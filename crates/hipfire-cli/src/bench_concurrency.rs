// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Concurrency sweep for `hipfire bench --concurrency`: a vLLM-style
//! closed-loop serving benchmark over the daemon wire protocol.
//!
//! Method (mirrors `vllm bench serve --max-concurrency K --request-rate inf
//! --num-prompts N`, and `sweep/conc_steady.py` of the 0.4.2 concurrency
//! audit):
//!
//! - **Closed loop at K.** K client lanes drain one FIFO of N ≫ K requests;
//!   a lane sends its next request the moment its previous one ends, so there
//!   is no burst barrier and no idle lane until the queue empties.
//! - **One prompt multiset for every K.** Request i is item `i % len` of the
//!   shape's prompt list, so every K does the same work.
//! - **Fixed output length.** The daemon has no `ignore_eos`; the `steady`
//!   shape's prompts all ask for long answers so greedy decoding runs to
//!   `max_tokens`. Every finish reason is reported.
//! - **Warmup discarded.** Before every sample, the 8 separate warmup prompts
//!   run closed-loop at the same K with 64-token outputs, so first use of a K
//!   shape never lands in a sample and no measured prompt is warm.
//! - **Per-run nonces.** Every request's first message starts with a
//!   `Q<run>-<i>.` tag unique to the (`--runs` round, request): every K of a
//!   round sends the same tokens, and a later round's repeat never hits the
//!   daemon's longest-common-prefix cache.
//! - **Chat turns, neutral sampling.** Requests carry `messages`, greedy
//!   sampling and neutral penalties (`repeat_penalty` 1.0): a non-neutral
//!   penalty takes the speculative decoders off their greedy routes.
//!
//! Metrics are vLLM's: output throughput = Σ output tokens / (last end −
//! first send), TTFT, TPOT = (end − first token) / (tokens − 1), plus the
//! steady window (every lane busy), SGLang's achieved concurrency
//! (Σ end-to-end / duration) and token-weighted τ.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::Path;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackendSel {
    /// The multi-slot engine: k streams advance together.
    Slots,
    /// No batching: the daemon serves one request at a time and the other
    /// lanes queue, as `hipfire serve` does with batching off.
    Sequential,
    /// The continuous-batching route `hipfire serve` runs with
    /// `serve.vmm_batch` on: `vmm` (Qwen3.5) or `fn-lanes` (Qwen4, single
    /// card or under `topology.pp` / `topology.ep`).
    Batch,
    Both,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WorkloadSel {
    Stateless,
    Multiturn,
    Both,
}

impl WorkloadSel {
    fn label(self) -> &'static str {
        match self {
            WorkloadSel::Stateless => "stateless",
            WorkloadSel::Multiturn => "multiturn",
            WorkloadSel::Both => "both",
        }
    }
    /// Turns per conversation: a multi-turn conversation sends a follow-up
    /// on the same lane as soon as its first answer is complete.
    fn turns(self) -> usize {
        if matches!(self, WorkloadSel::Multiturn) {
            2
        } else {
            1
        }
    }
}

/// Which request population the sweep sends.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BenchShape {
    /// The audit's 64 long-answer code prompts, fixed `max_tokens`, greedy.
    Steady,
    /// BetterBench 0.6.0's concurrency workload over its corpus directory.
    BetterBench,
}

impl BenchShape {
    pub fn label(self) -> &'static str {
        match self {
            BenchShape::Steady => "steady",
            BenchShape::BetterBench => "betterbench",
        }
    }
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
            .map_err(|_| anyhow!("--concurrency: not a number: {part}"))?;
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

/// Linear-interpolated percentile of sorted samples (numpy's default, and
/// `conc_steady.py`'s `pctl`). `None` when empty.
pub fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let k = (sorted.len() - 1) as f64 * p;
    let lo = k.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    Some(sorted[lo] + (sorted[hi] - sorted[lo]) * (k - lo as f64))
}

// ── Workloads ───────────────────────────────────────────────────────────────

/// The `steady` prompt set: 64 measured + 8 warmup code prompts, each asking
/// for a long answer. Byte-identical to the audit's `sweep/conc_prompts.json`
/// (md5 `cddf1f2a1a3b52da664352b2915dc36e`).
const STEADY_PROMPTS_JSON: &str = include_str!("bench_concurrency_prompts.json");

/// `steady` output length when `--max-tokens` is not given.
pub const STEADY_DEFAULT_MAX_TOKENS: u64 = 256;
/// Output length of the discarded warmup requests.
pub const WARMUP_TOKENS: u64 = 64;
/// BetterBench 0.6.0 sends 48 requests per concurrency level.
pub const BETTERBENCH_DEFAULT_REQUESTS: usize = 48;
/// Second user turn of the multi-turn arm.
pub const FOLLOWUP_PROMPT: &str = "Explain that again in one sentence.";

/// One request of a workload: its chat turn and output budget.
#[derive(Clone, Debug)]
pub struct Item {
    /// JSON array of `{role, content}` messages.
    pub messages: Value,
    pub max_tokens: u64,
}

/// Sampling sent on every request. Penalties are always neutral.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<u64>,
    pub seed: Option<u64>,
}

impl Sampling {
    pub const GREEDY: Sampling = Sampling {
        temperature: 0.0,
        top_p: 1.0,
        top_k: None,
        seed: None,
    };
    /// BetterBench's request sampling (temperature 0.7, top-p 0.95, top-k
    /// 20, seed 42 on every request).
    pub const BETTERBENCH: Sampling = Sampling {
        temperature: 0.7,
        top_p: 0.95,
        top_k: Some(20),
        seed: Some(42),
    };
}

pub struct Workload {
    pub measure: Vec<Item>,
    pub warmup: Vec<Item>,
    pub sampling: Sampling,
    /// Identity of the measured prompt source: md5 of the measured texts
    /// joined by `\n` (`steady`) or of the corpus files' bytes in sorted
    /// order (`betterbench`) — the same digests `conc_steady.py` records.
    pub source_md5: String,
}

impl Workload {
    /// Prepend `system` to every chat that has no system message, as
    /// `hipfire serve` does with `prompt.system`.
    pub fn inject_system(&mut self, system: &str) {
        for item in self.measure.iter_mut().chain(self.warmup.iter_mut()) {
            crate::serve::complete::inject_default_system_message(&mut item.messages, Some(system));
        }
    }
}

fn user_turn(text: &str) -> Value {
    serde_json::json!([{ "role": "user", "content": text }])
}

fn steady_prompt_lists() -> Result<(Vec<String>, Vec<String>)> {
    let v: Value = serde_json::from_str(STEADY_PROMPTS_JSON).context("embedded prompt set")?;
    let list = |key: &str| -> Result<Vec<String>> {
        v.get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("embedded prompt set has no {key} list"))?
            .iter()
            .map(|t| {
                t.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| anyhow!("embedded prompt set: non-string {key} entry"))
            })
            .collect()
    };
    Ok((list("measure")?, list("warmup")?))
}

fn steady_warmup(warm: &[String]) -> Vec<Item> {
    warm.iter()
        .map(|t| Item {
            messages: user_turn(t),
            max_tokens: WARMUP_TOKENS,
        })
        .collect()
}

/// The `steady` shape: request i is measured prompt `i % 64`, one user turn,
/// `max_tokens` each, greedy. `n` defaults to the 64 prompts.
pub fn steady_workload(n: Option<usize>, max_tokens: u64) -> Result<Workload> {
    let (measure, warm) = steady_prompt_lists()?;
    let n = n.unwrap_or(measure.len());
    if n == 0 {
        bail!("--num-prompts must be positive");
    }
    let texts: Vec<&str> = (0..n)
        .map(|i| measure[i % measure.len()].as_str())
        .collect();
    Ok(Workload {
        measure: texts
            .iter()
            .map(|t| Item {
                messages: user_turn(t),
                max_tokens,
            })
            .collect(),
        warmup: steady_warmup(&warm),
        sampling: Sampling::GREEDY,
        source_md5: format!("{:x}", md5::compute(texts.join("\n").as_bytes())),
    })
}

/// BetterBench 0.6.0's concurrency workload (`runner.py`
/// `concurrency_sweep`, `corpus.py` `load_corpus`): `flat` is every prompt of
/// the corpus's `*.jsonl` files in sorted file order, request i is
/// `flat[i % len(flat)]` with that prompt's own messages and `max_tokens`
/// (default 512), 48 requests per level, BetterBench's sampling. Warmup uses
/// the `steady` warmup prompts.
pub fn betterbench_workload(dir: &Path, n: Option<usize>) -> Result<Workload> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("--bb-corpus {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("jsonl"))
        .collect();
    files.sort();
    let mut raw = Vec::new();
    let mut flat = Vec::new();
    for path in &files {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let text = std::str::from_utf8(&bytes)
            .with_context(|| format!("{} is not UTF-8", path.display()))?;
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let o: Value = serde_json::from_str(line)
                .with_context(|| format!("{}:{}", path.display(), lineno + 1))?;
            let messages = o
                .get("messages")
                .filter(|m| m.as_array().is_some_and(|a| !a.is_empty()))
                .cloned()
                .ok_or_else(|| anyhow!("{}:{}: no messages", path.display(), lineno + 1))?;
            let max_tokens = match o.get("max_tokens") {
                None | Some(Value::Null) => 512,
                Some(v) => v.as_u64().ok_or_else(|| {
                    anyhow!(
                        "{}:{}: max_tokens is not a count",
                        path.display(),
                        lineno + 1
                    )
                })?,
            };
            flat.push(Item {
                messages,
                max_tokens,
            });
        }
        raw.extend_from_slice(&bytes);
    }
    if flat.is_empty() {
        bail!("--bb-corpus {}: no prompts in *.jsonl files", dir.display());
    }
    let n = n.unwrap_or(BETTERBENCH_DEFAULT_REQUESTS);
    if n == 0 {
        bail!("--num-prompts must be positive");
    }
    let (_, warm) = steady_prompt_lists()?;
    Ok(Workload {
        measure: (0..n).map(|i| flat[i % flat.len()].clone()).collect(),
        warmup: steady_warmup(&warm),
        sampling: Sampling::BETTERBENCH,
        source_md5: format!("{:x}", md5::compute(&raw)),
    })
}

/// `messages` with `tag` prepended to the first message's text, so the
/// prompt differs from every other request from its first content token and
/// the daemon's longest-common-prefix cache cannot serve it.
pub fn with_nonce(messages: &Value, tag: &str) -> Result<Value> {
    let mut out = messages.clone();
    let first = out
        .as_array_mut()
        .and_then(|a| a.first_mut())
        .ok_or_else(|| anyhow!("request has no messages"))?;
    let text = first
        .get("content")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("first message content must be a string"))?;
    first["content"] = Value::String(format!("{tag} {text}"));
    Ok(out)
}

fn last_user_text(messages: &Value) -> Option<&str> {
    messages
        .as_array()?
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))?
        .get("content")?
        .as_str()
}

/// The continuous-batching routes the batch sweep may measure.
pub fn is_batch_route(route: Option<&str>) -> bool {
    matches!(route, Some("vmm" | "fn-lanes"))
}

// ── Drivers ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DriverKind {
    Slots,
    Sequential,
    Batch,
}

/// One daemon, loaded once at max concurrency, driven over the wire.
pub struct Driver {
    kind: DriverKind,
    engine: hipfire_client::Engine,
    max_concurrency: usize,
    reasoning_on: bool,
    /// Monotonic sample counter, part of every request id so attempt ids
    /// never repeat.
    sample_seq: usize,
}

impl Driver {
    /// The multi-slot backend. `slot_capable` is the load reply's
    /// `experimental_multi_slot`: refusing here turns "this daemon was not
    /// loaded with the slot backend" into a reported result rather than a run
    /// that silently measures the ordinary sequential path.
    pub fn slots(
        engine: hipfire_client::Engine,
        max_concurrency: usize,
        slot_capable: bool,
        reasoning_on: bool,
    ) -> Result<Self> {
        if !slot_capable {
            bail!("daemon does not advertise experimental_multi_slot capability");
        }
        Ok(Self::new(
            DriverKind::Slots,
            engine,
            max_concurrency,
            reasoning_on,
        ))
    }

    /// The no-batching baseline: the daemon serves one request at a time.
    pub fn sequential(
        engine: hipfire_client::Engine,
        max_concurrency: usize,
        reasoning_on: bool,
    ) -> Self {
        Self::new(
            DriverKind::Sequential,
            engine,
            max_concurrency,
            reasoning_on,
        )
    }

    /// The continuous-batching route `hipfire serve` runs with
    /// `serve.vmm_batch` on (`vmm` or `fn-lanes`). `loaded` is the load
    /// reply; the driver refuses unless the daemon staged one of them, so the
    /// numbers can never describe another route.
    pub fn batch(
        engine: hipfire_client::Engine,
        max_concurrency: usize,
        loaded: &Value,
        reasoning_on: bool,
    ) -> Result<Self> {
        if loaded
            .get("continuous_batch_capable")
            .and_then(Value::as_bool)
            != Some(true)
        {
            bail!("daemon does not advertise continuous_batch_capable");
        }
        let route = loaded.get("continuous_batch_route").and_then(Value::as_str);
        if !is_batch_route(route) {
            bail!(
                "daemon staged continuous-batch route {route:?}, not the `vmm` or `fn-lanes` \
                 route `hipfire serve` runs with serve.vmm_batch"
            );
        }
        Ok(Self::new(
            DriverKind::Batch,
            engine,
            max_concurrency,
            reasoning_on,
        ))
    }

    fn new(
        kind: DriverKind,
        engine: hipfire_client::Engine,
        max_concurrency: usize,
        reasoning_on: bool,
    ) -> Self {
        Self {
            kind,
            engine,
            max_concurrency,
            reasoning_on,
            sample_seq: 0,
        }
    }

    pub fn label(&self) -> &'static str {
        match self.kind {
            DriverKind::Slots => "slots",
            DriverKind::Sequential => "noslots",
            DriverKind::Batch => "batch",
        }
    }

    /// Requests the daemon is given at once. The sequential baseline holds
    /// the other lanes' requests client-side, the way `hipfire serve` queues
    /// them with batching off; their queue time stays in TTFT.
    fn max_inflight(&self, k: usize) -> usize {
        match self.kind {
            DriverKind::Sequential => 1,
            DriverKind::Slots | DriverKind::Batch => k,
        }
    }

    /// Reject `k` above the started maximum instead of clamping. A clamped
    /// run would report a k=4 row that was really measured at k=2.
    pub fn check_k(&self, k: usize) -> Result<()> {
        if k > self.max_concurrency {
            bail!(
                "concurrency {k} exceeds {} backend maximum {}",
                self.label(),
                self.max_concurrency
            );
        }
        Ok(())
    }

    fn request(
        &self,
        messages: &Value,
        max_tokens: u64,
        s: &Sampling,
        id: &str,
        attempt: u64,
    ) -> Value {
        build_request(
            self.kind,
            self.reasoning_on,
            messages,
            max_tokens,
            s,
            id,
            attempt,
        )
    }
}

/// One generate request for `kind`.
///
/// `Batch` and `Sequential` send the wire request `hipfire serve` sends for a
/// chat completion (`messages` plus the last user turn as `prompt`, the
/// projected thinking switch), with `serve_continuous_batch` on the batch
/// route only. `Slots` keeps the slot backend's own answer-mode fields.
/// Every kind sends neutral penalties.
pub fn build_request(
    kind: DriverKind,
    reasoning_on: bool,
    messages: &Value,
    max_tokens: u64,
    s: &Sampling,
    id: &str,
    attempt_id: u64,
) -> Value {
    let mut r = serde_json::json!({
        "type": "generate",
        "id": id,
        "messages": messages,
        "max_tokens": max_tokens,
        "attempt_id": attempt_id,
        "temperature": s.temperature,
        "top_p": s.top_p,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "min_p": 0.0,
    });
    if let Some(k) = s.top_k {
        r["top_k"] = serde_json::json!(k);
    }
    if let Some(seed) = s.seed {
        r["seed"] = serde_json::json!(seed);
    }
    match kind {
        DriverKind::Slots => {
            r["experimental_multi_slot"] = Value::Bool(true);
            if !reasoning_on {
                r["max_think_tokens"] = serde_json::json!(1);
                r["assistant_prefix"] = serde_json::json!("closed_think");
                r["reasoning_effort"] = serde_json::json!("none");
            }
        }
        DriverKind::Sequential | DriverKind::Batch => {
            r["prompt"] = serde_json::json!(last_user_text(messages).unwrap_or("Hello"));
            r["thinking_enabled"] = Value::Bool(reasoning_on);
            r["assistant_prefix"] = serde_json::json!(if reasoning_on {
                "open_think"
            } else {
                "closed_think"
            });
            if kind == DriverKind::Batch {
                r["serve_continuous_batch"] = Value::Bool(true);
            }
        }
    }
    r
}

// ── Closed loop ─────────────────────────────────────────────────────────────

/// A daemon that sends nothing for this long fails the sample instead of
/// hanging the bench (finite resources).
const STALL_DEADLINE: Duration = Duration::from_secs(300);

/// One request (one turn) of a sample. Times are seconds from the start of
/// the closed loop.
#[derive(Clone, Debug, Default)]
pub struct RequestRecord {
    pub conv: usize,
    pub turn: usize,
    /// When its lane sent it (for the sequential baseline: when the lane
    /// would have sent it; it may wait client-side for the daemon).
    pub t_send: f64,
    pub t_first: Option<f64>,
    pub t_end: f64,
    /// Arrival time of every `token` frame.
    pub token_times: Vec<f64>,
    /// Committed output tokens: the daemon's `done.tokens`, or the token
    /// frame count when `done` has none.
    pub tokens: u64,
    pub finish: Option<String>,
    pub tau: Option<f64>,
    pub execution_mode: Option<String>,
    pub batched: Option<bool>,
    pub cached_tokens: u64,
    pub prompt_tokens: Option<u64>,
    pub err: Option<String>,
    pub text: String,
}

/// Every request of one closed-loop run, in completion order.
pub struct Sample {
    pub k: usize,
    pub records: Vec<RequestRecord>,
    /// Loop start to last completion: BetterBench's `aggregate_tps` wall.
    pub wall_s: f64,
}

enum Ev {
    Frame(Value),
    Closed,
}

struct Inflight {
    id: String,
    attempt: u64,
}

/// Run `items` through `k` closed-loop lanes and collect every request.
///
/// Each item is one conversation of `turns` turns; turn 2 is sent on the
/// same lane the moment turn 1 completes. Every first message carries
/// `<nonce>-<i>.`; request ids carry the unique `sample` number.
pub fn closed_loop(
    d: &Driver,
    items: &[Item],
    k: usize,
    sampling: &Sampling,
    turns: usize,
    nonce: &str,
    sample: usize,
) -> Result<Sample> {
    let label = d.label();
    let cap = d.max_inflight(k).max(1);
    let (tx, rx) = mpsc::channel::<(usize, Instant, Ev)>();
    let t0 = Instant::now();
    let secs = |at: Instant| at.saturating_duration_since(t0).as_secs_f64();

    let mut records: Vec<RequestRecord> = Vec::with_capacity(items.len() * turns);
    // Wire request and conversation messages per record, until dispatched.
    let mut wire: Vec<Option<Value>> = Vec::with_capacity(items.len() * turns);
    let mut convs: Vec<Value> = Vec::with_capacity(items.len());
    let mut ready: VecDeque<usize> = VecDeque::new();
    let mut inflight: HashMap<usize, Inflight> = HashMap::new();
    let mut next_item = 0usize;

    let id_of = |conv: usize| format!("bench-{label}-{sample}-{nonce}-{conv}");

    // Queue turn `turn` of conversation `conv` with `messages`, sent at `t`.
    let push = |records: &mut Vec<RequestRecord>,
                wire: &mut Vec<Option<Value>>,
                ready: &mut VecDeque<usize>,
                conv: usize,
                turn: usize,
                messages: &Value,
                t: f64| {
        let req = d.request(
            messages,
            items[conv].max_tokens,
            sampling,
            &id_of(conv),
            turn as u64 + 1,
        );
        records.push(RequestRecord {
            conv,
            turn,
            t_send: t,
            ..Default::default()
        });
        wire.push(Some(req));
        ready.push_back(records.len() - 1);
    };

    let start_conv = |convs: &mut Vec<Value>, next_item: &mut usize| -> Result<Option<usize>> {
        if *next_item >= items.len() {
            return Ok(None);
        }
        let conv = *next_item;
        *next_item += 1;
        convs.push(with_nonce(
            &items[conv].messages,
            &format!("{nonce}-{conv}."),
        )?);
        Ok(Some(conv))
    };

    for _ in 0..k.min(items.len()) {
        if let Some(conv) = start_conv(&mut convs, &mut next_item)? {
            let msgs = convs[conv].clone();
            push(&mut records, &mut wire, &mut ready, conv, 0, &msgs, 0.0);
        }
    }

    let dispatch = |records: &mut Vec<RequestRecord>,
                    wire: &mut Vec<Option<Value>>,
                    ready: &mut VecDeque<usize>,
                    inflight: &mut HashMap<usize, Inflight>|
     -> Result<()> {
        while inflight.len() < cap {
            let Some(r) = ready.pop_front() else {
                break;
            };
            let req = wire[r].take().expect("each request dispatches once");
            let (id, attempt, stream) = d
                .engine
                .submit_streaming(&req)
                .map_err(|e| anyhow!("{label} submit (conversation {}): {e}", records[r].conv))?;
            inflight.insert(r, Inflight { id, attempt });
            let tx = tx.clone();
            std::thread::spawn(move || {
                for frame in stream.iter() {
                    let terminal = matches!(
                        frame.get("type").and_then(Value::as_str),
                        Some("done") | Some("error")
                    );
                    if tx.send((r, Instant::now(), Ev::Frame(frame))).is_err() || terminal {
                        return;
                    }
                }
                let _ = tx.send((r, Instant::now(), Ev::Closed));
            });
        }
        Ok(())
    };

    let cancel_all = |inflight: &HashMap<usize, Inflight>| {
        for f in inflight.values() {
            // Cancel daemon-side, not just client-side: a daemon still
            // generating into a dropped channel pollutes the next sample.
            let _ = d.engine.cancel_attempt(&f.id, f.attempt);
            d.engine.release_attempt(&f.id, f.attempt);
        }
    };

    if let Err(e) = dispatch(&mut records, &mut wire, &mut ready, &mut inflight) {
        cancel_all(&inflight);
        return Err(e);
    }

    while !inflight.is_empty() || !ready.is_empty() {
        let (r, at, ev) = match rx.recv_timeout(STALL_DEADLINE) {
            Ok(x) => x,
            Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => {
                cancel_all(&inflight);
                bail!(
                    "{label} k={k}: no daemon frame for {} s with {} request(s) in flight",
                    STALL_DEADLINE.as_secs(),
                    inflight.len()
                );
            }
        };
        let Some(f) = inflight.get(&r) else {
            continue;
        };
        let t = secs(at);
        let rec = &mut records[r];
        let terminal = match ev {
            Ev::Frame(frame) => match frame.get("type").and_then(Value::as_str) {
                Some("token") => {
                    rec.t_first.get_or_insert(t);
                    rec.token_times.push(t);
                    if let Some(text) = frame.get("text").and_then(Value::as_str) {
                        rec.text.push_str(text);
                    }
                    false
                }
                Some("commit_ready") => {
                    if let Some(c) = frame.get("cached_tokens").and_then(Value::as_u64) {
                        rec.cached_tokens = rec.cached_tokens.max(c);
                    }
                    match d.engine.commit_attempt(&f.id, f.attempt) {
                        Ok(()) => false,
                        Err(e) => {
                            rec.err = Some(format!("commit: {e}"));
                            true
                        }
                    }
                }
                Some("done") => {
                    record_done(rec, &frame);
                    true
                }
                Some("error") => {
                    rec.err = Some(
                        frame
                            .get("message")
                            .and_then(Value::as_str)
                            .unwrap_or("daemon error")
                            .to_owned(),
                    );
                    true
                }
                // gen_start, reasoning, aborted, tool_calls, ...
                _ => false,
            },
            Ev::Closed => {
                rec.err = Some("stream closed without done".to_owned());
                true
            }
        };
        if !terminal {
            continue;
        }
        rec.t_end = t;
        if rec.tokens == 0 {
            rec.tokens = rec.token_times.len() as u64;
        }
        let f = inflight.remove(&r).expect("checked above");
        d.engine.release_attempt(&f.id, f.attempt);

        // The lane is free: its conversation's next turn, else the next item.
        let (conv, turn, ok) = (rec.conv, rec.turn, rec.err.is_none());
        if ok && turn + 1 < turns {
            let mut msgs = convs[conv].clone();
            if let Some(a) = msgs.as_array_mut() {
                a.push(serde_json::json!({ "role": "assistant", "content": records[r].text }));
                a.push(serde_json::json!({ "role": "user", "content": FOLLOWUP_PROMPT }));
            }
            convs[conv] = msgs.clone();
            push(
                &mut records,
                &mut wire,
                &mut ready,
                conv,
                turn + 1,
                &msgs,
                t,
            );
        } else if let Some(next) = start_conv(&mut convs, &mut next_item)? {
            let msgs = convs[next].clone();
            push(&mut records, &mut wire, &mut ready, next, 0, &msgs, t);
        }
        if let Err(e) = dispatch(&mut records, &mut wire, &mut ready, &mut inflight) {
            cancel_all(&inflight);
            return Err(e);
        }
    }

    Ok(Sample {
        k,
        records,
        wall_s: t0.elapsed().as_secs_f64(),
    })
}

/// Copy the `done` frame's accounting into the record (the fields
/// `hipfire serve` reports as `usage`, `timings` and `hipfire`).
fn record_done(rec: &mut RequestRecord, done: &Value) {
    let u = |key: &str| done.get(key).and_then(Value::as_u64);
    rec.tokens = u("tokens").or_else(|| u("completion_tokens")).unwrap_or(0);
    rec.finish = done
        .get("finish_reason")
        .and_then(Value::as_str)
        .map(str::to_owned);
    rec.tau = done.get("tau").and_then(Value::as_f64);
    rec.execution_mode = done
        .get("execution_mode")
        .and_then(Value::as_str)
        .map(str::to_owned);
    rec.batched = done
        .get("continuous_batch")
        .and_then(|c| c.get("executed"))
        .and_then(Value::as_bool);
    if let Some(c) = u("cached_tokens") {
        rec.cached_tokens = rec.cached_tokens.max(c);
    }
    rec.prompt_tokens = u("prompt_tokens")
        .or_else(|| u("prefill_tokens").map(|p| p + u("cached_tokens").unwrap_or(0)));
}

// ── Metrics ─────────────────────────────────────────────────────────────────

/// p50/p99 of a latency distribution, milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pctl {
    pub p50: f64,
    pub p99: f64,
}

fn pctl_ms(mut xs: Vec<f64>) -> Option<Pctl> {
    xs.sort_by(f64::total_cmp);
    Some(Pctl {
        p50: percentile(&xs, 0.5)? * 1000.0,
        p99: percentile(&xs, 0.99)? * 1000.0,
    })
}

/// The metrics of one sample (`conc_steady.py` `metrics`).
#[derive(Clone, Debug, Default)]
pub struct SampleMetrics {
    pub n: usize,
    pub n_ok: usize,
    pub output_tokens: u64,
    /// Last end − first send.
    pub duration_s: f64,
    /// vLLM output throughput: output tokens / duration.
    pub output_tok_s: f64,
    /// BetterBench `aggregate_tps`: output tokens / loop wall.
    pub bb_tok_s: f64,
    /// Throughput while every lane is busy (ramp and drain excluded).
    pub steady_tok_s: Option<f64>,
    pub ttft: Option<Pctl>,
    pub tpot: Option<Pctl>,
    /// SGLang achieved concurrency: Σ end-to-end / duration.
    pub concurrency: f64,
    /// Output tokens per verify window over the whole sample.
    pub tau: Option<f64>,
    pub finish: BTreeMap<String, usize>,
    pub routes: BTreeMap<String, usize>,
    pub batched: usize,
    pub cached_tokens: u64,
    /// Requests after the first turn of a conversation whose prompt the
    /// daemon served partly from cache.
    pub followup_prefix_hits: usize,
    pub prompt_tokens_mean: Option<f64>,
    pub errors: Vec<String>,
    pub text_md5: String,
}

impl SampleMetrics {
    pub fn from_sample(s: &Sample) -> Self {
        let recs = &s.records;
        let ok: Vec<&RequestRecord> = recs
            .iter()
            .filter(|r| r.err.is_none() && r.tokens > 0)
            .collect();
        let t0 = recs.iter().map(|r| r.t_send).fold(f64::INFINITY, f64::min);
        let t1 = recs.iter().map(|r| r.t_end).fold(0.0, f64::max);
        let duration_s = if recs.is_empty() { 0.0 } else { t1 - t0 };
        let output_tokens: u64 = ok.iter().map(|r| r.tokens).sum();
        let rate = |toks: u64, secs: f64| if secs > 0.0 { toks as f64 / secs } else { 0.0 };

        let ttft = pctl_ms(
            ok.iter()
                .filter_map(|r| Some(r.t_first? - r.t_send))
                .collect(),
        );
        let tpot = pctl_ms(
            ok.iter()
                .filter(|r| r.tokens > 1)
                .filter_map(|r| Some((r.t_end - r.t_first?) / (r.tokens - 1) as f64))
                .collect(),
        );
        let e2e: f64 = ok.iter().map(|r| r.t_end - r.t_send).sum();

        // Steady window: from the moment every one of the first K requests
        // has its first token to the first completion at/after the last send
        // (in-flight drops below K for good).
        let steady_tok_s = (|| {
            let mut by_send: Vec<&RequestRecord> = recs.iter().collect();
            by_send.sort_by(|a, b| a.t_send.total_cmp(&b.t_send));
            let w0 = by_send
                .iter()
                .take(s.k)
                .map(|r| r.t_first.unwrap_or(r.t_end))
                .fold(f64::NEG_INFINITY, f64::max);
            let last_send = by_send.last()?.t_send;
            let w1 = recs
                .iter()
                .map(|r| r.t_end)
                .filter(|&t| t >= last_send)
                .fold(f64::INFINITY, f64::min);
            if !w1.is_finite() || w1 <= w0 {
                return None;
            }
            let toks: f64 = ok
                .iter()
                .filter(|r| !r.token_times.is_empty())
                .map(|r| {
                    let per = r.tokens as f64 / r.token_times.len() as f64;
                    per * r
                        .token_times
                        .iter()
                        .filter(|&&t| w0 <= t && t <= w1)
                        .count() as f64
                })
                .sum();
            Some(toks / (w1 - w0))
        })();

        let tw: Vec<(f64, f64)> = ok
            .iter()
            .filter_map(|r| Some((r.tokens as f64, r.tau.filter(|&t| t > 0.0)?)))
            .collect();
        let tau = (!tw.is_empty()).then(|| {
            tw.iter().map(|(t, _)| t).sum::<f64>() / tw.iter().map(|(t, x)| t / x).sum::<f64>()
        });

        let mut finish = BTreeMap::new();
        let mut routes = BTreeMap::new();
        for r in recs {
            let key = match (&r.err, &r.finish) {
                (Some(_), _) => "error".to_owned(),
                (None, Some(f)) => f.clone(),
                (None, None) => "none".to_owned(),
            };
            *finish.entry(key).or_insert(0) += 1;
            let route = format!(
                "{}/batched={}",
                r.execution_mode.as_deref().unwrap_or("null"),
                r.batched.map_or("null".to_owned(), |b| b.to_string())
            );
            *routes.entry(route).or_insert(0) += 1;
        }
        let prompts: Vec<u64> = recs.iter().filter_map(|r| r.prompt_tokens).collect();
        let mut by_order: Vec<&RequestRecord> = recs.iter().collect();
        by_order.sort_by_key(|r| (r.conv, r.turn));
        let joined: Vec<&str> = by_order.iter().map(|r| r.text.as_str()).collect();

        SampleMetrics {
            n: recs.len(),
            n_ok: ok.len(),
            output_tokens,
            duration_s,
            output_tok_s: rate(output_tokens, duration_s),
            bb_tok_s: rate(output_tokens, s.wall_s),
            steady_tok_s,
            ttft,
            tpot,
            concurrency: if duration_s > 0.0 {
                e2e / duration_s
            } else {
                0.0
            },
            tau,
            finish,
            routes,
            batched: recs.iter().filter(|r| r.batched == Some(true)).count(),
            cached_tokens: recs.iter().map(|r| r.cached_tokens).sum(),
            followup_prefix_hits: recs
                .iter()
                .filter(|r| r.turn > 0 && r.cached_tokens > 0)
                .count(),
            prompt_tokens_mean: (!prompts.is_empty())
                .then(|| prompts.iter().sum::<u64>() as f64 / prompts.len() as f64),
            errors: recs.iter().filter_map(|r| r.err.clone()).collect(),
            text_md5: format!("{:x}", md5::compute(joined.join("\u{0}").as_bytes())),
        }
    }

    fn summary(map: &BTreeMap<String, usize>) -> String {
        map.iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// One progress line per sample.
    pub fn line(&self) -> String {
        let p =
            |x: Option<Pctl>| x.map_or("-".to_owned(), |x| format!("{:.0}/{:.0}", x.p50, x.p99));
        let q =
            |x: Option<Pctl>| x.map_or("-".to_owned(), |x| format!("{:.2}/{:.2}", x.p50, x.p99));
        format!(
            "out {:.2} tok/s  bb {:.2}  steady {}  ttft {} ms  tpot {} ms  in-flight {:.2}  tau {}  \
             {}/{} ok  {} tok in {:.1} s  finish {}  routes {}  cached {}  prompt {}  text_md5 {}",
            self.output_tok_s,
            self.bb_tok_s,
            self.steady_tok_s.map_or("-".to_owned(), |x| format!("{x:.2}")),
            p(self.ttft),
            q(self.tpot),
            self.concurrency,
            self.tau.map_or("-".to_owned(), |x| format!("{x:.3}")),
            self.n_ok,
            self.n,
            self.output_tokens,
            self.duration_s,
            Self::summary(&self.finish),
            Self::summary(&self.routes),
            self.cached_tokens,
            self.prompt_tokens_mean.map_or("-".to_owned(), |x| format!("{x:.1}")),
            self.text_md5,
        )
    }
}

/// All samples for one (backend, workload, k).
pub struct Point {
    pub backend: &'static str,
    pub workload: &'static str,
    pub k: usize,
    pub samples: Vec<SampleMetrics>,
}

fn med(samples: &[SampleMetrics], f: impl Fn(&SampleMetrics) -> Option<f64>) -> Option<f64> {
    let mut xs: Vec<f64> = samples.iter().filter_map(f).collect();
    median(&mut xs)
}

/// Render the comparison table: per point, the median over samples of each
/// metric, and the min–max of output throughput.
pub fn render_table(points: &[Point], shape: BenchShape) -> String {
    let bb = shape == BenchShape::BetterBench;
    let mut out = String::new();
    fn row(cells: [&str; 3], out_tok_s: &str, bb_col: Option<&str>, rest: [&str; 8]) -> String {
        let bb_col = bb_col.map_or(String::new(), |c| format!("{c:>12}  "));
        format!(
            "  {:<8} {:<10} {:>2}  {:<26}  {}{:>12}  {:>15}  {:>15}  {:>9}  {:>5}  {:>6}  {:<20}  {}\n",
            cells[0],
            cells[1],
            cells[2],
            out_tok_s,
            bb_col,
            rest[0],
            rest[1],
            rest[2],
            rest[3],
            rest[4],
            rest[5],
            rest[6],
            rest[7]
        )
    }
    out.push('\n');
    out.push_str(&row(
        ["backend", "workload", "k"],
        "out tok/s (min-max)",
        bb.then_some("bb agg tok/s"),
        [
            "steady tok/s",
            "TTFT p50/p99 ms",
            "TPOT p50/p99 ms",
            "in-flight",
            "tau",
            "tokens",
            "finish",
            "batched  cached  errors",
        ],
    ));
    let f2 = |x: Option<f64>| x.map_or("-".to_owned(), |x| format!("{x:.2}"));
    for p in points {
        let s = &p.samples;
        let Some(m) = med(s, |x| Some(x.output_tok_s)) else {
            continue;
        };
        let lo = s
            .iter()
            .map(|x| x.output_tok_s)
            .fold(f64::INFINITY, f64::min);
        let hi = s.iter().map(|x| x.output_tok_s).fold(0.0, f64::max);
        let ttft = format!(
            "{}/{}",
            med(s, |x| x.ttft.map(|t| t.p50)).map_or("-".to_owned(), |x| format!("{x:.0}")),
            med(s, |x| x.ttft.map(|t| t.p99)).map_or("-".to_owned(), |x| format!("{x:.0}")),
        );
        let tpot = format!(
            "{}/{}",
            f2(med(s, |x| x.tpot.map(|t| t.p50))),
            f2(med(s, |x| x.tpot.map(|t| t.p99))),
        );
        // Finish/batched/cached/errors are summed over samples: one
        // `stop` or one unbatched request in any sample must be visible.
        let mut finish: BTreeMap<String, usize> = BTreeMap::new();
        for x in s {
            for (k, v) in &x.finish {
                *finish.entry(k.clone()).or_insert(0) += v;
            }
        }
        let finish = SampleMetrics::summary(&finish);
        let n: usize = s.iter().map(|x| x.n).sum();
        let batched: usize = s.iter().map(|x| x.batched).sum();
        let cached: u64 = s.iter().map(|x| x.cached_tokens).sum();
        let errors: usize = s.iter().map(|x| x.errors.len()).sum();
        let bb_val = f2(med(s, |x| Some(x.bb_tok_s)));
        out.push_str(&row(
            [p.backend, p.workload, &p.k.to_string()],
            &format!("{m:.2} ({lo:.2}-{hi:.2})"),
            bb.then_some(bb_val.as_str()),
            [
                &f2(med(s, |x| x.steady_tok_s)),
                &ttft,
                &tpot,
                &f2(med(s, |x| Some(x.concurrency))),
                &med(s, |x| x.tau).map_or("-".to_owned(), |x| format!("{x:.3}")),
                &med(s, |x| Some(x.output_tokens as f64))
                    .map_or("-".to_owned(), |x| format!("{x:.0}")),
                &finish,
                &format!("{:>7}  {cached:>6}  {errors:>6}", format!("{batched}/{n}")),
            ],
        ));
    }
    out.push_str(
        "\n  out tok/s: vLLM output throughput, output tokens / (last end - first send), median\n\
         \x20 of the samples (min-max in brackets). steady: throughput while all k lanes were\n\
         \x20 busy. TTFT/TPOT: medians of per-sample percentiles. in-flight: achieved\n\
         \x20 concurrency, sum of request latencies / duration. tau: output tokens per verify\n\
         \x20 window. tokens: median output tokens per sample. finish/batched/cached/errors:\n\
         \x20 summed over samples (cached = prompt tokens served from the daemon's cache).\n",
    );
    if bb {
        out.push_str(
            "  bb agg: BetterBench aggregate_tps, output tokens / closed-loop wall clock.\n",
        );
    }
    out
}

/// Reject a multi-turn slot sample whose follow-up turns recorded no prefix
/// hits. The point of the slot multi-turn arm is that `SlotEngine` reuses the
/// conversation's KV; a faster turn 2 proves nothing on its own.
pub fn sample_is_valid(
    workload: WorkloadSel,
    backend: &str,
    m: &SampleMetrics,
) -> std::result::Result<(), String> {
    if matches!(workload, WorkloadSel::Multiturn)
        && backend == "slots"
        && m.followup_prefix_hits == 0
    {
        return Err(
            "multi-turn arm recorded no prefix hits — turn 2 did not reuse the \
             session KV, so this number does not measure what the arm claims"
                .to_string(),
        );
    }
    Ok(())
}

/// Run the interleaved sweep over one backend and collect points. Every
/// sample is preceded by a discarded warmup at the same k.
pub fn sweep_backend(
    d: &mut Driver,
    workload: &Workload,
    arms: &[WorkloadSel],
    points: &[usize],
    runs: usize,
    out: &mut Vec<Point>,
) -> Result<()> {
    let label = d.label();
    for &arm in arms {
        if arm == WorkloadSel::Both {
            continue;
        }
        let mut by_k: BTreeMap<usize, Vec<SampleMetrics>> = BTreeMap::new();
        for (i, k) in sweep_order(points, runs).into_iter().enumerate() {
            d.check_k(k)?;
            d.sample_seq += 1;
            let sample = d.sample_seq;
            // The nonce is per run (sweep round), not per sample: within a
            // round every k sends the identical token multiset, and the next
            // round's repeats differ from the first content token, so they
            // cannot hit the daemon's prompt cache.
            let round = i / points.len() + 1;
            let warm = closed_loop(
                d,
                &workload.warmup,
                k,
                &workload.sampling,
                1,
                &format!("W{round}"),
                sample,
            )?;
            if let Some(e) = warm.records.iter().find_map(|r| r.err.as_ref()) {
                bail!("{label}/{} k={k} warmup failed: {e}", arm.label());
            }
            let sample = closed_loop(
                d,
                &workload.measure,
                k,
                &workload.sampling,
                arm.turns(),
                &format!("Q{round}"),
                sample,
            )?;
            let m = SampleMetrics::from_sample(&sample);
            eprintln!(
                "  {label}/{} k={k} run {}: {}",
                arm.label(),
                round,
                m.line()
            );
            if let Some(e) = m.errors.first() {
                eprintln!(
                    "  [errors] {label}/{} k={k}: {} failed, first: {e}",
                    arm.label(),
                    m.errors.len()
                );
            }
            match sample_is_valid(arm, label, &m) {
                Ok(()) => by_k.entry(k).or_default().push(m),
                Err(why) => eprintln!("  [invalid] {label}/{} k={k}: {why}", arm.label()),
            }
        }
        for (k, samples) in by_k {
            out.push(Point {
                backend: label,
                workload: arm.label(),
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
    fn percentile_interpolates_like_numpy() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&xs, 0.5), Some(2.5));
        assert!((percentile(&xs, 0.99).unwrap() - 3.97).abs() < 1e-12);
        assert_eq!(percentile(&[], 0.5), None);
    }

    /// The embedded prompt set is the audit's: same measured multiset (the
    /// md5 `conc_steady.py` records as `measure_md5`), 8 separate warmups.
    #[test]
    fn steady_workload_is_the_audit_prompt_set() {
        let w = steady_workload(None, 256).unwrap();
        assert_eq!(w.measure.len(), 64);
        assert_eq!(w.warmup.len(), 8);
        assert_eq!(w.source_md5, "f0b077a766edc166a0c3f18cf81c169c");
        assert!(w.measure.iter().all(|i| i.max_tokens == 256));
        assert!(w.warmup.iter().all(|i| i.max_tokens == WARMUP_TOKENS));
        assert_eq!(w.sampling, Sampling::GREEDY);
        let (measure, warm) = steady_prompt_lists().unwrap();
        assert!(
            warm.iter().all(|t| !measure.contains(t)),
            "a warmup prompt is measured"
        );
        // N > 64 cycles the same multiset.
        let w = steady_workload(Some(66), 8).unwrap();
        assert_eq!(w.measure[64].messages, w.measure[0].messages);
    }

    #[test]
    fn betterbench_workload_cycles_sorted_files() {
        let dir = std::env::temp_dir().join(format!("hipfire-bb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("b.jsonl"),
            "{\"id\":\"b1\",\"messages\":[{\"role\":\"user\",\"content\":\"B1\"}],\"max_tokens\":7}\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("a.jsonl"),
            "# comment\n{\"id\":\"a1\",\"messages\":[{\"role\":\"user\",\"content\":\"A1\"}]}\n\n\
             {\"id\":\"a2\",\"messages\":[{\"role\":\"user\",\"content\":\"A2\"}],\"max_tokens\":9}\n",
        )
        .unwrap();
        std::fs::write(dir.join("ignored.txt"), "x").unwrap();
        let w = betterbench_workload(&dir, Some(5)).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        let firsts: Vec<&str> = w
            .measure
            .iter()
            .map(|i| i.messages[0]["content"].as_str().unwrap())
            .collect();
        assert_eq!(firsts, ["A1", "A2", "B1", "A1", "A2"]);
        let budgets: Vec<u64> = w.measure.iter().map(|i| i.max_tokens).collect();
        assert_eq!(budgets, [512, 9, 7, 512, 9]);
        assert_eq!(w.sampling, Sampling::BETTERBENCH);
        assert_eq!(w.warmup.len(), 8);
    }

    #[test]
    fn nonce_prefixes_the_first_message_only() {
        let m = serde_json::json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "again"}
        ]);
        let n = with_nonce(&m, "Q3-7.").unwrap();
        assert_eq!(n[0]["content"], "Q3-7. hi");
        assert_eq!(n[2]["content"], "again");
        assert!(with_nonce(&serde_json::json!([]), "Q").is_err());
    }

    /// The batch request is the chat request `hipfire serve` projects: chat
    /// messages, the batch opt-in, thinking off, neutral penalties.
    #[test]
    fn batch_request_is_a_serve_chat_turn_with_neutral_sampling() {
        let msgs = user_turn("hello");
        let r = build_request(
            DriverKind::Batch,
            false,
            &msgs,
            64,
            &Sampling::GREEDY,
            "c-1",
            2,
        );
        assert_eq!(r["serve_continuous_batch"], true);
        assert_eq!(r["type"], "generate");
        assert_eq!(r["id"], "c-1");
        assert_eq!(r["attempt_id"], 2);
        assert_eq!(r["max_tokens"], 64);
        assert_eq!(r["messages"], msgs);
        assert_eq!(r["prompt"], "hello");
        assert_eq!(r["temperature"], 0.0);
        assert_eq!(r["repeat_penalty"], 1.0);
        assert_eq!(r["presence_penalty"], 0.0);
        assert_eq!(r["thinking_enabled"], false);
        assert_eq!(r["assistant_prefix"], "closed_think");
        assert!(r.get("top_k").is_none() && r.get("seed").is_none());

        let r = build_request(
            DriverKind::Sequential,
            false,
            &msgs,
            64,
            &Sampling::BETTERBENCH,
            "c",
            1,
        );
        assert!(r.get("serve_continuous_batch").is_none());
        assert_eq!(r["top_k"], 20);
        assert_eq!(r["seed"], 42);
        assert_eq!(r["temperature"], 0.7);

        let r = build_request(
            DriverKind::Slots,
            false,
            &msgs,
            64,
            &Sampling::GREEDY,
            "c",
            1,
        );
        assert_eq!(r["experimental_multi_slot"], true);
        assert!(r.get("serve_continuous_batch").is_none());
        assert_eq!(r["max_think_tokens"], 1);
        assert_eq!(r["repeat_penalty"], 1.0);
    }

    fn rec(t_send: f64, t_first: f64, t_end: f64, tokens: u64) -> RequestRecord {
        let step = (t_end - t_first) / (tokens - 1) as f64;
        RequestRecord {
            t_send,
            t_first: Some(t_first),
            t_end,
            token_times: (0..tokens).map(|i| t_first + step * i as f64).collect(),
            tokens,
            finish: Some("length".into()),
            tau: Some(2.0),
            batched: Some(true),
            ..Default::default()
        }
    }

    /// Closed loop of 2 lanes over 3 requests: throughput is tokens over
    /// first-send-to-last-end, TPOT the per-request decode step.
    #[test]
    fn metrics_follow_the_vllm_definitions() {
        let s = Sample {
            k: 2,
            records: vec![
                rec(0.0, 0.5, 2.5, 5),
                rec(0.0, 0.5, 4.5, 5),
                rec(2.5, 3.0, 5.0, 5),
            ],
            wall_s: 5.5,
        };
        let m = SampleMetrics::from_sample(&s);
        assert_eq!(m.output_tokens, 15);
        assert_eq!(m.duration_s, 5.0);
        assert!((m.output_tok_s - 3.0).abs() < 1e-12);
        assert!((m.bb_tok_s - 15.0 / 5.5).abs() < 1e-12);
        assert!((m.ttft.unwrap().p50 - 500.0).abs() < 1e-9);
        // TPOT: 2.0/4, 4.0/4, 2.0/4 s → p50 0.5 s.
        assert!((m.tpot.unwrap().p50 - 500.0).abs() < 1e-9);
        // Σ e2e = 2.5 + 4.5 + 2.5 = 9.5 over 5 s.
        assert!((m.concurrency - 1.9).abs() < 1e-12);
        assert_eq!(m.tau, Some(2.0));
        assert_eq!(m.finish.get("length"), Some(&3));
        assert_eq!(m.batched, 3);
        // Window: both first-k requests have a token by 0.5; last send 2.5,
        // first end at/after it 2.5 → tokens with stamps in [0.5, 2.5].
        let st = m.steady_tok_s.unwrap();
        // req0: all 5 (0.5..2.5); req1: 0.5,1.5,2.5 → 3; req2: none.
        assert!((st - 8.0 / 2.0).abs() < 1e-12, "{st}");
    }

    #[test]
    fn failed_requests_count_no_tokens_but_are_reported() {
        let mut bad = rec(0.0, 0.5, 1.0, 3);
        bad.err = Some("boom".into());
        let s = Sample {
            k: 1,
            records: vec![rec(0.0, 0.5, 2.0, 4), bad],
            wall_s: 2.0,
        };
        let m = SampleMetrics::from_sample(&s);
        assert_eq!(m.output_tokens, 4);
        assert_eq!(m.n_ok, 1);
        assert_eq!(m.errors, vec!["boom".to_owned()]);
        assert_eq!(m.finish.get("error"), Some(&1));
    }

    #[test]
    fn render_table_reports_median_throughput_and_range() {
        let mk = |end: f64| {
            SampleMetrics::from_sample(&Sample {
                k: 1,
                records: vec![rec(0.0, 0.5, end, 101)],
                wall_s: end,
            })
        };
        let points = vec![Point {
            backend: "batch",
            workload: "stateless",
            k: 1,
            samples: vec![mk(1.0), mk(2.0), mk(4.0)],
        }];
        let table = render_table(&points, BenchShape::Steady);
        // 101 tokens over 1/2/4 s → 101 / 50.5 / 25.25; median 50.50.
        assert!(table.contains("50.50 (25.25-101.00)"), "{table}");
        assert!(table.contains("length=3"), "{table}");
        assert!(!table.contains("bb agg tok/s"), "{table}");
        assert!(render_table(&points, BenchShape::BetterBench).contains("bb agg"));
    }

    #[test]
    fn multiturn_without_prefix_hits_is_invalid_on_slots() {
        let none = SampleMetrics::default();
        let hits = SampleMetrics {
            followup_prefix_hits: 2,
            ..Default::default()
        };
        assert!(sample_is_valid(WorkloadSel::Multiturn, "slots", &none).is_err());
        assert!(sample_is_valid(WorkloadSel::Multiturn, "slots", &hits).is_ok());
        assert!(sample_is_valid(WorkloadSel::Multiturn, "batch", &none).is_ok());
        assert!(sample_is_valid(WorkloadSel::Stateless, "slots", &none).is_ok());
    }

    #[test]
    fn batch_route_admits_vmm_and_fn_lanes_only() {
        assert!(is_batch_route(Some("vmm")));
        assert!(is_batch_route(Some("fn-lanes")));
        assert!(!is_batch_route(Some("legacy")));
        assert!(!is_batch_route(None));
    }
}
