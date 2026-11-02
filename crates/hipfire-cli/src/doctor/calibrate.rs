// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Calibration for `hipfire doctor --npu`: one warm `npu_trunk=auto` daemon
//! alternates the diagnostic GPU-only route and the auto route request by
//! request (so thermal and DPM drift hits both arms alike), then `on` and
//! `emulate` daemons must agree on a short greedy generation.
//!
//! The daemon's stderr is piped and never echoed: only its `npu-trunk:` lines
//! are kept, and the report quotes measurements, not log text.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::{BufRead, BufReader};
use std::process::{ChildStderr, Stdio};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use hipfire_client::Engine;

use super::checks::Target;
use super::report::{
    Calibration, Exactness, ExactnessVerdict, PairWait, RateScales, Samples, SizeResult,
    SizeVerdict,
};

/// Prompt sizes (rows) of the A/B, smallest first.
const SIZES: [usize; 2] = [16384, 32768];
/// Alternating passes over `SIZES`; even passes run GPU first, odd passes auto first.
const REPS: usize = 2;
const WARMUP_ROWS: usize = 16384;
/// A measured prompt below this is not the regime the trunk plans for.
const MIN_ROWS: u64 = 9000;
const MAX_SEQ: u64 = 40960;
/// One 8192-row chunk plus a 1127-row tail: the smallest prompt that forms a pair.
const EXACT_ROWS: usize = 9319;
const EXACT_NONCE: u64 = 0x00e0_ac71;
const EXACT_TOKENS: u64 = 16;
/// Used until the first warm-up measures the real ratio; high on purpose so the
/// first prompt errs short of the context limit.
const DEFAULT_TOKENS_PER_WORD: f64 = 1.5;
/// `HIPFIRE_QWEN4_NPU_TRUNK_SAMPLE_LOG` in the process snapshot (the developer
/// channel; the daemon reads it through `hipfire_config::developer_bool`).
const SAMPLE_LOG_KEY: &str = "developer.qwen4_npu_trunk_sample_log";
/// Stderr must go quiet this long before a request counts as fully logged.
const SETTLE_QUIET: Duration = Duration::from_millis(120);
const SETTLE_CAP: Duration = Duration::from_secs(2);

// ───────────────────────── plan ─────────────────────────

/// What `run` will do, printed before the confirmation prompt.
pub(crate) fn plan_lines() -> Vec<String> {
    vec![
        "one daemon with npu_trunk=auto alternates GPU-only and auto prefills at 16K and 32K \
         rows, 2 reps each, on prompts that never hit the prompt cache"
            .to_owned(),
        "then npu_trunk=on and npu_trunk=emulate daemons each prefill the same prompt and \
         generate 16 tokens greedily; the two outputs must match"
            .to_owned(),
        "about 6–7 minutes on a Strix Halo (three model loads plus about 2 minutes of prefills)"
            .to_owned(),
        "uses the GPU and the NPU".to_owned(),
        "changes no setting and writes no file".to_owned(),
    ]
}

// ───────────────────────── session ─────────────────────────

/// One finished request, as the daemon's `done` event reported it.
#[derive(Clone, Debug)]
pub(crate) struct Generated {
    pub ttft_ms: f64,
    /// Prefill rows plus any prompt-cache prefix (`bench_prompt_tokens_from_done`).
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub text: String,
    /// Generated token count from `done` (the streamed piece count when absent).
    pub tokens: u64,
    /// Text of each streamed `token` event, in order.
    pub pieces: Vec<String>,
}

/// What `run_with` needs of a loaded daemon; `DaemonSession` is the real one.
pub(crate) trait Session {
    /// Select the diagnostic GPU-only route (`true`) or auto (`false`) for the
    /// requests that follow.
    fn set_route(&mut self, gpu: bool) -> Result<()>;
    /// Run one greedy request. Returns only after the request's stderr lines
    /// are collected, so `npu_lines` can be sliced per request.
    fn generate(&mut self, prompt: &str, max_tokens: u64) -> Result<Generated>;
    /// Every `npu-trunk:` / `WARN npu-trunk:` stderr line so far, load included.
    fn npu_lines(&self) -> Vec<String>;
    fn finish(self: Box<Self>) -> Result<()>;
}

/// A real `hipfire-daemon` child with the model loaded and stderr captured.
pub(crate) struct DaemonSession {
    engine: Engine,
    lines: Arc<Mutex<Vec<String>>>,
    reader: Option<JoinHandle<()>>,
}

impl DaemonSession {
    /// Resolve the model's config the way `hipfire bench` does, force
    /// `fusions.npu_trunk = mode`, spawn the daemon with piped stderr and load
    /// the model with `max_seq` context. `sample_log` turns on the per-pair
    /// `pair-sample` lines.
    pub(crate) fn start(
        paths: &crate::Paths,
        target: &Target,
        mode: &str,
        sample_log: bool,
        max_seq: u64,
    ) -> Result<Self> {
        let model_path = target.model_path.as_path();
        let model_arg = model_path.to_string_lossy().into_owned();
        let registry = hipfire_registry::load(&paths.registry).registry;
        let mut card = crate::model_card(paths, &registry, &model_arg);
        if card.pair().is_none() {
            if let Some(tag) = target.model_tag.as_deref() {
                card = crate::model_card_by_name(&registry, tag);
            }
        }
        let (tag, entry) = (card.tag().map(str::to_owned), card.entry().cloned());
        let resolved = crate::resolved_for_model(
            paths,
            tag.as_deref().unwrap_or(&model_arg),
            tag.as_deref(),
            entry.as_ref(),
        )?;
        // No download: the doctor writes no file. A missing PLE sidecar is an
        // error naming the pull command.
        crate::ple::ensure_ple_sidecar(
            entry.as_ref(),
            tag.as_deref(),
            model_path,
            crate::ple::request_from_config(&resolved),
            &crate::ple::load_target_arch(&resolved),
            false,
        )?;
        let mut process_config = hipfire_config::ProcessConfig::from_resolved(&resolved)?;
        process_config.values.set_cli("fusions.npu_trunk", mode)?;
        if sample_log {
            process_config.values.set_cli(SAMPLE_LOG_KEY, "1")?;
        }
        let engine = Engine::spawn_configured_with_stderr(
            &target.daemon,
            &BTreeMap::new(),
            &process_config,
            Some(model_path),
            Stdio::piped(),
        )?;
        let stderr = engine
            .take_stderr()
            .ok_or_else(|| anyhow!("the daemon's stderr was not captured"))?;
        let lines = Arc::new(Mutex::new(Vec::new()));
        let reader = spawn_reader(stderr, Arc::clone(&lines));
        let session = Self {
            engine,
            lines,
            reader: Some(reader),
        };
        session.engine.ping()?;
        let max_tokens = crate::config_u64(&resolved, "generation.max_tokens")?;
        let mut params = crate::load_params(
            &resolved,
            entry.as_ref(),
            &paths.models,
            model_path,
            max_tokens,
            None,
            None,
            tag.as_deref(),
            false,
            None,
        )?;
        crate::resolve_dflash_sidecar(
            &mut params,
            entry.as_ref(),
            &paths.models,
            model_path,
            tag.as_deref(),
        )?;
        crate::resolve_mtp_sidecar(
            &mut params,
            entry.as_ref(),
            &paths.models,
            model_path,
            &model_arg,
            tag.as_deref(),
        );
        params["max_seq"] = serde_json::json!(max_seq);
        session
            .engine
            .load(model_path, params)
            .with_context(|| format!("loading the model with npu_trunk={mode}"))?;
        session.settle();
        Ok(session)
    }

    pub(crate) fn set_route(&mut self, gpu: bool) -> Result<()> {
        let want = if gpu { "gpu" } else { "auto" };
        let reply = self
            .engine
            .request(&serde_json::json!({ "type": "npu_trunk_route", "route": want }))
            .context("selecting the diagnostic npu-trunk route")?;
        match reply.get("type").and_then(serde_json::Value::as_str) {
            Some("npu_trunk_route")
                if reply.get("route").and_then(serde_json::Value::as_str) == Some(want) =>
            {
                Ok(())
            }
            Some("error") => bail!(
                "the daemon refused the diagnostic route: {}",
                reply
                    .get("message")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("no message")
            ),
            _ => bail!("unexpected reply to npu_trunk_route: {reply}"),
        }
    }

    pub(crate) fn generate(&mut self, prompt: &str, max_tokens: u64) -> Result<Generated> {
        let request = crate::bench_generate_request(prompt, max_tokens);
        let mut pieces = Vec::new();
        let done = self.engine.generate(&request, |event| {
            if event.get("type").and_then(serde_json::Value::as_str) == Some("token") {
                if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                    pieces.push(text.to_owned());
                }
            }
            Ok(())
        })?;
        self.settle();
        let ttft_ms = done
            .get("ttft_ms")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| anyhow!("the daemon's done event carried no ttft_ms"))?;
        let prompt_tokens = crate::bench_prompt_tokens_from_done(&done)
            .ok_or_else(|| anyhow!("the daemon's done event carried no prompt token count"))?;
        let cached_tokens = done
            .get("cached_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let tokens = done
            .get("tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(pieces.len() as u64);
        Ok(Generated {
            ttft_ms,
            prompt_tokens,
            cached_tokens,
            text: pieces.concat(),
            tokens,
            pieces,
        })
    }

    pub(crate) fn npu_lines(&self) -> Vec<String> {
        lock(&self.lines).clone()
    }

    /// A clone of the engine handle, for a watchdog that calls `terminate()` /
    /// `exited()` while `generate` blocks on another thread.
    pub(crate) fn engine_handle(&self) -> Engine {
        self.engine.clone()
    }

    /// Unload, then kill and reap the daemon so its memory is free before the
    /// next one starts. The kill happens even when the unload fails.
    pub(crate) fn finish(self) -> Result<()> {
        let unloaded = self.engine.unload();
        self.engine.terminate();
        if let Some(reader) = self.reader {
            let _ = reader.join();
        }
        unloaded.map_err(anyhow::Error::from)
    }

    /// Wait until the stderr reader has caught up with the daemon. The daemon
    /// writes a request's lines before its `done`, so they are in the pipe
    /// already; this only waits out the reader thread's scheduling.
    fn settle(&self) {
        let deadline = Instant::now() + SETTLE_CAP;
        let mut count = lock(&self.lines).len();
        let mut quiet_since = Instant::now();
        while Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
            let now = lock(&self.lines).len();
            if now != count {
                count = now;
                quiet_since = Instant::now();
            } else if quiet_since.elapsed() >= SETTLE_QUIET {
                break;
            }
        }
    }
}

impl Session for DaemonSession {
    fn set_route(&mut self, gpu: bool) -> Result<()> {
        DaemonSession::set_route(self, gpu)
    }
    fn generate(&mut self, prompt: &str, max_tokens: u64) -> Result<Generated> {
        DaemonSession::generate(self, prompt, max_tokens)
    }
    fn npu_lines(&self) -> Vec<String> {
        DaemonSession::npu_lines(self)
    }
    fn finish(self: Box<Self>) -> Result<()> {
        DaemonSession::finish(*self)
    }
}

/// Poison-tolerant lock: a panicked reader thread must not hide the lines it
/// already collected.
fn lock(lines: &Mutex<Vec<String>>) -> MutexGuard<'_, Vec<String>> {
    lines.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Drain the daemon's stderr for its whole life (a full pipe would block it),
/// keeping only the trunk lines and echoing nothing.
fn spawn_reader(stderr: ChildStderr, lines: Arc<Mutex<Vec<String>>>) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut reader = BufReader::new(stderr);
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            let text = String::from_utf8_lossy(&buf);
            let line = text.trim_end();
            if is_trunk_line(line) {
                lock(&lines).push(line.to_owned());
            }
        }
    })
}

// ───────────────────────── log-line parsers ─────────────────────────
//
// Formats: `crates/hipfire-arch-qwen4/src/npu_trunk/{forward,admission}.rs`.

fn is_trunk_line(line: &str) -> bool {
    line.starts_with("npu-trunk:") || line.starts_with("WARN npu-trunk:")
}

/// The text after `npu-trunk: `, with an optional `WARN ` level prefix removed.
fn trunk_body(line: &str) -> Option<&str> {
    line.strip_prefix("WARN ")
        .unwrap_or(line)
        .strip_prefix("npu-trunk: ")
}

/// The load-time line of an armed `auto` trunk; carries the trunk route.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Armed {
    pub trunk_route: Option<String>,
}

/// `npu-trunk: mode=auto families=0xf contract=bf16_direct trunk_route=stock …`
pub(crate) fn parse_armed(line: &str) -> Option<Armed> {
    if line.starts_with("WARN ") {
        return None;
    }
    let body = trunk_body(line)?;
    if !body.starts_with("mode=auto families=") {
        return None;
    }
    let trunk_route = body
        .split_whitespace()
        .find_map(|token| token.strip_prefix("trunk_route="))
        .map(str::to_owned);
    Some(Armed { trunk_route })
}

/// `npu-trunk: off (<reason>)` or `WARN npu-trunk: off (<reason>); npu_trunk=auto
/// requested the NPU but …`; returns the reason.
pub(crate) fn parse_off(line: &str) -> Option<String> {
    let rest = trunk_body(line)?.strip_prefix("off (")?;
    let reason = match rest.find("); npu_trunk=auto requested") {
        Some(end) => &rest[..end],
        None => rest.strip_suffix(')').unwrap_or(rest),
    };
    Some(reason.to_owned())
}

/// `npu-trunk: rows=16384 families=… rates=gen3 gpu_x1.020 npu_x0.980[ frozen]`
pub(crate) fn parse_rates(line: &str) -> Option<RateScales> {
    let body = trunk_body(line)?;
    if !body.starts_with("rows=") {
        return None;
    }
    let tail = &body[body.find(" rates=gen")? + " rates=gen".len()..];
    let mut tokens = tail.split_whitespace();
    let generation = tokens.next()?.parse().ok()?;
    let gpu_x = tokens.next()?.strip_prefix("gpu_x")?.parse().ok()?;
    let npu_x = tokens.next()?.strip_prefix("npu_x")?.parse().ok()?;
    let frozen = tokens.next() == Some("frozen");
    Some(RateScales {
        gpu_x,
        npu_x,
        frozen,
        generation,
    })
}

/// `npu-trunk: pair-sample row=0 rows=8192+8192 … measured npu=… wait=12.5 pair=… | …`;
/// returns the measured `wait=` (ms the GPU stalled on the NPU).
pub(crate) fn parse_pair_wait_ms(line: &str) -> Option<f64> {
    let body = trunk_body(line)?.strip_prefix("pair-sample ")?;
    body.split_whitespace()
        .find_map(|token| token.strip_prefix("wait="))?
        .parse()
        .ok()
}

/// `npu-trunk: guard: pair at row 8192: the GPU waited 912 ms on the NPU, …`
pub(crate) fn is_guard_line(line: &str) -> bool {
    trunk_body(line).is_some_and(|body| body.starts_with("guard:"))
}

/// `npu-trunk: rows=16384 route=gpu (diagnostic)`
pub(crate) fn is_gpu_route_line(line: &str) -> bool {
    trunk_body(line)
        .is_some_and(|body| body.starts_with("rows=") && body.contains(" route=gpu (diagnostic)"))
}

/// `npu-trunk: pair rows=8192+1127 position=0 families=…` (`on`/`emulate`; logged once).
pub(crate) fn is_pair_line(line: &str) -> bool {
    trunk_body(line).is_some_and(|body| body.starts_with("pair rows="))
}

// ───────────────────────── statistics ─────────────────────────

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

/// (max - min) / median * 100 of one arm.
fn spread_pct(values: &[f64]) -> f64 {
    let med = median(values);
    if values.is_empty() || med <= 0.0 {
        return 0.0;
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (max - min) / med * 100.0
}

/// A size is a gain only when the saving exceeds the run-to-run spread of
/// either arm, a loss only when the loss does; anything else is noise.
fn size_verdict(saved_pct: f64, spread_pct: f64) -> SizeVerdict {
    if saved_pct > spread_pct {
        SizeVerdict::Gain
    } else if saved_pct < -spread_pct {
        SizeVerdict::Loss
    } else {
        SizeVerdict::NoMeasurableGain
    }
}

/// One prompt size from the GPU-only (`off`) and `auto` TTFT samples.
pub(crate) fn size_result(rows: u64, off: &[f64], auto: &[f64]) -> SizeResult {
    let off_median = median(off);
    let auto_median = median(auto);
    let saved_pct = if off_median > 0.0 {
        (off_median - auto_median) / off_median * 100.0
    } else {
        0.0
    };
    let spread = spread_pct(off).max(spread_pct(auto));
    SizeResult {
        rows,
        off: Samples {
            samples_ms: off.to_vec(),
            median_ms: off_median,
        },
        auto: Samples {
            samples_ms: auto.to_vec(),
            median_ms: auto_median,
        },
        saved_pct,
        spread_pct: spread,
        verdict: size_verdict(saved_pct, spread),
    }
}

fn pair_wait(lines: &[String]) -> Option<PairWait> {
    let waits: Vec<f64> = lines.iter().filter_map(|l| parse_pair_wait_ms(l)).collect();
    if waits.is_empty() {
        return None;
    }
    Some(PairWait {
        pairs: waits.len() as u64,
        median_ms: median(&waits),
        max_ms: waits.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    })
}

// ───────────────────────── synthetic prompts ─────────────────────────

const WORDS: &[&str] = &[
    "river", "stone", "window", "engine", "garden", "silver", "quiet", "harbor", "market",
    "lantern", "forest", "bridge", "copper", "meadow", "signal", "winter", "candle", "orchard",
    "thunder", "canvas", "pebble", "anchor", "violet", "compass", "ladder", "mirror", "village",
    "pocket", "saddle", "feather", "tunnel", "blanket", "kettle", "mountain", "ribbon", "shadow",
    "basket", "island", "needle", "paper", "summer", "castle", "bottle", "carpet", "thistle",
    "wagon", "marble", "whisper", "cabin", "morning", "valley", "button", "cloud", "doorway",
    "fabric", "glacier", "hammer", "journey", "kitchen", "letter", "mineral", "number", "ocean",
    "painter", "quarry", "rabbit", "season", "temple", "umbrella", "vessel", "weaver", "yellow",
    "zephyr", "balance", "chimney", "dolphin", "ember", "falcon", "granite", "horizon", "iron",
    "jasmine", "kernel", "lumber", "mosaic", "nectar", "oyster", "prairie", "rhythm", "sparrow",
    "timber", "velvet", "willow", "crystal", "dragon", "echo", "frost", "gentle", "honey",
];

/// splitmix64: a tiny deterministic generator; the prompts need no better.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// `words` pseudo-words of text under a first line that is a hash of `nonce`,
/// so two nonces never share a prefix longer than a character or two and the
/// daemon's prompt cache can never serve one from another.
fn render_prompt(nonce: u64, words: usize) -> String {
    let mut rng = SplitMix64(nonce);
    let tag = rng.next();
    let mut out = String::with_capacity(words * 8 + 80);
    let _ = writeln!(
        out,
        "{tag:016x} synthetic prefill workload, nothing to answer."
    );
    let mut in_sentence = 0usize;
    let mut sentence_len = 0usize;
    let mut sentences = 0usize;
    for index in 0..words {
        if in_sentence == 0 {
            sentence_len = 6 + (rng.next() % 9) as usize;
        }
        let word = WORDS[(rng.next() % WORDS.len() as u64) as usize];
        if in_sentence == 0 {
            let mut chars = word.chars();
            if let Some(first) = chars.next() {
                out.push(first.to_ascii_uppercase());
                out.push_str(chars.as_str());
            }
        } else {
            out.push(' ');
            out.push_str(word);
        }
        in_sentence += 1;
        if in_sentence == sentence_len || index + 1 == words {
            out.push('.');
            sentences += 1;
            in_sentence = 0;
            out.push(if sentences % 5 == 0 { '\n' } else { ' ' });
        }
    }
    out.truncate(out.trim_end().len());
    out
}

/// Turns a target token count into a word count from the tokens-per-word ratio
/// of this tokenizer. The first request to report a prompt length fixes the
/// ratio; until then a conservative default is used.
pub(crate) struct Sizer {
    tokens_per_word: f64,
    calibrated: bool,
}

impl Sizer {
    pub(crate) fn new() -> Self {
        Self {
            tokens_per_word: DEFAULT_TOKENS_PER_WORD,
            calibrated: false,
        }
    }

    /// The prompt for `nonce` aimed at `target_tokens`, and the word count that
    /// `observe` needs.
    pub(crate) fn prompt(&self, nonce: u64, target_tokens: usize) -> (String, usize) {
        let words = ((target_tokens as f64 / self.tokens_per_word).round() as usize).max(1);
        (render_prompt(nonce, words), words)
    }

    /// Record the measured prompt length of a prompt built with `words` words.
    /// Only the first observation counts, so every later prompt of one run is
    /// sized from the same ratio.
    pub(crate) fn observe(&mut self, prompt_tokens: u64, words: usize) {
        if self.calibrated || prompt_tokens == 0 || words == 0 {
            return;
        }
        self.tokens_per_word = prompt_tokens as f64 / words as f64;
        self.calibrated = true;
    }
}

/// Deterministic pseudo-text of about `target_tokens` tokens at the default
/// tokens-per-word ratio; use [`Sizer`] to land closer once a ratio is known.
pub(crate) fn synthetic_prompt(nonce: u64, target_tokens: usize) -> String {
    Sizer::new().prompt(nonce, target_tokens).0
}

// ───────────────────────── orchestration ─────────────────────────

/// Run the calibration against the real daemon.
pub(crate) fn run(
    paths: &crate::Paths,
    target: &Target,
    progress: &mut dyn FnMut(&str),
) -> Result<Calibration> {
    run_with(
        |mode, sample_log| {
            let session: Box<dyn Session> = Box::new(DaemonSession::start(
                paths, target, mode, sample_log, MAX_SEQ,
            )?);
            Ok(session)
        },
        progress,
    )
}

/// Per-run request state shared by every request of the auto session.
struct State {
    /// The route the session is on; a session starts on auto.
    gpu_route: bool,
    sizer: Sizer,
    nonce: u64,
}

struct Request {
    generated: Generated,
    /// The `npu-trunk:` lines this request logged.
    lines: Vec<String>,
}

fn issue(session: &mut dyn Session, state: &mut State, gpu: bool, rows: usize) -> Result<Request> {
    if state.gpu_route != gpu {
        session.set_route(gpu)?;
        state.gpu_route = gpu;
    }
    state.nonce += 1;
    let (prompt, words) = state.sizer.prompt(state.nonce, rows);
    let before = session.npu_lines().len();
    let generated = session.generate(&prompt, 1)?;
    state.sizer.observe(generated.prompt_tokens, words);
    let mut lines = session.npu_lines();
    let lines = lines.split_off(before.min(lines.len()));
    Ok(Request { generated, lines })
}

/// Checks that hold for every A/B request: a cold prefill on the route asked for.
fn check_request(request: &Request, gpu: bool, rows: usize) -> Result<()> {
    if request.generated.cached_tokens > 0 {
        bail!(
            "a {rows}-row calibration request was served {} rows from the prompt cache, so its \
             timing is not a cold prefill",
            request.generated.cached_tokens
        );
    }
    let gpu_lines = request
        .lines
        .iter()
        .filter(|line| is_gpu_route_line(line))
        .count();
    match (gpu, gpu_lines) {
        (true, 1) | (false, 0) => Ok(()),
        (true, 0) => bail!(
            "this daemon does not support the diagnostic GPU route (a GPU-only request logged no \
             `route=gpu (diagnostic)` line)"
        ),
        (true, n) => bail!("a GPU-only request logged {n} `route=gpu (diagnostic)` lines, not one"),
        (false, _) => bail!("an auto request ran on the diagnostic GPU route"),
    }
}

#[derive(Default)]
struct Arm {
    ms: Vec<f64>,
    rows: Vec<u64>,
}

impl Arm {
    fn push(&mut self, generated: &Generated) {
        self.ms.push(generated.ttft_ms);
        self.rows.push(generated.prompt_tokens);
    }
}

fn median_rows(rows: &[u64]) -> u64 {
    let values: Vec<f64> = rows.iter().map(|&r| r as f64).collect();
    median(&values).round() as u64
}

/// Drive the whole calibration through `open(mode, sample_log)`, which loads a
/// model in a fresh daemon with `npu_trunk` set to `mode`.
pub(crate) fn run_with(
    mut open: impl FnMut(&str, bool) -> Result<Box<dyn Session>>,
    progress: &mut dyn FnMut(&str),
) -> Result<Calibration> {
    let started = Instant::now();
    progress("[doctor] loading the model (npu_trunk=auto)…");
    let mut session = open("auto", true)?;
    let load_lines = session.npu_lines();
    let Some(armed) = load_lines.iter().find_map(|line| parse_armed(line)) else {
        let reason = load_lines
            .iter()
            .find_map(|line| parse_off(line))
            .unwrap_or_else(|| "the daemon logged no npu-trunk status line".to_owned());
        progress(&format!(
            "[doctor] the NPU trunk did not arm under npu_trunk=auto: {reason}"
        ));
        session.finish()?;
        return Ok(Calibration {
            armed: false,
            off_reason: Some(reason),
            trunk_route: None,
            sizes: Vec::new(),
            rate_scales: None,
            npu_wait_per_pair: None,
            guard_trips: 0,
            exactness: None,
            duration_s: started.elapsed().as_secs_f64(),
        });
    };

    let mut state = State {
        gpu_route: false,
        sizer: Sizer::new(),
        nonce: 0,
    };
    progress("[doctor] warming up (GPU-only, then auto, 16384 rows; not measured)…");
    for gpu in [true, false] {
        let request = issue(&mut *session, &mut state, gpu, WARMUP_ROWS)?;
        check_request(&request, gpu, WARMUP_ROWS)?;
    }

    let mut off: [Arm; SIZES.len()] = Default::default();
    let mut auto: [Arm; SIZES.len()] = Default::default();
    let mut auto_lines: Vec<String> = Vec::new();
    for rep in 0..REPS {
        for (index, &size) in SIZES.iter().enumerate() {
            // Alternate the order so a drift in clocks cannot favour one arm.
            let order = if rep % 2 == 0 {
                [true, false]
            } else {
                [false, true]
            };
            for gpu in order {
                let request = issue(&mut *session, &mut state, gpu, size)?;
                check_request(&request, gpu, size)?;
                if request.generated.prompt_tokens < MIN_ROWS {
                    bail!(
                        "a {size}-row calibration prompt came back as {} rows, below the \
                         {MIN_ROWS} the trunk plans for",
                        request.generated.prompt_tokens
                    );
                }
                progress(&format!(
                    "[doctor] {size} rows: {} {:.0} ms",
                    if gpu { "gpu" } else { "auto" },
                    request.generated.ttft_ms
                ));
                if gpu {
                    off[index].push(&request.generated);
                } else {
                    auto[index].push(&request.generated);
                    auto_lines.extend(request.lines);
                }
            }
        }
    }
    session.finish()?;

    let sizes: Vec<SizeResult> = (0..SIZES.len())
        .map(|i| {
            let rows: Vec<u64> = off[i].rows.iter().chain(&auto[i].rows).copied().collect();
            size_result(median_rows(&rows), &off[i].ms, &auto[i].ms)
        })
        .collect();
    let rate_scales = auto_lines.iter().rev().find_map(|line| parse_rates(line));
    let guard_trips = auto_lines.iter().filter(|l| is_guard_line(l)).count() as u64;
    let npu_wait_per_pair = pair_wait(&auto_lines);

    let exactness = exactness(&mut open, progress, &state.sizer)?;
    Ok(Calibration {
        armed: true,
        off_reason: None,
        trunk_route: armed.trunk_route,
        sizes,
        rate_scales,
        npu_wait_per_pair,
        guard_trips,
        exactness: Some(exactness),
        duration_s: started.elapsed().as_secs_f64(),
    })
}

/// `on` and `emulate` must generate the same greedy tokens from one prompt
/// that is large enough to form an NPU pair.
fn exactness(
    open: &mut impl FnMut(&str, bool) -> Result<Box<dyn Session>>,
    progress: &mut dyn FnMut(&str),
    sizer: &Sizer,
) -> Result<Exactness> {
    let (prompt, _) = sizer.prompt(EXACT_NONCE, EXACT_ROWS);
    progress("[doctor] exactness: loading the model (npu_trunk=on)…");
    let mut on = open("on", false)?;
    let on_run = on.generate(&prompt, EXACT_TOKENS)?;
    let exercised = on.npu_lines().iter().any(|line| is_pair_line(line));
    on.finish()?;
    if on_run.tokens == 0 {
        bail!("the exactness prompt generated no tokens under npu_trunk=on");
    }
    if !exercised {
        return Ok(Exactness {
            verdict: ExactnessVerdict::NotExercised,
            prompt_rows: on_run.prompt_tokens,
            tokens: on_run.tokens,
            first_difference: None,
        });
    }
    progress("[doctor] exactness: loading the model (npu_trunk=emulate)…");
    let mut emulate = open("emulate", false)?;
    let emulate_run = emulate.generate(&prompt, EXACT_TOKENS)?;
    emulate.finish()?;
    Ok(compare(&on_run, &emulate_run))
}

fn compare(on: &Generated, emulate: &Generated) -> Exactness {
    let mut first = on
        .pieces
        .iter()
        .zip(&emulate.pieces)
        .position(|(a, b)| a != b)
        .or_else(|| {
            (on.pieces.len() != emulate.pieces.len())
                .then(|| on.pieces.len().min(emulate.pieces.len()))
        });
    let same = first.is_none() && on.text == emulate.text && on.tokens == emulate.tokens;
    if !same && first.is_none() && on.tokens != emulate.tokens {
        first = Some(on.tokens.min(emulate.tokens) as usize);
    }
    Exactness {
        verdict: if same {
            ExactnessVerdict::Exact
        } else {
            ExactnessVerdict::Mismatch
        },
        prompt_rows: on.prompt_tokens,
        tokens: on.tokens,
        first_difference: first.map(|index| index as u64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    // Real-format lines (see the format strings quoted in each parser's docs).
    const ARMED: &str = "npu-trunk: mode=auto families=0xf contract=bf16_direct \
        trunk_route=overlay:b2 zab_source=stock-MQ6-DtoH pack_ms=1.000 download_ms=2.000 \
        resident_bytes=1053843456 npu_panels_bytes=1 gpu_twin_bytes=2 bank_b_bytes=3 \
        runtime=open raw=DtoH(no-retained-host-copy)";
    const CAL_LINE: &str = "npu-trunk: mode=auto contract=bf16 families=qkv,zab,q,kvidx \
        cal_key={} cal_sha256=ab n*=4736 T(16384)=1.0/2.0 T(65536)=3.0/4.0";
    const OFF: &str = "npu-trunk: off (requires Flash-Next family B)";
    const WARN_OFF: &str = "WARN npu-trunk: off (mmap heap: errno=11); npu_trunk=auto requested \
        the NPU but the trunk did not arm, prefill runs GPU-only (npu_trunk=on makes this a load \
        error)";
    const RATES: &str = "npu-trunk: rows=16384 families=qkv,zab,q,kvidx plan=[pipe \
        0..8192+8192..16384] predicted=-14.7% rates=gen3 gpu_x1.020 npu_x0.980";
    const RATES_FROZEN: &str = "npu-trunk: rows=32768 families=qkv,zab plan=[gpu] \
        predicted=0.0% rates=gen12 gpu_x1.100 npu_x0.900 frozen";
    const PAIR_SAMPLE: &str = "npu-trunk: pair-sample row=0 rows=8192+8192 families=qkv,zab \
        measured npu=410.2 window=520.0 wait=12.5 pair=1043.7 | plan npu=400.0 window=500.0 | \
        prior npu=400.0 window=500.0 | now gen2 gpu_x1.020 npu_x0.980";
    const GUARD: &str = "npu-trunk: guard: pair at row 8192: the GPU waited 912 ms on the NPU, \
        more than the 400 ms pairing saves it (pair 2400 ms; gpu_x1.000 npu_x1.000); the rest of \
        this request runs GPU-only";
    const GPU_ROUTE: &str = "npu-trunk: rows=16384 route=gpu (diagnostic)";
    const PAIR: &str = "npu-trunk: pair rows=8192+1127 position=0 families=qkv,zab,q,kvidx";

    #[test]
    fn parse_armed_reads_the_route() {
        assert_eq!(
            parse_armed(ARMED),
            Some(Armed {
                trunk_route: Some("overlay:b2".into())
            })
        );
        assert_eq!(
            parse_armed(&ARMED.replace("overlay:b2", "stock")),
            Some(Armed {
                trunk_route: Some("stock".into())
            })
        );
    }

    #[test]
    fn parse_armed_rejects_other_trunk_lines() {
        // The calibration line also starts `mode=auto` but is not the armed line.
        assert_eq!(parse_armed(CAL_LINE), None);
        assert_eq!(parse_armed(&ARMED.replace("mode=auto", "mode=on")), None);
        assert_eq!(parse_armed(OFF), None);
        assert_eq!(parse_armed(&format!("WARN {ARMED}")), None);
        assert_eq!(parse_armed("[qwen-cache resume] mode=auto families="), None);
    }

    #[test]
    fn parse_off_plain_and_warn() {
        assert_eq!(
            parse_off(OFF).as_deref(),
            Some("requires Flash-Next family B")
        );
        assert_eq!(parse_off(WARN_OFF).as_deref(), Some("mmap heap: errno=11"));
        assert_eq!(parse_off(ARMED), None);
        assert_eq!(parse_off(RATES), None);
    }

    #[test]
    fn parse_off_keeps_parentheses_inside_the_reason() {
        assert_eq!(
            parse_off("npu-trunk: off (calibration key mismatch: gpu (gfx1100))").as_deref(),
            Some("calibration key mismatch: gpu (gfx1100)")
        );
    }

    #[test]
    fn parse_rates_without_frozen() {
        assert_eq!(
            parse_rates(RATES),
            Some(RateScales {
                gpu_x: 1.02,
                npu_x: 0.98,
                frozen: false,
                generation: 3
            })
        );
    }

    #[test]
    fn parse_rates_with_frozen() {
        assert_eq!(
            parse_rates(RATES_FROZEN),
            Some(RateScales {
                gpu_x: 1.1,
                npu_x: 0.9,
                frozen: true,
                generation: 12
            })
        );
    }

    #[test]
    fn parse_rates_ignores_lines_without_rates() {
        assert_eq!(parse_rates(GPU_ROUTE), None);
        assert_eq!(parse_rates(PAIR_SAMPLE), None);
        assert_eq!(parse_rates(ARMED), None);
    }

    #[test]
    fn parse_pair_wait_reads_the_measured_wait() {
        assert_eq!(parse_pair_wait_ms(PAIR_SAMPLE), Some(12.5));
        assert_eq!(parse_pair_wait_ms(RATES), None);
        assert_eq!(parse_pair_wait_ms(PAIR), None);
    }

    #[test]
    fn guard_route_and_pair_lines_are_told_apart() {
        assert!(is_guard_line(GUARD));
        assert!(!is_guard_line(PAIR_SAMPLE));
        assert!(is_gpu_route_line(GPU_ROUTE));
        assert!(!is_gpu_route_line(RATES));
        assert!(is_pair_line(PAIR));
        assert!(!is_pair_line(PAIR_SAMPLE));
    }

    #[test]
    fn only_trunk_lines_are_kept() {
        assert!(is_trunk_line(ARMED));
        assert!(is_trunk_line(WARN_OFF));
        assert!(!is_trunk_line("[qwen-cache resume] rewound"));
        assert!(!is_trunk_line("  npu-trunk: indented"));
    }

    #[test]
    fn size_result_math() {
        let r = size_result(16384, &[1000.0, 1010.0], &[800.0, 810.0]);
        assert_eq!(r.rows, 16384);
        assert_eq!(r.off.median_ms, 1005.0);
        assert_eq!(r.auto.median_ms, 805.0);
        assert!((r.saved_pct - 19.900_497_5).abs() < 1e-6, "{}", r.saved_pct);
        // Both arms span 10 ms; the larger relative spread is the auto arm's.
        assert!((r.spread_pct - 10.0 / 805.0 * 100.0).abs() < 1e-9);
        assert_eq!(r.verdict, SizeVerdict::Gain);
    }

    #[test]
    fn size_verdicts_and_boundaries() {
        assert_eq!(size_verdict(5.0, 1.0), SizeVerdict::Gain);
        assert_eq!(size_verdict(-5.0, 1.0), SizeVerdict::Loss);
        assert_eq!(size_verdict(0.5, 1.0), SizeVerdict::NoMeasurableGain);
        assert_eq!(size_verdict(-0.5, 1.0), SizeVerdict::NoMeasurableGain);
        // Exactly at the spread is not enough either way.
        assert_eq!(size_verdict(1.0, 1.0), SizeVerdict::NoMeasurableGain);
        assert_eq!(size_verdict(-1.0, 1.0), SizeVerdict::NoMeasurableGain);
        assert_eq!(size_verdict(0.0, 0.0), SizeVerdict::NoMeasurableGain);
    }

    #[test]
    fn size_result_loss_and_spread_dominated_gain() {
        let loss = size_result(32768, &[1000.0, 1000.0], &[1200.0, 1200.0]);
        assert!((loss.saved_pct + 20.0).abs() < 1e-9);
        assert_eq!(loss.spread_pct, 0.0);
        assert_eq!(loss.verdict, SizeVerdict::Loss);
        // A 2% saving inside a ~20% spread proves nothing.
        let noisy = size_result(16384, &[900.0, 1100.0], &[880.0, 1080.0]);
        assert!(noisy.saved_pct > 0.0);
        assert!(noisy.spread_pct > noisy.saved_pct);
        assert_eq!(noisy.verdict, SizeVerdict::NoMeasurableGain);
    }

    #[test]
    fn pair_wait_summarises_samples() {
        let lines = vec![
            PAIR_SAMPLE.to_owned(),
            PAIR_SAMPLE.replace("wait=12.5", "wait=30.0"),
            PAIR_SAMPLE.replace("wait=12.5", "wait=2.0"),
            GUARD.to_owned(),
        ];
        let wait = pair_wait(&lines).unwrap();
        assert_eq!(wait.pairs, 3);
        assert_eq!(wait.median_ms, 12.5);
        assert_eq!(wait.max_ms, 30.0);
        assert!(pair_wait(&[GUARD.to_owned()]).is_none());
    }

    #[test]
    fn synthetic_prompt_is_deterministic() {
        assert_eq!(synthetic_prompt(7, 2000), synthetic_prompt(7, 2000));
        assert_ne!(synthetic_prompt(7, 2000), synthetic_prompt(8, 2000));
        let long = synthetic_prompt(7, 4000);
        let short = synthetic_prompt(7, 2000);
        assert!(long.len() > short.len());
    }

    #[test]
    fn synthetic_prompts_have_distinct_nonce_prefixes() {
        let mut tags = std::collections::BTreeSet::new();
        for nonce in 0..200u64 {
            let prompt = synthetic_prompt(nonce, 200);
            let tag = prompt.split_whitespace().next().unwrap().to_owned();
            assert_eq!(tag.len(), 16, "{prompt}");
            assert!(tags.insert(tag), "nonce {nonce} repeated a prefix");
        }
        // The tag is a hash of the nonce, so the first characters differ too.
        let a = synthetic_prompt(1, 200);
        let b = synthetic_prompt(2, 200);
        assert_ne!(a[..16], b[..16]);
    }

    #[test]
    fn sizer_uses_the_first_observed_ratio_only() {
        let mut sizer = Sizer::new();
        let (_, default_words) = sizer.prompt(1, 15000);
        assert_eq!(default_words, 10000);
        sizer.observe(12000, 10000);
        assert_eq!(sizer.prompt(2, 12000).1, 10000);
        sizer.observe(99999, 10000);
        assert_eq!(sizer.prompt(3, 12000).1, 10000);
    }

    // ───── fake session ─────

    #[derive(Clone)]
    struct Cfg {
        load_lines: Vec<String>,
        gpu_route_line: bool,
        cached: u64,
        auto_factor: f64,
        guard: bool,
        pairs: bool,
        on_text: String,
        emulate_text: String,
    }

    impl Cfg {
        fn armed() -> Self {
            Self {
                load_lines: vec![CAL_LINE.to_owned(), ARMED.to_owned()],
                gpu_route_line: true,
                cached: 0,
                auto_factor: 0.8,
                guard: false,
                pairs: true,
                on_text: "the river carried a quiet stone".to_owned(),
                emulate_text: "the river carried a quiet stone".to_owned(),
            }
        }
    }

    struct Fake {
        mode: String,
        cfg: Cfg,
        log: Rc<RefCell<Vec<String>>>,
        gpu: bool,
        requests: usize,
        lines: Vec<String>,
    }

    impl Session for Fake {
        fn set_route(&mut self, gpu: bool) -> Result<()> {
            self.gpu = gpu;
            Ok(())
        }
        fn generate(&mut self, prompt: &str, max_tokens: u64) -> Result<Generated> {
            let rows = (prompt.split_whitespace().count() as f64 * 1.2).round() as u64;
            let label = match (self.mode.as_str(), self.gpu) {
                ("auto", true) => "gpu",
                ("auto", false) => "auto",
                (other, _) => other,
            };
            self.log
                .borrow_mut()
                .push(format!("gen {label} {max_tokens}"));
            let n = self.requests;
            self.requests += 1;
            let mut text = String::new();
            match self.mode.as_str() {
                "auto" if self.gpu => {
                    if self.cfg.gpu_route_line {
                        self.lines.push(GPU_ROUTE.to_owned());
                    }
                }
                "auto" => {
                    self.lines.push(RATES.to_owned());
                    self.lines
                        .push(PAIR_SAMPLE.replace("wait=12.5", &format!("wait={}.0", 10 + n)));
                    self.lines
                        .push(PAIR_SAMPLE.replace("wait=12.5", &format!("wait={}.0", 20 + n)));
                    if self.cfg.guard {
                        self.lines.push(GUARD.to_owned());
                    }
                }
                "on" => {
                    if self.cfg.pairs {
                        self.lines.push(PAIR.to_owned());
                    }
                    text = self.cfg.on_text.clone();
                }
                _ => text = self.cfg.emulate_text.clone(),
            }
            // The first two requests are warm-ups: make them unmistakable.
            let ttft_ms = if n < 2 && self.mode == "auto" {
                99_999.0
            } else {
                rows as f64 * 0.5 * if self.gpu { 1.0 } else { self.cfg.auto_factor }
            };
            let pieces: Vec<String> = text
                .split(' ')
                .filter(|p| !p.is_empty())
                .map(str::to_owned)
                .collect();
            Ok(Generated {
                ttft_ms,
                prompt_tokens: rows,
                cached_tokens: self.cfg.cached,
                text,
                tokens: pieces.len() as u64,
                pieces,
            })
        }
        fn npu_lines(&self) -> Vec<String> {
            self.lines.clone()
        }
        fn finish(self: Box<Self>) -> Result<()> {
            self.log.borrow_mut().push(format!("finish {}", self.mode));
            Ok(())
        }
    }

    fn drive(cfg: Cfg) -> (Result<Calibration>, Vec<String>, Vec<String>) {
        let log = Rc::new(RefCell::new(Vec::new()));
        let log_open = Rc::clone(&log);
        let mut messages = Vec::new();
        let result = run_with(
            |mode, sample_log| {
                log_open
                    .borrow_mut()
                    .push(format!("open {mode} sample_log={sample_log}"));
                let load_lines = if mode == "auto" {
                    cfg.load_lines.clone()
                } else {
                    Vec::new()
                };
                let session: Box<dyn Session> = Box::new(Fake {
                    mode: mode.to_owned(),
                    cfg: cfg.clone(),
                    log: Rc::clone(&log_open),
                    gpu: false,
                    requests: 0,
                    lines: load_lines,
                });
                Ok(session)
            },
            &mut |message| messages.push(message.to_owned()),
        );
        let log = log.borrow().clone();
        (result, log, messages)
    }

    #[test]
    fn run_alternates_routes_and_discards_warmups() {
        let (result, log, messages) = drive(Cfg::armed());
        let calibration = result.unwrap();
        let generates: Vec<&str> = log
            .iter()
            .filter(|entry| entry.starts_with("gen "))
            .map(|entry| entry.as_str())
            .collect();
        assert_eq!(
            generates,
            [
                // warm-ups
                "gen gpu 1",
                "gen auto 1",
                // rep 0 (GPU first): 16K then 32K
                "gen gpu 1",
                "gen auto 1",
                "gen gpu 1",
                "gen auto 1",
                // rep 1 (auto first)
                "gen auto 1",
                "gen gpu 1",
                "gen auto 1",
                "gen gpu 1",
                // exactness
                "gen on 16",
                "gen emulate 16",
            ]
        );
        assert_eq!(
            log.iter()
                .filter(|e| e.starts_with("open ") || e.starts_with("finish "))
                .map(String::as_str)
                .collect::<Vec<_>>(),
            [
                "open auto sample_log=true",
                "finish auto",
                "open on sample_log=false",
                "finish on",
                "open emulate sample_log=false",
                "finish emulate",
            ]
        );
        assert!(calibration.armed);
        assert_eq!(calibration.off_reason, None);
        assert_eq!(calibration.trunk_route.as_deref(), Some("overlay:b2"));
        assert_eq!(calibration.sizes.len(), 2);
        for size in &calibration.sizes {
            assert_eq!(size.off.samples_ms.len(), 2);
            assert_eq!(size.auto.samples_ms.len(), 2);
            assert!(size.off.samples_ms.iter().all(|&ms| ms < 99_999.0));
            assert!(size.auto.samples_ms.iter().all(|&ms| ms < 99_999.0));
            assert!((size.saved_pct - 20.0).abs() < 1e-9);
            assert_eq!(size.verdict, SizeVerdict::Gain);
        }
        assert!(messages[0].contains("npu_trunk=auto"), "{messages:?}");
        assert!(
            messages
                .iter()
                .any(|m| m.starts_with("[doctor] 16384 rows: gpu ")),
            "{messages:?}"
        );
        assert!(
            messages
                .iter()
                .any(|m| m.starts_with("[doctor] 32768 rows: auto ")),
            "{messages:?}"
        );
    }

    #[test]
    fn run_sizes_prompts_within_three_percent() {
        let (result, _, _) = drive(Cfg::armed());
        let calibration = result.unwrap();
        for (size, target) in calibration.sizes.iter().zip([16384.0, 32768.0]) {
            let error = (size.rows as f64 - target).abs() / target;
            assert!(error < 0.03, "rows {} vs {target}", size.rows);
        }
    }

    #[test]
    fn run_collects_auto_diagnostics_from_measured_requests_only() {
        let (result, _, _) = drive(Cfg::armed());
        let calibration = result.unwrap();
        let rates = calibration.rate_scales.unwrap();
        assert_eq!((rates.generation, rates.frozen), (3, false));
        // Four measured auto requests, two pair samples each; the warm-up's are dropped.
        let wait = calibration.npu_wait_per_pair.unwrap();
        assert_eq!(wait.pairs, 8);
        assert_eq!(calibration.guard_trips, 0);
        let (guarded, _, _) = drive(Cfg {
            guard: true,
            ..Cfg::armed()
        });
        assert_eq!(guarded.unwrap().guard_trips, 4);
    }

    #[test]
    fn run_short_circuits_when_not_armed() {
        for (line, reason) in [
            (OFF, "requires Flash-Next family B"),
            (WARN_OFF, "mmap heap: errno=11"),
        ] {
            let (result, log, _) = drive(Cfg {
                load_lines: vec![line.to_owned()],
                ..Cfg::armed()
            });
            let calibration = result.unwrap();
            assert!(!calibration.armed);
            assert_eq!(calibration.off_reason.as_deref(), Some(reason));
            assert!(calibration.sizes.is_empty());
            assert!(calibration.exactness.is_none());
            assert_eq!(log, ["open auto sample_log=true", "finish auto"]);
        }
    }

    #[test]
    fn run_without_a_status_line_is_not_armed() {
        let (result, _, _) = drive(Cfg {
            load_lines: Vec::new(),
            ..Cfg::armed()
        });
        let calibration = result.unwrap();
        assert!(!calibration.armed);
        assert!(calibration
            .off_reason
            .unwrap()
            .contains("no npu-trunk status line"));
    }

    #[test]
    fn run_fails_when_the_gpu_route_is_not_logged() {
        let (result, _, _) = drive(Cfg {
            gpu_route_line: false,
            ..Cfg::armed()
        });
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains("does not support the diagnostic GPU route"),
            "{error}"
        );
    }

    #[test]
    fn run_fails_on_a_prompt_cache_hit() {
        let (result, _, _) = drive(Cfg {
            cached: 128,
            ..Cfg::armed()
        });
        let error = result.unwrap_err().to_string();
        assert!(error.contains("prompt cache"), "{error}");
    }

    #[test]
    fn exactness_exact_mismatch_and_not_exercised() {
        let (exact, _, _) = drive(Cfg::armed());
        let exact = exact.unwrap().exactness.unwrap();
        assert_eq!(exact.verdict, ExactnessVerdict::Exact);
        assert_eq!(exact.tokens, 6);
        assert_eq!(exact.first_difference, None);
        assert!(
            (exact.prompt_rows as f64 - 9319.0).abs() / 9319.0 < 0.03,
            "{}",
            exact.prompt_rows
        );

        let (mismatch, _, _) = drive(Cfg {
            emulate_text: "the river carried a loud stone".to_owned(),
            ..Cfg::armed()
        });
        let mismatch = mismatch.unwrap().exactness.unwrap();
        assert_eq!(mismatch.verdict, ExactnessVerdict::Mismatch);
        assert_eq!(mismatch.first_difference, Some(4));

        let (idle, log, _) = drive(Cfg {
            pairs: false,
            ..Cfg::armed()
        });
        let idle = idle.unwrap().exactness.unwrap();
        assert_eq!(idle.verdict, ExactnessVerdict::NotExercised);
        // Nothing to compare, so the emulate daemon is never started.
        assert!(!log.iter().any(|entry| entry.starts_with("open emulate")));
    }

    #[test]
    fn compare_reports_the_first_differing_piece_and_length_mismatch() {
        let make = |text: &str| {
            let pieces: Vec<String> = text.split(' ').map(str::to_owned).collect();
            Generated {
                ttft_ms: 1.0,
                prompt_tokens: 9319,
                cached_tokens: 0,
                text: pieces.concat(),
                tokens: pieces.len() as u64,
                pieces,
            }
        };
        let base = make("a b c d");
        assert_eq!(
            compare(&base, &make("a b c d")).verdict,
            ExactnessVerdict::Exact
        );
        let diff = compare(&base, &make("a b x d"));
        assert_eq!(
            (diff.verdict, diff.first_difference),
            (ExactnessVerdict::Mismatch, Some(2))
        );
        let short = compare(&base, &make("a b c"));
        assert_eq!(
            (short.verdict, short.first_difference),
            (ExactnessVerdict::Mismatch, Some(3))
        );
    }

    #[test]
    fn plan_lines_state_the_cost() {
        let plan = plan_lines().join("\n");
        assert!(plan.contains("about 6–7 minutes on a Strix Halo"));
        assert!(plan.contains("uses the GPU and the NPU"));
        assert!(plan.contains("changes no setting and writes no file"));
    }
}
