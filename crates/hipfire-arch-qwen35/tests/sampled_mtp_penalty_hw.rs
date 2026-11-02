// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Real-model Qwen3.x AR vs native sampled-MTP hardware oracle, with the
//! request penalties (repeat / presence / frequency).
//!
//! Both tests are `#[ignore]`d: they need a real HIP GPU and a Qwen3.5/3.6
//! trunk named by `HIPFIRE_MTP_BYTE_IDENTITY_MODEL` plus its MTP sidecar
//! (`HIPFIRE_MTP_BYTE_IDENTITY_HEAD`, default `model.with_extension("mtp")`),
//! the same gating as `mtp_step_oracle`. With the model env unset the tests
//! print a skip line and pass. Nothing here is built or run by default.
//!
//! # Tests
//!
//! - `sampled_mtp_penalties_match_ar_distribution`: for each prompt case,
//!   `TRIALS` cold trials per arm (default 2000; the certification size, a
//!   smaller positive override is allowed for exploration but proves nothing
//!   at alpha 1e-3), each trial with its own seed, keeping the first `TOKENS`
//!   generated IDs. The per-position marginals (`TOKENS` of them) and the
//!   joint (t1, t2) of the MTP arm are compared with AR's by a two-sample
//!   chi-square homogeneity test; every p-value must exceed `ALPHA = 1e-3`.
//! - `greedy_presence_penalty_matches_ar_ids`: temperature 0, repeat penalty
//!   1.0, presence penalty 1.5, frequency penalty 0, window
//!   `HIPFIRE_SAMPLED_MTP_REPEAT_WINDOW` (default 128); greedy MTP must emit
//!   byte-identical IDs to greedy AR for `HIPFIRE_SAMPLED_MTP_GREEDY_TOKENS`
//!   (>= 256, default 256) tokens per prompt case.
//!
//! # Arms
//!
//! - AR (`ar`): `ModelSlot::forward` over each prompt token from a
//!   `reset_state` cold start, then `hipfire_runtime::sampler::sample` on
//!   `slot.scratch.logits` / `sample_buf` / `repeat_buf`, then `forward` of each
//!   generated token at `pos = prompt.len() + generated.len() - 1`. The
//!   penalty history is the GENERATED IDs only (the prompt is never
//!   penalized; the first draw sees an empty history). The sampler window is
//!   `(repeat_buf bytes / 4).min(requested_window.max(1))`.
//! - MTP (`mtp`): the public `build_qwen35_mtp_speculator` (a boxed
//!   `MtpSpeculator<Qwen35MtpDrafter>`) through `Speculator::configure_request`
//!   (once per trial), `prefill` and `step`. `emitted` is the generated IDs
//!   including the seed, `position` starts at `prompt.len()` and advances by
//!   `step.emit.len()`, the pending seed is `step.next_seed`, and `max_emit` is
//!   the remaining token budget exactly. The request carries the same four
//!   penalty fields with the RAW requested `repeat_window` (normalizing it is
//!   the production speculator's job; the AR window above is recorded as
//!   `effective_window`).
//!
//! Each arm runs in its OWN FRESH PROCESS (the parent test re-executes this
//! test binary on the ignored internal test `sampled_mtp_penalty_arm`), so two
//! 35B sessions never share a process (allocator residue). The parent sets
//! `HIPFIRE_MTP_SAMPLED=1` for the MTP child; every other `HIPFIRE_*` flag is
//! inherited and recorded in `config.json`. The child reads the frozen config
//! from the internal env `HIPFIRE_SAMPLED_MTP_CONFIG` (a path), never from the
//! penalty/trial envs, and the arm is selected by `HIPFIRE_SAMPLED_MTP_ARM`
//! (`ar` | `mtp`). Those two internal envs are not for manual use.
//!
//! # Conventions
//!
//! - Sampled mode honours the real EOS: a trial stops after the EOS token and
//!   its IDs are PADDED with EOS to `TOKENS` (EOS is absorbing). The raw
//!   unpadded IDs are saved too, and `padding` is recorded in the report.
//! - Greedy mode is a FIXED-LENGTH oracle: both arms temporarily set the public
//!   `slot.config.eos_token = u32::MAX` so neither the AR loop nor the MTP
//!   verifier treats any token as EOS, and decoding continues past a real EOS
//!   to exactly the requested length. The real EOS is saved first and restored
//!   once the arm's trials finish (also on error, before the error
//!   propagates).
//! - Seeds: SplitMix64 of `arm_salt ^ case_salt ^ trial`, independent salts
//!   per arm. The AR sampler's xorshift32 state is the folded 32-bit hash
//!   (never 0). All seeds are persisted in `arm.json`.
//! - Chi-square: cells whose expected count is under 5 (against the smaller
//!   sample) are pooled in deterministic `BTreeMap` order; if the pooled cell
//!   still has expected < 5 it is merged into the smallest kept cell. A test
//!   with no remaining degree of freedom reports chi2 = 0, p = 1. A non-finite
//!   or out-of-range p is itself a failure.
//! - Degeneracy guard: a statistical test (every position and the joint)
//!   FAILS if it has df = 0 (one kept cell, e.g. every trial emitted the same
//!   token) or an effective sample size below
//!   `max(MIN_EFFECTIVE_SAMPLE_ABS, MIN_EFFECTIVE_SAMPLE_FRAC * min(n_ar, n_mtp))`
//!   (10 and 0.05: 100 at the certification size of 2000). The effective
//!   sample size is `min(n_ar, n_mtp) * (1 - sum_k q_k^2)` over the kept cells
//!   with pooled proportions `q_k` (the Gini-Simpson diversity times the
//!   sample): the expected number of trials that would disagree with an
//!   independent draw. A test that cannot distinguish two laws proves nothing,
//!   so it must never pass.
//!
//! # Render (the Qwen3.8 preamble trap)
//!
//! The prompt is `<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n`,
//! optionally followed by the closed think block `<think>\n\n</think>\n\n`.
//! Qwen3.8's chat template opens a think block in the generation prompt
//! (`'<think>\n'` when thinking is on, `'<think>\n\n</think>\n\n'` when
//! `enable_thinking is false`). Rendered open (no think text), the model
//! writes that preamble itself, deterministically: the first real-model run
//! (2026-10-10, qwen3.8-27b, 2000 trials per arm) emitted
//! `[<think>, "\n", 760, 1156]` in EVERY trial of both arms and both cases,
//! so every test had df = 0, p = 1 and the oracle passed vacuously. The
//! render is therefore per model: `HIPFIRE_SAMPLED_MTP_RENDER` = `auto`
//! (default), `open` or `think_closed`. `auto` reads the trunk's embedded
//! chat template and picks `think_closed` only when the template's
//! generation prompt itself opens a think block (it contains both the
//! `'<think>\n'` open literal and the `'<think>\n\n</think>\n\n'` closed
//! literal); otherwise (Qwen3.5/3.6 as tested before, or no template) it
//! keeps the open render. The chosen render and the reason are recorded in
//! `config.json`, and the degeneracy guard catches a wrong choice.
//!
//! # Environment
//!
//! `HIPFIRE_SAMPLED_MTP_TRIALS` 2000, `_TEMP` 0.7, `_TOP_P` 0.8, `_TOP_K` 20
//! (`none` = absent), `_MIN_P` 0, `_TOKENS` 4 (>= 4), `_REPEAT_PENALTY` 1.0,
//! `_REPEAT_WINDOW` 128, `_PRESENCE_PENALTY` 0, `_FREQUENCY_PENALTY` 0, all
//! parsed strictly (a typo panics rather than running the neutral config).
//! `HIPFIRE_SAMPLED_MTP_PROMPT`: `both` (default: an `animals` list case and
//! an `integers` list case), `animals`, `integers`, or any other text, used as
//! the user message of a single `custom` case. `_REPEAT_WINDOW` may be 0 or
//! above the 2048-cell buffer: AR normalizes it, MTP receives it raw.
//!
//! Output: a NEW directory per run under `HIPFIRE_SAMPLED_MTP_OUT`, default
//! `$HOME/.hipfire/oracles/qwen35-sampled-mtp` (anything under `/tmp` is
//! refused): `<test>-<config hash>-p<pid>-<unix nanos>/` holding `config.json`
//! (written before the spawns), per arm `<arm>.arm.json` (raw IDs, seeds,
//! per-position and joint counts, effective window, EOS, GPU arch, proposed /
//! accepted totals) plus `<arm>.stdout.log` / `<arm>.stderr.log`, and
//! `report.json` (config, per-test statistics, both arms' counts). The report
//! is written before any statistical assertion.
//!
//! # Run (no build is done by writing this file)
//!
//! ```bash
//! HIPFIRE_MTP_BYTE_IDENTITY_MODEL=/path/to/qwen3.5-35b-a3b.mq4 \
//! HIPFIRE_MTP_BYTE_IDENTITY_HEAD=/path/to/qwen3.5-35b-a3b.mtp \
//! HIPFIRE_SAMPLED_MTP_OUT=$HOME/.hipfire/oracles/qwen35-sampled-mtp \
//! HIPFIRE_SAMPLED_MTP_TRIALS=2000 HIPFIRE_SAMPLED_MTP_TEMP=0.7 \
//! HIPFIRE_SAMPLED_MTP_TOP_P=0.8 HIPFIRE_SAMPLED_MTP_TOP_K=20 \
//! HIPFIRE_SAMPLED_MTP_MIN_P=0 HIPFIRE_SAMPLED_MTP_TOKENS=4 \
//! HIPFIRE_SAMPLED_MTP_REPEAT_PENALTY=1.1 HIPFIRE_SAMPLED_MTP_REPEAT_WINDOW=128 \
//! HIPFIRE_SAMPLED_MTP_PRESENCE_PENALTY=0.5 HIPFIRE_SAMPLED_MTP_FREQUENCY_PENALTY=0.2 \
//! HIPFIRE_SAMPLED_MTP_PROMPT=both \
//! cargo test --release --manifest-path /home/kaden/ClaudeCode/warpfront/wt-swarm-penalties/Cargo.toml \
//!   -p hipfire-arch-qwen35 --test sampled_mtp_penalty_hw \
//!   sampled_mtp_penalties_match_ar_distribution -- --ignored --exact --nocapture --test-threads=1
//!
//! HIPFIRE_MTP_BYTE_IDENTITY_MODEL=/path/to/qwen3.5-35b-a3b.mq4 \
//! HIPFIRE_MTP_BYTE_IDENTITY_HEAD=/path/to/qwen3.5-35b-a3b.mtp \
//! HIPFIRE_SAMPLED_MTP_OUT=$HOME/.hipfire/oracles/qwen35-sampled-mtp \
//! HIPFIRE_SAMPLED_MTP_REPEAT_WINDOW=128 HIPFIRE_SAMPLED_MTP_GREEDY_TOKENS=256 \
//! cargo test --release --manifest-path /home/kaden/ClaudeCode/warpfront/wt-swarm-penalties/Cargo.toml \
//!   -p hipfire-arch-qwen35 --test sampled_mtp_penalty_hw \
//!   greedy_presence_penalty_matches_ar_ids -- --ignored --exact --nocapture --test-threads=1
//! ```

#![allow(clippy::all)]

use hipfire_arch_qwen35::mtp_head;
use hipfire_arch_qwen35::mtp_speculator::build_qwen35_mtp_speculator;
use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};
use hipfire_runtime::sampler::{self, SamplerConfig};
use hipfire_runtime::spec::{PrefillOutcome, SpecRequestConfig, Speculator};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const MODEL_ENV: &str = "HIPFIRE_MTP_BYTE_IDENTITY_MODEL";
const HEAD_ENV: &str = "HIPFIRE_MTP_BYTE_IDENTITY_HEAD";
const OUT_ENV: &str = "HIPFIRE_SAMPLED_MTP_OUT";
const TRIALS_ENV: &str = "HIPFIRE_SAMPLED_MTP_TRIALS";
const TEMP_ENV: &str = "HIPFIRE_SAMPLED_MTP_TEMP";
const TOP_P_ENV: &str = "HIPFIRE_SAMPLED_MTP_TOP_P";
const TOP_K_ENV: &str = "HIPFIRE_SAMPLED_MTP_TOP_K";
const MIN_P_ENV: &str = "HIPFIRE_SAMPLED_MTP_MIN_P";
const TOKENS_ENV: &str = "HIPFIRE_SAMPLED_MTP_TOKENS";
const REPEAT_PENALTY_ENV: &str = "HIPFIRE_SAMPLED_MTP_REPEAT_PENALTY";
const REPEAT_WINDOW_ENV: &str = "HIPFIRE_SAMPLED_MTP_REPEAT_WINDOW";
const PRESENCE_ENV: &str = "HIPFIRE_SAMPLED_MTP_PRESENCE_PENALTY";
const FREQUENCY_ENV: &str = "HIPFIRE_SAMPLED_MTP_FREQUENCY_PENALTY";
const PROMPT_ENV: &str = "HIPFIRE_SAMPLED_MTP_PROMPT";
const GREEDY_TOKENS_ENV: &str = "HIPFIRE_SAMPLED_MTP_GREEDY_TOKENS";
/// `auto` (default) | `open` | `think_closed`; see "Render" above.
const RENDER_ENV: &str = "HIPFIRE_SAMPLED_MTP_RENDER";
/// The closed think block appended after `assistant\n` by `think_closed`.
const THINK_CLOSED: &str = "<think>\n\n</think>\n\n";
/// Jinja-source literals (escaped `\n`) that mark a template whose
/// generation prompt opens a think block.
const TEMPLATE_THINK_OPEN: &str = r"'<think>\n'";
const TEMPLATE_THINK_CLOSED: &str = r"'<think>\n\n</think>\n\n'";
/// Internal: child arm selector (`ar` | `mtp`).
const ARM_ENV: &str = "HIPFIRE_SAMPLED_MTP_ARM";
/// Internal: path of the frozen `config.json` the child reads.
const CONFIG_ENV: &str = "HIPFIRE_SAMPLED_MTP_CONFIG";
const ARM_TEST: &str = "sampled_mtp_penalty_arm";

// Open-ended list prompts: every early position has real diversity under the
// certification sampling (a TRIALS=40 pilot on qwen3.8-27b kept 2-4 cells per
// position, n_eff ~0.5-0.75 n). A fixed story or a "repeat this line" prompt
// is near-deterministic for the first tokens and fails the degeneracy guard.
const ANIMALS: &str = "Name ten random animals, one per line, and nothing else.";
const INTEGERS: &str =
    "Write twenty random integers between 1 and 1000, separated by spaces, with no other text.";

const MAX_SEQ: usize = 4096;
/// `ModelSlotConfig::repeat_window`: sizes `scratch.repeat_buf` (F32 cells).
const SLOT_REPEAT_WINDOW: usize = 2048;
const MTP_K: usize = 3;
const ALPHA: f64 = 1e-3;
/// Degeneracy guard: minimum effective sample size of a statistical test is
/// `max(MIN_EFFECTIVE_SAMPLE_ABS, MIN_EFFECTIVE_SAMPLE_FRAC * min(n_ar, n_mtp))`.
const MIN_EFFECTIVE_SAMPLE_ABS: f64 = 10.0;
const MIN_EFFECTIVE_SAMPLE_FRAC: f64 = 0.05;
const MIN_TRIAL_SAMPLES_DEFAULT: usize = 2000;
const MIN_TOKENS: usize = 4;
const MIN_GREEDY_TOKENS: usize = 256;
const GREEDY_PRESENCE: f32 = 1.5;
const AR_SALT: u64 = 0xA11C_E5ED_0000_0001;
const MTP_SALT: u64 = 0x5EC0_17A9_0000_0002;
const ARMS: [&str; 2] = ["ar", "mtp"];
const PADDING: &str = "sampled trials that hit EOS stop there and are padded with EOS \
    (absorbing) to TOKENS; greedy trials run with EOS disabled and are never padded";

#[derive(Clone, Serialize, Deserialize)]
struct Case {
    name: String,
    user: String,
}

/// The frozen run configuration: written to `config.json` before any child
/// is spawned and the only thing the children read.
#[derive(Clone, Serialize, Deserialize)]
struct Config {
    /// `sampled` | `greedy`.
    mode: String,
    model: String,
    head: String,
    out_dir: String,
    config_hash: String,
    trials: usize,
    tokens: usize,
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: f32,
    repeat_penalty: f32,
    repeat_window: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
    cases: Vec<Case>,
    /// `open` | `think_closed` (see "Render").
    render: String,
    /// Why `render` was chosen (env override or template detection).
    render_reason: String,
    ar_salt: u64,
    mtp_salt: u64,
    max_seq: usize,
    slot_repeat_window: usize,
    mtp_k: usize,
    padding: String,
    /// Every `HIPFIRE_*` env var of the parent, as inherited by the children.
    flags: BTreeMap<String, String>,
    /// Env the parent additionally sets for the MTP child.
    mtp_child_flags: BTreeMap<String, String>,
}

#[derive(Serialize, Deserialize)]
struct CaseOutput {
    name: String,
    prompt_tokens: usize,
    trials: usize,
    tokens: usize,
    /// Per-trial seed handed to the arm (AR: the xorshift32 state).
    seeds: Vec<u64>,
    /// Raw generated IDs per trial (unpadded).
    ids: Vec<Vec<u32>>,
    /// Trials that ended before `tokens` on a real EOS.
    early_eos: usize,
    /// `position_counts[i]`: token id -> count at generated position `i`
    /// (after EOS padding).
    position_counts: Vec<BTreeMap<u32, u64>>,
    /// (t1, t2) joint, key `"t1,t2"`.
    joint_counts: BTreeMap<String, u64>,
    proposed: u64,
    accepted: u64,
    elapsed_secs: f64,
}

#[derive(Serialize, Deserialize)]
struct ArmOutput {
    arm: String,
    mode: String,
    gpu_arch: String,
    model: String,
    head: String,
    vocab_size: usize,
    /// The slot's real EOS (restored after greedy's sentinel).
    eos_token: u32,
    /// EOS visible to the arm while decoding (`u32::MAX` in greedy mode).
    arm_eos_token: u32,
    repeat_buf_capacity: usize,
    effective_window: usize,
    max_seq: usize,
    cases: Vec<CaseOutput>,
}

#[derive(Serialize)]
struct StatLine {
    case: String,
    test: String,
    chi2: f64,
    df: usize,
    p: f64,
    n_ar: u64,
    n_mtp: u64,
    kept_cells: usize,
    pooled_cells: usize,
    merged_residual: bool,
    /// `min(n_ar, n_mtp) * (1 - sum q_k^2)` over the kept cells.
    effective_sample: f64,
    min_effective_sample: f64,
    /// df = 0 or `effective_sample < min_effective_sample` (always a failure).
    degenerate: bool,
    pass: bool,
}

#[derive(Serialize)]
struct Report<'a> {
    test: &'a str,
    dir: String,
    alpha: f64,
    config: &'a Config,
    stats: &'a [StatLine],
    failures: &'a [String],
    passed: bool,
    ar: &'a ArmOutput,
    mtp: &'a ArmOutput,
}

// ─── env / config ───────────────────────────────────────────────────────────

fn strict<T>(name: &str, default: T) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Ok(v) => v.trim().parse().map_err(|e| format!("{name}={v}: {e}")),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(e) => Err(format!("{name}: {e}")),
    }
}

fn strict_finite(name: &str, default: f32) -> Result<f32, String> {
    let v: f32 = strict(name, default)?;
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("{name}={v} is not finite"))
    }
}

fn strict_top_k() -> Result<Option<u32>, String> {
    match std::env::var(TOP_K_ENV) {
        Err(_) => Ok(Some(20)),
        Ok(v) if v.trim() == "none" => Ok(None),
        Ok(v) => v
            .trim()
            .parse()
            .map(Some)
            .map_err(|e| format!("{TOP_K_ENV}={v}: {e}")),
    }
}

fn cases_from_env() -> Result<Vec<Case>, String> {
    let animals = Case {
        name: "animals".into(),
        user: ANIMALS.into(),
    };
    let integers = Case {
        name: "integers".into(),
        user: INTEGERS.into(),
    };
    match std::env::var(PROMPT_ENV) {
        Err(_) => Ok(vec![animals, integers]),
        Ok(v) => match v.as_str() {
            "both" => Ok(vec![animals, integers]),
            "animals" => Ok(vec![animals]),
            "integers" => Ok(vec![integers]),
            "" => Err(format!("{PROMPT_ENV} is set but empty")),
            text => Ok(vec![Case {
                name: "custom".into(),
                user: text.to_string(),
            }]),
        },
    }
}

fn model_and_head() -> Result<Option<(PathBuf, PathBuf)>, String> {
    let Some(model) = std::env::var_os(MODEL_ENV).map(PathBuf::from) else {
        return Ok(None);
    };
    let model =
        fs::canonicalize(&model).map_err(|e| format!("{MODEL_ENV}={}: {e}", model.display()))?;
    let head = std::env::var_os(HEAD_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| model.with_extension("mtp"));
    let head = fs::canonicalize(&head).map_err(|e| format!("MTP head {}: {e}", head.display()))?;
    Ok(Some((model, head)))
}

/// `(render, reason)` from `HIPFIRE_SAMPLED_MTP_RENDER`; `auto` inspects the
/// trunk's embedded chat template (see "Render" in the module docs).
fn render_from_env(model: &Path) -> Result<(String, String), String> {
    let mode = std::env::var(RENDER_ENV).unwrap_or_else(|_| "auto".into());
    match mode.as_str() {
        "open" | "think_closed" => Ok((mode.clone(), format!("{RENDER_ENV}={mode}"))),
        "auto" => {
            let hfq = hipfire_runtime::hfq::HfqFile::open(model)
                .map_err(|e| format!("open {} for its chat template: {e}", model.display()))?;
            Ok(match hfq.chat_template() {
                None => ("open".into(), "auto: no embedded chat template".into()),
                Some(t) if t.contains(TEMPLATE_THINK_OPEN) && t.contains(TEMPLATE_THINK_CLOSED) => {
                    (
                        "think_closed".into(),
                        "auto: the chat template's generation prompt opens a think block".into(),
                    )
                }
                Some(_) => (
                    "open".into(),
                    "auto: the chat template does not open a think block".into(),
                ),
            })
        }
        other => Err(format!(
            "{RENDER_ENV}={other:?}: expected auto | open | think_closed"
        )),
    }
}

/// `Ok(None)` = model env absent (skip). `greedy` fixes the sampling to the
/// presence-penalty greedy contract.
fn build_config(greedy: bool) -> Result<Option<Config>, String> {
    let Some((model, head)) = model_and_head()? else {
        return Ok(None);
    };
    let repeat_window: usize = strict(REPEAT_WINDOW_ENV, 128)?;
    let (trials, tokens, temp, top_p, top_k, min_p, repeat_penalty, presence, frequency);
    if greedy {
        trials = 1;
        tokens = strict(GREEDY_TOKENS_ENV, MIN_GREEDY_TOKENS)?;
        if tokens < MIN_GREEDY_TOKENS {
            return Err(format!(
                "{GREEDY_TOKENS_ENV}={tokens} < {MIN_GREEDY_TOKENS}"
            ));
        }
        temp = 0.0;
        top_p = 1.0;
        top_k = None;
        min_p = 0.0;
        repeat_penalty = 1.0;
        presence = GREEDY_PRESENCE;
        frequency = 0.0;
        // Window 0 / > 2048 are allowed (see the sampled branch).
    } else {
        trials = strict(TRIALS_ENV, MIN_TRIAL_SAMPLES_DEFAULT)?;
        if trials == 0 {
            return Err(format!("{TRIALS_ENV}=0"));
        }
        tokens = strict(TOKENS_ENV, MIN_TOKENS)?;
        if tokens < MIN_TOKENS {
            return Err(format!("{TOKENS_ENV}={tokens} < {MIN_TOKENS}"));
        }
        temp = strict_finite(TEMP_ENV, 0.7)?;
        if temp <= 0.0 {
            return Err(format!("{TEMP_ENV}={temp} must be > 0 (sampled oracle)"));
        }
        top_p = strict_finite(TOP_P_ENV, 0.8)?;
        if !(top_p > 0.0 && top_p <= 1.0) {
            return Err(format!("{TOP_P_ENV}={top_p} outside (0, 1]"));
        }
        top_k = strict_top_k()?;
        min_p = strict_finite(MIN_P_ENV, 0.0)?;
        if !(0.0..1.0).contains(&min_p) {
            return Err(format!("{MIN_P_ENV}={min_p} outside [0, 1)"));
        }
        repeat_penalty = strict_finite(REPEAT_PENALTY_ENV, 1.0)?;
        if repeat_penalty <= 0.0 {
            return Err(format!("{REPEAT_PENALTY_ENV}={repeat_penalty} must be > 0"));
        }
        presence = strict_finite(PRESENCE_ENV, 0.0)?;
        frequency = strict_finite(FREQUENCY_ENV, 0.0)?;
        if presence < 0.0 || frequency < 0.0 {
            return Err(format!(
                "{PRESENCE_ENV}={presence} / {FREQUENCY_ENV}={frequency} must be >= 0"
            ));
        }
        // `_REPEAT_WINDOW=0` (and > 2048) with a penalty is deliberately allowed:
        // AR floors the request window at 1 and clamps it to the buffer, and the
        // MTP request carries the RAW value, so production normalization is tested.
    }
    let flags: BTreeMap<String, String> = std::env::vars()
        .filter(|(k, _)| k.starts_with("HIPFIRE_"))
        .collect();
    let mut mtp_child_flags = BTreeMap::new();
    mtp_child_flags.insert("HIPFIRE_MTP_SAMPLED".to_string(), "1".to_string());
    let (render, render_reason) = render_from_env(&model)?;
    Ok(Some(Config {
        mode: if greedy { "greedy" } else { "sampled" }.into(),
        model: model.display().to_string(),
        head: head.display().to_string(),
        out_dir: String::new(),
        config_hash: String::new(),
        trials,
        tokens,
        temp,
        top_p,
        top_k,
        min_p,
        repeat_penalty,
        repeat_window,
        presence_penalty: presence,
        frequency_penalty: frequency,
        cases: cases_from_env()?,
        render,
        render_reason,
        ar_salt: AR_SALT,
        mtp_salt: MTP_SALT,
        max_seq: MAX_SEQ,
        slot_repeat_window: SLOT_REPEAT_WINDOW,
        mtp_k: MTP_K,
        padding: PADDING.into(),
        flags,
        mtp_child_flags,
    }))
}

fn config_hash(cfg: &Config) -> String {
    let mut c = cfg.clone();
    c.out_dir.clear();
    c.config_hash.clear();
    let mut h = Sha256::new();
    h.update(serde_json::to_vec(&c).expect("serialize config"));
    let hex = format!("{:x}", h.finalize());
    hex[..12].to_string()
}

fn out_base() -> PathBuf {
    let base = std::env::var_os(OUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME")
                .unwrap_or_else(|| panic!("neither {OUT_ENV} nor HOME is set"));
            PathBuf::from(home).join(".hipfire/oracles/qwen35-sampled-mtp")
        });
    assert!(
        !base.starts_with("/tmp"),
        "{} is under /tmp; pick a persistent output directory",
        base.display()
    );
    base
}

/// Creates the NEW per-run directory (never reuses one).
fn new_run_dir(test: &str, hash: &str) -> PathBuf {
    let base = out_base();
    fs::create_dir_all(&base).unwrap_or_else(|e| panic!("create {}: {e}", base.display()));
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = base.join(format!("{test}-{hash}-p{}-{nanos}", std::process::id()));
    fs::create_dir(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    dir
}

// ─── seeds ──────────────────────────────────────────────────────────────────

/// SplitMix64 of `salt ^ trial`, never zero.
fn trial_seed(salt: u64, trial: usize) -> u64 {
    let mut z = (salt ^ trial as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)).max(1)
}

fn case_salt(salt: u64, case_index: usize) -> u64 {
    salt ^ ((case_index as u64 + 1) << 40)
}

/// The 32-bit AR sampler state derived from a 64-bit trial seed, never 0.
fn fold_seed32(seed: u64) -> u32 {
    ((seed ^ (seed >> 32)) as u32).max(1)
}

// ─── chi-square ─────────────────────────────────────────────────────────────

/// ln Γ(x), Lanczos (g = 7, n = 9).
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularized upper incomplete gamma Q(a, x) (series / continued fraction).
fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let ln_pre = -x + a * x.ln() - ln_gamma(a);
    if x < a + 1.0 {
        let (mut sum, mut term, mut n) = (1.0 / a, 1.0 / a, a);
        for _ in 0..1000 {
            n += 1.0;
            term *= x / n;
            sum += term;
            if term.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        (1.0 - sum * ln_pre.exp()).clamp(0.0, 1.0)
    } else {
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-15 {
                break;
            }
        }
        (ln_pre.exp() * h).clamp(0.0, 1.0)
    }
}

struct Chi {
    stat: f64,
    df: usize,
    p: f64,
    n_a: u64,
    n_b: u64,
    kept: usize,
    pooled_cells: usize,
    merged_residual: bool,
    /// `nmin * (1 - sum q_k^2)` over the kept cells (pooled proportions).
    effective_sample: f64,
}

/// Two-sample chi-square homogeneity test of category counts `a` vs `b`.
/// Cells with expected count (against the smaller sample) under 5 are pooled
/// in key order; a pooled cell still under 5 is merged into the smallest kept
/// cell. No remaining degree of freedom reports `chi2 = 0, p = 1`.
fn homogeneity<K: Ord + Clone>(a: &BTreeMap<K, u64>, b: &BTreeMap<K, u64>) -> Chi {
    let n_a: u64 = a.values().sum();
    let n_b: u64 = b.values().sum();
    let mut cells: BTreeMap<K, (f64, f64)> = BTreeMap::new();
    for (k, &c) in a {
        cells.entry(k.clone()).or_default().0 += c as f64;
    }
    for (k, &c) in b {
        cells.entry(k.clone()).or_default().1 += c as f64;
    }
    if n_a == 0 || n_b == 0 {
        return Chi {
            stat: f64::NAN,
            df: 0,
            p: f64::NAN,
            n_a,
            n_b,
            kept: 0,
            pooled_cells: 0,
            merged_residual: false,
            effective_sample: 0.0,
        };
    }
    let (na, nb) = (n_a as f64, n_b as f64);
    let n = na + nb;
    let nmin = na.min(nb);
    let mut kept: Vec<(f64, f64)> = Vec::new();
    let mut pooled = (0.0, 0.0);
    let mut pooled_cells = 0usize;
    for &(x, y) in cells.values() {
        if (x + y) * nmin / n < 5.0 {
            pooled.0 += x;
            pooled.1 += y;
            pooled_cells += 1;
        } else {
            kept.push((x, y));
        }
    }
    let mut merged_residual = false;
    if pooled_cells > 0 {
        if kept.is_empty() || (pooled.0 + pooled.1) * nmin / n >= 5.0 {
            kept.push(pooled);
        } else {
            let mut smallest = 0;
            for (i, c) in kept.iter().enumerate() {
                if c.0 + c.1 < kept[smallest].0 + kept[smallest].1 {
                    smallest = i;
                }
            }
            kept[smallest].0 += pooled.0;
            kept[smallest].1 += pooled.1;
            merged_residual = true;
        }
    }
    let df = kept.len().saturating_sub(1);
    let simpson: f64 = kept.iter().map(|&(x, y)| ((x + y) / n).powi(2)).sum();
    let effective_sample = nmin * (1.0 - simpson);
    let (stat, p) = if df == 0 {
        (0.0, 1.0)
    } else {
        let mut stat = 0.0;
        for &(x, y) in &kept {
            let total = x + y;
            let (ea, eb) = (total * na / n, total * nb / n);
            stat += (x - ea).powi(2) / ea + (y - eb).powi(2) / eb;
        }
        (stat, gamma_q(df as f64 / 2.0, stat / 2.0))
    };
    Chi {
        stat,
        df,
        p,
        n_a,
        n_b,
        kept: kept.len(),
        pooled_cells,
        merged_residual,
        effective_sample,
    }
}

fn stat_line(case: &str, test: String, chi: Chi) -> StatLine {
    let valid = chi.p.is_finite() && (0.0..=1.0).contains(&chi.p) && chi.stat.is_finite();
    let min_effective_sample =
        MIN_EFFECTIVE_SAMPLE_ABS.max(MIN_EFFECTIVE_SAMPLE_FRAC * chi.n_a.min(chi.n_b) as f64);
    let degenerate = chi.df == 0 || !(chi.effective_sample >= min_effective_sample);
    StatLine {
        case: case.to_string(),
        test,
        chi2: chi.stat,
        df: chi.df,
        p: chi.p,
        n_ar: chi.n_a,
        n_mtp: chi.n_b,
        kept_cells: chi.kept,
        pooled_cells: chi.pooled_cells,
        merged_residual: chi.merged_residual,
        effective_sample: chi.effective_sample,
        min_effective_sample,
        degenerate,
        pass: valid && !degenerate && chi.p >= ALPHA,
    }
}

// ─── child arm ──────────────────────────────────────────────────────────────

/// `render` is `open` or `think_closed` (see "Render" in the module docs).
fn chat_prompt(tok: &Tokenizer, user: &str, render: &str) -> Vec<u32> {
    let mut t = Vec::new();
    t.extend(tok.encode("<|im_start|>"));
    t.extend(tok.encode("user"));
    t.extend(tok.encode("\n"));
    t.extend(tok.encode(user));
    t.extend(tok.encode("<|im_end|>"));
    t.extend(tok.encode("\n"));
    t.extend(tok.encode("<|im_start|>"));
    t.extend(tok.encode("assistant"));
    t.extend(tok.encode("\n"));
    if render == "think_closed" {
        t.extend(tok.encode(THINK_CLOSED));
    }
    t
}

fn count_ids(
    padded: &[Vec<u32>],
    tokens: usize,
) -> (Vec<BTreeMap<u32, u64>>, BTreeMap<String, u64>) {
    let mut positions: Vec<BTreeMap<u32, u64>> = vec![BTreeMap::new(); tokens];
    let mut joint: BTreeMap<String, u64> = BTreeMap::new();
    for ids in padded {
        for (i, &t) in ids.iter().enumerate() {
            *positions[i].entry(t).or_default() += 1;
        }
        *joint.entry(format!("{},{}", ids[0], ids[1])).or_default() += 1;
    }
    (positions, joint)
}

#[allow(clippy::too_many_arguments)]
fn finish_case(
    name: &str,
    prompt_tokens: usize,
    tokens: usize,
    pad: u32,
    seeds: Vec<u64>,
    raw: Vec<Vec<u32>>,
    proposed: u64,
    accepted: u64,
    elapsed_secs: f64,
) -> CaseOutput {
    let early_eos = raw.iter().filter(|ids| ids.len() < tokens).count();
    let padded: Vec<Vec<u32>> = raw
        .iter()
        .map(|ids| {
            let mut p = ids.clone();
            p.resize(tokens, pad);
            p
        })
        .collect();
    let (position_counts, joint_counts) = count_ids(&padded, tokens);
    CaseOutput {
        name: name.to_string(),
        prompt_tokens,
        trials: raw.len(),
        tokens,
        seeds,
        ids: raw,
        early_eos,
        position_counts,
        joint_counts,
        proposed,
        accepted,
        elapsed_secs,
    }
}

fn sampler_config(cfg: &Config, window: usize) -> SamplerConfig {
    SamplerConfig {
        temperature: cfg.temp,
        top_p: cfg.top_p,
        repeat_penalty: cfg.repeat_penalty,
        repeat_window: window,
        presence_penalty: cfg.presence_penalty,
        frequency_penalty: cfg.frequency_penalty,
        blocked_tokens: Vec::new(),
        top_k: cfg.top_k,
        min_p: (cfg.min_p > 0.0).then_some(cfg.min_p),
    }
}

fn spec_request(cfg: &Config, window: usize, seed: u64) -> SpecRequestConfig {
    SpecRequestConfig {
        temp: cfg.temp,
        top_p: cfg.top_p,
        top_k: cfg.top_k,
        min_p: cfg.min_p,
        cactus_delta: 0.0,
        rng_seed: seed,
        allow_ngram_modifier: false,
        repeat_penalty: cfg.repeat_penalty,
        repeat_window: window,
        presence_penalty: cfg.presence_penalty,
        frequency_penalty: cfg.frequency_penalty,
        ..SpecRequestConfig::default()
    }
}

fn ar_trial(
    gpu: &mut Gpu,
    slot: &mut ModelSlot,
    prompt: &[u32],
    sc: &SamplerConfig,
    seed: u32,
    tokens: usize,
    stop_eos: Option<u32>,
) -> Result<Vec<u32>, String> {
    let vocab = slot.config.vocab_size;
    slot.reset_state(gpu)
        .map_err(|e| format!("reset_state: {e:?}"))?;
    for (i, &t) in prompt.iter().enumerate() {
        slot.forward(gpu, t, i)
            .map_err(|e| format!("prompt forward {i}: {e:?}"))?;
    }
    let mut generated: Vec<u32> = Vec::with_capacity(tokens);
    let mut rng = seed;
    loop {
        let tok = sampler::sample(
            gpu,
            &slot.scratch.logits,
            &slot.scratch.sample_buf,
            &slot.scratch.repeat_buf,
            vocab,
            &generated,
            sc,
            &mut rng,
        );
        generated.push(tok);
        if generated.len() >= tokens || stop_eos == Some(tok) {
            return Ok(generated);
        }
        let pos = prompt.len() + generated.len() - 1;
        slot.forward(gpu, tok, pos)
            .map_err(|e| format!("decode forward {pos}: {e:?}"))?;
    }
}

fn ar_cases(
    gpu: &mut Gpu,
    slot: &mut ModelSlot,
    cfg: &Config,
    prompts: &[Vec<u32>],
    window: usize,
    stop_eos: Option<u32>,
    pad: u32,
) -> Result<Vec<CaseOutput>, String> {
    let sc = sampler_config(cfg, window);
    let mut out = Vec::new();
    for (ci, (case, prompt)) in cfg.cases.iter().zip(prompts).enumerate() {
        let started = std::time::Instant::now();
        let salt = case_salt(cfg.ar_salt, ci);
        let mut seeds = Vec::with_capacity(cfg.trials);
        let mut raw = Vec::with_capacity(cfg.trials);
        for trial in 0..cfg.trials {
            let seed = fold_seed32(trial_seed(salt, trial));
            seeds.push(seed as u64);
            raw.push(ar_trial(
                gpu, slot, prompt, &sc, seed, cfg.tokens, stop_eos,
            )?);
            if (trial + 1) % 100 == 0 {
                println!(
                    "ARM ar case {} trial {}/{}",
                    case.name,
                    trial + 1,
                    cfg.trials
                );
            }
        }
        out.push(finish_case(
            &case.name,
            prompt.len(),
            cfg.tokens,
            pad,
            seeds,
            raw,
            0,
            0,
            started.elapsed().as_secs_f64(),
        ));
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn mtp_trial(
    gpu: &mut Gpu,
    slot: &mut ModelSlot,
    spec: &mut dyn Speculator,
    prompt: &[u32],
    req: SpecRequestConfig,
    tokens: usize,
    stop_eos: Option<u32>,
    totals: &mut (u64, u64),
) -> Result<Vec<u32>, String> {
    slot.reset_state(gpu)
        .map_err(|e| format!("reset_state: {e:?}"))?;
    spec.configure_request(req);
    let first = match spec.prefill(gpu, slot, prompt, prompt, 0, false, None, &|| false)? {
        PrefillOutcome::Ready { first_token } => first_token,
        PrefillOutcome::Aborted => return Err("prefill aborted".into()),
    };
    let mut generated: Vec<u32> = vec![first];
    let mut position = prompt.len();
    let mut seed = first;
    while generated.len() < tokens && stop_eos != generated.last().copied() {
        let remaining = tokens - generated.len();
        let step = spec.step(
            gpu, slot, position, seed, &generated, None, req.temp, remaining,
        )?;
        if step.emit.is_empty() {
            return Err(format!("empty step at position {position}"));
        }
        if step.emit.len() > remaining {
            return Err(format!(
                "step emitted {} > max_emit {remaining} at position {position}",
                step.emit.len()
            ));
        }
        totals.0 += step.proposed as u64;
        totals.1 += step.accepted as u64;
        position += step.emit.len();
        seed = step.next_seed;
        for &t in step.emit.iter() {
            generated.push(t);
            if stop_eos == Some(t) {
                break;
            }
        }
    }
    Ok(generated)
}

fn mtp_cases(
    gpu: &mut Gpu,
    slot: &mut ModelSlot,
    spec: &mut dyn Speculator,
    cfg: &Config,
    prompts: &[Vec<u32>],
    window: usize,
    stop_eos: Option<u32>,
    pad: u32,
) -> Result<Vec<CaseOutput>, String> {
    if cfg.mode == "sampled" && !spec.supports_temp_verify() {
        return Err(
            "speculator does not advertise sampled verification (HIPFIRE_MTP_SAMPLED)".into(),
        );
    }
    let mut out = Vec::new();
    for (ci, (case, prompt)) in cfg.cases.iter().zip(prompts).enumerate() {
        let started = std::time::Instant::now();
        let salt = case_salt(cfg.mtp_salt, ci);
        let mut seeds = Vec::with_capacity(cfg.trials);
        let mut raw = Vec::with_capacity(cfg.trials);
        let mut totals = (0u64, 0u64);
        for trial in 0..cfg.trials {
            let seed = trial_seed(salt, trial);
            seeds.push(seed);
            let req = spec_request(cfg, window, seed);
            raw.push(mtp_trial(
                gpu,
                slot,
                spec,
                prompt,
                req,
                cfg.tokens,
                stop_eos,
                &mut totals,
            )?);
            if (trial + 1) % 100 == 0 {
                println!(
                    "ARM mtp case {} trial {}/{}",
                    case.name,
                    trial + 1,
                    cfg.trials
                );
            }
        }
        out.push(finish_case(
            &case.name,
            prompt.len(),
            cfg.tokens,
            pad,
            seeds,
            raw,
            totals.0,
            totals.1,
            started.elapsed().as_secs_f64(),
        ));
    }
    Ok(out)
}

fn free_session(gpu: &mut Gpu, slot: ModelSlot, spec: Option<Box<dyn Speculator>>) {
    gpu.invalidate_weight_caches();
    gpu.invalidate_graph_state();
    if let Some(spec) = spec {
        spec.free(gpu);
    }
    let _ = slot.kv_cache.free_gpu(gpu);
    slot.dn_state.free_gpu(gpu);
    let _ = slot.scratch.free_gpu(gpu);
    slot.weights.free_gpu(gpu);
    gpu.drain_pool();
}

fn run_arm(cfg: &Config, arm: &str) -> Result<ArmOutput, String> {
    let mut gpu = Gpu::init().map_err(|e| format!("Gpu::init: {e:?}"))?;
    let gpu_arch = format!("{}", gpu.arch);
    eprintln!("[oracle] arm={arm} mode={} gpu={gpu_arch}", cfg.mode);
    let slot_cfg = ModelSlotConfig {
        max_seq: cfg.max_seq,
        repeat_window: cfg.slot_repeat_window,
        ..ModelSlotConfig::default()
    };
    let model = Path::new(&cfg.model);
    let head_path = Path::new(&cfg.head);
    let mut slot = ModelSlot::load(&mut gpu, model, "sampled-mtp-penalty", slot_cfg)
        .map_err(|e| format!("load trunk: {e:?}"))?;
    let tokenizer = slot
        .load_tokenizer()
        .map_err(|e| format!("tokenizer: {e:?}"))?;
    let prompts: Vec<Vec<u32>> = cfg
        .cases
        .iter()
        .map(|c| chat_prompt(&tokenizer, &c.user, &cfg.render))
        .collect();
    let vocab_size = slot.config.vocab_size;
    let repeat_buf_capacity = slot.scratch.repeat_buf.buf.size() / 4;
    let effective_window = repeat_buf_capacity.min(cfg.repeat_window.max(1));
    let real_eos = slot.config.eos_token;
    let greedy = cfg.mode == "greedy";
    let mut spec: Option<Box<dyn Speculator>> = None;
    if arm == "mtp" {
        let head = mtp_head::load_mtp_head(head_path, &mut gpu, cfg.max_seq)
            .map_err(|e| format!("load mtp head: {e:?}"))?;
        if head.config.vocab_size != vocab_size {
            return Err(format!(
                "trunk/head vocab mismatch: {vocab_size} vs {}",
                head.config.vocab_size
            ));
        }
        spec = Some(build_qwen35_mtp_speculator(head, cfg.mtp_k, cfg.max_seq));
    }
    // Greedy fixed-length oracle: disable EOS in both arms (restored below).
    let (arm_eos, stop_eos) = if greedy {
        (u32::MAX, None)
    } else {
        (real_eos, Some(real_eos))
    };
    slot.config.eos_token = arm_eos;
    let cases = match (arm, spec.as_mut()) {
        ("ar", None) => ar_cases(
            &mut gpu,
            &mut slot,
            cfg,
            &prompts,
            effective_window,
            stop_eos,
            real_eos,
        ),
        ("mtp", Some(s)) => mtp_cases(
            &mut gpu,
            &mut slot,
            s.as_mut(),
            cfg,
            &prompts,
            cfg.repeat_window,
            stop_eos,
            real_eos,
        ),
        _ => Err(format!("unknown arm {arm}")),
    };
    slot.config.eos_token = real_eos;
    free_session(&mut gpu, slot, spec);
    Ok(ArmOutput {
        arm: arm.to_string(),
        mode: cfg.mode.clone(),
        gpu_arch,
        model: cfg.model.clone(),
        head: cfg.head.clone(),
        vocab_size,
        eos_token: real_eos,
        arm_eos_token: arm_eos,
        repeat_buf_capacity,
        effective_window,
        max_seq: cfg.max_seq,
        cases: cases?,
    })
}

/// One arm in a fresh process; spawned by the main tests below.
#[test]
#[ignore = "child process of the sampled MTP penalty oracle tests"]
fn sampled_mtp_penalty_arm() {
    let Ok(arm) = std::env::var(ARM_ENV) else {
        return;
    };
    let cfg_path = std::env::var_os(CONFIG_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{CONFIG_ENV} must name the frozen config.json"));
    let cfg: Config = serde_json::from_slice(
        &fs::read(&cfg_path).unwrap_or_else(|e| panic!("read {}: {e}", cfg_path.display())),
    )
    .expect("parse config.json");
    let out = run_arm(&cfg, &arm).unwrap_or_else(|e| panic!("{arm} arm: {e}"));
    let path = Path::new(&cfg.out_dir).join(format!("{arm}.arm.json"));
    fs::write(&path, serde_json::to_vec(&out).expect("serialize arm")).expect("write arm.json");
    println!("ARM {arm} wrote {}", path.display());
}

// ─── parent ─────────────────────────────────────────────────────────────────

fn spawn_arm(cfg: &Config, dir: &Path, arm: &str) -> ArmOutput {
    let exe = std::env::current_exe().expect("test binary path");
    let mut cmd = Command::new(exe);
    cmd.args([
        "--exact",
        ARM_TEST,
        "--ignored",
        "--nocapture",
        "--test-threads=1",
    ])
    .env(ARM_ENV, arm)
    .env(CONFIG_ENV, dir.join("config.json"))
    .current_dir(dir)
    .stdin(Stdio::null());
    if arm == "mtp" {
        for (k, v) in &cfg.mtp_child_flags {
            cmd.env(k, v);
        }
    }
    let output = cmd.output().unwrap_or_else(|e| panic!("spawn {arm}: {e}"));
    fs::write(dir.join(format!("{arm}.stdout.log")), &output.stdout).unwrap();
    fs::write(dir.join(format!("{arm}.stderr.log")), &output.stderr).unwrap();
    assert!(
        output.status.success(),
        "{arm} arm failed ({}); logs in {}",
        output.status,
        dir.display()
    );
    let path = dir.join(format!("{arm}.arm.json"));
    serde_json::from_slice(
        &fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display())),
    )
    .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// Writes `config.json`, runs both arms in fresh processes, and validates the
/// arms are comparable.
fn run_oracle(test: &str, mut cfg: Config) -> (PathBuf, Config, ArmOutput, ArmOutput) {
    cfg.config_hash = config_hash(&cfg);
    let dir = new_run_dir(test, &cfg.config_hash);
    cfg.out_dir = dir.display().to_string();
    fs::write(
        dir.join("config.json"),
        serde_json::to_vec_pretty(&cfg).expect("serialize config"),
    )
    .expect("write config.json");
    eprintln!("[oracle] {test}: output {}", dir.display());
    let ar = spawn_arm(&cfg, &dir, ARMS[0]);
    let mtp = spawn_arm(&cfg, &dir, ARMS[1]);
    assert_eq!(
        ar.effective_window, mtp.effective_window,
        "arms disagree on the penalty window"
    );
    assert_eq!(ar.eos_token, mtp.eos_token, "arms disagree on EOS");
    assert_eq!(ar.cases.len(), cfg.cases.len());
    assert_eq!(mtp.cases.len(), cfg.cases.len());
    (dir, cfg, ar, mtp)
}

fn write_report(
    test: &str,
    dir: &Path,
    cfg: &Config,
    stats: &[StatLine],
    failures: &[String],
    ar: &ArmOutput,
    mtp: &ArmOutput,
) {
    let report = Report {
        test,
        dir: dir.display().to_string(),
        alpha: ALPHA,
        config: cfg,
        stats,
        failures,
        passed: failures.is_empty(),
        ar,
        mtp,
    };
    fs::write(
        dir.join("report.json"),
        serde_json::to_vec_pretty(&report).expect("serialize report"),
    )
    .expect("write report.json");
}

#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_MTP_BYTE_IDENTITY_MODEL (Qwen3.x trunk) + MTP sidecar"]
fn sampled_mtp_penalties_match_ar_distribution() {
    let test = "sampled_mtp_penalties_match_ar_distribution";
    let cfg = match build_config(false) {
        Ok(Some(cfg)) => cfg,
        Ok(None) => {
            eprintln!("skipping: {MODEL_ENV} not set");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    let (dir, cfg, ar, mtp) = run_oracle(test, cfg);
    let mut stats: Vec<StatLine> = Vec::new();
    for (a, m) in ar.cases.iter().zip(&mtp.cases) {
        assert_eq!(a.name, m.name, "case order differs between arms");
        for pos in 0..cfg.tokens {
            let chi = homogeneity(&a.position_counts[pos], &m.position_counts[pos]);
            stats.push(stat_line(&a.name, format!("token {}", pos + 1), chi));
        }
        let chi = homogeneity(&a.joint_counts, &m.joint_counts);
        stats.push(stat_line(&a.name, "(t1,t2) joint".into(), chi));
    }
    let mut failures: Vec<String> = Vec::new();
    for s in &stats {
        let line = format!(
            "{} {}: chi2={:.2} df={} p={:.4} (n={}+{}, kept={}, pooled={}, merged={}, n_eff={:.1} min {:.1}{})\n",
            s.case,
            s.test,
            s.chi2,
            s.df,
            s.p,
            s.n_ar,
            s.n_mtp,
            s.kept_cells,
            s.pooled_cells,
            s.merged_residual,
            s.effective_sample,
            s.min_effective_sample,
            if s.degenerate { ", DEGENERATE" } else { "" }
        );
        eprint!("{line}");
        if !s.pass {
            failures.push(line);
        }
    }
    for (a, m) in ar.cases.iter().zip(&mtp.cases) {
        eprintln!(
            "{}: ar early_eos={} mtp early_eos={} mtp proposed={} accepted={}",
            a.name, a.early_eos, m.early_eos, m.proposed, m.accepted
        );
    }
    write_report(test, &dir, &cfg, &stats, &failures, &ar, &mtp);
    assert!(
        failures.is_empty(),
        "sampled MTP differs from AR under penalties (alpha {ALPHA}; report in {}):\n{}",
        dir.display(),
        failures.concat()
    );
}

#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_MTP_BYTE_IDENTITY_MODEL (Qwen3.x trunk) + MTP sidecar"]
fn greedy_presence_penalty_matches_ar_ids() {
    let test = "greedy_presence_penalty_matches_ar_ids";
    let cfg = match build_config(true) {
        Ok(Some(cfg)) => cfg,
        Ok(None) => {
            eprintln!("skipping: {MODEL_ENV} not set");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    let (dir, cfg, ar, mtp) = run_oracle(test, cfg);
    let mut stats: Vec<StatLine> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    for (a, m) in ar.cases.iter().zip(&mtp.cases) {
        assert_eq!(a.name, m.name, "case order differs between arms");
        let (ai, mi) = (&a.ids[0], &m.ids[0]);
        let mut line = None;
        if ai.len() != cfg.tokens || mi.len() != cfg.tokens {
            line = Some(format!(
                "{}: length ar={} mtp={} expected {}\n",
                a.name,
                ai.len(),
                mi.len(),
                cfg.tokens
            ));
        } else if let Some(i) = ai.iter().zip(mi).position(|(x, y)| x != y) {
            line = Some(format!(
                "{}: first mismatch at generated index {i}: ar={} mtp={}\n",
                a.name, ai[i], mi[i]
            ));
        }
        eprintln!(
            "{}: {} tokens, mtp proposed={} accepted={}, {}",
            a.name,
            ai.len(),
            m.proposed,
            m.accepted,
            if line.is_none() {
                "IDENTICAL"
            } else {
                "MISMATCH"
            }
        );
        stats.push(StatLine {
            case: a.name.clone(),
            test: "greedy ids identical".into(),
            chi2: 0.0,
            df: 0,
            p: if line.is_none() { 1.0 } else { 0.0 },
            n_ar: ai.len() as u64,
            n_mtp: mi.len() as u64,
            kept_cells: 0,
            pooled_cells: 0,
            merged_residual: false,
            effective_sample: 0.0,
            min_effective_sample: 0.0,
            degenerate: false,
            pass: line.is_none(),
        });
        if let Some(line) = line {
            eprint!("{line}");
            failures.push(line);
        }
    }
    write_report(test, &dir, &cfg, &stats, &failures, &ar, &mtp);
    assert!(
        failures.is_empty(),
        "greedy MTP IDs differ from AR under presence penalty (report in {}):\n{}",
        dir.display(),
        failures.concat()
    );
}

#[test]
fn degeneracy_guard_fails_single_cell_and_low_diversity() {
    let one: BTreeMap<u32, u64> = [(7, 2000)].into_iter().collect();
    let s = stat_line("c", "t".into(), homogeneity(&one, &one));
    assert!(s.degenerate && !s.pass, "df=0 must fail");
    // 99% one token: n_eff ~ 2000 * (1 - 0.9801 - 0.0001) = 39.6 < 100.
    let skew: BTreeMap<u32, u64> = [(1, 1980), (2, 20)].into_iter().collect();
    let s = stat_line("c", "t".into(), homogeneity(&skew, &skew));
    assert_eq!(s.df, 1);
    assert!(s.degenerate && !s.pass, "low effective sample must fail");
    let even: BTreeMap<u32, u64> = [(1, 1000), (2, 600), (3, 400)].into_iter().collect();
    let s = stat_line("c", "t".into(), homogeneity(&even, &even));
    assert!(
        !s.degenerate && s.pass,
        "diverse identical counts must pass"
    );
}

#[test]
fn think_template_literals_match_qwen38_generation_prompt() {
    let t = r"{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- else %}{{- '<think>\n' }}{%- endif %}{%- endif %}";
    assert!(t.contains(TEMPLATE_THINK_OPEN) && t.contains(TEMPLATE_THINK_CLOSED));
    let qwen3 = r"{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}";
    assert!(!qwen3.contains(TEMPLATE_THINK_OPEN));
}
