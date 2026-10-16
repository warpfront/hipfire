// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sampled Flash-Next MTP against AR sampling, in distribution.
//!
//! `#[ignore]`d: it needs a real HIP GPU and the canonical
//! `qwen3.8-flash-next-gptq3.mq4` artifact named by
//! `HIPFIRE_MTP_IDENTITY_MODEL`.
//!
//! Loading uses the shared production expert-residency policy:
//! `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=auto` or `N` places discrete-GPU
//! expert spill in pinned host RAM; auto reserves native MTP and QSA gather
//! workspace. Product code and sampled coverage, identity and law assertions
//! are unchanged.
//!
//! The serve battery's prose prompt (thinking off, rendered as `hipfire
//! serve` renders one user message) is decoded from a cold state
//! `HIPFIRE_SAMPLED_MTP_TRIALS` times (default 2000) per arm, each trial with
//! its own seed, keeping the first `TOKENS` generated IDs:
//!
//! - `ar`: `forward_chunk_final` / `forward_token_or_argmax` with the
//!   production host sampler (`sampler::sample_cpu`);
//! - `batched`: sampled native MTP, `HIPFIRE_MTP_INCREMENTAL=0`;
//! - `interleaved`: sampled native MTP, `HIPFIRE_MTP_INCREMENTAL=1`.
//! - `adaptive`: sampled native MTP on the default route (floor on, no route
//!   knob): calibration token, probe windows, and retirement when the floor
//!   decides it. Its per-seed bytes depend on wall-clock prices, so it is
//!   compared with AR in distribution only and never replayed.
//!
//! Trial seeds are SplitMix64-scrambled (`trial_seed`), distinct per arm:
//! the host AR sampler's xorshift32 maps consecutive seeds to nearly equal
//! first draws, which production avoids by hashing its request seeds.
//!
//! Each MTP arm's per-position marginals and its (t1, t2) joint are compared
//! with AR's by a two-sample chi-square homogeneity test (cells with fewer
//! than 5 expected draws pooled); every p-value must exceed `ALPHA`. Each MTP
//! arm on a forced floor-off route (`batched`, `interleaved`) also replays its
//! first `REPLAYS` seeds and must emit the same IDs; replay is asserted only
//! for those arms.
//! Temperature / top_p / top_k / min_p come from `HIPFIRE_SAMPLED_MTP_TEMP` /
//! `HIPFIRE_SAMPLED_MTP_TOP_P` / `HIPFIRE_SAMPLED_MTP_TOP_K` (unset = absent)
//! / `HIPFIRE_SAMPLED_MTP_MIN_P` (default 1.0 / 0.95 / absent / 0); both arms
//! receive them as a request would. Per-arm IDs and the report go to
//! `HIPFIRE_SAMPLED_MTP_OUT`; an arm whose `<arm>.ids.json` is already there
//! is reused, so the arms can run first as separate time-boxed invocations of
//! `sampled_mtp_distribution_arm` (set `HIPFIRE_SAMPLED_MTP_ARM`,
//! `HIPFIRE_MTP_SAMPLED=1`, `HIPFIRE_MTP_SAMPLED_MODE=leviathan` and the arm's
//! route knobs yourself: every arm checks it already runs under its
//! environment and fails naming the variable otherwise). Every arm runs the
//! speculative-rejection-sampling (Leviathan) verifier explicitly; the naive
//! verifier has its own test below.
//!
//! Repeat / presence / frequency penalties (default neutral) reach BOTH arms,
//! as a request's would: `HIPFIRE_SAMPLED_MTP_REPEAT_PENALTY` (1.0),
//! `HIPFIRE_SAMPLED_MTP_REPEAT_WINDOW` (0 = penalties off),
//! `HIPFIRE_SAMPLED_MTP_PRESENCE_PENALTY` (0.0) and
//! `HIPFIRE_SAMPLED_MTP_FREQUENCY_PENALTY` (0.0). The AR arm applies them
//! over the full rendered prompt plus its generated IDs (host `sample_cpu`);
//! the MTP arm installs them on its `SpecRequestConfig`. A malformed value
//! fails the run rather than falling back to neutral.
//!
//! `HIPFIRE_SAMPLED_MTP_PROMPT` selects the prompt fixture: `prose` (default,
//! the prompt above, so old neutral runs reproduce), `repeat` (a
//! repetition-heavy pattern-continuation prompt) or `window` (a prompt of
//! roughly 250 tokens whose distinctive head lies beyond a 128-token window,
//! so W=128 crosses the prompt / generated boundary from the first token;
//! the arm fails unless the prompt is longer than the window). In
//! `naive_sampled_mtp_emits_seeded_ar_ids` a non-`prose` fixture is added as
//! a third prompt.
//!
//! Every arm records its full config (sampling, penalties, fixture, trials,
//! `HIPFIRE_MTP_SAMPLED_MODE`, `HIPFIRE_MTP_INCREMENTAL`) in a sidecar
//! `<ids file stem>.config.json` and prints it. An existing ids file is
//! reused only when its recorded config equals the current one; a missing or
//! different config is a hard error. Use one `HIPFIRE_SAMPLED_MTP_OUT`
//! directory per config (e.g. `.../neutral`, `.../presence1.5`).
//!
//! # N-gram composition arms
//!
//! `sampled_mtp_ngram_matches_ar_distribution_on_flash_next` repeats the
//! comparison with the request-local n-gram pool armed (drafter built
//! `.with_ngram(Some(qwen4_ngram_mod_config((5, 3, 3))?))
//! .with_copy_lane(copy)`, request
//! `allow_ngram_modifier: true`, the same neutral T/top_p/top_k/min_p, no
//! penalties), `HIPFIRE_SAMPLED_MTP_TRIALS` (>= 2000) cold trials per arm.
//! These arms follow the product switch: `copy` is
//! `hipfire_config::mtp_ngram_copy_enabled_for_arch` of the loaded GPU
//! (`speculation.mtp_ngram_copy` / `HIPFIRE_MTP_NGRAM_COPY`, off by default:
//! the native-MTP n-gram takeover; `HIPFIRE_MTP_NGRAM_COPY=on` runs them on
//! the copy lane). The mode is recorded as `copy_lane` in every arm's config,
//! and the parent resolves it from each arm's receipt GPU, so ids of one mode
//! are never reused as the other's:
//!
//! - `ngram_batched` / `ngram_interleaved`: the prose prompt, so the pool
//!   mostly misses and the native window runs; compared with the existing
//!   `ar` arm (a window-source mix must leave the target law unchanged);
//!   native windows (`mtp_windows`) must occur;
//! - `copy_ar`: AR on the copied-span fixture (the reference);
//! - `copy_batched` / `copy_interleaved`: the copied-span fixture, whose
//!   prompt is a "rewrite this passage" request with the passage's first
//!   `COPY_PRIMED` tokens already in the assistant turn, so the pool proposes
//!   the passage's continuation on the first window (the fixture is
//!   pre-checked on the CPU against `MtpCopyContext` (copy lane) or
//!   `MtpNgramContext` (otherwise) for the prompt). Hits
//!   are REQUIRED: `request_stats().ngram_mod_windows`, summed over the
//!   trials, and the offered drafts must both be nonzero, in the arm and again
//!   in the parent from `<arm>.stats.json`.
//!
//! Every MTP arm is compared with its AR reference by the same per-position
//! marginals and (t1, t2) joint chi-square (every p-value > `ALPHA`), and
//! replays its first `REPLAYS` seeds: identical IDs and identical window
//! counters. Each arm runs in a fresh process of
//! `sampled_mtp_ngram_distribution_arm` (`ar` still comes from
//! `sampled_mtp_distribution_arm`; an arm whose `<arm>.ids.json` exists is
//! reused). The `HIPFIRE_SAMPLED_MTP_*_PENALTY` / `_REPEAT_WINDOW` env
//! reaches the n-gram arms and `copy_ar` exactly as above, so a presence-1.5
//! run checks penalized `p` on the n-gram rows (one output dir per config).
//!
//! # Long copy arms
//!
//! `sampled_mtp_copy_long_matches_ar_distribution_on_flash_next` compares
//! `copy_long_batched` and `copy_long_adaptive` with `copy_long_ar` on a longer
//! verbatim-rewrite fixture: 48 IDs per cold trial, at least 2000 trials.
//! These arms always run the copy lane (`.with_copy_lane(true)`, config
//! `copy_lane: true`, whatever the product switch says) with the copy triple
//! `hipfire_config::ngram_mod_copy_triple()` (5/1/63 plus its env
//! overrides), recorded in each config. The pool is CPU-prechecked for
//! 48 continuation tokens. Marginals at 0,1,2,3,7,11,15,23,31,47 and joints
//! (0,1), (7,8), (15,16), (31,32) use the same pooled homogeneity test and
//! `ALPHA`. Batched is forced floor-off and replays the first `REPLAYS` seeds
//! with identical IDs and window counters; adaptive uses the default floor
//! and route, with no replay. Each MTP arm must show n-gram windows and at
//! least one takeover verifying seed + >=4 candidates (>=5 rows); the report
//! also states whether public window data can count unconfirmed takeovers.
//! `HIPFIRE_SAMPLED_MTP_COPY_LONG_ARMS` (comma list, unset = both MTP arms)
//! names the MTP arms the parent runs and compares; each costs about as much
//! GPU time as the AR reference (~2 s per trial on an R9700), so a run may
//! name only `copy_long_adaptive`, the product route. Set it identically for
//! every shard and the parent.
//!
//! Run from the repository root (set `MODEL` and `OUT` first). Use the SAME
//! sampling/penalty/runtime environment and executable for all shards and
//! the parent; each shard's config/receipt/stats/evidence/text accompanies
//! `<arm>.<START>-<END>.ids.json`. Smaller disjoint ranges fit 600-second
//! holds; seeds use global trial indices, not shard-local indices:
//!
//! ```sh
//! export HIPFIRE_MTP_IDENTITY_MODEL="$MODEL" HIPFIRE_SAMPLED_MTP_OUT="$OUT"
//! export HIPFIRE_SAMPLED_MTP_TRIALS=2000 HIPFIRE_SAMPLED_MTP_PROMPT=prose
//! export HIPFIRE_MTP_SAMPLED=1 HIPFIRE_MTP_SAMPLED_MODE=leviathan
//! unset HIPFIRE_MTP_TRACE HIPFIRE_MTP_PHASE_TIMING
//! # Repeat START..END ranges to cover 0..2000 for EACH arm.
//! export HIPFIRE_SAMPLED_MTP_SHARD=0..10
//! export HIPFIRE_SAMPLED_MTP_ARM=copy_long_ar
//! unset HIPFIRE_MTP_INCREMENTAL HIPFIRE_MTP_AR_FLOOR
//! ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 cargo test --release -p hipfire-arch-qwen4 --test sampled_mtp_distribution_hw sampled_mtp_copy_long_distribution_arm -- --exact --ignored --nocapture --test-threads=1
//! export HIPFIRE_SAMPLED_MTP_ARM=copy_long_batched
//! export HIPFIRE_MTP_INCREMENTAL=0 HIPFIRE_MTP_AR_FLOOR=0
//! ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 cargo test --release -p hipfire-arch-qwen4 --test sampled_mtp_distribution_hw sampled_mtp_copy_long_distribution_arm -- --exact --ignored --nocapture --test-threads=1
//! export HIPFIRE_SAMPLED_MTP_ARM=copy_long_adaptive
//! unset HIPFIRE_MTP_INCREMENTAL HIPFIRE_MTP_AR_FLOOR
//! ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 cargo test --release -p hipfire-arch-qwen4 --test sampled_mtp_distribution_hw sampled_mtp_copy_long_distribution_arm -- --exact --ignored --nocapture --test-threads=1
//! unset HIPFIRE_SAMPLED_MTP_ARM HIPFIRE_SAMPLED_MTP_SHARD
//! ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 cargo test --release -p hipfire-arch-qwen4 --test sampled_mtp_distribution_hw sampled_mtp_copy_long_matches_ar_distribution_on_flash_next -- --exact --ignored --nocapture --test-threads=1
//! ```
//!
//! The parent config-verifies reusable shards, rejects overlaps and runs only
//! missing ranges in fresh children. Coverage is aggregate, never used to
//! select trials or shards. Existing four-token arms and configs are unchanged.
//!
//! `naive_sampled_mtp_emits_seeded_ar_ids` checks the stronger property of
//! `HIPFIRE_MTP_SAMPLED_MODE=naive`: with the same seed, every MTP route
//! (`batched`, `interleaved`, `adaptive` = route knob unset) emits the AR
//! arm's exact IDs. Cases: the serve bench's `lru_cache_pep8_strict` and
//! `prose_river_short` prompts × T0.7/top_p 0.8/top_k 20, T1.0/top_p 0.95
//! and T0.8/top_p 0.95/top_k 40/min_p 0.05 × `ID_SEEDS` seeds, `ID_TOKENS`
//! IDs each (cut after EOS). Arms run as fresh processes of
//! `sampled_mtp_identity_arm`; an arm whose `<arm>.identity.json` already
//! exists in `HIPFIRE_SAMPLED_MTP_OUT` is reused.
//!
//! Both identity arms read the same penalty and prompt-fixture env as above.
//!
//! # Arm environment and sidecars
//!
//! A parent resets what a benchmark shell may have left behind
//! (`HIPFIRE_MTP_INCREMENTAL`, `HIPFIRE_MTP_AR_FLOOR`, `HIPFIRE_MTP_TRACE`,
//! `HIPFIRE_MTP_PHASE_TIMING`) in every child: forced `batched` /
//! `interleaved` arms (also `ngram_*` / `copy_*`) set their route AND
//! `HIPFIRE_MTP_AR_FLOOR=0`; `ar`, `adaptive` and `*_adaptive` arms leave both
//! unset (floor on). `HIPFIRE_*` variables are not otherwise cleaned: they are
//! recorded.
//!
//! Every result's `<ids stem>.config.json` carries the config AND an identity:
//! the effective `HIPFIRE_*` environment (so the resident-sampling levers,
//! the floor and route knobs and the graph knobs), the executable's SHA-256,
//! a hash of the production sources compiled in, and the model artifact's and
//! `.ple-q4` / `.ple-q8` sidecars' sizes and SHA-256. A result is reused only
//! when that whole document equals the one expected now, which now includes
//! the n-gram arms (they had no sidecar). `<ids stem>.receipt.json` adds the
//! GPU arch / UUID / PCI BDF, the graph flags the runtime resolved, git state
//! and host (informational, not compared).
//!
//! MTP arms also write `<arm>.evidence.json` (route / window counters per
//! request, window rows, HIP call-count deltas around each window) and
//! `<arm>.text.txt` (decoded text of the first trials). Only quantities the
//! drafter and the runtime already report are recorded; nothing is inferred.
//! Failures are labelled: `DISTRIBUTION FAILURE (law)` for a p-value under
//! `ALPHA`, `COVERAGE FAILURE` for a run that did not exercise what it claims.
//!
//! # Adaptive late-position case
//!
//! Four output tokens end inside the floor's calibration window plus one probe
//! window, so the short `adaptive` arm cannot reach retirement.
//! `sampled_mtp_adaptive_floor_coverage_pilot_on_flash_next` runs a declared
//! cohort of `PILOT_TRIALS` adaptive requests of `HIPFIRE_SAMPLED_MTP_LATE_TOKENS`
//! (default and minimum 128) IDs and checks, from the per-window records,
//! that at least one request shows calibration (drafted 0, accepted 0,
//! `mtp_windows` 0 -> 1, one position) -> the source's
//! `MTP_FLOOR_PROBE_WINDOWS` probe windows -> `mtp_retired` false -> true ->
//! a later target-only window (`ar_windows` +1, still retired) that emitted at
//! index >= 31. The pilot is coverage, not statistics. If the fixture does
//! not retire, declare a longer `HIPFIRE_SAMPLED_MTP_LATE_TOKENS` or commit a
//! different fixture and restart the whole cohort; never force retirement,
//! drop non-retiring trials or condition on route history.
//! `sampled_mtp_adaptive_late_matches_ar_distribution_on_flash_next` then
//! compares adaptive MTP with AR over `HIPFIRE_SAMPLED_MTP_TRIALS` (default
//! and minimum 2000) independent trials per arm: early marginals 0..=3 and
//! the (t1, t2) joint, late marginals 31, 63, 95, 127 and joints (31, 32),
//! (63, 64), (95, 96), (126, 127), with the same homogeneity test and `ALPHA`,
//! and requires the aggregate coverage above. Trials are split into disjoint
//! `START..END` shards (`HIPFIRE_SAMPLED_MTP_SHARD`; seeds are
//! `trial_seed(salt, trial index)`, so no seed repeats and shards compose);
//! run `sampled_mtp_late_arm` directly per shard / per arm (`late_ar`,
//! `late_adaptive`) to fit fabric holds, then the aggregate test reuses every
//! config-verified shard and runs only the missing ranges.

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::expert_residency as residency;
use hipfire_arch_qwen4::mtp_spec::{qwen4_ngram_mod_config, Qwen4MtpDrafter};
use hipfire_arch_qwen4::Qwen4KvBackend;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::{ModelSource, SourcePayload};
use hipfire_runtime::ngram_mod::{MtpCopyContext, MtpNgramContext, NgramModConfig};
use hipfire_runtime::prompt_frame::{JinjaChatFrame, Message, Role};
use hipfire_runtime::sampler::{sample_cpu, SamplerConfig};
use hipfire_runtime::spec::{MtpDrafter, MtpRequestStats, SpecRequestConfig};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::{DType, Gpu};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[path = "common/hw_identity.rs"]
mod hw_identity;
use hw_identity::Counters;

const MODEL_ENV: &str = "HIPFIRE_MTP_IDENTITY_MODEL";
const OUT_ENV: &str = "HIPFIRE_SAMPLED_MTP_OUT";
const ARM_ENV: &str = "HIPFIRE_SAMPLED_MTP_ARM";
const TRIALS_ENV: &str = "HIPFIRE_SAMPLED_MTP_TRIALS";
const TEMP_ENV: &str = "HIPFIRE_SAMPLED_MTP_TEMP";
const TOP_P_ENV: &str = "HIPFIRE_SAMPLED_MTP_TOP_P";
const TOP_K_ENV: &str = "HIPFIRE_SAMPLED_MTP_TOP_K";
const MIN_P_ENV: &str = "HIPFIRE_SAMPLED_MTP_MIN_P";
const ARM_TEST: &str = "sampled_mtp_distribution_arm";
const REPEAT_PENALTY_ENV: &str = "HIPFIRE_SAMPLED_MTP_REPEAT_PENALTY";
const REPEAT_WINDOW_ENV: &str = "HIPFIRE_SAMPLED_MTP_REPEAT_WINDOW";
const PRESENCE_PENALTY_ENV: &str = "HIPFIRE_SAMPLED_MTP_PRESENCE_PENALTY";
const FREQUENCY_PENALTY_ENV: &str = "HIPFIRE_SAMPLED_MTP_FREQUENCY_PENALTY";
const PROMPT_ENV: &str = "HIPFIRE_SAMPLED_MTP_PROMPT";
const INCREMENTAL_ENV: &str = "HIPFIRE_MTP_INCREMENTAL";
const SAMPLED_MODE_ENV: &str = "HIPFIRE_MTP_SAMPLED_MODE";
const FLOOR_ENV: &str = "HIPFIRE_MTP_AR_FLOOR";
const SHARD_ENV: &str = "HIPFIRE_SAMPLED_MTP_SHARD";
const LATE_TOKENS_ENV: &str = "HIPFIRE_SAMPLED_MTP_LATE_TOKENS";
const LATE_ARM_TEST: &str = "sampled_mtp_late_arm";
/// Harness-control variables: they differ between a parent and each child
/// without changing what the arm computes, so identities exclude them.
const CONTROL_ENV: [&str; 4] = [MODEL_ENV, OUT_ENV, ARM_ENV, SHARD_ENV];

/// The serve battery's prose prompt (`scripts/serve_harness.py`).
const PROSE: &str = "Write a four-sentence story about a lighthouse keeper who finds something unexpected washed up on the rocks.";
const TOKENS: usize = 4;
const MAX_SEQ: usize = 4096;
const MTP_K: usize = 3;
const REPLAYS: usize = 5;
const ALPHA: f64 = 1e-3;
const ARMS: [&str; 4] = ["ar", "batched", "interleaved", "adaptive"];

/// The n-gram triple the pre-existing n-gram arms run (`n_match`, `n_min`,
/// `n_max`): the arch-16 default of `hipfire_config::ngram_mod_triple_for_arch`
/// with the copy lane off, on either lane.
const NGRAM_TRIPLE: (usize, usize, usize) = (5, 3, 3);
/// Fewest independent seeded trials per n-gram arm.
const NGRAM_MIN_TRIALS: usize = 2000;
const NGRAM_ARM_TEST: &str = "sampled_mtp_ngram_distribution_arm";
/// The AR reference arm of the copied-span fixture.
const COPY_AR_ARM: &str = "copy_ar";
const COPY_AR_SALT: u64 = 0xC0A11;

/// Copied-span fixture: the user asks for a rewrite that keeps the wording,
/// and the assistant turn already holds the passage's first `COPY_PRIMED`
/// tokens, so the last five context tokens before the first window's seed
/// (the passage's token `COPY_PRIMED`, when sampled) occur in the prompt.
const COPY_INSTRUCTION: &str = "Rewrite the following passage, keeping its wording wherever you can.\n\n";
const COPY_PASSAGE: &str = "The old lighthouse keeper climbed the spiral stairs every evening to light the great lamp, and every morning he polished the brass until it shone like gold.";
const COPY_PRIMED: usize = 8;

/// One n-gram combo arm.
struct NgramArm {
    name: &'static str,
    /// The AR arm whose distribution this arm must match.
    reference: &'static str,
    /// Runs the copied-span fixture (hits required on forced routes) rather
    /// than the prose prompt.
    copy: bool,
    /// `trial_seed` salt, distinct per arm.
    salt: u64,
}

/// A `*batched` / `*interleaved` name is a forced floor-off route and a
/// `*_adaptive` name the default route (`arm_route`).
static NGRAM_ARMS: [NgramArm; 6] = [
    NgramArm {
        name: "ngram_batched",
        reference: "ar",
        copy: false,
        salt: 0x4E1,
    },
    NgramArm {
        name: "ngram_interleaved",
        reference: "ar",
        copy: false,
        salt: 0x4E2,
    },
    NgramArm {
        name: "ngram_adaptive",
        reference: "ar",
        copy: false,
        salt: 0x4E3,
    },
    NgramArm {
        name: "copy_batched",
        reference: COPY_AR_ARM,
        copy: true,
        salt: 0xC0B1,
    },
    NgramArm {
        name: "copy_interleaved",
        reference: COPY_AR_ARM,
        copy: true,
        salt: 0xC0B2,
    },
    NgramArm {
        name: "copy_adaptive",
        reference: COPY_AR_ARM,
        copy: true,
        salt: 0xC0B3,
    },
];

/// SplitMix64 of `arm_salt ^ trial`, never zero.
fn trial_seed(arm_salt: u64, trial: usize) -> u64 {
    let mut z = (arm_salt ^ trial as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)).max(1)
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Parses `name` strictly (unset = `default`): a malformed value is an error,
/// so a typo cannot silently run the neutral config.
fn strict_env<T: std::str::FromStr>(name: &str, default: T) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Ok(v) => v.trim().parse().map_err(|e| format!("{name}={v}: {e}")),
        Err(_) => Ok(default),
    }
}

/// The request's repeat / presence / frequency penalty controls.
#[derive(Clone, Copy)]
struct Penalties {
    repeat_penalty: f32,
    repeat_window: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
}

impl Penalties {
    fn from_env() -> Result<Self, String> {
        let p = Self {
            repeat_penalty: strict_env(REPEAT_PENALTY_ENV, 1.0)?,
            repeat_window: strict_env(REPEAT_WINDOW_ENV, 0)?,
            presence_penalty: strict_env(PRESENCE_PENALTY_ENV, 0.0)?,
            frequency_penalty: strict_env(FREQUENCY_PENALTY_ENV, 0.0)?,
        };
        for (name, v) in [
            (REPEAT_PENALTY_ENV, p.repeat_penalty),
            (PRESENCE_PENALTY_ENV, p.presence_penalty),
            (FREQUENCY_PENALTY_ENV, p.frequency_penalty),
        ] {
            if !v.is_finite() {
                return Err(format!("{name}={v} is not finite"));
            }
        }
        Ok(p)
    }
}

/// The prompt a run decodes from (`HIPFIRE_SAMPLED_MTP_PROMPT`).
#[derive(Clone, Copy, PartialEq)]
enum Fixture {
    Prose,
    Repeat,
    Window,
}

impl Fixture {
    fn from_env() -> Result<Self, String> {
        match std::env::var(PROMPT_ENV).as_deref() {
            Err(_) | Ok("prose") => Ok(Self::Prose),
            Ok("repeat") => Ok(Self::Repeat),
            Ok("window") => Ok(Self::Window),
            Ok(other) => Err(format!(
                "{PROMPT_ENV}={other}: expected prose, repeat or window"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Prose => "prose",
            Self::Repeat => "repeat",
            Self::Window => "window",
        }
    }

    fn text(self) -> String {
        match self {
            Self::Prose => PROSE.to_string(),
            Self::Repeat => format!(
                "Continue this pattern for as long as you can, writing only the continuation: {}",
                "red green blue red green blue red green blue ".repeat(24)
            ),
            // Head (its distinctive names occur only here), filler, tail: with
            // a 128-token window the head has left the window before the first
            // generated token, and the filler drains out as tokens are added.
            Self::Window => format!(
                "Harbor log, early shift: the copper ferry Marigold, the gray tug Osprey, \
                 and the salt barge Tern waited by the cold quay while rain fell on the cranes. {}\
                 Now forget the harbor log. Write a four-sentence story about a lighthouse \
                 keeper who finds something unexpected washed up on the rocks.",
                "Tide table, ordinary entry: water rose slowly, then fell again, as it always \
                 does along this quiet coast. "
                    .repeat(8)
            ),
        }
    }

    /// The window fixture must really cross the penalty window.
    fn check_prompt(self, prompt_tokens: usize, pen: &Penalties) -> Result<(), String> {
        if self == Self::Window && prompt_tokens <= pen.repeat_window.max(128) + TOKENS {
            return Err(format!(
                "window fixture has only {prompt_tokens} prompt tokens; it must exceed the {} \
                 token penalty window",
                pen.repeat_window.max(128)
            ));
        }
        Ok(())
    }
}

/// Per-window draft budget of `Loaded::mtp_request_with_budget`, before the
/// remaining-token clamp.
#[derive(Clone, Copy)]
enum DraftBudget {
    /// A fixed budget: every arm but the long-copy ones passes `MTP_K`.
    Fixed(usize),
    /// The drafter's proposal capacity read at every window, after the
    /// request armed the n-gram source, as the product spec loop does
    /// (`mtp_draft_k(proposal_capacity(), max_emit)`). Read before
    /// `configure_request` it would be the previous request's (a fresh
    /// drafter's first request would run at `max_k`).
    ProposalCapacity,
}

/// One request's decode: IDs, window totals, final request stats and the
/// per-window records of `Loaded::mtp_request`.
struct MtpRun {
    ids: Vec<u32>,
    drafted: usize,
    accepted: usize,
    stats: MtpRequestStats,
    windows: Vec<WindowRec>,
}

/// Penalties, fixture and the arm's identity as recorded JSON fields.
fn config_base(
    kind: &str,
    arm: &str,
    pen: &Penalties,
    fixture: Fixture,
    identity: &Value,
) -> Value {
    serde_json::json!({
        "kind": kind,
        "arm": arm,
        "fixture": fixture.name(),
        "repeat_penalty": pen.repeat_penalty,
        "repeat_window": pen.repeat_window,
        "presence_penalty": pen.presence_penalty,
        "frequency_penalty": pen.frequency_penalty,
        "mtp_k": MTP_K,
        "identity": identity,
    })
}

fn dist_config(
    arm: &str,
    identity: &Value,
    pen: &Penalties,
    fixture: Fixture,
    sampling: (f32, f32, Option<u32>, f32),
    trials: usize,
) -> Value {
    let mut config = config_base("distribution", arm, pen, fixture, identity);
    config["temp"] = serde_json::json!(sampling.0);
    config["top_p"] = serde_json::json!(sampling.1);
    config["top_k"] = serde_json::json!(sampling.2);
    config["min_p"] = serde_json::json!(sampling.3);
    config["trials"] = serde_json::json!(trials);
    config["tokens"] = serde_json::json!(TOKENS);
    config["sampled_mode"] = serde_json::json!("leviathan");
    config
}

fn id_config(arm: &str, identity: &Value, pen: &Penalties, fixture: Fixture) -> Value {
    let mut config = config_base("identity", arm, pen, fixture, identity);
    config["tokens"] = serde_json::json!(ID_TOKENS);
    config["seeds"] = serde_json::json!(ID_SEEDS);
    config["sampled_mode"] = serde_json::json!("naive");
    config
}

/// An existing `ids_path` may only be reused (or overwritten) under the config
/// recorded next to it.
fn check_config(ids_path: &Path, expected: &serde_json::Value) -> Result<(), String> {
    let config_path = ids_path.with_extension("config.json");
    let recorded: serde_json::Value = match fs::read(&config_path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map_err(|e| format!("{}: {e}", config_path.display()))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(format!(
                "{} exists but {} does not: its config is unknown; remove it or use another {OUT_ENV}",
                ids_path.display(),
                config_path.display()
            ));
        }
        Err(e) => return Err(format!("{}: {e}", config_path.display())),
    };
    if &recorded != expected {
        return Err(format!(
            "{} was produced under a different config; remove it or use another {OUT_ENV}\n  recorded: {recorded}\n  current:  {expected}",
            ids_path.display()
        ));
    }
    Ok(())
}

fn write_config(ids_path: &Path, config: &serde_json::Value) -> Result<(), String> {
    let config_path = ids_path.with_extension("config.json");
    fs::write(
        &config_path,
        serde_json::to_vec_pretty(config).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{}: {e}", config_path.display()))
}

/// The informational receipt (GPU identity, resolved graph flags, git state)
/// next to the ids file: `<ids stem>.receipt.json`.
fn write_receipt(ids_path: &Path, receipt: &Value) -> Result<(), String> {
    let path = ids_path.with_extension("receipt.json");
    fs::write(
        &path,
        serde_json::to_vec_pretty(receipt).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{}: {e}", path.display()))
}

/// Sampling / trial / penalty / fixture settings of a distribution run.
struct DistSettings {
    trials: usize,
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: f32,
    pen: Penalties,
    fixture: Fixture,
}

impl DistSettings {
    fn from_env() -> Result<Self, String> {
        Ok(Self {
            trials: env_or(TRIALS_ENV, 2000),
            temp: env_or(TEMP_ENV, 1.0),
            top_p: env_or(TOP_P_ENV, 0.95),
            top_k: std::env::var(TOP_K_ENV)
                .ok()
                .map(|v| v.parse().map_err(|e| format!("{TOP_K_ENV}={v}: {e}")))
                .transpose()?,
            min_p: env_or(MIN_P_ENV, 0.0),
            pen: Penalties::from_env()?,
            fixture: Fixture::from_env()?,
        })
    }

    fn config(&self, arm: &str, identity: &Value) -> Value {
        dist_config(
            arm,
            identity,
            &self.pen,
            self.fixture,
            (self.temp, self.top_p, self.top_k, self.min_p),
            self.trials,
        )
    }
}

fn model_path() -> PathBuf {
    let path = std::env::var_os(MODEL_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{MODEL_ENV} must name qwen3.8-flash-next-gptq3.mq4"));
    assert!(
        path.is_file(),
        "{MODEL_ENV}={} is not a file",
        path.display()
    );
    path
}

fn out_dir() -> PathBuf {
    let dir = std::env::var_os(OUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("qwen4-sampled-mtp-{}", std::process::id()))
        });
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    dir
}

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
        1.0 - sum * ln_pre.exp()
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
        ln_pre.exp() * h
    }
}

/// Two-sample chi-square homogeneity test of category counts `a` vs `b`.
/// Categories whose expected count is under 5 in either sample are pooled.
/// Returns `(statistic, degrees of freedom, p-value)`.
fn homogeneity<K: std::hash::Hash + Eq + Clone>(a: &[K], b: &[K]) -> (f64, usize, f64) {
    let mut cells: HashMap<K, (f64, f64)> = HashMap::new();
    for key in a {
        cells.entry(key.clone()).or_default().0 += 1.0;
    }
    for key in b {
        cells.entry(key.clone()).or_default().1 += 1.0;
    }
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let n = na + nb;
    let mut kept: Vec<(f64, f64)> = Vec::new();
    let mut pooled = (0.0, 0.0);
    for &(x, y) in cells.values() {
        let total = x + y;
        if total * na.min(nb) / n < 5.0 {
            pooled.0 += x;
            pooled.1 += y;
        } else {
            kept.push((x, y));
        }
    }
    if pooled.0 + pooled.1 > 0.0 {
        kept.push(pooled);
    }
    let mut stat = 0.0;
    for &(x, y) in &kept {
        let total = x + y;
        let (ea, eb) = (total * na / n, total * nb / n);
        stat += (x - ea).powi(2) / ea + (y - eb).powi(2) / eb;
    }
    let df = kept.len().saturating_sub(1).max(1);
    (stat, df, gamma_q(df as f64 / 2.0, stat / 2.0))
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn sampled_mtp_matches_ar_distribution_on_flash_next() {
    let model = model_path();
    let dir = out_dir();
    let model_id = hw_identity::model_identity(&model, &dir).unwrap_or_else(|e| panic!("{e}"));
    let mut samples: Vec<Vec<Vec<u32>>> = Vec::new();
    let settings = DistSettings::from_env().unwrap_or_else(|e| panic!("{e}"));
    for arm in ARMS {
        let ids_path = dir.join(format!("{arm}.ids.json"));
        let expected = settings.config(
            arm,
            &arm_identity(arm, "leviathan", &model_id, &CONTROL_ENV),
        );
        if ids_path.exists() {
            eprintln!("reusing {}", ids_path.display());
        } else {
            spawn_arm(ARM_TEST, arm, "leviathan", &model, &dir, None, arm)
                .unwrap_or_else(|e| panic!("{e}"));
        }
        // Reused and fresh results alike must carry the config expected now.
        check_config(&ids_path, &expected).unwrap_or_else(|e| panic!("{e}"));
        samples.push(
            serde_json::from_slice(&fs::read(&ids_path).expect("arm ids")).expect("arm ids json"),
        );
    }
    let mut report = format!(
        "config (ar arm): {}\n",
        settings.config(
            "ar",
            &arm_identity("ar", "leviathan", &model_id, &CONTROL_ENV)
        )
    );
    let mut failures = Vec::new();
    let mut coverage = Vec::new();
    let ar = &samples[0];
    for (arm, mtp) in ARMS.iter().zip(&samples).skip(1) {
        compare_to_reference(arm, "ar", ar, mtp, &mut report, &mut failures);
        let evidence = read_json(&dir.join(format!("{arm}.evidence.json")))
            .unwrap_or_else(|e| panic!("{e}"));
        short_arm_coverage(arm, &evidence, &mut report, &mut coverage);
    }
    eprint!("{report}");
    fs::write(dir.join("report.txt"), &report).unwrap();
    assert!(
        failures.is_empty() && coverage.is_empty(),
        "{}",
        failure_message(&dir, &failures, &coverage)
    );
}

/// One arm in a fresh process; spawned by the test above.
#[test]
#[ignore = "child process of sampled_mtp_matches_ar_distribution_on_flash_next"]
fn sampled_mtp_distribution_arm() {
    let Ok(arm) = std::env::var(ARM_ENV) else {
        return;
    };
    if let Err(error) = run_arm(&arm, &model_path(), &out_dir()) {
        panic!("{arm} arm: {error}");
    }
}

/// A loaded Flash-Next target with its tokenizer and chat template.
struct Loaded {
    gpu: Gpu,
    bundle: Qwen4Bundle,
    tokenizer: Tokenizer,
    template: String,
    state: String,
}

fn load(model: &Path) -> Result<Loaded, String> {
    let mut hfq = HfqFile::open(model).map_err(|e| e.to_string())?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| e.to_string())?;
    let template = hfq.chat_template().ok_or("artifact has no chat template")?;
    if let Some(devices) = hipfire_config::devices::startup_devices() {
        let archs = devices
            .iter()
            .map(|device| device.arch.as_str())
            .collect::<Vec<_>>();
        if residency::keeps_host_memory_out_of_reclaim(hipfire_arch_qwen4::ARCH_ID, &archs) {
            hip_bridge::keep_host_memory_out_of_reclaim(&format!(
                "Qwen4 model on {}",
                archs.join(",")
            ));
        }
    }
    let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
    // Admission is GPU-arch aware: an expert-slab artifact admits only on a
    // supported arch. Initialising the GPU allocates no model memory.
    let receipt = hipfire_arch_qwen4::admit_hfqm_artifact_on(&hfq, Some(&gpu.arch))
        .map_err(|e| e.to_string())?;
    // The loader's forward scratch plan, decided from the actual payload
    // while the mapping is readable.
    let scratch_mode = hipfire_arch_qwen4::artifact::source_forward_scratch_mode(
        &gpu,
        &hfq,
        &receipt.config,
        &receipt.manifest,
    )
    .map_err(|e| e.to_string())?;
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        "",
        &gpu,
        &receipt.config,
    )?;
    let backend = Qwen4KvBackend::automatic(&gpu);
    let use_ranges = gpu.is_uma();
    let mut weights = receipt.manifest.weights.clone();
    let explicit = residency::expert_vram_layers_from_env()?;
    if use_ranges {
        if explicit.is_some() {
            return Err(format!(
                "qwen4: {} places experts in host RAM, which only a discrete GPU needs",
                residency::EXPERT_VRAM_LAYERS_ENV
            ));
        }
    } else {
        // A slab layer arena is one span of the container, not an index
        // tensor: it declares its byte size as its logical shape.
        let bytes_of = |entry: &hipfire_runtime::weight_manifest::WeightEntry| {
            if entry.name.ends_with(hipfire_arch_qwen4::expert_arena::ARENA_SUFFIX) {
                return match entry.logical_shape.as_slice() {
                    [bytes] => Some(*bytes as u64),
                    _ => None,
                };
            }
            hfq.tensor_data(&entry.name)
                .map(|(_, bytes)| bytes.len() as u64)
        };
        let (non_expert, layer_experts) = residency::resident_split(&weights, bytes_of)?;
        let experts = residency::routed_expert_bytes(&weights, bytes_of)?;
        let (free, _) = gpu.hip.get_vram_info().map_err(|e| e.to_string())?;
        let free = free as u64;
        let chunk =
            hipfire_arch_qwen4::gpu_forward::qwen4_prefill_chunk_requested(&gpu.arch, MAX_SEQ);
        let context = hipfire_arch_qwen4::Qwen4ContextCommit::for_expert_reserve(
            backend,
            gpu.is_uma(),
            MAX_SEQ,
            chunk,
            gpu.vmm_recommended_granularity().unwrap_or(1),
        );
        // Every oracle arm reserves the same native MTP footprint.
        let head = residency::language_head_dtype(&weights)
            .ok_or("qwen4: manifest has no language head")?;
        let row_capture =
            hipfire_arch_qwen4::mtp_spec::native_mtp_row_capture(&gpu, &receipt.config)
                .then_some(state_format.gdn);
        let mtp = hipfire_arch_qwen4::mtp_spec::native_mtp_device_bytes(
            &receipt.config,
            &context,
            chunk,
            MTP_K,
            head,
            row_capture,
            hipfire_arch_qwen4::mtp_spec::native_mtp_fill_scratch(&gpu),
        )
        .ok_or("qwen4: native MTP device bytes overflow")?;
        let gather = rdna_compute::tensor_ops::qsa_gather_workspace_needed(
            &gpu,
            state_format.qsa,
            state_format.qsa_f16_arena,
        )
        .then(|| {
            rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
                receipt.config.num_key_value_heads,
                context.tokens,
            )
            .map(|bytes| {
                (bytes as u64).saturating_sub(gpu.qsa_gather_scratch_bytes() as u64)
            })
            .ok_or("qwen4: QSA gather scratch size overflows")
        })
        .transpose()?;
        let reserve = residency::auto_vram_reserve(
            &receipt.config,
            &context,
            chunk,
            scratch_mode,
            state_format.layout(),
            Some(mtp),
            gather,
        )?;
        let placement = residency::resolve_expert_vram_layers(explicit, || {
            Ok(residency::fits_fully_resident(free, non_expert, experts, reserve))
        })?;
        if let Some(policy) = placement {
            let vram_layers = match policy {
                residency::ExpertVramLayers::Layers(layers) => layers,
                residency::ExpertVramLayers::Auto => {
                    let needed = non_expert
                        .checked_add(reserve)
                        .ok_or("qwen4: non-expert weights plus reserve overflow")?;
                    if free < needed {
                        return Err(format!(
                            "qwen4: {free} free VRAM bytes cannot hold {non_expert} non-expert weight bytes plus {reserve} reserve bytes even with every routed expert in host RAM"
                        ));
                    }
                    residency::auto_vram_layers(
                        free,
                        non_expert,
                        layer_experts,
                        receipt.config.num_hidden_layers,
                        reserve,
                    )
                }
            };
            let moved = residency::place_routed_experts(&mut weights, vram_layers);
            let host = residency::host_mapped_bytes(&weights, bytes_of)?;
            residency::check_host_ram(
                host,
                rdna_compute::kv_slots::mem_available_bytes(),
                residency::ttm_pool_estimate(),
            )?;
            residency::check_gtt_cap(host, residency::gtt_budget())?;
            eprintln!(
                "qwen4 routed experts: layers 0..{} in VRAM, {moved} tensors ({host} bytes) in pinned host RAM",
                vram_layers.min(receipt.config.num_hidden_layers)
            );
        } else {
            eprintln!(
                "qwen4 routed experts: {} MoE layers fully resident, 0 host-mapped bytes",
                residency::moe_layer_count(&receipt.config)
            );
        }
    }
    // Placement has finished reading payloads before UMA releases the mmap.
    if use_ranges {
        hfq.drop_mmap();
    }
    let mesh = DeviceMesh::single().map_err(|e| e.to_string())?;
    let expected = WeightOrigin::for_single(&mesh, &gpu);
    let source = HfqModelSource::from_hfq(hfq);
    let transaction = fulfill_manifest_from_payloads(
        &weights,
        &mesh,
        receipt.config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            if let Some(payload) = hipfire_arch_qwen4::expert_arena::recipe_arena_payload(
                &source,
                &receipt.manifest.identity.recipe,
                entry,
            ) {
                return payload;
            }
            if use_ranges || entry.residency.is_external() {
                return source
                    .tensor_range(&entry.name)
                    .map_err(|e| e.to_string())?
                    .map(SourcePayload::Range)
                    .ok_or_else(|| format!("missing tensor '{}'", entry.name));
            }
            let (info, bytes) = source
                .tensor_data(&entry.name)
                .ok_or_else(|| format!("missing tensor '{}'", entry.name))?;
            Ok(SourcePayload::Borrowed { info, bytes })
        },
    )
    .map_err(|e| e.to_string())?;
    let mut bundle = Qwen4Bundle::assemble_with_metadata(
        receipt.config,
        &receipt.manifest.identity.recipe,
        transaction,
        &receipt.placements,
        &mut gpu,
        MAX_SEQ,
        receipt.ple,
        state_format,
        backend,
    )
    .map_err(|e| e.to_string())?;
    bundle
        .attach_forward(&mut gpu, MAX_SEQ, scratch_mode)
        .map_err(|e| e.to_string())?;
    Ok(Loaded {
        gpu,
        bundle,
        tokenizer,
        template,
        state: format!("{state_format:?}"),
    })
}

impl Loaded {
    /// `text` as one user message, thinking off, rendered as `hipfire serve`
    /// renders it.
    fn render(&self, text: &str) -> Result<Vec<u32>, String> {
        self.render_with(text, false)
    }

    /// [`render`](Self::render) with the thinking switch given.
    fn render_with(&self, text: &str, thinking: bool) -> Result<Vec<u32>, String> {
        let user = Message {
            role: Role::User,
            content: text.to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let rendered = JinjaChatFrame {
            tokenizer: &self.tokenizer,
            template: &self.template,
            system: None,
            user: text,
            enable_thinking: thinking,
            bos_token: None,
            reasoning_strength: None,
            reasoning_effort: None,
        }
        .render_messages(&[user], None, None)?;
        Ok(self.tokenizer.encode(&rendered))
    }

    /// Seeded AR from a cold state with the production host sampler: up to
    /// `tokens` IDs, cut after EOS.
    fn ar_ids(
        &mut self,
        logits: &rdna_compute::GpuTensor,
        prompt: &[u32],
        cfg: &SamplerConfig,
        seed: u32,
        tokens: usize,
    ) -> Result<Vec<u32>, String> {
        let eos = self.bundle.config.eos_token_id;
        hipfire_runtime::llama::reset_cpu_sampler_rng(seed);
        self.bundle
            .reset(&mut self.gpu)
            .map_err(|e| e.to_string())?;
        self.bundle
            .forward_chunk_final(&mut self.gpu, prompt, logits, None)
            .map_err(|e| e.to_string())?;
        let mut history = prompt.to_vec();
        let mut ids = Vec::with_capacity(tokens);
        loop {
            let mut row = self.gpu.download_f32(logits).map_err(|e| e.to_string())?;
            let token = sample_cpu(&mut row, &history, cfg);
            ids.push(token);
            history.push(token);
            if ids.len() == tokens || token == eos {
                return Ok(ids);
            }
            self.bundle
                .forward_token_or_argmax(&mut self.gpu, Some(token), logits)
                .map_err(|e| e.to_string())?;
        }
    }

    /// Sampled native MTP from a cold state: up to `tokens` IDs, cut after
    /// EOS. Returns the IDs and the windows' (drafts, accepted) totals.
    fn mtp_ids(
        &mut self,
        drafter: &mut Qwen4MtpDrafter,
        prompt: &[u32],
        cfg: SpecRequestConfig,
        tokens: usize,
    ) -> Result<(Vec<u32>, usize, usize), String> {
        self.mtp_request(drafter, prompt, cfg, tokens)
            .map(|run| (run.ids, run.drafted, run.accepted))
    }

    /// [`mtp_ids`](Self::mtp_ids) plus the request's `request_stats()` and a
    /// record of every window: the stats and position before and after, the
    /// window's drafts / accepted / committed, and the HIP call-count deltas
    /// around the `mtp_step`. This is the one decode loop of every MTP arm.
    fn mtp_request(
        &mut self,
        drafter: &mut Qwen4MtpDrafter,
        prompt: &[u32],
        cfg: SpecRequestConfig,
        tokens: usize,
    ) -> Result<MtpRun, String> {
        self.mtp_request_with_budget(drafter, prompt, cfg, tokens, DraftBudget::Fixed(MTP_K))
    }

    /// [`mtp_request`](Self::mtp_request) with the per-window draft budget
    /// passed to `mtp_step` (before the remaining-token clamp) given: the
    /// long-copy arms pass [`DraftBudget::ProposalCapacity`] (the n-gram take
    /// cap's width once armed), every other arm `MTP_K`.
    fn mtp_request_with_budget(
        &mut self,
        drafter: &mut Qwen4MtpDrafter,
        prompt: &[u32],
        cfg: SpecRequestConfig,
        tokens: usize,
        draft_budget: DraftBudget,
    ) -> Result<MtpRun, String> {
        let eos = self.bundle.config.eos_token_id;
        drafter.configure_request(cfg);
        let mut seed_token = drafter
            .mtp_prefill(
                &mut self.gpu,
                &mut self.bundle,
                prompt,
                prompt,
                0,
                false,
                &|| false,
            )?
            .into_first_token()?;
        let mut ids = vec![seed_token];
        let (mut drafted, mut accepted) = (0, 0);
        let mut windows = Vec::new();
        while ids.len() < tokens && seed_token != eos {
            let position = self.bundle.state.position;
            let budget = match draft_budget {
                DraftBudget::Fixed(budget) => budget,
                DraftBudget::ProposalCapacity => MtpDrafter::proposal_capacity(drafter),
            };
            let k = budget.min(tokens - ids.len());
            let before = StatsSnap::of(&drafter.request_stats());
            let started = Counters::now();
            let window = drafter.mtp_step(
                &mut self.gpu,
                &mut self.bundle,
                position,
                seed_token,
                &ids,
                k,
                eos,
                None,
            )?;
            let counters = Counters::now().since(started);
            windows.push(WindowRec {
                index: ids.len(),
                position_before: position,
                position_after: self.bundle.state.position,
                k_budget: k,
                drafts: window.drafts_generated,
                accepted: window.accepted,
                committed: window.committed.len(),
                before,
                after: StatsSnap::of(&drafter.request_stats()),
                counters,
            });
            drafted += window.drafts_generated;
            accepted += window.accepted;
            seed_token = *window.committed.last().ok_or("empty MTP window")?;
            ids.extend_from_slice(&window.committed);
        }
        ids.truncate(tokens);
        Ok(MtpRun {
            ids,
            drafted,
            accepted,
            stats: drafter.request_stats(),
            windows,
        })
    }
}

fn ar_sampler(
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: f32,
    pen: &Penalties,
) -> SamplerConfig {
    SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty: pen.repeat_penalty,
        repeat_window: pen.repeat_window,
        presence_penalty: pen.presence_penalty,
        frequency_penalty: pen.frequency_penalty,
        blocked_tokens: Vec::new(),
        top_k,
        min_p: (min_p > 0.0).then_some(min_p),
    }
}

fn spec_request(
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: f32,
    seed: u64,
    pen: &Penalties,
) -> SpecRequestConfig {
    SpecRequestConfig {
        temp,
        top_p,
        top_k,
        min_p,
        cactus_delta: 0.0,
        rng_seed: seed,
        allow_ngram_modifier: false,
        repeat_penalty: pen.repeat_penalty,
        repeat_window: pen.repeat_window,
        presence_penalty: pen.presence_penalty,
        frequency_penalty: pen.frequency_penalty,
        ..SpecRequestConfig::default()
    }
}

/// [`spec_request`] with the n-gram pool armed (`allow_ngram_modifier`).
fn spec_request_ngram(
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: f32,
    seed: u64,
    pen: &Penalties,
) -> SpecRequestConfig {
    SpecRequestConfig {
        allow_ngram_modifier: true,
        ..spec_request(temp, top_p, top_k, min_p, seed, pen)
    }
}

fn new_drafter(loaded: &mut Loaded) -> Result<Qwen4MtpDrafter, String> {
    new_drafter_with(loaded, None, false)
}

/// A sampled-verify drafter; `ngram` is the pool configuration requests that
/// arm `allow_ngram_modifier` use (`None`: every window native) and
/// `copy_lane` runs that pool as the n-gram copy lane.
fn new_drafter_with(
    loaded: &mut Loaded,
    ngram: Option<NgramModConfig>,
    copy_lane: bool,
) -> Result<Qwen4MtpDrafter, String> {
    loaded
        .bundle
        .attach_mtp(&mut loaded.gpu, MAX_SEQ)
        .map_err(|e| e.to_string())?;
    let drafter = Qwen4MtpDrafter::new(MTP_K, MAX_SEQ, None)
        .with_ngram(ngram)
        .with_copy_lane(copy_lane);
    if !drafter.supports_temp_verify() {
        return Err("HIPFIRE_MTP_SAMPLED did not enable sampled verification".into());
    }
    Ok(drafter)
}

fn run_arm(arm: &str, model: &Path, dir: &Path) -> Result<(), String> {
    require_arm_env(arm, "leviathan")?;
    let DistSettings {
        trials,
        temp,
        top_p,
        top_k,
        min_p,
        pen,
        fixture,
    } = DistSettings::from_env()?;
    let env = own_env(&CONTROL_ENV);
    let settings_config = dist_config(
        arm,
        &own_identity(model, dir, &CONTROL_ENV)?,
        &pen,
        fixture,
        (temp, top_p, top_k, min_p),
        trials,
    );
    let ids_path = dir.join(format!("{arm}.ids.json"));
    if ids_path.exists() {
        check_config(&ids_path, &settings_config)?;
    }
    let mut loaded = load(model)?;
    let prompt = loaded.render(&fixture.text())?;
    fixture.check_prompt(prompt.len(), &pen)?;
    let eos = loaded.bundle.config.eos_token_id;
    println!(
        "ARM {arm} arch={} prompt_tokens={} trials={trials} temp={temp} top_p={top_p} top_k={top_k:?} min_p={min_p} repeat_penalty={} repeat_window={} presence_penalty={} frequency_penalty={} fixture={} state={} route_class={}",
        loaded.gpu.arch,
        prompt.len(),
        pen.repeat_penalty,
        pen.repeat_window,
        pen.presence_penalty,
        pen.frequency_penalty,
        fixture.name(),
        loaded.state,
        hw_identity::route_class(&env)
    );
    println!("ARM {arm} config {settings_config}");
    let started = std::time::Instant::now();
    let mut samples: Vec<Vec<u32>> = Vec::with_capacity(trials);
    if arm == "ar" {
        let logits = loaded
            .gpu
            .zeros(&[loaded.bundle.config.vocab_size], DType::F32)
            .map_err(|e| e.to_string())?;
        let cfg = ar_sampler(temp, top_p, top_k, min_p, &pen);
        for trial in 0..trials {
            let seed = trial_seed(0xA11, trial) as u32;
            samples.push(loaded.ar_ids(&logits, &prompt, &cfg, seed, TOKENS)?);
        }
        loaded.gpu.free_tensor(logits).map_err(|e| e.to_string())?;
    } else {
        let mut drafter = new_drafter(&mut loaded)?;
        let budget = floor_probe_budget()?;
        let mut evidence = Evidence::with_samples(EVIDENCE_SAMPLES);
        let mut run = |loaded: &mut Loaded, seed: u64| {
            let cfg = spec_request(temp, top_p, top_k, min_p, seed, &pen);
            loaded.mtp_request(&mut drafter, &prompt, cfg, TOKENS)
        };
        for trial in 0..trials {
            let seed = trial_seed(0x5EC, trial);
            let result = run(&mut loaded, seed)?;
            evidence.observe(trial, seed, &result, &loaded.tokenizer, budget, LATE_MARGINALS[0]);
            samples.push(result.ids);
        }
        // Only a forced floor-off route replays byte for byte; the adaptive
        // route's choices depend on wall-clock prices.
        if hw_identity::forced_floor_off(&env) {
            for trial in 0..REPLAYS.min(trials) {
                let replay = run(&mut loaded, trial_seed(0x5EC, trial))?;
                if replay.ids != samples[trial] {
                    return Err(format!(
                        "seed {} replayed {:?}, first run {:?}",
                        trial + 1,
                        replay.ids,
                        samples[trial]
                    ));
                }
            }
            println!(
                "ARM {arm} replayed {} seeds identically",
                REPLAYS.min(trials)
            );
        } else {
            println!(
                "ARM {arm} replay not asserted (route class {}): its routing depends on wall-clock prices",
                hw_identity::route_class(&env)
            );
        }
        evidence.write(dir, arm, budget, LATE_MARGINALS[0])?;
        Box::new(drafter).mtp_free(&mut loaded.gpu);
    }
    // Positions after an EOS read as EOS.
    for ids in &mut samples {
        ids.resize(TOKENS, eos);
    }
    println!(
        "ARM {arm} {trials} trials in {:.1}s; first sample: {:?}",
        started.elapsed().as_secs_f64(),
        loaded.tokenizer.decode(&samples[0])
    );
    fs::write(
        &ids_path,
        serde_json::to_vec(&samples).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    write_config(&ids_path, &settings_config)?;
    write_receipt(&ids_path, &hw_identity::receipt(&loaded.gpu, model))?;
    loaded
        .bundle
        .free_gpu(&mut loaded.gpu)
        .map_err(|e| e.to_string())
}

const ID_ARM_TEST: &str = "sampled_mtp_identity_arm";
const ID_ARMS: [&str; 4] = ["ar", "batched", "interleaved", "adaptive"];
const ID_TOKENS: usize = 128;
const ID_SEEDS: usize = 6;
/// (name, temperature, top_p, top_k, min_p).
const ID_SAMPLINGS: [(&str, f32, f32, Option<u32>, f32); 3] = [
    ("T0.7/p0.8/k20", 0.7, 0.8, Some(20), 0.0),
    ("T1.0/p0.95", 1.0, 0.95, None, 0.0),
    ("T0.8/p0.95/k40/minp0.05", 0.8, 0.95, Some(40), 0.05),
];
const ID_PROMPTS: [(&str, &str); 2] = [
    (
        "lru_cache",
        include_str!("../../../benchmarks/prompts/lru_cache_pep8_strict.txt"),
    ),
    (
        "prose_river",
        include_str!("../../../benchmarks/prompts/prose_river_short.txt"),
    ),
];

/// One identity case's result in an arm.
struct IdCase {
    case: String,
    ids: Vec<u32>,
    drafted: usize,
    accepted: usize,
}

impl IdCase {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "case": self.case,
            "ids": self.ids,
            "drafted": self.drafted,
            "accepted": self.accepted,
        })
    }

    fn from_json(value: &serde_json::Value) -> Self {
        let count = |key: &str| value[key].as_u64().expect("identity count") as usize;
        Self {
            case: value["case"].as_str().expect("identity case").to_string(),
            ids: serde_json::from_value(value["ids"].clone()).expect("identity ids"),
            drafted: count("drafted"),
            accepted: count("accepted"),
        }
    }
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn naive_sampled_mtp_emits_seeded_ar_ids() {
    let model = model_path();
    let dir = out_dir();
    let mut arms: Vec<Vec<IdCase>> = Vec::new();
    let pen = Penalties::from_env().unwrap_or_else(|e| panic!("{e}"));
    let fixture = Fixture::from_env().unwrap_or_else(|e| panic!("{e}"));
    let model_id = hw_identity::model_identity(&model, &dir).unwrap_or_else(|e| panic!("{e}"));
    for arm in ID_ARMS {
        let ids_path = dir.join(format!("{arm}.identity.json"));
        let expected = id_config(
            arm,
            &arm_identity(arm, "naive", &model_id, &CONTROL_ENV),
            &pen,
            fixture,
        );
        if ids_path.exists() {
            eprintln!("reusing {}", ids_path.display());
        } else {
            spawn_arm(
                ID_ARM_TEST,
                arm,
                "naive",
                &model,
                &dir,
                None,
                &format!("{arm}.identity"),
            )
            .unwrap_or_else(|e| panic!("{e}"));
        }
        // Reused and fresh results alike must carry the config expected now.
        check_config(&ids_path, &expected).unwrap_or_else(|e| panic!("{e}"));
        let json: Vec<serde_json::Value> =
            serde_json::from_slice(&fs::read(&ids_path).expect("arm identity ids"))
                .expect("arm identity json");
        arms.push(json.iter().map(IdCase::from_json).collect());
    }
    let mut report = String::new();
    let mut failures = 0usize;
    let ar = &arms[0];
    for (arm, cases) in ID_ARMS.iter().zip(&arms).skip(1) {
        assert_eq!(cases.len(), ar.len(), "{arm}: case count");
        let mut same = 0usize;
        for (a, m) in ar.iter().zip(cases) {
            assert_eq!(a.case, m.case, "{arm}: case order");
            match a.ids.iter().zip(&m.ids).position(|(x, y)| x != y) {
                None if a.ids.len() == m.ids.len() => same += 1,
                first => {
                    failures += 1;
                    report.push_str(&format!(
                        "{arm} {}: first differs at {} (ar {} ids, mtp {} ids)\n",
                        m.case,
                        first.unwrap_or(a.ids.len().min(m.ids.len())),
                        a.ids.len(),
                        m.ids.len()
                    ));
                }
            }
        }
        let (drafted, accepted) = cases
            .iter()
            .fold((0, 0), |(d, a), c| (d + c.drafted, a + c.accepted));
        report.push_str(&format!(
            "{arm}: {same}/{} cases identical to AR; drafts {drafted}, accepted {accepted}\n",
            ar.len()
        ));
    }
    eprint!("{report}");
    fs::write(dir.join("identity-report.txt"), &report).unwrap();
    assert_eq!(
        failures,
        0,
        "naive sampled MTP left AR's seeded IDs (logs in {}):\n{report}",
        dir.display()
    );
}

/// One identity arm in a fresh process; spawned by the test above.
#[test]
#[ignore = "child process of naive_sampled_mtp_emits_seeded_ar_ids"]
fn sampled_mtp_identity_arm() {
    let Ok(arm) = std::env::var(ARM_ENV) else {
        return;
    };
    if let Err(error) = run_identity_arm(&arm, &model_path(), &out_dir()) {
        panic!("{arm} identity arm: {error}");
    }
}

fn run_identity_arm(arm: &str, model: &Path, dir: &Path) -> Result<(), String> {
    require_arm_env(arm, "naive")?;
    let pen = Penalties::from_env()?;
    let fixture = Fixture::from_env()?;
    let config = id_config(
        arm,
        &own_identity(model, dir, &CONTROL_ENV)?,
        &pen,
        fixture,
    );
    let ids_path = dir.join(format!("{arm}.identity.json"));
    if ids_path.exists() {
        check_config(&ids_path, &config)?;
    }
    println!("ARM {arm} config {config}");
    let mut prompts: Vec<(String, String)> = ID_PROMPTS
        .iter()
        .map(|(name, text)| (name.to_string(), text.to_string()))
        .collect();
    if fixture != Fixture::Prose {
        prompts.push((fixture.name().to_string(), fixture.text()));
    }
    let mut loaded = load(model)?;
    let started = std::time::Instant::now();
    let logits = loaded
        .gpu
        .zeros(&[loaded.bundle.config.vocab_size], DType::F32)
        .map_err(|e| e.to_string())?;
    let mut drafter = if arm == "ar" {
        None
    } else {
        Some(new_drafter(&mut loaded)?)
    };
    let mut out = Vec::new();
    for (prompt_name, text) in &prompts {
        let prompt = loaded.render(text)?;
        if *prompt_name == fixture.name() {
            fixture.check_prompt(prompt.len(), &pen)?;
        }
        for (sampling, temp, top_p, top_k, min_p) in ID_SAMPLINGS {
            for n in 0..ID_SEEDS {
                let seed = trial_seed(0x1D, n) as u32;
                let case = format!("{prompt_name} {sampling} seed#{n}");
                let (ids, drafted, accepted) = match drafter.as_mut() {
                    None => {
                        let cfg = ar_sampler(temp, top_p, top_k, min_p, &pen);
                        let ids = loaded.ar_ids(&logits, &prompt, &cfg, seed, ID_TOKENS)?;
                        (ids, 0, 0)
                    }
                    Some(drafter) => {
                        let cfg = spec_request(temp, top_p, top_k, min_p, seed as u64, &pen);
                        loaded.mtp_ids(drafter, &prompt, cfg, ID_TOKENS)?
                    }
                };
                println!(
                    "ARM {arm} {case}: {} ids, drafts {drafted}, accepted {accepted}",
                    ids.len()
                );
                out.push(IdCase {
                    case,
                    ids,
                    drafted,
                    accepted,
                });
            }
        }
    }
    println!(
        "ARM {arm} {} cases in {:.1}s (arch {}, state {})",
        out.len(),
        started.elapsed().as_secs_f64(),
        loaded.gpu.arch,
        loaded.state
    );
    fs::write(
        &ids_path,
        serde_json::to_vec(&out.iter().map(IdCase::to_json).collect::<Vec<_>>())
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    write_config(&ids_path, &config)?;
    write_receipt(&ids_path, &hw_identity::receipt(&loaded.gpu, model))?;
    if let Some(drafter) = drafter {
        Box::new(drafter).mtp_free(&mut loaded.gpu);
    }
    loaded.gpu.free_tensor(logits).map_err(|e| e.to_string())?;
    loaded
        .bundle
        .free_gpu(&mut loaded.gpu)
        .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Arm environment, identity, coverage and failure classification
// ---------------------------------------------------------------------------

/// Route knob of a forced arm: a `batched` / `interleaved` name is a fixed
/// route (`HIPFIRE_MTP_INCREMENTAL` 0 / 1, floor off); every other arm
/// (`ar`, `adaptive`, `*_adaptive`, `late_*`) runs with no route knob.
fn arm_route(arm: &str) -> Option<&'static str> {
    if arm.ends_with("interleaved") {
        Some("1")
    } else if arm.ends_with("batched") {
        Some("0")
    } else {
        None
    }
}

/// What a parent changes in its child's environment. Every arm resets the
/// inherited route / floor / trace / timing state, so a benchmark shell's
/// forcing variables never reach a result; forced arms set their route and
/// switch the floor off explicitly; the adaptive and `ar` arms leave both
/// unset (floor on, no forced route).
fn arm_overrides(arm: &str, mode: &'static str) -> Vec<(&'static str, Option<&'static str>)> {
    let mut overrides: Vec<(&'static str, Option<&'static str>)> = vec![
        ("HIPFIRE_MTP_SAMPLED", Some("1")),
        (SAMPLED_MODE_ENV, Some(mode)),
        ("HIPFIRE_MTP_TRACE", None),
        ("HIPFIRE_MTP_PHASE_TIMING", None),
    ];
    match arm_route(arm) {
        Some(route) => {
            overrides.push((INCREMENTAL_ENV, Some(route)));
            overrides.push((FLOOR_ENV, Some("0")));
        }
        None => {
            overrides.push((INCREMENTAL_ENV, None));
            overrides.push((FLOOR_ENV, None));
        }
    }
    overrides
}

/// A child (or a separately launched arm) must already be running under its
/// arm's environment: a stale shell variable is an error, not a silently
/// different configuration.
fn require_arm_env(arm: &str, mode: &'static str) -> Result<(), String> {
    for (name, want) in arm_overrides(arm, mode) {
        let have = std::env::var(name).ok();
        if have.as_deref() != want {
            return Err(format!(
                "{name} is {have:?} but arm {arm} requires {want:?}; set it (or unset it) before running this arm directly"
            ));
        }
    }
    Ok(())
}

/// The identity a parent expects of `arm`'s child.
fn arm_identity(arm: &str, mode: &'static str, model: &Value, exclude: &[&str]) -> Value {
    hw_identity::config_identity(
        &hw_identity::env_snapshot(&arm_overrides(arm, mode), exclude),
        model,
    )
}

/// The environment of the running process (a child, or a direct arm run).
fn own_env(exclude: &[&str]) -> BTreeMap<String, String> {
    hw_identity::env_snapshot(&[], exclude)
}

/// The identity of the running process.
fn own_identity(model: &Path, dir: &Path, exclude: &[&str]) -> Result<Value, String> {
    Ok(hw_identity::config_identity(
        &own_env(exclude),
        &hw_identity::model_identity(model, dir)?,
    ))
}

/// Run `test` as a fresh child process for `arm` under the arm's environment;
/// `shard` is the late case's trial range. Logs go to `<log_stem>.out/.err`.
fn spawn_arm(
    test: &str,
    arm: &str,
    mode: &'static str,
    model: &Path,
    dir: &Path,
    shard: Option<(usize, usize)>,
    log_stem: &str,
) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| format!("test binary path: {e}"))?;
    let mut child = Command::new(exe);
    child
        .args([
            "--exact",
            test,
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ARM_ENV, arm)
        .env(MODEL_ENV, model)
        .env(OUT_ENV, dir)
        .stdin(Stdio::null());
    for (name, value) in arm_overrides(arm, mode) {
        match value {
            Some(value) => {
                child.env(name, value);
            }
            None => {
                child.env_remove(name);
            }
        }
    }
    match shard {
        Some((start, end)) => {
            child.env(SHARD_ENV, format!("{start}..{end}"));
        }
        None => {
            child.env_remove(SHARD_ENV);
        }
    }
    let output = child
        .output()
        .map_err(|e| format!("spawn {log_stem}: {e}"))?;
    fs::write(dir.join(format!("{log_stem}.out")), &output.stdout).map_err(|e| e.to_string())?;
    fs::write(dir.join(format!("{log_stem}.err")), &output.stderr).map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "{log_stem} arm failed ({}); logs in {}",
            output.status,
            dir.display()
        ));
    }
    Ok(())
}

/// The two kinds of failure never share a verdict: a distribution failure is
/// a statistic under `ALPHA` (a possible law bug); a coverage failure is a
/// run that did not exercise what it claims to (no law verdict either way).
fn failure_message(dir: &Path, distribution: &[String], coverage: &[String]) -> String {
    let mut message = String::new();
    if !distribution.is_empty() {
        message.push_str(&format!(
            "DISTRIBUTION FAILURE (law): sampled MTP differs from AR in distribution (alpha {ALPHA}; logs in {}):\n{}",
            dir.display(),
            distribution.concat()
        ));
    }
    if !coverage.is_empty() {
        message.push_str(&format!(
            "COVERAGE FAILURE (not a distribution result; logs in {}):\n{}",
            dir.display(),
            coverage.concat()
        ));
    }
    message
}

fn read_json(path: &Path) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{}: {e}", path.display()))
}

/// The `MTP_FLOOR_PROBE_WINDOWS` the binary under test was built with, read
/// from the embedded `mtp_spec.rs` source rather than copied here.
fn floor_probe_budget() -> Result<usize, String> {
    const SOURCE: &str = include_str!("../src/mtp_spec.rs");
    let line = SOURCE
        .lines()
        .find(|line| {
            line.trim_start()
                .starts_with("const MTP_FLOOR_PROBE_WINDOWS: usize")
        })
        .ok_or("MTP_FLOOR_PROBE_WINDOWS is not declared in mtp_spec.rs")?;
    line.split('=')
        .nth(1)
        .map(|value| value.trim().trim_end_matches(';').trim())
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|budget| *budget > 0)
        .ok_or_else(|| format!("cannot read the probe budget from {line:?}"))
}

/// What a window did to the request's counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct StatsSnap {
    mtp_windows: usize,
    ar_windows: usize,
    ngram_windows: usize,
    retired: bool,
}

impl StatsSnap {
    fn of(stats: &MtpRequestStats) -> Self {
        Self {
            mtp_windows: stats.mtp_windows,
            ar_windows: stats.ar_windows,
            ngram_windows: stats.ngram_mod_windows,
            retired: stats.mtp_retired,
        }
    }
}

/// One `mtp_step` as the decode loop observed it: only quantities the drafter
/// and the bundle already report, plus the HIP call-count deltas around it.
#[derive(Clone, Copy, Debug)]
struct WindowRec {
    /// Generated IDs before the window; its first committed ID is `ids[index]`.
    index: usize,
    position_before: usize,
    position_after: usize,
    /// The draft budget passed to `mtp_step`.
    k_budget: usize,
    drafts: usize,
    accepted: usize,
    committed: usize,
    before: StatsSnap,
    after: StatsSnap,
    counters: Counters,
}

/// Column names of [`WindowRec::row`]; the counter columns follow.
const WINDOW_COLUMNS: [&str; 11] = [
    "index",
    "position_before",
    "position_after",
    "k_budget",
    "drafts",
    "accepted",
    "committed",
    "mtp_windows_after",
    "ar_windows_after",
    "ngram_windows_after",
    "retired_after",
];

impl WindowRec {
    fn row(&self) -> Value {
        let mut row = vec![
            json!(self.index),
            json!(self.position_before),
            json!(self.position_after),
            json!(self.k_budget),
            json!(self.drafts),
            json!(self.accepted),
            json!(self.committed),
            json!(self.after.mtp_windows),
            json!(self.after.ar_windows),
            json!(self.after.ngram_windows),
            json!(self.after.retired),
        ];
        if let Value::Array(counters) = self.counters.to_row() {
            row.extend(counters);
        }
        Value::Array(row)
    }
}

/// A native window: `mtp_windows` moved by one, `ar_windows` did not.
fn is_native_window(window: &WindowRec) -> bool {
    window.after.mtp_windows == window.before.mtp_windows + 1
        && window.after.ar_windows == window.before.ar_windows
}

/// A target-only window after retirement: `ar_windows` moved by one,
/// `mtp_windows` did not.
fn is_ar_window(window: &WindowRec) -> bool {
    window.after.ar_windows == window.before.ar_windows + 1
        && window.after.mtp_windows == window.before.mtp_windows
}

/// The request's first window is the floor's AR calibration token: nothing
/// drafted or accepted, `mtp_windows` 0 -> 1 with `ar_windows` still 0, one
/// position advanced.
fn is_calibration_window(window: &WindowRec) -> bool {
    window.before.mtp_windows == 0
        && window.before.ar_windows == 0
        && !window.before.retired
        && window.drafts == 0
        && window.accepted == 0
        && window.committed == 1
        && is_native_window(window)
        && window.position_after == window.position_before + 1
}

/// One request's floor sequence, read off its window records.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct SeqCoverage {
    /// The first window was the calibration token.
    calibration_ok: bool,
    /// The window at which `mtp_retired` went false -> true.
    retired_at: Option<usize>,
    /// Native windows after calibration that completed before retirement
    /// (the retiring window counts when it was itself native).
    probes_before_retirement: usize,
    /// Generated index of every later target-only window that ran while
    /// `mtp_retired` stayed true.
    post_retirement_ar: Vec<usize>,
    /// Calibration, the whole probe budget, the ordinary false -> true
    /// retirement and a later AR window, all in one request.
    full: bool,
    /// `full`, and an AR window after retirement emitted at index >= `late_from`.
    late_after_retirement: bool,
}

impl SeqCoverage {
    fn to_json(&self) -> Value {
        json!({
            "calibration_ok": self.calibration_ok,
            "retired_at_step": self.retired_at,
            "probes_before_retirement": self.probes_before_retirement,
            "post_retirement_ar_indices": self.post_retirement_ar,
            "full": self.full,
            "late_after_retirement": self.late_after_retirement,
        })
    }
}

fn analyze_windows(windows: &[WindowRec], probe_budget: usize, late_from: usize) -> SeqCoverage {
    let mut coverage = SeqCoverage {
        calibration_ok: windows.first().is_some_and(is_calibration_window),
        ..SeqCoverage::default()
    };
    coverage.retired_at = windows
        .iter()
        .position(|window| !window.before.retired && window.after.retired);
    if let Some(at) = coverage.retired_at {
        let before = windows[1.min(at)..at]
            .iter()
            .filter(|window| is_native_window(window))
            .count();
        coverage.probes_before_retirement = before + usize::from(is_native_window(&windows[at]));
        coverage.post_retirement_ar = windows[at + 1..]
            .iter()
            .filter(|window| {
                is_ar_window(window) && window.before.retired && window.after.retired
            })
            .map(|window| window.index)
            .collect();
        coverage.full = coverage.calibration_ok
            && coverage.probes_before_retirement >= probe_budget
            && !coverage.post_retirement_ar.is_empty();
        coverage.late_after_retirement = coverage.full
            && coverage
                .post_retirement_ar
                .iter()
                .any(|index| *index >= late_from);
    }
    coverage
}

/// Aggregate window / floor evidence over an arm's trials. Every trial is
/// observed; nothing is selected by route history. `samples` keeps the first
/// `sample_limit` trials' decoded text and window rows, plus the first trial
/// with a complete floor sequence so its text can be read.
#[derive(Default)]
struct Evidence {
    sample_limit: usize,
    trials: usize,
    no_window_trials: usize,
    calibration_ok_trials: usize,
    retired_trials: usize,
    full_trials: usize,
    late_full_trials: usize,
    first_full_trial: Option<usize>,
    mtp_windows: usize,
    ar_windows: usize,
    ngram_windows: usize,
    steps: usize,
    drafted: usize,
    accepted: usize,
    post_retirement_ar_steps: usize,
    retire_steps: Vec<usize>,
    counters: [u64; 10],
    per_trial: Vec<Value>,
    samples: Vec<Value>,
    sampled_full: bool,
    texts: String,
}

impl Evidence {
    fn with_samples(sample_limit: usize) -> Self {
        Self {
            sample_limit,
            ..Self::default()
        }
    }

    fn observe(
        &mut self,
        trial: usize,
        seed: u64,
        run: &MtpRun,
        tokenizer: &Tokenizer,
        probe_budget: usize,
        late_from: usize,
    ) {
        let coverage = analyze_windows(&run.windows, probe_budget, late_from);
        self.trials += 1;
        self.no_window_trials += usize::from(run.windows.is_empty());
        self.calibration_ok_trials += usize::from(coverage.calibration_ok);
        if let Some(step) = coverage.retired_at {
            self.retired_trials += 1;
            self.retire_steps.push(step);
        }
        if coverage.full {
            self.full_trials += 1;
            self.first_full_trial.get_or_insert(trial);
            self.late_full_trials += usize::from(coverage.late_after_retirement);
        }
        self.mtp_windows += run.stats.mtp_windows;
        self.ar_windows += run.stats.ar_windows;
        self.ngram_windows += run.stats.ngram_mod_windows;
        self.steps += run.windows.len();
        self.drafted += run.drafted;
        self.accepted += run.accepted;
        self.post_retirement_ar_steps += coverage.post_retirement_ar.len();
        for window in &run.windows {
            for (total, delta) in self.counters.iter_mut().zip(window.counters.to_array()) {
                *total += delta;
            }
        }
        self.per_trial.push(json!({
            "trial": trial,
            "seed": seed,
            "ids": run.ids.len(),
            "steps": run.windows.len(),
            "drafted": run.drafted,
            "accepted": run.accepted,
            "mtp_windows": run.stats.mtp_windows,
            "ar_windows": run.stats.ar_windows,
            "ngram_mod_windows": run.stats.ngram_mod_windows,
            "mtp_retired": run.stats.mtp_retired,
            "coverage": coverage.to_json(),
        }));
        let wanted_full = coverage.full && !self.sampled_full;
        if self.samples.len() < self.sample_limit || wanted_full {
            self.sampled_full |= coverage.full;
            let text = tokenizer.decode(&run.ids);
            self.texts.push_str(&format!(
                "=== trial {trial} seed {seed} ({} ids, floor sequence full: {}) ===\n{text}\n\n",
                run.ids.len(),
                coverage.full
            ));
            let late_ids: Vec<Value> = coverage
                .post_retirement_ar
                .iter()
                .filter(|index| **index >= late_from)
                .filter_map(|index| run.ids.get(*index).map(|id| json!([index, id])))
                .collect();
            self.samples.push(json!({
                "trial": trial,
                "seed": seed,
                "ids": run.ids,
                "text": text,
                "coverage": coverage.to_json(),
                "post_retirement_late_ids": late_ids,
                "windows": run.windows.iter().map(WindowRec::row).collect::<Vec<_>>(),
            }));
        }
    }

    fn to_json(&self, probe_budget: usize, late_from: usize) -> Value {
        let counters: serde_json::Map<String, Value> = hw_identity::COUNTER_COLUMNS
            .iter()
            .zip(self.counters)
            .map(|(name, total)| ((*name).to_string(), json!(total)))
            .collect();
        let mut columns: Vec<&str> = WINDOW_COLUMNS.to_vec();
        columns.extend(hw_identity::COUNTER_COLUMNS);
        json!({
            "probe_budget": probe_budget,
            "late_from": late_from,
            "trials": self.trials,
            "no_window_trials": self.no_window_trials,
            "calibration_ok_trials": self.calibration_ok_trials,
            "retired_trials": self.retired_trials,
            "full_trials": self.full_trials,
            "late_full_trials": self.late_full_trials,
            "first_full_trial": self.first_full_trial,
            "mtp_windows": self.mtp_windows,
            "ar_windows": self.ar_windows,
            "ngram_windows": self.ngram_windows,
            "steps": self.steps,
            "drafted": self.drafted,
            "accepted": self.accepted,
            "post_retirement_ar_steps": self.post_retirement_ar_steps,
            "retire_steps": self.retire_steps,
            "counters": counters,
            "window_columns": columns,
            "per_trial": self.per_trial,
            "samples": self.samples,
        })
    }

    /// `<stem>.evidence.json` and the decoded sample text `<stem>.text.txt`.
    fn write(&self, dir: &Path, stem: &str, probe_budget: usize, late_from: usize) -> Result<(), String> {
        write_json(
            &dir.join(format!("{stem}.evidence.json")),
            &self.to_json(probe_budget, late_from),
        )?;
        let path = dir.join(format!("{stem}.text.txt"));
        fs::write(&path, &self.texts).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// Short (4-token) arm coverage from its evidence: a forced floor-off arm
/// must never touch the floor; the adaptive arm must start every request
/// with the calibration window. Four tokens do NOT exercise retirement; the
/// late case does.
fn short_arm_coverage(arm: &str, evidence: &Value, report: &mut String, coverage: &mut Vec<String>) {
    let count = |key: &str| evidence[key].as_u64().unwrap_or(0) as usize;
    let (trials, ar, retired, calibrated, no_window) = (
        count("trials"),
        count("ar_windows"),
        count("retired_trials"),
        count("calibration_ok_trials"),
        count("no_window_trials"),
    );
    let line = format!(
        "{arm}: trials={trials} steps={} mtp_windows={} ar_windows={ar} ngram_windows={} retired_trials={retired} calibration_first_window_trials={calibrated} no_window_trials={no_window} drafted={} accepted={}\n",
        count("steps"),
        count("mtp_windows"),
        count("ngram_windows"),
        count("drafted"),
        count("accepted"),
    );
    report.push_str(&line);
    if arm_route(arm).is_some() {
        if ar != 0 || retired != 0 || count("mtp_windows") == 0 {
            coverage.push(format!(
                "{line}  a forced floor-off arm must run native windows and never an AR window or retirement\n"
            ));
        }
    } else if calibrated + no_window != trials {
        coverage.push(format!(
            "{line}  the adaptive arm's first window must be the calibration token in every request\n"
        ));
    }
}

// ---------------------------------------------------------------------------
// N-gram composition arms
// ---------------------------------------------------------------------------

/// Per-position marginals and joints of `mtp` against its AR `reference`;
/// lines go to `report`, p-values under `ALPHA` to `failures`.
#[allow(clippy::too_many_arguments)]
fn compare_positions(
    arm: &str,
    reference: &str,
    ar: &[Vec<u32>],
    mtp: &[Vec<u32>],
    marginals: &[usize],
    joints: &[(usize, usize)],
    report: &mut String,
    failures: &mut Vec<String>,
) {
    for &position in marginals {
        let a: Vec<u32> = ar.iter().map(|ids| ids[position]).collect();
        let b: Vec<u32> = mtp.iter().map(|ids| ids[position]).collect();
        let (stat, df, p) = homogeneity(&a, &b);
        let line = format!(
            "{arm} vs {reference}, token {}: chi2={stat:.2} df={df} p={p:.4} (n={}+{})\n",
            position + 1,
            a.len(),
            b.len()
        );
        if p < ALPHA {
            failures.push(line.clone());
        }
        report.push_str(&line);
    }
    for &(first, second) in joints {
        let a: Vec<(u32, u32)> = ar.iter().map(|ids| (ids[first], ids[second])).collect();
        let b: Vec<(u32, u32)> = mtp.iter().map(|ids| (ids[first], ids[second])).collect();
        let (stat, df, p) = homogeneity(&a, &b);
        let line = format!(
            "{arm} vs {reference}, (t{},t{}) joint: chi2={stat:.2} df={df} p={p:.4}\n",
            first + 1,
            second + 1
        );
        if p < ALPHA {
            failures.push(line.clone());
        }
        report.push_str(&line);
    }
}

/// The four-token comparison: every position's marginal and the (t1, t2) joint.
fn compare_to_reference(
    arm: &str,
    reference: &str,
    ar: &[Vec<u32>],
    mtp: &[Vec<u32>],
    report: &mut String,
    failures: &mut Vec<String>,
) {
    let marginals: Vec<usize> = (0..TOKENS).collect();
    compare_positions(arm, reference, ar, mtp, &marginals, &[(0, 1)], report, failures);
}

/// An n-gram arm's recorded config (the `ar` arm uses [`dist_config`]):
/// `copy_lane` is the product switch the arm ran under
/// (`hipfire_config::mtp_ngram_copy_enabled_for_arch` of its GPU).
fn ngram_config(
    arm: &str,
    spec: Option<&NgramArm>,
    identity: &Value,
    sampling: (f32, f32, Option<u32>, f32),
    pen: &Penalties,
    trials: usize,
    copy_lane: bool,
) -> Value {
    let copy = spec.is_none_or(|spec| spec.copy);
    let mut config = config_base("ngram-distribution", arm, pen, Fixture::Prose, identity);
    config["fixture"] = json!(if copy { "copy" } else { "prose" });
    config["temp"] = json!(sampling.0);
    config["top_p"] = json!(sampling.1);
    config["top_k"] = json!(sampling.2);
    config["min_p"] = json!(sampling.3);
    config["trials"] = json!(trials);
    config["tokens"] = json!(TOKENS);
    config["sampled_mode"] = json!("leviathan");
    config["triple"] = json!([NGRAM_TRIPLE.0, NGRAM_TRIPLE.1, NGRAM_TRIPLE.2]);
    config["copy_lane"] = json!(copy_lane);
    config["salt"] = json!(spec.map_or(COPY_AR_SALT, |spec| spec.salt));
    config
}

const COPY_LONG_TOKENS: usize = 48;
/// Per-window draft budget of the long-copy arms, as the product spec loop
/// passes it (`mtp_draft_k(proposal_capacity(), max_emit)`): the drafter's
/// proposal capacity, which follows the n-gram take cap.
const COPY_LONG_BUDGET: &str = "proposal_capacity";
const COPY_LONG_ARM_TEST: &str = "sampled_mtp_copy_long_distribution_arm";
const COPY_LONG_ARMS: [&str; 3] = ["copy_long_ar", "copy_long_batched", "copy_long_adaptive"];
const COPY_LONG_MARGINALS: [usize; 10] = [0, 1, 2, 3, 7, 11, 15, 23, 31, 47];
const COPY_LONG_JOINTS: [(usize, usize); 4] = [(0, 1), (7, 8), (15, 16), (31, 32)];

fn copy_long_salt(arm: &str) -> u64 {
    match arm {
        "copy_long_ar" => 0xC091_A200,
        "copy_long_batched" => 0xC091_BA7C,
        "copy_long_adaptive" => 0xC091_ADA9,
        _ => unreachable!("unknown long-copy arm {arm}"),
    }
}

fn copy_long_config(
    arm: &str,
    identity: &Value,
    settings: &DistSettings,
    arch: &str,
    range: (usize, usize),
) -> Result<Value, String> {
    // The long-copy arms always run the copy lane, whatever the product
    // switch resolves to: its triple, not the arch's.
    let triple = hipfire_config::ngram_mod_copy_triple();
    // Validate the same pool config as the child, without loading a GPU.
    qwen4_ngram_mod_config(triple)?;
    let mut config = config_base(
        "copy-long-distribution", arm, &settings.pen, Fixture::Prose, identity,
    );
    config["fixture"] = json!("copy-long");
    config["instruction"] = json!(COPY_LONG_INSTRUCTION);
    config["passage"] = json!(COPY_LONG_PASSAGE);
    config["primed"] = json!(COPY_PRIMED);
    config["gpu_arch"] = json!(arch);
    config["triple"] = json!([triple.0, triple.1, triple.2]);
    config["copy_lane"] = json!(true);
    config["temp"] = json!(settings.temp);
    config["top_p"] = json!(settings.top_p);
    config["top_k"] = json!(settings.top_k);
    config["min_p"] = json!(settings.min_p);
    config["tokens"] = json!(COPY_LONG_TOKENS);
    config["draft_budget"] = json!(COPY_LONG_BUDGET);
    config["shard"] = json!([range.0, range.1]);
    config["salt"] = json!(copy_long_salt(arm));
    config["sampled_mode"] = json!("leviathan");
    config["marginals"] = json!(COPY_LONG_MARGINALS);
    config["joints"] = json!(COPY_LONG_JOINTS);
    Ok(config)
}

/// The GPU arch an arm's receipt (`<ids stem>.receipt.json`) recorded.
fn receipt_arch(ids_path: &Path) -> Result<String, String> {
    let receipt = read_json(&ids_path.with_extension("receipt.json"))?;
    receipt["gpu"]["arch"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| format!("{}: missing GPU arch receipt", ids_path.display()))
}

/// The receipt supplies the actual loaded GPU's arch, not a guessed default.
/// All other config fields, including its copy triple, are recomputed.
fn check_copy_long_shard(
    dir: &Path,
    arm: &str,
    range: (usize, usize),
    identity: &Value,
    settings: &DistSettings,
) -> Result<(), String> {
    let path = late_file(dir, arm, range, "ids.json");
    let receipt = read_json(&path.with_extension("receipt.json"))?;
    let arch = receipt["gpu"]["arch"]
        .as_str()
        .ok_or_else(|| format!("{}: missing GPU arch receipt", path.display()))?;
    check_config(&path, &copy_long_config(arm, identity, settings, arch, range)?)
}

fn ensure_copy_long_shards(
    model: &Path,
    dir: &Path,
    arm: &str,
    trials: usize,
    settings: &DistSettings,
    identity: &Value,
) -> Result<Vec<(usize, usize)>, String> {
    let shards = late_shards(dir, arm)?;
    for &range in &shards {
        check_copy_long_shard(dir, arm, range, identity, settings)?;
    }
    for gap in shard_gaps(&shards, trials)? {
        eprintln!("{arm}: running missing shard {}..{}", gap.0, gap.1);
        spawn_arm(
            COPY_LONG_ARM_TEST, arm, "leviathan", model, dir, Some(gap),
            &format!("{arm}.{}-{}", gap.0, gap.1),
        )?;
    }
    let shards = late_shards(dir, arm)?;
    for &range in &shards {
        check_copy_long_shard(dir, arm, range, identity, settings)?;
    }
    if !shard_gaps(&shards, trials)?.is_empty() {
        return Err(format!("{arm}: shards {shards:?} do not cover 0..{trials}"));
    }
    Ok(shards)
}

fn copy_long_coverage(arm: &str, parts: &[Value], report: &mut String) -> Vec<String> {
    let total = |key: &str| -> usize {
        parts.iter().map(|part| part[key].as_u64().expect("long-copy stats count") as usize).sum()
    };
    let windows = total("ngram_mod_windows");
    let wide = total("takeover_verified_ge5");
    let line = format!(
        "{arm}: trials={} ngram_mod_windows={windows} drafts={} accepted={} mtp_windows={} takeover_verified_ge5={wide}; unconfirmed_takeovers=not countable from public per-window data (confirmation/head calls are not exposed; HIP call deltas are not an unambiguous discriminator)\n",
        total("trials"), total("ngram_mod_drafts"), total("ngram_mod_accepted"), total("mtp_windows"),
    );
    report.push_str(&line);
    let mut failures = Vec::new();
    if windows == 0 || wide == 0 {
        failures.push(format!(
            "{line}  requires n-gram windows > 0 and at least one seed + >=4-candidate takeover\n"
        ));
    }
    failures
}

/// Keep the old arms' threshold behavior untouched; long-copy explicitly
/// requires every marginal and joint p-value to be strictly above ALPHA.
fn compare_copy_long(
    arm: &str,
    ar: &[Vec<u32>],
    mtp: &[Vec<u32>],
    report: &mut String,
    failures: &mut Vec<String>,
) {
    let positions = COPY_LONG_MARGINALS.iter().map(|&first| (first, None))
        .chain(COPY_LONG_JOINTS.iter().map(|&(first, second)| (first, Some(second))));
    for (first, second) in positions {
        let cells = |samples: &[Vec<u32>]| -> Vec<(u32, Option<u32>)> {
            samples.iter().map(|ids| (ids[first], second.map(|index| ids[index]))).collect()
        };
        let (stat, df, p) = homogeneity(&cells(ar), &cells(mtp));
        let label = second.map_or_else(
            || format!("position {first}"),
            |second| format!("joint ({first},{second})"),
        );
        let line = format!(
            "{arm} vs copy_long_ar, {label}: chi2={stat:.2} df={df} p={p:.6} (n={}+{})\n",
            ar.len(), mtp.len(),
        );
        if !(p > ALPHA) {
            failures.push(line.clone());
        }
        report.push_str(&line);
    }
}

/// Env var naming the MTP arms the long-copy parent compares with
/// `copy_long_ar` (see [`copy_long_compared_arms`]).
const COPY_LONG_ARMS_ENV: &str = "HIPFIRE_SAMPLED_MTP_COPY_LONG_ARMS";

/// MTP arms the long-copy parent runs and compares with `copy_long_ar`:
/// `HIPFIRE_SAMPLED_MTP_COPY_LONG_ARMS`, a comma list of `copy_long_batched`
/// / `copy_long_adaptive` (unset = both). `copy_long_adaptive` is the product
/// route; `copy_long_batched` adds the byte replay of its first seeds. Each
/// arm costs about as much GPU time as the AR reference, so a run may name
/// one; the report states which ran.
fn copy_long_compared_arms() -> Result<Vec<&'static str>, String> {
    let Ok(raw) = std::env::var(COPY_LONG_ARMS_ENV) else {
        return Ok(COPY_LONG_ARMS[1..].to_vec());
    };
    let mut arms = Vec::new();
    for name in raw.split(',').map(str::trim).filter(|name| !name.is_empty()) {
        let arm = COPY_LONG_ARMS[1..]
            .iter()
            .copied()
            .find(|&arm| arm == name)
            .ok_or_else(|| {
                format!(
                    "{COPY_LONG_ARMS_ENV}: unknown arm {name:?} (expected one of {:?})",
                    &COPY_LONG_ARMS[1..]
                )
            })?;
        if !arms.contains(&arm) {
            arms.push(arm);
        }
    }
    if arms.is_empty() {
        return Err(format!("{COPY_LONG_ARMS_ENV}={raw:?} names no arm"));
    }
    Ok(arms)
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4); long: use sampled_mtp_copy_long_distribution_arm shards under fabric holds"]
fn sampled_mtp_copy_long_matches_ar_distribution_on_flash_next() {
    let model = model_path();
    let dir = out_dir();
    let trials = late_trials().unwrap_or_else(|e| panic!("{e}"));
    let settings = DistSettings::from_env().unwrap_or_else(|e| panic!("{e}"));
    let model_id = hw_identity::model_identity(&model, &dir).unwrap_or_else(|e| panic!("{e}"));
    let compared = copy_long_compared_arms().unwrap_or_else(|e| panic!("{e}"));
    let mut samples: HashMap<&str, Vec<Vec<u32>>> = HashMap::new();
    let mut stats: HashMap<&str, Vec<Value>> = HashMap::new();
    for arm in std::iter::once(COPY_LONG_ARMS[0]).chain(compared.iter().copied()) {
        let identity = arm_identity(arm, "leviathan", &model_id, &LATE_CONTROL_ENV);
        let shards = ensure_copy_long_shards(&model, &dir, arm, trials, &settings, &identity)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut all = Vec::with_capacity(trials);
        let mut parts = Vec::new();
        for range in shards {
            let path = late_file(&dir, arm, range, "ids.json");
            let ids: Vec<Vec<u32>> = serde_json::from_value(
                read_json(&path).unwrap_or_else(|e| panic!("{e}")),
            ).expect("long-copy shard ids");
            assert_eq!(ids.len(), range.1 - range.0, "{arm}: shard {range:?} trials");
            assert!(ids.iter().all(|ids| ids.len() == COPY_LONG_TOKENS),
                "{arm}: shard {range:?} generated length");
            all.extend(ids);
            if arm != COPY_LONG_ARMS[0] {
                let part = read_json(&late_file(&dir, arm, range, "stats.json"))
                    .unwrap_or_else(|e| panic!("{e}"));
                assert_eq!(part["trials"].as_u64(), Some((range.1 - range.0) as u64),
                    "{arm}: shard {range:?} stats trials");
                // Require the corresponding per-window evidence, not IDs alone.
                let evidence = read_json(&late_file(&dir, arm, range, "evidence.json"))
                    .unwrap_or_else(|e| panic!("{e}"));
                assert_eq!(evidence["trials"], part["trials"], "{arm}: evidence trials");
                assert_eq!(evidence["ngram_windows"], part["ngram_mod_windows"],
                    "{arm}: evidence n-gram windows");
                parts.push(part);
            }
        }
        assert_eq!(all.len(), trials, "{arm}: aggregate trials");
        samples.insert(arm, all);
        stats.insert(arm, parts);
    }
    let mut report = format!(
        "long-copy case: {trials} trials per arm, {COPY_LONG_TOKENS} IDs, product n-gram triple per recorded GPU arch; marginals {COPY_LONG_MARGINALS:?}, joints {COPY_LONG_JOINTS:?}; MTP arms {compared:?}\n"
    );
    let mut failures = Vec::new();
    let mut coverage = Vec::new();
    for arm in &compared {
        compare_copy_long(
            arm, &samples[COPY_LONG_ARMS[0]], &samples[arm], &mut report, &mut failures,
        );
        coverage.extend(copy_long_coverage(arm, &stats[arm], &mut report));
    }
    eprint!("{report}");
    fs::write(dir.join("copy-long-report.txt"), &report).expect("long-copy report");
    assert!(failures.is_empty() && coverage.is_empty(), "{}",
        failure_message(&dir, &failures, &coverage));
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn sampled_mtp_ngram_matches_ar_distribution_on_flash_next() {
    let model = model_path();
    let dir = out_dir();
    let trials: usize = env_or(TRIALS_ENV, NGRAM_MIN_TRIALS);
    assert!(
        trials >= NGRAM_MIN_TRIALS,
        "{TRIALS_ENV}={trials}: the n-gram arms need at least {NGRAM_MIN_TRIALS} trials"
    );
    let model_id = hw_identity::model_identity(&model, &dir).unwrap_or_else(|e| panic!("{e}"));
    let settings = DistSettings::from_env().unwrap_or_else(|e| panic!("{e}"));
    let sampling = sampling_from_env().unwrap_or_else(|e| panic!("{e}"));
    let mut arms: Vec<(&str, Option<&NgramArm>)> = vec![("ar", None), (COPY_AR_ARM, None)];
    arms.extend(NGRAM_ARMS.iter().map(|arm| (arm.name, Some(arm))));
    let mut samples: HashMap<&str, Vec<Vec<u32>>> = HashMap::new();
    for (name, spec) in arms {
        let ids_path = dir.join(format!("{name}.ids.json"));
        let identity = arm_identity(name, "leviathan", &model_id, &CONTROL_ENV);
        if ids_path.exists() {
            eprintln!("reusing {}", ids_path.display());
        } else {
            let test = if name == "ar" {
                ARM_TEST
            } else {
                NGRAM_ARM_TEST
            };
            spawn_arm(test, name, "leviathan", &model, &dir, None, name)
                .unwrap_or_else(|e| panic!("{e}"));
        }
        // Reused and fresh results alike must carry the config expected now;
        // an n-gram arm's copy-lane mode is the product switch resolved for
        // the GPU its receipt names.
        let expected = if name == "ar" {
            settings.config("ar", &identity)
        } else {
            let arch = receipt_arch(&ids_path).unwrap_or_else(|e| panic!("{e}"));
            let copy_lane =
                hipfire_config::mtp_ngram_copy_enabled_for_arch(hipfire_arch_qwen4::ARCH_ID, &arch);
            ngram_config(name, spec, &identity, sampling, &settings.pen, trials, copy_lane)
        };
        check_config(&ids_path, &expected).unwrap_or_else(|e| panic!("{e}"));
        let ids: Vec<Vec<u32>> =
            serde_json::from_slice(&fs::read(&ids_path).expect("arm ids")).expect("arm ids json");
        assert!(
            ids.len() >= NGRAM_MIN_TRIALS,
            "{name}: {} trials in {}, need at least {NGRAM_MIN_TRIALS}",
            ids.len(),
            ids_path.display()
        );
        samples.insert(name, ids);
    }
    let mut report = String::new();
    let mut failures = Vec::new();
    let mut coverage = Vec::new();
    for arm in &NGRAM_ARMS {
        compare_to_reference(
            arm.name,
            arm.reference,
            &samples[arm.reference],
            &samples[arm.name],
            &mut report,
            &mut failures,
        );
        let stats_path = dir.join(format!("{}.stats.json", arm.name));
        let bytes = fs::read(&stats_path).unwrap_or_else(|e| {
            panic!(
                "{}: {e} (the arm writes its stats with its ids; delete {}.ids.json to rerun it)",
                stats_path.display(),
                arm.name
            )
        });
        let stats: Value = serde_json::from_slice(&bytes).expect("arm stats json");
        let count = |key: &str| stats[key].as_u64().expect("arm stats count") as usize;
        let (windows, drafts, accepted, native) = (
            count("ngram_mod_windows"),
            count("ngram_mod_drafts"),
            count("ngram_mod_accepted"),
            count("mtp_windows"),
        );
        let line = format!(
            "{}: ngram_mod_windows={windows} drafts={drafts} accepted={accepted} mtp_windows={native} (summed over {} trials)\n",
            arm.name,
            count("trials")
        );
        // The forced copy arms must hit (the pool proposes on the first
        // window). An adaptive copy arm spends its first window on the
        // floor's calibration token, so its hits are reported, not required.
        if arm.copy && arm_route(arm.name).is_some() && (windows == 0 || drafts == 0) {
            coverage.push(format!("{line}  the copied-span fixture produced no n-gram hits\n"));
        } else if !arm.copy && native == 0 {
            coverage.push(format!("{line}  no native MTP window ran on the prose prompt\n"));
        }
        report.push_str(&line);
        let evidence = read_json(&dir.join(format!("{}.evidence.json", arm.name)))
            .unwrap_or_else(|e| panic!("{e}"));
        short_arm_coverage(arm.name, &evidence, &mut report, &mut coverage);
    }
    eprint!("{report}");
    fs::write(dir.join("ngram-report.txt"), &report).unwrap();
    assert!(
        failures.is_empty() && coverage.is_empty(),
        "{}",
        failure_message(&dir, &failures, &coverage)
    );
}

/// One n-gram arm in a fresh process; spawned by the test above.
#[test]
#[ignore = "child process of sampled_mtp_ngram_matches_ar_distribution_on_flash_next"]
fn sampled_mtp_ngram_distribution_arm() {
    let Ok(arm) = std::env::var(ARM_ENV) else {
        return;
    };
    if let Err(error) = run_ngram_arm(&arm, &model_path(), &out_dir()) {
        panic!("{arm} arm: {error}");
    }
}

/// The long-copy fixture's instruction (the passage follows it verbatim).
const COPY_LONG_INSTRUCTION: &str =
    "Rewrite the following passage exactly, keeping every word and punctuation mark.\n\n";
/// A passage of well over `COPY_PRIMED + 1 + 63` tokens (about 160) with no
/// repeated five-token run, so the pool's proposal after the primed prefix is
/// the passage's own continuation.
const COPY_LONG_PASSAGE: &str = "Before the harbour woke each morning, the lighthouse keeper climbed the narrow spiral stairs, carrying a tin lantern and a loaf of yesterday's bread. At the top he trimmed the wick, wiped the salt from the great glass lens, and logged the weather in a small blue notebook. Gulls wheeled below him while fishing boats slid out past the breakwater, their engines murmuring across the cold grey water. Around noon a freighter signalled twice, and he answered with the old brass flag lamp his grandfather had once used. In the evening the beam swept slowly over the rocks, the village, and the dark line of the hills, and only when the last boat had returned did he finally sit down to eat.";
/// A takeover window counts as "verified >= 5 rows" when its `[seed,
/// candidates..]` block holds at least this many rows: `drafts + 1`, since a
/// takeover window's `drafts_generated` is its candidate count.
const COPY_LONG_ROWS_MIN: usize = 5;
/// Wide-takeover windows whose full row a shard's stats keep as proof (all
/// of them are counted).
const COPY_LONG_PROOF_ROWS: usize = 8;

/// The long-copy prompt: the rendered rewrite request plus the passage's first
/// `COPY_PRIMED` tokens as the start of the assistant turn. Fails unless the
/// passage tokens occur in the request and the pool (the product `pool`
/// config), seeded from this prompt, proposes the passage's continuation after
/// the passage's token `COPY_PRIMED` (the first window's seed when sampled):
/// at least `COPY_LONG_TOKENS` candidates, each equal to the passage. A pool
/// config with `n_max < COPY_LONG_TOKENS` (a product env override) is rejected.
fn copy_long_fixture(loaded: &Loaded, pool: NgramModConfig) -> Result<Vec<u32>, String> {
    if pool.n_max < COPY_LONG_TOKENS {
        return Err(format!(
            "long-copy needs a pool proposal of {COPY_LONG_TOKENS} tokens, but the product n_max is {}",
            pool.n_max
        ));
    }
    let need = COPY_PRIMED + 1 + pool.n_max;
    let passage = loaded.tokenizer.encode(COPY_LONG_PASSAGE);
    if passage.len() < need {
        return Err(format!(
            "long-copy passage has {} tokens, need {need}",
            passage.len()
        ));
    }
    let mut prompt = loaded.render(&format!("{COPY_LONG_INSTRUCTION}{COPY_LONG_PASSAGE}"))?;
    if !prompt
        .windows(passage.len())
        .any(|window| window == passage.as_slice())
    {
        return Err("the long passage's own tokens do not occur in the rendered request".into());
    }
    prompt.extend_from_slice(&passage[..COPY_PRIMED]);
    let mut context = MtpCopyContext::new(pool).map_err(|e| e.to_string())?;
    context.begin_request(&prompt);
    let seed = &passage[COPY_PRIMED..=COPY_PRIMED];
    let at_least = COPY_LONG_TOKENS;
    match context.propose(seed, pool.n_max) {
        Some(candidates)
            if candidates.len() >= at_least
                && candidates == &passage[COPY_PRIMED + 1..COPY_PRIMED + 1 + candidates.len()] =>
        {
            Ok(prompt)
        }
        other => Err(format!(
            "long-copy fixture: the pool proposed {other:?} after the passage's token {COPY_PRIMED}, expected the passage's next >= {at_least} tokens"
        )),
    }
}

/// A shard's n-gram counters and its verified-row takeover proof. A window is
/// a takeover iff `ngram_mod_windows` moved by one across it (only
/// `finish_takeover` moves it), and it verified >= `COPY_LONG_ROWS_MIN` rows iff
/// `drafts + 1 >= COPY_LONG_ROWS_MIN`. Both long arms run batched takeovers
/// (`require_arm_env` pins `HIPFIRE_MTP_INCREMENTAL` to 0 / unset; only 1
/// selects the interleaved takeover), so all those rows go through one verify
/// forward. Whether a takeover skipped the head's row-0 confirmation is not
/// public per-window data, so it is not counted.
#[derive(Default)]
struct CopyLongStats {
    trials: usize,
    ngram_mod_windows: usize,
    ngram_mod_drafts: usize,
    ngram_mod_accepted: usize,
    mtp_windows: usize,
    takeover_verified_ge5: usize,
    examples: Vec<Value>,
}

impl CopyLongStats {
    fn observe(&mut self, trial: usize, seed: u64, run: &MtpRun) -> Result<(), String> {
        self.trials += 1;
        self.ngram_mod_windows += run.stats.ngram_mod_windows;
        self.ngram_mod_drafts += run.stats.ngram_mod_drafts;
        self.ngram_mod_accepted += run.stats.ngram_mod_accepted;
        self.mtp_windows += run.stats.mtp_windows;
        let mut takeovers = 0usize;
        for window in &run.windows {
            let moved = window
                .after
                .ngram_windows
                .checked_sub(window.before.ngram_windows)
                .ok_or_else(|| format!("trial {trial}: ngram_mod_windows went backwards in a window"))?;
            match moved {
                0 => {}
                1 => {
                    takeovers += 1;
                    if window.drafts + 1 >= COPY_LONG_ROWS_MIN {
                        self.takeover_verified_ge5 += 1;
                        if self.examples.len() < COPY_LONG_PROOF_ROWS {
                            self.examples.push(json!({
                                "trial": trial,
                                "seed": seed,
                                "window": window.row(),
                            }));
                        }
                    }
                }
                _ => {
                    return Err(format!(
                        "trial {trial}: ngram_mod_windows moved by {moved} in one window"
                    ))
                }
            }
        }
        if takeovers != run.stats.ngram_mod_windows {
            return Err(format!(
                "trial {trial}: windows record {takeovers} takeovers, request_stats {}",
                run.stats.ngram_mod_windows
            ));
        }
        Ok(())
    }

    fn to_json(&self, range: (usize, usize)) -> Value {
        let mut columns: Vec<&str> = WINDOW_COLUMNS.to_vec();
        columns.extend(hw_identity::COUNTER_COLUMNS);
        json!({
            "trials": self.trials,
            "shard": [range.0, range.1],
            "ngram_mod_windows": self.ngram_mod_windows,
            "ngram_mod_drafts": self.ngram_mod_drafts,
            "ngram_mod_accepted": self.ngram_mod_accepted,
            "mtp_windows": self.mtp_windows,
            "draft_budget": COPY_LONG_BUDGET,
            "takeover_verified_ge5": self.takeover_verified_ge5,
            "takeover_verified_ge5_predicate": format!(
                "window with ngram_windows_after == ngram_windows_before + 1 (a takeover) and drafts + 1 >= {COPY_LONG_ROWS_MIN} ([seed, candidates..] rows, all verified in one batched forward: both long arms run batched takeovers)"
            ),
            "takeover_verified_ge5_examples": self.examples,
            "window_columns": columns,
            "unconfirmed_takeovers": Value::Null,
            "unconfirmed_takeovers_note": "not countable: the head-confirmation decision is not public per-window data and HIP call deltas do not discriminate it",
        })
    }
}

/// The deterministic columns of every window row (`WINDOW_COLUMNS`, no HIP
/// call counts): what an identical-seed replay must reproduce.
fn window_prefix_rows(run: &MtpRun) -> Vec<Vec<Value>> {
    run.windows
        .iter()
        .map(|window| match window.row() {
            Value::Array(row) => row[..WINDOW_COLUMNS.len()].to_vec(),
            _ => unreachable!("WindowRec::row is an array"),
        })
        .collect()
}

/// One long-copy shard in a fresh process; spawned by the parent test.
#[test]
#[ignore = "child process of sampled_mtp_copy_long_matches_ar_distribution_on_flash_next; runs one shard when run directly"]
fn sampled_mtp_copy_long_distribution_arm() {
    let Ok(arm) = std::env::var(ARM_ENV) else {
        return;
    };
    if let Err(error) = run_copy_long_arm(&arm, &model_path(), &out_dir()) {
        panic!("{arm} arm: {error}");
    }
}

fn run_copy_long_arm(arm: &str, model: &Path, dir: &Path) -> Result<(), String> {
    if !COPY_LONG_ARMS.contains(&arm) {
        return Err(format!("unknown long-copy arm {arm}"));
    }
    require_arm_env(arm, "leviathan")?;
    let settings = DistSettings::from_env()?;
    let trials = late_trials()?;
    let range = shard_range(arm, trials)?;
    let env = own_env(&LATE_CONTROL_ENV);
    let identity = own_identity(model, dir, &LATE_CONTROL_ENV)?;
    let ids_path = late_file(dir, arm, range, "ids.json");
    if ids_path.exists() {
        check_copy_long_shard(dir, arm, range, &identity, &settings)?;
        println!(
            "ARM {arm} shard {}..{} already complete under this config",
            range.0, range.1
        );
        return Ok(());
    }
    let mut loaded = load(model)?;
    // The long-copy arms always run the copy lane (see `copy_long_config`).
    let triple = hipfire_config::ngram_mod_copy_triple();
    let pool_config = qwen4_ngram_mod_config(triple)?;
    let config = copy_long_config(arm, &identity, &settings, &loaded.gpu.arch, range)?;
    let prompt = copy_long_fixture(&loaded, pool_config)?;
    let DistSettings {
        temp,
        top_p,
        top_k,
        min_p,
        pen,
        ..
    } = settings;
    let eos = loaded.bundle.config.eos_token_id;
    let budget = floor_probe_budget()?;
    println!(
        "ARM {arm} arch={} prompt_tokens={} shard={}..{} tokens={COPY_LONG_TOKENS} temp={temp} top_p={top_p} top_k={top_k:?} min_p={min_p} triple={triple:?} draft_budget={COPY_LONG_BUDGET} state={} route_class={} probe_budget={budget}",
        loaded.gpu.arch,
        prompt.len(),
        range.0,
        range.1,
        loaded.state,
        hw_identity::route_class(&env)
    );
    println!("ARM {arm} config {config}");
    let started = std::time::Instant::now();
    let (start, end) = range;
    let mut samples: Vec<Vec<u32>> = Vec::with_capacity(end - start);
    let progress = |trial: usize| {
        if (trial + 1 - start) % 25 == 0 {
            println!(
                "ARM {arm} {}/{} trials in {:.0}s",
                trial + 1 - start,
                end - start,
                started.elapsed().as_secs_f64()
            );
        }
    };
    let salt = copy_long_salt(arm);
    if arm == COPY_LONG_ARMS[0] {
        let logits = loaded
            .gpu
            .zeros(&[loaded.bundle.config.vocab_size], DType::F32)
            .map_err(|e| e.to_string())?;
        let cfg = ar_sampler(temp, top_p, top_k, min_p, &pen);
        for trial in start..end {
            let seed = trial_seed(salt, trial) as u32;
            samples.push(loaded.ar_ids(&logits, &prompt, &cfg, seed, COPY_LONG_TOKENS)?);
            progress(trial);
        }
        loaded.gpu.free_tensor(logits).map_err(|e| e.to_string())?;
    } else {
        let mut drafter = new_drafter_with(&mut loaded, Some(pool_config), true)?;
        let mut evidence = Evidence::with_samples(EVIDENCE_SAMPLES);
        let mut stats = CopyLongStats::default();
        // Only a forced floor-off route replays byte for byte; the adaptive
        // route's choices depend on wall-clock prices and are never replayed.
        let replays = hw_identity::forced_floor_off(&env);
        let mut firsts: Vec<(usize, u64, MtpRun)> = Vec::new();
        let run = |loaded: &mut Loaded, drafter: &mut Qwen4MtpDrafter, seed: u64| {
            let cfg = spec_request_ngram(temp, top_p, top_k, min_p, seed, &pen);
            loaded.mtp_request_with_budget(
                drafter,
                &prompt,
                cfg,
                COPY_LONG_TOKENS,
                DraftBudget::ProposalCapacity,
            )
        };
        for trial in start..end {
            let seed = trial_seed(salt, trial);
            let result = run(&mut loaded, &mut drafter, seed)?;
            if !result.stats.mtp_ngram {
                return Err(format!("trial {trial}: the n-gram pool was not armed"));
            }
            stats.observe(trial, seed, &result)?;
            evidence.observe(trial, seed, &result, &loaded.tokenizer, budget, LATE_MARGINALS[0]);
            if replays && trial < REPLAYS {
                samples.push(result.ids.clone());
                firsts.push((trial, seed, result));
            } else {
                samples.push(result.ids);
            }
            progress(trial);
        }
        for (trial, seed, first) in &firsts {
            let replay = run(&mut loaded, &mut drafter, *seed)?;
            if replay.ids != first.ids
                || replay.stats != first.stats
                || window_prefix_rows(&replay) != window_prefix_rows(first)
            {
                return Err(format!(
                    "trial {trial} replayed {:?} / {:?} / {:?}, first run {:?} / {:?} / {:?}",
                    replay.ids,
                    replay.stats,
                    window_prefix_rows(&replay),
                    first.ids,
                    first.stats,
                    window_prefix_rows(first)
                ));
            }
        }
        if replays {
            println!(
                "ARM {arm} replayed {} seeds identically (ids, request stats, window rows)",
                firsts.len()
            );
        } else {
            println!(
                "ARM {arm} replay not asserted (route class {})",
                hw_identity::route_class(&env)
            );
        }
        println!(
            "ARM {arm} ngram_mod_windows={} drafts={} accepted={} mtp_windows={} takeover_verified_ge5={} unconfirmed_takeovers=unknown",
            stats.ngram_mod_windows,
            stats.ngram_mod_drafts,
            stats.ngram_mod_accepted,
            stats.mtp_windows,
            stats.takeover_verified_ge5
        );
        write_json(&late_file(dir, arm, range, "stats.json"), &stats.to_json(range))?;
        evidence.write(
            dir,
            &format!("{arm}.{}-{}", range.0, range.1),
            budget,
            LATE_MARGINALS[0],
        )?;
        Box::new(drafter).mtp_free(&mut loaded.gpu);
    }
    // Positions after an EOS read as EOS; every trial is kept.
    for ids in &mut samples {
        ids.resize(COPY_LONG_TOKENS, eos);
    }
    println!(
        "ARM {arm} shard {}..{} in {:.1}s; first sample: {:?}",
        range.0,
        range.1,
        started.elapsed().as_secs_f64(),
        loaded.tokenizer.decode(&samples[0])
    );
    fs::write(
        &ids_path,
        serde_json::to_vec(&samples).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    write_config(&ids_path, &config)?;
    write_receipt(&ids_path, &hw_identity::receipt(&loaded.gpu, model))?;
    loaded
        .bundle
        .free_gpu(&mut loaded.gpu)
        .map_err(|e| e.to_string())
}

fn sampling_from_env() -> Result<(f32, f32, Option<u32>, f32), String> {
    let temp: f32 = env_or(TEMP_ENV, 1.0);
    let top_p: f32 = env_or(TOP_P_ENV, 0.95);
    let top_k: Option<u32> = std::env::var(TOP_K_ENV)
        .ok()
        .map(|v| v.parse().map_err(|e| format!("{TOP_K_ENV}={v}: {e}")))
        .transpose()?;
    let min_p: f32 = env_or(MIN_P_ENV, 0.0);
    Ok((temp, top_p, top_k, min_p))
}

/// The copied-span fixture's prompt: the rendered rewrite request plus the
/// passage's first `COPY_PRIMED` tokens as the start of the assistant turn.
/// Fails unless the passage tokens occur in the request and the pool of the
/// arm's lane (`MtpCopyContext` with `copy_lane`, else `MtpNgramContext`),
/// seeded from this prompt, proposes the passage's next `n_max` tokens after
/// the passage token `COPY_PRIMED` (the first window's seed when sampled).
fn copy_fixture(loaded: &Loaded, copy_lane: bool) -> Result<Vec<u32>, String> {
    let (_, n_min, n_max) = NGRAM_TRIPLE;
    let passage = loaded.tokenizer.encode(COPY_PASSAGE);
    if passage.len() < COPY_PRIMED + 1 + n_max.max(n_min) {
        return Err(format!(
            "copy passage has {} tokens, need {}",
            passage.len(),
            COPY_PRIMED + 1 + n_max.max(n_min)
        ));
    }
    let mut prompt = loaded.render(&format!("{COPY_INSTRUCTION}{COPY_PASSAGE}"))?;
    if !prompt
        .windows(passage.len())
        .any(|window| window == passage.as_slice())
    {
        return Err("the passage's own tokens do not occur in the rendered request".into());
    }
    prompt.extend_from_slice(&passage[..COPY_PRIMED]);
    let pool_config = qwen4_ngram_mod_config(NGRAM_TRIPLE)?;
    let seed = &passage[COPY_PRIMED..=COPY_PRIMED];
    let want = &passage[COPY_PRIMED + 1..COPY_PRIMED + 1 + n_max];
    let proposed: Option<Vec<u32>> = if copy_lane {
        let mut pool = MtpCopyContext::new(pool_config).map_err(|e| e.to_string())?;
        pool.begin_request(&prompt);
        pool.propose(seed, n_max).map(<[u32]>::to_vec)
    } else {
        let mut pool = MtpNgramContext::new(pool_config).map_err(|e| e.to_string())?;
        pool.begin_request(&prompt);
        pool.propose(seed, n_max).map(<[u32]>::to_vec)
    };
    match proposed {
        Some(candidates) if candidates == want => Ok(prompt),
        other => Err(format!(
            "copy fixture: the pool proposed {other:?} after the passage's token {COPY_PRIMED}, expected {want:?}"
        )),
    }
}

fn run_ngram_arm(arm: &str, model: &Path, dir: &Path) -> Result<(), String> {
    let spec = NGRAM_ARMS.iter().find(|spec| spec.name == arm);
    if spec.is_none() && arm != COPY_AR_ARM {
        return Err(format!("unknown n-gram arm {arm}"));
    }
    require_arm_env(arm, "leviathan")?;
    let trials: usize = env_or(TRIALS_ENV, NGRAM_MIN_TRIALS);
    if trials < NGRAM_MIN_TRIALS {
        return Err(format!(
            "{TRIALS_ENV}={trials}: the n-gram arms need at least {NGRAM_MIN_TRIALS} trials"
        ));
    }
    let sampling = sampling_from_env()?;
    let (temp, top_p, top_k, min_p) = sampling;
    // The env penalties reach both the AR reference and the n-gram arms.
    let pen = Penalties::from_env()?;
    let env = own_env(&CONTROL_ENV);
    let identity = own_identity(model, dir, &CONTROL_ENV)?;
    let ids_path = dir.join(format!("{arm}.ids.json"));
    let mut loaded = load(model)?;
    // The n-gram arms follow the product switch of the loaded GPU.
    let copy_lane = hipfire_config::mtp_ngram_copy_enabled_for_arch(
        hipfire_arch_qwen4::ARCH_ID,
        &loaded.gpu.arch,
    );
    let config = ngram_config(arm, spec, &identity, sampling, &pen, trials, copy_lane);
    if ids_path.exists() {
        check_config(&ids_path, &config)?;
    }
    let copy = spec.is_none_or(|spec| spec.copy);
    let prompt = if copy {
        copy_fixture(&loaded, copy_lane)?
    } else {
        loaded.render(PROSE)?
    };
    let eos = loaded.bundle.config.eos_token_id;
    println!(
        "ARM {arm} arch={} prompt_tokens={} trials={trials} temp={temp} top_p={top_p} top_k={top_k:?} min_p={min_p} triple={NGRAM_TRIPLE:?} copy_lane={copy_lane} state={} route_class={}",
        loaded.gpu.arch,
        prompt.len(),
        loaded.state,
        hw_identity::route_class(&env)
    );
    println!("ARM {arm} config {config}");
    let started = std::time::Instant::now();
    let mut samples: Vec<Vec<u32>> = Vec::with_capacity(trials);
    match spec {
        None => {
            let logits = loaded
                .gpu
                .zeros(&[loaded.bundle.config.vocab_size], DType::F32)
                .map_err(|e| e.to_string())?;
            let cfg = ar_sampler(temp, top_p, top_k, min_p, &pen);
            for trial in 0..trials {
                let seed = trial_seed(COPY_AR_SALT, trial) as u32;
                samples.push(loaded.ar_ids(&logits, &prompt, &cfg, seed, TOKENS)?);
            }
            loaded.gpu.free_tensor(logits).map_err(|e| e.to_string())?;
        }
        Some(spec) => {
            let pool_config = qwen4_ngram_mod_config(NGRAM_TRIPLE)?;
            let mut drafter = new_drafter_with(&mut loaded, Some(pool_config), copy_lane)?;
            let budget = floor_probe_budget()?;
            let mut evidence = Evidence::with_samples(EVIDENCE_SAMPLES);
            let mut run = |loaded: &mut Loaded, trial: usize| {
                let seed = trial_seed(spec.salt, trial);
                let cfg = spec_request_ngram(temp, top_p, top_k, min_p, seed, &pen);
                loaded
                    .mtp_request(&mut drafter, &prompt, cfg, TOKENS)
                    .map(|result| (result, seed))
            };
            let (mut windows, mut drafts, mut accepted, mut native) = (0usize, 0usize, 0usize, 0usize);
            let mut first_stats: Vec<MtpRequestStats> = Vec::new();
            for trial in 0..trials {
                let (result, seed) = run(&mut loaded, trial)?;
                let stats = result.stats;
                if !stats.mtp_ngram {
                    return Err(format!("trial {trial}: the n-gram pool was not armed"));
                }
                windows += stats.ngram_mod_windows;
                drafts += stats.ngram_mod_drafts;
                accepted += stats.ngram_mod_accepted;
                native += stats.mtp_windows;
                if trial < REPLAYS {
                    first_stats.push(stats);
                }
                evidence.observe(trial, seed, &result, &loaded.tokenizer, budget, LATE_MARGINALS[0]);
                samples.push(result.ids);
            }
            // Only a forced floor-off route replays byte for byte; the
            // adaptive route's choices depend on wall-clock prices.
            if hw_identity::forced_floor_off(&env) {
                for trial in 0..REPLAYS.min(trials) {
                    let (replay, _) = run(&mut loaded, trial)?;
                    if replay.ids != samples[trial] || replay.stats != first_stats[trial] {
                        return Err(format!(
                            "trial {trial} replayed {:?} / {:?}, first run {:?} / {:?}",
                            replay.ids, replay.stats, samples[trial], first_stats[trial]
                        ));
                    }
                }
                println!(
                    "ARM {arm} replayed {} seeds identically; ngram_mod_windows={windows} drafts={drafts} accepted={accepted} mtp_windows={native}",
                    REPLAYS.min(trials)
                );
            } else {
                println!(
                    "ARM {arm} replay not asserted (route class {}); ngram_mod_windows={windows} drafts={drafts} accepted={accepted} mtp_windows={native}",
                    hw_identity::route_class(&env)
                );
            }
            if spec.copy && arm_route(arm).is_some() && (windows == 0 || drafts == 0) {
                return Err(format!(
                    "the copied-span fixture produced no n-gram hits in {trials} trials (ngram_mod_windows={windows}, drafts={drafts})"
                ));
            }
            if !spec.copy && native == 0 {
                return Err(format!(
                    "no native MTP window ran in {trials} prose trials (ngram_mod_windows={windows})"
                ));
            }
            write_json(
                &dir.join(format!("{arm}.stats.json")),
                &json!({
                    "trials": trials,
                    "ngram_mod_windows": windows,
                    "ngram_mod_drafts": drafts,
                    "ngram_mod_accepted": accepted,
                    "mtp_windows": native,
                }),
            )?;
            evidence.write(dir, arm, budget, LATE_MARGINALS[0])?;
            Box::new(drafter).mtp_free(&mut loaded.gpu);
        }
    }
    // Positions after an EOS read as EOS.
    for ids in &mut samples {
        ids.resize(TOKENS, eos);
    }
    println!(
        "ARM {arm} {trials} trials in {:.1}s; first sample: {:?}",
        started.elapsed().as_secs_f64(),
        loaded.tokenizer.decode(&samples[0])
    );
    fs::write(
        &ids_path,
        serde_json::to_vec(&samples).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    write_config(&ids_path, &config)?;
    write_receipt(&ids_path, &hw_identity::receipt(&loaded.gpu, model))?;
    loaded
        .bundle
        .free_gpu(&mut loaded.gpu)
        .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Adaptive late-position case
// ---------------------------------------------------------------------------
//
// A long request under the default (adaptive) route: calibration token, the
// probe windows, the floor's ordinary retirement decision and the target-only
// windows after it, compared with AR at late positions. The short arms above
// cannot reach this: four output tokens end inside calibration plus one
// probe window.

const LATE_AR_ARM: &str = "late_ar";
const LATE_ADAPTIVE_ARM: &str = "late_adaptive";
const LATE_PILOT_ARM: &str = "late_pilot";
const LATE_AR_SALT: u64 = 0x1A7E_A200;
const LATE_MTP_SALT: u64 = 0x1A7E_5EC0;
const LATE_PILOT_SALT: u64 = 0x1A7E_9170;
/// Generated IDs per trial (`HIPFIRE_SAMPLED_MTP_LATE_TOKENS` may only raise it).
const LATE_TOKENS_DEFAULT: usize = 128;
/// Fewest independent trials per compared arm.
const LATE_MIN_TRIALS: usize = 2000;
/// Zero-based late marginals and adjacent joints, next to the early marginals.
const LATE_MARGINALS: [usize; 4] = [31, 63, 95, 127];
const LATE_JOINTS: [(usize, usize); 4] = [(31, 32), (63, 64), (95, 96), (126, 127)];
const LATE_EARLY_MARGINALS: [usize; 4] = [0, 1, 2, 3];
/// The declared coverage pilot: a small seed cohort that shows whether the
/// fixture and length exercise the floor. It carries no acceptance statistics.
const PILOT_TRIALS: usize = 24;
/// Trials whose decoded text and window rows an arm keeps.
const EVIDENCE_SAMPLES: usize = 8;
/// The late arms' parent / child control variables (`TRIALS` is excluded so
/// shards run under any total agree).
const LATE_CONTROL_ENV: [&str; 5] = [MODEL_ENV, OUT_ENV, ARM_ENV, SHARD_ENV, TRIALS_ENV];

fn late_salt(arm: &str) -> u64 {
    match arm {
        LATE_AR_ARM => LATE_AR_SALT,
        LATE_ADAPTIVE_ARM => LATE_MTP_SALT,
        LATE_PILOT_ARM => LATE_PILOT_SALT,
        _ => 0,
    }
}

fn late_tokens() -> Result<usize, String> {
    let tokens: usize = strict_env(LATE_TOKENS_ENV, LATE_TOKENS_DEFAULT)?;
    if tokens < LATE_TOKENS_DEFAULT {
        return Err(format!(
            "{LATE_TOKENS_ENV}={tokens}: the late indices need at least {LATE_TOKENS_DEFAULT} generated IDs"
        ));
    }
    Ok(tokens)
}

fn late_trials() -> Result<usize, String> {
    let trials: usize = strict_env(TRIALS_ENV, 2000)?;
    if trials < LATE_MIN_TRIALS {
        return Err(format!(
            "{TRIALS_ENV}={trials}: the late case needs at least {LATE_MIN_TRIALS} trials per arm"
        ));
    }
    Ok(trials)
}

/// `<arm>.<start>-<end>.<ext>` in `dir`.
fn late_file(dir: &Path, arm: &str, range: (usize, usize), ext: &str) -> PathBuf {
    dir.join(format!("{arm}.{}-{}.{ext}", range.0, range.1))
}

/// The trial range an arm process runs: `HIPFIRE_SAMPLED_MTP_SHARD=START..END`
/// or all of `0..total` (`total` is the pilot's cohort or the trial count).
fn shard_range(arm: &str, trials: usize) -> Result<(usize, usize), String> {
    let total = if arm == LATE_PILOT_ARM {
        PILOT_TRIALS
    } else {
        trials
    };
    let Ok(value) = std::env::var(SHARD_ENV) else {
        return Ok((0, total));
    };
    let (start, end) = value
        .split_once("..")
        .ok_or_else(|| format!("{SHARD_ENV}={value}: expected START..END"))?;
    let parse = |text: &str| {
        text.trim()
            .parse::<usize>()
            .map_err(|e| format!("{SHARD_ENV}={value}: {e}"))
    };
    let (start, end) = (parse(start)?, parse(end)?);
    if start >= end || end > total {
        return Err(format!(
            "{SHARD_ENV}={value}: the range must be non-empty and within 0..{total}"
        ));
    }
    Ok((start, end))
}

fn late_config(
    arm: &str,
    identity: &Value,
    settings: &DistSettings,
    tokens: usize,
    range: (usize, usize),
) -> Value {
    let mut config = config_base(
        "late-distribution",
        arm,
        &settings.pen,
        settings.fixture,
        identity,
    );
    config["temp"] = json!(settings.temp);
    config["top_p"] = json!(settings.top_p);
    config["top_k"] = json!(settings.top_k);
    config["min_p"] = json!(settings.min_p);
    config["tokens"] = json!(tokens);
    config["shard"] = json!([range.0, range.1]);
    config["salt"] = json!(late_salt(arm));
    config["sampled_mode"] = json!("leviathan");
    config["probe_budget"] = json!(floor_probe_budget().ok());
    config["late_marginals"] = json!(LATE_MARGINALS);
    config["late_joints"] = json!(LATE_JOINTS);
    config
}

/// Sorted `(start, end)` of every `<arm>.<start>-<end>.ids.json` in `dir`.
fn late_shards(dir: &Path, arm: &str) -> Result<Vec<(usize, usize)>, String> {
    let prefix = format!("{arm}.");
    let mut found = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let name = entry
            .map_err(|e| e.to_string())?
            .file_name()
            .to_string_lossy()
            .into_owned();
        let Some(rest) = name
            .strip_prefix(&prefix)
            .and_then(|rest| rest.strip_suffix(".ids.json"))
        else {
            continue;
        };
        let Some((start, end)) = rest.split_once('-') else {
            continue;
        };
        if let (Ok(start), Ok(end)) = (start.parse::<usize>(), end.parse::<usize>()) {
            found.push((start, end));
        }
    }
    found.sort_unstable();
    Ok(found)
}

/// Uncovered parts of `0..total` given sorted shards. An overlap would count
/// a seed twice and a shard beyond `total` is not part of this cohort: both
/// are errors, not something to patch around.
fn shard_gaps(shards: &[(usize, usize)], total: usize) -> Result<Vec<(usize, usize)>, String> {
    let mut gaps = Vec::new();
    let mut next = 0;
    for &(start, end) in shards {
        if start >= end || end > total {
            return Err(format!("shard {start}-{end} lies outside 0..{total}"));
        }
        if start < next {
            return Err(format!(
                "shard {start}-{end} overlaps an earlier shard ending at {next}: a seed would be counted twice"
            ));
        }
        if start > next {
            gaps.push((next, start));
        }
        next = end;
    }
    if next < total {
        gaps.push((next, total));
    }
    Ok(gaps)
}

/// Every shard of `arm` covering exactly `0..total`, each verified against
/// its recorded config; missing ranges are run as fresh child processes
/// (one per gap: use [`SHARD_ENV`] on `sampled_mtp_late_arm` to run them
/// separately, inside one fabric hold each).
#[allow(clippy::too_many_arguments)]
fn ensure_late_shards(
    model: &Path,
    dir: &Path,
    arm: &str,
    total: usize,
    tokens: usize,
    settings: &DistSettings,
    identity: &Value,
) -> Result<Vec<(usize, usize)>, String> {
    let verify = |shards: &[(usize, usize)]| -> Result<(), String> {
        for &range in shards {
            check_config(
                &late_file(dir, arm, range, "ids.json"),
                &late_config(arm, identity, settings, tokens, range),
            )?;
        }
        Ok(())
    };
    let shards = late_shards(dir, arm)?;
    verify(&shards)?;
    for gap in shard_gaps(&shards, total)? {
        eprintln!("{arm}: running missing shard {}..{}", gap.0, gap.1);
        spawn_arm(
            LATE_ARM_TEST,
            arm,
            "leviathan",
            model,
            dir,
            Some(gap),
            &format!("{arm}.{}-{}", gap.0, gap.1),
        )?;
    }
    let shards = late_shards(dir, arm)?;
    verify(&shards)?;
    if !shard_gaps(&shards, total)?.is_empty() {
        return Err(format!("{arm}: shards {shards:?} do not cover 0..{total}"));
    }
    Ok(shards)
}

/// Aggregate coverage of the adaptive arm from its shards' evidence, as
/// report lines plus coverage failures. Requires every request to start with
/// calibration and at least one request to show the whole floor sequence:
/// calibration, the probe budget, natural retirement, then a target-only
/// window after it that emitted at a late index. It does not require every
/// request to retire (routes are timing-dependent) and never forces one.
fn adaptive_coverage(parts: &[Value], budget: usize, report: &mut String) -> Vec<String> {
    let total = |key: &str| -> usize {
        parts
            .iter()
            .map(|part| part[key].as_u64().unwrap_or(0) as usize)
            .sum()
    };
    let (trials, calibrated, no_window, retired, full, late_full, steps) = (
        total("trials"),
        total("calibration_ok_trials"),
        total("no_window_trials"),
        total("retired_trials"),
        total("full_trials"),
        total("late_full_trials"),
        total("steps"),
    );
    let first_full = parts
        .iter()
        .filter_map(|part| part["first_full_trial"].as_u64())
        .min();
    let mut retire_steps: Vec<u64> = parts
        .iter()
        .flat_map(|part| part["retire_steps"].as_array().cloned().unwrap_or_default())
        .filter_map(|step| step.as_u64())
        .collect();
    retire_steps.sort_unstable();
    let median = retire_steps.get(retire_steps.len() / 2).copied();
    report.push_str(&format!(
        "adaptive floor coverage: trials={trials} probe_budget={budget} calibration_first_window_trials={calibrated} no_window_trials={no_window} retired_trials={retired} full_sequence_trials={full} late_full_sequence_trials={late_full} first_full_trial={first_full:?} retire_step(min/median/max)={:?}/{median:?}/{:?} mtp_windows={} ar_windows={} ngram_windows={} post_retirement_ar_steps={} drafted={} accepted={} steps={steps}\n",
        retire_steps.first(),
        retire_steps.last(),
        total("mtp_windows"),
        total("ar_windows"),
        total("ngram_windows"),
        total("post_retirement_ar_steps"),
        total("drafted"),
        total("accepted"),
    ));
    if steps > 0 {
        let mut means = String::from("per-window HIP call means (hip_bridge::launch_counters):");
        for name in hw_identity::COUNTER_COLUMNS {
            let sum: u64 = parts
                .iter()
                .map(|part| part["counters"][name].as_u64().unwrap_or(0))
                .sum();
            means.push_str(&format!(" {name}={:.2}", sum as f64 / steps as f64));
        }
        report.push_str(&means);
        report.push('\n');
    }
    let mut failures = Vec::new();
    if calibrated + no_window != trials {
        failures.push(format!(
            "only {calibrated} of {trials} requests began with the calibration window\n"
        ));
    }
    if retired == 0 {
        failures.push(format!(
            "no request retired naturally in {trials} trials: the fixture/length does not exercise retirement (declare a longer {LATE_TOKENS_ENV} or a different committed fixture and restart the whole cohort; do not force it)\n"
        ));
    }
    if full == 0 {
        failures.push(format!(
            "no request completed calibration -> {budget} probe windows -> retirement -> a later target-only window\n"
        ));
    } else if late_full == 0 {
        failures.push(format!(
            "no complete floor sequence emitted a post-retirement target-only ID at index >= {}\n",
            LATE_MARGINALS[0]
        ));
    }
    failures
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn sampled_mtp_adaptive_floor_coverage_pilot_on_flash_next() {
    let model = model_path();
    let dir = out_dir();
    let tokens = late_tokens().unwrap_or_else(|e| panic!("{e}"));
    let settings = DistSettings::from_env().unwrap_or_else(|e| panic!("{e}"));
    let budget = floor_probe_budget().unwrap_or_else(|e| panic!("{e}"));
    let model_id = hw_identity::model_identity(&model, &dir).unwrap_or_else(|e| panic!("{e}"));
    let range = (0, PILOT_TRIALS);
    let identity = arm_identity(LATE_PILOT_ARM, "leviathan", &model_id, &LATE_CONTROL_ENV);
    let expected = late_config(LATE_PILOT_ARM, &identity, &settings, tokens, range);
    let ids_path = late_file(&dir, LATE_PILOT_ARM, range, "ids.json");
    if ids_path.exists() {
        eprintln!("reusing {}", ids_path.display());
    } else {
        spawn_arm(
            LATE_ARM_TEST,
            LATE_PILOT_ARM,
            "leviathan",
            &model,
            &dir,
            Some(range),
            &format!("{LATE_PILOT_ARM}.{}-{}", range.0, range.1),
        )
        .unwrap_or_else(|e| panic!("{e}"));
    }
    check_config(&ids_path, &expected).unwrap_or_else(|e| panic!("{e}"));
    let evidence = read_json(&late_file(&dir, LATE_PILOT_ARM, range, "evidence.json"))
        .unwrap_or_else(|e| panic!("{e}"));
    let mut report = format!(
        "coverage pilot (declared cohort of {PILOT_TRIALS} seeds, {tokens} IDs, fixture {}; not acceptance statistics)\n",
        settings.fixture.name()
    );
    let failures = adaptive_coverage(std::slice::from_ref(&evidence), budget, &mut report);
    for trial in evidence["per_trial"].as_array().into_iter().flatten() {
        report.push_str(&format!("{trial}\n"));
    }
    eprint!("{report}");
    fs::write(dir.join("late-pilot-report.txt"), &report).unwrap();
    assert!(
        failures.is_empty(),
        "{}",
        failure_message(&dir, &[], &failures)
    );
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4); long: use sampled_mtp_late_arm shards under fabric holds"]
fn sampled_mtp_adaptive_late_matches_ar_distribution_on_flash_next() {
    let model = model_path();
    let dir = out_dir();
    let trials = late_trials().unwrap_or_else(|e| panic!("{e}"));
    let tokens = late_tokens().unwrap_or_else(|e| panic!("{e}"));
    let settings = DistSettings::from_env().unwrap_or_else(|e| panic!("{e}"));
    let budget = floor_probe_budget().unwrap_or_else(|e| panic!("{e}"));
    let model_id = hw_identity::model_identity(&model, &dir).unwrap_or_else(|e| panic!("{e}"));
    let mut ids: HashMap<&str, Vec<Vec<u32>>> = HashMap::new();
    let mut adaptive_evidence: Vec<Value> = Vec::new();
    for arm in [LATE_AR_ARM, LATE_ADAPTIVE_ARM] {
        let identity = arm_identity(arm, "leviathan", &model_id, &LATE_CONTROL_ENV);
        let shards = ensure_late_shards(&model, &dir, arm, trials, tokens, &settings, &identity)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut all: Vec<Vec<u32>> = Vec::with_capacity(trials);
        for range in shards {
            let part: Vec<Vec<u32>> = serde_json::from_value(
                read_json(&late_file(&dir, arm, range, "ids.json"))
                    .unwrap_or_else(|e| panic!("{e}")),
            )
            .expect("shard ids json");
            assert_eq!(part.len(), range.1 - range.0, "{arm} shard {range:?} trial count");
            assert!(
                part.iter().all(|ids| ids.len() == tokens),
                "{arm} shard {range:?} id count"
            );
            all.extend(part);
            if arm == LATE_ADAPTIVE_ARM {
                adaptive_evidence.push(
                    read_json(&late_file(&dir, arm, range, "evidence.json"))
                        .unwrap_or_else(|e| panic!("{e}")),
                );
            }
        }
        assert_eq!(all.len(), trials, "{arm}: trial count");
        ids.insert(arm, all);
    }
    let mut report = format!(
        "late-position case: {trials} trials per arm, {tokens} IDs, fixture {}, early marginals {LATE_EARLY_MARGINALS:?}, late marginals {LATE_MARGINALS:?}, joints {LATE_JOINTS:?}\n",
        settings.fixture.name()
    );
    let mut marginals = LATE_EARLY_MARGINALS.to_vec();
    marginals.extend(LATE_MARGINALS);
    let mut joints = vec![(0, 1)];
    joints.extend(LATE_JOINTS);
    let mut distribution = Vec::new();
    compare_positions(
        LATE_ADAPTIVE_ARM,
        LATE_AR_ARM,
        &ids[LATE_AR_ARM],
        &ids[LATE_ADAPTIVE_ARM],
        &marginals,
        &joints,
        &mut report,
        &mut distribution,
    );
    let coverage = adaptive_coverage(&adaptive_evidence, budget, &mut report);
    eprint!("{report}");
    fs::write(dir.join("late-report.txt"), &report).unwrap();
    assert!(
        distribution.is_empty() && coverage.is_empty(),
        "{}",
        failure_message(&dir, &distribution, &coverage)
    );
}

/// One late arm (`late_ar`, `late_adaptive`, `late_pilot`) over a trial range
/// in a fresh process. Spawned by the two tests above, and runnable directly
/// to compute one shard inside a time-boxed hold: set
/// `HIPFIRE_SAMPLED_MTP_ARM`, `HIPFIRE_SAMPLED_MTP_SHARD=START..END` and the
/// arm's environment (`require_arm_env` names anything that is wrong).
#[test]
#[ignore = "child process of the sampled_mtp_adaptive_* tests; runs one shard when run directly"]
fn sampled_mtp_late_arm() {
    let Ok(arm) = std::env::var(ARM_ENV) else {
        return;
    };
    if let Err(error) = run_late_arm(&arm, &model_path(), &out_dir()) {
        panic!("{arm} arm: {error}");
    }
}

fn run_late_arm(arm: &str, model: &Path, dir: &Path) -> Result<(), String> {
    if !matches!(arm, LATE_AR_ARM | LATE_ADAPTIVE_ARM | LATE_PILOT_ARM) {
        return Err(format!("unknown late arm {arm}"));
    }
    require_arm_env(arm, "leviathan")?;
    let settings = DistSettings::from_env()?;
    let tokens = late_tokens()?;
    let range = shard_range(arm, settings.trials)?;
    let env = own_env(&LATE_CONTROL_ENV);
    let identity = own_identity(model, dir, &LATE_CONTROL_ENV)?;
    let config = late_config(arm, &identity, &settings, tokens, range);
    let ids_path = late_file(dir, arm, range, "ids.json");
    if ids_path.exists() {
        check_config(&ids_path, &config)?;
        println!(
            "ARM {arm} shard {}..{} already complete under this config",
            range.0, range.1
        );
        return Ok(());
    }
    let DistSettings {
        temp,
        top_p,
        top_k,
        min_p,
        pen,
        fixture,
        ..
    } = settings;
    let mut loaded = load(model)?;
    let prompt = loaded.render(&fixture.text())?;
    fixture.check_prompt(prompt.len(), &pen)?;
    let eos = loaded.bundle.config.eos_token_id;
    let budget = floor_probe_budget()?;
    println!(
        "ARM {arm} arch={} prompt_tokens={} shard={}..{} tokens={tokens} temp={temp} top_p={top_p} top_k={top_k:?} min_p={min_p} fixture={} state={} route_class={} probe_budget={budget}",
        loaded.gpu.arch,
        prompt.len(),
        range.0,
        range.1,
        fixture.name(),
        loaded.state,
        hw_identity::route_class(&env)
    );
    println!("ARM {arm} config {config}");
    let started = std::time::Instant::now();
    let (start, end) = range;
    let mut samples: Vec<Vec<u32>> = Vec::with_capacity(end - start);
    let progress = |trial: usize| {
        if (trial + 1 - start) % 25 == 0 {
            println!(
                "ARM {arm} {}/{} trials in {:.0}s",
                trial + 1 - start,
                end - start,
                started.elapsed().as_secs_f64()
            );
        }
    };
    if arm == LATE_AR_ARM {
        let logits = loaded
            .gpu
            .zeros(&[loaded.bundle.config.vocab_size], DType::F32)
            .map_err(|e| e.to_string())?;
        let cfg = ar_sampler(temp, top_p, top_k, min_p, &pen);
        for trial in start..end {
            let seed = trial_seed(LATE_AR_SALT, trial) as u32;
            samples.push(loaded.ar_ids(&logits, &prompt, &cfg, seed, tokens)?);
            progress(trial);
        }
        loaded.gpu.free_tensor(logits).map_err(|e| e.to_string())?;
    } else {
        let mut evidence = Evidence::with_samples(if arm == LATE_PILOT_ARM {
            PILOT_TRIALS
        } else {
            EVIDENCE_SAMPLES
        });
        let salt = late_salt(arm);
        let mut drafter = new_drafter(&mut loaded)?;
        for trial in start..end {
            let seed = trial_seed(salt, trial);
            let cfg = spec_request(temp, top_p, top_k, min_p, seed, &pen);
            let result = loaded.mtp_request(&mut drafter, &prompt, cfg, tokens)?;
            evidence.observe(trial, seed, &result, &loaded.tokenizer, budget, LATE_MARGINALS[0]);
            samples.push(result.ids);
            progress(trial);
        }
        Box::new(drafter).mtp_free(&mut loaded.gpu);
        evidence.write(
            dir,
            &format!("{arm}.{}-{}", range.0, range.1),
            budget,
            LATE_MARGINALS[0],
        )?;
    }
    // Positions after an EOS read as EOS; every trial is kept.
    for ids in &mut samples {
        ids.resize(tokens, eos);
    }
    println!(
        "ARM {arm} shard {}..{} in {:.1}s; first sample: {:?}",
        range.0,
        range.1,
        started.elapsed().as_secs_f64(),
        loaded.tokenizer.decode(&samples[0])
    );
    fs::write(
        &ids_path,
        serde_json::to_vec(&samples).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    write_config(&ids_path, &config)?;
    write_receipt(&ids_path, &hw_identity::receipt(&loaded.gpu, model))?;
    loaded
        .bundle
        .free_gpu(&mut loaded.gpu)
        .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// GPU-free checks of the coverage logic
// ---------------------------------------------------------------------------

mod coverage_logic {
    use super::*;

    fn rec(
        index: usize,
        position: usize,
        before: (usize, usize, bool),
        after: (usize, usize, bool),
        drafts: usize,
        committed: usize,
    ) -> WindowRec {
        let snap = |(mtp_windows, ar_windows, retired): (usize, usize, bool)| StatsSnap {
            mtp_windows,
            ar_windows,
            ngram_windows: 0,
            retired,
        };
        WindowRec {
            index,
            position_before: position,
            position_after: position + committed,
            k_budget: 3,
            drafts,
            accepted: drafts.min(committed.saturating_sub(1)),
            committed,
            before: snap(before),
            after: snap(after),
            counters: Counters::default(),
        }
    }

    /// calibration, three probe windows (the last one retires), then AR.
    fn full_sequence(first_ar_index: usize) -> Vec<WindowRec> {
        vec![
            rec(1, 100, (0, 0, false), (1, 0, false), 0, 1),
            rec(2, 101, (1, 0, false), (2, 0, false), 3, 4),
            rec(6, 105, (2, 0, false), (3, 0, false), 3, 2),
            rec(8, 107, (3, 0, false), (4, 0, true), 3, 4),
            rec(first_ar_index, 111, (4, 0, true), (4, 1, true), 0, 1),
        ]
    }

    #[test]
    fn adaptive_sequence_full_when_every_stage_is_present() {
        let coverage = analyze_windows(&full_sequence(12), 3, 31);
        assert!(coverage.calibration_ok);
        assert_eq!(coverage.retired_at, Some(3));
        assert_eq!(coverage.probes_before_retirement, 3);
        assert_eq!(coverage.post_retirement_ar, vec![12]);
        assert!(coverage.full);
        assert!(!coverage.late_after_retirement);
        let late = analyze_windows(&full_sequence(40), 3, 31);
        assert!(late.full && late.late_after_retirement);
    }

    #[test]
    fn adaptive_sequence_needs_the_whole_probe_budget() {
        // Retirement is decided by the router at the start of the fourth
        // step (the AR window itself) after only two probes.
        let windows = vec![
            rec(1, 100, (0, 0, false), (1, 0, false), 0, 1),
            rec(2, 101, (1, 0, false), (2, 0, false), 3, 4),
            rec(6, 105, (2, 0, false), (3, 0, false), 3, 2),
            rec(8, 107, (3, 0, false), (3, 1, true), 0, 1),
            rec(9, 108, (3, 1, true), (3, 2, true), 0, 1),
        ];
        let coverage = analyze_windows(&windows, 3, 31);
        assert_eq!(coverage.probes_before_retirement, 2);
        assert!(!coverage.full);
        // ...and the same sequence satisfies a budget of two.
        assert!(analyze_windows(&windows, 2, 31).full);
    }

    #[test]
    fn adaptive_sequence_needs_an_ar_window_after_the_retiring_step() {
        let mut windows = full_sequence(12);
        windows.pop();
        let coverage = analyze_windows(&windows, 3, 31);
        assert_eq!(coverage.retired_at, Some(3));
        assert!(coverage.post_retirement_ar.is_empty());
        assert!(!coverage.full);
        // The retiring step itself being the AR window does not count as "later".
        let only_ar = vec![
            rec(1, 100, (0, 0, false), (1, 0, false), 0, 1),
            rec(2, 101, (1, 0, false), (2, 0, false), 3, 4),
            rec(6, 105, (2, 0, false), (3, 0, false), 3, 2),
            rec(8, 107, (3, 0, false), (4, 0, false), 3, 4),
            rec(12, 111, (4, 0, false), (4, 1, true), 0, 1),
        ];
        let coverage = analyze_windows(&only_ar, 3, 31);
        assert_eq!(coverage.retired_at, Some(4));
        assert_eq!(coverage.probes_before_retirement, 3);
        assert!(coverage.post_retirement_ar.is_empty() && !coverage.full);
    }

    #[test]
    fn adaptive_sequence_without_retirement_or_calibration_is_not_full() {
        let mut windows = full_sequence(12);
        windows.truncate(3);
        let coverage = analyze_windows(&windows, 3, 31);
        assert!(coverage.calibration_ok && coverage.retired_at.is_none() && !coverage.full);

        // A first window that drafted is not the calibration token.
        let mut windows = full_sequence(12);
        windows[0].drafts = 3;
        let coverage = analyze_windows(&windows, 3, 31);
        assert!(!coverage.calibration_ok && !coverage.full);
        assert!(analyze_windows(&[], 3, 31) == SeqCoverage::default());
    }

    #[test]
    fn late_shards_cover_exactly_once() {
        assert_eq!(shard_gaps(&[], 10), Ok(vec![(0, 10)]));
        assert_eq!(shard_gaps(&[(0, 4), (4, 10)], 10), Ok(vec![]));
        assert_eq!(
            shard_gaps(&[(2, 4), (6, 8)], 10),
            Ok(vec![(0, 2), (4, 6), (8, 10)])
        );
        assert!(shard_gaps(&[(0, 5), (4, 10)], 10).is_err(), "overlap");
        assert!(shard_gaps(&[(0, 12)], 10).is_err(), "beyond the cohort");
    }

    #[test]
    fn probe_budget_is_read_from_the_source_under_test() {
        assert!(floor_probe_budget().unwrap() >= 1);
    }

    #[test]
    fn forced_arms_reset_the_floor_and_adaptive_arms_clear_it() {
        let find = |arm: &str, name: &str| {
            arm_overrides(arm, "leviathan")
                .into_iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value)
        };
        assert_eq!(find("batched", FLOOR_ENV), Some(Some("0")));
        assert_eq!(find("batched", INCREMENTAL_ENV), Some(Some("0")));
        assert_eq!(find("copy_interleaved", INCREMENTAL_ENV), Some(Some("1")));
        for arm in ["ar", "adaptive", "ngram_adaptive", LATE_ADAPTIVE_ARM, LATE_PILOT_ARM] {
            assert_eq!(find(arm, FLOOR_ENV), Some(None), "{arm}");
            assert_eq!(find(arm, INCREMENTAL_ENV), Some(None), "{arm}");
        }
    }
}

// ---------------------------------------------------------------------------
// Thinking termination
// ---------------------------------------------------------------------------
//
// A sampled-speculation change must not stop a long thinking turn from
// closing. `sampled_mtp_thinking_terminates_like_ar_on_flash_next` renders a
// fixed built-in math prompt with thinking ON (the same chat frame as
// `render`, only `enable_thinking` differs), decodes it in plain AR (host
// sampler) and in sampled adaptive MTP (Leviathan) on the same binary with the
// same sampling, penalties and budget, over a fixed seed cohort
// (`trial_seed(salt, 0..seeds)`), and requires `</think>` in the decoded
// output of EVERY seed of BOTH arms. Nothing is selected after the fact: every
// seed counts, and a seed that does not close fails the test. No law,
// sampler or routing is changed; the arms are the production AR sampler and
// the production adaptive sampled MTP path (floor on, no route knob).
//
// Decoded text of every seed is written to `think_ar.think.txt` and
// `think_adaptive.think.txt`; per-seed IDs / close index to `*.think.json`;
// the MTP arm's per-window route / accepted / retirement counters to
// `think_adaptive.evidence.json`. Results carry the same config / identity /
// receipt sidecars as the other arms and are reused only under an equal config.
// `HIPFIRE_SAMPLED_MTP_THINK_TOKENS` (default 3500) and
// `HIPFIRE_SAMPLED_MTP_THINK_SEEDS` (default 4) change the declared budget and
// cohort for both arms together.

const THINK_AR_ARM: &str = "think_ar";
const THINK_ADAPTIVE_ARM: &str = "think_adaptive";
const THINK_ARM_TEST: &str = "sampled_mtp_think_arm";
const THINK_TOKENS_ENV: &str = "HIPFIRE_SAMPLED_MTP_THINK_TOKENS";
const THINK_SEEDS_ENV: &str = "HIPFIRE_SAMPLED_MTP_THINK_SEEDS";
/// Generated IDs per request, shared by both arms; must leave room in `MAX_SEQ`.
const THINK_TOKENS_DEFAULT: usize = 3500;
const THINK_SEEDS_DEFAULT: usize = 4;
const THINK_AR_SALT: u64 = 0x7A1A_A200;
const THINK_MTP_SALT: u64 = 0x7A1A_5EC0;
const THINK_CLOSE: &str = "</think>";
/// A fixed counting problem with a cheap-to-verify answer (the answer is 400)
/// that invites a long deliberate check; sent as one user message.
const THINK_PROMPT: &str = "How many positive integers less than 1000 are divisible by 3 or by 5 but not by 15? Work it out carefully, then verify the count with a second, independent method before giving the final answer.";

fn think_dims() -> Result<(usize, usize), String> {
    let tokens: usize = strict_env(THINK_TOKENS_ENV, THINK_TOKENS_DEFAULT)?;
    let seeds: usize = strict_env(THINK_SEEDS_ENV, THINK_SEEDS_DEFAULT)?;
    if tokens == 0 || tokens + 512 > MAX_SEQ || seeds == 0 {
        return Err(format!(
            "{THINK_TOKENS_ENV}={tokens} / {THINK_SEEDS_ENV}={seeds}: need 0 < tokens <= {} and seeds >= 1",
            MAX_SEQ - 512
        ));
    }
    Ok((tokens, seeds))
}

fn think_config(
    arm: &str,
    identity: &Value,
    settings: &DistSettings,
    tokens: usize,
    seeds: usize,
) -> Value {
    let mut config = config_base(
        "think-termination",
        arm,
        &settings.pen,
        Fixture::Prose,
        identity,
    );
    config["fixture"] = json!("think");
    config["prompt_sha256"] = json!(hw_identity::sha256_hex(THINK_PROMPT.as_bytes()));
    config["temp"] = json!(settings.temp);
    config["top_p"] = json!(settings.top_p);
    config["top_k"] = json!(settings.top_k);
    config["min_p"] = json!(settings.min_p);
    config["tokens"] = json!(tokens);
    config["seeds"] = json!(seeds);
    config["salt"] = json!(if arm == THINK_AR_ARM {
        THINK_AR_SALT
    } else {
        THINK_MTP_SALT
    });
    config["sampled_mode"] = json!("leviathan");
    config
}

/// One arm in a fresh process; spawned by the termination test below.
#[test]
#[ignore = "child process of sampled_mtp_thinking_terminates_like_ar_on_flash_next"]
fn sampled_mtp_think_arm() {
    let Ok(arm) = std::env::var(ARM_ENV) else {
        return;
    };
    if let Err(error) = run_think_arm(&arm, &model_path(), &out_dir()) {
        panic!("{arm} arm: {error}");
    }
}

fn run_think_arm(arm: &str, model: &Path, dir: &Path) -> Result<(), String> {
    if !matches!(arm, THINK_AR_ARM | THINK_ADAPTIVE_ARM) {
        return Err(format!("unknown thinking arm {arm}"));
    }
    require_arm_env(arm, "leviathan")?;
    let settings = DistSettings::from_env()?;
    let (tokens, seeds) = think_dims()?;
    let identity = own_identity(model, dir, &CONTROL_ENV)?;
    let config = think_config(arm, &identity, &settings, tokens, seeds);
    let ids_path = dir.join(format!("{arm}.ids.json"));
    if ids_path.exists() {
        check_config(&ids_path, &config)?;
        println!("ARM {arm} already complete under this config");
        return Ok(());
    }
    let DistSettings {
        temp,
        top_p,
        top_k,
        min_p,
        pen,
        ..
    } = settings;
    let env = own_env(&CONTROL_ENV);
    let mut loaded = load(model)?;
    let prompt = loaded.render_with(THINK_PROMPT, true)?;
    if prompt.len() + tokens + MTP_K + 1 > MAX_SEQ {
        return Err(format!(
            "prompt {} + {tokens} IDs does not fit MAX_SEQ {MAX_SEQ}",
            prompt.len()
        ));
    }
    let close_id = {
        let ids = loaded.tokenizer.encode(THINK_CLOSE);
        (ids.len() == 1).then(|| ids[0])
    };
    println!(
        "ARM {arm} arch={} prompt_tokens={} tokens={tokens} seeds={seeds} close_token={close_id:?} route_class={} state={}",
        loaded.gpu.arch,
        prompt.len(),
        hw_identity::route_class(&env),
        loaded.state
    );
    println!("ARM {arm} config {config}");
    let budget = floor_probe_budget()?;
    let mut evidence = Evidence::with_samples(seeds);
    let mut all_ids: Vec<Vec<u32>> = Vec::with_capacity(seeds);
    let mut records: Vec<Value> = Vec::with_capacity(seeds);
    let mut texts = String::new();
    let started = std::time::Instant::now();
    let logits = if arm == THINK_AR_ARM {
        Some(
            loaded
                .gpu
                .zeros(&[loaded.bundle.config.vocab_size], DType::F32)
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let mut drafter = if arm == THINK_ADAPTIVE_ARM {
        Some(new_drafter(&mut loaded)?)
    } else {
        None
    };
    for n in 0..seeds {
        let (ids, seed) = match (logits.as_ref(), drafter.as_mut()) {
            (Some(logits), None) => {
                let seed = trial_seed(THINK_AR_SALT, n);
                let cfg = ar_sampler(temp, top_p, top_k, min_p, &pen);
                (loaded.ar_ids(logits, &prompt, &cfg, seed as u32, tokens)?, seed)
            }
            (None, Some(drafter)) => {
                let seed = trial_seed(THINK_MTP_SALT, n);
                let cfg = spec_request(temp, top_p, top_k, min_p, seed, &pen);
                let run = loaded.mtp_request(drafter, &prompt, cfg, tokens)?;
                evidence.observe(n, seed, &run, &loaded.tokenizer, budget, LATE_MARGINALS[0]);
                (run.ids, seed)
            }
            _ => return Err("thinking arm state".into()),
        };
        let text = loaded.tokenizer.decode(&ids);
        let close_at = close_id.and_then(|id| ids.iter().position(|&token| token == id));
        let closed = close_at.is_some() || text.contains(THINK_CLOSE);
        println!(
            "ARM {arm} seed#{n}: {} ids, closed={closed} close_index={close_at:?}, {:.0}s",
            ids.len(),
            started.elapsed().as_secs_f64()
        );
        texts.push_str(&format!(
            "=== seed#{n} ({seed}): {} ids, closed={closed} ===\n{text}\n\n",
            ids.len()
        ));
        records.push(json!({
            "seed_index": n,
            "seed": seed,
            "ids": ids.len(),
            "closed": closed,
            "close_index": close_at,
        }));
        all_ids.push(ids);
    }
    if let Some(logits) = logits {
        loaded.gpu.free_tensor(logits).map_err(|e| e.to_string())?;
    }
    if let Some(drafter) = drafter {
        Box::new(drafter).mtp_free(&mut loaded.gpu);
        evidence.write(dir, arm, budget, LATE_MARGINALS[0])?;
    }
    write_json(
        &dir.join(format!("{arm}.think.json")),
        &json!({ "prompt_tokens": prompt.len(), "close_token": close_id, "seeds": records }),
    )?;
    let text_path = dir.join(format!("{arm}.think.txt"));
    fs::write(&text_path, &texts).map_err(|e| format!("{}: {e}", text_path.display()))?;
    fs::write(
        &ids_path,
        serde_json::to_vec(&all_ids).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    write_config(&ids_path, &config)?;
    write_receipt(&ids_path, &hw_identity::receipt(&loaded.gpu, model))?;
    loaded
        .bundle
        .free_gpu(&mut loaded.gpu)
        .map_err(|e| e.to_string())
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn sampled_mtp_thinking_terminates_like_ar_on_flash_next() {
    let model = model_path();
    let dir = out_dir();
    let settings = DistSettings::from_env().unwrap_or_else(|e| panic!("{e}"));
    let (tokens, seeds) = think_dims().unwrap_or_else(|e| panic!("{e}"));
    let model_id = hw_identity::model_identity(&model, &dir).unwrap_or_else(|e| panic!("{e}"));
    let mut closed = [0usize; 2];
    let mut report = format!(
        "thinking termination: {seeds} seeds, {tokens} IDs, prompt sha256 {}\n",
        hw_identity::sha256_hex(THINK_PROMPT.as_bytes())
    );
    for (slot, arm) in [THINK_AR_ARM, THINK_ADAPTIVE_ARM].into_iter().enumerate() {
        let identity = arm_identity(arm, "leviathan", &model_id, &CONTROL_ENV);
        let expected = think_config(arm, &identity, &settings, tokens, seeds);
        let ids_path = dir.join(format!("{arm}.ids.json"));
        if ids_path.exists() {
            eprintln!("reusing {}", ids_path.display());
        } else {
            spawn_arm(THINK_ARM_TEST, arm, "leviathan", &model, &dir, None, arm)
                .unwrap_or_else(|e| panic!("{e}"));
        }
        check_config(&ids_path, &expected).unwrap_or_else(|e| panic!("{e}"));
        let think =
            read_json(&dir.join(format!("{arm}.think.json"))).unwrap_or_else(|e| panic!("{e}"));
        for record in think["seeds"].as_array().into_iter().flatten() {
            closed[slot] += usize::from(record["closed"].as_bool().unwrap_or(false));
            report.push_str(&format!("{arm}: {record}\n"));
        }
    }
    report.push_str(&format!(
        "closed {THINK_CLOSE}: ar {}/{seeds}, sampled adaptive MTP {}/{seeds}\n",
        closed[0], closed[1]
    ));
    eprint!("{report}");
    fs::write(dir.join("think-report.txt"), &report).unwrap();
    // Decoded text of every seed: think_ar.think.txt, think_adaptive.think.txt.
    assert!(
        closed[0] == seeds,
        "plain AR closed {THINK_CLOSE} in only {}/{seeds} seeds within {tokens} IDs (decoded text in {}/think_ar.think.txt): the shared budget is too small for this prompt, so the comparison is void; raise {THINK_TOKENS_ENV}",
        closed[0],
        dir.display()
    );
    assert!(
        closed[1] == seeds,
        "THINKING TERMINATION FAILURE: plain AR closed {THINK_CLOSE} in {seeds}/{seeds} seeds but sampled adaptive MTP in {}/{seeds} (decoded text in {}/think_adaptive.think.txt)",
        closed[1],
        dir.display()
    );
}
