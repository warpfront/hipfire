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
//! # GDN state mode
//!
//! `HIPFIRE_GDN_TEST_STATE_QUANT=q8|fp32|q8_ef` picks the GDN recurrent-state
//! format (unset = `q8`, the historical fixture; anything else is refused).
//! The AR-vs-MTP comparisons of this file keep their Q8/F32 meaning; under
//! `q8_ef` the uninterrupted-AR law is not the target law of a rollback-reset
//! window, so they are diagnostics there and the q8_ef verdict is the
//! separate `sampled_mtp_q8ef_matches_verify_distribution_on_flash_next`
//! (verify-route law; see its section at the end of this file). The load
//! passes the value as `state_request` to `resolve_state_format`
//! and fails unless the resolved `GdnStateFormat` is the requested one; the
//! product's `HIPFIRE_STATE_QUANT` is never read. The parent resolves the mode
//! once and sets it in every child's environment (`arm_overrides`), so it is
//! part of every arm's recorded identity (`config.env` of the `.config.json`)
//! and a result produced under another mode is a hard config mismatch, never
//! reused. A separately launched arm must set it itself (like the other
//! variables `require_arm_env` pins) and fails naming it otherwise. Each arm
//! prints the request and the resolved format (`state_quant=... gdn_format=...`).
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
//! (`speculation.mtp_ngram_copy` / `HIPFIRE_MTP_NGRAM_COPY`; `auto` is the
//! copy lane on gfx1151 / gfx1201 and the native-MTP 5/3/3 takeover elsewhere;
//! `HIPFIRE_MTP_NGRAM_COPY=off` runs them on the 5/3/3 pool, `on` on the copy
//! lane). The mode is recorded as `copy_lane` in every arm's config,
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
//! `FLOOR_PROBE_WINDOWS` probe windows -> `mtp_retired` false -> true ->
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
//!
//! # PP2 topology arms (sampled Leviathan MTP under `topology.pp`)
//!
//! `HIPFIRE_SAMPLED_MTP_TOPOLOGY=pp2` runs the four short arms (`ar`,
//! `batched`, `interleaved`, `adaptive`) of the chi-square above on a two-card
//! pipeline mesh instead of one card, so sampled Leviathan MTP on `topology.pp`
//! is compared with AR sampling in distribution (unset or `single` keeps the
//! one-card arms, byte for byte unchanged). Both arms of the comparison load
//! the SAME artifact through the product loader (`hipfire_loader::load_model`
//! with `pp = 2`, the daemon's own mesh load, native MTP head attached on every
//! arm so placement and reserves match): the `ar` arm is PP AR (`pp_prefill_final`
//! / `pp_forward_token_or_argmax` logits on the output band's card, the
//! production host sampler `sample_cpu`, per-trial `reset_cpu_sampler_rng`);
//! the MTP arms drive `pp_mtp_reset` / `pp_mtp_prefill` / `pp_mtp_step`, the
//! entries the daemon's spec loop calls. A 2xR9700 box cannot hold the
//! mq3-xts / mq4-xts artifact on one card, so the PP2 `ar` arm is the
//! reference there (AR on one mesh and AR on one card are the same target law;
//! compare the PP2 AR arm with the one-card `ar` arm on the gptq3 artifact
//! separately if that cross-check is wanted). Everything else is the file's
//! own machinery: shards (`HIPFIRE_SAMPLED_MTP_SHARD`), merge
//! (`sampled_mtp_merge_short_shards`), per-position marginals and (t1, t2)
//! joint chi-square at `ALPHA`, evidence/coverage checks, config sidecars and
//! the replay of the first `REPLAYS` seeds on the forced floor-off arms (the
//! PP verify is deterministic, so a replay mismatch is a finding).
//!
//! The model is whatever `HIPFIRE_MTP_IDENTITY_MODEL` names (the 2xR9700
//! mq3-xts or mq4-xts artifact, or the gptq3 mq4 one); the topology and
//! `HIPFIRE_PP_LAYERS` (band layer counts in logical band order, summing to
//! the trunk layer count; unset = the loader's uniform split) are `HIPFIRE_*`
//! variables, so they are part of every arm's recorded identity and a result
//! produced under another split or topology is a hard config mismatch, never
//! reused. Only the four short arms take a PP2 topology (every other arm fails
//! naming the variable). The PP2 arms run neutral penalties only (penalized
//! sampled requests decode AR under PP and `HIPFIRE_MTP_SAMPLED_MODE=naive` is
//! refused there); do not set `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` (the PP load
//! refuses host-mapped experts), and leave both cards visible (no
//! `ROCR_VISIBLE_DEVICES=0` restriction). Run from the repository root:
//!
//! ```sh
//! export MODEL=/path/to/qwen3.8-flash-next.mq3-xts   # or the mq4-xts / gptq3 mq4 artifact
//! export OUT=/tmp/pp2-sampled
//! export HIPFIRE_MTP_IDENTITY_MODEL="$MODEL" HIPFIRE_SAMPLED_MTP_OUT="$OUT"
//! export HIPFIRE_SAMPLED_MTP_TOPOLOGY=pp2
//! # export HIPFIRE_PP_LAYERS=24,24      # optional band split, logical band order
//! export HIPFIRE_SAMPLED_MTP_TRIALS=2000 HIPFIRE_SAMPLED_MTP_PROMPT=prose
//! export HIPFIRE_MTP_SAMPLED=1 HIPFIRE_MTP_SAMPLED_MODE=leviathan
//! unset HIPFIRE_MTP_TRACE HIPFIRE_MTP_PHASE_TIMING HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS
//! T="cargo test --release -p hipfire-arch-qwen4 --test sampled_mtp_distribution_hw"
//! # Per arm, per shard (disjoint START..END ranges covering 0..2000; seeds are global):
//! export HIPFIRE_SAMPLED_MTP_SHARD=0..100
//! export HIPFIRE_SAMPLED_MTP_ARM=ar
//! unset HIPFIRE_MTP_INCREMENTAL HIPFIRE_MTP_AR_FLOOR
//! $T sampled_mtp_distribution_arm -- --exact --ignored --nocapture --test-threads=1
//! export HIPFIRE_SAMPLED_MTP_ARM=batched HIPFIRE_MTP_INCREMENTAL=0 HIPFIRE_MTP_AR_FLOOR=0
//! $T sampled_mtp_distribution_arm -- --exact --ignored --nocapture --test-threads=1
//! export HIPFIRE_SAMPLED_MTP_ARM=interleaved HIPFIRE_MTP_INCREMENTAL=1 HIPFIRE_MTP_AR_FLOOR=0
//! $T sampled_mtp_distribution_arm -- --exact --ignored --nocapture --test-threads=1
//! export HIPFIRE_SAMPLED_MTP_ARM=adaptive; unset HIPFIRE_MTP_INCREMENTAL HIPFIRE_MTP_AR_FLOOR
//! $T sampled_mtp_distribution_arm -- --exact --ignored --nocapture --test-threads=1
//! # ... repeat the four arms for every shard range, then (CPU only) join the shards:
//! unset HIPFIRE_SAMPLED_MTP_ARM HIPFIRE_SAMPLED_MTP_SHARD
//! $T sampled_mtp_merge_short_shards -- --exact --ignored --nocapture
//! # The verdict (reuses the merged, config-verified whole-arm files):
//! $T sampled_mtp_matches_ar_distribution_on_flash_next -- --exact --ignored --nocapture
//! ```
//!
//! Expected artifacts in `$OUT`, per arm and shard `<arm>.<START>-<END>`:
//! `.ids.json` (one `TOKENS`-id row per trial), `.ids.config.json` (config
//! with the PP identity: `HIPFIRE_SAMPLED_MTP_TOPOLOGY`, `HIPFIRE_PP_LAYERS`,
//! executable and model hashes), `.ids.receipt.json` (output card's GPU
//! identity) and, for the MTP arms, `.evidence.json` and `.text.txt`; after the
//! merge the same names without the range, and `report.txt` from the verdict
//! test (per-arm marginal and joint chi-square lines; every p-value must exceed
//! `ALPHA`, coverage failures are reported separately). Run unsharded
//! (`HIPFIRE_SAMPLED_MTP_SHARD` unset) and the parent test alone spawns the
//! four arms in fresh processes.

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
const TOPOLOGY_ENV: &str = "HIPFIRE_SAMPLED_MTP_TOPOLOGY";
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
const COPY_INSTRUCTION: &str =
    "Rewrite the following passage, keeping its wording wherever you can.\n\n";
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
        Ok(bytes) => {
            serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", config_path.display()))?
        }
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
        let evidence =
            read_json(&dir.join(format!("{arm}.evidence.json"))).unwrap_or_else(|e| panic!("{e}"));
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
    load_with_max_seq(model, MAX_SEQ)
}

/// [`load`] with the context (`max_seq`) the forward, the native-MTP reserve
/// and the assembled bundle are sized for.
fn load_with_max_seq(model: &Path, max_seq: usize) -> Result<Loaded, String> {
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
    let (state_request, state_format) = hw_identity::resolve_gdn_test_state_format(
        &hipfire_runtime::config::get().kv_mode,
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
            if entry
                .name
                .ends_with(hipfire_arch_qwen4::expert_arena::ARENA_SUFFIX)
            {
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
        let chunk = hipfire_arch_qwen4::gpu_forward::qwen4_prefill_chunk_requested(
            &gpu,
            max_seq,
            scratch_mode,
        );
        let context = hipfire_arch_qwen4::Qwen4ContextCommit::for_expert_reserve(
            backend,
            gpu.is_uma(),
            max_seq,
            chunk,
            gpu.vmm_recommended_granularity().unwrap_or(1),
        );
        // Every oracle arm reserves the same native MTP footprint.
        let head = residency::language_head_dtype(&weights)
            .ok_or("qwen4: manifest has no language head")?;
        let row_capture = hipfire_arch_qwen4::mtp_spec::native_mtp_capture(
            &gpu,
            &receipt.config,
            state_format.gdn,
            MTP_K,
        );
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
            .map(|bytes| (bytes as u64).saturating_sub(gpu.qsa_gather_scratch_bytes() as u64))
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
            Ok(residency::fits_fully_resident(
                free, non_expert, experts, reserve,
            ))
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
        max_seq,
        receipt.ple,
        state_format,
        backend,
    )
    .map_err(|e| e.to_string())?;
    bundle
        .attach_forward(&mut gpu, max_seq, scratch_mode)
        .map_err(|e| e.to_string())?;
    Ok(Loaded {
        gpu,
        bundle,
        tokenizer,
        template,
        state: format!(
            "state_quant={state_request} gdn_format={} {state_format:?}",
            state_format.gdn.name()
        ),
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
        render_prompt(&self.tokenizer, &self.template, text, thinking)
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
    new_drafter_with_max_seq(loaded, ngram, copy_lane, MAX_SEQ)
}

/// [`new_drafter_with`] for a target loaded with context `max_seq`
/// ([`load_with_max_seq`]).
fn new_drafter_with_max_seq(
    loaded: &mut Loaded,
    ngram: Option<NgramModConfig>,
    copy_lane: bool,
    max_seq: usize,
) -> Result<Qwen4MtpDrafter, String> {
    loaded
        .bundle
        .attach_mtp(&mut loaded.gpu, max_seq)
        .map_err(|e| e.to_string())?;
    let drafter = Qwen4MtpDrafter::new(MTP_K, max_seq, None)
        .with_ngram(ngram)
        .with_copy_lane(copy_lane);
    if !drafter.supports_temp_verify() {
        return Err("HIPFIRE_MTP_SAMPLED did not enable sampled verification".into());
    }
    Ok(drafter)
}

fn run_arm(arm: &str, model: &Path, dir: &Path) -> Result<(), String> {
    require_arm_env(arm, "leviathan")?;
    if topology_from_env()? == Topology::Pp2 {
        return run_arm_pp(arm, model, dir);
    }
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
    // `HIPFIRE_SAMPLED_MTP_SHARD=START..END` runs a slice of the global trial
    // indices (seeds stay global) and writes `<arm>.<START>-<END>.*`; unset
    // keeps the whole-arm `<arm>.*` files. `sampled_mtp_merge_short_shards`
    // joins shards into the whole-arm files the parent comparison reads.
    let (start, end) = shard_range(arm, trials)?;
    let sharded = (start, end) != (0, trials);
    let stem = if sharded {
        format!("{arm}.{start}-{end}")
    } else {
        arm.to_string()
    };
    let ids_path = dir.join(format!("{stem}.ids.json"));
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
    let mut samples: Vec<Vec<u32>> = Vec::with_capacity(end - start);
    if arm == "ar" {
        let logits = loaded
            .gpu
            .zeros(&[loaded.bundle.config.vocab_size], DType::F32)
            .map_err(|e| e.to_string())?;
        let cfg = ar_sampler(temp, top_p, top_k, min_p, &pen);
        for trial in start..end {
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
        for trial in start..end {
            let seed = trial_seed(0x5EC, trial);
            let result = run(&mut loaded, seed)?;
            evidence.observe(
                trial,
                seed,
                &result,
                &loaded.tokenizer,
                budget,
                LATE_MARGINALS[0],
            );
            samples.push(result.ids);
        }
        // Only a forced floor-off route replays byte for byte; the adaptive
        // route's choices depend on wall-clock prices.
        if hw_identity::forced_floor_off(&env) {
            let replays = REPLAYS.min(end - start);
            for local in 0..replays {
                let trial = start + local;
                let replay = run(&mut loaded, trial_seed(0x5EC, trial))?;
                if replay.ids != samples[local] {
                    return Err(format!(
                        "seed {} replayed {:?}, first run {:?}",
                        trial + 1,
                        replay.ids,
                        samples[local]
                    ));
                }
            }
            println!("ARM {arm} replayed {replays} seeds identically");
        } else {
            println!(
                "ARM {arm} replay not asserted (route class {}): its routing depends on wall-clock prices",
                hw_identity::route_class(&env)
            );
        }
        evidence.write(dir, &stem, budget, LATE_MARGINALS[0])?;
        Box::new(drafter).mtp_free(&mut loaded.gpu);
    }
    // Positions after an EOS read as EOS.
    for ids in &mut samples {
        ids.resize(TOKENS, eos);
    }
    println!(
        "ARM {arm} trials {start}..{end} in {:.1}s; first sample: {:?}",
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

// ---------------------------------------------------------------------------
// PP2 topology arms (`HIPFIRE_SAMPLED_MTP_TOPOLOGY=pp2`)
// ---------------------------------------------------------------------------

/// Where the short arms run. `Single` is the historical one-card load of
/// [`load`]; `Pp2` is a two-card pipeline mesh loaded through the product
/// loader (`HIPFIRE_PP_LAYERS` picks the band split).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Topology {
    Single,
    Pp2,
}

/// The pipeline degree of [`Topology::Pp2`].
const PP_DEGREE: usize = 2;

fn topology_from_env() -> Result<Topology, String> {
    match std::env::var(TOPOLOGY_ENV).as_deref() {
        Err(_) | Ok("") | Ok("single") => Ok(Topology::Single),
        Ok("pp2") => Ok(Topology::Pp2),
        Ok(other) => Err(format!(
            "{TOPOLOGY_ENV}={other:?}: expected `single` or `pp2`"
        )),
    }
}

/// `text` as one user message, thinking as given, rendered as `hipfire serve`
/// renders it.
fn render_prompt(
    tokenizer: &Tokenizer,
    template: &str,
    text: &str,
    thinking: bool,
) -> Result<Vec<u32>, String> {
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
        tokenizer,
        template,
        system: None,
        user: text,
        enable_thinking: thinking,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: None,
    }
    .render_messages(&[user], None, None)?;
    Ok(tokenizer.encode(&rendered))
}

/// The native MTP drafter behind the load's speculator, for the mesh entries.
fn pp_drafter(
    speculator: &mut dyn hipfire_runtime::spec::Speculator,
) -> Result<&mut Qwen4MtpDrafter, String> {
    speculator
        .drafter_any_mut()
        .and_then(|any| any.downcast_mut::<Qwen4MtpDrafter>())
        .ok_or_else(|| "the load's speculator is not the native Qwen4 MTP drafter".to_string())
}

/// The mesh's output band device (logits, the MTP head and the sampled verify's
/// reads live there) with the calling thread bound to it.
fn pp_owner(gpus: &mut hipfire_runtime::multi_gpu::Gpus) -> Result<&mut Gpu, String> {
    let owner = gpus
        .mesh_output_owner()
        .ok_or("the PP mesh has no output owner")?;
    let gpu = gpus
        .devices
        .get_mut(owner)
        .ok_or("the PP output owner is not a mesh device")?;
    gpu.bind_thread().map_err(|e| e.to_string())?;
    Ok(gpu)
}

/// A Flash-Next target loaded across a PP2 mesh by the product loader, with
/// its tokenizer, chat template and native MTP drafter (loaded on every arm,
/// used by the MTP arms only).
struct PpLoaded {
    model: hipfire_loader::LoadedModel,
    speculator: Box<dyn hipfire_runtime::spec::Speculator>,
    tokenizer: Tokenizer,
    template: String,
    state: String,
    /// The daemon-side `Gpu` the load was given; it outlives the model.
    _gpu: Gpu,
}

impl PpLoaded {
    fn load(model: &Path) -> Result<Self, String> {
        let path = model
            .to_str()
            .ok_or_else(|| format!("{} is not valid UTF-8", model.display()))?;
        let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
        let (state_request, _) = hw_identity::gdn_test_state_quant()?;
        let cask = hipfire_runtime::loader_api::CaskConfig::default();
        let mut spec = hipfire_runtime::loader_api::SpecLoadCfg::default();
        // The native head is attached on every arm so placement and reserves
        // are identical across the arms of one comparison.
        spec.mtp = Some(true);
        let mut loaded = hipfire_loader::load_model(
            path,
            MAX_SEQ,
            None,
            None,
            None,
            Some(state_request),
            &cask,
            PP_DEGREE,
            spec,
            &mut gpu,
        )?;
        if loaded.pp != PP_DEGREE {
            return Err(format!(
                "the product load resolved pp = {} but {TOPOLOGY_ENV}=pp2 needs {PP_DEGREE}",
                loaded.pp
            ));
        }
        let tokenizer = loaded.tokenizer.take().ok_or("the load has no tokenizer")?;
        let template = loaded
            .chat_template
            .take()
            .ok_or("the artifact has no chat template")?;
        let mut speculator = loaded
            .speculator
            .take()
            .ok_or("the load attached no speculator (is the native MTP head present?)")?;
        {
            let drafter = pp_drafter(speculator.as_mut())?;
            if !drafter.supports_temp_verify() {
                return Err("HIPFIRE_MTP_SAMPLED did not enable sampled verification".into());
            }
            if drafter.sampled_mode_is_naive() {
                return Err(
                    "HIPFIRE_MTP_SAMPLED_MODE=naive is refused under topology.pp; use leviathan"
                        .into(),
                );
            }
        }
        {
            let (bundle, _) = loaded
                .qwen4_mesh_mut()
                .ok_or("the PP2 load has no Qwen4 mesh bundle")?;
            if !bundle.pp_mesh_active() {
                return Err("the PP2 load has no active pipeline mesh".into());
            }
        }
        Ok(Self {
            model: loaded,
            speculator,
            tokenizer,
            template,
            state: format!("state_quant={state_request} (product loader, pp={PP_DEGREE})"),
            _gpu: gpu,
        })
    }

    fn parts(
        &mut self,
    ) -> Result<(&mut Qwen4Bundle, &mut hipfire_runtime::multi_gpu::Gpus), String> {
        self.model
            .qwen4_mesh_mut()
            .ok_or_else(|| "the PP2 model lost its Qwen4 mesh".to_string())
    }

    fn render(&self, text: &str) -> Result<Vec<u32>, String> {
        render_prompt(&self.tokenizer, &self.template, text, false)
    }

    /// One vocabulary-wide F32 logits row on the output band's device.
    fn alloc_logits(&mut self) -> Result<rdna_compute::GpuTensor, String> {
        let (bundle, gpus) = self.parts()?;
        let vocab = bundle.config.vocab_size;
        pp_owner(gpus)?
            .zeros(&[vocab], DType::F32)
            .map_err(|e| e.to_string())
    }

    fn free_logits(&mut self, logits: rdna_compute::GpuTensor) -> Result<(), String> {
        let (_, gpus) = self.parts()?;
        pp_owner(gpus)?
            .free_tensor(logits)
            .map_err(|e| e.to_string())
    }

    fn receipt(&mut self, model: &Path) -> Result<Value, String> {
        let (_, gpus) = self.parts()?;
        Ok(hw_identity::receipt(pp_owner(gpus)?, model))
    }

    /// [`Loaded::ar_ids`] on the mesh: seeded AR from a cold state with the
    /// production host sampler over the output band's logits row.
    fn ar_ids(
        &mut self,
        logits: &rdna_compute::GpuTensor,
        prompt: &[u32],
        cfg: &SamplerConfig,
        seed: u32,
        tokens: usize,
    ) -> Result<Vec<u32>, String> {
        let (bundle, gpus) = self.parts()?;
        let eos = bundle.config.eos_token_id;
        hipfire_runtime::llama::reset_cpu_sampler_rng(seed);
        // A cold prefill (`reused == 0`) resets every band.
        bundle
            .pp_prefill_final(gpus, prompt, 0, logits)
            .map_err(|e| e.to_string())?;
        let mut history = prompt.to_vec();
        let mut ids = Vec::with_capacity(tokens);
        loop {
            let mut row = pp_owner(gpus)?
                .download_f32(logits)
                .map_err(|e| e.to_string())?;
            let token = sample_cpu(&mut row, &history, cfg);
            ids.push(token);
            history.push(token);
            if ids.len() == tokens || token == eos {
                return Ok(ids);
            }
            bundle
                .pp_forward_token_or_argmax(gpus, Some(token), logits)
                .map_err(|e| e.to_string())?;
        }
    }

    /// [`Loaded::mtp_request`] on the mesh: the drafter's mesh entries with the
    /// model's `Gpus`, exactly as the daemon's spec loop drives them.
    fn mtp_request(
        &mut self,
        prompt: &[u32],
        cfg: SpecRequestConfig,
        tokens: usize,
    ) -> Result<MtpRun, String> {
        let (bundle, gpus) = self
            .model
            .qwen4_mesh_mut()
            .ok_or_else(|| "the PP2 model lost its Qwen4 mesh".to_string())?;
        let drafter = pp_drafter(self.speculator.as_mut())?;
        let eos = bundle.config.eos_token_id;
        bundle.pp_reset(gpus).map_err(|e| e.to_string())?;
        drafter.pp_mtp_reset(gpus, bundle)?;
        drafter.configure_request(cfg);
        let mut seed_token = drafter
            .pp_mtp_prefill(gpus, bundle, prompt, prompt, 0, false, &|| false)?
            .into_first_token()?;
        let mut ids = vec![seed_token];
        let (mut drafted, mut accepted) = (0, 0);
        let mut windows = Vec::new();
        while ids.len() < tokens && seed_token != eos {
            let position = bundle.state.position;
            let k = MTP_K.min(tokens - ids.len());
            let before = StatsSnap::of(&drafter.request_stats());
            let started = Counters::now();
            let window = drafter.pp_mtp_step(gpus, bundle, position, seed_token, &ids, k, eos)?;
            let counters = Counters::now().since(started);
            windows.push(WindowRec {
                index: ids.len(),
                position_before: position,
                position_after: bundle.state.position,
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

/// [`run_arm`] on a PP2 mesh: the same trials, seeds, shard files, config,
/// evidence and replay, with the PP load and the mesh decode loops. The load
/// is not freed (the arm is a fresh process that exits right after).
fn run_arm_pp(arm: &str, model: &Path, dir: &Path) -> Result<(), String> {
    let DistSettings {
        trials,
        temp,
        top_p,
        top_k,
        min_p,
        pen,
        fixture,
    } = DistSettings::from_env()?;
    if spec_request(temp, top_p, top_k, min_p, 0, &pen).penalties_active() {
        return Err(
            "the PP2 arms run neutral penalties only: penalized sampled MTP is refused under \
             topology.pp (the GPU penalty prepass refuses the mesh; such requests decode AR)"
                .into(),
        );
    }
    let env = own_env(&CONTROL_ENV);
    let settings_config = dist_config(
        arm,
        &own_identity(model, dir, &CONTROL_ENV)?,
        &pen,
        fixture,
        (temp, top_p, top_k, min_p),
        trials,
    );
    let (start, end) = shard_range(arm, trials)?;
    let sharded = (start, end) != (0, trials);
    let stem = if sharded {
        format!("{arm}.{start}-{end}")
    } else {
        arm.to_string()
    };
    let ids_path = dir.join(format!("{stem}.ids.json"));
    if ids_path.exists() {
        check_config(&ids_path, &settings_config)?;
    }
    let mut loaded = PpLoaded::load(model)?;
    let prompt = loaded.render(&fixture.text())?;
    fixture.check_prompt(prompt.len(), &pen)?;
    let eos = loaded.parts()?.0.config.eos_token_id;
    println!(
        "ARM {arm} topology=pp{PP_DEGREE} pp_layers={} prompt_tokens={} trials={trials} temp={temp} top_p={top_p} top_k={top_k:?} min_p={min_p} fixture={} state={} route_class={}",
        std::env::var("HIPFIRE_PP_LAYERS").unwrap_or_else(|_| "default".to_string()),
        prompt.len(),
        fixture.name(),
        loaded.state,
        hw_identity::route_class(&env)
    );
    println!("ARM {arm} config {settings_config}");
    let started = std::time::Instant::now();
    let mut samples: Vec<Vec<u32>> = Vec::with_capacity(end - start);
    if arm == "ar" {
        let logits = loaded.alloc_logits()?;
        let cfg = ar_sampler(temp, top_p, top_k, min_p, &pen);
        for trial in start..end {
            let seed = trial_seed(0xA11, trial) as u32;
            samples.push(loaded.ar_ids(&logits, &prompt, &cfg, seed, TOKENS)?);
        }
        loaded.free_logits(logits)?;
    } else {
        let budget = floor_probe_budget()?;
        let mut evidence = Evidence::with_samples(EVIDENCE_SAMPLES);
        for trial in start..end {
            let seed = trial_seed(0x5EC, trial);
            let cfg = spec_request(temp, top_p, top_k, min_p, seed, &pen);
            let result = loaded.mtp_request(&prompt, cfg, TOKENS)?;
            evidence.observe(
                trial,
                seed,
                &result,
                &loaded.tokenizer,
                budget,
                LATE_MARGINALS[0],
            );
            samples.push(result.ids);
        }
        // Only a forced floor-off route replays byte for byte; the adaptive
        // route's choices depend on wall-clock prices.
        if hw_identity::forced_floor_off(&env) {
            let replays = REPLAYS.min(end - start);
            for local in 0..replays {
                let trial = start + local;
                let cfg = spec_request(temp, top_p, top_k, min_p, trial_seed(0x5EC, trial), &pen);
                let replay = loaded.mtp_request(&prompt, cfg, TOKENS)?;
                if replay.ids != samples[local] {
                    return Err(format!(
                        "seed {} replayed {:?}, first run {:?}",
                        trial + 1,
                        replay.ids,
                        samples[local]
                    ));
                }
            }
            println!("ARM {arm} replayed {replays} seeds identically");
        } else {
            println!(
                "ARM {arm} replay not asserted (route class {}): its routing depends on wall-clock prices",
                hw_identity::route_class(&env)
            );
        }
        evidence.write(dir, &stem, budget, LATE_MARGINALS[0])?;
    }
    // Positions after an EOS read as EOS.
    for ids in &mut samples {
        ids.resize(TOKENS, eos);
    }
    println!(
        "ARM {arm} trials {start}..{end} in {:.1}s; first sample: {:?}",
        started.elapsed().as_secs_f64(),
        loaded.tokenizer.decode(&samples[0])
    );
    fs::write(
        &ids_path,
        serde_json::to_vec(&samples).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    write_config(&ids_path, &settings_config)?;
    let receipt = loaded.receipt(model)?;
    write_receipt(&ids_path, &receipt)
}

/// Joins the shards of each short arm (`<arm>.<START>-<END>.*` written by
/// `sampled_mtp_distribution_arm` under `HIPFIRE_SAMPLED_MTP_SHARD`) into the
/// whole-arm files the parent comparison reuses: ids concatenated in trial
/// order, the (identical) config and receipt of the first shard, evidence
/// counters summed and per-trial rows / texts concatenated. CPU only: no GPU,
/// no model read. Shards must cover `0..trials` exactly once and carry equal
/// configs; the parent still checks the merged config against the current
/// environment, so a stale or mixed shard set fails there.
#[test]
#[ignore = "CPU only: merges shards written by sampled_mtp_distribution_arm; needs HIPFIRE_SAMPLED_MTP_OUT"]
fn sampled_mtp_merge_short_shards() {
    let dir = out_dir();
    let trials: usize = env_or(TRIALS_ENV, 2000);
    for arm in ARMS {
        let shards = late_shards(&dir, arm).unwrap_or_else(|e| panic!("{e}"));
        assert!(!shards.is_empty(), "{arm}: no shards in {}", dir.display());
        let gaps = shard_gaps(&shards, trials).unwrap_or_else(|e| panic!("{arm}: {e}"));
        assert!(
            gaps.is_empty(),
            "{arm}: shards leave {gaps:?} of 0..{trials} uncovered"
        );
        let ids_file = |r: (usize, usize), ext: &str| late_file(&dir, arm, r, ext);
        let mut ids: Vec<Vec<u32>> = Vec::new();
        let config =
            read_json(&ids_file(shards[0], "ids.config.json")).unwrap_or_else(|e| panic!("{e}"));
        for &r in &shards {
            let c = read_json(&ids_file(r, "ids.config.json")).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(
                c, config,
                "{arm}: shard {r:?} was produced under a different config"
            );
            let part: Vec<Vec<u32>> =
                serde_json::from_slice(&fs::read(ids_file(r, "ids.json")).expect("shard ids"))
                    .expect("shard ids json");
            assert_eq!(
                part.len(),
                r.1 - r.0,
                "{arm}: shard {r:?} holds {} trials",
                part.len()
            );
            ids.extend(part);
        }
        assert_eq!(ids.len(), trials);
        let whole = dir.join(format!("{arm}.ids.json"));
        fs::write(&whole, serde_json::to_vec(&ids).unwrap()).unwrap();
        write_config(&whole, &config).unwrap_or_else(|e| panic!("{e}"));
        let receipt =
            read_json(&ids_file(shards[0], "ids.receipt.json")).unwrap_or_else(|e| panic!("{e}"));
        write_receipt(&whole, &receipt).unwrap_or_else(|e| panic!("{e}"));
        if arm != "ar" {
            let mut merged = read_json(&dir.join(format!(
                "{arm}.{}-{}.evidence.json",
                shards[0].0, shards[0].1
            )))
            .unwrap_or_else(|e| panic!("{e}"));
            let mut text = String::new();
            for (i, &r) in shards.iter().enumerate() {
                let stem = format!("{arm}.{}-{}", r.0, r.1);
                text.push_str(
                    &fs::read_to_string(dir.join(format!("{stem}.text.txt"))).unwrap_or_default(),
                );
                if i == 0 {
                    continue;
                }
                let e = read_json(&dir.join(format!("{stem}.evidence.json")))
                    .unwrap_or_else(|e| panic!("{e}"));
                for key in [
                    "trials",
                    "no_window_trials",
                    "calibration_ok_trials",
                    "retired_trials",
                    "full_trials",
                    "late_full_trials",
                    "mtp_windows",
                    "ar_windows",
                    "ngram_windows",
                    "steps",
                    "drafted",
                    "accepted",
                    "post_retirement_ar_steps",
                ] {
                    merged[key] =
                        json!(merged[key].as_u64().unwrap_or(0) + e[key].as_u64().unwrap_or(0));
                }
                for key in ["retire_steps", "per_trial"] {
                    let extra = e[key].as_array().cloned().unwrap_or_default();
                    merged[key].as_array_mut().expect(key).extend(extra);
                }
                let counters = merged["counters"].as_object_mut().expect("counters");
                for (name, total) in counters.iter_mut() {
                    *total = json!(
                        total.as_u64().unwrap_or(0) + e["counters"][name].as_u64().unwrap_or(0)
                    );
                }
                assert_eq!(
                    merged["probe_budget"], e["probe_budget"],
                    "{arm}: probe budget differs"
                );
                if merged["first_full_trial"].is_null() {
                    merged["first_full_trial"] = e["first_full_trial"].clone();
                }
            }
            assert_eq!(
                merged["trials"].as_u64(),
                Some(trials as u64),
                "{arm}: merged evidence trials"
            );
            write_json(&dir.join(format!("{arm}.evidence.json")), &merged)
                .unwrap_or_else(|e| panic!("{e}"));
            fs::write(dir.join(format!("{arm}.text.txt")), text).unwrap();
        }
        eprintln!(
            "{arm}: merged {} shards into {}",
            shards.len(),
            whole.display()
        );
    }
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
    let config = id_config(arm, &own_identity(model, dir, &CONTROL_ENV)?, &pen, fixture);
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
    let (state_request, _) = hw_identity::gdn_test_state_quant().unwrap_or_else(|e| panic!("{e}"));
    overrides.push((hw_identity::GDN_TEST_STATE_QUANT_ENV, Some(state_request)));
    overrides
}

/// A child (or a separately launched arm) must already be running under its
/// arm's environment: a stale shell variable is an error, not a silently
/// different configuration.
fn require_arm_env(arm: &str, mode: &'static str) -> Result<(), String> {
    // An unknown test mode (or q8_ef) is refused before the overrides below
    // resolve it.
    hw_identity::gdn_test_state_quant()?;
    // The topology switch is read by the four short leviathan arms only; every
    // other arm loads one card, so a stale pp2 setting would mislabel it.
    if (!ARMS.contains(&arm) || mode != "leviathan") && topology_from_env()? != Topology::Single {
        return Err(format!(
            "{TOPOLOGY_ENV} is set but arm {arm} (mode {mode}) runs on one card only: only the {ARMS:?} leviathan arms take a PP2 topology; unset {TOPOLOGY_ENV}"
        ));
    }
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

/// The `FLOOR_PROBE_WINDOWS` the binary under test was built with: the shared
/// economics module's constant, not a copy.
fn floor_probe_budget() -> Result<usize, String> {
    Ok(hipfire_runtime::spec_economics::FLOOR_PROBE_WINDOWS)
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
            .filter(|window| is_ar_window(window) && window.before.retired && window.after.retired)
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
    fn write(
        &self,
        dir: &Path,
        stem: &str,
        probe_budget: usize,
        late_from: usize,
    ) -> Result<(), String> {
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
fn short_arm_coverage(
    arm: &str,
    evidence: &Value,
    report: &mut String,
    coverage: &mut Vec<String>,
) {
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
    compare_positions(
        arm,
        reference,
        ar,
        mtp,
        &marginals,
        &[(0, 1)],
        report,
        failures,
    );
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
        "copy-long-distribution",
        arm,
        &settings.pen,
        Fixture::Prose,
        identity,
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
    check_config(
        &path,
        &copy_long_config(arm, identity, settings, arch, range)?,
    )
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
            COPY_LONG_ARM_TEST,
            arm,
            "leviathan",
            model,
            dir,
            Some(gap),
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
        parts
            .iter()
            .map(|part| part[key].as_u64().expect("long-copy stats count") as usize)
            .sum()
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
    let positions = COPY_LONG_MARGINALS
        .iter()
        .map(|&first| (first, None))
        .chain(
            COPY_LONG_JOINTS
                .iter()
                .map(|&(first, second)| (first, Some(second))),
        );
    for (first, second) in positions {
        let cells = |samples: &[Vec<u32>]| -> Vec<(u32, Option<u32>)> {
            samples
                .iter()
                .map(|ids| (ids[first], second.map(|index| ids[index])))
                .collect()
        };
        let (stat, df, p) = homogeneity(&cells(ar), &cells(mtp));
        let label = second.map_or_else(
            || format!("position {first}"),
            |second| format!("joint ({first},{second})"),
        );
        let line = format!(
            "{arm} vs copy_long_ar, {label}: chi2={stat:.2} df={df} p={p:.6} (n={}+{})\n",
            ar.len(),
            mtp.len(),
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
    for name in raw
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
    {
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
            let ids: Vec<Vec<u32>> =
                serde_json::from_value(read_json(&path).unwrap_or_else(|e| panic!("{e}")))
                    .expect("long-copy shard ids");
            assert_eq!(
                ids.len(),
                range.1 - range.0,
                "{arm}: shard {range:?} trials"
            );
            assert!(
                ids.iter().all(|ids| ids.len() == COPY_LONG_TOKENS),
                "{arm}: shard {range:?} generated length"
            );
            all.extend(ids);
            if arm != COPY_LONG_ARMS[0] {
                let part = read_json(&late_file(&dir, arm, range, "stats.json"))
                    .unwrap_or_else(|e| panic!("{e}"));
                assert_eq!(
                    part["trials"].as_u64(),
                    Some((range.1 - range.0) as u64),
                    "{arm}: shard {range:?} stats trials"
                );
                // Require the corresponding per-window evidence, not IDs alone.
                let evidence = read_json(&late_file(&dir, arm, range, "evidence.json"))
                    .unwrap_or_else(|e| panic!("{e}"));
                assert_eq!(evidence["trials"], part["trials"], "{arm}: evidence trials");
                assert_eq!(
                    evidence["ngram_windows"], part["ngram_mod_windows"],
                    "{arm}: evidence n-gram windows"
                );
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
            arm,
            &samples[COPY_LONG_ARMS[0]],
            &samples[arm],
            &mut report,
            &mut failures,
        );
        coverage.extend(copy_long_coverage(arm, &stats[arm], &mut report));
    }
    eprint!("{report}");
    fs::write(dir.join("copy-long-report.txt"), &report).expect("long-copy report");
    assert!(
        failures.is_empty() && coverage.is_empty(),
        "{}",
        failure_message(&dir, &failures, &coverage)
    );
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
            ngram_config(
                name,
                spec,
                &identity,
                sampling,
                &settings.pen,
                trials,
                copy_lane,
            )
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
            coverage.push(format!(
                "{line}  the copied-span fixture produced no n-gram hits\n"
            ));
        } else if !arm.copy && native == 0 {
            coverage.push(format!(
                "{line}  no native MTP window ran on the prose prompt\n"
            ));
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
                .ok_or_else(|| {
                    format!("trial {trial}: ngram_mod_windows went backwards in a window")
                })?;
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
            evidence.observe(
                trial,
                seed,
                &result,
                &loaded.tokenizer,
                budget,
                LATE_MARGINALS[0],
            );
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
        write_json(
            &late_file(dir, arm, range, "stats.json"),
            &stats.to_json(range),
        )?;
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
            let (mut windows, mut drafts, mut accepted, mut native) =
                (0usize, 0usize, 0usize, 0usize);
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
                evidence.observe(
                    trial,
                    seed,
                    &result,
                    &loaded.tokenizer,
                    budget,
                    LATE_MARGINALS[0],
                );
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
            assert_eq!(
                part.len(),
                range.1 - range.0,
                "{arm} shard {range:?} trial count"
            );
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
            evidence.observe(
                trial,
                seed,
                &result,
                &loaded.tokenizer,
                budget,
                LATE_MARGINALS[0],
            );
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

    /// Every arm's child is given the resolved GDN test mode explicitly.
    #[test]
    fn every_arm_forwards_the_gdn_test_mode() {
        let (request, _) = hw_identity::gdn_test_state_quant().unwrap();
        for arm in ["ar", "batched", "interleaved", "adaptive"] {
            let forwarded: Vec<_> = arm_overrides(arm, "leviathan")
                .into_iter()
                .filter(|(name, _)| *name == hw_identity::GDN_TEST_STATE_QUANT_ENV)
                .collect();
            assert_eq!(
                forwarded,
                vec![(hw_identity::GDN_TEST_STATE_QUANT_ENV, Some(request))],
                "{arm}"
            );
        }
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
        for arm in [
            "ar",
            "adaptive",
            "ngram_adaptive",
            LATE_ADAPTIVE_ARM,
            LATE_PILOT_ARM,
        ] {
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
                (
                    loaded.ar_ids(logits, &prompt, &cfg, seed as u32, tokens)?,
                    seed,
                )
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

// ---------------------------------------------------------------------------
// q8_ef verify-law case (PLAN G5)
// ---------------------------------------------------------------------------
//
// `sampled_mtp_q8ef_matches_verify_distribution_on_flash_next` checks sampled
// native-MTP against the Leviathan law applied to the ACTUAL verify-route
// target rows of the same route / reset history (live-only q8_ef zeroes the
// residual on every rollback and restore, so the target law of a window
// depends on that history and is not uninterrupted AR's). It runs under
// `--features lab`, which compiles the read-only window tap
// (`hipfire_arch_qwen4::lab_window`); without it the case refuses.
//
// # What is observed and exactly what is tested
//
// Per `mtp_step` the tap reports every proposal (rejected ones included) with
// the row `q` it was drawn from, the transformed target rows `p_j` the
// acceptance read (host and device route alike), each verdict's uniforms, the
// terminal draw, the GDN residual-reset counters and the ring half. The case
// feeds each window through these checks, none of which uses a quantity the
// law conditions on in the wrong direction:
//
// 0. Independent target rows. The case arms the forward-logits tap with
//    `keep_logits`, which records the RAW model logits of every verify
//    forward before the penalty prepass and every policy stage. For each row
//    the acceptance read, it rebuilds the target law itself: the history
//    `suffix_W(prompt || emitted || drafts[..j])`, the request's penalties
//    and blocked tokens (`apply_logit_policy_cpu` over the raw row), then
//    temperature / top_k / top_p / min_p (`SparseDist::build_from_logits`
//    under `SampleSpec::cpu_ar`), and requires the production row to equal
//    it (bit-exact on the host route, 1e-6 on a device-read row). A row with
//    no raw logits (tap unarmed, no forward matching the block) is an error,
//    never accepted on the production row alone. The proposal row `q` is the
//    one quantity not rebuilt (the draft head's logits are not tapped); its
//    law is established by items 1 to 3. The penalty history is not
//    recorded by the tap: the case reconstructs it from the prompt, the
//    committed stream and the window's drafts, and the equality with the
//    production row (already penalized) is what validates it; with active
//    penalties a wrong history changes the row and fails the window.
// 1. Window replay. The independent Leviathan replay (`q8ef_replay`: accept
//    `x` iff `q(x) > 0 && u q(x) < p(x)`, else the replacement from
//    `(p - q)+`, else the bonus from the last row) over the window's OWN
//    recorded rows and uniforms must reproduce the production committed IDs,
//    accepted count and uniform consumption; every draft must be the
//    inverse-CDF draw of its proposal row at the uniform recorded for it.
// 2. Exact one-step mixture (`q8ef_mixture_gap`). For every recorded row
//    pair `(p_j, q_j)` the output law of one Leviathan step,
//    `min(p, q) + (1 - sum min(p, q)) (p - q)+ / sum (p - q)+`, equals `p_j`
//    (to 1e-4): the accept / reject / residual correction restores `p_j`
//    from ANY proposal row, rejected proposals included.
// 3. Proposal law. Every drafted token, verified or not, has a PIT under its
//    own `q_j` that is Uniform(0,1) (`draft_pits`, by draft slot): the
//    drafts form an autoregressive chain independent of the verdicts.
// 4. Output law. The committed token at stream index `n` is a draw from the
//    target row `p_j` of the verify row that produced it, conditioned on the
//    committed history ONLY. This holds because the committed prefix equals
//    the verify block's prefix (accepted drafts ARE the committed tokens),
//    row `j` of a causal verify depends only on block tokens `0..=j`, and the
//    Leviathan theorem makes the token emitted at each stream position
//    `p_j`-distributed given the history. The PIT of every emitted token
//    under its own `p_j` (and the category counts against the per-trial
//    expectation `sum_i p_i(c)`) are therefore history-valid whatever the
//    history. They are NOT conditioned on the window's own accept/reject
//    verdict, on any rejected draft or on any uniform. Validity needs the
//    recorded `p_j` to be a function of the committed history only: if a
//    future verify batching made a row depend on later drafts, the tapped
//    rows would no longer be the law of the committed token and this check
//    would be invalid, so the case records the verify route and refuses
//    (error) any window whose rows the tap cannot attribute (a row gap, an
//    unmeasured draw offset, a non-finite logit or a tap error).
// 5. Stream uniforms. Every uniform of the window's request stream is
//    Uniform(0,1) (histogram, `uniform_hist`).
//
// Conditioning on the seed. Stream index 0 is the prefill draw of
// `mtp_prefill` (the request's seed token), outside every window and not
// observed by the tap. It is the conditioning seed of the first window: every
// statistic here is the law of stream indices >= 1 GIVEN the realized seed
// token and prompt. The law of the seed draw itself is not checked by this
// case (the receipt lists it as unobserved).
//
// # Coverage and completeness
//
// Required fixed positions (`Q8EF_MARGINALS`, `Q8EF_JOINTS`) must be observed
// in EVERY trial of the declared cohort: a position with fewer observations
// than trials (an EOS before it, an untapped window) is INCOMPLETE, never
// judged on the survivors and never a PASS. Coverage items (keep0 / partial /
// full, windows after a reset, ring halves, graph replay, forced batched /
// interleaved / adaptive, session / prefix / disk restore then sample,
// same-seed replay) each carry an observed count and a required count; any
// short item is INCOMPLETE. A law failure (p-value at or under `ALPHA`) is a
// FAIL whatever else is incomplete. Seeds are `trial_seed(salt, trial)` over
// global trial indices, shards compose, nothing is selected or retried.
//
// Restore arms. `session` restores the resident snapshot of the previous
// prompt; `prefix` restores the pool snapshot chain after a reset; `disk`
// demotes every snapshot to the disk tier before each trial
// (`Qwen4Bundle::session_clear`) so the plan can only be served from disk,
// after a publish process that cold-prefills the prompt and unloads. All
// three run the full declared cohort, are judged like the route arms, and
// every restoring prefill must zero the live residual (a reset-counter
// increase). The disk arm must also see its snapshot files vanish (promoted)
// during the restore.

const Q8EF_LAW_TEST: &str = "sampled_mtp_q8ef_matches_verify_distribution_on_flash_next";
const Q8EF_ARM_TEST: &str = "sampled_mtp_q8ef_arm";
const Q8EF_RECEIPT: &str = "q8ef_verify_law.coverage.json";
const Q8EF_MIN_TRIALS: usize = 2000;
/// Output IDs per trial: enough windows (K = `MTP_K`) for resets to precede
/// later sampled windows; the four cold IDs of the AR comparison are not.
const Q8EF_TOKENS: usize = 32;
const Q8EF_ROUTE_ARMS: [&str; 3] = ["q8ef_batched", "q8ef_interleaved", "q8ef_adaptive"];
const Q8EF_RESTORE_ARMS: [&str; 3] = [
    "q8ef_session_batched",
    "q8ef_prefix_batched",
    "q8ef_disk_batched",
];
const Q8EF_DISK_ARM: &str = "q8ef_disk_batched";
const Q8EF_STAGE_ENV: &str = "HIPFIRE_SAMPLED_MTP_Q8EF_STAGE";
/// Harness-control variables of the q8_ef arms (identity excludes them).
const Q8EF_CONTROL_ENV: [&str; 5] = [MODEL_ENV, OUT_ENV, ARM_ENV, SHARD_ENV, Q8EF_STAGE_ENV];
/// Positions of the category / PIT marginals (index 0 is the prefill draw, not
/// a verify row) and of the joint pairs.
const Q8EF_MARGINALS: [usize; 7] = [1, 2, 3, 4, 8, 16, 31];
const Q8EF_JOINTS: [(usize, usize); 4] = [(1, 2), (8, 9), (16, 17), (30, 31)];
const Q8EF_PIT_BINS: usize = 20;
const Q8EF_JOINT_GRID: usize = 4;
const Q8EF_MIN_EXPECTED: f64 = 5.0;
/// A coverage class needs at least this many observed windows in every arm
/// that must show it.
const Q8EF_MIN_WINDOWS: u64 = 10;
/// Salt of the PIT's auxiliary uniforms: a stream of its own, independent of
/// every production stream.
const Q8EF_PIT_SALT: u64 = 0x9178_E8EF_0001;
/// Tolerance of the exact one-step mixture (f32 rows).
const Q8EF_MIXTURE_TOL: f64 = 1e-4;
/// Context of every q8_ef arm: the 16384-token product context of the EF
/// plan, not the 4096 of the older arms; sized so a restore prompt of
/// `2 * chunk + 100` rows fits for any prefill chunk up to 8192.
const Q8EF_MAX_SEQ: usize = 16384;

/// The case's configuration contract: >= 2000 trials per arm and the q8_ef
/// mode (the law of any other mode is the AR comparison's, not this case's).
fn q8ef_check_plan(trials: usize, mode: &str) -> Result<(), String> {
    if mode != "q8_ef" {
        return Err(format!(
            "{Q8EF_LAW_TEST} runs under {}=q8_ef only (got {mode})",
            hw_identity::GDN_TEST_STATE_QUANT_ENV
        ));
    }
    if trials < Q8EF_MIN_TRIALS {
        return Err(format!(
            "{Q8EF_LAW_TEST} needs at least {Q8EF_MIN_TRIALS} trials per arm ({TRIALS_ENV}={trials})"
        ));
    }
    Ok(())
}

/// Trials of every arm: the route arms and every restore arm (session, prefix
/// and disk) run the same declared cohort of `HIPFIRE_SAMPLED_MTP_TRIALS`
/// (default and minimum 2000); no arm is shrunk.
struct Q8efCohort {
    trials: usize,
}

impl Q8efCohort {
    fn from_env() -> Result<Self, String> {
        Ok(Self {
            trials: strict_env(TRIALS_ENV, Q8EF_MIN_TRIALS)?,
        })
    }

    fn total(&self, _arm: &str) -> usize {
        self.trials
    }
}

fn q8ef_all_arms() -> impl Iterator<Item = &'static str> {
    Q8EF_ROUTE_ARMS.into_iter().chain(Q8EF_RESTORE_ARMS)
}

fn q8ef_salt(arm: &str) -> Result<u64, String> {
    Ok(match arm {
        "q8ef_batched" => 0xE8F1,
        "q8ef_interleaved" => 0xE8F2,
        "q8ef_adaptive" => 0xE8F3,
        "q8ef_session_batched" => 0xE8F4,
        "q8ef_prefix_batched" => 0xE8F5,
        Q8EF_DISK_ARM => 0xE8F6,
        other => return Err(format!("{other} is not a q8_ef arm")),
    })
}

/// Sum of a sparse target row's probabilities.
fn q8ef_row_mass(entries: &[(u32, f32)]) -> Result<f64, String> {
    let mut mass = 0.0f64;
    for &(token, prob) in entries {
        if !prob.is_finite() || prob < 0.0 {
            return Err(format!(
                "target row has probability {prob} for token {token}"
            ));
        }
        mass += f64::from(prob);
    }
    if mass > 0.0 && mass.is_finite() {
        Ok(mass)
    } else {
        Err("target row has no probability mass".into())
    }
}

/// Randomized probability-integral transform of `token` under its own row, in
/// the row's entry order: `(F(t^-) + u p(t)) / mass`. Uniform on [0, 1) when
/// `token` was drawn from that row and `u` is an independent uniform in
/// [0, 1), whatever the row (so whatever the history). A token outside the
/// row's support is an error, never clamped.
fn q8ef_pit(entries: &[(u32, f32)], token: u32, u: f64) -> Result<f64, String> {
    if !(0.0..1.0).contains(&u) {
        return Err(format!("PIT uniform {u} is outside [0, 1)"));
    }
    let mass = q8ef_row_mass(entries)?;
    let mut below = 0.0f64;
    for &(entry, prob) in entries {
        if entry == token {
            if prob <= 0.0 {
                return Err(format!("token {token} has zero target probability"));
            }
            return Ok((below + u * f64::from(prob)) / mass);
        }
        below += f64::from(prob);
    }
    Err(format!(
        "token {token} is outside the target row's support ({} entries)",
        entries.len()
    ))
}

fn q8ef_bin(x: f64, bins: usize) -> usize {
    ((x * bins as f64) as usize).min(bins - 1)
}

/// Chi-square of PIT values against Uniform(0, 1) in `bins` equal bins.
/// Returns `(statistic, degrees of freedom, p-value)`.
fn q8ef_pit_uniform(pits: &[f64], bins: usize) -> Result<(f64, usize, f64), String> {
    let expected = pits.len() as f64 / bins as f64;
    if bins < 2 || expected < Q8EF_MIN_EXPECTED {
        return Err(format!(
            "{} PIT values give {expected:.2} expected per bin over {bins} bins (< {Q8EF_MIN_EXPECTED})",
            pits.len()
        ));
    }
    let mut counts = vec![0.0f64; bins];
    for &x in pits {
        if !x.is_finite() || !(0.0..=1.0).contains(&x) {
            return Err(format!("PIT value {x} is outside [0, 1]"));
        }
        counts[q8ef_bin(x, bins)] += 1.0;
    }
    let stat: f64 = counts
        .iter()
        .map(|c| (c - expected).powi(2) / expected)
        .sum();
    let df = bins - 1;
    Ok((stat, df, gamma_q(df as f64 / 2.0, stat / 2.0)))
}

/// Chi-square of the PIT pairs of two positions of one trial against the
/// uniform distribution on the unit square (`grid` x `grid` cells): uniform
/// marginals AND independence, the history-valid joint test.
fn q8ef_pit_joint(pairs: &[(f64, f64)], grid: usize) -> Result<(f64, usize, f64), String> {
    let cells = grid * grid;
    let expected = pairs.len() as f64 / cells as f64;
    if grid < 2 || expected < Q8EF_MIN_EXPECTED {
        return Err(format!(
            "{} PIT pairs give {expected:.2} expected per cell over {cells} cells (< {Q8EF_MIN_EXPECTED})",
            pairs.len()
        ));
    }
    let mut counts = vec![0.0f64; cells];
    for &(a, b) in pairs {
        for x in [a, b] {
            if !x.is_finite() || !(0.0..=1.0).contains(&x) {
                return Err(format!("PIT value {x} is outside [0, 1]"));
            }
        }
        counts[q8ef_bin(a, grid) * grid + q8ef_bin(b, grid)] += 1.0;
    }
    let stat: f64 = counts
        .iter()
        .map(|c| (c - expected).powi(2) / expected)
        .sum();
    let df = cells - 1;
    Ok((stat, df, gamma_q(df as f64 / 2.0, stat / 2.0)))
}

/// Solves `a x = b` (row-major `n` x `n`) by Gaussian elimination with
/// partial pivoting; a pivot below `1e-9` of the largest diagonal is singular.
fn q8ef_solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Result<Vec<f64>, String> {
    let scale = (0..n).map(|i| a[i * n + i].abs()).fold(0.0f64, f64::max);
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&x, &y| a[x * n + col].abs().total_cmp(&a[y * n + col].abs()))
            .ok_or("empty covariance")?;
        if !(a[pivot * n + col].abs() > 1e-9 * scale) {
            return Err("category covariance is singular (rows without sampling variance)".into());
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
            }
            b.swap(col, pivot);
        }
        for row in col + 1..n {
            let factor = a[row * n + col] / a[col * n + col];
            if factor != 0.0 {
                for k in col..n {
                    a[row * n + k] -= factor * a[col * n + k];
                }
                b[row] -= factor * b[col];
            }
        }
    }
    let mut x = vec![0.0f64; n];
    for row in (0..n).rev() {
        let tail: f64 = (row + 1..n).map(|k| a[row * n + k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row * n + row];
    }
    Ok(x)
}

/// Category chi-square of emitted tokens against their OWN rows. `rows[i]` is
/// trial `i`'s target row (its history's law) and the token it emitted. The
/// expected count of a category is `sum_i p_i(c)`; the covariance of the
/// observed counts under the null is `sum_i (diag(p_i) - p_i p_i^T)`, so the
/// statistic is the Mahalanobis form `d^T S^-1 d` over all cells but the most
/// probable one, chi-square with `cells - 1` degrees of freedom. Tokens whose
/// expected count is under 5 are pooled into one rest cell (merged into the
/// most probable cell when even the rest is under 5). Returns `(statistic,
/// degrees of freedom, p-value)`.
fn q8ef_category_chi_square(rows: &[(&[(u32, f32)], u32)]) -> Result<(f64, usize, f64), String> {
    let mut normalized: Vec<Vec<(u32, f64)>> = Vec::with_capacity(rows.len());
    let mut expected: HashMap<u32, f64> = HashMap::new();
    for &(entries, token) in rows {
        let mass = q8ef_row_mass(entries)?;
        if !entries.iter().any(|&(t, p)| t == token && p > 0.0) {
            return Err(format!("token {token} is outside its target row's support"));
        }
        let row: Vec<(u32, f64)> = entries
            .iter()
            .filter(|&&(_, p)| p > 0.0)
            .map(|&(t, p)| (t, f64::from(p) / mass))
            .collect();
        for &(t, p) in &row {
            *expected.entry(t).or_default() += p;
        }
        normalized.push(row);
    }
    let mut kept: Vec<u32> = expected
        .iter()
        .filter(|&(_, &e)| e >= Q8EF_MIN_EXPECTED)
        .map(|(&t, _)| t)
        .collect();
    kept.sort_unstable();
    if kept.is_empty() {
        return Err(format!(
            "no token reaches {Q8EF_MIN_EXPECTED} expected draws"
        ));
    }
    let index: HashMap<u32, usize> = kept.iter().enumerate().map(|(i, &t)| (t, i)).collect();
    let rest_expected: f64 = expected
        .iter()
        .filter(|&(t, _)| !index.contains_key(t))
        .map(|(_, &e)| e)
        .sum();
    let rest_separate = rest_expected >= Q8EF_MIN_EXPECTED;
    let cells = kept.len() + usize::from(rest_separate);
    if cells < 2 {
        return Err("fewer than two cells: nothing to test".into());
    }
    let heaviest = (0..kept.len())
        .max_by(|&x, &y| {
            expected[&kept[x]]
                .total_cmp(&expected[&kept[y]])
                .then(y.cmp(&x))
        })
        .ok_or("no cells")?;
    let rest_cell = if rest_separate { kept.len() } else { heaviest };
    let cell_of = |token: u32| index.get(&token).copied().unwrap_or(rest_cell);
    let mut observed = vec![0.0f64; cells];
    let mut mean = vec![0.0f64; cells];
    let mut cov = vec![0.0f64; cells * cells];
    for (&(_, token), row) in rows.iter().zip(&normalized) {
        observed[cell_of(token)] += 1.0;
        let mut pi: Vec<(usize, f64)> = Vec::new();
        for &(t, p) in row {
            let cell = cell_of(t);
            match pi.iter_mut().find(|(c, _)| *c == cell) {
                Some(slot) => slot.1 += p,
                None => pi.push((cell, p)),
            }
        }
        for &(c, p) in &pi {
            mean[c] += p;
            cov[c * cells + c] += p;
            for &(d, q) in &pi {
                cov[c * cells + d] -= p * q;
            }
        }
    }
    let reference = (0..cells)
        .max_by(|&x, &y| mean[x].total_cmp(&mean[y]).then(y.cmp(&x)))
        .ok_or("no cells")?;
    let kept_cells: Vec<usize> = (0..cells).filter(|&c| c != reference).collect();
    let m = kept_cells.len();
    let mut a = vec![0.0f64; m * m];
    for (i, &c) in kept_cells.iter().enumerate() {
        for (j, &d) in kept_cells.iter().enumerate() {
            a[i * m + j] = cov[c * cells + d];
        }
    }
    let delta: Vec<f64> = kept_cells.iter().map(|&c| observed[c] - mean[c]).collect();
    let solved = q8ef_solve(a, delta.clone(), m)?;
    let stat: f64 = delta.iter().zip(&solved).map(|(d, x)| d * x).sum();
    Ok((stat, m, gamma_q(m as f64 / 2.0, stat / 2.0)))
}

// ---------------------------------------------------------------------------
// q8_ef verify-law case: window observations and the independent law replay
// ---------------------------------------------------------------------------

/// One native-MTP window as the lab tap reports it.
#[derive(Clone, Debug, PartialEq)]
struct Q8efWindow {
    /// Every drafted ID of the window (rejected ones included).
    drafts: Vec<u32>,
    /// Transformed target row of every verify row the acceptance read
    /// (`SparseDist::entries()` order), indexed by verify row.
    target_rows: Vec<Vec<(u32, f32)>>,
    /// The proposal row each draft was drawn from (a point mass is the row
    /// `[(draft, 1.0)]`).
    q_rows: Vec<Vec<(u32, f32)>>,
    /// The uniforms of the verdicts in consumption order: one per accept,
    /// two per reject (verdict, residual), one for the terminal bonus.
    uniforms: Vec<f32>,
    /// Every uniform the window drew from the request stream.
    stream_uniforms: Vec<f32>,
    /// What the window committed (accepted drafts plus the terminal token).
    committed: Vec<u32>,
    accepted: usize,
    eos: Option<u32>,
    /// `Qwen4State::gdn_residual_resets` before and after the window.
    resets: (u64, u64),
    /// Ring half the armed verify wrote, when the window ran one.
    ring_half: Option<u8>,
    /// hipGraph launches during the window.
    graph_launches: u64,
    /// `batched`, `interleaved` or `target_only`.
    route: String,
    device: bool,
    /// Target rows rebuilt from raw logits and equal to `target_rows`.
    independent_rows: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Q8efKeep {
    /// A target-only window: nothing was drafted.
    NoDrafts,
    /// Every draft rejected: the pre-window Q8 state is restored.
    Keep0,
    Partial,
    Full,
}

impl Q8efKeep {
    fn slot(self) -> usize {
        match self {
            Self::NoDrafts => 0,
            Self::Keep0 => 1,
            Self::Partial => 2,
            Self::Full => 3,
        }
    }
}

impl Q8efWindow {
    fn keep(&self) -> Q8efKeep {
        if self.drafts.is_empty() {
            Q8efKeep::NoDrafts
        } else if self.accepted == 0 {
            Q8efKeep::Keep0
        } else if self.accepted == self.drafts.len() {
            Q8efKeep::Full
        } else {
            Q8efKeep::Partial
        }
    }

    fn reset_delta(&self) -> u64 {
        self.resets.1.saturating_sub(self.resets.0)
    }

    /// What a same-route replay of the same seed must reproduce exactly:
    /// everything except the absolute reset counters and the graph launch
    /// count (first run captures, the replay launches).
    fn replay_key(&self) -> Self {
        Self {
            resets: (0, self.reset_delta()),
            graph_launches: 0,
            ..self.clone()
        }
    }
}

fn q8ef_prob(entries: &[(u32, f32)], token: u32) -> f32 {
    entries
        .iter()
        .find(|&&(t, _)| t == token)
        .map_or(0.0, |&(_, p)| p)
}

/// Inverse-CDF draw in entry order (the law's own `SparseDist::sample`).
fn q8ef_sample(entries: &[(u32, f32)], u: f32) -> u32 {
    let mut acc = 0.0f32;
    for &(token, p) in entries {
        acc += p;
        if u < acc {
            return token;
        }
    }
    entries.last().map_or(0, |&(token, _)| token)
}

/// Draw from the normalized residual `(p - q)+` (the law's `sample_residual`).
fn q8ef_residual(p: &[(u32, f32)], q: &[(u32, f32)], u: f32) -> u32 {
    let residual = |&(token, pt): &(u32, f32)| (token, (pt - q8ef_prob(q, token)).max(0.0));
    let mass: f32 = p.iter().map(|e| residual(e).1).sum();
    if !(mass > 0.0) {
        return q8ef_sample(p, u);
    }
    let target = u * mass;
    let mut acc = 0.0f32;
    let mut last = None;
    for (token, r) in p.iter().map(residual) {
        if r <= 0.0 {
            continue;
        }
        acc += r;
        if target < acc {
            return token;
        }
        last = Some(token);
    }
    last.unwrap_or_else(|| q8ef_sample(p, u))
}

fn q8ef_tape(w: &Q8efWindow, at: &mut usize) -> Result<f32, String> {
    let u = *w.uniforms.get(*at).ok_or_else(|| {
        format!(
            "the acceptance replay needs more than the {} recorded uniforms",
            w.uniforms.len()
        )
    })?;
    *at += 1;
    Ok(u)
}

/// Independent Leviathan replay of one window over its OWN recorded target
/// rows, proposal rows and uniforms: accept draft `x` iff `q(x) > 0` and
/// `u q(x) < p(x)`, else draw the replacement from `(p - q)+`; a fully
/// accepted window (without an accepted EOS) draws the bonus from the last
/// row. Returns `(committed, accepted, uniforms consumed)`.
fn q8ef_replay(w: &Q8efWindow) -> Result<(Vec<u32>, usize, usize), String> {
    if w.q_rows.len() != w.drafts.len() {
        return Err(format!(
            "{} proposal rows for {} drafts",
            w.q_rows.len(),
            w.drafts.len()
        ));
    }
    fn target_row(w: &Q8efWindow, j: usize) -> Result<&[(u32, f32)], String> {
        match w.target_rows.get(j) {
            Some(row) => Ok(row.as_slice()),
            None => Err(format!(
                "no recorded target row {j} ({} rows)",
                w.target_rows.len()
            )),
        }
    }
    let mut at = 0usize;
    let mut committed = Vec::with_capacity(w.drafts.len() + 1);
    let mut accepted = 0usize;
    for (j, (&draft, q)) in w.drafts.iter().zip(&w.q_rows).enumerate() {
        let p = target_row(w, j)?;
        let (qx, px) = (q8ef_prob(q, draft), q8ef_prob(p, draft));
        let u = q8ef_tape(w, &mut at)?;
        if qx > 0.0 && u * qx < px {
            committed.push(draft);
            accepted += 1;
            if w.eos == Some(draft) {
                return Ok((committed, accepted, at));
            }
        } else {
            let u = q8ef_tape(w, &mut at)?;
            committed.push(q8ef_residual(p, q, u));
            return Ok((committed, accepted, at));
        }
    }
    let p = target_row(w, w.drafts.len())?;
    let u = q8ef_tape(w, &mut at)?;
    committed.push(q8ef_sample(p, u));
    Ok((committed, accepted, at))
}

/// Largest `|out(x) - p(x)|` of the exact one-step Leviathan output law
/// `out(x) = min(p, q)(x) + (1 - sum min(p, q)) (p - q)+(x) / sum (p - q)+`
/// over the union support: zero (to rounding) for any pair of normalized rows.
fn q8ef_mixture_gap(p: &[(u32, f32)], q: &[(u32, f32)]) -> f64 {
    let mut tokens: Vec<u32> = p.iter().chain(q).map(|&(t, _)| t).collect();
    tokens.sort_unstable();
    tokens.dedup();
    let at = |row: &[(u32, f32)], t: u32| f64::from(q8ef_prob(row, t));
    let accept: f64 = tokens.iter().map(|&t| at(p, t).min(at(q, t))).sum();
    let excess: f64 = tokens.iter().map(|&t| (at(p, t) - at(q, t)).max(0.0)).sum();
    let reject = 1.0 - accept;
    tokens
        .iter()
        .map(|&t| {
            let residual = if excess > 0.0 {
                (at(p, t) - at(q, t)).max(0.0) / excess
            } else {
                0.0
            };
            (at(p, t).min(at(q, t)) + reject * residual - at(p, t)).abs()
        })
        .fold(0.0, f64::max)
}

/// A window passes when its recorded rows are valid distributions, every
/// draft lies in its proposal's support, each recorded `(p_j, q_j)` pair
/// satisfies the exact Leviathan mixture and the independent replay
/// reproduces the production verdict (committed IDs, accepted count,
/// uniforms consumed).
fn q8ef_check_window(w: &Q8efWindow) -> Result<(), String> {
    for (j, row) in w.target_rows.iter().enumerate() {
        q8ef_row_mass(row).map_err(|e| format!("target row {j}: {e}"))?;
    }
    for (j, (&draft, row)) in w.drafts.iter().zip(&w.q_rows).enumerate() {
        q8ef_row_mass(row).map_err(|e| format!("proposal row {j}: {e}"))?;
        if !(q8ef_prob(row, draft) > 0.0) {
            return Err(format!(
                "draft {j} ({draft}) lies outside its proposal's support"
            ));
        }
    }
    for (j, (p, q)) in w.target_rows.iter().zip(&w.q_rows).enumerate() {
        let gap = q8ef_mixture_gap(p, q);
        if !(gap <= Q8EF_MIXTURE_TOL) {
            return Err(format!(
                "the Leviathan mixture of rows {j} misses its target by {gap:.3e} (> {Q8EF_MIXTURE_TOL})"
            ));
        }
    }
    let (committed, accepted, used) = q8ef_replay(w)?;
    if committed != w.committed || accepted != w.accepted {
        return Err(format!(
            "production committed {:?} (accepted {}) but the independent replay of its own rows gives {:?} (accepted {})",
            w.committed, w.accepted, committed, accepted
        ));
    }
    if used != w.uniforms.len() {
        return Err(format!(
            "the replay consumed {used} uniforms, the window recorded {}",
            w.uniforms.len()
        ));
    }
    Ok(())
}

/// `[0, 1)` uniform of the PIT for trial `trial`, stream index `index`.
fn q8ef_pit_draw(salt: u64, trial: usize, index: usize) -> f64 {
    let z = trial_seed(salt ^ ((index as u64) << 40), trial);
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Where a trial stands between windows.
struct Q8efTrial {
    trial: usize,
    /// Stream index of the next committed token (index 0 is the prefill draw).
    next_index: usize,
    prev_keep: Option<Q8efKeep>,
    prev_reset: bool,
}

impl Q8efTrial {
    fn new(trial: usize) -> Self {
        Self {
            trial,
            next_index: 1,
            prev_keep: None,
            prev_reset: false,
        }
    }
}

fn q8ef_recorded(index: usize) -> bool {
    Q8EF_MARGINALS.contains(&index) || Q8EF_JOINTS.iter().any(|&(a, b)| a == index || b == index)
}

/// Everything one arm's windows contribute: coverage counters and the
/// history-valid statistics inputs.
#[derive(Default)]
struct Q8efEvidence {
    trials: usize,
    /// Trials that ended (EOS) before `Q8EF_TOKENS` IDs.
    early_trials: usize,
    /// Trials whose request began from a restored snapshot.
    restored_trials: usize,
    windows: u64,
    keep: [u64; 4],
    after_reset: u64,
    after_keep0: u64,
    ring_half: [u64; 2],
    graph_windows: u64,
    device_windows: u64,
    /// Target rows rebuilt from the raw verify logits and found equal to
    /// production's.
    independent_rows: u64,
    /// Same-route seed replays whose full window stream was identical.
    replays: u64,
    routes: BTreeMap<String, u64>,
    uniform_hist: [u64; Q8EF_PIT_BINS],
    category: BTreeMap<usize, Vec<(Vec<(u32, f32)>, u32)>>,
    pits: BTreeMap<usize, Vec<(usize, f64)>>,
    draft_pits: BTreeMap<usize, Vec<(usize, f64)>>,
}

impl Q8efEvidence {
    /// Check one window, count its coverage and record the PIT / category
    /// inputs of the stream positions it committed. A reset-contract
    /// violation (a rejected window that kept the residual, or an accepted /
    /// target-only window that zeroed it) is an error.
    fn observe(&mut self, salt: u64, ctx: &mut Q8efTrial, w: &Q8efWindow) -> Result<(), String> {
        let at = format!(
            "trial {} window at stream index {}",
            ctx.trial, ctx.next_index
        );
        q8ef_check_window(w).map_err(|e| format!("{at}: {e}"))?;
        let keep = w.keep();
        let reset = w.reset_delta();
        match keep {
            Q8efKeep::Full | Q8efKeep::NoDrafts if reset != 0 => {
                return Err(format!(
                    "{at}: {keep:?} window zeroed the residual ({reset} resets)"
                ));
            }
            Q8efKeep::Keep0 | Q8efKeep::Partial if reset == 0 => {
                return Err(format!("{at}: {keep:?} window left the residual unzeroed"));
            }
            _ => {}
        }
        for &u in &w.stream_uniforms {
            if !(0.0..1.0).contains(&u) {
                return Err(format!("{at}: stream uniform {u} is outside [0, 1)"));
            }
        }
        self.windows += 1;
        self.keep[keep.slot()] += 1;
        *self.routes.entry(w.route.clone()).or_default() += 1;
        if w.device {
            self.device_windows += 1;
        }
        if w.independent_rows != w.target_rows.len() {
            return Err(format!(
                "{at}: only {} of {} target rows were rebuilt from raw logits",
                w.independent_rows,
                w.target_rows.len()
            ));
        }
        self.independent_rows += w.independent_rows as u64;
        if ctx.prev_reset {
            self.after_reset += 1;
        }
        if ctx.prev_keep == Some(Q8efKeep::Keep0) {
            self.after_keep0 += 1;
        }
        if let Some(half) = w.ring_half {
            *self
                .ring_half
                .get_mut(usize::from(half))
                .ok_or_else(|| format!("{at}: ring half {half}"))? += 1;
        }
        if w.graph_launches > 0 {
            self.graph_windows += 1;
        }
        for &u in &w.stream_uniforms {
            self.uniform_hist[q8ef_bin(f64::from(u), Q8EF_PIT_BINS)] += 1;
        }
        for (j, (&draft, q)) in w.drafts.iter().zip(&w.q_rows).enumerate() {
            let u = q8ef_pit_draw(Q8EF_PIT_SALT ^ 0xD0, ctx.trial, (ctx.next_index << 4) + j);
            let pit = q8ef_pit(q, draft, u).map_err(|e| format!("{at}, draft {j}: {e}"))?;
            self.draft_pits.entry(j).or_default().push((ctx.trial, pit));
        }
        for (j, &token) in w.committed.iter().enumerate() {
            let index = ctx.next_index + j;
            if !q8ef_recorded(index) {
                continue;
            }
            let row = &w.target_rows[j];
            let pit = q8ef_pit(
                row,
                token,
                q8ef_pit_draw(Q8EF_PIT_SALT ^ salt, ctx.trial, index),
            )
            .map_err(|e| format!("{at}, committed token {j}: {e}"))?;
            self.pits.entry(index).or_default().push((ctx.trial, pit));
            if Q8EF_MARGINALS.contains(&index) {
                self.category
                    .entry(index)
                    .or_default()
                    .push((row.clone(), token));
            }
        }
        ctx.next_index += w.committed.len();
        ctx.prev_keep = Some(keep);
        ctx.prev_reset = reset > 0;
        Ok(())
    }

    fn to_json(&self) -> Result<Value, String> {
        let json_of = |e: serde_json::Error| e.to_string();
        Ok(json!({
            "trials": self.trials,
            "early_trials": self.early_trials,
            "restored_trials": self.restored_trials,
            "windows": self.windows,
            "keep": self.keep,
            "after_reset": self.after_reset,
            "after_keep0": self.after_keep0,
            "ring_half": self.ring_half,
            "graph_windows": self.graph_windows,
            "device_windows": self.device_windows,
            "independent_rows": self.independent_rows,
            "replays": self.replays,
            "routes": self.routes,
            "uniform_hist": self.uniform_hist,
            "category": serde_json::to_value(&self.category).map_err(json_of)?,
            "pits": serde_json::to_value(&self.pits).map_err(json_of)?,
            "draft_pits": serde_json::to_value(&self.draft_pits).map_err(json_of)?,
        }))
    }

    fn from_json(v: &Value) -> Result<Self, String> {
        fn get<T: serde::de::DeserializeOwned>(v: &Value, key: &str) -> Result<T, String> {
            serde_json::from_value(v[key].clone()).map_err(|e| format!("evidence {key}: {e}"))
        }
        Ok(Self {
            trials: get(v, "trials")?,
            early_trials: get(v, "early_trials")?,
            restored_trials: get(v, "restored_trials")?,
            windows: get(v, "windows")?,
            keep: get(v, "keep")?,
            after_reset: get(v, "after_reset")?,
            after_keep0: get(v, "after_keep0")?,
            ring_half: get(v, "ring_half")?,
            graph_windows: get(v, "graph_windows")?,
            device_windows: get(v, "device_windows")?,
            independent_rows: get(v, "independent_rows")?,
            replays: get(v, "replays")?,
            routes: get(v, "routes")?,
            uniform_hist: get(v, "uniform_hist")?,
            category: get(v, "category")?,
            pits: get(v, "pits")?,
            draft_pits: get(v, "draft_pits")?,
        })
    }

    /// Fold another shard of the same arm in (trial indices are global, so
    /// nothing is renumbered).
    fn merge(&mut self, other: Self) {
        self.trials += other.trials;
        self.early_trials += other.early_trials;
        self.restored_trials += other.restored_trials;
        self.windows += other.windows;
        self.after_reset += other.after_reset;
        self.after_keep0 += other.after_keep0;
        self.graph_windows += other.graph_windows;
        self.device_windows += other.device_windows;
        self.independent_rows += other.independent_rows;
        self.replays += other.replays;
        for i in 0..4 {
            self.keep[i] += other.keep[i];
        }
        for i in 0..2 {
            self.ring_half[i] += other.ring_half[i];
        }
        for i in 0..Q8EF_PIT_BINS {
            self.uniform_hist[i] += other.uniform_hist[i];
        }
        for (route, n) in other.routes {
            *self.routes.entry(route).or_default() += n;
        }
        for (index, rows) in other.category {
            self.category.entry(index).or_default().extend(rows);
        }
        for (index, pits) in other.pits {
            self.pits.entry(index).or_default().extend(pits);
        }
        for (slot, pits) in other.draft_pits {
            self.draft_pits.entry(slot).or_default().extend(pits);
        }
    }
}

/// Law verdict of one arm over `trials` declared trials: category and PIT
/// marginals at `Q8EF_MARGINALS`, joint PIT grids at `Q8EF_JOINTS`, the
/// proposal-law PIT by draft slot and the stream-uniform histogram. A p-value
/// at or under `ALPHA` is a law failure. A position (or slot) observed in
/// fewer trials than declared (an EOS before it, an untapped window), an
/// arm with fewer trials than declared, or a test that cannot be formed is
/// INCOMPLETE: nothing is judged on survivors and nothing is a pass by
/// default.
fn q8ef_judge(
    arm: &str,
    ev: &Q8efEvidence,
    trials: usize,
    report: &mut String,
    failures: &mut Vec<String>,
    incomplete: &mut Vec<String>,
) {
    let need = trials;
    report.push_str(&format!(
        "{arm}: {} of {trials} trials, {} windows, {} ended before {Q8EF_TOKENS} IDs\n",
        ev.trials, ev.windows, ev.early_trials
    ));
    if ev.trials != trials {
        incomplete.push(format!(
            "{arm}: evidence covers {} of the {trials} declared trials\n",
            ev.trials
        ));
    }
    report.push_str(&format!(
        "{arm}: {} target rows rebuilt from raw logits and equal to production's\n",
        ev.independent_rows
    ));
    if ev.independent_rows == 0 {
        incomplete.push(format!(
            "{arm}: no target row was rebuilt from raw logits (independent target law unchecked)\n"
        ));
    }
    let mut verdict = |what: String, result: Result<(f64, usize, f64), String>| match result {
        Ok((stat, df, p)) => {
            report.push_str(&format!("{arm} {what}: chi2={stat:.2} df={df} p={p:.4}\n"));
            if p <= ALPHA {
                failures.push(format!(
                    "{arm} {what}: p={p:.3e} <= {ALPHA} (chi2={stat:.2}, df={df})\n"
                ));
            }
        }
        Err(error) => incomplete.push(format!("{arm} {what}: cannot be judged: {error}\n")),
    };
    for &index in &Q8EF_MARGINALS {
        let rows: &[(Vec<(u32, f32)>, u32)] = match ev.category.get(&index) {
            Some(rows) => rows.as_slice(),
            None => &[],
        };
        if rows.len() != need {
            verdict(
                format!("category position {index}"),
                Err(format!("{} observations for {need} trials", rows.len())),
            );
        } else {
            let pairs: Vec<(&[(u32, f32)], u32)> = rows
                .iter()
                .map(|(row, token)| (row.as_slice(), *token))
                .collect();
            verdict(
                format!("category position {index}"),
                q8ef_category_chi_square(&pairs),
            );
        }
        let pits: Vec<f64> = ev
            .pits
            .get(&index)
            .map_or(Vec::new(), |v| v.iter().map(|&(_, pit)| pit).collect());
        if pits.len() != need {
            verdict(
                format!("PIT position {index}"),
                Err(format!("{} observations for {need} trials", pits.len())),
            );
        } else {
            verdict(
                format!("PIT position {index}"),
                q8ef_pit_uniform(&pits, Q8EF_PIT_BINS),
            );
        }
    }
    for &(a, b) in &Q8EF_JOINTS {
        let second_pits: Vec<(usize, f64)> = ev.pits.get(&b).cloned().unwrap_or_default();
        let second: HashMap<usize, f64> = second_pits.into_iter().collect();
        let pairs: Vec<(f64, f64)> = ev
            .pits
            .get(&a)
            .into_iter()
            .flatten()
            .filter_map(|&(trial, pa)| second.get(&trial).map(|&pb| (pa, pb)))
            .collect();
        if pairs.len() != need {
            verdict(
                format!("joint PIT ({a},{b})"),
                Err(format!(
                    "{} paired observations for {need} trials",
                    pairs.len()
                )),
            );
        } else {
            verdict(
                format!("joint PIT ({a},{b})"),
                q8ef_pit_joint(&pairs, Q8EF_JOINT_GRID),
            );
        }
    }
    for slot in 0..MTP_K {
        let pits: Vec<f64> = ev
            .draft_pits
            .get(&slot)
            .map_or(Vec::new(), |v| v.iter().map(|&(_, pit)| pit).collect());
        if pits.len() < need {
            verdict(
                format!("proposal PIT slot {slot}"),
                Err(format!("{} drafts for {need} trials", pits.len())),
            );
        } else {
            verdict(
                format!("proposal PIT slot {slot}"),
                q8ef_pit_uniform(&pits, Q8EF_PIT_BINS),
            );
        }
    }
    let n: u64 = ev.uniform_hist.iter().sum();
    let expected = n as f64 / Q8EF_PIT_BINS as f64;
    if expected < Q8EF_MIN_EXPECTED {
        verdict(
            "stream uniforms".into(),
            Err(format!("{n} uniforms over {Q8EF_PIT_BINS} bins")),
        );
    } else {
        let stat: f64 = ev
            .uniform_hist
            .iter()
            .map(|&c| (c as f64 - expected).powi(2) / expected)
            .sum();
        let df = Q8EF_PIT_BINS - 1;
        verdict(
            "stream uniforms".into(),
            Ok((stat, df, gamma_q(df as f64 / 2.0, stat / 2.0))),
        );
    }
}

/// One required coverage item with what the run observed.
struct Q8efCover {
    id: &'static str,
    what: &'static str,
    observed: u64,
    required: u64,
}

/// One counter of one arm's evidence.
type Q8efPick = fn(&Q8efEvidence) -> u64;

fn pick_windows(e: &Q8efEvidence) -> u64 {
    e.windows
}
fn pick_keep0(e: &Q8efEvidence) -> u64 {
    e.keep[Q8efKeep::Keep0.slot()]
}
fn pick_partial(e: &Q8efEvidence) -> u64 {
    e.keep[Q8efKeep::Partial.slot()]
}
fn pick_full(e: &Q8efEvidence) -> u64 {
    e.keep[Q8efKeep::Full.slot()]
}
fn pick_after_reset(e: &Q8efEvidence) -> u64 {
    e.after_reset
}
fn pick_after_keep0(e: &Q8efEvidence) -> u64 {
    e.after_keep0
}
fn pick_ring_half_0(e: &Q8efEvidence) -> u64 {
    e.ring_half[0]
}
fn pick_ring_half_1(e: &Q8efEvidence) -> u64 {
    e.ring_half[1]
}
fn pick_graph_windows(e: &Q8efEvidence) -> u64 {
    e.graph_windows
}
fn pick_replays(e: &Q8efEvidence) -> u64 {
    e.replays
}
fn pick_restored(e: &Q8efEvidence) -> u64 {
    e.restored_trials as u64
}

/// `pick` of arm `arm`; a missing arm counts as zero.
fn q8ef_arm_value(arms: &BTreeMap<String, Q8efEvidence>, arm: &str, pick: Q8efPick) -> u64 {
    match arms.get(arm) {
        Some(ev) => pick(ev),
        None => 0,
    }
}

/// The smallest `pick` over the given arms (what EVERY one of them shows).
fn q8ef_min_over(arms: &BTreeMap<String, Q8efEvidence>, names: &[&str], pick: Q8efPick) -> u64 {
    let mut min = u64::MAX;
    for name in names {
        min = min.min(q8ef_arm_value(arms, name, pick));
    }
    if names.is_empty() {
        0
    } else {
        min
    }
}

/// Windows of `arm` that ran verify route `route`.
fn q8ef_route_windows(arms: &BTreeMap<String, Q8efEvidence>, arm: &str, route: &str) -> u64 {
    match arms.get(arm) {
        Some(ev) => ev.routes.get(route).copied().unwrap_or(0),
        None => 0,
    }
}

fn q8ef_cover(id: &'static str, what: &'static str, observed: u64, required: u64) -> Q8efCover {
    Q8efCover {
        id,
        what,
        observed,
        required,
    }
}

/// Coverage of the whole run from every arm's evidence. A missing arm counts
/// as zero.
fn q8ef_coverage(arms: &BTreeMap<String, Q8efEvidence>, cohort: &Q8efCohort) -> Vec<Q8efCover> {
    let routes: &[&str] = &Q8EF_ROUTE_ARMS;
    let all: Vec<&str> = q8ef_all_arms().collect();
    let forced: &[&str] = &["q8ef_batched", "q8ef_interleaved"];
    let trials = cohort.trials as u64;
    vec![
        q8ef_cover("forced_batched_windows", "windows of the forced batched route (HIPFIRE_MTP_INCREMENTAL=0, floor off)", q8ef_route_windows(arms, "q8ef_batched", "batched"), Q8EF_MIN_WINDOWS),
        q8ef_cover("forced_interleaved_windows", "windows of the forced interleaved route (HIPFIRE_MTP_INCREMENTAL=1, floor off)", q8ef_route_windows(arms, "q8ef_interleaved", "interleaved"), Q8EF_MIN_WINDOWS),
        q8ef_cover("adaptive_product_windows", "windows of the default product route (floor on); wall-clock decisions recorded and tested, never replayed", q8ef_arm_value(arms, "q8ef_adaptive", pick_windows), Q8EF_MIN_WINDOWS),
        q8ef_cover("reject_keep0", "rejected windows that keep no draft, in every route arm", q8ef_min_over(arms, routes, pick_keep0), Q8EF_MIN_WINDOWS),
        q8ef_cover("reject_partial_keep", "rejected windows that keep part of their drafts, in every route arm", q8ef_min_over(arms, routes, pick_partial), Q8EF_MIN_WINDOWS),
        q8ef_cover("full_accept", "fully accepted windows, in every route arm", q8ef_min_over(arms, routes, pick_full), Q8EF_MIN_WINDOWS),
        q8ef_cover("sampled_window_after_ef_reset", "windows whose previous window zeroed the live residual, in every route arm", q8ef_min_over(arms, routes, pick_after_reset), Q8EF_MIN_WINDOWS),
        q8ef_cover("transaction_restore_then_sample", "windows right after a keep0 transaction restore, in every route arm", q8ef_min_over(arms, routes, pick_after_keep0), Q8EF_MIN_WINDOWS),
        q8ef_cover("session_restore_then_sample", "requests that began from a resident session snapshot", q8ef_arm_value(arms, "q8ef_session_batched", pick_restored), trials),
        q8ef_cover("prefix_restore_then_sample", "requests that began from a pool prefix snapshot chain", q8ef_arm_value(arms, "q8ef_prefix_batched", pick_restored), trials),
        q8ef_cover("disk_restore_then_sample", "requests that began from snapshots demoted to and read back from the disk tier", q8ef_arm_value(arms, Q8EF_DISK_ARM, pick_restored), trials),
        q8ef_cover("ring_half_0", "verifies written into ring half 0", all.iter().map(|arm| q8ef_arm_value(arms, arm, pick_ring_half_0)).sum::<u64>(), Q8EF_MIN_WINDOWS),
        q8ef_cover("ring_half_1", "verifies written into ring half 1", all.iter().map(|arm| q8ef_arm_value(arms, arm, pick_ring_half_1)).sum::<u64>(), Q8EF_MIN_WINDOWS),
        q8ef_cover("graph_replay", "windows that launched a captured hipGraph", all.iter().map(|arm| q8ef_arm_value(arms, arm, pick_graph_windows)).sum::<u64>(), Q8EF_MIN_WINDOWS),
        q8ef_cover("same_route_replay", "forced arms: seeds replayed with identical window streams (rows, drafts, uniforms, verdicts, resets), per arm", q8ef_min_over(arms, forced, pick_replays), REPLAYS as u64),
    ]
}

// ---------------------------------------------------------------------------
// q8_ef verify-law case: arm processes and the parent
// ---------------------------------------------------------------------------

fn q8ef_config(
    arm: &str,
    identity: &Value,
    settings: &DistSettings,
    total: usize,
    range: (usize, usize),
) -> Result<Value, String> {
    let mut config = config_base("q8ef-law", arm, &settings.pen, settings.fixture, identity);
    config["temp"] = json!(settings.temp);
    config["top_p"] = json!(settings.top_p);
    config["top_k"] = json!(settings.top_k);
    config["min_p"] = json!(settings.min_p);
    config["tokens"] = json!(Q8EF_TOKENS);
    config["max_seq"] = json!(Q8EF_MAX_SEQ);
    config["trials"] = json!(total);
    config["shard"] = json!([range.0, range.1]);
    config["salt"] = json!(q8ef_salt(arm)?);
    config["pit_salt"] = json!(Q8EF_PIT_SALT);
    config["sampled_mode"] = json!("leviathan");
    config["marginals"] = json!(Q8EF_MARGINALS);
    config["joints"] = json!(Q8EF_JOINTS);
    config["pit_bins"] = json!(Q8EF_PIT_BINS);
    config["joint_grid"] = json!(Q8EF_JOINT_GRID);
    config["replays"] = json!(REPLAYS);
    Ok(config)
}

/// Sorted `(start, end)` of every `<arm>.<start>-<end>.q8ef.json` in `dir`.
fn q8ef_shards(dir: &Path, arm: &str) -> Result<Vec<(usize, usize)>, String> {
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
            .and_then(|rest| rest.strip_suffix(".q8ef.json"))
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

/// [`spawn_arm`] for the q8_ef arm test, with the stage variable a disk
/// publish / restore stage the disk arm needs.
fn q8ef_spawn(
    arm: &str,
    mode: &'static str,
    model: &Path,
    dir: &Path,
    shard: (usize, usize),
    stage: Option<&str>,
    log_stem: &str,
) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| format!("test binary path: {e}"))?;
    let mut child = Command::new(exe);
    child
        .args([
            "--exact",
            Q8EF_ARM_TEST,
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ARM_ENV, arm)
        .env(MODEL_ENV, model)
        .env(OUT_ENV, dir)
        .env(SHARD_ENV, format!("{}..{}", shard.0, shard.1))
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
    match stage {
        Some(stage) => {
            child.env(Q8EF_STAGE_ENV, stage);
        }
        None => {
            child.env_remove(Q8EF_STAGE_ENV);
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

/// Every shard of `arm` covering exactly `0..total`, each verified against
/// its recorded config; missing ranges run as fresh child processes (the
/// disk arm a publish process and a restore process per gap).
fn q8ef_ensure_shards(
    model: &Path,
    dir: &Path,
    arm: &str,
    total: usize,
    settings: &DistSettings,
    identity: &Value,
) -> Result<Vec<(usize, usize)>, String> {
    let verify = |shards: &[(usize, usize)]| -> Result<(), String> {
        for &range in shards {
            check_config(
                &late_file(dir, arm, range, "q8ef.json"),
                &q8ef_config(arm, identity, settings, total, range)?,
            )?;
        }
        Ok(())
    };
    let shards = q8ef_shards(dir, arm)?;
    verify(&shards)?;
    for gap in shard_gaps(&shards, total)? {
        eprintln!("{arm}: running missing shard {}..{}", gap.0, gap.1);
        let stem = format!("{arm}.{}-{}", gap.0, gap.1);
        if arm == Q8EF_DISK_ARM {
            // The publish process cold-prefills the restore prompt and
            // unloads, demoting its snapshots to this shard's disk dir; the
            // restore process then runs the shard's trials against it.
            q8ef_spawn(
                arm,
                "leviathan",
                model,
                dir,
                gap,
                Some("publish"),
                &format!("{stem}.publish"),
            )?;
            q8ef_spawn(arm, "leviathan", model, dir, gap, Some("restore"), &stem)?;
        } else {
            q8ef_spawn(arm, "leviathan", model, dir, gap, None, &stem)?;
        }
    }
    let shards = q8ef_shards(dir, arm)?;
    verify(&shards)?;
    if !shard_gaps(&shards, total)?.is_empty() {
        return Err(format!("{arm}: shards {shards:?} do not cover 0..{total}"));
    }
    Ok(shards)
}

/// Deterministic restore-arm prompt tokens (content is irrelevant to the
/// law; only its length against the chunk size matters).
fn q8ef_prompt(seed: u64, len: usize) -> Vec<u32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (1000 + (state >> 33) % 99_000) as u32
        })
        .collect()
}

#[cfg(feature = "lab")]
mod q8ef_run {
    use super::*;
    use hipfire_arch_qwen4::lab_window::{
        LabDraw, LabForwardLogits, LabVerdict, LabVerifyRoute, LabWindow, LabWindowArm,
    };
    use hipfire_runtime::arch_model::ArchModel;
    use hipfire_runtime::sampler::apply_logit_policy_cpu;
    use hipfire_runtime::serve_contract::CacheDomain;
    use hipfire_runtime::session_cache::{SessionCache, SessionRoute};
    use hipfire_runtime::spec_sampling::{
        fill_penalty_history, naive_target_sampler, SampleSpec, SparseDist,
    };

    /// A recorded draw must be the stream uniform the window measured it at.
    fn draw_u(w: &LabWindow, draw: LabDraw) -> Result<f32, String> {
        if draw.index == usize::MAX {
            return Err("a draw's offset in the request stream could not be measured".into());
        }
        match w.uniforms.get(draw.index) {
            Some(&u) if u == draw.u && u.is_finite() => Ok(u),
            other => Err(format!(
                "draw {} records {} but the window's stream has {other:?}",
                draw.index, draw.u
            )),
        }
    }

    /// What the independent target-law construction needs besides the raw
    /// logits: the request's policy and the history inputs.
    pub(super) struct RowCtx<'a> {
        pub prompt: &'a [u32],
        /// The committed stream at the window start, the pending seed
        /// included (the `emitted` the window was called with).
        pub emitted: &'a [u32],
        pub cfg: &'a SpecRequestConfig,
    }

    /// Row `r` of one forward's kept raw logits.
    fn raw_row(f: &LabForwardLogits, r: usize) -> Result<&[f32], String> {
        let all: &[f32] = match f.logits.as_deref() {
            Some(all) => all,
            None => {
                return Err(
                    "the verify forward's raw logits were not kept (keep_logits is off)"
                        .to_string(),
                );
            }
        };
        let (start, end) = (r * f.vocab, (r + 1) * f.vocab);
        all.get(start..end)
            .ok_or_else(|| format!("raw logits hold {} values, row {r} needs {end}", all.len()))
    }

    /// The raw verify-forward logits of rows `0..rows` of the window's block
    /// `[seed] ++ drafts`: a batched window has one forward over the block,
    /// an interleaved or target-only window one single-row forward per row
    /// (in order). Replay forwards after a rollback are never matched.
    fn raw_rows<'a>(
        w: &'a LabWindow,
        drafts: &[u32],
        rows: usize,
    ) -> Result<Vec<&'a [f32]>, String> {
        let block: Vec<u32> = std::iter::once(w.seed_token)
            .chain(drafts.iter().copied())
            .collect();
        if rows > block.len() {
            return Err(format!("{rows} rows read from a {}-row block", block.len()));
        }
        let batched: Option<&'a LabForwardLogits> =
            w.forwards.iter().find(|f| !f.prefill && f.tokens == block);
        if let Some(f) = batched {
            if f.rows != block.len() {
                return Err(format!(
                    "the verify forward emitted {} rows for a {}-row block",
                    f.rows,
                    block.len()
                ));
            }
            let mut out: Vec<&'a [f32]> = Vec::with_capacity(rows);
            for r in 0..rows {
                out.push(raw_row(f, r)?);
            }
            return Ok(out);
        }
        let mut found: Vec<&'a [f32]> = Vec::with_capacity(rows);
        for f in w.forwards.iter() {
            if f.prefill || f.rows != 1 || f.tokens.len() != 1 {
                continue;
            }
            if found.len() < rows && f.tokens[0] == block[found.len()] {
                found.push(raw_row(f, 0)?);
            }
        }
        if found.len() < rows {
            return Err(format!(
                "only {} of {rows} verify rows have a matching target forward among {} recorded forwards",
                found.len(),
                w.forwards.len()
            ));
        }
        Ok(found)
    }

    /// Rebuild every target row the acceptance read from the RAW model
    /// logits, independently of the production row builder's inputs and
    /// route: penalties and blocked tokens over `suffix_W(prompt ‖ emitted ‖
    /// drafts[..j])` (`apply_logit_policy_cpu`), then temperature / top_k /
    /// top_p / min_p (`SparseDist::build_from_logits` under
    /// `SampleSpec::cpu_ar`). The production row must equal it: bit for bit
    /// on the host route, to 1e-6 on a row read back from the device. Returns
    /// the number of rows verified.
    fn rebuild_rows(
        w: &LabWindow,
        drafts: &[u32],
        target_rows: &[Vec<(u32, f32)>],
        ctx: &RowCtx,
    ) -> Result<usize, String> {
        let raw = raw_rows(w, drafts, target_rows.len())?;
        let policy = naive_target_sampler(ctx.cfg);
        let spec = SampleSpec::cpu_ar(ctx.cfg.temp, ctx.cfg.top_p, ctx.cfg.top_k, ctx.cfg.min_p);
        let window = ctx.cfg.penalty_window();
        let mut base = Vec::new();
        fill_penalty_history(&mut base, ctx.prompt, ctx.emitted, window);
        for (j, (raw, got)) in raw.iter().zip(target_rows).enumerate() {
            let mut history = base.clone();
            history.extend_from_slice(&drafts[..j]);
            let history: &[u32] = if window == 0 {
                &[]
            } else {
                &history[history.len().saturating_sub(window)..]
            };
            let mut row = raw.to_vec();
            apply_logit_policy_cpu(&mut row, history, &policy);
            let mut dist = SparseDist::default();
            dist.build_from_logits(&row, spec, &mut Vec::new())
                .map_err(|e| format!("row {j}: {e}"))?;
            let want = dist.entries();
            let from_device = w
                .rows
                .iter()
                .find(|r| r.row == j)
                .map_or(false, |r| r.from_device);
            let same = if from_device {
                want.len() == got.len()
                    && want
                        .iter()
                        .zip(got)
                        .all(|(a, b)| a.0 == b.0 && (a.1 - b.1).abs() <= 1e-6)
            } else {
                want == got.as_slice()
            };
            if !same {
                return Err(format!(
                    "row {j} ({}) differs from the production row: rebuilt {:?}..., production {:?}...",
                    if from_device { "device" } else { "host" },
                    &want[..want.len().min(4)],
                    &got[..got.len().min(4)]
                ));
            }
        }
        Ok(target_rows.len())
    }

    /// The tap's window as the law case consumes it. Any record the tap
    /// cannot attribute (error, unmeasured draw, row gap, non-finite logits,
    /// naive verifier) is an error: nothing is skipped.
    pub(super) fn window_from_lab(
        w: &LabWindow,
        eos: u32,
        graph_launches: u64,
        ctx: &RowCtx,
    ) -> Result<Q8efWindow, String> {
        if let Some(error) = &w.error {
            return Err(format!("window tap error: {error}"));
        }
        if !w.sampled || w.naive {
            return Err("not a sampled Leviathan window".into());
        }
        if w.forwards
            .iter()
            .any(|f| f.nonfinite.iter().any(|&n| n > 0))
        {
            return Err("non-finite logits in a verify forward of the window".into());
        }
        if w.proposals.len() < w.drafts_generated {
            return Err(format!(
                "{} proposals recorded for {} drafts generated",
                w.proposals.len(),
                w.drafts_generated
            ));
        }
        let (mut drafts, mut q_rows, mut tape) = (Vec::new(), Vec::new(), Vec::new());
        for (j, p) in w.proposals.iter().take(w.drafts_generated).enumerate() {
            let q = if p.q.is_empty() {
                if !p.point_mass {
                    return Err(format!("proposal {j} has no proposal row"));
                }
                vec![(p.token, 1.0)]
            } else {
                p.q.clone()
            };
            match p.draw {
                Some(draw) => {
                    let u = draw_u(w, draw).map_err(|e| format!("proposal {j}: {e}"))?;
                    if q8ef_sample(&q, u) != p.token {
                        return Err(format!(
                            "proposal {j}: draft {} is not the inverse-CDF draw of its proposal row at u={u}",
                            p.token
                        ));
                    }
                }
                None if p.point_mass => {}
                None => return Err(format!("proposal {j} has no draw")),
            }
            match p.verdict {
                Some(LabVerdict::Accept { draw }) => tape.push(draw_u(w, draw)?),
                Some(LabVerdict::Reject {
                    draw,
                    residual_draw,
                    ..
                }) => {
                    tape.push(draw_u(w, draw)?);
                    tape.push(draw_u(w, residual_draw)?);
                }
                None => {}
            }
            drafts.push(p.token);
            q_rows.push(q);
        }
        if let Some(bonus) = &w.bonus {
            tape.push(draw_u(w, bonus.draw)?);
        }
        let rows = w.rows.iter().map(|r| r.row).max().map_or(0, |max| max + 1);
        let mut target_rows: Vec<Option<Vec<(u32, f32)>>> = vec![None; rows];
        for r in &w.rows {
            if target_rows[r.row].replace(r.p.clone()).is_some() {
                return Err(format!("verify row {} was read twice", r.row));
            }
        }
        let target_rows = target_rows
            .into_iter()
            .enumerate()
            .map(|(j, row)| row.ok_or_else(|| format!("verify row {j} has no recorded target row")))
            .collect::<Result<Vec<_>, String>>()?;
        let independent_rows = rebuild_rows(w, &drafts, &target_rows, ctx)
            .map_err(|e| format!("independent target-law construction: {e}"))?;
        Ok(Q8efWindow {
            drafts,
            target_rows,
            q_rows,
            uniforms: tape,
            stream_uniforms: w.uniforms.clone(),
            committed: w.committed.clone(),
            accepted: w.accepted,
            eos: Some(eos),
            resets: (w.gdn_residual_resets_before, w.gdn_residual_resets_after),
            ring_half: w.ring_half,
            graph_launches,
            route: match w.verify_route {
                LabVerifyRoute::TargetOnly => "target_only",
                LabVerifyRoute::Batched => "batched",
                LabVerifyRoute::Interleaved => "interleaved",
            }
            .to_string(),
            device: w.device_route,
            independent_rows,
        })
    }

    /// One tapped sampled request: the same decode loop as
    /// `Loaded::mtp_request`, every window observed, checked and recorded
    /// into `ev`. Returns the windows and the number of IDs produced.
    #[allow(clippy::too_many_arguments)]
    fn request(
        loaded: &mut Loaded,
        drafter: &mut Qwen4MtpDrafter,
        prompt: &[u32],
        reused: usize,
        cfg: SpecRequestConfig,
        salt: u64,
        trial: usize,
        ev: &mut Q8efEvidence,
    ) -> Result<(Vec<Q8efWindow>, usize), String> {
        let eos = loaded.bundle.config.eos_token_id;
        drafter.configure_request(cfg);
        let resets = loaded.bundle.state.gdn_residual_resets();
        let mut seed_token = drafter
            .mtp_prefill(
                &mut loaded.gpu,
                &mut loaded.bundle,
                prompt,
                &prompt[reused..],
                reused,
                reused > 0,
                &|| false,
            )?
            .into_first_token()?;
        if reused > 0 && loaded.bundle.state.gdn_residual_resets() <= resets {
            return Err(format!(
                "trial {trial}: the restore of {reused} rows did not zero the live residual"
            ));
        }
        // Prefill-era taps (the seed draw is not a window) are discarded, not
        // observed.
        drafter.lab_take_window_observations();
        loaded.bundle.lab_take_forward_logits();
        let mut ids = vec![seed_token];
        let mut windows = Vec::new();
        let mut ctx = Q8efTrial::new(trial);
        while ids.len() < Q8EF_TOKENS && seed_token != eos {
            let position = loaded.bundle.state.position;
            let k = MTP_K.min(Q8EF_TOKENS - ids.len());
            let started = Counters::now();
            let window = drafter.mtp_step(
                &mut loaded.gpu,
                &mut loaded.bundle,
                position,
                seed_token,
                &ids,
                k,
                eos,
                None,
            )?;
            let graphs = Counters::now().since(started).graph_launch;
            let mut observed = drafter.lab_take_window_observations();
            if observed.len() != 1 {
                return Err(format!(
                    "trial {trial}: the tap produced {} observations for one window",
                    observed.len()
                ));
            }
            let row_ctx = RowCtx {
                prompt,
                emitted: &ids,
                cfg: &cfg,
            };
            let w = window_from_lab(&observed.remove(0), eos, graphs, &row_ctx)
                .map_err(|e| format!("trial {trial} stream index {}: {e}", ctx.next_index))?;
            if w.committed != window.committed || w.accepted != window.accepted {
                return Err(format!(
                    "trial {trial}: the tap saw committed {:?} accepted {} but mtp_step returned {:?} accepted {}",
                    w.committed, w.accepted, window.committed, window.accepted
                ));
            }
            ev.observe(salt, &mut ctx, &w)?;
            seed_token = *window.committed.last().ok_or("empty MTP window")?;
            ids.extend_from_slice(&window.committed);
            windows.push(w);
        }
        Ok((windows, ids.len()))
    }

    fn attach_cache(
        loaded: &mut Loaded,
        model: &Path,
        budget: u64,
        disk: Option<&Path>,
    ) -> Result<(), String> {
        let hfq = HfqFile::open(model).map_err(|e| e.to_string())?;
        let domain =
            CacheDomain::for_model(&hfq, &loaded.tokenizer, None, "qwen4", loaded.gpu.device_id);
        let cache = SessionCache::new(domain, budget);
        loaded.bundle.set_turn_snapshots(false);
        loaded.bundle.attach_session_cache(match disk {
            Some(dir) => cache.with_disk(dir, 1 << 40),
            None => cache,
        });
        Ok(())
    }

    /// A cold native-MTP prefill of `prompt` committed to the session cache.
    fn prime(
        loaded: &mut Loaded,
        drafter: &mut Qwen4MtpDrafter,
        prompt: &[u32],
        cfg: SpecRequestConfig,
    ) -> Result<(), String> {
        drafter.configure_request(cfg);
        drafter
            .mtp_prefill(
                &mut loaded.gpu,
                &mut loaded.bundle,
                prompt,
                prompt,
                0,
                false,
                &|| false,
            )?
            .into_first_token()?;
        loaded.bundle.session_commit();
        Ok(())
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Kind {
        Route,
        Session,
        Prefix,
        Disk,
    }

    pub(super) fn run_arm(arm: &str, model: &Path, dir: &Path) -> Result<(), String> {
        let kind = match arm {
            "q8ef_batched" | "q8ef_interleaved" | "q8ef_adaptive" => Kind::Route,
            "q8ef_session_batched" => Kind::Session,
            "q8ef_prefix_batched" => Kind::Prefix,
            Q8EF_DISK_ARM => Kind::Disk,
            other => return Err(format!("{other} is not a q8_ef arm")),
        };
        require_arm_env(arm, "leviathan")?;
        let cohort = Q8efCohort::from_env()?;
        let (mode, _) = hw_identity::gdn_test_state_quant()?;
        q8ef_check_plan(cohort.trials, mode)?;
        let settings = DistSettings::from_env()?;
        let total = cohort.total(arm);
        let range = shard_range(arm, total)?;
        let salt = q8ef_salt(arm)?;
        let stage = std::env::var(Q8EF_STAGE_ENV).ok();
        if kind == Kind::Disk && !matches!(stage.as_deref(), Some("publish" | "restore")) {
            return Err(format!(
                "{arm}: {Q8EF_STAGE_ENV} must be publish or restore"
            ));
        }
        let identity = own_identity(model, dir, &Q8EF_CONTROL_ENV)?;
        let config = q8ef_config(arm, &identity, &settings, total, range)?;
        let path = late_file(dir, arm, range, "q8ef.json");
        if stage.as_deref() != Some("publish") && path.exists() {
            return check_config(&path, &config);
        }
        let disk_dir = dir.join(format!("q8ef-disk-{}", range.0));
        let mut loaded = load_with_max_seq(model, Q8EF_MAX_SEQ)?;
        let route_prompt = loaded.render(&settings.fixture.text())?;
        settings
            .fixture
            .check_prompt(route_prompt.len(), &settings.pen)?;
        let chunk = loaded
            .bundle
            .spec_chunk_rows()
            .ok_or("the target has no speculative chunk rows")?;
        let long = 2 * chunk + 100;
        if kind != Kind::Route && long + Q8EF_TOKENS + 64 > Q8EF_MAX_SEQ {
            return Err(format!(
                "{arm}: a restore prompt of {long} rows (chunk {chunk}) does not fit the q8_ef context {Q8EF_MAX_SEQ}"
            ));
        }
        match kind {
            Kind::Route => {}
            Kind::Session => attach_cache(&mut loaded, model, 1, None)?,
            Kind::Prefix => attach_cache(&mut loaded, model, u64::MAX >> 1, None)?,
            Kind::Disk => {
                if stage.as_deref() == Some("publish") {
                    let _ = fs::remove_dir_all(&disk_dir);
                }
                fs::create_dir_all(&disk_dir).map_err(|e| e.to_string())?;
                attach_cache(&mut loaded, model, u64::MAX >> 1, Some(&disk_dir))?;
            }
        }
        let a = q8ef_prompt(0xa, long);
        let mut a2 = a[..chunk + 50].to_vec();
        a2.extend(q8ef_prompt(0xc, 100));
        let disk_prompt = q8ef_prompt(0xd, long);
        let mut drafter = new_drafter_with_max_seq(&mut loaded, None, false, Q8EF_MAX_SEQ)?;
        arm_taps(&mut loaded, &mut drafter);
        let prime_cfg = |seed: u64| -> SpecRequestConfig {
            spec_request(
                settings.temp,
                settings.top_p,
                settings.top_k,
                settings.min_p,
                seed,
                &settings.pen,
            )
        };
        if kind == Kind::Disk && stage.as_deref() == Some("publish") {
            // Unloading demotes the snapshots of this cold prefill to the
            // disk tier; the restore process must start from them.
            prime(&mut loaded, &mut drafter, &disk_prompt, prime_cfg(1))?;
            Box::new(drafter).mtp_free(&mut loaded.gpu);
            return loaded
                .bundle
                .free_gpu(&mut loaded.gpu)
                .map_err(|e| e.to_string());
        }
        let a_slice: &[u32] = a.as_slice();
        let a2_slice: &[u32] = a2.as_slice();
        let disk_slice: &[u32] = disk_prompt.as_slice();
        match kind {
            Kind::Session => prime(
                &mut loaded,
                &mut drafter,
                session_prompt_of(range.0 + 1, a_slice, a2_slice),
                prime_cfg(1),
            )?,
            Kind::Prefix => prime(&mut loaded, &mut drafter, a_slice, prime_cfg(1))?,
            _ => {}
        }
        let forced = matches!(kind, Kind::Route) && arm != "q8ef_adaptive";
        let started = std::time::Instant::now();
        let mut ev = Q8efEvidence::default();
        for trial in range.0..range.1 {
            let seed = trial_seed(salt, trial);
            let cfg = prime_cfg(seed);
            let mut disk_before = 0usize;
            let (prompt, reused): (&[u32], usize) = match kind {
                Kind::Route => (route_prompt.as_slice(), 0),
                Kind::Session | Kind::Prefix | Kind::Disk => {
                    let target: &[u32] = match kind {
                        Kind::Session => session_prompt_of(trial, a_slice, a2_slice),
                        Kind::Prefix => a_slice,
                        _ => disk_slice,
                    };
                    if kind == Kind::Prefix {
                        // A reset outside the cache drops the resident
                        // snapshot, so the plan can only offer the pool.
                        loaded
                            .bundle
                            .reset(&mut loaded.gpu)
                            .map_err(|e| e.to_string())?;
                    }
                    if kind == Kind::Disk {
                        // Demote every published snapshot to the disk tier
                        // (and drop the resident and live state), so the plan
                        // can only be served from disk.
                        loaded.bundle.session_clear(&mut loaded.gpu);
                        disk_before = snap_files(&disk_dir);
                        if disk_before == 0 {
                            return Err(format!(
                                "{arm}: trial {trial}: no snapshot file on disk after demoting the cache"
                            ));
                        }
                    }
                    let reused = loaded.bundle.session_plan(target, SessionRoute::Mtp);
                    if reused == 0 {
                        return Err(format!(
                            "{arm}: trial {trial}: the session cache offered no restore: this is not restore coverage"
                        ));
                    }
                    (target, reused)
                }
            };
            let (windows, produced) = request(
                &mut loaded,
                &mut drafter,
                prompt,
                reused,
                cfg,
                salt,
                trial,
                &mut ev,
            )?;
            loaded.bundle.session_commit();
            if kind == Kind::Disk {
                let after = snap_files(&disk_dir);
                if after >= disk_before {
                    return Err(format!(
                        "{arm}: trial {trial}: {disk_before} snapshot files before the restore and {after} after: the restore did not read the disk tier"
                    ));
                }
            }
            ev.trials += 1;
            if reused > 0 {
                ev.restored_trials += 1;
            }
            if produced < Q8EF_TOKENS {
                ev.early_trials += 1;
            }
            // Only a forced floor-off route replays byte for byte; the
            // adaptive route's choices depend on wall-clock prices.
            if forced && trial < REPLAYS {
                let mut scratch = Q8efEvidence::default();
                let (again, _) = request(
                    &mut loaded,
                    &mut drafter,
                    prompt,
                    0,
                    prime_cfg(seed),
                    salt,
                    trial,
                    &mut scratch,
                )?;
                let keys =
                    |ws: &[Q8efWindow]| ws.iter().map(Q8efWindow::replay_key).collect::<Vec<_>>();
                if keys(&again) != keys(&windows) {
                    let first = keys(&again)
                        .iter()
                        .zip(keys(&windows).iter())
                        .position(|(a, b)| a != b);
                    return Err(format!(
                        "{arm}: seed of trial {trial} replayed a different window stream (first difference at window {first:?}, {} vs {} windows)",
                        again.len(),
                        windows.len()
                    ));
                }
                ev.replays += 1;
            }
        }
        println!(
            "ARM {arm} trials {}..{} in {:.1}s: {} windows, keep {:?}, routes {:?}, replays {}",
            range.0,
            range.1,
            started.elapsed().as_secs_f64(),
            ev.windows,
            ev.keep,
            ev.routes,
            ev.replays
        );
        fs::write(
            &path,
            serde_json::to_vec(
                &json!({"arm": arm, "range": [range.0, range.1], "evidence": ev.to_json()?}),
            )
            .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        write_config(&path, &config)?;
        write_receipt(&path, &hw_identity::receipt(&loaded.gpu, model))?;
        Box::new(drafter).mtp_free(&mut loaded.gpu);
        loaded
            .bundle
            .free_gpu(&mut loaded.gpu)
            .map_err(|e| e.to_string())
    }

    /// The session arm alternates two prompts that share their first chunk,
    /// so each trial's plan restores the resident snapshot of the other.
    fn session_prompt_of<'a>(trial: usize, a: &'a [u32], a2: &'a [u32]) -> &'a [u32] {
        if trial % 2 == 0 {
            a2
        } else {
            a
        }
    }

    /// `.snap` files of a session-cache disk tier directory.
    fn snap_files(dir: &Path) -> usize {
        fs::read_dir(dir)
            .map(|entries| {
                entries
                    .filter(|e| {
                        e.as_ref()
                            .is_ok_and(|e| e.path().extension() == Some("snap".as_ref()))
                    })
                    .count()
            })
            .unwrap_or(0)
    }

    /// Arm the window and state taps for the whole process.
    fn arm_taps(loaded: &mut Loaded, drafter: &mut Qwen4MtpDrafter) {
        loaded.bundle.lab_observe(true, true);
        drafter.lab_observe_windows(LabWindowArm {
            armed: true,
            keep_logits: true,
        });
    }
}

fn q8ef_receipt(
    mode: &str,
    cohort: &Q8efCohort,
    arms: &BTreeMap<String, Q8efEvidence>,
    cover: &[Q8efCover],
    report: &str,
    failures: &[String],
    incomplete: &[String],
) -> Value {
    let verdict = if !failures.is_empty() {
        "FAIL"
    } else if !incomplete.is_empty() {
        "INCOMPLETE"
    } else {
        "PASS"
    };
    let arm_summary: BTreeMap<&str, Value> = arms
        .iter()
        .map(|(arm, ev)| {
            (
                arm.as_str(),
                json!({
                    "trials": ev.trials,
                    "early_trials": ev.early_trials,
                    "restored_trials": ev.restored_trials,
                    "windows": ev.windows,
                    "keep": {"no_drafts": ev.keep[0], "keep0": ev.keep[1], "partial": ev.keep[2], "full": ev.keep[3]},
                    "after_reset": ev.after_reset,
                    "after_keep0": ev.after_keep0,
                    "ring_half": ev.ring_half,
                    "graph_windows": ev.graph_windows,
                    "device_windows": ev.device_windows,
                    "independent_rows": ev.independent_rows,
                    "replays": ev.replays,
                    "routes": ev.routes,
                    "uniform_hist": ev.uniform_hist,
                }),
            )
        })
        .collect();
    let coverage: Vec<Value> = cover
        .iter()
        .map(|c| {
            json!({
                "item": c.id,
                "what": c.what,
                "observed": c.observed,
                "required": c.required,
                "status": if c.observed >= c.required { "COVERED" } else { "MISSING" },
            })
        })
        .collect();
    json!({
        "case": Q8EF_LAW_TEST,
        "verdict": verdict,
        "gdn_test_state_quant": mode,
        "alpha": ALPHA,
        "trials_per_arm": cohort.trials,
        "tokens_per_trial": Q8EF_TOKENS,
        "marginal_positions": Q8EF_MARGINALS,
        "joint_pairs": Q8EF_JOINTS,
        "scope": "stream indices >= 1: the tokens committed by tapped native-MTP windows, each judged under its own verify-route target row conditioned on the committed history",
        "unobserved_by_design": [
            "stream index 0 (the prefill seed draw: not a window, not observed by the tap; it is the conditioning seed of every window and its law is the AR comparison's)",
            "pipeline-parallel and expert-parallel mesh windows (not tapped)",
            "naive-interleaved proposals (draws only; this case runs the Leviathan verifier)"
        ],
        "arms": arm_summary,
        "coverage": coverage,
        "law_failures": failures,
        "incomplete": incomplete,
        "report": report,
    })
}

/// q8_ef verify-law case (see the section comment). The parent runs every
/// arm as fresh child processes (shards compose, configs are verified),
/// merges their evidence, judges each arm, evaluates coverage and writes
/// `q8ef_verify_law.coverage.json`. FAIL: a p-value at or under `ALPHA`.
/// INCOMPLETE: any short position, trial count or coverage item.
#[cfg(feature = "lab")]
#[test]
#[ignore = "needs a real HIP GPU, HIPFIRE_MTP_IDENTITY_MODEL and HIPFIRE_GDN_TEST_STATE_QUANT=q8_ef"]
fn sampled_mtp_q8ef_matches_verify_distribution_on_flash_next() {
    let model = model_path();
    let dir = out_dir();
    let cohort = Q8efCohort::from_env().unwrap_or_else(|e| panic!("{e}"));
    let (mode, _) = hw_identity::gdn_test_state_quant().unwrap_or_else(|e| panic!("{e}"));
    q8ef_check_plan(cohort.trials, mode).unwrap_or_else(|e| panic!("{e}"));
    let settings = DistSettings::from_env().unwrap_or_else(|e| panic!("{e}"));
    let model_id = hw_identity::model_identity(&model, &dir).unwrap_or_else(|e| panic!("{e}"));
    let mut arms: BTreeMap<String, Q8efEvidence> = BTreeMap::new();
    let (mut report, mut failures, mut incomplete) = (String::new(), Vec::new(), Vec::new());
    for arm in q8ef_all_arms() {
        let total = cohort.total(arm);
        let identity = arm_identity(arm, "leviathan", &model_id, &Q8EF_CONTROL_ENV);
        // An arm that errors (a window check, a restore that was not a
        // restore, a tap gap, an infrastructure failure) has no verdict: it
        // is recorded and the run is INCOMPLETE, never a PASS.
        let collected = q8ef_ensure_shards(&model, &dir, arm, total, &settings, &identity)
            .and_then(
                |shards: Vec<(usize, usize)>| -> Result<Q8efEvidence, String> {
                    let mut merged = Q8efEvidence::default();
                    for range in shards {
                        let part = read_json(&late_file(&dir, arm, range, "q8ef.json"))
                            .and_then(|v: Value| Q8efEvidence::from_json(&v["evidence"]))?;
                        merged.merge(part);
                    }
                    Ok(merged)
                },
            );
        arms.insert(
            arm.to_string(),
            collected.unwrap_or_else(|error| {
                incomplete.push(format!("{arm}: ARM ERROR (no verdict): {error}\n"));
                Q8efEvidence::default()
            }),
        );
    }
    for (arm, ev) in &arms {
        q8ef_judge(
            arm,
            ev,
            cohort.total(arm),
            &mut report,
            &mut failures,
            &mut incomplete,
        );
    }
    let cover = q8ef_coverage(&arms, &cohort);
    for item in &cover {
        if item.observed < item.required {
            incomplete.push(format!(
                "coverage {}: observed {} of {} required ({})\n",
                item.id, item.observed, item.required, item.what
            ));
        }
    }
    let receipt = q8ef_receipt(
        mode,
        &cohort,
        &arms,
        &cover,
        &report,
        &failures,
        &incomplete,
    );
    let path = dir.join(Q8EF_RECEIPT);
    write_json(&path, &receipt).unwrap_or_else(|e| panic!("{e}"));
    eprint!("{report}");
    eprintln!("receipt: {} verdict {}", path.display(), receipt["verdict"]);
    assert!(
        failures.is_empty() && incomplete.is_empty(),
        "{}",
        q8ef_failure_message(&dir, &failures, &incomplete)
    );
}

fn q8ef_failure_message(dir: &Path, failures: &[String], incomplete: &[String]) -> String {
    let mut message = String::new();
    if !failures.is_empty() {
        message.push_str(&format!(
            "DISTRIBUTION FAILURE (law): sampled q8_ef MTP differs from the verify-route law (alpha {ALPHA}; receipt in {}):\n{}",
            dir.display(),
            failures.concat()
        ));
    }
    if !incomplete.is_empty() {
        message.push_str(&format!(
            "INCOMPLETE (not a pass; receipt in {}):\n{}",
            dir.display(),
            incomplete.concat()
        ));
    }
    message
}

/// Without the lab tap the law case has nothing to observe: it refuses.
#[cfg(not(feature = "lab"))]
#[test]
#[ignore = "needs --features lab (the read-only window tap), a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL"]
fn sampled_mtp_q8ef_matches_verify_distribution_on_flash_next() {
    panic!(
        "{Q8EF_LAW_TEST}: REFUSED (unsupported), verdict INCOMPLETE: it observes the production sampled window through `hipfire_arch_qwen4::lab_window`, which exists only under --features lab; rebuild the test binary with `--features lab`"
    );
}

/// One q8_ef arm in a fresh process; spawned by the case above.
#[test]
#[ignore = "child process of sampled_mtp_q8ef_matches_verify_distribution_on_flash_next"]
fn sampled_mtp_q8ef_arm() {
    let Ok(arm) = std::env::var(ARM_ENV) else {
        return;
    };
    #[cfg(feature = "lab")]
    if let Err(error) = q8ef_run::run_arm(&arm, &model_path(), &out_dir()) {
        panic!("{arm} arm: {error}");
    }
    #[cfg(not(feature = "lab"))]
    panic!("{arm} arm: the q8_ef arms need --features lab");
}

mod q8ef_law_logic {
    use super::*;

    /// A row family with fixed, uneven weights (no randomness anywhere in
    /// these tests: every input is a deterministic quantile or grid point).
    fn family(width: u32, shift: u32) -> Vec<(u32, f32)> {
        let weights: Vec<f64> = (0..width)
            .map(|i| 1.0 + f64::from((i * 7919 + shift * 31) % 13))
            .collect();
        let mass: f64 = weights.iter().sum();
        weights
            .iter()
            .enumerate()
            .map(|(i, w)| (i as u32, (w / mass) as f32))
            .collect()
    }

    /// The token at quantile `x` of the normalized row and the in-token
    /// fraction `u` with `F(t^-) + u p(t) = x`.
    fn quantile(entries: &[(u32, f32)], x: f64) -> (u32, f64) {
        let mass: f64 = entries.iter().map(|&(_, p)| f64::from(p)).sum();
        let mut below = 0.0f64;
        for &(token, p) in entries {
            let p = f64::from(p) / mass;
            if x < below + p {
                return (token, ((x - below) / p).min(1.0 - 1e-12));
            }
            below += p;
        }
        (entries.last().expect("row").0, 1.0 - 1e-12)
    }

    /// `n` trials over three row widths; trial `i` of a family sits at the
    /// stratified quantile `(k + 0.5) / per` of its own row.
    fn stratified(n: usize) -> Vec<(Vec<(u32, f32)>, u32, f64)> {
        let families = [family(8, 1), family(12, 2), family(16, 3)];
        let per = n / 3;
        (0..n)
            .map(|i| {
                let entries = families[i % 3].clone();
                let (token, u) = quantile(&entries, ((i / 3) as f64 + 0.5) / per as f64);
                (entries, token, u)
            })
            .collect()
    }

    #[test]
    fn q8ef_pit_and_categories_accept_the_row_that_was_sampled() {
        let rows = stratified(3600);
        let pits: Vec<f64> = rows
            .iter()
            .map(|(entries, token, u)| q8ef_pit(entries, *token, *u).unwrap())
            .collect();
        let (_, df, p) = q8ef_pit_uniform(&pits, Q8EF_PIT_BINS).unwrap();
        assert_eq!(df, Q8EF_PIT_BINS - 1);
        assert!(
            p > 0.99,
            "stratified quantiles are uniform to the bin: p {p}"
        );
        let pairs: Vec<(&[(u32, f32)], u32)> = rows
            .iter()
            .map(|(entries, token, _)| (entries.as_slice(), *token))
            .collect();
        let (_, df, p) = q8ef_category_chi_square(&pairs).unwrap();
        assert_eq!(df, 15, "16 tokens, minus the reference cell");
        assert!(p > 0.5, "category counts sit on their expectation: p {p}");
    }

    #[test]
    fn q8ef_pit_and_categories_reject_a_different_row() {
        let rows = stratified(3600);
        // Judged under a uniform row over the same tokens: a different law.
        let flat = |entries: &[(u32, f32)]| -> Vec<(u32, f32)> {
            let w = 1.0 / entries.len() as f32;
            entries.iter().map(|&(t, _)| (t, w)).collect()
        };
        let pits: Vec<f64> = rows
            .iter()
            .map(|(entries, token, u)| q8ef_pit(&flat(entries), *token, *u).unwrap())
            .collect();
        let (_, _, p) = q8ef_pit_uniform(&pits, Q8EF_PIT_BINS).unwrap();
        assert!(p < ALPHA, "pit p {p}");
        let flats: Vec<Vec<(u32, f32)>> =
            rows.iter().map(|(entries, _, _)| flat(entries)).collect();
        let pairs: Vec<(&[(u32, f32)], u32)> = flats
            .iter()
            .zip(&rows)
            .map(|(row, (_, token, _))| (row.as_slice(), *token))
            .collect();
        let (_, _, p) = q8ef_category_chi_square(&pairs).unwrap();
        assert!(p < ALPHA, "category p {p}");
    }

    #[test]
    fn q8ef_category_test_pools_a_thin_tail() {
        // Three heavy tokens and a 400-token tail (each ~1.5 expected draws).
        let mut entries: Vec<(u32, f32)> = vec![(0, 0.4), (1, 0.25), (2, 0.15)];
        entries.extend((3..403u32).map(|t| (t, 0.2 / 400.0)));
        let tokens: Vec<u32> = (0..3000)
            .map(|k| quantile(&entries, (f64::from(k) + 0.5) / 3000.0).0)
            .collect();
        let rows: Vec<(&[(u32, f32)], u32)> =
            tokens.iter().map(|&t| (entries.as_slice(), t)).collect();
        let (_, df, p) = q8ef_category_chi_square(&rows).unwrap();
        assert_eq!(
            df, 3,
            "3 heavy cells + 1 pooled rest, minus the reference cell"
        );
        assert!(p > 0.5, "p {p}");

        // A tail too thin to stand alone merges into the heaviest cell.
        let mut thin: Vec<(u32, f32)> = vec![(0, 0.55), (1, 0.3), (2, 0.149)];
        thin.extend((3..13u32).map(|t| (t, 0.001 / 10.0)));
        let tokens: Vec<u32> = (0..3000)
            .map(|k| quantile(&thin, (f64::from(k) + 0.5) / 3000.0).0)
            .collect();
        let rows: Vec<(&[(u32, f32)], u32)> =
            tokens.iter().map(|&t| (thin.as_slice(), t)).collect();
        let (_, df, p) = q8ef_category_chi_square(&rows).unwrap();
        assert_eq!(df, 2);
        assert!(p > 0.5, "p {p}");
    }

    #[test]
    fn q8ef_statistics_refuse_what_they_cannot_judge() {
        let entries = [(1u32, 0.5f32), (2, 0.5)];
        assert!(q8ef_pit(&entries, 3, 0.5).is_err(), "outside the support");
        assert!(
            q8ef_pit(&[(1, 0.0), (2, 1.0)], 1, 0.5).is_err(),
            "zero probability"
        );
        assert!(q8ef_pit(&entries, 1, 1.0).is_err(), "u = 1");
        assert!(q8ef_pit(&entries, 1, -0.1).is_err(), "u < 0");
        assert!(q8ef_pit(&[(1, f32::NAN)], 1, 0.5).is_err(), "non-finite");
        assert!(q8ef_pit(&[(1, 0.0)], 1, 0.5).is_err(), "no mass");
        let pit = q8ef_pit(&entries, 2, 0.5).unwrap();
        assert!((pit - 0.75).abs() < 1e-12, "{pit}");
        assert!(
            q8ef_pit_uniform(&[0.5; 99], Q8EF_PIT_BINS).is_err(),
            "too few values"
        );
        assert!(
            q8ef_pit_uniform(&[1.5; 400], Q8EF_PIT_BINS).is_err(),
            "out of range"
        );
        assert!(
            q8ef_pit_joint(&[(0.5, 0.5); 79], Q8EF_JOINT_GRID).is_err(),
            "too few pairs"
        );
        let off_support: Vec<(&[(u32, f32)], u32)> = vec![(&entries[..], 9)];
        assert!(
            q8ef_category_chi_square(&off_support).is_err(),
            "off-support token"
        );
        // A deterministic row has no sampling variance to test against.
        let point = [(7u32, 1.0f32)];
        let rows: Vec<(&[(u32, f32)], u32)> = (0..50).map(|_| (&point[..], 7)).collect();
        assert!(q8ef_category_chi_square(&rows).is_err(), "single cell");
    }

    #[test]
    fn q8ef_joint_grid_accepts_a_uniform_square_and_rejects_dependence() {
        // 60 x 60 cell centres: exactly uniform on every 4 x 4 cell.
        let square: Vec<(f64, f64)> = (0..3600)
            .map(|i| {
                (
                    (i % 60) as f64 / 60.0 + 1.0 / 120.0,
                    (i / 60) as f64 / 60.0 + 1.0 / 120.0,
                )
            })
            .collect();
        let (stat, df, p) = q8ef_pit_joint(&square, Q8EF_JOINT_GRID).unwrap();
        assert_eq!(df, Q8EF_JOINT_GRID * Q8EF_JOINT_GRID - 1);
        assert!(stat < 1e-9 && p > 0.99, "stat {stat} p {p}");
        // Uniform marginals, perfectly dependent (b = a): only the diagonal cells.
        let diagonal: Vec<(f64, f64)> = square.iter().map(|&(a, _)| (a, a)).collect();
        let (_, _, p) = q8ef_pit_joint(&diagonal, Q8EF_JOINT_GRID).unwrap();
        assert!(p < ALPHA, "p {p}");
    }

    #[test]
    fn q8ef_plan_requires_the_mode_and_2000_trials() {
        assert!(q8ef_check_plan(2000, "q8_ef").is_ok());
        assert!(q8ef_check_plan(5000, "q8_ef").is_ok());
        assert!(q8ef_check_plan(1999, "q8_ef").unwrap_err().contains("2000"));
        for mode in ["q8", "fp32"] {
            let error = q8ef_check_plan(2000, mode).unwrap_err();
            assert!(
                error.contains(hw_identity::GDN_TEST_STATE_QUANT_ENV),
                "{error}"
            );
        }
    }
}

mod q8ef_window_logic {
    use super::*;

    fn window(
        drafts: &[u32],
        target_rows: &[&[(u32, f32)]],
        q_rows: &[&[(u32, f32)]],
        uniforms: &[f32],
        committed: &[u32],
        accepted: usize,
    ) -> Q8efWindow {
        Q8efWindow {
            drafts: drafts.to_vec(),
            target_rows: target_rows.iter().map(|r| r.to_vec()).collect(),
            q_rows: q_rows.iter().map(|r| r.to_vec()).collect(),
            uniforms: uniforms.to_vec(),
            committed: committed.to_vec(),
            accepted,
            eos: None,
            resets: (0, 0),
            ring_half: None,
            graph_launches: 0,
            route: "batched".into(),
            device: false,
            independent_rows: target_rows.len(),
            stream_uniforms: uniforms.to_vec(),
        }
    }

    const P: [(u32, f32); 2] = [(1, 0.6), (2, 0.4)];
    const POINT3: [(u32, f32); 1] = [(3, 1.0)];

    /// p = q accepts every draft for any u < 1, then the bonus is p's draw.
    fn full() -> Q8efWindow {
        window(
            &[1, 2],
            &[&P, &P, &P],
            &[&P, &P],
            &[0.9, 0.9, 0.5],
            &[1, 2, 1],
            2,
        )
    }

    /// Draft 3 has p = 0: rejected, replacement from the residual (= p).
    fn keep0() -> Q8efWindow {
        let mut w = window(&[3], &[&P, &P], &[&POINT3], &[0.1, 0.7], &[2], 0);
        w.resets = (4, 5);
        w
    }

    fn partial() -> Q8efWindow {
        let mut w = window(
            &[1, 3],
            &[&P, &P, &P],
            &[&P, &POINT3],
            &[0.2, 0.1, 0.1],
            &[1, 1],
            1,
        );
        w.resets = (5, 6);
        w
    }

    #[test]
    fn q8ef_replay_reproduces_full_reject_and_partial_windows() {
        assert_eq!(q8ef_replay(&full()).unwrap(), (vec![1, 2, 1], 2, 3));
        // Residual (p - q)+ = p; u = 0.7 lands in token 2's [0.6, 1.0).
        assert_eq!(q8ef_replay(&keep0()).unwrap(), (vec![2], 0, 2));
        // Accept draft 1, reject draft 3 (u 0.1), residual u = 0.1 -> token 1.
        assert_eq!(q8ef_replay(&partial()).unwrap(), (vec![1, 1], 1, 3));
        for w in [full(), keep0(), partial()] {
            q8ef_check_window(&w).unwrap();
        }
        assert_eq!(full().keep(), Q8efKeep::Full);
        assert_eq!(keep0().keep(), Q8efKeep::Keep0);
        assert_eq!(partial().keep(), Q8efKeep::Partial);
    }

    #[test]
    fn q8ef_replay_stops_at_an_accepted_eos_without_a_bonus() {
        let mut w = window(&[1, 2], &[&P, &P, &P], &[&P, &P], &[0.9], &[1], 1);
        w.eos = Some(1);
        assert_eq!(q8ef_replay(&w).unwrap(), (vec![1], 1, 1));
        q8ef_check_window(&w).unwrap();
    }

    #[test]
    fn q8ef_check_window_refuses_a_production_verdict_the_law_does_not_give() {
        let mut wrong_token = keep0();
        wrong_token.committed = vec![1];
        assert!(q8ef_check_window(&wrong_token)
            .unwrap_err()
            .contains("independent replay"));
        let mut wrong_accept = full();
        wrong_accept.accepted = 1;
        assert!(q8ef_check_window(&wrong_accept).is_err());
        let mut spare_uniform = full();
        spare_uniform.uniforms.push(0.3);
        assert!(q8ef_check_window(&spare_uniform)
            .unwrap_err()
            .contains("consumed 3"));
        let mut short_tape = full();
        short_tape.uniforms.pop();
        assert!(q8ef_check_window(&short_tape)
            .unwrap_err()
            .contains("more than"));
        let mut missing_row = full();
        missing_row.target_rows.pop();
        assert!(q8ef_check_window(&missing_row)
            .unwrap_err()
            .contains("no recorded target row"));
        let mut off_support = keep0();
        off_support.drafts = vec![9];
        assert!(q8ef_check_window(&off_support)
            .unwrap_err()
            .contains("support"));
        let mut bad_row = full();
        bad_row.target_rows[1] = vec![(1, f32::NAN)];
        assert!(q8ef_check_window(&bad_row).is_err());
    }

    #[test]
    fn q8ef_observe_counts_coverage_and_enforces_the_reset_contract() {
        let mut ev = Q8efEvidence::default();
        let mut ctx = Q8efTrial::new(0);
        let mut first = full();
        first.ring_half = Some(0);
        first.graph_launches = 2;
        ev.observe(1, &mut ctx, &first).unwrap();
        let mut second = keep0();
        second.ring_half = Some(1);
        ev.observe(1, &mut ctx, &second).unwrap();
        // The window after a keep0 reset.
        ev.observe(1, &mut ctx, &partial()).unwrap();
        ev.observe(1, &mut ctx, &full()).unwrap();
        assert_eq!(ev.windows, 4);
        assert_eq!(ev.keep, [0, 1, 1, 2]);
        assert_eq!(ev.after_keep0, 1);
        assert_eq!(ev.after_reset, 2, "after the keep0 and after the partial");
        assert_eq!(ev.ring_half, [1, 1]);
        assert_eq!(ev.graph_windows, 1);
        assert_eq!(ctx.next_index, 1 + 3 + 1 + 2 + 3);

        // A rejected window that kept the residual, and an accepted one that zeroed it.
        let mut kept = keep0();
        kept.resets = (7, 7);
        assert!(ev
            .observe(1, &mut ctx, &kept)
            .unwrap_err()
            .contains("unzeroed"));
        let mut zeroed = full();
        zeroed.resets = (7, 8);
        assert!(ev
            .observe(1, &mut ctx, &zeroed)
            .unwrap_err()
            .contains("zeroed the residual"));
    }

    #[test]
    fn q8ef_observe_records_pit_and_category_inputs_at_the_declared_positions() {
        let mut ev = Q8efEvidence::default();
        let mut ctx = Q8efTrial::new(7);
        // Stream indices 1, 2, 3 (a full window: positions 1..=3 are marginals).
        ev.observe(1, &mut ctx, &full()).unwrap();
        for index in [1usize, 2, 3] {
            assert!(Q8EF_MARGINALS.contains(&index));
            assert_eq!(ev.pits[&index].len(), 1, "{index}");
            assert_eq!(ev.pits[&index][0].0, 7);
            assert_eq!(ev.category[&index].len(), 1, "{index}");
        }
        assert_eq!(ev.category[&1][0].1, 1);
        assert_eq!(ev.category[&2][0].1, 2);
        // Token 2 under P = [(1, .6), (2, .4)]: PIT in [0.6, 1.0).
        let pit = ev.pits[&2][0].1;
        assert!((0.6..1.0).contains(&pit), "{pit}");
        // The draw is a pure function of (salt, trial, index).
        assert_eq!(q8ef_pit_draw(1, 7, 2), q8ef_pit_draw(1, 7, 2));
        assert_ne!(q8ef_pit_draw(1, 7, 2), q8ef_pit_draw(1, 8, 2));
        assert!((0.0..1.0).contains(&q8ef_pit_draw(1, 7, 2)));
    }

    #[test]
    fn q8ef_judge_is_incomplete_without_observations_and_a_failure_on_a_wrong_law() {
        let empty = Q8efEvidence::default();
        let (mut report, mut failures, mut incomplete) = (String::new(), Vec::new(), Vec::new());
        q8ef_judge(
            "batched",
            &empty,
            2000,
            &mut report,
            &mut failures,
            &mut incomplete,
        );
        assert!(failures.is_empty());
        // The trial count, every marginal (category + PIT), every joint,
        // every draft slot and the stream uniforms.
        assert_eq!(
            incomplete.len(),
            2 + 2 * Q8EF_MARGINALS.len() + Q8EF_JOINTS.len() + MTP_K + 1
        );

        // Position 1: 2000 trials of a fixed row whose observed tokens ignore it
        // (every trial emits token 2 although p(2) = 0.4): a law failure.
        let mut ev = Q8efEvidence::default();
        ev.trials = 2000;
        for trial in 0..2000usize {
            let pit = q8ef_pit(&P, 2, q8ef_pit_draw(1, trial, 1)).unwrap();
            ev.pits.entry(1).or_default().push((trial, pit));
            ev.category.entry(1).or_default().push((P.to_vec(), 2));
        }
        let (mut report, mut failures, mut incomplete) = (String::new(), Vec::new(), Vec::new());
        q8ef_judge(
            "batched",
            &ev,
            2000,
            &mut report,
            &mut failures,
            &mut incomplete,
        );
        assert!(
            failures.iter().any(|f| f.contains("category position 1")),
            "{failures:?}"
        );
        assert!(
            failures.iter().any(|f| f.contains("PIT position 1")),
            "{failures:?}"
        );
        assert!(report.contains("category position 1"));
    }

    #[test]
    fn q8ef_judge_never_judges_survivors() {
        // 1999 of 2000 trials reached position 1: INCOMPLETE, not a verdict
        // on the survivors (no 90% rule).
        let mut ev = Q8efEvidence::default();
        ev.trials = 2000;
        for trial in 0..1999usize {
            let pit = q8ef_pit(&P, 1, q8ef_pit_draw(1, trial, 1)).unwrap();
            ev.pits.entry(1).or_default().push((trial, pit));
            ev.category.entry(1).or_default().push((P.to_vec(), 1));
        }
        let (mut report, mut failures, mut incomplete) = (String::new(), Vec::new(), Vec::new());
        q8ef_judge(
            "batched",
            &ev,
            2000,
            &mut report,
            &mut failures,
            &mut incomplete,
        );
        assert!(failures.is_empty(), "{failures:?}");
        assert!(
            incomplete.iter().any(|m| m.contains("category position 1")
                && m.contains("1999 observations for 2000 trials")),
            "{incomplete:?}"
        );
        assert!(
            incomplete
                .iter()
                .any(|m| m.contains("PIT position 1")
                    && m.contains("1999 observations for 2000 trials")),
            "{incomplete:?}"
        );
    }

    #[test]
    fn q8ef_mixture_gap_is_zero_for_normalized_rows_and_catches_an_unnormalized_target() {
        let q = [(1u32, 0.2f32), (2, 0.3), (3, 0.5)];
        assert!(q8ef_mixture_gap(&P, &q) < 1e-6);
        assert!(q8ef_mixture_gap(&P, &P) < 1e-6);
        assert!(q8ef_mixture_gap(&P, &POINT3) < 1e-6);
        // A target row with mass 0.9 is not a distribution.
        let short = [(1u32, 0.9f32)];
        let q = [(1u32, 0.3f32), (2, 0.7)];
        assert!(q8ef_mixture_gap(&short, &q) > 0.05);
        let mut w = full();
        w.target_rows[0] = short.to_vec();
        w.q_rows[0] = q.to_vec();
        assert!(q8ef_check_window(&w).unwrap_err().contains("mixture"));
    }

    #[test]
    fn q8ef_observe_records_proposal_pits_and_the_uniform_histogram() {
        let mut ev = Q8efEvidence::default();
        let mut ctx = Q8efTrial::new(3);
        ev.observe(1, &mut ctx, &full()).unwrap();
        assert_eq!(ev.draft_pits[&0].len(), 1);
        assert_eq!(ev.draft_pits[&1].len(), 1);
        assert_eq!(
            ev.uniform_hist.iter().sum::<u64>(),
            3,
            "one per stream uniform"
        );
        assert_eq!(ev.routes["batched"], 1);
        let mut bad = full();
        bad.stream_uniforms = vec![1.5, 0.1, 0.1];
        assert!(ev
            .observe(1, &mut ctx, &bad)
            .unwrap_err()
            .contains("outside [0, 1)"));
    }

    #[test]
    fn q8ef_evidence_survives_json_and_shards_merge() {
        let mut a = Q8efEvidence::default();
        let mut ctx = Q8efTrial::new(0);
        a.observe(1, &mut ctx, &full()).unwrap();
        a.trials = 1;
        a.replays = 2;
        let mut b = Q8efEvidence::default();
        let mut ctx = Q8efTrial::new(1);
        b.observe(1, &mut ctx, &keep0()).unwrap();
        b.trials = 1;
        let back = Q8efEvidence::from_json(&a.to_json().unwrap()).unwrap();
        assert_eq!((back.trials, back.windows, back.replays), (1, 1, 2));
        assert_eq!(back.keep, a.keep);
        assert_eq!(back.category, a.category);
        for (index, pits) in &a.pits {
            for (x, y) in pits.iter().zip(&back.pits[index]) {
                assert_eq!(x.0, y.0);
                assert!((x.1 - y.1).abs() < 1e-12);
            }
        }
        let mut merged = back;
        merged.merge(b);
        assert_eq!(merged.trials, 2);
        assert_eq!(merged.windows, 2);
        assert_eq!(merged.keep, [0, 1, 0, 1]);
        assert_eq!(
            merged.pits[&1].len(),
            2,
            "trial 0 and trial 1 both reached index 1"
        );
        assert_eq!(merged.routes["batched"], 2);
    }

    #[test]
    fn q8ef_replay_key_ignores_absolute_resets_and_graph_launches_only() {
        let mut a = keep0();
        let mut b = keep0();
        b.resets = (40, 41);
        b.graph_launches = 9;
        assert_eq!(a.replay_key(), b.replay_key());
        a.uniforms[0] = 0.2;
        assert_ne!(a.replay_key(), b.replay_key());
        let mut c = keep0();
        c.resets = (4, 6);
        assert_ne!(
            keep0().replay_key(),
            c.replay_key(),
            "a different reset delta differs"
        );
    }

    fn cohort() -> Q8efCohort {
        Q8efCohort { trials: 2000 }
    }

    #[test]
    fn q8ef_cohort_sizes_each_arm_and_salts_are_unique() {
        let c = cohort();
        assert_eq!(c.total("q8ef_batched"), 2000);
        assert_eq!(c.total("q8ef_session_batched"), 2000);
        assert_eq!(c.total(Q8EF_DISK_ARM), 2000, "the disk arm is not shrunk");
        let mut salts: Vec<u64> = q8ef_all_arms().map(|a| q8ef_salt(a).unwrap()).collect();
        salts.sort_unstable();
        salts.dedup();
        assert_eq!(salts.len(), 6);
        assert!(q8ef_salt("ar").is_err());
        // The route knobs of the existing arm machinery reach every q8_ef arm.
        assert_eq!(arm_route("q8ef_batched"), Some("0"));
        assert_eq!(arm_route("q8ef_interleaved"), Some("1"));
        assert_eq!(arm_route("q8ef_adaptive"), None);
        for arm in Q8EF_RESTORE_ARMS {
            assert_eq!(arm_route(arm), Some("0"), "{arm} is forced batched");
        }
    }

    fn covered_arm() -> Q8efEvidence {
        let mut ev = Q8efEvidence::default();
        ev.windows = 100;
        ev.keep = [0, 20, 20, 60];
        ev.after_reset = 40;
        ev.after_keep0 = 20;
        ev.ring_half = [50, 50];
        ev.graph_windows = 90;
        ev.replays = REPLAYS as u64;
        ev.routes.insert("batched".into(), 100);
        ev.routes.insert("interleaved".into(), 100);
        ev
    }

    #[test]
    fn q8ef_coverage_is_missing_until_every_item_is_observed() {
        let c = cohort();
        let mut arms: BTreeMap<String, Q8efEvidence> = BTreeMap::new();
        let none = q8ef_coverage(&arms, &c);
        assert!(none.iter().all(|i| i.observed == 0 && i.required > 0));
        for arm in Q8EF_ROUTE_ARMS {
            arms.insert(arm.to_string(), covered_arm());
        }
        for (arm, restored) in [
            ("q8ef_session_batched", 2000usize),
            ("q8ef_prefix_batched", 2000),
            (Q8EF_DISK_ARM, 2000),
        ] {
            let mut ev = covered_arm();
            ev.restored_trials = restored;
            arms.insert(arm.to_string(), ev);
        }
        let all = q8ef_coverage(&arms, &c);
        let short: Vec<&str> = all
            .iter()
            .filter(|i| i.observed < i.required)
            .map(|i| i.id)
            .collect();
        assert!(short.is_empty(), "{short:?}");
        // One missing restore, one missing half: exactly those items are short.
        arms.get_mut("q8ef_prefix_batched").unwrap().restored_trials = 1999;
        for ev in arms.values_mut() {
            ev.ring_half[1] = 0;
        }
        let all = q8ef_coverage(&arms, &c);
        let short: Vec<&str> = all
            .iter()
            .filter(|i| i.observed < i.required)
            .map(|i| i.id)
            .collect();
        assert_eq!(short, ["prefix_restore_then_sample", "ring_half_1"]);
    }

    #[test]
    fn q8ef_receipt_verdict_is_fail_then_incomplete_then_pass() {
        let c = cohort();
        let arms = BTreeMap::new();
        let cover = q8ef_coverage(&arms, &c);
        let fail = q8ef_receipt("q8_ef", &c, &arms, &cover, "", &["x".into()], &["y".into()]);
        assert_eq!(fail["verdict"], "FAIL");
        let incomplete = q8ef_receipt("q8_ef", &c, &arms, &cover, "", &[], &["y".into()]);
        assert_eq!(incomplete["verdict"], "INCOMPLETE");
        let pass = q8ef_receipt("q8_ef", &c, &arms, &[], "", &[], &[]);
        assert_eq!(pass["verdict"], "PASS");
        assert!(pass["unobserved_by_design"].as_array().unwrap().len() == 3);
        assert!(pass["scope"]
            .as_str()
            .unwrap()
            .contains("stream indices >= 1"));
        assert_eq!(
            incomplete["coverage"].as_array().unwrap().len(),
            cover.len()
        );
        let message = q8ef_failure_message(Path::new("d"), &["x\n".into()], &["y\n".into()]);
        assert!(message.contains("DISTRIBUTION FAILURE (law)") && message.contains("INCOMPLETE"));
    }
}
