// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Greedy Flash-Next MTP token identity against AR on a real prompt.
//!
//! `#[ignore]`d: it needs a real HIP GPU and the Flash-Next artifact named by
//! `HIPFIRE_MTP_IDENTITY_MODEL`: the pinned split base
//! (`qwen3.8-flash-next-base.mq4-xts` with its `.ple-q8` / `.ple-q4` rows sidecar,
//! `.trunk-overlay` and `.expert-hotset` next to it) or a monolithic
//! `qwen3.8-flash-next-gptq3.mq4` with embedded rows (explicitly running it
//! without one fails rather than passing as model proof).
//!
//! Each arm loads through the product loader, `hipfire_loader::load_qwen4_rank`
//! (the serve load): source admission attaches the arch-preferred PLE rows sidecar
//! (`memory.ple_format`, auto = q8 on gfx1201, q4 on gfx1151) and the child prints
//! one `greedy_mtp_identity ple_rows: format=<fmt> origin=sidecar path=<p>` line
//! only when a sidecar is in use; the loader applies the trunk overlay and, on
//! gfx1201 discrete loads, the expert hot set, and places the remaining experts
//! with the automatic residency policy (`HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` pins
//! it). The placement reserves the native MTP head for every arm (K=3, the AR arm
//! included), which attaches in the loader. Prompts, identity/coverage
//! assertions and production-source checks are unchanged.
//!
//! The rendered `tests/fixtures/qwen4_mtp_identity_p1.txt` prompt (GDN
//! recurrent state per `HIPFIRE_GDN_TEST_STATE_QUANT`, Q8 by default,
//! `max_seq` 16384, K=3, temperature 0) is prefilled once per
//! arm and 1200 greedy token IDs are generated in process through the public
//! bundle and `MtpDrafter` contracts, not the daemon wire. Every arm runs in
//! its own fresh child process of this test binary, because route knobs are
//! read from a process-start snapshot:
//!
//! - `ar`: `forward_chunk_final` then single-row `forward_token_or_argmax`;
//! - `batched`: `HIPFIRE_MTP_INCREMENTAL=0`;
//! - `incremental`: `HIPFIRE_MTP_INCREMENTAL=1`;
//! - `adaptive`: route knob unset;
//! - `full-head`: `HIPFIRE_MTP_INCREMENTAL=0`, `HIPFIRE_MTP_DRAFT_HEAD=full`.
//! - `ngram-batched`, `ngram-incremental`, `ngram-adaptive`: the drafter is built
//!   `.with_ngram(Some(qwen4_ngram_mod_config(ngram_mod_triple_for_arch(16, arch))))
//!   .with_copy_lane(mtp_ngram_copy_enabled_for_arch(16, arch))`; the product
//!   switch `speculation.mtp_ngram_copy` / `HIPFIRE_MTP_NGRAM_COPY` is set
//!   explicitly by the child: `0` for these legacy takeover arms (`1` for the
//!   derived-copy/wide-copy arms). The `NGRAM` line prints `copy_lane=`, and
//!   the request is armed with `allow_ngram_modifier: true`, with route knob
//!   `HIPFIRE_MTP_INCREMENTAL` `0` / `1` / unset;
//! - `ngram-long`: `ngram-adaptive` with the inherited triple
//!   `HIPFIRE_NGRAM_MOD_N_MATCH=24 HIPFIRE_NGRAM_MOD_N_MIN=48 HIPFIRE_NGRAM_MOD_N_MAX=64`
//!   (`qwen4_ngram_mod_config` caps `n_max` to 63).
//!
//! Each MTP arm must emit AR's exact IDs. The route is confirmed from the
//! drafter's own `HIPFIRE_MTP_TRACE` window events (batched windows carry
//! `"baseline":true`), so a silent fallback cannot pass as coverage.
//! Per-arm IDs, decoded text and logs go to `HIPFIRE_MTP_IDENTITY_OUT`
//! (default: a fresh directory under the system temp dir).
//!
//! A second case, `seasons`, renders the serve battery's factual prompt
//! exactly as `hipfire serve` does for a single user message with thinking
//! disabled (`render_messages`, no system block; the token IDs are pinned to
//! the serve-captured ones) and generates 256 tokens. Its short context
//! exercises the 2..=8-row HC projections whose summation order once
//! differed from the single-row decode (Gate041z seasons divergence).
//!
//! A third case, `copy`, renders the committed copy-heavy
//! `benchmarks/prompts/code_edit_rewrite_copy.txt` (sha256 pinned) the way `p1`
//! is rendered and generates 1200 tokens, so the n-gram pool actually hits.
//!
//! The n-gram tests (`greedy_mtp_ngram_matches_ar_on_flash_next_{p1,seasons,copy}`)
//! run `ar` plus the four n-gram arms over a case with the same identity check.
//! `HIPFIRE_MTP_TRACE` windows are counted by `"source"` (`ngram` / `mtp`) and
//! `"verify_route"` (`batched` / `interleaved`): `ngram-batched` must have no
//! interleaved windows, `ngram-incremental` no batched windows, and on `copy`
//! every n-gram arm must take at least one n-gram window.
//! `greedy_mtp_ngram_batched_is_deterministic_on_flash_next_copy` runs
//! `ngram-batched` on `copy` in two fresh children and requires identical IDs.
//! The tests spawn GPU children: run with `--test-threads=1`.
//!
//! Every arm's child starts from a reset environment (the route, floor,
//! trace, timing, draft-head and n-gram variables a benchmark shell may have
//! left are removed, then the arm's own are set) and is checked against it.
//! Each arm writes `<arm>.identity.json`: the effective `HIPFIRE_*`
//! environment (levers, floor, route and graph knobs), the executable's
//! SHA-256, a hash of the production sources compiled in, the model's and its
//! `.ple-q4` / `.ple-q8` / `.trunk-overlay` / `.expert-hotset` sidecars' SHA-256, and a receipt (GPU arch / UUID /
//! PCI BDF, resolved graph flags, git state). The parent requires it to equal
//! the identity it expects, and removes any earlier result of the arm first.
//! `artifacts_used` records what the product load actually used, not what sits
//! next to the model: the PLE rows source (format, `sidecar` / `embedded`, path,
//! sha256), the trunk overlay the loader applied (the parent requires its
//! `qwen4 trunk overlay: applied` line; a skipped or REFUSED overlay fails) and
//! the expert hot set the load bound (the tier lease's sidecar sha256 must equal
//! the hashed `<model>.expert-hotset`; a hot set that applies on the device but
//! was skipped fails, and one is never applied where it is not applicable).
//! MTP arms also write `<arm>.windows.json`: per window the position, drafts,
//! accepted, committed, the drafter's own window / retirement counters and the
//! HIP call-count deltas around the step; the decoded text is `<arm>.txt`.
//!
//! # GDN state mode and graph policy
//!
//! `HIPFIRE_GDN_TEST_STATE_QUANT=q8|fp32` picks the GDN recurrent-state format
//! (unset = `q8`, the historical fixture). `q8_ef` refuses before model loading:
//! use `q8_ef_product_routes_hw`, where cross-route equality is diagnostic.
//! The value is passed as `state_request` to `resolve_state_format`
//! and the run fails unless the resolved `GdnStateFormat` is the requested one
//! (the product's `HIPFIRE_STATE_QUANT` is never read). The parent resolves the
//! mode once and sets it in every child's environment, so it is part of each
//! arm's identity (`config.env`) and a child started without it refuses; every
//! arm records `gdn_state` (`request`, resolved `name`) in `<arm>.identity.json`
//! and the parent writes both as the `summary.txt` header. The per-arm expert
//! placement and native-MTP sizing use the resolved format. G2 (F32 MTP == AR)
//! runs as
//! `HIPFIRE_GDN_TEST_STATE_QUANT=fp32 HIPFIRE_MTP_IDENTITY_MODEL=$M <test-bin> --ignored --test-threads=1 greedy_mtp_matches_ar_on_flash_next`.
//!
//! Qwen4 reads no `HIPFIRE_GRAPH` / `HIPFIRE_AR_GRAPH` / `HIPFIRE_GRAPH_MOE`
//! (only Qwen3.5 does; `gpu_forward.rs` documents that no Qwen4 production path
//! opens a hipGraph capture). Its graph-like switch is the retained-replay
//! backend `HIPFIRE_REPLAY_BACKEND` (`Gpu::init` ->
//! `ReplayController::from_config`): `hip` (default when unset here) launches
//! eagerly, `auto` records the AR decode forward once and replays it; verify
//! forwards (2+ rows) always launch eager. The parent resolves it (`hip`/`off`
//! -> `hip`, `auto`/`redline` -> `auto`, `shadow`; others refused), forwards it
//! to every child, each child fails unless `gpu.replay.request()` is that
//! backend, and `<arm>.identity.json` (`graph_policy`) plus the `summary.txt`
//! header record it. Run G2 as a "graph0" parent with `HIPFIRE_REPLAY_BACKEND=hip`
//! and a "graph1" parent with `HIPFIRE_REPLAY_BACKEND=auto`.
//!
//! `greedy_mtp_derived_k_vs_incumbent_on_flash_next_prompts` compares fresh
//! adaptive incumbent/derived children on all three prompts, with n-grams
//! armed, copy lane off and greedy sampling. It requires eligible request
//! receipts and at least one changed decision. Every generated ID of both
//! arms must equal an independent AR child, even when their verify routes
//! differ. Decoded attractor checks are additional, not an identity oracle.
//! This is not HIPFIRE_GRAPH coverage. Both arms run the load's unqualified default K (no
//! explicit `mtp_k`), so the derived chooser's legal ceiling is the qualified
//! row-exact rows less the seed, never a shallow test K; every `derived_k`
//! receipt carries its price mode (`off_pool` here), legal cap, row limit and
//! width, and none may exceed the qualified rows.
//!
//! `greedy_mtp_derived_k_copy_vs_incumbent_on_flash_next_prompts` is the same
//! pair with the n-gram copy lane on (`HIPFIRE_MTP_NGRAM_COPY=1`, mode
//! `on_copy`; same arm names, child `greedy_mtp_derived_k_copy_identity_arm`).
//! Each arm's `request` receipt must name `copy_lane` true. On the copy-heavy
//! prompt the derived arm must show `window` lines of source `ngram` and
//! `derived_k` lines of source `ngram` that drafted; a window whose mode has no
//! measured price prints `derived_k_unsupported` and runs the incumbent, so an
//! unmeasured OnCopy profile fails this gate instead of passing on the flag.
//!
//! `greedy_mtp_wide_windows_match_ar_on_flash_next` is PLAN G3: native and
//! real-pool copy windows at R=9,16,17,32,33,48,63,64; native/copy/native
//! return; interior captured-prefix repair; native retirement with copy still
//! active; ragged token-budget tails; a deterministic AR-derived diagnostic
//! EOS; and unforced product-derived native/copy choosers. Fresh children
//! explicitly carry width, source scenario, F32/Q8 state and KV modes. All
//! IDs are compared to AR, not only final text. `coverage.json` contains
//! actual `window`/post-success `window_commit` receipt widths for the cells
//! actually reached; a filtered run records its reached subset, never the
//! whole G3. The five existing selectors choose one declared cell; all absent
//! runs the whole matrix. A configured K63, missing receipt, missing partial
//! repair or unexecuted cell fails.
//! Build with `--features reference-parity`; absent controls fail explicitly.
//! Admission and identity use the oracle's live effective row-cap getter;
//! each actual window's `row_limit` receipt remains authoritative.
//! Run this battery once with `HIPFIRE_REPLAY_BACKEND=hip`, once with `auto`;
//! identities record the resolved backend. These are eager verify windows
//! bound to the actual singleton AR policy, not hipGraph capture evidence.

use hipfire_arch_qwen4::expert_residency as residency;
use hipfire_arch_qwen4::mtp_spec::{
    native_mtp_row_capture, qwen4_ngram_mod_config, Qwen4MtpDrafter,
};
use hipfire_arch_qwen4::weights::Qwen4TierRecipe;
use hipfire_arch_qwen4::Qwen4KvBackend;
use hipfire_dispatch::pipeline::DraftHeadPolicy;
use hipfire_loader::{load_qwen4_rank, Qwen4RankShared};
use hipfire_runtime::loader_api::{ModelSource, SpecLoadCfg};
use hipfire_runtime::ple_sidecar::{PleRowsChoice, PleRowsOrigin};
use hipfire_runtime::prompt_frame::{JinjaChatFrame, Message, Role};
use hipfire_runtime::spec::{MtpDrafter, SpecRequestConfig};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{DType, Gpu};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[path = "common/hw_identity.rs"]
mod hw_identity;
use hw_identity::Counters;

const MODEL_ENV: &str = "HIPFIRE_MTP_IDENTITY_MODEL";
const OUT_ENV: &str = "HIPFIRE_MTP_IDENTITY_OUT";
const ARM_ENV: &str = "HIPFIRE_MTP_IDENTITY_ARM";
const CASE_ENV: &str = "HIPFIRE_MTP_IDENTITY_CASE";
const ARM_TEST: &str = "greedy_mtp_identity_arm";
const DERIVED_ARM_TEST: &str = "greedy_mtp_derived_k_identity_arm";
const DERIVED_COPY_ARM_TEST: &str = "greedy_mtp_derived_k_copy_identity_arm";
const DERIVED_K_ENV: &str = "HIPFIRE_QWEN4_MTP_DERIVED_K";
/// Harness-control variables: they differ between a parent and each child
/// without changing what the arm computes, so identities exclude them.
const CONTROL_ENV: [&str; 4] = [MODEL_ENV, OUT_ENV, ARM_ENV, CASE_ENV];
/// The launch-backend variable, the only graph-like switch Qwen4 honors in
/// process: `ReplayController::from_config` (`rdna-compute/src/replay.rs`,
/// called from `Gpu::init`) reads it. `HIPFIRE_GRAPH` / `HIPFIRE_AR_GRAPH` /
/// `HIPFIRE_GRAPH_MOE` are read by Qwen3.5 only; no Qwen4 source reads them.
const REPLAY_BACKEND_ENV: &str = "HIPFIRE_REPLAY_BACKEND";
const GRAPH_POLICY_NOTE: &str = "graph policy is the retained-replay backend: hip = eager launches, auto = AR decode records then replays a retained body; verify forwards (2+ rows) always launch eager; HIPFIRE_GRAPH is not read by Qwen4";

/// The launch backend this process requests, canonical (`hip`, `auto`,
/// `shadow`); unset or empty is `hip`, as `ReplayBackendRequest::from_config`
/// reads it. An unknown value is refused, never relabelled.
fn replay_backend_request() -> Result<&'static str, String> {
    match std::env::var(REPLAY_BACKEND_ENV) {
        Err(std::env::VarError::NotPresent) => Ok("hip"),
        Err(std::env::VarError::NotUnicode(value)) => {
            Err(format!("{REPLAY_BACKEND_ENV}={value:?} is not UTF-8"))
        }
        Ok(value) => match value.to_ascii_lowercase().as_str() {
            "" | "hip" | "off" => Ok("hip"),
            "auto" | "redline" => Ok("auto"),
            "shadow" => Ok("shadow"),
            other => Err(format!(
                "{REPLAY_BACKEND_ENV}={other:?} is not a known backend (accepted: hip, off, auto, redline, shadow)"
            )),
        },
    }
}

/// The canonical name of the backend the runtime resolved.
fn replay_backend_name(request: rdna_compute::replay::ReplayBackendRequest) -> &'static str {
    use rdna_compute::replay::ReplayBackendRequest;
    match request {
        ReplayBackendRequest::Hip => "hip",
        ReplayBackendRequest::Shadow => "shadow",
        ReplayBackendRequest::Auto => "auto",
    }
}
/// Column names of a `<arm>.windows.json` row (the HIP call counts follow).
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

const PROMPT: &[u8] = include_bytes!("fixtures/qwen4_mtp_identity_p1.txt");
/// `hipx:~/pm-wave/fns3/in/p1.txt` (md5 `c91f8380a035dc842416b1be5efa8197`).
const PROMPT_SHA256: &str = "4ba1d4d971b99ace1fd294e1cdb087c432d87a155c203bddef87552ce02353a7";
const PROMPT_TOKENS: usize = 1047;
const MAX_SEQ: usize = 16384;
const MTP_K: usize = 3;

/// The serve battery's factual prompt (`scripts/serve_harness.py`).
const SEASONS: &str = "What causes the seasons on Earth? Answer in exactly three sentences.";
/// Prompt IDs `hipfire serve` prefilled for SEASONS (Gate041z, daemon diag dump).
const SEASONS_IDS: [u32; 25] = [
    248045, 846, 198, 3710, 10814, 279, 15127, 383, 8964, 30, 21134, 303, 6681, 2250, 22157, 13,
    248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
];

/// The committed copy-heavy prompt (`benchmarks/prompts/code_edit_rewrite_copy.txt`).
const COPY_PROMPT: &[u8] = include_bytes!("../../../benchmarks/prompts/code_edit_rewrite_copy.txt");
const COPY_PROMPT_SHA256: &str = "5f223a71ca57719c256968a15031f78483459ac22fb649690e3f9077b211e79d";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Case {
    P1,
    Seasons,
    Copy,
}

impl Case {
    fn name(self) -> &'static str {
        match self {
            Case::P1 => "p1",
            Case::Seasons => "seasons",
            Case::Copy => "copy",
        }
    }

    fn generated(self) -> usize {
        match self {
            Case::P1 => 1200,
            Case::Seasons => 256,
            Case::Copy => 1200,
        }
    }
}

#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    incremental: Option<&'static str>,
    draft_head: Option<&'static str>,
    route: Route,
    /// Build the drafter with the n-gram pool and arm `allow_ngram_modifier`.
    ngram: bool,
    /// `HIPFIRE_NGRAM_MOD_{N_MATCH,N_MIN,N_MAX}` for the child, if overridden.
    ngram_triple: Option<(&'static str, &'static str, &'static str)>,
    /// None: legacy arm; Some(false/true): derived-policy incumbent/candidate.
    derived_k: Option<bool>,
    /// Explicit product copy-lane switch for this arm (`speculation.mtp_ngram_copy` /
    /// `HIPFIRE_MTP_NGRAM_COPY`), never inherited from a benchmark shell.
    /// Derived COPYON and wide-copy scenarios enable it.
    copy_lane: bool,
    wide: Option<WideRun>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Route {
    Ar,
    Batched,
    Incremental,
    Adaptive,
}

/// Explicit child configuration, included in config.env, never inferred from
/// an inherited shell knob. The row override is laboratory admission, not a
/// production-capability or performance certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WideRun {
    rows: &'static str,
    scenario: &'static str,
    state: &'static str,
    kv: &'static str,
}

const WIDE_ROWS: [&str; 8] = ["9", "16", "17", "32", "33", "48", "63", "64"];
const WIDE_ROWS_ENV: &str = "HIPFIRE_MTP_IDENTITY_VERIFY_ROWS";
const WIDE_SCENARIO_ENV: &str = "HIPFIRE_MTP_IDENTITY_SCENARIO";
const WIDE_CHILD: &str = "greedy_mtp_wide_identity_arm";

fn wide_arm(run: WideRun) -> Arm {
    Arm {
        name: run.scenario,
        route: if run.scenario == "ar" {
            Route::Ar
        } else {
            Route::Adaptive
        },
        incremental: None,
        ngram: matches!(
            run.scenario,
            "copy" | "return" | "retire" | "product" | "product-copy"
        ),
        copy_lane: matches!(run.scenario, "copy" | "return" | "retire" | "product-copy"),
        derived_k: matches!(run.scenario, "product" | "product-copy").then_some(true),
        wide: Some(run),
        ..ARMS[0]
    }
}

/// Declared G3 candidate scenarios for a prompt at a requested width. The
/// product choosers run only at the row64 ceiling; `ar` is the reference,
/// never a candidate. No new scenario admission.
fn wide_scenarios(case: Case, rows_is_64: bool) -> &'static [&'static str] {
    match (case, rows_is_64) {
        (Case::P1, true) => &["native", "budget", "eos", "product"],
        (Case::P1, false) => &["native", "budget", "eos"],
        (Case::Copy, true) => &["copy", "return", "retire", "product-copy"],
        (Case::Copy, false) => &["copy", "return", "retire"],
        (Case::Seasons, _) => unreachable!("wide prompt matrix"),
    }
}

/// Pure selector for one declared G3 cell: `[rows, scenario, case, state, kv]`.
/// All absent runs the whole matrix; any present requires all five, each
/// naming a declared cell (`ar` is the reference, never a candidate).
fn parse_wide_selector(selector: [Option<&str>; 5]) -> Result<Option<(WideRun, Case)>, String> {
    if selector == [None, None, None, None, None] {
        return Ok(None);
    }
    let missing = selector
        .iter()
        .zip([
            WIDE_ROWS_ENV,
            WIDE_SCENARIO_ENV,
            CASE_ENV,
            hw_identity::GDN_TEST_STATE_QUANT_ENV,
            "HIPFIRE_KV_MODE",
        ])
        .find(|(value, _)| value.is_none())
        .map(|(_, name)| name);
    if let Some(name) = missing {
        return Err(format!(
            "partial wide selector at {name}: all five of {WIDE_ROWS_ENV}, {WIDE_SCENARIO_ENV}, {CASE_ENV}, {} and HIPFIRE_KV_MODE are required together",
            hw_identity::GDN_TEST_STATE_QUANT_ENV
        ));
    }
    let [rows, scenario, case, state, kv] = selector.map(|value| value.expect("all five present"));
    let rows = WIDE_ROWS
        .into_iter()
        .find(|candidate| *candidate == rows)
        .ok_or_else(|| format!("{WIDE_ROWS_ENV}={rows:?} is not a declared G3 width"))?;
    let case = match case {
        "p1" => Case::P1,
        "copy" => Case::Copy,
        other => {
            return Err(format!(
                "{CASE_ENV}={other:?} is not a declared G3 prompt (accepted: p1, copy)"
            ));
        }
    };
    let state = match state {
        "q8" => "q8",
        "fp32" => "fp32",
        other => {
            return Err(format!(
                "{}={other:?} is not a declared G3 state (accepted: q8, fp32)",
                hw_identity::GDN_TEST_STATE_QUANT_ENV
            ));
        }
    };
    let kv = match kv {
        "q8" => "q8",
        "f32" => "f32",
        other => {
            return Err(format!(
                "HIPFIRE_KV_MODE={other:?} is not a declared G3 KV mode (accepted: q8, f32)"
            ));
        }
    };
    let scenario = wide_scenarios(case, rows == "64")
        .iter()
        .find(|&&candidate| candidate == scenario)
        .copied()
        .ok_or_else(|| {
            format!(
                "{WIDE_SCENARIO_ENV}={scenario:?} is not a declared candidate for {} at R{rows} (ar is the reference, never a candidate)",
                case.name()
            )
        })?;
    Ok(Some((
        WideRun {
            rows,
            scenario,
            state,
            kv,
        },
        case,
    )))
}

/// Read the five existing wide selectors. All absent runs the whole matrix;
/// any present requires all five and a declared cell. Rejects non-UTF8 and
/// partial/invalid selectors before any model, directory or child work.
fn wide_selector() -> Result<Option<(WideRun, Case)>, String> {
    fn selector_env(name: &str) -> Result<Option<String>, String> {
        match std::env::var(name) {
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(value)) => {
                Err(format!("{name}={value:?} is not UTF-8"))
            }
            Ok(value) => Ok(Some(value)),
        }
    }
    let rows = selector_env(WIDE_ROWS_ENV)?;
    let scenario = selector_env(WIDE_SCENARIO_ENV)?;
    let case = selector_env(CASE_ENV)?;
    let state = selector_env(hw_identity::GDN_TEST_STATE_QUANT_ENV)?;
    let kv = selector_env("HIPFIRE_KV_MODE")?;
    parse_wide_selector([
        rows.as_deref(),
        scenario.as_deref(),
        case.as_deref(),
        state.as_deref(),
        kv.as_deref(),
    ])
}

const ARMS: [Arm; 5] = [
    Arm {
        name: "ar",
        incremental: None,
        draft_head: None,
        route: Route::Ar,
        ngram: false,
        ngram_triple: None,
        derived_k: None,
        copy_lane: false,
        wide: None,
    },
    Arm {
        name: "batched",
        incremental: Some("0"),
        draft_head: None,
        route: Route::Batched,
        ngram: false,
        ngram_triple: None,
        derived_k: None,
        copy_lane: false,
        wide: None,
    },
    Arm {
        name: "incremental",
        incremental: Some("1"),
        draft_head: None,
        route: Route::Incremental,
        ngram: false,
        ngram_triple: None,
        derived_k: None,
        copy_lane: false,
        wide: None,
    },
    Arm {
        name: "adaptive",
        incremental: None,
        draft_head: None,
        route: Route::Adaptive,
        ngram: false,
        ngram_triple: None,
        derived_k: None,
        copy_lane: false,
        wide: None,
    },
    Arm {
        name: "full-head",
        incremental: Some("0"),
        draft_head: Some("full"),
        route: Route::Batched,
        ngram: false,
        ngram_triple: None,
        derived_k: None,
        copy_lane: false,
        wide: None,
    },
];

/// AR plus the n-gram combined arms.
const NGRAM_ARMS: [Arm; 5] = [
    ARMS[0],
    Arm {
        name: "ngram-batched",
        incremental: Some("0"),
        draft_head: None,
        route: Route::Batched,
        ngram: true,
        ngram_triple: None,
        derived_k: None,
        copy_lane: false,
        wide: None,
    },
    Arm {
        name: "ngram-incremental",
        incremental: Some("1"),
        draft_head: None,
        route: Route::Incremental,
        ngram: true,
        ngram_triple: None,
        derived_k: None,
        copy_lane: false,
        wide: None,
    },
    Arm {
        name: "ngram-adaptive",
        incremental: None,
        draft_head: None,
        route: Route::Adaptive,
        ngram: true,
        ngram_triple: None,
        derived_k: None,
        copy_lane: false,
        wide: None,
    },
    Arm {
        name: "ngram-long",
        incremental: None,
        draft_head: None,
        route: Route::Adaptive,
        ngram: true,
        ngram_triple: Some(("24", "48", "64")),
        derived_k: None,
        copy_lane: false,
        wide: None,
    },
];

/// Product adaptive routing and floor, differing only in the derived-K flag.
const DERIVED_ARMS: [Arm; 2] = [
    Arm {
        name: "derived-incumbent",
        derived_k: Some(false),
        ..NGRAM_ARMS[3]
    },
    Arm {
        name: "derived",
        derived_k: Some(true),
        ..NGRAM_ARMS[3]
    },
];

/// The COPYON pair: the same two arms with the n-gram copy lane on. Same arm
/// names, so the receipts of a campaign read both modes alike.
const DERIVED_COPY_ARMS: [Arm; 2] = [
    Arm {
        copy_lane: true,
        ..DERIVED_ARMS[0]
    },
    Arm {
        copy_lane: true,
        ..DERIVED_ARMS[1]
    },
];

fn model_path() -> PathBuf {
    let (_, format) = hw_identity::gdn_test_state_quant().unwrap_or_else(|e| panic!("{e}"));
    assert!(
        format != hipfire_arch_qwen4::GdnStateFormat::Q8Ef,
        "q8_ef cross-route equality is diagnostic; run q8_ef_product_routes_hw instead of greedy_mtp_identity_hw"
    );
    let path = std::env::var_os(MODEL_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{MODEL_ENV} must name the Flash-Next base (qwen3.8-flash-next-base.mq4-xts or qwen3.8-flash-next-gptq3.mq4)"));
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
            std::env::temp_dir().join(format!("qwen4-mtp-identity-{}", std::process::id()))
        });
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    dir
}

/// Batched and incremental window counts from the drafter's trace events.
fn window_routes(stderr: &str) -> (usize, usize) {
    let windows = stderr.lines().filter(|line| {
        line.starts_with("QWEN4_MTP_TRACE ") && line.contains("\"event\":\"window\"")
    });
    windows.fold((0, 0), |(batched, incremental), line| {
        if line.contains("\"baseline\":true") {
            (batched + 1, incremental)
        } else {
            (batched, incremental + 1)
        }
    })
}

/// Trace windows by `"source"` and `"verify_route"`.
#[derive(Default, Clone, Copy)]
struct WindowCounts {
    ngram: usize,
    mtp: usize,
    batched: usize,
    interleaved: usize,
}

fn window_counts(stderr: &str) -> WindowCounts {
    let mut counts = WindowCounts::default();
    let windows = stderr.lines().filter(|line| {
        line.starts_with("QWEN4_MTP_TRACE ") && line.contains("\"event\":\"window\"")
    });
    for line in windows {
        if line.contains("\"source\":\"ngram\"") {
            counts.ngram += 1;
        } else if line.contains("\"source\":\"mtp\"") {
            counts.mtp += 1;
        }
        if line.contains("\"verify_route\":\"batched\"") {
            counts.batched += 1;
        } else if line.contains("\"verify_route\":\"interleaved\"") {
            counts.interleaved += 1;
        }
    }
    counts
}

/// Route and source assertions for an n-gram arm; returns the counts.
fn ngram_window_check(arm: Arm, case: Case, stderr: &str) -> WindowCounts {
    let w = window_counts(stderr);
    assert_eq!(
        w.ngram + w.mtp,
        w.batched + w.interleaved,
        "{} arm: every window must carry a source and a verify_route",
        arm.name
    );
    let route_ok = match arm.route {
        Route::Ar => w.batched == 0 && w.interleaved == 0,
        Route::Batched => w.batched > 0 && w.interleaved == 0,
        Route::Incremental => w.batched == 0 && w.interleaved > 0,
        Route::Adaptive => w.batched > 0,
    };
    assert!(
        route_ok,
        "{} arm took the wrong verify_route: {} batched and {} interleaved windows ({} ngram, {} mtp)",
        arm.name, w.batched, w.interleaved, w.ngram, w.mtp
    );
    if case == Case::Copy {
        assert!(
            w.ngram > 0,
            "{} arm took no n-gram window on the copy-heavy case ({} mtp windows)",
            arm.name,
            w.mtp
        );
    }
    w
}

fn first_difference(ar: &[u32], mtp: &[u32]) -> Option<usize> {
    ar.iter()
        .zip(mtp)
        .position(|(a, b)| a != b)
        .or_else(|| (ar.len() != mtp.len()).then(|| ar.len().min(mtp.len())))
}

/// What the product load does with `<model>.expert-hotset` on this device,
/// decided from the loader's own public policy (`HotsetPolicy`, `PlacementPolicy`,
/// `decide_whole_first`, `preflight`) with the inputs `load_qwen4_rank` builds for
/// a single, non-pipelined load: `Ok(path)` applies, `Err(reason)` does not.
fn hotset_expectation(
    model: &Path,
    gpu: &Gpu,
    recipe: &Qwen4TierRecipe,
) -> Result<Result<PathBuf, String>, String> {
    use hipfire_arch_qwen4::expert_hotset as hotsets;
    let policy = hotsets::HotsetPolicy::from_env()?;
    let placement = hotsets::PlacementPolicy::from_env()?;
    let env = hotsets::HotsetEnv {
        gpu_arch: &gpu.arch,
        unified_memory: gpu.is_uma(),
        expert_parallel: false,
        explicit_vram_layers: residency::expert_vram_layers_from_env()?.is_some(),
        source: match recipe {
            Qwen4TierRecipe::FlashNextPacked { .. } => hotsets::HotsetSource::Aggregate,
            Qwen4TierRecipe::FlashNextSlab { .. } => hotsets::HotsetSource::SlabArena,
            Qwen4TierRecipe::StockFp8 => hotsets::HotsetSource::StockFp8,
            _ => hotsets::HotsetSource::Nonaggregate("not an aggregate MQ4 artifact".into()),
        },
    };
    if hotsets::decide_whole_first(placement, &env, false) == hotsets::WholeFirstDecision::Active {
        return Ok(Err(format!(
            "{}=whole-first keeps whole-layer placement",
            hotsets::PLACEMENT_POLICY_ENV
        )));
    }
    Ok(hotsets::preflight(&policy, model, &env).map_err(|skip| skip.to_string()))
}

/// The hot set the load actually bound (`applied`, the sidecar sha256 the tier
/// lease carries) against the expectation. A hot set the device should have
/// used but the load skipped, or one applied where it should not be, fails the
/// arm; an applied one must be the very file the identity hashed.
fn hotset_receipt(
    expected: &Result<PathBuf, String>,
    applied: Option<&str>,
    model_id: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    match (expected, applied) {
        (Ok(path), Some(sha)) => {
            if model_id["expert-hotset"]["sha256"].as_str() != Some(sha) {
                return Err(format!(
                    "the load applied a hot set with sha256 {sha}, not the hashed {} ({})",
                    path.display(),
                    model_id["expert-hotset"]["sha256"]
                ));
            }
            Ok(serde_json::json!({
                "applied": true,
                "path": path.display().to_string(),
                "sha256": sha,
            }))
        }
        (Ok(path), None) => Err(format!(
            "{} applies on this device but the load placed no hot set (whole-layer placement); refusing an unfaithful placement",
            path.display()
        )),
        (Err(reason), None) => Ok(serde_json::json!({
            "applied": false,
            "not_applicable": reason,
        })),
        (Err(reason), Some(sha)) => Err(format!(
            "the load applied hot set {sha} though it is not applicable here ({reason})"
        )),
    }
}

/// The PLE rows source the load resolved. A sidecar must be exactly the
/// `<model>.ple-<format>` file the identity hashed.
fn ple_rows_receipt(
    choice: Option<&PleRowsChoice>,
    model: &Path,
    model_id: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let Some(choice) = choice else {
        return Ok(serde_json::json!({"format": null, "origin": "none", "path": null}));
    };
    let format = choice.format.name();
    match &choice.origin {
        PleRowsOrigin::Embedded => Ok(serde_json::json!({
            "format": format,
            "origin": "embedded",
            "path": model.display().to_string(),
        })),
        PleRowsOrigin::Sidecar(path) => {
            let mut hashed = model.as_os_str().to_os_string();
            hashed.push(format!(".ple-{format}"));
            if path.as_os_str() != hashed.as_os_str() {
                return Err(format!(
                    "PLE rows resolved to {}, not the hashed {}",
                    path.display(),
                    PathBuf::from(hashed).display()
                ));
            }
            let key = format!("ple-{format}");
            let sha = model_id[key.as_str()]["sha256"]
                .as_str()
                .ok_or_else(|| format!("identity holds no sha256 for {key}"))?;
            Ok(serde_json::json!({
                "format": format,
                "origin": "sidecar",
                "path": path.display().to_string(),
                "sha256": sha,
            }))
        }
    }
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (Flash-Next base with matching PLE rows)"]
fn greedy_mtp_matches_ar_on_flash_next_p1() {
    run_case(Case::P1, &ARMS, "p1");
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (Flash-Next base with matching PLE rows)"]
fn greedy_mtp_matches_ar_on_flash_next_seasons() {
    run_case(Case::Seasons, &ARMS, "seasons");
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (Flash-Next base with matching PLE rows)"]
fn greedy_mtp_ngram_matches_ar_on_flash_next_p1() {
    run_case(Case::P1, &NGRAM_ARMS, "ngram-p1");
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (Flash-Next base with matching PLE rows)"]
fn greedy_mtp_ngram_matches_ar_on_flash_next_seasons() {
    run_case(Case::Seasons, &NGRAM_ARMS, "ngram-seasons");
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (Flash-Next base with matching PLE rows)"]
fn greedy_mtp_ngram_matches_ar_on_flash_next_copy() {
    run_case(Case::Copy, &NGRAM_ARMS, "ngram-copy");
}

/// Same-shape determinism: `ngram-batched` on `copy` in two fresh children.
#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (Flash-Next base with matching PLE rows)"]
fn greedy_mtp_ngram_batched_is_deterministic_on_flash_next_copy() {
    let model = model_path();
    let base = out_dir().join("ngram-copy-determinism");
    let arm = NGRAM_ARMS
        .iter()
        .copied()
        .find(|arm| arm.name == "ngram-batched")
        .expect("ngram-batched arm");
    let mut runs = Vec::new();
    for run in 1..=2 {
        let dir = base.join(format!("run{run}"));
        fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
        let result = spawn_arm(arm, Case::Copy, &model, &dir);
        let w = ngram_window_check(arm, Case::Copy, &result.stderr);
        eprintln!(
            "ngram-batched run {run}: {} ngram and {} mtp windows",
            w.ngram, w.mtp
        );
        runs.push(result.ids);
    }
    if let Some(index) = first_difference(&runs[0], &runs[1]) {
        panic!(
            "ngram-batched copy runs differ at generated index {index} (logs in {})",
            base.display()
        );
    }
}

/// This is an identity/coherence gate, not a speed or hipGraph gate.
#[test]
#[ignore = "requires supported real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL; fails if ineligible or no changed derived-K decision"]
fn greedy_mtp_derived_k_vs_incumbent_on_flash_next_prompts() {
    run_derived_vs_incumbent(&DERIVED_ARMS, "derived-vs-incumbent");
}

/// The COPYON pair of the gate above: the n-gram copy lane is on in both
/// arms. It additionally requires, from the existing traces alone, that the
/// derived arm priced and took copy takeovers (`derived_k` lines of source
/// `ngram` in mode `on_copy` that drafted, `window` lines of source `ngram`)
/// on the copy-heavy prompt, so an enabled flag with no copy takeover, or one
/// that fell back to the incumbent because no OnCopy price is measured, fails.
#[test]
#[ignore = "requires supported real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL; fails if ineligible, no OnCopy price or no changed derived-K copy decision"]
fn greedy_mtp_derived_k_copy_vs_incumbent_on_flash_next_prompts() {
    run_derived_vs_incumbent(&DERIVED_COPY_ARMS, "derived-copy-vs-incumbent");
}

/// The shared body of the two derived gates: `arms` is the incumbent/derived
/// pair of one cost mode.
fn run_derived_vs_incumbent(arms: &[Arm; 2], dir_name: &str) {
    let copy = arms[0].copy_lane;
    assert_eq!(arms[1].copy_lane, copy, "the pair shares one cost mode");
    let mode = if copy { "on_copy" } else { "off_pool" };
    let model = model_path();
    let base = out_dir().join(dir_name);
    let mut changed = 0;
    let mut takeovers = 0;
    for case in [Case::P1, Case::Seasons, Case::Copy] {
        let dir = base.join(case.name());
        fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
        // Reserve the same load-default native-head capacity for AR too:
        // a smaller AR reserve can change the automatic expert placement.
        let ar = spawn_arm(
            Arm {
                derived_k: Some(false),
                ..ARMS[0]
            },
            case,
            &model,
            &dir,
        );
        let incumbent = spawn_arm(arms[0], case, &model, &dir);
        let derived = spawn_arm(arms[1], case, &model, &dir);
        let events = mtp_trace_events(&derived.stderr);
        // Each arm's own request receipt names the copy lane it ran.
        for (label, arm_events) in [
            ("incumbent", mtp_trace_events(&incumbent.stderr)),
            ("derived", events.clone()),
        ] {
            let requests: Vec<_> = arm_events
                .iter()
                .filter(|event| event["event"] == "request")
                .collect();
            assert_eq!(
                requests.len(),
                1,
                "{}: {label} arm must print exactly one request receipt",
                case.name()
            );
            assert_eq!(
                requests[0]["copy_lane"],
                serde_json::json!(copy),
                "{}: {label} arm request receipt names the wrong copy lane: {}",
                case.name(),
                requests[0]
            );
        }
        let requests: Vec<_> = events
            .iter()
            .filter(|event| event["event"] == "derived_k_request")
            .collect();
        assert_eq!(
            requests.len(),
            1,
            "{}: missing/duplicate derived request receipt",
            case.name()
        );
        assert_eq!(
            requests[0]["eligible"],
            serde_json::json!(true),
            "{}: unsupported derived-K request: {}",
            case.name(),
            requests[0]
        );
        let decisions: Vec<_> = events
            .iter()
            .filter(|event| event["event"] == "derived_k")
            .collect();
        for event in &decisions {
            assert_eq!(event["version"], "fn-phase1-v1/replay2-v1");
            assert_eq!(event["verify_route"], "batched");
            assert!(
                event["changed"].is_boolean(),
                "invalid derived receipt: {event}"
            );
            // Prices are keyed by the arm's own cost mode, never the other's.
            assert_eq!(
                event["mode"],
                serde_json::json!(mode),
                "{}: derived price mode: {event}",
                case.name()
            );
            // Only the physical and explicit ceilings, never above the
            // qualified row-exact rows the bundle reported.
            let legal = event["legal_cap"].as_u64().expect("legal_cap");
            let rows = event["row_limit"].as_u64().expect("row_limit");
            let drafted = event["drafted_k"].as_u64().expect("drafted_k");
            assert!(
                legal < rows && drafted <= legal,
                "{}: width beyond the qualified rows or the legal cap: {event}",
                case.name()
            );
        }
        for event in events
            .iter()
            .filter(|event| event["event"] == "derived_k_unsupported")
        {
            assert_eq!(
                event["mode"],
                serde_json::json!(mode),
                "{}: unsupported receipt names the wrong mode: {event}",
                case.name()
            );
        }
        let ngram_decisions = decisions
            .iter()
            .filter(|event| {
                event["source"] == "ngram" && event["drafted_k"].as_u64().unwrap_or(0) > 0
            })
            .count();
        let ngram_windows = events
            .iter()
            .filter(|event| event["event"] == "window" && event["source"] == "ngram")
            .count();
        if copy {
            takeovers += ngram_decisions;
            if matches!(case, Case::Copy) {
                assert!(
                    ngram_windows > 0 && ngram_decisions > 0,
                    "{}: copy arm took no priced copy takeover ({ngram_windows} ngram windows, {ngram_decisions} derived n-gram decisions, {} unsupported receipts): no measured OnCopy price, or no copy takeover (logs in {})",
                    case.name(),
                    events
                        .iter()
                        .filter(|event| event["event"] == "derived_k_unsupported")
                        .count(),
                    dir.display()
                );
            }
        }
        let prompt_changed = decisions
            .iter()
            .filter(|event| event["changed"] == true)
            .count();
        changed += prompt_changed;
        let same_routes = identity_window_sequence(&dir, arms[0], &incumbent.stderr)
            == identity_window_sequence(&dir, arms[1], &derived.stderr);
        let difference = first_difference(&incumbent.ids, &derived.ids);
        let text = fs::read_to_string(dir.join("derived.txt")).expect("derived decoded text");
        let attractor = decoded_text_has_attractor(&text);
        let summary = format!(
            "{}{}: same_routes={same_routes}, {}id_equal={}, first_difference={difference:?}, consulted={}, changed={prompt_changed}, attractor={attractor}, reached_token_cap={} (cap alone is not runaway; HIPFIRE_GRAPH not exercised)\n",
            derived.header,
            case.name(),
            if same_routes { "same_route_" } else { "cross_route_" },
            difference.is_none(),
            decisions.len(),
            derived.ids.len() == case.generated(),
        );
        eprint!("{summary}");
        fs::write(dir.join("summary.txt"), summary).unwrap();
        for (label, ids) in [("incumbent", &incumbent.ids), ("derived", &derived.ids)] {
            assert_eq!(
                first_difference(&ar.ids[..ids.len()], ids),
                None,
                "{}: {label} differs from AR (logs in {})",
                case.name(),
                dir.display()
            );
        }
        assert!(
            !attractor,
            "{}: derived decoded output is empty or an attractor/runaway loop (logs in {})",
            case.name(),
            dir.display()
        );
    }
    assert!(
        changed > 0,
        "no exercise: no derived_k window changed=true across the prompts (logs in {})",
        base.display()
    );
    assert!(
        !copy || takeovers > 0,
        "no exercise: no priced copy takeover across the prompts (logs in {})",
        base.display()
    );
}

fn mtp_trace_events(stderr: &str) -> Vec<serde_json::Value> {
    stderr
        .lines()
        .filter_map(|line| line.strip_prefix("QWEN4_MTP_TRACE "))
        .map(|json| serde_json::from_str(json).expect("valid QWEN4_MTP_TRACE JSON"))
        .collect()
}

/// PLAN G3: run separately with HIPFIRE_REPLAY_BACKEND=hip and auto.
/// Refused/missing F32/Q8 cells fail; this is not a capability promotion.
/// The five existing selectors (rows, scenario, case, state, kv) choose one
/// declared cell; when all are absent the parent runs the whole matrix.
/// `coverage.json` records only reached cells actually executed, so a filtered
/// run is a reached subset, never the whole G3.
#[test]
#[ignore = "requires real GPU/model/sidecars and row64 kernels; run --test-threads=1"]
fn greedy_mtp_wide_windows_match_ar_on_flash_next() {
    let selected = wide_selector().unwrap_or_else(|e| panic!("{e}"));
    let model = model_path();
    let base = out_dir().join("wide-g3");
    let mut cells = Vec::new();
    for state in ["q8", "fp32"] {
        for kv in ["q8", "f32"] {
            for rows in WIDE_ROWS {
                let run = WideRun {
                    rows,
                    scenario: "ar",
                    state,
                    kv,
                };
                for case in [Case::P1, Case::Copy] {
                    if let Some((selected_run, selected_case)) = selected {
                        if state != selected_run.state
                            || kv != selected_run.kv
                            || rows != selected_run.rows
                            || case != selected_case
                        {
                            continue;
                        }
                    }
                    let dir = base.join(format!("{state}-{kv}-r{rows}-{}", case.name()));
                    fs::create_dir_all(&dir).expect("create wide cell directory");
                    let reference = spawn_arm(wide_arm(run), case, &model, &dir);
                    assert_eq!(
                        reference.ids.len(),
                        case.generated(),
                        "complete AR reference"
                    );
                    // The product chooser has no forced width: one ceiling64
                    // campaign per precision cell, not eight identical runs.
                    let scenarios: &[&str] = wide_scenarios(case, rows == "64");
                    for &scenario in scenarios {
                        if let Some((selected_run, _)) = selected {
                            if scenario != selected_run.scenario {
                                continue;
                            }
                        }
                        let cell = WideRun { scenario, ..run };
                        let candidate = spawn_arm(wide_arm(cell), case, &model, &dir);
                        let end = if scenario == "eos" {
                            wide_stop(&reference.ids).1 + 1
                        } else {
                            wide_budget(cell, reference.ids.len())
                        };
                        assert_eq!(
                            first_difference(&reference.ids[..end], &candidate.ids), None,
                            "{state}/{kv}/R{rows}/{scenario}/{}: every ID must equal AR (logs in {})",
                            case.name(), dir.display()
                        );
                        let reached = check_wide_receipts(&candidate.stderr, cell);
                        cells.push(serde_json::json!({
                            "state": state, "kv": kv, "requested_rows": rows,
                            "scenario": scenario, "case": case.name(),
                            "ids": candidate.ids.len(), "reached": reached,
                        }));
                        fs::write(
                            base.join("coverage.json"),
                            serde_json::to_vec_pretty(&cells).expect("coverage JSON"),
                        )
                        .expect("write reached-window coverage");
                    }
                }
            }
        }
    }
}

fn wide_budget(run: WideRun, normal: usize) -> usize {
    if run.scenario == "budget" {
        2 * run.rows.parse::<usize>().expect("wide rows") + 5
    } else {
        normal
    }
}

/// Deterministic laboratory EOS from AR, never a fabricated draft. Pass it
/// through the real acceptance/terminal-repair path as the stop token.
fn wide_stop(reference: &[u32]) -> (u32, usize) {
    reference
        .iter()
        .copied()
        .enumerate()
        .find(|(index, token)| {
            *index >= 64 && reference[..*index].iter().all(|prior| prior != token)
        })
        .map(|(index, token)| (token, index))
        .expect("AR fixture needs a first-occurrence stop token beyond row64")
}

/// Missing actual extent or successful repair receipts are a hard failure.
fn check_wide_receipts(stderr: &str, run: WideRun) -> serde_json::Value {
    let events = mtp_trace_events(stderr);
    let width = run.rows.parse::<u64>().expect("requested rows");
    let windows: Vec<_> = events
        .iter()
        .filter(|event| event["event"] == "window")
        .collect();
    assert!(
        !windows.is_empty(),
        "{}: no actual verify windows",
        run.scenario
    );
    let mut widths = std::collections::BTreeSet::new();
    let mut sources = Vec::new();
    let mut partial = false;
    let mut retired_copy = false;
    for (window_index, window) in windows.iter().enumerate() {
        let reached = window["verify_rows"]
            .as_u64()
            .expect("actual verify_rows receipt");
        let limit = window["row_limit"]
            .as_u64()
            .expect("effective row_limit receipt");
        assert!(reached > 0 && reached <= limit && limit <= 64, "{window}");
        assert!(
            window["verify_route"] == "batched" || window["verify_route"] == "interleaved",
            "unknown actual verify route: {window}"
        );
        if window["verify_route"] == "batched" {
            assert_eq!(
                window["rows"]
                    .as_array()
                    .expect("verified draft rows")
                    .len() as u64
                    + 1,
                reached,
                "actual target extent, not configured K: {window}"
            );
        } else {
            assert_eq!(reached, 1, "interleaved is not wide coverage");
        }
        widths.insert(reached);
        let source = window["source"].as_str().expect("actual source receipt");
        assert!(
            matches!(source, "mtp" | "ngram"),
            "unknown actual source: {window}"
        );
        sources.push(source);
        let commits: Vec<_> = events
            .iter()
            .filter(|event| {
                event["event"] == "window_commit"
                    && event["position"] == window["position"]
                    && event["source"] == window["source"]
            })
            .collect();
        assert_eq!(
            commits.len(),
            1,
            "missing successful repair/commit receipt: {window}"
        );
        let commit = commits[0];
        assert_eq!(commit["verified_rows"].as_u64(), Some(reached));
        let keep = commit["keep"].as_u64().expect("committed state prefix");
        assert!(keep > 0, "{commit}");
        if window["verify_route"] == "batched" {
            assert!(keep <= reached, "{commit}");
        }
        if keep < reached && window["verify_route"] == "batched" {
            assert_eq!(
                commit["repair_route"], "native_capture",
                "must exercise real native prefix repair"
            );
            let wanted_source = if run.scenario == "copy" {
                "ngram"
            } else {
                "mtp"
            };
            partial |= source == wanted_source
                && reached == width
                && keep > 1
                && window_index + 1 < windows.len();
        }
        retired_copy |= source == "ngram" && commit["retired"] == true && reached == width;
        if run.scenario == "retire" && commit["retired"] == true {
            assert_ne!(
                source, "mtp",
                "native drafting resurrected after retirement"
            );
        }
    }
    if matches!(run.scenario, "product" | "product-copy") {
        let decisions: Vec<_> = events
            .iter()
            .filter(|event| event["event"] == "derived_k")
            .collect();
        assert!(!decisions.is_empty(), "product chooser was not consulted");
        assert!(
            !events
                .iter()
                .any(|event| event["event"] == "derived_k_unsupported"),
            "unpriced/unsupported product mode is not coverage"
        );
        for event in decisions {
            let legal = event["legal_cap"].as_u64().expect("derived legal cap");
            let rows = event["row_limit"].as_u64().expect("derived effective rows");
            let drafted = event["drafted_k"].as_u64().expect("derived actual drafts");
            assert!(drafted <= legal && legal < rows && rows <= 64, "{event}");
        }
        assert!(
            widths.iter().any(|&rows| rows > 8),
            "product-derived chooser never actually verified a wide window"
        );
        if run.scenario == "product-copy" {
            assert!(
                windows.iter().any(|event| event["source"] == "ngram"
                    && event["verify_rows"].as_u64().unwrap_or(0) > 8),
                "product copy chooser never reached wide verify"
            );
        } else {
            assert!(
                windows.iter().any(|event| event["source"] == "mtp"
                    && event["verify_rows"].as_u64().unwrap_or(0) > 8),
                "product native chooser never reached wide verify"
            );
        }
    } else {
        let source = if matches!(run.scenario, "copy" | "return" | "retire") {
            "ngram"
        } else {
            "mtp"
        };
        assert!(
            windows
                .iter()
                .any(|event| event["source"] == source
                    && event["verify_rows"].as_u64() == Some(width)),
            "{}: R{width} never executed on {source}; configured K does not count",
            run.scenario
        );
        if matches!(run.scenario, "native" | "copy") {
            assert!(
                partial,
                "R{width}: no interior partial rollback followed by continuation"
            );
        }
        if run.scenario == "return" {
            let native = sources
                .iter()
                .position(|source| *source == "mtp")
                .expect("initial native");
            let copy = sources
                .iter()
                .enumerate()
                .skip(native + 1)
                .find(|(_, source)| **source == "ngram")
                .map(|(i, _)| i)
                .expect("copy takeover");
            assert!(
                sources[copy + 1..].contains(&"mtp"),
                "copy never returned to native"
            );
        }
        if run.scenario == "retire" {
            assert!(
                retired_copy,
                "no full-width copy window after actual native retirement"
            );
            assert!(
                sources.contains(&"mtp"),
                "retirement case never drafted natively"
            );
        }
        if run.scenario == "budget" {
            assert!(
                widths.iter().any(|&rows| rows < width),
                "no ragged end-of-budget verify"
            );
        }
    }
    serde_json::json!({"widths": widths, "sources": sources,
        "interior_partial": partial, "retired_copy": retired_copy})
}

#[test]
#[ignore = "private fresh child for greedy_mtp_wide_windows_match_ar_on_flash_next"]
fn greedy_mtp_wide_identity_arm() {
    let rows = std::env::var(WIDE_ROWS_ENV).expect("explicit child width");
    let rows = WIDE_ROWS
        .into_iter()
        .find(|candidate| *candidate == rows)
        .expect("supported G3 width");
    let scenario = std::env::var(WIDE_SCENARIO_ENV).expect("explicit child scenario");
    let scenario = [
        "ar",
        "native",
        "copy",
        "return",
        "retire",
        "budget",
        "eos",
        "product",
        "product-copy",
    ]
    .into_iter()
    .find(|candidate| *candidate == scenario)
    .expect("supported G3 scenario");
    assert_eq!(std::env::var(ARM_ENV).as_deref(), Ok(scenario));
    let state = hw_identity::gdn_test_state_quant()
        .expect("explicit admitted GDN state")
        .0;
    assert!(
        matches!(state, "q8" | "fp32"),
        "Q8Ef is not admitted for greedy identity"
    );
    let kv = match std::env::var("HIPFIRE_KV_MODE").as_deref() {
        Ok("q8") => "q8",
        Ok("f32") => "f32",
        _ => panic!("explicit admitted KV mode required"),
    };
    let case = match std::env::var(CASE_ENV).as_deref() {
        Ok("p1") => Case::P1,
        Ok("copy") => Case::Copy,
        _ => panic!("explicit wide prompt required"),
    };
    run_arm(
        wide_arm(WideRun {
            rows,
            scenario,
            state,
            kv,
        }),
        case,
        &model_path(),
        &out_dir(),
    )
    .unwrap_or_else(|error| panic!("{scenario}/R{rows}: {error}"));
}

#[test]
fn wide_child_overrides_carry_width_source_and_precision() {
    assert_eq!(Case::P1.generated(), 1200);
    assert_eq!(Case::Copy.generated(), 1200);
    assert_eq!(Case::Seasons.generated(), 256);
    assert_eq!(
        parse_wide_selector([None, None, None, None, None]).expect("all absent"),
        None
    );
    for state in ["q8", "fp32"] {
        for kv in ["q8", "f32"] {
            for rows in WIDE_ROWS {
                for case in [Case::P1, Case::Copy] {
                    let scenarios: &[&str] = wide_scenarios(case, rows == "64");
                    for &scenario in scenarios {
                        let (run, resolved) = parse_wide_selector([
                            Some(rows),
                            Some(scenario),
                            Some(case.name()),
                            Some(state),
                            Some(kv),
                        ])
                        .expect("declared selector")
                        .expect("complete selector resolves");
                        assert_eq!(run.rows, rows);
                        assert_eq!(run.scenario, scenario);
                        assert_eq!(run.state, state);
                        assert_eq!(run.kv, kv);
                        assert_eq!(resolved, case);
                    }
                    for (index, name) in [
                        WIDE_ROWS_ENV,
                        WIDE_SCENARIO_ENV,
                        CASE_ENV,
                        hw_identity::GDN_TEST_STATE_QUANT_ENV,
                        "HIPFIRE_KV_MODE",
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let mut partial = [
                            Some("64"),
                            Some("native"),
                            Some("p1"),
                            Some("q8"),
                            Some("q8"),
                        ];
                        partial[index] = None;
                        assert!(
                            parse_wide_selector(partial).is_err(),
                            "partial selector missing {name} must fail"
                        );
                    }
                }
            }
        }
    }
    for bad in ["8", "65"] {
        assert!(
            parse_wide_selector([
                Some(bad),
                Some("native"),
                Some("p1"),
                Some("q8"),
                Some("q8")
            ])
            .is_err(),
            "invalid rows {bad} must fail"
        );
    }
    for bad in ["ar", "bogus"] {
        assert!(
            parse_wide_selector([Some("64"), Some(bad), Some("p1"), Some("q8"), Some("q8")])
                .is_err(),
            "invalid scenario {bad} must fail"
        );
    }
    assert!(
        parse_wide_selector([Some("64"), Some("copy"), Some("p1"), Some("q8"), Some("q8")])
            .is_err()
    );
    assert!(parse_wide_selector([
        Some("64"),
        Some("native"),
        Some("copy"),
        Some("q8"),
        Some("q8")
    ])
    .is_err());
    assert!(parse_wide_selector([
        Some("64"),
        Some("product"),
        Some("copy"),
        Some("q8"),
        Some("q8")
    ])
    .is_err());
    assert!(parse_wide_selector([
        Some("9"),
        Some("product"),
        Some("p1"),
        Some("q8"),
        Some("q8")
    ])
    .is_err());
    assert!(parse_wide_selector([
        Some("9"),
        Some("product-copy"),
        Some("copy"),
        Some("q8"),
        Some("q8")
    ])
    .is_err());
    assert!(parse_wide_selector([
        Some("64"),
        Some("native"),
        Some("seasons"),
        Some("q8"),
        Some("q8")
    ])
    .is_err());
    assert!(parse_wide_selector([
        Some("64"),
        Some("native"),
        Some("p1"),
        Some("q8ef"),
        Some("q8")
    ])
    .is_err());
    assert!(parse_wide_selector([
        Some("64"),
        Some("native"),
        Some("p1"),
        Some("q8"),
        Some("f16")
    ])
    .is_err());
    assert!(
        parse_wide_selector([Some("64"), Some("ar"), Some("p1"), Some("q8"), Some("q8")]).is_err()
    );
    for rows in WIDE_ROWS {
        for scenario in [
            "native",
            "copy",
            "return",
            "retire",
            "budget",
            "eos",
            "product",
            "product-copy",
        ] {
            let arm = wide_arm(WideRun {
                rows,
                scenario,
                state: "fp32",
                kv: "q8",
            });
            let overrides = arm_overrides(arm).expect("controlled environment");
            let mut names = std::collections::BTreeSet::new();
            for (name, _) in &overrides {
                assert!(
                    names.insert(*name),
                    "duplicate override {name} breaks child validation"
                );
            }
            let value = |name| {
                overrides
                    .iter()
                    .find(|(key, _)| *key == name)
                    .and_then(|(_, value)| *value)
            };
            assert_eq!(value(WIDE_ROWS_ENV), Some(rows));
            assert_eq!(value(WIDE_SCENARIO_ENV), Some(scenario));
            assert_eq!(value("HIPFIRE_QWEN4_WIDE_VERIFY_ROWS"), Some("64"));
            assert_eq!(value(hw_identity::GDN_TEST_STATE_QUANT_ENV), Some("fp32"));
            assert_eq!(value("HIPFIRE_KV_MODE"), Some("q8"));
            assert_eq!(
                value("HIPFIRE_MTP_NGRAM_COPY"),
                Some(if arm.copy_lane { "1" } else { "0" })
            );
            assert_eq!(value("HIPFIRE_MTP_INCREMENTAL"), None);
        }
    }
}

#[test]
#[should_panic(expected = "actual verify_rows receipt")]
fn wide_receipts_refuse_configured_k_without_actual_extent() {
    let run = WideRun {
        rows: "64",
        scenario: "native",
        state: "q8",
        kv: "q8",
    };
    check_wide_receipts(
        "QWEN4_MTP_TRACE {\"event\":\"window\",\"source\":\"mtp\",\"k\":63,\"verify_route\":\"batched\"}",
        run,
    );
}

#[test]
#[should_panic(expected = "missing successful repair/commit receipt")]
fn wide_receipts_refuse_uncommitted_window() {
    let event = serde_json::json!({
        "event": "window", "position": 100, "source": "mtp",
        "k": 63, "verify_route": "batched", "verify_rows": 64,
        "row_limit": 64, "rows": vec![serde_json::json!({"match": true}); 63],
    });
    let trace = format!("QWEN4_MTP_TRACE {event}");
    check_wide_receipts(
        &trace,
        WideRun {
            rows: "64",
            scenario: "native",
            state: "q8",
            kv: "q8",
        },
    );
}

/// Include AR floor/probe/derived-AR windows, which have no `window` trace.
/// The returned window receipt supplies actual drafts, not the trace's K budget.
fn identity_window_sequence(dir: &Path, arm: Arm, stderr: &str) -> Vec<(String, u64)> {
    let mut routes = BTreeMap::new();
    for event in mtp_trace_events(stderr) {
        if event["event"] == "window" {
            let position = event["position"].as_u64().expect("window trace position");
            let route = event["verify_route"].as_str().expect("window verify_route");
            assert!(
                route == "batched" || route == "interleaved",
                "unknown verify route: {event}"
            );
            assert!(
                routes.insert(position, route.to_string()).is_none(),
                "duplicate window trace at {position}"
            );
        }
    }
    let receipt: serde_json::Value = serde_json::from_slice(
        &fs::read(dir.join(format!("{}.windows.json", arm.name))).expect("window receipt"),
    )
    .expect("window receipt JSON");
    let columns = receipt["columns"].as_array().expect("window columns");
    let position_col = columns
        .iter()
        .position(|name| name == "position_before")
        .expect("position column");
    let drafts_col = columns
        .iter()
        .position(|name| name == "drafts")
        .expect("drafts column");
    let sequence = receipt["windows"]
        .as_array()
        .expect("window rows")
        .iter()
        .map(|row| {
            let position = row[position_col].as_u64().expect("window position");
            let drafts = row[drafts_col].as_u64().expect("window drafts");
            let route = routes.remove(&position).unwrap_or_else(|| {
                assert_eq!(drafts, 0, "untraced non-AR window at {position}");
                "ar".to_string()
            });
            (route, drafts)
        })
        .collect::<Vec<_>>();
    assert!(
        !sequence.is_empty(),
        "{}: no window route exercise",
        arm.name
    );
    assert!(routes.is_empty(), "{}: unaccounted window traces", arm.name);
    sequence
}

/// Same whitespace-token thresholds as scripts/serve_harness.py's
/// `_token_attractor` / `_response_has_attractor`, including separate channels.
/// Fixed-length generation alone is not a runaway failure.
fn decoded_text_has_attractor(text: &str) -> bool {
    fn channel_has_attractor(text: &str) -> bool {
        let words: Vec<_> = text.split_whitespace().collect();
        let ratio_bad = |window: &[&str], minimum: f64| {
            let mut counts = BTreeMap::new();
            for word in window {
                *counts.entry(*word).or_insert(0usize) += 1;
            }
            counts.len() as f64 / (window.len() as f64) < minimum
                || counts.values().copied().max().unwrap_or(0) as f64 / (window.len() as f64) > 0.50
        };
        if words.len() >= 8
            && (ratio_bad(&words[..words.len().min(128)], 0.15)
                || ratio_bad(&words[words.len().saturating_sub(128)..], 0.30))
        {
            return true;
        }
        let half = &words[words.len() / 2..];
        if half.len() < 6 {
            return false;
        }
        let mut grams = BTreeMap::new();
        for gram in half.windows(3) {
            *grams.entry(gram).or_insert(0usize) += 1;
        }
        let repeated: usize = grams.values().copied().filter(|&count| count > 1).sum();
        repeated as f64 / ((half.len() - 2) as f64) > 0.50
    }
    if text.trim().is_empty() {
        return true;
    }
    match text.split_once("</think>") {
        Some((thinking, visible)) => {
            channel_has_attractor(thinking.trim_start_matches("<think>"))
                || channel_has_attractor(visible)
        }
        None => channel_has_attractor(text),
    }
}

/// Explicit invocation without its identity environment must fail.
#[test]
#[ignore = "child process of greedy_mtp_derived_k_vs_incumbent_on_flash_next_prompts; identity env required"]
fn greedy_mtp_derived_k_identity_arm() {
    let name = std::env::var(ARM_ENV).unwrap_or_else(|_| panic!("{ARM_ENV} is required"));
    let arm = if name == "ar" {
        Arm {
            derived_k: Some(false),
            ..ARMS[0]
        }
    } else {
        DERIVED_ARMS
            .iter()
            .copied()
            .find(|arm| arm.name == name)
            .unwrap_or_else(|| panic!("unknown derived {ARM_ENV}={name}"))
    };
    let case = match std::env::var(CASE_ENV).as_deref() {
        Ok("p1") => Case::P1,
        Ok("seasons") => Case::Seasons,
        Ok("copy") => Case::Copy,
        other => panic!("unknown/missing {CASE_ENV}={other:?}"),
    };
    let dir = std::env::var_os(OUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{OUT_ENV} is required"));
    assert!(
        dir.is_dir(),
        "{OUT_ENV} must name the parent's output directory"
    );
    if let Err(error) = run_arm(arm, case, &model_path(), &dir) {
        panic!("{} arm: {error}", arm.name);
    }
}

/// The COPYON pair's child process; explicit invocation without its identity
/// environment must fail.
#[test]
#[ignore = "child process of greedy_mtp_derived_k_copy_vs_incumbent_on_flash_next_prompts; identity env required"]
fn greedy_mtp_derived_k_copy_identity_arm() {
    let name = std::env::var(ARM_ENV).unwrap_or_else(|_| panic!("{ARM_ENV} is required"));
    let arm = DERIVED_COPY_ARMS
        .iter()
        .copied()
        .find(|arm| arm.name == name)
        .unwrap_or_else(|| panic!("unknown derived copy {ARM_ENV}={name}"));
    let case = match std::env::var(CASE_ENV).as_deref() {
        Ok("p1") => Case::P1,
        Ok("seasons") => Case::Seasons,
        Ok("copy") => Case::Copy,
        other => panic!("unknown/missing {CASE_ENV}={other:?}"),
    };
    let dir = std::env::var_os(OUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{OUT_ENV} is required"));
    assert!(
        dir.is_dir(),
        "{OUT_ENV} must name the parent's output directory"
    );
    if let Err(error) = run_arm(arm, case, &model_path(), &dir) {
        panic!("{} arm: {error}", arm.name);
    }
}

struct ArmRun {
    ids: Vec<u32>,
    stderr: String,
    /// `summary.txt` header lines: the arm's recorded GDN format and graph policy.
    header: String,
}

/// What a parent changes in an arm's child environment: everything that can
/// force a route or alter the drafter is cleared, then the arm's own settings
/// are applied. The AR floor is left at its default (on) for every arm. The
/// GDN test mode (`HIPFIRE_GDN_TEST_STATE_QUANT`, default `q8`) and the launch
/// backend (`HIPFIRE_REPLAY_BACKEND`, default `hip`) are resolved once by the
/// parent and forwarded explicitly, so a child never runs a different mode.
fn arm_overrides(arm: Arm) -> Result<Vec<(&'static str, Option<&'static str>)>, String> {
    let (n_match, n_min, n_max) = if arm.wide.is_some() {
        (Some("5"), Some("1"), Some("63"))
    } else {
        match arm.ngram_triple {
            Some((n_match, n_min, n_max)) => (Some(n_match), Some(n_min), Some(n_max)),
            None => (None, None, None),
        }
    };
    let state_request = match arm.wide {
        Some(run) => run.state,
        None => hw_identity::gdn_test_state_quant()?.0,
    };
    let mut overrides = vec![
        ("HIPFIRE_MTP_AR_FLOOR", None),
        ("HIPFIRE_MTP_PHASE_TIMING", None),
        ("HIPFIRE_MTP_NGRAM", None),
        ("HIPFIRE_MTP_FLOOR_MIN_WINDOWS", None),
        ("HIPFIRE_MTP_FLOOR_AR_PROBE", None),
        ("HIPFIRE_MTP_FLOOR_TRACE", None),
        (
            "HIPFIRE_MTP_NGRAM_COPY",
            Some(if arm.copy_lane { "1" } else { "0" }),
        ),
        (WIDE_ROWS_ENV, arm.wide.map(|run| run.rows)),
        (WIDE_SCENARIO_ENV, arm.wide.map(|run| run.scenario)),
        (DERIVED_K_ENV, (arm.derived_k == Some(true)).then_some("1")),
        ("HIPFIRE_MTP_TRACE", (arm.route != Route::Ar).then_some("1")),
        ("HIPFIRE_MTP_INCREMENTAL", arm.incremental),
        ("HIPFIRE_MTP_DRAFT_HEAD", arm.draft_head),
        ("HIPFIRE_NGRAM_MOD_N_MATCH", n_match),
        ("HIPFIRE_NGRAM_MOD_N_MIN", n_min),
        ("HIPFIRE_NGRAM_MOD_N_MAX", n_max),
        (hw_identity::GDN_TEST_STATE_QUANT_ENV, Some(state_request)),
        (REPLAY_BACKEND_ENV, Some(replay_backend_request()?)),
    ];
    if let Some(run) = arm.wide {
        overrides.extend([
            ("HIPFIRE_QWEN4_WIDE_VERIFY_ROWS", Some("64")),
            ("HIPFIRE_KV_MODE", Some(run.kv)),
        ]);
    }
    Ok(overrides)
}

/// The native MTP draft window of an arm. A derived-policy arm runs the load's
/// unqualified default (`hipfire_config::mtp_k_default` over the GPU's
/// qualified row-exact rows, no explicit `mtp_k`), so the incumbent and the
/// derived chooser are not both pinned to a shallow test K; every other arm
/// keeps this test's [`MTP_K`].
fn arm_mtp_k(arm: Arm, gpu_arch: &str) -> usize {
    if arm.wide.is_some() {
        63
    } else if arm.derived_k.is_some() {
        hipfire_config::mtp_k_default(
            hipfire_arch_qwen4::ARCH_ID,
            rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows(gpu_arch),
        )
    } else {
        MTP_K
    }
}

/// The child must already run under its arm's environment.
fn require_arm_env(arm: Arm) -> Result<(), String> {
    for (name, want) in arm_overrides(arm)? {
        let have = std::env::var(name).ok();
        if have.as_deref() != want {
            return Err(format!(
                "{name} is {have:?} but arm {} requires {want:?} (the parent forwards it explicitly)",
                arm.name
            ));
        }
    }
    Ok(())
}

/// Run one arm in a fresh child of this test binary and collect its IDs and
/// trace.
fn spawn_arm(arm: Arm, case: Case, model: &Path, dir: &Path) -> ArmRun {
    let exe = std::env::current_exe().expect("test binary path");
    let model_id = hw_identity::model_identity(model, dir).unwrap_or_else(|e| panic!("{e}"));
    let expected = hw_identity::config_identity(
        &hw_identity::env_snapshot(
            &arm_overrides(arm).unwrap_or_else(|e| panic!("{e}")),
            &CONTROL_ENV,
        ),
        &model_id,
    );
    // An earlier result of this arm must never stand in for this run.
    for stale in ["ids.json", "txt", "identity.json", "windows.json"] {
        let _ = fs::remove_file(dir.join(format!("{}.{stale}", arm.name)));
    }
    let mut child = Command::new(&exe);
    child
        .args([
            "--exact",
            if arm.wide.is_some() {
                WIDE_CHILD
            } else {
                match (arm.derived_k.is_some(), arm.copy_lane) {
                    (true, true) => DERIVED_COPY_ARM_TEST,
                    (true, false) => DERIVED_ARM_TEST,
                    (false, _) => ARM_TEST,
                }
            },
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ARM_ENV, arm.name)
        .env(CASE_ENV, case.name())
        .env(MODEL_ENV, model)
        .env(OUT_ENV, dir)
        .stdin(Stdio::null());
    for (name, value) in arm_overrides(arm).unwrap_or_else(|e| panic!("{e}")) {
        match value {
            Some(value) => {
                child.env(name, value);
            }
            None => {
                child.env_remove(name);
            }
        }
    }
    let output = child
        .output()
        .unwrap_or_else(|e| panic!("spawn {} arm: {e}", arm.name));
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    fs::write(dir.join(format!("{}.out", arm.name)), &output.stdout).unwrap();
    fs::write(dir.join(format!("{}.err", arm.name)), &output.stderr).unwrap();
    assert!(
        output.status.success(),
        "{} arm failed ({}); logs in {}",
        arm.name,
        output.status,
        dir.display()
    );
    let ids: Vec<u32> = serde_json::from_slice(
        &fs::read(dir.join(format!("{}.ids.json", arm.name))).expect("arm ids"),
    )
    .expect("arm ids json");
    if arm.derived_k.is_some() || arm.wide.is_some() {
        assert!(
            !ids.is_empty() && ids.len() <= case.generated(),
            "{} arm empty or over-budget id count",
            arm.name
        );
    } else {
        assert_eq!(ids.len(), case.generated(), "{} arm id count", arm.name);
    }
    let recorded: serde_json::Value = serde_json::from_slice(
        &fs::read(dir.join(format!("{}.identity.json", arm.name))).expect("arm identity"),
    )
    .expect("arm identity json");
    if arm.derived_k.is_some() && arm.wide.is_none() && ids.len() < case.generated() {
        assert_eq!(
            ids.last().copied().map(u64::from),
            recorded["eos_token"].as_u64(),
            "{}: short output without actual EOS is not identity coverage",
            arm.name
        );
    }
    let (want_request, want_format) = match arm.wide {
        Some(run) => (
            run.state,
            match run.state {
                "q8" => hipfire_arch_qwen4::GdnStateFormat::Q8,
                "fp32" => hipfire_arch_qwen4::GdnStateFormat::F32,
                _ => panic!("wide greedy gate admits only q8/fp32 state"),
            },
        ),
        None => hw_identity::gdn_test_state_quant().unwrap_or_else(|e| panic!("{e}")),
    };
    let want_backend = replay_backend_request().unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        recorded["config"],
        expected,
        "{} arm ran under a different identity than the parent expects (logs in {})",
        arm.name,
        dir.display()
    );
    assert_eq!(
        recorded["gdn_state"],
        serde_json::json!({"request": want_request, "resolved": want_format.name()}),
        "{} arm resolved a different GDN state format than {} requests (logs in {})",
        arm.name,
        hw_identity::GDN_TEST_STATE_QUANT_ENV,
        dir.display()
    );
    assert_eq!(
        recorded["graph_policy"]["replay_backend"],
        serde_json::json!(want_backend),
        "{} arm ran a different launch backend than {REPLAY_BACKEND_ENV} requests (logs in {})",
        arm.name,
        dir.display()
    );
    if let Some(run) = arm.wide {
        assert_eq!(recorded["wide"]["requested_rows"], run.rows);
        assert_eq!(recorded["wide"]["scenario"], run.scenario);
        assert_eq!(recorded["wide"]["state"], run.state);
        assert_eq!(recorded["wide"]["kv_requested"], run.kv);
        if run.scenario == "eos" {
            assert_eq!(recorded["termination"]["stopped_at_eos"], true);
            assert_eq!(
                recorded["termination"]["stop_token"].as_u64(),
                ids.last().copied().map(u64::from)
            );
        }
        assert!(
            recorded["wide"]["effective_rows"]
                .as_u64()
                .expect("effective bundle rows")
                >= run.rows.parse::<u64>().expect("requested rows"),
            "child load did not actually admit the requested extent"
        );
    }
    // The artifacts the arm actually used, not merely the ones next to the
    // model: the PLE sidecar marker the runner reads, and the loader's own
    // trunk overlay `applied` line (a skipped or REFUSED overlay is not the
    // campaign trunk).
    let used = &recorded["artifacts_used"];
    let ple = &used["ple_rows"];
    let marker_prefix = "greedy_mtp_identity ple_rows:";
    let markers: Vec<&str> = stderr
        .lines()
        .filter(|line| line.starts_with(marker_prefix))
        .collect();
    if ple["origin"] == "sidecar" {
        let want = format!(
            "{marker_prefix} format={} origin=sidecar path={}",
            ple["format"].as_str().unwrap_or("?"),
            ple["path"].as_str().unwrap_or("?")
        );
        assert_eq!(
            markers,
            vec![want.as_str()],
            "{} arm must print exactly one PLE sidecar marker (logs in {})",
            arm.name,
            dir.display()
        );
    } else {
        assert!(
            markers.is_empty(),
            "{} arm printed a PLE sidecar marker without using a sidecar",
            arm.name
        );
    }
    if let Some(path) = used["trunk_overlay"]["expected_path"].as_str() {
        let applied = format!("QT44 tensors from {path} (");
        assert!(
            stderr.lines().any(|line| {
                line.starts_with("  qwen4 trunk overlay: applied ") && line.contains(&applied)
            }),
            "{} arm: the loader did not apply the trunk overlay {path} (logs in {})",
            arm.name,
            dir.display()
        );
    }
    if let Some(enabled) = arm.derived_k {
        assert_eq!(
            recorded["derived_k"],
            serde_json::json!(if enabled { Some("1") } else { None }),
            "{} arm derived-K flag identity",
            arm.name
        );
    }
    let header = format!(
        "gdn_state_quant={want_request} gdn_format={}\nreplay_backend={want_backend} replay_state={} HIPFIRE_GRAPH={} ({GRAPH_POLICY_NOTE})\n",
        want_format.name(),
        recorded["graph_policy"]["replay_state"],
        recorded["graph_policy"]["HIPFIRE_GRAPH"],
    );
    ArmRun {
        ids,
        stderr,
        header,
    }
}

fn run_case(case: Case, arms: &[Arm], dir_name: &str) {
    let model = model_path();
    let dir = out_dir().join(dir_name);
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    let mut ids: Vec<(Arm, Vec<u32>)> = Vec::new();
    let mut failures = Vec::new();
    let mut summary = String::new();
    let mut header = String::new();
    for &arm in arms {
        let run = spawn_arm(arm, case, &model, &dir);
        let arm_ids = run.ids;
        let (batched, incremental) = window_routes(&run.stderr);
        let route_ok = match arm.route {
            Route::Ar => batched == 0 && incremental == 0,
            Route::Batched => batched > 0 && incremental == 0,
            Route::Incremental => batched == 0 && incremental > 0,
            Route::Adaptive => batched > 0,
        };
        assert!(
            route_ok,
            "{} arm took the wrong route: {batched} batched and {incremental} interleaved windows",
            arm.name
        );
        let extra = if arm.ngram {
            let w = ngram_window_check(arm, case, &run.stderr);
            format!(", {} ngram and {} mtp windows", w.ngram, w.mtp)
        } else {
            String::new()
        };
        let line = match ids.first() {
            None => format!("{}: {} ids\n", arm.name, arm_ids.len()),
            Some((_, ar)) => match first_difference(ar, &arm_ids) {
                None => format!(
                    "{}: identical to ar ({batched} batched, {incremental} interleaved windows{extra})\n",
                    arm.name
                ),
                Some(index) => {
                    let line = format!(
                        "{}: first difference at generated index {index}: ar {} vs mtp {} ({batched} batched, {incremental} interleaved windows{extra})\n",
                        arm.name, ar[index], arm_ids[index]
                    );
                    failures.push(line.clone());
                    line
                }
            },
        };
        eprint!("{line}");
        if header.is_empty() {
            header = run.header.clone();
            eprint!("{header}");
        }
        summary.push_str(&line);
        ids.push((arm, arm_ids));
    }
    fs::write(dir.join("summary.txt"), format!("{header}{summary}")).unwrap();
    assert!(
        failures.is_empty(),
        "greedy MTP differs from AR (logs in {}):\n{}",
        dir.display(),
        failures.concat()
    );
}

/// One arm in a fresh process; spawned by `spawn_arm`.
#[test]
#[ignore = "child process of the greedy_mtp_*_flash_next_* tests"]
fn greedy_mtp_identity_arm() {
    let Ok(name) = std::env::var(ARM_ENV) else {
        return;
    };
    let arm = ARMS
        .iter()
        .chain(NGRAM_ARMS.iter())
        .copied()
        .find(|arm| arm.name == name)
        .unwrap_or_else(|| panic!("unknown {ARM_ENV}={name}"));
    let case = match std::env::var(CASE_ENV).as_deref() {
        Ok("p1") => Case::P1,
        Ok("seasons") => Case::Seasons,
        Ok("copy") => Case::Copy,
        other => panic!("unknown {CASE_ENV}={other:?}"),
    };
    let dir = out_dir();
    if let Err(error) = run_arm(arm, case, &model_path(), &dir) {
        panic!("{} arm: {error}", arm.name);
    }
}

fn run_arm(arm: Arm, case: Case, model: &Path, dir: &Path) -> Result<(), String> {
    require_arm_env(arm)?;
    if arm.wide.is_some() && !cfg!(feature = "reference-parity") {
        return Err("wide G3 requires --features reference-parity for real transaction receipts and laboratory controls".into());
    }
    let model_str = model
        .to_str()
        .ok_or_else(|| format!("{} is not UTF-8", model.display()))?;
    let model_id = hw_identity::model_identity(model, dir)?;
    let mut source = ModelSource::from_path(model_str)?;
    let (tokenizer, template) = {
        let ModelSource::Hfq(hfq) = &source else {
            return Err("model is not an HFQ artifact".into());
        };
        (
            Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| e.to_string())?,
            hfq.chat_template().ok_or("artifact has no chat template")?,
        )
    };
    let frame = |user: &'static str, system: Option<&'static str>| JinjaChatFrame {
        tokenizer: &tokenizer,
        template: &template,
        system,
        user,
        enable_thinking: false,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: None,
    };
    let tokens = match case {
        Case::P1 => {
            let digest = Sha256::digest(PROMPT);
            let digest: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
            if digest != PROMPT_SHA256 {
                return Err(format!(
                    "prompt fixture sha256 {digest}, expected {PROMPT_SHA256}"
                ));
            }
            let prompt: &'static str = std::str::from_utf8(PROMPT).map_err(|e| e.to_string())?;
            let tokens = tokenizer.encode(&frame(prompt, Some("")).render()?);
            if tokens.len() != PROMPT_TOKENS {
                return Err(format!(
                    "rendered prompt is {} tokens, expected {PROMPT_TOKENS}",
                    tokens.len()
                ));
            }
            tokens
        }
        Case::Copy => {
            let digest = Sha256::digest(COPY_PROMPT);
            let digest: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
            if digest != COPY_PROMPT_SHA256 {
                return Err(format!(
                    "copy prompt sha256 {digest}, expected {COPY_PROMPT_SHA256}"
                ));
            }
            let prompt: &'static str =
                std::str::from_utf8(COPY_PROMPT).map_err(|e| e.to_string())?;
            tokenizer.encode(&frame(prompt, Some("")).render()?)
        }
        Case::Seasons => {
            // `hipfire serve` renders an OpenAI messages request through
            // `render_messages` (no system block when none is sent).
            let user = Message {
                role: Role::User,
                content: SEASONS.to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            };
            let rendered = frame(SEASONS, None).render_messages(&[user], None, None)?;
            let tokens = tokenizer.encode(&rendered);
            if tokens != SEASONS_IDS {
                return Err(format!(
                    "rendered seasons prompt {tokens:?} is not the serve prompt {SEASONS_IDS:?}"
                ));
            }
            tokens
        }
    };
    hipfire_loader::prepare_host_memory_for(model);
    let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
    let mtp_k = arm_mtp_k(arm, &gpu.arch);
    let replay_backend = replay_backend_name(gpu.replay.request());
    let requested_backend = replay_backend_request()?;
    if replay_backend != requested_backend {
        return Err(format!(
            "{REPLAY_BACKEND_ENV} resolved to the {replay_backend} backend, not the {requested_backend} the parent forwarded"
        ));
    }
    // Source admission as the daemon runs it before the carrier loads: resolve and
    // attach the PLE rows sidecar (a split base has no embedded rows). The carrier
    // re-resolves the same source idempotently.
    let ple_choice =
        hipfire_loader::admission::resolve_qwen4_ple_rows(&mut source, model_str, &gpu.arch)?;
    let ple_receipt = ple_rows_receipt(ple_choice.as_ref(), model, &model_id)?;
    if let Some(choice) = &ple_choice {
        if let PleRowsOrigin::Sidecar(path) = &choice.origin {
            eprintln!(
                "greedy_mtp_identity ple_rows: format={} origin=sidecar path={}",
                choice.format.name(),
                path.display()
            );
        }
    }
    // What the product load does with the trunk overlay and the expert hot set
    // next to this model on this device. The carrier below decides and applies
    // both itself; these are the expectations its outcome is checked against, so
    // a skipped or refused sidecar never passes as the campaign placement.
    let (overlay_expected, hotset_expected) = {
        let ModelSource::Hfq(hfq) = &source else {
            return Err("model source is not an HFQ artifact".into());
        };
        let admitted = hipfire_arch_qwen4::admit_hfqm_artifact_on(hfq, Some(&gpu.arch))
            .map_err(|e| e.to_string())?;
        let overlay =
            hipfire_arch_qwen4::resolve_trunk_overlay(hfq, model, &gpu.arch, &admitted.config)?
                .map(|overlay| overlay.path().display().to_string());
        let hotset = hotset_expectation(model, &gpu, &admitted.manifest.identity.recipe)?;
        (overlay, hotset)
    };
    let (state_request, want_gdn) = hw_identity::gdn_test_state_quant()?;
    let backend = Qwen4KvBackend::automatic(&gpu);
    let vmm_granularity = match backend {
        Qwen4KvBackend::Vmm => gpu
            .vmm_recommended_granularity()
            .map_err(|e| format!("qwen4: VMM granularity: {e}"))?,
        Qwen4KvBackend::Legacy => 1,
    };
    // The product loader: admission, PLE rows, trunk overlay, expert hot set /
    // automatic placement, weight fulfillment, assembly, forward and the native
    // MTP head, exactly as `hipfire serve` loads this artifact. `max_k` sizes the
    // MTP reserve of the placement with this test's K for every arm, the AR arm
    // included. `state_quant` is the GDN test mode's canonical request.
    let shared = Qwen4RankShared {
        continuous_batch: false,
        path: model_str.to_string(),
        max_seq: MAX_SEQ,
        sequence: None,
        vision_reserve_bytes: 0,
        backend,
        vmm_granularity,
        kv_mode: hipfire_runtime::config::get().kv_mode.clone(),
        state_quant: Some(state_request.to_string()),
        spec: SpecLoadCfg::default(),
        native_mtp: true,
        max_k: mtp_k,
    };
    let loaded = load_qwen4_rank(&mut gpu, None, &shared, None, None, source)?;
    if loaded.max_seq != MAX_SEQ {
        return Err(format!(
            "the loader admitted max_seq {}, not {MAX_SEQ}",
            loaded.max_seq
        ));
    }
    if !loaded.native_mtp {
        return Err("the product load declined the native MTP head; no arm runs without it".into());
    }
    let mut bundle = loaded.bundle;
    let state_format = bundle.state_format();
    if state_format.gdn != want_gdn {
        return Err(format!(
            "{}={state_request} resolved to GDN state {:?}, expected {want_gdn:?}",
            hw_identity::GDN_TEST_STATE_QUANT_ENV,
            state_format.gdn
        ));
    }
    let hotset_receipt = hotset_receipt(
        &hotset_expected,
        bundle
            .weights
            .expert_tiers()
            .and_then(|lease| lease.hotset_sidecar_sha256()),
        &model_id,
    )?;
    let vocab = bundle.config.vocab_size;
    let capture = native_mtp_row_capture(&gpu, &bundle.config);
    if let Some(run) = arm.wide {
        #[cfg(not(feature = "reference-parity"))]
        {
            return Err(format!(
                "wide G3 {} requires --features reference-parity for the live effective row cap",
                run.scenario
            ));
        }
        #[cfg(feature = "reference-parity")]
        {
            let want_rows = run.rows.parse::<usize>().map_err(|e| e.to_string())?;
            let effective =
                hipfire_arch_qwen4::wide_verify_oracle::dev_effective_row_exact_rows(&bundle, &gpu);
            if effective < want_rows || !capture {
                return Err(format!(
                    "wide G3 refused: requested R{want_rows}, effective rows={effective}, row_capture={capture}"
                ));
            }
            let want_qsa = match run.kv {
                "q8" => hipfire_arch_qwen4::QsaKvFormat::Q8,
                "f32" => hipfire_arch_qwen4::QsaKvFormat::F32,
                _ => return Err("wide G3 only admits Q8/F32 KV".into()),
            };
            if state_format.qsa != want_qsa {
                return Err(format!(
                    "requested KV {want_qsa:?}, resolved {:?}",
                    state_format.qsa
                ));
            }
        }
    }
    let head = hipfire_config::developer_var("HIPFIRE_MTP_DRAFT_HEAD")
        .unwrap_or_else(|_| "mq2r".to_string());
    let head_copy = DraftHeadPolicy::parse(&head).copy;
    if (arm.draft_head == Some("full")) == head_copy.is_some() {
        return Err(format!(
            "draft head policy {head:?} resolved to copy {head_copy:?}"
        ));
    }
    println!(
        "ARM {} arch={} prompt_tokens={} state_quant={state_request} gdn_format={} state={state_format:?} replay_backend={replay_backend} backend={} row_capture={capture} draft_head={head}",
        arm.name,
        gpu.arch,
        tokens.len(),
        state_format.gdn.name(),
        backend.name()
    );
    let mut ids = Vec::with_capacity(case.generated() + mtp_k);
    let mut termination = serde_json::json!({
        "budget": case.generated(), "stopped_at_eos": false,
    });
    if arm.route == Route::Ar {
        let logits = gpu.zeros(&[vocab], DType::F32).map_err(|e| e.to_string())?;
        bundle
            .forward_chunk_final(&mut gpu, &tokens, &logits, None)
            .map_err(|e| e.to_string())?;
        while ids.len() < case.generated() {
            ids.push(
                bundle
                    .forward_token_or_argmax(&mut gpu, None, &logits)
                    .map_err(|e| e.to_string())?,
            );
        }
        gpu.free_tensor(logits).map_err(|e| e.to_string())?;
    } else {
        if arm.route == Route::Batched && !capture {
            return Err("this GPU's GDN route does not capture verify rows".into());
        }
        let mut drafter = Qwen4MtpDrafter::new(mtp_k, MAX_SEQ, None);
        if arm.ngram {
            let triple = hipfire_config::ngram_mod_triple_for_arch(16, &gpu.arch);
            if arm.ngram_triple.is_some() && triple != (24, 48, 64) {
                return Err(format!(
                    "ngram-long resolved triple {triple:?}, expected (24, 48, 64)"
                ));
            }
            let config = qwen4_ngram_mod_config(triple)?;
            let copy_lane = hipfire_config::mtp_ngram_copy_enabled_for_arch(16, &gpu.arch);
            if arm.derived_k.is_some() && copy_lane != arm.copy_lane {
                return Err(format!(
                    "arm {} runs copy_lane={}, but the product switch resolved copy_lane={copy_lane}",
                    arm.name, arm.copy_lane
                ));
            }
            println!(
                "NGRAM triple={triple:?} n_match={} n_min={} n_max={} copy_lane={copy_lane}",
                config.n_match, config.n_min, config.n_max
            );
            drafter = drafter.with_ngram(Some(config)).with_copy_lane(copy_lane);
            drafter.configure_request(SpecRequestConfig {
                allow_ngram_modifier: true,
                ..SpecRequestConfig::default()
            });
        } else if arm.wide.is_some() {
            drafter.configure_request(SpecRequestConfig::default());
        }
        let budget = arm
            .wide
            .map_or(case.generated(), |run| wide_budget(run, case.generated()));
        let diagnostic_eos = if arm.wide.is_some_and(|run| run.scenario == "eos") {
            let reference: Vec<u32> = serde_json::from_slice(
                &fs::read(dir.join("ar.ids.json")).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            Some(wide_stop(&reference).0)
        } else {
            None
        };
        if let Some(run) = arm.wide {
            if !matches!(run.scenario, "product" | "product-copy") {
                #[cfg(feature = "reference-parity")]
                drafter.lab_force_window_drafts(Some(
                    run.rows.parse::<usize>().map_err(|e| e.to_string())? - 1,
                ))?;
            }
        }
        let mut seed = drafter
            .mtp_prefill(&mut gpu, &mut bundle, &tokens, &tokens, 0, false, &|| false)?
            .into_first_token()?;
        ids.push(seed);
        let eos = diagnostic_eos.unwrap_or(bundle.config.eos_token_id);
        let stop_at_eos =
            diagnostic_eos.is_some() || (arm.derived_k.is_some() && arm.wide.is_none());
        let mut windows: Vec<serde_json::Value> = Vec::new();
        #[cfg(feature = "reference-parity")]
        let mut retired = false;
        while ids.len() < budget && !(stop_at_eos && ids.last() == Some(&eos)) {
            let mut k_budget = mtp_k;
            if let Some(run) = arm.wide {
                let product = matches!(run.scenario, "product" | "product-copy");
                k_budget = mtp_k.min(budget - ids.len() - 1);
                if !product {
                    #[cfg(feature = "reference-parity")]
                    {
                        let drafts = (run.rows.parse::<usize>().map_err(|e| e.to_string())? - 1)
                            .min(k_budget);
                        drafter.lab_force_window_drafts(Some(drafts))?;
                        let copy_seen = drafter.request_stats().ngram_mod_windows > 0;
                        let copy_enabled = match run.scenario {
                            "return" => windows.len() >= 2 && !copy_seen,
                            "retire" => windows.len() >= 2,
                            "copy" => true,
                            _ => false,
                        };
                        drafter.lab_force_copy_enabled(Some(copy_enabled));
                        if run.scenario == "retire" && windows.len() >= 2 && !retired {
                            drafter.lab_retire_native()?;
                            retired = true;
                        }
                        k_budget = drafts;
                    }
                }
            }
            let position = bundle.state.position;
            let started = Counters::now();
            let window = drafter.mtp_step(
                &mut gpu,
                &mut bundle,
                position,
                seed,
                &ids,
                k_budget,
                eos,
                None,
            )?;
            if arm.wide.is_some() && window.committed.len() > budget - ids.len() {
                return Err("wide window exceeded its remaining token budget".into());
            }
            let counters = Counters::now().since(started);
            let after = drafter.request_stats();
            let mut row = vec![
                serde_json::json!(ids.len()),
                serde_json::json!(position),
                serde_json::json!(bundle.state.position),
                serde_json::json!(k_budget),
                serde_json::json!(window.drafts_generated),
                serde_json::json!(window.accepted),
                serde_json::json!(window.committed.len()),
                serde_json::json!(after.mtp_windows),
                serde_json::json!(after.ar_windows),
                serde_json::json!(after.ngram_mod_windows),
                serde_json::json!(after.mtp_retired),
            ];
            row.extend(counters.to_array().map(|count| serde_json::json!(count)));
            windows.push(serde_json::Value::Array(row));
            seed = *window.committed.last().ok_or("empty MTP window")?;
            ids.extend_from_slice(&window.committed);
            if stop_at_eos {
                if let Some(end) = ids.iter().position(|&id| id == eos) {
                    ids.truncate(end + 1);
                }
            }
        }
        if arm.wide.is_none() {
            ids.truncate(case.generated());
        }
        if arm.wide.is_some_and(|run| run.scenario == "eos") && ids.last() != Some(&eos) {
            return Err("diagnostic EOS was not actually reached".into());
        }
        termination = serde_json::json!({
            "budget": budget,
            "stop_token": eos,
            "diagnostic_eos": diagnostic_eos.is_some(),
            "stopped_at_eos": stop_at_eos && ids.last() == Some(&eos),
            "generated": ids.len(),
            "position": bundle.state.position,
        });
        let stats = drafter.request_stats();
        println!("STATS {stats:?}");
        if arm.ngram && !stats.mtp_ngram {
            return Err("n-gram pool was not armed for the request".into());
        }
        let mut columns: Vec<&str> = WINDOW_COLUMNS.to_vec();
        columns.extend(hw_identity::COUNTER_COLUMNS);
        fs::write(
            dir.join(format!("{}.windows.json", arm.name)),
            serde_json::to_vec_pretty(&serde_json::json!({
                "columns": columns,
                "windows": windows,
                "request_stats": format!("{stats:?}"),
            }))
            .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        Box::new(drafter).mtp_free(&mut gpu);
    }
    fs::write(
        dir.join(format!("{}.ids.json", arm.name)),
        serde_json::to_vec(&ids).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    fs::write(
        dir.join(format!("{}.txt", arm.name)),
        tokenizer.decode(&ids),
    )
    .map_err(|e| e.to_string())?;
    let effective_rows: Option<usize> = if arm.wide.is_some() {
        #[cfg(feature = "reference-parity")]
        {
            Some(
                hipfire_arch_qwen4::wide_verify_oracle::dev_effective_row_exact_rows(&bundle, &gpu),
            )
        }
        #[cfg(not(feature = "reference-parity"))]
        {
            return Err(
                "wide G3 requires --features reference-parity for the live effective row cap"
                    .into(),
            );
        }
    } else {
        None
    };
    let identity =
        hw_identity::config_identity(&hw_identity::env_snapshot(&[], &CONTROL_ENV), &model_id);
    fs::write(
        dir.join(format!("{}.identity.json", arm.name)),
        serde_json::to_vec_pretty(&serde_json::json!({
            "arm": arm.name,
            "case": case.name(),
            "config": identity,
            "derived_k": std::env::var(DERIVED_K_ENV).ok(),
            "eos_token": bundle.config.eos_token_id,
            "termination": termination,
            "wide": arm.wide.map(|run| serde_json::json!({
                "requested_rows": run.rows,
                "scenario": run.scenario,
                "state": run.state,
                "kv_requested": run.kv,
                "qsa_resolved": format!("{:?}", state_format.qsa),
                "effective_rows": effective_rows,
            })),
            "gdn_state": {
                "request": state_request,
                "resolved": state_format.gdn.name(),
            },
            "graph_policy": {
                "replay_backend": replay_backend,
                "replay_state": format!("{:?}", gpu.replay.state()),
                "HIPFIRE_GRAPH": std::env::var("HIPFIRE_GRAPH").ok(),
                "note": GRAPH_POLICY_NOTE,
            },
            "receipt": hw_identity::receipt(&gpu, model),
            "artifacts_used": {
                "loader": "hipfire_loader::load_qwen4_rank",
                "placement_native_mtp_k": mtp_k,
                "ple_rows": ple_receipt,
                "trunk_overlay": {
                    "expected_path": overlay_expected,
                    "sha256": overlay_expected.as_ref().map(|_| model_id["trunk-overlay"]["sha256"].clone()),
                    "applied_evidence": "stderr `qwen4 trunk overlay: applied` line, required by the parent",
                },
                "expert_hotset": hotset_receipt,
            },
        }))
        .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    bundle.free_gpu(&mut gpu).map_err(|e| e.to_string())
}
