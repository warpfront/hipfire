// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen4_slowlink_quality — native S1 production-route quality probe (Q1).
//!
//! One binary, five separate GPU processes (`--phase baseline|s1a|s1b|beta-f32|
//! s1-f32`, one model load each) and two CPU-only phases (`--phase
//! check-plan|score`). Every GPU phase loads the
//! artifact fresh through the product loader
//! (`slowlink_load::slowlink_load_bundle(.., None, AdmissionPurpose::Product)`),
//! asserts the actual placement / scratch / tile against the plan, and then
//! runs its cells from a freshly reset state. The parent harness owns the
//! environment (selector, D2 off, PLE q4, no MTP / cache / hot set); this
//! program only ASSERTS it and fails on any mismatch. There is no fallback
//! route, no synthesized receipt and no stubbed number.
//!
//! # Phases and cells
//! * `baseline` (selector `0`): per primary context a teacher-forced cell and
//!   a short greedy cell, then the 3400-token resume prompt as one direct call
//!   (`resume-cold`) and as the exact split `1537 + 1863` (`resume-split`), the
//!   fourth q8 resume arm (all through `forward_chunk_final`, 4096 scratch).
//! * `s1a` (selector `1`/auto): FIRST the seven determinism cases (greedy,
//!   every logits row captured), THEN the primary cells and the resume cells
//!   (split `1537 + 1863` as two direct calls, then the one-call 3400 direct).
//! * `s1b`: the seven determinism cases only, identical order and config.
//! * `beta-f32` / `s1-f32` (the f32 diagnostic pair, `HIPFIRE_STATE_QUANT=f32`,
//!   KV still q8, everything else identical): `resume-cold` on the baseline
//!   route / `resume-split` on the S1 route.
//!
//! Pinned class flags (the original j1968 cells), asserted on every route:
//! `HIPFIRE_STATE_QUANT=q8` (f32 in the two f32 phases), `HIPFIRE_KV_MODE=q8`,
//! `HIPFIRE_GRAPH=0`, `HIPFIRE_NORMALIZE_PROMPT=0`,
//! `HIPFIRE_CHAT_CURRENT_DATE=2026-10-17`, `HIPFIRE_OFFLOAD_EXEC=pcie`,
//! `HIPFIRE_PLE_FORMAT=q4`, `HIPFIRE_QWEN4_D2_DEFAULT=0`; and the loaded
//! request state's GDN / QSA K-V storage is asserted from the bundle itself.
//! The baseline routes (`baseline`, `beta-f32`) additionally run with the
//! explicit N9 override `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=9` and
//! `HIPFIRE_PREFILL_CHUNK_ROWS=4096` (placement `explicit-env`, 9 resident
//! layers, 4096 scratch); the S1 routes (`s1a`, `s1b`, `s1-f32`) MUST leave
//! the layer override unset (any value declines S1) and the admission picks
//! the plan's resident layers. Values are read through `hipfire_config::developer_var`, the
//! process snapshot the product itself reads (every `HIPFIRE_*` variable is
//! captured once at start; the program never sets or edits the environment).
//!
//! Cell names: `det-<case id>`, `primary-<ctx>-teacher`, `primary-<ctx>-greedy`,
//! `resume-split`, `resume-cold`.
//!
//! Row `r` of a teacher cell is the logits row after the prefill plus `r`
//! teacher tokens (row 0 is the final prefill row), so a primary cell scores
//! `512` rows after feeding the first `511` held-out tokens, and the resume
//! cell `32` rows after the `31` shared teacher ids. A greedy cell captures the
//! seed row and the row after EVERY generated token, including the last one
//! and an EOS (the daemon forwards a token before it checks for EOS):
//! `rows = ids + 1`.
//!
//! # Output directory (fresh only)
//! `--out-dir` must not exist (it is created) or be an empty directory; every
//! file is `create_new`. Per cell: `<cell>.logits.f32le` (rows back to back,
//! canonical little-endian IEEE-754 bits, `vocab * 4` bytes per row, no header)
//! and `<cell>.rows.jsonl`. Per phase: `cells.jsonl` (one provenance record per
//! cell with the actual receipts), `phase-summary.json` (written last: its
//! presence means the phase completed), and for the determinism cases the
//! greedy-ids compatible `ids.jsonl` / `receipts.jsonl`.
//!
//! A row line carries `cell`, `row`, `input_token` (null for the seed row),
//! `context_position` (tokens in the context the row conditions on),
//! `next_token` (the teacher / generated token that followed it, null where
//! none), `logits_sha256` (over the row's canonical bytes), `bin_offset`,
//! `top1_*` / `top2_*` (index, value, raw f32 bits; ties resolve to the lower
//! index) and `top12_margin`, plus `neg_inf_count`. NaN and `+inf` logits fail
//! the run; `-inf` is zero mass and is counted. The cell's `stream_sha256` is
//! the SHA-256 of its whole `.logits.f32le` file.
//!
//! # Scoring
//! `--phase score` re-reads two phases' dumps (verifying every row and stream
//! hash), pairs rows by index (same absolute position and same next token),
//! and writes `kl.jsonl` / `kl-summary.json` with `KL(reference || candidate)`
//! in f64 nats (max-shifted log-softmax, Neumaier-compensated sums). `-inf`
//! logits are zero probability mass; NaN, `+inf` and an all-`-inf` reference
//! are errors; a candidate without mass where the reference has some is an
//! infinite divergence (reported as `kl_nats: null`, `kl_status: "infinite"`).
//! No confidence interval is computed or implied.
//!
//! `--phase score-f32` (CPU) scores ONLY the f32 resume pair
//! (`resume-f32-s1split-vs-betacold`: `beta-f32` `resume-cold` against `s1-f32`
//! `resume-split`) from `--beta-f32-dir` / `--s1-f32-dir`; no baseline or `s1a`
//! directory is read and no q8 pair is recomputed. It reuses the very same
//! row/hash verification and KL scorer as `score`, and writes `kl.jsonl` and
//! `kl-summary.json` (`scope: "f32-pair-only"`) to its own fresh `--out-dir`.

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::slowlink_admission::{
    resident_layer_switch_is_override, AdmissionPurpose, SlowlinkMode, S1_RESIDENT_LAYERS_ENV,
};
use hipfire_arch_qwen4::slowlink_load::slowlink_load_bundle;
use hipfire_arch_qwen4::{GdnStateFormat, QsaKvFormat};
use hipfire_config::memory::{ple_sidecar_path, PleFormat};
use hipfire_runtime::prompt_frame::{JinjaChatFrame, Message};
use hipfire_runtime::tokenizer::{maybe_normalize_prompt, Tokenizer};
use hipfire_runtime::verified_cache::{StatKey, VerifiedCache};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "\
usage:
  qwen4_slowlink_quality --phase baseline|s1a|s1b|beta-f32|s1-f32 --plan PLAN.json --manifest MANIFEST.json --model MODEL.hfq --out-dir NEW_DIR
  qwen4_slowlink_quality --phase score --plan PLAN.json --baseline-dir DIR --s1a-dir DIR [--s1b-dir DIR] [--beta-f32-dir DIR --s1-f32-dir DIR] --out-dir NEW_DIR
  qwen4_slowlink_quality --phase score-f32 --plan PLAN.json --beta-f32-dir DIR --s1-f32-dir DIR --out-dir NEW_DIR
  qwen4_slowlink_quality --phase check-plan --plan PLAN.json
  qwen4_slowlink_quality --help

Every path is absolute. --out-dir must not exist (it is created) or be an empty
directory; nothing is ever overwritten. The five GPU phases are separate
processes with the same plan and the same launch configuration (the parent sets
HIPFIRE_QWEN4_SLOWLINK_PREFILL=0 for baseline and beta-f32, 1 or unset for
s1a/s1b/s1-f32; baseline and beta-f32 also set HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=9
and HIPFIRE_PREFILL_CHUNK_ROWS=4096 (expect.baseline_env); s1a/s1b/s1-f32 leave
HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS unset (expect.env_absent: any value declines
S1); beta-f32/s1-f32 set HIPFIRE_STATE_QUANT=f32.

PLAN.json (strict: unknown keys are refused), schema \"hipfire.qwen4.slowlink_quality.plan.v1\":
  max_seq                 >= 16897 and >= every cell's prompt + continuation + 1
  artifact                {path, size, sha256}    the model's pinned identity
  expect                  {resident_layers: 9, baseline_tile_rows: 4096, s1_tile_rows: 768,
                           s1_layer_major_min_rows, ple_format, env: {NAME: VALUE},
                           env_absent: [NAME], baseline_env: {NAME: VALUE}}
                           asserted against the live process; env MUST carry the 8
                           pinned class flags (STATE_QUANT=q8, KV_MODE=q8, GRAPH=0,
                           NORMALIZE_PROMPT=0, CHAT_CURRENT_DATE=2026-10-17,
                           OFFLOAD_EXEC=pcie, PLE_FORMAT=q4, QWEN4_D2_DEFAULT=0);
                           baseline_env (baseline/beta-f32 only) MUST carry
                           HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=9 and
                           HIPFIRE_PREFILL_CHUNK_ROWS=4096; env_absent (S1 routes
                           only) MUST name HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS
  determinism             {case_ids: [7 ids, run order], max_new_tokens: 32, thinking,
                           cases: [{id, prompt_ids | messages(+tools), resume_at?,
                                    expected_prompt_tokens, retained_ids?}]}
  primary                 {heldout_rows: 512, greedy_new_tokens: 32,
                           contexts: [{id: ctx16k, prefix_ids: 16384, heldout_ids: 512,
                                       prefix_sha256, heldout_sha256, sources: [...]},
                                      {id: ctx4k, prefix_ids: 4097, heldout_ids: 512, ...}]}
                           sources: [{label, path, sha256, concat_offset, token_count}]
                           contiguously cover prefix||heldout; sha256 is over the ids as u32 LE.
  resume                  {prompt_ids: 3400, prompt_sha256, split_at: 1537,
                           teacher_ids: 31, teacher_sha256, retained_s1_ids: 32}
                           teacher_ids are the first 31 of retained_s1_ids.

MANIFEST.json (strict, written by the parent harness, never by this program),
schema \"hipfire.qwen4.slowlink_quality.manifest.v1\":
  binary {sha256, ..}     sha256 must equal this running executable
  source {..}             source / worktree identity, embedded verbatim
  launch {..}?            command, environment, device, locks; embedded verbatim
  model  {path, dev, ino, size, mtime_sec, mtime_nsec, sha256}
                          trusted identity tuple; dev/ino/size/mtime must equal the
                          model's stat, path/size/sha256 must equal plan.artifact.
                          The model is never read or hashed here (ctime is recorded
                          as metadata only).
";

const PLAN_SCHEMA: &str = "hipfire.qwen4.slowlink_quality.plan.v1";
const MANIFEST_SCHEMA: &str = "hipfire.qwen4.slowlink_quality.manifest.v1";
const CELL_SCHEMA: &str = "hipfire.qwen4.slowlink_quality.cell.v1";
const ROW_SCHEMA: &str = "hipfire.qwen4.slowlink_quality.row.v1";
const SUMMARY_SCHEMA: &str = "hipfire.qwen4.slowlink_quality.phase_summary.v1";
const KL_ROW_SCHEMA: &str = "hipfire.qwen4.slowlink_quality.kl_row.v1";
const KL_SUMMARY_SCHEMA: &str = "hipfire.qwen4.slowlink_quality.kl_summary.v1";
const DETERMINISM_SCHEMA: &str = "hipfire.qwen4.slowlink_quality.determinism.v1";

/// Production placement the probe is defined at: 9 resident trunk layers.
const EXPECTED_RESIDENT_LAYERS: usize = 9;
/// Incumbent forward scratch rows (the baseline arm's tile and chunk).
const BASELINE_TILE_ROWS: usize = 4096;
/// The admitted S1 product tile.
const S1_TILE_ROWS: usize = 768;
const DET_CASES: usize = 7;
const DET_MAX_NEW: usize = 32;
const HELDOUT_ROWS: usize = 512;
const GREEDY_NEW: usize = 32;
const CTX_16K: &str = "ctx16k";
const CTX_16K_PREFIX: usize = 16384;
const CTX_4K: &str = "ctx4k";
const CTX_4K_PREFIX: usize = 4097;
const RESUME_PROMPT_TOKENS: usize = 3400;
const RESUME_SPLIT_AT: usize = 1537;
const RESUME_TEACHER_IDS: usize = 31;
const RESUME_RETAINED_IDS: usize = 32;
/// The 16K context: its prefix, the held-out rows and the trailer slot.
const MIN_MAX_SEQ: usize = CTX_16K_PREFIX + HELDOUT_ROWS + 1;
const RESUME_COLD: &str = "resume-cold";
const RESUME_SPLIT: &str = "resume-split";
/// The class flags the original j1968 cells ran with (greedy-s1 / greedy-beta
/// `spawn-env.json`), pinned on EVERY route and every phase. The f32 phases
/// differ in `HIPFIRE_STATE_QUANT` only (see [`effective_env`]).
const PINNED_ENV: [(&str, &str); 8] = [
    ("HIPFIRE_STATE_QUANT", "q8"),
    ("HIPFIRE_KV_MODE", "q8"),
    ("HIPFIRE_GRAPH", "0"),
    ("HIPFIRE_NORMALIZE_PROMPT", "0"),
    ("HIPFIRE_CHAT_CURRENT_DATE", "2026-10-17"),
    ("HIPFIRE_OFFLOAD_EXEC", "pcie"),
    ("HIPFIRE_PLE_FORMAT", "q4"),
    ("HIPFIRE_QWEN4_D2_DEFAULT", "0"),
];
const STATE_QUANT_ENV: &str = "HIPFIRE_STATE_QUANT";
/// The N9 override of the baseline routes (`baseline`, `beta-f32`): an explicit
/// layer count pins the placement at 9 resident layers (`placement
/// "explicit-env"`). It declines the S1 selector, so on the S1 routes
/// (`s1a`, `s1b`, `s1-f32`) it MUST stay unset and the admission picks the 9.
const EXPLICIT_LAYERS_ENV: &str = "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS";
/// The incumbent forward scratch / chunk of the baseline routes.
const CHUNK_ROWS_ENV: &str = "HIPFIRE_PREFILL_CHUNK_ROWS";

// ── command line ───────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Baseline,
    S1a,
    S1b,
    /// STATE_QUANT=f32 diagnostic, baseline route: the one-call 3400 resume.
    BetaF32,
    /// STATE_QUANT=f32 diagnostic, S1 route: the 1537 + 1863 split resume.
    S1F32,
    Score,
    /// CPU: the f32 resume pair only (no q8 directory is read or rescored).
    ScoreF32,
    CheckPlan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Baseline,
    S1,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Baseline => "baseline",
            Arm::S1 => "s1",
        }
    }
}

impl Phase {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "baseline" => Ok(Phase::Baseline),
            "s1a" => Ok(Phase::S1a),
            "s1b" => Ok(Phase::S1b),
            "beta-f32" => Ok(Phase::BetaF32),
            "s1-f32" => Ok(Phase::S1F32),
            "score" => Ok(Phase::Score),
            "score-f32" => Ok(Phase::ScoreF32),
            "check-plan" => Ok(Phase::CheckPlan),
            other => Err(format!(
                "--phase: {other:?} is not baseline, s1a, s1b, beta-f32, s1-f32, score, score-f32 or check-plan"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Phase::Baseline => "baseline",
            Phase::S1a => "s1a",
            Phase::S1b => "s1b",
            Phase::BetaF32 => "beta-f32",
            Phase::S1F32 => "s1-f32",
            Phase::Score => "score",
            Phase::ScoreF32 => "score-f32",
            Phase::CheckPlan => "check-plan",
        }
    }

    /// The arm a GPU phase runs; `None` for the CPU phases.
    fn arm(self) -> Option<Arm> {
        match self {
            Phase::Baseline | Phase::BetaF32 => Some(Arm::Baseline),
            Phase::S1a | Phase::S1b | Phase::S1F32 => Some(Arm::S1),
            Phase::Score | Phase::ScoreF32 | Phase::CheckPlan => None,
        }
    }

    /// The two f32-state diagnostic phases (`HIPFIRE_STATE_QUANT=f32`, KV q8).
    fn state_f32(self) -> bool {
        matches!(self, Phase::BetaF32 | Phase::S1F32)
    }

    fn runs_determinism(self) -> bool {
        matches!(self, Phase::S1a | Phase::S1b)
    }

    fn runs_primary_and_resume(self) -> bool {
        matches!(self, Phase::Baseline | Phase::S1a)
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Args {
    phase: Phase,
    plan: PathBuf,
    manifest: Option<PathBuf>,
    model: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    baseline_dir: Option<PathBuf>,
    s1a_dir: Option<PathBuf>,
    s1b_dir: Option<PathBuf>,
    beta_f32_dir: Option<PathBuf>,
    s1_f32_dir: Option<PathBuf>,
}

fn absolute(flag: &str, value: String) -> Result<PathBuf, String> {
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err(format!("{flag} must be an absolute path"));
    }
    Ok(path)
}

fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<Args, String> {
    let mut args = args.into_iter();
    let mut phase = None;
    let mut plan = None;
    let mut manifest = None;
    let mut model = None;
    let mut out_dir = None;
    let mut baseline_dir = None;
    let mut s1a_dir = None;
    let mut s1b_dir = None;
    let mut beta_f32_dir = None;
    let mut s1_f32_dir = None;
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--phase" => phase = Some(Phase::parse(&value()?)?),
            "--plan" => plan = Some(absolute("--plan", value()?)?),
            "--manifest" => manifest = Some(absolute("--manifest", value()?)?),
            "--model" => model = Some(absolute("--model", value()?)?),
            "--out-dir" => out_dir = Some(absolute("--out-dir", value()?)?),
            "--baseline-dir" => baseline_dir = Some(absolute("--baseline-dir", value()?)?),
            "--s1a-dir" => s1a_dir = Some(absolute("--s1a-dir", value()?)?),
            "--s1b-dir" => s1b_dir = Some(absolute("--s1b-dir", value()?)?),
            "--beta-f32-dir" => beta_f32_dir = Some(absolute("--beta-f32-dir", value()?)?),
            "--s1-f32-dir" => s1_f32_dir = Some(absolute("--s1-f32-dir", value()?)?),
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    let phase = phase.ok_or("--phase is required")?;
    let plan = plan.ok_or("--plan is required")?;
    let args = Args {
        phase,
        plan,
        manifest,
        model,
        out_dir,
        baseline_dir,
        s1a_dir,
        s1b_dir,
        beta_f32_dir,
        s1_f32_dir,
    };
    args.check_flags()?;
    Ok(args)
}

/// One phase's flag presence table: `(flag, present)` rows.
type FlagRows<'a> = &'a [(&'a str, bool)];

impl Args {
    /// Required flags present, inapplicable ones refused.
    fn check_flags(&self) -> Result<(), String> {
        let gpu = [
            ("--manifest", self.manifest.is_some()),
            ("--model", self.model.is_some()),
            ("--out-dir", self.out_dir.is_some()),
        ];
        let score = [
            ("--baseline-dir", self.baseline_dir.is_some()),
            ("--s1a-dir", self.s1a_dir.is_some()),
            ("--out-dir", self.out_dir.is_some()),
        ];
        let score_f32 = [
            ("--beta-f32-dir", self.beta_f32_dir.is_some()),
            ("--s1-f32-dir", self.s1_f32_dir.is_some()),
            ("--out-dir", self.out_dir.is_some()),
        ];
        let optional_score = [
            ("--s1b-dir", self.s1b_dir.is_some()),
            ("--beta-f32-dir", self.beta_f32_dir.is_some()),
            ("--s1-f32-dir", self.s1_f32_dir.is_some()),
        ];
        let (required, optional): (FlagRows<'_>, FlagRows<'_>) = match self.phase {
            Phase::Baseline | Phase::S1a | Phase::S1b | Phase::BetaF32 | Phase::S1F32 => {
                (&gpu, &[])
            }
            Phase::Score => (&score, &optional_score),
            Phase::ScoreF32 => (&score_f32, &[]),
            Phase::CheckPlan => (&[], &[]),
        };
        for (flag, present) in required {
            if !present {
                return Err(format!(
                    "{flag} is required for --phase {}",
                    self.phase.name()
                ));
            }
        }
        let allowed: BTreeSet<&str> = required
            .iter()
            .chain(optional.iter())
            .map(|(flag, _)| *flag)
            .collect();
        let all = [
            ("--manifest", self.manifest.is_some()),
            ("--model", self.model.is_some()),
            ("--out-dir", self.out_dir.is_some()),
            ("--baseline-dir", self.baseline_dir.is_some()),
            ("--s1a-dir", self.s1a_dir.is_some()),
            ("--s1b-dir", self.s1b_dir.is_some()),
            ("--beta-f32-dir", self.beta_f32_dir.is_some()),
            ("--s1-f32-dir", self.s1_f32_dir.is_some()),
        ];
        for (flag, present) in all {
            if present && !allowed.contains(flag) {
                return Err(format!(
                    "{flag} does not apply to --phase {}",
                    self.phase.name()
                ));
            }
        }
        Ok(())
    }
}

// ── hashing ────────────────────────────────────────────────────────────

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// SHA-256 over `ids` as little-endian `u32`s.
fn sha_u32(ids: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for id in ids {
        hasher.update(id.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Streamed SHA-256 of a (small) file: plan, manifest, this executable, a dump.
/// NEVER called on the model.
fn file_sha256(path: &Path) -> Result<String, String> {
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

// ── plan ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: String,
    max_seq: usize,
    artifact: Artifact,
    expect: Expect,
    determinism: Determinism,
    primary: Primary,
    resume: Resume,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    path: String,
    size: u64,
    sha256: String,
}

/// Sealed execution recipe at `manifest.source.execution_recipe`: the ONLY
/// new-target authority. `schema` + `recipe_sha256` pin the canonical recipe
/// bytes; the `expect` triple is explicit sealed planned content the candidate
/// plan, the actual selected placement, and every layer-major receipt MUST
/// equal. A null `resident_layer_set` is the front placement
/// (`0..resident_layers`); the count is genuine planned content, never a
/// forced env value.
const EXECUTION_RECIPE_SCHEMA: &str = "hipfire.qwen4.slowlink_quality.execution_recipe.v1";

/// Sealed planned placement of one execution recipe (`expect` block only; the
/// recipe file digest + per-entry verified-cache digests gate the artifacts).
#[derive(Debug, Clone)]
struct SealedExpect {
    resident_layers: usize,
    resident_layer_set: Option<Vec<usize>>,
    placement: String,
    cache_home: String,
    cache_path: String,
    cache_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expect {
    resident_layers: usize,
    baseline_tile_rows: usize,
    s1_tile_rows: usize,
    /// S1-arm prefill calls of at least this many rows MUST run layer-major;
    /// shorter ones may take the incumbent route (the receipt says which).
    s1_layer_major_min_rows: usize,
    ple_format: String,
    env: BTreeMap<String, String>,
    /// S1 routes only (`s1a`, `s1b`, `s1-f32`): names that must be UNSET; must
    /// include `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` (any value declines S1).
    env_absent: Vec<String>,
    /// Baseline routes only (`baseline`, `beta-f32`), required in addition to
    /// `env`: must carry `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` = `resident_layers`
    /// (the explicit N9 override) and `HIPFIRE_PREFILL_CHUNK_ROWS` =
    /// `baseline_tile_rows` (the 4096 scratch).
    baseline_env: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Determinism {
    case_ids: Vec<String>,
    max_new_tokens: usize,
    thinking: bool,
    cases: Vec<DetCase>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct DetCase {
    id: String,
    prompt_ids: Option<Vec<u32>>,
    messages: Option<Vec<Message>>,
    tools: Option<Vec<Value>>,
    resume_at: Option<usize>,
    expected_prompt_tokens: usize,
    /// The ids an earlier run retained for this case (comparison only: a
    /// mismatch is reported, never silently accepted or enforced).
    retained_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Primary {
    heldout_rows: usize,
    greedy_new_tokens: usize,
    contexts: Vec<PrimaryContext>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrimaryContext {
    id: String,
    prefix_ids: Vec<u32>,
    heldout_ids: Vec<u32>,
    prefix_sha256: String,
    heldout_sha256: String,
    sources: Vec<Source>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    label: String,
    path: String,
    sha256: String,
    /// Offset of this source's first token in `prefix_ids || heldout_ids`.
    concat_offset: u64,
    token_count: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Resume {
    prompt_ids: Vec<u32>,
    prompt_sha256: String,
    split_at: usize,
    teacher_ids: Vec<u32>,
    teacher_sha256: String,
    retained_s1_ids: Vec<u32>,
}

fn valid_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 48
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

fn det_cell(id: &str) -> String {
    format!("det-{id}")
}

fn primary_cell(context: &str, kind: &str) -> String {
    format!("primary-{context}-{kind}")
}

impl PrimaryContext {
    /// The held-out ids fed after the prefill: every row but the last one's
    /// token, so `rows = feed + 1 = heldout rows`.
    fn feed(&self) -> &[u32] {
        &self.heldout_ids[..self.heldout_ids.len() - 1]
    }

    fn next_tokens(&self) -> Vec<Option<u32>> {
        self.heldout_ids.iter().map(|&t| Some(t)).collect()
    }
}

impl Resume {
    /// Row `i` is followed by `teacher_ids[i]`; the last row's follower is not
    /// part of the shared teacher history.
    fn next_tokens(&self) -> Vec<Option<u32>> {
        self.teacher_ids
            .iter()
            .map(|&t| Some(t))
            .chain(std::iter::once(None))
            .collect()
    }
}

impl Plan {
    fn from_slice(bytes: &[u8]) -> Result<Plan, String> {
        let plan: Plan = serde_json::from_slice(bytes).map_err(|e| format!("plan JSON: {e}"))?;
        plan.validate()?;
        Ok(plan)
    }

    #[cfg(test)]
    fn from_value(value: Value) -> Result<Plan, String> {
        let plan: Plan = serde_json::from_value(value).map_err(|e| format!("plan JSON: {e}"))?;
        plan.validate()?;
        Ok(plan)
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema != PLAN_SCHEMA {
            return Err(format!(
                "plan schema {:?} is not {PLAN_SCHEMA:?}",
                self.schema
            ));
        }
        self.artifact.validate()?;
        self.expect.validate()?;
        self.determinism.validate()?;
        self.primary.validate()?;
        self.resume.validate()?;
        let required = self.required_max_seq();
        if self.max_seq < MIN_MAX_SEQ || self.max_seq < required {
            return Err(format!(
                "max_seq {} is below the required {} (>= {MIN_MAX_SEQ} and every cell's prompt + continuation + 1)",
                self.max_seq,
                required.max(MIN_MAX_SEQ)
            ));
        }
        Ok(())
    }

    /// Smallest `max_seq` every cell with a known prompt length fits in
    /// (`prompt + continuation + 1`, the greedy driver's rule). A rendered
    /// `messages` prompt is checked after rendering.
    fn required_max_seq(&self) -> usize {
        let mut required = 0usize;
        for case in &self.determinism.cases {
            if let Some(ids) = &case.prompt_ids {
                required = required.max(ids.len() + self.determinism.max_new_tokens + 1);
            }
        }
        for context in &self.primary.contexts {
            let prefix = context.prefix_ids.len();
            required = required.max(prefix + context.heldout_ids.len() + 1);
            required = required.max(prefix + self.primary.greedy_new_tokens + 1);
        }
        required.max(self.resume.prompt_ids.len() + self.resume.teacher_ids.len() + 1)
    }

    /// Every id any cell feeds, against the loaded vocabulary.
    fn check_vocab(&self, vocab: usize) -> Result<(), String> {
        let check = |what: &str, ids: &[u32]| -> Result<(), String> {
            match ids.iter().find(|&&id| id as usize >= vocab) {
                Some(bad) => Err(format!(
                    "{what}: id {bad} is outside the vocabulary ({vocab})"
                )),
                None => Ok(()),
            }
        };
        for case in &self.determinism.cases {
            if let Some(ids) = &case.prompt_ids {
                check(&format!("case {}", case.id), ids)?;
            }
        }
        for context in &self.primary.contexts {
            check(
                &format!("context {} prefix", context.id),
                &context.prefix_ids,
            )?;
            check(
                &format!("context {} heldout", context.id),
                &context.heldout_ids,
            )?;
        }
        check("resume prompt", &self.resume.prompt_ids)?;
        check("resume teacher", &self.resume.teacher_ids)
    }

    /// The cells a phase produces, in run order.
    fn cell_names(&self, phase: Phase) -> Vec<String> {
        let mut names = Vec::new();
        if phase.runs_determinism() {
            names.extend(self.determinism.case_ids.iter().map(|id| det_cell(id)));
        }
        if phase.runs_primary_and_resume() {
            for context in &self.primary.contexts {
                names.push(primary_cell(&context.id, "teacher"));
                names.push(primary_cell(&context.id, "greedy"));
            }
            if phase == Phase::S1a {
                names.push(RESUME_SPLIT.to_string());
            }
            names.push(RESUME_COLD.to_string());
            if phase == Phase::Baseline {
                // The exact baseline split: the fourth q8 resume arm.
                names.push(RESUME_SPLIT.to_string());
            }
        }
        match phase {
            Phase::BetaF32 => names.push(RESUME_COLD.to_string()),
            Phase::S1F32 => names.push(RESUME_SPLIT.to_string()),
            _ => {}
        }
        names
    }
}

impl Artifact {
    fn validate(&self) -> Result<(), String> {
        if !Path::new(&self.path).is_absolute() {
            return Err(format!("artifact.path {:?} is not absolute", self.path));
        }
        if self.size == 0 {
            return Err("artifact.size must be > 0".into());
        }
        if !is_sha256_hex(&self.sha256) {
            return Err("artifact.sha256 is not 64 lowercase hex digits".into());
        }
        Ok(())
    }
}

impl Expect {
    fn validate(&self) -> Result<(), String> {
        if self.resident_layers != EXPECTED_RESIDENT_LAYERS {
            return Err(format!(
                "expect.resident_layers is {}, the production probe requires {EXPECTED_RESIDENT_LAYERS}",
                self.resident_layers
            ));
        }
        if self.baseline_tile_rows != BASELINE_TILE_ROWS {
            return Err(format!(
                "expect.baseline_tile_rows is {}, the baseline scratch is {BASELINE_TILE_ROWS}",
                self.baseline_tile_rows
            ));
        }
        if self.s1_tile_rows != S1_TILE_ROWS {
            return Err(format!(
                "expect.s1_tile_rows is {}, the admitted S1 tile is {S1_TILE_ROWS}",
                self.s1_tile_rows
            ));
        }
        if self.s1_layer_major_min_rows == 0 {
            return Err("expect.s1_layer_major_min_rows must be > 0".into());
        }
        if self.ple_format.is_empty() {
            return Err("expect.ple_format is empty".into());
        }
        for name in self
            .env
            .keys()
            .chain(self.baseline_env.keys())
            .chain(self.env_absent.iter())
        {
            if name.is_empty() {
                return Err("expect.env / env_absent holds an empty name".into());
            }
        }
        if let Some(name) = self
            .env_absent
            .iter()
            .find(|name| self.env.contains_key(name.as_str()))
        {
            return Err(format!("expect: {name} is both required and absent"));
        }
        for (name, want) in PINNED_ENV {
            if self.env.get(name).map(String::as_str) != Some(want) {
                return Err(format!(
                    "expect.env[{name}] must be {want:?}: the original j1968 class flag, pinned on every route"
                ));
            }
        }
        if !self
            .env_absent
            .iter()
            .any(|name| name == EXPLICIT_LAYERS_ENV)
        {
            return Err(format!(
                "expect.env_absent (S1 routes) must name {EXPLICIT_LAYERS_ENV} (an explicit layer count declines the selector)"
            ));
        }
        let resident = self.resident_layers.to_string();
        let chunk = self.baseline_tile_rows.to_string();
        for (name, want) in [
            (EXPLICIT_LAYERS_ENV, resident.as_str()),
            (CHUNK_ROWS_ENV, chunk.as_str()),
        ] {
            if self.baseline_env.get(name).map(String::as_str) != Some(want) {
                return Err(format!(
                    "expect.baseline_env[{name}] must be {want:?}: the baseline routes pin N9 explicitly with the 4096 scratch"
                ));
            }
        }
        if let Some(name) = self
            .baseline_env
            .keys()
            .find(|name| self.env.contains_key(name.as_str()))
        {
            return Err(format!("expect: {name} is in both env and baseline_env"));
        }
        Ok(())
    }
}

impl Determinism {
    fn validate(&self) -> Result<(), String> {
        if self.cases.len() != DET_CASES {
            return Err(format!(
                "determinism.cases has {} cases, expected {DET_CASES}",
                self.cases.len()
            ));
        }
        let ids: Vec<&str> = self.cases.iter().map(|case| case.id.as_str()).collect();
        let declared: Vec<&str> = self.case_ids.iter().map(String::as_str).collect();
        if ids != declared {
            return Err("determinism.case_ids does not match the cases' ids in order".into());
        }
        let unique: BTreeSet<&str> = ids.iter().copied().collect();
        if unique.len() != ids.len() {
            return Err("determinism.case_ids has duplicates".into());
        }
        if self.max_new_tokens != DET_MAX_NEW {
            return Err(format!(
                "determinism.max_new_tokens is {}, expected {DET_MAX_NEW}",
                self.max_new_tokens
            ));
        }
        for case in &self.cases {
            case.validate(self.max_new_tokens)?;
        }
        Ok(())
    }
}

impl DetCase {
    fn validate(&self, max_new: usize) -> Result<(), String> {
        let at = |e: &str| format!("case {:?}: {e}", self.id);
        if !valid_name(&self.id) {
            return Err(at("id must be 1-48 of [A-Za-z0-9_-]"));
        }
        if self.expected_prompt_tokens == 0 {
            return Err(at("expected_prompt_tokens must be > 0"));
        }
        match (&self.prompt_ids, &self.messages) {
            (Some(_), Some(_)) => return Err(at("both prompt_ids and messages given")),
            (None, None) => return Err(at("needs prompt_ids or messages")),
            (Some(ids), None) => {
                if ids.is_empty() {
                    return Err(at("prompt_ids is empty"));
                }
                if ids.len() != self.expected_prompt_tokens {
                    return Err(at(&format!(
                        "prompt_ids has {} ids but expected_prompt_tokens is {}",
                        ids.len(),
                        self.expected_prompt_tokens
                    )));
                }
                if self.tools.is_some() {
                    return Err(at("tools needs messages"));
                }
                if let Some(resume_at) = self.resume_at {
                    if resume_at == 0 || resume_at >= ids.len() {
                        return Err(at(&format!(
                            "resume_at {resume_at} is not interior to {} prompt_ids",
                            ids.len()
                        )));
                    }
                }
            }
            (None, Some(messages)) => {
                if messages.is_empty() {
                    return Err(at("messages is empty"));
                }
                if self.resume_at.is_some() {
                    return Err(at("resume_at needs prompt_ids"));
                }
            }
        }
        if let Some(retained) = &self.retained_ids {
            if retained.is_empty() || retained.len() > max_new {
                return Err(at(&format!(
                    "retained_ids has {} ids, expected 1..={max_new}",
                    retained.len()
                )));
            }
        }
        Ok(())
    }
}

impl Primary {
    fn validate(&self) -> Result<(), String> {
        if self.heldout_rows != HELDOUT_ROWS {
            return Err(format!(
                "primary.heldout_rows is {}, expected {HELDOUT_ROWS}",
                self.heldout_rows
            ));
        }
        if self.greedy_new_tokens != GREEDY_NEW {
            return Err(format!(
                "primary.greedy_new_tokens is {}, expected {GREEDY_NEW}",
                self.greedy_new_tokens
            ));
        }
        let expected = [(CTX_16K, CTX_16K_PREFIX), (CTX_4K, CTX_4K_PREFIX)];
        if self.contexts.len() != expected.len() {
            return Err(format!(
                "primary.contexts has {} contexts, expected {}",
                self.contexts.len(),
                expected.len()
            ));
        }
        for (context, (id, prefix)) in self.contexts.iter().zip(expected) {
            context.validate(id, prefix, self.heldout_rows)?;
        }
        Ok(())
    }
}

impl PrimaryContext {
    fn validate(&self, id: &str, prefix: usize, heldout: usize) -> Result<(), String> {
        let at = |e: String| format!("context {id}: {e}");
        if self.id != id {
            return Err(at(format!(
                "id is {:?}, expected {id:?} in this position",
                self.id
            )));
        }
        if self.prefix_ids.len() != prefix {
            return Err(at(format!(
                "prefix_ids has {} ids, expected {prefix}",
                self.prefix_ids.len()
            )));
        }
        if self.heldout_ids.len() != heldout {
            return Err(at(format!(
                "heldout_ids has {} ids, expected {heldout}",
                self.heldout_ids.len()
            )));
        }
        for (what, declared, ids) in [
            ("prefix_sha256", &self.prefix_sha256, &self.prefix_ids),
            ("heldout_sha256", &self.heldout_sha256, &self.heldout_ids),
        ] {
            if !is_sha256_hex(declared) {
                return Err(at(format!("{what} is not 64 lowercase hex digits")));
            }
            let computed = sha_u32(ids);
            if &computed != declared {
                return Err(at(format!(
                    "{what} mismatch (declared {declared}, computed {computed})"
                )));
            }
        }
        if self.sources.is_empty() {
            return Err(at("sources is empty".into()));
        }
        let mut offset = 0u64;
        for (i, source) in self.sources.iter().enumerate() {
            if source.label.is_empty() || !Path::new(&source.path).is_absolute() {
                return Err(at(format!("source {i}: label empty or path not absolute")));
            }
            if !is_sha256_hex(&source.sha256) {
                return Err(at(format!(
                    "source {i}: sha256 is not 64 lowercase hex digits"
                )));
            }
            if source.token_count == 0 {
                return Err(at(format!("source {i}: token_count is 0")));
            }
            if source.concat_offset != offset {
                return Err(at(format!(
                    "source {i}: concat_offset {} is not contiguous (expected {offset})",
                    source.concat_offset
                )));
            }
            offset = offset
                .checked_add(source.token_count)
                .ok_or_else(|| at("sources overflow".into()))?;
        }
        let total = (self.prefix_ids.len() + self.heldout_ids.len()) as u64;
        if offset != total {
            return Err(at(format!(
                "sources cover {offset} tokens, prefix + heldout is {total}"
            )));
        }
        Ok(())
    }
}

impl Resume {
    fn validate(&self) -> Result<(), String> {
        if self.prompt_ids.len() != RESUME_PROMPT_TOKENS {
            return Err(format!(
                "resume.prompt_ids has {} ids, expected {RESUME_PROMPT_TOKENS}",
                self.prompt_ids.len()
            ));
        }
        if self.split_at != RESUME_SPLIT_AT {
            return Err(format!(
                "resume.split_at is {}, expected {RESUME_SPLIT_AT}",
                self.split_at
            ));
        }
        if self.teacher_ids.len() != RESUME_TEACHER_IDS {
            return Err(format!(
                "resume.teacher_ids has {} ids, expected {RESUME_TEACHER_IDS}",
                self.teacher_ids.len()
            ));
        }
        if self.retained_s1_ids.len() != RESUME_RETAINED_IDS {
            return Err(format!(
                "resume.retained_s1_ids has {} ids, expected {RESUME_RETAINED_IDS}",
                self.retained_s1_ids.len()
            ));
        }
        if self.retained_s1_ids[..RESUME_TEACHER_IDS] != self.teacher_ids[..] {
            return Err(format!(
                "resume.teacher_ids is not the first {RESUME_TEACHER_IDS} ids of resume.retained_s1_ids"
            ));
        }
        for (what, declared, ids) in [
            ("prompt_sha256", &self.prompt_sha256, &self.prompt_ids),
            ("teacher_sha256", &self.teacher_sha256, &self.teacher_ids),
        ] {
            if !is_sha256_hex(declared) {
                return Err(format!("resume.{what} is not 64 lowercase hex digits"));
            }
            let computed = sha_u32(ids);
            if &computed != declared {
                return Err(format!(
                    "resume.{what} mismatch (declared {declared}, computed {computed})"
                ));
            }
        }
        Ok(())
    }
}

// ── external manifest and model identity ───────────────────────────────

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    binary: Value,
    source: Value,
    launch: Option<Value>,
    model: ModelTuple,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelTuple {
    path: String,
    dev: u64,
    ino: u64,
    size: u64,
    mtime_sec: i64,
    mtime_nsec: i64,
    sha256: String,
}

/// `stat` of the model (ctime is metadata only, never compared).
#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelStat {
    dev: u64,
    ino: u64,
    size: u64,
    mtime_sec: i64,
    mtime_nsec: i64,
    ctime_sec: i64,
    ctime_nsec: i64,
}

fn stat_model(path: &Path) -> Result<ModelStat, String> {
    use std::os::unix::fs::MetadataExt;
    let meta =
        std::fs::metadata(path).map_err(|e| format!("stat model {}: {e}", path.display()))?;
    if !meta.is_file() {
        return Err(format!("model {} is not a regular file", path.display()));
    }
    Ok(ModelStat {
        dev: meta.dev(),
        ino: meta.ino(),
        size: meta.size(),
        mtime_sec: meta.mtime(),
        mtime_nsec: meta.mtime_nsec(),
        ctime_sec: meta.ctime(),
        ctime_nsec: meta.ctime_nsec(),
    })
}

impl Manifest {
    fn from_slice(bytes: &[u8], artifact: &Artifact) -> Result<Manifest, String> {
        let manifest: Manifest =
            serde_json::from_slice(bytes).map_err(|e| format!("manifest JSON: {e}"))?;
        manifest.validate(artifact)?;
        Ok(manifest)
    }

    fn validate(&self, artifact: &Artifact) -> Result<(), String> {
        if self.schema != MANIFEST_SCHEMA {
            return Err(format!(
                "manifest schema {:?} is not {MANIFEST_SCHEMA:?}",
                self.schema
            ));
        }
        let binary_sha = self.binary_sha256()?;
        if !is_sha256_hex(binary_sha) {
            return Err("manifest.binary.sha256 is not 64 lowercase hex digits".into());
        }
        match self.source.as_object() {
            Some(map) if !map.is_empty() => {}
            _ => return Err("manifest.source must be a non-empty object".into()),
        }
        if let Some(launch) = &self.launch {
            if !launch.is_object() {
                return Err("manifest.launch must be an object".into());
            }
        }
        if !is_sha256_hex(&self.model.sha256) {
            return Err("manifest.model.sha256 is not 64 lowercase hex digits".into());
        }
        if self.model.path != artifact.path
            || self.model.size != artifact.size
            || self.model.sha256 != artifact.sha256
        {
            return Err(format!(
                "manifest.model (path {:?}, size {}, sha256 {}) differs from plan.artifact (path {:?}, size {}, sha256 {})",
                self.model.path,
                self.model.size,
                self.model.sha256,
                artifact.path,
                artifact.size,
                artifact.sha256
            ));
        }
        Ok(())
    }

    fn binary_sha256(&self) -> Result<&str, String> {
        self.binary
            .get("sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| "manifest.binary.sha256 is missing".to_string())
    }
}

/// The trusted tuple against the observed `stat`: device, inode, size and
/// mtime must all be equal. The pinned hash is trusted from the manifest;
/// the model is never read here.
fn verify_model_stat(tuple: &ModelTuple, stat: &ModelStat) -> Result<(), String> {
    let mut problems = Vec::new();
    if tuple.dev != stat.dev {
        problems.push(format!("dev {} != {}", tuple.dev, stat.dev));
    }
    if tuple.ino != stat.ino {
        problems.push(format!("ino {} != {}", tuple.ino, stat.ino));
    }
    if tuple.size != stat.size {
        problems.push(format!("size {} != {}", tuple.size, stat.size));
    }
    if tuple.mtime_sec != stat.mtime_sec || tuple.mtime_nsec != stat.mtime_nsec {
        problems.push(format!(
            "mtime {}.{:09} != {}.{:09}",
            tuple.mtime_sec, tuple.mtime_nsec, stat.mtime_sec, stat.mtime_nsec
        ));
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "model identity differs from the trusted manifest: {}",
            problems.join("; ")
        ))
    }
}

/// Read `manifest.source.execution_recipe` when present. `None` is the
/// untouched original replay path (literal original N9 plan, literal original
/// baseline_env N9 baseline, stat-only model identity). `Some` is the ONLY
/// new-target authority: schema + recipe file sha + sealed expect triple.
fn parse_execution_recipe(manifest: &Manifest) -> Result<Option<SealedExpect>, String> {
    let Some(recipe) = manifest.source.get("execution_recipe") else {
        return Ok(None);
    };
    let map = recipe
        .as_object()
        .ok_or_else(|| "manifest.source.execution_recipe must be an object".to_string())?;
    let schema = map
        .get("schema")
        .and_then(Value::as_str)
        .ok_or_else(|| "manifest.source.execution_recipe.schema is missing".to_string())?;
    if schema != EXECUTION_RECIPE_SCHEMA {
        return Err(format!(
            "manifest.source.execution_recipe.schema is {schema:?}, expected {EXECUTION_RECIPE_SCHEMA:?}"
        ));
    }
    let recipe_sha = map
        .get("recipe_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "manifest.source.execution_recipe.recipe_sha256 is missing".to_string())?;
    if !is_sha256_hex(recipe_sha) {
        return Err(
            "manifest.source.execution_recipe.recipe_sha256 is not 64 lowercase hex digits".into(),
        );
    }
    let expect = map
        .get("expect")
        .and_then(Value::as_object)
        .ok_or_else(|| "manifest.source.execution_recipe.expect must be an object".to_string())?;
    let resident_layers = expect
        .get("resident_layers")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            "manifest.source.execution_recipe.expect.resident_layers is missing".to_string()
        })? as usize;
    if resident_layers == 0 {
        return Err("manifest.source.execution_recipe.expect.resident_layers must be > 0".into());
    }
    let resident_layer_set = match expect.get("resident_layer_set") {
        None | Some(Value::Null) => None,
        Some(Value::Array(items)) => {
            let mut set = Vec::with_capacity(items.len());
            for item in items {
                let layer = item.as_u64().ok_or_else(|| {
                    "manifest.source.execution_recipe.expect.resident_layer_set holds a non-integer"
                        .to_string()
                })? as usize;
                set.push(layer);
            }
            if set.len() != resident_layers {
                return Err(format!(
                    "execution_recipe resident_layer_set has {} entries, expected {resident_layers}",
                    set.len()
                ));
            }
            let mut sorted = set.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() != set.len() {
                return Err("execution_recipe resident_layer_set has duplicate layers".into());
            }
            if sorted != set {
                return Err("execution_recipe resident_layer_set must be sorted ascending".into());
            }
            Some(set)
        }
        Some(_) => {
            return Err(
                "manifest.source.execution_recipe.expect.resident_layer_set must be an array or null"
                    .into(),
            );
        }
    };
    let placement = expect
        .get("placement")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            "manifest.source.execution_recipe.expect.placement is missing".to_string()
        })?;
    if placement.is_empty() {
        return Err("manifest.source.execution_recipe.expect.placement must be non-empty".into());
    }
    let verified = map
        .get("verified_cache")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            "manifest.source.execution_recipe.verified_cache must be an object".to_string()
        })?;
    let cache_home = verified
        .get("home")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            "manifest.source.execution_recipe.verified_cache.home is missing".to_string()
        })?;
    let cache_path = verified
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            "manifest.source.execution_recipe.verified_cache.path is missing".to_string()
        })?;
    let cache_sha256 = verified
        .get("sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            "manifest.source.execution_recipe.verified_cache.sha256 is missing".to_string()
        })?;
    if cache_home.is_empty() || cache_path.is_empty() {
        return Err(
            "manifest.source.execution_recipe.verified_cache.home/path must be non-empty".into(),
        );
    }
    if !is_sha256_hex(cache_sha256) {
        return Err(
            "manifest.source.execution_recipe.verified_cache.sha256 is not 64 lowercase hex digits"
                .into(),
        );
    }
    Ok(Some(SealedExpect {
        resident_layers,
        resident_layer_set,
        placement: placement.to_string(),
        cache_home: cache_home.to_string(),
        cache_path: cache_path.to_string(),
        cache_sha256: cache_sha256.to_string(),
    }))
}

/// Gate the candidate plan against the sealed recipe when one is present:
/// `Plan.expect.resident_layers` MUST equal the sealed count. No-recipe plans
/// keep the literal original validation untouched.
fn check_plan_against_recipe(plan: &Plan, recipe: Option<&SealedExpect>) -> Result<(), String> {
    let Some(sealed) = recipe else {
        return Ok(());
    };
    if plan.expect.resident_layers != sealed.resident_layers {
        return Err(format!(
            "plan.expect.resident_layers is {}, the sealed recipe requires {}",
            plan.expect.resident_layers, sealed.resident_layers
        ));
    }
    if plan.expect.resident_layers == 0 {
        return Err("plan.expect.resident_layers must be > 0".into());
    }
    Ok(())
}

/// Identity BEFORE any load via the REAL `VerifiedCache` (recipe path only;
/// the untouched original replay path never calls this): the recipe's
/// verified-cache file digest must match, the cache home must equal the real
/// `hipfire_home()` root, and the actual model AND `<model>.ple-q4` sidecar
/// digest AND stat must hit the cache and equal the sealed manifest recipe
/// (model vs `manifest.model.sha256`/`artifact.sha256`; q4 via the per-entry
/// `verified_cache.entries[]` sha + stat). Reject missing/stale/wrong digest
/// with no hash fallback, no `remember()`, no `disabled()` substitution.
fn verify_verified_identity(
    manifest: &Manifest,
    artifact: &Artifact,
    sealed: &SealedExpect,
    model: &Path,
) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;
    let home = hipfire_config::hipfire_home().map_err(|e| format!("hipfire_home: {e}"))?;
    if Path::new(&sealed.cache_home) != home {
        return Err(format!(
            "execution_recipe verified_cache.home {:?} differs from the real hipfire home {}",
            sealed.cache_home,
            home.display()
        ));
    }
    let cache_bytes = std::fs::read(Path::new(&sealed.cache_path))
        .map_err(|e| format!("read verified cache {}: {e}", sealed.cache_path))?;
    if sha256_hex(&cache_bytes) != sealed.cache_sha256 {
        return Err("execution_recipe verified_cache.sha256 differs from the cache file".into());
    }
    let entries: Value = manifest
        .source
        .pointer("/execution_recipe/verified_cache/entries")
        .cloned()
        .ok_or_else(|| {
            "manifest.source.execution_recipe.verified_cache.entries is missing".to_string()
        })?;
    let entries = entries.as_array().ok_or_else(|| {
        "manifest.source.execution_recipe.verified_cache.entries must be an array".to_string()
    })?;
    let expected_for = |path: &Path| -> Result<String, String> {
        let want = path.display().to_string();
        for entry in entries {
            let map = entry
                .as_object()
                .ok_or_else(|| "verified_cache.entries holds a non-object entry".to_string())?;
            if map.get("path").and_then(Value::as_str) == Some(want.as_str()) {
                let sha = map
                    .get("sha256")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("verified_cache entry for {want} has no sha256"))?;
                if !is_sha256_hex(sha) {
                    return Err(format!(
                        "verified_cache entry for {want} is not 64 lowercase hex digits"
                    ));
                }
                let meta =
                    std::fs::metadata(path).map_err(|e| format!("stat {}: {e}", path.display()))?;
                let check = |field: &str, got: u64, want: u64| -> Result<(), String> {
                    if got != want {
                        return Err(format!("verified_cache entry for {want} has stale {field}"));
                    }
                    Ok(())
                };
                check(
                    "dev",
                    meta.dev(),
                    map.get("dev")
                        .and_then(Value::as_u64)
                        .ok_or_else(|| format!("verified_cache entry for {want} has no dev"))?,
                )?;
                check(
                    "ino",
                    meta.ino(),
                    map.get("ino")
                        .and_then(Value::as_u64)
                        .ok_or_else(|| format!("verified_cache entry for {want} has no ino"))?,
                )?;
                check(
                    "size",
                    meta.size(),
                    map.get("size")
                        .and_then(Value::as_u64)
                        .ok_or_else(|| format!("verified_cache entry for {want} has no size"))?,
                )?;
                let mtime_sec = map
                    .get("mtime_sec")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| format!("verified_cache entry for {want} has no mtime_sec"))?;
                let mtime_nsec = map
                    .get("mtime_nsec")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| format!("verified_cache entry for {want} has no mtime_nsec"))?;
                if meta.mtime() != mtime_sec || meta.mtime_nsec() != mtime_nsec {
                    return Err(format!("verified_cache entry for {want} has stale mtime"));
                }
                return Ok(sha.to_string());
            }
        }
        Err(format!("verified_cache.entries has no entry for {want}"))
    };
    let check_one = |path: &Path, expected: &str, what: &str| -> Result<(), String> {
        let meta = std::fs::metadata(path).map_err(|e| format!("stat {}: {e}", path.display()))?;
        let key = StatKey::of(&meta);
        let cache = VerifiedCache::in_home(&home);
        let digest = cache.verified_digest(path, &key).ok_or_else(|| {
            format!(
                "{what} {} has no verified-cache hit (missing, corrupt, stale or unrecorded): refusing load",
                path.display()
            )
        })?;
        if !digest.eq_ignore_ascii_case(expected) {
            return Err(format!(
                "verified {what} digest {digest} differs from the sealed manifest recipe"
            ));
        }
        Ok(())
    };
    check_one(model, &manifest.model.sha256, "model")?;
    if manifest.model.sha256 != artifact.sha256 {
        return Err("manifest.model.sha256 differs from plan.artifact.sha256".into());
    }
    let q4 = ple_sidecar_path(model, PleFormat::Q4);
    let q4_expected = expected_for(&q4)?;
    check_one(&q4, &q4_expected, "PLE sidecar")?;
    Ok(())
}

// ── numerics: row statistics, canonical bytes, KL ──────────────────────

/// Top-2 of one logits row. Ties resolve to the LOWER index (the greedy
/// argmax rule), so the second entry of an exact tie is the next-lowest index.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RowStats {
    top1_id: u32,
    top1: f32,
    top2_id: u32,
    top2: f32,
    /// `-inf` logits (zero probability mass).
    neg_inf: usize,
}

impl RowStats {
    fn margin(&self) -> f64 {
        f64::from(self.top1) - f64::from(self.top2)
    }
}

/// Statistics of a capturable row: NaN and `+inf` are refused, `-inf` is
/// counted, and at least two finite logits must exist (a finite margin).
fn row_stats(row: &[f32]) -> Result<RowStats, String> {
    let mut best: Option<(u32, f32)> = None;
    let mut second: Option<(u32, f32)> = None;
    let mut neg_inf = 0usize;
    for (i, &x) in row.iter().enumerate() {
        if x.is_nan() {
            return Err(format!("logit {i} is NaN"));
        }
        if x == f32::INFINITY {
            return Err(format!("logit {i} is +inf"));
        }
        if x == f32::NEG_INFINITY {
            neg_inf += 1;
            continue;
        }
        let candidate = (i as u32, x);
        match best {
            None => best = Some(candidate),
            Some((_, b)) if x > b => {
                second = best;
                best = Some(candidate);
            }
            Some(_) => match second {
                None => second = Some(candidate),
                Some((_, s)) if x > s => second = Some(candidate),
                Some(_) => {}
            },
        }
    }
    match (best, second) {
        (Some((top1_id, top1)), Some((top2_id, top2))) => Ok(RowStats {
            top1_id,
            top1,
            top2_id,
            top2,
            neg_inf,
        }),
        _ => Err(format!(
            "fewer than two finite logits in a row of {}",
            row.len()
        )),
    }
}

/// The canonical row bytes: each f32's IEEE-754 bits as 4 little-endian bytes,
/// on any host. `out` is reused.
fn encode_row_le(row: &[f32], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(row.len() * 4);
    for &x in row {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
}

fn decode_row_le(bytes: &[u8], out: &mut Vec<f32>) -> Result<(), String> {
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "row of {} bytes is not a whole number of f32",
            bytes.len()
        ));
    }
    out.clear();
    // The length guard above leaves no remainder.
    let (chunks, _) = bytes.as_chunks::<4>();
    out.extend(
        chunks
            .iter()
            .map(|c| f32::from_bits(u32::from_le_bytes(*c))),
    );
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
enum KlError {
    Empty,
    LengthMismatch {
        reference: usize,
        candidate: usize,
    },
    Nan {
        side: &'static str,
        index: usize,
    },
    PositiveInfinity {
        side: &'static str,
        index: usize,
    },
    /// Every reference logit is `-inf`: there is no distribution.
    ZeroSupportReference,
}

impl std::fmt::Display for KlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KlError::Empty => write!(f, "empty logits row"),
            KlError::LengthMismatch {
                reference,
                candidate,
            } => write!(f, "reference has {reference} logits, candidate {candidate}"),
            KlError::Nan { side, index } => write!(f, "{side} logit {index} is NaN"),
            KlError::PositiveInfinity { side, index } => {
                write!(f, "{side} logit {index} is +inf")
            }
            KlError::ZeroSupportReference => {
                write!(f, "reference has no finite logit (zero support)")
            }
        }
    }
}

/// Neumaier-compensated f64 sum.
#[derive(Default)]
struct Neumaier {
    sum: f64,
    compensation: f64,
}

impl Neumaier {
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.compensation += (self.sum - t) + x;
        } else {
            self.compensation += (x - t) + self.sum;
        }
        self.sum = t;
    }

    fn total(&self) -> f64 {
        self.sum + self.compensation
    }
}

/// `(max finite logit, ln Σ exp(x - max))` of a row, or `None` when every
/// logit is `-inf`. NaN and `+inf` are errors.
fn log_sum_exp_shifted(row: &[f32], side: &'static str) -> Result<Option<(f64, f64)>, KlError> {
    let mut max = f64::NEG_INFINITY;
    for (index, &x) in row.iter().enumerate() {
        if x.is_nan() {
            return Err(KlError::Nan { side, index });
        }
        if x == f32::INFINITY {
            return Err(KlError::PositiveInfinity { side, index });
        }
        if x != f32::NEG_INFINITY {
            max = max.max(f64::from(x));
        }
    }
    if max == f64::NEG_INFINITY {
        return Ok(None);
    }
    let mut sum = Neumaier::default();
    for &x in row {
        if x != f32::NEG_INFINITY {
            sum.add((f64::from(x) - max).exp());
        }
    }
    Ok(Some((max, sum.total().ln())))
}

/// `KL(reference || candidate)` in nats from two raw logits rows, in f64.
///
/// Both rows are max-shifted log-softmaxes; the sum runs over the reference's
/// support (a `-inf` reference logit is zero mass and contributes 0) with a
/// compensated accumulator, so an additive shift of either row changes the
/// result only by rounding. A candidate `-inf` where the reference has mass is
/// an infinite divergence: `Ok(f64::INFINITY)`. The value may be a few ulps
/// below zero for (near-)identical rows.
fn kl_divergence_f64(reference: &[f32], candidate: &[f32]) -> Result<f64, KlError> {
    if reference.is_empty() || candidate.is_empty() {
        return Err(KlError::Empty);
    }
    if reference.len() != candidate.len() {
        return Err(KlError::LengthMismatch {
            reference: reference.len(),
            candidate: candidate.len(),
        });
    }
    let (rmax, rln) =
        log_sum_exp_shifted(reference, "reference")?.ok_or(KlError::ZeroSupportReference)?;
    let Some((cmax, cln)) = log_sum_exp_shifted(candidate, "candidate")? else {
        return Ok(f64::INFINITY);
    };
    let mut total = Neumaier::default();
    for (&r, &c) in reference.iter().zip(candidate) {
        if r == f32::NEG_INFINITY {
            continue;
        }
        if c == f32::NEG_INFINITY {
            return Ok(f64::INFINITY);
        }
        let lp = (f64::from(r) - rmax) - rln;
        let lq = (f64::from(c) - cmax) - cln;
        total.add(lp.exp() * (lp - lq));
    }
    Ok(total.total())
}

// ── fresh output ───────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct OutputDir {
    root: PathBuf,
}

impl OutputDir {
    /// A directory that did not exist (it is created) or exists empty. Anything
    /// else (a file, a symlink, a non-empty directory) is refused.
    fn create_fresh(path: &Path) -> Result<Self, String> {
        if !path.is_absolute() {
            return Err(format!("output {} is not absolute", path.display()));
        }
        match std::fs::symlink_metadata(path) {
            Ok(meta) => {
                if !meta.is_dir() {
                    return Err(format!(
                        "refusing output {}: it exists and is not a directory",
                        path.display()
                    ));
                }
                let mut entries = std::fs::read_dir(path)
                    .map_err(|e| format!("read output {}: {e}", path.display()))?;
                if entries.next().is_some() {
                    return Err(format!(
                        "refusing output {}: the directory is not empty",
                        path.display()
                    ));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| format!("create {}: {e}", parent.display()))?;
                }
                std::fs::create_dir(path)
                    .map_err(|e| format!("create output {}: {e}", path.display()))?;
            }
            Err(e) => return Err(format!("stat output {}: {e}", path.display())),
        }
        Ok(Self {
            root: path.to_path_buf(),
        })
    }

    fn create_file(&self, name: &str) -> Result<std::fs::File, String> {
        if name.is_empty() || name.contains('/') || name == "." || name == ".." {
            return Err(format!("invalid output file name {name:?}"));
        }
        let path = self.root.join(name);
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| format!("create {}: {e}", path.display()))
    }

    fn write_new(&self, name: &str, bytes: &[u8]) -> Result<(), String> {
        let mut file = self.create_file(name)?;
        file.write_all(bytes)
            .and_then(|()| file.sync_all())
            .map_err(|e| format!("write {name}: {e}"))
    }
}

struct RowRecord {
    index: usize,
    input_token: Option<u32>,
    position: usize,
    sha256: String,
    stats: RowStats,
    offset: u64,
}

struct CellOutput {
    rows: usize,
    bin_name: String,
    bin_bytes: u64,
    rows_name: String,
    rows_sha256: String,
    stream_sha256: String,
}

/// One cell's streamed dump: rows are written, hashed and forgotten; only the
/// per-row metadata is kept (a few hundred small records at most).
struct CellWriter {
    dir: OutputDir,
    name: String,
    vocab: usize,
    bin: std::fs::File,
    bin_name: String,
    stream: Sha256,
    bytes: u64,
    buf: Vec<u8>,
    records: Vec<RowRecord>,
}

impl CellWriter {
    fn create(dir: &OutputDir, name: &str, vocab: usize) -> Result<Self, String> {
        let bin_name = format!("{name}.logits.f32le");
        let bin = dir.create_file(&bin_name)?;
        Ok(Self {
            dir: dir.clone(),
            name: name.to_string(),
            vocab,
            bin,
            bin_name,
            stream: Sha256::new(),
            bytes: 0,
            buf: Vec::with_capacity(vocab * 4),
            records: Vec::new(),
        })
    }

    /// Append one row: stats (NaN / `+inf` refused), canonical bytes, per-row
    /// and stream hash. `position` is the context length the row conditions on.
    fn push_row(
        &mut self,
        input_token: Option<u32>,
        position: usize,
        row: &[f32],
    ) -> Result<RowStats, String> {
        if row.len() != self.vocab {
            return Err(format!(
                "cell {}: row of {} logits, vocab is {}",
                self.name,
                row.len(),
                self.vocab
            ));
        }
        let stats = row_stats(row)
            .map_err(|e| format!("cell {} row {}: {e}", self.name, self.records.len()))?;
        encode_row_le(row, &mut self.buf);
        let sha256 = sha256_hex(&self.buf);
        self.bin
            .write_all(&self.buf)
            .map_err(|e| format!("write {}: {e}", self.bin_name))?;
        self.stream.update(&self.buf);
        self.records.push(RowRecord {
            index: self.records.len(),
            input_token,
            position,
            sha256,
            stats,
            offset: self.bytes,
        });
        self.bytes += self.buf.len() as u64;
        Ok(stats)
    }

    fn row_json(&self, record: &RowRecord, next_token: Option<u32>) -> Value {
        json!({
            "schema": ROW_SCHEMA,
            "cell": self.name,
            "row": record.index,
            "input_token": record.input_token,
            "context_position": record.position,
            "next_token": next_token,
            "logits_sha256": record.sha256,
            "bin_offset": record.offset,
            "top1_id": record.stats.top1_id,
            "top1_logit": record.stats.top1,
            "top1_logit_bits": record.stats.top1.to_bits(),
            "top2_id": record.stats.top2_id,
            "top2_logit": record.stats.top2,
            "top2_logit_bits": record.stats.top2.to_bits(),
            "top12_margin": record.stats.margin(),
            "neg_inf_count": record.stats.neg_inf,
        })
    }

    /// Close the dump and write the rows file. `next_tokens[i]` is the token
    /// that followed row `i` (one entry per row).
    fn finish(mut self, next_tokens: &[Option<u32>]) -> Result<CellOutput, String> {
        if next_tokens.len() != self.records.len() {
            return Err(format!(
                "cell {}: {} rows but {} next-token entries",
                self.name,
                self.records.len(),
                next_tokens.len()
            ));
        }
        self.bin
            .flush()
            .and_then(|()| self.bin.sync_all())
            .map_err(|e| format!("sync {}: {e}", self.bin_name))?;
        let mut text = String::new();
        for (record, next) in self.records.iter().zip(next_tokens) {
            text.push_str(&self.row_json(record, *next).to_string());
            text.push('\n');
        }
        let rows_name = format!("{}.rows.jsonl", self.name);
        self.dir.write_new(&rows_name, text.as_bytes())?;
        Ok(CellOutput {
            rows: self.records.len(),
            bin_name: self.bin_name,
            bin_bytes: self.bytes,
            rows_name,
            rows_sha256: sha256_hex(text.as_bytes()),
            stream_sha256: format!("{:x}", self.stream.finalize()),
        })
    }
}

// ── row sources: the teacher and greedy capture loops ──────────────────

/// What a capture loop drives. The GPU implementation forwards through the
/// bundle; the CPU tests script it.
trait RowSource {
    /// Read the current logits row (prefill's final row, or the last forward's).
    fn read_row(&mut self, out: &mut Vec<f32>) -> Result<(), String>;
    /// Forward one teacher token.
    fn forward_teacher(&mut self, token: u32) -> Result<(), String>;
    /// Forward `pending`, or (`None`) the device argmax of the row the last
    /// forward left; returns the token that ran.
    fn forward_greedy(&mut self, pending: Option<u32>) -> Result<u32, String>;
    /// Tokens in the model's context.
    fn position(&self) -> usize;
}

type OnRow<'a> = &'a mut dyn FnMut(Option<u32>, usize, &[f32]) -> Result<RowStats, String>;

fn expect_position<S: RowSource>(src: &S, want: usize, what: &str) -> Result<(), String> {
    let got = src.position();
    if got != want {
        return Err(format!("context position {got} {what}, expected {want}"));
    }
    Ok(())
}

/// Seed row, then one row after each `feed` token: `feed.len() + 1` rows, row
/// `i` conditioned on the prefill plus `feed[..i]`.
fn teacher_capture<S: RowSource>(
    src: &mut S,
    feed: &[u32],
    base: usize,
    row: &mut Vec<f32>,
    on_row: OnRow<'_>,
) -> Result<(), String> {
    expect_position(src, base, "after the prefill")?;
    src.read_row(row)?;
    on_row(None, base, row.as_slice())?;
    for (i, &token) in feed.iter().enumerate() {
        src.forward_teacher(token)?;
        let at = base + i + 1;
        expect_position(src, at, &format!("after teacher token {i}"))?;
        src.read_row(row)?;
        on_row(Some(token), at, row.as_slice())?;
    }
    Ok(())
}

/// The daemon's greedy loop with every row captured. The first token is the
/// host argmax of the seed row, every later one the device argmax of the
/// previous forward's row. A token is FORWARDED before the stop check, so the
/// row after the last token (an EOS included) is captured: `rows = ids + 1`.
/// Returns the ids and `"stop"` (the first terminator, included) or `"length"`.
fn greedy_capture<S: RowSource>(
    src: &mut S,
    max_new: usize,
    base: usize,
    is_stop: &dyn Fn(u32) -> bool,
    row: &mut Vec<f32>,
    on_row: OnRow<'_>,
) -> Result<(Vec<u32>, &'static str), String> {
    expect_position(src, base, "after the prefill")?;
    src.read_row(row)?;
    let seed = on_row(None, base, row.as_slice())?;
    let host = hipfire_runtime::llama::argmax(row.as_slice());
    if host != seed.top1_id {
        return Err(format!(
            "host argmax {host} differs from the seed row's top-1 {}",
            seed.top1_id
        ));
    }
    let mut last_top1 = seed.top1_id;
    let mut pending = Some(host);
    let mut ids = Vec::with_capacity(max_new);
    while ids.len() < max_new {
        let from_device = pending.is_none();
        let token = src.forward_greedy(pending.take())?;
        if from_device && token != last_top1 {
            return Err(format!(
                "device argmax token {token} differs from the previous row's top-1 {last_top1} (row {})",
                ids.len()
            ));
        }
        ids.push(token);
        let at = base + ids.len();
        expect_position(src, at, &format!("after generated token {}", ids.len() - 1))?;
        src.read_row(row)?;
        let stats = on_row(Some(token), at, row.as_slice())?;
        last_top1 = stats.top1_id;
        if is_stop(token) {
            return Ok((ids, "stop"));
        }
    }
    Ok((ids, "length"))
}

/// Consecutive `(start, end)` forward calls covering `0..len` in `chunk`-row
/// pieces (the baseline's manual prefill).
fn chunk_ranges(len: usize, chunk: usize) -> Vec<(usize, usize)> {
    let chunk = chunk.max(1);
    let mut ranges = Vec::with_capacity(len / chunk + 1);
    let mut start = 0;
    while start < len {
        let end = (start + chunk).min(len);
        ranges.push((start, end));
        start = end;
    }
    ranges
}

fn validate_ranges(ranges: &[(usize, usize)], len: usize) -> Result<(), String> {
    let mut at = 0;
    for &(start, end) in ranges {
        if start != at || end <= start {
            return Err(format!(
                "prefill range {start}..{end} does not continue at {at}"
            ));
        }
        at = end;
    }
    if at != len {
        return Err(format!("prefill ranges end at {at}, the prompt has {len}"));
    }
    Ok(())
}

// ── actual route / placement assertions ────────────────────────────────

fn mode_name(mode: SlowlinkMode) -> &'static str {
    match mode {
        SlowlinkMode::Off => "off",
        SlowlinkMode::Force => "force",
        SlowlinkMode::Auto => "auto",
    }
}

/// The selector the parent launched with: baseline is `0`, S1 anything else.
fn check_mode(arm: Arm, mode: SlowlinkMode) -> Result<(), String> {
    match (arm, mode) {
        (Arm::Baseline, SlowlinkMode::Off) => Ok(()),
        (Arm::S1, SlowlinkMode::Force | SlowlinkMode::Auto) => Ok(()),
        (arm, mode) => Err(format!(
            "the {} arm needs the S1 selector {}, but the process selector is {}",
            arm.name(),
            if arm == Arm::Baseline {
                "off (0)"
            } else {
                "on (1 or unset)"
            },
            mode_name(mode)
        )),
    }
}

/// The environment a route must carry: `expect.env` plus, on the baseline
/// routes, `expect.baseline_env` (explicit N9, 4096 scratch), with the one
/// difference of the f32 diagnostic phases applied: they run
/// `HIPFIRE_STATE_QUANT=f32` with the KV mode still q8.
fn effective_env(expect: &Expect, arm: Arm, state_f32: bool) -> BTreeMap<String, String> {
    let mut env = expect.env.clone();
    if arm == Arm::Baseline {
        env.extend(expect.baseline_env.clone());
    }
    if state_f32 {
        env.insert(STATE_QUANT_ENV.to_string(), "f32".to_string());
    }
    env
}

/// The effective environment of the live process (`effective_env`; plus, on
/// the S1 routes only, `expect.env_absent` must be unset), through `lookup`.
/// Returns the observed values for the record.
fn check_env(
    expect: &Expect,
    arm: Arm,
    state_f32: bool,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<Value, String> {
    let mut problems = Vec::new();
    let mut observed = serde_json::Map::new();
    for (name, want) in &effective_env(expect, arm, state_f32) {
        let got = lookup(name);
        if got.as_deref() != Some(want.as_str()) {
            problems.push(format!("{name} is {got:?}, expected {want:?}"));
        }
        observed.insert(name.clone(), json!(got));
    }
    let absent: &[String] = if arm == Arm::S1 {
        &expect.env_absent
    } else {
        &[]
    };
    for name in absent {
        let got = lookup(name);
        if got.is_some() {
            problems.push(format!("{name} is set ({got:?}) but must be absent"));
        }
        observed.insert(name.clone(), json!(got));
    }
    if problems.is_empty() {
        Ok(Value::Object(observed))
    } else {
        Err(format!(
            "launch environment mismatch: {}",
            problems.join("; ")
        ))
    }
}

/// What the loaded bundle actually is.
#[derive(Debug, Clone)]
struct LoadFacts {
    mode: SlowlinkMode,
    placement: String,
    slowlink_profile: bool,
    resident_layers: usize,
    host_mapped_tensors: usize,
    /// The forward's scratch rows (`Qwen4Bundle::spec_chunk_rows`).
    scratch_rows: Option<usize>,
    profile_tile_rows: Option<usize>,
    profile_resident_layers: Option<usize>,
    profile_is_product: Option<bool>,
    ple_format: String,
    /// The loaded request state's GDN recurrent / QSA K-V storage
    /// (`Qwen4Bundle::state_format`).
    gdn_format: GdnStateFormat,
    qsa_format: QsaKvFormat,
}

fn validate_load(
    arm: Arm,
    state_f32: bool,
    facts: &LoadFacts,
    expect: &Expect,
    sealed: Option<&SealedExpect>,
) -> Result<(), String> {
    let mut problems = Vec::new();
    if let Err(e) = check_mode(arm, facts.mode) {
        problems.push(e);
    }
    if facts.resident_layers != expect.resident_layers {
        problems.push(format!(
            "{} resident layers, expected {}",
            facts.resident_layers, expect.resident_layers
        ));
    }
    if let Some(sealed) = sealed {
        if facts.resident_layers != sealed.resident_layers {
            problems.push(format!(
                "{} resident layers, the sealed recipe requires {}",
                facts.resident_layers, sealed.resident_layers
            ));
        }
        if facts.placement != sealed.placement {
            problems.push(format!(
                "placement {:?}, the sealed recipe requires {:?}",
                facts.placement, sealed.placement
            ));
        }
        if matches!(
            hipfire_config::developer_var(S1_RESIDENT_LAYERS_ENV).ok().as_deref(),
            Some(raw) if resident_layer_switch_is_override(Some(raw))
        ) {
            problems.push(format!(
                "{S1_RESIDENT_LAYERS_ENV} overrides the resident set: the sealed recipe's explicit set is the only authority"
            ));
        }
    }
    if facts.host_mapped_tensors == 0 {
        problems
            .push("no host-mapped tensors: the placement is not the host-link placement".into());
    }
    if facts.ple_format != expect.ple_format {
        problems.push(format!(
            "PLE rows {:?}, expected {:?}",
            facts.ple_format, expect.ple_format
        ));
    }
    let want_gdn = if state_f32 {
        GdnStateFormat::F32
    } else {
        GdnStateFormat::Q8
    };
    if facts.gdn_format != want_gdn || facts.qsa_format != QsaKvFormat::Q8 {
        problems.push(format!(
            "request state is GDN {:?} / QSA K-V {:?}, expected GDN {want_gdn:?} / QSA K-V Q8",
            facts.gdn_format, facts.qsa_format
        ));
    }
    let want_scratch = match arm {
        Arm::Baseline => expect.baseline_tile_rows,
        Arm::S1 => expect.s1_tile_rows,
    };
    if facts.scratch_rows != Some(want_scratch) {
        problems.push(format!(
            "forward scratch is {:?} rows, expected {want_scratch}",
            facts.scratch_rows
        ));
    }
    match arm {
        Arm::Baseline => {
            if facts.placement != "explicit-env" {
                problems.push(format!(
                    "placement {:?}, expected \"explicit-env\" (the explicit layer-count override)",
                    facts.placement
                ));
            }
            if facts.slowlink_profile || facts.profile_tile_rows.is_some() {
                problems.push("an S1 profile is attached to the baseline load".into());
            }
        }
        Arm::S1 => {
            if facts.placement != "slowlink-profile" {
                problems.push(format!(
                    "placement {:?}, expected \"slowlink-profile\"",
                    facts.placement
                ));
            }
            if !facts.slowlink_profile {
                problems.push("no S1 profile was admitted".into());
            }
            if facts.profile_tile_rows != Some(expect.s1_tile_rows) {
                problems.push(format!(
                    "profile tile {:?}, expected {}",
                    facts.profile_tile_rows, expect.s1_tile_rows
                ));
            }
            if facts.profile_resident_layers != Some(facts.resident_layers) {
                problems.push(format!(
                    "profile resident layers {:?} differ from the loaded {}",
                    facts.profile_resident_layers, facts.resident_layers
                ));
            }
            if facts.profile_is_product != Some(true) {
                problems.push("the S1 profile was not admitted for the product purpose".into());
            }
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "actual load does not match the plan: {}",
            problems.join("; ")
        ))
    }
}

fn receipt_u64(receipt: &Value, key: &str) -> Result<u64, String> {
    receipt
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("receipt field {key:?} is missing or not an unsigned integer"))
}

fn receipt_str<'a>(receipt: &'a Value, key: &str) -> Result<&'a str, String> {
    receipt
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("receipt field {key:?} is missing or not a string"))
}

/// One prefill call's actual receipt against the arm: route, selector label,
/// entry position, rows and tile. Returns the checked fields for the record.
/// With a sealed recipe the layer-major count AND set MUST equal the recipe's
/// explicit planned placement (null set = front `0..count`); without one the
/// literal original N9 checks stand untouched.
fn validate_receipt(
    arm: Arm,
    mode: SlowlinkMode,
    expect: &Expect,
    receipt: &Value,
    entry: usize,
    rows: usize,
    sealed: Option<&SealedExpect>,
) -> Result<Value, String> {
    let route = receipt_str(receipt, "route")?;
    let selected_by = receipt_str(receipt, "selected_by")?;
    let entry_pos = receipt_u64(receipt, "entry_pos")? as usize;
    let prompt_rows = receipt_u64(receipt, "prompt_rows")? as usize;
    let tile_rows = receipt_u64(receipt, "tile_rows")? as usize;
    let retained = receipt_u64(receipt, "retained_scratch_rows")? as usize;
    let mut problems = Vec::new();
    if entry_pos != entry {
        problems.push(format!("entry_pos {entry_pos}, expected {entry}"));
    }
    if prompt_rows != rows {
        problems.push(format!("prompt_rows {prompt_rows}, expected {rows}"));
    }
    let want_tile = match arm {
        Arm::Baseline => expect.baseline_tile_rows,
        Arm::S1 => expect.s1_tile_rows,
    };
    if tile_rows != want_tile || retained != want_tile {
        problems.push(format!(
            "tile_rows {tile_rows} / retained_scratch_rows {retained}, expected {want_tile}"
        ));
    }
    match arm {
        Arm::Baseline => {
            if route != "incumbent" {
                problems.push(format!("route {route:?}, expected \"incumbent\""));
            }
            if selected_by != "env0" {
                problems.push(format!("selected_by {selected_by:?}, expected \"env0\""));
            }
        }
        Arm::S1 => {
            let want_selected = match mode {
                SlowlinkMode::Force => "env1",
                SlowlinkMode::Auto => "auto",
                SlowlinkMode::Off => "(selector off)",
            };
            if selected_by != want_selected {
                problems.push(format!(
                    "selected_by {selected_by:?}, expected {want_selected:?}"
                ));
            }
            match route {
                "layer-major" => {
                    let status = receipt.get("status").and_then(Value::as_str);
                    if status != Some("ok") {
                        problems.push(format!("layer-major status {status:?}, expected \"ok\""));
                    }
                    match receipt_u64(receipt, "resident_layers") {
                        Ok(layers) if layers as usize == expect.resident_layers => {}
                        Ok(layers) => problems.push(format!(
                            "layer-major resident_layers {layers}, expected {}",
                            expect.resident_layers
                        )),
                        Err(e) => problems.push(e),
                    }
                    if let Some(sealed) = sealed {
                        match receipt.get("resident_layer_set") {
                            Some(Value::Array(items)) => {
                                let mut got = Vec::with_capacity(items.len());
                                let mut malformed = false;
                                for item in items {
                                    match item.as_u64() {
                                        Some(layer) => got.push(layer as usize),
                                        None => {
                                            malformed = true;
                                            break;
                                        }
                                    }
                                }
                                if malformed {
                                    problems.push(
                                        "layer-major resident_layer_set holds a non-integer".into(),
                                    );
                                } else {
                                    let want: Vec<usize> = match &sealed.resident_layer_set {
                                        Some(set) => set.clone(),
                                        None => (0..sealed.resident_layers).collect(),
                                    };
                                    if got != want {
                                        problems.push(format!(
                                            "layer-major resident_layer_set {got:?} differs from the sealed recipe {want:?}"
                                        ));
                                    }
                                    if got.len() != sealed.resident_layers {
                                        problems.push(format!(
                                            "layer-major resident_layer_set has {} entries, the sealed recipe requires {}",
                                            got.len(),
                                            sealed.resident_layers
                                        ));
                                    }
                                }
                            }
                            _ => problems
                                .push("layer-major receipt has no resident_layer_set array".into()),
                        }
                    }
                }
                "incumbent" => {
                    if rows >= expect.s1_layer_major_min_rows {
                        problems.push(format!(
                            "a {rows}-row call (>= {}) took the incumbent route: bypass {}",
                            expect.s1_layer_major_min_rows,
                            receipt.get("bypass_reason").unwrap_or(&Value::Null)
                        ));
                    }
                }
                other => problems.push(format!(
                    "route {other:?} is neither layer-major nor incumbent"
                )),
            }
        }
    }
    if !problems.is_empty() {
        return Err(format!(
            "slowlink receipt check failed: {}",
            problems.join("; ")
        ));
    }
    Ok(json!({
        "route": route,
        "selected_by": selected_by,
        "entry_pos": entry_pos,
        "prompt_rows": prompt_rows,
        "tile_rows": tile_rows,
        "retained_scratch_rows": retained,
        "resident_layers": receipt.get("resident_layers").cloned().unwrap_or(Value::Null),
        "host_layers": receipt.get("host_layers").cloned().unwrap_or(Value::Null),
        "sweeps": receipt.get("sweeps").cloned().unwrap_or(Value::Null),
        "status": receipt.get("status").cloned().unwrap_or(Value::Null),
        "bypass_reason": receipt.get("bypass_reason").cloned().unwrap_or(Value::Null),
    }))
}

// ── GPU context ────────────────────────────────────────────────────────

struct Ctx {
    gpu: Gpu,
    bundle: Qwen4Bundle,
    /// The ONE reused vocab-row device allocation: every prefill's final row
    /// and every forward's row lands here.
    logits: GpuTensor,
    vocab: usize,
    eos: u32,
    tokenizer: Tokenizer,
    template: Option<String>,
}

/// How a prompt is prefilled.
#[derive(Clone, Debug, PartialEq, Eq)]
enum PrefillPlan {
    /// `prefill_final(prompt, 0)`: the daemon's AR route from a fresh state.
    Production,
    /// `reset`, then consecutive `forward_chunk_final` calls over these ranges.
    Direct(Vec<(usize, usize)>),
}

impl PrefillPlan {
    fn label(&self) -> &'static str {
        match self {
            PrefillPlan::Production => "prefill_final",
            PrefillPlan::Direct(ranges) if ranges.len() == 1 => "direct-single-call",
            PrefillPlan::Direct(_) => "direct-consecutive-calls",
        }
    }
}

struct PrefillResult {
    calls: Vec<(usize, usize)>,
    receipts: Vec<Value>,
    checks: Vec<Value>,
    seconds: f64,
    /// LAB-only actual witness (`lab` only). `None` = absent/non-pass (no
    /// grouped launch observed) with ordinary success untouched.
    #[cfg(feature = "lab")]
    fn_witness: Option<Value>,
    #[cfg(not(feature = "lab"))]
    fn_witness: (),
}

impl Ctx {
    fn prefill(
        &mut self,
        prompt: &[u32],
        how: &PrefillPlan,
        arm: Arm,
        mode: SlowlinkMode,
        expect: &Expect,
        sealed: Option<&SealedExpect>,
    ) -> Result<PrefillResult, String> {
        if prompt.is_empty() {
            return Err("empty prompt".into());
        }
        let calls: Vec<(usize, usize)> = match how {
            PrefillPlan::Production => vec![(0, prompt.len())],
            PrefillPlan::Direct(ranges) => {
                validate_ranges(ranges, prompt.len())?;
                ranges.clone()
            }
        };
        #[cfg(feature = "lab")]
        self.gpu.lab_fn_begin();
        match Self::prefill_inner(self, prompt, &calls, how, arm, mode, expect, sealed) {
            Ok((receipts, checks, seconds)) => Ok(PrefillResult {
                calls,
                receipts,
                checks,
                seconds,
                #[cfg(feature = "lab")]
                fn_witness: Self::lab_fn_export(&mut self.gpu),
                #[cfg(not(feature = "lab"))]
                fn_witness: (),
            }),
            Err(error) => {
                #[cfg(feature = "lab")]
                self.gpu.lab_fn_reset();
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_inner(
        ctx: &mut Ctx,
        prompt: &[u32],
        calls: &[(usize, usize)],
        how: &PrefillPlan,
        arm: Arm,
        mode: SlowlinkMode,
        expect: &Expect,
        sealed: Option<&SealedExpect>,
    ) -> Result<(Vec<Value>, Vec<Value>, f64), String> {
        let started = Instant::now();
        let mut receipts = Vec::with_capacity(calls.len());
        let mut checks = Vec::with_capacity(calls.len());
        match how {
            PrefillPlan::Production => {
                ctx.bundle
                    .prefill_final(&mut ctx.gpu, prompt, 0, &ctx.logits)
                    .map_err(|e| format!("prefill: {e}"))?;
                ctx.record_receipt(
                    0,
                    &calls[0],
                    arm,
                    mode,
                    expect,
                    sealed,
                    &mut receipts,
                    &mut checks,
                )?;
            }
            PrefillPlan::Direct(_) => {
                ctx.bundle
                    .reset(&mut ctx.gpu)
                    .map_err(|e| format!("reset: {e}"))?;
                for (call, &(start, end)) in calls.iter().enumerate() {
                    ctx.bundle
                        .forward_chunk_final(&mut ctx.gpu, &prompt[start..end], &ctx.logits, None)
                        .map_err(|e| format!("direct prefill {start}..{end}: {e}"))?;
                    ctx.record_receipt(
                        call,
                        &(start, end),
                        arm,
                        mode,
                        expect,
                        sealed,
                        &mut receipts,
                        &mut checks,
                    )?;
                }
            }
        }
        ctx.gpu
            .hip
            .device_synchronize()
            .map_err(|e| format!("device synchronize: {e}"))?;
        let seconds = started.elapsed().as_secs_f64();
        if ctx.bundle.state.position != prompt.len() {
            return Err(format!(
                "after the prefill the context holds {} tokens, expected {}",
                ctx.bundle.state.position,
                prompt.len()
            ));
        }
        Ok((receipts, checks, seconds))
    }

    #[allow(clippy::too_many_arguments)]
    fn record_receipt(
        &self,
        call: usize,
        range: &(usize, usize),
        arm: Arm,
        mode: SlowlinkMode,
        expect: &Expect,
        sealed: Option<&SealedExpect>,
        receipts: &mut Vec<Value>,
        checks: &mut Vec<Value>,
    ) -> Result<(), String> {
        let receipt = self
            .bundle
            .slowlink_receipt()
            .ok_or_else(|| format!("no slowlink receipt for prefill call {call}"))?;
        let check = validate_receipt(
            arm,
            mode,
            expect,
            &receipt,
            range.0,
            range.1 - range.0,
            sealed,
        )
        .map_err(|e| format!("prefill call {call} ({}..{}): {e}", range.0, range.1))?;
        receipts.push(receipt);
        checks.push(check);
        Ok(())
    }

    /// LAB-only actual witness export: drain the `Gpu` window and map owned
    /// records to JSON. `None` = absent/non-pass with ordinary success
    /// untouched. Full geometry uses the actual existing identity methods
    /// (module/container/device resolved at the launch boundary).
    #[cfg(feature = "lab")]
    fn lab_fn_export(gpu: &mut Gpu) -> Option<Value> {
        let (records, overflow) = gpu.lab_fn_end()?;
        let launches: Vec<Value> = records
            .into_iter()
            .map(|record| {
                json!({
                    "symbol": record.symbol,
                    "module": record.module,
                    "container_sha256": record.container_sha256,
                    "device_sha256": record.device_sha256,
                    "grid": record.grid,
                    "block": record.block,
                    "down": record.down,
                    "width": record.width,
                    "grouped_rows": record.grouped_rows,
                    "n_experts": record.n_experts,
                    "total_slots": record.total_slots,
                    "source_rows": record.source_rows,
                    "host_mapped": record.host_mapped,
                    "bf16_round_trip": record.bf16_round_trip,
                    "chunk_rows": record.chunk_rows,
                })
            })
            .collect();
        Some(json!({
            "schema": "hipfire.qwen4.lab_fn_witness.v1",
            "overflowed": overflow.0,
            "complete": overflow.0 == 0,
            "launches": launches,
        }))
    }
}

/// The bundle behind [`RowSource`]; disjoint borrows so the stop predicate can
/// hold the tokenizer meanwhile.
struct GpuSource<'a> {
    gpu: &'a mut Gpu,
    bundle: &'a mut Qwen4Bundle,
    logits: &'a GpuTensor,
}

impl RowSource for GpuSource<'_> {
    fn read_row(&mut self, out: &mut Vec<f32>) -> Result<(), String> {
        self.gpu
            .download_f32_into(self.logits, out)
            .map_err(|e| format!("download logits row: {e}"))
    }

    fn forward_teacher(&mut self, token: u32) -> Result<(), String> {
        self.bundle
            .forward_token(self.gpu, token, self.logits, None)
            .map_err(|e| format!("teacher forward: {e}"))
    }

    fn forward_greedy(&mut self, pending: Option<u32>) -> Result<u32, String> {
        self.bundle
            .forward_token_or_argmax(self.gpu, pending, self.logits)
            .map_err(|e| format!("decode: {e}"))
    }

    fn position(&self) -> usize {
        self.bundle.state.position
    }
}

/// Load the artifact the product way, assert what actually loaded, and
/// allocate the single reused logits row.
fn load_ctx(
    model: &Path,
    arm: Arm,
    state_f32: bool,
    plan: &Plan,
    sealed: Option<&SealedExpect>,
) -> Result<(Ctx, Value), String> {
    let mode = SlowlinkMode::from_env();
    check_mode(arm, mode)?;
    hipfire_loader::prepare_host_memory_for(model);
    let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
    let loaded = slowlink_load_bundle(
        &mut gpu,
        model,
        plan.max_seq,
        None,
        AdmissionPurpose::Product,
    )?;
    let profile = loaded.bundle.slowlink_profile();
    let facts = LoadFacts {
        mode,
        placement: loaded.placement.to_string(),
        slowlink_profile: loaded.slowlink_profile,
        resident_layers: loaded.resident_layers,
        host_mapped_tensors: loaded.host_mapped_tensors,
        scratch_rows: loaded.bundle.spec_chunk_rows(),
        profile_tile_rows: profile.map(|p| p.tile_rows),
        profile_resident_layers: profile.map(|p| p.resident_layers),
        profile_is_product: profile.map(|p| p.purpose == AdmissionPurpose::Product),
        ple_format: loaded.ple_format.clone(),
        gdn_format: loaded.bundle.state_format().gdn,
        qsa_format: loaded.bundle.state_format().qsa,
    };
    validate_load(arm, state_f32, &facts, &plan.expect, sealed)?;
    plan.check_vocab(loaded.vocab)?;
    let profile_json = match profile {
        Some(p) => json!({
            "tile_rows": p.tile_rows,
            "window_rows": p.window_rows,
            "resident_layers": p.resident_layers,
            "baseline_layers": p.baseline_layers,
            "checkpoint_stride_rows": p.checkpoint_stride_rows,
            "link_gbps": p.link_gbps,
            "device_snapshot_budget_bytes": p.device_snapshot_budget_bytes,
            "purpose": "product",
        }),
        None => Value::Null,
    };
    let facts_json = json!({
        "selector": mode_name(mode),
        "placement": facts.placement,
        "slowlink_profile": facts.slowlink_profile,
        "resident_layers": facts.resident_layers,
        "host_mapped_tensors": facts.host_mapped_tensors,
        "scratch_rows": facts.scratch_rows,
        "ple_format": facts.ple_format,
        "state_format": {
            "gdn": format!("{:?}", facts.gdn_format),
            "qsa_kv": format!("{:?}", facts.qsa_format),
        },
        "vocab": loaded.vocab,
        "eos_token": loaded.eos_token,
        "num_layers": loaded.num_layers,
        "arch_id": loaded.arch_id,
        "gpu_arch": gpu.arch,
        "profile": profile_json,
    });
    let logits = gpu
        .zeros(&[loaded.vocab], DType::F32)
        .map_err(|e| e.to_string())?;
    eprintln!(
        "qwen4_slowlink_quality: loaded {} on {} ({}): placement {} resident layers {} scratch {:?} vocab {}",
        model.display(),
        gpu.arch,
        arm.name(),
        facts.placement,
        facts.resident_layers,
        facts.scratch_rows,
        loaded.vocab
    );
    Ok((
        Ctx {
            gpu,
            bundle: loaded.bundle,
            logits,
            vocab: loaded.vocab,
            eos: loaded.eos_token,
            tokenizer: loaded.tokenizer,
            template: loaded.chat_template,
        },
        facts_json,
    ))
}

// ── determinism prompts ────────────────────────────────────────────────

struct ResolvedCase {
    id: String,
    ids: Vec<u32>,
    resume_at: Option<usize>,
    retained: Option<Vec<u32>>,
}

/// Whether a chat template reads the render-time date (the wall clock).
fn template_reads_date(template: &str) -> bool {
    template.contains("current_date") || template.contains("strftime_now")
}

/// The ids of every determinism case: `prompt_ids` verbatim, `messages`
/// rendered the way the daemon's AR route renders them (and the greedy-ids
/// driver does), then checked against `expected_prompt_tokens`.
fn resolve_det_cases(plan: &Plan, ctx: &Ctx) -> Result<Vec<ResolvedCase>, String> {
    let mut resolved = Vec::with_capacity(plan.determinism.cases.len());
    for case in &plan.determinism.cases {
        let ids = match (&case.prompt_ids, &case.messages) {
            (Some(ids), None) => ids.clone(),
            (None, Some(messages)) => {
                let template = ctx.template.as_deref().ok_or_else(|| {
                    format!("case {}: the artifact has no chat_template", case.id)
                })?;
                if template_reads_date(template)
                    && hipfire_config::developer_var("HIPFIRE_CHAT_CURRENT_DATE")
                        .map_or(true, |date| date.trim().is_empty())
                {
                    return Err(format!(
                        "case {}: the chat_template reads the current date; pin HIPFIRE_CHAT_CURRENT_DATE=YYYY-MM-DD",
                        case.id
                    ));
                }
                let mut messages = messages.clone();
                for entry in &mut messages {
                    if !entry.content.is_empty() {
                        if let Cow::Owned(normalized) = maybe_normalize_prompt(&entry.content) {
                            entry.content = normalized;
                        }
                    }
                }
                let tools = case.tools.as_deref().filter(|tools| !tools.is_empty());
                let frame = JinjaChatFrame {
                    tokenizer: &ctx.tokenizer,
                    template,
                    system: None,
                    user: "",
                    enable_thinking: plan.determinism.thinking,
                    bos_token: None,
                    reasoning_strength: None,
                    reasoning_effort: None,
                };
                let rendered = frame
                    .render_messages(&messages, tools, None)
                    .map_err(|e| format!("case {}: qwen4 Jinja render failed: {e}", case.id))?;
                ctx.tokenizer.encode(&rendered)
            }
            _ => {
                return Err(format!(
                    "case {}: neither or both of prompt_ids / messages",
                    case.id
                ))
            }
        };
        if ids.len() != case.expected_prompt_tokens {
            return Err(format!(
                "case {}: the prompt is {} tokens, the plan expects {}",
                case.id,
                ids.len(),
                case.expected_prompt_tokens
            ));
        }
        if let Some(bad) = ids.iter().find(|&&id| id as usize >= ctx.vocab) {
            return Err(format!(
                "case {}: prompt id {bad} is outside the vocabulary ({})",
                case.id, ctx.vocab
            ));
        }
        let required = ids.len() + plan.determinism.max_new_tokens + 1;
        if required > plan.max_seq {
            return Err(format!(
                "case {}: prompt {} + max_new {} + trailer 1 exceeds max_seq {}",
                case.id,
                ids.len(),
                plan.determinism.max_new_tokens,
                plan.max_seq
            ));
        }
        resolved.push(ResolvedCase {
            id: case.id.clone(),
            ids,
            resume_at: case.resume_at,
            retained: case.retained_ids.clone(),
        });
    }
    Ok(resolved)
}

// ── phase runner ───────────────────────────────────────────────────────

enum Body<'a> {
    /// Feed `feed`, row per step; `next_tokens` has one entry per row.
    Teacher {
        feed: &'a [u32],
        next_tokens: Vec<Option<u32>>,
        teacher_sha256: String,
    },
    Greedy {
        max_new: usize,
    },
}

struct CellDone {
    ids: Vec<u32>,
    finish: Option<&'static str>,
    receipts: Vec<Value>,
    prompt_tokens: usize,
}

struct Run<'a> {
    plan: &'a Plan,
    sealed: Option<&'a SealedExpect>,
    phase: Phase,
    arm: Arm,
    mode: SlowlinkMode,
    ctx: Ctx,
    dir: OutputDir,
    cells_file: std::fs::File,
    ids_file: Option<std::fs::File>,
    receipts_file: Option<std::fs::File>,
    row_buf: Vec<f32>,
    done: Vec<Value>,
    names: Vec<String>,
}

fn append_line(file: &mut std::fs::File, what: &str, line: &str) -> Result<(), String> {
    writeln!(file, "{line}")
        .and_then(|()| file.flush())
        .and_then(|()| file.sync_data())
        .map_err(|e| format!("write {what}: {e}"))
}

impl Run<'_> {
    fn direct_plan(&self, len: usize) -> PrefillPlan {
        match self.arm {
            Arm::Baseline => {
                PrefillPlan::Direct(chunk_ranges(len, self.plan.expect.baseline_tile_rows))
            }
            Arm::S1 => PrefillPlan::Direct(vec![(0, len)]),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_cell(
        &mut self,
        name: &str,
        kind: &'static str,
        context: Option<&str>,
        prompt: &[u32],
        how: PrefillPlan,
        body: Body<'_>,
        extra: Value,
    ) -> Result<CellDone, String> {
        let continuation = match &body {
            Body::Teacher { feed, .. } => feed.len(),
            Body::Greedy { max_new } => *max_new,
        };
        if prompt.len() + continuation + 1 > self.plan.max_seq {
            return Err(format!(
                "cell {name}: prompt {} + continuation {continuation} + trailer 1 exceeds max_seq {}",
                prompt.len(),
                self.plan.max_seq
            ));
        }
        let cell_started = Instant::now();
        let pre = self
            .ctx
            .prefill(
                prompt,
                &how,
                self.arm,
                self.mode,
                &self.plan.expect,
                self.sealed,
            )
            .map_err(|e| format!("cell {name}: {e}"))?;
        let base = prompt.len();
        let mut writer = CellWriter::create(&self.dir, name, self.ctx.vocab)?;
        let rows_started = Instant::now();
        let mut ids: Vec<u32> = Vec::new();
        let mut finish: Option<&'static str> = None;
        let final_position;
        {
            let c = &mut self.ctx;
            let eos = c.eos;
            let tokenizer = &c.tokenizer;
            let mut src = GpuSource {
                gpu: &mut c.gpu,
                bundle: &mut c.bundle,
                logits: &c.logits,
            };
            let mut on_row = |input: Option<u32>, position: usize, row: &[f32]| {
                writer.push_row(input, position, row)
            };
            match &body {
                Body::Teacher { feed, .. } => {
                    teacher_capture(&mut src, feed, base, &mut self.row_buf, &mut on_row)
                        .map_err(|e| format!("cell {name}: {e}"))?;
                }
                Body::Greedy { max_new } => {
                    let stop = |token: u32| token == eos || tokenizer.is_terminator(token);
                    let (generated, why) = greedy_capture(
                        &mut src,
                        *max_new,
                        base,
                        &stop,
                        &mut self.row_buf,
                        &mut on_row,
                    )
                    .map_err(|e| format!("cell {name}: {e}"))?;
                    ids = generated;
                    finish = Some(why);
                }
            }
            final_position = src.position();
        }
        let rows_seconds = rows_started.elapsed().as_secs_f64();
        let (next_tokens, expected_rows) = match &body {
            Body::Teacher {
                feed, next_tokens, ..
            } => (next_tokens.clone(), feed.len() + 1),
            Body::Greedy { .. } => (
                ids.iter()
                    .map(|&t| Some(t))
                    .chain(std::iter::once(None))
                    .collect(),
                ids.len() + 1,
            ),
        };
        let out = writer.finish(&next_tokens)?;
        if out.rows != expected_rows {
            return Err(format!(
                "cell {name}: {} rows written, expected {expected_rows}",
                out.rows
            ));
        }
        let mut cell = json!({
            "schema": CELL_SCHEMA,
            "phase": self.phase.name(),
            "arm": self.arm.name(),
            "cell": name,
            "kind": kind,
            "context": context,
            "prompt_tokens": base,
            "prompt_sha256": sha_u32(prompt),
            "prefill": {
                "structure": how.label(),
                "calls": pre.calls.iter().map(|&(s, e)| json!({"start": s, "end": e, "rows": e - s})).collect::<Vec<_>>(),
                "receipts": pre.receipts,
                "receipt_checks": pre.checks,
                "seconds": pre.seconds,
            },
            "rows": out.rows,
            "vocab": self.ctx.vocab,
            "bin": out.bin_name,
            "bin_bytes": out.bin_bytes,
            "rows_file": out.rows_name,
            "rows_file_sha256": out.rows_sha256,
            "stream_sha256": out.stream_sha256,
            "rows_seconds": rows_seconds,
            "cell_seconds": cell_started.elapsed().as_secs_f64(),
            "final_position": final_position,
        });
        if let Value::Object(map) = &mut cell {
            if let Body::Teacher {
                feed,
                teacher_sha256,
                ..
            } = &body
            {
                map.insert("teacher_feed_tokens".into(), json!(feed.len()));
                map.insert("teacher_sha256".into(), json!(teacher_sha256));
            }
            if let Some(why) = finish {
                map.insert("token_ids".into(), json!(ids));
                map.insert("token_ids_sha256".into(), json!(sha_u32(&ids)));
                map.insert("finish_reason".into(), json!(why));
                map.insert("text".into(), json!(self.ctx.tokenizer.decode(&ids)));
            }
            if let Value::Object(extra) = extra {
                for (key, value) in extra {
                    map.insert(key, value);
                }
            }
            // A case that carries earlier-retained ids records the comparison
            // (reported, never enforced here).
            if let Some(retained) = map.get("retained_ids").filter(|v| v.is_array()).cloned() {
                map.insert("retained_ids_match".into(), json!(retained == json!(ids)));
            }
            #[cfg(feature = "lab")]
            map.insert("fn_witness".into(), pre.fn_witness.unwrap_or(Value::Null));
        }
        append_line(&mut self.cells_file, "cells.jsonl", &cell.to_string())?;
        self.done.push(json!({
            "cell": name,
            "rows": out.rows,
            "stream_sha256": out.stream_sha256,
            "prompt_tokens": base,
            "finish_reason": finish,
        }));
        self.names.push(name.to_string());
        eprintln!(
            "qwen4_slowlink_quality: {} cell {name}: prompt {base}, {} rows, prefill {:.2}s, rows {rows_seconds:.2}s",
            self.phase.name(),
            out.rows,
            pre.seconds
        );
        Ok(CellDone {
            ids,
            finish,
            receipts: pre.receipts,
            prompt_tokens: base,
        })
    }

    fn run_det_case(&mut self, case: &ResolvedCase) -> Result<(), String> {
        let how = match case.resume_at {
            None => PrefillPlan::Production,
            Some(at) => PrefillPlan::Direct(vec![(0, at), (at, case.ids.len())]),
        };
        let name = det_cell(&case.id);
        let done = self.run_cell(
            &name,
            "determinism",
            None,
            &case.ids,
            how,
            Body::Greedy {
                max_new: self.plan.determinism.max_new_tokens,
            },
            json!({
                "case_id": case.id,
                "resume_at": case.resume_at,
                "retained_ids": case.retained,
            }),
        )?;
        let finish = done
            .finish
            .ok_or_else(|| format!("cell {name}: greedy cell without a finish reason"))?;
        if let Some(retained) = &case.retained {
            eprintln!(
                "qwen4_slowlink_quality: case {} ids {} the retained ids",
                case.id,
                if retained == &done.ids {
                    "match"
                } else {
                    "DIFFER from"
                }
            );
        }
        let ids_line = json!({
            "id": case.id,
            "token_ids": done.ids,
            "finish_reason": finish,
            "prompt_tokens": done.prompt_tokens,
        })
        .to_string();
        let receipts_line = json!({"id": case.id, "slowlink": done.receipts}).to_string();
        if let Some(file) = self.ids_file.as_mut() {
            append_line(file, "ids.jsonl", &ids_line)?;
        }
        if let Some(file) = self.receipts_file.as_mut() {
            append_line(file, "receipts.jsonl", &receipts_line)?;
        }
        Ok(())
    }

    fn run_primary(&mut self, index: usize) -> Result<(), String> {
        let plan = self.plan;
        let context = &plan.primary.contexts[index];
        let how = self.direct_plan(context.prefix_ids.len());
        let extra = json!({
            "context_id": context.id,
            "prefix_tokens": context.prefix_ids.len(),
            "prefix_sha256": context.prefix_sha256,
            "heldout_sha256": context.heldout_sha256,
            "sources": context
                .sources
                .iter()
                .map(|s| json!({
                    "label": s.label, "path": s.path, "sha256": s.sha256,
                    "concat_offset": s.concat_offset, "token_count": s.token_count,
                }))
                .collect::<Vec<_>>(),
        });
        self.run_cell(
            &primary_cell(&context.id, "teacher"),
            "primary-teacher",
            Some(&context.id),
            &context.prefix_ids,
            how.clone(),
            Body::Teacher {
                feed: context.feed(),
                next_tokens: context.next_tokens(),
                teacher_sha256: context.heldout_sha256.clone(),
            },
            extra.clone(),
        )?;
        // A separate reset + prefill: never the already teacher-advanced state.
        self.run_cell(
            &primary_cell(&context.id, "greedy"),
            "primary-greedy",
            Some(&context.id),
            &context.prefix_ids,
            how,
            Body::Greedy {
                max_new: plan.primary.greedy_new_tokens,
            },
            extra,
        )?;
        Ok(())
    }

    fn run_resume(&mut self, split: bool) -> Result<(), String> {
        let plan = self.plan;
        let resume = &plan.resume;
        let (name, how) = if split {
            (
                RESUME_SPLIT,
                PrefillPlan::Direct(vec![
                    (0, resume.split_at),
                    (resume.split_at, resume.prompt_ids.len()),
                ]),
            )
        } else {
            (
                RESUME_COLD,
                PrefillPlan::Direct(vec![(0, resume.prompt_ids.len())]),
            )
        };
        self.run_cell(
            name,
            "resume",
            None,
            &resume.prompt_ids,
            how,
            Body::Teacher {
                feed: &resume.teacher_ids,
                next_tokens: resume.next_tokens(),
                teacher_sha256: resume.teacher_sha256.clone(),
            },
            json!({"split_at": split.then_some(resume.split_at)}),
        )?;
        Ok(())
    }
}

fn required<'a>(value: &'a Option<PathBuf>, flag: &str) -> Result<&'a PathBuf, String> {
    value.as_ref().ok_or_else(|| format!("{flag} is required"))
}

fn same_file_path(a: &Path, b: &Path) -> Result<bool, String> {
    let a = std::fs::canonicalize(a).map_err(|e| format!("canonicalize {}: {e}", a.display()))?;
    let b = std::fs::canonicalize(b).map_err(|e| format!("canonicalize {}: {e}", b.display()))?;
    Ok(a == b)
}

fn run_gpu_phase(args: &Args, plan: &Plan, plan_bytes: &[u8]) -> Result<(), String> {
    let phase = args.phase;
    let arm = phase
        .arm()
        .ok_or_else(|| "internal: not a GPU phase".to_string())?;
    let started = Instant::now();
    let model = required(&args.model, "--model")?;
    let manifest_path = required(&args.manifest, "--manifest")?;
    let out_dir = required(&args.out_dir, "--out-dir")?;

    // Provenance first, before anything is created or loaded.
    let manifest_bytes = std::fs::read(manifest_path)
        .map_err(|e| format!("read {}: {e}", manifest_path.display()))?;
    let manifest = Manifest::from_slice(&manifest_bytes, &plan.artifact)?;
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let exe_sha = file_sha256(&exe)?;
    if exe_sha != manifest.binary_sha256()? {
        return Err(format!(
            "this executable ({}) hashes to {exe_sha}, the manifest names {}",
            exe.display(),
            manifest.binary_sha256()?
        ));
    }
    if !same_file_path(model, Path::new(&plan.artifact.path))? {
        return Err(format!(
            "--model {} is not the plan's artifact {}",
            model.display(),
            plan.artifact.path
        ));
    }
    let stat_before = stat_model(model)?;
    verify_model_stat(&manifest.model, &stat_before)?;
    let recipe = parse_execution_recipe(&manifest)?;
    check_plan_against_recipe(plan, recipe.as_ref())?;
    if let Some(sealed) = recipe.as_ref() {
        verify_verified_identity(&manifest, &plan.artifact, sealed, model)?;
    }
    let env_observed = check_env(&plan.expect, arm, phase.state_f32(), &|name| {
        hipfire_config::developer_var(name).ok()
    })?;
    check_mode(arm, SlowlinkMode::from_env())?;

    let dir = OutputDir::create_fresh(out_dir)?;
    let cells_file = dir.create_file("cells.jsonl")?;
    let (ids_file, receipts_file) = if phase.runs_determinism() {
        (
            Some(dir.create_file("ids.jsonl")?),
            Some(dir.create_file("receipts.jsonl")?),
        )
    } else {
        (None, None)
    };

    let (ctx, load_facts) = load_ctx(model, arm, phase.state_f32(), plan, recipe.as_ref())?;
    let mode = SlowlinkMode::from_env();
    let det_cases = if phase.runs_determinism() {
        resolve_det_cases(plan, &ctx)?
    } else {
        Vec::new()
    };
    let mut run = Run {
        plan,
        sealed: recipe.as_ref(),
        phase,
        arm,
        mode,
        ctx,
        dir: dir.clone(),
        cells_file,
        ids_file,
        receipts_file,
        row_buf: Vec::new(),
        done: Vec::new(),
        names: Vec::new(),
    };
    // The determinism cases run FIRST, in the plan's order.
    for case in &det_cases {
        run.run_det_case(case)?;
    }
    if phase.runs_primary_and_resume() {
        for index in 0..plan.primary.contexts.len() {
            run.run_primary(index)?;
        }
        if phase == Phase::S1a {
            run.run_resume(true)?;
        }
        run.run_resume(false)?;
        if phase == Phase::Baseline {
            // The exact baseline split 1537 + 1863 (q8): the fourth q8 arm.
            run.run_resume(true)?;
        }
    }
    match phase {
        Phase::BetaF32 => run.run_resume(false)?,
        Phase::S1F32 => run.run_resume(true)?,
        _ => {}
    }
    let expected_names = plan.cell_names(phase);
    if run.names != expected_names {
        return Err(format!(
            "ran cells {:?}, the phase defines {:?}",
            run.names, expected_names
        ));
    }
    let stat_after = stat_model(model)?;
    if stat_after != stat_before {
        return Err("the model's stat changed while the phase ran".into());
    }
    let summary = json!({
        "schema": SUMMARY_SCHEMA,
        "complete": true,
        "phase": phase.name(),
        "arm": arm.name(),
        "state_f32": phase.state_f32(),
        "plan": {
            "path": args.plan.display().to_string(),
            "sha256": sha256_hex(plan_bytes),
            "max_seq": plan.max_seq,
            "expect": {
                "resident_layers": plan.expect.resident_layers,
                "baseline_tile_rows": plan.expect.baseline_tile_rows,
                "s1_tile_rows": plan.expect.s1_tile_rows,
                "s1_layer_major_min_rows": plan.expect.s1_layer_major_min_rows,
                "ple_format": plan.expect.ple_format,
            },
            "execution_recipe": recipe.as_ref().map(|sealed| json!({
                "resident_layers": sealed.resident_layers,
                "resident_layer_set": sealed.resident_layer_set,
                "placement": sealed.placement,
            })).unwrap_or(Value::Null),
        },
        "manifest": {
            "path": manifest_path.display().to_string(),
            "sha256": sha256_hex(&manifest_bytes),
            "content": {
                "schema": manifest.schema,
                "binary": manifest.binary,
                "source": manifest.source,
                "launch": manifest.launch,
                "model": {
                    "path": manifest.model.path,
                    "dev": manifest.model.dev,
                    "ino": manifest.model.ino,
                    "size": manifest.model.size,
                    "mtime_sec": manifest.model.mtime_sec,
                    "mtime_nsec": manifest.model.mtime_nsec,
                    "sha256": manifest.model.sha256,
                },
            },
        },
        "binary": {"path": exe.display().to_string(), "sha256": exe_sha},
        "model": {
            "path": model.display().to_string(),
            "pinned_sha256": manifest.model.sha256,
            "hashed_here": false,
            "verified_cache": recipe.as_ref().map(|sealed| json!({
                "home": sealed.cache_home,
                "path": sealed.cache_path,
                "sha256": sealed.cache_sha256,
            })).unwrap_or(Value::Null),
            "stat": {
                "dev": stat_before.dev,
                "ino": stat_before.ino,
                "size": stat_before.size,
                "mtime_sec": stat_before.mtime_sec,
                "mtime_nsec": stat_before.mtime_nsec,
                "ctime_sec": stat_before.ctime_sec,
                "ctime_nsec": stat_before.ctime_nsec,
                "ctime_is_metadata_only": true,
            },
        },
        "environment": env_observed,
        "load": load_facts,
        "cells": run.done,
        "cell_order": run.names,
        "wall_seconds": started.elapsed().as_secs_f64(),
    });
    let text = serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?;
    dir.write_new("phase-summary.json", text.as_bytes())?;
    eprintln!(
        "qwen4_slowlink_quality: phase {} complete: {} cells in {:.0}s",
        phase.name(),
        run.names.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

// ── CPU phases: check-plan, score ──────────────────────────────────────

fn run_check_plan(args: &Args, plan: &Plan, plan_bytes: &[u8]) -> Result<(), String> {
    let cells = |phase: Phase| plan.cell_names(phase);
    let report = json!({
        "plan": args.plan.display().to_string(),
        "plan_sha256": sha256_hex(plan_bytes),
        "status": "valid",
        "max_seq": plan.max_seq,
        "required_max_seq": plan.required_max_seq().max(MIN_MAX_SEQ),
        "cells": {
            "baseline": cells(Phase::Baseline),
            "s1a": cells(Phase::S1a),
            "s1b": cells(Phase::S1b),
            "beta-f32": cells(Phase::BetaF32),
            "s1-f32": cells(Phase::S1F32),
        },
        "primary_rows": plan.primary.heldout_rows,
        "resume_rows": plan.resume.teacher_ids.len() + 1,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}

/// One phase directory's `cells.jsonl`, by cell name.
#[derive(Debug)]
struct CellIndex {
    dir: PathBuf,
    cells: BTreeMap<String, Value>,
}

fn read_cells(dir: &Path) -> Result<CellIndex, String> {
    let summary_path = dir.join("phase-summary.json");
    let summary: Value = serde_json::from_slice(&std::fs::read(&summary_path).map_err(|e| {
        format!(
            "read {} (an incomplete phase has none): {e}",
            summary_path.display()
        )
    })?)
    .map_err(|e| format!("parse {}: {e}", summary_path.display()))?;
    if summary.get("complete") != Some(&Value::Bool(true)) {
        return Err(format!(
            "{} is not a complete phase",
            summary_path.display()
        ));
    }
    let path = dir.join("cells.jsonl");
    let file = std::fs::File::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut cells = BTreeMap::new();
    for (i, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("read {} line {}: {e}", path.display(), i + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line)
            .map_err(|e| format!("parse {} line {}: {e}", path.display(), i + 1))?;
        let name = value
            .get("cell")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{} line {}: no cell name", path.display(), i + 1))?
            .to_string();
        if cells.insert(name.clone(), value).is_some() {
            return Err(format!("{}: duplicate cell {name}", path.display()));
        }
    }
    Ok(CellIndex {
        dir: dir.to_path_buf(),
        cells,
    })
}

fn cell_field<'a>(cell: &'a Value, key: &str) -> Result<&'a str, String> {
    cell.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("cell record field {key:?} is missing or not a string"))
}

fn cell_uint(cell: &Value, key: &str) -> Result<u64, String> {
    cell.get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("cell record field {key:?} is missing or not an unsigned integer"))
}

impl CellIndex {
    fn cell(&self, name: &str) -> Result<&Value, String> {
        self.cells
            .get(name)
            .ok_or_else(|| format!("{} has no cell {name:?}", self.dir.display()))
    }
}

/// One row's metadata as the dump wrote it.
struct RowMetaIn {
    position: u64,
    next_token: Option<u64>,
    sha256: String,
    top1_id: u64,
    margin: f64,
}

fn read_rows_meta(dir: &Path, cell: &Value) -> Result<Vec<RowMetaIn>, String> {
    let name = cell_field(cell, "rows_file")?;
    let path = dir.join(name);
    let bytes = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if sha256_hex(&bytes) != cell_field(cell, "rows_file_sha256")? {
        return Err(format!(
            "{}: rows file hash differs from cells.jsonl",
            path.display()
        ));
    }
    let text = String::from_utf8(bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let value: Value = serde_json::from_str(line)
            .map_err(|e| format!("{} line {}: {e}", path.display(), i + 1))?;
        if value.get("row").and_then(Value::as_u64) != Some(i as u64) {
            return Err(format!(
                "{} line {}: rows are not in order",
                path.display(),
                i + 1
            ));
        }
        rows.push(RowMetaIn {
            position: cell_uint(&value, "context_position")?,
            next_token: value.get("next_token").and_then(Value::as_u64),
            sha256: cell_field(&value, "logits_sha256")?.to_string(),
            top1_id: cell_uint(&value, "top1_id")?,
            margin: value
                .get("top12_margin")
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("{} line {}: no top12_margin", path.display(), i + 1))?,
        });
    }
    if rows.len() as u64 != cell_uint(cell, "rows")? {
        return Err(format!(
            "{}: {} rows, cells.jsonl says {}",
            path.display(),
            rows.len(),
            cell_uint(cell, "rows")?
        ));
    }
    Ok(rows)
}

/// Sequential reader of one cell's dump that re-verifies every row hash and,
/// at the end, the stream hash.
struct BinReader {
    file: std::fs::File,
    path: PathBuf,
    buf: Vec<u8>,
    stream: Sha256,
    claimed_stream: String,
    claimed_bytes: u64,
    read_bytes: u64,
}

impl BinReader {
    fn open(dir: &Path, cell: &Value) -> Result<Self, String> {
        let path = dir.join(cell_field(cell, "bin")?);
        let file =
            std::fs::File::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let vocab = cell_uint(cell, "vocab")? as usize;
        Ok(Self {
            file,
            path,
            buf: vec![0u8; vocab * 4],
            stream: Sha256::new(),
            claimed_stream: cell_field(cell, "stream_sha256")?.to_string(),
            claimed_bytes: cell_uint(cell, "bin_bytes")?,
            read_bytes: 0,
        })
    }

    fn next_row(&mut self, expect_sha: &str, row: &mut Vec<f32>) -> Result<(), String> {
        self.file
            .read_exact(&mut self.buf)
            .map_err(|e| format!("read {}: {e}", self.path.display()))?;
        if sha256_hex(&self.buf) != expect_sha {
            return Err(format!(
                "{}: row at byte {} does not match its recorded hash",
                self.path.display(),
                self.read_bytes
            ));
        }
        self.stream.update(&self.buf);
        self.read_bytes += self.buf.len() as u64;
        decode_row_le(&self.buf, row)
    }

    fn finish(self) -> Result<(), String> {
        let mut probe = [0u8; 1];
        let mut file = self.file;
        let extra = file
            .read(&mut probe)
            .map_err(|e| format!("read {}: {e}", self.path.display()))?;
        if extra != 0 || self.read_bytes != self.claimed_bytes {
            return Err(format!(
                "{}: {} bytes read, {} recorded, trailing data {}",
                self.path.display(),
                self.read_bytes,
                self.claimed_bytes,
                extra != 0
            ));
        }
        let stream = format!("{:x}", self.stream.finalize());
        if stream != self.claimed_stream {
            return Err(format!(
                "{}: stream hash {stream} differs from the recorded {}",
                self.path.display(),
                self.claimed_stream
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Side {
    Baseline,
    S1a,
    BetaF32,
    S1F32,
}

/// The phase directories a score run reads. `score-f32` reads the f32 pair
/// only; the q8 directories are not needed (nor recomputed).
struct Sources {
    baseline: Option<CellIndex>,
    s1a: Option<CellIndex>,
    beta_f32: Option<CellIndex>,
    s1_f32: Option<CellIndex>,
}

impl Sources {
    fn pick(&self, side: Side) -> Result<&CellIndex, String> {
        let (index, flag) = match side {
            Side::Baseline => (&self.baseline, "--baseline-dir"),
            Side::S1a => (&self.s1a, "--s1a-dir"),
            Side::BetaF32 => (&self.beta_f32, "--beta-f32-dir"),
            Side::S1F32 => (&self.s1_f32, "--s1-f32-dir"),
        };
        index
            .as_ref()
            .ok_or_else(|| format!("{flag} was not given"))
    }
}

struct PairSpec {
    name: String,
    reference: (Side, String),
    candidate: (Side, String),
    expected_rows: usize,
}

/// The f32 diagnostic pair: the baseline-route cold 3400 call against the S1
/// split 1537 + 1863, both at `HIPFIRE_STATE_QUANT=f32`.
fn f32_pair_spec(plan: &Plan) -> PairSpec {
    PairSpec {
        name: "resume-f32-s1split-vs-betacold".into(),
        reference: (Side::BetaF32, RESUME_COLD.into()),
        candidate: (Side::S1F32, RESUME_SPLIT.into()),
        expected_rows: plan.resume.teacher_ids.len() + 1,
    }
}

fn pair_specs(plan: &Plan, with_f32: bool) -> Vec<PairSpec> {
    let mut pairs = Vec::new();
    for context in &plan.primary.contexts {
        let cell = primary_cell(&context.id, "teacher");
        pairs.push(PairSpec {
            name: format!("primary-{}", context.id),
            reference: (Side::Baseline, cell.clone()),
            candidate: (Side::S1a, cell),
            expected_rows: plan.primary.heldout_rows,
        });
    }
    let resume_rows = plan.resume.teacher_ids.len() + 1;
    pairs.push(PairSpec {
        name: "resume-s1split".into(),
        reference: (Side::Baseline, RESUME_COLD.into()),
        candidate: (Side::S1a, RESUME_SPLIT.into()),
        expected_rows: resume_rows,
    });
    pairs.push(PairSpec {
        name: "resume-s1cold".into(),
        reference: (Side::Baseline, RESUME_COLD.into()),
        candidate: (Side::S1a, RESUME_COLD.into()),
        expected_rows: resume_rows,
    });
    pairs.push(PairSpec {
        name: "resume-s1cold-vs-s1split".into(),
        reference: (Side::S1a, RESUME_COLD.into()),
        candidate: (Side::S1a, RESUME_SPLIT.into()),
        expected_rows: resume_rows,
    });
    // The fourth q8 arm: the exact baseline split, against the baseline cold
    // call (the boundary alone) and against both S1 arms.
    pairs.push(PairSpec {
        name: "resume-betasplit".into(),
        reference: (Side::Baseline, RESUME_COLD.into()),
        candidate: (Side::Baseline, RESUME_SPLIT.into()),
        expected_rows: resume_rows,
    });
    pairs.push(PairSpec {
        name: "resume-s1split-vs-betasplit".into(),
        reference: (Side::Baseline, RESUME_SPLIT.into()),
        candidate: (Side::S1a, RESUME_SPLIT.into()),
        expected_rows: resume_rows,
    });
    pairs.push(PairSpec {
        name: "resume-s1cold-vs-betasplit".into(),
        reference: (Side::Baseline, RESUME_SPLIT.into()),
        candidate: (Side::S1a, RESUME_COLD.into()),
        expected_rows: resume_rows,
    });
    if with_f32 {
        pairs.push(f32_pair_spec(plan));
    }
    pairs
}

fn score_pair(spec: &PairSpec, sources: &Sources, kl_out: &mut dyn Write) -> Result<Value, String> {
    let at = |e: String| format!("pair {}: {e}", spec.name);
    let ref_index = sources.pick(spec.reference.0).map_err(at)?;
    let cand_index = sources.pick(spec.candidate.0).map_err(at)?;
    let ref_cell = ref_index.cell(&spec.reference.1).map_err(at)?;
    let cand_cell = cand_index.cell(&spec.candidate.1).map_err(at)?;
    for key in ["prompt_sha256", "teacher_sha256"] {
        if ref_cell.get(key) != cand_cell.get(key) {
            return Err(at(format!(
                "{key} differs between the reference and candidate cells"
            )));
        }
    }
    let vocab = cell_uint(ref_cell, "vocab").map_err(at)?;
    if vocab != cell_uint(cand_cell, "vocab").map_err(at)? {
        return Err(at("vocab differs".into()));
    }
    let ref_meta = read_rows_meta(&ref_index.dir, ref_cell).map_err(at)?;
    let cand_meta = read_rows_meta(&cand_index.dir, cand_cell).map_err(at)?;
    if ref_meta.len() != cand_meta.len() {
        return Err(at(format!(
            "reference has {} rows, candidate {}",
            ref_meta.len(),
            cand_meta.len()
        )));
    }
    let mut ref_bin = BinReader::open(&ref_index.dir, ref_cell).map_err(at)?;
    let mut cand_bin = BinReader::open(&cand_index.dir, cand_cell).map_err(at)?;
    let (mut ref_row, mut cand_row) = (Vec::new(), Vec::new());
    let mut sum = Neumaier::default();
    let (mut max_kl, mut max_row) = (f64::NEG_INFINITY, 0usize);
    let (mut infinite, mut top1_agree) = (0usize, 0usize);
    for (i, (r, c)) in ref_meta.iter().zip(&cand_meta).enumerate() {
        if r.position != c.position || r.next_token != c.next_token {
            return Err(at(format!(
                "row {i}: the histories differ (position {} vs {}, next token {:?} vs {:?})",
                r.position, c.position, r.next_token, c.next_token
            )));
        }
        ref_bin.next_row(&r.sha256, &mut ref_row).map_err(at)?;
        cand_bin.next_row(&c.sha256, &mut cand_row).map_err(at)?;
        let kl = kl_divergence_f64(&ref_row, &cand_row).map_err(|e| at(format!("row {i}: {e}")))?;
        if kl.is_finite() {
            sum.add(kl);
            if kl > max_kl {
                max_kl = kl;
                max_row = i;
            }
        } else {
            infinite += 1;
        }
        if r.top1_id == c.top1_id {
            top1_agree += 1;
        }
        let line = json!({
            "schema": KL_ROW_SCHEMA,
            "pair": spec.name,
            "row": i,
            "context_position": r.position,
            "next_token": r.next_token,
            "kl_nats": kl.is_finite().then_some(kl),
            "kl_status": if kl.is_finite() { "ok" } else { "infinite" },
            "reference_top1_id": r.top1_id,
            "candidate_top1_id": c.top1_id,
            "top1_match": r.top1_id == c.top1_id,
            "reference_margin": r.margin,
            "candidate_margin": c.margin,
        });
        writeln!(kl_out, "{line}").map_err(|e| at(format!("write kl.jsonl: {e}")))?;
    }
    ref_bin.finish().map_err(at)?;
    cand_bin.finish().map_err(at)?;
    let rows = ref_meta.len();
    let finite = rows - infinite;
    Ok(json!({
        "schema": KL_SUMMARY_SCHEMA,
        "pair": spec.name,
        "reference": {"cell": spec.reference.1, "stream_sha256": ref_cell.get("stream_sha256")},
        "candidate": {"cell": spec.candidate.1, "stream_sha256": cand_cell.get("stream_sha256")},
        "rows": rows,
        "expected_rows": spec.expected_rows,
        "complete": rows == spec.expected_rows,
        "infinite_rows": infinite,
        "kl_sum_nats": (finite > 0).then(|| sum.total()),
        // Arithmetic mean of the rows present when every row is finite; no
        // interval is computed.
        "kl_mean_nats": (infinite == 0 && rows > 0).then(|| sum.total() / rows as f64),
        "kl_max_nats": (finite > 0).then_some(max_kl),
        "kl_max_row": (finite > 0).then_some(max_row),
        "top1_agree_rows": top1_agree,
    }))
}

/// Per determinism case: ids, finish, row count, stream hash and every row hash
/// compared between two independent S1 runs.
fn compare_determinism(plan: &Plan, a: &CellIndex, b: &CellIndex) -> Result<Value, String> {
    let mut cases = Vec::new();
    for id in &plan.determinism.case_ids {
        let name = det_cell(id);
        let (ca, cb) = (a.cell(&name)?, b.cell(&name)?);
        let (ma, mb) = (read_rows_meta(&a.dir, ca)?, read_rows_meta(&b.dir, cb)?);
        let first_diff = ma.iter().zip(&mb).position(|(x, y)| x.sha256 != y.sha256);
        let ids_equal = ca.get("token_ids") == cb.get("token_ids");
        let identical = ids_equal
            && ca.get("finish_reason") == cb.get("finish_reason")
            && ca.get("stream_sha256") == cb.get("stream_sha256")
            && ma.len() == mb.len()
            && first_diff.is_none();
        cases.push(json!({
            "case": id,
            "cell": name,
            "identical": identical,
            "token_ids_equal": ids_equal,
            "finish_reason": [ca.get("finish_reason"), cb.get("finish_reason")],
            "rows": [ma.len(), mb.len()],
            "first_differing_row": first_diff,
            "stream_sha256": [ca.get("stream_sha256"), cb.get("stream_sha256")],
        }));
    }
    let all = cases.iter().all(|c| c["identical"] == Value::Bool(true));
    Ok(json!({
        "schema": DETERMINISM_SCHEMA,
        "all_identical": all,
        "cases": cases,
    }))
}

fn run_score(args: &Args, plan: &Plan, f32_only: bool) -> Result<(), String> {
    let sources = if f32_only {
        Sources {
            baseline: None,
            s1a: None,
            beta_f32: Some(read_cells(required(&args.beta_f32_dir, "--beta-f32-dir")?)?),
            s1_f32: Some(read_cells(required(&args.s1_f32_dir, "--s1-f32-dir")?)?),
        }
    } else {
        let sources = Sources {
            baseline: Some(read_cells(required(&args.baseline_dir, "--baseline-dir")?)?),
            s1a: Some(read_cells(required(&args.s1a_dir, "--s1a-dir")?)?),
            beta_f32: args.beta_f32_dir.as_deref().map(read_cells).transpose()?,
            s1_f32: args.s1_f32_dir.as_deref().map(read_cells).transpose()?,
        };
        if sources.beta_f32.is_some() != sources.s1_f32.is_some() {
            return Err("--beta-f32-dir and --s1-f32-dir are given together or not at all".into());
        }
        sources
    };
    let dir = OutputDir::create_fresh(required(&args.out_dir, "--out-dir")?)?;
    let mut kl_file = std::io::BufWriter::new(dir.create_file("kl.jsonl")?);
    let mut summaries = Vec::new();
    let specs = if f32_only {
        vec![f32_pair_spec(plan)]
    } else {
        pair_specs(plan, sources.beta_f32.is_some())
    };
    for spec in specs {
        let summary = score_pair(&spec, &sources, &mut kl_file)?;
        eprintln!(
            "qwen4_slowlink_quality: scored {}: {} rows, mean {:?} nats, top-1 agree {}/{}",
            spec.name,
            summary["rows"],
            summary["kl_mean_nats"],
            summary["top1_agree_rows"],
            summary["rows"]
        );
        summaries.push(summary);
    }
    kl_file
        .flush()
        .map_err(|e| format!("flush kl.jsonl: {e}"))?;
    let mut report = json!({
        "scope": if f32_only { "f32-pair-only" } else { "full" },
        "pairs": summaries,
    });
    if let Some(s1b_dir) = &args.s1b_dir {
        let s1b = read_cells(s1b_dir)?;
        let s1a = sources
            .s1a
            .as_ref()
            .ok_or_else(|| "--s1b-dir needs --s1a-dir".to_string())?;
        let determinism = compare_determinism(plan, s1a, &s1b)?;
        dir.write_new(
            "determinism.json",
            serde_json::to_string_pretty(&determinism)
                .map_err(|e| e.to_string())?
                .as_bytes(),
        )?;
        report["determinism_all_identical"] = determinism["all_identical"].clone();
    }
    dir.write_new(
        "kl-summary.json",
        serde_json::to_string_pretty(&report)
            .map_err(|e| e.to_string())?
            .as_bytes(),
    )?;
    Ok(())
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let args = parse_args(argv).map_err(|e| format!("{e}\n{USAGE}"))?;
    let plan_bytes =
        std::fs::read(&args.plan).map_err(|e| format!("read {}: {e}", args.plan.display()))?;
    let plan = Plan::from_slice(&plan_bytes)?;
    match args.phase {
        Phase::CheckPlan => run_check_plan(&args, &plan, &plan_bytes),
        Phase::Score => run_score(&args, &plan, false),
        Phase::ScoreF32 => run_score(&args, &plan, true),
        Phase::Baseline | Phase::S1a | Phase::S1b | Phase::BetaF32 | Phase::S1F32 => {
            run_gpu_phase(&args, &plan, &plan_bytes)
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("qwen4_slowlink_quality: {error}");
        std::process::exit(1);
    }
}

// ── CPU tests (no GPU, no model) ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "qwen4_slowlink_quality-{tag}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn ids(count: usize, salt: u32) -> Vec<u32> {
        (0..count as u32).map(|i| (i * 7 + salt) % 1000).collect()
    }

    fn source(label: &str, offset: usize, count: usize) -> Value {
        json!({
            "label": label,
            "path": format!("/data/{label}.json"),
            "sha256": sha256_hex(label.as_bytes()),
            "concat_offset": offset,
            "token_count": count,
        })
    }

    fn context(id: &str, prefix: usize, salt: u32) -> Value {
        let prefix_ids = ids(prefix, salt);
        let heldout_ids = ids(HELDOUT_ROWS, salt + 3);
        json!({
            "id": id,
            "prefix_sha256": sha_u32(&prefix_ids),
            "heldout_sha256": sha_u32(&heldout_ids),
            "prefix_ids": prefix_ids,
            "heldout_ids": heldout_ids,
            "sources": [
                source("a", 0, prefix),
                source("b", prefix, HELDOUT_ROWS),
            ],
        })
    }

    fn det_case(index: usize) -> Value {
        let id = format!("case{index}");
        let prompt = ids(10 + index, index as u32);
        json!({
            "id": id,
            "prompt_ids": prompt,
            "expected_prompt_tokens": 10 + index,
        })
    }

    fn pinned_env() -> Value {
        let mut map = serde_json::Map::new();
        for (name, value) in PINNED_ENV {
            map.insert(name.to_string(), json!(value));
        }
        Value::Object(map)
    }

    fn sample_plan_value() -> Value {
        let retained = ids(RESUME_RETAINED_IDS, 11);
        let teacher = retained[..RESUME_TEACHER_IDS].to_vec();
        let resume_prompt = ids(RESUME_PROMPT_TOKENS, 5);
        let cases: Vec<Value> = (0..DET_CASES).map(det_case).collect();
        let case_ids: Vec<String> = (0..DET_CASES).map(|i| format!("case{i}")).collect();
        json!({
            "schema": PLAN_SCHEMA,
            "max_seq": 16897,
            "artifact": {"path": "/models/m.hfq", "size": 125u64, "sha256": sha256_hex(b"model")},
            "expect": {
                "resident_layers": 9,
                "baseline_tile_rows": 4096,
                "s1_tile_rows": 768,
                "s1_layer_major_min_rows": 1537,
                "ple_format": "q4",
                "env": pinned_env(),
                "env_absent": ["HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS"],
                "baseline_env": {
                    "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS": "9",
                    "HIPFIRE_PREFILL_CHUNK_ROWS": "4096",
                },
            },
            "determinism": {
                "case_ids": case_ids,
                "max_new_tokens": 32,
                "thinking": false,
                "cases": cases,
            },
            "primary": {
                "heldout_rows": 512,
                "greedy_new_tokens": 32,
                "contexts": [
                    context(CTX_16K, CTX_16K_PREFIX, 1),
                    context(CTX_4K, CTX_4K_PREFIX, 2),
                ],
            },
            "resume": {
                "prompt_sha256": sha_u32(&resume_prompt),
                "prompt_ids": resume_prompt,
                "split_at": 1537,
                "teacher_sha256": sha_u32(&teacher),
                "teacher_ids": teacher,
                "retained_s1_ids": retained,
            },
        })
    }

    fn plan_err(mutate: impl FnOnce(&mut Value)) -> String {
        let mut value = sample_plan_value();
        mutate(&mut value);
        match Plan::from_value(value) {
            Ok(_) => panic!("the mutated plan validated"),
            Err(e) => e,
        }
    }

    // ── plan bounds, counts, shared history ──

    #[test]
    fn plan_validates_and_counts_cells() {
        let plan = Plan::from_value(sample_plan_value()).unwrap();
        assert_eq!(plan.required_max_seq(), 16897);
        let names = |phase| plan.cell_names(phase);
        assert_eq!(names(Phase::Baseline).len(), 6);
        assert_eq!(names(Phase::S1a).len(), DET_CASES + 4 + 2);
        assert_eq!(names(Phase::S1b).len(), DET_CASES);
        // The determinism cases come FIRST in s1a, in the plan's order.
        assert_eq!(&names(Phase::S1a)[..DET_CASES], &names(Phase::S1b)[..]);
        assert_eq!(
            names(Phase::Baseline),
            vec![
                "primary-ctx16k-teacher",
                "primary-ctx16k-greedy",
                "primary-ctx4k-teacher",
                "primary-ctx4k-greedy",
                "resume-cold",
                "resume-split",
            ]
        );
        assert_eq!(
            &names(Phase::S1a)[DET_CASES..],
            &[
                "primary-ctx16k-teacher",
                "primary-ctx16k-greedy",
                "primary-ctx4k-teacher",
                "primary-ctx4k-greedy",
                "resume-split",
                "resume-cold",
            ]
        );
        assert_eq!(names(Phase::BetaF32), vec!["resume-cold"]);
        assert_eq!(names(Phase::S1F32), vec!["resume-split"]);
    }

    #[test]
    fn plan_bounds_are_enforced() {
        assert!(plan_err(|v| {
            v["max_seq"] = json!(16896);
        })
        .contains("max_seq 16896 is below the required"));
        assert!(plan_err(|v| {
            v["primary"]["contexts"][0]["heldout_ids"]
                .as_array_mut()
                .unwrap()
                .pop();
        })
        .contains("context ctx16k: heldout_ids has 511 ids, expected 512"));
        assert!(plan_err(|v| {
            v["primary"]["contexts"][1]["prefix_ids"]
                .as_array_mut()
                .unwrap()
                .pop();
        })
        .contains("context ctx4k: prefix_ids has 4096 ids, expected 4097"));
        assert!(plan_err(|v| {
            v["primary"]["contexts"][0]["prefix_sha256"] = json!(sha256_hex(b"x"));
        })
        .contains("prefix_sha256 mismatch"));
        assert!(plan_err(|v| {
            v["primary"]["contexts"][0]["heldout_ids"][3] = json!(999_999);
        })
        .contains("heldout_sha256 mismatch"));
        assert!(plan_err(|v| {
            v["primary"]["contexts"][0]["sources"][1]["concat_offset"] = json!(5);
        })
        .contains("not contiguous"));
        assert!(plan_err(|v| {
            v["primary"]["contexts"][1]["sources"][1]["token_count"] = json!(100);
        })
        .contains("sources cover"));
        assert!(plan_err(|v| {
            v["expect"]["resident_layers"] = json!(5);
        })
        .contains("expect.resident_layers is 5"));
        assert!(plan_err(|v| {
            v["expect"]["baseline_tile_rows"] = json!(8192);
        })
        .contains("baseline_tile_rows"));
        assert!(plan_err(|v| {
            v["expect"]["s1_tile_rows"] = json!(1024);
        })
        .contains("s1_tile_rows"));
        assert!(plan_err(|v| {
            v["expect"]["env_absent"] = json!(["HIPFIRE_QWEN4_D2_DEFAULT"]);
        })
        .contains("both required and absent"));
        assert!(plan_err(|v| {
            v["expect"]["env"]["HIPFIRE_STATE_QUANT"] = json!("f32");
        })
        .contains("expect.env[HIPFIRE_STATE_QUANT] must be \"q8\""));
        assert!(plan_err(|v| {
            v["expect"]["env"]
                .as_object_mut()
                .unwrap()
                .remove("HIPFIRE_KV_MODE");
        })
        .contains("expect.env[HIPFIRE_KV_MODE]"));
        assert!(plan_err(|v| {
            v["expect"]["env_absent"] = json!([]);
        })
        .contains("expect.env_absent (S1 routes) must name"));
        assert!(plan_err(|v| {
            v["expect"]["baseline_env"]["HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS"] = json!("8");
        })
        .contains("expect.baseline_env[HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS] must be \"9\""));
        assert!(plan_err(|v| {
            v["expect"]["baseline_env"]
                .as_object_mut()
                .unwrap()
                .remove("HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS");
        })
        .contains("expect.baseline_env[HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS]"));
        assert!(plan_err(|v| {
            v["expect"]["baseline_env"]["HIPFIRE_PREFILL_CHUNK_ROWS"] = json!("8192");
        })
        .contains("expect.baseline_env[HIPFIRE_PREFILL_CHUNK_ROWS] must be \"4096\""));
        assert!(plan_err(|v| {
            v["expect"]["env"]["HIPFIRE_PREFILL_CHUNK_ROWS"] = json!("4096");
        })
        .contains("is in both env and baseline_env"));
        assert!(plan_err(|v| {
            v["expect"].as_object_mut().unwrap().remove("baseline_env");
        })
        .contains("baseline_env"));
        assert!(plan_err(|v| {
            v["artifact"]["sha256"] = json!("abc");
        })
        .contains("artifact.sha256"));
        assert!(plan_err(|v| {
            v["unexpected"] = json!(1);
        })
        .contains("unexpected"));
    }

    #[test]
    fn plan_case_ids_and_resume_are_enforced() {
        assert!(plan_err(|v| {
            v["determinism"]["cases"].as_array_mut().unwrap().pop();
        })
        .contains("has 6 cases, expected 7"));
        assert!(plan_err(|v| {
            v["determinism"]["case_ids"]
                .as_array_mut()
                .unwrap()
                .swap(0, 1);
        })
        .contains("case_ids does not match"));
        assert!(plan_err(|v| {
            v["determinism"]["cases"][2]["resume_at"] = json!(500);
        })
        .contains("resume_at 500 is not interior"));
        assert!(plan_err(|v| {
            v["determinism"]["cases"][2]["expected_prompt_tokens"] = json!(99);
        })
        .contains("expected_prompt_tokens is 99"));
        assert!(plan_err(|v| {
            v["determinism"]["cases"][1]["messages"] = json!([{"role": "user", "content": "x"}]);
        })
        .contains("both prompt_ids and messages"));
        assert!(plan_err(|v| {
            v["determinism"]["max_new_tokens"] = json!(16);
        })
        .contains("max_new_tokens is 16"));
        assert!(plan_err(|v| {
            v["resume"]["split_at"] = json!(1536);
        })
        .contains("resume.split_at is 1536"));
        // The shared teacher history is exactly the first 31 retained ids.
        assert!(plan_err(|v| {
            let mut teacher = ids(RESUME_TEACHER_IDS, 77);
            v["resume"]["teacher_sha256"] = json!(sha_u32(&teacher));
            v["resume"]["teacher_ids"] = json!(teacher.split_off(0));
        })
        .contains("is not the first 31 ids of resume.retained_s1_ids"));
        assert!(plan_err(|v| {
            v["resume"]["retained_s1_ids"].as_array_mut().unwrap().pop();
        })
        .contains("retained_s1_ids has 31 ids, expected 32"));
        // A rendered `messages` case is valid at plan time.
        let mut value = sample_plan_value();
        value["determinism"]["cases"][6] = json!({
            "id": "case6",
            "messages": [{"role": "user", "content": "hello"}],
            "expected_prompt_tokens": 594,
            "retained_ids": [1, 2, 3],
        });
        Plan::from_value(value).unwrap();
    }

    #[test]
    fn plan_vocab_check_names_the_offender() {
        let plan = Plan::from_value(sample_plan_value()).unwrap();
        plan.check_vocab(1000).unwrap();
        let err = plan.check_vocab(990).unwrap_err();
        assert!(err.contains("outside the vocabulary (990)"), "{err}");
    }

    #[test]
    fn shared_history_and_row_counts() {
        let plan = Plan::from_value(sample_plan_value()).unwrap();
        for context in &plan.primary.contexts {
            // 511 forwards -> 512 rows, row r followed by heldout[r].
            assert_eq!(context.feed().len(), HELDOUT_ROWS - 1);
            assert_eq!(context.feed(), &context.heldout_ids[..HELDOUT_ROWS - 1]);
            assert_eq!(context.next_tokens().len(), HELDOUT_ROWS);
            assert_eq!(context.next_tokens()[0], Some(context.heldout_ids[0]));
            assert_eq!(context.next_tokens()[511], Some(context.heldout_ids[511]));
        }
        // 31 shared teacher ids -> seed + 31 forwards = 32 rows; the last
        // row's follower is not part of the shared history.
        let next = plan.resume.next_tokens();
        assert_eq!(next.len(), 32);
        assert_eq!(next[30], Some(plan.resume.teacher_ids[30]));
        assert_eq!(next[31], None);
        // Both prefill structures of the 3400-token prompt tile it exactly.
        validate_ranges(&[(0, 1537), (1537, 3400)], 3400).unwrap();
        validate_ranges(&[(0, 3400)], 3400).unwrap();
        assert!(validate_ranges(&[(0, 1537), (1538, 3400)], 3400).is_err());
        assert!(validate_ranges(&[(0, 1537)], 3400).is_err());
        // The baseline's manual chunks.
        assert_eq!(
            chunk_ranges(16384, 4096),
            vec![(0, 4096), (4096, 8192), (8192, 12288), (12288, 16384)]
        );
        assert_eq!(chunk_ranges(4097, 4096), vec![(0, 4096), (4096, 4097)]);
        validate_ranges(&chunk_ranges(4097, 4096), 4097).unwrap();
    }

    /// Scripted rows for a [`RowSource`]; `rows[k]` is the row after `k` forwards.
    struct Scripted {
        rows: Vec<Vec<f32>>,
        at: usize,
        base: usize,
        fed: Vec<u32>,
        greedy_inputs: Vec<Option<u32>>,
        corrupt_device_argmax: Option<u32>,
    }

    impl Scripted {
        fn new(rows: Vec<Vec<f32>>, base: usize) -> Self {
            Self {
                rows,
                at: 0,
                base,
                fed: Vec::new(),
                greedy_inputs: Vec::new(),
                corrupt_device_argmax: None,
            }
        }
    }

    impl RowSource for Scripted {
        fn read_row(&mut self, out: &mut Vec<f32>) -> Result<(), String> {
            out.clear();
            out.extend_from_slice(self.rows.get(self.at).ok_or("script exhausted")?);
            Ok(())
        }

        fn forward_teacher(&mut self, token: u32) -> Result<(), String> {
            self.fed.push(token);
            self.at += 1;
            Ok(())
        }

        fn forward_greedy(&mut self, pending: Option<u32>) -> Result<u32, String> {
            self.greedy_inputs.push(pending);
            let token = match (pending, self.corrupt_device_argmax) {
                (Some(t), _) => t,
                (None, Some(bad)) => bad,
                (None, None) => hipfire_runtime::llama::argmax(&self.rows[self.at]),
            };
            self.fed.push(token);
            self.at += 1;
            Ok(token)
        }

        fn position(&self) -> usize {
            self.base + self.at
        }
    }

    fn peak(vocab: usize, at: usize, other: usize) -> Vec<f32> {
        let mut row = vec![0.0f32; vocab];
        row[at] = 5.0;
        row[other] = 1.0;
        row
    }

    type Seen = Vec<(Option<u32>, usize, RowStats)>;

    #[test]
    fn teacher_capture_feeds_the_same_history_in_every_arm() {
        let feed: Vec<u32> = vec![4, 5, 6, 7];
        // Two arms prefill differently but the teacher history is identical.
        let mut positions = Vec::new();
        for _arm in 0..2 {
            let rows: Vec<Vec<f32>> = (0..5).map(|k| peak(8, k, 7)).collect();
            let mut src = Scripted::new(rows, 100);
            let mut seen: Seen = Vec::new();
            let mut buf = Vec::new();
            let mut on_row =
                |input: Option<u32>, position: usize, row: &[f32]| -> Result<RowStats, String> {
                    let stats = row_stats(row)?;
                    seen.push((input, position, stats));
                    Ok(stats)
                };
            teacher_capture(&mut src, &feed, 100, &mut buf, &mut on_row).unwrap();
            assert_eq!(src.fed, feed);
            assert_eq!(seen.len(), feed.len() + 1);
            assert_eq!(seen[0].0, None);
            assert_eq!(seen[0].1, 100);
            assert_eq!(seen[4].1, 104);
            assert_eq!(
                seen.iter().map(|s| s.0).collect::<Vec<_>>(),
                vec![None, Some(4), Some(5), Some(6), Some(7)]
            );
            positions.push(seen.iter().map(|s| s.1).collect::<Vec<_>>());
        }
        assert_eq!(positions[0], positions[1]);
    }

    #[test]
    fn teacher_capture_refuses_position_drift() {
        struct Drifting(Scripted);
        impl RowSource for Drifting {
            fn read_row(&mut self, out: &mut Vec<f32>) -> Result<(), String> {
                self.0.read_row(out)
            }
            fn forward_teacher(&mut self, token: u32) -> Result<(), String> {
                self.0.forward_teacher(token)
            }
            fn forward_greedy(&mut self, pending: Option<u32>) -> Result<u32, String> {
                self.0.forward_greedy(pending)
            }
            fn position(&self) -> usize {
                self.0.position() + 1
            }
        }
        let rows = vec![peak(4, 0, 1), peak(4, 1, 2)];
        let mut src = Drifting(Scripted::new(rows, 10));
        let mut buf = Vec::new();
        let mut on_row = |_: Option<u32>, _: usize, row: &[f32]| row_stats(row);
        let err = teacher_capture(&mut src, &[1], 10, &mut buf, &mut on_row).unwrap_err();
        assert!(
            err.contains("context position 11 after the prefill, expected 10"),
            "{err}"
        );
    }

    // ── EOS last-forward semantics ──

    #[test]
    fn greedy_forwards_eos_and_captures_its_row() {
        // vocab 6, eos = 5. Seed argmax 2 -> forward 2 -> argmax 4 -> forward 4
        // -> argmax 5 (EOS) -> forward 5 -> row captured -> stop.
        let rows = vec![
            peak(6, 2, 0),
            peak(6, 4, 0),
            peak(6, 5, 0),
            peak(6, 1, 0), // the row AFTER the EOS forward
        ];
        let mut src = Scripted::new(rows, 7);
        let mut seen: Seen = Vec::new();
        let mut buf = Vec::new();
        let mut on_row =
            |input: Option<u32>, position: usize, row: &[f32]| -> Result<RowStats, String> {
                let stats = row_stats(row)?;
                seen.push((input, position, stats));
                Ok(stats)
            };
        let (ids, finish) =
            greedy_capture(&mut src, 32, 7, &|t| t == 5, &mut buf, &mut on_row).unwrap();
        assert_eq!(ids, vec![2, 4, 5]);
        assert_eq!(finish, "stop");
        // The EOS token IS the last id and WAS forwarded.
        assert_eq!(src.fed, vec![2, 4, 5]);
        // seed + one row per forwarded token, EOS's row last.
        assert_eq!(seen.len(), ids.len() + 1);
        assert_eq!(seen[3].0, Some(5));
        assert_eq!(seen[3].1, 10);
        assert_eq!(seen[3].2.top1_id, 1);
        // The first step ran the host argmax, later ones the device argmax.
        assert_eq!(src.greedy_inputs, vec![Some(2), None, None]);
    }

    #[test]
    fn greedy_length_finish_captures_every_forward() {
        let rows = vec![peak(6, 1, 0), peak(6, 2, 0), peak(6, 3, 0)];
        let mut src = Scripted::new(rows, 0);
        let mut count = 0usize;
        let mut buf = Vec::new();
        let mut on_row = |_: Option<u32>, _: usize, row: &[f32]| {
            count += 1;
            row_stats(row)
        };
        let (ids, finish) =
            greedy_capture(&mut src, 2, 0, &|t| t == 5, &mut buf, &mut on_row).unwrap();
        assert_eq!(ids, vec![1, 2]);
        assert_eq!(finish, "length");
        assert_eq!(count, 3);
        assert_eq!(src.fed.len(), 2);
    }

    #[test]
    fn greedy_refuses_a_device_argmax_that_disagrees_with_the_row() {
        let rows = vec![peak(6, 1, 0), peak(6, 2, 0), peak(6, 3, 0)];
        let mut src = Scripted::new(rows, 0);
        src.corrupt_device_argmax = Some(4);
        let mut buf = Vec::new();
        let mut on_row = |_: Option<u32>, _: usize, row: &[f32]| row_stats(row);
        let err = greedy_capture(&mut src, 4, 0, &|_| false, &mut buf, &mut on_row).unwrap_err();
        assert!(err.contains("device argmax token 4 differs"), "{err}");
    }

    // ── row statistics, top-2 tie order, non-finite policy ──

    #[test]
    fn top2_ties_resolve_to_the_lower_index() {
        let stats = row_stats(&[1.0, 5.0, 5.0, 3.0]).unwrap();
        assert_eq!((stats.top1_id, stats.top2_id), (1, 2));
        assert_eq!(stats.margin(), 0.0);
        let stats = row_stats(&[5.0, 5.0, 5.0]).unwrap();
        assert_eq!((stats.top1_id, stats.top2_id), (0, 1));
        let stats = row_stats(&[2.0, 7.5, -1.0, 7.0]).unwrap();
        assert_eq!((stats.top1_id, stats.top2_id), (1, 3));
        assert_eq!(stats.margin(), 0.5);
        // The best value arriving last still displaces both.
        let stats = row_stats(&[1.0, 2.0, 9.0]).unwrap();
        assert_eq!((stats.top1_id, stats.top2_id), (2, 1));
        // Ties agree with the greedy argmax.
        assert_eq!(
            row_stats(&[0.0, 3.0, 3.0]).unwrap().top1_id,
            hipfire_runtime::llama::argmax(&[0.0, 3.0, 3.0])
        );
    }

    #[test]
    fn row_stats_non_finite_policy() {
        assert!(row_stats(&[1.0, f32::NAN]).unwrap_err().contains("NaN"));
        assert!(row_stats(&[f32::INFINITY, 1.0])
            .unwrap_err()
            .contains("+inf"));
        let stats = row_stats(&[f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY, 0.5]).unwrap();
        assert_eq!(stats.neg_inf, 2);
        assert_eq!((stats.top1_id, stats.top2_id), (1, 3));
        assert!(row_stats(&[f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY])
            .unwrap_err()
            .contains("fewer than two finite"));
        assert!(row_stats(&[1.0]).is_err());
    }

    // ── canonical bytes and hashes ──

    #[test]
    fn canonical_row_bytes_and_hash() {
        let mut bytes = Vec::new();
        encode_row_le(&[1.0, -0.0, 2.5], &mut bytes);
        assert_eq!(
            bytes,
            vec![0, 0, 0x80, 0x3f, 0, 0, 0, 0x80, 0, 0, 0x20, 0x40]
        );
        assert_eq!(
            sha256_hex(&bytes),
            "60d6c502d760a348a356a109d80444c7e88b26abfcd4aed1f6ce28d518c0fab2"
        );
        // The buffer is reused, not appended to.
        encode_row_le(&[0.5, -1.0], &mut bytes);
        assert_eq!(bytes, vec![0, 0, 0, 0x3f, 0, 0, 0x80, 0xbf]);
        assert_eq!(
            sha256_hex(&bytes),
            "031a5e93e726630d2e7087e4356829de6907683f3a2aba769e5155e7a068d167"
        );
        // Round trip preserves the bits, including -0.0 and -inf.
        let original = [1.0f32, -0.0, f32::NEG_INFINITY, 2.5e-39];
        encode_row_le(&original, &mut bytes);
        let mut decoded = Vec::new();
        decode_row_le(&bytes, &mut decoded).unwrap();
        for (a, b) in original.iter().zip(&decoded) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert!(decode_row_le(&[1, 2, 3], &mut decoded).is_err());
        assert_eq!(
            sha_u32(&[1, 2, 3]),
            "4636993d3e1da4e9d6b8f87b79e8f7c6d018580d52661950eabc3845c5897a4d"
        );
        assert!(is_sha256_hex(&sha256_hex(b"")));
        assert!(!is_sha256_hex("ABC"));
    }

    #[test]
    fn cell_writer_streams_rows_hashes_and_refuses_reuse() {
        let root = scratch_dir("cellwriter");
        let dir = OutputDir::create_fresh(&root).unwrap();
        let mut writer = CellWriter::create(&dir, "cell-a", 3).unwrap();
        let rows = [[1.0f32, -0.0, 2.5], [0.25, f32::NEG_INFINITY, 4.0]];
        let mut whole = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            let stats = writer
                .push_row(if i == 0 { None } else { Some(9) }, 40 + i, row)
                .unwrap();
            assert_eq!(stats.top1_id, 2);
            let mut one = Vec::new();
            encode_row_le(row, &mut one);
            whole.extend_from_slice(&one);
        }
        // Wrong width and NaN rows are refused before anything is written.
        assert!(writer.push_row(None, 0, &[1.0, 2.0]).is_err());
        assert!(writer.push_row(None, 0, &[1.0, f32::NAN, 2.0]).is_err());
        let out = writer.finish(&[Some(9), None]).unwrap();
        assert_eq!(out.rows, 2);
        assert_eq!(out.bin_bytes, 24);
        let on_disk = std::fs::read(root.join(&out.bin_name)).unwrap();
        assert_eq!(on_disk, whole);
        assert_eq!(out.stream_sha256, sha256_hex(&on_disk));
        let rows_text = std::fs::read_to_string(root.join(&out.rows_name)).unwrap();
        assert_eq!(out.rows_sha256, sha256_hex(rows_text.as_bytes()));
        let lines: Vec<Value> = rows_text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["input_token"], Value::Null);
        assert_eq!(lines[0]["context_position"], 40);
        assert_eq!(lines[0]["next_token"], 9);
        assert_eq!(lines[1]["input_token"], 9);
        assert_eq!(lines[1]["next_token"], Value::Null);
        assert_eq!(lines[1]["neg_inf_count"], 1);
        assert_eq!(lines[1]["bin_offset"], 12);
        assert_eq!(lines[0]["logits_sha256"], sha256_hex(&on_disk[..12]));
        assert_eq!(lines[1]["logits_sha256"], sha256_hex(&on_disk[12..]));
        assert_eq!(lines[0]["top1_logit_bits"], 2.5f32.to_bits());
        // A second writer for the same cell never overwrites.
        assert!(CellWriter::create(&dir, "cell-a", 3).is_err());
        // The next-token list must match the rows.
        let mut short = CellWriter::create(&dir, "cell-b", 3).unwrap();
        short.push_row(None, 0, &rows[0]).unwrap();
        assert!(short.finish(&[]).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    // ── fresh output refusal ──

    #[test]
    fn output_directory_must_be_fresh() {
        let root = scratch_dir("fresh");
        // Absent: created.
        let dir = OutputDir::create_fresh(&root).unwrap();
        // Empty existing directory: accepted.
        OutputDir::create_fresh(&root).unwrap();
        // Files are create_new.
        dir.write_new("a.json", b"{}").unwrap();
        assert!(dir.create_file("a.json").is_err());
        assert!(dir.write_new("a.json", b"{}").is_err());
        assert_eq!(std::fs::read(root.join("a.json")).unwrap(), b"{}");
        // Non-empty directory: refused, and nothing is touched.
        let err = OutputDir::create_fresh(&root).unwrap_err();
        assert!(err.contains("not empty"), "{err}");
        // A file or a relative path: refused.
        let file = root.join("a.json");
        assert!(OutputDir::create_fresh(&file)
            .unwrap_err()
            .contains("not a directory"));
        assert!(OutputDir::create_fresh(Path::new("relative/out")).is_err());
        // Names cannot escape the directory.
        assert!(dir.create_file("../escape").is_err());
        assert!(dir.create_file("").is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    // ── stable KL ──

    fn naive_kl(p_logits: &[f64], q_logits: &[f64]) -> f64 {
        let softmax = |l: &[f64]| -> Vec<f64> {
            let z: f64 = l.iter().map(|x| x.exp()).sum();
            l.iter().map(|x| x.exp() / z).collect()
        };
        let (p, q) = (softmax(p_logits), softmax(q_logits));
        p.iter()
            .zip(&q)
            .filter(|(p, _)| **p > 0.0)
            .map(|(p, q)| p * (p / q).ln())
            .sum()
    }

    #[test]
    fn kl_of_equal_rows_is_zero() {
        let row = [1.0f32, 2.0, 3.0, f32::NEG_INFINITY, -4.5];
        assert_eq!(kl_divergence_f64(&row, &row).unwrap(), 0.0);
        let big = [30000.0f32, 29999.0, 29998.0];
        assert_eq!(kl_divergence_f64(&big, &big).unwrap(), 0.0);
    }

    #[test]
    fn kl_is_invariant_to_an_additive_shift() {
        let r = [0.5f32, 1.5, -2.0, 0.0, 3.25];
        for shift in [4.0f32, -8.0, 1024.0] {
            let c: Vec<f32> = r.iter().map(|x| x + shift).collect();
            let kl = kl_divergence_f64(&r, &c).unwrap();
            assert!(kl.abs() < 1e-12, "shift {shift}: {kl}");
            let kl = kl_divergence_f64(&c, &r).unwrap();
            assert!(kl.abs() < 1e-12, "shift {shift}: {kl}");
        }
        // Shifting only the candidate of a genuinely different pair changes nothing.
        let q = [0.0f32, 1.0, -1.0, 2.0, 0.5];
        let qs: Vec<f32> = q.iter().map(|x| x + 16.0).collect();
        let a = kl_divergence_f64(&r, &q).unwrap();
        let b = kl_divergence_f64(&r, &qs).unwrap();
        assert!((a - b).abs() < 1e-12);
        assert!(a > 0.0);
    }

    #[test]
    fn kl_nontrivial_matches_the_closed_form_and_a_naive_reference() {
        // p = (1/2, 1/4, 1/4), q = (1/4, 1/2, 1/4):
        // KL(p||q) = 1/2 ln 2 + 1/4 ln(1/2) + 0 = ln(2) / 4.
        let p = [0.5f64.ln() as f32, 0.25f64.ln() as f32, 0.25f64.ln() as f32];
        let q = [0.25f64.ln() as f32, 0.5f64.ln() as f32, 0.25f64.ln() as f32];
        let kl = kl_divergence_f64(&p, &q).unwrap();
        assert!((kl - 2f64.ln() / 4.0).abs() < 1e-6, "{kl}");
        // Dyadic logits: compare with an independent naive softmax in f64.
        let rl = [0.0f32, 1.0, -1.0, 2.0, 0.5];
        let ql = [1.0f32, 0.0, -1.0, 0.25, 1.5];
        let want = naive_kl(
            &rl.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
            &ql.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
        );
        let got = kl_divergence_f64(&rl, &ql).unwrap();
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        assert!(got > 0.0);
        // KL is not symmetric.
        let back = kl_divergence_f64(&ql, &rl).unwrap();
        assert!((back - got).abs() > 1e-6);
        // A huge logit gap stays finite and stable.
        let kl = kl_divergence_f64(&[0.0, -50.0], &[-50.0, 0.0]).unwrap();
        assert!((kl - 50.0).abs() < 1e-9, "{kl}");
    }

    #[test]
    fn kl_negative_infinity_is_zero_mass() {
        // Reference mass only on 0 and 2 (the -inf entry contributes nothing).
        let r = [0.0f32, f32::NEG_INFINITY, 0.0];
        let c = [0.0f32, -5.0, 1.0];
        let want = naive_kl(&[0.0, -1e300, 0.0], &[0.0, -5.0, 1.0]);
        let got = kl_divergence_f64(&r, &c).unwrap();
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        assert!(got > 0.0);
        // -inf in the same slot of both rows is a shared zero-mass token.
        let both = kl_divergence_f64(
            &[1.0, f32::NEG_INFINITY, 2.0],
            &[1.5, f32::NEG_INFINITY, 2.0],
        )
        .unwrap();
        let want = naive_kl(&[1.0, -1e300, 2.0], &[1.5, -1e300, 2.0]);
        assert!((both - want).abs() < 1e-12);
        // The candidate losing all its mass where the reference has some is
        // an infinite divergence, not an error and not a fake number.
        assert_eq!(
            kl_divergence_f64(&[0.0, 1.0], &[0.0, f32::NEG_INFINITY]).unwrap(),
            f64::INFINITY
        );
        assert_eq!(
            kl_divergence_f64(&[0.0, 1.0], &[f32::NEG_INFINITY, f32::NEG_INFINITY]).unwrap(),
            f64::INFINITY
        );
        // Mass the candidate puts where the reference has none is not free: it
        // takes probability from the shared support through the candidate's
        // normalizer, so KL = -ln q0 = ln(1 + e^3) for reference (1, 0 mass)
        // against candidate logits (0, 3).
        let extra = kl_divergence_f64(&[0.0, f32::NEG_INFINITY], &[0.0, 3.0]).unwrap();
        let want = (1.0 + 3f64.exp()).ln();
        assert!((extra - want).abs() < 1e-12, "{extra} vs {want}");
        // The same zero-mass slot in both rows costs nothing.
        let same = kl_divergence_f64(&[0.0, f32::NEG_INFINITY], &[0.0, f32::NEG_INFINITY]).unwrap();
        assert!(same.abs() < 1e-12, "{same}");
    }

    #[test]
    fn kl_rejects_nan_positive_infinity_and_zero_support() {
        assert_eq!(
            kl_divergence_f64(&[0.0, f32::NAN], &[0.0, 1.0]),
            Err(KlError::Nan {
                side: "reference",
                index: 1
            })
        );
        assert_eq!(
            kl_divergence_f64(&[0.0, 1.0], &[f32::NAN, 1.0]),
            Err(KlError::Nan {
                side: "candidate",
                index: 0
            })
        );
        assert_eq!(
            kl_divergence_f64(&[f32::INFINITY, 1.0], &[0.0, 1.0]),
            Err(KlError::PositiveInfinity {
                side: "reference",
                index: 0
            })
        );
        assert_eq!(
            kl_divergence_f64(&[0.0, 1.0], &[0.0, f32::INFINITY]),
            Err(KlError::PositiveInfinity {
                side: "candidate",
                index: 1
            })
        );
        assert_eq!(
            kl_divergence_f64(&[f32::NEG_INFINITY, f32::NEG_INFINITY], &[0.0, 1.0]),
            Err(KlError::ZeroSupportReference)
        );
        assert_eq!(kl_divergence_f64(&[], &[]), Err(KlError::Empty));
        assert_eq!(
            kl_divergence_f64(&[0.0, 1.0], &[0.0]),
            Err(KlError::LengthMismatch {
                reference: 2,
                candidate: 1
            })
        );
        // A NaN in a zero-support candidate is still refused.
        assert!(kl_divergence_f64(&[0.0, 1.0], &[f32::NEG_INFINITY, f32::NAN]).is_err());
    }

    // ── actual route / placement / env assertions ──

    fn expect() -> Expect {
        Plan::from_value(sample_plan_value()).unwrap().expect
    }

    fn baseline_facts() -> LoadFacts {
        LoadFacts {
            mode: SlowlinkMode::Off,
            placement: "explicit-env".into(),
            slowlink_profile: false,
            resident_layers: 9,
            host_mapped_tensors: 120,
            scratch_rows: Some(4096),
            profile_tile_rows: None,
            profile_resident_layers: None,
            profile_is_product: None,
            ple_format: "q4".into(),
            gdn_format: GdnStateFormat::Q8,
            qsa_format: QsaKvFormat::Q8,
        }
    }

    fn s1_facts() -> LoadFacts {
        LoadFacts {
            mode: SlowlinkMode::Force,
            placement: "slowlink-profile".into(),
            slowlink_profile: true,
            resident_layers: 9,
            host_mapped_tensors: 120,
            scratch_rows: Some(768),
            profile_tile_rows: Some(768),
            profile_resident_layers: Some(9),
            profile_is_product: Some(true),
            ple_format: "q4".into(),
            gdn_format: GdnStateFormat::Q8,
            qsa_format: QsaKvFormat::Q8,
        }
    }

    #[test]
    fn load_assertions_fail_on_any_unexpected_placement() {
        let e = expect();
        validate_load(Arm::Baseline, false, &baseline_facts(), &e, None).unwrap();
        validate_load(Arm::S1, false, &s1_facts(), &e, None).unwrap();
        let mut facts = s1_facts();
        facts.mode = SlowlinkMode::Auto;
        validate_load(Arm::S1, false, &facts, &e, None).unwrap();
        // The f32 diagnostic loads the exact GDN state with the KV still q8.
        let mut f32_facts = baseline_facts();
        f32_facts.gdn_format = GdnStateFormat::F32;
        validate_load(Arm::Baseline, true, &f32_facts, &e, None).unwrap();
        assert!(validate_load(Arm::Baseline, false, &f32_facts, &e, None)
            .unwrap_err()
            .contains("expected GDN Q8"));
        assert!(
            validate_load(Arm::Baseline, true, &baseline_facts(), &e, None)
                .unwrap_err()
                .contains("expected GDN F32")
        );
        let mut f32_kv = s1_facts();
        f32_kv.qsa_format = QsaKvFormat::F32;
        assert!(validate_load(Arm::S1, false, &f32_kv, &e, None)
            .unwrap_err()
            .contains("QSA K-V Q8"));

        let bad = |arm, mutate: &dyn Fn(&mut LoadFacts)| {
            let mut facts = if arm == Arm::S1 {
                s1_facts()
            } else {
                baseline_facts()
            };
            mutate(&mut facts);
            validate_load(arm, false, &facts, &e, None).unwrap_err()
        };
        assert!(bad(Arm::S1, &|f| f.resident_layers = 8).contains("8 resident layers, expected 9"));
        assert!(bad(Arm::S1, &|f| f.profile_tile_rows = Some(1024)).contains("profile tile"));
        assert!(bad(Arm::S1, &|f| f.scratch_rows = Some(4096)).contains("forward scratch"));
        assert!(bad(Arm::S1, &|f| f.placement = "explicit-env".into()).contains("slowlink-profile"));
        assert!(bad(Arm::S1, &|f| f.slowlink_profile = false).contains("no S1 profile"));
        assert!(bad(Arm::S1, &|f| f.mode = SlowlinkMode::Off).contains("S1 selector"));
        assert!(bad(Arm::S1, &|f| f.profile_is_product = Some(false)).contains("product purpose"));
        assert!(bad(Arm::Baseline, &|f| f.scratch_rows = Some(8192)).contains("expected 4096"));
        assert!(
            bad(Arm::Baseline, &|f| f.placement = "slowlink-profile".into())
                .contains("explicit-env")
        );
        assert!(
            bad(Arm::Baseline, &|f| f.placement = "incumbent-auto".into()).contains("explicit-env")
        );
        assert!(bad(Arm::Baseline, &|f| f.resident_layers = 5)
            .contains("5 resident layers, expected 9"));
        assert!(bad(Arm::Baseline, &|f| f.mode = SlowlinkMode::Force).contains("off (0)"));
        assert!(bad(Arm::Baseline, &|f| {
            f.slowlink_profile = true;
            f.profile_tile_rows = Some(768);
        })
        .contains("attached to the baseline"));
        assert!(bad(Arm::Baseline, &|f| f.host_mapped_tensors = 0).contains("no host-mapped"));
        assert!(bad(Arm::Baseline, &|f| f.ple_format = "q8".into()).contains("PLE rows"));
    }

    fn incumbent_receipt(entry: u64, rows: u64, tile: u64, selected: &str) -> Value {
        json!({
            "schema": "hipfire.qwen4.slowlink_receipt.v1",
            "route": "incumbent",
            "selected_by": selected,
            "bypass_reason": "TooShort { rows: 512, minimum: 1537 }",
            "entry_pos": entry,
            "prompt_rows": rows,
            "tile_rows": tile,
            "retained_scratch_rows": tile,
            "sweeps": 0,
            "resident_layers": 0,
            "host_layers": 0,
        })
    }

    fn layer_major_receipt(entry: u64, rows: u64, selected: &str, layers: u64) -> Value {
        json!({
            "schema": "hipfire.qwen4.slowlink_receipt.v1",
            "route": "layer-major",
            "selected_by": selected,
            "bypass_reason": null,
            "status": "ok",
            "entry_pos": entry,
            "prompt_rows": rows,
            "tile_rows": 768,
            "retained_scratch_rows": 768,
            "sweeps": 4,
            "resident_layers": layers,
            "resident_layer_set": [0, 3, 7, 15, 19, 23, 27, 39, 43],
            "host_layers": 31,
        })
    }

    #[test]
    fn receipts_are_checked_against_the_arm() {
        let e = expect();
        // Baseline: every call incumbent, env0, the 4096 scratch, in place.
        let ok = validate_receipt(
            Arm::Baseline,
            SlowlinkMode::Off,
            &e,
            &incumbent_receipt(4096, 4096, 4096, "env0"),
            4096,
            4096,
            None,
        )
        .unwrap();
        assert_eq!(ok["route"], "incumbent");
        assert_eq!(ok["tile_rows"], 4096);
        let err = validate_receipt(
            Arm::Baseline,
            SlowlinkMode::Off,
            &e,
            &layer_major_receipt(0, 4096, "env0", 9),
            0,
            4096,
            None,
        )
        .unwrap_err();
        assert!(err.contains("expected \"incumbent\""), "{err}");
        let err = validate_receipt(
            Arm::Baseline,
            SlowlinkMode::Off,
            &e,
            &incumbent_receipt(0, 4096, 768, "env0"),
            0,
            4096,
            None,
        )
        .unwrap_err();
        assert!(err.contains("expected 4096"), "{err}");
        let err = validate_receipt(
            Arm::Baseline,
            SlowlinkMode::Off,
            &e,
            &incumbent_receipt(0, 4000, 4096, "env0"),
            0,
            4096,
            None,
        )
        .unwrap_err();
        assert!(err.contains("prompt_rows 4000, expected 4096"), "{err}");
        // S1: a long call must be layer-major at N9 and the 768 tile.
        validate_receipt(
            Arm::S1,
            SlowlinkMode::Force,
            &e,
            &layer_major_receipt(0, 16384, "env1", 9),
            0,
            16384,
            None,
        )
        .unwrap();
        validate_receipt(
            Arm::S1,
            SlowlinkMode::Auto,
            &e,
            &layer_major_receipt(1537, 1863, "auto", 9),
            1537,
            1863,
            None,
        )
        .unwrap();
        let err = validate_receipt(
            Arm::S1,
            SlowlinkMode::Force,
            &e,
            &incumbent_receipt(0, 16384, 768, "env1"),
            0,
            16384,
            None,
        )
        .unwrap_err();
        assert!(err.contains("took the incumbent route"), "{err}");
        let err = validate_receipt(
            Arm::S1,
            SlowlinkMode::Force,
            &e,
            &layer_major_receipt(0, 16384, "env1", 8),
            0,
            16384,
            None,
        )
        .unwrap_err();
        assert!(err.contains("resident_layers 8, expected 9"), "{err}");
        let err = validate_receipt(
            Arm::S1,
            SlowlinkMode::Force,
            &e,
            &layer_major_receipt(0, 16384, "auto", 9),
            0,
            16384,
            None,
        )
        .unwrap_err();
        assert!(
            err.contains("selected_by \"auto\", expected \"env1\""),
            "{err}"
        );
        // A short S1 call may take the incumbent route, at the S1 scratch.
        validate_receipt(
            Arm::S1,
            SlowlinkMode::Force,
            &e,
            &incumbent_receipt(0, 512, 768, "env1"),
            0,
            512,
            None,
        )
        .unwrap();
        // An aborted layer-major receipt is not accepted.
        let mut aborted = layer_major_receipt(0, 2000, "env1", 9);
        aborted["status"] = json!("aborted: boom");
        assert!(
            validate_receipt(Arm::S1, SlowlinkMode::Force, &e, &aborted, 0, 2000, None).is_err()
        );
        // A missing field is an error, never a default.
        let mut missing = layer_major_receipt(0, 2000, "env1", 9);
        missing.as_object_mut().unwrap().remove("tile_rows");
        assert!(
            validate_receipt(Arm::S1, SlowlinkMode::Force, &e, &missing, 0, 2000, None)
                .unwrap_err()
                .contains("tile_rows")
        );
    }

    #[test]
    fn launch_environment_is_asserted() {
        fn env_lookup<'a>(
            map: &'a BTreeMap<&'a str, &'a str>,
            patch: Option<(&'a str, Option<&'a str>)>,
        ) -> impl Fn(&str) -> Option<String> + 'a {
            move |name: &str| match patch {
                Some((patched, value)) if patched == name => value.map(str::to_string),
                _ => map.get(name).map(|v| v.to_string()),
            }
        }
        let e = expect();
        // Baseline routes: class flags + the explicit N9 override + 4096 scratch.
        let mut baseline: BTreeMap<&str, &str> = PINNED_ENV.into_iter().collect();
        baseline.insert(EXPLICIT_LAYERS_ENV, "9");
        baseline.insert(CHUNK_ROWS_ENV, "4096");
        // S1 routes: class flags only; the layer override stays unset.
        let s1: BTreeMap<&str, &str> = PINNED_ENV.into_iter().collect();

        let lookup = env_lookup(&baseline, None);
        let observed = check_env(&e, Arm::Baseline, false, &lookup).unwrap();
        assert_eq!(observed[EXPLICIT_LAYERS_ENV], "9");
        assert_eq!(observed[CHUNK_ROWS_ENV], "4096");
        assert_eq!(observed["HIPFIRE_STATE_QUANT"], "q8");
        assert_eq!(observed["HIPFIRE_QWEN4_D2_DEFAULT"], "0");
        // The baseline needs EXACTLY 9: a wrong count or an unset override is refused.
        for patch in [Some("8"), Some("10"), None] {
            let patched = env_lookup(&baseline, Some((EXPLICIT_LAYERS_ENV, patch)));
            let err = check_env(&e, Arm::Baseline, false, &patched).unwrap_err();
            assert!(err.contains(EXPLICIT_LAYERS_ENV), "{err}");
        }
        let scratch = env_lookup(&baseline, Some((CHUNK_ROWS_ENV, Some("8192"))));
        assert!(check_env(&e, Arm::Baseline, false, &scratch)
            .unwrap_err()
            .contains(CHUNK_ROWS_ENV));

        let s1_lookup = env_lookup(&s1, None);
        let observed = check_env(&e, Arm::S1, false, &s1_lookup).unwrap();
        assert_eq!(observed[EXPLICIT_LAYERS_ENV], Value::Null);
        // Any fixed override declines S1 and is refused on every S1 route,
        // the baseline's own 9 included.
        for value in ["9", "8", "0"] {
            let leaked = env_lookup(&s1, Some((EXPLICIT_LAYERS_ENV, Some(value))));
            for f32_state in [false, true] {
                let err = check_env(&e, Arm::S1, f32_state, &leaked).unwrap_err();
                assert!(err.contains("must be absent"), "{err}");
            }
        }
        assert!(check_env(&e, Arm::S1, false, &lookup)
            .unwrap_err()
            .contains("must be absent"));
        // The S1 environment does not satisfy the baseline routes.
        assert!(check_env(&e, Arm::Baseline, false, &s1_lookup).is_err());

        // f32 diagnostic pair: STATE_QUANT=f32, everything else (KV q8) unchanged.
        let mut baseline_f32 = baseline.clone();
        baseline_f32.insert(STATE_QUANT_ENV, "f32");
        let mut s1_f32 = s1.clone();
        s1_f32.insert(STATE_QUANT_ENV, "f32");
        let beta_lookup = env_lookup(&baseline_f32, None);
        let observed = check_env(&e, Arm::Baseline, true, &beta_lookup).unwrap();
        assert_eq!(observed["HIPFIRE_STATE_QUANT"], "f32");
        assert_eq!(observed["HIPFIRE_KV_MODE"], "q8");
        assert_eq!(observed[EXPLICIT_LAYERS_ENV], "9");
        let s1_f32_lookup = env_lookup(&s1_f32, None);
        let observed = check_env(&e, Arm::S1, true, &s1_f32_lookup).unwrap();
        assert_eq!(observed["HIPFIRE_STATE_QUANT"], "f32");
        assert_eq!(observed[EXPLICIT_LAYERS_ENV], Value::Null);
        // A q8 process is refused in an f32 phase and the reverse.
        assert!(check_env(&e, Arm::Baseline, true, &lookup)
            .unwrap_err()
            .contains("HIPFIRE_STATE_QUANT"));
        assert!(check_env(&e, Arm::S1, false, &s1_f32_lookup)
            .unwrap_err()
            .contains("HIPFIRE_STATE_QUANT"));
        // The other pinned flags are refused when wrong or unset.
        let wrong = env_lookup(&baseline, Some(("HIPFIRE_QWEN4_D2_DEFAULT", Some("1"))));
        assert!(check_env(&e, Arm::Baseline, false, &wrong)
            .unwrap_err()
            .contains("expected \"0\""));
        let unset = |_: &str| None;
        assert!(check_env(&e, Arm::Baseline, false, &unset).is_err());
        assert!(check_env(&e, Arm::S1, false, &unset).is_err());
    }

    #[test]
    fn arm_selector_pairing() {
        check_mode(Arm::Baseline, SlowlinkMode::Off).unwrap();
        check_mode(Arm::S1, SlowlinkMode::Force).unwrap();
        check_mode(Arm::S1, SlowlinkMode::Auto).unwrap();
        assert!(check_mode(Arm::Baseline, SlowlinkMode::Auto).is_err());
        assert!(check_mode(Arm::S1, SlowlinkMode::Off).is_err());
    }

    // ── manifest and model identity ──

    fn manifest_json(plan: &Plan, exe_sha: &str) -> Value {
        json!({
            "schema": MANIFEST_SCHEMA,
            "binary": {"sha256": exe_sha, "path": "/bin/x"},
            "source": {"commit": "40a4e3bee", "worktree": "/wt"},
            "launch": {"command": "x"},
            "model": {
                "path": plan.artifact.path,
                "dev": 64769u64, "ino": 1234u64, "size": plan.artifact.size,
                "mtime_sec": 1_700_000_000i64, "mtime_nsec": 123_456_789i64,
                "sha256": plan.artifact.sha256,
            },
        })
    }

    #[test]
    fn manifest_and_model_identity_tuple() {
        let plan = Plan::from_value(sample_plan_value()).unwrap();
        let exe = sha256_hex(b"exe");
        let bytes = serde_json::to_vec(&manifest_json(&plan, &exe)).unwrap();
        let manifest = Manifest::from_slice(&bytes, &plan.artifact).unwrap();
        assert_eq!(manifest.binary_sha256().unwrap(), exe);
        let stat = ModelStat {
            dev: 64769,
            ino: 1234,
            size: plan.artifact.size,
            mtime_sec: 1_700_000_000,
            mtime_nsec: 123_456_789,
            ctime_sec: 1_800_000_000,
            ctime_nsec: 5,
        };
        verify_model_stat(&manifest.model, &stat).unwrap();
        // ctime is metadata only: a different ctime is accepted.
        let mut other_ctime = stat.clone();
        other_ctime.ctime_sec += 99;
        verify_model_stat(&manifest.model, &other_ctime).unwrap();
        // Device, inode, size and mtime each refuse.
        for mutate in [
            (|s: &mut ModelStat| s.dev += 1) as fn(&mut ModelStat),
            |s| s.ino += 1,
            |s| s.size += 1,
            |s| s.mtime_sec += 1,
            |s| s.mtime_nsec += 1,
        ] {
            let mut changed = stat.clone();
            mutate(&mut changed);
            assert!(verify_model_stat(&manifest.model, &changed)
                .unwrap_err()
                .contains("differs from the trusted manifest"));
        }
        // The manifest's pinned artifact must be the plan's.
        let mut wrong = manifest_json(&plan, &exe);
        wrong["model"]["sha256"] = json!(sha256_hex(b"other"));
        assert!(
            Manifest::from_slice(&serde_json::to_vec(&wrong).unwrap(), &plan.artifact)
                .unwrap_err()
                .contains("differs from plan.artifact")
        );
        let mut wrong = manifest_json(&plan, &exe);
        wrong["model"]["path"] = json!("/elsewhere/m.hfq");
        assert!(
            Manifest::from_slice(&serde_json::to_vec(&wrong).unwrap(), &plan.artifact).is_err()
        );
        let mut wrong = manifest_json(&plan, "nothex");
        assert!(
            Manifest::from_slice(&serde_json::to_vec(&wrong).unwrap(), &plan.artifact)
                .unwrap_err()
                .contains("binary.sha256")
        );
        wrong = manifest_json(&plan, &exe);
        wrong["source"] = json!({});
        assert!(
            Manifest::from_slice(&serde_json::to_vec(&wrong).unwrap(), &plan.artifact).is_err()
        );
        wrong = manifest_json(&plan, &exe);
        wrong["extra"] = json!(1);
        assert!(
            Manifest::from_slice(&serde_json::to_vec(&wrong).unwrap(), &plan.artifact).is_err()
        );
    }

    fn recipe_manifest(plan: &Plan, exe_sha: &str, layers: usize, set: Value) -> Value {
        let mut manifest = manifest_json(plan, exe_sha);
        manifest["source"] = json!({
            "commit": "40a4e3bee",
            "worktree": "/wt",
            "execution_recipe": {
                "schema": EXECUTION_RECIPE_SCHEMA,
                "recipe_sha256": sha256_hex(b"recipe"),
                "device": {"arch": "gfx1201"},
                "source": {"origin": "sealed"},
                "verified_cache": {
                    "home": "/home/u/.hipfire",
                    "path": "/home/u/.hipfire/cache/verified-artifacts.json",
                    "sha256": sha256_hex(b"cache"),
                    "entries": [],
                },
                "expect": {
                    "resident_layers": layers,
                    "resident_layer_set": set,
                    "placement": "slowlink-profile",
                },
                "resources": {},
            },
        });
        manifest
    }

    #[test]
    fn sealed_recipe_gates_plan_count_and_receipt_set() {
        let plan = Plan::from_value(sample_plan_value()).unwrap();
        let exe = sha256_hex(b"exe");
        // No recipe: the untouched original N9 replay path.
        let bytes = serde_json::to_vec(&manifest_json(&plan, &exe)).unwrap();
        let manifest = Manifest::from_slice(&bytes, &plan.artifact).unwrap();
        assert!(parse_execution_recipe(&manifest).unwrap().is_none());
        check_plan_against_recipe(&plan, None).unwrap();
        // A genuine planned count seals: plan == recipe == actual.
        let sealed_value = recipe_manifest(&plan, &exe, 9, Value::Null);
        let bytes = serde_json::to_vec(&sealed_value).unwrap();
        let sealed_manifest = Manifest::from_slice(&bytes, &plan.artifact).unwrap();
        let sealed = parse_execution_recipe(&sealed_manifest).unwrap().unwrap();
        assert_eq!(sealed.resident_layers, 9);
        assert!(sealed.resident_layer_set.is_none());
        check_plan_against_recipe(&plan, Some(&sealed)).unwrap();
        // An arbitrary count never seals: the candidate plan must equal it.
        let other_value = recipe_manifest(&plan, &exe, 7, Value::Null);
        let bytes = serde_json::to_vec(&other_value).unwrap();
        let other_manifest = Manifest::from_slice(&bytes, &plan.artifact).unwrap();
        let other = parse_execution_recipe(&other_manifest).unwrap().unwrap();
        assert!(check_plan_against_recipe(&plan, Some(&other))
            .unwrap_err()
            .contains("requires 7"));
        // An explicit set pins the layer-major receipts exactly.
        let spread = vec![0, 3, 7, 15, 19, 23, 27, 39, 43];
        let set_value = recipe_manifest(&plan, &exe, 9, json!(spread));
        let bytes = serde_json::to_vec(&set_value).unwrap();
        let set_manifest = Manifest::from_slice(&bytes, &plan.artifact).unwrap();
        let set_sealed = parse_execution_recipe(&set_manifest).unwrap().unwrap();
        assert_eq!(set_sealed.resident_layer_set, Some(spread.clone()));
        let receipt = layer_major_receipt(0, 16384, "env1", 9);
        let e = expect();
        validate_receipt(
            Arm::S1,
            SlowlinkMode::Force,
            &e,
            &receipt,
            0,
            16384,
            Some(&set_sealed),
        )
        .unwrap();
        let mut wrong_set = receipt.clone();
        wrong_set["resident_layer_set"] = json!([0, 1, 2]);
        assert!(validate_receipt(
            Arm::S1,
            SlowlinkMode::Force,
            &e,
            &wrong_set,
            0,
            16384,
            Some(&set_sealed)
        )
        .unwrap_err()
        .contains("differs from the sealed recipe"));
        // A recipe layer-major receipt with no set is refused, never defaulted.
        let mut bare = layer_major_receipt(0, 16384, "env1", 9);
        bare.as_object_mut().unwrap().remove("resident_layer_set");
        assert!(validate_receipt(
            Arm::S1,
            SlowlinkMode::Force,
            &e,
            &bare,
            0,
            16384,
            Some(&set_sealed)
        )
        .unwrap_err()
        .contains("no resident_layer_set array"));
        // The load itself must equal the sealed count, not the old literal.
        let mut facts = s1_facts();
        facts.resident_layers = 7;
        assert!(validate_load(Arm::S1, false, &facts, &e, Some(&set_sealed))
            .unwrap_err()
            .contains("requires 9"));
    }

    #[test]
    fn stat_model_reads_a_real_file_and_refuses_a_directory() {
        let root = scratch_dir("stat");
        let dir = OutputDir::create_fresh(&root).unwrap();
        dir.write_new("m.bin", b"0123456789").unwrap();
        let stat = stat_model(&root.join("m.bin")).unwrap();
        assert_eq!(stat.size, 10);
        assert_eq!(stat, stat_model(&root.join("m.bin")).unwrap());
        assert!(stat_model(&root).is_err());
        assert!(stat_model(&root.join("missing")).is_err());
        assert_eq!(
            file_sha256(&root.join("m.bin")).unwrap(),
            sha256_hex(b"0123456789")
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    // ── command line ──

    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn command_line_per_phase() {
        let gpu = parse_args(argv(&[
            "--phase",
            "s1a",
            "--plan",
            "/p.json",
            "--manifest",
            "/m.json",
            "--model",
            "/x.hfq",
            "--out-dir",
            "/out",
        ]))
        .unwrap();
        assert_eq!(gpu.phase, Phase::S1a);
        assert_eq!(gpu.phase.arm(), Some(Arm::S1));
        assert_eq!(Phase::Baseline.arm(), Some(Arm::Baseline));
        assert!(Phase::Score.arm().is_none());
        assert!(Phase::ScoreF32.arm().is_none());
        assert_eq!(Phase::parse("score-f32").unwrap(), Phase::ScoreF32);
        parse_args(argv(&[
            "--phase",
            "score-f32",
            "--plan",
            "/p.json",
            "--beta-f32-dir",
            "/bf",
            "--s1-f32-dir",
            "/sf",
            "--out-dir",
            "/o",
        ]))
        .unwrap();
        // score-f32 needs both f32 directories and takes no q8 directory.
        assert!(parse_args(argv(&[
            "--phase",
            "score-f32",
            "--plan",
            "/p.json",
            "--beta-f32-dir",
            "/bf",
            "--out-dir",
            "/o"
        ]))
        .unwrap_err()
        .contains("--s1-f32-dir is required"));
        assert!(parse_args(argv(&[
            "--phase",
            "score-f32",
            "--plan",
            "/p.json",
            "--beta-f32-dir",
            "/bf",
            "--s1-f32-dir",
            "/sf",
            "--out-dir",
            "/o",
            "--baseline-dir",
            "/b"
        ]))
        .unwrap_err()
        .contains("does not apply"));
        assert!(parse_args(argv(&[
            "--phase",
            "score-f32",
            "--plan",
            "/p.json",
            "--beta-f32-dir",
            "/bf",
            "--s1-f32-dir",
            "/sf",
            "--out-dir",
            "/o",
            "--s1a-dir",
            "/a"
        ]))
        .unwrap_err()
        .contains("does not apply"));
        parse_args(argv(&["--phase", "check-plan", "--plan", "/p.json"])).unwrap();
        parse_args(argv(&[
            "--phase",
            "beta-f32",
            "--plan",
            "/p.json",
            "--manifest",
            "/m.json",
            "--model",
            "/x.hfq",
            "--out-dir",
            "/out",
        ]))
        .unwrap();
        assert_eq!(Phase::parse("s1-f32").unwrap(), Phase::S1F32);
        assert_eq!(Phase::S1F32.arm(), Some(Arm::S1));
        assert_eq!(Phase::BetaF32.arm(), Some(Arm::Baseline));
        assert!(Phase::BetaF32.state_f32() && Phase::S1F32.state_f32());
        assert!(!Phase::Baseline.state_f32() && !Phase::S1a.state_f32());
        parse_args(argv(&[
            "--phase",
            "score",
            "--plan",
            "/p.json",
            "--baseline-dir",
            "/b",
            "--s1a-dir",
            "/a",
            "--beta-f32-dir",
            "/bf",
            "--s1-f32-dir",
            "/sf",
            "--out-dir",
            "/o",
        ]))
        .unwrap();
        assert!(parse_args(argv(&[
            "--phase",
            "s1-f32",
            "--plan",
            "/p.json",
            "--manifest",
            "/m.json",
            "--model",
            "/x",
            "--out-dir",
            "/o",
            "--beta-f32-dir",
            "/bf"
        ]))
        .unwrap_err()
        .contains("does not apply"));
        parse_args(argv(&[
            "--phase",
            "score",
            "--plan",
            "/p.json",
            "--baseline-dir",
            "/b",
            "--s1a-dir",
            "/a",
            "--s1b-dir",
            "/c",
            "--out-dir",
            "/o",
        ]))
        .unwrap();
        // Missing, relative and inapplicable flags are refused.
        assert!(
            parse_args(argv(&["--phase", "baseline", "--plan", "/p.json"]))
                .unwrap_err()
                .contains("--manifest is required")
        );
        assert!(
            parse_args(argv(&["--phase", "check-plan", "--plan", "p.json"]))
                .unwrap_err()
                .contains("absolute")
        );
        assert!(parse_args(argv(&[
            "--phase",
            "check-plan",
            "--plan",
            "/p.json",
            "--out-dir",
            "/o"
        ]))
        .unwrap_err()
        .contains("does not apply"));
        assert!(parse_args(argv(&[
            "--phase",
            "s1b",
            "--plan",
            "/p.json",
            "--manifest",
            "/m",
            "--model",
            "/x",
            "--out-dir",
            "/o",
            "--s1a-dir",
            "/a"
        ]))
        .unwrap_err()
        .contains("does not apply"));
        assert!(parse_args(argv(&["--phase", "bogus", "--plan", "/p.json"])).is_err());
        assert!(parse_args(argv(&["--plan", "/p.json"]))
            .unwrap_err()
            .contains("--phase"));
        assert!(parse_args(argv(&["--phase", "s1a", "--wat", "x"]))
            .unwrap_err()
            .contains("unknown"));
        assert!(USAGE.contains("--phase baseline|s1a|s1b"));
    }

    // ── stream/row verification used by the scorer ──

    #[test]
    fn scorer_pairs_dumped_rows_end_to_end() {
        // Two real dumps written by the CellWriter, scored by the same code
        // path the score phase uses.
        let root = scratch_dir("score");
        let dir_a = OutputDir::create_fresh(&root.join("a")).unwrap();
        let dir_b = OutputDir::create_fresh(&root.join("b")).unwrap();
        let reference = [[1.0f32, 2.0, 3.0, f32::NEG_INFINITY], [0.0, 1.0, 0.5, 0.25]];
        let candidate = [[1.0f32, 2.0, 3.0, f32::NEG_INFINITY], [0.5, 1.0, 0.0, 0.25]];
        let mut records = Vec::new();
        for (dir, rows) in [(&dir_a, &reference), (&dir_b, &candidate)] {
            let mut writer = CellWriter::create(dir, "pair-cell", 4).unwrap();
            for (i, row) in rows.iter().enumerate() {
                writer
                    .push_row(if i == 0 { None } else { Some(7) }, 20 + i, row)
                    .unwrap();
            }
            let out = writer.finish(&[Some(7), None]).unwrap();
            let cell = json!({
                "cell": "pair-cell", "vocab": 4, "rows": out.rows,
                "bin": out.bin_name, "bin_bytes": out.bin_bytes,
                "rows_file": out.rows_name, "rows_file_sha256": out.rows_sha256,
                "stream_sha256": out.stream_sha256,
                "prompt_sha256": "p", "teacher_sha256": "t",
            });
            records.push(cell);
        }
        let index = |dir: &OutputDir, cell: &Value| CellIndex {
            dir: dir.root.clone(),
            cells: [("pair-cell".to_string(), cell.clone())]
                .into_iter()
                .collect(),
        };
        let sources = Sources {
            baseline: Some(index(&dir_a, &records[0])),
            s1a: Some(index(&dir_b, &records[1])),
            beta_f32: None,
            s1_f32: None,
        };
        let spec = PairSpec {
            name: "t".into(),
            reference: (Side::Baseline, "pair-cell".into()),
            candidate: (Side::S1a, "pair-cell".into()),
            expected_rows: 2,
        };
        let mut sink: Vec<u8> = Vec::new();
        let summary = score_pair(&spec, &sources, &mut sink).unwrap();
        assert_eq!(summary["rows"], 2);
        assert_eq!(summary["complete"], true);
        assert_eq!(summary["infinite_rows"], 0);
        assert_eq!(summary["top1_agree_rows"], 2);
        let lines: Vec<Value> = String::from_utf8(sink)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["kl_nats"], 0.0);
        let want = kl_divergence_f64(&reference[1], &candidate[1]).unwrap();
        assert_eq!(lines[1]["kl_nats"].as_f64().unwrap(), want);
        assert!(want > 0.0);
        assert_eq!(lines[1]["context_position"], 21);
        // Corrupting one byte of a dump is caught by the row hash.
        let bin = root.join("b").join("pair-cell.logits.f32le");
        let mut bytes = std::fs::read(&bin).unwrap();
        bytes[0] ^= 1;
        std::fs::write(&bin, &bytes).unwrap();
        let mut sink: Vec<u8> = Vec::new();
        let err = score_pair(&spec, &sources, &mut sink).unwrap_err();
        assert!(err.contains("does not match its recorded hash"), "{err}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn score_f32_scores_only_the_f32_pair_without_q8_directories() {
        let root = scratch_dir("scoref32");
        let plan = Plan::from_value(sample_plan_value()).unwrap();
        let spec = f32_pair_spec(&plan);
        assert_eq!(spec.name, "resume-f32-s1split-vs-betacold");
        assert_eq!(spec.expected_rows, RESUME_TEACHER_IDS + 1);
        assert_eq!(spec.reference.1, RESUME_COLD);
        assert_eq!(spec.candidate.1, RESUME_SPLIT);
        // The full score keeps the same pair, unchanged, when the f32 dirs are given.
        let full = pair_specs(&plan, true);
        assert_eq!(
            full.last().map(|p| p.name.as_str()),
            Some(spec.name.as_str())
        );
        assert!(pair_specs(&plan, false).iter().all(|p| p.name != spec.name));

        let rows = RESUME_TEACHER_IDS + 1;
        let mut indexes = Vec::new();
        for (arm, cell, bump) in [("beta", RESUME_COLD, 0.0f32), ("s1", RESUME_SPLIT, 0.25)] {
            let dir = OutputDir::create_fresh(&root.join(arm)).unwrap();
            let mut writer = CellWriter::create(&dir, cell, 4).unwrap();
            for r in 0..rows {
                let row = [1.0f32 + r as f32 * 0.5, 2.0 + bump, 0.5, f32::NEG_INFINITY];
                writer
                    .push_row(if r == 0 { None } else { Some(7) }, 3400 + r, &row)
                    .unwrap();
            }
            let out = writer.finish(&plan.resume.next_tokens()).unwrap();
            let record = json!({
                "cell": cell, "vocab": 4, "rows": out.rows,
                "bin": out.bin_name, "bin_bytes": out.bin_bytes,
                "rows_file": out.rows_name, "rows_file_sha256": out.rows_sha256,
                "stream_sha256": out.stream_sha256,
                "prompt_sha256": "p", "teacher_sha256": "t",
            });
            indexes.push(CellIndex {
                dir: dir.root.clone(),
                cells: [(cell.to_string(), record)].into_iter().collect(),
            });
        }
        let s1_f32 = indexes.pop().unwrap();
        let beta_f32 = indexes.pop().unwrap();
        let sources = Sources {
            baseline: None,
            s1a: None,
            beta_f32: Some(beta_f32),
            s1_f32: Some(s1_f32),
        };
        let mut sink: Vec<u8> = Vec::new();
        let summary = score_pair(&spec, &sources, &mut sink).unwrap();
        assert_eq!(summary["rows"], rows);
        assert_eq!(summary["complete"], true);
        assert_eq!(summary["infinite_rows"], 0);
        assert!(summary["kl_mean_nats"].as_f64().unwrap() > 0.0);
        assert_eq!(String::from_utf8(sink).unwrap().lines().count(), rows);
        // Any q8 side is reported as not given, never silently recomputed.
        assert!(sources
            .pick(Side::Baseline)
            .unwrap_err()
            .contains("--baseline-dir was not given"));
        assert!(sources
            .pick(Side::S1a)
            .unwrap_err()
            .contains("--s1a-dir was not given"));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// SHA-256 of the retained j1968 `greedy-s1/receipts.jsonl` the fixtures
    /// below were taken from (`holds/collected/hold1b-r8-40a4e3bee-56bafa4e6-j1968/
    /// OUT/greedy-s1/receipts.jsonl`, ids `resume` calls 0 and 1, and `p1536`):
    /// the actual producer's JSON, every key and value as retained, with only the
    /// run-to-run timing values trimmed to what each record held.
    const J1968_S1_RECEIPTS_SHA256: &str =
        "e5e5818c88dda727c97c2f8a003a99d1bd0a4be997c5bcebfcf14905dfe63b3f";

    /// j1968 `resume` call 0: the layer-major 1537-row prefix (entry 0).
    fn real_resume_first() -> Value {
        json!({
            "schema": "hipfire.qwen4.slowlink_receipt.v1",
            "route": "layer-major", "selected_by": "env1", "bypass_reason": null,
            "status": "ok", "entry_pos": 0, "prompt_rows": 1537,
            "windows": [{"start": 0, "rows": 1537}],
            "launch_cuts": [768, 1536, 1537], "session_cuts": [],
            "tile_rows": 768, "retained_scratch_rows": 768, "sweeps": 1,
            "resident_layers": 9, "host_layers": 39,
            "host_expert_bytes_total": 52140441600u64, "submitted_bytes": 52140441600u64,
            "useful_bytes": 20101017600u64, "unused_prefetched_bytes": 32039424000u64,
            "discarded_group_payload_bytes": 0,
            "reserve": {"tile": 947662596u64, "arena": 887535872u64, "ring": 167190528u64,
                        "lookahead": 1337524224u64, "tables": 147456u64, "total": 3476572740u64},
            "spans_ms": {"wall": 7608.303967999999, "copy": 7547.773854690138,
                         "compute": 229.99508666992188, "route_barrier": 552.874143, "nonexpert": 0.0},
            "overlap": {"cross_layer_intersections": 38, "cross_layer_overlap_ms": 47.222015380859375,
                        "copy_idle_ms": 57.441673278808594, "page_full_stalls": 646,
                        "copy_starved_stalls": 595},
            "timeline_overflow": 0, "peak_vram_bytes": 23024631808u64,
            "vram_measurement": {"method": "hipMemGetInfo_post_request_sample", "continuous_peak": false},
            "injection_consumed": false,
        })
    }

    /// j1968 `resume` call 1: the layer-major resumed 1863 rows (entry 1537).
    fn real_resume_second() -> Value {
        let mut receipt = real_resume_first();
        let map = receipt.as_object_mut().unwrap();
        map.insert("entry_pos".into(), json!(1537));
        map.insert("prompt_rows".into(), json!(1863));
        map.insert("windows".into(), json!([{"start": 0, "rows": 1863}]));
        map.insert("launch_cuts".into(), json!([768, 1536, 1863]));
        map.insert("useful_bytes".into(), json!(20751206400u64));
        map.insert("unused_prefetched_bytes".into(), json!(31389235200u64));
        map.insert("peak_vram_bytes".into(), json!(23033020416u64));
        map.insert(
            "spans_ms".into(),
            json!({"wall": 7530.319555, "copy": 7430.3494803509675,
                   "compute": 838.3013000488281, "route_barrier": 939.218257, "nonexpert": 0.0}),
        );
        map.insert(
            "overlap".into(),
            json!({"cross_layer_intersections": 40, "cross_layer_overlap_ms": 148.23165893554688,
                   "copy_idle_ms": 95.24016761779785, "page_full_stalls": 646,
                   "copy_starved_stalls": 556}),
        );
        receipt
    }

    /// j1968 `p1536`: the one-row-short bypass (an incumbent receipt: no `status`,
    /// no `session_cuts`, `bypass_reason` TooShort, S1 scratch of 768 rows).
    fn real_p1536_bypass() -> Value {
        json!({
            "schema": "hipfire.qwen4.slowlink_receipt.v1",
            "route": "incumbent", "selected_by": "env1",
            "bypass_reason": "TooShort { rows: 1536, minimum: 1537 }",
            "entry_pos": 0, "prompt_rows": 1536,
            "windows": [{"start": 0, "rows": 1536}], "launch_cuts": [1536],
            "tile_rows": 768, "retained_scratch_rows": 768, "sweeps": 0,
            "resident_layers": 0, "host_layers": 0, "host_expert_bytes_total": 0,
            "submitted_bytes": 0, "useful_bytes": 0, "unused_prefetched_bytes": 0,
            "reserve": {"tile": 0, "arena": 0, "ring": 0, "lookahead": 0, "tables": 0, "total": 0},
            "spans_ms": {"wall": 0.0, "copy": 0.0, "compute": 0.0, "route_barrier": 0.0, "nonexpert": 0.0},
            "overlap": {"cross_layer_intersections": 0, "cross_layer_overlap_ms": 0.0,
                        "copy_idle_ms": 0.0, "page_full_stalls": 0, "copy_starved_stalls": 0},
            "peak_vram_bytes": 0, "injection_consumed": false,
        })
    }

    #[test]
    fn actual_j1968_s1_receipts_pass_the_s1_checks() {
        assert!(is_sha256_hex(J1968_S1_RECEIPTS_SHA256));
        let e = expect();
        // The retained one-layer-major 1537 prefix and the resumed 1863 (entry 1537).
        let first = validate_receipt(
            Arm::S1,
            SlowlinkMode::Force,
            &e,
            &real_resume_first(),
            0,
            1537,
            None,
        )
        .unwrap();
        assert_eq!(first["route"], "layer-major");
        assert_eq!(first["status"], "ok");
        assert_eq!(first["tile_rows"], 768);
        assert_eq!(first["resident_layers"], 9);
        let second = validate_receipt(
            Arm::S1,
            SlowlinkMode::Force,
            &e,
            &real_resume_second(),
            1537,
            1863,
            None,
        )
        .unwrap();
        assert_eq!(second["entry_pos"], 1537);
        assert_eq!(second["prompt_rows"], 1863);
        assert_eq!(second["host_layers"], 39);
        // The two calls are not interchangeable: entry and rows are checked.
        assert!(validate_receipt(
            Arm::S1,
            SlowlinkMode::Force,
            &e,
            &real_resume_second(),
            0,
            1863,
            None,
        )
        .unwrap_err()
        .contains("entry_pos 1537, expected 0"));
        assert!(validate_receipt(
            Arm::S1,
            SlowlinkMode::Force,
            &e,
            &real_resume_first(),
            0,
            1536,
            None,
        )
        .unwrap_err()
        .contains("prompt_rows 1537, expected 1536"));
        // Retained under selector `1` (env1): an `auto` process is refused.
        assert!(validate_receipt(
            Arm::S1,
            SlowlinkMode::Auto,
            &e,
            &real_resume_first(),
            0,
            1537,
            None,
        )
        .unwrap_err()
        .contains("selected_by \"env1\", expected \"auto\""));
        // The short bypass: accepted below the 1537 minimum, with its reason kept.
        let bypass = validate_receipt(
            Arm::S1,
            SlowlinkMode::Force,
            &e,
            &real_p1536_bypass(),
            0,
            1536,
            None,
        )
        .unwrap();
        assert_eq!(bypass["route"], "incumbent");
        assert_eq!(bypass["status"], Value::Null);
        assert_eq!(
            bypass["bypass_reason"],
            "TooShort { rows: 1536, minimum: 1537 }"
        );
        assert_eq!(bypass["tile_rows"], 768);
        // ...but the same bypass on a call that must be layer-major is refused.
        let mut long = real_p1536_bypass();
        long["prompt_rows"] = json!(1537);
        assert!(
            validate_receipt(Arm::S1, SlowlinkMode::Force, &e, &long, 0, 1537, None)
                .unwrap_err()
                .contains("took the incumbent route")
        );
        // S1 receipts never satisfy the baseline arm (route, selector, 4096 scratch).
        let err = validate_receipt(
            Arm::Baseline,
            SlowlinkMode::Off,
            &e,
            &real_resume_first(),
            0,
            1537,
            None,
        )
        .unwrap_err();
        assert!(err.contains("expected \"incumbent\""), "{err}");
        assert!(
            err.contains("selected_by \"env1\", expected \"env0\""),
            "{err}"
        );
        assert!(err.contains("expected 4096"), "{err}");
    }

    /// The baseline (selector off) receipt exactly as the producer
    /// (`incumbent_receipt_json`, via `Qwen4Bundle::slowlink_receipt`) builds it:
    /// `selected_by` env0, `bypass_reason` null (a kill switch is "not asked",
    /// not a refusal), the forward's scratch rows as the tile, no `status`, and
    /// the stored incumbent window `(0, rows)` relative to the call's entry.
    fn producer_incumbent_off(entry: u64, rows: u64, tile: u64) -> Value {
        json!({
            "schema": "hipfire.qwen4.slowlink_receipt.v1",
            "route": "incumbent", "selected_by": "env0", "bypass_reason": null,
            "entry_pos": entry, "prompt_rows": rows,
            "windows": [{"start": 0, "rows": rows}], "launch_cuts": [rows],
            "tile_rows": tile, "retained_scratch_rows": tile,
            "sweeps": 0, "resident_layers": 0, "host_layers": 0,
            "host_expert_bytes_total": 0, "submitted_bytes": 0, "useful_bytes": 0,
            "unused_prefetched_bytes": 0,
            "reserve": {"tile": 0, "arena": 0, "ring": 0, "lookahead": 0, "tables": 0, "total": 0},
            "spans_ms": {"wall": 0.0, "copy": 0.0, "compute": 0.0, "route_barrier": 0.0, "nonexpert": 0.0},
            "overlap": {"cross_layer_intersections": 0, "cross_layer_overlap_ms": 0.0,
                        "copy_idle_ms": 0.0, "page_full_stalls": 0, "copy_starved_stalls": 0},
            "peak_vram_bytes": 0, "injection_consumed": false,
        })
    }

    #[test]
    fn baseline_mode_off_receipts_carry_no_profile_and_pass_the_baseline_checks() {
        let e = expect();
        // The baseline cold 3400 call, the split calls and the 4096-row chunks.
        for (entry, rows) in [
            (0u64, 3400u64),
            (0, 1537),
            (1537, 1863),
            (0, 4096),
            (4096, 4096),
            (12288, 4096),
            (4096, 1),
        ] {
            validate_receipt(
                Arm::Baseline,
                SlowlinkMode::Off,
                &e,
                &producer_incumbent_off(entry, rows, 4096),
                entry as usize,
                rows as usize,
                None,
            )
            .unwrap();
        }
        // No profile fields are needed or consulted: the receipt has no status
        // and zero S1 counters, and the validation still names the scratch.
        let off = producer_incumbent_off(0, 3400, 4096);
        assert!(off.get("status").is_none());
        assert_eq!(off["resident_layers"], 0);
        let mut small = producer_incumbent_off(0, 3400, 768);
        small["retained_scratch_rows"] = json!(768);
        assert!(
            validate_receipt(Arm::Baseline, SlowlinkMode::Off, &e, &small, 0, 3400, None)
                .unwrap_err()
                .contains("expected 4096")
        );
        // A baseline receipt is refused on an S1 route even at the S1 scratch.
        assert!(validate_receipt(
            Arm::S1,
            SlowlinkMode::Force,
            &e,
            &producer_incumbent_off(0, 3400, 768),
            0,
            3400,
            None,
        )
        .unwrap_err()
        .contains("selected_by \"env0\", expected \"env1\""));
    }
}
