// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! R4 driver: the search phase machine over `pm_loop` subprocesses.
//!
//! `pm-census tune plan` prints the [`DomainV1`] of one corpus node;
//! `pm-census tune run` measures it. The tuner never edits kernel source and
//! never links a GPU: every measurement is a fresh `pm_loop` process
//! (`BIN <candidate args> --time 50 --rounds 6 --max-rounds R --time-budget-s S
//! --out DIR/obj [phase flags]`), whose stdout JSON record is kept verbatim in
//! `DIR/records/<phase>-<candidate_id>-<replicate>.json`.
//!
//! Phases, each gated on the previous one:
//!
//! 0. `offline` (`--offline`, no device): the incumbent and every domain
//!    candidate are emitted and M7-certified. A candidate that exits non-zero,
//!    reports an `error.stage`, or whose resolved baseline is not the corpus
//!    node's launch symbol is `pruned` (reason kept). The rest are classified by
//!    the code bytes of the launched function: `code_identical_control` (equal to
//!    the shipped incumbent's), `duplicate_of:<id>` (equal to an earlier
//!    candidate's) or `search`. If the *incumbent* spec itself is unusable or
//!    resolves another baseline the whole run is refused.
//! 1. `aa`: the incumbent with `--null-control aa`, `--aa N` fresh processes;
//!    `crate::noise::analyze` turns them into the noise floor
//!    (`DIR/noise.json`, passed to every later phase).
//! 2. `discovery`: every `search` and `code_identical_control` candidate once.
//! 3. `confirm`: every search candidate whose discovery passes the m7,
//!    correctness, abba and noise-floor gates is re-measured in `--confirm N`
//!    further fresh processes. A winner needs `promote::is_winner`.
//!
//! The code-identical control is the negative control: it emits the incumbent's
//! own bytes, so a separable *and* material discovery for it means the harness
//! (placement, order, drift) manufactures effects, and every winner of the run
//! is void.
//!
//! Resume: an invocation whose record file already parses as JSON is reused, not
//! re-run (ledger `reused: true`; reused work consumes no budget). Budgets stop
//! launching new processes (`--max-processes`, `--max-wall-s`) and `stop` names
//! the phase that was cut.

use super::{
    domain, promote, store, CandidateSpecV1, ClaimPolicyV1, DomainV1, Refusal, TuneEvidenceV1,
    TuneKeyV1, TuneRecordV1, RECORD_SCHEMA,
};
use hipfire_atlas::workload::WorkloadCorpusV1;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const RUN_SCHEMA: &str = "hipfire.tune_run.v1";

const USAGE: &str =
    "usage: pm-census tune plan --corpus FILE --case ID_PREFIX --node N --tokens T\n\
       pm-census tune run --corpus FILE --case ID_PREFIX --node N --tokens T \
--pm-loop BIN --out DIR --commit SHA40 [--store FILE] [--time-budget-s S] \
[--max-rounds R] [--max-processes N] [--max-wall-s S] [--aa N] [--confirm N]";

const FAMILY_WISE: &str = "Discovery admits a candidate at sign-test p <= 1e-3 per candidate; \
that is a screen, not the claim. Each discovery-passing candidate is then re-measured in \
held-out fresh processes and must again be separable at p <= 1e-3, material and above the \
A/A noise floor, which bounds the per-candidate false-winner rate at about 1e-6 under \
independence of the fresh processes (k candidates => at most k * 1e-6). The code-identical \
control voids every winner of a run in which it too is separable and material.";

/// Everything `run_all` needs besides the domain.
#[derive(Clone, Debug)]
pub struct RunConfig {
    pub pm_loop: PathBuf,
    pub out: PathBuf,
    /// hipfire commit (40 hex) recorded in every store key.
    pub commit: String,
    pub store: Option<PathBuf>,
    pub time_budget_s: f64,
    pub max_rounds: u64,
    pub max_processes: usize,
    pub max_wall_s: f64,
    pub policy: ClaimPolicyV1,
}

// ---------------------------------------------------------------------------
// Record parsing

fn at<'a>(v: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter()
        .try_fold(v, |v, k| v.get(*k))
        .filter(|x| !x.is_null())
}

fn at_str(v: &Value, path: &[&str]) -> Option<String> {
    at(v, path)?.as_str().map(str::to_owned)
}

/// `resources[].code_sha256` of the row whose `symbol` is the arm's launch symbol.
fn code_sha256(rec: &Value, arm: &str) -> Option<String> {
    let sym = at(rec, &[arm, "launch_symbol"])?.as_str()?;
    at(rec, &[arm, "resources"])?
        .as_array()?
        .iter()
        .find(|r| r.get("symbol").and_then(Value::as_str) == Some(sym))?
        .get("code_sha256")?
        .as_str()
        .map(str::to_owned)
}

/// The gate-relevant fields of one `pm_loop` record; absent fields stay `None`.
pub fn parse_attempt(
    rec: &Value,
    candidate_id: &str,
    phase: &str,
    replicate: u32,
    record_path: &str,
    exit_code: i32,
) -> super::AttemptV1 {
    let t = |k: &str| at(rec, &["timing", k]);
    super::AttemptV1 {
        candidate_id: candidate_id.to_owned(),
        phase: phase.to_owned(),
        replicate,
        record_path: record_path.to_owned(),
        exit_code,
        error_stage: at_str(rec, &["error", "stage"]),
        error_message: at_str(rec, &["error", "message"]),
        m7_status: at_str(rec, &["candidate", "m7", "status"]),
        correctness_pass: at(rec, &["correctness", "pass"]).and_then(Value::as_bool),
        identical_to_baseline: at(rec, &["correctness", "identical_to_baseline"])
            .and_then(Value::as_bool),
        baseline_launch_symbol: at_str(rec, &["baseline", "launch_symbol"]),
        candidate_launch_symbol: at_str(rec, &["candidate", "launch_symbol"]),
        baseline_co_sha256: at_str(rec, &["baseline", "co_sha256"]),
        candidate_co_sha256: at_str(rec, &["candidate", "co_sha256"]),
        baseline_code_sha256: code_sha256(rec, "baseline"),
        candidate_code_sha256: code_sha256(rec, "candidate"),
        separable: t("separable").and_then(Value::as_bool),
        direction: t("direction").and_then(Value::as_str).map(str::to_owned),
        sign_p: t("sign_p").and_then(Value::as_f64),
        block_ratio: t("block_ratio").and_then(Value::as_f64),
        material: t("material").and_then(Value::as_bool),
        rounds_run: t("rounds_run").and_then(Value::as_u64),
        noise_floor: at(rec, &["timing", "vs_noise_floor", "floor"]).and_then(Value::as_f64),
        exceeds_noise_floor: at(rec, &["timing", "vs_noise_floor", "exceeds"])
            .and_then(Value::as_bool),
        device_uuid: at_str(rec, &["device", "uuid"]),
        fabric_job: at_str(rec, &["device", "fabric_job"]),
    }
}

/// Why an offline attempt cannot be searched, if it cannot.
fn prune_reason(a: &super::AttemptV1, node_symbol: &str) -> Option<String> {
    if a.exit_code != 0 || a.error_stage.is_some() {
        return Some(format!(
            "exit_code {}: {}: {}",
            a.exit_code,
            a.error_stage.as_deref().unwrap_or("no error stage"),
            a.error_message.as_deref().unwrap_or("no error message")
        ));
    }
    if a.baseline_launch_symbol.as_deref() != Some(node_symbol) {
        return Some(format!(
            "baseline_mismatch: pm_loop baseline {} is not the corpus incumbent {node_symbol}",
            a.baseline_launch_symbol.as_deref().unwrap_or("<none>")
        ));
    }
    None
}

/// Largest A/A floor among the groups of this kernel, shape and baseline.
fn aa_floor(doc: &Value, kernel: &str, shape: &Value, baseline_sha: &str) -> Option<f64> {
    doc["groups"]
        .as_array()?
        .iter()
        .filter(|g| {
            g["aa"] == true
                && g["kernel"] == kernel
                && g["shape"] == *shape
                && g["baseline_sha256"] == baseline_sha
        })
        .filter_map(|g| g["floor"].as_f64())
        .max_by(f64::total_cmp)
}

/// Geometric-mean speedup over discovery and confirmations (the promotion
/// formula), `None` without a usable block ratio.
fn speedup(ev: &TuneEvidenceV1) -> Option<f64> {
    let ratios: Vec<f64> = std::iter::once(&ev.discovery)
        .chain(&ev.confirms)
        .filter_map(|a| a.block_ratio)
        .filter(|r| r.is_finite() && *r > 0.0)
        .collect();
    if ratios.is_empty() {
        return None;
    }
    let g = (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len() as f64).exp();
    Some(1.0 / g)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Run `pm_loop` with `args`, stdin closed. Linux refuses `execve` with
/// ETXTBSY while any process still holds the executable open for writing; a
/// sibling thread's fork can inherit a write descriptor of a just-written
/// binary until that child execs. That window is milliseconds, so ETXTBSY is
/// retried a bounded number of times; every other error returns at once.
fn run_pm_loop(pm_loop: &Path, args: &[String]) -> std::io::Result<std::process::Output> {
    const ETXTBSY: i32 = 26;
    let mut attempt = 0;
    loop {
        match std::process::Command::new(pm_loop)
            .args(args)
            .stdin(std::process::Stdio::null())
            .output()
        {
            Err(e) if e.raw_os_error() == Some(ETXTBSY) && attempt < 20 => {
                attempt += 1;
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            r => return r,
        }
    }
}

// ---------------------------------------------------------------------------
// Process driver

struct Invoked {
    attempt: super::AttemptV1,
    /// The raw record bytes when stdout was a JSON object.
    bytes: Option<Vec<u8>>,
    rec: Value,
}

struct Driver<'a> {
    cfg: &'a RunConfig,
    start: Instant,
    processes: usize,
    stop: Option<Value>,
    noise_path: Option<PathBuf>,
}

/// Exit code of a reused record that has no `.exit` sidecar, from its content.
fn infer_exit(rec: &Value) -> i32 {
    if rec.get("error").is_some() {
        3
    } else if at(rec, &["correctness", "pass"]).and_then(Value::as_bool) == Some(false) {
        1
    } else {
        0
    }
}

/// Whether a stored record may be reused for `spec`: always for non-`.rip`
/// candidates; for a `rip` knob only when the record's `candidate.rip.sha256`
/// equals the sha256 of the script's current bytes (an edited or missing script,
/// or a record without that field, is re-run).
fn rip_record_current(spec: &CandidateSpecV1, rec: &Value) -> bool {
    let Some(path) = spec.knobs.get("rip") else {
        return true;
    };
    let Ok(bytes) = std::fs::read(path) else {
        return false;
    };
    let now: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    at(rec, &["candidate", "rip", "sha256"]).and_then(Value::as_str) == Some(now.as_str())
}

impl Driver<'_> {
    fn budget_cut(&self) -> Option<&'static str> {
        if self.processes >= self.cfg.max_processes {
            Some("max_processes")
        } else if self.start.elapsed().as_secs_f64() >= self.cfg.max_wall_s {
            Some("max_wall_s")
        } else {
            None
        }
    }

    fn ledger(&self, v: Value) -> Result<(), String> {
        let path = self.cfg.out.join("ledger.jsonl");
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        writeln!(f, "{v}").map_err(|e| format!("{}: {e}", path.display()))
    }

    /// One `pm_loop` invocation (or its reused record). `Ok(None)`: the budget
    /// is spent and `stop` names `phase`.
    fn invoke(
        &mut self,
        phase: &str,
        spec: &CandidateSpecV1,
        replicate: u32,
    ) -> Result<Option<Invoked>, String> {
        let id = spec.id();
        let rec_dir = self.cfg.out.join("records");
        let stem = format!("{phase}-{id}-{replicate}");
        let json_path = rec_dir.join(format!("{stem}.json"));
        let record_path = json_path.display().to_string();

        if let Ok(bytes) = std::fs::read(&json_path) {
            if let Ok(rec) = serde_json::from_slice::<Value>(&bytes) {
                // A `.rip` candidate's id hashes the script path, not its bytes:
                // reuse only a record measured from the current script content.
                if rec.is_object() && rip_record_current(spec, &rec) {
                    let exit_code = std::fs::read_to_string(rec_dir.join(format!("{stem}.exit")))
                        .ok()
                        .and_then(|s| s.trim().parse().ok())
                        .unwrap_or_else(|| infer_exit(&rec));
                    let attempt =
                        parse_attempt(&rec, &id, phase, replicate, &record_path, exit_code);
                    self.ledger(json!({
                        "phase": phase, "candidate_id": id, "replicate": replicate,
                        "exit_code": exit_code, "record_path": record_path,
                        "wall_s": 0.0, "reused": true,
                    }))?;
                    return Ok(Some(Invoked {
                        attempt,
                        bytes: Some(bytes),
                        rec,
                    }));
                }
            }
        }

        if let Some(reason) = self.budget_cut() {
            self.stop.get_or_insert_with(
                || json!({"phase": phase, "reason": reason, "processes": self.processes}),
            );
            return Ok(None);
        }

        let mut args = spec.pm_loop_args();
        for (k, v) in [
            ("--time", "50".to_owned()),
            ("--rounds", "6".to_owned()),
            ("--max-rounds", self.cfg.max_rounds.to_string()),
            ("--time-budget-s", self.cfg.time_budget_s.to_string()),
            ("--out", self.cfg.out.join("obj").display().to_string()),
        ] {
            args.push(k.to_owned());
            args.push(v);
        }
        match phase {
            "offline" => args.push("--offline".to_owned()),
            "aa" => {
                args.push("--null-control".to_owned());
                args.push("aa".to_owned());
            }
            _ => {
                if let Some(n) = &self.noise_path {
                    args.push("--noise-floor".to_owned());
                    args.push(n.display().to_string());
                }
            }
        }

        let t0 = Instant::now();
        let out = run_pm_loop(&self.cfg.pm_loop, &args)
            .map_err(|e| format!("{}: {e}", self.cfg.pm_loop.display()))?;
        let wall_s = t0.elapsed().as_secs_f64();
        self.processes += 1;
        let exit_code = out.status.code().unwrap_or(-1);

        let err_path = rec_dir.join(format!("{stem}.err"));
        std::fs::write(&err_path, &out.stderr)
            .map_err(|e| format!("{}: {e}", err_path.display()))?;
        let exit_path = rec_dir.join(format!("{stem}.exit"));
        std::fs::write(&exit_path, exit_code.to_string())
            .map_err(|e| format!("{}: {e}", exit_path.display()))?;
        // The record is written last: its presence is the resume marker.
        std::fs::write(&json_path, &out.stdout).map_err(|e| format!("{record_path}: {e}"))?;

        let (rec, bytes) = match serde_json::from_slice::<Value>(&out.stdout) {
            Ok(v) if v.is_object() => (v, Some(out.stdout)),
            _ => (
                json!({"error": {
                    "stage": "driver",
                    "message": format!(
                        "pm_loop stdout is not a JSON object (exit {exit_code}); see {}",
                        err_path.display()
                    ),
                }}),
                None,
            ),
        };
        let attempt = parse_attempt(&rec, &id, phase, replicate, &record_path, exit_code);
        self.ledger(json!({
            "phase": phase, "candidate_id": id, "replicate": replicate,
            "exit_code": exit_code, "record_path": record_path,
            "wall_s": wall_s, "reused": false,
        }))?;
        Ok(Some(Invoked {
            attempt,
            bytes,
            rec,
        }))
    }
}

// ---------------------------------------------------------------------------
// Phase machine

struct Cand {
    spec: CandidateSpecV1,
    id: String,
    role: String,
    reason: Option<String>,
    offline: Option<super::AttemptV1>,
    discovery: Option<super::AttemptV1>,
    confirms: Vec<super::AttemptV1>,
}

impl Cand {
    fn is_control(&self) -> bool {
        self.role == "code_identical_control"
    }
}

/// Run every phase of `domain` under `cfg`; writes `DIR/result.json` and
/// returns it.
pub fn run_all(domain: &DomainV1, cfg: &RunConfig) -> Result<Value, String> {
    let bin_bytes = std::fs::read(&cfg.pm_loop)
        .map_err(|e| format!("pm_loop {}: {e}", cfg.pm_loop.display()))?;
    let pm_loop_sha256 = sha256_hex(&bin_bytes);
    drop(bin_bytes);
    for d in [cfg.out.join("records"), cfg.out.join("obj")] {
        std::fs::create_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let policy = &cfg.policy;
    let mut drv = Driver {
        cfg,
        start: Instant::now(),
        processes: 0,
        stop: None,
        noise_path: None,
    };
    let incumbent_id = domain.incumbent.id();
    let mut cands: Vec<Cand> = domain
        .candidates
        .iter()
        .map(|s| Cand {
            id: s.id(),
            spec: s.clone(),
            role: "unscreened".to_owned(),
            reason: None,
            offline: None,
            discovery: None,
            confirms: Vec::new(),
        })
        .collect();

    // Phase 0: offline legality screen (incumbent first, then the domain).
    let mut offline: BTreeMap<String, super::AttemptV1> = BTreeMap::new();
    let mut specs: Vec<&CandidateSpecV1> = vec![&domain.incumbent];
    specs.extend(domain.candidates.iter().filter(|c| c.id() != incumbent_id));
    for spec in specs {
        match drv.invoke("offline", spec, 0)? {
            Some(i) => {
                offline.insert(spec.id(), i.attempt);
            }
            None => break,
        }
        if spec.id() == incumbent_id {
            let a = &offline[&incumbent_id];
            if let Some(reason) = prune_reason(a, &domain::baseline_symbol(domain)) {
                return Err(format!(
                    "incumbent {} (epi {}) refused: {reason}",
                    incumbent_id,
                    domain
                        .incumbent
                        .knobs
                        .get("epi")
                        .map_or("?", String::as_str)
                ));
            }
        }
    }
    let mut seen: Vec<(String, String)> = Vec::new();
    for c in &mut cands {
        let Some(a) = offline.get(&c.id) else {
            continue;
        };
        c.offline = Some(a.clone());
        if let Some(r) = prune_reason(a, &domain::baseline_symbol(domain)) {
            c.role = "pruned".to_owned();
            c.reason = Some(r);
            continue;
        }
        let code = a.candidate_code_sha256.clone();
        if let Some(code) = &code {
            if let Some((_, first)) = seen.iter().find(|(s, _)| s == code) {
                c.role = format!("duplicate_of:{first}");
                continue;
            }
            seen.push((code.clone(), c.id.clone()));
        }
        c.role = if code.is_some() && code == a.baseline_code_sha256 {
            "code_identical_control".to_owned()
        } else {
            "search".to_owned()
        };
    }

    // Phase 1: A/A floor of the incumbent.
    let mut aa_attempts: Vec<super::AttemptV1> = Vec::new();
    let mut aa_inputs: Vec<(String, Vec<u8>)> = Vec::new();
    let mut aa_ref: Option<(Value, String)> = None;
    if drv.stop.is_none() {
        for rep in 0..policy.aa_processes as u32 {
            let Some(i) = drv.invoke("aa", &domain.incumbent, rep)? else {
                break;
            };
            if let Some(b) = i.bytes {
                aa_inputs.push((i.attempt.record_path.clone(), b));
            }
            if aa_ref.is_none() && i.attempt.block_ratio.is_some() {
                if let Some(sha) = i.attempt.baseline_co_sha256.clone() {
                    aa_ref = Some((i.rec["shape"].clone(), sha));
                }
            }
            aa_attempts.push(i.attempt);
        }
    }
    let aa_runs = aa_attempts
        .iter()
        .filter(|a| a.block_ratio.is_some())
        .count();
    let mut noise_floor = None;
    if drv.stop.is_none() {
        let doc = crate::noise::analyze(&aa_inputs)?;
        let noise_path = cfg.out.join("noise.json");
        let text = serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?;
        std::fs::write(&noise_path, text).map_err(|e| format!("{}: {e}", noise_path.display()))?;
        if let Some((shape, sha)) = &aa_ref {
            noise_floor = aa_floor(&doc, &domain.kernel, shape, sha);
        }
        drv.noise_path = Some(noise_path);
    }

    // Phase 2: discovery of every search candidate and the control.
    if drv.stop.is_none() {
        for c in &mut cands {
            if c.role != "search" && !c.is_control() {
                continue;
            }
            match drv.invoke("discovery", &c.spec, 0)? {
                Some(i) => c.discovery = Some(i.attempt),
                None => break,
            }
        }
    }

    // Phase 3: held-out confirmation of the discovery survivors.
    let evidence = |c: &Cand, discovery: &super::AttemptV1| TuneEvidenceV1 {
        candidate: c.spec.clone(),
        discovery: discovery.clone(),
        confirms: c.confirms.clone(),
        noise_floor,
        aa_runs,
        kld_state: None,
        kld_token: None,
    };
    if drv.stop.is_none() {
        let screen = ClaimPolicyV1 {
            confirm_processes: 0,
            ..policy.clone()
        };
        'confirm: for ci in 0..cands.len() {
            if cands[ci].role != "search" {
                continue;
            }
            let Some(d) = cands[ci].discovery.clone() else {
                continue;
            };
            let eligible = match promote::can_promote(&evidence(&cands[ci], &d), &screen) {
                Ok(_) => true,
                Err(r) => r.stage == "kld_state",
            };
            if !eligible {
                continue;
            }
            for rep in 0..policy.confirm_processes as u32 {
                match drv.invoke("confirm", &cands[ci].spec.clone(), rep)? {
                    Some(i) => cands[ci].confirms.push(i.attempt),
                    None => break 'confirm,
                }
            }
        }
    }

    // Verdicts, negative control, store records.
    let controls: Vec<Value> = cands
        .iter()
        .filter(|c| c.is_control())
        .map(|c| {
            let d = c.discovery.as_ref();
            json!({
                "id": c.id,
                "separable": d.and_then(|d| d.separable),
                "material": d.and_then(|d| d.material),
                "block_ratio": d.and_then(|d| d.block_ratio),
            })
        })
        .collect();
    let control_failed = cands.iter().any(|c| {
        c.is_control()
            && c.discovery
                .as_ref()
                .is_some_and(|d| d.separable == Some(true) && d.material == Some(true))
    });
    let negative_control = json!({
        "status": if control_failed { "negative_control_failed" }
                  else if controls.is_empty() { "no_control_ran" } else { "pass" },
        "controls": controls,
        "note": if control_failed {
            "a code-identical candidate was separable and material: the harness manufactures \
             effects here, so every winner of this run is void"
        } else {
            "a code-identical candidate must not be separable and material"
        },
    });

    let mut winners: Vec<String> = Vec::new();
    let mut best: Option<(f64, Value)> = None;
    let mut cand_json: Vec<Value> = Vec::new();
    let mut records: Vec<TuneRecordV1> = Vec::new();
    let key_sha = |a: &super::AttemptV1, o: &Option<super::AttemptV1>| {
        let pick =
            |f: fn(&super::AttemptV1) -> Option<String>| f(a).or_else(|| o.as_ref().and_then(f));
        (
            pick(|a| a.baseline_code_sha256.clone()),
            pick(|a| a.candidate_co_sha256.clone()),
            pick(|a| a.candidate_code_sha256.clone()),
        )
    };
    for c in &cands {
        let mut entry = json!({
            "id": c.id, "knobs": c.spec.knobs, "role": c.role, "reason": c.reason,
            "offline": c.offline, "discovery": c.discovery, "confirms": c.confirms,
            "verdict": "not_winner", "speedup": null, "promotion": null,
        });
        if let Some(d) = &c.discovery {
            let ev = evidence(c, d);
            let would_win = c.role == "search" && promote::is_winner(&ev, policy);
            let mut promotion = promote::can_promote(&ev, policy);
            let win = would_win && !control_failed;
            if would_win && control_failed {
                let passed = match &promotion {
                    Ok(p) => p.passed.clone(),
                    Err(r) => r.passed.clone(),
                };
                promotion = Err(Refusal {
                    stage: "negative_control".to_owned(),
                    reason: "the code-identical control was separable and material in this run"
                        .to_owned(),
                    passed,
                });
            }
            let sp = speedup(&ev);
            entry["verdict"] = json!(if win {
                "winner"
            } else if would_win {
                "void_negative_control"
            } else {
                "not_winner"
            });
            entry["speedup"] = json!(sp);
            entry["promotion"] = serde_json::to_value(&promotion).map_err(|e| e.to_string())?;
            if win {
                winners.push(c.id.clone());
                if let Some(s) = sp {
                    if best.as_ref().map_or(true, |(b, _)| s > *b) {
                        best = Some((s, json!({"id": c.id, "knobs": c.spec.knobs, "speedup": s})));
                    }
                }
            }
            if let (Some(b), Some(co), Some(code)) = key_sha(d, &c.offline) {
                records.push(TuneRecordV1 {
                    schema: RECORD_SCHEMA.to_owned(),
                    key: TuneKeyV1 {
                        arch: c.spec.arch.clone(),
                        kernel: c.spec.kernel.clone(),
                        shape: c.spec.shape.clone(),
                        // An attempt measured NOT identical to the baseline makes the record
                        // numeric (the KLD/state gate decides); unmeasured stays exact.
                        numeric_contract: if std::iter::once(&ev.discovery)
                            .chain(ev.confirms.iter())
                            .any(|a| a.identical_to_baseline == Some(false))
                        {
                            "numeric"
                        } else {
                            "exact"
                        }
                        .to_owned(),
                        mode: "eager".to_owned(),
                        baseline_code_sha256: b,
                        candidate_co_sha256: co,
                        candidate_code_sha256: code,
                        commit: cfg.commit.clone(),
                        pm_loop_sha256: pm_loop_sha256.clone(),
                    },
                    domain_hash: domain.domain_hash.clone(),
                    candidate: c.spec.clone(),
                    verdict: if win { "winner" } else { "not_winner" }.to_owned(),
                    evidence: ev,
                    promotion,
                });
            }
        }
        cand_json.push(entry);
    }
    if let Some(path) = &cfg.store {
        for r in &records {
            store::append(path, r)?;
        }
    }

    let result = json!({
        "schema": RUN_SCHEMA,
        "domain": domain,
        "policy": policy,
        "family_wise": FAMILY_WISE,
        "noise_floor": noise_floor,
        "aa_runs": aa_runs,
        "candidates": cand_json,
        "winners": winners,
        "best": best.map(|(_, v)| v),
        "negative_control": negative_control,
        "stop": drv.stop,
        "pm_loop_sha256": pm_loop_sha256,
    });
    let path = cfg.out.join("result.json");
    let text = serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(result)
}

// ---------------------------------------------------------------------------
// CLI

fn flags(args: &[String], allowed: &[&str]) -> Result<BTreeMap<String, String>, String> {
    let mut m = BTreeMap::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let k = a
            .strip_prefix("--")
            .filter(|k| allowed.contains(k))
            .ok_or_else(|| format!("unknown argument {a:?}\n{USAGE}"))?;
        let v = it
            .next()
            .ok_or_else(|| format!("--{k} needs a value\n{USAGE}"))?;
        if m.insert(k.to_owned(), v.clone()).is_some() {
            return Err(format!("--{k} given twice"));
        }
    }
    Ok(m)
}

fn req<'a>(m: &'a BTreeMap<String, String>, k: &str) -> Result<&'a str, String> {
    m.get(k)
        .map(String::as_str)
        .ok_or_else(|| format!("missing --{k}\n{USAGE}"))
}

fn num<T: std::str::FromStr>(
    m: &BTreeMap<String, String>,
    k: &str,
    default: Option<T>,
) -> Result<T, String> {
    match m.get(k) {
        Some(v) => v
            .parse()
            .map_err(|_| format!("--{k}: cannot parse {v:?} as a number")),
        None => default.ok_or_else(|| format!("missing --{k}\n{USAGE}")),
    }
}

/// The unique corpus case whose `case_id` starts with `prefix`, enumerated.
pub fn load_domain(
    corpus: &Path,
    prefix: &str,
    node: usize,
    tokens: u64,
) -> Result<DomainV1, String> {
    if prefix.is_empty() {
        return Err("--case must be a non-empty case_id prefix".to_owned());
    }
    let corpus = WorkloadCorpusV1::read_jsonl(corpus)?;
    let hits: Vec<_> = corpus
        .cases
        .iter()
        .filter(|c| c.case_id.starts_with(prefix))
        .collect();
    match hits.as_slice() {
        [] => Err(format!("no case_id starts with {prefix:?}")),
        [c] => domain::enumerate(c, node, tokens),
        many => Err(format!(
            "{} cases start with {prefix:?} ({}); give a longer prefix",
            many.len(),
            many.iter()
                .map(|c| &c.case_id[..c.case_id.len().min(12)])
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// `pm-census tune plan|run ...`; prints one JSON document.
pub fn cli(args: &[String]) -> Result<(), String> {
    let Some((sub, rest)) = args.split_first() else {
        return Err(USAGE.to_owned());
    };
    let selectors = ["corpus", "case", "node", "tokens"];
    match sub.as_str() {
        "plan" => {
            let m = flags(rest, &selectors)?;
            let d = load_domain(
                Path::new(req(&m, "corpus")?),
                req(&m, "case")?,
                num(&m, "node", None)?,
                num(&m, "tokens", None)?,
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&d).map_err(|e| e.to_string())?
            );
            Ok(())
        }
        "run" => {
            let allowed = [
                "corpus",
                "case",
                "node",
                "tokens",
                "pm-loop",
                "out",
                "commit",
                "store",
                "time-budget-s",
                "max-rounds",
                "max-processes",
                "max-wall-s",
                "aa",
                "confirm",
            ];
            let m = flags(rest, &allowed)?;
            let d = load_domain(
                Path::new(req(&m, "corpus")?),
                req(&m, "case")?,
                num(&m, "node", None)?,
                num(&m, "tokens", None)?,
            )?;
            let commit = req(&m, "commit")?;
            if commit.len() != 40 || !commit.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(format!("--commit {commit:?} is not a 40-hex sha"));
            }
            let defaults = ClaimPolicyV1::default();
            let cfg = RunConfig {
                pm_loop: PathBuf::from(req(&m, "pm-loop")?),
                out: PathBuf::from(req(&m, "out")?),
                commit: commit.to_ascii_lowercase(),
                store: m.get("store").map(PathBuf::from),
                time_budget_s: num(&m, "time-budget-s", Some(30.0))?,
                max_rounds: num(&m, "max-rounds", Some(64))?,
                max_processes: num(&m, "max-processes", Some(40))?,
                max_wall_s: num(&m, "max-wall-s", Some(1080.0))?,
                policy: ClaimPolicyV1 {
                    aa_processes: num(&m, "aa", Some(defaults.aa_processes))?,
                    confirm_processes: num(&m, "confirm", Some(defaults.confirm_processes))?,
                    ..defaults
                },
            };
            let result = run_all(&d, &cfg)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
            );
            Ok(())
        }
        _ => Err(USAGE.to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::domain::test_case;
    use std::os::unix::fs::PermissionsExt;

    const NODE: &str = "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8";
    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "pm-census-tune-run-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn domain() -> DomainV1 {
        let case = test_case("a3b", NODE, &[("M", 1024), ("m_total", 69376)]);
        domain::enumerate(&case, 0, 8192).unwrap()
    }

    /// A 64-hex stand-in sha for a readable tag.
    fn h(tag: &str) -> String {
        Sha256::digest(tag.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// A schema-1 record. `kind`: `offline`, `aa` or `run`.
    fn record(kind: &str, cand_code: &str, timing: Value) -> Value {
        let base_code = "base_code";
        let cand_co = if cand_code == base_code {
            "co_base"
        } else {
            "co_cand"
        };
        let sym = |c: &str| {
            if c == base_code {
                NODE.to_owned()
            } else {
                format!("{NODE}_{c}")
            }
        };
        let mut r = json!({
            "schema": 1, "kernel": "qwen4_moe_sym", "arch": "gfx1201",
            "shape": {"preset": "a3b", "T": 8192},
            "baseline": {"co_sha256": h("co_base"), "launch_symbol": NODE,
                "resources": [{"symbol": "other", "code_sha256": h("zz")},
                              {"symbol": NODE, "code_sha256": h(base_code)}]},
            "candidate": {"co_sha256": h(cand_co), "launch_symbol": sym(cand_code),
                "resources": [{"symbol": sym(cand_code), "code_sha256": h(cand_code)}],
                "m7": {"status": "pass"}},
        });
        if kind == "offline" {
            r["offline"] = json!(true);
        } else {
            r["device"] = json!({"uuid": "GPU-1", "fabric_job": "job-9"});
            r["correctness"] = json!({"pass": true, "identical_to_baseline": true});
            r["timing"] = timing;
        }
        if kind == "aa" {
            r["null_control"] = json!({"kind": "aa"});
        }
        r
    }

    fn timing(ratio: f64, separable: bool, material: bool) -> Value {
        json!({
            "separable": separable, "direction": if ratio < 1.0 { "faster" } else { "slower" },
            "sign_p": if separable { 1e-4 } else { 0.4 }, "block_ratio": ratio,
            "material": material, "rounds_run": 12,
            "vs_noise_floor": {"floor": 0.0003, "effect": (ratio - 1.0).abs(),
                               "exceeds": (ratio - 1.0).abs() > 0.0003, "group": "g"},
        })
    }

    fn canned(dir: &Path, mode: &str, epi: &str, rec: &Value, exit: i32) {
        let d = dir.join("canned");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(format!("{mode}-{epi}.json")), rec.to_string()).unwrap();
        std::fs::write(d.join(format!("{mode}-{epi}.exit")), exit.to_string()).unwrap();
    }

    fn script(dir: &Path) -> PathBuf {
        let p = dir.join("fake_pm_loop.sh");
        std::fs::write(
            &p,
            "#!/bin/sh\n\
             d=\"$(dirname \"$0\")/canned\"\n\
             mode=run; epi=; prev=\n\
             for a in \"$@\"; do\n\
               case \"$a\" in\n\
                 --offline) mode=offline ;;\n\
                 --null-control) mode=aa ;;\n\
               esac\n\
               if [ \"$prev\" = \"--epi\" ]; then epi=\"$a\"; fi\n\
               prev=\"$a\"\n\
             done\n\
             cat \"$d/$mode-$epi.json\"\n\
             exit \"$(cat \"$d/$mode-$epi.exit\")\"\n",
        )
        .unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    /// gate_up: faster search candidate; gate_up_nt4: pruned offline (exit 3);
    /// gate_up_nt8 (= incumbent twin): code-identical control, not separable.
    fn standard(dir: &Path, control_effect: bool) {
        let faster = timing(0.9, true, true);
        canned(
            dir,
            "offline",
            "gate_up",
            &record("offline", "code_a", Value::Null),
            0,
        );
        canned(
            dir,
            "offline",
            "gate_up_nt4",
            &json!({"schema": 1, "error": {"stage": "m7", "message": "M7 fail"}}),
            3,
        );
        canned(
            dir,
            "offline",
            "gate_up_nt8",
            &record("offline", "base_code", Value::Null),
            0,
        );
        canned(
            dir,
            "aa",
            "gate_up_nt8",
            &record("aa", "base_code", timing(1.0003, false, false)),
            0,
        );
        canned(dir, "run", "gate_up", &record("run", "code_a", faster), 0);
        let ctl = if control_effect {
            timing(0.97, true, true)
        } else {
            timing(1.0001, false, false)
        };
        canned(
            dir,
            "run",
            "gate_up_nt8",
            &record("run", "base_code", ctl),
            0,
        );
    }

    fn cfg(dir: &Path) -> RunConfig {
        RunConfig {
            pm_loop: script(dir),
            out: dir.join("out"),
            commit: COMMIT.to_owned(),
            store: Some(dir.join("store.jsonl")),
            time_budget_s: 30.0,
            max_rounds: 64,
            max_processes: 40,
            max_wall_s: 1080.0,
            policy: ClaimPolicyV1::default(),
        }
    }

    fn cand<'a>(r: &'a Value, epi: &str) -> &'a Value {
        r["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["knobs"]["epi"] == epi)
            .unwrap()
    }

    #[test]
    fn parse_attempt_reads_every_field() {
        let mut rec = record("run", "code_a", timing(0.9, true, true));
        rec["error"] = json!({"stage": "x", "message": "boom"});
        let a = parse_attempt(&rec, "id", "discovery", 2, "p.json", 1);
        assert_eq!(
            (a.candidate_id.as_str(), a.phase.as_str(), a.replicate),
            ("id", "discovery", 2)
        );
        assert_eq!((a.record_path.as_str(), a.exit_code), ("p.json", 1));
        assert_eq!(a.error_stage.as_deref(), Some("x"));
        assert_eq!(a.error_message.as_deref(), Some("boom"));
        assert_eq!(a.m7_status.as_deref(), Some("pass"));
        assert_eq!(
            (a.correctness_pass, a.identical_to_baseline),
            (Some(true), Some(true))
        );
        assert_eq!(a.baseline_launch_symbol.as_deref(), Some(NODE));
        assert_eq!(
            a.candidate_launch_symbol.as_deref(),
            Some(format!("{NODE}_code_a").as_str())
        );
        assert_eq!(a.baseline_co_sha256, Some(h("co_base")));
        assert_eq!(a.candidate_co_sha256, Some(h("co_cand")));
        assert_eq!(a.baseline_code_sha256, Some(h("base_code")));
        assert_eq!(a.candidate_code_sha256, Some(h("code_a")));
        assert_eq!(
            (a.separable, a.direction.as_deref()),
            (Some(true), Some("faster"))
        );
        assert_eq!(
            (a.sign_p, a.block_ratio, a.material),
            (Some(1e-4), Some(0.9), Some(true))
        );
        assert_eq!(a.rounds_run, Some(12));
        assert_eq!(
            (a.noise_floor, a.exceeds_noise_floor),
            (Some(0.0003), Some(true))
        );
        assert_eq!(
            (a.device_uuid.as_deref(), a.fabric_job.as_deref()),
            (Some("GPU-1"), Some("job-9"))
        );
        let none = parse_attempt(&json!({}), "id", "offline", 0, "p", 0);
        assert!(none.m7_status.is_none() && none.block_ratio.is_none());
        assert!(none.baseline_code_sha256.is_none() && none.noise_floor.is_none());
    }

    #[test]
    fn full_run_finds_winner_and_keeps_control_quiet() {
        let dir = tmp("full");
        standard(&dir, false);
        let cfg = cfg(&dir);
        let d = domain();
        let r = run_all(&d, &cfg).unwrap();
        let gu = &d.candidates[0];
        assert_eq!(r["schema"], RUN_SCHEMA);
        assert_eq!(r["winners"], json!([gu.id()]));
        assert_eq!(r["best"]["id"], gu.id());
        assert!((r["best"]["speedup"].as_f64().unwrap() - 1.0 / 0.9).abs() < 1e-9);
        assert_eq!(r["aa_runs"], 3);
        assert!((r["noise_floor"].as_f64().unwrap() - 0.0003).abs() < 1e-12);
        assert_eq!(r["negative_control"]["status"], "pass");
        assert_eq!(r["stop"], Value::Null);
        assert_eq!(r["pm_loop_sha256"].as_str().unwrap().len(), 64);

        assert_eq!(cand(&r, "gate_up")["role"], "search");
        assert_eq!(cand(&r, "gate_up")["verdict"], "winner");
        assert_eq!(cand(&r, "gate_up")["confirms"].as_array().unwrap().len(), 1);
        assert_eq!(cand(&r, "gate_up_nt4")["role"], "pruned");
        assert!(cand(&r, "gate_up_nt4")["reason"]
            .as_str()
            .unwrap()
            .contains("m7"));
        assert!(cand(&r, "gate_up_nt4")["discovery"].is_null());
        assert_eq!(cand(&r, "gate_up_nt8")["role"], "code_identical_control");
        assert_eq!(cand(&r, "gate_up_nt8")["verdict"], "not_winner");

        // Records exist for the control and the winner only; the winner is
        // bit-identical, so the kld_state stage passes on the exact-output proof.
        let (recs, bad) = store::import(&std::fs::read_to_string(dir.join("store.jsonl")).unwrap());
        assert!(bad.is_empty(), "{bad:?}");
        assert_eq!(recs.len(), 2);
        let win = recs.iter().find(|x| x.candidate == *gu).unwrap();
        assert_eq!(win.verdict, "winner");
        assert_eq!(win.key.numeric_contract, "exact");
        assert_eq!(win.key.commit, COMMIT);
        assert_eq!(win.key.baseline_code_sha256, h("base_code"));
        assert_eq!(win.key.candidate_code_sha256, h("code_a"));
        assert_eq!(win.key.candidate_co_sha256, h("co_cand"));
        assert_eq!(win.domain_hash, d.domain_hash);
        assert_eq!(win.evidence.confirms.len(), 1);
        assert!(win.promotion.is_ok(), "{:?}", win.promotion);
        let ctl = recs
            .iter()
            .find(|x| x.candidate == d.candidates[2])
            .unwrap();
        assert_eq!(ctl.verdict, "not_winner");

        let ledger = std::fs::read_to_string(cfg.out.join("ledger.jsonl")).unwrap();
        assert_eq!(ledger.lines().count(), 3 + 3 + 2 + 1);
        assert!(ledger.lines().all(|l| l.contains("\"reused\":false")));
        assert!(cfg.out.join("noise.json").exists());
        assert_eq!(
            serde_json::from_slice::<Value>(&std::fs::read(cfg.out.join("result.json")).unwrap())
                .unwrap(),
            // Both sides through the same text parse (serde_json floats are not
            // round-trip exact without `float_roundtrip`).
            serde_json::from_str::<Value>(&r.to_string()).unwrap()
        );
    }

    #[test]
    fn resume_reuses_every_record_and_reproduces_result() {
        let dir = tmp("resume");
        standard(&dir, false);
        let mut cfg = cfg(&dir);
        cfg.store = None;
        let d = domain();
        run_all(&d, &cfg).unwrap();
        let first = std::fs::read(cfg.out.join("result.json")).unwrap();
        // Any fresh process would now fail: its canned output is gone.
        std::fs::remove_dir_all(dir.join("canned")).unwrap();
        run_all(&d, &cfg).unwrap();
        assert_eq!(std::fs::read(cfg.out.join("result.json")).unwrap(), first);
        let ledger = std::fs::read_to_string(cfg.out.join("ledger.jsonl")).unwrap();
        let lines: Vec<&str> = ledger.lines().collect();
        assert_eq!(lines.len(), 18);
        assert!(lines[..9].iter().all(|l| l.contains("\"reused\":false")));
        assert!(lines[9..].iter().all(|l| l.contains("\"reused\":true")));
    }

    #[test]
    fn separable_material_control_voids_winners() {
        let dir = tmp("control");
        standard(&dir, true);
        let cfg = cfg(&dir);
        let d = domain();
        let r = run_all(&d, &cfg).unwrap();
        assert_eq!(r["negative_control"]["status"], "negative_control_failed");
        assert_eq!(r["winners"], json!([]));
        assert_eq!(r["best"], Value::Null);
        assert_eq!(cand(&r, "gate_up")["verdict"], "void_negative_control");
        assert_eq!(
            cand(&r, "gate_up")["promotion"]["Err"]["stage"],
            "negative_control"
        );
        let (recs, _) = store::import(&std::fs::read_to_string(dir.join("store.jsonl")).unwrap());
        let win = recs
            .iter()
            .find(|x| x.candidate == d.candidates[0])
            .unwrap();
        assert_eq!(win.verdict, "not_winner");
        assert_eq!(
            win.promotion.as_ref().unwrap_err().stage,
            "negative_control"
        );
    }

    #[test]
    fn process_budget_names_the_cut_phase() {
        let dir = tmp("budget");
        standard(&dir, false);
        let mut cfg = cfg(&dir);
        cfg.max_processes = 2;
        let r = run_all(&domain(), &cfg).unwrap();
        assert_eq!(r["stop"]["phase"], "offline");
        assert_eq!(r["stop"]["reason"], "max_processes");
        assert_eq!(r["winners"], json!([]));
        assert!(!cfg.out.join("noise.json").exists());
    }

    #[test]
    fn baseline_mismatch_prunes_a_candidate_but_refuses_a_mismatched_incumbent() {
        let dir = tmp("mismatch");
        standard(&dir, false);
        let mut other = record("offline", "code_a", Value::Null);
        other["baseline"]["launch_symbol"] = json!("qwen4_moe_flat_baseline");
        canned(&dir, "offline", "gate_up", &other, 0);
        let r = run_all(&domain(), &cfg(&dir)).unwrap();
        assert_eq!(cand(&r, "gate_up")["role"], "pruned");
        assert!(cand(&r, "gate_up")["reason"]
            .as_str()
            .unwrap()
            .starts_with("baseline_mismatch: pm_loop baseline qwen4_moe_flat_baseline is not"));
        assert_eq!(r["winners"], json!([]));

        let dir = tmp("mismatch-inc");
        standard(&dir, false);
        canned(&dir, "offline", "gate_up_nt8", &other, 0);
        let e = run_all(&domain(), &cfg(&dir)).unwrap_err();
        assert!(
            e.contains("incumbent") && e.contains("baseline_mismatch"),
            "{e}"
        );
    }

    #[test]
    fn duplicate_code_is_not_searched_twice() {
        let dir = tmp("dup");
        standard(&dir, false);
        canned(
            &dir,
            "offline",
            "gate_up_nt4",
            &record("offline", "code_a", Value::Null),
            0,
        );
        let d = domain();
        let r = run_all(&d, &cfg(&dir)).unwrap();
        assert_eq!(
            cand(&r, "gate_up_nt4")["role"],
            format!("duplicate_of:{}", d.candidates[0].id())
        );
        assert!(cand(&r, "gate_up_nt4")["discovery"].is_null());
    }

    #[test]
    fn plan_selects_a_unique_case_by_prefix() {
        let dir = tmp("plan");
        let case = test_case("a3b", NODE, &[("M", 1024), ("m_total", 69376)]);
        let corpus = WorkloadCorpusV1::new(vec![case.clone()]).unwrap();
        let path = dir.join("corpus.jsonl");
        corpus.write_jsonl(&path).unwrap();
        let d = load_domain(&path, &case.case_id[..8], 0, 8192).unwrap();
        assert_eq!(d.case_id, case.case_id);
        assert_eq!(d.candidates.len(), 3);
        assert!(load_domain(&path, "zz", 0, 8192)
            .unwrap_err()
            .contains("no case_id"));
        assert!(load_domain(&path, "", 0, 8192).is_err());
        assert!(cli(&["plan".into(), "--bogus".into(), "1".into()]).is_err());
        assert!(cli(&[]).is_err());
    }

    #[test]
    fn rip_record_reused_only_for_current_script_bytes() {
        let d = tmp("ripsha");
        std::fs::create_dir_all(&d).unwrap();
        let rip = d.join("c.rip");
        std::fs::write(&rip, b"one").unwrap();
        let mut knobs = BTreeMap::new();
        knobs.insert("rip".to_owned(), rip.display().to_string());
        let spec = CandidateSpecV1 {
            kernel: "iu4_gemm".into(),
            arch: "gfx1201".into(),
            shape: "M=128,N=128,K=256".into(),
            knobs,
        };
        let sha = |b: &[u8]| -> String {
            Sha256::digest(b)
                .iter()
                .map(|x| format!("{x:02x}"))
                .collect()
        };
        let rec = json!({"candidate": {"rip": {"sha256": sha(b"one")}}});
        assert!(rip_record_current(&spec, &rec));
        std::fs::write(&rip, b"two").unwrap();
        assert!(!rip_record_current(&spec, &rec));
        assert!(!rip_record_current(&spec, &json!({"candidate": {}})));
        std::fs::remove_file(&rip).unwrap();
        assert!(!rip_record_current(&spec, &rec));
        let mut plain = spec.clone();
        plain.knobs.clear();
        assert!(rip_record_current(&plain, &json!({})));
    }
}
