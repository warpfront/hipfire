// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! G4.7 Qwen3.5 reset/rollback hardware tests.
//!
//! All three tests are `#[ignore]`d: they need a real HIP GPU, a real Qwen3.5
//! model file, and a daemon binary built with `--features
//! serve-fault-inject` (the `test_fault_after_*` request fields are compiled
//! out otherwise; a fault request that returns `done` instead of `error`
//! fails with a rebuild hint).
//!
//! The tests drive the daemon in batch JSONL mode over piped stdin (the same
//! shape as `scripts/pp-gate.sh`): load → generate(s) → unload, then assert
//! on the parsed wire events:
//!
//! - fault after prefill / after first decode → exactly one `error`
//!   terminal with wire-visible `rolled_back=true`, no `done`, no tokens;
//! - the next clean request serves `done` with bytes identical to a
//!   fresh-daemon run (no assistant-cache store of the failed turn, no
//!   recurrent/KV drift);
//! - unload completes.
//!
//! Each fault case runs 3× in fresh processes (`HIPFIRE_RESET_ROUNDS`
//! overrides for time-boxed runs; default 3).
//!
//! The `dflash_*` test additionally needs `HIPFIRE_DFLASH_DRAFT` (a DFlash
//! draft); it forces the draft through the daemon env so every turn takes the
//! speculative-decode route, and asserts `dflash:true` on each clean `done`
//! so a silent AR fallback cannot pass as spec coverage.
//!
//! Reset-phase HIP failure (`rolled_back=false` on the wire) is not
//! forceable on healthy hardware; that half of the contract is pinned by
//! `unattested_epilogue_error_carries_reset_context` and
//! `pp_abort_unattested_error_shape_stable` in
//! `tests/qwen35_reset_terminals.rs`.

#![allow(clippy::all)]

use serde_json::Value;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

static HW_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    HW_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

const PROMPT: &str = "Write a one-sentence greeting.";
const MAX_TOKENS: u64 = 64;
const SEED: u64 = 12345;

fn default_model_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    PathBuf::from(home).join(".hipfire/models/qwen3.8-27b.mq4-xt")
}

fn model_path(var: &str) -> Option<PathBuf> {
    match std::env::var(var) {
        Ok(p) => {
            let p = PathBuf::from(p);
            if !p.is_file() {
                panic!("{var}={} is not a readable file", p.display());
            }
            Some(p)
        }
        Err(_) => {
            let p = default_model_path();
            if !p.is_file() {
                eprintln!("skip: {} unset and {} missing", var, p.display());
                return None;
            }
            Some(p)
        }
    }
}

fn rounds() -> usize {
    std::env::var("HIPFIRE_RESET_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3)
}
/// DFlash draft for the spec-route test. Skips when unset (there is no
/// default draft path to fall back to). The spec test passes the draft
/// explicitly through the daemon env, overriding any ambient value, so its
/// turns take the speculative-decode route by construction.
fn draft_path() -> Option<PathBuf> {
    match std::env::var("HIPFIRE_DFLASH_DRAFT") {
        Ok(p) => {
            let p = PathBuf::from(p);
            if !p.is_file() {
                panic!(
                    "HIPFIRE_DFLASH_DRAFT={} is not a readable file",
                    p.display()
                );
            }
            Some(p)
        }
        Err(_) => {
            eprintln!("skip: HIPFIRE_DFLASH_DRAFT unset (spec-route test needs a DFlash draft)");
            None
        }
    }
}

fn daemon_bin() -> PathBuf {
    if let Ok(p) = std::env::var("HIPFIRE_DAEMON_BIN") {
        let p = PathBuf::from(p);
        assert!(
            p.is_file(),
            "HIPFIRE_DAEMON_BIN={} is not a file",
            p.display()
        );
        return p;
    }
    // Crate name is hipfire-daemon but the bin target is `daemon`.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/daemon")
}
fn has_gpu() -> bool {
    // ROCm kernel driver node; absent on CPU-only boxes.
    std::path::Path::new("/dev/kfd").exists() || std::path::Path::new("/dev/dri").exists()
}

/// Interactive daemon session: requests are sent line-by-line and responses
/// are read back, so the two-phase `commit_ready` → `commit` handshake works
/// (a batch pipe applies a pre-sent commit before the worker waits, which the
/// daemon drops as stale and the turn ends `aborted`).
struct Session {
    stdin: std::process::ChildStdin,
    lines: std::sync::mpsc::Receiver<String>,
    child: std::process::Child,
    stderr: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

/// Per-receive ceiling: load is seconds, a 64-token debug generate is under
/// a minute; anything beyond this is a hung daemon, not a slow one.
const STEP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

impl Session {
    fn spawn(extra_env: &[(&str, &str)]) -> Self {
        let bin = daemon_bin();
        assert!(
            bin.is_file(),
            "daemon binary missing at {} — build it first: cargo build -p hipfire-daemon --locked --features serve-fault-inject",
            bin.display()
        );
        let mut child = Command::new(&bin)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("HIPFIRE_DETERMINISTIC", "1")
            .envs(extra_env.iter().copied())
            .spawn()
            .unwrap_or_else(|e| panic!("spawn daemon {}: {e}", bin.display()));
        let stdin = child.stdin.take().expect("daemon stdin");
        let stdout = child.stdout.take().expect("daemon stdout");
        // Reader thread: stdout lines → channel (daemon flushes per event).
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = std::io::BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                match std::io::BufRead::read_line(&mut reader, &mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let _ = tx.send(line.trim_end().to_string());
                    }
                }
            }
        });
        // Drain thread: keep the last stderr lines for failure reports.
        let stderr = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let stderr_child = child.stderr.take().expect("daemon stderr");
        let stderr_ref = stderr.clone();
        std::thread::spawn(move || {
            let mut reader = std::io::BufReader::new(stderr_child);
            let mut line = String::new();
            loop {
                line.clear();
                match std::io::BufRead::read_line(&mut reader, &mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let mut log = stderr_ref.lock().unwrap_or_else(|e| e.into_inner());
                        log.push(line.trim_end().to_string());
                        if log.len() > 40 {
                            log.remove(0);
                        }
                    }
                }
            }
        });
        Session {
            stdin,
            lines: rx,
            child,
            stderr,
        }
    }

    fn send(&mut self, v: &Value) {
        writeln!(self.stdin, "{v}").expect("write daemon stdin");
        self.stdin.flush().expect("flush daemon stdin");
    }

    fn recv(&self, context: &str) -> Value {
        let line = self
            .lines
            .recv_timeout(STEP_TIMEOUT)
            .unwrap_or_else(|_| panic!("{context}: timed out waiting for daemon line"));
        serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("{context}: daemon stdout line is not JSON: {line:?}: {e}"))
    }

    fn stderr_tail(&self) -> String {
        self.stderr.lock().unwrap().join("\n")
    }

    fn close(mut self, context: &str) {
        drop(self.stdin);
        let status = self.child.wait().expect("daemon wait");
        assert!(status.success(), "{context}: daemon exited {status}");
    }
}

fn events_for_id<'a>(events: &'a [Value], id: &str) -> Vec<&'a Value> {
    events
        .iter()
        .filter(|e| e.get("id").and_then(|v| v.as_str()) == Some(id))
        .collect()
}

fn token_text(events: &[&Value]) -> String {
    events
        .iter()
        .filter(|e| e.get("type").and_then(|v| v.as_str()) == Some("token"))
        .map(|e| e.get("text").and_then(|v| v.as_str()).unwrap_or(""))
        .collect()
}

fn load_req(model: &str, pp: Option<u64>) -> Value {
    let mut params = serde_json::json!({ "max_seq": 2048 });
    if let Some(pp) = pp {
        params["pp"] = pp.into();
    }
    serde_json::json!({ "type": "load", "model": model, "params": params })
}

fn gen_req(id: &str, attempt: u64, fault_prefill: bool, fault_decode: bool) -> Value {
    let mut req = serde_json::json!({
        "type": "generate",
        "id": id,
        "attempt_id": attempt,
        "prompt": PROMPT,
        "temperature": 0.0,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "thinking_enabled": false,
    });
    if fault_prefill {
        req["test_fault_after_prefill"] = true.into();
    }
    if fault_decode {
        req["test_fault_after_first_decode"] = true.into();
    }
    req
}

fn commit_req(id: &str, attempt: u64) -> Value {
    serde_json::json!({ "type": "commit", "id": id, "attempt_id": attempt })
}

/// Assert the fault request produced exactly one attested error terminal and
/// nothing else client-visible; panic with a rebuild hint when the fault did
/// not fire (daemon built without the feature).
fn assert_fault_terminal(events: &[Value], id: &str, context: &str, stderr_tail: &str) {
    let events = events_for_id(events, id);
    let types: Vec<String> = events
        .iter()
        .map(|e| {
            e.get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("?")
                .to_string()
        })
        .collect();
    if types.iter().any(|t| t == "done") {
        panic!(
            "{context}: fault did not fire for {id} (got done) — rebuild the daemon with \
             --features serve-fault-inject and set HIPFIRE_DAEMON_BIN. stderr tail:\n{}",
            stderr_tail
        );
    }
    assert_eq!(
        types.iter().filter(|t| *t == "gen_start").count(),
        1,
        "{context}: expected one gen_start for {id}, got {types:?}"
    );
    let errors: Vec<&Value> = events
        .iter()
        .filter(|e| e.get("type").and_then(|v| v.as_str()) == Some("error"))
        .copied()
        .collect();
    assert_eq!(
        errors.len(),
        1,
        "{context}: expected exactly one error terminal for {id}, got {types:?}"
    );
    assert_eq!(
        errors[0].get("rolled_back"),
        Some(&Value::Bool(true)),
        "{context}: error terminal must carry attested rolled_back=true: {}",
        errors[0]
    );
    assert!(
        !types.iter().any(|t| t == "done" || t == "aborted"),
        "{context}: fault must not emit done/aborted for {id}: {types:?}"
    );
    assert!(
        !types.iter().any(|t| t == "token"),
        "{context}: fault seam fires before token visibility for {id}: {types:?}"
    );
}

/// Send load and wait for `loaded`.
fn session_load(s: &mut Session, model: &str, pp: Option<u64>, context: &str) {
    s.send(&load_req(model, pp));
    loop {
        let e = s.recv(context);
        if e.get("type").and_then(|v| v.as_str()) == Some("loaded") {
            return;
        }
    }
}

/// Run one generate to its terminal, answering `commit_ready` with `commit`.
/// Fault turns end at `error` before any handshake. Returns the id-scoped
/// events.
fn session_generate(
    s: &mut Session,
    id: &str,
    attempt: u64,
    fault_prefill: bool,
    fault_decode: bool,
    context: &str,
) -> Vec<Value> {
    s.send(&gen_req(id, attempt, fault_prefill, fault_decode));
    let mut scoped = Vec::new();
    loop {
        let e = s.recv(context);
        let ty = e
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string();
        let mine = e.get("id").and_then(|v| v.as_str()) == Some(id);
        if mine && ty == "commit_ready" {
            s.send(&commit_req(id, attempt));
        }
        let terminal = mine && (ty == "done" || ty == "error");
        if mine {
            scoped.push(e);
        }
        if terminal {
            return scoped;
        }
    }
}

/// Send unload and wait for `unloaded`.
fn session_unload(s: &mut Session, context: &str) {
    s.send(&serde_json::json!({ "type": "unload" }));
    loop {
        let e = s.recv(context);
        if e.get("type").and_then(|v| v.as_str()) == Some("unloaded") {
            return;
        }
    }
}

/// Assert one committed `done`. Single-AR `done` carries `finish_reason stop`;
/// the PP `done` envelope has no `finish_reason`, so there the commit is
/// proven by exactly-one-`done` with no `aborted` (the abort path emits the
/// `aborted`+`done(aborted)` pair instead). With `expect_dflash`, the `done`
/// must also carry the spec route's `dflash:true` marker, so a silent AR
/// fallback cannot pass as spec coverage.
fn assert_done_stop(events: &[Value], context: &str, what: &str, pp: bool, expect_dflash: bool) {
    let done: Vec<&Value> = events
        .iter()
        .filter(|e| e.get("type").and_then(|v| v.as_str()) == Some("done"))
        .collect();
    assert_eq!(
        done.len(),
        1,
        "{context}: {what} must end in exactly one done: {:?}",
        events.iter().map(|e| e.get("type")).collect::<Vec<_>>()
    );
    if pp {
        assert!(
            !events
                .iter()
                .any(|e| e.get("type").and_then(|v| v.as_str()) == Some("aborted")),
            "{context}: {what} must commit, not abort"
        );
    } else {
        assert_eq!(
            done[0].get("finish_reason").and_then(|v| v.as_str()),
            Some("stop"),
            "{context}: {what} must commit (finish stop)"
        );
    }
    if expect_dflash {
        assert_eq!(
            done[0].get("dflash"),
            Some(&Value::Bool(true)),
            "{context}: {what} must take the spec route (done.dflash=true)"
        );
    }
}

/// Clean-generate text for one fresh daemon process (load → gen → unload).
fn clean_bytes(
    model: &str,
    pp: Option<u64>,
    extra_env: &[(&str, &str)],
    context: &str,
    expect_dflash: bool,
) -> String {
    let id = "fresh";
    let mut s = Session::spawn(extra_env);
    session_load(&mut s, model, pp, context);
    let events = session_generate(&mut s, id, 7001, false, false, context);
    assert_done_stop(
        &events,
        context,
        "fresh clean generate",
        pp.is_some(),
        expect_dflash,
    );
    let text = token_text(&events_for_id(&events, id));
    assert!(
        !text.is_empty(),
        "{context}: fresh clean generate emitted no tokens"
    );
    session_unload(&mut s, context);
    s.close(context);
    text
}

/// One fault-matrix round in a fresh daemon process: prefill fault, decode
/// fault, then a clean retry whose bytes must equal the fresh baseline.
fn fault_round(
    model: &str,
    pp: Option<u64>,
    extra_env: &[(&str, &str)],
    baseline: &str,
    context: &str,
    expect_dflash: bool,
) {
    let mut s = Session::spawn(extra_env);
    session_load(&mut s, model, pp, context);
    let tail = s.stderr_tail();
    let prefill = session_generate(&mut s, "fault-prefill", 7002, true, false, context);
    assert_fault_terminal(&prefill, "fault-prefill", context, &tail);
    let decode = session_generate(&mut s, "fault-decode", 7003, false, true, context);
    assert_fault_terminal(&decode, "fault-decode", context, &tail);
    let retry = session_generate(&mut s, "retry", 7004, false, false, context);
    assert_done_stop(
        &retry,
        context,
        "post-fault retry",
        pp.is_some(),
        expect_dflash,
    );
    let retry_text = token_text(&events_for_id(&retry, "retry"));
    assert_eq!(
        retry_text, baseline,
        "{context}: post-fault retry bytes must equal the fresh-daemon baseline \
         (failed turn stored no cache and left no drift)"
    );
    session_unload(&mut s, context);
    s.close(context);
}

#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_QWEN35_RESET_MODEL (dense Qwen3.5 MQ4) + daemon built with --features serve-fault-inject"]
fn single_qwen35_reset_rolls_back_and_recovers() {
    let _guard = lock();
    if !has_gpu() {
        eprintln!("skip: no GPU (/dev/kfd and /dev/dri absent)");
        return;
    }
    let Some(model) = model_path("HIPFIRE_QWEN35_RESET_MODEL") else {
        return;
    };
    let model = model.display().to_string();
    let extra_env: &[(&str, &str)] = &[];
    for round in 0..rounds() {
        let ctx = format!("single round {round}");
        let baseline = clean_bytes(&model, None, extra_env, &ctx, false);
        fault_round(&model, None, extra_env, &baseline, &ctx, false);
    }
    // Report identity for the run log.
    let mut fp = HashMap::new();
    fp.insert("model", model);
    fp.insert("prompt", PROMPT.to_string());
    eprintln!("single_qwen35_reset_rolls_back_and_recovers ok: {fp:?}");
}

#[test]
#[ignore = "requires 2x HIP GPU (HIPFIRE_HAVE_2_GPU=1) + HIPFIRE_PP_RESET_MODEL + daemon built with --features serve-fault-inject"]
fn pp_qwen35_reset_rolls_back_and_recovers() {
    let _guard = lock();
    if std::env::var("HIPFIRE_HAVE_2_GPU").ok().as_deref() != Some("1") {
        eprintln!("skip: HIPFIRE_HAVE_2_GPU not set");
        return;
    }
    if !has_gpu() {
        eprintln!("skip: no GPU (/dev/kfd and /dev/dri absent)");
        return;
    }
    let Some(model) = model_path("HIPFIRE_PP_RESET_MODEL") else {
        return;
    };
    let model = model.display().to_string();
    // Two uniform-VRAM devices (preflight rejects imbalanced pairs). Default
    // 0,1; override when a sibling run occupies a device, e.g. 1,2.
    let devices = std::env::var("HIPFIRE_PP_DEVICES").unwrap_or_else(|_| "0,1".to_string());
    let extra_env = [
        ("HIP_VISIBLE_DEVICES", devices.as_str()),
        ("ROCR_VISIBLE_DEVICES", devices.as_str()),
    ];
    for round in 0..rounds() {
        let ctx = format!("pp round {round}");
        // pp:2 load routes every generate through generate_multi
        // (select_generation_route pins pp>1 to PipelineParallel).
        let baseline = clean_bytes(&model, Some(2), &extra_env, &ctx, false);
        fault_round(&model, Some(2), &extra_env, &baseline, &ctx, false);
    }
    let mut fp = HashMap::new();
    fp.insert("model", model);
    fp.insert("prompt", PROMPT.to_string());
    eprintln!("pp_qwen35_reset_rolls_back_and_recovers ok: {fp:?}");
}

#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_QWEN35_RESET_MODEL (dense Qwen3.5 MQ4) + HIPFIRE_DFLASH_DRAFT (DFlash draft) + daemon built with --features serve-fault-inject"]
fn dflash_qwen35_reset_rolls_back_and_recovers() {
    let _guard = lock();
    if !has_gpu() {
        eprintln!("skip: no GPU (/dev/kfd and /dev/dri absent)");
        return;
    }
    let Some(model) = model_path("HIPFIRE_QWEN35_RESET_MODEL") else {
        return;
    };
    let Some(draft) = draft_path() else {
        return;
    };
    let model = model.display().to_string();
    let draft = draft.display().to_string();
    let extra_env = [("HIPFIRE_DFLASH_DRAFT", draft.as_str())];
    for round in 0..rounds() {
        let ctx = format!("dflash round {round}");
        let baseline = clean_bytes(&model, None, &extra_env, &ctx, true);
        fault_round(&model, None, &extra_env, &baseline, &ctx, true);
    }
    // Report identity for the run log.
    let mut fp = HashMap::new();
    fp.insert("model", model);
    fp.insert("draft", draft);
    fp.insert("prompt", PROMPT.to_string());
    eprintln!("dflash_qwen35_reset_rolls_back_and_recovers ok: {fp:?}");
}
