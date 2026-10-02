// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 (Flash-Next) prompt-cache and MTP hardware tests.
//!
//! Every test is `#[ignore]`d: they need a real HIP GPU, the Flash-Next
//! model (`HIPFIRE_QWEN4_CACHE_MODEL`, default
//! `~/.hipfire/models/qwen3.8-flash-next.mq4.hfq`) and a release daemon
//! (`HIPFIRE_DAEMON_BIN`, default `target/release/daemon`).
//!
//! One interactive daemon session runs a five-turn conversation where every
//! turn's `messages` purely extend the previous turn, then asserts the
//! `cached_tokens` pattern on each `done`: a pure extension reuses the prefix,
//! except an MTP turn right after an AR turn (the AR decode does not feed the
//! MTP head, so that hit is demoted to a cold miss).  A third test checks that
//! seeded sampled MTP emits seeded AR's exact text.

#![allow(clippy::all)]

use serde_json::Value;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

static HW_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    HW_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

const USERS: [&str; 5] = [
    "Name one prime number and nothing else.",
    "Now name a larger one.",
    "Add those two numbers.",
    "Is the sum even? Answer yes or no.",
    "Say thanks in one word.",
];

fn model_path() -> Option<PathBuf> {
    let path = match std::env::var("HIPFIRE_QWEN4_CACHE_MODEL") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
            PathBuf::from(home).join(".hipfire/models/qwen3.8-flash-next.mq4.hfq")
        }
    };
    if path.is_file() {
        Some(path)
    } else {
        eprintln!("skip: Qwen4 cache model missing at {}", path.display());
        None
    }
}

fn daemon_bin() -> PathBuf {
    match std::env::var("HIPFIRE_DAEMON_BIN") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/release/daemon"),
    }
}

/// Interactive daemon session: requests are sent line-by-line so the
/// two-phase `commit_ready` → `commit` handshake works.
struct Session {
    stdin: std::process::ChildStdin,
    lines: std::sync::mpsc::Receiver<String>,
    child: std::process::Child,
    stderr: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

/// Per-receive ceiling. Flash-Next load reads ~125 GB of weights.
const STEP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2400);

impl Session {
    fn spawn(extra_env: &[(&str, &str)]) -> Self {
        let bin = daemon_bin();
        assert!(
            bin.is_file(),
            "daemon binary missing at {} — build it first: cargo build --release -p hipfire-daemon --locked",
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
        let line = self.lines.recv_timeout(STEP_TIMEOUT).unwrap_or_else(|_| {
            panic!(
                "{context}: timed out waiting for daemon line\n{}",
                self.stderr_tail()
            )
        });
        serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("{context}: daemon stdout line is not JSON: {line:?}: {e}"))
    }

    fn stderr_tail(&self) -> String {
        self.stderr
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .join("\n")
    }

    fn close(mut self, context: &str) {
        self.send(&serde_json::json!({ "type": "unload" }));
        loop {
            let e = self.recv(context);
            if e.get("type").and_then(|v| v.as_str()) == Some("unloaded") {
                break;
            }
        }
        drop(self.stdin);
        let status = self.child.wait().expect("daemon wait");
        assert!(status.success(), "{context}: daemon exited {status}");
    }
}

fn load(s: &mut Session, model: &str, mtp: bool, context: &str) {
    s.send(&serde_json::json!({
        "type": "load",
        "model": model,
        "params": {
            "max_seq": 2048,
            "dflash_mode": "off",
            "mtp_mode": if mtp { "on" } else { "off" },
            "ngram_draft": false,
        },
    }));
    loop {
        let e = s.recv(context);
        match e.get("type").and_then(|v| v.as_str()) {
            Some("loaded") => return,
            Some("error") => panic!("{context}: load failed: {e}\n{}", s.stderr_tail()),
            _ => {}
        }
    }
}

/// One committed turn with `sampling` (temperature/seed/penalty fields merged
/// into the request): returns (`done` event, concatenated token text).
fn turn(
    s: &mut Session,
    id: &str,
    attempt: u64,
    messages: &[Value],
    sampling: &Value,
    context: &str,
) -> (Value, String) {
    let prompt = messages.last().unwrap()["content"].clone();
    let mut request = serde_json::json!({
        "type": "generate",
        "id": id,
        "attempt_id": attempt,
        "prompt": prompt,
        "messages": messages,
        "max_tokens": 96,
        "thinking_enabled": false,
    });
    for (key, value) in sampling.as_object().expect("sampling is an object") {
        request[key] = value.clone();
    }
    s.send(&request);
    let mut text = String::new();
    loop {
        let e = s.recv(context);
        if e.get("id").and_then(|v| v.as_str()) != Some(id) {
            continue;
        }
        match e.get("type").and_then(|v| v.as_str()).unwrap_or("?") {
            "token" => text.push_str(e.get("text").and_then(|v| v.as_str()).unwrap_or("")),
            "commit_ready" => s.send(&serde_json::json!({
                "type": "commit", "id": id, "attempt_id": attempt,
            })),
            "done" => return (e, text),
            "error" => panic!("{context}: turn ended in error: {e}\n{}", s.stderr_tail()),
            _ => {}
        }
    }
}

/// Run the five-turn greedy conversation; `penalties[i]` is turn `i`'s
/// repeat penalty (1.0 → native MTP when loaded; anything else → AR, which
/// applies it). Returns per-turn (`cached_tokens`, `mtp` marker).
fn conversation(extra_env: &[(&str, &str)], mtp: bool, penalties: [f64; 5]) -> Vec<(u64, bool)> {
    let Some(model) = model_path() else {
        return Vec::new();
    };
    let model = model.to_string_lossy().into_owned();
    let mut s = Session::spawn(extra_env);
    load(&mut s, &model, mtp, "load");
    let mut messages: Vec<Value> = Vec::new();
    let mut out = Vec::new();
    for (i, (user, penalty)) in USERS.iter().zip(penalties).enumerate() {
        messages.push(serde_json::json!({ "role": "user", "content": user }));
        let context = format!("turn {}", i + 1);
        let (done, text) = turn(
            &mut s,
            &format!("q4c-{i}"),
            i as u64 + 1,
            &messages,
            &serde_json::json!({ "temperature": 0.0, "repeat_penalty": penalty, "seed": 12345 }),
            &context,
        );
        let cached = done
            .get("cached_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let is_mtp = done.get("mtp").and_then(|v| v.as_bool()) == Some(true);
        eprintln!("{context}: RP={penalty} cached={cached} mtp={is_mtp} text={text:?}");
        assert!(
            !text.trim().is_empty(),
            "{context}: empty text; done={done}"
        );
        messages.push(serde_json::json!({ "role": "assistant", "content": text }));
        out.push((cached, is_mtp));
    }
    s.close("unload");
    out
}

#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_QWEN4_CACHE_MODEL (qwen3.8-flash-next) + release daemon"]
fn qwen4_mtp_ar_prompt_cache_alternation() {
    let _g = lock();
    let turns = conversation(&[], true, [1.0, 1.0, 1.1, 1.0, 1.0]);
    if turns.is_empty() {
        return;
    }
    let cached: Vec<u64> = turns.iter().map(|t| t.0).collect();
    let mtp: Vec<bool> = turns.iter().map(|t| t.1).collect();
    assert_eq!(
        mtp,
        [true, true, false, true, true],
        "route per turn (cached={cached:?})"
    );
    assert_eq!(cached[0], 0, "turn 1 is cold: {cached:?}");
    assert!(cached[1] > 0, "MTP→MTP hits: {cached:?}");
    assert!(cached[2] > 0, "MTP→AR hits: {cached:?}");
    assert_eq!(cached[3], 0, "AR→MTP demotes to a miss: {cached:?}");
    assert!(
        cached[4] > 0,
        "MTP→MTP hits after a cold MTP turn: {cached:?}"
    );
}

#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_QWEN4_CACHE_MODEL (qwen3.8-flash-next) + release daemon"]
fn qwen4_ar_prompt_cache_reuses_prefix() {
    let _g = lock();
    let turns = conversation(&[], false, [1.0; 5]);
    if turns.is_empty() {
        return;
    }
    let cached: Vec<u64> = turns.iter().map(|t| t.0).collect();
    assert_eq!(cached[0], 0, "turn 1 is cold: {cached:?}");
    assert!(
        cached[1..].iter().all(|&c| c > 0),
        "AR turns 2-5 hit: {cached:?}"
    );

    let disabled = conversation(&[("HIPFIRE_QWEN_PROMPT_CACHE", "0")], false, [1.0; 5]);
    let cached: Vec<u64> = disabled.iter().map(|t| t.0).collect();
    assert!(cached.iter().all(|&c| c == 0), "cache disabled: {cached:?}");
}

/// Seeded sampled MTP must emit seeded AR's exact tokens on both verify
/// routes: every emitted token is one draw from the shared sampler RNG, and
/// a 2..8-row verify forward is bitwise the single-row decode. The code case
/// is a batched-verify row that once rounded the final HC read differently.
#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_QWEN4_CACHE_MODEL (qwen3.8-flash-next) + release daemon"]
fn qwen4_seeded_sampled_mtp_matches_ar() {
    let _g = lock();
    let Some(model) = model_path() else {
        return;
    };
    let model = model.to_string_lossy().into_owned();
    let prose = "Write a short story opening about a lighthouse keeper who finds a strange object on the beach.";
    let code = "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two example calls.";
    let cases: Vec<(&str, f64, u64)> = (1..=4u64)
        .map(|seed| (prose, 0.7, seed))
        .chain([(code, 1.0, 6)])
        .collect();
    let run = |mtp: bool, incremental: &str| -> Vec<String> {
        let mut s = Session::spawn(&[
            ("HIPFIRE_QWEN_PROMPT_CACHE", "0"),
            ("HIPFIRE_MTP_INCREMENTAL", incremental),
        ]);
        load(&mut s, &model, mtp, "load");
        let mut texts = Vec::new();
        for (index, &(prompt, temperature, seed)) in cases.iter().enumerate() {
            let context = format!("mtp={mtp} incremental={incremental} case={index}");
            let messages = [serde_json::json!({ "role": "user", "content": prompt })];
            let sampling =
                serde_json::json!({ "temperature": temperature, "top_p": 0.95, "seed": seed });
            let attempt = index as u64 + 1;
            let (done, text) = turn(
                &mut s,
                &format!("q4s-{index}"),
                attempt,
                &messages,
                &sampling,
                &context,
            );
            let is_mtp = done.get("mtp").and_then(|v| v.as_bool()) == Some(true);
            assert_eq!(is_mtp, mtp, "{context}: route; done={done}");
            texts.push(text);
        }
        s.close("unload");
        texts
    };
    let ar = run(false, "1");
    assert_eq!(run(true, "1"), ar, "interleaved verify");
    assert_eq!(run(true, "0"), ar, "batched verify");
}
