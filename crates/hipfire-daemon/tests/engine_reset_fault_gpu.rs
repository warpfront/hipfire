// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! HIPF-563: the engine batch driver fails closed when a cancel's rollback
//! is not attested (HIPF-533), and a core-queued abort stamps `loop legacy`
//! (HIPF-518). Ignored: real GPU + a Qwen4 fixture (`HIPFIRE_ENGINE_FIXTURE`).
//! Run with `--features serve-fault-inject -- --ignored`.
//!
//! The daemon binary is driven over its JSONL protocol; an `abort` carrying
//! `test_fault_engine_reset: true` makes the next engine rollback fail.
//! - x: abort a resident request at a token (tick settle site);
//! - xc: abort at `commit_ready` (`settle_terminals` site).
//! Both must give the cancelled request one fail-closed `error`
//! (`rolled_back: false`, no `done`), then `BatchDriveError::Poisoned`: the
//! next request is refused with "state poisoned".
//! - q: 5 requests on 4 lanes, the queued 5th aborted: `done` with `loop legacy`.
#![cfg(feature = "serve-fault-inject")]

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;

const COUNT: &str = "List the numbers from 1 to 200, one per line, as digits.";
const YES: &str = "Reply with exactly the word: yes";

struct Daemon {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<Value>,
}

impl Daemon {
    fn load(model: &str) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_daemon"))
            .env("HIPFIRE_ENGINE_LOOP", "engine")
            .env("HIPFIRE_PLE_FORMAT", "q4")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn daemon");
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if let Ok(v) = serde_json::from_str::<Value>(&line) {
                    if tx.send(v).is_err() {
                        break;
                    }
                }
            }
        });
        let mut d = Self { child, stdin, rx };
        // mtp off: with MTP a client cancel keeps the fed state and never rolls back.
        d.send(json!({"type": "load", "model": model, "params": {
            "max_seq": 4096, "mtp_mode": "off", "continuous_batch_size": 4,
            "serve_vmm_batch": {"sampled": true}}}));
        loop {
            let e = d.recv();
            match e["type"].as_str() {
                Some("loaded") => break,
                Some("error") => panic!("load failed: {e}"),
                _ => {}
            }
        }
        // Precondition: the fixture serves on the engine loop (else every
        // assertion below would read as a product defect).
        d.generate("W1", YES, 8);
        let w = d.run(&["W1"], None, None, None);
        let w = &w[0].1[0];
        assert!(
            w["type"] == "done" && w["loop"] == "engine",
            "fixture {model} does not serve on the engine loop (needs a Qwen4 \
             Flash-Next file + HIPFIRE_ENGINE_LOOP=engine): {w}"
        );
        d
    }

    fn send(&mut self, msg: Value) {
        writeln!(self.stdin, "{msg}").unwrap();
        self.stdin.flush().unwrap();
    }

    fn recv(&self) -> Value {
        self.rx
            .recv_timeout(Duration::from_secs(600))
            .expect("daemon event (exited or stalled)")
    }

    fn generate(&mut self, id: &str, prompt: &str, max_tokens: u32) {
        self.send(json!({"type": "generate", "id": id, "attempt_id": 1,
            "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
            "max_tokens": max_tokens, "thinking_enabled": false, "serve_continuous_batch": true}));
    }

    fn abort(&mut self, id: &str, fault: bool) {
        self.send(json!({"type": "abort", "id": id, "attempt_id": 1,
            "test_fault_engine_reset": fault}));
    }

    /// Run until every id in `ids` has its terminal; commit every
    /// `commit_ready` except `abort_commit`'s (aborted with the fault), abort
    /// `abort_tok.0` with the fault at its `abort_tok.1`-th token, and abort
    /// `abort_queued.1` (no fault) at `abort_queued.0`'s first token.
    /// Returns each id's terminal events (`done` / `error`).
    fn run(
        &mut self,
        ids: &[&str],
        abort_tok: Option<(&str, usize)>,
        abort_commit: Option<&str>,
        abort_queued: Option<(&str, &str)>,
    ) -> Vec<(String, Vec<Value>)> {
        let mut out: Vec<(String, Vec<Value>, usize)> =
            ids.iter().map(|id| (id.to_string(), vec![], 0)).collect();
        let mut pending = ids.len();
        while pending > 0 {
            let e = self.recv();
            let Some(i) = out.iter().position(|(id, ..)| e["id"] == id.as_str()) else {
                continue;
            };
            let id = out[i].0.clone();
            match e["type"].as_str().unwrap_or("") {
                "token" | "reasoning" => {
                    out[i].2 += 1;
                    let n = out[i].2;
                    if abort_tok == Some((id.as_str(), n)) {
                        self.abort(&id, true);
                    }
                    if let Some((first, queued)) = abort_queued {
                        if first == id && n == 1 {
                            self.abort(queued, false);
                        }
                    }
                }
                "commit_ready" => {
                    if abort_commit == Some(id.as_str()) {
                        self.abort(&id, true);
                    } else {
                        self.send(json!({"type": "commit", "id": id, "attempt_id": 1}));
                    }
                }
                "done" | "error" => {
                    out[i].1.push(e);
                    pending -= 1;
                }
                _ => {}
            }
        }
        // A late second terminal would show up here.
        std::thread::sleep(Duration::from_millis(500));
        while let Ok(e) = self.rx.try_recv() {
            if let Some(i) = out.iter().position(|(id, ..)| e["id"] == id.as_str()) {
                if matches!(e["type"].as_str(), Some("done" | "error")) {
                    out[i].1.push(e);
                }
            }
        }
        out.into_iter().map(|(id, t, _)| (id, t)).collect()
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// `HIPFIRE_ENGINE_FIXTURE`, else the `qwen3.8:flash-next` file in the model store.
fn fixture() -> String {
    std::env::var("HIPFIRE_ENGINE_FIXTURE").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("HOME");
        format!("{home}/.hipfire/models/qwen3.8-flash-next-base.mq4-xts")
    })
}

/// The cancelled request's one fail-closed terminal, then the poisoned batch.
fn assert_fail_closed_then_poisoned(d: &mut Daemon, terminals: &[Value]) {
    assert_eq!(terminals.len(), 1, "exactly one terminal: {terminals:?}");
    let t = &terminals[0];
    assert_eq!(t["type"], "error", "no done: {t}");
    assert_eq!(t["rolled_back"], false, "{t}");
    let message = t["message"].as_str().unwrap_or("");
    assert!(message.contains("injected engine reset fault"), "{t}");
    // Err(BatchDriveError::Poisoned): the daemon drops the scheduler.
    d.generate("P1", YES, 16);
    let p = d.run(&["P1"], None, None, None);
    let p = &p[0].1[0];
    assert_eq!(p["type"], "error", "{p}");
    assert!(
        p["message"]
            .as_str()
            .is_some_and(|m| m.contains("state poisoned")),
        "{p}"
    );
}

#[test]
#[ignore = "real GPU + Qwen4 fixture"]
fn engine_cancel_with_failed_reset_poisons_the_batch() {
    let model = fixture();

    // x: tick site.
    let mut d = Daemon::load(&model);
    d.generate("X1", COUNT, 200);
    let r = d.run(&["X1"], Some(("X1", 20)), None, None);
    assert_fail_closed_then_poisoned(&mut d, &r[0].1);
    drop(d);

    // xc: settle_terminals site.
    let mut d = Daemon::load(&model);
    d.generate("XC1", YES, 48);
    let r = d.run(&["XC1"], None, Some("XC1"), None);
    assert_fail_closed_then_poisoned(&mut d, &r[0].1);
    drop(d);

    // q: a core-queued abort is never admitted: loop legacy.
    let mut d = Daemon::load(&model);
    let ids = ["Q1", "Q2", "Q3", "Q4", "Q5"];
    for id in ids {
        d.generate(id, COUNT, 64);
    }
    let r = d.run(&ids, None, None, Some(("Q1", "Q5")));
    for (id, t) in &r {
        assert_eq!(t.len(), 1, "{id}: {t:?}");
        assert_eq!(t[0]["type"], "done", "{id}: {}", t[0]);
    }
    let q5 = &r[4].1[0];
    assert_eq!(q5["finish_reason"], "aborted", "{q5}");
    assert_eq!(q5["loop"], "legacy", "{q5}");
    assert_eq!(r[0].1[0]["loop"], "engine", "{}", r[0].1[0]);
}
