// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native-MTP n-gram takeover fill oracle on a real Flash-Next model.
//!
//! `#[ignore]`d: it needs a real HIP GPU and the `qwen3.8-flash-next-gptq3.mq4`
//! artifact named by `HIPFIRE_MTP_IDENTITY_MODEL` (explicitly running it
//! without one fails rather than passing as model proof).
//!
//! Run:
//!
//! ```text
//! HIPFIRE_MTP_IDENTITY_MODEL=/path/to/qwen3.8-flash-next-gptq3.mq4 \
//! HIPFIRE_DEVICES=0 \
//! cargo test -p hipfire-arch-qwen4 --features reference-parity \
//!     --test mtp_takeover_fill_hw -- --ignored --exact mtp_takeover_fill_hw \
//!     --nocapture --test-threads=1
//! ```
//!
//! Every arm runs `state_parity::run_mtp_takeover_fill_oracle(model, 4096)` in
//! its own fresh child process of this test binary, because route knobs are
//! read from a process-start snapshot:
//!
//! - `batched-fill`: `HIPFIRE_MTP_INCREMENTAL=0`, `HIPFIRE_QWEN4_MTP_BATCHED_FILL=1`;
//! - `serial-fill`: `HIPFIRE_MTP_INCREMENTAL=0`, `HIPFIRE_QWEN4_MTP_BATCHED_FILL=0`;
//! - `interleaved`: `HIPFIRE_MTP_INCREMENTAL=1`, `HIPFIRE_QWEN4_MTP_BATCHED_FILL=0`.
//!
//! Each child prints exactly one `MTP_TAKEOVER_FILL_SUMMARY {json}` line on
//! stdout. The parent requires the batched and serial `windows` arrays to be
//! identical JSON (digests, pending hidden, positions), and for every arm:
//! `rejected_tail.same_state`, `pairing.paired_hidden_row == -1`, the three
//! `pairing` (`rel_l2`, `forced_rel_l2`, `pending_rel_l2`) `< 0.1 *` their
//! shifted/lagging counterparts
//! and a non-empty `native_miss.committed`; the batched and serial arms also
//! need `native_miss.drafts_generated >= 1`. The route is confirmed from the
//! drafter's own `HIPFIRE_MTP_TRACE` window events: n-gram takeover windows
//! must carry the arm's `verify_route` and a native `source=mtp` window must
//! exist, so a silent fallback cannot pass as coverage.
//!
//! The oracle (asserted inside the helper, one load per process) drives these
//! takeover sequences: partial acceptance (1/3), full acceptance (3/3), zero
//! acceptance (0/3) and an EOS accepted at draft index 1, plus first-reject
//! positions 0, 1 and 2. Per window, `consumed` is `accepted + 1` (minus the
//! accepted EOS where one ends the window), `committed` and the target/MTP
//! positions follow from it, and `pending_hidden` is the hidden of the last
//! consumed verify row. The target state is F32 (`GdnStateFormat`), mirroring
//! `run_mtp_fill_digest`, on the automatically selected VMM/legacy KV backend;
//! the report carries `state_format` and `kv_backend`. The raw GDN/KV family
//! digests and the pending-hidden digest must be byte-equal to the one-row AR
//! reference (40 forwards), and the unconsumed physical head K/V/raw-index
//! rows must be unchanged as well as the live family bytes. The report also
//! carries the forced pending/family byte-identical booleans, which are not
//! asserted across the 4-row/1-row schedule. Draft/target pairing is token
//! `p` with the target hidden of `p - 1` (a prompt's position 0 with its own;
//! the pending hidden holds head position - 1, the pairing every draft step
//! runs): the head rows of the takeover and of the one-row forced route must
//! be decisively closer (`rel_l2 < 0.1 *`) to one-row references built with
//! hidden `p - 1` than to ones built with hidden `p`, and the pending hidden
//! after the window (row `P + 3`) closer to the forced route's than to row
//! `P + 2`. The parent additionally requires the batched-fill and serial-fill
//! `windows` arrays to be exactly equal. The interleaved arm repeats the
//! sequence and the native miss; no cross-route digest equality is required of
//! it.

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const MODEL_ENV: &str = "HIPFIRE_MTP_IDENTITY_MODEL";
const ARM_ENV: &str = "HIPFIRE_MTP_TAKEOVER_FILL_ARM";
const ARM_TEST: &str = "mtp_takeover_fill_arm";
const SUMMARY_PREFIX: &str = "MTP_TAKEOVER_FILL_SUMMARY ";
const TRACE_PREFIX: &str = "QWEN4_MTP_TRACE ";
const MAX_SEQ: usize = 4096;

#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    /// `HIPFIRE_MTP_INCREMENTAL`.
    incremental: &'static str,
    /// `HIPFIRE_QWEN4_MTP_BATCHED_FILL`.
    batched_fill: &'static str,
    /// Expected `verify_route` of every n-gram takeover window.
    verify_route: &'static str,
    /// Whether the native miss must have generated drafts.
    needs_drafts: bool,
}

const ARMS: [Arm; 3] = [
    Arm {
        name: "batched-fill",
        incremental: "0",
        batched_fill: "1",
        verify_route: "batched",
        needs_drafts: true,
    },
    Arm {
        name: "serial-fill",
        incremental: "0",
        batched_fill: "0",
        verify_route: "batched",
        needs_drafts: true,
    },
    Arm {
        name: "interleaved",
        incremental: "1",
        batched_fill: "0",
        verify_route: "interleaved",
        needs_drafts: false,
    },
];

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

/// The single summary line of one arm's stdout.
fn summary_of(arm: &Arm, stdout: &str) -> Value {
    let lines: Vec<&str> = stdout
        .lines()
        .filter_map(|line| line.split_once(SUMMARY_PREFIX).map(|(_, json)| json))
        .collect();
    assert_eq!(
        lines.len(),
        1,
        "{} arm printed {} {SUMMARY_PREFIX:?} lines, expected 1\nstdout:\n{stdout}",
        arm.name,
        lines.len()
    );
    let summary: Value = serde_json::from_str(lines[0])
        .unwrap_or_else(|e| panic!("{} arm summary is not JSON ({e}): {}", arm.name, lines[0]));
    assert_eq!(
        summary["arm"].as_str(),
        Some(arm.name),
        "{} arm summary names the wrong arm",
        arm.name
    );
    assert!(
        summary["result"].is_object(),
        "{} arm summary has no helper result: {summary}",
        arm.name
    );
    summary
}

#[test]
fn takeover_summary_accepts_libtest_prefix() {
    let stdout = concat!(
        "test mtp_takeover_fill_arm ... MTP_TAKEOVER_FILL_SUMMARY ",
        "{\"arm\":\"batched-fill\",\"result\":{}}\n",
        "ok\n"
    );
    assert_eq!(summary_of(&ARMS[0], stdout)["result"], json!({}));
}

/// `event == "window"` trace objects emitted by the drafter on stderr.
fn trace_windows(arm: &Arm, stderr: &str) -> Vec<Value> {
    stderr
        .lines()
        .filter_map(|line| line.strip_prefix(TRACE_PREFIX))
        .map(|json| {
            serde_json::from_str::<Value>(json)
                .unwrap_or_else(|e| panic!("{} arm trace is not JSON ({e}): {json}", arm.name))
        })
        .filter(|event| event["event"].as_str() == Some("window"))
        .collect()
}

fn check_trace(arm: &Arm, stderr: &str) {
    let windows = trace_windows(arm, stderr);
    let ngram: Vec<&Value> = windows
        .iter()
        .filter(|w| w["source"].as_str() == Some("ngram"))
        .collect();
    let native: Vec<&Value> = windows
        .iter()
        .filter(|w| w["source"].as_str() == Some("mtp"))
        .collect();
    assert!(
        !ngram.is_empty(),
        "{} arm traced no source=ngram takeover window (silent fallback?)",
        arm.name
    );
    assert!(
        !native.is_empty(),
        "{} arm traced no source=mtp native window (silent fallback?)",
        arm.name
    );
    for window in &ngram {
        assert_eq!(
            window["verify_route"].as_str(),
            Some(arm.verify_route),
            "{} arm ngram window took the wrong route: {window}",
            arm.name
        );
        let baseline = window["baseline"].as_bool().unwrap_or(false);
        assert_eq!(
            baseline,
            arm.verify_route == "batched",
            "{} arm ngram window baseline flag disagrees with its route: {window}",
            arm.name
        );
    }
    eprintln!(
        "{}: {} ngram windows ({}), {} native mtp windows",
        arm.name,
        ngram.len(),
        arm.verify_route,
        native.len()
    );
}

fn check_result(arm: &Arm, result: &Value) {
    let windows = result["windows"]
        .as_array()
        .unwrap_or_else(|| panic!("{} arm result has no windows array", arm.name));
    assert!(
        !windows.is_empty(),
        "{} arm result has no windows",
        arm.name
    );

    assert_eq!(
        result["rejected_tail"]["same_state"],
        Value::Bool(true),
        "{} arm rejected tail changed state: {}",
        arm.name,
        result["rejected_tail"]
    );

    assert_eq!(
        result["pairing"]["paired_hidden_row"],
        json!(-1),
        "{} arm pairing must be token p with the hidden of p - 1",
        arm.name
    );
    // (correct, wrong) pairs: takeover head rows vs the one-row references
    // (paired p-1 vs own-row p), the one-row forced route likewise, and the
    // pending hidden vs the forced route's and vs a pending one row behind.
    for (correct, wrong) in [
        ("rel_l2", "shifted_rel_l2"),
        ("forced_rel_l2", "forced_shifted_rel_l2"),
        ("pending_rel_l2", "pending_lag_rel_l2"),
    ] {
        let near = result["pairing"][correct]
            .as_f64()
            .unwrap_or_else(|| panic!("{} arm pairing.{correct} missing", arm.name));
        let far = result["pairing"][wrong]
            .as_f64()
            .unwrap_or_else(|| panic!("{} arm pairing.{wrong} missing", arm.name));
        assert!(
            near < 0.1 * far,
            "{} arm pairing is not decisively aligned: {correct} {near} vs {wrong} {far}",
            arm.name
        );
    }

    let committed = result["native_miss"]["committed"]
        .as_array()
        .unwrap_or_else(|| panic!("{} arm native_miss.committed missing", arm.name));
    assert!(
        !committed.is_empty(),
        "{} arm native miss committed nothing",
        arm.name
    );
    if arm.needs_drafts {
        let drafts = result["native_miss"]["drafts_generated"]
            .as_u64()
            .unwrap_or_else(|| panic!("{} arm native_miss.drafts_generated missing", arm.name));
        assert!(
            drafts >= 1,
            "{} arm native miss generated no drafts",
            arm.name
        );
    }
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn mtp_takeover_fill_hw() {
    let model = model_path();
    let exe = std::env::current_exe().expect("test binary path");
    let mut windows: Vec<(&'static str, Value)> = Vec::new();
    for arm in ARMS {
        let output = Command::new(&exe)
            .args([
                "--exact",
                ARM_TEST,
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(ARM_ENV, arm.name)
            .env(MODEL_ENV, &model)
            .env_remove("HIPFIRE_MTP_DRAFT_HEAD")
            .env_remove("HIPFIRE_MTP_K")
            .env_remove("HIPFIRE_MTP_SAMPLED_MODE")
            .env("HIPFIRE_MTP_INCREMENTAL", arm.incremental)
            .env("HIPFIRE_QWEN4_MTP_BATCHED_FILL", arm.batched_fill)
            .env("HIPFIRE_MTP_TRACE", "1")
            .stdin(Stdio::null())
            .output()
            .unwrap_or_else(|e| panic!("spawn {} arm: {e}", arm.name));
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        assert!(
            output.status.success(),
            "{} arm failed ({})\nstdout:\n{stdout}\nstderr:\n{stderr}",
            arm.name,
            output.status
        );
        let summary = summary_of(&arm, &stdout);
        println!("{SUMMARY_PREFIX}{summary}");
        let result = &summary["result"];
        check_result(&arm, result);
        check_trace(&arm, &stderr);
        windows.push((arm.name, result["windows"].clone()));
    }

    let batched = &windows[0];
    let serial = &windows[1];
    assert_eq!(
        batched.1, serial.1,
        "{} and {} windows differ (families, pending hidden, positions)",
        batched.0, serial.0
    );
}

/// One arm in a fresh process; spawned by `mtp_takeover_fill_hw`.
#[test]
#[ignore = "child process of mtp_takeover_fill_hw"]
fn mtp_takeover_fill_arm() {
    let name = std::env::var(ARM_ENV).unwrap_or_else(|_| {
        panic!("{ARM_ENV} must name a takeover fill arm; run mtp_takeover_fill_hw instead")
    });
    let arm = ARMS
        .iter()
        .find(|arm| arm.name == name)
        .unwrap_or_else(|| panic!("unknown {ARM_ENV}={name}"));
    let result = hipfire_arch_qwen4::state_parity::run_mtp_takeover_fill_oracle(
        Path::new(&model_path()),
        MAX_SEQ,
    )
    .unwrap_or_else(|e| panic!("{} arm oracle: {e}", arm.name));
    let summary = json!({
        "arm": arm.name,
        "incremental": arm.incremental,
        "batched_fill": arm.batched_fill,
        "verify_route": arm.verify_route,
        "result": result,
    });
    println!("{SUMMARY_PREFIX}{summary}");
}
