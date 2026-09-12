// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! G4.8 vision/OCR request-lifecycle matrix: Qwen35-VL and dots.ocr
//! cancellation + error epilogues.
//!
//! Covers every vision loop exit named in the G4.8 plan: cancel before
//! prefill, cancel during decode, forward/argmax/spec faults — each paired
//! with its fail-closed counterpart (canonical rollback, exactly one
//! terminal, second-turn reuse equal to a fresh load, unload/reload clean).
//!
//! # Fixtures (real hardware, real artifacts — no emulation)
//!
//! - `HIPFIRE_VISION_FIXTURE`: `"TEXT_HFQ,VISION_HFQ"` — e.g.
//!   `~/.hipfire/models/qwen3.8-27b.mq4-xt,~/.hipfire/models/qwen3.8-27b-vision.hfq`.
//!   The comma-separated pair is required: trunk first, tower sidecar second.
//! - `HIPFIRE_DOTS_OCR_FIXTURE`: path to the dots.ocr artifact, e.g.
//!   `~/.hipfire/models/dots-ocr.q8.hfq`.
//! - Images (override with `HIPFIRE_VISION_IMAGE` / `HIPFIRE_DOTS_IMAGE`):
//!   `benchmarks/vision/images/general_qa.jpg` (VL) and
//!   `benchmarks/images/dots_ocr_smoke_001.jpg` (dots.ocr). Both are
//!   in-repo real photographs, not synthetic fixtures.
//!
//! # Invocation (parent runs on gfx1201)
//!
//! ```sh
//! HIPFIRE_VISION_FIXTURE="$HOME/.hipfire/models/qwen3.8-27b.mq4-xt,$HOME/.hipfire/models/qwen3.8-27b-vision.hfq" \
//! HIPFIRE_DOTS_OCR_FIXTURE="$HOME/.hipfire/models/dots-ocr.q8.hfq" \
//! cargo test -p hipfire-generate --locked --test vision_lifecycle_tests -- --ignored
//! ```
//!
//! # Design
//!
//! `generate_vl` / `generate_vl_dots_ocr` write to `&mut std::io::Stdout`,
//! which cannot be substituted in-process — so each matrix runs in a CHILD
//! process (this same test binary re-executed with
//! `HIPFIRE_VISION_LIFECYCLE_CHILD=vl|dots`), whose piped stdout the parent
//! parses for the exact terminal accounting. Abort injection crosses the
//! pipe: the child runs a stdin reader thread applying `ABORT <id>
//! <attempt>` via `apply_terminal_control`; the parent sends it after
//! `READY <id>` (prefill cancel) or after the first `token` line
//! (decode cancel). Faults are armed in-process via `common::arm_vision_fault` /
//! `HIPFIRE_DOTS_FAULT` hooks, which exercise the production error
//! epilogues without touching production behavior when unset.

#![allow(clippy::all)]

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Duration;

// ── shared: fixture-env parsing (CPU-testable) ──────────────────────────

/// Split `HIPFIRE_VISION_FIXTURE` (`"TEXT,VISION"`) into the trunk/sidecar
/// pair. Exactly one comma is required — a bare path is a configuration
/// error, not a default (silently running trunk-only would void every
/// vision assertion below).
fn split_fixture_pair(var: &str, value: &str) -> Result<(String, String), String> {
    let mut parts = value.split(',');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(text), Some(vision), None)
            if !text.trim().is_empty() && !vision.trim().is_empty() =>
        {
            Ok((text.trim().to_string(), vision.trim().to_string()))
        }
        _ => Err(format!(
            "{var} must be \"TEXT_HFQ,VISION_HFQ\" (got {value:?})"
        )),
    }
}

#[test]
fn vision_fixture_pair_parses_trunk_and_sidecar() {
    let (text, vision) = split_fixture_pair(
        "HIPFIRE_VISION_FIXTURE",
        "/m/qwen3.8-27b.mq4-xt,/m/qwen3.8-27b-vision.hfq",
    )
    .expect("valid pair");
    assert_eq!(text, "/m/qwen3.8-27b.mq4-xt");
    assert_eq!(vision, "/m/qwen3.8-27b-vision.hfq");
}

#[test]
fn vision_fixture_pair_rejects_bare_or_triple_paths() {
    assert!(split_fixture_pair("HIPFIRE_VISION_FIXTURE", "/m/only-text.hfq").is_err());
    assert!(split_fixture_pair("HIPFIRE_VISION_FIXTURE", "/a,/b,/c").is_err());
    assert!(split_fixture_pair("HIPFIRE_VISION_FIXTURE", ",/b").is_err());
    assert!(split_fixture_pair("HIPFIRE_VISION_FIXTURE", "/a,").is_err());
}

// ── shared: transcript accounting (CPU-testable shape) ──────────────────

/// One parsed JSONL event line carrying a request id.
#[derive(Debug, Clone)]
struct Event {
    id: String,
    typ: String,
    finish_reason: Option<String>,
    rolled_back: Option<bool>,
    text: Option<String>,
}

fn parse_event(line: &str) -> Option<Event> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    if v.get("id").and_then(|id| id.as_str()).is_none() {
        return None;
    }
    Some(Event {
        id: v["id"].as_str().unwrap_or("").to_string(),
        typ: v["type"].as_str().unwrap_or("").to_string(),
        finish_reason: v
            .get("finish_reason")
            .and_then(|r| r.as_str())
            .map(str::to_string),
        rolled_back: v.get("rolled_back").and_then(|r| r.as_bool()),
        text: v.get("text").and_then(|t| t.as_str()).map(str::to_string),
    })
}

/// Terminal outcome for one request id: exactly one of success / cancel /
/// error — the G4 exactly-once contract.
#[derive(Debug, PartialEq, Eq)]
enum TerminalOutcome {
    Success,
    Cancel,
    Error,
}

fn classify_terminal(events: &[Event]) -> Option<TerminalOutcome> {
    let mut done = 0;
    let mut aborted = 0;
    let mut aborted_done = 0;
    let mut error = 0;
    for e in events {
        match e.typ.as_str() {
            "done" if e.finish_reason.as_deref() == Some("aborted") => aborted_done += 1,
            "done" => done += 1,
            "aborted" => aborted += 1,
            "error" => error += 1,
            _ => {}
        }
    }
    match (done, aborted, aborted_done, error) {
        (1, 0, 0, 0) => Some(TerminalOutcome::Success),
        (0, 1, 1, 0) => Some(TerminalOutcome::Cancel),
        (0, 0, 0, 1) => Some(TerminalOutcome::Error),
        _ => None,
    }
}

#[test]
fn terminal_classifier_accepts_exactly_one_outcome() {
    let mk = |typ: &str, finish: Option<&str>| Event {
        id: "t".into(),
        typ: typ.into(),
        finish_reason: finish.map(str::to_string),
        rolled_back: None,
        text: None,
    };
    assert_eq!(
        classify_terminal(&[mk("done", None)]),
        Some(TerminalOutcome::Success)
    );
    assert_eq!(
        classify_terminal(&[mk("aborted", None), mk("done", Some("aborted"))]),
        Some(TerminalOutcome::Cancel)
    );
    assert_eq!(
        classify_terminal(&[mk("error", None)]),
        Some(TerminalOutcome::Error)
    );
    // Doubles and mixtures are contract violations, never an outcome.
    assert_eq!(
        classify_terminal(&[mk("done", None), mk("done", None)]),
        None
    );
    assert_eq!(
        classify_terminal(&[mk("error", None), mk("done", None)]),
        None
    );
    assert_eq!(classify_terminal(&[mk("aborted", None)]), None);
}

const CHILD_ENV: &str = "HIPFIRE_VISION_LIFECYCLE_CHILD";
const CHILD_TEST: &str = "vision_lifecycle_child";

/// Re-execute this test binary as a matrix child with piped stdio.
fn spawn_child(scenario: &str) -> (Child, BufReader<ChildStdout>, ChildStdin) {
    let exe = std::env::current_exe().expect("current test exe");
    let mut child = Command::new(exe)
        .arg("--exact")
        .arg(CHILD_TEST)
        .arg("--nocapture")
        .env(CHILD_ENV, scenario)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn matrix child");
    let stdout = child.stdout.take().expect("child stdout");
    let stdin = child.stdin.take().expect("child stdin");
    // Watchdog: a wedged GPU child must fail the test, never hang it.
    let pid = child.id();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(1800));
        let _ = Command::new("kill").arg("-9").arg(pid.to_string()).status();
    });
    (child, BufReader::new(stdout), stdin)
}

/// Parent-side transcript: parsed events + section markers in order.
#[derive(Default)]
struct Transcript {
    events: Vec<Event>,
    sections: Vec<String>,
}
fn events_for(t: &Transcript, id: &str) -> Vec<Event> {
    t.events.iter().filter(|e| e.id == id).cloned().collect()
}

fn gen_starts(t: &Transcript, id: &str) -> usize {
    events_for(t, id)
        .iter()
        .filter(|e| e.typ == "gen_start")
        .count()
}

fn token_text(t: &Transcript, id: &str) -> String {
    events_for(t, id)
        .iter()
        .filter_map(|e| e.text.clone())
        .collect::<String>()
}

fn assert_outcome(t: &Transcript, id: &str, want: TerminalOutcome) {
    let ev = events_for(t, id);
    assert_eq!(
        classify_terminal(&ev),
        Some(want),
        "request {id}: terminal contract violated ({} events: {:?})",
        ev.len(),
        ev.iter().map(|e| e.typ.clone()).collect::<Vec<_>>(),
    );
    assert_eq!(
        gen_starts(t, id),
        1,
        "request {id}: must start exactly once"
    );
}

// ── child entry point ───────────────────────────────────────────────────

/// Hidden workhorse: runs the named matrix when (and only when) the parent
/// spawned this binary with `HIPFIRE_VISION_LIFECYCLE_CHILD` set. A normal
/// `cargo test` run executes this test with the env unset and passes
/// vacuously without touching the GPU.
#[test]
fn vision_lifecycle_child() {
    let scenario = std::env::var(CHILD_ENV).unwrap_or_default();
    if scenario.is_empty() {
        return;
    }
    match scenario.as_str() {
        "vl" => child_vl_matrix(),
        "dots" => child_dots_matrix(),
        other => panic!("unknown matrix {other:?}"),
    }
}

// ── ignored parent tests ────────────────────────────────────────────────

/// Device-global VRAM assertions cannot run concurrently on one GPU, and two
/// 27B-class loads do not fit side by side — serialize the matrices.
static MATRIX_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Qwen35-VL lifecycle matrix. Requires `HIPFIRE_VISION_FIXTURE` (see
/// module docs). Ignored: ~2×27B loads plus nine short turns on gfx1201.
#[test]
#[ignore = "requires HIP GPU + HIPFIRE_VISION_FIXTURE (qwen3.8-27b.mq4-xt + qwen3.8-27b-vision.hfq)"]
fn vl_lifecycle_matrix() {
    // Recover (don't propagate) poisoning: a failed matrix must not wedge its
    // sibling — each child re-establishes its own GPU state from scratch.
    let _matrix_guard = MATRIX_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (mut child, mut reader, mut stdin) = spawn_child("vl");
    // Drive abort injection: prefill cancels fire on READY, decode cancels
    // on the first streamed token.
    let mut line = String::new();
    let mut transcript = Transcript::default();
    let mut t6_tokens = 0;
    let mut aborted_t6 = false;
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
        let trimmed = line.trim().to_string();
        if let Some(rest) = trimmed.strip_prefix("SECTION ") {
            transcript.sections.push(rest.to_string());
        } else if trimmed.starts_with('{') {
            if let Some(e) = parse_event(&trimmed) {
                if e.id == "g48-vl-t2" && e.typ == "gen_start" {
                    writeln!(stdin, "ABORT g48-vl-t2 2").expect("abort t2");
                    stdin.flush().expect("flush abort");
                }
                if e.id == "g48-vl-t6" && e.typ == "token" {
                    t6_tokens += 1;
                    if t6_tokens == 1 && !aborted_t6 {
                        aborted_t6 = true;
                        writeln!(stdin, "ABORT g48-vl-t6 6").expect("abort t6");
                        stdin.flush().expect("flush abort");
                    }
                }
                transcript.events.push(e);
            }
        } else if trimmed.strip_prefix("READY ").is_some() {
            // READY markers carry no JSON; nothing to record.
        }
    }
    let status = child.wait().expect("child wait");
    assert!(status.success(), "vl matrix child failed: {status}");

    for section in [
        "load-ok",
        "g48-vl-t1",
        "g48-vl-t2",
        "g48-vl-t3",
        "g48-vl-t4",
        "unload-ok",
        "reload-ok",
        "g48-vl-t5",
        "g48-vl-t6",
        "g48-vl-t7",
        "g48-vl-t8",
        "g48-vl-t9",
    ] {
        assert!(
            transcript.sections.contains(&section.to_string()),
            "missing SECTION {section} (got {:?})",
            transcript.sections
        );
    }
    // Positive route: baseline success.
    assert_outcome(&transcript, "g48-vl-t1", TerminalOutcome::Success);
    // Cancel before prefill: zero-token cancelled pair, never done.
    assert_outcome(&transcript, "g48-vl-t2", TerminalOutcome::Cancel);
    // Injected prefill fault: single error, attested rollback.
    assert_outcome(&transcript, "g48-vl-t3", TerminalOutcome::Error);
    let t3err: Vec<_> = events_for(&transcript, "g48-vl-t3")
        .into_iter()
        .filter(|e| e.typ == "error")
        .collect();
    assert_eq!(t3err.len(), 1);
    assert_eq!(
        t3err[0].rolled_back,
        Some(true),
        "prefill fault must attest rollback"
    );
    // Every turn that recovers from a cancel or an injected fault must decode
    // exactly what an untouched fresh context decodes. The reference is t5 —
    // the turn after a full unload/reload, whose buffers provably cannot hold
    // state from an earlier turn — so an equality failure here is residue from
    // the cancel/fault path and nothing else.
    //
    // t1 is deliberately NOT the reference: the warmup turn leaves the context
    // populated, so t1 decodes APPENDED at a nonzero sequence offset (different
    // rope phase, different tile count) while t4/t5/t7/t9 decode fresh.
    // Requiring byte equality across that offset asserts something no
    // quantized-KV tier guarantees: measured top-1 margins at the divergent
    // step are 0.24-1.39 logits, so the argmax flip lands differently per tier
    // (Asym3 happened to agree; Fwht2/Fwht3/Fwht4 all pick the other token,
    // and all four tiers agree with each other in each context).
    let t1 = token_text(&transcript, "g48-vl-t1");
    assert!(!t1.is_empty(), "baseline appended turn streamed no text");
    let fresh = token_text(&transcript, "g48-vl-t5");
    assert!(!fresh.is_empty(), "fresh-load turn streamed no text");
    // Reuse after cancel + injected prefill fault.
    assert_eq!(token_text(&transcript, "g48-vl-t4"), fresh);
    // Cancel during decode, then reuse.
    assert_outcome(&transcript, "g48-vl-t6", TerminalOutcome::Cancel);
    assert_eq!(token_text(&transcript, "g48-vl-t7"), fresh);
    // Injected decode fault, then reuse.
    assert_outcome(&transcript, "g48-vl-t8", TerminalOutcome::Error);
    assert_eq!(token_text(&transcript, "g48-vl-t9"), fresh);
}

/// dots.ocr lifecycle matrix (AR + n-gram spec + text-only paths).
/// Requires `HIPFIRE_DOTS_OCR_FIXTURE`. Ignored: needs HIP GPU.
#[test]
#[ignore = "requires HIP GPU + HIPFIRE_DOTS_OCR_FIXTURE (dots-ocr.q8.hfq)"]
fn dots_ocr_lifecycle_matrix() {
    let _matrix_guard = MATRIX_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (mut child, mut reader, mut stdin) = spawn_child("dots");
    let mut line = String::new();
    let mut transcript = Transcript::default();
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut aborted: HashMap<String, bool> = HashMap::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
        let trimmed = line.trim().to_string();
        if let Some(rest) = trimmed.strip_prefix("SECTION ") {
            transcript.sections.push(rest.to_string());
        } else if trimmed.starts_with('{') {
            if let Some(e) = parse_event(&trimmed) {
                if e.typ == "gen_start" && (e.id == "g48-dots-d2") {
                    writeln!(stdin, "ABORT g48-dots-d2 102").expect("abort d2");
                    stdin.flush().expect("flush abort");
                }
                if e.typ == "token" && (e.id == "g48-dots-d9" || e.id == "g48-dots-x2") {
                    let n = counts.entry(e.id.clone()).or_insert(0);
                    *n += 1;
                    if *n == 1 && !aborted.get(&e.id).copied().unwrap_or(false) {
                        aborted.insert(e.id.clone(), true);
                        let attempt = if e.id == "g48-dots-d9" { 109 } else { 203 };
                        writeln!(stdin, "ABORT {} {attempt}", e.id).expect("abort decode");
                        stdin.flush().expect("flush abort");
                    }
                }
                transcript.events.push(e);
            }
        }
    }
    let status = child.wait().expect("child wait");
    assert!(status.success(), "dots matrix child failed: {status}");

    for section in [
        "load-ok",
        "g48-dots-d1",
        "g48-dots-d2",
        "g48-dots-d3",
        "g48-dots-d4",
        "g48-dots-d5",
        "g48-dots-d6",
        "g48-dots-d7",
        "g48-dots-d8",
        "g48-dots-d9",
        "g48-dots-d10",
        "unload-ok",
        "reload-spec-ok",
        "g48-dots-s1",
        "g48-dots-s2",
        "g48-dots-s3",
        "g48-dots-x1",
        "g48-dots-x2",
        "g48-dots-x3",
        "unload2-ok",
    ] {
        assert!(
            transcript.sections.contains(&section.to_string()),
            "missing SECTION {section} (got {:?})",
            transcript.sections
        );
    }
    let d1 = token_text(&transcript, "g48-dots-d1");
    assert!(!d1.is_empty(), "baseline turn streamed no text");
    assert_outcome(&transcript, "g48-dots-d1", TerminalOutcome::Success);
    assert_outcome(&transcript, "g48-dots-d2", TerminalOutcome::Cancel);
    assert_outcome(&transcript, "g48-dots-d3", TerminalOutcome::Error);
    for id in ["g48-dots-d4", "g48-dots-d6", "g48-dots-d8", "g48-dots-d10"] {
        assert_outcome(&transcript, id, TerminalOutcome::Success);
        assert_eq!(token_text(&transcript, id), d1, "{id} must equal baseline");
    }
    for id in ["g48-dots-d5", "g48-dots-d7"] {
        assert_outcome(&transcript, id, TerminalOutcome::Error);
        let errs: Vec<_> = events_for(&transcript, id)
            .into_iter()
            .filter(|e| e.typ == "error")
            .collect();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].rolled_back, Some(true), "{id} must attest rollback");
    }
    assert_outcome(&transcript, "g48-dots-d9", TerminalOutcome::Cancel);
    // Spec path: success, injected spec fault (error, never done), reuse.
    let s1 = token_text(&transcript, "g48-dots-s1");
    assert_outcome(&transcript, "g48-dots-s1", TerminalOutcome::Success);
    assert_eq!(s1, d1, "spec decode must be byte-identical to AR");
    assert_outcome(&transcript, "g48-dots-s2", TerminalOutcome::Error);
    assert_outcome(&transcript, "g48-dots-s3", TerminalOutcome::Success);
    assert_eq!(token_text(&transcript, "g48-dots-s3"), d1);
    // Text-only path: success, decode cancel, reuse.
    let x1 = token_text(&transcript, "g48-dots-x1");
    assert_outcome(&transcript, "g48-dots-x1", TerminalOutcome::Success);
    assert_outcome(&transcript, "g48-dots-x2", TerminalOutcome::Cancel);
    assert_outcome(&transcript, "g48-dots-x3", TerminalOutcome::Success);
    assert_eq!(token_text(&transcript, "g48-dots-x3"), x1);
}

// ── child-side GPU implementation ───────────────────────────────────────
// Everything below runs only in the matrix child (guarded by CHILD_ENV).

#[cfg(test)]
fn emit_marker(text: &str) {
    println!("{text}");
    std::io::stdout().flush().expect("flush marker");
}

#[cfg(test)]
fn vram_free_bytes(gpu: &rdna_compute::Gpu) -> usize {
    gpu.hip.get_vram_info().expect("child VRAM query").0 as usize
}

/// Pay the one-time ROCm first-allocation reservation up front (mirrors the
/// DSpark seam tests) so `vram_base` measures only what the load under test
/// owns. 64 MiB slack on the unload comparisons, same as DSpark.
#[cfg(test)]
const VRAM_SLACK_BYTES: usize = 64 << 20;

#[cfg(test)]
fn warm_first_alloc(gpu: &mut rdna_compute::Gpu) {
    const WARM: usize = 1 << 20;
    let t = gpu
        .upload_raw(&vec![0u8; WARM], &[WARM])
        .expect("first-alloc warm-up upload");
    gpu.free_tensor(t).expect("first-alloc warm-up free");
    gpu.drain_pool();
}

#[cfg(test)]
fn spawn_committer(id: &str, attempt: u64) {
    use hipfire_engine::terminal::{apply_terminal_control, terminal_control};
    let id = id.to_string();
    std::thread::spawn(move || {
        for _ in 0..12_000 {
            let ready = terminal_control()
                .mu
                .lock()
                .map(|g| {
                    g.active
                        .as_ref()
                        .is_some_and(|a| a.id == id && a.attempt_id == attempt && a.ready)
                })
                .unwrap_or(false);
            if ready {
                apply_terminal_control("commit", &id, attempt);
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    });
}

#[cfg(test)]
fn spawn_abort_injector() {
    use hipfire_engine::terminal::apply_terminal_control;
    use std::io::BufRead as _;
    std::thread::spawn(|| {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 3 && parts[0] == "ABORT" {
                if let Ok(attempt) = parts[2].parse::<u64>() {
                    apply_terminal_control("abort", parts[1], attempt);
                }
            }
        }
    });
}

#[cfg(test)]
fn admit_and_load(
    gpu: &mut rdna_compute::Gpu,
    path: &str,
    vision: Option<&str>,
    max_seq: usize,
    spec: hipfire_runtime::loader_api::SpecLoadCfg,
) -> hipfire_loader::LoadedModel {
    use hipfire_config::Deepseek4ComputePlacement;
    let admission = hipfire_loader::admission::admit_source(
        path,
        1,
        1,
        None,
        None,
        gpu.arch.as_str(),
        vision,
        None,
        max_seq,
    )
    .expect("admit fixture source");
    hipfire_loader::load_admitted_with_gemma4_drafter(
        admission,
        path,
        max_seq,
        None,
        Deepseek4ComputePlacement::Single,
        None,
        None,
        hipfire_loader::GEMMA4_EAGLE_DRAFT_LEN,
        None,
        None,
        None,
        &hipfire_runtime::loader_api::CaskConfig::default(),
        1,
        spec,
        gpu,
    )
    .expect("load fixture model")
}

#[cfg(test)]
fn repo_image(default: &str, override_var: &str) -> String {
    if let Ok(p) = std::env::var(override_var) {
        return p;
    }
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{manifest}/../../{default}")
}

#[cfg(test)]
fn vl_params<'a>(
    id: &'a str,
    prompt: &'a str,
    image: &'a str,
    max_tokens: usize,
) -> hipfire_generate::vision::GenerateVLParams<'a> {
    use hipfire_generate::vision::ImageSource;
    use hipfire_runtime::prompt_frame::AssistantPrefix;
    hipfire_generate::vision::GenerateVLParams {
        id,
        prompt,
        system_prompt: None,
        image_source: ImageSource::Path(image),
        temp: 0.0,
        top_p: 1.0,
        max_tokens,
        repeat_penalty: 1.0,
        repeat_window: 0,
        max_think_tokens: 0,
        assistant_prefix: AssistantPrefix::Plain,
        seed: 0xC10C,
    }
}

/// One VL turn: terminal-control activation, optional committer (success
/// turns only — error/cancel turns must observe no commit), READY marker
/// for parent-driven abort injection, generate, teardown.
#[cfg(test)]
fn vl_turn(
    m: &mut hipfire_loader::LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    id: &str,
    attempt: u64,
    image: &str,
    prompt: &str,
    max_tokens: usize,
    commit: bool,
) {
    use hipfire_engine::terminal::{
        activate_terminal_control, clear_terminal_control, BatchAttemptScope,
    };
    activate_terminal_control(id, attempt);
    let _scope = BatchAttemptScope::enter_singleton(attempt);
    if commit {
        spawn_committer(id, attempt);
    }
    emit_marker(&format!("READY {id}"));
    let params = vl_params(id, prompt, image, max_tokens);
    let mut out = std::io::stdout();
    hipfire_generate::vision::generate_vl(m, gpu, &mut out, &params);
    std::io::stdout().flush().expect("flush turn");
    clear_terminal_control();
    emit_marker(&format!("SECTION {id}"));
}

#[cfg(test)]
fn child_vl_matrix() {
    let fixture =
        std::env::var("HIPFIRE_VISION_FIXTURE").expect("HIPFIRE_VISION_FIXTURE must be set");
    let (text_path, vision_path) =
        split_fixture_pair("HIPFIRE_VISION_FIXTURE", &fixture).expect("valid fixture pair");
    let image = repo_image(
        "benchmarks/vision/images/general_qa.jpg",
        "HIPFIRE_VISION_IMAGE",
    );
    assert!(
        std::path::Path::new(&image).exists(),
        "vision image missing: {image}"
    );
    spawn_abort_injector();
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    warm_first_alloc(&mut gpu);

    let mut m = admit_and_load(
        &mut gpu,
        &text_path,
        Some(&vision_path),
        8192,
        hipfire_runtime::loader_api::SpecLoadCfg::default(),
    );
    emit_marker("SECTION load-ok");
    let prompt = "Describe this image in detail.";
    // Warmup turn BEFORE the VRAM baseline: first-use kernel compiles land
    // here, so later per-turn/unload accounting measures owned state only.
    // Attempt 99 is unique to this throwaway turn (parent never aborts it).
    vl_turn(&mut m, &mut gpu, "g48-vl-warm", 99, &image, prompt, 8, true);
    let vram_base = vram_free_bytes(&gpu);

    // T1 baseline; T2 cancel-before-prefill; T3 injected prefill fault;
    // T4 reuse-after-fault.
    vl_turn(&mut m, &mut gpu, "g48-vl-t1", 1, &image, prompt, 8, true);
    vl_turn(&mut m, &mut gpu, "g48-vl-t2", 2, &image, prompt, 8, false);
    hipfire_generate::common::arm_vision_fault(Some("prefill"));
    vl_turn(&mut m, &mut gpu, "g48-vl-t3", 3, &image, prompt, 8, false);
    hipfire_generate::common::arm_vision_fault(None);
    vl_turn(&mut m, &mut gpu, "g48-vl-t4", 4, &image, prompt, 8, true);

    // Unload/reload: paired accounting, then fresh-load equality.
    hipfire_loader::unload_model(m, &mut gpu).expect("unload after matrix");
    gpu.drain_pool();
    let vram_after = vram_free_bytes(&gpu);
    assert!(
        vram_after + VRAM_SLACK_BYTES >= vram_base,
        "VL unload leaked VRAM: base {vram_base} after {vram_after}"
    );
    emit_marker("SECTION unload-ok");
    let mut m = admit_and_load(
        &mut gpu,
        &text_path,
        Some(&vision_path),
        8192,
        hipfire_runtime::loader_api::SpecLoadCfg::default(),
    );
    emit_marker("SECTION reload-ok");
    vl_turn(&mut m, &mut gpu, "g48-vl-t5", 5, &image, prompt, 8, true);
    // T6 cancel-during-decode (parent aborts on first token; wider budget
    // so decode is still in flight); T7 reuse.
    vl_turn(&mut m, &mut gpu, "g48-vl-t6", 6, &image, prompt, 32, false);
    vl_turn(&mut m, &mut gpu, "g48-vl-t7", 7, &image, prompt, 8, true);
    // T8 injected decode fault; T9 reuse.
    hipfire_generate::common::arm_vision_fault(Some("decode"));
    vl_turn(&mut m, &mut gpu, "g48-vl-t8", 8, &image, prompt, 8, false);
    hipfire_generate::common::arm_vision_fault(None);
    vl_turn(&mut m, &mut gpu, "g48-vl-t9", 9, &image, prompt, 8, true);
}

/// One dots.ocr image turn (AR or, when a speculator is loaded, n-gram).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn dots_turn(
    m: &mut hipfire_loader::LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    id: &str,
    attempt: u64,
    image: &str,
    prompt: &str,
    max_tokens: usize,
    commit: bool,
) {
    use hipfire_engine::terminal::{
        activate_terminal_control, clear_terminal_control, BatchAttemptScope,
    };
    activate_terminal_control(id, attempt);
    let _scope = BatchAttemptScope::enter_singleton(attempt);
    if commit {
        spawn_committer(id, attempt);
    }
    emit_marker(&format!("READY {id}"));
    let params = vl_params(id, prompt, image, max_tokens);
    let mut out = std::io::stdout();
    hipfire_generate::vision::generate_vl_dots_ocr(m, gpu, &mut out, &params);
    std::io::stdout().flush().expect("flush turn");
    clear_terminal_control();
    emit_marker(&format!("SECTION {id}"));
}

/// One dots.ocr text-only turn.
#[cfg(test)]
fn dots_text_turn(
    m: &mut hipfire_loader::LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    id: &str,
    attempt: u64,
    prompt: &str,
    max_tokens: usize,
    commit: bool,
) {
    use hipfire_engine::terminal::{
        activate_terminal_control, clear_terminal_control, BatchAttemptScope,
    };
    activate_terminal_control(id, attempt);
    let _scope = BatchAttemptScope::enter_singleton(attempt);
    if commit {
        spawn_committer(id, attempt);
    }
    emit_marker(&format!("READY {id}"));
    let mut out = std::io::stdout();
    hipfire_generate::vision::generate_dots_ocr_text(
        m, gpu, &mut out, id, prompt, None, 0.0, 1.0, max_tokens,
    );
    std::io::stdout().flush().expect("flush turn");
    clear_terminal_control();
    emit_marker(&format!("SECTION {id}"));
}

#[cfg(test)]
fn child_dots_matrix() {
    let path =
        std::env::var("HIPFIRE_DOTS_OCR_FIXTURE").expect("HIPFIRE_DOTS_OCR_FIXTURE must be set");
    let image = repo_image(
        "benchmarks/images/dots_ocr_smoke_001.jpg",
        "HIPFIRE_DOTS_IMAGE",
    );
    assert!(
        std::path::Path::new(&image).exists(),
        "dots image missing: {image}"
    );
    spawn_abort_injector();
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    warm_first_alloc(&mut gpu);

    // max_seq 8192: the smoke image yields 4880 visual tokens plus prompt,
    // so 2048 would trip the KV-budget guard before prefill.
    let mut m = admit_and_load(
        &mut gpu,
        &path,
        None,
        8192,
        hipfire_runtime::loader_api::SpecLoadCfg::default(),
    );
    emit_marker("SECTION load-ok");
    let prompt = "Transcribe all visible text in reading order.";
    // Warmup turn BEFORE the VRAM baseline: first-use kernel compiles land
    // here (rope/attention/gather kernels compile on first use), so later
    // per-turn/unload accounting measures owned state only.
    dots_turn(
        &mut m,
        &mut gpu,
        "g48-dots-warm",
        100,
        &image,
        prompt,
        16,
        true,
    );
    let vram_base = vram_free_bytes(&gpu);

    // AR: baseline, prefill cancel, prefill fault, reuse; argmax fault,
    // reuse; decode fault, reuse; decode cancel, reuse.
    dots_turn(
        &mut m,
        &mut gpu,
        "g48-dots-d1",
        101,
        &image,
        prompt,
        16,
        true,
    );
    dots_turn(
        &mut m,
        &mut gpu,
        "g48-dots-d2",
        102,
        &image,
        prompt,
        16,
        false,
    );
    hipfire_generate::common::arm_dots_fault(Some("prefill"));
    dots_turn(
        &mut m,
        &mut gpu,
        "g48-dots-d3",
        103,
        &image,
        prompt,
        16,
        false,
    );
    hipfire_generate::common::arm_dots_fault(None);
    dots_turn(
        &mut m,
        &mut gpu,
        "g48-dots-d4",
        104,
        &image,
        prompt,
        16,
        true,
    );
    hipfire_generate::common::arm_dots_fault(Some("argmax"));
    dots_turn(
        &mut m,
        &mut gpu,
        "g48-dots-d5",
        105,
        &image,
        prompt,
        16,
        false,
    );
    hipfire_generate::common::arm_dots_fault(None);
    dots_turn(
        &mut m,
        &mut gpu,
        "g48-dots-d6",
        106,
        &image,
        prompt,
        16,
        true,
    );
    hipfire_generate::common::arm_dots_fault(Some("decode"));
    dots_turn(
        &mut m,
        &mut gpu,
        "g48-dots-d7",
        107,
        &image,
        prompt,
        16,
        false,
    );
    hipfire_generate::common::arm_dots_fault(None);
    dots_turn(
        &mut m,
        &mut gpu,
        "g48-dots-d8",
        108,
        &image,
        prompt,
        16,
        true,
    );
    dots_turn(
        &mut m,
        &mut gpu,
        "g48-dots-d9",
        109,
        &image,
        prompt,
        32,
        false,
    );
    dots_turn(
        &mut m,
        &mut gpu,
        "g48-dots-d10",
        110,
        &image,
        prompt,
        16,
        true,
    );

    // Unload/reload with the n-gram drafter armed: spec success must be
    // byte-identical to AR; spec fault fails closed; reuse recovers.
    hipfire_loader::unload_model(m, &mut gpu).expect("unload dots AR model");
    gpu.drain_pool();
    let vram_after = vram_free_bytes(&gpu);
    assert!(
        vram_after + VRAM_SLACK_BYTES >= vram_base,
        "dots unload leaked VRAM: base {vram_base} after {vram_after}"
    );
    emit_marker("SECTION unload-ok");
    let mut m = admit_and_load(
        &mut gpu,
        &path,
        None,
        8192,
        hipfire_runtime::loader_api::SpecLoadCfg {
            ngram_draft: Some(true),
            ..Default::default()
        },
    );
    assert!(
        m.speculator.is_some(),
        "ngram speculator must load for the spec phase"
    );
    emit_marker("SECTION reload-spec-ok");
    // Warmup spec turn: first-use verify-path kernels/scratch land here so
    // the final unload accounting (against the phase-1 baseline) stays clean.
    dots_turn(
        &mut m,
        &mut gpu,
        "g48-dots-swarm",
        114,
        &image,
        prompt,
        16,
        true,
    );
    dots_turn(
        &mut m,
        &mut gpu,
        "g48-dots-s1",
        111,
        &image,
        prompt,
        16,
        true,
    );
    hipfire_generate::common::arm_dots_fault(Some("spec"));
    dots_turn(
        &mut m,
        &mut gpu,
        "g48-dots-s2",
        112,
        &image,
        prompt,
        16,
        false,
    );
    hipfire_generate::common::arm_dots_fault(None);
    dots_turn(
        &mut m,
        &mut gpu,
        "g48-dots-s3",
        113,
        &image,
        prompt,
        16,
        true,
    );

    // Text-only path on the same bundle: success, decode cancel, reuse.
    let text_prompt = "List three office supplies.";
    dots_text_turn(&mut m, &mut gpu, "g48-dots-x1", 201, text_prompt, 16, true);
    dots_text_turn(&mut m, &mut gpu, "g48-dots-x2", 202, text_prompt, 32, false);
    dots_text_turn(&mut m, &mut gpu, "g48-dots-x3", 203, text_prompt, 16, true);

    hipfire_loader::unload_model(m, &mut gpu).expect("final dots unload");
    gpu.drain_pool();
    let vram_final = vram_free_bytes(&gpu);
    assert!(
        vram_final + VRAM_SLACK_BYTES >= vram_base,
        "dots final unload leaked VRAM: base {vram_base} final {vram_final}"
    );
    emit_marker("SECTION unload2-ok");
}
