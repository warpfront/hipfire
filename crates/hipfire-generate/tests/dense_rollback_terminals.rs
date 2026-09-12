// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! G4.10 dense-route terminal contract tests (no GPU).
//!
//! Pins the observable wire contract the G4.10 dense-rollback work must
//! preserve or establish:
//!
//! - success epilogue: the exact envelope sequence a clean turn emits
//!   (`gen_start` → `token`* → `done`) is byte-stable before/after the
//!   rollback reroute — the reroute only touches failure/cancel exits;
//! - failure epilogue: one correlated `error`, never `done`, with the
//!   attested `rolled_back` flag (and reset context when unattested);
//! - cancel epilogue: attested keeps the `aborted`+`done(aborted)` pair,
//!   unattested collapses to a single error with no `done`;
//! - fault hooks: one-shot, default-off (no production behavior change).

#![allow(clippy::all)]

use hipfire_engine::terminal::*;
use hipfire_generate::ar::{emit_active_route_done_value, emit_generation_start, GenerationRoute, GenerationRouteScope};
use hipfire_generate::common::{
    arm_generation_fault_after_first_decode, arm_generation_fault_after_prefill,
    emit_fail_closed_error, emit_spec_cancel_after_rollback, take_generation_fault_after_first_decode,
    take_generation_fault_after_prefill, RollbackEpilogue,
};
static TERMINAL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    // Recover from poisoning so one failing test cannot cascade into the
    // rest (the failing test still fails on its own assertion).
    TERMINAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn fresh_attempt(attempt: u64) {
    batch_clear_all_terminals();
    clear_terminal_control();
    set_active_attempt_id(attempt);
    assert_eq!(hipfire_generate::ar::active_generation_route(), None);
}

/// Fresh attempt with staged terminal control, mirroring the daemon's
/// singleton admission: `done` terminals deliver without a client handshake.
fn fresh_controlled_attempt(id: &str, attempt: u64) {
    fresh_attempt(attempt);
    activate_terminal_control(id, attempt);
}

fn parse_lines(output: &[u8]) -> Vec<serde_json::Value> {
    std::str::from_utf8(output)
        .expect("UTF-8 events")
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).expect("JSON event"))
        .collect()
}

/// Scripted LFM2-style success: gen_start → one token → done. Mirrors the
/// envelope shapes `generate_lfm2moe` emits on its clean path, so any
/// success-path byte change breaks this test.
#[test]
fn lfm_success_envelope_sequence_byte_stable() {
    let _guard = lock();
    let id = "g410-success";
    fresh_controlled_attempt(id, 901);
    let mut output = Vec::new();
    {
        let _route = GenerationRouteScope::enter(GenerationRoute::LfmAr, id);
        emit_generation_start(GenerationRoute::LfmAr, &mut output, id, false);
        let token = serde_json::json!({
            "type": "token",
            "id": id,
            "text": " Paris",
            "attempt_id": active_attempt_id(),
        });
        output.extend_from_slice(token.to_string().as_bytes());
        output.push(b'\n');
        let pending_done = serde_json::json!({
            "type": "done",
            "id": id,
            "tokens": 1,
            "tok_s": 10.0,
            "prefill_ms": 5,
            "total_ms": 105,
            "attempt_id": active_attempt_id(),
        });
        emit_active_route_done_value(&mut output, &pending_done);
        assert_eq!(
            hipfire_generate::ar::active_generation_route(),
            None,
            "done must release the route"
        );
    }
    let events = parse_lines(&output);
    assert_eq!(events.len(), 3, "success emits exactly 3 envelopes: {events:?}");
    assert_eq!(events[0]["type"], "gen_start");
    assert_eq!(events[0]["id"], id);
    assert_eq!(events[0]["attempt_id"], 901);
    assert_eq!(events[1]["type"], "token");
    assert_eq!(events[1]["id"], id);
    assert_eq!(events[1]["text"], " Paris");
    assert_eq!(events[1]["attempt_id"], 901);
    assert_eq!(events[2]["type"], "done");
    assert_eq!(events[2]["id"], id);
    assert_eq!(events[2]["tokens"], 1);
    assert_eq!(events[2]["attempt_id"], 901);
    // Exactly one terminal was claimed: a second claim on the same key fails.
    assert!(
        !claim_wire_terminal(id, 901),
        "success turn must claim exactly one terminal"
    );
    fresh_attempt(0);
}

/// Fail-closed error with an attested epilogue: exactly one `error` with
/// `rolled_back=true`, never `done`, terminal claimed exactly once.
#[test]
fn dense_fail_closed_error_attested_is_single_terminal() {
    let _guard = lock();
    let id = "g410-fail-attested";
    fresh_controlled_attempt(id, 902);
    let mut output = Vec::new();
    {
        let _route = GenerationRouteScope::enter(GenerationRoute::LfmAr, id);
        emit_generation_start(GenerationRoute::LfmAr, &mut output, id, false);
        let ep = RollbackEpilogue { rolled_back: true, context: None };
        emit_fail_closed_error(&mut output, Some(id), "lfm2moe decode failed: test", "internal", false, &ep);
    }
    let events = parse_lines(&output);
    assert_eq!(events.len(), 2, "gen_start + one error, no done: {events:?}");
    assert_eq!(events[1]["type"], "error");
    assert_eq!(events[1]["id"], id);
    assert_eq!(events[1]["class"], "internal");
    assert_eq!(events[1]["retryable"], false);
    assert_eq!(events[1]["rolled_back"], true);
    assert_eq!(events[1]["attempt_id"], 902);
    assert!(
        !claim_wire_terminal(id, 902),
        "failure turn must claim exactly one terminal"
    );
    fresh_attempt(0);
}

/// Fail-closed error with an UNATTESTED epilogue: `rolled_back=false` with
/// the reset context visible in the message — the "reset error visible and
/// unattested" contract — still exactly one terminal, still no `done`.
#[test]
fn dense_fail_closed_error_unattested_is_visible_single_terminal() {
    let _guard = lock();
    let id = "g410-fail-unattested";
    fresh_controlled_attempt(id, 903);
    let mut output = Vec::new();
    {
        let _route = GenerationRouteScope::enter(GenerationRoute::MiniMaxAr, id);
        emit_generation_start(GenerationRoute::MiniMaxAr, &mut output, id, false);
        let ep = RollbackEpilogue {
            rolled_back: false,
            context: Some("device_synchronize failed: test-reset".to_string()),
        };
        emit_fail_closed_error(&mut output, Some(id), "minimax decode failed: test", "internal", false, &ep);
    }
    let events = parse_lines(&output);
    assert_eq!(events.len(), 2, "gen_start + one error, no done: {events:?}");
    assert_eq!(events[1]["type"], "error");
    assert_eq!(events[1]["rolled_back"], false);
    let message = events[1]["message"].as_str().expect("error message");
    assert!(
        message.contains("device_synchronize failed: test-reset"),
        "reset context must stay visible: {message}"
    );
    assert!(
        !claim_wire_terminal(id, 903),
        "unattested failure must claim exactly one terminal"
    );
    fresh_attempt(0);
}

/// Attested cancel keeps the fold-compatible `aborted`+`done(aborted)` pair.
#[test]
fn spec_cancel_attested_keeps_aborted_done_pair() {
    let _guard = lock();
    let id = "g410-cancel-attested";
    fresh_controlled_attempt(id, 904);
    let mut output = Vec::new();
    {
        let _route = GenerationRouteScope::enter(GenerationRoute::LfmAr, id);
        emit_generation_start(GenerationRoute::LfmAr, &mut output, id, false);
        let ep = RollbackEpilogue { rolled_back: true, context: None };
        emit_spec_cancel_after_rollback(&mut output, id, 3, &ep);
    }
    let events = parse_lines(&output);
    let types: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap_or("?")).collect();
    assert_eq!(
        types,
        vec!["gen_start", "aborted", "done"],
        "attested cancel keeps the pair: {events:?}"
    );
    assert_eq!(events[2]["finish_reason"], "aborted");
    assert!(
        !claim_wire_terminal(id, 904),
        "cancel turn must claim exactly one terminal"
    );
    fresh_attempt(0);
}

/// Unattested cancel collapses to a single error with no `done`.
#[test]
fn spec_cancel_unattested_is_single_error_no_done() {
    let _guard = lock();
    let id = "g410-cancel-unattested";
    fresh_controlled_attempt(id, 905);
    let mut output = Vec::new();
    {
        let _route = GenerationRouteScope::enter(GenerationRoute::CohereAr, id);
        emit_generation_start(GenerationRoute::CohereAr, &mut output, id, false);
        let ep = RollbackEpilogue {
            rolled_back: false,
            context: Some("lfm2moe.reset: test-reset".to_string()),
        };
        emit_spec_cancel_after_rollback(&mut output, id, 3, &ep);
    }
    let events = parse_lines(&output);
    assert_eq!(events.len(), 2, "gen_start + one error, never done: {events:?}");
    assert_eq!(events[1]["type"], "error");
    assert_eq!(events[1]["rolled_back"], false);
    assert!(
        !claim_wire_terminal(id, 905),
        "unattested cancel must claim exactly one terminal"
    );
    fresh_attempt(0);
}

/// Fault hooks are one-shot and default-off: production paths that only
/// `take_*` see no behavior change unless a test armed them.
#[test]
fn generation_fault_hooks_one_shot_default_off() {
    let _guard = lock();
    assert!(!take_generation_fault_after_prefill());
    assert!(!take_generation_fault_after_first_decode());
    arm_generation_fault_after_prefill(true);
    assert!(take_generation_fault_after_prefill());
    assert!(!take_generation_fault_after_prefill());
    arm_generation_fault_after_first_decode(true);
    assert!(take_generation_fault_after_first_decode());
    assert!(!take_generation_fault_after_first_decode());
    arm_generation_fault_after_prefill(false);
    arm_generation_fault_after_first_decode(false);
    assert!(!take_generation_fault_after_prefill());
    assert!(!take_generation_fault_after_first_decode());
}

/// Scripted LLaMA-style success: gen_start → one token → done. Mirrors the
/// envelope shapes the LLaMA AR loop emits on its clean path (G4.10(b)), so
/// any success-path byte change from the prefill/decode/cancel reroute
/// breaks this test.
#[test]
fn llama_success_envelope_sequence_byte_stable() {
    let _guard = lock();
    let id = "g410b-success";
    fresh_controlled_attempt(id, 911);
    let mut output = Vec::new();
    {
        let _route = GenerationRouteScope::enter(GenerationRoute::LlamaAr, id);
        emit_generation_start(GenerationRoute::LlamaAr, &mut output, id, false);
        let token = serde_json::json!({
            "type": "token",
            "id": id,
            "text": " Paris",
            "attempt_id": active_attempt_id(),
        });
        output.extend_from_slice(token.to_string().as_bytes());
        output.push(b'\n');
        let pending_done = serde_json::json!({
            "type": "done",
            "id": id,
            "tokens": 1,
            "tok_s": 10.0,
            "prefill_tokens": 7,
            "prefill_ms": 5.0,
            "prefill_tok_s": 1400.0,
            "decode_tok_s": 10.0,
            "ttft_ms": 5.0,
            "attempt_id": active_attempt_id(),
        });
        emit_active_route_done_value(&mut output, &pending_done);
        assert_eq!(
            hipfire_generate::ar::active_generation_route(),
            None,
            "done must release the route"
        );
    }
    let events = parse_lines(&output);
    assert_eq!(events.len(), 3, "success emits exactly 3 envelopes: {events:?}");
    assert_eq!(events[0]["type"], "gen_start");
    assert_eq!(events[0]["id"], id);
    assert_eq!(events[0]["attempt_id"], 911);
    assert_eq!(events[1]["type"], "token");
    assert_eq!(events[1]["id"], id);
    assert_eq!(events[1]["text"], " Paris");
    assert_eq!(events[1]["attempt_id"], 911);
    assert_eq!(events[2]["type"], "done");
    assert_eq!(events[2]["id"], id);
    assert_eq!(events[2]["tokens"], 1);
    assert_eq!(events[2]["prefill_tokens"], 7);
    assert_eq!(events[2]["attempt_id"], 911);
    // Exactly one terminal was claimed: a second claim on the same key fails.
    assert!(
        !claim_wire_terminal(id, 911),
        "success turn must claim exactly one terminal"
    );
    fresh_attempt(0);
}
