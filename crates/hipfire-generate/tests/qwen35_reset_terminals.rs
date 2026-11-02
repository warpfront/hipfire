// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! G4.7 Qwen3.5 reset/rollback wire characterization (no GPU).
//!
//! Pins the observable wire payloads of the Qwen35 single-AR (S1/S2/S4) and
//! pipeline-parallel (P2/P3) failure terminals BEFORE the G4.7 attested
//! reroute, so the reroute diff is reviewable line-by-line:
//!
//! - S1/S2 today: `write_error` (attempt emitter, route-aware when a route
//!   scope is active) with `rolled_back=false`, class `internal`.
//! - S4 today: `dense::emit_active_attempt_error` with class `transient`,
//!   retryable, `rolled_back=false`.
//! - P2 since E2: `emit_fail_closed_error_for_route(PipelineParallel, …)`
//!   with attested `rolled_back=true`.
//! - P3 abort today: attested `aborted`+`done(aborted)`; unattested collapses
//!   to one correlated error (shape pinned via the same emitters
//!   `emit_pipeline_cancel_after_rollback` uses).
//!
//! E2/E3 update the P2/S1/S2/S4 assertions to the attested shape
//! (`rolled_back=true`); the abort-shape and policy-gate tests never change.

#![allow(clippy::all)]

use hipfire_engine::terminal::*;
use hipfire_generate::ar::{
    emit_generation_cancel, emit_generation_start, qwen_ar_forward_fail_action,
    qwen_ar_forward_fail_message, write_error, GenerationRoute, GenerationRouteScope,
};
use hipfire_generate::common::{
    attest_rollback_steps, emit_fail_closed_error, emit_fail_closed_error_for_route,
    RollbackEpilogue,
};

static TERMINAL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    TERMINAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn fresh_attempt(attempt: u64) {
    // Route state is owned by `GenerationRouteScope` guards (each test enters
    // and drops its own); terminal latches live in `hipfire_engine`.
    clear_terminal_control();
    set_active_attempt_id(attempt);
    assert_eq!(hipfire_generate::ar::active_generation_route(), None);
}

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
fn teardown() {
    // `GenerationRouteScope` guards dropped by the caller restore the route.
    clear_terminal_control();
    set_active_attempt_id(0);
}

/// S1 prefill shape today: one `internal`/non-retryable error with
/// `rolled_back=false`, no `done`, no tokens.
#[test]
fn qwen_single_prefill_fail_error_shape_unattested() {
    let _guard = lock();
    let id = "g47-s1-prefill";
    fresh_controlled_attempt(id, 71_001);
    let mut sink = Vec::new();
    {
        let _scope = GenerationRouteScope::enter(GenerationRoute::QwenAr, id);
        emit_generation_start(GenerationRoute::QwenAr, &mut sink, id, false);
        write_error(
            &mut sink,
            id,
            &qwen_ar_forward_fail_message("forward_prefill_batch", "injected"),
        );
    }
    let events = parse_lines(&sink);
    assert_eq!(events[0]["type"], "gen_start");
    let errors: Vec<&serde_json::Value> = events
        .iter()
        .filter(|event| event["type"] == "error")
        .collect();
    assert_eq!(
        errors.len(),
        1,
        "S1 must emit exactly one error: {events:?}"
    );
    assert_eq!(errors[0]["id"], id);
    assert_eq!(
        errors[0]["message"], "forward_prefill_batch: injected",
        "S1 message shape: {events:?}"
    );
    assert_eq!(errors[0]["class"], "internal");
    assert_eq!(errors[0]["retryable"], false);
    assert_eq!(
        errors[0]["rolled_back"], false,
        "S1 is unattested today (E3 flips to true)"
    );
    assert!(
        events.iter().all(|event| event["type"] != "done"),
        "S1 must not emit done: {events:?}"
    );
    teardown();
}

/// S2 decode shape today: same contract as S1 with the decode phase label.
#[test]
fn qwen_single_decode_fail_error_shape_unattested() {
    let _guard = lock();
    let id = "g47-s2-decode";
    fresh_controlled_attempt(id, 71_002);
    let mut sink = Vec::new();
    {
        let _scope = GenerationRouteScope::enter(GenerationRoute::QwenAr, id);
        emit_generation_start(GenerationRoute::QwenAr, &mut sink, id, false);
        write_error(
            &mut sink,
            id,
            &qwen_ar_forward_fail_message("forward_scratch decode", "injected"),
        );
    }
    let events = parse_lines(&sink);
    let errors: Vec<&serde_json::Value> = events
        .iter()
        .filter(|event| event["type"] == "error")
        .collect();
    assert_eq!(
        errors.len(),
        1,
        "S2 must emit exactly one error: {events:?}"
    );
    assert_eq!(errors[0]["message"], "forward_scratch decode: injected");
    assert_eq!(errors[0]["class"], "internal");
    assert_eq!(errors[0]["retryable"], false);
    assert_eq!(
        errors[0]["rolled_back"], false,
        "S2 is unattested today (E3 flips to true)"
    );
    assert!(
        events.iter().all(|event| event["type"] != "done"),
        "S2 must not emit done: {events:?}"
    );
    teardown();
}

/// S4 adaptive-downshift shape today: `transient`/retryable error with
/// `rolled_back=false`.
#[test]
fn qwen_single_downshift_fail_error_shape_unattested() {
    let _guard = lock();
    let id = "g47-s4-downshift";
    fresh_controlled_attempt(id, 71_003);
    let mut sink = Vec::new();
    {
        let _scope = GenerationRouteScope::enter(GenerationRoute::QwenAr, id);
        emit_generation_start(GenerationRoute::QwenAr, &mut sink, id, false);
        hipfire_generate::dense::emit_active_attempt_error(
            &mut sink,
            Some(id),
            "adaptive KV transition failed during decode: injected",
            "transient",
            true,
            false,
        );
    }
    let events = parse_lines(&sink);
    let errors: Vec<&serde_json::Value> = events
        .iter()
        .filter(|event| event["type"] == "error")
        .collect();
    assert_eq!(
        errors.len(),
        1,
        "S4 must emit exactly one error: {events:?}"
    );
    assert_eq!(errors[0]["class"], "transient");
    assert_eq!(errors[0]["retryable"], true);
    assert_eq!(
        errors[0]["rolled_back"], false,
        "S4 is unattested today (E3 flips to true)"
    );
    assert!(
        events.iter().all(|event| event["type"] != "done"),
        "S4 must not emit done: {events:?}"
    );
    teardown();
}

/// P2 prefill shape (E2): attested PP rollback renders `rolled_back=true`
/// with the message verbatim (no context appended when attested).
#[test]
fn pp_prefill_fail_error_shape_attested() {
    let _guard = lock();
    let id = "g47-p2-prefill";
    fresh_controlled_attempt(id, 71_004);
    let mut sink = Vec::new();
    {
        let _scope = GenerationRouteScope::enter(GenerationRoute::PipelineParallel, id);
        emit_generation_start(GenerationRoute::PipelineParallel, &mut sink, id, false);
        let ep = RollbackEpilogue {
            rolled_back: true,
            context: None,
        };
        emit_fail_closed_error_for_route(
            GenerationRoute::PipelineParallel,
            &mut sink,
            Some(id),
            "forward_prefill_batch_multi: injected",
            "validation",
            false,
            &ep,
        );
    }
    let events = parse_lines(&sink);
    let errors: Vec<&serde_json::Value> = events
        .iter()
        .filter(|event| event["type"] == "error")
        .collect();
    assert_eq!(
        errors.len(),
        1,
        "P2 must emit exactly one error: {events:?}"
    );
    assert_eq!(
        errors[0]["message"],
        "forward_prefill_batch_multi: injected"
    );
    assert_eq!(errors[0]["class"], "validation");
    assert_eq!(errors[0]["retryable"], false);
    assert_eq!(errors[0]["rolled_back"], true, "P2 is attested since E2");
    assert!(
        events.iter().all(|event| event["type"] != "done"),
        "P2 must not emit done: {events:?}"
    );
    teardown();
}

/// P2 decode shape (E2): same attested contract with the decode label.
#[test]
fn pp_decode_fail_error_shape_attested() {
    let _guard = lock();
    let id = "g47-p2-decode";
    fresh_controlled_attempt(id, 71_005);
    let mut sink = Vec::new();
    {
        let _scope = GenerationRouteScope::enter(GenerationRoute::PipelineParallel, id);
        emit_generation_start(GenerationRoute::PipelineParallel, &mut sink, id, false);
        let ep = RollbackEpilogue {
            rolled_back: true,
            context: None,
        };
        emit_fail_closed_error_for_route(
            GenerationRoute::PipelineParallel,
            &mut sink,
            Some(id),
            "forward_scratch_multi decode: injected",
            "validation",
            false,
            &ep,
        );
    }
    let events = parse_lines(&sink);
    let errors: Vec<&serde_json::Value> = events
        .iter()
        .filter(|event| event["type"] == "error")
        .collect();
    assert_eq!(
        errors.len(),
        1,
        "P2 must emit exactly one error: {events:?}"
    );
    assert_eq!(
        errors[0]["message"],
        "forward_scratch_multi decode: injected"
    );
    assert_eq!(errors[0]["rolled_back"], true);
    assert!(
        events.iter().all(|event| event["type"] != "done"),
        "P2 must not emit done: {events:?}"
    );
    teardown();
}

/// P3 abort shape, attested branch: `aborted` + `done(aborted)`, never `error`.
/// E2 keeps this terminal identical.
#[test]
fn pp_abort_attested_cancel_shape_stable() {
    let _guard = lock();
    let id = "g47-p3-abort-ok";
    fresh_controlled_attempt(id, 71_006);
    let mut sink = Vec::new();
    {
        let _scope = GenerationRouteScope::enter(GenerationRoute::PipelineParallel, id);
        emit_generation_start(GenerationRoute::PipelineParallel, &mut sink, id, false);
        emit_generation_cancel(GenerationRoute::PipelineParallel, &mut sink, id, 2);
    }
    let events = parse_lines(&sink);
    assert!(
        events.iter().all(|event| event["type"] != "error"),
        "attested abort must not emit error: {events:?}"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event["type"] == "aborted")
            .count(),
        1
    );
    let done: Vec<&serde_json::Value> = events
        .iter()
        .filter(|event| event["type"] == "done")
        .collect();
    assert_eq!(done.len(), 1);
    assert_eq!(done[0]["finish_reason"], "aborted");
    teardown();
}

/// P3 abort shape, unattested branch: one correlated error carrying the reset
/// context, no `done`. E2 keeps this terminal identical.
#[test]
fn pp_abort_unattested_error_shape_stable() {
    let _guard = lock();
    let id = "g47-p3-abort-fail";
    fresh_controlled_attempt(id, 71_007);
    let mut sink = Vec::new();
    {
        let _scope = GenerationRouteScope::enter(GenerationRoute::PipelineParallel, id);
        emit_generation_start(GenerationRoute::PipelineParallel, &mut sink, id, false);
        let ep = RollbackEpilogue {
            rolled_back: false,
            context: Some("pp rank1 device_synchronize: injected".to_string()),
        };
        emit_fail_closed_error_for_route(
            GenerationRoute::PipelineParallel,
            &mut sink,
            Some(id),
            "client cancelled; fail-closed rollback could not be attested",
            "validation",
            false,
            &ep,
        );
    }
    let events = parse_lines(&sink);
    let errors: Vec<&serde_json::Value> = events
        .iter()
        .filter(|event| event["type"] == "error")
        .collect();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0]["rolled_back"], false);
    assert!(
        errors[0]["message"]
            .as_str()
            .unwrap()
            .contains("pp rank1 device_synchronize"),
        "unattested abort must carry reset context: {events:?}"
    );
    assert!(
        events.iter().all(|event| event["type"] != "done"),
        "unattested abort must not emit done: {events:?}"
    );
    teardown();
}

/// Fail-closed policy gate every Qwen AR forward-failure site consults:
/// errors are emitted, uncommitted state is cold-reset, the failed token is
/// never emitted, adaptive poison stays sticky. S1/S2/S4/S5 all branch on it.
#[test]
fn qwen_forward_fail_action_policy_stable() {
    let action = qwen_ar_forward_fail_action();
    assert!(action.emit_request_error);
    assert!(action.reset_uncommitted_state);
    assert!(!action.emit_failed_token);
    assert!(!action.clear_adaptive_poison);
}

/// Unattested epilogue rendering (reset-phase failure half of the HW
/// contract): `rolled_back=false` stays wire-visible and the reset context is
/// appended to the message. Reset-phase HIP failure is not forceable on
/// healthy hardware, so this unit pins the shape the ignored HW test cannot
/// reach; the HW test pins the attested (`true`) rendering.
#[test]
fn unattested_epilogue_error_carries_reset_context() {
    let _guard = lock();
    let id = "g47-unattested-ctx";
    fresh_controlled_attempt(id, 71_008);
    let mut sink = Vec::new();
    {
        let _scope = GenerationRouteScope::enter(GenerationRoute::QwenAr, id);
        emit_generation_start(GenerationRoute::QwenAr, &mut sink, id, false);
        let ep = RollbackEpilogue {
            rolled_back: false,
            context: Some("device_synchronize failed: injected".to_string()),
        };
        emit_fail_closed_error(
            &mut sink,
            Some(id),
            "forward_prefill_batch: injected",
            "internal",
            false,
            &ep,
        );
    }
    let events = parse_lines(&sink);
    let errors: Vec<&serde_json::Value> = events
        .iter()
        .filter(|event| event["type"] == "error")
        .collect();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0]["rolled_back"], false);
    let message = errors[0]["message"].as_str().unwrap();
    assert!(
        message.contains("device_synchronize failed: injected"),
        "unattested error must carry reset context: {events:?}"
    );
    assert_eq!(
        message
            .matches("device_synchronize failed: injected")
            .count(),
        1,
        "reset context must be appended exactly once: {events:?}"
    );
    teardown();
}

/// Attestation combiner PP case: one per-device sync failure vetoes the whole
/// epilogue and names the device.
#[test]
fn attest_pp_device_sync_failure_names_device() {
    let ep = attest_rollback_steps(
        &[(
            "pp rank1 device_synchronize",
            Err("719: launch failure".to_string()),
        )],
        Ok(()),
    );
    assert!(!ep.rolled_back);
    assert!(
        ep.context.unwrap().contains("pp rank1 device_synchronize"),
        "combiner must name the failing device"
    );
}

/// Attestation combiner PP case: every device attested → `rolled_back=true`
/// with no context.
#[test]
fn attest_pp_all_devices_ok_attests() {
    let ep = attest_rollback_steps(
        &[
            ("pp rank0 device_synchronize", Ok(())),
            ("pp rank1 device_synchronize", Ok(())),
        ],
        Ok(()),
    );
    assert!(ep.rolled_back);
    assert!(ep.context.is_none());
}

/// S1/S2 shape since E3: attested rollback renders `rolled_back=true` with
/// the message verbatim. Class/retryable unchanged (`internal`/false).
#[test]
fn qwen_single_prefill_fail_error_shape_attested() {
    let _guard = lock();
    let id = "g47-s1-attested";
    fresh_controlled_attempt(id, 71_009);
    let mut sink = Vec::new();
    {
        let _scope = GenerationRouteScope::enter(GenerationRoute::QwenAr, id);
        emit_generation_start(GenerationRoute::QwenAr, &mut sink, id, false);
        let ep = RollbackEpilogue {
            rolled_back: true,
            context: None,
        };
        emit_fail_closed_error(
            &mut sink,
            Some(id),
            "forward_prefill_batch: injected",
            "internal",
            false,
            &ep,
        );
    }
    let events = parse_lines(&sink);
    let errors: Vec<&serde_json::Value> = events
        .iter()
        .filter(|event| event["type"] == "error")
        .collect();
    assert_eq!(
        errors.len(),
        1,
        "S1 must emit exactly one error: {events:?}"
    );
    assert_eq!(errors[0]["message"], "forward_prefill_batch: injected");
    assert_eq!(errors[0]["class"], "internal");
    assert_eq!(errors[0]["retryable"], false);
    assert_eq!(errors[0]["rolled_back"], true);
    assert!(
        events.iter().all(|event| event["type"] != "done"),
        "S1 must not emit done: {events:?}"
    );
    teardown();
}

/// S4 shape since E3: attested rollback keeps `transient`/retryable and
/// flips only `rolled_back` to true.
#[test]
fn qwen_single_downshift_fail_error_shape_attested() {
    let _guard = lock();
    let id = "g47-s4-attested";
    fresh_controlled_attempt(id, 71_010);
    let mut sink = Vec::new();
    {
        let _scope = GenerationRouteScope::enter(GenerationRoute::QwenAr, id);
        emit_generation_start(GenerationRoute::QwenAr, &mut sink, id, false);
        let ep = RollbackEpilogue {
            rolled_back: true,
            context: None,
        };
        emit_fail_closed_error(
            &mut sink,
            Some(id),
            "adaptive KV transition failed during decode: injected",
            "transient",
            true,
            &ep,
        );
    }
    let events = parse_lines(&sink);
    let errors: Vec<&serde_json::Value> = events
        .iter()
        .filter(|event| event["type"] == "error")
        .collect();
    assert_eq!(
        errors.len(),
        1,
        "S4 must emit exactly one error: {events:?}"
    );
    assert_eq!(errors[0]["class"], "transient");
    assert_eq!(errors[0]["retryable"], true);
    assert_eq!(errors[0]["rolled_back"], true);
    assert!(
        events.iter().all(|event| event["type"] != "done"),
        "S4 must not emit done: {events:?}"
    );
    teardown();
}

/// Direct single-AR fail-closed error-emitter wire shape: `gpu`/retryable
/// error with attested `rolled_back=true`. This synthetic test does not
/// replace the ignored hardware production seam.
#[test]
fn qwen_single_fail_closed_writer_wire_shape() {
    let _guard = lock();
    let id = "g47-seam-decode";
    fresh_controlled_attempt(id, 71_011);
    let mut sink = Vec::new();
    {
        let _scope = GenerationRouteScope::enter(GenerationRoute::QwenAr, id);
        emit_generation_start(GenerationRoute::QwenAr, &mut sink, id, false);
        let ep = RollbackEpilogue {
            rolled_back: true,
            context: None,
        };
        emit_fail_closed_error(
            &mut sink,
            Some(id),
            "injected fault after first decode",
            "gpu",
            true,
            &ep,
        );
    }
    let events = parse_lines(&sink);
    let errors: Vec<&serde_json::Value> = events
        .iter()
        .filter(|event| event["type"] == "error")
        .collect();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0]["message"], "injected fault after first decode");
    assert_eq!(errors[0]["class"], "gpu");
    assert_eq!(errors[0]["retryable"], true);
    assert_eq!(errors[0]["rolled_back"], true);
    assert_eq!(
        events.len(),
        2,
        "single fail-closed writer must emit only gen_start and error: {events:?}"
    );
    assert_eq!(events[0]["type"], "gen_start");
    assert_eq!(events[1]["type"], "error");
    teardown();
}

/// Direct pipeline-parallel fail-closed error-emitter wire shape with
/// attested `rolled_back=true`. This synthetic test does not replace the
/// ignored hardware production seam.
#[test]
fn pp_fail_closed_writer_wire_shape() {
    let _guard = lock();
    let id = "g47-pp-seam-decode";
    fresh_controlled_attempt(id, 71_012);
    let mut sink = Vec::new();
    {
        let _scope = GenerationRouteScope::enter(GenerationRoute::PipelineParallel, id);
        emit_generation_start(GenerationRoute::PipelineParallel, &mut sink, id, false);
        let ep = RollbackEpilogue {
            rolled_back: true,
            context: None,
        };
        emit_fail_closed_error_for_route(
            GenerationRoute::PipelineParallel,
            &mut sink,
            Some(id),
            "injected fault after first decode",
            "gpu",
            true,
            &ep,
        );
    }
    let events = parse_lines(&sink);
    let errors: Vec<&serde_json::Value> = events
        .iter()
        .filter(|event| event["type"] == "error")
        .collect();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0]["message"], "injected fault after first decode");
    assert_eq!(errors[0]["rolled_back"], true);
    assert_eq!(errors[0]["id"], id);
    assert_eq!(
        events.len(),
        2,
        "PP fail-closed writer must emit only gen_start and error: {events:?}"
    );
    assert_eq!(events[0]["type"], "gen_start");
    assert_eq!(events[1]["type"], "error");
    teardown();
}

/// Sealed Single-MoE expert mutation failure is one attested error terminal.
/// The routed expert has already written its private result, but publication
/// into the residual is faulted before any token or commit visibility.
#[test]
fn qwen_single_sealed_expert_mutation_fail_error_shape_attested() {
    let _guard = lock();
    let id = "g5-sealed-expert-mutation";
    fresh_controlled_attempt(id, 75_001);
    let mut sink = Vec::new();
    {
        let _scope = GenerationRouteScope::enter(GenerationRoute::QwenAr, id);
        emit_generation_start(GenerationRoute::QwenAr, &mut sink, id, false);
        let ep = RollbackEpilogue {
            rolled_back: true,
            context: None,
        };
        emit_fail_closed_error(
            &mut sink,
            Some(id),
            "sealed MoE expert mutation before combine publication: injected",
            "gpu",
            true,
            &ep,
        );
    }
    let events = parse_lines(&sink);
    let errors: Vec<&serde_json::Value> = events
        .iter()
        .filter(|event| event["type"] == "error")
        .collect();
    assert_eq!(
        errors.len(),
        1,
        "sealed expert fault must emit one error: {events:?}"
    );
    assert_eq!(errors[0]["rolled_back"], true);
    assert_eq!(errors[0]["class"], "gpu");
    assert_eq!(errors[0]["retryable"], true);
    assert!(
        errors[0]["message"]
            .as_str()
            .unwrap()
            .contains("sealed MoE expert mutation before combine publication"),
        "fault message must identify the sealed boundary: {events:?}"
    );
    assert!(
        events.iter().all(|event| !matches!(
            event["type"].as_str(),
            Some("token" | "committed" | "done" | "aborted")
        )),
        "sealed expert fault must publish no post-error events: {events:?}"
    );
    assert_eq!(events[0]["type"], "gen_start");
    assert_eq!(events[1]["type"], "error");
    teardown();
}
