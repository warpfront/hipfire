// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Extracted from `crates/hipfire-runtime/examples/daemon.rs`
//! `#[cfg(test)] mod terminal_control_tests` (9 of 10 tests).
//! The `glimmer_marker_suffix_is_char_boundary_safe` test remains in
//! `daemon.rs` because it exercises `super::glimmer_longest_marker_suffix`,
//! a Glimmer-specific helper that is not part of `hipfire-engine`.

use hipfire_engine::emit::emit_active_attempt_error;
use hipfire_engine::terminal::{
    activate_terminal_control, active_batch_generation, adopt_singleton_transfer,
    apply_terminal_control, await_client_terminal_commit, batch_bind_active,
    batch_clear_all_terminals, batch_clear_terminal, batch_clear_terminal_at_generation,
    batch_handoff_to_singleton_and_clear, batch_mark_ready_with_pending, batch_terminal_control,
    batch_terminal_generation, check_abort, claim_terminal, claim_terminal_at_generation,
    claim_wire_terminal, clear_terminal_control, emit_aborted_terminal_after_abort,
    emit_staged_terminal_done, mark_terminal_control_ready, set_active_attempt_id,
    terminal_control, terminal_generation, wait_terminal_control_decision, BatchAttemptScope,
    ClientTerminalDecision, LaneTicket, TerminalControlDecision,
};
use std::sync::{Arc, Barrier, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

/// Serializes all tests in this module: they share the process-global
/// terminal-control singleton and would race under `cargo test` parallelism.
fn test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}

/// Acquire the module lock and reset shared state. Hold the returned
/// guard for the full test body (including any helper threads joined
/// before drop).
fn begin_test() -> MutexGuard<'static, ()> {
    let guard = test_lock();
    clear_terminal_control();
    set_active_attempt_id(0);
    guard
}

fn reset() {
    clear_terminal_control();
    set_active_attempt_id(0);
}

fn batch_announce_terminal(id: &str, attempt_id: u64) -> bool {
    hipfire_engine::terminal::batch_announce_terminal(id, attempt_id).is_some()
}

fn batch_transition_to_queued(id: &str, attempt_id: u64) -> bool {
    batch_terminal_generation(id, attempt_id).is_some_and(|generation| {
        hipfire_engine::terminal::batch_transition_to_queued(id, attempt_id, generation)
    })
}
fn decision_of(id: &str, attempt_id: u64) -> Option<TerminalControlDecision> {
    let g = terminal_control().mu.lock().unwrap();
    g.active.as_ref().and_then(|a| {
        if a.id == id && a.attempt_id == attempt_id {
            a.decision
        } else {
            None
        }
    })
}

fn is_ready(id: &str, attempt_id: u64) -> bool {
    let g = terminal_control().mu.lock().unwrap();
    match g.active.as_ref() {
        Some(a) if a.id == id && a.attempt_id == attempt_id => a.ready,
        _ => false,
    }
}

#[test]
fn early_commit_before_ready_cannot_commit() {
    let _lock = begin_test();
    activate_terminal_control("r1", 7);
    apply_terminal_control("commit", "r1", 7);
    assert_eq!(
        decision_of("r1", 7),
        None,
        "commit before ready must be ignored"
    );
    assert!(!check_abort("r1"));
    // After ready, a fresh matching commit should still be required.
    assert!(mark_terminal_control_ready("r1", 7));
    assert_eq!(decision_of("r1", 7), None);
    apply_terminal_control("commit", "r1", 7);
    assert_eq!(decision_of("r1", 7), Some(TerminalControlDecision::Commit));
    reset();
}

#[test]
fn exact_id_and_attempt_correlation() {
    let _lock = begin_test();
    activate_terminal_control("r1", 3);
    // Wrong id
    apply_terminal_control("abort", "r2", 3);
    assert_eq!(decision_of("r1", 3), None);
    assert!(!check_abort("r1"));
    // Wrong attempt
    apply_terminal_control("abort", "r1", 99);
    assert_eq!(decision_of("r1", 3), None);
    assert!(!check_abort("r1"));
    // Exact match
    apply_terminal_control("abort", "r1", 3);
    assert_eq!(decision_of("r1", 3), Some(TerminalControlDecision::Abort));
    assert!(check_abort("r1"));
    // check_abort only matches active id
    assert!(!check_abort("r2"));
    reset();
}

#[test]
fn stale_and_malformed_controls_ignored() {
    let _lock = begin_test();
    activate_terminal_control("live", 5);
    // No active match: stale id/attempt
    apply_terminal_control("abort", "stale", 5);
    apply_terminal_control("commit", "live", 1);
    apply_terminal_control("abort", "live", 1);
    // Unknown kind
    apply_terminal_control("nope", "live", 5);
    assert_eq!(decision_of("live", 5), None);
    assert!(!is_ready("live", 5));
    assert!(!check_abort("live"));
    // Empty active: control without activation is a no-op
    clear_terminal_control();
    apply_terminal_control("abort", "live", 5);
    assert!(terminal_control().mu.lock().unwrap().active.is_none());
    reset();
}

#[test]
fn abort_before_ready_wins() {
    let _lock = begin_test();
    activate_terminal_control("r1", 2);
    apply_terminal_control("abort", "r1", 2);
    assert!(check_abort("r1"));
    // Ready after abort does not clear abort; commit cannot overwrite.
    assert!(mark_terminal_control_ready("r1", 2));
    apply_terminal_control("commit", "r1", 2);
    assert_eq!(decision_of("r1", 2), Some(TerminalControlDecision::Abort));
    assert_eq!(
        wait_terminal_control_decision("r1", 2, Duration::from_millis(50)),
        ClientTerminalDecision::Abort
    );
    reset();
}

#[test]
fn abort_after_ready_wins() {
    let _lock = begin_test();
    activate_terminal_control("r1", 4);
    assert!(mark_terminal_control_ready("r1", 4));
    apply_terminal_control("abort", "r1", 4);
    assert_eq!(decision_of("r1", 4), Some(TerminalControlDecision::Abort));
    // Subsequent commit ignored once decided
    apply_terminal_control("commit", "r1", 4);
    assert_eq!(decision_of("r1", 4), Some(TerminalControlDecision::Abort));
    assert!(check_abort("r1"));
    reset();
}

#[test]
fn matching_ready_commit_succeeds() {
    let _lock = begin_test();
    activate_terminal_control("r1", 8);
    set_active_attempt_id(8);
    let mut sink = Vec::new();
    let pending = serde_json::json!({
        "type": "done",
        "id": "r1",
        "attempt_id": 8,
        "finish_reason": "stop",
        "tokens": 3,
    });
    let handle = std::thread::spawn(|| {
        // Spin until ready, then commit.
        for _ in 0..200 {
            if is_ready("r1", 8) {
                apply_terminal_control("commit", "r1", 8);
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("never became ready");
    });
    let decision = await_client_terminal_commit(&mut sink, "r1", &pending);
    handle.join().unwrap();
    assert_eq!(decision, ClientTerminalDecision::Commit);
    let line = std::str::from_utf8(&sink).unwrap().trim();
    let v: serde_json::Value = serde_json::from_str(line).unwrap();
    assert_eq!(v["type"], "commit_ready");
    assert_eq!(v["id"], "r1");
    assert_eq!(v["attempt_id"], 8);
    assert_eq!(v["finish_reason"], "stop");
    assert_eq!(v["tokens"], 3);
    // commit_ready is pending_done with only type changed.
    let mut as_done = v.clone();
    as_done["type"] = serde_json::json!("done");
    assert_eq!(as_done, pending);
    reset();
}

#[test]
fn timeout_classifies_abort() {
    let _lock = begin_test();
    activate_terminal_control("r1", 11);
    assert!(mark_terminal_control_ready("r1", 11));
    let decision = wait_terminal_control_decision("r1", 11, Duration::from_millis(30));
    assert_eq!(decision, ClientTerminalDecision::Abort);
    assert_eq!(decision_of("r1", 11), Some(TerminalControlDecision::Abort));
    assert!(check_abort("r1"));
    reset();
}

#[test]
fn commit_ready_json_carries_full_pending_done() {
    let _lock = begin_test();
    activate_terminal_control("req-x", 42);
    set_active_attempt_id(42);
    // Pre-latch abort so await returns immediately after emit.
    apply_terminal_control("abort", "req-x", 42);
    let pending = serde_json::json!({
        "type": "done",
        "id": "req-x",
        "attempt_id": 42,
        "finish_reason": "length",
        "tokens": 7,
        "tok_s": 1.5,
    });
    let mut sink = Vec::new();
    let decision = await_client_terminal_commit(&mut sink, "req-x", &pending);
    assert_eq!(decision, ClientTerminalDecision::Abort);
    let line = std::str::from_utf8(&sink).unwrap().trim();
    let v: serde_json::Value = serde_json::from_str(line).unwrap();
    assert_eq!(v["type"], "commit_ready");
    assert_eq!(v["id"], "req-x");
    assert_eq!(v["attempt_id"], 42);
    assert_eq!(v["finish_reason"], "length");
    assert_eq!(v["tokens"], 7);
    assert_eq!(v["tok_s"], 1.5);
    // Only one line
    assert_eq!(sink.iter().filter(|&&b| b == b'\n').count(), 1);
    reset();
}

#[test]
fn check_abort_matches_active_attempt_only() {
    let _lock = begin_test();
    activate_terminal_control("same", 1);
    apply_terminal_control("abort", "same", 1);
    assert!(check_abort("same"));
    // New activation clears prior abort latch.
    activate_terminal_control("same", 2);
    assert!(!check_abort("same"));
    apply_terminal_control("abort", "same", 2);
    assert!(check_abort("same"));
    reset();
}

#[test]
fn terminal_claim_is_exactly_once_under_race() {
    let _lock = begin_test();
    activate_terminal_control("race", 77);
    let barrier = Arc::new(Barrier::new(3));
    let mut joins = Vec::new();
    for _ in 0..2 {
        let barrier = Arc::clone(&barrier);
        joins.push(std::thread::spawn(move || {
            barrier.wait();
            claim_terminal("race", 77)
        }));
    }
    barrier.wait();
    let claimed = joins
        .into_iter()
        .map(|join| join.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(claimed.iter().filter(|&&value| value).count(), 1);
    assert_eq!(claimed.iter().filter(|&&value| !value).count(), 1);
    reset();
}

#[test]
fn terminal_claim_rejects_mismatch_and_late_writers() {
    let _lock = begin_test();
    activate_terminal_control("active", 91);
    assert!(!claim_terminal("active", 92));
    assert!(!claim_terminal("other", 91));
    assert!(!claim_terminal("active", 0));
    assert!(claim_terminal("active", 91));
    clear_terminal_control();
    assert!(!claim_terminal("active", 91));
    set_active_attempt_id(91);
    let mut late = Vec::new();
    emit_active_attempt_error(&mut late, Some("active"), "late", "runtime", false, true);
    assert!(late.is_empty(), "late writer must not reach the wire");
    reset();
}

#[test]
fn singleton_generation_reuse_rejects_old_writer() {
    let _lock = begin_test();
    activate_terminal_control("reuse", 7);
    let first = terminal_generation("reuse", 7).expect("first generation");
    clear_terminal_control();
    assert!(!claim_terminal("reuse", 7));

    activate_terminal_control("reuse", 7);
    let second = terminal_generation("reuse", 7).expect("second generation");
    assert_ne!(first, second);
    assert!(!claim_terminal_at_generation("reuse", 7, first));
    assert!(claim_terminal_at_generation("reuse", 7, second));
    reset();
}

#[test]
fn singleton_delayed_writer_stays_inert_across_multiple_lifecycles() {
    let _lock = begin_test();
    assert!(!claim_terminal("inactive", 99));

    activate_terminal_control("delayed", 17);
    let first = terminal_generation("delayed", 17).expect("first generation");
    clear_terminal_control();

    activate_terminal_control("other", 18);
    let second = terminal_generation("other", 18).expect("second generation");
    assert_ne!(first, second);
    assert!(!claim_terminal("delayed", 17));
    assert!(!claim_terminal_at_generation("delayed", 17, first));
    clear_terminal_control();

    activate_terminal_control("delayed", 17);
    let third = terminal_generation("delayed", 17).expect("third generation");
    assert_ne!(first, third);
    assert!(!claim_terminal_at_generation("delayed", 17, first));
    assert!(claim_terminal_at_generation("delayed", 17, third));
    reset();
}

#[test]
fn active_done_error_race_has_one_wire_terminal() {
    let _lock = begin_test();
    activate_terminal_control("semantic-race", 88);
    let pending = serde_json::json!({
        "type": "done",
        "id": "semantic-race",
        "attempt_id": 88,
        "finish_reason": "stop",
    });
    let done = std::thread::spawn({
        let pending = pending.clone();
        move || {
            set_active_attempt_id(88);
            let mut sink = Vec::new();
            emit_staged_terminal_done(&mut sink, &pending);
            sink
        }
    });
    let error = std::thread::spawn(|| {
        set_active_attempt_id(88);
        let mut sink = Vec::new();
        emit_active_attempt_error(
            &mut sink,
            Some("semantic-race"),
            "racing failure",
            "runtime",
            false,
            true,
        );
        sink
    });
    let done = done.join().unwrap();
    let error = error.join().unwrap();
    let lines = done
        .iter()
        .chain(error.iter())
        .filter(|&&byte| byte == b'\n')
        .count();
    assert_eq!(lines, 1);
    reset();
}

#[test]
fn batch_wire_claims_are_keyed_and_reusable_after_retirement() {
    let _lock = begin_test();
    batch_clear_all_terminals();
    assert!(batch_announce_terminal("lane-a", 1));
    assert!(batch_announce_terminal("lane-b", 1));
    assert!(batch_transition_to_queued("lane-a", 1));
    assert!(batch_transition_to_queued("lane-b", 1));
    let generation_a = batch_terminal_generation("lane-a", 1).expect("lane-a admission");
    let generation_b = batch_terminal_generation("lane-b", 1).expect("lane-b admission");
    assert!(batch_bind_active(
        "lane-a",
        1,
        generation_a,
        LaneTicket {
            lane: 0,
            generation: 1,
            admission: generation_a,
        }
    ));
    assert!(batch_bind_active(
        "lane-b",
        1,
        generation_b,
        LaneTicket {
            lane: 1,
            generation: 1,
            admission: generation_b,
        }
    ));
    {
        let _scope = BatchAttemptScope::enter_for("lane-a", 1);
        assert!(claim_wire_terminal("lane-a", 1));
        assert!(!claim_wire_terminal("lane-a", 1));
        assert!(!claim_wire_terminal("lane-a", 2));
    }
    {
        let _scope = BatchAttemptScope::enter_for("lane-b", 1);
        assert!(claim_wire_terminal("lane-b", 1));
    }
    let stale = BatchAttemptScope::enter_for("lane-a", 1);
    batch_clear_terminal("lane-a", 1);
    assert!(!claim_wire_terminal("lane-a", 1));
    drop(stale);
    assert!(batch_announce_terminal("lane-a", 1));
    {
        let _scope = BatchAttemptScope::enter_for("lane-a", 1);
        assert!(claim_wire_terminal("lane-a", 1));
    }
    batch_clear_all_terminals();
    reset();
}

#[test]
fn batch_generation_reuse_rejects_stale_scope_without_retirement_growth() {
    let _lock = begin_test();
    batch_clear_all_terminals();
    assert!(batch_announce_terminal("reuse", 41));
    let stale = BatchAttemptScope::enter_for("reuse", 41);
    batch_clear_terminal("reuse", 41);
    assert!(batch_announce_terminal("reuse", 41));
    assert!(!claim_wire_terminal("reuse", 41));
    drop(stale);
    {
        let _fresh = BatchAttemptScope::enter_for("reuse", 41);
        assert!(claim_wire_terminal("reuse", 41));
    }
    batch_clear_terminal("reuse", 41);

    for attempt in 1..=256 {
        let id = format!("churn-{attempt}");
        assert!(batch_announce_terminal(&id, attempt + 1000));
        let _scope = BatchAttemptScope::enter_for(&id, attempt + 1000);
        assert!(claim_wire_terminal(&id, attempt + 1000));
        batch_clear_terminal(&id, attempt + 1000);
    }
    let state = batch_terminal_control().mu.lock().unwrap();
    assert!(state.entries.is_empty());
    drop(state);
    batch_clear_all_terminals();
    reset();
}

#[test]
fn batch_conditional_cleanup_preserves_reused_key() {
    let _lock = begin_test();
    let id = "batch-reuse";
    let attempt_id = 41;
    assert!(batch_announce_terminal(id, attempt_id));
    let scope_a = BatchAttemptScope::enter_for(id, attempt_id);
    let generation_a = scope_a.admission_generation().expect("generation A");
    assert!(batch_clear_terminal_at_generation(
        id,
        attempt_id,
        generation_a
    ));

    assert!(batch_announce_terminal(id, attempt_id));
    let generation_b = batch_terminal_generation(id, attempt_id).expect("generation B");
    assert_ne!(generation_a, generation_b);
    assert!(!batch_clear_terminal_at_generation(
        id,
        attempt_id,
        generation_a
    ));
    assert_eq!(
        batch_terminal_generation(id, attempt_id),
        Some(generation_b)
    );
    drop(scope_a);

    let scope_b = BatchAttemptScope::enter_for(id, attempt_id);
    assert_eq!(scope_b.admission_generation(), Some(generation_b));
    assert!(batch_clear_terminal_at_generation(
        id,
        attempt_id,
        generation_b
    ));
    drop(scope_b);
    reset();
}

#[test]
fn batch_rebind_clears_tls_without_binding_reused_generation() {
    let _lock = begin_test();
    let id = "batch-rebind";
    let attempt_id = 42;
    assert!(batch_announce_terminal(id, attempt_id));
    let mut scope_a = BatchAttemptScope::enter_for(id, attempt_id);
    let generation_a = scope_a.admission_generation().expect("generation A");
    assert!(batch_clear_terminal_at_generation(
        id,
        attempt_id,
        generation_a
    ));

    assert!(batch_announce_terminal(id, attempt_id));
    let generation_b = batch_terminal_generation(id, attempt_id).expect("generation B");
    scope_a.rebind_for(attempt_id);
    assert_eq!(active_batch_generation(), None);
    assert!(!batch_clear_terminal_at_generation(
        id,
        attempt_id,
        generation_a
    ));
    assert_eq!(
        batch_terminal_generation(id, attempt_id),
        Some(generation_b)
    );

    let scope_b = BatchAttemptScope::enter_for(id, attempt_id);
    assert_eq!(scope_b.admission_generation(), Some(generation_b));
    assert!(batch_clear_terminal_at_generation(
        id,
        attempt_id,
        generation_b
    ));
    drop(scope_b);
    drop(scope_a);
    reset();
}

#[test]
fn singleton_handoff_abort_is_atomic_at_each_transfer_phase() {
    let _lock = begin_test();
    let id = "handoff-abort-phase";
    let attempt_id = 88_u64;

    for phase in 0..3 {
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);
        let generation = hipfire_engine::terminal::batch_announce_terminal(id, attempt_id)
            .expect("batch generation");
        activate_terminal_control(id, attempt_id);

        let abort_at_phase = || {
            let gate = Arc::new(Barrier::new(2));
            let worker_gate = Arc::clone(&gate);
            let worker = std::thread::spawn(move || {
                worker_gate.wait();
                apply_terminal_control("abort", id, attempt_id);
                hipfire_engine::terminal::batch_apply_terminal_control("abort", id, attempt_id);
            });
            gate.wait();
            worker.join().expect("abort worker");
        };

        let transfer = if phase == 0 {
            // Abort before the handoff snapshots either owner.
            abort_at_phase();
            batch_handoff_to_singleton_and_clear(id, attempt_id, generation)
                .expect("singleton handoff")
        } else {
            let transfer = batch_handoff_to_singleton_and_clear(id, attempt_id, generation)
                .expect("singleton handoff");
            if phase == 1 {
                // Abort after the exact snapshot/tombstone boundary but before
                // adoption; the tombstone must retain it.
                abort_at_phase();
                assert!(check_abort(id));
            } else {
                // Abort after adoption must hit the restored singleton directly.
                assert!(adopt_singleton_transfer(id, attempt_id, transfer.clone()));
                abort_at_phase();
                assert!(check_abort(id));
            }
            transfer
        };

        if phase != 2 {
            assert!(adopt_singleton_transfer(id, attempt_id, transfer));
        }
        assert!(check_abort(id), "abort must survive phase {phase}");

        let _scope = BatchAttemptScope::enter_singleton(attempt_id);
        assert!(claim_wire_terminal(id, attempt_id));
        assert!(
            !claim_wire_terminal(id, attempt_id),
            "singleton terminal must be claimed exactly once"
        );
        drop(_scope);
        clear_terminal_control();
        let next_generation = hipfire_engine::terminal::batch_announce_terminal(id, attempt_id)
            .expect("next generation after old terminal");
        assert_ne!(generation, next_generation);
        assert!(batch_clear_terminal_at_generation(
            id,
            attempt_id,
            next_generation
        ));
    }
    batch_clear_all_terminals();
    clear_terminal_control();
    set_active_attempt_id(0);
}

#[test]
fn singleton_handoff_tombstone_serializes_abort_and_same_key_admission() {
    let _lock = begin_test();
    let id = "handoff-admission-barrier";
    let attempt_id = 89_u64;
    let generation = hipfire_engine::terminal::batch_announce_terminal(id, attempt_id)
        .expect("batch generation");
    activate_terminal_control(id, attempt_id);
    let transfer = batch_handoff_to_singleton_and_clear(id, attempt_id, generation)
        .expect("singleton handoff");

    let gate = Arc::new(Barrier::new(3));
    let abort_gate = Arc::clone(&gate);
    let abort_worker = std::thread::spawn(move || {
        abort_gate.wait();
        apply_terminal_control("abort", id, attempt_id);
        hipfire_engine::terminal::batch_apply_terminal_control("abort", id, attempt_id);
    });
    let announce_gate = Arc::clone(&gate);
    let (announce_tx, announce_rx) = std::sync::mpsc::channel();
    let announce_worker = std::thread::spawn(move || {
        announce_gate.wait();
        announce_tx
            .send(hipfire_engine::terminal::batch_announce_terminal(
                id, attempt_id,
            ))
            .expect("announce result");
    });
    gate.wait();
    abort_worker.join().expect("abort worker");
    announce_worker.join().expect("announce worker");
    assert!(
        announce_rx.recv().expect("announce result").is_none(),
        "same-key B cannot enter while A transfer tombstone is live"
    );

    assert!(adopt_singleton_transfer(id, attempt_id, transfer));
    assert!(check_abort(id), "abort during transfer must be adopted");
    assert!(
        hipfire_engine::terminal::batch_announce_terminal(id, attempt_id).is_none(),
        "same-key B remains blocked until A closes"
    );
    let _scope = BatchAttemptScope::enter_singleton(attempt_id);
    assert!(claim_wire_terminal(id, attempt_id));
    assert!(!claim_wire_terminal(id, attempt_id));
    drop(_scope);
    clear_terminal_control();

    let generation_b = hipfire_engine::terminal::batch_announce_terminal(id, attempt_id)
        .expect("B admitted after A release");
    assert_ne!(generation, generation_b);
    assert!(batch_clear_terminal_at_generation(
        id,
        attempt_id,
        generation_b
    ));
    batch_clear_all_terminals();
    set_active_attempt_id(0);
}

#[test]
fn singleton_handoff_scope_cannot_bind_reannounced_batch_owner() {
    let _lock = begin_test();
    batch_clear_all_terminals();
    let id = "singleton-handoff-reuse";
    let attempt_id = 77;
    activate_terminal_control(id, attempt_id);
    let singleton_generation = terminal_generation(id, attempt_id).expect("singleton generation A");
    let generation_a = hipfire_engine::terminal::batch_announce_terminal(id, attempt_id)
        .expect("batch generation A");
    let transfer =
        batch_handoff_to_singleton_and_clear(id, attempt_id, generation_a).expect("handoff A");

    // A fresh batch admission for the same wire key is blocked while A's
    // tombstone is pending and remains blocked until A is adopted and closed.
    assert!(adopt_singleton_transfer(id, attempt_id, transfer));
    assert_eq!(
        terminal_generation(id, attempt_id),
        Some(singleton_generation)
    );
    assert!(
        hipfire_engine::terminal::batch_announce_terminal(id, attempt_id).is_none(),
        "batch generation B must wait for singleton A to close"
    );
    clear_terminal_control();
    let generation_b = hipfire_engine::terminal::batch_announce_terminal(id, attempt_id)
        .expect("batch generation B after A terminal cleanup");
    assert_ne!(generation_a, generation_b);
    let scope_a = BatchAttemptScope::enter_singleton(attempt_id);
    assert_eq!(scope_a.admission_generation(), None);
    assert_eq!(active_batch_generation(), None);
    assert!(
        !claim_wire_terminal(id, attempt_id),
        "singleton A must not claim batch B"
    );
    assert_eq!(
        batch_terminal_generation(id, attempt_id),
        Some(generation_b)
    );
    assert!(
        !batch_clear_terminal_at_generation(id, attempt_id, generation_a),
        "singleton A must not clear batch B"
    );
    assert_eq!(
        batch_terminal_generation(id, attempt_id),
        Some(generation_b)
    );
    drop(scope_a);

    assert!(batch_clear_terminal_at_generation(
        id,
        attempt_id,
        generation_b
    ));
    clear_terminal_control();
    set_active_attempt_id(0);
}

#[test]
fn batch_admission_token_reuse_rejects_stale_producer_operations() {
    let _lock = begin_test();
    batch_clear_all_terminals();
    let id = "batch-owner-reuse";
    let attempt_id = 43;
    let generation_a =
        hipfire_engine::terminal::batch_announce_terminal(id, attempt_id).expect("generation A");
    assert!(hipfire_engine::terminal::batch_transition_to_queued(
        id,
        attempt_id,
        generation_a
    ));
    let ticket_a = LaneTicket {
        lane: 0,
        generation: 1,
        admission: generation_a,
    };
    assert!(batch_bind_active(id, attempt_id, generation_a, ticket_a));
    assert!(batch_clear_terminal_at_generation(
        id,
        attempt_id,
        generation_a
    ));

    let generation_b =
        hipfire_engine::terminal::batch_announce_terminal(id, attempt_id).expect("generation B");
    assert_ne!(generation_a, generation_b);
    assert!(!hipfire_engine::terminal::batch_transition_to_queued(
        id,
        attempt_id,
        generation_a
    ));
    assert!(hipfire_engine::terminal::batch_transition_to_queued(
        id,
        attempt_id,
        generation_b
    ));
    let ticket_b = LaneTicket {
        lane: 0,
        generation: 2,
        admission: generation_b,
    };
    assert!(!batch_bind_active(id, attempt_id, generation_a, ticket_a));
    assert!(batch_bind_active(id, attempt_id, generation_b, ticket_b));
    let pending = serde_json::json!({
        "type": "done",
        "id": id,
        "attempt_id": attempt_id,
    });
    assert!(!batch_mark_ready_with_pending(
        id,
        attempt_id,
        generation_a,
        ticket_a,
        pending.clone()
    ));
    assert!(batch_mark_ready_with_pending(
        id,
        attempt_id,
        generation_b,
        ticket_b,
        pending
    ));
    assert!(!batch_clear_terminal_at_generation(
        id,
        attempt_id,
        generation_a
    ));
    assert_eq!(
        batch_terminal_generation(id, attempt_id),
        Some(generation_b)
    );
    assert!(batch_clear_terminal_at_generation(
        id,
        attempt_id,
        generation_b
    ));
    reset();
}

#[test]
fn abort_helper_emits_exactly_one_correlated_aborted_terminal() {
    let _lock = begin_test();
    set_active_attempt_id(41);
    activate_terminal_control("req-abort", 41);
    // Latch Abort the way the stdin reader does on client cancel.
    apply_terminal_control("abort", "req-abort", 41);

    let mut out = Vec::new();
    assert!(
        emit_aborted_terminal_after_abort(&mut out, "req-abort", 3),
        "abort terminal must be delivered"
    );
    // Exactly-once: the wire-terminal claim is consumed, so a repeat call (or
    // a racing error path) must emit nothing more.
    assert!(
        !emit_aborted_terminal_after_abort(&mut out, "req-abort", 3),
        "repeat abort terminal must be suppressed"
    );

    let text = String::from_utf8(out).expect("terminal output is JSONL UTF-8");
    let events: Vec<serde_json::Value> = text
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).expect("each terminal line is JSON"))
        .collect();
    assert_eq!(
        events.len(),
        2,
        "expected aborted + aborted-done pair, got: {text}"
    );

    assert_eq!(
        events[0].get("type").and_then(|v| v.as_str()),
        Some("aborted")
    );
    assert_eq!(
        events[0].get("id").and_then(|v| v.as_str()),
        Some("req-abort")
    );
    assert_eq!(
        events[0].get("attempt_id").and_then(|v| v.as_u64()),
        Some(41)
    );

    let done_count = events
        .iter()
        .filter(|v| v.get("type").and_then(|v| v.as_str()) == Some("done"))
        .count();
    assert_eq!(done_count, 1, "exactly one done envelope, got: {text}");
    assert_eq!(events[1].get("type").and_then(|v| v.as_str()), Some("done"));
    assert_eq!(
        events[1].get("finish_reason").and_then(|v| v.as_str()),
        Some("aborted"),
        "never a success done on abort, got: {text}"
    );
    assert_eq!(
        events[1].get("id").and_then(|v| v.as_str()),
        Some("req-abort")
    );
    assert_eq!(
        events[1].get("attempt_id").and_then(|v| v.as_u64()),
        Some(41)
    );
    reset();
}
