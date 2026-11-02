//! GPU-free pin for #822: a client abort during a speculative prefill (MTP,
//! DSpark) is a cancel (`aborted` + `done`), never a `validation` error; a real
//! prefill failure stays the fail-closed `validation` error.

#![allow(clippy::all)]

use hipfire_engine::terminal::*;
use hipfire_generate::ar::{emit_generation_start, GenerationRoute, GenerationRouteScope};
use hipfire_generate::common::RollbackEpilogue;
use hipfire_generate::qwen::{
    classify_spec_prefill, emit_spec_prefill_terminal, SpecPrefillTerminal,
};
use hipfire_runtime::spec::PrefillOutcome;

static TERMINAL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn attested() -> RollbackEpilogue {
    RollbackEpilogue {
        rolled_back: true,
        context: None,
    }
}

/// Events the route emits for `terminal` after an attested rollback.
fn events_for(id: &str, attempt: u64, terminal: &SpecPrefillTerminal) -> Vec<serde_json::Value> {
    let _guard = TERMINAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    clear_terminal_control();
    set_active_attempt_id(attempt);
    activate_terminal_control(id, attempt);
    let mut sink = Vec::new();
    {
        let _scope = GenerationRouteScope::enter(GenerationRoute::QwenAr, id);
        emit_generation_start(GenerationRoute::QwenAr, &mut sink, id, false);
        emit_spec_prefill_terminal(&mut sink, id, terminal, &attested());
    }
    clear_terminal_control();
    set_active_attempt_id(0);
    std::str::from_utf8(&sink)
        .expect("UTF-8 events")
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).expect("JSON event"))
        .collect()
}

fn types(events: &[serde_json::Value]) -> Vec<&str> {
    events
        .iter()
        .map(|e| e["type"].as_str().unwrap_or(""))
        .collect()
}

#[test]
fn ready_prefill_yields_the_seed() {
    let ready = Ok(PrefillOutcome::Ready { first_token: 42 });
    assert_eq!(classify_spec_prefill(ready), Ok(42));
}

#[test]
fn mtp_prefill_abort_is_classified_as_a_cancel() {
    assert_eq!(
        classify_spec_prefill(Ok(PrefillOutcome::Aborted)),
        Err(SpecPrefillTerminal::Cancelled)
    );
}

#[test]
fn real_prefill_error_is_classified_as_a_failure() {
    assert_eq!(
        classify_spec_prefill(Err("empty prompt".into())),
        Err(SpecPrefillTerminal::Failed("prefill: empty prompt".into()))
    );
}

#[test]
fn aborted_prefill_emits_aborted_and_done_with_no_validation_error() {
    let events = events_for("req-822-cancel", 82_201, &SpecPrefillTerminal::Cancelled);
    let kinds = types(&events);
    assert!(kinds.contains(&"aborted"), "{events:?}");
    assert!(kinds.contains(&"done"), "{events:?}");
    assert!(!kinds.contains(&"error"), "{events:?}");
    assert!(
        !events.iter().any(|e| e["class"] == "validation"),
        "{events:?}"
    );
}

#[test]
fn failed_prefill_emits_one_validation_error_and_no_done() {
    let terminal = SpecPrefillTerminal::Failed("prefill: empty prompt".into());
    let events = events_for("req-822-fail", 82_202, &terminal);
    let errors: Vec<&serde_json::Value> = events.iter().filter(|e| e["type"] == "error").collect();
    assert_eq!(errors.len(), 1, "{events:?}");
    assert_eq!(errors[0]["class"], "validation");
    assert_eq!(errors[0]["rolled_back"], true);
    assert!(!types(&events).contains(&"done"), "{events:?}");
}
