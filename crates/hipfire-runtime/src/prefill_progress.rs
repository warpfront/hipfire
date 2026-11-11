// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Daemon `prefill_progress` JSONL event: how many prompt tokens a request has
//! prefilled so far.
//!
//! Wire shape (one whole line, `serde_json`-built because `id` is
//! user-supplied):
//!
//! ```text
//! {"type":"prefill_progress","id":"r1","attempt_id":3,"done":2048,"total":9000}
//! ```
//!
//! Rules every emitter shares:
//! - emitted after a prefill chunk completes, **only while `done < total`**
//!   (a single-chunk prompt, or the final chunk, produces no event — the first
//!   token/`done` event already says prefill finished);
//! - `done`/`total` count prompt tokens this request prefills (a cache-hit
//!   prefix is not counted in either);
//! - on by default; `HIPFIRE_PREFILL_PROGRESS=0` (read through
//!   [`hipfire_config::developer_var`]) suppresses every event.
//!
//! Chunk loops that own the request writer call [`emit_to`]. Loops that live
//! in crates without the writer (the Qwen3.5 MTP/DFlash prompt prefill) call
//! [`report`], which writes through the thread's [`scope`] — installed for the
//! request by `hipfire-generate`'s `GenerationRouteScope` — but only inside a
//! [`prompt_phase`]: the same chunk loops also run mid-request (forced
//! advances, terminal-prefix repair) after tokens have streamed, and those must
//! stay silent. The prompt-prefill entry point opens the phase. The default scope
//! sink writes to process stdout, the same handle the daemon writes token
//! events to (`Stdout` serialises whole `write_all` calls and the daemon never
//! wraps it in a `BufWriter`, so lines cannot interleave or reorder).

use std::cell::{Cell, RefCell};
use std::io::Write;
use std::sync::LazyLock;

/// Kill-switch variable (`0` disables the event; anything else keeps it on).
pub const ENV_VAR: &str = "HIPFIRE_PREFILL_PROGRESS";

/// Pure gate: the event is on unless the value is exactly `"0"`.
pub fn gate(value: Option<&str>) -> bool {
    value != Some("0")
}

/// Process-wide gate, read once through the developer-var snapshot.
pub fn enabled() -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| gate(hipfire_config::developer_var(ENV_VAR).ok().as_deref()));
    *ENABLED
}

/// The `done < total` rule: a chunk reports only while prefill is incomplete.
/// `done == 0` and `total == 0` never report (nothing prefilled / nothing to do).
pub fn should_report(done: usize, total: usize) -> bool {
    done > 0 && done < total
}

/// Build the event envelope. `attempt_id` is omitted when `None`.
pub fn event(id: &str, attempt_id: Option<u64>, done: usize, total: usize) -> serde_json::Value {
    let mut value = serde_json::json!({
        "type": "prefill_progress",
        "id": id,
    });
    if let (Some(attempt_id), Some(map)) = (attempt_id, value.as_object_mut()) {
        map.insert("attempt_id".to_owned(), attempt_id.into());
    }
    if let Some(map) = value.as_object_mut() {
        map.insert("done".to_owned(), done.into());
        map.insert("total".to_owned(), total.into());
    }
    value
}

/// One whole newline-terminated wire line, or `None` when `gated` is off or the
/// `done < total` rule says no event. `gated` is the resolved kill-switch.
fn line(
    gated: bool,
    id: &str,
    attempt_id: Option<u64>,
    done: usize,
    total: usize,
) -> Option<Vec<u8>> {
    if !gated || !should_report(done, total) {
        return None;
    }
    let mut bytes = event(id, attempt_id, done, total).to_string().into_bytes();
    bytes.push(b'\n');
    Some(bytes)
}

fn write_line<W: Write + ?Sized>(
    out: &mut W,
    gated: bool,
    id: &str,
    attempt_id: Option<u64>,
    done: usize,
    total: usize,
) {
    if let Some(bytes) = line(gated, id, attempt_id, done, total) {
        // One `write_all` of the whole line keeps it atomic w.r.t. other
        // writers on the same handle; flush so it precedes the first token.
        let _ = out.write_all(&bytes);
        let _ = out.flush();
    }
}

/// Emit one progress event to `out` (the request's own writer). No-op when the
/// kill switch is off or `done >= total`.
pub fn emit_to<W: Write + ?Sized>(
    out: &mut W,
    id: &str,
    attempt_id: Option<u64>,
    done: usize,
    total: usize,
) {
    write_line(out, enabled(), id, attempt_id, done, total);
}

type Sink = Box<dyn FnMut(&[u8])>;

struct Target {
    id: String,
    attempt_id: Option<u64>,
    sink: Sink,
}

thread_local! {
    static ACTIVE: RefCell<Option<Target>> = const { RefCell::new(None) };
    static PROMPT_PHASE: Cell<bool> = const { Cell::new(false) };
}

/// Guard returned by [`prompt_phase`]; restores the previous phase on drop.
#[must_use = "the prompt phase ends when the guard drops"]
pub struct PromptPhase {
    previous: bool,
}

impl Drop for PromptPhase {
    fn drop(&mut self) {
        PROMPT_PHASE.with(|phase| phase.set(self.previous));
    }
}

/// Mark this thread as inside the request's prompt prefill, so [`report`]
/// calls from shared chunk loops are live. Hold the guard across the prompt
/// prefill only (never across decode, forced advances, or repairs).
pub fn prompt_phase() -> PromptPhase {
    PromptPhase {
        previous: PROMPT_PHASE.with(|phase| phase.replace(true)),
    }
}

/// Guard returned by [`scope`]; restores the previous target on drop.
#[must_use = "the progress target is removed when the guard drops"]
pub struct Scope {
    previous: Option<Target>,
}

impl Drop for Scope {
    fn drop(&mut self) {
        let previous = self.previous.take();
        ACTIVE.with(|active| *active.borrow_mut() = previous);
    }
}

/// Install this thread's progress target for one request. Events go to process
/// stdout (the daemon's JSONL channel).
pub fn scope(id: &str, attempt_id: Option<u64>) -> Scope {
    scope_with_sink(
        id,
        attempt_id,
        Box::new(|bytes| {
            let mut stdout = std::io::stdout().lock();
            let _ = stdout.write_all(bytes);
            let _ = stdout.flush();
        }),
    )
}

/// [`scope`] with an explicit sink (tests, embedders). The sink receives one
/// whole newline-terminated line per call.
pub fn scope_with_sink(id: &str, attempt_id: Option<u64>, sink: Sink) -> Scope {
    let target = Target {
        id: id.to_owned(),
        attempt_id,
        sink,
    };
    let previous = ACTIVE.with(|active| active.borrow_mut().replace(target));
    Scope { previous }
}

fn report_gated(gated: bool, done: usize, total: usize) {
    if !PROMPT_PHASE.with(Cell::get) {
        return;
    }
    ACTIVE.with(|active| {
        if let Some(target) = active.borrow_mut().as_mut() {
            if let Some(bytes) = line(gated, &target.id, target.attempt_id, done, total) {
                (target.sink)(&bytes);
            }
        }
    });
}

/// Report prompt prefill progress for the thread's active request. No-op
/// outside a [`prompt_phase`], without an active [`scope`], with the kill
/// switch off, or at `done >= total`.
pub fn report(done: usize, total: usize) {
    report_gated(enabled(), done, total);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    fn parse(bytes: &[u8]) -> serde_json::Value {
        let text = std::str::from_utf8(bytes).expect("utf8");
        assert!(text.ends_with('\n'));
        assert_eq!(text.matches('\n').count(), 1, "exactly one line: {text:?}");
        serde_json::from_str(text.trim_end()).expect("valid json")
    }

    #[test]
    fn event_shape_with_attempt() {
        let v = event("r1", Some(7), 2048, 9000);
        assert_eq!(
            v,
            serde_json::json!({
                "type": "prefill_progress",
                "id": "r1",
                "attempt_id": 7,
                "done": 2048,
                "total": 9000,
            })
        );
    }

    #[test]
    fn event_omits_absent_attempt() {
        let v = event("r1", None, 1, 2);
        assert!(v.get("attempt_id").is_none());
        assert_eq!(v["done"], 1);
        assert_eq!(v["total"], 2);
    }

    #[test]
    fn event_line_escapes_hostile_id() {
        let id = "a\"b\\c\nd}";
        let bytes = line(true, id, Some(1), 1, 2).expect("line");
        let v = parse(&bytes);
        assert_eq!(v["id"], id);
        assert_eq!(v["type"], "prefill_progress");
    }

    #[test]
    fn done_must_be_below_total() {
        assert!(should_report(1, 2));
        assert!(should_report(2048, 9000));
        assert!(!should_report(0, 9000));
        assert!(!should_report(9000, 9000));
        assert!(!should_report(9001, 9000));
        assert!(!should_report(0, 0));
        assert!(line(true, "r", Some(1), 512, 512).is_none());
        assert!(line(true, "r", Some(1), 512, 1024).is_some());
    }

    #[test]
    fn kill_switch_gate() {
        assert!(gate(None));
        assert!(gate(Some("1")));
        assert!(gate(Some("")));
        assert!(!gate(Some("0")));
        assert!(line(gate(Some("0")), "r", Some(1), 512, 1024).is_none());
        assert!(line(gate(None), "r", Some(1), 512, 1024).is_some());
    }

    #[test]
    fn write_line_emits_one_flushed_line() {
        let mut out: Vec<u8> = Vec::new();
        write_line(&mut out, true, "r1", Some(2), 512, 1024);
        assert_eq!(parse(&out)["done"], 512);
        out.clear();
        write_line(&mut out, true, "r1", Some(2), 1024, 1024);
        write_line(&mut out, false, "r1", Some(2), 512, 1024);
        assert!(out.is_empty());
    }

    #[test]
    fn scope_routes_report_and_restores_previous() {
        let seen: Rc<RefCell<Vec<serde_json::Value>>> = Rc::default();
        let outer = seen.clone();
        let _outer = scope_with_sink(
            "outer",
            Some(1),
            Box::new(move |bytes| outer.borrow_mut().push(parse(bytes))),
        );
        let _phase = prompt_phase();
        {
            let inner_seen: Rc<RefCell<Vec<serde_json::Value>>> = Rc::default();
            let inner = inner_seen.clone();
            let _inner = scope_with_sink(
                "inner",
                None,
                Box::new(move |bytes| inner.borrow_mut().push(parse(bytes))),
            );
            report_gated(true, 256, 1024);
            report_gated(true, 1024, 1024);
            report_gated(false, 512, 1024);
            let inner_events = inner_seen.borrow();
            assert_eq!(inner_events.len(), 1);
            assert_eq!(inner_events[0]["id"], "inner");
            assert!(inner_events[0].get("attempt_id").is_none());
        }
        report_gated(true, 512, 1024);
        let events = seen.borrow();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["id"], "outer");
        assert_eq!(events[0]["attempt_id"], 1);
        assert_eq!(events[0]["done"], 512);
    }

    #[test]
    fn report_without_scope_is_noop() {
        let _phase = prompt_phase();
        report_gated(true, 1, 2);
    }

    #[test]
    fn report_is_silent_outside_prompt_phase() {
        let seen: Rc<RefCell<Vec<serde_json::Value>>> = Rc::default();
        let sink = seen.clone();
        let _scope = scope_with_sink(
            "r",
            Some(1),
            Box::new(move |bytes| sink.borrow_mut().push(parse(bytes))),
        );
        report_gated(true, 512, 1024);
        assert!(seen.borrow().is_empty());
        {
            let _phase = prompt_phase();
            report_gated(true, 512, 1024);
            assert_eq!(seen.borrow().len(), 1);
        }
        report_gated(true, 512, 1024);
        assert_eq!(seen.borrow().len(), 1, "phase ended with its guard");
    }
}
