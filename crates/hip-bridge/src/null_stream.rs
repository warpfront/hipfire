// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Null-stream submission ledger (`HIPFIRE_NULL_STREAM_LEDGER=1`, developer).
//!
//! Every [`crate::HipRuntime`] call that puts work on the HIP legacy null stream (the synchronous
//! `hipMemcpy` / `hipMemset` family, the `*_default` and `stream: None` async variants, a kernel
//! launch, an event record or a stream wait without a stream) notes its caller here. With the env
//! set, a site prints one line the first time it submits and again at every power of two of its
//! count, so a served run shows which sites still feed the null stream in steady state and how
//! often. Off (the default), a note is one relaxed atomic load.
//!
//! Decode and verify steps run inside a [`steady_step`] scope, and a ledger line reads
//! `in_step=1` for a submission made while one is open on any thread. The regression guard is
//! `HIPFIRE_NULL_STREAM_LEDGER=deny`: such a submission panics with its site, so a new per-step
//! null-stream site fails the run that reaches it instead of slipping in.
//!
//! Why it matters: the null stream is its own hardware queue. When a busy serve keeps submitting to
//! it, that queue holds packets (the implicit null-stream barriers wait on the compute stream) and
//! competes for the CP pipe with the compute queue; when the queue mapper puts both on one pipe the
//! CP time-slices them and every dispatch on that device pays the slice.

use std::collections::HashMap;
use std::panic::Location;
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::Mutex;

/// The ledger switch (developer): `1` logs, `deny` also panics on an in-step submission.
pub const NULL_STREAM_LEDGER_ENV: &str = "HIPFIRE_NULL_STREAM_LEDGER";

const UNKNOWN: u8 = 0;
const OFF: u8 = 1;
const ON: u8 = 2;
const DENY: u8 = 3;
static STATE: AtomicU8 = AtomicU8::new(UNKNOWN);
static COUNTS: Mutex<Option<HashMap<(&'static str, u32, &'static str), u64>>> = Mutex::new(None);
/// Open [`SteadyStep`] scopes, process-wide: a step's work may be submitted from rank threads.
static STEPS_OPEN: AtomicU32 = AtomicU32::new(0);

fn state() -> u8 {
    match STATE.load(Ordering::Relaxed) {
        UNKNOWN => {
            let s = match hipfire_config::developer_var(NULL_STREAM_LEDGER_ENV).as_deref() {
                Ok("1") => ON,
                Ok("deny") => DENY,
                _ => OFF,
            };
            STATE.store(s, Ordering::Relaxed);
            s
        }
        s => s,
    }
}

/// One decode or verify step in flight (see [`steady_step`]); closes on drop.
#[must_use = "the scope closes when the guard drops"]
pub struct SteadyStep(());

impl Drop for SteadyStep {
    fn drop(&mut self) {
        STEPS_OPEN.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Open a steady-state step scope: from here to the guard's drop, every null-stream submission is
/// per-step work (ledger `in_step=1`; a panic under `deny`). Wrap a decode step or verify window,
/// never load, prefill setup or teardown.
pub fn steady_step() -> SteadyStep {
    STEPS_OPEN.fetch_add(1, Ordering::Relaxed);
    SteadyStep(())
}

/// Steps a request runs before its decode is steady state: graph capture, first-use kernel loads
/// and scratch growth happen in them.
pub const WARM_STEPS: usize = 2;

/// [`steady_step`] for a request's `step`-th decode step or verify window (0-based), `None`
/// during the first [`WARM_STEPS`] and when the step's compute is not `on_stream` (a route that
/// computes on the null stream itself has nothing for its null-stream copies to wait behind).
pub fn steady_step_after_warmup(step: usize, on_stream: bool) -> Option<SteadyStep> {
    (on_stream && step >= WARM_STEPS).then(steady_step)
}

/// Note one null-stream submission of `api` by `site`.
pub fn note(api: &'static str, site: &'static Location<'static>) {
    let s = state();
    if s == OFF {
        return;
    }
    let in_step = STEPS_OPEN.load(Ordering::Relaxed) > 0;
    if s == DENY && in_step {
        panic!(
            "[null-stream] {api} at {}:{} inside a steady step ({NULL_STREAM_LEDGER_ENV}=deny): \
             a per-step copy must stay on the compute stream (dtoh_stream_ordered)",
            site.file(),
            site.line()
        );
    }
    let mut guard = COUNTS.lock().unwrap_or_else(|p| p.into_inner());
    let count = {
        let map = guard.get_or_insert_with(HashMap::new);
        let entry = map.entry((site.file(), site.line(), api)).or_insert(0);
        *entry += 1;
        *entry
    };
    drop(guard);
    if count.is_power_of_two() {
        let unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        eprintln!(
            "[null-stream] {api} at {}:{} count {count} in_step={} unix_ms={unix_ms}",
            site.file(),
            site.line(),
            u8::from(in_step)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deny_panics_only_inside_a_steady_step() {
        STATE.store(DENY, Ordering::Relaxed);
        note("hipMemcpy D2H", Location::caller());
        let step = steady_step();
        let hit = std::panic::catch_unwind(|| note("hipMemcpy D2H", Location::caller()));
        drop(step);
        assert!(
            hit.is_err(),
            "an in-step null-stream submission must panic under deny"
        );
        note("hipMemcpy D2H", Location::caller());
        assert_eq!(STEPS_OPEN.load(Ordering::Relaxed), 0);
        STATE.store(UNKNOWN, Ordering::Relaxed);
    }
}
