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
//! Why it matters: the null stream is its own hardware queue. When a busy serve keeps submitting to
//! it, that queue holds packets (the implicit null-stream barriers wait on the compute stream) and
//! competes for the CP pipe with the compute queue; when the queue mapper puts both on one pipe the
//! CP time-slices them and every dispatch on that device pays the slice.

use std::collections::HashMap;
use std::panic::Location;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;

/// The ledger switch (developer).
pub const NULL_STREAM_LEDGER_ENV: &str = "HIPFIRE_NULL_STREAM_LEDGER";

const UNKNOWN: u8 = 0;
const OFF: u8 = 1;
const ON: u8 = 2;
static STATE: AtomicU8 = AtomicU8::new(UNKNOWN);
static COUNTS: Mutex<Option<HashMap<(&'static str, u32, &'static str), u64>>> = Mutex::new(None);

fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        ON => true,
        OFF => false,
        _ => {
            let on = hipfire_config::developer_var(NULL_STREAM_LEDGER_ENV).as_deref() == Ok("1");
            STATE.store(if on { ON } else { OFF }, Ordering::Relaxed);
            on
        }
    }
}

/// Note one null-stream submission of `api` by `site`.
pub fn note(api: &'static str, site: &'static Location<'static>) {
    if !enabled() {
        return;
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
            "[null-stream] {api} at {}:{} count {count} unix_ms={unix_ms}",
            site.file(),
            site.line()
        );
    }
}
