// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Coarse phase timers for [`crate::task_graph::certify_pool`], behind
//! `railgun-cert pool --timings`.
//!
//! Disabled, [`timed`] is one relaxed atomic load around the closure. Enabled,
//! every call is recorded under the path of the enclosing `timed` calls of the
//! same thread (`certify_pool > prove_pool > t6 derive_lifted`); [`report`]
//! prints the paths in first-seen order with call count, total and self time
//! (total minus the totals of the direct children).

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

static ENABLED: AtomicBool = AtomicBool::new(false);
static LOG: Mutex<Vec<Row>> = Mutex::new(Vec::new());
static NOTES: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());

thread_local! {
    static STACK: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

struct Row {
    path: Vec<String>,
    calls: u64,
    total: Duration,
}

/// Start recording (the `--timings` flag).
pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

/// Whether [`enable`] was called.
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Run `f`, recording its wall time under `label` when timings are enabled.
pub fn timed<T>(label: &str, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let path = STACK.with(|s| {
        let mut s = s.borrow_mut();
        s.push(label.to_owned());
        s.clone()
    });
    let start = Instant::now();
    let out = f();
    let elapsed = start.elapsed();
    STACK.with(|s| {
        s.borrow_mut().pop();
    });
    let mut log = LOG.lock().unwrap_or_else(|e| e.into_inner());
    match log.iter_mut().find(|r| r.path == path) {
        Some(r) => {
            r.calls += 1;
            r.total += elapsed;
        }
        None => log.push(Row {
            path,
            calls: 1,
            total: elapsed,
        }),
    }
    out
}

/// Record a work-counter line (`visits 123 of 4096`, ...) under the enclosing phase; `text`
/// is evaluated only when timings are enabled.
pub fn note(text: impl FnOnce() -> String) {
    if !enabled() {
        return;
    }
    let path = STACK.with(|s| s.borrow().join(" > "));
    NOTES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((path, text()));
}

/// The recorded phases as a text table (empty when nothing was recorded); the log is
/// cleared, so the next pool object starts from zero.
pub fn take_report() -> String {
    let log = std::mem::take(&mut *LOG.lock().unwrap_or_else(|e| e.into_inner()));
    if log.is_empty() {
        return String::new();
    }
    let mut order: Vec<usize> = (0..log.len()).collect();
    // parents print before children, siblings in first-seen order
    order.sort_by(|&a, &b| {
        let (pa, pb) = (&log[a].path, &log[b].path);
        let key = |i: usize, p: &Vec<String>| -> Vec<usize> {
            (1..=p.len())
                .map(|n| log.iter().position(|r| r.path[..] == p[..n]).unwrap_or(i))
                .collect()
        };
        key(a, pa).cmp(&key(b, pb))
    });
    let mut out = String::from("phase                                                      calls     total ms      self ms\n");
    for &i in &order {
        let r = &log[i];
        let children: Duration = log
            .iter()
            .filter(|c| c.path.len() == r.path.len() + 1 && c.path[..r.path.len()] == r.path[..])
            .map(|c| c.total)
            .sum();
        let label = format!(
            "{}{}",
            "  ".repeat(r.path.len() - 1),
            r.path.last().map_or("", String::as_str)
        );
        out.push_str(&format!(
            "{label:<58} {:>5} {:>12.1} {:>12.1}\n",
            r.calls,
            r.total.as_secs_f64() * 1e3,
            r.total.saturating_sub(children).as_secs_f64() * 1e3,
        ));
    }
    let notes = std::mem::take(&mut *NOTES.lock().unwrap_or_else(|e| e.into_inner()));
    if !notes.is_empty() {
        out.push_str("work counters\n");
        for (path, text) in notes {
            out.push_str(&format!("  {path}: {text}\n"));
        }
    }
    out
}
