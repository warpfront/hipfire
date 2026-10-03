// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! `HIPFIRE_REDLINE_GAP_TIMING=1`: per-token host breakdown of the plain-AR
//! decode step (embedding launch, position copy, retained-PM4 patch and
//! submit/wait with its in-IB GPU span, or the HIP-graph launch), plus the
//! host time between consecutive decode steps. Diagnostic only; the disabled
//! path reads no clocks. One summary line per run of consecutive positions
//! (at most 128 steps) on stderr.

use std::cell::RefCell;
use std::sync::LazyLock;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub enum Slot {
    Embed = 0,
    PosCopy,
    Patch,
    SubmitWait,
    GpuSpan,
    GraphLaunch,
    Forward,
    Between,
}

const SLOTS: usize = 8;
const NAMES: [&str; SLOTS] = [
    "embed", "pos_copy", "patch", "submit_wait", "gpu_span", "graph_launch", "forward", "between",
];
const WINDOW: usize = 128;

struct State {
    current: [u64; SLOTS],
    tokens: Vec<[u64; SLOTS]>,
    route: &'static str,
    first_pos: usize,
    last_pos: Option<usize>,
    last_end: Option<Instant>,
}

static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    hipfire_config::process_value("HIPFIRE_REDLINE_GAP_TIMING").as_deref() == Some("1")
});

thread_local! {
    // The decode step runs on one thread; per-thread state needs no lock.
    static STATE: RefCell<State> = const {
        RefCell::new(State {
            current: [0; SLOTS],
            tokens: Vec::new(),
            route: "",
            first_pos: 0,
            last_pos: None,
            last_end: None,
        })
    };
}

pub fn enabled() -> bool {
    *ENABLED
}

pub fn now() -> Option<Instant> {
    enabled().then(Instant::now)
}

pub fn add_since(slot: Slot, start: Option<Instant>) {
    if let Some(start) = start {
        add_ns(slot, start.elapsed().as_nanos() as u64);
    }
}

pub fn add_ns(slot: Slot, ns: u64) {
    if !enabled() {
        return;
    }
    STATE.with_borrow_mut(|state| state.current[slot as usize] += ns);
}

/// Start of one decode step.
pub fn begin_forward() -> Option<Instant> {
    let start = now()?;
    STATE.with_borrow_mut(|state| state.current = [0; SLOTS]);
    Some(start)
}

/// End of one decode step at `pos` on `route`. A position discontinuity, a
/// route change or a full window flushes one summary line; `between` (host
/// time from the previous step's end to this step's start) is recorded only
/// for steps that continue the previous position.
pub fn end_forward(start: Option<Instant>, pos: usize, route: &'static str) {
    let Some(start) = start else {
        return;
    };
    let end = Instant::now();
    STATE.with_borrow_mut(|state| {
        let continues = state.last_pos == Some(pos.wrapping_sub(1)) && state.route == route;
        if !continues {
            flush(state);
        }
        if state.tokens.is_empty() {
            state.first_pos = pos;
            state.route = route;
        }
        state.current[Slot::Forward as usize] = end.duration_since(start).as_nanos() as u64;
        state.current[Slot::Between as usize] = match (continues, state.last_end) {
            (true, Some(last)) => start.duration_since(last).as_nanos() as u64,
            _ => u64::MAX,
        };
        let record = state.current;
        state.tokens.push(record);
        state.last_pos = Some(pos);
        state.last_end = Some(end);
        if state.tokens.len() >= WINDOW {
            flush(state);
        }
    });
}

fn flush(state: &mut State) {
    let tokens = std::mem::take(&mut state.tokens);
    if tokens.is_empty() {
        return;
    }
    let mut line = format!(
        "[gap-timing] route={} first_pos={} n={}",
        state.route,
        state.first_pos,
        tokens.len()
    );
    for (slot, name) in NAMES.iter().enumerate() {
        let mut values = tokens
            .iter()
            .map(|t| t[slot])
            .filter(|&v| v != u64::MAX)
            .collect::<Vec<_>>();
        if values.is_empty() {
            continue;
        }
        values.sort_unstable();
        let mean = values.iter().sum::<u64>() as f64 / values.len() as f64;
        let median = values[values.len() / 2];
        line.push_str(&format!(" {name}_us={:.2}/{:.2}", mean / 1e3, median as f64 / 1e3));
    }
    eprintln!("{line} (mean/median)");
}
