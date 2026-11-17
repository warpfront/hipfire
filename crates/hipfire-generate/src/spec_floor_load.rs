// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The driver-owned speculate-vs-AR floor's prices are fixed at load
//! (SPEC-AR-FLOOR S2a-5, verdict §7.11).
//!
//! A qwen35 DFlash load whose speculator owns the floor
//! (`FloorControl::Driver`) is calibrated once, after the drafter is attached
//! and before the daemon acks `loaded`: a fixed built-in text is prefilled
//! through the speculator's own `prefill`, then at least
//! [`CALIBRATION_MIN_WINDOWS`] speculative windows (`step`) and at least
//! [`CALIBRATION_MIN_AR`] AR tokens (`ar_step`) run on the load's own route
//! and executors. Their loop-to-loop wall times go through the same price
//! rules a request's do (first-use windows and the first
//! [`crate::qwen::SPEC_FLOOR_UNPRICED_AR`] AR tokens of a run are never a
//! price) until the base context bucket pins (≥ 4 kept AR samples, ≥ 8 kept
//! full-block window samples). The pin is frozen as the load's snapshot; then
//! the speculator ([`Speculator::reset_to_load`]) and the target slot (KV and
//! DeltaNet state) are reset, so the first request sees a fresh load.
//!
//! No request ever calibrates or changes its load's prices. A load whose
//! calibration cannot run (no singleton DFlash state, pipeline / expert
//! parallel, no tokenizer, a context too small for the text) or does not
//! pin within [`CALIBRATION_MAX_WINDOWS`] windows and [`CALIBRATION_MAX_AR`]
//! AR tokens has no snapshot: every request on it runs the count line, and
//! the load says so once.

use hipfire_arch_qwen35::speculative::ModelSlot;
use hipfire_loader::spec_build::Qwen35SlotGuard;
use hipfire_loader::LoadedModel;
use hipfire_runtime::spec::{DriverFloor, FloorControl, FloorSnapshotSource, Speculator};
use hipfire_runtime::spec_prices::{PinnedPrices, PIN_MIN_WINDOW_SAMPLES};
use std::time::Instant;

/// The built-in calibration text: plain prose, encoded without a chat
/// frame. Its prices are the AR token's and the full block's, which do not
/// depend on what the text says; only its length matters (the windows stay
/// in the base context bucket).
pub const CALIBRATION_TEXT: &str = "The harbor town woke slowly on the first cold morning of the \
season. Fishermen carried coils of rope down to the water while the baker on the corner \
pulled the first loaves from his oven and set them on the wide stone sill to cool. A thin \
fog drifted in from the sea and settled over the rooftops, so that the church tower seemed \
to float above the street without any foundation at all. Children walked to school in \
pairs, their breath showing in small white clouds, and the shopkeepers swept the steps in \
front of their doors and called greetings across the square. Down at the end of the long \
pier an old woman sat on an upturned crate and mended a net, her fingers moving quickly \
through the knots she had tied a thousand times before. She had watched the boats leave \
and return for more than sixty years, and she could tell from the color of the water and \
the shape of the clouds whether the catch would be good. Today, she said to anyone who \
stopped to listen, the wind would turn by noon and the boats would come home early, heavy \
with silver fish and tired men who would want soup, bread, and a quiet evening by the fire.";

/// Fewest speculative windows a calibration runs (§7.11: at least 12, so
/// the full-block cell keeps 8 samples past the width's first-use window
/// and the cell's discarded first observation).
pub const CALIBRATION_MIN_WINDOWS: usize = 12;
/// Speculative windows after which a calibration that has not kept
/// [`PIN_MIN_WINDOW_SAMPLES`] full-block samples gives up (lookup-narrowed
/// or first-use windows are not full-block samples).
pub const CALIBRATION_MAX_WINDOWS: usize = 40;
/// Fewest AR tokens a calibration runs: the first two of the run are never
/// a price, which leaves the four AR samples a pin needs.
pub const CALIBRATION_MIN_AR: usize = 6;
/// AR tokens after which a calibration that has not pinned gives up (a
/// retained route's tape recording marks its AR tokens first-use).
pub const CALIBRATION_MAX_AR: usize = 32;

/// What one load-time calibration measured.
#[derive(Clone, Debug)]
pub struct SpecFloorCalibration {
    /// The base bucket's pin, once both cells held enough samples.
    pub snapshot: Option<PinnedPrices>,
    /// Speculative windows run.
    pub windows: usize,
    /// AR tokens run.
    pub ar_steps: usize,
    /// Wall time of the prefill, the windows and the AR tokens.
    pub wall_ms: f64,
    /// Kept AR and full-block window samples when it stopped.
    pub kept: (usize, usize),
}

fn driver(spec: &mut dyn Speculator) -> Option<&mut DriverFloor> {
    match spec.floor_control() {
        FloorControl::Driver(df) => Some(df),
        FloorControl::Producer | FloorControl::Unsupported => None,
    }
}

/// The calibration's loop-to-loop clock: the request loop's pricing rules
/// (`SpecFloorRun::mark` / `observe`) without a floor decision.
#[derive(Default)]
struct CalibrationClock {
    mark: Option<Instant>,
    first_use: Option<rdna_compute::FirstUseMarker>,
    /// The step since the last mark: verified rows (0 = AR) and position
    /// before it.
    pending: Option<(usize, usize)>,
    /// AR tokens observed since the last speculative window.
    ar_run: usize,
}

impl CalibrationClock {
    /// Price the pending step with the wall time since the last mark, unless
    /// it paid first-use cost (its width's or route's first, or a first-use
    /// marker change across it).
    fn mark(
        &mut self,
        df: &mut DriverFloor,
        now: Instant,
        first_use: rdna_compute::FirstUseMarker,
    ) {
        let previous = self.mark.replace(now);
        let paid = self
            .first_use
            .replace(first_use)
            .is_some_and(|before| before != first_use);
        if let (Some((rows, ctx)), Some(at)) = (self.pending.take(), previous) {
            let us = now.saturating_duration_since(at).as_secs_f64() * 1e6;
            let cold = df.first_use(rows) | paid;
            if !cold {
                crate::qwen::spec_floor_record_price(df, rows, ctx, us as f32, self.ar_run);
            }
            self.ar_run = if rows == 0 { self.ar_run + 1 } else { 0 };
        }
    }

    fn done(&mut self, rows: usize, ctx: usize) {
        self.pending = Some((rows, ctx));
    }
}

/// Run the load-time calibration through `spec` on `slot` from a fresh
/// state: prefill `prompt`, then speculative windows until the full-block
/// cell holds its samples (at least [`CALIBRATION_MIN_WINDOWS`]), then AR
/// tokens until the base bucket pins (at least [`CALIBRATION_MIN_AR`]).
/// Measures only: target and drafter are left advanced, for
/// [`reset_after_spec_floor_calibration`]. `Err` for a speculator without a
/// driver-owned floor, a context that cannot hold the run, or a GPU error.
pub fn run_spec_floor_calibration(
    gpu: &mut rdna_compute::Gpu,
    spec: &mut dyn Speculator,
    slot: &mut ModelSlot,
    prompt: &[u32],
    ctx_capacity: usize,
) -> Result<SpecFloorCalibration, String> {
    use hipfire_runtime::spec::{PrefillOutcome, SpecTarget};
    let block = spec.block_size();
    let budget = CALIBRATION_MAX_WINDOWS * block + CALIBRATION_MAX_AR;
    if prompt.is_empty() || prompt.len() + budget + block >= ctx_capacity {
        return Err(format!(
            "context {ctx_capacity} cannot hold the {}-token text and {budget} decode rows",
            prompt.len()
        ));
    }
    let route = crate::qwen::spec_floor_route_key(gpu);
    let full_rows = {
        let df = driver(spec).ok_or("the speculator has no driver-owned floor")?;
        df.prices.set_route(route);
        df.prices.full_rows()
    };
    let t0 = Instant::now();
    slot.session_request_write_limit(prompt.len() + budget + block);
    let first = match spec.prefill(gpu, slot, prompt, prompt, 0, false, None, &|| false)? {
        PrefillOutcome::Ready { first_token } => first_token,
        PrefillOutcome::Aborted => return Err("prefill aborted".into()),
    };
    slot.map_ahead_for_decode(gpu, prompt.len(), budget, block);
    let base = prompt.len();
    let (mut position, mut seed) = (base, first);
    let mut emitted = Vec::with_capacity(budget + 1);
    emitted.push(first);
    let mut clock = CalibrationClock::default();
    let mut windows = 0;
    loop {
        let marker = gpu.first_use_marker();
        let df = driver(spec).ok_or("the driver-owned floor went away")?;
        clock.mark(df, Instant::now(), marker);
        let full = df.prices.kept(full_rows, base) >= PIN_MIN_WINDOW_SAMPLES;
        if (windows >= CALIBRATION_MIN_WINDOWS && full) || windows == CALIBRATION_MAX_WINDOWS {
            break;
        }
        let step = spec.step(gpu, slot, position, seed, &emitted, None, 0.0, usize::MAX)?;
        if step.emit.is_empty() {
            return Err(format!("window at {position} emitted nothing"));
        }
        clock.done(crate::qwen::spec_floor_rows(false, step.proposed), position);
        position += step.emit.len();
        emitted.extend_from_slice(&step.emit);
        seed = step.next_seed;
        windows += 1;
    }
    let mut ar_steps = 0;
    let snapshot = loop {
        let marker = gpu.first_use_marker();
        let df = driver(spec).ok_or("the driver-owned floor went away")?;
        clock.mark(df, Instant::now(), marker);
        let pinned = df.prices.pinned(base);
        if (ar_steps >= CALIBRATION_MIN_AR && pinned.is_some()) || ar_steps == CALIBRATION_MAX_AR {
            break pinned;
        }
        let next = spec.ar_step(gpu, slot, position, seed, &emitted)?;
        clock.done(0, position);
        position += 1;
        emitted.push(next);
        seed = next;
        ar_steps += 1;
    };
    let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
    let df = driver(spec).ok_or("the driver-owned floor went away")?;
    Ok(SpecFloorCalibration {
        snapshot,
        windows,
        ar_steps,
        wall_ms,
        kept: (df.prices.kept(0, base), df.prices.kept(full_rows, base)),
    })
}

/// The reset after a calibration (and the sequence the redline fixtures
/// use): the drafter back to a fresh load's state
/// ([`Speculator::reset_to_load`]) with empty price cells, and the target's
/// KV cache and DeltaNet state cleared. The snapshot is untouched.
pub fn reset_after_spec_floor_calibration(
    gpu: &mut rdna_compute::Gpu,
    spec: &mut dyn Speculator,
    slot: &mut ModelSlot,
) -> Result<(), String> {
    spec.reset_to_load(gpu)?;
    if let Some(df) = driver(spec) {
        df.clear_prices();
    }
    crate::redline::redline_reset_qwen_slot(gpu, slot)
}

/// Why a load has no snapshot, logged once.
fn count_line(reason: &str) {
    eprintln!(
        "[spec-floor] no load snapshot ({reason}): every greedy DFlash request on this load runs the count line"
    );
}

/// Give a qwen35 DFlash load with a driver-owned floor its frozen prices
/// before it accepts a request (see the module doc). Loads without a
/// driver-owned floor, and loads with `HIPFIRE_SPEC_RETIRE` set (every
/// request runs the count line), are left alone. `Err` only when the reset
/// after a calibration failed: the target and drafter state is then unknown
/// and the load must not be published.
pub fn prepare_spec_floor(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
    let Some(spec) = m.speculator.as_mut() else {
        return Ok(());
    };
    if driver(spec.as_mut()).is_none() {
        return Ok(());
    }
    if hipfire_config::developer_var("HIPFIRE_SPEC_RETIRE").is_ok() {
        eprintln!("[spec-floor] HIPFIRE_SPEC_RETIRE is set: every request runs the count line; no load calibration");
        return Ok(());
    }
    if m.pp > 1 || m.ep.is_some() {
        count_line("pipeline or expert parallel load");
        return Ok(());
    }
    let Some(tokenizer) = m.tokenizer.as_ref() else {
        count_line("no tokenizer");
        return Ok(());
    };
    let prompt = tokenizer.encode(CALIBRATION_TEXT);
    let ctx_capacity = spec.ctx_capacity().min(m.physical_cap);
    let mut guard = match Qwen35SlotGuard::take(&mut m.state) {
        Ok(guard) => guard,
        Err(e) => {
            count_line(&e);
            return Ok(());
        }
    };
    let slot = match guard.model_slot() {
        Ok(slot) => slot,
        Err(e) => {
            count_line(&e);
            return Ok(());
        }
    };
    let measured = run_spec_floor_calibration(gpu, spec.as_mut(), slot, &prompt, ctx_capacity);
    let reset = reset_after_spec_floor_calibration(gpu, spec.as_mut(), slot);
    drop(guard);
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    reset.map_err(|e| format!("speculation floor calibration reset failed: {e}"))?;
    let calibration = match measured {
        Ok(calibration) => calibration,
        Err(e) => {
            count_line(&format!("calibration failed: {e}"));
            return Ok(());
        }
    };
    let Some(prices) = calibration.snapshot else {
        count_line(&format!(
            "calibration did not pin: {} of 4 AR and {} of {PIN_MIN_WINDOW_SAMPLES} full-block samples after {} windows and {} AR tokens",
            calibration.kept.0, calibration.kept.1, calibration.windows, calibration.ar_steps
        ));
        return Ok(());
    };
    let df = m
        .speculator
        .as_mut()
        .and_then(|spec| driver(spec.as_mut()))
        .ok_or("the driver-owned floor went away")?;
    df.freeze_snapshot(prices, FloorSnapshotSource::LoadCalibrated);
    let full = prices.window_us[prices.full_rows];
    eprintln!(
        "[spec-floor] load-calibrated: ar_us={:.1} full-block window_us={:.1} ({} rows) break_even={:.3}; {} windows, {} AR tokens, {:.0} ms",
        prices.ar_us,
        full,
        prices.full_rows,
        full / prices.ar_us,
        calibration.windows,
        calibration.ar_steps,
        calibration.wall_ms
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::spec_prices::{ArRouteKey, PIN_MIN_AR_SAMPLES};
    use std::time::Duration;

    fn marker(kernels: usize) -> rdna_compute::FirstUseMarker {
        rdna_compute::FirstUseMarker {
            kernels,
            recorded_launches: 0,
            graph_blobs: 0,
            replay_state: rdna_compute::replay::ReplayState::Ready,
        }
    }

    /// The calibration loop on a scripted clock: `window_ms(i)` per window,
    /// `ar_ms(i)` per AR token, `tape` AR tokens that move the first-use
    /// marker (a retained route recording its tape). Returns the pin and the
    /// windows and AR tokens it ran.
    fn scripted(
        window_ms: impl Fn(usize) -> f64,
        ar_ms: impl Fn(usize) -> f64,
        tape: usize,
    ) -> (Option<PinnedPrices>, usize, usize) {
        let mut df = DriverFloor::new(16, false);
        df.prices.set_route(ArRouteKey::new(2, true));
        let base = 300;
        let mut clock = CalibrationClock::default();
        let mut now = Instant::now();
        let (mut windows, mut position, mut kernels) = (0, base, 0);
        loop {
            clock.mark(&mut df, now, marker(kernels));
            let full = df.prices.kept(16, base) >= PIN_MIN_WINDOW_SAMPLES;
            if (windows >= CALIBRATION_MIN_WINDOWS && full) || windows == CALIBRATION_MAX_WINDOWS {
                break;
            }
            now += Duration::from_secs_f64(window_ms(windows) / 1e3);
            clock.done(16, position);
            position += 3;
            windows += 1;
        }
        let mut ar = 0;
        let pinned = loop {
            clock.mark(&mut df, now, marker(kernels));
            let pinned = df.prices.pinned(base);
            if (ar >= CALIBRATION_MIN_AR && pinned.is_some()) || ar == CALIBRATION_MAX_AR {
                break pinned;
            }
            if ar < tape {
                kernels += 1;
            }
            now += Duration::from_secs_f64(ar_ms(ar) / 1e3);
            clock.done(0, position);
            position += 1;
            ar += 1;
        };
        (pinned, windows, ar)
    }

    #[test]
    fn a_steady_load_pins_in_12_windows_and_6_ar_tokens() {
        // The first window pays its width's first use, the cell discards the
        // second: windows 3..=10 are the 8 kept samples, so 12 windows hold
        // them. The first two AR tokens pay re-capture and are not priced.
        let (pinned, windows, ar) = scripted(
            |i| if i == 0 { 900.0 } else { 39.0 },
            |i| if i < 2 { 30.0 } else { 23.4 },
            0,
        );
        assert_eq!((windows, ar), (CALIBRATION_MIN_WINDOWS, CALIBRATION_MIN_AR));
        let pinned = pinned.expect("pinned");
        assert!((pinned.ar_us - 23_400.0).abs() < 1.0, "{pinned:?}");
        assert!((pinned.window(16) - 39_000.0).abs() < 1.0, "{pinned:?}");
        assert_eq!(PIN_MIN_AR_SAMPLES, 4);
    }

    #[test]
    fn a_recording_tape_extends_the_ar_phase_until_four_priced_tokens() {
        // A retained route recording its tape across the first 8 AR tokens:
        // each moves the first-use marker and is not priced. The run is 8
        // tokens long by then, so tokens 9..=12 price and the bucket pins
        // after the 12th.
        let (pinned, _, ar) = scripted(|_| 39.0, |_| 23.4, 8);
        assert_eq!(ar, 12);
        assert!(pinned.is_some());
    }

    #[test]
    fn a_calibration_that_cannot_pin_stops_at_its_caps() {
        // Every AR token moves the marker: no AR price, no pin.
        let (pinned, windows, ar) = scripted(|_| 39.0, |_| 23.4, usize::MAX);
        assert_eq!(pinned, None);
        assert_eq!((windows, ar), (CALIBRATION_MIN_WINDOWS, CALIBRATION_MAX_AR));
    }

    #[test]
    fn the_calibration_text_stays_in_the_base_bucket() {
        // About 4 characters per token: the text, the windows' rows and the
        // AR tokens stay below the first bucket edge.
        let rows = CALIBRATION_MAX_WINDOWS * 16 + CALIBRATION_MAX_AR;
        assert!(
            CALIBRATION_TEXT.len() / 3 + rows < hipfire_runtime::spec_prices::PRICE_BUCKET_EDGES[0]
        );
    }
}
