// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Björn Bösel

//! Per-request speculative-decode retirement: switch a request from spec
//! windows to plain AR tokens once its drafter cannot pay for the cycle.
//!
//! A spec cycle commits `emit` tokens for the price of one cycle; an AR step
//! commits one token for the price of one AR step. Speculation wins while
//! `emit / cycle_time > 1 / ar_time`, i.e. while the tokens a window of cycles
//! commits stay above its cost in AR steps, `sum(cycle_time) / ar_time`. That
//! cost is the break-even line, modelled per cycle as linear in the cycle's
//! verified rows (a verify of `r` rows costs a fixed part plus about one
//! row's projections per extra row) and in context length (the verify
//! attention scans `r x L` keys per cycle where AR scans `L`):
//!
//! ```text
//! break_even(rows, L) = base + per_row * rows + per_1k_ctx * L / 1024
//! ```
//!
//! The decision is a pure function of the committed-token counts, the verified
//! rows and the position, never of wall-clock time, so a request retires at
//! the same cycle on every run with the same draws (a live timing fit flips
//! near-tie decisions between runs, and verify width changes greedy numerics).
//! Like the slot engine's MTP retirement it judges whole windows and needs
//! [`RETIRE_WINDOWS`] failing windows in a row, so one low-acceptance stretch
//! (a templated opening) does not retire a drafter that pays for itself
//! afterwards. Retirement is sticky for the rest of the request.
//!
//! A short answer (a few dozen tokens) never completes two windows, so the
//! request's first [`EARLY_CYCLES`] cycles get an early verdict: they retire
//! the request at once when they commit fewer tokens than their cost by more
//! than [`EARLY_MARGIN`] tokens per cycle (a drafter below break-even from
//! the start; Raon-1-mini's short `creative_short` answers are 42-47 tokens
//! over 22-30 cycles, and its prose drafts commit about 1.67 tokens per cycle
//! against a 1.90-step cycle on the 7900 XTX). A drafter within the margin of
//! break-even (where speculation and AR cost about the same) waits for the
//! window verdicts.
//!
//! Default break-even (`HIPFIRE_SPEC_RETIRE` unset, and the drafter states no
//! measured line of its own, [`crate::spec::Speculator::retire_break_even`]):
//! [`DEFAULT_BREAK_EVEN`], context-flat 1.5 tokens per cycle, just below the
//! smallest cost ratio measured on any box (gfx1151, Qwen3.8-27B MQ4 + MQ4
//! DFlash: 1.53 at 100 tokens, 1.61 at 2K, 1.73 at 32K; gfx1100 about 2 at
//! short context). A request below it loses to AR on every measured box. A box
//! whose verify cost grows with context takes its measured line through the
//! env knob.

/// Spec cycles per judged window.
pub const RETIRE_WINDOW: usize = 16;
/// Consecutive failing windows before the request retires to AR.
pub const RETIRE_WINDOWS: usize = 2;
/// Cycles of the request's first-window early verdict.
pub const EARLY_CYCLES: usize = 8;
/// Tokens per cycle below the break-even line that the early verdict needs
/// (`emitted + EARLY_MARGIN * EARLY_CYCLES < cost`).
pub const EARLY_MARGIN: f64 = 0.1;

/// Break-even tokens per spec cycle that verified `rows` target rows at
/// context length `L`: `base + per_row * rows + per_1k_ctx * L / 1024`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreakEven {
    pub base: f64,
    pub per_1k_ctx: f64,
    pub per_row: f64,
}

impl BreakEven {
    pub fn at(&self, rows: usize, position: usize) -> f64 {
        self.base + self.per_row * rows as f64 + self.per_1k_ctx * position as f64 / 1024.0
    }
}

pub const DEFAULT_BREAK_EVEN: BreakEven = BreakEven {
    base: 1.5,
    per_1k_ctx: 0.0,
    per_row: 0.0,
};

/// Parse `HIPFIRE_SPEC_RETIRE`: unset = `Ok(None)` (the drafter's line, else
/// [`DEFAULT_BREAK_EVEN`]); `0` / `off` = never retire; `<base>[,<per_1k_ctx>
/// [,<per_row>]]` = that line. Anything else is an error (a typo must not
/// silently fall back to the default).
pub fn parse_break_even(value: Option<&str>) -> Result<Option<Option<BreakEven>>, String> {
    let Some(raw) = value.map(str::trim) else {
        return Ok(None);
    };
    if raw == "0" || raw.eq_ignore_ascii_case("off") {
        return Ok(Some(None));
    }
    let bad =
        || format!("HIPFIRE_SPEC_RETIRE={raw:?}: expected off | <base>[,<per_1k_ctx>[,<per_row>]]");
    let mut parts = raw.split(',').map(str::trim);
    let num = |s: Option<&str>| -> Result<Option<f64>, String> {
        match s {
            None => Ok(None),
            Some(s) => match s.parse::<f64>() {
                Ok(v) if v.is_finite() && v >= 0.0 => Ok(Some(v)),
                _ => Err(bad()),
            },
        }
    };
    let base = num(parts.next())?.ok_or_else(bad)?;
    let per_1k_ctx = num(parts.next())?.unwrap_or(0.0);
    let per_row = num(parts.next())?.unwrap_or(0.0);
    if parts.next().is_some() {
        return Err(bad());
    }
    Ok(Some(Some(BreakEven {
        base,
        per_1k_ctx,
        per_row,
    })))
}

/// The line a request retires on: `HIPFIRE_SPEC_RETIRE` when set (`parsed`
/// from [`parse_break_even`]), else the drafter's own measured line, else
/// [`DEFAULT_BREAK_EVEN`]. `None` never retires.
pub fn resolve_break_even(
    parsed: Option<Option<BreakEven>>,
    drafter: Option<BreakEven>,
) -> Option<BreakEven> {
    parsed.unwrap_or(Some(drafter.unwrap_or(DEFAULT_BREAK_EVEN)))
}

/// One request's retirement state. Feed every spec cycle to [`Self::observe`].
pub struct SpecRetire {
    break_even: Option<BreakEven>,
    cycles: usize,
    emitted: usize,
    /// Break-even tokens of the window's cycles so far.
    cost: f64,
    failed_windows: usize,
    /// Whole windows judged so far (the early verdict runs before the first).
    windows: usize,
    retired_at: Option<usize>,
}

impl SpecRetire {
    /// `None` never retires.
    pub fn new(break_even: Option<BreakEven>) -> Self {
        Self {
            break_even,
            cycles: 0,
            emitted: 0,
            cost: 0.0,
            failed_windows: 0,
            windows: 0,
            retired_at: None,
        }
    }

    pub fn is_retired(&self) -> bool {
        self.retired_at.is_some()
    }

    /// Context position at the cycle that retired the request.
    pub fn retired_at(&self) -> Option<usize> {
        self.retired_at
    }

    /// Record one spec cycle that verified `rows` target rows, committed
    /// `emitted` tokens and left the request at absolute `position`. Returns
    /// true on the cycle that retires.
    pub fn observe(&mut self, emitted: usize, rows: usize, position: usize) -> bool {
        let Some(break_even) = self.break_even else {
            return false;
        };
        if self.is_retired() {
            return false;
        }
        self.cycles += 1;
        self.emitted += emitted;
        self.cost += break_even.at(rows, position);
        if self.cycles == EARLY_CYCLES
            && self.windows == 0
            && (self.emitted as f64) + EARLY_MARGIN * (EARLY_CYCLES as f64) < self.cost
        {
            self.retired_at = Some(position);
            return true;
        }
        if self.cycles < RETIRE_WINDOW {
            return false;
        }
        let failed = (self.emitted as f64) < self.cost;
        self.windows += 1;
        self.cycles = 0;
        self.emitted = 0;
        self.cost = 0.0;
        if failed {
            self.failed_windows += 1;
        } else {
            self.failed_windows = 0;
        }
        if self.failed_windows < RETIRE_WINDOWS {
            return false;
        }
        self.retired_at = Some(position);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(r: &mut SpecRetire, cycles: usize, emitted: usize, position: &mut usize) -> bool {
        feed_rows(r, cycles, emitted, 2, position)
    }

    fn feed_rows(
        r: &mut SpecRetire,
        cycles: usize,
        emitted: usize,
        rows: usize,
        position: &mut usize,
    ) -> bool {
        let mut retired = false;
        for _ in 0..cycles {
            *position += emitted;
            retired |= r.observe(emitted, rows, *position);
        }
        retired
    }

    #[test]
    fn retires_after_consecutive_failing_windows_only() {
        let mut r = SpecRetire::new(Some(DEFAULT_BREAK_EVEN));
        let mut pos = 0;
        // A passing first window (no early verdict), then the window verdicts.
        assert!(!feed(&mut r, RETIRE_WINDOW, 3, &mut pos));
        assert!(!feed(&mut r, RETIRE_WINDOW, 1, &mut pos));
        // A passing window resets the streak.
        assert!(!feed(&mut r, RETIRE_WINDOW, 3, &mut pos));
        assert!(!feed(&mut r, RETIRE_WINDOW, 1, &mut pos));
        assert!(!r.is_retired());
        // The retiring cycle is the last one of the second failing window.
        assert!(!feed(&mut r, RETIRE_WINDOW - 1, 1, &mut pos));
        assert!(r.observe(1, 2, pos + 1));
        assert_eq!(r.retired_at(), Some(pos + 1));
        // Sticky: later cycles never report another retirement.
        assert!(!feed(&mut r, RETIRE_WINDOW * 4, 1, &mut pos));
    }

    #[test]
    fn early_verdict_retires_a_far_drafter_after_its_first_cycles_only() {
        // 1 token per cycle against 1.5: 8 tokens < 12 - 2 at cycle 8.
        let mut r = SpecRetire::new(Some(DEFAULT_BREAK_EVEN));
        let mut pos = 0;
        assert!(!feed(&mut r, EARLY_CYCLES - 1, 1, &mut pos));
        assert!(r.observe(1, 2, pos + 1));
        // At break-even (alternating 1 and 2, 12 tokens for a cost of 12): no
        // early verdict; the window verdicts judge it.
        let mut near = SpecRetire::new(Some(DEFAULT_BREAK_EVEN));
        for i in 0..EARLY_CYCLES {
            assert!(!near.observe(1 + i % 2, 2, 10 + i));
        }
        // Only the request's first window has an early verdict.
        let mut later = SpecRetire::new(Some(DEFAULT_BREAK_EVEN));
        let mut pos = 0;
        assert!(!feed(&mut later, RETIRE_WINDOW, 3, &mut pos));
        assert!(!feed(&mut later, EARLY_CYCLES, 1, &mut pos));
        assert!(!later.is_retired());
    }

    #[test]
    fn mean_at_break_even_keeps_speculating() {
        // 1.5 tokens per cycle: alternating 1 and 2 is exactly break-even.
        let mut r = SpecRetire::new(Some(DEFAULT_BREAK_EVEN));
        for i in 0..RETIRE_WINDOW * 8 {
            assert!(!r.observe(1 + i % 2, 2, 100 + i));
        }
    }

    #[test]
    fn context_slope_retires_at_long_context_only() {
        // Same drafter (3 tokens/cycle) against base 1.5 + 0.1/1K: break-even
        // is 1.7 at 2K and 4.7 at 32K.
        let line = BreakEven {
            base: 1.5,
            per_1k_ctx: 0.1,
            per_row: 0.0,
        };
        let mut short = SpecRetire::new(Some(line));
        let mut pos = 2048;
        assert!(!feed(&mut short, RETIRE_WINDOW * 4, 3, &mut pos));
        let mut long = SpecRetire::new(Some(line));
        let mut pos = 32768;
        assert!(feed(&mut long, RETIRE_WINDOW * 2, 3, &mut pos));
    }

    #[test]
    fn verified_rows_price_each_cycle() {
        // 1.2 + 0.1 per verified row: a 2-row cycle breaks even at 1.4 tokens,
        // a 9-row cycle at 2.1. 2 tokens per cycle pays for 2-row cycles only.
        let line = BreakEven {
            base: 1.2,
            per_1k_ctx: 0.0,
            per_row: 0.1,
        };
        let mut narrow = SpecRetire::new(Some(line));
        let mut pos = 0;
        assert!(!feed_rows(&mut narrow, RETIRE_WINDOW * 4, 2, 2, &mut pos));
        let mut wide = SpecRetire::new(Some(line));
        let mut pos = 0;
        assert!(feed_rows(&mut wide, RETIRE_WINDOW * 2, 2, 9, &mut pos));
        // A window mixing both is judged on its summed cost: 8 x 1.4 + 8 x 2.1
        // = 28 break-even tokens against 32 committed, so it pays.
        let mut mixed = SpecRetire::new(Some(line));
        let mut pos = 0;
        for _ in 0..RETIRE_WINDOW * 2 {
            assert!(!feed_rows(&mut mixed, 1, 2, 2, &mut pos));
            assert!(!feed_rows(&mut mixed, 1, 2, 9, &mut pos));
        }
    }

    #[test]
    fn disabled_never_retires() {
        let mut r = SpecRetire::new(None);
        let mut pos = 0;
        assert!(!feed(&mut r, RETIRE_WINDOW * 8, 1, &mut pos));
    }

    #[test]
    fn parses_env_forms() {
        assert_eq!(parse_break_even(None), Ok(None));
        assert_eq!(parse_break_even(Some("off")), Ok(Some(None)));
        assert_eq!(parse_break_even(Some("0")), Ok(Some(None)));
        assert_eq!(
            parse_break_even(Some("2")),
            Ok(Some(Some(BreakEven {
                base: 2.0,
                per_1k_ctx: 0.0,
                per_row: 0.0
            })))
        );
        assert_eq!(
            parse_break_even(Some(" 1.6 , 0.004 ")),
            Ok(Some(Some(BreakEven {
                base: 1.6,
                per_1k_ctx: 0.004,
                per_row: 0.0
            })))
        );
        assert_eq!(
            parse_break_even(Some("1.2,0,0.15")),
            Ok(Some(Some(BreakEven {
                base: 1.2,
                per_1k_ctx: 0.0,
                per_row: 0.15
            })))
        );
        for bad in ["", "x", "1,2,3,4", "-1", "1,nan", "on", "1,0,-0.1"] {
            assert!(parse_break_even(Some(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn env_then_drafter_then_default() {
        let drafter = BreakEven {
            base: 1.9,
            per_1k_ctx: 0.0,
            per_row: 0.05,
        };
        assert_eq!(resolve_break_even(None, Some(drafter)), Some(drafter));
        assert_eq!(resolve_break_even(None, None), Some(DEFAULT_BREAK_EVEN));
        assert_eq!(resolve_break_even(Some(None), Some(drafter)), None);
        let env = BreakEven {
            base: 2.0,
            per_1k_ctx: 0.0,
            per_row: 0.0,
        };
        assert_eq!(
            resolve_break_even(Some(Some(env)), Some(drafter)),
            Some(env)
        );
    }
}
