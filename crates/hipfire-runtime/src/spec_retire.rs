// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Björn Bösel

//! Per-request speculative-decode retirement: switch a request from spec
//! windows to plain AR tokens once its drafter cannot pay for the cycle.
//!
//! A spec cycle commits `emit` tokens for the price of one cycle; an AR step
//! commits one token for the price of one AR step. Speculation wins while
//! `emit / cycle_time > 1 / ar_time`, i.e. while the mean tokens per cycle
//! stays above the cost ratio `cycle_time / ar_time`. That ratio is the
//! break-even advance, modelled as a line in context length because the verify
//! attention scans `B x L` keys per cycle where AR scans `L`:
//!
//! ```text
//! break_even(L) = base + per_1k_ctx * L / 1024
//! ```
//!
//! The decision is a pure function of the committed-token counts and the
//! position, never of wall-clock time, so a greedy request retires at the same
//! cycle on every run (a live timing fit flips near-tie decisions between
//! runs, and verify width changes greedy numerics). Like the slot engine's MTP
//! retirement it judges whole windows and needs [`RETIRE_WINDOWS`] failing
//! windows in a row, so one low-acceptance stretch (a templated opening) does
//! not retire a drafter that pays for itself afterwards. Retirement is sticky
//! for the rest of the request.
//!
//! Default break-even (`HIPFIRE_SPEC_RETIRE` unset): [`DEFAULT_BREAK_EVEN`],
//! context-flat 1.5 tokens per cycle, just below the smallest cost ratio
//! measured on any box (gfx1151, Qwen3.8-27B MQ4 + MQ4 DFlash: 1.53 at 100
//! tokens, 1.61 at 2K, 1.73 at 32K; gfx1100 about 2 at short context). A
//! request below it loses to AR on every measured box. A box whose verify cost
//! grows with context takes its measured line through the env knob.

/// Spec cycles per judged window.
pub const RETIRE_WINDOW: usize = 16;
/// Consecutive failing windows before the request retires to AR.
pub const RETIRE_WINDOWS: usize = 2;

/// Break-even tokens per spec cycle at context length `L`:
/// `base + per_1k_ctx * L / 1024`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreakEven {
    pub base: f64,
    pub per_1k_ctx: f64,
}

impl BreakEven {
    pub fn at(&self, position: usize) -> f64 {
        self.base + self.per_1k_ctx * position as f64 / 1024.0
    }
}

pub const DEFAULT_BREAK_EVEN: BreakEven = BreakEven {
    base: 1.5,
    per_1k_ctx: 0.0,
};

/// Parse `HIPFIRE_SPEC_RETIRE`: unset = [`DEFAULT_BREAK_EVEN`]; `0` / `off` =
/// never retire; `<base>[,<per_1k_ctx>]` = that line. Anything else is an error
/// (a typo must not silently fall back to the default).
pub fn parse_break_even(value: Option<&str>) -> Result<Option<BreakEven>, String> {
    let Some(raw) = value.map(str::trim) else {
        return Ok(Some(DEFAULT_BREAK_EVEN));
    };
    if raw == "0" || raw.eq_ignore_ascii_case("off") {
        return Ok(None);
    }
    let bad = || format!("HIPFIRE_SPEC_RETIRE={raw:?}: expected off | <base>[,<per_1k_ctx>]");
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
    if parts.next().is_some() {
        return Err(bad());
    }
    Ok(Some(BreakEven { base, per_1k_ctx }))
}

/// One request's retirement state. Feed every spec cycle to [`Self::observe`].
pub struct SpecRetire {
    break_even: Option<BreakEven>,
    cycles: usize,
    emitted: usize,
    failed_windows: usize,
    retired_at: Option<usize>,
}

impl SpecRetire {
    /// `None` never retires.
    pub fn new(break_even: Option<BreakEven>) -> Self {
        Self {
            break_even,
            cycles: 0,
            emitted: 0,
            failed_windows: 0,
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

    /// Record one spec cycle that committed `emitted` tokens and left the
    /// request at absolute `position`. Returns true on the cycle that retires.
    pub fn observe(&mut self, emitted: usize, position: usize) -> bool {
        let Some(break_even) = self.break_even else {
            return false;
        };
        if self.is_retired() {
            return false;
        }
        self.cycles += 1;
        self.emitted += emitted;
        if self.cycles < RETIRE_WINDOW {
            return false;
        }
        let mean = self.emitted as f64 / self.cycles as f64;
        self.cycles = 0;
        self.emitted = 0;
        if mean < break_even.at(position) {
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
        let mut retired = false;
        for _ in 0..cycles {
            *position += emitted;
            retired |= r.observe(emitted, *position);
        }
        retired
    }

    #[test]
    fn retires_after_consecutive_failing_windows_only() {
        let mut r = SpecRetire::new(Some(DEFAULT_BREAK_EVEN));
        let mut pos = 0;
        assert!(!feed(&mut r, RETIRE_WINDOW, 1, &mut pos));
        // A passing window resets the streak.
        assert!(!feed(&mut r, RETIRE_WINDOW, 3, &mut pos));
        assert!(!feed(&mut r, RETIRE_WINDOW, 1, &mut pos));
        assert!(!r.is_retired());
        // The retiring cycle is the last one of the second failing window.
        assert!(!feed(&mut r, RETIRE_WINDOW - 1, 1, &mut pos));
        assert!(r.observe(1, pos + 1));
        assert_eq!(r.retired_at(), Some(pos + 1));
        // Sticky: later cycles never report another retirement.
        assert!(!feed(&mut r, RETIRE_WINDOW * 4, 1, &mut pos));
    }

    #[test]
    fn mean_at_break_even_keeps_speculating() {
        // 1.5 tokens per cycle: alternating 1 and 2 is exactly break-even.
        let mut r = SpecRetire::new(Some(DEFAULT_BREAK_EVEN));
        for i in 0..RETIRE_WINDOW * 8 {
            assert!(!r.observe(1 + i % 2, 100 + i));
        }
    }

    #[test]
    fn context_slope_retires_at_long_context_only() {
        // Same drafter (3 tokens/cycle) against base 1.5 + 0.1/1K: break-even
        // is 1.7 at 2K and 4.7 at 32K.
        let line = BreakEven {
            base: 1.5,
            per_1k_ctx: 0.1,
        };
        let mut short = SpecRetire::new(Some(line));
        let mut pos = 2048;
        assert!(!feed(&mut short, RETIRE_WINDOW * 4, 3, &mut pos));
        let mut long = SpecRetire::new(Some(line));
        let mut pos = 32768;
        assert!(feed(&mut long, RETIRE_WINDOW * 2, 3, &mut pos));
    }

    #[test]
    fn disabled_never_retires() {
        let mut r = SpecRetire::new(None);
        let mut pos = 0;
        assert!(!feed(&mut r, RETIRE_WINDOW * 8, 1, &mut pos));
    }

    #[test]
    fn parses_env_forms() {
        assert_eq!(parse_break_even(None), Ok(Some(DEFAULT_BREAK_EVEN)));
        assert_eq!(parse_break_even(Some("off")), Ok(None));
        assert_eq!(parse_break_even(Some("0")), Ok(None));
        assert_eq!(
            parse_break_even(Some("2")),
            Ok(Some(BreakEven {
                base: 2.0,
                per_1k_ctx: 0.0
            }))
        );
        assert_eq!(
            parse_break_even(Some(" 1.6 , 0.004 ")),
            Ok(Some(BreakEven {
                base: 1.6,
                per_1k_ctx: 0.004
            }))
        );
        for bad in ["", "x", "1,2,3", "-1", "1,nan", "on"] {
            assert!(parse_break_even(Some(bad)).is_err(), "{bad:?}");
        }
    }
}
