//! The single self-overwriting stderr line `hipfire run` shows while the
//! engine prefills a long prompt, before the first token can appear.

use std::time::{Duration, Instant};

/// Lines are redrawn at most this often.
pub(crate) const INTERVAL: Duration = Duration::from_millis(500);

/// `prefill 512/2048 tokens (25%), ETA 3s`, as drawn after a leading `\r`.
/// The ETA extrapolates the rate over `elapsed` (time since the request was
/// sent); `?` until a rate exists.
pub(crate) fn format_line(done: u64, total: u64, elapsed: Duration) -> String {
    let eta = match hipfire_client::prefill_eta_secs(done, total, elapsed) {
        Some(secs) => format!("{secs:.0}"),
        None => "?".to_owned(),
    };
    format!(
        "  prefill {done}/{total} tokens ({}%), ETA {eta}s",
        hipfire_client::prefill_percent(done, total)
    )
}

/// Throttled writer state for the prefill line. Each method returns the
/// exact text to write to stderr (`None` = write nothing), so the caller owns
/// the I/O and tests need no terminal.
pub(crate) struct PrefillLine {
    enabled: bool,
    started: Instant,
    last_draw: Option<Instant>,
    /// Width of the line currently on screen (0 = nothing to clear).
    drawn: usize,
}

impl PrefillLine {
    /// `enabled` is false off a terminal and under `--json`; then nothing is
    /// ever returned. `started` is when the request was sent.
    pub(crate) fn new(enabled: bool, started: Instant) -> Self {
        Self {
            enabled,
            started,
            last_draw: None,
            drawn: 0,
        }
    }

    /// The redraw for a progress report at `now`, if one is due.
    pub(crate) fn update(&mut self, done: u64, total: u64, now: Instant) -> Option<String> {
        if !self.enabled {
            return None;
        }
        if self
            .last_draw
            .is_some_and(|last| now.saturating_duration_since(last) < INTERVAL)
        {
            return None;
        }
        self.last_draw = Some(now);
        let line = format_line(done, total, now.saturating_duration_since(self.started));
        // A shorter redraw (e.g. `ETA 10s` -> `ETA 9s`) blanks the previous
        // line's tail; `drawn` keeps the widest width on screen for `clear`.
        let pad = self.drawn.saturating_sub(line.len());
        self.drawn = self.drawn.max(line.len());
        Some(format!("\r{line}{}", " ".repeat(pad)))
    }

    /// Erase the line (before the first token, or at the end/error).
    pub(crate) fn clear(&mut self) -> Option<String> {
        if self.drawn == 0 {
            return None;
        }
        let width = std::mem::take(&mut self.drawn);
        Some(format!("\r{}\r", " ".repeat(width)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_format_reports_tokens_percent_and_eta() {
        // 512 tokens in 2 s -> 256 tok/s -> 1536 left -> 6 s.
        assert_eq!(
            format_line(512, 2048, Duration::from_secs(2)),
            "  prefill 512/2048 tokens (25%), ETA 6s"
        );
        assert_eq!(
            format_line(0, 2048, Duration::from_secs(1)),
            "  prefill 0/2048 tokens (0%), ETA ?s"
        );
        assert_eq!(
            format_line(2047, 2048, Duration::from_secs(8)),
            "  prefill 2047/2048 tokens (99%), ETA 0s"
        );
    }

    #[test]
    fn redraw_is_throttled_and_cleared_to_the_drawn_width() {
        let t0 = Instant::now();
        let mut line = PrefillLine::new(true, t0);
        assert_eq!(line.clear(), None, "nothing drawn yet");
        let first = line
            .update(512, 2048, t0 + Duration::from_secs(2))
            .expect("first report draws at once");
        assert_eq!(first, "\r  prefill 512/2048 tokens (25%), ETA 6s");
        assert_eq!(
            line.update(1024, 2048, t0 + Duration::from_millis(2300)),
            None,
            "inside 500 ms"
        );
        let second = line
            .update(1024, 2048, t0 + Duration::from_secs(3))
            .expect("due after 500 ms");
        assert_eq!(second, "\r  prefill 1024/2048 tokens (50%), ETA 3s");
        let cleared = line.clear().expect("clears what is on screen");
        assert_eq!(cleared, format!("\r{}\r", " ".repeat(second.len() - 1)));
        assert_eq!(line.clear(), None, "cleared once");
    }

    #[test]
    fn disabled_line_never_writes() {
        let t0 = Instant::now();
        let mut line = PrefillLine::new(false, t0);
        assert_eq!(line.update(1, 2, t0 + Duration::from_secs(1)), None);
        assert_eq!(line.clear(), None);
    }
}
