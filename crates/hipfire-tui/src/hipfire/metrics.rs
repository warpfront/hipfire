// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Live serve `/metrics` for the Dashboard: a Prometheus text parser, an
//! in-memory ring of recent scrapes, and the windowed rates and quantiles
//! derived from the oldest and newest scrape in it. History is never
//! persisted; it is dropped when serve goes away or restarts.

use std::{
    collections::{HashMap, VecDeque},
    time::Instant,
};

/// About 3 minutes of history at the worker's 1.5 s scrape cadence.
const RING_CAP: usize = 120;

const REQUESTS: &str = "hipfire_requests_total";
const FAILED: &str = "hipfire_requests_failed_total";
const REJECTED: &str = "hipfire_admission_rejected_total";
const PROMPT: &str = "hipfire_prompt_tokens_total";
const COMPLETION: &str = "hipfire_completion_tokens_total";
const CACHED: &str = "hipfire_cached_prompt_tokens_total";
const UPTIME: &str = "hipfire_uptime_seconds";
const TTFT: &str = "hipfire_ttft_milliseconds";
const TPOT: &str = "hipfire_time_per_output_token_milliseconds";

/// One parsed `/metrics` body.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scrape {
    /// Unlabeled `hipfire_*` series (counters, gauges, `_sum`, `_count`).
    pub values: HashMap<String, f64>,
    /// Histogram name -> `(le, cumulative count)`, sorted by `le`; `+Inf` last.
    pub buckets: HashMap<String, Vec<(f64, f64)>>,
}

/// Parse Prometheus text exposition. Keeps unlabeled `hipfire_*` samples and
/// `_bucket{le="…"}` rows; comments, other labeled series (`model_info`) and
/// foreign names are ignored.
pub fn parse(body: &str) -> Scrape {
    let mut scrape = Scrape::default();
    for line in body.lines() {
        let mut parts = line.split_whitespace();
        let (Some(series), Some(value)) = (parts.next(), parts.next()) else {
            continue;
        };
        if !series.starts_with("hipfire_") {
            continue;
        }
        let Ok(value) = value.parse::<f64>() else {
            continue;
        };
        let Some((name, labels)) = series.split_once('{') else {
            scrape.values.insert(series.to_string(), value);
            continue;
        };
        let le = labels
            .strip_prefix("le=\"")
            .and_then(|l| l.strip_suffix("\"}"))
            .and_then(|l| l.parse::<f64>().ok());
        if let (Some(hist), Some(le)) = (name.strip_suffix("_bucket"), le) {
            scrape
                .buckets
                .entry(hist.to_string())
                .or_default()
                .push((le, value));
        }
    }
    for rows in scrape.buckets.values_mut() {
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    }
    scrape
}

/// What the Dashboard's "Live metrics" card shows.
#[derive(Clone, Debug, PartialEq)]
pub enum MetricsView {
    /// Fewer than two scrapes since the ring was last cleared.
    Collecting,
    Live(LiveMetrics),
    /// Serve answered `/metrics` with an HTTP error (older serve: 404).
    Unavailable,
    /// Serve unreachable.
    Offline,
}

/// Rates and quantiles over the ring window (oldest to newest scrape).
#[derive(Clone, Debug, PartialEq)]
pub struct LiveMetrics {
    pub window_s: f64,
    pub req_s: f64,
    /// Generated tokens per second, from the completion-token counter.
    pub tok_s: f64,
    pub errors_per_min: f64,
    pub rejects_per_min: f64,
    /// `cached_prompt / prompt` token deltas; `None` without prompt tokens.
    pub cache_hit_pct: Option<f64>,
    /// `(p50, p95)` in ms; `None` without observations in the window.
    pub ttft_ms: Option<(f64, f64)>,
    pub tpot_ms: Option<(f64, f64)>,
    /// Mean speculative tau; `None` without spec requests in the window.
    pub tau: Option<f64>,
    /// Per-scrape-interval tok/s, oldest first (Sparkline input).
    pub tok_s_series: Vec<u64>,
    /// Per-scrape-interval req/s × 1000, oldest first (Sparkline input).
    pub req_s_series: Vec<u64>,
}

#[derive(Debug, Default)]
pub struct MetricsRing {
    samples: VecDeque<(Instant, Scrape)>,
}

impl MetricsRing {
    /// Fold one scrape attempt in and return what the card should show. A
    /// failed scrape (`Err` carries `Offline` or `Unavailable`) drops history,
    /// and so does a restart: uptime or any `_total` counter going backwards.
    pub fn record(&mut self, at: Instant, scrape: Result<Scrape, MetricsView>) -> MetricsView {
        let scrape = match scrape {
            Ok(scrape) => scrape,
            Err(view) => {
                self.samples.clear();
                return view;
            }
        };
        if self
            .samples
            .back()
            .is_some_and(|(_, last)| restarted(last, &scrape))
        {
            self.samples.clear();
        }
        if self.samples.len() == RING_CAP {
            self.samples.pop_front();
        }
        self.samples.push_back((at, scrape));
        self.live()
            .map_or(MetricsView::Collecting, MetricsView::Live)
    }

    fn live(&self) -> Option<LiveMetrics> {
        let (t0, a) = self.samples.front()?;
        let (t1, b) = self.samples.back()?;
        let dt = t1.duration_since(*t0).as_secs_f64();
        if dt <= 0.0 {
            return None;
        }
        let d = |k: &str| delta(a, b, k);
        let prompt = d(PROMPT);
        let tau_count = d("hipfire_spec_tau_count");
        Some(LiveMetrics {
            window_s: dt,
            req_s: d(REQUESTS) / dt,
            tok_s: d(COMPLETION) / dt,
            errors_per_min: d(FAILED) * 60.0 / dt,
            rejects_per_min: d(REJECTED) * 60.0 / dt,
            cache_hit_pct: (prompt > 0.0).then(|| 100.0 * d(CACHED) / prompt),
            ttft_ms: window_quantiles(a, b, TTFT),
            tpot_ms: window_quantiles(a, b, TPOT),
            tau: (tau_count > 0.0).then(|| d("hipfire_spec_tau_sum") / tau_count),
            tok_s_series: self.series(COMPLETION, 1.0),
            req_s_series: self.series(REQUESTS, 1000.0),
        })
    }

    fn series(&self, key: &str, scale: f64) -> Vec<u64> {
        self.samples
            .iter()
            .zip(self.samples.iter().skip(1))
            .map(|((ta, a), (tb, b))| {
                let dt = tb.duration_since(*ta).as_secs_f64();
                if dt > 0.0 {
                    (delta(a, b, key) / dt * scale).round().max(0.0) as u64
                } else {
                    0
                }
            })
            .collect()
    }
}

fn delta(a: &Scrape, b: &Scrape, key: &str) -> f64 {
    b.values.get(key).unwrap_or(&0.0) - a.values.get(key).unwrap_or(&0.0)
}

fn restarted(prev: &Scrape, next: &Scrape) -> bool {
    prev.values.iter().any(|(k, v)| {
        (k == UPTIME || k.ends_with("_total")) && next.values.get(k).is_some_and(|n| n < v)
    })
}

/// p50/p95 over the bucket increase between two scrapes of histogram `name`.
fn window_quantiles(a: &Scrape, b: &Scrape, name: &str) -> Option<(f64, f64)> {
    let old = a.buckets.get(name);
    let rows: Vec<(f64, f64)> = b
        .buckets
        .get(name)?
        .iter()
        .map(|&(le, n)| {
            let before = old
                .and_then(|rows| rows.iter().find(|r| r.0 == le))
                .map_or(0.0, |r| r.1);
            (le, n - before)
        })
        .collect();
    Some((
        histogram_quantile(0.5, &rows)?,
        histogram_quantile(0.95, &rows)?,
    ))
}

/// PromQL `histogram_quantile` over cumulative `(le, count)` rows ending in
/// `+Inf`: linear interpolation inside the bucket holding the rank, the lower
/// bound of the first bucket taken as 0, and a rank in `+Inf` reported as the
/// highest finite bound.
fn histogram_quantile(q: f64, rows: &[(f64, f64)]) -> Option<f64> {
    let total = rows.last()?.1;
    if total <= 0.0 {
        return None;
    }
    let rank = q * total;
    let i = rows.iter().position(|r| r.1 >= rank)?;
    let (le, count) = rows[i];
    let (lo, below) = if i == 0 { (0.0, 0.0) } else { rows[i - 1] };
    if le.is_infinite() {
        return (i > 0).then_some(lo);
    }
    Some(lo + (le - lo) * (rank - below) / (count - below))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const CAPTURE: &str = include_str!("testdata/metrics_qwen35_9b.txt");

    fn body(requests: u64, completion: u64, uptime: u64) -> String {
        format!("{REQUESTS} {requests}\n{COMPLETION} {completion}\n{UPTIME} {uptime}\n{FAILED} 0\n")
    }

    fn view_after(ring: &mut MetricsRing, at: Instant, text: &str) -> MetricsView {
        ring.record(at, Ok(parse(text)))
    }

    fn live(view: MetricsView) -> LiveMetrics {
        match view {
            MetricsView::Live(live) => live,
            other => panic!("expected Live, got {other:?}"),
        }
    }

    #[test]
    fn parses_captured_serve_body() {
        let s = parse(CAPTURE);
        assert_eq!(s.values[REQUESTS], 2.0);
        assert_eq!(s.values[COMPLETION], 400.0);
        assert_eq!(s.values[PROMPT], 32.0);
        assert_eq!(s.values["hipfire_context_capacity_tokens"], 8192.0);
        assert_eq!(
            s.values["hipfire_request_latency_milliseconds_sum"],
            8510.442
        );
        assert_eq!(s.values["hipfire_spec_tau_count"], 0.0);
        // Labeled non-bucket series and comments are ignored.
        assert!(!s
            .values
            .keys()
            .any(|k| k.contains('{') || k.starts_with('#')));
        assert!(!s.values.contains_key("hipfire_model_info"));
        let ttft = &s.buckets[TTFT];
        assert_eq!(ttft.len(), 13);
        assert_eq!(ttft[2], (50.0, 1.0));
        assert_eq!(*ttft.last().unwrap(), (f64::INFINITY, 2.0));
        assert_eq!(s.buckets[TPOT].len(), 12);
        assert_eq!(s.buckets.len(), 5);
    }

    #[test]
    fn rates_over_ring_window() {
        let t0 = Instant::now();
        let mut ring = MetricsRing::default();
        let first = "hipfire_requests_total 2\nhipfire_completion_tokens_total 400\n\
                     hipfire_prompt_tokens_total 32\nhipfire_cached_prompt_tokens_total 0\n\
                     hipfire_requests_failed_total 0\nhipfire_admission_rejected_total 1\n\
                     hipfire_spec_tau_sum 0\nhipfire_spec_tau_count 0\n";
        assert_eq!(view_after(&mut ring, t0, first), MetricsView::Collecting);
        view_after(&mut ring, t0 + Duration::from_secs(1), &body(2, 400, 1));
        let last = "hipfire_requests_total 5\nhipfire_completion_tokens_total 1000\n\
                    hipfire_prompt_tokens_total 132\nhipfire_cached_prompt_tokens_total 50\n\
                    hipfire_requests_failed_total 1\nhipfire_admission_rejected_total 1\n\
                    hipfire_spec_tau_sum 12\nhipfire_spec_tau_count 4\n";
        let m = live(view_after(&mut ring, t0 + Duration::from_secs(3), last));
        assert_eq!(m.window_s, 3.0);
        assert_eq!(m.req_s, 1.0);
        assert_eq!(m.tok_s, 200.0);
        assert_eq!(m.errors_per_min, 20.0);
        assert_eq!(m.rejects_per_min, 0.0);
        assert_eq!(m.cache_hit_pct, Some(50.0));
        assert_eq!(m.tau, Some(3.0));
        // Interval 0->1s: no change; 1->3s: 3 req, 600 tok over 2s.
        assert_eq!(m.tok_s_series, vec![0, 300]);
        assert_eq!(m.req_s_series, vec![0, 1500]);
        assert_eq!(m.ttft_ms, None, "no histogram in window");
    }

    #[test]
    fn quantiles_from_bucket_deltas() {
        let t0 = Instant::now();
        let mut ring = MetricsRing::default();
        let hist = |b10, b50, b100, inf| {
            format!(
                "{TTFT}_bucket{{le=\"10\"}} {b10}\n{TTFT}_bucket{{le=\"50\"}} {b50}\n\
                 {TTFT}_bucket{{le=\"100\"}} {b100}\n{TTFT}_bucket{{le=\"+Inf\"}} {inf}\n"
            )
        };
        // 2 old observations at <=50 ms must not count in the window.
        view_after(&mut ring, t0, &hist(0, 2, 2, 2));
        // Window delta: 4 at (10,50], 4 at (50,100], 2 above 100.
        let m = live(view_after(
            &mut ring,
            t0 + Duration::from_secs(2),
            &hist(0, 6, 10, 12),
        ));
        // p50: rank 5 in (50,100] holding 4..8 -> 50 + 50 * (5-4)/4.
        // p95: rank 9.5 falls in +Inf -> highest finite bound.
        assert_eq!(m.ttft_ms, Some((62.5, 100.0)));
        assert_eq!(m.tpot_ms, None);
        // No new observations -> no quantile.
        let m = live(view_after(
            &mut ring,
            t0 + Duration::from_secs(4),
            &hist(0, 6, 10, 12),
        ));
        assert_eq!(m.ttft_ms, Some((62.5, 100.0)), "window still from oldest");
    }

    #[test]
    fn histogram_quantile_edges() {
        let rows = [(10.0, 4.0), (20.0, 4.0), (f64::INFINITY, 4.0)];
        assert_eq!(
            histogram_quantile(0.5, &rows),
            Some(5.0),
            "first bucket from 0"
        );
        assert_eq!(histogram_quantile(0.5, &[(f64::INFINITY, 3.0)]), None);
        assert_eq!(
            histogram_quantile(0.5, &[(10.0, 0.0), (f64::INFINITY, 0.0)]),
            None
        );
    }

    #[test]
    fn counter_or_uptime_decrease_clears_ring() {
        let t0 = Instant::now();
        let s = |n| t0 + Duration::from_secs(n);
        let mut ring = MetricsRing::default();
        view_after(&mut ring, s(0), &body(5, 100, 10));
        live(view_after(&mut ring, s(1), &body(6, 150, 11)));
        // Counter went backwards: serve restarted, history must go.
        assert_eq!(
            view_after(&mut ring, s(2), &body(0, 0, 11)),
            MetricsView::Collecting
        );
        live(view_after(&mut ring, s(3), &body(1, 10, 12)));
        // Uptime alone going backwards is a restart too.
        assert_eq!(
            view_after(&mut ring, s(4), &body(1, 10, 2)),
            MetricsView::Collecting
        );
    }

    #[test]
    fn failed_scrape_clears_ring() {
        let t0 = Instant::now();
        let s = |n| t0 + Duration::from_secs(n);
        let mut ring = MetricsRing::default();
        view_after(&mut ring, s(0), &body(1, 10, 1));
        live(view_after(&mut ring, s(1), &body(2, 20, 2)));
        assert_eq!(
            ring.record(s(2), Err(MetricsView::Offline)),
            MetricsView::Offline
        );
        // Same counters as before going offline: still only one sample.
        assert_eq!(
            view_after(&mut ring, s(3), &body(2, 20, 3)),
            MetricsView::Collecting
        );
        assert_eq!(
            ring.record(s(4), Err(MetricsView::Unavailable)),
            MetricsView::Unavailable
        );
        assert_eq!(
            view_after(&mut ring, s(5), &body(2, 20, 5)),
            MetricsView::Collecting
        );
    }

    #[test]
    fn ring_keeps_last_cap_samples() {
        let t0 = Instant::now();
        let mut ring = MetricsRing::default();
        let mut view = MetricsView::Collecting;
        for i in 0..(RING_CAP as u64 + 10) {
            view = view_after(&mut ring, t0 + Duration::from_secs(i), &body(i, i, i));
        }
        assert_eq!(live(view).window_s, (RING_CAP - 1) as f64);
    }
}
