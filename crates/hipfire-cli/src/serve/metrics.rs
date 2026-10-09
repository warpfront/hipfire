// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Prometheus exposition for the serve path.
//!
//! The daemon already emits `ttft_ms`, `prefill_tok_s`, `decode_tok_s` and
//! `latency_ms` on every `done` event. Until this module existed the gateway
//! kept exactly one of them — `ServeMeta::recent_tok_s`, a scalar overwritten by
//! each request — so per-request telemetry flowed through the process and was
//! discarded. `/stats` reported counters but no distribution, which is why the
//! periodic decode-stall burst fixed in `dacce7470` had to be diagnosed with a
//! bespoke harness: a p50 and a p99 would have shown it immediately.
//!
//! No new dependency. Prometheus text format is a few lines to emit, and the
//! histogram is fixed-bucket with atomic counters, so scraping never blocks a
//! request. Buckets are chosen for interactive LLM serving: single-digit
//! milliseconds is meaningless here, tens of seconds is a hang.

use std::sync::atomic::{AtomicU64, Ordering};

use super::{EngineState, ServeMeta};

/// Upper bounds in milliseconds for TTFT and request latency. `+Inf` is
/// implicit and emitted last.
const LATENCY_BUCKETS_MS: &[f64] = &[
    10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 30000.0, 60000.0,
];

/// Upper bounds in tokens per second for decode and prefill throughput.
const TOK_S_BUCKETS: &[f64] = &[
    1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0, 2500.0,
    5000.0,
];

/// Upper bounds in milliseconds for time per output token.
const TPOT_BUCKETS_MS: &[f64] = &[
    2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 200.0, 500.0,
];

/// Fixed-bucket histogram over f64 observations, lock-free.
#[derive(Debug)]
pub(crate) struct Histogram {
    bounds: &'static [f64],
    buckets: Box<[AtomicU64]>,
    inf: AtomicU64,
    sum_milli: AtomicU64,
}

impl Histogram {
    fn new(bounds: &'static [f64]) -> Self {
        Self {
            bounds,
            buckets: bounds.iter().map(|_| AtomicU64::new(0)).collect(),
            inf: AtomicU64::new(0),
            sum_milli: AtomicU64::new(0),
        }
    }

    pub(crate) fn observe(&self, value: f64) {
        if !value.is_finite() || value < 0.0 {
            return;
        }
        match self.bounds.iter().position(|bound| value <= *bound) {
            Some(i) => self.buckets[i].fetch_add(1, Ordering::Relaxed),
            None => self.inf.fetch_add(1, Ordering::Relaxed),
        };
        // Store the sum scaled by 1000 so it stays integral and lock-free.
        self.sum_milli
            .fetch_add((value * 1000.0) as u64, Ordering::Relaxed);
    }

    fn count(&self) -> u64 {
        self.buckets
            .iter()
            .map(|b| b.load(Ordering::Relaxed))
            .sum::<u64>()
            + self.inf.load(Ordering::Relaxed)
    }

    fn sum(&self) -> f64 {
        self.sum_milli.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Prometheus histogram: cumulative buckets, then `_sum` and `_count`.
    fn render(&self, out: &mut String, name: &str, help: &str) {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} histogram\n"));
        let mut cumulative = 0u64;
        for (bucket, bound) in self.buckets.iter().zip(self.bounds) {
            cumulative += bucket.load(Ordering::Relaxed);
            out.push_str(&format!("{name}_bucket{{le=\"{bound}\"}} {cumulative}\n"));
        }
        cumulative += self.inf.load(Ordering::Relaxed);
        out.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {cumulative}\n"));
        out.push_str(&format!("{name}_sum {}\n", self.sum()));
        out.push_str(&format!("{name}_count {}\n", self.count()));
    }
}

/// Prometheus summary without quantiles: `_sum` and `_count` only, from which
/// a scraper derives the mean over any window.
#[derive(Debug, Default)]
pub(crate) struct Summary {
    sum_milli: AtomicU64,
    count: AtomicU64,
}

impl Summary {
    fn observe(&self, value: f64) {
        if !value.is_finite() || value < 0.0 {
            return;
        }
        self.sum_milli
            .fetch_add((value * 1000.0) as u64, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    fn render(&self, out: &mut String, name: &str, help: &str) {
        let sum = self.sum_milli.load(Ordering::Relaxed) as f64 / 1000.0;
        let count = self.count.load(Ordering::Relaxed);
        out.push_str(&format!(
            "# HELP {name} {help}\n# TYPE {name} summary\n{name}_sum {sum}\n{name}_count {count}\n"
        ));
    }
}

/// Label value escaped per the text exposition format: `\`, `"` and newline.
fn escape_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// Everything the gateway can report without asking the daemon.
#[derive(Debug)]
pub(crate) struct Metrics {
    pub(crate) ttft_ms: Histogram,
    pub(crate) latency_ms: Histogram,
    pub(crate) tpot_ms: Histogram,
    pub(crate) decode_tok_s: Histogram,
    pub(crate) prefill_tok_s: Histogram,
    pub(crate) spec_tau: Summary,
    pub(crate) requests_total: AtomicU64,
    pub(crate) requests_failed: AtomicU64,
    pub(crate) admission_rejected: AtomicU64,
    pub(crate) prompt_tokens: AtomicU64,
    pub(crate) completion_tokens: AtomicU64,
    pub(crate) cached_prompt_tokens: AtomicU64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            ttft_ms: Histogram::new(LATENCY_BUCKETS_MS),
            latency_ms: Histogram::new(LATENCY_BUCKETS_MS),
            tpot_ms: Histogram::new(TPOT_BUCKETS_MS),
            decode_tok_s: Histogram::new(TOK_S_BUCKETS),
            prefill_tok_s: Histogram::new(TOK_S_BUCKETS),
            spec_tau: Summary::default(),
            requests_total: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
            admission_rejected: AtomicU64::new(0),
            prompt_tokens: AtomicU64::new(0),
            completion_tokens: AtomicU64::new(0),
            cached_prompt_tokens: AtomicU64::new(0),
        }
    }
}

impl Metrics {
    /// One successful request: count it and record its gateway-measured wall
    /// time. Image requests stop here; chat requests go through `observe_done`.
    pub(crate) fn observe_request(&self, latency_ms: f64) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.latency_ms.observe(latency_ms);
    }

    /// Fold one completed chat attempt: its `done` payload plus the timing the
    /// gateway measured for it.
    ///
    /// Every `done` field is optional: a stream aborted by the client, or a
    /// daemon build that predates a field, simply contributes nothing rather
    /// than a zero. A zero would drag every percentile toward the floor and
    /// make the histogram lie in the reassuring direction. For the same reason
    /// `ttft_ms` is `None` when the attempt produced no token at all.
    pub(crate) fn observe_done(
        &self,
        done: &serde_json::Value,
        ttft_ms: Option<f64>,
        latency_ms: f64,
    ) {
        // ttft and latency are deliberately NOT taken from `done` even when
        // present. They are measured at the gateway, so that every path reports
        // them on the same basis and the interval includes admission queueing --
        // the component an operator needs when latency rises.
        self.observe_request(latency_ms);
        if let Some(t) = ttft_ms {
            self.ttft_ms.observe(t);
        }
        // The `done` envelope is PATH-DEPENDENT. The rich builder
        // (`hipfire_generate::common::emit_qwen_dflash_done_terminal`) carries
        // prefill_tok_s, decode_tok_s, ttft_ms, tau and cycles; the simple decode
        // path emits only {type,id,tokens,tok_s}. So `prefill_tok_s` fills on the
        // qwen/DFlash routes and stays empty on a plain lfm2.5 decode -- verified
        // by scraping both.
        let f64_field = |key: &str| done.get(key).and_then(serde_json::Value::as_f64);
        let u64_field = |key: &str| done.get(key).and_then(serde_json::Value::as_u64);
        if let Some(v) = f64_field("decode_tok_s").or_else(|| f64_field("tok_s")) {
            self.decode_tok_s.observe(v);
        }
        if let Some(v) = f64_field("prefill_tok_s") {
            self.prefill_tok_s.observe(v);
        }
        if let Some(v) = f64_field("tau") {
            self.spec_tau.observe(v);
        }
        let cached = u64_field("cached_tokens");
        if let Some(v) = cached {
            self.cached_prompt_tokens.fetch_add(v, Ordering::Relaxed);
        }
        // Same rule as `completion_usage`: Qwen routes report prefill + cached
        // instead of `prompt_tokens`.
        let prompt = u64_field("prompt_tokens")
            .or_else(|| u64_field("prefill_tokens").map(|p| p + cached.unwrap_or(0)));
        if let Some(v) = prompt {
            self.prompt_tokens.fetch_add(v, Ordering::Relaxed);
        }
        let tokens = u64_field("tokens").or_else(|| u64_field("completion_tokens"));
        if let Some(n) = tokens {
            self.completion_tokens.fetch_add(n, Ordering::Relaxed);
        }
        // TPOT spans first token to end, so it needs a first token and at
        // least one token after it.
        if let (Some(ttft), Some(n @ 2..)) = (ttft_ms, tokens) {
            self.tpot_ms.observe((latency_ms - ttft) / (n - 1) as f64);
        }
    }

    pub(crate) fn record_failure(&self) {
        self.requests_failed.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_admission_rejected(&self) {
        self.admission_rejected.fetch_add(1, Ordering::Relaxed);
    }

    /// Prometheus text exposition (v0.0.4). Serve facts come from `ServeMeta`,
    /// never from the runtime lock (see `/health`).
    pub(crate) fn render(
        &self,
        meta: &ServeMeta,
        queue_depth: usize,
        queue_capacity: usize,
    ) -> String {
        let mut out = String::with_capacity(4096);

        let counter = |out: &mut String, name: &str, help: &str, v: u64| {
            out.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} counter\n{name} {v}\n"
            ));
        };
        let gauge = |out: &mut String, name: &str, help: &str, v: f64| {
            out.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {v}\n"
            ));
        };
        let flag = |on: bool| if on { 1.0 } else { 0.0 };

        counter(
            &mut out,
            "hipfire_requests_total",
            "Completed generation requests.",
            self.requests_total.load(Ordering::Relaxed),
        );
        counter(
            &mut out,
            "hipfire_requests_failed_total",
            "Chat and image requests answered with an error, or whose stream ended in an error event. Oversized or unparseable bodies and admission rejections are not counted here.",
            self.requests_failed.load(Ordering::Relaxed),
        );
        counter(
            &mut out,
            "hipfire_admission_rejected_total",
            "Requests refused by admission control (queue full, or queue wait timed out).",
            self.admission_rejected.load(Ordering::Relaxed),
        );
        counter(
            &mut out,
            "hipfire_prompt_tokens_total",
            "Prompt tokens of completed chat requests, as reported by the daemon.",
            self.prompt_tokens.load(Ordering::Relaxed),
        );
        counter(
            &mut out,
            "hipfire_completion_tokens_total",
            "Generated tokens of completed chat requests, as reported by the daemon.",
            self.completion_tokens.load(Ordering::Relaxed),
        );
        counter(
            &mut out,
            "hipfire_cached_prompt_tokens_total",
            "Prompt tokens served from the prefix cache, as reported by the daemon.",
            self.cached_prompt_tokens.load(Ordering::Relaxed),
        );
        counter(
            &mut out,
            "hipfire_retries_total",
            "Chat attempts retried after a transient daemon failure.",
            meta.retries_attempted,
        );
        counter(
            &mut out,
            "hipfire_retries_succeeded_total",
            "Chat requests that completed on a retry.",
            meta.retries_succeeded,
        );
        gauge(
            &mut out,
            "hipfire_queue_depth",
            "Requests currently in flight or queued.",
            queue_depth as f64,
        );
        gauge(
            &mut out,
            "hipfire_queue_capacity",
            "Admission control capacity.",
            queue_capacity as f64,
        );
        gauge(
            &mut out,
            "hipfire_uptime_seconds",
            "Seconds since the gateway started.",
            meta.started.elapsed().as_secs() as f64,
        );
        gauge(
            &mut out,
            "hipfire_engine_up",
            "1 while the daemon is up, 0 while it is restarting or unhealthy.",
            flag(meta.engine_state == EngineState::Up),
        );
        gauge(
            &mut out,
            "hipfire_model_loading",
            "1 while a model load is in progress, 0 otherwise.",
            flag(meta.loading_model.is_some()),
        );
        gauge(
            &mut out,
            "hipfire_context_capacity_tokens",
            "Context (KV capacity) the resident model was loaded with; 0 when none is resident.",
            meta.n_ctx as f64,
        );
        // No sample at all while no model is resident: the value is the label.
        out.push_str(
            "# HELP hipfire_model_info The resident model.\n# TYPE hipfire_model_info gauge\n",
        );
        if let Some(model) = &meta.current_model {
            out.push_str(&format!(
                "hipfire_model_info{{model=\"{}\"}} 1\n",
                escape_label(model)
            ));
        }

        self.ttft_ms.render(
            &mut out,
            "hipfire_ttft_milliseconds",
            "Time to first token, per request.",
        );
        self.latency_ms.render(
            &mut out,
            "hipfire_request_latency_milliseconds",
            "Wall time per request.",
        );
        self.tpot_ms.render(
            &mut out,
            "hipfire_time_per_output_token_milliseconds",
            "Time per output token after the first, per chat request.",
        );
        self.decode_tok_s.render(
            &mut out,
            "hipfire_decode_tokens_per_second",
            "Decode throughput per request.",
        );
        self.prefill_tok_s.render(
            &mut out,
            "hipfire_prefill_tokens_per_second",
            "Prefill throughput per request.",
        );
        self.spec_tau.render(
            &mut out,
            "hipfire_spec_tau",
            "Speculative decoding mean accepted length (tau), per request that reports one.",
        );

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Integral values become JSON integers, as the daemon emits token counts.
    fn done(pairs: &[(&str, f64)]) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        for (k, v) in pairs {
            let value = if v.fract() == 0.0 {
                serde_json::json!(*v as u64)
            } else {
                serde_json::json!(v)
            };
            m.insert((*k).to_string(), value);
        }
        serde_json::Value::Object(m)
    }

    #[test]
    fn histogram_buckets_are_cumulative_and_sum_is_exact() {
        let h = Histogram::new(LATENCY_BUCKETS_MS);
        for v in [5.0, 30.0, 300.0, 90_000.0] {
            h.observe(v);
        }
        let mut s = String::new();
        h.render(&mut s, "t", "help");
        // 5 -> le=10; 30 -> le=50; 300 -> le=500; 90000 -> +Inf
        assert!(s.contains("t_bucket{le=\"10\"} 1"), "{s}");
        assert!(s.contains("t_bucket{le=\"50\"} 2"), "{s}");
        assert!(s.contains("t_bucket{le=\"500\"} 3"), "{s}");
        assert!(s.contains("t_bucket{le=\"+Inf\"} 4"), "{s}");
        assert!(s.contains("t_count 4"), "{s}");
        assert!(s.contains("t_sum 90335"), "{s}");
    }

    #[test]
    fn tok_s_lands_in_tok_s_buckets() {
        // These used to share the 10..60000 ms latency buckets.
        let m = Metrics::default();
        for v in [0.5, 37.7, 75.0, 120.0, 6000.0] {
            m.observe_done(&done(&[("decode_tok_s", v)]), None, 1.0);
        }
        let mut s = String::new();
        m.decode_tok_s.render(&mut s, "d", "help");
        assert!(s.contains("d_bucket{le=\"1\"} 1\n"), "{s}");
        assert!(s.contains("d_bucket{le=\"30\"} 1\n"), "{s}");
        assert!(s.contains("d_bucket{le=\"50\"} 2\n"), "{s}");
        assert!(s.contains("d_bucket{le=\"75\"} 3\n"), "{s}");
        assert!(s.contains("d_bucket{le=\"150\"} 4\n"), "{s}");
        assert!(s.contains("d_bucket{le=\"5000\"} 4\n"), "{s}");
        assert!(s.contains("d_bucket{le=\"+Inf\"} 5\n"), "{s}");
        assert!(!s.contains("le=\"60000\""), "{s}");
    }

    #[test]
    fn absent_fields_do_not_observe_a_zero() {
        // A zero would pull every percentile toward the floor -- a histogram that
        // lies in the reassuring direction is worse than a missing one.
        let m = Metrics::default();
        m.observe_done(&done(&[("prefill_tok_s", 400.0)]), None, 120.0);
        assert_eq!(m.prefill_tok_s.count(), 1);
        assert_eq!(
            m.decode_tok_s.count(),
            0,
            "absent decode must not be recorded"
        );
        assert_eq!(m.requests_total.load(Ordering::Relaxed), 1);
        assert_eq!(m.ttft_ms.count(), 0, "no first token => no ttft sample");
        assert_eq!(m.latency_ms.count(), 1);
        assert_eq!(m.spec_tau.count.load(Ordering::Relaxed), 0);
        assert_eq!(m.tpot_ms.count(), 0);
        let out = m.render(&ServeMeta::new("t".to_owned()), 0, 1);
        for series in [
            "hipfire_prompt_tokens_total 0\n",
            "hipfire_completion_tokens_total 0\n",
            "hipfire_cached_prompt_tokens_total 0\n",
            "hipfire_spec_tau_count 0\n",
        ] {
            assert!(out.contains(series), "{series}: {out}");
        }
        // ttft/latency come from the gateway, never from the done payload.
        m.observe_done(
            &done(&[("ttft_ms", 1.0), ("latency_ms", 1.0)]),
            Some(30.0),
            200.0,
        );
        assert_eq!(m.ttft_ms.count(), 1);
        assert_eq!(m.ttft_ms.sum(), 30.0);
        assert_eq!(m.latency_ms.sum(), 320.0);
    }

    #[test]
    fn token_counters_and_tau_fold_present_fields() {
        let m = Metrics::default();
        m.observe_done(
            &done(&[
                ("prompt_tokens", 100.0),
                ("prefill_tokens", 36.0),
                ("cached_tokens", 64.0),
                ("tokens", 20.0),
                ("completion_tokens", 999.0),
                ("tau", 2.5),
            ]),
            None,
            1.0,
        );
        // `completion_tokens` is only the fallback for a missing `tokens`;
        // prefill + cached stands in for a missing `prompt_tokens`, as in usage.
        m.observe_done(
            &done(&[
                ("completion_tokens", 5.0),
                ("prefill_tokens", 7.0),
                ("cached_tokens", 3.0),
            ]),
            None,
            1.0,
        );
        assert_eq!(m.prompt_tokens.load(Ordering::Relaxed), 110);
        assert_eq!(m.cached_prompt_tokens.load(Ordering::Relaxed), 67);
        assert_eq!(m.completion_tokens.load(Ordering::Relaxed), 25);
        let mut s = String::new();
        m.spec_tau.render(&mut s, "tau", "help");
        assert!(
            s.contains("# TYPE tau summary\ntau_sum 2.5\ntau_count 1\n"),
            "{s}"
        );
    }

    #[test]
    fn tpot_excludes_first_token_and_skips_degenerate_requests() {
        let m = Metrics::default();
        // (1000 - 200) / (5 - 1) = 200 ms.
        m.observe_done(&done(&[("tokens", 5.0)]), Some(200.0), 1000.0);
        assert_eq!(m.tpot_ms.count(), 1);
        assert_eq!(m.tpot_ms.sum(), 200.0);
        // No first token, a single token, or no token count: no sample.
        m.observe_done(&done(&[("tokens", 5.0)]), None, 1000.0);
        m.observe_done(&done(&[("tokens", 1.0)]), Some(200.0), 1000.0);
        m.observe_done(&done(&[]), Some(200.0), 1000.0);
        assert_eq!(m.tpot_ms.count(), 1);
        // The fallback count feeds TPOT too.
        m.observe_done(&done(&[("completion_tokens", 3.0)]), Some(100.0), 300.0);
        assert_eq!(m.tpot_ms.count(), 2);
        assert_eq!(m.tpot_ms.sum(), 300.0);
    }

    #[test]
    fn decode_falls_back_to_tok_s() {
        let m = Metrics::default();
        m.observe_done(&done(&[("tok_s", 37.7)]), None, 1.0);
        assert_eq!(m.decode_tok_s.count(), 1);
    }

    #[test]
    fn negative_and_nan_are_ignored() {
        let h = Histogram::new(LATENCY_BUCKETS_MS);
        h.observe(-1.0);
        h.observe(f64::NAN);
        h.observe(f64::INFINITY);
        assert_eq!(h.count(), 0);
    }

    #[test]
    fn model_info_label_is_escaped() {
        assert_eq!(escape_label("plain"), "plain");
        assert_eq!(escape_label("a\\b\"c\nd"), "a\\\\b\\\"c\\nd");
        let meta = ServeMeta {
            current_model: Some("C:\\models\\\"q\".hfq".to_owned()),
            ..ServeMeta::new("t".to_owned())
        };
        let out = Metrics::default().render(&meta, 0, 1);
        assert!(
            out.contains("hipfire_model_info{model=\"C:\\\\models\\\\\\\"q\\\".hfq\"} 1\n"),
            "{out}"
        );
    }

    #[test]
    fn render_reports_serve_state() {
        let m = Metrics::default();
        let meta = ServeMeta {
            current_model: Some("qwen3.8-27b".to_owned()),
            loading_model: Some("qwen3.8-27b".to_owned()),
            engine_state: EngineState::Restarting,
            n_ctx: 32768,
            retries_attempted: 4,
            retries_succeeded: 3,
            ..ServeMeta::new("t".to_owned())
        };
        let out = m.render(&meta, 3, 16);
        for name in [
            "hipfire_requests_total",
            "hipfire_queue_depth",
            "hipfire_ttft_milliseconds",
            "hipfire_time_per_output_token_milliseconds",
            "hipfire_decode_tokens_per_second",
            "hipfire_spec_tau",
            "hipfire_model_info",
        ] {
            assert!(
                out.contains(&format!("# HELP {name} ")),
                "missing HELP {name}"
            );
            assert!(
                out.contains(&format!("# TYPE {name} ")),
                "missing TYPE {name}"
            );
        }
        for sample in [
            "hipfire_queue_depth 3\n",
            "hipfire_engine_up 0\n",
            "hipfire_model_loading 1\n",
            "hipfire_context_capacity_tokens 32768\n",
            "hipfire_retries_total 4\n",
            "hipfire_retries_succeeded_total 3\n",
            "hipfire_model_info{model=\"qwen3.8-27b\"} 1\n",
        ] {
            assert!(out.contains(sample), "{sample}: {out}");
        }
        // No resident model: the info series has no sample at all.
        let idle = m.render(&ServeMeta::new("t".to_owned()), 0, 16);
        assert!(!idle.contains("hipfire_model_info{"), "{idle}");
        assert!(idle.contains("hipfire_engine_up 1\n"), "{idle}");
    }
}
