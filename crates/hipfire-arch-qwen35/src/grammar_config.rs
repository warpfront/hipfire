// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen35 grammar env resolver — restores the two operator tunables that
//! became no-ops when `grammar.rs` was unified into `saddle-core` (B1).
//!
//! Before the merge `crates/hipfire-arch-qwen35/src/grammar.rs:128-153`
//! read the variables via `hipfire_config::developer_var`:
//!   - `HIPFIRE_QWEN35_NGRAM_MIN_REPEATS` clamped to `2..=32` (default 6)
//!   - `HIPFIRE_QWEN35_NGRAM_LEN_MIN` clamped to `1..=32` (default 3)
//!
//! The unified `saddle_core::grammar::json::Config` parameterized these but
//! no caller wired the env — `Matcher::new` always used `Config::default()`
//! (256, 6, 3, 32). This module restores the env handling on the caller side
//! so `saddle-core` stays free of `hipfire-config` (the layering contract).
//!
//! Validation reproduces the pre-merge logic exactly:
//!   `developer_var(name).ok().and_then(|s| s.parse().ok()).filter(|n| in_range).unwrap_or(default)`
//! Out-of-range and unparseable values fall back to the default, they are not
//! clamped.

use crate::grammar;

/// Resolve the qwen35 grammar `Config` from the two `HIPFIRE_QWEN35_*` env vars.
///
/// Reads `HIPFIRE_QWEN35_NGRAM_MIN_REPEATS` (`2..=32`, default 6) and
/// `HIPFIRE_QWEN35_NGRAM_LEN_MIN` (`1..=32`, default 3). When unset,
/// unparseable, or out of range the field falls back to `Config::default()`.
///
/// The resolver consults `hipfire_config::developer_var` (the process
/// snapshot). For an operator the snapshot is built from the process env at
/// startup, so ambient `HIPFIRE_QWEN35_*` values are honoured. Unit tests
/// exercise the pure parser via [`resolve_one_from`] without mutating the
/// process-global snapshot.
pub fn resolve_qwen35_grammar_config() -> grammar::Config {
    let defaults = grammar::Config::default();
    let ngram_min_repeats = resolve_one(
        "HIPFIRE_QWEN35_NGRAM_MIN_REPEATS",
        defaults.ngram_min_repeats,
        2,
        32,
    );
    let ngram_len_min = resolve_one(
        "HIPFIRE_QWEN35_NGRAM_LEN_MIN",
        defaults.ngram_len_min,
        1,
        32,
    );
    grammar::Config {
        ngram_window: defaults.ngram_window,
        ngram_min_repeats,
        ngram_len_min,
        ngram_len_max: defaults.ngram_len_max,
    }
}

/// Alias kept for ergonomic import from the daemon example.
pub fn resolve_grammar_config() -> grammar::Config {
    resolve_qwen35_grammar_config()
}

fn resolve_one(name: &str, default: usize, lo: usize, hi: usize) -> usize {
    // Pre-merge: developer_var(name).ok().and_then(|s| s.parse().ok()).filter(|n| n >= lo && n <= hi).unwrap_or(default)
    resolve_one_from(hipfire_config::developer_var(name).ok(), default, lo, hi)
}

/// Pure parse/clamp path shared by production and unit tests.
fn resolve_one_from(raw: Option<String>, default: usize, lo: usize, hi: usize) -> usize {
    raw.and_then(|s| s.parse().ok())
        .filter(|n| *n >= lo && *n <= hi)
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_when_unset() {
        // Pure path: absent raw → default fields from Config::default().
        let d = grammar::Config::default();
        assert_eq!(
            resolve_one_from(None, d.ngram_min_repeats, 2, 32),
            d.ngram_min_repeats
        );
        assert_eq!(
            resolve_one_from(None, d.ngram_len_min, 1, 32),
            d.ngram_len_min
        );
    }

    #[test]
    fn reads_min_repeats_from_value() {
        assert_eq!(resolve_one_from(Some("10".into()), 6, 2, 32), 10);
    }

    #[test]
    fn reads_len_min_from_value() {
        assert_eq!(resolve_one_from(Some("7".into()), 3, 1, 32), 7);
    }

    #[test]
    fn out_of_range_falls_back_to_default() {
        let d = grammar::Config::default();

        // Below lower bound
        assert_eq!(
            resolve_one_from(Some("1".into()), d.ngram_min_repeats, 2, 32),
            d.ngram_min_repeats,
            "1 is below 2..=32 for NGRAM_MIN_REPEATS, pre-merge would fall back to default 6"
        );
        // Above upper bound
        assert_eq!(
            resolve_one_from(Some("33".into()), d.ngram_min_repeats, 2, 32),
            d.ngram_min_repeats,
            "33 is above 2..=32, should fall back"
        );
        // Len below 1
        assert_eq!(
            resolve_one_from(Some("0".into()), d.ngram_len_min, 1, 32),
            d.ngram_len_min,
            "0 is below 1..=32 for NGRAM_LEN_MIN, should fall back to 3"
        );
        // Len above 32
        assert_eq!(
            resolve_one_from(Some("99".into()), d.ngram_len_min, 1, 32),
            d.ngram_len_min,
            "99 >32 should fall back"
        );
        // Unparseable
        assert_eq!(
            resolve_one_from(Some("abc".into()), d.ngram_len_min, 1, 32),
            d.ngram_len_min,
            "unparseable should fall back"
        );
        assert_eq!(
            resolve_one_from(Some("notanumber".into()), d.ngram_min_repeats, 2, 32),
            d.ngram_min_repeats,
            "unparseable min_repeats should fall back"
        );
    }
}
