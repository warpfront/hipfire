// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Published decode/prefill receipts and the fixed speed classes `hipfire fit`
//! reports from them.
//!
//! The table is `registry/receipts-v1.json`, compiled into every binary. It is
//! deliberately NOT part of the wire registry (`registry/v1.json` is
//! untouched), so old clients keep parsing the fetched registry unchanged.
//!
//! # What a receipt is
//!
//! One number the repository already publishes (the registry `desc` text, the
//! README, `docs/multi-gpu.md`, `docs/BENCHMARKS.md`, the changelog), tied to
//! exactly one registry tag, one GPU arch, one topology, one placement, one
//! metric (decode or prefill), one route and one token count, with the
//! measurement date and a verbatim `quote` from the cited source. Nothing is
//! derived, averaged, converted or interpolated, and a receipt is never applied
//! across tag, arch, topology or placement: [`for_model`] matches the tag and
//! the arch exactly and the caller must show the row's own topology and
//! placement beside the number. The unit tests re-read every cited source and
//! fail when a quote is not a verbatim substring of the cited line (or file)
//! or does not contain the published number.
//!
//! `placement` is `resident`, `host-experts` (Flash-Next experts mapped in host
//! memory) or `unstated` (the source does not say; recorded as such instead of
//! guessed). Rows from `docs/BENCHMARKS.md` are marked `historical`: that page
//! states that none of its rows is a current baseline.
//!
//! # Speed classes (a product decision, not a measurement)
//!
//! | metric  | fast        | interactive     | usable         | slow        |
//! |---------|-------------|-----------------|----------------|-------------|
//! | decode  | >= 100 tok/s | >= 30 tok/s    | >= 10 tok/s    | < 10 tok/s  |
//! | prefill | >= 2000 tok/s | >= 500 tok/s  | >= 100 tok/s   | < 100 tok/s |
//!
//! Each band's lower bound is inclusive; [`class_of`] maps a value to the
//! highest band it reaches and treats a non-finite value as `slow`.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::LazyLock;

const BUNDLED_RECEIPTS: &str = include_str!("../../../registry/receipts-v1.json");

/// Schema version of `registry/receipts-v1.json`.
pub const RECEIPTS_SCHEMA_VERSION: u32 = 1;

/// What a receipt measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Metric {
    /// Generation speed; `tokens` is the generated length (for example 128).
    Decode,
    /// Prompt-processing speed; `tokens` is the prompt length (for example 8192).
    Prefill,
}

/// One published measurement. See the module docs for the rules every row obeys.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    /// Current registry tag (a key of `registry/v1.json` `models`).
    pub tag: String,
    /// GPU architecture id, for example `gfx1201`.
    pub arch: String,
    /// `single`, `ep<N>`, `pp<N>` or `tp<N>`.
    pub topology: String,
    /// `resident`, `host-experts` or `unstated`.
    pub placement: String,
    pub metric: Metric,
    /// `ar`, `mtp`, `dflash`, `ngram` or `mtp+ngram`, as the source states.
    pub route: String,
    /// Generated tokens (decode) or prompt tokens (prefill).
    pub tokens: u32,
    /// KV depth when the source states one.
    pub context: Option<u32>,
    pub tok_s: f64,
    /// Measurement date `YYYY-MM-DD` (see the row `note` when it is a document
    /// date rather than a measurement date).
    pub date: String,
    /// Repo-relative path, optionally `:LINE` (1-based) of the quoted line.
    pub source: String,
    /// Verbatim substring of the cited source that contains the number.
    pub quote: String,
    /// True when the source declares its rows historical.
    pub historical: bool,
    pub note: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptFile {
    schema_version: u32,
    policy: String,
    receipts: Vec<Receipt>,
}

static BUNDLED: LazyLock<ReceiptFile> = LazyLock::new(|| {
    // The file is compiled in and covered by the tests below; a parse failure
    // here is a build defect, not a runtime condition.
    serde_json::from_str(BUNDLED_RECEIPTS).expect("bundled registry/receipts-v1.json must parse")
});

/// Every bundled receipt, in file order.
pub fn bundled() -> &'static [Receipt] {
    &BUNDLED.receipts
}

/// The one-sentence policy the bundled file states.
pub fn policy() -> &'static str {
    &BUNDLED.policy
}

/// Receipts whose tag and arch both equal the arguments exactly: decode rows
/// first, then prefill, newest date first within each metric (file order
/// breaks ties). Never returns another tag's or arch's row.
pub fn for_model(tag: &str, arch: &str) -> Vec<&'static Receipt> {
    let mut rows: Vec<&'static Receipt> = bundled()
        .iter()
        .filter(|row| row.tag == tag && row.arch == arch)
        .collect();
    rows.sort_by(|a, b| {
        metric_rank(a.metric)
            .cmp(&metric_rank(b.metric))
            .then_with(|| b.date.cmp(&a.date))
    });
    rows
}

fn metric_rank(metric: Metric) -> u8 {
    match metric {
        Metric::Decode => 0,
        Metric::Prefill => 1,
    }
}

/// Fixed speed band of a published tok/s value (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SpeedClass {
    Fast,
    Interactive,
    Usable,
    Slow,
}

impl fmt::Display for SpeedClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Fast => "fast",
            Self::Interactive => "interactive",
            Self::Usable => "usable",
            Self::Slow => "slow",
        })
    }
}

/// Lower bounds (inclusive) of fast, interactive and usable, in tok/s.
fn thresholds(metric: Metric) -> (f64, f64, f64) {
    match metric {
        Metric::Decode => (100.0, 30.0, 10.0),
        Metric::Prefill => (2000.0, 500.0, 100.0),
    }
}

/// The class of `tok_s` for `metric`. A NaN compares below every bound and is
/// classed `slow`.
pub fn class_of(metric: Metric, tok_s: f64) -> SpeedClass {
    let (fast, interactive, usable) = thresholds(metric);
    if tok_s >= fast {
        SpeedClass::Fast
    } else if tok_s >= interactive {
        SpeedClass::Interactive
    } else if tok_s >= usable {
        SpeedClass::Usable
    } else {
        SpeedClass::Slow
    }
}

/// Human-readable band of `class` for `metric`, for example `">= 100 tok/s"`.
pub fn class_band(metric: Metric, class: SpeedClass) -> &'static str {
    match (metric, class) {
        (Metric::Decode, SpeedClass::Fast) => ">= 100 tok/s",
        (Metric::Decode, SpeedClass::Interactive) => ">= 30 and < 100 tok/s",
        (Metric::Decode, SpeedClass::Usable) => ">= 10 and < 30 tok/s",
        (Metric::Decode, SpeedClass::Slow) => "< 10 tok/s",
        (Metric::Prefill, SpeedClass::Fast) => ">= 2000 tok/s",
        (Metric::Prefill, SpeedClass::Interactive) => ">= 500 and < 2000 tok/s",
        (Metric::Prefill, SpeedClass::Usable) => ">= 100 and < 500 tok/s",
        (Metric::Prefill, SpeedClass::Slow) => "< 100 tok/s",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::fs;

    fn repo_file(path: &str) -> String {
        let full = format!("{}/../../{path}", env!("CARGO_MANIFEST_DIR"));
        fs::read_to_string(&full)
            .unwrap_or_else(|error| panic!("cannot read cited source {full}: {error}"))
    }

    /// Splits `path[:LINE]` into the path and the optional 1-based line.
    fn split_source(source: &str) -> (&str, Option<usize>) {
        match source.rsplit_once(':') {
            Some((path, line)) if !line.is_empty() && line.bytes().all(|b| b.is_ascii_digit()) => {
                (path, Some(line.parse().expect("line number")))
            }
            _ => (source, None),
        }
    }

    /// `1234567.5` -> `1,234,567.5`.
    fn group_thousands(number: &str) -> String {
        let (int, frac) = match number.split_once('.') {
            Some((int, frac)) => (int, Some(frac)),
            None => (number, None),
        };
        let mut out = String::new();
        for (i, ch) in int.chars().enumerate() {
            if i > 0 && (int.len() - i) % 3 == 0 {
                out.push(',');
            }
            out.push(ch);
        }
        if let Some(frac) = frac {
            out.push('.');
            out.push_str(frac);
        }
        out
    }

    /// True when `text` prints `tok_s` the way a source would: plain or with
    /// thousands separators, one to three decimals, standing alone (not a
    /// suffix or prefix of a longer number).
    fn prints_number(text: &str, tok_s: f64) -> bool {
        let mut forms = vec![format!("{tok_s}")];
        for decimals in 1..=3 {
            forms.push(format!("{tok_s:.decimals$}"));
        }
        let grouped: Vec<String> = forms.iter().map(|form| group_thousands(form)).collect();
        forms.extend(grouped);
        let bytes = text.as_bytes();
        forms.iter().any(|form| {
            text.match_indices(form.as_str()).any(|(start, _)| {
                let end = start + form.len();
                let before_ok = start == 0 || !(bytes[start - 1].is_ascii_digit());
                let after_ok = end >= bytes.len()
                    || !(bytes[end].is_ascii_digit()
                        || (matches!(bytes[end], b'.' | b',')
                            && bytes.get(end + 1).is_some_and(u8::is_ascii_digit)));
                before_ok && after_ok
            })
        })
    }

    fn valid_date(date: &str) -> bool {
        let b = date.as_bytes();
        if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
            return false;
        }
        let digits = |range: std::ops::Range<usize>| b[range].iter().all(u8::is_ascii_digit);
        if !(digits(0..4) && digits(5..7) && digits(8..10)) {
            return false;
        }
        let month: u32 = date[5..7].parse().unwrap();
        let day: u32 = date[8..10].parse().unwrap();
        (1..=12).contains(&month) && (1..=31).contains(&day)
    }

    fn valid_arch(arch: &str) -> bool {
        arch.strip_prefix("gfx").is_some_and(|rest| {
            !rest.is_empty() && rest.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        })
    }

    fn valid_topology(topology: &str) -> bool {
        topology == "single"
            || ["ep", "pp", "tp"].iter().any(|prefix| {
                topology.strip_prefix(prefix).is_some_and(|n| {
                    !n.is_empty()
                        && n.bytes().all(|b| b.is_ascii_digit())
                        && n.parse::<u32>().is_ok_and(|n| n >= 2)
                })
            })
    }

    #[test]
    fn bundled_file_parses_with_schema_one() {
        assert_eq!(BUNDLED.schema_version, RECEIPTS_SCHEMA_VERSION);
        assert!(!policy().trim().is_empty());
        assert!(!bundled().is_empty());
    }

    #[test]
    fn rows_are_unique_per_cell() {
        let mut seen = HashSet::new();
        for row in bundled() {
            let key = (
                &row.tag,
                &row.arch,
                &row.topology,
                &row.placement,
                row.metric,
                &row.route,
                row.tokens,
                row.context,
            );
            assert!(seen.insert(key), "duplicate receipt cell: {key:?}");
        }
    }

    #[test]
    fn every_tag_is_a_current_registry_tag() {
        let registry = crate::bundled().expect("bundled registry parses");
        for row in bundled() {
            assert!(
                registry.models.contains_key(&row.tag),
                "receipt tag {:?} is not a current bundled-registry tag (source {})",
                row.tag,
                row.source
            );
        }
    }

    #[test]
    fn row_fields_are_well_formed() {
        for row in bundled() {
            let at = format!("{} {} {}", row.tag, row.arch, row.source);
            assert!(valid_arch(&row.arch), "bad arch: {at}");
            assert!(row.tok_s.is_finite() && row.tok_s > 0.0, "bad tok_s: {at}");
            assert!(row.tokens > 0, "bad token count: {at}");
            assert!(valid_date(&row.date), "bad date {:?}: {at}", row.date);
            assert!(
                valid_topology(&row.topology),
                "bad topology {:?}: {at}",
                row.topology
            );
            assert!(
                matches!(
                    row.placement.as_str(),
                    "resident" | "host-experts" | "unstated"
                ),
                "bad placement {:?}: {at}",
                row.placement
            );
            assert!(
                matches!(
                    row.route.as_str(),
                    "ar" | "mtp" | "dflash" | "ngram" | "mtp+ngram"
                ),
                "bad route {:?}: {at}",
                row.route
            );
            assert!(!row.quote.trim().is_empty(), "empty quote: {at}");
        }
    }

    #[test]
    fn benchmarks_md_rows_are_historical() {
        let mut seen = 0;
        for row in bundled() {
            let (path, _) = split_source(&row.source);
            if path == "docs/BENCHMARKS.md" {
                seen += 1;
                assert!(
                    row.historical,
                    "BENCHMARKS.md row must be historical: {} {}",
                    row.tag, row.source
                );
            }
        }
        // No minimum count: e7571f03e pointed BENCHMARKS.md at the README's
        // dated table and dropped the competitor-comparison rows, orphaning
        // the receipts that quoted them. Any receipt still sourced from the
        // file must be historical; whether one exists at all depends on the
        // doc, not the receipts.
        let _ = seen;
    }

    #[test]
    fn every_quote_is_verbatim_in_its_source_and_prints_the_number() {
        for row in bundled() {
            let (path, line) = split_source(&row.source);
            let text = repo_file(path);
            match line {
                Some(line) => {
                    let cited = text
                        .lines()
                        .nth(line - 1)
                        .unwrap_or_else(|| panic!("{} has no line {line}", row.source));
                    assert!(
                        cited.contains(&row.quote),
                        "quote is not a substring of {}:\n  quote: {}\n  line:  {cited}",
                        row.source,
                        row.quote
                    );
                }
                None => assert!(
                    text.contains(&row.quote),
                    "quote not found in {path}: {}",
                    row.quote
                ),
            }
            assert!(
                prints_number(&row.quote, row.tok_s),
                "quote does not contain {} as printed: {} ({})",
                row.tok_s,
                row.quote,
                row.source
            );
        }
    }

    #[test]
    fn number_matching_respects_digit_boundaries() {
        assert!(prints_number("203.9 gfx1201", 203.9));
        assert!(prints_number("prefill 3,943 tok/s", 3943.0));
        assert!(prints_number("| **353** | 168 |", 353.0));
        assert!(prints_number("about 15.0 tok/s.", 15.0));
        assert!(!prints_number("1203.9 gfx1201", 203.9));
        assert!(!prints_number("203.95 gfx1201", 203.9));
        assert!(!prints_number("13,943 tok/s", 3943.0));
    }

    #[test]
    fn for_model_matches_tag_and_arch_exactly() {
        let sample = &bundled()[0];
        let rows = for_model(&sample.tag, &sample.arch);
        assert!(!rows.is_empty());
        assert!(rows
            .iter()
            .all(|row| row.tag == sample.tag && row.arch == sample.arch));
        let expected = bundled()
            .iter()
            .filter(|row| row.tag == sample.tag && row.arch == sample.arch)
            .count();
        assert_eq!(rows.len(), expected);

        assert!(
            for_model("qwen3.6:35b-a3b", "gfx1201").is_empty(),
            "no prefix or alias matching"
        );
        assert!(
            for_model("qwen3.6:35b-a3b-mq4r", "gfx942").is_empty(),
            "no cross-arch matching"
        );
        assert!(
            for_model("qwen3.6:35b-a3b-mq4r", "GFX1201").is_empty(),
            "arch match is exact"
        );
        assert!(for_model("", "").is_empty());
    }

    #[test]
    fn for_model_orders_decode_then_prefill_newest_first() {
        let mut checked = false;
        for row in bundled() {
            let rows = for_model(&row.tag, &row.arch);
            let mut last_rank = 0;
            let mut last_date: Option<&str> = None;
            for entry in &rows {
                let rank = metric_rank(entry.metric);
                assert!(
                    rank >= last_rank,
                    "decode rows precede prefill rows for {}",
                    row.tag
                );
                if rank != last_rank {
                    last_date = None;
                }
                if let Some(previous) = last_date {
                    assert!(
                        previous >= entry.date.as_str(),
                        "newest date first within a metric"
                    );
                }
                last_rank = rank;
                last_date = Some(entry.date.as_str());
            }
            checked |= rows.iter().any(|r| r.metric == Metric::Decode)
                && rows.iter().any(|r| r.metric == Metric::Prefill);
        }
        assert!(checked, "expected a cell with both decode and prefill rows");
    }

    #[test]
    fn mq4r_registry_row_is_present() {
        let rows = for_model("qwen3.6:35b-a3b-mq4r", "gfx1201");
        let decode = rows
            .iter()
            .find(|row| row.metric == Metric::Decode && row.tokens == 128 && row.route == "ar")
            .expect("TG128 AR registry row");
        assert_eq!(decode.tok_s, 203.9);
        assert_eq!(decode.source.split(':').next(), Some("registry/v1.json"));
        assert!(!decode.historical);
    }

    #[test]
    fn decode_class_boundaries() {
        use SpeedClass::*;
        assert_eq!(class_of(Metric::Decode, 100.0), Fast);
        assert_eq!(class_of(Metric::Decode, 99.99), Interactive);
        assert_eq!(class_of(Metric::Decode, 30.0), Interactive);
        assert_eq!(class_of(Metric::Decode, 29.99), Usable);
        assert_eq!(class_of(Metric::Decode, 10.0), Usable);
        assert_eq!(class_of(Metric::Decode, 9.99), Slow);
        assert_eq!(class_of(Metric::Decode, 0.0), Slow);
        assert_eq!(class_of(Metric::Decode, f64::NAN), Slow);
    }

    #[test]
    fn prefill_class_boundaries() {
        use SpeedClass::*;
        assert_eq!(class_of(Metric::Prefill, 2000.0), Fast);
        assert_eq!(class_of(Metric::Prefill, 1999.9), Interactive);
        assert_eq!(class_of(Metric::Prefill, 500.0), Interactive);
        assert_eq!(class_of(Metric::Prefill, 499.9), Usable);
        assert_eq!(class_of(Metric::Prefill, 100.0), Usable);
        assert_eq!(class_of(Metric::Prefill, 99.9), Slow);
    }

    #[test]
    fn class_display_and_bands() {
        assert_eq!(SpeedClass::Fast.to_string(), "fast");
        assert_eq!(SpeedClass::Interactive.to_string(), "interactive");
        assert_eq!(SpeedClass::Usable.to_string(), "usable");
        assert_eq!(SpeedClass::Slow.to_string(), "slow");
        assert_eq!(class_band(Metric::Decode, SpeedClass::Fast), ">= 100 tok/s");
        assert_eq!(class_band(Metric::Decode, SpeedClass::Slow), "< 10 tok/s");
        assert_eq!(
            class_band(Metric::Prefill, SpeedClass::Fast),
            ">= 2000 tok/s"
        );
        assert_eq!(class_band(Metric::Prefill, SpeedClass::Slow), "< 100 tok/s");
        for metric in [Metric::Decode, Metric::Prefill] {
            for class in [
                SpeedClass::Fast,
                SpeedClass::Interactive,
                SpeedClass::Usable,
                SpeedClass::Slow,
            ] {
                assert!(class_band(metric, class).contains("tok/s"));
            }
        }
    }

    #[test]
    fn metric_serializes_lowercase() {
        assert_eq!(
            serde_json::to_string(&Metric::Decode).unwrap(),
            "\"decode\""
        );
        assert_eq!(
            serde_json::from_str::<Metric>("\"prefill\"").unwrap(),
            Metric::Prefill
        );
        assert_eq!(
            serde_json::to_string(&SpeedClass::Interactive).unwrap(),
            "\"interactive\""
        );
    }
}
