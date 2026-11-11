// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Measured trade-offs quoted by the Settings preview and `hipfire config`.
//!
//! One static table, transcribed from the Settings spec §2.3. Every number and
//! source string below is a measurement somebody recorded; `None` means the
//! spec says "unknown" and the UI must say so rather than guess. Rows are per
//! GPU arch (and per PR #902 state where that changes the pair compared), and
//! [`evidence_for`] never answers with another arch's row.
//!
//! "Closer" is the relative drop in KLD against the full-precision reference
//! (lower KLD = output closer to the BF16 model). Speed is prefill throughput
//! in tok/s unless the metric says otherwise; "slower" is the time ratio
//! `speed_from / speed_to`.

use serde::Serialize;

use crate::effective::A8Prefill;

pub const PROMPT_PROCESSING: &str = "prompt-processing";
pub const FLASH_NEXT_EXPERTS: &str = "flash-next-experts";
pub const FLASH_NEXT_PLE: &str = "flash-next-ple";
pub const FLASH_NEXT_PLE_DISK: &str = "flash-next-ple-disk";
pub const SHORTER_THINKING: &str = "shorter-thinking";
pub const A8_TO_IU4: &str = "a8-to-iu4";

/// A named secondary measurement pair (`from` → `to`), e.g. code24 KLD.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Extra {
    pub metric: &'static str,
    pub from: f64,
    pub to: f64,
}

/// One row of the §2.3 table: the move from the "Fast" side to the
/// "Accurate" side of one preview line on one arch.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct EvidenceRow {
    /// Preview line id (the `PROMPT_PROCESSING`… constants).
    pub line_id: &'static str,
    /// Exact GPU arch the numbers were measured on; `None` = every arch.
    pub arch: Option<&'static str>,
    /// PR #902 states this row applies to; empty = any.
    pub a8: &'static [A8Prefill],
    /// KLD against the full-precision reference, before → after.
    pub kld_from: Option<f64>,
    pub kld_to: Option<f64>,
    /// What the KLD was measured on (dataset / reference).
    pub kld_metric: Option<&'static str>,
    /// Second KLD pair (code24), when the spec quotes one.
    pub kld_extra: Option<Extra>,
    /// Throughput, before → after.
    pub speed_from: Option<f64>,
    pub speed_to: Option<f64>,
    /// Unit and workload of `speed_*` (`"pp8192 tok/s"`).
    pub speed_metric: Option<&'static str>,
    /// Second throughput pair, when the spec quotes one.
    pub speed_extra: Option<Extra>,
    /// Sidecar sizes in GB (q4, q8) for the disk row.
    pub disk_gb: Option<(f64, f64)>,
    /// Decode-speed effect as the spec words it; `None` = unknown / not given.
    pub typing: Option<&'static str>,
    /// Qualifier the preview must show next to the numbers.
    pub caveat: Option<&'static str>,
    /// Where the numbers come from; `None` = no source.
    pub source: Option<&'static str>,
}

const NONE: EvidenceRow = EvidenceRow {
    line_id: "",
    arch: None,
    a8: &[],
    kld_from: None,
    kld_to: None,
    kld_metric: None,
    kld_extra: None,
    speed_from: None,
    speed_to: None,
    speed_metric: None,
    speed_extra: None,
    disk_gb: None,
    typing: None,
    caveat: None,
    source: None,
};

const A8_LANDED: &[A8Prefill] = &[A8Prefill::OptIn, A8Prefill::DefaultOn];

static ROWS: &[EvidenceRow] = &[
    // Prompt processing, A8 (#902), gfx1151.
    EvidenceRow {
        line_id: PROMPT_PROCESSING,
        arch: Some("gfx1151"),
        a8: A8_LANDED,
        kld_from: Some(0.069498),
        kld_to: Some(0.039374),
        kld_metric: Some("WT2 KLD"),
        kld_extra: Some(Extra {
            metric: "code24 KLD",
            from: 0.048487,
            to: 0.027297,
        }),
        speed_from: Some(974.2),
        speed_to: Some(603.8),
        speed_metric: Some("pp8192 tok/s"),
        speed_extra: Some(Extra {
            metric: "pp32000 tok/s",
            from: 777.0,
            to: 521.4,
        }),
        typing: Some("unchanged (prefill, batch ≥ 64)"),
        source: Some("PR #902 body (qwen3.8-27b.mq4-xts, HIP 7.2, 85 W socket)"),
        ..NONE
    },
    // Prompt processing, iu4 off (no #902), gfx1151.
    EvidenceRow {
        line_id: PROMPT_PROCESSING,
        arch: Some("gfx1151"),
        a8: &[A8Prefill::NotLanded],
        kld_from: Some(0.069498),
        kld_to: Some(0.040313),
        kld_metric: Some("WT2 KLD"),
        kld_extra: Some(Extra {
            metric: "code24 KLD",
            from: 0.048487,
            to: 0.027348,
        }),
        speed_from: Some(974.2),
        speed_to: Some(347.0),
        speed_metric: Some("pp8192 tok/s"),
        typing: Some("unchanged"),
        source: Some("PR #902 body"),
        ..NONE
    },
    // Prompt processing, native fp8, gfx1201.
    EvidenceRow {
        line_id: PROMPT_PROCESSING,
        arch: Some("gfx1201"),
        kld_from: Some(0.069571),
        kld_to: Some(0.041859),
        kld_metric: Some("WT2 KLD"),
        kld_extra: Some(Extra {
            metric: "code24 KLD",
            from: 0.052119,
            to: 0.029867,
        }),
        speed_from: Some(4641.0),
        speed_to: Some(3852.0),
        speed_metric: Some("pp8192 tok/s (≈)"),
        typing: Some("unchanged"),
        caveat: Some("cross-card: A4 on card C vs fp8 on card A, 300 W"),
        source: Some("AGENTS.md:639-642; CHANGELOG.md:1770"),
        ..NONE
    },
    // Prompt processing, iu4 off, gfx1100.
    EvidenceRow {
        line_id: PROMPT_PROCESSING,
        arch: Some("gfx1100"),
        kld_from: Some(0.076078),
        kld_to: Some(0.057307),
        kld_metric: Some("WT2 KLD"),
        speed_from: Some(1405.1),
        speed_to: Some(1093.9),
        speed_metric: Some("XTX pp8192 tok/s"),
        typing: Some("unchanged (48.92 vs 48.82 tok/s)"),
        caveat: Some(
            "old checkpoint qwen3.8-27b.mq4-xt; current mq4-xts: unknown (AGENTS.md:643-644 pins A4 only)",
        ),
        source: Some("CHANGELOG.md:1887"),
        ..NONE
    },
    // Flash-Next experts, gfx1100 (7900 XTX): symmetric IU4 requested by default.
    EvidenceRow {
        line_id: FLASH_NEXT_EXPERTS,
        arch: Some("gfx1100"),
        kld_from: Some(0.0905),
        kld_to: Some(0.0612),
        kld_metric: Some("BF16-ref WT2 KLD"),
        speed_from: Some(955.5),
        speed_to: Some(943.9),
        speed_metric: Some("XTX pp8192 tok/s"),
        typing: Some("unchanged (route is prefill chunks ≥ 512 rows)"),
        caveat: Some("both arms had expert placement pinned at 8 VRAM layers"),
        source: Some(
            "CHANGELOG.md:27 (Flash-Next on gfx1100 (7900 XTX): the symmetric-IU4 MoE prefill route is requested by default)",
        ),
        ..NONE
    },
    // Flash-Next experts, gfx1151.
    EvidenceRow {
        line_id: FLASH_NEXT_EXPERTS,
        arch: Some("gfx1151"),
        kld_from: Some(0.1027),
        kld_to: Some(0.0661),
        kld_metric: Some("BF16-ref KLD"),
        speed_from: Some(1928.5),
        speed_to: Some(1455.0),
        speed_metric: Some("pp8192 tok/s (≈)"),
        typing: Some("unchanged (route is prefill chunks ≥ 512 rows)"),
        source: Some("CHANGELOG.md:1033, 1281"),
        ..NONE
    },
    // Flash-Next experts, gfx1201: only Strix Halo was measured.
    EvidenceRow {
        line_id: FLASH_NEXT_EXPERTS,
        arch: Some("gfx1201"),
        typing: Some("unchanged"),
        caveat: Some("only Strix Halo measured"),
        ..NONE
    },
    // Flash-Next word table q4 → q8, gfx1151.
    EvidenceRow {
        line_id: FLASH_NEXT_PLE,
        arch: Some("gfx1151"),
        kld_from: Some(0.104865),
        kld_to: Some(0.103252),
        kld_metric: Some("WT2 KLD"),
        speed_from: Some(1828.0),
        speed_to: Some(1599.0),
        speed_metric: Some("Ciru 8K steady tok/s (≈)"),
        caveat: Some("arch not stated in the entry"),
        source: Some("CHANGELOG.md:651-652"),
        ..NONE
    },
    // Flash-Next word table, disk size (no arch).
    EvidenceRow {
        line_id: FLASH_NEXT_PLE_DISK,
        disk_gb: Some((29.12, 54.4)),
        source: Some("CHANGELOG.md:232"),
        ..NONE
    },
    // Shorter thinking, every arch.
    EvidenceRow {
        line_id: SHORTER_THINKING,
        caveat: Some("fewer generated tokens per answer [INFERENCE]"),
        ..NONE
    },
    // A8 → IU4 (Low-power, #902 default-on), gfx1151. The speed pair is the
    // A8 row's, read the other way ("1.61× less prefill time").
    EvidenceRow {
        line_id: A8_TO_IU4,
        arch: Some("gfx1151"),
        a8: &[A8Prefill::DefaultOn],
        kld_from: Some(0.0394),
        kld_to: Some(0.0695),
        kld_metric: Some("WT2 KLD"),
        speed_from: Some(603.8),
        speed_to: Some(974.2),
        speed_metric: Some("pp8192 tok/s"),
        typing: Some("unchanged"),
        source: Some("PR #902"),
        ..NONE
    },
];

/// Every row, in spec table order.
pub fn rows() -> &'static [EvidenceRow] {
    ROWS
}

/// The row for `line_id` on exactly `arch` (or an all-arch row) that applies
/// in the given #902 state. Never another arch's row.
pub fn evidence_for(line_id: &str, arch: &str, a8: A8Prefill) -> Option<&'static EvidenceRow> {
    ROWS.iter().find(|row| {
        row.line_id == line_id
            && row.arch.is_none_or(|row_arch| row_arch == arch)
            && (row.a8.is_empty() || row.a8.contains(&a8))
    })
}

/// Relative KLD drop `(from − to) / from`, as a fraction (0.43 = 43% closer).
/// Negative when the output moved away from the reference. `None` when either
/// KLD is unknown or `from` is not positive.
pub fn closer_pct(row: &EvidenceRow) -> Option<f64> {
    let (from, to) = (row.kld_from?, row.kld_to?);
    (from > 0.0).then(|| (from - to) / from)
}

/// Time ratio `speed_from / speed_to` (1.61 = takes 1.61× as long). Below 1 the
/// move is faster. `None` when either speed is unknown or not positive.
pub fn slower_x(row: &EvidenceRow) -> Option<f64> {
    let (from, to) = (row.speed_from?, row.speed_to?);
    (from > 0.0 && to > 0.0).then(|| from / to)
}

/// `x` to two significant figures, no trailing exponent: 43.34 → "43",
/// 1.538 → "1.5", 0.0723 → "0.072", 1234 → "1200".
pub fn two_sig(x: f64) -> String {
    if x == 0.0 || !x.is_finite() {
        return format!("{x}");
    }
    let mut digits = x.abs().log10().floor() as i32;
    let mut decimals = (1 - digits).max(0) as usize;
    let mut text = format!("{x:.decimals$}");
    // Rounding can carry into the next power of ten (9.96 → "10.0").
    let rounded: f64 = text.parse().unwrap_or(x);
    let carried = rounded.abs().log10().floor() as i32;
    if carried != digits {
        digits = carried;
        decimals = (1 - digits).max(0) as usize;
        text = format!("{rounded:.decimals$}");
    }
    if digits >= 2 {
        // Round to two significant figures in the integer part.
        let scale = 10f64.powi(digits - 1);
        text = format!("{:.0}", (x / scale).round() * scale);
    }
    text
}

/// "43% closer" (or "N% further" when the KLD rose); `None` when unknown.
pub fn format_closer(row: &EvidenceRow) -> Option<String> {
    let pct = closer_pct(row)? * 100.0;
    let word = if pct < 0.0 { "further" } else { "closer" };
    Some(format!("{}% {word}", two_sig(pct.abs())))
}

/// "1.6× slower" (or "1.6× faster" when the move takes less time, "same
/// speed" when equal); `None` when unknown.
pub fn format_slower(row: &EvidenceRow) -> Option<String> {
    let ratio = slower_x(row)?;
    Some(if ratio > 1.0 {
        format!("{}× slower", two_sig(ratio))
    } else if ratio < 1.0 {
        format!("{}× faster", two_sig(1.0 / ratio))
    } else {
        "same speed".to_owned()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(line: &str, arch: &str, a8: A8Prefill) -> &'static EvidenceRow {
        evidence_for(line, arch, a8).unwrap_or_else(|| panic!("{line} {arch} {a8:?}"))
    }

    #[test]
    fn formatter_matches_the_spec_copy() {
        let a8 = row(PROMPT_PROCESSING, "gfx1151", A8Prefill::OptIn);
        assert_eq!(format_closer(a8).as_deref(), Some("43% closer"));
        assert_eq!(format_slower(a8).as_deref(), Some("1.6× slower"));
        let iu4_off = row(PROMPT_PROCESSING, "gfx1151", A8Prefill::NotLanded);
        assert_eq!(format_closer(iu4_off).as_deref(), Some("42% closer"));
        assert_eq!(format_slower(iu4_off).as_deref(), Some("2.8× slower"));
        let fp8 = row(PROMPT_PROCESSING, "gfx1201", A8Prefill::NotLanded);
        assert_eq!(format_closer(fp8).as_deref(), Some("40% closer"));
        assert_eq!(format_slower(fp8).as_deref(), Some("1.2× slower"));
        let xtx = row(PROMPT_PROCESSING, "gfx1100", A8Prefill::NotLanded);
        assert_eq!(format_closer(xtx).as_deref(), Some("25% closer"));
        assert_eq!(format_slower(xtx).as_deref(), Some("1.3× slower"));
        let experts = row(FLASH_NEXT_EXPERTS, "gfx1151", A8Prefill::NotLanded);
        assert_eq!(format_closer(experts).as_deref(), Some("36% closer"));
        assert_eq!(format_slower(experts).as_deref(), Some("1.3× slower"));
        let ple = row(FLASH_NEXT_PLE, "gfx1151", A8Prefill::NotLanded);
        assert_eq!(format_closer(ple).as_deref(), Some("1.5% closer"));
        assert_eq!(format_slower(ple).as_deref(), Some("1.1× slower"));
    }

    #[test]
    fn helpers_return_the_raw_ratios() {
        let a8 = row(PROMPT_PROCESSING, "gfx1151", A8Prefill::OptIn);
        assert!((closer_pct(a8).unwrap() - 0.4334).abs() < 1e-3);
        assert!((slower_x(a8).unwrap() - 974.2 / 603.8).abs() < 1e-12);
    }

    #[test]
    fn a8_to_iu4_reads_worse_and_faster() {
        let r = row(A8_TO_IU4, "gfx1151", A8Prefill::DefaultOn);
        assert!(closer_pct(r).unwrap() < 0.0);
        assert_eq!(format_closer(r).as_deref(), Some("76% further"));
        assert_eq!(format_slower(r).as_deref(), Some("1.6× faster"));
        assert!(evidence_for(A8_TO_IU4, "gfx1151", A8Prefill::OptIn).is_none());
        assert!(evidence_for(A8_TO_IU4, "gfx1151", A8Prefill::NotLanded).is_none());
    }

    #[test]
    fn unknowns_stay_none() {
        let experts = row(FLASH_NEXT_EXPERTS, "gfx1201", A8Prefill::NotLanded);
        assert_eq!(closer_pct(experts), None);
        assert_eq!(slower_x(experts), None);
        assert_eq!(format_closer(experts), None);
        assert_eq!(format_slower(experts), None);
        assert_eq!(experts.source, None);
        let think = row(SHORTER_THINKING, "gfx1030", A8Prefill::OptIn);
        assert_eq!(closer_pct(think), None);
        assert_eq!(slower_x(think), None);
        assert_eq!(think.typing, None);
        assert_eq!(think.source, None);
        // Word-table typing is unknown in the spec.
        let ple = row(FLASH_NEXT_PLE, "gfx1151", A8Prefill::NotLanded);
        assert_eq!(ple.typing, None);
    }

    #[test]
    fn lookup_never_returns_another_archs_row() {
        for arch in ["gfx1030", "gfx1101", "gfx942", ""] {
            for a8 in [A8Prefill::NotLanded, A8Prefill::OptIn, A8Prefill::DefaultOn] {
                assert!(evidence_for(PROMPT_PROCESSING, arch, a8).is_none());
                assert!(evidence_for(FLASH_NEXT_EXPERTS, arch, a8).is_none());
                assert!(evidence_for(FLASH_NEXT_PLE, arch, a8).is_none());
            }
        }
        for row in rows() {
            for a8 in [A8Prefill::NotLanded, A8Prefill::OptIn, A8Prefill::DefaultOn] {
                if let Some(arch) = row.arch {
                    let found = evidence_for(row.line_id, arch, a8);
                    assert!(found.is_none_or(|f| f.arch == Some(arch)));
                }
            }
        }
        // gfx1151 prompt-processing picks the row for the A8 state.
        assert_eq!(
            row(PROMPT_PROCESSING, "gfx1151", A8Prefill::DefaultOn).kld_to,
            Some(0.039374)
        );
        assert_eq!(
            row(PROMPT_PROCESSING, "gfx1151", A8Prefill::NotLanded).kld_to,
            Some(0.040313)
        );
        // Shorter thinking applies to every arch.
        assert!(evidence_for(SHORTER_THINKING, "gfx1030", A8Prefill::OptIn).is_some());
    }

    #[test]
    fn rows_quote_the_spec_verbatim() {
        let a8 = row(PROMPT_PROCESSING, "gfx1151", A8Prefill::OptIn);
        assert_eq!(a8.kld_extra.unwrap().from, 0.048487);
        assert_eq!(a8.kld_extra.unwrap().to, 0.027297);
        assert_eq!(a8.speed_extra.unwrap().from, 777.0);
        assert_eq!(a8.speed_extra.unwrap().to, 521.4);
        assert_eq!(
            a8.source,
            Some("PR #902 body (qwen3.8-27b.mq4-xts, HIP 7.2, 85 W socket)")
        );
        let fp8 = row(PROMPT_PROCESSING, "gfx1201", A8Prefill::NotLanded);
        assert_eq!((fp8.kld_from, fp8.kld_to), (Some(0.069571), Some(0.041859)));
        assert_eq!(fp8.kld_extra.unwrap().from, 0.052119);
        assert_eq!(fp8.kld_extra.unwrap().to, 0.029867);
        assert!(fp8.caveat.unwrap().starts_with("cross-card"));
        let disk = row(FLASH_NEXT_PLE_DISK, "gfx1151", A8Prefill::NotLanded);
        assert_eq!(disk.disk_gb, Some((29.12, 54.4)));
        assert_eq!(disk.source, Some("CHANGELOG.md:232"));
    }

    #[test]
    fn two_significant_figures() {
        for (x, want) in [
            (43.34, "43"),
            (1.538, "1.5"),
            (1.6134, "1.6"),
            (0.0723, "0.072"),
            (9.96, "10"),
            (99.6, "100"),
            (1234.0, "1200"),
            (0.5, "0.50"),
        ] {
            assert_eq!(two_sig(x), want, "{x}");
        }
    }
}
