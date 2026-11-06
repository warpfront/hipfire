// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `[moe-entry]` receipts: which routed-MoE kernel entry a launcher selected for a row count.
//!
//! With `HIPFIRE_CB_PHASES=1` (the developer diagnostic that also enables the serve
//! `[fn-lane-step]` line) every routed-MoE launcher prints ONE stderr line per distinct
//! `(stage, entry, rows)` of the process, so a card run attests which gate/up and down kernels
//! the 9..64-row verify steps really took (`rocprofv3` kernel names agree with the `entry`).
//! Unset, the receipt is one cached boolean check.

use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

static ON: LazyLock<bool> = LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_CB_PHASES").is_ok_and(|v| v.trim() == "1")
});

static SEEN: LazyLock<Mutex<HashSet<(&'static str, String, usize)>>> =
    LazyLock::new(Default::default);

/// The receipt line (pure).
pub(crate) fn line(stage: &str, entry: &str, rows: usize) -> String {
    format!("[moe-entry] stage={stage} rows={rows} entry={entry}")
}

/// Print the receipt once per `(stage, entry, rows)` when `HIPFIRE_CB_PHASES=1`.
pub(crate) fn moe_entry_receipt(stage: &'static str, entry: &str, rows: usize) {
    if !*ON {
        return;
    }
    let first =
        SEEN.lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert((stage, entry.to_owned(), rows));
    if first {
        eprintln!("{}", line(stage, entry, rows));
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_line_names_stage_rows_and_entry() {
        assert_eq!(
            super::line("down", "qwen4_moe_down_a8_grouped_nodz_gfx1201", 32),
            "[moe-entry] stage=down rows=32 entry=qwen4_moe_down_a8_grouped_nodz_gfx1201"
        );
    }

    /// The gate/up entry the wide verify pass takes at every row count of the 1..=64-row window
    /// (the `g32` family above 8 rows up to `WIDE_VERIFY_MAX_ROWS`; the incumbent `o4_r8_x4` /
    /// `o4_r2_x4` elsewhere), shipped and planar, with and without dead stores.
    #[test]
    fn the_g32_gate_up_entry_is_selected_per_row_count() {
        use crate::dispatch::Gpu;
        let max = crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS;
        assert_eq!(max, 64);
        for rows in 0..=max + 8 {
            let on = Gpu::moe32_gu_applies(1, rows, true);
            assert_eq!(on, (9..=max).contains(&rows), "rows {rows}");
            assert!(!Gpu::moe32_gu_applies(0, rows, true), "mode 0 is off");
            assert!(!Gpu::moe32_gu_applies(1, rows, false), "gfx1201 only");
            assert_eq!(Gpu::moe32_gu_applies(2, rows, true), on);
        }
        for planar in [false, true] {
            for rows in 9..=max {
                for nodz in [false, true] {
                    let (module, _src, entry) = Gpu::moe32_gu_entry(planar, rows, nodz);
                    assert!(module.ends_with("_g32"), "{module}");
                    let class = if rows > 16 { "_g32_r8" } else { "_g32_r4" };
                    let want_suffix = if nodz { "_nodz" } else { "" };
                    assert!(
                        entry.ends_with(&format!("{class}{want_suffix}")),
                        "rows {rows} planar {planar} nodz {nodz}: {entry}"
                    );
                    assert_eq!(entry.contains("g256v2p"), planar, "{entry}");
                }
            }
        }
    }
}
