// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201 LOADcnt retires mixed BUFFER/GLOBAL load windows in issue order
//! (ROCm 10.0 `SIInsertWaitcnts.cpp`: both are LOAD_CNT's single
//! `VMEM_ACCESS` event). Golden: the oracle-passed QKVZA progressive
//! schedule (8 buffer header/packed loads then 8 global x loads, waits
//! 14/12/11/10/9/6/2/1/0) is accepted by the independent text replay and by
//! M7 with obligations {}; one wait loosened by a single slot is refused by both.
#![cfg(feature = "toolchain")]
use peacemaker_isa::Arch;

const QKVZA: &str = include_str!("fixtures/pm_decode/fused_qkvza_progressive_loadcnt.s");
const SYMBOL: &str = "fused_qkvza_mq4g256v2";

fn m7(text: &str, tag: &str) -> Result<serde_json::Value, String> {
    let elf = peacemaker_isa::native::assemble(text, Arch::Gfx1201)?;
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("load-fifo-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("{SYMBOL}.co"));
    std::fs::write(&path, &elf).map_err(|e| e.to_string())?;
    let result = peacemaker_isa::pm_check::m7(&path, "gfx1201", SYMBOL);
    std::fs::remove_dir_all(&dir).ok();
    result
}

#[test]
fn qkvza_progressive_mixed_window_is_m7_clean() {
    peacemaker_isa::ledger_replay::replay_waits(QKVZA, Arch::Gfx1201).expect("text replay");
    let summary = m7(QKVZA, "golden").expect("M7 obligations {}");
    eprintln!("M7 {summary}");
    assert_eq!(summary["obligations"], serde_json::json!({}));
    assert_eq!(summary["ambiguous_delays"], 0);
}

/// `s_wait_loadcnt 0xe` retires the two oldest loads (v[40:41], v[42:43]);
/// `0xf` retires only v[40:41], so the next instruction's read of v[42:43]
/// is a real hazard both checkers must still report.
#[test]
fn qkvza_insufficient_wait_is_refused() {
    let first = "\ts_wait_loadcnt 0xe\n";
    assert_eq!(QKVZA.matches(first).count(), 1);
    let loose = QKVZA.replacen(first, "\ts_wait_loadcnt 0xf\n", 1);
    assert!(peacemaker_isa::ledger_replay::replay_waits(&loose, Arch::Gfx1201).is_err());
    let err = m7(&loose, "loose").expect_err("M7 must refuse the loosened wait");
    assert!(err.contains("wait-raw-vmem-load"), "{err}");
}
