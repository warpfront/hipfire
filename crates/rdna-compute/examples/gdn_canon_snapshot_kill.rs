// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S0 kill 4 (CHUNK-CANONICAL-DESIGN §6 S0, §4.1, §7 G2): snapshot an `Open`
//! stream at p = 1/25/63/64/65/511/512/513/521, overwrite every live persistent
//! buffer, restore, continue past the next seam, seal — and require the result
//! to be bit-identical to the unsplit reference.
//!
//! The kill is exercised against the typed hook
//! [`canon::stream::GdnStreamUnderTest`]:
//!
//! * **CPU model leg.** `RetainedReferenceStream` is the oracle; the correct
//!   `BoundedTileStream` must pass at every p (Open and Sealed snapshots, with
//!   `Peek` before and after the restore, both Q8Ef and plain Q8, off-grid
//!   `phase_start`); each fault mutant is run separately and **must be
//!   detected**: missing pending-raw carry, missing tile-entry state, missing
//!   EF (read and snapshot), stale-Q8 read at `pending==0`, duplicate conv,
//!   plain-Q8 call-index rounding, poison read of the carry capacity, plus the
//!   current call-end quantization and F32-without-raw-carry. A mutant that
//!   escapes makes the receipt `oracle_invalid`.
//! * **Plug-in point.** The "Plug-in slot" in `run()` constructs the streams that are *really*
//!   under test. Today: the current production gfx1201 scan route (no carry,
//!   call-end Q8Ef quantization) — it must FAIL off-grid p (fail-before) and
//!   passes only where p is an absolute seam (512). S2/S3/S4 add their
//!   `GdnStreamUnderTest` implementations here and the same matrix runs.
//!
//! Build: cargo build --release -p rdna-compute --features lab,deltanet --example gdn_canon_snapshot_kill
//! Run:   HIPFIRE_GRAPH=0 gdn_canon_snapshot_kill --json <receipt.json> [--no-gpu] [--hv 48|32]
//! Exit: 0 all pass | 1 production fails the kill (fail-before) | 2 oracle invalid | 3 environment.

#[path = "gdn_canon_common/mod.rs"]
mod canon;

use canon::gpu::*;
use canon::kills::*;
use canon::matrix::*;
use canon::model::*;
use canon::util::*;
use rdna_compute::gdn_stream::GdnStateQuant;
use rdna_compute::Gpu;
use serde_json::{json, Value};

fn cpu_case(quant: GdnStateQuant, init_position: u64, total: usize) -> CpuCase {
    let mut schedules = Vec::new();
    for &p in &KILL_POSITIONS {
        schedules.push(Schedule::snapshot_at(p, total, false, true));
    }
    for &p in &KILL_POSITIONS {
        schedules.push(Schedule::snapshot_at(p, total, true, true));
    }
    if init_position % 64 != 0 {
        // positions where the absolute tile / seam grid is hit exactly from an off-grid phase_start
        for p in [
            64 - (init_position % 64) as usize,
            512 - (init_position % 512) as usize,
        ] {
            schedules.push(Schedule::snapshot_at(p, total, false, true));
            schedules.push(Schedule::snapshot_at(p, total, true, true));
        }
    }
    CpuCase {
        quant,
        init_position,
        total_rows: total,
        schedules,
    }
}

fn run() -> R<i32> {
    let args = Args::from_env();
    let json_path = args.value("--json");
    let hv: usize = args
        .value("--hv")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(48);
    let total: usize = args
        .value("--total")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1100);

    // ── CPU model leg ────────────────────────────────────────────────────────
    let cases = vec![
        cpu_case(GdnStateQuant::Q8Ef, 0, total),
        cpu_case(GdnStateQuant::Q8Ef, 37, total),
        cpu_case(GdnStateQuant::Q8, 0, total),
    ];
    let faults: Vec<Fault> = Fault::ALL_FAULTS.to_vec();
    let t0 = std::time::Instant::now();
    let cpu = run_cpu_suite(&cases, &faults)?;
    eprintln!(
        "cpu leg: model_ok={} detected={:?} escaped={:?} ({:.1}s)",
        cpu.model_ok,
        cpu.detected,
        cpu.escaped,
        t0.elapsed().as_secs_f64()
    );
    let required = [
        "missing_pending_raw_carry",
        "missing_tile_entry_state",
        "missing_ef_read",
        "snapshot_drops_ef",
        "stale_q8_read",
        "duplicate_conv",
        "call_index_rounding",
        "call_end_quantization",
    ];
    let missing_required: Vec<&str> = required
        .iter()
        .copied()
        .filter(|r| !cpu.detected.iter().any(|d| d == r))
        .collect();
    let oracle_valid = cpu.model_ok && cpu.escaped.is_empty() && missing_required.is_empty();

    // ── plug-in streams (device) ─────────────────────────────────────────────
    let mut plug_json: Vec<Value> = Vec::new();
    let mut prod_fails = false;
    let mut arch = "none".to_string();
    if !args.has("--no-gpu") {
        let mut gpu = match Gpu::init() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("gdn_canon_snapshot_kill: GPU init failed: {e}");
                return Ok(EXIT_ENV);
            }
        };
        arch = gpu.arch.clone();
        if arch != "gfx1201" {
            eprintln!("gdn_canon_snapshot_kill: device leg needs gfx1201 (got {arch})");
            return Ok(EXIT_ENV);
        }
        if hipfire_config::developer_var("HIPFIRE_GDN_SCAN_OUT_EMU").is_ok() {
            return Err("unset HIPFIRE_GDN_SCAN_OUT_EMU".into());
        }
        let consts = GpuConsts::new(&mut gpu, hv)?;
        let init = gen_gpu_initial(hv, 0);
        let data = gen_gpu_rows(hv, total);
        // Open-snapshot schedules only: production has no Seal/Peek semantics. Peeks are
        // omitted because the reference here is the same stream run as one call.
        let scheds: Vec<Schedule> = KILL_POSITIONS
            .iter()
            .map(|&p| Schedule::snapshot_at(p, total, false, false))
            .collect();
        {
            // ── Plug-in slot ─────────────────────────────────────────────────
            // S2/S3/S4: construct your GdnStreamUnderTest here (plain Box<dyn>), then
            //   evaluate_candidate(&mut yours, &mut reference, &init, &data, &scheds,
            //                      RefMode::SameSchedule|SingleCall, &cfg)
            // The device production route below is the only plug-in today.
            let mut stream = GpuProductionStream::new(&mut gpu, &consts)?;
            let evals =
                evaluate_self_single(&mut stream, &init, &data, &scheds, &RunCfg { recipe_id: 0 })?;
            let mut cells = Vec::new();
            for (e, &p) in evals.iter().zip(KILL_POSITIONS.iter()) {
                if !e.diff.pass {
                    prod_fails = true;
                }
                let mut j = eval_json(e);
                j["p"] = json!(p);
                j["p_is_absolute_seam"] = json!(p % 512 == 0);
                cells.push(j);
            }
            plug_json.push(json!({
                "plug_in": "current production gfx1201 scan route (no carry; call-end Q8Ef quantization)",
                "hv": hv,
                "total_rows": total,
                "reference": "same device route, one call over all rows",
                "expected_before": "fail at every p except an absolute seam (512)",
                "cells": cells,
            }));
            stream.free()?;
        }
        consts.free(&mut gpu)?;
    }

    let verdict = if !oracle_valid {
        "oracle_invalid"
    } else if prod_fails {
        "fail"
    } else {
        "pass"
    };
    let receipt = envelope(
        "gdn_canon_snapshot_kill",
        &arch,
        verdict,
        json!({
            "kill": 4,
            "oracle_valid": oracle_valid,
            "required_faults_missing_detection": missing_required,
            "positions": KILL_POSITIONS,
            "seeds": {"model": MODEL_SEED, "data": DATA_SEED, "init": INIT_SEED},
            "hook": "canon::stream::GdnStreamUnderTest + canon::kills::evaluate_candidate",
            "cpu": cpu.json,
            "plug_ins": plug_json,
        }),
    );
    write_receipt(&json_path, &receipt)?;
    println!(
        "{}",
        serde_json::to_string(
            &json!({"oracle": "gdn_canon_snapshot_kill", "verdict": verdict, "oracle_valid": oracle_valid, "faults_escaped": cpu.escaped, "required_missing": missing_required})
        )?
    );
    Ok(match verdict {
        "pass" => EXIT_PASS,
        "fail" => EXIT_KILL_FAILS,
        _ => EXIT_ORACLE_INVALID,
    })
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("gdn_canon_snapshot_kill: {e}");
            std::process::exit(EXIT_ENV);
        }
    }
}
