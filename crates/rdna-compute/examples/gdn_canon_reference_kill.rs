// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S0 kill 3 (CHUNK-CANONICAL-DESIGN §6 S0): full-512 retained-input reference
//! versus bounded-tile replay, off-grid splits, nonzero incoming EF, final seal.
//!
//! Two legs, one receipt:
//!
//! * **CPU leg (no GPU).** `RetainedReferenceStream` (slow, retains the whole
//!   512-row frame, recomputes it each append) is the oracle. It must be
//!   partition-invariant, the correct `BoundedTileStream` (FP32 tile-entry +
//!   raw pre-conv carry of the incomplete microtile, replay only that tile,
//!   quantize at absolute seams and at `seal`) must be bit-identical to it on
//!   every schedule (fixed outer chunks 1/25/63/64/65/511/512/513/521, seeded
//!   random partitions, mid-stream `Seal`, off-grid `phase_start`), and every
//!   fault mutant (incl. current call-end quantization) must be *detected*,
//!   otherwise the oracle is blind and the run is `oracle_invalid`.
//! * **GPU leg (gfx1201 production scan).** The existing prep → KKT → scan
//!   route is run as one call over all rows (stride-512 canonical definition)
//!   and under the same partitions. Current production quantizes Q8Ef at every
//!   outer call end and restarts the 512 grid at call row 0, so off-grid splits
//!   must FAIL (fail-before); splits whose boundaries are absolute 512 seams
//!   are the positive control.
//!
//! Build: cargo build --release -p rdna-compute --features lab,deltanet --example gdn_canon_reference_kill
//! Run:   HIPFIRE_GRAPH=0 gdn_canon_reference_kill --json <receipt.json> [--no-gpu] [--hv 48|32]
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

const CHUNKS: [usize; 9] = [1, 25, 63, 64, 65, 511, 512, 513, 521];

fn cpu_schedules(total: usize, quick: bool) -> Vec<Schedule> {
    let mut v = vec![Schedule::single(total)];
    for c in CHUNKS {
        if quick && c == 1 {
            continue;
        }
        v.push(Schedule::fixed_chunk(total, c));
    }
    for seed in 0..4u64 {
        v.push(Schedule::random_partition(total, SEED_BASE | (0x40 + seed)));
    }
    // Mode fence in the middle of a phase: a genuine seal at an off-grid position, then
    // a new phase whose first tile / frame is clipped to the next absolute boundary.
    v.push(Schedule {
        name: format!("seal@200[{total}]"),
        steps: vec![Step::Append(200), Step::Seal, Step::Append(total - 200)],
    });
    v.push(Schedule {
        name: format!("seal@64+seal@511[{total}]"),
        steps: vec![
            Step::Append(64),
            Step::Seal,
            Step::Append(447),
            Step::Seal,
            Step::Append(total - 511),
        ],
    });
    v
}

fn run() -> R<i32> {
    let args = Args::from_env();
    let json_path = args.value("--json");
    let hv: usize = args
        .value("--hv")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(48);
    let total_cpu: usize = args
        .value("--total-cpu")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1100);
    let total_gpu: usize = args
        .value("--total-gpu")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1100);
    let quick = args.has("--quick");

    // ── CPU leg ──────────────────────────────────────────────────────────────
    let cases = vec![
        CpuCase {
            quant: GdnStateQuant::Q8Ef,
            init_position: 0,
            total_rows: total_cpu,
            schedules: cpu_schedules(total_cpu, quick),
        },
        CpuCase {
            quant: GdnStateQuant::Q8Ef,
            // off-grid phase_start (genuine decode -> prefill boundary)
            init_position: 37,
            total_rows: total_cpu,
            schedules: vec![
                Schedule::single(total_cpu),
                Schedule::fixed_chunk(total_cpu, 25),
                Schedule::fixed_chunk(total_cpu, 64),
                Schedule::fixed_chunk(total_cpu, 65),
                Schedule::fixed_chunk(total_cpu, 512),
                Schedule::random_partition(total_cpu, SEED_BASE | 0x50),
            ],
        },
        CpuCase {
            quant: GdnStateQuant::Q8,
            init_position: 0,
            total_rows: total_cpu,
            schedules: cpu_schedules(total_cpu, true),
        },
    ];
    // Faults observable by partition / seal schedules. The snapshot-only faults
    // (missing pending carry, missing tile entry, snapshot drops EF) are killed by
    // gdn_canon_snapshot_kill, whose schedules actually snapshot.
    let faults: Vec<Fault> = vec![
        Fault::CallEndQuant,
        Fault::NaiveEntryCarry,
        Fault::MissingEf,
        Fault::StaleQ8,
        Fault::DuplicateConv,
        Fault::CallIndexRounding,
        Fault::PendingCapacityRead,
    ];
    let t0 = std::time::Instant::now();
    let cpu = run_cpu_suite(&cases, &faults)?;
    let cpu_secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "cpu leg: model_ok={} detected={:?} escaped={:?} ({cpu_secs:.1}s)",
        cpu.model_ok, cpu.detected, cpu.escaped
    );
    let oracle_valid = cpu.model_ok
        && cpu.escaped.is_empty()
        && cpu.detected.iter().any(|f| f == "call_end_quantization");

    // ── GPU leg ──────────────────────────────────────────────────────────────
    let mut gpu_json = json!({"skipped": true});
    let mut gpu_fails = false;
    let mut arch = "none".to_string();
    if !args.has("--no-gpu") {
        let mut gpu = match Gpu::init() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("gdn_canon_reference_kill: GPU init failed: {e}");
                return Ok(EXIT_ENV);
            }
        };
        arch = gpu.arch.clone();
        if arch != "gfx1201" {
            eprintln!("gdn_canon_reference_kill: GPU leg needs gfx1201 (got {arch})");
            return Ok(EXIT_ENV);
        }
        if hipfire_config::developer_var("HIPFIRE_GDN_SCAN_OUT_EMU").is_ok() {
            return Err("unset HIPFIRE_GDN_SCAN_OUT_EMU".into());
        }
        let consts = GpuConsts::new(&mut gpu, hv)?;
        let init = gen_gpu_initial(hv, 0);
        let data = gen_gpu_rows(hv, total_gpu);
        let mut scheds: Vec<Schedule> = [25usize, 63, 64, 65, 511, 512, 513, 521, 1024]
            .iter()
            .map(|c| Schedule::fixed_chunk(total_gpu, *c))
            .collect();
        for seed in 0..3u64 {
            scheds.push(Schedule::random_partition(
                total_gpu,
                SEED_BASE | (0x60 + seed),
            ));
        }
        let mut stream = GpuProductionStream::new(&mut gpu, &consts)?;
        let evals =
            evaluate_self_single(&mut stream, &init, &data, &scheds, &RunCfg { recipe_id: 0 })?;
        let route_of_last = stream.routes.last().cloned();
        let mut cells: Vec<Value> = Vec::new();
        for e in &evals {
            // Boundaries that are all absolute 512 seams (or the end) are the positive
            // control: production's call-end quantization coincides with a seam.
            let seam_aligned =
                e.schedule.starts_with("chunk512[") || e.schedule.starts_with("chunk1024[");
            if !e.diff.pass && seam_aligned {
                // The control failing means the oracle itself is broken or production is
                // non-deterministic: not a clean fail-before.
                gpu_fails = true;
            }
            if !e.diff.pass {
                gpu_fails = true;
            }
            let mut j = eval_json(e);
            j["seam_aligned_positive_control"] = json!(seam_aligned);
            cells.push(j);
        }
        let control_ok = evals
            .iter()
            .filter(|e| e.schedule.starts_with("chunk512[") || e.schedule.starts_with("chunk1024["))
            .all(|e| e.diff.pass);
        let off_grid_failing = evals
            .iter()
            .filter(|e| {
                !(e.schedule.starts_with("chunk512[") || e.schedule.starts_with("chunk1024["))
                    && !e.schedule.starts_with("random")
            })
            .filter(|e| !e.diff.pass)
            .count();
        gpu_json = json!({
            "arch": arch,
            "hv": hv,
            "total_rows": total_gpu,
            "reference": "production route, one call over all rows (stride 512)",
            "candidate": "production route, same rows in outer calls (call-end Q8Ef quantization, grid restarts)",
            "seam_aligned_positive_control_passes": control_ok,
            "off_grid_fixed_chunk_schedules_failing": off_grid_failing,
            "last_route": route_of_last.map(|r| json!({"rows": r.rows, "kkt_batched": r.kkt_batched, "mseg": r.mseg, "segments": r.segments})),
            "cells": cells,
        });
        stream.free()?;
        consts.free(&mut gpu)?;
    }

    let verdict = if !oracle_valid {
        "oracle_invalid"
    } else if gpu_fails {
        "fail"
    } else {
        "pass"
    };
    let receipt = envelope(
        "gdn_canon_reference_kill",
        &arch,
        verdict,
        json!({
            "kill": 3,
            "oracle_valid": oracle_valid,
            "semantics": "absolute 512 seams; FP32 tile-entry + raw pre-conv carry; quantize at seams and seal only",
            "cpu_tile_arithmetic": "stand-in with tile-local gate cumulation (tile-origin dependent rounding), causal prefix property; see gdn_canon_common/model.rs",
            "seeds": {"model": MODEL_SEED, "data": DATA_SEED, "init": INIT_SEED},
            "cpu": cpu.json,
            "gpu": gpu_json,
        }),
    );
    write_receipt(&json_path, &receipt)?;
    println!(
        "{}",
        serde_json::to_string(
            &json!({"oracle": "gdn_canon_reference_kill", "verdict": verdict, "oracle_valid": oracle_valid, "faults_escaped": cpu.escaped})
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
            eprintln!("gdn_canon_reference_kill: {e}");
            std::process::exit(EXIT_ENV);
        }
    }
}
