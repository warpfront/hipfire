//! No-model S0 mechanism oracle for mutable expert-pointer table contents.
//!
//! This is an original lab harness. Fixture layout follows the existing
//! moe_few.rs tests by Kaden Schutt; production launchers and replay machinery
//! are used unchanged. It does NOT validate the sealed-MoE residency contract.
//! The two projections have independent fixed inputs: this is not an MoE
//! forward, a logits oracle, or evidence for a dynamic-cache speedup.
//!
//! Default is plan-only. Run with --execute [--hip-graph] on exact gfx1201.
//! The caller must hold the machine's shared GPU lock for the entire process.
#[path = "support/fn_expert_fixture.rs"]
mod fixture;
use fixture::*;
use rdna_compute::replay::{ReplayBackendRequest, ReplayController, ReplayQuiescence};
use rdna_compute::Gpu;
use serde_json::{json, Value};

#[allow(clippy::too_many_arguments)]
fn exercise(
    gpu: &mut Gpu,
    arena: &mut Arena,
    graph: &mut Option<hip_bridge::Graph>,
    exec: &mut Option<hip_bridge::GraphExec>,
    idle: &mut bool,
    want_graph: bool,
    report: &mut Value,
) -> R<()> {
    check(
        gpu.arch == "gfx1201",
        format!("S0 requires exact gfx1201, got {}", gpu.arch),
    )?;
    report["arch"] = json!(gpu.arch);
    report["device_id"] = json!(gpu.device_id);
    hip(gpu.ensure_capture_stream())?;
    let f = Fixture::build(gpu, arena, report)?;
    f.publish(gpu, arena, cases()[0])?;
    f.reset(gpu, arena)?;
    // Materialize every kernel before either capture mode is opened.
    gpu.replay = ReplayController::new(ReplayBackendRequest::Hip);
    *idle = false;
    f.launch(gpu)?;
    hip(gpu.hip.device_synchronize())?;
    *idle = true;
    let warm = f.read(gpu, arena)?;

    f.reset(gpu, arena)?;
    gpu.replay = ReplayController::new_manual_pm4();
    gpu.replay.set_forward_eligible(true);
    hip(gpu.replay.begin_capture())?;
    *idle = false;
    let before = launch_count();
    f.launch(gpu)?;
    hip(gpu.replay.finish_capture())?;
    hip(gpu.hip.device_synchronize())?;
    *idle = true;
    check(
        launch_count().saturating_sub(before) == 2,
        "capture launch census",
    )?;
    check(
        gpu.replay.recorded_launches().len() == 2,
        "recorded tape must contain two launches",
    )?;
    check(
        gpu.replay.recorded_launches()[0].kernel == GU_SYMBOL
            && gpu.replay.recorded_launches()[1].kernel == DN_SYMBOL,
        "recorded kernel order differs",
    )?;
    check(
        f.read(gpu, arena)?.differences(&warm) == [0; 3],
        "capture changed math",
    )?;
    let (dispatches, _, _) = gpu.replay.prepare_pm4_prefix(gpu.device_id as usize, 2)?;
    check(
        dispatches == 2 && gpu.replay.prepared_pm4_plan_ready(),
        "PM4 prepare census",
    )?;
    let tape = Tape {
        launches: gpu.replay.recorded_launches().to_vec(),
        prepared: gpu
            .replay
            .prepared_route_identity()
            .ok_or("PM4 identity absent")?,
        gu_table: f.gu_table.buf.as_ptr() as usize,
        dn_table: f.dn_table.buf.as_ptr() as usize,
    };
    let launches_json: Vec<Value> = tape
        .launches
        .iter()
        .map(|l| {
            json!({
                "kernel": l.kernel, "grid": l.grid, "block": l.block,
                "shared_mem": l.shared_mem, "kernarg_bytes": l.kernarg.len(),
                "kernarg_sha256": hash(&l.kernarg)
            })
        })
        .collect();
    report["capture"] = json!({
        "captures": 1, "dispatches": dispatches, "launches": launches_json,
        "prepared_identity": format!("{:?}", tape.prepared),
        "all_table_updates_resets_readbacks_outside_capture": true,
        "warm_vs_recorded_math_exact": true
    });

    if want_graph {
        f.reset(gpu, arena)?;
        let before = launch_count();
        // This helper is the existing stream-capture API used in moe_few.rs.
        // The resulting graph captures these two warmed kernels only.
        hip(gpu.begin_stream_capture())?;
        if let Err(error) = f.launch(gpu) {
            if let Ok(g) = gpu.end_stream_capture() {
                *graph = Some(g);
            }
            return Err(format!("hipGraph capture body: {error}"));
        }
        *graph = Some(hip(gpu.end_stream_capture())?);
        *exec = Some(hip(gpu.hip.graph_instantiate(graph.as_ref().unwrap()))?);
        check(
            launch_count().saturating_sub(before) == 2,
            "hipGraph captured launch census",
        )?;
        tape.verify(gpu, &f)?;
    }
    report["hip_graph_tested"] = json!(want_graph);
    report["hip_graph_scope"] =
        json!("Optional low-level mechanism only; no production Qwen4 hipGraph decode claim");
    let mut baseline: Option<Snapshot> = None;
    let mut pm4_total = 0usize;
    for case in cases() {
        check(
            *idle,
            "attempted table mutation before prior arm retirement",
        )?;
        let publication = f.publish(gpu, arena, case)?;
        let mut case_reference: Option<Snapshot> = None;
        let mut arms = vec![Arm::Eager, Arm::RecordedHip, Arm::Pm4];
        if want_graph {
            arms.push(Arm::HipGraph);
        }
        for which in arms {
            let (out, mut proof) = arm(gpu, arena, &f, &tape, exec.as_ref(), which, idle)?;
            let differences = baseline.as_ref().map_or([0; 3], |b| out.differences(b));
            if case.wrong_gu {
                check(
                    differences[0] > 0 && differences[1] > 0 && differences[2] == 0,
                    format!(
                        "{} {}: GU negative inconclusive: {differences:?}",
                        case.name,
                        which.name()
                    ),
                )?;
            } else if case.wrong_dn {
                check(
                    differences[0] == 0 && differences[1] == 0 && differences[2] > 0,
                    format!(
                        "{} {}: DN negative inconclusive: {differences:?}",
                        case.name,
                        which.name()
                    ),
                )?;
            } else {
                check(
                    differences == [0; 3],
                    format!(
                        "{} {}: residency changes bytes: {differences:?}",
                        case.name,
                        which.name()
                    ),
                )?;
            }
            if let Some(expected) = &case_reference {
                check(
                    out.differences(expected) == [0; 3],
                    format!(
                        "{} {}: disagrees with same-case eager",
                        case.name,
                        which.name()
                    ),
                )?;
            }
            proof["case"] = json!(case.name);
            proof["table_publication"] = publication.clone();
            proof["f32_word_differences_vs_initial_vram"] = json!(differences);
            proof["same_case_eager_exact"] = json!(true);
            proof["status"] = json!("PASS");
            report["checks"].as_array_mut().unwrap().push(proof);
            if matches!(which, Arm::Pm4) {
                pm4_total += 1;
            }
            if baseline.is_none() {
                baseline = Some(Snapshot {
                    gate: out.gate.clone(),
                    up: out.up.clone(),
                    down: out.down.clone(),
                });
            }
            if case_reference.is_none() {
                case_reference = Some(out);
            }
        }
    }
    tape.verify(gpu, &f)?;
    report["pm4_total_replays"] = json!(pm4_total);
    report["checks_expected"] = json!(cases().len() * if want_graph { 4 } else { 3 });
    report["all_checks_complete"] = json!(true);
    Ok(())
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mut report = json!({
        "schema": 1, "scope": "mechanism_only", "no_model": true,
        "production_contract_validated": false, "performance_claim": false,
        "checks": [], "all_checks_complete": false,
        "limitations": [
            "No sealed-MoE mutable-residency API or immutable_identity waiver is tested.",
            "No full-model logits, routing, recurrent state or generation is tested.",
            "No cache policy, copy overlap, throughput or latency benefit is measured.",
            "Independent fixed GU and DN inputs isolate table indirection; no activation chain.",
            "Caller must own GPU coordination locks; this binary never acquires or removes them."
        ]
    });
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("fn_expert_table_indirection --execute [--hip-graph]\nNo model; exact gfx1201; caller holds shared GPU lock. Default: PLAN_ONLY.");
        return;
    }
    if args.iter().any(|a| a != "--execute" && a != "--hip-graph") {
        report["status"] = json!("FAIL");
        report["error"] = json!("unknown argument");
        println!("{report}");
        std::process::exit(1);
    }
    if !args.iter().any(|a| a == "--execute") {
        report["status"] = json!("PLAN_ONLY");
        report["planned_cases"] = json!(cases().len());
        println!("{report}");
        return;
    }
    let want_graph = args.iter().any(|a| a == "--hip-graph");
    let mut gpu = match Gpu::init() {
        Ok(gpu) => gpu,
        Err(e) => {
            report["status"] = json!("FAIL");
            report["error"] = json!(format!("GPU init: {e}"));
            println!("{report}");
            std::process::exit(1);
        }
    };
    let mut arena = Arena::default();
    let mut graph = None;
    let mut exec = None;
    let mut idle = true;
    let outcome = exercise(
        &mut gpu,
        &mut arena,
        &mut graph,
        &mut exec,
        &mut idle,
        want_graph,
        &mut report,
    );
    if let Err(error) = &outcome {
        report["error"] = json!(error);
    }
    let mut cleanup = Vec::new();
    if idle {
        if let Err(failure) = gpu.replay.shutdown() {
            cleanup.push(format!(
                "retained shutdown: {}; {:?}",
                failure.error, failure.quiescence
            ));
            idle = failure.quiescence == ReplayQuiescence::Proven;
        }
    }
    if idle {
        if let Some(e) = exec.take() {
            if let Err(e) = gpu.hip.graph_exec_destroy(e) {
                cleanup.push(e.to_string());
            }
        }
        if let Some(g) = graph.take() {
            if let Err(e) = gpu.hip.graph_destroy(g) {
                cleanup.push(e.to_string());
            }
        }
        if let Err(e) = arena.release(&mut gpu) {
            cleanup.push(e);
        }
    } else {
        cleanup.push("GPU quiescence unknown: no explicit free/drop of captured owners or GPU/controller; process must terminate".into());
    }
    report["cleanup_errors"] = json!(cleanup);
    report["gpu_quiescence_proven_at_cleanup"] = json!(idle);
    let pass = outcome.is_ok() && cleanup.is_empty();
    report["status"] = json!(if pass { "PASS" } else { "FAIL" });
    println!("{report}");
    if !idle {
        // Graph/GraphExec are non-Drop handles; their explicit destroy calls
        // were skipped above. Retain the actual allocation/controller owners.
        std::mem::forget(arena);
        std::mem::forget(gpu);
    }
    if !pass {
        std::process::exit(1);
    }
}
