// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Suites built on [`super::matrix`]: the CPU model suite (correct bounded
//! replay + every fault mutant vs the retained-input reference) and the generic
//! plug-in evaluator S2/S3/S4 call with their real stream implementations.

use super::matrix::*;
use super::model::*;
use super::stream::*;
use super::util::*;
use rayon::prelude::*;
use rdna_compute::gdn_stream::GdnStateQuant;
use serde_json::{json, Value};

pub const MODEL_SEED: u64 = SEED_BASE | 0x11;
pub const DATA_SEED: u64 = SEED_BASE | 0x22;
pub const INIT_SEED: u64 = SEED_BASE | 0x33;

/// How the reference side of an [`evaluate_candidate`] comparison is run.
#[derive(Clone, Copy, Debug)]
pub enum RefMode {
    /// Reference executes the same schedule (valid for Peek/Seal/Snapshot steps;
    /// the retained reference is partition-invariant by construction).
    SameSchedule,
    /// Reference executes one append of all rows: the canonical definition for
    /// seal-free schedules (partition invariance is the claim under test).
    SingleCall,
}

#[derive(Clone, Debug)]
pub struct CandidateEval {
    pub schedule: String,
    pub candidate: String,
    pub diff: Diff,
}

/// **Plug-in hook.** Drive any [`GdnStreamUnderTest`] through `scheds` against
/// `reference` (any implementation; typically
/// [`RetainedReferenceStream`] or, for device streams, the same device stream
/// run as one call) and return one bitwise [`Diff`] per schedule.
pub fn evaluate_candidate(
    cand: &mut dyn GdnStreamUnderTest,
    reference: &mut dyn GdnStreamUnderTest,
    init: &InitialState,
    data: &RawRows,
    scheds: &[Schedule],
    ref_mode: RefMode,
    cfg: &RunCfg,
) -> R<Vec<CandidateEval>> {
    let g = cand.geometry();
    let check_meta = cand.reports_stream_meta();
    let mut out = Vec::new();
    for s in scheds {
        let rs = match ref_mode {
            RefMode::SameSchedule => s.clone(),
            RefMode::SingleCall => Schedule::single(s.total_rows()),
        };
        let r = run_schedule(reference, init, data, &rs, cfg)?;
        let c = run_schedule(cand, init, data, s, cfg)?;
        let diff = compare_runs(&g, &r, &c, check_meta, cfg.recipe_id != 0);
        out.push(CandidateEval {
            schedule: s.name.clone(),
            candidate: cand.label(),
            diff,
        });
    }
    Ok(out)
}

/// Like [`evaluate_candidate`] with [`RefMode::SingleCall`] when only one
/// stream object exists (device streams own the `&mut Gpu`): the same stream is
/// first run as one append of all rows (the canonical definition for
/// seal-free schedules), then under each schedule.
pub fn evaluate_self_single(
    stream: &mut dyn GdnStreamUnderTest,
    init: &InitialState,
    data: &RawRows,
    scheds: &[Schedule],
    cfg: &RunCfg,
) -> R<Vec<CandidateEval>> {
    let g = stream.geometry();
    let check_meta = stream.reports_stream_meta();
    let mut out = Vec::new();
    for s in scheds {
        let r = run_schedule(stream, init, data, &Schedule::single(s.total_rows()), cfg)?;
        let c = run_schedule(stream, init, data, s, cfg)?;
        out.push(CandidateEval {
            schedule: s.name.clone(),
            candidate: stream.label(),
            diff: compare_runs(&g, &r, &c, check_meta, cfg.recipe_id != 0),
        });
    }
    Ok(out)
}

fn crashed_diff(why: &str) -> Diff {
    Diff {
        pass: false,
        rows: 0,
        outputs_identical: false,
        first_diff_row: Some(0),
        out_maxabs: f64::NAN,
        out_nonfinite_mismatch: 0,
        final_state: StateDiff::default(),
        mid_seal_mismatches: 0,
        peek_mismatches: 0,
        meta_mismatches: vec![why.to_string()],
    }
}

pub fn eval_json(e: &CandidateEval) -> Value {
    json!({"schedule": e.schedule, "candidate": e.candidate, "diff": e.diff.json()})
}

// ───────────────────────────── CPU suite ─────────────────────────────────────

pub struct CpuCase {
    pub quant: GdnStateQuant,
    pub init_position: u64,
    pub total_rows: usize,
    pub schedules: Vec<Schedule>,
}

pub struct CpuSuiteResult {
    pub json: Value,
    /// Correct model matches the reference on every schedule and the reference
    /// is partition-invariant where checked.
    pub model_ok: bool,
    /// Faults that never produced a diff on any schedule of any case (oracle
    /// would be blind to them).
    pub escaped: Vec<String>,
    /// Faults observed (diff ≠ ∅) at least once.
    pub detected: Vec<String>,
}

struct ScheduleOutcome {
    schedule: String,
    ref_invariant: Option<bool>,
    model: Diff,
    faults: Vec<(Fault, Diff)>,
}

fn qname(q: GdnStateQuant) -> &'static str {
    match q {
        GdnStateQuant::Q8 => "q8",
        GdnStateQuant::Q8Ef => "q8ef",
        GdnStateQuant::F32 => "f32",
    }
}

pub fn run_cpu_suite(cases: &[CpuCase], faults: &[Fault]) -> R<CpuSuiteResult> {
    let mut case_json = Vec::new();
    let mut model_ok = true;
    let mut detections: std::collections::BTreeMap<&'static str, usize> = Default::default();
    for f in faults {
        detections.insert(f.name(), 0);
    }
    for case in cases {
        let g = cpu_geometry(case.quant);
        let consts = ModelConsts::new(&g, MODEL_SEED);
        let init = gen_initial(&g, INIT_SEED, case.init_position);
        let data = gen_rows(&g, DATA_SEED, case.total_rows);
        let cfg = RunCfg {
            recipe_id: model_plan(&g).recipe_id,
        };
        let applicable: Vec<Fault> = faults
            .iter()
            .copied()
            .filter(|f| f.applies(case.quant))
            .collect();

        let outcomes: Vec<Result<ScheduleOutcome, String>> = case
            .schedules
            .par_iter()
            .map(|s| -> Result<ScheduleOutcome, String> {
                (|| -> R<ScheduleOutcome> {
                    let mut reference = RetainedReferenceStream::new(g, consts.clone(), &init);
                    let partition_only = s.is_partition_only();
                    // Canonical run: same schedule (Seal/Peek/Snapshot steps change what
                    // is being asked), or the single call for pure partitions.
                    let (canon, ref_invariant) = if partition_only {
                        let single = run_schedule(
                            &mut reference,
                            &init,
                            &data,
                            &Schedule::single(s.total_rows()),
                            &cfg,
                        )?;
                        // Re-running the (slow) retained reference under the partition proves
                        // invariance; skip it for very fine partitions (cost O(calls * frame)).
                        let inv = if s.steps.len() <= 64 {
                            let under = run_schedule(&mut reference, &init, &data, s, &cfg)?;
                            Some(compare_runs(&g, &single, &under, true, true).pass)
                        } else {
                            None
                        };
                        (single, inv)
                    } else {
                        (run_schedule(&mut reference, &init, &data, s, &cfg)?, None)
                    };
                    let mut model_stream =
                        BoundedTileStream::new(g, consts.clone(), Fault::None, &init);
                    let m = run_schedule(&mut model_stream, &init, &data, s, &cfg)?;
                    let model = compare_runs(&g, &canon, &m, true, true);
                    let mut fs = Vec::new();
                    for &f in &applicable {
                        // A faulted stream may also crash (e.g. a poison-sized slice); that is a detection.
                        let attempt =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                let mut bad = BoundedTileStream::new(g, consts.clone(), f, &init);
                                run_schedule(&mut bad, &init, &data, s, &cfg)
                                    .map_err(|e| e.to_string())
                            }));
                        let d = match attempt {
                            Ok(Ok(r)) => compare_runs(&g, &canon, &r, true, true),
                            Ok(Err(e)) => crashed_diff(&format!("error: {e}")),
                            Err(_) => crashed_diff("panic"),
                        };
                        fs.push((f, d));
                    }
                    Ok(ScheduleOutcome {
                        schedule: s.name.clone(),
                        ref_invariant,
                        model,
                        faults: fs,
                    })
                })()
                .map_err(|e| e.to_string())
            })
            .collect();

        let mut sched_json = Vec::new();
        for o in outcomes {
            let o = o?;
            if !o.model.pass || o.ref_invariant == Some(false) {
                model_ok = false;
            }
            let mut fj = serde_json::Map::new();
            for (f, d) in &o.faults {
                if !d.numeric_pass() {
                    *detections.get_mut(f.name()).unwrap() += 1;
                }
                fj.insert(
                    f.name().into(),
                    json!({"detected": !d.numeric_pass(), "meta_detected": !d.meta_mismatches.is_empty(), "first_diff_row": d.first_diff_row,
                           "out_maxabs": d.out_maxabs,
                           "state": d.final_state.codes + d.final_state.ef + d.final_state.conv + d.final_state.scales,
                           "peek_mismatches": d.peek_mismatches,
                           "mid_seal_mismatches": d.mid_seal_mismatches,
                           "meta_mismatches": d.meta_mismatches.len()}),
                );
            }
            sched_json.push(json!({
                "schedule": o.schedule,
                "reference_partition_invariant": o.ref_invariant,
                "correct_model": o.model.json(),
                "faults": fj,
            }));
        }
        case_json.push(json!({
            "quant": qname(case.quant),
            "init_position": case.init_position,
            "total_rows": case.total_rows,
            "geometry": {"hk": g.hk, "hv": g.hv, "micro_rows": g.micro_rows, "commit_rows": g.commit_rows},
            "schedules": sched_json,
        }));
    }
    let escaped: Vec<String> = detections
        .iter()
        .filter(|(name, n)| **n == 0 && faults.iter().any(|f| f.name() == **name))
        .map(|(n, _)| n.to_string())
        .collect();
    // A fault that never applied to any case (e.g. Q8-only fault in a Q8Ef-only run) is
    // not an escape.
    let applied: std::collections::BTreeSet<&str> = cases
        .iter()
        .flat_map(|c| {
            faults
                .iter()
                .filter(move |f| f.applies(c.quant))
                .map(|f| f.name())
        })
        .collect();
    let escaped: Vec<String> = escaped
        .into_iter()
        .filter(|n| applied.contains(n.as_str()))
        .collect();
    let detected: Vec<String> = detections
        .iter()
        .filter(|(_, n)| **n > 0)
        .map(|(k, _)| k.to_string())
        .collect();
    let detections_json: serde_json::Map<String, Value> = detections
        .iter()
        .map(|(k, v)| (k.to_string(), json!(v)))
        .collect();
    Ok(CpuSuiteResult {
        json: json!({
            "model_ok": model_ok,
            "fault_detection_counts": detections_json,
            "faults_escaped": escaped,
            "cases": case_json,
        }),
        model_ok,
        escaped,
        detected,
    })
}
