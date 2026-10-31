// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lane-suite glue of `sampled_mtp_distribution_hw` (`HIPFIRE_SAMPLED_MTP_EXECUTOR=lanes`).
//!
//! A child module of that test so it shares its config, identity and
//! comparison helpers; the host-only logic (parsers, arm set, merge key and
//! planner, per-lane splitting, route coverage) is [`lane_suite`], with its own
//! CPU tests. This module is the arm-process entry, the shard merger and the
//! parent verdict. Lane arms run in their own `HIPFIRE_SAMPLED_MTP_OUT`
//! directory; the parent never launches lane arms (it would start an unbounded
//! cohort), it only judges merged, config-verified whole-arm files.

use super::*;

/// Fewest trials of a lane distribution verdict, per arm and lane.
const LANE_MIN_TRIALS: usize = 2000;

/// Harness-control variables of the lane suite: the shared [`CONTROL_ENV`] plus
/// the fabric admission token, which names the job (a different value per
/// shard job) and changes nothing the arm computes, so shards run in different
/// fabric jobs share one identity.
pub(super) const LANE_CONTROL_ENV: [&str; 5] = [
    MODEL_ENV,
    OUT_ENV,
    ARM_ENV,
    SHARD_ENV,
    "HIPFIRE_FABRIC_ADMISSION",
];

/// Why a topology has no lane executor: only EP2 runs lane arms in this
/// build. Single-card and PP2 lane arms validate their whole configuration
/// and then refuse, writing no ids, so nothing can be merged or judged from a
/// run that did not happen.
fn lane_topology_unsupported(topology: lane_suite::Topology) -> String {
    format!(
        "the lane executor runs topology ep2 only ({} lane arms are not implemented in this build); this arm validated its configuration and ran no trials",
        topology.as_str()
    )
}

fn lane_state_format() -> Result<&'static str, String> {
    Ok(hw_identity::gdn_test_state_quant()?.0)
}

/// A lane arm's recorded config: the shared distribution config plus the
/// `lane_suite` block (executor, topology, lanes, route, state format, range).
fn lane_arm_config(
    arm: &str,
    identity: &Value,
    settings: &DistSettings,
    placement: lane_suite::Placement,
    range: (usize, usize),
) -> Result<Value, String> {
    lane_suite::lane_config(
        settings.config(arm, identity),
        arm,
        placement,
        lane_state_format()?,
        range,
    )
}

/// One lane arm process. Everything an arm would need is validated first (the
/// environment contract, neutral penalties, the shard range, the config and any
/// existing result's config), so a wrong invocation fails for its own reason;
/// then the EP2 executor ([`super::lane_ep`]) runs the shard, and the other
/// topologies refuse explicitly.
pub(super) fn run_lane_arm(
    arm: &str,
    model: &Path,
    dir: &Path,
    placement: lane_suite::Placement,
) -> Result<(), String> {
    let settings = DistSettings::from_env()?;
    if spec_request(
        settings.temp,
        settings.top_p,
        settings.top_k,
        settings.min_p,
        0,
        &settings.pen,
    )
    .penalties_active()
    {
        return Err(
            "the lane arms run neutral penalties only: penalized sampled requests decode AR on the mesh topologies"
                .into(),
        );
    }
    let range = shard_range(arm, settings.trials)?;
    let config = lane_arm_config(
        arm,
        &own_identity(model, dir, &LANE_CONTROL_ENV)?,
        &settings,
        placement,
        range,
    )?;
    let stem = if range != (0, settings.trials) {
        format!("{arm}.{}-{}", range.0, range.1)
    } else {
        arm.to_string()
    };
    let ids_path = dir.join(format!("{stem}.ids.json"));
    if ids_path.exists() {
        check_config(&ids_path, &config)?;
    }
    println!(
        "ARM {arm} executor={} topology={} lanes={} trials {}..{} of {} rows_per_trial={}",
        placement.executor.as_str(),
        placement.topology.as_str(),
        placement.lanes,
        range.0,
        range.1,
        settings.trials,
        placement.lanes
    );
    println!("ARM {arm} config {config}");
    match placement.topology {
        lane_suite::Topology::Ep2 => {
            super::lane_ep::run_ep_arm(arm, model, dir, placement, &settings, range, &config, &stem)
        }
        other => Err(lane_topology_unsupported(other)),
    }
}

fn dir_file_names(dir: &Path) -> Result<Vec<String>, String> {
    let mut names = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        names.push(
            entry
                .map_err(|e| e.to_string())?
                .file_name()
                .to_string_lossy()
                .into_owned(),
        );
    }
    names.sort();
    Ok(names)
}

fn read_rows(path: &Path) -> Result<Vec<Vec<u32>>, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Joins the lane shards of `dir` into the whole-arm files the verdict reads.
/// CPU only. The set is planned first ([`lane_suite::plan_lane_merge`]): a
/// mixed, incomplete, overlapping or mis-labelled set writes nothing. The
/// merged set must also be the one this process's environment names
/// (topology, lanes, state format), so a shard set from another configuration
/// cannot be merged under this one's label.
pub(super) fn merge_lane_suite(
    dir: &Path,
    trials: usize,
    placement: lane_suite::Placement,
) -> Result<(), String> {
    let names = dir_file_names(dir)?;
    lane_suite::reject_foreign_shards(&names, placement.topology)?;
    let mut rows: BTreeMap<(String, (usize, usize)), Vec<Vec<u32>>> = BTreeMap::new();
    let mut by_arm: BTreeMap<String, Vec<lane_suite::LaneShard>> = BTreeMap::new();
    for (arm, range) in lane_suite::shard_files(&names) {
        let part = read_rows(&late_file(dir, &arm, range, "ids.json"))?;
        let config = read_json(&late_file(dir, &arm, range, "ids.config.json"))?;
        by_arm
            .entry(arm.clone())
            .or_default()
            .push(lane_suite::LaneShard {
                file_range: range,
                config,
                rows: part.len(),
            });
        rows.insert((arm, range), part);
    }
    let plan = lane_suite::plan_lane_merge(&by_arm, trials, placement.topology)?;
    if plan.topology != placement.topology.as_str()
        || plan.lanes != placement.lanes
        || plan.state_format != lane_state_format()?
    {
        return Err(format!(
            "the shards are topology {} / {} lanes / state {}, but this process's environment names topology {} / {} lanes / state {}: merge under the environment that produced them",
            plan.topology,
            plan.lanes,
            plan.state_format,
            placement.topology.as_str(),
            placement.lanes,
            lane_state_format()?
        ));
    }
    for (arm, ranges) in &plan.arms {
        let mut ids: Vec<Vec<u32>> = Vec::with_capacity(trials * plan.lanes);
        let mut evidence = Vec::with_capacity(ranges.len());
        let mut text = String::new();
        for &range in ranges {
            ids.extend(rows.remove(&(arm.clone(), range)).expect("planned shard"));
            evidence.push(read_json(&late_file(dir, arm, range, "evidence.json"))?);
            text.push_str(
                &fs::read_to_string(late_file(dir, arm, range, "text.txt")).unwrap_or_default(),
            );
        }
        let config = lane_suite::merged_config(
            &by_arm[arm]
                .iter()
                .find(|shard| shard.file_range == ranges[0])
                .expect("planned shard")
                .config,
            trials,
        );
        let merged_evidence = lane_suite::merge_evidence(&evidence)?;
        let whole = dir.join(format!("{arm}.ids.json"));
        fs::write(&whole, serde_json::to_vec(&ids).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        write_config(&whole, &config)?;
        write_receipt(
            &whole,
            &read_json(&late_file(dir, arm, ranges[0], "ids.receipt.json"))?,
        )?;
        write_json(&dir.join(format!("{arm}.evidence.json")), &merged_evidence)?;
        fs::write(dir.join(format!("{arm}.text.txt")), text).map_err(|e| e.to_string())?;
        eprintln!(
            "{arm}: merged {} shards into {}",
            ranges.len(),
            whole.display()
        );
    }
    Ok(())
}

/// The lane verdict: every lane arm's merged, config-verified result against
/// the AR reference lanes, pooled and per lane, plus route coverage. The
/// parent never launches a missing arm. `Err` carries the failure message.
pub(super) fn lane_verdict(
    model: &Path,
    dir: &Path,
    placement: lane_suite::Placement,
) -> Result<String, String> {
    let settings = DistSettings::from_env()?;
    if settings.trials < LANE_MIN_TRIALS {
        return Err(format!(
            "{TRIALS_ENV}={}: the lane verdict needs at least {LANE_MIN_TRIALS} trials per arm",
            settings.trials
        ));
    }
    let trials = settings.trials;
    let model_id = hw_identity::model_identity(model, dir)?;
    let cohort = lane_suite::cohort_arms(placement.topology);
    let mut samples: BTreeMap<&str, Vec<Vec<u32>>> = BTreeMap::new();
    for &arm in cohort {
        let ids_path = dir.join(format!("{arm}.ids.json"));
        if !ids_path.exists() {
            return Err(format!(
                "{} is missing: run the {arm} shards covering 0..{trials}, then sampled_mtp_merge_short_shards; the lane verdict never launches lane arms",
                ids_path.display()
            ));
        }
        let expected = lane_arm_config(
            arm,
            &arm_identity(arm, "leviathan", &model_id, &LANE_CONTROL_ENV),
            &settings,
            placement,
            (0, trials),
        )?;
        check_config(&ids_path, &expected)?;
        let rows = read_rows(&ids_path)?;
        if rows.len() != trials * placement.lanes {
            return Err(format!(
                "{arm}: {} rows, expected {trials} trials x {} lanes",
                rows.len(),
                placement.lanes
            ));
        }
        samples.insert(arm, rows);
    }
    let evidence = cohort
        .iter()
        .map(|arm| Ok((*arm, read_json(&dir.join(format!("{arm}.evidence.json")))?)))
        .collect::<Result<BTreeMap<&str, Value>, String>>()?;
    let mut judged =
        judge_lane_samples(&samples, &evidence, placement, trials, lane_state_format()?)?;
    // The adaptive arm's floor-on route, from its recorded config (the identity
    // check above already compared it with this environment's).
    let adaptive_config = read_json(&dir.join("lane_adaptive.ids.config.json"))?;
    if let Some(failure) = lane_suite::adaptive_floor_failure(&adaptive_config) {
        judged.coverage.push(format!("{failure}\n"));
    }
    fs::write(dir.join("report.txt"), &judged.report).map_err(|e| e.to_string())?;
    eprint!("{}", judged.report);
    if judged.failures.is_empty() && judged.coverage.is_empty() {
        Ok(judged.report)
    } else {
        Err(failure_message(dir, &judged.failures, &judged.coverage))
    }
}

/// The statistical and coverage outcome of a lane cohort.
struct Judged {
    report: String,
    /// p-value failures (`DISTRIBUTION FAILURE (law)`).
    failures: Vec<String>,
    /// Route-coverage failures (`COVERAGE FAILURE`).
    coverage: Vec<String>,
}

/// Every non-reference lane arm against `lane_ar`, pooled and (with more than
/// one lane) per lane, plus every arm's route coverage from its evidence. Pure:
/// no file, device or model.
fn judge_lane_samples(
    samples: &BTreeMap<&str, Vec<Vec<u32>>>,
    evidence: &BTreeMap<&str, Value>,
    placement: lane_suite::Placement,
    trials: usize,
    state_format: &str,
) -> Result<Judged, String> {
    let mut report = format!(
        "lane suite: topology={} lanes={} trials={trials} state={state_format}\n",
        placement.topology.as_str(),
        placement.lanes,
    );
    let mut failures = Vec::new();
    let mut coverage = Vec::new();
    let get = |arm: &str| {
        samples
            .get(arm)
            .ok_or_else(|| format!("no samples for {arm}"))
    };
    let reference = get(lane_suite::LANE_REFERENCE_ARM)?;
    let reference_lanes = lane_suite::per_lane_rows(reference, placement.lanes)?;
    for &arm in lane_suite::cohort_arms(placement.topology) {
        let arm_evidence = evidence
            .get(arm)
            .ok_or_else(|| format!("no evidence for {arm}"))?;
        for line in lane_suite::lane_coverage(
            placement.topology,
            arm,
            placement.lanes,
            trials,
            arm_evidence,
        ) {
            coverage.push(format!("{line}\n"));
        }
        if arm == lane_suite::LANE_REFERENCE_ARM {
            continue;
        }
        let rows = get(arm)?;
        compare_to_reference(
            arm,
            lane_suite::LANE_REFERENCE_ARM,
            reference,
            rows,
            &mut report,
            &mut failures,
        );
        if placement.lanes > 1 {
            let lanes = lane_suite::per_lane_rows(rows, placement.lanes)?;
            for (lane, (want, got)) in reference_lanes.iter().zip(&lanes).enumerate() {
                compare_to_reference(
                    &format!("{arm}[lane {lane}]"),
                    &format!("{}[lane {lane}]", lane_suite::LANE_REFERENCE_ARM),
                    want,
                    got,
                    &mut report,
                    &mut failures,
                );
            }
        }
    }
    Ok(Judged {
        report,
        failures,
        coverage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placement(lanes: usize) -> lane_suite::Placement {
        lane_suite::Placement {
            executor: lane_suite::Executor::Lanes,
            topology: lane_suite::Topology::Ep2,
            lanes,
        }
    }

    #[test]
    fn only_the_ep2_topology_has_a_lane_executor() {
        for topology in [lane_suite::Topology::Single, lane_suite::Topology::Pp2] {
            let message = lane_topology_unsupported(topology);
            assert!(message.contains("ep2 only"), "{message}");
            assert!(message.contains(topology.as_str()), "{message}");
            assert!(message.contains("ran no trials"), "{message}");
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lane-run-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A deterministic row generator: `TOKENS` ids per row drawn from a small
    /// alphabet whose weights are `weights`, with a weak dependence of the
    /// second id on the first so the (t1, t2) joint is not a product.
    fn rows(count: usize, weights: [u64; 3], stream: u64) -> Vec<Vec<u32>> {
        let mut state = stream.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let total: u64 = weights.iter().sum();
        (0..count)
            .map(|_| {
                let mut row = Vec::with_capacity(TOKENS);
                for position in 0..TOKENS {
                    let mut draw = next() % total;
                    let mut id = 0u32;
                    for (index, weight) in weights.iter().enumerate() {
                        if draw < *weight {
                            id = index as u32;
                            break;
                        }
                        draw -= weight;
                    }
                    if position == 1 && row[0] == 2 {
                        id = (id + 1) % 3;
                    }
                    row.push(id);
                }
                row
            })
            .collect()
    }

    fn evidence(arm: &str, trials: u64, lanes: u64) -> Value {
        let requests = trials * lanes;
        let (mtp, ar, retired, cal, nowin, copy, pm) = match arm {
            "lane_ar" => (0, 90, 0, 0, requests, 0, 0),
            "lane_batched" => (90, 0, 0, 0, 0, 0, 0),
            "lane_adaptive" => (90, 9, 1, requests, 0, 0, 0),
            _ => (90, 0, 0, 0, 0, 30, 30),
        };
        json!({
            "trials": trials, "requests": requests, "mtp_windows": mtp, "ar_windows": ar,
            "retired_trials": retired, "calibration_ok_trials": cal, "no_window_trials": nowin,
            "copy_windows": copy, "copy_point_mass_windows": pm,
        })
    }

    fn cohort(
        lanes: usize,
        trials: usize,
        weights: impl Fn(&str) -> [u64; 3],
    ) -> (
        BTreeMap<&'static str, Vec<Vec<u32>>>,
        BTreeMap<&'static str, Value>,
    ) {
        let mut samples = BTreeMap::new();
        let mut evidence_by_arm = BTreeMap::new();
        for (index, arm) in lane_suite::EP2_ARMS.iter().enumerate() {
            samples.insert(*arm, rows(trials * lanes, weights(arm), 11 + index as u64));
            evidence_by_arm.insert(*arm, evidence(arm, trials as u64, lanes as u64));
        }
        (samples, evidence_by_arm)
    }

    #[test]
    fn judge_passes_equal_distributions_pooled_and_per_lane() {
        let (samples, evidence) = cohort(8, 600, |_| [5, 3, 2]);
        let judged = judge_lane_samples(&samples, &evidence, placement(8), 600, "q8").unwrap();
        assert!(judged.failures.is_empty(), "{}", judged.failures.concat());
        assert!(judged.coverage.is_empty(), "{}", judged.coverage.concat());
        for arm in ["lane_batched", "lane_adaptive"] {
            assert!(
                judged.report.contains(&format!("{arm}[lane 0]")),
                "{arm} pooled only"
            );
            assert!(judged.report.contains(&format!("{arm}[lane 7]")));
        }
    }

    #[test]
    fn judge_flags_one_bad_lane_that_pooling_would_hide() {
        // Lane 3 of lane_batched draws from a different law; the other seven
        // lanes and the reference share one. The pooled test sees a seven
        // eighths mixture, the per-lane test must catch the lane.
        let (mut samples, evidence) = cohort(8, 2000, |_| [5, 3, 2]);
        let bad = rows(2000, [2, 3, 5], 99);
        let batched = samples.get_mut("lane_batched").unwrap();
        for (trial, row) in bad.into_iter().enumerate() {
            batched[trial * 8 + 3] = row;
        }
        let judged = judge_lane_samples(&samples, &evidence, placement(8), 2000, "q8").unwrap();
        assert!(
            judged
                .failures
                .iter()
                .any(|f| f.contains("lane_batched[lane 3]")),
            "{}",
            judged.report
        );
        assert!(
            !judged
                .failures
                .iter()
                .any(|f| f.contains("lane_batched[lane 4]")),
            "an unaffected lane must not fail"
        );
        assert!(judged.coverage.is_empty());
    }

    #[test]
    fn judge_reports_coverage_failures_separately_from_the_law() {
        let (samples, mut by_arm) = cohort(1, 600, |_| [5, 3, 2]);
        by_arm.insert("lane_batched", {
            let mut value = evidence("lane_batched", 600, 1);
            value["ar_windows"] = json!(3);
            value
        });
        let judged = judge_lane_samples(&samples, &by_arm, placement(1), 600, "q8").unwrap();
        assert!(judged.failures.is_empty());
        assert!(
            judged.coverage.iter().any(|c| c.contains("lane_batched")),
            "{:?}",
            judged.coverage
        );
        // One lane: pooled only, no per-lane lines.
        assert!(!judged.report.contains("[lane 0]"));
        // Missing inputs are errors, never skipped arms.
        let mut short = samples.clone();
        short.remove("lane_adaptive");
        assert!(judge_lane_samples(&short, &by_arm, placement(1), 600, "q8").is_err());
        let mut no_evidence = by_arm.clone();
        no_evidence.remove("lane_ar");
        assert!(judge_lane_samples(&samples, &no_evidence, placement(1), 600, "q8").is_err());
    }

    // ---- shard merge on disk

    const TRIALS_FIXTURE: usize = 4;

    fn write_shard(dir: &Path, arm: &str, range: (usize, usize), topology: &str, artifact: &str) {
        let lanes = 8;
        let mut base = json!({
            "kind": "distribution", "arm": arm, "fixture": "prose",
            "repeat_penalty": 1.0, "repeat_window": 0, "presence_penalty": 0.0,
            "frequency_penalty": 0.0, "mtp_k": 3, "temp": 1.0, "top_p": 0.95,
            "top_k": null, "min_p": 0.0, "trials": TRIALS_FIXTURE, "tokens": 4,
            "sampled_mode": "leviathan",
            "identity": {
                "model": {"size": 1, "sha256": artifact},
                "env": {
                    lane_suite::EXECUTOR_ENV: "lanes",
                    lane_suite::LANES_ENV: "8",
                    lane_suite::TOPOLOGY_ENV: topology,
                },
            },
        });
        let placement = lane_suite::Placement {
            executor: lane_suite::Executor::Lanes,
            topology: lane_suite::Topology::parse(Some(topology)).unwrap(),
            lanes,
        };
        let state = lane_state_format().unwrap();
        if state != "q8" {
            base["identity"]["env"][lane_suite::STATE_QUANT_ENV] = json!(state);
        }
        let config = lane_suite::lane_config(base, arm, placement, state, range).unwrap();
        let stem = format!("{arm}.{}-{}", range.0, range.1);
        let ids: Vec<Vec<u32>> = (0..(range.1 - range.0) * lanes)
            .map(|row| vec![range.0 as u32, row as u32])
            .collect();
        fs::write(
            dir.join(format!("{stem}.ids.json")),
            serde_json::to_vec(&ids).unwrap(),
        )
        .unwrap();
        fs::write(
            dir.join(format!("{stem}.ids.config.json")),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        fs::write(
            dir.join(format!("{stem}.ids.receipt.json")),
            b"{\"gpu\":\"x\"}",
        )
        .unwrap();
        let mut shard_evidence = evidence(arm, (range.1 - range.0) as u64, lanes as u64);
        shard_evidence["per_trial"] = json!((range.0..range.1).collect::<Vec<_>>());
        fs::write(
            dir.join(format!("{stem}.evidence.json")),
            serde_json::to_vec(&shard_evidence).unwrap(),
        )
        .unwrap();
        fs::write(
            dir.join(format!("{stem}.text.txt")),
            format!("{arm} {range:?}\n"),
        )
        .unwrap();
    }

    fn write_all(dir: &Path) {
        for arm in lane_suite::EP2_ARMS {
            write_shard(dir, arm, (0, 2), "ep2", "eafaff94");
            write_shard(dir, arm, (2, 4), "ep2", "eafaff94");
        }
    }

    #[test]
    fn merge_writes_whole_arm_files_for_a_complete_set() {
        let dir = scratch("merge-ok");
        write_all(&dir);
        merge_lane_suite(&dir, TRIALS_FIXTURE, placement(8)).unwrap();
        for arm in lane_suite::EP2_ARMS {
            let ids = read_rows(&dir.join(format!("{arm}.ids.json"))).unwrap();
            assert_eq!(ids.len(), TRIALS_FIXTURE * 8, "{arm}");
            // Shards concatenated in trial order.
            assert_eq!(ids[0], vec![0, 0]);
            assert_eq!(ids[16], vec![2, 0]);
            let config = read_json(&dir.join(format!("{arm}.ids.config.json"))).unwrap();
            assert_eq!(config["lane_suite"]["range"], json!([0, TRIALS_FIXTURE]));
            let merged = read_json(&dir.join(format!("{arm}.evidence.json"))).unwrap();
            assert_eq!(merged["trials"], json!(TRIALS_FIXTURE));
            assert_eq!(merged["requests"], json!(TRIALS_FIXTURE * 8));
            assert_eq!(merged["per_trial"], json!([0, 1, 2, 3]));
            assert!(dir.join(format!("{arm}.ids.receipt.json")).exists());
            let text = fs::read_to_string(dir.join(format!("{arm}.text.txt"))).unwrap();
            assert!(text.starts_with(&format!("{arm} (0, 2)")), "{text}");
            // The merged evidence is a coverage-clean cohort for its arm.
            assert!(
                lane_suite::lane_coverage(
                    lane_suite::Topology::Ep2,
                    arm,
                    8,
                    TRIALS_FIXTURE,
                    &merged
                )
                .is_empty(),
                "{arm}"
            );
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    fn assert_nothing_merged(dir: &Path) {
        for arm in lane_suite::EP2_ARMS {
            assert!(
                !dir.join(format!("{arm}.ids.json")).exists(),
                "{arm}: a refused merge must write nothing"
            );
        }
    }

    #[test]
    fn merge_refuses_incomplete_and_mixed_sets_and_writes_nothing() {
        // A missing arm.
        let dir = scratch("merge-arm");
        write_all(&dir);
        for file in fs::read_dir(&dir).unwrap() {
            let path = file.unwrap().path();
            if path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("lane_adaptive.")
            {
                fs::remove_file(path).unwrap();
            }
        }
        let error = merge_lane_suite(&dir, TRIALS_FIXTURE, placement(8)).unwrap_err();
        assert!(
            error.contains("incomplete arm set") && error.contains("lane_adaptive"),
            "{error}"
        );
        assert_nothing_merged(&dir);
        fs::remove_dir_all(&dir).unwrap();

        // A missing range of one arm.
        let dir = scratch("merge-range");
        write_all(&dir);
        for ext in ["ids.json", "ids.config.json"] {
            fs::remove_file(dir.join(format!("lane_batched.2-4.{ext}"))).unwrap();
        }
        let error = merge_lane_suite(&dir, TRIALS_FIXTURE, placement(8)).unwrap_err();
        assert!(error.contains("incomplete shard set"), "{error}");
        assert_nothing_merged(&dir);
        fs::remove_dir_all(&dir).unwrap();

        // A shard from another artifact.
        let dir = scratch("merge-artifact");
        write_all(&dir);
        write_shard(&dir, "lane_adaptive", (2, 4), "ep2", "541cfb55");
        let error = merge_lane_suite(&dir, TRIALS_FIXTURE, placement(8)).unwrap_err();
        assert!(
            error.contains("mixed") && error.contains("artifact"),
            "{error}"
        );
        assert_nothing_merged(&dir);
        fs::remove_dir_all(&dir).unwrap();

        // A singleton shard left in the lane directory.
        let dir = scratch("merge-foreign");
        write_all(&dir);
        fs::write(dir.join("ar.0-2.ids.json"), b"[]").unwrap();
        let error = merge_lane_suite(&dir, TRIALS_FIXTURE, placement(8)).unwrap_err();
        assert!(
            error.contains("mixed shard set") && error.contains("\"ar\""),
            "{error}"
        );
        assert_nothing_merged(&dir);
        fs::remove_dir_all(&dir).unwrap();

        // The copy lane has no ep2 cohort: its shard is foreign under ep2.
        let dir = scratch("merge-copy");
        write_all(&dir);
        write_shard(&dir, "lane_copy", (0, 4), "ep2", "eafaff94");
        let error = merge_lane_suite(&dir, TRIALS_FIXTURE, placement(8)).unwrap_err();
        assert!(
            error.contains("mixed shard set") && error.contains("lane_copy"),
            "{error}"
        );
        assert_nothing_merged(&dir);
        fs::remove_dir_all(&dir).unwrap();

        // The process environment names another topology / lane count than the shards.
        let dir = scratch("merge-env");
        write_all(&dir);
        let wrong_lanes = merge_lane_suite(&dir, TRIALS_FIXTURE, placement(1)).unwrap_err();
        assert!(wrong_lanes.contains("environment"), "{wrong_lanes}");
        assert_nothing_merged(&dir);
        fs::remove_dir_all(&dir).unwrap();
        // Shards recorded under pp2 are a valid ep2-cohort set by arm name, but
        // not this ep2 process's.
        let dir = scratch("merge-topology");
        for arm in lane_suite::EP2_ARMS {
            write_shard(&dir, arm, (0, 4), "pp2", "eafaff94");
        }
        let error = merge_lane_suite(&dir, TRIALS_FIXTURE, placement(8)).unwrap_err();
        assert!(error.contains("environment"), "{error}");
        assert_nothing_merged(&dir);
        fs::remove_dir_all(&dir).unwrap();
    }
}
