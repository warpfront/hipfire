// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! R6/R7 promotion pipeline: `can_promote` evaluates ordered stages and refuses
//! at the first failure. The final `kld_state` stage accepts either the
//! bit-identical exact-output proof or, for numeric candidates, promotion-grade
//! KLD evidence: the R9 token-bootstrap evidence (`kld_token`, [`kld_token_stage`])
//! when attached, else the R7 paired per-chunk evidence ([`kld_state_stage`]); a
//! numeric candidate without that evidence is never promoted.

use super::{
    ktok, rowclass, AttemptV1, ClaimPolicyV1, KldRunV1, KldTokenEvidenceV2, PromotionV1, Refusal,
    TuneEvidenceV1,
};
use serde_json::Value;

pub const STAGES: [&str; 6] = [
    "m7",
    "correctness",
    "abba",
    "noise_floor",
    "confirm",
    "kld_state",
];

fn refuse(idx: usize, reason: String) -> Refusal {
    Refusal {
        stage: STAGES[idx].to_owned(),
        reason,
        passed: STAGES[..idx].iter().map(|s| (*s).to_owned()).collect(),
    }
}

fn attempts(ev: &TuneEvidenceV1) -> impl Iterator<Item = (String, &AttemptV1)> {
    std::iter::once(("discovery".to_owned(), &ev.discovery)).chain(
        ev.confirms
            .iter()
            .enumerate()
            .map(|(i, a)| (format!("confirm[{i}]"), a)),
    )
}

fn stage_m7(ev: &TuneEvidenceV1) -> Result<(), String> {
    for (n, a) in attempts(ev) {
        if a.m7_status.as_deref() != Some("pass") {
            return Err(format!(
                "{n}: m7_status is {:?}, need \"pass\"",
                a.m7_status
            ));
        }
    }
    Ok(())
}

fn stage_correctness(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> Result<(), String> {
    for (n, a) in attempts(ev) {
        if a.correctness_pass != Some(true) {
            return Err(format!(
                "{n}: correctness_pass is {:?}, need true",
                a.correctness_pass
            ));
        }
        match a.identical_to_baseline {
            Some(true) => {}
            Some(false) if policy.admit_numeric => {}
            Some(false) => {
                return Err(format!(
                    "{n}: identical_to_baseline is false; NUMERIC candidates are not admitted by this policy (admit_numeric=false)"
                ))
            }
            None => {
                return Err(format!(
                    "{n}: identical_to_baseline is None; need an explicit exact/numeric determination"
                ))
            }
        }
    }
    Ok(())
}

/// ABBA criteria for one attempt (stage 3).
fn abba_ok(n: &str, a: &AttemptV1, policy: &ClaimPolicyV1) -> Result<(), String> {
    if a.separable != Some(true) {
        return Err(format!("{n}: separable is {:?}, need true", a.separable));
    }
    if a.direction.as_deref() != Some("faster") {
        return Err(format!(
            "{n}: direction is {:?}, need \"faster\"",
            a.direction
        ));
    }
    match a.sign_p {
        Some(p) if p <= policy.sign_p_max => {}
        other => {
            return Err(format!(
                "{n}: sign_p {other:?} not <= {}",
                policy.sign_p_max
            ))
        }
    }
    if a.material != Some(true) {
        return Err(format!("{n}: material is {:?}, need true", a.material));
    }
    match a.block_ratio {
        Some(r) if (r - 1.0).abs() >= policy.min_effect => {}
        other => {
            return Err(format!(
                "{n}: block_ratio {other:?} effect below min_effect {}",
                policy.min_effect
            ))
        }
    }
    Ok(())
}

/// Noise-floor criteria for one attempt against the evidence floor (stage 4).
fn noise_ok(n: &str, a: &AttemptV1, floor: f64) -> Result<(), String> {
    match a.block_ratio {
        Some(r) if (r - 1.0).abs() > floor => {}
        other => {
            return Err(format!(
                "{n}: block_ratio {other:?} effect does not exceed noise floor {floor}"
            ))
        }
    }
    if a.exceeds_noise_floor != Some(true) {
        return Err(format!(
            "{n}: exceeds_noise_floor is {:?}, need true",
            a.exceeds_noise_floor
        ));
    }
    Ok(())
}

fn stage_abba(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> Result<(), String> {
    abba_ok("discovery", &ev.discovery, policy)
}

fn stage_noise(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> Result<(), String> {
    if ev.aa_runs < policy.aa_processes {
        return Err(format!(
            "aa_runs {} < required {}",
            ev.aa_runs, policy.aa_processes
        ));
    }
    let Some(floor) = ev.noise_floor else {
        return Err("no noise floor measured; a missing floor is never a pass".to_owned());
    };
    noise_ok("discovery", &ev.discovery, floor)
}

fn stage_confirm(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> Result<(), String> {
    if ev.confirms.len() < policy.confirm_processes {
        return Err(format!(
            "{} confirmation(s) < required {}",
            ev.confirms.len(),
            policy.confirm_processes
        ));
    }
    let floor = ev
        .noise_floor
        .ok_or_else(|| "no noise floor for confirmation".to_owned())?;
    for (i, c) in ev.confirms.iter().enumerate() {
        let n = format!("confirm[{i}]");
        abba_ok(&n, c, policy)?;
        noise_ok(&n, c, floor)?;
        if c.candidate_code_sha256 != ev.discovery.candidate_code_sha256
            || c.candidate_code_sha256.is_none()
        {
            return Err(format!("{n}: candidate_code_sha256 differs from discovery"));
        }
        if c.baseline_code_sha256 != ev.discovery.baseline_code_sha256
            || c.baseline_code_sha256.is_none()
        {
            return Err(format!("{n}: baseline_code_sha256 differs from discovery"));
        }
    }
    Ok(())
}

/// Run/evidence consistency checks shared by every `kld_state` path.
fn kld_consistency(ev: &TuneEvidenceV1, i: &KldRunV1, c: &KldRunV1) -> Result<(), String> {
    if i.role != "incumbent" {
        return Err(format!("incumbent run role is {:?}", i.role));
    }
    if c.role != "candidate" {
        return Err(format!("candidate run role is {:?}", c.role));
    }
    for (n, r) in [("incumbent", i), ("candidate", c)] {
        if r.scoring_mode != "prefill" {
            return Err(format!(
                "{n}: scoring_mode is {:?}, need \"prefill\"",
                r.scoring_mode
            ));
        }
        for var in ["HIPFIRE_GRAPH", "HIPFIRE_NORMALIZE_PROMPT"] {
            if r.env.get(var).map(String::as_str) != Some("0") {
                return Err(format!(
                    "{n}: env {var} is {:?}, need \"0\"",
                    r.env.get(var)
                ));
            }
        }
        if !r.fresh_home {
            return Err(format!("{n}: fresh_home is false"));
        }
    }
    let same = [
        ("reference_sha256", &i.reference_sha256, &c.reference_sha256),
        ("model_sha256", &i.model_sha256, &c.model_sha256),
        ("evaluator_sha256", &i.evaluator_sha256, &c.evaluator_sha256),
        ("kv_mode", &i.kv_mode, &c.kv_mode),
        ("kv_v", &i.kv_v, &c.kv_v),
    ];
    for (n, a, b) in same {
        if a != b {
            return Err(format!("{n} differs between incumbent and candidate runs"));
        }
    }
    if i.max_chunks != c.max_chunks {
        return Err(format!(
            "max_chunks differs between runs ({} vs {})",
            i.max_chunks, c.max_chunks
        ));
    }
    if i.tune_table_path.is_some() {
        return Err("incumbent run used a tune table (tune_table_path is set)".to_owned());
    }
    match (
        &c.selected_code_object_sha256,
        &ev.discovery.candidate_co_sha256,
    ) {
        (Some(a), Some(b)) if a == b => {}
        (a, b) => {
            return Err(format!(
                "candidate selected_code_object_sha256 {a:?} != discovery candidate_co_sha256 {b:?}"
            ))
        }
    }
    Ok(())
}

/// Consistency of a token-bootstrap evidence: every stratum's run pair passes
/// [`kld_consistency`] and carries hash-recorded per-token files; the strata
/// agree on model, evaluator, KV flags and selected code object, and use
/// distinct references (a repeated reference would count tokens twice).
fn kld_token_consistency(ev: &TuneEvidenceV1, t: &KldTokenEvidenceV2) -> Result<(), String> {
    let Some(first) = t.strata.first() else {
        return Err("kld_token evidence has no strata".to_owned());
    };
    let mut labels = std::collections::BTreeSet::new();
    let mut refs = std::collections::BTreeSet::new();
    for s in &t.strata {
        let why = |e: String| format!("stratum {}: {e}", s.label);
        kld_consistency(ev, &s.incumbent, &s.candidate).map_err(why)?;
        for r in [&s.incumbent, &s.candidate] {
            if r.ktok_path.is_none() || r.ktok_sha256.is_none() {
                return Err(why(format!("{} run has no ktok_path/ktok_sha256", r.role)));
            }
        }
        if !labels.insert(s.label.as_str()) {
            return Err(format!("stratum label {:?} repeats", s.label));
        }
        if !refs.insert(s.incumbent.reference_sha256.as_str()) {
            return Err(why("reference_sha256 repeats another stratum's".to_owned()));
        }
        let (a, b) = (&first.incumbent, &s.incumbent);
        let same = [
            ("model_sha256", &a.model_sha256, &b.model_sha256),
            ("evaluator_sha256", &a.evaluator_sha256, &b.evaluator_sha256),
            ("kv_mode", &a.kv_mode, &b.kv_mode),
            ("kv_v", &a.kv_v, &b.kv_v),
        ];
        for (n, x, y) in same {
            if x != y {
                return Err(why(format!("{n} differs from stratum {}", first.label)));
            }
        }
        if s.candidate.selected_code_object_sha256 != first.candidate.selected_code_object_sha256 {
            return Err(why(format!(
                "candidate selected_code_object_sha256 differs from stratum {}",
                first.label
            )));
        }
    }
    Ok(())
}

/// Numeric stage-6 verdict of a token-bootstrap evidence: promotion grade,
/// consistent runs, both one-sided 95% upper bounds below `kld_margin` and at
/// least [`ktok::STRATUM_MIN_CHUNKS`] chunks per stratum
/// ([`ktok::judge_bounds`]). Refusal is the default.
fn kld_token_stage(
    ev: &TuneEvidenceV1,
    t: &KldTokenEvidenceV2,
    policy: &ClaimPolicyV1,
) -> Result<(), String> {
    if t.grade != "promotion" {
        return Err(format!(
            "kld_token grade is {:?}; screening-grade evidence never promotes (need \"promotion\")",
            t.grade
        ));
    }
    kld_token_consistency(ev, t)?;
    ktok::judge_bounds(t, policy.kld_margin)
}

/// Stage 6: model-level KLD / state-equality evidence. Exposed so callers can
/// judge it independently of stages 1-5.
pub fn kld_state_stage(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> Result<(), String> {
    let all_identical = attempts(ev).all(|(_, a)| a.identical_to_baseline == Some(true));
    if all_identical {
        // Bit-identical path: the exact-output proof suffices; any attached
        // state evidence must agree.
        if let Some(k) = &ev.kld_state {
            if !k.exact {
                return Err(
                    "kld_state evidence attached to an identical-to-baseline candidate has exact=false (kldseq bytes differ)"
                        .to_owned(),
                );
            }
            kld_consistency(ev, &k.incumbent, &k.candidate)?;
        }
        if let Some(t) = &ev.kld_token {
            if !t.stat.exact {
                return Err(
                    "kld_token evidence attached to an identical-to-baseline candidate has exact=false (nonzero per-token differences)"
                        .to_owned(),
                );
            }
            kld_token_consistency(ev, t)?;
        }
        return Ok(());
    }
    if let Some(t) = &ev.kld_token {
        return kld_token_stage(ev, t, policy);
    }
    let Some(k) = &ev.kld_state else {
        return Err(
            "numeric candidate has no kld_state evidence; no pass without evidence".to_owned(),
        );
    };
    if k.grade != "promotion" {
        return Err(format!(
            "kld_state grade is {:?}; screening-grade evidence never promotes (need \"promotion\")",
            k.grade
        ));
    }
    kld_consistency(ev, &k.incumbent, &k.candidate)?;
    let min = policy.kld_promotion_min_chunks;
    if k.incumbent.max_chunks < min {
        return Err(format!(
            "max_chunks {} < kld_promotion_min_chunks {min}",
            k.incumbent.max_chunks
        ));
    }
    if k.paired.n < min {
        return Err(format!(
            "paired.n {} < kld_promotion_min_chunks {min}",
            k.paired.n
        ));
    }
    if k.paired.ci95_hi.is_nan() || k.paired.ci95_hi > policy.kld_margin {
        return Err(format!(
            "paired ci95_hi {} exceeds kld_margin {}",
            k.paired.ci95_hi, policy.kld_margin
        ));
    }
    Ok(())
}

/// Stage name of the row-class gate ([`super::rowclass::gate`]). It is not in
/// [`STAGES`]: it runs between `correctness` and `abba`, and only for records
/// the gate [`applies`](super::rowclass::applies) to, which then also list it in
/// the `passed` stages.
pub const ROWCLASS_STAGE: &str = "rowclass";

/// Stage name of the MoE route-evidence gate: MoE evidence whose every attempt
/// ran only the `uniform` synthetic route is refused (`uniform-only route
/// evidence`); `route=hist:` / `skew` evidence passes.
pub const ROUTE_STAGE: &str = "route_evidence";

/// `route=<kind>` of a shape text such as `a3b,T=8192,route=hist:f.json`: the
/// kind before any `:` argument.
fn route_in_shape_text(s: &str) -> Option<String> {
    s.split(',')
        .find_map(|kv| kv.trim().strip_prefix("route="))
        .map(|v| v.split(':').next().unwrap_or(v).trim().to_owned())
}

/// The route kind a pm_loop record states: top-level `route` (string, or an
/// object's `kind`), else `shape.route`, else a `route=` key of a textual
/// `shape`.
fn record_route(rec: &Value) -> Option<String> {
    let kind = |v: &Value| match v {
        Value::String(s) => Some(s.split(':').next().unwrap_or(s).trim().to_owned()),
        Value::Object(o) => o.get("kind").and_then(Value::as_str).map(str::to_owned),
        _ => None,
    };
    if let Some(k) = rec.get("route").and_then(kind) {
        return Some(k);
    }
    match rec.get("shape")? {
        Value::String(s) => route_in_shape_text(s),
        shape => shape.get("route").and_then(kind),
    }
}

fn mentions_moe(v: Option<&Value>) -> bool {
    v.and_then(Value::as_str)
        .is_some_and(|s| s.to_ascii_lowercase().contains("moe"))
}

/// MoE evidence: a `moe` kernel/node name, or a `route` shape key anywhere.
/// `Ok(true)` = the gate judged the evidence.
fn stage_route(ev: &TuneEvidenceV1, recs: &[(String, Value)]) -> Result<bool, Refusal> {
    let routes: Vec<(String, Option<String>)> = attempts(ev)
        .map(|(n, _)| {
            let rec = recs.iter().find(|(rn, _)| *rn == n).map(|(_, r)| r);
            let route = rec
                .and_then(record_route)
                .or_else(|| route_in_shape_text(&ev.candidate.shape));
            (n, route)
        })
        .collect();
    let moe = ev.candidate.kernel.to_ascii_lowercase().contains("moe")
        || recs
            .iter()
            .any(|(_, r)| mentions_moe(r.get("kernel")) || mentions_moe(r.get("node")))
        || routes.iter().any(|(_, r)| r.is_some());
    if !moe {
        return Ok(false);
    }
    if routes.iter().all(|(_, r)| r.as_deref() == Some("uniform")) {
        return Err(Refusal {
            stage: ROUTE_STAGE.to_owned(),
            reason: "uniform-only route evidence: every attempt ran route=uniform; \
                     re-measure under route=hist:<route_hist.json> or route=skew"
                .to_owned(),
            passed: STAGES[..2].iter().map(|s| (*s).to_owned()).collect(),
        });
    }
    Ok(true)
}

/// Parsed JSON of every attempt's `record_path` that can be read. An unreadable
/// or non-JSON record is not a row-class record; its other stages still judge it.
fn load_records(ev: &TuneEvidenceV1) -> Vec<(String, Value)> {
    attempts(ev)
        .filter_map(|(n, a)| {
            let v = serde_json::from_slice(&std::fs::read(&a.record_path).ok()?).ok()?;
            Some((n, v))
        })
        .collect()
}

/// Row-class gate over every governed record. `Ok(true)` = at least one record
/// was judged.
fn stage_rowclass(
    recs: &[(String, Value)],
    policy: &ClaimPolicyV1,
    route_ran: bool,
) -> Result<bool, Refusal> {
    let mut ran = false;
    for (n, r) in recs {
        if !rowclass::applies(r, policy) {
            continue;
        }
        ran = true;
        rowclass::gate(r, policy).map_err(|e| Refusal {
            stage: ROWCLASS_STAGE.to_owned(),
            reason: format!("{n}: {e}"),
            passed: passed_before(2, route_ran, false),
        })?;
    }
    Ok(ran)
}

/// `passed` of a refusal at stage `idx` when the route / row-class gates did / did not run.
fn passed_before(idx: usize, route_ran: bool, rowclass_ran: bool) -> Vec<String> {
    let mut p: Vec<String> = STAGES[..idx].iter().map(|s| (*s).to_owned()).collect();
    if idx >= 2 {
        if route_ran {
            p.insert(2, ROUTE_STAGE.to_owned());
        }
        if rowclass_ran {
            p.insert(if route_ran { 3 } else { 2 }, ROWCLASS_STAGE.to_owned());
        }
    }
    p
}

/// Stages 1-5; `Ok((route_ran, rowclass_ran))`.
fn run_stages_1_to_5(
    ev: &TuneEvidenceV1,
    policy: &ClaimPolicyV1,
    recs: &[(String, Value)],
) -> Result<(bool, bool), Refusal> {
    stage_m7(ev).map_err(|r| refuse(0, r))?;
    stage_correctness(ev, policy).map_err(|r| refuse(1, r))?;
    let route_ran = stage_route(ev, recs)?;
    let ran = stage_rowclass(recs, policy, route_ran)?;
    let late = |idx: usize, reason: String| Refusal {
        passed: passed_before(idx, route_ran, ran),
        ..refuse(idx, reason)
    };
    stage_abba(ev, policy).map_err(|r| late(2, r))?;
    stage_noise(ev, policy).map_err(|r| late(3, r))?;
    stage_confirm(ev, policy).map_err(|r| late(4, r))?;
    Ok((route_ran, ran))
}

/// Stages 1-5 all pass (KLD/state not considered). Attempt records are read from
/// their `record_path`s for the row-class gate; see [`is_winner_with_records`].
pub fn is_winner(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> bool {
    is_winner_with_records(ev, policy, &load_records(ev))
}

/// [`is_winner`] with the attempts' parsed records (`(attempt name, record)`)
/// supplied by the caller.
pub fn is_winner_with_records(
    ev: &TuneEvidenceV1,
    policy: &ClaimPolicyV1,
    recs: &[(String, Value)],
) -> bool {
    run_stages_1_to_5(ev, policy, recs).is_ok()
}

pub fn can_promote(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> Result<PromotionV1, Refusal> {
    can_promote_with_records(ev, policy, &load_records(ev))
}

/// [`can_promote`] with the attempts' parsed records supplied by the caller. A
/// `hip_loop --rowclasses` record is refused if any admitted row class is
/// `Slower` (listing the classes), if its regime does not fit the LLC, or, for a
/// `contract:` node whose contract admits several classes, if it has no sweep.
pub fn can_promote_with_records(
    ev: &TuneEvidenceV1,
    policy: &ClaimPolicyV1,
    recs: &[(String, Value)],
) -> Result<PromotionV1, Refusal> {
    let (route_ran, ran) = run_stages_1_to_5(ev, policy, recs)?;
    kld_state_stage(ev, policy).map_err(|r| Refusal {
        passed: passed_before(5, route_ran, ran),
        ..refuse(5, r)
    })?;
    let ratios: Vec<f64> = attempts(ev).filter_map(|(_, a)| a.block_ratio).collect();
    let g = (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len() as f64).exp();
    let mut passed: Vec<String> = STAGES.iter().map(|s| (*s).to_owned()).collect();
    if route_ran {
        passed.insert(2, ROUTE_STAGE.to_owned());
    }
    if ran {
        passed.insert(if route_ran { 3 } else { 2 }, ROWCLASS_STAGE.to_owned());
    }
    Ok(PromotionV1 {
        passed,
        speedup: 1.0 / g,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::{
        CandidateSpecV1, KldStateEvidenceV1, KldStratumV2, PairedKldV1, TokenBootstrapV2,
    };
    use std::collections::BTreeMap;

    fn attempt(phase: &str) -> AttemptV1 {
        AttemptV1 {
            candidate_id: "c".into(),
            phase: phase.into(),
            replicate: 0,
            record_path: "r.json".into(),
            exit_code: 0,
            error_stage: None,
            error_message: None,
            m7_status: Some("pass".into()),
            correctness_pass: Some(true),
            identical_to_baseline: Some(true),
            baseline_launch_symbol: None,
            candidate_launch_symbol: None,
            baseline_co_sha256: None,
            candidate_co_sha256: Some("co".into()),
            baseline_code_sha256: Some("b".into()),
            candidate_code_sha256: Some("c".into()),
            separable: Some(true),
            direction: Some("faster".into()),
            sign_p: Some(1e-5),
            block_ratio: Some(0.95),
            material: Some(true),
            rounds_run: Some(10),
            noise_floor: Some(0.01),
            exceeds_noise_floor: Some(true),
            device_uuid: None,
            fabric_job: None,
        }
    }

    fn ev() -> TuneEvidenceV1 {
        TuneEvidenceV1 {
            candidate: CandidateSpecV1 {
                kernel: "k".into(),
                arch: "gfx1100".into(),
                shape: "s".into(),
                knobs: BTreeMap::new(),
            },
            discovery: attempt("discovery"),
            confirms: vec![attempt("confirm")],
            noise_floor: Some(0.01),
            aa_runs: 3,
            kld_state: None,
            kld_token: None,
        }
    }

    fn stage_of(ev: &TuneEvidenceV1) -> String {
        can_promote(ev, &ClaimPolicyV1::default())
            .unwrap_err()
            .stage
    }

    fn run(role: &str) -> KldRunV1 {
        KldRunV1 {
            role: role.into(),
            model_path: "m".into(),
            model_sha256: "m".into(),
            reference_path: "r".into(),
            reference_sha256: "r".into(),
            evaluator_path: "e".into(),
            evaluator_sha256: "e".into(),
            hipfire_commit: "c".into(),
            scoring_mode: "prefill".into(),
            max_chunks: 32,
            kv_mode: "q8".into(),
            kv_v: "q8".into(),
            env: BTreeMap::from([
                ("HIPFIRE_GRAPH".to_owned(), "0".to_owned()),
                ("HIPFIRE_NORMALIZE_PROMPT".to_owned(), "0".to_owned()),
            ]),
            fresh_home: true,
            tune_table_path: None,
            tune_table_sha256: None,
            selected_code_object_sha256: (role == "candidate").then(|| "co".to_owned()),
            kldseq_path: "k".into(),
            kldseq_sha256: "k".into(),
            ktok_path: None,
            ktok_sha256: None,
            device_uuid: "u".into(),
            fabric_job: Some("j".into()),
            slice_mean_kld: 0.01,
        }
    }

    fn kld(grade: &str, exact: bool, ci95_hi: f64) -> KldStateEvidenceV1 {
        KldStateEvidenceV1 {
            grade: grade.into(),
            incumbent: run("incumbent"),
            candidate: run("candidate"),
            paired: PairedKldV1 {
                n: 32,
                mean_diff: 0.0,
                sd_diff: 0.0001,
                ci95_lo: -0.0001,
                ci95_hi,
            },
            exact,
        }
    }

    fn numeric(k: Option<KldStateEvidenceV1>) -> TuneEvidenceV1 {
        let mut e = ev();
        e.discovery.identical_to_baseline = Some(false);
        e.confirms[0].identical_to_baseline = Some(false);
        e.kld_state = k;
        e
    }

    fn kld_reason(e: &TuneEvidenceV1) -> String {
        let r = can_promote(e, &ClaimPolicyV1::default()).unwrap_err();
        assert_eq!(r.stage, "kld_state");
        assert_eq!(r.passed, STAGES[..5].to_vec());
        r.reason
    }

    #[test]
    fn identical_path_passes_without_kld() {
        let e = ev();
        let p = ClaimPolicyV1::default();
        assert!(is_winner(&e, &p));
        let g = can_promote(&e, &p).unwrap();
        assert_eq!(g.passed, STAGES.to_vec());
        assert!(g.speedup > 1.0);
        // consistent exact evidence is fine
        let mut e2 = ev();
        e2.kld_state = Some(kld("promotion", true, 0.0));
        assert!(can_promote(&e2, &p).is_ok());
    }

    #[test]
    fn identical_with_non_exact_or_inconsistent_kld_refuses() {
        let mut e = ev();
        e.kld_state = Some(kld("promotion", false, 0.0));
        assert!(kld_reason(&e).contains("exact=false"));
        let mut e = ev();
        let mut k = kld("promotion", true, 0.0);
        k.candidate.scoring_mode = "decode".into();
        e.kld_state = Some(k);
        assert!(kld_reason(&e).contains("scoring_mode"));
    }

    #[test]
    fn numeric_requires_evidence_and_promotion_grade() {
        assert!(kld_reason(&numeric(None)).contains("no pass without evidence"));
        let r = kld_reason(&numeric(Some(kld("screen", false, 0.0))));
        assert!(r.contains("screening"), "{r}");
    }

    #[test]
    fn numeric_consistency_checks_each_refuse() {
        type Mutator = Box<dyn Fn(&mut KldStateEvidenceV1)>;
        let cases: Vec<(&str, Mutator)> = vec![
            (
                "scoring_mode",
                Box::new(|k| k.incumbent.scoring_mode = "decode".into()),
            ),
            (
                "reference_sha256",
                Box::new(|k| k.candidate.reference_sha256 = "x".into()),
            ),
            (
                "model_sha256",
                Box::new(|k| k.candidate.model_sha256 = "x".into()),
            ),
            (
                "evaluator_sha256",
                Box::new(|k| k.candidate.evaluator_sha256 = "x".into()),
            ),
            ("max_chunks", Box::new(|k| k.candidate.max_chunks = 40)),
            ("kv_mode", Box::new(|k| k.candidate.kv_mode = "x".into())),
            ("kv_v", Box::new(|k| k.candidate.kv_v = "x".into())),
            (
                "HIPFIRE_GRAPH",
                Box::new(|k| {
                    k.candidate.env.insert("HIPFIRE_GRAPH".into(), "1".into());
                }),
            ),
            (
                "HIPFIRE_NORMALIZE_PROMPT",
                Box::new(|k| {
                    k.incumbent.env.remove("HIPFIRE_NORMALIZE_PROMPT");
                }),
            ),
            ("fresh_home", Box::new(|k| k.incumbent.fresh_home = false)),
            (
                "tune table",
                Box::new(|k| k.incumbent.tune_table_path = Some("t.json".into())),
            ),
            (
                "selected_code_object_sha256",
                Box::new(|k| {
                    k.candidate.selected_code_object_sha256 = Some("other".into());
                }),
            ),
            (
                "selected_code_object_sha256",
                Box::new(|k| {
                    k.candidate.selected_code_object_sha256 = None;
                }),
            ),
            (
                "min chunks",
                Box::new(|k| {
                    k.incumbent.max_chunks = 8;
                    k.candidate.max_chunks = 8;
                }),
            ),
            ("paired.n", Box::new(|k| k.paired.n = 8)),
        ];
        for (needle, m) in cases {
            let mut k = kld("promotion", false, 0.0);
            m(&mut k);
            let r = kld_reason(&numeric(Some(k)));
            let needle = if needle == "min chunks" {
                "max_chunks"
            } else {
                needle
            };
            assert!(r.contains(needle), "{needle}: {r}");
        }
    }

    #[test]
    fn numeric_margin_gate() {
        let p = ClaimPolicyV1::default();
        let ok = numeric(Some(kld("promotion", false, 0.0005)));
        assert!(is_winner(&ok, &p));
        let g = can_promote(&ok, &p).unwrap();
        assert_eq!(g.passed, STAGES.to_vec());
        let r = kld_reason(&numeric(Some(kld("promotion", false, 0.0006))));
        assert!(r.contains("ci95_hi"), "{r}");
        assert!(kld_state_stage(&numeric(Some(kld("promotion", false, 0.0))), &p).is_ok());
    }

    #[test]
    fn each_stage_refuses_at_itself() {
        let p = ClaimPolicyV1::default();
        let mut e = ev();
        e.discovery.m7_status = Some("fail".into());
        assert_eq!(stage_of(&e), "m7");
        assert!(!is_winner(&e, &p));

        let mut e = ev();
        e.discovery.identical_to_baseline = Some(false);
        let strict = ClaimPolicyV1 {
            admit_numeric: false,
            ..ClaimPolicyV1::default()
        };
        let r = can_promote(&e, &strict).unwrap_err();
        assert_eq!(r.stage, "correctness");
        assert!(r.reason.contains("NUMERIC candidates are not admitted"));
        let mut e = ev();
        e.discovery.identical_to_baseline = None;
        assert_eq!(stage_of(&e), "correctness");
        let mut e = ev();
        e.discovery.correctness_pass = Some(false);
        assert_eq!(stage_of(&e), "correctness");

        let mut e = ev();
        e.discovery.separable = Some(false);
        assert_eq!(stage_of(&e), "abba");
        let mut e = ev();
        e.discovery.direction = Some("slower".into());
        e.discovery.block_ratio = Some(1.05);
        assert_eq!(stage_of(&e), "abba");
        let mut e = ev();
        e.discovery.block_ratio = Some(0.998);
        assert_eq!(stage_of(&e), "abba");

        let mut e = ev();
        e.noise_floor = None;
        assert_eq!(stage_of(&e), "noise_floor");
        let mut e = ev();
        e.aa_runs = 2;
        assert_eq!(stage_of(&e), "noise_floor");
        let mut e = ev();
        e.noise_floor = Some(0.2);
        assert_eq!(stage_of(&e), "noise_floor");

        let mut e = ev();
        e.confirms.clear();
        assert_eq!(stage_of(&e), "confirm");
        let mut e = ev();
        e.confirms[0].direction = Some("slower".into());
        e.confirms[0].block_ratio = Some(1.05);
        assert_eq!(stage_of(&e), "confirm");
        let mut e = ev();
        e.confirms[0].candidate_code_sha256 = Some("other".into());
        assert_eq!(stage_of(&e), "confirm");
    }

    fn stratum(label: &str, refsha: &str, n_chunks: usize) -> KldStratumV2 {
        let mut i = run("incumbent");
        let mut c = run("candidate");
        for r in [&mut i, &mut c] {
            r.reference_sha256 = refsha.into();
            r.ktok_path = Some("t".into());
            r.ktok_sha256 = Some("t".into());
        }
        KldStratumV2 {
            label: label.into(),
            incumbent: i,
            candidate: c,
            n_chunks,
            tokens_per_chunk: 1023,
            mean_diff: -0.0003,
            sd_token: 0.05,
            sd_chunk: 0.004,
            deff: 1.0,
            top1pct_abs_share: 0.1,
            exact: false,
        }
    }

    fn token_ev(grade: &str, n: usize, hi: f64, exact: bool) -> KldTokenEvidenceV2 {
        KldTokenEvidenceV2 {
            schema: crate::tune::KLD_TOKEN_SCHEMA.into(),
            grade: grade.into(),
            strata: vec![stratum("wt2", "a", n), stratum("ag", "b", n)],
            stat: TokenBootstrapV2 {
                stat: crate::tune::KLD_TOKEN_STAT.into(),
                n_strata: 2,
                n_chunks: 2 * n,
                n_tokens: 2 * n * 1023,
                point: if exact { 0.0 } else { -0.0003 },
                se: 0.0004,
                boot_hi95: hi,
                normal_hi95: hi,
                boot_b: 20_000,
                seed: 1,
                sd_token: 0.05,
                sd_chunk: 0.004,
                top1pct_abs_share: 0.1,
                exact,
            },
        }
    }

    fn numeric_token(t: KldTokenEvidenceV2) -> TuneEvidenceV1 {
        let mut e = numeric(None);
        e.kld_token = Some(t);
        e
    }

    #[test]
    fn token_bootstrap_decides_the_numeric_verdict() {
        let p = ClaimPolicyV1::default();
        // Both bounds below the margin, 24 chunks per stratum: promotes even though
        // the R7 per-chunk evidence is absent (and would refuse at this sd).
        assert!(can_promote(&numeric_token(token_ev("promotion", 24, 0.0004, false)), &p).is_ok());
        // Either bound at/above the margin refuses with the numbers.
        let r = kld_reason(&numeric_token(token_ev("promotion", 24, 0.0005, false)));
        assert!(r.contains("boot_hi95") && r.contains("0.0005"), "{r}");
        let mut t = token_ev("promotion", 24, 0.0004, false);
        t.stat.normal_hi95 = 0.0007;
        assert!(kld_reason(&numeric_token(t)).contains("normal_hi95"));
        // Per-stratum floor.
        let r = kld_reason(&numeric_token(token_ev("promotion", 19, 0.0, false)));
        assert!(r.contains("per-stratum floor"), "{r}");
        // Screening grade never promotes.
        let r = kld_reason(&numeric_token(token_ev("screen", 24, 0.0, false)));
        assert!(r.contains("screening"), "{r}");
        // Exact identity of the per-token diffs passes as in R7.
        assert!(can_promote(&numeric_token(token_ev("promotion", 24, 0.0, true)), &p).is_ok());
        // kld_token takes precedence over a refusing kld_state, never the reverse.
        let mut e = numeric_token(token_ev("promotion", 24, 0.0004, false));
        e.kld_state = Some(kld("promotion", false, 0.5));
        assert!(can_promote(&e, &p).is_ok());
    }

    #[test]
    fn token_bootstrap_consistency_checks_each_refuse() {
        type Mutator = Box<dyn Fn(&mut KldTokenEvidenceV2)>;
        let cases: Vec<(&str, Mutator)> = vec![
            (
                "scoring_mode",
                Box::new(|t| t.strata[1].incumbent.scoring_mode = "decode".into()),
            ),
            (
                "repeats",
                Box::new(|t| {
                    t.strata[1].incumbent.reference_sha256 = "a".into();
                    t.strata[1].candidate.reference_sha256 = "a".into();
                }),
            ),
            (
                "model_sha256",
                Box::new(|t| {
                    t.strata[1].incumbent.model_sha256 = "x".into();
                    t.strata[1].candidate.model_sha256 = "x".into();
                }),
            ),
            (
                "ktok_path",
                Box::new(|t| t.strata[0].candidate.ktok_path = None),
            ),
            (
                "selected_code_object_sha256",
                Box::new(|t| t.strata[1].candidate.selected_code_object_sha256 = Some("zz".into())),
            ),
            ("label", Box::new(|t| t.strata[1].label = "wt2".into())),
            ("no strata", Box::new(|t| t.strata.clear())),
        ];
        for (name, m) in cases {
            let mut t = token_ev("promotion", 24, 0.0, false);
            m(&mut t);
            let r = kld_reason(&numeric_token(t));
            assert!(r.contains(name) || r.contains("strata"), "{name}: {r}");
        }
    }

    #[test]
    fn identical_path_checks_attached_token_evidence() {
        let p = ClaimPolicyV1::default();
        let mut e = ev();
        e.kld_token = Some(token_ev("promotion", 24, 0.0, true));
        assert!(can_promote(&e, &p).is_ok());
        let mut e = ev();
        e.kld_token = Some(token_ev("promotion", 24, 0.0, false));
        assert!(kld_reason(&e).contains("exact=false"));
    }

    #[test]
    fn frozen_v1_serialization_omits_the_r9_fields() {
        let e = ev();
        let s = serde_json::to_string(&e).unwrap();
        assert!(!s.contains("kld_token"), "{s}");
        let mut r = run("incumbent");
        let s = serde_json::to_string(&r).unwrap();
        assert!(!s.contains("ktok"), "{s}");
        // Old records (no ktok fields) still load; new ones round trip.
        let back: KldRunV1 = serde_json::from_str(&s).unwrap();
        assert_eq!(back, r);
        r.ktok_path = Some("p".into());
        r.ktok_sha256 = Some("h".into());
        let back: KldRunV1 = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(back, r);
        let back: TuneEvidenceV1 = serde_json::from_str(&s_ev(&e)).unwrap();
        assert_eq!(back, e);
    }

    fn s_ev(e: &TuneEvidenceV1) -> String {
        serde_json::to_string(e).unwrap()
    }

    fn rc_class(rows: u64, ratio: f64, pf: f64, ps: f64) -> Value {
        serde_json::json!({
            "rows": rows, "shape": "M=128,K=2560", "identical_to_baseline": true,
            "timing": {"ratio_median": ratio, "p_faster": pf, "p_slower": ps, "rounds": []},
            "admitted": {"baseline": true, "candidate": true},
        })
    }

    fn rc_record(classes: Vec<Value>) -> Value {
        serde_json::json!({
            "node": "contract:generic_gemm.f32_tiled_rows.mpw16",
            "vs": "contract:generic_gemm.f32_batched.wave_per_output",
            "arch": "gfx1201", "regime": "hot", "bytes_per_call": 1.0e6, "rowclasses": classes,
        })
    }

    #[test]
    fn rowclass_veto_and_sweep_requirement_in_promotion() {
        let p = ClaimPolicyV1::default();
        let e = ev();
        let recs = |r: Value| vec![("discovery".to_owned(), r)];
        // Non-contract records keep today's behaviour: no records, no gate.
        assert!(can_promote_with_records(&e, &p, &[]).is_ok());
        let plain = serde_json::json!({"node": "mq6_res", "arch": "gfx1201"});
        assert!(can_promote_with_records(&e, &p, &recs(plain)).is_ok());

        // Faster everywhere admitted: promoted, and the stage is listed.
        let good = rc_record(vec![
            rc_class(1, 0.8, 1e-4, 1.0),
            rc_class(512, 0.8, 1e-4, 1.0),
        ]);
        let g = can_promote_with_records(&e, &p, &recs(good.clone())).unwrap();
        assert_eq!(g.passed[2], ROWCLASS_STAGE);
        assert_eq!(g.passed.len(), 7);
        assert!(is_winner_with_records(&e, &p, &recs(good)));

        // Any admitted Slower class vetoes, naming the class.
        let bad = rc_record(vec![
            rc_class(1, 1.6, 1.0, 1e-4),
            rc_class(512, 0.8, 1e-4, 1.0),
        ]);
        let r = can_promote_with_records(&e, &p, &recs(bad.clone())).unwrap_err();
        assert_eq!(r.stage, ROWCLASS_STAGE);
        assert_eq!(r.passed, STAGES[..2].to_vec());
        assert!(
            r.reason.contains("rows=1 ") && r.reason.contains("veto"),
            "{}",
            r.reason
        );
        assert!(!is_winner_with_records(&e, &p, &recs(bad)));

        // Regime refusal flows through the gate.
        let mut big = rc_record(vec![rc_class(512, 0.8, 1e-4, 1.0)]);
        big["bytes_per_call"] = serde_json::json!(1u64 << 30);
        let r = can_promote_with_records(&e, &p, &recs(big)).unwrap_err();
        assert!(r.reason.contains("last-level cache"), "{}", r.reason);

        // A single shape for a contract admitted at many classes needs a sweep.
        let mut single = rc_record(vec![]);
        single.as_object_mut().unwrap().remove("rowclasses");
        let r = can_promote_with_records(&e, &p, &recs(single.clone())).unwrap_err();
        assert_eq!(r.stage, ROWCLASS_STAGE);
        assert!(r.reason.contains("no row-class sweep"), "{}", r.reason);
        let off = ClaimPolicyV1 {
            require_rowclasses: false,
            ..p.clone()
        };
        assert!(can_promote_with_records(&e, &off, &recs(single)).is_ok());

        // Later-stage refusals list the row-class stage once it ran.
        let mut late = ev();
        late.discovery.sign_p = Some(0.5);
        let ok = rc_record(vec![rc_class(512, 0.8, 1e-4, 1.0)]);
        let r = can_promote_with_records(&late, &p, &recs(ok)).unwrap_err();
        assert_eq!(r.stage, "abba");
        assert_eq!(r.passed, vec!["m7", "correctness", ROWCLASS_STAGE]);
    }

    #[test]
    fn rowclass_records_are_read_from_record_path() {
        let dir = std::env::temp_dir().join(format!("pmrc-promote-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rec.json");
        let bad = rc_record(vec![
            rc_class(1, 1.6, 1.0, 1e-4),
            rc_class(512, 0.8, 1e-4, 1.0),
        ]);
        std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
        let mut e = ev();
        e.discovery.record_path = path.to_str().unwrap().to_owned();
        let p = ClaimPolicyV1::default();
        assert_eq!(can_promote(&e, &p).unwrap_err().stage, ROWCLASS_STAGE);
        assert!(!is_winner(&e, &p));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn moe_ev(shape: &str) -> TuneEvidenceV1 {
        let mut e = ev();
        e.candidate.kernel = "qwen4_moe_sym".into();
        e.candidate.shape = shape.into();
        e
    }

    fn moe_rec(route: Value) -> Value {
        serde_json::json!({"kernel": "qwen4_moe_sym", "shape": {"E": 256, "route": route}})
    }

    #[test]
    fn uniform_only_moe_evidence_is_refused() {
        let p = ClaimPolicyV1::default();
        // Textual shape on the candidate (no records readable).
        let r = can_promote_with_records(&moe_ev("a3b,T=8192,route=uniform"), &p, &[]).unwrap_err();
        assert_eq!(r.stage, ROUTE_STAGE);
        assert!(
            r.reason.contains("uniform-only route evidence"),
            "{}",
            r.reason
        );
        assert_eq!(r.passed, STAGES[..2].to_vec());
        // Per-attempt records: shape.route string and an object `kind`.
        let uni = |v: Value| {
            vec![
                ("discovery".to_owned(), moe_rec(v.clone())),
                ("confirm[0]".to_owned(), moe_rec(v)),
            ]
        };
        let e = moe_ev("a3b,T=8192");
        for route in [
            serde_json::json!("uniform"),
            serde_json::json!({"kind": "uniform"}),
        ] {
            let r = can_promote_with_records(&e, &p, &uni(route)).unwrap_err();
            assert_eq!(r.stage, ROUTE_STAGE);
        }
        // A `route` shape key makes the evidence MoE even if the kernel name is not.
        let mut k = ev();
        k.candidate.shape = "route=uniform".into();
        assert_eq!(
            can_promote_with_records(&k, &p, &[]).unwrap_err().stage,
            ROUTE_STAGE
        );
        assert!(!is_winner_with_records(
            &e,
            &p,
            &uni(serde_json::json!("uniform"))
        ));
    }

    #[test]
    fn hist_or_skew_moe_evidence_passes_the_route_gate() {
        let p = ClaimPolicyV1::default();
        for route in [
            serde_json::json!("hist:/x/route_hist.json"),
            serde_json::json!({"kind": "hist", "path": "/x/route_hist.json", "id": "ab", "layer": null}),
            serde_json::json!("skew"),
            serde_json::json!("skew5"),
        ] {
            let recs = vec![
                ("discovery".to_owned(), moe_rec(route.clone())),
                ("confirm[0]".to_owned(), moe_rec(route.clone())),
            ];
            let g = can_promote_with_records(&moe_ev("a3b"), &p, &recs).unwrap();
            assert_eq!(g.passed[2], ROUTE_STAGE, "{route}");
        }
        assert!(can_promote_with_records(&moe_ev("a3b,T=8192,route=hist:f.json"), &p, &[]).is_ok());
        // One non-uniform attempt is enough: not "every attempt uniform".
        let mixed = vec![
            (
                "discovery".to_owned(),
                moe_rec(serde_json::json!("uniform")),
            ),
            (
                "confirm[0]".to_owned(),
                moe_rec(serde_json::json!("hist:f.json")),
            ),
        ];
        assert!(can_promote_with_records(&moe_ev("a3b"), &p, &mixed).is_ok());
        // Non-MoE evidence is untouched, even with no route anywhere.
        let g = can_promote_with_records(&ev(), &p, &[]).unwrap();
        assert!(!g.passed.contains(&ROUTE_STAGE.to_owned()));
    }

    #[test]
    fn route_gate_precedes_rowclass_and_later_stages_list_it() {
        let p = ClaimPolicyV1::default();
        let mut late = moe_ev("route=skew");
        late.discovery.sign_p = Some(0.5);
        let r = can_promote_with_records(&late, &p, &[]).unwrap_err();
        assert_eq!(r.stage, "abba");
        assert_eq!(r.passed, vec!["m7", "correctness", ROUTE_STAGE]);
    }

    #[test]
    fn policy_defaults_for_rowclass_fields_load_from_old_json() {
        let p: ClaimPolicyV1 = serde_json::from_str(
            r#"{"sign_p_max":0.001,"min_effect":0.005,"confirm_processes":1,"aa_processes":3}"#,
        )
        .unwrap();
        assert!(p.require_rowclasses && p.shares.is_empty());
        let q: ClaimPolicyV1 = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(p, q);
    }
}
