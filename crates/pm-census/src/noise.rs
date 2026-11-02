// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Measurement-noise harness over `pm_loop` records (`pm-census noise`).
//!
//! `pm_loop` times a candidate against a baseline with ABBA blocks inside ONE
//! process and reports an exact paired sign test over its rounds. That test
//! answers "is this process's A/B difference larger than its own round-to-round
//! scatter", not "would a fresh process see the same effect". This module turns
//! many `pm_loop` records into the evidence a promotion decision needs.
//!
//! # Why fresh-process runs
//!
//! Everything that is constant inside one process is invisible to the in-process
//! test: the placement of the two loaded code objects, the allocator's buffer
//! addresses, the clock/power state the card settled into, the order effects of
//! one particular schedule. Two byte-identical objects have come out
//! "separable" at 0.2-0.3% inside a single process. Those effects are redrawn
//! by every new process, so only independent processes can estimate them. Each
//! input record is therefore one run (one process), and every count below is a
//! count of runs, never of rounds. Rounds inside one run are not independent
//! evidence of the process-level effect; they only feed the pooled round-ratio
//! dispersion, which describes within-run scatter.
//!
//! # Why the floor comes from A/A
//!
//! An A/A run (`pm_loop --null-control aa`, or any record whose candidate
//! object is byte-identical to the baseline) has a true effect of exactly 1.0.
//! The distribution of `|block_ratio - 1|` over fresh-process A/A runs is the
//! pure measurement-and-placement noise of that kernel, shape and baseline
//! object. The `floor` is the 95th percentile of that distribution (linear
//! interpolation between order statistics); `false_separable_rate` and
//! `false_material_rate` are the fractions of A/A runs that `pm_loop` itself
//! called separable / material, i.e. its empirical false-positive rate on this
//! workload. The floor is estimated only from `>= 3` runs; below that it is
//! `None` with `status: "insufficient_runs"`: it is never guessed. A/A noise is
//! kernel and shape specific, so an A/B group is only judged against an A/A
//! group with the same kernel, canonical shape JSON and baseline sha256.
//!
//! # A/B verdicts
//!
//! For an A/B group the run directions (`block_ratio < 1` is faster) get an
//! exact two-sided binomial sign test (`min(1, 2 * P(X <= n - majority))`,
//! `X ~ Binomial(n, 1/2)`, ties against the majority: 3 same-direction runs
//! give p = 0.25, 11 give p = 2/2048). The verdict, in order:
//!
//! * `insufficient_runs`: fewer than 3 fresh-process runs;
//! * `no_null_control`: no A/A group with the same kernel, shape and baseline
//!   sha256 and a floor;
//! * `inconsistent`: the runs do not all point the same way;
//! * `within_noise`: `|median_effect| <= floor` (`median_effect` is the median
//!   block ratio minus 1);
//! * `above_floor`: all runs same direction, `|median_effect| > floor`, and
//!   every run `material` under `pm_loop`'s own gate;
//! * `not_material`: same direction and above the floor, but at least one run
//!   missed `pm_loop`'s `--min-effect` gate.
//!
//! # Repeated looks (what the p-values are not)
//!
//! `pm_loop` keeps adding ABBA rounds until its test separates, a round cap or
//! a time budget (an adaptive stopping rule), and an autotuner looks at many
//! candidates. A per-run `p <= 1e-3` is therefore not a family-wide
//! false-positive bound, and the sign-test p of this module is likewise a
//! per-group number: it is not corrected for the number of candidates or looks.
//! The A/A rates and floor measure the realised false-positive behaviour of the
//! same procedure on identical objects, which is the control the adaptive rule
//! needs; a confirmatory, pre-registered fresh-process run set is still
//! required before a family-wide claim.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const NOISE_SCHEMA: &str = "hipfire.pm_noise.v1";
/// Fewest fresh-process runs for a floor or an A/B verdict.
pub const MIN_RUNS: usize = 3;
/// Quantile of the A/A `|ratio - 1|` distribution that defines the floor.
const FLOOR_QUANTILE: f64 = 0.95;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Linear-interpolation quantile (`q` in `[0, 1]`, position `q * (n - 1)`) of
/// `v`; `None` for an empty slice or a non-finite value.
pub fn quantile(v: &[f64], q: f64) -> Option<f64> {
    if v.is_empty() || v.iter().any(|x| !x.is_finite()) || !(0.0..=1.0).contains(&q) {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let pos = q * (s.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    Some(s[lo] + (s[hi] - s[lo]) * (pos - lo as f64))
}

/// Exact two-sided sign-test p of `majority` same-direction runs out of `n`:
/// `min(1, 2 * P(X <= n - majority))`, `X ~ Binomial(n, 1/2)`. Runs that are not
/// in the majority (other direction and ties) count against it.
pub fn sign_test_p(majority: usize, n: usize) -> f64 {
    if n == 0 || majority > n {
        return 1.0;
    }
    let tail = n - majority;
    let mut term = 0.5f64.powi(n as i32);
    let mut sum = term;
    for i in 0..tail {
        term *= (n - i) as f64 / (i + 1) as f64;
        sum += term;
    }
    (2.0 * sum).min(1.0)
}

/// Dispersion of positive finite values; same definition as the `pm_loop`
/// `noise` fields (MAD unscaled, linear-interpolated quantiles, `log_sd` the
/// sample sd of `ln v`).
#[derive(Clone, Debug, PartialEq)]
pub struct Dispersion {
    pub n: usize,
    pub median: f64,
    pub mad: f64,
    pub iqr: f64,
    pub p05: f64,
    pub p95: f64,
    pub min: f64,
    pub max: f64,
    pub cv: f64,
    pub log_sd: f64,
}

impl Dispersion {
    pub fn of(v: &[f64]) -> Result<Dispersion, String> {
        if v.is_empty() || v.iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return Err("dispersion values must be nonempty, positive and finite".into());
        }
        let q = |xs: &[f64], p: f64| quantile(xs, p).expect("validated nonempty finite");
        let sd = |xs: &[f64]| {
            if xs.len() < 2 {
                return 0.0;
            }
            let m = xs.iter().sum::<f64>() / xs.len() as f64;
            (xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() - 1) as f64).sqrt()
        };
        let median = q(v, 0.5);
        let dev: Vec<f64> = v.iter().map(|x| (x - median).abs()).collect();
        let logs: Vec<f64> = v.iter().map(|x| x.ln()).collect();
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let d = Dispersion {
            n: v.len(),
            median,
            mad: q(&dev, 0.5),
            iqr: q(v, 0.75) - q(v, 0.25),
            p05: q(v, 0.05),
            p95: q(v, 0.95),
            min: q(v, 0.0),
            max: q(v, 1.0),
            cv: sd(v) / mean,
            log_sd: sd(&logs),
        };
        if !(d.cv.is_finite() && d.log_sd.is_finite()) {
            return Err("nonfinite dispersion".into());
        }
        Ok(d)
    }

    pub fn json(&self) -> Value {
        json!({"n": self.n, "median": self.median, "mad": self.mad, "iqr": self.iqr,
            "p05": self.p05, "p95": self.p95, "min": self.min, "max": self.max,
            "cv": self.cv, "log_sd": self.log_sd})
    }
}

/// One fresh-process `pm_loop` run.
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    /// `timing.block_ratio`: median candidate block / median baseline block.
    pub block_ratio: f64,
    pub separable: bool,
    pub material: bool,
    /// `timing.noise.round_ratios` (empty for records that predate the field).
    pub round_ratios: Vec<f64>,
}

impl Run {
    /// `"faster"` (`block_ratio < 1`), `"slower"` or `"tie"`.
    pub fn direction(&self) -> &'static str {
        if self.block_ratio < 1.0 {
            "faster"
        } else if self.block_ratio > 1.0 {
            "slower"
        } else {
            "tie"
        }
    }
}

/// Runs sharing kernel, canonical shape, baseline sha256, candidate sha256 and
/// null-control kind.
#[derive(Clone, Debug)]
pub struct Group {
    pub kernel: String,
    pub shape: Value,
    pub baseline_sha256: String,
    pub candidate_sha256: String,
    /// `null_control.kind` of the records (`Some("aa")`), else `None`.
    pub null_control: Option<String>,
    pub runs: Vec<Run>,
}

/// Key-sorted, whitespace-free JSON: equal values give equal strings whatever
/// the map ordering of the parser.
pub fn canonical(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let body: Vec<String> = keys
                .into_iter()
                .map(|k| format!("{}:{}", Value::String(k.clone()), canonical(&m[k])))
                .collect();
            format!("{{{}}}", body.join(","))
        }
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(canonical).collect::<Vec<_>>().join(",")
        ),
        other => other.to_string(),
    }
}

impl Group {
    fn key(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}",
            self.kernel,
            canonical(&self.shape),
            self.baseline_sha256,
            self.candidate_sha256,
            self.null_control.as_deref().unwrap_or("-")
        )
    }

    /// Stable group id: first 16 hex digits of the SHA-256 of the key.
    pub fn id(&self) -> String {
        hex(&Sha256::digest(self.key().as_bytes()))[..16].to_owned()
    }

    /// `null_control` is `aa`, or the candidate object is byte-identical to
    /// the baseline.
    pub fn is_aa(&self) -> bool {
        self.null_control.as_deref() == Some("aa") || self.candidate_sha256 == self.baseline_sha256
    }

    /// Same workload as `other` for noise purposes: kernel, canonical shape and
    /// baseline sha256.
    fn same_workload(&self, other: &Group) -> bool {
        self.kernel == other.kernel
            && self.baseline_sha256 == other.baseline_sha256
            && canonical(&self.shape) == canonical(&other.shape)
    }

    fn effects(&self) -> Vec<f64> {
        self.runs
            .iter()
            .map(|r| (r.block_ratio - 1.0).abs())
            .collect()
    }

    /// A/A floor: q95 of `|block_ratio - 1|` over runs (`None` below
    /// [`MIN_RUNS`]).
    pub fn floor(&self) -> Option<f64> {
        if self.runs.len() < MIN_RUNS {
            return None;
        }
        quantile(&self.effects(), FLOOR_QUANTILE)
    }

    fn pooled_round_ratios(&self) -> Vec<f64> {
        self.runs
            .iter()
            .flat_map(|r| r.round_ratios.iter().copied())
            .collect()
    }

    /// q95 of the pooled `|round_ratio - 1|` (`None` below [`MIN_RUNS`] runs or
    /// without round ratios).
    pub fn round_floor(&self) -> Option<f64> {
        if self.runs.len() < MIN_RUNS {
            return None;
        }
        let e: Vec<f64> = self
            .pooled_round_ratios()
            .iter()
            .map(|r| (r - 1.0).abs())
            .collect();
        quantile(&e, FLOOR_QUANTILE)
    }

    fn json_common(&self) -> serde_json::Map<String, Value> {
        let n = self.runs.len();
        let pooled = self.pooled_round_ratios();
        let mut m = serde_json::Map::new();
        m.insert("id".into(), json!(self.id()));
        m.insert("kernel".into(), json!(self.kernel));
        m.insert("shape".into(), self.shape.clone());
        m.insert("baseline_sha256".into(), json!(self.baseline_sha256));
        m.insert("candidate_sha256".into(), json!(self.candidate_sha256));
        m.insert("null_control".into(), json!(self.null_control));
        m.insert("aa".into(), json!(self.is_aa()));
        m.insert("runs".into(), json!(n));
        m.insert(
            "separable_runs".into(),
            json!(self.runs.iter().filter(|r| r.separable).count()),
        );
        m.insert(
            "material_runs".into(),
            json!(self.runs.iter().filter(|r| r.material).count()),
        );
        m.insert(
            "block_ratios".into(),
            json!(self.runs.iter().map(|r| r.block_ratio).collect::<Vec<_>>()),
        );
        m.insert(
            "directions".into(),
            json!(self.runs.iter().map(Run::direction).collect::<Vec<_>>()),
        );
        m.insert(
            "round_ratio".into(),
            Dispersion::of(&pooled)
                .map(|d| d.json())
                .unwrap_or(Value::Null),
        );
        m
    }

    fn json_aa(&self) -> Value {
        let mut m = self.json_common();
        let n = self.runs.len() as f64;
        let rate = |c: usize| c as f64 / n;
        m.insert(
            "false_separable_rate".into(),
            json!(rate(self.runs.iter().filter(|r| r.separable).count())),
        );
        m.insert(
            "false_material_rate".into(),
            json!(rate(self.runs.iter().filter(|r| r.material).count())),
        );
        m.insert("floor".into(), json!(self.floor()));
        m.insert("round_floor".into(), json!(self.round_floor()));
        m.insert(
            "status".into(),
            json!(if self.runs.len() < MIN_RUNS {
                "insufficient_runs"
            } else {
                "ok"
            }),
        );
        Value::Object(m)
    }

    fn json_ab(&self, all: &[Group]) -> Value {
        let mut m = self.json_common();
        let n = self.runs.len();
        let faster = self
            .runs
            .iter()
            .filter(|r| r.direction() == "faster")
            .count();
        let slower = self
            .runs
            .iter()
            .filter(|r| r.direction() == "slower")
            .count();
        let ratios: Vec<f64> = self.runs.iter().map(|r| r.block_ratio).collect();
        let median_effect = quantile(&ratios, 0.5).map(|m| m - 1.0);
        // The widest floor among the matching A/A groups (conservative).
        let control = all
            .iter()
            .filter(|g| g.is_aa() && g.same_workload(self))
            .filter_map(|g| g.floor().map(|f| (f, g.id())))
            .max_by(|a, b| a.0.total_cmp(&b.0));
        let verdict = if n < MIN_RUNS {
            "insufficient_runs"
        } else if control.is_none() {
            "no_null_control"
        } else if faster != n && slower != n {
            "inconsistent"
        } else {
            let floor = control.as_ref().map(|c| c.0).unwrap_or(f64::NAN);
            let effect = median_effect.map(f64::abs).unwrap_or(f64::NAN);
            if effect <= floor {
                "within_noise"
            } else if self.runs.iter().all(|r| r.material) {
                "above_floor"
            } else {
                "not_material"
            }
        };
        m.insert("faster_runs".into(), json!(faster));
        m.insert("slower_runs".into(), json!(slower));
        m.insert("sign_p".into(), json!(sign_test_p(faster.max(slower), n)));
        m.insert("median_effect".into(), json!(median_effect));
        m.insert("floor".into(), json!(control.as_ref().map(|c| c.0)));
        m.insert("null_control_group".into(), json!(control.map(|c| c.1)));
        m.insert("verdict".into(), json!(verdict));
        Value::Object(m)
    }
}

fn str_at<'a>(rec: &'a Value, path: &[&str]) -> Option<&'a str> {
    path.iter().try_fold(rec, |v, k| v.get(*k))?.as_str()
}

/// The group key fields and the run of one `pm_loop` record.
fn parse_record(rec: &Value) -> Result<(Group, Run), String> {
    if rec["schema"] != 1 {
        return Err("not a pm_loop schema-1 record".into());
    }
    let kernel = rec["kernel"].as_str().ok_or("no kernel")?.to_owned();
    if rec["shape"].is_null() {
        return Err("no shape".into());
    }
    let sha = |who: &str| -> Result<String, String> {
        str_at(rec, &[who, "co_sha256"])
            .or_else(|| str_at(rec, &[who, "sha256"]))
            .map(str::to_owned)
            .ok_or_else(|| format!("no {who} sha256"))
    };
    let t = &rec["timing"];
    let block_ratio = t["block_ratio"]
        .as_f64()
        .filter(|r| r.is_finite() && *r > 0.0)
        .ok_or("no timing.block_ratio (run did not reach timing)")?;
    let mut round_ratios = Vec::new();
    if let Some(a) = t["noise"]["round_ratios"].as_array() {
        for v in a {
            round_ratios.push(
                v.as_f64()
                    .filter(|r| r.is_finite() && *r > 0.0)
                    .ok_or("timing.noise.round_ratios has a non-positive or non-numeric entry")?,
            );
        }
    }
    let group = Group {
        kernel,
        shape: rec["shape"].clone(),
        baseline_sha256: sha("baseline")?,
        candidate_sha256: sha("candidate")?,
        null_control: str_at(rec, &["null_control", "kind"]).map(str::to_owned),
        runs: Vec::new(),
    };
    Ok((
        group,
        Run {
            block_ratio,
            separable: t["separable"] == true,
            material: t["material"] == true,
            round_ratios,
        },
    ))
}

/// Records of one input: a single JSON object, or JSON lines.
pub fn parse_records(text: &str) -> Result<Vec<Value>, String> {
    if let Ok(v) = serde_json::from_str::<Value>(text) {
        return match v {
            Value::Object(_) => Ok(vec![v]),
            _ => Err("expected a JSON object or JSON lines of objects".into()),
        };
    }
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        if !v.is_object() {
            return Err(format!("line {}: not a JSON object", i + 1));
        }
        out.push(v);
    }
    if out.is_empty() {
        return Err("no records".into());
    }
    Ok(out)
}

/// Group the records of `inputs` (`(path, bytes)`); records that did not reach
/// timing are reported in `skipped`, never silently dropped.
pub fn analyze(inputs: &[(String, Vec<u8>)]) -> Result<Value, String> {
    let mut groups: BTreeMap<String, Group> = BTreeMap::new();
    let mut skipped = Vec::new();
    let mut input_rows = Vec::new();
    for (path, bytes) in inputs {
        input_rows.push(json!({"path": path, "sha256": hex(&Sha256::digest(bytes))}));
        let text = std::str::from_utf8(bytes).map_err(|e| format!("{path}: {e}"))?;
        for (i, rec) in parse_records(text)
            .map_err(|e| format!("{path}: {e}"))?
            .iter()
            .enumerate()
        {
            match parse_record(rec) {
                Ok((g, run)) => groups.entry(g.key()).or_insert(g).runs.push(run),
                Err(reason) => skipped.push(json!({"path": path, "record": i, "reason": reason})),
            }
        }
    }
    let all: Vec<Group> = groups.into_values().collect();
    let rows: Vec<Value> = all
        .iter()
        .map(|g| {
            if g.is_aa() {
                g.json_aa()
            } else {
                g.json_ab(&all)
            }
        })
        .collect();
    Ok(json!({"schema": NOISE_SCHEMA, "groups": rows, "skipped": skipped, "inputs": input_rows}))
}

/// `pm-census noise FILES... [--out FILE]`: prints the noise document and, with
/// `--out`, also writes it pretty-printed to FILE.
pub fn cli(args: &[String]) -> Result<(), String> {
    const USAGE: &str = "usage: pm-census noise FILES... [--out FILE]";
    let mut files = Vec::new();
    let mut out = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => {
                out = Some(
                    it.next()
                        .ok_or_else(|| format!("--out needs a value ({USAGE})"))?,
                )
            }
            f if f.starts_with("--") => return Err(format!("unknown flag {f} ({USAGE})")),
            f => files.push(f),
        }
    }
    if files.is_empty() {
        return Err(format!("no input files ({USAGE})"));
    }
    let inputs = files
        .into_iter()
        .map(|f| {
            std::fs::read(f)
                .map(|b| (f.to_owned(), b))
                .map_err(|e| format!("{f}: {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let doc = analyze(&inputs)?;
    if let Some(path) = out {
        let pretty = serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?;
        std::fs::write(path, pretty + "\n").map_err(|e| format!("{path}: {e}"))?;
    }
    println!(
        "{}",
        serde_json::to_string(&doc).map_err(|e| e.to_string())?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn rec(
        base: &str,
        cand: &str,
        nc: Option<&str>,
        ratio: f64,
        separable: bool,
        material: bool,
        rr: &[f64],
    ) -> Value {
        let mut v = json!({
            "schema": 1, "kernel": "iu4_gemm", "shape": {"M": 128, "N": 64},
            "baseline": {"co_sha256": base}, "candidate": {"co_sha256": cand},
            "timing": {"block_ratio": ratio, "separable": separable, "material": material,
                "noise": {"round_ratios": rr}},
        });
        if let Some(k) = nc {
            v["null_control"] = json!({"kind": k});
        }
        v
    }

    fn jsonl(recs: &[Value]) -> Vec<(String, Vec<u8>)> {
        let text: Vec<String> = recs.iter().map(|r| r.to_string()).collect();
        vec![("runs.jsonl".to_owned(), text.join("\n").into_bytes())]
    }

    fn aa_recs(ratios: &[f64]) -> Vec<Value> {
        ratios
            .iter()
            .map(|r| rec("b1", "b1", Some("aa"), *r, false, false, &[*r, *r]))
            .collect()
    }

    fn ab_recs(ratios: &[f64], material: &[bool]) -> Vec<Value> {
        ratios
            .iter()
            .zip(material)
            .map(|(r, m)| rec("b1", "c1", None, *r, true, *m, &[*r]))
            .collect()
    }

    fn verdict_of(recs: Vec<Value>) -> (String, Value) {
        let doc = analyze(&jsonl(&recs)).unwrap();
        let g = doc["groups"]
            .as_array()
            .unwrap()
            .iter()
            .find(|g| g["aa"] == false)
            .unwrap()
            .clone();
        (g["verdict"].as_str().unwrap().to_owned(), g)
    }

    #[test]
    fn sign_test_exact_values() {
        assert_eq!(sign_test_p(3, 3), 0.25);
        assert_eq!(sign_test_p(11, 11), 2.0 / 2048.0);
        // 2 of 3 in the majority: 2 * P(X <= 1) = 2 * 4/8 -> capped at 1.
        assert_eq!(sign_test_p(2, 3), 1.0);
        // 4 of 5: 2 * (1 + 5) / 32.
        assert_eq!(sign_test_p(4, 5), 12.0 / 32.0);
        assert_eq!(sign_test_p(0, 0), 1.0);
        assert_eq!(sign_test_p(5, 4), 1.0);
    }

    #[test]
    fn quantile_is_linear_interpolation() {
        let v = [5.0, 1.0, 4.0, 2.0, 3.0];
        assert_eq!(quantile(&v, 0.0), Some(1.0));
        assert_eq!(quantile(&v, 0.5), Some(3.0));
        assert_eq!(quantile(&v, 1.0), Some(5.0));
        assert!((quantile(&v, 0.95).unwrap() - 4.8).abs() < 1e-12);
        assert!((quantile(&v, 0.25).unwrap() - 2.0).abs() < 1e-12);
        assert_eq!(quantile(&[2.0, 4.0], 0.5), Some(3.0));
        assert_eq!(quantile(&[], 0.5), None);
        assert_eq!(quantile(&[1.0, f64::NAN], 0.5), None);
        assert_eq!(quantile(&[1.0], 1.5), None);
    }

    #[test]
    fn dispersion_known_values() {
        let d = Dispersion::of(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!((d.n, d.median, d.mad, d.iqr), (5, 3.0, 1.0, 2.0));
        assert!((d.p05 - 1.2).abs() < 1e-12 && (d.p95 - 4.8).abs() < 1e-12);
        assert_eq!((d.min, d.max), (1.0, 5.0));
        assert!((d.cv - 2.5f64.sqrt() / 3.0).abs() < 1e-12);
        let logs: Vec<f64> = (1..=5).map(|x| f64::from(x).ln()).collect();
        let m = logs.iter().sum::<f64>() / 5.0;
        let sd = (logs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / 4.0).sqrt();
        assert!((d.log_sd - sd).abs() < 1e-12);
        let one = Dispersion::of(&[2.0]).unwrap();
        assert_eq!((one.mad, one.cv, one.log_sd), (0.0, 0.0, 0.0));
        assert!(Dispersion::of(&[]).is_err());
        assert!(Dispersion::of(&[1.0, 0.0]).is_err());
    }

    #[test]
    fn aa_floor_is_q95_with_linear_interpolation() {
        // |effects| sorted: .001 .002 .003 .004 .010 -> position 3.8.
        let doc = analyze(&jsonl(&aa_recs(&[1.003, 0.998, 1.010, 0.999, 1.004]))).unwrap();
        let g = &doc["groups"][0];
        assert_eq!(g["aa"], true);
        let floor = g["floor"].as_f64().unwrap();
        assert!(
            (floor - (0.004 + (0.010 - 0.004) * 0.8)).abs() < 1e-12,
            "{floor}"
        );
        assert_eq!(g["status"], "ok");
        assert_eq!(g["runs"], 5);
        assert_eq!(g["false_separable_rate"], 0.0);
        assert_eq!(g["false_material_rate"], 0.0);
        // Pooled round ratios: 10 values, round floor is their own q95.
        assert_eq!(g["round_ratio"]["n"], 10);
        let effects: Vec<f64> = [1.003f64, 0.998, 1.010, 0.999, 1.004]
            .iter()
            .flat_map(|r| [(r - 1.0).abs(); 2])
            .collect();
        let rf = g["round_floor"].as_f64().unwrap();
        assert!((rf - quantile(&effects, 0.95).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn aa_false_positive_rates_and_insufficient_runs() {
        let mut recs = aa_recs(&[1.001, 1.002, 0.999, 1.004]);
        recs[0]["timing"]["separable"] = json!(true);
        recs[1]["timing"]["separable"] = json!(true);
        recs[1]["timing"]["material"] = json!(true);
        let doc = analyze(&jsonl(&recs)).unwrap();
        let g = &doc["groups"][0];
        assert_eq!(g["separable_runs"], 2);
        assert_eq!(g["material_runs"], 1);
        assert_eq!(g["false_separable_rate"], 0.5);
        assert_eq!(g["false_material_rate"], 0.25);
        let few = analyze(&jsonl(&aa_recs(&[1.001, 1.002]))).unwrap();
        let g = &few["groups"][0];
        assert!(g["floor"].is_null() && g["round_floor"].is_null());
        assert_eq!(g["status"], "insufficient_runs");
    }

    #[test]
    fn aa_detected_by_identical_object_without_null_control() {
        let recs: Vec<Value> = [1.001, 0.999, 1.002]
            .iter()
            .map(|r| rec("b1", "b1", None, *r, false, false, &[]))
            .collect();
        let doc = analyze(&jsonl(&recs)).unwrap();
        assert_eq!(doc["groups"][0]["aa"], true);
        assert!(doc["groups"][0]["floor"].is_number());
        assert!(doc["groups"][0]["round_ratio"].is_null());
    }

    #[test]
    fn verdicts() {
        let aa = aa_recs(&[1.001, 0.999, 1.002]); // floor 0.0019
        let join = |mut a: Vec<Value>, b: Vec<Value>| {
            a.extend(b);
            a
        };
        // insufficient_runs: two A/B runs.
        let (v, g) = verdict_of(join(aa.clone(), ab_recs(&[0.9, 0.9], &[true, true])));
        assert_eq!(v, "insufficient_runs");
        assert_eq!(g["runs"], 2);
        // no_null_control: no A/A group at all, or one with another baseline.
        let (v, _) = verdict_of(ab_recs(&[0.9, 0.9, 0.9], &[true; 3]));
        assert_eq!(v, "no_null_control");
        let other_base: Vec<Value> = [1.001, 0.999, 1.002]
            .iter()
            .map(|r| rec("b2", "b2", Some("aa"), *r, false, false, &[]))
            .collect();
        let (v, _) = verdict_of(join(other_base, ab_recs(&[0.9, 0.9, 0.9], &[true; 3])));
        assert_eq!(v, "no_null_control");
        // An A/A group with too few runs for a floor is no null control either.
        let (v, _) = verdict_of(join(
            aa_recs(&[1.001, 1.002]),
            ab_recs(&[0.9; 3], &[true; 3]),
        ));
        assert_eq!(v, "no_null_control");
        // inconsistent: mixed directions.
        let (v, _) = verdict_of(join(aa.clone(), ab_recs(&[0.95, 1.05, 0.96], &[true; 3])));
        assert_eq!(v, "inconsistent");
        // within_noise: same direction, median effect 0.001 <= floor 0.0019.
        let (v, g) = verdict_of(join(
            aa.clone(),
            ab_recs(&[0.9990, 0.9985, 0.9995], &[false; 3]),
        ));
        assert_eq!(v, "within_noise");
        assert!((g["median_effect"].as_f64().unwrap() + 0.001).abs() < 1e-12);
        assert!((g["floor"].as_f64().unwrap() - 0.0019).abs() < 1e-12);
        assert_eq!(g["sign_p"], 0.25);
        assert_eq!(g["faster_runs"], 3);
        // above_floor: same direction, effect above the floor, every run material.
        let (v, g) = verdict_of(join(aa.clone(), ab_recs(&[0.95, 0.96, 0.94], &[true; 3])));
        assert_eq!(v, "above_floor");
        assert!(g["null_control_group"].is_string());
        // not_material: above the floor but one run misses the pm_loop gate.
        let (v, _) = verdict_of(join(aa, ab_recs(&[0.95, 0.96, 0.94], &[true, false, true])));
        assert_eq!(v, "not_material");
    }

    #[test]
    fn larger_a_a_floor_wins_and_shape_key_order_is_canonical() {
        let mut recs = aa_recs(&[1.001, 0.999, 1.002]);
        // Same shape written with another key order, a second A/A group.
        let mut wide: Vec<Value> = [1.05, 0.95, 1.04]
            .iter()
            .map(|r| rec("b1", "b1", None, *r, false, false, &[]))
            .collect();
        for w in &mut wide {
            w["shape"] = serde_json::from_str(r#"{"N": 64, "M": 128}"#).unwrap();
        }
        recs.extend(wide);
        recs.extend(ab_recs(&[0.97, 0.97, 0.97], &[true; 3]));
        let (v, g) = verdict_of(recs);
        // A/A groups differ by null_control kind (two groups), the widest floor applies.
        assert!(g["floor"].as_f64().unwrap() > 0.03, "{g}");
        assert_eq!(v, "within_noise");
    }

    #[test]
    fn parses_single_object_and_jsonl_and_reports_skips() {
        let good = rec("b1", "c1", None, 0.9, true, true, &[0.9]);
        let single = vec![("one.json".to_owned(), good.to_string().into_bytes())];
        let doc = analyze(&single).unwrap();
        assert_eq!(doc["schema"], NOISE_SCHEMA);
        assert_eq!(doc["groups"][0]["runs"], 1);
        assert_eq!(doc["inputs"][0]["path"], "one.json");
        assert_eq!(doc["inputs"][0]["sha256"].as_str().unwrap().len(), 64);
        // A failed run (no timing) and a wrong schema are listed, not dropped.
        let mut failed = good.clone();
        failed["timing"] = json!({"status": "refused: candidate failed correctness"});
        let mut other = good.clone();
        other["schema"] = json!(2);
        let doc = analyze(&jsonl(&[good, failed, other])).unwrap();
        assert_eq!(doc["groups"][0]["runs"], 1);
        assert_eq!(doc["skipped"].as_array().unwrap().len(), 2);
        assert_eq!(doc["skipped"][0]["record"], 1);
        // Garbage is an error.
        let bad = vec![("bad".to_owned(), b"not json\n{}".to_vec())];
        assert!(analyze(&bad).is_err());
        assert!(analyze(&[("empty".to_owned(), Vec::new())]).is_err());
    }

    #[test]
    fn group_id_is_stable_and_key_sensitive() {
        let (mut a, _) = parse_record(&rec("b1", "c1", None, 0.9, true, true, &[])).unwrap();
        let id = a.id();
        assert_eq!(id.len(), 16);
        assert_eq!(id, a.id());
        a.candidate_sha256 = "c2".into();
        assert_ne!(id, a.id());
    }

    #[test]
    fn cli_writes_out_file_and_validates_args() {
        let dir = std::env::temp_dir().join(format!("pm_noise_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("runs.jsonl");
        let text: Vec<String> = aa_recs(&[1.001, 0.999, 1.002])
            .iter()
            .map(|r| r.to_string())
            .collect();
        std::fs::write(&input, text.join("\n")).unwrap();
        let out = dir.join("noise.json");
        let s = |p: &std::path::Path| p.display().to_string();
        cli(&[s(&input), "--out".into(), s(&out)]).unwrap();
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(doc["schema"], NOISE_SCHEMA);
        assert_eq!(doc["groups"][0]["aa"], true);
        assert!(cli(&[]).is_err());
        assert!(cli(&["--bogus".into()]).is_err());
        assert!(cli(&[s(&input), "--out".into()]).is_err());
        assert!(cli(&[s(&dir.join("missing.json"))]).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
