// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Row-class objective for contract-derived `hip_loop --rowclasses` records
//! (PM/rip-loop Slice 2, section B).
//!
//! A row-class record carries `rowclasses: [{rows, shape, identical_to_baseline,
//! timing: {ratio_median, p_faster, p_slower, rounds, ..}, admitted: {baseline,
//! candidate}}]` plus `regime` (`hot` | `pool`), `bytes_per_call`, `node`, `vs`
//! and `arch`. `ratio_median` is candidate / baseline, so `< 1` is faster.
//!
//! * [`verdicts`]: per-class `Faster | Slower | Tie` under
//!   [`ClaimPolicyV1::sign_p_max`] / [`ClaimPolicyV1::min_effect`]. A class is
//!   `Faster` iff `p_faster <= sign_p_max` and `ratio <= 1 - min_effect`, `Slower`
//!   iff `p_slower <= sign_p_max` and `ratio >= 1 + min_effect`, else `Tie`.
//! * [`gate`]: the promotion gate `promote` applies to a row-class record: the
//!   regime must fit the arch's last-level cache, no admitted class may be
//!   `Slower` (per-class veto), at least one must be `Faster`, and, when
//!   [`ClaimPolicyV1::shares`] is set, the share-weighted material
//!   `sum(share * (1 - ratio)) >= min_effect`.
//! * [`derived_min_rows`] / [`derived_max_rows`]: the widest row predicate the
//!   evidence supports.
//! * [`receipt`] and the `pm-census tune selector-receipt` CLI ([`cli`]): the
//!   `pm.selector_receipt.v1` document `scripts/check-selector-receipts.py`
//!   checks against the source constants.

use super::{kld::sha256_hex, ClaimPolicyV1};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub const RECEIPT_SCHEMA: &str = "pm.selector_receipt.v1";

const MIB: u64 = 1 << 20;

/// Last-level cache of an arch in bytes (`hot` records may not exceed it).
pub fn llc_bytes(arch: &str) -> Option<u64> {
    match arch {
        "gfx1201" => Some(64 * MIB),
        "gfx1100" => Some(96 * MIB),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Faster,
    Slower,
    Tie,
}

/// The sign-test verdict of one row class.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ClassVerdict {
    pub rows: u64,
    /// candidate / baseline median block ratio.
    pub ratio: f64,
    pub p_faster: f64,
    pub p_slower: f64,
    pub verdict: Verdict,
    /// Both arms admit this row count (`admitted.baseline && admitted.candidate`).
    pub admitted: bool,
    /// The shape of the record the class was measured at.
    pub shape: String,
}

/// Classify one `(ratio, p_faster, p_slower)` triple.
pub fn classify(ratio: f64, p_faster: f64, p_slower: f64, policy: &ClaimPolicyV1) -> Verdict {
    if p_faster <= policy.sign_p_max && ratio <= 1.0 - policy.min_effect {
        Verdict::Faster
    } else if p_slower <= policy.sign_p_max && ratio >= 1.0 + policy.min_effect {
        Verdict::Slower
    } else {
        Verdict::Tie
    }
}

fn num(v: &Value, path: &[&str]) -> Option<f64> {
    path.iter()
        .try_fold(v, |v, k| v.get(*k))
        .and_then(Value::as_f64)
}

/// The record's `shape` as text (a string, or the compact JSON of anything else).
fn record_shape(rec: &Value) -> Option<String> {
    match rec.get("shape")? {
        Value::String(s) => Some(s.clone()),
        Value::Null => None,
        other => Some(other.to_string()),
    }
}

/// `admitted: {baseline, candidate}` (or a bare bool); absent = admitted.
fn class_admitted(c: &Value) -> Result<bool, String> {
    match c.get("admitted") {
        None | Some(Value::Null) => Ok(true),
        Some(Value::Bool(b)) => Ok(*b),
        Some(a) => {
            let arm = |k: &str| {
                a.get(k)
                    .and_then(Value::as_bool)
                    .ok_or_else(|| format!("admitted.{k} missing or not a bool"))
            };
            Ok(arm("baseline")? && arm("candidate")?)
        }
    }
}

/// Whether the record is a row-class record (`rowclasses` present).
pub fn has_rowclasses(rec: &Value) -> bool {
    rec.get("rowclasses").is_some_and(|r| !r.is_null())
}

/// Whether the record measures a contract-derived node.
pub fn is_contract(rec: &Value) -> bool {
    rec.get("node")
        .and_then(Value::as_str)
        .is_some_and(|n| n.starts_with("contract:"))
}

/// Per-class verdicts of one row-class record, ascending by `rows`. A class that
/// neither arm admits and that carries no timing (clipped) is dropped; an
/// admitted class without a complete timing is an error, never a pass.
pub fn verdicts(rec: &Value, policy: &ClaimPolicyV1) -> Result<Vec<ClassVerdict>, String> {
    let classes = rec
        .get("rowclasses")
        .and_then(Value::as_array)
        .ok_or("record has no rowclasses array")?;
    let rec_shape = record_shape(rec);
    let mut out = Vec::with_capacity(classes.len());
    for (i, c) in classes.iter().enumerate() {
        let rows = c
            .get("rows")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("rowclasses[{i}]: rows missing or not an integer"))?;
        let admitted =
            class_admitted(c).map_err(|e| format!("rowclasses[{i}] (rows={rows}): {e}"))?;
        let t = |k: &str| num(c, &["timing", k]);
        let (Some(ratio), Some(p_faster), Some(p_slower)) =
            (t("ratio_median"), t("p_faster"), t("p_slower"))
        else {
            if admitted {
                return Err(format!(
                    "rowclasses[{i}] (rows={rows}): admitted class lacks timing.ratio_median/p_faster/p_slower"
                ));
            }
            continue;
        };
        if !(ratio.is_finite() && ratio > 0.0) {
            return Err(format!(
                "rowclasses[{i}] (rows={rows}): ratio_median {ratio} not finite positive"
            ));
        }
        if !((0.0..=1.0).contains(&p_faster) && (0.0..=1.0).contains(&p_slower)) {
            return Err(format!(
                "rowclasses[{i}] (rows={rows}): p_faster {p_faster} / p_slower {p_slower} outside [0, 1]"
            ));
        }
        let shape = rec_shape
            .clone()
            .or_else(|| {
                c.get("shape").map(|s| match s {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
            })
            .unwrap_or_else(|| "?".to_owned());
        out.push(ClassVerdict {
            rows,
            ratio,
            p_faster,
            p_slower,
            verdict: classify(ratio, p_faster, p_slower, policy),
            admitted,
            shape,
        });
    }
    out.sort_by_key(|c| c.rows);
    for w in out.windows(2) {
        if w[0].rows == w[1].rows {
            return Err(format!("rows={} appears twice in one record", w[0].rows));
        }
    }
    Ok(out)
}

/// Merge per-shape verdict sets: a class is `Slower` if `Slower` in any shape,
/// `Faster` only if `Faster` in every shape that admits it. The reported
/// numbers are the worst case (the slowest `Slower` entry, else the highest ratio).
/// A class counts as admitted when at least one shape admits it; non-admitted
/// entries never contribute.
pub fn merge(sets: &[Vec<ClassVerdict>]) -> Vec<ClassVerdict> {
    let mut by_rows: BTreeMap<u64, Vec<&ClassVerdict>> = BTreeMap::new();
    for c in sets.iter().flatten() {
        by_rows.entry(c.rows).or_default().push(c);
    }
    let worst = |v: &[&ClassVerdict]| -> ClassVerdict {
        (*v.iter()
            .max_by(|a, b| a.ratio.total_cmp(&b.ratio))
            .expect("non-empty"))
        .clone()
    };
    by_rows
        .into_iter()
        .map(|(rows, entries)| {
            let adm: Vec<&ClassVerdict> = entries.iter().copied().filter(|c| c.admitted).collect();
            if adm.is_empty() {
                return worst(&entries);
            }
            let slower: Vec<&ClassVerdict> = adm
                .iter()
                .copied()
                .filter(|c| c.verdict == Verdict::Slower)
                .collect();
            let mut m = if !slower.is_empty() {
                worst(&slower)
            } else {
                worst(&adm)
            };
            m.rows = rows;
            m.admitted = true;
            m.verdict = if !slower.is_empty() {
                Verdict::Slower
            } else if adm.iter().all(|c| c.verdict == Verdict::Faster) {
                Verdict::Faster
            } else {
                Verdict::Tie
            };
            m
        })
        .collect()
}

/// Smallest admitted `Faster` class `c` such that no admitted class `>= c` is
/// `Slower`; `None` if there is none.
pub fn derived_min_rows(classes: &[ClassVerdict]) -> Option<u64> {
    let mut best = None;
    let mut adm: Vec<&ClassVerdict> = classes.iter().filter(|c| c.admitted).collect();
    adm.sort_by_key(|c| std::cmp::Reverse(c.rows));
    for c in adm {
        match c.verdict {
            Verdict::Slower => break,
            Verdict::Faster => best = Some(c.rows),
            Verdict::Tie => {}
        }
    }
    best
}

/// Largest admitted `Faster` class `c` such that no admitted class `<= c` is
/// `Slower`; `None` if there is none.
pub fn derived_max_rows(classes: &[ClassVerdict]) -> Option<u64> {
    let mut best = None;
    let mut adm: Vec<&ClassVerdict> = classes.iter().filter(|c| c.admitted).collect();
    adm.sort_by_key(|c| c.rows);
    for c in adm {
        match c.verdict {
            Verdict::Slower => break,
            Verdict::Faster => best = Some(c.rows),
            Verdict::Tie => {}
        }
    }
    best
}

/// Predicate direction of a selector constant: `Min` enables the candidate at
/// `rows >= value`, `Max` at `rows <= value`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Min,
    Max,
}

impl Kind {
    pub fn parse(s: &str) -> Result<Kind, String> {
        match s {
            "min" => Ok(Kind::Min),
            "max" => Ok(Kind::Max),
            o => Err(format!("--kind {o:?}: want min or max")),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Min => "min",
            Kind::Max => "max",
        }
    }
}

/// Whether `value` as a `kind` predicate is covered by `classes` (merged): the
/// value is inside the derived domain, no admitted class on the admitted side
/// of the predicate is `Slower`, and the admitted side holds at least one
/// measured class (an unmeasured side is never covered).
pub fn covers(classes: &[ClassVerdict], kind: Kind, value: u64) -> bool {
    let side = |c: &&ClassVerdict| {
        c.admitted
            && match kind {
                Kind::Min => c.rows >= value,
                Kind::Max => c.rows <= value,
            }
    };
    let within = match kind {
        Kind::Min => derived_min_rows(classes).is_some_and(|d| value >= d),
        Kind::Max => derived_max_rows(classes).is_some_and(|d| value <= d),
    };
    within
        && classes.iter().filter(side).next().is_some()
        && classes
            .iter()
            .filter(side)
            .all(|c| c.verdict != Verdict::Slower)
}

/// `sum(share * (1 - ratio))` over the admitted classes that have a share, and
/// how many classes matched.
pub fn weighted_material(classes: &[ClassVerdict], shares: &BTreeMap<u64, f64>) -> (f64, usize) {
    classes
        .iter()
        .filter(|c| c.admitted)
        .filter_map(|c| shares.get(&c.rows).map(|s| s * (1.0 - c.ratio)))
        .fold((0.0, 0), |(m, n), x| (m + x, n + 1))
}

/// `rows=share,rows=share` (the `--shares` argument).
pub fn parse_shares(s: &str) -> Result<BTreeMap<u64, f64>, String> {
    let mut m = BTreeMap::new();
    for part in s.split(',').filter(|p| !p.is_empty()) {
        let (r, v) = part
            .split_once('=')
            .ok_or_else(|| format!("--shares {part:?}: want rows=share"))?;
        let rows: u64 = r.parse().map_err(|_| format!("--shares: bad rows {r:?}"))?;
        let share: f64 = v
            .parse()
            .map_err(|_| format!("--shares: bad share {v:?}"))?;
        if !(share.is_finite() && share >= 0.0) {
            return Err(format!("--shares: share {share} not a finite non-negative"));
        }
        if m.insert(rows, share).is_some() {
            return Err(format!("--shares: rows={rows} given twice"));
        }
    }
    Ok(m)
}

fn bytes_per_call(rec: &Value, c: &Value) -> Option<f64> {
    c.get("bytes_per_call")
        .and_then(Value::as_f64)
        .or_else(|| rec.get("bytes_per_call").and_then(Value::as_f64))
}

/// Refuse a `hot` record whose working set exceeds the arch's last-level cache
/// (it would not measure what a hot-cache regime means) and a record that does
/// not state its regime.
pub fn check_regime(rec: &Value) -> Result<(), String> {
    let regime = rec
        .get("regime")
        .and_then(Value::as_str)
        .ok_or("record has no regime (need \"hot\" or \"pool\")")?;
    match regime {
        "pool" => return Ok(()),
        "hot" => {}
        o => return Err(format!("regime {o:?}: need \"hot\" or \"pool\"")),
    }
    let arch = rec.get("arch").and_then(Value::as_str).unwrap_or("");
    let llc = llc_bytes(arch)
        .ok_or_else(|| format!("hot regime on unknown arch {arch:?}: no last-level cache size"))?;
    let classes = rec
        .get("rowclasses")
        .and_then(Value::as_array)
        .ok_or("record has no rowclasses array")?;
    for c in classes {
        if !class_admitted(c).unwrap_or(false) {
            continue;
        }
        let rows = c.get("rows").and_then(Value::as_u64).unwrap_or(0);
        let b = bytes_per_call(rec, c).ok_or_else(|| {
            format!("hot regime but rows={rows} has no bytes_per_call to compare with the LLC")
        })?;
        if b > llc as f64 {
            return Err(format!(
                "regime is hot but rows={rows} moves {b} bytes/call > {arch} last-level cache {} MiB; re-measure with --pool",
                llc / MIB
            ));
        }
    }
    Ok(())
}

/// Whether the contract named by a `contract:<abi id | export symbol>` node
/// admits more than one row class (`WorkV1.admit`): any open-ended or ranged
/// domain does. A contract with no work spec cannot be proven single-class, so
/// it counts as spanning.
pub fn admit_spans_classes(node: &str) -> bool {
    use hipfire_kernel_abi::families;
    let name = node.strip_prefix("contract:").unwrap_or(node);
    let abi = families::all().find(|a| a.id == name || a.exports.iter().any(|e| e.symbol == name));
    let Some(abi) = abi else { return true };
    match families::work_for(abi.id) {
        Some(w) => w
            .work
            .admit
            .max_rows
            .map_or(true, |m| m > w.work.admit.min_rows),
        None => true,
    }
}

/// Whether the row-class gate governs a record: it carries a sweep, or it is a
/// contract record whose contract admits more than one class and the policy
/// requires a sweep. Records of other nodes keep their single-shape stages.
pub fn applies(rec: &Value, policy: &ClaimPolicyV1) -> bool {
    has_rowclasses(rec)
        || (policy.require_rowclasses
            && is_contract(rec)
            && admit_spans_classes(rec.get("node").and_then(Value::as_str).unwrap_or("")))
}

/// Promotion gate for one record; see the module docs. Returns the verdicts.
/// A contract record without a sweep is refused ("no row-class sweep") when
/// the policy requires one and its contract admits more than one class.
pub fn gate(rec: &Value, policy: &ClaimPolicyV1) -> Result<Vec<ClassVerdict>, String> {
    if !has_rowclasses(rec) {
        let node = rec.get("node").and_then(Value::as_str).unwrap_or("");
        if policy.require_rowclasses && is_contract(rec) && admit_spans_classes(node) {
            return Err(format!(
                "no row-class sweep: {node} admits more than one row class and the record is a single shape; rerun with --rowclasses"
            ));
        }
        return Err("record has no rowclasses".to_owned());
    }
    check_regime(rec)?;
    let v = verdicts(rec, policy)?;
    if !policy.admit_numeric {
        if let Some(c) = rec
            .get("rowclasses")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .find(|c| {
                class_admitted(c).unwrap_or(false)
                    && c.get("identical_to_baseline").and_then(Value::as_bool) == Some(false)
            })
        {
            return Err(format!(
                "rows={}: identical_to_baseline is false; NUMERIC candidates are not admitted by this policy (admit_numeric=false)",
                c.get("rows").and_then(Value::as_u64).unwrap_or(0)
            ));
        }
    }
    let slower: Vec<String> = v
        .iter()
        .filter(|c| c.admitted && c.verdict == Verdict::Slower)
        .map(|c| {
            format!(
                "rows={} (ratio {:.4}, p_slower {:e})",
                c.rows, c.ratio, c.p_slower
            )
        })
        .collect();
    if !slower.is_empty() {
        return Err(format!(
            "row-class veto: candidate is Slower at {} admitted class(es): {}",
            slower.len(),
            slower.join(", ")
        ));
    }
    if !v.iter().any(|c| c.admitted && c.verdict == Verdict::Faster) {
        return Err("no admitted row class is Faster".to_owned());
    }
    if !policy.shares.is_empty() {
        let (m, n) = weighted_material(&v, &policy.shares);
        if n == 0 {
            return Err("shares given but none matches an admitted measured row class".to_owned());
        }
        if m < policy.min_effect {
            return Err(format!(
                "share-weighted material {m:.6} < min_effect {} over {n} class(es)",
                policy.min_effect
            ));
        }
    }
    Ok(v)
}

fn same_str(recs: &[Value], key: &str) -> Result<String, String> {
    let mut it = recs.iter().map(|r| r.get(key).and_then(Value::as_str));
    let first = it
        .next()
        .flatten()
        .ok_or_else(|| format!("record 0 has no {key}"))?;
    for (i, v) in it.enumerate() {
        if v != Some(first) {
            return Err(format!(
                "record {} has {key} {:?} != {first:?}; receipts merge only identical judgements",
                i + 1,
                v
            ));
        }
    }
    Ok(first.to_owned())
}

/// Build a `pm.selector_receipt.v1` from `(path, file bytes, parsed record)`
/// triples. Every record must pass the regime check and agree on
/// `node`/`vs`/`arch`; classes merge across shapes ([`merge`]).
pub fn receipt(
    recs: &[(String, Vec<u8>, Value)],
    constant: &str,
    path: &str,
    value: u64,
    kind: Kind,
    policy: &ClaimPolicyV1,
    judge_argv: &[String],
) -> Result<Value, String> {
    if recs.is_empty() {
        return Err("no --record given".to_owned());
    }
    // hip_loop writes `vs` as `{baseline:{contract,..}, candidate:{..}}`; older/synthetic
    // records carry a plain string. Normalize to `contract:<baseline id>`.
    let values: Vec<Value> = recs
        .iter()
        .map(|r| {
            let mut v = r.2.clone();
            if let Some(id) = v
                .pointer("/vs/baseline/contract")
                .and_then(Value::as_str)
                .map(|s| format!("contract:{s}"))
            {
                v["vs"] = Value::String(id);
            }
            v
        })
        .collect();
    let candidate = same_str(&values, "node")?;
    let incumbent = same_str(&values, "vs")?;
    let arch = same_str(&values, "arch")?;
    let mut sets = Vec::new();
    let mut shapes: Vec<String> = Vec::new();
    for (p, _, r) in recs {
        if !has_rowclasses(r) {
            return Err(format!("{p}: not a row-class record (no rowclasses)"));
        }
        check_regime(r).map_err(|e| format!("{p}: {e}"))?;
        let v = verdicts(r, policy).map_err(|e| format!("{p}: {e}"))?;
        for c in &v {
            if !shapes.contains(&c.shape) {
                shapes.push(c.shape.clone());
            }
        }
        sets.push(v);
    }
    let merged = merge(&sets);
    let mut classes: Vec<&ClassVerdict> = sets.iter().flatten().collect();
    classes.sort_by(|a, b| (a.rows, &a.shape).cmp(&(b.rows, &b.shape)));
    let class_json: Vec<Value> = classes
        .iter()
        .map(|c| {
            json!({
                "rows": c.rows, "ratio": c.ratio, "p_faster": c.p_faster,
                "p_slower": c.p_slower, "verdict": c.verdict, "shape": c.shape,
                "admitted": c.admitted,
            })
        })
        .collect();
    let mut doc = json!({
        "schema": RECEIPT_SCHEMA,
        "constant": constant,
        "path": path,
        "value": value,
        "kind": kind.as_str(),
        "arch": arch,
        "candidate": candidate,
        "incumbent": incumbent,
        "shapes": shapes,
        "classes": class_json,
        "derived_min_rows": derived_min_rows(&merged),
        "derived_max_rows": derived_max_rows(&merged),
        "covers": covers(&merged, kind, value),
        "sign_p_max": policy.sign_p_max,
        "min_effect": policy.min_effect,
        "records": recs.iter().map(|(p, b, _)| json!({"path": p, "sha256": sha256_hex(b)})).collect::<Vec<_>>(),
        "judge_argv": judge_argv,
    });
    if !policy.shares.is_empty() {
        let (m, n) = weighted_material(&merged, &policy.shares);
        doc["material"] = json!({"value": m, "classes": n, "shares": policy.shares});
    }
    Ok(doc)
}

const USAGE: &str = "usage: pm-census tune selector-receipt --record R.json [--record R2.json ...] --constant NAME --path FILE --value N [--kind min|max] [--shares rows=share,...] [--policy P.json] --out receipts/selectors/NAME.json";

/// `pm-census tune selector-receipt ...` (`args` starts at `selector-receipt`).
pub fn cli(args: &[String]) -> Result<(), String> {
    let mut args = args.iter();
    match args.next().map(String::as_str) {
        Some("selector-receipt") => {}
        _ => return Err(USAGE.to_owned()),
    }
    let argv: Vec<String> = std::iter::once("pm-census".to_owned())
        .chain(std::iter::once("tune".to_owned()))
        .chain(std::iter::once("selector-receipt".to_owned()))
        .chain(args.clone().cloned())
        .collect();
    let mut m: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    while let Some(f) = args.next() {
        let name = f
            .strip_prefix("--")
            .filter(|n| {
                [
                    "record", "constant", "path", "value", "kind", "shares", "policy", "out",
                ]
                .contains(n)
            })
            .ok_or_else(|| format!("unknown argument {f}\n{USAGE}"))?;
        let v = args.next().ok_or_else(|| format!("{f} needs a value"))?;
        m.entry(name).or_default().push(v);
    }
    let one = |k: &str| -> Result<Option<&str>, String> {
        match m.get(k).map(Vec::as_slice) {
            None => Ok(None),
            Some([v]) => Ok(Some(*v)),
            Some(_) => Err(format!("--{k} given more than once")),
        }
    };
    let need = |k: &str| one(k)?.ok_or_else(|| format!("--{k} is required\n{USAGE}"));
    let value: u64 = need("value")?
        .parse()
        .map_err(|_| "--value: want a non-negative integer".to_owned())?;
    let kind = Kind::parse(one("kind")?.unwrap_or("min"))?;
    let mut policy: ClaimPolicyV1 = match one("policy")? {
        Some(p) => {
            let t = std::fs::read_to_string(p).map_err(|e| format!("read {p}: {e}"))?;
            serde_json::from_str(&t).map_err(|e| format!("parse {p}: {e}"))?
        }
        None => ClaimPolicyV1::default(),
    };
    if let Some(s) = one("shares")? {
        policy.shares = parse_shares(s)?;
    }
    let mut recs = Vec::new();
    for p in m.get("record").map(Vec::as_slice).unwrap_or_default() {
        let bytes = std::fs::read(p).map_err(|e| format!("read {p}: {e}"))?;
        let v: Value = serde_json::from_slice(&bytes).map_err(|e| format!("parse {p}: {e}"))?;
        recs.push(((*p).to_owned(), bytes, v));
    }
    let doc = receipt(
        &recs,
        need("constant")?,
        need("path")?,
        value,
        kind,
        &policy,
        &argv,
    )?;
    let out = need("out")?;
    if let Some(dir) = std::path::Path::new(out)
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
    {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    }
    let text = serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())? + "\n";
    std::fs::write(out, &text).map_err(|e| format!("write {out}: {e}"))?;
    println!("{text}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pol() -> ClaimPolicyV1 {
        ClaimPolicyV1::default()
    }

    /// One admitted class whose rounds are `faster`/`slower` of `r` (p = 2^-count).
    fn class(rows: u64, ratio: f64, p_faster: f64, p_slower: f64) -> Value {
        json!({
            "rows": rows, "shape": format!("M={rows},K=2560"),
            "identical_to_baseline": true,
            "timing": {"ratio_median": ratio, "p_faster": p_faster, "p_slower": p_slower, "rounds": []},
            "admitted": {"baseline": true, "candidate": true},
        })
    }
    fn fast(rows: u64) -> Value {
        class(rows, 0.8, 1e-4, 1.0)
    }
    fn slow(rows: u64) -> Value {
        class(rows, 1.6, 1.0, 1e-4)
    }
    fn tie(rows: u64) -> Value {
        class(rows, 1.0, 0.5, 0.5)
    }
    fn record(classes: Vec<Value>) -> Value {
        json!({
            "tool": "hip_loop", "node": "contract:gemm_f32_tiled_rows",
            "vs": "contract:gemm_f32_batched", "arch": "gfx1201", "shape": "M=128,K=2560",
            "regime": "hot", "bytes_per_call": 1.0e6, "rowclasses": classes,
        })
    }
    /// tiled-vs-batched decode signature: slower at 1, tie at 2..256, faster >= 512.
    fn signature() -> Value {
        record(vec![
            slow(1),
            slow(2),
            tie(256),
            fast(512),
            fast(1024),
            fast(4096),
        ])
    }

    #[test]
    fn verdicts_follow_sign_p_and_min_effect() {
        let v = verdicts(&signature(), &pol()).unwrap();
        let of = |r| v.iter().find(|c| c.rows == r).unwrap().verdict;
        assert_eq!(of(1), Verdict::Slower);
        assert_eq!(of(256), Verdict::Tie);
        assert_eq!(of(512), Verdict::Faster);
        // significant sign but effect below min_effect is a tie
        assert_eq!(classify(0.999, 1e-6, 1.0, &pol()), Verdict::Tie);
        // big effect but p above sign_p_max is a tie
        assert_eq!(classify(0.5, 0.01, 1.0, &pol()), Verdict::Tie);
        assert_eq!(classify(2.0, 1.0, 0.01, &pol()), Verdict::Tie);
        // boundaries are inclusive
        let p = pol();
        assert_eq!(
            classify(1.0 - p.min_effect, p.sign_p_max, 1.0, &p),
            Verdict::Faster
        );
        assert_eq!(
            classify(1.0 + p.min_effect, 1.0, p.sign_p_max, &p),
            Verdict::Slower
        );
        assert!(v.windows(2).all(|w| w[0].rows < w[1].rows));
        assert!(v.iter().all(|c| c.shape == "M=128,K=2560"));
    }

    #[test]
    fn malformed_classes_are_errors_not_passes() {
        let mut r = signature();
        r["rowclasses"][0]["timing"]
            .as_object_mut()
            .unwrap()
            .remove("p_slower");
        assert!(verdicts(&r, &pol()).unwrap_err().contains("rows=1"));
        let mut r = signature();
        r["rowclasses"][0]["timing"]["ratio_median"] = json!(-1.0);
        assert!(verdicts(&r, &pol()).is_err());
        // a clipped class (not admitted, no timing) is dropped
        let mut r = signature();
        r["rowclasses"].as_array_mut().unwrap().push(json!({
            "rows": 16384, "admitted": {"baseline": true, "candidate": false}
        }));
        assert_eq!(verdicts(&r, &pol()).unwrap().len(), 6);
        // duplicate rows
        let r = record(vec![fast(1), fast(1)]);
        assert!(verdicts(&r, &pol()).unwrap_err().contains("twice"));
    }

    #[test]
    fn veto_refuses_with_the_class_list() {
        let e = gate(&signature(), &pol()).unwrap_err();
        assert!(e.contains("row-class veto"), "{e}");
        assert!(e.contains("rows=1 ") && e.contains("rows=2 "), "{e}");
        assert!(!e.contains("rows=512"), "{e}");
        // no Slower class: passes
        let ok = record(vec![tie(1), tie(2), fast(512), fast(4096)]);
        assert_eq!(gate(&ok, &pol()).unwrap().len(), 4);
        // a Slower class that the arms do not both admit does not veto
        let mut r = signature();
        r["rowclasses"][0]["admitted"]["candidate"] = json!(false);
        r["rowclasses"][1]["admitted"] = json!(false);
        assert!(gate(&r, &pol()).is_ok());
        // all ties: nothing Faster
        let e = gate(&record(vec![tie(1), tie(512)]), &pol()).unwrap_err();
        assert!(e.contains("no admitted row class is Faster"), "{e}");
    }

    #[test]
    fn numeric_class_refused_when_policy_forbids() {
        let mut r = record(vec![fast(512)]);
        r["rowclasses"][0]["identical_to_baseline"] = json!(false);
        assert!(gate(&r, &pol()).is_ok());
        let strict = ClaimPolicyV1 {
            admit_numeric: false,
            ..pol()
        };
        assert!(gate(&r, &strict).unwrap_err().contains("NUMERIC"));
    }

    #[test]
    fn share_weighted_material() {
        let r = record(vec![fast(512), tie(1)]);
        // 0.8 ratio at 512: 1 * 0.2 = 0.2 >= min_effect
        let mut p = pol();
        p.shares = BTreeMap::from([(512, 0.01), (1, 0.99)]);
        // 0.01 * 0.2 + 0.99 * 0 = 0.002 < 0.005
        let e = gate(&r, &p).unwrap_err();
        assert!(e.contains("share-weighted material"), "{e}");
        p.shares = BTreeMap::from([(512, 0.5), (1, 0.5)]);
        assert!(gate(&r, &p).is_ok());
        p.shares = BTreeMap::from([(99, 1.0)]);
        assert!(gate(&r, &p).unwrap_err().contains("none matches"));
        assert_eq!(
            parse_shares("1=0.25,512=0.75").unwrap(),
            BTreeMap::from([(1, 0.25), (512, 0.75)])
        );
        assert!(parse_shares("1=x").is_err() && parse_shares("1").is_err());
        assert!(parse_shares("1=0.5,1=0.5").is_err());
    }

    #[test]
    fn derived_domain_min_and_max() {
        let v = verdicts(&signature(), &pol()).unwrap();
        assert_eq!(derived_min_rows(&v), Some(512));
        assert_eq!(derived_max_rows(&v), None); // Slower at the smallest class
                                                // Faster at the small end, Slower at the large end: a MAX predicate
        let r = record(vec![fast(1), fast(8), tie(64), slow(512), slow(4096)]);
        let v = verdicts(&r, &pol()).unwrap();
        assert_eq!(derived_min_rows(&v), None);
        assert_eq!(derived_max_rows(&v), Some(8));
        // a Slower hole above a Faster class keeps min above the hole
        let r = record(vec![fast(2), slow(16), fast(64), tie(256), fast(512)]);
        assert_eq!(derived_min_rows(&verdicts(&r, &pol()).unwrap()), Some(64));
        // non-admitted classes are ignored
        let mut r = record(vec![slow(1), fast(512)]);
        r["rowclasses"][0]["admitted"]["baseline"] = json!(false);
        assert_eq!(derived_min_rows(&verdicts(&r, &pol()).unwrap()), Some(512));
    }

    #[test]
    fn regime_refusal() {
        let mut r = signature();
        assert!(check_regime(&r).is_ok());
        r["bytes_per_call"] = json!(65u64 * MIB);
        let e = check_regime(&r).unwrap_err();
        assert!(
            e.contains("last-level cache 64 MiB") && e.contains("--pool"),
            "{e}"
        );
        // 64 MiB is not over; gfx1100 has 96 MiB
        r["bytes_per_call"] = json!(64u64 * MIB);
        assert!(check_regime(&r).is_ok());
        r["arch"] = json!("gfx1100");
        r["bytes_per_call"] = json!(90u64 * MIB);
        assert!(check_regime(&r).is_ok());
        r["bytes_per_call"] = json!(97u64 * MIB);
        assert!(check_regime(&r).is_err());
        // pool accepts any size; a missing regime or unknown arch is refused
        r["regime"] = json!("pool");
        assert!(check_regime(&r).is_ok());
        r.as_object_mut().unwrap().remove("regime");
        assert!(check_regime(&r).unwrap_err().contains("no regime"));
        let mut r = signature();
        r["arch"] = json!("gfx9999");
        assert!(check_regime(&r).unwrap_err().contains("unknown arch"));
        // per-class bytes_per_call overrides the record level
        let mut r = signature();
        r["rowclasses"][4]["bytes_per_call"] = json!(200u64 * MIB);
        assert!(check_regime(&r).unwrap_err().contains("rows=1024"));
        // the gate refuses it too
        assert!(gate(&r, &pol()).unwrap_err().contains("last-level cache"));
    }

    #[test]
    fn sweep_required_for_contract_records() {
        let mut r = signature();
        r.as_object_mut().unwrap().remove("rowclasses");
        let e = gate(&r, &pol()).unwrap_err();
        assert!(e.contains("no row-class sweep"), "{e}");
        let off = ClaimPolicyV1 {
            require_rowclasses: false,
            ..pol()
        };
        assert!(!gate(&r, &off).unwrap_err().contains("no row-class sweep"));
        // the F32 tiled contract is admitted from 512 up: spans classes
        assert!(admit_spans_classes(
            "contract:generic_gemm.f32_tiled_rows.mpw16"
        ));
        assert!(admit_spans_classes("contract:gemm_f32_tiled_rows"));
    }

    fn rec_bytes(r: &Value, name: &str) -> (String, Vec<u8>, Value) {
        (name.to_owned(), serde_json::to_vec(r).unwrap(), r.clone())
    }

    #[test]
    fn receipt_covers_true_and_false() {
        let rec = rec_bytes(&signature(), "a.json");
        let mk = |value, kind| {
            receipt(
                &[rec.clone()],
                "F32_TILED_ROWS_MIN_N",
                "x.rs",
                value,
                kind,
                &pol(),
                &["a".into()],
            )
            .unwrap()
        };
        let d = mk(512, Kind::Min);
        assert_eq!(d["schema"], RECEIPT_SCHEMA);
        assert_eq!(d["covers"], true);
        assert_eq!(d["derived_min_rows"], 512);
        assert!(d["derived_max_rows"].is_null());
        assert_eq!(d["candidate"], "contract:gemm_f32_tiled_rows");
        assert_eq!(d["incumbent"], "contract:gemm_f32_batched");
        assert_eq!(d["kind"], "min");
        assert_eq!(d["records"][0]["path"], "a.json");
        assert_eq!(d["records"][0]["sha256"].as_str().unwrap().len(), 64);
        assert_eq!(d["classes"].as_array().unwrap().len(), 6);
        assert_eq!(d["judge_argv"][0], "a");
        // above the derived min is fine; below is not
        assert_eq!(mk(1024, Kind::Min)["covers"], true);
        assert_eq!(mk(256, Kind::Min)["covers"], false);
        assert_eq!(mk(1, Kind::Min)["covers"], false);
        // nothing measured at or above the value: not covered
        assert_eq!(mk(1 << 20, Kind::Min)["covers"], false);
        // a MAX predicate is false when no Faster prefix exists
        assert_eq!(mk(512, Kind::Max)["covers"], false);
        let small = rec_bytes(&record(vec![fast(1), fast(8), slow(512)]), "b.json");
        let mx = |v| {
            receipt(&[small.clone()], "N_MAX", "x.rs", v, Kind::Max, &pol(), &[]).unwrap()["covers"]
                .clone()
        };
        assert_eq!(mx(8), true);
        assert_eq!(mx(1), true);
        assert_eq!(mx(64), false); // beyond derived max
    }

    #[test]
    fn receipt_merges_shapes_and_slower_wins() {
        let a = rec_bytes(&signature(), "a.json");
        // a second shape that is Slower at 512: the merged class is Slower
        let mut r2 = record(vec![slow(1), fast(1024), fast(4096)]);
        r2["rowclasses"][0] = slow(512);
        r2["shape"] = json!("M=17,K=10240");
        let b = rec_bytes(&r2, "b.json");
        let d = receipt(
            &[a.clone(), b.clone()],
            "C",
            "x.rs",
            1024,
            Kind::Min,
            &pol(),
            &[],
        )
        .unwrap();
        assert_eq!(d["derived_min_rows"], 1024);
        assert_eq!(d["covers"], true);
        assert_eq!(d["shapes"].as_array().unwrap().len(), 2);
        // 512 is not covered now
        let d = receipt(&[a.clone(), b], "C", "x.rs", 512, Kind::Min, &pol(), &[]).unwrap();
        assert_eq!(d["covers"], false);
        assert!(d["classes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["shape"] == "M=17,K=10240"));
        // per-shape entries survive, sorted by (rows, shape)
        let rows: Vec<u64> = d["classes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["rows"].as_u64().unwrap())
            .collect();
        assert!(rows.windows(2).all(|w| w[0] <= w[1]));
        // merge keeps a class Faster only if Faster in every admitting shape
        let m = merge(&[
            verdicts(&record(vec![fast(512)]), &pol()).unwrap(),
            verdicts(&record(vec![tie(512)]), &pol()).unwrap(),
        ]);
        assert_eq!(m[0].verdict, Verdict::Tie);
        // a non-admitting shape does not weaken a Faster class
        let mut other = record(vec![tie(512)]);
        other["rowclasses"][0]["admitted"]["candidate"] = json!(false);
        let m = merge(&[
            verdicts(&record(vec![fast(512)]), &pol()).unwrap(),
            verdicts(&other, &pol()).unwrap(),
        ]);
        assert_eq!(m[0].verdict, Verdict::Faster);
    }

    #[test]
    fn receipt_refuses_inconsistent_or_hot_over_llc_records() {
        let a = rec_bytes(&signature(), "a.json");
        let mut other = signature();
        other["node"] = json!("contract:other");
        let e = receipt(
            &[a.clone(), rec_bytes(&other, "o.json")],
            "C",
            "x.rs",
            1,
            Kind::Min,
            &pol(),
            &[],
        )
        .unwrap_err();
        assert!(e.contains("node"), "{e}");
        let mut hot = signature();
        hot["bytes_per_call"] = json!(1u64 << 30);
        let e = receipt(
            &[rec_bytes(&hot, "h.json")],
            "C",
            "x.rs",
            512,
            Kind::Min,
            &pol(),
            &[],
        )
        .unwrap_err();
        assert!(
            e.contains("h.json") && e.contains("last-level cache"),
            "{e}"
        );
        let mut single = signature();
        single.as_object_mut().unwrap().remove("rowclasses");
        assert!(receipt(
            &[rec_bytes(&single, "s.json")],
            "C",
            "x.rs",
            1,
            Kind::Min,
            &pol(),
            &[]
        )
        .unwrap_err()
        .contains("rowclasses"));
        assert!(receipt(&[], "C", "x.rs", 1, Kind::Min, &pol(), &[]).is_err());
    }

    #[test]
    fn cli_writes_receipt_and_rejects_bad_args() {
        let dir = std::env::temp_dir().join(format!("pmrc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let rp = dir.join("r.json");
        std::fs::write(&rp, serde_json::to_vec(&signature()).unwrap()).unwrap();
        let out = dir.join("sub/NAME.json");
        let s = |v: &str| v.to_owned();
        let args = vec![
            s("selector-receipt"),
            s("--record"),
            rp.to_str().unwrap().to_owned(),
            s("--constant"),
            s("NAME"),
            s("--path"),
            s("a.rs"),
            s("--value"),
            s("512"),
            s("--out"),
            out.to_str().unwrap().to_owned(),
        ];
        cli(&args).unwrap();
        let d: Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
        assert_eq!(d["covers"], true);
        assert_eq!(d["constant"], "NAME");
        assert!(d["judge_argv"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a == "--constant"));
        let mut bad = args.clone();
        bad.push(s("--bogus"));
        bad.push(s("1"));
        assert!(cli(&bad).unwrap_err().contains("unknown argument"));
        assert!(cli(&args[..2]).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
