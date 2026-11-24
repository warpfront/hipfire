//! Class table, row interface, and verdicts.

use crate::digest::{first_difference, FirstDifference, HostState, StateDigest};
use crate::receipt::Receipt;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Instant;

// ---- class table (classes.toml) ------------------------------------------

#[derive(Clone, Debug, Deserialize)]
pub struct Invariant {
    pub id: String,
    pub name: String,
    pub always_exact: String,
    #[serde(default)]
    pub not_compared: Option<String>,
    /// The named arithmetic difference a row may bound; absent = never bounded.
    #[serde(default)]
    pub bounded_difference: Option<String>,
    #[serde(default)]
    pub bounded_form: Option<String>,
    #[serde(default)]
    pub bounded_requires_family_declaration: bool,
    #[serde(default)]
    pub transitional: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ClassTable {
    pub invariant: Vec<Invariant>,
}

pub const EMBEDDED_CLASSES: &str = include_str!("../classes.toml");

impl ClassTable {
    pub fn parse(text: &str) -> Result<ClassTable, String> {
        let t: ClassTable = toml::from_str(text).map_err(|e| format!("classes.toml: {e}"))?;
        let mut seen = std::collections::BTreeSet::new();
        for i in &t.invariant {
            if !seen.insert(i.id.clone()) {
                return Err(format!("classes.toml: duplicate invariant {}", i.id));
            }
            if i.bounded_difference.as_deref() == Some("") {
                return Err(format!("classes.toml: {} has an unnamed bounded difference", i.id));
            }
        }
        Ok(t)
    }

    pub fn embedded() -> ClassTable {
        Self::parse(EMBEDDED_CLASSES).expect("embedded classes.toml is valid")
    }

    pub fn get(&self, id: &str) -> Option<&Invariant> {
        self.invariant.iter().find(|i| i.id == id)
    }
}

// ---- row interface -------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonClass {
    Exact,
    Bounded,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Recipe {
    pub name: String,
    /// Kernels this recipe must resolve for the row to count as covering it.
    pub required_kernels: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecipePair {
    pub reference: Recipe,
    pub candidate: Recipe,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputSchedule {
    pub seed: u64,
    pub steps: usize,
}

/// Per-card budgets keyed by the named arithmetic difference.
pub type Budgets = BTreeMap<String, f64>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RowSpec {
    pub fixture: String,
    pub invariant: String,
    pub recipe_pair: RecipePair,
    pub input_schedule: InputSchedule,
    pub comparison_class: ComparisonClass,
    /// For a bounded row: the named difference (must equal the table's).
    #[serde(default)]
    pub bounded_difference: Option<String>,
    #[serde(default)]
    pub budgets: Budgets,
}

/// One executed arm: audit log text plus the state after every step.
pub struct ArmRun {
    pub audit_log: String,
    pub steps: Vec<HostState>,
}

/// An arm executes a recipe over a schedule. The card half implements this over
/// real engines; the self-test implements it over a host fake.
pub trait Arm {
    fn run(&mut self, recipe: &Recipe, schedule: &InputSchedule) -> Result<ArmRun, String>;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Verdict {
    Pass,
    Fail { first_difference: FirstDifference },
    /// The intended recipe did not run on at least one arm.
    NotCovered { missing: Vec<String> },
    /// Bounded class and no budget for the named difference.
    Unqualified { difference: String },
}

impl Verdict {
    pub fn name(&self) -> &'static str {
        match self {
            Verdict::Pass => "pass",
            Verdict::Fail { .. } => "fail",
            Verdict::NotCovered { .. } => "not_covered",
            Verdict::Unqualified { .. } => "unqualified",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RowMetric {
    pub step: usize,
    pub state_equal: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct RowReport {
    pub fixture: String,
    pub invariant: String,
    pub executed_receipts: (Receipt, Receipt),
    pub per_row_metrics: Vec<RowMetric>,
    pub first_difference: Option<FirstDifference>,
    #[serde(flatten)]
    pub verdict: Verdict,
    pub wall_ms: f64,
    /// Peak resident set of the runner process (`VmHWM`), KiB, when readable.
    pub peak_rss_kib: Option<u64>,
}

/// Why a row was refused before running.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    NoClass(String),
    BoundedNotAllowed { invariant: String },
    DifferenceMismatch { invariant: String, got: Option<String>, table: String },
    Arm(String),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::NoClass(i) => write!(f, "refused: invariant {i} has no class in classes.toml"),
            Refusal::BoundedNotAllowed { invariant } => {
                write!(f, "refused: {invariant} is always exact, a bounded class is not allowed")
            }
            Refusal::DifferenceMismatch { invariant, got, table } => write!(
                f,
                "refused: {invariant} bounded difference {got:?} does not match the table's {table:?}"
            ),
            Refusal::Arm(e) => write!(f, "arm failed: {e}"),
        }
    }
}

pub fn peak_rss_kib() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    s.lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
}

/// Decide a row's class standing before anything runs. `Ok(Some(d))` = bounded
/// with named difference `d`; `Ok(None)` = exact.
fn admit(table: &ClassTable, row: &RowSpec) -> Result<Option<String>, Refusal> {
    let inv = table.get(&row.invariant).ok_or_else(|| Refusal::NoClass(row.invariant.clone()))?;
    match row.comparison_class {
        ComparisonClass::Exact => Ok(None),
        ComparisonClass::Bounded => {
            let Some(tbl) = &inv.bounded_difference else {
                return Err(Refusal::BoundedNotAllowed { invariant: inv.id.clone() });
            };
            if row.bounded_difference.as_deref() != Some(tbl.as_str()) {
                return Err(Refusal::DifferenceMismatch {
                    invariant: inv.id.clone(),
                    got: row.bounded_difference.clone(),
                    table: tbl.clone(),
                });
            }
            Ok(Some(tbl.clone()))
        }
    }
}

/// Run one row. Order of standing: no class / not allowed -> refused;
/// recipe not executed -> `not_covered`; bounded without budget ->
/// `unqualified`; otherwise compare state digests step by step.
pub fn run_row(
    table: &ClassTable,
    row: &RowSpec,
    reference: &mut dyn Arm,
    candidate: &mut dyn Arm,
) -> Result<RowReport, Refusal> {
    let bounded = admit(table, row)?;
    let t0 = Instant::now();
    let a = reference.run(&row.recipe_pair.reference, &row.input_schedule).map_err(Refusal::Arm)?;
    let b = candidate.run(&row.recipe_pair.candidate, &row.input_schedule).map_err(Refusal::Arm)?;
    let ra = Receipt::parse(&a.audit_log).map_err(Refusal::Arm)?;
    let rb = Receipt::parse(&b.audit_log).map_err(Refusal::Arm)?;

    let mut report = RowReport {
        fixture: row.fixture.clone(),
        invariant: row.invariant.clone(),
        executed_receipts: (ra.clone(), rb.clone()),
        per_row_metrics: Vec::new(),
        first_difference: None,
        verdict: Verdict::Pass,
        wall_ms: 0.0,
        peak_rss_kib: None,
    };

    let mut missing = ra.missing(&row.recipe_pair.reference.required_kernels);
    for m in rb.missing(&row.recipe_pair.candidate.required_kernels) {
        if !missing.contains(&m) {
            missing.push(m);
        }
    }
    if !missing.is_empty() {
        report.verdict = Verdict::NotCovered { missing };
    } else if let Some(d) = bounded.clone().filter(|d| !row.budgets.contains_key(d)) {
        report.verdict = Verdict::Unqualified { difference: d };
    } else if bounded.is_some() {
        // A budgeted bounded row needs logit-level metrics from the card half.
        report.verdict = Verdict::Unqualified {
            difference: format!("{}: bounded comparison needs the card runner", row.invariant),
        };
    } else {
        let da: Vec<StateDigest> = a.steps.iter().map(HostState::digest).collect();
        let db: Vec<StateDigest> = b.steps.iter().map(HostState::digest).collect();
        for step in 0..da.len().max(db.len()) {
            let diff = match (da.get(step), db.get(step)) {
                (Some(x), Some(y)) => first_difference(x, y),
                _ => Some(FirstDifference {
                    step,
                    component: "schedule".into(),
                    row: None,
                    owner: "schedule".into(),
                    detail: format!("arm step counts differ: {} vs {}", da.len(), db.len()),
                }),
            };
            report.per_row_metrics.push(RowMetric { step, state_equal: diff.is_none() });
            if let Some(mut d) = diff {
                d.step = step;
                if report.first_difference.is_none() {
                    report.first_difference = Some(d.clone());
                    report.verdict = Verdict::Fail { first_difference: d };
                }
            }
        }
    }
    report.wall_ms = t0.elapsed().as_secs_f64() * 1e3;
    report.peak_rss_kib = peak_rss_kib();
    Ok(report)
}
