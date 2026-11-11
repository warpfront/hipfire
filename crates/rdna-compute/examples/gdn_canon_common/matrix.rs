// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Schedule runner and bit-exact comparison for [`GdnStreamUnderTest`]
//! implementations (CHUNK-CANONICAL-DESIGN §6 S0 kills 3 and 4, §7 G1/G2).
//!
//! A *schedule* is a list of steps applied to one stream: appends of arbitrary
//! (off-grid) lengths, `Seal`, a non-mutating `Peek` of the would-be sealed
//! state, and `SnapshotClobberRestore` (snapshot → destroy every live
//! persistent buffer → restore). Every schedule ends with an implicit final
//! `seal`. Two runs are compared **bitwise**: every published output row,
//! every mid-run seal/peek state, the final sealed state (Q8 codes, scales, EF,
//! conv) and — for implementations that report it — the stream metadata
//! against the absolute-grid formula [`model::expected_meta`].

use super::model::expected_meta;
use super::stream::*;
use super::util::*;
use rdna_compute::gdn_stream::*;
use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub enum Step {
    Append(usize),
    /// Real prompt end / mode fence (transitions to `Sealed`).
    Seal,
    /// `peek_seal` (non-mutating); compared against the reference's peek.
    Peek,
    SnapshotClobberRestore,
}

#[derive(Clone, Debug)]
pub struct Schedule {
    pub name: String,
    pub steps: Vec<Step>,
}

impl Schedule {
    pub fn total_rows(&self) -> usize {
        self.steps
            .iter()
            .map(|s| if let Step::Append(n) = s { *n } else { 0 })
            .sum()
    }
    pub fn single(total: usize) -> Self {
        Self {
            name: format!("single[{total}]"),
            steps: vec![Step::Append(total)],
        }
    }
    pub fn fixed_chunk(total: usize, chunk: usize) -> Self {
        let mut steps = vec![];
        let mut left = total;
        while left > 0 {
            let n = chunk.min(left);
            steps.push(Step::Append(n));
            left -= n;
        }
        Self {
            name: format!("chunk{chunk}[{total}]"),
            steps,
        }
    }
    pub fn random_partition(total: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut steps = vec![];
        let mut left = total;
        while left > 0 {
            let n = match rng.below(6) {
                0 => 1 + rng.below(3),
                1 => 1 + rng.below(63),
                2 => 64 * (1 + rng.below(3)),
                3 => 1 + rng.below(130),
                4 => 500 + rng.below(30),
                _ => 1 + rng.below(70),
            }
            .min(left);
            steps.push(Step::Append(n));
            left -= n;
        }
        Self {
            name: format!("random[{total}]#{seed:x}"),
            steps,
        }
    }
    /// Snapshot Open at `p` (optionally Sealed first), overwrite, restore,
    /// continue to `total`. Peeks before/after the restore prove the restored
    /// representation is the same semantic state.
    pub fn snapshot_at(p: usize, total: usize, sealed: bool, peeks: bool) -> Self {
        let mut steps = vec![Step::Append(p)];
        if sealed {
            steps.push(Step::Seal);
        }
        if peeks {
            steps.push(Step::Peek);
        }
        steps.push(Step::SnapshotClobberRestore);
        if peeks {
            steps.push(Step::Peek);
        }
        if total > p {
            steps.push(Step::Append(total - p));
        }
        Self {
            name: format!(
                "snapshot_{}@{p}[{total}]",
                if sealed { "sealed" } else { "open" }
            ),
            steps,
        }
    }
    /// Same appends with `Peek`/`SnapshotClobberRestore` removed and, when no
    /// `Seal` step is present, merged into a single append.
    pub fn is_partition_only(&self) -> bool {
        self.steps.iter().all(|s| matches!(s, Step::Append(_)))
    }
}

#[derive(Clone, Debug)]
pub struct RunResult {
    pub outputs: Vec<f32>,
    pub metas: Vec<GdnStreamMeta>,
    pub peeks: Vec<SealedState>,
    pub mid_seals: Vec<SealedState>,
    pub final_state: SealedState,
    pub expected_metas: Vec<GdnStreamMeta>,
}

pub struct RunCfg {
    /// 0 = do not compare recipe ids.
    pub recipe_id: u64,
}

/// Drive `s` through `sched` with `data` rows (starting at the schedule's
/// first row), from sealed `init`.
pub fn run_schedule(
    s: &mut dyn GdnStreamUnderTest,
    init: &InitialState,
    data: &RawRows,
    sched: &Schedule,
    cfg: &RunCfg,
) -> R<RunResult> {
    let g = s.geometry();
    s.reset(init)?;
    let mut cursor = 0usize;
    let mut phase = GdnStreamPhase::Sealed;
    let mut position = init.position;
    let mut phase_start = init.position;
    let mut res = RunResult {
        outputs: vec![],
        metas: vec![],
        peeks: vec![],
        mid_seals: vec![],
        final_state: SealedState {
            position: 0,
            codes: vec![],
            scales: vec![],
            ef: vec![],
            conv: vec![],
        },
        expected_metas: vec![],
    };
    for step in &sched.steps {
        match step {
            Step::Append(n) => {
                if cursor + n > data.rows {
                    return Err(
                        format!("schedule {} needs more rows than supplied", sched.name).into(),
                    );
                }
                let rows = data.slice(&g, cursor, cursor + n);
                res.outputs.extend(s.append(&rows)?);
                cursor += n;
                if *n > 0 {
                    if phase == GdnStreamPhase::Sealed {
                        phase = GdnStreamPhase::Open;
                        phase_start = position;
                    }
                    position += *n as u64;
                }
            }
            Step::Seal => {
                res.mid_seals.push(s.seal()?);
                phase = GdnStreamPhase::Sealed;
            }
            Step::Peek => res.peeks.push(s.peek_seal()?),
            Step::SnapshotClobberRestore => {
                let snap = s.snapshot()?;
                s.clobber_live()?;
                s.restore(&snap)?;
            }
        }
        res.metas.push(s.meta());
        let mut m = expected_meta(&g, cfg.recipe_id, phase, position, phase_start, 0);
        m.recipe_id = cfg.recipe_id;
        res.expected_metas.push(m);
    }
    res.final_state = s.seal()?;
    Ok(res)
}

fn meta_matches(a: &GdnStreamMeta, e: &GdnStreamMeta, check_recipe: bool) -> bool {
    a.schema_version == e.schema_version
        && (!check_recipe || a.recipe_id == e.recipe_id)
        && a.phase == e.phase
        && a.position == e.position
        && a.phase_start == e.phase_start
        && a.frame_base == e.frame_base
        && a.tile_base == e.tile_base
        && a.pending_rows == e.pending_rows
}

#[derive(Clone, Debug)]
pub struct Diff {
    pub pass: bool,
    pub rows: usize,
    pub outputs_identical: bool,
    pub first_diff_row: Option<usize>,
    pub out_maxabs: f64,
    pub out_nonfinite_mismatch: usize,
    pub final_state: StateDiff,
    pub mid_seal_mismatches: usize,
    pub peek_mismatches: usize,
    pub meta_mismatches: Vec<String>,
}

impl Diff {
    /// Bitwise agreement of every published output, mid-run seal/peek and the
    /// final sealed state — excluding the metadata schema check.
    pub fn numeric_pass(&self) -> bool {
        self.outputs_identical
            && self.final_state.is_empty()
            && self.mid_seal_mismatches == 0
            && self.peek_mismatches == 0
    }

    pub fn json(&self) -> Value {
        json!({
            "pass": self.pass,
            "numeric_pass": self.numeric_pass(),
            "rows": self.rows,
            "outputs_identical": self.outputs_identical,
            "first_diff_row": self.first_diff_row,
            "out_maxabs": self.out_maxabs,
            "out_nonfinite_mismatch": self.out_nonfinite_mismatch,
            "final_state": {
                "position": self.final_state.position,
                "q8_codes_diff": self.final_state.codes,
                "scales_diff": self.final_state.scales,
                "ef_diff": self.final_state.ef,
                "conv_diff": self.final_state.conv,
            },
            "mid_seal_mismatches": self.mid_seal_mismatches,
            "peek_mismatches": self.peek_mismatches,
            "meta_mismatches": self.meta_mismatches,
        })
    }
}

/// Bitwise comparison of a candidate run against a reference run of the same
/// schedule family. `check_meta`: compare the candidate's reported metadata
/// with the absolute-grid formula (needs `Stream::reports_stream_meta`).
pub fn compare_runs(
    g: &StreamGeometry,
    reference: &RunResult,
    cand: &RunResult,
    check_meta: bool,
    check_recipe: bool,
) -> Diff {
    let ow = g.out_width();
    let first = first_diff_f32(&reference.outputs, &cand.outputs);
    let (maxabs, nonfinite) = maxabs_diff(&reference.outputs, &cand.outputs);
    let final_state = reference.final_state.diff(&cand.final_state);
    let mid = reference
        .mid_seals
        .iter()
        .zip(&cand.mid_seals)
        .filter(|(a, b)| !a.diff(b).is_empty())
        .count()
        + reference.mid_seals.len().abs_diff(cand.mid_seals.len());
    let peeks = reference
        .peeks
        .iter()
        .zip(&cand.peeks)
        .filter(|(a, b)| !a.diff(b).is_empty())
        .count()
        + reference.peeks.len().abs_diff(cand.peeks.len());
    let mut meta_bad = vec![];
    if check_meta {
        for (i, (m, e)) in cand.metas.iter().zip(&cand.expected_metas).enumerate() {
            if !meta_matches(m, e, check_recipe) {
                meta_bad.push(format!(
                    "step{i}: got {{phase:{:?} pos:{} start:{} frame:{} tile:{} pending:{}}} expected {{phase:{:?} pos:{} start:{} frame:{} tile:{} pending:{}}}",
                    m.phase, m.position, m.phase_start, m.frame_base, m.tile_base, m.pending_rows,
                    e.phase, e.position, e.phase_start, e.frame_base, e.tile_base, e.pending_rows
                ));
                if meta_bad.len() >= 4 {
                    break;
                }
            }
        }
    }
    let outputs_identical = first.is_none();
    Diff {
        pass: outputs_identical
            && final_state.is_empty()
            && mid == 0
            && peeks == 0
            && meta_bad.is_empty(),
        rows: reference.outputs.len() / ow.max(1),
        outputs_identical,
        first_diff_row: first.map(|i| i / ow.max(1)),
        out_maxabs: maxabs,
        out_nonfinite_mismatch: nonfinite,
        final_state,
        mid_seal_mismatches: mid,
        peek_mismatches: peeks,
        meta_mismatches: meta_bad,
    }
}

/// The off-grid positions of design §6 S0 kill 4 / §7 G3.
pub const KILL_POSITIONS: [usize; 9] = [1, 25, 63, 64, 65, 511, 512, 513, 521];
