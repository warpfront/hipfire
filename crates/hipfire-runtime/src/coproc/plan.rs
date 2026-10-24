// SPDX-License-Identifier: Apache-2.0
//! Pure deterministic crossover planner of a co-processor stage. No clock or hardware input.
//!
//! Chunk pairs are the micro-batches: a `Step::Pipe` runs two banks of rows layer-major so the co-processor
//! computes one bank's owned projections while the GPU runs the other bank's layer (`coproc::schedule`); the
//! first co-processor chain of a pair is the pipeline fill. The model prices the GPU-only chunk, each bank's
//! per-kind GPU segment and co-processor chains, steady-state stalls, DRAM contention and the fill, all from a
//! [`Calibration`] of one [`StageMap`].
use super::calibration::OutFamily;
use super::stage::{FamilySet, StageMap, GROUPS, KINDS};
use super::{CalKey, Calibration, FamilyCalibration};
use sha2::{Digest, Sha256};
use std::ops::Range;

pub const CHUNK: usize = 8192;
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    Gpu { chunk: Range<usize> },
    Pipe { a: Range<usize>, b: Range<usize> },
}
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub steps: Vec<Step>,
    pub t_gpu_ms: f64,
    pub t_pipe_ms: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct GpuChunk {
    pub total: f64,
    /// Per layer of each kind: everything but the owned-eligible input projections.
    pub rest: [f64; KINDS],
    /// Per group: the GPU cost of its input projections over all layers of its kind.
    pub trunk: [f64; GROUPS],
}
#[derive(Clone, Copy, Debug)]
pub struct BankSegments {
    pub gpu: [f64; KINDS],
    pub npu: [f64; KINDS],
    pub npu_out: [f64; KINDS],
    pub bytes: [f64; KINDS],
    pub chunk: GpuChunk,
}
#[derive(Clone, Copy, Debug)]
pub struct PairTime {
    pub t: f64,
    pub stall: f64,
    pub extra: f64,
    pub fill: f64,
    pub gpu_only: f64,
}

fn layers(map: &StageMap, kind: usize) -> f64 {
    map.kinds[kind].layers as f64
}
pub fn gpu_chunk(map: &StageMap, cal: &Calibration, rows: usize, prefix: usize) -> GpuChunk {
    let g = &cal.gpu;
    let f = rows as f64 / CHUNK as f64;
    let pre = prefix as f64 / CHUNK as f64;
    let trunk = map
        .groups
        .map(|group| g.trunk_in_ms_per_8192[group.name] * f);
    let rest = [
        (g.uniform_per_layer_ms + g.gdn_core_per_layer_ms) * f,
        (g.uniform_per_layer_ms
            + g.qsa_per_layer_ms
            + g.qsa_growth_per_layer_ms_per_8192_prefix * pre)
            * f,
    ];
    GpuChunk {
        total: layers(map, 0) * rest[0]
            + layers(map, 1) * rest[1]
            + trunk.iter().sum::<f64>()
            + g.fixed_per_chunk_ms,
        rest,
        trunk,
    }
}
fn npu_ms_with(rows: usize, cols: usize, k: usize, panel: usize, tops: f64) -> f64 {
    let padded = cols.div_ceil(panel) * panel;
    2.0 * rows as f64 * padded as f64 * k as f64 / (tops * 1e12) * 1e3
}
/// Co-run NPU compute time of `cols` input-projection columns of `kind` over `rows` rows.
pub fn npu_ms(map: &StageMap, cal: &Calibration, kind: usize, rows: usize, cols: usize) -> f64 {
    npu_ms_with(
        rows,
        cols,
        map.kinds[kind].in_k,
        cal.npu.panel_cols,
        cal.npu.tops_corun,
    )
}
/// Out family of layer kind `k`: its calibration and out-projection block.
fn out_family<'a>(
    map: &StageMap,
    cal: &'a Calibration,
    k: usize,
) -> (&'a FamilyCalibration, &'a OutFamily) {
    let family = &cal.families[map.kinds[k].out_family];
    (
        family,
        family
            .out
            .as_ref()
            .expect("validated: out families carry an out block"),
    )
}
/// Co-run NPU compute time of an out family: its simulator alone rate scaled by the in-proj co-run factor.
fn out_ms(map: &StageMap, cal: &Calibration, k: usize, rows: usize) -> f64 {
    let (family, o) = out_family(map, cal, k);
    let tops = o.tops_alone * cal.npu.tops_corun
        / cal
            .npu
            .tops_alone
            .expect("validated: out families need npu.tops_alone");
    npu_ms_with(rows, family.cols, o.k, o.panel_cols, tops)
}
/// What the planner moves to the co-processor: whole in-proj groups (`StageMap::groups`) and the out families
/// (one per kind). `supported` is false for an empty or partial-group selection, which stays GPU-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Moved {
    pub inproj: [bool; GROUPS],
    pub out: [bool; KINDS],
    pub supported: bool,
}
impl Moved {
    pub fn new(inproj: [bool; GROUPS], out: [bool; KINDS]) -> Self {
        Self {
            inproj,
            out,
            supported: inproj.iter().chain(&out).any(|moved| *moved),
        }
    }
    /// The groups of an ownership set; no out family.
    pub fn of(map: &StageMap, families: FamilySet) -> Self {
        Self {
            supported: map.supports(families),
            ..Self::new(map.owned_groups(families), [false; KINDS])
        }
    }
}
pub fn bank_segments(
    map: &StageMap,
    cal: &Calibration,
    families: FamilySet,
    rows: usize,
    prefix: usize,
) -> BankSegments {
    bank_segments_moved(map, cal, Moved::of(map, families), rows, prefix)
}
/// Per-layer-kind GPU segment, NPU in-proj / out chains and NPU DRAM bytes of one bank visit. A moved out family
/// removes its GPU o-proj GEMM from the GPU segment and adds its K-scaled pack.
pub fn bank_segments_moved(
    map: &StageMap,
    cal: &Calibration,
    moved: Moved,
    rows: usize,
    prefix: usize,
) -> BankSegments {
    let chunk = gpu_chunk(map, cal, rows, prefix);
    let f = rows as f64 / CHUNK as f64;
    let mut cols = [0; KINDS];
    let mut kept = [0.0; KINDS];
    let mut consume = [0.0; KINDS];
    for (i, group) in map.groups.iter().enumerate() {
        let kind = group.kind;
        if moved.inproj[i] {
            let family = &cal.families[group.name];
            cols[kind] += family.cols;
            if cal.glue.consume_mode == "retile" {
                consume[kind] +=
                    cal.glue.consume_ms_per_col_8192[&family.out_dtype] * family.cols as f64;
            }
        } else {
            kept[kind] += chunk.trunk[i];
        }
    }
    let mut gpu = [
        chunk.rest[0] + kept[0] / layers(map, 0),
        chunk.rest[1] + kept[1] / layers(map, 1),
    ];
    let mut npu = [0.0; KINDS];
    let mut npu_out = [0.0; KINDS];
    let mut bytes = [0.0; KINDS];
    for k in 0..KINDS {
        let in_k = map.kinds[k].in_k as f64;
        if cols[k] != 0 {
            gpu[k] += cal.glue.pack_ms_8192 * f;
            npu[k] = npu_ms(map, cal, k, rows, cols[k]) + cal.glue.sync_ms;
        }
        gpu[k] += consume[k] * f;
        if cols[k] != 0 || moved.out[k] {
            gpu[k] += cal.glue.visit_ms_8192 * f;
        }
        let padded = cols[k].div_ceil(cal.npu.panel_cols) * cal.npu.panel_cols;
        bytes[k] = 2.0 * rows as f64 * padded as f64 * in_k / 1e3 * cal.npu.bytes_per_kop;
        if moved.out[k] {
            let (family, o) = out_family(map, cal, k);
            gpu[k] += (cal.glue.pack_ms_8192 * o.k as f64 / in_k - o.gpu_ms_per_8192) * f;
            npu_out[k] = out_ms(map, cal, k, rows) + cal.glue.sync_ms;
            bytes[k] += 2.0 * rows as f64 * family.cols as f64 * o.k as f64 / 1e3 * o.bytes_per_kop;
        }
    }
    BankSegments {
        gpu,
        npu,
        npu_out,
        bytes,
        chunk,
    }
}
/// Extra GPU ms while NPU chains `npu_this` (moving `nbytes` of DRAM traffic) overlap the other bank's GPU segment:
/// the legacy burst term plus the per-byte contention (`ms_per_gb`).
pub fn contention_extra(cal: &Calibration, gpu_other: f64, npu_this: f64, nbytes: f64) -> f64 {
    (cal.contention.alpha_overlap - 1.0) * gpu_other.min(npu_this)
        + cal.contention.ms_per_gb * nbytes / 1e9
}
/// Per attention layer stall of the order `C_A C_B P_A P_B` (a moved kind-1 out family): I(l+1,A) overlaps P_B(l),
/// I(l+1,B) overlaps C_A(l+1), O_A overlaps C_B, O_B overlaps P_A; each chain stalls by `max(0, chain - window)`.
/// Windows are the measured 16K-pair Core/post phases scaled by rows (Core is linear in the prefix); the post window
/// loses the GPU o-proj GEMM and gains the K-scaled pack; the in-proj chain is the calibrated chain scaled by its
/// columns.
fn qsa_order_stall(
    map: &StageMap,
    cal: &Calibration,
    moved: Moved,
    (r1, p1): (usize, usize),
    (r2, p2): (usize, usize),
    o1: f64,
    o2: f64,
) -> f64 {
    let q = cal
        .qsa_order
        .as_ref()
        .expect("validated: the attention out family requires qsa_order");
    let o = &out_family(map, cal, 1).1;
    let (f1, f2) = (r1 as f64 / CHUNK as f64, r2 as f64 / CHUNK as f64);
    let core = |p: usize| q.core_a_ms + (q.core_b_ms - q.core_a_ms) * p as f64 / CHUNK as f64;
    let post = q.post_ms - o.gpu_ms_per_8192
        + cal.glue.pack_ms_8192 * o.k as f64 / map.kinds[1].in_k as f64;
    let qsa_cols = |all: bool| -> usize {
        map.groups
            .iter()
            .enumerate()
            .filter(|(i, g)| g.kind == 1 && (all || moved.inproj[*i]))
            .map(|(_, g)| cal.families[g.name].cols)
            .sum()
    };
    let reference = npu_ms(map, cal, 1, CHUNK, qsa_cols(true));
    let cols = qsa_cols(false);
    let chain = |r: usize| {
        if cols == 0 {
            0.0
        } else {
            q.inproj_chain_ms * npu_ms(map, cal, 1, r, cols) / reference
        }
    };
    let (c_a, c_b) = (core(p1) * f1, core(p2) * f2);
    let (p_a, p_b) = (post * f1, post * f2);
    (chain(r1) - p_b).max(0.0)
        + (chain(r2) - c_a).max(0.0)
        + (o1 - c_b).max(0.0)
        + (o2 - p_a).max(0.0)
}
/// Steady-state stall per layer. A moved attention out family forces the order `C_A C_B P_A P_B`
/// (`qsa_order_stall`). Everywhere else the NPU chains of a bank (in-proj, plus the out chain when moved) stall the
/// GPU by what exceeds the other bank's GPU window: zero at balanced pairs (fill only), real for an unbalanced pair.
#[allow(clippy::too_many_arguments)]
fn layer_stall(
    map: &StageMap,
    cal: &Calibration,
    moved: Moved,
    k: usize,
    a: (usize, usize),
    b: (usize, usize),
    sa: &BankSegments,
    sb: &BankSegments,
) -> f64 {
    if k == 1 && moved.out[1] {
        return qsa_order_stall(map, cal, moved, a, b, sa.npu_out[1], sb.npu_out[1]);
    }
    (sb.npu[k] + sb.npu_out[k] - sa.gpu[k]).max(0.0)
        + (sa.npu[k] + sa.npu_out[k] - sb.gpu[k]).max(0.0)
}
#[allow(clippy::too_many_arguments)]
pub fn pair_time(
    map: &StageMap,
    cal: &Calibration,
    families: FamilySet,
    r1: usize,
    p1: usize,
    r2: usize,
    p2: usize,
) -> PairTime {
    pair_time_moved(map, cal, Moved::of(map, families), r1, p1, r2, p2)
}
#[allow(clippy::too_many_arguments)]
pub fn pair_time_moved(
    map: &StageMap,
    cal: &Calibration,
    moved: Moved,
    r1: usize,
    p1: usize,
    r2: usize,
    p2: usize,
) -> PairTime {
    let a = bank_segments_moved(map, cal, moved, r1, p1);
    let b = bank_segments_moved(map, cal, moved, r2, p2);
    let mut pr = PairTime {
        t: 0.0,
        stall: 0.0,
        extra: 0.0,
        fill: if a.npu[0] != 0.0 { a.npu[0] } else { a.npu[1] },
        gpu_only: a.chunk.total + b.chunk.total,
    };
    for k in 0..KINDS {
        let layers = layers(map, k);
        let s = layer_stall(map, cal, moved, k, (r1, p1), (r2, p2), &a, &b);
        let e = contention_extra(cal, a.gpu[k], b.npu[k] + b.npu_out[k], b.bytes[k])
            + contention_extra(cal, b.gpu[k], a.npu[k] + a.npu_out[k], a.bytes[k]);
        pr.t += layers * (a.gpu[k] + b.gpu[k] + s + e);
        pr.stall += layers * s;
        pr.extra += layers * e;
    }
    pr.t += pr.fill + cal.gpu.fixed_per_chunk_ms * 2.0;
    pr
}
fn beats(cal: &Calibration, pipe: f64, gpu: f64) -> bool {
    pipe < (1.0 - cal.margin) * gpu
}
fn half(rows: usize) -> usize {
    (rows / 2).div_ceil(64) * 64
}
#[derive(Clone, Copy)]
struct Choice {
    consumed: usize,
    split: Option<usize>,
}

/// GPU wins exact ties; natural earlier pairs win ties against rebalance/re-bank.
/// Only whole groups move; partial-group masks stay GPU-only.
pub fn plan(map: &StageMap, cal: &Calibration, families: FamilySet, rows: usize) -> Plan {
    plan_moved(map, cal, Moved::of(map, families), rows)
}
pub fn plan_moved(map: &StageMap, cal: &Calibration, moved: Moved, rows: usize) -> Plan {
    plan_with(map, cal, moved, rows, false)
}
/// `plan` restricted to what the executor runs: natural 8192 + B pairs and the rebalanced last pair. The single-chunk
/// split and the re-banked plan are skipped, so a request never gets a plan whose pairs `executable_plan` would
/// project back to GPU chunks (with live rate scales the re-banked plan can win the model by a fraction of a percent).
pub fn plan_executable(
    map: &StageMap,
    cal: &Calibration,
    families: FamilySet,
    rows: usize,
) -> Plan {
    plan_with(map, cal, Moved::of(map, families), rows, true)
}
fn plan_with(
    map: &StageMap,
    cal: &Calibration,
    moved: Moved,
    rows: usize,
    executable_only: bool,
) -> Plan {
    let len = rows.div_ceil(CHUNK);
    let g: Vec<_> = (0..len)
        .map(|i| gpu_chunk(map, cal, (rows - i * CHUNK).min(CHUNK), i * CHUNK).total)
        .collect();
    let t_gpu_ms = g.iter().sum();
    if !moved.supported {
        return Plan {
            steps: (0..len)
                .map(|i| Step::Gpu {
                    chunk: i * CHUNK..((i + 1) * CHUNK).min(rows),
                })
                .collect(),
            t_gpu_ms,
            t_pipe_ms: t_gpu_ms,
        };
    }
    let mut best = vec![0.0; len + 1];
    let mut choice = vec![
        Choice {
            consumed: 1,
            split: None
        };
        len + 1
    ];
    for i in 1..=len {
        best[i] = best[i - 1] + g[i - 1];
        if i >= 2 {
            let start = (i - 2) * CHUNK;
            let last = (rows - (i - 1) * CHUNK).min(CHUNK);
            let pr = pair_time_moved(map, cal, moved, CHUNK, start, last, start + CHUNK);
            if beats(cal, pr.t, pr.gpu_only) && best[i - 2] + pr.t < best[i] {
                best[i] = best[i - 2] + pr.t;
                choice[i] = Choice {
                    consumed: 2,
                    split: Some(CHUNK),
                };
            }
        }
    }
    if len >= 2 && rows % CHUNK != 0 {
        let start = (len - 2) * CHUNK;
        let total = rows - start;
        let h1 = half(total);
        let h2 = total - h1;
        let pr = pair_time_moved(map, cal, moved, h1, start, h2, start + h1);
        if h1.min(h2) >= cal.gpu.min_bank_rows
            && beats(cal, pr.t, g[len - 2] + g[len - 1])
            && best[len - 2] + pr.t < best[len]
        {
            best[len] = best[len - 2] + pr.t;
            choice[len] = Choice {
                consumed: 2,
                split: Some(h1),
            };
        }
    }
    if !executable_only && len == 1 && rows >= 2 * cal.gpu.min_bank_rows {
        let h1 = half(rows);
        let t = pair_time_moved(map, cal, moved, h1, 0, rows - h1, h1).t + cal.gpu.boundary_tax_ms;
        if beats(cal, t, g[0]) && t < best[1] {
            best[1] = t;
            choice[1] = Choice {
                consumed: 1,
                split: Some(h1),
            };
        }
    }
    let nb = 2 * rows.div_ceil(2 * CHUNK);
    if !executable_only && len >= 2 && nb > 0 && rows / nb >= cal.gpu.min_bank_rows {
        let size = (rows / nb).div_ceil(64) * 64;
        let last = rows as isize - (size * (nb - 1)) as isize;
        // Python can construct a nonpositive final bank for very small banks;
        // no executable row range exists in that case.
        if last > 0 {
            let mut t = 0.0;
            let mut eligible = true;
            for k in (0..nb).step_by(2) {
                let r2 = if k + 2 == nb { last as usize } else { size };
                let pr = pair_time_moved(map, cal, moved, size, k * size, r2, (k + 1) * size);
                t += pr.t;
                if cal.margin != 0.0 && !beats(cal, pr.t, pr.gpu_only) {
                    eligible = false;
                }
            }
            t += cal.gpu.boundary_tax_ms * nb.saturating_sub(len) as f64;
            if eligible && beats(cal, t, t_gpu_ms) && t < best[len] {
                return Plan {
                    steps: (0..nb)
                        .step_by(2)
                        .map(|k| Step::Pipe {
                            a: k * size..(k + 1) * size,
                            b: (k + 1) * size..if k + 2 == nb { rows } else { (k + 2) * size },
                        })
                        .collect(),
                    t_gpu_ms,
                    t_pipe_ms: t,
                };
            }
        }
    }
    let mut steps = Vec::with_capacity(len);
    let mut i = len;
    while i > 0 {
        let c = choice[i];
        let start = (i - c.consumed) * CHUNK;
        let end = (i * CHUNK).min(rows);
        steps.push(match c.split {
            Some(split) => Step::Pipe {
                a: start..start + split,
                b: start + split..end,
            },
            None => Step::Gpu { chunk: start..end },
        });
        i -= c.consumed;
    }
    steps.reverse();
    Plan {
        steps,
        t_gpu_ms,
        t_pipe_ms: best[len],
    }
}
/// First 64-row grid point in 1024..=49152 whose suffix always beats GPU-only.
pub fn crossover(map: &StageMap, cal: &Calibration, families: FamilySet) -> Option<usize> {
    let mut first = None;
    for rows in (1024..=6 * CHUNK).step_by(64) {
        let p = plan(map, cal, families, rows);
        if p.t_pipe_ms < p.t_gpu_ms {
            if first.is_none() {
                first = Some(rows);
            }
        } else {
            first = None;
        }
    }
    first
}

pub const SELECTION_ROWS: [usize; 3] = [16384, 32768, 65536];
/// One row of the load-time eligibility table, in ascending subset-bit order.
#[derive(Clone, Debug)]
pub struct SelectionRow {
    pub families: FamilySet,
    /// Names of the subset's groups, in map order.
    pub groups: Vec<&'static str>,
    pub t_pipe_ms: [f64; 3],
    pub t_gpu_ms: [f64; 3],
    pub relative: [f64; 3],
    pub mean_relative: f64,
    pub beats_margin_16384: bool,
    /// Every family of the subset is `selectable`; only these can win.
    pub selectable: bool,
}
#[derive(Clone, Debug)]
pub struct SelectionReport {
    pub subsets: Vec<SelectionRow>,
    /// Minimum-mean candidate before the 16384-row margin gate.
    pub best_families: FamilySet,
}
/// Mean relative time, then fewer groups, then the sorted group names.
fn compare_selection(map: &StageMap, a: &SelectionRow, b: &SelectionRow) -> std::cmp::Ordering {
    let names = map.sorted_group_names();
    a.mean_relative
        .total_cmp(&b.mean_relative)
        .then_with(|| a.groups.len().cmp(&b.groups.len()))
        .then_with(|| {
            names
                .into_iter()
                .filter(|name| a.groups.contains(name))
                .cmp(names.into_iter().filter(|name| b.groups.contains(name)))
        })
}
/// Names of the groups whose bit is set in `bits`, in map order.
pub fn group_names(map: &StageMap, bits: u8) -> Vec<&'static str> {
    map.groups
        .iter()
        .enumerate()
        .filter(|(i, _)| bits & (1 << i) != 0)
        .map(|(_, g)| g.name)
        .collect()
}
/// Whether every group of `bits` is `selectable` in the calibration.
pub fn selectable(map: &StageMap, cal: &Calibration, bits: u8) -> bool {
    group_names(map, bits)
        .into_iter()
        .all(|name| cal.families[name].selectable)
}
/// Select the minimum mean T_pipe/T_gpu across the three fixed prompt sizes among the subsets whose every family is
/// `selectable` (the out families are not in the in-proj mask and stay at zero selection).
/// Gate that winner (not a runner-up) by the strict 16384-row margin.
pub fn select_families(map: &StageMap, cal: &Calibration) -> (FamilySet, SelectionReport) {
    let mut subsets = Vec::with_capacity((1 << GROUPS) - 1);
    for bits in 1..1u8 << GROUPS {
        let families = map.subset(bits);
        let plans = SELECTION_ROWS.map(|rows| plan(map, cal, families, rows));
        let t_pipe_ms = std::array::from_fn(|i| plans[i].t_pipe_ms);
        let t_gpu_ms = std::array::from_fn(|i| plans[i].t_gpu_ms);
        // A zero-cost GPU baseline has no possible strict improvement.
        let relative = std::array::from_fn(|i| {
            if t_gpu_ms[i] == 0.0 {
                1.0
            } else {
                t_pipe_ms[i] / t_gpu_ms[i]
            }
        });
        subsets.push(SelectionRow {
            families,
            groups: group_names(map, bits),
            t_pipe_ms,
            t_gpu_ms,
            relative,
            mean_relative: relative.iter().sum::<f64>() / 3.0,
            beats_margin_16384: beats(cal, t_pipe_ms[0], t_gpu_ms[0]),
            selectable: selectable(map, cal, bits),
        });
    }
    let Some(best) = subsets
        .iter()
        .filter(|row| row.selectable)
        .min_by(|a, b| compare_selection(map, a, b))
    else {
        return (
            FamilySet::NONE,
            SelectionReport {
                subsets,
                best_families: FamilySet::NONE,
            },
        );
    };
    let families = if best.beats_margin_16384 {
        best.families
    } else {
        FamilySet::NONE
    };
    let best_families = best.families;
    (
        families,
        SelectionReport {
            subsets,
            best_families,
        },
    )
}

/// Admit any nonempty calibrated family subset.
pub fn auto_admission(
    map: &StageMap,
    cal: &Calibration,
    current: &CalKey,
) -> Result<FamilySet, String> {
    cal.key_matches(current)?;
    let (families, _) = select_families(map, cal);
    if families.is_empty() {
        return Err("calibration selects no families".into());
    }
    Ok(families)
}

/// Whether `a` + `b` is a pair the executor can run: the natural 8192 + B pair
/// (B in `[minimum, 8192]`) or the planner's last-two-chunk rebalanced pair, two
/// banks of about equal rows (64-row aligned split, each in `[minimum, 8192]`)
/// that end the request. Split and re-bank geometries match neither.
fn executable_pair(a: &Range<usize>, b: &Range<usize>, rows: usize, minimum: usize) -> bool {
    if a.start % CHUNK != 0 || b.start != a.end {
        return false;
    }
    let natural = a.len() == CHUNK
        && (minimum..=CHUNK).contains(&b.len())
        && b.end == (a.start + 2 * CHUNK).min(rows);
    let rebalanced = a.len() != CHUNK
        && a.len() % 64 == 0
        && a.len().min(b.len()) >= minimum
        && a.len().max(b.len()) <= CHUNK
        && b.end == rows
        && b.end - a.start > CHUNK;
    natural || rebalanced
}

/// Project S5 onto the executor: natural pairs at the natural chunk boundaries
/// and the rebalanced last pair (`pipe-rebalanced`). Split/rebanked steps stay
/// GPU-only; never substitute force-pairs.
pub fn executable_plan(
    map: &StageMap,
    cal: &Calibration,
    planned: &Plan,
    rows: usize,
    n_star: Option<usize>,
) -> Plan {
    executable_plan_for(map, cal, map.all(), planned, rows, n_star)
}
/// `executable_plan` whose predicted `t_pipe_ms` prices each pair with `families` (the set `planned` was made for).
pub fn executable_plan_for(
    map: &StageMap,
    cal: &Calibration,
    families: FamilySet,
    planned: &Plan,
    rows: usize,
    n_star: Option<usize>,
) -> Plan {
    let minimum = 512.max(cal.gpu.min_bank_rows);
    let armed = n_star.is_some_and(|n| rows >= n);
    let mut steps = Vec::with_capacity(rows.div_ceil(CHUNK));
    let mut at = 0;
    let mut t_pipe_ms = 0.0;
    while at < rows {
        let pair = armed
            .then(|| {
                planned.steps.iter().find_map(|step| match step {
                    Step::Pipe { a, b }
                        if a.start == at && executable_pair(a, b, rows, minimum) =>
                    {
                        Some((a.clone(), b.clone()))
                    }
                    _ => None,
                })
            })
            .flatten();
        if let Some((a, b)) = pair {
            t_pipe_ms += pair_time(map, cal, families, a.len(), a.start, b.len(), b.start).t;
            at = b.end;
            steps.push(Step::Pipe { a, b });
        } else {
            let end = (at + CHUNK).min(rows);
            t_pipe_ms += gpu_chunk(map, cal, end - at, at).total;
            steps.push(Step::Gpu { chunk: at..end });
            at = end;
        }
    }
    Plan {
        steps,
        t_gpu_ms: planned.t_gpu_ms,
        t_pipe_ms,
    }
}

/// Rows of the planned pair starting at `offset`: bank A, then bank B.
pub fn planned_pair(planned: &Plan, offset: usize) -> Option<(usize, usize)> {
    planned.steps.iter().find_map(|step| match step {
        Step::Pipe { a, b } if a.start == offset => Some((a.len(), b.len())),
        _ => None,
    })
}

pub fn planned_pair_rows(planned: &Plan, offset: usize) -> Option<usize> {
    planned_pair(planned, offset).map(|(a, b)| a + b)
}
/// The max-window budget: a bank below `CHUNK` rows runs the co-processor program of its own padded row count
/// (`padded_rows`), one hardware context per distinct count, at most `max_contexts` per load. Admits the counts of
/// `executable`'s pairs into `admitted` in first-use order (the order the co-processor meets them). A pair that would
/// exceed the budget is dropped and `replan` rebuilds the plan from the remaining pairs, so its rows run GPU-native
/// rather than through a full `CHUNK`-row chain.
pub fn admit_wave_budget(
    executable: Plan,
    admitted: &mut Vec<usize>,
    max_contexts: usize,
    padded_rows: impl Fn(usize) -> usize,
    replan: impl Fn(&Plan) -> Plan,
) -> Plan {
    let mut kept = executable.clone();
    kept.steps.retain(|step| {
        let Step::Pipe { a, b } = step else {
            return true;
        };
        let mut next = admitted.clone();
        for rows in [a.len(), b.len()].map(&padded_rows) {
            if rows < CHUNK && !next.contains(&rows) {
                next.push(rows);
            }
        }
        let fits = next.len() <= max_contexts;
        if fits {
            *admitted = next;
        }
        fits
    });
    if kept.steps.len() == executable.steps.len() {
        executable
    } else {
        replan(&kept)
    }
}
/// Load-time log context, all caller-supplied deterministic state.
pub struct LoadLogKey<'a> {
    pub current: &'a CalKey,
    pub mode: &'a str,
    pub families: FamilySet,
}
pub fn load_log_line(
    map: &StageMap,
    cal: &Calibration,
    key: &LoadLogKey<'_>,
    n_star: Option<usize>,
    t16384: &Plan,
    t65536: &Plan,
) -> String {
    let tag = map.tag;
    if let Err(reason) = cal.key_matches(key.current) {
        return format!("{tag}: off ({reason})");
    }
    let bytes = serde_json::to_vec(cal).expect("validated calibration JSON");
    let hash = Sha256::digest(&bytes);
    let names: Vec<_> = map
        .groups
        .iter()
        .zip(map.owned_groups(key.families))
        .filter_map(|(g, yes)| yes.then_some(g.name))
        .collect();
    format!("{tag}: mode={} contract={} families={} cal_key={} cal_sha256={:x} n*={} T(16384)={:.6}/{:.6} T(65536)={:.6}/{:.6}",
        key.mode, cal.npu.epilogue, names.join(","), serde_json::to_string(&cal.key).expect("key JSON"), hash,
        n_star.map_or_else(|| "none".into(), |n| n.to_string()), t16384.t_pipe_ms, t16384.t_gpu_ms, t65536.t_pipe_ms, t65536.t_gpu_ms)
}
pub fn request_log_line(map: &StageMap, rows: usize, plan: &Plan) -> String {
    let steps: Vec<_> = plan
        .steps
        .iter()
        .map(|s| match s {
            Step::Gpu { chunk } => format!("gpu {}..{}", chunk.start, chunk.end),
            Step::Pipe { a, b } => format!("pipe {}..{}+{}..{}", a.start, a.end, b.start, b.end),
        })
        .collect();
    let percent = if plan.t_gpu_ms == 0.0 {
        0.0
    } else {
        100.0 * (plan.t_pipe_ms - plan.t_gpu_ms) / plan.t_gpu_ms
    };
    format!(
        "{}: rows={rows} plan=[{}] predicted={percent:+.1}%",
        map.tag,
        steps.join(" | ")
    )
}

#[cfg(test)]
mod tests {
    use super::super::stage::{tests::DENSE, LayerKindDecl};
    use super::*;

    fn calibration() -> Calibration {
        let group = |cols: usize, layer: &str| serde_json::json!({"layer": layer, "cols": cols, "out_dtype": "bf16"});
        serde_json::from_value(serde_json::json!({
            "schema_version": 4,
            "key": {"device": {"arch": "gfx1151", "pci_bdf": "0000:c5:00.0", "npu_fw_version": "test"},
                    "kernel": {"uname_release": "test"}, "model_trunk_index_sha256": "0".repeat(64),
                    "epilogue": "bf16", "trunk_route": "stock"},
            "gpu": {"trunk_in_ms_per_8192": {"qkv": 550.0, "zab": 365.0, "q": 220.0, "kv": 44.0},
                    "uniform_per_layer_ms": 40.0, "gdn_core_per_layer_ms": 8.0, "qsa_per_layer_ms": 30.0,
                    "qsa_growth_per_layer_ms_per_8192_prefix": 2.0, "fixed_per_chunk_ms": 10.0,
                    "boundary_tax_ms": 300.0, "min_bank_rows": 2048},
            "families": {"qkv": group(10240, "gdn"), "zab": group(6240, "gdn"), "q": group(12288, "fa"),
                         "kv": group(2048, "fa")},
            "glue": {"pack_ms_8192": 1.0, "sync_ms": 4.0, "consume_mode": "direct",
                     "consume_ms_per_col_8192": {"bf16": 0.0, "f32": 0.0}},
            "npu": {"tops_corun": 200.0, "bytes_per_kop": 2.4, "epilogue": "bf16", "panel_cols": 512},
            "contention": {"alpha_overlap": 1.1, "model": "test"},
            "margin": 0.0
        }))
        .unwrap()
    }

    /// The planner prices a map's own layer counts and reduction depth, not Flash-Next's 36/12 layers and K = 2560.
    #[test]
    fn costs_follow_the_map_layer_counts_and_reduction_depth() {
        let cal = calibration();
        cal.validate(&DENSE).unwrap();
        let chunk = gpu_chunk(&DENSE, &cal, CHUNK, 0);
        assert_eq!(chunk.total, 48.0 * 48.0 + 16.0 * 70.0 + 1179.0 + 10.0);
        let shallow = StageMap {
            kinds: DENSE.kinds.map(|k| LayerKindDecl {
                in_k: k.in_k / 2,
                ..k
            }),
            ..DENSE
        };
        assert_eq!(
            npu_ms(&DENSE, &cal, 0, CHUNK, 10240),
            2.0 * npu_ms(&shallow, &cal, 0, CHUNK, 10240)
        );
        let segments = bank_segments(&DENSE, &cal, DENSE.subset(0b0001), CHUNK, 0);
        // qkv moved: its whole-stack GPU cost leaves the per-layer GDN segment, the FA segment keeps all of its own.
        assert_eq!(segments.gpu[0], 48.0 + 365.0 / 48.0 + 1.0);
        assert_eq!(segments.gpu[1], 70.0 + 264.0 / 16.0);
        assert!(segments.npu[0] > 0.0 && segments.npu[1] == 0.0);
        let plan = plan(&DENSE, &cal, DENSE.all(), 16384);
        assert!(matches!(plan.steps[..], [Step::Pipe { .. }]), "{plan:?}");
    }
}
