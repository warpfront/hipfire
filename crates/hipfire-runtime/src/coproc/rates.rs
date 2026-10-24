// SPDX-License-Identifier: Apache-2.0
//! Live rate scales for the `auto` co-processor plan. The built-in calibration is the prior; measured pairs
//! learn a GPU-window scale and an NPU-chain scale that multiply the prior's terms (`scaled_calibration`).
//! Forced modes (`on`, `emulate`) never use the estimator: they plan from the static calibration only.
use super::plan::{
    bank_segments_moved, contention_extra, gpu_chunk, group_names, pair_time_moved,
    plan_executable, selectable, Moved,
};
use super::stage::{FamilySet, StageMap, GROUPS};
use super::Calibration;

/// Weight of a new sample in the scale EWMAs.
pub const EWMA_ALPHA: f64 = 0.25;
/// Raw ratios and scales are clamped to `SCALE_MIN..=SCALE_MAX`.
pub const SCALE_MIN: f64 = 0.5;
pub const SCALE_MAX: f64 = 2.0;
/// Samples whose modelled window or NPU time is below this many ms are ignored.
pub const MIN_MODEL_MS: f64 = 50.0;
/// Relative EWMA/snapshot difference above which an unfrozen snapshot takes the EWMA values.
pub const APPLY_EPS: f64 = 0.01;
/// Valid samples needed before the snapshot may freeze.
pub const FREEZE_AFTER: u32 = 6;
/// Consecutive valid samples that must each move both EWMAs by less than `FREEZE_EPS` (relative) to freeze.
pub const FREEZE_WINDOW: u32 = 4;
pub const FREEZE_EPS: f64 = 0.01;
/// A frozen snapshot reopens after `REOPEN_CONFIRM` consecutive valid samples whose raw ratio deviates from the
/// frozen scale by more than `REOPEN_EPS` (relative) in the same direction.
pub const REOPEN_EPS: f64 = 0.05;
pub const REOPEN_CONFIRM: u32 = 3;

/// Modelled times of one executed pair, comparable with the measured NPU chain and GPU window sums.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairModel {
    pub npu_ms: f64,
    pub window_ms: f64,
    pub gpu_only_ms: f64,
    pub savings_ms: f64,
}

/// Model of one executed pair. `kinds[l]` is the kind (`StageMap::kinds` index) of layer `l`; `rows`/`prefix`
/// describe banks A, B.
/// `npu_ms`: the sum of the in-proj NPU chain (`npu[k]` of `bank_segments_moved`) over every (layer, bank) whose
/// kind has a nonzero chain under `moved`. `window_ms`: the sum over the same chains except (0, A) (its window is
/// bank-B initialization) of the GPU window hiding the chain, plus its contention: chain (l, A), l >= 1, is hidden
/// by bank B's segment of layer l-1; chain (l, B) by bank A's segment of layer l.
/// `gpu_only_ms`: both banks run GPU-only. `savings_ms`: what pairing saves the GPU before any NPU wait: the moved
/// families' GPU projection time minus the pair's GPU glue (pack, consume, visit) and its DRAM contention. The pair
/// loses to GPU-only once the GPU's measured NPU waits exceed it; unlike `gpu_only_ms` it carries no prefix-dependent
/// term, so the model's error at depth does not move it.
pub fn pair_model(
    map: &StageMap,
    cal: &Calibration,
    moved: Moved,
    rows: [usize; 2],
    prefix: [usize; 2],
    kinds: &[usize],
) -> PairModel {
    let a = bank_segments_moved(map, cal, moved, rows[0], prefix[0]);
    let b = bank_segments_moved(map, cal, moved, rows[1], prefix[1]);
    let (mut npu_ms, mut window_ms) = (0.0, 0.0);
    for (l, &k) in kinds.iter().enumerate() {
        if a.npu[k] != 0.0 {
            npu_ms += a.npu[k];
            if l >= 1 {
                let window = b.gpu[kinds[l - 1]];
                window_ms += window + contention_extra(cal, window, a.npu[k], a.bytes[k]);
            }
        }
        if b.npu[k] != 0.0 {
            npu_ms += b.npu[k];
            let window = a.gpu[k];
            window_ms += window + contention_extra(cal, window, b.npu[k], b.bytes[k]);
        }
    }
    let mut savings_ms =
        -pair_time_moved(map, cal, moved, rows[0], prefix[0], rows[1], prefix[1]).extra;
    for (s, (r, p)) in [&a, &b]
        .into_iter()
        .zip([(rows[0], prefix[0]), (rows[1], prefix[1])])
    {
        let chunk = gpu_chunk(map, cal, r, p);
        let moved_trunk: f64 = (0..GROUPS)
            .filter(|&i| moved.inproj[i])
            .map(|i| chunk.trunk[i])
            .sum();
        let segments: f64 = kinds.iter().map(|&kind| s.gpu[kind]).sum();
        let glue = segments - (chunk.total - cal.gpu.fixed_per_chunk_ms - moved_trunk);
        savings_ms += moved_trunk - glue;
    }
    PairModel {
        npu_ms,
        window_ms,
        savings_ms,
        gpu_only_ms: gpu_chunk(map, cal, rows[0], prefix[0]).total
            + gpu_chunk(map, cal, rows[1], prefix[1]).total,
    }
}

/// One measured pair against its model: summed NPU chain time and summed GPU window time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RateSample {
    pub npu_measured_ms: f64,
    pub npu_model_ms: f64,
    pub window_measured_ms: f64,
    pub window_model_ms: f64,
}

/// The estimator's current calibration (`scaled_calibration(prior, gpu_scale, npu_scale)`). `generation` advances
/// whenever it changes; `frozen` means it is fixed until a reopen.
#[derive(Clone, Debug)]
pub struct RateSnapshot {
    pub calibration: Calibration,
    pub generation: u64,
    pub gpu_scale: f64,
    pub npu_scale: f64,
    pub samples: u32,
    pub frozen: bool,
}

/// Prior calibration scaled by learned GPU-window and NPU-chain rates (EWMA of measured/model, clamped), frozen once
/// settled. Owned by one process; the forced modes never create one.
pub struct RateEstimator {
    prior: Calibration,
    snapshot: RateSnapshot,
    ewma: Option<(f64, f64)>,
    /// Consecutive valid samples that moved both EWMAs by less than `FREEZE_EPS`.
    calm: u32,
    /// Frozen only: direction of the current run of deviating samples and its length.
    deviating: Option<((i8, i8), u32)>,
}

fn relative(value: f64, base: f64) -> f64 {
    (value - base).abs() / base
}

impl RateEstimator {
    pub fn new(prior: Calibration) -> Self {
        let snapshot = RateSnapshot {
            calibration: prior.clone(),
            generation: 0,
            gpu_scale: 1.0,
            npu_scale: 1.0,
            samples: 0,
            frozen: false,
        };
        Self {
            prior,
            snapshot,
            ewma: None,
            calm: 0,
            deviating: None,
        }
    }
    pub fn prior(&self) -> &Calibration {
        &self.prior
    }
    pub fn snapshot(&self) -> &RateSnapshot {
        &self.snapshot
    }

    fn publish(&mut self, gpu_scale: f64, npu_scale: f64) {
        self.snapshot.calibration = scaled_calibration(&self.prior, gpu_scale, npu_scale);
        self.snapshot.gpu_scale = gpu_scale;
        self.snapshot.npu_scale = npu_scale;
        self.snapshot.generation += 1;
    }

    /// Feeds one measured pair. Invalid samples (non-finite, model below `MIN_MODEL_MS`, measured not positive)
    /// are ignored. Returns true iff the snapshot changed (its generation advanced).
    pub fn observe(&mut self, sample: &RateSample) -> bool {
        let values = [
            sample.npu_measured_ms,
            sample.npu_model_ms,
            sample.window_measured_ms,
            sample.window_model_ms,
        ];
        if !values.iter().all(|v| v.is_finite())
            || sample.npu_model_ms < MIN_MODEL_MS
            || sample.window_model_ms < MIN_MODEL_MS
            || sample.npu_measured_ms <= 0.0
            || sample.window_measured_ms <= 0.0
        {
            return false;
        }
        let raw_gpu =
            (sample.window_measured_ms / sample.window_model_ms).clamp(SCALE_MIN, SCALE_MAX);
        let raw_npu = (sample.npu_measured_ms / sample.npu_model_ms).clamp(SCALE_MIN, SCALE_MAX);
        self.snapshot.samples = self.snapshot.samples.saturating_add(1);
        let (gpu, npu) = match self.ewma {
            None => {
                self.calm = 0;
                (raw_gpu, raw_npu)
            }
            Some((g, n)) => {
                let next = (
                    g + EWMA_ALPHA * (raw_gpu - g),
                    n + EWMA_ALPHA * (raw_npu - n),
                );
                self.calm = if relative(next.0, g) < FREEZE_EPS && relative(next.1, n) < FREEZE_EPS
                {
                    self.calm + 1
                } else {
                    0
                };
                next
            }
        };
        self.ewma = Some((gpu, npu));
        if self.snapshot.frozen {
            let direction = |raw: f64, base: f64| -> i8 {
                let d = (raw - base) / base;
                if d > REOPEN_EPS {
                    1
                } else if d < -REOPEN_EPS {
                    -1
                } else {
                    0
                }
            };
            let dir = (
                direction(raw_gpu, self.snapshot.gpu_scale),
                direction(raw_npu, self.snapshot.npu_scale),
            );
            if dir == (0, 0) {
                self.deviating = None;
                return false;
            }
            let run = match self.deviating {
                Some((d, n)) if d == dir => n + 1,
                _ => 1,
            };
            self.deviating = Some((dir, run));
            if run < REOPEN_CONFIRM {
                return false;
            }
            self.snapshot.frozen = false;
            self.deviating = None;
            self.calm = 0;
            self.publish(gpu, npu);
            return true;
        }
        let changed = relative(gpu, self.snapshot.gpu_scale) > APPLY_EPS
            || relative(npu, self.snapshot.npu_scale) > APPLY_EPS;
        if changed {
            self.publish(gpu, npu);
        }
        if self.snapshot.samples >= FREEZE_AFTER && self.calm >= FREEZE_WINDOW {
            self.snapshot.frozen = true;
            self.deviating = None;
        }
        changed
    }
}

/// `prior` with its GPU terms multiplied by `gpu_scale` and its NPU terms by `npu_scale` (rates divide).
/// GPU: every `gpu.*` ms term (not `min_bank_rows`), glue `pack_ms_8192`, `visit_ms_8192` and
/// `consume_ms_per_col_8192`, out-family `gpu_ms_per_8192` and `qsa_order`, all x `gpu_scale`.
/// NPU: `npu.tops_corun`, `npu.tops_alone` and out-family `tops_alone` / `npu_scale`; `glue.sync_ms` x `npu_scale`.
/// Nothing else changes; scales of 1.0 reproduce `prior` exactly.
pub fn scaled_calibration(prior: &Calibration, gpu_scale: f64, npu_scale: f64) -> Calibration {
    let mut cal = prior.clone();
    let g = &mut cal.gpu;
    for v in g.trunk_in_ms_per_8192.values_mut() {
        *v *= gpu_scale;
    }
    for v in [
        &mut g.uniform_per_layer_ms,
        &mut g.gdn_core_per_layer_ms,
        &mut g.qsa_per_layer_ms,
        &mut g.qsa_growth_per_layer_ms_per_8192_prefix,
        &mut g.fixed_per_chunk_ms,
        &mut g.boundary_tax_ms,
    ] {
        *v *= gpu_scale;
    }
    cal.glue.pack_ms_8192 *= gpu_scale;
    cal.glue.visit_ms_8192 *= gpu_scale;
    for v in cal.glue.consume_ms_per_col_8192.values_mut() {
        *v *= gpu_scale;
    }
    cal.glue.sync_ms *= npu_scale;
    cal.npu.tops_corun /= npu_scale;
    if let Some(t) = &mut cal.npu.tops_alone {
        *t /= npu_scale;
    }
    for out in cal.families.values_mut().filter_map(|f| f.out.as_mut()) {
        out.gpu_ms_per_8192 *= gpu_scale;
        out.tops_alone /= npu_scale;
    }
    if let Some(q) = &mut cal.qsa_order {
        for v in [
            &mut q.core_a_ms,
            &mut q.core_b_ms,
            &mut q.post_ms,
            &mut q.inproj_chain_ms,
        ] {
            *v *= gpu_scale;
        }
    }
    cal
}

/// Fastest group subset of `allowed` for a `rows`-row request: the minimum `plan_executable(..).t_pipe_ms` over the
/// `select_families` subsets that lie within `allowed` and whose families are `selectable`. Ties go to fewer groups,
/// then the sorted group names. `FamilySet::NONE` when no subset's `t_pipe_ms` is below its `t_gpu_ms`.
pub fn choose_families(
    map: &StageMap,
    cal: &Calibration,
    allowed: FamilySet,
    rows: usize,
) -> FamilySet {
    let names = |bits: u8| -> Vec<&'static str> {
        let mut names = group_names(map, bits);
        names.sort_unstable();
        names
    };
    let mut best: Option<(f64, usize, Vec<&'static str>, FamilySet)> = None;
    for bits in 1..1u8 << GROUPS {
        let subset = map.subset(bits);
        if !subset.is_subset_of(allowed) || !selectable(map, cal, bits) {
            continue;
        }
        let p = plan_executable(map, cal, subset, rows);
        if p.t_pipe_ms >= p.t_gpu_ms {
            continue;
        }
        let key = (p.t_pipe_ms, bits.count_ones() as usize, names(bits));
        let better = best.as_ref().is_none_or(|(t, count, order, _)| {
            key.0
                .total_cmp(t)
                .then_with(|| key.1.cmp(count))
                .then_with(|| key.2.cmp(order))
                .is_lt()
        });
        if better {
            best = Some((key.0, key.1, key.2, subset));
        }
    }
    best.map_or(FamilySet::NONE, |(.., subset)| subset)
}
