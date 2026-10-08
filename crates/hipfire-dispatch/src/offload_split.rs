// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Pass-back: **scheduled co-inference of the layers spilled into system RAM**
//! (`memory.offload_exec=passback`).
//!
//! The partial-offload feature spills a contiguous prefix of layers to pinned
//! host RAM (`memory.gpu_layer_budget`) and executes their weight-reading GEMVs
//! through one of two engines: `pcie` — the GPU reading the host-mapped weights
//! over the link — or `cpu` — the CPU SIMD kernels reading host RAM directly.
//! Pass-back **runs both engines on the same spilled step, concurrently**. It is a
//! mixture of those two existing paths working in concert, not a third engine and
//! not a fallback: the GPU arm *is* the `pcie` path restricted to a row range, the
//! CPU arm *is* the `cpu` path restricted to the rest, and `share = 0` is
//! byte-identical to `memory.offload_exec=cpu`.
//!
//! Why it pays: a CPU-only step is a host sync point — a blocking D2H, a host
//! GEMV, a blocking H2D — so the GPU sits idle for most of its wall
//! (`docs/perf-checkpoints/2026-10-01-offload-passback-headroom-idle.md` measured
//! 77.3–94.6 % on the 9B at 8–24 of 32 layers spilled), while a CPU GEMV stream
//! and a GPU PCIe read of the *same* host-mapped pages are near-additive up to
//! 75–78 GB/s combined against ~52 GB/s for the CPU alone. Co-inference spends
//! that idle window on the same host bytes.
//!
//! **The schedule is per step, not per layer.** Consecutive layers are serially
//! dependent through the residual stream, so moving a whole *layer* between the
//! engines only swaps who idles; the mixture has to be *within* a step. A spilled
//! step's output rows are independent, so the step is divided by output rows: rows
//! `[0, g)` go to the GPU (the `pcie` path) and rows `[g, m)` stay on the CPU (the
//! `cpu` path), concurrently. A step that ends up wholly on one engine is a
//! **degenerate point of this schedule** (`g → 0` for an all-CPU step), not a
//! validation failure and not a fallback to a worse path — the eligibility gates
//! below only bound which steps can be co-inferenced at all.
//!
//! Two properties make it correct and cheap:
//!
//! * **Both arms are the existing paths, over a row range.** The GPU arm is
//!   `pcie` mode's launch restricted to rows `[0, g)`
//!   ([`crate::pipeline::steps::launch_op_rows`]); the CPU arm is `cpu` mode's
//!   step restricted to rows `[g, m)` ([`crate::cpu_exec::cpu_arm_prepare`] /
//!   [`crate::cpu_exec::cpu_arm_finish`]). No third engine and no new numerics,
//!   which is also why a share of `0` is byte-identical to
//!   `memory.offload_exec=cpu`.
//! * **No second stream and no events.** The overlap is an ordering result:
//!   issue the blocking D2H *before* enqueueing the GPU arm (so it drains only
//!   the step's producer), enqueue the GPU arm (async), run the CPU multiply
//!   while that kernel executes, then do the blocking H2D of the CPU's rows —
//!   which is stream-ordered after the GPU arm because both are on the same
//!   (default) stream. The copies are `k*4` bytes down and `(m-g)*4` up (~30 KB
//!   at 9B shapes) against tens of MB of weight bytes, so a two-stream design
//!   would buy the overlap of a ~2 µs copy.
//!
//! The **share** `g` is scheduled, not hardcoded: the optimum is a property of
//! the host (link width, DRAM peak, core count, AVX2), so the first
//! split-eligible step of each `(dtype, k)` seeds itself by timing both engines
//! on that step's own weight buffer, and every split step then refines the share
//! from its own arm timings. No host constant is load-bearing: the probe
//! accelerates convergence and the online controller corrects it.
//!
//! The controller reads the blocking H2D (`join`) relative to the shape's own
//! no-wait floor — the smallest join it has seen, i.e. what the copy costs when
//! the GPU arm finished first. Only the excess over that floor can be a wait for
//! the GPU arm; when there is one, `cpu_ns + excess` recovers the arm's duration
//! and the share moves toward the balance point. When there is not (the common
//! regime: a CPU multiply of hundreds of microseconds against a
//! tens-of-microseconds copy) the CPU is the straggler and the share rises.
//! Reading the raw join as a GPU duration in that regime is how a split ends up
//! pinned to its own floor with both engines idle in turn.
//!
//! Scope is the dense qwen3.5 seam only (the step lists that carry
//! `qkv_via_execute_steps`, `qkvza_via_execute_steps`,
//! `gate_up_via_execute_steps` and the dense `Step::GemvResidual` sites), plus
//! the dense FFN down-projection, which `weight_gemv_swiglu_residual` hands over
//! as a `Step::GemvResidual` when passback is on (it cannot be seen as a step
//! otherwise — the SiLU is fused into its launch). MoE / routed-expert paths
//! never reach it and get no arms here.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use hipfire_config::memory::PassbackShare;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::context::DispatchCtx;
use crate::cpu_exec::{self, HostExec, StepTiming};
use crate::families::gemv::{weight_row_bytes, WeightRef};
use crate::pipeline::steps::{launch_op_rows, GemvInput, Step};
use crate::types::{dtype_rotation_plan, DispatchError, KernelKey, RotationPlan};

/// Split offsets are rounded down to a multiple of this many rows: 2 covers the
/// residual kernels' `row0 = blockIdx.x << 1` + `float2` store at `y + row0` (an
/// odd offset would misalign that store) and 8 keeps every covered format's
/// weight byte offset 4-byte aligned (their row strides are multiples of 8).
const ROW_ALIGN: usize = 8;

/// Below this many weight bytes the extra launch + join outweighs the overlap;
/// the step runs wholly on the CPU. At 9B shapes the covered projections are
/// 4–50 MB, so this only catches the tiny ones (the DeltaNet beta/alpha rows,
/// ~64 KB).
pub const MIN_SPLIT_BYTES: usize = 2 << 20;

/// First-trial share before either engine has been timed, and the floor/ceiling
/// of every scheduled share. 0.375 is the 2026-10-01 probe's balance point on
/// gfx1201 (27.3 GB/s link vs 46 GB/s contended CPU) — a starting guess the
/// controller corrects, not a hardware assumption the feature depends on.
const DEFAULT_GPU_SHARE: f64 = 0.375;
const SHARE_MIN: f64 = 0.05;
const SHARE_MAX: f64 = 0.50;

/// Reps per engine in the seeding probe: the median of three is enough to reject
/// a cold first touch, and each rep is one full pass over the step's weight
/// buffer.
const PROBE_REPS: usize = 3;

/// How much of the controller's proposal to apply per adjustment, and how often
/// to adjust.
const ALPHA: f64 = 0.3;
const APPLY_EVERY: u64 = 4;

/// EWMA weight for a rate sample, and the convergence latch: a shape stops being
/// perturbed once it has applied this many adjustments and the last four were all
/// small.
const EWMA_ALPHA: f64 = 0.25;
const FROZEN_APPLIED: u32 = 64;
const FREEZE_EPS: f64 = 0.005;

/// The latch's re-open band, on the proposal axis (`ALPHA ×` the balance-point
/// move). A latched shape keeps measuring, and unlatches when a freshly computed
/// proposal exceeds this in the same direction for [`REOPEN_CONFIRM`]
/// consecutive adjustments. Its floor is the freeze band: a shape latches with
/// its last four proposals under `FREEZE_EPS` (0.005), so a band at or below
/// that would re-open the latch on the very noise the freeze rejected — measured
/// on the model, `REOPEN_EPS = 0.002` re-opens a *clean* static plant nine times
/// in 1024 steps. `0.01` is twice the freeze band, which holds a margin for the
/// rate EWMAs to settle after latching. It is deliberately tighter than the
/// original `0.02`: on the balance-point axis that was a `0.02 / ALPHA ≈ 0.067`
/// dead zone, so a slow host-load drift could slide the optimum by up to 6.7
/// points before the latch noticed — a step change tripped it, a ramp did not.
/// On the model's ramp, `0.01` cuts the drift regret by ~28 % with no static- or
/// step-scenario cost. The band is not field-identifiable: a four-arm A/B (0.005
/// without the streak, 0.005/0.01/0.02 with it, 3 interleaved rounds at 8 and 16
/// of 32 spilled) left every arm inside its own run-to-run spread and flipped the
/// ordering between budgets, so the static fixture does not resolve the band —
/// the 0.005 penalty an earlier spread measured did not reproduce. So the value
/// rests on the freeze-band margin and the drift model, not on a field win.
const REOPEN_EPS: f64 = 0.01;

/// Consecutive out-of-band proposals, in the same direction, a latched shape must
/// produce before it unlatches. One is not enough: a host-load transient or a
/// single badly-sampled join can push one proposal past the band, and unlatching
/// on it hands the controller the noise it was latched to reject. A genuine move
/// drives the balance point one way for many adjustments, while noise flips sign,
/// so requiring a same-sign run separates the two without a bigger dead zone.
/// It is insurance, not a win: the model's genuine-move scenarios cost ~1–2 µs of
/// extra reopen latency per move and no measured throughput change, while it
/// removes the reopens a noisy static host produced without it.
const REOPEN_CONFIRM: i32 = 3;

/// `enabled` + `share` supplied by the caller, so the parity test and the
/// planner tests need no process snapshot.
#[derive(Clone, Copy, Debug)]
pub struct PassbackOptions {
    pub enabled: bool,
    pub share: PassbackShare,
}

impl PassbackOptions {
    /// The configuration for this process: `memory.offload_exec=passback` and
    /// `memory.offload_passback_share`.
    pub fn from_process() -> Self {
        PassbackOptions {
            enabled: enabled(),
            share: hipfire_config::memory::offload_passback_share(),
        }
    }
}

/// The seam's entry point: reads the process config snapshot.
pub fn run(gpu: &mut Gpu, ctx: &DispatchCtx, step: &Step) -> Result<bool, DispatchError> {
    run_with(gpu, ctx, step, &PassbackOptions::from_process())
}

/// `memory.offload_exec == passback` (via the cached CPU-exec predicate, which
/// covers both modes that execute spilled steps host-side).
pub fn enabled() -> bool {
    cpu_exec::passback_enabled()
}

/// Whether `dtype` can be *split*: a CPU decoder exists **and** its vector row dot
/// is available. With the scalar decoder the CPU rate (~10 GMAC/s) is well below
/// the GPU's PCIe read rate, so the best share is all-GPU — which plain `pcie`
/// does better. The load-time coverage line counts a splittable layer with this.
pub fn passback_capable_format(dtype: DType) -> bool {
    cpu_exec::cpu_quant_for(dtype).is_some_and(|q| hipfire_cpu::simd::row_dot_enabled(q, None))
}

/// Bytes per weight row for `dtype` at `k`, or `None` when there is no CPU
/// decoder for it.
///
/// Re-exported for callers that size a probe without depending on `hipfire-cpu`
/// (the CLI's `hipfire offload-bench` route reaches this through
/// `hipfire-runtime`): the CPU decoder is the single source of truth for the row
/// stride, which is also the invariant the feasibility gate checks the weight's
/// byte length against.
pub fn row_bytes_for(dtype: DType, k: usize) -> Option<usize> {
    cpu_exec::cpu_quant_for(dtype).map(|q| hipfire_cpu::gemv::row_bytes(q, k))
}

/// The share of a step's rows the GPU takes, chosen from `(m, row_bytes, share)`.
///
/// `None` when the step must run wholly on the CPU: a share that is not strictly
/// inside `(0, 1)`, fewer than two alignment quanta of rows, or a weight below
/// [`MIN_SPLIT_BYTES`]. `share = 0` therefore lands here, which is what makes
/// `memory.offload_passback_share = 0` byte-identical to
/// `memory.offload_exec=cpu`.
pub(crate) fn plan_rows(m: usize, row_bytes: usize, share: f64) -> Option<usize> {
    if !(share > 0.0 && share < 1.0) {
        return None;
    }
    if m < 2 * ROW_ALIGN {
        return None;
    }
    if row_bytes.checked_mul(m)? < MIN_SPLIT_BYTES {
        return None;
    }
    let g = align_down((m as f64 * share) as usize, ROW_ALIGN);
    Some(g.clamp(ROW_ALIGN, m - ROW_ALIGN))
}

fn align_down(v: usize, align: usize) -> usize {
    v / align * align
}

fn clamp_share(share: f64) -> f64 {
    share.clamp(SHARE_MIN, SHARE_MAX)
}

// ── The scheduler ──────────────────────────────────────

/// An exponentially weighted moving average of one engine's measured rate.
#[derive(Clone, Copy, Debug)]
struct Ewma {
    value: f64,
}

impl Ewma {
    fn new(sample: f64) -> Self {
        Ewma { value: sample }
    }

    /// The first sample *is* the state; later samples move it by [`EWMA_ALPHA`].
    fn update(&mut self, sample: f64) {
        self.value += EWMA_ALPHA * (sample - self.value);
    }
}

/// A shape's split state: the share in force, the two engines' contended rates,
/// and the convergence bookkeeping.
struct ShapeState {
    share: f64,
    r_cpu: Option<Ewma>,
    r_gpu: Option<Ewma>,
    samples: u64,
    /// Steps whose join exceeded the shape's no-wait floor plus the margin — the
    /// only ones `r_gpu` is sampled from. Printed with the shape so an operator can
    /// see whether the GPU rate the share rests on was ever actually observed.
    gpu_samples: u64,
    /// The smallest blocking-H2D time seen for this shape: the step's own
    /// no-wait floor, i.e. what the copy costs when the GPU arm finished first.
    /// Every larger join is measured *against* it, so the controller needs no
    /// host constant for the copy (the measured one is 30–50 µs for the ~30 KB
    /// this mode copies back — above any threshold guessed from the link rate).
    min_join_ns: Option<u64>,
    /// The last adjustment's proposal and whether that step's join had waited.
    last_target: Option<f64>,
    last_waited: bool,
    applied: u32,
    /// Times this shape has unlatched (frozen -> unfrozen). Nonzero proves the
    /// fixture actually exercised the re-open path, which a `frozen` flag alone
    /// cannot show, and it is the first number to read when a share has moved
    /// after latching.
    reopens: u32,
    /// The last four adjustment magnitudes, seeded above the freeze threshold so
    /// a shape cannot freeze on its first few no-op adjustments.
    deltas: [f64; 4],
    frozen: bool,
    /// Consecutive out-of-band proposals while latched, signed by direction
    /// (positive = the balance point moved up). Reset whenever a proposal lands
    /// back inside the band.
    reopen_streak: i32,
}

impl ShapeState {
    fn new(share: f64) -> Self {
        ShapeState {
            share,
            r_cpu: None,
            r_gpu: None,
            samples: 0,
            gpu_samples: 0,
            min_join_ns: None,
            last_target: None,
            last_waited: false,
            applied: 0,
            reopens: 0,
            deltas: [f64::MAX; 4],
            frozen: false,
            reopen_streak: 0,
        }
    }
}

/// Keyed by `(dtype, k)`: the CPU kernels differ per format and their throughput
/// per `k`.
struct SplitScheduler {
    shapes: BTreeMap<(DType, usize), ShapeState>,
    /// Shapes already seeded in this process, so the two-engine probe (a full
    /// pass on each engine) runs once per shape, not once per layer.
    probed: BTreeSet<(DType, usize)>,
    /// A calibration installed by [`seed`], which replaces the seeding probe for
    /// every shape that has no state yet.
    fallback: Option<SplitCalibration>,
}

impl Default for SplitScheduler {
    fn default() -> Self {
        SplitScheduler {
            shapes: BTreeMap::new(),
            probed: BTreeSet::new(),
            fallback: None,
        }
    }
}

static SCHEDULER: LazyLock<Mutex<SplitScheduler>> =
    LazyLock::new(|| Mutex::new(SplitScheduler::default()));

/// One shape's read-only scheduler state, for diagnostics and the bench.
#[derive(Clone, Copy, Debug)]
pub struct ShapeSnapshot {
    pub dtype: DType,
    pub k: usize,
    pub share: f64,
    pub cpu_bytes_per_s: Option<f64>,
    pub gpu_bytes_per_s: Option<f64>,
    pub samples: u64,
    /// Steps whose join exceeded the shape's no-wait floor plus the margin — the
    /// only ones the GPU rate above can come from.
    pub gpu_samples: u64,
    /// The smallest blocking-H2D time seen for this shape: the no-wait floor the
    /// controller measures every join against, and the reason a "wait" needs no
    /// host constant.
    pub min_join_ns: Option<u64>,
    pub last_waited: bool,
    pub last_target: Option<f64>,
    pub applied: u32,
    pub frozen: bool,
    /// Times this shape has unlatched; see `ShapeState::reopens`.
    pub reopens: u32,
}

/// Every shape the scheduler has state for. Read-only; no device, no locks held
/// on return.
pub fn scheduler_snapshot() -> Vec<ShapeSnapshot> {
    let Ok(scheduler) = SCHEDULER.lock() else {
        return Vec::new();
    };
    scheduler.shapes.iter().map(shape_of).collect()
}

/// Install a calibration measured elsewhere (by [`probe_synthetic`], the
/// `hipfire offload-bench` path, or a test) as this process's starting point: a
/// shape with no state yet starts from this share and rate, and no seeding probe
/// runs for it. The online controller still refines it.
pub fn seed(calibration: &SplitCalibration) {
    let Ok(mut scheduler) = SCHEDULER.lock() else {
        return;
    };
    scheduler.fallback = Some(*calibration);
}

/// The share in force for `dtype`'s `k`, if the scheduler has state for it.
///
/// The split trace reads this so the printed share is the one actually used,
/// pinned or scheduled.
pub(crate) fn shape_state(dtype: DType, k: usize) -> Option<ShapeSnapshot> {
    let scheduler = SCHEDULER.lock().ok()?;
    let state = scheduler.shapes.get(&(dtype, k))?;
    Some(shape_of((&(dtype, k), state)))
}

fn shape_of(((dtype, k), state): (&(DType, usize), &ShapeState)) -> ShapeSnapshot {
    ShapeSnapshot {
        dtype: *dtype,
        k: *k,
        share: state.share,
        cpu_bytes_per_s: state.r_cpu.map(|e| e.value),
        gpu_bytes_per_s: state.r_gpu.map(|e| e.value),
        samples: state.samples,
        gpu_samples: state.gpu_samples,
        min_join_ns: state.min_join_ns,
        last_waited: state.last_waited,
        last_target: state.last_target,
        applied: state.applied,
        frozen: state.frozen,
        reopens: state.reopens,
    }
}

/// The scheduler's fallback share, when one was installed.
fn fallback_share() -> Option<f64> {
    SCHEDULER.lock().ok()?.fallback.map(|c| c.share)
}

/// Whether this shape has already been probed (or seeded) in this process.
fn is_seeded(key: (DType, usize)) -> bool {
    SCHEDULER
        .lock()
        .map(|scheduler| scheduler.probed.contains(&key))
        .unwrap_or(true)
}

/// Insert a shape's state from a share, once. `probed` is marked so a later step
/// of the same shape neither probes nor re-seeds it.
fn seed_shape(key: (DType, usize), share: f64) {
    let Ok(mut scheduler) = SCHEDULER.lock() else {
        return;
    };
    scheduler.probed.insert(key);
    scheduler
        .shapes
        .entry(key)
        .or_insert_with(|| ShapeState::new(clamp_share(share)));
}

/// Record a pinned share so the trace and the snapshot can report it.
fn note_pinned(key: (DType, usize), share: f64) {
    let Ok(mut scheduler) = SCHEDULER.lock() else {
        return;
    };
    scheduler
        .shapes
        .entry(key)
        .or_insert_with(|| ShapeState::new(share));
}

/// The share to use for a shape whose state should already exist.
fn scheduled_share(key: (DType, usize)) -> f64 {
    SCHEDULER
        .lock()
        .ok()
        .and_then(|scheduler| {
            scheduler
                .shapes
                .get(&key)
                .map(|state| state.share)
                .or_else(|| scheduler.fallback.map(|c| c.share))
        })
        .unwrap_or(DEFAULT_GPU_SHARE)
}

/// Hysteresis on the join, in nanoseconds: an excess over the shape's no-wait
/// floor smaller than this is host-timer noise around the copy, not a wait.
///
/// Needed because the floor is the *minimum* join ever seen, so most joins exceed
/// it by a few microseconds; without the margin nearly every step would read as a
/// wait and the samples would be noise. Ten microseconds is well above that jitter
/// and far below the waits this mode acts on (hundreds of microseconds to
/// milliseconds). It is a timer-resolution constant, not a host performance one.
const JOIN_MARGIN_NS: u64 = 10_000;

/// The excess of `join_ns` over the shape's no-wait floor that can be a wait: the
/// hysteresis margin, or a quarter of the floor for a host whose copy is itself
/// slow (the copy's own jitter scales with it).
fn waited_extra_ns(join_ns: u64, floor_ns: Option<u64>) -> u64 {
    let Some(floor) = floor_ns else {
        return 0;
    };
    let margin = JOIN_MARGIN_NS.max(floor / 4);
    join_ns.saturating_sub(floor.saturating_add(margin))
}

/// The controller's proposal after one step's measurements.
///
/// With both rates present, move toward the balance point
/// `r_gpu / (r_gpu + r_cpu)`. With no GPU rate yet and a join that sat at the
/// shape's no-wait floor — the common regime, where the H2D of the CPU's rows
/// costs tens of microseconds against a CPU multiply of hundreds — the CPU is the
/// straggler, so raise the share. Otherwise the step carries no usable signal and
/// the share is unchanged.
fn next_share(cur: f64, r_cpu: Option<f64>, r_gpu: Option<f64>, gpu_waited: bool) -> f64 {
    match (r_cpu, r_gpu) {
        (Some(cpu), Some(gpu)) if cpu + gpu > 0.0 => {
            clamp_share(cur + ALPHA * (clamp_share(gpu / (gpu + cpu)) - cur))
        }
        _ if !gpu_waited => clamp_share(cur + 0.02),
        _ => cur,
    }
}

/// One split step's measurements. Rates are bytes per **second**, matching
/// [`SplitCalibration`]; the controller only uses them as a ratio.
///
/// `cpu_ns` is the CPU arm's own duration (the blocking D2H plus the multiply) and
/// `join_ns` the blocking H2D, which is `copy + max(0, gpu_ns - cpu_ns)`.
///
/// The join is read relative to the shape's own no-wait floor ([`min_join_ns`]),
/// never against an absolute threshold: only the excess over that floor can be a
/// wait for the GPU arm, and in that case `cpu_ns + excess` recovers the arm's
/// duration. Treating the raw join as a GPU duration in the other regime — the
/// common one, where the CPU multiply is hundreds of microseconds against a
/// tens-of-microseconds copy — reports a GPU rate an order of magnitude too low
/// and drags the share to its floor with both engines idle in turn.
fn observe(
    key: (DType, usize),
    rows_cpu: usize,
    rows_gpu: usize,
    row_bytes: usize,
    cpu_ns: u64,
    join_ns: u64,
) {
    let Ok(mut scheduler) = SCHEDULER.lock() else {
        return;
    };
    let Some(state) = scheduler.shapes.get_mut(&key) else {
        return;
    };
    if cpu_ns > 0 {
        let sample = bytes_per_s(rows_cpu as f64 * row_bytes as f64, cpu_ns);
        state.r_cpu = Some(match state.r_cpu {
            Some(mut ewma) => {
                ewma.update(sample);
                ewma
            }
            None => Ewma::new(sample),
        });
    }
    // The join is `copy + max(0, gpu_ns - cpu_ns)`, and only its excess over this
    // shape's own smallest join can be a wait for the GPU arm. Measuring against
    // that floor (rather than a fixed microsecond threshold, or the CPU's own
    // multiply) is what makes the sample honest on a host where the copy alone
    // costs more than the link rate suggests: at the floor the CPU is the
    // straggler and the share rises; above it the excess *is* the GPU arm's
    // overrun and its duration is recoverable as `cpu_ns + excess`.
    let baseline = state.min_join_ns;
    state.min_join_ns = Some(baseline.map_or(join_ns, |m| m.min(join_ns)));
    let waited_extra = waited_extra_ns(join_ns, baseline);
    let gpu_waited = waited_extra > 0;
    if gpu_waited && cpu_ns + waited_extra > 0 {
        let sample = bytes_per_s(
            rows_gpu as f64 * row_bytes as f64,
            cpu_ns + waited_extra,
        );
        state.r_gpu = Some(match state.r_gpu {
            Some(mut ewma) => {
                ewma.update(sample);
                ewma
            }
            None => Ewma::new(sample),
        });
        state.gpu_samples += 1;
    }
    state.samples += 1;
    state.last_waited = gpu_waited;
    if state.samples % APPLY_EVERY != 0 {
        return;
    }
    // Mirrors `next_share`'s two signal branches: only an adjustment that had
    // information counts toward freezing the shape.
    let had_signal = (state.r_cpu.is_some() && state.r_gpu.is_some()) || !gpu_waited;
    let next = next_share(
        state.share,
        state.r_cpu.map(|e| e.value),
        state.r_gpu.map(|e| e.value),
        gpu_waited,
    );
    // A latched shape keeps *measuring* — the rate EWMAs above never stop — so it
    // can notice that the balance point has moved. If the fresh proposal departs
    // from the latched share by more than the re-open band, and does so in the
    // same direction for [`REOPEN_CONFIRM`] consecutive adjustments, unlatch and
    // track it; otherwise hold. This is what lets the latch be periodic without
    // being permanent: it absorbs the converged value (and its noise) but follows
    // a genuine move (the balance point is a property of the host, so only host
    // state — competing load, thermal/DPM — moves it; a shape's rows and the
    // context length do not).
    // Requiring the run to hold one direction is what keeps a noisy static host
    // from unlatching the shape onto the noise the latch exists to reject.
    if state.frozen {
        state.reopen_streak = advance_reopen_streak(state.reopen_streak, next - state.share);
        if state.reopen_streak.unsigned_abs() < REOPEN_CONFIRM.unsigned_abs() {
            return;
        }
        state.frozen = false;
        state.reopens += 1;
        state.reopen_streak = 0;
        state.deltas = [f64::MAX; 4];
    }
    let delta = (next - state.share).abs();
    state.share = next;
    state.last_target = Some(next);
    if !had_signal {
        return;
    }
    state.applied += 1;
    state.deltas.rotate_left(1);
    state.deltas[3] = delta;
    if state.applied >= FROZEN_APPLIED && state.deltas.iter().all(|d| *d < FREEZE_EPS) {
        state.frozen = true;
    }
}

fn bytes_per_s(bytes: f64, ns: u64) -> f64 {
    bytes * 1e9 / ns.max(1) as f64
}

/// Fold one latched-step proposal into its re-open streak. A proposal inside
/// [`REOPEN_EPS`] clears the streak; one outside it extends a same-direction run
/// or restarts the streak in the new direction. `REOPEN_CONFIRM` consecutive
/// same-direction proposals are what actually unlatch the shape.
fn advance_reopen_streak(current: i32, off: f64) -> i32 {
    if off.abs() <= REOPEN_EPS {
        return 0;
    }
    let dir = if off > 0.0 { 1 } else { -1 };
    if current.signum() == dir {
        current + dir
    } else {
        dir
    }
}

// ── The seeding probe ──────────────────────────────────

/// What one two-engine measurement of a `(format, m, k)` shape found.
#[derive(Clone, Copy, Debug)]
pub struct SplitCalibration {
    /// The CPU arm's rate over the shape's weight bytes.
    pub cpu_bytes_per_s: f64,
    /// The GPU arm's rate over the same bytes.
    pub gpu_bytes_per_s: f64,
    /// `clamp(gpu / (gpu + cpu), SHARE_MIN, SHARE_MAX)`: the starting share.
    pub share: f64,
}

fn hip_err(e: hip_bridge::HipError) -> DispatchError {
    DispatchError::Hip(e.to_string())
}

/// The activation a probe should read, and whether the CPU arm would rotate it.
fn probe_activation<'a>(step: &'a Step<'a>, w: &WeightRef) -> Option<(&'a GpuTensor, bool)> {
    match step {
        Step::Gemv { input, .. } | Step::GemvResidual { input, .. } => match input {
            GemvInput::Raw(t) => Some((*t, dtype_rotation_plan(w.dtype) == RotationPlan::FwhtG256)),
            GemvInput::Prerotated(t) => Some((*t, false)),
        },
        _ => None,
    }
}

/// `GemvInput` holds references, so a probe step borrows the *real* input rather
/// than owning a copy: a `Raw` probe therefore rotates exactly as the real arm
/// will.
fn clone_input<'a>(input: &GemvInput<'a>) -> GemvInput<'a> {
    match input {
        GemvInput::Raw(t) => GemvInput::Raw(t),
        GemvInput::Prerotated(t) => GemvInput::Prerotated(t),
    }
}

fn median(mut samples: Vec<f64>) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples[samples.len() / 2]
}

/// Time both engines on `probe`'s weight buffer, `reps` times each, and return
/// the median rates. `Ok(None)` when the format has no launchable GPU arm (the
/// scheduler then keeps [`DEFAULT_GPU_SHARE`] and the online controller converges
/// from there).
///
/// Both engines are timed *alone*, sequentially, so both rates are overestimates
/// and the derived share is only a starting point.
fn measure_both(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    probe: &Step,
    w: &WeightRef,
    q: hipfire_cpu::quant::CpuQuant,
    row_bytes: usize,
    x_host: &[f32],
    reps: usize,
) -> Result<Option<SplitCalibration>, DispatchError> {
    let reps = reps.max(1);
    let Some(bytes) = gpu.host_bytes(w.buf) else {
        return Ok(None);
    };
    let mut y = vec![0.0f32; w.m];
    let mut cpu_ns = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        hipfire_cpu::gemv::gemv(q, bytes, w.m, w.k, x_host, &mut y);
        cpu_ns.push(t.elapsed().as_nanos() as f64);
    }
    let mut gpu_ns = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        if launch_op_rows(gpu, ctx, probe, Some(0..w.m)).is_err() {
            return Ok(None);
        }
        // A real blocking sync, NOT `sync_with_deadline`: that one polls with
        // `SYNC_POLL_INTERVAL` (2 ms) of sleep granularity, which on a sub-
        // millisecond kernel *is* the measurement — it reports ~4 GB/s for an
        // 8 MiB weight whose true rate is ~32 GB/s. Deadline-bearing paths still
        // use it; a timed benchmark must not.
        gpu.hip.device_synchronize().map_err(hip_err)?;
        gpu_ns.push(t.elapsed().as_nanos() as f64);
    }
    let size = (w.m * row_bytes) as f64;
    let cpu_bytes_per_s = bytes_per_s(size, median(cpu_ns).max(1.0) as u64);
    let gpu_bytes_per_s = bytes_per_s(size, median(gpu_ns).max(1.0) as u64);
    let share = clamp_share(gpu_bytes_per_s / (gpu_bytes_per_s + cpu_bytes_per_s));
    Ok(Some(SplitCalibration {
        cpu_bytes_per_s,
        gpu_bytes_per_s,
        share,
    }))
}

/// The seeding probe (4.3): time both engines on **this step's own weight
/// buffer**, once per shape per process.
///
/// The GPU arm is measured through a probe `Step` built over a scratch output,
/// never through the real step: for the residual form a probe write into the real
/// accumulator would be double-counted by the following real arm.
pub fn probe_weight(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    step: &Step,
    w: &WeightRef,
    reps: usize,
) -> Result<Option<SplitCalibration>, DispatchError> {
    let Some(q) = cpu_exec::cpu_quant_for(w.dtype) else {
        return Ok(None);
    };
    let Some(row_bytes) = weight_row_bytes(w) else {
        return Ok(None);
    };
    if row_bytes == 0 || w.m == 0 || w.k == 0 {
        return Ok(None);
    }
    let Some((activation, rotate_input)) = probe_activation(step, w) else {
        return Ok(None);
    };
    // The activation as the CPU arm will see it, once: the real arm downloads (and
    // rotates) it once per step, so re-deriving it per rep would measure a
    // transform the step does not repeat.
    let (x_host, _) = cpu_exec::prepare_activation(gpu, w, activation, rotate_input)?;
    let scratch = gpu.alloc_tensor(&[w.m], DType::F32).map_err(hip_err)?;
    let acc_scratch = match step {
        Step::GemvResidual { .. } => match gpu.alloc_tensor(&[w.m], DType::F32) {
            Ok(acc) => Some(acc),
            Err(e) => {
                let _ = gpu.free_tensor(scratch);
                return Err(hip_err(e));
            }
        },
        _ => None,
    };
    let input = match step {
        Step::Gemv { input, .. } => input,
        Step::GemvResidual { input, .. } => input,
        _ => unreachable!("probe_activation admitted only the two GEMV shapes"),
    };
    let input = clone_input(input);
    let probe = match acc_scratch.as_ref() {
        Some(acc) => Step::GemvResidual {
            w,
            input,
            residual: acc,
            out: acc,
        },
        None => Step::Gemv {
            w,
            input,
            out: &scratch,
        },
    };
    let result = measure_both(gpu, ctx, &probe, w, q, row_bytes, &x_host, reps);
    let _ = gpu.free_tensor(scratch);
    if let Some(acc) = acc_scratch {
        let _ = gpu.free_tensor(acc);
    }
    result
}

/// The same two-engine measurement over a **synthetic** host-mapped weight, for
/// `hipfire offload-bench`: no model and no step list needed.
///
/// Allocates, uses and frees its own buffers (a host-mapped weight of `m` rows —
/// system RAM the GPU reads over the link — an f32 activation of `k`, an f32
/// output of `m`). The weight's *contents* are irrelevant: both engines decode
/// the same bytes and the measurement is memory traffic.
pub fn probe_synthetic(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    dtype: DType,
    m: usize,
    k: usize,
    reps: usize,
) -> Result<SplitCalibration, DispatchError> {
    let q = cpu_exec::cpu_quant_for(dtype).ok_or_else(|| {
        DispatchError::Hip(format!(
            "passback probe: no CPU decoder for {dtype:?}; nothing to measure"
        ))
    })?;
    let row_bytes = hipfire_cpu::gemv::row_bytes(q, k);
    if m == 0 || k == 0 || row_bytes == 0 {
        return Err(DispatchError::Hip(
            "passback probe: m, k and the format's row stride must all be non-zero".into(),
        ));
    }
    let weight = gpu
        .upload_raw_host(&vec![0u8; m * row_bytes], &[m, k])
        .map_err(hip_err)?;
    let x_act = match gpu.upload_f32(&vec![0.0f32; k], &[k]) {
        Ok(t) => t,
        Err(e) => {
            let _ = gpu.free_tensor(weight);
            return Err(hip_err(e));
        }
    };
    let scratch = match gpu.alloc_tensor(&[m], DType::F32) {
        Ok(t) => t,
        Err(e) => {
            let _ = gpu.free_tensor(x_act);
            let _ = gpu.free_tensor(weight);
            return Err(hip_err(e));
        }
    };
    let result = {
        let w_ref = WeightRef {
            buf: &weight,
            dtype,
            m,
            k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        };
        // `Raw`, not `Prerotated`: the probe needs no producer to have warmed the
        // rotation scratch, and the extra rotate is `k` floats against MBs of
        // weight bytes. The real split of a `Raw` step does exactly this too.
        let probe = Step::Gemv {
            w: &w_ref,
            input: GemvInput::Raw(&x_act),
            out: &scratch,
        };
        let rotate = dtype_rotation_plan(dtype) == RotationPlan::FwhtG256;
        let (x_host, _) = cpu_exec::prepare_activation(gpu, &w_ref, &x_act, rotate)?;
        measure_both(gpu, ctx, &probe, &w_ref, q, row_bytes, &x_host, reps)?
    };
    let _ = gpu.free_tensor(scratch);
    let _ = gpu.free_tensor(x_act);
    let _ = gpu.free_tensor(weight);
    result.ok_or_else(|| {
        DispatchError::Hip(format!(
            "passback probe: {dtype:?} has no launchable GPU arm at m={m}, k={k}"
        ))
    })
}

// ── The seam ───────────────────────────────────────────

/// The weight of a row-splittable step, or `None` for any other kind.
fn step_weight<'a, 'b>(step: &'b Step<'a>) -> Option<&'a WeightRef<'a>> {
    match step {
        Step::Gemv { w, .. } | Step::GemvResidual { w, .. } => Some(*w),
        _ => None,
    }
}

/// The step's output rows that must be covered by the two arms, in order: the
/// output, plus the residual for the residual form.
fn output_extent(w: &WeightRef, step: &Step) -> bool {
    let out_ok = |t: &GpuTensor| t.dtype == DType::F32 && t.numel() >= w.m;
    match step {
        Step::Gemv { out, .. } => out_ok(out),
        Step::GemvResidual { residual, out, .. } => out_ok(residual) && out_ok(out),
        _ => false,
    }
}

/// Execute `step` as a pass-back split, or `Ok(false)` to leave it wholly to
/// `memory.offload_exec=cpu`.
///
/// Every refusal below is silent-with-a-fallback rather than an error: the mode
/// has a correct whole-CPU route for every step, so an unsplittable shape runs
/// exactly as `cpu` mode runs it (never as `pcie`).
pub fn run_with(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    step: &Step,
    opts: &PassbackOptions,
) -> Result<bool, DispatchError> {
    if !opts.enabled {
        return Ok(false);
    }
    // 1–2: the CPU must be able to take the step at all, and the step must be one
    // of the four row-splittable shapes.
    let Some(plan) = cpu_exec::plan_step(gpu, step) else {
        return Ok(false);
    };
    let Some(w) = step_weight(step) else {
        return Ok(false);
    };
    let residual = matches!(step, Step::GemvResidual { .. });
    if residual && KernelKey::for_gemv_residual(w.dtype).is_err() {
        // The GPU arm of a residual step is `dispatch_residual`, whose dtype set
        // is exactly `for_gemv_residual`'s. A `Raw` step outside it would take
        // `launch_op`'s multi-launch fallback (GEMV into a whole-tensor scratch,
        // then `residual += out`), and a `Prerotated` one would simply fail to
        // launch. Both must fall back to the whole-CPU step — this mode is never
        // allowed to be less robust than `memory.offload_exec=cpu`, which handles
        // every format the CPU decodes.
        return Ok(false);
    }
    // 3–6: host-mapped, unpadded, and consistent with the CPU decoder.
    if !gpu.host_located(w.buf) || w.row_stride != 0 || w.m == 0 {
        return Ok(false);
    }
    let Some(row_bytes) = weight_row_bytes(w) else {
        return Ok(false);
    };
    let Some(q) = cpu_exec::cpu_quant_for(w.dtype) else {
        return Ok(false);
    };
    // The correctness-critical invariant: it is what makes `&host_bytes[g *
    // row_bytes..]` the CPU arm's row `g`. A mismatch falls back rather than
    // reading the wrong rows.
    if row_bytes != hipfire_cpu::gemv::row_bytes(q, w.k) {
        return Ok(false);
    }
    if !hipfire_cpu::simd::row_dot_enabled(q, None) {
        return Ok(false);
    }
    // 7: both write targets must be F32 and cover every row.
    if !output_extent(w, step) {
        return Ok(false);
    }
    // 8: the share.
    let key = (w.dtype, w.k);
    let share = match opts.share {
        PassbackShare::Share(f) => {
            note_pinned(key, f);
            f
        }
        PassbackShare::Auto => {
            if !is_seeded(key) {
                match fallback_share() {
                    Some(f) => seed_shape(key, f),
                    None => {
                        // A failed or unavailable probe must not fail a step that
                        // plain `cpu` mode would have run: fall back to
                        // `DEFAULT_GPU_SHARE` and let the online controller correct
                        // it from the arm timings.
                        let calibration = probe_weight(gpu, ctx, step, w, PROBE_REPS)
                            .ok()
                            .flatten()
                            .map(|c| c.share);
                        seed_shape(key, calibration.unwrap_or(DEFAULT_GPU_SHARE));
                    }
                }
            }
            scheduled_share(key)
        }
    };
    // 9: is this shape splittable at this share?
    let Some(g) = plan_rows(w.m, row_bytes, share) else {
        return Ok(false);
    };

    // The ordering *is* the mechanism: prepare (blocking D2H) before the GPU arm,
    // GPU arm async, CPU multiply, then the blocking H2D of the CPU's rows as the
    // join.
    let step_start = Instant::now();
    let arm = cpu_exec::cpu_arm_prepare(gpu, &plan, g..w.m)?;
    launch_op_rows(gpu, ctx, step, Some(0..g))?;
    let (gemv_ns, join_ns) = cpu_exec::cpu_arm_finish(gpu, &arm)?;
    let cpu_ns = arm.d2h_ns + gemv_ns;
    observe(key, w.m - g, g, row_bytes, cpu_ns, join_ns);
    cpu_exec::finish_step(
        arm.q,
        w,
        arm.rotate_input,
        residual,
        HostExec::Split { gpu_rows: g },
        step_start,
        StepTiming {
            d2h_ns: arm.d2h_ns,
            gemv_ns,
            h2d_ns: join_ns,
        },
    );
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_rows_balances_and_refuses() {
        // The plan's pinned example: 0.375 of 12288 rows at 2720 B/row.
        assert_eq!(plan_rows(12288, 2720, 0.375), Some(4608));
        // A share outside (0, 1) never splits — 0 is the cpu byte-identity twin.
        assert_eq!(plan_rows(12288, 2720, 0.0), None);
        assert_eq!(plan_rows(12288, 2720, 1.0), None);
        assert_eq!(plan_rows(12288, 2720, -0.1), None);
        assert_eq!(plan_rows(12288, 2720, f64::NAN), None);
        // Too few rows to align on.
        assert_eq!(plan_rows(8, 1 << 20, 0.5), None);
        // A weight below the minimum split size.
        assert_eq!(plan_rows(1024, 1024, 0.5), None);
        assert!(1024 * 1024 < MIN_SPLIT_BYTES);
    }

    #[test]
    fn plan_rows_is_aligned_and_inside_its_bounds() {
        for m in [16usize, 17, 63, 1000, 12288, 40961] {
            for share in [0.01f64, 0.05, 0.1, 0.375, 0.5, 0.9, 0.99] {
                let Some(g) = plan_rows(m, 2720, share) else {
                    continue;
                };
                assert_eq!(g % ROW_ALIGN, 0, "m={m} share={share} g={g}");
                assert!(g >= ROW_ALIGN && g <= m - ROW_ALIGN, "m={m} share={share} g={g}");
            }
        }
    }

    #[test]
    fn next_share_moves_toward_the_balance_point() {
        // CPU at 52 GB/s, GPU at 27.3 GB/s -> balance 0.344; from 0.375 it moves
        // down by ALPHA times the gap.
        let moved = next_share(0.375, Some(52e9), Some(27.3e9), true);
        let target = 27.3 / (27.3 + 52.0);
        assert!(moved < 0.375 && moved > target, "moved={moved}");
        assert!((moved - (0.375 + ALPHA * (target - 0.375))).abs() < 1e-12);
        // A join that did not outlast the CPU's multiply means the CPU is the
        // straggler: the share rises by 0.02 ...
        assert!(
            (next_share(0.2, Some(52e9), None, false) - 0.22).abs() < 1e-12,
            "the push-up branch must fire on the common no-wait regime"
        );
        // ... and clamps at the ceiling.
        assert_eq!(next_share(SHARE_MAX, Some(52e9), None, false), SHARE_MAX);
        // A join that waited with no GPU rate yet is no information: unchanged.
        assert_eq!(next_share(0.3, Some(52e9), None, true), 0.3);
        // With both rates the balance point wins even when the join waited.
        let up = next_share(SHARE_MIN, Some(1.0), Some(1e12), true);
        assert!(up > SHARE_MIN && up <= SHARE_MAX, "up={up}");
        assert!((up - (SHARE_MIN + ALPHA * (SHARE_MAX - SHARE_MIN))).abs() < 1e-12);
    }

    #[test]
    fn observe_raises_the_share_at_the_join_floor_and_samples_above_it() {
        // A distinct key so the process-global scheduler cannot interfere with
        // the other tests in this binary.
        let key = (DType::Q8_0, 1234);
        seed_shape(key, 0.375);

        // A join that sits at the shape's own floor is a copy, not a wait: the
        // CPU is the straggler, no GPU rate is observable, and the share rises.
        for _ in 0..4 {
            observe(key, 3072, 1024, 2176, 500_000, 40_000);
        }
        let floor = shape_state(key.0, key.1).expect("seeded");
        assert_eq!(floor.gpu_samples, 0, "no wait, no GPU sample");
        assert!(
            floor.share > 0.375,
            "the share must rise while every join sits at the floor, got {}",
            floor.share
        );

        // A join *at* the floor, or within the hysteresis margin of it, is still
        // noise: no sample, and the share keeps rising.
        for _ in 0..4 {
            observe(key, 3072, 1024, 2176, 500_000, 45_000);
        }
        let noise = shape_state(key.0, key.1).expect("seeded");
        assert_eq!(
            noise.gpu_samples, 0,
            "an excess inside the margin is not a wait"
        );
        assert!(noise.share > floor.share, "the share keeps rising");

        // A join that *exceeds* the floor by more than the margin is a wait: the
        // GPU arm's duration is recoverable as `cpu_ns + excess`, and its (slow)
        // rate pulls the share toward the balance point it implies.
        for _ in 0..4 {
            observe(key, 3072, 1024, 2176, 500_000, 600_000);
        }
        let waited = shape_state(key.0, key.1).expect("seeded");
        assert!(waited.gpu_samples > 0, "a wait must yield a GPU sample");
        assert!(
            waited.share < noise.share,
            "a slow GPU arm must pull the share down ({} -> {})",
            noise.share,
            waited.share
        );
        // 1024 rows * 2176 B over (500 µs CPU arm + 550 µs excess over floor +
        // margin) = 2.12 GB/s: the sample is the excess, not the raw join.
        let r_gpu = waited.gpu_bytes_per_s.expect("sampled");
        let expected = 1024.0 * 2176.0 * 1e9 / 1_050_000.0;
        assert!(
            (r_gpu - expected).abs() / expected < 0.01,
            "sampled {r_gpu:e}, expected {expected:e}"
        );
        assert!(waited.cpu_bytes_per_s.is_some());
    }

    #[test]
    fn a_latched_shape_reopens_when_the_balance_point_moves() {
        // A distinct key so the process-global scheduler cannot interfere.
        let key = (DType::Q8_0, 4242);
        seed_shape(key, 0.246);
        // First join sets the shape's no-wait floor; later joins overrun it by
        // 10 µs, so `r_gpu` is sampled and the balance point is
        // 1024·2176/(500 µs + 10 µs) over that plus the CPU arm — 0.2463, i.e.
        // the seeded share, so the shape converges and latches.
        observe(key, 3072, 1024, 2176, 500_000, 40_000);
        for _ in 0..(APPLY_EVERY as usize * (FROZEN_APPLIED as usize + 8)) {
            observe(key, 3072, 1024, 2176, 500_000, 60_000);
        }
        let latched = shape_state(key.0, key.1).expect("seeded");
        assert!(
            latched.frozen,
            "a stable balance point must latch: share={} applied={}",
            latched.share, latched.applied
        );
        let before = latched.share;

        // The GPU arm is given four times the rows: the balance point jumps to
        // ~0.57, far outside the re-open band. The latch must re-open and track it
        // — this is the long-context / changed-host-load case the latch exists to
        // not miss.
        for _ in 0..(APPLY_EVERY as usize * 8) {
            observe(key, 3072, 4096, 2176, 500_000, 60_000);
        }
        let after = shape_state(key.0, key.1).expect("seeded");
        assert!(!after.frozen, "a moved balance point must re-open the latch");
        assert!(
            after.share > before + 0.05,
            "the share must follow the move: {before} -> {}",
            after.share
        );
    }

    // ── Latch dynamics harness ─────────────────────────────────────────
    //
    // A latch can only be exercised by a *moving* balance point, which no static
    // fixture produces, so `simulate` drives the real `observe` /
    // `scheduled_share` / `plan_rows` path over a scripted host-load schedule.
    // The plant is the one the module documents: `join = copy + max(0, gpu_ns -
    // cpu_ns)`, with deterministic jitter so joins cross the no-wait floor.
    struct Plant {
        m: usize,
        rb: usize,
        copy_ns: u64,
        jitter: u64,
        seed: u64,
    }
    impl Plant {
        fn noise(&mut self) -> i64 {
            self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = (self.seed >> 33) as i64;
            (x % (2 * self.jitter as i64 + 1)) - self.jitter as i64
        }
        /// `(cpu_ns, join_ns)`, both carrying deterministic jitter so joins
        /// cross the shape's no-wait floor the way they do on a real host.
        fn measure(&mut self, g: usize, r_cpu: f64, r_gpu: f64) -> (u64, u64) {
            let cpu = ((self.m - g) as f64 * self.rb as f64 * 1e9 / r_cpu) as i64 + self.noise();
            let gpu = (g as f64 * self.rb as f64 * 1e9 / r_gpu) as i64 + self.noise();
            let cpu = cpu.max(1) as u64;
            let gpu = gpu.max(1) as u64;
            (cpu, self.copy_ns + gpu.saturating_sub(cpu))
        }
    }

    /// The field 9B shape: `m=12288 k=4096` at 2176 B/row, a ~40 µs no-wait
    /// floor and ±60 µs host jitter (the checkpoint's `d2h=0.03 / join=0.04ms`).
    fn plant(seed: u64) -> Plant {
        Plant { m: 12288, rb: 2176, copy_ns: 40_000, jitter: 60_000, seed }
    }

    /// Drive a shape through the real `observe`/`scheduled_share`/`plan_rows`
    /// path over a scripted host-load schedule, returning `(reopens,
    /// final_share)`. `reopens` counts frozen -> unfrozen transitions after the
    /// shape first latched.
    fn simulate(
        key: (DType, usize),
        mut plant: Plant,
        start: f64,
        steps: usize,
        rates: &dyn Fn(usize) -> (f64, f64),
    ) -> (u32, f64) {
        seed_shape(key, start);
        let mut was_frozen = false;
        let mut reopens = 0u32;
        let mut share = start;
        for step in 0..steps {
            share = scheduled_share(key);
            let g = plan_rows(plant.m, plant.rb, share).expect("splittable");
            let (r_cpu, r_gpu) = rates(step);
            let (cpu_ns, join_ns) = plant.measure(g, r_cpu, r_gpu);
            let frozen = shape_state(key.0, key.1).expect("seeded").frozen;
            if was_frozen && !frozen {
                reopens += 1;
            }
            was_frozen = frozen;
            observe(key, plant.m - g, g, plant.rb, cpu_ns, join_ns);
        }
        (reopens, share)
    }

    #[test]
    fn a_latched_shape_holds_through_a_noisy_static_host() {
        // A static optimum must not re-open the latch: this is the property the
        // streak and the band exist to preserve. A 0.005 band measured worst in
        // one real-bench spread (9c722823c) and thrashes this model under heavy
        // jitter; the bench never established why, so the shipped band is 0.01
        // (twice the freeze band) and the streak rejects the sign-alternating
        // noise that would unlatch it.
        let (reopens, end) = simulate((DType::Q8_0, 9101), plant(7), 0.36, 1024, &|_| (35e9, 18e9));
        assert_eq!(reopens, 0, "a static optimum re-opened the latch");
        assert_eq!(
            shape_state(DType::Q8_0, 9101).expect("seeded").reopens,
            0,
            "the trace-visible re-open counter must agree"
        );
        let optimum = 18.0 / (35.0 + 18.0);
        assert!((end - optimum).abs() < 0.05, "share {end} vs optimum {optimum}");
    }

    #[test]
    fn a_latched_shape_follows_a_sustained_move() {
        // The changed-host-load case: the balance point steps and the latch must
        // reopen and track it.
        let moved = |s: usize| if s < 512 { (35e9, 18e9) } else { (22e9, 18e9) };
        let (reopens, end) = simulate((DType::Q8_0, 9102), plant(11), 0.36, 1024, &moved);
        assert!(reopens >= 1, "a step change did not re-open the latch");
        assert!(
            shape_state(DType::Q8_0, 9102).expect("seeded").reopens >= 1,
            "the trace-visible re-open counter must move with the latch"
        );
        let optimum = 18.0 / (22.0 + 18.0);
        assert!((end - optimum).abs() < 0.03, "share {end} vs optimum {optimum}");
    }

    #[test]
    fn a_latched_shape_follows_a_slow_drift() {
        // The reopen band is on the *proposal* axis (ALPHA x the balance-point
        // move), so a ramp that shifts the optimum a little per adjustment must
        // still trip the latch more than once instead of locking at its first
        // plateau — the regression the original fixed-0.02 band had.
        let ramp = |s: usize| {
            let t = (s as f64 / 4096.0).min(1.0);
            (35e9 - 13e9 * t, 18e9)
        };
        let (reopens, end) = simulate((DType::Q8_0, 9103), plant(31), 0.36, 4096, &ramp);
        assert!(reopens >= 2, "a slow drift left the latch shut: reopens={reopens}");
        let optimum = 18.0 / (18.0 + 22.0);
        assert!(end > optimum - 0.05, "the share did not track the drift: {end} vs {optimum}");
    }

    #[test]
    fn a_reopen_needs_a_same_sign_run() {
        assert_eq!(advance_reopen_streak(0, 0.001), 0, "in band clears the run");
        assert_eq!(advance_reopen_streak(0, 0.03), 1);
        assert_eq!(advance_reopen_streak(1, 0.03), 2, "a same-sign run extends");
        assert_eq!(advance_reopen_streak(1, -0.03), -1, "noise flips sign, the run restarts");
        assert_eq!(advance_reopen_streak(-2, -0.03), -3);
        assert_eq!(advance_reopen_streak(-2, 0.001), 0);
    }

    #[test]
    fn ewma_seeds_on_the_first_sample() {
        let mut e = Ewma::new(10.0);
        assert_eq!(e.value, 10.0);
        e.update(20.0);
        assert_eq!(e.value, 10.0 + EWMA_ALPHA * 10.0);
        e.update(0.0);
        assert!(e.value > 0.0 && e.value < 20.0);
    }

    #[test]
    fn share_always_lands_inside_the_ceiling_and_floor() {
        for start in [0.0f64, 0.05, 0.375, 0.5, 0.9] {
            for (cpu, gpu) in [(1e9, 1e12), (1e12, 1e9), (1.0, 1.0)] {
                for waited in [true, false] {
                    let s = next_share(start, Some(cpu), Some(gpu), waited);
                    assert!((SHARE_MIN..=SHARE_MAX).contains(&s), "s={s}");
                }
            }
        }
    }
}
