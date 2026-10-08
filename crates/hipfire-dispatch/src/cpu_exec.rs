// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! CPU execution of host-mapped weight steps (`memory.offload_exec=cpu`).
//!
//! Partial offload ([`hipfire_config::memory::gpu_layer_budget`]) places whole
//! layers in host-mapped system RAM, but by default the GPU kernels still
//! dereference those bytes — once per token, across PCIe. With
//! `memory.offload_exec=cpu` the steps whose weight tensor is host-mapped
//! execute on the CPU instead, so a spilled layer's per-token cost is bound by
//! the link (27.1 GB/s measured on the host in
//! `docs/perf-checkpoints/2026-09-26-llamacpp-offload-scaling-baseline.md`)
//! rather than by device DRAM — which is what llama.cpp's CPU backend buys with
//! its `-ngl` spill.
//!
//! Scope is deliberately the *weight-reading* ops only — [`Step::Gemv`] and
//! [`Step::GemvResidual`], plus the one op family that fused them
//! (`hipfire_runtime::llama::weight_gemv_swiglu_residual`, which splits itself
//! and calls [`run_host_mapped_gemv_residual`]). Attention, softmax, rmsnorm,
//! RoPE, qk-norm, the KV write, the flash-attention families and the DeltaNet
//! recurrence stay on the GPU, and so does the KV cache's residency: this
//! changes *who multiplies*, never *what is spilled*.
//!
//! The numerical contract is llama.cpp-level, not bit-identity — the reference
//! behaviour for this feature is llama.cpp's own partial offload, which is not
//! numerically transparent either (same record, "Correctness gate"). Acceptance
//! is coherence plus task-correct output plus a *measured* divergence; the
//! per-format device-vs-CPU distances are recorded in
//! `crates/hipfire-arch-qwen35/tests/gpu_gemv_parity.rs`.
//!
//! Two properties of the launcher this path has to reproduce exactly, because
//! getting either wrong produces plausible-looking wrong numbers rather than an
//! error:
//!
//! * **Rotation.** `Raw` inputs are FWHT-rotated exactly when the dtype's
//!   `dtype_rotation_plan` says so; the weights are stored post-rotation.
//! * **AWQ.** A weight carrying an `awq_scale` sidecar was pre-scaled by `s` at
//!   quantize time, and the *rotate* step (not the GEMV) divides the activation
//!   by it. That applies to a `Raw` input we rotate ourselves, and equally to
//!   the fused down-projection this path splits. It must **not** be applied
//!   again to a `Prerotated` input, whose producer already did it.
//!
//! Known gap, stated rather than papered over: the launcher *verifies* a
//! `Prerotated` buffer's rotation tag (`plan` + `awq`) and errors on a mismatch,
//! while a bare `GpuTensor` carries no tag, so a CPU step cannot make that check
//! for a pre-rotated input. A mismatch is a producer bug that fails loudly on
//! the GPU arms of the same model, which is why the check was not duplicated
//! into the step representation.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use rdna_compute::{DType, Gpu, GpuTensor};

use hipfire_cpu::epilogue::residual_add;
use hipfire_cpu::gemv::gemv;
use hipfire_cpu::quant::{divide_by_awq_scale, rotate_x, CpuQuant};

use crate::families::gemv::WeightRef;
use crate::pipeline::steps::{GemvInput, Step};
use crate::types::{dtype_rotation_plan, DispatchError, RotationPlan};

/// `memory.offload_exec` resolved once from the process snapshot, mirroring the
/// `forward_lowered_enabled` precedent: read once, never per step.
///
/// One cached snapshot backs both predicates below, so the mode is resolved
/// exactly once for the process rather than once per boolean derived from it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum HostExecMode {
    Pcie,
    Cpu,
    Passback,
}

fn host_exec_mode() -> HostExecMode {
    static MODE: LazyLock<HostExecMode> = LazyLock::new(|| {
        match hipfire_config::memory::offload_exec() {
            hipfire_config::memory::OffloadExec::Cpu => HostExecMode::Cpu,
            hipfire_config::memory::OffloadExec::Passback => HostExecMode::Passback,
            hipfire_config::memory::OffloadExec::Pcie => HostExecMode::Pcie,
        }
    });
    *MODE
}

/// `memory.offload_exec` executes some or all of a spilled step host-side
/// (`cpu` or `passback`), so the CPU multiplies its weights
/// ([`passback_enabled`] distinguishes the split form). Every host-exec
/// mechanism in this module keys on this predicate, so both modes are covered
/// without renaming a call site.
pub fn cpu_exec_enabled() -> bool {
    host_exec_mode() != HostExecMode::Pcie
}

/// `memory.offload_exec == passback`: the CPU runs a spilled step and hands part
/// of its output rows back to the GPU ([`crate::offload_split`]).
pub fn passback_enabled() -> bool {
    host_exec_mode() == HostExecMode::Passback
}

/// (steps executed on the CPU, host-mapped steps that stayed on the GPU).
///
/// The second number is the honest coverage failure signal: it counts steps whose
/// weight tensor *was* host-mapped while CPU execution was enabled but whose
/// shape never reached the CPU — either an unsupported quant format or a fused
/// launch the seam did not split. Zero means every host-mapped step this process
/// executed went to the CPU.
static CPU_STEPS: AtomicUsize = AtomicUsize::new(0);
static HOST_MAPPED_GPU_STEPS: AtomicUsize = AtomicUsize::new(0);

/// Cumulative wall time spent *inside* CPU-executed steps, entry to return: the
/// D2H drain, the multiplication, the H2D. Paired with [`CPU_STEPS`] for the
/// GPU-idle accounting in [`report_idle`].
///
/// A pass-back *split* step is deliberately NOT charged here: its wall contains
/// real GPU work, so folding it in would inflate the "GPU-idle lower bound" that
/// [`report_idle`] documents. Split steps have their own counters below.
static CPU_STEP_WALL_NS: AtomicU64 = AtomicU64::new(0);

/// Pass-back split steps: how many, how many of their output rows went to the
/// GPU, and their total wall. Counted separately from [`CPU_STEPS`] for the
/// reason above; [`split_counters`] is the reader.
static SPLIT_STEPS: AtomicUsize = AtomicUsize::new(0);
static SPLIT_GPU_ROWS: AtomicU64 = AtomicU64::new(0);
static SPLIT_WALL_NS: AtomicU64 = AtomicU64::new(0);

pub fn cpu_exec_counters() -> (usize, usize) {
    (
        CPU_STEPS.load(Ordering::Relaxed),
        HOST_MAPPED_GPU_STEPS.load(Ordering::Relaxed),
    )
}

/// `(split steps, rows handed to the GPU)`.
pub fn split_counters() -> (usize, u64) {
    (
        SPLIT_STEPS.load(Ordering::Relaxed),
        SPLIT_GPU_ROWS.load(Ordering::Relaxed),
    )
}

/// Which engine(s) executed a step, for [`finish_step`]'s accounting.
#[derive(Clone, Copy)]
pub(crate) enum HostExec {
    /// The whole step ran on the CPU (`memory.offload_exec=cpu`).
    Cpu,
    /// A pass-back split: the GPU ran the first `gpu_rows` output rows, the CPU
    /// the rest.
    Split { gpu_rows: usize },
}

/// Wall-clock partition of a decode run into CPU-step time and everything else.
///
/// A CPU-executed step is a host sync point: its D2H drains the stream, the
/// multiplication and the H2D run on the host, so the compute units execute
/// nothing for its whole duration — *provided* nothing else is using the device.
/// Over a decode-only span (prefill is GPU-side and never enters the seam) the
/// fraction of the wall spent inside those steps is therefore a **lower bound**
/// on the GPU's idle fraction, and the headroom a "hand some of the spilled
/// work back to an idle GPU" scheme would be spending. `HIPFIRE_CPU_EXEC_TRACE=1`
/// prints it per window; it is not a correctness signal and nothing depends on
/// it.
#[derive(Clone, Copy)]
struct IdleWindow {
    anchor: Instant,
    cpu_ns: u64,
    steps: usize,
    split_steps: usize,
    split_rows: u64,
    split_wall_ns: u64,
}

/// `None` until the first report; a window is the span between two consecutive
/// reports. Windows rather than a cumulative ratio because `hipfire bench
/// --runs N` decodes several times inside one process and the gaps between runs
/// are neither GPU-idle nor CPU-step time — a cumulative ratio would dilute
/// every window that spanned one.
static IDLE_WINDOW: Mutex<Option<IdleWindow>> = Mutex::new(None);

/// Print the CPU-step share of the wall since the last report. Called from
/// [`trace_step`] at the same doubling schedule as the per-shape lines, so the
/// last windows of a run cover the bulk of it and the cold first steps sit in
/// the first, discarded, window.
fn report_idle(steps: usize) {
    let total_cpu_ns = CPU_STEP_WALL_NS.load(Ordering::Relaxed);
    let (total_split_steps, total_split_rows) = split_counters();
    let total_split_wall = SPLIT_WALL_NS.load(Ordering::Relaxed);
    let Ok(mut window) = IDLE_WINDOW.lock() else {
        return;
    };
    let now = Instant::now();
    let Some(previous) = window.as_mut() else {
        *window = Some(IdleWindow {
            anchor: now,
            cpu_ns: total_cpu_ns,
            steps,
            split_steps: total_split_steps,
            split_rows: total_split_rows,
            split_wall_ns: total_split_wall,
        });
        return;
    };
    let span = now.duration_since(previous.anchor);
    let cpu_ns = total_cpu_ns.saturating_sub(previous.cpu_ns);
    let window_steps = steps - previous.steps;
    let split_steps = total_split_steps.saturating_sub(previous.split_steps);
    let split_rows = total_split_rows.saturating_sub(previous.split_rows);
    let split_wall_ns = total_split_wall.saturating_sub(previous.split_wall_ns);
    previous.anchor = now;
    previous.cpu_ns = total_cpu_ns;
    previous.steps = steps;
    previous.split_steps = total_split_steps;
    previous.split_rows = total_split_rows;
    previous.split_wall_ns = total_split_wall;
    let span_ns = span.as_nanos() as u64;
    if span_ns == 0 {
        return;
    }
    let per_step_us = cpu_ns as f64 / window_steps.max(1) as f64 / 1e3;
    // A split step's wall contains GPU work, so it is excluded from the CPU-idle
    // numerator (see CPU_STEP_WALL_NS) — but the window must not hide it either,
    // hence the suffix.
    let split_note = if split_steps == 0 {
        String::new()
    } else {
        format!(
            "; {split_steps} split steps ({split_rows} rows on GPU, {:.0}ms split wall) not \
             counted as CPU-idle",
            split_wall_ns as f64 / 1e6,
        )
    };
    eprintln!(
        "cpu exec: idle {:.1}% — window ending at step {steps} covers {window_steps} steps: \
         {:.0}ms wall, {:.0}ms on CPU ({per_step_us:.0}us CPU/step) \
         — GPU-idle lower bound, see report_idle{split_note}",
        100.0 * cpu_ns as f64 / span_ns as f64,
        span_ns as f64 / 1e6,
        cpu_ns as f64 / 1e6,
    );
}

/// Per-step wall-time split, in nanoseconds: device→host, the GEMV itself,
/// host→device — keyed by step *shape*, not summed across the process.
///
/// The structural risk of this feature is that a CPU step is a host sync point,
/// so it serializes against the surrounding GPU work; these three numbers say
/// whether a slow `cpu` arm is paying for copies or for arithmetic. Keying them
/// by shape is the whole point: a process-wide sum divided by the total step
/// count is a *cumulative mean* that every shape reports identically (and that
/// early cold steps inflate), which reads as per-shape attribution and is not.
#[derive(Clone, Copy, Default)]
struct StepStats {
    calls: u64,
    d2h_ns: u64,
    gemv_ns: u64,
    /// Whole-CPU steps: the blocking H2D. Split steps: the join (the blocking H2D,
    /// which in a split also carries the GPU arm's wait).
    h2d_ns: u64,
    /// The shape's kind, latched by its first call: a shape either always splits
    /// or never does, because the decision is a property of the shape and the
    /// scheduler's share is clamped away from 0 and 1.
    split: bool,
    /// Split-only: rows the CPU arm multiplied, and rows the GPU arm took, summed
    /// over the shape's calls so the throughput figures below can be derived.
    split_cpu_rows: u64,
    split_gpu_rows: u64,
}

/// `(quant, m, k, rotated, residual, awq)` → running per-shape totals.
static SHAPES: LazyLock<Mutex<BTreeMap<(u8, usize, usize, bool, bool, bool), StepStats>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// `DType` → the decoder for it, for exactly the formats `hipfire_cpu` can
/// decode. `None` means the step stays on the GPU (over PCIe): not a correctness
/// problem, only a smaller bandwidth win, and the load-time coverage line names
/// the format.
///
/// Invariant, pinned by `cpu_quant_rotation_agrees_with_plan`: for every dtype
/// here, [`CpuQuant::is_fwht_g256`] agrees with [`dtype_rotation_plan`], so the
/// seam and the launcher cannot disagree about whether a `Raw` activation
/// arrives rotated.
pub fn cpu_quant_for(dtype: DType) -> Option<CpuQuant> {
    match dtype {
        DType::MQ4G256 => Some(CpuQuant::Mq4G256),
        DType::MQ4G256V2 => Some(CpuQuant::Mq4G256V2),
        DType::MQ6G256 => Some(CpuQuant::Mq6G256),
        DType::MQ6G256V2 => Some(CpuQuant::Mq6G256V2),
        DType::MQ5G256 => Some(CpuQuant::Mq5G256),
        DType::MQ5G256V2 => Some(CpuQuant::Mq5G256V2),
        DType::MQ4CG256 => Some(CpuQuant::Mq4CG256),
        DType::MQ3G256 => Some(CpuQuant::Mq3G256),
        DType::MQ3G256V2 => Some(CpuQuant::Mq3G256V2),
        DType::MQ3G256Lloyd => Some(CpuQuant::Mq3G256Lloyd),
        DType::MQ2G256 => Some(CpuQuant::Mq2G256),
        DType::MQ2G256V2 => Some(CpuQuant::Mq2G256V2),
        DType::MQ2G256Lloyd => Some(CpuQuant::Mq2G256Lloyd),
        DType::MQ2G256LloydU => Some(CpuQuant::Mq2G256LloydU),
        DType::MQ4G256Lloyd => Some(CpuQuant::Mq4G256Lloyd),
        DType::HFQ6G256 => Some(CpuQuant::Hfq6G256),
        DType::HFQ4G256 => Some(CpuQuant::Hfq4G256),
        DType::HFQ4G128 => Some(CpuQuant::Hfq4G128),
        DType::HFQ3G256 => Some(CpuQuant::Hfq3G256),
        DType::HFQ3G128 => Some(CpuQuant::Hfq3G128),
        DType::HFQ2G256 => Some(CpuQuant::Hfq2G256),
        DType::HFQ2G128 => Some(CpuQuant::Hfq2G128),
        DType::TQ2G128 => Some(CpuQuant::Tq2G128),
        DType::BQ1G128 => Some(CpuQuant::Bq1G128),
        DType::Q8_0 => Some(CpuQuant::Q8F16),
        DType::F16 => Some(CpuQuant::F16),
        DType::F32 => Some(CpuQuant::F32),
        DType::BF16 => Some(CpuQuant::Bf16),
        _ => None,
    }
}

/// Whether a model with this resolved offload split has CPU-executed steps at
/// all: CPU execution selected *and* a non-empty spilled prefix
/// (`Qwen35Config::i_gpu_start`). With no spill every weight stays
/// device-resident, [`Gpu::host_located`] is false everywhere, and not a single
/// step can move to the CPU — `memory.offload_exec=cpu` must not change a number
/// there.
///
/// Conservative by construction: a spill whose every weight is an unsupported
/// format also reports `true`, which costs a hipGraph but never correctness.
pub fn cpu_offload_active(i_gpu_start: usize) -> bool {
    cpu_exec_enabled() && i_gpu_start > 0
}

/// Whether `memory.offload_exec=cpu` conflicts with a retained-replay backend.
///
/// Pure truth table (the CPU-exec decision, the resolved split, and whether a
/// replay controller is in play) so it is testable without a process snapshot.
pub fn cpu_exec_redline_conflict(cpu_exec: bool, i_gpu_start: usize, replay_enabled: bool) -> bool {
    cpu_exec && i_gpu_start > 0 && replay_enabled
}

/// Refuse `memory.offload_exec=cpu` together with a retained-replay (Redline)
/// backend, naming both keys.
///
/// Redline does not go through `execute_steps`' launch funnel: its tape records
/// GPU launches and replays them, so a CPU-executed step would run while the
/// tape is built and then be *absent* from every replay — the replayed route
/// would keep computing from the activations that step should have refreshed.
/// Nothing in the tape can express that, so the load fails instead of running a
/// route whose output is silently stale.
///
/// `replay_enabled` is the controller's own predicate
/// (`ReplayController::is_enabled`), not a config string: whether Redline is in
/// play depends on the runtime route decision and certification state, so the
/// caller with the `Gpu` supplies the fact. Shadow controllers are refused too
/// (conservatively — shadow does not change the launch route, so the refusal is
/// stricter than strictly necessary there).
pub fn reject_cpu_exec_under_redline(
    i_gpu_start: usize,
    replay_enabled: bool,
) -> Result<(), String> {
    if cpu_exec_redline_conflict(cpu_exec_enabled(), i_gpu_start, replay_enabled) {
        let mode = hipfire_config::memory::offload_exec();
        return Err(format!(
            "memory.offload_exec={mode} conflicts with the retained-replay (Redline) backend: the \
             replay tape records GPU launches and does not execute the CPU-executed steps, so a \
             replayed route would compute from stale activations. Set memory.offload_exec=pcie \
             (HIPFIRE_OFFLOAD_EXEC=pcie) or replay.backend=hip"
        ));
    }
    Ok(())
}

/// Log the capture-disable decision once per process. A CPU-executed step is a
/// host sync point (a D2H and an H2D around the multiplication), so a graph that
/// contained one could neither be recorded nor replayed correctly.
pub fn log_capture_disabled_once() {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        if passback_enabled() {
            eprintln!("passback: hipGraph capture disabled (host-executed steps present)");
        } else {
            eprintln!("cpu exec: hipGraph capture disabled (CPU-executed steps present)");
        }
    });
}

/// `out[0..m] = W · x[0..k]` on the CPU, over a host-mapped weight.
///
/// `rotate_x` must be `true` exactly when the launcher would self-rotate a `Raw`
/// input for this dtype (i.e. `dtype_rotation_plan(w.dtype) == FwhtG256`), which
/// [`RotationPlan`] is the single source of truth for; the dispatch seam derives
/// it so no caller has to.
pub fn run_host_mapped_gemv(
    gpu: &Gpu,
    w: &WeightRef,
    x: &GpuTensor,
    rotate_input: bool,
    out: &GpuTensor,
) -> Result<(), DispatchError> {
    let plan = CpuStep {
        w,
        x,
        rotate_input,
        out,
        residual: None,
    };
    let step_start = Instant::now();
    let arm = cpu_arm_prepare(gpu, &plan, 0..w.m)?;
    let (gemv_ns, h2d_ns) = cpu_arm_finish(gpu, &arm)?;
    finish_step(
        arm.q,
        w,
        rotate_input,
        false,
        HostExec::Cpu,
        step_start,
        StepTiming {
            d2h_ns: arm.d2h_ns,
            gemv_ns,
            h2d_ns,
        },
    );
    Ok(())
}

/// Read the activation for a CPU step, applying the same pre-rotation
/// transforms the launcher's rotate step would, and return it with the elapsed
/// nanoseconds — the trace's `d2h` figure, which therefore covers the AWQ divide
/// and the FWHT whenever they apply, not just the copy.
///
/// * AWQ (`w.awq_scale`): the quantizer pre-scaled the weights by `s` and the
///   rotate kernel divides the activation by it (`(W·s)·(x/s) = W·x`). This
///   happens *inside the rotation*, so it applies only when we rotate — a
///   `Prerotated` input already had it applied by whoever produced it (the
///   launcher *checks* that via the rotation tag; a CPU step cannot, which is
///   the one semantic gap this path has — see the module docs).
/// * FWHT (`rotate_input`): unless the input arrived pre-rotated.
///
/// `pub(crate)` because the pass-back's seeding probe reuses the exact activation
/// transform the real step will apply, rather than re-deriving it.
pub(crate) fn prepare_activation(
    gpu: &Gpu,
    w: &WeightRef,
    x: &GpuTensor,
    rotate_input: bool,
) -> Result<(Vec<f32>, u64), DispatchError> {
    let t0 = Instant::now();
    let mut host = download_f32(gpu, x, w.k)?;
    if rotate_input {
        if let Some(scale) = w.awq_scale {
            let scale = download_f32(gpu, scale, w.k)?;
            divide_by_awq_scale(&mut host, &scale);
        }
        rotate_x(&mut host);
    }
    Ok((host, t0.elapsed().as_nanos() as u64))
}

/// The CPU arm of a host-executed step: every blocking device read already done
/// ([`cpu_arm_prepare`]), the multiply and the blocking write-back still pending
/// ([`cpu_arm_finish`]).
///
/// The two phases exist so a pass-back split can interleave a GPU launch
/// *between* them: a blocking D2H issued after that launch would drain it, so the
/// read must complete first and the write is the join (stream-ordered after the
/// GPU arm on the same stream). `memory.offload_exec=cpu` runs the two back to
/// back, which is the same sequence of device operations as the single-shot form
/// it replaced.
pub(crate) struct CpuArm<'a> {
    /// The weight's CPU decoder.
    pub(crate) q: CpuQuant,
    /// Whether the activation was rotated (and AWQ-divided) — i.e. whether the
    /// launcher would have self-rotated this input. Mirrored here so the
    /// pass-back can hand it to [`finish_step`] without keeping the plan alive.
    pub(crate) rotate_input: bool,
    /// Elapsed time of the blocking reads: the activation, plus the accumulator
    /// when the step is a residual form. The trace's and the scheduler's `d2h`.
    pub(crate) d2h_ns: u64,
    w: &'a WeightRef<'a>,
    /// The row range this arm multiplies.
    rows: Range<usize>,
    /// Where the result is written: the step's `out`, or its `residual` for the
    /// residual form (a residual step never writes `out` — see the GemvResidual
    /// arms in `steps.rs`).
    target: &'a GpuTensor,
    /// The activation for `rows`, already rotated and AWQ-divided exactly as the
    /// launcher's rotate step would have done.
    x_host: Vec<f32>,
    /// The accumulator read back for the residual form, *before* the multiply, so
    /// a concurrently executing GPU arm cannot race this read.
    acc_host: Option<Vec<f32>>,
}

/// Phase one of the CPU arm: perform every blocking device read for `rows` and
/// return the arm. MUST complete before any GPU arm of the same step is
/// enqueued, because the reads drain the stream.
pub(crate) fn cpu_arm_prepare<'a>(
    gpu: &Gpu,
    cpu: &CpuStep<'a>,
    rows: Range<usize>,
) -> Result<CpuArm<'a>, DispatchError> {
    let q = host_mapped_quant(gpu, cpu.w)?;
    let t0 = Instant::now();
    let (x_host, _) = prepare_activation(gpu, cpu.w, cpu.x, cpu.rotate_input)?;
    let acc_host = match cpu.residual {
        Some(acc) => Some(download_f32(
            gpu,
            &acc.sub_offset(rows.start, rows.len()),
            rows.len(),
        )?),
        None => None,
    };
    Ok(CpuArm {
        q,
        rotate_input: cpu.rotate_input,
        d2h_ns: t0.elapsed().as_nanos() as u64,
        w: cpu.w,
        rows,
        target: cpu.residual.unwrap_or(cpu.out),
        x_host,
        acc_host,
    })
}

/// Phase two of the CPU arm: the multiply over `rows`, the residual add when
/// there is one, and the blocking write-back. Returns `(gemv_ns, h2d_ns)`.
///
/// The write-back is the join of a pass-back split: same stream as the GPU arm, so
/// it waits for it, and the two arms' rows are disjoint, so nothing else is
/// needed. For a split step `h2d_ns` therefore also carries the GPU arm's wait,
/// which is the trace's `join` figure.
pub(crate) fn cpu_arm_finish(gpu: &Gpu, arm: &CpuArm<'_>) -> Result<(u64, u64), DispatchError> {
    let rows = arm.rows.clone();
    let bytes = gpu
        .host_bytes(arm.w.buf)
        .ok_or_else(|| cpu_err("weight tensor is host-mapped but has no host pointer"))?;
    let row_bytes = crate::families::gemv::weight_row_bytes(arm.w)
        .ok_or_else(|| cpu_err("weight byte length is not a whole number of rows"))?;
    let w_rows = &bytes[rows.start * row_bytes..];
    let mut y = vec![0.0f32; rows.len()];
    let t1 = Instant::now();
    gemv(arm.q, w_rows, rows.len(), arm.w.k, &arm.x_host, &mut y);
    let gemv_ns = t1.elapsed().as_nanos() as u64;
    let t2 = Instant::now();
    let target = arm.target.sub_offset(rows.start, rows.len());
    match arm.acc_host.as_ref() {
        Some(acc_host) => {
            let mut acc = acc_host.clone();
            residual_add(&mut acc, &y);
            upload_f32(gpu, &target, &acc)?;
        }
        None => upload_f32(gpu, &target, &y)?,
    }
    let h2d_ns = t2.elapsed().as_nanos() as u64;
    Ok((gemv_ns, h2d_ns))
}

/// `acc += W · x` on the CPU, over a host-mapped weight — the residual form the
/// fused down-projection kernels use (`WithSwiGLUResidual`, `WithResidual`).
///
/// The accumulation is in place in `acc`, matching both fused arms: a residual
/// step never writes its `out` scratch, so writing only `out` when the two do not
/// alias would leave every downstream activation stale.
pub fn run_host_mapped_gemv_residual(
    gpu: &Gpu,
    w: &WeightRef,
    x: &GpuTensor,
    acc: &GpuTensor,
) -> Result<(), DispatchError> {
    // The fused down-projection this splits always feeds the CPU path an
    // unrotated activation (the SiLU output), so it rotates when the dtype says
    // the weights were encoded post-rotation — the same rule `run_step` derives
    // from a `Raw` input.
    let rotate_input = dtype_rotation_plan(w.dtype) == RotationPlan::FwhtG256;
    let plan = CpuStep {
        w,
        x,
        rotate_input,
        out: acc,
        residual: Some(acc),
    };
    let step_start = Instant::now();
    let arm = cpu_arm_prepare(gpu, &plan, 0..w.m)?;
    let (gemv_ns, h2d_ns) = cpu_arm_finish(gpu, &arm)?;
    finish_step(
        arm.q,
        w,
        rotate_input,
        true,
        HostExec::Cpu,
        step_start,
        StepTiming {
            d2h_ns: arm.d2h_ns,
            gemv_ns,
            h2d_ns,
        },
    );
    Ok(())
}

/// Close out an executed step: count it, charge a whole-CPU step's wall time to
/// the GPU-idle account, and feed the per-shape trace. One call site per executed
/// step, so the counters and the trace can never disagree about how many ran.
///
/// A pass-back split step is counted in [`SPLIT_STEPS`]/[`SPLIT_GPU_ROWS`]/
/// [`SPLIT_WALL_NS`] instead of [`CPU_STEPS`]/[`CPU_STEP_WALL_NS`], because its
/// wall contains real GPU work and would otherwise inflate the CPU-idle lower
/// bound [`report_idle`] documents.
///
/// `pub(crate)` so the pass-back's split step reports through the same single
/// call site.
pub(crate) fn finish_step(
    q: CpuQuant,
    w: &WeightRef,
    rotated: bool,
    residual: bool,
    exec: HostExec,
    step_start: Instant,
    timing: StepTiming,
) {
    match exec {
        HostExec::Cpu => {
            CPU_STEPS.fetch_add(1, Ordering::Relaxed);
            CPU_STEP_WALL_NS
                .fetch_add(step_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        HostExec::Split { gpu_rows } => {
            SPLIT_STEPS.fetch_add(1, Ordering::Relaxed);
            SPLIT_GPU_ROWS.fetch_add(gpu_rows as u64, Ordering::Relaxed);
            SPLIT_WALL_NS.fetch_add(step_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
    trace_step(q, w, rotated, residual, exec, timing);
}

/// Whether [`run_host_mapped_gemv`] / [`run_host_mapped_gemv_residual`] can drive
/// this weight: host-mapped *and* a decodable format.
pub fn host_mapped_cpu_capable(gpu: &Gpu, w: &WeightRef) -> bool {
    cpu_exec_enabled() && gpu.host_located(w.buf) && cpu_quant_for(w.dtype).is_some()
}

fn host_mapped_quant(gpu: &Gpu, w: &WeightRef) -> Result<CpuQuant, DispatchError> {
    if !cpu_exec_enabled() {
        return Err(cpu_err("cpu exec is not enabled (memory.offload_exec)"));
    }
    if !gpu.host_located(w.buf) {
        return Err(cpu_err("weight tensor is not host-mapped"));
    }
    cpu_quant_for(w.dtype).ok_or_else(|| {
        cpu_err(&format!(
            "no CPU decoder for dtype {:?}; this step must stay on the GPU",
            w.dtype
        ))
    })
}

/// A host-mapped weight step that the CPU can execute.
pub(crate) struct CpuStep<'a> {
    w: &'a WeightRef<'a>,
    x: &'a GpuTensor,
    /// Whether the activation must be FWHT-rotated (and AWQ-divided) before the
    /// multiply. Load-bearing: `true` means "the launcher would rotate this",
    /// `false` means someone already did — rotating a pre-rotated input again is
    /// a silent $\mathcal{R}^2$ error that still looks like plausible
    /// activations.
    rotate_input: bool,
    out: &'a GpuTensor,
    /// `Some` for `Step::GemvResidual`, which accumulates in place into the
    /// residual (never into `out` — see the GemvResidual arms in `steps.rs`).
    residual: Option<&'a GpuTensor>,
}

/// Plan a CPU execution for `step`, or `None` when it must stay on the GPU:
/// CPU execution disabled, weight not host-mapped, or a format
/// [`cpu_quant_for`] does not cover.
pub(crate) fn plan_step<'a>(gpu: &Gpu, step: &'a Step<'a>) -> Option<CpuStep<'a>> {
    if !cpu_exec_enabled() {
        return None;
    }
    let (w, input, out, residual) = match step {
        Step::Gemv { w, input, out } => (*w, input, *out, None),
        Step::GemvResidual {
            w,
            input,
            residual,
            out,
        } => (*w, input, *out, Some(*residual)),
        _ => return None,
    };
    if !gpu.host_located(w.buf) || cpu_quant_for(w.dtype).is_none() {
        return None;
    }
    // Mirror `GemvFamily::run_input`: a `Raw` input is rotated by the launcher
    // *and only when* the dtype's plan is `FwhtG256`; a `Prerotated` input is
    // never touched again.
    let rotate_input = matches!(input, GemvInput::Raw(_))
        && dtype_rotation_plan(w.dtype) == RotationPlan::FwhtG256;
    Some(CpuStep {
        w,
        x: match input {
            GemvInput::Raw(t) | GemvInput::Prerotated(t) => t,
        },
        rotate_input,
        out,
        residual,
    })
}

/// Whether `step` reads a host-mapped weight, regardless of whether the CPU can
/// decode it — the coverage counter's question.
pub(crate) fn reads_host_mapped_weight(gpu: &Gpu, step: &Step) -> bool {
    match step {
        Step::Gemv { w, .. } | Step::GemvResidual { w, .. } => gpu.host_located(w.buf),
        _ => false,
    }
}

/// Execute a planned CPU step, whole (`memory.offload_exec=cpu`).
///
/// The order is the observable contract: D2H x, [D2H acc], gemv, [add], H2D.
pub(crate) fn run_step(gpu: &Gpu, plan: &CpuStep) -> Result<(), DispatchError> {
    let step_start = Instant::now();
    let arm = cpu_arm_prepare(gpu, plan, 0..plan.w.m)?;
    let (gemv_ns, h2d_ns) = cpu_arm_finish(gpu, &arm)?;
    finish_step(
        arm.q,
        plan.w,
        plan.rotate_input,
        plan.residual.is_some(),
        HostExec::Cpu,
        step_start,
        StepTiming {
            d2h_ns: arm.d2h_ns,
            gemv_ns,
            h2d_ns,
        },
    );
    Ok(())
}

/// Count a host-mapped step that is about to be launched on the GPU anyway.
pub(crate) fn count_host_mapped_gpu_step() {
    HOST_MAPPED_GPU_STEPS.fetch_add(1, Ordering::Relaxed);
}

fn cpu_err(msg: &str) -> DispatchError {
    DispatchError::Hip(format!("cpu exec: {msg}"))
}

fn download_f32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Result<Vec<f32>, DispatchError> {
    assert!(
        t.numel() >= n,
        "cpu exec: tensor has {} elements, need {n}",
        t.numel()
    );
    let mut out = vec![0.0f32; n];
    // Safety: `out` is a live `n`-element f32 buffer; the slice covers exactly
    // those bytes and does not outlive the call.
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.memcpy_dtoh_auto(bytes, &t.buf)
        .map_err(|e| cpu_err(&format!("D2H: {e}")))?;
    Ok(out)
}

fn upload_f32(gpu: &Gpu, t: &GpuTensor, v: &[f32]) -> Result<(), DispatchError> {
    assert!(
        t.numel() >= v.len(),
        "cpu exec: destination has {} elements, source has {}",
        t.numel(),
        v.len()
    );
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    gpu.memcpy_htod_auto(&t.buf, bytes)
        .map_err(|e| cpu_err(&format!("H2D: {e}")))
}

/// One step's own wall-time split, handed to [`trace_step`].
///
/// For a pass-back split step `h2d_ns` is the *join* (the blocking H2D, which
/// waits for the concurrently launched GPU arm), and `gemv_ns` is the CPU arm's
/// multiply — which is what the scheduler's `observe` samples.
pub(crate) struct StepTiming {
    pub(crate) d2h_ns: u64,
    pub(crate) gemv_ns: u64,
    pub(crate) h2d_ns: u64,
}

/// Per-shape step accounting under `HIPFIRE_CPU_EXEC_TRACE=1`: one line per
/// distinct step shape at its first call and then at every doubling of that
/// shape's call count, followed by the running counters.
///
/// Both halves of that schedule matter. The `calls=1` line is the *cold* first
/// step of the shape (host-mapped page first touch, rayon pool wake-up); the
/// later lines are the shape's own steady state, which is the only number worth
/// quoting. Printing the whole process-wide mean per shape — as this did before
/// — gives every shape the same figure and inflates it with the cold steps of
/// whatever ran first, which reads as attribution and is not.
///
/// The counters are the coverage signal: `host-mapped steps still on GPU` must be
/// 0 for a model the CPU covers, so a shape that silently never reaches the seam
/// shows up as a number rather than as a mystery.
///
/// A pass-back shape prints its own line instead (`split: …`): for it `join` is
/// the blocking H2D, `copy + max(0, gpu_ns - cpu_ns)`. What the scheduler's
/// controller reads is the join's excess over the shape's own **no-wait floor**
/// (the smallest join seen), so `gpu samples` on that line says how often the
/// GPU rate its share rests on was actually observed: zero means the share is
/// still being pushed up because every join so far sat at the floor, and
/// `gpu≥GB/s` is then a lower bound that says nothing about the real rate (which
/// is why it is printed as a bound).
fn trace_step(
    q: CpuQuant,
    w: &WeightRef,
    rotated: bool,
    residual: bool,
    exec: HostExec,
    timing: StepTiming,
) {
    if hipfire_config::developer_var("HIPFIRE_CPU_EXEC_TRACE").is_err() {
        return;
    }
    let key = (q as u8, w.m, w.k, rotated, residual, w.awq_scale.is_some());
    let Ok(mut shapes) = SHAPES.lock() else {
        return;
    };
    let stats = shapes.entry(key).or_default();
    if stats.calls == 0 {
        stats.split = matches!(exec, HostExec::Split { .. });
    }
    stats.calls += 1;
    stats.d2h_ns += timing.d2h_ns;
    stats.gemv_ns += timing.gemv_ns;
    stats.h2d_ns += timing.h2d_ns;
    if let HostExec::Split { gpu_rows } = exec {
        stats.split_gpu_rows += gpu_rows as u64;
        stats.split_cpu_rows += (w.m - gpu_rows) as u64;
    }
    let stats = *stats;
    // The GPU-idle window report rides the *global* step count's doubling
    // schedule, so it exists even for a spill whose every shape is new — and a
    // pure pass-back run still has a global step count (the splits).
    let (on_cpu, on_gpu) = cpu_exec_counters();
    let (splits, _) = split_counters();
    let total = on_cpu + splits;
    if total.is_power_of_two() {
        report_idle(total);
    }
    if !stats.calls.is_power_of_two() {
        return;
    }
    let (m, k) = (w.m, w.k);
    // ns -> ms, averaged over *this shape's* calls so far.
    let per_ms = |ns: u64| ns as f64 / 1e6 / stats.calls as f64;
    if stats.split {
        // The CPU arm runs while the GPU arm streams the *same* host pages: `cpu`
        // is the contended rate and `gpu≥` the lower bound the controller samples
        // (see `offload_split`). bytes/ns is GB/s numerically.
        let row_bytes = (w.buf.buf.size() / w.m.max(1)) as f64;
        let cpu_arm_ns = (stats.d2h_ns + stats.gemv_ns).max(1) as f64;
        let whole_ns = (stats.d2h_ns + stats.gemv_ns + stats.h2d_ns).max(1) as f64;
        let cpu_gbs = stats.split_cpu_rows as f64 * row_bytes / cpu_arm_ns;
        let gpu_gbs = stats.split_gpu_rows as f64 * row_bytes / whole_ns;
        let (share, control) = match crate::offload_split::shape_state(w.dtype, w.k) {
            Some(s) => (
                format!("{:.3}", s.share),
                format!(
                    " (gpu samples {}, floor={}, last waited={}, target={}, {} applied, {} reopens{})",
                    s.gpu_samples,
                    s.min_join_ns
                        .map(|ns| format!("{:.2}ms", ns as f64 / 1e6))
                        .unwrap_or_else(|| "—".into()),
                    s.last_waited,
                    s.last_target
                        .map(|t| format!("{t:.3}"))
                        .unwrap_or_else(|| "—".into()),
                    s.applied,
                    s.reopens,
                    if s.frozen { ", frozen" } else { "" },
                ),
            ),
            None => ("n/a".to_string(), String::new()),
        };
        eprintln!(
            "split: step gemv m={m} k={k} quant={q:?} gpu_rows={:.0}/{m} share={share}{control} \
             rotated={rotated} residual={residual} awq={} | {} calls | {total} steps ({splits} \
             split), {on_gpu} host-mapped steps still on GPU | mean per split step: \
             d2h={:.2}ms gemv={:.2}ms join={:.2}ms | cpu={cpu_gbs:.1}GB/s gpu\u{2265}{gpu_gbs:.1}GB/s",
            stats.split_gpu_rows as f64 / stats.calls as f64,
            w.awq_scale.is_some(),
            stats.calls,
            per_ms(stats.d2h_ns),
            per_ms(stats.gemv_ns),
            per_ms(stats.h2d_ns),
        );
        return;
    }
    eprintln!(
        "cpu exec: step gemv m={m} k={k} quant={q:?} rotated={rotated} residual={residual} \
         awq={} | {} calls | {on_cpu} steps on CPU, {on_gpu} host-mapped steps still on GPU | \
         mean per step: d2h={:.2}ms gemv={:.2}ms h2d={:.2}ms",
        w.awq_scale.is_some(),
        stats.calls,
        per_ms(stats.d2h_ns),
        per_ms(stats.gemv_ns),
        per_ms(stats.h2d_ns)
    );
}
