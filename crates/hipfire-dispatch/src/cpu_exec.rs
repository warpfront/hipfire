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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use rdna_compute::{DType, Gpu, GpuTensor};

use hipfire_cpu::epilogue::residual_add;
use hipfire_cpu::gemv::gemv;
use hipfire_cpu::quant::{divide_by_awq_scale, rotate_x, CpuQuant};

use crate::families::gemv::WeightRef;
use crate::pipeline::steps::{GemvInput, Step};
use crate::types::{dtype_rotation_plan, DispatchError, RotationPlan};

/// Cached `memory.offload_exec == Cpu`, mirroring the `forward_lowered_enabled`
/// precedent: resolved once from the process snapshot, never re-read per step.
pub fn cpu_exec_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        hipfire_config::memory::offload_exec() == hipfire_config::memory::OffloadExec::Cpu
    });
    *ENABLED
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

pub fn cpu_exec_counters() -> (usize, usize) {
    (
        CPU_STEPS.load(Ordering::Relaxed),
        HOST_MAPPED_GPU_STEPS.load(Ordering::Relaxed),
    )
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
    h2d_ns: u64,
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
        return Err(
            "memory.offload_exec=cpu conflicts with the retained-replay (Redline) backend: the \
             replay tape records GPU launches and does not execute the CPU-executed steps, so a \
             replayed route would compute from stale activations. Set memory.offload_exec=pcie \
             (HIPFIRE_OFFLOAD_EXEC=pcie) or replay.backend=hip"
                .to_string(),
        );
    }
    Ok(())
}

/// Log the capture-disable decision once per process. A CPU-executed step is a
/// host sync point (a D2H and an H2D around the multiplication), so a graph that
/// contained one could neither be recorded nor replayed correctly.
pub fn log_capture_disabled_once() {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        eprintln!("cpu exec: hipGraph capture disabled (CPU-executed steps present)");
    });
}

/// `out[0..m] = W · x[0..k]` on the CPU, over a host-mapped weight.
///
/// `rotate_x` must be `true` exactly when the launcher would self-rotate a `Raw`
/// input for this dtype (i.e. `dtype_rotation_plan(w.dtype) == FwhtG256`), which
/// [`RotationPlan`] is the single source of truth for; [`run_gemv`] derives it
/// so no caller has to.
pub fn run_host_mapped_gemv(
    gpu: &Gpu,
    w: &WeightRef,
    x: &GpuTensor,
    rotate_input: bool,
    out: &GpuTensor,
) -> Result<(), DispatchError> {
    let q = host_mapped_quant(gpu, w)?;
    let (m, k) = (w.m, w.k);
    let bytes = gpu
        .host_bytes(w.buf)
        .ok_or_else(|| cpu_err("weight tensor is host-mapped but has no host pointer"))?;
    let (x_host, d2h_ns) = prepare_activation(gpu, w, x, rotate_input)?;
    let mut y = vec![0.0f32; m];
    let t1 = Instant::now();
    gemv(q, bytes, m, k, &x_host, &mut y);
    let gemv_ns = t1.elapsed().as_nanos() as u64;
    let t2 = Instant::now();
    upload_f32(gpu, out, &y)?;
    let h2d_ns = t2.elapsed().as_nanos() as u64;
    CPU_STEPS.fetch_add(1, Ordering::Relaxed);
    trace_step(
        q,
        w,
        rotate_input,
        false,
        StepTiming {
            d2h_ns,
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
fn prepare_activation(
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
    let q = host_mapped_quant(gpu, w)?;
    let (m, k) = (w.m, w.k);
    let rotate_input = dtype_rotation_plan(w.dtype) == RotationPlan::FwhtG256;
    let bytes = gpu
        .host_bytes(w.buf)
        .ok_or_else(|| cpu_err("weight tensor is host-mapped but has no host pointer"))?;
    let (x_host, d2h_ns) = prepare_activation(gpu, w, x, rotate_input)?;
    let mut y = vec![0.0f32; m];
    let t1 = Instant::now();
    gemv(q, bytes, m, k, &x_host, &mut y);
    let gemv_ns = t1.elapsed().as_nanos() as u64;
    let t2 = Instant::now();
    let mut acc_host = download_f32(gpu, acc, m)?;
    residual_add(&mut acc_host, &y);
    upload_f32(gpu, acc, &acc_host)?;
    let h2d_ns = t2.elapsed().as_nanos() as u64;
    CPU_STEPS.fetch_add(1, Ordering::Relaxed);
    trace_step(
        q,
        w,
        rotate_input,
        true,
        StepTiming {
            d2h_ns,
            gemv_ns,
            h2d_ns,
        },
    );
    Ok(())
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

/// Rotation disposition of a step's input. The distinction is load-bearing:
/// `Raw` means "the launcher would rotate this", `Prerotated` means someone
/// already did — rotating a `Prerotated` input again is a silent
/// $\mathcal{R}^2$ error that still looks like plausible activations.
enum CpuInput<'a> {
    Raw(&'a GpuTensor),
    Prerotated(&'a GpuTensor),
}

impl<'a> CpuInput<'a> {
    fn tensor(&self) -> &'a GpuTensor {
        match self {
            CpuInput::Raw(t) | CpuInput::Prerotated(t) => t,
        }
    }
}

/// A host-mapped weight step that the CPU can execute.
pub(crate) struct CpuStep<'a> {
    w: &'a WeightRef<'a>,
    input: CpuInput<'a>,
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
    Some(CpuStep {
        w,
        input: match input {
            GemvInput::Raw(t) => CpuInput::Raw(t),
            GemvInput::Prerotated(t) => CpuInput::Prerotated(t),
        },
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

/// Execute a planned CPU step.
pub(crate) fn run_step(gpu: &Gpu, plan: &CpuStep) -> Result<(), DispatchError> {
    match plan.residual {
        // The fused down-projection this splits always feeds the CPU path an
        // unrotated activation (the SiLU output), so it rotates when the dtype
        // says the weights were encoded post-rotation.
        Some(residual) => run_host_mapped_gemv_residual(gpu, plan.w, plan.input.tensor(), residual),
        // Mirror `GemvFamily::run_input`: a `Raw` input is rotated by the
        // launcher *and only when* the dtype's plan is FwhtG256; a `Prerotated`
        // input is never touched again.
        None => run_host_mapped_gemv(
            gpu,
            plan.w,
            plan.input.tensor(),
            matches!(plan.input, CpuInput::Raw(_))
                && dtype_rotation_plan(plan.w.dtype) == RotationPlan::FwhtG256,
            plan.out,
        ),
    }
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
struct StepTiming {
    d2h_ns: u64,
    gemv_ns: u64,
    h2d_ns: u64,
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
fn trace_step(q: CpuQuant, w: &WeightRef, rotated: bool, residual: bool, timing: StepTiming) {
    if hipfire_config::developer_var("HIPFIRE_CPU_EXEC_TRACE").is_err() {
        return;
    }
    let key = (q as u8, w.m, w.k, rotated, residual, w.awq_scale.is_some());
    let Ok(mut shapes) = SHAPES.lock() else {
        return;
    };
    let stats = shapes.entry(key).or_default();
    stats.calls += 1;
    stats.d2h_ns += timing.d2h_ns;
    stats.gemv_ns += timing.gemv_ns;
    stats.h2d_ns += timing.h2d_ns;
    let stats = *stats;
    if !stats.calls.is_power_of_two() {
        return;
    }
    let (m, k) = (w.m, w.k);
    let (on_cpu, on_gpu) = cpu_exec_counters();
    // ns -> ms, averaged over *this shape's* calls so far.
    let per_ms = |ns: u64| ns as f64 / 1e6 / stats.calls as f64;
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
