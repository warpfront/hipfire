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

use hipfire_cpu::epilogue::{residual_add, silu_mul};
use hipfire_cpu::gemv::gemv;
use hipfire_cpu::quant::{divide_by_awq_scale, rotate_x, rotate_x_128_v2, CpuQuant, InputRotation};

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
    static MODE: LazyLock<HostExecMode> =
        LazyLock::new(|| match hipfire_config::memory::offload_exec() {
            hipfire_config::memory::OffloadExec::Cpu => HostExecMode::Cpu,
            hipfire_config::memory::OffloadExec::Passback => HostExecMode::Passback,
            hipfire_config::memory::OffloadExec::Pcie => HostExecMode::Pcie,
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

/// Whether the CPU expert splice is armed: `memory.offload_exec=cpu` (or
/// `passback`, which splits only dense steps and runs routed experts on this
/// splice) and the `HIPFIRE_MOE_CPU_EXPERTS=0` kill-switch unset. This is the *model-level*
/// decision; the loader additionally requires a host-placed packed layer with no
/// AWQ sidecar, and records the armed sink on the layer weights.
pub fn moe_cpu_experts_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        cpu_exec_enabled()
            && hipfire_config::developer_var("HIPFIRE_MOE_CPU_EXPERTS")
                .ok()
                .as_deref()
                != Some("0")
    });
    *ENABLED
}

/// The splice pool configuration. Knobs (both developer vars, read once when
/// first needed): `HIPFIRE_CPU_EXPERT_THREADS` (worker count; `0`/unset = one
/// per physical core in the process's allowed CPU set) and
/// `HIPFIRE_CPU_EXPERT_PIN` (`0` = do not pin workers; default on).
fn splice_pool_config() -> hipfire_cpu::splice_pool::SplicePoolConfig {
    use hipfire_cpu::splice_pool::SplicePoolConfig;
    static CFG: LazyLock<SplicePoolConfig> = LazyLock::new(|| SplicePoolConfig {
        threads: hipfire_config::developer_var("HIPFIRE_CPU_EXPERT_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0),
        pin: hipfire_config::developer_bool("HIPFIRE_CPU_EXPERT_PIN", true),
    });
    *CFG
}

/// The pinned physical-core pool the routed-expert splice runs on.
pub(crate) fn in_splice_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    hipfire_cpu::splice_pool::install(splice_pool_config(), f)
}

/// Streaming-read bandwidth of host RAM measured on the splice pool: the
/// pinned pool (one worker per physical core) the routed-expert splice runs
/// on, so the figure is what that splice can actually pull from system
/// memory. `hipfire calibrate` records it.
pub fn splice_read_bandwidth(bytes: usize, reps: usize) -> hipfire_cpu::membw::ReadBandwidth {
    in_splice_pool(|| hipfire_cpu::membw::read_bandwidth(bytes, reps))
}

/// Worker count of the splice pool (builds the pool on first call, with the
/// same config every splice uses). For arming logs only.
pub fn splice_pool_threads() -> usize {
    hipfire_cpu::splice_pool::splice_pool(splice_pool_config()).current_num_threads()
}

/// Whether the expanded MoE timing trace is armed: the existing
/// `HIPFIRE_CPU_EXEC_TRACE` diagnostic. Resolved per call (not cached) so the
/// debug script's trace-off control run in the same process snapshot shape
/// still works; the lookup is a snapshot-map read, off the hot arithmetic.
pub fn moe_trace_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_CPU_EXEC_TRACE").is_ok()
}
/// (steps executed on the CPU, host-mapped GEMV steps that stayed on the GPU).
///
/// The second number is the honest coverage-failure signal for the step seam
/// only: it counts `Step::Gemv` / `Step::GemvResidual` steps whose weight tensor
/// *was* host-mapped while CPU execution was enabled but whose shape never
/// reached the CPU — either an unsupported quant format or a fused launch the
/// seam did not split. The MoE tier is structurally invisible to it: the grouped
/// host read a wide prefill deliberately keeps on the GPU is never counted, nor
/// is the `llama.rs` fused split. The MoE tier's coverage is reported by the
/// load-time coverage line (`cpu exec: N/N spilled layers fully covered by the
/// CPU seam … uncovered quants: …`), which remains the authority there. Zero
/// means every host-mapped GEMV step this process executed went to the CPU.
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

/// Slots per `CpuQuant` discriminant (`q as usize`) in one splice call's
/// selected experts. Length 32 covers the 28 current variants with headroom;
/// indices at/above the variant count stay zero. Printed as `NamexCount`
/// pairs via [`cpu_quant_mix_name`].
const MOE_QUANT_MIX_LEN: usize = 32;

/// Per-shape totals for the MoE expert splice: the d2h / gemv / h2d split plus
/// the coarse setup/middle/combine phases and total wall of one
/// `(quant, dim, mi, top_k, rows, graded)` shape's calls. `quant` is the first
/// selected expert's gate_up format (a representative); `graded` says the call's
/// experts were not all one tier.
#[derive(Clone, Copy, Default)]
struct MoeStepStats {
    calls: usize,
    /// Four D2H submissions, including HIP API overhead.
    d2h_enqueue_ns: u64,
    /// Single completion barrier for all four downloads.
    d2h_complete_ns: u64,
    /// All experts' gate_up GEMV.
    gu_ns: u64,
    /// All experts' down GEMV.
    dn_ns: u64,
    h2d_ns: u64,
    /// Resolving the selected slots to host experts (ids/spans), before any
    /// arithmetic.
    setup_ns: u64,
    /// All experts' SiLU + optional hidden rotation, the whole per-token loop.
    middle_ns: u64,
    /// All experts' routing-weighted accumulate into the residual.
    combine_ns: u64,
    /// The whole splice call's wall time, exclusive of the trace print itself.
    total_ns: u64,
    /// GU activation-prep/metadata vs Rayon decode-and-dot execution split.
    /// `gu_ns` stays the whole region (`prep + exec`); `prep` is 0 until the
    /// traced `gemv_shared_sourced` sibling lands.
    gu_prep_ns: u64,
    gu_exec_ns: u64,
    /// DN activation-prep/metadata vs Rayon decode-and-dot execution split.
    dn_prep_ns: u64,
    dn_exec_ns: u64,
    /// CPU middle split: gather vs SiLU vs FWHT, summed worker time across the
    /// slot-parallel middle region (may exceed `middle_ns`).
    middle_gather_ns: u64,
    middle_silu_ns: u64,
    middle_fwht_ns: u64,
    /// Host time to drain the producers before downloads (trace only). Includes
    /// the synchronization API cost, not physical copy time.
    d2h_wait_ns: u64,
    /// Selected-slot GU / DN quant mixture summed across this shape's calls.
    gu_mix: [u64; MOE_QUANT_MIX_LEN],
    dn_mix: [u64; MOE_QUANT_MIX_LEN],
}

/// `(quant, dim, mi, top_k, rows, graded)` → running per-shape totals.
static MOE_SHAPES: LazyLock<Mutex<BTreeMap<(u8, usize, usize, usize, usize, bool), MoeStepStats>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// `DType` → the decoder for it, for exactly the formats `hipfire_cpu` can
/// decode. `None` means the step stays on the GPU (over PCIe): not a correctness
/// problem, only a smaller bandwidth win, and the load-time coverage line names
/// the format.
///
/// Invariant, pinned by `cpu_quant_rotation_agrees_with_plan`: for every dtype
/// here, [`CpuQuant::input_rotation`] agrees with [`dtype_rotation_plan`]
/// (FWHT-256 <-> `FwhtG256`, FWHT-128 <-> `FwhtG128`), so the seam and the
/// launcher cannot disagree about whether a `Raw` activation arrives rotated.

/// Every `DType` [`cpu_quant_for`] maps to `Some`: the declaration for
/// consumers and tests (both `cpu_quant_for` pins iterate this). The match in
/// [`cpu_quant_for`] keeps its explicit per-dtype arms; this lists the same set.
pub const CPU_EXECUTED_DTYPES: &[DType] = &[
    DType::MQ4G256,
    DType::MQ4G256V2,
    DType::MQ6G256,
    DType::MQ6G256V2,
    DType::MQ5G256,
    DType::MQ5G256V2,
    DType::MQ4CG256,
    DType::MQ3G256,
    DType::MQ3G256V2,
    DType::MQ3G256Lloyd,
    DType::MQ2G256,
    DType::MQ2G256V2,
    DType::MQ2G256Lloyd,
    DType::MQ2G256LloydU,
    DType::MQ4G256Lloyd,
    DType::MQ4G128V2,
    DType::MFP4G32E8,
    DType::HFQ6G256,
    DType::HFQ4G256,
    DType::HFQ4G128,
    DType::HFQ3G256,
    DType::HFQ3G128,
    DType::HFQ2G256,
    DType::HFQ2G128,
    DType::TQ2G128,
    DType::BQ1G128,
    DType::Q8_0,
    DType::F16,
    DType::F32,
    DType::BF16,
    DType::MQ3I4G128R256,
    DType::MQ3I4G128R128,
    DType::E8PI4G128R256,
    DType::E8PI4G128R128,
];

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
        DType::MQ4G128V2 => Some(CpuQuant::Mq4G128V2),
        DType::MFP4G32E8 => Some(CpuQuant::Mfp4G32E8),
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
        // Planar expert-slab payloads (codes plane, then scales plane): routed
        // experts only, never a row-addressed dense step ([`dense_cpu_quant_for`]).
        DType::MQ3I4G128R256 => Some(CpuQuant::Mq3I4G128R256),
        DType::MQ3I4G128R128 => Some(CpuQuant::Mq3I4G128R128),
        DType::E8PI4G128R256 => Some(CpuQuant::E8pI4G128R256),
        DType::E8PI4G128R128 => Some(CpuQuant::E8pI4G128R128),
        _ => None,
    }
}

/// [`cpu_quant_for`] restricted to the row-addressed formats a dense host step
/// can execute: the dense paths slice `w.buf` into `m` equal row strides, which
/// a planar expert-slab payload (all code rows, then all scale rows) does not
/// have, so its dtypes stay on the GPU there.
pub fn dense_cpu_quant_for(dtype: DType) -> Option<CpuQuant> {
    cpu_quant_for(dtype).filter(|q| !q.is_planar())
}

/// The CPU row-dot kernel this host runs for `dtype`
/// ([`hipfire_cpu::simd::row_dot_kernel`]: `avx2-fma-f16c`, `avx2-fma` or
/// `scalar`); `None` for a dtype without a CPU decoder.
pub fn cpu_kernel_for(dtype: DType) -> Option<&'static str> {
    cpu_quant_for(dtype).map(hipfire_cpu::simd::row_dot_kernel)
}

/// Whether a `Raw` input of this dtype is rotated by the launcher — i.e. the
/// CPU path must rotate it itself: `FwhtG256` (every FWHT-256 format) or, for
/// the one CPU-executed FWHT-128 format (qt 53), `FwhtG128`.
pub(crate) fn raw_input_is_rotated(dtype: DType) -> bool {
    match dtype_rotation_plan(dtype) {
        RotationPlan::FwhtG256 => true,
        RotationPlan::FwhtG128 => {
            cpu_quant_for(dtype).is_some_and(|q| q.input_rotation() == InputRotation::FwhtG128V2)
        }
        _ => false,
    }
}

/// Whether a model with this many host-placed weight layers has CPU-executed
/// steps at all: CPU execution selected *and* a non-empty host placement. With
/// no spill every weight stays device-resident, [`Gpu::host_located`] is false
/// everywhere, and not a single step can move to the CPU — `memory.offload_exec=cpu`
/// must not change a number there.
///
/// `host_weight_layers` counts every layer with a host-mapped weight the CPU may
/// multiply: whole-layer spills (the step seam) *and* routed-expert layers (the
/// MoE CPU-down splice). Conservative by construction: a spill whose every weight
/// is an unsupported format also reports `true`, which costs a hipGraph but never
/// correctness.
pub fn cpu_offload_active(host_weight_layers: usize) -> bool {
    cpu_exec_enabled() && host_weight_layers > 0
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
        match cpu_quant_for(w.dtype).map(CpuQuant::input_rotation) {
            Some(InputRotation::FwhtG128V2) => rotate_x_128_v2(&mut host),
            _ => rotate_x(&mut host),
        }
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
    let rotate_input = raw_input_is_rotated(w.dtype);
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

/// One selected expert's host-resident weight bytes for the CPU splice.
///
/// Each projection carries its own format, so a graded layer's experts (MQ6 hot
/// / MQ4 mid / MQ3-Lloyd cold) resolve per expert, and an expert whose two
/// projections differ is expressible too. The slices are the expert's *own*
/// extents — no stride padding: `gate_up` is exactly
/// `tensor_bytes(gate_up_quant, 2 * mi, dim)` bytes and `down` exactly
/// `tensor_bytes(down_quant, dim, mi)` (`hipfire_cpu::gemv::tensor_bytes`,
/// which covers the planar expert-slab payloads too). The engine validates and
/// refuses a short slice rather than reading past it.
///
/// Owns nothing, so the splice can hold a whole window's resolved slots in a
/// fixed-capacity stack array instead of a heap `Vec` (`Copy` is what makes
/// `[CpuMoeExpert; N]` initializable without an allocation).
#[derive(Clone, Copy)]
pub struct CpuMoeExpert<'a> {
    pub gate_up_quant: CpuQuant,
    pub gate_up: &'a [u8],
    pub down_quant: CpuQuant,
    pub down: &'a [u8],
}

/// Reusable host workspaces for the MoE CPU splice, kept in a thread-local so a
/// per-token decode does not re-allocate them every layer. Resized on demand,
/// never shrunk — the buffers are the splice's, not the caller's.
///
/// The first four are the arithmetic `f32` workspaces. The rest is index metadata
/// (counts and offsets, no borrows), rebuilt each call in reusable storage so
/// the warm path makes no heap allocation: the selected slots are grouped by
/// expert id by a counting sort, which needs no per-group `Vec`. The
/// borrow-carrying job descriptors cannot live here at all — they borrow these
/// very buffers — so they are answered on demand through
/// [`MoeJobSource`] instead.
#[derive(Default)]
struct MoeScratch {
    gu: Vec<f32>,
    hidden: Vec<f32>,
    dn: Vec<f32>,
    /// Slots in first-seen group order: group `g` owns
    /// `order[group_start[g]..group_start[g+1]]`.
    order: Vec<u32>,
    /// `groups + 1` start offsets into [`Self::order`].
    group_start: Vec<u32>,
    /// The global expert id of each group.
    group_id: Vec<u32>,
    /// Group index of each slot.
    slot_group: Vec<u32>,
    /// Position of each slot within its group (its column in the group's
    /// row-major output run).
    slot_si: Vec<u32>,
    /// Element offset of each group's output run, plus a trailing total
    /// (`groups + 1`); refilled per projection with that projection's row count.
    group_off: Vec<usize>,
}

thread_local! {
    static MOE_SCRATCH: std::cell::RefCell<MoeScratch> =
        const { std::cell::RefCell::new(MoeScratch {
            gu: Vec::new(),
            hidden: Vec::new(),
            dn: Vec::new(),
            order: Vec::new(),
            group_start: Vec::new(),
            group_id: Vec::new(),
            slot_group: Vec::new(),
            slot_si: Vec::new(),
            group_off: Vec::new(),
        }) };
    static CPU_TRANSFER: std::cell::RefCell<Option<hip_bridge::PinnedTransferBuffer>> =
        const { std::cell::RefCell::new(None) };
}

/// The top-k scratch's expert id, decoded exactly as the loader packed it: the
/// router writes a `u16` id as the low bits of an `f32` bit pattern.
fn expert_id(raw: f32) -> usize {
    (raw.to_bits() as i32) as u16 as usize
}

/// BF16 round-to-nearest-even of a finite `f32`, returned widened to `f32`;
/// non-finite values pass through unchanged. Integer-bit arithmetic identical to
/// the GPU `rt` helpers (`router_bf16_rt`, `rot128_bf16_rt`): add `0x7fff` plus
/// the kept LSB, then truncate the low 16 bits. Signed zero and the sign of
/// every finite value are preserved; a finite value that rounds past the BF16
/// maximum carries into the exponent and becomes ±Inf, as on the GPU.
#[inline]
fn bf16_rt(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xffff_0000)
}

/// `gate_up` extent of one expert: `[2 * mi, dim]` in its format, checked.
fn expert_gu_bytes(q: CpuQuant, dim: usize, mi: usize) -> Result<usize, DispatchError> {
    let rows = mi
        .checked_mul(2)
        .ok_or_else(|| cpu_err("moe cpu expert splice: gate_up extent overflows"))?;
    tensor_extent(q, rows, dim)
        .ok_or_else(|| cpu_err("moe cpu expert splice: gate_up extent overflows"))
}

/// `down` extent of one expert: `[dim, mi]` in its format, checked.
fn expert_down_bytes(q: CpuQuant, dim: usize, mi: usize) -> Result<usize, DispatchError> {
    tensor_extent(q, dim, mi).ok_or_else(|| cpu_err("moe cpu expert splice: down extent overflows"))
}

/// `hipfire_cpu::gemv::tensor_bytes(q, m, k)` with the multiplication checked:
/// one row's bytes (codes plus, for a planar format, its scales) times `m`.
fn tensor_extent(q: CpuQuant, m: usize, k: usize) -> Option<usize> {
    hipfire_cpu::gemv::tensor_bytes(q, 1, k).checked_mul(m)
}

/// Staging for [`moe_cpu_experts`]' borrowed-expert window: one preallocated
/// stack of `64 * 10 = 640` slots (the full FN logical window at top_k 10),
/// written exactly once per slot with no 640-zeroing and no per-window heap for
/// 1..=640 slots. Resolver passes fill it in physical tiles of at most
/// [`MOE_PHYS_TILE`] slots (4 rows × top_k 10), but the whole logical window's
/// slots resolve in this single staging so the proven arithmetic sites run
/// unchanged; the single window-wide `MoeScratch` reservation still covers the
/// full logical window. Wider non-FN windows (>640) keep the heap fallback.
const MOE_PHYS_SLOTS: usize = 64 * 10;
/// Physical tile width (slots) one resolver pass fills: 4 rows × top_k 10.
const MOE_PHYS_TILE: usize = 4 * 10;

/// CPU FFN for a window of `rows` tokens' routed experts: recompute every
/// selected expert (gate_up GEMV + SiLU + hidden-rotate + down GEMV) on the CPU
/// from the intact host blobs and accumulate the routing-weighted result into
/// the residual, one row at a time.
///
/// `x_rot` is the already-rotated activation the gate_up projection consumes; the
/// post-SiLU hidden is rotated here exactly as the fused kernel would (per the
/// *down* projection's format). The sealed MoE program of a CPU-deferred call
/// ([`MoeParams::cpu_deferred_experts`](crate::families::moe::MoeParams::cpu_deferred_experts))
/// omits every routed GPU stage, so this supplies the whole routed contribution.
/// `rows` is the window width (1 for decode, up
/// to the narrow MTP verify width) and `top_k` is `num_experts_per_tok`; both are
/// explicit so a tensor's spare capacity can never silently become the batch
/// size.
///
/// `resolve` maps a selected expert's global id to its host bytes. It is called
/// once per selected slot (`rows * top_k` calls, row-major then rank order) up
/// front, may be asked for the same id more than once, and must be a pure
/// function of the id.
///
/// `bf16_round_trip` selects the GPU BF16 storage boundaries of the grouped
/// QT44/QT53 decode route instead of plain f32 arithmetic. `false` is the
/// original arithmetic, byte for byte. `true` applies
/// `rt` = BF16 round-to-nearest-even (non-finite passes through) at exactly
/// these points: each gathered gate and up value before SiLU; the activated
/// hidden `rt(silu(rt g) * rt u)` before the FWHT (the rotated down input stays
/// f32); and the combine, which starts every row at `rt(residual)` and then adds
/// that row's selected experts in ASCENDING EXPERT ID (equal ids: lower rank
/// first — not routing-rank order), each step
/// `acc = rt(acc + rt(rt(down) * rt(weight)))`. The gate/up and down GEMVs stay
/// f32 with the CPU kernels' own summation order, which intentionally differs
/// from the GPU's reduction tree.
///
/// Fail-closed by construction: an unknown expert id, unavailable host bytes, an
/// undecodable dtype, an AWQ sidecar (a bare `gemv` would ignore the per-expert
/// scale), graph capture / replay recording (a CPU step is a host sync point), a
/// `dim` that is not a multiple of 256, an `mi` that is not a multiple of 128 (or
/// of the selected down format's group width: 256 for FWHT-256 formats, 128 for
/// the QT53 `Mq4G128V2` down), or short scratch are all errors,
/// never silent zeros.
#[allow(clippy::too_many_arguments)]
pub fn moe_cpu_experts<'a>(
    gpu: &Gpu,
    dim: usize,
    mi: usize,
    rows: usize,
    top_k: usize,
    x_rot: &GpuTensor,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
    residual: &GpuTensor,
    awq: bool,
    bf16_round_trip: bool,
    resolve: impl Fn(usize) -> Result<CpuMoeExpert<'a>, DispatchError>,
) -> Result<(), DispatchError> {
    if gpu.graphs.capture_mode || gpu.replay.is_recording() {
        return Err(cpu_err(
            "moe cpu expert splice refuses graph capture / replay recording: a CPU step is a host sync point",
        ));
    }
    if awq {
        return Err(cpu_err(
            "moe cpu expert splice refuses AWQ weights: a bare gemv would ignore the per-expert scale",
        ));
    }
    if rows == 0 || top_k == 0 {
        return Err(cpu_err(
            "moe cpu expert splice: rows and top_k must be non-zero",
        ));
    }
    if dim % 256 != 0 || mi % 128 != 0 {
        return Err(cpu_err(&format!(
            "moe cpu expert splice: dim={dim} must be a multiple of 256 and mi={mi} of 128"
        )));
    }
    let slots = rows
        .checked_mul(top_k)
        .ok_or_else(|| cpu_err("moe cpu expert splice: rows × top_k overflows"))?;
    let n_x = rows
        .checked_mul(dim)
        .ok_or_else(|| cpu_err("moe cpu expert splice: rows × dim overflows"))?;
    if x_rot.numel() < n_x || residual.numel() < n_x {
        return Err(cpu_err(&format!(
            "moe cpu expert splice: activation/residual need {n_x} elements"
        )));
    }
    if topk_indices.numel() < slots || topk_weights.numel() < slots {
        return Err(cpu_err(&format!(
            "moe cpu expert splice: top-k scratch needs {slots} elements"
        )));
    }
    // The splice's whole wall time, started after validation and stopped just
    // before the trace print (which is not part of the call's cost).
    let t_total = Instant::now();
    gpu.bind_thread()
        .map_err(|e| cpu_err(&format!("bind: {e}")))?;
    let stage_bytes = n_x
        .checked_add(slots)
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| cpu_err("moe cpu expert splice: transfer extent overflows"))?;
    // Check every span before the first async enqueue: a later HIP size assert
    // must never unwind while a download still writes the reusable stage.
    for (tensor, n) in [
        (x_rot, n_x),
        (topk_indices, slots),
        (topk_weights, slots),
        (residual, n_x),
    ] {
        if tensor.buf.size() < n * 4 {
            return Err(cpu_err("moe cpu expert splice: transfer buffer too small"));
        }
    }
    with_cpu_transfer(gpu, stage_bytes / 4, |transfer| {
        MOE_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let mut timing = MoeStepTiming::default();
            // An event inserted here is after the producers and cannot measure
            // the host wait for them. Trace only: drain before copying to
            // separate readiness from copy/API cost. Controls wait once after
            // enqueueing all four downloads.
            if moe_trace_enabled() {
                let t_wait = Instant::now();
                match gpu.active_stream.as_ref() {
                    Some(stream) => gpu.hip.stream_synchronize(stream),
                    None => gpu.hip.stream_synchronize_default(),
                }
                .map_err(|e| cpu_err(&format!("D2H producer drain: {e}")))?;
                timing.d2h_wait_ns = t_wait.elapsed().as_nanos() as u64;
            }
            let t_enqueue = Instant::now();
            // Enqueue through owned raw extents, not host slices: the previous
            // return may still read this allocation. FIFO ordering prevents
            // these D2H writes from overtaking it on the same stream.
            download_transfer(gpu, transfer, x_rot, 0, n_x)?;
            download_transfer(gpu, transfer, topk_indices, n_x, slots)?;
            download_transfer(gpu, transfer, topk_weights, n_x + slots, slots)?;
            download_transfer(gpu, transfer, residual, n_x + 2 * slots, n_x)?;
            timing.d2h_enqueue_ns = t_enqueue.elapsed().as_nanos() as u64;
            let t_complete = Instant::now();
            transfer
                .complete(&gpu.hip)
                .map_err(|e| cpu_err(&format!("D2H completion: {e}")))?;
            timing.d2h_complete_ns = t_complete.elapsed().as_nanos() as u64;
            let values = transfer_values(transfer, stage_bytes / 4);
            let (x, rest) = values.split_at_mut(n_x);
            let (ti, rest) = rest.split_at_mut(slots);
            let (tw, acc) = rest.split_at_mut(slots);
            // Resolve every selected slot up front: a bad id or a short span is
            // an error before arithmetic. Warm decode/verify resolves into a
            // fixed stack array; the FN 5..64-row lab window resolves into one
            // preallocated stack staging of 640 borrowed slots, written exactly
            // once per slot (no 640-zeroing for row 1, no per-window heap), in
            // physical <=40-slot tiles that reuse the proven numerical sites.
            // Wider non-FN windows retain the heap fallback.
            let t_setup = Instant::now();
            let mut stack: [std::mem::MaybeUninit<CpuMoeExpert<'a>>; MOE_PHYS_SLOTS] =
                [const { std::mem::MaybeUninit::uninit() }; MOE_PHYS_SLOTS];
            let heap: Vec<CpuMoeExpert<'a>>;
            let experts: &[CpuMoeExpert<'a>] = if slots <= MOE_PHYS_TILE {
                for (dst, s) in stack[..slots].iter_mut().zip(0..slots) {
                    dst.write(resolve(expert_id(ti[s]))?);
                }
                // SAFETY: every element 0..slots was just written exactly once.
                unsafe {
                    std::slice::from_raw_parts(stack.as_ptr() as *const CpuMoeExpert<'a>, slots)
                }
            } else if slots <= MOE_PHYS_SLOTS {
                // Tile the resolve over physical <=40-slot passes; the staging
                // lifetime is the whole call, not the tile loop.
                let mut done = 0usize;
                while done < slots {
                    let n = (slots - done).min(MOE_PHYS_TILE);
                    for (dst, s) in stack[done..done + n].iter_mut().zip(done..done + n) {
                        dst.write(resolve(expert_id(ti[s]))?);
                    }
                    done += n;
                }
                // SAFETY: every element 0..slots was written exactly once above.
                unsafe {
                    std::slice::from_raw_parts(stack.as_ptr() as *const CpuMoeExpert<'a>, slots)
                }
            } else {
                heap = (0..slots)
                    .map(|s| resolve(expert_id(ti[s])))
                    .collect::<Result<Vec<_>, _>>()?;
                &heap
            };
            timing.setup_ns = t_setup.elapsed().as_nanos() as u64;
            // Actual GU/DN tier mixture, independent of the first expert label.
            for e in experts.iter() {
                let gi = (e.gate_up_quant as usize).min(MOE_QUANT_MIX_LEN - 1);
                let di = (e.down_quant as usize).min(MOE_QUANT_MIX_LEN - 1);
                timing.gu_mix[gi] = timing.gu_mix[gi].saturating_add(1);
                timing.dn_mix[di] = timing.dn_mix[di].saturating_add(1);
            }
            // The whole CPU side of the splice (gate_up, the SwiGLU/rotate
            // middle, down, combine) runs on the pinned physical-core pool.
            let scratch_ref: &mut MoeScratch = &mut scratch;
            in_splice_pool(|| {
                moe_cpu_experts_host(
                    dim,
                    mi,
                    rows,
                    top_k,
                    experts,
                    x,
                    ti,
                    tw,
                    acc,
                    scratch_ref,
                    &mut timing,
                    bf16_round_trip,
                )
            })?;
            let t_h2d = Instant::now();
            upload_transfer(gpu, transfer, residual, n_x + 2 * slots, n_x)?;
            timing.h2d_ns = t_h2d.elapsed().as_nanos() as u64;
            CPU_STEPS.fetch_add(1, Ordering::Relaxed);
            let mixed = experts.iter().any(|e| {
                e.gate_up_quant != experts[0].gate_up_quant || e.down_quant != experts[0].down_quant
            });
            timing.total_ns = t_total.elapsed().as_nanos() as u64;
            trace_moe_step(
                experts[0].gate_up_quant,
                dim,
                mi,
                top_k,
                rows,
                mixed,
                timing,
            );
            Ok(())
        })
    })
}

/// The splice's arithmetic over already-resolved host slices — split out from
/// [`moe_cpu_experts`] so it is testable without a device. It accumulates into
/// `acc` and records its `gate_up`/SiLU+rotate/down/combine region
/// timings into `timing`'s matching fields (the caller owns the D2H/H2D/total
/// fields).
///
/// Slots are grouped by expert id: one weight, one shared-weight projection,
/// every token that selected it. A weight row's bytes are therefore read once
/// for all of that expert's tokens (row-major, see
/// [`hipfire_cpu::gemv::gemv_shared_sourced`]), and all groups of a projection go
/// to the CPU in one rayon region.
///
/// Every working set is a reusable field of `scratch` — the `f32` buffers and the
/// index metadata — and the grouped window is described to the kernel through
/// [`MoeJobSource`] instead of a materialized job slice, so a warm call performs
/// no heap allocation.
#[allow(clippy::too_many_arguments)]
fn moe_cpu_experts_host(
    dim: usize,
    mi: usize,
    rows: usize,
    top_k: usize,
    experts: &[CpuMoeExpert<'_>],
    x: &[f32],
    ti: &[f32],
    tw: &[f32],
    residual_acc: &mut [f32],
    scratch: &mut MoeScratch,
    timing: &mut MoeStepTiming,
    bf16_round_trip: bool,
) -> Result<(), DispatchError> {
    let slots = rows * top_k;
    if experts.len() != slots {
        return Err(cpu_err(
            "moe cpu expert splice: expert count != rows × top_k",
        ));
    }
    let n_x = rows * dim;
    if x.len() < n_x || residual_acc.len() < n_x || ti.len() < slots || tw.len() < slots {
        return Err(cpu_err("moe cpu expert splice: scratch too small"));
    }
    moe_slots_compute(
        dim,
        mi,
        top_k,
        slots,
        None,
        experts,
        x,
        |s| expert_id(ti[s]) as u32,
        scratch,
        timing,
        bf16_round_trip,
        None,
    )?;
    // The down outputs are row-major per group, so a token's vector is strided:
    // `slot_group`/`slot_si` place each slot in its group's run, and the residual
    // accumulates in the original row-major, rank-minor order.
    let t_combine = Instant::now();
    {
        let dn = &scratch.dn[..dim * slots];
        if bf16_round_trip {
            // GPU `moe_down_combine_top10_batched` order: each row starts from
            // rt(residual) and adds its selected ranks in ascending expert id
            // (ties: lower rank, NOT routing-rank order), each step
            // acc = rt(acc + rt(rt(down) * rt(weight))). The next rank is the
            // smallest (expert id, rank) key strictly after the previous one — a
            // tight O(top_k²) scan that needs no extra buffer.
            for t in 0..rows {
                let accrow = &mut residual_acc[t * dim..(t + 1) * dim];
                for a in accrow.iter_mut() {
                    *a = bf16_rt(*a);
                }
                let mut prev: Option<(usize, usize)> = None;
                for _ in 0..top_k {
                    let mut best: Option<(usize, usize)> = None;
                    for j in 0..top_k {
                        let key = (expert_id(ti[t * top_k + j]), j);
                        if prev.map_or(false, |p| key <= p) {
                            continue;
                        }
                        if best.map_or(true, |b| key < b) {
                            best = Some(key);
                        }
                    }
                    let Some((_, j)) = best else { break };
                    prev = best;
                    let s = t * top_k + j;
                    let gi = scratch.slot_group[s] as usize;
                    let si = scratch.slot_si[s] as usize;
                    let n = group_n(&scratch.group_start, gi);
                    let base = scratch.group_off[gi];
                    let w = bf16_rt(tw[s]);
                    for (r, a) in accrow.iter_mut().enumerate() {
                        let out = bf16_rt(dn[base + r * n + si]);
                        *a = bf16_rt(*a + bf16_rt(out * w));
                    }
                }
            }
        } else {
            for s in 0..slots {
                let t = s / top_k;
                let gi = scratch.slot_group[s] as usize;
                let si = scratch.slot_si[s] as usize;
                let n = group_n(&scratch.group_start, gi);
                let base = scratch.group_off[gi];
                let w = tw[s];
                let accrow = &mut residual_acc[t * dim..(t + 1) * dim];
                for (r, a) in accrow.iter_mut().enumerate() {
                    *a += w * dn[base + r * n + si];
                }
            }
        }
    }
    timing.combine_ns = t_combine.elapsed().as_nanos() as u64;
    Ok(())
}

/// One expert's span contract: both projections' group widths divide the
/// shapes and both retained spans cover their extents.
fn validate_expert_spans(dim: usize, mi: usize, e: &CpuMoeExpert<'_>) -> Result<(), DispatchError> {
    if dim % e.gate_up_quant.k_multiple() != 0 || mi % e.down_quant.k_multiple() != 0 {
        return Err(cpu_err(&format!(
            "moe cpu expert splice: dim={dim} / mi={mi} not a multiple of the expert's gate_up / down group width"
        )));
    }
    let gu_need = expert_gu_bytes(e.gate_up_quant, dim, mi)?;
    let dn_need = expert_down_bytes(e.down_quant, dim, mi)?;
    if e.gate_up.len() < gu_need {
        return Err(cpu_err(&format!(
            "moe cpu expert splice: expert gate_up span {} < {gu_need} bytes",
            e.gate_up.len()
        )));
    }
    if e.down.len() < dn_need {
        return Err(cpu_err(&format!(
            "moe cpu expert splice: expert down span {} < {dn_need} bytes",
            e.down.len()
        )));
    }
    Ok(())
}
/// Fixed-size per-call CPU-arm trace record for the opt-in
/// `HIPFIRE_CPU_SPLIT_TRACE` diagnostic. Stack-owned; crosses the Rayon pool
/// boundary as an owned return, never thread-local state. Inclusive totals
/// (`cpu_total_ns`, `pool_install_ns`, `pool_body_ns`, `middle_ns`, `gu_ns`,
/// `dn_ns`) overlap their disjoint children by design: reconciliation is
/// `child_sum <= inclusive`, except `gu_ns == gu_prep_ns + gu_exec_ns` and
/// `dn_ns == dn_prep_ns + dn_exec_ns` (exact: the traced GEMV measures its
/// whole call as one outer span with `prep = total - exec`).
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct CpuArmTrace {
    /// Same `Instant` origin as the split join's existing trace-entry clock.
    /// Boundary fields are nanosecond offsets, not durations or GPU times.
    trace_epoch: Option<Instant>,
    /// Absent when no host slot reaches the corresponding phase.
    pub middle_start_ns: Option<u64>,
    pub middle_end_ns: Option<u64>,
    pub scatter_start_ns: Option<u64>,
    pub scatter_end_ns: Option<u64>,
    /// Public-entry inclusive total (validation + pool install + return).
    pub cpu_total_ns: u64,
    /// Entry validation alone.
    pub validate_ns: u64,
    /// Pool install total (dispatch + body + return).
    pub pool_install_ns: u64,
    /// In-pool closure body total.
    pub pool_body_ns: u64,
    /// Grouping/scratch prep inside the grouped compute.
    pub group_ns: u64,
    pub gu_ns: u64,
    pub gu_prep_ns: u64,
    pub gu_exec_ns: u64,
    pub middle_ns: u64,
    pub gather_ns: u64,
    pub silu_ns: u64,
    pub fwht_ns: u64,
    pub dn_ns: u64,
    pub dn_prep_ns: u64,
    pub dn_exec_ns: u64,
    /// Output scatter of down values into the expanded layout.
    pub scatter_ns: u64,
    /// Distinct experts from the existing grouping (group count).
    pub unique_experts: u32,
}

impl CpuArmTrace {
    fn offset_ns(&self, stamp: Instant) -> Option<u64> {
        self.trace_epoch.map(|epoch| {
            stamp
                .saturating_duration_since(epoch)
                .as_nanos()
                .min(u64::MAX as u128) as u64
        })
    }
}

fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

/// The gate_up GEMV → SwiGLU → hidden rotation → down GEMV core shared by the
/// full-residual splice ([`moe_cpu_experts_host`]) and the sparse per-slot
/// entry ([`moe_cpu_host_slot_outputs`]).
///
/// `slots` entries are computed; entry `a` uses `experts[a]`, group key
/// `group_key(a)` (entries with equal keys MUST share one expert's bytes) and
/// the activation row `flat(a) / top_k`, where `flat(a)` is `flat_of[a]` or
/// `a` itself when `flat_of` is `None`. On return `scratch` holds the down
/// outputs (`dn`) with `slot_group` / `slot_si` / `group_start` / `group_off`
/// locating entry `a`'s vector: `dn[group_off[g] + r * n(g) + slot_si[a]]`.
/// Every span is validated before the first GEMV; only `scratch` is written.
#[allow(clippy::too_many_arguments)]
fn moe_slots_compute(
    dim: usize,
    mi: usize,
    top_k: usize,
    slots: usize,
    flat_of: Option<&[u32]>,
    experts: &[CpuMoeExpert<'_>],
    x: &[f32],
    group_key: impl Fn(usize) -> u32,
    scratch: &mut MoeScratch,
    timing: &mut MoeStepTiming,
    bf16_round_trip: bool,
    mut trace: Option<&mut CpuArmTrace>,
) -> Result<(), DispatchError> {
    debug_assert_eq!(experts.len(), slots);
    let t_group = trace.is_some().then(Instant::now);
    // Group the selected slots by expert id, in first-seen order, validating each
    // expert's spans as we go. A counting sort into the reusable index vectors
    // keeps each group's members contiguous in `order` without a per-group `Vec`:
    // assign every slot its group index and collect the group ids, then
    // prefix-sum the sizes into `group_start` and place the slots. (`group_off`
    // doubles as the fill cursor here, before it holds output offsets.)
    scratch.group_id.clear();
    scratch.slot_group.resize(slots, 0);
    for s in 0..slots {
        validate_expert_spans(dim, mi, &experts[s])?;
        let eid = group_key(s);
        let gi = match scratch.group_id.iter().position(|&g| g == eid) {
            Some(gi) => gi,
            None => {
                scratch.group_id.push(eid);
                scratch.group_id.len() - 1
            }
        };
        scratch.slot_group[s] = gi as u32;
    }
    let groups = scratch.group_id.len();
    scratch.group_off.clear();
    scratch.group_off.resize(groups, 0);
    for s in 0..slots {
        scratch.group_off[scratch.slot_group[s] as usize] += 1;
    }
    scratch.group_start.clear();
    scratch.group_start.resize(groups + 1, 0);
    let mut acc = 0u32;
    for gi in 0..groups {
        scratch.group_start[gi] = acc;
        acc += scratch.group_off[gi] as u32;
    }
    scratch.group_start[groups] = acc;
    for gi in 0..groups {
        scratch.group_off[gi] = scratch.group_start[gi] as usize;
    }
    scratch.order.resize(slots, 0);
    scratch.slot_si.resize(slots, 0);
    for s in 0..slots {
        let gi = scratch.slot_group[s] as usize;
        let at = scratch.group_off[gi];
        scratch.order[at] = s as u32;
        scratch.slot_si[s] = at as u32 - scratch.group_start[gi];
        scratch.group_off[gi] += 1;
    }
    if let Some(t) = trace.as_mut().map(|t| &mut **t) {
        t.unique_experts = scratch.group_id.len() as u32;
    }

    // gate_up: one shared-weight projection per group, every group in one rayon
    // region. Output is row-major per group (`out[r * n + t]`), so a weight row's
    // bytes are read once for all of that expert's tokens.
    scratch.gu.resize(2 * mi * slots, 0.0);
    scratch.hidden.resize(mi * slots, 0.0);
    scratch.dn.resize(dim * slots, 0.0);
    scratch.group_off.resize(groups + 1, 0);
    fill_group_off(&scratch.group_start, &mut scratch.group_off, 2 * mi);
    if let Some(t) = trace.as_mut().map(|t| &mut **t) {
        t.group_ns = t_group.map_or(0, elapsed_ns);
    }
    let gu_split = {
        let src = MoeJobSource {
            experts,
            order: &scratch.order,
            group_start: &scratch.group_start,
            act: x,
            row_len: dim,
            top_k,
            flat_of,
            proj: MoeProj::GateUp,
        };
        hipfire_cpu::gemv::gemv_shared_sourced_traced(2 * mi, dim, &mut scratch.gu, &src, None)
    };
    timing.gu_ns = gu_split.total_ns;
    timing.gu_prep_ns = gu_split.prep_ns;
    timing.gu_exec_ns = gu_split.exec_ns;
    if let Some(t) = trace.as_mut().map(|t| &mut **t) {
        t.gu_ns = gu_split.total_ns;
        t.gu_prep_ns = gu_split.prep_ns;
        t.gu_exec_ns = gu_split.exec_ns;
    }
    // SwiGLU + hidden rotation, per token. The gate/up pair sits at strided
    // indices in the row-major output, so gather it; the rotation is the *down*
    // projection's contract (its weights are stored post-rotation).
    let t_middle = Instant::now();
    if let Some(t) = trace.as_mut().map(|t| &mut **t) {
        t.middle_start_ns = t.offset_ns(t_middle);
    }
    // Slot-parallel: every slot's gather -> SiLU -> BF16 -> FWHT touches only its
    // own `hidden` row and reads `gu` (written before this region), so the slots
    // are independent and the result is byte-identical to the serial loop. The
    // sub-phase counters are summed worker time (they can exceed `middle_ns`).
    let gather_ns = AtomicU64::new(0);
    let silu_ns = AtomicU64::new(0);
    let fwht_ns = AtomicU64::new(0);
    {
        let gu = &scratch.gu[..2 * mi * slots];
        let (order, group_start, group_off) =
            (&scratch.order, &scratch.group_start, &scratch.group_off);
        let (slot_group, slot_si) = (&scratch.slot_group, &scratch.slot_si);
        hipfire_cpu::splice_pool::par_for_each_chunk_mut(
            &mut scratch.hidden[..mi * slots],
            mi,
            |s, hid| {
                let gi = slot_group[s] as usize;
                let si = slot_si[s] as usize;
                let n = group_n(group_start, gi);
                let base = group_off[gi];
                let first = order[group_start[gi] as usize] as usize;
                let down_rotation = experts[first].down_quant.input_rotation();
                // Gate/up are gathered into a stack buffer one `MIDDLE_CHUNK`-wide
                // column block at a time (a whole slot for every real shape);
                // SiLU is elementwise, so chunking changes no bit.
                let mut buf = [0.0f32; 2 * MIDDLE_CHUNK];
                let (mut gather_t, mut silu_t) = (0u64, 0u64);
                let mut j0 = 0;
                while j0 < mi {
                    let c = (mi - j0).min(MIDDLE_CHUNK);
                    let t_gather = Instant::now();
                    for j in 0..c {
                        let (g, u) = (
                            gu[base + (j0 + j) * n + si],
                            gu[base + (mi + j0 + j) * n + si],
                        );
                        buf[j] = if bf16_round_trip { bf16_rt(g) } else { g };
                        buf[c + j] = if bf16_round_trip { bf16_rt(u) } else { u };
                    }
                    gather_t += elapsed_ns(t_gather);
                    let t_silu = Instant::now();
                    silu_mul(&buf[..2 * c], &mut hid[j0..j0 + c]);
                    silu_t += elapsed_ns(t_silu);
                    j0 += c;
                }
                let t_silu = Instant::now();
                if bf16_round_trip {
                    // The activated hidden is stored BF16 before the FWHT; the
                    // rotated output below stays f32.
                    for h in hid.iter_mut() {
                        *h = bf16_rt(*h);
                    }
                }
                gather_ns.fetch_add(gather_t, Ordering::Relaxed);
                silu_ns.fetch_add(silu_t + elapsed_ns(t_silu), Ordering::Relaxed);
                if down_rotation != InputRotation::None {
                    let t_fwht = Instant::now();
                    match down_rotation {
                        InputRotation::FwhtG128V2 => rotate_x_128_v2(hid),
                        _ => rotate_x(hid),
                    }
                    fwht_ns.fetch_add(elapsed_ns(t_fwht), Ordering::Relaxed);
                }
            },
        );
    }
    let (gather_ns, silu_ns, fwht_ns) = (
        gather_ns.into_inner(),
        silu_ns.into_inner(),
        fwht_ns.into_inner(),
    );
    if let Some(t) = trace.as_mut().map(|t| &mut **t) {
        t.middle_end_ns = t.offset_ns(Instant::now());
    }
    timing.middle_ns = t_middle.elapsed().as_nanos() as u64;
    timing.middle_gather_ns = gather_ns;
    timing.middle_silu_ns = silu_ns;
    timing.middle_fwht_ns = fwht_ns;
    if let Some(t) = trace.as_mut().map(|t| &mut **t) {
        t.middle_ns = timing.middle_ns;
        t.gather_ns = gather_ns;
        t.silu_ns = silu_ns;
        t.fwht_ns = fwht_ns;
    }
    scratch.group_off.resize(groups + 1, 0);
    fill_group_off(&scratch.group_start, &mut scratch.group_off, dim);
    let dn_split = {
        let src = MoeJobSource {
            experts,
            order: &scratch.order,
            group_start: &scratch.group_start,
            act: &scratch.hidden[..mi * slots],
            row_len: mi,
            top_k,
            flat_of,
            proj: MoeProj::Down,
        };
        hipfire_cpu::gemv::gemv_shared_sourced_traced(dim, mi, &mut scratch.dn, &src, None)
    };
    timing.dn_ns = dn_split.total_ns;
    timing.dn_prep_ns = dn_split.prep_ns;
    timing.dn_exec_ns = dn_split.exec_ns;
    if let Some(t) = trace.as_mut().map(|t| &mut **t) {
        t.dn_ns = dn_split.total_ns;
        t.dn_prep_ns = dn_split.prep_ns;
        t.dn_exec_ns = dn_split.exec_ns;
    }
    Ok(())
}

/// Validated `(rows × top_k, rows × dim, rows × top_k × dim)` of a host-slot call.
fn host_slot_extents(
    dim: usize,
    mi: usize,
    rows: usize,
    top_k: usize,
) -> Result<(usize, usize, usize), DispatchError> {
    if rows == 0 || top_k == 0 {
        return Err(cpu_err(
            "moe cpu host slots: rows and top_k must be non-zero",
        ));
    }
    if dim % 256 != 0 || mi % 128 != 0 {
        return Err(cpu_err(&format!(
            "moe cpu host slots: dim={dim} must be a multiple of 256 and mi={mi} of 128"
        )));
    }
    let slots = rows
        .checked_mul(top_k)
        .ok_or_else(|| cpu_err("moe cpu host slots: rows × top_k overflows"))?;
    let n_x = rows
        .checked_mul(dim)
        .ok_or_else(|| cpu_err("moe cpu host slots: rows × dim overflows"))?;
    let n_out = slots
        .checked_mul(dim)
        .ok_or_else(|| cpu_err("moe cpu host slots: rows × top_k × dim overflows"))?;
    Ok((slots, n_x, n_out))
}

/// `true` when `id` names a real expert: the router's `-1` sentinel and any
/// other out-of-range id are unrouted.
#[inline]
fn route_id_valid(id: i32, n_experts: usize) -> bool {
    id >= 0 && (id as usize) < n_experts
}

/// Expert ids kept on the stack by [`ExpertReps`] before it heap-allocates.
const EXPERT_REP_STACK: usize = 512;

fn expert_spans_differ(o: &CpuMoeExpert<'_>, e: &CpuMoeExpert<'_>) -> bool {
    o.gate_up_quant != e.gate_up_quant
        || o.down_quant != e.down_quant
        || o.gate_up.as_ptr() != e.gate_up.as_ptr()
        || o.gate_up.len() != e.gate_up.len()
        || o.down.as_ptr() != e.down.as_ptr()
        || o.down.len() != e.down.len()
}

/// Per-expert first-seen representative (index into `host_slots`/`experts`).
/// Same-span equality is an equivalence relation and every earlier same-id
/// slot already matched the representative (or the call errored), so comparing
/// against the first occurrence alone accepts/rejects exactly what the former
/// pairwise scan did and names the same `other` slot. O(1) per slot.
struct ExpertReps {
    stack: [u32; EXPERT_REP_STACK],
    heap: Vec<u32>,
    n: usize,
}

impl ExpertReps {
    const NONE: u32 = u32::MAX;

    fn new(n_experts: usize) -> Self {
        let heap = if n_experts > EXPERT_REP_STACK {
            vec![Self::NONE; n_experts]
        } else {
            Vec::new()
        };
        Self {
            stack: [Self::NONE; EXPERT_REP_STACK],
            heap,
            n: n_experts,
        }
    }

    /// Record slot position `i` for `id`; return the position
    /// of the first earlier same-id slot whose spans differ, if any. `id` MUST
    /// be `< n_experts`.
    fn conflict(&mut self, id: usize, i: usize, experts: &[CpuMoeExpert<'_>]) -> Option<usize> {
        debug_assert!(id < self.n);
        let cell = if self.n > EXPERT_REP_STACK {
            &mut self.heap[id]
        } else {
            &mut self.stack[id]
        };
        if *cell == Self::NONE {
            *cell = i as u32;
            return None;
        }
        let r = *cell as usize;
        if expert_spans_differ(&experts[r], &experts[i]) {
            Some(r)
        } else {
            None
        }
    }
}

/// Slot-ownership bitmask words kept on the stack (1024 slots) before a heap
/// fallback.
const HOST_SLOT_MASK_STACK_WORDS: usize = 16;

/// Validate a sparse host-slot plan against the original route, with no
/// arithmetic and no allocation for up to 1024 slots. This is the exact
/// precondition [`moe_cpu_host_slot_outputs`] enforces first, so a caller can
/// run it at prepare time (before any GPU tail is enqueued) and know the later
/// arithmetic call cannot fail on the plan.
///
/// * `ids`: the ORIGINAL router ids, `[rows * top_k]` i32 (at least that many).
///   An id outside `0..n_experts` (including `-1`) is an invalid slot.
/// * `gpu_ids`: the masked GPU route, when the caller has it host-side. `Some`
///   verifies the ownership partition: a host slot is exactly `-1` there, a
///   valid non-host slot is unchanged (`== ids[s]`), an invalid slot is owned
///   by neither arm (host-unlisted and invalid in `gpu_ids`).
/// * `host_slots`: flat slots `row * top_k + rank`, any order, no duplicates,
///   each with a valid id.
/// * `experts[i]`: the retained spans of `host_slots[i]`. Slots naming the same
///   expert id MUST carry the identical spans (same pointers, lengths and
///   quants), so grouping by id can never mix weights.
/// * Every span's group width and extent is checked; `awq` is refused.
/// * `x_len` / `out_len`: the element counts of the rotated activation and the
///   output extent the later arithmetic call will receive; they must cover
///   `rows * dim` and `rows * top_k * dim`.
#[allow(clippy::too_many_arguments)]
pub fn validate_moe_cpu_host_slot_outputs(
    dim: usize,
    mi: usize,
    rows: usize,
    top_k: usize,
    n_experts: usize,
    ids: &[i32],
    gpu_ids: Option<&[i32]>,
    host_slots: &[u32],
    experts: &[CpuMoeExpert<'_>],
    x_len: usize,
    awq: bool,
    out_len: usize,
) -> Result<(), DispatchError> {
    if awq {
        return Err(cpu_err(
            "moe cpu host slots refuse AWQ weights: a bare gemv would ignore the per-expert scale",
        ));
    }
    let (slots, n_x, n_out) = host_slot_extents(dim, mi, rows, top_k)?;
    if x_len < n_x || out_len < n_out {
        return Err(cpu_err(&format!(
            "moe cpu host slots: activation needs {n_x} and output {n_out} elements"
        )));
    }
    if ids.len() < slots || gpu_ids.is_some_and(|g| g.len() < slots) {
        return Err(cpu_err(&format!(
            "moe cpu host slots: route ids need {slots} elements"
        )));
    }
    if experts.len() != host_slots.len() {
        return Err(cpu_err("moe cpu host slots: span count != host slot count"));
    }
    let words = slots.div_ceil(64);
    let mut stack = [0u64; HOST_SLOT_MASK_STACK_WORDS];
    let mut heap: Vec<u64>;
    let owned: &mut [u64] = if words <= HOST_SLOT_MASK_STACK_WORDS {
        &mut stack[..words]
    } else {
        heap = vec![0u64; words];
        &mut heap
    };
    let mut reps = ExpertReps::new(n_experts);
    for (i, (&slot, e)) in host_slots.iter().zip(experts).enumerate() {
        let s = slot as usize;
        if s >= slots {
            return Err(cpu_err(&format!(
                "moe cpu host slots: host slot {s} out of range (rows × top_k = {slots})"
            )));
        }
        if owned[s / 64] >> (s % 64) & 1 != 0 {
            return Err(cpu_err(&format!(
                "moe cpu host slots: host slot {s} listed twice"
            )));
        }
        owned[s / 64] |= 1 << (s % 64);
        let id = ids[s];
        if !route_id_valid(id, n_experts) {
            return Err(cpu_err(&format!(
                "moe cpu host slots: host slot {s} has invalid expert id {id} (n_experts={n_experts})"
            )));
        }
        validate_expert_spans(dim, mi, e)?;
        if let Some(other) = reps.conflict(id as usize, i, experts) {
            let other = host_slots[other];
            return Err(cpu_err(&format!(
                "moe cpu host slots: expert id {id} resolved to different spans (host slots {other} and {s})"
            )));
        }
    }
    if let Some(g) = gpu_ids {
        for s in 0..slots {
            let host = owned[s / 64] >> (s % 64) & 1 != 0;
            let valid = route_id_valid(ids[s], n_experts);
            let ok = if host {
                g[s] == -1
            } else if valid {
                g[s] == ids[s]
            } else {
                !route_id_valid(g[s], n_experts)
            };
            if !ok {
                return Err(cpu_err(&format!(
                    "moe cpu host slots: slot {s} ownership mismatch (original id {}, gpu id {}, host={host})",
                    ids[s], g[s]
                )));
            }
        }
    }
    Ok(())
}

fn reserve_to<T>(v: &mut Vec<T>, n: usize) {
    if v.capacity() < n {
        v.reserve(n - v.len());
    }
}

/// Pre-grow the calling thread's host-slot scratch for up to `max_slots`
/// computed slots so a later [`moe_cpu_host_slot_outputs`] on the SAME thread
/// grows nothing. (The GEMV's own per-thread activation-sum cache lives in
/// `hipfire_cpu` and is not reserved here.) The D2 executor prewarms this once
/// per layer at `prepare` time for its full allocated capacity (64 rows × 10 =
/// 640 slots), so no logical 1..64 window regrows it on the hot path.
pub fn reserve_moe_cpu_host_slot_scratch(
    dim: usize,
    mi: usize,
    max_slots: usize,
) -> Result<(), DispatchError> {
    let ovf = || cpu_err("moe cpu host slots: scratch extent overflows");
    let gu = max_slots.checked_mul(2 * mi).ok_or_else(ovf)?;
    let hid = max_slots.checked_mul(mi).ok_or_else(ovf)?;
    let dn = max_slots.checked_mul(dim).ok_or_else(ovf)?;
    MOE_SCRATCH.with(|cell| {
        let mut s = cell.borrow_mut();
        reserve_to(&mut s.gu, gu);
        reserve_to(&mut s.hidden, hid);
        reserve_to(&mut s.dn, dn);
        reserve_to(&mut s.order, max_slots);
        reserve_to(&mut s.group_start, max_slots + 1);
        reserve_to(&mut s.group_id, max_slots);
        reserve_to(&mut s.slot_group, max_slots);
        reserve_to(&mut s.slot_si, max_slots);
        reserve_to(&mut s.group_off, max_slots + 1);
    });
    Ok(())
}

/// Real host-slot scratch capacity (logical ROWS) of THIS thread for geometry
/// `(dim, mi, top_k)`: `min_slots / top_k` where `min_slots` is the minimum
/// over every reserved `MOE_SCRATCH` vector of `capacity /
/// per-slot-component-stride` — i.e. the largest row count the current
/// reservation can serve without any `Vec` growth on the hot path. Units are
/// rows, not slots: a 640-slot reserve at top_k 10 reports 64, and the legacy
/// 40-slot reserve reports 4. `moe_cpu_host_slot_outputs` needs per computed
/// slot: `2*mi` gu + `mi` hidden + `dim` dn elements, one
/// `order`/`group_id`/`slot_group`/`slot_si` entry each, one
/// `group_start`/`group_off` entry each (the `+1` trailing offset is covered
/// by the same reservation).
/// Pure reader: no allocation, no growth, no mutation; `0` when `dim/mi/top_k`
/// is zero, when any stride divides nothing, when `top_k` exceeds the slot
/// minimum, or when the thread holds no reservation yet; `usize::MAX` is never
/// returned (every quotient is bounded by a real `Vec::capacity`). Callers
/// MUST NOT clamp a slots-scale value into this rows-scale result.
pub fn moe_cpu_host_slot_scratch_capacity_rows(dim: usize, mi: usize, top_k: usize) -> usize {
    if dim == 0 || mi == 0 || top_k == 0 {
        return 0;
    }
    let stride_2mi = match (2usize).checked_mul(mi) {
        Some(v) if v > 0 => v,
        _ => return 0,
    };
    MOE_SCRATCH.with(|cell| {
        let s = cell.borrow();
        let mut slots = usize::MAX;
        // f32 arithmetic buffers: capacity elements / per-slot elements.
        slots = slots.min(s.gu.capacity() / stride_2mi);
        slots = slots.min(s.hidden.capacity() / mi);
        slots = slots.min(s.dn.capacity() / dim);
        // Per-slot index metadata: one entry per computed slot.
        slots = slots.min(s.order.capacity());
        slots = slots.min(s.group_id.capacity());
        slots = slots.min(s.slot_group.capacity());
        slots = slots.min(s.slot_si.capacity());
        // group_start/group_off reserve max_slots+1 entries (groups+1 offsets);
        // the trailing offset consumes one, the rest cover one group each, and
        // one group always covers >=1 slot, so capacity-1 bounds the slots.
        slots = slots.min(s.group_start.capacity().saturating_sub(1));
        slots = slots.min(s.group_off.capacity().saturating_sub(1));
        if slots == usize::MAX {
            0
        } else {
            slots / top_k
        }
    })
}

/// Host-only per-slot expert outputs for a split (CPU cold / GPU hot) MoE
/// layer. Validates the whole plan first ([`validate_moe_cpu_host_slot_outputs`], same
/// arguments), then computes, for each listed host slot, the expert's
/// UNWEIGHTED down-projection output on the pinned splice pool:
/// gate_up GEMV → `rt(g)`, `rt(u)` → SiLU·mul → `rt(hidden)` → down-format
/// FWHT → down GEMV, i.e. D1's `bf16_round_trip=true` arithmetic up to the
/// down output.
///
/// `x_rot` is the rotated F32 activation `[rows, dim]` (rows without host
/// slots are never read). `out` is the GPU expanded layout `[rows, top_k, dim]`:
/// host slot `s` receives RAW F32 down values at `out[s * dim .. (s + 1) * dim]`
/// (no BF16 rounding of the output — the canonical combine applies it), exactly
/// the representation `gemv_*_moe_down_*_indexed_batched_expanded` stores.
/// Every other `out` element is neither read nor written, so the caller may
/// leave stale data (even NaN) there. No weights, residual or GPU memory is
/// touched; no router weight is applied.
///
/// Each element is the same per-element dot the single-row GEMV computes, so
/// the bytes of a slot depend only on its expert and its activation row — not
/// on `host_slots` order, which slots share an expert, or the worker count
/// (under one CPU ISA policy).
#[allow(clippy::too_many_arguments)]
pub fn moe_cpu_host_slot_outputs(
    dim: usize,
    mi: usize,
    rows: usize,
    top_k: usize,
    n_experts: usize,
    ids: &[i32],
    gpu_ids: Option<&[i32]>,
    host_slots: &[u32],
    experts: &[CpuMoeExpert<'_>],
    x_rot: &[f32],
    awq: bool,
    out: &mut [f32],
) -> Result<(), DispatchError> {
    validate_moe_cpu_host_slot_outputs(
        dim,
        mi,
        rows,
        top_k,
        n_experts,
        ids,
        gpu_ids,
        host_slots,
        experts,
        x_rot.len(),
        awq,
        out.len(),
    )?;
    if host_slots.is_empty() {
        return Ok(());
    }
    MOE_SCRATCH.with(|cell| {
        let mut guard = cell.borrow_mut();
        let scratch: &mut MoeScratch = &mut guard;
        in_splice_pool(|| {
            host_slot_outputs_core(
                dim, mi, top_k, ids, host_slots, experts, x_rot, scratch, out, None,
            )
        })
    })
}
/// Crate-private timed entry sharing the checked implementation of
/// [`moe_cpu_host_slot_outputs`]: same validation, same core, same pool.
/// Normal callers keep the public API; the CPU-split trace calls this so the
/// fixed-size [`CpuArmTrace`] crosses the Rayon install as an owned return,
/// never thread-local state. No allocation, no output; timing only.
/// `trace_epoch` is the caller's existing join-entry `Instant`; all optional
/// phase boundaries use nanosecond offsets from that same origin. An empty
/// host set leaves boundaries absent rather than manufacturing zero stamps.
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_cpu_host_slot_outputs_traced(
    dim: usize,
    mi: usize,
    rows: usize,
    top_k: usize,
    n_experts: usize,
    ids: &[i32],
    gpu_ids: Option<&[i32]>,
    host_slots: &[u32],
    experts: &[CpuMoeExpert<'_>],
    x_rot: &[f32],
    awq: bool,
    out: &mut [f32],
    trace_epoch: Instant,
) -> Result<CpuArmTrace, DispatchError> {
    let mut trace = CpuArmTrace {
        trace_epoch: Some(trace_epoch),
        ..CpuArmTrace::default()
    };
    let t_total = Instant::now();
    let t_validate = Instant::now();
    validate_moe_cpu_host_slot_outputs(
        dim,
        mi,
        rows,
        top_k,
        n_experts,
        ids,
        gpu_ids,
        host_slots,
        experts,
        x_rot.len(),
        awq,
        out.len(),
    )?;
    trace.validate_ns = elapsed_ns(t_validate);
    if host_slots.is_empty() {
        trace.cpu_total_ns = elapsed_ns(t_total);
        return Ok(trace);
    }
    let t_install = Instant::now();
    let result = MOE_SCRATCH.with(|cell| {
        let mut guard = cell.borrow_mut();
        let scratch: &mut MoeScratch = &mut guard;
        // Owned fixed-size record across the Rayon install: the pool body
        // timer runs INSIDE the install closure, so dispatch/return overhead
        // stays in `pool_install_ns`, not `pool_body_ns`.
        in_splice_pool(|| {
            let t_body = Instant::now();
            let result = host_slot_outputs_core(
                dim,
                mi,
                top_k,
                ids,
                host_slots,
                experts,
                x_rot,
                scratch,
                out,
                Some(&mut trace),
            );
            let body_ns = elapsed_ns(t_body);
            (result, body_ns, trace)
        })
    });
    let (result, body_ns, returned) = result;
    trace = returned;
    trace.pool_install_ns = elapsed_ns(t_install);
    trace.pool_body_ns = body_ns;
    result?;
    trace.cpu_total_ns = elapsed_ns(t_total);
    Ok(trace)
}

/// The validated arithmetic of [`moe_cpu_host_slot_outputs`], run in whatever
/// rayon pool is current. Split out so tests can pin the worker count.
#[allow(clippy::too_many_arguments)]
fn host_slot_outputs_core(
    dim: usize,
    mi: usize,
    top_k: usize,
    ids: &[i32],
    host_slots: &[u32],
    experts: &[CpuMoeExpert<'_>],
    x_rot: &[f32],
    scratch: &mut MoeScratch,
    out: &mut [f32],
    mut trace: Option<&mut CpuArmTrace>,
) -> Result<(), DispatchError> {
    let mut timing = MoeStepTiming::default();
    moe_slots_compute(
        dim,
        mi,
        top_k,
        host_slots.len(),
        Some(host_slots),
        experts,
        x_rot,
        |a| ids[host_slots[a] as usize] as u32,
        scratch,
        &mut timing,
        true,
        trace.as_mut().map(|t| &mut **t),
    )?;
    let t_scatter = trace.is_some().then(Instant::now);
    if let (Some(t), Some(s0)) = (trace.as_mut(), t_scatter) {
        t.scatter_start_ns = t.offset_ns(s0);
    }
    for (a, &slot) in host_slots.iter().enumerate() {
        let gi = scratch.slot_group[a] as usize;
        let si = scratch.slot_si[a] as usize;
        let n = group_n(&scratch.group_start, gi);
        let base = scratch.group_off[gi];
        let dst = &mut out[slot as usize * dim..(slot as usize + 1) * dim];
        for (r, o) in dst.iter_mut().enumerate() {
            *o = scratch.dn[base + r * n + si];
        }
    }
    if let Some(t) = trace.as_mut() {
        t.scatter_end_ns = t.offset_ns(Instant::now());
    }
    if let (Some(t), Some(s0)) = (trace.as_mut(), t_scatter) {
        t.scatter_ns = elapsed_ns(s0);
    }
    Ok(())
}

/// Reusable host workspace of [`moe_cpu_grouped_rows_bf16`] (the SP short-prompt
/// prefill's CPU arm). It owns its own [`MoeScratch`] — never the `MOE_SCRATCH`
/// thread-local the D2 decode executors reserve and read capacity from — so SP
/// windows cannot perturb D2's scratch capacity readings. Memory is bounded by
/// the chunk size, not by the window: at hidden 2560 / mi 640 it is
/// `17,920 B` per chunk slot (`2*mi` gate_up + `mi` hidden + `dim` down f32).
/// Grow-only; one owner (the SP runtime), one call at a time.
#[derive(Default)]
pub struct SpCpuScratch {
    moe: MoeScratch,
    /// Activation row of each chunk slot (`flat_of`, with `top_k = 1`).
    flat: Vec<u32>,
    /// Group key of each chunk slot: its index in the caller's expert list.
    key: Vec<u32>,
    /// Output slot index of each chunk slot.
    out_idx: Vec<u32>,
    /// Output-slot ownership bitmap of the validation pass.
    seen: Vec<u64>,
}

impl SpCpuScratch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-grow for chunks of up to `max_chunk_slots` so a later call with
    /// `max_chunk_slots <= ` this value grows no arithmetic buffer.
    pub fn reserve(
        &mut self,
        dim: usize,
        mi: usize,
        max_chunk_slots: usize,
    ) -> Result<(), DispatchError> {
        let ovf = || cpu_err("moe cpu grouped rows: scratch extent overflows");
        let gu = max_chunk_slots.checked_mul(2 * mi).ok_or_else(ovf)?;
        let hid = max_chunk_slots.checked_mul(mi).ok_or_else(ovf)?;
        let dn = max_chunk_slots.checked_mul(dim).ok_or_else(ovf)?;
        let s = &mut self.moe;
        reserve_to(&mut s.gu, gu);
        reserve_to(&mut s.hidden, hid);
        reserve_to(&mut s.dn, dn);
        reserve_to(&mut s.order, max_chunk_slots);
        reserve_to(&mut s.group_start, max_chunk_slots + 1);
        reserve_to(&mut s.group_id, max_chunk_slots);
        reserve_to(&mut s.slot_group, max_chunk_slots);
        reserve_to(&mut s.slot_si, max_chunk_slots);
        reserve_to(&mut s.group_off, max_chunk_slots + 1);
        reserve_to(&mut self.flat, max_chunk_slots);
        reserve_to(&mut self.key, max_chunk_slots);
        reserve_to(&mut self.out_idx, max_chunk_slots);
        Ok(())
    }
}

/// One CPU-owned expert of an SP layer: its host byte spans plus every output
/// the expert produces.
///
/// `rows[i] = (token row, out index)`: activation row `token row` of `x_rot` is
/// run through this expert and its BF16 down output lands at
/// `out_bf16[out index * dim ..][..dim]`. Every out index is written by exactly
/// one entry of exactly one expert across the whole call.
#[derive(Clone, Copy)]
pub struct SpCpuExpert<'a> {
    pub spans: CpuMoeExpert<'a>,
    pub rows: &'a [(u32, u32)],
}

/// What one [`moe_cpu_grouped_rows_bf16`] call did. Times are wall ns of the
/// calling thread; `compute_ns` is the summed `moe_slots_compute` time and
/// `scatter_ns` the summed BF16 scatter time over all chunks.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpCpuStats {
    /// Experts with at least one row.
    pub experts: usize,
    /// Output rows written.
    pub slots: usize,
    /// `moe_slots_compute` calls made.
    pub chunks: usize,
    pub validate_ns: u64,
    pub compute_ns: u64,
    pub scatter_ns: u64,
}

/// BF16 bits of `x`, round-to-nearest-even.
///
/// Finite values and ±Inf are bit-equal to the kernel contract
/// (`slab_moe::slowlink_selftest::bf`/`rbf`, and [`bf16_rt`]): add `0x7fff`
/// plus the kept LSB and keep the high half, so a finite value that rounds
/// past the BF16 maximum becomes ±Inf and signed zero keeps its sign. NaN
/// becomes a quiet NaN with the sign and the high payload bits kept: the
/// carry of the rounding add must never turn a low-payload NaN into ±Inf.
#[inline]
pub fn bf16_bits_rne(x: f32) -> u16 {
    let u = x.to_bits();
    if x.is_nan() {
        return ((u >> 16) as u16) | 0x0040;
    }
    (u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

/// Raw destination pointer shared by the scatter workers. Soundness: the
/// validation pass proved every out index unique and in range, so no two
/// workers' `dim`-long destination rows overlap.
struct SpOutPtr(*mut u16);

impl SpOutPtr {
    #[inline]
    fn get(&self) -> *mut u16 {
        self.0
    }
}

// SAFETY: see the type's doc comment; the pointee is only reached through
// disjoint `dim`-long ranges while the owning `&mut [u16]` is borrowed.
unsafe impl Send for SpOutPtr {}
unsafe impl Sync for SpOutPtr {}

/// O(slots + experts) plan check of [`moe_cpu_grouped_rows_bf16`]: each
/// expert's spans are checked once (not once per slot, and never pairwise),
/// every row index is in `x_rot`, and every out index is in range and claimed
/// by exactly one entry (a bitmap, one bit per output row). Returns
/// `(experts with rows, total slots)`. Reads no weight bytes and writes only
/// `seen`.
fn sp_rows_validate(
    dim: usize,
    mi: usize,
    x_rows: usize,
    out_slots: usize,
    experts: &[SpCpuExpert<'_>],
    seen: &mut Vec<u64>,
) -> Result<(usize, usize), DispatchError> {
    seen.clear();
    seen.resize(out_slots.div_ceil(64), 0);
    let (mut live, mut slots) = (0usize, 0usize);
    for e in experts {
        if e.rows.is_empty() {
            continue;
        }
        validate_expert_spans(dim, mi, &e.spans)?;
        live += 1;
        slots = slots
            .checked_add(e.rows.len())
            .filter(|&s| s <= u32::MAX as usize)
            .ok_or_else(|| cpu_err("moe cpu grouped rows: slot count overflows u32"))?;
        for &(row, out) in e.rows {
            if row as usize >= x_rows {
                return Err(cpu_err(&format!(
                    "moe cpu grouped rows: token row {row} out of range (x has {x_rows} rows)"
                )));
            }
            let out = out as usize;
            if out >= out_slots {
                return Err(cpu_err(&format!(
                    "moe cpu grouped rows: out index {out} out of range (out has {out_slots} rows)"
                )));
            }
            let (w, b) = (out / 64, 1u64 << (out % 64));
            if seen[w] & b != 0 {
                return Err(cpu_err(&format!(
                    "moe cpu grouped rows: out index {out} assigned twice"
                )));
            }
            seen[w] |= b;
        }
    }
    Ok((live, slots))
}

/// SP CPU arm: for every listed expert and `(token row, out index)` pair,
/// compute the expert's UNWEIGHTED down output for that activation row and
/// store it BF16 round-to-nearest-even at `out_bf16[out index * dim ..][..dim]`.
///
/// The arithmetic is exactly [`moe_cpu_host_slot_outputs`]'s
/// (`moe_slots_compute(.., bf16_round_trip = true, ..)`: gate_up GEMV → rt(g),
/// rt(u) → SiLU·mul → rt(hidden) → down-format FWHT → down GEMV); only the
/// representation differs: the result is rounded with [`bf16_bits_rne`] into a
/// compact `[slots, dim]` u16 table the caller addresses by out index, instead
/// of an F32 `[rows, top_k, dim]` expanded layout. Each expert row's bytes
/// therefore depend only on (expert, activation row): not on the chunking, the
/// order of experts or rows, or the worker count.
///
/// `x_rot` is the rotated F32 activation `[x_rows, dim]` (rows nobody names are
/// never read). `max_chunk_slots` (≥ 1) bounds the slots one
/// `moe_slots_compute` call holds, and with it the scratch: chunks are
/// expert-aligned (whole experts are packed greedily; an expert with more rows
/// than the cap is split on its own). Every plan defect is an error before any
/// output row is written; out rows no expert names are neither read nor
/// written. Runs on the splice pool; the whole call is one `install`.
pub fn moe_cpu_grouped_rows_bf16(
    dim: usize,
    mi: usize,
    x_rot: &[f32],
    experts: &[SpCpuExpert<'_>],
    max_chunk_slots: usize,
    scratch: &mut SpCpuScratch,
    out_bf16: &mut [u16],
) -> Result<SpCpuStats, DispatchError> {
    sp_rows_entry(
        dim,
        mi,
        x_rot,
        experts,
        max_chunk_slots,
        scratch,
        out_bf16,
        |run| in_splice_pool(run),
    )
}

/// The public entry with the pool abstracted: validation first (O(slots +
/// experts), no pool, no output write), then `run_in_pool` executes the
/// arithmetic in whatever rayon pool it chooses. Lets tests pin the worker
/// count while sharing every other line with the public path.
#[allow(clippy::too_many_arguments)]
fn sp_rows_entry(
    dim: usize,
    mi: usize,
    x_rot: &[f32],
    experts: &[SpCpuExpert<'_>],
    max_chunk_slots: usize,
    scratch: &mut SpCpuScratch,
    out_bf16: &mut [u16],
    run_in_pool: impl FnOnce(
        &mut (dyn FnMut() -> Result<(), DispatchError> + Send),
    ) -> Result<(), DispatchError>,
) -> Result<SpCpuStats, DispatchError> {
    if dim == 0 || mi == 0 || max_chunk_slots == 0 {
        return Err(cpu_err(
            "moe cpu grouped rows: dim, mi and max_chunk_slots must be non-zero",
        ));
    }
    if x_rot.len() % dim != 0 {
        return Err(cpu_err(&format!(
            "moe cpu grouped rows: x length {} is not a multiple of dim {dim}",
            x_rot.len()
        )));
    }
    let t_validate = Instant::now();
    let (live, slots) = sp_rows_validate(
        dim,
        mi,
        x_rot.len() / dim,
        out_bf16.len() / dim,
        experts,
        &mut scratch.seen,
    )?;
    let mut stats = SpCpuStats {
        experts: live,
        slots,
        validate_ns: elapsed_ns(t_validate),
        ..SpCpuStats::default()
    };
    if slots == 0 {
        return Ok(stats);
    }
    let mut body = || {
        sp_rows_core(
            dim,
            mi,
            x_rot,
            experts,
            max_chunk_slots,
            scratch,
            out_bf16,
            &mut stats,
        )
    };
    run_in_pool(&mut body)?;
    Ok(stats)
}

/// The validated arithmetic of [`moe_cpu_grouped_rows_bf16`], run in whatever
/// rayon pool is current.
#[allow(clippy::too_many_arguments)]
fn sp_rows_core(
    dim: usize,
    mi: usize,
    x_rot: &[f32],
    experts: &[SpCpuExpert<'_>],
    max_chunk_slots: usize,
    scratch: &mut SpCpuScratch,
    out_bf16: &mut [u16],
    stats: &mut SpCpuStats,
) -> Result<(), DispatchError> {
    let cap = max_chunk_slots.min(stats.slots).max(1);
    let SpCpuScratch {
        moe,
        flat,
        key,
        out_idx,
        ..
    } = scratch;
    let mut spans: Vec<CpuMoeExpert<'_>> = Vec::with_capacity(cap);
    let out_ptr = SpOutPtr(out_bf16.as_mut_ptr());
    let (mut ei, mut off) = (0usize, 0usize);
    while ei < experts.len() {
        spans.clear();
        flat.clear();
        key.clear();
        out_idx.clear();
        while ei < experts.len() {
            let e = &experts[ei];
            let remaining = e.rows.len() - off;
            if remaining == 0 {
                ei += 1;
                off = 0;
                continue;
            }
            let room = cap - spans.len();
            // Expert-aligned: an expert that does not fit the open chunk starts
            // the next one (and is split there only if it alone exceeds `cap`).
            if !spans.is_empty() && remaining > room {
                break;
            }
            let take = remaining.min(room);
            for &(row, out) in &e.rows[off..off + take] {
                spans.push(e.spans);
                flat.push(row);
                key.push(ei as u32);
                out_idx.push(out);
            }
            off += take;
            if off == e.rows.len() {
                ei += 1;
                off = 0;
            }
            if spans.len() == cap {
                break;
            }
        }
        let n = spans.len();
        if n == 0 {
            break;
        }
        let t_compute = Instant::now();
        let mut timing = MoeStepTiming::default();
        moe_slots_compute(
            dim,
            mi,
            1,
            n,
            Some(&flat[..n]),
            &spans,
            x_rot,
            |a| key[a],
            moe,
            &mut timing,
            true,
            None,
        )?;
        stats.compute_ns += elapsed_ns(t_compute);
        let t_scatter = Instant::now();
        {
            let (dn, slot_group, slot_si) = (&moe.dn, &moe.slot_group, &moe.slot_si);
            let (group_start, group_off) = (&moe.group_start, &moe.group_off);
            let out_ptr = &out_ptr;
            hipfire_cpu::splice_pool::par_for_each_chunk_mut(&mut out_idx[..n], 1, |a, o| {
                let gi = slot_group[a] as usize;
                let si = slot_si[a] as usize;
                let w = group_n(group_start, gi);
                let base = group_off[gi];
                // SAFETY: `sp_rows_validate` proved `o[0] < out_bf16.len() / dim`
                // and that no other slot of the call has this out index, so this
                // `dim`-long range is in bounds and written by this worker only.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(out_ptr.get().add(o[0] as usize * dim), dim)
                };
                for (r, d) in dst.iter_mut().enumerate() {
                    *d = bf16_bits_rne(dn[base + r * w + si]);
                }
            });
        }
        stats.scatter_ns += elapsed_ns(t_scatter);
        stats.chunks += 1;
    }
    Ok(())
}

/// Host twin of the canonical BF16 MoE combine
/// (`moe_down_combine_top10_batched` / `_grouped_top10`, generalized to any
/// `top_k`): the reducer that owns route weighting, fed by UNWEIGHTED raw F32
/// per-slot down values in the GPU expanded layout.
///
/// `expanded` is `[rows, top_k, dim]` (slot `s` at `s * dim`); `target` is
/// `[rows, dim]`, the residual on entry and the combined result on return. Per
/// row every element starts at `rt(target)` and each LIVE slot (id in
/// `0..n_experts`) is added in ascending `(expert id, rank)` order as
/// `acc = rt(acc + rt(rt(down) * rt(weight)))`, `rt` = BF16 round-to-nearest-even.
/// Slots with an invalid / sentinel id contribute nothing and neither their
/// `expanded` range nor their weight is read, so stale or NaN data there is
/// harmless.
///
/// # Panics
/// If an input is shorter than its extent.
#[allow(clippy::too_many_arguments)]
pub fn moe_canonical_combine_reference(
    dim: usize,
    rows: usize,
    top_k: usize,
    n_experts: usize,
    ids: &[i32],
    weights: &[f32],
    expanded: &[f32],
    target: &mut [f32],
) {
    let slots = rows * top_k;
    assert!(
        ids.len() >= slots
            && weights.len() >= slots
            && expanded.len() >= slots * dim
            && target.len() >= rows * dim,
        "moe_canonical_combine_reference: an input is shorter than its extent"
    );
    for t in 0..rows {
        let row = &mut target[t * dim..(t + 1) * dim];
        for a in row.iter_mut() {
            *a = bf16_rt(*a);
        }
        let mut prev: Option<(i32, usize)> = None;
        loop {
            let mut best: Option<(i32, usize)> = None;
            for j in 0..top_k {
                let id = ids[t * top_k + j];
                if !route_id_valid(id, n_experts) {
                    continue;
                }
                let key = (id, j);
                if prev.is_some_and(|p| key <= p) {
                    continue;
                }
                if best.map_or(true, |b| key < b) {
                    best = Some(key);
                }
            }
            let Some((_, j)) = best else { break };
            prev = best;
            let s = t * top_k + j;
            let w = bf16_rt(weights[s]);
            for (a, &d) in row.iter_mut().zip(&expanded[s * dim..(s + 1) * dim]) {
                *a = bf16_rt(*a + bf16_rt(bf16_rt(d) * w));
            }
        }
    }
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
            CPU_STEP_WALL_NS.fetch_add(step_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        HostExec::Split { gpu_rows } => {
            SPLIT_STEPS.fetch_add(1, Ordering::Relaxed);
            SPLIT_GPU_ROWS.fetch_add(gpu_rows as u64, Ordering::Relaxed);
            SPLIT_WALL_NS.fetch_add(step_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
    trace_step(q, w, rotated, residual, exec, timing);
}

/// Which projection of a selected expert a [`MoeJobSource`] job reads.
#[derive(Clone, Copy)]
enum MoeProj {
    GateUp,
    Down,
}

/// Group `gi`'s member count from the prefix-summed `group_start`.
/// Column block (gate/up pairs) the middle region gathers per step on its stack.
const MIDDLE_CHUNK: usize = 2048;

fn group_n(group_start: &[u32], gi: usize) -> usize {
    (group_start[gi + 1] - group_start[gi]) as usize
}

/// Fill `group_off` (length `groups + 1`) with each group's element offset into a
/// projection buffer whose per-group run is `per_row * n(g)` elements — `2 * mi`
/// for gate_up, `dim` for down — plus a trailing total.
fn fill_group_off(group_start: &[u32], group_off: &mut [usize], per_row: usize) {
    let groups = group_start.len() - 1;
    let mut acc = 0usize;
    for gi in 0..groups {
        group_off[gi] = acc;
        acc += per_row * group_n(group_start, gi);
    }
    group_off[groups] = acc;
}

/// The splice's grouped window presented to
/// [`hipfire_cpu::gemv::gemv_shared_sourced`] without materializing a job slice:
/// every read is answered from the reusable index metadata (`order`,
/// `group_start`) plus the activation buffer and the resolved experts, so no
/// per-call descriptor allocation is needed and nothing borrows out of the
/// thread-local scratch after the call.
struct MoeJobSource<'a, 's> {
    experts: &'s [CpuMoeExpert<'a>],
    order: &'s [u32],
    group_start: &'s [u32],
    act: &'s [f32],
    /// Row width of `act`: `dim` for gate_up, `mi` for down.
    row_len: usize,
    top_k: usize,
    /// `Some`: entry `a`'s activation row is `flat_of[a] / top_k` (sparse host
    /// slots); `None`: entry `a` is flat slot `a`.
    flat_of: Option<&'s [u32]>,
    proj: MoeProj,
}

impl<'a, 's> MoeJobSource<'a, 's> {
    /// A group's expert — its first-seen slot; every slot in the group resolved to
    /// the same expert id, hence the same bytes.
    fn first_expert(&self, j: usize) -> &'s CpuMoeExpert<'a> {
        &self.experts[self.order[self.group_start[j] as usize] as usize]
    }
}

impl<'a, 's> hipfire_cpu::gemv::SharedJobSource for MoeJobSource<'a, 's> {
    fn jobs(&self) -> usize {
        self.group_start.len() - 1
    }
    fn n(&self, j: usize) -> usize {
        group_n(self.group_start, j)
    }
    fn quant(&self, j: usize) -> CpuQuant {
        let e = self.first_expert(j);
        match self.proj {
            MoeProj::GateUp => e.gate_up_quant,
            MoeProj::Down => e.down_quant,
        }
    }
    fn packed(&self, j: usize) -> &[u8] {
        let e = self.first_expert(j);
        match self.proj {
            MoeProj::GateUp => e.gate_up,
            MoeProj::Down => e.down,
        }
    }
    fn xs_row(&self, j: usize, t: usize) -> &[f32] {
        let slot = self.order[self.group_start[j] as usize + t] as usize;
        // gate_up consumes token `slot / top_k`'s activation row; down consumes
        // the slot's own post-SiLU hidden row.
        let row = match self.proj {
            MoeProj::GateUp => self.flat_of.map_or(slot, |f| f[slot] as usize) / self.top_k,
            MoeProj::Down => slot,
        };
        &self.act[row * self.row_len..(row + 1) * self.row_len]
    }
}

/// Per-shape accounting for the MoE splice under
/// `HIPFIRE_CPU_EXEC_TRACE=1`: one line per `(quant, dim, mi, top_k, rows,
/// graded)` at its first call and at every doubling, so the `calls=1` line is
/// the cold first step and later lines are steady state. Keyed by shape rather
/// than summed process-wide, for the same reason [`trace_step`] is.
#[allow(clippy::too_many_arguments)]
fn trace_moe_step(
    q: CpuQuant,
    dim: usize,
    mi: usize,
    k: usize,
    rows: usize,
    graded: bool,
    timing: MoeStepTiming,
) {
    if hipfire_config::developer_var("HIPFIRE_CPU_EXEC_TRACE").is_err() {
        return;
    }
    let Ok(mut shapes) = MOE_SHAPES.lock() else {
        return;
    };
    let stats = shapes
        .entry((q as u8, dim, mi, k, rows, graded))
        .or_default();
    stats.calls += 1;
    stats.d2h_enqueue_ns += timing.d2h_enqueue_ns;
    stats.d2h_complete_ns += timing.d2h_complete_ns;
    stats.gu_ns += timing.gu_ns;
    stats.dn_ns += timing.dn_ns;
    stats.h2d_ns += timing.h2d_ns;
    stats.setup_ns += timing.setup_ns;
    stats.middle_ns += timing.middle_ns;
    stats.combine_ns += timing.combine_ns;
    stats.total_ns += timing.total_ns;
    stats.gu_prep_ns += timing.gu_prep_ns;
    stats.gu_exec_ns += timing.gu_exec_ns;
    stats.dn_prep_ns += timing.dn_prep_ns;
    stats.dn_exec_ns += timing.dn_exec_ns;
    stats.middle_gather_ns += timing.middle_gather_ns;
    stats.middle_silu_ns += timing.middle_silu_ns;
    stats.middle_fwht_ns += timing.middle_fwht_ns;
    stats.d2h_wait_ns += timing.d2h_wait_ns;
    for i in 0..MOE_QUANT_MIX_LEN {
        stats.gu_mix[i] += timing.gu_mix[i] as u64;
        stats.dn_mix[i] += timing.dn_mix[i] as u64;
    }
    let stats = *stats;
    if !stats.calls.is_power_of_two() {
        return;
    }
    let (on_cpu, on_gpu) = cpu_exec_counters();
    let per_ms = |ns: u64| ns as f64 / 1e6 / stats.calls as f64;
    // Enqueue + completion is copy/API wall after the diagnostic producer drain;
    // neither region is a physical PCIe transfer measurement.
    let d2h_copy_ns = stats.d2h_enqueue_ns + stats.d2h_complete_ns;
    eprintln!(
        "cpu exec: moe expert splice dim={dim} mi={mi} k={k} rows={rows} quant={q:?} \
         graded={graded} | {} calls | {on_cpu} steps on CPU, {on_gpu} host-mapped steps still \
         on GPU | mean per call: d2h_enqueue={:.2}ms d2h_complete={:.2}ms gu={:.2}ms dn={:.2}ms \
         h2d={:.2}ms setup={:.2}ms middle={:.2}ms combine={:.2}ms total={:.2}ms \
         | gu_prep={:.2}ms gu_exec={:.2}ms dn_prep={:.2}ms dn_exec={:.2}ms \
         gather={:.2}ms silu={:.2}ms fwht={:.2}ms d2h_wait={:.2}ms d2h_copy_api={:.2}ms \
         | gu_mix=[{}] dn_mix=[{}]",
        stats.calls,
        per_ms(stats.d2h_enqueue_ns),
        per_ms(stats.d2h_complete_ns),
        per_ms(stats.gu_ns),
        per_ms(stats.dn_ns),
        per_ms(stats.h2d_ns),
        per_ms(stats.setup_ns),
        per_ms(stats.middle_ns),
        per_ms(stats.combine_ns),
        per_ms(stats.total_ns),
        per_ms(stats.gu_prep_ns),
        per_ms(stats.gu_exec_ns),
        per_ms(stats.dn_prep_ns),
        per_ms(stats.dn_exec_ns),
        per_ms(stats.middle_gather_ns),
        per_ms(stats.middle_silu_ns),
        per_ms(stats.middle_fwht_ns),
        per_ms(stats.d2h_wait_ns),
        per_ms(d2h_copy_ns),
        format_quant_mix(&stats.gu_mix, stats.calls),
        format_quant_mix(&stats.dn_mix, stats.calls),
    );
}

/// `CpuQuant` discriminant (`q as usize`) → short variant name for the
/// per-call quant-mixture print. Order matches the enum declaration; unknown
/// indices (past the current variant count) print as `q{N}`.
fn cpu_quant_mix_name(idx: usize) -> &'static str {
    match idx {
        0 => "Mq4G256",
        1 => "Mq4G256V2",
        2 => "Mq6G256",
        3 => "Mq3G256",
        4 => "Mq3G256Lloyd",
        5 => "Hfq6G256",
        6 => "Hfq3G256",
        7 => "Hfq2G256",
        8 => "Hfq4G128",
        9 => "Hfq3G128",
        10 => "Hfq2G128",
        11 => "Mq5G256",
        12 => "Mq2G256",
        13 => "Mq2G256Lloyd",
        14 => "Mq2G256LloydU",
        15 => "Mq4G256Lloyd",
        16 => "Mq4CG256",
        17 => "Mq6G256V2",
        18 => "Mq5G256V2",
        19 => "Mq3G256V2",
        20 => "Mq2G256V2",
        21 => "Tq2G128",
        22 => "Bq1G128",
        23 => "Hfq4G256",
        24 => "F16",
        25 => "F32",
        26 => "Bf16",
        27 => "Q8F16",
        _ => "q?",
    }
}

/// Mean slots per call per quant (`NamexMean` pairs, one decimal) from summed
/// mixture counts. Empty (trace shapes that predate the mix) prints `-`.
fn format_quant_mix(mix: &[u64; MOE_QUANT_MIX_LEN], calls: usize) -> String {
    let mut out = String::new();
    for (i, &sum) in mix.iter().enumerate() {
        if sum == 0 {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        let mean = sum as f64 / calls.max(1) as f64;
        out.push_str(&format!("{}x{mean:.1}", cpu_quant_mix_name(i)));
    }
    if out.is_empty() {
        out.push('-');
    }
    out
}

/// One splice call's wall-time split, handed to [`trace_moe_step`].
#[derive(Clone, Copy, Default)]
struct MoeStepTiming {
    d2h_enqueue_ns: u64,
    d2h_complete_ns: u64,
    gu_ns: u64,
    dn_ns: u64,
    h2d_ns: u64,
    /// Resolving the selected slots to host experts.
    setup_ns: u64,
    /// SiLU + optional hidden rotation, the whole per-token loop.
    middle_ns: u64,
    /// The routing-weighted accumulate into the residual.
    combine_ns: u64,
    /// The whole splice call's wall time.
    total_ns: u64,
    /// GU prep (activation-sum/metadata) vs Rayon decode-and-dot execution.
    /// `gu_ns` stays the whole region (`prep + exec`); `prep_ns` is 0 until
    /// the traced `gemv_shared_sourced` sibling lands.
    gu_prep_ns: u64,
    gu_exec_ns: u64,
    /// DN prep (activation-sum/metadata) vs Rayon decode-and-dot execution.
    dn_prep_ns: u64,
    dn_exec_ns: u64,
    /// CPU middle split: gather vs SiLU vs FWHT sums across the per-token loop.
    middle_gather_ns: u64,
    middle_silu_ns: u64,
    middle_fwht_ns: u64,
    /// Trace-only host producer drain, including synchronization API overhead.
    d2h_wait_ns: u64,
    /// Selected-slot GU / DN quant mixture of this call, indexed by
    /// `CpuQuant as usize`; summed into the shape stats by [`trace_moe_step`].
    gu_mix: [u16; MOE_QUANT_MIX_LEN],
    dn_mix: [u16; MOE_QUANT_MIX_LEN],
}

/// Whether [`run_host_mapped_gemv`] / [`run_host_mapped_gemv_residual`] can drive
/// this weight: host-mapped *and* a decodable format.
pub fn host_mapped_cpu_capable(gpu: &Gpu, w: &WeightRef) -> bool {
    cpu_exec_enabled() && gpu.host_located(w.buf) && dense_cpu_quant_for(w.dtype).is_some()
}

/// `HIPFIRE_MOE_CPU_ORACLE=1`: run the same host-mapped expert bytes that fed the
/// GPU's routed-expert kernels through the CPU SIMD expert FFN
///
/// Attribution, not a fallback. If the CPU path over the same blobs reproduces the
/// GPU's own per-expert down outputs, the packed layout and the GPU's read of
/// host-mapped memory are sound — a wrong token then cannot be explained by either
/// and must come from the routing, the dtype/arm selection, or a non-expert tensor.
/// If they disagree, the printed rank and element localize it in one run, which a
/// resident control arm cannot do on a card too small to hold the model.
///
/// Lives here rather than in the arch because this crate owns the CPU seam and the
/// arch intentionally has no `hipfire-cpu` dependency. Both candidate inputs are
/// tried (activation as stored, and freshly rotated), so the report is evidence
/// about which one the kernels consume instead of assuming it.
#[allow(clippy::too_many_arguments)]
pub fn moe_cpu_oracle_report(
    gpu: &Gpu,
    quant: CpuQuant,
    dim: usize,
    mi: usize,
    k: usize,
    gate_up_owner: &GpuTensor,
    down_owner: &GpuTensor,
    x_norm: &GpuTensor,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
    down_expanded: &GpuTensor,
    layer_idx: u16,
    // `ffn.expert_down_awq_ptrs.is_some()`. Printed because it is a hard
    // precondition of any CPU down projection: a bare `gemv` ignores a
    // per-expert AWQ scale, so a fixture that gains sidecars later must not
    // silently keep taking the CPU path. The index scan that established its
    // absence here is a 60 MB window, not a guarantee.
    down_awq: bool,
) {
    let (x, ti, tw, dn_gpu) = match (
        download_f32(gpu, x_norm, dim),
        download_f32(gpu, topk_indices, k),
        download_f32(gpu, topk_weights, k),
        download_f32(gpu, down_expanded, k * dim),
    ) {
        (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
        _ => {
            eprintln!("[moe-cpu-oracle] layer {layer_idx}: scratch download failed");
            return;
        }
    };
    let (gu, dn) = match (gpu.host_bytes(gate_up_owner), gpu.host_bytes(down_owner)) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            eprintln!("[moe-cpu-oracle] layer {layer_idx}: expert blobs are not host-mapped");
            return;
        }
    };
    let blobs = hipfire_cpu::moe::ExpertBlobs {
        gate_up: gu,
        down: dn,
        gate_up_stride: hipfire_cpu::gemv::tensor_bytes(quant, 2 * mi, dim),
        down_stride: hipfire_cpu::gemv::tensor_bytes(quant, dim, mi),
        quant,
        dim,
        mi,
    };
    // Which side is empty is the question, so report magnitudes, not just the
    // worst relative difference: a layer whose GPU expert output is identically
    // zero and whose CPU output is not is a *missing contribution*, a different
    // fault from a layer that computes a similar-but-different number.
    let sum_abs = |v: &[f32]| v.iter().map(|x| x.abs() as f64).sum::<f64>();
    let zeros = |v: &[f32]| v.iter().filter(|x| **x == 0.0).count();
    let mut xin = x[..dim.min(x.len())].to_vec();
    if dim % 256 == 0 {
        rotate_x(&mut xin);
    }
    let mut cpu_all = vec![0.0f32; k * dim];
    let mut routed_ranks = 0usize;
    for krank in 0..k.min(ti.len()).min(tw.len()) {
        let expert = (ti[krank].to_bits() as i32) as u16 as usize;
        if hipfire_cpu::moe::run_experts(
            &blobs,
            &xin,
            &[(expert, 1.0)],
            &mut cpu_all[krank * dim..(krank + 1) * dim],
        )
        .is_ok()
        {
            routed_ranks += 1;
        }
    }
    eprintln!(
        "[moe-cpu-oracle] layer {layer_idx} ranks {routed_ranks}/{k} down_awq={down_awq} |x| {:.4e} (zeros {}) |gpu-down| {:.4e} (zeros {}/{}) |cpu-down| {:.4e} (zeros {})",
        sum_abs(&x[..dim.min(x.len())]),
        zeros(&x[..dim.min(x.len())]),
        sum_abs(&dn_gpu),
        zeros(&dn_gpu),
        dn_gpu.len(),
        sum_abs(&cpu_all),
        zeros(&cpu_all),
    );
    for rotated in [false, true] {
        if x.len() < dim || (rotated && dim % 256 != 0) {
            break;
        }
        let mut xin = x[..dim].to_vec();
        if rotated {
            rotate_x(&mut xin);
        }
        let mut worst_rel = 0.0f64;
        let mut worst_at = (usize::MAX, usize::MAX);
        let mut worst_vals = (0.0f32, 0.0f32);
        let mut ranks = 0usize;
        for krank in 0..k.min(ti.len()).min(tw.len()) {
            let expert = (ti[krank].to_bits() as i32) as u16 as usize;
            let base = krank * dim;
            if base + dim > dn_gpu.len() {
                break;
            }
            let mut out = vec![0.0f32; dim];
            if hipfire_cpu::moe::run_experts(&blobs, &xin, &[(expert, 1.0)], &mut out).is_err() {
                continue;
            }
            ranks += 1;
            for j in 0..dim {
                let (a, b) = (out[j] as f64, dn_gpu[base + j] as f64);
                let rel = (a - b).abs() / a.abs().max(b.abs()).max(1e-6);
                if rel > worst_rel {
                    worst_rel = rel;
                    worst_at = (krank, j);
                    worst_vals = (a as f32, b as f32);
                }
            }
        }
        eprintln!(
            "[moe-cpu-oracle] layer {layer_idx} quant {quant:?} ranks {ranks} input {}: max |cpu-gpu|/max|.| = {worst_rel:.3e} at (rank, elem) {worst_at:?} cpu={} gpu={}",
            if rotated { "fwht-rotated" } else { "as-stored" },
            worst_vals.0,
            worst_vals.1,
        );
    }
}

fn host_mapped_quant(gpu: &Gpu, w: &WeightRef) -> Result<CpuQuant, DispatchError> {
    if !cpu_exec_enabled() {
        return Err(cpu_err("cpu exec is not enabled (memory.offload_exec)"));
    }
    if !gpu.host_located(w.buf) {
        return Err(cpu_err("weight tensor is not host-mapped"));
    }
    dense_cpu_quant_for(w.dtype).ok_or_else(|| {
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
    if !gpu.host_located(w.buf) || dense_cpu_quant_for(w.dtype).is_none() {
        return None;
    }
    // Mirror `GemvFamily::run_input`: a `Raw` input is rotated by the launcher
    // *and only when* the dtype's plan is `FwhtG256`; a `Prerotated` input is
    // never touched again.
    let rotate_input = matches!(input, GemvInput::Raw(_)) && raw_input_is_rotated(w.dtype);
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
/// decode it — the coverage counter's question. Matches `Step::Gemv` /
/// `Step::GemvResidual` in the step seam only: the MoE tier's grouped host read
/// (which wide prefill deliberately keeps on the GPU) and the `llama.rs` fused
/// split are structurally invisible to it — see the counter docs above.
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
    let mut out = vec![0.0f32; n];
    download_into_f32(gpu, t, n, &mut out)?;
    Ok(out)
}

/// [`download_f32`] into a caller-owned buffer, so the MoE splice's thread-local
/// workspaces are reused instead of re-allocated every layer.
fn download_into_f32(
    gpu: &Gpu,
    t: &GpuTensor,
    n: usize,
    out: &mut [f32],
) -> Result<(), DispatchError> {
    assert!(
        t.numel() >= n,
        "cpu exec: tensor has {} elements, need {n}",
        t.numel()
    );
    assert!(
        out.len() >= n,
        "cpu exec: buffer has {} elements, need {n}",
        out.len()
    );
    // Safety: `out[..n]` is a live `n`-element f32 buffer; the slice covers
    // exactly those bytes and does not outlive the call.
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.memcpy_dtoh_auto(bytes, &t.buf)
        .map_err(|e| cpu_err(&format!("D2H: {e}")))
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

/// Reuse owned pinned storage across CPU steps. Growth/device transitions
/// drop an owner that drains pending accesses; the warm same-stream path has
/// no return-side host wait, event record, or extra staging copy.
fn with_cpu_transfer<R>(
    gpu: &Gpu,
    values: usize,
    run: impl FnOnce(&mut hip_bridge::PinnedTransferBuffer) -> Result<R, DispatchError>,
) -> Result<R, DispatchError> {
    if gpu.graphs.capture_mode || gpu.replay.is_recording() {
        return Err(cpu_err(
            "CPU transfer cannot run during capture or retained recording",
        ));
    }
    gpu.bind_thread()
        .map_err(|e| cpu_err(&format!("bind: {e}")))?;
    let bytes = values
        .checked_mul(4)
        .ok_or_else(|| cpu_err("CPU transfer extent overflows"))?;
    CPU_TRANSFER.with(|cell| {
        let mut owner = cell.borrow_mut();
        if owner
            .as_ref()
            .is_none_or(|b| b.size() < bytes || b.device() != gpu.device_id)
        {
            // Retire the old allocation before binding/allocating the new one.
            // Its destructor restores this caller's current device.
            drop(owner.take());
            *owner = Some(
                gpu.hip
                    .transfer_buffer(bytes)
                    .map_err(|e| cpu_err(&format!("pinned transfer stage: {e}")))?,
            );
        }
        let result = run(owner.as_mut().expect("stage allocated above"));
        // Earlier enqueues can survive a later error. Protect their device
        // allocations as well as the stage before returning the failure.
        if result.is_err() && gpu.hip.device_synchronize().is_err() {
            std::mem::forget(owner.take());
        }
        result
    })
}

/// Prewarm the calling thread's owned pinned transfer stage for `values` f32
/// elements (checked `values*4` bytes) on `gpu`'s device, reusing the existing
/// [`with_cpu_transfer`] allocation/reuse/drain lifecycle without running any
/// math: growth or a device transition drops an owner that drains pending
/// accesses; the warm same-device path keeps the existing owner untouched (no
/// hot allocation). Admission calls this at load/prewarm after memory
/// admission; refuses capture/retained-recording like every other transfer use.
pub fn reserve_cpu_transfer_values(gpu: &Gpu, values: usize) -> Result<(), DispatchError> {
    with_cpu_transfer(gpu, values, |_| Ok(()))
}

/// Real pinned-transfer capacity (bytes) of the calling thread's stage for
/// `device_id`: `0` when no stage exists, when it belongs to another device,
/// or while it is pending (an in-flight transfer the caller must not reuse),
/// otherwise its actual [`hip_bridge::PinnedTransferBuffer::size`]. Pure
/// reader: no allocation, no growth, no mutation.
pub fn cpu_transfer_capacity_bytes(device_id: i32) -> usize {
    CPU_TRANSFER.with(|cell| {
        let owner = cell.borrow();
        match owner.as_ref() {
            Some(b) if b.device() == device_id && !b.is_pending() => b.size(),
            _ => 0,
        }
    })
}

fn transfer_values(transfer: &mut hip_bridge::PinnedTransferBuffer, n: usize) -> &mut [f32] {
    let bytes = &mut transfer.as_bytes_mut()[..n * 4];
    // SAFETY: hipHostMalloc is f32-aligned, bytes are initialized, and the
    // owner permits host access only after queued GPU accesses have completed.
    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<f32>(), n) }
}

fn download_transfer(
    gpu: &Gpu,
    transfer: &mut hip_bridge::PinnedTransferBuffer,
    tensor: &GpuTensor,
    offset: usize,
    n: usize,
) -> Result<(), DispatchError> {
    if tensor.numel() < n {
        return Err(cpu_err("D2H tensor extent too small"));
    }
    transfer
        .download_async(
            &gpu.hip,
            &tensor.buf,
            offset * 4,
            n * 4,
            gpu.active_stream.as_ref(),
        )
        .map_err(|e| cpu_err(&format!("D2H enqueue: {e}")))
}

fn upload_transfer(
    gpu: &Gpu,
    transfer: &mut hip_bridge::PinnedTransferBuffer,
    tensor: &GpuTensor,
    offset: usize,
    n: usize,
) -> Result<(), DispatchError> {
    if tensor.numel() < n {
        return Err(cpu_err("H2D tensor extent too small"));
    }
    transfer
        .upload_async(
            &gpu.hip,
            &tensor.buf,
            offset * 4,
            n * 4,
            gpu.active_stream.as_ref(),
        )
        .map_err(|e| cpu_err(&format!("H2D enqueue: {e}")))
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

/// One MoE layer window's wall-time split on the CPU-deferred route, handed to
/// [`moe_window_trace`]. All spans are host wall (`Instant`), adjacent not
/// nested: `gpu_sealed_ns` covers the sealed GPU program (`execute_steps`),
/// `cpu_splice_ns` the CPU splice (`moe_cpu_experts`), and `other_ns` the
/// caller-side gap between/around them (scratch access, flag checks). The
/// caller records all three from one outer span so they reconcile against
/// `total_ns`; nothing here launches, syncs, or allocates on the hot path
/// beyond the guide struct itself.
#[derive(Clone, Copy, Default, Debug)]
pub struct MoeWindowTiming {
    /// Whole window: sealed GPU program + CPU splice + caller-side gap.
    pub total_ns: u64,
    /// Sealed GPU program (`execute_steps`): router/norm/shared GPU stages — and,
    /// since the CPU routed-expert splice became a stage of that program, the
    /// splice's cost as well on the `cpu_deferred` path.
    pub gpu_sealed_ns: u64,
    /// CPU splice (`moe_cpu_experts`): whole routed FFN on the CPU.
    ///
    /// Reports **0** on the `cpu_deferred` path since the splice moved inside the
    /// sealed program — its cost is part of `gpu_sealed_ns` there, and the seam's
    /// own per-shape splice trace (`moe expert splice … steps on CPU …`) carries
    /// the per-call figure. Kept for the non-deferred shape of this struct.
    pub cpu_splice_ns: u64,
    /// Caller-side gap: `total - sealed - splice` (scratch, flags, hooks).
    /// The subtraction is exact only while `cpu_splice_ns` is 0 — which it is by
    /// construction on the `cpu_deferred` path, since the splice's cost is already
    /// inside `gpu_sealed_ns`; a producer that starts populating `cpu_splice_ns`
    /// again must keep the two disjoint.
    pub other_ns: u64,
    /// Forward width that selected this route (1 = decode, ≤4 = narrow).
    pub rows: usize,
    /// Routed-expert count per token (`num_experts_per_tok`).
    pub top_k: usize,
}

/// Per-shape totals for [`moe_window_trace`]: one line per `(layer, rows,
/// top_k)` at its first window and at every doubling — the same power-of-two
/// schedule as [`trace_moe_step`], so the `calls=1` line is the cold first
/// window and later lines are steady state. Keyed per layer (not just shape)
/// so the printed `layer=` owns its `calls` count.
#[derive(Clone, Copy, Default)]
struct MoeWindowStats {
    calls: u64,
    total_ns: u64,
    gpu_sealed_ns: u64,
    cpu_splice_ns: u64,
    other_ns: u64,
}

/// `(layer, rows, top_k)` → running per-layer window totals.
static MOE_WINDOWS: LazyLock<Mutex<BTreeMap<(usize, usize, usize), MoeWindowStats>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// Opt-in aggregated per-layer window timing on the CPU-deferred MoE route,
/// under the same `HIPFIRE_CPU_EXEC_TRACE=1` gate as [`trace_moe_step`].
///
/// The arch caller records one window per MoE layer: the sealed GPU program
/// span, the CPU splice span, and the caller-side gap, all from host wall
/// timestamps around calls it already makes. No GPU events, no new device
/// sync, no execution-order change. Summaries print outside the measured
/// region (power-of-two schedule), and tracing is disabled by default with a
/// single snapshot-map read on the hot path.
pub fn moe_window_trace(layer_idx: usize, timing: MoeWindowTiming) {
    if moe_trace_enabled() {
        moe_window_trace_inner(layer_idx, timing);
    }
}

fn moe_window_trace_inner(layer_idx: usize, timing: MoeWindowTiming) {
    let Ok(mut windows) = MOE_WINDOWS.lock() else {
        return;
    };
    let stats = windows
        .entry((layer_idx, timing.rows, timing.top_k))
        .or_default();
    stats.calls += 1;
    stats.total_ns += timing.total_ns;
    stats.gpu_sealed_ns += timing.gpu_sealed_ns;
    stats.cpu_splice_ns += timing.cpu_splice_ns;
    stats.other_ns += timing.other_ns;
    let stats = *stats;
    if !stats.calls.is_power_of_two() {
        return;
    }
    let per_ms = |ns: u64| ns as f64 / 1e6 / stats.calls as f64;
    eprintln!(
        "cpu exec: moe window layer={layer_idx} rows={} k={} | {} calls | \
         mean per window: sealed_gpu={:.2}ms cpu_splice={:.2}ms other={:.2}ms total={:.2}ms",
        timing.rows,
        timing.top_k,
        stats.calls,
        per_ms(stats.gpu_sealed_ns),
        per_ms(stats.cpu_splice_ns),
        per_ms(stats.other_ns),
        per_ms(stats.total_ns),
    );
}

/// One MoE layer's GPU-vs-CPU interleave on the batched-slots route, handed to
/// [`moe_layer_trace`]. Host wall around the two halves the layer already
/// executes: `attn_ns` covers the attention half (QKVZA/recurrence or
/// QKV/attention + output projection, host wall including queued GPU work, not
/// device-only time), `moe_ns` covers the MoE FFN body (`prefill_moe_ffn_body`
/// including any CPU splice). Adjacent, not nested; `total_ns` is the outer
/// span so the parts reconcile. No GPU events, no new sync, no order change.
#[derive(Clone, Copy, Default, Debug)]
pub struct MoeLayerTiming {
    pub total_ns: u64,
    pub attn_ns: u64,
    pub moe_ns: u64,
    pub rows: usize,
}

/// Per-layer totals for [`moe_layer_trace`]: one line per `(layer, attn-kind,
/// rows)` at its first window and at every doubling — same power-of-two
/// schedule as the splice. Keyed per layer (not just shape) so the printed
/// `layer=` owns its `calls` count.
#[derive(Clone, Copy, Default)]
struct MoeLayerStats {
    calls: u64,
    total_ns: u64,
    attn_ns: u64,
    moe_ns: u64,
}

/// `(layer, full_attn, rows)` → running per-layer totals.
static MOE_LAYERS: LazyLock<Mutex<BTreeMap<(usize, bool, usize), MoeLayerStats>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// Opt-in per-layer GPU/CPU interleave timing on the batched-slots MoE route,
/// under the same `HIPFIRE_CPU_EXEC_TRACE=1` gate. The layer caller records
/// host wall around the attention half and the MoE FFN body it already runs;
/// summaries print outside the measured region. Disabled by default with a
/// single snapshot-map read on the hot path.
pub fn moe_layer_trace(layer_idx: usize, is_full_attn: bool, timing: MoeLayerTiming) {
    if moe_trace_enabled() {
        moe_layer_trace_inner(layer_idx, is_full_attn, timing);
    }
}

fn moe_layer_trace_inner(layer_idx: usize, is_full_attn: bool, timing: MoeLayerTiming) {
    let Ok(mut layers) = MOE_LAYERS.lock() else {
        return;
    };
    let stats = layers
        .entry((layer_idx, is_full_attn, timing.rows))
        .or_default();
    stats.calls += 1;
    stats.total_ns += timing.total_ns;
    stats.attn_ns += timing.attn_ns;
    stats.moe_ns += timing.moe_ns;
    let stats = *stats;
    if !stats.calls.is_power_of_two() {
        return;
    }
    let per_ms = |ns: u64| ns as f64 / 1e6 / stats.calls as f64;
    eprintln!(
        "cpu exec: moe layer layer={layer_idx} {} rows={} | {} calls | \
         mean per layer: attn={:.2}ms moe_ffn={:.2}ms total={:.2}ms",
        if is_full_attn { "fullattn" } else { "deltanet" },
        timing.rows,
        stats.calls,
        per_ms(stats.attn_ns),
        per_ms(stats.moe_ns),
        per_ms(stats.total_ns),
    );
}

/// One batched serve-engine step's wall-time split on the MoE route, handed to
/// [`moe_step_trace`]. Host wall around work the step already does:
/// `forward_ns` covers the batched forward (`ModelRig::forward_step` →
/// `forward_batch_slots_graphed_opts`, all layers), `sample_ns` the sampler
/// (`sample_per_slot`, GPU kernels + launch), `sync_readback_ns` the device
/// synchronize + token-id D2H, and `emit_ns` the accept/commit/emit loop.
/// Adjacent, not nested; `total_ns` is the outer step span so the parts
/// reconcile. No GPU events, no new sync, no order change.
#[derive(Clone, Copy, Default, Debug)]
pub struct MoeStepTime {
    pub total_ns: u64,
    pub forward_ns: u64,
    pub sample_ns: u64,
    pub sync_readback_ns: u64,
    pub emit_ns: u64,
    pub rows: usize,
}

/// Per-shape totals for [`moe_step_trace`]: one line per `rows` at its first
/// step and at every doubling — same power-of-two schedule as the splice.
#[derive(Clone, Copy, Default)]
struct MoeServeStepStats {
    calls: u64,
    total_ns: u64,
    forward_ns: u64,
    sample_ns: u64,
    sync_readback_ns: u64,
    emit_ns: u64,
}

/// `rows` → running per-shape step totals.
static MOE_STEPS: LazyLock<Mutex<BTreeMap<usize, MoeServeStepStats>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// Opt-in whole-step timing on the batched MoE route, under the same
/// `HIPFIRE_CPU_EXEC_TRACE=1` gate. The serve engine records host wall around
/// the forward, the sampler, the sync/readback, and the emit loop it already
/// runs; summaries print outside the measured region. Disabled by default
/// with a single snapshot-map read on the hot path.
pub fn moe_step_trace(timing: MoeStepTime) {
    if moe_trace_enabled() {
        moe_step_trace_inner(timing);
    }
}

fn moe_step_trace_inner(timing: MoeStepTime) {
    let Ok(mut steps) = MOE_STEPS.lock() else {
        return;
    };
    let stats = steps.entry(timing.rows).or_default();
    stats.calls += 1;
    stats.total_ns += timing.total_ns;
    stats.forward_ns += timing.forward_ns;
    stats.sample_ns += timing.sample_ns;
    stats.sync_readback_ns += timing.sync_readback_ns;
    stats.emit_ns += timing.emit_ns;
    let stats = *stats;
    if !stats.calls.is_power_of_two() {
        return;
    }
    let per_ms = |ns: u64| ns as f64 / 1e6 / stats.calls as f64;
    eprintln!(
        "cpu exec: moe step rows={} | {} calls | \
         mean per step: forward={:.2}ms sample={:.2}ms sync_readback={:.2}ms emit={:.2}ms total={:.2}ms",
        timing.rows,
        stats.calls,
        per_ms(stats.forward_ns),
        per_ms(stats.sample_ns),
        per_ms(stats.sync_readback_ns),
        per_ms(stats.emit_ns),
        per_ms(stats.total_ns),
    );
}

/// Per-shape step accounting under `HIPFIRE_CPU_EXEC_TRACE=1`: one line per
/// distinct step shape at its first call and then at every doubling of that
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

#[cfg(test)]
mod moe_splice_tests {
    use super::*;

    /// One packed group of weights for `q`, mirroring `hipfire_cpu`'s own test
    /// fixture: a valid, finite header plus a bijective payload, so a wrong
    /// offset or format fails rather than matching by symmetry.
    fn group_bytes(q: CpuQuant, salt: usize) -> Vec<u8> {
        let mut b: Vec<u8> = (0..q.group_bytes())
            .map(|i| (((i + salt) * 37 + 11) & 0xFF) as u8)
            .collect();
        let f32x2 = |a: f32, c: f32| {
            let mut o = [0u8; 8];
            o[..4].copy_from_slice(&a.to_le_bytes());
            o[4..].copy_from_slice(&c.to_le_bytes());
            o
        };
        match q {
            CpuQuant::Mq4G256V2 => {
                // fp16 [s0 z0 s1 z1] = 0.0625, -0.25, 0.5, 0.75.
                for (i, v) in [0x2c00u16, 0xb400, 0x3800, 0x3a00].iter().enumerate() {
                    b[2 * i..2 * i + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
            CpuQuant::Mq4G128V2 => {
                // fp16 [scale][zero] = 0.0625, -0.25.
                b[..2].copy_from_slice(&0x2c00u16.to_le_bytes());
                b[2..4].copy_from_slice(&0xb400u16.to_le_bytes());
            }
            CpuQuant::Mq4G256 => b[..8].copy_from_slice(&f32x2(0.03125, -0.5)),
            CpuQuant::Mq6G256 => b[..8].copy_from_slice(&f32x2(0.0078125, -0.125)),
            CpuQuant::Mq3G256Lloyd => {
                const CB: [u16; 8] = [
                    0xbc00, 0xb800, 0xb400, 0xb000, 0x3000, 0x3400, 0x3800, 0x3c00,
                ];
                for (k, v) in CB.iter().enumerate() {
                    b[2 * k..2 * k + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
            _ => {}
        }
        b
    }

    /// A `[m, k]` weight tensor's bytes, group by group. Planar formats
    /// (`[codes plane][f16 scale plane]`, `tensor_bytes(q, m, k)` long): an
    /// LCG-derived code plane (every E8P word / 3-bit code is legal) and finite,
    /// normal, positive f16 scales.
    pub(super) fn weight_bytes(q: CpuQuant, m: usize, k: usize, salt: usize) -> Vec<u8> {
        if q.is_planar() {
            let codes = m * hipfire_cpu::gemv::row_bytes(q, k);
            let mut s = (salt as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
            let mut next = move || {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as u32
            };
            let mut out = Vec::with_capacity(hipfire_cpu::gemv::tensor_bytes(q, m, k));
            out.extend((0..codes).map(|_| next() as u8));
            for _ in 0..m * (k / 128) {
                let scale = 0x2800u16 + (next() % 0x1400) as u16;
                out.extend_from_slice(&scale.to_le_bytes());
            }
            assert_eq!(out.len(), hipfire_cpu::gemv::tensor_bytes(q, m, k));
            return out;
        }
        let groups = k / q.group_elems();
        let mut out = Vec::with_capacity(m * groups * q.group_bytes());
        for row in 0..m {
            for g in 0..groups {
                out.extend_from_slice(&group_bytes(q, salt + row * groups + g));
            }
        }
        out
    }

    /// The top-k scratch stores an id as the low bits of an `f32` bit pattern
    /// (`expert_id` inverts this).
    fn id_bits(e: usize) -> f32 {
        f32::from_bits(e as u32)
    }

    fn x_of(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.017).sin() * 0.5).collect()
    }

    /// A uniform layer: the multirow engine must reproduce repeated single-row
    /// CPU expert evaluation, measured against the existing per-expert reference
    /// ([`hipfire_cpu::moe::run_experts`]) over the same packed blobs.
    #[test]
    fn uniform_multirow_matches_run_experts() {
        let (dim, mi) = (256usize, 256usize);
        let (rows, top_k, n_exp) = (4usize, 3usize, 4usize);
        let q = CpuQuant::Mq4G256;
        let gu_stride = hipfire_cpu::gemv::row_bytes(q, dim) * 2 * mi;
        let dn_stride = hipfire_cpu::gemv::row_bytes(q, mi) * dim;
        let mut gu_all = Vec::with_capacity(n_exp * gu_stride);
        let mut dn_all = Vec::with_capacity(n_exp * dn_stride);
        for e in 0..n_exp {
            gu_all.extend_from_slice(&weight_bytes(q, 2 * mi, dim, 100 + e));
            dn_all.extend_from_slice(&weight_bytes(q, dim, mi, 200 + e));
        }
        // Repeated experts across rows, and the higher-id experts selected too.
        let idx: [usize; 12] = [0, 2, 3, 2, 2, 0, 3, 0, 1, 1, 3, 3];
        let wts: [f32; 12] = [0.5, 0.3, 0.2, 0.4, 0.4, 0.2, 0.25, 0.35, 0.4, 0.6, 0.2, 0.2];
        let x = x_of(rows * dim);
        let seed: Vec<f32> = (0..rows * dim).map(|i| 0.1 * (i % 7) as f32).collect();
        let ti: Vec<f32> = idx.iter().map(|&e| id_bits(e)).collect();

        let experts: Vec<CpuMoeExpert> = (0..rows * top_k)
            .map(|s| {
                let e = idx[s];
                CpuMoeExpert {
                    gate_up_quant: q,
                    gate_up: &gu_all[e * gu_stride..(e + 1) * gu_stride],
                    down_quant: q,
                    down: &dn_all[e * dn_stride..(e + 1) * dn_stride],
                }
            })
            .collect();
        let mut scratch = MoeScratch::default();
        let mut acc = seed.clone();
        moe_cpu_experts_host(
            dim,
            mi,
            rows,
            top_k,
            &experts,
            &x,
            &ti,
            &wts,
            &mut acc,
            &mut scratch,
            &mut MoeStepTiming::default(),
            false,
        )
        .unwrap();

        let blobs = hipfire_cpu::moe::ExpertBlobs {
            gate_up: &gu_all,
            down: &dn_all,
            gate_up_stride: gu_stride,
            down_stride: dn_stride,
            quant: q,
            dim,
            mi,
        };
        let mut reference = seed.clone();
        for t in 0..rows {
            let routing: Vec<(usize, f32)> = (0..top_k)
                .map(|j| (idx[t * top_k + j], wts[t * top_k + j]))
                .collect();
            hipfire_cpu::moe::run_experts(
                &blobs,
                &x[t * dim..(t + 1) * dim],
                &routing,
                &mut reference[t * dim..(t + 1) * dim],
            )
            .unwrap();
        }
        assert_eq!(
            acc, reference,
            "multirow CPU output != repeated single-row run_experts"
        );
    }

    /// A graded layer: experts of different tiers, including one whose gate_up
    /// and down formats differ, and a higher-id cold (MQ3-Lloyd) expert. The
    /// reference is the per-(row, rank) single-expert evaluation built from the
    /// crate's own `gemv` primitive.
    #[test]
    fn graded_multirow_matches_per_expert_gemv() {
        let (dim, mi) = (256usize, 256usize);
        let (rows, top_k, n_exp) = (3usize, 4usize, 4usize);
        let gu_q = [
            CpuQuant::Mq6G256,
            CpuQuant::Mq4G256,
            CpuQuant::Mq3G256Lloyd,
            CpuQuant::Mq6G256,
        ];
        let dn_q = [
            CpuQuant::Mq6G256,
            CpuQuant::Mq4G256,
            CpuQuant::Mq3G256Lloyd,
            CpuQuant::Mq3G256Lloyd,
        ];
        let gu: Vec<Vec<u8>> = (0..n_exp)
            .map(|e| weight_bytes(gu_q[e], 2 * mi, dim, 300 + e))
            .collect();
        let dn: Vec<Vec<u8>> = (0..n_exp)
            .map(|e| weight_bytes(dn_q[e], dim, mi, 400 + e))
            .collect();
        let idx: [usize; 12] = [0, 1, 2, 3, 2, 2, 3, 0, 3, 1, 1, 2];
        let wts: [f32; 12] = [0.4, 0.3, 0.2, 0.1, 0.5, 0.3, 0.2, 0.4, 0.6, 0.25, 0.35, 0.4];
        let x = x_of(rows * dim);
        let seed: Vec<f32> = (0..rows * dim).map(|i| 0.05 * (i % 5) as f32).collect();
        let ti: Vec<f32> = idx.iter().map(|&e| id_bits(e)).collect();

        let experts: Vec<CpuMoeExpert> = (0..rows * top_k)
            .map(|s| {
                let e = idx[s];
                CpuMoeExpert {
                    gate_up_quant: gu_q[e],
                    gate_up: &gu[e],
                    down_quant: dn_q[e],
                    down: &dn[e],
                }
            })
            .collect();
        let mut scratch = MoeScratch::default();
        let mut acc = seed.clone();
        moe_cpu_experts_host(
            dim,
            mi,
            rows,
            top_k,
            &experts,
            &x,
            &ti,
            &wts,
            &mut acc,
            &mut scratch,
            &mut MoeStepTiming::default(),
            false,
        )
        .unwrap();

        let mut reference = seed.clone();
        for t in 0..rows {
            for j in 0..top_k {
                let s = t * top_k + j;
                let e = idx[s];
                let mut gv = vec![0.0f32; 2 * mi];
                gemv(
                    gu_q[e],
                    &gu[e],
                    2 * mi,
                    dim,
                    &x[t * dim..(t + 1) * dim],
                    &mut gv,
                );
                let mut hid = vec![0.0f32; mi];
                silu_mul(&gv, &mut hid);
                if dn_q[e].is_fwht_g256() {
                    rotate_x(&mut hid);
                }
                let mut dout = vec![0.0f32; dim];
                gemv(dn_q[e], &dn[e], dim, mi, &hid, &mut dout);
                for (a, v) in reference[t * dim..(t + 1) * dim].iter_mut().zip(dout) {
                    *a += wts[s] * v;
                }
            }
        }
        assert_eq!(
            acc, reference,
            "graded multirow CPU output != per-expert gemv reference"
        );
    }

    /// A short expert span is an error, never a silent read past it.
    #[test]
    fn short_expert_span_is_an_error() {
        let (dim, mi) = (256usize, 256usize);
        let q = CpuQuant::Mq4G256;
        let short = vec![0u8; 8];
        let experts = vec![CpuMoeExpert {
            gate_up_quant: q,
            gate_up: &short,
            down_quant: q,
            down: &short,
        }];
        let mut scratch = MoeScratch::default();
        let err = moe_cpu_experts_host(
            dim,
            mi,
            1,
            1,
            &experts,
            &vec![0.0f32; dim],
            &[id_bits(0)],
            &[1.0],
            &mut vec![0.0f32; dim],
            &mut scratch,
            &mut MoeStepTiming::default(),
            false,
        )
        .expect_err("a short expert span must fail");
        assert!(format!("{err}").contains("span"), "unexpected error: {err}");
    }

    /// An expert count that disagrees with `rows × top_k` is an error.
    #[test]
    fn expert_count_mismatch_is_an_error() {
        let (dim, mi) = (256usize, 256usize);
        let q = CpuQuant::Mq4G256;
        let gu = weight_bytes(q, 2 * mi, dim, 1);
        let dn = weight_bytes(q, dim, mi, 2);
        let experts = vec![CpuMoeExpert {
            gate_up_quant: q,
            gate_up: &gu,
            down_quant: q,
            down: &dn,
        }];
        let mut scratch = MoeScratch::default();
        // rows=2, top_k=1 needs 2 experts; only 1 supplied.
        assert!(moe_cpu_experts_host(
            dim,
            mi,
            2,
            1,
            &experts,
            &vec![0.0f32; 2 * dim],
            &[id_bits(0), id_bits(0)],
            &[1.0, 1.0],
            &mut vec![0.0f32; 2 * dim],
            &mut scratch,
            &mut MoeStepTiming::default(),
            false,
        )
        .is_err());
    }

    /// Flash-Next's shape: gate_up in qt 44 (`k = dim`), down in qt 53 with
    /// `k = mi = 640` — not a multiple of 256, rotated with the 128-wide
    /// transform. The splice must equal the per-expert reference built from
    /// `gemv` + `rotate_x_128_v2` bit for bit, and agree with the original-basis
    /// product (dequantized, inverse-rotated weights · the *unrotated* hidden),
    /// which a wrong or missing down rotation would not.
    #[test]
    fn qt53_down_k640_uses_the_g128_rotation() {
        assert_eq!(cpu_quant_for(DType::MQ4G128V2), Some(CpuQuant::Mq4G128V2));
        assert!(CPU_EXECUTED_DTYPES.contains(&DType::MQ4G128V2));
        let (dim, mi) = (256usize, 640usize);
        let (rows, top_k, n_exp) = (2usize, 3usize, 3usize);
        let (gq, dq) = (CpuQuant::Mq4G256V2, CpuQuant::Mq4G128V2);
        let gu: Vec<Vec<u8>> = (0..n_exp)
            .map(|e| weight_bytes(gq, 2 * mi, dim, 500 + e))
            .collect();
        let dn: Vec<Vec<u8>> = (0..n_exp)
            .map(|e| weight_bytes(dq, dim, mi, 600 + e))
            .collect();
        let idx: [usize; 6] = [0, 1, 2, 2, 0, 1];
        let wts: [f32; 6] = [0.4, 0.35, 0.25, 0.5, 0.3, 0.2];
        let x = x_of(rows * dim);
        let seed: Vec<f32> = (0..rows * dim).map(|i| 0.05 * (i % 5) as f32).collect();
        let ti: Vec<f32> = idx.iter().map(|&e| id_bits(e)).collect();
        let experts: Vec<CpuMoeExpert> = (0..rows * top_k)
            .map(|s| CpuMoeExpert {
                gate_up_quant: gq,
                gate_up: &gu[idx[s]],
                down_quant: dq,
                down: &dn[idx[s]],
            })
            .collect();
        let mut scratch = MoeScratch::default();
        let mut acc = seed.clone();
        moe_cpu_experts_host(
            dim,
            mi,
            rows,
            top_k,
            &experts,
            &x,
            &ti,
            &wts,
            &mut acc,
            &mut scratch,
            &mut MoeStepTiming::default(),
            false,
        )
        .unwrap();

        let mut reference = seed.clone();
        let mut original_basis = seed.clone();
        // Σ |w||h| per element: the f32 rounding scale. The fixture's hidden is
        // large with heavy cancellation, so the tolerance is relative to this,
        // not to the (small) result.
        let mut mag = vec![0.0f64; rows * dim];
        for t in 0..rows {
            for j in 0..top_k {
                let s = t * top_k + j;
                let e = idx[s];
                let mut gv = vec![0.0f32; 2 * mi];
                gemv(gq, &gu[e], 2 * mi, dim, &x[t * dim..(t + 1) * dim], &mut gv);
                let mut hid = vec![0.0f32; mi];
                silu_mul(&gv, &mut hid);
                let unrotated = hid.clone();
                rotate_x_128_v2(&mut hid);
                let mut dout = vec![0.0f32; dim];
                gemv(dq, &dn[e], dim, mi, &hid, &mut dout);
                for (a, v) in reference[t * dim..(t + 1) * dim].iter_mut().zip(dout) {
                    *a += wts[s] * v;
                }
                // Original-basis product: inverse-rotated weights · unrotated hidden.
                let rb = hipfire_cpu::gemv::row_bytes(dq, mi);
                for r in 0..dim {
                    let mut dot = 0.0f64;
                    let mut abs = 0.0f64;
                    for g in 0..mi / 128 {
                        let mut w = [0.0f32; 128];
                        hipfire_cpu::quant::dequant_group(dq, &dn[e][r * rb + g * 68..], &mut w);
                        for i in 0..128 {
                            let term = w[i] as f64 * unrotated[g * 128 + i] as f64;
                            dot += term;
                            abs += term.abs();
                        }
                    }
                    original_basis[t * dim + r] += wts[s] * dot as f32;
                    mag[t * dim + r] += wts[s] as f64 * abs;
                }
            }
        }
        assert_eq!(acc, reference, "qt53 splice != per-expert gemv reference");
        for (i, (a, b)) in acc.iter().zip(&original_basis).enumerate() {
            let tol = 1e-4 * (mag[i] as f32).max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "element {i}: rotated-basis {a} vs original-basis {b} (tol {tol})"
            );
        }
    }

    /// `bf16_rt` is the GPU `rt`: ties-to-even on the dropped 16 bits, sign and
    /// signed zero preserved, ±Inf and NaN (payload included) passed through, and
    /// a finite value past the BF16 maximum carries into ±Inf.
    #[test]
    fn bf16_rt_rounds_ties_to_even_and_passes_nonfinite() {
        let rt_bits = |bits: u32| bf16_rt(f32::from_bits(bits)).to_bits();
        // Exactly half way: even kept mantissa stays, odd kept mantissa rounds up.
        assert_eq!(rt_bits(0x3f80_8000), 0x3f80_0000);
        assert_eq!(rt_bits(0x3f81_8000), 0x3f82_0000);
        // Just below / above half way.
        assert_eq!(rt_bits(0x3f80_7fff), 0x3f80_0000);
        assert_eq!(rt_bits(0x3f80_8001), 0x3f81_0000);
        // Negative mirrors positive (magnitude rounding, sign kept).
        assert_eq!(rt_bits(0xbf80_8000), 0xbf80_0000);
        assert_eq!(rt_bits(0xbf81_8000), 0xbf82_0000);
        assert_eq!(rt_bits(0xbf80_8001), 0xbf81_0000);
        // Signed zero, and a negative denormal that rounds to -0.
        assert_eq!(rt_bits(0x0000_0000), 0x0000_0000);
        assert_eq!(rt_bits(0x8000_0000), 0x8000_0000);
        assert_eq!(rt_bits(0x8000_0001), 0x8000_0000);
        // Already-BF16 values are fixed points.
        assert_eq!(rt_bits(0x4049_0000), 0x4049_0000);
        // Non-finite: unchanged bit for bit.
        for bits in [
            0x7f80_0000u32,
            0xff80_0000,
            0x7fc0_0000,
            0x7f80_0001,
            0xffc1_2345,
        ] {
            assert_eq!(rt_bits(bits), bits);
        }
        // Largest finite f32 rounds up into +Inf / -Inf.
        assert_eq!(bf16_rt(f32::MAX), f32::INFINITY);
        assert_eq!(bf16_rt(f32::MIN), f32::NEG_INFINITY);
    }

    /// Scalar reference for the BF16 route: plain `gemv` projections with explicit
    /// BF16 boundaries — rt(gate), rt(up), rt(activated hidden) before the FWHT,
    /// the rotated down input left f32, rt(residual) at the start of each row, and
    /// per step `acc = rt(acc + rt(rt(down) * rt(weight)))`. `rank_order` and
    /// `round_rotated` switch in the two WRONG variants the real route must not
    /// match: routing-rank combine order and a BF16 round of the rotated hidden.
    #[allow(clippy::too_many_arguments)]
    fn bf16_reference(
        q: CpuQuant,
        gu: &[Vec<u8>],
        dn: &[Vec<u8>],
        idx: &[usize],
        wts: &[f32],
        x: &[f32],
        seed: &[f32],
        (dim, mi, rows, top_k): (usize, usize, usize, usize),
        rank_order: bool,
        round_rotated: bool,
    ) -> Vec<f32> {
        let mut out: Vec<f32> = seed.iter().map(|&v| bf16_rt(v)).collect();
        for t in 0..rows {
            let mut order: Vec<usize> = (0..top_k).collect();
            if !rank_order {
                order.sort_by_key(|&j| (idx[t * top_k + j], j));
            }
            for j in order {
                let s = t * top_k + j;
                let e = idx[s];
                let mut gv = vec![0.0f32; 2 * mi];
                gemv(q, &gu[e], 2 * mi, dim, &x[t * dim..(t + 1) * dim], &mut gv);
                for g in gv.iter_mut() {
                    *g = bf16_rt(*g);
                }
                let mut hid = vec![0.0f32; mi];
                silu_mul(&gv, &mut hid);
                for h in hid.iter_mut() {
                    *h = bf16_rt(*h);
                }
                if q.is_fwht_g256() {
                    rotate_x(&mut hid);
                }
                if round_rotated {
                    for h in hid.iter_mut() {
                        *h = bf16_rt(*h);
                    }
                }
                let mut dout = vec![0.0f32; dim];
                gemv(q, &dn[e], dim, mi, &hid, &mut dout);
                let w = bf16_rt(wts[s]);
                for (a, v) in out[t * dim..(t + 1) * dim].iter_mut().zip(dout) {
                    *a = bf16_rt(*a + bf16_rt(bf16_rt(v) * w));
                }
            }
        }
        out
    }

    /// The `bf16_round_trip = true` host path equals the scalar BF16-boundary
    /// reference bit for bit, on a window whose routing ranks are NOT in expert-id
    /// order (including a repeated id in one row, which must tie-break by rank).
    /// The two wrong variants — routing-rank combine order, and rounding the
    /// rotated hidden — must differ, and the false route must differ from true
    /// while staying plain f32.
    #[test]
    fn bf16_route_matches_scalar_reference_with_expert_id_combine_order() {
        let (dim, mi) = (256usize, 256usize);
        let (rows, top_k, n_exp) = (2usize, 3usize, 4usize);
        let q = CpuQuant::Mq4G256;
        let gu: Vec<Vec<u8>> = (0..n_exp)
            .map(|e| weight_bytes(q, 2 * mi, dim, 700 + e))
            .collect();
        let dn: Vec<Vec<u8>> = (0..n_exp)
            .map(|e| weight_bytes(q, dim, mi, 800 + e))
            .collect();
        // Row 0 ranks are 3, 0, 2 (descending/unsorted); row 1 repeats id 2.
        let idx: [usize; 6] = [3, 0, 2, 2, 2, 1];
        let wts: [f32; 6] = [0.4, 0.35, 0.25, 0.5, 0.3, 0.2];
        let x = x_of(rows * dim);
        // Not BF16-exact, so the initial rt(residual) is observable.
        let seed: Vec<f32> = (0..rows * dim)
            .map(|i| 0.1234567 * (i % 7) as f32 + 1.0001)
            .collect();
        let ti: Vec<f32> = idx.iter().map(|&e| id_bits(e)).collect();
        let experts: Vec<CpuMoeExpert> = (0..rows * top_k)
            .map(|s| CpuMoeExpert {
                gate_up_quant: q,
                gate_up: &gu[idx[s]],
                down_quant: q,
                down: &dn[idx[s]],
            })
            .collect();
        let run = |bf16: bool| {
            let mut scratch = MoeScratch::default();
            let mut acc = seed.clone();
            moe_cpu_experts_host(
                dim,
                mi,
                rows,
                top_k,
                &experts,
                &x,
                &ti,
                &wts,
                &mut acc,
                &mut scratch,
                &mut MoeStepTiming::default(),
                bf16,
            )
            .unwrap();
            acc
        };
        let geom = (dim, mi, rows, top_k);
        let reference = bf16_reference(q, &gu, &dn, &idx, &wts, &x, &seed, geom, false, false);
        let acc = run(true);
        assert_eq!(
            acc, reference,
            "bf16 splice != scalar BF16-boundary reference"
        );
        assert!(
            acc.iter().all(|v| bf16_rt(*v) == *v),
            "every stored element must be BF16-exact"
        );
        let rank_order = bf16_reference(q, &gu, &dn, &idx, &wts, &x, &seed, geom, true, false);
        assert_ne!(acc, rank_order, "fixture is not sensitive to combine order");
        let round_rotated = bf16_reference(q, &gu, &dn, &idx, &wts, &x, &seed, geom, false, true);
        assert_ne!(
            acc, round_rotated,
            "rotated hidden must not be rounded before the down GEMV"
        );
        assert_ne!(run(false), acc, "true and false routes must differ");
    }

    // ── sparse host-slot arithmetic (D2 split) ─────────────────────────────

    const HS_DIM: usize = 256;
    const HS_MI: usize = 128;
    /// QT53's down format: 128-element groups admit `mi = 128`.
    const HS_Q: CpuQuant = CpuQuant::Mq4G128V2;
    /// A quiet-NaN payload standing in for stale scratch.
    const STALE_BITS: u32 = 0x7fc0_dead;

    fn stale(n: usize) -> Vec<f32> {
        vec![f32::from_bits(STALE_BITS); n]
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn all_stale(v: &[f32]) -> bool {
        v.iter().all(|x| x.to_bits() == STALE_BITS)
    }

    struct HostFixture {
        gu: Vec<Vec<u8>>,
        dn: Vec<Vec<u8>>,
    }

    fn host_fixture(n_exp: usize) -> HostFixture {
        HostFixture {
            gu: (0..n_exp)
                .map(|e| weight_bytes(HS_Q, 2 * HS_MI, HS_DIM, 5000 + 131 * e))
                .collect(),
            dn: (0..n_exp)
                .map(|e| weight_bytes(HS_Q, HS_DIM, HS_MI, 9000 + 257 * e))
                .collect(),
        }
    }

    impl HostFixture {
        fn expert(&self, id: i32) -> CpuMoeExpert<'_> {
            CpuMoeExpert {
                gate_up_quant: HS_Q,
                gate_up: &self.gu[id as usize],
                down_quant: HS_Q,
                down: &self.dn[id as usize],
            }
        }

        fn spans(&self, ids: &[i32], host: &[u32]) -> Vec<CpuMoeExpert<'_>> {
            host.iter().map(|&s| self.expert(ids[s as usize])).collect()
        }
    }

    /// The host route with every host slot masked to the `-1` sentinel.
    fn masked_route(ids: &[i32], host: &[u32]) -> Vec<i32> {
        let mut g = ids.to_vec();
        for &s in host {
            g[s as usize] = -1;
        }
        g
    }

    /// Run the host-slot entry: `pool = None` through the public API (the
    /// splice pool), `Some` through the same validated arithmetic in a pool of
    /// a chosen worker count.
    #[allow(clippy::too_many_arguments)]
    fn hs_run(
        pool: Option<&rayon::ThreadPool>,
        rows: usize,
        top_k: usize,
        n_exp: usize,
        ids: &[i32],
        gpu_ids: Option<&[i32]>,
        host: &[u32],
        experts: &[CpuMoeExpert<'_>],
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), DispatchError> {
        match pool {
            None => moe_cpu_host_slot_outputs(
                HS_DIM, HS_MI, rows, top_k, n_exp, ids, gpu_ids, host, experts, x, false, out,
            ),
            Some(pool) => {
                validate_moe_cpu_host_slot_outputs(
                    HS_DIM,
                    HS_MI,
                    rows,
                    top_k,
                    n_exp,
                    ids,
                    gpu_ids,
                    host,
                    experts,
                    x.len(),
                    false,
                    out.len(),
                )?;
                if host.is_empty() {
                    return Ok(());
                }
                let mut scratch = MoeScratch::default();
                pool.install(|| {
                    host_slot_outputs_core(
                        HS_DIM,
                        HS_MI,
                        top_k,
                        ids,
                        host,
                        experts,
                        x,
                        &mut scratch,
                        out,
                        None,
                    )
                })
            }
        }
    }
    /// The crate-private timed entry is byte-equal to the public entry and
    /// rejects the same bad plans with the same errors, over the existing
    /// fixture. Inclusive totals overlap their disjoint children by design:
    /// GU/DN equal prep + exec exactly, every other child sum reconciles
    /// against its inclusive total with slack.
    #[test]
    fn traced_host_slot_entry_is_byte_equal_and_validates_like_public() {
        let (rows, top_k, n_exp) = (2usize, 3usize, 5usize);
        let fx = host_fixture(n_exp);
        let slots = rows * top_k;
        let ids: Vec<i32> = vec![4, 1, 3, 1, 0, 2];
        let host: Vec<u32> = vec![0, 2, 3, 5];
        let gpu = masked_route(&ids, &host);
        let experts = fx.spans(&ids, &host);
        let x = x_of(rows * HS_DIM);
        let mut out_plain = stale(slots * HS_DIM);
        let mut out_traced = stale(slots * HS_DIM);
        moe_cpu_host_slot_outputs(
            HS_DIM,
            HS_MI,
            rows,
            top_k,
            n_exp,
            &ids,
            Some(&gpu),
            &host,
            &experts,
            &x,
            false,
            &mut out_plain,
        )
        .unwrap();
        let trace_epoch = Instant::now();
        let trace = moe_cpu_host_slot_outputs_traced(
            HS_DIM,
            HS_MI,
            rows,
            top_k,
            n_exp,
            &ids,
            Some(&gpu),
            &host,
            &experts,
            &x,
            false,
            &mut out_traced,
            trace_epoch,
        )
        .unwrap();
        assert_eq!(
            bits(&out_plain),
            bits(&out_traced),
            "traced entry bytes != public entry bytes"
        );
        assert_eq!(trace.gu_ns, trace.gu_prep_ns + trace.gu_exec_ns);
        assert_eq!(trace.dn_ns, trace.dn_prep_ns + trace.dn_exec_ns);
        assert!(trace.gather_ns + trace.silu_ns + trace.fwht_ns > 0);
        assert!(
            trace.group_ns + trace.gu_ns + trace.middle_ns + trace.dn_ns + trace.scatter_ns
                <= trace.pool_body_ns
        );
        assert!(trace.pool_body_ns <= trace.pool_install_ns);
        assert!(trace.validate_ns + trace.pool_install_ns <= trace.cpu_total_ns);
        let middle_start = trace.middle_start_ns.expect("nonempty middle start");
        let middle_end = trace.middle_end_ns.expect("nonempty middle end");
        let scatter_start = trace.scatter_start_ns.expect("nonempty scatter start");
        let scatter_end = trace.scatter_end_ns.expect("nonempty scatter end");
        assert!(middle_start <= middle_end);
        assert!(middle_end <= scatter_start);
        assert!(scatter_start <= scatter_end);
        assert!(scatter_end <= elapsed_ns(trace_epoch));
        // Host slots name experts {4, 3, 1, 2}: four distinct groups.
        assert_eq!(trace.unique_experts, 4);
        // The same bad plan fails the same way, leaving `out` untouched.
        let dup = [0u32, 0];
        let mut dup_plain = stale(slots * HS_DIM);
        let mut dup_traced = stale(slots * HS_DIM);
        let plain_err = moe_cpu_host_slot_outputs(
            HS_DIM,
            HS_MI,
            rows,
            top_k,
            n_exp,
            &ids,
            Some(&gpu),
            &dup,
            &fx.spans(&ids, &dup),
            &x,
            false,
            &mut dup_plain,
        )
        .map(|_| ())
        .expect_err("duplicate host slot must fail");
        let traced_err = moe_cpu_host_slot_outputs_traced(
            HS_DIM,
            HS_MI,
            rows,
            top_k,
            n_exp,
            &ids,
            Some(&gpu),
            &dup,
            &fx.spans(&ids, &dup),
            &x,
            false,
            &mut dup_traced,
            Instant::now(),
        )
        .map(|_| ())
        .expect_err("duplicate host slot must fail");
        assert_eq!(format!("{plain_err}"), format!("{traced_err}"));
        assert!(all_stale(&dup_plain) && all_stale(&dup_traced));
        // Exact-single-slot unit case (no env, no global state): one host
        // slot naming expert 2, one group, hierarchy exact where required.
        let one_ids: Vec<i32> = vec![2, 0, 1, 0, 1, 2];
        let one_host: Vec<u32> = vec![0];
        let one_gpu = masked_route(&one_ids, &one_host);
        let one_experts = fx.spans(&one_ids, &one_host);
        let mut one_out = stale(slots * HS_DIM);
        let one = moe_cpu_host_slot_outputs_traced(
            HS_DIM,
            HS_MI,
            rows,
            top_k,
            n_exp,
            &one_ids,
            Some(&one_gpu),
            &one_host,
            &one_experts,
            &x,
            false,
            &mut one_out,
            Instant::now(),
        )
        .unwrap();
        assert_eq!(one.unique_experts, 1);
        assert_eq!(one.gu_ns, one.gu_prep_ns + one.gu_exec_ns);
        assert_eq!(one.dn_ns, one.dn_prep_ns + one.dn_exec_ns);
        // Empty host set: valid plan, writes nothing, zero child stages.
        let empty_host: Vec<u32> = vec![];
        let empty_experts: Vec<CpuMoeExpert<'_>> = vec![];
        let mut empty_out = stale(slots * HS_DIM);
        let empty = moe_cpu_host_slot_outputs_traced(
            HS_DIM,
            HS_MI,
            rows,
            top_k,
            n_exp,
            &ids,
            Some(&ids),
            &empty_host,
            &empty_experts,
            &x,
            false,
            &mut empty_out,
            Instant::now(),
        )
        .unwrap();
        assert_eq!(empty.unique_experts, 0);
        assert_eq!(empty.group_ns, 0);
        assert_eq!(empty.gu_ns, 0);
        assert_eq!(empty.middle_ns, 0);
        assert_eq!(empty.dn_ns, 0);
        assert_eq!(empty.scatter_ns, 0);
        assert_eq!(empty.pool_body_ns, 0);
        assert_eq!(empty.middle_start_ns, None);
        assert_eq!(empty.middle_end_ns, None);
        assert_eq!(empty.scatter_start_ns, None);
        assert_eq!(empty.scatter_end_ns, None);
        assert!(all_stale(&empty_out));
    }

    /// Scalar per-slot reference: plain `gemv` projections with the GPU's BF16
    /// points (rt(gate), rt(up), rt(activated hidden) before the FWHT) and RAW
    /// F32 down output.
    fn slot_reference(fx: &HostFixture, id: i32, x_row: &[f32]) -> Vec<f32> {
        let e = id as usize;
        let mut gv = vec![0.0f32; 2 * HS_MI];
        gemv(HS_Q, &fx.gu[e], 2 * HS_MI, HS_DIM, x_row, &mut gv);
        for g in gv.iter_mut() {
            *g = bf16_rt(*g);
        }
        let mut hid = vec![0.0f32; HS_MI];
        silu_mul(&gv, &mut hid);
        for h in hid.iter_mut() {
            *h = bf16_rt(*h);
        }
        match HS_Q.input_rotation() {
            InputRotation::FwhtG128V2 => rotate_x_128_v2(&mut hid),
            InputRotation::FwhtG256 => rotate_x(&mut hid),
            InputRotation::None => {}
        }
        let mut dout = vec![0.0f32; HS_DIM];
        gemv(HS_Q, &fx.dn[e], HS_DIM, HS_MI, &hid, &mut dout);
        dout
    }

    /// `rows` tokens of `top_k` slots (`ids`/`w` are `[rows * top_k]`, `x` is
    /// `[rows, HS_DIM]`) under the given host/GPU slot masks (bit `s` = slot
    /// `s` is cold/host). The host values are installed into the GPU expanded
    /// layout (masked and invalid slots hold the kernel's zeros, then invalid
    /// slots are NaN-poisoned) and the canonical combine runs ONCE over the
    /// window with the original route. Every row's bytes must equal BOTH the
    /// combine over the unsplit window AND the one-row AR combine (`rows = 1`
    /// host-slot values + combine of that row alone) — the row-invariance
    /// contract D2 admits widths 2..4 under. Returns whether a two-partial
    /// (GPU arm first, then host arm) accumulation ever differed, which proves
    /// the fixture can see an ordering error.
    fn run_mask_case_rows(
        rows: usize,
        top_k: usize,
        ids: &[i32],
        w: &[f32],
        x: &[f32],
        masks: &[u64],
    ) -> bool {
        let (slots, n_exp) = (rows * top_k, 12usize);
        assert!(slots <= 640 && ids.len() == slots && w.len() == slots && x.len() == rows * HS_DIM);
        let fx = host_fixture(n_exp);
        let valid = |s: usize| route_id_valid(ids[s], n_exp);
        let range = |s: usize| s * HS_DIM..(s + 1) * HS_DIM;
        let every: Vec<u32> = (0..slots as u32).filter(|&s| valid(s as usize)).collect();
        let mut full = stale(slots * HS_DIM);
        let full_route = masked_route(ids, &every);
        hs_run(
            None,
            rows,
            top_k,
            n_exp,
            ids,
            Some(&full_route),
            &every,
            &fx.spans(ids, &every),
            x,
            &mut full,
        )
        .unwrap();
        for s in 0..slots {
            if valid(s) {
                let row = s / top_k;
                assert_eq!(
                    bits(&full[range(s)]),
                    bits(&slot_reference(
                        &fx,
                        ids[s],
                        &x[row * HS_DIM..(row + 1) * HS_DIM]
                    )),
                    "slot {s} != scalar reference"
                );
            } else {
                assert!(all_stale(&full[range(s)]), "invalid slot {s} was written");
            }
        }
        let residual0: Vec<f32> = (0..HS_DIM).map(|c| 1.0 + 0.013 * c as f32).collect();
        let mut reference = vec![0.0f32; rows * HS_DIM];
        for t in 0..rows {
            reference[t * HS_DIM..(t + 1) * HS_DIM].copy_from_slice(&residual0);
        }
        moe_canonical_combine_reference(HS_DIM, rows, top_k, n_exp, ids, w, &full, &mut reference);
        // One-row AR per token: its own host-slot run and its own combine.
        let mut ar = vec![0.0f32; rows * HS_DIM];
        for t in 0..rows {
            let r_ids = &ids[t * top_k..(t + 1) * top_k];
            let r_host: Vec<u32> = (0..top_k as u32)
                .filter(|&s| route_id_valid(r_ids[s as usize], n_exp))
                .collect();
            let mut r_full = stale(top_k * HS_DIM);
            hs_run(
                None,
                1,
                top_k,
                n_exp,
                r_ids,
                None,
                &r_host,
                &fx.spans(r_ids, &r_host),
                &x[t * HS_DIM..(t + 1) * HS_DIM],
                &mut r_full,
            )
            .unwrap();
            for &s in &r_host {
                assert_eq!(
                    bits(&r_full[s as usize * HS_DIM..(s as usize + 1) * HS_DIM]),
                    bits(&full[range(t * top_k + s as usize)]),
                    "row {t} slot {s}: window host value != one-row AR"
                );
            }
            let mut row = residual0.clone();
            moe_canonical_combine_reference(
                HS_DIM,
                1,
                top_k,
                n_exp,
                r_ids,
                &w[t * top_k..(t + 1) * top_k],
                &r_full,
                &mut row,
            );
            ar[t * HS_DIM..(t + 1) * HS_DIM].copy_from_slice(&row);
            assert_eq!(
                bits(&row),
                bits(&reference[t * HS_DIM..(t + 1) * HS_DIM]),
                "row {t}: one-row AR combine != window combine of the unsplit values"
            );
        }
        let mut partial_differs = false;
        for &mask in masks {
            let host: Vec<u32> = (0..slots as u32)
                .filter(|&s| (mask >> s) & 1 == 1 && valid(s as usize))
                .collect();
            let gpu_route = masked_route(ids, &host);
            let mut host_out = stale(slots * HS_DIM);
            hs_run(
                None,
                rows,
                top_k,
                n_exp,
                ids,
                Some(&gpu_route),
                &host,
                &fx.spans(ids, &host),
                x,
                &mut host_out,
            )
            .unwrap();
            // GPU arm: valid unmasked slots carry their value; masked and
            // invalid slots are the kernel's zero stores.
            let mut expanded = vec![0.0f32; slots * HS_DIM];
            for s in 0..slots {
                if valid(s) && !host.contains(&(s as u32)) {
                    expanded[range(s)].copy_from_slice(&full[range(s)]);
                }
            }
            // Slot install, after the GPU down.
            for s in 0..slots {
                if host.contains(&(s as u32)) {
                    assert_eq!(
                        bits(&host_out[range(s)]),
                        bits(&full[range(s)]),
                        "mask {mask:#b}: host slot {s} differs from its unsplit value"
                    );
                    expanded[range(s)].copy_from_slice(&host_out[range(s)]);
                } else {
                    assert!(
                        all_stale(&host_out[range(s)]),
                        "mask {mask:#b}: non-host slot {s} was written"
                    );
                }
            }
            for s in 0..slots {
                if !valid(s) {
                    expanded[range(s)].fill(f32::NAN);
                }
            }
            let mut got = vec![0.0f32; rows * HS_DIM];
            for t in 0..rows {
                got[t * HS_DIM..(t + 1) * HS_DIM].copy_from_slice(&residual0);
            }
            moe_canonical_combine_reference(
                HS_DIM, rows, top_k, n_exp, ids, w, &expanded, &mut got,
            );
            assert_eq!(
                bits(&got),
                bits(&reference),
                "mask {mask:#b}: split join != unsplit"
            );
            assert_eq!(
                bits(&got),
                bits(&ar),
                "mask {mask:#b}: split join != per-row one-row AR"
            );

            // Two-partial: the GPU arm reduces first (host slots masked to -1),
            // then the host arm reduces over its result.
            let mut host_only = ids.to_vec();
            for s in 0..slots {
                if !host.contains(&(s as u32)) {
                    host_only[s] = -1;
                }
            }
            let mut two = vec![0.0f32; rows * HS_DIM];
            for t in 0..rows {
                two[t * HS_DIM..(t + 1) * HS_DIM].copy_from_slice(&residual0);
            }
            moe_canonical_combine_reference(
                HS_DIM, rows, top_k, n_exp, &gpu_route, w, &expanded, &mut two,
            );
            moe_canonical_combine_reference(
                HS_DIM, rows, top_k, n_exp, &host_only, w, &expanded, &mut two,
            );
            if bits(&two) != bits(&reference) {
                partial_differs = true;
            }
        }
        partial_differs
    }

    /// One ten-slot, one-row route under all 1024 host/GPU masks.
    fn run_mask_case(ids: [i32; 10], w: [f32; 10]) -> bool {
        let masks: Vec<u64> = (0..1024).collect();
        run_mask_case_rows(1, 10, &ids, &w, &x_of(HS_DIM), &masks)
    }

    /// Wide oracle: `masks` are bit-vectors over slots (1 = cold/host), one
    /// `u64` word per 64 slots, so the full 640-slot logical window is covered
    /// without enumerating 2^640 subsets. Same invariant as
    /// [`run_mask_case_rows`]: every row's joined bytes equal the one-row AR of
    /// that token.
    fn run_mask_case_rows_wide(
        rows: usize,
        top_k: usize,
        ids: &[i32],
        w: &[f32],
        x: &[f32],
        masks: &[Vec<u64>],
    ) {
        let (slots, n_exp) = (rows * top_k, 12usize);
        assert!(slots <= 640 && ids.len() == slots && w.len() == slots && x.len() == rows * HS_DIM);
        let fx = host_fixture(n_exp);
        let valid = |s: usize| route_id_valid(ids[s], n_exp);
        let is_host = |mask: &[u64], s: usize| (mask[s / 64] >> (s % 64)) & 1 == 1;
        let range = |s: usize| s * HS_DIM..(s + 1) * HS_DIM;
        let every: Vec<u32> = (0..slots as u32).filter(|&s| valid(s as usize)).collect();
        let mut full = stale(slots * HS_DIM);
        let full_route = masked_route(ids, &every);
        hs_run(
            None,
            rows,
            top_k,
            n_exp,
            ids,
            Some(&full_route),
            &every,
            &fx.spans(ids, &every),
            x,
            &mut full,
        )
        .unwrap();
        // Every slot equals the scalar per-slot reference; invalid slots stay stale.
        for s in 0..slots {
            if valid(s) {
                let row = s / top_k;
                assert_eq!(
                    bits(&full[range(s)]),
                    bits(&slot_reference(
                        &fx,
                        ids[s],
                        &x[row * HS_DIM..(row + 1) * HS_DIM]
                    )),
                    "wide slot {s} != scalar reference"
                );
            } else {
                assert!(
                    all_stale(&full[range(s)]),
                    "wide invalid slot {s} was written"
                );
            }
        }
        let residual0: Vec<f32> = (0..HS_DIM).map(|c| 1.0 + 0.013 * c as f32).collect();
        let mut reference = vec![0.0f32; rows * HS_DIM];
        for t in 0..rows {
            reference[t * HS_DIM..(t + 1) * HS_DIM].copy_from_slice(&residual0);
        }
        moe_canonical_combine_reference(HS_DIM, rows, top_k, n_exp, ids, w, &full, &mut reference);
        // One-row AR per token: its own host-slot run and its own combine.
        let mut ar = vec![0.0f32; rows * HS_DIM];
        for t in 0..rows {
            let r_ids = &ids[t * top_k..(t + 1) * top_k];
            let r_host: Vec<u32> = (0..top_k as u32)
                .filter(|&s| route_id_valid(r_ids[s as usize], n_exp))
                .collect();
            let mut r_full = stale(top_k * HS_DIM);
            hs_run(
                None,
                1,
                top_k,
                n_exp,
                r_ids,
                None,
                &r_host,
                &fx.spans(r_ids, &r_host),
                &x[t * HS_DIM..(t + 1) * HS_DIM],
                &mut r_full,
            )
            .unwrap();
            for &s in &r_host {
                assert_eq!(
                    bits(&r_full[s as usize * HS_DIM..(s as usize + 1) * HS_DIM]),
                    bits(&full[range(t * top_k + s as usize)]),
                    "wide row {t} slot {s}: window host value != one-row AR"
                );
            }
            let mut row = residual0.clone();
            moe_canonical_combine_reference(
                HS_DIM,
                1,
                top_k,
                n_exp,
                r_ids,
                &w[t * top_k..(t + 1) * top_k],
                &r_full,
                &mut row,
            );
            ar[t * HS_DIM..(t + 1) * HS_DIM].copy_from_slice(&row);
            assert_eq!(
                bits(&row),
                bits(&reference[t * HS_DIM..(t + 1) * HS_DIM]),
                "wide row {t}: one-row AR combine != window combine of the unsplit values"
            );
        }
        // Split join under every wide mask: host slots installed into the GPU
        // expanded layout, invalid slots NaN-poisoned, joined once over the
        // window; must equal BOTH the unsplit combine and the per-row AR.
        for mask in masks {
            let host: Vec<u32> = (0..slots as u32)
                .filter(|&s| is_host(mask, s as usize) && valid(s as usize))
                .collect();
            let gpu_route = masked_route(ids, &host);
            let mut host_out = stale(slots * HS_DIM);
            hs_run(
                None,
                rows,
                top_k,
                n_exp,
                ids,
                Some(&gpu_route),
                &host,
                &fx.spans(ids, &host),
                x,
                &mut host_out,
            )
            .unwrap();
            let mut expanded = vec![0.0f32; slots * HS_DIM];
            for s in 0..slots {
                if valid(s) && !host.contains(&(s as u32)) {
                    expanded[range(s)].copy_from_slice(&full[range(s)]);
                }
            }
            for s in 0..slots {
                if host.contains(&(s as u32)) {
                    assert_eq!(
                        bits(&host_out[range(s)]),
                        bits(&full[range(s)]),
                        "wide mask: host slot {s} differs from its unsplit value"
                    );
                    expanded[range(s)].copy_from_slice(&host_out[range(s)]);
                } else {
                    assert!(
                        all_stale(&host_out[range(s)]),
                        "wide mask: non-host slot {s} was written"
                    );
                }
            }
            for s in 0..slots {
                if !valid(s) {
                    expanded[range(s)].fill(f32::NAN);
                }
            }
            let mut got = vec![0.0f32; rows * HS_DIM];
            for t in 0..rows {
                got[t * HS_DIM..(t + 1) * HS_DIM].copy_from_slice(&residual0);
            }
            moe_canonical_combine_reference(
                HS_DIM, rows, top_k, n_exp, ids, w, &expanded, &mut got,
            );
            assert_eq!(
                bits(&got),
                bits(&reference),
                "wide mask: split join != unsplit"
            );
            assert_eq!(
                bits(&got),
                bits(&ar),
                "wide mask: split join != per-row one-row AR"
            );
        }
    }

    /// Deterministic masks for windows too wide to enumerate; the all-cold,
    /// all-hot and per-row-only / per-row-excluded masks are always included.
    /// Narrow windows (<=64 slots) keep the exact legacy `u64` cover.
    fn sampled_masks(slots: usize, top_k: usize, n: usize) -> Vec<u64> {
        assert!(slots <= 64, "narrow masks cover at most 64 slots");
        let all = if slots == 64 {
            u64::MAX
        } else {
            (1u64 << slots) - 1
        };
        let mut row0 = 0u64;
        for s in 0..top_k {
            row0 |= 1 << s;
        }
        let mut masks = vec![0, all];
        for t in 0..slots / top_k {
            masks.push(row0 << (t * top_k));
            masks.push(all & !(row0 << (t * top_k)));
        }
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        while masks.len() < n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            masks.push((state >> 7) & all);
        }
        masks
    }

    /// Wide deterministic masks: one `u64` word per 64 slots (10 words for the
    /// full 640-slot logical window). The all-cold, all-hot and strided
    /// per-row masks are always included, then the LCG fills to `n` masks;
    /// every mask word beyond `slots` stays zero.
    fn sampled_masks_wide(slots: usize, top_k: usize, n: usize) -> Vec<Vec<u64>> {
        assert!(slots <= 640, "logical window is at most 640 slots");
        let words = slots.div_ceil(64);
        let mut set = |mask: &mut Vec<u64>, s: usize| {
            mask[s / 64] |= 1 << (s % 64);
        };
        let mut full = vec![0u64; words];
        for s in 0..slots {
            set(&mut full, s);
        }
        let rows = slots / top_k;
        // All single-row masks only for narrow windows; wide windows sample a
        // strided subset (row 0, middle, last) to keep the mask count bounded.
        let stride = (rows / 8).max(1);
        let mut masks = vec![vec![0u64; words], full.clone()];
        let mut t = 0usize;
        while t < rows {
            let mut row = vec![0u64; words];
            for s in 0..top_k {
                set(&mut row, (t * top_k + s) % slots);
            }
            let mut excl = full.clone();
            for (e, r) in excl.iter_mut().zip(&row) {
                *e &= !r;
            }
            masks.push(row);
            masks.push(excl);
            t += stride;
        }
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        while masks.len() < n {
            let mut mask = vec![0u64; words];
            for w in 0..words {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                mask[w] = (state >> 7) | (state << 57);
            }
            for s in slots..words * 64 {
                mask[s / 64] &= !(1 << (s % 64));
            }
            masks.push(mask);
        }
        masks
    }

    /// Interleaved ids, a repeated id, cancelling and extreme weights (a ±1e30
    /// pair on one expert, a ±1e37 pair on another, a sub-normal, ±0).
    #[test]
    fn canonical_join_is_byte_identical_for_all_1024_masks() {
        let ids = [7, 2, 9, 2, 8, 5, 0, 5, 3, 4];
        let w = [
            0.5, 1.0e30, 0.35, -1.0e30, 0.3, 1.0e37, 1.0e-38, -1.0e37, 0.0, -0.0,
        ];
        assert!(
            run_mask_case(ids, w),
            "fixture cannot distinguish a two-partial combine from the canonical one"
        );
    }

    /// A sentinel (`-1`) and an out-of-range id: owned by neither arm, their
    /// weights NaN/inf and their expanded ranges NaN, never read.
    #[test]
    fn canonical_join_ignores_sentinel_and_invalid_slots_for_all_1024_masks() {
        let ids = [7, -1, 9, 2, 8, 5, 40, 5, 3, 2];
        let w = [
            0.5,
            f32::NAN,
            0.35,
            1.0e30,
            0.3,
            1.0e37,
            f32::INFINITY,
            -1.0e37,
            0.0,
            -1.0e30,
        ];
        assert!(
            run_mask_case(ids, w),
            "fixture cannot distinguish a two-partial combine from the canonical one"
        );
    }

    /// A 2..4-row, ten-slot-per-row window under sampled host/GPU masks
    /// (per-row cold subsets differ inside one window), distinct and identical
    /// tokens, with and without invalid slots. Rows repeat experts across ranks
    /// and across rows.
    fn multirow_join(rows: usize, n_masks: usize, identical_tokens: bool, with_invalid: bool) {
        let top_k = 10;
        let base_ids: [[i32; 10]; 4] = [
            [7, 2, 9, 2, 8, 5, 0, 5, 3, 4],
            [2, 7, 5, 9, 0, 8, 3, 4, 5, 2],
            [9, 9, 1, 11, 6, 6, 10, 2, 7, 0],
            [4, 3, 5, 0, 8, 9, 2, 7, 2, 5],
        ];
        let base_w: [[f32; 10]; 4] = [
            [
                0.5, 1.0e30, 0.35, -1.0e30, 0.3, 1.0e37, 1.0e-38, -1.0e37, 0.0, -0.0,
            ],
            [
                0.25, -1.0e30, 0.4, 1.0e30, 0.15, -1.0e37, 3.0e-39, 1.0e37, 0.7, 0.01,
            ],
            [
                0.125, 0.3, -0.2, 0.9, 1.0e30, -1.0e30, 0.6, 0.05, 1.0e37, -1.0e37,
            ],
            [0.45, 0.35, 0.2, -0.0, 0.0, 0.33, 0.66, -0.5, 1.0e-30, 2.0],
        ];
        assert!(rows >= 1 && rows <= 4, "narrow harness covers 1..4 rows");
        let mut ids = Vec::new();
        let mut w = Vec::new();
        for t in 0..rows {
            let mut row = base_ids[t];
            let mut wr = base_w[t];
            if with_invalid {
                row[(1 + 3 * t) % 10] = -1;
                wr[(1 + 3 * t) % 10] = f32::NAN;
                row[(4 + 5 * t) % 10] = 40;
                wr[(4 + 5 * t) % 10] = f32::INFINITY;
            }
            ids.extend_from_slice(&row);
            w.extend_from_slice(&wr);
        }
        let mut x = x_of(rows * HS_DIM);
        if identical_tokens {
            for t in 1..rows {
                let (head, tail) = x.split_at_mut(t * HS_DIM);
                tail[..HS_DIM].copy_from_slice(&head[..HS_DIM]);
            }
        }
        let slots = rows * top_k;
        let masks = if rows == 1 {
            (0..1024u64).collect()
        } else {
            sampled_masks(slots, top_k, n_masks)
        };
        assert!(
            run_mask_case_rows(rows, top_k, &ids, &w, &x, &masks),
            "fixture cannot distinguish a two-partial combine from the canonical one"
        );
    }

    #[test]
    fn canonical_join_multirow_windows_match_one_row_ar_for_sampled_masks() {
        for rows in 2..=4usize {
            for identical in [false, true] {
                for invalid in [false, true] {
                    multirow_join(rows, 1024, identical, invalid);
                }
            }
        }
    }

    /// Wide row builder behind [`multirow_join_wide`]: the same interleaved /
    /// repeated / sentinel / extreme-weight fixture, row `t` cycling the four
    /// proven narrow rows so every 1..64 width repeats established cover.
    fn multirow_join_wide(rows: usize, n_masks: usize, identical_tokens: bool, with_invalid: bool) {
        let top_k = 10;
        let base_ids: [[i32; 10]; 4] = [
            [7, 2, 9, 2, 8, 5, 0, 5, 3, 4],
            [2, 7, 5, 9, 0, 8, 3, 4, 5, 2],
            [9, 9, 1, 11, 6, 6, 10, 2, 7, 0],
            [4, 3, 5, 0, 8, 9, 2, 7, 2, 5],
        ];
        let base_w: [[f32; 10]; 4] = [
            [
                0.5, 1.0e30, 0.35, -1.0e30, 0.3, 1.0e37, 1.0e-38, -1.0e37, 0.0, -0.0,
            ],
            [
                0.25, -1.0e30, 0.4, 1.0e30, 0.15, -1.0e37, 3.0e-39, 1.0e37, 0.7, 0.01,
            ],
            [
                0.125, 0.3, -0.2, 0.9, 1.0e30, -1.0e30, 0.6, 0.05, 1.0e37, -1.0e37,
            ],
            [0.45, 0.35, 0.2, -0.0, 0.0, 0.33, 0.66, -0.5, 1.0e-30, 2.0],
        ];
        assert!(rows >= 1 && rows <= 64, "logical window is 1..64 rows");
        let mut ids = Vec::new();
        let mut w = Vec::new();
        for t in 0..rows {
            let mut row = base_ids[t % 4];
            let mut wr = base_w[t % 4];
            if with_invalid {
                row[(1 + 3 * t) % 10] = -1;
                wr[(1 + 3 * t) % 10] = f32::NAN;
                row[(4 + 5 * t) % 10] = 40;
                wr[(4 + 5 * t) % 10] = f32::INFINITY;
            }
            ids.extend_from_slice(&row);
            w.extend_from_slice(&wr);
        }
        let mut x = x_of(rows * HS_DIM);
        if identical_tokens {
            for t in 1..rows {
                let (head, tail) = x.split_at_mut(t * HS_DIM);
                tail[..HS_DIM].copy_from_slice(&head[..HS_DIM]);
            }
        }
        let masks = sampled_masks_wide(rows * top_k, top_k, n_masks);
        run_mask_case_rows_wide(rows, top_k, &ids, &w, &x, &masks);
    }

    /// Logical window 1..64: every width's rows are byte-identical to the
    /// one-row AR of each token (distinct and identical tokens, with and
    /// without invalid slots), over sampled host/GPU masks including the
    /// all-cold, all-hot and per-row masks. Widths 2..=4 repeat the proven
    /// legacy cover; 5..=64 are the new lab-wide storage widths (no device
    /// needed: the same device-free host-slot entry the D2 join runs).
    #[test]
    fn logical_window_1_to_64_matches_one_row_ar() {
        for rows in 1..=64usize {
            // Narrow widths keep the legacy 256-mask density; wide widths use
            // 40 masks (all-cold, all-hot, strided per-row, LCG) — every one of
            // the 64 widths still runs the full three-way byte check.
            let n_masks = if rows <= 8 { 256 } else { 40 };
            for identical in [false, true] {
                for invalid in [false, true] {
                    multirow_join_wide(rows, n_masks, identical, invalid);
                }
            }
        }
    }

    /// Exhaustive 2^10 masks over a two-row, five-slot window (every cold
    /// subset of both rows), with a repeated expert inside and across rows.
    #[test]
    fn canonical_join_two_row_window_is_byte_identical_for_all_1024_masks() {
        let ids = [7, 2, 7, 2, 5, 2, 7, 5, -1, 40];
        let w = [
            0.5,
            1.0e30,
            0.35,
            -1.0e30,
            0.3,
            1.0e37,
            1.0e-38,
            -1.0e37,
            f32::NAN,
            f32::INFINITY,
        ];
        let masks: Vec<u64> = (0..1024).collect();
        for identical in [false, true] {
            let mut x = x_of(2 * HS_DIM);
            if identical {
                let (a, b) = x.split_at_mut(HS_DIM);
                b.copy_from_slice(a);
            }
            assert!(
                run_mask_case_rows(2, 5, &ids, &w, &x, &masks),
                "fixture cannot distinguish a two-partial combine from the canonical one"
            );
        }
    }

    // ── D2 row-invariance oracle across formats ────────────────────────────

    /// An expert pool of shape `(dim, mi)` with separate gate_up (`gq`) and down
    /// (`dq`) formats: the row formats (D2's placement is per expert, a graded
    /// split is its own case in `graded_multirow_matches_per_expert_gemv`) and
    /// the planar E8P-I4 / W3 slab pairs (R256 gate_up, R128 down).
    struct QFix {
        gq: CpuQuant,
        dq: CpuQuant,
        dim: usize,
        mi: usize,
        gu: Vec<Vec<u8>>,
        dn: Vec<Vec<u8>>,
    }

    fn qfix(gq: CpuQuant, dq: CpuQuant, dim: usize, mi: usize, n_exp: usize) -> QFix {
        QFix {
            gq,
            dq,
            dim,
            mi,
            gu: (0..n_exp)
                .map(|e| weight_bytes(gq, 2 * mi, dim, 3100 + 173 * e))
                .collect(),
            dn: (0..n_exp)
                .map(|e| weight_bytes(dq, dim, mi, 7700 + 311 * e))
                .collect(),
        }
    }

    impl QFix {
        fn expert(&self, id: i32) -> CpuMoeExpert<'_> {
            CpuMoeExpert {
                gate_up_quant: self.gq,
                gate_up: &self.gu[id as usize],
                down_quant: self.dq,
                down: &self.dn[id as usize],
            }
        }

        fn run(
            &self,
            rows: usize,
            top_k: usize,
            ids: &[i32],
            host: &[u32],
            x: &[f32],
            out: &mut [f32],
        ) {
            let spans: Vec<_> = host.iter().map(|&s| self.expert(ids[s as usize])).collect();
            moe_cpu_host_slot_outputs(
                self.dim,
                self.mi,
                rows,
                top_k,
                self.gu.len(),
                ids,
                None,
                host,
                &spans,
                x,
                false,
                out,
            )
            .unwrap();
        }

        /// Plain per-slot reference: single-row `gemv` projections with the
        /// GPU's BF16 points and raw F32 down output. The input rotation is the
        /// DOWN quant's contract (as in `moe_slots_compute`).
        fn reference(&self, id: i32, x_row: &[f32]) -> Vec<f32> {
            let e = id as usize;
            let mut gv = vec![0.0f32; 2 * self.mi];
            gemv(self.gq, &self.gu[e], 2 * self.mi, self.dim, x_row, &mut gv);
            for g in gv.iter_mut() {
                *g = bf16_rt(*g);
            }
            let mut hid = vec![0.0f32; self.mi];
            silu_mul(&gv, &mut hid);
            for h in hid.iter_mut() {
                *h = bf16_rt(*h);
            }
            match self.dq.input_rotation() {
                InputRotation::FwhtG128V2 => rotate_x_128_v2(&mut hid),
                InputRotation::FwhtG256 => rotate_x(&mut hid),
                InputRotation::None => {}
            }
            let mut dout = vec![0.0f32; self.dim];
            gemv(self.dq, &self.dn[e], self.dim, self.mi, &hid, &mut dout);
            dout
        }
    }

    /// D2 admits windows of 2..4 rows; each token's host slots (the cold
    /// experts of ITS route) and the canonically joined row must be byte
    /// identical to the one-row AR of that token at every row position, across
    /// formats (cached-Σx MQ4/MQ6, uncached, and the planar E8P-I4 / W3 slab
    /// pairs with a distinct gate_up and down quant), identical vs distinct
    /// tokens, repeated experts across rows, changing cold sets, invalid slots,
    /// accepted prefix replays (narrower windows, run in both orders so a wide
    /// window's scratch is reused by a narrow one) and AR calls in between.
    #[test]
    fn d2_window_rows_are_byte_identical_to_one_row_ar_for_every_position() {
        let (top_k, n_exp) = (3usize, 8usize);
        let scenarios: [[[i32; 3]; 4]; 4] = [
            [[2, 0, 4]; 4],
            [[2, 0, 4], [0, 4, 2], [4, 2, 1], [1, 2, 0]],
            [[2, -1, 4], [40, 0, 4], [2, 0, -1], [-1, -1, 3]],
            [[0, 1, 1], [3, 4, 5], [6, 7, 0], [1, 3, 5]],
        ];
        // Cold experts per call (bit e = expert e is cold).
        let cold_sets = [0xFFu8, 0x55, 0xAA, 0x05, 0x00, 0x24, 0x81, 0x1B];
        let shapes = [
            (CpuQuant::Mq4G128V2, CpuQuant::Mq4G128V2, 256usize, 128usize),
            (CpuQuant::Mq4G256, CpuQuant::Mq4G256, 256, 256),
            (CpuQuant::Mq6G256, CpuQuant::Mq6G256, 256, 256),
            (CpuQuant::Mq4G256V2, CpuQuant::Mq4G256V2, 256, 256),
            // Planar slab formats: gate_up and down use distinct row widths.
            (CpuQuant::E8pI4G128R256, CpuQuant::E8pI4G128R128, 256, 128),
            (CpuQuant::Mq3I4G128R256, CpuQuant::Mq3I4G128R128, 256, 128),
        ];
        for (gq, dq, dim, mi) in shapes {
            let fx = qfix(gq, dq, dim, mi, n_exp);
            let wts: Vec<f32> = (0..12)
                .map(|s| (0.11 + 0.063 * s as f32) * if s % 3 == 1 { -1.0 } else { 1.0 })
                .collect();
            for (si, scenario) in scenarios.iter().enumerate() {
                for identical in [false, true] {
                    let mut x = x_of(4 * dim);
                    if identical {
                        for t in 1..4 {
                            let (head, tail) = x.split_at_mut(t * dim);
                            tail[..dim].copy_from_slice(&head[..dim]);
                        }
                    }
                    for &cold in &cold_sets {
                        let is_host = |id: i32| route_id_valid(id, n_exp) && (cold >> id) & 1 == 1;
                        // One-row AR per token (its own call, its own cold slots).
                        let ar: Vec<(Vec<u32>, Vec<f32>)> = (0..4)
                            .map(|t| {
                                let r_ids = &scenario[t];
                                let host: Vec<u32> = (0..top_k as u32)
                                    .filter(|&s| is_host(r_ids[s as usize]))
                                    .collect();
                                let mut out = stale(top_k * dim);
                                fx.run(
                                    1,
                                    top_k,
                                    r_ids,
                                    &host,
                                    &x[t * dim..(t + 1) * dim],
                                    &mut out,
                                );
                                (host, out)
                            })
                            .collect();
                        // Narrow widths keep the proven legacy cover; 5..8 extend
                        // it one row at a time and 64 is the full logical window
                        // (same row-cycled scenario rows, same one-row AR oracle).
                        let mut window_outs: Vec<(usize, Vec<f32>)> = Vec::new();
                        for width in [2usize, 3, 4, 3, 2, 5, 6, 7, 8, 64] {
                            let ids: Vec<i32> = (0..width)
                                .flat_map(|t| scenario[t % 4].iter().copied())
                                .collect();
                            let host: Vec<u32> = (0..(width * top_k) as u32)
                                .filter(|&s| is_host(ids[s as usize]))
                                .collect();
                            let xw: Vec<f32> = (0..width)
                                .flat_map(|t| x[t % 4 * dim..(t % 4 + 1) * dim].to_vec())
                                .collect();
                            let mut out = stale(width * top_k * dim);
                            fx.run(width, top_k, &ids, &host, &xw, &mut out);
                            let ctx = format!(
                                "{gq:?}/{dq:?} scenario {si} identical={identical} cold={cold:#04x} width={width}"
                            );
                            for t in 0..width {
                                // Row `t` cycles the 4 proven scenario rows; the
                                // one-row AR oracle (`ar`) and scalar reference
                                // index the same cycled row.
                                let cyc = t % 4;
                                for j in 0..top_k {
                                    let s = t * top_k + j;
                                    let range = s * dim..(s + 1) * dim;
                                    if host.contains(&(s as u32)) {
                                        let one = &ar[cyc].1[j * dim..(j + 1) * dim];
                                        assert_eq!(
                                            bits(&out[range.clone()]),
                                            bits(one),
                                            "{ctx}: row {t} rank {j} host value != one-row AR"
                                        );
                                        assert_eq!(
                                            bits(&out[range]),
                                            bits(
                                                &fx.reference(ids[s], &xw[t * dim..(t + 1) * dim])
                                            ),
                                            "{ctx}: row {t} rank {j} != scalar reference"
                                        );
                                    } else {
                                        assert!(all_stale(&out[range]), "{ctx}: slot {s} written");
                                    }
                                }
                            }
                            // Accepted-prefix replay: a narrower window over the
                            // same leading rows reproduces the wider one's bytes,
                            // restricted to widths the legacy harness recorded
                            // (2..4 over the first 4 scenario rows).
                            for (pw, pout) in &window_outs {
                                if width > 4 || *pw > 4 {
                                    continue;
                                }
                                let n = (*pw).min(width) * top_k * dim;
                                assert_eq!(
                                    bits(&out[..n]),
                                    bits(&pout[..n]),
                                    "{ctx}: width {width} prefix != width {pw}"
                                );
                            }
                            window_outs.push((width, out.clone()));

                            // Canonical join: GPU arm values for non-host valid
                            // slots (scalar reference), host slots from the
                            // window, invalid slots NaN; combined over the
                            // window vs per-row one-row AR joins. Weights cycle
                            // with the rows (`wts` covers the 4-row harness).
                            let residual: Vec<f32> =
                                (0..dim).map(|c| 1.0 + 0.013 * c as f32).collect();
                            let mut expanded = vec![0.0f32; width * top_k * dim];
                            let mut ar_expanded = vec![vec![0.0f32; top_k * dim]; width];
                            for t in 0..width {
                                for j in 0..top_k {
                                    let s = t * top_k + j;
                                    let id = ids[s];
                                    let dst = s * dim..(s + 1) * dim;
                                    let r_dst = j * dim..(j + 1) * dim;
                                    if !route_id_valid(id, n_exp) {
                                        expanded[dst].fill(f32::NAN);
                                        ar_expanded[t][r_dst].fill(f32::NAN);
                                    } else if is_host(id) {
                                        expanded[dst.clone()].copy_from_slice(&out[dst]);
                                        ar_expanded[t][r_dst.clone()]
                                            .copy_from_slice(&ar[t % 4].1[r_dst]);
                                    } else {
                                        let v = fx.reference(id, &xw[t * dim..(t + 1) * dim]);
                                        expanded[dst].copy_from_slice(&v);
                                        ar_expanded[t][r_dst].copy_from_slice(&v);
                                    }
                                }
                            }
                            let mut joined = vec![0.0f32; width * dim];
                            for t in 0..width {
                                joined[t * dim..(t + 1) * dim].copy_from_slice(&residual);
                            }
                            // Row-cycled weights for the wide combine.
                            let wts_wide: Vec<f32> = (0..width)
                                .flat_map(|t| {
                                    let cyc = t % 4;
                                    wts[cyc * top_k..(cyc + 1) * top_k].to_vec()
                                })
                                .collect();
                            moe_canonical_combine_reference(
                                dim,
                                width,
                                top_k,
                                n_exp,
                                &ids,
                                &wts_wide,
                                &expanded,
                                &mut joined,
                            );
                            for t in 0..width {
                                let cyc = t % 4;
                                let mut row = residual.clone();
                                moe_canonical_combine_reference(
                                    dim,
                                    1,
                                    top_k,
                                    n_exp,
                                    &scenario[cyc],
                                    &wts[cyc * top_k..(cyc + 1) * top_k],
                                    &ar_expanded[t],
                                    &mut row,
                                );
                                assert_eq!(
                                    bits(&joined[t * dim..(t + 1) * dim]),
                                    bits(&row),
                                    "{ctx}: row {t} joined != one-row AR join"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Same slot ⇒ same bytes, whatever the window grouping, host-slot order,
    /// number of rows per call, or worker count; and they match the scalar
    /// per-slot reference. Rows with no host slot are never read.
    #[test]
    fn host_slot_bytes_independent_of_grouping_order_and_worker_count() {
        let (rows, top_k, n_exp) = (4usize, 3usize, 5usize);
        let fx = host_fixture(n_exp);
        let ids: Vec<i32> = vec![4, 1, 3, 1, 1, 0, 3, 2, 4, 0, 2, 1];
        let x = x_of(rows * HS_DIM);
        let range = |s: usize| s * HS_DIM..(s + 1) * HS_DIM;
        let every: Vec<u32> = (0..12).collect();
        let run = |pool: Option<&rayon::ThreadPool>, host: &[u32], x: &[f32]| {
            let mut out = stale(12 * HS_DIM);
            hs_run(
                pool,
                rows,
                top_k,
                n_exp,
                &ids,
                None,
                host,
                &fx.spans(&ids, host),
                x,
                &mut out,
            )
            .unwrap();
            out
        };
        let full = run(None, &every, &x);
        assert!(full.iter().all(|v| v.is_finite()));
        for s in 0..12 {
            let row = s / top_k;
            assert_eq!(
                bits(&full[range(s)]),
                bits(&slot_reference(
                    &fx,
                    ids[s],
                    &x[row * HS_DIM..(row + 1) * HS_DIM]
                )),
                "slot {s} != scalar reference"
            );
        }
        // Host-slot order.
        let mut reversed = every.clone();
        reversed.reverse();
        let interleaved: Vec<u32> = (0..12u32)
            .filter(|s| s % 2 == 0)
            .chain((0..12u32).filter(|s| s % 2 == 1))
            .collect();
        for order in [&reversed, &interleaved] {
            assert_eq!(
                bits(&run(None, order, &x)),
                bits(&full),
                "host-slot order changed bytes"
            );
        }
        // Each slot alone (group size 1): only that slot is written.
        for s in 0..12u32 {
            let out = run(None, &[s], &x);
            assert_eq!(
                bits(&out[range(s as usize)]),
                bits(&full[range(s as usize)]),
                "slot {s} alone"
            );
            for o in 0..12 {
                if o != s as usize {
                    assert!(
                        all_stale(&out[range(o)]),
                        "slot {o} written by a call for slot {s}"
                    );
                }
            }
        }
        // One row per call.
        for t in 0..rows {
            let mut out = stale(top_k * HS_DIM);
            let host = [0u32, 1, 2];
            let row_ids = &ids[t * top_k..(t + 1) * top_k];
            hs_run(
                None,
                1,
                top_k,
                n_exp,
                row_ids,
                None,
                &host,
                &fx.spans(row_ids, &host),
                &x[t * HS_DIM..(t + 1) * HS_DIM],
                &mut out,
            )
            .unwrap();
            assert_eq!(
                bits(&out),
                bits(&full[t * top_k * HS_DIM..(t + 1) * top_k * HS_DIM]),
                "row {t} window"
            );
        }
        // Worker count.
        for threads in [1usize, 2, 3, 7] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            assert_eq!(
                bits(&run(Some(&pool), &every, &x)),
                bits(&full),
                "{threads} workers"
            );
            assert_eq!(
                bits(&run(Some(&pool), &reversed, &x)),
                bits(&full),
                "{threads} workers, reversed"
            );
        }
        // Row 1 has no host slot: NaN there must never be read.
        let mut poisoned = x.clone();
        poisoned[HS_DIM..2 * HS_DIM].fill(f32::NAN);
        let sparse = [10u32, 2, 7];
        let out = run(None, &sparse, &poisoned);
        for s in 0..12usize {
            if sparse.contains(&(s as u32)) {
                assert_eq!(
                    bits(&out[range(s)]),
                    bits(&full[range(s)]),
                    "sparse slot {s}"
                );
            } else {
                assert!(all_stale(&out[range(s)]), "slot {s} written");
            }
        }
    }

    fn expect_err(r: Result<(), DispatchError>, needle: &str) {
        let e = r.expect_err(needle);
        assert!(
            format!("{e}").contains(needle),
            "expected {needle:?}, got: {e}"
        );
    }

    /// Every defect of the plan is an error from the validator, before any
    /// arithmetic; the arithmetic entry leaves `out` untouched on failure.
    #[test]
    fn host_slot_plan_validation_rejects_bad_plans() {
        let (rows, top_k, n_exp) = (2usize, 2usize, 4usize);
        let fx = host_fixture(n_exp);
        let ids = [0i32, 1, 2, 3];
        let (xl, ol) = (rows * HS_DIM, rows * top_k * HS_DIM);
        let val = |ids: &[i32],
                   gpu: Option<&[i32]>,
                   host: &[u32],
                   ex: &[CpuMoeExpert<'_>],
                   xl: usize,
                   awq: bool,
                   ol: usize| {
            validate_moe_cpu_host_slot_outputs(
                HS_DIM, HS_MI, rows, top_k, n_exp, ids, gpu, host, ex, xl, awq, ol,
            )
        };
        let host = [0u32, 3];
        let ex = fx.spans(&ids, &host);
        val(&ids, Some(&[-1, 1, 2, -1]), &host, &ex, xl, false, ol).unwrap();
        val(&ids, None, &host, &ex, xl, false, ol).unwrap();
        val(&ids, None, &[], &[], xl, false, ol).unwrap();
        // A router-invalid slot owned by neither arm is a valid plan.
        let bad = [0i32, -1, 2, 3];
        let bad_ex = fx.spans(&bad, &host);
        val(&bad, Some(&[-1, -1, 2, -1]), &host, &bad_ex, xl, false, ol).unwrap();
        val(&bad, Some(&[-1, 77, 2, -1]), &host, &bad_ex, xl, false, ol).unwrap();

        let dup = [1u32, 1];
        expect_err(
            val(&ids, None, &dup, &fx.spans(&ids, &dup), xl, false, ol),
            "twice",
        );
        expect_err(
            val(&ids, None, &[4], &[fx.expert(0)], xl, false, ol),
            "out of range",
        );
        expect_err(
            val(&bad, None, &[1], &[fx.expert(0)], xl, false, ol),
            "invalid expert id",
        );
        let big = [0i32, 9, 2, 3];
        expect_err(
            val(&big, None, &[1], &[fx.expert(0)], xl, false, ol),
            "invalid expert id",
        );
        expect_err(
            val(&ids, None, &host, &ex[..1], xl, false, ol),
            "span count",
        );
        let short = CpuMoeExpert {
            gate_up_quant: HS_Q,
            gate_up: &fx.gu[0][..16],
            down_quant: HS_Q,
            down: &fx.dn[0],
        };
        expect_err(val(&ids, None, &[0], &[short], xl, false, ol), "span");
        expect_err(val(&ids, None, &host, &ex, xl, true, ol), "AWQ");
        let same = [1i32, 1, 2, 3];
        let h2 = [0u32, 1];
        expect_err(
            val(
                &same,
                None,
                &h2,
                &[fx.expert(1), fx.expert(2)],
                xl,
                false,
                ol,
            ),
            "different spans",
        );
        // Ownership partition against the masked GPU route.
        expect_err(
            val(&ids, Some(&[0, 1, 2, -1]), &host, &ex, xl, false, ol),
            "ownership",
        );
        expect_err(
            val(&ids, Some(&[-1, -1, 2, -1]), &host, &ex, xl, false, ol),
            "ownership",
        );
        expect_err(
            val(&ids, Some(&[-1, 1, 3, -1]), &host, &ex, xl, false, ol),
            "ownership",
        );
        expect_err(
            val(&bad, Some(&[-1, 1, 2, -1]), &host, &bad_ex, xl, false, ol),
            "ownership",
        );
        // Extents.
        expect_err(val(&ids, None, &host, &ex, xl - 1, false, ol), "activation");
        expect_err(val(&ids, None, &host, &ex, xl, false, ol - 1), "activation");
        expect_err(val(&ids[..3], None, &host, &ex, xl, false, ol), "route ids");
        expect_err(
            validate_moe_cpu_host_slot_outputs(
                HS_DIM,
                HS_MI,
                0,
                top_k,
                n_exp,
                &ids,
                None,
                &[],
                &[],
                0,
                false,
                0,
            ),
            "non-zero",
        );
        expect_err(
            validate_moe_cpu_host_slot_outputs(
                255,
                HS_MI,
                rows,
                top_k,
                n_exp,
                &ids,
                None,
                &[],
                &[],
                xl,
                false,
                ol,
            ),
            "multiple of 256",
        );

        // The arithmetic entry refuses the same plans without writing.
        let x = x_of(xl);
        let mut out = stale(ol);
        let e = moe_cpu_host_slot_outputs(
            HS_DIM,
            HS_MI,
            rows,
            top_k,
            n_exp,
            &ids,
            None,
            &dup,
            &fx.spans(&ids, &dup),
            &x,
            false,
            &mut out,
        )
        .expect_err("duplicate host slot");
        assert!(format!("{e}").contains("twice"));
        assert!(all_stale(&out));
        let e = moe_cpu_host_slot_outputs(
            HS_DIM, HS_MI, rows, top_k, n_exp, &ids, None, &host, &ex, &x, true, &mut out,
        )
        .expect_err("awq");
        assert!(format!("{e}").contains("AWQ"));
        assert!(all_stale(&out));
    }

    /// After `reserve_moe_cpu_host_slot_scratch`, a call within that slot count
    /// grows no scratch buffer, and the capacity getter reports rows, not slots.
    #[test]
    fn reserved_scratch_is_not_regrown_by_the_host_slot_call() {
        reserved_scratch_holds(4, 3);
        // Full logical-window prewarm: one 640-slot reserve, then a 64-row call
        // covering the whole window grows nothing.
        reserved_scratch_holds(64, 10);
    }

    fn reserved_scratch_holds(rows: usize, top_k: usize) {
        let n_exp = 5usize;
        reserve_moe_cpu_host_slot_scratch(HS_DIM, HS_MI, rows * top_k).unwrap();
        // Row units, not slots: legacy 40-slot reserve reports 4 rows at
        // top_k 10; the 640-slot prewarm reports 64.
        assert_eq!(
            moe_cpu_host_slot_scratch_capacity_rows(HS_DIM, HS_MI, top_k),
            rows
        );
        assert_eq!(moe_cpu_host_slot_scratch_capacity_rows(0, HS_MI, top_k), 0);
        assert_eq!(moe_cpu_host_slot_scratch_capacity_rows(HS_DIM, 0, top_k), 0);
        assert_eq!(moe_cpu_host_slot_scratch_capacity_rows(HS_DIM, HS_MI, 0), 0);
        let caps = || {
            MOE_SCRATCH.with(|c| {
                let s = c.borrow();
                [
                    s.gu.capacity(),
                    s.hidden.capacity(),
                    s.dn.capacity(),
                    s.order.capacity(),
                    s.group_start.capacity(),
                    s.group_id.capacity(),
                    s.slot_group.capacity(),
                    s.slot_si.capacity(),
                    s.group_off.capacity(),
                ]
            })
        };
        let before = caps();
        let fx = host_fixture(n_exp);
        let slots = rows * top_k;
        let ids: Vec<i32> = (0..slots).map(|s| (s % n_exp) as i32).collect();
        let host: Vec<u32> = (0..slots as u32).collect();
        let x = x_of(rows * HS_DIM);
        let mut out = stale(slots * HS_DIM);
        hs_run(
            None,
            rows,
            top_k,
            n_exp,
            &ids,
            None,
            &host,
            &fx.spans(&ids, &host),
            &x,
            &mut out,
        )
        .unwrap();
        assert_eq!(caps(), before);
    }
}

#[cfg(test)]
mod slot_check_tests {
    use super::*;

    fn exp<'a>(q: CpuQuant, gu: &'a [u8], dn: &'a [u8]) -> CpuMoeExpert<'a> {
        CpuMoeExpert {
            gate_up_quant: q,
            gate_up: gu,
            down_quant: q,
            down: dn,
        }
    }

    /// Former O(n^2) predicate: first (id, other-position) conflict.
    fn reference(ids: &[i32], hs: &[u32], ex: &[CpuMoeExpert<'_>]) -> Option<usize> {
        for (i, e) in ex.iter().enumerate() {
            for (&other, o) in hs[..i].iter().zip(ex) {
                if ids[other as usize] == ids[hs[i] as usize] && expert_spans_differ(o, e) {
                    return Some(other as usize);
                }
            }
            let _ = i;
        }
        None
    }

    fn fast(ids: &[i32], hs: &[u32], ex: &[CpuMoeExpert<'_>], n_experts: usize) -> Option<usize> {
        let mut reps = ExpertReps::new(n_experts);
        for i in 0..ex.len() {
            if let Some(o) = reps.conflict(ids[hs[i] as usize] as usize, i, ex) {
                return Some(hs[o] as usize);
            }
        }
        None
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[test]
    fn linear_matches_pairwise_reference() {
        let bufs: Vec<Vec<u8>> = (0..6).map(|i| vec![0u8; 64 + i]).collect();
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for case in 0..4000 {
            let n_experts = [4usize, 64, 511, 512, 513, 700][case % 6];
            let slots = 1 + (rng.next() as usize % 40);
            let ids: Vec<i32> = (0..slots)
                .map(|_| (rng.next() as usize % n_experts.min(6)) as i32)
                .collect();
            let mut hs: Vec<u32> = (0..slots as u32).collect();
            for k in (1..hs.len()).rev() {
                hs.swap(k, rng.next() as usize % (k + 1));
            }
            hs.truncate(1 + rng.next() as usize % slots);
            // Adversarial: mostly consistent per-id spans with sparse corruption.
            let corrupt = case % 3 == 0;
            let ex: Vec<CpuMoeExpert<'_>> = hs
                .iter()
                .map(|&s| {
                    let id = ids[s as usize] as usize;
                    let mut b = id % bufs.len();
                    if corrupt && rng.next() % 7 == 0 {
                        b = rng.next() as usize % bufs.len();
                    }
                    exp(CpuQuant::Mq4G256V2, &bufs[b], &bufs[b])
                })
                .collect();
            assert_eq!(
                reference(&ids, &hs, &ex),
                fast(&ids, &hs, &ex, n_experts),
                "case {case}"
            );
        }
    }

    #[test]
    fn linear_all_same_and_all_conflicting() {
        let a = vec![0u8; 8];
        let b = vec![0u8; 8];
        let ids = vec![3i32; 100];
        let hs: Vec<u32> = (0..100).collect();
        let same: Vec<_> = (0..100).map(|_| exp(CpuQuant::Mq4G256V2, &a, &a)).collect();
        assert_eq!(fast(&ids, &hs, &same, 8), None);
        assert_eq!(reference(&ids, &hs, &same), None);
        let mut mixed = same.clone();
        mixed[57] = exp(CpuQuant::Mq4G256V2, &b, &b);
        assert_eq!(fast(&ids, &hs, &mixed, 8), Some(0));
        assert_eq!(reference(&ids, &hs, &mixed), Some(0));
    }

    #[test]
    fn linear_scaling_smoke() {
        let a = vec![0u8; 8];
        let n = 16384usize;
        let ids: Vec<i32> = (0..n).map(|i| (i % 256) as i32).collect();
        let hs: Vec<u32> = (0..n as u32).collect();
        let ex: Vec<_> = (0..n).map(|_| exp(CpuQuant::Mq4G256V2, &a, &a)).collect();
        let t = std::time::Instant::now();
        // Same shape as the predicate loop of validate_moe_cpu_host_slot_outputs.
        let r = fast(&ids, &hs, &ex, 256);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("slot-check 16384 slots: {ms:.3} ms");
        assert_eq!(r, None);
        if !cfg!(debug_assertions) {
            assert!(ms < 20.0, "{ms} ms");
        }
    }
}

/// SP grouped-rows BF16 entry ([`moe_cpu_grouped_rows_bf16`]) at the real
/// Flash-Next dims (hidden 2560, mi 640) over synthetic expert bytes in a real
/// CPU quant format.
#[cfg(test)]
mod sp_rows_tests {
    use super::moe_splice_tests::weight_bytes;
    use super::*;

    const DIM: usize = 2560;
    const MI: usize = 640;
    const Q: CpuQuant = CpuQuant::Mq4G128V2;
    const N_EXP: usize = 6;
    const ROWS: usize = 8;
    const TOP_K: usize = 4;
    const SLOTS: usize = ROWS * TOP_K;
    /// A quiet-NaN payload standing in for "untouched".
    const CANARY: u16 = 0x7fc1;

    struct Ctx {
        gu: Vec<Vec<u8>>,
        dn: Vec<Vec<u8>>,
        ids: Vec<i32>,
        x: Vec<f32>,
        /// Per expert: `(token row, out index)`; out indices run in (expert,
        /// slot) order, so they are a permutation of the host slot numbers.
        per_expert: Vec<Vec<(u32, u32)>>,
        /// `out_of_slot[s]` = out index of host slot `s`.
        out_of_slot: Vec<u32>,
        /// `bf16_bits_rne(moe_cpu_host_slot_outputs)` per out index, `[SLOTS, DIM]`.
        reference: Vec<u16>,
    }

    impl Ctx {
        fn expert(&self, e: usize) -> CpuMoeExpert<'_> {
            CpuMoeExpert {
                gate_up_quant: Q,
                gate_up: &self.gu[e],
                down_quant: Q,
                down: &self.dn[e],
            }
        }

        fn experts(&self, order: &[usize]) -> Vec<SpCpuExpert<'_>> {
            order
                .iter()
                .map(|&e| SpCpuExpert {
                    spans: self.expert(e),
                    rows: &self.per_expert[e],
                })
                .collect()
        }
    }

    fn build_ctx() -> Ctx {
        let gu: Vec<Vec<u8>> = (0..N_EXP)
            .map(|e| weight_bytes(Q, 2 * MI, DIM, 5000 + 131 * e))
            .collect();
        let dn: Vec<Vec<u8>> = (0..N_EXP)
            .map(|e| weight_bytes(Q, DIM, MI, 9000 + 257 * e))
            .collect();
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        // Each token routes to TOP_K distinct experts.
        let mut ids = Vec::with_capacity(SLOTS);
        for _ in 0..ROWS {
            let mut picked: Vec<i32> = Vec::new();
            while picked.len() < TOP_K {
                let e = (next() % N_EXP as u64) as i32;
                if !picked.contains(&e) {
                    picked.push(e);
                }
            }
            ids.extend(picked);
        }
        let x: Vec<f32> = (0..ROWS * DIM)
            .map(|i| ((i as f32) * 0.0137 + (i / DIM) as f32).sin() * 0.5)
            .collect();
        let mut per_expert = vec![Vec::new(); N_EXP];
        let mut out_of_slot = vec![0u32; SLOTS];
        let mut counter = 0u32;
        for e in 0..N_EXP {
            for s in 0..SLOTS {
                if ids[s] as usize == e {
                    per_expert[e].push(((s / TOP_K) as u32, counter));
                    out_of_slot[s] = counter;
                    counter += 1;
                }
            }
        }
        assert_eq!(counter as usize, SLOTS);
        let mut ctx = Ctx {
            gu,
            dn,
            ids,
            x,
            per_expert,
            out_of_slot,
            reference: Vec::new(),
        };
        // The per-slot oracle: the existing checked D2 entry, F32 expanded layout.
        let host: Vec<u32> = (0..SLOTS as u32).collect();
        let spans: Vec<CpuMoeExpert<'_>> =
            ctx.ids.iter().map(|&e| ctx.expert(e as usize)).collect();
        let mut f32_out = vec![f32::from_bits(0x7fc0_dead); SLOTS * DIM];
        moe_cpu_host_slot_outputs(
            DIM,
            MI,
            ROWS,
            TOP_K,
            N_EXP,
            &ctx.ids,
            None,
            &host,
            &spans,
            &ctx.x,
            false,
            &mut f32_out,
        )
        .unwrap();
        assert!(f32_out.iter().all(|v| v.is_finite()));
        assert!(f32_out.iter().any(|&v| v != 0.0));
        let mut reference = vec![0u16; SLOTS * DIM];
        for s in 0..SLOTS {
            let o = ctx.out_of_slot[s] as usize;
            for r in 0..DIM {
                reference[o * DIM + r] = bf16_bits_rne(f32_out[s * DIM + r]);
            }
        }
        drop(spans);
        ctx.reference = reference;
        ctx
    }

    static CTX: LazyLock<Ctx> = LazyLock::new(build_ctx);

    fn canary_out(rows: usize) -> Vec<u16> {
        vec![CANARY; rows * DIM]
    }

    fn run_in(
        pool: Option<&rayon::ThreadPool>,
        x: &[f32],
        experts: &[SpCpuExpert<'_>],
        cap: usize,
        out: &mut [u16],
    ) -> Result<SpCpuStats, DispatchError> {
        let mut scratch = SpCpuScratch::new();
        match pool {
            None => moe_cpu_grouped_rows_bf16(DIM, MI, x, experts, cap, &mut scratch, out),
            Some(pool) => sp_rows_entry(DIM, MI, x, experts, cap, &mut scratch, out, |run| {
                pool.install(|| run())
            }),
        }
    }

    fn expect_err(r: Result<SpCpuStats, DispatchError>, needle: &str) {
        let e = r.expect_err(needle);
        assert!(
            format!("{e}").contains(needle),
            "expected {needle:?}, got: {e}"
        );
    }

    const ALL: [usize; N_EXP] = [0, 1, 2, 3, 4, 5];

    /// Every SP row equals `bf16_rne(moe_cpu_host_slot_outputs)` of the same
    /// (expert, activation row) bit for bit; rows past the plan stay canary.
    #[test]
    fn sp_rows_equal_host_slot_outputs_bitwise() {
        let c = &*CTX;
        let mut out = canary_out(SLOTS + 3);
        let stats = run_in(None, &c.x, &c.experts(&ALL), 4096, &mut out).unwrap();
        assert_eq!(stats.slots, SLOTS);
        assert_eq!(stats.experts, N_EXP);
        assert_eq!(stats.chunks, 1);
        assert_eq!(
            out[..SLOTS * DIM],
            c.reference[..],
            "rows != host slot outputs"
        );
        assert!(out[SLOTS * DIM..].iter().all(|&b| b == CANARY));
        // The route really exercised more than one expert per token.
        assert!(c.per_expert.iter().filter(|r| !r.is_empty()).count() > 1);
    }

    /// The chunk size changes no byte: 1, 7, 64, 4096 and unbounded, and
    /// whatever order the experts are listed in.
    #[test]
    fn sp_rows_chunk_invariant() {
        let c = &*CTX;
        let reversed: Vec<usize> = ALL.iter().rev().copied().collect();
        for cap in [1usize, 7, 64, 4096, usize::MAX] {
            for order in [&ALL[..], &reversed[..]] {
                let mut out = canary_out(SLOTS);
                let stats = run_in(None, &c.x, &c.experts(order), cap, &mut out).unwrap();
                assert_eq!(out, c.reference, "cap {cap}, order {order:?}");
                assert_eq!(stats.slots, SLOTS);
                match cap {
                    1 => assert_eq!(stats.chunks, SLOTS),
                    4096 | usize::MAX => assert_eq!(stats.chunks, 1),
                    _ => assert!(stats.chunks >= SLOTS.div_ceil(cap)),
                }
            }
        }
    }

    /// Same bytes at 1, 4 and 16 workers (and under chunking).
    #[test]
    fn sp_rows_worker_invariant() {
        let c = &*CTX;
        for workers in [1usize, 4, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            for cap in [7usize, 4096] {
                let mut out = canary_out(SLOTS);
                run_in(Some(&pool), &c.x, &c.experts(&ALL), cap, &mut out).unwrap();
                assert_eq!(out, c.reference, "{workers} workers, cap {cap}");
            }
        }
    }

    /// Rows no expert names are never written, and activation rows no expert
    /// names are never read; every plan defect is an error that leaves `out`
    /// untouched.
    #[test]
    fn sp_rows_untouched_canary() {
        let c = &*CTX;
        // Sparse plan: experts 1 and 3 only; out has extra rows; x rows that
        // neither expert names are NaN-poisoned.
        let sub = [1usize, 3];
        let mut named = [false; SLOTS];
        let mut used = [false; ROWS];
        for &e in &sub {
            for &(row, out) in &c.per_expert[e] {
                named[out as usize] = true;
                used[row as usize] = true;
            }
        }
        assert!(named.iter().any(|&n| !n) && used.iter().any(|&u| !u));
        let mut x = c.x.clone();
        for (t, &u) in used.iter().enumerate() {
            if !u {
                x[t * DIM..(t + 1) * DIM].fill(f32::NAN);
            }
        }
        let mut out = canary_out(SLOTS + 4);
        run_in(None, &x, &c.experts(&sub), 5, &mut out).unwrap();
        for o in 0..SLOTS + 4 {
            let row = &out[o * DIM..(o + 1) * DIM];
            if o < SLOTS && named[o] {
                assert_eq!(row, &c.reference[o * DIM..(o + 1) * DIM], "named row {o}");
            } else {
                assert!(row.iter().all(|&b| b == CANARY), "row {o} written");
            }
        }
        // No experts / empty row lists: nothing runs, nothing is written.
        let mut out = canary_out(2);
        let stats = run_in(None, &c.x, &[], 8, &mut out).unwrap();
        assert_eq!((stats.slots, stats.chunks), (0, 0));
        let none = [SpCpuExpert {
            spans: c.expert(0),
            rows: &[],
        }];
        run_in(None, &c.x, &none, 8, &mut out).unwrap();
        assert!(out.iter().all(|&b| b == CANARY));

        // Defects: the first expert is valid and the defect comes last, so a
        // partial write before the error would be visible.
        let rows0 = &c.per_expert[0];
        let dup = [(0u32, rows0[0].1)];
        let oob_out = [(0u32, SLOTS as u32)];
        let oob_row = [(ROWS as u32, 0u32)];
        let short = CpuMoeExpert {
            gate_up: &c.gu[0][..c.gu[0].len() - 1],
            ..c.expert(0)
        };
        let good = SpCpuExpert {
            spans: c.expert(0),
            rows: rows0,
        };
        let cases: [(&str, SpCpuExpert<'_>); 4] = [
            (
                "assigned twice",
                SpCpuExpert {
                    spans: c.expert(1),
                    rows: &dup,
                },
            ),
            (
                "out index",
                SpCpuExpert {
                    spans: c.expert(1),
                    rows: &oob_out,
                },
            ),
            (
                "token row",
                SpCpuExpert {
                    spans: c.expert(1),
                    rows: &oob_row,
                },
            ),
            (
                "gate_up span",
                SpCpuExpert {
                    spans: short,
                    rows: &oob_row,
                },
            ),
        ];
        for (needle, bad) in cases {
            let mut out = canary_out(SLOTS);
            expect_err(run_in(None, &c.x, &[good, bad], 8, &mut out), needle);
            assert!(out.iter().all(|&b| b == CANARY), "{needle}: out written");
        }
        let mut out = canary_out(SLOTS);
        expect_err(
            run_in(None, &c.x, &c.experts(&ALL), 0, &mut out),
            "non-zero",
        );
        expect_err(
            run_in(None, &c.x[..DIM + 1], &c.experts(&ALL), 8, &mut out),
            "multiple of dim",
        );
        assert!(out.iter().all(|&b| b == CANARY));
    }

    /// Validation is O(slots + experts): 40,960 slots (the 4K window at
    /// top_k 10) over 512 experts validate in bounded time, where the pairwise
    /// predicate would take ~8e8 comparisons. Error paths run the same pass.
    #[test]
    fn sp_rows_validation_is_linear() {
        const EXPERTS: usize = 512;
        const PER: usize = 80;
        let c = &*CTX;
        // Interleaved out indices: every expert's rows are spread across the table.
        let rows: Vec<Vec<(u32, u32)>> = (0..EXPERTS)
            .map(|e| {
                (0..PER)
                    .map(|i| ((i % 4096) as u32, (i * EXPERTS + e) as u32))
                    .collect()
            })
            .collect();
        let experts: Vec<SpCpuExpert<'_>> = rows
            .iter()
            .map(|r| SpCpuExpert {
                spans: c.expert(0),
                rows: r,
            })
            .collect();
        let slots = EXPERTS * PER;
        assert_eq!(slots, 40_960);
        let mut seen = Vec::new();
        let t = Instant::now();
        let (live, n) = sp_rows_validate(DIM, MI, 4096, slots, &experts, &mut seen).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("sp rows validate {slots} slots: {ms:.3} ms");
        assert_eq!((live, n), (EXPERTS, slots));
        assert!(ms < 250.0, "validation took {ms} ms");

        // Through the public entry: a bad last expert fails after the full
        // pass, before any pool work or output write. The output table is
        // zero-filled lazily, so no 200 MB is touched.
        let short = CpuMoeExpert {
            down: &c.dn[0][..c.dn[0].len() - 1],
            ..c.expert(0)
        };
        let mut with_bad = experts.clone();
        with_bad.push(SpCpuExpert {
            spans: short,
            rows: &[(0, 0)],
        });
        let x = vec![0.0f32; 4096 * DIM];
        let mut out = vec![0u16; slots * DIM];
        let t = Instant::now();
        expect_err(run_in(None, &x, &with_bad, 4096, &mut out), "down span");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("sp rows rejected plan {slots} slots: {ms:.3} ms");
        assert!(ms < 250.0, "rejection took {ms} ms");
    }

    /// The kernel contract's BF16 encoder, verbatim from
    /// `slab_moe::slowlink_selftest::rbf`.
    fn kernel_rbf(x: f32) -> f32 {
        if x.is_nan() {
            return f32::NAN;
        }
        let u = x.to_bits();
        if (!u & 0x7f80_0000) != 0 {
            f32::from_bits(u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff_0000)
        } else {
            f32::from_bits(u & 0xffff_0000)
        }
    }

    fn check_range(lo: u32, hi: u32) -> u64 {
        let mut mismatches = 0u64;
        for u in lo..=hi {
            let x = f32::from_bits(u);
            let got = bf16_bits_rne(x);
            if x.is_nan() {
                // Quiet NaN of the same sign: never Inf, never a number.
                let back = f32::from_bits((got as u32) << 16);
                if !(back.is_nan() && (got & 0x8000) == ((u >> 16) as u16 & 0x8000)) {
                    mismatches += 1;
                }
            } else if got != (kernel_rbf(x).to_bits() >> 16) as u16 {
                mismatches += 1;
            }
        }
        mismatches
    }

    /// All 2^32 f32 bit patterns: finite and ±Inf equal the kernel's `bf`, NaN
    /// stays NaN. Includes the round-up-to-Inf boundary and every subnormal.
    #[test]
    fn bf16_bits_rne_exhaustive() {
        use rayon::prelude::*;
        let bad: u64 = (0u32..=0xffff)
            .into_par_iter()
            .map(|hi| check_range(hi << 16, (hi << 16) | 0xffff))
            .sum();
        assert_eq!(bad, 0, "bf16_bits_rne disagrees with the kernel contract");
        // Spot values: RNE ties, overflow to Inf, signed zero, NaN payloads.
        assert_eq!(bf16_bits_rne(1.0), 0x3f80);
        assert_eq!(bf16_bits_rne(f32::from_bits(0x3f80_8000)), 0x3f80); // tie → even
        assert_eq!(bf16_bits_rne(f32::from_bits(0x3f81_8000)), 0x3f82); // tie → even
        assert_eq!(bf16_bits_rne(f32::MAX), 0x7f80); // rounds past BF16 max → +Inf
        assert_eq!(bf16_bits_rne(f32::MIN), 0xff80);
        assert_eq!(bf16_bits_rne(-0.0), 0x8000);
        assert_eq!(bf16_bits_rne(f32::INFINITY), 0x7f80);
        assert_eq!(bf16_bits_rne(f32::NEG_INFINITY), 0xff80);
        for u in [0x7f80_0001u32, 0x7fff_ffff, 0xff80_0001, 0xffff_ffff] {
            let b = bf16_bits_rne(f32::from_bits(u));
            assert!(
                f32::from_bits((b as u32) << 16).is_nan(),
                "{u:#x} -> {b:#x}"
            );
        }
    }
}
