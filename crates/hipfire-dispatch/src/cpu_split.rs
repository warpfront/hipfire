// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Eager CPU-cold/GPU-hot slot transaction. The original route never changes.
//!
//! `prepare` is the sole input-readiness wait, before the GPU routed tail.
//! `join` runs scoped CPU arithmetic while that tail executes, then queues cold
//! slot installation behind its masked zero stores. Neither arm reads residual.
//!
//! Lab diagnostics (`configure`, `SplitParityMode`) are OFF by default. When off,
//! every hot path pays exactly one `Cell<bool>` read and keeps the production
//! byte order, stream order and overlap. Timing/Full modes add the endpoint
//! override, an optional serial join and host-clock/buffer capture; they never
//! run in production.
//!
//! Two opt-in scheduling variants (both byte-exact, defaults unchanged, state
//! machines in [`crate::cpwait`]): `HIPFIRE_QWEN4_D2_READY=event` replaces the
//! blocking input synchronization of `prepare` with an event-proven readiness
//! spin; `HIPFIRE_QWEN4_D2_JOIN=cpwait` (mapped return, all-cold layers) queues a
//! command-processor `hipStreamWaitValue32` and the canonical combine BEFORE the
//! CPU arithmetic (`cpwait_arm`), then `join` runs the arithmetic and releases the
//! wait with a generation store. Nothing may synchronize the stream between those
//! two calls; every error between them stores the abort value first.

use crate::cpu_exec::{self, CpuArmTrace, CpuMoeExpert};
use crate::cpwait::{
    spin_until_ready, ArmId, Disarm, GatePort, JoinGate, JoinTarget, SpinOutcome, SystemClock,
    Watchdog, READY_SPIN_BUDGET,
};
use crate::families::moe::RoutedExpertWeights;
use crate::types::DispatchError;
use hip_bridge::{
    HipError, MappedHostBuffer, MappedPublished, PinnedTransferBuffer, SignalGen, SignalWord,
    SIGNAL_ABORT,
};
use hipfire_config::memory::{D2Join, D2Ready, D2Return};
use hipfire_cpu::quant::CpuQuant;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::time::Instant;

pub const CPU_SPLIT_MAX_ROWS: usize = 4;

/// Actual allocated storage bound (rows) of a [`CpuSplitExecutor`]: 64, the
/// same value as `hipfire_config::QWEN4_VERIFY_MAX_ROWS`. Storage only, NOT a
/// qualification or a default: the legacy default route still admits at most
/// [`CPU_SPLIT_MAX_ROWS`] rows (see [`cpu_splice_window`]), and a 5..64-row call
/// additionally needs an explicit lab-wide qualification (verify scope) from the
/// routing/admission owners. Never metadata-bump [`CPU_SPLIT_MAX_ROWS`].
pub const CPU_SPLIT_STORAGE_MAX_ROWS: usize = 64;

const _: () = assert!(CPU_SPLIT_STORAGE_MAX_ROWS == hipfire_config::QWEN4_VERIFY_MAX_ROWS);
const _: () = assert!(CPU_SPLIT_STORAGE_MAX_ROWS == 64);

/// Whether a call of `rows` real rows over `route_rows` routing rows takes the
/// CPU routed-expert window on a layer that is `armed`, under an explicit
/// `qualified_max_rows` cap: decode and a verify window of at most that many
/// rows, and only when the real rows are the whole routing window. A call over
/// a wider routing window (a short tail of a long prefill whose `route_rows` is
/// the canonical chunk) takes the GPU route.
///
/// `qualified_max_rows` is the routing/admission owner's qualified width for
/// this call (4 on the default route; up to [`CPU_SPLIT_STORAGE_MAX_ROWS`] only
/// under an explicit lab-wide verify scope). Callers MUST clamp it to
/// [`CPU_SPLIT_STORAGE_MAX_ROWS`]; the executor independently refuses any `rows`
/// beyond its allocated [`CpuSplitExecutor::capacity_rows`].
pub const fn cpu_splice_window_qualified(
    armed: bool,
    rows: usize,
    route_rows: usize,
    qualified_max_rows: usize,
) -> bool {
    armed && rows != 0 && rows == route_rows && rows <= qualified_max_rows
}

/// Whether a call of `rows` real rows over `route_rows` routing rows takes the
/// CPU routed-expert window on a layer that is `armed`: decode and a verify
/// window of at most [`CPU_SPLIT_MAX_ROWS`] rows, and only when the real rows
/// are the whole routing window. A call over a wider routing window (a
/// 1..=4-row tail of a long prefill whose `route_rows` is the canonical chunk)
/// takes the GPU route. The D1 seal (`seal_moe_for_layer`) applies this
/// predicate directly; the D2 executor gate
/// ([`crate::families::moe::cpu_split_executor`]) sees only the row count, so
/// the qwen4 prefill seal binds an executor-less view of the layer's weights
/// (`Qwen4MoeLayerRuntime::routed_view`) for every call this rejects.
pub const fn cpu_splice_window(armed: bool, rows: usize, route_rows: usize) -> bool {
    cpu_splice_window_qualified(armed, rows, route_rows, CPU_SPLIT_MAX_ROWS)
}

const MAX_SLOTS: usize = CPU_SPLIT_STORAGE_MAX_ROWS * 10;

fn err(message: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(format!("CPU tier split: {message}"))
}

/// CPU-readable bytes of ANY expert of a layer, hot or cold, independent of
/// tier owners. Implemented by the lab over the artifact mmap.
pub trait CpuShadow: Send + Sync {
    fn expert(&self, layer: usize, expert: usize) -> Option<CpuMoeExpert<'_>>;
}

/// Frozen placement for one diagnostic run.
/// `AllHot`: every VALID slot is a GPU slot (host_count = 0, masked ids == ids).
/// `AllCold`: every VALID slot is a host slot (masked ids all -1); EVERY valid
/// slot's CPU bytes come from `shadow`. Invalid ids stay in neither arm.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SplitEndpoint {
    Natural,
    AllHot,
    AllCold,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SplitParityMode {
    Off,
    Timing,
    Full,
}

#[derive(Clone)]
pub struct SplitParityConfig {
    pub mode: SplitParityMode,
    /// `None` = every armed layer; other layers run production unchanged.
    pub layers: Option<Vec<usize>>,
    pub endpoint: SplitEndpoint,
    /// `join` drains the stream once BEFORE the CPU slot arithmetic.
    pub serial: bool,
    /// REQUIRED when `endpoint == AllCold`.
    pub shadow: Option<Arc<dyn CpuShadow>>,
    /// Record cap; extra calls run (endpoint/serial still apply) unrecorded.
    pub max_records: usize,
}

/// Host `Instant` nanoseconds; only filled for wanted layers in Timing/Full.
/// In Full mode the pre-combine drain makes these serialized, not overlap, times.
#[derive(Clone, Debug, Default)]
pub struct SplitTiming {
    /// prepare() entry -> stream idle AFTER the D2 combine.
    pub wall_ns: u64,
    /// prepare() wall (D2H + completes + payload validate + mask H2D enqueue).
    pub prepare_ns: u64,
    /// End of prepare -> start of join (host time enqueueing the masked GPU tail).
    pub gpu_enqueue_ns: u64,
    /// CPU slot arithmetic inside join (`moe_cpu_host_slot_outputs`).
    pub join_cpu_ns: u64,
    /// join() wall incl. installing H2D enqueues (and the leading sync if serial).
    pub join_total_ns: u64,
    /// End of join -> stream idle after the combine.
    pub tail_ns: u64,
    pub cold_slots: u32,
    pub hot_slots: u32,
    pub invalid_slots: u32,
    pub serial: bool,
}

/// Cold-result transport of one executor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum D2ReturnMode {
    /// Cold slots are copied host-to-device into `down_expanded`, queued
    /// behind the GPU tail's masked zero stores (the original transport).
    #[default]
    Copy,
    /// The CPU writes the cold slots into an owned mapped, coherent host
    /// allocation and the unchanged canonical combine reads it directly. No
    /// cold-slot H2D is issued. All-cold layers only.
    Mapped,
}

impl D2ReturnMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Copy => "copy",
            Self::Mapped => "mapped",
        }
    }
}

/// Why a layer cannot use the mapped return, or `None`. The mapped view
/// REPLACES the combine's expanded input, so it must hold every live slot:
/// only a layer whose every expert is cold qualifies.
fn mapped_return_refusal(placement: &[u8]) -> Option<&'static str> {
    if placement.is_empty() {
        return Some("empty ownership plan");
    }
    if !placement.iter().all(|&p| p == 1) {
        return Some(
            "mixed hot/cold placement: the mapped return replaces the whole combine input, so only an all-cold layer can use it (copy transport kept)",
        );
    }
    None
}

/// A transport decision and the capability reason when the request was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReturnDecision {
    pub mode: D2ReturnMode,
    pub refusal: Option<&'static str>,
}

/// Capability routing of the requested transport for one layer's ownership
/// plan: `Mapped` is honoured on all-cold plans only, every other request or
/// plan keeps `Copy` and (for a refused `Mapped`) says why.
pub fn select_return_mode(request: D2Return, placement: &[u8]) -> ReturnDecision {
    match request {
        D2Return::Copy => ReturnDecision {
            mode: D2ReturnMode::Copy,
            refusal: None,
        },
        D2Return::Mapped => match mapped_return_refusal(placement) {
            None => ReturnDecision {
                mode: D2ReturnMode::Mapped,
                refusal: None,
            },
            Some(reason) => ReturnDecision {
                mode: D2ReturnMode::Copy,
                refusal: Some(reason),
            },
        },
    }
}

/// A join decision and the capability reason when the request was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JoinDecision {
    pub mode: D2Join,
    pub refusal: Option<&'static str>,
}

/// Capability routing of the requested join for one layer: `CpWait` is
/// honoured only on an all-cold layer that returns through the mapped view and
/// whose device can execute stream value waits. Every other request keeps
/// `Host`, and a refused `CpWait` says why. Never silent.
pub fn select_join_mode(
    request: D2Join,
    return_mode: D2ReturnMode,
    placement: &[u8],
    capable: bool,
) -> JoinDecision {
    match request {
        D2Join::Host => JoinDecision {
            mode: D2Join::Host,
            refusal: None,
        },
        D2Join::CpWait => {
            let refusal = if placement.is_empty() {
                Some("empty ownership plan")
            } else if !placement.iter().all(|&p| p == 1) {
                Some("mixed hot/cold placement: the CP-wait join needs the mapped return, which only an all-cold layer can use (host join kept)")
            } else if return_mode != D2ReturnMode::Mapped {
                Some("the CP-wait join needs the mapped return (HIPFIRE_QWEN4_D2_RETURN=mapped); the copy transport is installed behind the GPU tail (host join kept)")
            } else if !capable {
                Some("the device does not report hipDeviceAttributeCanUseStreamWaitValue (host join kept)")
            } else {
                None
            };
            match refusal {
                None => JoinDecision {
                    mode: D2Join::CpWait,
                    refusal: None,
                },
                Some(reason) => JoinDecision {
                    mode: D2Join::Host,
                    refusal: Some(reason),
                },
            }
        }
    }
}

/// How `prepare` waits for its input downloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputWait {
    /// Blocking stream synchronizations (the original boundary).
    StreamSync,
    /// An event proof polled within [`READY_SPIN_BUDGET`].
    EventSpin,
}

/// The input wait of a configured readiness. The same under the host and the
/// CP-wait join: the CP-wait join arms no readiness word of its own.
fn input_wait(ready: D2Ready) -> InputWait {
    match ready {
        D2Ready::Sync => InputWait::StreamSync,
        D2Ready::Event => InputWait::EventSpin,
    }
}
/// Fixed-size per-call join trace for the opt-in `HIPFIRE_CPU_SPLIT_TRACE`
/// diagnostic (mapped + CP-wait successful joins only). Stack-owned; no
/// allocation. The `cpu_exec` sub-record crosses the Rayon pool as the owned
/// return of the crate-private timed entry. Inclusive totals overlap disjoint
/// children by design (`join_total_ns` over resolve/prove/cpu_total/zero/
/// publish/mark/release; `cpu_total_ns` over validate/pool_install;
/// `pool_install_ns` over `pool_body_ns`; `pool_body_ns` over
/// group/GU/middle/DN/scatter) — reconciliation is `child_sum <= inclusive`,
/// except `gu/dn == prep + exec` exactly.
#[derive(Clone, Copy, Default, Debug)]
struct CpuJoinTrace {
    resolve_ns: u64,
    prove_ns: u64,
    trace: CpuArmTrace,
    zero_ns: u64,
    zero_slots: u64,
    publish_ns: u64,
    mark_ns: u64,
    release_ns: u64,
    /// Enclosures in nanoseconds from the existing join trace-entry `Instant`.
    /// Retirement encloses the existing HIP event-record call, not GPU time.
    retirement_start_ns: Option<u64>,
    retirement_end_ns: Option<u64>,
    /// Encloses `JoinGate::release`, including its actual join-word store.
    /// This is not the broader legacy `release_ns` bookkeeping duration.
    release_start_ns: Option<u64>,
    release_end_ns: Option<u64>,
    join_word_device: usize,
    /// Join entry through release; measured BEFORE the single log line, so no
    /// logging cost is included.
    join_total_ns: u64,
}

/// Numeric JSON or literal null, without allocation. A skipped CPU phase has
/// no boundary; zero would falsely position it at the join-entry epoch.
struct TraceNs(Option<u64>);

impl std::fmt::Display for TraceNs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(value) => write!(f, "{value}"),
            None => f.write_str("null"),
        }
    }
}

/// Zero every slot row of `out` that is not a host slot: exactly the bytes the
/// copy transport's `down_expanded` holds there (the masked GPU tail stores
/// zero for slots owned by neither arm). `host_slots` is strictly ascending.
fn zero_unwritten_slots(out: &mut [f32], dim: usize, host_slots: &[u32]) {
    let mut next = 0usize;
    for &slot in host_slots {
        let slot = slot as usize;
        out[next * dim..slot * dim].fill(0.0);
        next = slot + 1;
    }
    out[next * dim..].fill(0.0);
}

/// What the executor did for one recorded call (lab records only).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SplitTransport {
    /// Transport this call used; `Copy` for every call of a copy executor and
    /// for a lab all-hot call of a mapped one.
    pub mode: D2ReturnMode,
    /// Cold-slot H2D enqueues this call issued (zero on a mapped call).
    pub output_copy_calls: u64,
    /// Mapped generations this call published (1 on a mapped call).
    pub mapped_publications: u64,
    /// Retirements that actually blocked, prepare and join of this call.
    pub mapped_retire_waits: u64,
}

/// Per-generation protocol of the mapped return, independent of HIP so the
/// ordering is testable without a device:
/// `Idle -> Published (join) -> Lent (combine_outputs) -> InFlight
/// (combine_enqueued) -> Idle (next prepare retires)`. A lab all-hot call
/// bypasses the mapped view (`Bypassed`); any failure is `Poisoned`.
/// The CP-wait join reverses join and combine: `Idle -> Gated (combine queued
/// behind the wait) -> InFlight (join published and marked it in flight)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReturnStage {
    Idle,
    Published,
    Lent,
    InFlight,
    Bypassed,
    Poisoned,
    Gated,
}

/// Which tensor the canonical combine must read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CombineSource {
    Expanded,
    Mapped,
}

#[derive(Clone, Copy, Debug)]
struct ReturnProtocol {
    stage: ReturnStage,
}

impl ReturnProtocol {
    const fn new() -> Self {
        Self {
            stage: ReturnStage::Idle,
        }
    }

    /// `prepare`: `Ok(true)` when the previous generation's consumer must be
    /// retired before anything can write the buffer. A published generation
    /// whose combine was never enqueued is a protocol error.
    fn admit_prepare(&mut self) -> Result<bool, &'static str> {
        match self.stage {
            ReturnStage::Idle => Ok(false),
            ReturnStage::Bypassed => {
                self.stage = ReturnStage::Idle;
                Ok(false)
            }
            ReturnStage::InFlight => Ok(true),
            ReturnStage::Published | ReturnStage::Lent => {
                Err("the previous generation was published but its combine was never enqueued")
            }
            ReturnStage::Gated => {
                Err("the previous generation queued its combine behind a CP wait but never joined")
            }
            ReturnStage::Poisoned => Err("the mapped return is poisoned"),
        }
    }

    fn retired(&mut self) {
        if self.stage == ReturnStage::InFlight {
            self.stage = ReturnStage::Idle;
        }
    }

    /// `join` may write only over a retired buffer.
    fn begin_join(&self) -> Result<(), &'static str> {
        match self.stage {
            ReturnStage::Idle => Ok(()),
            _ => Err("join over a mapped generation that was not retired"),
        }
    }

    fn published(&mut self) {
        self.stage = ReturnStage::Published;
    }
    fn bypassed(&mut self) {
        self.stage = ReturnStage::Bypassed;
    }
    fn poison(&mut self) {
        self.stage = ReturnStage::Poisoned;
    }

    /// CP-wait `cpwait_arm`: the consumer of the generation about to be
    /// written is queued behind the wait; the retired buffer is lent to it
    /// before any host store.
    fn arm_gated(&mut self) -> Result<CombineSource, &'static str> {
        match self.stage {
            ReturnStage::Idle => {
                self.stage = ReturnStage::Gated;
                Ok(CombineSource::Mapped)
            }
            ReturnStage::Gated => Err("cpwait_arm twice for one generation"),
            ReturnStage::Poisoned => Err("cpwait_arm after a failed transaction"),
            _ => Err("cpwait_arm over a mapped generation that was not retired"),
        }
    }

    /// CP-wait `join` may write only into the generation armed for it.
    fn begin_join_gated(&self) -> Result<(), &'static str> {
        match self.stage {
            ReturnStage::Gated => Ok(()),
            _ => Err("cpwait join without an armed generation"),
        }
    }

    /// CP-wait `join` published the generation and marked its (already queued)
    /// consumer in flight.
    fn joined_gated(&mut self) {
        self.stage = ReturnStage::InFlight;
    }

    /// `combine_outputs`: the published view is handed out once per generation.
    fn lend(&mut self) -> Result<CombineSource, &'static str> {
        match self.stage {
            ReturnStage::Published => {
                self.stage = ReturnStage::Lent;
                Ok(CombineSource::Mapped)
            }
            ReturnStage::Bypassed => Ok(CombineSource::Expanded),
            ReturnStage::Idle => Err("combine_outputs before a successful join of this generation"),
            ReturnStage::Lent => Err("combine_outputs twice for one generation"),
            ReturnStage::InFlight => Err("combine_outputs after the generation was consumed"),
            ReturnStage::Gated => Err("combine_outputs on a CP-wait generation (use cpwait_arm)"),
            ReturnStage::Poisoned => Err("combine_outputs after a failed join"),
        }
    }

    /// `combine_enqueued`: `Ok(true)` when the published token must be marked
    /// in flight, `Ok(false)` for a bypassed (copy) call.
    fn enqueued(&mut self) -> Result<bool, &'static str> {
        match self.stage {
            ReturnStage::Lent => {
                self.stage = ReturnStage::InFlight;
                Ok(true)
            }
            ReturnStage::Bypassed => {
                self.stage = ReturnStage::Idle;
                Ok(false)
            }
            ReturnStage::Published => Err("combine_enqueued without combine_outputs"),
            ReturnStage::Idle => Err("combine_enqueued without a published generation"),
            ReturnStage::InFlight => Err("combine_enqueued twice for one generation"),
            ReturnStage::Gated => Err("combine_enqueued on a CP-wait generation"),
            ReturnStage::Poisoned => Err("combine_enqueued after a failed join"),
        }
    }
}

/// One record per recorded layer call, in call order. Timing mode leaves every
/// tensor Vec empty.
#[derive(Clone, Debug)]
pub struct SplitParityRecord {
    pub layer: usize,
    pub rows: usize,
    pub dim: usize,
    pub mi: usize,
    pub top_k: usize,
    pub endpoint: SplitEndpoint,
    /// `placement()` clone, 0 hot / 1 cold.
    pub placement: Vec<u8>,
    /// [rows*dim] bit-exact executor input.
    pub x_rot: Vec<f32>,
    /// [rows*top_k] ORIGINAL route as consumed.
    pub ids: Vec<i32>,
    /// [rows*top_k]
    pub weights: Vec<f32>,
    /// [rows*dim] combine target BEFORE the D2 combine.
    pub target_in: Vec<f32>,
    /// [rows*top_k*dim] expanded exactly as the combine consumed it.
    pub d2_expanded: Vec<f32>,
    /// [rows*dim] combine target AFTER the D2 combine.
    pub d2_target: Vec<f32>,
    /// [rows*top_k*dim] all-GPU tail on ORIGINAL ids, run after the D2 combine.
    pub gpu_expanded: Vec<f32>,
    pub timing: SplitTiming,
    /// Transport of this call (filled at `join`).
    pub transport: SplitTransport,
}

/// Hard ceiling on `max_records` (guards unbounded lab memory).
pub const SPLIT_PARITY_MAX_RECORDS: usize = 1 << 20;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Prepared,
    Joined,
    Before,
    After,
}

struct Open {
    t0: Instant,
    t_prep_end: Instant,
    t_join_end: Instant,
    stage: Stage,
    full: bool,
    rec: SplitParityRecord,
}

struct ParityState {
    mode: SplitParityMode,
    endpoint: SplitEndpoint,
    serial: bool,
    shadow: Option<Arc<dyn CpuShadow>>,
    max_records: usize,
    records: Vec<SplitParityRecord>,
    open: Option<Open>,
}

fn sync_stream(gpu: &Gpu) -> Result<(), DispatchError> {
    match gpu.active_stream.as_ref() {
        Some(s) => gpu.hip.stream_synchronize(s),
        None => gpu.hip.stream_synchronize_default(),
    }
    .map_err(err)?;
    gpu.take_device_wait_error().map_err(err)
}

fn read_f32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Result<Vec<f32>, DispatchError> {
    if t.buf.size() < n * 4 {
        return Err(err("parity capture tensor too small"));
    }
    let mut v = vec![0.0f32; n];
    // SAFETY: `v` owns exactly n f32s; every byte pattern is a valid f32.
    let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).map_err(err)?;
    gpu.take_device_wait_error().map_err(err)?;
    Ok(v)
}

fn ns(from: Instant, to: Instant) -> u64 {
    to.saturating_duration_since(from)
        .as_nanos()
        .min(u64::MAX as u128) as u64
}

struct Transaction {
    input: PinnedTransferBuffer,
    mask: PinnedTransferBuffer,
    output: PinnedTransferBuffer,
    ids: [i32; MAX_SLOTS],
    gpu_ids: [i32; MAX_SLOTS],
    host_slots: [u32; MAX_SLOTS],
    host_count: usize,
    rows: usize,
    ready: bool,
    failed: bool,
    /// Set by `prepare` only for an `AllCold` diagnostic run.
    cold_shadow: Option<Arc<dyn CpuShadow>>,
    /// Mapped-return generation protocol (inert for a copy executor).
    ret: ReturnProtocol,
    /// Published token of the current mapped generation until it is marked in
    /// flight by `combine_enqueued`.
    token: Option<MappedPublished>,
    /// This call reads the mapped view (mapped executor, not a lab all-hot call).
    mapped_call: bool,
    /// `mapped_retire_waits` when this call's `prepare` began.
    waits_base: u64,
    /// Generation of this call's join word (CP-wait executors).
    cp_gen: u32,
}

/// The mapped cold-result return: one owning mapped host allocation and its
/// non-owning device view. The view is `Borrowed` and is never passed to a free
/// path; `buf` strictly outlives every use of it (both die together in
/// `free_gpu`, after a device synchronize, and `buf` quiesces itself on drop).
struct MappedReturn {
    view: GpuTensor,
    buf: RefCell<MappedHostBuffer>,
}

/// The CP-wait join of one executor: a join word the CPU writes after the
/// slots, the gate that funnels every error into an abort, and the watchdog
/// arm. Dropping it stops the watchdog arm first, then releases any queued wait
/// (the word's drop stores the abort value before it drains).
struct CpWait {
    gate: RefCell<JoinGate>,
    /// The watchdog arm of the queued wait, if any.
    arm: Cell<Option<ArmId>>,
    target: Arc<JoinTarget>,
    watchdog: &'static Watchdog<SystemClock>,
    gen: Cell<SignalGen>,
    joins: Cell<u64>,
    join_word: SignalWord,
}

/// [`GatePort`] over a [`CpWait`]'s join word, its watchdog and (when present)
/// the device.
struct CpPort<'a> {
    cp: &'a CpWait,
    gpu: Option<&'a Gpu>,
}

impl GatePort for CpPort<'_> {
    fn store_join(&self, value: u32) {
        self.cp.join_word.store_release(value);
    }

    fn disarm_watchdog(&self) -> Disarm {
        match self.cp.arm.take() {
            Some(id) => self.cp.watchdog.disarm(id),
            None => Disarm::Unknown,
        }
    }

    fn device_sync(&self) -> Result<(), String> {
        match self.gpu {
            Some(gpu) => gpu.hip.device_synchronize().map_err(|e| e.to_string()),
            None => Err("no device handle for a drain".into()),
        }
    }
}

impl CpWait {
    fn next_gen(&self) -> Result<u32, DispatchError> {
        let mut gen = self.gen.get();
        let value = gen.next().map_err(err)?;
        self.gen.set(gen);
        Ok(value)
    }

    /// The error funnel with a device: abort FIRST (if a wait is queued), poison,
    /// then drain.
    fn fail(&self, gpu: &Gpu) -> Result<(), String> {
        self.target.poison();
        self.gate.borrow_mut().fail(&CpPort {
            cp: self,
            gpu: Some(gpu),
        })
    }

    /// Release a queued wait without any device call. `true` when this stored
    /// the abort value.
    fn abort_only(&self) -> bool {
        match self.gate.try_borrow_mut() {
            Ok(mut gate) => {
                let stored = gate.abort(&CpPort {
                    cp: self,
                    gpu: None,
                });
                if stored {
                    self.target.poison();
                }
                stored
            }
            Err(_) => {
                // The gate is mid-transition on this thread: the abort value is
                // always safe to store, and a stuck wait never is.
                self.join_word.store_release(SIGNAL_ABORT);
                self.target.poison();
                true
            }
        }
    }
}

impl Drop for CpWait {
    fn drop(&mut self) {
        if let Some(id) = self.arm.take() {
            self.watchdog.disarm(id);
        }
    }
}

/// Preallocated per-layer transaction and immutable expert ownership plan.
/// A runtime must release this with `free_gpu` while its device is bound, just
/// like its other model-owned tensors. CPU work never outlives `join`.
pub struct CpuSplitExecutor {
    /// CP-wait join; `None` selects the host join. Declared FIRST so it drops
    /// first: its signal word releases any queued wait before the transfer
    /// buffers (which synchronize the device) or the mapped buffer drop.
    cpwait: Option<CpWait>,
    dim: usize,
    mi: usize,
    top_k: usize,
    max_rows: usize,
    device: i32,
    placement: Box<[u8]>,
    masked_indices: GpuTensor,
    state: RefCell<Transaction>,
    /// One-read gate for every diagnostic path; false == production.
    diag_on: Cell<bool>,
    /// A record is open for the in-flight transaction.
    recording: Cell<bool>,
    /// The open record is Full and still awaits `parity_reference`.
    ref_pending: Cell<bool>,
    layer: Cell<usize>,
    parity: RefCell<ParityState>,
    /// Mapped cold-result return; `None` selects the copy transport.
    mapped: Option<MappedReturn>,
    /// Why a requested mapped return was refused (copy transport kept).
    return_refusal: Option<String>,
    /// Cold-slot H2D enqueues, ever.
    output_copy_calls: Cell<u64>,
    /// Mapped generations published, ever.
    mapped_publications: Cell<u64>,
    /// Mapped retirements that actually blocked, ever.
    mapped_retire_waits: Cell<u64>,
    /// Input readiness mechanism (`HIPFIRE_QWEN4_D2_READY`); `Sync` is the
    /// original boundary.
    ready: D2Ready,
    /// Readiness polls (event queries), ever.
    ready_polls: Cell<u64>,
    /// Blocking stream synchronizations event readiness still needed (spin
    /// budget exhausted, or a buffer the proof does not cover), ever.
    ready_fallback_syncs: Cell<u64>,
    /// Why a requested CP-wait join was refused (host join kept).
    join_refusal: Option<String>,
    /// Opt-in `HIPFIRE_CPU_SPLIT_TRACE=1`, read once at construction and
    /// cached. Independent of `diag_on`/parity: it never enables them and they
    /// never enable it. Disabled: no clock reads, records, or output.
    cpu_trace: bool,
}

/// The single env value that enables the CPU-arm trace; every other setting
/// (including unset) keeps the zero-overhead path.
fn cpu_split_trace_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_CPU_SPLIT_TRACE").is_ok_and(|v| v == "1")
}

/// Allocated slot/element extents of a `max_rows`-row [`CpuSplitExecutor`]:
/// `(slots, input_values, mask_values, output_values)` with `slots = max_rows
/// * top_k` (at most 640 = [`CPU_SPLIT_STORAGE_MAX_ROWS`] × 10), `input_values
/// = max_rows * dim + 2 * slots` (original x, then ids, then weights — f32
/// each), `mask_values = slots` and `output_values = slots * dim`. Checked:
/// `0` or more than [`CPU_SPLIT_STORAGE_MAX_ROWS`] rows, and any overflowing
/// extent, is an error before any device allocation exists, so a malformed
/// request can never partially mutate an executor.
pub fn cpu_split_storage_extents(
    max_rows: usize,
    dim: usize,
    top_k: usize,
) -> Result<(usize, usize, usize, usize), DispatchError> {
    if max_rows == 0 || max_rows > CPU_SPLIT_STORAGE_MAX_ROWS {
        return Err(err("unsupported row cap"));
    }
    let slots = max_rows
        .checked_mul(top_k)
        .ok_or_else(|| err("slot overflow"))?;
    let input_values = max_rows
        .checked_mul(dim)
        .and_then(|v| slots.checked_mul(2).and_then(|t| v.checked_add(t)))
        .ok_or_else(|| err("input overflow"))?;
    let output_values = slots
        .checked_mul(dim)
        .ok_or_else(|| err("output overflow"))?;
    Ok((slots, input_values, slots, output_values))
}

impl CpuSplitExecutor {
    pub fn new(
        gpu: &Gpu,
        dim: usize,
        mi: usize,
        top_k: usize,
        max_rows: usize,
        placement: &[u8],
    ) -> Result<Self, DispatchError> {
        if dim == 0
            || dim % 256 != 0
            || mi == 0
            || mi % 128 != 0
            || top_k != 10
            || placement.is_empty()
            || placement.len() > i32::MAX as usize
            || placement.iter().any(|&v| v > 1)
        {
            return Err(err("unsupported shape or ownership plan"));
        }
        if max_rows == 0 || max_rows > CPU_SPLIT_STORAGE_MAX_ROWS {
            return Err(err("unsupported row cap"));
        }
        let (slots, input_values, _, output_values) =
            cpu_split_storage_extents(max_rows, dim, top_k)?;
        let input_bytes = input_values
            .checked_mul(4)
            .ok_or_else(|| err("input byte overflow"))?;
        let mask_bytes = slots
            .checked_mul(4)
            .ok_or_else(|| err("mask byte overflow"))?;
        let output_bytes = output_values
            .checked_mul(4)
            .ok_or_else(|| err("output byte overflow"))?;
        let input = gpu.hip.transfer_buffer(input_bytes).map_err(err)?;
        let mask = gpu.hip.transfer_buffer(mask_bytes).map_err(err)?;
        let output = gpu.hip.transfer_buffer(output_bytes).map_err(err)?;
        // Allocate last: every preceding fallible operation has a draining owner.
        let masked_indices = GpuTensor {
            buf: gpu.hip.malloc(mask_bytes).map_err(err)?,
            shape: vec![slots],
            dtype: DType::F32,
        };
        Ok(Self {
            dim,
            mi,
            top_k,
            max_rows,
            device: gpu.device_id,
            placement: placement.into(),
            masked_indices,
            state: RefCell::new(Transaction {
                input,
                mask,
                output,
                ids: [-1; MAX_SLOTS],
                gpu_ids: [-1; MAX_SLOTS],
                host_slots: [0; MAX_SLOTS],
                host_count: 0,
                rows: 0,
                ready: false,
                failed: false,
                cold_shadow: None,
                ret: ReturnProtocol::new(),
                token: None,
                mapped_call: false,
                waits_base: 0,
                cp_gen: 0,
            }),
            diag_on: Cell::new(false),
            recording: Cell::new(false),
            ref_pending: Cell::new(false),
            layer: Cell::new(0),
            parity: RefCell::new(ParityState {
                mode: SplitParityMode::Off,
                endpoint: SplitEndpoint::Natural,
                serial: false,
                shadow: None,
                max_records: 0,
                records: Vec::new(),
                open: None,
            }),
            mapped: None,
            return_refusal: None,
            output_copy_calls: Cell::new(0),
            mapped_publications: Cell::new(0),
            mapped_retire_waits: Cell::new(0),
            cpwait: None,
            ready: D2Ready::Sync,
            ready_polls: Cell::new(0),
            ready_fallback_syncs: Cell::new(0),
            join_refusal: None,
            cpu_trace: cpu_split_trace_enabled(),
        })
    }

    pub fn masked_indices(&self) -> &GpuTensor {
        &self.masked_indices
    }
    pub fn placement(&self) -> &[u8] {
        &self.placement
    }
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }
    /// Actually allocated row capacity of this executor (the `max_rows` it was
    /// built with, 1..=[`CPU_SPLIT_STORAGE_MAX_ROWS`]): the truthful bound the
    /// routing predicate clamps a qualified width against. Not a constant, not a
    /// default, not a qualification: a 64-capacity executor still routes 4-wide
    /// by default until a lab-wide verify scope qualifies wider.
    pub fn capacity_rows(&self) -> usize {
        self.max_rows
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn mi(&self) -> usize {
        self.mi
    }
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// The transport this executor's cold results use.
    pub fn return_mode(&self) -> D2ReturnMode {
        if self.mapped.is_some() {
            D2ReturnMode::Mapped
        } else {
            D2ReturnMode::Copy
        }
    }

    /// Why a requested mapped return was refused (the copy transport is kept).
    pub fn return_refusal(&self) -> Option<&str> {
        self.return_refusal.as_deref()
    }

    /// Bytes of the owned mapped cold-result allocation; 0 on the copy transport.
    pub fn mapped_capacity_bytes(&self) -> usize {
        self.mapped.as_ref().map_or(0, |m| m.view.buf.size())
    }

    /// `(output_copy_calls, mapped_publications, mapped_retire_waits)`, each
    /// cumulative: cold-slot H2D enqueues, mapped generations published and
    /// retirements that actually blocked.
    pub fn transport_counters(&self) -> (u64, u64, u64) {
        (
            self.output_copy_calls.get(),
            self.mapped_publications.get(),
            self.mapped_retire_waits.get(),
        )
    }

    /// The input readiness mechanism of this executor's `prepare`.
    pub fn ready_mode(&self) -> D2Ready {
        self.ready
    }

    /// `(ready_polls, ready_fallback_syncs)`, each cumulative: readiness polls
    /// (event queries) and the blocking stream synchronizations event readiness
    /// still needed.
    pub fn ready_counters(&self) -> (u64, u64) {
        (self.ready_polls.get(), self.ready_fallback_syncs.get())
    }

    /// Select event readiness (`HIPFIRE_QWEN4_D2_READY=event`). Call once, at
    /// load, before the first `prepare`. The default is the original stream
    /// synchronization. Either join uses the configured readiness.
    pub fn enable_event_ready(&mut self) -> Result<D2Ready, DispatchError> {
        let st = self.state.get_mut();
        if st.failed || st.ready {
            return Err(err("readiness selection with an active transaction"));
        }
        self.ready = D2Ready::Event;
        Ok(D2Ready::Event)
    }

    /// The join this executor uses.
    pub fn join_mode(&self) -> D2Join {
        if self.cpwait.is_some() {
            D2Join::CpWait
        } else {
            D2Join::Host
        }
    }

    /// Why a requested CP-wait join was refused (the host join is kept).
    pub fn join_refusal(&self) -> Option<&str> {
        self.join_refusal.as_deref()
    }

    /// `(cpwait_joins, cpwait_aborts, watchdog_fires)`, each cumulative: calls
    /// released with their generation, waits released with the abort value by
    /// an error path, and abort stores by the watchdog.
    pub fn cpwait_counters(&self) -> (u64, u64, u64) {
        match self.cpwait.as_ref() {
            Some(cp) => (
                cp.joins.get(),
                cp.gate.try_borrow().map_or(0, |g| g.aborts()),
                cp.target.fires(),
            ),
            None => (0, 0, 0),
        }
    }

    /// Switch this executor to the CP-wait join, or keep the host join and
    /// record why. Call once, at load, after the return mode is selected and
    /// before the first `prepare`, and only when the operator asked for
    /// `cpwait`. Capability routing, never silent: a copy transport, a mixed
    /// hot/cold plan, a device whose `hipDeviceAttributeCanUseStreamWaitValue`
    /// is 0, a failed signal allocation or watchdog start is a refusal that
    /// leaves [`Self::join_refusal`] set and the host join running.
    pub fn enable_cpwait_join(&mut self, gpu: &Gpu) -> Result<D2Join, DispatchError> {
        if self.cpwait.is_some() || self.join_refusal.is_some() {
            return Err(err("the join was already selected"));
        }
        {
            let st = self.state.get_mut();
            if st.failed || st.ready {
                return Err(err("join selection with an active transaction"));
            }
        }
        if gpu.device_id != self.device {
            return Err(err("join selection on another device"));
        }
        let precheck = select_join_mode(D2Join::CpWait, self.return_mode(), &self.placement, true);
        if let Some(reason) = precheck.refusal {
            self.join_refusal = Some(reason.to_string());
            return Ok(D2Join::Host);
        }
        if self.diag_on.get() {
            self.join_refusal =
                Some("a lab parity configuration is active (host join kept)".into());
            return Ok(D2Join::Host);
        }
        gpu.bind_thread().map_err(err)?;
        let capable = match gpu.hip.can_use_stream_wait_value(gpu.device_id) {
            Ok(capable) => capable,
            Err(error) => {
                self.join_refusal =
                    Some(format!("capability query failed: {error} (host join kept)"));
                return Ok(D2Join::Host);
            }
        };
        let decision =
            select_join_mode(D2Join::CpWait, self.return_mode(), &self.placement, capable);
        if let Some(reason) = decision.refusal {
            self.join_refusal = Some(reason.to_string());
            return Ok(D2Join::Host);
        }
        let watchdog = match crate::cpwait::global_watchdog() {
            Ok(watchdog) => watchdog,
            Err(error) => {
                self.join_refusal = Some(format!("{error} (host join kept)"));
                return Ok(D2Join::Host);
            }
        };
        let join_word = match gpu.hip.signal_word() {
            Ok(word) => word,
            Err(error) => {
                self.join_refusal = Some(format!(
                    "signal word allocation failed: {error} (host join kept)"
                ));
                return Ok(D2Join::Host);
            }
        };
        let abort_ptr = join_word.host_ptr();
        let target = Arc::new(JoinTarget::new(Box::new(move || {
            // SAFETY: the join word outlives every watchdog arm: `CpWait`
            // disarms its arm before its word drops, and `Watchdog::disarm` is
            // serialized with this call.
            unsafe { abort_ptr.store(SIGNAL_ABORT) }
        })));
        self.cpwait = Some(CpWait {
            gate: RefCell::new(JoinGate::new()),
            arm: Cell::new(None),
            target,
            watchdog,
            gen: Cell::new(SignalGen::new()),
            joins: Cell::new(0),
            join_word,
        });
        Ok(D2Join::CpWait)
    }

    /// Switch this executor's cold return to the mapped transport, or keep
    /// the copy transport and record why. Call once, at load, before the first
    /// `prepare`, and only when the operator asked for `mapped`. Capability
    /// routing, never a silent numeric fallback: a mixed hot/cold plan, an
    /// allocation failure or a missing/null device alias is a refusal that
    /// leaves [`Self::return_refusal`] set and the original copy transport
    /// running. A failed alias is NEVER replaced by the host pointer.
    pub fn enable_mapped_return(&mut self, gpu: &Gpu) -> Result<D2ReturnMode, DispatchError> {
        if self.mapped.is_some() || self.return_refusal.is_some() {
            return Err(err("the cold-result return was already selected"));
        }
        {
            let st = self.state.get_mut();
            if st.failed || st.ready {
                return Err(err(
                    "cold-result return selection with an active transaction",
                ));
            }
        }
        if gpu.device_id != self.device {
            return Err(err("cold-result return selection on another device"));
        }
        let decision = select_return_mode(D2Return::Mapped, &self.placement);
        if let Some(reason) = decision.refusal {
            self.return_refusal = Some(reason.to_string());
            return Ok(D2ReturnMode::Copy);
        }
        gpu.bind_thread().map_err(err)?;
        let (_, _, _, output_values) =
            cpu_split_storage_extents(self.max_rows, self.dim, self.top_k)
                .map_err(|e| err(format!("{e:?}")))?;
        let values = output_values;
        let bytes = values
            .checked_mul(4)
            .ok_or_else(|| err("mapped byte overflow"))?;
        let buffer = match gpu.hip.mapped_host_buffer(bytes) {
            Ok(buffer) => buffer,
            Err(error) => {
                self.return_refusal = Some(format!(
                    "mapped host allocation of {bytes} bytes failed: {error}"
                ));
                return Ok(D2ReturnMode::Copy);
            }
        };
        // SAFETY: `buffer` owns the mapped allocation and is stored next to
        // this Borrowed view in `MappedReturn`; the pair is only dropped
        // together (`free_gpu`, after a device synchronize) and the view is
        // never given to a free path.
        let view = GpuTensor {
            buf: unsafe { buffer.device_buffer().alias() },
            shape: vec![values],
            dtype: DType::F32,
        };
        self.mapped = Some(MappedReturn {
            view,
            buf: RefCell::new(buffer),
        });
        Ok(D2ReturnMode::Mapped)
    }

    /// Prove the mapped buffer's previous consumer finished: a cheap poll
    /// first, a blocking wait only if it is still outstanding (counted).
    fn prove_retired(&self, gpu: &Gpu, buf: &mut MappedHostBuffer) -> Result<(), DispatchError> {
        if buf.try_retire(&gpu.hip).map_err(err)? {
            return Ok(());
        }
        self.mapped_retire_waits
            .set(self.mapped_retire_waits.get() + 1);
        buf.retire(&gpu.hip).map_err(err)
    }

    /// The tensor the canonical combine must read as its expanded slot rows:
    /// `expanded` on the copy transport; on the mapped transport the published
    /// mapped view, handed out ONCE after a successful `join` of this
    /// generation. Refused before `join`, after a failed `join`, after the
    /// generation was consumed and on a second request. Pair every call with
    /// [`Self::combine_enqueued`] right after the combine is enqueued.
    pub fn combine_outputs<'a>(
        &'a self,
        expanded: &'a GpuTensor,
    ) -> Result<&'a GpuTensor, DispatchError> {
        let Some(mapped) = self.mapped.as_ref() else {
            return Ok(expanded);
        };
        if self.cpwait.is_some() {
            return Err(err(
                "combine_outputs on a CP-wait executor (use cpwait_arm)",
            ));
        }
        let mut st = self
            .state
            .try_borrow_mut()
            .map_err(|_| err("combine_outputs during an active transaction"))?;
        if st.failed {
            return Err(err("combine_outputs after a failed transaction"));
        }
        if st.ready {
            return Err(err("combine_outputs before join"));
        }
        match st.ret.lend().map_err(err)? {
            CombineSource::Expanded => Ok(expanded),
            CombineSource::Mapped => Ok(&mapped.view),
        }
    }

    /// Call right after the canonical combine is enqueued: on the mapped
    /// transport records the in-flight consumer of the published generation (a
    /// retirement event on the active stream) so the next `prepare` can prove
    /// it finished. No-op on the copy transport. An error poisons the
    /// transaction.
    pub fn combine_enqueued(&self, gpu: &Gpu) -> Result<(), DispatchError> {
        let Some(mapped) = self.mapped.as_ref() else {
            return Ok(());
        };
        if self.cpwait.is_some() {
            return Err(err(
                "combine_enqueued on a CP-wait executor (the join marks the consumer)",
            ));
        }
        let mut st = self
            .state
            .try_borrow_mut()
            .map_err(|_| err("combine_enqueued during an active transaction"))?;
        if st.failed {
            return Err(err("combine_enqueued after a failed transaction"));
        }
        let result = (|| {
            if !st.ret.enqueued().map_err(err)? {
                return Ok(());
            }
            let token = st
                .token
                .take()
                .ok_or_else(|| err("combine_enqueued lost the published token"))?;
            gpu.bind_thread().map_err(err)?;
            let mut buf = mapped
                .buf
                .try_borrow_mut()
                .map_err(|_| err("mapped return buffer is in use"))?;
            buf.mark_gpu_reading(&gpu.hip, token, gpu.active_stream.as_ref())
                .map_err(err)
        })();
        if result.is_err() {
            st.failed = true;
            st.ret.poison();
            self.drain_failed(gpu);
            if self.diag_on.get() {
                self.close_open();
            }
        }
        result
    }

    /// Drain after a failed transaction. A CP-wait executor aborts any queued
    /// wait BEFORE the device synchronize (a synchronize with a pending wait
    /// never returns) and poisons itself.
    fn drain_failed(&self, gpu: &Gpu) {
        match self.cpwait.as_ref() {
            Some(cp) => {
                let _ = cp.fail(gpu);
            }
            None => {
                let _ = gpu.hip.device_synchronize();
            }
        }
    }

    /// Whether this executor joins through the CP wait. The sealed program is
    /// lowered in the CP-wait order (combine before join) iff this is true.
    pub fn cpwait_active(&self) -> bool {
        self.cpwait.is_some()
    }

    /// CP-wait join, step 1: after the routed GPU tail is enqueued and BEFORE
    /// the CPU arithmetic, queue `hipStreamWaitValue32(join_word >= gen)` on the
    /// active stream and hand out the mapped view for the canonical combine,
    /// which the caller enqueues next on that stream. Nothing the stream queues
    /// afterwards runs until [`Self::join`] releases the wait (or an abort
    /// does), so no code may synchronize the stream or device between this call
    /// and `join`. A failure aborts (ABORT stored before any drain) and
    /// poisons; the request must be discarded.
    pub fn cpwait_arm<'a>(&'a self, gpu: &Gpu) -> Result<&'a GpuTensor, DispatchError> {
        let (Some(cp), Some(mapped)) = (self.cpwait.as_ref(), self.mapped.as_ref()) else {
            return Err(err("cpwait_arm on an executor without the CP-wait join"));
        };
        let mut st = self
            .state
            .try_borrow_mut()
            .map_err(|_| err("cpwait_arm during an active transaction"))?;
        if st.failed {
            return Err(err("cpwait_arm after a failed transaction"));
        }
        if !st.ready || !st.mapped_call {
            return Err(err("cpwait_arm outside a prepared mapped call"));
        }
        let result = (|| {
            if cp.target.is_poisoned() {
                return Err(err("the join watchdog fired; the CP-wait join is poisoned"));
            }
            self.gate(gpu, st.rows)?;
            st.ret.arm_gated().map_err(err)?;
            let gen = st.cp_gen;
            // Armed BEFORE the enqueue: a failed call may still have queued the
            // packet, so from here every error must abort.
            cp.gate.borrow_mut().arm(gen).map_err(err)?;
            cp.arm.set(Some(cp.watchdog.arm(cp.target.clone())));
            gpu.hip
                .stream_wait_signal_gte(gpu.active_stream.as_ref(), &cp.join_word, gen)
                .map_err(err)
        })();
        if let Err(e) = result {
            st.failed = true;
            st.ret.poison();
            let _ = cp.fail(gpu);
            return Err(e);
        }
        Ok(&mapped.view)
    }

    /// Release a queued CP wait with the abort value and poison, without any
    /// device call. For an owner that is abandoning a call between
    /// [`Self::cpwait_arm`] and [`Self::join`] (a failed later stage, a dropped
    /// program): the combine then consumes garbage and the request must be
    /// discarded. A no-op when no wait is queued.
    pub fn abort_armed(&self) {
        let Some(cp) = self.cpwait.as_ref() else {
            return;
        };
        if !cp.abort_only() {
            return;
        }
        if let Ok(mut st) = self.state.try_borrow_mut() {
            st.failed = true;
            st.ret.poison();
        }
    }

    /// Wait for the three input downloads of `prepare`, then retire the older
    /// mask/output transfers, by the configured readiness ([`input_wait`]) under
    /// either join. `Sync` is the original boundary: three stream
    /// synchronizations, byte for byte. `Event` arms a proof after the downloads
    /// and spins on it ([`READY_SPIN_BUDGET`]); a spin that expires, or an older
    /// buffer the proof does not cover, falls back to the blocking
    /// synchronization and is counted.
    fn await_inputs(&self, gpu: &Gpu, st: &mut Transaction) -> Result<(), DispatchError> {
        match input_wait(self.ready) {
            InputWait::StreamSync => {
                st.input.complete(&gpu.hip).map_err(err)?;
                // These may synchronize the stream, intentionally BEFORE its tail.
                st.mask.complete(&gpu.hip).map_err(err)?;
                st.output.complete(&gpu.hip).map_err(err)
            }
            InputWait::EventSpin => self.await_event_ready(gpu, st),
        }
    }

    fn await_event_ready(&self, gpu: &Gpu, st: &mut Transaction) -> Result<(), DispatchError> {
        let mut proof = st.input.arm_ready(&gpu.hip).map_err(err)?;
        let start = Instant::now();
        let outcome = spin_until_ready(
            || st.input.try_complete_ready(&gpu.hip, &mut proof),
            || start.elapsed(),
            READY_SPIN_BUDGET,
        )
        .map_err(err)?;
        self.ready_polls
            .set(self.ready_polls.get() + outcome.polls());
        if let SpinOutcome::TimedOut { .. } = outcome {
            st.input.complete_ready(&gpu.hip, &mut proof).map_err(err)?;
            self.ready_fallback_syncs
                .set(self.ready_fallback_syncs.get() + 1);
        }
        // The completed proof retires every older same-stream transfer.
        for older in [&mut st.mask, &mut st.output] {
            if !older.retire_behind(&proof).map_err(err)? {
                older.complete(&gpu.hip).map_err(err)?;
                self.ready_fallback_syncs
                    .set(self.ready_fallback_syncs.get() + 1);
            }
        }
        Ok(())
    }

    /// Freeze one diagnostic run for this layer. Replaces any previous
    /// configuration and drops its records. Takes no GPU: the next `prepare`
    /// retires all pinned owners before touching them, so no drain is needed
    /// here, but an in-flight (prepared, un-joined) or failed transaction is
    /// refused. `Off` restores production exactly and requires a natural,
    /// non-serial config. A layer outside `layers` runs production unchanged.
    pub fn configure(&self, layer: usize, cfg: SplitParityConfig) -> Result<(), DispatchError> {
        let st = self
            .state
            .try_borrow()
            .map_err(|_| err("configure during an active transaction"))?;
        if st.failed || st.ready {
            return Err(err("configure with a failed or unfinished transaction"));
        }
        drop(st);
        let active = cfg.mode != SplitParityMode::Off;
        if active && self.cpwait.is_some() {
            return Err(err(
                "parity lab modes drain the stream and are refused under the CP-wait join",
            ));
        }
        if !active {
            if cfg.endpoint != SplitEndpoint::Natural || cfg.serial {
                return Err(err(
                    "parity mode Off requires the natural endpoint and serial=false",
                ));
            }
        } else if cfg.max_records == 0 || cfg.max_records > SPLIT_PARITY_MAX_RECORDS {
            return Err(err(format!(
                "max_records must be 1..={SPLIT_PARITY_MAX_RECORDS}"
            )));
        }
        if cfg.endpoint == SplitEndpoint::AllCold && cfg.shadow.is_none() {
            return Err(err("AllCold endpoint requires a CPU shadow"));
        }
        let wanted = active && cfg.layers.as_ref().map_or(true, |l| l.contains(&layer));
        let mut p = self
            .parity
            .try_borrow_mut()
            .map_err(|_| err("configure re-entered"))?;
        self.recording.set(false);
        self.ref_pending.set(false);
        self.layer.set(layer);
        p.open = None;
        p.records = Vec::new();
        if wanted {
            p.mode = cfg.mode;
            p.endpoint = cfg.endpoint;
            p.serial = cfg.serial;
            p.shadow = cfg.shadow;
            p.max_records = cfg.max_records;
        } else {
            p.mode = SplitParityMode::Off;
            p.endpoint = SplitEndpoint::Natural;
            p.serial = false;
            p.shadow = None;
            p.max_records = 0;
        }
        self.diag_on.set(wanted);
        Ok(())
    }

    /// Drain recorded parity records in call order.
    pub fn take_parity_records(&self) -> Vec<SplitParityRecord> {
        std::mem::take(&mut self.parity.borrow_mut().records)
    }

    /// True from a successful `prepare` that opened a record until that record
    /// completes (after `parity_after_combine` in Timing, `parity_reference` in
    /// Full) or errors. False in production and for unrecorded calls.
    pub fn wants_parity(&self) -> bool {
        self.recording.get()
    }

    /// True while the open record is Full and still awaits the unmasked GPU
    /// reference tail + `parity_reference`.
    pub fn wants_reference(&self) -> bool {
        self.ref_pending.get()
    }

    fn close_open(&self) {
        self.recording.set(false);
        self.ref_pending.set(false);
        if let Ok(mut p) = self.parity.try_borrow_mut() {
            p.open = None;
        }
    }

    /// Shared hook frame: no-op unless a record is open; validates rows and
    /// eager state; any error discards the open record. `f` returns true when
    /// the record is complete and must be pushed.
    fn hook(
        &self,
        gpu: &Gpu,
        rows: usize,
        f: impl FnOnce(&Gpu, &mut Open) -> Result<bool, DispatchError>,
    ) -> Result<(), DispatchError> {
        if !self.recording.get() {
            return Ok(());
        }
        let mut p = self.parity.borrow_mut();
        let result = (|| {
            let open = p
                .open
                .as_mut()
                .ok_or_else(|| err("parity record not open"))?;
            if open.rec.rows != rows {
                return Err(err("parity rows differ from prepare"));
            }
            self.gate(gpu, rows)?;
            f(gpu, open)
        })();
        match result {
            Ok(false) => Ok(()),
            Ok(true) => {
                let open = p.open.take().ok_or_else(|| err("parity record vanished"))?;
                p.records.push(open.rec);
                self.recording.set(false);
                self.ref_pending.set(false);
                Ok(())
            }
            Err(e) => {
                p.open = None;
                self.recording.set(false);
                self.ref_pending.set(false);
                Err(e)
            }
        }
    }

    /// Call after join, BEFORE the canonical combine. Full: drains the stream and
    /// captures `target` (combine input) and `expanded` as the combine will read
    /// them (lab only, serializes). Timing: no-op. Production/unrecorded: no-op.
    pub fn parity_before_combine(
        &self,
        gpu: &Gpu,
        rows: usize,
        expanded: &GpuTensor,
        target: &GpuTensor,
    ) -> Result<(), DispatchError> {
        self.hook(gpu, rows, |gpu, open| {
            if open.stage != Stage::Joined {
                return Err(err("parity_before_combine out of order"));
            }
            if open.full {
                let (nt, ne) = (rows * self.dim, rows * self.top_k * self.dim);
                sync_stream(gpu)?;
                open.rec.target_in = read_f32(gpu, target, nt)?;
                open.rec.d2_expanded = read_f32(gpu, expanded, ne)?;
                open.stage = Stage::Before;
            }
            Ok(false)
        })
    }

    /// Call right after the canonical combine is enqueued. Drains the stream
    /// (the one timing sync), stamps wall/tail times BEFORE any readback, then
    /// Timing completes the record; Full captures the combined `target` and
    /// waits for `parity_reference`.
    pub fn parity_after_combine(
        &self,
        gpu: &Gpu,
        rows: usize,
        target: &GpuTensor,
    ) -> Result<(), DispatchError> {
        self.hook(gpu, rows, |gpu, open| {
            let expect = if open.full {
                Stage::Before
            } else {
                Stage::Joined
            };
            if open.stage != expect {
                return Err(err("parity_after_combine out of order"));
            }
            sync_stream(gpu)?;
            let now = Instant::now();
            open.rec.timing.wall_ns = ns(open.t0, now);
            open.rec.timing.tail_ns = ns(open.t_join_end, now);
            if open.full {
                open.rec.d2_target = read_f32(gpu, target, rows * self.dim)?;
                open.stage = Stage::After;
                Ok(false)
            } else {
                Ok(true)
            }
        })
    }

    /// Full only: call after the caller reran the UNMASKED all-GPU tail into the
    /// same `expanded` (after the combine consumed it). Captures `gpu_expanded`
    /// and completes the record. No-op for Timing/unrecorded.
    pub fn parity_reference(
        &self,
        gpu: &Gpu,
        rows: usize,
        expanded: &GpuTensor,
    ) -> Result<(), DispatchError> {
        self.hook(gpu, rows, |gpu, open| {
            if !open.full {
                return Ok(false);
            }
            if open.stage != Stage::After {
                return Err(err("parity_reference out of order"));
            }
            sync_stream(gpu)?;
            open.rec.gpu_expanded = read_f32(gpu, expanded, rows * self.top_k * self.dim)?;
            Ok(true)
        })
    }

    fn resolve<'a>(
        &self,
        gpu: &Gpu,
        experts: &'a dyn RoutedExpertWeights,
        shadow: Option<&'a dyn CpuShadow>,
        id: usize,
    ) -> Option<CpuMoeExpert<'a>> {
        match shadow {
            Some(s) => s.expert(self.layer.get(), id),
            None => experts.cpu_splice_expert(gpu, id),
        }
    }

    fn gate(&self, gpu: &Gpu, rows: usize) -> Result<(), DispatchError> {
        if rows == 0 || rows > self.max_rows || gpu.device_id != self.device {
            return Err(err("invalid width or device"));
        }
        let capture = gpu
            .active_stream
            .as_ref()
            .map(|s| gpu.hip.stream_is_capturing(s))
            .transpose()
            .map_err(err)?
            .unwrap_or(false);
        if gpu.graphs.capture_mode
            || gpu.graphs.verify.capturing.is_some()
            || gpu.graphs.replay.capturing.is_some()
            || capture
            || gpu.replay.is_recording()
            || gpu.replay.retained_body_active()
            || gpu.replay.should_route_aql()
            || gpu.replay.should_route_pm4()
            || gpu.replay.shadow_body_route().is_some()
            || gpu.replay.state() == rdna_compute::replay::ReplayState::ShadowValidated
        {
            return Err(err(
                "requires eager HIP; capture, retained replay, AQL/PM4 and shadow are refused",
            ));
        }
        gpu.bind_thread().map_err(err)
    }

    /// Download ORIGINAL route/rotated activation and retire all old pinned
    /// reads before launching any routed GPU kernels for this generation.
    pub fn prepare(
        &self,
        gpu: &Gpu,
        rows: usize,
        x_rot: &GpuTensor,
        ids: &GpuTensor,
        weights: &GpuTensor,
        experts: &dyn RoutedExpertWeights,
    ) -> Result<(), DispatchError> {
        let diag = self.diag_on.get();
        let t0 = diag.then(Instant::now);
        self.gate(gpu, rows)?;
        let rows_x = rows
            .checked_mul(self.dim)
            .ok_or_else(|| err("input overflow"))?;
        let slots_x = (rows
            .checked_mul(self.top_k)
            .ok_or_else(|| err("slot overflow")))?;
        if x_rot.buf.size() < rows_x * 4
            || ids.buf.size() < slots_x * 4
            || weights.buf.size() < slots_x * 4
            || experts.len() != self.placement.len()
        {
            return Err(err("input extent or expert plan mismatch"));
        }
        // Admission may run on a different thread than orchestration. Reserve
        // its reusable CPU scratch here, before the parallel GPU/CPU phase.
        let capacity_slots = self
            .max_rows
            .checked_mul(self.top_k)
            .ok_or_else(|| err("slot overflow"))?;
        cpu_exec::reserve_moe_cpu_host_slot_scratch(self.dim, self.mi, capacity_slots)?;
        let (endpoint, shadow) = if diag {
            // A record left open by an aborted caller never leaks into this call.
            self.close_open();
            let p = self.parity.borrow();
            let shadow = if p.endpoint == SplitEndpoint::AllCold {
                Some(
                    p.shadow
                        .clone()
                        .ok_or_else(|| err("AllCold endpoint lost its CPU shadow"))?,
                )
            } else {
                None
            };
            (p.endpoint, shadow)
        } else {
            (SplitEndpoint::Natural, None)
        };
        let mut st = self.state.borrow_mut();
        if st.failed || st.ready {
            return Err(err("failed or unfinished transaction"));
        }
        let result = (|| {
            if let Some(cp) = self.cpwait.as_ref() {
                if cp.target.is_poisoned() {
                    return Err(err("the join watchdog fired; the CP-wait join is poisoned"));
                }
                cp.gate.borrow_mut().begin_call().map_err(err)?;
                st.cp_gen = cp.next_gen()?;
            }
            let retire_first = if self.mapped.is_some() {
                st.waits_base = self.mapped_retire_waits.get();
                st.ret.admit_prepare().map_err(err)?
            } else {
                false
            };
            st.input
                .download_async(
                    &gpu.hip,
                    &x_rot.buf,
                    0,
                    rows_x * 4,
                    gpu.active_stream.as_ref(),
                )
                .map_err(err)?;
            st.input
                .download_async(
                    &gpu.hip,
                    &ids.buf,
                    rows_x * 4,
                    slots_x * 4,
                    gpu.active_stream.as_ref(),
                )
                .map_err(err)?;
            st.input
                .download_async(
                    &gpu.hip,
                    &weights.buf,
                    (rows_x + slots_x) * 4,
                    slots_x * 4,
                    gpu.active_stream.as_ref(),
                )
                .map_err(err)?;
            self.await_inputs(gpu, &mut st)?;
            if retire_first {
                // The input proof (a synchronization, or the event that
                // follows the previous layer's work on this stream) puts the
                // previous generation's combine behind it, so this is a poll,
                // not a wait. Nothing can write the buffer before it.
                let mapped = self
                    .mapped
                    .as_ref()
                    .ok_or_else(|| err("mapped retirement without a mapped return"))?;
                let mut buf = mapped
                    .buf
                    .try_borrow_mut()
                    .map_err(|_| err("mapped return buffer is in use"))?;
                self.prove_retired(gpu, &mut buf)?;
                st.ret.retired();
            }
            for s in 0..slots_x {
                let at = (rows_x + s) * 4;
                let id =
                    i32::from_ne_bytes(st.input.as_bytes_mut()[at..at + 4].try_into().unwrap());
                st.ids[s] = id;
                st.gpu_ids[s] = -1;
            }
            st.host_count = 0;
            for s in 0..slots_x {
                let id = st.ids[s];
                // Invalid original slots belong to neither arm; preserve them
                // for the existing router-fault reporting/canonical reducer.
                if id < 0 || id as usize >= self.placement.len() {
                    continue;
                }
                let cold = match endpoint {
                    SplitEndpoint::Natural => self.placement[id as usize] == 1,
                    SplitEndpoint::AllHot => false,
                    SplitEndpoint::AllCold => true,
                };
                if cold {
                    let n = st.host_count;
                    st.host_slots[n] = s as u32;
                    st.host_count += 1;
                } else {
                    st.gpu_ids[s] = id;
                }
            }
            self.validate_payloads(gpu, &st, rows, experts, shadow.as_deref())?;
            for s in 0..slots_x {
                let bytes = st.gpu_ids[s].to_ne_bytes();
                st.mask.as_bytes_mut()[s * 4..s * 4 + 4].copy_from_slice(&bytes);
            }
            st.mask
                .upload_async(
                    &gpu.hip,
                    &self.masked_indices.buf,
                    0,
                    slots_x * 4,
                    gpu.active_stream.as_ref(),
                )
                .map_err(err)?;
            st.mapped_call = self.mapped.is_some() && endpoint != SplitEndpoint::AllHot;
            st.rows = rows;
            st.ready = true;
            Ok(())
        })();
        if result.is_err() {
            st.failed = true;
            // Keep this transaction poisoned even after a successful drain:
            // a partially advanced model request must never silently retry.
            self.drain_failed(gpu);
            if diag {
                self.close_open();
            }
        } else {
            st.cold_shadow = shadow;
            if let Some(t0) = t0 {
                self.open_record(&mut st, t0, rows, endpoint);
            }
        }
        result
    }

    /// Lab only: open the parity record for a successfully prepared call.
    fn open_record(&self, st: &mut Transaction, t0: Instant, rows: usize, endpoint: SplitEndpoint) {
        let prepare_ns = ns(t0, Instant::now());
        let slots = rows
            .checked_mul(self.top_k)
            .expect("open record after checked prepare");
        let nx = rows
            .checked_mul(self.dim)
            .expect("open record after checked prepare");
        let mut p = self.parity.borrow_mut();
        if p.records.len() >= p.max_records {
            return;
        }
        let full = p.mode == SplitParityMode::Full;
        let cold = st.host_count;
        let hot = st.gpu_ids[..slots].iter().filter(|&&v| v >= 0).count();
        let mut rec = SplitParityRecord {
            layer: self.layer.get(),
            rows,
            dim: self.dim,
            mi: self.mi,
            top_k: self.top_k,
            endpoint,
            placement: Vec::new(),
            x_rot: Vec::new(),
            ids: Vec::new(),
            weights: Vec::new(),
            target_in: Vec::new(),
            d2_expanded: Vec::new(),
            d2_target: Vec::new(),
            gpu_expanded: Vec::new(),
            transport: SplitTransport::default(),
            timing: SplitTiming {
                prepare_ns,
                cold_slots: cold as u32,
                hot_slots: hot as u32,
                invalid_slots: (slots - hot - cold) as u32,
                serial: p.serial,
                ..SplitTiming::default()
            },
        };
        if full {
            rec.placement = self.placement.to_vec();
            rec.ids = st.ids[..slots].to_vec();
            let bytes = st.input.as_bytes_mut();
            let f32s = |range: std::ops::Range<usize>| -> Vec<f32> {
                bytes[range.start * 4..range.end * 4]
                    .chunks_exact(4)
                    .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
                    .collect()
            };
            rec.x_rot = f32s(0..nx);
            rec.weights = f32s(nx + slots..nx + 2 * slots);
        }
        let now = Instant::now();
        p.open = Some(Open {
            t0,
            t_prep_end: now,
            t_join_end: now,
            stage: Stage::Prepared,
            full,
            rec,
        });
        self.recording.set(true);
        self.ref_pending.set(full);
    }

    fn validate_payloads(
        &self,
        gpu: &Gpu,
        st: &Transaction,
        rows: usize,
        experts: &dyn RoutedExpertWeights,
        shadow: Option<&dyn CpuShadow>,
    ) -> Result<(), DispatchError> {
        let dummy = CpuMoeExpert {
            gate_up_quant: CpuQuant::F32,
            gate_up: &[],
            down_quant: CpuQuant::F32,
            down: &[],
        };
        // `resolved` is preallocated executor-width staging (640 slots): filled
        // once per call, no per-window heap growth beyond this fixed array.
        let mut resolved = [dummy; MAX_SLOTS];
        for (j, &slot) in st.host_slots[..st.host_count].iter().enumerate() {
            let id = st.ids[slot as usize] as usize;
            resolved[j] = self
                .resolve(gpu, experts, shadow, id)
                .ok_or_else(|| err(format!("cold expert {id} has no retained host spans")))?;
        }
        cpu_exec::validate_moe_cpu_host_slot_outputs(
            self.dim,
            self.mi,
            rows,
            self.top_k,
            self.placement.len(),
            &st.ids[..rows * self.top_k],
            Some(&st.gpu_ids[..rows * self.top_k]),
            &st.host_slots[..st.host_count],
            &resolved[..st.host_count],
            rows * self.dim,
            false,
            rows * self.top_k * self.dim,
        )
    }

    /// GPU down is already queued. NO stream completion belongs in this
    /// method: it would turn the intended CPU/GPU overlap into serialization.
    /// (Lab `serial=true` is the one deliberate exception: a single drain
    /// before the CPU arithmetic.) Cold slots are installed as one H2D per run
    /// of adjacent slots: identical bytes and stream order, fewer HIP calls.
    pub fn join(
        &self,
        gpu: &Gpu,
        rows: usize,
        expanded: &GpuTensor,
        experts: &dyn RoutedExpertWeights,
    ) -> Result<(), DispatchError> {
        // Trace-only entry clock: BEFORE gate/validation so `join_total_ns`
        // spans the whole successful join. Existing parity `tj0` semantics
        // unchanged (it still starts after the state checks).
        // Every new *_start_ns / *_end_ns uses this SAME Instant origin and
        // saturated nanosecond conversion as the existing CPU join trace.
        // These are CPU clock enclosures; the retirement API's profiler
        // interval constrains clock translation, never an assumed offset.
        let trace_on = self.cpu_trace && self.cpwait.is_some();
        let trace_entry = trace_on.then(Instant::now);
        self.gate(gpu, rows)?;
        let slots = rows
            .checked_mul(self.top_k)
            .ok_or_else(|| err("slot overflow"))?;
        let out_values = slots
            .checked_mul(self.dim)
            .ok_or_else(|| err("output overflow"))?;
        if expanded.buf.size() < out_values * 4 {
            return Err(err("short expanded output"));
        }
        let mut st = self.state.borrow_mut();
        if st.failed || !st.ready || st.rows != rows {
            return Err(err("join without matching prepare"));
        }
        let diag = self.diag_on.get();
        let tj0 = diag.then(Instant::now);
        let serial = diag && self.parity.borrow().serial;
        let shadow = st.cold_shadow.clone();
        let copies0 = self.output_copy_calls.get();
        let publications0 = self.mapped_publications.get();
        let result = (|| {
            if serial {
                sync_stream(gpu)?;
            }
            let dummy = CpuMoeExpert {
                gate_up_quant: CpuQuant::F32,
                gate_up: &[],
                down_quant: CpuQuant::F32,
                down: &[],
            };
            let mut resolved = [dummy; MAX_SLOTS];
            for (j, &slot) in st.host_slots[..st.host_count].iter().enumerate() {
                resolved[j] = self
                    .resolve(
                        gpu,
                        experts,
                        shadow.as_deref(),
                        st.ids[slot as usize] as usize,
                    )
                    .ok_or_else(|| err("immutable cold payload disappeared"))?;
            }
            let resolve_ns = trace_entry.map_or(0, |t| ns(t, Instant::now()));
            let Transaction {
                input,
                output,
                ids,
                gpu_ids,
                host_slots,
                host_count,
                ret,
                token,
                mapped_call,
                ..
            } = &mut *st;
            // SAFETY: hipHostMalloc storage is aligned, initialized and idle.
            // Input completion and previous output retirement happened BEFORE
            // the GPU tail; neither buffer has been submitted since then.
            let x = unsafe {
                let n_x = rows
                    .checked_mul(self.dim)
                    .ok_or_else(|| err("input overflow"))?;
                std::slice::from_raw_parts(input.as_bytes_mut().as_ptr().cast::<f32>(), n_x)
            };
            if *mapped_call {
                // Mapped return: the cold slots go straight into the mapped
                // allocation the canonical combine will read; NO cold-slot H2D.
                let mapped = self
                    .mapped
                    .as_ref()
                    .ok_or_else(|| err("mapped call without a mapped return"))?;
                if let Some(cp) = self.cpwait.as_ref() {
                    if cp.target.is_poisoned() {
                        return Err(err("the join watchdog fired; the CP-wait join is poisoned"));
                    }
                    ret.begin_join_gated().map_err(err)?;
                } else {
                    ret.begin_join().map_err(err)?;
                }
                let mut buf = mapped
                    .buf
                    .try_borrow_mut()
                    .map_err(|_| err("mapped return buffer is in use"))?;
                // `prepare` retired the previous consumer; prove it again
                // before the first host store (no-op when already Retired).
                let t_prove = trace_on.then(Instant::now);
                self.prove_retired(gpu, &mut buf)?;
                let prove_ns = t_prove.map_or(0, |t| ns(t, Instant::now()));
                let mut write = buf.begin_write().map_err(err)?;
                let all = write.f32_mut().map_err(err)?;
                let rows_out = &mut all[..out_values];
                let t_cpu = (diag || trace_on).then(Instant::now);
                let arm = if trace_on {
                    Some(cpu_exec::moe_cpu_host_slot_outputs_traced(
                        self.dim,
                        self.mi,
                        rows,
                        self.top_k,
                        self.placement.len(),
                        &ids[..slots],
                        Some(&gpu_ids[..slots]),
                        &host_slots[..*host_count],
                        &resolved[..*host_count],
                        x,
                        false,
                        rows_out,
                        trace_entry.expect("trace_on supplies the join trace epoch"),
                    )?)
                } else {
                    cpu_exec::moe_cpu_host_slot_outputs(
                        self.dim,
                        self.mi,
                        rows,
                        self.top_k,
                        self.placement.len(),
                        &ids[..slots],
                        Some(&gpu_ids[..slots]),
                        &host_slots[..*host_count],
                        &resolved[..*host_count],
                        x,
                        false,
                        rows_out,
                    )?;
                    None
                };
                // Slots owned by neither arm hold zero in the copy transport's
                // expanded rows (masked zero stores); match those bytes.
                let t_zero = trace_on.then(Instant::now);
                zero_unwritten_slots(rows_out, self.dim, &host_slots[..*host_count]);
                let (zero_ns, zero_slots) = match t_zero {
                    Some(t) => (ns(t, Instant::now()), (slots - *host_count) as u64),
                    None => (0, 0),
                };
                let cpu_ns = t_cpu.map_or(0, |t| ns(t, Instant::now()));
                let t_publish = trace_on.then(Instant::now);
                let published = write.publish();
                let publish_ns = t_publish.map_or(0, |t| ns(t, Instant::now()));
                if let Some(cp) = self.cpwait.as_ref() {
                    // The combine is already queued behind the wait: record its
                    // retirement event at the stream tail, publish the ledger
                    // transition, then release the wait with this generation.
                    let t_mark = trace_on.then(Instant::now);
                    buf.mark_gpu_reading(&gpu.hip, published, gpu.active_stream.as_ref())
                        .map_err(err)?;
                    let retirement_end_ns = trace_entry.map(|epoch| ns(epoch, Instant::now()));
                    let mark_ns = t_mark.map_or(0, |t| ns(t, Instant::now()));
                    // Release spans ALL ledger/watchdog/release bookkeeping
                    // after the mark: joined_gated, the publication counter,
                    // the wait release, and the join counter.
                    let t_release = trace_on.then(Instant::now);
                    ret.joined_gated();
                    self.mapped_publications
                        .set(self.mapped_publications.get() + 1);
                    let release_start_ns = trace_entry.map(|epoch| ns(epoch, Instant::now()));
                    cp.gate
                        .borrow_mut()
                        .release(&CpPort { cp, gpu: Some(gpu) })
                        .map_err(err)?;
                    let release_end_ns = trace_entry.map(|epoch| ns(epoch, Instant::now()));
                    cp.joins.set(cp.joins.get() + 1);
                    let release_ns = t_release.map_or(0, |t| ns(t, Instant::now()));
                    let trace = arm.map(|t| CpuJoinTrace {
                        resolve_ns,
                        prove_ns,
                        trace: t,
                        zero_ns,
                        zero_slots,
                        publish_ns,
                        mark_ns,
                        release_ns,
                        retirement_start_ns: trace_entry
                            .zip(t_mark)
                            .map(|(epoch, start)| ns(epoch, start)),
                        retirement_end_ns,
                        release_start_ns,
                        release_end_ns,
                        // The same non-owning device view used to queue the wait;
                        // never the host-mapped address or an executor surrogate.
                        join_word_device: cp.join_word.device_buffer().as_ptr() as usize,
                        join_total_ns: trace_entry.map_or(0, |t| ns(t, Instant::now())),
                    });
                    return Ok((cpu_ns, trace));
                }
                *token = Some(published);
                ret.published();
                self.mapped_publications
                    .set(self.mapped_publications.get() + 1);
                return Ok((cpu_ns, None));
            }
            let out = unsafe {
                std::slice::from_raw_parts_mut(
                    output.as_bytes_mut().as_mut_ptr().cast::<f32>(),
                    out_values,
                )
            };
            let t_cpu = diag.then(Instant::now);
            cpu_exec::moe_cpu_host_slot_outputs(
                self.dim,
                self.mi,
                rows,
                self.top_k,
                self.placement.len(),
                &ids[..slots],
                Some(&gpu_ids[..slots]),
                &host_slots[..*host_count],
                &resolved[..*host_count],
                x,
                false,
                out,
            )?;
            let cpu_ns = t_cpu.map_or(0, |t| ns(t, Instant::now()));
            let hs = &host_slots[..*host_count];
            let mut i = 0;
            while i < hs.len() {
                // Row-major ascending slots: adjacent slots are adjacent in both
                // the pinned output and `expanded`, so a run is one contiguous copy.
                let mut j = i + 1;
                while j < hs.len() && hs[j] == hs[j - 1] + 1 {
                    j += 1;
                }
                let offset = hs[i] as usize * self.dim * 4;
                let len = (j - i) * self.dim * 4;
                // Non-owning DeviceBuffer view: no shape Vec allocation and no
                // GPU arithmetic. Same-stream FIFO orders the down zero stores
                // before these writes and the canonical combine after them.
                let dst = expanded.buf.byte_view(offset, len);
                output
                    .upload_async(&gpu.hip, &dst, offset, len, gpu.active_stream.as_ref())
                    .map_err(err)?;
                self.output_copy_calls.set(self.output_copy_calls.get() + 1);
                i = j;
            }
            if self.mapped.is_some() {
                ret.bypassed();
            }
            Ok((cpu_ns, None))
        })();
        st.ready = false;
        match result {
            Ok((cpu_ns, arm)) => {
                if let Some(tj0) = tj0 {
                    if self.recording.get() {
                        let now = Instant::now();
                        let mut p = self.parity.borrow_mut();
                        if let Some(o) = p.open.as_mut() {
                            o.rec.timing.gpu_enqueue_ns = ns(o.t_prep_end, tj0);
                            o.rec.timing.join_cpu_ns = cpu_ns;
                            o.rec.timing.join_total_ns = ns(tj0, now);
                            o.t_join_end = now;
                            o.stage = Stage::Joined;
                            o.rec.transport = SplitTransport {
                                mode: if st.mapped_call {
                                    D2ReturnMode::Mapped
                                } else {
                                    D2ReturnMode::Copy
                                },
                                output_copy_calls: self.output_copy_calls.get() - copies0,
                                mapped_publications: self.mapped_publications.get() - publications0,
                                mapped_retire_waits: self.mapped_retire_waits.get() - st.waits_base,
                            };
                        }
                    }
                }
                // Opt-in diagnostic: exactly one numeric JSON line per mapped +
                // CP-wait successful join, AFTER generation release and all
                // publication bookkeeping above. Error/abort paths return early
                // and emit nothing. No output while a generation is unreleased.
                if let (true, Some(j)) = (trace_on, arm) {
                    // EVERY successful mapped + CP-wait join, including a
                    // zero-host-slot call (parser admits unique_experts == 0).
                    if self.cpwait.is_some() {
                        let t = &j.trace;
                        eprintln!(
                            "QWEN4_CPU_ARM {{\"rows\":{rows},\"dim\":{},\"top_k\":{},\"host_count\":{},\"unique_experts\":{},\"executor\":{},\"cpwait\":1,\"cp_gen\":{},\"cp_joins\":{},\"resolve_ns\":{},\"prove_ns\":{},\"cpu_total_ns\":{},\"validate_ns\":{},\"pool_install_ns\":{},\"pool_body_ns\":{},\"group_ns\":{},\"gu_ns\":{},\"gu_prep_ns\":{},\"gu_exec_ns\":{},\"middle_ns\":{},\"gather_ns\":{},\"silu_ns\":{},\"fwht_ns\":{},\"dn_ns\":{},\"dn_prep_ns\":{},\"dn_exec_ns\":{},\"scatter_ns\":{},\"zero_ns\":{},\"zero_slots\":{},\"publish_ns\":{},\"mark_ns\":{},\"release_ns\":{},\"join_total_ns\":{},\"middle_start_ns\":{},\"middle_end_ns\":{},\"scatter_start_ns\":{},\"scatter_end_ns\":{},\"retirement_start_ns\":{},\"retirement_end_ns\":{},\"release_start_ns\":{},\"release_end_ns\":{},\"join_word_device\":{}}}",
                            self.dim,
                            self.top_k,
                            st.host_count,
                            t.unique_experts,
                            self as *const Self as usize,
                            st.cp_gen,
                            self.cpwait.as_ref().map_or(0, |cp| cp.joins.get()),
                            j.resolve_ns,
                            j.prove_ns,
                            t.cpu_total_ns,
                            t.validate_ns,
                            t.pool_install_ns,
                            t.pool_body_ns,
                            t.group_ns,
                            t.gu_ns,
                            t.gu_prep_ns,
                            t.gu_exec_ns,
                            t.middle_ns,
                            t.gather_ns,
                            t.silu_ns,
                            t.fwht_ns,
                            t.dn_ns,
                            t.dn_prep_ns,
                            t.dn_exec_ns,
                            t.scatter_ns,
                            j.zero_ns,
                            j.zero_slots,
                            j.publish_ns,
                            j.mark_ns,
                            j.release_ns,
                            j.join_total_ns,
                            TraceNs(t.middle_start_ns),
                            TraceNs(t.middle_end_ns),
                            TraceNs(t.scatter_start_ns),
                            TraceNs(t.scatter_end_ns),
                            TraceNs(j.retirement_start_ns),
                            TraceNs(j.retirement_end_ns),
                            TraceNs(j.release_start_ns),
                            TraceNs(j.release_end_ns),
                            j.join_word_device,
                        );
                    }
                }
                Ok(())
            }
            Err(e) => {
                st.failed = true;
                st.ret.poison();
                self.drain_failed(gpu);
                if diag {
                    self.close_open();
                }
                Err(e)
            }
        }
    }

    /// Per-layer live host<->device transfer volume: original x/ids/weights
    /// download, masked ids upload, and on the copy transport just the cold F32
    /// expanded slot returns (H2D). On the mapped transport no cold slot is
    /// copied: that return is [`Self::mapped_return_bytes`]. No residual.
    /// Checked: overflow reports `(usize::MAX, usize::MAX)` (a malformed width
    /// the executor itself would refuse) so byte census can never wrap.
    pub fn transfer_bytes(&self, rows: usize, cold_slots: usize) -> (usize, usize) {
        Self::transfer_bytes_for(
            self.dim,
            self.top_k,
            self.mapped.is_some(),
            rows,
            cold_slots,
        )
    }

    /// Device-free transfer census behind [`Self::transfer_bytes`]: the same
    /// checked byte formula over explicit geometry, so lab tests can pin the
    /// 64-row numbers without a device.
    pub fn transfer_bytes_for(
        dim: usize,
        top_k: usize,
        mapped: bool,
        rows: usize,
        cold_slots: usize,
    ) -> (usize, usize) {
        let up: Option<usize> = (|| {
            4usize
                .checked_mul(rows)?
                .checked_mul(dim.checked_add(2 * top_k)?)
        })();
        let returned = if mapped {
            Some(0)
        } else {
            (|| 4usize.checked_mul(cold_slots)?.checked_mul(dim))()
        };
        let down: Option<usize> = (|| {
            4usize
                .checked_mul(rows)?
                .checked_mul(top_k)?
                .checked_add(returned?)
        })();
        match (up, down) {
            (Some(up), Some(down)) => (up, down),
            _ => (usize::MAX, usize::MAX),
        }
    }

    /// Bytes the canonical combine reads remotely from the mapped allocation
    /// for `cold_slots` cold slot rows (PCIe reads, not H2D copies); 0 on the
    /// copy transport. Checked: overflow reports `usize::MAX`.
    pub fn mapped_return_bytes(&self, cold_slots: usize) -> usize {
        if self.mapped.is_some() {
            4usize
                .checked_mul(cold_slots)
                .and_then(|v| v.checked_mul(self.dim))
                .unwrap_or(usize::MAX)
        } else {
            0
        }
    }

    /// Storage byte census of a `max_rows`-row executor: `(input, mask,
    /// output)` live pinned/device bytes (f32 elements × 4).
    /// Checked: overflow reports `(usize::MAX, usize::MAX, usize::MAX)`.
    pub fn storage_bytes(max_rows: usize, dim: usize, top_k: usize) -> (usize, usize, usize) {
        let bytes: Option<(usize, usize, usize)> = (|| {
            let (_slots, input_values, mask_values, output_values) =
                cpu_split_storage_extents(max_rows, dim, top_k).ok()?;
            Some((
                input_values.checked_mul(4)?,
                mask_values.checked_mul(4)?,
                output_values.checked_mul(4)?,
            ))
        })();
        bytes.unwrap_or((usize::MAX, usize::MAX, usize::MAX))
    }

    pub fn free_gpu(self, gpu: &Gpu) -> Result<(), HipError> {
        gpu.bind_thread()?;
        // A queued CP wait is released with the abort value BEFORE any drain: a
        // device synchronize with a pending wait never returns.
        self.abort_armed();
        if let Err(e) = gpu.hip.device_synchronize() {
            // Both pinned copy sources, the mapped cold-result allocation and
            // the mask allocation may still be referenced. Retain the whole
            // owner if retirement is unproved.
            std::mem::forget(self);
            return Err(e);
        }
        let Self {
            masked_indices,
            state,
            mapped,
            cpwait,
            ..
        } = self;
        // The CP-wait word goes first: it stores the abort value, drains the
        // device and frees itself, and the watchdog arm is already disarmed.
        drop(cpwait);
        drop(state);
        // Drop the owning allocation (it quiesces its consumer itself and
        // retains the memory if that fails), then the Borrowed view with it.
        // The view is never passed to `hip.free`.
        drop(mapped);
        gpu.hip.free(masked_indices.buf)
    }
}

#[cfg(test)]
mod return_tests {
    use super::*;
    use hipfire_config::memory::parse_d2_return;

    /// One whole generation: prepare, join, hand out the view, enqueue.
    fn run_generation(p: &mut ReturnProtocol) {
        assert_eq!(p.admit_prepare(), Ok(p.stage == ReturnStage::InFlight));
        p.retired();
        p.begin_join().unwrap();
        p.published();
        assert_eq!(p.lend(), Ok(CombineSource::Mapped));
        assert_eq!(p.enqueued(), Ok(true));
    }

    #[test]
    fn legacy_default_scope_stays_four_wide() {
        assert_eq!(CPU_SPLIT_MAX_ROWS, 4);
        assert_eq!(CPU_SPLIT_STORAGE_MAX_ROWS, 64);
        assert_eq!(
            CPU_SPLIT_STORAGE_MAX_ROWS,
            hipfire_config::QWEN4_VERIFY_MAX_ROWS
        );
        assert_eq!(MAX_SLOTS, 640);
        // Default route: decode + the proven 2..4 verify widths, whole window only.
        for rows in 1..=4usize {
            assert!(cpu_splice_window(true, rows, rows));
            assert!(cpu_splice_window_qualified(
                true,
                rows,
                rows,
                CPU_SPLIT_MAX_ROWS
            ));
        }
        for rows in [0usize, 5, 8, 64] {
            assert!(!cpu_splice_window(true, rows, rows));
        }
        // A short tail of a long prefill is never the whole routing window.
        assert!(!cpu_splice_window(true, 4, 8));
        assert!(!cpu_splice_window_qualified(true, 4, 8, 64));
        // Qualified lab-wide scope admits the full 1..64 window; unqualified stays 4.
        for rows in [5usize, 8, 33, 64] {
            assert!(cpu_splice_window_qualified(
                true,
                rows,
                rows,
                CPU_SPLIT_STORAGE_MAX_ROWS
            ));
            assert!(cpu_splice_window_qualified(true, rows, rows, 64));
        }
        assert!(!cpu_splice_window_qualified(
            true,
            65,
            65,
            CPU_SPLIT_STORAGE_MAX_ROWS
        ));
        assert!(!cpu_splice_window_qualified(
            true,
            64,
            32,
            CPU_SPLIT_STORAGE_MAX_ROWS
        ));
        assert!(!cpu_splice_window_qualified(
            false,
            1,
            1,
            CPU_SPLIT_STORAGE_MAX_ROWS
        ));
    }

    #[test]
    fn storage_extents_cover_64_rows_at_top_k_10() {
        let (slots, input, mask, output) = cpu_split_storage_extents(64, 256, 10).unwrap();
        assert_eq!(slots, 640);
        assert_eq!(input, 64 * 256 + 2 * 640);
        assert_eq!(mask, 640);
        assert_eq!(output, 640 * 256);
        let (in_b, mask_b, out_b) = CpuSplitExecutor::storage_bytes(64, 256, 10);
        assert_eq!(in_b, input * 4);
        assert_eq!(mask_b, mask * 4);
        assert_eq!(out_b, output * 4);
        // Transport census for the worst-case all-cold copy call: x/ids/weights
        // down the input side, masked ids + every cold slot back. Device-free
        // formula pins the exact 64-row numbers (dim 256, top_k 10).
        let (up, down) = CpuSplitExecutor::transfer_bytes_for(256, 10, false, 64, 640);
        assert_eq!(up, 4 * 64 * (256 + 20));
        assert_eq!(down, 4 * 64 * 10 + 4 * 640 * 256);
        let (mup, mdown) = CpuSplitExecutor::transfer_bytes_for(256, 10, true, 64, 640);
        assert_eq!((mup, mdown), (up, 4 * 64 * 10));
        // Malformed widths refuse before any device allocation exists.
        assert!(cpu_split_storage_extents(0, 256, 10).is_err());
        assert!(cpu_split_storage_extents(65, 256, 10).is_err());
        assert_eq!(
            CpuSplitExecutor::storage_bytes(65, 256, 10),
            (usize::MAX, usize::MAX, usize::MAX)
        );
    }

    #[test]
    fn combine_outputs_refuses_before_join() {
        let mut p = ReturnProtocol::new();
        assert!(p.lend().is_err());
        assert_eq!(p.stage, ReturnStage::Idle);
        // A prepared-but-unjoined generation is still `Idle`.
        assert_eq!(p.admit_prepare(), Ok(false));
        assert!(p.lend().is_err());
    }

    #[test]
    fn combine_outputs_refuses_after_a_failed_join() {
        let mut p = ReturnProtocol::new();
        p.admit_prepare().unwrap();
        p.poison();
        assert!(p.lend().is_err());
        assert!(p.enqueued().is_err());
        assert!(
            p.admit_prepare().is_err(),
            "a poisoned return never serves another call"
        );
        assert!(p.begin_join().is_err());
    }

    #[test]
    fn combine_outputs_hands_out_the_view_once_per_generation() {
        let mut p = ReturnProtocol::new();
        p.admit_prepare().unwrap();
        p.begin_join().unwrap();
        p.published();
        assert_eq!(p.lend(), Ok(CombineSource::Mapped));
        assert!(p.lend().is_err(), "twice");
        assert_eq!(p.enqueued(), Ok(true));
        assert!(p.lend().is_err(), "after the token was consumed");
    }

    #[test]
    fn combine_enqueued_requires_a_published_and_lent_generation() {
        let mut p = ReturnProtocol::new();
        assert!(p.enqueued().is_err(), "nothing published");
        p.admit_prepare().unwrap();
        assert!(p.enqueued().is_err(), "prepared only");
        p.begin_join().unwrap();
        p.published();
        assert!(
            p.enqueued().is_err(),
            "published but combine_outputs never asked"
        );
        assert_eq!(p.lend(), Ok(CombineSource::Mapped));
        assert_eq!(p.enqueued(), Ok(true));
        assert!(p.enqueued().is_err(), "twice");
    }

    #[test]
    fn next_prepare_retires_the_consumer_before_any_write() {
        let mut p = ReturnProtocol::new();
        run_generation(&mut p);
        assert_eq!(p.stage, ReturnStage::InFlight);
        assert!(
            p.begin_join().is_err(),
            "no write over an unretired consumer"
        );
        assert_eq!(
            p.admit_prepare(),
            Ok(true),
            "next prepare must retire first"
        );
        assert!(
            p.begin_join().is_err(),
            "still unretired until the proof lands"
        );
        p.retired();
        p.begin_join().unwrap();
    }

    #[test]
    fn repeated_generations_alternate_retire_and_publish() {
        let mut p = ReturnProtocol::new();
        for generation in 0..8 {
            assert_eq!(
                p.stage == ReturnStage::InFlight,
                generation > 0,
                "generation {generation}"
            );
            run_generation(&mut p);
        }
    }

    #[test]
    fn a_skipped_combine_enqueued_is_a_protocol_error_at_the_next_prepare() {
        let mut p = ReturnProtocol::new();
        p.admit_prepare().unwrap();
        p.begin_join().unwrap();
        p.published();
        assert!(p.admit_prepare().is_err(), "published, never handed out");
        let mut q = ReturnProtocol::new();
        q.admit_prepare().unwrap();
        q.begin_join().unwrap();
        q.published();
        q.lend().unwrap();
        assert!(
            q.admit_prepare().is_err(),
            "handed out, never marked in flight"
        );
    }

    #[test]
    fn a_bypassed_call_reads_the_expanded_rows_and_never_marks_a_token() {
        let mut p = ReturnProtocol::new();
        run_generation(&mut p);
        assert_eq!(p.admit_prepare(), Ok(true));
        p.retired();
        p.begin_join().unwrap();
        p.bypassed();
        assert_eq!(p.lend(), Ok(CombineSource::Expanded));
        assert_eq!(p.enqueued(), Ok(false));
        assert_eq!(p.stage, ReturnStage::Idle);
        // An un-enqueued bypassed call does not block the next prepare.
        p.begin_join().unwrap();
        p.bypassed();
        assert_eq!(p.admit_prepare(), Ok(false));
        p.begin_join().unwrap();
    }

    #[test]
    fn mixed_placement_selects_copy_with_a_reason() {
        let decision = select_return_mode(D2Return::Mapped, &[1, 1, 0, 1]);
        assert_eq!(decision.mode, D2ReturnMode::Copy);
        assert!(decision
            .refusal
            .is_some_and(|r| r.contains("mixed hot/cold")));
        let all_hot = select_return_mode(D2Return::Mapped, &[0, 0, 0]);
        assert_eq!(all_hot.mode, D2ReturnMode::Copy);
        assert!(all_hot.refusal.is_some());
        let empty = select_return_mode(D2Return::Mapped, &[]);
        assert_eq!(empty.mode, D2ReturnMode::Copy);
        assert!(empty.refusal.is_some());
    }

    #[test]
    fn all_cold_placement_selects_mapped() {
        let decision = select_return_mode(D2Return::Mapped, &[1; 512]);
        assert_eq!(
            decision,
            ReturnDecision {
                mode: D2ReturnMode::Mapped,
                refusal: None
            }
        );
    }

    #[test]
    fn the_default_and_the_kill_value_select_copy_without_a_refusal() {
        for placement in [&[1u8, 1, 1][..], &[1, 0][..]] {
            let decision = select_return_mode(D2Return::Copy, placement);
            assert_eq!(
                decision,
                ReturnDecision {
                    mode: D2ReturnMode::Copy,
                    refusal: None
                }
            );
        }
        assert_eq!(D2ReturnMode::default(), D2ReturnMode::Copy);
    }

    #[test]
    fn flag_spellings_drive_the_selection() {
        let all_cold = [1u8; 8];
        let request =
            |raw: Option<&str>| parse_d2_return(raw).map(|r| select_return_mode(r, &all_cold).mode);
        assert_eq!(request(None), Ok(D2ReturnMode::Copy));
        assert_eq!(request(Some("copy")), Ok(D2ReturnMode::Copy));
        assert_eq!(request(Some("mapped")), Ok(D2ReturnMode::Mapped));
        assert!(request(Some("pinned")).is_err());
        assert!(request(Some("")).is_err());
    }

    #[test]
    fn unwritten_slots_are_zeroed_and_host_slots_untouched() {
        let dim = 3;
        let mut out = vec![f32::NAN; 5 * dim];
        for slot in [1usize, 2, 4] {
            out[slot * dim..(slot + 1) * dim].fill(slot as f32);
        }
        zero_unwritten_slots(&mut out, dim, &[1, 2, 4]);
        for slot in 0..5usize {
            let want = if [1, 2, 4].contains(&slot) {
                slot as f32
            } else {
                0.0
            };
            assert!(
                out[slot * dim..(slot + 1) * dim]
                    .iter()
                    .all(|v| v.to_bits() == want.to_bits()),
                "slot {slot}"
            );
        }
        let mut none = vec![f32::NAN; 2 * dim];
        zero_unwritten_slots(&mut none, dim, &[]);
        assert!(none.iter().all(|v| v.to_bits() == 0));
        let mut all = vec![7.0f32; 2 * dim];
        zero_unwritten_slots(&mut all, dim, &[0, 1]);
        assert!(all.iter().all(|&v| v == 7.0));
    }

    // ── CP-wait join ──

    /// One whole CP-wait generation: prepare, arm (combine queued), join.
    fn run_gated_generation(p: &mut ReturnProtocol) {
        assert_eq!(p.admit_prepare(), Ok(p.stage == ReturnStage::InFlight));
        p.retired();
        assert_eq!(p.arm_gated(), Ok(CombineSource::Mapped));
        p.begin_join_gated().unwrap();
        p.joined_gated();
    }

    #[test]
    fn a_gated_generation_arms_before_it_joins_and_retires_at_the_next_prepare() {
        let mut p = ReturnProtocol::new();
        for generation in 0..6 {
            assert_eq!(
                p.stage == ReturnStage::InFlight,
                generation > 0,
                "generation {generation}"
            );
            run_gated_generation(&mut p);
            assert_eq!(p.stage, ReturnStage::InFlight);
        }
    }

    #[test]
    fn a_gated_join_needs_an_armed_generation_and_the_host_join_refuses_one() {
        let mut p = ReturnProtocol::new();
        assert!(p.begin_join_gated().is_err(), "join before arm");
        p.admit_prepare().unwrap();
        assert!(p.begin_join_gated().is_err(), "prepared only");
        p.arm_gated().unwrap();
        assert!(p.arm_gated().is_err(), "armed twice");
        assert!(
            p.begin_join().is_err(),
            "the host join never writes an armed generation"
        );
        assert!(
            p.lend().is_err() && p.enqueued().is_err(),
            "the host combine path is closed"
        );
        assert!(
            p.admit_prepare().is_err(),
            "an armed generation that never joined blocks the next call"
        );
    }

    #[test]
    fn an_armed_generation_cannot_start_over_an_unretired_consumer_or_a_poisoned_return() {
        let mut p = ReturnProtocol::new();
        run_gated_generation(&mut p);
        assert!(
            p.arm_gated().is_err(),
            "the previous consumer is still in flight"
        );
        p.poison();
        assert!(p.arm_gated().is_err());
        assert!(p.begin_join_gated().is_err());
    }

    #[test]
    fn cpwait_is_refused_on_the_copy_transport_and_on_mixed_placements() {
        let all_cold = [1u8; 8];
        let mixed = [1u8, 0, 1, 1];
        let host = select_join_mode(D2Join::Host, D2ReturnMode::Mapped, &all_cold, true);
        assert_eq!(
            host,
            JoinDecision {
                mode: D2Join::Host,
                refusal: None
            }
        );
        let copy = select_join_mode(D2Join::CpWait, D2ReturnMode::Copy, &all_cold, true);
        assert_eq!(copy.mode, D2Join::Host);
        assert!(copy
            .refusal
            .is_some_and(|r| r.contains("needs the mapped return")));
        let mix = select_join_mode(D2Join::CpWait, D2ReturnMode::Mapped, &mixed, true);
        assert_eq!(mix.mode, D2Join::Host);
        assert!(mix.refusal.is_some_and(|r| r.contains("mixed hot/cold")));
        let mix_copy = select_join_mode(D2Join::CpWait, D2ReturnMode::Copy, &mixed, true);
        assert_eq!(mix_copy.mode, D2Join::Host);
        assert!(mix_copy
            .refusal
            .is_some_and(|r| r.contains("mixed hot/cold")));
        let empty = select_join_mode(D2Join::CpWait, D2ReturnMode::Mapped, &[], true);
        assert!(empty.refusal.is_some());
        assert_eq!(empty.mode, D2Join::Host);
    }

    #[test]
    fn cpwait_arms_no_readiness_word_and_uses_the_configured_readiness() {
        // Both joins wait for the inputs through the one configured mechanism.
        assert_eq!(input_wait(D2Ready::Sync), InputWait::StreamSync);
        assert_eq!(input_wait(D2Ready::Event), InputWait::EventSpin);
        // The executor arms no readiness word: no production code names one.
        let production = include_str!("cpu_split.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for word_path in ["arm_ready_word", "try_complete_ready_word", "ready_word"] {
            assert!(
                !production.contains(word_path),
                "{word_path} reappeared in the executor"
            );
        }
    }

    #[test]
    fn cpwait_is_refused_without_the_stream_wait_value_capability() {
        let refused = select_join_mode(D2Join::CpWait, D2ReturnMode::Mapped, &[1; 4], false);
        assert_eq!(refused.mode, D2Join::Host);
        assert!(refused
            .refusal
            .is_some_and(|r| r.contains("hipDeviceAttributeCanUseStreamWaitValue")));
    }

    #[test]
    fn cpwait_is_selected_on_a_capable_all_cold_mapped_layer() {
        let ok = select_join_mode(D2Join::CpWait, D2ReturnMode::Mapped, &[1; 512], true);
        assert_eq!(
            ok,
            JoinDecision {
                mode: D2Join::CpWait,
                refusal: None
            }
        );
    }

    #[test]
    fn join_flag_spellings_drive_the_selection() {
        use hipfire_config::memory::parse_d2_join;
        let request = |raw: Option<&str>| {
            parse_d2_join(raw)
                .map(|r| select_join_mode(r, D2ReturnMode::Mapped, &[1; 8], true).mode)
        };
        assert_eq!(request(None), Ok(D2Join::Host));
        assert_eq!(request(Some("host")), Ok(D2Join::Host));
        assert_eq!(request(Some("cpwait")), Ok(D2Join::CpWait));
        assert!(request(Some("wait")).is_err());
        assert!(request(Some("")).is_err());
    }

    #[test]
    fn ready_flag_spellings_parse() {
        use hipfire_config::memory::parse_d2_ready;
        assert_eq!(parse_d2_ready(None), Ok(D2Ready::Sync));
        assert_eq!(parse_d2_ready(Some("sync")), Ok(D2Ready::Sync));
        assert_eq!(parse_d2_ready(Some("event")), Ok(D2Ready::Event));
        assert!(parse_d2_ready(Some("spin")).is_err());
        assert!(parse_d2_ready(Some("")).is_err());
    }
}
