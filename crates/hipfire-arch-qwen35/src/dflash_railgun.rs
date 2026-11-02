// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Railgun design A slice A6: the certified-window owner, admission, and the
//! window transaction (`docs/design/2026-10-25-dflash-railgun-window.md` §6, §8).
//!
//! [`RailgunWindowOwner`] is owned by the `DflashSpeculator` (one per
//! speculator, built only when `HIPFIRE_DFLASH_RAILGUN_WINDOW=1` is read at
//! build). `spec_step_dflash_with_railgun` asks it, per window:
//!
//! 1. [`admit_static`] (once per owner state: exact gfx1201, dense 27B-class
//!    qwen35, FP8 KV, fixed-16 all-sliding DFlash2 draft, no tree / retained
//!    PM4 / online tuner) and [`admit_window`] (greedy temp 0, no CACTUS /
//!    PLD / n-gram / repeat penalty / diagnostics, fixed B=16, lookup
//!    Copy/Probe stay on HIP, armed Tail lookup is `Whole16` only when
//!    `NoTailAction` proves every legal tail decision declines, otherwise
//!    `HostSplice`).
//! 2. [`RailgunWindowOwner::run`]: the §6 transaction. Everything before the
//!    queue is touched is refusable to the ordinary HIP route with nothing
//!    changed; everything after is not retried.
//!
//! An unsupported window takes the existing HIP route with a counted,
//! log-once [`RailgunRefusal`]. The GPU-facing half — turning the A4/A5
//! emissions into an owned `PreparedWindow` — is the [`WindowBuilder`] the
//! integration installs; without one every window refuses with
//! [`RailgunRefusal::EngineUnavailable`] and runs on HIP.

use std::collections::{BTreeMap, HashSet};

use hip_bridge::{HipError, HipResult, Stream};
use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashWeights, DraftCtxMode};
use railgun::window::{
    BoundWindow, PreparedWindow, Quiescence, Refusal as WindowRefusal, WindowBindings, WindowCache,
    WindowExecution, WindowHostDelta, WindowInputs, WindowShape,
};
use rdna_compute::Gpu;

use crate::dflash_draft_emit::{draft_family_key, DraftFamilyKey};
use crate::speculative::{
    DeltaNetSnapshot, GdnTape, HiddenCommitDelta, HiddenStateRingBuffer, ModelSlot, VerifyScratch,
};

/// The developer variable (read once at speculator build; `1` enables).
pub const RAILGUN_WINDOW_VAR: &str = "HIPFIRE_DFLASH_RAILGUN_WINDOW";

/// The fixed chain window of the certified route.
pub const WINDOW_ROWS: usize = 16;

/// Qwen3.5-27B dense dimensions (`dim` / FFN `hidden_dim` / GQA group).
const DENSE_27B_DIM: usize = 5120;
const DENSE_27B_FFN: usize = 17408;
const DENSE_27B_GQA_GROUP: usize = 6;

/// `HIPFIRE_DFLASH_RAILGUN_WINDOW`, `1` = on. Read once by the caller.
pub fn window_var_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_DFLASH_RAILGUN_WINDOW")
        .ok()
        .as_deref()
        == Some("1")
}

// ------------------------------------------------------------- refusals

/// Why a window ran on the ordinary HIP route. `key()` is the counter / log
/// key; the same reason is logged once per owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RailgunRefusal {
    // static (owner-level)
    NotGfx1201,
    NotDenseQwen35,
    Not27bClass,
    KvNotFp8,
    DraftNotFixed16,
    DraftNotAllSliding,
    TreeVerify,
    RetainedPm4,
    OnlineTune,
    // per window
    Sampling,
    Cactus,
    Pld,
    NgramOverride,
    RepeatPenalty,
    HostPath,
    Diagnostics,
    CtxSlice,
    NoGdnTape,
    BlockNot16,
    LookupCopy,
    LookupProbe,
    ColdSeedOrRebuild,
    ShapeNotWarm,
    /// `Whole16` only: the verify's GDN requant reads the stochastic-rounding
    /// frame (Q8 state, EF off), and the incumbent HIP-graph verify bakes the
    /// frame at capture (a replay reserves none), so a railgun window cannot
    /// reproduce its bytes. With EF on the requant is deterministic round-to-
    /// nearest and ignores the frame (`kernels/src/gated_delta_net_q8.hip:158-163`,
    /// `gated_delta_net_q8_fast.hip:533-535`); FP32 and Q4 state take no frame.
    GdnFrameDependent,
    /// The window manifest does not admit this window's node sequence (or its
    /// bind witnesses). Decided on the host right after bind, before anything is
    /// staged, sealed or enqueued, so nothing needs undoing.
    ManifestRefused,
    /// `HIPFIRE_DFLASH_RAILGUN_DEBUG`: the pre-submit self-check found the
    /// lowered state (kernarg buffers, IB dispatch dimensions) different from
    /// the bound images and grids. Nothing was dispatched; the first mismatch
    /// is printed.
    LoweringSelfCheck,
    /// This window's skeleton is not built yet: its planning and lowering run
    /// off the critical path (a background build), so this window takes the
    /// HIP route and later windows at the shape use the skeleton.
    SkeletonBuilding,
    // engine
    EngineUnavailable,
    PrepareRefused,
    BindRefused,
    StageFailed,
    SubmitRefused,
}

impl RailgunRefusal {
    pub fn key(self) -> &'static str {
        match self {
            Self::NotGfx1201 => "not_gfx1201",
            Self::NotDenseQwen35 => "not_dense_qwen35",
            Self::Not27bClass => "not_27b_class",
            Self::KvNotFp8 => "kv_not_fp8",
            Self::DraftNotFixed16 => "draft_not_fixed16",
            Self::DraftNotAllSliding => "draft_not_all_sliding",
            Self::TreeVerify => "tree_verify",
            Self::RetainedPm4 => "retained_pm4",
            Self::OnlineTune => "online_tune",
            Self::Sampling => "sampling",
            Self::Cactus => "cactus",
            Self::Pld => "pld",
            Self::NgramOverride => "ngram_override",
            Self::RepeatPenalty => "repeat_penalty",
            Self::HostPath => "host_path",
            Self::Diagnostics => "diagnostics",
            Self::CtxSlice => "ctx_slice",
            Self::NoGdnTape => "no_gdn_tape",
            Self::BlockNot16 => "block_not_16",
            Self::LookupCopy => "lookup_copy",
            Self::LookupProbe => "lookup_probe",
            Self::ColdSeedOrRebuild => "cold_seed_or_rebuild",
            Self::ShapeNotWarm => "shape_not_warm",
            Self::GdnFrameDependent => "gdn_frame_dependent",
            Self::ManifestRefused => "manifest_refused",
            Self::LoweringSelfCheck => "lowering_self_check",
            Self::SkeletonBuilding => "skeleton_building",
            Self::EngineUnavailable => "engine_unavailable",
            Self::PrepareRefused => "prepare_refused",
            Self::BindRefused => "bind_refused",
            Self::StageFailed => "stage_failed",
            Self::SubmitRefused => "submit_refused",
        }
    }
}

// ------------------------------------------------------------ admission

/// Facts of the loaded owners [`admit_static`] needs.
#[derive(Clone, Debug)]
pub struct StaticFacts {
    pub arch: String,
    pub num_experts: usize,
    pub dim: usize,
    pub ffn_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub kv_fp8: bool,
    /// Any other KV quantization (Q8/asym/HFQ4/FWHT/BF16/INT8) is set.
    pub kv_other_quant: bool,
    pub draft_block: usize,
    pub ctx_mode: DraftCtxMode,
    pub retained_pm4_admitted: bool,
    pub online_tuner: bool,
}

/// Owner-level admission (design §8): exact gfx1201, single GPU (the
/// singleton path only; the dense-TP mesh and VMM lanes never reach here),
/// dense 27B-class qwen35, FP8 KV, fixed-16 all-sliding draft.
pub fn admit_static(f: &StaticFacts) -> Result<(), RailgunRefusal> {
    if f.arch != "gfx1201" {
        return Err(RailgunRefusal::NotGfx1201);
    }
    if f.num_experts != 0 {
        return Err(RailgunRefusal::NotDenseQwen35);
    }
    if f.dim != DENSE_27B_DIM
        || f.ffn_dim != DENSE_27B_FFN
        || f.n_kv_heads == 0
        || f.n_heads / f.n_kv_heads != DENSE_27B_GQA_GROUP
    {
        return Err(RailgunRefusal::Not27bClass);
    }
    if !f.kv_fp8 || f.kv_other_quant {
        return Err(RailgunRefusal::KvNotFp8);
    }
    if f.draft_block != WINDOW_ROWS {
        return Err(RailgunRefusal::DraftNotFixed16);
    }
    match f.ctx_mode {
        DraftCtxMode::Windowed { w, w_full } if w == w_full => {}
        _ => return Err(RailgunRefusal::DraftNotAllSliding),
    }
    if f.retained_pm4_admitted {
        return Err(RailgunRefusal::RetainedPm4);
    }
    if f.online_tuner {
        return Err(RailgunRefusal::OnlineTune);
    }
    Ok(())
}

/// What the lookup layer decided for this window, as far as railgun cares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookupFacts {
    /// No lookup state, or a window that offers nothing.
    None,
    /// Pre-draft Copy window: keeps its HIP owner.
    Copy,
    /// Price probe: keeps its HIP owner.
    Probe,
    /// Tail offered and `NoTailAction` proved every legal decision declines.
    TailDeclinesAll,
    /// Tail offered and some decision may take: the host splice is real.
    TailMayAct,
}

/// Per-window facts [`admit_window`] needs.
#[derive(Clone, Debug)]
pub struct WindowFacts {
    pub b: usize,
    pub greedy: bool,
    pub cactus: bool,
    pub pld: bool,
    pub ngram_cache: bool,
    pub repeat_penalty: bool,
    pub host_path: bool,
    pub diagnostics: bool,
    pub ctx_slice: bool,
    pub gdn_tape: bool,
    /// The target's GDN state requant ignores the stochastic-rounding frame:
    /// FP32/Q4 state, or Q8 with error feedback on (deterministic rounding).
    /// False for Q8 with EF off, where the verify's bytes depend on a frame the
    /// graph bakes and railgun cannot reproduce.
    pub gdn_frame_independent: bool,
    pub lookup: LookupFacts,
}

/// Window-level admission. `Ok(execution)` picks `Whole16` or `HostSplice`.
pub fn admit_window(f: &WindowFacts) -> Result<WindowExecution, RailgunRefusal> {
    if !f.greedy {
        return Err(RailgunRefusal::Sampling);
    }
    if f.cactus {
        return Err(RailgunRefusal::Cactus);
    }
    if f.pld {
        return Err(RailgunRefusal::Pld);
    }
    if f.ngram_cache {
        return Err(RailgunRefusal::NgramOverride);
    }
    if f.repeat_penalty {
        return Err(RailgunRefusal::RepeatPenalty);
    }
    if f.host_path {
        return Err(RailgunRefusal::HostPath);
    }
    if f.diagnostics {
        return Err(RailgunRefusal::Diagnostics);
    }
    if f.ctx_slice {
        return Err(RailgunRefusal::CtxSlice);
    }
    if !f.gdn_tape {
        return Err(RailgunRefusal::NoGdnTape);
    }
    match f.lookup {
        LookupFacts::Copy => return Err(RailgunRefusal::LookupCopy),
        LookupFacts::Probe => return Err(RailgunRefusal::LookupProbe),
        _ => {}
    }
    if f.b != WINDOW_ROWS {
        return Err(RailgunRefusal::BlockNot16);
    }
    match f.lookup {
        // The host splice runs the draft segment only: no GDN verify node.
        LookupFacts::TailMayAct => Ok(WindowExecution::HostSplice),
        _ if !f.gdn_frame_independent => Err(RailgunRefusal::GdnFrameDependent),
        _ => Ok(WindowExecution::Whole16),
    }
}

// -------------------------------------------------------------- counters

/// Windows by route, so a perf run can prove which route ran.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RailgunCounters {
    pub whole16: u64,
    pub host_splice: u64,
    /// HIP windows with the refusal that sent them there.
    pub hip_refused: BTreeMap<&'static str, u64>,
}

impl RailgunCounters {
    pub fn hip_total(&self) -> u64 {
        self.hip_refused.values().sum()
    }
}

// --------------------------------------------------------------- engine

/// Everything a [`WindowBuilder`] may touch for one window. `gpu`/owners are
/// the live singleton's; the builder plans nothing it cannot undo.
pub struct WindowRequest<'a> {
    pub gpu: &'a mut Gpu,
    pub target: &'a mut ModelSlot,
    pub draft_weights: &'a DflashWeights,
    pub draft_cfg: &'a DflashConfig,
    pub draft_scratch: &'a mut DflashScratch,
    pub hidden_rb: &'a mut HiddenStateRingBuffer,
    pub verify_scratch: &'a VerifyScratch,
    pub gdn_tape: Option<&'a mut GdnTape>,
    /// Everything the owner knows; `max_tiles` is left 0 for the builder to
    /// fill (attention partial-buffer geometry is its binding key's).
    pub shape: WindowShape,
    pub seed: u32,
    pub mask_token: u32,
    pub execution: WindowExecution,
}

/// A prepared window's bound view, as returned by the builder.
pub struct PreparedInputs {
    /// `shape` with `max_tiles` resolved.
    pub shape: WindowShape,
    pub bindings: WindowBindings,
    pub inputs: WindowInputs,
}

/// Proposals (and, for `Whole16`, verify picks) read back after completion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowResults {
    /// `[seed, d1..d15]`, the proposal block.
    pub block: Vec<u32>,
    /// The 16 target picks (`Whole16` only).
    pub argmax_per_pos: Option<Vec<u32>>,
}

/// The `Refusal::Malformed` message a [`WindowBuilder::prepare`] returns when
/// the window's skeleton is being built in the background: the owner counts it
/// as [`RailgunRefusal::SkeletonBuilding`] (a HIP window), not a failure.
pub const SKELETON_BUILDING: &str = "skeleton building in the background";

/// GPU-facing half of a certified window. The integration implements it by
/// planning the A4 draft and A5 verify emissions, running A2 admission,
/// authoring the owned program and creating the `PreparedWindow` over an
/// `OwnedWindowLowering` `WindowQueue`.
///
/// Contract:
/// * `prepare` runs only for a shape whose family already ran once on HIP
///   (lazy scratch/JIT materialized), may advance the global GDN requant
///   frame counter exactly as HIP would (the owner restores it on any
///   pre-queue refusal), and mutates no GPU or host window state.
/// * `stage` enqueues the window inputs (pinned pack + one async H2D + one
///   fence record) on `stream`; `await_staged` is the one host wait on that
///   fence.
/// * `read_results` runs after `Submission::complete`.
pub trait WindowBuilder: Send {
    fn prepare<'c>(
        &mut self,
        cache: &'c mut WindowCache,
        request: &mut WindowRequest<'_>,
    ) -> Result<(&'c mut PreparedWindow, PreparedInputs), WindowRefusal>;

    /// Manifest admission of this window's plan and (assumed) bind witnesses.
    /// Pure host work, called right after `bind` and before anything is
    /// mutated or enqueued: a refusal here leaves the HIP fallback an
    /// untouched preimage. With a manifest, an admitted plan is certified
    /// exact on `bound` here when it has no conditional node, and after the
    /// real witness is attached (`seal_positions`) when it has.
    fn admit(&mut self, prepared: &PreparedWindow, bound: &mut BoundWindow) -> Result<(), String>;

    fn stage(&mut self, gpu: &Gpu, inputs: &WindowInputs, stream: &Stream) -> HipResult<()>;

    fn await_staged(&mut self, gpu: &Gpu) -> HipResult<()>;

    /// `HIPFIRE_DFLASH_RAILGUN_DEBUG`: one stderr line describing the bound
    /// window, before submit. No-op unless the builder was asked to trace.
    fn trace_window(&mut self, _prepared: &PreparedWindow, _shape: &WindowShape) {}

    /// `HIPFIRE_DFLASH_RAILGUN_TWIN`: execute the bound window's node images
    /// through ordinary HIP launches instead of the PM4 lowering, on the owner
    /// stream, and prove the stream idle. `Ok(false)` when twin mode is off
    /// (the queue runs the window). An `Err` leaves the window partly executed.
    fn twin_run(&mut self, _gpu: &mut Gpu, _prepared: &PreparedWindow) -> HipResult<bool> {
        Ok(false)
    }

    /// §A9.4: for a plan with conditional (loaded-index) nodes, host-check the
    /// window's physical positions against the existing producer, upload
    /// exactly those bytes into the dedicated sealed positions allocation on
    /// the owner stream, wait for that H2D (the stream is idle for it on
    /// return, success or error), and attach the resulting bind witness to
    /// `bound`. A plan without conditional nodes is a no-op. Runs after
    /// `await_staged` and before `submit`.
    fn seal_positions(
        &mut self,
        request: &mut WindowRequest<'_>,
        prepared: &PreparedWindow,
        bound: &mut BoundWindow,
        inputs: &WindowInputs,
    ) -> HipResult<()>;

    /// The window was refused after `admit` and before anything was queued
    /// (the stream is proven idle): release the sealed positions, forget a
    /// pending certification and retire the staging upload (idempotent).
    fn abandon_positions(&mut self);

    fn read_results(&mut self, gpu: &Gpu, execution: WindowExecution) -> HipResult<WindowResults>;

    /// Drop builder-held plans/bindings (reset, eviction, capacity change).
    fn invalidate(&mut self);

    /// Free the builder's device resources at speculator teardown (the owner
    /// has already dropped every prepared window).
    fn release(self: Box<Self>, gpu: &mut Gpu);
}

/// Outcome of a window that ran on railgun.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RailgunWindowRun {
    pub execution: WindowExecution,
    pub block: Vec<u32>,
    /// `Whole16` only.
    pub verify: Option<RailgunVerify>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RailgunVerify {
    pub argmax_per_pos: Vec<u32>,
    /// `WindowVerifyReceipt::same_as_hip_graph`.
    pub same_as_hip_graph: bool,
    pub tape_populated: bool,
}

/// Either the window ran, or it did not start and the HIP route owns it
/// (nothing changed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RailgunAttempt {
    Ran(RailgunWindowRun),
    Refused(RailgunRefusal),
}

fn err(msg: impl AsRef<str>) -> HipError {
    HipError::new(0, msg.as_ref())
}

// ----------------------------------------------------------------- owner

/// The per-speculator owner of the certified window route.
pub struct RailgunWindowOwner {
    builder: Option<Box<dyn WindowBuilder>>,
    cache: WindowCache,
    counters: RailgunCounters,
    logged: HashSet<RailgunRefusal>,
    static_verdict: Option<Result<(), RailgunRefusal>>,
    /// Families whose window already ran once on HIP (plans are built only
    /// off such a window).
    warm: Vec<DraftFamilyKey>,
    generation: u64,
    quarantined: Option<String>,
    last_route: Option<&'static str>,
}

/// Borrowed pieces of the live window, owned by `spec_step_dflash`.
pub struct WindowOwners<'a> {
    pub gpu: &'a mut Gpu,
    pub target: &'a mut ModelSlot,
    pub draft_weights: &'a DflashWeights,
    pub draft_cfg: &'a DflashConfig,
    pub draft_scratch: &'a mut DflashScratch,
    pub hidden_rb: &'a mut HiddenStateRingBuffer,
    pub verify_scratch: &'a VerifyScratch,
    pub gdn_tape: Option<&'a mut GdnTape>,
    pub target_snap: &'a mut DeltaNetSnapshot,
    pub position: usize,
    pub seed: u32,
}

impl RailgunWindowOwner {
    /// A new owner; the variable was already read (once) by the caller.
    pub fn new() -> Self {
        Self {
            builder: None,
            cache: WindowCache::new(),
            counters: RailgunCounters::default(),
            logged: HashSet::new(),
            static_verdict: None,
            warm: Vec::new(),
            generation: 0,
            quarantined: None,
            last_route: None,
        }
    }

    /// Install the GPU-facing builder (the integration engine).
    pub fn install_builder(&mut self, builder: Box<dyn WindowBuilder>) {
        self.builder = Some(builder);
    }

    pub fn counters(&self) -> &RailgunCounters {
        &self.counters
    }

    /// Route label of the last window: `whole16`, `hostsplice` or
    /// `hip:<reason>` (cycle-trace tag).
    pub fn last_route(&self) -> Option<&'static str> {
        self.last_route
    }

    /// Start of a speculator step: the route label belongs to one window.
    pub fn clear_route(&mut self) {
        self.last_route = None;
    }

    pub fn is_quarantined(&self) -> bool {
        self.quarantined.is_some()
    }

    /// Drop every prepared window and warm mark (reset, eviction, capacity
    /// or mapping change). In-flight entries cannot exist between windows.
    pub fn invalidate(&mut self) {
        self.cache = WindowCache::new();
        self.warm.clear();
        self.static_verdict = None;
        if let Some(b) = self.builder.as_mut() {
            b.invalidate();
        }
    }

    /// Speculator teardown: drop every prepared window (nothing is in flight
    /// between windows), then free the builder's device resources.
    pub fn release(self, gpu: &mut Gpu) {
        let RailgunWindowOwner { builder, cache, .. } = self;
        drop(cache);
        if let Some(b) = builder {
            b.release(gpu);
        }
    }

    /// Count (and log once per reason) a window that runs on HIP.
    pub fn refuse(&mut self, why: RailgunRefusal, detail: &str) {
        *self.counters.hip_refused.entry(why.key()).or_insert(0) += 1;
        self.last_route = Some(hip_label(why));
        if self.logged.insert(why) {
            if detail.is_empty() {
                eprintln!("[dflash] railgun window: HIP route ({})", why.key());
            } else {
                eprintln!(
                    "[dflash] railgun window: HIP route ({}): {detail}",
                    why.key()
                );
            }
        }
    }

    /// Cached owner-level verdict (recomputed after [`Self::invalidate`]).
    pub fn static_verdict(
        &mut self,
        facts: impl FnOnce() -> StaticFacts,
    ) -> Result<(), RailgunRefusal> {
        if self.static_verdict.is_none() {
            self.static_verdict = Some(admit_static(&facts()));
        }
        self.static_verdict.expect("just set")
    }

    /// Shape the window at `position` would have, from the live owners.
    pub fn shape_hint(
        position: usize,
        draft_scratch: &DflashScratch,
        hidden_rb: &HiddenStateRingBuffer,
        compact_offset: usize,
        physical_cap: usize,
    ) -> Option<WindowShape> {
        let DraftCtxMode::Windowed { w, w_full } = draft_scratch.ctx_mode else {
            return None;
        };
        let thlog = &draft_scratch.thlog;
        Some(WindowShape {
            position: position as u64,
            logical_ctx_rows: thlog.abs_positions().len().min(position) as u64,
            projected_rows: thlog.proj_cached_rows() as u64,
            full_cached_rows: thlog.full_cached_rows() as u64,
            compact_offset: compact_offset as u64,
            ring_head: hidden_rb.head as u64,
            swa_w: w as u64,
            full_w: w_full as u64,
            max_position: physical_cap.saturating_sub(1) as u64,
            max_tiles: 0,
            verify_rows: WINDOW_ROWS as u64,
            draft_rows: WINDOW_ROWS as u64,
            generation: 0,
        })
    }

    /// A window of `shape`'s family completed on HIP: later windows of the
    /// family may be planned.
    pub fn note_hip_window(&mut self, shape: &WindowShape) {
        if let Ok(key) = draft_family_key(shape) {
            if !self.warm.contains(&key) {
                self.warm.push(key);
            }
        }
    }

    /// The §6 transaction for one admitted window.
    ///
    /// Before the queue is touched (plan, bind, input staging, snapshot
    /// SAVE) every failure returns `Refused` with the GDN frame counter
    /// restored: the HIP route then runs the window unchanged (the SAVE it
    /// repeats is idempotent). After submit nothing is restored or freed
    /// while PM4 may run: unknown quiescence quarantines the owner; proven
    /// quiescence propagates the error (no automatic retry).
    pub fn run(
        &mut self,
        owners: WindowOwners<'_>,
        execution: WindowExecution,
    ) -> HipResult<RailgunAttempt> {
        if let Some(why) = &self.quarantined {
            return Err(err(format!("railgun window owner is quarantined: {why}")));
        }
        let WindowOwners {
            gpu,
            target,
            draft_weights,
            draft_cfg,
            draft_scratch,
            hidden_rb,
            verify_scratch,
            gdn_tape,
            target_snap,
            position,
            seed,
        } = owners;
        if self.builder.is_none() {
            return Ok(self.refused(
                RailgunRefusal::EngineUnavailable,
                "no window builder installed",
            ));
        }
        let Some(mut shape) = Self::shape_hint(
            position,
            draft_scratch,
            hidden_rb,
            target.kv_cache.compact_offset,
            target.kv_cache.physical_cap,
        ) else {
            return Ok(self.refused(RailgunRefusal::DraftNotAllSliding, "legacy context mode"));
        };
        // Family 4 (cold seed / reset / eviction / rebuild) and unseen
        // families keep the HIP route; a window that ran once there warms
        // the family for the next one.
        match draft_family_key(&shape) {
            Err(e) => return Ok(self.refused(RailgunRefusal::ColdSeedOrRebuild, &e)),
            Ok(key) if !self.warm.contains(&key) => {
                return Ok(self.refused(RailgunRefusal::ShapeNotWarm, ""));
            }
            Ok(_) => {}
        }
        self.generation += 1;
        shape.generation = self.generation;
        if gpu.active_stream.is_none() {
            gpu.active_stream = Some(gpu.hip.stream_create()?);
        }

        let frame_checkpoint = rdna_compute::norm::gdn_requant_frame_checkpoint();
        let tape_expected = gdn_tape.is_some();
        let Self {
            builder,
            cache,
            counters,
            logged,
            quarantined,
            last_route,
            ..
        } = self;
        let builder = builder.as_mut().expect("checked above");

        let mut request = WindowRequest {
            gpu,
            target,
            draft_weights,
            draft_cfg,
            draft_scratch,
            hidden_rb,
            verify_scratch,
            gdn_tape,
            shape,
            seed,
            mask_token: draft_cfg.mask_token_id,
            execution,
        };
        let mut pre_queue_refusal =
            |why: RailgunRefusal, detail: String| -> HipResult<RailgunAttempt> {
                rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
                *counters.hip_refused.entry(why.key()).or_insert(0) += 1;
                *last_route = Some(hip_label(why));
                if logged.insert(why) {
                    eprintln!(
                        "[dflash] railgun window: HIP route ({}): {detail}",
                        why.key()
                    );
                }
                Ok(RailgunAttempt::Refused(why))
            };

        // 1. Plan + bind (host only). `prepare` builds a missing skeleton in
        //    the owner's 32-entry cache off a family that already ran on HIP.
        let (prepared, prep) = match builder.prepare(cache, &mut request) {
            Ok(v) => v,
            Err(WindowRefusal::Malformed(m)) if m == SKELETON_BUILDING => {
                return pre_queue_refusal(RailgunRefusal::SkeletonBuilding, m)
            }
            Err(e) => return pre_queue_refusal(RailgunRefusal::PrepareRefused, e.to_string()),
        };
        if prepared.plan().execution() != execution {
            return pre_queue_refusal(
                RailgunRefusal::PrepareRefused,
                format!(
                    "prepared {:?} for a {execution:?} window",
                    prepared.plan().execution()
                ),
            );
        }
        if execution == WindowExecution::Whole16
            && prepared.plan().tape_populated() != tape_expected
        {
            return pre_queue_refusal(
                RailgunRefusal::PrepareRefused,
                "plan tape flag differs from the HIP route's tape decision".into(),
            );
        }
        let mut bound = match prepared.bind(&prep.shape, &prep.bindings, &prep.inputs) {
            Ok(b) => b,
            Err(e) => return pre_queue_refusal(RailgunRefusal::BindRefused, e.to_string()),
        };

        // 1b. Manifest admission, host-only, BEFORE anything is mutated or
        //     enqueued (snapshot SAVE, staging, sealed positions): a refusal
        //     leaves the HIP fallback an untouched preimage.
        if let Err(e) = builder.admit(&*prepared, &mut bound) {
            return pre_queue_refusal(RailgunRefusal::ManifestRefused, e);
        }

        // 2. SAVE + input staging on the owner stream, one fence, then the
        //    entry acquire happens inside the lowered segment. The SAVE is
        //    moved before the draft only for `Whole16` (draft mutates no
        //    target KV/DN state).
        let staged: HipResult<()> = (|| {
            let gpu: &Gpu = &*request.gpu;
            let stream = gpu
                .active_stream
                .as_ref()
                .ok_or_else(|| err("owner stream missing"))?;
            if execution == WindowExecution::Whole16 {
                target_snap.save_from_async_on(&request.target.dn_state, gpu, stream)?;
            }
            builder.stage(gpu, &prep.inputs, stream)?;
            builder.await_staged(gpu)
        })();
        let staged = staged.and_then(|()| {
            builder.seal_positions(&mut request, &*prepared, &mut bound, &prep.inputs)
        });
        if let Err(e) = staged {
            // Roll back everything the failed attempt touched, so the HIP
            // fallback starts from the untouched preimage:
            //  * the stream is proven idle FIRST (the async SAVE, the staging
            //    H2D and the positions upload may still be in flight);
            //  * sealed positions retire (`Retired`), a pending certification
            //    and the staging upload are dropped (`abandon_positions`);
            //  * the GDN frame counter is restored (`pre_queue_refusal`);
            //  * the snapshot needs no undo: the HIP route repeats its
            //    synchronous `save_from`, which overwrites the async SAVE with
            //    the same S(p) bytes (a Whole16 window skips it only on success);
            //  * host deltas are only published after completion, and the
            //    ring/watermarks were never touched.
            let idle = request
                .gpu
                .active_stream
                .as_ref()
                .map(|s| request.gpu.hip.stream_synchronize(s))
                .unwrap_or(Ok(()));
            return match idle {
                Ok(()) => {
                    builder.abandon_positions();
                    pre_queue_refusal(RailgunRefusal::StageFailed, e.to_string())
                }
                Err(sync) => {
                    *quarantined = Some(format!("stage failed ({e}); stream not idle ({sync})"));
                    Err(err(format!("railgun window: stage failed ({e}) and the stream did not drain ({sync}); owner quarantined")))
                }
            };
        }

        // 2b. Developer arms: trace the bound window; in twin mode run its
        //     node images through HIP instead of the PM4 queue (whose queue is
        //     then a no-op, so submit/complete only drive the state machine).
        builder.trace_window(&*prepared, &prep.shape);
        if let Err(e) = builder.twin_run(&mut *request.gpu, &*prepared) {
            *quarantined = Some(format!("twin run failed: {e}"));
            return Err(err(format!(
                "railgun window: twin run failed ({e}); owner quarantined"
            )));
        }

        // 3. Submit.
        let submission = match prepared.submit(bound) {
            Ok(s) => s,
            // The pre-submit self-check of the lowered state failed: nothing
            // was dispatched and the queue is quiescent, so this is a refusal
            // to the HIP route, not an execution failure.
            Err(WindowRefusal::SubmitFailed {
                message,
                quiescence: Quiescence::Proven,
            }) if message.starts_with(rdna_compute::railgun_window::LOWERING_SELF_CHECK) => {
                builder.abandon_positions();
                // Ask the HIP runtime whether it tracks an allocation over the
                // mismatching kernarg buffers (VA aliasing) and say so.
                if let Some(probe) = request.gpu.railgun_probe_self_check_failure() {
                    eprintln!("[dflash-railgun]   HIP allocation probe: {probe}");
                }
                return pre_queue_refusal(RailgunRefusal::LoweringSelfCheck, message);
            }
            Err(WindowRefusal::SubmitFailed {
                message,
                quiescence,
            }) => {
                return Err(after_submit_failure(
                    quarantined,
                    quiescence,
                    format!("submit: {message}"),
                ));
            }
            Err(other) => {
                builder.abandon_positions();
                return pre_queue_refusal(RailgunRefusal::SubmitRefused, other.to_string());
            }
        };

        // 4. Complete (queue quiescence proven), then publish the host delta
        //    once and read the proposals / verify picks.
        let complete = match submission.complete() {
            Ok(c) => c,
            Err(f) => {
                return Err(after_submit_failure(
                    quarantined,
                    f.quiescence,
                    f.to_string(),
                ));
            }
        };
        publish_host_delta(
            &complete.host_delta,
            request.draft_scratch,
            request.hidden_rb,
            execution,
        )?;
        let results = builder.read_results(&*request.gpu, execution)?;
        if results.block.len() != WINDOW_ROWS || results.block[0] != seed {
            return Err(err(format!(
                "railgun window: proposal block of {} words (seed {} vs {})",
                results.block.len(),
                results.block.first().copied().unwrap_or(u32::MAX),
                seed
            )));
        }
        let verify = match (execution, results.argmax_per_pos) {
            (WindowExecution::Whole16, Some(picks)) if picks.len() == WINDOW_ROWS => {
                if complete.receipt.tape_populated && !tape_expected {
                    return Err(err(
                        "railgun window: plan populated a GDN tape the owner does not hold",
                    ));
                }
                Some(RailgunVerify {
                    argmax_per_pos: picks,
                    same_as_hip_graph: complete.receipt.same_as_hip_graph,
                    tape_populated: complete.receipt.tape_populated,
                })
            }
            (WindowExecution::HostSplice, None) => None,
            (_, picks) => {
                return Err(err(format!(
                    "railgun window: {execution:?} returned {} verify picks",
                    picks.map_or(0, |p| p.len())
                )));
            }
        };
        match execution {
            WindowExecution::Whole16 => {
                counters.whole16 += 1;
                *last_route = Some("whole16");
            }
            WindowExecution::HostSplice => {
                counters.host_splice += 1;
                *last_route = Some("hostsplice");
            }
        }
        Ok(RailgunAttempt::Ran(RailgunWindowRun {
            execution,
            block: results.block,
            verify,
        }))
    }

    fn refused(&mut self, why: RailgunRefusal, detail: &str) -> RailgunAttempt {
        self.refuse(why, detail);
        RailgunAttempt::Refused(why)
    }
}

impl Default for RailgunWindowOwner {
    fn default() -> Self {
        Self::new()
    }
}

/// `hip:<reason>` label (static: counters/trace keep no allocation).
fn hip_label(why: RailgunRefusal) -> &'static str {
    match why {
        RailgunRefusal::NotGfx1201 => "hip:not_gfx1201",
        RailgunRefusal::NotDenseQwen35 => "hip:not_dense_qwen35",
        RailgunRefusal::Not27bClass => "hip:not_27b_class",
        RailgunRefusal::KvNotFp8 => "hip:kv_not_fp8",
        RailgunRefusal::DraftNotFixed16 => "hip:draft_not_fixed16",
        RailgunRefusal::DraftNotAllSliding => "hip:draft_not_all_sliding",
        RailgunRefusal::TreeVerify => "hip:tree_verify",
        RailgunRefusal::RetainedPm4 => "hip:retained_pm4",
        RailgunRefusal::OnlineTune => "hip:online_tune",
        RailgunRefusal::Sampling => "hip:sampling",
        RailgunRefusal::Cactus => "hip:cactus",
        RailgunRefusal::Pld => "hip:pld",
        RailgunRefusal::NgramOverride => "hip:ngram_override",
        RailgunRefusal::RepeatPenalty => "hip:repeat_penalty",
        RailgunRefusal::HostPath => "hip:host_path",
        RailgunRefusal::Diagnostics => "hip:diagnostics",
        RailgunRefusal::CtxSlice => "hip:ctx_slice",
        RailgunRefusal::NoGdnTape => "hip:no_gdn_tape",
        RailgunRefusal::BlockNot16 => "hip:block_not_16",
        RailgunRefusal::LookupCopy => "hip:lookup_copy",
        RailgunRefusal::LookupProbe => "hip:lookup_probe",
        RailgunRefusal::ColdSeedOrRebuild => "hip:cold_seed_or_rebuild",
        RailgunRefusal::ShapeNotWarm => "hip:shape_not_warm",
        RailgunRefusal::GdnFrameDependent => "hip:gdn_frame_dependent",
        RailgunRefusal::ManifestRefused => "hip:manifest_refused",
        RailgunRefusal::LoweringSelfCheck => "hip:lowering_self_check",
        RailgunRefusal::SkeletonBuilding => "hip:skeleton_building",
        RailgunRefusal::EngineUnavailable => "hip:engine_unavailable",
        RailgunRefusal::PrepareRefused => "hip:prepare_refused",
        RailgunRefusal::BindRefused => "hip:bind_refused",
        RailgunRefusal::StageFailed => "hip:stage_failed",
        RailgunRefusal::SubmitRefused => "hip:submit_refused",
    }
}

/// Failure after the queue was (or may have been) touched: never restore or
/// free. Unknown quiescence quarantines the owner.
fn after_submit_failure(
    quarantined: &mut Option<String>,
    quiescence: Quiescence,
    message: String,
) -> HipError {
    match quiescence {
        Quiescence::Unknown => {
            eprintln!("[dflash] railgun window owner quarantined: {message}");
            *quarantined = Some(message.clone());
            err(format!(
                "railgun window failed with unknown queue quiescence (owner quarantined): {message}"
            ))
        }
        Quiescence::Proven => err(format!(
            "railgun window failed with the queue proven quiescent (no automatic retry): {message}"
        )),
    }
}

/// Publish the bound [`WindowHostDelta`] once, after completion and before
/// the accepted-prefix scatter (§6): draft projection watermark
/// (`mark_proj_cached`, which also advances the full-cache watermark) and the
/// hidden ring head/written. Each `from` must equal the live value.
fn publish_host_delta(
    delta: &WindowHostDelta,
    draft_scratch: &mut DflashScratch,
    hidden_rb: &mut HiddenStateRingBuffer,
    execution: WindowExecution,
) -> HipResult<()> {
    if delta.upload_watermark.is_some() {
        return Err(err(
            "railgun window: unexpected upload watermark (the GPU-resident route never uploads target_hidden)",
        ));
    }
    if let Some(t) = delta.projection_watermark {
        let live = draft_scratch.thlog.proj_cached_rows() as u64;
        if live != t.from {
            return Err(err(format!(
                "railgun window: stale projection watermark (live {live}, planned from {})",
                t.from
            )));
        }
        draft_scratch.thlog.mark_proj_cached(t.to as usize);
    }
    match (execution, delta.ring_head, delta.ring_written) {
        (WindowExecution::Whole16, Some(head), Some(written)) => {
            if head.from != hidden_rb.head as u64 || written.from != hidden_rb.written as u64 {
                return Err(err("railgun window: stale hidden ring delta"));
            }
            hidden_rb.apply_commit_delta(HiddenCommitDelta {
                head_from: head.from as usize,
                head_to: head.to as usize,
                written_from: written.from as usize,
                written_to: written.to as usize,
            });
            Ok(())
        }
        (WindowExecution::HostSplice, None, None) => Ok(()),
        _ => Err(err(format!(
            "railgun window: {execution:?} ring delta does not match its segment"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> StaticFacts {
        StaticFacts {
            arch: "gfx1201".into(),
            num_experts: 0,
            dim: 5120,
            ffn_dim: 17408,
            n_heads: 24,
            n_kv_heads: 4,
            kv_fp8: true,
            kv_other_quant: false,
            draft_block: 16,
            ctx_mode: DraftCtxMode::Windowed {
                w: 2048,
                w_full: 2048,
            },
            retained_pm4_admitted: false,
            online_tuner: false,
        }
    }

    fn window() -> WindowFacts {
        WindowFacts {
            b: 16,
            greedy: true,
            cactus: false,
            pld: false,
            ngram_cache: false,
            repeat_penalty: false,
            host_path: false,
            diagnostics: false,
            ctx_slice: false,
            gdn_tape: true,
            gdn_frame_independent: true,
            lookup: LookupFacts::None,
        }
    }

    #[test]
    fn static_admission_refusals_are_exact() {
        assert_eq!(admit_static(&facts()), Ok(()));
        let cases: Vec<(Box<dyn Fn(&mut StaticFacts)>, RailgunRefusal)> = vec![
            (
                Box::new(|f| f.arch = "gfx1100".into()),
                RailgunRefusal::NotGfx1201,
            ),
            (
                Box::new(|f| f.arch = "gfx1200".into()),
                RailgunRefusal::NotGfx1201,
            ),
            (
                Box::new(|f| f.num_experts = 256),
                RailgunRefusal::NotDenseQwen35,
            ),
            (Box::new(|f| f.dim = 4096), RailgunRefusal::Not27bClass),
            (Box::new(|f| f.n_heads = 32), RailgunRefusal::Not27bClass),
            (Box::new(|f| f.kv_fp8 = false), RailgunRefusal::KvNotFp8),
            (
                Box::new(|f| f.kv_other_quant = true),
                RailgunRefusal::KvNotFp8,
            ),
            (
                Box::new(|f| f.draft_block = 8),
                RailgunRefusal::DraftNotFixed16,
            ),
            (
                Box::new(|f| {
                    f.ctx_mode = DraftCtxMode::Windowed {
                        w: 2048,
                        w_full: 8192,
                    }
                }),
                RailgunRefusal::DraftNotAllSliding,
            ),
            (
                Box::new(|f| f.ctx_mode = DraftCtxMode::Legacy),
                RailgunRefusal::DraftNotAllSliding,
            ),
            (
                Box::new(|f| f.retained_pm4_admitted = true),
                RailgunRefusal::RetainedPm4,
            ),
            (
                Box::new(|f| f.online_tuner = true),
                RailgunRefusal::OnlineTune,
            ),
        ];
        for (mutate, want) in cases {
            let mut f = facts();
            mutate(&mut f);
            assert_eq!(admit_static(&f), Err(want), "{want:?}");
        }
    }

    #[test]
    fn window_admission_picks_whole16_hostsplice_or_refuses() {
        assert_eq!(admit_window(&window()), Ok(WindowExecution::Whole16));
        let mut w = window();
        w.lookup = LookupFacts::TailDeclinesAll;
        assert_eq!(admit_window(&w), Ok(WindowExecution::Whole16));
        w.lookup = LookupFacts::TailMayAct;
        assert_eq!(admit_window(&w), Ok(WindowExecution::HostSplice));
        w.lookup = LookupFacts::Copy;
        assert_eq!(admit_window(&w), Err(RailgunRefusal::LookupCopy));
        w.lookup = LookupFacts::Probe;
        assert_eq!(admit_window(&w), Err(RailgunRefusal::LookupProbe));
        let cases: Vec<(Box<dyn Fn(&mut WindowFacts)>, RailgunRefusal)> = vec![
            (Box::new(|f| f.greedy = false), RailgunRefusal::Sampling),
            (Box::new(|f| f.cactus = true), RailgunRefusal::Cactus),
            (Box::new(|f| f.pld = true), RailgunRefusal::Pld),
            (
                Box::new(|f| f.ngram_cache = true),
                RailgunRefusal::NgramOverride,
            ),
            (
                Box::new(|f| f.repeat_penalty = true),
                RailgunRefusal::RepeatPenalty,
            ),
            (Box::new(|f| f.host_path = true), RailgunRefusal::HostPath),
            (
                Box::new(|f| f.diagnostics = true),
                RailgunRefusal::Diagnostics,
            ),
            (Box::new(|f| f.ctx_slice = true), RailgunRefusal::CtxSlice),
            (Box::new(|f| f.gdn_tape = false), RailgunRefusal::NoGdnTape),
            (Box::new(|f| f.b = 9), RailgunRefusal::BlockNot16),
            (
                Box::new(|f| f.gdn_frame_independent = false),
                RailgunRefusal::GdnFrameDependent,
            ),
        ];
        for (mutate, want) in cases {
            let mut f = window();
            mutate(&mut f);
            assert_eq!(admit_window(&f), Err(want), "{want:?}");
        }
    }

    #[test]
    fn refusals_count_and_log_once_per_reason() {
        let mut owner = RailgunWindowOwner::new();
        owner.refuse(RailgunRefusal::Sampling, "");
        owner.refuse(RailgunRefusal::Sampling, "");
        owner.refuse(RailgunRefusal::BlockNot16, "");
        assert_eq!(owner.counters().hip_refused["sampling"], 2);
        assert_eq!(owner.counters().hip_refused["block_not_16"], 1);
        assert_eq!(owner.counters().hip_total(), 3);
        assert_eq!(owner.last_route(), Some("hip:block_not_16"));
        assert_eq!(owner.counters().whole16, 0);
        // The manifest refusal is its own counted key and trace label.
        owner.refuse(RailgunRefusal::ManifestRefused, "");
        assert_eq!(owner.counters().hip_refused["manifest_refused"], 1);
        assert_eq!(owner.last_route(), Some("hip:manifest_refused"));
        owner.refuse(RailgunRefusal::LoweringSelfCheck, "");
        assert_eq!(owner.counters().hip_refused["lowering_self_check"], 1);
        assert_eq!(owner.last_route(), Some("hip:lowering_self_check"));
        owner.refuse(RailgunRefusal::SkeletonBuilding, "");
        assert_eq!(owner.counters().hip_refused["skeleton_building"], 1);
        assert_eq!(owner.last_route(), Some("hip:skeleton_building"));
    }

    #[test]
    fn warm_marks_drop_on_invalidate() {
        let mut owner = RailgunWindowOwner::new();
        let shape = WindowShape {
            position: 4000,
            logical_ctx_rows: 4000,
            projected_rows: 3990,
            swa_w: 2048,
            full_w: 2048,
            ..WindowShape::default()
        };
        owner.note_hip_window(&shape);
        assert_eq!(owner.warm.len(), 1);
        owner.invalidate();
        assert!(owner.warm.is_empty());
    }
}
