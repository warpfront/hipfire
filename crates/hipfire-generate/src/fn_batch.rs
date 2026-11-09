// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next exact lanes (greedy, or sampled with host draws): AR lanes, and per-lane native MTP lanes when the
//! staged store carries the MTP head (`serve.batch_spec`, default on;
//! `serve.batch_spec = false` / `HIPFIRE_SERVE_BATCH_SPEC=0` stages AR lanes
//! only). Prompt IDs come from `qwen::render_qwen4_prompt`,
//! also used by `generate_qwen4_ar` and the native MTP singleton. Emission
//! reuses `generate_ar_with_forward`'s `QwenArSemanticProducer`, incremental
//! tokenizer decode, raw commit, committed-event, think-budget and done helpers.
//! Unlike the Qwen35 driver there is no loop guard or tool-markup parser when
//! tools are absent. A prompt's seed is held until its AR forward completes:
//! the singleton forwards the token BEFORE committing/classifying it. The
//! executor's AR pick is the following token, not the token just forwarded.
//! Each step the driver plans which decoding lanes run (one per row of the
//! step's row budget), asks the executor for a step-synchronous batched draft
//! of them (`draft_step`, budget `min(K, tokens left - 1, context left - 1)`,
//! zero while think-budget tokens are forced), then plans one row (`Ar`: the
//! pending seed) or `1 + d` rows (`Verify`: `[seed, drafts..]`) per lane. A
//! lane's advance commits `committed_ids`; the tokens it FORWARDED are the plan
//! seed followed by `committed_ids` without its last entry (the last is the
//! next seed), emitted in order through `emit_forwarded`. The first finish
//! (stop, terminator, think budget, length) ends the lane and drops the rest
//! of the window; `length` from the executor applies to the last forwarded
//! token only. A forced think-budget close queued mid-window follows the
//! window's own tokens. A draft-step failure errors every lane it named. The
//! done value carries the lane's MTP counters (`mtp`, `mtp_windows`,
//! `ar_windows`, `mtp_retired`, `mtp_drafts`, `mtp_accepted`).
//! A running singleton is promoted into a lane when a second request arrives
//! (`generate_spec` hands a `Qwen4PromotedRequest`, adopted before the loop
//! under the same AttemptKey; its already-streamed pending seed is not
//! emitted again). No lane session cache or parking is performed here.
//! Timing fields measure this execution and are not byte-identical wall-clock
//! values. `cached_tokens` is zero: S3 uses cold lane state, not singleton
//! session restore. Content/reasoning bytes and semantic finish classification
//! use the singleton code; compute-ID identity is the lane executor's oracle.
//! One scheduler loop serves every topology through `Owner`/`LaneExec`: single
//! and expert-parallel models run the bundle's executor on the rank-0 device;
//! a pipeline-parallel model (`qwen4_mesh_mut`) admits, releases, force-seeds
//! and steps through the `pp_*` lane APIs and `Qwen4PpLaneExecutor` over the
//! mesh's `Gpus`. Promotion of a running singleton into a lane is single-only
//! (PP and EP refuse it), so the promoted-adoption path only runs on single.
//! Because an EP/PP singleton could never join its peers, a lone EP/PP request
//! is not run as a singleton at all: the daemon admits it straight into this
//! driver (`qwen4_lone_request_takes_lane`), where it is a cohort of one and
//! every later arrival joins at the next step (`lone_arrival_hands_back` keeps
//! the singleton only for a request whose session cache would be restored).
//! Sampled requests (neutral penalties; the lane store's `serve.batch_sampled`)
//! install a `LaneSampling` right after admission: the lane draws on the host
//! with the AR singleton's sampler and the request seed, and MTP lanes verify
//! sampled windows naively. Promoted lanes are greedy-only.
//! Single stores run every decode lane through the mesh
//! lane protocol's one-rank reducer (`hipfire_arch_qwen4::lane_rank`): the
//! driver's four entries are the protocol's phases (`draft_step`: plan, begin,
//! draft rounds, seal; `forward_step`: verify, decide, prepare; `commit_step`:
//! publish and settle the step's price), and the lane table's committed IDs
//! and pending seed are the mirror of the root's published step (the driver
//! plans rows from it and emits `forwarded_tokens` from the advance the
//! root's output produced). The driver keeps request termination (EOS, stop,
//! length, think budget, cancel): the root never closes a lane.
//! An expert-parallel store staged with native-MTP lanes
//! (`Qwen4Bundle::ep_mesh_lanes_armed`) splits each tick: the executor plan
//! carries prefill tiles only, and every chosen decoding lane runs as one
//! root-planned mesh window (`ep_mesh_decode_step`: rank 0 routes, drafts and
//! decides, every rank replays the ordered messages) whose advance feeds the
//! same emission path (`forwarded_tokens` from the window's seed). The two
//! halves fail independently: an error in one errors only its own lanes and
//! the other half's committed tokens are still emitted. The executor half's
//! prefill ticks (first tokens) are emitted between the halves, before the
//! mesh window (`HIPFIRE_LANE_FIRST_TOKEN_BEFORE_MESH`; `emit_tick` is the one
//! per-tick emission both positions use). Route mix by step
//! concurrency is logged once per drive (`[fn-lane-mesh-mix]`). An AR EP store
//! (kill switch `HIPFIRE_EP_MTP_LANES=0`) keeps the executor
//! path unchanged.

use crate::ar::*;
use crate::batch::{
    drain_qwen_batch_inbox, handoff_started_in_think, is_vmm_batch_request_eligible,
    qwen4_lone_request_takes_lane, Qwen4PromotedRequest,
};
use crate::common::{emit_committed_event, fit_max_tokens_if};
use hipfire_arch_qwen4::bundle::{Qwen4Bundle, Qwen4PpLaneExecutor};
use hipfire_arch_qwen4::ep_mesh_serve::{MeshDecodeRequest, MeshStepFacts};
use hipfire_arch_qwen4::lane::pages::{
    long_prefill_route_rows, page_budget_enabled, reservation_rows, PageDecision, PageVerdict,
    LANE_PAGE_BUDGET_REFUSED,
};
use hipfire_arch_qwen4::lane::{LaneSampling, LaneSpecStats, LaneStatus};
use hipfire_arch_qwen4::lane_exec::{
    LaneDraft, LaneRoute, Qwen4LaneExecutor, PROVISION_MAP_REFUSED,
};
use hipfire_engine::emit::*;
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_engine::turn_cache::*;
use hipfire_loader::LoadedModel;
use hipfire_runtime::emit_text::ThinkBudget;
use hipfire_runtime::external_rows::ServingGuard;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::session_cache::SessionRoute;
use hipfire_runtime::slot_batch::{
    BatchStepPlan, RequestAdvance, RequestEpoch, RequestRows, RequestStepKind, RowRange, StepOutput,
};
use rdna_compute::Gpu;
use std::collections::{HashSet, VecDeque};
use std::io::{Stdout, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const ROUTE: GenerationRoute = GenerationRoute::Qwen4Ar;

/// Whether a request just assigned a lane, with no other lane active and none
/// queued (`lone`), goes back to the singleton route instead of running as a
/// lane.
/// - Single device (`lone_lane` false): a request that arrived after the
///   driver started (`initial` false) hands back, as before; the driver's own
///   entry cohort never does, because the singleton can be promoted into a lane
///   when a peer arrives.
/// - EP/PP (`lone_lane`): the singleton can NOT be promoted, so a lone request
///   stays a lane (a cohort of one; peers join at the next step) — except when
///   the singleton route would restore session state (`session_reuse`), which
///   lanes (cold state, no session cache) cannot.
pub(crate) fn lone_arrival_hands_back(
    lone: bool,
    lone_lane: bool,
    initial: bool,
    session_reuse: impl FnOnce() -> bool,
) -> bool {
    lone && if lone_lane { session_reuse() } else { !initial }
}

static NEXT_EPOCH: AtomicU64 = AtomicU64::new(1);

/// The next request epoch of a lane owner: shared by admission and the
/// singleton promotion, so tags never collide.
pub(crate) fn next_lane_epoch() -> Result<RequestEpoch, String> {
    let tag = NEXT_EPOCH
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| "qwen4 lane epoch exhausted".to_string())?;
    Ok(RequestEpoch {
        request_tag: tag,
        owner_generation: 1,
    })
}

struct LaneCtl {
    epoch: RequestEpoch,
    slot: usize,
    semantic: Option<QwenArSemanticProducer>,
    bytes: Vec<u8>,
    budget: ThinkBudget,
    forced: VecDeque<u32>,
    started_in_think: bool,
    max_tokens: usize,
    start: Instant,
    decode_start: Option<Instant>,
    prefill_ms: f64,
    cache: Option<(TurnCacheAction, Vec<u32>)>,
    /// A promoted lane's pending seed was already streamed by the singleton:
    /// its first decode advance does not emit the plan seed again.
    skip_first_forward: bool,
    /// The lane's decode map-ahead ran (once per request, at its first
    /// planned decode step).
    mapped_ahead: bool,
    /// Whole-request PLE serving guard: owned by the lane's control for its
    /// full lifespan (admission or promotion until commit, cancel or error
    /// drops the control), so the PLE prewarm readers stay paused for the
    /// lane's prefill and decode, never just for a scheduler tick.
    _serving: Option<ServingGuard>,
}

/// Whether the lanes live on a pipeline-parallel mesh (band-scoped stores on
/// every band, driven through the mesh's `Gpus`) rather than on the single
/// or expert-parallel bundle (rank-0 `Gpu`).
fn pp_active(model: &LoadedModel) -> Result<bool, String> {
    Ok(model
        .qwen4()
        .ok_or("qwen4 bundle disappeared")?
        .pp_mesh_active())
}

/// Free `epoch`'s lane on every owner of the topology (host-only).
fn release_epoch(model: &mut LoadedModel, epoch: &RequestEpoch) -> Result<(), String> {
    if pp_active(model)? {
        let (bundle, gpus) = model
            .qwen4_mesh_mut()
            .ok_or("qwen4 topology.pp: the pipeline devices are not attached")?;
        bundle
            .pp_release_lane(gpus, epoch)
            .map_err(|e| e.to_string())
    } else {
        model
            .qwen4_mut()
            .ok_or("qwen4 bundle disappeared")?
            .release_lane(epoch)
            .map_err(|e| e.to_string())
    }
}

fn release(model: &mut LoadedModel, ctl: &mut LaneCtl) -> Result<(), String> {
    if !ctl.epoch.is_admitted() {
        return Ok(());
    }
    release_epoch(model, &ctl.epoch)?;
    ctl.epoch.owner_generation = 0;
    Ok(())
}

/// The lane owner of the loaded topology: the Qwen4 bundle with the device
/// handles its lane operations need. Single and expert-parallel models use
/// the rank-0 `gpu`; a pipeline-parallel model uses the mesh's `Gpus` (the
/// output-band root bundle owns the mesh and its authoritative lane table).
struct Owner<'a> {
    bundle: &'a mut Qwen4Bundle,
    gpu: &'a mut Gpu,
    gpus: Option<&'a mut Gpus>,
}

impl<'a> Owner<'a> {
    fn of(model: &'a mut LoadedModel, gpu: &'a mut Gpu) -> Result<Self, String> {
        if pp_active(model)? {
            let (bundle, gpus) = model
                .qwen4_mesh_mut()
                .ok_or("qwen4 topology.pp: the pipeline devices are not attached")?;
            Ok(Self {
                bundle,
                gpu,
                gpus: Some(gpus),
            })
        } else {
            Ok(Self {
                bundle: model.qwen4_mut().ok_or("qwen4 bundle disappeared")?,
                gpu,
                gpus: None,
            })
        }
    }

    /// Admit into `slot` when the page budget chose one (every EP rank takes the root's
    /// choice), else the lowest free lane.
    fn admit(
        &mut self,
        epoch: RequestEpoch,
        prompt: Vec<u32>,
        slot: Option<usize>,
    ) -> Result<usize, String> {
        let admitted = match self.gpus.as_deref_mut() {
            Some(gpus) => self.bundle.pp_admit_lane(gpus, epoch, prompt),
            None => self
                .bundle
                .admit_lane_in(&mut *self.gpu, epoch, prompt, slot),
        };
        admitted.map_err(|e| e.to_string())
    }

    /// The lane page budget's decision for a request reserving `rows` context rows: the
    /// verdict and the slot to admit into (a pipeline-parallel mesh has no per-rank lane
    /// pages: always fits, no slot choice).
    fn page_decision(&mut self, rows: usize) -> Result<PageDecision, String> {
        if self.gpus.is_some() {
            return Ok(PageDecision::UNBUDGETED);
        }
        self.bundle.lane_page_decision(&mut *self.gpu, rows)
    }

    /// The elastic reserve is on for this owner (an EP load with the D2 tail and a positive cap;
    /// never on a pipeline-parallel mesh).
    fn elastic_active(&self) -> bool {
        self.gpus.is_none() && self.bundle.elastic_active()
    }

    /// Reclaim the elastic bank on every rank; whether any rank freed one. `false` without the
    /// reserve.
    fn elastic_reclaim(&mut self, why: &str) -> bool {
        self.elastic_active() && self.bundle.elastic_reclaim_all(&mut *self.gpu, why)
    }

    /// Reserve `rows` context rows for the just-admitted lane of `epoch` (host-only; a pipeline
    /// parallel mesh keeps none).
    fn reserve_pages(&mut self, epoch: &RequestEpoch, rows: usize) -> Result<(), String> {
        if self.gpus.is_some() {
            return Ok(());
        }
        self.bundle
            .reserve_lane_pages(epoch, rows)
            .map_err(|e| e.to_string())
    }

    /// Install the request's sampler on a just-admitted lane (host-only).
    fn set_sampling(&mut self, epoch: &RequestEpoch, sampling: LaneSampling) -> Result<(), String> {
        let set = match self.gpus.as_deref_mut() {
            Some(gpus) => self.bundle.pp_set_lane_sampling(gpus, epoch, sampling),
            None => self.bundle.set_lane_sampling(epoch, sampling),
        };
        set.map_err(|e| e.to_string())
    }

    /// Make a just-admitted lane a `min_tokens` lane (host-only): its windows
    /// run through an EOS. Not available on a pipeline-parallel mesh.
    fn set_ignore_eos(&mut self, epoch: &RequestEpoch) -> Result<(), String> {
        if self.gpus.is_some() {
            return Err("min_tokens is not supported on pipeline-parallel lanes".to_string());
        }
        self.bundle
            .set_lane_ignore_eos(epoch)
            .map_err(|e| e.to_string())
    }

    fn force_seed(&mut self, epoch: &RequestEpoch, token: u32) -> Result<(), String> {
        let forced = match self.gpus.as_deref_mut() {
            Some(gpus) => self.bundle.pp_force_lane_seed(gpus, epoch, token),
            None => self.bundle.force_lane_seed(epoch, token),
        };
        forced.map_err(|e| e.to_string())
    }

    fn executor(&mut self) -> Result<LaneExec<'_>, String> {
        let executor = match self.gpus.as_deref_mut() {
            Some(gpus) => self.bundle.pp_lane_executor(gpus).map(LaneExec::Pp),
            None => self
                .bundle
                .lane_executor()
                .map(|executor| LaneExec::Single(executor, &mut *self.gpu)),
        };
        executor.map_err(|e| e.to_string())
    }
}

/// The exact AR lane executor of the topology behind one call surface.
enum LaneExec<'a> {
    /// Single or expert-parallel (the executor fans out across EP ranks itself).
    Single(Qwen4LaneExecutor<'a>, &'a mut Gpu),
    Pp(Qwen4PpLaneExecutor<'a>),
}

impl LaneExec<'_> {
    fn exact_prefill_chunk_len(&self, epoch: &RequestEpoch) -> Result<usize, String> {
        match self {
            Self::Single(executor, _) => executor.exact_prefill_chunk_len(epoch),
            Self::Pp(executor) => executor.exact_prefill_chunk_len(epoch),
        }
    }

    fn map_ahead_decode(&mut self, epoch: &RequestEpoch, remaining_new_tokens: usize) {
        match self {
            Self::Single(executor, gpu) => {
                executor.map_ahead_decode(gpu, epoch, remaining_new_tokens)
            }
            Self::Pp(executor) => executor.map_ahead_decode(epoch, remaining_new_tokens),
        }
    }

    fn draft_step(&mut self, requests: &[(RequestEpoch, usize)]) -> Result<Vec<LaneDraft>, String> {
        match self {
            Self::Single(executor, gpu) => executor.draft_step(gpu, requests),
            Self::Pp(executor) => executor.draft_step(requests),
        }
    }

    fn provision_step(&mut self, plan: &BatchStepPlan) -> Result<(), String> {
        match self {
            Self::Single(executor, gpu) => executor.provision_step(gpu, plan),
            Self::Pp(executor) => executor.provision_step(plan),
        }
    }

    fn forward_step(&mut self, plan: &BatchStepPlan) -> Result<StepOutput, String> {
        match self {
            Self::Single(executor, gpu) => executor.forward_step(gpu, plan),
            Self::Pp(executor) => executor.forward_step(plan),
        }
    }

    fn commit_step(
        &mut self,
        plan: &BatchStepPlan,
        output: StepOutput,
    ) -> Result<Vec<RequestAdvance>, String> {
        match self {
            Self::Single(executor, gpu) => executor.commit_step(gpu, plan, output),
            Self::Pp(executor) => executor.commit_step(plan, output),
        }
    }

    fn abort_step(&mut self, plan: &BatchStepPlan) {
        match self {
            Self::Single(executor, _) => executor.abort_step(plan),
            Self::Pp(executor) => executor.abort_step(plan),
        }
    }
}

/// The tokens a decode advance forwarded: the plan seed, then every committed
/// id but the last (the next pending seed). `skip_seed` drops the plan seed
/// (a promoted lane's was streamed by the singleton).
fn forwarded_tokens(seed: u32, committed: &[u32], skip_seed: bool) -> Vec<u32> {
    let window = &committed[..committed.len().saturating_sub(1)];
    if skip_seed {
        window.to_vec()
    } else {
        std::iter::once(seed)
            .chain(window.iter().copied())
            .collect()
    }
}

/// `HIPFIRE_LANE_EARLY_FIRST_TOKEN`: unset / anything but exact `0` streams a lane's
/// prompt-completing pick in the step whose prefill committed it; `0` (kill switch) defers it
/// to the lane's first decode advance (as its plan seed, after that step's window).
pub const LANE_EARLY_FIRST_TOKEN_ENV: &str = "HIPFIRE_LANE_EARLY_FIRST_TOKEN";

fn early_first_token_from(value: Option<&str>) -> bool {
    value.map(str::trim) != Some("0")
}

static LANE_EARLY_FIRST_TOKEN: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    early_first_token_from(
        hipfire_config::developer_var(LANE_EARLY_FIRST_TOKEN_ENV)
            .ok()
            .as_deref(),
    )
});

fn early_first_token_enabled() -> bool {
    *LANE_EARLY_FIRST_TOKEN
}

/// `HIPFIRE_LANE_FIRST_TOKEN_BEFORE_MESH`: unset / anything but exact `0` streams the executor
/// half's prefill ticks (the early first tokens, abort / stop / EOS / length finishes and the
/// `prefill_progress` events of the lanes whose prompt chunk this step ran) right after the
/// executor commits and BEFORE the step's EP mesh decode window runs; `0` (kill switch)
/// emits them after that window, with the rest of the tick. Meaningful only with
/// `HIPFIRE_LANE_EARLY_FIRST_TOKEN` on, and only on a step that carries a mesh window.
pub const LANE_FIRST_TOKEN_BEFORE_MESH_ENV: &str = "HIPFIRE_LANE_FIRST_TOKEN_BEFORE_MESH";

/// `early`: the resolved `HIPFIRE_LANE_EARLY_FIRST_TOKEN` (off: nothing to move, today's order).
fn first_token_before_mesh_from(early: bool, value: Option<&str>) -> bool {
    early && value.map(str::trim) != Some("0")
}

static LANE_FIRST_TOKEN_BEFORE_MESH: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    first_token_before_mesh_from(
        early_first_token_enabled(),
        hipfire_config::developer_var(LANE_FIRST_TOKEN_BEFORE_MESH_ENV)
            .ok()
            .as_deref(),
    )
});

fn first_token_before_mesh_enabled() -> bool {
    *LANE_FIRST_TOKEN_BEFORE_MESH
}

/// `HIPFIRE_LANE_ADMIT_COALESCE_MS`: the quiet gap (ms) a lane-admission tick waits for one
/// more arrival before its new prompts prefill (unset: [`ADMIT_COALESCE_DEFAULT_MS`]; `0`
/// is the kill switch; values above [`ADMIT_COALESCE_MAX_MS`] are clamped). The whole wait of
/// one tick is at most [`ADMIT_COALESCE_GAPS`] gaps.
pub const LANE_ADMIT_COALESCE_ENV: &str = "HIPFIRE_LANE_ADMIT_COALESCE_MS";
const ADMIT_COALESCE_DEFAULT_MS: u64 = 2;
const ADMIT_COALESCE_MAX_MS: u64 = 20;
const ADMIT_COALESCE_GAPS: u32 = 5;
/// Queued requests that make a tick a burst on their own; below it only a lone fresh request
/// that stays a lane arms the wait ([`coalesce_arms`], `HIPFIRE_LANE_ADMIT_COALESCE_FIRST`).
const ADMIT_COALESCE_MIN_QUEUED: usize = 2;

fn admit_coalesce_from(value: Option<&str>) -> Option<Duration> {
    let ms = match value.map(str::trim).filter(|v| !v.is_empty()) {
        None => ADMIT_COALESCE_DEFAULT_MS,
        Some(v) => v.parse::<u64>().unwrap_or(ADMIT_COALESCE_DEFAULT_MS),
    };
    (ms > 0).then(|| Duration::from_millis(ms.min(ADMIT_COALESCE_MAX_MS)))
}

static LANE_ADMIT_COALESCE: std::sync::LazyLock<Option<Duration>> =
    std::sync::LazyLock::new(|| {
        admit_coalesce_from(
            hipfire_config::developer_var(LANE_ADMIT_COALESCE_ENV)
                .ok()
                .as_deref(),
        )
    });

fn admit_coalesce_gap() -> Option<Duration> {
    *LANE_ADMIT_COALESCE
}

/// `HIPFIRE_LANE_ADMIT_COALESCE_FIRST`: unset / anything but exact `0` also arms the admission
/// wait on a single queued request whose front can take a free lane now, where a lone request
/// stays a lane (`qwen4_lone_request_takes_lane`: EP / PP lanes): the first arrival waits one
/// gap for a second (a gap with no arrival ends the wait), re-armed per arrival within the
/// tick's whole budget. `0` (kill switch) arms it from [`ADMIT_COALESCE_MIN_QUEUED`] queued
/// requests only, as before. A topology that hands a lone request to the singleton route
/// (single device, PP with a speculator) never waits for it, whatever this says;
/// `HIPFIRE_LANE_ADMIT_COALESCE_MS=0` still turns the whole wait off.
pub const LANE_ADMIT_COALESCE_FIRST_ENV: &str = "HIPFIRE_LANE_ADMIT_COALESCE_FIRST";

fn admit_coalesce_first_from(value: Option<&str>) -> bool {
    value.map(str::trim) != Some("0")
}

static LANE_ADMIT_COALESCE_FIRST: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    admit_coalesce_first_from(
        hipfire_config::developer_var(LANE_ADMIT_COALESCE_FIRST_ENV)
            .ok()
            .as_deref(),
    )
});

fn admit_coalesce_first_enabled() -> bool {
    *LANE_ADMIT_COALESCE_FIRST
}

/// Whether a tick's admission waits for stragglers (the caller also needs the queue front to
/// be assignable and a gap configured): a burst of [`ADMIT_COALESCE_MIN_QUEUED`] or more
/// queued requests always does; a lone one only with `first` on, not already held by the page
/// budget (`held_by_pages`: a wait cannot admit it sooner) and where it stays a lane
/// (`lone_takes_lane`, consulted only then).
fn coalesce_arms(
    first: bool,
    queued: usize,
    held_by_pages: bool,
    lone_takes_lane: impl FnOnce() -> bool,
) -> bool {
    queued >= ADMIT_COALESCE_MIN_QUEUED
        || (first && queued == 1 && !held_by_pages && lone_takes_lane())
}

/// The timeout of the tick's next straggler wait, `elapsed` into it: one `gap`, never past
/// the whole budget of [`ADMIT_COALESCE_GAPS`] gaps, and none once no free lane is left for
/// one more (`room`). The wait ends at the first gap that brings no arrival.
fn coalesce_next_wait(gap: Duration, elapsed: Duration, room: bool) -> Option<Duration> {
    let left = (gap * ADMIT_COALESCE_GAPS).saturating_sub(elapsed);
    (room && !left.is_zero()).then(|| gap.min(left))
}

/// A promoted request that cannot continue anywhere (its singleton state
/// already left the resident owners): release its lane and emit one
/// generation error for the key.
fn fail_promoted(
    model: &mut LoadedModel,
    stdout: &mut Stdout,
    key: &AttemptKey,
    admission: BatchGeneration,
    epoch: &RequestEpoch,
    message: &str,
) {
    let released = release_epoch(model, epoch);
    let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
    emit_generation_error(
        ROUTE,
        stdout,
        Some(&key.id),
        message,
        "internal",
        false,
        released.is_ok(),
    );
    batch_clear_terminal_at_generation(&key.id, key.attempt_id, admission);
}

/// Install a promoted singleton as a Running lane under its AttemptKey (no
/// new `gen_start`). On failure the lane is released and one generation error
/// is emitted for the key; the request cannot continue anywhere.
fn adopt_promoted(
    sched: &mut ContinuousBatchScheduler,
    model: &mut LoadedModel,
    controls: &mut [Option<LaneCtl>],
    stdout: &mut Stdout,
    promoted: Qwen4PromotedRequest,
) {
    let Qwen4PromotedRequest {
        pending,
        progress,
        producer,
        epoch,
        slot,
        seed_emitted,
        max_tokens,
    } = promoted;
    let key = pending.key.clone();
    let admission = pending.admission;
    let started_in_think = pending.started_in_think;
    let max_think_tokens = pending.max_think_tokens;
    let start = progress.created_at;
    // The request's decode began on the resident singleton at its prefill end,
    // not at adoption: the done-event decode rate spans the whole decode.
    let decode_start = progress.prefill_done_at.unwrap_or_else(Instant::now);
    let prefill_ms = progress.prefill_done_at.map_or(0.0, |t| {
        t.saturating_duration_since(start).as_secs_f64() * 1000.0
    });
    let bytes = model.tokenizer.as_ref().map(|tokenizer| {
        let mut bytes = Vec::new();
        for &token in &progress.streamed_tokens {
            tokenizer.decode_token_bytes_into(token, &mut bytes);
        }
        bytes
    });
    let prompt_len = progress.prompt_len;
    let adopted = bytes.and_then(|bytes| {
        sched
            .adopt_running(pending, progress)
            .map(|ticket| (ticket, bytes))
    });
    let Some((ticket, bytes)) = adopted else {
        fail_promoted(
            model,
            stdout,
            &key,
            admission,
            &epoch,
            "qwen4 lane promotion failed: no lane or tokenizer for the promoted request",
        );
        return;
    };
    // The promoted lane's remaining growth is promised to the page budget like an admitted
    // lane's (best effort: it is already running and cannot be queued).
    let max_seq = model.max_seq;
    if let Some(bundle) = model.qwen4_mut() {
        let rows = reservation_rows(
            prompt_len,
            max_tokens,
            bundle.lane_page_draft_rows(),
            max_seq,
        );
        if let Err(error) = bundle.reserve_lane_pages(&epoch, rows) {
            eprintln!("  qwen4 lane page budget: promoted lane not reserved: {error}");
        }
    }
    controls[ticket.lane] = Some(LaneCtl {
        epoch,
        slot,
        semantic: Some(producer),
        bytes,
        budget: ThinkBudget::new(max_think_tokens),
        forced: VecDeque::new(),
        started_in_think,
        max_tokens,
        start,
        decode_start: Some(decode_start),
        prefill_ms,
        cache: None,
        skip_first_forward: seed_emitted,
        mapped_ahead: false,
        _serving: model.ple_serving_guard(),
    });
}

/// The wire class and retryability of a failed lane step: a mapping refusal
/// ([`PROVISION_MAP_REFUSED`], the lane could not commit the device pages its context needs), a
/// page-budget refusal ([`LANE_PAGE_BUDGET_REFUSED`], the request can never fit the free device
/// pages) or a device out-of-memory failure outside a context map is a capacity condition, answered
/// `overload` + retryable (HTTP 503, the request may be retried when other lanes finish); every
/// other failure keeps the caller's class and is not retryable.
fn lane_error_wire<'a>(message: &str, class: &'a str) -> (&'a str, bool) {
    if message.contains(PROVISION_MAP_REFUSED) || message.contains(LANE_PAGE_BUDGET_REFUSED) {
        ("overload", true)
    } else {
        crate::common::fail_closed_wire(message, class, false)
    }
}

fn lane_error(
    sched: &mut ContinuousBatchScheduler,
    model: &mut LoadedModel,
    controls: &mut [Option<LaneCtl>],
    idx: usize,
    stdout: &mut Stdout,
    message: &str,
    class: &str,
) -> Result<(), String> {
    let (key, admission) = match &sched.lanes[idx] {
        BatchLane::Running(l) | BatchLane::Seeding(l) => (l.key.clone(), l.ticket.admission),
        BatchLane::AwaitingClient(l) => (l.key.clone(), l.ticket.admission),
        BatchLane::Empty { .. } => return Ok(()),
    };
    let released = controls[idx].as_mut().map_or(Ok(()), |c| release(model, c));
    let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
    let (class, retryable) = lane_error_wire(message, class);
    emit_generation_error(
        ROUTE,
        stdout,
        Some(&key.id),
        message,
        class,
        retryable,
        released.is_ok(),
    );
    let _ = sched.abort_lane(idx, &key, admission);
    controls[idx] = None;
    released
}

/// Requests put back in line after a step failed on a capacity condition: how often each went back
/// and which already announced their `gen_start` (it must not be announced twice).
#[derive(Default)]
struct RequeueBook {
    started: HashSet<AttemptKey>,
    counts: std::collections::HashMap<AttemptKey, u32>,
}

/// Times one request may be put back in line before it is failed instead (a request that keeps
/// failing its mappings must not cycle forever).
const MAX_REQUEUES: u32 = 2;

impl RequeueBook {
    /// Whether `key` already announced its start; consumes the mark.
    fn take_started(&mut self, key: &AttemptKey) -> bool {
        self.started.remove(key)
    }

    /// Whether `key` may go back in line once more; counts it when it may.
    fn allow(&mut self, key: &AttemptKey) -> bool {
        if self.counts.len() > 4096 {
            self.counts.clear();
        }
        let count = self.counts.entry(key.clone()).or_insert(0);
        if *count >= MAX_REQUEUES {
            return false;
        }
        *count += 1;
        true
    }
}

/// A lane of a failed step. A mapping refusal ([`PROVISION_MAP_REFUSED`]: the device ran out of pages the
/// lane needed, on a rank the budget could not see) of a request that has emitted nothing yet goes back to
/// the FRONT of the queue (bounded by [`MAX_REQUEUES`]) and is admitted again at a later step boundary, when
/// the page budget re-measures the device: no client error, no second `gen_start`. A refused request that
/// already streamed tokens, a request out of requeues and every other failure fail exactly as before
/// ([`lane_error`]: a mapping refusal is a retryable 503 overload). Mapped pages cannot be handed back to
/// the device (a VMM range that is unmapped and mapped again translates to its old pages on gfx1201, see
/// hip-bridge `vmm.rs`); they stay with the lane's slot, which the budget counts as mapped and reuses.
fn lane_step_failed(
    sched: &mut ContinuousBatchScheduler,
    model: &mut LoadedModel,
    controls: &mut [Option<LaneCtl>],
    idx: usize,
    stdout: &mut Stdout,
    message: &str,
    requeue: &mut RequeueBook,
) -> Result<(), String> {
    if page_budget_enabled() && message.contains(PROVISION_MAP_REFUSED) {
        let candidate = match &sched.lanes[idx] {
            BatchLane::Running(l) | BatchLane::Seeding(l) if l.streamed_tokens.is_empty() => {
                Some((l.key.clone(), l.ticket.admission))
            }
            _ => None,
        };
        if let Some((key, admission)) = candidate {
            if requeue.allow(&key) {
                let released = controls[idx].as_mut().map_or(Ok(()), |c| release(model, c));
                if released.is_ok() && sched.requeue_lane(idx, &key, admission) {
                    controls[idx] = None;
                    requeue.started.insert(key.clone());
                    eprintln!(
                        "  qwen4 lane page budget: requeued {} attempt {} after a mapping refusal mid-step ({message})",
                        key.id, key.attempt_id
                    );
                    return Ok(());
                }
            }
        }
    }
    lane_error(sched, model, controls, idx, stdout, message, "gpu")
}

fn cancel_lane(
    sched: &mut ContinuousBatchScheduler,
    model: &mut LoadedModel,
    controls: &mut [Option<LaneCtl>],
    idx: usize,
    stdout: &mut Stdout,
) -> Result<(), String> {
    let (key, admission, generated) = match &sched.lanes[idx] {
        BatchLane::Running(l) | BatchLane::Seeding(l) => {
            (l.key.clone(), l.ticket.admission, l.streamed_tokens.len())
        }
        BatchLane::AwaitingClient(l) => (
            l.key.clone(),
            l.ticket.admission,
            l.pending_done
                .get("tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize,
        ),
        BatchLane::Empty { .. } => return Ok(()),
    };
    if let Some(c) = controls[idx].as_mut() {
        release(model, c)?;
    }
    let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
    emit_generation_cancel(ROUTE, stdout, &key.id, generated);
    let _ = sched.abort_lane(idx, &key, admission);
    controls[idx] = None;
    Ok(())
}

fn render_request(
    model: &mut LoadedModel,
    request: &BatchPendingRequest,
) -> Result<(Vec<u32>, bool), (String, &'static str)> {
    if request.max_tokens == 0 {
        return Err(("max_tokens must be > 0".to_string(), "validation"));
    }
    let msg = &request.original_msg;
    let prompt = hipfire_runtime::tokenizer::maybe_normalize_prompt(
        msg.get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("Hello"),
    );
    let history = hipfire_engine::prompt::parse_generate_messages(msg)
        .map_err(|e| (format!("invalid messages field: {e}"), "validation"))?;
    let tools = msg
        .get("tools")
        .and_then(|v| v.as_array())
        .filter(|a| !a.is_empty())
        .map(|a| a.as_slice());
    let raw_effort = msg
        .get("reasoning_effort")
        .or_else(|| msg.get("thinking_mode"))
        .and_then(|v| v.as_str());
    let (enabled, effort) = hipfire_engine::prompt::qwen_jinja_reasoning(
        msg.get("thinking_enabled").and_then(|v| v.as_bool()),
        raw_effort,
        request.max_think_tokens,
    );
    crate::qwen::render_qwen4_prompt(
        model,
        crate::qwen::Qwen4PromptRoute::Ar,
        msg.get("system").and_then(|v| v.as_str()),
        &prompt,
        request.assistant_prefix,
        tools,
        history.as_deref(),
        enabled,
        effort.as_deref(),
    )
}

/// A rendered prompt: its token ids and whether generation starts inside a think block, or the
/// refusal and its wire class.
type Rendered = Result<(Vec<u32>, bool), (String, &'static str)>;

/// Generation budget of a lane request over its rendered `prompt_len`
/// (`max_tokens_fit` clamps it to the context left), the one rule the page
/// gate's reservation and the admission checks share.
fn lane_request_max_tokens(req: &BatchPendingRequest, prompt_len: usize, max_seq: usize) -> usize {
    fit_max_tokens_if(
        req.original_msg
            .get("max_tokens_fit")
            .and_then(|v| v.as_bool())
            == Some(true),
        req.max_tokens,
        prompt_len + 1,
        max_seq,
    )
}

/// What the lane page budget does with the queue front ([`gate_decision`]).
#[derive(Debug, PartialEq, Eq)]
enum GateDecision {
    /// Assign it a lane.
    Admit,
    /// Keep it queued; ask again at the next step boundary.
    Queue(hipfire_arch_qwen4::lane::pages::PageShortfall),
    /// It can never fit: assign it a lane only to refuse it (a retryable overload).
    Refuse(hipfire_arch_qwen4::lane::pages::PageShortfall),
}

/// The gate's decision for a budget `verdict` with `active_lanes` lanes running on the
/// scheduler. A request that does not fit waits only while a running lane can still
/// change the answer; with no lane active, waiting would spin.
fn gate_decision(verdict: PageVerdict, active_lanes: usize) -> GateDecision {
    match verdict {
        PageVerdict::Fit => GateDecision::Admit,
        PageVerdict::Wait(short) if active_lanes > 0 => GateDecision::Queue(short),
        PageVerdict::Wait(short) | PageVerdict::Never(short) => GateDecision::Refuse(short),
    }
}

/// A request held back by the page budget, kept across ticks so the wait is announced once and
/// its prompt is rendered once.
struct PageWait {
    key: AttemptKey,
    since: Instant,
    ticks: u64,
    announced: bool,
    rendered: Option<Rendered>,
}

/// The page gate's answer for the queue front.
enum PageGate {
    /// Assign the front a lane. Carries the prompt the gate rendered (so it is not rendered again)
    /// and the refusal a request that can never fit gets once it holds a lane.
    Proceed {
        rendered: Option<Rendered>,
        refusal: Option<String>,
        /// The slot the budget chose (`None`: the lowest free one).
        slot: Option<usize>,
    },
    /// Leave the front queued: stop assigning this tick.
    Wait,
}

impl PageGate {
    fn proceed(rendered: Option<Rendered>, refusal: Option<String>, slot: Option<usize>) -> Self {
        Self::Proceed {
            rendered,
            refusal,
            slot,
        }
    }
}

/// The page budget's decision for the queue front reserving `rows` rows. `None`: the budget could
/// not be measured, which the gate answers by admitting `id` without it (logged here).
fn gate_page_decision(
    m: &mut LoadedModel,
    gpu: &mut Gpu,
    rows: usize,
    id: &str,
) -> Result<Option<PageDecision>, String> {
    match Owner::of(m, gpu)?.page_decision(rows) {
        Ok(decision) => Ok(Some(decision)),
        Err(error) => {
            eprintln!("  qwen4 lane page budget: {error}; admitting {id} without the budget");
            Ok(None)
        }
    }
}

/// The lane page budget at the queue front (see `hipfire_arch_qwen4::lane::pages`): before the
/// front request takes a lane, reserve-check its worst-case context growth against every rank's
/// free device pages. `Proceed` with nothing when the kill switch is set
/// (`HIPFIRE_QWEN4_LANE_PAGE_BUDGET=0`), nothing is assignable, the request goes back to the
/// singleton, the admission below refuses it on its own terms, or the budget cannot be measured.
fn page_budget_gate(
    sched: &ContinuousBatchScheduler,
    m: &mut LoadedModel,
    gpu: &mut Gpu,
    controls: &[Option<LaneCtl>],
    initial: &HashSet<AttemptKey>,
    requeued: &RequeueBook,
    wait: &mut Option<PageWait>,
) -> Result<PageGate, String> {
    if !page_budget_enabled() {
        return Ok(PageGate::proceed(None, None, None));
    }
    let Some(key) = sched.assignable_front().cloned() else {
        return Ok(PageGate::proceed(None, None, None));
    };
    let Some(req) = sched.pending.get(&key) else {
        return Ok(PageGate::proceed(None, None, None));
    };
    if !batch_is_current(&key.id, key.attempt_id, req.admission) {
        return Ok(PageGate::proceed(None, None, None));
    }
    let mut state = match wait.take() {
        Some(state) if state.key == key => state,
        _ => PageWait {
            key: key.clone(),
            since: Instant::now(),
            ticks: 0,
            announced: false,
            rendered: None,
        },
    };
    let rendered = state
        .rendered
        .take()
        .unwrap_or_else(|| render_request(m, req));
    // A request the singleton route takes back is not a lane request: no budget.
    let lone = sched.inbox.len() == 1 && !controls.iter().flatten().any(|c| c.epoch.is_admitted());
    let takes_lane = qwen4_lone_request_takes_lane(m);
    if lone_arrival_hands_back(
        lone,
        takes_lane,
        initial.contains(&key) || requeued.started.contains(&key),
        || {
            rendered.as_ref().is_ok_and(|(prompt, _)| {
                crate::qwen::plan_qwen4_session(m, prompt, SessionRoute::Ar) > 0
            })
        },
    ) {
        return Ok(PageGate::proceed(Some(rendered), None, None));
    }
    let Ok((prompt, _)) = &rendered else {
        return Ok(PageGate::proceed(Some(rendered), None, None));
    };
    let max_tokens = lane_request_max_tokens(req, prompt.len(), m.max_seq);
    let fits_context = prompt
        .len()
        .checked_add(max_tokens)
        .and_then(|n| n.checked_add(1))
        .is_some_and(|n| n <= m.max_seq);
    if req.max_tokens == 0 || prompt.is_empty() || !fits_context {
        // The admission below refuses it as a validation / context-length error.
        return Ok(PageGate::proceed(Some(rendered), None, None));
    }
    let prompt_len = prompt.len();
    let rows = reservation_rows(
        prompt_len,
        max_tokens,
        Owner::of(m, gpu)?.bundle.lane_page_draft_rows(),
        m.max_seq,
    );
    // The elastic reserve (PLAN-cp15 L2): a first prefill tile of >= 512 route rows builds the F16
    // weight shadows out of the memory the bank holds, so every rank gives the bank back before
    // the budget is read (the page numbers then see the freed bytes).
    {
        let mut owner = Owner::of(m, gpu)?;
        if owner.elastic_active()
            && long_prefill_route_rows(owner.bundle.lane_prefill_tile_len(prompt, 0))
        {
            owner.elastic_reclaim("admit");
        }
    }
    let Some(mut decision) = gate_page_decision(m, gpu, rows, &key.id)? else {
        return Ok(PageGate::proceed(Some(rendered), None, None));
    };
    // A request the budget cannot admit while a bank is lent decides again once it is back.
    if !matches!(decision.verdict, PageVerdict::Fit) && Owner::of(m, gpu)?.elastic_reclaim("pages")
    {
        let Some(again) = gate_page_decision(m, gpu, rows, &key.id)? else {
            return Ok(PageGate::proceed(Some(rendered), None, None));
        };
        decision = again;
    }
    let slot = decision.slot;
    match gate_decision(decision.verdict, sched.active_count()) {
        GateDecision::Admit => {
            if state.announced {
                eprintln!(
                    "  qwen4 lane page budget: admitted {} attempt {} after {:.0} ms and {} step boundaries queued",
                    key.id,
                    key.attempt_id,
                    state.since.elapsed().as_secs_f64() * 1000.0,
                    state.ticks
                );
            }
            Ok(PageGate::proceed(Some(rendered), None, slot))
        }
        GateDecision::Queue(short) => {
            state.ticks += 1;
            if !state.announced {
                state.announced = true;
                eprintln!(
                    "  qwen4 lane page budget: queued {} attempt {}: {prompt_len} prompt + {max_tokens} max tokens reserve {rows} context rows; {short}",
                    key.id, key.attempt_id
                );
            }
            state.rendered = Some(rendered);
            *wait = Some(state);
            Ok(PageGate::Wait)
        }
        GateDecision::Refuse(short) => {
            eprintln!(
                "  qwen4 lane page budget: refused {} attempt {}: {prompt_len} prompt + {max_tokens} max tokens reserve {rows} context rows and cannot fit with no lane active; {short}",
                key.id, key.attempt_id
            );
            Ok(PageGate::proceed(
                Some(rendered),
                Some(format!(
                    "{LANE_PAGE_BUDGET_REFUSED}the request's context ({rows} rows) does not fit the free device memory: {short}"
                )),
                slot,
            ))
        }
    }
}

/// The serve loop's idle count for the elastic reserve (`Qwen4Bundle::elastic_tick`): step
/// boundaries since the last prefill tile of >= 512 route rows or elastic reclaim.
#[derive(Default)]
struct ElasticIdle {
    steps: usize,
    /// The root bundle's reclaim count when `steps` last restarted.
    reclaims: u64,
}

/// Step-boundary housekeeping of the elastic reserve; a no-op without it. `step` is
/// `Some(long_prefill)` after a step (`long_prefill`: it held a tile of >= 512 route rows) and
/// `None` on an idle iteration (the loop's 2 ms idle sleep), which counts like a step so an idle
/// server gives back the caches and re-lends without waiting for traffic. The idle count restarts
/// after a long prefill or any reclaim (a gate, backstop or borrow reclaim all move the bundle's reclaim count); the
/// tick runs only while no request is queued and no lane prefills. Never fails the loop: a failed
/// tick is logged by the bundle and the next one asks again.
fn elastic_boundary(
    sched: &ContinuousBatchScheduler,
    m: &mut LoadedModel,
    gpu: &mut Gpu,
    controls: &[Option<LaneCtl>],
    idle: &mut ElasticIdle,
    step: Option<bool>,
) {
    match Owner::of(m, gpu) {
        Ok(owner) if owner.elastic_active() => {}
        _ => return,
    }
    let prefilling = m
        .qwen4()
        .and_then(|bundle| bundle.lanes())
        .is_some_and(|store| {
            controls
                .iter()
                .flatten()
                .filter(|ctl| ctl.epoch.is_admitted())
                .any(|ctl| {
                    store
                        .lane(&ctl.epoch)
                        .is_some_and(|lane| lane.host.status == LaneStatus::Prefilling)
                })
        });
    let Ok(mut owner) = Owner::of(m, gpu) else {
        return;
    };
    let reclaims = owner.bundle.elastic_reclaim_epoch();
    if reclaims != idle.reclaims || step == Some(true) {
        idle.steps = 0;
    } else {
        idle.steps = idle.steps.saturating_add(1);
    }
    idle.reclaims = reclaims;
    if prefilling || !sched.inbox.is_empty() {
        return;
    }
    owner.bundle.elastic_tick(&mut *owner.gpu, idle.steps);
}

/// A failed `draft_step`: its error and every lane it named.
struct DraftFailure {
    error: String,
    epochs: Vec<RequestEpoch>,
}

/// One scheduler tick's work: the lane executor's plan (everything but the
/// mesh lanes' decode) and, on an armed EP mesh store, the decoding lanes the
/// root plans and the ranks replay as one mesh step ([`MeshDecodeRequest`]).
struct PlannedStep {
    plan: BatchStepPlan,
    mesh: Vec<MeshDecodeRequest>,
}

/// `HIPFIRE_CB_PHASES=1` (developer diagnostic): one `[fn-lane-step]` stderr
/// line per lane step. `host` is the wall time from the previous step's end
/// to this plan (terminal decisions, inbox, admission); `plan` holds the
/// batched draft (one D2H per draft step); `fwd` is provision + forward
/// (synchronized by the pick download); then `commit` (the executor's commit and the
/// mesh decode window) and `emit`, which also holds the prefill ticks emitted before the
/// mesh window (`HIPFIRE_LANE_FIRST_TOKEN_BEFORE_MESH`). Rows by
/// kind, decoding/active lanes, the queue and the committed tokens follow.
struct StepTrace {
    origin: Instant,
    last: Instant,
    step: u64,
}

impl StepTrace {
    fn new() -> Option<Self> {
        let now = Instant::now();
        hipfire_config::developer_var("HIPFIRE_CB_PHASES")
            .is_ok_and(|v| v.trim() == "1")
            .then_some(Self {
                origin: now,
                last: now,
                step: 0,
            })
    }

    /// `marks`: plan start, forward start, commit start, commit end. `early_emit`: the
    /// emit time spent inside the commit window (prefill ticks before the mesh window),
    /// reported under `emit` and not `commit`.
    #[allow(clippy::too_many_arguments)]
    fn line(
        &mut self,
        marks: [Instant; 4],
        plan: &BatchStepPlan,
        mesh: &MeshStepFacts,
        tokens: usize,
        accepted: usize,
        active: usize,
        queued: usize,
        early_emit: Duration,
    ) {
        let now = Instant::now();
        let ms = |a: Instant, b: Instant| b.saturating_duration_since(a).as_secs_f64() * 1000.0;
        let early_ms = early_emit.as_secs_f64() * 1000.0;
        let (mut pf, mut pf_rows, mut ar, mut ver, mut ver_rows) = (0, 0, 0, 0, 0);
        for r in &plan.requests {
            match r.kind {
                RequestStepKind::Prefill => {
                    pf += 1;
                    pf_rows += r.rows.len;
                }
                RequestStepKind::Ar => ar += 1,
                RequestStepKind::Verify { .. } => {
                    ver += 1;
                    ver_rows += r.rows.len;
                }
                RequestStepKind::Forced => {}
            }
        }
        eprintln!("[fn-lane-step] n={} t={:.1} host={:.2} plan={:.2} fwd={:.2} commit={:.2} emit={:.2} active={active} queued={queued} \
            pf={pf} pf_rows={pf_rows} ar={ar} ver={ver} ver_rows={ver_rows} mesh_lanes={} mesh_native={} mesh_copy={} mesh_ar={} tokens={tokens} accepted={accepted}",
            self.step, ms(self.origin, marks[0]), ms(self.last, marks[0]), ms(marks[0], marks[1]), ms(marks[1], marks[2]),
            (ms(marks[2], marks[3]) - early_ms).max(0.0), ms(marks[3], now) + early_ms, mesh.lanes, mesh.native_windows, mesh.copy_windows, mesh.ar_windows);
        hip_bridge::site_ledger::mark_step(self.step, mesh.lanes, pf_rows, tokens);
        self.step += 1;
        self.last = now;
    }
}

/// Route mix of one drive's EP mesh decode steps, keyed by the lanes a step
/// carried (the step's concurrency): windows by final source (native drafts,
/// copy candidates, AR rows), drafts offered/accepted and tokens emitted.
/// Logged once when the drive ends (`[fn-lane-mesh-mix]`), so a receipt needs
/// no per-step diagnostic flag.
#[derive(Default)]
struct MeshMix {
    by_lanes: std::collections::BTreeMap<usize, MeshStepFacts>,
    steps: std::collections::BTreeMap<usize, usize>,
}

impl MeshMix {
    fn note(&mut self, facts: &MeshStepFacts) {
        if facts.lanes == 0 {
            return;
        }
        let sum = self.by_lanes.entry(facts.lanes).or_default();
        sum.lanes += facts.lanes;
        sum.native_windows += facts.native_windows;
        sum.copy_windows += facts.copy_windows;
        sum.ar_windows += facts.ar_windows;
        sum.offered += facts.offered;
        sum.accepted += facts.accepted;
        sum.emitted += facts.emitted;
        *self.steps.entry(facts.lanes).or_default() += 1;
    }

    fn report(&self) {
        for (lanes, sum) in &self.by_lanes {
            eprintln!(
                "[fn-lane-mesh-mix] lanes={lanes} steps={} windows={} native={} copy={} ar={} \
                 offered={} accepted={} emitted={}",
                self.steps.get(lanes).copied().unwrap_or(0),
                sum.native_windows + sum.copy_windows + sum.ar_windows,
                sum.native_windows,
                sum.copy_windows,
                sum.ar_windows,
                sum.offered,
                sum.accepted,
                sum.emitted
            );
        }
    }
}

/// Build directly rather than using `BatchPlanner::forced_prefill_len`: that
/// planner charges exact singleton prefill tiles against the AR row budget,
/// so a tile larger than the budget could never run. Here only decode lanes
/// use that budget (one lane per budget row; a lane's verify block may carry
/// more rows); every prefilling owner gets at most one indivisible tile.
/// `streamed[i]` is the tokens control `i` has streamed so far.
fn plan_step(
    model: &mut LoadedModel,
    gpu: &mut Gpu,
    controls: &mut [Option<LaneCtl>],
    streamed: &[usize],
    row_budget: usize,
    cursor: &mut usize,
    max_lanes: usize,
) -> Result<Result<PlannedStep, DraftFailure>, String> {
    let max_seq = model.max_seq;
    let mut owner = Owner::of(model, gpu)?;
    let mut order: Vec<_> = controls
        .iter()
        .flatten()
        .filter(|c| c.epoch.is_admitted())
        .map(|c| (c.slot, c.epoch))
        .collect();
    order.sort_by_key(|&(slot, _)| slot);
    let store = owner.bundle.lanes().ok_or("qwen4 lanes not staged")?;
    let spec_k = store.spec().map(|s| s.k);
    // Armed EP native-MTP lanes: decode runs through the mesh lane protocol
    // (root plans, every rank replays); the executor plans prefill only.
    let mesh_mode = owner.bundle.ep_mesh_lanes_armed();
    let decoding: Vec<_> = order
        .iter()
        .filter_map(|&(slot, epoch)| {
            store
                .lane(&epoch)
                .filter(|l| l.host.status == LaneStatus::Decoding)
                .map(|_| slot)
        })
        .collect();
    let count = decoding.len();
    let chosen: Vec<_> = (0..count.min(row_budget))
        .map(|j| decoding[(*cursor + j) % count])
        .collect();
    if count != 0 {
        *cursor = (*cursor + count.min(row_budget)) % count;
    }
    for c in controls
        .iter_mut()
        .flatten()
        .filter(|c| chosen.contains(&c.slot))
    {
        if let Some(token) = c.forced.pop_front() {
            owner.force_seed(&c.epoch, token)?;
        }
    }
    // Once per request, before its first draft/decode step (and so before
    // any provision): map the lane's context for the rest of its token
    // budget (best effort, `provision_step` grows on demand otherwise).
    if !chosen.is_empty() {
        let mut executor = owner.executor()?;
        for (idx, c) in controls.iter_mut().enumerate() {
            let Some(c) = c
                .as_mut()
                .filter(|c| !c.mapped_ahead && chosen.contains(&c.slot))
            else {
                continue;
            };
            c.mapped_ahead = true;
            executor.map_ahead_decode(&c.epoch, c.max_tokens.saturating_sub(streamed[idx]));
        }
    }
    let mut drafts: Vec<LaneDraft> = Vec::new();
    let mut mesh_requests: Vec<MeshDecodeRequest> = Vec::new();
    if let Some(k) = spec_k {
        let mut requests = Vec::new();
        for (idx, c) in controls.iter().enumerate() {
            let Some(c) = c.as_ref().filter(|c| chosen.contains(&c.slot)) else {
                continue;
            };
            let position = owner
                .bundle
                .lanes()
                .and_then(|s| s.lane(&c.epoch))
                .map(|l| l.state.position)
                .ok_or("admitted qwen4 lane disappeared")?;
            let budget = if c.forced.is_empty() {
                k.min(c.max_tokens.saturating_sub(streamed[idx]).saturating_sub(1))
                    .min(max_seq.saturating_sub(position).saturating_sub(1))
            } else {
                0
            };
            requests.push((c.epoch, budget));
        }
        if mesh_mode {
            // The root derives depth itself (native up to `k`, copy up to the
            // copy take cap): it gets the tokens left, not the native budget.
            mesh_requests = controls
                .iter()
                .enumerate()
                .filter_map(|(idx, c)| {
                    let c = c.as_ref().filter(|c| chosen.contains(&c.slot))?;
                    Some(MeshDecodeRequest {
                        epoch: c.epoch,
                        tokens_left: if c.forced.is_empty() {
                            c.max_tokens.saturating_sub(streamed[idx]).max(1)
                        } else {
                            1
                        },
                    })
                })
                .collect();
        } else if !requests.is_empty() {
            let mut executor = owner.executor()?;
            match executor.draft_step(&requests) {
                Ok(d) => drafts = d,
                Err(error) => {
                    return Ok(Err(DraftFailure {
                        error,
                        epochs: requests.iter().map(|r| r.0).collect(),
                    }));
                }
            }
        }
    }
    let store = owner.bundle.lanes().ok_or("qwen4 lanes not staged")?;
    let mut rows = Vec::new();
    for (slot, epoch) in order {
        let lane = store
            .lane(&epoch)
            .ok_or("admitted qwen4 lane disappeared")?;
        match lane.host.status {
            LaneStatus::Prefilling => rows.push((
                slot,
                epoch,
                RequestStepKind::Prefill,
                lane.host.prefilled,
                0,
                Vec::new(),
            )),
            LaneStatus::Decoding if mesh_mode && chosen.contains(&slot) => {}
            LaneStatus::Decoding if chosen.contains(&slot) => {
                let position = lane.state.position;
                match drafts.iter().find(|d| d.epoch == epoch) {
                    // A batched window, or (a copy-armed store only) any
                    // window that carries the root's copy candidates.
                    Some(LaneDraft { route, drafts, .. })
                        if matches!(route, LaneRoute::Batched { .. }) || !drafts.is_empty() =>
                    {
                        if drafts.is_empty() {
                            return Err("qwen4 batched lane window drafted no tokens".into());
                        }
                        rows.push((
                            slot,
                            epoch,
                            RequestStepKind::Verify {
                                draft_len: drafts.len(),
                            },
                            position,
                            1 + drafts.len(),
                            drafts.clone(),
                        ));
                    }
                    Some(_) => {
                        rows.push((slot, epoch, RequestStepKind::Ar, position, 1, Vec::new()))
                    }
                    None if spec_k.is_none() => {
                        rows.push((slot, epoch, RequestStepKind::Ar, position, 1, Vec::new()))
                    }
                    None => return Err("qwen4 draft step omitted a planned lane".into()),
                }
            }
            LaneStatus::Decoding => {}
            LaneStatus::Poisoned => return Err("qwen4 lane poisoned outside a step".into()),
        }
    }
    {
        let executor = owner.executor()?;
        for (_, epoch, kind, _, len, _) in &mut rows {
            if *kind == RequestStepKind::Prefill {
                *len = executor.exact_prefill_chunk_len(epoch)?;
            }
        }
    }
    let store = owner.bundle.lanes().ok_or("qwen4 lanes not staged")?;
    let mut plan = BatchStepPlan::default();
    plan.batch.m_per_slot = vec![0; max_lanes];
    for (slot, epoch, kind, position, len, drafted) in rows {
        if len == 0 {
            return Err("qwen4 exact prefill tile is empty".into());
        }
        let lane = store
            .lane(&epoch)
            .ok_or("admitted qwen4 lane disappeared")?;
        let begin = plan.batch.tokens.len();
        if kind == RequestStepKind::Prefill {
            let end = position
                .checked_add(len)
                .ok_or("qwen4 tile position overflow")?;
            let slice = lane
                .host
                .prompt
                .get(position..end)
                .ok_or("qwen4 prefill tile exceeds prompt")?;
            plan.batch.tokens.extend_from_slice(slice);
        } else {
            // The root's pending seed (a PP band's lane never reaches the
            // single-card root; its host mirror is its whole state).
            plan.batch.tokens.push(
                store
                    .root_seed(&epoch)
                    .or(lane.host.pending_seed)
                    .ok_or("decoding lane has no pending seed")?,
            );
            plan.batch.tokens.extend_from_slice(&drafted);
        }
        for j in 0..len {
            plan.batch
                .positions
                .push(i32::try_from(position + j).map_err(|_| "lane position exceeds i32")?);
            plan.batch
                .row_slot
                .push(i32::try_from(slot).map_err(|_| "lane slot exceeds i32")?);
        }
        plan.batch.m_per_slot[slot] = len;
        plan.requests.push(RequestRows {
            epoch,
            rows: RowRange { begin, len },
            kind,
        });
        if kind == RequestStepKind::Prefill {
            plan.prefill_rows += len;
        } else {
            plan.decode_rows += len;
        }
    }
    Ok(Ok(PlannedStep {
        plan,
        mesh: mesh_requests,
    }))
}

/// Whether the token just committed ends a lane's turn: the filter or a stop
/// sequence stopped it, or it is an EOS / terminator id while stops are active.
/// Below `min_tokens` (`stops_active` false) an EOS id is an ordinary token:
/// the lane's windows ran through it and the stream continues.
fn turn_ends(stopped: bool, stops_active: bool, is_eos: bool) -> bool {
    stopped || (stops_active && is_eos)
}

fn emit_forwarded(
    stdout: &mut Stdout,
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    lane: &mut QwenBatchLane,
    ctl: &mut LaneCtl,
    token: u32,
    eos: u32,
    adv_length: bool,
) -> Result<Option<bool>, String> {
    let fed = ctl.bytes.len();
    tokenizer.decode_token_bytes_into(token, &mut ctl.bytes);
    let new_bytes = &ctl.bytes[fed..];
    let elapsed = ctl.start.elapsed().as_millis() as u64;
    lane.first_token_at.get_or_insert_with(Instant::now);
    let id = &lane.key.id;
    let producer = ctl.semantic.as_mut().ok_or("lane producer missing")?;
    let stopped = producer
        .commit_and_classify(
            stdout,
            token,
            || {
                let pos = qwen_ar_raw_commit_token(
                    &mut lane.conversation_tokens,
                    &mut lane.streamed_tokens,
                    &mut lane.seq_pos,
                    token,
                    QwenArRawCommitDisposition::ClassifiedVisible,
                );
                (pos, new_bytes)
            },
            |pos, output| emit_committed_event(output, id, token, pos, elapsed),
        )
        .map_err(|e| e.to_string())?;
    if turn_ends(
        stopped,
        producer.last_stops_active,
        token == eos || tokenizer.is_terminator(token),
    ) {
        return Ok(Some(false));
    }
    match qwen_ar_think_budget_step(&mut ctl.budget, &ctl.bytes, ctl.started_in_think, tokenizer) {
        QwenArThinkBudgetAction::Continue => {}
        QwenArThinkBudgetAction::Close(tokens) => ctl.forced.extend(tokens),
        QwenArThinkBudgetAction::Stop => return Ok(Some(false)),
    }
    Ok((lane.streamed_tokens.len() >= ctl.max_tokens || adv_length).then_some(true))
}

fn finish_lane(
    sched: &mut ContinuousBatchScheduler,
    model: &mut LoadedModel,
    controls: &mut [Option<LaneCtl>],
    idx: usize,
    stdout: &mut Stdout,
    length: bool,
) -> Result<(), String> {
    let ctl = controls[idx].as_mut().ok_or("lane control missing")?;
    let BatchLane::Running(lane) = &sched.lanes[idx] else {
        return Err("finished lane is not running".into());
    };
    let key = lane.key.clone();
    let admission = lane.ticket.admission;
    let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
    let (finish, visible) = ctl
        .semantic
        .take()
        .ok_or("lane producer missing")?
        .finish(stdout, length)
        .map_err(|e| e.to_string())?;
    let now = Instant::now();
    let generated = lane.streamed_tokens.len();
    let total = now.duration_since(ctl.start).as_secs_f64();
    let decode = ctl
        .decode_start
        .map_or(0.0, |t| now.duration_since(t).as_secs_f64());
    let mut done = qwen_ar_done_value(
        &key.id,
        finish.finish_reason,
        generated,
        batch_finite_rate(generated, total),
        lane.prompt_len,
        ctl.prefill_ms,
        batch_finite_rate(lane.prompt_len, ctl.prefill_ms / 1000.0),
        batch_finite_rate(generated, decode),
        ctl.prefill_ms,
        0,
        "",
    );
    stage_terminal_tool_calls(&mut done, finish.finish_reason, &finish.wire_tool_calls);
    attach_continuous_batch_route_evidence(
        &mut done,
        sched.lanes.len(),
        idx,
        sched.lane_capacity,
        lane.max_active_lanes.max(1),
    );
    // Per-stream MTP counters (before `release` drops the lane), under the
    // singleton's done keys; `mtp_drafts`/`mtp_accepted` and the per-position
    // `mtp_depth_*` arrays are lane-only.
    if let Some(stats) = model
        .qwen4()
        .and_then(|b| b.lane_spec_stats_for(&ctl.epoch))
    {
        attach_mtp_lane_stats(&mut done, &stats);
    }
    let action = qwen_ar_cache_action(&finish, &visible);
    if action.store {
        let tokenizer = model.tokenizer.as_ref().ok_or("tokenizer disappeared")?;
        let im_end = tokenizer.encode("<|im_end|>");
        let im_end = if im_end.len() == 1 {
            Some(im_end[0])
        } else {
            None
        };
        let nl = tokenizer.encode("\n").into_iter().collect();
        ctl.cache = Some((
            action,
            crate::qwen::qwen_dflash_cache_seq(&lane.streamed_tokens, im_end, &nl),
        ));
    }
    release(model, ctl)?;
    let mut ready = done.clone();
    ready["type"] = serde_json::json!("commit_ready");
    if !sched.mark_awaiting_commit(idx, done) {
        return Err("qwen4 mark_awaiting_commit failed".into());
    }
    writeln!(stdout, "{ready}")
        .and_then(|()| stdout.flush())
        .map_err(|e| e.to_string())
}

/// Whether `advance` is the executor's tick of a lane whose prompt chunk this step ran: not a
/// decode advance and not an EP mesh window (which carries its seed).
fn is_prefill_tick(
    plan: &BatchStepPlan,
    advance: &RequestAdvance,
    decode_seed: Option<u32>,
) -> bool {
    decode_seed.is_none()
        && plan
            .requests
            .iter()
            .find(|r| r.epoch == advance.epoch)
            .is_some_and(|r| r.kind == RequestStepKind::Prefill)
}

/// Emit one lane's committed tick: a prefill tick streams `prefill_progress`, stamps the
/// prefill timers and (early first token) the lane's first committed token; any other tick
/// streams its forwarded window (`forwarded_tokens`). `decode_seed` is the mesh window's
/// pending seed, `None` for the executor's own ticks. Terminal finishes and per-lane errors
/// are handled here; `Err` is a driver-level failure.
#[allow(clippy::too_many_arguments)]
fn emit_tick(
    sched: &mut ContinuousBatchScheduler,
    m: &mut LoadedModel,
    gpu: &mut Gpu,
    controls: &mut [Option<LaneCtl>],
    stdout: &mut Stdout,
    plan: &BatchStepPlan,
    eos: u32,
    advance: RequestAdvance,
    decode_seed: Option<u32>,
) -> Result<(), String> {
    let idx = controls
        .iter()
        .position(|c| c.as_ref().is_some_and(|c| c.epoch == advance.epoch))
        .ok_or("qwen4 advance has no request owner")?;
    let request = plan.requests.iter().find(|r| r.epoch == advance.epoch);
    let is_prefill = is_prefill_tick(plan, &advance, decode_seed);
    let plan_seed = match (decode_seed, request) {
        (Some(seed), _) => seed,
        (None, Some(r)) => plan.batch.tokens[r.rows.begin],
        (None, None) => return Err("qwen4 advance not in plan".into()),
    };
    if is_prefill {
        // `prefill_progress` for the lane's prompt chunk (the
        // last chunk is silent: `done < total` only).
        if let BatchLane::Running(lane) | BatchLane::Seeding(lane) = &sched.lanes[idx] {
            let progress = Owner::of(m, gpu).ok().and_then(|owner| {
                let cell = owner.bundle.lanes()?.lane(&advance.epoch)?;
                Some((cell.host.prefilled, cell.host.prompt.len()))
            });
            if let Some((done, total)) = progress {
                crate::common::emit_prefill_progress(
                    stdout,
                    &lane.key.id,
                    Some(lane.key.attempt_id),
                    done,
                    total,
                );
            }
        }
        if !advance.committed_ids.is_empty() {
            let ctl = controls[idx].as_mut().ok_or("lane control missing")?;
            ctl.prefill_ms = ctl.start.elapsed().as_secs_f64() * 1000.0;
            ctl.decode_start = Some(Instant::now());
            if let BatchLane::Running(lane) = &mut sched.lanes[idx] {
                lane.prefill_done_at = ctl.decode_start;
            }
        }
        let length_finish = advance.finish.as_deref() == Some("length");
        // The prompt-completing pick streams in THIS step (it is the lane's
        // committed first token; its next decode advance then skips it as the
        // plan seed), not one decode step later with that step's window.
        // `HIPFIRE_LANE_EARLY_FIRST_TOKEN=0` restores the deferred emission.
        let early = early_first_token_enabled()
            && !advance.committed_ids.is_empty()
            && matches!(sched.lanes[idx], BatchLane::Running(_))
            && controls[idx]
                .as_ref()
                .is_some_and(|c| !c.skip_first_forward);
        if early {
            if let BatchLane::Running(lane) = &sched.lanes[idx] {
                if batch_check_abort(&lane.key.id, lane.key.attempt_id, lane.ticket.admission) {
                    return cancel_lane(sched, m, controls, idx, stdout);
                }
            }
            let BatchLane::Running(lane) = &mut sched.lanes[idx] else {
                return Err("prefilled lane is not running".into());
            };
            let _scope = BatchAttemptScope::enter_for_generation(
                &lane.key.id,
                lane.key.attempt_id,
                lane.ticket.admission,
            );
            let tokenizer = m.tokenizer.as_ref().ok_or("tokenizer disappeared")?;
            let ctl = controls[idx].as_mut().ok_or("lane control missing")?;
            ctl.skip_first_forward = true;
            let emitted = emit_forwarded(
                stdout,
                tokenizer,
                lane,
                ctl,
                advance.committed_ids[0],
                eos,
                length_finish,
            );
            return match emitted {
                Ok(Some(length)) => match finish_lane(sched, m, controls, idx, stdout, length) {
                    Ok(()) => Ok(()),
                    Err(error) => lane_error(sched, m, controls, idx, stdout, &error, "validation"),
                },
                Ok(None) => Ok(()),
                Err(error) => lane_error(sched, m, controls, idx, stdout, &error, "validation"),
            };
        }
        if length_finish {
            finish_lane(sched, m, controls, idx, stdout, true)?;
        }
        return Ok(());
    }
    // All step participants committed before any bytes are published.
    if let BatchLane::Running(lane) = &sched.lanes[idx] {
        if batch_check_abort(&lane.key.id, lane.key.attempt_id, lane.ticket.admission) {
            return cancel_lane(sched, m, controls, idx, stdout);
        }
    }
    // Forwarded tokens: the plan seed, then every committed id but
    // the last (the next pending seed). One entry for an AR row. A
    // promoted lane's first plan seed was streamed by the singleton.
    let skip_seed = controls[idx]
        .as_mut()
        .is_some_and(|c| std::mem::take(&mut c.skip_first_forward));
    let forwarded = forwarded_tokens(plan_seed, &advance.committed_ids, skip_seed);
    let length_finish = advance.finish.as_deref() == Some("length");
    let _scope = match &sched.lanes[idx] {
        BatchLane::Running(lane) => BatchAttemptScope::enter_for_generation(
            &lane.key.id,
            lane.key.attempt_id,
            lane.ticket.admission,
        ),
        _ => return Err("advanced lane is not running".into()),
    };
    if forwarded.is_empty() && length_finish {
        // The already-streamed promoted seed was the only forwarded token.
        return match finish_lane(sched, m, controls, idx, stdout, true) {
            Ok(()) => Ok(()),
            Err(error) => lane_error(sched, m, controls, idx, stdout, &error, "validation"),
        };
    }
    for (i, &token) in forwarded.iter().enumerate() {
        let BatchLane::Running(lane) = &mut sched.lanes[idx] else {
            return Err("advanced lane is not running".into());
        };
        let tokenizer = m.tokenizer.as_ref().ok_or("tokenizer disappeared")?;
        let emitted = emit_forwarded(
            stdout,
            tokenizer,
            lane,
            controls[idx].as_mut().ok_or("lane control missing")?,
            token,
            eos,
            length_finish && i + 1 == forwarded.len(),
        );
        match emitted {
            Ok(Some(length)) => {
                if let Err(error) = finish_lane(sched, m, controls, idx, stdout, length) {
                    lane_error(sched, m, controls, idx, stdout, &error, "validation")?;
                }
                break;
            }
            Ok(None) => {}
            Err(error) => {
                lane_error(sched, m, controls, idx, stdout, &error, "validation")?;
                break;
            }
        }
    }
    Ok(())
}

/// Serve concurrently admitted Qwen4 requests (greedy, or sampled with
/// neutral penalties when the lane store is sampled) using the exact lane
/// executor. All terminal ownership remains in `ContinuousBatchScheduler`.
/// `promoted` is a running singleton moved into a lane (`generate_spec`): it
/// is adopted as a Running lane before the loop and continues from its
/// pending seed.
pub fn drive_qwen4_lane_batch(
    sched: &mut ContinuousBatchScheduler,
    gpu: &mut Gpu,
    m: &mut LoadedModel,
    params: VmmBatchParams,
    stdout: &mut Stdout,
    inbox: &mut DaemonInbox,
    promoted: Option<Qwen4PromotedRequest>,
) -> Result<(), BatchDriveError> {
    // Held for the whole drive (every return path, error teardown included):
    // covers requests that are queued, being rendered or handed back to the
    // singleton route, which own no lane control yet, plus the promoted
    // adoption below.
    let _serving = m.ple_serving_guard();
    let setup = m
        .qwen4()
        .ok_or("lane model is not qwen4")
        .and_then(|bundle| {
            let store = bundle.lanes().ok_or("qwen4 lanes not staged")?;
            let (max_lanes, row_budget) = (
                store.max_lanes(),
                store.row_budget().min(params.max_batch_tokens),
            );
            if row_budget == 0 || sched.max_batch > max_lanes {
                return Err("qwen4 lane scheduler/budget does not match staged store");
            }
            Ok((max_lanes, row_budget, bundle.config.eos_token_id))
        });
    let (max_lanes, row_budget, eos) = match setup {
        Ok(setup) => setup,
        Err(message) => {
            if let Some(p) = &promoted {
                fail_promoted(
                    m,
                    stdout,
                    &p.pending.key,
                    p.pending.admission,
                    &p.epoch,
                    message,
                );
            }
            return Err(BatchDriveError::Gpu(message.into()));
        }
    };
    let mut controls: Vec<Option<LaneCtl>> = (0..sched.max_batch).map(|_| None).collect();
    let mut cursor = 0;
    let mut trace = StepTrace::new();
    let mut mix = MeshMix::default();
    // Step boundaries since the last >= 512-row prefill or elastic reclaim (the elastic reserve's
    // idle count).
    let mut elastic_idle = ElasticIdle::default();
    // The request the page budget is holding at the queue front, if any.
    let mut page_wait: Option<PageWait> = None;
    // Requests a failed step put back in line (see `lane_step_failed`).
    let mut requeue = RequeueBook::default();
    let batch_size = sched.max_batch;
    // Requests already queued at entry; the promoted lane never passes the
    // inbox, so it is never handed back to the singleton.
    let initial: HashSet<_> = sched.inbox.iter().cloned().collect();
    if let Some(promoted) = promoted {
        adopt_promoted(sched, m, &mut controls, stdout, promoted);
    }
    let result = (|| -> Result<(), String> {
        loop {
            // Terminal decisions must not block other lanes' GPU work.
            for idx in 0..sched.max_batch {
                let decision = match &sched.lanes[idx] {
                    BatchLane::AwaitingClient(t) => {
                        if batch_check_abort(&t.key.id, t.key.attempt_id, t.ticket.admission)
                            || Instant::now() >= t.deadline
                        {
                            Some(false)
                        } else if batch_poll_decision(
                            &t.key.id,
                            t.key.attempt_id,
                            t.ticket.admission,
                        ) == Some(ClientTerminalDecision::Commit)
                        {
                            Some(true)
                        } else {
                            None
                        }
                    }
                    BatchLane::Running(l) | BatchLane::Seeding(l)
                        if batch_check_abort(&l.key.id, l.key.attempt_id, l.ticket.admission) =>
                    {
                        Some(false)
                    }
                    _ => None,
                };
                match decision {
                    Some(false) => cancel_lane(sched, m, &mut controls, idx, stdout)?,
                    Some(true) => {
                        let BatchLane::AwaitingClient(t) = &sched.lanes[idx] else {
                            unreachable!()
                        };
                        let key = t.key.clone();
                        let admission = t.ticket.admission;
                        let done = t.pending_done.clone();
                        let _scope = BatchAttemptScope::enter_for_generation(
                            &key.id,
                            key.attempt_id,
                            admission,
                        );
                        if !sched.commit_lane_retain_terminal(idx, &key, admission) {
                            lane_error(
                                sched,
                                m,
                                &mut controls,
                                idx,
                                stdout,
                                "qwen4 commit_lane failed",
                                "internal",
                            )?;
                        } else {
                            if let Some(ctl) = controls[idx].as_mut() {
                                if let Some((action, tokens)) = ctl.cache.take() {
                                    let tokenizer =
                                        m.tokenizer.as_ref().ok_or("tokenizer disappeared")?;
                                    apply_turn_cache_action(
                                        |fp, seq| {
                                            m.asst_turn_cache.insert(
                                                fp,
                                                crate::qwen::qwen_cached_turn_entry(
                                                    tokenizer,
                                                    seq,
                                                    ctl.started_in_think,
                                                ),
                                            );
                                        },
                                        &action,
                                        tokens,
                                    );
                                }
                            }
                            emit_generation_done_value(ROUTE, stdout, &done);
                            batch_clear_terminal_at_generation(&key.id, key.attempt_id, admission);
                            controls[idx] = None;
                        }
                    }
                    None => {}
                }
            }
            for key in sched.inbox.iter().cloned().collect::<Vec<_>>() {
                if let Some(req) = sched.pending.get(&key) {
                    let admission = req.admission;
                    if batch_check_abort(&key.id, key.attempt_id, admission) {
                        let _scope = BatchAttemptScope::enter_for_generation(
                            &key.id,
                            key.attempt_id,
                            admission,
                        );
                        if !requeue.take_started(&key) {
                            emit_generation_start(ROUTE, stdout, &key.id, false);
                        }
                        emit_generation_cancel(ROUTE, stdout, &key.id, 0);
                        sched.abort_queued(&key, admission);
                    }
                }
            }
            let tokenizer = m.tokenizer.as_ref().ok_or("tokenizer missing")?;
            let mut barrier = drain_qwen_batch_inbox(
                sched,
                m,
                batch_size,
                is_vmm_batch_request_eligible,
                tokenizer,
                m.chat_template.as_ref(),
                ROUTE,
                stdout,
                inbox,
                true,
            )?;
            // Burst coalescing: a tick that already holds a burst (two or more queued requests
            // whose front can take a free lane now), or a lone fresh request where it stays a
            // lane (EP / PP lanes: `coalesce_arms`), waits while arrivals keep coming (each at
            // most `gap` after the last, the whole wait capped at `ADMIT_COALESCE_GAPS` gaps,
            // and only while a free lane is left for one more), so a burst's stragglers
            // prefill in the SAME wave instead of one wave later. A lone request waits one gap
            // at most (a gap with no arrival ends the wait); a queue behind a full lane set and
            // a lone request a single device hands back to the singleton route never wait.
            if let Some(gap) = admit_coalesce_gap().filter(|_| barrier.is_none()) {
                if coalesce_arms(
                    admit_coalesce_first_enabled(),
                    sched.inbox.len(),
                    page_wait.is_some(),
                    || qwen4_lone_request_takes_lane(m),
                ) && sched.assignable_front().is_some()
                {
                    let started = Instant::now();
                    while barrier.is_none() {
                        let room = sched.inbox.len() < sched.empty_lanes().len();
                        let Some(wait) = coalesce_next_wait(gap, started.elapsed(), room) else {
                            break;
                        };
                        match inbox.recv_timeout(wait) {
                            Ok(msg) => inbox.push_front(msg),
                            Err(_) => break,
                        }
                        barrier = drain_qwen_batch_inbox(
                            sched,
                            m,
                            batch_size,
                            is_vmm_batch_request_eligible,
                            tokenizer,
                            m.chat_template.as_ref(),
                            ROUTE,
                            stdout,
                            inbox,
                            true,
                        )?;
                    }
                }
            }
            if let Some(barrier) = barrier {
                inbox.push_front(barrier);
            }
            // The page budget gates the queue front before it takes a lane: a request
            // whose worst-case context growth does not fit the free device pages stays
            // queued, as for a full lane set, and is asked again at the next step
            // boundary (never failed mid-step).
            loop {
                let (pre_rendered, budget_refusal, budget_slot) = match page_budget_gate(
                    sched,
                    m,
                    gpu,
                    &controls,
                    &initial,
                    &requeue,
                    &mut page_wait,
                )? {
                    PageGate::Wait => break,
                    PageGate::Proceed {
                        rendered,
                        refusal,
                        slot,
                    } => (rendered, refusal, slot),
                };
                let Some((key, ticket)) = sched.try_assign_one() else {
                    break;
                };
                let req = sched
                    .pending
                    .get(&key)
                    .cloned()
                    .ok_or("assigned request missing")?;
                let lone = sched.inbox.is_empty()
                    && !controls.iter().flatten().any(|c| c.epoch.is_admitted());
                // The prompt rendered for the session-reuse probe below (or by the
                // page gate), kept so the lane does not render it twice.
                let mut probed = pre_rendered;
                if lone_arrival_hands_back(
                    lone,
                    qwen4_lone_request_takes_lane(m),
                    initial.contains(&key) || requeue.started.contains(&key),
                    || {
                        let rendered = probed.take().unwrap_or_else(|| render_request(m, &req));
                        let reusable = rendered.as_ref().is_ok_and(|(prompt, _)| {
                            crate::qwen::plan_qwen4_session(m, prompt, SessionRoute::Ar) > 0
                        });
                        probed = Some(rendered);
                        reusable
                    },
                ) {
                    inbox.push_front(handoff_started_in_think(
                        sched,
                        ticket.lane,
                        &key,
                        &req,
                        ROUTE,
                    )?);
                    continue;
                }
                let rendered = probed.unwrap_or_else(|| render_request(m, &req));
                let _scope = BatchAttemptScope::enter_for_generation(
                    &key.id,
                    key.attempt_id,
                    ticket.admission,
                );
                let (prompt, started) = match rendered {
                    Ok(r) => r,
                    Err((error, class)) => {
                        lane_error(sched, m, &mut controls, ticket.lane, stdout, &error, class)?;
                        continue;
                    }
                };
                let max_tokens = lane_request_max_tokens(&req, prompt.len(), m.max_seq);
                let required = prompt
                    .len()
                    .checked_add(max_tokens)
                    .and_then(|n| n.checked_add(1));
                let refusal = if req.max_tokens == 0 {
                    Some(("max_tokens must be > 0".to_string(), "validation"))
                } else if prompt.is_empty() {
                    Some((
                        "qwen4 AR prompt rendered to zero tokens".to_string(),
                        "validation",
                    ))
                } else if required.is_none_or(|n| n > m.max_seq) {
                    Some((format!("qwen4 request exceeds context window: prompt={} + max_tokens={} + trailer=1 > max_seq={}", prompt.len(), max_tokens, m.max_seq), "context_length"))
                } else if let Some(message) = budget_refusal {
                    Some((message, "overload"))
                } else {
                    None
                };
                if let Some((message, class)) = refusal {
                    lane_error(
                        sched,
                        m,
                        &mut controls,
                        ticket.lane,
                        stdout,
                        &message,
                        class,
                    )?;
                    continue;
                }
                let lane_rows = reservation_rows(
                    prompt.len(),
                    max_tokens,
                    Owner::of(m, gpu)?.bundle.lane_page_draft_rows(),
                    m.max_seq,
                );
                // vLLM-compatible `min_tokens`: validated against `max_tokens`
                // at the gateway and the daemon, so a malformed value cannot
                // reach a lane; a stray one is refused rather than guessed at.
                let min_tokens = match hipfire_runtime::stop_sequence::parse_min_tokens_field(
                    req.original_msg.get("min_tokens"),
                    req.max_tokens,
                ) {
                    Ok(n) => n,
                    Err(message) => {
                        lane_error(
                            sched,
                            m,
                            &mut controls,
                            ticket.lane,
                            stdout,
                            &message,
                            "validation",
                        )?;
                        continue;
                    }
                };
                let stop =
                    hipfire_runtime::stop_sequence::parse_stop_field(req.original_msg.get("stop"))
                        .map_err(|e| e.to_string())?;
                let epoch = next_lane_epoch()?;
                let admitted = Owner::of(m, gpu)?.admit(epoch, prompt.clone(), budget_slot);
                let slot = match admitted {
                    Ok(slot) => slot,
                    Err(e) => {
                        lane_error(
                            sched,
                            m,
                            &mut controls,
                            ticket.lane,
                            stdout,
                            &e.to_string(),
                            "context_length",
                        )?;
                        continue;
                    }
                };
                // The page budget's reservation: from here the lane's worst-case growth
                // is promised to it (cleared when the lane is released).
                if let Err(error) = Owner::of(m, gpu)?.reserve_pages(&epoch, lane_rows) {
                    release_epoch(m, &epoch)?;
                    lane_error(
                        sched,
                        m,
                        &mut controls,
                        ticket.lane,
                        stdout,
                        &error,
                        "internal",
                    )?;
                    continue;
                }
                if req.sampling.temp > 0.0 {
                    let s = &req.sampling;
                    let sampling = LaneSampling {
                        temperature: s.temp,
                        top_p: s.top_p,
                        top_k: s.top_k,
                        min_p: s.min_p,
                        seed: request_seed_for(&key, req.client_seed),
                        seeded: req.client_seed.is_some(),
                    };
                    if let Err(error) = Owner::of(m, gpu)?.set_sampling(&epoch, sampling) {
                        release_epoch(m, &epoch)?;
                        lane_error(
                            sched,
                            m,
                            &mut controls,
                            ticket.lane,
                            stdout,
                            &error,
                            "validation",
                        )?;
                        continue;
                    }
                }
                if min_tokens > 0 {
                    // The lane's windows run through an EOS; the driver holds EOS
                    // and stops inactive below `min_tokens` (`emit_forwarded`).
                    if let Err(error) = Owner::of(m, gpu)?.set_ignore_eos(&epoch) {
                        release_epoch(m, &epoch)?;
                        lane_error(
                            sched,
                            m,
                            &mut controls,
                            ticket.lane,
                            stdout,
                            &error,
                            "unsupported",
                        )?;
                        continue;
                    }
                }
                // A request put back in line after a mapping refusal already announced its start.
                if !requeue.take_started(&key) {
                    emit_generation_start(ROUTE, stdout, &key.id, started);
                }
                // The singleton tools parser maps an empty array to None.
                let tools = req
                    .original_msg
                    .get("tools")
                    .and_then(|v| v.as_array())
                    .is_some_and(|a| !a.is_empty());
                controls[ticket.lane] = Some(LaneCtl {
                    epoch,
                    slot,
                    semantic: Some(
                        QwenArSemanticProducer::new_with_tool_protocol(&key.id, started, tools)
                            .with_stop(&stop)
                            .with_min_tokens(min_tokens, 0),
                    ),
                    bytes: Vec::new(),
                    budget: ThinkBudget::new(req.max_think_tokens),
                    forced: VecDeque::new(),
                    started_in_think: started,
                    max_tokens,
                    start: Instant::now(),
                    decode_start: None,
                    prefill_ms: 0.0,
                    cache: None,
                    skip_first_forward: false,
                    mapped_ahead: false,
                    _serving: m.ple_serving_guard(),
                });
                let BatchLane::Running(lane) = &mut sched.lanes[ticket.lane] else {
                    return Err("assigned lane is not running".into());
                };
                lane.prompt_len = prompt.len();
                lane.seq_pos = prompt.len();
                lane.conversation_tokens = prompt;
            }
            if sched.active_count() == 0 && sched.inbox.is_empty() {
                return Ok(());
            }
            let streamed: Vec<usize> = sched
                .lanes
                .iter()
                .map(|l| match l {
                    BatchLane::Running(l) | BatchLane::Seeding(l) => l.streamed_tokens.len(),
                    _ => 0,
                })
                .collect();
            let t_plan = Instant::now();
            let PlannedStep {
                plan,
                mesh: mesh_requests,
            } = match plan_step(
                m,
                gpu,
                &mut controls,
                &streamed,
                row_budget,
                &mut cursor,
                max_lanes,
            )? {
                Ok(planned) => planned,
                Err(failure) => {
                    for epoch in &failure.epochs {
                        if let Some(idx) = controls
                            .iter()
                            .position(|c| c.as_ref().is_some_and(|c| c.epoch == *epoch))
                        {
                            lane_step_failed(
                                sched,
                                m,
                                &mut controls,
                                idx,
                                stdout,
                                &format!("qwen4 lane draft failed: {}", failure.error),
                                &mut requeue,
                            )?;
                        }
                    }
                    continue;
                }
            };
            if plan.requests.is_empty() && mesh_requests.is_empty() {
                elastic_boundary(sched, m, gpu, &controls, &mut elastic_idle, None);
                std::thread::sleep(Duration::from_millis(2));
                continue;
            }
            let t_fwd = Instant::now();
            let mut t_commit = t_fwd;
            // The executor's plan first (prefill tiles; every row of a store
            // without the mesh lane owner), then the mesh lanes' decode step.
            // A failure errors only its own lanes: what the other half
            // already committed must still be emitted below.
            let mut ticks: Vec<(RequestAdvance, Option<u32>)> = Vec::new();
            if !plan.requests.is_empty() {
                let stepped = {
                    let mut owner = Owner::of(m, gpu)?;
                    let mut executor = owner.executor()?;
                    let result = executor
                        .provision_step(&plan)
                        .and_then(|()| executor.forward_step(&plan))
                        .and_then(|out| {
                            t_commit = Instant::now();
                            executor.commit_step(&plan, out)
                        });
                    if result.is_err() {
                        executor.abort_step(&plan);
                    }
                    result
                };
                match stepped {
                    Ok(advances) => ticks.extend(advances.into_iter().map(|a| (a, None))),
                    Err(error) => {
                        for request in &plan.requests {
                            if let Some(idx) = controls
                                .iter()
                                .position(|c| c.as_ref().is_some_and(|c| c.epoch == request.epoch))
                            {
                                lane_step_failed(
                                    sched,
                                    m,
                                    &mut controls,
                                    idx,
                                    stdout,
                                    &format!("qwen4 lane step failed: {error}"),
                                    &mut requeue,
                                )?;
                            }
                        }
                    }
                }
            }
            // First tokens before the mesh window: on a step that carries one, the executor
            // half's prefill ticks (a lane's committed first token with its abort / stop / EOS /
            // length finish, `prefill_progress`) go out now rather than after the mesh decode
            // window (14-21 ms) that would hold them. The mesh window carries only lanes that
            // were Decoding when the step was planned, never a lane this step prefilled, so a
            // lane finished here is not in it. Every other tick keeps its place below.
            let mut early_emit = Duration::ZERO;
            if first_token_before_mesh_enabled() && !mesh_requests.is_empty() {
                let t_early = Instant::now();
                let (prefill, rest): (Vec<_>, Vec<_>) = ticks
                    .into_iter()
                    .partition(|(advance, seed)| is_prefill_tick(&plan, advance, *seed));
                ticks = rest;
                for (advance, seed) in prefill {
                    emit_tick(
                        sched,
                        m,
                        gpu,
                        &mut controls,
                        stdout,
                        &plan,
                        eos,
                        advance,
                        seed,
                    )?;
                }
                early_emit = t_early.elapsed();
            }
            let mut mesh_facts = MeshStepFacts::default();
            if !mesh_requests.is_empty() {
                let stepped = {
                    let mut owner = Owner::of(m, gpu)?;
                    owner
                        .bundle
                        .ep_mesh_decode_step(&mut *owner.gpu, &mesh_requests)
                };
                match stepped {
                    Ok((decoded, facts)) => {
                        mesh_facts = facts;
                        mix.note(&mesh_facts);
                        ticks.extend(decoded.into_iter().map(|d| (d.advance, Some(d.seed))));
                    }
                    Err(error) => {
                        for request in &mesh_requests {
                            if let Some(idx) = controls
                                .iter()
                                .position(|c| c.as_ref().is_some_and(|c| c.epoch == request.epoch))
                            {
                                lane_step_failed(
                                    sched,
                                    m,
                                    &mut controls,
                                    idx,
                                    stdout,
                                    &format!("qwen4 lane step failed: {error}"),
                                    &mut requeue,
                                )?;
                            }
                        }
                    }
                }
            }
            let t_done = Instant::now();
            let decoded = trace.is_some().then(|| {
                ticks
                    .iter()
                    .filter(|(a, seed)| {
                        seed.is_some()
                            || plan
                                .requests
                                .iter()
                                .any(|r| r.epoch == a.epoch && r.kind != RequestStepKind::Prefill)
                    })
                    .fold((0, 0), |(t, c), (a, _)| {
                        (t + a.committed_ids.len(), c + a.accepted_drafts)
                    })
            });
            for (advance, decode_seed) in ticks {
                emit_tick(
                    sched,
                    m,
                    gpu,
                    &mut controls,
                    stdout,
                    &plan,
                    eos,
                    advance,
                    decode_seed,
                )?;
            }
            if let (Some(trace), Some((tokens, accepted))) = (trace.as_mut(), decoded) {
                let active = controls
                    .iter()
                    .flatten()
                    .filter(|c| c.epoch.is_admitted())
                    .count();
                trace.line(
                    [t_plan, t_fwd, t_commit, t_done],
                    &plan,
                    &mesh_facts,
                    tokens,
                    accepted,
                    active,
                    sched.inbox.len(),
                    early_emit,
                );
            }
            // The elastic reserve's step boundary (idle count, cache give-back, re-lend).
            let long_prefill = plan.requests.iter().any(|r| {
                r.kind == RequestStepKind::Prefill && long_prefill_route_rows(r.rows.len)
            });
            elastic_boundary(sched, m, gpu, &controls, &mut elastic_idle, Some(long_prefill));
        }
    })();
    mix.report();
    if let Err(error) = result {
        let mut cleanup_error = None;
        for idx in 0..sched.max_batch {
            if let Err(e) = lane_error(sched, m, &mut controls, idx, stdout, &error, "gpu") {
                cleanup_error.get_or_insert(e);
            }
        }
        for (key, req) in &sched.pending {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, req.admission);
            emit_generation_error(
                ROUTE,
                stdout,
                Some(&key.id),
                &error,
                "gpu",
                false,
                cleanup_error.is_none(),
            );
        }
        sched.fail_all_active();
        return Err(match cleanup_error {
            Some(e) => BatchDriveError::Poisoned(format!("{error}; lane release failed: {e}")),
            None => BatchDriveError::Gpu(error),
        });
    }
    Ok(())
}

/// A lane's MTP counters on its done event, under the singleton's keys
/// (`mtp`, `mtp_windows`, `ar_windows`, `mtp_retired`, `tau`, `cycles`) plus
/// the lane-only `mtp_drafts`, `mtp_accepted` and the per-position arrays
/// `mtp_depth_offered` / `mtp_depth_reached` / `mtp_depth_accepted` (index
/// `d - 1` is depth `d`, trimmed after the last non-zero bin;
/// `mtp_depth_accepted[d] / mtp_depth_reached[d]` is the per-position
/// acceptance `a_d` over native windows). `cycles` counts verify windows
/// (native, `ArSynced` and copy) and `tau` is accepted drafts per cycle, the
/// singleton's definitions. Nothing but the lane's own window counters feeds
/// them; a lane that ran no window reports only `mtp: true`.
fn attach_mtp_lane_stats(done: &mut serde_json::Value, stats: &LaneSpecStats) {
    done["mtp"] = serde_json::json!(true);
    if stats.windows == 0 && stats.ar_windows == 0 && stats.copy_windows == 0 && !stats.retired {
        return;
    }
    done["mtp_windows"] = serde_json::json!(stats.windows);
    done["ar_windows"] = serde_json::json!(stats.ar_windows);
    done["mtp_retired"] = serde_json::json!(stats.retired);
    done["mtp_drafts"] = serde_json::json!(stats.drafts);
    done["mtp_accepted"] = serde_json::json!(stats.accepted);
    if stats.copy_windows != 0 {
        done["mtp_copy_windows"] = serde_json::json!(stats.copy_windows);
        done["mtp_copy_drafts"] = serde_json::json!(stats.copy_drafts);
        done["mtp_copy_accepted"] = serde_json::json!(stats.copy_accepted);
    }
    let cycles = stats.windows + stats.copy_windows;
    if cycles != 0 {
        let tau = (stats.accepted + stats.copy_accepted) as f64 / cycles as f64;
        done["cycles"] = serde_json::json!(cycles);
        done["tau"] = serde_json::json!((tau * 100.0).round() / 100.0);
    }
    let bins = stats
        .depth_offered
        .iter()
        .rposition(|&n| n != 0)
        .map_or(0, |last| last + 1);
    if bins != 0 {
        done["mtp_depth_offered"] = serde_json::json!(&stats.depth_offered[..bins]);
        done["mtp_depth_reached"] = serde_json::json!(&stats.depth_reached[..bins]);
        done["mtp_depth_accepted"] = serde_json::json!(&stats.depth_accepted[..bins]);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_mapping_refusal_is_a_retryable_overload_and_other_failures_keep_their_class() {
        let refused = format!(
            "qwen4 lane step failed: {}16301 positions for RequestEpoch {{ request_tag: 40 }}: hipMemCreate out of memory",
            hipfire_arch_qwen4::lane_exec::PROVISION_MAP_REFUSED
        );
        assert_eq!(super::lane_error_wire(&refused, "gpu"), ("overload", true));
        assert_eq!(
            super::lane_error_wire("qwen4 lane step failed: forward failed", "gpu"),
            ("gpu", false)
        );
        assert_eq!(
            super::lane_error_wire("qwen4 lane draft failed: x", "gpu"),
            ("gpu", false)
        );
    }

    /// A lane step that ran out of device memory outside a context map is a retryable overload too.
    #[test]
    fn a_device_out_of_memory_step_is_a_retryable_overload() {
        assert_eq!(
            super::lane_error_wire(
                "qwen4 lane step failed: forward_step: GpuPool: hipMalloc(2088960 B): out of memory",
                "gpu"
            ),
            ("overload", true)
        );
    }

    /// A request goes back in line a bounded number of times, announces its start once, and a request
    /// that was never requeued announces it normally.
    #[test]
    fn a_requeued_request_is_bounded_and_announces_its_start_once() {
        use super::{RequeueBook, MAX_REQUEUES};
        use hipfire_engine::terminal::AttemptKey;
        let mut book = RequeueBook::default();
        let key = AttemptKey::new("r1", 1);
        let other = AttemptKey::new("r2", 2);
        assert!(
            !book.take_started(&key),
            "never requeued: announce the start"
        );
        for _ in 0..MAX_REQUEUES {
            assert!(book.allow(&key));
        }
        assert!(
            !book.allow(&key),
            "out of requeues: the request fails instead"
        );
        assert!(book.allow(&other), "the bound is per request");
        book.started.insert(key.clone());
        assert!(
            book.take_started(&key),
            "requeued: the start was already announced"
        );
        assert!(!book.take_started(&key), "consumed once");
    }

    /// Every failure site of a lane step (draft, executor, mesh) goes through `lane_step_failed`, which
    /// requeues only a mapping refusal of a request that streamed nothing and only with the budget on.
    #[test]
    fn every_step_failure_site_goes_through_the_requeue_rule() {
        let src = include_str!("fn_batch.rs");
        let prod = &src[..src.find("#[cfg(test)]").expect("test module")];
        let drive = prod.find("pub fn drive_qwen4_lane_batch").expect("drive");
        let body = &prod[drive..];
        assert_eq!(body.matches("lane_step_failed(").count(), 3);
        let rule = prod.find("fn lane_step_failed(").expect("rule");
        let rule_src = &prod[rule..(rule + 2400).min(prod.len())];
        assert!(rule_src.contains("page_budget_enabled()"));
        assert!(rule_src.contains("PROVISION_MAP_REFUSED"));
        assert!(rule_src.contains("streamed_tokens.is_empty()"));
    }

    /// A request the page budget can never fit is refused as a retryable overload, and so is
    /// a mid-step mapping refusal; the budget's message does not read as one of the others.
    #[test]
    fn a_page_budget_refusal_is_a_retryable_overload() {
        let refused = format!(
            "{}the request's context (16384 rows) does not fit the free device memory: rank 1: ...",
            hipfire_arch_qwen4::lane::pages::LANE_PAGE_BUDGET_REFUSED
        );
        assert_eq!(
            super::lane_error_wire(&refused, "overload"),
            ("overload", true)
        );
        // Whatever class the caller passed, the marker decides.
        assert_eq!(super::lane_error_wire(&refused, "gpu"), ("overload", true));
    }

    /// The gate queues a request that does not fit while a lane can still change the answer,
    /// admits one that fits, and refuses (never spins on) one that cannot fit with no lane active.
    #[test]
    fn the_gate_queues_while_lanes_run_and_refuses_when_waiting_cannot_help() {
        use super::{gate_decision, GateDecision};
        use hipfire_arch_qwen4::lane::pages::{PageShortfall, PageVerdict, RankPages};
        let short = PageShortfall {
            pages: RankPages {
                rank: 1,
                need: 4 << 30,
                free: 1 << 30,
                ..RankPages::default()
            },
            margin: 256 << 20,
        };
        assert_eq!(gate_decision(PageVerdict::Fit, 0), GateDecision::Admit);
        assert_eq!(gate_decision(PageVerdict::Fit, 8), GateDecision::Admit);
        assert_eq!(
            gate_decision(PageVerdict::Wait(short), 3),
            GateDecision::Queue(short)
        );
        // The store says wait but the scheduler has nothing running: waiting would spin.
        assert_eq!(
            gate_decision(PageVerdict::Wait(short), 0),
            GateDecision::Refuse(short)
        );
        assert_eq!(
            gate_decision(PageVerdict::Never(short), 3),
            GateDecision::Refuse(short)
        );
    }

    /// The gate sits before the lane assignment and the reservation after the admission, and the
    /// kill switch short-circuits the gate before anything is read or rendered.
    #[test]
    fn the_page_gate_precedes_assignment_and_the_kill_switch_precedes_the_gate() {
        let src = include_str!("fn_batch.rs");
        let prod = &src[..src.find("#[cfg(test)]").expect("test module")];
        let drive = prod.find("pub fn drive_qwen4_lane_batch").expect("drive");
        let body = &prod[drive..];
        let gate = body.find("page_budget_gate(").expect("gate call");
        let assign = body.find("sched.try_assign_one()").expect("assignment");
        let admit = body
            .find(".admit(epoch, prompt.clone(), budget_slot)")
            .expect("admission");
        let reserve = body
            .find(".reserve_pages(&epoch, lane_rows)")
            .expect("reservation");
        assert!(gate < assign && assign < admit && admit < reserve);
        let gate_fn = prod.find("fn page_budget_gate(").expect("gate fn");
        let gate_src = &prod[gate_fn..];
        let switch = gate_src
            .find("if !page_budget_enabled()")
            .expect("kill switch");
        let first_read = gate_src.find("sched.assignable_front()").expect("peek");
        assert!(
            switch < first_read,
            "the kill switch is the first thing the gate checks"
        );
    }

    /// The elastic reserve gives its bank back before the page numbers are read: the `admit`
    /// reclaim precedes the first page decision, and a non-Fit verdict reclaims and decides again;
    /// the loop ticks the reserve only after the step's commit.
    #[test]
    fn the_elastic_reclaim_precedes_the_page_decision() {
        let src = include_str!("fn_batch.rs");
        let prod = &src[..src.find("#[cfg(test)]").expect("test module")];
        let gate_fn = prod.find("fn page_budget_gate(").expect("gate fn");
        let gate_src = &prod[gate_fn..];
        let admit = gate_src
            .find("elastic_reclaim(\"admit\")")
            .expect("admit reclaim");
        let first = gate_src
            .find("gate_page_decision(")
            .expect("first page decision");
        let pages = gate_src
            .find("elastic_reclaim(\"pages\")")
            .expect("retry reclaim");
        assert!(
            admit < first && first < pages,
            "reclaim, decide, reclaim again on a non-Fit verdict"
        );
        assert!(
            gate_src[pages..].contains("gate_page_decision("),
            "a retry decision follows the retry reclaim"
        );
        let drive = prod.find("pub fn drive_qwen4_lane_batch").expect("drive");
        let body = &prod[drive..];
        let commit = body.find("executor.commit_step(&plan, out)").expect("commit");
        let tick = body
            .find("elastic_boundary(sched, m, gpu, &controls, &mut elastic_idle, Some(")
            .expect("tick");
        assert!(commit < tick, "the elastic tick follows the step's commit");
    }

    use super::{attach_mtp_lane_stats, forwarded_tokens, LaneCtl, LaneSpecStats};
    use hipfire_runtime::emit_text::ThinkBudget;
    use hipfire_runtime::external_rows::FetchActivity;
    use hipfire_runtime::slot_batch::RequestEpoch;
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    fn lane_ctl(serving: Option<hipfire_runtime::external_rows::ServingGuard>) -> LaneCtl {
        LaneCtl {
            epoch: RequestEpoch {
                request_tag: 1,
                owner_generation: 1,
            },
            slot: 0,
            semantic: None,
            bytes: Vec::new(),
            budget: ThinkBudget::new(0),
            forced: VecDeque::new(),
            started_in_think: false,
            max_tokens: 1,
            start: Instant::now(),
            decode_start: None,
            prefill_ms: 0.0,
            cache: None,
            skip_first_forward: false,
            mapped_ahead: false,
            _serving: serving,
        }
    }

    #[test]
    fn a_lane_turn_ignores_an_eos_id_below_min_tokens_only() {
        use super::turn_ends;
        // Stops active: an EOS id ends the turn; so does a filter / stop match.
        assert!(turn_ends(false, true, true));
        assert!(turn_ends(true, true, false));
        assert!(!turn_ends(false, true, false));
        // Below `min_tokens`: the EOS id is ordinary, a (suspended) stop never
        // reports, and nothing else changes.
        assert!(!turn_ends(false, false, true));
        assert!(!turn_ends(false, false, false));
    }

    #[test]
    fn lane_control_owns_the_serving_guard_for_its_whole_lifespan() {
        let activity = Arc::new(FetchActivity::new());
        let mut controls: Vec<Option<LaneCtl>> = vec![None, None];
        controls[0] = Some(lane_ctl(Some(activity.serving())));
        controls[1] = Some(lane_ctl(Some(activity.serving())));
        assert_eq!(activity.in_flight(), 2);
        // Scheduler ticks (taking a lane's control mutably) never release it.
        let _ = controls[0].as_mut();
        assert_eq!(activity.in_flight(), 2);
        // Commit / cancel / error all end a lane by dropping its control.
        controls[0] = None;
        assert_eq!(activity.in_flight(), 1);
        assert!(!activity.idle_for(Duration::ZERO));
        controls.clear();
        assert_eq!(activity.in_flight(), 0);
        assert!(activity.idle_for(Duration::ZERO));
    }

    #[test]
    fn lane_control_releases_the_serving_guard_on_unwind() {
        let activity = Arc::new(FetchActivity::new());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ctl = lane_ctl(Some(activity.serving()));
            assert_eq!(activity.in_flight(), 1);
            panic!("lane step failed");
        }));
        assert!(result.is_err());
        assert_eq!(activity.in_flight(), 0);
    }

    /// The per-position arrays come from the windows' (offered, accepted)
    /// pairs: windows (3, 3), (3, 1), (2, 0), (1, 1) give offered 4/3/2,
    /// reached 4/2/1 (a window reaches depth d when its first d-1 drafts
    /// matched) and accepted 3/1/1, and the done keys carry the singleton's
    /// `cycles`/`tau` over the same windows.
    #[test]
    fn lane_done_carries_per_position_acceptance_and_cycles() {
        let mut stats = LaneSpecStats::default();
        for (offered, accepted) in [(3usize, 3usize), (3, 1), (2, 0), (1, 1)] {
            stats.windows += 1;
            stats.drafts += offered as u64;
            stats.accepted += accepted as u64;
            stats.record_depths(offered, accepted);
        }
        let mut done = serde_json::json!({});
        attach_mtp_lane_stats(&mut done, &stats);
        assert_eq!(done["mtp"], true);
        assert_eq!(done["mtp_windows"], 4);
        assert_eq!(done["mtp_drafts"], 9);
        assert_eq!(done["mtp_accepted"], 5);
        assert_eq!(done["mtp_depth_offered"], serde_json::json!([4, 3, 2]));
        assert_eq!(done["mtp_depth_reached"], serde_json::json!([4, 2, 1]));
        assert_eq!(done["mtp_depth_accepted"], serde_json::json!([3, 1, 1]));
        assert_eq!(done["cycles"], 4);
        assert_eq!(done["tau"], 1.25);
        assert!(done.get("mtp_copy_windows").is_none());
    }

    /// A lane that ran no window reports only `mtp: true` (no zero arrays, no
    /// tau): the gateway must not fold a zero into its distributions.
    #[test]
    fn lane_without_windows_reports_no_counters() {
        let mut done = serde_json::json!({});
        attach_mtp_lane_stats(&mut done, &LaneSpecStats::default());
        assert_eq!(done, serde_json::json!({ "mtp": true }));
    }

    /// AR-synced windows (no drafts) are cycles but add no depth; a copy
    /// window adds to cycles and tau, not to the native per-position arrays.
    #[test]
    fn ar_and_copy_windows_count_as_cycles_not_depths() {
        let mut stats = LaneSpecStats::default();
        stats.windows = 2; // one native (2 offered, 2 accepted), one ArSynced
        stats.drafts = 2;
        stats.accepted = 2;
        stats.record_depths(2, 2);
        stats.copy_windows = 1;
        stats.copy_drafts = 4;
        stats.copy_accepted = 1;
        let mut done = serde_json::json!({});
        attach_mtp_lane_stats(&mut done, &stats);
        assert_eq!(done["cycles"], 3);
        assert_eq!(done["tau"], 1.0);
        assert_eq!(done["mtp_depth_offered"], serde_json::json!([1, 1]));
        assert_eq!(done["mtp_depth_accepted"], serde_json::json!([1, 1]));
        assert_eq!(done["mtp_copy_accepted"], 1);
    }

    /// Source-level: every lane control carries the guard, and the driver
    /// takes its own before any setup, render, handoff or teardown path.
    #[test]
    fn every_lane_control_and_the_drive_take_a_serving_guard() {
        let src = include_str!("fn_batch.rs");
        let prod = &src[..src.find("#[cfg(test)]").expect("test module")];
        assert_eq!(
            prod.matches("LaneCtl {").count(),
            3,
            "decl + admission + promotion"
        );
        assert_eq!(
            prod.matches("_serving: ").count(),
            3,
            "decl + admission + promotion"
        );
        assert_eq!(prod.matches("_serving: m.ple_serving_guard()").count(), 1);
        assert_eq!(
            prod.matches("_serving: model.ple_serving_guard()").count(),
            1
        );
        let drive = prod.find("pub fn drive_qwen4_lane_batch").expect("drive");
        let guard = prod[drive..]
            .find("let _serving = m.ple_serving_guard();")
            .expect("drive guard");
        let setup = prod[drive..].find("let setup =").expect("setup");
        assert!(
            guard < setup,
            "the drive guard must precede setup and promoted adoption"
        );
    }

    #[test]
    fn forwarded_tokens_are_seed_then_all_committed_but_last() {
        assert_eq!(forwarded_tokens(7, &[8, 9, 10], false), vec![7, 8, 9]);
        // An AR row commits one id: only the plan seed was forwarded.
        assert_eq!(forwarded_tokens(7, &[8], false), vec![7]);
    }

    #[test]
    fn promoted_first_advance_drops_the_streamed_plan_seed() {
        assert_eq!(forwarded_tokens(7, &[8, 9, 10], true), vec![8, 9]);
        assert!(forwarded_tokens(7, &[8], true).is_empty());
        assert!(forwarded_tokens(7, &[], true).is_empty());
    }

    #[test]
    fn early_first_token_is_on_unless_exact_zero() {
        use super::early_first_token_from;
        for on in [None, Some(""), Some("1"), Some("on"), Some(" 1 ")] {
            assert!(early_first_token_from(on), "{on:?}");
        }
        for off in [Some("0"), Some(" 0\n")] {
            assert!(!early_first_token_from(off), "{off:?}");
        }
    }

    #[test]
    fn first_token_before_mesh_is_on_unless_exact_zero_and_needs_the_early_first_token() {
        use super::first_token_before_mesh_from;
        for on in [None, Some(""), Some("1"), Some("on"), Some(" 1 ")] {
            assert!(first_token_before_mesh_from(true, on), "{on:?}");
        }
        for off in [Some("0"), Some(" 0\n")] {
            assert!(!first_token_before_mesh_from(true, off), "{off:?}");
        }
        // With the early first token off there is no first token to move: today's order.
        for any in [None, Some("1"), Some("0")] {
            assert!(!first_token_before_mesh_from(false, any), "{any:?}");
        }
    }

    #[test]
    fn only_the_executors_prefill_rows_are_prefill_ticks() {
        use super::is_prefill_tick;
        use hipfire_runtime::slot_batch::{
            BatchStepPlan, RequestAdvance, RequestRows, RequestStepKind, RowRange,
        };
        let epoch = |tag| RequestEpoch {
            request_tag: tag,
            owner_generation: 1,
        };
        let rows = |tag, begin, kind| RequestRows {
            epoch: epoch(tag),
            rows: RowRange { begin, len: 1 },
            kind,
        };
        let advance = |tag| RequestAdvance {
            epoch: epoch(tag),
            committed_ids: vec![5],
            committed_position: 1,
            accepted_drafts: 0,
            verified_rows: 0,
            finish: None,
        };
        let plan = BatchStepPlan {
            requests: vec![
                rows(1, 0, RequestStepKind::Prefill),
                rows(2, 1, RequestStepKind::Ar),
                rows(3, 2, RequestStepKind::Verify { draft_len: 1 }),
            ],
            ..Default::default()
        };
        assert!(is_prefill_tick(&plan, &advance(1), None));
        assert!(!is_prefill_tick(&plan, &advance(2), None));
        assert!(!is_prefill_tick(&plan, &advance(3), None));
        // A mesh window carries its seed and is never in the executor's plan.
        assert!(!is_prefill_tick(&plan, &advance(1), Some(9)));
        assert!(!is_prefill_tick(&plan, &advance(4), None));
    }

    /// The order is the point: the executor's prefill ticks leave between the executor's
    /// commit and the mesh decode window, behind the kill switch and only on a step that
    /// carries a mesh window; every other tick keeps the single emission loop after it. The
    /// mesh window's requests derive from lanes that were already Decoding at plan time,
    /// so a lane a step prefilled (and may finish by its first token) is never in it.
    #[test]
    fn prefill_ticks_are_emitted_before_the_mesh_decode_step() {
        let src = include_str!("fn_batch.rs");
        let prod = &src[..src.find("#[cfg(test)]").expect("test module")];
        let drive = prod.find("pub fn drive_qwen4_lane_batch").expect("drive");
        let body = &prod[drive..];
        let commit = body
            .find("executor.commit_step(&plan, out)")
            .expect("executor commit");
        let early = body
            .find("if first_token_before_mesh_enabled() && !mesh_requests.is_empty()")
            .expect("before-mesh emission");
        let mesh = body.find("ep_mesh_decode_step").expect("mesh half");
        let rest = body
            .find("for (advance, decode_seed) in ticks")
            .expect("remaining emission");
        assert!(
            commit < early && early < mesh && mesh < rest,
            "commit, prefill ticks, mesh window, then every other tick"
        );
        let block = &body[early..mesh];
        assert!(block.contains("is_prefill_tick(&plan, advance, *seed)"));
        assert!(block.contains("emit_tick("));
        // Both positions share one per-tick emission; the loop body is not duplicated.
        assert_eq!(prod.matches("fn emit_tick(").count(), 1);
        assert_eq!(body.matches("emit_tick(").count(), 2);
        assert_eq!(
            prod.matches("emit_forwarded(").count(),
            3,
            "decl + 2 in emit_tick"
        );
        // The emit time spent ahead of the mesh window is reported under `emit`.
        assert!(body.contains("early_emit = t_early.elapsed();"));
        assert!(body[body.find("trace.line(").expect("trace")..].contains("early_emit,"));
        // plan_step: mesh requests come from `chosen`, which come from Decoding lanes;
        // prefill rows come from Prefilling lanes only.
        let plan_fn = prod.find("fn plan_step(").expect("plan_step");
        let plan_src = &prod[plan_fn..drive];
        let decoding = plan_src.find("let decoding: Vec<_>").expect("decoding set");
        assert!(plan_src[decoding..].contains("l.host.status == LaneStatus::Decoding"));
        assert!(plan_src.contains("let chosen: Vec<_> = (0..count.min(row_budget))"));
        assert!(plan_src.contains("decoding[(*cursor + j) % count]"));
        let mesh_build = plan_src
            .find("mesh_requests = controls")
            .expect("mesh requests");
        assert!(plan_src[mesh_build..].contains("chosen.contains(&c.slot)"));
        assert!(plan_src.contains("LaneStatus::Prefilling => rows.push(("));
    }

    #[test]
    fn admit_coalesce_gap_defaults_to_2ms_and_zero_kills_it() {
        use super::{admit_coalesce_from, coalesce_next_wait};
        use std::time::Duration;
        let ms = Duration::from_millis;
        assert_eq!(admit_coalesce_from(None), Some(ms(2)));
        assert_eq!(admit_coalesce_from(Some("")), Some(ms(2)));
        assert_eq!(admit_coalesce_from(Some("x")), Some(ms(2)));
        assert_eq!(admit_coalesce_from(Some(" 5 ")), Some(ms(5)));
        assert_eq!(admit_coalesce_from(Some("500")), Some(ms(20)));
        assert_eq!(admit_coalesce_from(Some("0")), None);
        // A lone request's whole wait is its first gap (the timeout is the gap itself, at the
        // default and at the clamp), and a gap that brings no arrival ends it.
        for gap in [1, 2, 5, 20].map(ms) {
            assert_eq!(coalesce_next_wait(gap, Duration::ZERO, true), Some(gap));
        }
    }

    #[test]
    fn admit_coalesce_first_is_on_unless_exact_zero() {
        use super::admit_coalesce_first_from;
        for on in [None, Some(""), Some("1"), Some("on"), Some(" 1 ")] {
            assert!(admit_coalesce_first_from(on), "{on:?}");
        }
        for off in [Some("0"), Some(" 0\n")] {
            assert!(!admit_coalesce_first_from(off), "{off:?}");
        }
    }

    #[test]
    fn the_wait_budget_is_five_gaps_with_a_free_lane_left() {
        use super::{coalesce_next_wait, ADMIT_COALESCE_GAPS};
        let gap = Duration::from_millis(2);
        // One gap at a time, the last one trimmed to the budget, none past it.
        assert_eq!(
            coalesce_next_wait(gap, Duration::from_millis(9), true),
            Some(Duration::from_millis(1))
        );
        assert_eq!(
            coalesce_next_wait(gap, gap * ADMIT_COALESCE_GAPS, true),
            None
        );
        assert_eq!(
            coalesce_next_wait(gap, gap * (ADMIT_COALESCE_GAPS + 3), true),
            None
        );
        // No free lane left for one more: no wait at any point.
        assert_eq!(coalesce_next_wait(gap, Duration::ZERO, false), None);
    }

    /// The drive's wait is the pure rules plus the inbox: armed by `coalesce_arms` (the front
    /// assignable, a gap configured, no barrier), one `recv_timeout` per gap, drained after
    /// each arrival, ended by the first gap with none. A lone request is only asked about its
    /// topology after the switch and the page hold.
    #[test]
    fn the_drive_waits_with_the_arming_and_timeout_rules() {
        let src = include_str!("fn_batch.rs");
        let prod = &src[..src.find("#[cfg(test)]").expect("test module")];
        let drive = prod.find("pub fn drive_qwen4_lane_batch").expect("drive");
        let body = &prod[drive..];
        let at = body
            .find("if let Some(gap) = admit_coalesce_gap()")
            .expect("coalescing block");
        let end = at
            + body[at..]
                .find("if let Some(barrier) = barrier")
                .expect("block end");
        let block = &body[at..end];
        assert!(block.contains(".filter(|_| barrier.is_none())"));
        let arms = block.find("coalesce_arms(").expect("arming rule");
        let rest = &block[arms..];
        let first = rest.find("admit_coalesce_first_enabled()").expect("switch");
        let held = rest.find("page_wait.is_some()").expect("page hold");
        let topology = rest
            .find("|| qwen4_lone_request_takes_lane(m)")
            .expect("lone topology");
        let front = rest
            .find("sched.assignable_front().is_some()")
            .expect("assignable front");
        assert!(first < held && held < topology && topology < front);
        assert!(rest.contains("coalesce_next_wait(gap, started.elapsed(), room)"));
        let recv = rest.find("inbox.recv_timeout(wait)").expect("one gap");
        let none = rest.find("Err(_) => break").expect("empty gap ends");
        let drain = rest.find("drain_qwen_batch_inbox(").expect("drain");
        assert!(recv < none && none < drain);
        assert_eq!(block.matches("recv_timeout(").count(), 1);
    }

    #[test]
    fn an_early_first_token_is_skipped_as_the_next_plan_seed() {
        // The prefill step streams the pick and sets `skip_first_forward`; the source keeps
        // that order (flag before the emit) and only for a running lane not already skipping.
        let src = include_str!("fn_batch.rs");
        let prod = &src[..src.find("#[cfg(test)]").expect("test module")];
        let at = prod
            .find("let early = early_first_token_enabled()")
            .expect("early emit");
        let body = &prod[at..];
        let flag = body
            .find("ctl.skip_first_forward = true;")
            .expect("skip flag");
        let emit = body
            .find("advance.committed_ids[0]")
            .expect("emit the pick");
        assert!(flag < emit);
        assert!(body[..flag].contains("!c.skip_first_forward"));
    }

    #[test]
    fn mesh_mix_sums_windows_by_the_lanes_a_step_carried() {
        use super::{MeshMix, MeshStepFacts};
        let mut mix = MeshMix::default();
        let facts = |lanes, native, copy, ar| MeshStepFacts {
            lanes,
            native_windows: native,
            copy_windows: copy,
            ar_windows: ar,
            offered: native * 3,
            accepted: native,
            emitted: native * 2 + copy + ar,
            native_mode: ar == 0,
        };
        mix.note(&facts(2, 2, 0, 0));
        mix.note(&facts(2, 1, 1, 0));
        mix.note(&facts(8, 3, 2, 3));
        // A step with no lane (never produced) is not a bucket.
        mix.note(&MeshStepFacts::default());
        assert_eq!(mix.by_lanes.len(), 2);
        assert_eq!(mix.steps[&2], 2);
        assert_eq!(mix.steps[&8], 1);
        let two = mix.by_lanes[&2];
        assert_eq!(
            (
                two.native_windows,
                two.copy_windows,
                two.ar_windows,
                two.offered,
                two.accepted
            ),
            (3, 1, 0, 9, 3)
        );
    }

    /// The mesh half of a tick must never swallow the executor half's
    /// committed tokens (and vice versa): both halves' failures error only
    /// their own lanes and the advances of the other still flow to emission.
    #[test]
    fn a_tick_emits_both_halves_even_when_one_errors() {
        let src = include_str!("fn_batch.rs");
        let prod = &src[..src.find("#[cfg(test)]").expect("test module")];
        let drive = prod.find("pub fn drive_qwen4_lane_batch").expect("drive");
        let body = &prod[drive..];
        let exec = body
            .find("executor.commit_step(&plan, out)")
            .expect("executor half");
        let mesh = body.find("ep_mesh_decode_step").expect("mesh half");
        let emit = body
            .find("for (advance, decode_seed) in ticks")
            .expect("emission");
        assert!(
            exec < mesh && mesh < emit,
            "executor half, mesh half, then one emission loop"
        );
        let between = &body[exec..emit];
        assert!(
            !between.contains("continue;"),
            "a failed half must not skip the other half's emission"
        );
    }

    #[test]
    fn lone_arrival_hand_back_policy_by_topology() {
        use super::lone_arrival_hands_back;
        // Single device: only a post-entry arrival hands back, and the session
        // probe is never consulted (promotion covers the entry cohort).
        for (lone, initial, want) in [
            (true, false, true),
            (true, true, false),
            (false, false, false),
            (false, true, false),
        ] {
            assert_eq!(
                lone_arrival_hands_back(lone, false, initial, || panic!("single never probes")),
                want
            );
        }
        // EP/PP: a lone request stays a lane, entry cohort or not, unless the
        // singleton would restore a session; with peers it never probes.
        for initial in [false, true] {
            assert!(!lone_arrival_hands_back(true, true, initial, || false));
            assert!(lone_arrival_hands_back(true, true, initial, || true));
            assert!(!lone_arrival_hands_back(false, true, initial, || {
                panic!("peers present: no probe")
            }));
        }
    }

    /// The lone-request wait follows the hand-back policy: it is armed exactly where a lone
    /// request stays a lane (EP / PP lanes), never where the singleton route takes it (single
    /// device, PP with a speculator: a post-entry lone arrival is handed back, so a wait would
    /// only delay its singleton start). On a lane topology a session-reuse hand-back still
    /// waited its one gap: the probe runs at assignment, after the wait.
    #[test]
    fn a_lone_request_waits_only_where_it_stays_a_lane_and_the_switch_is_on() {
        use super::{coalesce_arms, lone_arrival_hands_back};
        // A burst arms whatever the switch and topology, without asking the topology.
        for first in [true, false] {
            for queued in [2usize, 3, 8] {
                assert!(coalesce_arms(first, queued, false, || {
                    panic!("burst: no topology probe")
                }));
            }
            assert!(!coalesce_arms(first, 0, false, || panic!("empty queue")));
        }
        // A lone fresh request: on where it stays a lane, off with the kill switch (today's
        // `>= 2` rule), never when the page budget already holds it.
        assert!(coalesce_arms(true, 1, false, || true));
        assert!(!coalesce_arms(true, 1, false, || false));
        assert!(!coalesce_arms(false, 1, false, || {
            panic!("kill switch: no topology probe")
        }));
        assert!(!coalesce_arms(true, 1, true, || {
            panic!("page-held: no topology probe")
        }));
        for (topology, speculator, armed) in [
            ("single", false, false),
            ("single", true, false),
            ("pp", true, false),
            ("pp", false, true),
            ("ep", false, true),
            ("ep", true, true),
        ] {
            let lone_lane = crate::batch::qwen4_lone_lane(topology, speculator);
            assert_eq!(
                coalesce_arms(true, 1, false, || lone_lane),
                armed,
                "{topology} speculator={speculator}"
            );
            if !armed {
                assert!(
                    lone_arrival_hands_back(true, lone_lane, false, || panic!("no probe")),
                    "{topology} speculator={speculator}: a lone arrival without a wait takes the singleton route"
                );
            }
        }
    }

    /// Scripted lane model: a lane's next id is a pure function of its own
    /// prompt and position (greedy), never of the cohort it runs in.
    fn scripted_next(prompt: &[u32], generated: usize) -> u32 {
        let h = prompt.iter().fold(0x9e37_79b9u32, |h, &t| {
            h.rotate_left(5) ^ t.wrapping_mul(0x85eb_ca6b)
        });
        h.wrapping_add((generated as u32).wrapping_mul(2_654_435_761)) % 50_000
    }

    /// The coalescing configuration a simulated driver runs with.
    #[derive(Clone, Copy)]
    struct Coalesce {
        /// `HIPFIRE_LANE_ADMIT_COALESCE_MS` (`0`: the whole wait is off).
        gap_ms: u64,
        /// `HIPFIRE_LANE_ADMIT_COALESCE_FIRST`.
        first: bool,
    }

    const COALESCE_DEFAULT: Coalesce = Coalesce {
        gap_ms: 2,
        first: true,
    };

    /// [`simulate_burst_at`] for `n` requests of which the last `late` land during the first
    /// lane step (50 ms in, long past the coalescing budget) and the rest are in the inbox when
    /// the driver first drains.
    fn simulate_burst(
        topology: &str,
        n: usize,
        late: usize,
        steps: usize,
    ) -> Option<(Vec<usize>, Vec<Vec<u32>>)> {
        let arrivals: Vec<u64> = (0..n).map(|i| if i < n - late { 0 } else { 50 }).collect();
        simulate_burst_at(topology, COALESCE_DEFAULT, &arrivals, steps)
            .map(|(active, ids, _)| (active, ids))
    }

    /// Daemon + driver admission for requests that reach an idle server, as
    /// `daemon dispatch -> enqueue -> drive_qwen4_lane_batch` does it: the
    /// daemon sees the FIRST request with no peer visible yet; request `i` reaches the
    /// driver `arrivals_ms[i]` ms after its first drain (`0`: already in the inbox then;
    /// non-decreasing). The first tick runs the driver's coalescing wait with its own arming
    /// rule and timeouts (`coalesce_arms`, `coalesce_next_wait`); whatever lands after it joins
    /// at the next step. Returns the per-step active lane counts (None when the first request
    /// would have run as a singleton), every lane's scripted ids and the first tick's wait in
    /// ms.
    fn simulate_burst_at(
        topology: &str,
        coalesce: Coalesce,
        arrivals_ms: &[u64],
        steps: usize,
    ) -> Option<(Vec<usize>, Vec<Vec<u32>>, u64)> {
        // The terminal registry is process-global and other lib tests clear it
        // wholesale: hold the crate's generation test lock for the whole run.
        let _registry = crate::ar::generation_test_lock();
        use hipfire_engine::scheduler::{
            BatchPendingRequest, BatchSampling, ContinuousBatchScheduler,
        };
        use hipfire_engine::terminal::{
            batch_announce_terminal, batch_clear_terminal, batch_terminal_generation, AttemptKey,
        };
        let n = arrivals_ms.len();
        assert!(n >= 1 && arrivals_ms[0] == 0 && arrivals_ms.windows(2).all(|w| w[0] <= w[1]));
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(7000);
        let base = NEXT.fetch_add(n as u64, std::sync::atomic::Ordering::SeqCst);
        let make = |i: usize| {
            let attempt = base + i as u64;
            let id = format!("burst-{attempt}");
            let admission = batch_announce_terminal(&id, attempt).expect("admission");
            let key = AttemptKey::new(&id, attempt);
            BatchPendingRequest {
                admission,
                key: key.clone(),
                original_msg: serde_json::json!({"type": "generate", "id": id}),
                prompt: format!("p{i}"),
                prompt_tokens: vec![i as u32 + 1, 7, 11],
                started_in_think: false,
                system: None,
                assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                max_think_tokens: 0,
                max_tokens: steps,
                client_seed: None,
                sampling: BatchSampling {
                    temp: 1.0,
                    top_p: 0.95,
                    top_k: Some(20),
                    min_p: None,
                    repeat_penalty: 1.0,
                    presence_penalty: 0.0,
                    frequency_penalty: 0.0,
                    repeat_window: 128,
                },
            }
        };
        // Daemon: `batch_request_eligible_for_route` on the VMM route, first
        // request, no other batched generate visible.
        let peer_pending = false;
        let lone_lane = crate::batch::qwen4_lone_lane(topology, false);
        if !(lone_lane || peer_pending) {
            for i in 0..n {
                batch_clear_terminal(&format!("burst-{}", base + i as u64), base + i as u64);
            }
            return None;
        }
        let mut sched = ContinuousBatchScheduler::new(8, 4096);
        let requests: Vec<_> = (0..n).map(make).collect();
        let keys: Vec<_> = requests.iter().map(|r| r.key.clone()).collect();
        assert!(sched.enqueue(requests[0].clone()));
        let initial: std::collections::HashSet<_> = sched.inbox.iter().cloned().collect();
        let mut arrived = 1;
        let mut admitted = vec![false; n];
        let mut lanes: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut ids = vec![Vec::new(); n];
        let mut active_per_step = Vec::new();
        let mut waited_ms = 0u64;
        for step in 0..steps {
            // Drain: what is already there at the first drain, then everything that landed
            // since (the late ones show up after the first step).
            let due = if step == 0 {
                arrivals_ms.iter().take_while(|&&at| at == 0).count()
            } else {
                n
            };
            while arrived < due {
                assert!(sched.enqueue(requests[arrived].clone()));
                arrived += 1;
            }
            // The driver's coalescing wait (first tick only: later ticks have nothing queued).
            if step == 0
                && coalesce.gap_ms > 0
                && super::coalesce_arms(coalesce.first, sched.inbox.len(), false, || lone_lane)
                && sched.assignable_front().is_some()
            {
                let gap = Duration::from_millis(coalesce.gap_ms);
                loop {
                    let room = sched.inbox.len() < sched.empty_lanes().len();
                    let Some(wait) =
                        super::coalesce_next_wait(gap, Duration::from_millis(waited_ms), room)
                    else {
                        break;
                    };
                    let wait = wait.as_millis() as u64;
                    match arrivals_ms.get(arrived) {
                        Some(&at) if at - waited_ms <= wait => {
                            waited_ms = at;
                            assert!(sched.enqueue(requests[arrived].clone()));
                            arrived += 1;
                        }
                        _ => {
                            waited_ms += wait;
                            break;
                        }
                    }
                }
            }
            while let Some((key, ticket)) = sched.try_assign_one() {
                let i = keys.iter().position(|k| *k == key).unwrap();
                let lone = sched.inbox.is_empty() && !admitted.iter().any(|&a| a);
                assert!(
                    !super::lone_arrival_hands_back(
                        lone,
                        lone_lane,
                        initial.contains(&key),
                        || false
                    ),
                    "request {i} must not go back to a singleton"
                );
                admitted[i] = true;
                lanes.push((i, requests[i].prompt_tokens.clone()));
                let _ = ticket;
            }
            active_per_step.push(sched.active_count());
            for (i, history) in &mut lanes {
                let next = scripted_next(&requests[*i].prompt_tokens, ids[*i].len());
                ids[*i].push(next);
                history.push(next);
            }
        }
        for (i, r) in requests.iter().enumerate() {
            batch_clear_terminal(&r.key.id, base + i as u64);
        }
        Some((active_per_step, ids, waited_ms))
    }

    #[test]
    fn simultaneous_arrivals_on_ep_and_pp_decode_as_lanes_from_the_first_step() {
        for topology in ["ep", "pp"] {
            for n in [1usize, 4, 8] {
                // Simultaneous arrivals are lanes from step 0 with the lone-first wait on or
                // killed, and with the whole wait off: nothing to join, same ids.
                for coalesce in [
                    COALESCE_DEFAULT,
                    Coalesce {
                        first: false,
                        ..COALESCE_DEFAULT
                    },
                    Coalesce {
                        gap_ms: 0,
                        ..COALESCE_DEFAULT
                    },
                ] {
                    let (active, ids, _) = simulate_burst_at(topology, coalesce, &vec![0; n], 6)
                        .expect("EP/PP lone request takes a lane");
                    assert_eq!(active, vec![n; 6], "{topology} C{n}: active from step 0");
                    for (i, lane_ids) in ids.iter().enumerate() {
                        let solo = (0..6)
                            .map(|g| scripted_next(&[i as u32 + 1, 7, 11], g))
                            .collect::<Vec<_>>();
                        assert_eq!(lane_ids, &solo, "{topology} C{n} lane {i}: ids unchanged");
                    }
                }
            }
        }
    }

    #[test]
    fn late_arrivals_join_the_lone_ep_lane_at_the_next_step() {
        // The daemon dispatched request 0 alone; 7 more land during its first
        // step (past the coalescing wait): they are lanes at step 1 (active 8), not a
        // queue behind a singleton, and every lane's ids are the solo ids.
        for first in [true, false] {
            let arrivals = [0, 50, 50, 50, 50, 50, 50, 50];
            let coalesce = Coalesce {
                first,
                ..COALESCE_DEFAULT
            };
            let (active, ids, _) = simulate_burst_at("ep", coalesce, &arrivals, 6).expect("lane");
            assert_eq!(active, vec![1, 8, 8, 8, 8, 8], "first={first}");
            // A late lane starts at its own position 0 at the step it joins.
            for (i, lane_ids) in ids.iter().enumerate() {
                let expect = if i == 0 { 6 } else { 5 };
                assert_eq!(lane_ids.len(), expect, "lane {i}");
                let solo = (0..lane_ids.len())
                    .map(|g| scripted_next(&[i as u32 + 1, 7, 11], g))
                    .collect::<Vec<_>>();
                assert_eq!(lane_ids, &solo);
            }
        }
        assert_eq!(
            simulate_burst("ep", 8, 7, 6).expect("lane").0,
            vec![1, 8, 8, 8, 8, 8]
        );
    }

    #[test]
    fn a_lone_first_request_waits_exactly_one_gap_and_the_kill_switch_restores_no_wait() {
        for topology in ["ep", "pp"] {
            let run = |coalesce| {
                simulate_burst_at(topology, coalesce, &[0], 3).expect("EP/PP lone request lane")
            };
            // A lone request with no second arrival: one gap, not more (<= 2 ms by default).
            let (active, _, waited) = run(COALESCE_DEFAULT);
            assert_eq!((active, waited), (vec![1, 1, 1], 2), "{topology}");
            let (_, _, waited) = run(Coalesce {
                gap_ms: 20,
                ..COALESCE_DEFAULT
            });
            assert_eq!(waited, 20, "{topology}: one gap at the clamp");
            // `HIPFIRE_LANE_ADMIT_COALESCE_FIRST=0`: today's `>= 2` rule, no wait for a lone one.
            let (_, _, waited) = run(Coalesce {
                first: false,
                ..COALESCE_DEFAULT
            });
            assert_eq!(waited, 0, "{topology}: kill switch");
            // `HIPFIRE_LANE_ADMIT_COALESCE_MS=0`: the whole wait is off.
            let (_, _, waited) = run(Coalesce {
                gap_ms: 0,
                ..COALESCE_DEFAULT
            });
            assert_eq!(waited, 0, "{topology}: gap off");
        }
    }

    #[test]
    fn an_arrival_inside_the_first_gap_joins_the_first_step() {
        for topology in ["ep", "pp"] {
            let at = |arrivals: &[u64], coalesce| {
                simulate_burst_at(topology, coalesce, arrivals, 5).expect("lane")
            };
            let off = Coalesce {
                first: false,
                ..COALESCE_DEFAULT
            };
            // Two arrivals 1 ms after the first: the lone first request waits for them (each
            // re-arms the gap; one empty gap ends it) and all three prefill in step 0. With the
            // kill switch the first request runs alone and they join one step later.
            let (active, ids, waited) = at(&[0, 1, 1], COALESCE_DEFAULT);
            assert_eq!((active, waited), (vec![3; 5], 3), "{topology}");
            let solo = |i: usize, len: usize| {
                (0..len)
                    .map(|g| scripted_next(&[i as u32 + 1, 7, 11], g))
                    .collect::<Vec<_>>()
            };
            for (i, lane_ids) in ids.iter().enumerate() {
                assert_eq!(lane_ids, &solo(i, 5), "{topology} lane {i}");
            }
            let (active, ids, waited) = at(&[0, 1, 1], off);
            assert_eq!((active, waited), (vec![1, 3, 3, 3, 3], 0), "{topology}");
            for (i, lane_ids) in ids.iter().enumerate() {
                let len = if i == 0 { 5 } else { 4 };
                assert_eq!(lane_ids, &solo(i, len), "{topology} lane {i}");
            }
            // A burst (two queued at the first drain) waits under either switch.
            for coalesce in [COALESCE_DEFAULT, off] {
                let (active, _, waited) = at(&[0, 0, 1], coalesce);
                assert_eq!((active, waited), (vec![3; 5], 3), "{topology}");
            }
            // An arrival past the gap is one step later; it never extends the wait.
            let (active, _, waited) = at(&[0, 3], COALESCE_DEFAULT);
            assert_eq!((active, waited), (vec![1, 2, 2, 2, 2], 2), "{topology}");
        }
    }

    #[test]
    fn the_first_wait_is_capped_at_five_gaps_and_by_free_lanes() {
        // Arrivals every 2 ms keep re-arming the 2 ms gap, but the tick's whole wait stops at
        // 5 gaps (10 ms): the sixth arrival (12 ms) joins one step later.
        let (active, _, waited) =
            simulate_burst_at("ep", COALESCE_DEFAULT, &[0, 2, 4, 6, 8, 10, 12], 4).expect("lane");
        assert_eq!((active, waited), (vec![6, 7, 7, 7], 10));
        // Seven arrivals 1 ms after the first fill all 8 lanes: no gap is spent past the
        // last free lane.
        let (active, _, waited) =
            simulate_burst_at("ep", COALESCE_DEFAULT, &[0, 1, 1, 1, 1, 1, 1, 1], 3).expect("lane");
        assert_eq!((active, waited), (vec![8, 8, 8], 1));
    }

    #[test]
    fn single_device_lone_request_keeps_the_singleton_route() {
        // Promotion exists on a single device: the lone request stays a
        // singleton (unchanged), as does a PP server with a speculator; an EP
        // server with a speculator admits it as a lane.
        assert!(simulate_burst("single", 8, 0, 3).is_none());
        assert!(crate::batch::qwen4_lone_lane("ep", true));
        assert!(!crate::batch::qwen4_lone_lane("pp", true));
        // Neither singleton-route topology waits for a lone request, with the new switch on.
        for (topology, speculator) in [("single", false), ("single", true), ("pp", true)] {
            let lone_lane = crate::batch::qwen4_lone_lane(topology, speculator);
            assert!(!lone_lane, "{topology} speculator={speculator}");
            assert!(
                !super::coalesce_arms(true, 1, false, || lone_lane),
                "{topology} speculator={speculator}"
            );
        }
    }
}
