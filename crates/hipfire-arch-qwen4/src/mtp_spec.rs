// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native Qwen4 MTP acceptance and transaction-shape helpers.
//!
//! The runtime owns the speculative loop and its pending-seed contract.  This
//! module records the native MTP count convention and lowers a verified
//! window (greedy prefix match, or with `speculation.mtp_sampled` a sampled
//! verify from [`hipfire_runtime::spec_sampling`]: speculative rejection
//! sampling, or SpecInfer naive sampling under
//! `HIPFIRE_MTP_SAMPLED_MODE=naive`) onto the canonical runtime types.  In
//! particular, the seed is never copied into `MtpWindow::committed` or
//! `SpecStep::emit`.
//! Target rollback counts accepted drafts only; the position helpers take the
//! consumed-row count, which adds the seed.

use crate::bundle::{penalty_prepass_enabled, BundleError, Qwen4Bundle, Qwen4PpWindow};
use crate::gpu_forward::{Qwen4OutputRows, QWEN4_SPEC_VERIFY_ROWS};
use crate::mtp_gpu::{
    mtp_append_det_enabled, mtp_append_det_requested, prompt_fill_uses_rows, MtpAppendScratch,
    MtpFillContext, MtpGpuStateSnapshot, MTP_FILL_ROWS,
};
#[cfg(any(test, feature = "reference-parity"))]
use crate::reference_mtp::{MtpError, Qwen4MtpState};
use crate::state::Qwen4StateSnapshot;
use hipfire_runtime::device_mesh::DimKind;
use hipfire_runtime::multi_gpu::executor::MeshRoute;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::ngram_mod::{MtpCopyContext, MtpNgramContext, NgramModConfig};
use hipfire_runtime::sampler::{sample_cpu, PenaltyTable, SamplerConfig};
use hipfire_runtime::session_cache::SessionRoute;
use hipfire_runtime::spec::{
    accept_greedy_prefix, terminal_prefix_replay, GreedyAccept, MtpDrafter, MtpRequestStats,
    MtpSpeculator, MtpWindow, PrefillOutcome, SpecAdvance, SpecGrammar, SpecRequestConfig,
    SpecScratch, SpecStep, SpecTarget, Speculator,
};
use hipfire_runtime::spec_economics::{
    derived_offer_bin, fn_measured_ar_price, fn_measured_depth_counts, fn_measured_load_identity,
    fn_measured_reference_supported, fn_measured_window_price, fn_phase1_ar_price,
    fn_phase1_window_price, native_margin_bin, CalibrationCounts, ContextBucket, DerivedAction,
    DerivedCard, DerivedCostMode, DerivedDepth, DerivedFeature, DerivedPrice, DerivedRequest,
    DerivedSource, NativeMarginBin, ResidualKey, DERIVED_MAX_DRAFTS, FN_DERIVED_VERSION,
};
#[cfg(test)]
use hipfire_runtime::spec_economics::{fresh_agreement, Agreement, MAX_DEPTH};
use hipfire_runtime::spec_economics::{
    observe_agreement, observe_agreement_counts, valid_us, Action, NativeWindow, WindowStats,
    AGREEMENT_PRIOR, DEFAULT_DECAY,
};
use hipfire_runtime::spec_sampling::{
    accept_naive_prefix, accept_sampled_prefix, naive_target_sampler, verify_sampled_draft,
    DraftVerdict, PenaltyHistory, SampleSpec, SparseDist, SpecRng,
};
use rdna_compute::profile::{unix_micros, Span, SpanProfiler};
use rdna_compute::spec_sampling_cpu_ar::{
    spec_cpu_ar_alloc_f32, spec_cpu_ar_alloc_u32, SpecCpuArPolicy, SpecDistRows, SPEC_CPU_AR_CAP,
    SPEC_CPU_AR_EOS_NONE,
};
use rdna_compute::{Gpu, GpuTensor};
use smallvec::SmallVec;

/// The operation context of the one native MTP driver: how the target and the
/// head's device are reached.
///
/// `Single` is the historical `&mut Gpu` every call used to take (one device
/// holds the whole target and the head). `Mesh` is a qwen4 `topology.pp`
/// mesh: the head, the drafter tensors and the spec hidden capture live on the
/// output band's device ([`Self::gpu`] resolves and binds it at the point of
/// use), while every target operation runs through the bundle's PP entries
/// over all bands. Both arms run the same driver bodies; only the operation
/// wrappers below dispatch.
pub(crate) enum Qwen4Ops<'a> {
    Single(&'a mut Gpu),
    Mesh(&'a mut Gpus),
}

impl<'a> Qwen4Ops<'a> {
    /// A mesh context; the output owner must exist (checked once here so the
    /// per-call resolution below cannot fail for a healthy mesh).
    pub(crate) fn mesh(gpus: &'a mut Gpus) -> Result<Self, String> {
        mesh_output_gpu(gpus)?;
        Ok(Self::Mesh(gpus))
    }

    pub(crate) fn is_mesh(&self) -> bool {
        matches!(self, Self::Mesh(_))
    }

    /// The device of the MTP head and every drafter tensor: the caller's `Gpu`
    /// (Single) or the mesh's output band device with the thread bound to it
    /// (Mesh). A coordinator leaves the thread on another device, so the
    /// mesh arm rebinds on every call.
    pub(crate) fn gpu(&mut self) -> Result<&mut Gpu, String> {
        match self {
            Self::Single(gpu) => Ok(&mut **gpu),
            Self::Mesh(gpus) => mesh_output_gpu(gpus),
        }
    }

    /// The device of a path that only exists on one card (n-gram takeover,
    /// derived K): refused under a mesh. Sampled Leviathan verification is not
    /// one of them: it reads the output band's device through [`Self::gpu`].
    pub(crate) fn single(&mut self, what: &str) -> Result<&mut Gpu, String> {
        match self {
            Self::Single(gpu) => Ok(&mut **gpu),
            Self::Mesh(_) => Err(format!(
                "Qwen4 native MTP {what} is not supported under qwen4 topology.pp"
            )),
        }
    }

    fn gpus(&mut self) -> Result<&mut Gpus, String> {
        match self {
            Self::Mesh(gpus) => Ok(&mut **gpus),
            Self::Single(_) => {
                Err("Qwen4 native MTP: no pipeline mesh in this context".to_string())
            }
        }
    }
}

/// The output band's device with the calling thread bound to it.
fn mesh_output_gpu(gpus: &mut Gpus) -> Result<&mut Gpu, String> {
    let owner = gpus
        .mesh_output_owner()
        .ok_or_else(|| "Qwen4 native MTP: the mesh has no output owner".to_string())?;
    let gpu = gpus
        .devices
        .get_mut(owner)
        .ok_or_else(|| "Qwen4 native MTP: the output device is unavailable".to_string())?;
    gpu.bind_thread()
        .map_err(|error| format!("Qwen4 native MTP: bind the output device: {error}"))?;
    Ok(gpu)
}

/// Wait for the head's queued work on the output device before a mesh target
/// forward, whose wide-hidden capture and logits writes ride the band's own
/// stream and so are not ordered behind the head's stream work (the spec
/// hidden rows, the pending hidden and the head's scratch are read by that
/// work).
fn mesh_quiesce_head(gpus: &mut Gpus) -> Result<(), String> {
    mesh_output_gpu(gpus)?
        .hip
        .device_synchronize()
        .map_err(|error| format!("Qwen4 native MTP: quiesce the output device: {error}"))
}

/// A target rollback ticket: a local state's snapshot, or the aggregate
/// window of a PP mesh (one local ticket per band plus the output head,
/// held by the mesh). A mesh ticket is never accepted by a local
/// `validate_ticket` and a local one never by the mesh: every consumer
/// matches the variant its [`Qwen4Ops`] arm expects.
#[derive(Clone, Copy)]
pub(crate) enum TargetTicket {
    Local(Qwen4StateSnapshot),
    Mesh(Qwen4PpWindow),
}

/// The MTP head's rollback ticket: local, or (Mesh) the same aggregate
/// window [`TargetTicket::Mesh`] carries, whose head piece the head-only
/// operations address.
#[derive(Clone, Copy)]
pub(crate) enum HeadTicket {
    Local(MtpGpuStateSnapshot),
    Mesh(Qwen4PpWindow),
}

impl TargetTicket {
    fn local(self) -> Result<Qwen4StateSnapshot, String> {
        match self {
            Self::Local(ticket) => Ok(ticket),
            Self::Mesh(_) => Err("Qwen4 native MTP: a mesh ticket on a local target".to_string()),
        }
    }

    fn mesh(self) -> Result<Qwen4PpWindow, String> {
        match self {
            Self::Mesh(ticket) => Ok(ticket),
            Self::Local(_) => Err("Qwen4 native MTP: a local ticket on a mesh target".to_string()),
        }
    }
}

impl HeadTicket {
    fn local(self) -> Result<MtpGpuStateSnapshot, String> {
        match self {
            Self::Local(ticket) => Ok(ticket),
            Self::Mesh(_) => Err("Qwen4 native MTP: a mesh ticket on a local head".to_string()),
        }
    }
}
use std::ops::ControlFlow;
use std::time::Instant;

/// `HIPFIRE_MTP_FLOOR_TRACE=1` (or `2`): one `QWEN4_FLOOR_TRACE` JSON line
/// per AR token, calibration, native and external window on stderr.
fn floor_trace_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_MTP_FLOOR_TRACE")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "2"))
}

/// `HIPFIRE_MTP_FLOOR_TRACE=2`: besides every `QWEN4_FLOOR_TRACE` line, one
/// `QWEN4_FLOOR_CALL` line per call into the floor's public API, with its
/// exact inputs and outputs, so a replay can feed the same call sequence into
/// a re-implementation and compare every return value bit for bit
/// ([`hipfire_runtime::spec_economics::replay_call_log`]).
fn floor_call_trace_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_MTP_FLOOR_TRACE").is_ok_and(|value| value == "2")
}

/// The prefix of the floor's `HIPFIRE_MTP_FLOOR_TRACE=1` state lines.
const FLOOR_TRACE_PREFIX: &str = "QWEN4_FLOOR_TRACE";

/// The prefix of the floor's `HIPFIRE_MTP_FLOOR_TRACE=2` call-log lines.
const FLOOR_CALL_PREFIX: &str = "QWEN4_FLOOR_CALL";

/// `floor` with the exact call log on stderr under `HIPFIRE_MTP_FLOOR_TRACE=2`
/// (its `install` line: what built it, `owner`, and whether the floor is
/// `enabled` for its request); `floor` itself otherwise.
pub(crate) fn floor_with_call_log(
    floor: Qwen4Floor,
    owner: &'static str,
    enabled: bool,
) -> Qwen4Floor {
    if floor_call_trace_enabled() {
        floor.with_call_log_enabled(FLOOR_CALL_PREFIX, owner, enabled)
    } else {
        floor
    }
}

/// A finite `f32` as `{}` (shortest round trip); `null` otherwise, which a
/// JSON number cannot be.
struct JsonF32(f32);

impl std::fmt::Display for JsonF32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_finite() {
            write!(f, "{}", self.0)
        } else {
            f.write_str("null")
        }
    }
}

/// A finite `f64` as `{}`; `null` otherwise.
struct JsonF64(f64);

impl std::fmt::Display for JsonF64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_finite() {
            write!(f, "{}", self.0)
        } else {
            f.write_str("null")
        }
    }
}

/// `HIPFIRE_MTP_PHASE_TIMING=1`: per-window phase lines on stderr.
fn phase_timing_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_MTP_PHASE_TIMING").is_ok_and(|value| value == "1")
}

/// Wall microseconds since `start`, after the GPU drained: the window's GPU
/// tail is included. Timing mode only (it synchronizes the device); a sync
/// failure only loses the tail, never the window.
fn synced_wall_us(gpu: &Gpu, start: Instant) -> f64 {
    let _ = gpu.hip.device_synchronize();
    start.elapsed().as_secs_f64() * 1e6
}

/// Device-side phase timing for one native MTP window.
///
/// Enabled by `HIPFIRE_MTP_PHASE_TIMING=1`.  Each phase is one `hipEvent` pair
/// recorded on the null stream and resolved once at window end, so the
/// instrument never synchronizes mid-window and never changes launch order:
/// the number it prints is the GPU time the phase actually occupies.
pub(crate) struct MtpPhaseTimers {
    spans: SpanProfiler,
    open: Option<Span>,
    /// A replica rank's timers record and resolve like rank 0's (the window's
    /// device work must not depend on the rank) but print nothing.
    silent: bool,
}

impl MtpPhaseTimers {
    pub(crate) fn new() -> Self {
        Self {
            spans: SpanProfiler::new(phase_timing_enabled()),
            open: None,
            silent: false,
        }
    }

    /// `new()` that prints nothing when `silent` (an EP follower rank).
    fn silenced(mut self, silent: bool) -> Self {
        self.silent = silent;
        self
    }

    fn enabled(&self) -> bool {
        self.spans.enabled()
    }

    /// Close the open phase and open `label`.
    fn mark(&mut self, gpu: &Gpu, label: &'static str) {
        self.spans.end(&gpu.hip, None, self.open.take());
        self.open = self.spans.begin(&gpu.hip, None, label);
    }

    /// Resolve every recorded pair and print one line.  `fields` carries the
    /// per-window scalars (position, k, accepted drafts).  Repeated labels are
    /// summed, so a per-row phase reads as its total across the window.
    fn finish(mut self, gpu: &Gpu, tag: &str, fields: &str) {
        self.spans.end(&gpu.hip, None, self.open.take());
        let totals = self.spans.resolve(&gpu.hip, None);
        if totals.is_empty() || self.silent {
            return;
        }
        let phases = totals
            .iter()
            .map(|total| {
                format!(
                    "\"{}\":{{\"us\":{:.1},\"t0\":{},\"t1\":{}}}",
                    total.label, total.us, total.first_start_unix_us, total.last_end_unix_us
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!("{tag} {{\"event\":\"mtp_phase\",{fields},\"phases_us\":{{{phases}}}}}");
    }

    /// Drop the recorded spans without printing (a native window that handed
    /// over to a takeover): the events are resolved so none stays live.
    fn discard(mut self, gpu: &Gpu) {
        self.spans.end(&gpu.hip, None, self.open.take());
        let _ = self.spans.resolve(&gpu.hip, None);
    }
}

/// Accepted drafts the target commits before the pending seed.
///
/// An accepted EOS draft (no bonus follows it) stays the pending seed, so
/// neither target nor MTP state consumes it until the terminal flush; a bonus
/// EOS is already predicted after the accepted prefix. This is also the
/// captured target-hidden row that produced the next pending seed.
pub fn target_commit_accept_len(accepted: &GreedyAccept) -> usize {
    let accepted_eos = accepted.hit_eos && accepted.committed.len() == accepted.accepted;
    accepted.accepted - usize::from(accepted_eos)
}

/// With `speculation.mtp_sampled` off (`HIPFIRE_MTP_SAMPLED=0`; on by
/// default), native MTP verifies greedy target picks only, so a sampled
/// request fails closed here (the router keeps such requests on AR).
pub fn require_native_greedy(temp: f32) -> Result<(), String> {
    if !temp.is_finite() || temp.abs() > 1.0e-6 {
        return Err(
            "Qwen4 native MTP verifies greedy requests only while speculation.mtp_sampled is off (HIPFIRE_MTP_SAMPLED=0)"
                .to_string(),
        );
    }
    Ok(())
}

/// Validate a native MTP prefill request before touching either owner.
///
/// A cold fill is the complete prompt from position zero. A cache hit fills
/// exactly `prompt_tokens[start_pos..]` after the bundle restored its
/// session snapshot at `start_pos`, so target and MTP resume at the same
/// absolute position.
pub fn validate_native_mtp_prefill_request(
    prompt_tokens: &[u32],
    fill_tokens: &[u32],
    start_pos: usize,
    cache_hit: bool,
) -> Result<(), String> {
    if prompt_tokens.is_empty() {
        return Err("Qwen4 native MTP prefill requires at least one prompt token".to_string());
    }
    if !cache_hit {
        if start_pos != 0 {
            return Err(format!(
                "Qwen4 native MTP cold prefill requires position zero, got {start_pos}"
            ));
        }
        if fill_tokens != prompt_tokens {
            return Err("Qwen4 native MTP prefill requires a complete prompt fill".to_string());
        }
        return Ok(());
    }
    if start_pos == 0 || start_pos >= prompt_tokens.len() {
        return Err(format!(
            "Qwen4 native MTP cache hit at {start_pos} needs a non-empty suffix of a {}-token prompt",
            prompt_tokens.len()
        ));
    }
    if fill_tokens != &prompt_tokens[start_pos..] {
        return Err("Qwen4 native MTP cache hit must fill exactly the prompt suffix".to_string());
    }
    Ok(())
}

/// Absolute target positions occupied by the committed verify prefix.
///
/// `position` is the seed's position and `num_accepted_tokens` includes that
/// seed, so the returned range has exactly `num_accepted_tokens` rows.  The
/// bonus is predicted, not consumed, and is therefore intentionally absent.
pub fn committed_target_positions(
    position: usize,
    num_accepted_tokens: usize,
) -> Result<Vec<usize>, String> {
    let end = position
        .checked_add(num_accepted_tokens)
        .ok_or_else(|| "Qwen4 native MTP target position overflow".to_string())?;
    Ok((position..end).collect())
}

#[cfg(any(test, feature = "reference-parity"))]
/// Compact target-aligned MTP QSA rows after a partial acceptance.
///
/// The caller supplies the seed position and the native count.  Rows at or
/// after the committed end are rejected-tail state and are discarded from the
/// sparse selection.  Full main K/V is restored/replayed by the target
/// transaction; this helper only handles the MTP side index row.
pub fn compact_native_qsa_selection(
    state: &mut Qwen4MtpState,
    position: usize,
    num_accepted_tokens: usize,
) -> Result<(), String> {
    let retained_end = position
        .checked_add(num_accepted_tokens)
        .ok_or_else(|| "Qwen4 native MTP QSA position overflow".to_string())?;
    if retained_end > state.qsa.position {
        return Err(format!(
            "Qwen4 native MTP QSA commit end {retained_end} exceeds active position {}",
            state.qsa.position
        ));
    }
    let retained = state
        .qsa
        .selected_indices
        .iter()
        .copied()
        .filter(|&row| row < retained_end)
        .collect::<Vec<_>>();
    state
        .qsa
        .compact_selection(&retained)
        .map_err(|error| format!("Qwen4 native MTP QSA compaction: {error}"))
}

#[cfg(any(test, feature = "reference-parity"))]
/// Compute the target's post-commit position without mutating either owner.
pub fn native_commit_position(
    position: usize,
    num_accepted_tokens: usize,
) -> Result<usize, String> {
    position
        .checked_add(num_accepted_tokens)
        .ok_or_else(|| "Qwen4 native MTP commit position overflow".to_string())
}

/// Apply the runtime's one shared greedy acceptance rule to native MTP picks,
/// refusing (rather than debug-asserting) a verifier that returned no bonus
/// slot.  The seed is not an argument: it is already represented by the
/// target block and is excluded from `committed` by construction.
pub fn accept_native_greedy(
    drafts: &[u32],
    target_picks: &[u32],
    eos: Option<u32>,
) -> Result<GreedyAccept, String> {
    if target_picks.len() < drafts.len().saturating_add(1) {
        return Err(format!(
            "Qwen4 native MTP verifier returned {} picks for {} drafts; one bonus pick is required",
            target_picks.len(),
            drafts.len()
        ));
    }
    Ok(accept_greedy_prefix(drafts, target_picks, eos))
}

/// Lower an MTP window directly to the runtime's pending-seed result.
/// `MtpWindow::committed` already excludes the seed, so this is a 1:1 emit
/// mapping and never performs a DFlash-style seed re-echo transformation.
pub fn window_to_spec_step(window: MtpWindow) -> Result<SpecStep, String> {
    let next_seed = *window
        .committed
        .last()
        .ok_or_else(|| "Qwen4 native MTP committed no token (would stall decode)".to_string())?;
    if window.accepted > window.drafts_generated {
        return Err(format!(
            "Qwen4 native MTP accepted {} drafts out of {}",
            window.accepted, window.drafts_generated
        ));
    }
    Ok(SpecStep::new(
        window.committed,
        next_seed,
        window.drafts_generated,
        window.accepted,
    ))
}
/// Reusable Qwen4 target-side verify scratch.  GPU output buffers and the
/// captured wide hidden rows belong to the bundle; this object owns only the
/// checked-out rollback ticket and its fixed block capacity.
pub struct Qwen4SpecScratch {
    block_size: usize,
    target_snapshot: Option<TargetTicket>,
    /// The pre-window ticket of the last committed window (start, end)
    /// kept for [`SpecTarget::repair_committed_prefix`]; the next snapshot,
    /// reset or restore supersedes it.
    retained: Option<(Qwen4StateSnapshot, usize, usize)>,
    #[cfg(feature = "reference-parity")]
    fault_after_snapshot: Option<usize>,
    #[cfg(feature = "reference-parity")]
    fault_receipt: Option<rdna_compute::LabKernelFaultReceipt>,
    #[cfg(all(feature = "reference-parity", feature = "kernel-timing"))]
    profile_after_snapshot: bool,
    #[cfg(all(feature = "reference-parity", feature = "kernel-timing"))]
    kernel_report: Option<hip_bridge::KernelTimingReport>,
}

/// [`SpecTarget::spec_advance`] on one rank inside an EP fan-out (mesh
/// detached, never aborting): the final row's argmax. This rank's prompt
/// nesting spans the whole advance, so a borrowed EP reserve restores once, at
/// its completion, on every rank (`spec_advance_from` nests inside).
fn replica_spec_advance(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: usize,
    reset: bool,
) -> Result<u32, String> {
    crate::ep::bracket_prompt(
        (bundle, gpu),
        |(bundle, _)| bundle.share_prompt_begin(),
        |(bundle, gpu)| match bundle.spec_advance(gpu, tokens, start_pos, reset, &|| false, None)? {
            SpecAdvance::Ready { last_argmax, .. } => Ok(last_argmax),
            SpecAdvance::Aborted => {
                Err("Qwen4 spec_advance aborted inside an EP fan-out".to_string())
            }
        },
        |(bundle, gpu), ok| {
            bundle
                .share_prompt_end(gpu, ok)
                .map_err(|error| error.to_string())
        },
    )
}

/// [`MtpDrafter::mtp_prefill`] on one rank inside an EP fan-out: the native-MTP
/// fill is one whole-prompt entry on this rank, so the borrowed reserve is
/// restored once, at its completion, after the last head continuation consumed
/// the wide capture (`mtp_prefill_ops` nests inside).
#[allow(clippy::too_many_arguments)]
fn replica_mtp_prefill(
    drafter: &mut Qwen4MtpDrafter,
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    prompt_tokens: &[u32],
    fill_tokens: &[u32],
    start_pos: usize,
    cache_hit: bool,
) -> Result<PrefillOutcome, String> {
    crate::ep::bracket_prompt(
        (drafter, (bundle, gpu)),
        |(_, (bundle, _))| bundle.share_prompt_begin_mtp_fill(),
        |(drafter, (bundle, gpu))| {
            drafter.mtp_prefill(
                &mut **gpu,
                &mut **bundle,
                prompt_tokens,
                fill_tokens,
                start_pos,
                cache_hit,
                &|| false,
            )
        },
        |(_, (bundle, gpu)), ok| {
            bundle
                .share_prompt_end(gpu, ok)
                .map_err(|error| error.to_string())
        },
    )
}

impl SpecScratch for Qwen4SpecScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn free(self: Box<Self>, _gpu: &mut Gpu) {
        // A live ticket is consumed by verify/commit before the speculator is
        // released.  Bundle teardown also owns the arena, so there is no
        // independent GPU allocation to free here.
        debug_assert!(
            self.target_snapshot.is_none(),
            "Qwen4 verify scratch dropped with an active target snapshot"
        );
    }

    fn clear(&mut self) {
        self.target_snapshot = None;
        self.retained = None;
    }
}

/// The operations of the one native MTP driver that touch the target (and
/// the per-band state it spans), dispatched on the operation context: the
/// `Single` arm is each call the driver made before the context existed; the
/// `Mesh` arm is the same operation over every band of the PP mesh through
/// the bundle's PP entries. The head's own operations (`mtp_*`, the spec
/// hidden copies) stay local to the root bundle and take [`Qwen4Ops::gpu`].
impl Qwen4Bundle {
    /// Canonical rows one target forward takes.
    fn ops_chunk_rows(&self, ops: &Qwen4Ops<'_>) -> Option<usize> {
        match ops {
            Qwen4Ops::Single(_) => self.spec_chunk_rows(),
            Qwen4Ops::Mesh(_) => self.pp_chunk_rows(),
        }
    }

    /// The target (every band, under a mesh) reset to a cold state, head
    /// included (the bundle's own reset covers the head).
    fn ops_reset(&mut self, ops: &mut Qwen4Ops<'_>) -> Result<(), String> {
        let reset = match ops {
            Qwen4Ops::Single(gpu) => self.reset(gpu),
            Qwen4Ops::Mesh(gpus) => self.pp_reset(gpus),
        };
        reset.map_err(|error| format!("Qwen4 reset_recurrent: {error}"))
    }

    /// Row-capture rings (`rows` rows) on every band that holds GDN layers.
    fn ops_ensure_row_capture(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        rows: usize,
    ) -> Result<(), String> {
        match ops {
            Qwen4Ops::Single(gpu) => {
                let history = self.config.linear_conv_kernel_dim - 1;
                self.state
                    .ensure_row_capture(gpu, rows, history)
                    .map_err(|error| error.to_string())
            }
            Qwen4Ops::Mesh(gpus) => self
                .pp_ensure_row_capture(gpus, rows)
                .map(|_| ())
                .map_err(|error| error.to_string()),
        }
    }

    /// Output and top-1 views of `rows` verify rows of the spec buffers (the
    /// same views [`Qwen4Bundle::spec_forward_rows`] hands the forward).
    fn mesh_head_views(&self, rows: usize) -> Result<(GpuTensor, GpuTensor), String> {
        let logits_len = rows
            .checked_mul(self.config.vocab_size)
            .ok_or_else(|| "Qwen4 spec logits overflow".to_string())?;
        let top1_len = rows
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| "Qwen4 spec argmax overflow".to_string())?;
        let logits = self
            .spec_logits
            .as_ref()
            .filter(|logits| logits.numel() >= logits_len)
            .ok_or_else(|| "Qwen4 spec logits are not attached or too small".to_string())?
            .sub_offset(0, logits_len);
        let top1 = self
            .spec_top1
            .as_ref()
            .filter(|top1| top1.byte_size() >= top1_len)
            .ok_or_else(|| "Qwen4 spec argmax is not attached or too small".to_string())?
            .sub_offset(0, top1_len);
        Ok((logits, top1))
    }

    /// The first `rows` rows of the spec hidden capture.
    fn mesh_wide_view(&self, rows: usize) -> Result<GpuTensor, String> {
        let width = self
            .config
            .hc_count
            .checked_mul(self.config.hidden_size)
            .ok_or_else(|| "Qwen4 spec hidden width overflow".to_string())?;
        let len = rows
            .checked_mul(width)
            .ok_or_else(|| "Qwen4 spec hidden row overflow".to_string())?;
        Ok(self
            .spec_hidden
            .as_ref()
            .filter(|hidden| hidden.numel() >= len)
            .ok_or_else(|| "Qwen4 spec hidden is not allocated or too small".to_string())?
            .sub_offset(0, len))
    }

    /// Read `count` argmax rows of `top1` from the output device.
    fn mesh_read_picks(
        &mut self,
        gpus: &mut Gpus,
        top1: &GpuTensor,
        count: usize,
    ) -> Result<Vec<u32>, String> {
        let bytes_len = count * std::mem::size_of::<i32>();
        if self.spec_host_top1.len() < bytes_len {
            return Err("Qwen4 spec host argmax capacity is too small".to_string());
        }
        let gpu = mesh_output_gpu(gpus)?;
        gpu.hip
            .memcpy_dtoh(&mut self.spec_host_top1[..bytes_len], &top1.buf)
            .map_err(|error| error.to_string())?;
        Ok(self.spec_host_top1[..bytes_len]
            .chunks_exact(4)
            .map(|bytes| u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect())
    }

    /// One verify-shaped forward over `tokens` with the wide hidden captured:
    /// [`Qwen4Bundle::spec_forward_rows`] (`read_picks`) or its resident form
    /// (no pick readback, an empty `Vec`). Under a mesh every band runs the
    /// block on the Verify route (the output band captures the final
    /// pre-collapse wide hidden and computes every row's logits).
    fn ops_spec_rows(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        tokens: &[u32],
        read_picks: bool,
    ) -> Result<Vec<u32>, String> {
        match ops {
            Qwen4Ops::Single(gpu) => if read_picks {
                self.spec_forward_rows(gpu, tokens, true)
            } else {
                self.spec_forward_rows_resident(gpu, tokens, true)
                    .map(|()| Vec::new())
            }
            .map_err(|error| error.to_string()),
            Qwen4Ops::Mesh(gpus) => {
                let chunk = self
                    .pp_chunk_rows()
                    .ok_or_else(|| "Qwen4 mesh chunk is not attached".to_string())?;
                if tokens.is_empty() || tokens.len() > chunk {
                    return Err(format!(
                        "Qwen4 mesh verify block length {} is outside 1..={chunk}",
                        tokens.len()
                    ));
                }
                mesh_quiesce_head(gpus)?;
                let (logits, top1) = self.mesh_head_views(tokens.len())?;
                let wide = self.mesh_wide_view(tokens.len())?;
                self.pp_forward_verify(gpus, tokens, &logits, Some(&top1), &wide)
                    .map_err(|error| error.to_string())?;
                if read_picks {
                    self.mesh_read_picks(gpus, &top1, tokens.len())
                } else {
                    Ok(Vec::new())
                }
            }
        }
    }

    /// [`Qwen4Bundle::spec_capture_token`] (`read_pick`) or its resident form
    /// (returns 0): one chunk-route row with the wide hidden in spec row 0.
    fn ops_capture_token(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        token: u32,
        read_pick: bool,
    ) -> Result<u32, String> {
        match ops {
            Qwen4Ops::Single(gpu) => {
                if read_pick {
                    self.spec_capture_token(gpu, token)
                        .map_err(|error| error.to_string())
                } else {
                    self.spec_capture_token_resident(gpu, token)
                        .map(|()| 0)
                        .map_err(|error| error.to_string())
                }
            }
            Qwen4Ops::Mesh(_) => {
                let picks = self.ops_spec_rows(ops, std::slice::from_ref(&token), read_pick)?;
                if !read_pick {
                    return Ok(0);
                }
                picks
                    .into_iter()
                    .next()
                    .ok_or_else(|| "Qwen4 spec capture returned no argmax".to_string())
            }
        }
    }

    /// [`Qwen4Bundle::spec_ar_token`]: one ordinary target-only token (the
    /// Decode route under a mesh), its argmax read back, no capture.
    fn ops_ar_token(&mut self, ops: &mut Qwen4Ops<'_>, token: u32) -> Result<u32, String> {
        match ops {
            Qwen4Ops::Single(gpu) => self
                .spec_ar_token(gpu, token)
                .map_err(|error| error.to_string()),
            Qwen4Ops::Mesh(gpus) => {
                mesh_quiesce_head(gpus)?;
                let (logits, top1) = self.mesh_head_views(1)?;
                self.pp_forward(
                    gpus,
                    std::slice::from_ref(&token),
                    Some(&logits),
                    Some(&top1),
                    Qwen4OutputRows::Final,
                    MeshRoute::Decode,
                    None,
                )
                .map_err(|error| error.to_string())?;
                self.mesh_read_picks(gpus, &top1, 1)?
                    .into_iter()
                    .next()
                    .ok_or_else(|| "Qwen4 spec AR token returned no argmax".to_string())
            }
        }
    }

    /// The trunk tile of a prompt chunk with wide-hidden capture:
    /// `spec_prefill_rows` when `needs_pick`, else `spec_prefill_rows_silent`.
    /// Under a mesh the chunk runs through every band on the Prefill route
    /// and the output band captures the chunk's wide rows for the head.
    fn ops_prefill_rows(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        tokens: &[u32],
        needs_pick: bool,
    ) -> Result<Option<u32>, String> {
        match ops {
            Qwen4Ops::Single(gpu) => {
                if needs_pick {
                    self.spec_prefill_rows(gpu, tokens, true)
                        .map(Some)
                        .map_err(|error| error.to_string())
                } else {
                    self.spec_prefill_rows_silent(gpu, tokens)
                        .map(|_| None)
                        .map_err(|error| error.to_string())
                }
            }
            Qwen4Ops::Mesh(gpus) => {
                let chunk = self
                    .pp_chunk_rows()
                    .ok_or_else(|| "Qwen4 mesh chunk is not attached".to_string())?;
                if tokens.is_empty() || tokens.len() > chunk {
                    return Err(format!(
                        "Qwen4 mesh prefill chunk length {} is outside 1..={chunk}",
                        tokens.len()
                    ));
                }
                mesh_quiesce_head(gpus)?;
                let wide = self.mesh_wide_view(tokens.len())?;
                if needs_pick {
                    let (logits, top1) = self.mesh_head_views(1)?;
                    self.pp_forward(
                        gpus,
                        tokens,
                        Some(&logits),
                        Some(&top1),
                        Qwen4OutputRows::Final,
                        MeshRoute::Prefill,
                        Some(&wide),
                    )
                    .map_err(|error| error.to_string())?;
                    self.mesh_read_picks(gpus, &top1, 1)?
                        .into_iter()
                        .next()
                        .map(Some)
                        .ok_or_else(|| "Qwen4 prefill produced no argmax".to_string())
                } else {
                    self.pp_forward(
                        gpus,
                        tokens,
                        None,
                        None,
                        Qwen4OutputRows::Final,
                        MeshRoute::Prefill,
                        Some(&wide),
                    )
                    .map_err(|error| error.to_string())?;
                    Ok(None)
                }
            }
        }
    }

    /// Clear the row-capture arm flag after a verify (every band, under a
    /// mesh).
    fn ops_disarm(&mut self, ops: &Qwen4Ops<'_>) {
        match ops {
            Qwen4Ops::Single(_) => self.state.row_capture_armed = false,
            Qwen4Ops::Mesh(_) => self.pp_window_arm(false),
        }
    }

    /// Whether the captured rows of `ticket`'s `rows`-row verify can repair a
    /// partial accept without replaying the kept rows.
    fn ops_ring_eligible(&self, ticket: TargetTicket, rows: usize) -> bool {
        match ticket {
            TargetTicket::Local(_) => self.state.row_capture_rows() >= rows,
            TargetTicket::Mesh(window) => window.row_repair_ready(),
        }
    }

    /// Keep the first `keep` rows of the armed verify `ticket` brackets
    /// without re-running them: each band's own row-aware repair. The target
    /// only; the head is handled by the caller. The ticket stays active.
    fn ops_rollback_rows(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        ticket: TargetTicket,
        keep: usize,
        block: &[u32],
        ple_normed: Option<&GpuTensor>,
    ) -> Result<(), String> {
        match ops {
            Qwen4Ops::Single(gpu) => {
                let ple_normed = ple_normed
                    .ok_or_else(|| "Qwen4 local row rollback has no PLE rows".to_string())?;
                self.rollback_verify_rows_retain_with(gpu, ticket.local()?, keep, block, ple_normed)
                    .map_err(|error| error.to_string())
            }
            Qwen4Ops::Mesh(gpus) => self
                .pp_window_repair_keep(gpus, &ticket.mesh()?, keep, block)
                .map_err(|error| error.to_string()),
        }
    }

    /// Drop the head's rejected tail (`keep` rows kept) of a retained
    /// window, host only.
    fn ops_head_truncate_retain(&mut self, head: HeadTicket, keep: usize) -> Result<(), String> {
        match head {
            HeadTicket::Local(ticket) => self
                .mtp_truncate_retain(ticket, keep)
                .map_err(|error| error.to_string()),
            HeadTicket::Mesh(window) => self
                .pp_window_head_truncate_retain(&window, keep)
                .map_err(|error| error.to_string()),
        }
    }

    /// The head alone back to the window's start, its ticket kept active.
    fn ops_head_restore_retain(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        head: HeadTicket,
    ) -> Result<(), String> {
        match head {
            HeadTicket::Local(ticket) => self
                .mtp_restore_retain(ops.gpu()?, ticket)
                .map_err(|error| error.to_string()),
            HeadTicket::Mesh(window) => self
                .pp_window_head_restore_retain(ops.gpus()?, &window)
                .map_err(|error| error.to_string()),
        }
    }
    /// The pre-window tickets `target`/`head` still validate (no owner moved
    /// past them since they were taken).
    fn ops_window_valid(&self, target: TargetTicket, head: HeadTicket) -> Result<(), String> {
        match (target, head) {
            (TargetTicket::Local(target), HeadTicket::Local(head)) => {
                self.validate_commit(target)
                    .map_err(|error| error.to_string())?;
                self.mtp_validate_commit(head)
                    .map_err(|error| error.to_string())
            }
            (TargetTicket::Mesh(window), HeadTicket::Mesh(_)) => {
                if self.pp_window_live(&window) {
                    Ok(())
                } else {
                    Err("Qwen4 mesh window ticket is stale".to_string())
                }
            }
            _ => Err("Qwen4 native MTP: target and head tickets disagree".to_string()),
        }
    }

    /// End a window whose every fallible target/head operation succeeded:
    /// validate every owner's ticket before invalidating any, then either
    /// commit (consume) or retain (keep rollback-able) the tickets of all of
    /// them. Under a mesh this is the aggregate finalize over every band and
    /// the head.
    fn ops_finish_window(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        target: TargetTicket,
        head: HeadTicket,
        retain: bool,
    ) -> Result<(), String> {
        match (target, head) {
            (TargetTicket::Local(target), HeadTicket::Local(head)) => {
                if retain {
                    self.retain_window(target, head)
                        .map_err(|error| error.to_string())
                } else {
                    self.validate_commit(target)
                        .map_err(|error| error.to_string())?;
                    self.mtp_validate_commit(head)
                        .map_err(|error| error.to_string())?;
                    self.commit_validated(target);
                    self.mtp_commit_validated(head);
                    Ok(())
                }
            }
            (TargetTicket::Mesh(window), HeadTicket::Mesh(_)) => {
                let gpus = ops.gpus()?;
                self.pp_window_finalize(gpus, &window, retain)
                    .map_err(|error| error.to_string())
            }
            _ => Err("Qwen4 native MTP: target and head tickets disagree".to_string()),
        }
    }

    /// Release a retained or abandoned window's tickets (host only; a ticket
    /// a snapshot, reset or restore already superseded is left alone).
    fn ops_release_window(&mut self, target: TargetTicket, head: HeadTicket) {
        match (target, head) {
            (TargetTicket::Local(target), HeadTicket::Local(head)) => {
                self.release_window(target, head)
            }
            (TargetTicket::Mesh(window), _) => self.pp_window_release(&window),
            _ => {}
        }
    }

    /// Physical CPU-owned verify storage, including the seed. A legacy load
    /// retains its default window; an admitted+prewarmed load uses its actual
    /// reserved extent. Qualification remains a separate per-call bound.
    pub(crate) fn spec_verify_rows_cap(&self) -> usize {
        let storage = self.cpu_storage_capacity_rows();
        let cpu = self.has_cpu_deferred_experts() || storage > 0;
        let default = spec_verify_rows_cap(cpu);
        if cpu {
            storage.max(default)
        } else {
            default
        }
    }

    /// Refuse a `rows`-row speculative verify or replay on an armed CPU
    /// splice load before any state moves ([`check_spec_verify_rows`]).
    fn require_spec_verify_rows(&self, rows: usize, what: &str) -> Result<(), String> {
        check_spec_verify_rows(self.spec_verify_rows_cap(), rows, what)
    }

    /// The verify transaction behind [`SpecTarget::verify_block`]: snapshot,
    /// row-armed forward, restore on error, end-position check. `read_picks`
    /// downloads every row's argmax; `false` leaves the picks on the device
    /// and returns an empty `Vec` (the logits, hidden capture and state are
    /// identical).
    ///
    /// Under a mesh the aggregate window (every band and the head, armed
    /// first) was opened by the driver before it drafted and sits in
    /// `scratch`: verify only runs the block, and an error leaves the ticket
    /// for the driver's single aggregate rollback.
    fn verify_block_rows(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        read_picks: bool,
    ) -> Result<Vec<u32>, String> {
        let mesh = ops.is_mesh();
        // Drafter-internal under EP: runs only inside a fan-out (mesh detached).
        if !mesh {
            self.ep_refuse("speculative decoding (verify block)")
                .map_err(|error| error.to_string())?;
        }
        let block_size = {
            let s = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .ok_or("Qwen4 verify_block: scratch is not Qwen4SpecScratch")?;
            match (&s.target_snapshot, mesh) {
                (Some(TargetTicket::Local(_)), false) => {
                    return Err("Qwen4 verify_block: target snapshot is already active".to_string());
                }
                (None, true) => {
                    return Err("Qwen4 verify_block: the mesh window is not open".to_string());
                }
                (Some(_), true) | (Some(TargetTicket::Mesh(_)), false) => {
                    if matches!(s.target_snapshot, Some(TargetTicket::Mesh(_))) && mesh {
                        // The aggregate window the driver opened before it drafted.
                    } else {
                        return Err(
                            "Qwen4 verify_block: the ticket does not match the context".to_string()
                        );
                    }
                }
                _ => {}
            }
            s.block_size
        };
        if block.is_empty() || block.len() > block_size {
            return Err(format!(
                "Qwen4 verify block length {} is outside scratch capacity {block_size}",
                block.len()
            ));
        }
        if self.state.position != position {
            return Err(format!(
                "Qwen4 verify_block position mismatch: expected {}, got {position}",
                self.state.position
            ));
        }
        let end_pos = position
            .checked_add(block.len())
            .ok_or_else(|| "Qwen4 verify position overflow".to_string())?;
        if end_pos > self.state.max_seq_len {
            return Err(format!(
                "Qwen4 verify end {end_pos} exceeds context capacity {}",
                self.state.max_seq_len
            ));
        }
        self.require_spec_verify_rows(block.len(), "Qwen4 verify block")?;
        if !mesh {
            // The verify leaves per-row rollback points, so a rejected suffix is
            // dropped without re-running the accepted rows (armed before the
            // snapshot: the GDN states then need no copy).
            self.state.row_capture_armed =
                block.len() >= 2 && self.state.row_capture_rows() >= block.len();
            let snapshot = match self.snapshot(ops.gpu()?) {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    self.state.row_capture_armed = false;
                    return Err(error.to_string());
                }
            };
            scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .ok_or("Qwen4 verify_block: scratch is not Qwen4SpecScratch")?
                .target_snapshot = Some(TargetTicket::Local(snapshot));
        }
        // A single-GPU verify's row launchers may take the gfx1151 PM row
        // kernels (column-identical to the 1-row incumbents, as in the lane step).
        let rows_scope = match ops {
            Qwen4Ops::Single(gpu) => Some(std::mem::replace(&mut gpu.lane_rows_scope, true)),
            Qwen4Ops::Mesh(_) => None,
        };
        #[cfg(feature = "reference-parity")]
        let inject_fault = {
            let s = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .ok_or("Qwen4 verify_block: scratch is not Qwen4SpecScratch")?;
            std::mem::take(&mut s.fault_after_snapshot)
        };
        #[cfg(feature = "reference-parity")]
        let result = (|| {
            #[cfg(feature = "kernel-timing")]
            let profile = {
                let s = scratch
                    .as_any_mut()
                    .downcast_mut::<Qwen4SpecScratch>()
                    .ok_or("Qwen4 verify profiling: wrong scratch type")?;
                std::mem::take(&mut s.profile_after_snapshot)
            };
            #[cfg(feature = "kernel-timing")]
            if profile {
                ops.gpu()?
                    .lab_begin_kernel_timing()
                    .map_err(|e| e.to_string())?;
            }
            let result = match inject_fault {
                Some(n) => match ops.gpu()?.lab_arm_kernel_fault_after(n) {
                    Ok(()) => {
                        let result = self.ops_spec_rows(ops, block, read_picks);
                        let receipt = ops.gpu()?.lab_disarm_kernel_fault();
                        if let Some(s) = scratch.as_any_mut().downcast_mut::<Qwen4SpecScratch>() {
                            s.fault_receipt = Some(receipt);
                        }
                        result
                    }
                    Err(error) => Err(error.to_string()),
                },
                None => self.ops_spec_rows(ops, block, read_picks),
            };
            #[cfg(feature = "kernel-timing")]
            if profile {
                if result.is_err() {
                    ops.gpu()?
                        .lab_cancel_kernel_timing()
                        .map_err(|e| e.to_string())?;
                } else {
                    match ops.gpu()?.lab_finish_kernel_timing() {
                        Ok(report) => {
                            if let Some(s) = scratch.as_any_mut().downcast_mut::<Qwen4SpecScratch>()
                            {
                                s.kernel_report = Some(report);
                            }
                        }
                        Err(error) => {
                            let cancel = ops.gpu()?.lab_cancel_kernel_timing();
                            return Err(match cancel {
                                Ok(()) => error.to_string(),
                                Err(cancel) => {
                                    format!("{error}; collector cancellation failed: {cancel}")
                                }
                            });
                        }
                    }
                }
            }
            result
        })();
        #[cfg(not(feature = "reference-parity"))]
        let result = self.ops_spec_rows(ops, block, read_picks);
        if let (Some(previous), Qwen4Ops::Single(gpu)) = (rows_scope, &mut *ops) {
            gpu.lane_rows_scope = previous;
        }
        self.ops_disarm(ops);
        if let Err(error) = &result {
            if mesh {
                return Err(error.clone());
            }
            let snapshot = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .and_then(|s| s.target_snapshot.take());
            if let Some(snapshot) = snapshot {
                self.restore(ops.gpu()?, snapshot.local()?)
                    .map_err(|restore| format!("{error}; target restore failed: {restore}"))?;
            }
            return Err(error.clone());
        }
        if self.state.position != end_pos {
            let mismatch = format!(
                "Qwen4 verify ended at {}, expected {end_pos}",
                self.state.position
            );
            if mesh {
                return Err(mismatch);
            }
            let snapshot = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .and_then(|s| s.target_snapshot.take());
            if let Some(snapshot) = snapshot {
                self.restore(ops.gpu()?, snapshot.local()?)
                    .map_err(|restore| format!("{mismatch}; target restore failed: {restore}"))?;
            }
            return Err(mismatch);
        }
        result
    }

    /// [`SpecTarget::verify_block`] for a window whose acceptance consumes the
    /// verify's resident logits on the device: the same transaction without
    /// the argmax pick readback.
    fn verify_block_resident(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        self.verify_block_rows(ops, block, position, scratch, false)
            .map(|_| ())
    }

    /// The non-EP advance body of [`SpecTarget::spec_advance`]: reset when asked, validate, then run the rows. A reset
    /// advance is re-entrant after an NPU fault (it starts cold again). With
    /// `hidden_out` and an armed DFlash2 drafter every row's target features
    /// are appended to it and the final row's logits are returned too.
    fn spec_advance_from(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        // The whole advance is one prompt for decode-share: the donor tail is
        // restored once, after its last tile (`share_finish_prompt`).
        self.share_prompt_begin();
        let result = self.spec_advance_tiles(gpu, tokens, start_pos, reset, abort, hidden_out);
        let finish = self
            .share_prompt_end(gpu, result.is_ok())
            .map_err(|error| error.to_string());
        result.and_then(|advance| finish.map(|()| advance))
    }

    /// The tile loop of [`Self::spec_advance_from`].
    fn spec_advance_tiles(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
        mut hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        if self.dflash.is_none() {
            hidden_out = None;
        }
        let capture = hidden_out.is_some();
        if reset {
            self.reset_recurrent(gpu)?;
        }
        if tokens.is_empty() {
            return Err("Qwen4 spec_advance cannot process an empty token slice".to_string());
        }
        if self.state.position != start_pos {
            return Err(format!(
                "Qwen4 spec_advance position mismatch: expected {}, got {start_pos}",
                self.state.position
            ));
        }
        let end_pos = start_pos
            .checked_add(tokens.len())
            .ok_or_else(|| "Qwen4 spec position overflow".to_string())?;
        if end_pos > self.state.max_seq_len {
            return Err(format!(
                "Qwen4 spec advance end {end_pos} exceeds context capacity {}",
                self.state.max_seq_len
            ));
        }
        let max_chunk = self
            .execution
            .as_ref()
            .ok_or_else(|| "Qwen4 forward resources are not attached".to_string())?
            .prompt_chunk_rows();
        self.begin_npu_trunk_request(gpu)
            .map_err(|error| error.to_string())?;
        let plan_end = self
            .session_lineage_after(start_pos)
            .map_or(end_pos, |p| p.min(end_pos));
        if let Some(forward) = &mut self.execution {
            // The pair executor runs layer-major outside the step
            // interpreter, where the DFlash2 layer capture has no hook.
            if capture {
                forward.npu_trunk_request_gpu_only("DFlash2 layer capture");
            }
            forward
                .plan_npu_trunk_request(gpu, plan_end - start_pos, start_pos)
                .map_err(|error| error.to_string())?;
        }
        let mut offset = 0usize;
        let mut last_argmax = None;
        while offset < tokens.len() {
            if abort() {
                self.reset_recurrent(gpu)?;
                return Ok(SpecAdvance::Aborted);
            }
            self.poll_npu_trunk(gpu)
                .map_err(|error| error.to_string())?;
            let planned = self.npu_trunk_planned();
            let rows = self.prefill_schedule_rows(start_pos + offset, end_pos, max_chunk);
            let end = offset + rows;
            // A planned pair group (rows above one chunk) runs through the pair
            // executor inside the forward, in every NPU mode.
            let covered = planned && rows > max_chunk;
            self.set_ple_lookahead(&tokens[end..]);
            if covered && end < tokens.len() {
                self.spec_prefill_rows_without_pick(gpu, &tokens[offset..end], capture)
                    .map_err(|error| error.to_string())?;
            } else {
                last_argmax = Some(
                    self.spec_prefill_rows(gpu, &tokens[offset..end], capture)
                        .map_err(|error| error.to_string())?,
                );
            }
            if let Some(out) = hidden_out.as_deref_mut() {
                self.read_dflash_capture(gpu, rows, out)
                    .map_err(|error| error.to_string())?;
                // A long prompt keeps only the rows the draft reads.
                self.trim_dflash_rows(out);
            }
            if self.session_next_boundary() == Some(start_pos + end) {
                self.spec_advance_session_boundary(gpu, start_pos + end)
                    .map_err(|e| e.to_string())?;
            }
            offset = end;
        }
        if self.state.position != end_pos {
            return Err(format!(
                "Qwen4 spec advance ended at {}, expected {end_pos}",
                self.state.position
            ));
        }
        let last_logits = if capture {
            let mut logits = Vec::new();
            self.spec_row_logits(gpu, 0, &mut logits)
                .map_err(|error| error.to_string())?;
            Some(logits)
        } else {
            None
        };
        Ok(SpecAdvance::Ready {
            last_argmax: last_argmax.expect("non-empty spec advance produced no argmax"),
            last_logits,
        })
    }

    /// The verify scratch of a native driver: `block_size` rows of spec
    /// hidden capture (on the output device under a mesh) and an empty
    /// ticket slot.
    fn ops_new_spec_scratch(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        if !ops.is_mesh() {
            self.ep_refuse("speculative decoding (verify scratch)")
                .map_err(|error| error.to_string())?;
        }
        let block_size = block_size.max(1);
        let max_chunk = match ops {
            // The block is the prompt tile the spec hidden capture holds: the
            // prompt schedule capacity, not the resident bank of a
            // decode-share load (S4-MAXCHUNK class a).
            Qwen4Ops::Single(_) => self
                .execution
                .as_ref()
                .ok_or_else(|| {
                    "Qwen4 spec scratch requires attached forward resources".to_string()
                })?
                .prompt_chunk_rows(),
            Qwen4Ops::Mesh(_) => self
                .pp_chunk_rows()
                .ok_or_else(|| "Qwen4 spec scratch requires an attached mesh".to_string())?,
        };
        if block_size > max_chunk {
            return Err(format!(
                "Qwen4 spec block size {block_size} exceeds forward capacity {max_chunk}"
            ));
        }
        // An armed DFlash2 drafter reads its target layers, not the MTP
        // head's wide hidden.
        if self.dflash.is_some() {
            self.ensure_dflash_capture(ops.gpu()?, block_size)
        } else {
            self.ensure_spec_hidden(ops.gpu()?, block_size)
        }
        .map_err(|error| error.to_string())?;
        Ok(Box::new(Qwen4SpecScratch {
            block_size,
            target_snapshot: None,
            retained: None,
            #[cfg(feature = "reference-parity")]
            fault_after_snapshot: None,
            #[cfg(feature = "reference-parity")]
            fault_receipt: None,
            #[cfg(all(feature = "reference-parity", feature = "kernel-timing"))]
            profile_after_snapshot: false,
            #[cfg(all(feature = "reference-parity", feature = "kernel-timing"))]
            kernel_report: None,
        }))
    }
}

/// Developer hooks of the wide-verify oracle (`wide_verify_oracle`): thin
/// crate-internal entry points onto the private verify transaction, compiled
/// only for the parity tools. No production route calls them.
#[cfg(feature = "reference-parity")]
impl Qwen4Bundle {
    /// [`Self::verify_block_rows`] exactly: the production verify transaction.
    pub(crate) fn dev_verify_block_rows(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        read_picks: bool,
    ) -> Result<Vec<u32>, String> {
        self.verify_block_rows(
            &mut Qwen4Ops::Single(gpu),
            block,
            position,
            scratch,
            read_picks,
        )
    }

    pub(crate) fn dev_verify_fault_after_snapshot(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        scratch: &mut dyn SpecScratch,
        successful_launches: usize,
    ) -> Result<crate::wide_verify_oracle::DevVerifyFaultOutcome, String> {
        let s = scratch
            .as_any_mut()
            .downcast_mut::<Qwen4SpecScratch>()
            .ok_or("Qwen4 fault injection: wrong scratch type")?;
        s.fault_after_snapshot = Some(successful_launches);
        s.fault_receipt = None;
        let position = self.state.position;
        let result = self.dev_verify_block_rows(gpu, block, position, scratch, false);
        let s = scratch
            .as_any_mut()
            .downcast_mut::<Qwen4SpecScratch>()
            .ok_or("Qwen4 fault injection: wrong scratch type")?;
        s.fault_after_snapshot = None;
        let receipt = s.fault_receipt.take();
        match result {
            Err(verify_error) => {
                let receipt = receipt.filter(|r| r.injected).ok_or_else(|| {
                    format!("verify failed without the requested kernel fault: {verify_error}")
                })?;
                Ok(crate::wide_verify_oracle::DevVerifyFaultOutcome {
                    verify_error,
                    receipt,
                })
            }
            Ok(_) => {
                self.dev_rollback_verify(gpu, scratch)?;
                Err("requested kernel fault did not fire; successful verify was rolled back".into())
            }
        }
    }

    #[cfg(feature = "kernel-timing")]
    pub(crate) fn dev_verify_resident_profiled(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        scratch: &mut dyn SpecScratch,
    ) -> Result<hip_bridge::KernelTimingReport, String> {
        let s = scratch
            .as_any_mut()
            .downcast_mut::<Qwen4SpecScratch>()
            .ok_or("Qwen4 verify profiling: wrong scratch type")?;
        s.profile_after_snapshot = true;
        s.kernel_report = None;
        let position = self.state.position;
        let result = self.dev_verify_block_rows(gpu, block, position, scratch, false);
        let s = scratch
            .as_any_mut()
            .downcast_mut::<Qwen4SpecScratch>()
            .ok_or("Qwen4 verify profiling: wrong scratch type")?;
        s.profile_after_snapshot = false;
        let report = s.kernel_report.take();
        result?;
        match report {
            Some(report) => Ok(report),
            None => {
                self.dev_rollback_verify(gpu, scratch)?;
                Err("verify succeeded without a kernel report; transaction rolled back".into())
            }
        }
    }

    /// Rewind a finished local verify without committing: take its target
    /// ticket out of scratch, reject a mesh ticket, then restore and consume
    /// the local snapshot (also releasing its arena entry on copy failure).
    pub(crate) fn dev_rollback_verify(
        &mut self,
        gpu: &mut Gpu,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        let snapshot = scratch
            .as_any_mut()
            .downcast_mut::<Qwen4SpecScratch>()
            .ok_or("Qwen4 rollback: scratch is not Qwen4SpecScratch")?
            .target_snapshot
            .take()
            .ok_or("Qwen4 rollback: target snapshot is not active")?
            .local()?;
        self.restore(gpu, snapshot)
            .map_err(|error| error.to_string())
    }

    /// The production native partial-accept finish of the verify the scratch
    /// ticket brackets, target side only: keep the first `keep` rows of the
    /// armed `block.len()`-row verify by the retain-path recurrence repair
    /// (`rollback_verify_rows_retain`, the captured-row route the batched
    /// window uses), or none for a full accept, then commit the ticket and
    /// consume it. The generic `commit_prefix` (restore + replay) is a
    /// different path. Refuses, before any mutation, a keep outside
    /// `1..=block.len()`, a missing ticket and, for a partial keep, a capture
    /// ring that does not hold the verify's rows (it never falls back to the
    /// replay). The ticket stays active on any error.
    pub(crate) fn dev_commit_captured_prefix(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        keep: usize,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        if block.is_empty() || keep == 0 || keep > block.len() {
            return Err(format!(
                "Qwen4 native commit: keep {keep} outside 1..={}",
                block.len()
            ));
        }
        let scratch = scratch
            .as_any_mut()
            .downcast_mut::<Qwen4SpecScratch>()
            .ok_or("Qwen4 native commit: scratch is not Qwen4SpecScratch")?;
        let ticket = scratch
            .target_snapshot
            .ok_or("Qwen4 native commit: target snapshot is not active")?;
        let snapshot = ticket.local()?;
        self.validate_commit(snapshot)
            .map_err(|error| error.to_string())?;
        let verified_end = self.state.position;
        let start = verified_end
            .checked_sub(block.len())
            .ok_or("Qwen4 native commit: position below the verified block")?;
        if keep < block.len() {
            if !self.ops_ring_eligible(ticket, block.len()) {
                return Err(format!(
                    "Qwen4 native commit: the {}-row capture ring does not hold the {}-row verify (no native repair route)",
                    self.state.row_capture_rows(),
                    block.len()
                ));
            }
            self.rollback_verify_rows_retain(gpu, snapshot, keep, block)
                .map_err(|error| error.to_string())?;
        }
        if self.state.position != start + keep {
            return Err(format!(
                "Qwen4 native commit ended at position {}, expected {}",
                self.state.position,
                start + keep
            ));
        }
        self.commit(gpu, snapshot)
            .map_err(|error| error.to_string())?;
        scratch.target_snapshot = None;
        self.state.row_capture_armed = false;
        Ok(())
    }
}

impl SpecTarget for Qwen4Bundle {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.reset(gpu)
            .map_err(|error| format!("Qwen4 reset_recurrent: {error}"))
    }

    fn retry_reset_eligible(&self) -> bool {
        true
    }

    fn session_request_write_limit(&mut self, through: usize) {
        Qwen4Bundle::session_request_write_limit(self, through);
    }

    fn new_spec_scratch(
        &mut self,
        gpu: &mut Gpu,
        block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        self.ops_new_spec_scratch(&mut Qwen4Ops::Single(gpu), block_size)
    }

    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        // Expert parallelism: every rank advances in one fan-out. Rank 0
        // checks the request and polls the abort once, here; inside the
        // fan-out every rank uses a never-abort closure, so none stops alone.
        if self.ep.is_some() {
            if tokens.is_empty() {
                if reset {
                    self.reset_recurrent(gpu)?;
                }
                return Err("Qwen4 spec_advance cannot process an empty token slice".to_string());
            }
            if abort() {
                self.reset_recurrent(gpu)?;
                return Ok(SpecAdvance::Aborted);
            }
            if hidden_out.is_some() && self.dflash.is_some() {
                return Err("DFlash2 is not supported under qwen4 expert parallelism".to_string());
            }
            let last_argmax = self
                .ep_fanout(
                    "spec_advance",
                    |bundle| {
                        replica_spec_advance(bundle, gpu, tokens, start_pos, reset)
                            .map_err(BundleError::Forward)
                    },
                    &|bundle, peer_gpu, _| {
                        replica_spec_advance(bundle, peer_gpu, tokens, start_pos, reset)
                    },
                )
                .map_err(|error| error.to_string())?;
            return Ok(SpecAdvance::Ready {
                last_argmax,
                last_logits: None,
            });
        }
        // An NPU fault latches the route off. A reset advance starts cold, so it reruns whole GPU-only (the replan is always
        // row 0); a forced advance mid-turn cannot be replayed and fails closed, the route staying off for later requests.
        if reset {
            let mut hidden_out = hidden_out;
            let start = hidden_out.as_deref().map_or(0, Vec::len);
            return crate::npu_trunk::containment::rerun_after_fault(
                self,
                tokens.len(),
                0,
                |bundle, _| {
                    // A rerun starts cold: drop the faulted attempt's rows.
                    if let Some(out) = hidden_out.as_deref_mut() {
                        out.truncate(start);
                    }
                    bundle.spec_advance_from(
                        gpu,
                        tokens,
                        start_pos,
                        true,
                        abort,
                        hidden_out.as_deref_mut(),
                    )
                },
                |bundle| bundle.take_npu_trunk_fault(),
                |_| 0,
            );
        }
        let result = self.spec_advance_from(gpu, tokens, start_pos, false, abort, hidden_out);
        if result.is_err() {
            self.take_npu_trunk_fault();
        }
        result
    }

    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        let picks =
            self.verify_block_rows(&mut Qwen4Ops::Single(gpu), block, position, scratch, true)?;
        if let Some(out) = hidden_out.filter(|_| self.dflash.is_some()) {
            self.read_dflash_capture(gpu, block.len(), out)
                .map_err(|error| error.to_string())?;
        }
        Ok(picks)
    }

    fn verify_block_logits(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        self.verify_block_rows(&mut Qwen4Ops::Single(gpu), block, position, scratch, false)?;
        let logits = self
            .spec_logits_rows(0, block.len())
            .and_then(|rows| gpu.download_f32(&rows).map_err(BundleError::Hip))
            .map_err(|error| error.to_string())?;
        if let Some(out) = hidden_out.filter(|_| self.dflash.is_some()) {
            self.read_dflash_capture(gpu, block.len(), out)
                .map_err(|error| error.to_string())?;
        }
        Ok(logits)
    }

    fn dflash_extract_layers(&self) -> Option<&[usize]> {
        self.dflash_layers()
    }

    fn set_dflash_extract_layers(&mut self, layers: Vec<usize>) {
        self.set_dflash_layers(layers);
    }

    fn embed_row(&mut self, gpu: &mut Gpu, token_id: u32) -> Result<Vec<f32>, String> {
        self.dflash_embed_row(gpu, token_id)
            .map_err(|error| error.to_string())
    }

    fn set_dflash_feature_window(&mut self, rows: usize) {
        self.set_dflash_window(rows);
    }

    fn abandon_verify(
        &mut self,
        gpu: &mut Gpu,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        let ticket = scratch
            .as_any_mut()
            .downcast_mut::<Qwen4SpecScratch>()
            .ok_or("Qwen4 abandon_verify: scratch is not Qwen4SpecScratch")?
            .target_snapshot
            .take();
        match ticket {
            None => Ok(()),
            Some(ticket) => self
                .restore(gpu, ticket.local()?)
                .map_err(|error| error.to_string()),
        }
    }

    fn lm_head_logits_device(
        &mut self,
        gpu: &mut Gpu,
        hidden_rows: &GpuTensor,
        n: usize,
    ) -> Result<GpuTensor, String> {
        self.dflash_lm_head(gpu, hidden_rows, n)
            .map_err(|error| error.to_string())
    }

    fn commit_prefix(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        accept_len: usize,
        position: usize,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        self.ep_refuse("speculative decoding (commit prefix)")
            .map_err(|error| error.to_string())?;
        if block.is_empty() {
            return Err("Qwen4 commit_prefix cannot commit an empty block".to_string());
        }
        // Before the snapshot ticket is consumed: a block wider than the CPU
        // splice window was never verified, so nothing is restored or replayed.
        self.require_spec_verify_rows(block.len(), "Qwen4 commit block")?;
        let draft_len = block.len() - 1;
        if accept_len > draft_len {
            return Err(format!(
                "Qwen4 commit_prefix accepts {accept_len} drafts out of {draft_len}"
            ));
        }
        let verified_end = position
            .checked_add(block.len())
            .ok_or_else(|| "Qwen4 commit position overflow".to_string())?;
        if self.state.position != verified_end {
            return Err(format!(
                "Qwen4 commit_prefix target position mismatch: expected {verified_end}, got {}",
                self.state.position
            ));
        }
        let committed_end = position
            .checked_add(accept_len + 1)
            .ok_or_else(|| "Qwen4 commit position overflow".to_string())?;
        let scratch = scratch
            .as_any_mut()
            .downcast_mut::<Qwen4SpecScratch>()
            .ok_or("Qwen4 commit_prefix: scratch is not Qwen4SpecScratch")?;
        scratch.retained = None;
        let snapshot = scratch
            .target_snapshot
            .take()
            .ok_or("Qwen4 commit_prefix: target snapshot is not active")?
            .local()?;
        if accept_len == draft_len {
            self.validate_commit(snapshot)
                .map_err(|error| error.to_string())?;
            self.retain_target_validated(snapshot);
            scratch.retained = Some((snapshot, position, committed_end));
            return Ok(());
        }
        // A verify that left per-row rollback points (row capture armed for
        // this block) drops the rejected suffix in place, as native MTP
        // does; otherwise restore and replay the kept rows.
        if block.len() >= 2 && self.state.row_capture_rows() >= block.len() {
            if let Err(error) =
                self.rollback_verify_rows_retain(gpu, snapshot, accept_len + 1, block)
            {
                return Err(match self.restore(gpu, snapshot) {
                    Ok(()) => error.to_string(),
                    Err(restore) => format!("{error}; target restore failed: {restore}"),
                });
            }
            self.validate_commit(snapshot)
                .map_err(|error| error.to_string())?;
            self.retain_target_validated(snapshot);
            scratch.retained = Some((snapshot, position, committed_end));
        } else {
            self.restore(gpu, snapshot)
                .map_err(|error| error.to_string())?;
            // The DFlash2 drafter already read this window's features from
            // the verify; the replay re-captures only the MTP head's wide
            // hidden.
            let capture = self.dflash.is_none();
            self.spec_forward_rows(gpu, &block[..accept_len + 1], capture)
                .map(|_| ())
                .map_err(|error| error.to_string())?;
            // No window retained: the restore consumed the ticket.
        }
        if self.state.position != committed_end {
            return Err(format!(
                "Qwen4 commit_prefix replay ended at {}, expected {committed_end}",
                self.state.position
            ));
        }
        Ok(())
    }

    fn repair_committed_prefix(
        &mut self,
        gpu: &mut Gpu,
        scratch: &mut dyn SpecScratch,
        position: usize,
        replay: &[u32],
    ) -> Result<bool, String> {
        let Some((ticket, start, end)) = scratch
            .as_any_mut()
            .downcast_mut::<Qwen4SpecScratch>()
            .ok_or("Qwen4 repair_committed_prefix: scratch is not Qwen4SpecScratch")?
            .retained
            .take()
        else {
            return Ok(false);
        };
        let live = self.validate_commit(ticket).is_ok();
        if !live
            || start != position
            || replay.is_empty()
            || start + replay.len() > end
            || self.state.position != end
        {
            if live {
                self.commit_validated(ticket);
            }
            return Ok(false);
        }
        let replayed = self
            .restore_retain(gpu, ticket)
            .map_err(|error| format!("Qwen4 terminal repair restore: {error}"))
            .and_then(|()| {
                // The drafter keeps the window's feature rows: no capture.
                self.spec_forward_rows(gpu, replay, self.dflash.is_none())
                    .map_err(|error| format!("Qwen4 terminal repair replay: {error}"))
            });
        if self.validate_commit(ticket).is_ok() {
            self.commit_validated(ticket);
        }
        replayed?;
        if self.state.position != start + replay.len() {
            return Err(format!(
                "Qwen4 terminal repair ended at {}, expected {}",
                self.state.position,
                start + replay.len()
            ));
        }
        Ok(true)
    }

    fn eos_token(&self) -> u32 {
        self.config.eos_token_id
    }

    fn ctx_capacity(&self) -> usize {
        self.state.max_seq_len
    }

    fn map_ahead_for_decode(
        &mut self,
        gpu: &mut Gpu,
        position: usize,
        remaining_new_tokens: usize,
        max_draft_rows: usize,
    ) {
        Qwen4Bundle::map_ahead_for_decode(
            self,
            gpu,
            position,
            remaining_new_tokens,
            max_draft_rows,
        );
    }
}

/// How a native batched window's drafting closure ended.
enum NativeOutcome {
    /// The window ran to its verdict.
    Window(MtpWindow),
    /// The head's first draft equals the pool's first candidate: the window
    /// is abandoned before any draft is kept or verified, and the n-gram
    /// takeover runs instead.
    Takeover,
}

/// Clock and cost of the native work a takeover was confirmed through (the
/// batched switch or the interleaved probe), for the takeover's phase line.
#[derive(Clone, Copy)]
struct TakeoverProbe {
    /// The native window's start: the takeover's `window_us` includes the probe.
    start: Instant,
    /// Synced wall microseconds from `start` to the switch (timing mode only).
    probe_us: f64,
}

fn takeover_probe(gpu: &Gpu, start: Instant) -> TakeoverProbe {
    TakeoverProbe {
        start,
        probe_us: if phase_timing_enabled() {
            synced_wall_us(gpu, start)
        } else {
            0.0
        },
    }
}

/// Cost of a batched window that drafts K tokens (index K: K draft steps,
/// a (K+1)-row verify, rollback), in interleaved-route emitted tokens (one
/// single-row forward plus one draft step). gfx1151, Qwen3.8-Flash-Next,
/// measured: interleaved ~33.5 ms/token, K=2 window ~62.5 ms, K=3 ~75.4 ms;
/// the others from the few-row forward's growth (2 rows ~50 ms, 5 ~69,
/// 8 ~98) plus ~2.7 ms per draft step.
pub(crate) const MTP_WINDOW_COST: [f32; 8] = [1.0, 1.7, 1.87, 2.25, 2.55, 2.75, 3.0, 3.6];
/// The singleton drafter's decay on gfx1151 (~30-window memory). The floor
/// retires a request to AR, stickily, once no native route's measured cost
/// per expected token beats AR. With the ~8-window memory an early run of
/// rejections or one slow window retired requests that native MTP was
/// winning: on sampled HumanEval/0-9 (Strix Halo), /1 and /6 retired in every
/// run. Decaying only the agreement at 0.97 still retired /6 in every run.
const MTP_AGREEMENT_DECAY_GFX1151: f32 = 0.97;
/// Emitted tokens a request's first takeover is assumed to yield (one
/// pseudo-window): the offline 5/3/3 TC/Hermes accepted-prefix-plus-bonus on
/// a pool hit was 2.9-3.2 (`release-0.4.1/mtp-ngram-plan.md` §1).
const NGRAM_YIELD_PRIOR: f32 = 3.0;
/// Per declined pool hit, the takeover yield history decays by this, so a
/// source that lost to native MTP is re-tried once the evidence ages out.
const NGRAM_YIELD_RECOVERY: f32 = 0.95;
/// Draft depths of the request-local n-gram acceptance table: one per verify
/// row of the spec capacity below the seed, the longest proposal the pool
/// can make (`qwen4_ngram_mod_config` caps `n_max` there).
const NGRAM_DEPTHS: usize = QWEN4_SPEC_VERIFY_ROWS - 1;
/// Fewest candidates of a takeover that may skip the head's row-0
/// confirmation, once the copy source is established (see
/// [`NGRAM_UNCONFIRMED_MIN_ACCEPT`]). Checked against teacher-forced replays
/// of the real agentic chain (Halo `h1`, 17 requests at 45K-220K, and the
/// R9700 smoke chain; `release-0.4.3/ngram-copy-research/rowzero-tune.md`,
/// a simulation, not a hardware measurement): 5 instead of the provisional 4
/// kept the replay's tokens per AR-cost unit (-0.03% Halo, +0.38% R9700)
/// and cut blind first-candidate divergences (Halo 10.3% of skipped
/// confirmations vs 11.5%). Lowering [`NGRAM_UNCONFIRMED_MIN_ACCEPT`] to 0.8
/// bought at most 1.2% and diverged on 19% of skipped confirmations.
const NGRAM_UNCONFIRMED_MIN_TAKE: usize = 5;
/// Depth-1 n-gram acceptance (`ngram_agreement[0]`'s ratio) an unconfirmed
/// takeover needs (kept at 0.9 by the replay grid above).
const NGRAM_UNCONFIRMED_MIN_ACCEPT: f32 = 0.9;
/// Decayed windows of evidence (`ngram_agreement[0]`'s weight) behind that
/// acceptance before an unconfirmed takeover may run: the table starts at the
/// prior's weight, which no confirmation-free takeover may rest on (kept at
/// 4 by the replay grid above; 2 and 8 were no better).
const NGRAM_UNCONFIRMED_MIN_EVIDENCE: f32 = 4.0;
/// Smallest probability that a verify row's whole draft prefix is accepted
/// for the row (plus its draft step) to pay: ~6 ms of a ~47 ms window
/// emitting ~2-3.5 tokens.
pub(crate) const MTP_ROW_WORTH: f32 = 0.2;
/// Best per-draft acceptance [`draft_accept_estimate`] gives.
pub(crate) const DRAFT_ACCEPT_MAX: f32 = 0.98;

/// Acceptance of a draft whose exact logit leads its runner-up (among the
/// re-scored candidates) by `margin`; measured on the committed sweep prompts
/// (gfx1151, Qwen3.8-Flash-Next, all depths pooled).
pub(crate) fn draft_accept_estimate(margin: f32) -> f32 {
    match margin {
        m if m < 0.25 => 0.3,
        m if m < 1.5 => 0.5,
        m if m < 3.0 => 0.7,
        m if m < 6.0 => 0.82,
        _ => DRAFT_ACCEPT_MAX,
    }
}

/// `QWEN4_MTP_TRACE` per-row `{draft, pick, match}` objects.
fn trace_rows(drafts: &[u32], picks: &[u32]) -> String {
    drafts
        .iter()
        .zip(picks)
        .map(|(draft, pick)| {
            format!(
                "{{\"draft\":{draft},\"pick\":{pick},\"match\":{}}}",
                draft == pick
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// The `,"gdn_residual_resets":N` suffix of a `QWEN4_MTP_TRACE` window line:
/// the submitted full-owner live EF residual zero sequences of `state`
/// (initial reset, restores, partial rollbacks); empty unless the GDN format
/// carries a live residual, so Q8/F32 trace lines are unchanged.
fn gdn_residual_trace_field(state: &crate::state::Qwen4State) -> String {
    if state.gdn_format().has_residual() {
        format!(",\"gdn_residual_resets\":{}", state.gdn_residual_resets())
    } else {
        String::new()
    }
}

/// Speculative windows a singleton request's floor observes before it may
/// retire.  Measured routing still starts after
/// [`hipfire_runtime::spec_economics::FLOOR_PROBE_WINDOWS`];
/// only the sticky retirement waits.  The first windows of a response accept
/// poorly far more often than its body: on Strix Halo real-text serve, with
/// retirement at the probe budget, 5 of 9 sampled 32K requests retired within
/// their first windows and ran the whole response at AR speed (28–29 tok/s
/// against 40–46 when drafting continued).  `HIPFIRE_MTP_FLOOR_MIN_WINDOWS=<n>`
/// (n ≥ 1) overrides it; `3` restores retirement at the probe budget.
/// Continuous-batching lanes keep retirement at the probe budget.
const MTP_SINGLETON_FLOOR_MIN_WINDOWS: usize = 16;

/// Ordinary AR tokens a singleton request's floor times, at the moment it
/// would retire, to re-price AR where the request is now (not where it
/// started).  The first decode token is not a representative AR sample under
/// the Flash-Next expert hot set: its cost is a property of that one token's
/// routed experts (VRAM hot set against host-mapped), 7–26% off the request's
/// mean AR token on the R9700 traces (humaneval_0 19.6 ms against 23.7 ms,
/// bare_factual 23.2 against 29.5), and AR tokens spread 20–28 ms inside one
/// response.  Six probes bound the mean's error to a few percent.
const MTP_SINGLETON_FLOOR_AR_PROBE_TOKENS: usize = 6;

/// The singleton request's retirement threshold:
/// `HIPFIRE_MTP_FLOOR_MIN_WINDOWS` when it parses as an integer ≥ 1, else
/// [`MTP_SINGLETON_FLOOR_MIN_WINDOWS`].
fn singleton_floor_min_windows() -> usize {
    hipfire_config::developer_var("HIPFIRE_MTP_FLOOR_MIN_WINDOWS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(MTP_SINGLETON_FLOOR_MIN_WINDOWS)
}

/// The singleton drafter's agreement / floor-cost / n-gram-yield decay on
/// `arch`: [`MTP_AGREEMENT_DECAY_GFX1151`] on gfx1151, where the longer
/// memory was measured, else [`DEFAULT_DECAY`]. The lanes keep
/// [`DEFAULT_DECAY`].
fn singleton_agreement_decay(arch: &str) -> f32 {
    if arch == "gfx1151" {
        MTP_AGREEMENT_DECAY_GFX1151
    } else {
        DEFAULT_DECAY
    }
}

/// The qwen4 route's floor: native K 1..7 ([`MTP_WINDOW_COST`]) and a takeover slot per
/// n-gram depth (`NGRAM_DEPTHS + 1`).
pub(crate) type Qwen4Floor =
    hipfire_runtime::spec_economics::Floor<{ MTP_WINDOW_COST.len() }, { NGRAM_DEPTHS + 1 }>;

/// Fresh AR samples a singleton request's floor takes at a would-be
/// retirement (see [`Qwen4Floor::with_retire_evidence`]).
/// `HIPFIRE_MTP_FLOOR_AR_PROBE=<n>` overrides it; `0` is the kill switch and
/// restores the single-calibration-token floor.
fn singleton_floor_ar_probe_tokens() -> usize {
    hipfire_config::developer_var("HIPFIRE_MTP_FLOOR_AR_PROBE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(MTP_SINGLETON_FLOOR_AR_PROBE_TOKENS)
}

fn elapsed_us(start: Instant) -> f32 {
    (start.elapsed().as_secs_f64() * 1e6) as f32
}

/// Wall time since `start`, after the device drained: a window's cost
/// includes its whole asynchronous GPU tail, so no tail is billed to the next
/// window (or to the AR calibration).
fn synced_elapsed_us(gpu: &Gpu, start: Instant) -> Result<f32, String> {
    gpu.hip
        .device_synchronize()
        .map_err(|error| format!("Qwen4 MTP floor window sync: {error}"))?;
    Ok(elapsed_us(start))
}
/// Sampled verification algorithm (`HIPFIRE_MTP_SAMPLED_MODE`, resolved once
/// per drafter).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampledMode {
    /// Speculative rejection sampling (Leviathan et al. 2023, Chen et al.
    /// 2023): drafts drawn from `q`, emitted stream equals AR in
    /// distribution. The default.
    Leviathan,
    /// SpecInfer naive sampling: argmax drafts, one AR-sampler draw per verify
    /// row, accept iff equal. The emitted stream is AR's seeded stream.
    Naive,
}

impl SampledMode {
    /// `naive` selects [`SampledMode::Naive`]; unset or `leviathan` the
    /// default. Any other value is refused rather than silently defaulted.
    fn from_env() -> Result<Self, String> {
        match hipfire_config::developer_var("HIPFIRE_MTP_SAMPLED_MODE").as_deref() {
            Ok("naive") => Ok(Self::Naive),
            Ok("leviathan") | Err(_) => Ok(Self::Leviathan),
            Ok(other) => Err(format!(
                "HIPFIRE_MTP_SAMPLED_MODE={other:?}: expected `leviathan` or `naive`"
            )),
        }
    }
}

/// Per-request sampled verification (`speculation.mtp_sampled`, temperature
/// above zero).
///
/// Target policy: the Qwen4 AR producer is host `sampler::sample_cpu` over
/// the FULL rendered prompt plus the generated tokens, with the request's
/// repeat / presence / frequency penalties over the trailing `repeat_window`
/// tokens, then temperature, top_p, top_k and min_p. [`Self::policy`]
/// ([`naive_target_sampler`]) is that sampler and is the target policy of
/// BOTH modes; [`Self::history`] supplies its token history. The history
/// penalized at verify row `i` (the row that scores the token after `i`
/// accepted drafts) is `H_i = suffix_W(P ‖ E ‖ drafts[..i])`, with `P` the
/// prompt, `E` the tokens emitted before the window (pending seed included),
/// and `W` the penalty window. Rebuilt from `E` every window
/// ([`Self::begin_window`]): a rejected draft or a pruned proposal never
/// enters it. A neutral request has window 0 and does no history work.
///
/// [`SampledMode::Leviathan`]:
/// - `p`, a verify row's target distribution: the logit row penalized by
///   `H_i`, then truncated by [`SampleSpec::cpu_ar`] with the request's
///   temperature, top_p, top_k and min_p — exactly what the Qwen4 AR
///   producer samples (`sampler::sample_cpu` → `llama::sample_top_k_p`: a
///   20- or 64-wide pool, the top_k cap with absent = 20 and 0 = 64, min_p,
///   nucleus).
/// - `q`, a draft step's distribution: the same policy and truncation
///   applied to the draft head's 8 re-scored candidates and their exact
///   logits (the whole draft row when the head does not re-score), penalized
///   by the same `H_i` as the verify row the draft is proposed for. The
///   draft token is drawn from this `q`, and the verdict reads this `q`.
///   Tokens of `p` outside the 8 candidates are reached through the residual
///   or the bonus draw.
///
/// Accept with probability `min(1, p/q)`, otherwise emit a draw from
/// `(p - q)+`; a window whose drafts are all accepted emits its bonus from
/// `p`. One request-seeded stream supplies every draw.
///
/// [`SampledMode::Naive`] (`naive` set): drafts stay the head's argmax; each
/// verify row the verdict reads is drawn with `sampler::sample_cpu`, the
/// penalty history `H_i` and the AR producer's sampler config, from the
/// process-wide AR sampler RNG, and a draft is accepted iff it equals its
/// row's draw ([`accept_naive_prefix`]). Every emitted token is one draw, in
/// AR's order.
struct SampledVerify {
    spec: SampleSpec,
    rng: SpecRng,
    /// [`SampledMode::Naive`]: one AR-sampler draw per verify row, argmax
    /// drafts. `false`: Leviathan.
    naive: bool,
    /// The AR producer's sampler ([`naive_target_sampler`]): the target
    /// policy of both modes, penalties included.
    policy: SamplerConfig,
    /// [`Self::policy`] without its penalty stage: the host policy of a
    /// target row the GPU prepass already penalized.
    row_policy: SamplerConfig,
    /// Target rows take the GPU penalty prepass
    /// ([`Qwen4Bundle::apply_spec_penalty_table`], bit-identical to the host
    /// penalty stage) before their download: `policy` has a penalty stage
    /// and `HIPFIRE_QWEN4_PENALTY_PREPASS` is not `0`. Draft `q`s keep the
    /// host policy over the same history rows.
    gpu_rows: bool,
    /// The prepass's reused per-row token table.
    table: PenaltyTable,
    /// Penalty history: prompt tail, tokens emitted before the window and
    /// the window's kept drafts; window 0 when no penalty is active.
    history: PenaltyHistory,
    /// The request's prompt is in `history`. Set by the request's first
    /// prefill; a later prefill of the same request is a strict-prefix
    /// realign whose `prompt_tokens` replay the prompt AND emitted tokens,
    /// which `begin_window(emitted)` already supplies.
    prompt_set: bool,
    /// Host copy of one logit row.
    host: Vec<f32>,
    scratch: Vec<(u32, f32)>,
    target: SparseDist,
    /// `q` of each draft step in the current window (Leviathan only).
    drafts: Vec<SparseDist>,
    /// Point-mass `q`s of drafts that were not drawn from a distribution
    /// (`accept_leviathan` with `point_mass`), reused across windows.
    points: Vec<SparseDist>,
    /// Resident Leviathan acceptance (`HIPFIRE_QWEN4_MTP_DEVICE_ACCEPT`):
    /// target rows built on the device from the verify's resident logits and
    /// ordered acceptance there. `None` until `Qwen4MtpDrafter::ensure_resources`
    /// provisions it for an eligible request; moved to the drafter's spares
    /// by `configure_request` (it has no `Gpu`) and freed only by `mtp_free`.
    device: Option<DeviceAccept>,
    /// The request passes the device-accept predicate that needs no `Gpu`
    /// ([`device_accept_eligible`] and the env switch); the architecture and
    /// pool-capacity checks happen when `device` is provisioned.
    device_eligible: bool,
    /// The current window took a device row build or device accept; read by
    /// the `accept_route` trace field. Reset by [`Self::begin_window`].
    route_device: bool,
    /// `HIPFIRE_QWEN4_MTP_SAMPLED_PICKS` is `1`: a device-route window
    /// downloads no argmax picks unless trace or phase timers need them
    /// ([`Self::device_skip_picks`]).
    sampled_picks: bool,
    /// The window being observed (`Qwen4MtpDrafter::lab_observe_windows`);
    /// `None` unarmed, so every hook is a single test and does no work.
    #[cfg(feature = "lab")]
    lab: Option<Box<crate::lab_window::LabWindow>>,
}

impl SampledVerify {
    fn new(cfg: SpecRequestConfig, max_k: usize, mode: SampledMode) -> Self {
        let naive = mode == SampledMode::Naive;
        if naive {
            // The AR producer's per-request seeding of the shared sampler
            // RNG (`generate` already did it with this seed on the daemon
            // route; repeating it keeps a direct caller replayable).
            hipfire_runtime::llama::reset_cpu_sampler_rng(cfg.rng_seed as u32);
        }
        let policy = naive_target_sampler(&cfg);
        let spec = SampleSpec::cpu_ar(cfg.temp, cfg.top_p, cfg.top_k, cfg.min_p);
        let gpu_rows = PenaltyTable::flags_for(&policy) != 0 && penalty_prepass_enabled();
        let device_eligible = mtp_device_accept_enabled()
            && device_accept_eligible(
                naive,
                cfg.penalty_window(),
                gpu_rows,
                policy.blocked_tokens.is_empty(),
                &spec,
            );
        Self {
            spec,
            rng: SpecRng::new(cfg.rng_seed),
            drafts: if naive {
                Vec::new()
            } else {
                vec![SparseDist::default(); max_k]
            },
            points: Vec::new(),
            naive,
            gpu_rows,
            row_policy: policy.without_penalties(),
            policy,
            table: PenaltyTable::default(),
            history: PenaltyHistory::new(cfg.penalty_window()),
            prompt_set: false,
            host: Vec::new(),
            scratch: Vec::with_capacity(SampleSpec::MAX_POOL),
            target: SparseDist::default(),
            device: None,
            device_eligible,
            route_device: false,
            sampled_picks: mtp_sampled_picks_enabled(),
            #[cfg(feature = "lab")]
            lab: None,
        }
    }

    /// Leviathan mode draws the drafts from `q`; naive keeps the argmax.
    fn draws_drafts(&self) -> bool {
        !self.naive
    }

    /// Start a window: rebuild the penalty history from the emitted tokens
    /// (the pending seed included); returns the history length before drafts.
    fn begin_window(&mut self, emitted: &[u32]) -> usize {
        self.route_device = false;
        self.history.begin_window(emitted)
    }

    /// GPU prepass of physical verify rows `first..first + hist_rows.len()`,
    /// row `first + i` penalized by history row `hist_rows.start + i`, in one
    /// launch queued behind the forward that wrote them.
    fn penalize_rows(
        &mut self,
        gpu: &mut Gpu,
        bundle: &mut Qwen4Bundle,
        first: usize,
        hist_rows: std::ops::Range<usize>,
    ) -> Result<(), String> {
        let vocab = bundle.config.vocab_size;
        self.table.reset(&self.policy);
        for hist_row in hist_rows {
            self.table
                .push_row(self.history.row(hist_row), &self.policy, vocab);
        }
        bundle
            .apply_spec_penalty_table(gpu, first, &self.table)
            .map_err(|error| error.to_string())
    }

    /// Naive mode: physical verify row `row`'s target token, one AR-sampler
    /// draw penalized by history row `hist_row`.
    fn naive_draw(
        &mut self,
        gpu: &mut Gpu,
        bundle: &mut Qwen4Bundle,
        row: usize,
        hist_row: usize,
    ) -> Result<u32, String> {
        if !self.naive {
            return Err("Qwen4 sampled MTP: naive draw outside naive mode".to_string());
        }
        if self.gpu_rows {
            self.penalize_rows(gpu, bundle, row, hist_row..hist_row + 1)?;
        }
        self.draw_penalized(gpu, bundle, row, hist_row)
    }

    /// [`Self::naive_draw`] of a row the GPU prepass already penalized when
    /// [`Self::gpu_rows`] (the host applies the whole policy otherwise).
    fn draw_penalized(
        &mut self,
        gpu: &Gpu,
        bundle: &Qwen4Bundle,
        row: usize,
        hist_row: usize,
    ) -> Result<u32, String> {
        bundle
            .spec_row_logits(gpu, row, &mut self.host)
            .map_err(|error| error.to_string())?;
        let (policy, history) = if self.gpu_rows {
            (&self.row_policy, &[][..])
        } else {
            (&self.policy, self.history.row(hist_row))
        };
        let token = sample_cpu(&mut self.host, history, policy);
        #[cfg(feature = "lab")]
        if let Some(lab) = self.lab.as_mut() {
            lab.note_naive_draw(hist_row, token);
        }
        Ok(token)
    }

    /// Draw the draft token from the last MTP prediction's `q` (penalized by
    /// history row `hist_row`), kept as the window's draft distribution
    /// `index`.
    fn sample_draft(
        &mut self,
        gpu: &Gpu,
        bundle: &Qwen4Bundle,
        index: usize,
        hist_row: usize,
    ) -> Result<u32, String> {
        let q = self
            .drafts
            .get_mut(index)
            .ok_or_else(|| format!("Qwen4 sampled MTP draft index {index} exceeds K"))?;
        bundle
            .mtp_draft_dist(
                gpu,
                self.spec,
                self.history.row(hist_row),
                &self.policy,
                &mut self.host,
                &mut self.scratch,
                q,
            )
            .map_err(|error| error.to_string())?;
        #[cfg(feature = "lab")]
        let lab_before = self.rng;
        let token = q.sample(self.rng.next_f32());
        #[cfg(feature = "lab")]
        if let Some(lab) = self.lab.as_mut() {
            lab.note_draft(token, q.entries(), &lab_before);
        }
        Ok(token)
    }

    /// Load physical verify row `row`'s `p` (penalized by history row
    /// `hist_row`) into `self.target`.
    fn load_target(
        &mut self,
        gpu: &mut Gpu,
        bundle: &mut Qwen4Bundle,
        row: usize,
        hist_row: usize,
    ) -> Result<(), String> {
        if self.gpu_rows {
            self.penalize_rows(gpu, bundle, row, hist_row..hist_row + 1)?;
        }
        let (policy, history) = if self.gpu_rows {
            (&self.row_policy, &[][..])
        } else {
            (&self.policy, self.history.row(hist_row))
        };
        bundle
            .spec_row_dist(
                gpu,
                row,
                self.spec,
                history,
                policy,
                &mut self.host,
                &mut self.scratch,
                &mut self.target,
            )
            .map_err(|error| error.to_string())?;
        #[cfg(feature = "lab")]
        if let Some(lab) = self.lab.as_mut() {
            lab.note_row(hist_row, self.target.entries(), false);
        }
        Ok(())
    }

    /// Leviathan acceptance over a batched verify: `drafts[i]` is verified
    /// against `p_i` penalized by `H_i` (physical verify row == history
    /// row; with [`Self::gpu_rows`] every row is penalized up front by one
    /// prepass launch). `q_i` is the distribution the draft was drawn from
    /// (`self.drafts[i]`), or with `point_mass` the point mass on the draft.
    fn accept_leviathan(
        &mut self,
        gpu: &mut Gpu,
        bundle: &mut Qwen4Bundle,
        drafts: &[u32],
        point_mass: bool,
        eos: u32,
    ) -> Result<GreedyAccept, String> {
        self.history.rewind_drafts();
        for &draft in drafts {
            self.history.push_draft(draft);
        }
        if self.gpu_rows {
            self.penalize_rows(gpu, bundle, 0, 0..drafts.len() + 1)?;
        }
        #[cfg(feature = "lab")]
        let lab_before = self.rng;
        if self.device_ready(drafts.len()) {
            return self.accept_leviathan_device(gpu, bundle, drafts, point_mass, eos);
        }
        let SampledVerify {
            spec,
            rng,
            policy,
            row_policy,
            gpu_rows,
            history,
            host,
            scratch,
            target,
            drafts: draft_dists,
            points,
            #[cfg(feature = "lab")]
            lab,
            ..
        } = self;
        let qs = leviathan_qs(&mut *points, &*draft_dists, drafts, point_mass)?;
        let history = &*history;
        let spec = *spec;
        let gpu_rows = *gpu_rows;
        let gpu = &*gpu;
        let bundle = &*bundle;
        let acceptance = accept_sampled_prefix(drafts, qs, Some(eos), rng, target, |row, out| {
            let (policy, row_history) = if gpu_rows {
                (&*row_policy, &[][..])
            } else {
                (&*policy, history.row(row))
            };
            bundle
                .spec_row_dist(gpu, row, spec, row_history, policy, host, scratch, out)
                .map_err(|error| error.to_string())?;
            #[cfg(feature = "lab")]
            if let Some(lab) = lab.as_mut() {
                lab.note_row(row, out.entries(), false);
            }
            Ok(())
        })?;
        #[cfg(feature = "lab")]
        if let Some(lab) = lab.as_mut() {
            lab.note_batched(
                drafts,
                qs,
                point_mass,
                &acceptance.committed,
                acceptance.accepted,
                acceptance.hit_eos,
                &lab_before,
                None,
            );
        }
        Ok(acceptance)
    }

    /// Naive acceptance over a batched verify: row `i`'s draw is penalized
    /// by `H_i` and recorded in `picks[i]`.
    fn accept_naive(
        &mut self,
        gpu: &mut Gpu,
        bundle: &mut Qwen4Bundle,
        drafts: &[u32],
        eos: u32,
        picks: &mut [u32],
    ) -> Result<GreedyAccept, String> {
        self.history.rewind_drafts();
        for &draft in drafts {
            self.history.push_draft(draft);
        }
        if self.gpu_rows {
            self.penalize_rows(gpu, bundle, 0, 0..drafts.len() + 1)?;
        }
        let acceptance = accept_naive_prefix(drafts, Some(eos), |row| {
            let draw = self.draw_penalized(gpu, bundle, row, row)?;
            picks[row] = draw;
            Ok(draw)
        })?;
        #[cfg(feature = "lab")]
        if let Some(lab) = self.lab.as_mut() {
            lab.note_naive_batched(
                drafts,
                &acceptance.committed,
                acceptance.accepted,
                acceptance.hit_eos,
            );
        }
        Ok(acceptance)
    }

    /// Host one-draft verdict of an interleaved window against `self.target`
    /// (`p`, loaded by [`Self::load_target`]) and the window's draft
    /// distribution 0, or with `point_mass` the point mass `points[0]` the
    /// caller set on `draft`. Consumes one uniform on accept, two on reject.
    #[cfg_attr(not(feature = "lab"), allow(clippy::let_and_return))]
    fn host_step_verdict(&mut self, draft: u32, point_mass: bool) -> DraftVerdict {
        let q = if point_mass {
            &self.points[0]
        } else {
            &self.drafts[0]
        };
        #[cfg(feature = "lab")]
        let lab_before = self.rng;
        let verdict = verify_sampled_draft(&self.target, q, draft, &mut self.rng);
        #[cfg(feature = "lab")]
        if let Some(lab) = self.lab.as_mut() {
            lab.note_step(draft, q.entries(), point_mass, verdict, &lab_before);
        }
        verdict
    }

    /// The bonus draw of an all-accepted interleaved window, or the target
    /// draw of a target-only window, from `self.target`: one uniform.
    #[cfg_attr(not(feature = "lab"), allow(clippy::let_and_return))]
    fn host_bonus(&mut self) -> u32 {
        #[cfg(feature = "lab")]
        let lab_before = self.rng;
        let token = self.target.sample(self.rng.next_f32());
        #[cfg(feature = "lab")]
        if let Some(lab) = self.lab.as_mut() {
            lab.note_bonus(token, &lab_before);
        }
        token
    }
}

/// `HIPFIRE_QWEN4_MTP_DEVICE_ACCEPT`: sampled Leviathan windows build their
/// target rows on the device from the verify's resident logits and run the
/// ordered acceptance there (opt-in: only an exact `1` enables; unset or any
/// other value keeps the host rows and host accept). Read once per request
/// by [`SampledVerify::new`].
pub(crate) fn mtp_device_accept_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_QWEN4_MTP_DEVICE_ACCEPT").as_deref() == Ok("1")
}

/// `HIPFIRE_QWEN4_MTP_SAMPLED_PICKS`: a device-route sampled window skips the
/// unused greedy argmax pick download (opt-in: only an exact `1` enables;
/// otherwise the picks download). Read once per request by
/// [`SampledVerify::new`].
fn mtp_sampled_picks_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_QWEN4_MTP_SAMPLED_PICKS").as_deref() == Ok("1")
}

/// Exact supported architecture with the resident device sampler available.
fn mtp_device_sampler_supported(gpu: &Gpu) -> bool {
    (gpu.arch_caps.is_gfx1151() || gpu.arch_caps.is_gfx1201()) && gpu.spec_cpu_ar_device_supported()
}

/// Device-accept eligibility that needs no `Gpu`: Leviathan (not naive), no
/// penalty history window, no GPU penalty prepass, no blocked tokens, a
/// positive finite temperature and a pool/cap the 64-entry device rows hold.
fn device_accept_eligible(
    naive: bool,
    history_window: usize,
    gpu_rows: bool,
    no_blocked_tokens: bool,
    spec: &SampleSpec,
) -> bool {
    !naive
        && history_window == 0
        && !gpu_rows
        && no_blocked_tokens
        && spec.temperature > 0.0
        && spec.temperature.is_finite()
        && (1..=SPEC_CPU_AR_CAP).contains(&spec.pool)
        && (1..=spec.pool).contains(&spec.cap)
}

/// The window's draft distributions: the point mass of every draft, or the
/// distributions the drafts were drawn from.
fn leviathan_qs<'a>(
    points: &'a mut Vec<SparseDist>,
    draft_dists: &'a [SparseDist],
    drafts: &[u32],
    point_mass: bool,
) -> Result<&'a [SparseDist], String> {
    if point_mass {
        if points.len() < drafts.len() {
            points.resize_with(drafts.len(), SparseDist::default);
        }
        for (point, &draft) in points.iter_mut().zip(drafts) {
            point.set_point_mass(draft);
        }
        Ok(&points[..drafts.len()])
    } else {
        draft_dists
            .get(..drafts.len())
            .ok_or_else(|| "Qwen4 sampled MTP: more drafts than K".to_string())
    }
}

/// Advance `rng` by `draws` uniforms (the authoritative stream moves only by
/// the draws a device acceptance consumed).
fn advance_rng(rng: &mut SpecRng, draws: usize) {
    for _ in 0..draws {
        rng.next_u64();
    }
}

/// The next `tape.len()` uniforms of `rng` in order, without advancing it:
/// a clone draws them, so verdict `j` reads `tape[j]`, its residual
/// `tape[j + 1]` and the bonus `tape[k]`, exactly the draws the host accept
/// takes from the authoritative stream.
fn fill_uniform_tape(rng: &SpecRng, tape: &mut [f32]) {
    let mut clone = *rng;
    for slot in tape.iter_mut() {
        *slot = clone.next_f32();
    }
}

/// What the two result words of an ordered device accept mean for a window
/// of `k` proposals, with the real request EOS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DeviceVerdict {
    /// Proposals accepted, `0..=k`.
    accepted: usize,
    /// The residual replacement (rejection) or bonus (all accepted); `None`
    /// when the kernel stopped on an accepted proposal equal to ITS `eos`
    /// argument (no terminal draw).
    terminal: Option<u32>,
    /// Uniforms of the host stream the window consumed: `accepted` after a
    /// stopping accept, `accepted + 2` after a rejection, `k + 1` when every
    /// proposal was accepted. Independent of the real EOS.
    consumed: usize,
    /// The window hit the REQUEST's EOS: the last accepted proposal (stopping
    /// accept) or the terminal token equals `eos`. A stop the kernel took for
    /// a proposal that is not the request EOS (the incremental one-row
    /// verdict passes `eos = proposal`) is not an EOS hit.
    hit_eos: bool,
}

/// Decode the kernel's `[accepted, terminal]` words for `drafts` against the
/// request's `eos`. `terminal == u32::MAX` is the stopping accept.
fn decode_device_verdict(
    drafts: &[u32],
    words: [u32; 2],
    eos: u32,
) -> Result<DeviceVerdict, String> {
    let k = drafts.len();
    let accepted = words[0] as usize;
    if accepted > k {
        return Err(format!(
            "Qwen4 sampled MTP device accept reported {accepted} accepted of {k} proposals"
        ));
    }
    if words[1] == SPEC_CPU_AR_EOS_NONE {
        if accepted == 0 {
            return Err(
                "Qwen4 sampled MTP device accept stopped with no accepted proposal".to_string(),
            );
        }
        return Ok(DeviceVerdict {
            accepted,
            terminal: None,
            consumed: accepted,
            hit_eos: drafts[accepted - 1] == eos,
        });
    }
    Ok(DeviceVerdict {
        accepted,
        terminal: Some(words[1]),
        consumed: if accepted < k { accepted + 2 } else { k + 1 },
        hit_eos: words[1] == eos,
    })
}

/// The committed tokens of a decoded window: `drafts[..accepted]` then the
/// terminal token, when there is one.
fn device_accept_result(drafts: &[u32], verdict: &DeviceVerdict) -> GreedyAccept {
    let mut committed = Vec::with_capacity(drafts.len() + 1);
    committed.extend_from_slice(&drafts[..verdict.accepted]);
    committed.extend(verdict.terminal);
    GreedyAccept {
        committed,
        accepted: verdict.accepted,
        hit_eos: verdict.hit_eos,
    }
}

/// `accept_route` field of the window traces: `device_parallel` when the
/// window built target rows or accepted on the device, else `host` (greedy,
/// naive, ineligible, unsupported and `HIPFIRE_QWEN4_MTP_DEVICE_ACCEPT` not `1`).
fn accept_route_label(sampled: Option<&SampledVerify>) -> &'static str {
    if sampled.is_some_and(|s| s.route_device) {
        "device_parallel"
    } else {
        "host"
    }
}

fn u32_bytes(words: &[u32]) -> &[u8] {
    // SAFETY: any `u32` slice is valid as `len * 4` initialized bytes.
    unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words)) }
}

fn f32_bytes(words: &[f32]) -> &[u8] {
    // SAFETY: any `f32` slice is valid as `len * 4` initialized bytes.
    unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words)) }
}

/// Whether queued device work may still read or write the host staging bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StageState {
    Idle,
    /// Copies or kernels were queued and no fence covers them yet.
    Enqueued,
    /// The fence event was recorded behind every queued copy, kernel and
    /// result readback.
    Fenced,
}

/// Host staging of one device accept's uploads and result readback, reused
/// by every window (allocated once with the buffers). Dropped while queued
/// work may still touch it, its memory is leaked instead of freed.
struct DeviceStaging {
    counts: Vec<u32>,
    ids: Vec<u32>,
    probabilities: Vec<f32>,
    proposals: Vec<u32>,
    tape: Vec<f32>,
    /// The 8 result bytes: the stable destination of the asynchronous
    /// readback on an active stream.
    result: Vec<u8>,
    state: StageState,
}

impl Drop for DeviceStaging {
    fn drop(&mut self) {
        if self.state != StageState::Idle {
            std::mem::forget(std::mem::take(&mut self.counts));
            std::mem::forget(std::mem::take(&mut self.ids));
            std::mem::forget(std::mem::take(&mut self.probabilities));
            std::mem::forget(std::mem::take(&mut self.proposals));
            std::mem::forget(std::mem::take(&mut self.tape));
            std::mem::forget(std::mem::take(&mut self.result));
        }
    }
}

/// Device buffers and host staging of the resident Leviathan acceptance,
/// sized once for `capacity` proposals (`capacity + 1` target rows) and
/// reused by every window and every later request of the drafter. Allocated
/// by `ensure_resources` outside the timed window and any graph capture.
struct DeviceAccept {
    /// Target rows `0..=capacity`.
    target: SpecDistRows,
    /// Draft rows `0..capacity`.
    draft: SpecDistRows,
    /// `u32[capacity]`.
    proposals: GpuTensor,
    /// `f32[capacity + 1]`.
    uniforms: GpuTensor,
    /// `u32[2]`: accepted count, terminal token.
    result: GpuTensor,
    capacity: usize,
    staging: DeviceStaging,
    /// Recorded behind the window's queued copies, accept kernel and (on an
    /// active stream) result readback; waited on only to retire an
    /// interrupted window.
    fence: Option<hip_bridge::Event>,
}

impl DeviceAccept {
    fn new(gpu: &mut Gpu, capacity: usize) -> Result<Self, String> {
        let capacity = capacity.max(1);
        let fail = |what: &str, error: &dyn std::fmt::Display| {
            format!("Qwen4 sampled MTP device accept {what}: {error}")
        };
        gpu.spec_cpu_ar_precompile()
            .map_err(|error| fail("precompile", &error))?;
        gpu.spec_cpu_ar_rows_parallel_precompile()
            .map_err(|error| fail("rows precompile", &error))?;
        let target_rows = capacity + 1;
        let cap = SPEC_CPU_AR_CAP;
        // `u32`/`f32` words per buffer, in the order consumed below; the
        // crate has no integer dtype, so every buffer is an F32-typed tensor.
        let words = [
            target_rows,
            target_rows * cap,
            target_rows * cap,
            capacity,
            capacity * cap,
            capacity * cap,
            capacity,
            capacity + 1,
            2,
        ];
        let mut tensors: Vec<GpuTensor> = Vec::with_capacity(words.len());
        for (index, &n) in words.iter().enumerate() {
            let tensor = if matches!(index, 2 | 5 | 7) {
                spec_cpu_ar_alloc_f32(gpu, n)
            } else {
                spec_cpu_ar_alloc_u32(gpu, n)
            };
            match tensor {
                Ok(tensor) => tensors.push(tensor),
                Err(error) => {
                    for tensor in tensors {
                        let _ = gpu.free_tensor(tensor);
                    }
                    return Err(fail("allocation", &error));
                }
            }
        }
        let fence = match gpu
            .hip
            .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)
        {
            Ok(fence) => fence,
            Err(error) => {
                for tensor in tensors {
                    let _ = gpu.free_tensor(tensor);
                }
                return Err(fail("fence", &error));
            }
        };
        let mut it = tensors.into_iter();
        let mut next = || {
            it.next()
                .expect("nine device accept buffers allocated above")
        };
        let target = SpecDistRows {
            counts: next(),
            ids: next(),
            probabilities: next(),
            capacity_rows: target_rows,
        };
        let draft = SpecDistRows {
            counts: next(),
            ids: next(),
            probabilities: next(),
            capacity_rows: capacity,
        };
        let proposals = next();
        let uniforms = next();
        let result = next();
        Ok(Self {
            target,
            draft,
            proposals,
            uniforms,
            result,
            capacity,
            staging: DeviceStaging {
                counts: vec![0; capacity],
                ids: vec![0; capacity * cap],
                probabilities: vec![0.0; capacity * cap],
                proposals: vec![0; capacity],
                tape: vec![0.0; capacity + 1],
                result: vec![0; 8],
                state: StageState::Idle,
            },
            fence: Some(fence),
        })
    }

    /// Wait until no queued work touches the staging bytes. On failure the
    /// state is unchanged, so the bytes stay unwritable and the next call
    /// retries. Idle after every successful accept (its result wait already
    /// ordered everything), so the normal flow waits on nothing here.
    fn drain(&mut self, gpu: &Gpu) -> Result<(), String> {
        let wait = match (self.staging.state, self.fence.as_ref()) {
            (StageState::Idle, _) => Ok(()),
            (StageState::Fenced, Some(fence)) => gpu.hip.event_synchronize(fence),
            // An unfenced enqueue (the fence record failed) or a lost fence:
            // drain the whole device rather than trust the stream order.
            _ => gpu.hip.device_synchronize(),
        };
        wait.map_err(|error| format!("Qwen4 sampled MTP device accept staging drain: {error}"))?;
        self.staging.state = StageState::Idle;
        Ok(())
    }

    /// Release the device buffers after retiring any pending copy. A drain
    /// that fails leaks everything (nothing proves the device stopped reading).
    fn free(mut self, gpu: &mut Gpu) {
        if self.drain(gpu).is_err() {
            std::mem::forget(self);
            return;
        }
        let Self {
            target,
            draft,
            proposals,
            uniforms,
            result,
            fence,
            ..
        } = self;
        let _ = target.free(gpu);
        let _ = draft.free(gpu);
        let _ = gpu.free_tensor(proposals);
        let _ = gpu.free_tensor(uniforms);
        let _ = gpu.free_tensor(result);
        if let Some(fence) = fence {
            let _ = gpu.hip.event_destroy(fence);
        }
    }

    /// Enqueue the parallel target-row builder over `rows` resident logit
    /// rows into target rows `0..rows`. Stream-ordered; no readback.
    fn build_rows(
        &self,
        gpu: &mut Gpu,
        logits: &GpuTensor,
        vocab: usize,
        rows: usize,
        policy: SpecCpuArPolicy,
    ) -> Result<(), String> {
        if rows > self.target.capacity_rows {
            return Err(format!(
                "Qwen4 sampled MTP device accept needs {rows} target rows (capacity {})",
                self.target.capacity_rows
            ));
        }
        gpu.spec_cpu_ar_rows_parallel_enqueue(logits, &self.target, vocab, rows, policy)
            .map_err(|error| format!("Qwen4 sampled MTP device target rows: {error}"))
    }

    /// Read target rows `0..rows` of the resident device distributions back
    /// as `SparseDist::entries()`-ordered rows (blocking). Lab only: the
    /// production accept never reads the rows back.
    #[cfg(feature = "lab")]
    fn lab_target_rows(
        &self,
        gpu: &Gpu,
        rows: usize,
    ) -> Result<Vec<crate::lab_window::LabDist>, String> {
        if rows > self.target.capacity_rows {
            return Err(format!(
                "Qwen4 sampled MTP lab target rows: {rows} exceed capacity {}",
                self.target.capacity_rows
            ));
        }
        let cap = SPEC_CPU_AR_CAP;
        let read = |tensor: &GpuTensor| {
            gpu.download_raw_bytes(tensor)
                .map_err(|error| format!("Qwen4 sampled MTP lab target rows readback: {error}"))
        };
        let counts = read(&self.target.counts)?;
        let ids = read(&self.target.ids)?;
        let probabilities = read(&self.target.probabilities)?;
        let word = |bytes: &[u8], index: usize| {
            [
                bytes[4 * index],
                bytes[4 * index + 1],
                bytes[4 * index + 2],
                bytes[4 * index + 3],
            ]
        };
        (0..rows)
            .map(|row| {
                let count = u32::from_ne_bytes(word(&counts, row)) as usize;
                if count > cap {
                    return Err(format!(
                        "Qwen4 sampled MTP lab target row {row} holds {count} entries (cap {cap})"
                    ));
                }
                Ok((0..count)
                    .map(|slot| {
                        let at = row * cap + slot;
                        (
                            u32::from_ne_bytes(word(&ids, at)),
                            f32::from_ne_bytes(word(&probabilities, at)),
                        )
                    })
                    .collect())
            })
            .collect()
    }

    /// One ordered accept over the resident target rows `0..=k`: uploads the
    /// draft rows `qs`, the `proposals` and the next `k + 1` uniforms of `rng`
    /// (the stream itself is not advanced) with stream-ordered asynchronous
    /// copies, enqueues the accept kernel, records the fence behind it (and
    /// behind the queued readback on an active stream) and reads the two
    /// result words back: one blocking `memcpy_dtoh_auto` on the default
    /// stream, or `memcpy_dtoh_async` plus one `stream_synchronize` on an
    /// active stream. That readback is the window's only host wait.
    /// `k == 0` is the bonus draw from target row 0. `eos` is the kernel's
    /// stop token (`SPEC_CPU_AR_EOS_NONE` stops at none).
    fn accept(
        &mut self,
        gpu: &mut Gpu,
        qs: &[SparseDist],
        proposals: &[u32],
        rng: &SpecRng,
        eos: u32,
    ) -> Result<[u32; 2], String> {
        let k = proposals.len();
        if qs.len() != k || k > self.capacity {
            return Err(format!(
                "Qwen4 sampled MTP device accept got {} draft rows for {k} proposals (capacity {})",
                qs.len(),
                self.capacity
            ));
        }
        self.drain(gpu)?;
        let cap = SPEC_CPU_AR_CAP;
        let staging = &mut self.staging;
        for (row, q) in qs.iter().enumerate() {
            let entries = q.entries();
            if entries.len() > cap {
                return Err(format!(
                    "Qwen4 sampled MTP draft row {row} has {} entries (device rows hold {cap})",
                    entries.len()
                ));
            }
            let base = row * cap;
            staging.counts[row] = entries.len() as u32;
            for (slot, &(token, probability)) in entries.iter().enumerate() {
                staging.ids[base + slot] = token;
                staging.probabilities[base + slot] = probability;
            }
            staging.ids[base + entries.len()..base + cap].fill(0);
            staging.probabilities[base + entries.len()..base + cap].fill(0.0);
        }
        staging.proposals[..k].copy_from_slice(proposals);
        fill_uniform_tape(rng, &mut staging.tape[..k + 1]);
        gpu.bind_thread()
            .map_err(|error| format!("Qwen4 sampled MTP device accept bind: {error}"))?;
        // Pending from before the first queued copy until the fence is
        // recorded behind every queued operation.
        staging.state = StageState::Enqueued;
        let upload = |gpu: &Gpu, dst: &GpuTensor, src: &[u8]| -> Result<(), String> {
            match gpu.active_stream.as_ref() {
                Some(stream) => gpu.hip.memcpy_htod_async(&dst.buf, src, stream),
                None => gpu.hip.memcpy_htod_async_default(&dst.buf, src),
            }
            .map_err(|error| format!("Qwen4 sampled MTP device accept upload: {error}"))
        };
        if k > 0 {
            upload(gpu, &self.draft.counts, u32_bytes(&staging.counts[..k]))?;
            upload(gpu, &self.draft.ids, u32_bytes(&staging.ids[..k * cap]))?;
            upload(
                gpu,
                &self.draft.probabilities,
                f32_bytes(&staging.probabilities[..k * cap]),
            )?;
            upload(gpu, &self.proposals, u32_bytes(&staging.proposals[..k]))?;
        }
        upload(gpu, &self.uniforms, f32_bytes(&staging.tape[..k + 1]))?;
        gpu.spec_accept_ordered_enqueue(
            &self.target,
            &self.draft,
            &self.proposals,
            &self.uniforms,
            &self.result,
            k,
            eos,
        )
        .map_err(|error| format!("Qwen4 sampled MTP device accept: {error}"))?;
        let fence = self
            .fence
            .as_ref()
            .ok_or_else(|| "Qwen4 sampled MTP device accept fence is missing".to_string())?;
        match gpu.active_stream.as_ref() {
            Some(stream) => {
                gpu.hip
                    .memcpy_dtoh_async(&mut staging.result, &self.result.buf, Some(stream))
                    .map_err(|error| {
                        format!("Qwen4 sampled MTP device accept result queue: {error}")
                    })?;
                gpu.hip.event_record(fence, Some(stream)).map_err(|error| {
                    format!("Qwen4 sampled MTP device accept fence record: {error}")
                })?;
                staging.state = StageState::Fenced;
                gpu.hip.stream_synchronize(stream).map_err(|error| {
                    format!("Qwen4 sampled MTP device accept result wait: {error}")
                })?;
            }
            None => {
                gpu.hip.event_record(fence, None).map_err(|error| {
                    format!("Qwen4 sampled MTP device accept fence record: {error}")
                })?;
                staging.state = StageState::Fenced;
                gpu.memcpy_dtoh_auto(&mut staging.result, &self.result.buf)
                    .map_err(|error| format!("Qwen4 sampled MTP device accept result: {error}"))?;
            }
        }
        // The result wait ordered every queued copy, the kernel and the
        // readback itself: nothing is left to drain.
        staging.state = StageState::Idle;
        // The target rows this accept consumed came from the router producers; a
        // non-finite router row recorded its (layer, row) in the device fault word.
        gpu.take_device_wait_error()
            .map_err(|error| format!("Qwen4 sampled MTP device accept: {error}"))?;
        let bytes = &staging.result;
        Ok([
            u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            u32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        ])
    }
}

impl SampledVerify {
    /// This window of `k` proposals runs on the device route: a Leviathan
    /// request that passed [`device_accept_eligible`], has its resident
    /// buffers provisioned (exact supported architecture) and `1 <= k <=` the
    /// buffers' capacity. `k == 0` windows (the target-only draw) stay host.
    /// The private verify/enqueue paths select on this: the resident logits
    /// are consumed by `device_load_target` / `accept_leviathan`.
    fn device_ready(&self, k: usize) -> bool {
        k >= 1
            && self.draws_drafts()
            && self.device_eligible
            && self
                .device
                .as_ref()
                .is_some_and(|device| k <= device.capacity)
    }

    /// This window's per-row greedy picks are unused: the device route
    /// consumes the resident logits, the `HIPFIRE_QWEN4_MTP_SAMPLED_PICKS`
    /// lever is on, and `keep_picks` (trace or phase timers, which print the
    /// rows) is off. The verify / row forward then skips the pick download.
    fn device_skip_picks(&self, k: usize, keep_picks: bool) -> bool {
        !keep_picks && self.sampled_picks && self.device_ready(k)
    }

    fn device_policy(&self) -> SpecCpuArPolicy {
        SpecCpuArPolicy::new(
            self.spec.temperature,
            self.spec.top_p,
            self.spec.pool as u32,
            self.spec.cap as u32,
            self.spec.min_p,
        )
    }

    /// Device twin of [`Self::load_target`] for the one-row interleaved
    /// windows: build physical verify row `row`'s `p` into device target row
    /// 0, stream-ordered after the forward that wrote it. No readback.
    fn device_load_target(
        &mut self,
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        row: usize,
    ) -> Result<(), String> {
        let policy = self.device_policy();
        let logits = bundle
            .spec_logits_rows(row, 1)
            .map_err(|error| error.to_string())?;
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| "Qwen4 sampled MTP device accept is not provisioned".to_string())?;
        device.build_rows(gpu, &logits, bundle.config.vocab_size, 1, policy)?;
        self.route_device = true;
        Ok(())
    }

    /// The one-row verdict of an interleaved window over the device target
    /// row 0 ([`Self::device_load_target`]): `draft` against its `q` (the
    /// window's draft distribution 0, or with `point_mass` the point mass on
    /// `draft`). The kernel's stop token is the proposal itself, so an
    /// accepted proposal never reads a bonus row; the caller applies the real
    /// EOS rule. Consumes one uniform on accept, two on reject.
    fn device_step_verdict(
        &mut self,
        gpu: &mut Gpu,
        draft: u32,
        point_mass: bool,
    ) -> Result<DraftVerdict, String> {
        let SampledVerify {
            device,
            rng,
            drafts,
            points,
            route_device,
            #[cfg(feature = "lab")]
            lab,
            ..
        } = self;
        let q = if point_mass {
            if points.is_empty() {
                points.push(SparseDist::default());
            }
            points[0].set_point_mass(draft);
            &points[0]
        } else {
            drafts
                .first()
                .ok_or_else(|| "Qwen4 sampled MTP has no draft distribution".to_string())?
        };
        let device = device
            .as_mut()
            .ok_or_else(|| "Qwen4 sampled MTP device accept is not provisioned".to_string())?;
        #[cfg(feature = "lab")]
        let lab_before = *rng;
        let words = device.accept(gpu, std::slice::from_ref(q), &[draft], rng, draft)?;
        let verdict = decode_device_verdict(&[draft], words, SPEC_CPU_AR_EOS_NONE)?;
        advance_rng(rng, verdict.consumed);
        *route_device = true;
        #[cfg(feature = "lab")]
        if let Some(lab) = lab.as_mut() {
            let rows = device.lab_target_rows(&*gpu, 1)?;
            lab.device_route = true;
            lab.device_words.push(words);
            let row = lab.rows.len();
            lab.note_row(row, &rows[0], true);
            let observed = match verdict.terminal {
                None => DraftVerdict::Accept,
                Some(token) => DraftVerdict::Reject(token),
            };
            lab.note_step(draft, q.entries(), point_mass, observed, &lab_before);
        }
        Ok(match verdict.terminal {
            None => DraftVerdict::Accept,
            Some(token) => DraftVerdict::Reject(token),
        })
    }

    /// The bonus draw of an interleaved window from device target row 0
    /// ([`Self::device_load_target`]): one uniform.
    fn device_bonus(&mut self, gpu: &mut Gpu) -> Result<u32, String> {
        let SampledVerify {
            device,
            rng,
            route_device,
            #[cfg(feature = "lab")]
            lab,
            ..
        } = self;
        let device = device
            .as_mut()
            .ok_or_else(|| "Qwen4 sampled MTP device accept is not provisioned".to_string())?;
        #[cfg(feature = "lab")]
        let lab_before = *rng;
        let words = device.accept(gpu, &[], &[], rng, SPEC_CPU_AR_EOS_NONE)?;
        let verdict = decode_device_verdict(&[], words, SPEC_CPU_AR_EOS_NONE)?;
        advance_rng(rng, verdict.consumed);
        *route_device = true;
        #[cfg(feature = "lab")]
        if let Some(lab) = lab.as_mut() {
            let rows = device.lab_target_rows(&*gpu, 1)?;
            lab.device_route = true;
            lab.device_words.push(words);
            let row = lab.rows.len();
            lab.note_row(row, &rows[0], true);
            if let Some(token) = verdict.terminal {
                lab.note_bonus(token, &lab_before);
            }
        }
        verdict
            .terminal
            .ok_or_else(|| "Qwen4 sampled MTP device bonus returned no token".to_string())
    }

    /// Device twin of the host branch of [`Self::accept_leviathan`]: every
    /// target row `0..=k` is built from the resident verify logits in one
    /// launch, the draft rows, proposals and uniform tape are uploaded
    /// stream-ordered, and one 8-byte result is read back. The authoritative
    /// stream advances only by the consumed draws.
    fn accept_leviathan_device(
        &mut self,
        gpu: &mut Gpu,
        bundle: &mut Qwen4Bundle,
        drafts: &[u32],
        point_mass: bool,
        eos: u32,
    ) -> Result<GreedyAccept, String> {
        let k = drafts.len();
        let policy = self.device_policy();
        let logits = bundle
            .spec_logits_rows(0, k + 1)
            .map_err(|error| error.to_string())?;
        let vocab = bundle.config.vocab_size;
        let SampledVerify {
            device,
            rng,
            drafts: draft_dists,
            points,
            route_device,
            #[cfg(feature = "lab")]
            lab,
            ..
        } = self;
        let qs = leviathan_qs(&mut *points, &*draft_dists, drafts, point_mass)?;
        let device = device
            .as_mut()
            .ok_or_else(|| "Qwen4 sampled MTP device accept is not provisioned".to_string())?;
        device.build_rows(gpu, &logits, vocab, k + 1, policy)?;
        #[cfg(feature = "lab")]
        let lab_before = *rng;
        let words = device.accept(gpu, qs, drafts, rng, eos)?;
        let verdict = decode_device_verdict(drafts, words, eos)?;
        advance_rng(rng, verdict.consumed);
        *route_device = true;
        #[cfg(feature = "lab")]
        if let Some(lab) = lab.as_mut() {
            // The rows the ordered accept read: through the rejection (or
            // bonus) row, or only the accepted rows of a stopping accept.
            let read = verdict.accepted + usize::from(verdict.terminal.is_some());
            for (row, p) in device.lab_target_rows(&*gpu, read)?.iter().enumerate() {
                lab.note_row(row, p, true);
            }
            lab.device_route = true;
            let observed = device_accept_result(drafts, &verdict);
            lab.note_batched(
                drafts,
                qs,
                point_mass,
                &observed.committed,
                observed.accepted,
                observed.hit_eos,
                &lab_before,
                Some(words),
            );
        }
        Ok(device_accept_result(drafts, &verdict))
    }
}

/// What a promoted singleton's drafter hands its lane
/// ([`Qwen4MtpDrafter::promotion_parts`]).
pub struct Qwen4PromotionParts<'a> {
    /// The next window's row-0 hidden (F32, `hc_count * hidden`).
    pub pending_hidden: &'a GpuTensor,
    /// Full request-capacity agreement (performance only), borrowed until the
    /// receiving lane takes ownership at promotion.
    pub agreement: &'a [(f32, f32)],
    /// The floor retired the request: the head lags, the lane must stay
    /// `ArRetired`.
    pub retired: bool,
}

/// The route of one native window, chosen on rank 0 by
/// [`Qwen4MtpDrafter::plan_window`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WindowRoute {
    /// The floor retired the request before this window: the head-free AR
    /// token, with no resource pass.
    ArRetired,
    /// The floor chose AR (retiring the request) while routing this window:
    /// the head-free AR token after the resource pass.
    ArRouted,
    /// The request's first window: the AR calibration token after the
    /// resource pass.
    ArCalibrate,
    /// A native window: `depth` 0 is the interleaved route, otherwise the
    /// batched route drafting at most `depth` rows.
    Native { depth: usize },
}

/// Every timing- or host-state-dependent decision of one native window, made
/// once on rank 0 (the floor and the n-gram gate read wall clocks and request
/// history): the route, the draft budget, the n-gram hit and the forced-route
/// and trace environment. [`Qwen4MtpDrafter::run_window`] runs a plan on every
/// rank and takes nothing else from those states, so all ranks issue the same
/// device ops.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WindowPlan {
    pub(crate) route: WindowRoute,
    /// The native draft budget: the caller's `k` clamped to the head's `max_k`.
    pub(crate) k: usize,
    pub(crate) trace: bool,
    /// The floor is neither bypassed by a forced route nor opted out.
    pub(crate) floor_active: bool,
    /// The pool proposed a takeover this window (the head's first draft still
    /// has to confirm it). With the copy lane the take-length chooser
    /// accepted it as well.
    pub(crate) ngram_hit: bool,
    /// The head's row-0 first draft must equal the first candidate before
    /// the takeover runs: always so without the copy lane, and with it until
    /// the copy source is established ([`NGRAM_UNCONFIRMED_MIN_TAKE`]).
    /// Meaningful only with `ngram_hit`.
    pub(crate) ngram_confirm: bool,
    /// The takeover candidates. Filled only for replication
    /// ([`Qwen4MtpDrafter::attach_candidates`]); rank 0 reads its own pool copy.
    pub(crate) ngram_candidates: Vec<u32>,
}

/// A target-less drafter call rank 0 ran, owed to every EP peer's drafter at
/// the start of the next fan-out. `mtp_reset_for_realign` is `mtp_reset` for
/// this drafter, so it queues [`ReplayOp::Reset`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ReplayOp {
    Configure(SpecRequestConfig),
    Reset,
}

/// What a peer's drafter is built from: rank 0's construction parameters.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReplicaParams {
    max_k: usize,
    ctx_capacity: usize,
    end_of_turn: Option<u32>,
    ngram_config: Option<NgramModConfig>,
    copy_lane: bool,
    sampled_enabled: bool,
}

/// The host-visible part of a window every EP rank must agree on:
/// `(committed, accepted, drafts_generated)`.
fn window_key(window: Result<MtpWindow, String>) -> Result<(Vec<u32>, usize, usize), String> {
    window.map(|window| (window.committed, window.accepted, window.drafts_generated))
}

/// The head rows a retired request owes. Once the floor retires a request,
/// its AR tokens advance the target alone (no head work per token); each
/// token's target wide hidden is captured into spec hidden row `i` while the
/// head stays at `start`. [`Qwen4MtpDrafter::catch_up_head`] appends the owed
/// rows teacher-forced (token `p` with the target hidden of `p - 1`, as a
/// takeover appends its kept rows), when they fill
/// one batched fill and before every teacher-forced advance (the terminal
/// pending-seed flush among them). A retired request therefore still ends
/// with its head at the target, and the conversation's next turn continues
/// live.
///
/// While rows are owed nothing else writes spec hidden: a retired request
/// runs only head-free AR tokens until it catches up, and every reset or
/// prefill drops the record (its owners are reset or restored). Nothing
/// writes the drafter's pending hidden either: it stays the hidden of
/// `start - 1` (the head's position - 1), which owed token 0 pairs with
/// (token `i` pairs with spec hidden row `i - 1`); the catch-up ends it on
/// row `owed - 1`. Host only.
#[derive(Debug, Default)]
struct HeadLag {
    /// The head position: where the first owed row goes.
    start: usize,
    /// The owed tokens, at positions `start..start + tokens.len()`; token
    /// `i`'s own target hidden is spec hidden row `i` (the pair of token
    /// `i + 1`).
    tokens: Vec<u32>,
}

impl HeadLag {
    /// The spec hidden row the AR token at `position` captures into: the
    /// next row. Refused unless `position` continues the owed rows and a row
    /// below `capacity` is free.
    fn next_row(&self, position: usize, capacity: usize) -> Result<usize, String> {
        let row = self.tokens.len();
        if row != 0 && self.start + row != position {
            return Err(format!(
                "Qwen4 MTP head lag: AR token at {position} does not continue the owed rows {}..{}",
                self.start,
                self.start + row
            ));
        }
        if row >= capacity {
            return Err(format!(
                "Qwen4 MTP head lag: {row} owed rows leave no free row of {capacity}"
            ));
        }
        Ok(row)
    }

    /// Owe the token at `position`, whose hidden went to [`Self::next_row`].
    fn push(&mut self, position: usize, token: u32) {
        debug_assert!(self.tokens.is_empty() || self.start + self.tokens.len() == position);
        if self.tokens.is_empty() {
            self.start = position;
        }
        self.tokens.push(token);
    }

    fn clear(&mut self) {
        self.tokens.clear();
    }
}

/// Where a committed window sat: its first row's position (`start`, the
/// seed's), the position of the owners after it (`end`) and its seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WindowSpan {
    start: usize,
    end: usize,
    seed: u32,
}

/// The last committed window and the two pre-window tickets whose arena
/// entries were kept active past the commit ([`Qwen4Bundle::retain_window`]),
/// so a terminal stop inside the window can restore both owners to its start
/// and replay only the consumed prefix. Held until the next window, forced
/// advance, prefill, reset or request configuration; the next snapshot, reset
/// or session restore of either owner also supersedes the tickets, which then
/// stop validating. The drafter's `retained_pending` holds the window's
/// pre-window pending hidden (the target hidden of `span.start - 1`, which
/// the seed's head row pairs with): the replay needs it after the window
/// moved `pending_hidden` on.
#[derive(Clone, Copy)]
struct RetainedWindow {
    span: WindowSpan,
    target: TargetTicket,
    head: HeadTicket,
}

/// The tokens a terminal repair replays after restoring `span`'s pre-window
/// state: `seed` plus all but the last of `consumed` (the last stays pending
/// for the terminal flush). `None` when the repair does not apply: it is for
/// another window, nothing was consumed (there is nothing to replay), or the
/// replay would pass the rows the window committed.
fn terminal_repair_replay(
    span: WindowSpan,
    window_start: usize,
    window_seed: u32,
    consumed: &[u32],
) -> Option<SmallVec<[u32; 8]>> {
    if consumed.is_empty() || (span.start, span.seed) != (window_start, window_seed) {
        return None;
    }
    let replay = terminal_prefix_replay(window_seed, consumed);
    (span.start + replay.len() <= span.end).then_some(replay)
}

/// Native GPU MTP drafter.  The MTP operator/state stay model-owned by the
/// target bundle; this adapter owns only the reusable verifier scratch, the
/// pending target-hidden row that seeds each MTP window (the hidden of the
/// head's position - 1) and a pre-window copy of it for a terminal repair.
pub struct Qwen4MtpDrafter {
    max_k: usize,
    /// Only a user-specified K limits the derived policy; the load default
    /// remains the incumbent chooser's starting point.
    explicit_mtp_k: Option<usize>,
    ctx_capacity: usize,
    /// Rows a 2+ row verify forward of the target runs row-exact on this GPU
    /// ([`Qwen4Bundle::row_exact_verify_rows`]); the copy lane's n-gram take
    /// cap is one below it. Starts at the few-row limit and, with
    /// `copy_lane`, is refreshed at every window entry that has the GPU and
    /// the target.
    row_exact_rows: usize,
    /// Most rows (the seed included) one speculative verify or replay of the
    /// target may take ([`Qwen4Bundle::spec_verify_rows_cap`]): the CPU
    /// splice window on an armed load, unbounded otherwise. Latched from the
    /// target at every window entry and resource pass
    /// ([`Self::bind_spec_rows`]); it bounds the proposal capacity, so the
    /// caller's draft budget, the copy-lane take and every n-gram takeover
    /// stay inside it.
    spec_rows_cap: usize,
    request: SpecRequestConfig,
    scratch: Option<Box<dyn SpecScratch>>,
    pending_hidden: Option<GpuTensor>,
    row_hidden: Option<GpuTensor>,
    /// Prompt rows one chunked prefill call may capture; mirrors the attached
    /// forward's chunk capacity and sizes the spec hidden capture buffer.
    prefill_rows: usize,
    /// Recent agreement per draft depth as decayed (accepted, compared)
    /// counts (see `observe_agreement`); picks the verify route and draft
    /// depth when `HIPFIRE_MTP_INCREMENTAL` is unset.
    agreement: Vec<(f32, f32)>,
    /// `speculation.mtp_sampled`, resolved at construction.
    sampled_enabled: bool,
    /// `HIPFIRE_MTP_SAMPLED_MODE`, resolved at construction; an invalid
    /// value fails sampled requests with its message.
    sampled_mode: Result<SampledMode, String>,
    /// The current request's sampled verification; `None` for greedy.
    sampled: Option<SampledVerify>,
    /// Row-batched operator buffers of the batched prompt-fill Append pass
    /// (`HIPFIRE_QWEN4_MTP_BATCHED_FILL`); `None` where the pass cannot run
    /// (see `Qwen4MtpGpu::append_rows_supported`) or has not been enabled yet.
    append_scratch: Option<MtpAppendScratch>,
    /// Chat end-of-turn token (`<|im_end|>`). Windows stop at it instead of
    /// the config EOS so an accepted end-of-turn stays pending for the
    /// terminal flush: committing the draft tail past it leaves a
    /// strict-prefix terminal this drafter cannot repair (the spec terminal
    /// reset path).
    end_of_turn: Option<u32>,
    /// Tokenizer's `</think>` token; only newly emitted closes may re-arm.
    think_close: Option<u32>,
    /// History length at the last plan; the first plan observes only its seed.
    emitted_seen: Option<usize>,
    /// N-gram pool configuration resolved at load for this GPU
    /// (`hipfire_config::ngram_mod_triple_for_arch`); `None` disables the
    /// n-gram takeover for every request.
    ngram_config: Option<NgramModConfig>,
    /// The copy lane (`hipfire_config::mtp_ngram_copy_enabled_for_arch`,
    /// [`Self::with_copy_lane`]) runs the n-gram source only when set, and
    /// only with `ngram_config` and a request that arms n-gram. Unset: the
    /// native-MTP n-gram takeover (`ngram`, the yield gate `ngram_yield`, the
    /// head's row-0 confirmation of every takeover).
    copy_lane: bool,
    /// Request-local n-gram proposal owner of the takeover without the copy
    /// lane, allocated on the first armed request and reused (cleared and
    /// reseeded) by every later one.
    ngram: Option<MtpNgramContext>,
    /// Request-local n-gram proposal owner of the copy lane, allocated on the
    /// first armed request and reused (cleared and reseeded) by every later
    /// one. Only the pool of the active mode is ever allocated.
    ngram_copy: Option<MtpCopyContext>,
    /// This request may take a window from the n-gram pool
    /// (`SpecRequestConfig::allow_ngram_modifier`). Without the copy lane a
    /// pool hit takes the window only if the head's own first draft equals
    /// the hit's first candidate and the yield gate passes; otherwise the
    /// window is native. With it a hit takes the window when the take-length
    /// chooser ([`Self::ngram_hit`]) prices a takeover at least as well as
    /// the native window it displaces; the head's own first draft must then
    /// equal the first candidate, unless the copy source is established
    /// ([`NGRAM_UNCONFIRMED_MIN_TAKE`]).
    ngram_active: bool,
    /// The current window's pool candidates, copied out of the pool (reused).
    /// Valid only when `ngram_hit` returned true for the window.
    ngram_candidates: Vec<u32>,
    /// Decayed (emitted tokens, windows) of this request's takeover windows:
    /// the n-gram source's own yield, kept apart from the native per-depth
    /// agreement (see `ngram_wins`). Unused by the copy lane.
    ngram_yield: (f32, f32),
    /// Copy lane: this request's decayed (accepted, compared) counts of the n-gram
    /// source per candidate depth, the pool's own acceptance kept apart from
    /// the native per-depth agreement and observed by shadow on every window
    /// that had a proposal, whichever source ran it
    /// ([`Self::observe_ngram_shadow`]).
    ngram_agreement: Vec<(f32, f32)>,
    /// The full pool proposal (up to `n_max`) of the last plan that had one
    /// (reused buffer).
    ngram_proposal: Vec<u32>,
    /// `emitted.len()` at the plan that made `ngram_proposal`, until the
    /// next plan has compared it with the committed tokens.
    ngram_proposed_at: Option<usize>,
    /// The last shadow observation accepted its first candidate: the copy
    /// source is still on the text being emitted. An unconfirmed takeover
    /// needs it, so a divergence costs at most one blind window.
    ngram_last_first_accepted: bool,
    /// What the last [`Self::ngram_hit`] decided, for the plan and its trace;
    /// `None` when the pool was not asked or made no proposal.
    ngram_decision: Option<NgramDecision>,
    /// Bit `d`: this drafter already ran a takeover of `d` candidates, so
    /// that row count's one-time verify work is done (see
    /// [`Self::floor_observe_external_window`]). Lives as long as the
    /// drafter, across requests.
    ngram_warm_depths: u64,
    /// Single-row and additional-row verify milliseconds for the loaded arch.
    ngram_row_cost: Option<(f32, f32)>,
    /// Per-window decay of `agreement`, `ngram_agreement` and the floor's
    /// measured costs on the loaded arch ([`singleton_agreement_decay`]).
    agreement_decay: f32,
    /// `[seed, candidates..]` of the current takeover window (reused).
    takeover_block: Vec<u32>,
    /// Same-request AR floor; reset by request configuration or a newly
    /// emitted think-close after retirement. Realignment and head refills
    /// otherwise preserve sticky retirement.
    floor: Qwen4Floor,
    /// `HIPFIRE_MTP_AR_FLOOR` (`0` opts out), read once per request by
    /// `configure_request`; false restores the pre-floor chooser.
    floor_enabled: bool,
    /// The head rows the retired request owes (see [`HeadLag`]); empty
    /// while the head sits at the target.
    head_lag: HeadLag,
    /// Request-local wire counters surfaced by `request_stats`; reset only
    /// by `configure_request`.
    stats: MtpRequestStats,
    /// Resident-accept buffers of earlier requests, kept for reuse because
    /// `configure_request` has no `Gpu` to free them; adopted by the next
    /// eligible sampled request and freed only by `mtp_free`.
    device_spare: Vec<DeviceAccept>,
    /// An EP peer's replica of rank 0's drafter: it never plans a window, never
    /// feeds its own clocks to the floor and prints no trace.
    follower: bool,
    /// Target-less calls (`configure_request`, `mtp_reset`) the EP peers'
    /// drafters have not replayed yet. Compacted on push (see
    /// [`Self::push_replay`]): at most one configure and at most three entries,
    /// whether or not a mesh ever drains it.
    replay: Vec<ReplayOp>,
    /// The last committed window, while a terminal stop inside it is still
    /// repairable ([`RetainedWindow`]). Dropped by the next window, forced
    /// advance, prefill, reset or request configuration.
    retained: Option<RetainedWindow>,
    /// The target hidden of `retained`'s `span.start - 1`: the pending hidden
    /// the window began from, saved before the window moved `pending_hidden`
    /// on (a terminal replay's first head row pairs with it). Valid only
    /// while `retained` is `Some`.
    retained_pending: Option<GpuTensor>,
    /// Bit `d`: this drafter already ran a batched window drafting `d`
    /// tokens, so that row count's one-time verify work is done (see
    /// [`Self::floor_observe_native`]). Lives as long as the drafter, across
    /// requests.
    warm_depths: u64,
    /// This drafter already timed a calibration-path AR token, so the
    /// decode route's one-time work is done (see [`Self::floor_observe_ar`]).
    /// Lives as long as the drafter, across requests.
    warm_ar: bool,
    /// The last native window, priced for the floor only when the next
    /// window is planned (see [`Self::finish_floor_window`]).
    floor_pending: Option<FloorPending>,
    /// Derived-K state of the current request (`HIPFIRE_QWEN4_MTP_DERIVED_K`
    /// read once by `configure_request`); `None` when off, on a follower, or
    /// for a sampled request.
    derived: Option<DerivedPhase>,
    /// Absolute context position of the window being planned, carried by the
    /// singleton `mtp_step` only (other plan callers leave it `None` and never
    /// consult); never `emitted.len()`.
    derived_position: Option<usize>,
    /// Last admitted row-exact verify capacity, including the seed row.
    /// Refreshed from the bundle before planning; profile extent is not a cap.
    verify_row_limit: usize,
    /// Immutable measured-load descriptor for this load (`None` in tests and
    /// in any loader that has not supplied one): the `LoadIdentityMuse`
    /// `Qwen4MeasuredLoadIdentity` shared once per load, borrowed by
    /// reference per request, never hashed or re-serialized per window. A
    /// directly constructed drafter without the descriptor (synthetic CPU
    /// tests included) stays a real incumbent for the derived policy:
    /// synthetic tables exercise the inner binder without ever claiming
    /// production eligibility.
    measured_load_identity:
        Option<std::sync::Arc<crate::measured_identity::Qwen4MeasuredLoadIdentity>>,
    /// Reusable native-margin scratch: the actual logit margin of every head
    /// row drafted this native window (`None` for a non-finite or negative
    /// margin, which carries no evidence). Reserved for [`DERIVED_MAX_DRAFTS`]
    /// once per drafter and reused by every window, so no allocation happens
    /// mid-window. Reset at each native draft entry, appended by every actual
    /// head step, consumed (as bins) by the post-row re-solve and by the lab
    /// forced-window receipt. Present (allocated) only while derived is bound
    /// or a lab forced-cost window is armed; the off path stores nothing.
    derived_margins: Option<Vec<Option<f32>>>,
    #[cfg(feature = "reference-parity")]
    lab_force_drafts: Option<usize>,
    #[cfg(feature = "reference-parity")]
    lab_copy_enabled: Option<bool>,
    #[cfg(feature = "reference-parity")]
    lab_forced_window: Option<crate::wide_verify_oracle::DevForcedWindow>,
    #[cfg(feature = "reference-parity")]
    lab_confirmation: Option<bool>,
    #[cfg(feature = "reference-parity")]
    lab_requested_confirmation: bool,
    #[cfg(feature = "reference-parity")]
    lab_actual_route: Option<crate::wide_verify_oracle::DevWindowRoute>,
    /// Lab window observation arming (`lab_observe_windows`).
    #[cfg(feature = "lab")]
    lab_arm: crate::lab_window::LabWindowArm,
    /// The window of a request without sampled verification, while open.
    #[cfg(feature = "lab")]
    lab_open: Option<Box<crate::lab_window::LabWindow>>,
    /// Closed window observations awaiting `lab_take_window_observation`.
    #[cfg(feature = "lab")]
    lab_done: std::collections::VecDeque<crate::lab_window::LabWindow>,
}

#[cfg(feature = "reference-parity")]
impl Qwen4MtpDrafter {
    /// Pin qualification width, not acceptance or tokens. Clearing it restores
    /// the production planner; the setting survives prefill and reset.
    pub fn lab_force_window_drafts(&mut self, drafts: Option<usize>) -> Result<(), String> {
        if drafts.is_some_and(|d| d > DERIVED_MAX_DRAFTS) {
            return Err("forced draft width exceeds physical executor capacity 63".into());
        }
        self.lab_force_drafts = drafts;
        self.refresh_chooser_capacity();
        Ok(())
    }

    /// Gate copy offers without clearing either pool or resetting evidence.
    /// None restores the configured source policy.
    pub fn lab_force_copy_enabled(&mut self, enabled: Option<bool>) {
        self.lab_copy_enabled = enabled;
    }

    pub fn lab_retire_native(&mut self) -> Result<(), String> {
        if self.follower {
            return Err("lab retirement requires the singleton planner".into());
        }
        self.floor.seed_retired(true);
        self.stats.mtp_retired = true;
        Ok(())
    }

    fn lab_policy_forced(&self) -> bool {
        self.lab_force_drafts.is_some() || self.lab_forced_window.is_some()
    }

    fn lab_ngram_width(&self) -> Option<usize> {
        use crate::wide_verify_oracle::DevWindowSource;
        if self.lab_copy_enabled == Some(false) {
            return None;
        }
        match self.lab_forced_window {
            Some(r) if r.source == DevWindowSource::Ngram => Some(r.drafts),
            Some(_) => None,
            None => self.lab_force_drafts.filter(|&d| d > 0),
        }
    }

    fn lab_plan_window(
        &mut self,
        emitted: &[u32],
        caller_k: usize,
        trace: bool,
        plan_start: Instant,
    ) -> Result<WindowPlan, String> {
        use crate::wide_verify_oracle::DevWindowSource;
        let request = self.lab_forced_window;
        let d = request
            .map_or_else(|| self.lab_force_drafts.unwrap_or(0), |r| r.drafts)
            .min(caller_k);
        self.ngram_candidates.clear();
        self.ngram_proposal.clear();
        let offered_hit = self.ngram_hit(emitted, caller_k);
        let hit = match request.map(|r| r.source) {
            Some(DevWindowSource::Mtp | DevWindowSource::ArRouted) => false,
            _ => offered_hit,
        };
        if request.is_some_and(|r| r.source == DevWindowSource::Ngram) && !hit {
            return Err(format!(
                "unsupported real ngram offer: requested {d}, available {}",
                self.ngram_proposal.len()
            ));
        }
        if request.is_some_and(|r| r.source == DevWindowSource::Mtp) && self.floor.retired() {
            return Err("unsupported forced native window: request is retired".into());
        }
        let confirm = hit && self.ngram_decision.is_none_or(|v| v.confirm);
        self.lab_requested_confirmation = confirm;
        self.trace_ngram_plan(trace, emitted, plan_start, self.floor.retired());
        let route = if self.floor.retired() {
            WindowRoute::ArRetired
        } else if request.is_some_and(|r| r.source == DevWindowSource::ArRouted) {
            WindowRoute::ArRouted
        } else {
            WindowRoute::Native { depth: d }
        };
        Ok(WindowPlan {
            route,
            k: d,
            trace,
            floor_active: false,
            ngram_hit: hit,
            ngram_confirm: !hit || confirm,
            ngram_candidates: Vec::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dev_run_forced_window(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        position: usize,
        seed: u32,
        emitted: &[u32],
        eos: u32,
        request: crate::wide_verify_oracle::DevForcedWindow,
    ) -> Result<crate::wide_verify_oracle::DevWindowMeasurement, String> {
        use crate::wide_verify_oracle::{dev_verify_caps, DevWindowMeasurement, DevWindowSource};
        if self.follower || bundle.ep_mesh().is_some() {
            return Err("forced window timing supports singleton only".into());
        }
        if self.lab_forced_window.is_some() {
            return Err("forced window timing is already active".into());
        }
        if request.drafts > DERIVED_MAX_DRAFTS
            || (request.source == DevWindowSource::ArRouted) != (request.drafts == 0)
        {
            return Err("forced window requires AR K0 or native/real-ngram K1..63".into());
        }
        if self.sampled.is_some() {
            return Err("forced window timing requires greedy verification".into());
        }
        if request.source == DevWindowSource::ArRouted && self.floor.retired() {
            return Err("unsupported ArRouted profile: request is already retired".into());
        }
        if bundle.state.position != position || emitted.last() != Some(&seed) {
            return Err("forced window position or pending seed does not match live state".into());
        }
        let caps = dev_verify_caps(bundle, gpu);
        let cpu = bundle.has_cpu_deferred_experts() || bundle.cpu_storage_capacity_rows() > 0;
        let qualified = if cpu {
            let lab_armed =
                rdna_compute::prefill_invariant::qwen4_wide_verify_override_present(&gpu.arch)
                    && !gpu.replay.is_recording()
                    && !gpu.graphs.capture_mode;
            bundle.cpu_qualified_verify_rows(gpu, lab_armed)
        } else {
            caps.row_exact_verify_rows
        };
        let limit = if request.timing_only && !cpu {
            caps.spec_verify_rows_cap
        } else {
            caps.spec_verify_rows_cap.min(qualified)
        };
        if request.drafts + 1 > limit {
            return Err(format!(
                "unsupported forced width {} rows; effective cap {limit}",
                request.drafts + 1
            ));
        }
        let before = self.stats.ngram_mod_windows;
        let saved_copy = self.lab_copy_enabled;
        self.lab_forced_window = Some(request);
        self.refresh_chooser_capacity();
        self.lab_copy_enabled = None;
        self.lab_confirmation = None;
        self.lab_requested_confirmation = false;
        self.lab_actual_route = None;
        // The lab receipt copies the scratch out below; the batched draft
        // loop arms it when derived or forced-cost measurement is active.
        let timed = (|| {
            gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
            let start = Instant::now();
            let window = self.mtp_step(
                gpu,
                bundle,
                position,
                seed,
                emitted,
                request.drafts,
                eos,
                None,
            )?;
            gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
            let total_us = start.elapsed().as_secs_f64() * 1e6;
            let source = if self.stats.ngram_mod_windows > before {
                DevWindowSource::Ngram
            } else if request.source == DevWindowSource::ArRouted {
                DevWindowSource::ArRouted
            } else {
                DevWindowSource::Mtp
            };
            // Busy window? No: a requested/actual mismatch keeps the observed
            // window as a raw incomplete measurement (the curve consumer gates
            // `complete` on requested == actual and on the margins length).
            // Only a real transaction failure (position/seed/caps/route latch)
            // is `Err` here.
            let outcome_mismatch =
                source != request.source || window.drafts_generated != request.drafts;
            let native_draft_margins = if source == DevWindowSource::Mtp {
                // The lab receipt copies the captured vector out once;
                // production keeps no copy. Native only: every other route
                // is empty. `None` is an actual invalid head-step margin,
                // never padding: the curve length gate preserves an
                // incomplete raw receipt.
                self.derived_margins
                    .as_ref()
                    .map(|scratch| scratch.iter().copied().collect::<Vec<Option<f32>>>())
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            Ok(DevWindowMeasurement {
                requested_drafts: request.drafts,
                actual_drafts: window.drafts_generated,
                offered: self.ngram_proposal.len(),
                accepted_prefix: window.accepted,
                source,
                route: self
                    .lab_actual_route
                    .ok_or("forced window did not latch an execution route")?,
                cost_mode: self.derived_cost_mode(),
                requested_confirmation: self.lab_requested_confirmation,
                head_confirmation: self.lab_confirmation,
                terminal: window.committed.contains(&self.end_of_turn.unwrap_or(eos)),
                total_us,
                window,
                native_draft_margins,
                outcome_mismatch,
            })
        })();
        self.lab_forced_window = None;
        self.lab_copy_enabled = saved_copy;
        timed
    }
}

/// A finished native window awaiting its floor price.
#[derive(Clone, Copy, Debug)]
struct FloorPending {
    start: Instant,
    route: NativeWindow,
    window: WindowStats,
    k: usize,
    /// The drafter's first batched window at this drafted depth.
    first_use: bool,
}

/// What [`Qwen4MtpDrafter::ngram_hit`] decided for one window that had a pool
/// proposal: the plan reads `take` and `confirm`, the `ngram_plan` trace all
/// of it (`rate` and `native_rate` stay 0 when the proposal was shorter than
/// `n_min` or the budget and was never priced).
#[derive(Clone, Copy, Debug)]
struct NgramDecision {
    /// The proposal's length.
    available: usize,
    /// Candidates taken; 0 when the proposal was declined.
    take: usize,
    /// The head's row-0 draft must confirm the first candidate.
    confirm: bool,
    /// Best takeover rate (expected emitted tokens per AR step of cost).
    rate: f32,
    /// The native window's rate it was compared with.
    native_rate: f32,
    /// Depth-1 n-gram acceptance.
    a1: f32,
}

// ---------------------------------------------------------------------------
// Derived K (`HIPFIRE_QWEN4_MTP_DERIVED_K=1`, developer-only, default off).
//
// One E/T objective (`hipfire_runtime::spec_economics::DerivedRequest`) over
// the frozen `fn-phase1-v1` calibration and the measured total window prices
// of the request's cost mode. (a) Before the window it enumerates every
// measured, supported prefix up to the legal ceiling (the qualified row-exact
// verify rows less the seed, the physical draft rows, the head's own K that an
// explicit `speculation.mtp_k` sets, the caller's remaining budget and, for an
// n-gram take, the pool's offer) and runs the global maximum: it neither
// inherits the incumbent's depth, its fresh-request shallow cap nor the
// margin prune's fixed worth as a ceiling, and may widen as well as narrow.
// (b) After each drafted row it re-solves the same objective over the
// remaining measured prefixes `k >= drafted` (stopping now is `k == drafted`;
// a drafted row cannot be discarded and the total price of every such `k`
// already contains the heads spent so far exactly once). It picks no source
// and never bypasses the n-gram confirmation. Forced routes, the floor's own
// AR calibration/retirement and every sampled, PP, EP, lane or unidentified
// request keep the incumbent. A window whose cost mode (OffPool, or OnCopy
// with the copy lane) has no measured price keeps the incumbent and says so
// in the trace; the modes' prices are never mixed. Off (the flag unset or any
// value but `1`) the drafter holds no derived state at all: no allocation, no
// table query, no trace line and no extra Floor call.
// ---------------------------------------------------------------------------

/// A frozen median window price lookup: `(median_us, context_pooled)`;
/// `None` = no measured price in that mode (never another mode's).
type DerivedWindowPrice =
    fn(DerivedCard, DerivedCostMode, DerivedSource, ContextBucket, u8, bool) -> Option<(f64, bool)>;

/// Frozen calibration counts and window/AR prices. Production binds
/// [`FN_PHASE1_TABLES`]; a test binds scripted tables, so the planner hook is
/// exercised without the generated table's values.
#[derive(Clone, Copy)]
struct DerivedTables {
    counts: fn(
        DerivedCard,
        DerivedCostMode,
        DerivedSource,
        u8,
        DerivedFeature,
    ) -> Option<CalibrationCounts>,
    window_price: DerivedWindowPrice,
    ar_price: fn(DerivedCard, DerivedCostMode, ContextBucket) -> Option<(f64, bool)>,
}

const FN_PHASE1_TABLES: DerivedTables = DerivedTables {
    counts: fn_measured_depth_counts,
    window_price: fn_measured_window_price,
    ar_price: fn_measured_ar_price,
};

/// The PLE rows' class the calibration was measured with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DerivedPle {
    /// An attached Q8F16 sidecar (the R9700 receipt).
    Q8Sidecar,
    /// An attached Q4LM sidecar (the Halo receipt).
    Q4Sidecar,
    /// Embedded shards, absent rows or any other encoding: unknown class.
    Other,
}

/// What the loaded target is, bound once per request from the first prefill's
/// GPU and bundle ([`Qwen4MtpDrafter::derived_bind_target`]). The loader ->
/// runtime identity closure (immutable verified artifact/sidecar plus resolved
/// placement/state/route, cheap per-request reference) is owned separately;
/// `DerivedIdentity` does not re-derive or re-hash it here. A missing or
/// mismatched book keeps the real incumbent: no invented prices.
#[derive(Clone, Copy, Debug)]
struct DerivedIdentity<'a> {
    /// `gpu.arch`: exactly `gfx1201` (R9700) or `gfx1151` (Halo).
    arch: &'a str,
    /// `config.profile` is `Qwen4Profile::FlashNext`.
    flash_next: bool,
    /// The loaded manifest's recipe is `Qwen4TierRecipe::FlashNextPacked`.
    packed_recipe: bool,
    /// The config declares the one-layer MTP head and the bundle attached it.
    mtp_head: bool,
    /// No `expert_keep` subset.
    full_experts: bool,
    ple: DerivedPle,
    qsa: crate::QsaKvFormat,
    gdn: crate::GdnStateFormat,
    /// No PP mesh, EP mesh or placement, lane store or follower replica.
    singleton: bool,
}

/// The card a frozen calibration exists for, or the reason this identity is
/// outside it. Exact arch strings; per card the receipts' state class: R9700
/// QSA fp8 + GDN q8 + Q8F16 PLE sidecar, Halo QSA q8 + GDN f32 + Q4LM sidecar.
#[cfg(test)]
fn derived_eligibility(id: &DerivedIdentity<'_>) -> Result<DerivedCard, &'static str> {
    derived_arithmetic_eligibility(id, false)
}

/// The generic book may qualify Q4 on gfx1201 without changing the live
/// arithmetic. Every other architecture, recipe, state and placement law stays
/// identical to the legacy receipt.
fn derived_arithmetic_eligibility(
    id: &DerivedIdentity<'_>,
    qualified_generic: bool,
) -> Result<DerivedCard, &'static str> {
    let card = match id.arch {
        "gfx1201" => DerivedCard::R9700,
        "gfx1151" => DerivedCard::Halo,
        _ => return Err("unsupported_arch"),
    };
    if !id.singleton {
        return Err("not_singleton");
    }
    if !id.flash_next {
        return Err("profile");
    }
    if !id.packed_recipe {
        return Err("recipe");
    }
    if !id.mtp_head {
        return Err("no_mtp_head");
    }
    if !id.full_experts {
        return Err("expert_keep");
    }
    let (ple, state_ok) = match card {
        DerivedCard::R9700 => (
            DerivedPle::Q8Sidecar,
            matches!(
                (id.qsa, id.gdn),
                (crate::QsaKvFormat::Fp8, crate::GdnStateFormat::Q8)
            ),
        ),
        DerivedCard::Halo => (
            DerivedPle::Q4Sidecar,
            matches!(
                (id.qsa, id.gdn),
                (crate::QsaKvFormat::Q8, crate::GdnStateFormat::F32)
            ),
        ),
    };
    if id.ple != ple
        && !(qualified_generic && card == DerivedCard::R9700 && id.ple == DerivedPle::Q4Sidecar)
    {
        return Err("ple_class");
    }
    if !state_ok {
        return Err("state_format");
    }
    Ok(card)
}

/// Admission is an exact cached product descriptor match to a nonempty,
/// qualified measured profile, never merely the same card/mode or PLE class.
fn derived_measured_eligibility(
    id: &DerivedIdentity<'_>,
    cached: Option<&str>,
    admitted: impl FnOnce(DerivedCard) -> Option<&'static str>,
) -> Result<DerivedCard, &'static str> {
    let card = derived_arithmetic_eligibility(id, true)?;
    match (cached, admitted(card)) {
        (Some(cached), Some(profile)) if cached == profile => Ok(card),
        (Some(_), Some(_)) => Err("measured_load_mismatch"),
        _ => Err("measured_load_missing"),
    }
}

/// The identity of a loaded target. `shared_scope`: the request does not own
/// the whole model on one device (a PP mesh or a follower replica).
fn derived_identity_of<'a>(
    bundle: &Qwen4Bundle,
    arch: &'a str,
    shared_scope: bool,
) -> DerivedIdentity<'a> {
    use hipfire_runtime::external_rows::RowEncoding;
    let ple = match bundle.ple_source() {
        Some(crate::artifact::Qwen4PleRows::Sidecar {
            encoding: RowEncoding::Q8F16,
            ..
        }) => DerivedPle::Q8Sidecar,
        Some(crate::artifact::Qwen4PleRows::Sidecar {
            encoding: RowEncoding::Q4Lm,
            ..
        }) => DerivedPle::Q4Sidecar,
        _ => DerivedPle::Other,
    };
    let state = bundle.state_format();
    DerivedIdentity {
        arch,
        flash_next: matches!(
            bundle.config.profile,
            crate::config::Qwen4Profile::FlashNext
        ),
        packed_recipe: matches!(
            bundle.weights.manifest.identity.recipe,
            crate::Qwen4TierRecipe::FlashNextPacked { .. }
        ),
        mtp_head: bundle.config.mtp.is_some() && bundle.mtp.is_some(),
        full_experts: bundle.config.expert_keep.is_none()
            && bundle.weights.manifest.expert_keep.is_none(),
        ple,
        qsa: state.qsa,
        gdn: state.gdn,
        singleton: !shared_scope
            && !bundle.pp_mesh_active()
            && bundle.ep_mesh().is_none()
            && bundle
                .weights
                .ep_placement()
                .is_none_or(|placement| placement.ranks <= 1)
            && bundle.lanes().is_none(),
    }
}

/// A request's derived-K lifecycle. The drafter's `derived` is `None` when
/// the flag is off, the drafter is a follower replica, or the request is
/// sampled.
enum DerivedPhase {
    /// Configured with the flag on; the first prefill binds the identity.
    Unbound,
    /// Bound, and outside every frozen calibration: the incumbent runs.
    Ineligible,
    Bound(Box<DerivedState>),
}

/// A bound request's chooser state. The policy vectors are provisioned once
/// for the physical maximum ([`DERIVED_MAX_DRAFTS`] depths, one more price)
/// and reused by every window whatever rows the executor qualifies; only the
/// sparse residual cells follow the admitted capacity.
struct DerivedState {
    card: DerivedCard,
    tables: DerivedTables,
    request: DerivedRequest,
    capacity: usize,
    depths: Vec<DerivedDepth>,
    prices: Vec<DerivedPrice>,
    resolve: Vec<DerivedPrice>,
    keys: Vec<ResidualKey>,
    pending: Option<DerivedPending>,
    /// Cost mode the bind-time identity was admitted under: a mode flip
    /// after bind invalidates the window (see [`Self::derived_route_changed`]).
    mode_hint: Option<DerivedCostMode>,
    /// Reusable per-depth re-solve inputs: `depths` with every measured
    /// legal depth re-keyed to the latest observed margin cell. Reused by
    /// every re-solve and every window; never reallocated mid-window.
    margined: Vec<DerivedDepth>,
    /// Reusable commit bins of the executed rows' observed margins, indexed
    /// `row - 1`. Reused by every commit; never reallocated per window.
    margin_bins: Vec<Option<NativeMarginBin>>,
    /// Last re-solve's evidence: the committed row whose margin re-keyed
    /// `margined`, and how many depths carried a measured margin cell.
    /// `measured == 0` means the re-solve ran on the unconditional evidence.
    last_margin: Option<(usize, NativeMarginBin)>,
    last_measured: usize,
    /// Refreshed from the actual bundle/GPU before each production plan.
    /// Scripted CPU table tests have no device reference.
    reference: Option<bool>,
    #[cfg(test)]
    consults: u32,
    #[cfg(test)]
    observes: u32,
    #[cfg(test)]
    resolves: u32,
}

impl DerivedState {
    fn new(card: DerivedCard, tables: DerivedTables, capacity: usize) -> Self {
        Self {
            card,
            tables,
            request: DerivedRequest::with_max_drafts(card, capacity),
            capacity,
            mode_hint: None,
            depths: Vec::with_capacity(DERIVED_MAX_DRAFTS),
            prices: Vec::with_capacity(DERIVED_MAX_DRAFTS + 1),
            resolve: Vec::with_capacity(DERIVED_MAX_DRAFTS + 1),
            margined: Vec::with_capacity(DERIVED_MAX_DRAFTS),
            margin_bins: Vec::with_capacity(DERIVED_MAX_DRAFTS),
            keys: Vec::with_capacity(DERIVED_MAX_DRAFTS),
            pending: None,
            last_margin: None,
            last_measured: 0,
            reference: None,
            #[cfg(test)]
            consults: 0,
            #[cfg(test)]
            observes: 0,
            #[cfg(test)]
            resolves: 0,
        }
    }

    fn set_capacity(&mut self, capacity: usize) {
        if capacity == self.capacity {
            return;
        }
        self.capacity = capacity;
        // Only the sparse residual cells follow the admitted capacity; the
        // policy vectors were provisioned for the physical maximum.
        self.request.set_capacity(capacity);
    }
}

/// What a consulted window needs at its observation boundary.
#[derive(Clone, Copy, Debug)]
struct DerivedPending {
    /// The source whose depths were priced (the incumbent's).
    source: DerivedSource,
    /// The cost mode the prices were taken in.
    mode: DerivedCostMode,
    /// Pre-window price identity, not an observed confirmation outcome.
    confirmation: bool,
    /// Absolute context position of the window.
    position: usize,
    /// The incumbent width: the native depth or the n-gram take.
    incumbent: u8,
    /// `None`: the chooser found no priced action; `Some(0)`: AR.
    chosen: Option<u8>,
    /// The chosen action's total window cost and expected tokens.
    chosen_cost_us: f64,
    chosen_expected: f64,
    /// The widest measured legal prefix the window priced.
    hi: u8,
    /// The legal draft ceiling of this window before the caller's budget.
    legal: usize,
    /// The caller's draft budget, whose `+ 1` is the window's commit cap.
    caller_k: usize,
    /// Some priced action used the all-context pooled median.
    pooled: bool,
    /// The n-gram takeover ran (the head's confirmation matched), as opposed
    /// to the native window that replaces an unconfirmed hit.
    ran_ngram: bool,
}

/// Replay2's context label of `bucket`.
fn derived_context_label(bucket: ContextBucket) -> &'static str {
    match bucket {
        ContextBucket::Le2K => "<=2K",
        ContextBucket::Le8K => "<=8K",
        ContextBucket::Le16K => "<=16K",
        ContextBucket::Over16K => "32K+",
    }
}

/// The `source` label of a derived trace line.
fn derived_source_label(source: DerivedSource) -> &'static str {
    match source {
        DerivedSource::Mtp => "mtp",
        DerivedSource::Ngram => "ngram",
    }
}

/// The `mode` label of a derived trace line.
fn derived_mode_label(mode: DerivedCostMode) -> &'static str {
    match mode {
        DerivedCostMode::OffPool => "off_pool",
        DerivedCostMode::OnCopy => "on_copy",
    }
}

impl Qwen4MtpDrafter {
    /// The request's derived-K state: reset by every configuration. `enabled`
    /// is the flag, read once by [`MtpDrafter::configure_request`]. A
    /// follower replica never plans; a sampled request (a non-finite or
    /// positive temperature) creates no state and says so once.
    fn derived_configure(&mut self, cfg: &SpecRequestConfig, enabled: bool) {
        self.derived = None;
        self.derived_position = None;
        if !enabled || self.follower {
            return;
        }
        if !cfg.temp.is_finite() || cfg.temp > 1.0e-6 {
            self.trace_derived_request(false, Some("sampled"), None);
            return;
        }
        self.derived = Some(DerivedPhase::Unbound);
    }

    /// Bind the request's identity outcome (first call only). The mode hint
    /// is stamped from the drafter's current cost mode so a later flip
    /// invalidates the window.
    fn derived_bind(&mut self, outcome: Result<DerivedCard, &'static str>, tables: DerivedTables) {
        if !matches!(self.derived, Some(DerivedPhase::Unbound)) {
            return;
        }
        let mode = self.derived_cost_mode();
        match outcome {
            Ok(card) => {
                let mut state =
                    DerivedState::new(card, tables, self.derived_legal_cap(DerivedSource::Mtp));
                state.mode_hint = Some(mode);
                self.derived = Some(DerivedPhase::Bound(Box::new(state)));
                self.refresh_chooser_capacity();
                self.trace_derived_request(true, None, Some(card));
            }
            Err(reason) => {
                self.derived = Some(DerivedPhase::Ineligible);
                self.trace_derived_request(false, Some(reason), None);
            }
        }
    }

    /// Bind from the first prefill's GPU and target. A PP mesh is not a
    /// singleton. The cached canonical JSON (supplied once by the loader via
    /// [`Self::with_measured_load_identity`]) is compared once here against
    /// the runtime's admitted identity for the request's cost mode; any
    /// mismatch (or any missing descriptor) keeps the real incumbent via
    /// `Ineligible`, never invented prices.
    fn derived_bind_target(&mut self, ops: &mut Qwen4Ops<'_>, target: &mut dyn SpecTarget) {
        let mesh = ops.is_mesh();
        let arch = ops.gpu().map(|gpu| gpu.arch.clone()).unwrap_or_default();
        let shared_scope = mesh || self.follower;
        let mode = self.derived_cost_mode();
        // Typed descriptor, compared as its cached canonical JSON string:
        // no re-serialization per request, no hashing. Any mismatch or
        // missing descriptor keeps the real incumbent via `Ineligible`.
        let measured = self
            .measured_load_identity
            .as_deref()
            .map(|identity| identity.canonical_json());
        let outcome = match Self::bundle(target) {
            Ok(bundle) => derived_measured_eligibility(
                &derived_identity_of(bundle, &arch, shared_scope),
                measured,
                |card| fn_measured_load_identity(card, mode),
            ),
            Err(_) => Err("not_qwen4_bundle"),
        };
        self.derived_bind(outcome, FN_PHASE1_TABLES);
    }

    /// The once-per-request `derived_k_request` trace line.
    fn trace_derived_request(
        &self,
        eligible: bool,
        reason: Option<&str>,
        card: Option<DerivedCard>,
    ) {
        let enabled =
            hipfire_config::developer_var("HIPFIRE_MTP_TRACE").is_ok_and(|value| value == "1");
        if !enabled || self.follower {
            return;
        }
        let reason = reason.map_or_else(|| "null".to_string(), |reason| format!("\"{reason}\""));
        let card = match card {
            Some(DerivedCard::R9700) => "\"R9700\"",
            Some(DerivedCard::Halo) => "\"Halo\"",
            None => "null",
        };
        eprintln!(
            "QWEN4_MTP_TRACE {{\"event\":\"derived_k_request\",\"version\":\"{FN_DERIVED_VERSION}\",\"eligible\":{eligible},\"reason\":{reason},\"card\":{card}}}"
        );
    }

    /// A consulted window awaits its observation.
    fn derived_pending_open(&self) -> bool {
        matches!(&self.derived, Some(DerivedPhase::Bound(state)) if state.pending.is_some())
    }

    /// Drop the consulted window without an update (a failed or rolled-back
    /// window, a reset, a prefill or a terminal repair). The scratch is left
    /// alone on purpose: the lab receipt reads it after this call on the
    /// failed-window path, and the next batched window re-arms from its own
    /// entry reset. Stale margins never leak into a later re-solve because
    /// every batched entry resets before the first head step.
    fn derived_drop_pending(&mut self) {
        if let Some(DerivedPhase::Bound(state)) = self.derived.as_mut() {
            state.pending = None;
        }
    }

    /// The window's n-gram takeover ran, so the queried n-gram source
    /// executed (a confirmation miss never reaches it).
    fn derived_note_takeover(&mut self) {
        if let Some(DerivedPhase::Bound(state)) = self.derived.as_mut() {
            if let Some(pending) = state.pending.as_mut() {
                pending.ran_ngram = true;
            }
        }
    }

    /// The planner hook, after the incumbent chose a batched native route of
    /// depth >= 1: consult the chooser over the incumbent source's prefixes.
    /// The live target/state/placement/mode identity is re-checked once per
    /// request bind (see [`Self::derived_bind_target`]); a relevant
    /// transition after bind invalidates to `Ineligible` here: a copy-lane
    /// mode flip, or an AR-reference fallback/record transition away from the
    /// reference the admitted profile was measured with. The production
    /// window refreshes the exact bundle/GPU reference before consulting;
    /// no per-window hashing or serialization is needed.
    fn derived_plan(&mut self, plan: &mut WindowPlan, caller_k: usize) {
        if self.derived_route_changed(plan) {
            self.derived = Some(DerivedPhase::Ineligible);
            return;
        }
        let Some(position) = self.derived_position else {
            return;
        };
        if !matches!(self.derived, Some(DerivedPhase::Bound(_))) {
            return;
        }
        let Some(DerivedPhase::Bound(mut state)) = self.derived.take() else {
            return;
        };
        self.derived_consult(&mut state, plan, caller_k, position);
        self.derived = Some(DerivedPhase::Bound(state));
    }

    /// Whether the window's live execution route no longer matches the bound
    /// identity: a copy-lane mode flip, or an AR-reference fallback/record
    /// transition away from the admitted profile's reference. The production
    /// caller supplies the exact current reference; scripted CPU table tests
    /// exercise the same optimizer without a live device.
    fn derived_route_changed(&self, plan: &WindowPlan) -> bool {
        let Some(DerivedPhase::Bound(state)) = self.derived.as_ref() else {
            return false;
        };
        let mode = self.derived_cost_mode();
        if state.mode_hint != Some(mode) {
            return true;
        }
        let source = if plan.ngram_hit {
            DerivedSource::Ngram
        } else {
            DerivedSource::Mtp
        };
        state.reference.is_some_and(|recorded| {
            !fn_measured_reference_supported(state.card, mode, source, recorded)
        })
    }

    /// The mode a window's prices are keyed by: the copy lane routes (and so
    /// costs) every window differently from the default pool route, so its
    /// prices are a separate measured profile, never the pool's.
    fn derived_cost_mode(&self) -> DerivedCostMode {
        if self.copy_lane {
            DerivedCostMode::OnCopy
        } else {
            DerivedCostMode::OffPool
        }
    }

    /// The admitted request ceiling, independent of historical profile extent.
    fn incumbent_max_k(&self) -> usize {
        self.max_k
            .min(self.explicit_mtp_k.unwrap_or(DERIVED_MAX_DRAFTS))
            .min(self.verify_row_limit.saturating_sub(1))
            .min(self.draft_rows_cap())
            .min(DERIVED_MAX_DRAFTS)
    }

    /// The incumbent ceiling when off; otherwise only physical/qualified
    /// capacity and an explicit user override, never the load's default K.
    fn k_ceiling(&self) -> usize {
        #[cfg(feature = "reference-parity")]
        if self.lab_policy_forced() {
            return self
                .lab_forced_window
                .map(|r| r.drafts)
                .or(self.lab_force_drafts)
                .unwrap_or(0)
                .min(self.draft_rows_cap());
        }
        if matches!(
            self.derived,
            Some(DerivedPhase::Unbound | DerivedPhase::Bound(_))
        ) {
            self.derived_legal_cap(DerivedSource::Mtp)
        } else {
            self.incumbent_max_k()
        }
    }

    /// Every source obeys the same physical/qualified and explicit user caps.
    /// Real n-gram availability is an additional bound at consultation.
    fn derived_legal_cap(&self, _source: DerivedSource) -> usize {
        self.explicit_mtp_k
            .unwrap_or(DERIVED_MAX_DRAFTS)
            .min(self.verify_row_limit.saturating_sub(1))
            .min(self.draft_rows_cap())
            .min(DERIVED_MAX_DRAFTS)
    }

    /// Refresh the executor's qualified capacity before planning, without
    /// changing the profile or promoting an unqualified load.
    fn derived_refresh_rows(&mut self, rows: usize) {
        self.verify_row_limit = rows;
        self.refresh_chooser_capacity();
    }

    fn refresh_chooser_capacity(&mut self) {
        let capacity = self.k_ceiling().min(self.draft_rows_cap());
        if self.agreement.len() != capacity {
            self.agreement.resize(capacity, AGREEMENT_PRIOR);
        }
        self.floor.set_capacity(capacity);
        let derived_capacity = self.derived_legal_cap(DerivedSource::Mtp);
        if let Some(DerivedPhase::Bound(state)) = self.derived.as_mut() {
            state.set_capacity(derived_capacity);
        }
        if self.derived_margins_active() {
            // Provisioned for the physical maximum, never the current rows:
            // the executor's qualified rows can change between windows and
            // the per-window path must not allocate.
            let scratch = self.derived_margins.get_or_insert_with(Vec::new);
            scratch.reserve(DERIVED_MAX_DRAFTS.saturating_sub(scratch.len()));
        }
    }

    /// Price, decide and apply (a). Without a measured priced drafted width
    /// (every width unpriced in this window's cost mode: an AR price alone
    /// would leave the chooser nothing to compare) nothing is consulted, so
    /// the window is not observed either; the incumbent runs and the trace
    /// says why.
    fn derived_consult(
        &mut self,
        state: &mut DerivedState,
        plan: &mut WindowPlan,
        caller_k: usize,
        position: usize,
    ) {
        let WindowRoute::Native { depth } = plan.route else {
            return;
        };
        let (source, incumbent) = if plan.ngram_hit {
            (DerivedSource::Ngram, self.ngram_candidates.len())
        } else {
            (DerivedSource::Mtp, depth)
        };
        let legal = self.derived_legal_cap(source);
        state.set_capacity(legal);
        // Prefixes the window can run: the legal ceiling, the caller's
        // remaining budget and, for a take, what the pool offered.
        let mut cap = legal.min(caller_k);
        if source == DerivedSource::Ngram {
            cap = cap.min(self.ngram_proposal.len());
        }
        let mode = self.derived_cost_mode();
        let bucket = ContextBucket::of(position);
        let (card, tables) = (state.card, state.tables);
        state.prices.clear();
        let mut pooled = false;
        let mut hi = 0u8;
        for width in 1..=cap.min(DERIVED_MAX_DRAFTS) {
            let k = width as u8;
            if let Some((cost_us, pool)) =
                (tables.window_price)(card, mode, source, bucket, k, plan.ngram_confirm)
            {
                state.prices.push(DerivedPrice { k, cost_us });
                pooled |= pool;
                hi = k;
            }
        }
        if hi == 0 {
            self.trace_derived_unsupported(
                plan,
                position,
                source,
                mode,
                incumbent,
                legal,
                caller_k,
                "no_measured_price",
            );
            return;
        }
        if let Some((cost_us, pool)) = (tables.ar_price)(card, mode, bucket) {
            state.prices.insert(0, DerivedPrice { k: 0, cost_us });
            pooled |= pool;
        }
        // The n-gram feature of every depth of this window: the pool's full
        // offer (kept before the incumbent's take truncated it), the matched
        // key's exact length and the last executed n-gram window's first token.
        let ngram_feature = match source {
            DerivedSource::Mtp => None,
            DerivedSource::Ngram => {
                let meta = if self.copy_lane {
                    self.ngram_copy
                        .as_ref()
                        .and_then(MtpCopyContext::last_proposal)
                } else {
                    self.ngram.as_ref().and_then(MtpNgramContext::last_proposal)
                };
                Some((
                    derived_offer_bin(Some(self.ngram_proposal.len())),
                    meta.and_then(|meta| u32::try_from(meta.key_len).ok()),
                    state.request.last_first(DerivedSource::Ngram),
                ))
            }
        };
        state.keys.clear();
        state.depths.clear();
        for depth in 1..=hi {
            let feature = match ngram_feature {
                None => DerivedFeature::Native,
                Some((offered, key_len, last_first)) => DerivedFeature::Ngram {
                    offered,
                    key_len,
                    last_first,
                },
            };
            let key = ResidualKey {
                source,
                depth,
                feature,
            };
            let Some(counts) = (tables.counts)(card, mode, source, depth, feature) else {
                break;
            };
            state.keys.push(key);
            state.depths.push(DerivedDepth { key, counts });
        }
        // A price alone cannot support an uncalibrated depth. In particular
        // wide OnCopy never borrows the old population's unseen .8 prior.
        state
            .prices
            .retain(|p| usize::from(p.k) <= state.depths.len());
        hi = state.prices.iter().map(|p| p.k).max().unwrap_or(0);
        if hi == 0 {
            self.trace_derived_unsupported(
                plan,
                position,
                source,
                mode,
                incumbent,
                legal,
                caller_k,
                "no_measured_counts",
            );
            return;
        }
        state.depths.truncate(usize::from(hi));
        state.keys.truncate(usize::from(hi));
        let incumbent_u8 = u8::try_from(incumbent).unwrap_or(u8::MAX);
        let decision = state
            .request
            .choose(&state.depths, &state.prices, incumbent_u8);
        let chosen = decision.map(|decision| match decision.action {
            DerivedAction::Ar => 0,
            DerivedAction::Prefix(k) => k,
        });
        state.pending = Some(DerivedPending {
            source,
            mode,
            position,
            incumbent: incumbent_u8,
            confirmation: plan.ngram_confirm,
            chosen,
            chosen_cost_us: decision.map_or(f64::NAN, |decision| decision.cost_us),
            chosen_expected: decision.map_or(f64::NAN, |decision| decision.expected),
            hi,
            legal,
            caller_k,
            pooled,
            ran_ngram: false,
        });
        #[cfg(test)]
        {
            state.consults += 1;
        }
        match chosen {
            // AR is per window: the head-free AR token after the resource
            // pass and head catch-up, with no hit and no sticky retirement.
            Some(0) => {
                plan.route = WindowRoute::ArRouted;
                plan.ngram_hit = false;
                plan.ngram_confirm = true;
            }
            // Native: the draft loop may run up to the widest measured legal
            // prefix; the post-row re-solve ([`Self::derived_resolve`]) stops
            // it at the objective's width. An n-gram take is exactly the
            // chosen prefix of the pool's offer (the confirmation and the
            // native fallback depth stay the incumbent's).
            Some(width) => match source {
                DerivedSource::Mtp => {
                    plan.route = WindowRoute::Native {
                        depth: usize::from(hi),
                    };
                    plan.k = plan.k.max(usize::from(hi));
                }
                DerivedSource::Ngram => {
                    self.ngram_candidates.clear();
                    self.ngram_candidates
                        .extend_from_slice(&self.ngram_proposal[..usize::from(width)]);
                }
            },
            None => {}
        }
    }

    /// This window's draft loop is driven by the derived objective: a
    /// consulted native-MTP window that chose a drafted width. The incumbent's
    /// fixed-worth margin prune does not run there.
    fn derived_loop_active(&self) -> bool {
        matches!(
            &self.derived,
            Some(DerivedPhase::Bound(state))
                if state.pending.is_some_and(|pending| {
                    pending.source == DerivedSource::Mtp
                        && pending.chosen.is_some_and(|k| k > 0)
                })
        )
    }

    /// Whether the drafter stores native draft margins this window: derived
    /// is bound or the lab forced-cost window is armed. The off path (no
    /// derived state, no forced measurement) stores nothing.
    fn derived_margins_active(&self) -> bool {
        if matches!(self.derived, Some(DerivedPhase::Bound(_))) {
            return true;
        }
        #[cfg(feature = "reference-parity")]
        if self.lab_forced_window.is_some() {
            return true;
        }
        false
    }

    /// Clear the capacity-sized margin scratch provisioned at admission/bind.
    /// Inactive windows neither store margins nor free reusable storage.
    fn derived_margins_reset(&mut self) {
        if let Some(scratch) = self.derived_margins.as_mut() {
            scratch.clear();
        }
    }

    /// Append one actual head-step margin to the scratch, when armed (see
    /// [`Self::derived_margins_reset`]). A non-finite or negative margin is
    /// stored as `None`: it carries no evidence and the re-solve falls back
    /// to the unconditional depth evidence.
    fn derived_margins_push(&mut self, margin: f32) {
        if !self.derived_margins_active() {
            return;
        }
        let capacity = self.k_ceiling().min(self.draft_rows_cap());
        let Some(scratch) = self.derived_margins.as_mut() else {
            return;
        };
        if scratch.len() >= capacity {
            return;
        }
        scratch.push(if margin.is_finite() && margin >= 0.0 {
            Some(margin)
        } else {
            None
        });
    }

    /// Bin of the margin observed at committed draft row `observed_depth`
    /// (1-based), or `None` when the row produced no usable margin.
    fn derived_margin_bin(&self, observed_depth: usize) -> Option<NativeMarginBin> {
        let scratch = self.derived_margins.as_ref()?;
        let margin = (*scratch.get(observed_depth.checked_sub(1)?)?)?;
        native_margin_bin(margin)
    }

    /// Trace a dropped (non-finite or negative) head-step margin: explicit
    /// diagnostic, no invented bin. Gated on `HIPFIRE_MTP_TRACE=1` and never
    /// from a follower.
    fn trace_derived_margin_dropped(&self, position: usize, observed_depth: usize) {
        if self.follower {
            return;
        }
        if !hipfire_config::developer_var("HIPFIRE_MTP_TRACE").is_ok_and(|value| value == "1") {
            return;
        }
        eprintln!(
            "QWEN4_MTP_TRACE {{\"event\":\"derived_k_margin_dropped\",\"version\":\"{FN_DERIVED_VERSION}\",\"position\":{position},\"observed_depth\":{observed_depth},\"reason\":\"nonfinite_or_negative\"}}"
        );
    }

    /// Re-key `state.margined` in place to the margin observed at committed
    /// row `observed_depth`: every measured legal depth `1..=hi` (the whole
    /// re-solve domain, since acceptance of every depth is still unknown
    /// after the current row) reads its margin-conditional counts, falling
    /// back to the pre-decision unconditional depth where the margin is
    /// unusable or the cell is unsupported. Only the legal `hi` governs:
    /// the sparse price count is not the maximum K. Counts how many depths
    /// carried a measured cell and records the evidence in
    /// `last_margin`/`last_measured` for the truthful commit trace.
    /// `observed = u8::MAX` is unrepresentable: depths are contiguous
    /// `1..=hi` with `hi <= DERIVED_MAX_DRAFTS` (63).
    fn derived_rekey_margined(
        state: &mut DerivedState,
        mode: DerivedCostMode,
        observed_depth: usize,
        bin: Option<NativeMarginBin>,
    ) {
        let Some(bin) = bin else {
            state.margined.clear();
            state.margined.extend_from_slice(&state.depths);
            state.last_margin = None;
            state.last_measured = 0;
            return;
        };
        let observed = u8::try_from(observed_depth).unwrap_or(u8::MAX);
        let card = state.card;
        state.margined.clear();
        state.last_measured = 0;
        if observed == u8::MAX {
            state.margined.extend_from_slice(&state.depths);
            state.last_margin = None;
            return;
        }
        state.last_margin = Some((observed_depth, bin));
        let mut measured = 0usize;
        state.margined.extend(state.depths.iter().map(|depth| {
            let key = ResidualKey {
                source: DerivedSource::Mtp,
                depth: depth.key.depth,
                feature: DerivedFeature::NativeMargin {
                    observed_depth: observed,
                    margin: bin,
                },
            };
            match (state.tables.counts)(
                card,
                mode,
                DerivedSource::Mtp,
                depth.key.depth,
                key.feature,
            ) {
                Some(counts) => {
                    measured += 1;
                    DerivedDepth { key, counts }
                }
                None => *depth,
            }
        }));
        state.last_measured = measured;
        if measured == 0 {
            // Every cell fell back: the re-solve ran on the unconditional
            // evidence, so no margin row is claimed as its evidence.
            state.last_margin = None;
        }
    }

    /// (b): the single global re-solve over the remaining measured legal
    /// prefixes `k >= drafted`. Reads the most recently observed margin (the
    /// caller passes the latest `drafted` row); `None` is the same
    /// unconditional re-solve. The current margin informs ALL target depths
    /// `1..=hi`, not only `k >= drafted`: the whole draft's acceptance is
    /// still unknown evidence about the prefix. Non-derived production never
    /// runs this. Returns whether to draft another row.
    fn derived_resolve(&mut self, drafted: usize) -> bool {
        let observed_depth = drafted;
        let bin = self.derived_margin_bin(observed_depth);
        let mode = self.derived_cost_mode();
        // Disjoint borrows: the drafter's scratch bins the margin while the
        // bound state re-solves in place — no allocation, no borrow conflict.
        let Some(DerivedPhase::Bound(state)) = self.derived.as_mut() else {
            return false;
        };
        let Some(pending) = state.pending else {
            return false;
        };
        if pending.source != DerivedSource::Mtp {
            return false;
        }
        Self::derived_rekey_margined(state, mode, observed_depth, bin);
        state.resolve.clear();
        state.resolve.extend(
            state
                .prices
                .iter()
                .copied()
                .filter(|price| price.k >= 1 && usize::from(price.k) >= drafted),
        );
        #[cfg(test)]
        {
            state.resolves += 1;
        }
        let decision =
            state
                .request
                .choose(&state.margined, &state.resolve, pending.chosen.unwrap_or(0));
        if let (Some(decision), Some(pending)) = (decision, state.pending.as_mut()) {
            pending.chosen = Some(match decision.action {
                DerivedAction::Ar => 0,
                DerivedAction::Prefix(k) => k,
            });
            pending.chosen_cost_us = decision.cost_us;
            pending.chosen_expected = decision.expected;
        }
        decision.is_some_and(|decision| {
            matches!(decision.action, DerivedAction::Prefix(k) if usize::from(k) > drafted)
        })
    }

    /// The `derived_k_unsupported` trace line: a window the chooser could not
    /// price in its cost mode, which therefore runs the incumbent.
    #[allow(clippy::too_many_arguments)]
    fn trace_derived_unsupported(
        &self,
        plan: &WindowPlan,
        position: usize,
        source: DerivedSource,
        mode: DerivedCostMode,
        incumbent: usize,
        legal: usize,
        caller_k: usize,
        reason: &str,
    ) {
        if !plan.trace || self.follower {
            return;
        }
        eprintln!(
            "QWEN4_MTP_TRACE {{\"event\":\"derived_k_unsupported\",\"version\":\"{FN_DERIVED_VERSION}\",\"source\":\"{}\",\"mode\":\"{}\",\"position\":{position},\"context\":\"{}\",\"reason\":\"{reason}\",\"incumbent_k\":{incumbent},\"legal_cap\":{legal},\"caller_k\":{caller_k},\"row_limit\":{},\"user_k\":{},\"confirmation\":{},\"offered\":{}}}",
            derived_source_label(source),
            derived_mode_label(mode),
            derived_context_label(ContextBucket::of(position)),
            self.verify_row_limit,
            self.explicit_mtp_k.map_or_else(|| "null".to_owned(), |k| k.to_string()),
            plan.ngram_confirm,
            self.ngram_proposal.len()
        );
    }

    /// Observe a consulted window exactly once at its verified-window
    /// boundary (where the `window` trace receipt is written, before the
    /// emitter's final publish): a successful window updates the residuals,
    /// a failed one only drops the pending.
    ///
    /// Only the queried source's own execution is credited: an AR window, or
    /// an n-gram hit whose confirmation missed and fell back to the native
    /// window, ages the residuals with `executed == 0` and donates nothing.
    /// `accepted` is clamped to what the window actually committed and to the
    /// caller's window cap `caller_k + 1`. A window is terminal when it
    /// committed the end-of-turn token or hit that cap. The caller bounds
    /// every window by `k = min(capacity, remaining - 1)` (there is no
    /// separate max-output setting here), so a window that commits
    /// `caller_k + 1` tokens accepted every executed draft: marking it
    /// terminal censors no rejection and changes no counted outcome. Stop
    /// strings and cancellation clip tokens only after the window is emitted
    /// and end the request, so no later decision exists to bias; they are
    /// not terminal here.
    fn derived_observe(&mut self, plan: &WindowPlan, eos: u32, result: &Result<MtpWindow, String>) {
        if !matches!(self.derived, Some(DerivedPhase::Bound(_))) {
            return;
        }
        let Some(DerivedPhase::Bound(mut state)) = self.derived.take() else {
            return;
        };
        if let (Some(pending), Ok(window)) = (state.pending.take(), result) {
            let ran = match (plan.route, pending.source) {
                (WindowRoute::Native { .. }, DerivedSource::Mtp) => !plan.ngram_hit,
                (WindowRoute::Native { .. }, DerivedSource::Ngram) => {
                    plan.ngram_hit && pending.ran_ngram
                }
                _ => false,
            };
            let committed = window.committed.len();
            let cap = pending.caller_k.saturating_add(1);
            let drafted = if ran {
                window.drafts_generated.min(state.keys.len())
            } else {
                0
            };
            let accepted = if ran {
                window.accepted.min(committed).min(cap)
            } else {
                0
            };
            let terminal = window.committed.contains(&eos) || committed == cap;
            let observed = u8::try_from(drafted).unwrap_or(u8::MAX);
            // Commit-only evidence, exactly once at this boundary: single
            // aging, base update verbatim, then per-row margin bumps on
            // reached depths with identical censoring. Bins fill the reused
            // `margin_bins` scratch in place (no per-window allocation);
            // disjoint borrows keep `self.derived` out of the fill.
            let (evidenced, commit_measured) =
                if ran && pending.source == DerivedSource::Mtp && drafted > 0 {
                    let margins = self.derived_margins.as_ref();
                    state.margin_bins.clear();
                    state.margin_bins.extend((0..drafted).map(|row| {
                        margins
                            .and_then(|scratch| scratch.get(row))
                            .copied()
                            .flatten()
                            .and_then(native_margin_bin)
                    }));
                    let evidenced = state.margin_bins.iter().any(|bin| bin.is_some());
                    if evidenced {
                        state.request.observe_with_margins(
                            &state.keys,
                            &state.margin_bins,
                            observed,
                            accepted,
                            terminal,
                        );
                    } else {
                        state
                            .request
                            .observe(&state.keys, observed, accepted, terminal);
                    }
                    (evidenced, true)
                } else {
                    state
                        .request
                        .observe(&state.keys, observed, accepted, terminal);
                    (false, false)
                };
            #[cfg(test)]
            {
                state.observes += 1;
            }
            {
                let margin_evidence = if evidenced { "measured" } else { "unmeasured" };
                let margin_rows = state.last_measured;
                let margin_bin = state
                    .last_margin
                    .map(|(_, bin)| bin.index().to_string())
                    .unwrap_or_else(|| "null".to_string());
                let margin_depth = state
                    .last_margin
                    .map(|(depth, _)| depth.to_string())
                    .unwrap_or_else(|| "null".to_string());
                if plan.trace && !self.follower {
                    let priced = state
                        .prices
                        .iter()
                        .map(|price| price.k.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    let chosen = pending
                        .chosen
                        .map_or_else(|| "null".to_string(), |k| k.to_string());
                    let changed = pending.chosen.is_some_and(|k| k != pending.incumbent);
                    eprintln!(
                        "QWEN4_MTP_TRACE {{\"event\":\"derived_k\",\"version\":\"{FN_DERIVED_VERSION}\",\"source\":\"{}\",\"mode\":\"{}\",\"verify_route\":\"batched\",\"position\":{},\"context\":\"{}\",\"incumbent_k\":{},\"chosen_k\":{chosen},\"changed\":{changed},\"priced_k\":[{priced}],\"context_pooled\":{},\"legal_cap\":{},\"caller_k\":{},\"widest_priced_k\":{},\"row_limit\":{},\"user_k\":{},\"chosen_cost_us\":{},\"chosen_expected\":{},\"drafted_k\":{drafted},\"accepted\":{accepted},\"confirmation\":{},\"offered\":{},\"price_boundary\":\"total_window\",\"margin_evidence\":\"{margin_evidence}\",\"margin_observed_depth\":{margin_depth},\"margin_bin\":{margin_bin},\"margin_measured_depths\":{margin_rows},\"commit_margins\":{commit_measured}}}",
                        derived_source_label(pending.source),
                        derived_mode_label(pending.mode),
                        pending.position,
                        derived_context_label(ContextBucket::of(pending.position)),
                        pending.incumbent,
                        pending.pooled,
                        pending.legal,
                        pending.caller_k,
                        pending.hi,
                        self.verify_row_limit,
                        self.explicit_mtp_k.map_or_else(|| "null".to_owned(), |k| k.to_string()),
                        JsonF64(pending.chosen_cost_us),
                        JsonF64(pending.chosen_expected),
                        pending.confirmation,
                        self.ngram_proposal.len()
                    );
                }
            }
        }
        self.derived = Some(DerivedPhase::Bound(state));
    }
}

impl Qwen4MtpDrafter {
    pub fn new(max_k: usize, ctx_capacity: usize, end_of_turn: Option<u32>) -> Self {
        let max_k = max_k.clamp(1, DERIVED_MAX_DRAFTS);
        let capacity =
            max_k.min(rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS.saturating_sub(1));
        let mut floor = floor_with_call_log(Qwen4Floor::new(MTP_WINDOW_COST), "drafter_new", true);
        floor.set_capacity(capacity);
        Self {
            max_k,
            explicit_mtp_k: None,
            ctx_capacity,
            request: SpecRequestConfig::default(),
            scratch: None,
            pending_hidden: None,
            row_hidden: None,
            prefill_rows: 0,
            agreement: vec![AGREEMENT_PRIOR; capacity],
            sampled_enabled: hipfire_config::mtp_sampled_enabled(),
            sampled_mode: SampledMode::from_env(),
            sampled: None,
            append_scratch: None,
            end_of_turn,
            think_close: None,
            emitted_seen: None,
            ngram_config: None,
            row_exact_rows: rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS,
            verify_row_limit: rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS,
            spec_rows_cap: usize::MAX,
            copy_lane: false,
            ngram: None,
            ngram_copy: None,
            ngram_active: false,
            ngram_candidates: Vec::new(),
            ngram_yield: (0.0, 0.0),
            ngram_agreement: Vec::new(),
            ngram_proposal: Vec::new(),
            ngram_proposed_at: None,
            ngram_last_first_accepted: false,
            ngram_decision: None,
            ngram_warm_depths: 0,
            ngram_row_cost: None,
            agreement_decay: DEFAULT_DECAY,
            takeover_block: Vec::new(),
            floor,
            floor_enabled: true,
            head_lag: HeadLag::default(),
            stats: MtpRequestStats::default(),
            device_spare: Vec::new(),
            follower: false,
            replay: Vec::new(),
            retained: None,
            retained_pending: None,
            warm_depths: 0,
            warm_ar: false,
            floor_pending: None,
            derived: None,
            derived_position: None,
            measured_load_identity: None,
            derived_margins: None,
            #[cfg(feature = "reference-parity")]
            lab_force_drafts: None,
            #[cfg(feature = "reference-parity")]
            lab_copy_enabled: None,
            #[cfg(feature = "reference-parity")]
            lab_forced_window: None,
            #[cfg(feature = "reference-parity")]
            lab_confirmation: None,
            #[cfg(feature = "reference-parity")]
            lab_requested_confirmation: false,
            #[cfg(feature = "reference-parity")]
            lab_actual_route: None,
            #[cfg(feature = "lab")]
            lab_arm: crate::lab_window::LabWindowArm::default(),
            #[cfg(feature = "lab")]
            lab_open: None,
            #[cfg(feature = "lab")]
            lab_done: std::collections::VecDeque::new(),
        }
    }

    /// Supply this load's immutable measured-load descriptor: the
    /// `Qwen4MeasuredLoadIdentity` cached canonical JSON (`canonical_json`)
    /// from the loader's carriers factory. Stored once as the shared `Arc`,
    /// borrowed by reference at request bind; `None` keeps the real incumbent.
    /// No per-window hashing or serialization, no second cache.
    pub fn with_measured_load_identity(
        mut self,
        identity: Option<std::sync::Arc<crate::measured_identity::Qwen4MeasuredLoadIdentity>>,
    ) -> Self {
        self.measured_load_identity = identity;
        self
    }

    /// Preserve the distinction between a load default and a user's cap.
    pub fn with_explicit_mtp_k_override(mut self, value: Option<usize>) -> Self {
        self.explicit_mtp_k = value.map(|k| k.min(DERIVED_MAX_DRAFTS));
        self
    }

    /// Enable n-gram takeover windows with `config` for requests that arm
    /// `allow_ngram_modifier`; `None` keeps every window native.
    pub fn with_ngram(mut self, config: Option<NgramModConfig>) -> Self {
        self.ngram_config = config;
        let capacity = config.map_or(0, |config| config.n_max.min(NGRAM_DEPTHS));
        self.ngram_agreement.resize(capacity, AGREEMENT_PRIOR);
        self
    }

    /// Run the n-gram source as the copy lane (`on`: take-length chooser,
    /// shadow acceptance, retired requests keeping it, confirmation skipped
    /// once the source is established) instead of the native-MTP takeover.
    /// It runs only when an n-gram config is present
    /// ([`Self::with_ngram`]) and the request arms n-gram; off by default.
    pub fn with_copy_lane(mut self, on: bool) -> Self {
        self.copy_lane = on;
        self
    }

    /// Restrict sampled verification to loads that allow it: `false` turns
    /// it off (sampled requests stay on AR), `true` keeps what
    /// `speculation.mtp_sampled` resolved to; it never turns it on.
    pub fn with_sampled_allowed(mut self, allowed: bool) -> Self {
        self.sampled_enabled &= allowed;
        self
    }

    /// `HIPFIRE_MTP_SAMPLED_MODE=naive` resolved at construction. Under
    /// `topology.pp` that mode is refused (its draws come from the
    /// process-wide AR sampler), so the router keeps such sampled requests on
    /// the PP AR producer.
    pub fn sampled_mode_is_naive(&self) -> bool {
        matches!(self.sampled_mode, Ok(SampledMode::Naive))
    }

    /// The configured request carries a non-neutral repeat / presence /
    /// frequency penalty (the router's definition: the window is not
    /// consulted).
    pub fn request_has_nonneutral_penalties(&self) -> bool {
        self.request.repeat_penalty != 1.0
            || self.request.presence_penalty != 0.0
            || self.request.frequency_penalty != 0.0
    }

    /// A sampled request needs sampled verification enabled (and a valid
    /// `HIPFIRE_MTP_SAMPLED_MODE`).
    fn require_supported_request(&self) -> Result<(), String> {
        if self.sampled.is_some() {
            Ok(())
        } else if let (true, Err(error)) = (self.sampled_enabled, &self.sampled_mode) {
            require_native_greedy(self.request.temp).map_err(|_| error.clone())
        } else {
            require_native_greedy(self.request.temp)
        }
    }

    fn bundle<'a>(target: &'a mut dyn SpecTarget) -> Result<&'a mut Qwen4Bundle, String> {
        target
            .as_any_mut()
            .downcast_mut::<Qwen4Bundle>()
            .ok_or_else(|| "Qwen4MtpDrafter: target is not a Qwen4Bundle".to_string())
    }

    /// Latch this load's speculative verify-row cap from the target (the CPU
    /// splice flag is fixed at load, so the value never changes under a
    /// drafter).
    fn bind_spec_rows(&mut self, target: &mut dyn SpecTarget) -> Result<(), String> {
        self.spec_rows_cap = Self::bundle(target)?.spec_verify_rows_cap();
        self.refresh_chooser_capacity();
        Ok(())
    }

    /// Most drafts one window of this load verifies: the row cap less the seed.
    fn draft_rows_cap(&self) -> usize {
        self.spec_rows_cap.saturating_sub(1)
    }

    /// Refuse a plan that would verify more rows than the CPU splice window
    /// before any device state moves.
    fn check_plan_rows(&self, plan: &WindowPlan) -> Result<(), String> {
        let candidates = if self.follower {
            plan.ngram_candidates.len()
        } else {
            self.ngram_candidates.len()
        };
        check_spec_verify_rows(
            self.spec_rows_cap,
            window_verify_rows(plan.route, plan.ngram_hit, candidates),
            "Qwen4 native MTP window",
        )
    }

    fn ensure_resources(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        target: &mut dyn SpecTarget,
    ) -> Result<(), String> {
        self.bind_arch(ops.gpu()?);
        self.bind_spec_rows(target)?;
        // The copy lane's take cap reads the single card's row-exact rows; a
        // mesh never arms n-gram (refused before any state changes), so only
        // an armed pool under a mesh reaches the refusal.
        if self.copy_lane && (self.ngram_active || !ops.is_mesh()) {
            let rows = Self::bundle(target)?.row_exact_verify_rows(ops.single("n-gram takeover")?);
            self.row_exact_rows = rows;
        }
        // Every native chooser follows the admitted executor, not a profile's
        // historical last measured depth. Mesh paths remain conservative.
        if !ops.is_mesh() {
            let rows = Self::bundle(target)?.row_exact_verify_rows(ops.single("derived K")?);
            self.derived_refresh_rows(rows);
        }
        let width = {
            let bundle = Self::bundle(target)?;
            bundle.mtp_position().map_err(|error| error.to_string())?;
            bundle
                .config
                .hc_count
                .checked_mul(bundle.config.hidden_size)
                .ok_or_else(|| "Qwen4 MTP hidden width overflow".to_string())?
        };
        let capacity = self.proposal_capacity();
        let copy_lane = self.copy_lane;
        let prefill_rows = {
            let bundle = Self::bundle(target)?;
            // Every band's ring under a mesh: a verify's per-row repair is
            // taken by each band from its own captured rows.
            if native_mtp_row_capture(ops.gpu()?, &bundle.config) {
                // Verify blocks up to the arch's row-exact limit keep capture
                // rollback: the ring holds the wider of the window (a copy
                // lane's `proposal_capacity + 1`, else an MTP window
                // `max_k + 1`) and the per-GPU row-exact verify rows (the
                // census charges the same ring, `native_mtp_capture`).
                let window_rows = if copy_lane {
                    capacity + 1
                } else {
                    self.k_ceiling() + 1
                };
                let capture_rows = native_mtp_capture_rows(ops.gpu()?, window_rows);
                bundle.ops_ensure_row_capture(ops, capture_rows)?;
            }
            bundle.ops_chunk_rows(ops).unwrap_or(1).max(1)
        };
        if copy_lane {
            // An armed copy-lane takeover verifies up to `take_cap + 1`
            // rows, which can exceed `max_k + 1` (and a request may arm
            // n-gram after the first one allocated): a scratch too small for
            // it is replaced.
            let block_rows = (capacity + 1).max(prefill_rows);
            let too_small = self
                .scratch
                .as_mut()
                .and_then(|scratch| scratch.as_any_mut().downcast_mut::<Qwen4SpecScratch>())
                .is_some_and(|scratch| scratch.block_size < block_rows);
            if too_small {
                if let Some(old) = self.scratch.take() {
                    old.free(ops.gpu()?);
                }
            }
            if self.scratch.is_none() {
                let scratch = Self::bundle(target)?.ops_new_spec_scratch(ops, block_rows)?;
                self.scratch = Some(scratch);
            }
        } else if self.scratch.is_none() {
            let scratch = Self::bundle(target)?
                .ops_new_spec_scratch(ops, (self.k_ceiling() + 1).max(prefill_rows))?;
            self.scratch = Some(scratch);
        }
        self.prefill_rows = prefill_rows;
        if self.pending_hidden.is_none() {
            self.pending_hidden = Some(
                ops.gpu()?
                    .zeros(&[width], rdna_compute::DType::F32)
                    .map_err(|error| format!("Qwen4 MTP pending hidden allocation: {error}"))?,
            );
        }
        if self.row_hidden.is_none() {
            self.row_hidden = Some(
                ops.gpu()?
                    .zeros(&[width], rdna_compute::DType::F32)
                    .map_err(|error| format!("Qwen4 MTP row hidden allocation: {error}"))?,
            );
        }
        if self.retained_pending.is_none() {
            self.retained_pending = Some(
                ops.gpu()?
                    .zeros(&[width], rdna_compute::DType::F32)
                    .map_err(|error| {
                        format!("Qwen4 MTP retained pending hidden allocation: {error}")
                    })?,
            );
        }
        if self.append_scratch.is_none() && mtp_fill_scratch_wanted(ops.gpu()?) {
            let bundle = Self::bundle(target)?;
            if bundle.mtp_append_rows_supported(ops.gpu()?) {
                let rows = MTP_FILL_ROWS.min(prefill_rows);
                let scratch = MtpAppendScratch::new(ops.gpu()?, &bundle.config, rows, prefill_rows)
                    .map_err(|error| format!("Qwen4 MTP append scratch allocation: {error}"))?;
                self.append_scratch = Some(scratch);
            }
        }
        self.ensure_device_accept(ops.gpu()?)?;
        Ok(())
    }

    /// Provision the current sampled request's resident acceptance buffers
    /// (and compile their kernels) before any timed window: a no-op unless
    /// the request is device-eligible on an exact supported architecture.
    /// Reuses the request's own buffers, then a spare of an earlier request;
    /// grows (free, reallocate) only when the capacity is too small. Buffers
    /// the current request cannot use stay in the spares (never dropped).
    fn ensure_device_accept(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        let capacity = self.proposal_capacity();
        let Some(sampled) = self.sampled.as_mut() else {
            return Ok(());
        };
        if !sampled.device_eligible || !sampled.draws_drafts() {
            return Ok(());
        }
        if sampled
            .device
            .as_ref()
            .is_some_and(|device| device.capacity >= capacity)
        {
            return Ok(());
        }
        if !mtp_device_sampler_supported(gpu) {
            return Ok(());
        }
        // Provision point-mass q storage before the first takeover window.
        sampled.points.resize_with(capacity, || {
            let mut point = SparseDist::default();
            point.set_point_mass(0);
            point
        });
        if let Some(small) = sampled.device.take() {
            self.device_spare.push(small);
        }
        let found = self
            .device_spare
            .iter()
            .position(|device| device.capacity >= capacity);
        let device = match found {
            Some(index) => self.device_spare.swap_remove(index),
            None => {
                // This request outgrows every spare: retire them first.
                for small in std::mem::take(&mut self.device_spare) {
                    small.free(gpu);
                }
                DeviceAccept::new(gpu, capacity)?
            }
        };
        if let Some(sampled) = self.sampled.as_mut() {
            sampled.device = Some(device);
        }
        Ok(())
    }

    fn row_hidden(&self) -> Result<&GpuTensor, String> {
        self.row_hidden
            .as_ref()
            .ok_or_else(|| "Qwen4 MTP row hidden is not allocated".to_string())
    }

    /// Incremental verify: interleave one target row with one draft step and
    /// stop at the first rejection.
    ///
    /// Every row is committed as it is produced, so the target state advances
    /// by exactly the rows that were verified and the head state advances with
    /// it: no snapshot, no restore, no re-forward of the accepted prefix.  A
    /// cycle costs `accepted + 1` single-row forwards instead of a `k+1`-row
    /// batch plus a replay of the same prefix, and no draft step is wasted past
    /// the rejection point. This is the default; `HIPFIRE_MTP_INCREMENTAL=0`
    /// explicitly selects batching.
    /// QSA reselects on row 0 of each window, including consecutive windows
    /// at nonzero request positions; later rows reuse that window's selection.
    /// With `sampled`, each row's draft is drawn from its `q` and verified
    /// against the row's `p` instead of matched against the argmax.
    #[allow(clippy::too_many_arguments)]
    fn mtp_step_incremental(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        k: usize,
        eos: u32,
        trace: bool,
        mut sampled: Option<&mut SampledVerify>,
    ) -> Result<MtpWindow, String> {
        let mut timers = MtpPhaseTimers::new().silenced(self.follower);
        let window_start = Instant::now();
        let mut committed: Vec<u32> = Vec::with_capacity(k + 1);
        let mut drafts: Vec<u32> = Vec::with_capacity(k);
        let mut picks: Vec<u32> = Vec::with_capacity(k + 1);
        let mut token = seed;
        let mut accepted = 0usize;
        let mut row = 0usize;
        let device_rows = sampled.as_deref().is_some_and(|s| s.device_ready(k));
        let skip_picks = sampled
            .as_deref()
            .is_some_and(|s| s.device_skip_picks(k, trace || timers.enabled()));
        loop {
            let token_position = position
                .checked_add(row)
                .ok_or_else(|| "Qwen4 MTP incremental position overflow".to_string())?;
            timers.mark(ops.gpu()?, "target_row");
            let mut pick = Self::bundle(target)?.ops_capture_token(ops, token, !skip_picks)?;
            match sampled.as_deref_mut() {
                // Naive: the row's target token is its AR-sampler draw.
                Some(s) if !s.draws_drafts() => {
                    pick = s.naive_draw(ops.gpu()?, Self::bundle(target)?, 0, row)?
                }
                Some(s) if device_rows => {
                    s.device_load_target(ops.gpu()?, Self::bundle(target)?, 0)?
                }
                Some(s) => s.load_target(ops.gpu()?, Self::bundle(target)?, 0, row)?,
                None => {}
            }
            if !skip_picks {
                picks.push(pick);
            }
            let row_hidden = self.row_hidden()?;
            {
                let bundle = Self::bundle(target)?;
                bundle
                    .copy_spec_hidden_row_to(ops.gpu()?, 0, row_hidden)
                    .map_err(|error| error.to_string())?;
            }
            if row == k {
                // Every draft matched: this row's pick is the bonus.  The head
                // must still consume the row's token so its own position lands on
                // the committed end, exactly as the batched path's full-accept
                // branch does with one extra `mtp_forward_token`.
                timers.mark(ops.gpu()?, "draft_step");
                {
                    // `pending` is the target hidden of `token_position - 1`.
                    let pending = self.pending_hidden()?;
                    let bundle = Self::bundle(target)?;
                    bundle
                        .mtp_append_token(ops.gpu()?, token, Some(pending), token_position)
                        .map_err(|error| error.to_string())?;
                }
                committed.push(match sampled.as_deref_mut() {
                    Some(s) if device_rows => s.device_bonus(ops.gpu()?)?,
                    Some(s) if s.draws_drafts() => s.host_bonus(),
                    _ => pick,
                });
                break;
            }
            timers.mark(ops.gpu()?, "draft_step");
            // Every row pairs its token with the target hidden of the row
            // before it, which this route has verified by now: `pending`
            // carries it (the window's pending hidden for row 0, then the
            // previous row's own, moved on when its draft is accepted).
            let mut draft = {
                let pending = self.pending_hidden()?;
                let bundle = Self::bundle(target)?;
                bundle
                    .mtp_forward_token(ops.gpu()?, token, Some(pending), token_position, row == 0)
                    .map_err(|error| error.to_string())?
            };
            let verdict = match sampled.as_deref_mut() {
                Some(s) if s.draws_drafts() => {
                    draft = s.sample_draft(ops.gpu()?, Self::bundle(target)?, 0, row)?;
                    if device_rows {
                        s.device_step_verdict(ops.gpu()?, draft, false)?
                    } else {
                        s.host_step_verdict(draft, false)
                    }
                }
                _ if draft == pick => DraftVerdict::Accept,
                _ => DraftVerdict::Reject(pick),
            };
            drafts.push(draft);
            match verdict {
                DraftVerdict::Accept => {
                    accepted += 1;
                    committed.push(draft);
                    // The accepted draft is history for the next row (and
                    // the bonus); a rejected one never enters it.
                    if let Some(s) = sampled.as_deref_mut() {
                        s.history.push_draft(draft);
                    }
                    if draft == eos {
                        break;
                    }
                    // The next row pairs with this row's own target hidden.
                    {
                        let pending = self.pending_hidden()?;
                        ops.gpu()?
                            .copy_d2d(row_hidden, pending, pending.byte_size())
                            .map_err(|error| {
                                format!("Qwen4 MTP incremental pending hidden copy: {error}")
                            })?;
                    }
                    token = draft;
                    row += 1;
                }
                DraftVerdict::Reject(replacement) => {
                    committed.push(replacement);
                    break;
                }
            }
        }
        if committed.is_empty() {
            return Err("Qwen4 MTP incremental window committed no token".to_string());
        }
        // The next window's step-0 conditioning hidden is the hidden of the last
        // committed token, which is the row just captured: the baseline path
        // copies row `target_commit_accept_len` (= the last processed row) of its verify
        // capture, and this is the same row of the same forward shape.
        {
            let row_hidden = self.row_hidden()?;
            let pending = self.pending_hidden()?;
            ops.gpu()?
                .copy_d2d(row_hidden, pending, pending.byte_size())
                .map_err(|error| format!("Qwen4 MTP incremental pending hidden copy: {error}"))?;
        }
        let committed_end = position
            .checked_add(committed.len())
            .ok_or_else(|| "Qwen4 MTP incremental commit overflow".to_string())?;
        let mtp_end = Self::bundle(target)?
            .mtp_position()
            .map_err(|error| error.to_string())?;
        if mtp_end != committed_end {
            return Err(format!(
                "Qwen4 MTP incremental transaction ended at mtp={mtp_end}, expected {committed_end}"
            ));
        }
        if (trace || timers.enabled()) && !self.follower {
            let rows = drafts
                .iter()
                .zip(picks.iter())
                .map(|(draft, pick)| {
                    format!(
                        "{{\"draft\":{draft},\"pick\":{pick},\"match\":{}}}",
                        draft == pick
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            let gdn_resets = gdn_residual_trace_field(&Self::bundle(target)?.state);
            let row_limit = Self::bundle(target)?.row_exact_verify_rows(ops.gpu()?);
            eprintln!(
                "QWEN4_MTP_TRACE {{\"event\":\"window\",\"source\":\"mtp\",\"verify_route\":\"interleaved\",\"accept_route\":\"{}\",\"position\":{position},\"k\":{k},\"accepted\":{accepted},\"verify_rows\":1,\"row_limit\":{row_limit},\"rows\":[{rows}],\"committed\":{committed:?}{gdn_resets}}}",
                accept_route_label(sampled.as_deref())
            );
        }
        if timers.enabled() {
            let window_us = synced_wall_us(ops.gpu()?, window_start);
            let fields = format!(
                "\"source\":\"mtp\",\"verify_route\":\"interleaved\",\"accept_route\":\"{}\",\"position\":{position},\"k\":{},\"accepted\":{accepted},\"emitted\":{},\"window_us\":{window_us:.1},\"pairing\":\"teacher-forced\",\"t_end\":{}",
                accept_route_label(sampled.as_deref()),
                drafts.len(),
                committed.len(),
                unix_micros()
            );
            timers.finish(ops.gpu()?, "QWEN4_MTP_PHASE", &fields);
        }
        Ok(MtpWindow {
            committed,
            accepted,
            drafts_generated: drafts.len(),
        })
    }

    /// See the free [`observe_agreement`].
    fn observe_agreement(&mut self, window: &MtpWindow) {
        observe_agreement(&mut self.agreement, window, self.agreement_decay);
    }

    /// With no agreement evidence, start at K=3 (or the smaller budget);
    /// evidence may still grow the chooser to the unchanged configured cap.
    fn batched_depth(&self, k: usize) -> usize {
        if self.agreement.iter().all(|&entry| entry == AGREEMENT_PRIOR) {
            k.min(3)
        } else {
            self.floor.provisional_depth(&self.agreement, k)
        }
    }

    /// Price both sources on the same clock; unsupported arches retain the
    /// original chooser. Depth drafts require depth + 1 verify rows.
    fn ngram_window_cost(&self, depth: usize) -> f32 {
        let Some((single, extra)) = self.ngram_row_cost else {
            return MTP_WINDOW_COST.get(depth).copied().unwrap_or(f32::INFINITY);
        };
        if self.floor_enabled {
            if let Some(cost) = self.floor.batched_cost(depth) {
                return cost;
            }
        }
        let ar_ms = if self.floor_enabled && valid_us(self.floor.ar_us()) {
            self.floor.ar_us() / 1000.0
        } else {
            single
        };
        // Explicit few-row buckets from the measured fits. Nine or more
        // rows use a different (expert-sharing) batched route: without its
        // measurements do not extrapolate or assume it is cheap.
        let row_ms = if extra == 6.9 {
            [30.5, 37.4, 44.3, 51.2, 58.1, 65.0, 71.9, 78.8]
        } else {
            [30.3, 45.1, 59.9, 74.7, 89.5, 104.3, 119.1, 133.9]
        };
        row_ms.get(depth).copied().unwrap_or(f32::INFINITY) / ar_ms
    }

    /// Best expected emitted tokens per unit of window cost the native route
    /// offers at the current agreement (the interleaved route is 1.0): on a
    /// priced arch with a measured floor, the floor's cheapest candidate
    /// within `k` (the depth the chooser would run,
    /// [`Qwen4Floor::candidate_rate`]); before that, `batched_depth`'s
    /// objective over [`Self::ngram_window_cost`].
    fn native_rate(&self, k: usize) -> f32 {
        let depths = k.min(self.agreement.len());
        if self.floor_enabled && self.ngram_row_cost.is_some() {
            if let Some(rate) = self.floor.candidate_rate(&self.agreement, depths) {
                return rate;
            }
        }
        let mut best = 1.0f32;
        let mut prefix = 1.0f32;
        let mut expected = 1.0f32;
        let depths = self.agreement.iter().take(depths);
        for (i, &(accepted, total)) in depths.enumerate() {
            let cost = self.ngram_window_cost(i + 1);
            prefix *= accepted / total.max(f32::MIN_POSITIVE);
            expected += prefix;
            best = best.max(expected / cost);
        }
        best
    }

    /// Whether an `n`-candidate takeover is expected to emit at least as much
    /// per unit cost as the native window it displaces. N-gram hits and native
    /// drafts can succeed on the same spans (a pool hit on a span the model
    /// rewrites displaces a better native window), so the source is chosen by
    /// each source's own measured yield: the takeover's decayed emitted per
    /// window (prior `NGRAM_YIELD_PRIOR` tokens, one pseudo-window) against
    /// the native agreement's best rate. Declined hits let the takeover
    /// history decay back toward the prior, so the n-gram source is re-tried.
    fn ngram_wins(&self, n: usize, k: usize) -> bool {
        let (emitted, windows) = self.ngram_yield;
        let expected = (emitted + NGRAM_YIELD_PRIOR) / (windows + 1.0);
        let cost = self.ngram_window_cost(n);
        expected / cost >= self.native_rate(k.min(self.incumbent_max_k()))
    }

    /// Extend only a well-established copy source, never a fresh or uncertain
    /// match. Configured n_max remains the hard bound, so defaults stay 5/3/3.
    fn ngram_takeover_depth(&self, matched: usize, current: usize) -> usize {
        let base = matched.min(current);
        if matched <= current
            || self.ngram_row_cost.is_none()
            || self.stats.ngram_mod_windows < 5
            || self.stats.ngram_mod_drafts == 0
        {
            return base;
        }
        let acceptance = self.stats.ngram_mod_accepted as f32 / self.stats.ngram_mod_drafts as f32;
        if acceptance < 0.95 {
            return base;
        }
        let (emitted, windows) = self.ngram_yield;
        let mut expected = (emitted + NGRAM_YIELD_PRIOR) / (windows + 1.0);
        let mut best = (base, expected / self.ngram_window_cost(base));
        let mut prefix = acceptance.powi(base as i32);
        for depth in base + 1..=matched {
            prefix *= acceptance;
            expected += prefix;
            let rate = expected / self.ngram_window_cost(depth);
            if rate > best.1 && rate >= self.native_rate(current.min(self.incumbent_max_k())) {
                best = (depth, rate);
            }
        }
        best.0
    }

    /// Candidates a copy-lane takeover may take at most
    /// ([`qwen4_ngram_take_cap`] at this GPU's row-exact verify rows); 0 with
    /// no pool configured.
    fn ngram_take_cap(&self) -> usize {
        self.ngram_config.map_or(0, |config| {
            qwen4_ngram_take_cap(&config, self.row_exact_rows).min(self.draft_rows_cap())
        })
    }

    /// Window cost of a takeover of `d` candidates in AR-step units: the
    /// request's measured takeover costs on a priced arch with the floor on
    /// ([`Qwen4Floor::takeover_cost`]); before that, and on unpriced arches,
    /// the static price [`Self::ngram_window_cost`].
    fn takeover_cost(&self, d: usize) -> f32 {
        if self.floor_enabled && self.ngram_row_cost.is_some() {
            if let Some(cost) = self.floor.takeover_cost(d) {
                return cost;
            }
        }
        self.ngram_window_cost(d)
    }

    /// Compare the last plan's n-gram proposal with the tokens committed
    /// since (`emitted[start..]`, the window that followed it) and observe
    /// the pool's per-depth acceptance, whichever source ran that window:
    /// native, takeover and AR windows alike. `m` is the common prefix of
    /// proposal and committed tokens and the comparison is censored at the
    /// committed length, so depths past it are neither accepted nor
    /// rejected. On a greedy stream the match is exactly what a verify of
    /// the proposal would have accepted. On a sampled stream it is an
    /// unbiased estimate of the point-mass acceptance: an independent
    /// target draw equals a candidate `c` with probability `p(c)`.
    fn observe_ngram_shadow(&mut self, emitted: &[u32]) {
        let Some(start) = self.ngram_proposed_at else {
            return;
        };
        if emitted.len() < start {
            self.ngram_proposed_at = None;
            return;
        }
        if emitted.len() == start {
            return;
        }
        let truth = &emitted[start..];
        let matched = self
            .ngram_proposal
            .iter()
            .zip(truth)
            .take_while(|(proposed, committed)| proposed == committed)
            .count();
        let horizon = self.ngram_proposal.len().min(truth.len());
        observe_agreement_counts(
            &mut self.ngram_agreement,
            matched,
            horizon,
            self.agreement_decay,
        );
        self.ngram_last_first_accepted = matched >= 1;
        self.ngram_proposed_at = None;
    }

    /// Route with the floor opted out (`HIPFIRE_MTP_AR_FLOOR=0`): the
    /// pre-floor per-window chooser, `0` = interleaved, else batched at that
    /// depth.
    fn floor_off_depth(&self, k: usize) -> usize {
        self.batched_depth(k)
    }
    /// One native window of the AR floor: the ordinary target-only token
    /// `seed` at `position`, with a sampled draw when the request is sampled.
    ///
    /// `calibrate` (the request's first window) times the token including
    /// that draw, then, outside the timing, keeps the head synced exactly as
    /// a `k = 0` interleaved window does (the ordinary forward's wide hidden,
    /// head append of the seed with the pending hidden of `position - 1`, then
    /// the pending hidden moves to the seed's own)
    /// before recording the AR price.  Otherwise (retired) it does no head
    /// work and never checks the head position: the head owes the token's
    /// row ([`Self::owe_head_row`]).
    fn mtp_ar_step(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        calibrate: bool,
    ) -> Result<MtpWindow, String> {
        let end = position
            .checked_add(1)
            .ok_or_else(|| "Qwen4 MTP AR step position overflow".to_string())?;
        {
            let bundle = Self::bundle(target)?;
            let target_position = bundle.state.position;
            let mtp_position = if calibrate {
                bundle.mtp_position().map_err(|error| error.to_string())?
            } else {
                position
            };
            if target_position != position || mtp_position != position {
                return Err(format!(
                    "Qwen4 MTP AR step position mismatch: target={target_position}, mtp={mtp_position}, position={position}"
                ));
            }
        }
        if calibrate {
            // Prefill's trailing head work must not be billed to AR.
            ops.gpu()?
                .hip
                .device_synchronize()
                .map_err(|error| format!("Qwen4 MTP AR calibration sync: {error}"))?;
        }
        let started = Instant::now();
        let mut sampled = self.sampled.take();
        let drawn = (|| -> Result<u32, String> {
            let mut pick = Self::bundle(target)?.ops_ar_token(ops, seed)?;
            match sampled.as_mut() {
                Some(s) if !s.draws_drafts() => {
                    pick = s.naive_draw(ops.gpu()?, Self::bundle(target)?, 0, 0)?
                }
                Some(s) => {
                    s.load_target(ops.gpu()?, Self::bundle(target)?, 0, 0)?;
                    pick = s.host_bonus();
                }
                None => {}
            }
            Ok(pick)
        })();
        self.sampled = sampled;
        let pick = drawn?;
        let us = elapsed_us(started);
        let target_end = Self::bundle(target)?.state.position;
        if target_end != end {
            return Err(format!(
                "Qwen4 MTP AR step ended at target={target_end}, expected {end}"
            ));
        }
        if calibrate {
            let row_hidden = self.row_hidden()?;
            let pending = self.pending_hidden()?;
            {
                let bundle = Self::bundle(target)?;
                bundle
                    .copy_ar_hidden_to(ops.gpu()?, row_hidden)
                    .map_err(|error| error.to_string())?;
                // The seed pairs with the target hidden of `position - 1`.
                bundle
                    .mtp_append_token(ops.gpu()?, seed, Some(pending), position)
                    .map_err(|error| error.to_string())?;
                let mtp_end = bundle.mtp_position().map_err(|error| error.to_string())?;
                if mtp_end != end {
                    return Err(format!(
                        "Qwen4 MTP AR calibration ended at mtp={mtp_end}, expected {end}"
                    ));
                }
            }
            let gpu = ops.gpu()?;
            gpu.copy_d2d(row_hidden, pending, pending.byte_size())
                .map_err(|error| {
                    format!("Qwen4 MTP AR calibration pending hidden copy: {error}")
                })?;
            // Drain the calibration's head work so the first probe window
            // does not bill it.
            gpu.hip
                .device_synchronize()
                .map_err(|error| format!("Qwen4 MTP AR calibration tail sync: {error}"))?;
            self.floor_observe_ar(us);
            self.stats.mtp_windows += 1;
            self.stats.floor_ar_windows += 1;
            self.stats.mtp_retired = self.floor.retired();
        } else {
            self.owe_head_row(ops, target, position, seed)?;
            if floor_trace_enabled() {
                eprintln!(
                    "{}",
                    self.floor
                        .trace_line(FLOOR_TRACE_PREFIX, &self.agreement, "ar", us, 1, 0)
                );
            }
            self.stats.ar_windows += 1;
        }
        Ok(MtpWindow {
            committed: vec![pick],
            accepted: 0,
            drafts_generated: 0,
        })
    }

    /// Owed head rows a retired request holds before it catches up: one
    /// batched decode fill (`MtpAppendScratch` rows), inside the spec hidden
    /// capture both are sized from.
    fn head_lag_capacity(&self) -> usize {
        MTP_FILL_ROWS.min(self.prefill_rows).max(1)
    }

    /// The retired AR token `seed` at `position`, just forwarded by the
    /// target alone: copy its wide hidden (valid until the next forward) to
    /// the next spec hidden row and owe its head row. A full record catches
    /// the head up at once.
    fn owe_head_row(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
    ) -> Result<(), String> {
        let capacity = self.head_lag_capacity();
        let row = self.head_lag.next_row(position, capacity)?;
        Self::bundle(target)?
            .copy_ar_hidden_to_spec_row(ops.gpu()?, row)
            .map_err(|error| error.to_string())?;
        self.head_lag.push(position, seed);
        if self.head_lag.tokens.len() >= capacity {
            self.catch_up_head(ops, target)?;
        }
        Ok(())
    }

    /// Append the head rows a retired request owes ([`HeadLag`]),
    /// teacher-forced (token `p` with the target hidden of `p - 1`: the
    /// pending hidden, still the head's position - 1, for the first owed
    /// token) exactly as a takeover appends its kept rows: one batched decode
    /// fill from two rows when it runs, else one Append per row. Head and
    /// target then sit at the same
    /// position, and the pending hidden is the last row's, as after any
    /// teacher-forced advance. A no-op when nothing is owed.
    fn catch_up_head(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        target: &mut dyn SpecTarget,
    ) -> Result<(), String> {
        let owed = self.head_lag.tokens.len();
        if owed == 0 {
            return Ok(());
        }
        let start = self.head_lag.start;
        let end = start + owed;
        let bundle = Self::bundle(target)?;
        let head = bundle.mtp_position().map_err(|error| error.to_string())?;
        if head != start || bundle.state.position != end {
            return Err(format!(
                "Qwen4 MTP head catch-up position mismatch: target={}, mtp={head}, owed rows {start}..{end}",
                bundle.state.position
            ));
        }
        {
            let row_hidden = self
                .row_hidden
                .as_ref()
                .ok_or_else(|| "Qwen4 MTP row hidden is not allocated".to_string())?;
            let pending = self
                .pending_hidden
                .as_ref()
                .ok_or_else(|| "Qwen4 MTP pending hidden is not allocated".to_string())?;
            fill_head_rows(
                bundle,
                ops.gpu()?,
                self.append_scratch.as_mut(),
                row_hidden,
                pending,
                &self.head_lag.tokens,
                start,
            )?;
        }
        let head = bundle.mtp_position().map_err(|error| error.to_string())?;
        if head != end {
            return Err(format!(
                "Qwen4 MTP head catch-up ended at mtp={head}, expected {end}"
            ));
        }
        let pending = self
            .pending_hidden
            .as_ref()
            .ok_or_else(|| "Qwen4 MTP pending hidden is not allocated".to_string())?;
        bundle
            .copy_spec_hidden_row_to(ops.gpu()?, owed - 1, pending)
            .map_err(|error| error.to_string())?;
        self.head_lag.clear();
        Ok(())
    }

    /// Drop the retained window (see [`RetainedWindow`]): the owners move on,
    /// so a terminal stop can no longer be repaired from it.
    fn release_retained(&mut self, target: &mut dyn SpecTarget) {
        if let Some(window) = self.retained.take() {
            if let Ok(bundle) = Self::bundle(target) {
                bundle.ops_release_window(window.target, window.head);
            }
        }
    }

    /// Whether the one-row windows (interleaved native and takeover) take
    /// pre-window tickets: only when the request has stop sequences, which
    /// can end a turn inside a window no other signal predicts. The batched
    /// routes snapshot for their own rollback and retain those tickets for
    /// free; the one-row routes pay a snapshot copy per window for it. Never
    /// under EP (state snapshots are refused there).
    fn keeps_one_row_windows(&self, target: &mut dyn SpecTarget) -> bool {
        self.request.stop_sequences
            && Self::bundle(target).is_ok_and(|bundle| bundle.ep_mesh().is_none())
    }

    /// Snapshot target and head before a one-row window. Under a mesh one
    /// aggregate window (every band and the head, never armed: the window's
    /// target rows are one-row forwards) spans up to `rows` rows.
    fn open_window_tickets(
        ops: &mut Qwen4Ops<'_>,
        target: &mut dyn SpecTarget,
        rows: usize,
    ) -> Result<(TargetTicket, HeadTicket), String> {
        let bundle = Self::bundle(target)?;
        if ops.is_mesh() {
            let gpus = ops.gpus()?;
            let window = bundle
                .pp_window_open_unarmed(gpus, rows, true)
                .map_err(|error| error.to_string())?;
            return Ok((TargetTicket::Mesh(window), HeadTicket::Mesh(window)));
        }
        let gpu = ops.gpu()?;
        let target_ticket = bundle.snapshot(gpu).map_err(|error| error.to_string())?;
        match bundle.mtp_snapshot(gpu) {
            Ok(head_ticket) => Ok((
                TargetTicket::Local(target_ticket),
                HeadTicket::Local(head_ticket),
            )),
            Err(error) => {
                bundle.commit_validated(target_ticket);
                Err(error.to_string())
            }
        }
    }

    /// Save the pending hidden (the target hidden of the coming window's
    /// `start - 1`) into `retained_pending`, before the window moves
    /// `pending_hidden` on: a terminal repair replays from it.
    fn save_retained_pending(&self, gpu: &Gpu) -> Result<(), String> {
        let pending = self.pending_hidden()?;
        let saved = self
            .retained_pending
            .as_ref()
            .ok_or_else(|| "Qwen4 MTP retained pending hidden is not allocated".to_string())?;
        gpu.copy_d2d(pending, saved, pending.byte_size())
            .map_err(|error| format!("Qwen4 MTP retained pending hidden copy: {error}"))
    }

    /// The tickets of a one-row window when the request keeps them
    /// ([`Self::keeps_one_row_windows`]), with the pre-window pending hidden
    /// saved beside them. `rows` is the mesh aggregate window's span (see
    /// [`Self::open_window_tickets`]).
    fn open_one_row_tickets(
        &self,
        ops: &mut Qwen4Ops<'_>,
        target: &mut dyn SpecTarget,
        rows: usize,
    ) -> Result<Option<(TargetTicket, HeadTicket)>, String> {
        if !self.keeps_one_row_windows(target) {
            return Ok(None);
        }
        self.save_retained_pending(ops.gpu()?)?;
        Self::open_window_tickets(ops, target, rows).map(Some)
    }

    /// End a one-row window's tickets: a committed window retains them (it
    /// ended at the target's position), a failed one releases them; the
    /// window's own error handling is unchanged.
    fn close_window_tickets(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        target: &mut dyn SpecTarget,
        tickets: Option<(TargetTicket, HeadTicket)>,
        start: usize,
        seed: u32,
        committed: bool,
    ) {
        let Some((target_ticket, head_ticket)) = tickets else {
            return;
        };
        let Ok(bundle) = Self::bundle(target) else {
            return;
        };
        if committed
            && bundle
                .ops_finish_window(ops, target_ticket, head_ticket, true)
                .is_ok()
        {
            self.retained = Some(RetainedWindow {
                span: WindowSpan {
                    start,
                    end: bundle.state.position,
                    seed,
                },
                target: target_ticket,
                head: head_ticket,
            });
        } else {
            bundle.ops_release_window(target_ticket, head_ticket);
        }
    }

    /// The restore-and-replay body of [`MtpDrafter::mtp_repair_terminal_prefix`]:
    /// both owners back to the window's start, then `replay` forwarded through
    /// the target and appended to the head teacher-forced, exactly as a
    /// takeover appends its kept rows ([`fill_head_rows`]): the seed's head
    /// row pairs with the window's pre-window pending hidden
    /// (`retained_pending`; `pending_hidden` has moved on since), each later
    /// row with the replayed row before it. The retained tickets stay active
    /// for the caller to release.
    fn replay_window_prefix(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        target: &mut dyn SpecTarget,
        window: RetainedWindow,
        replay: &[u32],
    ) -> Result<(), String> {
        let start = window.span.start;
        let end = start + replay.len();
        let bundle = Self::bundle(target)?;
        bundle.require_spec_verify_rows(replay.len(), "Qwen4 terminal repair replay")?;
        match (window.target, window.head) {
            (TargetTicket::Mesh(ticket), HeadTicket::Mesh(_)) => {
                // Every band and the head back to the window's start in one
                // aggregate restore (the ticket stays valid), then the same
                // replay forward over every band.
                bundle
                    .pp_window_restore(ops.gpus()?, &ticket, true)
                    .map_err(|error| format!("Qwen4 terminal repair window restore: {error}"))?;
            }
            (TargetTicket::Local(target_ticket), HeadTicket::Local(head_ticket)) => {
                bundle
                    .restore_retain(ops.gpu()?, target_ticket)
                    .map_err(|error| format!("Qwen4 terminal repair target restore: {error}"))?;
                bundle
                    .mtp_restore_retain(ops.gpu()?, head_ticket)
                    .map_err(|error| format!("Qwen4 terminal repair head restore: {error}"))?;
            }
            _ => return Err("Qwen4 terminal repair: target and head tickets disagree".to_string()),
        }
        // `replay` is never empty: a repair needs a consumed token.
        bundle
            .ops_spec_rows(ops, replay, false)
            .map_err(|error| format!("Qwen4 terminal repair replay: {error}"))?;
        {
            let row_hidden = self
                .row_hidden
                .as_ref()
                .ok_or_else(|| "Qwen4 MTP row hidden is not allocated".to_string())?;
            let pending = self
                .pending_hidden
                .as_ref()
                .ok_or_else(|| "Qwen4 MTP pending hidden is not allocated".to_string())?;
            let saved = self
                .retained_pending
                .as_ref()
                .ok_or_else(|| "Qwen4 MTP retained pending hidden is not allocated".to_string())?;
            // The head is back at `start`: its row pairs with the hidden of
            // `start - 1`, which the window saved.
            ops.gpu()?
                .copy_d2d(saved, pending, pending.byte_size())
                .map_err(|error| format!("Qwen4 terminal repair pending restore: {error}"))?;
            fill_head_rows(
                bundle,
                ops.gpu()?,
                self.append_scratch.as_mut(),
                row_hidden,
                pending,
                replay,
                start,
            )
            .map_err(|error| format!("Qwen4 terminal repair head fill: {error}"))?;
        }
        let pending = self
            .pending_hidden
            .as_ref()
            .ok_or_else(|| "Qwen4 MTP pending hidden is not allocated".to_string())?;
        bundle
            .copy_spec_hidden_row_to(ops.gpu()?, replay.len() - 1, pending)
            .map_err(|error| error.to_string())?;
        let head = bundle.mtp_position().map_err(|error| error.to_string())?;
        if bundle.state.position != end || head != end {
            return Err(format!(
                "Qwen4 terminal repair ended at target={} mtp={head}, expected {end}",
                bundle.state.position
            ));
        }
        Ok(())
    }

    /// Observe one timed calibration-path AR token (the request's
    /// calibration token or a retire probe; `us` from its forward to its
    /// draw) for the floor; a replica rank's floor is never consulted.  The
    /// drafter's first such token is not observed: it pays the decode
    /// route's one-time kernel binds (on Halo the process's first decode
    /// token cost 55.6 ms and the next request's 30.0 ms, about 140 ms in the
    /// ags run whose depth collapsed, and 9.35 s once), and as the AR price
    /// it would let every losing request keep drafting (nothing measured
    /// costs as much per token) and price each extra draft at several real
    /// rows (the floor's row-cost prior, 0.27 AR steps).  The floor still needs the
    /// sample, so the next window is a calibration token again: one extra AR
    /// token per drafter, the same rule as its first window at each depth
    /// ([`Self::finish_floor_window`]).
    fn floor_observe_ar(&mut self, us: f32) {
        let first_use = !std::mem::replace(&mut self.warm_ar, true);
        if self.follower {
            return;
        }
        let event = if first_use {
            "calibration_first_use"
        } else if self.floor.calibrated() {
            "ar_probe"
        } else {
            "calibration"
        };
        if !first_use {
            self.floor.observe_calibration(us, &self.agreement);
        }
        if floor_trace_enabled() {
            eprintln!(
                "{}",
                self.floor
                    .trace_line(FLOOR_TRACE_PREFIX, &self.agreement, event, us, 1, 0)
            );
        }
    }

    /// Count one finished native window that started at `start` and set its
    /// floor observation aside, unless `bypass` (a developer-forced route,
    /// `HIPFIRE_MTP_INCREMENTAL=0|1`, or `HIPFIRE_MTP_AR_FLOOR=0`) skips the
    /// floor.  The window is priced by the wall time from its start to the
    /// next window's plan ([`Self::close_floor_window`]): that interval holds
    /// its whole GPU tail (the next window's first draft readback waits for
    /// it) without the device sync a synced window end cost every window.
    /// The first batched window this drafter runs at a drafted depth is
    /// observed unpriced (an unusable timing: it spends the probe, records
    /// no cost): the process's first speculative window binds the whole
    /// verify route's kernels (gfx1151 Flash-Next: ~+17-20 ms, ~200 kernel
    /// binds, `HIPFIRE_MODULE_LOAD_AUDIT`) and would otherwise price its
    /// depth out for the whole request; a later first use of a row count
    /// binds five row-specialised symbols (~0-2 ms).
    fn finish_floor_window(
        &mut self,
        bypass: bool,
        route: NativeWindow,
        window: WindowStats,
        k: usize,
        start: Instant,
    ) {
        self.stats.mtp_windows += 1;
        let first_use = route == NativeWindow::Batched && {
            let bit = 1u64 << window.drafts.min(DERIVED_MAX_DRAFTS);
            let first = self.warm_depths & bit == 0;
            self.warm_depths |= bit;
            first
        };
        if bypass || self.follower {
            return;
        }
        // A window run without a plan in between ends where this one began.
        self.close_floor_window(start);
        self.floor_pending = Some(FloorPending {
            start,
            route,
            window,
            k,
            first_use,
        });
    }

    /// Price the pending native window from its start to `end`.
    fn close_floor_window(&mut self, end: Instant) {
        if let Some(pending) = self.floor_pending {
            let us = end.saturating_duration_since(pending.start).as_secs_f64() * 1e6;
            self.price_floor_window(us as f32);
        }
    }

    /// Observe the pending native window for the floor at `us`.
    fn price_floor_window(&mut self, us: f32) {
        let Some(pending) = self.floor_pending.take() else {
            return;
        };
        let priced_us = if pending.first_use { f32::NAN } else { us };
        self.floor.observe_window(
            pending.route,
            pending.window,
            priced_us,
            pending.k,
            &self.agreement,
        );
        if floor_trace_enabled() {
            let event = match pending.route {
                NativeWindow::Interleaved => "native_interleaved",
                NativeWindow::Batched if pending.first_use => "native_batched_first_use",
                NativeWindow::Batched => "native_batched",
            };
            eprintln!(
                "{}",
                self.floor.trace_line(
                    FLOOR_TRACE_PREFIX,
                    &self.agreement,
                    event,
                    us,
                    pending.window.emitted,
                    pending.window.drafts
                )
            );
        }
        self.stats.mtp_retired = self.floor.retired();
    }

    /// [`Self::finish_floor_window`] priced at `us` at once, a batched
    /// window routed at the depth it drafted.
    #[cfg(test)]
    fn floor_observe_native(
        &mut self,
        bypass: bool,
        route: NativeWindow,
        window: &MtpWindow,
        us: f32,
        k: usize,
    ) {
        let stats = WindowStats::of(window, window.drafts_generated);
        self.finish_floor_window(bypass, route, stats, k, Instant::now());
        self.price_floor_window(us);
    }

    /// Whether the floor retired this request to head-free AR.  A caller
    /// that composes an external drafter must check this before every
    /// takeover: after retirement the head lags the target and must not be
    /// extended.
    #[allow(dead_code)]
    pub(crate) fn floor_retired(&self) -> bool {
        self.floor.retired()
    }

    /// Whether the next window must be a timed AR token: the request's first
    /// (the calibration token; the first two on a drafter's first request,
    /// [`Self::floor_observe_ar`]), or one of the AR probes a would-be
    /// retirement takes ([`Qwen4Floor::with_retire_evidence`]).  False for a
    /// developer-forced route (`HIPFIRE_MTP_INCREMENTAL` `0|1`) and with the
    /// floor opted out (`HIPFIRE_MTP_AR_FLOOR=0`), which bypass the floor.  A
    /// caller that composes an external drafter must not take over while this
    /// is true: the calibration windows must be the request's first and the
    /// probes consecutive.
    pub(crate) fn floor_needs_calibration(&self) -> bool {
        self.floor_enabled && self.floor.needs_ar_sample() && !Self::floor_forced()
    }

    /// `floor_retired() || floor_needs_calibration()`: the exact guard an
    /// external takeover must pass before it runs.
    pub(crate) fn floor_takeover_blocked(&self) -> bool {
        self.floor.retired() || self.floor_needs_calibration()
    }

    /// What a promoted singleton hands its lane: the next window's row-0
    /// hidden, the per-depth agreement and whether the floor retired the
    /// request (the head then lags the target, so the lane must stay
    /// head-free). Err when no prefill ran (the drafter holds no pending
    /// hidden). The drafter keeps no in-flight window state between windows;
    /// an active target snapshot in its verify scratch is a ticket of the
    /// bundle's trunk arena, which `Qwen4Bundle::prepare_promotion` refuses
    /// (the scratch cannot be downcast through `&self`). The last committed
    /// window's retained tickets (`RetainedWindow`) do not block a
    /// promotion: they only serve a terminal repair, and the promoted request
    /// leaves this drafter.
    pub fn promotion_parts(&self) -> Result<Qwen4PromotionParts<'_>, String> {
        let pending_hidden = self.pending_hidden.as_ref().ok_or_else(|| {
            "Qwen4 MTP promotion: the drafter holds no request resources".to_string()
        })?;
        Ok(Qwen4PromotionParts {
            pending_hidden,
            agreement: &self.agreement,
            retired: self.floor_enabled && self.floor.retired(),
        })
    }

    /// Developer-forced route that bypasses the floor.
    fn floor_forced() -> bool {
        matches!(
            hipfire_config::developer_var("HIPFIRE_MTP_INCREMENTAL").as_deref(),
            Ok("0" | "1")
        )
    }

    /// Report one finished external window: `drafts` offered, `emitted`
    /// tokens committed, `wall_us` of its whole window (drafting, verify and
    /// commit).  Counts toward the shared probe budget, prices external
    /// windows separately from the native agreement, and feeds the realized
    /// mixture of all windows.  A request stays alive on a losing native
    /// route only while external windows are measured and unblocked and the
    /// decayed realized per-token cost of all its windows is strictly below
    /// AR; it retires at the first window where that stops holding.
    /// Returns `true` when
    /// external takeover must stop for the rest of the request; never with
    /// the floor opted out (`HIPFIRE_MTP_AR_FLOOR=0`).
    pub(crate) fn floor_observe_external_window(
        &mut self,
        emitted: usize,
        wall_us: f32,
        drafts: usize,
    ) -> bool {
        // With the copy lane, the first takeover of a candidate count binds
        // that row count's specialised verify symbols, as the first native
        // batched window of a drafted depth does
        // ([`Self::finish_floor_window`]): observed unpriced, so it spends
        // the probe and records no cost.
        let first_use = if self.copy_lane {
            let bit = 1u64 << drafts.min(NGRAM_DEPTHS);
            let first = self.ngram_warm_depths & bit == 0;
            self.ngram_warm_depths |= bit;
            first
        } else {
            false
        };
        if !self.floor_enabled || self.follower {
            return false;
        }
        let stop = if self.copy_lane {
            let priced_us = if first_use { f32::NAN } else { wall_us };
            self.floor
                .observe_external_copy(emitted, priced_us, drafts, &self.agreement)
        } else {
            self.floor
                .observe_external(emitted, wall_us, drafts, &self.agreement)
        };
        if floor_trace_enabled() {
            eprintln!(
                "{}",
                self.floor.trace_line(
                    FLOOR_TRACE_PREFIX,
                    &self.agreement,
                    "external",
                    wall_us,
                    emitted,
                    drafts
                )
            );
        }
        self.stats.mtp_retired = self.floor.retired();
        stop
    }

    fn pending_hidden(&self) -> Result<&GpuTensor, String> {
        self.pending_hidden
            .as_ref()
            .ok_or_else(|| "Qwen4 MTP pending hidden is not allocated".to_string())
    }

    /// The pending target-hidden row, for the parity harnesses.
    pub(crate) fn pending_hidden_for_parity(&self) -> Result<&GpuTensor, String> {
        self.pending_hidden()
    }

    /// One n-gram takeover window: the pool's `candidates` replace the MTP
    /// head's drafts for this window.
    ///
    /// `[seed, candidates..]` is verified through the same target executor
    /// as a native window: the batched `verify_block` with row-capture
    /// rollback, or under `HIPFIRE_MTP_INCREMENTAL=1` the interleaved
    /// one-row route that stops at the first rejection. Acceptance is the
    /// shared rule of the request: greedy prefix match, or sampled with each
    /// candidate's draft distribution the point mass δ(candidate) (accept with
    /// probability p(candidate), otherwise draw from p without it), or the
    /// naive AR-sampler draw per row.
    ///
    /// The head never proposed these rows, so after the target keeps its
    /// consumed rows (`seed` plus the accepted candidates; an accepted
    /// end-of-turn stays pending) the MTP head appends exactly those rows,
    /// teacher-forced: token `p` paired with the target hidden of position
    /// `p - 1` (the seed with the pending hidden), the pairing of the draft
    /// steps, the prompt fill and `mtp_forced_advance`, through the
    /// batched append where it runs. The pending hidden becomes the last
    /// consumed row's, so the next native window drafts as it would after a
    /// prompt fill ending there. A rejected tail never reaches either owner.
    ///
    /// This takes the window unconditionally; `mtp_step` calls it (through
    /// `run_takeover`) only for a pool hit the head's own first draft
    /// confirmed (with the copy lane: a pool hit the take-length chooser
    /// accepted and, until the copy source is established, the head's own
    /// first draft confirmed).
    #[allow(clippy::too_many_arguments)]
    pub fn mtp_takeover_step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        candidates: &[u32],
        eos: u32,
    ) -> Result<MtpWindow, String> {
        self.takeover_window(gpu, target, position, seed, emitted, candidates, eos, None)
    }

    /// `mtp_takeover_step`, with the confirmation probe's clock when the
    /// takeover was reached through one.
    #[allow(clippy::too_many_arguments)]
    fn takeover_window(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        candidates: &[u32],
        eos: u32,
        probe: Option<TakeoverProbe>,
    ) -> Result<MtpWindow, String> {
        let eos = self.end_of_turn.unwrap_or(eos);
        self.require_supported_request()?;
        // The window's penalty history restarts from `emitted` (idempotent
        // after `mtp_step`'s own `begin_window`).
        if let Some(s) = self.sampled.as_mut() {
            s.begin_window(emitted);
        }
        if candidates.is_empty() {
            return Err("Qwen4 MTP takeover has no candidates".to_string());
        }
        // Before any head or target mutation: a takeover wider than the CPU
        // splice window is refused here, not mid-verify.
        self.bind_spec_rows(target)?;
        check_spec_verify_rows(
            self.spec_rows_cap,
            candidates.len() + 1,
            "Qwen4 n-gram takeover window",
        )?;
        self.ensure_resources(&mut Qwen4Ops::Single(&mut *gpu), target)?;
        {
            let bundle = Self::bundle(target)?;
            let vocab = bundle.config.vocab_size;
            if let Some(&token) = candidates.iter().find(|&&token| token as usize >= vocab) {
                return Err(format!(
                    "Qwen4 MTP takeover candidate {token} is outside the {vocab}-token vocabulary"
                ));
            }
            let target_position = bundle.state.position;
            let mtp_position = bundle.mtp_position().map_err(|error| error.to_string())?;
            if target_position != position || mtp_position != position {
                return Err(format!(
                    "Qwen4 MTP takeover position mismatch: target={target_position}, mtp={mtp_position}, position={position}"
                ));
            }
        }
        let trace =
            hipfire_config::developer_var("HIPFIRE_MTP_TRACE").is_ok_and(|value| value == "1");
        let interleaved = hipfire_config::developer_var("HIPFIRE_MTP_INCREMENTAL")
            .is_ok_and(|value| value == "1");
        #[cfg(feature = "reference-parity")]
        let interleaved = interleaved && self.lab_forced_window.is_none();
        #[cfg(feature = "reference-parity")]
        {
            self.lab_actual_route = Some(if interleaved {
                crate::wide_verify_oracle::DevWindowRoute::Interleaved
            } else {
                crate::wide_verify_oracle::DevWindowRoute::Batched
            });
        }
        let mut sampled = self.sampled.take();
        let window = if interleaved {
            let tickets = self.open_one_row_tickets(
                &mut Qwen4Ops::Single(&mut *gpu),
                target,
                candidates.len() + 1,
            )?;
            let window = self.takeover_interleaved(
                gpu,
                target,
                position,
                seed,
                candidates,
                eos,
                trace,
                sampled.as_mut(),
                probe,
            );
            self.close_window_tickets(
                &mut Qwen4Ops::Single(&mut *gpu),
                target,
                tickets,
                position,
                seed,
                window.is_ok(),
            );
            window
        } else {
            self.takeover_batched(
                gpu,
                target,
                position,
                seed,
                candidates,
                eos,
                trace,
                sampled.as_mut(),
                probe,
            )
        };
        self.sampled = sampled;
        window
    }

    /// With the request armed, ask the pool for a candidate run over the
    /// authoritative history (prompt plus `emitted`, which holds the seed) and
    /// copy it into `ngram_candidates`. A hit is reported only if the yield
    /// gate (`ngram_wins`) passes; a declined hit decays the takeover yield
    /// history toward its prior (`NGRAM_YIELD_RECOVERY`). A hit is not yet a
    /// takeover: `mtp_step` runs it only after the head's own first draft
    /// equals `ngram_candidates[0]`. With the copy lane the decision is
    /// [`Self::copy_hit`]'s.
    fn ngram_hit(&mut self, emitted: &[u32], k: usize) -> bool {
        #[cfg(feature = "reference-parity")]
        if self.lab_copy_enabled == Some(false) {
            return false;
        }
        if self.copy_lane {
            self.copy_hit(emitted, k)
        } else {
            // The floor's calibration token must be the request's first window,
            // and a retired request's head lags the target: no takeover then.
            let blocked = self.floor_takeover_blocked();
            #[cfg(feature = "reference-parity")]
            let blocked = blocked && !self.lab_policy_forced();
            if !self.ngram_active || blocked {
                return false;
            }
            // `k` already includes the remaining emit budget: n-gram extension may
            // exceed the native depth but never this caller cap (a takeover commits
            // seed + candidates to both device owners before the host sees them).
            let limit = self.ngram_config.map_or(k, |config| config.n_max.min(k));
            let Some(ctx) = self.ngram.as_mut() else {
                return false;
            };
            let Some(candidates) = ctx.propose(emitted, limit) else {
                return false;
            };
            self.ngram_candidates.clear();
            self.ngram_candidates.extend_from_slice(candidates);
            // The pool's full offer, kept before the incumbent's take
            // truncates it: the derived chooser prices (and may widen to)
            // prefixes of it, and bins the offer, not the take.
            if matches!(self.derived, Some(DerivedPhase::Bound(_))) {
                self.ngram_proposal.clear();
                self.ngram_proposal
                    .extend_from_slice(&self.ngram_candidates);
            }
            #[cfg(feature = "reference-parity")]
            if self.lab_policy_forced() {
                self.ngram_proposal.clone_from(&self.ngram_candidates);
                if let Some(d) = self.lab_ngram_width() {
                    if d > self.ngram_candidates.len() || d > k {
                        return false;
                    }
                    self.ngram_candidates.truncate(d);
                    return d > 0;
                }
            }
            let native_depth = k.min(self.incumbent_max_k());
            let depth = self.ngram_takeover_depth(self.ngram_candidates.len(), native_depth);
            if self.ngram_config.is_some_and(|config| depth < config.n_min) {
                return false;
            }
            self.ngram_candidates.truncate(depth);
            if depth <= native_depth && !self.ngram_wins(self.ngram_candidates.len(), k) {
                self.ngram_yield.0 *= NGRAM_YIELD_RECOVERY;
                self.ngram_yield.1 *= NGRAM_YIELD_RECOVERY;
                return false;
            }
            true
        }
    }

    /// The copy lane's pool decision: ask the pool for a proposal over the
    /// authoritative history (prompt plus `emitted`, which holds the seed),
    /// up to `n_max` tokens: the whole proposal is kept for the shadow
    /// observation of [`Self::observe_ngram_shadow`], but at most
    /// `min(take cap, k)` candidates are taken. The take length is the
    /// `d` in `n_min..=min(proposal, cap)` maximizing the expected emitted
    /// tokens per unit [`Self::takeover_cost`], with the expected tokens
    /// `1 + Σ_i Π_{j<=i} a_j` of this request's own n-gram acceptance `a_j`
    /// per depth (shorter on ties); it is taken when that rate is at least
    /// the native window's ([`Self::native_rate`]), or, on a request the
    /// floor retired, at least AR's 1.0 token per AR step: retirement says
    /// native drafting loses to AR, not that the copy source does. The take
    /// is copied into `ngram_candidates`. A declined proposal still leaves
    /// the shadow state set. Whether the head's row-0 draft must still
    /// confirm the take is decided here too
    /// ([`NGRAM_UNCONFIRMED_MIN_TAKE`]). Only the floor's AR calibration
    /// (and its probes) bars the pool.
    fn copy_hit(&mut self, emitted: &[u32], k: usize) -> bool {
        self.ngram_decision = None;
        // The floor's calibration token must be the request's first window.
        let blocked = self.floor_needs_calibration();
        #[cfg(feature = "reference-parity")]
        let blocked = blocked && !self.lab_policy_forced();
        if !self.ngram_active || blocked {
            return false;
        }
        let Some(config) = self.ngram_config else {
            return false;
        };
        let Some(ctx) = self.ngram_copy.as_mut() else {
            return false;
        };
        // The extra length past the take is only observed, never taken.
        let Some(proposal) = ctx.propose(emitted, config.n_max) else {
            return false;
        };
        self.ngram_proposal.clear();
        self.ngram_proposal.extend_from_slice(proposal);
        self.ngram_proposed_at = Some(emitted.len());
        let available = self.ngram_proposal.len();
        // `k` already includes the remaining emit budget: a takeover commits
        // seed + candidates to both device owners before the host sees them.
        let cap = self.ngram_take_cap().min(k).min(self.ngram_agreement.len());
        let (a1_sum, a1_weight) = self.ngram_agreement[0];
        let a1 = a1_sum / a1_weight.max(f32::MIN_POSITIVE);
        let mut decision = NgramDecision {
            available,
            take: 0,
            confirm: true,
            rate: 0.0,
            native_rate: 0.0,
            a1,
        };
        #[cfg(feature = "reference-parity")]
        if let Some(d) = self.lab_ngram_width() {
            // Bypass the take-length chooser only. These are the exact
            // source's real tokens, never fabricated longer offers.
            if d > available || d > k {
                self.ngram_decision = Some(decision);
                return false;
            }
            decision.take = d;
            decision.confirm = !(d >= NGRAM_UNCONFIRMED_MIN_TAKE
                && a1 >= NGRAM_UNCONFIRMED_MIN_ACCEPT
                && a1_weight >= NGRAM_UNCONFIRMED_MIN_EVIDENCE
                && self.ngram_last_first_accepted);
            self.ngram_candidates.clear();
            self.ngram_candidates
                .extend_from_slice(&self.ngram_proposal[..d]);
            self.ngram_decision = Some(decision);
            return d > 0;
        }
        let hi = available.min(cap);
        let lo = config.n_min.max(1);
        if hi >= lo {
            let native_rate = if self.floor.retired() {
                1.0
            } else {
                self.native_rate(k.min(self.incumbent_max_k()))
            };
            let (mut prefix, mut expected) = (1.0f32, 1.0f32);
            let mut best = (0usize, 0.0f32);
            for d in 1..=hi {
                let (accepted, weight) = self.ngram_agreement[d - 1];
                prefix *= accepted / weight.max(f32::MIN_POSITIVE);
                expected += prefix;
                if d < lo {
                    continue;
                }
                let rate = expected / self.takeover_cost(d);
                if best.0 == 0 || rate > best.1 {
                    best = (d, rate);
                }
            }
            decision.rate = best.1;
            decision.native_rate = native_rate;
            // A retired request's alternative is one AR row, so a takeover
            // rejected at its first candidate wastes the whole verify (a
            // native window that agreed with the same wrong candidate would
            // have failed with it): once retired, take only while the last
            // shadow observation accepted its first candidate. A copy span
            // that resumes re-arms after one AR token.
            let retired_cold = self.floor.retired() && !self.ngram_last_first_accepted;
            if best.1 >= native_rate && !retired_cold {
                decision.take = best.0;
                decision.confirm = !(best.0 >= NGRAM_UNCONFIRMED_MIN_TAKE
                    && a1 >= NGRAM_UNCONFIRMED_MIN_ACCEPT
                    && a1_weight >= NGRAM_UNCONFIRMED_MIN_EVIDENCE
                    && self.ngram_last_first_accepted);
                self.ngram_candidates.clear();
                self.ngram_candidates
                    .extend_from_slice(&self.ngram_proposal[..best.0]);
            }
        }
        self.ngram_decision = Some(decision);
        decision.take > 0
    }

    /// Interleaved-route confirmation of a pool hit: run the head's row-0
    /// draft step on a scratch copy of the head (snapshot, one
    /// `mtp_forward_token`, restore, and the draft policy restored too) and
    /// compare its argmax with `ngram_candidates[0]`. The head ends exactly as
    /// it began either way.
    fn probe_first_draft(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
    ) -> Result<bool, String> {
        let Some(&candidate) = self.ngram_candidates.first() else {
            return Ok(false);
        };
        let pending = self
            .pending_hidden
            .as_ref()
            .ok_or_else(|| "Qwen4 MTP pending hidden is not allocated".to_string())?;
        let bundle = Self::bundle(target)?;
        let policy = bundle
            .mtp_draft_request_state()
            .map_err(|error| error.to_string())?;
        let ticket = bundle
            .mtp_snapshot(gpu)
            .map_err(|error| error.to_string())?;
        let drafted = bundle.mtp_forward_token(gpu, seed, Some(pending), position, true);
        let restored = bundle.mtp_restore(gpu, ticket);
        let policy_restored = bundle.set_mtp_draft_request_state(policy);
        let first = drafted.map_err(|error| error.to_string())?;
        restored.map_err(|error| error.to_string())?;
        policy_restored.map_err(|error| error.to_string())?;
        Ok(first == candidate)
    }

    /// The takeover window over `ngram_candidates` (a chosen take), with the
    /// `ngram_mod_*` counters and the floor updated
    /// ([`Self::finish_takeover`]). Takeover results never feed the native
    /// agreement table. Without the copy lane the takeover's own yield and
    /// the pool's low-acceptance clear are updated too; the copy lane never
    /// clears the pool on low acceptance: the take-length chooser backs off
    /// on the measured acceptance, where the pool's five-low-window clear
    /// would drop the request's prompt for good.
    #[allow(clippy::too_many_arguments)]
    fn run_takeover(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        eos: u32,
        probe: Option<TakeoverProbe>,
    ) -> Result<MtpWindow, String> {
        self.derived_note_takeover();
        // The floor prices the takeover over its whole window: from the
        // native window whose row-0 step confirmed it, to the drained head
        // fill after commit.
        let wall_start = probe
            .as_ref()
            .map_or_else(Instant::now, |probe| probe.start);
        let candidates = std::mem::take(&mut self.ngram_candidates);
        let window = self.takeover_window(
            gpu,
            target,
            position,
            seed,
            emitted,
            &candidates,
            eos,
            probe,
        );
        self.ngram_candidates = candidates;
        let window = window?;
        let wall_us = if self.floor_enabled && !Self::floor_forced() {
            synced_elapsed_us(gpu, wall_start)?
        } else {
            f32::NAN
        };
        self.finish_takeover(&window, wall_us);
        Ok(window)
    }

    /// The host-side bookkeeping of a finished takeover: the request's
    /// `ngram_mod_*` counters and the floor's external-window price
    /// (`wall_us`, drafting through the drained head fill), which may stop
    /// takeovers for the rest of the request. Without the copy lane it also
    /// updates the takeover yield (`ngram_yield`) and feeds the pool's
    /// low-acceptance clear (`observe_result`).
    fn finish_takeover(&mut self, window: &MtpWindow, wall_us: f32) {
        if !self.copy_lane {
            let decay = self.agreement_decay;
            self.ngram_yield.0 = self.ngram_yield.0 * decay + window.committed.len() as f32;
            self.ngram_yield.1 = self.ngram_yield.1 * decay + 1.0;
        }
        self.stats.ngram_mod_windows += 1;
        self.stats.ngram_mod_drafts += window.drafts_generated;
        self.stats.ngram_mod_accepted += window.accepted;
        if !self.copy_lane {
            if let Some(ctx) = self.ngram.as_mut() {
                ctx.observe_result(window.drafts_generated, window.accepted);
            }
        }
        if self.floor_enabled
            && !Self::floor_forced()
            && self.floor_observe_external_window(
                window.committed.len(),
                wall_us,
                window.drafts_generated,
            )
        {
            self.ngram_active = false;
        }
    }

    /// Batched takeover: one `[seed, candidates..]` verify, rollback to the
    /// consumed rows, teacher-forced head append. Two-owner transaction as in
    /// the native batched window: any failure restores target and head.
    #[allow(clippy::too_many_arguments)]
    fn takeover_batched(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        candidates: &[u32],
        eos: u32,
        trace: bool,
        mut sampled: Option<&mut SampledVerify>,
        probe: Option<TakeoverProbe>,
    ) -> Result<MtpWindow, String> {
        let n = candidates.len();
        let mut block = std::mem::take(&mut self.takeover_block);
        block.clear();
        block.push(seed);
        block.extend_from_slice(candidates);
        let mut timers = MtpPhaseTimers::new().silenced(self.follower);
        let window_start = probe.map_or_else(Instant::now, |probe| probe.start);
        let mut accepted_drafts = 0usize;
        let mut emitted_len = 0usize;
        let mut snapshot = match Self::bundle(target)
            .and_then(|bundle| bundle.mtp_snapshot(gpu).map_err(|error| error.to_string()))
        {
            Ok(ticket) => Some(ticket),
            Err(error) => {
                self.takeover_block = block;
                return Err(error);
            }
        };
        let result = (|| -> Result<MtpWindow, String> {
            // The verify's PLE rows are SSD-resident: start reading them now.
            Self::bundle(target)?.warm_ple_rows(&block);
            timers.mark(gpu, "verify");
            let bundle = Self::bundle(target)?;
            let scratch = self
                .scratch
                .as_mut()
                .ok_or_else(|| "Qwen4 MTP takeover verify scratch is not allocated".to_string())?;
            let skip_picks = sampled
                .as_deref()
                .is_some_and(|s| s.device_skip_picks(n, trace || timers.enabled()));
            let mut target_picks = if skip_picks {
                // Resident acceptance reads this window's logits on the
                // device: no argmax pick is downloaded or consulted.
                bundle.verify_block_resident(
                    &mut Qwen4Ops::Single(&mut *gpu),
                    &block,
                    position,
                    scratch.as_mut(),
                )?;
                Vec::new()
            } else {
                let target_picks =
                    bundle.verify_block(gpu, &block, position, scratch.as_mut(), None)?;
                if target_picks.len() < block.len() {
                    return Err(format!(
                        "Qwen4 MTP takeover verifier returned {} rows for {n} candidates",
                        target_picks.len()
                    ));
                }
                target_picks
            };
            // The request's shared acceptance: each candidate row's `p` is
            // penalized with the window history plus the candidates before it,
            // and a sampled candidate's `q` is the point mass on it. `accept`
            // times the target-row work (penalty prepass, row downloads, host
            // distributions) apart from the verify forward.
            timers.mark(gpu, "accept");
            let acceptance = match sampled.as_deref_mut() {
                Some(s) if !s.draws_drafts() => {
                    s.accept_naive(gpu, bundle, candidates, eos, &mut target_picks)?
                }
                Some(s) => s.accept_leviathan(gpu, bundle, candidates, true, eos)?,
                None => accept_native_greedy(candidates, &target_picks, Some(eos))?,
            };
            accepted_drafts = acceptance.accepted;
            let target_accept_len = target_commit_accept_len(&acceptance);
            let consumed = target_accept_len + 1;
            let target_scratch = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .ok_or("Qwen4 MTP takeover target scratch type changed")?;
            let target_snapshot = target_scratch
                .target_snapshot
                .ok_or("Qwen4 MTP takeover target snapshot disappeared")?
                .local()?;
            if consumed < block.len() && bundle.state.row_capture_rows() >= block.len() {
                timers.mark(gpu, "target_rollback");
                bundle
                    .rollback_verify_rows_retain(gpu, target_snapshot, consumed, &block)
                    .map_err(|error| error.to_string())?;
            } else if consumed < block.len() {
                timers.mark(gpu, "target_replay");
                bundle
                    .restore_retain(gpu, target_snapshot)
                    .map_err(|error| error.to_string())?;
                bundle
                    .spec_forward_rows(gpu, &block[..consumed], true)
                    .map_err(|error| error.to_string())?;
            }
            // The kept rows' captured hiddens are spec hidden rows
            // `0..consumed` (the verify's, or the replay's).
            timers.mark(gpu, "head_fill");
            {
                let row_hidden = self
                    .row_hidden
                    .as_ref()
                    .ok_or_else(|| "Qwen4 MTP row hidden is not allocated".to_string())?;
                let pending = self
                    .pending_hidden
                    .as_ref()
                    .ok_or_else(|| "Qwen4 MTP pending hidden is not allocated".to_string())?;
                fill_head_rows(
                    bundle,
                    gpu,
                    self.append_scratch.as_mut(),
                    row_hidden,
                    pending,
                    &block[..consumed],
                    position,
                )?;
            }
            timers.mark(gpu, "commit");
            let committed_end = position + consumed;
            let mtp_end = bundle.mtp_position().map_err(|error| error.to_string())?;
            if bundle.state.position != committed_end || mtp_end != committed_end {
                return Err(format!(
                    "Qwen4 MTP takeover ended at target={} mtp={mtp_end}, expected {committed_end}",
                    bundle.state.position
                ));
            }
            let pending_hidden = self
                .pending_hidden
                .as_ref()
                .ok_or_else(|| "Qwen4 MTP pending hidden is not allocated".to_string())?;
            // The window's pre-window pending hidden, kept for a terminal
            // repair before the copy below moves `pending_hidden` on.
            let retained_pending = self
                .retained_pending
                .as_ref()
                .ok_or_else(|| "Qwen4 MTP retained pending hidden is not allocated".to_string())?;
            gpu.copy_d2d(pending_hidden, retained_pending, pending_hidden.byte_size())
                .map_err(|error| format!("Qwen4 MTP retained pending hidden copy: {error}"))?;
            bundle
                .copy_spec_hidden_row_to(gpu, target_accept_len, pending_hidden)
                .map_err(|error| error.to_string())?;
            if trace && !self.follower {
                let gdn_resets = gdn_residual_trace_field(&bundle.state);
                let verify_rows = n + 1;
                let row_limit = bundle.row_exact_verify_rows(gpu);
                eprintln!(
                    "QWEN4_MTP_TRACE {{\"event\":\"window\",\"source\":\"ngram\",\"verify_route\":\"batched\",\"accept_route\":\"{}\",\"position\":{position},\"k\":{n},\"accepted\":{},\"verify_rows\":{verify_rows},\"row_limit\":{row_limit},\"rows\":[{}],\"committed\":{:?},\"baseline\":true{gdn_resets}}}",
                    accept_route_label(sampled.as_deref()),
                    acceptance.accepted,
                    trace_rows(candidates, &target_picks),
                    acceptance.committed
                );
            }
            let mtp_ticket = snapshot
                .as_ref()
                .copied()
                .expect("MTP snapshot remains active until transaction commit");
            bundle
                .retain_window(target_snapshot, mtp_ticket)
                .map_err(|error| error.to_string())?;
            target_scratch.target_snapshot = None;
            snapshot = None;
            self.retained = Some(RetainedWindow {
                span: WindowSpan {
                    start: position,
                    end: committed_end,
                    seed,
                },
                target: TargetTicket::Local(target_snapshot),
                head: HeadTicket::Local(mtp_ticket),
            });
            emitted_len = acceptance.committed.len();
            Ok(MtpWindow {
                committed: acceptance.committed,
                accepted: acceptance.accepted,
                drafts_generated: n,
            })
        })();
        self.takeover_block = block;
        if timers.enabled() {
            let window_us = synced_wall_us(gpu, window_start);
            let probe_field = probe.map_or_else(String::new, |probe| {
                format!("\"probe_us\":{:.1},", probe.probe_us)
            });
            let accept_route = accept_route_label(sampled.as_deref());
            let fields = format!(
                "\"source\":\"ngram\",\"verify_route\":\"batched\",\"accept_route\":\"{accept_route}\",{probe_field}\"position\":{position},\"k\":{n},\"accepted\":{accepted_drafts},\"emitted\":{emitted_len},\"window_us\":{window_us:.1},\"t_end\":{}",
                unix_micros()
            );
            timers.finish(gpu, "QWEN4_MTP_PHASE", &fields);
        }
        if let Err(error) = &result {
            if let Err(rollback) = self.rollback_failed_window(
                &mut Qwen4Ops::Single(&mut *gpu),
                target,
                snapshot.take().map(HeadTicket::Local),
            ) {
                return Err(format!("{error}; rollback failed: {rollback}"));
            }
        }
        result
    }

    /// Restore both owners after a failed batched window: the target from
    /// the verify scratch's active ticket, the head from `mtp_ticket`. Errors
    /// name every restore that failed. Under a mesh the aggregate ticket
    /// restores every band and the head in one transaction (consuming it).
    fn rollback_failed_window(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        target: &mut dyn SpecTarget,
        mtp_ticket: Option<HeadTicket>,
    ) -> Result<(), String> {
        let mut rollback_errors = Vec::new();
        let target_ticket = match self.scratch.as_mut() {
            Some(scratch) => match scratch.as_any_mut().downcast_mut::<Qwen4SpecScratch>() {
                Some(scratch) => scratch.target_snapshot.take(),
                None => {
                    rollback_errors.push("target rollback scratch type changed".to_string());
                    None
                }
            },
            None => None,
        };
        match (target_ticket, mtp_ticket) {
            (Some(TargetTicket::Mesh(window)), _) | (None, Some(HeadTicket::Mesh(window))) => {
                // A window the finalize already restored and killed (or a
                // reset invalidated) has nothing left to restore.
                if let Err(rollback) = ops.gpus().and_then(|gpus| {
                    Self::bundle(target).and_then(|bundle| {
                        if !bundle.pp_window_live(&window) {
                            return Ok(());
                        }
                        bundle
                            .pp_window_restore(gpus, &window, false)
                            .map_err(|restore| restore.to_string())
                    })
                }) {
                    rollback_errors.push(format!("window rollback failed: {rollback}"));
                }
            }
            (target_ticket, mtp_ticket) => {
                if let Some(ticket) = target_ticket {
                    if let Err(rollback) = ticket.local().and_then(|ticket| {
                        Self::bundle(target).and_then(|bundle| {
                            bundle
                                .restore(ops.gpu()?, ticket)
                                .map_err(|restore| restore.to_string())
                        })
                    }) {
                        rollback_errors.push(format!("target rollback failed: {rollback}"));
                    }
                }
                if let Some(ticket) = mtp_ticket {
                    if let Err(rollback) = ticket.local().and_then(|ticket| {
                        Self::bundle(target).and_then(|bundle| {
                            bundle
                                .mtp_restore(ops.gpu()?, ticket)
                                .map_err(|restore| restore.to_string())
                        })
                    }) {
                        rollback_errors.push(format!("MTP rollback failed: {rollback}"));
                    }
                }
            }
        }
        if rollback_errors.is_empty() {
            Ok(())
        } else {
            Err(rollback_errors.join("; "))
        }
    }

    /// Interleaved takeover (`HIPFIRE_MTP_INCREMENTAL=1`): the native
    /// one-row target route with the candidates in place of head drafts.
    /// Each row commits as it is produced and the head appends it
    /// teacher-forced; the window stops at the first rejection.
    #[allow(clippy::too_many_arguments)]
    fn takeover_interleaved(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        candidates: &[u32],
        eos: u32,
        trace: bool,
        mut sampled: Option<&mut SampledVerify>,
        probe: Option<TakeoverProbe>,
    ) -> Result<MtpWindow, String> {
        let n = candidates.len();
        let mut timers = MtpPhaseTimers::new().silenced(self.follower);
        let window_start = probe.map_or_else(Instant::now, |probe| probe.start);
        let mut committed: Vec<u32> = Vec::with_capacity(n + 1);
        let mut picks: Vec<u32> = Vec::with_capacity(n + 1);
        // Row `r`'s penalty history is the window base plus the candidates
        // accepted before it.
        if let Some(s) = sampled.as_deref_mut() {
            s.history.rewind_drafts();
        }
        let mut token = seed;
        let mut accepted = 0usize;
        let mut row = 0usize;
        let device_rows = sampled.as_deref().is_some_and(|s| s.device_ready(n));
        let skip_picks = sampled
            .as_deref()
            .is_some_and(|s| s.device_skip_picks(n, trace || timers.enabled()));
        loop {
            let token_position = position + row;
            timers.mark(gpu, "target_row");
            let mut pick = if skip_picks {
                Self::bundle(target)?
                    .spec_capture_token_resident(gpu, token)
                    .map_err(|error| error.to_string())?;
                0
            } else {
                Self::bundle(target)?
                    .spec_capture_token(gpu, token)
                    .map_err(|error| error.to_string())?
            };
            match sampled.as_deref_mut() {
                Some(s) if !s.draws_drafts() => {
                    pick = s.naive_draw(gpu, Self::bundle(target)?, 0, row)?
                }
                Some(s) if device_rows => s.device_load_target(gpu, Self::bundle(target)?, 0)?,
                Some(s) => s.load_target(gpu, Self::bundle(target)?, 0, row)?,
                None => {}
            }
            if !skip_picks {
                picks.push(pick);
            }
            timers.mark(gpu, "head_fill");
            let row_hidden = self
                .row_hidden
                .as_ref()
                .ok_or_else(|| "Qwen4 MTP row hidden is not allocated".to_string())?;
            {
                let bundle = Self::bundle(target)?;
                bundle
                    .copy_spec_hidden_row_to(gpu, 0, row_hidden)
                    .map_err(|error| error.to_string())?;
                // (token p, hidden p - 1): `pending` carries the previous
                // row's hidden (the window's pending hidden for row 0).
                let pending = self
                    .pending_hidden
                    .as_ref()
                    .ok_or_else(|| "Qwen4 MTP pending hidden is not allocated".to_string())?;
                bundle
                    .mtp_append_token(gpu, token, Some(pending), token_position)
                    .map_err(|error| error.to_string())?;
            }
            if row == n {
                committed.push(match sampled.as_deref_mut() {
                    Some(s) if device_rows => s.device_bonus(gpu)?,
                    Some(s) if s.draws_drafts() => s.host_bonus(),
                    _ => pick,
                });
                break;
            }
            let candidate = candidates[row];
            let verdict = match sampled.as_deref_mut() {
                Some(s) if s.draws_drafts() => {
                    if s.points.is_empty() {
                        s.points.push(SparseDist::default());
                    }
                    s.points[0].set_point_mass(candidate);
                    if device_rows {
                        s.device_step_verdict(gpu, candidate, true)?
                    } else {
                        s.host_step_verdict(candidate, true)
                    }
                }
                _ if candidate == pick => DraftVerdict::Accept,
                _ => DraftVerdict::Reject(pick),
            };
            match verdict {
                DraftVerdict::Accept => {
                    accepted += 1;
                    committed.push(candidate);
                    if let Some(s) = sampled.as_deref_mut() {
                        s.history.push_draft(candidate);
                    }
                    if candidate == eos {
                        break;
                    }
                    // The next row pairs with this row's own target hidden.
                    {
                        let pending = self.pending_hidden()?;
                        gpu.copy_d2d(row_hidden, pending, pending.byte_size())
                            .map_err(|error| {
                                format!("Qwen4 MTP takeover pending hidden copy: {error}")
                            })?;
                    }
                    token = candidate;
                    row += 1;
                }
                DraftVerdict::Reject(replacement) => {
                    committed.push(replacement);
                    break;
                }
            }
        }
        // The next window's row-0 hidden is the last consumed row's.
        {
            let row_hidden = self.row_hidden()?;
            let pending = self.pending_hidden()?;
            gpu.copy_d2d(row_hidden, pending, pending.byte_size())
                .map_err(|error| format!("Qwen4 MTP takeover pending hidden copy: {error}"))?;
        }
        let committed_end = position + row + 1;
        let bundle = Self::bundle(target)?;
        let mtp_end = bundle.mtp_position().map_err(|error| error.to_string())?;
        if bundle.state.position != committed_end || mtp_end != committed_end {
            return Err(format!(
                "Qwen4 MTP interleaved takeover ended at target={} mtp={mtp_end}, expected {committed_end}",
                bundle.state.position
            ));
        }
        if trace && !self.follower {
            let gdn_resets = gdn_residual_trace_field(&bundle.state);
            let row_limit = bundle.row_exact_verify_rows(gpu);
            eprintln!(
                "QWEN4_MTP_TRACE {{\"event\":\"window\",\"source\":\"ngram\",\"verify_route\":\"interleaved\",\"accept_route\":\"{}\",\"position\":{position},\"k\":{n},\"accepted\":{accepted},\"verify_rows\":1,\"row_limit\":{row_limit},\"rows\":[{}],\"committed\":{committed:?}{gdn_resets}}}",
                accept_route_label(sampled.as_deref()),
                trace_rows(candidates, &picks)
            );
        }
        if timers.enabled() {
            let window_us = synced_wall_us(gpu, window_start);
            let probe_field = probe.map_or_else(String::new, |probe| {
                format!("\"probe_us\":{:.1},", probe.probe_us)
            });
            let accept_route = accept_route_label(sampled.as_deref());
            let fields = format!(
                "\"source\":\"ngram\",\"verify_route\":\"interleaved\",\"accept_route\":\"{accept_route}\",{probe_field}\"position\":{position},\"k\":{n},\"accepted\":{accepted},\"emitted\":{},\"window_us\":{window_us:.1},\"t_end\":{}",
                committed.len(),
                unix_micros()
            );
            timers.finish(gpu, "QWEN4_MTP_PHASE", &fields);
        }
        Ok(MtpWindow {
            committed,
            accepted,
            drafts_generated: n,
        })
    }
}

/// Close the open phase of `timers` (when given) and open `label`.
fn mark_phase(timers: &mut Option<&mut MtpPhaseTimers>, gpu: &Gpu, label: &'static str) {
    if let Some(timers) = timers.as_deref_mut() {
        timers.mark(gpu, label);
    }
}

/// The body of `mtp_prefill`'s chunk loop for the chunk `chunk` at `pos`:
/// the trunk tile with wide-hidden capture (`spec_prefill_rows` when
/// `needs_pick`, else `spec_prefill_rows_silent`), then the head append of
/// every row (`append_scratch` given and batched fill or the DET prompt fill
/// on: sub-chunks of `scratch.rows()` rows, a lone row per-row except under
/// DET, which takes it through the same rows kernels; otherwise per-row),
/// with token `p` paired with the target hidden of `p - 1` (`pending` for the
/// chunk's first row, which prompt position 0 takes from its own row),
/// leaving `pending` on the chunk's last
/// row. Returns the trunk pick when `needs_pick`. The boundary, session and
/// pick bookkeeping stay with the caller; `abort` is polled before every
/// sub-chunk (or row) and a hit resets the target and breaks. Every rows pass is
/// [`MtpFillContext::PromptFill`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn mtp_prefill_chunk(
    bundle: &mut Qwen4Bundle,
    ops: &mut Qwen4Ops<'_>,
    chunk: &[u32],
    pos: usize,
    needs_pick: bool,
    pending: &GpuTensor,
    mut append_scratch: Option<&mut MtpAppendScratch>,
    reuse_trunk_ids: bool,
    abort: &dyn Fn() -> bool,
) -> Result<ControlFlow<(), Option<u32>>, String> {
    let max_chunk = bundle.ops_chunk_rows(ops).unwrap_or(1).max(1);
    let paired = !ops.is_mesh() && bundle.npu_trunk_planned() && chunk.len() > max_chunk;
    // The pair's banks and its interior snapshot row: the target forward runs
    // them through the pair executor, and the head appends follow the same
    // banks and cut.
    let shape = if paired {
        Some(
            bundle
                .npu_trunk_pair_shape(pos, chunk.len(), max_chunk)
                .filter(|shape| shape.total() == chunk.len())
                .ok_or_else(|| {
                    "Qwen4 npu-trunk: scheduled pair group has no matching plan".to_string()
                })?,
        )
    } else {
        None
    };
    let pick = bundle.ops_prefill_rows(ops, chunk, needs_pick)?;
    if let Some(shape) = &shape {
        // The target captured both banks together, but head append boundaries,
        // prompt-id uploads and pending rows remain those of the sequential run.
        for piece in shape.pieces() {
            let tokens = &chunk[piece.clone()];
            if mtp_append_prefill_chunk(
                bundle,
                ops,
                tokens,
                pos + piece.start,
                piece.start,
                pending,
                append_scratch.as_deref_mut(),
                false,
                abort,
            )?
            .is_break()
            {
                return Ok(ControlFlow::Break(()));
            }
            if piece.end < chunk.len() {
                // The pair's interior snapshot row is a prompt boundary like
                // any other: `pending` is the target hidden of its last row
                // (`piece.end - 1`) after this piece's head append, and the
                // snapshot must carry it or every restore refuses it.
                if bundle.session_next_boundary() == Some(pos + piece.end) {
                    bundle
                        .session_save_boundary_hidden(ops.gpu()?, pending)
                        .map_err(|e| e.to_string())?;
                }
                bundle
                    .publish_pair_session_boundary(ops.gpu()?, pos + piece.end)
                    .map_err(|e| e.to_string())?;
            }
        }
    } else if mtp_append_prefill_chunk(
        bundle,
        ops,
        chunk,
        pos,
        0,
        pending,
        append_scratch,
        reuse_trunk_ids,
        abort,
    )?
    .is_break()
    {
        return Ok(ControlFlow::Break(()));
    }
    Ok(ControlFlow::Continue(pick))
}

#[allow(clippy::too_many_arguments)]
fn mtp_append_prefill_chunk(
    bundle: &mut Qwen4Bundle,
    ops: &mut Qwen4Ops<'_>,
    chunk: &[u32],
    pos: usize,
    hidden_row0: usize,
    pending: &GpuTensor,
    append_scratch: Option<&mut MtpAppendScratch>,
    reuse_trunk_ids: bool,
    abort: &dyn Fn() -> bool,
) -> Result<ControlFlow<()>, String> {
    // The head's batched rows route: the incumbent batched fill
    // (`HIPFIRE_QWEN4_MTP_BATCHED_FILL`, on by default), or the DET prompt
    // fill (exact gfx1151, `HIPFIRE_QWEN4_MTP_APPEND_DET` on unless `0`), which uses
    // the scratch even with batched fill off. Without a scratch (the head
    // does not support the rows route) the chunk is per-row either way.
    let det = mtp_append_det_enabled(ops.gpu()?);
    let batched = mtp_batched_fill_enabled();
    if let Some(scratch) = append_scratch.filter(|_| batched || det) {
        // The chunk's ids upload once, here: the target forward's pick
        // readback has already drained the stream (a silent chunk
        // leaves the copy waiting for that forward, once per chunk
        // instead of once per sub-chunk). Each sub-chunk then embeds
        // from its rows of that upload; the bytes are the chunk's own
        // tokens as i32, exactly what `append_rows` would upload
        // itself. With the trunk-id reuse lever on, the head keeps
        // that lever's own ids (and its own upload when they differ).
        let chunk_ids_uploaded =
            prefill_uploads_chunk_ids(reuse_trunk_ids, chunk.len(), scratch.prompt_id_rows());
        if chunk_ids_uploaded {
            scratch
                .upload_prompt_ids(ops.gpu()?, chunk)
                .map_err(|error| error.to_string())?;
        }
        // Map the head's QSA context through the chunk's last row once,
        // here (the device is idle after the target forward), instead
        // of growing inside each sub-chunk's Append: a VMM grow
        // zero-fills with a blocking memset. Same pages, same zero
        // fill, all before any Append writes; each sub-chunk's own
        // mapping check then finds the coverage in place.
        bundle
            .mtp_map_through(ops.gpu()?, pos + chunk.len() - 1)
            .map_err(|error| error.to_string())?;
        // One Append pass per sub-chunk of `prompt_fill_sub_rows` rows
        // (`scratch.rows()` batched; one row at a time with batched fill
        // off under DET). Token `p` pairs with the target hidden of
        // `p - 1`: chunk row 0 with `pending` (the previous chunk's last
        // row; prompt position 0 has no predecessor and takes its own
        // hidden), row `i` with spec hidden row `i - 1`.
        let sub_rows = prompt_fill_sub_rows(batched, scratch.rows());
        if pos == 0 {
            bundle
                .copy_spec_hidden_row_to(ops.gpu()?, hidden_row0, pending)
                .map_err(|error| error.to_string())?;
        }
        if prompt_fill_uses_rows(det, 1) {
            // DET: the first row takes the same rows kernels as every other
            // prompt row, so its bytes do not depend on the chunk boundary.
            let prompt_ids = if chunk_ids_uploaded {
                Some(
                    scratch
                        .prompt_ids_view(0, 1)
                        .map_err(|error| error.to_string())?,
                )
            } else {
                None
            };
            bundle
                .mtp_append_rows_hidden(
                    ops.gpu()?,
                    MtpFillContext::PromptFill,
                    scratch,
                    &chunk[..1],
                    pending,
                    prefill_trunk_ids_row(reuse_trunk_ids, 0),
                    prompt_ids.as_ref(),
                    pos,
                )
                .map_err(|error| error.to_string())?;
        } else {
            bundle
                .mtp_append_token(ops.gpu()?, chunk[0], Some(pending), pos)
                .map_err(|error| error.to_string())?;
        }
        let mut off = 1;
        while off < chunk.len() {
            if abort() {
                bundle.ops_reset(ops)?;
                return Ok(ControlFlow::Break(()));
            }
            let n = (chunk.len() - off).min(sub_rows);
            let position = pos
                .checked_add(off)
                .ok_or_else(|| "Qwen4 native MTP prefill position overflow".to_string())?;
            if !prompt_fill_uses_rows(det, n) {
                // The incumbent multirow projections need two rows or
                // more; a lone row is exactly the per-row Append (its
                // token upload stays the per-row path's own, blocking
                // one). Under DET a lone row takes the rows route below
                // instead, through the same kernels as every other row.
                bundle
                    .copy_spec_hidden_row_to(ops.gpu()?, hidden_row0 + off - 1, pending)
                    .map_err(|error| error.to_string())?;
                bundle
                    .mtp_append_token(ops.gpu()?, chunk[off], Some(pending), position)
                    .map_err(|error| error.to_string())?;
            } else {
                let prompt_ids = if chunk_ids_uploaded {
                    Some(
                        scratch
                            .prompt_ids_view(off, n)
                            .map_err(|error| error.to_string())?,
                    )
                } else {
                    None
                };
                bundle
                    .mtp_append_rows(
                        ops.gpu()?,
                        MtpFillContext::PromptFill,
                        scratch,
                        &chunk[off..off + n],
                        hidden_row0 + off - 1,
                        prefill_trunk_ids_row(reuse_trunk_ids, off),
                        prompt_ids.as_ref(),
                        position,
                    )
                    .map_err(|error| error.to_string())?;
            }
            off += n;
        }
        // The per-row loop leaves `pending` on the chunk's last row: the
        // next draft step's row-0 hidden.
        bundle
            .copy_spec_hidden_row_to(ops.gpu()?, hidden_row0 + chunk.len() - 1, pending)
            .map_err(|error| error.to_string())?;
    } else {
        for (index, &token) in chunk.iter().enumerate() {
            if abort() {
                bundle.ops_reset(ops)?;
                return Ok(ControlFlow::Break(()));
            }
            let position = pos
                .checked_add(index)
                .ok_or_else(|| "Qwen4 native MTP prefill position overflow".to_string())?;
            // (token p, hidden p - 1): `pending` still holds the hidden of
            // `p - 1`; prompt position 0 has no predecessor and takes its own.
            if position == 0 {
                bundle
                    .copy_spec_hidden_row_to(ops.gpu()?, hidden_row0 + index, pending)
                    .map_err(|error| error.to_string())?;
            }
            bundle
                .mtp_append_token(ops.gpu()?, token, Some(pending), position)
                .map_err(|error| error.to_string())?;
            bundle
                .copy_spec_hidden_row_to(ops.gpu()?, hidden_row0 + index, pending)
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(ControlFlow::Continue(()))
}

/// Teacher-forced head rows for `tokens` at `start..`, whose target hiddens
/// are spec hidden rows `0..tokens.len()` while `pending` holds the hidden of
/// `start - 1` (the head's position - 1): token `i` pairs with the hidden of
/// position `start + i - 1`, so token 0 with `pending` and token `i` with spec
/// row `i - 1`. Token 0 is one Append; the rest take one batched decode fill
/// when `fill` takes them, else one Append per row through `row_hidden`.
/// `pending` is left as it was: the caller moves it to the last row's hidden.
#[allow(clippy::too_many_arguments)]
fn fill_head_rows(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    fill: Option<&mut MtpAppendScratch>,
    row_hidden: &GpuTensor,
    pending: &GpuTensor,
    tokens: &[u32],
    start: usize,
) -> Result<(), String> {
    let Some((&first, rest)) = tokens.split_first() else {
        return Ok(());
    };
    bundle
        .mtp_append_token(gpu, first, Some(pending), start)
        .map_err(|error| error.to_string())?;
    if rest.is_empty() {
        return Ok(());
    }
    match fill
        .filter(|fill| rest.len() >= 2 && rest.len() <= fill.rows() && mtp_batched_fill_enabled())
    {
        Some(fill) => bundle
            .mtp_append_rows(
                gpu,
                MtpFillContext::DecodeFill,
                fill,
                rest,
                0,
                None,
                None,
                start + 1,
            )
            .map_err(|error| error.to_string()),
        None => {
            for (row, &token) in rest.iter().enumerate() {
                bundle
                    .copy_spec_hidden_row_to(gpu, row, row_hidden)
                    .map_err(|error| error.to_string())?;
                bundle
                    .mtp_append_token(gpu, token, Some(row_hidden), start + 1 + row)
                    .map_err(|error| error.to_string())?;
            }
            Ok(())
        }
    }
}

/// What [`finish_native_window`] needs from a batched window's verify.
pub(crate) struct NativeFinish<'a> {
    /// Seed position: the window's first row.
    pub position: usize,
    /// `[seed, drafts..]`, the verify's rows.
    pub block: &'a [u32],
    pub acceptance: GreedyAccept,
    /// The target ticket taken before the verify.
    pub target_snapshot: TargetTicket,
    /// The head ticket taken before the window's draft steps.
    pub mtp_ticket: HeadTicket,
    /// Destination of the next window's row-0 hidden; on entry the
    /// pre-window pending hidden (the target hidden of `position - 1`), and
    /// the teacher-forced rebuild's scratch row in between.
    pub pending_hidden: &'a GpuTensor,
    /// Row `i` = block row `i`'s wide hidden (F32, `hc_count * hidden`).
    pub verify_hidden: &'a GpuTensor,
    /// The verify's PLE convolution input rows from block row 0 (a local
    /// target only: every mesh band keeps its own captured inputs).
    pub ple_normed: Option<&'a GpuTensor>,
    /// Keep both pre-window tickets rollback-able after the commit
    /// ([`Qwen4Bundle::retain_window`]) instead of consuming them: the
    /// singleton drafter's windows, which a terminal stop may cut short. The
    /// caller records the tickets and releases them.
    pub retain: bool,
}

/// A batched window's tail, from the target's accepted length through the
/// ticket commit: ring rollback (or the replay when the ring does not cover
/// the verify), head commit (truncate to the seed's row, then the kept drafts' rows rebuilt teacher-forced), the
/// position checks, the pending hidden copy and the commit of both tickets.
/// `mtp_step` and the lane executor both end their windows here; the caller
/// clears its own scratch ticket and head ticket after `Ok`.
pub(crate) fn finish_native_window(
    picks: &mut Qwen4Bundle,
    ops: &mut Qwen4Ops<'_>,
    f: NativeFinish<'_>,
    mut timers: Option<&mut MtpPhaseTimers>,
) -> Result<MtpWindow, String> {
    let NativeFinish {
        position,
        block,
        acceptance,
        target_snapshot,
        mtp_ticket,
        pending_hidden,
        verify_hidden,
        ple_normed,
        retain,
    } = f;
    let drafts = block
        .get(1..)
        .ok_or_else(|| "Qwen4 native MTP window has no seed row".to_string())?;
    let k = drafts.len();
    let target_accept_len = target_commit_accept_len(&acceptance);
    let full_accept = target_accept_len == k;

    // Keep both pre-window tickets active until every replay and hidden
    // copy succeeds. A retained restore lets the outer rollback repair
    // both owners if either side's GPU work fails.
    // Under a mesh a full target restore also restores the head to the
    // window's start (one aggregate restore): the head then replays below.
    let mut head_at_start = false;
    if !full_accept && picks.ops_ring_eligible(target_snapshot, block.len()) {
        mark_phase(&mut timers, ops.gpu()?, "target_rollback");
        picks.ops_rollback_rows(
            ops,
            target_snapshot,
            target_accept_len + 1,
            block,
            ple_normed,
        )?;
    } else if !full_accept {
        mark_phase(&mut timers, ops.gpu()?, "target_replay");
        match target_snapshot {
            TargetTicket::Local(ticket) => picks
                .restore_retain(ops.gpu()?, ticket)
                .map_err(|error| error.to_string())?,
            TargetTicket::Mesh(window) => {
                picks
                    .pp_window_restore(ops.gpus()?, &window, true)
                    .map_err(|error| error.to_string())?;
                head_at_start = true;
            }
        }
        picks.ops_spec_rows(ops, &block[..target_accept_len + 1], true)?;
    }
    mark_phase(&mut timers, ops.gpu()?, "mtp_commit");
    // The draft chain's row 0 (the seed with the pending hidden of
    // `position - 1`) is the only head row of the window kept as written: its
    // later rows were conditioned on the head's own previous-step hidden.
    // Keep it (or write it again when the head is back at the window's start)
    // and rebuild the kept drafts' rows teacher-forced, (token `block[i + 1]`,
    // target verify hidden row `i`), the pairing of the prompt fill, the
    // takeover fill and the AR catch-up. Head rows feed drafts only; target
    // state is untouched.
    match head_finish(k, head_at_start) {
        HeadFinish::SeedFromStart { restore } => {
            // No draft step ran (`k == 0`) or a mesh target restore already
            // rewound the head: the head appends the seed with the pending
            // hidden of `position - 1`, on a fresh selection.
            if restore {
                picks.ops_head_restore_retain(ops, mtp_ticket)?;
            }
            picks
                .mtp_append_token(ops.gpu()?, block[0], Some(pending_hidden), position)
                .map_err(|error| error.to_string())?;
        }
        HeadFinish::KeepSeedRow => {
            picks.ops_head_truncate_retain(mtp_ticket, 1)?;
        }
    }
    for row in 0..target_accept_len {
        picks
            .copy_hidden_row(ops.gpu()?, verify_hidden, row, pending_hidden)
            .map_err(|error| error.to_string())?;
        picks
            .mtp_append_token(
                ops.gpu()?,
                block[row + 1],
                Some(pending_hidden),
                position + 1 + row,
            )
            .map_err(|error| error.to_string())?;
    }
    mark_phase(&mut timers, ops.gpu()?, "commit");
    let committed_end = position
        .checked_add(target_accept_len + 1)
        .ok_or_else(|| "Qwen4 MTP commit position overflow".to_string())?;
    if picks.state.position != committed_end
        || picks.mtp_position().map_err(|error| error.to_string())? != committed_end
    {
        return Err(format!(
            "Qwen4 native MTP transaction ended at target={} mtp={}, expected {committed_end}",
            picks.state.position,
            picks.mtp_position().map_err(|error| error.to_string())?
        ));
    }
    picks
        .copy_hidden_row(ops.gpu()?, verify_hidden, target_accept_len, pending_hidden)
        .map_err(|error| error.to_string())?;

    let window = MtpWindow {
        committed: acceptance.committed,
        accepted: acceptance.accepted,
        drafts_generated: drafts.len(),
    };
    // All fallible GPU operations are complete. Validate every owner's
    // ticket (every band and the head, under a mesh) before invalidating
    // any, then perform the no-copy commit boundary; a retaining window keeps
    // the arena entries instead, so a terminal stop inside it can still roll
    // every owner back.
    picks.ops_finish_window(ops, target_snapshot, mtp_ticket, retain)?;
    Ok(window)
}

/// How a batched window's head reaches the committed position. Every route
/// ends with the kept drafts' rows rebuilt teacher-forced after the seed's
/// row ([`finish_native_window`]); the routes differ only in how the seed's
/// row stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HeadFinish {
    /// The draft chain's row 0 is the seed's row as written: truncate the
    /// head to it.
    KeepSeedRow,
    /// The head is back (or goes back: `restore`; a mesh target restore has
    /// already done it) at the window's start: append the seed's row.
    SeedFromStart { restore: bool },
}

/// The head branch of [`finish_native_window`]. A window that drafted nothing
/// (`k == 0`) has no row 0 to keep, and `head_at_start` (the target restore
/// of a mesh window, the replay route, ring not eligible, already put the
/// head back at the window's start) leaves none either.
fn head_finish(k: usize, head_at_start: bool) -> HeadFinish {
    if k == 0 || head_at_start {
        HeadFinish::SeedFromStart {
            restore: !head_at_start,
        }
    } else {
        HeadFinish::KeepSeedRow
    }
}

impl Qwen4MtpDrafter {
    /// Single-row and additional-row verify milliseconds of the loaded arch
    /// (`None`: the legacy chooser prices the n-gram source).
    fn ngram_row_cost_for(gpu: &Gpu) -> Option<(f32, f32)> {
        if gpu.arch_caps.is_gfx1151() {
            Some((30.5, 6.9))
        } else if gpu.arch_caps.is_gfx1201() {
            Some((30.3, 14.8))
        } else {
            None
        }
    }

    /// Bind the loaded arch's n-gram row prices and singleton decay (the
    /// request's floor takes the decay too).
    fn bind_arch(&mut self, gpu: &Gpu) {
        self.ngram_row_cost = Self::ngram_row_cost_for(gpu);
        self.agreement_decay = singleton_agreement_decay(&gpu.arch);
        self.floor.set_decay(self.agreement_decay);
    }

    /// Every timing- or host-state-dependent decision of the next window,
    /// host only: the forced-route environment, the floor's retirement and
    /// calibration, the n-gram pool's hit (and its yield gate; with the copy
    /// lane its take length), the floor's
    /// route and depth, and the trace flag. Rank 0 only: it mutates the floor,
    /// the n-gram history and the request stats exactly as the decision chain
    /// of a single-card window did, in the same order. `emitted` holds the
    /// pending seed; `k` is the caller's draft budget (checked against the
    /// proposal capacity here).
    pub(crate) fn plan_window(&mut self, emitted: &[u32], k: usize) -> Result<WindowPlan, String> {
        let incremental = hipfire_config::developer_var("HIPFIRE_MTP_INCREMENTAL").ok();
        let trace =
            hipfire_config::developer_var("HIPFIRE_MTP_TRACE").is_ok_and(|value| value == "1");
        self.plan_window_with(emitted, k, incremental.as_deref(), trace)
    }

    /// Observe only newly emitted tokens, never a close in the prompt or a
    /// close already seen by an earlier plan. No sampler or hidden state moves.
    fn observe_think_close(&mut self, emitted: &[u32], floor_active: bool) {
        let start = self
            .emitted_seen
            .unwrap_or_else(|| emitted.len().saturating_sub(1))
            .min(emitted.len());
        self.emitted_seen = Some(emitted.len());
        if floor_active
            && self.floor.retired()
            && self
                .think_close
                .is_some_and(|close| emitted[start..].contains(&close))
        {
            self.floor.rearm();
            self.agreement.fill(AGREEMENT_PRIOR);
            self.stats.mtp_retired = false;
        }
    }

    /// [`Self::plan_window`] with the environment already read.
    fn plan_window_with(
        &mut self,
        emitted: &[u32],
        k: usize,
        incremental: Option<&str>,
        trace: bool,
    ) -> Result<WindowPlan, String> {
        // A plan that never ran leaves no consulted window behind.
        self.derived_drop_pending();
        // The previous native window ends here (its price includes its GPU
        // tail); this plan reads its cost.
        let plan_start = Instant::now();
        self.close_floor_window(plan_start);
        // The window that followed the last proposal has committed: compare.
        if self.copy_lane {
            self.observe_ngram_shadow(emitted);
        }
        self.require_supported_request()?;
        if k > self.proposal_capacity() {
            return Err(format!(
                "Qwen4 native MTP draft budget {k} exceeds the proposal capacity {}",
                self.proposal_capacity()
            ));
        }
        // A developer-forced route is a diagnostic override: the floor
        // neither calibrates nor retires, so the pinned route runs as before.
        // `HIPFIRE_MTP_AR_FLOOR=0` (`floor_enabled == false`) bypasses the
        // floor the same way and restores the pre-floor chooser.
        let forced = matches!(incremental, Some("0" | "1"));
        let floor_active = self.floor_enabled && !forced;
        self.observe_think_close(emitted, floor_active);
        #[cfg(feature = "reference-parity")]
        if self.lab_policy_forced() {
            return self.lab_plan_window(emitted, k, trace, plan_start);
        }
        // Otherwise the window drafts natively, within the head's own K.
        let native_k = k.min(self.incumbent_max_k());
        let caller_k = k;
        let ar = |route| WindowPlan {
            route,
            k: native_k,
            trace,
            floor_active,
            ngram_hit: false,
            ngram_confirm: true,
            ngram_candidates: Vec::new(),
        };
        if floor_active && self.floor.retired() {
            if !self.copy_lane {
                return Ok(ar(WindowRoute::ArRetired));
            }
            // The copy source may still take the window: the head is caught
            // up before a takeover ([`Self::run_window`]).
            let ngram_hit = self.ngram_hit(emitted, k);
            self.trace_ngram_plan(trace, emitted, plan_start, true);
            return Ok(WindowPlan {
                route: WindowRoute::ArRetired,
                k: native_k,
                trace,
                floor_active,
                ngram_hit,
                ngram_confirm: !ngram_hit || self.ngram_decision.is_none_or(|d| d.confirm),
                ngram_candidates: Vec::new(),
            });
        }
        if floor_active && self.floor.needs_ar_sample() {
            return Ok(ar(WindowRoute::ArCalibrate));
        }
        // A pool hit may take this window, but only if the head's own first
        // draft equals the hit's first candidate: confirmed on the route the
        // window runs (batched: inside the drafting loop, reusing the row-0
        // step; interleaved: a standalone probe). `mtp_windows` counts only
        // the windows that end native, and a confirmation miss leaves the
        // takeover yield history untouched. With the copy lane a pool
        // proposal takes the window at a take length the chooser prices at
        // least as well as native, and until the copy source is established
        // the head's own first draft must still equal the first candidate.
        let ngram_hit = self.ngram_hit(emitted, k);
        if self.copy_lane {
            self.trace_ngram_plan(trace, emitted, plan_start, false);
        }
        let k = native_k;
        // Route: a (k+1)-row batched verify costs about 2.5 single-row
        // forwards, so it pays only while drafts keep being accepted; the
        // interleaved verify costs one forward per emitted token and wastes no
        // draft. `HIPFIRE_MTP_INCREMENTAL=0|1` forces a route.
        let depth = match incremental {
            Some("0") => k,
            Some("1") => 0,
            _ if !self.floor_enabled => self.floor_off_depth(k),
            _ => match {
                let provisional = self.batched_depth(k);
                let route = self.floor.route(&self.agreement, k, provisional);
                self.stats.mtp_retired = self.floor.retired();
                route
            } {
                Action::ArHeadFree => return Ok(ar(WindowRoute::ArRouted)),
                Action::ArSynced => return Ok(ar(WindowRoute::ArCalibrate)),
                Action::Interleaved => 0,
                Action::Native(depth) => depth,
            },
        };
        let mut plan = WindowPlan {
            route: WindowRoute::Native { depth },
            k,
            trace,
            floor_active,
            ngram_hit,
            ngram_confirm: !ngram_hit || self.ngram_decision.is_none_or(|d| d.confirm),
            ngram_candidates: Vec::new(),
        };
        // A native incumbent's depth is not an economic ceiling, including
        // an unforced interleaved incumbent. Explicit/floor AR routes above
        // remain authoritative.
        if floor_active {
            self.derived_plan(&mut plan, caller_k);
        }
        Ok(plan)
    }

    /// The `request` trace event (`HIPFIRE_MTP_TRACE=1`, rank 0): what the
    /// request configured, the context its `window` events are read in.
    /// Non-finite floats print as `null`.
    fn trace_request(&self, cfg: &SpecRequestConfig) {
        let enabled =
            hipfire_config::developer_var("HIPFIRE_MTP_TRACE").is_ok_and(|value| value == "1");
        if !enabled || self.follower {
            return;
        }
        let ngram = match (self.ngram_active, self.ngram_config) {
            (true, Some(config)) => format!(
                ",\"n_match\":{},\"n_min\":{},\"n_max\":{}",
                config.n_match, config.n_min, config.n_max
            ),
            _ => String::new(),
        };
        eprintln!(
            "QWEN4_MTP_TRACE {{\"event\":\"request\",\"k\":{},\"copy_lane\":{},\"floor\":{},\"temp\":{},\"repeat_penalty\":{},\"repeat_window\":{},\"presence_penalty\":{},\"frequency_penalty\":{},\"ngram\":{}{ngram},\"sampled\":{}}}",
            self.max_k,
            self.copy_lane,
            self.floor_enabled,
            JsonF32(cfg.temp),
            JsonF32(cfg.repeat_penalty),
            cfg.repeat_window,
            JsonF32(cfg.presence_penalty),
            JsonF32(cfg.frequency_penalty),
            self.ngram_active,
            self.sampled.is_some()
        );
    }

    /// The `ngram_plan` trace line (`HIPFIRE_MTP_TRACE=1`, rank 0) of the
    /// decision [`Self::ngram_hit`] just made, when the pool made a proposal.
    fn trace_ngram_plan(&self, trace: bool, emitted: &[u32], plan_start: Instant, retired: bool) {
        if !trace || self.follower {
            return;
        }
        if let Some(decision) = self.ngram_decision {
            let meta = if self.copy_lane {
                self.ngram_copy
                    .as_ref()
                    .and_then(MtpCopyContext::last_proposal)
            } else {
                self.ngram.as_ref().and_then(MtpNgramContext::last_proposal)
            };
            let source = meta.map_or_else(String::new, |meta| {
                let source_pos = meta
                    .source_pos
                    .map_or_else(String::new, |pos| format!(",\"source_pos\":{pos}"));
                format!(",\"key_len\":{}{source_pos}", meta.key_len)
            });
            eprintln!(
                "QWEN4_MTP_TRACE {{\"event\":\"ngram_plan\",\"emitted\":{},\"available\":{},\"take\":{},\"confirm\":{},\"rate\":{:.3},\"native_rate\":{:.3},\"a1\":{:.3},\"plan_us\":{:.3},\"retired\":{retired}{source}}}",
                emitted.len(),
                decision.available,
                decision.take,
                decision.confirm,
                decision.rate,
                decision.native_rate,
                decision.a1,
                elapsed_us(plan_start)
            );
        }
    }

    /// Copy this window's pool candidates into `plan` for the replica ranks
    /// (rank 0 reads its own copy, so a single-card window never allocates).
    pub(crate) fn attach_candidates(&self, plan: &mut WindowPlan) {
        if plan.ngram_hit {
            plan.ngram_candidates.clone_from(&self.ngram_candidates);
        }
    }

    /// The parameters an EP peer's replica is constructed from.
    pub(crate) fn replica_params(&self) -> ReplicaParams {
        ReplicaParams {
            max_k: self.max_k,
            ctx_capacity: self.ctx_capacity,
            end_of_turn: self.end_of_turn,
            ngram_config: self.ngram_config,
            copy_lane: self.copy_lane,
            sampled_enabled: self.sampled_enabled,
        }
    }

    /// An EP peer's drafter: rank 0's construction parameters, as a follower.
    pub(crate) fn new_replica(params: ReplicaParams) -> Self {
        let mut replica = Self::new(params.max_k, params.ctx_capacity, params.end_of_turn)
            .with_ngram(params.ngram_config)
            .with_copy_lane(params.copy_lane);
        replica.sampled_enabled = params.sampled_enabled;
        replica.follower = true;
        replica
    }

    /// Whether the n-gram source is armed for the request. Rank 0 may disarm
    /// it mid-request (the floor stopped external takeovers); a replica
    /// mirrors the value at the start of every fan-out so its proposal
    /// capacity never differs from rank 0's.
    pub(crate) fn ngram_active(&self) -> bool {
        self.ngram_active
    }

    pub(crate) fn mirror_ngram_active(&mut self, active: bool) {
        self.ngram_active = active;
    }

    /// The target-less calls the peers' drafters still owe, oldest first.
    pub(crate) fn pending_replay(&self) -> &[ReplayOp] {
        &self.replay
    }

    /// The peers replayed [`Self::pending_replay`]: drop it.
    pub(crate) fn clear_replay(&mut self) {
        self.replay.clear();
    }

    /// Apply `ops` (rank 0's target-less calls, in order) to this replica.
    pub(crate) fn replay(&mut self, gpu: &mut Gpu, ops: &[ReplayOp]) -> Result<(), String> {
        for op in ops {
            match *op {
                ReplayOp::Configure(cfg) => MtpDrafter::configure_request(self, cfg),
                ReplayOp::Reset => MtpDrafter::mtp_reset(self, gpu)?,
            }
        }
        Ok(())
    }

    /// Queue a target-less call for the peers, compacted: a configure
    /// supersedes every earlier one (it rebuilds every request-local state
    /// and does not touch what a reset clears, so the two commute) and
    /// consecutive resets collapse, leaving at most
    /// `[Reset, Configure, Reset]`. A replica queues nothing.
    fn push_replay(&mut self, op: ReplayOp) {
        if self.follower {
            return;
        }
        match op {
            ReplayOp::Configure(_) => {
                self.replay
                    .retain(|queued| !matches!(queued, ReplayOp::Configure(_)));
                self.replay
                    .dedup_by(|a, b| matches!((a, b), (ReplayOp::Reset, ReplayOp::Reset)));
            }
            ReplayOp::Reset => {
                if matches!(self.replay.last(), Some(ReplayOp::Reset)) {
                    return;
                }
            }
        }
        self.replay.push(op);
    }

    /// `HIPFIRE_MTP_SAMPLED_MODE=naive` draws from the process-wide AR sampler,
    /// which every rank of an EP mesh would advance: refused under EP, before
    /// any fan-out (an error inside one poisons the mesh).
    pub(crate) fn refuse_naive_under_ep(&self) -> Result<(), String> {
        match &self.sampled {
            Some(sampled) if !sampled.draws_drafts() => Err(
                "Qwen4 native MTP: HIPFIRE_MTP_SAMPLED_MODE=naive is unsupported under qwen4 EP \
                 (its draws come from the process-wide AR sampler every rank shares); use the \
                 default leviathan mode or set topology.ep = 1 (HIPFIRE_QWEN4_EP=0)"
                    .to_string(),
            ),
            _ => Ok(()),
        }
    }

    /// Run the window `plan` chose (see [`Self::plan_window`]) on this rank:
    /// everything of a native window but the timing- and host-state-dependent
    /// decisions, which only the plan carries. On rank 0 this is the old
    /// single-card window verbatim; a replica rank runs the identical device
    /// program on its own device (taking the n-gram candidates from the plan)
    /// and feeds none of its own clocks to the floor.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_window(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        eos: u32,
        plan: &WindowPlan,
    ) -> Result<MtpWindow, String> {
        self.run_window_ops(
            &mut Qwen4Ops::Single(gpu),
            target,
            position,
            seed,
            emitted,
            eos,
            plan,
        )
    }

    /// [`Self::run_window`] over an operation context: the one window body of
    /// a single card and of a PP mesh. With `feature = "lab"` and the drafter
    /// armed (`lab_observe_windows`) the window is also observed; nothing of
    /// that touches the window's result.
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(feature = "lab"), allow(clippy::let_and_return))]
    pub(crate) fn run_window_ops(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        eos: u32,
        plan: &WindowPlan,
    ) -> Result<MtpWindow, String> {
        #[cfg(feature = "lab")]
        let observed = self.lab_window_begin(target, position, seed, plan);
        let prior_ngram = self.stats.ngram_mod_windows;
        let result = self.run_window_ops_body(ops, target, position, seed, emitted, eos, plan);
        if plan.trace && !self.follower {
            if let Ok(window) = &result {
                let ngram = self.stats.ngram_mod_windows > prior_ngram;
                if ngram || matches!(plan.route, WindowRoute::Native { .. }) {
                    let source = if ngram { "ngram" } else { "mtp" };
                    let mut interleaved = if ngram {
                        hipfire_config::developer_var("HIPFIRE_MTP_INCREMENTAL").as_deref()
                            == Ok("1")
                    } else {
                        matches!(plan.route, WindowRoute::Native { depth: 0 })
                    };
                    #[cfg(feature = "reference-parity")]
                    if ngram && self.lab_forced_window.is_some() {
                        interleaved = false;
                    }
                    let verified_rows = if interleaved {
                        1
                    } else {
                        window.drafts_generated + 1
                    };
                    if let Ok(bundle) = Self::bundle(target) {
                        let keep = bundle.state.position.saturating_sub(position);
                        let repair_route = if interleaved || keep >= verified_rows {
                            "none"
                        } else if self
                            .retained
                            .is_some_and(|r| bundle.ops_ring_eligible(r.target, verified_rows))
                        {
                            "native_capture"
                        } else {
                            "replay"
                        };
                        eprintln!(
                            "QWEN4_MTP_TRACE {{\"event\":\"window_commit\",\"source\":\"{source}\",\"position\":{position},\"verified_rows\":{verified_rows},\"keep\":{keep},\"repair_route\":\"{repair_route}\",\"retired\":{}}}",
                            self.floor.retired()
                        );
                    }
                }
            }
        }
        // Derived K: the verified-window boundary, where the `window` trace
        // receipt is written; failed windows are dropped without an update.
        if self.derived_pending_open() {
            let eos = self.end_of_turn.unwrap_or(eos);
            self.derived_observe(plan, eos, &result);
        }
        #[cfg(feature = "lab")]
        if observed {
            self.lab_window_end(target, &result);
        }
        result
    }

    /// The window body of [`Self::run_window_ops`].
    #[allow(clippy::too_many_arguments)]
    fn run_window_ops_body(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        eos: u32,
        plan: &WindowPlan,
    ) -> Result<MtpWindow, String> {
        // A plan wider than the CPU splice window is refused before any
        // state moves (an armed load's drafter plans within it already).
        self.bind_spec_rows(target)?;
        self.check_plan_rows(plan)?;
        let eos = self.end_of_turn.unwrap_or(eos);
        // The window's penalty history is rebuilt from the tokens emitted so
        // far (the pending seed included); rejected drafts of earlier windows
        // never enter it.
        if let Some(s) = self.sampled.as_mut() {
            s.begin_window(emitted);
        }
        if self.follower && plan.ngram_hit {
            self.ngram_candidates.clear();
            self.ngram_candidates
                .extend_from_slice(&plan.ngram_candidates);
        }
        let (hit, k, trace, floor_active) = (plan.ngram_hit, plan.k, plan.trace, plan.floor_active);
        // The window before this one stays repairable only until now.
        self.release_retained(target);
        if plan.route == WindowRoute::ArRetired {
            if !hit {
                #[cfg(feature = "reference-parity")]
                {
                    self.lab_actual_route =
                        Some(crate::wide_verify_oracle::DevWindowRoute::TargetOnly);
                }
                return self.mtp_ar_step(ops, target, position, seed, false);
            }
            // The n-gram takeover is a one-card route: refused under a mesh
            // before any resource or head work.
            ops.single("n-gram takeover")?;
            // A retired request's head lags the target: catch it up (the
            // owed rows appended as a forced advance does) before any head
            // work, then confirm the first candidate or take the window.
            self.ensure_resources(ops, target)?;
            self.catch_up_head(ops, target)?;
            if plan.ngram_confirm {
                let matched =
                    self.probe_first_draft(ops.single("n-gram takeover")?, target, position, seed)?;
                #[cfg(feature = "reference-parity")]
                {
                    self.lab_confirmation = Some(matched);
                }
                if trace && self.copy_lane && !self.follower {
                    eprintln!(
                        "QWEN4_MTP_TRACE {{\"event\":\"ngram_confirm\",\"position\":{position},\"match\":{matched}}}"
                    );
                }
                if !matched {
                    // The head sits at `position`, so the AR step's owed row
                    // starts a fresh lag there.
                    return self.mtp_ar_step(ops, target, position, seed, false);
                }
            }
            return self.run_takeover(
                ops.single("n-gram takeover")?,
                target,
                position,
                seed,
                emitted,
                eos,
                None,
            );
        }
        self.ensure_resources(ops, target)?;
        // Re-arming may leave retired AR rows owed on every EP rank. Catch
        // them up before calibration or any draft overwrites spec hidden;
        // pending stays the head's predecessor hidden until catch-up uses it.
        self.catch_up_head(ops, target)?;
        #[cfg(feature = "reference-parity")]
        if !matches!(plan.route, WindowRoute::Native { .. }) {
            self.lab_actual_route = Some(crate::wide_verify_oracle::DevWindowRoute::TargetOnly);
        }
        let depth = match plan.route {
            WindowRoute::Native { depth } => depth,
            WindowRoute::ArCalibrate => return self.mtp_ar_step(ops, target, position, seed, true),
            WindowRoute::ArRetired | WindowRoute::ArRouted => {
                return self.mtp_ar_step(ops, target, position, seed, false)
            }
        };
        // An established copy source skips the head's row-0 confirmation:
        // the takeover runs before any route work, with no head step.
        if hit && !plan.ngram_confirm {
            return self.run_takeover(
                ops.single("n-gram takeover")?,
                target,
                position,
                seed,
                emitted,
                eos,
                None,
            );
        }
        if depth == 0 {
            #[cfg(feature = "reference-parity")]
            {
                self.lab_actual_route =
                    Some(crate::wide_verify_oracle::DevWindowRoute::Interleaved);
            }
            let window_start = Instant::now();
            if hit {
                let matched =
                    self.probe_first_draft(ops.single("n-gram takeover")?, target, position, seed)?;
                #[cfg(feature = "reference-parity")]
                {
                    self.lab_confirmation = Some(matched);
                }
                if trace && self.copy_lane && !self.follower {
                    eprintln!(
                        "QWEN4_MTP_TRACE {{\"event\":\"ngram_confirm\",\"position\":{position},\"match\":{matched}}}"
                    );
                }
                if matched {
                    let probe = takeover_probe(ops.single("n-gram takeover")?, window_start);
                    return self.run_takeover(
                        ops.single("n-gram takeover")?,
                        target,
                        position,
                        seed,
                        emitted,
                        eos,
                        Some(probe),
                    );
                }
            }
            let tickets = self.open_one_row_tickets(ops, target, k + 1)?;
            let mut sampled = self.sampled.take();
            let window = self.mtp_step_incremental(
                ops,
                target,
                position,
                seed,
                k,
                eos,
                trace,
                sampled.as_mut(),
            );
            self.sampled = sampled;
            self.close_window_tickets(ops, target, tickets, position, seed, window.is_ok());
            let window = window?;
            self.observe_agreement(&window);
            let stats = WindowStats::of(&window, 0);
            self.finish_floor_window(
                !floor_active,
                NativeWindow::Interleaved,
                stats,
                k,
                window_start,
            );
            return Ok(window);
        }
        // The interleaved route issues no head draft steps of its own (the
        // head reads the target's rows teacher-forced): no margin exists.
        // History picks the route and the most drafts; the drafts' own
        // margins stop early.
        let k_max = depth;
        let window_start = Instant::now();
        {
            let bundle = Self::bundle(target)?;
            let target_position = bundle.state.position;
            let mtp_position = bundle.mtp_position().map_err(|error| error.to_string())?;
            if target_position != position || mtp_position != position {
                return Err(format!(
                    "Qwen4 MTP step position mismatch: target={}, mtp={}, position={position}",
                    target_position, mtp_position
                ));
            }
        }
        // The draft policy is outside the head snapshot: saved here so a
        // confirmed takeover can undo the abandoned row-0 step.
        let draft_state = Self::bundle(target)?
            .mtp_draft_request_state()
            .map_err(|error| error.to_string())?;
        // The head's pre-window ticket. Under a mesh one aggregate window
        // (armed first, then every band and the head snapshotted) is taken
        // here, before the drafts, and the verify only runs the block: the
        // draft steps touch only the head.
        let mut snapshot = if ops.is_mesh() {
            let window = Self::bundle(target)?
                .pp_window_open(ops.gpus()?, k_max + 1, true)
                .map_err(|error| error.to_string())?;
            let Some(scratch) = self
                .scratch
                .as_mut()
                .and_then(|scratch| scratch.as_any_mut().downcast_mut::<Qwen4SpecScratch>())
            else {
                Self::bundle(target)?
                    .ops_release_window(TargetTicket::Mesh(window), HeadTicket::Mesh(window));
                return Err("Qwen4 native MTP verify scratch is not allocated".to_string());
            };
            scratch.target_snapshot = Some(TargetTicket::Mesh(window));
            Some(HeadTicket::Mesh(window))
        } else {
            let bundle = Self::bundle(target)?;
            Some(HeadTicket::Local(
                bundle
                    .mtp_snapshot(ops.gpu()?)
                    .map_err(|error| error.to_string())?,
            ))
        };
        let mut timers = MtpPhaseTimers::new().silenced(self.follower);
        let mut accepted_drafts = 0usize;
        let mut drafts_verified = 0usize;
        let mut emitted_len = 0usize;
        let mut sampled = self.sampled.take();
        let result = (|| -> Result<NativeOutcome, String> {
            if self.derived_margins_active() {
                self.derived_margins_reset();
            }
            let mut drafts = Vec::with_capacity(k_max);
            let mut margins: Vec<f32> = Vec::with_capacity(k_max);
            let mut input = seed;
            // The verify's PLE rows are SSD-resident: start reading each
            // token's rows as soon as it is known, while the drafts run.
            let mut known = Vec::with_capacity(k_max + 1);
            known.push(seed);
            Self::bundle(target)?.warm_ple_rows(&known);
            // Probability the whole draft prefix is accepted, from each
            // draft's exact logit margin over its runner-up.
            let mut prefix = 1.0f32;
            // The derived objective drives this window's depth: its re-solve
            // after each row replaces the incumbent's fixed-worth margin prune.
            let derived_loop = self.derived_loop_active();
            #[cfg(feature = "reference-parity")]
            let derived_loop = derived_loop || self.lab_policy_forced();
            // Every proposal starts with a fresh QSA selection; only later
            // draft rows within this window reuse it.
            for index in 0..k_max {
                let hidden = if index == 0 {
                    Some(self.pending_hidden()?)
                } else {
                    None
                };
                let token_position = position
                    .checked_add(index)
                    .ok_or_else(|| "Qwen4 MTP step position overflow".to_string())?;
                let argmax = Self::bundle(target)?
                    .mtp_forward_token(ops.gpu()?, input, hidden, token_position, index == 0)
                    .map_err(|error| error.to_string())?;
                // Head-confirmed takeover: the head's own first draft (the
                // row-0 argmax, before any sampled draw) equals the pool's
                // first candidate. Nothing is kept or verified yet, so the
                // window is abandoned here; the target was never touched.
                if index == 0 && hit {
                    let matched = self.ngram_candidates.first() == Some(&argmax);
                    #[cfg(feature = "reference-parity")]
                    {
                        self.lab_confirmation = Some(matched);
                    }
                    if trace && self.copy_lane && !self.follower {
                        eprintln!(
                            "QWEN4_MTP_TRACE {{\"event\":\"ngram_confirm\",\"position\":{position},\"match\":{matched}}}"
                        );
                    }
                    if matched {
                        return Ok(NativeOutcome::Takeover);
                    }
                }
                input = argmax;
                if let Some(s) = sampled.as_mut().filter(|s| s.draws_drafts()) {
                    input = s.sample_draft(ops.gpu()?, Self::bundle(target)?, index, index)?;
                }
                let margin = Self::bundle(target)?.mtp_draft_margin();
                prefix *= draft_accept_estimate(margin);
                // A verify row pays only while its whole prefix is likely
                // accepted; the first draft always rides.
                if !derived_loop && index > 0 && prefix < MTP_ROW_WORTH {
                    break;
                }
                margins.push(margin);
                drafts.push(input);
                let drafted = drafts.len();
                // The actual head-step margin is causal evidence the moment it
                // is read: store it before any prune or re-solve, so the lab
                // receipt and the post-row re-solve share one capture.
                self.derived_margins_push(margin);
                if !margin.is_finite() || margin < 0.0 {
                    self.trace_derived_margin_dropped(position, drafted);
                }
                // Only a kept draft enters the penalty history: a pruned
                // proposal broke out above.
                if let Some(s) = sampled.as_mut() {
                    s.history.push_draft(input);
                }
                if !derived_loop && prefix * DRAFT_ACCEPT_MAX < MTP_ROW_WORTH {
                    break;
                }
                if self.derived_loop_active() && !self.derived_resolve(drafted) {
                    break;
                }
                if index + 1 < k_max {
                    known.push(input);
                    Self::bundle(target)?.warm_ple_rows(&known);
                }
            }
            let k = drafts.len();
            let mut block = Vec::with_capacity(k + 1);
            block.push(seed);
            block.extend_from_slice(&drafts);
            #[cfg(feature = "reference-parity")]
            {
                self.lab_actual_route = Some(crate::wide_verify_oracle::DevWindowRoute::Batched);
            }
            timers.mark(ops.gpu()?, "verify");
            let skip_picks = sampled
                .as_ref()
                .is_some_and(|s| s.device_skip_picks(k, trace || timers.enabled()));
            let picks = Self::bundle(target)?;
            let pending_hidden = self
                .pending_hidden
                .as_ref()
                .ok_or_else(|| "Qwen4 native MTP pending hidden is not allocated".to_string())?;
            let scratch = self
                .scratch
                .as_mut()
                .ok_or_else(|| "Qwen4 native MTP verify scratch is not allocated".to_string())?;
            let mut target_picks = if skip_picks {
                // Resident acceptance reads this window's logits on the
                // device: no argmax pick is downloaded or consulted.
                picks
                    .verify_block_resident(ops, &block, position, scratch.as_mut())
                    .map_err(|error| error.to_string())?;
                Vec::new()
            } else {
                let target_picks = picks
                    .verify_block_rows(ops, &block, position, scratch.as_mut(), true)
                    .map_err(|error| error.to_string())?;
                if sampled.is_some() && target_picks.len() < k + 1 {
                    return Err(format!(
                        "Qwen4 sampled MTP verifier returned {} rows for {k} drafts",
                        target_picks.len()
                    ));
                }
                target_picks
            };
            // `accept`: the target-row work (penalty prepass, row downloads,
            // host distributions) apart from the verify forward.
            timers.mark(ops.gpu()?, "accept");
            let acceptance = match sampled.as_mut() {
                // Naive: each row the verdict reads is replaced by its draw.
                Some(s) if !s.draws_drafts() => {
                    s.accept_naive(ops.gpu()?, picks, &drafts, eos, &mut target_picks)?
                }
                Some(s) => s.accept_leviathan(ops.gpu()?, picks, &drafts, false, eos)?,
                None => accept_native_greedy(&drafts, &target_picks, Some(eos))?,
            };
            accepted_drafts = acceptance.accepted;
            drafts_verified = drafts.len();
            emitted_len = acceptance.committed.len();
            if trace && !self.follower {
                let rows = drafts
                    .iter()
                    .zip(target_picks.iter())
                    .zip(margins.iter())
                    .map(|((draft, pick), margin)| {
                        format!(
                            "{{\"draft\":{draft},\"pick\":{pick},\"match\":{},\"margin\":{margin:.4}}}",
                            draft == pick
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                let gdn_resets = gdn_residual_trace_field(&picks.state);
                let verify_rows = k + 1;
                let row_limit = picks.row_exact_verify_rows(ops.gpu()?);
                eprintln!(
                    "QWEN4_MTP_TRACE {{\"event\":\"window\",\"source\":\"mtp\",\"verify_route\":\"batched\",\"accept_route\":\"{}\",\"position\":{position},\"k\":{k},\"accepted\":{accepted_drafts},\"verify_rows\":{verify_rows},\"row_limit\":{row_limit},\"rows\":[{rows}],\"committed\":{:?},\"baseline\":true{gdn_resets}}}",
                    accept_route_label(sampled.as_ref()),
                    acceptance.committed
                );
            }
            let target_scratch = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .ok_or("Qwen4 native MTP target scratch type changed")?;
            let target_snapshot = target_scratch
                .target_snapshot
                .ok_or("Qwen4 native MTP target snapshot disappeared")?;
            let mtp_ticket = snapshot
                .as_ref()
                .copied()
                .expect("MTP snapshot remains active until transaction commit");
            // The window's pre-window pending hidden, kept for a terminal
            // repair before the finish moves `pending_hidden` on.
            let retained_pending = self.retained_pending.as_ref().ok_or_else(|| {
                "Qwen4 native MTP retained pending hidden is not allocated".to_string()
            })?;
            ops.gpu()?
                .copy_d2d(pending_hidden, retained_pending, pending_hidden.byte_size())
                .map_err(|error| {
                    format!("Qwen4 native MTP retained pending hidden copy: {error}")
                })?;
            // The whole captured hidden and the verify's PLE rows from row 0.
            let verify_hidden = {
                let hidden = picks
                    .spec_hidden
                    .as_ref()
                    .ok_or_else(|| "Qwen4 native MTP spec hidden is not allocated".to_string())?;
                hidden.sub_offset(0, hidden.numel())
            };
            let ple_normed = if ops.is_mesh() {
                None
            } else {
                Some(
                    picks
                        .lane_ple_normed_rows(0, block.len())
                        .map_err(|error| error.to_string())?,
                )
            };
            let window = finish_native_window(
                picks,
                ops,
                NativeFinish {
                    position,
                    block: &block,
                    acceptance,
                    target_snapshot,
                    mtp_ticket,
                    pending_hidden,
                    verify_hidden: &verify_hidden,
                    ple_normed: ple_normed.as_ref(),
                    retain: true,
                },
                Some(&mut timers),
            )?;
            target_scratch.target_snapshot = None;
            snapshot = None;
            self.retained = Some(RetainedWindow {
                span: WindowSpan {
                    start: position,
                    end: picks.state.position,
                    seed,
                },
                target: target_snapshot,
                head: mtp_ticket,
            });
            Ok(NativeOutcome::Window(window))
        })();
        self.sampled = sampled;
        let result = match result {
            Ok(NativeOutcome::Window(window)) => Ok(window),
            Ok(NativeOutcome::Takeover) => {
                // No target verify ran (`target_snapshot` is None), so this
                // restores only the head; the draft policy goes back with it.
                timers.discard(ops.gpu()?);
                if let Err(rollback) = self.rollback_failed_window(ops, target, snapshot.take()) {
                    return Err(format!(
                        "Qwen4 MTP takeover switch could not restore the head: {rollback}"
                    ));
                }
                Self::bundle(target)?
                    .set_mtp_draft_request_state(draft_state)
                    .map_err(|error| error.to_string())?;
                let probe = takeover_probe(ops.single("n-gram takeover")?, window_start);
                return self.run_takeover(
                    ops.single("n-gram takeover")?,
                    target,
                    position,
                    seed,
                    emitted,
                    eos,
                    Some(probe),
                );
            }
            Err(error) => Err(error),
        };
        if let Ok(window) = &result {
            self.observe_agreement(window);
            // A derived window was routed at the width it drafted.
            let cap = if self.derived_loop_active() {
                window.drafts_generated
            } else {
                k_max
            };
            let stats = WindowStats::of(window, cap);
            self.finish_floor_window(!floor_active, NativeWindow::Batched, stats, k, window_start);
        }
        if timers.enabled() {
            let window_us = synced_wall_us(ops.gpu()?, window_start);
            let accept_route = accept_route_label(self.sampled.as_ref());
            let fields = format!(
                "\"source\":\"mtp\",\"verify_route\":\"batched\",\"accept_route\":\"{accept_route}\",\"position\":{position},\"k\":{drafts_verified},\"accepted\":{accepted_drafts},\"emitted\":{emitted_len},\"window_us\":{window_us:.1},\"t_end\":{}",
                unix_micros()
            );
            timers.finish(ops.gpu()?, "QWEN4_MTP_PHASE", &fields);
        }
        if let Err(error) = &result {
            if let Err(rollback) = self.rollback_failed_window(ops, target, snapshot.take()) {
                return Err(format!("{error}; rollback failed: {rollback}"));
            }
        }
        result
    }

    /// What a PP mesh does not serve, refused before any state changes:
    /// naive sampled verification (its draws come from the process-wide
    /// sampler, not the request-seeded stream), a penalized sampled request
    /// (the GPU penalty prepass refuses the mesh) and the n-gram takeover
    /// (one-card route). Greedy requests and unpenalized Leviathan sampled
    /// requests run: every device read of the verify (draft `q`, target rows,
    /// bonus) goes through [`Qwen4Ops::gpu`], the output band's device.
    fn refuse_unsupported_under_pp(&self) -> Result<(), String> {
        if !self.request.temp.is_finite() {
            return Err(
                "Qwen4 native MTP under qwen4 topology.pp needs a finite request temperature"
                    .to_string(),
            );
        }
        match self.sampled.as_ref() {
            Some(sampled) if !sampled.draws_drafts() => {
                return Err("Qwen4 native MTP under qwen4 topology.pp does not support \
                     HIPFIRE_MTP_SAMPLED_MODE=naive (its draws come from the process-wide AR \
                     sampler); use the default leviathan mode"
                    .to_string());
            }
            Some(_) if self.request.penalties_active() => {
                return Err(
                    "Qwen4 native MTP under qwen4 topology.pp does not support sampled requests \
                     with repeat/presence/frequency penalties (the GPU penalty prepass refuses \
                     the pipeline mesh); such requests decode AR"
                        .to_string(),
                );
            }
            Some(_) => {}
            None if self.request.temp.abs() > 1.0e-6 => {
                return Err(
                    "Qwen4 native MTP under qwen4 topology.pp verifies greedy requests only \
                     while sampled verification is off (speculation.mtp_sampled / \
                     HIPFIRE_MTP_SAMPLED=0)"
                        .to_string(),
                );
            }
            None => {}
        }
        if self.ngram_active {
            return Err(
                "Qwen4 native MTP under qwen4 topology.pp does not support the n-gram takeover \
                 (allow_ngram_modifier with an armed pool)"
                    .to_string(),
            );
        }
        Ok(())
    }

    /// The body of [`MtpDrafter::mtp_prefill`] past its EP fan-out, over an
    /// operation context. Under a mesh the target prefills through every band
    /// chunk by chunk; each chunk's head continuation runs on the output
    /// device before the next chunk's wide capture can overwrite its rows.
    pub(crate) fn mtp_prefill_ops(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        target: &mut dyn SpecTarget,
        prompt_tokens: &[u32],
        start_pos: usize,
        cache_hit: bool,
        abort: &dyn Fn() -> bool,
    ) -> Result<PrefillOutcome, String> {
        // The whole prompt is one decode-share / EP-reserve prompt: the donor
        // tail is restored once, after its last tile (`share_finish_prompt`).
        // `!ops.is_mesh()` names the single-context route; it does not exclude
        // per-rank EP execution (each EP rank runs this body on a `Single` context
        // inside the fan-out). A PP mesh never takes decode-share or the EP
        // reserve, and fails closed if one is somehow loaded.
        let single = !ops.is_mesh();
        if !single && Self::bundle(target)?.ep_reserve_active() {
            return Err(
                "Qwen4 native-MTP prefill over a PP mesh cannot run on an EP \
                        reserve-sharing load"
                    .to_string(),
            );
        }
        if single {
            Self::bundle(target)?.share_prompt_begin_mtp_fill();
        }
        let result =
            self.mtp_prefill_tiles(ops, target, prompt_tokens, start_pos, cache_hit, abort);
        if !single {
            return result;
        }
        let ok = result.is_ok();
        let finish = Self::bundle(target).and_then(|bundle| {
            bundle
                .share_prompt_end(ops.gpu()?, ok)
                .map_err(|error| error.to_string())
        });
        result.and_then(|outcome| finish.map(|()| outcome))
    }

    /// The tile loop of [`Self::mtp_prefill_ops`].
    fn mtp_prefill_tiles(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        target: &mut dyn SpecTarget,
        prompt_tokens: &[u32],
        start_pos: usize,
        cache_hit: bool,
        abort: &dyn Fn() -> bool,
    ) -> Result<PrefillOutcome, String> {
        // Bind the derived-K identity once per request, before any device
        // work (the request's first prefill; a realigning re-prefill keeps it).
        if matches!(self.derived, Some(DerivedPhase::Unbound)) {
            self.derived_bind_target(ops, target);
        }
        if ops.is_mesh() {
            self.refuse_unsupported_under_pp()?;
            if cache_hit || start_pos != 0 {
                return Err(
                    "Qwen4 native MTP prefill under qwen4 topology.pp is cold only: MTP \
                     prefix reuse (session restore) is not supported under the pipeline mesh"
                        .to_string(),
                );
            }
        }
        self.agreement.fill(AGREEMENT_PRIOR);
        // The request's n-gram history starts from the full canonical prompt
        // (cache hit or miss), once per request: a realignment re-prefill
        // keeps the history it already holds.
        if self.ngram_active {
            if let Some(ctx) = self.ngram.as_mut().filter(|ctx| !ctx.is_begun()) {
                ctx.begin_request(prompt_tokens);
            }
            if let Some(ctx) = self.ngram_copy.as_mut().filter(|ctx| !ctx.is_begun()) {
                ctx.begin_request(prompt_tokens);
            }
        }
        // A miss resets target, head and draft policy; a hit restores the
        // session snapshot at `start_pos` into all three. Either way no head
        // row is owed afterwards (the prefill overwrites spec hidden).
        let reused = if cache_hit { start_pos } else { 0 };
        self.head_lag.clear();
        self.derived_drop_pending();
        self.release_retained(target);
        if ops.is_mesh() {
            // Cold only (checked above): every band and the head reset, no
            // session record is begun or published on the MTP route.
            Self::bundle(target)?.ops_reset(ops)?;
        } else {
            let bundle = Self::bundle(target)?;
            // A bounded side request may park the committed live conversation
            // inside `session_begin`, a capture at the live position that no
            // prefill boundary produced. A restore of that park refuses a
            // snapshot without the boundary row, so keep the live one: with
            // target and head aligned, `pending` is the target hidden of row
            // `position - 1`. The row is position-stamped; a capture at any
            // other position never reads it.
            if let Some(pending) = self.pending_hidden.as_ref() {
                let position = bundle.state.position;
                if position > 0 && bundle.mtp_position().is_ok_and(|head| head == position) {
                    bundle
                        .session_save_boundary_hidden(ops.gpu()?, pending)
                        .map_err(|error| error.to_string())?;
                }
            }
            bundle
                .session_begin(ops.gpu()?, prompt_tokens, reused, SessionRoute::Mtp)
                .map_err(|error| error.to_string())?;
        }
        self.ensure_resources(ops, target)?;
        {
            let bundle = Self::bundle(target)?;
            let target_position = bundle.state.position;
            let mtp_position = bundle.mtp_position().map_err(|error| error.to_string())?;
            if target_position != start_pos || mtp_position != start_pos {
                return Err(format!(
                    "Qwen4 MTP prefill position mismatch: target={}, mtp={}, start={start_pos}",
                    target_position, mtp_position
                ));
            }
        }
        // The first fill row pairs with the target hidden of `start_pos - 1`,
        // which the session snapshot carries (saved at its boundary below)
        // and a live continuation already holds in `pending`: a snapshot
        // restore puts it there; `false` (live, no restore) leaves the live one.
        // A mesh prefill is cold only (checked above), so it has no restore.
        if cache_hit && start_pos > 0 {
            let pending = self.pending_hidden()?;
            Self::bundle(target)?
                .session_restore_boundary_hidden(ops.gpu()?, pending)
                .map_err(|error| error.to_string())?;
        }
        // Field borrows (not `pending_hidden()`): the batched fill below also
        // borrows `append_scratch` mutably.
        let pending = self
            .pending_hidden
            .as_ref()
            .ok_or_else(|| "Qwen4 MTP pending hidden is not allocated".to_string())?;
        // Unfiltered: `mtp_prefill_chunk` decides per chunk whether the
        // scratch is used (batched fill, or the DET prompt fill).
        let mut batched_scratch = self.append_scratch.as_mut();
        // The trunk's uploaded ids are another device's under a mesh (the
        // first band's): the head uploads its own copy on the output device.
        let reuse_trunk_ids = mtp_reuse_prefill_ids_enabled() && !ops.is_mesh();
        let mut first_token = None;
        // One chunked target forward per chunk instead of one single-row forward
        // per prompt token: the shared forward already captures the whole
        // chunk's wide hidden, and the head then consumes each row in order.
        // Cost goes from ~1 single-row forward per prompt token to the ordinary
        // chunked prefill rate plus one head step per token.
        // Each prompt token selects with its own query and the pooled keys
        // visible at that position, regardless of target prefill chunking.
        let skip_intermediate_pick = mtp_skip_intermediate_pick_enabled();
        let chunk_rows = self.prefill_rows.max(1);
        let mut pos = start_pos;
        // The whole remaining prompt goes through the slow-link selector
        // BEFORE the per-chunk loop splits it: a qualified request runs its
        // target layer-major (one expert sweep per window) and this head
        // consumes the bounded per-tile hidden rows the final layer emits, in
        // order: token `p` pairs with the target hidden of `p - 1` (`pending`
        // for a tile's first row, prompt position 0 with its own), a due
        // session cut is published once its tile is appended, and the head and
        // `pending` are part of the transaction. The selector declining
        // (before any mutation) leaves the incumbent loop below untouched.
        let capture_eligible = if !ops.is_mesh() && start_pos < prompt_tokens.len() {
            let eligible = Self::bundle(target)?.slowlink_capture_eligible();
            if let Err(why) = eligible {
                tracing::debug!(
                    bypass_reason = why,
                    prompt_rows = prompt_tokens.len() - start_pos,
                    "qwen4 MTP prompt fill: slow-link capture bypassed",
                );
            }
            eligible.is_ok()
        } else {
            false
        };
        if capture_eligible {
            let aborted = std::cell::Cell::new(false);
            let mut sink = |gpu: &mut Gpu,
                            bundle: &mut Qwen4Bundle,
                            tile: crate::slowlink_prefill::PromptTile<'_>|
             -> Result<(), String> {
                if abort() {
                    aborted.set(true);
                    return Err("Qwen4 native MTP prefill aborted".to_string());
                }
                let end = tile
                    .start
                    .checked_add(tile.rows)
                    .ok_or_else(|| "Qwen4 native MTP prefill position overflow".to_string())?;
                let chunk = prompt_tokens
                    .get(tile.start..end)
                    .ok_or_else(|| "Qwen4 slow-link tile is outside the prompt".to_string())?;
                let flow = {
                    let mut tile_ops = Qwen4Ops::Single(&mut *gpu);
                    // The wide rows of this tile are `spec_hidden` rows
                    // `0..rows`; the abort poll is the sink's own (a reset
                    // inside an open transaction would orphan its tickets).
                    mtp_append_prefill_chunk(
                        bundle,
                        &mut tile_ops,
                        chunk,
                        tile.start,
                        0,
                        pending,
                        batched_scratch.as_deref_mut(),
                        false,
                        &|| false,
                    )?
                };
                if flow.is_break() {
                    return Err("Qwen4 native MTP tile append stopped".to_string());
                }
                if tile.last {
                    // The window end: the snapshot keeps `pending` (the target
                    // hidden of `end - 1`); the transaction's commit publishes it.
                    if bundle.session_next_boundary() == Some(end) {
                        bundle
                            .session_save_boundary_hidden(gpu, pending)
                            .map_err(|error| error.to_string())?;
                    }
                } else if bundle.slowlink_cut_due(end) {
                    bundle
                        .slowlink_publish_cut_mtp(gpu, pending)
                        .map_err(|error| error.to_string())?;
                }
                // `prefill_progress` (silent outside the request's prompt
                // phase and past the last tile).
                hipfire_runtime::prefill_progress::report(
                    end - start_pos,
                    prompt_tokens.len() - start_pos,
                );
                Ok(())
            };
            let outcome = Self::bundle(target)?.slowlink_try_capture(
                ops.gpu()?,
                &prompt_tokens[start_pos..],
                pending,
                &mut sink,
            );
            match outcome {
                Ok(Some(seed)) => {
                    first_token = Some(seed);
                    pos = prompt_tokens.len();
                }
                // Declined before any mutation: the incumbent loop runs.
                Ok(None) => {}
                Err(error) => {
                    if aborted.get() {
                        Self::bundle(target)?.ops_reset(ops)?;
                        return Ok(PrefillOutcome::Aborted);
                    }
                    // The transaction restored target and head to the entry
                    // state; put the entry `pending` hidden back with them,
                    // but only from a backup taken for the window the state
                    // was restored to (each window saves before its begin).
                    let message = error.to_string();
                    if !Self::bundle(target)?.slowlink_pending_restorable() {
                        return Err(message);
                    }
                    let restored =
                        Self::bundle(target)?.slowlink_restore_pending(ops.gpu()?, pending);
                    return Err(match restored {
                        Ok(()) => message,
                        Err(restore) => {
                            format!("{message}; pending hidden restore failed: {restore}")
                        }
                    });
                }
            }
        }
        while pos < prompt_tokens.len() {
            if abort() {
                Self::bundle(target)?.ops_reset(ops)?;
                return Ok(PrefillOutcome::Aborted);
            }
            // A chunk ends at the session cache's next snapshot boundary when
            // one lies inside the natural chunk, else at the natural chunk end.
            let rows = if ops.is_mesh() {
                // Every band consumes the same canonical chunk descriptors.
                chunk_rows.min(prompt_tokens.len() - pos)
            } else {
                Self::bundle(target)?
                    .poll_npu_trunk(ops.gpu()?)
                    .map_err(|error| error.to_string())?;
                Self::bundle(target)?.prefill_schedule_rows(pos, prompt_tokens.len(), chunk_rows)
            };
            let natural = pos
                .checked_add(rows)
                .ok_or_else(|| "Qwen4 native MTP prefill position overflow".to_string())?;
            let end = natural;
            if end <= pos || end > natural {
                return Err(format!(
                    "Qwen4 native MTP prefill chunk end {end} is outside ({pos}, {natural}]"
                ));
            }
            let chunk = &prompt_tokens[pos..end];
            if !ops.is_mesh() {
                Self::bundle(target)?.set_ple_lookahead(&prompt_tokens[end..]);
            }
            // Only the prompt's last chunk's pick is read (the seed): an
            // earlier chunk still captures its full wide hidden for the head
            // but skips the final hyper, LM head, argmax and readback.
            let needs_pick = if chunk.len() > chunk_rows {
                end == prompt_tokens.len()
            } else {
                mtp_prefill_chunk_needs_pick(end, prompt_tokens.len(), skip_intermediate_pick)
            };
            let ControlFlow::Continue(pick) = mtp_prefill_chunk(
                Self::bundle(target)?,
                ops,
                chunk,
                pos,
                needs_pick,
                pending,
                batched_scratch.as_deref_mut(),
                reuse_trunk_ids,
                abort,
            )?
            else {
                return Ok(PrefillOutcome::Aborted);
            };
            // Target and head both hold the exact state at `end` only now: a
            // due snapshot boundary is reported here, after the forward and the
            // head append of this chunk.
            if !ops.is_mesh() && Self::bundle(target)?.session_next_boundary() == Some(end) {
                // `pending` is the target hidden of `end - 1` now: the
                // snapshot keeps it for the restore's first fill row.
                Self::bundle(target)?
                    .session_save_boundary_hidden(ops.gpu()?, pending)
                    .map_err(|error| error.to_string())?;
                Self::bundle(target)?
                    .session_at_boundary(ops.gpu()?, &prompt_tokens[..end])
                    .map_err(|error| error.to_string())?;
            }
            if pick.is_some() {
                first_token = pick;
            }
            pos = end;
            hipfire_runtime::prefill_progress::report(
                pos - start_pos,
                prompt_tokens.len() - start_pos,
            );
        }
        let pick = first_token.expect("non-empty MTP prefill produced no seed");
        // The seed is the first emitted token: sampled, it is drawn from the
        // last prompt row's `p` (naive: with the AR sampler itself), exactly
        // as AR draws its first token. Its penalty history is the full prompt
        // (cold and cache-hit alike); `fill_tokens` is only the uncached tail.
        let first_token = match self.sampled.as_mut() {
            Some(s) => {
                if !s.prompt_set {
                    s.history.set_prompt(prompt_tokens);
                    s.prompt_set = true;
                }
                if s.draws_drafts() {
                    s.load_target(ops.gpu()?, Self::bundle(target)?, 0, 0)?;
                    s.target.sample(s.rng.next_f32())
                } else {
                    s.naive_draw(ops.gpu()?, Self::bundle(target)?, 0, 0)?
                }
            }
            None => pick,
        };
        Ok(PrefillOutcome::Ready { first_token })
    }

    /// The body of [`MtpDrafter::mtp_forced_advance`] past its EP fan-out.
    pub(crate) fn mtp_forced_advance_ops(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        target: &mut dyn SpecTarget,
        tokens: &[u32],
        start_pos: usize,
        abort: &dyn Fn() -> bool,
    ) -> Result<bool, String> {
        if ops.is_mesh() {
            self.refuse_unsupported_under_pp()?;
        }
        // The retained repair window ends before any resource change or forced
        // append: a live retained ticket pins the row-capture rings, so release
        // it first (as every other step/prefill entry does).
        self.release_retained(target);
        self.ensure_resources(ops, target)?;
        // A retired request's head lags by the rows it owes: append them
        // first, so these rows follow them and the advance (the terminal
        // pending-seed flush among them) ends with head and target together.
        self.catch_up_head(ops, target)?;
        let pending = self.pending_hidden()?;
        for (index, &token) in tokens.iter().enumerate() {
            if abort() {
                return Ok(true);
            }
            let position = start_pos
                .checked_add(index)
                .ok_or_else(|| "Qwen4 native MTP forced position overflow".to_string())?;
            let bundle = Self::bundle(target)?;
            bundle.ops_capture_token(ops, token, true)?;
            // The token pairs with the target hidden of `position - 1`, which
            // `pending` holds; `pending` then takes this token's own.
            bundle
                .mtp_append_token(ops.gpu()?, token, Some(pending), position)
                .map_err(|error| error.to_string())?;
            bundle
                .copy_spec_hidden_row_to(ops.gpu()?, 0, pending)
                .map_err(|error| error.to_string())?;
        }
        Ok(true)
    }

    /// The body of [`MtpDrafter::mtp_repair_terminal_prefix`] over an
    /// operation context.
    pub(crate) fn repair_terminal_prefix_ops(
        &mut self,
        ops: &mut Qwen4Ops<'_>,
        target: &mut dyn SpecTarget,
        window_start: usize,
        window_seed: u32,
        consumed: &[u32],
    ) -> Result<bool, String> {
        self.derived_drop_pending();
        let Some(window) = self.retained.take() else {
            return Ok(false);
        };
        let replay = terminal_repair_replay(window.span, window_start, window_seed, consumed);
        let bundle = Self::bundle(target)?;
        let usable = replay
            .as_ref()
            .is_some_and(|rows| rows.len() <= bundle.spec_verify_rows_cap())
            && bundle.ep_mesh().is_none()
            && bundle.state.position == window.span.end
            && bundle
                .mtp_position()
                .is_ok_and(|head| head == window.span.end)
            && bundle.ops_window_valid(window.target, window.head).is_ok();
        let replay = match replay {
            Some(replay) if usable => replay,
            _ => {
                bundle.ops_release_window(window.target, window.head);
                return Ok(false);
            }
        };
        let repaired = self.replay_window_prefix(ops, target, window, &replay);
        Self::bundle(target)?.ops_release_window(window.target, window.head);
        repaired.map(|()| true)
    }

    /// The mesh context of a PP entry: the bundle must be the output-band root
    /// of an attached mesh and `gpus` that mesh's devices.
    fn pp_ops<'g>(bundle: &Qwen4Bundle, gpus: &'g mut Gpus) -> Result<Qwen4Ops<'g>, String> {
        if !bundle.pp_mesh_active() {
            return Err("Qwen4 native MTP: no pipeline mesh is attached".to_string());
        }
        // The supplied `Gpus` must be the mesh this bundle was attached to
        // (pure check, before any resource or checkpoint work).
        bundle
            .pp_require_ready(gpus)
            .map_err(|error| error.to_string())?;
        Qwen4Ops::mesh(gpus)
    }

    /// [`MtpDrafter::mtp_prefill`] on a PP mesh: a cold greedy prefill of the
    /// whole prompt through every band, the head filled chunk by chunk on the
    /// output device. `fill_tokens`/`start_pos`/`cache_hit` are validated like
    /// the single-card entry; prefix reuse is refused (the MTP route has no
    /// aggregate session record). Returns the first-token seed, or `Aborted`.
    pub fn pp_mtp_prefill(
        &mut self,
        gpus: &mut Gpus,
        bundle: &mut Qwen4Bundle,
        prompt_tokens: &[u32],
        fill_tokens: &[u32],
        start_pos: usize,
        cache_hit: bool,
        abort: &dyn Fn() -> bool,
    ) -> Result<PrefillOutcome, String> {
        self.require_supported_request()?;
        validate_native_mtp_prefill_request(prompt_tokens, fill_tokens, start_pos, cache_hit)?;
        self.refuse_unsupported_under_pp()?;
        if cache_hit || start_pos != 0 {
            return Err(
                "Qwen4 native MTP prefill under qwen4 topology.pp is cold only: MTP \
                 prefix reuse (session restore) is not supported under the pipeline mesh"
                    .to_string(),
            );
        }
        let mut ops = Self::pp_ops(bundle, gpus)?;
        self.mtp_prefill_ops(&mut ops, bundle, prompt_tokens, start_pos, cache_hit, abort)
    }

    /// [`MtpDrafter::mtp_step`] on a PP mesh: plan the window on the host,
    /// then run the one window body over every band and the output-band head.
    #[allow(clippy::too_many_arguments)]
    pub fn pp_mtp_step(
        &mut self,
        gpus: &mut Gpus,
        bundle: &mut Qwen4Bundle,
        position: usize,
        seed: u32,
        emitted: &[u32],
        k: usize,
        eos: u32,
    ) -> Result<MtpWindow, String> {
        self.require_supported_request()?;
        self.refuse_unsupported_under_pp()?;
        let mut ops = Self::pp_ops(bundle, gpus)?;
        // The floor's pricing and decay read the output device's arch before
        // the window's resource pass sets them again (to the same values).
        self.bind_arch(ops.gpu()?);
        let plan = self.plan_window(emitted, k)?;
        self.run_window_ops(&mut ops, bundle, position, seed, emitted, eos, &plan)
    }

    /// [`MtpDrafter::mtp_forced_advance`] on a PP mesh. `Ok(true)`: the
    /// drafter advanced the target itself (or aborted).
    pub fn pp_mtp_forced_advance(
        &mut self,
        gpus: &mut Gpus,
        bundle: &mut Qwen4Bundle,
        tokens: &[u32],
        start_pos: usize,
        abort: &dyn Fn() -> bool,
    ) -> Result<bool, String> {
        if tokens.is_empty() || abort() {
            return Ok(true);
        }
        self.refuse_unsupported_under_pp()?;
        let mut ops = Self::pp_ops(bundle, gpus)?;
        self.mtp_forced_advance_ops(&mut ops, bundle, tokens, start_pos, abort)
    }

    /// [`MtpDrafter::mtp_repair_terminal_prefix`] on a PP mesh: every band
    /// and the head restored to the retained window's start (one aggregate
    /// restore), then the consumed prefix replayed through every band and
    /// appended to the head. `Ok(false)`: nothing retained / not applicable;
    /// the caller takes its reset path.
    pub fn pp_mtp_repair_terminal_prefix(
        &mut self,
        gpus: &mut Gpus,
        bundle: &mut Qwen4Bundle,
        window_start: usize,
        window_seed: u32,
        consumed: &[u32],
    ) -> Result<bool, String> {
        if self.retained.is_none() {
            return Ok(false);
        }
        let mut ops = Self::pp_ops(bundle, gpus)?;
        self.repair_terminal_prefix_ops(&mut ops, bundle, window_start, window_seed, consumed)
    }

    /// [`MtpDrafter::mtp_reset`] on a PP mesh: the drafter's own state only
    /// (the mesh's trunk reset is `Qwen4Bundle::pp_reset`). Tickets of a
    /// window the mesh reset already invalidated are dropped silently.
    pub fn pp_mtp_reset(
        &mut self,
        gpus: &mut Gpus,
        bundle: &mut Qwen4Bundle,
    ) -> Result<(), String> {
        // Pure identity check first: a foreign `Gpus` releases nothing and
        // memsets nothing.
        bundle
            .pp_require_ready(gpus)
            .map_err(|error| error.to_string())?;
        if let Some(window) = self.retained.take() {
            bundle.ops_release_window(window.target, window.head);
        }
        if let Some(scratch) = self
            .scratch
            .as_mut()
            .and_then(|scratch| scratch.as_any_mut().downcast_mut::<Qwen4SpecScratch>())
        {
            if let Some(TargetTicket::Mesh(window)) = scratch.target_snapshot.take() {
                bundle.ops_release_window(TargetTicket::Mesh(window), HeadTicket::Mesh(window));
            }
        }
        let gpu = mesh_output_gpu(gpus)?;
        MtpDrafter::mtp_reset(self, gpu)
    }

    /// [`MtpDrafter::mtp_reset_for_realign`] on a PP mesh (identical to
    /// [`Self::pp_mtp_reset`], as on a single card).
    pub fn pp_mtp_reset_for_realign(
        &mut self,
        gpus: &mut Gpus,
        bundle: &mut Qwen4Bundle,
    ) -> Result<(), String> {
        self.pp_mtp_reset(gpus, bundle)
    }
}

#[cfg(feature = "lab")]
impl Qwen4MtpDrafter {
    /// Oracle reference: the AR-floor window (`calibrate` = the request's
    /// first, head-synced window; else the retired head-free one) of
    /// `mtp_step`, after the resources `mtp_step` ensures.
    pub fn lab_ar_window(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        calibrate: bool,
    ) -> Result<MtpWindow, String> {
        let mut ops = Qwen4Ops::Single(gpu);
        self.ensure_resources(&mut ops, target)?;
        self.mtp_ar_step(&mut ops, target, position, seed, calibrate)
    }

    /// F32 bytes of the pending hidden row.
    pub fn lab_pending_hidden_bytes(&self, gpu: &Gpu) -> Result<Vec<u8>, String> {
        let pending = self.pending_hidden()?;
        gpu.hip
            .device_synchronize()
            .map_err(|error| format!("Qwen4 MTP lab pending hidden sync: {error}"))?;
        let mut bytes = vec![0u8; pending.byte_size()];
        gpu.hip
            .memcpy_dtoh(&mut bytes, &pending.buf)
            .map_err(|error| format!("Qwen4 MTP lab pending hidden download: {error}"))?;
        Ok(bytes)
    }

    /// Arm (or disarm) the production-window observation. Armed, every
    /// `mtp_step` window leaves one [`crate::lab_window::LabWindow`] for
    /// [`Self::lab_take_window_observation`]; the bundle's state and
    /// forward-logits taps are armed at each window's start. Disarming drops
    /// every unread observation. Read-only: nothing here changes a draw, a
    /// verdict, a route or a launch.
    pub fn lab_observe_windows(&mut self, arm: crate::lab_window::LabWindowArm) {
        self.lab_arm = arm;
        if !arm.armed {
            self.lab_open = None;
            self.lab_done.clear();
            if let Some(sampled) = self.sampled.as_mut() {
                sampled.lab = None;
            }
        }
    }

    /// The oldest unread window observation (one per observed `mtp_step`).
    pub fn lab_take_window_observation(&mut self) -> Option<crate::lab_window::LabWindow> {
        self.lab_done.pop_front()
    }

    /// Every unread window observation, oldest first.
    pub fn lab_take_window_observations(&mut self) -> Vec<crate::lab_window::LabWindow> {
        self.lab_done.drain(..).collect()
    }

    /// Open the window record before `plan` runs; `false` when unarmed (no
    /// work done).
    fn lab_window_begin(
        &mut self,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        plan: &WindowPlan,
    ) -> bool {
        use crate::lab_window::{LabVerifyRoute, LabWindow};
        if !self.lab_arm.armed {
            return false;
        }
        let Ok(bundle) = Self::bundle(target) else {
            return false;
        };
        bundle.lab_observe(true, self.lab_arm.keep_logits);
        let pre_state = bundle.state.lab_take_state_events();
        let pre_forwards = bundle.lab_take_forward_logits();
        let resets = bundle.state.gdn_residual_resets();
        let route = match plan.route {
            WindowRoute::ArRetired | WindowRoute::ArRouted | WindowRoute::ArCalibrate => {
                LabVerifyRoute::TargetOnly
            }
            WindowRoute::Native { depth: 0 } => LabVerifyRoute::Interleaved,
            WindowRoute::Native { .. } => LabVerifyRoute::Batched,
        };
        let (sampled, naive, rng) = match self.sampled.as_ref() {
            Some(s) => (true, s.naive, Some(s.rng)),
            None => (false, false, None),
        };
        let window = Box::new(LabWindow::begin(
            position,
            seed,
            format!("{:?}", plan.route),
            plan.k,
            plan.ngram_hit,
            route,
            sampled,
            naive,
            rng,
            resets,
            pre_state,
            pre_forwards,
        ));
        match self.sampled.as_mut() {
            Some(s) => s.lab = Some(window),
            None => self.lab_open = Some(window),
        }
        true
    }

    /// Close the window record `lab_window_begin` opened and queue it.
    fn lab_window_end(&mut self, target: &mut dyn SpecTarget, result: &Result<MtpWindow, String>) {
        let open = self.sampled.as_mut().and_then(|s| s.lab.take());
        let Some(mut window) = open.or_else(|| self.lab_open.take()) else {
            return;
        };
        let (state, forwards, resets) = match Self::bundle(target) {
            Ok(bundle) => (
                bundle.state.lab_take_state_events(),
                bundle.lab_take_forward_logits(),
                bundle.state.gdn_residual_resets(),
            ),
            Err(_) => (
                crate::lab_window::LabStateEvents::default(),
                Vec::new(),
                window.gdn_residual_resets_before,
            ),
        };
        let rng_end = self.sampled.as_ref().map(|s| s.rng);
        if let Some(s) = self.sampled.as_ref() {
            window.device_route |= s.route_device;
        }
        window.finish(result, rng_end.as_ref(), resets, state, forwards);
        self.lab_done.push_back(*window);
    }
}

impl Qwen4MtpDrafter {
    /// The prefill after validation and the EP branch: restore or reset target, head and draft policy at `start_pos`
    /// (`cache_hit`: the session snapshot there; else cold), fill the rest, and draw the seed. Re-entrant after an NPU
    /// fault: [`Self::mtp_prefill`] reruns it from the re-planned start, and the n-gram history and the sampled
    /// penalty prompt are set once per request.
    fn mtp_prefill_from(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        prompt_tokens: &[u32],
        start_pos: usize,
        cache_hit: bool,
        abort: &dyn Fn() -> bool,
    ) -> Result<PrefillOutcome, String> {
        self.mtp_prefill_ops(
            &mut Qwen4Ops::Single(gpu),
            target,
            prompt_tokens,
            start_pos,
            cache_hit,
            abort,
        )
    }
}

/// The rerun context of [`Qwen4MtpDrafter::mtp_prefill`]: both owners, and the restart row the fault check re-planned.
struct MtpPrefillRun<'a> {
    drafter: &'a mut Qwen4MtpDrafter,
    target: &'a mut dyn SpecTarget,
    restart: usize,
}

impl MtpDrafter for Qwen4MtpDrafter {
    fn mtp_prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        prompt_tokens: &[u32],
        fill_tokens: &[u32],
        start_pos: usize,
        cache_hit: bool,
        abort: &dyn Fn() -> bool,
    ) -> Result<PrefillOutcome, String> {
        self.require_supported_request()?;
        validate_native_mtp_prefill_request(prompt_tokens, fill_tokens, start_pos, cache_hit)?;
        // Expert parallelism: rank 0 and every peer's replica run the whole
        // prefill in one fan-out. A request this drafter cannot serve under EP
        // is refused, and an abort is polled, here on rank 0, before any
        // rank starts: an error or an abort inside the fan-out would strand the
        // others (inside it every rank uses a never-abort closure).
        if Self::bundle(target)?.ep_mesh().is_some() {
            self.refuse_naive_under_ep()?;
            if abort() {
                target.reset_recurrent(gpu)?;
                return Ok(PrefillOutcome::Aborted);
            }
            // Inside the fan-out every rank runs never-abort (`replica_mtp_prefill`).
            return Self::bundle(target)?.ep_drafter_fanout(
                "mtp_prefill",
                gpu,
                self,
                |drafter, bundle, gpu| {
                    replica_mtp_prefill(
                        drafter,
                        bundle,
                        gpu,
                        prompt_tokens,
                        fill_tokens,
                        start_pos,
                        cache_hit,
                    )
                },
                &|drafter, bundle, gpu| {
                    replica_mtp_prefill(
                        drafter,
                        bundle,
                        gpu,
                        prompt_tokens,
                        fill_tokens,
                        start_pos,
                        cache_hit,
                    )
                },
            );
        }
        // An NPU fault latches the route off; the request then reruns from the re-planned session start with the exact
        // GPU-only schedule (`mtp_prefill_from` restores or resets target, head and draft policy before its first row).
        let prompt_len = prompt_tokens.len();
        let mut run = MtpPrefillRun {
            drafter: self,
            target,
            restart: 0,
        };
        crate::npu_trunk::containment::rerun_after_fault(
            &mut run,
            prompt_len,
            start_pos,
            |run, start| {
                let hit = if start == start_pos {
                    cache_hit
                } else {
                    start > 0
                };
                run.drafter.mtp_prefill_from(
                    gpu,
                    &mut *run.target,
                    prompt_tokens,
                    start,
                    hit,
                    abort,
                )
            },
            |run| {
                let Ok(bundle) = Qwen4MtpDrafter::bundle(&mut *run.target) else {
                    return false;
                };
                if !bundle.take_npu_trunk_fault() {
                    return false;
                }
                run.restart =
                    bundle.session_plan_capped(prompt_tokens, SessionRoute::Mtp, prompt_len);
                true
            },
            |run| run.restart,
        )
    }

    /// `plan_window` (rank 0) then `run_window`. Single card: the two run
    /// back to back on this drafter, in the order the whole window once ran
    /// in. Under EP rank 0 plans and every rank runs the plan in one fan-out
    /// (see [`Qwen4Bundle::ep_drafter_fanout`]); the hosts agree on the
    /// window.
    fn mtp_step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        k: usize,
        eos: u32,
        _grammar: Option<&mut dyn SpecGrammar>,
    ) -> Result<MtpWindow, String> {
        let ep = Self::bundle(target)?.ep_mesh().is_some();
        if ep {
            self.refuse_naive_under_ep()?;
        }
        // The floor's n-gram pricing and decay read the arch before the
        // window's resource pass sets them again (to the same values).
        self.bind_arch(gpu);
        self.bind_spec_rows(target)?;
        let rows = Self::bundle(target)?.row_exact_verify_rows(gpu);
        if self.copy_lane {
            self.row_exact_rows = rows;
        }
        self.derived_refresh_rows(rows);
        #[cfg(feature = "reference-parity")]
        if self.lab_policy_forced() {
            if ep {
                return Err("forced windows require singleton execution".into());
            }
            let bundle = Self::bundle(target)?;
            let caps = crate::wide_verify_oracle::dev_verify_caps(bundle, gpu);
            let cpu = bundle.has_cpu_deferred_experts() || bundle.cpu_storage_capacity_rows() > 0;
            let qualified = if cpu {
                let lab_armed =
                    rdna_compute::prefill_invariant::qwen4_wide_verify_override_present(&gpu.arch)
                        && !gpu.replay.is_recording()
                        && !gpu.graphs.capture_mode;
                bundle.cpu_qualified_verify_rows(gpu, lab_armed)
            } else {
                caps.row_exact_verify_rows
            };
            let timing_only = self.lab_forced_window.is_some_and(|r| r.timing_only);
            let limit = if timing_only && !cpu {
                caps.spec_verify_rows_cap
            } else {
                caps.spec_verify_rows_cap.min(qualified)
            };
            let d = self
                .lab_forced_window
                .map(|r| r.drafts)
                .or(self.lab_force_drafts)
                .unwrap_or(0)
                .min(k);
            check_spec_verify_rows(limit, d + 1, "forced Qwen4 window")?;
        }
        // The absolute position is the derived-K context bucket's input; only
        // this singleton route carries it (EP and PP never consult).
        if !ep && self.derived.is_some() {
            self.derived_position = Some(position);
            let reference = Self::bundle(target)?.ar_verify_reference(gpu);
            if let Some(DerivedPhase::Bound(state)) = self.derived.as_mut() {
                if let Some(reference) = reference {
                    state.reference = Some(matches!(
                        reference,
                        rdna_compute::prefill_invariant::Qwen4VerifyReference::Recorded
                    ));
                } else {
                    self.derived = Some(DerivedPhase::Ineligible);
                }
            }
        }
        let plan = self.plan_window(emitted, k);
        self.derived_position = None;
        let mut plan = plan?;
        if !ep {
            return self.run_window(gpu, target, position, seed, emitted, eos, &plan);
        }
        self.attach_candidates(&mut plan);
        let plan = &plan;
        let (committed, accepted, drafts_generated) = Self::bundle(target)?.ep_drafter_fanout(
            "mtp_step",
            gpu,
            self,
            |drafter, bundle, gpu| {
                window_key(drafter.run_window(gpu, bundle, position, seed, emitted, eos, plan))
            },
            &|drafter, bundle, gpu| {
                window_key(drafter.run_window(gpu, bundle, position, seed, emitted, eos, plan))
            },
        )?;
        Ok(MtpWindow {
            committed,
            accepted,
            drafts_generated,
        })
    }

    fn mtp_forced_advance(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        tokens: &[u32],
        start_pos: usize,
        abort: &dyn Fn() -> bool,
    ) -> Result<bool, String> {
        if tokens.is_empty() {
            return Ok(true);
        }
        if abort() {
            return Ok(true);
        }
        // Expert parallelism: rank 0 decided above that no abort is pending;
        // every rank advances the rows in one fan-out (never-abort inside)
        // and the hosts agree.
        if Self::bundle(target)?.ep_mesh().is_some() {
            let never = || false;
            return Self::bundle(target)?.ep_drafter_fanout(
                "mtp_forced_advance",
                gpu,
                self,
                |drafter, bundle, gpu| {
                    drafter.mtp_forced_advance(gpu, bundle, tokens, start_pos, &never)
                },
                &|drafter, bundle, gpu| {
                    drafter.mtp_forced_advance(gpu, bundle, tokens, start_pos, &never)
                },
            );
        }
        self.mtp_forced_advance_ops(&mut Qwen4Ops::Single(gpu), target, tokens, start_pos, abort)
    }

    /// Repair a terminal stop inside the last committed window: restore the
    /// pre-window target and head tickets the window retained, then replay
    /// only the consumed prefix (`seed` plus all but the last consumed token,
    /// which stays pending for the caller's terminal flush). Target and head
    /// both advance teacher-forced over the replay, the pending hidden becomes
    /// the last replayed row's, and the retained tickets are released.
    ///
    /// `Ok(false)` leaves the caller on its reset path: nothing retained (a
    /// one-row window without `SpecRequestConfig::stop_sequences`, an AR
    /// window, an EP mesh, anything between the window and this call), a
    /// different window, no consumed token, or a state that moved since the
    /// window (the head's seed row pairs with the pre-window pending hidden
    /// the window saved in `retained_pending`). An `Err` means the restore or
    /// replay itself failed and the owners are in an unknown state.
    fn mtp_repair_terminal_prefix(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        window_start: usize,
        window_seed: u32,
        consumed: &[u32],
    ) -> Result<bool, String> {
        self.repair_terminal_prefix_ops(
            &mut Qwen4Ops::Single(gpu),
            target,
            window_start,
            window_seed,
            consumed,
        )
    }

    fn mtp_reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.push_replay(ReplayOp::Reset);
        self.head_lag.clear();
        self.derived_drop_pending();
        self.retained = None;
        // A reset's work is not the last window's: leave it unpriced.
        self.floor_pending = None;
        if let Some(scratch) = self.scratch.as_mut() {
            if let Some(scratch) = scratch.as_any_mut().downcast_mut::<Qwen4SpecScratch>() {
                scratch.target_snapshot = None;
            }
        }
        if let Some(hidden) = self.pending_hidden.as_ref() {
            gpu.hip
                .memset(&hidden.buf, 0, hidden.buf.size())
                .map_err(|error| format!("Qwen4 native MTP pending reset: {error}"))?;
        }
        Ok(())
    }

    fn mtp_free(self: Box<Self>, gpu: &mut Gpu) {
        let Self {
            scratch,
            pending_hidden,
            append_scratch,
            sampled,
            device_spare,
            retained_pending,
            ..
        } = *self;
        if let Some(scratch) = scratch {
            scratch.free(gpu);
        }
        if let Some(hidden) = pending_hidden {
            let _ = gpu.free_tensor(hidden);
        }
        if let Some(hidden) = retained_pending {
            let _ = gpu.free_tensor(hidden);
        }
        if let Some(scratch) = append_scratch {
            let _ = scratch.free_gpu(gpu);
        }
        for device in device_spare
            .into_iter()
            .chain(sampled.and_then(|state| state.device))
        {
            device.free(gpu);
        }
    }

    /// Every drafter tensor lives on the output device. Under a PP mesh that
    /// is the mesh's output owner (not necessarily device 0): an absent owner
    /// or a failed bind leaks the buffers rather than freeing them on another
    /// band's device. A single card or an EP mesh frees on the first device,
    /// the trait default.
    fn mtp_free_multi(mut self: Box<Self>, gpus: &mut Gpus) {
        if gpus.mesh.size_of(DimKind::Pp) <= 1 {
            if let Some(dev) = gpus.devices.first_mut() {
                let _ = dev.bind_thread();
                self.mtp_free(dev);
            }
            return;
        }
        let owner = gpus.mesh_output_owner();
        match owner.and_then(|owner| gpus.devices.get_mut(owner)) {
            Some(dev) if dev.bind_thread().is_ok() => {
                // Only the PP branch also frees the row hidden carry.
                if let Some(hidden) = self.row_hidden.take() {
                    let _ = dev.free_tensor(hidden);
                }
                self.mtp_free(dev);
            }
            _ => std::mem::forget(self),
        }
    }

    fn k(&self) -> usize {
        self.k_ceiling().min(self.draft_rows_cap())
    }

    /// An armed n-gram pool may offer up to its `n_max` candidates (with the
    /// copy lane: take up to [`qwen4_ngram_take_cap`] candidates, never the
    /// longer proposal it observes); native windows still draft at most K.
    /// Every caller stays inside actual reserved verify storage. CPU-owned
    /// product qualification remains narrow; only the explicitly armed lab
    /// path may use a separately admitted and prewarmed wider window.
    fn proposal_capacity(&self) -> usize {
        if matches!(
            self.derived,
            Some(DerivedPhase::Unbound | DerivedPhase::Bound(_))
        ) {
            return self.derived_legal_cap(DerivedSource::Mtp);
        }
        let ceiling = self.k_ceiling();
        let capacity = if self.copy_lane {
            if self.ngram_active {
                ceiling.max(self.ngram_take_cap())
            } else {
                ceiling
            }
        } else {
            match (self.ngram_active, self.ngram_config) {
                (true, Some(config)) => ceiling.max(config.n_max),
                _ => ceiling,
            }
        };
        capacity.min(self.draft_rows_cap())
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn requires_greedy(&self) -> bool {
        !self.sampled_enabled
    }

    /// With `speculation.mtp_sampled`, a sampled request verifies by
    /// speculative rejection sampling, or naive sampling under
    /// `HIPFIRE_MTP_SAMPLED_MODE=naive`; greedy requests keep the argmax match.
    fn configure_request(&mut self, cfg: SpecRequestConfig) {
        self.request = cfg;
        self.emitted_seen = None;
        self.head_lag.clear();
        self.retained = None;
        self.floor_pending = None;
        self.floor_enabled =
            hipfire_config::developer_var("HIPFIRE_MTP_AR_FLOOR").as_deref() != Ok("0");
        self.floor.clear_call_log();
        self.floor.reset_request(
            singleton_floor_min_windows(),
            self.agreement_decay,
            singleton_floor_ar_probe_tokens(),
        );
        if floor_call_trace_enabled() {
            self.floor
                .set_call_log_enabled(FLOOR_CALL_PREFIX, "drafter", self.floor_enabled);
        }
        if let Some(device) = self.sampled.as_mut().and_then(|state| state.device.take()) {
            self.device_spare.push(device);
        }
        let sampled = self.sampled_enabled && cfg.temp.is_finite() && cfg.temp > 1.0e-6;
        self.sampled = match (sampled, &self.sampled_mode) {
            (true, Ok(mode)) => Some(SampledVerify::new(cfg, self.max_k, *mode)),
            _ => None,
        };
        // N-gram takeovers: greedy and sampled, thinking on or off. The pool
        // allocation is reused; its history restarts with this request.
        self.stats = MtpRequestStats::default();
        self.ngram_active = false;
        self.ngram_yield = (0.0, 0.0);
        self.ngram_agreement.fill(AGREEMENT_PRIOR);
        self.ngram_proposal.clear();
        self.ngram_proposed_at = None;
        self.ngram_last_first_accepted = false;
        self.ngram_decision = None;
        if let Some(ctx) = self.ngram.as_mut() {
            ctx.reset_request();
        }
        if let Some(ctx) = self.ngram_copy.as_mut() {
            ctx.reset_request();
        }
        if let (true, Some(config)) = (cfg.allow_ngram_modifier, self.ngram_config) {
            if self.copy_lane {
                if self
                    .ngram_copy
                    .as_ref()
                    .is_none_or(|ctx| ctx.config() != &config)
                {
                    self.ngram_copy = MtpCopyContext::new(config).ok();
                }
                self.ngram_active = self.ngram_copy.is_some();
            } else {
                if self
                    .ngram
                    .as_ref()
                    .is_none_or(|ctx| ctx.config() != &config)
                {
                    self.ngram = MtpNgramContext::new(config).ok();
                }
                self.ngram_active = self.ngram.is_some();
            }
        }
        self.stats.mtp_ngram = self.ngram_active;
        self.trace_request(&cfg);
        let derived =
            hipfire_config::developer_var("HIPFIRE_QWEN4_MTP_DERIVED_K").as_deref() == Ok("1");
        if let Some(scratch) = self.derived_margins.as_mut() {
            scratch.clear();
        }
        self.derived_configure(&cfg, derived);
        self.refresh_chooser_capacity();
        self.push_replay(ReplayOp::Configure(cfg));
    }

    fn supports_temp_verify(&self) -> bool {
        self.sampled_enabled
    }

    fn request_stats(&self) -> MtpRequestStats {
        let mut stats = self.stats;
        stats.ngram_mod_accept_rate = if stats.ngram_mod_drafts > 0 {
            let rate = stats.ngram_mod_accepted as f64 / stats.ngram_mod_drafts as f64;
            (rate * 1000.0).round() / 1000.0
        } else {
            0.0
        };
        stats
    }
}

/// Build the generic runtime adapter around the native Qwen4 GPU MTP core.
/// `end_of_turn` is the tokenizer's `<|im_end|>` id, when it has one;
/// `think_close` is the tokenizer's `</think>` id, when it has one;
/// `ngram` is the n-gram pool configuration requests may arm, and
/// `copy_lane` runs that pool as the copy lane
/// (`hipfire_config::mtp_ngram_copy_enabled_for_arch`); `gdn` is the loaded
/// GDN state format, which gates sampled verification
/// ([`qwen4_mtp_sampled_allowed`]).
pub fn build_qwen4_mtp_speculator(
    max_k: usize,
    explicit_mtp_k: Option<usize>,
    ctx_capacity: usize,
    end_of_turn: Option<u32>,
    think_close: Option<u32>,
    ngram: Option<NgramModConfig>,
    copy_lane: bool,
    gdn: crate::GdnStateFormat,
    measured_load_identity: Option<
        std::sync::Arc<crate::measured_identity::Qwen4MeasuredLoadIdentity>,
    >,
) -> Box<dyn Speculator> {
    let mut drafter = Qwen4MtpDrafter::new(max_k, ctx_capacity, end_of_turn)
        .with_explicit_mtp_k_override(explicit_mtp_k)
        .with_ngram(ngram)
        .with_copy_lane(copy_lane)
        .with_sampled_allowed(qwen4_mtp_sampled_allowed(gdn))
        .with_measured_load_identity(measured_load_identity);
    drafter.think_close = think_close;
    Box::new(MtpSpeculator::new(drafter))
}

/// Whether a load with GDN state format `gdn` may verify sampled requests
/// (temperature > 0) through native MTP. Under `q8_ef` they run AR instead
/// until the sampled-MTP law (G5) is verified on a GPU; greedy native MTP is
/// unchanged. Every other format keeps `speculation.mtp_sampled`. The match
/// is exhaustive so a new format forces a decision here.
pub fn qwen4_mtp_sampled_allowed(gdn: crate::GdnStateFormat) -> bool {
    match gdn {
        crate::GdnStateFormat::Q8Ef => false,
        crate::GdnStateFormat::F32 | crate::GdnStateFormat::Q8 => true,
    }
}

/// The n-gram pool configuration for a `(n_match, n_min, n_max)` triple
/// (`hipfire_config::ngram_mod_triple_for_arch`). Without the copy lane a
/// takeover verifies `n + 1` rows, so `n_max` is capped one below the spec
/// verify capacity (and `n_min` with it); with it `n_max` is the proposal
/// length under the same cap and [`qwen4_ngram_take_cap`] is the shorter
/// take. A triple the pool cannot run is an error, never a silently
/// different policy.
pub fn qwen4_ngram_mod_config(
    (n_match, n_min, n_max): (usize, usize, usize),
) -> Result<NgramModConfig, String> {
    if n_match == 0 || n_max == 0 || n_max > 64 || n_min > n_max {
        return Err(format!(
            "invalid n-gram triple n_match={n_match} n_min={n_min} n_max={n_max} (need 1 <= n_match, n_min <= n_max <= 64)"
        ));
    }
    let n_max = n_max.min(crate::gpu_forward::QWEN4_SPEC_VERIFY_ROWS - 1);
    Ok(NgramModConfig {
        n_match,
        n_min: n_min.min(n_max),
        n_max,
        ..NgramModConfig::default()
    })
}

/// Candidates a copy-lane takeover takes at most for `config`: its `n_max`, and never
/// more rows (the seed included) than a verify forward runs bitwise like
/// the single-row decode (`row_exact_rows`,
/// `rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows`).
pub fn qwen4_ngram_take_cap(config: &NgramModConfig, row_exact_rows: usize) -> usize {
    config.n_max.min(row_exact_rows.saturating_sub(1))
}

/// Default speculative verify rows before any wider CPU-owned storage was
/// admitted and prewarmed. The bundle's physical cap additionally reads that
/// actual reservation; this default never claims wider qualification.
/// An unarmed GPU load has no CPU storage restriction.
pub(crate) const fn spec_verify_rows_cap(cpu_splice: bool) -> usize {
    if cpu_splice {
        crate::program::QWEN4_CPU_SPLICE_MAX_ROWS
    } else {
        usize::MAX
    }
}

/// `Ok` while a `rows`-row speculative forward fits `cap`
/// ([`spec_verify_rows_cap`]); otherwise the CPU splice refusal, to be
/// returned before the forward's first state mutation.
pub(crate) fn check_spec_verify_rows(cap: usize, rows: usize, what: &str) -> Result<(), String> {
    if rows <= cap {
        return Ok(());
    }
    Err(format!(
        "{what} of {rows} rows exceeds the {cap}-row CPU routed-expert window; \
         wider verify would run the GPU grouped read, not the CPU decode arithmetic \
         (set memory.offload_exec=pcie or cap the draft length at K<={})",
        cap.saturating_sub(1)
    ))
}

/// Rows the widest target forward of a planned window verifies: the seed plus
/// the native drafts (`depth`; the interleaved route and the AR routes are
/// single-row steps) or, on an n-gram takeover, the seed plus its candidates.
fn window_verify_rows(route: WindowRoute, ngram_hit: bool, candidates: usize) -> usize {
    let native = match route {
        WindowRoute::Native { depth } => depth + 1,
        WindowRoute::ArRetired | WindowRoute::ArRouted | WindowRoute::ArCalibrate => 1,
    };
    if ngram_hit {
        native.max(candidates + 1)
    } else {
        native
    }
}

/// The most drafts any window of a native MTP request on `gpu_arch` verifies
/// (the seed row excluded): the head's `max_k` (at most [`DERIVED_MAX_DRAFTS`],
/// the physical executor width), or, with the n-gram copy lane on
/// (`hipfire_config::mtp_ngram_copy_enabled_for_arch`) where the n-gram
/// source arms (`hipfire_config::mtp_ngram_enabled_for_arch`), the copy
/// lane's take cap ([`qwen4_ngram_take_cap`] at the arch's row-exact verify
/// rows). Admission sizes the verify rows and the GDN row capture by it: on
/// gfx1201 (K=3) with the copy lane on, an armed takeover verifies 8 rows
/// where native windows verify 4; without it admission sizes the native
/// windows alone. Sizing only: this never widens the incumbent chooser's
/// execution depths or prunes.
pub fn native_mtp_proposal_drafts(max_k: usize, gpu_arch: &str) -> usize {
    if hipfire_config::developer_var("HIPFIRE_QWEN4_MTP_DERIVED_K").as_deref() == Ok("1") {
        return DERIVED_MAX_DRAFTS;
    }
    if !(hipfire_config::mtp_ngram_enabled_for_arch(crate::ARCH_ID, gpu_arch)
        && hipfire_config::mtp_ngram_copy_enabled_for_arch(crate::ARCH_ID, gpu_arch))
    {
        // Admission reserves the requested executor extent before the load.
        // Runtime agreement and cost evidence follow its admitted capacity;
        // the finite historical profile does not impose a tuning ceiling.
        return max_k.min(DERIVED_MAX_DRAFTS);
    }
    qwen4_ngram_mod_config(hipfire_config::ngram_mod_triple_for_arch(
        crate::ARCH_ID,
        gpu_arch,
    ))
    .map_or(max_k, |config| {
        max_k.max(qwen4_ngram_take_cap(
            &config,
            rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows(gpu_arch),
        ))
    })
}

/// `HIPFIRE_QWEN4_MTP_BATCHED_FILL`: the batched prompt-fill Append pass is on
/// unless set to `0`; read at every `mtp_prefill`.
pub(crate) fn mtp_batched_fill_enabled() -> bool {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_MTP_BATCHED_FILL", true)
}

/// Whether this GPU's prompt fill wants the Append scratch: batched fill on,
/// or the DET prompt fill in force (`HIPFIRE_QWEN4_MTP_APPEND_DET`, default on; exact
/// gfx1151, which uses the scratch even with batched fill off). The head's
/// own support ([`Qwen4Bundle::mtp_append_rows_supported`]) still applies.
pub(crate) fn mtp_fill_scratch_wanted(gpu: &Gpu) -> bool {
    mtp_batched_fill_enabled() || mtp_append_det_enabled(gpu)
}

/// [`mtp_fill_scratch_wanted`] from the environment alone, for admission
/// arithmetic that has no GPU: an upper bound on every arch.
pub(crate) fn mtp_fill_scratch_requested() -> bool {
    mtp_batched_fill_enabled() || mtp_append_det_requested()
}

/// Rows of one prompt-fill Append pass: `capacity` (the scratch rows) with
/// batched fill on; with it off (the scratch is then only present for the DET
/// prompt fill) one row at a time, which the DET kernels make byte-equal to
/// the batched passes.
fn prompt_fill_sub_rows(batched: bool, capacity: usize) -> usize {
    if batched {
        capacity
    } else {
        1
    }
}

/// `HIPFIRE_QWEN4_MTP_REUSE_PREFILL_IDS`: the prompt-fill head Append embeds
/// from the token ids the trunk forward of the same chunk already uploaded
/// (checked against the chunk's tokens), instead of uploading them again,
/// only when set to `1` (default off: the warm path was not output-identical);
/// read at every `mtp_prefill`.
pub(crate) fn mtp_reuse_prefill_ids_enabled() -> bool {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_MTP_REUSE_PREFILL_IDS", false)
}

/// `Qwen4Bundle::mtp_append_rows`'s `trunk_ids_row` for a prompt-fill
/// sub-chunk starting at chunk row `off`: the sub-chunk's rows are that row
/// of the chunk's trunk upload when reuse is on.
fn prefill_trunk_ids_row(reuse: bool, off: usize) -> Option<usize> {
    reuse.then_some(off)
}

/// Whether a prompt-fill chunk of `chunk_rows` rows uploads its token ids once
/// into the `capacity`-row prompt id buffer for its sub-chunks to embed from:
/// not with the trunk-id reuse lever on (it supplies its own ids), not for a
/// lone row (the per-row Append uploads its own), and not past the buffer.
fn prefill_uploads_chunk_ids(reuse: bool, chunk_rows: usize, capacity: usize) -> bool {
    !reuse && (2..=capacity).contains(&chunk_rows)
}

/// `HIPFIRE_QWEN4_MTP_SKIP_INTERMEDIATE_PICK`: a non-final `mtp_prefill` chunk
/// skips the LM head, argmax and readback unless set to `0`; read at every
/// `mtp_prefill`.
pub(crate) fn mtp_skip_intermediate_pick_enabled() -> bool {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_MTP_SKIP_INTERMEDIATE_PICK", true)
}

/// Whether the `mtp_prefill` chunk ending at `end` (an exclusive prompt
/// position, whether it ends at a capture boundary or at the natural chunk
/// end) runs the LM head and reads its argmax back. Only the prompt's last
/// chunk's pick is read (the seed, and the last row's logits for a sampled
/// draw), so with the skip on every chunk ending before the prompt end is
/// silent; the skip off keeps the pick on every chunk.
pub(crate) fn mtp_prefill_chunk_needs_pick(
    end: usize,
    prompt_len: usize,
    skip_intermediate: bool,
) -> bool {
    !skip_intermediate || end == prompt_len
}
/// Whether this GPU's GDN route captures verify rows: row capture rides the
/// few-row persistent GDN recurrence.
pub fn native_mtp_row_capture(gpu: &Gpu, config: &crate::Qwen4Config) -> bool {
    gpu.arch_caps.has_gfx11_plus_simt()
        && config.linear_key_head_dim == 128
        && config.linear_value_head_dim == 128
        && config.linear_conv_kernel_dim == 4
}

/// Rows of the native MTP GDN capture ring on this GPU for a window of
/// `window_rows` rows: the wider of the window and the arch's row-exact verify
/// rows, so verify blocks up to the row-exact limit keep capture rollback.
pub fn native_mtp_capture_rows(gpu: &Gpu, window_rows: usize) -> usize {
    window_rows.max(crate::gpu_forward::qwen4_row_exact_verify_rows(gpu))
}

/// The `row_capture` argument of [`native_mtp_device_bytes`] /
/// [`native_mtp_request_bytes`] for this GPU: the GDN state format and the
/// ring rows ([`native_mtp_capture_rows`] of a `max_k`-draft window) where
/// [`native_mtp_row_capture`], else `None`.
pub fn native_mtp_capture(
    gpu: &Gpu,
    config: &crate::Qwen4Config,
    gdn: crate::GdnStateFormat,
    max_k: usize,
) -> Option<(crate::GdnStateFormat, usize)> {
    native_mtp_row_capture(gpu, config).then(|| {
        (
            gdn,
            native_mtp_capture_rows(gpu, native_mtp_window_rows(max_k)),
        )
    })
}
/// Public byte bridge for the loader's resource snapshot: the device bytes of
/// the ACTUAL capture ring this config/GDN owns at `rows` rows over
/// `owned_gdn_layers` GDN layers, via the one
/// [`crate::state::Qwen4State::row_capture_bytes_scoped`] formula (never a
/// parallel memory planner). Full single-device layer count, as the settled
/// state uses; PP/EP bands stay absent upstream. `None` on arithmetic
/// overflow or an unsupported layer predicate.
pub fn qwen4_capture_bytes(
    config: &crate::Qwen4Config,
    gdn: crate::GdnStateFormat,
    rows: usize,
    owned_gdn_layers: usize,
) -> Option<u64> {
    if owned_gdn_layers == 0 || owned_gdn_layers != config.n_linear_layers() {
        return None;
    }
    crate::state::Qwen4State::row_capture_bytes_scoped(config, gdn, rows, owned_gdn_layers)
        .and_then(|bytes| u64::try_from(bytes).ok())
}

/// Whether a native MTP request on this GPU allocates the prompt-fill Append
/// scratch (`MtpAppendScratch`): exact gfx1151 or gfx1201 with
/// `HIPFIRE_QWEN4_MTP_BATCHED_FILL` on, or exact gfx1151 with
/// `HIPFIRE_QWEN4_MTP_APPEND_DET` on (the default; the DET prompt fill needs the scratch
/// even with batched fill off).
pub fn native_mtp_fill_scratch(gpu: &Gpu) -> bool {
    (gpu.arch_caps.is_gfx1151() || gpu.arch_caps.is_gfx1201()) && mtp_fill_scratch_wanted(gpu)
}

/// Device bytes native MTP adds to a load whose QSA context storage is
/// `context` (its admitted `max_seq` and committed part) with a
/// `chunk_rows` prefill chunk, windows of up to `max_k` drafts (pass
/// [`native_mtp_proposal_drafts`], which counts n-gram takeovers) and the
/// language head stored as `head_dtype`: what the attached head
/// (`Qwen4MtpGpu`) commits, plus the larger of its build scratch (released
/// at attach) and what the first speculative request allocates after it —
/// the verify hidden rows (`max_k + 1` rows or one forward chunk, whichever
/// is larger), the pending, row and retained-pending hidden carries, with `row_capture`
/// (the GDN state format and ring rows, [`native_mtp_capture`]) the GDN
/// capture ring in that format, and with `batched_fill`
/// (where [`native_mtp_fill_scratch`]) the prompt-fill Append scratch.
pub fn native_mtp_device_bytes(
    config: &crate::Qwen4Config,
    context: &crate::Qwen4ContextCommit,
    chunk_rows: usize,
    max_k: usize,
    head_dtype: rdna_compute::DType,
    row_capture: Option<(crate::GdnStateFormat, usize)>,
    batched_fill: bool,
) -> Option<u64> {
    let request = native_mtp_request_bytes(
        config,
        context.max_seq,
        chunk_rows,
        max_k,
        row_capture,
        batched_fill,
    )?;
    let (resident, scratch) =
        crate::mtp_gpu::Qwen4MtpGpu::device_bytes(config, context, head_dtype, false)?;
    u64::try_from(resident)
        .ok()?
        .checked_add(u64::try_from(scratch).ok()?.max(request))
}

/// Rows a native MTP request sizes for windows of up to `max_k` drafts (pass
/// [`native_mtp_proposal_drafts`], which counts n-gram takeovers): the seed
/// row plus `max_k` clamped to the widest verify forward
/// (`QWEN4_SPEC_VERIFY_ROWS - 1` drafts). The verify hidden rows and the GDN
/// row capture of every band are sized by it.
pub(crate) fn native_mtp_window_rows(max_k: usize) -> usize {
    max_k.clamp(1, QWEN4_SPEC_VERIFY_ROWS - 1) + 1
}

/// Device bytes the first speculative request of a native MTP load of
/// `max_seq` tokens allocates once and keeps (after the head attached): the
/// verify hidden rows (`max_k + 1` rows or one `chunk_rows` forward chunk,
/// whichever is larger; `on` reserves two chunks), the pending, row and retained-pending hidden
/// carries, with
/// `row_capture` (format, ring rows) the GDN capture ring, and with
/// `batched_fill` the batched prompt-fill scratch and one chunk of prompt ids.
pub fn native_mtp_request_bytes(
    config: &crate::Qwen4Config,
    max_seq: usize,
    chunk_rows: usize,
    max_k: usize,
    row_capture: Option<(crate::GdnStateFormat, usize)>,
    batched_fill: bool,
) -> Option<u64> {
    native_mtp_request_allocation_sizes(
        config,
        max_seq,
        chunk_rows,
        max_k,
        row_capture,
        batched_fill,
    )?
    .into_iter()
    .try_fold(0u64, |total, bytes| total.checked_add(bytes))
}

/// [`native_mtp_request_bytes`] per allocation: the verify hidden rows, the
/// pending, row and retained-pending hidden carries (one `hc * hidden` F32
/// row each), the GDN capture ring's allocations
/// ([`crate::state::Qwen4State::row_capture_allocation_sizes_scoped`]) and
/// the batched prompt-fill scratch's
/// ([`MtpAppendScratch::allocation_sizes`]). Zero-byte entries are omitted;
/// `None` on overflow.
pub(crate) fn native_mtp_request_allocation_sizes(
    config: &crate::Qwen4Config,
    max_seq: usize,
    chunk_rows: usize,
    max_k: usize,
    row_capture: Option<(crate::GdnStateFormat, usize)>,
    batched_fill: bool,
) -> Option<Vec<u64>> {
    let rows = native_mtp_window_rows(max_k);
    let hidden_row = config
        .hc_count
        .checked_mul(config.hidden_size)?
        .checked_mul(std::mem::size_of::<f32>())?;
    let verify_rows = crate::bundle::spec_hidden_capacity_rows(
        rows.max(max_seq.min(chunk_rows)),
        chunk_rows,
        crate::npu_trunk::NpuTrunkMode::from_env().ok() == Some(crate::npu_trunk::NpuTrunkMode::On),
    )?;
    let mut sizes = vec![
        u64::try_from(verify_rows.checked_mul(hidden_row)?).ok()?,
        u64::try_from(hidden_row).ok()?,
        u64::try_from(hidden_row).ok()?,
        u64::try_from(hidden_row).ok()?,
    ];
    if let Some((gdn, ring_rows)) = row_capture {
        sizes.extend(
            crate::state::Qwen4State::row_capture_allocation_sizes_scoped(
                config,
                gdn,
                ring_rows.max(rows),
                config.n_linear_layers(),
            )?,
        );
    }
    if batched_fill {
        sizes.extend(MtpAppendScratch::allocation_sizes(
            config,
            MTP_FILL_ROWS.min(chunk_rows).max(1),
            chunk_rows.max(1),
        )?);
    }
    sizes.retain(|&bytes| bytes > 0);
    Some(sizes)
}

#[cfg(any(test, feature = "reference-parity"))]
/// Convert an MTP request-state error into the erased runtime error type.
pub fn mtp_error(error: MtpError) -> String {
    format!("Qwen4 native MTP: {error}")
}

#[cfg(test)]
mod adapter_recovery_tests;
#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::spec_economics::{
        batched_depth, replay_call_log, CapStats, ReplayReport, FLOOR_PROBE_WINDOWS,
    };

    fn mtp_contexts() -> Vec<crate::Qwen4ContextCommit> {
        use crate::kv_backend::Qwen4KvBackend;
        let granularity = 2 << 20;
        vec![
            crate::Qwen4ContextCommit::legacy(4096),
            crate::Qwen4ContextCommit::legacy(262_144),
            crate::Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 8193, 1025, granularity),
            crate::Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 262_144, 1536, granularity),
            crate::Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 262_144, 8192, granularity),
        ]
    }

    fn sum(sizes: &[u64]) -> u64 {
        sizes.iter().sum()
    }

    /// The per-allocation request inventory is the byte query's own plan:
    /// the verify hidden rows, three hidden carries, the capture ring's
    /// allocations and the fill scratch's, summing to
    /// `native_mtp_request_bytes`, with overflow refused alike.
    #[test]
    fn native_mtp_request_allocation_sizes_sum_to_the_request_bytes() {
        use crate::GdnStateFormat as G;
        let config = crate::config::compact_test_config();
        for chunk_rows in [1usize, 4, 512, 768, 1536, 8192] {
            for drafts in [1usize, 3, 7, 63] {
                for capture in [
                    None,
                    Some((G::Q8, 2)),
                    Some((G::Q8Ef, 8)),
                    Some((G::F32, hipfire_config::QWEN4_VERIFY_MAX_ROWS)),
                ] {
                    for fill in [false, true] {
                        let sizes = native_mtp_request_allocation_sizes(
                            &config, 4096, chunk_rows, drafts, capture, fill,
                        )
                        .expect("request inventory");
                        assert_eq!(
                            Some(sum(&sizes)),
                            native_mtp_request_bytes(
                                &config, 4096, chunk_rows, drafts, capture, fill
                            ),
                            "{chunk_rows} {drafts} {capture:?} {fill}"
                        );
                        assert!(sizes.iter().all(|&bytes| bytes > 0));
                        let carry = (config.hc_count * config.hidden_size * 4) as u64;
                        assert_eq!(sizes[1..4], [carry; 3]);
                        let ring = capture.map_or(0, |(gdn, rows)| {
                            crate::state::Qwen4State::row_capture_allocation_sizes_scoped(
                                &config,
                                gdn,
                                rows.max(native_mtp_window_rows(drafts)),
                                config.n_linear_layers(),
                            )
                            .unwrap()
                            .len()
                        });
                        let append = if fill {
                            MtpAppendScratch::allocation_sizes(
                                &config,
                                MTP_FILL_ROWS.min(chunk_rows).max(1),
                                chunk_rows.max(1),
                            )
                            .unwrap()
                            .len()
                        } else {
                            0
                        };
                        assert_eq!(sizes.len(), 4 + ring + append, "{capture:?} {fill}");
                    }
                }
            }
        }
        assert_eq!(
            native_mtp_request_allocation_sizes(&config, 4096, usize::MAX, 3, None, true),
            None
        );
        for rows in [usize::MAX, usize::MAX / 2, 1usize << 62] {
            assert_eq!(
                native_mtp_request_allocation_sizes(
                    &config,
                    4096,
                    1536,
                    3,
                    Some((crate::GdnStateFormat::Q8, rows)),
                    false
                ),
                None
            );
        }
    }

    /// The two production inventories compose to the byte query: the resident
    /// head allocations plus the larger (by sum) of the head's build scratch
    /// and the first request's allocations, the request on a tie, summing to
    /// `native_mtp_device_bytes` for every context commit, chunk, draft count,
    /// head dtype and capture.
    #[test]
    fn native_mtp_allocation_sizes_sum_to_the_device_bytes() {
        use crate::GdnStateFormat as G;
        use rdna_compute::DType;
        let config = crate::config::compact_test_config();
        let composed = |context: &crate::Qwen4ContextCommit,
                        chunk_rows: usize,
                        drafts: usize,
                        head: DType,
                        capture: Option<(G, usize)>,
                        fill: bool|
         -> Option<Vec<u64>> {
            let request = native_mtp_request_allocation_sizes(
                &config,
                context.max_seq,
                chunk_rows,
                drafts,
                capture,
                fill,
            )?;
            let (mut sizes, scratch) =
                crate::mtp_gpu::Qwen4MtpGpu::allocation_sizes(&config, context, head, false)?;
            sizes.extend(
                if scratch.iter().sum::<u64>() > request.iter().sum::<u64>() {
                    scratch
                } else {
                    request
                },
            );
            Some(sizes)
        };
        for context in mtp_contexts() {
            for chunk_rows in [1usize, 4, 512, 768, 1536] {
                for drafts in [1usize, 3, 63] {
                    for head in [DType::BF16, DType::Q8_0, DType::MQ6G256V2, DType::F32] {
                        for capture in [None, Some((G::Q8Ef, 8))] {
                            for fill in [false, true] {
                                let sizes =
                                    composed(&context, chunk_rows, drafts, head, capture, fill);
                                let bytes = native_mtp_device_bytes(
                                    &config, &context, chunk_rows, drafts, head, capture, fill,
                                );
                                assert_eq!(
                                    sizes.as_deref().map(sum),
                                    bytes,
                                    "{context:?} {chunk_rows} {drafts} {head:?} {capture:?} {fill}"
                                );
                                if let Some(sizes) = sizes {
                                    assert!(sizes.iter().all(|&bytes| bytes > 0));
                                }
                            }
                        }
                    }
                }
            }
        }
        let huge = crate::Qwen4ContextCommit::legacy(usize::MAX);
        assert_eq!(composed(&huge, 1536, 3, DType::BF16, None, false), None);
        assert_eq!(
            native_mtp_device_bytes(&config, &huge, 1536, 3, DType::BF16, None, false),
            None
        );
    }

    #[test]
    fn head_finish_single_card_keeps_the_seed_row_and_rebuilds_the_rest() {
        // A window that drafted keeps its draft chain's row 0 (the seed with
        // the pending hidden of `position - 1`) whatever the acceptance.
        assert_eq!(head_finish(3, false), HeadFinish::KeepSeedRow);
        assert_eq!(head_finish(1, false), HeadFinish::KeepSeedRow);
        // A zero-draft window has no row 0: restore and append the seed.
        assert_eq!(
            head_finish(0, false),
            HeadFinish::SeedFromStart { restore: true }
        );
    }

    #[test]
    fn head_finish_after_a_mesh_restore_appends_the_seed_without_restoring_again() {
        // The mesh replay route restored target and head together: the head is
        // at the window's start, so the seed's row is appended as it is.
        assert_eq!(
            head_finish(3, true),
            HeadFinish::SeedFromStart { restore: false }
        );
        assert_eq!(
            head_finish(1, true),
            HeadFinish::SeedFromStart { restore: false }
        );
    }

    #[test]
    fn mesh_tickets_are_never_accepted_as_local_ones() {
        let window = Qwen4PpWindow::for_test(7, 4, true, true);
        let target = TargetTicket::Mesh(window);
        assert!(target.local().is_err());
        assert!(target.mesh().is_ok());
        let head = HeadTicket::Mesh(window);
        assert!(head.local().is_err());
    }

    #[test]
    fn pp_serves_greedy_and_unpenalized_leviathan_requests_without_the_ngram_pool() {
        let sampled = SpecRequestConfig {
            temp: 0.8,
            top_p: 0.9,
            rng_seed: 7,
            ..SpecRequestConfig::default()
        };
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None);
        drafter.sampled_enabled = true;
        drafter.sampled_mode = Ok(SampledMode::Leviathan);
        assert!(drafter.refuse_unsupported_under_pp().is_ok());
        // A sampled Leviathan request with neutral penalties runs on the mesh:
        // every read of its verify goes through the output band's device.
        MtpDrafter::configure_request(&mut drafter, sampled);
        assert!(drafter.sampled.is_some());
        assert!(drafter.refuse_unsupported_under_pp().is_ok());
        // Penalties need the GPU prepass, which refuses the mesh.
        MtpDrafter::configure_request(
            &mut drafter,
            SpecRequestConfig {
                repeat_penalty: 1.1,
                repeat_window: 64,
                ..sampled
            },
        );
        let penalized = drafter.refuse_unsupported_under_pp().unwrap_err();
        assert!(penalized.contains("penalties"), "{penalized}");
        // Naive draws come from the process-wide sampler.
        drafter.sampled_mode = Ok(SampledMode::Naive);
        MtpDrafter::configure_request(&mut drafter, sampled);
        let naive = drafter.refuse_unsupported_under_pp().unwrap_err();
        assert!(naive.contains("naive"), "{naive}");
        // Never silently greedy: a nonzero temperature with sampled
        // verification off is refused (the typed entry bypasses the AR gate).
        drafter.sampled_mode = Ok(SampledMode::Leviathan);
        drafter.sampled_enabled = false;
        MtpDrafter::configure_request(&mut drafter, sampled);
        assert!(drafter.sampled.is_none());
        let disabled = drafter.refuse_unsupported_under_pp().unwrap_err();
        assert!(disabled.contains("greedy"), "{disabled}");
        // An ordinary greedy request is accepted with the flag on or off.
        MtpDrafter::configure_request(&mut drafter, SpecRequestConfig::default());
        assert!(drafter.refuse_unsupported_under_pp().is_ok());
        drafter.sampled_enabled = true;
        assert!(drafter.refuse_unsupported_under_pp().is_ok());
        // A non-finite temperature is never admitted.
        drafter.request.temp = f32::NAN;
        assert!(drafter.refuse_unsupported_under_pp().is_err());
        // An armed n-gram pool is a one-card route.
        drafter.request.temp = 0.0;
        drafter.ngram_active = true;
        let pool = drafter.refuse_unsupported_under_pp().unwrap_err();
        assert!(pool.contains("n-gram"), "{pool}");
    }
    use crate::reference_mtp::MtpQsaGeometry;
    use hipfire_runtime::spec_sampling::sample_residual;

    /// The `mtp_prefill` chunk ends for a fill of `prompt_len` rows from
    /// `start`, with natural chunks of `chunk_rows` and capture boundaries at
    /// `captures` (a chunk ends at the first boundary inside it), mirroring
    /// the loop's `next_prefix_capture` planning.
    fn prefill_chunk_ends(
        start: usize,
        prompt_len: usize,
        chunk_rows: usize,
        captures: &[usize],
    ) -> Vec<usize> {
        let mut ends = Vec::new();
        let mut pos = start;
        while pos < prompt_len {
            let natural = (pos + chunk_rows).min(prompt_len);
            let end = captures
                .iter()
                .copied()
                .find(|&boundary| boundary > pos && boundary <= natural)
                .unwrap_or(natural);
            ends.push(end);
            pos = end;
        }
        ends
    }

    fn picking_chunks(ends: &[usize], prompt_len: usize, skip: bool) -> Vec<bool> {
        ends.iter()
            .map(|&end| mtp_prefill_chunk_needs_pick(end, prompt_len, skip))
            .collect()
    }

    #[test]
    fn prefill_pick_only_on_the_final_chunk_when_skipping() {
        // One chunk is always the final one and keeps its pick.
        assert_eq!(picking_chunks(&[100], 100, true), vec![true]);
        // Natural split, no capture boundary: 100 rows in chunks of 40.
        let ends = prefill_chunk_ends(0, 100, 40, &[]);
        assert_eq!(ends, vec![40, 80, 100]);
        assert_eq!(picking_chunks(&ends, 100, true), vec![false, false, true]);
    }

    #[test]
    fn prefill_capture_splits_are_silent_unless_they_end_the_prompt() {
        // A turn anchor and a periodic boundary split the prompt mid-way: both
        // split chunks are non-final and skip the pick.
        let ends = prefill_chunk_ends(0, 100, 64, &[30, 90]);
        assert_eq!(ends, vec![30, 90, 100]);
        assert_eq!(picking_chunks(&ends, 100, true), vec![false, false, true]);
        // The end-of-prompt capture boundary closes the final chunk: it keeps
        // the full logits/readback for the seed.
        let ends = prefill_chunk_ends(0, 100, 64, &[64, 100]);
        assert_eq!(ends, vec![64, 100]);
        assert_eq!(picking_chunks(&ends, 100, true), vec![false, true]);
        // A split just before the end leaves a one-row final chunk that picks.
        let ends = prefill_chunk_ends(0, 100, 64, &[99]);
        assert_eq!(ends, vec![64, 99, 100]);
        assert_eq!(picking_chunks(&ends, 100, true), vec![false, false, true]);
    }

    #[test]
    fn prefill_cache_hit_suffix_keeps_the_final_pick() {
        // A prefix-cache hit fills only `start..prompt_len`; the decision keys
        // on the absolute prompt end, so a one-chunk suffix still picks.
        let ends = prefill_chunk_ends(8192, 8300, 8192, &[]);
        assert_eq!(ends, vec![8300]);
        assert_eq!(picking_chunks(&ends, 8300, true), vec![true]);
        let ends = prefill_chunk_ends(8192, 8300, 64, &[8200]);
        assert_eq!(ends, vec![8200, 8264, 8300]);
        assert_eq!(picking_chunks(&ends, 8300, true), vec![false, false, true]);
    }

    #[test]
    fn prefill_opt_out_picks_every_chunk() {
        let ends = prefill_chunk_ends(0, 100, 64, &[30, 90]);
        assert_eq!(picking_chunks(&ends, 100, false), vec![true; ends.len()]);
    }

    #[test]
    fn prefill_head_append_names_the_trunk_id_row_only_when_reuse_is_on() {
        assert_eq!(prefill_trunk_ids_row(true, 0), Some(0));
        assert_eq!(prefill_trunk_ids_row(true, 1024), Some(1024));
        assert_eq!(prefill_trunk_ids_row(false, 0), None);
        assert_eq!(prefill_trunk_ids_row(false, 1024), None);
    }

    #[test]
    fn prefill_head_append_uploads_the_chunk_ids_once_unless_reuse_or_oversized() {
        // A chunk of two rows or more has a multirow sub-chunk to feed.
        assert!(prefill_uploads_chunk_ids(false, 8192, 8192));
        assert!(prefill_uploads_chunk_ids(false, 1025, 8192));
        assert!(prefill_uploads_chunk_ids(false, 2, 8192));
        // A lone row is the per-row Append, which never reads the upload (under
        // DET it takes the rows route with the scratch's own upload).
        assert!(!prefill_uploads_chunk_ids(false, 1, 8192));
        // The reuse lever keeps its own ids.
        assert!(!prefill_uploads_chunk_ids(true, 8192, 8192));
        // A chunk past the buffer's capacity keeps the per-sub-chunk upload.
        assert!(!prefill_uploads_chunk_ids(false, 8193, 8192));
    }

    /// The sub-chunk sizes `mtp_prefill_chunk` walks for `chunk_rows`.
    fn prompt_fill_passes(chunk_rows: usize, batched: bool, capacity: usize) -> Vec<usize> {
        let sub_rows = prompt_fill_sub_rows(batched, capacity);
        let mut passes = Vec::new();
        let mut off = 0;
        while off < chunk_rows {
            let n = (chunk_rows - off).min(sub_rows);
            passes.push(n);
            off += n;
        }
        passes
    }

    #[test]
    fn prompt_fill_sub_chunks_are_scratch_rows_batched_and_single_rows_otherwise() {
        assert_eq!(prompt_fill_passes(2500, true, 1024), [1024, 1024, 452]);
        assert_eq!(prompt_fill_passes(1, true, 1024), [1]);
        assert_eq!(prompt_fill_passes(1025, true, 1024), [1024, 1]);
        // Batched fill off (scratch present only for the DET prompt fill):
        // one row at a time.
        assert_eq!(prompt_fill_passes(3, false, 1024), [1, 1, 1]);
        // Every pass covers the chunk exactly once, in order.
        for batched in [true, false] {
            for rows in [1usize, 2, 7, 1023, 1024, 1025, 4097] {
                assert_eq!(
                    prompt_fill_passes(rows, batched, 1024)
                        .iter()
                        .sum::<usize>(),
                    rows
                );
            }
        }
    }

    #[test]
    fn det_prompt_fill_sends_every_pass_through_rows_and_default_keeps_the_lone_row_per_row() {
        // DET: a trailing lone row (1025 rows) and the row-at-a-time passes
        // all take the rows route.
        for batched in [true, false] {
            assert!(prompt_fill_passes(1025, batched, 1024)
                .into_iter()
                .all(|n| crate::mtp_gpu::prompt_fill_uses_rows(true, n)));
        }
        // Default (DET off, batched on): only the lone trailing row stays per-row.
        let uses_rows: Vec<bool> = prompt_fill_passes(1025, true, 1024)
            .into_iter()
            .map(|n| crate::mtp_gpu::prompt_fill_uses_rows(false, n))
            .collect();
        assert_eq!(uses_rows, [true, false]);
    }

    #[test]
    fn native_acceptance_lowers_to_the_runtime_window_and_step() {
        let result = accept_native_greedy(&[10, 11], &[10, 99, 100], None).unwrap();
        assert_eq!(result.committed, vec![10, 99]);
        assert_eq!(result.accepted, 1);
        assert!(!result.hit_eos);
        assert_eq!(target_commit_accept_len(&result), 1);

        let step = window_to_spec_step(MtpWindow {
            committed: result.committed,
            accepted: result.accepted,
            drafts_generated: 2,
        })
        .unwrap();
        assert_eq!(step.emit.as_slice(), &[10, 99]);
        assert_eq!(step.next_seed, 99);
        assert_eq!(step.proposed, 2);
        assert_eq!(step.accepted, 1);
    }

    #[test]
    fn native_zero_partial_full_and_eos_counts_are_explicit() {
        let zero = accept_native_greedy(&[], &[42], None).unwrap();
        assert_eq!(zero.committed, vec![42]);
        assert_eq!(target_commit_accept_len(&zero), 0);

        let partial = accept_native_greedy(&[10, 11], &[10, 12, 100], None).unwrap();
        assert_eq!(target_commit_accept_len(&partial), 1);

        let full = accept_native_greedy(&[10, 11], &[10, 11, 12], None).unwrap();
        assert_eq!(full.committed, vec![10, 11, 12]);
        assert_eq!(target_commit_accept_len(&full), 2);

        let eos = accept_native_greedy(&[10, 99], &[10, 99, 12], Some(99)).unwrap();
        assert_eq!(eos.committed, vec![10, 99]);
        assert_eq!(eos.accepted, 2);
        assert!(eos.hit_eos);

        assert!(require_native_greedy(-0.0).is_ok());
        assert!(require_native_greedy(1.0e-6).is_ok());
        assert!(require_native_greedy(1.0e-5).is_err());
        assert!(require_native_greedy(f32::INFINITY).is_err());
        assert!(require_native_greedy(f32::NEG_INFINITY).is_err());
        assert!(require_native_greedy(f32::NAN).is_err());
    }

    #[test]
    fn accepted_eos_stays_pending_for_terminal_flush() {
        let accepted = accept_native_greedy(&[10, 99], &[10, 99, 12], Some(99)).unwrap();
        assert_eq!(target_commit_accept_len(&accepted), 1);
        assert_eq!(
            committed_target_positions(7, target_commit_accept_len(&accepted)).unwrap(),
            vec![7]
        );
        assert_eq!(
            native_commit_position(7, target_commit_accept_len(&accepted)).unwrap(),
            8
        );

        let bonus = accept_native_greedy(&[10, 11], &[10, 99, 12], Some(99)).unwrap();
        assert!(bonus.hit_eos);
        assert_eq!(target_commit_accept_len(&bonus), 1);
    }

    #[test]
    fn zero_draft_replays_seed_before_terminal_flush() {
        let zero = accept_native_greedy(&[], &[42], Some(99)).unwrap();
        assert_eq!(target_commit_accept_len(&zero), 0);
        assert_eq!(
            committed_target_positions(7, target_commit_accept_len(&zero) + 1).unwrap(),
            vec![7]
        );
        assert_eq!(
            native_commit_position(7, target_commit_accept_len(&zero) + 1).unwrap(),
            8
        );
    }

    #[test]
    fn terminal_repair_replays_the_seed_and_all_but_the_last_consumed_token() {
        // A window at position 10 that kept the seed and three tokens.
        let span = WindowSpan {
            start: 10,
            end: 14,
            seed: 7,
        };
        let replay = |consumed: &[u32]| {
            terminal_repair_replay(span, 10, 7, consumed).map(|replay| replay.to_vec())
        };
        // The last consumed token stays pending for the terminal flush.
        assert_eq!(replay(&[20]), Some(vec![7]));
        assert_eq!(replay(&[20, 21]), Some(vec![7, 20]));
        assert_eq!(replay(&[20, 21, 22]), Some(vec![7, 20, 21]));
        // Consuming a whole window with an accepted end-of-turn left pending
        // replays exactly the rows the window committed.
        assert_eq!(replay(&[20, 21, 22, 23]), Some(vec![7, 20, 21, 22]));
    }

    #[test]
    fn terminal_repair_declines_what_it_cannot_restore() {
        let span = WindowSpan {
            start: 10,
            end: 14,
            seed: 7,
        };
        // Nothing consumed: there is nothing to replay.
        assert!(terminal_repair_replay(span, 10, 7, &[]).is_none());
        // Another window (position or seed).
        assert!(terminal_repair_replay(span, 11, 7, &[20]).is_none());
        assert!(terminal_repair_replay(span, 10, 8, &[20]).is_none());
        // More consumed than the window committed.
        assert!(terminal_repair_replay(span, 10, 7, &[20, 21, 22, 23, 24]).is_none());
    }

    #[test]
    fn stop_sequences_is_off_by_default_and_replays_to_peers() {
        assert!(!SpecRequestConfig::default().stop_sequences);
        let cfg = SpecRequestConfig {
            stop_sequences: true,
            ..SpecRequestConfig::default()
        };
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None);
        MtpDrafter::configure_request(&mut drafter, cfg);
        assert!(drafter.request.stop_sequences);
        assert_eq!(drafter.pending_replay(), &[ReplayOp::Configure(cfg)]);
        // A new request without stop sequences drops the flag and any window
        // kept for the previous one.
        MtpDrafter::configure_request(&mut drafter, SpecRequestConfig::default());
        assert!(!drafter.request.stop_sequences);
        assert!(drafter.retained.is_none());
    }

    #[test]
    fn native_zero_one_all_acceptance_advances_mtp_state_in_lockstep() {
        let cases = [
            (vec![], vec![42], 0usize),
            (vec![10], vec![10, 42], 1usize),
            (vec![10, 11], vec![10, 11, 42], 2usize),
        ];
        for (drafts, target_picks, expected_accept_len) in cases {
            let acceptance = accept_native_greedy(&drafts, &target_picks, None).unwrap();
            assert_eq!(target_commit_accept_len(&acceptance), expected_accept_len);
            let consumed = target_commit_accept_len(&acceptance) + 1;
            let expected_position = native_commit_position(7, consumed).unwrap();
            let mut state = Qwen4MtpState::new(MtpQsaGeometry {
                q_heads: 2,
                kv_heads: 1,
                head_dim: 2,
                index_heads: 1,
                index_dim: 2,
                compress_ratio: 2,
                budget: 4,
                max_seq_len: 16,
                rotary_dim: 2,
                rope: rdna_compute::rope::Rope {
                    theta: 10_000.0,
                    scaling: rdna_compute::rope::RopeScaling::None,
                },
            })
            .unwrap();
            state.position = 7;
            state.qsa.position = 7;
            let snapshot = state.begin_transaction();
            state.forced_advance(consumed).unwrap();
            state.commit(snapshot).unwrap();
            assert_eq!(state.position, expected_position);
            assert_eq!(state.qsa.position, expected_position);
            assert_eq!(state.step_index, consumed);
        }
    }

    #[test]
    fn compact_native_selection_keeps_only_target_aligned_prefix() {
        let mut state = Qwen4MtpState::new(MtpQsaGeometry {
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            index_heads: 1,
            index_dim: 2,
            compress_ratio: 2,
            budget: 4,
            max_seq_len: 16,
            rotary_dim: 2,
            rope: rdna_compute::rope::Rope {
                theta: 10_000.0,
                scaling: rdna_compute::rope::RopeScaling::None,
            },
        })
        .unwrap();
        state.qsa.position = 6;
        state.qsa.selected_indices = vec![0, 1, 3, 5];
        compact_native_qsa_selection(&mut state, 1, 3).unwrap();
        assert_eq!(state.qsa.selected_indices, vec![0, 1, 3]);
        assert_eq!(committed_target_positions(1, 3).unwrap(), vec![1, 2, 3]);
        assert_eq!(native_commit_position(1, 3).unwrap(), 4);
    }

    #[test]
    fn malformed_target_picks_are_rejected_before_acceptance() {
        assert!(accept_native_greedy(&[2], &[], None).is_err());
        assert!(accept_native_greedy(&[2, 3], &[2, 3], None).is_err());
    }

    #[test]
    fn sampled_verify_carries_penalties_into_policy_and_history() {
        let cfg = SpecRequestConfig {
            temp: 0.7,
            repeat_penalty: 1.15,
            repeat_window: 64,
            presence_penalty: 0.5,
            frequency_penalty: 0.25,
            ..SpecRequestConfig::default()
        };
        assert_eq!(cfg.penalty_window(), 64);
        for mode in [SampledMode::Leviathan, SampledMode::Naive] {
            let s = SampledVerify::new(cfg, 4, mode);
            assert_eq!(s.policy.repeat_penalty, 1.15);
            assert_eq!(s.policy.repeat_window, 64);
            assert_eq!(s.policy.presence_penalty, 0.5);
            assert_eq!(s.policy.frequency_penalty, 0.25);
            assert_eq!(s.history.window(), 64);
            assert_eq!(s.draws_drafts(), mode == SampledMode::Leviathan);
        }
    }

    #[test]
    fn sampled_verify_neutral_request_has_no_history_window() {
        let neutral = SpecRequestConfig {
            temp: 0.7,
            ..SpecRequestConfig::default()
        };
        assert_eq!(neutral.penalty_window(), 0);
        let s = SampledVerify::new(neutral, 4, SampledMode::Leviathan);
        assert_eq!(s.history.window(), 0);
        assert_eq!(s.policy.repeat_penalty, 1.0);
        assert_eq!(s.drafts.len(), 4);
        // A window with no penalty never buffers history.
        let mut s = s;
        s.history.set_prompt(&[1, 2, 3]);
        assert_eq!(s.begin_window(&[4, 5]), 0);
        s.history.push_draft(6);
        assert!(s.history.row(1).is_empty());

        // A repeat window with no active penalty is also window 0.
        let inert = SpecRequestConfig {
            temp: 0.7,
            repeat_window: 64,
            ..SpecRequestConfig::default()
        };
        assert_eq!(
            SampledVerify::new(inert, 4, SampledMode::Leviathan)
                .history
                .window(),
            0
        );
    }

    #[test]
    fn sampled_verify_history_rows_follow_prompt_emitted_and_drafts() {
        let cfg = SpecRequestConfig {
            temp: 0.7,
            repeat_penalty: 1.2,
            repeat_window: 4,
            ..SpecRequestConfig::default()
        };
        let mut s = SampledVerify::new(cfg, 4, SampledMode::Naive);
        s.history.set_prompt(&[1, 2, 3]);
        assert_eq!(s.history.row(0), &[1, 2, 3]);
        // `emitted` already holds the pending seed.
        s.begin_window(&[7]);
        assert_eq!(s.history.row(0), &[1, 2, 3, 7]);
        s.history.push_draft(8);
        s.history.push_draft(9);
        assert_eq!(s.history.row(2), &[3, 7, 8, 9]);
        s.history.rewind_drafts();
        assert_eq!(s.history.row(0), &[1, 2, 3, 7]);
    }

    #[test]
    fn ngram_source_runs_only_while_its_yield_matches_native() {
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None);
        assert!(drafter.ngram_wins(3, 3), "a fresh request tries the pool");
        // A pool emitting 2 tokens per 3-candidate window (the Halo smoke's
        // code_edit_rewrite_copy pattern) loses to native MTP agreeing at 0.95.
        drafter.agreement.fill((0.95, 1.0));
        drafter.ngram_yield = (16.0, 8.0);
        assert!(!drafter.ngram_wins(3, 3));
        // A verbatim copy (4 tokens per window) beats it.
        drafter.ngram_yield = (32.0, 8.0);
        assert!(drafter.ngram_wins(3, 3));
    }

    #[test]
    fn ngram_arch_pricing_refuses_r9700_copy_but_keeps_halo_winner() {
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None);
        drafter.agreement.fill((0.95, 1.0));
        drafter.agreement[1] = (0.86 / 0.95, 1.0);
        // Five takeovers averaging 2.4 emitted tokens; preserve the prior.
        drafter.ngram_yield = (12.0, 5.0);
        drafter.ngram_row_cost = Some((30.3, 14.8));
        assert!((drafter.ngram_window_cost(3) - 74.7 / 30.3).abs() < 1e-5);
        assert!(!drafter.ngram_wins(3, 2));
        drafter.ngram_row_cost = Some((30.5, 6.9));
        drafter.ngram_yield = (17.0, 5.0);
        assert!((drafter.ngram_window_cost(3) - 51.2 / 30.5).abs() < 1e-5);
        assert!(drafter.ngram_wins(3, 2));
    }

    #[test]
    fn ngram_pricing_uses_request_measurements_after_probing() {
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None);
        drafter.ngram_row_cost = Some((30.3, 14.8));
        drafter.floor.seed_ar_us(30_000.0);
        drafter.floor.seed_batched(3, (240_000.0, 2.0));
        assert!(drafter.floor.batched_cost(3).is_none());
        drafter.floor.seed_probes_done();
        assert_eq!(drafter.floor.batched_cost(3), Some(4.0));
        assert_eq!(drafter.ngram_window_cost(3), 4.0);
        assert!(!drafter.ngram_wins(3, 2));
        drafter.floor.seed_batched(3, (30_000.0, 2.0));
        assert!(drafter.ngram_wins(3, 2));
        drafter.ngram_row_cost = None;
        assert_eq!(drafter.ngram_window_cost(3), MTP_WINDOW_COST[3]);
    }

    #[test]
    fn cost_aware_ngram_depth_extends_only_confident_long_matches() {
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None);
        drafter.ngram_row_cost = Some((30.5, 6.9));
        drafter.ngram_yield = (20.0, 5.0);
        assert_eq!(drafter.ngram_takeover_depth(8, 3), 3);
        drafter.stats.ngram_mod_windows = 5;
        drafter.stats.ngram_mod_drafts = 15;
        drafter.stats.ngram_mod_accepted = 15;
        assert_eq!(drafter.ngram_takeover_depth(8, 3), 7);
        assert_eq!(drafter.ngram_takeover_depth(3, 3), 3);
        assert_eq!(drafter.ngram_takeover_depth(63, 3), 7);
        assert!(drafter.ngram_window_cost(8).is_infinite());
        drafter.stats.ngram_mod_accepted = 12;
        assert_eq!(drafter.ngram_takeover_depth(8, 3), 3);
    }

    /// A drafter whose pool is trained on `0..21` (so `[0, 1]` continues as
    /// `[2, 3, 4, ..]`) and whose takeover history is confident and
    /// well-established: the cost-aware chooser extends past the native depth.
    fn armed_ngram_drafter(n_min: usize, n_max: usize) -> Qwen4MtpDrafter {
        let config = NgramModConfig {
            capacity: 1024,
            n_match: 2,
            n_min,
            n_max,
        };
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None).with_ngram(Some(config));
        let mut pool = MtpNgramContext::new(config).expect("pool");
        pool.begin_request(&(0..21u32).collect::<Vec<_>>());
        drafter.ngram = Some(pool);
        drafter.ngram_active = true;
        drafter.floor_enabled = false;
        drafter.ngram_row_cost = Some((30.5, 6.9));
        drafter.stats.ngram_mod_windows = 5;
        drafter.stats.ngram_mod_drafts = 15;
        drafter.stats.ngram_mod_accepted = 15;
        drafter.ngram_yield = (20.0, 5.0);
        drafter
    }

    #[test]
    fn ngram_hit_never_proposes_past_the_callers_budget() {
        let emitted = [0u32, 1];

        // The fixture is live: a full budget takes the whole trained chain.
        let mut drafter = armed_ngram_drafter(3, 3);
        assert!(drafter.ngram_hit(&emitted, 3));
        assert_eq!(drafter.ngram_candidates, vec![2, 3, 4]);

        // A budget below n_min refuses the hit instead of extending past it.
        for k in 0..3 {
            let mut drafter = armed_ngram_drafter(3, 3);
            assert!(!drafter.ngram_hit(&emitted, k), "k={k}");
        }

        // With n_min = 1 every admitted proposal fits the budget; k = 0 is no
        // takeover. (`ngram_candidates` is only meaningful after a true hit.)
        for k in 0..=8usize {
            let mut drafter = armed_ngram_drafter(1, 8);
            let hit = drafter.ngram_hit(&emitted, k);
            assert_eq!(hit, k > 0, "k={k}");
            if hit {
                assert!(
                    drafter.ngram_candidates.len() <= k,
                    "k={k}: {} candidates exceed the budget",
                    drafter.ngram_candidates.len()
                );
            }
        }

        // The cap is the caller's budget, not the native depth: a large
        // budget still extends past `max_k = 3` for a configured longer match.
        let mut drafter = armed_ngram_drafter(3, 8);
        assert!(drafter.ngram_hit(&emitted, 8));
        assert!(
            drafter.ngram_candidates.len() > 3,
            "{:?}",
            drafter.ngram_candidates
        );
    }

    #[test]
    fn ngram_takeover_consumes_exactly_the_emitted_rows_and_leaves_the_bonus_pending() {
        let emitted = [0u32, 1];
        let start = 1256usize;
        for k in 1..=8usize {
            let mut drafter = armed_ngram_drafter(1, 8);
            assert!(drafter.ngram_hit(&emitted, k), "k={k}");
            let candidates = drafter.ngram_candidates.clone();
            assert!(candidates.len() <= k, "k={k}");

            // The verifier accepts every candidate and picks bonus 99.
            let mut picks = candidates.clone();
            picks.push(99);
            let acceptance = accept_native_greedy(&candidates, &picks, None).unwrap();
            assert_eq!(acceptance.accepted, candidates.len());
            let consumed = target_commit_accept_len(&acceptance) + 1;
            let step = window_to_spec_step(MtpWindow {
                committed: acceptance.committed.clone(),
                accepted: acceptance.accepted,
                drafts_generated: candidates.len(),
            })
            .unwrap()
            .cap_emit(k + 1);

            // No release-only host clipping: the clip is a no-op and the rows
            // the target and head consumed equal the emitted count.
            assert_eq!(step.emit.len(), acceptance.committed.len(), "k={k}");
            assert_eq!(consumed, step.emit.len(), "k={k}");
            assert_eq!(step.accepted, candidates.len(), "k={k}");
            // The bonus is the pending seed: emitted, but not yet a device row.
            assert_eq!(step.next_seed, 99, "k={k}");
            let owners = native_commit_position(start, consumed).unwrap();
            assert_eq!(owners, start + step.emit.len(), "k={k}");
            // The terminal pending-seed flush writes exactly that one row to
            // both owners and meets the host cursor.
            assert_eq!(
                native_commit_position(owners, 1).unwrap(),
                owners + 1,
                "k={k}"
            );
            if k == 1 {
                assert_eq!(owners, 1258);
                assert_eq!(native_commit_position(owners, 1).unwrap(), 1259);
            }
        }
    }

    fn plan_of(drafter: &mut Qwen4MtpDrafter, k: usize, incremental: Option<&str>) -> WindowPlan {
        drafter
            .plan_window_with(&[0, 1], k, incremental, false)
            .expect("plan")
    }

    #[test]
    fn rearm_retired_thinking_recalibrates_then_probes_with_original_thresholds() {
        for temp in [0.0, 0.8] {
            let mut drafter = Qwen4MtpDrafter::new(7, 4096, None);
            drafter.think_close = Some(99);
            drafter.sampled_enabled = true;
            drafter.sampled_mode = Ok(SampledMode::Leviathan);
            let cfg = SpecRequestConfig {
                temp,
                rng_seed: 123,
                ..Default::default()
            };
            drafter.configure_request(cfg);
            drafter.floor_enabled = true;
            drafter.floor = Qwen4Floor::with_retire_after(MTP_WINDOW_COST, 5)
                .with_decay(0.97)
                .with_retire_evidence(2);
            assert_eq!(
                drafter
                    .plan_window_with(&[10], 7, None, false)
                    .unwrap()
                    .route,
                WindowRoute::ArCalibrate
            );
            drafter.agreement.fill((0.0, 2.0));
            drafter.floor.observe_calibration(30.0, &drafter.agreement);
            for _ in 0..5 {
                drafter.floor.observe_native(
                    NativeWindow::Interleaved,
                    &window(0, 1),
                    90.0,
                    7,
                    &drafter.agreement,
                );
            }
            run_probes(&mut drafter.floor, &drafter.agreement, 2, 30.0);
            assert!(drafter.floor.retired());
            drafter.stats.mtp_retired = true;
            drafter.head_lag.push(100, 10);
            let next_draw = |drafter: &Qwen4MtpDrafter| {
                drafter.sampled.as_ref().map(|state| {
                    let mut rng = state.rng;
                    rng.next_u64()
                })
            };
            let sampled_draw = next_draw(&drafter);
            assert_eq!(
                drafter
                    .plan_window_with(&[10, 11], 7, None, false)
                    .unwrap()
                    .route,
                WindowRoute::ArRetired
            );
            assert_eq!(
                drafter
                    .plan_window_with(&[10, 11, 99], 7, None, false)
                    .unwrap()
                    .route,
                WindowRoute::ArCalibrate
            );
            assert!(!drafter.floor.retired() && !drafter.stats.mtp_retired);
            assert!(!drafter.floor.calibrated());
            let floor_state = drafter.floor.diagnostics();
            assert_eq!((floor_state.observed, floor_state.probes), (0, 0));
            assert_eq!(drafter.floor.diagnostics().retire_after, 5);
            assert_eq!(drafter.floor.diagnostics().probe_tokens, 2);
            assert_eq!(drafter.floor.diagnostics().decay, 0.97);
            assert_eq!(drafter.floor.diagnostics().combined, (0.0, 0.0));
            assert_eq!(drafter.floor.diagnostics().slow, (0.0, 0.0));
            assert_eq!(
                drafter.floor.diagnostics().caps,
                [CapStats::<{ MTP_WINDOW_COST.len() }>::default(); MTP_WINDOW_COST.len()]
            );
            assert_eq!(drafter.request, cfg);
            assert_eq!(
                next_draw(&drafter),
                sampled_draw,
                "re-arm must not advance or reseed RNG"
            );
            // Planning preserves owed rows/pending: run_window catches up every
            // rank before the new calibration can overwrite their hidden rows.
            assert_eq!(drafter.head_lag.tokens, vec![10]);
            drafter.floor.observe_calibration(30.0, &drafter.agreement);
            assert_eq!(
                drafter
                    .plan_window_with(&[10, 11, 99], 7, None, false)
                    .unwrap()
                    .route,
                WindowRoute::Native { depth: 3 }
            );
        }
    }

    #[test]
    fn rearm_each_think_close_is_consumed_once_even_after_another_retirement() {
        let mut drafter = Qwen4MtpDrafter::new(7, 4096, None);
        drafter.think_close = Some(99);
        drafter.floor.seed_retired(true);
        drafter.observe_think_close(&[10, 99], true);
        assert!(!drafter.floor.retired());
        drafter.floor.seed_retired(true);
        drafter.observe_think_close(&[10, 99], true);
        drafter.observe_think_close(&[10, 99, 12], true);
        assert!(
            drafter.floor.retired(),
            "the old close must not re-arm again"
        );
        drafter.observe_think_close(&[10, 99, 12, 99], true);
        assert!(!drafter.floor.retired(), "a distinct new close may re-arm");
        drafter.configure_request(SpecRequestConfig::default());
        assert_eq!(drafter.emitted_seen, None);
    }

    #[test]
    fn rearm_requires_a_new_close_and_an_active_retired_floor() {
        for (close, retired, active) in [
            (None, true, true),
            (Some(99), false, true),
            (Some(99), true, false),
        ] {
            let mut drafter = Qwen4MtpDrafter::new(7, 4096, None);
            drafter.think_close = close;
            drafter.floor.seed_retired(retired);
            drafter.floor.seed_observed(42);
            drafter.observe_think_close(&[99], active);
            assert_eq!(drafter.floor.retired(), retired);
            assert_eq!(drafter.floor.diagnostics().observed, 42);
        }
        let mut drafter = Qwen4MtpDrafter::new(7, 4096, None);
        drafter.think_close = Some(99);
        drafter.floor.seed_retired(true);
        // A first plan may carry a prompt prefix in a direct caller's history:
        // only its pending seed is newly generated, not that older close.
        drafter.observe_think_close(&[99, 10], true);
        assert!(drafter.floor.retired());
    }

    #[test]
    fn rearm_fresh_chooser_starts_at_three_without_lowering_the_native_cap() {
        let cap = hipfire_config::mtp_k_default(
            16,
            rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows("gfx1151"),
        );
        assert_eq!(cap, 7);
        let mut drafter = Qwen4MtpDrafter::new(cap, 4096, None);
        drafter.floor.observe_calibration(100.0, &drafter.agreement);
        for k in 1..=cap {
            let plan = plan_of(&mut drafter, k, None);
            assert_eq!(plan.route, WindowRoute::Native { depth: k.min(3) });
            assert_eq!(plan.k, k);
        }
        assert_eq!(drafter.k(), 7);
        assert_eq!(
            plan_of(&mut drafter, cap, Some("0")).route,
            WindowRoute::Native { depth: 7 }
        );
        assert_eq!(
            plan_of(&mut drafter, cap, Some("1")).route,
            WindowRoute::Native { depth: 0 }
        );
    }

    #[test]
    fn rearm_fresh_chooser_can_grow_from_three_to_seven() {
        let mut drafter = Qwen4MtpDrafter::new(7, 4096, None);
        drafter.floor.observe_calibration(30.0, &drafter.agreement);
        let mut reached = false;
        for i in 0..100 {
            let plan = plan_of(&mut drafter, 7, None);
            let WindowRoute::Native { depth } = plan.route else {
                panic!("native route");
            };
            if i == 0 {
                assert_eq!(depth, 3);
            }
            reached |= depth == 7;
            let kept = window(depth, depth);
            drafter.observe_agreement(&kept);
            drafter.floor.observe_window(
                NativeWindow::Batched,
                WindowStats::of(&kept, depth),
                60.0 + depth as f32 * 2.0,
                7,
                &drafter.agreement,
            );
        }
        assert!(
            reached,
            "positive measured evidence must still reach the cap"
        );
        assert_eq!(drafter.k(), 7);
    }

    #[test]
    fn plan_window_checks_retirement_then_calibration_before_the_ngram_pool() {
        // A retired request runs head-free AR: the pool is never asked and the
        // takeover yield history is untouched.
        let mut drafter = armed_ngram_drafter(1, 8);
        drafter.floor_enabled = true;
        drafter.floor.seed_retired(true);
        drafter.ngram_yield = (20.0, 5.0);
        let plan = plan_of(&mut drafter, 3, None);
        assert_eq!(plan.route, WindowRoute::ArRetired);
        assert!(plan.floor_active && !plan.ngram_hit);
        assert!(
            drafter.ngram_candidates.is_empty(),
            "the pool was never asked"
        );
        assert_eq!(drafter.ngram_yield, (20.0, 5.0));

        // The request's first window calibrates AR before any pool hit.
        let mut drafter = armed_ngram_drafter(1, 8);
        drafter.floor_enabled = true;
        drafter.ngram_yield = (20.0, 5.0);
        let plan = plan_of(&mut drafter, 3, None);
        assert_eq!(plan.route, WindowRoute::ArCalibrate);
        assert!(!plan.ngram_hit && drafter.ngram_candidates.is_empty());
        assert_eq!(drafter.ngram_yield, (20.0, 5.0));

        // A developer-forced route bypasses the floor entirely.
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None);
        let plan = plan_of(&mut drafter, 3, Some("1"));
        assert_eq!(plan.route, WindowRoute::Native { depth: 0 });
        assert!(!plan.floor_active);
        let plan = plan_of(&mut drafter, 2, Some("0"));
        assert_eq!(plan.route, WindowRoute::Native { depth: 2 });
        assert_eq!(plan.k, 2);
        assert!(!drafter.floor.calibrated() && !drafter.floor.retired());
    }

    #[test]
    fn plan_window_routes_through_the_floor_and_retires_in_the_plan() {
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None);
        assert!(plan_window_budget_error(&mut drafter, 4).contains("proposal capacity"));
        drafter.floor.seed_calibrated(30_000.0);
        // Probing: the Halo-table depth of the agreement prior, clamped to K.
        let plan = plan_of(&mut drafter, 3, None);
        assert_eq!(
            plan.route,
            WindowRoute::Native {
                depth: batched_depth(&drafter.agreement, 3, &MTP_WINDOW_COST)
            }
        );
        assert_eq!(plan.k, 3);
        assert!(!drafter.floor.retired() && !drafter.stats.mtp_retired);
        // Past the probe budget and the grace with no measured winner, the
        // routing itself retires the request (rank 0's state) and says AR.
        drafter.floor.seed_probes_done();
        let retire_after = drafter.floor.diagnostics().retire_after;
        drafter.floor.seed_observed(retire_after);
        let plan = plan_of(&mut drafter, 3, None);
        assert_eq!(plan.route, WindowRoute::ArRouted);
        assert!(drafter.floor.retired() && drafter.stats.mtp_retired);
        // Retirement is sticky: the next window skips the resource pass.
        assert_eq!(plan_of(&mut drafter, 3, None).route, WindowRoute::ArRetired);
    }

    fn plan_window_budget_error(drafter: &mut Qwen4MtpDrafter, k: usize) -> String {
        drafter
            .plan_window_with(&[0, 1], k, None, false)
            .expect_err("a budget past the proposal capacity is refused")
    }

    #[test]
    fn plan_window_runs_the_ngram_gate_exactly_as_the_old_chain_and_shares_candidates_on_demand() {
        // Floor off: the pre-floor chooser; the hit is the pool's.
        let mut drafter = armed_ngram_drafter(1, 8);
        let mut plan = plan_of(&mut drafter, 3, None);
        assert!(plan.ngram_hit && !plan.floor_active);
        assert_eq!(
            plan.route,
            WindowRoute::Native {
                depth: batched_depth(&drafter.agreement, 3, &MTP_WINDOW_COST)
            }
        );
        // Single card allocates nothing for the plan; EP attaches the copy.
        assert!(plan.ngram_candidates.is_empty());
        drafter.attach_candidates(&mut plan);
        assert!(!plan.ngram_candidates.is_empty());
        assert_eq!(plan.ngram_candidates, drafter.ngram_candidates);

        // A declined hit decays the takeover yield history in the plan, as the
        // old chain did before routing.
        let mut drafter = armed_ngram_drafter(1, 8);
        drafter.stats.ngram_mod_windows = 0;
        drafter.ngram_yield = (0.0, 100.0);
        let plan = plan_of(&mut drafter, 3, None);
        assert!(!plan.ngram_hit);
        assert!(drafter.ngram_yield.1 < 100.0);
        let mut declined = plan.clone();
        drafter.attach_candidates(&mut declined);
        assert!(declined.ngram_candidates.is_empty());
    }

    #[test]
    fn replay_queue_compacts_and_a_replica_queues_nothing() {
        let cfg = |seed: u64| SpecRequestConfig {
            rng_seed: seed,
            ..SpecRequestConfig::default()
        };
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None);
        drafter.push_replay(ReplayOp::Reset);
        drafter.push_replay(ReplayOp::Reset);
        assert_eq!(drafter.pending_replay(), &[ReplayOp::Reset]);
        MtpDrafter::configure_request(&mut drafter, cfg(1));
        drafter.push_replay(ReplayOp::Reset);
        drafter.push_replay(ReplayOp::Reset);
        assert_eq!(
            drafter.pending_replay(),
            &[
                ReplayOp::Reset,
                ReplayOp::Configure(cfg(1)),
                ReplayOp::Reset
            ]
        );
        // A later configure supersedes the earlier one.
        MtpDrafter::configure_request(&mut drafter, cfg(2));
        assert_eq!(
            drafter.pending_replay(),
            &[ReplayOp::Reset, ReplayOp::Configure(cfg(2))]
        );
        // A single-card load never drains the queue: it stays bounded.
        for seed in 0..10_000u64 {
            MtpDrafter::configure_request(&mut drafter, cfg(seed));
            drafter.push_replay(ReplayOp::Reset);
            drafter.push_replay(ReplayOp::Reset);
            assert!(drafter.pending_replay().len() <= 3);
        }
        assert_eq!(
            drafter.pending_replay().last(),
            Some(&ReplayOp::Reset),
            "the order of the surviving ops is the call order"
        );
        drafter.clear_replay();
        assert!(drafter.pending_replay().is_empty());

        // A replica is built from rank 0's parameters and queues nothing.
        let leader = Qwen4MtpDrafter::new(3, 4096, Some(7)).with_ngram(None);
        let mut replica = Qwen4MtpDrafter::new_replica(leader.replica_params());
        assert!(replica.follower && !leader.follower);
        assert_eq!(
            (replica.max_k, replica.ctx_capacity, replica.end_of_turn),
            (3, 4096, Some(7))
        );
        MtpDrafter::configure_request(&mut replica, cfg(5));
        replica.push_replay(ReplayOp::Reset);
        assert!(replica.pending_replay().is_empty());
        replica.mirror_ngram_active(true);
        assert!(replica.ngram_active());
    }

    #[test]
    fn q8_ef_sampled_requests_route_to_ar() {
        use crate::GdnStateFormat;
        assert!(!qwen4_mtp_sampled_allowed(GdnStateFormat::Q8Ef));
        assert!(qwen4_mtp_sampled_allowed(GdnStateFormat::Q8));
        assert!(qwen4_mtp_sampled_allowed(GdnStateFormat::F32));
        let speculator = build_qwen4_mtp_speculator(
            3,
            None,
            4096,
            None,
            None,
            None,
            false,
            GdnStateFormat::Q8Ef,
            None,
        );
        assert!(speculator.requires_greedy());
        assert!(!speculator.supports_temp_verify());

        // A replica mirrors its leader's sampled gate.
        let leader = Qwen4MtpDrafter::new(3, 4096, None).with_sampled_allowed(false);
        assert!(!leader.sampled_enabled);
        let replica = Qwen4MtpDrafter::new_replica(leader.replica_params());
        assert!(!replica.sampled_enabled);
        let leader = Qwen4MtpDrafter::new(3, 4096, None);
        let replica = Qwen4MtpDrafter::new_replica(leader.replica_params());
        assert_eq!(replica.sampled_enabled, leader.sampled_enabled);
    }

    #[test]
    fn q8_and_fp32_sampled_mtp_unchanged() {
        use crate::GdnStateFormat;
        let sampled = hipfire_config::mtp_sampled_enabled();
        let plain = Qwen4MtpDrafter::new(3, 4096, None);
        assert_eq!(plain.sampled_enabled, sampled);
        for gdn in [GdnStateFormat::Q8, GdnStateFormat::F32] {
            let speculator =
                build_qwen4_mtp_speculator(3, None, 4096, None, None, None, false, gdn, None);
            assert_eq!(speculator.supports_temp_verify(), sampled, "{gdn:?}");
            assert_eq!(speculator.requires_greedy(), !sampled, "{gdn:?}");
            let gated = Qwen4MtpDrafter::new(3, 4096, None)
                .with_sampled_allowed(qwen4_mtp_sampled_allowed(gdn));
            assert_eq!(gated.sampled_enabled, plain.sampled_enabled, "{gdn:?}");
        }
    }

    #[test]
    fn replica_rank_never_feeds_its_clocks_to_the_floor() {
        let mut leader = Qwen4MtpDrafter::new(3, 4096, None);
        let mut replica = Qwen4MtpDrafter::new_replica(leader.replica_params());
        let window = MtpWindow {
            committed: vec![1, 2],
            accepted: 1,
            drafts_generated: 1,
        };
        for drafter in [&mut leader, &mut replica] {
            drafter.floor.seed_calibrated(1_000_000.0);
            drafter.floor_observe_native(false, NativeWindow::Interleaved, &window, 10.0, 1);
            assert!(!drafter.floor_observe_external_window(2, 10.0, 1));
        }
        assert_eq!(leader.floor.diagnostics().observed, 2);
        assert_eq!(replica.floor.diagnostics().observed, 0);
        assert!(!replica.floor.retired());
        // Both count the window (stats are rank-local bookkeeping).
        assert_eq!(leader.stats.mtp_windows, replica.stats.mtp_windows);
        assert_eq!(window_key(Ok(window.clone())), Ok((vec![1, 2], 1, 1)));
    }

    #[test]
    fn unobserved_depth_relaxes_to_the_prior_so_a_rejected_depth_is_re_explored() {
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None);
        let window = |accepted: usize, drafts_generated: usize| MtpWindow {
            committed: vec![0; accepted + 1],
            accepted,
            drafts_generated,
        };
        assert_eq!(
            drafter.batched_depth(3),
            3,
            "the prior drafts the full depth"
        );
        // One depth-3 rejection drops that depth below its threshold.
        drafter.observe_agreement(&window(2, 3));
        assert_eq!(drafter.batched_depth(3), 2, "the rejected depth is dropped");
        // Depth 3 is never compared again (k=2 windows), yet must come back.
        let mut recovered_after = None;
        for n in 1..=40 {
            drafter.observe_agreement(&window(2, 2));
            if drafter.batched_depth(3) == 3 {
                recovered_after = Some(n);
                break;
            }
        }
        let n = recovered_after.expect("depth 3 must be re-explored");
        assert!(n < 40, "depth 3 returned only after {n} windows");
    }

    #[test]
    fn singleton_decay_is_long_on_gfx1151_only_and_reaches_the_request_floor() {
        assert_eq!(DEFAULT_DECAY, 0.875);
        assert_eq!(MTP_AGREEMENT_DECAY_GFX1151, 0.97);
        assert_eq!(singleton_agreement_decay("gfx1151"), 0.97);
        for arch in [
            "",
            "gfx1150",
            "gfx1152",
            "gfx1201",
            "gfx1200",
            "gfx1100",
            "gfx942",
            "gfx1151:sramecc+",
            "GFX1151",
        ] {
            assert_eq!(singleton_agreement_decay(arch), 0.875, "{arch}");
        }
        assert_eq!(Qwen4Floor::new(MTP_WINDOW_COST).diagnostics().decay, 0.875);
        // Binding an arch selects one decay for agreement and every floor
        // cost pair; configuring another request must preserve that choice.
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None);
        assert_eq!(drafter.agreement_decay, 0.875);
        for arch in ["gfx1151", "gfx1201", "gfx1151", "gfx942"] {
            let decay = singleton_agreement_decay(arch);
            drafter.agreement_decay = decay;
            drafter.agreement.fill(AGREEMENT_PRIOR);
            MtpDrafter::configure_request(&mut drafter, SpecRequestConfig::default());
            assert_eq!(drafter.floor.diagnostics().decay, decay, "{arch}");
            drafter.observe_agreement(&window(0, 1));
            let (prior_accepted, prior_total) = AGREEMENT_PRIOR;
            assert_eq!(
                drafter.agreement[0],
                (prior_accepted * decay, prior_total * decay + 1.0),
                "{arch}"
            );
            drafter.floor.observe_calibration(100.0, &drafter.agreement);
            let mut combined = (0.0, 0.0);
            for route in [NativeWindow::Interleaved, NativeWindow::Batched] {
                for _ in 0..2 {
                    drafter
                        .floor
                        .observe_native(route, &window(1, 1), 40.0, 3, &drafter.agreement);
                    combined = (combined.0 * decay + 40.0, combined.1 * decay + 2.0);
                }
            }
            assert_eq!(drafter.floor.diagnostics().combined, combined);
            assert_eq!(
                drafter.floor.diagnostics().interleaved,
                (40.0 * decay + 40.0, 2.0 * decay + 2.0)
            );
            assert_eq!(
                drafter.floor.diagnostics().batched[1],
                (40.0 * decay + 40.0, decay + 1.0)
            );
            for _ in 0..2 {
                assert!(!drafter
                    .floor
                    .observe_external(2, 40.0, 1, &drafter.agreement));
                combined = (combined.0 * decay + 40.0, combined.1 * decay + 2.0);
            }
            assert_eq!(drafter.floor.diagnostics().combined, combined);
            assert_eq!(
                drafter.floor.diagnostics().external,
                (40.0 * decay + 40.0, 2.0 * decay + 2.0)
            );
            drafter.floor.seed_retired(true);
            drafter.ngram_yield = (8.0, 2.0);
            let agreement = drafter.agreement.clone();
            MtpDrafter::configure_request(&mut drafter, SpecRequestConfig::default());
            assert_eq!(drafter.agreement_decay, decay);
            assert_eq!(drafter.floor.diagnostics().decay, decay);
            assert!(!drafter.floor.retired());
            assert!(!drafter.floor.calibrated());
            assert_eq!(drafter.floor.diagnostics().observed, 0);
            assert_eq!(drafter.floor.diagnostics().combined, (0.0, 0.0));
            assert_eq!(drafter.floor.diagnostics().external, (0.0, 0.0));
            assert_eq!(drafter.floor.diagnostics().interleaved, (0.0, 0.0));
            assert_eq!(
                drafter.floor.diagnostics().batched,
                [(0.0, 0.0); MTP_WINDOW_COST.len()]
            );
            assert_eq!(drafter.ngram_yield, (0.0, 0.0));
            // Agreement restarts during prefill, not configure_request.
            assert_eq!(drafter.agreement, agreement);
        }
    }

    #[test]
    fn ngram_config_caps_n_max_below_the_verify_rows() {
        let config = qwen4_ngram_mod_config((5, 3, 3)).expect("arch-16 default");
        assert_eq!((config.n_match, config.n_min, config.n_max), (5, 3, 3));
        assert_eq!(config.capacity, NgramModConfig::default().capacity);
        // [seed, 64 candidates] would be 65 verify rows: one too many.
        let config = qwen4_ngram_mod_config((24, 48, 64)).expect("inherited triple");
        assert_eq!((config.n_match, config.n_min, config.n_max), (24, 48, 63));
        assert!(qwen4_ngram_mod_config((0, 1, 3)).is_err());
        assert!(qwen4_ngram_mod_config((5, 4, 3)).is_err());
        assert!(qwen4_ngram_mod_config((5, 3, 0)).is_err());
        assert!(qwen4_ngram_mod_config((5, 3, 65)).is_err());
    }

    fn agreement(accepted: f32, total: f32) -> Agreement {
        [(accepted, total); MAX_DEPTH]
    }

    fn window(accepted: usize, drafts: usize) -> MtpWindow {
        MtpWindow {
            committed: vec![0; accepted + 1],
            accepted,
            drafts_generated: drafts,
        }
    }

    fn calibrated(ar_us: f32, agreement: &[(f32, f32)]) -> Qwen4Floor {
        let mut floor = Qwen4Floor::new(MTP_WINDOW_COST);
        floor.observe_calibration(ar_us, agreement);
        floor
    }

    /// `n` batched windows routed at and drafting `depth`, every draft
    /// accepted, `us` each.
    fn batched_windows(
        floor: &mut Qwen4Floor,
        agreement: &[(f32, f32)],
        depth: usize,
        us: f32,
        n: usize,
    ) {
        for _ in 0..n {
            floor.observe_native(
                NativeWindow::Batched,
                &window(depth, depth),
                us,
                7,
                agreement,
            );
        }
    }

    #[test]
    fn retired_head_lag_owes_contiguous_rows_from_the_head_position() {
        let mut lag = HeadLag::default();
        // The first owed token fixes the head position; later tokens must
        // continue it row by row (spec hidden row i = the hidden of position
        // start + i, which pairs with token i + 1; token 0 pairs with the
        // pending hidden of start - 1).
        assert_eq!(lag.next_row(34_120, 4), Ok(0));
        lag.push(34_120, 11);
        assert_eq!(lag.next_row(34_121, 4), Ok(1));
        let gap = lag.next_row(34_122, 4).unwrap_err();
        assert!(gap.contains("does not continue"), "{gap}");
        let back = lag.next_row(34_120, 4).unwrap_err();
        assert!(back.contains("does not continue"), "{back}");
        lag.push(34_121, 12);
        lag.push(34_122, 13);
        lag.push(34_123, 14);
        assert_eq!(lag.start, 34_120);
        assert_eq!(lag.tokens, [11, 12, 13, 14]);
        // A full record has no free row: the drafter catches up first.
        let full = lag.next_row(34_124, 4).unwrap_err();
        assert!(full.contains("no free row"), "{full}");
        // After a catch-up (or a reset) the next owed token starts anew.
        lag.clear();
        assert_eq!(lag.next_row(40_000, 4), Ok(0));
        lag.push(40_000, 21);
        assert_eq!(lag.start, 40_000);
        assert_eq!(lag.tokens, [21]);
    }

    #[test]
    fn retired_head_lag_capacity_is_one_batched_fill_and_requests_drop_it() {
        let mut drafter = Qwen4MtpDrafter::new(4, 65_536, None);
        assert_eq!(drafter.head_lag_capacity(), 1, "no resources yet");
        drafter.prefill_rows = 8192;
        assert_eq!(drafter.head_lag_capacity(), MTP_FILL_ROWS);
        drafter.prefill_rows = 512;
        assert_eq!(drafter.head_lag_capacity(), 512);
        // A new request never inherits owed rows of an earlier one.
        drafter.head_lag.push(7, 3);
        MtpDrafter::configure_request(&mut drafter, SpecRequestConfig::default());
        assert!(drafter.head_lag.tokens.is_empty());
    }

    /// Deterministic draws in `[0, 1)` (xorshift32).
    struct Draws(u32);

    impl Draws {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            (self.0 >> 8) as f32 / (1u32 << 24) as f32
        }
    }

    /// The depth of each of `windows` native windows of a singleton request
    /// (0: interleaved, 8: a retired request's AR token), driven the way the
    /// drafter drives the floor: calibrated at `ar_us`, the table depth during
    /// the probes (`forced`: every window at that depth, floor bypassed), AR
    /// probe tokens priced at `ar_us`, and every batched window drafting its
    /// routed depth up to the margin prune's cut, accepting drafts while a
    /// draw stays below the acceptance and costing `window_us[drafted]`;
    /// `stream(window)` is `(acceptance, drafts the prune keeps)`.
    fn chooser_run(
        window_us: [f32; 8],
        ar_us: f32,
        decay: f32,
        k: usize,
        windows: usize,
        forced: Option<usize>,
        seed: u32,
        stream: impl Fn(usize) -> (f32, usize),
    ) -> ChooserRun {
        let mut agreement = [AGREEMENT_PRIOR; MAX_DEPTH];
        let mut floor =
            Qwen4Floor::with_retire_after(MTP_WINDOW_COST, MTP_SINGLETON_FLOOR_MIN_WINDOWS)
                .with_decay(decay)
                .with_retire_evidence(MTP_SINGLETON_FLOOR_AR_PROBE_TOKENS);
        let mut draws = Draws(seed.max(1));
        let mut run = ChooserRun {
            depths: Vec::with_capacity(windows),
            tokens: 0.0,
            us: 0.0,
        };
        while run.depths.len() < windows {
            if forced.is_none() && floor.needs_ar_sample() {
                floor.observe_calibration(ar_us, &agreement);
                run.tokens += 1.0;
                run.us += ar_us;
                continue;
            }
            let (p, kept) = stream(run.depths.len());
            let route = match forced {
                Some(depth) => Action::Native(depth),
                None => floor.route(
                    &agreement,
                    k,
                    batched_depth(&agreement, k, &MTP_WINDOW_COST),
                ),
            };
            let (cap, drafts, accepted, us, route) = match route {
                Action::ArHeadFree => {
                    run.depths.push(8);
                    run.tokens += 1.0;
                    run.us += ar_us;
                    continue;
                }
                Action::ArSynced => continue,
                Action::Interleaved => {
                    let accepted = (0..k).take_while(|_| draws.next() < p).count();
                    let us = ar_us * (accepted + 1) as f32;
                    (
                        0,
                        (accepted + 1).min(k),
                        accepted,
                        us,
                        NativeWindow::Interleaved,
                    )
                }
                Action::Native(cap) => {
                    let drafted = cap.min(kept.max(1));
                    let accepted = (0..drafted).take_while(|_| draws.next() < p).count();
                    (
                        cap,
                        drafted,
                        accepted,
                        window_us[drafted],
                        NativeWindow::Batched,
                    )
                }
            };
            observe_agreement(&mut agreement, &window(accepted, drafts), decay);
            let stats = WindowStats {
                emitted: accepted + 1,
                drafts,
                cap,
            };
            floor.observe_window(route, stats, us, k, &agreement);
            run.depths.push(cap);
            run.tokens += (accepted + 1) as f32;
            run.us += us;
        }
        run
    }

    /// One [`chooser_run`]: the depth of every native window (0:
    /// interleaved, 8: a retired request's AR token) and the emitted tokens
    /// and wall time of the whole request, AR tokens included.
    struct ChooserRun {
        depths: Vec<usize>,
        tokens: f32,
        us: f32,
    }

    impl ChooserRun {
        fn tokens_per_ms(&self) -> f32 {
            self.tokens / self.us * 1000.0
        }
    }

    /// Halo gfx1151 Flash-Next mq4-xts window medians by drafted depth
    /// (fnrw-halo forced K=5/7 and adaptive traces) and its AR step.
    const HALO_WINDOW_US: [f32; 8] = [
        0.0, 38_500.0, 44_500.0, 51_700.0, 59_800.0, 67_000.0, 85_000.0, 92_600.0,
    ];
    const HALO_AR_US: f32 = 28_450.0;
    /// R9700 gfx1201: 48.7 ms at one draft plus 12.4 ms per extra row
    /// (fnrw-r9700 `DEPTH-RECORDS.md`), and its AR step.
    const R9700_WINDOW_US: [f32; 8] = [
        0.0, 48_700.0, 61_100.0, 73_500.0, 85_900.0, 98_300.0, 110_700.0, 123_100.0,
    ];
    const R9700_AR_US: f32 = 28_600.0;

    /// Sixteen seeded 60-window requests of one stream through
    /// [`chooser_run`] with a budget of 7.
    #[derive(Debug)]
    struct ChooserScenario {
        /// The chooser's mean tokens per ms.
        rate: f32,
        /// Mean tokens per ms of every fixed depth 1..=7 (index 0 unused).
        forced: [f32; 8],
        /// The chooser's windows from the 20th on, by depth (8: retired).
        late: [usize; 9],
    }

    impl ChooserScenario {
        fn best_depth(&self) -> usize {
            (1..8)
                .max_by(|&a, &b| self.forced[a].total_cmp(&self.forced[b]))
                .expect("depths")
        }

        /// Share of the late native windows (retired AR tokens excluded)
        /// whose depth `keep` admits.
        fn late_share(&self, keep: impl Fn(usize) -> bool) -> f32 {
            let kept: usize = (0..8).filter(|&d| keep(d)).map(|d| self.late[d]).sum();
            kept as f32 / self.late[..8].iter().sum::<usize>().max(1) as f32
        }
    }

    fn chooser_scenario(
        window_us: [f32; 8],
        ar_us: f32,
        decay: f32,
        stream: impl Fn(usize) -> (f32, usize) + Copy,
    ) -> ChooserScenario {
        let seeds = 1..=16u32;
        let mean = |forced: Option<usize>| {
            seeds
                .clone()
                .map(|seed| {
                    chooser_run(window_us, ar_us, decay, 7, 60, forced, seed * 7919, stream)
                        .tokens_per_ms()
                })
                .sum::<f32>()
                / seeds.clone().count() as f32
        };
        let mut scenario = ChooserScenario {
            rate: mean(None),
            forced: [0.0; 8],
            late: [0; 9],
        };
        for depth in 1..8 {
            scenario.forced[depth] = mean(Some(depth));
        }
        for seed in seeds {
            let run = chooser_run(window_us, ar_us, decay, 7, 60, None, seed * 7919, stream);
            for &depth in &run.depths[20..] {
                scenario.late[depth] += 1;
            }
        }
        scenario
    }

    /// The lock-in the chooser exists to escape: a code request whose first
    /// windows reject and are pruned to two drafts (the measured-only
    /// chooser then ran depth 2 for the whole request, as humaneval_0, lru
    /// and records_parser did on Halo) climbs to the measured optimum, 5,
    /// and never settles on the dearer 6 or 7.
    #[test]
    fn chooser_climbs_out_of_a_pruned_slow_start() {
        let code = |w: usize| if w < 4 { (0.3, 2) } else { (0.93, 7) };
        let s = chooser_scenario(
            HALO_WINDOW_US,
            HALO_AR_US,
            MTP_AGREEMENT_DECAY_GFX1151,
            code,
        );
        assert_eq!(s.best_depth(), 5, "{s:?}");
        assert!(s.late_share(|d| d == 5) >= 0.7, "{s:?}");
        // One-row probes past the optimum stay a small share of windows.
        assert!(s.late_share(|d| d >= 6) <= 0.08, "{s:?}");
        assert!(s.rate >= 0.95 * s.forced[5], "{s:?}");
        assert!(s.rate >= 1.1 * s.forced[2], "{s:?}");
    }

    /// Where deeper rows lose a budget of 7 stays shallow.  R9700 row prices
    /// at its forced-K measured acceptance (0.8: fixed depth 5 -8%, 7 -18%
    /// against 3, as measured there) run 4 or deeper only on near-ties and
    /// stay within a few percent of the best fixed depth; at prose
    /// acceptance (0.72) on either card the chooser stays at 2-3.
    #[test]
    fn chooser_stays_shallow_where_deeper_rows_lose() {
        let r9700 = chooser_scenario(R9700_WINDOW_US, R9700_AR_US, DEFAULT_DECAY, |_| (0.8, 7));
        assert!(matches!(r9700.best_depth(), 2 | 3), "{r9700:?}");
        assert!(r9700.forced[5] < 0.94 * r9700.forced[3], "{r9700:?}");
        assert!(
            r9700.late_share(|d| (5..8).contains(&d)) <= 0.1,
            "{r9700:?}"
        );
        // The calibration token and three table-depth probe windows are in
        // the chooser's rate, not in the fixed depths'.
        assert!(
            r9700.rate >= 0.93 * r9700.forced[r9700.best_depth()],
            "{r9700:?}"
        );
        let prose = |w: usize| if w < 4 { (0.3, 2) } else { (0.72, 7) };
        for s in [
            chooser_scenario(R9700_WINDOW_US, R9700_AR_US, DEFAULT_DECAY, |_| (0.72, 7)),
            chooser_scenario(
                HALO_WINDOW_US,
                HALO_AR_US,
                MTP_AGREEMENT_DECAY_GFX1151,
                prose,
            ),
        ] {
            assert!(matches!(s.best_depth(), 2 | 3), "{s:?}");
            assert!(s.late_share(|d| (1..=3).contains(&d)) >= 0.85, "{s:?}");
            assert!(s.late_share(|d| (5..8).contains(&d)) <= 0.03, "{s:?}");
            assert!(s.rate >= 0.93 * s.forced[s.best_depth()], "{s:?}");
        }
    }

    /// A prose-like stream the margin prune splits: confident spans draft
    /// the whole cap and accept 0.92 per draft, uncertain ones are cut to a
    /// single draft that is accepted 0.45 of the time.  A cap past the
    /// shallow optimum of the drafted-depth model costs the uncertain windows
    /// nothing, so the chooser reaches the best fixed cap with the prune on
    /// (Halo forced K=4 with the prune: prose +8.4%, where the drafted-depth
    /// chooser stayed at 2).
    #[test]
    fn chooser_prices_the_margin_prune_into_the_cap() {
        let mixed = |w: usize| {
            let h = (w as u32).wrapping_mul(2_654_435_761) >> 24;
            if h % 100 < 55 {
                (0.92, 7)
            } else {
                (0.45, 1)
            }
        };
        let halo = chooser_scenario(
            HALO_WINDOW_US,
            HALO_AR_US,
            MTP_AGREEMENT_DECAY_GFX1151,
            mixed,
        );
        assert!(halo.best_depth() >= 3, "{halo:?}");
        assert!(
            halo.rate >= 0.95 * halo.forced[halo.best_depth()],
            "{halo:?}"
        );
        assert!(
            halo.late_share(|d| d + 1 >= halo.best_depth() && d <= halo.best_depth() + 1) >= 0.7,
            "{halo:?}"
        );
        let r9700 = chooser_scenario(R9700_WINDOW_US, R9700_AR_US, DEFAULT_DECAY, mixed);
        assert!(
            r9700.rate >= 0.93 * r9700.forced[r9700.best_depth()],
            "{r9700:?}"
        );
    }

    /// The same rule finds depth where a dearer row still pays: R9700 row
    /// prices at code acceptance.
    #[test]
    fn chooser_goes_deep_on_r9700_costs_when_acceptance_pays() {
        let s = chooser_scenario(R9700_WINDOW_US, R9700_AR_US, DEFAULT_DECAY, |_| (0.93, 7));
        assert!(s.best_depth() >= 4, "{s:?}");
        assert!(s.late_share(|d| d >= 4 && d < 8) >= 0.6, "{s:?}");
        assert!(s.rate >= 0.95 * s.forced[s.best_depth()], "{s:?}");
    }

    /// Run the AR probe tokens a would-be retirement asks for, each costing
    /// `us`, asserting the route stays `ArSample` until the last.
    fn run_probes(floor: &mut Qwen4Floor, agreement: &[(f32, f32)], n: usize, us: f32) {
        for _ in 0..n {
            assert!(floor.needs_ar_sample());
            assert_eq!(floor.route(agreement, 3, 3), Action::ArSynced);
            floor.observe_calibration(us, agreement);
        }
    }

    /// The singleton path turns retire evidence on (default
    /// `MTP_SINGLETON_FLOOR_AR_PROBE_TOKENS`), the lane floor does not, and
    /// a pending probe blocks every takeover and is planned as an AR token.
    #[test]
    fn floor_retire_evidence_is_singleton_only_and_blocks_takeover_while_probing() {
        assert_eq!(
            Qwen4Floor::new(MTP_WINDOW_COST).diagnostics().probe_tokens,
            0
        );
        let mut drafter = floor_drafter(true);
        MtpDrafter::configure_request(&mut drafter, SpecRequestConfig::default());
        assert_eq!(
            drafter.floor.diagnostics().probe_tokens,
            singleton_floor_ar_probe_tokens()
        );
        assert_eq!(
            drafter.floor.diagnostics().probe_tokens,
            MTP_SINGLETON_FLOOR_AR_PROBE_TOKENS
        );
        drafter.floor.seed_calibrated(30_000.0);
        drafter.floor.seed_probe_left(2);
        assert!(drafter.floor_needs_calibration());
        assert!(drafter.floor_takeover_blocked());
        assert_eq!(
            plan_of(&mut drafter, 3, None).route,
            WindowRoute::ArCalibrate
        );
        drafter.floor.seed_probe_left(0);
        assert!(!drafter.floor_needs_calibration());
        assert!(!drafter.floor_takeover_blocked());
    }

    fn floor_drafter(floor_enabled: bool) -> Qwen4MtpDrafter {
        let mut drafter = Qwen4MtpDrafter::new(7, 4096, None);
        drafter.floor_enabled = floor_enabled;
        drafter
    }

    #[test]
    fn floor_optout_never_calibrates_or_blocks_takeover() {
        let drafter = floor_drafter(false);
        assert!(!drafter.floor.calibrated());
        assert!(!drafter.floor_needs_calibration());
        assert!(!drafter.floor_takeover_blocked());
        assert!(!drafter.floor_retired());
    }

    #[test]
    fn floor_optout_external_windows_never_stop_takeover_or_retire() {
        let mut drafter = floor_drafter(false);
        for _ in 0..(4 * FLOOR_PROBE_WINDOWS) {
            assert!(!drafter.floor_observe_external_window(0, f32::INFINITY, 4));
            assert!(!drafter.floor_observe_external_window(1, 1.0e9, 4));
        }
        assert!(!drafter.floor.retired());
        assert_eq!(drafter.floor.diagnostics().probes, 0);
        assert!(!drafter.request_stats().mtp_retired);
        assert_eq!(drafter.request_stats().ar_windows, 0);
        assert!(!drafter.floor_takeover_blocked());
    }

    #[test]
    fn floor_optout_native_windows_count_without_observing_the_floor() {
        let mut drafter = floor_drafter(false);
        let good = window(2, 2);
        for _ in 0..(2 * FLOOR_PROBE_WINDOWS) {
            drafter.floor_observe_native(true, NativeWindow::Batched, &good, 1.0e9, 3);
            drafter.floor_observe_native(true, NativeWindow::Interleaved, &good, 1.0e9, 3);
        }
        let stats = drafter.request_stats();
        assert_eq!(stats.mtp_windows, 4 * FLOOR_PROBE_WINDOWS);
        assert!(!stats.mtp_retired);
        assert_eq!(stats.ar_windows, 0);
        assert_eq!(drafter.floor.diagnostics().probes, 0);
    }

    /// The first batched window a drafter runs at a drafted depth (the
    /// process's first verify at that row count, one-time kernel work
    /// included) spends a probe but is not priced; later windows at that
    /// depth are, in this request and every later one.  Interleaved windows
    /// are always priced.
    #[test]
    fn floor_prices_no_first_window_at_a_drafted_depth() {
        let good = agreement(1.8, 2.0);
        let mut drafter = floor_drafter(true);
        drafter.floor.observe_calibration(30_000.0, &good);
        drafter.floor_observe_native(false, NativeWindow::Batched, &window(3, 3), 90_000.0, 7);
        assert_eq!(drafter.floor.diagnostics().batched[3], (0.0, 0.0));
        assert_eq!(drafter.floor.diagnostics().fit, [0.0; 5]);
        assert_eq!(drafter.floor.diagnostics().probes, 1);
        drafter.floor_observe_native(false, NativeWindow::Batched, &window(3, 3), 75_000.0, 7);
        assert_eq!(drafter.floor.diagnostics().batched[3], (75_000.0, 1.0));
        drafter.floor_observe_native(false, NativeWindow::Interleaved, &window(1, 2), 40_000.0, 7);
        assert_eq!(drafter.floor.diagnostics().interleaved, (40_000.0, 2.0));
        // A forced route (bypass) still runs the depth's first window.
        drafter.floor_observe_native(true, NativeWindow::Batched, &window(5, 5), 1.0e9, 7);
        MtpDrafter::configure_request(&mut drafter, SpecRequestConfig::default());
        drafter.floor.observe_calibration(30_000.0, &good);
        for depth in [3, 5] {
            drafter.floor_observe_native(
                false,
                NativeWindow::Batched,
                &window(depth, depth),
                75_000.0,
                7,
            );
            assert_eq!(
                drafter.floor.diagnostics().batched[depth],
                (75_000.0, 1.0),
                "depth {depth}"
            );
        }
    }

    #[test]
    fn floor_optout_control_floor_on_still_retires_on_terrible_external_windows() {
        let mut drafter = floor_drafter(true);
        let bad = agreement(0.0, 2.0);
        drafter.floor.observe_calibration(30.0, &bad);
        let mut stopped = false;
        for _ in 0..30 {
            if drafter.floor_observe_external_window(1, 1.0e9, 4) {
                stopped = true;
                break;
            }
        }
        assert!(stopped);
    }

    #[test]
    fn floor_optout_depth_is_the_pre_floor_batched_depth() {
        let mut drafter = floor_drafter(false);
        for (accepted, total) in [(0.0, 2.0), (1.0, 2.0), (1.8, 2.0), (2.0, 2.0)] {
            drafter.agreement.fill((accepted, total));
            for k in 0..=drafter.max_k {
                assert_eq!(drafter.floor_off_depth(k), drafter.batched_depth(k));
            }
        }
        // Perfect agreement pays for batching; none falls back to interleaved.
        drafter.agreement.fill((2.0, 2.0));
        assert!(drafter.floor_off_depth(drafter.max_k) > 0);
        drafter.agreement.fill((0.0, 2.0));
        assert_eq!(drafter.floor_off_depth(drafter.max_k), 0);
    }

    // ---- Resident Leviathan acceptance (`HIPFIRE_QWEN4_MTP_DEVICE_ACCEPT`):
    // the pure window law. The ordered-accept kernel is modeled on the host
    // with the host-law primitives driven by the explicit uniform tape the
    // device receives (`fill_uniform_tape`), and compared with
    // `accept_sampled_prefix` / `verify_sampled_draft` over the same
    // distributions: committed tokens, accepted count, EOS flag and the
    // authoritative stream's later outputs. No GPU is involved.

    const DEV_NO_EOS: u32 = 99;

    fn dev_spec(top_k: Option<u32>) -> SampleSpec {
        SampleSpec::cpu_ar(1.0, 1.0, top_k, 0.0)
    }

    fn dev_logits(salt: u32) -> Vec<f32> {
        (0..10u32)
            .map(|token| {
                let hash = token
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(salt.wrapping_mul(40_503))
                    >> 7;
                (hash & 0xff) as f32 / 32.0
            })
            .collect()
    }

    /// All mass on `top` (every other probability below f32 resolution).
    fn dev_peaked(top: usize) -> Vec<f32> {
        let mut logits = vec![0.0f32; 10];
        logits[top] = 30.0;
        logits
    }

    fn dev_dist(logits: &[f32], spec: SampleSpec) -> SparseDist {
        let mut dist = SparseDist::default();
        dist.build_from_logits(logits, spec, &mut Vec::new())
            .expect("test logits have finite mass");
        dist
    }

    fn dev_point(token: u32) -> SparseDist {
        let mut dist = SparseDist::default();
        dist.set_point_mass(token);
        dist
    }

    /// Host model of `spec_accept_ordered_u32`: row `j`'s verdict reads
    /// `tape[j]`, its residual `tape[j + 1]`, the bonus `tape[k]`; an accepted
    /// proposal equal to `eos` stops without a bonus (`terminal = u32::MAX`).
    fn dev_simulate_accept(
        target: &[SparseDist],
        qs: &[SparseDist],
        proposals: &[u32],
        tape: &[f32],
        eos: u32,
    ) -> [u32; 2] {
        let k = proposals.len();
        for row in 0..k {
            let (p, q, x) = (&target[row], &qs[row], proposals[row]);
            let (qx, px) = (q.prob(x), p.prob(x));
            if qx > 0.0 && tape[row] * qx < px {
                if x == eos {
                    return [(row + 1) as u32, SPEC_CPU_AR_EOS_NONE];
                }
            } else {
                return [row as u32, sample_residual(p, q, tape[row + 1])];
            }
        }
        [k as u32, target[k].sample(tape[k])]
    }

    /// One batched window: the device reconstruction against the host
    /// accept over the same rows and the same stream.
    fn dev_check_window(
        target: &[SparseDist],
        qs: &[SparseDist],
        drafts: &[u32],
        eos: u32,
        seed: u64,
    ) -> DeviceVerdict {
        let k = drafts.len();
        assert_eq!(target.len(), k + 1);
        let base = SpecRng::new(seed);
        let mut host_rng = base;
        let mut scratch = SparseDist::default();
        let want = accept_sampled_prefix(
            drafts,
            qs,
            Some(eos),
            &mut host_rng,
            &mut scratch,
            |row, out| {
                *out = target[row].clone();
                Ok(())
            },
        )
        .unwrap();
        let mut tape = vec![0.0f32; k + 1];
        fill_uniform_tape(&base, &mut tape);
        let words = dev_simulate_accept(target, qs, drafts, &tape, eos);
        let verdict = decode_device_verdict(drafts, words, eos).unwrap();
        let got = device_accept_result(drafts, &verdict);
        assert_eq!(got.committed, want.committed, "seed {seed}");
        assert_eq!(got.accepted, want.accepted, "seed {seed}");
        assert_eq!(got.hit_eos, want.hit_eos, "seed {seed}");
        let mut device_rng = base;
        advance_rng(&mut device_rng, verdict.consumed);
        for _ in 0..4 {
            assert_eq!(device_rng.next_u64(), host_rng.next_u64(), "seed {seed}");
        }
        verdict
    }

    /// `q == p` over the full pool on every row; proposals are each row's
    /// most probable token, so every one is accepted for any `u < 1`.
    fn dev_matched(k: usize) -> (Vec<SparseDist>, Vec<SparseDist>, Vec<u32>) {
        let spec = dev_spec(None);
        let target: Vec<SparseDist> = (0..=k as u32)
            .map(|row| dev_dist(&dev_logits(row + 1), spec))
            .collect();
        let qs = target[..k].to_vec();
        let drafts = (0..k).map(|row| target[row].entries()[0].0).collect();
        (target, qs, drafts)
    }

    /// Like [`dev_matched`] but row `at`'s `p` keeps two candidates while its
    /// `q` has the full pool, and its proposal lies outside `p`'s support:
    /// rejected for every `u`.
    fn dev_reject_at(k: usize, at: usize) -> (Vec<SparseDist>, Vec<SparseDist>, Vec<u32>) {
        let (mut target, mut qs, mut drafts) = dev_matched(k);
        let logits = dev_logits(at as u32 + 1);
        target[at] = dev_dist(&logits, dev_spec(Some(2)));
        qs[at] = dev_dist(&logits, dev_spec(None));
        drafts[at] = qs[at]
            .entries()
            .iter()
            .map(|entry| entry.0)
            .find(|&token| target[at].prob(token) == 0.0)
            .expect("a full pool holds a token outside a two-candidate pool");
        (target, qs, drafts)
    }

    #[test]
    fn device_accept_all_accepted_takes_the_bonus_and_consumes_k_plus_one() {
        let (target, qs, drafts) = dev_matched(4);
        for seed in 0..32 {
            let verdict = dev_check_window(&target, &qs, &drafts, DEV_NO_EOS, seed);
            assert_eq!(verdict.accepted, 4);
            assert_eq!(verdict.consumed, 5);
            assert!(verdict.terminal.is_some());
            assert!(!verdict.hit_eos);
        }
    }

    #[test]
    fn device_accept_first_and_last_rejection_consume_accepted_plus_two() {
        for (k, at) in [(4, 0), (4, 3), (3, 1)] {
            let (target, qs, drafts) = dev_reject_at(k, at);
            for seed in 0..32 {
                let verdict = dev_check_window(&target, &qs, &drafts, DEV_NO_EOS, seed);
                assert_eq!(verdict.accepted, at, "k {k} reject at {at}");
                assert_eq!(verdict.consumed, at + 2);
                let replacement = verdict.terminal.expect("a rejection carries a replacement");
                assert!(target[at].prob(replacement) > 0.0);
                let committed = device_accept_result(&drafts, &verdict).committed;
                assert_eq!(committed.len(), at + 1);
                assert_eq!(committed.last(), Some(&replacement));
            }
        }
    }

    #[test]
    fn device_accept_stops_at_an_accepted_eos_without_a_bonus() {
        let (target, qs, drafts) = dev_matched(4);
        let eos = drafts[1];
        let stop = drafts.iter().position(|&draft| draft == eos).unwrap();
        for seed in 0..32 {
            let verdict = dev_check_window(&target, &qs, &drafts, eos, seed);
            assert_eq!(verdict.accepted, stop + 1);
            assert_eq!(verdict.terminal, None);
            assert_eq!(verdict.consumed, stop + 1);
            assert!(verdict.hit_eos);
            let committed = device_accept_result(&drafts, &verdict).committed;
            assert_eq!(committed.len(), stop + 1);
            assert_eq!(committed.last(), Some(&eos));
        }
    }

    #[test]
    fn device_accept_rejection_replacement_that_is_the_eos_hits_eos() {
        // p = {0, 1} at one half each, q the full pool with 0 and 1 level:
        // token 2 is outside p, rejected, and the residual is 0 or 1.
        let mut logits0 = vec![0.0f32; 10];
        logits0[0] = 5.0;
        logits0[1] = 5.0;
        let target = vec![
            dev_dist(&logits0, dev_spec(Some(2))),
            dev_dist(&dev_logits(9), dev_spec(None)),
        ];
        let qs = vec![dev_dist(&logits0, dev_spec(None))];
        let drafts = [2u32];
        let (mut hit, mut miss) = (false, false);
        for seed in 0..256 {
            let verdict = dev_check_window(&target, &qs, &drafts, 0, seed);
            assert_eq!(verdict.accepted, 0);
            assert_eq!(verdict.consumed, 2);
            match verdict.terminal {
                Some(0) => {
                    hit = true;
                    assert!(verdict.hit_eos);
                }
                Some(_) => {
                    miss = true;
                    assert!(!verdict.hit_eos);
                }
                None => panic!("a rejection has a terminal token"),
            }
        }
        assert!(hit && miss, "both residual outcomes occur across 256 seeds");
    }

    #[test]
    fn device_accept_point_mass_drafts_match_the_host_law() {
        let tops = [2usize, 5, 3];
        let mut target: Vec<SparseDist> = tops
            .iter()
            .map(|&top| dev_dist(&dev_peaked(top), dev_spec(None)))
            .collect();
        target.push(dev_dist(&dev_peaked(6), dev_spec(None)));
        let accepted_drafts: Vec<u32> = tops.iter().map(|&top| top as u32).collect();
        let qs: Vec<SparseDist> = accepted_drafts
            .iter()
            .map(|&token| dev_point(token))
            .collect();
        for seed in 0..32 {
            let verdict = dev_check_window(&target, &qs, &accepted_drafts, DEV_NO_EOS, seed);
            assert_eq!(verdict.accepted, 3);
            assert_eq!(verdict.terminal, Some(6));
            assert_eq!(verdict.consumed, 4);
        }
        // The last proposal is not the target's token: rejected, replaced by it.
        let mut rejected = accepted_drafts.clone();
        rejected[2] = 4;
        let qs: Vec<SparseDist> = rejected.iter().map(|&token| dev_point(token)).collect();
        for seed in 0..32 {
            let verdict = dev_check_window(&target, &qs, &rejected, DEV_NO_EOS, seed);
            assert_eq!(verdict.accepted, 2);
            assert_eq!(verdict.terminal, Some(3));
            assert_eq!(verdict.consumed, 4);
        }
    }

    #[test]
    fn device_accept_single_proposal_windows_match_the_host_law() {
        let (target, qs, drafts) = dev_matched(1);
        for seed in 0..32 {
            let verdict = dev_check_window(&target, &qs, &drafts, DEV_NO_EOS, seed);
            assert_eq!((verdict.accepted, verdict.consumed), (1, 2));
        }
        let (target, qs, drafts) = dev_reject_at(1, 0);
        for seed in 0..32 {
            let verdict = dev_check_window(&target, &qs, &drafts, DEV_NO_EOS, seed);
            assert_eq!((verdict.accepted, verdict.consumed), (0, 2));
        }
    }

    /// The incremental one-row verdict: the kernel is told `eos = proposal`
    /// (so an accepted proposal never reads a bonus row) and the caller
    /// applies the real EOS rule. Target row 1 is poisoned: it is never read.
    fn dev_check_step(
        p: &SparseDist,
        q: &SparseDist,
        draft: u32,
        real_eos: u32,
        seed: u64,
    ) -> (DraftVerdict, DeviceVerdict) {
        let base = SpecRng::new(seed);
        let mut host_rng = base;
        let want = verify_sampled_draft(p, q, draft, &mut host_rng);
        let mut tape = [0.0f32; 2];
        fill_uniform_tape(&base, &mut tape);
        let target = [p.clone(), dev_point(9)];
        let words = dev_simulate_accept(&target, std::slice::from_ref(q), &[draft], &tape, draft);
        let verdict = decode_device_verdict(&[draft], words, real_eos).unwrap();
        let got = match verdict.terminal {
            None => DraftVerdict::Accept,
            Some(token) => DraftVerdict::Reject(token),
        };
        assert_eq!(got, want, "seed {seed}");
        let mut device_rng = base;
        advance_rng(&mut device_rng, verdict.consumed);
        for _ in 0..4 {
            assert_eq!(device_rng.next_u64(), host_rng.next_u64(), "seed {seed}");
        }
        (want, verdict)
    }

    #[test]
    fn device_incremental_verdict_real_eos_proposal_accepted_stops_and_consumes_one() {
        let p = dev_dist(&dev_peaked(4), dev_spec(None));
        let q = dev_point(4);
        for seed in 0..16 {
            let (verdict, decoded) = dev_check_step(&p, &q, 4, 4, seed);
            assert_eq!(verdict, DraftVerdict::Accept);
            assert_eq!(decoded.consumed, 1);
            assert_eq!(decoded.terminal, None);
            assert!(decoded.hit_eos, "the proposal is the request EOS");
        }
    }

    #[test]
    fn device_incremental_verdict_artificial_stop_is_not_an_eos_hit() {
        let p = dev_dist(&dev_peaked(4), dev_spec(None));
        let q = dev_point(4);
        for seed in 0..16 {
            let (verdict, decoded) = dev_check_step(&p, &q, 4, DEV_NO_EOS, seed);
            assert_eq!(verdict, DraftVerdict::Accept);
            assert_eq!(decoded.consumed, 1);
            assert!(
                !decoded.hit_eos,
                "a kernel-only stop is not the request EOS"
            );
        }
    }

    #[test]
    fn device_incremental_verdict_rejection_consumes_two_and_flags_a_replacement_eos() {
        // The proposal 2 is not the target's token 4: rejected, replaced by 4.
        let p = dev_dist(&dev_peaked(4), dev_spec(None));
        let q = dev_point(2);
        for seed in 0..16 {
            let (verdict, decoded) = dev_check_step(&p, &q, 2, 4, seed);
            assert_eq!(verdict, DraftVerdict::Reject(4));
            assert_eq!(decoded.consumed, 2);
            assert!(
                decoded.hit_eos,
                "the residual replacement is the request EOS"
            );
            let (_, decoded) = dev_check_step(&p, &q, 2, DEV_NO_EOS, seed);
            assert_eq!(decoded.consumed, 2);
            assert!(!decoded.hit_eos);
            // A rejected proposal that is itself the request EOS is not an EOS hit.
            let (_, decoded) = dev_check_step(&p, &q, 2, 2, seed);
            assert!(!decoded.hit_eos);
        }
    }

    #[test]
    fn device_incremental_verdict_matches_the_host_law_on_general_rows() {
        let spec = dev_spec(None);
        for salt in 0..6u32 {
            let p = dev_dist(&dev_logits(salt + 1), dev_spec(Some(3)));
            let q = dev_dist(&dev_logits(salt + 20), spec);
            for seed in 0..24 {
                // Draw the proposal from q, as the window does.
                let mut rng = SpecRng::new(seed + 1000);
                let draft = q.sample(rng.next_f32());
                let _ = dev_check_step(&p, &q, draft, DEV_NO_EOS, seed);
                let _ = dev_check_step(&p, &q, draft, draft, seed);
            }
        }
    }

    #[test]
    fn device_bonus_draws_from_row_zero_and_consumes_one() {
        for salt in 0..6u32 {
            let p = dev_dist(&dev_logits(salt + 3), dev_spec(None));
            for seed in 0..24 {
                let base = SpecRng::new(seed);
                let mut host_rng = base;
                let want = p.sample(host_rng.next_f32());
                let mut tape = [0.0f32; 1];
                fill_uniform_tape(&base, &mut tape);
                let words = dev_simulate_accept(
                    std::slice::from_ref(&p),
                    &[],
                    &[],
                    &tape,
                    SPEC_CPU_AR_EOS_NONE,
                );
                let verdict = decode_device_verdict(&[], words, SPEC_CPU_AR_EOS_NONE).unwrap();
                assert_eq!(verdict.terminal, Some(want));
                assert_eq!(verdict.consumed, 1);
                let mut device_rng = base;
                advance_rng(&mut device_rng, verdict.consumed);
                assert_eq!(device_rng.next_u64(), host_rng.next_u64());
            }
        }
        assert!(decode_device_verdict(&[], [0, SPEC_CPU_AR_EOS_NONE], DEV_NO_EOS).is_err());
    }

    #[test]
    fn device_verdict_decoding_rejects_impossible_words() {
        let drafts = [5u32, 6, 7];
        let reject = decode_device_verdict(&drafts, [1, 4], DEV_NO_EOS).unwrap();
        assert_eq!(
            (reject.accepted, reject.consumed, reject.terminal),
            (1, 3, Some(4))
        );
        let bonus = decode_device_verdict(&drafts, [3, 4], DEV_NO_EOS).unwrap();
        assert_eq!(
            (bonus.accepted, bonus.consumed, bonus.terminal),
            (3, 4, Some(4))
        );
        let stop = decode_device_verdict(&drafts, [2, SPEC_CPU_AR_EOS_NONE], 6).unwrap();
        assert_eq!(
            (stop.accepted, stop.consumed, stop.terminal, stop.hit_eos),
            (2, 2, None, true)
        );
        let result = device_accept_result(&drafts, &stop);
        assert_eq!(
            (result.committed.as_slice(), result.accepted, result.hit_eos),
            (&[5u32, 6][..], 2, true)
        );
        assert!(decode_device_verdict(&drafts, [4, 1], DEV_NO_EOS).is_err());
        assert!(decode_device_verdict(&drafts, [0, SPEC_CPU_AR_EOS_NONE], DEV_NO_EOS).is_err());
    }

    #[test]
    fn uniform_tape_is_the_next_draws_and_leaves_the_stream_alone() {
        let base = SpecRng::new(77);
        let mut tape = [0.0f32; 5];
        fill_uniform_tape(&base, &mut tape);
        let mut walk = base;
        for &slot in &tape {
            assert_eq!(slot, walk.next_f32());
        }
        let mut untouched = base;
        let mut reference = base;
        assert_eq!(untouched.next_u64(), reference.next_u64());
    }

    #[test]
    fn device_accept_eligibility_is_neutral_leviathan_only() {
        let spec = SampleSpec::cpu_ar(0.7, 0.9, None, 0.0);
        assert!(device_accept_eligible(false, 0, false, true, &spec));
        assert!(
            !device_accept_eligible(true, 0, false, true, &spec),
            "naive"
        );
        assert!(
            !device_accept_eligible(false, 64, false, true, &spec),
            "history window"
        );
        assert!(
            !device_accept_eligible(false, 0, true, true, &spec),
            "gpu penalty rows"
        );
        assert!(
            !device_accept_eligible(false, 0, false, false, &spec),
            "blocked tokens"
        );
        for temperature in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let bad = SampleSpec {
                temperature,
                ..spec
            };
            assert!(
                !device_accept_eligible(false, 0, false, true, &bad),
                "{temperature}"
            );
        }
        let wide = SampleSpec::cpu_ar(0.7, 0.9, Some(0), 0.0);
        assert!(device_accept_eligible(false, 0, false, true, &wide));
        assert!(!device_accept_eligible(
            false,
            0,
            false,
            true,
            &SampleSpec {
                pool: 65,
                cap: 65,
                ..spec
            }
        ));
        assert!(!device_accept_eligible(
            false,
            0,
            false,
            true,
            &SampleSpec { cap: 0, ..spec }
        ));
        assert!(!device_accept_eligible(
            false,
            0,
            false,
            true,
            &SampleSpec {
                cap: spec.pool + 1,
                ..spec
            }
        ));
    }

    #[test]
    fn sampled_verify_device_predicate_follows_request_and_resources() {
        let neutral = SpecRequestConfig {
            temp: 0.7,
            ..SpecRequestConfig::default()
        };
        let s = SampledVerify::new(neutral, 4, SampledMode::Leviathan);
        // Eligible by request, but nothing is provisioned without a `Gpu`.
        assert!(s.device_eligible || !mtp_device_accept_enabled());
        for k in 0..=4 {
            assert!(!s.device_ready(k));
        }
        assert!(!SampledVerify::new(neutral, 4, SampledMode::Naive).device_eligible);
        let penalized = SpecRequestConfig {
            repeat_penalty: 1.15,
            repeat_window: 64,
            ..neutral
        };
        assert!(!SampledVerify::new(penalized, 4, SampledMode::Leviathan).device_eligible);
        assert_eq!(accept_route_label(None), "host");
        assert_eq!(accept_route_label(Some(&s)), "host");
    }

    #[cfg(feature = "lab")]
    #[test]
    fn lab_window_hooks_do_nothing_unarmed_and_never_move_the_stream() {
        use crate::lab_window::{LabVerdict, LabVerifyRoute, LabWindow};
        let cfg = SpecRequestConfig {
            temp: 0.7,
            ..SpecRequestConfig::default()
        };
        let rng_word = |rng: &SpecRng| format!("{rng:?}");
        // Unarmed: the hooks leave no record and draw exactly what the plain
        // acceptance primitives draw.
        let mut plain = SampledVerify::new(cfg, 4, SampledMode::Leviathan);
        assert!(plain.lab.is_none());
        plain.target.set_point_mass(5);
        plain.drafts[0].set_point_mass(5);
        let mut reference = plain.rng;
        let verdict = plain.host_step_verdict(5, false);
        assert_eq!(
            verdict,
            verify_sampled_draft(&dev_point(5), &dev_point(5), 5, &mut reference)
        );
        assert_eq!(plain.host_bonus(), 5);
        reference.next_f32();
        assert!(plain.lab.is_none(), "an unarmed hook records nothing");
        assert_eq!(rng_word(&plain.rng), rng_word(&reference));
        // Armed: the same draws, observed.
        let mut armed = SampledVerify::new(cfg, 4, SampledMode::Leviathan);
        armed.target.set_point_mass(5);
        armed.drafts[0].set_point_mass(5);
        let start = armed.rng;
        armed.lab = Some(Box::new(LabWindow::begin(
            0,
            1,
            "Native { depth: 0 }".to_string(),
            4,
            false,
            LabVerifyRoute::Interleaved,
            true,
            false,
            Some(start),
            0,
            Default::default(),
            Vec::new(),
        )));
        armed
            .lab
            .as_mut()
            .expect("armed")
            .note_row(0, armed.target.entries(), false);
        assert_eq!(armed.host_step_verdict(5, false), verdict);
        armed
            .lab
            .as_mut()
            .expect("armed")
            .note_row(1, armed.target.entries(), false);
        assert_eq!(armed.host_bonus(), 5);
        assert_eq!(
            rng_word(&armed.rng),
            rng_word(&reference),
            "observing moved no draw"
        );
        let window = armed.lab.take().expect("armed");
        let tape = crate::lab_window::peek_uniforms(&start, 2);
        assert_eq!(window.proposals.len(), 1);
        assert_eq!(window.proposals[0].token, 5);
        assert_eq!(
            window.proposals[0].verdict,
            Some(LabVerdict::Accept {
                draw: crate::lab_window::LabDraw {
                    index: 0,
                    u: tape[0]
                }
            })
        );
        let bonus = window.bonus.expect("bonus noted");
        assert_eq!((bonus.row, bonus.token, bonus.draw.index), (1, 5, 1));
        assert_eq!(bonus.draw.u, tape[1]);
        assert_eq!(window.rows.len(), 2);
    }

    #[cfg(feature = "lab")]
    #[test]
    fn lab_drafter_arm_is_off_by_default_and_disarming_drops_observations() {
        use crate::lab_window::LabWindowArm;
        let mut drafter = Qwen4MtpDrafter::new(4, 128, None);
        assert!(!drafter.lab_arm.armed);
        assert!(drafter.lab_take_window_observation().is_none());
        assert!(drafter.lab_take_window_observations().is_empty());
        drafter.lab_observe_windows(LabWindowArm {
            armed: true,
            keep_logits: true,
        });
        assert!(drafter.lab_arm.armed && drafter.lab_arm.keep_logits);
        drafter.lab_observe_windows(LabWindowArm::default());
        assert!(!drafter.lab_arm.armed);
        assert!(drafter.lab_open.is_none() && drafter.lab_done.is_empty());
    }

    /// An n-gram acceptance table with `ratio` at every depth backed by
    /// `weight` windows of evidence.
    fn ngram_table(ratio: f32, weight: f32) -> Vec<(f32, f32)> {
        vec![(ratio * weight, weight); NGRAM_DEPTHS]
    }

    #[test]
    fn ngram_take_runs_only_while_its_acceptance_beats_native() {
        let emitted = [0u32, 1];
        // A fresh request (the prior's 0.8 at the prior's weight) prices the
        // pool below native MTP agreeing at 0.9, but the proposal is still
        // kept for the shadow observation.
        let mut drafter = armed_copy_drafter(1, 8);
        drafter.agreement.fill((0.9, 1.0));
        drafter.ngram_agreement.fill(AGREEMENT_PRIOR);
        assert!(!drafter.ngram_hit(&emitted, 8));
        assert_eq!(drafter.ngram_proposed_at, Some(2));
        assert_eq!(drafter.ngram_proposal.len(), 8);
        assert_eq!(drafter.ngram_decision.map(|d| d.take), Some(0));
        // A source measured at the native acceptance pays from a long take
        // (the best rate sits at 6 of the 7 candidates the cap allows)...
        drafter.ngram_agreement = ngram_table(0.9, 20.0);
        assert!(drafter.ngram_hit(&emitted, 8));
        assert_eq!(drafter.ngram_candidates.len(), 6);
        // ... a measured 0.8 source does not ...
        drafter.ngram_agreement = ngram_table(0.8, 20.0);
        assert!(!drafter.ngram_hit(&emitted, 8));
        // ... and the budget `k` bounds the take.
        drafter.ngram_agreement = ngram_table(0.99, 20.0);
        assert!(drafter.ngram_hit(&emitted, 4));
        assert_eq!(drafter.ngram_candidates.len(), 4);
    }

    #[test]
    fn ngram_take_length_follows_the_arch_row_price() {
        let emitted = [0u32, 1];
        for ((single, extra), takes) in [((30.5, 6.9), true), ((30.3, 14.8), false)] {
            let mut drafter = armed_copy_drafter(1, 8);
            drafter.ngram_row_cost = Some((single, extra));
            drafter.agreement.fill((0.92, 1.0));
            drafter.ngram_agreement = ngram_table(0.9, 20.0);
            // Halo's cheap rows pay for the long take; the R9700's dearer
            // rows leave the 0.9 source below the native window at every
            // length.
            assert_eq!(drafter.ngram_hit(&emitted, 8), takes, "{single} {extra}");
            if takes {
                assert_eq!(drafter.ngram_candidates.len(), 6);
            }
        }
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None);
        drafter.ngram_row_cost = Some((30.3, 14.8));
        assert!((drafter.ngram_window_cost(3) - 74.7 / 30.3).abs() < 1e-5);
        drafter.ngram_row_cost = Some((30.5, 6.9));
        assert!((drafter.ngram_window_cost(3) - 51.2 / 30.5).abs() < 1e-5);
        assert!(drafter.ngram_window_cost(8).is_infinite());
    }

    #[test]
    fn ngram_pricing_uses_request_measurements_after_probing_copy_lane() {
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None).with_copy_lane(true);
        drafter.ngram_row_cost = Some((30.3, 14.8));
        drafter.floor.seed_ar_us(30_000.0);
        drafter.floor.seed_batched(3, (240_000.0, 2.0));
        assert!(drafter.floor.batched_cost(3).is_none());
        assert!(drafter.floor.takeover_cost(3).is_none());
        drafter.floor.seed_probes_done();
        assert_eq!(drafter.floor.batched_cost(3), Some(4.0));
        assert_eq!(drafter.ngram_window_cost(3), 4.0);
        // A takeover with no measured takeover window is priced as the
        // native window of its row count.
        assert_eq!(drafter.takeover_cost(3), 4.0);
        drafter.floor.seed_external_depth(3, (60_000.0, 2.0));
        assert_eq!(drafter.takeover_cost(3), 1.0);
        drafter.floor_enabled = false;
        assert!((drafter.takeover_cost(3) - 74.7 / 30.3).abs() < 1e-5);
        drafter.ngram_row_cost = None;
        assert_eq!(drafter.ngram_window_cost(3), MTP_WINDOW_COST[3]);
        assert_eq!(drafter.takeover_cost(3), MTP_WINDOW_COST[3]);
    }

    #[test]
    fn ngram_take_length_stops_where_the_chain_stops_paying() {
        let emitted = [0u32, 1];
        let mut drafter = armed_copy_drafter(1, 8);
        drafter.agreement.fill((0.9, 1.0));
        // Depths 1-3 accept at 0.99, deeper ones at 0.3: the fourth row adds
        // 0.29 tokens for a 0.55 AR-step dearer window.
        let mut table = ngram_table(0.99, 20.0);
        for entry in &mut table[3..] {
            *entry = (6.0, 20.0);
        }
        drafter.ngram_agreement = table;
        assert!(drafter.ngram_hit(&emitted, 8));
        assert_eq!(drafter.ngram_candidates, vec![2, 3, 4]);
        // The chosen take is the one the decision recorded.
        let decision = drafter.ngram_decision.expect("decision");
        assert_eq!((decision.available, decision.take), (8, 3));
        assert!(decision.rate >= decision.native_rate);
    }

    /// A drafter whose pool is trained on `0..21` (so `[0, 1]` continues as
    /// `[2, 3, 4, ..]`) and whose n-gram source is measured confident and
    /// well-established (0.99 over 20 windows) on the static Halo prices
    /// (floor off): the chooser takes the longest chain that pays.
    fn armed_copy_drafter(n_min: usize, n_max: usize) -> Qwen4MtpDrafter {
        let config = NgramModConfig {
            capacity: 1024,
            n_match: 2,
            n_min,
            n_max,
        };
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None)
            .with_ngram(Some(config))
            .with_copy_lane(true);
        let mut pool = MtpCopyContext::new(config).expect("pool");
        pool.begin_request(&(0..21u32).collect::<Vec<_>>());
        drafter.ngram_copy = Some(pool);
        drafter.ngram_active = true;
        drafter.floor_enabled = false;
        drafter.ngram_row_cost = Some((30.5, 6.9));
        drafter.ngram_agreement = ngram_table(0.99, 20.0);
        drafter
    }

    #[test]
    fn plan_window_checks_retirement_then_calibration_before_the_ngram_pool_copy_lane() {
        // A retired request still asks the pool: native drafting lost to AR,
        // not the copy source, which must beat AR's 1.0 token per step.
        let mut drafter = armed_copy_drafter(1, 8);
        drafter.floor_enabled = true;
        drafter.floor.seed_calibrated(30_000.0);
        drafter.floor.seed_probes_done();
        drafter.floor.seed_retired(true);
        // Retired and cold (no shadow observation has accepted a first
        // candidate yet): the pool proposes, the shadow state is set, and
        // nothing is taken.
        let plan = plan_of(&mut drafter, 3, None);
        assert_eq!(plan.route, WindowRoute::ArRetired);
        assert!(plan.floor_active && !plan.ngram_hit);
        assert_eq!(drafter.ngram_proposed_at, Some(2));
        // Once the last observation accepted its first candidate, it takes.
        drafter.ngram_last_first_accepted = true;
        let mut plan = plan_of(&mut drafter, 3, None);
        assert_eq!(plan.route, WindowRoute::ArRetired);
        assert!(plan.floor_active && plan.ngram_hit);
        assert_eq!(drafter.ngram_proposed_at, Some(2));
        assert_eq!(drafter.ngram_decision.map(|d| d.native_rate), Some(1.0));
        // The replicas get the candidates on demand.
        drafter.attach_candidates(&mut plan);
        assert!(!plan.ngram_candidates.is_empty());
        assert_eq!(plan.ngram_candidates, drafter.ngram_candidates);
        // A source that loses to AR itself is declined.
        drafter.ngram_agreement = ngram_table(0.0, 20.0);
        let plan = plan_of(&mut drafter, 3, None);
        assert_eq!(plan.route, WindowRoute::ArRetired);
        assert!(!plan.ngram_hit && plan.ngram_confirm);

        // The request's first window calibrates AR before any pool hit.
        let mut drafter = armed_copy_drafter(1, 8);
        drafter.floor_enabled = true;
        let plan = plan_of(&mut drafter, 3, None);
        assert_eq!(plan.route, WindowRoute::ArCalibrate);
        assert!(!plan.ngram_hit && drafter.ngram_candidates.is_empty());
        assert!(drafter.ngram_proposed_at.is_none() && drafter.ngram_decision.is_none());

        // A developer-forced route bypasses the floor entirely.
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None).with_copy_lane(true);
        let plan = plan_of(&mut drafter, 3, Some("1"));
        assert_eq!(plan.route, WindowRoute::Native { depth: 0 });
        assert!(!plan.floor_active);
        let plan = plan_of(&mut drafter, 2, Some("0"));
        assert_eq!(plan.route, WindowRoute::Native { depth: 2 });
        assert_eq!(plan.k, 2);
        assert!(!drafter.floor.calibrated() && !drafter.floor.retired());
    }

    #[test]
    fn plan_window_runs_the_ngram_gate_exactly_as_the_old_chain_and_shares_candidates_on_demand_copy_lane(
    ) {
        // Floor off: the pre-floor chooser; the hit is the pool's.
        let mut drafter = armed_copy_drafter(1, 8);
        let mut plan = plan_of(&mut drafter, 3, None);
        assert!(plan.ngram_hit && !plan.floor_active);
        assert_eq!(
            plan.route,
            WindowRoute::Native {
                depth: batched_depth(&drafter.agreement, 3, &MTP_WINDOW_COST)
            }
        );
        // Single card allocates nothing for the plan; EP attaches the copy.
        assert!(plan.ngram_candidates.is_empty());
        drafter.attach_candidates(&mut plan);
        assert!(!plan.ngram_candidates.is_empty());
        assert_eq!(plan.ngram_candidates, drafter.ngram_candidates);

        // A declined proposal is no takeover, but it stays set for the
        // shadow observation and the plan's confirm flag defaults safe.
        let mut drafter = armed_copy_drafter(1, 8);
        drafter.ngram_agreement = ngram_table(0.1, 20.0);
        let plan = plan_of(&mut drafter, 3, None);
        assert!(!plan.ngram_hit && plan.ngram_confirm);
        assert_eq!(drafter.ngram_proposed_at, Some(2));
        assert_eq!(
            drafter.ngram_decision.map(|d| (d.available, d.take)),
            Some((8, 0))
        );
        let mut declined = plan.clone();
        drafter.attach_candidates(&mut declined);
        assert!(declined.ngram_candidates.is_empty());
    }

    #[test]
    fn singleton_decay_is_long_on_gfx1151_only_and_reaches_the_request_floor_copy_lane() {
        assert_eq!(DEFAULT_DECAY, 0.875);
        assert_eq!(MTP_AGREEMENT_DECAY_GFX1151, 0.97);
        assert_eq!(singleton_agreement_decay("gfx1151"), 0.97);
        for arch in [
            "",
            "gfx1150",
            "gfx1152",
            "gfx1201",
            "gfx1200",
            "gfx1100",
            "gfx942",
            "gfx1151:sramecc+",
            "GFX1151",
        ] {
            assert_eq!(singleton_agreement_decay(arch), 0.875, "{arch}");
        }
        assert_eq!(Qwen4Floor::new(MTP_WINDOW_COST).diagnostics().decay, 0.875);
        // Binding an arch selects one decay for agreement and every floor
        // cost pair; configuring another request must preserve that choice.
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None);
        assert_eq!(drafter.agreement_decay, 0.875);
        for arch in ["gfx1151", "gfx1201", "gfx1151", "gfx942"] {
            let decay = singleton_agreement_decay(arch);
            drafter.agreement_decay = decay;
            drafter.agreement.fill(AGREEMENT_PRIOR);
            MtpDrafter::configure_request(&mut drafter, SpecRequestConfig::default());
            assert_eq!(drafter.floor.diagnostics().decay, decay, "{arch}");
            drafter.observe_agreement(&window(0, 1));
            let (prior_accepted, prior_total) = AGREEMENT_PRIOR;
            assert_eq!(
                drafter.agreement[0],
                (prior_accepted * decay, prior_total * decay + 1.0),
                "{arch}"
            );
            drafter.floor.observe_calibration(100.0, &drafter.agreement);
            let mut combined = (0.0, 0.0);
            for route in [NativeWindow::Interleaved, NativeWindow::Batched] {
                for _ in 0..2 {
                    drafter
                        .floor
                        .observe_native(route, &window(1, 1), 40.0, 3, &drafter.agreement);
                    combined = (combined.0 * decay + 40.0, combined.1 * decay + 2.0);
                }
            }
            assert_eq!(drafter.floor.diagnostics().combined, combined);
            assert_eq!(
                drafter.floor.diagnostics().interleaved,
                (40.0 * decay + 40.0, 2.0 * decay + 2.0)
            );
            assert_eq!(
                drafter.floor.diagnostics().batched[1],
                (40.0 * decay + 40.0, decay + 1.0)
            );
            for _ in 0..2 {
                assert!(!drafter
                    .floor
                    .observe_external_copy(2, 40.0, 1, &drafter.agreement));
                combined = (combined.0 * decay + 40.0, combined.1 * decay + 2.0);
            }
            assert_eq!(drafter.floor.diagnostics().combined, combined);
            assert_eq!(
                drafter.floor.diagnostics().external,
                (40.0 * decay + 40.0, 2.0 * decay + 2.0)
            );
            drafter.floor.seed_retired(true);
            drafter.ngram_agreement = ngram_table(0.9, 20.0);
            drafter.ngram_proposal = vec![1, 2, 3];
            drafter.ngram_proposed_at = Some(3);
            let agreement = drafter.agreement.clone();
            MtpDrafter::configure_request(&mut drafter, SpecRequestConfig::default());
            assert_eq!(drafter.agreement_decay, decay);
            assert_eq!(drafter.floor.diagnostics().decay, decay);
            assert!(!drafter.floor.retired());
            assert!(!drafter.floor.calibrated());
            assert_eq!(drafter.floor.diagnostics().observed, 0);
            assert_eq!(drafter.floor.diagnostics().combined, (0.0, 0.0));
            assert_eq!(drafter.floor.diagnostics().external, (0.0, 0.0));
            assert_eq!(drafter.floor.diagnostics().interleaved, (0.0, 0.0));
            assert_eq!(
                drafter.floor.diagnostics().batched,
                [(0.0, 0.0); MTP_WINDOW_COST.len()]
            );
            assert!(drafter
                .ngram_agreement
                .iter()
                .all(|&entry| entry == AGREEMENT_PRIOR));
            assert!(drafter.ngram_proposal.is_empty() && drafter.ngram_proposed_at.is_none());
            // Agreement restarts during prefill, not configure_request.
            assert_eq!(drafter.agreement, agreement);
        }
    }

    // ---- The n-gram copy lane (CPU: the real planner and take-length
    // chooser, simulated windows).

    /// `n` pseudo-random tokens in `base..base + 40_000` (xorshift32).
    fn random_tokens(seed: u32, base: u32, n: usize) -> Vec<u32> {
        let mut state = seed.max(1);
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                base + state % 40_000
            })
            .collect()
    }

    /// Halo gfx1151 takeover window: AR step plus 6.43 ms per candidate row
    /// plus the 1 ms teacher-forced head fill.
    fn halo_takeover_us(candidates: usize) -> f32 {
        28_700.0 + 6_430.0 * candidates as f32 + 1_000.0
    }

    /// Native acceptance per draft of the simulated head (and the chance its
    /// row-0 argmax is the true next token).
    const NATIVE_ACCEPT: f32 = 0.9;

    /// A prompt holding `span` between filler no output token comes from.
    fn prompt_around(span: &[u32]) -> Vec<u32> {
        let mut prompt = random_tokens(3, 50_000, 50);
        prompt.extend_from_slice(span);
        prompt.extend(random_tokens(4, 60_000, 20));
        prompt
    }

    /// What the copy-lane harness ran in one window.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum CopyWindow {
        Ar,
        Native,
        Takeover {
            take: usize,
            accepted: usize,
            confirm: bool,
        },
    }

    struct CopyRun {
        /// `(emitted.len() at the plan, what ran)` per window.
        windows: Vec<(usize, CopyWindow)>,
        /// The plan's route of each window.
        routes: Vec<WindowRoute>,
        /// Whether the pool made a proposal at the window's plan.
        proposed: Vec<bool>,
        /// Whether the floor had retired the request at the window's plan.
        retired: Vec<bool>,
        /// `(tokens, us)` of each window.
        costs: Vec<(f32, f32)>,
        /// The drafter still armed the n-gram lane at the end.
        ngram_active: bool,
        tokens: f32,
        us: f32,
    }

    impl CopyRun {
        fn tokens_per_ms(&self) -> f32 {
            self.tokens / self.us * 1000.0
        }

        /// `(window index, take, accepted, confirm)` of every takeover.
        fn takeovers(&self) -> Vec<(usize, usize, usize, bool)> {
            self.windows
                .iter()
                .enumerate()
                .filter_map(|(index, &(_, window))| match window {
                    CopyWindow::Takeover {
                        take,
                        accepted,
                        confirm,
                    } => Some((index, take, accepted, confirm)),
                    _ => None,
                })
                .collect()
        }

        /// Index of the first window planned at or after `position` tokens.
        fn first_window_from(&self, position: usize) -> usize {
            self.windows
                .iter()
                .position(|&(start, _)| start >= position)
                .expect("a window past the position")
        }
    }

    /// A singleton drafter on the Halo prices (decay, floor and row costs),
    /// the n-gram pool seeded with `prompt` and armed when `ngram`.
    fn copy_drafter(prompt: &[u32], ngram: bool) -> Qwen4MtpDrafter {
        let config = qwen4_ngram_mod_config((5, 1, 63)).expect("copy-lane triple");
        let mut drafter = Qwen4MtpDrafter::new(5, 4096, None)
            .with_ngram(Some(config))
            .with_copy_lane(true);
        drafter.agreement_decay = MTP_AGREEMENT_DECAY_GFX1151;
        drafter.ngram_row_cost = Some((30.5, 6.9));
        MtpDrafter::configure_request(
            &mut drafter,
            SpecRequestConfig {
                allow_ngram_modifier: ngram,
                ..SpecRequestConfig::default()
            },
        );
        if let Some(pool) = drafter.ngram_copy.as_mut() {
            pool.begin_request(prompt);
        }
        assert_eq!(drafter.ngram_active, ngram);
        drafter
    }

    /// [`copy_run_with`] at [`NATIVE_ACCEPT`] throughout, the head's row-0
    /// argmax the true token.
    fn copy_run(prompt: &[u32], out: &[u32], ngram: bool) -> CopyRun {
        copy_run_with(prompt, out, ngram, |_| NATIVE_ACCEPT, false)
    }

    /// Drive `plan_window_with` over the output stream `out` (its first token
    /// the seed): calibration AR, native windows accepting each draft with
    /// `accept(emitted so far)` priced at the Halo native costs, retired AR
    /// tokens, takeovers matched against the stream and priced at
    /// [`halo_takeover_us`]. The head's confirmation passes when the first
    /// candidate is the true token and its draw accepts, or always with
    /// `head_follows_pool` (a head that shares the pool's wrong guess). The
    /// head's catch-up before a retired takeover is host bookkeeping only.
    fn copy_run_with(
        prompt: &[u32],
        out: &[u32],
        ngram: bool,
        accept: impl Fn(usize) -> f32,
        head_follows_pool: bool,
    ) -> CopyRun {
        copy_run_at(
            prompt,
            out,
            ngram,
            accept,
            head_follows_pool,
            rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS,
        )
    }

    /// [`copy_run_with`] on a drafter whose verify forward runs
    /// `row_exact_rows` rows row-exact.
    fn copy_run_at(
        prompt: &[u32],
        out: &[u32],
        ngram: bool,
        accept: impl Fn(usize) -> f32,
        head_follows_pool: bool,
        row_exact_rows: usize,
    ) -> CopyRun {
        let mut drafter = copy_drafter(prompt, ngram);
        drafter.row_exact_rows = row_exact_rows;
        let mut draws = Draws(0x9E37_79B9);
        let mut emitted = vec![out[0]];
        let mut run = CopyRun {
            windows: Vec::new(),
            routes: Vec::new(),
            proposed: Vec::new(),
            retired: Vec::new(),
            costs: Vec::new(),
            ngram_active: ngram,
            tokens: 0.0,
            us: 0.0,
        };
        while emitted.len() < out.len() {
            let start = emitted.len();
            let truth = &out[start..];
            let accept = accept(start);
            let k = MtpDrafter::proposal_capacity(&drafter);
            let plan = drafter
                .plan_window_with(&emitted, k, None, false)
                .expect("plan");
            let takeover = plan.ngram_hit
                && (!plan.ngram_confirm
                    || head_follows_pool
                    || (drafter.ngram_candidates[0] == truth[0] && draws.next() < accept));
            let (kind, committed, us) = if takeover {
                let candidates = drafter.ngram_candidates.clone();
                let accepted = candidates
                    .iter()
                    .zip(truth)
                    .take_while(|(candidate, token)| candidate == token)
                    .count();
                let committed = (accepted + 1).min(truth.len());
                let window = MtpWindow {
                    committed: truth[..committed].to_vec(),
                    accepted,
                    drafts_generated: candidates.len(),
                };
                let us = halo_takeover_us(candidates.len());
                drafter.finish_takeover(&window, us);
                let take = candidates.len();
                (
                    CopyWindow::Takeover {
                        take,
                        accepted,
                        confirm: plan.ngram_confirm,
                    },
                    committed,
                    us,
                )
            } else {
                match plan.route {
                    WindowRoute::ArCalibrate => {
                        drafter
                            .floor
                            .observe_calibration(HALO_AR_US, &drafter.agreement);
                        (CopyWindow::Ar, 1, HALO_AR_US)
                    }
                    WindowRoute::ArRetired | WindowRoute::ArRouted => {
                        (CopyWindow::Ar, 1, HALO_AR_US)
                    }
                    WindowRoute::Native { depth } => {
                        let (route, drafts, accepted, us) = if depth == 0 {
                            let accepted =
                                (0..plan.k).take_while(|_| draws.next() < accept).count();
                            let us = HALO_AR_US * (accepted + 1) as f32;
                            (
                                NativeWindow::Interleaved,
                                (accepted + 1).min(plan.k),
                                accepted,
                                us,
                            )
                        } else {
                            let accepted = (0..depth).take_while(|_| draws.next() < accept).count();
                            (
                                NativeWindow::Batched,
                                depth,
                                accepted,
                                HALO_WINDOW_US[depth],
                            )
                        };
                        let committed = (accepted + 1).min(truth.len());
                        let window = MtpWindow {
                            committed: truth[..committed].to_vec(),
                            accepted,
                            drafts_generated: drafts,
                        };
                        drafter.observe_agreement(&window);
                        drafter.floor_observe_native(false, route, &window, us, plan.k);
                        (CopyWindow::Native, committed, us)
                    }
                }
            };
            run.windows.push((start, kind));
            run.routes.push(plan.route);
            run.proposed.push(drafter.ngram_decision.is_some());
            run.retired.push(drafter.floor.retired());
            run.costs.push((committed as f32, us));
            run.tokens += committed as f32;
            run.us += us;
            emitted.extend_from_slice(&truth[..committed]);
        }
        run.ngram_active = drafter.ngram_active;
        run
    }

    #[test]
    fn ngram_take_cap_is_one_below_the_row_exact_rows() {
        let config = qwen4_ngram_mod_config((5, 1, 63)).expect("copy-lane triple");
        for (rows, cap) in [(8usize, 7usize), (16, 15), (32, 31)] {
            assert_eq!(qwen4_ngram_take_cap(&config, rows), cap, "rows={rows}");
        }
        let short = NgramModConfig { n_max: 5, ..config };
        assert_eq!(qwen4_ngram_take_cap(&short, 32), 5);
        assert_eq!(qwen4_ngram_take_cap(&config, 0), 0);
    }

    #[test]
    fn cpu_splice_caps_speculative_verify_rows_at_the_narrow_window() {
        assert_eq!(spec_verify_rows_cap(false), usize::MAX);
        assert_eq!(spec_verify_rows_cap(true), 4);
        assert!(check_spec_verify_rows(4, 4, "block").is_ok());
        assert!(check_spec_verify_rows(4, 1, "block").is_ok());
        let error = check_spec_verify_rows(4, 5, "Qwen4 verify block").unwrap_err();
        assert!(error.contains("5 rows") && error.contains("CPU routed-expert window"));
        assert!(error.contains("K<=3"), "{error}");
        assert!(check_spec_verify_rows(usize::MAX, 64, "block").is_ok());
    }

    #[test]
    fn window_rows_follow_the_widest_forward_of_the_plan() {
        assert_eq!(
            window_verify_rows(WindowRoute::Native { depth: 3 }, false, 0),
            4
        );
        assert_eq!(
            window_verify_rows(WindowRoute::Native { depth: 0 }, false, 0),
            1
        );
        assert_eq!(window_verify_rows(WindowRoute::ArRouted, false, 0), 1);
        assert_eq!(window_verify_rows(WindowRoute::ArRetired, true, 3), 4);
        assert_eq!(
            window_verify_rows(WindowRoute::Native { depth: 2 }, true, 5),
            6
        );
        assert_eq!(
            window_verify_rows(WindowRoute::Native { depth: 5 }, true, 2),
            6
        );
    }

    #[test]
    fn admitted_cpu_storage_does_not_promote_unqualified_product_windows() {
        let mut drafter = Qwen4MtpDrafter::new(63, 4096, None);
        drafter.spec_rows_cap = 64;
        drafter.derived_refresh_rows(4);
        assert_eq!((drafter.k(), drafter.proposal_capacity()), (3, 3));
        assert_eq!(drafter.agreement.len(), 3);
        assert!(drafter.plan_window_with(&[1], 63, None, false).is_err());
    }

    #[test]
    fn an_armed_cpu_splice_bounds_capacity_k_and_every_plan_to_three_drafts() {
        // Native windows: K beyond the window is clamped out of the capacity,
        // so the speculative loop's own budget (`min(capacity, max_emit - 1)`)
        // never reaches 4 and a direct over-budget call is refused.
        let mut drafter = Qwen4MtpDrafter::new(6, 4096, None);
        assert_eq!(MtpDrafter::proposal_capacity(&drafter), 6);
        drafter.spec_rows_cap = 4;
        assert_eq!(MtpDrafter::proposal_capacity(&drafter), 3);
        assert_eq!(MtpDrafter::k(&drafter), 3);
        assert_eq!(plan_of(&mut drafter, 3, None).k, 3);
        let error = plan_window_budget_error(&mut drafter, 4);
        assert!(error.contains("exceeds the proposal capacity 3"), "{error}");
        // An unarmed load keeps the head's full K.
        drafter.spec_rows_cap = usize::MAX;
        assert_eq!(MtpDrafter::proposal_capacity(&drafter), 6);
    }

    #[test]
    fn an_armed_cpu_splice_caps_the_ngram_sources_at_three_candidates() {
        // The non-copy takeover offers up to n_max candidates.
        let mut drafter = armed_ngram_drafter(1, 8);
        assert!(MtpDrafter::proposal_capacity(&drafter) >= 8);
        drafter.spec_rows_cap = 4;
        assert_eq!(MtpDrafter::proposal_capacity(&drafter), 3);
        // The copy lane's take cap is one below the row-exact rows, and no
        // more than one below the splice window either way.
        let mut copy = armed_copy_drafter(1, 63);
        copy.row_exact_rows = 32;
        assert_eq!(copy.ngram_take_cap(), 31);
        copy.spec_rows_cap = 4;
        assert_eq!(copy.ngram_take_cap(), 3);
        assert_eq!(MtpDrafter::proposal_capacity(&copy), 3);
    }

    #[test]
    fn drafter_take_cap_and_capacity_follow_its_row_exact_rows() {
        let mut drafter = armed_copy_drafter(1, 63);
        assert_eq!(
            drafter.row_exact_rows,
            rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS
        );
        assert_eq!(drafter.ngram_take_cap(), 7);
        drafter.row_exact_rows = 16;
        assert_eq!(drafter.ngram_take_cap(), 15);
        assert!(MtpDrafter::proposal_capacity(&drafter) >= 15);
    }

    #[test]
    fn copy_stream_takes_past_seven_when_more_rows_are_row_exact() {
        let span = random_tokens(11, 1_000, 600);
        let prompt = prompt_around(&span);
        let out = span[..500].to_vec();
        let wide = copy_run_at(&prompt, &out, true, |_| NATIVE_ACCEPT, false, 32);
        let takes: Vec<usize> = wide.takeovers().iter().map(|t| t.1).collect();
        assert!(takes.iter().any(|&take| take > 7), "{takes:?}");
        assert!(takes.iter().all(|&take| take <= 31), "{takes:?}");
    }

    #[test]
    fn copy_stream_takes_the_cap_and_accepts_long_runs() {
        let span = random_tokens(11, 1_000, 600);
        let prompt = prompt_around(&span);
        let out = span[..500].to_vec();
        let run = copy_run(&prompt, &out, true);
        let native = copy_run(&prompt, &out, false);
        let cap = qwen4_ngram_mod_config((5, 1, 63))
            .map(|c| qwen4_ngram_take_cap(&c, rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS))
            .unwrap();
        assert_eq!(cap, rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS - 1);

        let takeovers = run.takeovers();
        let late = &run.windows[20..];
        let at_cap = late
            .iter()
            .filter(|(_, w)| matches!(w, CopyWindow::Takeover { take, .. } if *take == cap))
            .count();
        let share = at_cap as f32 / late.len() as f32;
        let mean_accepted =
            takeovers.iter().map(|t| t.2).sum::<usize>() as f32 / takeovers.len().max(1) as f32;
        let speedup = run.tokens_per_ms() / native.tokens_per_ms();
        let confirmed_late = takeovers.iter().filter(|t| t.0 >= 20 && t.3).count();
        let measured = format!(
            "windows {} takeovers {} share_at_cap {share:.3} mean_accepted {mean_accepted:.3} \
             tok/ms {:.4} native {:.4} ratio {speedup:.3} confirmed_late {confirmed_late}",
            run.windows.len(),
            takeovers.len(),
            run.tokens_per_ms(),
            native.tokens_per_ms()
        );
        assert!(share >= 0.8, "{measured}");
        assert!(mean_accepted >= 6.0, "{measured}");
        assert!(speedup >= 1.2, "{measured}");
        // Once the copy source is established the head's row-0 step is skipped.
        assert_eq!(confirmed_late, 0, "{measured}");
    }

    #[test]
    fn prose_without_matches_stays_native() {
        let span = random_tokens(11, 1_000, 600);
        let prompt = prompt_around(&span);
        let out = random_tokens(77, 100_000, 300);
        let on = copy_run(&prompt, &out, true);
        let off = copy_run(&prompt, &out, false);
        assert!(on.takeovers().is_empty(), "{:?}", on.takeovers());
        assert_eq!(on.routes, off.routes);
        assert_eq!(on.windows, off.windows);
    }

    #[test]
    fn copy_then_divergence_backs_off_and_recovers() {
        let span = random_tokens(11, 1_000, 600);
        let prompt = prompt_around(&span);
        let mut out = span[..150].to_vec();
        // 25 blocks of a prompt key (the pool proposes its continuation in the
        // prompt, which is wrong) followed by one token the prompt never has.
        let wrong = random_tokens(21, 200_000, 25);
        for (block, &token) in wrong.iter().enumerate() {
            let key = 160 + 17 * block;
            out.extend_from_slice(&span[key..key + 5]);
            out.push(token);
        }
        assert_eq!(out.len(), 300);
        out.extend_from_slice(&span[300..450]);
        let run = copy_run(&prompt, &out, true);

        let first_divergent = run.first_window_from(150);
        let second_copy = run.first_window_from(300);
        let takeovers = run.takeovers();
        let divergent: Vec<_> = takeovers
            .iter()
            .filter(|t| t.0 >= first_divergent && t.0 < second_copy)
            .collect();
        // A proposal reaches the plan only where the last five tokens are a
        // key, so in the divergent span the n-gram table learns (and a
        // takeover is weighed) only at those windows.
        let proposals: Vec<usize> = (first_divergent..second_copy)
            .filter(|&window| run.proposed[window])
            .collect();
        let measured = format!(
            "divergence at window {first_divergent}, second copy at {second_copy}, \
             proposals at {proposals:?}, takeovers {takeovers:?}"
        );
        // Takeovers stop within six proposals of the divergence ...
        assert!(
            divergent
                .iter()
                .all(|t| proposals.iter().take(6).any(|&window| window == t.0)),
            "{measured}"
        );
        // ... after at most one that took the window unconfirmed: its
        // rejected first candidate clears `ngram_last_first_accepted`, so
        // every later take there needs the head's confirmation.
        assert!(divergent.iter().filter(|t| !t.3).count() <= 1, "{measured}");
        assert!(
            divergent.iter().filter(|t| !t.3).all(|t| t.2 == 0),
            "{measured}"
        );
        // They resume within ten windows of the second copy span.
        assert!(
            takeovers
                .iter()
                .any(|t| t.0 >= second_copy && t.0 <= second_copy + 10),
            "{measured}"
        );
    }

    #[test]
    fn take_never_exceeds_budget_or_row_exact_cap() {
        for n_max in [3usize, 7, 63] {
            for k in 0..=10usize {
                let mut drafter = armed_copy_drafter(1, n_max);
                let take_cap = n_max.min(rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS - 1);
                assert_eq!(drafter.ngram_take_cap(), take_cap);
                assert_eq!(
                    MtpDrafter::proposal_capacity(&drafter),
                    drafter.max_k.max(take_cap),
                    "n_max={n_max}"
                );
                let hit = drafter.ngram_hit(&[0, 1], k);
                assert_eq!(hit, k > 0, "n_max={n_max} k={k}");
                if hit {
                    let take = drafter.ngram_candidates.len();
                    assert!(take <= k, "n_max={n_max} k={k} take={take}");
                    assert!(
                        take < rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS,
                        "n_max={n_max} k={k}"
                    );
                    assert!(take <= take_cap, "n_max={n_max} k={k}");
                }
                // The proposal itself is observed up to `n_max`, past the cap
                // (the copy runs to the history end: `2..=20, 0, 1`).
                assert_eq!(
                    drafter.ngram_proposal.len(),
                    n_max.min(21),
                    "n_max={n_max} k={k}"
                );
            }
        }
        // Disarmed, the capacity is the head's own K.
        let mut drafter = armed_copy_drafter(1, 63);
        drafter.ngram_active = false;
        assert_eq!(MtpDrafter::proposal_capacity(&drafter), drafter.max_k);
    }

    #[test]
    fn shadow_observation_censors_at_committed_length() {
        let ratio = |(accepted, total): (f32, f32)| accepted / total;
        let fresh = |proposal: &[u32], at: usize| {
            let mut drafter = Qwen4MtpDrafter::new(3, 4096, None);
            drafter.ngram_agreement = vec![AGREEMENT_PRIOR; 4];
            drafter.ngram_proposal = proposal.to_vec();
            drafter.ngram_proposed_at = Some(at);
            drafter
        };
        let prior = ratio(AGREEMENT_PRIOR);

        // Proposal [a, b, c, d] against committed [a, b]: depths 1-2 accepted,
        // nothing compared beyond.
        let (a, b, c, d) = (10u32, 11, 12, 13);
        let mut drafter = fresh(&[a, b, c, d], 1);
        drafter.observe_ngram_shadow(&[99, a, b]);
        assert!(drafter.ngram_proposed_at.is_none());
        for depth in 0..2 {
            let entry = drafter.ngram_agreement[depth];
            assert!(
                entry.1 > AGREEMENT_PRIOR.1 + 0.5,
                "depth {depth}: {entry:?}"
            );
            assert!(ratio(entry) > prior, "depth {depth}: {entry:?}");
        }
        for depth in 2..drafter.ngram_agreement.len() {
            let entry = drafter.ngram_agreement[depth];
            assert!(
                (entry.1 - AGREEMENT_PRIOR.1).abs() < 1e-4,
                "depth {depth}: {entry:?}"
            );
        }

        // Committed [a, x, ..]: depth 1 accepted, depth 2 rejected, nothing
        // compared beyond.
        let mut drafter = fresh(&[a, b, c, d], 1);
        drafter.observe_ngram_shadow(&[99, a, 77, 78]);
        let [first, second, third] = [0, 1, 2].map(|depth| drafter.ngram_agreement[depth]);
        assert!(ratio(first) > prior && first.1 > AGREEMENT_PRIOR.1 + 0.5);
        assert!(ratio(second) < prior && second.1 > AGREEMENT_PRIOR.1 + 0.5);
        assert!((third.1 - AGREEMENT_PRIOR.1).abs() < 1e-4);

        // A proposal shorter than the committed tokens is censored at its own
        // length: all of it accepted, none of the bonus compared.
        let mut drafter = fresh(&[a, b], 1);
        drafter.observe_ngram_shadow(&[99, a, b, c, d]);
        assert!(ratio(drafter.ngram_agreement[1]) > prior);
        assert!((drafter.ngram_agreement[2].1 - AGREEMENT_PRIOR.1).abs() < 1e-4);

        // No committed token yet: nothing is observed and the proposal stays.
        let mut drafter = fresh(&[a, b], 3);
        drafter.observe_ngram_shadow(&[99, a, b]);
        assert_eq!(drafter.ngram_proposed_at, Some(3));
        assert!(drafter
            .ngram_agreement
            .iter()
            .all(|&entry| entry == AGREEMENT_PRIOR));
    }

    #[test]
    fn takeover_cost_prefers_measured_then_neighbour_then_native() {
        let close = |a: f32, b: f32| (a - b).abs() < 1e-3;
        let good = agreement(1.8, 2.0);
        let ar = 30_000.0;
        let mut floor = calibrated(ar, &good);
        // No price while the probe budget lasts.
        assert!(floor.takeover_cost(3).is_none());
        batched_windows(&mut floor, &good, 3, 75_000.0, FLOOR_PROBE_WINDOWS);
        let row_us = floor.fit_line().expect("a priced native depth").2 as f32;

        // (c) No takeover measured: the native window cost at that row count,
        // the fit's line past the native table.
        assert!(close(floor.takeover_cost(3).unwrap(), 75_000.0 / ar));
        assert!(close(
            floor.takeover_cost(5).unwrap(),
            (75_000.0 + 2.0 * row_us) / ar
        ));
        assert!(floor.diagnostics().batched.get(9).is_none());
        assert!(close(
            floor.takeover_cost(9).unwrap(),
            (75_000.0 + 6.0 * row_us) / ar
        ));

        // (a) A measured takeover depth is its own mean.
        floor.observe_external_copy(8, 90_000.0, 7, &good);
        assert!(close(floor.takeover_cost(7).unwrap(), 90_000.0 / ar));
        // (b) Neighbours move the nearest measured takeover by the row slope,
        // ahead of the native model.
        assert!(close(
            floor.takeover_cost(5).unwrap(),
            (90_000.0 - 2.0 * row_us) / ar
        ));
        assert!(close(
            floor.takeover_cost(9).unwrap(),
            (90_000.0 + 2.0 * row_us) / ar
        ));
        assert!(close(
            floor.takeover_cost(3).unwrap(),
            (90_000.0 - 4.0 * row_us) / ar
        ));

        // An unusable timing (a drafter's first takeover of a row count)
        // records nothing: the neighbour still prices it.
        floor.observe_external_copy(6, f32::NAN, 5, &good);
        assert_eq!(floor.diagnostics().external_depth[5], (0.0, 0.0));
        assert!(close(
            floor.takeover_cost(5).unwrap(),
            (90_000.0 - 2.0 * row_us) / ar
        ));
        floor.observe_external_copy(6, 80_000.0, 5, &good);
        assert!(close(floor.takeover_cost(5).unwrap(), 80_000.0 / ar));

        // Equidistant measured depths: the one with more windows prices it.
        let mut floor = calibrated(ar, &good);
        batched_windows(&mut floor, &good, 3, 75_000.0, FLOOR_PROBE_WINDOWS);
        let row_us = floor.fit_line().expect("a priced native depth").2 as f32;
        floor.observe_external_copy(3, 60_000.0, 2, &good);
        for _ in 0..3 {
            floor.observe_external_copy(7, 100_000.0, 6, &good);
        }
        assert!(close(
            floor.takeover_cost(4).unwrap(),
            (100_000.0 - 2.0 * row_us) / ar
        ));

        // The drafter observes its first takeover of each row count unpriced.
        let mut drafter = floor_drafter(true);
        drafter.copy_lane = true;
        drafter.floor.observe_calibration(ar, &drafter.agreement);
        for _ in 0..3 {
            drafter.floor_observe_external_window(8, 90_000.0, 7);
        }
        let decay = drafter.floor.diagnostics().decay;
        let (sum, windows) = drafter.floor.diagnostics().external_depth[7];
        assert!((windows - (1.0 + decay)).abs() < 1e-5, "{windows}");
        assert!((sum - 90_000.0 * (1.0 + decay)).abs() < 1.0, "{sum}");
        assert_eq!(drafter.ngram_warm_depths, 1 << 7);
    }

    /// A prompt holding `span`, and for every five-token window of
    /// `divergent` that window followed by eight tokens that are never the
    /// stream's next: wherever the stream sits in `divergent` the pool
    /// proposes a wrong candidate 0.
    fn prompt_with_wrong_chains(span: &[u32], divergent: &[u32]) -> Vec<u32> {
        let mut prompt = prompt_around(span);
        for (at, key) in divergent.windows(5).enumerate() {
            prompt.extend_from_slice(key);
            prompt.extend(random_tokens(1_000 + at as u32, 300_000, 8));
        }
        prompt
    }

    #[test]
    fn retired_request_keeps_copy_takeovers() {
        let span = random_tokens(11, 1_000, 600);
        let prompt = prompt_around(&span);
        // Prose the pool never proposes on, where the native head accepts a
        // draft with 0.3 (the floor retires the request), then a verbatim copy.
        let prose = random_tokens(77, 100_000, 120);
        let copy_start = prose.len();
        let mut out = prose;
        out.extend_from_slice(&span[..300]);
        let run = copy_run_with(
            &prompt,
            &out,
            true,
            |emitted| {
                if emitted < copy_start {
                    0.3
                } else {
                    NATIVE_ACCEPT
                }
            },
            false,
        );

        let first_copy = run.first_window_from(copy_start);
        assert!(
            run.retired[first_copy],
            "the floor never retired the request"
        );
        assert!(
            run.proposed[..first_copy].iter().all(|&proposed| !proposed),
            "the pool proposed on prose"
        );
        let copy = &run.windows[first_copy..];
        let steady = &copy[10..];
        let at_cap = steady
            .iter()
            .filter(|(_, w)| matches!(w, CopyWindow::Takeover { take: 7, .. }))
            .count();
        let share = at_cap as f32 / steady.len() as f32;
        let (tokens, us) = run.costs[first_copy..]
            .iter()
            .fold((0.0f32, 0.0f32), |(t, u), &(tokens, us)| {
                (t + tokens, u + us)
            });
        let speedup = (tokens / us) * HALO_AR_US;
        let measured = format!(
            "retired at plan of window {first_copy}, copy windows {}, takeovers {}, \
             share_at_cap_from_10th {share:.3}, copy tok/ms {:.4}, AR-only {:.4}, ratio {speedup:.3}",
            copy.len(),
            run.takeovers().iter().filter(|t| t.0 >= first_copy).count(),
            tokens / us * 1000.0,
            1000.0 / HALO_AR_US,
        );
        assert!(share >= 0.7, "{measured}");
        // The request stays retired: no native window after retirement.
        assert!(
            run.windows
                .iter()
                .zip(&run.retired)
                .all(|(&(_, window), &retired)| !(retired && window == CopyWindow::Native)),
            "{measured}"
        );
        assert!(run.retired.last().copied().unwrap_or(false), "{measured}");
        assert!(speedup >= 2.0, "{measured}");
    }

    #[test]
    fn retired_takeovers_stop_when_they_lose_to_ar() {
        let span = random_tokens(11, 1_000, 600);
        let divergent = random_tokens(21, 150_000, 200);
        let prompt = prompt_with_wrong_chains(&span, &divergent);
        // Retire on prose, copy long enough to establish the source (so the
        // first divergent proposal is unconfirmed), then diverge for good:
        // every proposal's first candidate is wrong.
        let prose = random_tokens(77, 100_000, 120);
        let diverge_at = prose.len() + 200;
        let mut out = prose.clone();
        out.extend_from_slice(&span[..200]);
        out.extend_from_slice(&divergent);
        for head_follows_pool in [false, true] {
            let run = copy_run_with(
                &prompt,
                &out,
                true,
                |emitted| {
                    if emitted < prose.len() {
                        0.3
                    } else {
                        NATIVE_ACCEPT
                    }
                },
                head_follows_pool,
            );
            let first = run.first_window_from(diverge_at);
            assert!(run.retired[first], "the floor never retired the request");
            let takeovers: Vec<_> = run
                .takeovers()
                .into_iter()
                .filter(|t| t.0 >= first)
                .collect();
            let blind = takeovers.iter().filter(|t| !t.3).count();
            let confirmed = takeovers.len() - blind;
            let measured = format!(
                "head_follows_pool {head_follows_pool}: {} windows from the divergence, \
                 {blind} blind and {confirmed} confirmed takeovers, lane armed at the end {}",
                run.windows.len() - first,
                run.ngram_active
            );
            // At most one takeover runs without the head's confirmation, and
            // it is rejected whole.
            assert!(blind <= 1, "{measured}");
            assert!(
                takeovers.iter().filter(|t| !t.3).all(|t| t.2 == 0),
                "{measured}"
            );
            // Once retired, a rejected first candidate stops every take
            // (confirmed or not) until a shadow observation accepts one
            // again, whether or not the head agrees with the wrong pool.
            assert_eq!(confirmed, 0, "{measured}");
        }

        // A retired request's takeovers that cost at least AR per token
        // disarm the lane (the first of a row count is observed unpriced).
        let mut drafter = copy_drafter(&prompt, true);
        drafter.floor.seed_calibrated(HALO_AR_US);
        drafter.floor.seed_probes_done();
        drafter.floor.seed_retired(true);
        let wasted = MtpWindow {
            committed: vec![0],
            accepted: 0,
            drafts_generated: 4,
        };
        drafter.finish_takeover(&wasted, halo_takeover_us(4));
        assert!(
            drafter.ngram_active,
            "an unpriced takeover cannot block the lane"
        );
        drafter.finish_takeover(&wasted, halo_takeover_us(4));
        assert!(!drafter.ngram_active && drafter.floor.retired());
        assert_eq!(drafter.request_stats().ngram_mod_windows, 2);
        assert!(drafter.request_stats().mtp_retired);
    }

    /// A request that arms the n-gram source.
    fn ngram_request() -> SpecRequestConfig {
        SpecRequestConfig {
            allow_ngram_modifier: true,
            ..SpecRequestConfig::default()
        }
    }

    #[test]
    fn copy_lane_off_is_beta_defaults() {
        for n_max in [3usize, 8] {
            let config = qwen4_ngram_mod_config((5, 3, n_max)).expect("beta-style triple");
            let mut drafter = Qwen4MtpDrafter::new(3, 4096, None).with_ngram(Some(config));
            assert!(!drafter.copy_lane);
            assert_eq!(MtpDrafter::proposal_capacity(&drafter), 3);
            MtpDrafter::configure_request(&mut drafter, ngram_request());
            assert!(drafter.ngram_active);
            assert_eq!(
                MtpDrafter::proposal_capacity(&drafter),
                drafter.max_k.max(config.n_max)
            );
            assert_eq!(drafter.ngram_yield, (0.0, 0.0));
        }
        if !hipfire_config::mtp_ngram_copy_enabled_for_arch(crate::ARCH_ID, "gfx1201") {
            assert_eq!(native_mtp_proposal_drafts(3, "gfx1201"), 3);
        }
    }

    #[test]
    fn copy_lane_arms_its_own_pool_only() {
        let config = qwen4_ngram_mod_config((5, 3, 3)).expect("triple");
        for copy in [false, true] {
            let mut drafter = Qwen4MtpDrafter::new(3, 4096, None)
                .with_ngram(Some(config))
                .with_copy_lane(copy);
            assert_eq!(drafter.copy_lane, copy);
            MtpDrafter::configure_request(&mut drafter, SpecRequestConfig::default());
            assert!(
                !drafter.ngram_active && drafter.ngram.is_none() && drafter.ngram_copy.is_none()
            );
            MtpDrafter::configure_request(&mut drafter, ngram_request());
            assert!(drafter.ngram_active, "copy={copy}");
            assert_eq!(drafter.ngram.is_some(), !copy, "copy={copy}");
            assert_eq!(drafter.ngram_copy.is_some(), copy, "copy={copy}");
            // A later request that does not arm n-gram disarms it and keeps the
            // pool of the lane's mode only.
            MtpDrafter::configure_request(&mut drafter, SpecRequestConfig::default());
            assert!(!drafter.ngram_active);
            assert_eq!(drafter.ngram.is_some(), !copy, "copy={copy}");
            assert_eq!(drafter.ngram_copy.is_some(), copy, "copy={copy}");
            // An EP peer's replica runs the same lane.
            let replica = Qwen4MtpDrafter::new_replica(drafter.replica_params());
            assert_eq!(replica.copy_lane, copy);
        }
    }

    /// What [`halo_floor_request`] measured from window `late` on.
    #[derive(Debug)]
    struct HaloFloorRun {
        one_draft: usize,
        native: usize,
        tokens: f64,
        us: f64,
        /// The floor retired the request by its last window.
        retired: bool,
    }

    /// One request through the drafter as `mtp_step` drives it (plan, route,
    /// observe) with budget `k`, Halo's floor decay and n-gram disarmed: the
    /// drafter's first AR token priced at `first_ar_us` and every later one
    /// at [`HALO_AR_US`], each through [`Qwen4MtpDrafter::floor_observe_ar`];
    /// `warm`, a drafter that already ran its first AR token, so this
    /// request's calibration token is priced at `first_ar_us` (what beta did
    /// with the process's first decode token).  A batched window costs
    /// [`HALO_WINDOW_US`] of its drafts, accepting each draft while a draw
    /// stays below `acceptance(window)`; the interleaved route one AR step
    /// per emitted token.
    fn halo_floor_request(
        k: usize,
        first_ar_us: f32,
        warm: bool,
        seed: u32,
        windows: usize,
        late: usize,
        acceptance: impl Fn(usize) -> f32,
    ) -> HaloFloorRun {
        let mut drafter = Qwen4MtpDrafter::new(k, 32_768, None);
        drafter.agreement_decay = MTP_AGREEMENT_DECAY_GFX1151;
        drafter.warm_ar = warm;
        MtpDrafter::configure_request(&mut drafter, SpecRequestConfig::default());
        drafter.floor_enabled = true;
        let mut draws = Draws(seed.max(1));
        let mut emitted = vec![0u32];
        let mut ar_tokens = 0usize;
        let mut run = HaloFloorRun {
            one_draft: 0,
            native: 0,
            tokens: 0.0,
            us: 0.0,
            retired: false,
        };
        for w in 0..windows {
            let plan = loop {
                let plan = drafter
                    .plan_window_with(&emitted, k, None, false)
                    .expect("plan");
                if plan.route != WindowRoute::ArCalibrate {
                    break plan;
                }
                let us = if ar_tokens == 0 {
                    first_ar_us
                } else {
                    HALO_AR_US
                };
                ar_tokens += 1;
                drafter.floor_observe_ar(us);
                emitted.push(0);
                if w >= late {
                    run.tokens += 1.0;
                    run.us += f64::from(us);
                }
            };
            let p = acceptance(w);
            let (accepted, us) = match plan.route {
                WindowRoute::Native { depth } => {
                    let rows = if depth == 0 { plan.k } else { depth };
                    let accepted = (0..rows).take_while(|_| draws.next() < p).count();
                    let (drafts, us, route) = if depth == 0 {
                        let us = HALO_AR_US * (accepted + 1) as f32;
                        ((accepted + 1).min(plan.k), us, NativeWindow::Interleaved)
                    } else {
                        (depth, HALO_WINDOW_US[depth], NativeWindow::Batched)
                    };
                    let window = window(accepted, drafts);
                    drafter.observe_agreement(&window);
                    drafter.floor_observe_native(false, route, &window, us, plan.k);
                    if w >= late {
                        run.native += 1;
                        run.one_draft += usize::from(depth == 1);
                    }
                    (accepted, us)
                }
                _ => (0, HALO_AR_US),
            };
            emitted.resize(emitted.len() + accepted + 1, 0);
            if w >= late {
                run.tokens += (accepted + 1) as f64;
                run.us += f64::from(us);
            }
        }
        run.retired = drafter.floor.retired();
        run
    }

    /// The floor never prices AR from the drafter's first AR token (the
    /// process's first decode token pays the decode route's one-time kernel
    /// binds: Halo 55.6 ms against 30.0 ms, 9.35 s once): that token leaves
    /// the floor uncalibrated and the next window calibrates again on the
    /// real price.  The drafter's next request calibrates on its first token.
    #[test]
    fn floor_never_prices_ar_from_the_drafters_first_ar_token() {
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None);
        let requests: [&[f32]; 2] = [&[9_353_409.0, HALO_AR_US], &[31_000.0]];
        for (request, ar_tokens) in requests.into_iter().enumerate() {
            MtpDrafter::configure_request(&mut drafter, SpecRequestConfig::default());
            drafter.floor_enabled = true;
            for &us in ar_tokens {
                let plan = drafter
                    .plan_window_with(&[0], 3, None, false)
                    .expect("plan");
                assert_eq!(plan.route, WindowRoute::ArCalibrate, "request {request}");
                drafter.floor_observe_ar(us);
            }
            assert!(drafter.floor.calibrated(), "request {request}");
            assert_eq!(drafter.floor.ar_us(), ar_tokens[ar_tokens.len() - 1]);
            let plan = drafter
                .plan_window_with(&[0], 3, None, false)
                .expect("plan");
            assert!(
                matches!(plan.route, WindowRoute::Native { .. }),
                "request {request}: {:?}",
                plan.route
            );
        }
    }

    /// Speculation that genuinely loses still retires to AR after an
    /// inflated first decode token (4.7 AR tokens, the Halo ags run's
    /// ~140 ms against ~30 ms): at 0.2 acceptance every Halo option costs
    /// more per token than AR (one draft: 38.5 ms for 1.2 tokens against
    /// 28.45 ms).  Priced from that token (beta: the same token as a warm
    /// drafter's calibration), every option looks cheaper than AR and the
    /// request drafts to its end.  At 0.72 acceptance it keeps drafting.
    #[test]
    fn losing_speculation_retires_after_an_inflated_first_ar_token() {
        let first = 4.7 * HALO_AR_US;
        for k in [7usize, 3] {
            for seed in 1..=8u32 {
                let seed = seed * 7919;
                let fixed = halo_floor_request(k, first, false, seed, 200, 100, |_| 0.2);
                assert!(
                    fixed.retired && fixed.native == 0,
                    "k={k} seed={seed}: {fixed:?}"
                );
                let beta = halo_floor_request(k, first, true, seed, 200, 100, |_| 0.2);
                assert!(
                    !beta.retired && beta.native == 100,
                    "k={k} seed={seed}: {beta:?}"
                );
                let good = halo_floor_request(k, first, false, seed, 200, 100, |_| 0.72);
                assert!(
                    !good.retired && good.native == 100,
                    "k={k} seed={seed}: {good:?}"
                );
            }
        }
    }

    /// The Halo ags depth collapse and its root fix.  Priced from an
    /// inflated first decode token (4.7 AR tokens), the cost fit's prior row
    /// (the floor's row-cost prior, 0.27 AR steps) is several real rows: native
    /// acceptance dips to 0.45 for 200 windows (the copy lane takes the
    /// confident spans and leaves the native windows the rest) and
    /// recovers to 0.72, and the floor keeps one draft after the recovery
    /// (Halo ags: 625 native windows at one draft, 72% accepted).  Priced
    /// from the drafter's second AR token it climbs back.  Budgets 7
    /// (gfx1151's `auto`) and 3 (gfx1201's).
    #[test]
    fn inflated_first_ar_token_no_longer_locks_one_draft() {
        let run = |k: usize, warm: bool| {
            let (mut one_draft, mut native, mut tokens, mut us) = (0, 0, 0.0, 0.0);
            for seed in 1..=16u32 {
                let r = halo_floor_request(k, 4.7 * HALO_AR_US, warm, seed * 7919, 600, 450, |w| {
                    if (100..300).contains(&w) {
                        0.45
                    } else {
                        0.72
                    }
                });
                one_draft += r.one_draft;
                native += r.native;
                tokens += r.tokens;
                us += r.us;
            }
            (one_draft as f64 / native.max(1) as f64, tokens / us * 1e6)
        };
        for k in [7usize, 3] {
            let (beta_share, beta_rate) = run(k, true);
            let (fixed_share, fixed_rate) = run(k, false);
            eprintln!("k={k} first AR token 4.7x: one-draft share / tok/s beta {beta_share:.3} {beta_rate:.2}, fixed {fixed_share:.3} {fixed_rate:.2}");
            assert!(beta_share >= 0.5, "k={k}: beta locks: {beta_share}");
            assert!(fixed_share <= 0.05, "k={k}: {fixed_share}");
            assert!(
                fixed_rate >= 1.05 * beta_rate,
                "k={k}: {fixed_rate} vs {beta_rate}"
            );
        }
    }

    thread_local! {
        static FLOOR_CALL_LINES: std::cell::RefCell<Vec<String>> =
            std::cell::RefCell::new(Vec::new());
    }

    fn capture_floor_call(line: &str) {
        FLOOR_CALL_LINES.with(|lines| lines.borrow_mut().push(line.to_string()));
    }

    /// `lines` replayed on the qwen4 route's floor: its window-cost table and
    /// its takeover slots.
    fn replay_qwen4_floor_log<'a>(lines: impl Iterator<Item = &'a str>) -> ReplayReport {
        replay_call_log::<{ MTP_WINDOW_COST.len() }, { NGRAM_DEPTHS + 1 }>(
            lines,
            FLOOR_CALL_PREFIX,
            MTP_WINDOW_COST,
        )
    }

    /// A singleton-configured qwen4 floor driven the way the drafter drives it
    /// (calibration and AR probes, batched and interleaved windows, takeover
    /// windows, a decay rebind and a re-arm, a terrible stretch that pushes it
    /// toward retirement) with its exact call log captured.
    #[test]
    fn floor_call_log_replays_a_scripted_qwen4_floor_bit_identically() {
        FLOOR_CALL_LINES.with(|lines| lines.borrow_mut().clear());
        let decay = MTP_AGREEMENT_DECAY_GFX1151;
        let mut agreement = fresh_agreement();
        let mut floor =
            Qwen4Floor::with_retire_after(MTP_WINDOW_COST, MTP_SINGLETON_FLOOR_MIN_WINDOWS)
                .with_decay(decay)
                .with_retire_evidence(MTP_SINGLETON_FLOOR_AR_PROBE_TOKENS)
                .with_call_log_to(FLOOR_CALL_PREFIX, "test", true, capture_floor_call);
        let mut state = 0x2545_f491u32;
        let mut draw = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for step in 0..160usize {
            // From step 80 every window costs far more than AR.
            let slow = if step >= 80 { 6.0 } else { 1.0 };
            if floor.needs_ar_sample() {
                floor.observe_calibration(28_000.0 + (draw() % 4_000) as f32, &agreement);
                continue;
            }
            let provisional = floor.provisional_depth(&agreement, 7);
            match floor.route(&agreement, 7, provisional) {
                Action::ArHeadFree => {
                    floor.observe_external_copy(3, 60_000.0, 3, &agreement);
                }
                Action::ArSynced => {}
                Action::Interleaved => {
                    let accepted = (draw() % 2) as usize;
                    let w = window(accepted, 1);
                    observe_agreement(&mut agreement, &w, decay);
                    let us = slow * 30_000.0 * (accepted + 1) as f32;
                    floor.observe_native(NativeWindow::Interleaved, &w, us, 7, &agreement);
                }
                Action::Native(depth) => {
                    let accepted = (draw() % (depth as u32 + 1)) as usize;
                    let w = window(accepted, depth);
                    observe_agreement(&mut agreement, &w, decay);
                    let us = slow * (35_000.0 + 8_000.0 * depth as f32 + (draw() % 3_000) as f32);
                    floor.observe_native(NativeWindow::Batched, &w, us, 7, &agreement);
                }
            }
            if step % 11 == 0 {
                floor.observe_external(2, slow * 70_000.0, 3, &agreement);
            }
            if step % 13 == 0 {
                let drafts = 1 + (draw() % 7) as usize;
                floor.observe_external_copy(drafts, slow * 50_000.0, drafts, &agreement);
            }
            if step == 60 {
                floor.set_decay(DEFAULT_DECAY);
            }
            if step == 120 {
                floor.rearm();
                agreement = fresh_agreement();
            }
        }
        let lines = FLOOR_CALL_LINES.with(|lines| std::mem::take(&mut *lines.borrow_mut()));
        assert!(lines[0].contains("\"op\":\"install\""), "{}", lines[0]);
        assert!(lines.iter().any(|line| line.contains("\"op\":\"route\"")));
        assert!(lines
            .iter()
            .any(|line| line.contains("\"op\":\"observe_native\"")));
        assert!(lines
            .iter()
            .any(|line| line.contains("\"op\":\"observe_external_copy\"")));
        let report = replay_qwen4_floor_log(lines.iter().map(String::as_str));
        assert_eq!(report.malformed, 0, "{:?}", report.first_malformed);
        assert_eq!(report.orphan_calls, 0);
        assert_eq!(report.floors.len(), 1);
        assert!(report.calls() > 0);
        assert_eq!(report.calls(), lines.len() - 1);
        assert_eq!(
            report.mismatches(),
            0,
            "{:?}",
            report.floors[0].first_mismatch
        );
    }

    // ---- Derived K (HIPFIRE_QWEN4_MTP_DERIVED_K) --------------------------

    fn derived_counts_narrow(
        _: DerivedCard,
        _: DerivedCostMode,
        _: DerivedSource,
        depth: u8,
        _: DerivedFeature,
    ) -> Option<CalibrationCounts> {
        Some(CalibrationCounts {
            depth_s: if depth == 1 { 10_000 } else { 0 },
            depth_n: 10_000,
            cell_s: 0,
            cell_n: 0,
        })
    }

    fn derived_counts_dead(
        _: DerivedCard,
        _: DerivedCostMode,
        _: DerivedSource,
        _: u8,
        _: DerivedFeature,
    ) -> Option<CalibrationCounts> {
        Some(CalibrationCounts {
            depth_s: 0,
            depth_n: 10_000,
            cell_s: 0,
            cell_n: 0,
        })
    }

    fn derived_window_price(
        _: DerivedCard,
        _: DerivedCostMode,
        _: DerivedSource,
        _: ContextBucket,
        k: u8,
        _: bool,
    ) -> Option<(f64, bool)> {
        Some((30_000.0 + 2_000.0 * f64::from(k), false))
    }

    fn derived_window_price_missing(
        _: DerivedCard,
        _: DerivedCostMode,
        _: DerivedSource,
        _: ContextBucket,
        _: u8,
        _: bool,
    ) -> Option<(f64, bool)> {
        None
    }

    /// Only the OffPool profile is measured; OnCopy has no price at all.
    fn derived_window_price_off_only(
        _: DerivedCard,
        mode: DerivedCostMode,
        _: DerivedSource,
        _: ContextBucket,
        k: u8,
        _: bool,
    ) -> Option<(f64, bool)> {
        (mode == DerivedCostMode::OffPool).then(|| (30_000.0 + 2_000.0 * f64::from(k), false))
    }

    /// Two measured profiles that disagree: a steep per-row price off the
    /// copy route, a nearly flat one on it.
    fn derived_window_price_by_mode(
        _: DerivedCard,
        mode: DerivedCostMode,
        _: DerivedSource,
        _: ContextBucket,
        k: u8,
        _: bool,
    ) -> Option<(f64, bool)> {
        let per_row = match mode {
            DerivedCostMode::OffPool => 8_000.0,
            DerivedCostMode::OnCopy => 100.0,
        };
        Some((30_000.0 + per_row * f64::from(k), false))
    }

    /// A non-monotone curve (widths 1..=3 only): the adjacent step 1 -> 2
    /// loses rate but width 3 wins globally.
    fn derived_window_price_nonmonotone(
        _: DerivedCard,
        _: DerivedCostMode,
        _: DerivedSource,
        _: ContextBucket,
        k: u8,
        _: bool,
    ) -> Option<(f64, bool)> {
        match k {
            1 => Some((30_000.0, false)),
            2 => Some((45_000.0, false)),
            3 => Some((40_000.0, false)),
            _ => None,
        }
    }

    fn derived_ar_price(
        _: DerivedCard,
        _: DerivedCostMode,
        _: ContextBucket,
    ) -> Option<(f64, bool)> {
        Some((25_000.0, false))
    }

    fn derived_ar_price_missing(
        _: DerivedCard,
        _: DerivedCostMode,
        _: ContextBucket,
    ) -> Option<(f64, bool)> {
        None
    }

    /// Every depth near-certain (acceptance ~1.0).
    fn derived_counts_certain(
        _: DerivedCard,
        _: DerivedCostMode,
        _: DerivedSource,
        _: u8,
        _: DerivedFeature,
    ) -> Option<CalibrationCounts> {
        Some(CalibrationCounts {
            depth_s: 10_000,
            depth_n: 10_000,
            cell_s: 0,
            cell_n: 0,
        })
    }

    /// Every depth conditionally accepted at 0.5.
    fn derived_counts_half(
        _: DerivedCard,
        _: DerivedCostMode,
        _: DerivedSource,
        _: u8,
        _: DerivedFeature,
    ) -> Option<CalibrationCounts> {
        Some(CalibrationCounts {
            depth_s: 5_000,
            depth_n: 10_000,
            cell_s: 0,
            cell_n: 0,
        })
    }

    /// Every depth conditionally accepted at 0.8.
    fn derived_counts_eighty(
        _: DerivedCard,
        _: DerivedCostMode,
        _: DerivedSource,
        _: u8,
        _: DerivedFeature,
    ) -> Option<CalibrationCounts> {
        Some(CalibrationCounts {
            depth_s: 8_000,
            depth_n: 10_000,
            cell_s: 0,
            cell_n: 0,
        })
    }

    /// Depth 1 near-certain, deeper depths never accepted: the narrowest
    /// prefix (K=1) beats AR and every wider prefix.
    const DERIVED_NARROW: DerivedTables = DerivedTables {
        counts: derived_counts_narrow,
        window_price: derived_window_price,
        ar_price: derived_ar_price,
    };
    /// Nothing is ever accepted: AR beats every priced prefix.
    const DERIVED_DEAD: DerivedTables = DerivedTables {
        counts: derived_counts_dead,
        window_price: derived_window_price,
        ar_price: derived_ar_price,
    };
    /// AR is priced but no drafted width is.
    const DERIVED_UNPRICED: DerivedTables = DerivedTables {
        counts: derived_counts_dead,
        window_price: derived_window_price_missing,
        ar_price: derived_ar_price,
    };
    /// Every depth near-certain and rows cheap: the widest legal prefix wins.
    const DERIVED_CERTAIN: DerivedTables = DerivedTables {
        counts: derived_counts_certain,
        window_price: derived_window_price,
        ar_price: derived_ar_price,
    };
    /// Measured OffPool prices only.
    const DERIVED_OFF_ONLY: DerivedTables = DerivedTables {
        counts: derived_counts_certain,
        window_price: derived_window_price_off_only,
        ar_price: derived_ar_price,
    };
    /// Different measured profiles per cost mode.
    const DERIVED_BY_MODE: DerivedTables = DerivedTables {
        counts: derived_counts_half,
        window_price: derived_window_price_by_mode,
        ar_price: derived_ar_price,
    };
    /// A non-monotone window curve at acceptance 0.8, no AR price.
    const DERIVED_NONMONOTONE: DerivedTables = DerivedTables {
        counts: derived_counts_eighty,
        window_price: derived_window_price_nonmonotone,
        ar_price: derived_ar_price_missing,
    };

    fn derived_test_identity(arch: &str) -> DerivedIdentity<'_> {
        DerivedIdentity {
            arch,
            flash_next: true,
            packed_recipe: true,
            mtp_head: true,
            full_experts: true,
            ple: DerivedPle::Q8Sidecar,
            qsa: crate::QsaKvFormat::Fp8,
            gdn: crate::GdnStateFormat::Q8,
            singleton: true,
        }
    }

    fn derived_test_floor() -> Qwen4Floor {
        Qwen4Floor::with_retire_after(MTP_WINDOW_COST, MTP_SINGLETON_FLOOR_MIN_WINDOWS)
            .with_decay(DEFAULT_DECAY)
            .with_retire_evidence(MTP_SINGLETON_FLOOR_AR_PROBE_TOKENS)
            .with_call_log_to(FLOOR_CALL_PREFIX, "test", true, capture_floor_call)
    }

    /// A calibrated drafter whose floor call log is captured: its first plan
    /// routes `Native { depth: batched_depth(prior, 3) }`.
    fn derived_floor_drafter() -> Qwen4MtpDrafter {
        FLOOR_CALL_LINES.with(|lines| lines.borrow_mut().clear());
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None).with_explicit_mtp_k_override(Some(3));
        drafter.floor = derived_test_floor();
        drafter.floor.seed_calibrated(30_000.0);
        drafter
    }

    fn derived_test_cfg(temp: f32) -> SpecRequestConfig {
        SpecRequestConfig {
            temp,
            ..SpecRequestConfig::default()
        }
    }

    /// Flag on, greedy, bound eligible on the R9700 identity, position set as
    /// `mtp_step` carries it.
    fn derived_bind_test(drafter: &mut Qwen4MtpDrafter, tables: DerivedTables) {
        drafter.derived_configure(&derived_test_cfg(0.0), true);
        drafter.derived_bind(
            derived_eligibility(&derived_test_identity("gfx1201")),
            tables,
        );
        drafter.derived_position = Some(1500);
    }

    fn derived_drafter(tables: DerivedTables) -> Qwen4MtpDrafter {
        let mut drafter = derived_floor_drafter();
        derived_bind_test(&mut drafter, tables);
        drafter
    }

    fn derived_state(drafter: &Qwen4MtpDrafter) -> &DerivedState {
        match &drafter.derived {
            Some(DerivedPhase::Bound(state)) => state,
            _ => panic!("derived state is not bound"),
        }
    }

    fn derived_incumbent_depth(plan: &WindowPlan) -> usize {
        match plan.route {
            WindowRoute::Native { depth } => depth,
            other => panic!("expected a native route, got {other:?}"),
        }
    }

    /// The scripted plan sequence of a drafter and its captured Floor calls,
    /// with the process-global `seq` and `floor` identifiers stripped (they
    /// count every floor and call in the test binary, across threads); every
    /// other byte of each call line is compared.
    fn derived_scripted_plans(mut drafter: Qwen4MtpDrafter) -> (Vec<WindowPlan>, Vec<String>) {
        let plans: Vec<WindowPlan> = (0..12).map(|_| plan_of(&mut drafter, 3, None)).collect();
        let lines = FLOOR_CALL_LINES.with(|lines| std::mem::take(&mut *lines.borrow_mut()));
        let lines = lines
            .into_iter()
            .map(|line| {
                let op = line
                    .find("\"op\":")
                    .unwrap_or_else(|| panic!("floor call line without an op: {line}"));
                let head = line
                    .find('{')
                    .unwrap_or_else(|| panic!("floor call line without a body: {line}"));
                format!("{}{}", &line[..=head], &line[op..])
            })
            .collect();
        (plans, lines)
    }

    #[test]
    fn derived_k_off_keeps_the_floor_calls_and_plans_byte_identical() {
        let mut unset = derived_floor_drafter();
        // A position carried by a caller is inert while the state is absent.
        unset.derived_position = Some(1500);
        let (unset_plans, unset_lines) = derived_scripted_plans(unset);
        let mut zero = derived_floor_drafter();
        zero.derived_configure(&derived_test_cfg(0.0), false);
        zero.derived_position = Some(1500);
        assert!(zero.derived.is_none());
        let (zero_plans, zero_lines) = derived_scripted_plans(zero);
        assert!(unset_lines
            .iter()
            .any(|line| line.contains("\"op\":\"route\"")));
        assert_eq!(unset_plans, zero_plans);
        assert_eq!(unset_lines, zero_lines);

        // Enabled but sampled: no state is created and nothing changes.
        for temp in [0.8f32, f32::NAN] {
            let mut sampled = derived_floor_drafter();
            sampled.derived_configure(&derived_test_cfg(temp), true);
            sampled.derived_bind(Ok(DerivedCard::R9700), DERIVED_NARROW);
            sampled.derived_position = Some(1500);
            assert!(sampled.derived.is_none(), "temp={temp}");
            let (plans, lines) = derived_scripted_plans(sampled);
            assert_eq!(plans, unset_plans, "temp={temp}");
            assert_eq!(lines, unset_lines, "temp={temp}");
        }
    }

    #[test]
    fn derived_k_ineligible_identity_names_its_reason_and_retains_the_incumbent() {
        assert_eq!(
            derived_eligibility(&derived_test_identity("gfx1100")),
            Err("unsupported_arch")
        );
        assert_eq!(
            derived_eligibility(&derived_test_identity("gfx1201")),
            Ok(DerivedCard::R9700)
        );
        let halo = DerivedIdentity {
            ple: DerivedPle::Q4Sidecar,
            qsa: crate::QsaKvFormat::Q8,
            gdn: crate::GdnStateFormat::F32,
            ..derived_test_identity("gfx1151")
        };
        assert_eq!(derived_eligibility(&halo), Ok(DerivedCard::Halo));
        // Each card keeps only its own calibrated state class.
        assert_eq!(
            derived_eligibility(&DerivedIdentity {
                ple: DerivedPle::Q4Sidecar,
                ..derived_test_identity("gfx1201")
            }),
            Err("ple_class")
        );
        let q4 = DerivedIdentity {
            ple: DerivedPle::Q4Sidecar,
            ..derived_test_identity("gfx1201")
        };
        // The production helper requires the exact nonempty qualified book
        // descriptor. Empty shipping books cannot turn Q4 into the Q8 receipt.
        let profile = "sealed-qualified-q4-profile";
        assert_eq!(
            derived_measured_eligibility(&q4, Some(profile), |_| Some(profile)),
            Ok(DerivedCard::R9700)
        );
        for (cached, admitted) in [
            (None, None),
            (Some(profile), None),
            (None, Some(profile)),
            (Some(profile), Some("different-state-or-placement")),
        ] {
            assert!(derived_measured_eligibility(&q4, cached, |_| admitted).is_err());
        }
        assert_eq!(
            derived_measured_eligibility(
                &DerivedIdentity {
                    singleton: false,
                    ..q4
                },
                Some(profile),
                |_| Some(profile),
            ),
            Err("not_singleton")
        );
        assert_eq!(
            derived_eligibility(&DerivedIdentity {
                gdn: crate::GdnStateFormat::F32,
                ..derived_test_identity("gfx1201")
            }),
            Err("state_format")
        );
        assert_eq!(
            derived_eligibility(&DerivedIdentity {
                qsa: crate::QsaKvFormat::Fp8,
                ..halo
            }),
            Err("state_format")
        );
        assert_eq!(
            derived_eligibility(&DerivedIdentity {
                singleton: false,
                ..derived_test_identity("gfx1201")
            }),
            Err("not_singleton")
        );
        assert_eq!(
            derived_eligibility(&DerivedIdentity {
                packed_recipe: false,
                ..derived_test_identity("gfx1201")
            }),
            Err("recipe")
        );

        let (base_plans, base_lines) = derived_scripted_plans(derived_floor_drafter());
        let mut drafter = derived_floor_drafter();
        drafter.derived_configure(&derived_test_cfg(0.0), true);
        assert!(matches!(drafter.derived, Some(DerivedPhase::Unbound)));
        // Unbound: the window consults nothing.
        drafter.derived_position = Some(1500);
        let (plans, lines) = derived_scripted_plans(drafter);
        assert_eq!((plans, lines), (base_plans.clone(), base_lines.clone()));

        let mut drafter = derived_floor_drafter();
        drafter.derived_configure(&derived_test_cfg(0.0), true);
        drafter.derived_bind(
            derived_eligibility(&derived_test_identity("gfx1100")),
            DERIVED_NARROW,
        );
        assert!(matches!(drafter.derived, Some(DerivedPhase::Ineligible)));
        drafter.derived_position = Some(1500);
        let (plans, lines) = derived_scripted_plans(drafter);
        assert_eq!((plans, lines), (base_plans, base_lines));
    }

    #[test]
    fn derived_k_chooses_the_objective_width_and_observes_each_consulted_window_once() {
        let (base_plans, _) = derived_scripted_plans(derived_floor_drafter());
        let incumbent = derived_incumbent_depth(&base_plans[0]);
        assert!(incumbent >= 2, "incumbent depth {incumbent}");

        let mut drafter = derived_drafter(DERIVED_NARROW);
        let plan = plan_of(&mut drafter, 3, None);
        // The draft loop may run to the widest measured legal prefix (here
        // the head's K = 3); the objective's width (1) is where it stops.
        assert_eq!(plan.route, WindowRoute::Native { depth: 3 });
        assert!(!plan.ngram_hit);
        assert!(drafter.derived_loop_active());
        {
            let state = derived_state(&drafter);
            assert_eq!(state.consults, 1);
            assert_eq!(state.observes, 0);
            let pending = state.pending.expect("a consulted window holds its pending");
            assert_eq!(pending.source, DerivedSource::Mtp);
            assert_eq!(pending.mode, DerivedCostMode::OffPool);
            assert_eq!(usize::from(pending.incumbent), incumbent);
            assert_eq!(pending.chosen, Some(1));
            assert_eq!((pending.hi, pending.legal), (3, 3));
            assert_eq!(pending.caller_k, 3);
            assert_eq!(pending.position, 1500);
            assert_eq!(state.keys.len(), 3);
            assert!(state.keys.iter().enumerate().all(|(index, key)| {
                key.source == DerivedSource::Mtp
                    && usize::from(key.depth) == index + 1
                    && key.feature == DerivedFeature::Native
            }));
            let priced: Vec<u8> = state.prices.iter().map(|price| price.k).collect();
            assert_eq!(priced, vec![0, 1, 2, 3]);
        }
        // After the first drafted row the re-solve over `k >= 1` stops there.
        assert!(!drafter.derived_resolve(1));
        assert_eq!(derived_state(&drafter).resolves, 1);
        let window = MtpWindow {
            committed: vec![5, 6],
            accepted: 1,
            drafts_generated: 1,
        };
        drafter.derived_observe(&plan, 99, &Ok(window));
        {
            let state = derived_state(&drafter);
            assert_eq!(state.observes, 1);
            assert!(state.pending.is_none());
            assert_eq!(state.request.last_first(DerivedSource::Mtp), Some(true));
        }
        // Without a new consult nothing is observed again.
        assert!(!drafter.derived_pending_open());
        drafter.derived_observe(
            &plan,
            99,
            &Ok(MtpWindow {
                committed: vec![5],
                accepted: 0,
                drafts_generated: 1,
            }),
        );
        let state = derived_state(&drafter);
        assert_eq!(state.observes, 1);
        assert_eq!(state.request.last_first(DerivedSource::Mtp), Some(true));
    }

    /// A bound, calibrated drafter of head K `max_k` whose bundle reported
    /// `rows` qualified row-exact verify rows.
    fn derived_wide_drafter(max_k: usize, rows: usize, tables: DerivedTables) -> Qwen4MtpDrafter {
        FLOOR_CALL_LINES.with(|lines| lines.borrow_mut().clear());
        let mut drafter =
            Qwen4MtpDrafter::new(max_k, 4096, None).with_explicit_mtp_k_override(Some(max_k));
        drafter.floor = derived_test_floor();
        drafter.floor.seed_calibrated(30_000.0);
        derived_bind_test(&mut drafter, tables);
        drafter.derived_refresh_rows(rows);
        drafter
    }

    fn derived_pending(drafter: &Qwen4MtpDrafter) -> DerivedPending {
        derived_state(drafter)
            .pending
            .expect("a consulted window holds its pending")
    }

    /// Synthetic CPU evidence only: neither success nor rejection is certain.
    fn derived_uncertain_counts<const BOUNDARY: u8>(
        _: DerivedCard,
        _: DerivedCostMode,
        _: DerivedSource,
        depth: u8,
        _: DerivedFeature,
    ) -> Option<CalibrationCounts> {
        Some(CalibrationCounts {
            depth_s: if depth <= BOUNDARY { 19_600 } else { 200 },
            depth_n: 20_000,
            cell_s: 0,
            cell_n: 0,
        })
    }

    fn derived_uncertain_cost(
        _: DerivedCard,
        _: DerivedCostMode,
        _: DerivedSource,
        _: ContextBucket,
        k: u8,
        _: bool,
    ) -> Option<(f64, bool)> {
        Some((30_000.0 + 100.0 * f64::from(k), false))
    }

    fn assert_uncertain_boundary<const BOUNDARY: u8>() {
        let tables = DerivedTables {
            counts: derived_uncertain_counts::<BOUNDARY>,
            window_price: derived_uncertain_cost,
            ar_price: derived_ar_price,
        };
        // Independent exhaustive finite-cost reference: directly integrate the
        // posterior survival chain, without invoking the runtime objective.
        let reference = |cap: usize| {
            let (mut survival, mut emitted) = (1.0, 1.0);
            let mut best = (0usize, 1.0 / 25_000.0);
            for k in 1..=cap {
                let successes = if k <= usize::from(BOUNDARY) {
                    19_600.0
                } else {
                    200.0
                };
                survival *= (successes + 1.6) / 20_002.0;
                emitted += survival;
                let rate = emitted / (30_000.0 + 100.0 * k as f64);
                if rate > best.1 {
                    best = (k, rate);
                }
            }
            best.0
        };
        assert_eq!(reference(63), usize::from(BOUNDARY));
        for (rows, explicit, budget) in [(64, 63, 63), (33, 63, 32), (64, 16, 16), (64, 63, 11)] {
            let mut drafter = derived_wide_drafter(63, rows, tables);
            drafter.explicit_mtp_k = Some(explicit);
            drafter.refresh_chooser_capacity();
            let cap = (rows - 1).min(explicit).min(budget);
            let storage = (drafter.agreement.as_ptr(), drafter.agreement.capacity());
            let plan = plan_of(&mut drafter, budget, None);
            let optimum = reference(cap);
            assert_eq!(derived_pending(&drafter).chosen, Some(optimum as u8));
            assert!(optimum <= cap);
            for drafted in 1..optimum {
                assert!(drafter.derived_resolve(drafted));
            }
            assert!(!drafter.derived_resolve(optimum));
            assert_eq!(
                storage,
                (drafter.agreement.as_ptr(), drafter.agreement.capacity())
            );
            assert_eq!(plan.route, WindowRoute::Native { depth: cap });
        }
    }

    #[test]
    fn native_agreement_tracks_admitted_capacity_and_observes_depth_sixty_three() {
        let mut drafter = Qwen4MtpDrafter::new(63, 4096, None);
        drafter.derived_refresh_rows(64);
        assert_eq!(drafter.k(), 63);
        assert_eq!(drafter.agreement.len(), 63);
        drafter.observe_agreement(&window(63, 63));
        assert!(drafter.agreement[62].0 > AGREEMENT_PRIOR.0);
        let storage = (drafter.agreement.as_ptr(), drafter.agreement.capacity());
        drafter.refresh_chooser_capacity();
        assert_eq!(
            storage,
            (drafter.agreement.as_ptr(), drafter.agreement.capacity())
        );
        drafter.explicit_mtp_k = Some(16);
        drafter.refresh_chooser_capacity();
        assert_eq!((drafter.k(), drafter.agreement.len()), (16, 16));
        drafter.spec_rows_cap = 4;
        drafter.refresh_chooser_capacity();
        assert_eq!((drafter.k(), drafter.agreement.len()), (3, 3));
        assert_eq!(
            storage,
            (drafter.agreement.as_ptr(), drafter.agreement.capacity())
        );
        drafter.explicit_mtp_k = Some(0);
        drafter.refresh_chooser_capacity();
        assert_eq!((drafter.k(), drafter.agreement.len()), (0, 0));
    }

    #[test]
    fn derived_k_uncertain_boundaries_beyond_ten_match_finite_reference() {
        assert_uncertain_boundary::<16>();
        assert_uncertain_boundary::<32>();
        assert_uncertain_boundary::<63>();
    }

    #[test]
    fn derived_k_wide_capacity_without_prices_or_qualification_never_promotes() {
        let mut unpriced = derived_wide_drafter(63, 64, DERIVED_UNPRICED);
        let _ = plan_of(&mut unpriced, 63, None);
        assert!(derived_state(&unpriced).pending.is_none());
        assert_eq!(derived_state(&unpriced).consults, 0);
        let mut unqualified = derived_wide_drafter(63, 8, DERIVED_CERTAIN);
        let _ = plan_of(&mut unqualified, 7, None);
        assert_eq!(derived_pending(&unqualified).chosen, Some(7));
        assert!(unqualified.plan_window_with(&[1], 63, None, false).is_err());
    }

    #[test]
    fn derived_k_widens_beyond_the_incumbent_up_to_the_legal_cap() {
        let mut plain = Qwen4MtpDrafter::new(7, 4096, None);
        plain.floor = derived_test_floor();
        plain.floor.seed_calibrated(30_000.0);
        let incumbent = derived_incumbent_depth(&plan_of(&mut plain, 7, None));
        assert!(incumbent < 7, "incumbent depth {incumbent}");

        let mut drafter = derived_wide_drafter(7, 8, DERIVED_CERTAIN);
        let plan = plan_of(&mut drafter, 7, None);
        assert_eq!(plan.route, WindowRoute::Native { depth: 7 });
        assert!(plan.k >= 7);
        let pending = derived_pending(&drafter);
        assert_eq!(pending.chosen, Some(7));
        assert_eq!(usize::from(pending.incumbent), incumbent);
        assert_eq!((pending.hi, pending.legal, pending.caller_k), (7, 7, 7));
        // The stationary re-solve keeps extending to the objective's width.
        for drafted in 1..7 {
            assert!(drafter.derived_resolve(drafted), "drafted {drafted}");
        }
        assert!(!drafter.derived_resolve(7));
    }

    #[test]
    fn derived_k_legal_cap_is_the_qualified_rows_the_physical_rows_and_the_explicit_k() {
        // Qualified rows 5 (4 drafts), whatever the head's K.
        let mut drafter = derived_wide_drafter(7, 5, DERIVED_CERTAIN);
        assert_eq!((drafter.k(), drafter.proposal_capacity()), (4, 4));
        let plan = plan_of(&mut drafter, 4, None);
        assert_eq!(plan.route, WindowRoute::Native { depth: 4 });
        assert_eq!(derived_pending(&drafter).chosen, Some(4));
        assert_eq!(derived_pending(&drafter).legal, 4);
        // The physical rows (a CPU splice window) bind it below the qualified ones.
        let mut drafter = derived_wide_drafter(7, 8, DERIVED_CERTAIN);
        drafter.spec_rows_cap = 3;
        assert_eq!(drafter.k(), 2);
        let _ = plan_of(&mut drafter, 2, None);
        assert_eq!(derived_pending(&drafter).chosen, Some(2));
        // An explicit K below the qualified rows is the cap, not the old 3.
        let mut drafter = derived_wide_drafter(2, 8, DERIVED_CERTAIN);
        assert_eq!(drafter.k(), 2);
        let _ = plan_of(&mut drafter, 2, None);
        assert_eq!(derived_pending(&drafter).chosen, Some(2));
        let mut drafter = derived_wide_drafter(5, 8, DERIVED_CERTAIN);
        let _ = plan_of(&mut drafter, 5, None);
        assert_eq!(derived_pending(&drafter).chosen, Some(5));
        // An unqualified default never exceeds the qualified rows (8 today),
        // even with a huge explicit K and a physical capacity of 64.
        let mut drafter = derived_wide_drafter(63, 8, DERIVED_CERTAIN);
        drafter.spec_rows_cap = 64;
        assert_eq!(drafter.k(), 7);
        let _ = plan_of(&mut drafter, 7, None);
        assert_eq!(derived_pending(&drafter).chosen, Some(7));
        assert_eq!(derived_pending(&drafter).legal, 7);
        // A caller budget below the cap (the remaining tokens) binds too.
        let mut drafter = derived_wide_drafter(7, 8, DERIVED_CERTAIN);
        let _ = plan_of(&mut drafter, 2, None);
        assert_eq!(derived_pending(&drafter).chosen, Some(2));
    }

    #[test]
    fn derived_k_reaches_the_physical_executor_only_with_a_qualified_getter_and_allocates_nothing()
    {
        let mut drafter = derived_wide_drafter(63, 64, DERIVED_CERTAIN);
        drafter.spec_rows_cap = 64;
        assert_eq!(drafter.k(), 63);
        let capacities = |drafter: &Qwen4MtpDrafter| {
            let state = derived_state(drafter);
            (
                state.depths.capacity(),
                state.prices.capacity(),
                state.resolve.capacity(),
                state.keys.capacity(),
            )
        };
        let before = capacities(&drafter);
        assert!(before.0 >= 63 && before.1 >= 64 && before.2 >= 64 && before.3 >= 63);
        let plan = plan_of(&mut drafter, 63, None);
        assert_eq!(plan.route, WindowRoute::Native { depth: 63 });
        assert_eq!(derived_pending(&drafter).chosen, Some(63));
        for drafted in 1..63 {
            assert!(drafter.derived_resolve(drafted));
        }
        assert!(!drafter.derived_resolve(63));
        drafter.derived_observe(
            &plan,
            99,
            &Ok(MtpWindow {
                committed: vec![1; 64],
                accepted: 63,
                drafts_generated: 63,
            }),
        );
        assert_eq!(capacities(&drafter), before);
    }

    #[test]
    fn derived_k_off_or_ineligible_keeps_the_incumbent_ceiling() {
        let off = Qwen4MtpDrafter::new(63, 4096, None);
        let qualified = rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS - 1;
        assert_eq!((off.k(), off.proposal_capacity()), (qualified, qualified));
        let mut ineligible = Qwen4MtpDrafter::new(63, 4096, None);
        ineligible.derived_configure(&derived_test_cfg(0.0), true);
        ineligible.derived_bind(
            derived_eligibility(&derived_test_identity("gfx1100")),
            DERIVED_CERTAIN,
        );
        assert_eq!(ineligible.k(), qualified);
        let mut sampled = Qwen4MtpDrafter::new(63, 4096, None);
        sampled.derived_configure(&derived_test_cfg(0.8), true);
        assert_eq!(sampled.k(), qualified);
    }

    #[test]
    fn derived_k_prices_each_cost_mode_from_its_own_profile() {
        // Synthetic CPU tables use the inner binder, not shipping admission.
        // Set the execution mode before binding, as the loader/request path
        // does: changing it afterwards deliberately invalidates the identity.
        let bound_in_mode = |tables, copy_lane| {
            let mut drafter = derived_floor_drafter();
            drafter.copy_lane = copy_lane;
            derived_bind_test(&mut drafter, tables);
            drafter.derived_refresh_rows(8);
            drafter
        };
        // Off the copy route the steep profile loses to AR ...
        let mut off = bound_in_mode(DERIVED_BY_MODE, false);
        assert_eq!(off.derived_cost_mode(), DerivedCostMode::OffPool);
        assert_eq!(
            derived_state(&off).mode_hint,
            Some(DerivedCostMode::OffPool)
        );
        let plan = plan_of(&mut off, 3, None);
        assert_eq!(plan.route, WindowRoute::ArRouted);
        assert_eq!(derived_pending(&off).mode, DerivedCostMode::OffPool);
        assert_eq!(derived_pending(&off).chosen_cost_us, 25_000.0);
        assert_eq!(derived_state(&off).prices[1].cost_us, 38_000.0);
        // ... on it the flat profile makes the widest prefix win.
        let mut copy = bound_in_mode(DERIVED_BY_MODE, true);
        assert_eq!(copy.derived_cost_mode(), DerivedCostMode::OnCopy);
        assert_eq!(
            derived_state(&copy).mode_hint,
            Some(DerivedCostMode::OnCopy)
        );
        let plan = plan_of(&mut copy, 3, None);
        assert_eq!(plan.route, WindowRoute::Native { depth: 3 });
        let pending = derived_pending(&copy);
        assert_eq!(pending.mode, DerivedCostMode::OnCopy);
        assert_eq!(pending.chosen, Some(3));
        assert_eq!(pending.chosen_cost_us, 30_300.0);
        assert_eq!(derived_state(&copy).prices[1].cost_us, 30_100.0);

        // A mode with no measured profile is unsupported: the incumbent runs
        // and the OffPool price is never borrowed.
        let mut base = derived_floor_drafter();
        base.copy_lane = true;
        let base_plan = plan_of(&mut base, 3, None);
        let mut unsupported = bound_in_mode(DERIVED_OFF_ONLY, true);
        let plan = plan_of(&mut unsupported, 3, None);
        assert_eq!(plan, base_plan);
        assert_eq!(derived_state(&unsupported).consults, 0);
        assert!(derived_state(&unsupported).pending.is_none());
        let mut supported = bound_in_mode(DERIVED_OFF_ONLY, false);
        let _ = plan_of(&mut supported, 3, None);
        assert_eq!(derived_state(&supported).consults, 1);

        // A genuinely changed execution mode must not reuse either bound
        // profile, even when the synthetic table prices both modes.
        for copy_lane in [false, true] {
            let mut changed = bound_in_mode(DERIVED_BY_MODE, copy_lane);
            changed.copy_lane = !copy_lane;
            let mut incumbent = derived_floor_drafter();
            incumbent.copy_lane = !copy_lane;
            assert_eq!(
                plan_of(&mut changed, 3, None),
                plan_of(&mut incumbent, 3, None)
            );
            assert!(matches!(changed.derived, Some(DerivedPhase::Ineligible)));
        }
    }

    #[test]
    fn derived_k_production_tables_never_price_the_copy_mode() {
        // The production table, through the same fn pointers the planner binds.
        for card in [DerivedCard::R9700, DerivedCard::Halo] {
            for bucket in [
                ContextBucket::Le2K,
                ContextBucket::Le8K,
                ContextBucket::Le16K,
                ContextBucket::Over16K,
            ] {
                assert!(fn_phase1_ar_price(card, DerivedCostMode::OnCopy, bucket).is_none());
                for k in 1..=DERIVED_MAX_DRAFTS as u8 {
                    for source in [DerivedSource::Mtp, DerivedSource::Ngram] {
                        assert!(
                            fn_phase1_window_price(
                                card,
                                DerivedCostMode::OnCopy,
                                source,
                                bucket,
                                k,
                                true
                            )
                            .is_none(),
                            "{card:?} {bucket:?} {source:?} k={k}"
                        );
                    }
                }
            }
            // The OffPool profile is the measured one, widths never interpolated.
            let price = FN_PHASE1_TABLES.window_price;
            for k in 1..=7 {
                assert!(price(
                    card,
                    DerivedCostMode::OffPool,
                    DerivedSource::Mtp,
                    ContextBucket::Le2K,
                    k,
                    true
                )
                .is_some());
            }
            assert!(price(
                card,
                DerivedCostMode::OffPool,
                DerivedSource::Mtp,
                ContextBucket::Le2K,
                8,
                true
            )
            .is_none());
        }
    }

    #[test]
    fn derived_k_resolves_the_global_objective_not_the_first_failed_marginal() {
        use hipfire_runtime::spec_economics::DerivedYield;
        // The adjacent step 1 -> 2 loses rate ...
        let first = DerivedYield::seed().append(0.8);
        let second = first.append(0.8);
        assert!(!first
            .objective(30_000.0)
            .marginal_to(second.objective(45_000.0))
            .improves_rate());
        // ... yet width 3 is the global optimum, so the post-row re-solve
        // keeps drafting through the failed step instead of stopping at 1.
        let mut drafter = derived_wide_drafter(3, 8, DERIVED_NONMONOTONE);
        let plan = plan_of(&mut drafter, 3, None);
        assert_eq!(plan.route, WindowRoute::Native { depth: 3 });
        assert_eq!(derived_pending(&drafter).chosen, Some(3));
        assert!(drafter.derived_resolve(1));
        assert!(drafter.derived_resolve(2));
        assert!(!drafter.derived_resolve(3));
        assert_eq!(derived_state(&drafter).resolves, 3);
    }

    #[test]
    fn derived_k_draft_loop_is_not_driven_by_the_incumbent_prune_without_a_consult() {
        let mut drafter = derived_drafter(DERIVED_DEAD);
        // An AR window and an unconsulted window leave the incumbent loop.
        let _ = plan_of(&mut drafter, 3, None);
        assert!(!drafter.derived_loop_active());
        let mut drafter = derived_drafter(DERIVED_UNPRICED);
        let _ = plan_of(&mut drafter, 3, None);
        assert!(!drafter.derived_loop_active());
        let mut drafter = derived_drafter(DERIVED_NARROW);
        assert!(!drafter.derived_loop_active());
        let _ = plan_of(&mut drafter, 3, None);
        assert!(drafter.derived_loop_active());
    }

    #[test]
    fn derived_k_chooses_ar_per_window_without_retiring_the_floor() {
        let mut drafter = derived_drafter(DERIVED_DEAD);
        let plan = plan_of(&mut drafter, 3, None);
        assert_eq!(plan.route, WindowRoute::ArRouted);
        assert!(!plan.ngram_hit);
        assert!(!drafter.floor.retired());
        assert_eq!(
            derived_state(&drafter).pending.expect("pending").chosen,
            Some(0)
        );
        drafter.derived_observe(
            &plan,
            99,
            &Ok(MtpWindow {
                committed: vec![9],
                accepted: 0,
                drafts_generated: 0,
            }),
        );
        let state = derived_state(&drafter);
        assert_eq!(state.observes, 1);
        // AR ages the residuals but learns no draft outcome.
        assert_eq!(state.request.last_first(DerivedSource::Mtp), None);
        // The next window is planned from the floor again.
        let next = plan_of(&mut drafter, 3, None);
        assert_eq!(next.route, WindowRoute::ArRouted);
        assert!(!drafter.floor.retired());
    }

    #[test]
    fn derived_k_without_a_priced_width_does_not_consult() {
        let (base_plans, _) = derived_scripted_plans(derived_floor_drafter());
        let mut drafter = derived_drafter(DERIVED_UNPRICED);
        let plan = plan_of(&mut drafter, 3, None);
        assert_eq!(plan, base_plans[0]);
        let state = derived_state(&drafter);
        assert_eq!(state.consults, 0);
        assert!(state.pending.is_none());
        // A window nobody consulted observes nothing.
        assert!(!drafter.derived_pending_open());
    }

    #[test]
    fn derived_k_forced_and_floor_off_routes_bypass_the_chooser() {
        for incremental in [Some("0"), Some("1")] {
            let mut drafter = derived_drafter(DERIVED_DEAD);
            let plan = plan_of(&mut drafter, 3, incremental);
            assert!(!plan.floor_active);
            assert_ne!(plan.route, WindowRoute::ArRouted);
            assert_eq!(derived_state(&drafter).consults, 0);
        }
        let mut drafter = derived_drafter(DERIVED_DEAD);
        drafter.floor_enabled = false;
        let plan = plan_of(&mut drafter, 3, None);
        assert!(!plan.floor_active);
        assert_ne!(plan.route, WindowRoute::ArRouted);
        assert_eq!(derived_state(&drafter).consults, 0);
        // A caller that carries no absolute position never consults.
        let mut drafter = derived_drafter(DERIVED_DEAD);
        drafter.derived_position = None;
        let plan = plan_of(&mut drafter, 3, None);
        assert_ne!(plan.route, WindowRoute::ArRouted);
        assert_eq!(derived_state(&drafter).consults, 0);
    }

    fn derived_ngram_drafter(tables: Option<DerivedTables>) -> Qwen4MtpDrafter {
        FLOOR_CALL_LINES.with(|lines| lines.borrow_mut().clear());
        let mut drafter = armed_ngram_drafter(1, 8);
        drafter.floor_enabled = true;
        drafter.floor = derived_test_floor();
        drafter.floor.seed_calibrated(30_000.0);
        if let Some(tables) = tables {
            derived_bind_test(&mut drafter, tables);
        }
        drafter
    }

    #[test]
    fn derived_k_narrows_only_the_ngram_take_and_credits_the_source_that_ran() {
        let mut base = derived_ngram_drafter(None);
        let base_plan = plan_of(&mut base, 3, None);
        assert!(base_plan.ngram_hit, "the fixture's pool hit");
        let base_take = base.ngram_candidates.len();
        assert!(base_take >= 2, "take {base_take}");

        let mut drafter = derived_ngram_drafter(Some(DERIVED_NARROW));
        let plan = plan_of(&mut drafter, 3, None);
        // The incumbent's source, native cap and confirmation are kept;
        // only the take is narrowed.
        assert!(plan.ngram_hit);
        assert_eq!(plan.route, base_plan.route);
        assert_eq!(plan.ngram_confirm, base_plan.ngram_confirm);
        assert_eq!(drafter.ngram_candidates.len(), 1);
        assert_eq!(drafter.ngram_candidates[..], base.ngram_candidates[..1]);
        assert!(drafter.ngram_proposal.len() >= base_take);
        let key_len = drafter
            .ngram
            .as_ref()
            .and_then(MtpNgramContext::last_proposal)
            .map(|meta| meta.key_len as u32);
        {
            let state = derived_state(&drafter);
            let pending = state.pending.expect("pending");
            assert_eq!(pending.source, DerivedSource::Ngram);
            assert_eq!(usize::from(pending.incumbent), base_take);
            assert_eq!(pending.chosen, Some(1));
            assert_eq!(usize::from(pending.hi), drafter.ngram_proposal.len().min(3));
            assert_eq!(state.keys.len(), usize::from(pending.hi));
            assert_eq!(
                state.keys[0].feature,
                DerivedFeature::Ngram {
                    offered: derived_offer_bin(Some(drafter.ngram_proposal.len())),
                    key_len,
                    last_first: None,
                }
            );
        }
        // The takeover ran: its outcome is the n-gram source's.
        drafter.derived_note_takeover();
        drafter.derived_observe(
            &plan,
            99,
            &Ok(MtpWindow {
                committed: vec![2, 3],
                accepted: 1,
                drafts_generated: 1,
            }),
        );
        let state = derived_state(&drafter);
        assert_eq!(state.observes, 1);
        assert_eq!(state.request.last_first(DerivedSource::Ngram), Some(true));
        assert_eq!(state.request.last_first(DerivedSource::Mtp), None);

        // A confirmation miss falls back to the native window: the queried
        // source did not execute, so its residuals only age and the native
        // outcome is not donated to them.
        let mut missed = derived_ngram_drafter(Some(DERIVED_NARROW));
        let plan = plan_of(&mut missed, 3, None);
        assert!(plan.ngram_hit);
        missed.derived_observe(
            &plan,
            99,
            &Ok(MtpWindow {
                committed: vec![7, 8],
                accepted: 1,
                drafts_generated: 1,
            }),
        );
        let state = derived_state(&missed);
        assert_eq!(state.observes, 1);
        assert_eq!(state.request.last_first(DerivedSource::Ngram), None);
        assert_eq!(state.request.last_first(DerivedSource::Mtp), None);

        // AR chosen for an ngram window disables the hit for that window only.
        let mut ar = derived_ngram_drafter(Some(DERIVED_DEAD));
        let plan = plan_of(&mut ar, 3, None);
        assert_eq!(plan.route, WindowRoute::ArRouted);
        assert!(!plan.ngram_hit);
        assert!(!ar.floor.retired());
    }

    #[test]
    fn derived_k_widens_an_ngram_take_to_the_pools_offer() {
        let mut base = derived_ngram_drafter(None);
        let base_plan = plan_of(&mut base, 7, None);
        assert!(base_plan.ngram_hit, "the fixture's pool hit");
        let base_take = base.ngram_candidates.len();

        let mut drafter = derived_ngram_drafter(Some(DERIVED_CERTAIN));
        let plan = plan_of(&mut drafter, 7, None);
        assert!(plan.ngram_hit);
        assert_eq!(plan.route, base_plan.route);
        let offered = drafter.ngram_proposal.len();
        let width = offered.min(7);
        assert!(width >= base_take);
        // The take is exactly the chosen prefix of the pool's whole offer.
        assert_eq!(
            drafter.ngram_candidates[..],
            drafter.ngram_proposal[..width]
        );
        let pending = derived_pending(&drafter);
        assert_eq!(pending.source, DerivedSource::Ngram);
        assert_eq!(pending.mode, DerivedCostMode::OffPool);
        assert_eq!(pending.chosen, Some(width as u8));
        assert_eq!(pending.legal, 7);
    }

    fn derived_copy_drafter(tables: Option<DerivedTables>) -> Qwen4MtpDrafter {
        FLOOR_CALL_LINES.with(|lines| lines.borrow_mut().clear());
        let mut drafter = armed_copy_drafter(1, 8);
        drafter.floor_enabled = true;
        drafter.floor = derived_test_floor();
        drafter.floor.seed_calibrated(30_000.0);
        if let Some(tables) = tables {
            derived_bind_test(&mut drafter, tables);
        }
        drafter
    }

    #[test]
    fn derived_k_prices_the_copy_lane_takeover_in_its_own_mode() {
        let mut base = derived_copy_drafter(None);
        let base_plan = plan_of(&mut base, 7, None);
        assert!(
            base_plan.ngram_hit,
            "the fixture's copy proposal takes the window"
        );
        let base_candidates = base.ngram_candidates.clone();

        // A measured OnCopy profile prices the take in that mode.
        let mut drafter = derived_copy_drafter(Some(DERIVED_BY_MODE));
        let plan = plan_of(&mut drafter, 7, None);
        assert!(plan.ngram_hit);
        let pending = derived_pending(&drafter);
        assert_eq!(
            (pending.source, pending.mode),
            (DerivedSource::Ngram, DerivedCostMode::OnCopy)
        );
        let width = usize::from(pending.chosen.expect("a priced take"));
        assert!(width >= 1 && width <= drafter.ngram_proposal.len().min(7));
        assert_eq!(
            drafter.ngram_candidates[..],
            drafter.ngram_proposal[..width]
        );

        // Only the OffPool profile is measured: the copy window keeps the
        // incumbent's take and consults nothing.
        let mut unsupported = derived_copy_drafter(Some(DERIVED_OFF_ONLY));
        let plan = plan_of(&mut unsupported, 7, None);
        assert_eq!(plan, base_plan);
        assert_eq!(unsupported.ngram_candidates, base_candidates);
        assert_eq!(derived_state(&unsupported).consults, 0);
        assert!(derived_state(&unsupported).pending.is_none());
    }

    #[test]
    fn derived_k_failed_window_and_resets_do_not_donate_outcomes() {
        let mut drafter = derived_drafter(DERIVED_NARROW);
        let plan = plan_of(&mut drafter, 3, None);
        assert!(drafter.derived_pending_open());
        drafter.derived_observe(&plan, 99, &Err("verify failed".to_string()));
        {
            let state = derived_state(&drafter);
            assert_eq!((state.consults, state.observes), (1, 0));
            assert!(state.pending.is_none());
            assert_eq!(state.request.last_first(DerivedSource::Mtp), None);
        }
        // The lifecycle clears (prefill, reset, terminal repair) drop it too.
        let plan = plan_of(&mut drafter, 3, None);
        drafter.derived_drop_pending();
        assert!(!drafter.derived_pending_open());
        drafter.derived_observe(
            &plan,
            99,
            &Ok(MtpWindow {
                committed: vec![5, 6],
                accepted: 1,
                drafts_generated: 1,
            }),
        );
        assert_eq!(derived_state(&drafter).observes, 0);
        // A stale pending is replaced by the next consult, never observed.
        let _ = plan_of(&mut drafter, 3, None);
        let _ = plan_of(&mut drafter, 3, None);
        assert_eq!(derived_state(&drafter).consults, 4);
        assert_eq!(derived_state(&drafter).observes, 0);
        // A new request starts unbound with no residuals.
        drafter.derived_configure(&derived_test_cfg(0.0), true);
        assert!(matches!(drafter.derived, Some(DerivedPhase::Unbound)));
        assert!(drafter.derived_position.is_none());
    }

    #[test]
    fn derived_k_clamps_acceptance_and_censors_a_terminal_rejection() {
        let mut drafter = derived_drafter(DERIVED_NARROW);
        let window = |committed: Vec<u32>, accepted: usize| MtpWindow {
            committed,
            accepted,
            drafts_generated: 1,
        };
        // An end-of-turn that rejects the draft is censored: history unknown.
        let plan = plan_of(&mut drafter, 3, None);
        drafter.derived_observe(&plan, 99, &Ok(window(vec![99], 0)));
        assert_eq!(
            derived_state(&drafter)
                .request
                .last_first(DerivedSource::Mtp),
            None
        );
        // A rejection that is not terminal is a failure at depth 1.
        let plan = plan_of(&mut drafter, 3, None);
        drafter.derived_observe(&plan, 99, &Ok(window(vec![7], 0)));
        assert_eq!(
            derived_state(&drafter)
                .request
                .last_first(DerivedSource::Mtp),
            Some(false)
        );
        // An over-claimed acceptance is clamped to what was committed.
        let plan = plan_of(&mut drafter, 3, None);
        drafter.derived_observe(&plan, 99, &Ok(window(vec![1, 2], 5)));
        let state = derived_state(&drafter);
        assert_eq!(state.request.last_first(DerivedSource::Mtp), Some(true));
        assert_eq!(state.observes, 3);
    }

    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/floor_call_log_replay.rs"
    ));
}
