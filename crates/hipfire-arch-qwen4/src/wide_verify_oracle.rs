// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Developer hooks of the wide-verify oracle (`examples/qwen4_wide_verify`).
//!
//! The oracle compares the PRODUCTION verify transaction
//! (`Qwen4Bundle::verify_block_rows` through `SpecTarget::verify_block`:
//! snapshot, row-armed `spec_forward_rows` with `prefill_scope = false`,
//! `commit_prefix`) against sequential `forward_token` decode. These hooks
//! only expose that transaction, its resident logits and the live state's
//! digests to the example; no production route calls them, and the module is
//! compiled only with `reference-parity`.

use crate::bundle::Qwen4Bundle;
use crate::cache_quality::cq_state_families;
use crate::mtp_spec::native_mtp_row_capture;
use hipfire_runtime::session_cache::SessionRoute;
use hipfire_runtime::spec::SpecScratch;
use rdna_compute::Gpu;

/// Live qualified and physical bounds. Timing-only callers may exceed only
/// `row_exact_verify_rows`, never the other two or the executor's 64 rows.
#[derive(Clone, Copy, Debug)]
pub struct DevVerifyCaps {
    pub spec_verify_rows_cap: usize,
    pub forward_max_chunk: usize,
    pub row_exact_verify_rows: usize,
}

pub fn dev_verify_caps(bundle: &Qwen4Bundle, gpu: &Gpu) -> DevVerifyCaps {
    let forward_max_chunk = bundle.execution.as_ref().map_or(0, |f| f.scratch.max_chunk);
    let logits = bundle
        .spec_logits
        .as_ref()
        .map_or(0, |t| t.numel() / bundle.config.vocab_size.max(1));
    let top1 = bundle.spec_top1.as_ref().map_or(0, |t| t.byte_size() / 4);
    let physical = crate::gpu_forward::qwen4_spec_logit_rows(forward_max_chunk)
        .min(logits)
        .min(top1)
        .min(bundle.spec_verify_rows_cap())
        .min(bundle.native_mtp_cpu_window_rows().unwrap_or(usize::MAX));
    DevVerifyCaps {
        spec_verify_rows_cap: physical,
        forward_max_chunk,
        row_exact_verify_rows: bundle.row_exact_verify_rows(gpu),
    }
}

/// Effective row-exact verify rows of the live bundle: the frozen
/// `row_exact_verify_rows` getter (admitted arithmetic with AR reference,
/// CPU-expert restriction, physical capacity, allocated scratch).
pub fn dev_effective_row_exact_rows(bundle: &Qwen4Bundle, gpu: &Gpu) -> usize {
    bundle.row_exact_verify_rows(gpu)
}

/// The captured-input recurrence repair path, not restore plus replay.
pub fn dev_native_retain_supported(bundle: &Qwen4Bundle, n: usize) -> bool {
    n == 1 || (n >= 2 && bundle.state.row_capture_rows() >= n)
}

pub fn dev_commit_captured_prefix(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    block: &[u32],
    keep: usize,
    scratch: &mut dyn SpecScratch,
) -> Result<(), String> {
    bundle.dev_commit_captured_prefix(gpu, block, keep, scratch)
}

/// `accept_len` excludes the seed; `keep` in the captured-prefix API includes it.
pub fn dev_commit_native_retain(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    block: &[u32],
    accept_len: usize,
    scratch: &mut dyn SpecScratch,
) -> Result<(), String> {
    let keep = accept_len
        .checked_add(1)
        .ok_or("native accept length overflow")?;
    dev_commit_captured_prefix(bundle, gpu, block, keep, scratch)
}

pub fn dev_head_pending_hidden(
    drafter: &crate::mtp_spec::Qwen4MtpDrafter,
    gpu: &Gpu,
) -> Result<Vec<u8>, String> {
    let values = gpu
        .download_f32(drafter.pending_hidden_for_parity()?)
        .map_err(|e| e.to_string())?;
    Ok(values.into_iter().flat_map(f32::to_le_bytes).collect())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DevWindowSource {
    ArRouted,
    Mtp,
    Ngram,
}

/// Actual execution branch, latched by the executor rather than inferred
/// from a requested width or an environment setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DevWindowRoute {
    TargetOnly,
    Batched,
    Interleaved,
}

#[derive(Clone, Copy, Debug)]
pub struct DevForcedWindow {
    pub source: DevWindowSource,
    pub drafts: usize,
    /// Explicit diagnostic opt-in: never an identity qualification.
    pub timing_only: bool,
}

pub struct DevWindowMeasurement {
    pub requested_drafts: usize,
    pub actual_drafts: usize,
    pub offered: usize,
    pub accepted_prefix: usize,
    pub source: DevWindowSource,
    pub route: DevWindowRoute,
    pub requested_confirmation: bool,
    pub cost_mode: hipfire_runtime::spec_economics::DerivedCostMode,
    pub head_confirmation: Option<bool>,
    pub terminal: bool,
    pub total_us: f64,
    pub window: hipfire_runtime::spec::MtpWindow,
    /// Actual head-row logit margins of this native window: entry `j` is the
    /// margin read at committed draft row `j + 1` (`Some` iff finite and
    /// non-negative, else `None`). Native only; every other source is empty.
    /// The curve consumer gates `complete` on requested == actual and on
    /// this length matching `actual_drafts`. Copied out of the drafter
    /// scratch once for the receipt; production paths keep no copy.
    pub native_draft_margins: Vec<Option<f32>>,
    /// The observed window did not match the request (source or width):
    /// the curve consumer keeps it as a raw incomplete receipt
    /// (`complete: false` + `missing_reason`), never a priced window.
    /// Only real transaction failures (position/seed/caps, route latch)
    /// surface as `Err`.
    pub outcome_mismatch: bool,
}

pub struct DevVerifyFaultOutcome {
    pub verify_error: String,
    pub receipt: rdna_compute::LabKernelFaultReceipt,
}

/// Inject through the real verify transaction after its snapshot is open.
/// `0` fails its first kernel; `n > 0` permits exactly n successful kernel
/// submissions before failure. Disarming precedes automatic target restore.
pub fn dev_verify_resident_fault_after_snapshot(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    block: &[u32],
    scratch: &mut dyn SpecScratch,
    successful_launches: usize,
) -> Result<DevVerifyFaultOutcome, String> {
    bundle.dev_verify_fault_after_snapshot(gpu, block, scratch, successful_launches)
}

/// Kernel-only eager-HIP launch events from the production verify's forward,
/// excluding its snapshot. Prepare the GPU's event pool before calling this;
/// a successful result leaves the normal verify ticket active for commit/undo.
#[cfg(feature = "kernel-timing")]
pub fn dev_verify_resident_profiled(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    block: &[u32],
    scratch: &mut dyn SpecScratch,
) -> Result<hip_bridge::KernelTimingReport, String> {
    bundle.dev_verify_resident_profiled(gpu, block, scratch)
}

/// One real production window. No manufactured copy candidates or component
/// sums: the device-synced total includes lookup, planning, head alignment,
/// verification, acceptance, rollback, head fill and the commit boundary.
#[allow(clippy::too_many_arguments)]
pub fn dev_run_forced_window(
    drafter: &mut crate::mtp_spec::Qwen4MtpDrafter,
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    position: usize,
    seed: u32,
    emitted: &[u32],
    eos: u32,
    request: DevForcedWindow,
) -> Result<DevWindowMeasurement, String> {
    drafter.dev_run_forced_window(bundle, gpu, position, seed, emitted, eos, request)
}

/// Exact arithmetic reference of the last multi-row verify attempt. Singleton
/// cells are unbound and must not reuse a previous multi-row receipt.
#[cfg(feature = "lab")]
pub fn dev_last_verify_reference(
    bundle: &Qwen4Bundle,
) -> Option<rdna_compute::prefill_invariant::Qwen4VerifyReference> {
    bundle.last_verify_reference()
}

/// Checked device allocation plan, not a measured peak or host-locked charge.
#[cfg(feature = "lab")]
pub fn dev_row_capture_planned_bytes(bundle: &Qwen4Bundle, rows: usize) -> Option<usize> {
    bundle.row_capture_planned_bytes(rows)
}

/// Make the GDN row-capture ring hold at least `rows` rows, so
/// `verify_block_rows` arms capture for blocks of up to `rows` rows (its rule:
/// `block.len() >= 2 && row_capture_rows() >= block.len()`).
///
/// Mirrors the product drafter's `ensure_resources`: the ring is allocated
/// only where `native_mtp_row_capture` says the GDN route captures rows. On
/// any other GPU this is a no-op and [`dev_row_capture_rows`] stays 0 (the
/// verify then runs unarmed, exactly as the product does there).
pub fn dev_arm_row_capture(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    rows: usize,
) -> Result<(), String> {
    if rows < 2 || !native_mtp_row_capture(gpu, &bundle.config) {
        return Ok(());
    }
    let conv_history = bundle
        .config
        .linear_conv_kernel_dim
        .checked_sub(1)
        .ok_or("linear_conv_kernel_dim is 0")?;
    bundle
        .state
        .ensure_row_capture(gpu, rows, conv_history)
        .map_err(|error| error.to_string())
}

/// Whether this GPU/config captures verify rows at all (the product drafter's
/// `native_mtp_row_capture` gate).
pub fn dev_row_capture_supported(bundle: &Qwen4Bundle, gpu: &Gpu) -> bool {
    native_mtp_row_capture(gpu, &bundle.config)
}

/// Rows the GDN row-capture ring currently holds (0 = none allocated).
pub fn dev_row_capture_rows(bundle: &Qwen4Bundle) -> usize {
    bundle.state.row_capture_rows()
}

/// Whether `verify_block_rows` arms row capture for an `n`-row block now: its
/// own arming rule, evaluated on the live state.
pub fn dev_would_arm(bundle: &Qwen4Bundle, n: usize) -> bool {
    n >= 2 && bundle.state.row_capture_rows() >= n
}

/// Live target position (tokens committed to the resident state).
pub fn dev_position(bundle: &Qwen4Bundle) -> usize {
    bundle.state.position
}

/// Run the production verify transaction over `block` at the live position
/// (picks downloaded, hidden captured) and download every row's logits:
/// returns `(picks, logits)` with `logits` row-major `block.len() * vocab`.
/// The state is left advanced by `block.len()` with the verify's snapshot
/// ticket active in `scratch`; finish with `SpecTarget::commit_prefix` or
/// [`dev_rollback_verify`].
/// A logits synchronization, view or download failure rolls the verify back
/// and consumes its ticket; a rollback failure is reported with the read error.
pub fn dev_verify_logits(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    block: &[u32],
    scratch: &mut dyn SpecScratch,
) -> Result<(Vec<u32>, Vec<f32>), String> {
    let position = bundle.state.position;
    let picks = bundle.dev_verify_block_rows(gpu, block, position, scratch, true)?;
    let result = (|| -> Result<Vec<f32>, String> {
        gpu.hip
            .device_synchronize()
            .map_err(|error| error.to_string())?;
        let rows = bundle
            .spec_logits_rows(0, block.len())
            .map_err(|error| error.to_string())?;
        gpu.download_f32(&rows).map_err(|error| error.to_string())
    })();
    match result {
        Ok(logits) => Ok((picks, logits)),
        Err(error) => {
            bundle
                .dev_rollback_verify(gpu, scratch)
                .map_err(|rollback| format!("{error}; target rollback failed: {rollback}"))?;
            Err(error)
        }
    }
}

/// The same verify transaction with the logits left resident and no pick
/// readback (`verify_block_resident`'s route): the timed variant.
pub fn dev_verify_resident(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    block: &[u32],
    scratch: &mut dyn SpecScratch,
) -> Result<(), String> {
    let position = bundle.state.position;
    bundle
        .dev_verify_block_rows(gpu, block, position, scratch, false)
        .map(|_| ())
}

/// Undo a finished verify without committing: restore the verify's snapshot
/// ticket (the state it started from) and consume it.
pub fn dev_rollback_verify(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    scratch: &mut dyn SpecScratch,
) -> Result<(), String> {
    bundle.dev_rollback_verify(gpu, scratch)
}

/// SHA-256 hex of every part of the live target state at the live position,
/// as `cache_quality::cq_state_families` computes them
/// (`SessionState::snapshot_parts` on the AR route: part 0 is the metadata,
/// then every fixed part, then the valid rows of every row stream). Keys are
/// `part.<i>`, in part order.
pub fn dev_state_digest(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
) -> Result<Vec<(String, String)>, String> {
    let position = bundle.state.position;
    let families = cq_state_families(gpu, bundle, SessionRoute::Ar, position)?;
    let mut parts: Vec<(usize, String, String)> = families
        .into_iter()
        .map(|(key, family)| {
            let index = key
                .strip_prefix("part.")
                .and_then(|index| index.parse::<usize>().ok())
                .ok_or_else(|| format!("unexpected state part key '{key}'"))?;
            Ok((index, key, family.digest))
        })
        .collect::<Result<_, String>>()?;
    parts.sort_by_key(|(index, _, _)| *index);
    Ok(parts
        .into_iter()
        .map(|(_, key, digest)| (key, digest))
        .collect())
}

/// A taken snapshot of the MTP head (state and the draft head's request-local
/// policy), bracketing a timed run of [`dev_head_chain`]; consume it with
/// [`dev_head_rollback`].
pub struct DevHeadTicket {
    snapshot: crate::mtp_gpu::MtpGpuStateSnapshot,
    draft: hipfire_dispatch::pipeline::DraftHeadRequestState,
}

/// The bundle has an attached MTP head.
pub fn dev_head_attached(bundle: &Qwen4Bundle) -> bool {
    bundle.mtp_position().is_ok()
}

/// The head's position.
pub fn dev_head_position(bundle: &Qwen4Bundle) -> Result<usize, String> {
    bundle.mtp_position().map_err(|error| error.to_string())
}

/// Snapshot the head and its draft request state before a chain.
pub fn dev_head_open(bundle: &mut Qwen4Bundle, gpu: &mut Gpu) -> Result<DevHeadTicket, String> {
    let snapshot = bundle
        .mtp_snapshot(gpu)
        .map_err(|error| error.to_string())?;
    let draft = bundle
        .mtp_draft_request_state()
        .map_err(|error| error.to_string())?;
    Ok(DevHeadTicket { snapshot, draft })
}

/// Exactly `depth` production batched-route head steps (the drafting loop's
/// `mtp_forward_token` calls: row 0 reads `pending` and selects QSA afresh,
/// later rows chain the previous draft) without the margin prune: no sync and
/// no download beyond what each step does. Returns the drafts.
pub fn dev_head_chain(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    seed: u32,
    pending: &rdna_compute::GpuTensor,
    position: usize,
    depth: usize,
) -> Result<Vec<u32>, String> {
    let mut drafts = Vec::with_capacity(depth);
    let mut input = seed;
    for index in 0..depth {
        let hidden = (index == 0).then_some(pending);
        let token_position = position
            .checked_add(index)
            .ok_or_else(|| "Qwen4 MTP step position overflow".to_string())?;
        input = bundle
            .mtp_forward_token(gpu, input, hidden, token_position, index == 0)
            .map_err(|error| error.to_string())?;
        drafts.push(input);
    }
    Ok(drafts)
}

/// Restore the head and its draft request state to the ticket's.
pub fn dev_head_rollback(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    ticket: DevHeadTicket,
) -> Result<(), String> {
    bundle
        .mtp_restore(gpu, ticket.snapshot)
        .map_err(|error| error.to_string())?;
    bundle
        .set_mtp_draft_request_state(ticket.draft)
        .map_err(|error| error.to_string())
}
