// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native Qwen4 MTP acceptance and transaction-shape helpers.
//!
//! The runtime owns the speculative loop and its pending-seed contract.  This
//! module only records the native MTP count convention and lowers an already
//! verified greedy result onto the canonical runtime types.  In particular,
//! the seed is never copied into `MtpWindow::committed` or `SpecStep::emit`.
//! Target rollback counts accepted drafts only; the position helpers take the
//! consumed-row count, which adds the seed.

use crate::bundle::Qwen4Bundle;
#[cfg(any(test, feature = "reference-parity"))]
use crate::reference_mtp::{MtpError, Qwen4MtpState};
use crate::state::Qwen4StateSnapshot;
use hipfire_runtime::spec::{
    accept_greedy_prefix, GreedyAccept, MtpDrafter, MtpSpeculator, MtpWindow, SpecAdvance,
    SpecGrammar, SpecRequestConfig, SpecScratch, SpecStep, SpecTarget, Speculator,
};
use rdna_compute::profile::{unix_micros, Span, SpanProfiler};
use rdna_compute::{Gpu, GpuTensor};
use std::time::Instant;

/// Device-side phase timing for one native MTP window.
///
/// Enabled by `HIPFIRE_MTP_PHASE_TIMING=1`.  Each phase is one `hipEvent` pair
/// recorded on the null stream and resolved once at window end, so the
/// instrument never synchronizes mid-window and never changes launch order:
/// the number it prints is the GPU time the phase actually occupies.
struct MtpPhaseTimers {
    spans: SpanProfiler,
    open: Option<Span>,
}

impl MtpPhaseTimers {
    fn new() -> Self {
        Self {
            spans: SpanProfiler::new(
                std::env::var("HIPFIRE_MTP_PHASE_TIMING").is_ok_and(|value| value == "1"),
            ),
            open: None,
        }
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
        if totals.is_empty() {
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

/// Native MTP uses greedy target picks only.  A sampled request must fail
/// closed until an exact distribution-verification implementation exists.
pub fn require_native_greedy(temp: f32) -> Result<(), String> {
    if !temp.is_finite() || temp.abs() > 1.0e-6 {
        return Err(
            "Qwen4 native MTP supports greedy verification only; sampled MTP requires exact distribution verification"
                .to_string(),
        );
    }
    Ok(())
}

/// Refuse native MTP prefill requests that would reuse a cached suffix.
///
/// Native MTP prefill is currently a cold, position-zero operation.  Silent
/// truncation of a cache-hit suffix would leave target and MTP state at
/// different positions, so callers must reject it before touching either
/// owner.
pub fn validate_native_mtp_prefill_request(
    prompt_tokens: &[u32],
    fill_tokens: &[u32],
    start_pos: usize,
    cache_hit: bool,
) -> Result<(), String> {
    if prompt_tokens.is_empty() {
        return Err("Qwen4 native MTP prefill requires at least one prompt token".to_string());
    }
    if cache_hit {
        return Err("Qwen4 native MTP prefill refuses cache-hit suffix reuse".to_string());
    }
    if start_pos != 0 {
        return Err(format!(
            "Qwen4 native MTP prefill requires position zero, got {start_pos}"
        ));
    }
    if fill_tokens != prompt_tokens {
        return Err("Qwen4 native MTP prefill requires a complete prompt fill".to_string());
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
    target_snapshot: Option<Qwen4StateSnapshot>,
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

    fn new_spec_scratch(
        &mut self,
        gpu: &mut Gpu,
        block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        let block_size = block_size.max(1);
        let max_chunk = self
            .execution
            .as_ref()
            .ok_or_else(|| "Qwen4 spec scratch requires attached forward resources".to_string())?
            .scratch
            .max_chunk;
        if block_size > max_chunk {
            return Err(format!(
                "Qwen4 spec block size {block_size} exceeds forward capacity {max_chunk}"
            ));
        }
        self.ensure_spec_hidden(gpu, block_size)
            .map_err(|error| error.to_string())?;
        Ok(Box::new(Qwen4SpecScratch {
            block_size,
            target_snapshot: None,
        }))
    }

    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
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
            .scratch
            .max_chunk;
        let mut offset = 0usize;
        let mut last_argmax = None;
        while offset < tokens.len() {
            if abort() {
                self.reset_recurrent(gpu)?;
                return Ok(SpecAdvance::Aborted);
            }
            let end = (offset + max_chunk).min(tokens.len());
            let picks = self
                .spec_forward_rows(gpu, &tokens[offset..end], false)
                .map_err(|error| error.to_string())?;
            last_argmax = picks.last().copied();
            offset = end;
        }
        if self.state.position != end_pos {
            return Err(format!(
                "Qwen4 spec advance ended at {}, expected {end_pos}",
                self.state.position
            ));
        }
        Ok(SpecAdvance::Ready {
            last_argmax: last_argmax.expect("non-empty spec advance produced no argmax"),
            last_logits: None,
        })
    }

    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        let block_size = {
            let s = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .ok_or("Qwen4 verify_block: scratch is not Qwen4SpecScratch")?;
            if s.target_snapshot.is_some() {
                return Err("Qwen4 verify_block: target snapshot is already active".to_string());
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
        let snapshot = self.snapshot(gpu).map_err(|error| error.to_string())?;
        scratch
            .as_any_mut()
            .downcast_mut::<Qwen4SpecScratch>()
            .ok_or("Qwen4 verify_block: scratch is not Qwen4SpecScratch")?
            .target_snapshot = Some(snapshot);
        let result = self
            .spec_forward_rows(gpu, block, true)
            .map_err(|error| error.to_string());
        if let Err(error) = &result {
            let snapshot = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .and_then(|s| s.target_snapshot.take());
            if let Some(snapshot) = snapshot {
                self.restore(gpu, snapshot)
                    .map_err(|restore| format!("{error}; target restore failed: {restore}"))?;
            }
            return Err(error.clone());
        }
        if self.state.position != end_pos {
            let mismatch = format!(
                "Qwen4 verify ended at {}, expected {end_pos}",
                self.state.position
            );
            let snapshot = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .and_then(|s| s.target_snapshot.take());
            if let Some(snapshot) = snapshot {
                self.restore(gpu, snapshot)
                    .map_err(|restore| format!("{mismatch}; target restore failed: {restore}"))?;
            }
            return Err(mismatch);
        }
        result
    }

    fn commit_prefix(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        accept_len: usize,
        position: usize,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        if block.is_empty() {
            return Err("Qwen4 commit_prefix cannot commit an empty block".to_string());
        }
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
        let snapshot = scratch
            .as_any_mut()
            .downcast_mut::<Qwen4SpecScratch>()
            .ok_or("Qwen4 commit_prefix: scratch is not Qwen4SpecScratch")?
            .target_snapshot
            .take()
            .ok_or("Qwen4 commit_prefix: target snapshot is not active")?;
        if accept_len == draft_len {
            return self
                .commit(gpu, snapshot)
                .map_err(|error| error.to_string());
        }
        self.restore(gpu, snapshot)
            .map_err(|error| error.to_string())?;
        self.spec_forward_rows(gpu, &block[..accept_len + 1], true)
            .map(|_| ())
            .map_err(|error| error.to_string())?;
        if self.state.position != committed_end {
            return Err(format!(
                "Qwen4 commit_prefix replay ended at {}, expected {committed_end}",
                self.state.position
            ));
        }
        Ok(())
    }

    fn eos_token(&self) -> u32 {
        self.config.eos_token_id
    }

    fn ctx_capacity(&self) -> usize {
        self.state.max_seq_len
    }
}

/// Draft-step hidden conditioning mode; see [`Qwen4MtpDrafter::draft_pairing`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DraftPairing {
    HeadState,
    AlignedHead,
    AlignedTarget,
}

/// Native GPU MTP drafter.  The MTP operator/state stay model-owned by the
/// target bundle; this adapter owns only the reusable verifier scratch and one
/// pending target-hidden row needed to seed each MTP window.
pub struct Qwen4MtpDrafter {
    max_k: usize,
    ctx_capacity: usize,
    request: SpecRequestConfig,
    scratch: Option<Box<dyn SpecScratch>>,
    pending_hidden: Option<GpuTensor>,
    row_hidden: Option<GpuTensor>,
    /// Prompt rows one chunked prefill call may capture; mirrors the attached
    /// forward's chunk capacity and sizes the spec hidden capture buffer.
    prefill_rows: usize,
}

impl Qwen4MtpDrafter {
    pub fn new(max_k: usize, ctx_capacity: usize) -> Self {
        Self {
            max_k: max_k.clamp(1, 10),
            ctx_capacity,
            request: SpecRequestConfig::default(),
            scratch: None,
            pending_hidden: None,
            row_hidden: None,
            prefill_rows: 0,
        }
    }

    fn bundle<'a>(target: &'a mut dyn SpecTarget) -> Result<&'a mut Qwen4Bundle, String> {
        target
            .as_any_mut()
            .downcast_mut::<Qwen4Bundle>()
            .ok_or_else(|| "Qwen4MtpDrafter: target is not a Qwen4Bundle".to_string())
    }

    fn ensure_resources(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
    ) -> Result<(), String> {
        let width = {
            let bundle = Self::bundle(target)?;
            bundle.mtp_position().map_err(|error| error.to_string())?;
            bundle
                .config
                .hc_count
                .checked_mul(bundle.config.hidden_size)
                .ok_or_else(|| "Qwen4 MTP hidden width overflow".to_string())?
        };
        let prefill_rows = {
            let bundle = Self::bundle(target)?;
            bundle.spec_chunk_rows().unwrap_or(1).max(1)
        };
        if self.scratch.is_none() {
            let scratch = target.new_spec_scratch(gpu, (self.max_k + 1).max(prefill_rows))?;
            self.scratch = Some(scratch);
        }
        self.prefill_rows = prefill_rows;
        if self.pending_hidden.is_none() {
            self.pending_hidden = Some(
                gpu.zeros(&[width], rdna_compute::DType::F32)
                    .map_err(|error| format!("Qwen4 MTP pending hidden allocation: {error}"))?,
            );
        }
        if self.row_hidden.is_none() {
            self.row_hidden = Some(
                gpu.zeros(&[width], rdna_compute::DType::F32)
                    .map_err(|error| format!("Qwen4 MTP row hidden allocation: {error}"))?,
            );
        }
        Ok(())
    }

    fn row_hidden(&self) -> Result<&GpuTensor, String> {
        self.row_hidden
            .as_ref()
            .ok_or_else(|| "Qwen4 MTP row hidden is not allocated".to_string())
    }

    /// Native MTP draft-step conditioning.  `HeadState` is the historical
    /// chain: row 0 is fed the window's pending hidden and every later row is
    /// fed the head's own previous-step hidden (`forward_token` with no target hidden).
    /// `AlignedHead` feeds the target's hidden of the token being processed at
    /// row 0 and the head's own hidden afterwards; `AlignedTarget` feeds the
    /// target's same-row hidden at every row.  Selectable so the pairing can be
    /// measured rather than argued: `HIPFIRE_MTP_PAIRING=aligned-head|aligned`.
    fn draft_pairing() -> DraftPairing {
        match std::env::var("HIPFIRE_MTP_PAIRING").as_deref() {
            Ok("aligned-head") => DraftPairing::AlignedHead,
            Ok("aligned") => DraftPairing::AlignedTarget,
            _ => DraftPairing::HeadState,
        }
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
    fn mtp_step_incremental(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        k: usize,
        eos: u32,
        trace: bool,
    ) -> Result<MtpWindow, String> {
        let pairing = Self::draft_pairing();
        let mut timers = MtpPhaseTimers::new();
        let mut committed: Vec<u32> = Vec::with_capacity(k + 1);
        let mut drafts: Vec<u32> = Vec::with_capacity(k);
        let mut picks: Vec<u32> = Vec::with_capacity(k + 1);
        let mut token = seed;
        let mut accepted = 0usize;
        let mut row = 0usize;
        loop {
            let token_position = position
                .checked_add(row)
                .ok_or_else(|| "Qwen4 MTP incremental position overflow".to_string())?;
            timers.mark(gpu, "target_row");
            let pick = {
                let bundle = Self::bundle(target)?;
                bundle
                    .spec_capture_token(gpu, token)
                    .map_err(|error| error.to_string())?
            };
            picks.push(pick);
            let row_hidden = self.row_hidden()?;
            {
                let bundle = Self::bundle(target)?;
                bundle
                    .copy_spec_hidden_row_to(gpu, 0, row_hidden)
                    .map_err(|error| error.to_string())?;
            }
            if row == k {
                // Every draft matched: this row's pick is the bonus.  The head
                // must still consume the row's token so its own position lands on
                // the committed end, exactly as the batched path's full-accept
                // branch does with one extra `mtp_forward_token`.
                timers.mark(gpu, "draft_step");
                let hidden = match pairing {
                    DraftPairing::HeadState => None,
                    DraftPairing::AlignedHead | DraftPairing::AlignedTarget => Some(row_hidden),
                };
                {
                    let bundle = Self::bundle(target)?;
                    bundle
                        .mtp_advance_token(gpu, token, hidden, token_position, row == 0)
                        .map_err(|error| error.to_string())?;
                }
                committed.push(pick);
                break;
            }
            timers.mark(gpu, "draft_step");
            let hidden = match pairing {
                DraftPairing::HeadState if row > 0 => None,
                DraftPairing::AlignedHead if row > 0 => None,
                DraftPairing::HeadState => Some(self.pending_hidden()?),
                DraftPairing::AlignedHead | DraftPairing::AlignedTarget => Some(row_hidden),
            };
            let draft = {
                let bundle = Self::bundle(target)?;
                bundle
                    .mtp_forward_token(gpu, token, hidden, token_position, row == 0)
                    .map_err(|error| error.to_string())?
            };
            drafts.push(draft);
            if draft == pick {
                accepted += 1;
                committed.push(draft);
                if draft == eos {
                    break;
                }
                token = draft;
                row += 1;
            } else {
                committed.push(pick);
                break;
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
            gpu.copy_d2d(row_hidden, pending, pending.byte_size())
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
        if trace || timers.enabled() {
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
            eprintln!(
                "QWEN4_MTP_TRACE {{\"event\":\"window\",\"position\":{position},\"k\":{k},\"accepted\":{accepted},\"rows\":[{rows}],\"committed\":{committed:?}}}"
            );
        }
        if timers.enabled() {
            let fields = format!(
                "\"position\":{position},\"k\":{k},\"accepted\":{accepted},\"pairing\":\"{}\",\"t_end\":{}",
                match pairing {
                    DraftPairing::HeadState => "head-state",
                    DraftPairing::AlignedHead => "aligned-head",
                    DraftPairing::AlignedTarget => "aligned-target",
                },
                unix_micros()
            );
            timers.finish(gpu, "QWEN4_MTP_PHASE", &fields);
        }
        Ok(MtpWindow {
            committed,
            accepted,
            drafts_generated: drafts.len(),
        })
    }

    fn pending_hidden(&self) -> Result<&GpuTensor, String> {
        self.pending_hidden
            .as_ref()
            .ok_or_else(|| "Qwen4 MTP pending hidden is not allocated".to_string())
    }
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
    ) -> Result<u32, String> {
        require_native_greedy(self.request.temp)?;
        validate_native_mtp_prefill_request(prompt_tokens, fill_tokens, start_pos, cache_hit)?;
        // Native Qwen4 MTP has no exact target+MTP suffix rehydration yet.
        // Always discard any AR or stale MTP prefix and rebuild the complete
        // rendered prompt from position zero.
        target.reset_recurrent(gpu)?;
        self.ensure_resources(gpu, target)?;
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
        let pending = self.pending_hidden()?;
        let mut first_token = None;
        // One chunked target forward per chunk instead of one single-row forward
        // per prompt token: the shared forward already captures the whole
        // chunk's wide hidden, and the head then consumes each row in order.
        // Cost goes from ~1 single-row forward per prompt token to the ordinary
        // chunked prefill rate plus one head step per token.
        // Each prompt token selects with its own query and the pooled keys
        // visible at that position, regardless of target prefill chunking.
        let chunk_rows = self.prefill_rows.max(1);
        for (chunk_index, chunk) in fill_tokens.chunks(chunk_rows).enumerate() {
            if abort() {
                target.reset_recurrent(gpu)?;
                return Err("Qwen4 native MTP prefill aborted".to_string());
            }
            let base = chunk_index * chunk_rows;
            let pick = Self::bundle(target)?
                .spec_prefill_rows(gpu, chunk)
                .map_err(|error| error.to_string())?;
            for (index, &token) in chunk.iter().enumerate() {
                if abort() {
                    target.reset_recurrent(gpu)?;
                    return Err("Qwen4 native MTP prefill aborted".to_string());
                }
                let position = start_pos
                    .checked_add(base)
                    .and_then(|value| value.checked_add(index))
                    .ok_or_else(|| "Qwen4 native MTP prefill position overflow".to_string())?;
                let bundle = Self::bundle(target)?;
                bundle
                    .copy_spec_hidden_row_to(gpu, index, pending)
                    .map_err(|error| error.to_string())?;
                bundle
                    .mtp_advance_token(gpu, token, Some(pending), position, true)
                    .map_err(|error| error.to_string())?;
            }
            first_token = Some(pick);
        }
        Ok(first_token.expect("non-empty MTP prefill produced no seed"))
    }

    fn mtp_step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        _emitted: &[u32],
        k: usize,
        eos: u32,
        _grammar: Option<&mut dyn SpecGrammar>,
    ) -> Result<MtpWindow, String> {
        require_native_greedy(self.request.temp)?;
        if k > self.max_k {
            return Err(format!(
                "Qwen4 native MTP draft budget {k} exceeds configured K {}",
                self.max_k
            ));
        }
        self.ensure_resources(gpu, target)?;
        let trace = std::env::var("HIPFIRE_MTP_TRACE").is_ok_and(|value| value == "1");
        // Incremental verification is the default on every GPU; batching is opt-in.
        let incremental = !matches!(std::env::var("HIPFIRE_MTP_INCREMENTAL").as_deref(), Ok("0"));
        if incremental {
            return self.mtp_step_incremental(gpu, target, position, seed, k, eos, trace);
        }
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
        let mut snapshot = {
            let bundle = Self::bundle(target)?;
            Some(
                bundle
                    .mtp_snapshot(gpu)
                    .map_err(|error| error.to_string())?,
            )
        };
        let mut timers = MtpPhaseTimers::new();
        let mut accepted_drafts = 0usize;
        let window_start = Instant::now();
        let result = (|| -> Result<MtpWindow, String> {
            let mut drafts = Vec::with_capacity(k);
            let mut input = seed;
            // Every proposal starts with a fresh QSA selection; only later
            // draft rows within this window reuse it.
            for index in 0..k {
                let hidden = if index == 0 {
                    Some(self.pending_hidden()?)
                } else {
                    None
                };
                let token_position = position
                    .checked_add(index)
                    .ok_or_else(|| "Qwen4 MTP step position overflow".to_string())?;
                input = Self::bundle(target)?
                    .mtp_forward_token(gpu, input, hidden, token_position, index == 0)
                    .map_err(|error| error.to_string())?;
                drafts.push(input);
            }
            let mut block = Vec::with_capacity(k + 1);
            block.push(seed);
            block.extend_from_slice(&drafts);
            timers.mark(gpu, "verify");
            let picks = Self::bundle(target)?;
            let pending_hidden = self
                .pending_hidden
                .as_ref()
                .ok_or_else(|| "Qwen4 native MTP pending hidden is not allocated".to_string())?;
            let scratch = self
                .scratch
                .as_mut()
                .ok_or_else(|| "Qwen4 native MTP verify scratch is not allocated".to_string())?;
            let target_picks = picks
                .verify_block(gpu, &block, position, scratch.as_mut(), None)
                .map_err(|error| error.to_string())?;
            let acceptance = accept_native_greedy(&drafts, &target_picks, Some(eos))?;
            accepted_drafts = acceptance.accepted;
            if trace {
                let rows = drafts
                    .iter()
                    .zip(target_picks.iter())
                    .map(|(draft, pick)| {
                        format!(
                            "{{\"draft\":{draft},\"pick\":{pick},\"match\":{}}}",
                            draft == pick
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!(
                    "QWEN4_MTP_TRACE {{\"event\":\"window\",\"position\":{position},\"k\":{k},\"accepted\":{accepted_drafts},\"rows\":[{rows}],\"committed\":{:?},\"baseline\":true}}",
                    acceptance.committed
                );
            }
            let target_accept_len = target_commit_accept_len(&acceptance);
            let full_accept = target_accept_len == k;
            let target_scratch = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .ok_or("Qwen4 native MTP target scratch type changed")?;
            let target_snapshot = target_scratch
                .target_snapshot
                .ok_or("Qwen4 native MTP target snapshot disappeared")?;

            // Keep both pre-window tickets active until every replay and hidden
            // copy succeeds. A retained restore lets the outer rollback repair
            // both owners if either side's GPU work fails.
            if !full_accept {
                timers.mark(gpu, "target_replay");
                picks
                    .restore_retain(gpu, target_snapshot)
                    .map_err(|error| error.to_string())?;
                picks
                    .spec_forward_rows(gpu, &block[..target_accept_len + 1], true)
                    .map(|_| ())
                    .map_err(|error| error.to_string())?;
            }
            timers.mark(gpu, "mtp_replay");
            let mtp_ticket = snapshot
                .as_ref()
                .copied()
                .expect("MTP snapshot remains active until transaction commit");
            if full_accept && k > 0 {
                let last_draft = *drafts
                    .last()
                    .ok_or_else(|| "Qwen4 native MTP full accept has no final draft".to_string())?;
                let last_position = position
                    .checked_add(k)
                    .ok_or_else(|| "Qwen4 MTP step position overflow".to_string())?;
                picks
                    .mtp_advance_token(gpu, last_draft, None, last_position, false)
                    .map_err(|error| error.to_string())?;
            } else {
                picks
                    .mtp_restore_retain(gpu, mtp_ticket)
                    .map_err(|error| error.to_string())?;
                // Snapshot restore brings back the pre-window QSA selection;
                // replay must reselect on its first row.
                for (index, &token) in block[..target_accept_len + 1].iter().enumerate() {
                    let hidden = if index == 0 {
                        Some(pending_hidden)
                    } else {
                        None
                    };
                    let token_position = position
                        .checked_add(index)
                        .ok_or_else(|| "Qwen4 MTP step position overflow".to_string())?;
                    picks
                        .mtp_advance_token(gpu, token, hidden, token_position, index == 0)
                        .map_err(|error| error.to_string())?;
                }
            }
            timers.mark(gpu, "commit");
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
                .copy_spec_hidden_row_to(gpu, target_accept_len, pending_hidden)
                .map_err(|error| error.to_string())?;

            let window = MtpWindow {
                committed: acceptance.committed,
                accepted: acceptance.accepted,
                drafts_generated: drafts.len(),
            };
            // All fallible GPU operations are complete. Validate both tickets
            // before invalidating either arena, then perform the no-copy commit
            // boundary and clear the target scratch ticket.
            picks
                .validate_commit(target_snapshot)
                .map_err(|error| error.to_string())?;
            picks
                .mtp_validate_commit(mtp_ticket)
                .map_err(|error| error.to_string())?;
            picks.commit_validated(target_snapshot);
            picks.mtp_commit_validated(mtp_ticket);
            target_scratch.target_snapshot = None;
            snapshot = None;
            Ok(window)
        })();
        if timers.enabled() {
            let fields = format!(
                "\"position\":{position},\"k\":{k},\"accepted\":{accepted_drafts},\"window_us\":{:.1},\"t_end\":{}",
                window_start.elapsed().as_secs_f64() * 1e6,
                unix_micros()
            );
            timers.finish(gpu, "QWEN4_MTP_PHASE", &fields);
        }
        if let Err(error) = &result {
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
            if let Some(ticket) = target_ticket {
                if let Err(rollback) = Self::bundle(target).and_then(|bundle| {
                    bundle
                        .restore(gpu, ticket)
                        .map_err(|restore| restore.to_string())
                }) {
                    rollback_errors.push(format!("target rollback failed: {rollback}"));
                }
            }
            if let Some(ticket) = snapshot.take() {
                if let Err(rollback) = Self::bundle(target).and_then(|bundle| {
                    bundle
                        .mtp_restore(gpu, ticket)
                        .map_err(|restore| restore.to_string())
                }) {
                    rollback_errors.push(format!("MTP rollback failed: {rollback}"));
                }
            }
            if !rollback_errors.is_empty() {
                return Err(format!(
                    "{error}; rollback failed: {}",
                    rollback_errors.join("; ")
                ));
            }
        }
        result
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
        self.ensure_resources(gpu, target)?;
        let pending = self.pending_hidden()?;
        for (index, &token) in tokens.iter().enumerate() {
            if abort() {
                return Ok(true);
            }
            let position = start_pos
                .checked_add(index)
                .ok_or_else(|| "Qwen4 native MTP forced position overflow".to_string())?;
            let bundle = Self::bundle(target)?;
            bundle
                .spec_capture_token(gpu, token)
                .map_err(|error| error.to_string())?;
            bundle
                .copy_spec_hidden_row_to(gpu, 0, pending)
                .map_err(|error| error.to_string())?;
            bundle
                .mtp_advance_token(gpu, token, Some(pending), position, true)
                .map_err(|error| error.to_string())?;
        }
        Ok(true)
    }

    fn mtp_reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
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
            ..
        } = *self;
        if let Some(scratch) = scratch {
            scratch.free(gpu);
        }
        if let Some(hidden) = pending_hidden {
            let _ = gpu.free_tensor(hidden);
        }
    }

    fn k(&self) -> usize {
        self.max_k
    }

    fn proposal_capacity(&self) -> usize {
        self.max_k
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn requires_greedy(&self) -> bool {
        true
    }

    fn configure_request(&mut self, cfg: SpecRequestConfig) {
        self.request = cfg;
    }

    fn supports_temp_verify(&self) -> bool {
        false
    }
}

/// Build the generic runtime adapter around the native Qwen4 GPU MTP core.
pub fn build_qwen4_mtp_speculator(max_k: usize, ctx_capacity: usize) -> Box<dyn Speculator> {
    Box::new(MtpSpeculator::new(Qwen4MtpDrafter::new(
        max_k,
        ctx_capacity,
    )))
}

#[cfg(any(test, feature = "reference-parity"))]
/// Convert an MTP request-state error into the erased runtime error type.
pub fn mtp_error(error: MtpError) -> String {
    format!("Qwen4 native MTP: {error}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference_mtp::MtpQsaGeometry;

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
                rope_theta: 10_000,
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
            rope_theta: 10_000,
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
}
