// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Target-generic DFlash2 speculator (`name() == "dflash2"`).
//!
//! DFlash2 (z-lab `dflash/model.py`, `DFlash2DraftModel`) drafts a block
//! `[seed, MASK × (B-1)]` in one non-causal sliding-window forward conditioned
//! on the target's features at its `target_layer_ids`, then a candidate
//! selector walks one path through the per-position top-K of the draft
//! logits (the TARGET head over the draft's final rows):
//! `score = unary + <pred_codebook[prev] ⊙ proj(h_t), succ_codebook[cand]>`.
//!
//! - Greedy: the selector's argmax walk; acceptance is the longest prefix
//!   equal to the target's per-row argmax, so the output is the AR stream.
//! - Sampled: the selector draws each row from `q = softmax(score / T)` over
//!   its candidates; a draft is accepted iff `u * q(x) < p(x)` and a rejection
//!   draws from the residual `(p - q)+` (`_rejection_sample` with
//!   `draft_indices`), with `p` the AR sampler's own truncated distribution
//!   ([`SparseDist::build_from_logits`]), so emitted tokens are distributed
//!   as AR's.
//! - Verify depth: a window verifies only the draft prefix whose running
//!   product of selector confidences (each row's top-candidate mass, fixed
//!   before that row's draw) stays at or above [`VERIFY_CONFIDENCE`] (at
//!   least one draft); the draft forward still proposes the whole block.
//!   Whether row `r` is verified depends on the drafts before it only, so a
//!   sampled verified row is still distributed as its `q` (a cut on the drawn
//!   token's own mass is not: see the target-law test below).
//!
//! The target is reached only through [`SpecTarget`]: it captures the feature
//! rows (`dflash_extract_layers`), embeds the noise block and applies its head
//! to the draft rows on the device ([`SpecTarget::lm_head_logits_device`]).
//! Chain only: there is no tree arm (the selector is the proposal).

use crate::dflash::{draft_forward_based, propose_candidates_device, DflashConfig, DflashScratch};
use crate::dflash::{DflashWeights, DFLASH_BLOCK_MAX};
use crate::hfq::HfqFile;
use crate::llama::argmax;
use crate::sampler::{apply_logit_policy_cpu, SamplerConfig};
use crate::spec::{
    accept_greedy_prefix, terminal_prefix_replay, PrefillOutcome, RowPick, SpecAdvance,
    SpecGrammar, SpecRequestConfig, SpecScratch, SpecStep, SpecTarget, Speculator,
};
use crate::spec_sampling::{
    accept_sampled_prefix, PenaltyHistory, SampleSpec, SparseDist, SpecRng,
};
use rdna_compute::Gpu;
use std::path::Path;

/// One draft window as the oracle sees it: what the draft forward and the
/// selector produced before any verify ([`trace_windows_begin`]).
#[derive(Clone, Debug, PartialEq)]
pub struct DraftWindowTrace {
    pub position: usize,
    pub seed: u32,
    pub block: usize,
    /// The selector's drafted tokens (`block - 1`).
    pub tokens: Vec<u32>,
    /// Bit patterns of the selector's per-row confidences.
    pub confidence_bits: Vec<u32>,
    /// FNV-1a over the bit patterns of the draft's final hidden rows `1..block`.
    pub hidden_fnv: u64,
    /// Device-synced ms of the draft forward alone.
    pub draft_ms: f64,
    /// Device-synced ms of the target head + selector over the draft rows.
    pub selector_ms: f64,
}

static WINDOW_TRACE: std::sync::Mutex<Option<Vec<DraftWindowTrace>>> = std::sync::Mutex::new(None);
/// 0 = follow `HIPFIRE_DFLASH_DRAFT_GRAPH`, 1 = force off, 2 = force on.
static DRAFT_GRAPH_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Developer oracle hook: from now on every speculator window records a
/// [`DraftWindowTrace`] (and device-syncs around the draft forward to time
/// it). Process-wide; [`trace_windows_take`] ends it.
pub fn trace_windows_begin() {
    *WINDOW_TRACE.lock().unwrap() = Some(Vec::new());
}

/// The windows traced since [`trace_windows_begin`]; tracing stops.
pub fn trace_windows_take() -> Vec<DraftWindowTrace> {
    WINDOW_TRACE.lock().unwrap().take().unwrap_or_default()
}

fn tracing_windows() -> bool {
    WINDOW_TRACE.lock().unwrap().is_some()
}

/// Developer oracle hook: force the graph-captured draft window on (`Some(true)`),
/// off (`Some(false)`), or back to `HIPFIRE_DFLASH_DRAFT_GRAPH` (`None`) for every
/// speculator built after this call.
pub fn set_draft_graph_override(on: Option<bool>) {
    let v = match on {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    DRAFT_GRAPH_OVERRIDE.store(v, std::sync::atomic::Ordering::Relaxed);
}

fn fnv1a_f32_bits(v: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in v {
        for byte in x.to_bits().to_le_bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// Graph counters of the most recently freed speculator's scratch (oracle read-out).
static LAST_GRAPH_STATS: std::sync::Mutex<Option<crate::dflash::DraftGraphStats>> =
    std::sync::Mutex::new(None);

/// Draft-graph counters of the last speculator freed (see [`Speculator::free`]).
pub fn last_draft_graph_stats() -> Option<crate::dflash::DraftGraphStats> {
    LAST_GRAPH_STATS.lock().unwrap().clone()
}

/// Target-generic DFlash2 chain speculator. Owns the draft, its scratch, the
/// cumulative committed feature rows (`[rows × feature_width]` F32,
/// the draft's context) and the target-minted verify scratch.
pub struct Dflash2Speculator {
    weights: DflashWeights,
    scratch: DflashScratch,
    config: DflashConfig,
    /// Feature rows of committed positions `host_base..`, in order: grows by
    /// exactly `accepted + 1` rows per window (the bonus row arrives as the
    /// next seed). With a windowed draft only the window is kept
    /// ([`trim_feature_rows`]); otherwise `host_base` stays 0.
    target_hidden_host: Vec<f32>,
    host_base: usize,
    /// Token of every committed position `0..host_end()`: tells a session
    /// reuse whether the rows below its restart belong to that prompt.
    host_tokens: Vec<u32>,
    /// Rows the draft reads (its declared sliding window); `None` keeps every
    /// row.
    feature_window: Option<usize>,
    verify_scratch: Option<Box<dyn SpecScratch>>,
    block_size: usize,
    ctx_capacity: usize,
    /// Embedding row of the mask token (the draft's noise), filled once.
    mask_row: Vec<f32>,
    request: SpecRequestConfig,
    rng: SpecRng,
    /// The request's penalty history (window 0 while its penalties are
    /// neutral): seeded at prefill, rebuilt from the loop's `emitted` at every
    /// window ([`PenaltyHistory`]).
    history: PenaltyHistory,
    /// The request's penalty policy for [`apply_logit_policy_cpu`], the same
    /// arithmetic AR's `sample_cpu` runs before its temperature stage.
    penalty_cfg: SamplerConfig,
    /// The target's measured break-even line for this draft
    /// ([`SpecTarget::dflash_retire_break_even`]); `Some` arms the AR step.
    retire: Option<crate::spec_retire::BreakEven>,
}

impl Dflash2Speculator {
    fn row_elems(&self) -> usize {
        self.config.feature_width()
    }

    fn sampled(&self) -> bool {
        self.request.temp > 1e-6
    }

    fn sample_spec(&self) -> SampleSpec {
        SampleSpec::cpu_ar(
            self.request.temp,
            self.request.top_p,
            self.request.top_k,
            self.request.min_p,
        )
    }

    /// One draw (sampled) or argmax (greedy) from the target row `logits`
    /// after the request's penalties over `self.history.row(0)`: AR's
    /// `sample_cpu` pick. The window ([`PenaltyHistory::begin_window`] /
    /// `set_prompt`) must be open.
    fn pick_penalized_row(&mut self, logits: &mut [f32]) -> Result<u32, String> {
        apply_logit_policy_cpu(logits, self.history.row(0), &self.penalty_cfg);
        if self.sampled() {
            self.sample_row(logits)
        } else {
            Ok(argmax(logits))
        }
    }

    /// One draw from the target row `logits` with the AR sampler's law.
    fn sample_row(&mut self, logits: &[f32]) -> Result<u32, String> {
        let mut p = SparseDist::default();
        p.build_from_logits(logits, self.sample_spec(), &mut Vec::new())?;
        Ok(p.sample(self.rng.next_f32()))
    }

    fn verify_scratch(&mut self) -> Result<&mut Box<dyn SpecScratch>, String> {
        self.verify_scratch
            .as_mut()
            .ok_or_else(|| "Dflash2Speculator: verify scratch already freed".to_string())
    }

    /// Committed positions the host rows end at.
    fn host_end(&self) -> usize {
        self.host_base + self.target_hidden_host.len() / self.row_elems()
    }

    fn check_host_rows(&self, position: usize) -> Result<(), String> {
        if self.target_hidden_host.len() % self.row_elems() == 0 && self.host_end() == position {
            Ok(())
        } else {
            Err(format!(
                "Dflash2Speculator: {} feature floats from row {}, expected rows up to {position}",
                self.target_hidden_host.len(),
                self.host_base
            ))
        }
    }

    /// After the target appended rows through position `end` (and may have
    /// trimmed leading rows itself): re-derive the base, then keep the window.
    fn rebase_host(&mut self, end: usize) -> Result<(), String> {
        let rows = self.target_hidden_host.len() / self.row_elems();
        self.host_base = end.checked_sub(rows).ok_or_else(|| {
            format!("Dflash2Speculator: {rows} feature rows end past position {end}")
        })?;
        self.trim_host();
        Ok(())
    }

    fn trim_host(&mut self) {
        if let Some(window) = self.feature_window {
            let row_elems = self.row_elems();
            self.host_base += trim_feature_rows(&mut self.target_hidden_host, row_elems, window);
        }
    }

    fn clear_host(&mut self) {
        self.target_hidden_host.clear();
        self.target_hidden_host.shrink_to_fit();
        self.host_base = 0;
        self.host_tokens.clear();
        self.scratch.reset_upload_tracking();
    }

    /// The target continues `prompt` at `start` (a session restore or a live
    /// continuation): keep the host rows below `start` or restart the draft
    /// context empty there ([`draft_resume_base`]). A hit the draft cannot
    /// resume fails; the caller asks [`Speculator::can_resume_at`] first.
    fn reuse_host(&mut self, prompt: &[u32], start: usize) -> Result<(), String> {
        let base = draft_resume_base(
            &self.host_tokens,
            self.host_base,
            self.feature_window,
            prompt,
            start,
        )
        .ok_or_else(|| {
            format!(
                "Dflash2Speculator: feature rows {}..{} cannot resume the draft at {start} of a {}-token prompt",
                self.host_base,
                self.host_end(),
                prompt.len()
            )
        })?;
        if base == start {
            self.target_hidden_host.clear();
            self.host_tokens.clear();
            self.host_tokens.extend_from_slice(&prompt[..start]);
        } else {
            self.target_hidden_host
                .truncate((start - base) * self.row_elems());
            self.host_tokens.truncate(start);
        }
        self.host_base = base;
        // Device copies of rows at or past `start` are stale now.
        self.scratch.reset_upload_tracking();
        Ok(())
    }

    /// A window that failed after its verify started must not leave the
    /// target holding the verify's rollback ticket (the next verify would
    /// refuse): roll the target back to the window's start.
    fn settle<T>(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        result: Result<T, String>,
    ) -> Result<T, String> {
        let Err(error) = result else {
            return result;
        };
        match self.verify_scratch.as_mut() {
            Some(vs) => match target.abandon_verify(gpu, vs.as_mut()) {
                Ok(()) => Err(error),
                Err(rollback) => Err(format!("{error}; target rollback failed: {rollback}")),
            },
            None => Err(error),
        }
    }

    /// One target token, no draft: the committed row `[seed]`
    /// ([`SpecTarget::commit_row`], no rollback snapshot needed) with its
    /// feature capture, then the greedy or sampled pick, over the penalized
    /// row when the request has penalties (history = the loop's `emitted`,
    /// seed last, which the caller opened with `begin_window`).
    fn step_one_token(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
    ) -> Result<SpecStep, String> {
        let mut rows = Vec::with_capacity(self.row_elems());
        let sampled = self.sampled();
        let penalized = self.request.penalties_active();
        let vs = self.verify_scratch()?;
        let pick = target.commit_row(
            gpu,
            seed,
            position,
            vs.as_mut(),
            Some(&mut rows),
            sampled || penalized,
        )?;
        let token = match pick {
            RowPick::Logits(mut logits) if penalized => self.pick_penalized_row(&mut logits)?,
            RowPick::Logits(logits) => self.sample_row(&logits)?,
            RowPick::Argmax(token) => token,
        };
        self.append_rows(&rows, &[seed])?;
        Ok(SpecStep::new(std::iter::once(token), token, 0, 0).cap_emit(1))
    }

    /// Append the feature rows of the committed `tokens`.
    fn append_rows(&mut self, rows: &[f32], tokens: &[u32]) -> Result<(), String> {
        let elems = tokens.len() * self.row_elems();
        if rows.len() < elems {
            return Err(format!(
                "Dflash2Speculator: target captured {} feature floats, need {elems}",
                rows.len()
            ));
        }
        self.target_hidden_host.extend_from_slice(&rows[..elems]);
        self.host_tokens.extend_from_slice(tokens);
        self.trim_host();
        Ok(())
    }
}

impl Speculator for Dflash2Speculator {
    fn name(&self) -> &'static str {
        "dflash2"
    }

    fn prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        prompt_tokens: &[u32],
        prefill_tokens: &[u32],
        prefill_start: usize,
        cache_hit: bool,
        _resume_from: Option<usize>,
        abort: &dyn Fn() -> bool,
    ) -> Result<PrefillOutcome, String> {
        // The penalty history of the request: the rendered prompt (the whole
        // prompt, not the cache-fill suffix) unless the request penalizes
        // generated tokens only; opens the window of the first token's row.
        self.history
            .set_prompt(if self.request.penalty_generated_only {
                &[]
            } else {
                prompt_tokens
            });
        // A cold prefill resets the target and a hit restores or continues
        // it: either way no verify ticket is outstanding.
        if let Some(vs) = self.verify_scratch.as_mut() {
            vs.clear();
        }
        let (fill, start) = if cache_hit {
            self.reuse_host(prompt_tokens, prefill_start)?;
            (prefill_tokens, prefill_start)
        } else {
            self.clear_host();
            (prompt_tokens, 0)
        };
        self.check_host_rows(start)?;
        let advance = target.spec_advance(
            gpu,
            fill,
            start,
            !cache_hit,
            abort,
            Some(&mut self.target_hidden_host),
        )?;
        let (last_argmax, last_logits) = match advance {
            SpecAdvance::Aborted => return Ok(PrefillOutcome::Aborted),
            SpecAdvance::Ready {
                last_argmax,
                last_logits,
            } => (last_argmax, last_logits),
        };
        self.rebase_host(start + fill.len())?;
        self.host_tokens.extend_from_slice(fill);
        self.check_host_rows(start + fill.len())?;
        let penalized = self.request.penalties_active();
        let first_token = match last_logits {
            // The first token sees the history's prompt tail; an empty one
            // (generated-only) leaves the row untouched.
            Some(mut logits) if penalized && !self.history.row(0).is_empty() => {
                self.pick_penalized_row(&mut logits)?
            }
            Some(logits) if self.sampled() => self.sample_row(&logits)?,
            None if self.sampled() => {
                return Err(
                    "Dflash2Speculator: a sampled prefill needs the target's final-row logits"
                        .into(),
                );
            }
            None if penalized && !self.history.row(0).is_empty() => {
                return Err(
                    "Dflash2Speculator: a penalized prefill needs the target's final-row logits"
                        .into(),
                );
            }
            _ => last_argmax,
        };
        Ok(PrefillOutcome::Ready { first_token })
    }

    fn step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        _grammar: Option<&mut dyn SpecGrammar>,
        temp: f32,
        max_emit: usize,
    ) -> Result<SpecStep, String> {
        if max_emit == 0 {
            return Err("Dflash2Speculator: max_emit=0 (no remaining output budget)".into());
        }
        let penalized = self.request.penalties_active();
        if (temp > 1e-6) != self.sampled() {
            return Err(format!(
                "Dflash2Speculator: step temperature {temp} differs from the request's {}",
                self.request.temp
            ));
        }
        self.check_host_rows(position)?;
        if penalized {
            // Row 0 of this window sees `prompt ‖ emitted` (seed included).
            self.history.begin_window(emitted);
        }
        if max_emit == 1 {
            let result = self.step_one_token(gpu, target, position, seed);
            return self.settle(gpu, target, result);
        }
        // Block of b rows: b-1 drafts + bonus never exceed the budget.
        let b = self.block_size.min(max_emit).max(2);
        let h = self.config.hidden;
        let drafts_n = b - 1;

        // Noise block [seed, MASK × (b-1)]: raw target embedding rows.
        if self.mask_row.is_empty() {
            self.mask_row = target.embed_row(gpu, self.config.mask_token_id)?;
        }
        let seed_row = target.embed_row(gpu, seed)?;
        if seed_row.len() != h || self.mask_row.len() != h {
            return Err(format!(
                "Dflash2Speculator: target embedding rows have {} floats, the draft needs {h}",
                seed_row.len()
            ));
        }
        let mut noise = Vec::with_capacity(b * h);
        noise.extend_from_slice(&seed_row);
        for _ in 1..b {
            noise.extend_from_slice(&self.mask_row);
        }
        // Context rows sit at their absolute positions, the block after them.
        let positions_q: Vec<i32> = (position as i32..(position + b) as i32).collect();
        let positions_k: Vec<i32> = (0..(position + b) as i32).collect();
        let traced = tracing_windows();
        let t_draft = traced.then(|| {
            let _ = gpu.hip.device_synchronize();
            std::time::Instant::now()
        });
        draft_forward_based(
            gpu,
            &self.weights,
            &self.config,
            Some(&noise),
            Some(self.target_hidden_host.as_slice()),
            self.host_base,
            &positions_q,
            &positions_k,
            b,
            position,
            &mut self.scratch,
            false,
        )
        .map_err(|e| format!("Dflash2Speculator draft_forward: {e}"))?;
        let draft_ms = t_draft.map(|t| {
            let _ = gpu.hip.device_synchronize();
            t.elapsed().as_secs_f64() * 1e3
        });
        let t_selector = traced.then(std::time::Instant::now);

        // Selector over the target head's logits of the draft rows 1..b.
        let hidden_rows = self.scratch.x.sub_offset(h, drafts_n * h);
        let logits = target.lm_head_logits_device(gpu, &hidden_rows, drafts_n)?;
        let sampled = self.sampled();
        let uniforms: Vec<f32> = if sampled {
            (0..drafts_n).map(|_| self.rng.next_f32()).collect()
        } else {
            Vec::new()
        };
        let proposal = propose_candidates_device(
            gpu,
            &self.weights,
            &self.scratch,
            &hidden_rows,
            &logits,
            drafts_n,
            seed,
            if sampled { temp } else { 0.0 },
            sampled.then_some(uniforms.as_slice()),
        )
        .map_err(|e| format!("Dflash2Speculator selector: {e}"))?;
        if let (Some(draft_ms), Some(t_selector)) = (draft_ms, t_selector) {
            let selector_ms = t_selector.elapsed().as_secs_f64() * 1e3;
            let hidden = gpu
                .download_f32(&hidden_rows)
                .map_err(|e| format!("Dflash2Speculator trace download: {e}"))?;
            let trace = DraftWindowTrace {
                position,
                seed,
                block: b,
                tokens: proposal.tokens.clone(),
                confidence_bits: proposal.confidence.iter().map(|c| c.to_bits()).collect(),
                hidden_fnv: fnv1a_f32_bits(&hidden[..drafts_n * h]),
                draft_ms,
                selector_ms,
            };
            if let Some(log) = WINDOW_TRACE.lock().unwrap().as_mut() {
                log.push(trace);
            }
        }
        // Verify only the confident prefix of the drafts: an unconfident draft
        // rarely survives the verify but still costs its target row. The cut is
        // draft-side only, so greedy output stays AR's and sampled acceptance
        // keeps AR's law.
        let drafts = &proposal.tokens[..verify_depth(&proposal.confidence)];
        let mut block = Vec::with_capacity(drafts.len() + 1);
        block.push(seed);
        block.extend_from_slice(drafts);
        let result = self.verify_window(gpu, target, position, &block, &proposal, max_emit);
        self.settle(gpu, target, result)
    }

    fn on_forced_advance(
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
        // Forced tokens must land in the feature rows too, or the next draft
        // conditions on a short prefix.
        self.check_host_rows(start_pos)?;
        if let SpecAdvance::Ready { .. } = target.spec_advance(
            gpu,
            tokens,
            start_pos,
            false,
            abort,
            Some(&mut self.target_hidden_host),
        )? {
            self.rebase_host(start_pos + tokens.len())?;
            self.host_tokens.extend_from_slice(tokens);
            self.check_host_rows(start_pos + tokens.len())?;
        }
        Ok(true)
    }

    fn supports_ar_step(&self) -> bool {
        self.retire.is_some()
    }

    /// A one-row window: the target's verify of `[seed]` (its recorded
    /// one-row graph after the first use) with the feature capture, the
    /// greedy or sampled pick of [`Self::step_one_token`], and the committed
    /// row appended to the draft context, which therefore stays whole.
    fn ar_step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
    ) -> Result<u32, String> {
        if self.retire.is_none() {
            return Err("Dflash2Speculator: this target has no AR step".into());
        }
        self.check_host_rows(position)?;
        if self.request.penalties_active() {
            self.history.begin_window(emitted);
        }
        let result = self.step_one_token(gpu, target, position, seed);
        self.settle(gpu, target, result).map(|step| step.next_seed)
    }

    fn supports_sampled_ar_step(&self) -> bool {
        true
    }

    fn supports_penalties(&self) -> bool {
        true
    }

    fn retire_break_even(&self) -> Option<crate::spec_retire::BreakEven> {
        self.retire
    }

    fn reset(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        self.clear_host();
        // The target was reset with the drafter: drop a stale verify ticket.
        if let Some(vs) = self.verify_scratch.as_mut() {
            vs.clear();
        }
        Ok(())
    }

    fn repair_terminal_prefix(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        window_start: usize,
        window_seed: u32,
        consumed: &[u32],
    ) -> Result<bool, String> {
        let replay = terminal_prefix_replay(window_seed, consumed);
        let end = window_start + replay.len();
        // The replay re-runs committed rows only, so their feature rows stand.
        if replay.is_empty()
            || window_start < self.host_base
            || self.host_tokens.get(window_start..end) != Some(replay.as_slice())
        {
            return Ok(false);
        }
        let vs = self.verify_scratch()?;
        if !target.repair_committed_prefix(gpu, vs.as_mut(), window_start, &replay)? {
            return Ok(false);
        }
        let row_elems = self.row_elems();
        self.target_hidden_host
            .truncate((end - self.host_base) * row_elems);
        self.host_tokens.truncate(end);
        // Device copies of rows at or past `end` are stale now.
        self.scratch.reset_upload_tracking();
        Ok(true)
    }

    fn reset_state_evidence(&self) -> Option<crate::spec::SpecResetEvidence> {
        let th = &self.scratch.thlog;
        Some(crate::spec::SpecResetEvidence {
            drafter_reset: self.target_hidden_host.is_empty()
                && th.uploaded_rows() == 0
                && th.proj_cached_rows() == 0
                && th.full_cached_rows() == 0,
            checkpoint_empty: true,
        })
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn can_resume_at(&self, prompt: &[u32], start: usize) -> bool {
        draft_resume_base(
            &self.host_tokens,
            self.host_base,
            self.feature_window,
            prompt,
            start,
        )
        .is_some()
    }

    fn supports_temp_verify(&self) -> bool {
        true
    }

    fn supports_chain_nucleus_verify(&self) -> bool {
        true
    }

    fn requires_greedy(&self) -> bool {
        false
    }

    fn configure_request(&mut self, cfg: SpecRequestConfig) {
        self.rng = SpecRng::new(cfg.rng_seed);
        self.history = PenaltyHistory::new(cfg.penalty_window());
        self.penalty_cfg = SamplerConfig {
            repeat_penalty: cfg.repeat_penalty,
            repeat_window: cfg.repeat_window,
            presence_penalty: cfg.presence_penalty,
            frequency_penalty: cfg.frequency_penalty,
            ..SamplerConfig::greedy()
        };
        self.request = cfg;
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        let Dflash2Speculator {
            weights,
            scratch,
            verify_scratch,
            ..
        } = *self;
        weights.free_gpu(gpu);
        *LAST_GRAPH_STATS.lock().unwrap() = Some(scratch.draft_graph_stats());
        scratch.free_gpu(gpu);
        if let Some(vs) = verify_scratch {
            vs.free(gpu);
        }
    }
}

impl Dflash2Speculator {
    /// Verify `block` (`[seed, drafts…]`), accept, commit the kept prefix
    /// and append its feature rows.
    fn verify_window(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        block: &[u32],
        proposal: &crate::dflash::DflashCandidateProposal,
        max_emit: usize,
    ) -> Result<SpecStep, String> {
        let drafts = &block[1..];
        let mut rows = Vec::with_capacity(block.len() * self.row_elems());
        let mut vs = self.verify_scratch.take().ok_or("verify scratch freed")?;
        let sampled = self.sampled();
        let penalized = self.request.penalties_active();
        let verdict = if sampled || penalized {
            // Penalized rows come back as full logits for greedy too: the
            // device argmax cannot see the penalties. Neutral requests keep
            // their routes (sampled: logits; greedy: the device argmax).
            let vocab = self.config.vocab_size;
            target
                .verify_block_logits(gpu, block, position, vs.as_mut(), Some(&mut rows))
                .and_then(|mut logits| {
                    if penalized {
                        penalize_verify_rows(
                            &mut logits,
                            vocab,
                            drafts,
                            &mut self.history,
                            &self.penalty_cfg,
                        )?;
                    }
                    if sampled {
                        self.accept_sampled(proposal, drafts, &logits)
                    } else {
                        accept_greedy_rows(drafts, &logits, vocab)
                    }
                })
        } else {
            target
                .verify_block(gpu, block, position, vs.as_mut(), Some(&mut rows))
                .map(|picks| accept_greedy_prefix(drafts, &picks, None))
        };
        self.verify_scratch = Some(vs);
        let verdict = verdict?;
        let accepted = verdict.accepted;
        let bonus = *verdict
            .committed
            .last()
            .ok_or("Dflash2Speculator: empty verdict")?;
        target.commit_prefix(
            gpu,
            block,
            accepted,
            position,
            self.verify_scratch()?.as_mut(),
        )?;
        self.append_rows(&rows, &block[..=accepted])?;
        let emit = drafts[..accepted]
            .iter()
            .copied()
            .chain(std::iter::once(bonus));
        Ok(SpecStep::new(emit, bonus, drafts.len(), accepted).cap_emit(max_emit))
    }

    /// Sampled acceptance of one window against the verify's `logits`
    /// (`(drafts + 1) × vocab`): the selector's sparse `q` per draft row.
    fn accept_sampled(
        &mut self,
        proposal: &crate::dflash::DflashCandidateProposal,
        drafts: &[u32],
        logits: &[f32],
    ) -> Result<crate::spec::GreedyAccept, String> {
        let vocab = self.config.vocab_size;
        if logits.len() != (drafts.len() + 1) * vocab {
            return Err(format!(
                "Dflash2Speculator: verify returned {} logits for {} rows of {vocab}",
                logits.len(),
                drafts.len() + 1
            ));
        }
        let q = selector_q(proposal, drafts.len())?;
        let spec = self.sample_spec();
        let mut scratch = Vec::new();
        accept_sampled_prefix(
            drafts,
            &q,
            None,
            &mut self.rng,
            &mut SparseDist::default(),
            |row, out| {
                out.build_from_logits(&logits[row * vocab..(row + 1) * vocab], spec, &mut scratch)
            },
        )
    }
}

/// Penalize every row of a verify window's `logits` (`(drafts + 1) × vocab`)
/// in place with AR's policy arithmetic ([`apply_logit_policy_cpu`]): row `r`
/// is the target's row after `drafts[..r]`, so it sees the window's base
/// history (`prompt ‖ emitted` or `emitted`, seed last) plus those drafts,
/// trailing `repeat_window` tokens. `history` must be at its window's start
/// ([`PenaltyHistory::begin_window`]) and is left there.
fn penalize_verify_rows(
    logits: &mut [f32],
    vocab: usize,
    drafts: &[u32],
    history: &mut PenaltyHistory,
    cfg: &SamplerConfig,
) -> Result<(), String> {
    if logits.len() != (drafts.len() + 1) * vocab {
        return Err(format!(
            "Dflash2Speculator: verify returned {} logits for {} rows of {vocab}",
            logits.len(),
            drafts.len() + 1
        ));
    }
    for (row, logit_row) in logits.chunks_exact_mut(vocab).enumerate() {
        apply_logit_policy_cpu(logit_row, history.row(row), cfg);
        if let Some(&draft) = drafts.get(row) {
            history.push_draft(draft);
        }
    }
    history.rewind_drafts();
    Ok(())
}

/// Greedy acceptance of one window against its target `logits` rows
/// (`(drafts + 1) × vocab`, penalized or not): the per-row argmax, the rule
/// AR's `sample_cpu` applies at temperature 0, then the shared accept-prefix
/// rule.
fn accept_greedy_rows(
    drafts: &[u32],
    logits: &[f32],
    vocab: usize,
) -> Result<crate::spec::GreedyAccept, String> {
    if logits.len() != (drafts.len() + 1) * vocab {
        return Err(format!(
            "Dflash2Speculator: verify returned {} logits for {} rows of {vocab}",
            logits.len(),
            drafts.len() + 1
        ));
    }
    let picks: Vec<u32> = logits.chunks_exact(vocab).map(argmax).collect();
    Ok(accept_greedy_prefix(drafts, &picks, None))
}

/// The selector's per-row draft distributions of the first `rows` rows: row
/// `r` puts its softmax probability on each of its top-K candidates
/// (`draft_indices` of the reference `_rejection_sample`).
fn selector_q(
    proposal: &crate::dflash::DflashCandidateProposal,
    rows: usize,
) -> Result<Vec<SparseDist>, String> {
    let k = proposal.top_k;
    let probs = proposal
        .probabilities
        .as_ref()
        .ok_or("Dflash2Speculator: sampled selector returned no q")?;
    if proposal.candidates.len() < rows * k || probs.len() < rows * k {
        return Err("Dflash2Speculator: selector q shape mismatch".into());
    }
    Ok((0..rows)
        .map(|r| {
            let mut q = SparseDist::default();
            q.set_entries(
                proposal.candidates[r * k..(r + 1) * k]
                    .iter()
                    .copied()
                    .zip(probs[r * k..(r + 1) * k].iter().copied()),
            );
            q
        })
        .collect())
}

/// Raw feature rows the draft projects per `fc` GEMM through its staging
/// buffer ([`DflashScratch::new_windowed_staged`]): every window's new rows
/// (`<= DFLASH_BLOCK_MAX + 1`) in one GEMM, a cold prompt's window in
/// `window / 128` of them. 128 Raon-1-mini feature rows (51200 F32) are 26 MB
/// where the raw window ring was 0.42 GB.
const FEATURE_STAGE_ROWS: usize = 128;

/// Selector-confidence product below which a window stops verifying drafts.
/// HIPF-135, Flash-Next on gfx1151 (verify ~36 ms + ~6 ms per row): over 6122
/// block-8 windows, products 0.3-0.5 maximize the simulated tok/s (+18% over
/// the best fixed depth).
const VERIFY_CONFIDENCE: f32 = 0.5;

/// Drafts a window verifies: the longest prefix whose running product of
/// selector confidences stays at or above [`VERIFY_CONFIDENCE`], at least one.
/// `confidence[r]` must not depend on row `r`'s own draw
/// ([`crate::dflash::selector_row_confidence`]): the cut is then a stopping
/// rule on the earlier drafts, which keeps sampled acceptance exact.
fn verify_depth(confidence: &[f32]) -> usize {
    let mut product = 1.0;
    let confident = confidence
        .iter()
        .take_while(|&&c| {
            product *= c;
            product >= VERIFY_CONFIDENCE
        })
        .count();
    confident.max(1).min(confidence.len())
}

/// The draft scratch [`build_dflash2_speculator`] runs on: an all-sliding
/// draft was trained over its declared window, so it reads its context
/// through the projected rows only and stages the raw feature rows from the
/// host ([`DflashScratch::new_windowed_staged`]) instead of keeping a
/// window-high raw ring on the device; a split draft keeps the raw window
/// (its full layer reads it); an unwindowed draft keeps every row. Running
/// the draft over any other span than its declared window is a
/// train/inference mask mismatch.
pub fn new_draft_scratch(
    gpu: &mut Gpu,
    config: &DflashConfig,
    block: usize,
    ctx_capacity: usize,
    has_mq: bool,
) -> Result<DflashScratch, String> {
    match config.declared_window {
        Some(w) if config.all_layers_sliding => DflashScratch::new_windowed_staged(
            gpu,
            config,
            block,
            w,
            ctx_capacity,
            has_mq,
            FEATURE_STAGE_ROWS,
        ),
        Some(w) => DflashScratch::new_windowed(gpu, config, block, w, w, ctx_capacity, has_mq),
        None => DflashScratch::new_with_mq(gpu, config, block, ctx_capacity, has_mq),
    }
    .map_err(|e| format!("{e}"))
}

/// Build a DFlash2 speculator for `target` from a converted draft HFQ
/// (`dflash_convert`, F16 or MQ). `block` is the runtime proposal width
/// (seed included), `2..=DFLASH_BLOCK_MAX`; the caller bounds it by the
/// target's exact verify rows. Refuses a draft without the candidate
/// selector (legacy DFlash).
pub fn build_dflash2_speculator(
    gpu: &mut Gpu,
    draft_hfq_path: &str,
    target: &mut dyn SpecTarget,
    ctx_capacity: usize,
    block: usize,
) -> Result<Box<dyn Speculator>, String> {
    let draft_hfq = HfqFile::open(Path::new(draft_hfq_path)).map_err(|e| format!("{e}"))?;
    let config = DflashConfig::from_hfq(&draft_hfq)
        .ok_or_else(|| "DFlash2 draft: missing or malformed `dflash` metadata".to_string())?;
    if config.selector_rank.is_none() || config.selector_top_k.is_none() {
        return Err(format!(
            "draft '{draft_hfq_path}' is a legacy DFlash draft (no candidate selector); \
             this target runs DFlash2 drafts only"
        ));
    }
    if !(2..=DFLASH_BLOCK_MAX).contains(&block) {
        return Err(format!(
            "DFlash2 block {block} is outside 2..={DFLASH_BLOCK_MAX}"
        ));
    }
    let target_streams = target.dflash_streams_per_layer();
    if target_streams != config.target_streams_per_layer {
        return Err(format!(
            "DFlash2 draft '{draft_hfq_path}' expects target_streams_per_layer {} but the target \
             captures {target_streams} stream(s) per extract layer",
            config.target_streams_per_layer
        ));
    }
    let weights = DflashWeights::load(gpu, &draft_hfq, &config).map_err(|e| format!("{e}"))?;
    if !weights.has_candidate_selector() {
        weights.free_gpu(gpu);
        return Err("DFlash2 draft: candidate selector tensors are missing".into());
    }
    let scratch = new_draft_scratch(gpu, &config, block, ctx_capacity, weights.has_mq);
    let mut scratch = match scratch {
        Ok(scratch) => scratch,
        Err(e) => {
            weights.free_gpu(gpu);
            return Err(e);
        }
    };
    match DRAFT_GRAPH_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => scratch.set_draft_graph(false),
        2 => scratch.set_draft_graph(true),
        _ => {}
    }
    // An all-sliding draft reads only its last `W` context rows: the host and
    // the target keep just those (a split draft's full layer reads them all).
    let feature_window = config.declared_window.filter(|_| config.all_layers_sliding);
    target.set_dflash_extract_layers(config.target_layer_ids.clone());
    if let Some(window) = feature_window {
        target.set_dflash_feature_window(window);
    }
    let retire = target.dflash_retire_break_even(gpu, weights.has_mq);
    let verify_scratch = match target.new_spec_scratch(gpu, block) {
        Ok(vs) => vs,
        Err(e) => {
            target.set_dflash_extract_layers(Vec::new());
            weights.free_gpu(gpu);
            scratch.free_gpu(gpu);
            return Err(e);
        }
    };
    Ok(Box::new(Dflash2Speculator {
        weights,
        scratch,
        config,
        target_hidden_host: Vec::new(),
        host_base: 0,
        host_tokens: Vec::new(),
        feature_window,
        verify_scratch: Some(verify_scratch),
        block_size: block,
        ctx_capacity,
        mask_row: Vec::new(),
        request: SpecRequestConfig::default(),
        rng: SpecRng::new(0),
        history: PenaltyHistory::default(),
        penalty_cfg: SamplerConfig::greedy(),
        retire,
    }))
}

/// Where the draft context restarts when the target continues `prompt` at
/// `start` (a session restore or a live continuation). The host holds rows
/// `host_base..host_tokens.len()`, committed for `host_tokens`. The draft's
/// next forward reads rows `prompt.len() - window..` (`window` `None`: every
/// row) and needs all of them held; rows below `start` can only come from
/// the host, and only when committed for this prompt's prefix.
/// `Some(host_base)` keeps the host rows, `Some(start)` restarts empty at
/// `start` (the forward reads nothing below it) and `None` cannot resume.
fn draft_resume_base(
    host_tokens: &[u32],
    host_base: usize,
    window: Option<usize>,
    prompt: &[u32],
    start: usize,
) -> Option<usize> {
    let first_read = window.map_or(0, |w| prompt.len().saturating_sub(w));
    if start <= first_read {
        return Some(start);
    }
    let kept = host_base <= first_read
        && host_tokens
            .get(..start)
            .is_some_and(|held| prompt.get(..start) == Some(held));
    kept.then_some(host_base)
}

/// Keep the last `keep` rows of `rows_elems`-wide feature rows once the
/// buffer holds more than `2 * keep` rows (amortized one copy per `keep`
/// appended rows). Returns the rows dropped from the front.
pub fn trim_feature_rows(rows: &mut Vec<f32>, row_elems: usize, keep: usize) -> usize {
    let held = rows.len() / row_elems;
    if keep == 0 || held <= 2 * keep {
        return 0;
    }
    let drop = held - keep;
    rows.drain(..drop * row_elems);
    drop
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dflash::DflashCandidateProposal;

    /// The reference `_rejection_sample` with `draft_indices` (z-lab
    /// `dflash/model.py`): a draft `x ~ q` is accepted iff `u * q(x) < p(x)`,
    /// and a rejection draws from `(p - q)+` renormalized, `q` being the
    /// selector's mass on its candidates only. Then the first emitted token is
    /// distributed as `p`, the acceptance rate is `Σ min(p, q)`, and a
    /// rejection never lands where `q >= p`.
    #[test]
    fn selector_sparse_q_rejection_matches_the_reference_law() {
        // Candidates 4 and 5 are outside p's support; token 0 is outside q's.
        let proposal = DflashCandidateProposal {
            tokens: vec![1],
            candidates: vec![1, 2, 4, 5],
            probabilities: Some(vec![0.5, 0.2, 0.2, 0.1]),
            selected_probabilities: None,
            confidence: vec![0.5],
            top_k: 4,
        };
        let q = selector_q(&proposal, 1).unwrap();
        let p = [(0u32, 0.4f32), (1, 0.3), (2, 0.2), (3, 0.1)];
        let (mut target, mut rng) = (SparseDist::default(), SpecRng::new(7));
        let n = 400_000u64;
        let (mut first, mut accepted) = ([0u64; 8], 0u64);
        for _ in 0..n {
            let x = q[0].sample(rng.next_f32());
            let verdict =
                accept_sampled_prefix(&[x], &q, None, &mut rng, &mut target, |row, out| {
                    match row {
                        0 => out.set_entries(p),
                        _ => out.set_point_mass(7),
                    }
                    Ok(())
                })
                .unwrap();
            first[verdict.committed[0] as usize] += 1;
            if verdict.accepted == 1 {
                accepted += 1;
                assert_eq!(
                    verdict.committed,
                    vec![x, 7],
                    "full accept takes the bonus row"
                );
            } else {
                // (p - q)+ = {0: 0.4, 3: 0.1}: tokens 1 and 2 have q >= p.
                assert!(matches!(verdict.committed[0], 0 | 3), "{verdict:?}");
            }
        }
        for (token, pt) in p {
            let freq = first[token as usize] as f64 / n as f64;
            assert!(
                (freq - pt as f64).abs() < 0.004,
                "token {token}: {freq} vs {pt}"
            );
        }
        assert_eq!(
            first[4] + first[5],
            0,
            "q-only candidates are never emitted"
        );
        let rate = accepted as f64 / n as f64;
        assert!(
            (rate - 0.5).abs() < 0.004,
            "acceptance {rate} vs Σ min(p, q) = 0.5"
        );
    }

    #[test]
    fn selector_q_keeps_each_rows_candidates() {
        let proposal = DflashCandidateProposal {
            tokens: vec![3, 9],
            candidates: vec![3, 4, 9, 8],
            probabilities: Some(vec![0.75, 0.25, 0.6, 0.4]),
            selected_probabilities: None,
            confidence: vec![0.75, 0.6],
            top_k: 2,
        };
        let q = selector_q(&proposal, 2).unwrap();
        assert_eq!(
            (q[0].prob(3), q[0].prob(4), q[0].prob(9)),
            (0.75, 0.25, 0.0)
        );
        assert_eq!((q[1].prob(9), q[1].prob(8), q[1].prob(3)), (0.6, 0.4, 0.0));
        assert_eq!(
            selector_q(&proposal, 1).unwrap().len(),
            1,
            "a verified prefix"
        );
        assert!(selector_q(&proposal, 3).is_err(), "shape is checked");
    }

    /// The host keeps at most `2 * keep` rows, and after a trim exactly the
    /// last `keep`, in order: the draft's window always survives.
    #[test]
    fn trimmed_feature_rows_keep_the_last_window_in_order() {
        let row = 3;
        let mut rows: Vec<f32> = Vec::new();
        let mut base = 0usize;
        for position in 0..100u32 {
            rows.extend((0..row).map(|i| (position * 10 + i as u32) as f32));
            base += trim_feature_rows(&mut rows, row, 8);
            let held = rows.len() / row;
            assert!(held <= 16, "position {position}: {held} rows");
            assert_eq!(base + held, position as usize + 1);
            // Row r of the buffer is position base + r.
            assert_eq!(rows[0], (base as u32 * 10) as f32);
            assert!(held >= 8.min(position as usize + 1));
        }
        assert_eq!(
            trim_feature_rows(&mut rows, row, 0),
            0,
            "keep 0 never trims"
        );
    }

    /// A session reuse keeps the host rows only when they were committed for
    /// this prompt and hold every row the draft's next forward reads; a
    /// suffix of at least one window needs no host rows. Anything else must
    /// prefill cold: restarting empty at `start` with a shorter suffix is
    /// the draft forward's `base <= ctx - window` assert (HIPF-188).
    #[test]
    fn draft_resume_keeps_this_prompts_rows_or_needs_a_window_suffix() {
        // 20-token prompt, window 4: the first draft forward reads rows 16..20.
        let prompt: Vec<u32> = (0..20).collect();
        let w = Some(4);
        let held = &prompt[..19];
        assert_eq!(draft_resume_base(held, 14, w, &prompt, 18), Some(14));
        assert_eq!(
            draft_resume_base(held, 16, w, &prompt, 19),
            Some(16),
            "live continuation"
        );
        let mut foreign = held.to_vec();
        foreign[3] = 99;
        assert_eq!(
            draft_resume_base(&foreign, 14, w, &prompt, 18),
            None,
            "rows of another prompt"
        );
        assert_eq!(
            draft_resume_base(held, 17, w, &prompt, 18),
            None,
            "rows 17.. miss row 16"
        );
        assert_eq!(
            draft_resume_base(held, 19, w, &prompt, 18),
            None,
            "start below a trimmed base"
        );
        assert_eq!(
            draft_resume_base(&[], 0, w, &prompt, 18),
            None,
            "rows cleared by a reset"
        );
        assert_eq!(
            draft_resume_base(&foreign, 17, w, &prompt, 16),
            Some(16),
            "a window-long suffix restarts empty"
        );
        assert_eq!(draft_resume_base(&foreign, 17, w, &prompt, 0), Some(0));
        // A draft that reads every row keeps rows from 0 or prefills cold.
        assert_eq!(draft_resume_base(held, 0, None, &prompt, 18), Some(0));
        assert_eq!(draft_resume_base(&[], 0, None, &prompt, 18), None);
        assert_eq!(draft_resume_base(&[], 0, None, &prompt, 0), Some(0));
    }

    #[test]
    fn verify_depth_keeps_the_confident_prefix() {
        // Running products 0.9, 0.72, 0.504 | 0.25: three drafts.
        assert_eq!(verify_depth(&[0.9, 0.8, 0.7, 0.5, 1.0]), 3);
        // A product exactly at the threshold is kept.
        assert_eq!(verify_depth(&[1.0, VERIFY_CONFIDENCE, 0.99]), 2);
        // An unconfident first draft is still verified, a later rise never revives the chain.
        assert_eq!(verify_depth(&[0.1, 1.0, 1.0]), 1);
        assert_eq!(verify_depth(&[1.0; 7]), 7);
    }

    /// Sampled target law across the verify-depth cut. Two-row chain: `x1 ~
    /// q1`, `x2 ~ q2(.|x1)` (the selector's predecessor dependence), depth by
    /// [`verify_depth`] on the row confidences, acceptance by
    /// [`accept_sampled_prefix`] against `p1` / `p2(.|t1)`; a window that ends
    /// after one token continues with an exact draw from `p2(.|t1)`. The
    /// emitted pair must follow `p1(t1) p2(t2|t1)`. With the production
    /// confidence (top-candidate mass, [`selector_row_confidence`]) it does;
    /// with the drawn token's own mass the second row is verified more often
    /// when its draw is the likely one, and `P(t2 = c | t1 = a)` rises from 0.3
    /// to 0.42 (`0.6 * 0.5 + 0.4 * 0.3`).
    #[test]
    fn sampled_verify_depth_keeps_the_target_law() {
        use crate::dflash::selector_row_confidence;
        let (a, b, c, d) = (1u32, 2u32, 3u32, 4u32);
        let dist = |e: &[(u32, f32)]| {
            let mut s = SparseDist::default();
            s.set_entries(e.iter().copied());
            s
        };
        let q1 = [(a, 0.9f32), (b, 0.1)];
        let q2 = |x: u32| {
            if x == a {
                [(c, 0.6f32), (d, 0.4)]
            } else {
                [(c, 0.5), (d, 0.5)]
            }
        };
        let p1 = [(a, 0.5f32), (b, 0.5)];
        let p2 = |x: u32| {
            if x == a {
                [(c, 0.3f32), (d, 0.7)]
            } else {
                [(c, 0.5), (d, 0.5)]
            }
        };
        let scores = |q: &[(u32, f32)]| q.iter().map(|&(_, p)| p.ln()).collect::<Vec<f32>>();
        let run = |drawn_mass: bool| {
            let mut rng = SpecRng::new(11);
            let mut target = SparseDist::default();
            let mut pairs = std::collections::HashMap::<(u32, u32), u64>::new();
            let n = 400_000u64;
            for _ in 0..n {
                let x1 = dist(&q1).sample(rng.next_f32());
                let x2 = dist(&q2(x1)).sample(rng.next_f32());
                let confidence = if drawn_mass {
                    vec![dist(&q1).prob(x1), dist(&q2(x1)).prob(x2)]
                } else {
                    vec![
                        selector_row_confidence(&scores(&q1)),
                        selector_row_confidence(&scores(&q2(x1))),
                    ]
                };
                let depth = verify_depth(&confidence);
                let drafts = &[x1, x2][..depth];
                let q: Vec<SparseDist> = [dist(&q1), dist(&q2(x1))][..depth].to_vec();
                let verdict =
                    accept_sampled_prefix(drafts, &q, None, &mut rng, &mut target, |row, out| {
                        match row {
                            0 => out.set_entries(p1),
                            _ => out.set_entries(p2(drafts[0])),
                        }
                        Ok(())
                    })
                    .unwrap();
                let t1 = verdict.committed[0];
                let t2 = match verdict.committed.get(1) {
                    Some(&t) => t,
                    None => dist(&p2(t1)).sample(rng.next_f32()),
                };
                *pairs.entry((t1, t2)).or_default() += 1;
            }
            let freq =
                |t1: u32, t2: u32| pairs.get(&(t1, t2)).copied().unwrap_or(0) as f64 / n as f64;
            freq(a, c) / (freq(a, c) + freq(a, d))
        };
        // P(t1 = a) = 0.5: the conditional is estimated on ~200k pairs.
        let exact = run(false);
        assert!(
            (exact - 0.3).abs() < 0.005,
            "P(c | a) = {exact}, target 0.3"
        );
        let biased = run(true);
        assert!(
            (biased - 0.42).abs() < 0.005,
            "the drawn-mass cut must show its bias: P(c | a) = {biased}"
        );
    }

    fn penalty_cfg(window: usize) -> SamplerConfig {
        SamplerConfig {
            repeat_penalty: 1.1,
            repeat_window: window,
            presence_penalty: 1.5,
            frequency_penalty: 0.25,
            ..SamplerConfig::greedy()
        }
    }

    /// The shared predicate and the generated-only switch: a neutral request
    /// has no history window, a penalized one keeps `repeat_window`, and the
    /// default penalizes the prompt too (existing MTP users are unchanged).
    #[test]
    fn penalty_request_defaults_keep_the_prompt_in_the_history() {
        let neutral = SpecRequestConfig::default();
        assert!(!neutral.penalty_generated_only);
        assert_eq!(neutral.penalty_window(), 0);
        let penalized = SpecRequestConfig {
            presence_penalty: 1.5,
            repeat_window: 64,
            ..SpecRequestConfig::default()
        };
        assert!(penalized.penalties_active());
        assert_eq!(penalized.penalty_window(), 64);
        assert!(!penalized.penalty_generated_only);
        // A window of 0 disables every penalty.
        assert!(!SpecRequestConfig {
            presence_penalty: 1.5,
            ..SpecRequestConfig::default()
        }
        .penalties_active());
    }

    /// Greedy verify of a penalized window: each row's pick is the argmax of
    /// the row `sample_cpu` would penalize against `suffix_W(base ‖
    /// drafts[..r])`, a row whose raw argmax was already emitted flips to the
    /// runner-up, and the accept rule keeps the draft prefix equal to those
    /// picks.
    #[test]
    fn penalized_greedy_verify_flips_emitted_argmax_like_sample_cpu() {
        use crate::sampler::sample_cpu;
        let vocab = 6usize;
        let seed = 2u32;
        let base = [4u32, 4, seed]; // emitted so far, pending seed last
        let raw: [[f32; 6]; 4] = [
            [0.1, 0.2, 3.0, 2.6, 0.0, 0.3], // raw argmax 2, the pending seed
            [0.0, 0.1, 0.2, 3.0, 2.9, 0.1], // raw argmax 3, row 0's draft
            [0.0, 0.0, 0.1, 0.2, 3.0, 2.9], // raw argmax 4, emitted twice
            [0.5, 2.0, 0.1, 0.2, 0.0, 0.3],
        ];
        for window in [2usize, 3, 8, 64] {
            let cfg = penalty_cfg(window);
            let mut logits: Vec<f32> = raw.iter().flatten().copied().collect();
            let drafts = [3u32, 4, 5];
            let mut history = PenaltyHistory::new(window);
            history.set_prompt(&[]);
            history.begin_window(&base);
            penalize_verify_rows(&mut logits, vocab, &drafts, &mut history, &cfg).unwrap();
            assert_eq!(
                history.tokens(),
                &base[base.len().saturating_sub(window)..],
                "drafts rewound to the window's base"
            );
            let mut reference = Vec::new();
            let mut picks = Vec::new();
            for (r, row) in raw.iter().enumerate() {
                let mut hist = base.to_vec();
                hist.extend_from_slice(&drafts[..r.min(drafts.len())]);
                let mut row = row.to_vec();
                picks.push(sample_cpu(&mut row, &hist, &cfg));
                reference.push(row);
            }
            for (r, row) in logits.chunks_exact(vocab).enumerate() {
                assert_eq!(row, &reference[r][..], "window {window} row {r}");
            }
            let verdict = accept_greedy_rows(&drafts, &logits, vocab).unwrap();
            // The expected window is rebuilt from `sample_cpu`'s own picks.
            assert_eq!(
                verdict.committed,
                {
                    let mut kept = Vec::new();
                    for (i, &d) in drafts.iter().enumerate() {
                        if picks[i] == d {
                            kept.push(d);
                        } else {
                            break;
                        }
                    }
                    kept.push(picks[kept.len()]);
                    kept
                },
                "window {window}"
            );
        }
        // With the full history the first row's raw argmax (the seed) is
        // penalized to the runner-up, which then matches a draft of it.
        let cfg = penalty_cfg(64);
        let mut logits: Vec<f32> = raw[..2].iter().flatten().copied().collect();
        let mut history = PenaltyHistory::new(64);
        history.set_prompt(&[]);
        history.begin_window(&base);
        penalize_verify_rows(&mut logits, vocab, &[3], &mut history, &cfg).unwrap();
        let verdict = accept_greedy_rows(&[3], &logits, vocab).unwrap();
        assert_eq!(verdict.accepted, 1, "the runner-up draft is accepted");
        // The raw argmax would have rejected it.
        let raw_flat: Vec<f32> = raw[..2].iter().flatten().copied().collect();
        assert_eq!(
            accept_greedy_rows(&[3], &raw_flat, vocab).unwrap().accepted,
            0
        );
        // A wrong-length buffer is refused.
        assert!(penalize_verify_rows(&mut logits[..5], vocab, &[3], &mut history, &cfg).is_err());
        assert!(accept_greedy_rows(&[3], &raw_flat[..5], vocab).is_err());
    }

    /// Sampled verify under presence + frequency + repeat penalties: the
    /// first two emitted tokens follow `p0(t1) · p1(t2 | t1)` with `p` the
    /// softmax of each row penalized against `emitted ‖ earlier drafts`; row 1
    /// is penalized against the window's earlier accepted draft. Leaving the
    /// drafts out of the history (the bug this guards) visibly breaks it.
    #[test]
    fn sampled_verify_follows_the_penalized_target_law() {
        let vocab = 4usize;
        let seed = 3u32;
        let cfg = SamplerConfig {
            repeat_penalty: 1.2,
            repeat_window: 6,
            presence_penalty: 1.5,
            frequency_penalty: 0.5,
            ..SamplerConfig::greedy()
        };
        let raw0 = [0.4f32, 0.1, 0.0, 1.2];
        let raw1 = |prev: u32| -> [f32; 4] {
            let mut r = [0.2f32, 0.5, 0.3, 0.1];
            r[prev as usize] += 1.5;
            r
        };
        // The target's raw row after `[.., seed, drafts[..r]]`.
        let raw_rows = |drafts: &[u32]| -> Vec<f32> {
            let mut rows = raw0.to_vec();
            for r in 1..=drafts.len() {
                rows.extend_from_slice(&raw1(drafts[r - 1]));
            }
            rows
        };
        let softmax = |row: &[f32]| -> Vec<f64> {
            let max = row.iter().copied().fold(f32::MIN, f32::max) as f64;
            let e: Vec<f64> = row.iter().map(|&v| (v as f64 - max).exp()).collect();
            let z: f64 = e.iter().sum();
            e.iter().map(|v| v / z).collect()
        };
        let penalized = |row: &[f32], history: &[u32]| -> Vec<f32> {
            let mut row = row.to_vec();
            apply_logit_policy_cpu(&mut row, history, &cfg);
            row
        };
        // Exact law of two emitted tokens after the pending seed.
        let p0 = softmax(&penalized(&raw0, &[seed]));
        let expected = |t1: u32, t2: u32| -> f64 {
            p0[t1 as usize] * softmax(&penalized(&raw1(t1), &[seed, t1]))[t2 as usize]
        };
        let dist = |e: &[(u32, f32)]| {
            let mut s = SparseDist::default();
            s.set_entries(e.iter().copied());
            s
        };
        let q1 = [(0u32, 0.4f32), (1, 0.3), (2, 0.2), (3, 0.1)];
        let q2 = |x: u32| -> [(u32, f32); 4] {
            let mut q = [(0u32, 0.25f32); 4];
            for (i, e) in q.iter_mut().enumerate() {
                *e = (i as u32, if i as u32 == (x + 1) % 4 { 0.55 } else { 0.15 });
            }
            q
        };
        let spec = SampleSpec::cpu_ar(1.0, 1.0, None, 0.0);
        let max_deviation = |with_drafts: bool| -> f64 {
            let mut rng = SpecRng::new(23);
            let mut history = PenaltyHistory::new(cfg.repeat_window);
            history.set_prompt(&[]);
            let (mut target, mut scratch) = (SparseDist::default(), Vec::new());
            let mut pairs = std::collections::HashMap::<(u32, u32), u64>::new();
            let n = 300_000u64;
            for _ in 0..n {
                let x1 = dist(&q1).sample(rng.next_f32());
                let x2 = dist(&q2(x1)).sample(rng.next_f32());
                // Depth by the earlier drafts only: a seed-equal first draft
                // is verified alone.
                let depth = if x1 == seed { 1 } else { 2 };
                let drafts = &[x1, x2][..depth];
                let q: Vec<SparseDist> = [dist(&q1), dist(&q2(x1))][..depth].to_vec();
                let mut logits = raw_rows(drafts);
                history.begin_window(&[seed]);
                if with_drafts {
                    penalize_verify_rows(&mut logits, vocab, drafts, &mut history, &cfg).unwrap();
                } else {
                    // The bug: every row sees the base history only.
                    for row in logits.chunks_exact_mut(vocab) {
                        apply_logit_policy_cpu(row, history.row(0), &cfg);
                    }
                }
                let verdict =
                    accept_sampled_prefix(drafts, &q, None, &mut rng, &mut target, |row, out| {
                        out.build_from_logits(
                            &logits[row * vocab..(row + 1) * vocab],
                            spec,
                            &mut scratch,
                        )
                    })
                    .unwrap();
                let t1 = verdict.committed[0];
                let t2 = match verdict.committed.get(1) {
                    Some(&t) => t,
                    None => {
                        // A one-token window continues as an AR step over
                        // `emitted = [seed, t1]`.
                        history.begin_window(&[seed, t1]);
                        let mut row = raw1(t1).to_vec();
                        apply_logit_policy_cpu(&mut row, history.row(0), &cfg);
                        let mut p = SparseDist::default();
                        p.build_from_logits(&row, spec, &mut scratch).unwrap();
                        p.sample(rng.next_f32())
                    }
                };
                *pairs.entry((t1, t2)).or_default() += 1;
            }
            let mut worst = 0.0f64;
            for t1 in 0..vocab as u32 {
                for t2 in 0..vocab as u32 {
                    let freq = pairs.get(&(t1, t2)).copied().unwrap_or(0) as f64 / n as f64;
                    worst = worst.max((freq - expected(t1, t2)).abs());
                }
            }
            worst
        };
        let exact = max_deviation(true);
        assert!(exact < 0.004, "penalized pair law deviates by {exact}");
        let without = max_deviation(false);
        assert!(
            without > 0.01,
            "a base-only history must show its bias, deviation {without}"
        );
    }
}
