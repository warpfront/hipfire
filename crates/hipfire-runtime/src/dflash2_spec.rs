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
//!   product of selector confidences stays at or above
//!   [`VERIFY_CONFIDENCE`] (at least one draft); the draft forward still
//!   proposes the whole block.
//!
//! The target is reached only through [`SpecTarget`]: it captures the feature
//! rows (`dflash_extract_layers`), embeds the noise block and applies its head
//! to the draft rows on the device ([`SpecTarget::lm_head_logits_device`]).
//! Chain only: there is no tree arm (the selector is the proposal).

use crate::dflash::{draft_forward_based, propose_candidates_device, DflashConfig, DflashScratch};
use crate::dflash::{DflashWeights, DFLASH_BLOCK_MAX};
use crate::hfq::HfqFile;
use crate::spec::{
    accept_greedy_prefix, terminal_prefix_replay, PrefillOutcome, SpecAdvance, SpecGrammar,
    SpecRequestConfig, SpecScratch, SpecStep, SpecTarget, Speculator,
};
use crate::spec_sampling::{accept_sampled_prefix, SampleSpec, SparseDist, SpecRng};
use rdna_compute::Gpu;
use std::path::Path;

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

    /// Verify `[seed]` alone: one target token, no draft.
    fn step_one_token(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
    ) -> Result<SpecStep, String> {
        let block = [seed];
        let mut rows = Vec::with_capacity(self.row_elems());
        let token = if self.sampled() {
            let mut vs = self.verify_scratch.take().ok_or("verify scratch freed")?;
            let logits =
                target.verify_block_logits(gpu, &block, position, vs.as_mut(), Some(&mut rows));
            self.verify_scratch = Some(vs);
            self.sample_row(&logits?)?
        } else {
            let vs = self.verify_scratch()?;
            target.verify_block(gpu, &block, position, vs.as_mut(), Some(&mut rows))?[0]
        };
        target.commit_prefix(gpu, &block, 0, position, self.verify_scratch()?.as_mut())?;
        self.append_rows(&rows, &block)?;
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
        let first_token = if self.sampled() {
            let logits = last_logits.ok_or(
                "Dflash2Speculator: a sampled prefill needs the target's final-row logits",
            )?;
            self.sample_row(&logits)?
        } else {
            last_argmax
        };
        Ok(PrefillOutcome::Ready { first_token })
    }

    fn step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        _emitted: &[u32],
        _grammar: Option<&mut dyn SpecGrammar>,
        temp: f32,
        max_emit: usize,
    ) -> Result<SpecStep, String> {
        if max_emit == 0 {
            return Err("Dflash2Speculator: max_emit=0 (no remaining output budget)".into());
        }
        if self.request.penalties_active() {
            return Err("Dflash2Speculator: token penalties are not supported".into());
        }
        if (temp > 1e-6) != self.sampled() {
            return Err(format!(
                "Dflash2Speculator: step temperature {temp} differs from the request's {}",
                self.request.temp
            ));
        }
        self.check_host_rows(position)?;
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
        let verdict = if self.sampled() {
            target
                .verify_block_logits(gpu, block, position, vs.as_mut(), Some(&mut rows))
                .and_then(|logits| self.accept_sampled(proposal, drafts, &logits))
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

/// Selector-confidence product below which a window stops verifying drafts.
/// HIPF-135, Flash-Next on gfx1151 (verify ~36 ms + ~6 ms per row): over 6122
/// block-8 windows, products 0.3-0.5 maximize the simulated tok/s (+18% over
/// the best fixed depth).
const VERIFY_CONFIDENCE: f32 = 0.5;

/// Drafts a window verifies: the longest prefix whose running product of
/// selector confidences stays at or above [`VERIFY_CONFIDENCE`], at least one.
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
    // The draft was trained over its declared sliding window; running it
    // over any other span is a train/inference mask mismatch.
    let scratch = match config.declared_window {
        Some(w) => {
            DflashScratch::new_windowed(gpu, &config, block, w, w, ctx_capacity, weights.has_mq)
        }
        None => DflashScratch::new_with_mq(gpu, &config, block, ctx_capacity, weights.has_mq),
    };
    let scratch = match scratch {
        Ok(scratch) => scratch,
        Err(e) => {
            weights.free_gpu(gpu);
            return Err(format!("{e}"));
        }
    };
    // An all-sliding draft reads only its last `W` context rows: the host and
    // the target keep just those (a split draft's full layer reads them all).
    let feature_window = config.declared_window.filter(|_| config.all_layers_sliding);
    target.set_dflash_extract_layers(config.target_layer_ids.clone());
    if let Some(window) = feature_window {
        target.set_dflash_feature_window(window);
    }
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
}
