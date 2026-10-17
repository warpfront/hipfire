// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 speculative decode on the engine seam: the arch-22 EAGLE drafter
//! as an [`MtpDrafter`] (driven by the generic `MtpSpeculator`) and
//! [`SpecTarget`] for [`Gemma4Bundle`].
//!
//! A window at `position` holds the pending seed (emitted, not yet in the
//! KV) and the target's post-norm hidden of `position - 1`, the hidden that
//! predicted it. The drafter drafts `k` tokens at the constant query
//! position `position`; one batched verify writes `[seed, drafts..]` at
//! `position..` and returns per-row argmax (greedy) or per-row logits
//! (sampled or penalized requests, logprobs) and post-norm hidden; the accept
//! rule commits the accepted drafts plus the bonus. Gemma 4 KV is
//! position-indexed, so rejected rows need no rollback: the next verify
//! overwrites them.
//!
//! Sampled and penalized requests verify by SpecInfer naive sampling: drafts
//! stay the draft head's argmax, verify row `i` draws one token with the AR
//! producer's sampler (`sampler::sample_cpu` and its process-wide RNG,
//! penalized against `suffix_W(prompt ‖ emitted ‖ drafts[..i])`), and a
//! draft is accepted iff it equals its row's draw. Every emitted token is one
//! AR draw in AR's order, so the output follows the AR distribution.

use crate::bundle::Gemma4Bundle;
use crate::drafter::{
    draft_step, target_embedding_table, Gemma4DrafterConfig, Gemma4DrafterScratch,
    Gemma4DrafterWeights, DRAFTER_ARCH_ID,
};
use crate::forward::{decode_step, forward_batch_spec};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{argmax, reset_cpu_sampler_rng, sample_top_k_p};
use hipfire_runtime::sampler::{sample_cpu, SamplerConfig};
use hipfire_runtime::session_cache::SessionRoute;
use hipfire_runtime::spec::{
    accept_greedy_prefix, MtpDrafter, MtpWindow, PrefillOutcome, SpecAdvance, SpecGrammar,
    SpecRequestConfig, SpecScratch, SpecTarget,
};
use hipfire_runtime::spec_sampling::{accept_naive_prefix, naive_target_sampler, PenaltyHistory};
use rdna_compute::{DType, Gpu, GpuTensor};

fn gemma4(target: &mut dyn SpecTarget) -> Result<&mut Gemma4Bundle, String> {
    target
        .as_any_mut()
        .downcast_mut::<Gemma4Bundle>()
        .ok_or_else(|| "gemma4 drafter: target is not a Gemma 4 bundle".to_string())
}

/// The target policy of a sampled or penalized request: the sampler the
/// Gemma 4 AR producer (`generate_gemma4`) draws every token with, and the
/// penalty history it sees there (prompt ‖ generated).
// ponytail: naive acceptance (argmax drafts); Leviathan with the draft head's
// distribution as `q` would accept more drafts at temp > 0.
struct TargetPolicy {
    sampler: SamplerConfig,
    history: PenaltyHistory,
    /// The request's prompt is in `history` (its first prefill sets it).
    prompt_set: bool,
    /// Penalized copy of the row being drawn.
    scratch: Vec<f32>,
}

impl TargetPolicy {
    fn new(cfg: &SpecRequestConfig) -> Self {
        // The AR producer's per-request seeding of the shared sampler RNG.
        reset_cpu_sampler_rng(cfg.rng_seed as u32);
        Self {
            sampler: naive_target_sampler(cfg),
            history: PenaltyHistory::new(cfg.penalty_window()),
            prompt_set: false,
            scratch: Vec::new(),
        }
    }

    /// One AR draw from `row` penalized by `history`.
    fn draw(sampler: &SamplerConfig, scratch: &mut Vec<f32>, row: &[f32], history: &[u32]) -> u32 {
        if history.is_empty() {
            // No penalty stage: `sample_cpu` would not touch the row.
            return sample_top_k_p(
                row,
                sampler.temperature,
                sampler.top_p,
                sampler.top_k,
                sampler.min_p,
            );
        }
        scratch.clear();
        scratch.extend_from_slice(row);
        sample_cpu(scratch, history, sampler)
    }

    /// The prefill's first token. `prompt` is the full rendered prompt (on a
    /// realign it also replays the emitted tokens, which AR's history holds).
    fn draw_prefill(&mut self, row: &[f32], prompt: &[u32]) -> u32 {
        if !self.prompt_set {
            self.history.set_prompt(prompt);
            self.prompt_set = true;
        }
        let tail = &prompt[prompt.len().saturating_sub(self.history.window())..];
        Self::draw(&self.sampler, &mut self.scratch, row, tail)
    }

    /// Verify row `index`'s token, penalized by history row `index`.
    fn draw_row(&mut self, row: &[f32], index: usize) -> u32 {
        Self::draw(
            &self.sampler,
            &mut self.scratch,
            row,
            self.history.row(index),
        )
    }
}

/// The arch-22 EAGLE drafter riding a Gemma 4 target.
pub struct Gemma4Drafter {
    config: Gemma4DrafterConfig,
    weights: Gemma4DrafterWeights,
    scratch: Gemma4DrafterScratch,
    /// Target post-norm hidden of the position before the pending seed.
    pending_hidden: GpuTensor,
    /// Per-row post-norm hidden of a verify block, `[(k + 1) · dim]`.
    verify_hidden: GpuTensor,
    k: usize,
    ctx_capacity: usize,
    vocab: usize,
    /// The request's target policy when it samples (`temp > 1e-6`) or
    /// penalizes; `None` verifies by greedy argmax.
    policy: Option<TargetPolicy>,
    /// The request asked for logprobs: `rows` backs `token_logits`.
    logprobs: bool,
    /// Host target logits `[n · vocab]` behind the last emission: the prefill
    /// row, a `k = 0` decode row, or a window's verify rows.
    rows: Vec<f32>,
}

impl Gemma4Drafter {
    /// Load the drafter at `path` for `target`, drafting `draft_len` tokens
    /// per window (the loader admits 1..=5).
    pub fn load(
        path: &str,
        draft_len: usize,
        target: &Gemma4Bundle,
        gpu: &mut Gpu,
    ) -> Result<Self, String> {
        let hfq = HfqFile::open(std::path::Path::new(path))
            .map_err(|e| format!("open gemma4 drafter: {e}"))?;
        if hfq.arch_id != DRAFTER_ARCH_ID {
            return Err(format!(
                "gemma4 EAGLE drafter must be arch_id={DRAFTER_ARCH_ID} (gemma4_unified_assistant); got arch_id={} — a DFlash draft goes in params.draft on qwen3.5 targets, not params.drafter",
                hfq.arch_id
            ));
        }
        let config = Gemma4DrafterConfig::from_hfq(&hfq)?;
        let dim = target.config.dim;
        if config.backbone_hidden != dim {
            return Err(format!(
                "drafter backbone_hidden ({}) != target hidden ({dim}) — this drafter was trained against a different target width",
                config.backbone_hidden
            ));
        }
        if target_embedding_table(&target.weights).is_none() {
            return Err(format!(
                "gemma4 drafter: target embedding format {:?} has no batched lookup",
                target.weights.embd_format
            ));
        }
        let weights = Gemma4DrafterWeights::load(&hfq, &config, gpu)?;
        let scratch = match Gemma4DrafterScratch::new(gpu, &config, &weights) {
            Ok(scratch) => scratch,
            Err(e) => {
                weights.free_gpu(gpu);
                return Err(e);
            }
        };
        let hidden = gpu.zeros(&[dim], DType::F32).and_then(|pending| {
            match gpu.zeros(&[(draft_len + 1) * dim], DType::F32) {
                Ok(verify) => Ok((pending, verify)),
                Err(e) => {
                    let _ = gpu.free_tensor(pending);
                    Err(e)
                }
            }
        });
        let (pending_hidden, verify_hidden) = match hidden {
            Ok(v) => v,
            Err(e) => {
                scratch.free_gpu(gpu);
                weights.free_gpu(gpu);
                return Err(format!("gemma4 drafter: hidden buffers: {e:?}"));
            }
        };
        Ok(Self {
            config,
            weights,
            scratch,
            pending_hidden,
            verify_hidden,
            k: draft_len,
            ctx_capacity: target.state.max_seq,
            vocab: target.config.vocab_size,
            policy: None,
            logprobs: false,
            rows: Vec::new(),
        })
    }

    pub fn config(&self) -> &Gemma4DrafterConfig {
        &self.config
    }

    /// The target's post-norm hidden of its last processed position becomes
    /// the pending seed's conditioning hidden.
    fn take_target_hidden(&self, gpu: &mut Gpu, bundle: &Gemma4Bundle) -> Result<(), String> {
        let bytes = bundle.config.dim * 4;
        gpu.hip
            .memcpy_dtod_at(&self.pending_hidden.buf, 0, &bundle.state.tmp.buf, 0, bytes)
            .map_err(|e| format!("gemma4 drafter: capture hidden: {e:?}"))
    }

    /// Keep a single-row emission's logits for `token_logits`.
    fn keep_row(&mut self, logits: Vec<f32>) {
        if self.logprobs {
            self.rows = logits;
        }
    }
}

impl MtpDrafter for Gemma4Drafter {
    fn mtp_prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        prompt_tokens: &[u32],
        _fill_tokens: &[u32],
        start_pos: usize,
        _cache_hit: bool,
        _abort: &dyn Fn() -> bool,
    ) -> Result<PrefillOutcome, String> {
        let bundle = gemma4(target)?;
        let logits = bundle.prefill(gpu, prompt_tokens, start_pos, SessionRoute::Mtp)?;
        self.take_target_hidden(gpu, bundle)?;
        let first_token = match self.policy.as_mut() {
            None => argmax(&logits),
            Some(policy) => policy.draw_prefill(&logits, prompt_tokens),
        };
        self.keep_row(logits);
        Ok(PrefillOutcome::Ready { first_token })
    }

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
        let bundle = gemma4(target)?;
        if bundle.state.n_tokens != position {
            return Err(format!(
                "gemma4 drafter: window at {position} but the target is at {}",
                bundle.state.n_tokens
            ));
        }
        let dim = bundle.config.dim;
        if let Some(policy) = self.policy.as_mut() {
            policy.history.begin_window(emitted);
        }
        if k == 0 {
            // The seed alone: one ordinary decode step.
            let logits = decode_step(
                &bundle.config,
                &bundle.weights,
                &mut bundle.state,
                gpu,
                seed,
                position as u32,
            )?;
            self.take_target_hidden(gpu, bundle)?;
            let token = match self.policy.as_mut() {
                None => argmax(&logits),
                Some(policy) => policy.draw_row(&logits, 0),
            };
            self.keep_row(logits);
            return Ok(MtpWindow {
                committed: vec![token],
                accepted: 0,
                drafts_generated: 0,
            });
        }
        let k = k.min(self.k);
        self.scratch
            .begin_round(gpu, &self.pending_hidden, position)?;
        let mut block = Vec::with_capacity(k + 1);
        block.push(seed);
        for _ in 0..k {
            let prev = *block.last().expect("seeded block");
            block.push(draft_step(
                gpu,
                &self.weights,
                &self.config,
                &mut self.scratch,
                &bundle.weights,
                &bundle.state,
                &bundle.config,
                prev,
                position,
            )?);
        }
        let hidden = self.verify_hidden.sub_offset(0, block.len() * dim);
        let mut picks = Vec::with_capacity(block.len());
        let greedy = self.policy.is_none();
        let want_rows = !greedy || self.logprobs;
        forward_batch_spec(
            &bundle.config,
            &bundle.weights,
            &mut bundle.state,
            gpu,
            &block,
            position,
            Some(&hidden),
            greedy.then_some(&mut picks),
            want_rows.then_some(&mut self.rows),
        )?;
        let accept = match self.policy.as_mut() {
            None => accept_greedy_prefix(&block[1..], &picks, Some(eos)),
            Some(policy) => {
                for &draft in &block[1..] {
                    policy.history.push_draft(draft);
                }
                let (vocab, rows) = (self.vocab, &self.rows);
                accept_naive_prefix(&block[1..], Some(eos), |row| {
                    Ok(policy.draw_row(&rows[row * vocab..(row + 1) * vocab], row))
                })?
            }
        };
        let committed = accept.committed.len();
        // The seed and every committed token but the last are in the KV; the
        // last stays pending, conditioned on the hidden of the row before it.
        bundle.state.n_tokens = position + committed;
        let row = hidden.sub_offset((committed - 1) * dim, dim);
        gpu.hip
            .memcpy_dtod_at(&self.pending_hidden.buf, 0, &row.buf, 0, dim * 4)
            .map_err(|e| format!("gemma4 drafter: capture verify hidden: {e:?}"))?;
        Ok(MtpWindow {
            committed: accept.committed,
            accepted: accept.accepted,
            drafts_generated: k,
        })
    }

    fn mtp_forced_advance(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        tokens: &[u32],
        start_pos: usize,
        _abort: &dyn Fn() -> bool,
    ) -> Result<bool, String> {
        let bundle = gemma4(target)?;
        if !tokens.is_empty() {
            bundle.forward_rows(gpu, tokens, start_pos)?;
            self.take_target_hidden(gpu, bundle)?;
        }
        Ok(true)
    }

    fn mtp_reset(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        // Stateless between windows: prefill restages the conditioning hidden.
        Ok(())
    }

    fn mtp_free(self: Box<Self>, gpu: &mut Gpu) {
        let d = *self;
        let _ = gpu.free_tensor(d.pending_hidden);
        let _ = gpu.free_tensor(d.verify_hidden);
        d.scratch.free_gpu(gpu);
        d.weights.free_gpu(gpu);
    }

    fn k(&self) -> usize {
        self.k
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn requires_greedy(&self) -> bool {
        false
    }

    fn configure_request(&mut self, cfg: SpecRequestConfig) {
        self.logprobs = cfg.logprobs;
        self.policy = (cfg.temp > 1e-6 || cfg.penalties_active()).then(|| TargetPolicy::new(&cfg));
    }

    fn token_logits(&self, index: usize) -> Option<&[f32]> {
        if !self.logprobs {
            return None;
        }
        self.rows.get(index * self.vocab..(index + 1) * self.vocab)
    }

    fn name(&self) -> &'static str {
        "eagle"
    }
}

/// Gemma 4 verify needs no scratch of its own: KV is position-indexed and
/// nothing is snapshotted before a verify.
struct NoScratch;

impl SpecScratch for NoScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn free(self: Box<Self>, _gpu: &mut Gpu) {}
}

impl SpecTarget for Gemma4Bundle {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        hipfire_runtime::arch_model::ArchModel::reset_session_state(self, gpu)
    }

    fn new_spec_scratch(
        &mut self,
        _gpu: &mut Gpu,
        _block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        Ok(Box::new(NoScratch))
    }

    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        _abort: &dyn Fn() -> bool,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        if reset {
            self.state.reset();
        }
        if self.state.n_tokens != start_pos {
            return Err(format!(
                "gemma4 spec_advance from {start_pos} but the state is at {}",
                self.state.n_tokens
            ));
        }
        let logits = self.forward_rows(gpu, tokens, start_pos)?;
        Ok(SpecAdvance::Ready {
            last_argmax: argmax(&logits),
            last_logits: Some(logits),
        })
    }

    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        _scratch: &mut dyn SpecScratch,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        if let [token] = block {
            let logits = decode_step(
                &self.config,
                &self.weights,
                &mut self.state,
                gpu,
                *token,
                position as u32,
            )?;
            return Ok(vec![argmax(&logits)]);
        }
        let mut picks = Vec::with_capacity(block.len());
        forward_batch_spec(
            &self.config,
            &self.weights,
            &mut self.state,
            gpu,
            block,
            position,
            None,
            Some(&mut picks),
            None,
        )?;
        Ok(picks)
    }

    fn commit_prefix(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        accept_len: usize,
        position: usize,
        _scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        // Rows past the committed prefix are overwritten by the next verify.
        self.state.n_tokens = position + accept_len + 1;
        Ok(())
    }

    fn eos_token(&self) -> u32 {
        self.eos_tok
    }

    fn ctx_capacity(&self) -> usize {
        self.state.max_seq
    }
}
