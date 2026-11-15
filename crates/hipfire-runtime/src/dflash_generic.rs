// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Target-generic chain-mode DFlash speculator.
//!
//! This is the arch-free twin of `hipfire_arch_qwen35::speculative::spec_step_dflash`
//! / `dflash_spec::DflashSpeculator`. It drives the SAME block-diffusion drafter
//! forward ([`crate::dflash`]) but verifies through the arch-generic
//! [`SpecTarget`] trait instead of a concrete qwen35 `ModelSlot`.
//!
//! Because the target is reached only through `SpecTarget`, the whole
//! DeltaNet-specific apparatus the qwen35 path carries — recurrent snapshot
//! ([`DeltaNetSnapshot`]), the GDN innovation tape ([`GdnTape`]), the hidden-state
//! ring buffer, and the post-verify rewind/replay — DISAPPEARS. For a stateless
//! dense-attention target (LLaMA / plain Qwen3) verify is one block-parallel
//! forward whose accepted-prefix KV is already correct; nothing to rewind.
//!
//! Mechanically the generic skeleton extracted from `spec_step_dflash` is:
//!   1. build the masked block `[seed, MASK, …, MASK]`,
//!   2. draft it in ONE [`draft_forward`] (broadcast the target's mask-token
//!      embedding as the noise input; positions per the qwen35 path),
//!   3. derive `drafts` by applying the TARGET lm_head to the draft hidden rows
//!      (`draft_scratch.x` rows `1..b`) and argmax-ing,
//!   4. verify `[seed, drafts…]` through [`SpecTarget::verify_block`], which
//!      returns the per-position target argmax AND the per-position hidden rows,
//!   5. greedy-accept the longest matching prefix + bonus, append ONLY the
//!      committed-prefix hidden to `target_hidden_host`, and advance.
//!
//! The chain verify supports BOTH greedy (temp≈0) and a distribution-EXACT
//! temp>0 path: SpecInfer NAIVE sampling (draw `x ~ softmax(target_logits/temp)`
//! per position, accept the drafted token it lands on, else emit the draw as the
//! bonus — see [`crate::spec_sampling::naive_sample_chain`]). So
//! [`supports_temp_verify`](Speculator::supports_temp_verify) is `true`.

use crate::dflash::{draft_forward, DflashConfig, DflashScratch, DflashWeights};
use crate::hfq::HfqFile;
use crate::llama;
use crate::spec::{
    accept_greedy_prefix, PrefillOutcome, SpecAdvance, SpecGrammar, SpecRequestConfig, SpecScratch,
    SpecStep, SpecTarget, Speculator,
};
use crate::spec_sampling::naive_sample_chain;
use rdna_compute::Gpu;
use std::path::Path;

/// Target-generic chain-mode DFlash speculator.
///
/// Owns the loaded draft weights/scratch/config, the cumulative target-hidden
/// host buffer (`[committed_rows × feature_width]` f32, row-major,
/// extract-layers ascending), and the arch-specific verify scratch the target
/// minted via [`SpecTarget::new_spec_scratch`]. The target itself is borrowed
/// per call — never owned — exactly like the qwen35 `DflashSpeculator`.
pub struct GenericDflashSpeculator {
    weights: DflashWeights,
    scratch: DflashScratch,
    config: DflashConfig,
    /// Cumulative committed target-hidden rows. Authoritative CPU shadow handed
    /// to `draft_forward` each cycle (the generic path always uses the host
    /// buffer — there is no D2D scatter fast path here, unlike qwen35's GPU-side
    /// hidden_rb). Grows by exactly `accept+1` rows per accepted cycle.
    target_hidden_host: Vec<f32>,
    /// Per-target verify scratch, minted by `SpecTarget::new_spec_scratch`.
    /// `Option` so `free` can move it out for explicit GPU release.
    verify_scratch: Option<Box<dyn SpecScratch>>,
    block_size: usize,
    ctx_capacity: usize,
    /// Xorshift64* RNG state for the temp>0 chain naive sampler. Bit-compatible
    /// with the qwen35 spec sampler's seed so the chain draws are deterministic.
    rng_state: u64,
    /// Per-request temperature from [`Speculator::set_sampling`]. Used only for
    /// the post-prefill first-token draw (`step` receives `temp` as an argument).
    /// Default 0 → historical greedy argmax.
    sample_temp: f32,
    /// Per-request nucleus/top-k truncation from [`Speculator::set_sampling`].
    /// Defaults `1.0` / `0` (disabled) so a speculator whose `set_sampling` is
    /// never called behaves exactly as today — byte-identical draws via the
    /// `top_k == 0 && top_p >= 0.999` fast path in [`naive_sample_chain`].
    sample_top_p: f32,
    sample_top_k: usize,
}

impl GenericDflashSpeculator {
    /// Hidden rows (`h` wide) per feature row: `num_extract × streams`, so
    /// `feature_rows() * hidden == config.feature_width()`.
    fn feature_rows(&self) -> usize {
        self.config.num_extract() * self.config.target_streams_per_layer
    }

    /// True one-token window: verify only `[seed]`, emit exactly one target
    /// token, commit prefix length 0 (full accept of a b=1 block). No draft
    /// forward, no post-commit `cap_emit` truncation of a longer commit.
    ///
    /// Greedy (`temp≈0`): argmax of the seed row. Sampled: one draw from
    /// `softmax(logits/temp)` via [`naive_sample_chain`] with empty drafts —
    /// distribution-exact, never an argmax substitute.
    fn step_one_token(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        temp: f32,
    ) -> Result<SpecStep, String> {
        let ne = self.feature_rows();
        let h = self.config.hidden;
        let vocab = self.config.vocab_size;
        let block = [seed];

        let ctx_elems = position * ne * h;
        assert_eq!(
            self.target_hidden_host.len(),
            ctx_elems,
            "target_hidden_host len {} != position {} * ne {} * h {}",
            self.target_hidden_host.len(),
            position,
            ne,
            h
        );

        let vs = self
            .verify_scratch
            .as_mut()
            .ok_or("GenericDflashSpeculator: verify scratch already freed")?;
        let mut block_hidden: Vec<f32> = Vec::with_capacity(ne * h);

        let token = if temp > 1e-6 {
            let logits = target.verify_block_logits(
                gpu,
                &block,
                position,
                vs.as_mut(),
                Some(&mut block_hidden),
            )?;
            debug_assert_eq!(logits.len(), vocab);
            // Empty drafts → single bonus draw at row 0 (distribution-exact).
            let (accepted, bonus) = naive_sample_chain(
                &logits,
                &[],
                vocab,
                temp,
                self.sample_top_p,
                self.sample_top_k,
                &mut self.rng_state,
            );
            debug_assert_eq!(
                accepted, 0,
                "empty-draft naive_sample_chain accepts nothing"
            );
            bonus
        } else {
            let pick =
                target.verify_block(gpu, &block, position, vs.as_mut(), Some(&mut block_hidden))?;
            debug_assert_eq!(pick.len(), 1);
            pick[0]
        };
        debug_assert_eq!(block_hidden.len(), ne * h);

        // b=1 full accept: accept_len == block.len()-1 == 0.
        target.commit_prefix(gpu, &block, 0, position, vs.as_mut())?;

        // H2: seed row only (accepted drafts = 0).
        let keep_elems = committed_block_hidden_elems(0, ne, h);
        self.target_hidden_host
            .extend_from_slice(&block_hidden[..keep_elems]);
        debug_assert_eq!(
            self.target_hidden_host.len(),
            committed_host_len(position, 0, ne, h),
            "host buffer length mismatch after one-token commit"
        );

        let step = SpecStep::new(std::iter::once(token), token, 0, 0);
        debug_assert_eq!(step.emit.len(), 1);
        // Defensive only — one-token path never overshoots.
        Ok(step.cap_emit(1))
    }

    /// Shared chain-verify tail: H2 hidden truncation + SpecStep construction.
    ///
    /// Called after both the greedy and temp>0 chain arms have computed `accepted`
    /// and `bonus` and committed the prefix via `commit_prefix`. Appends the first
    /// `accepted + 1` rows of `block_hidden` to `target_hidden_host` (H2: seed +
    /// accepted drafts; the bonus row is NOT appended — it materialises next cycle
    /// as the block seed) and constructs the `SpecStep` for the daemon.
    ///
    /// MUST be called with identical arguments from both arms: the call order
    /// (extend_from_slice → debug_assert_eq → SpecStep::new) is the same in both.
    fn finish_chain(
        &mut self,
        block_hidden: &[f32],
        drafts: &[u32],
        accepted: usize,
        bonus: u32,
        position: usize,
        ne: usize,
        h: usize,
        max_emit: usize,
    ) -> Result<SpecStep, String> {
        // Append ONLY the committed-prefix hidden — the first accepted+1 rows of
        // block_hidden (seed + accepted drafts). The bonus's hidden is NOT appended:
        // its proper hidden materialises on the NEXT cycle's verify when it is
        // forwarded as block[0]. This grows the prefix by accept+1, matching
        // draft_forward's contract.
        // draft_forward owns the uploaded_target_hidden_rows cursor (it delta-uploads
        // the appended host rows next step); the generic path never scatters to GPU
        // itself, so we must NOT set it here.
        let keep_elems = committed_block_hidden_elems(accepted, ne, h);
        self.target_hidden_host
            .extend_from_slice(&block_hidden[..keep_elems]);
        debug_assert_eq!(
            self.target_hidden_host.len(),
            committed_host_len(position, accepted, ne, h),
            "host buffer length mismatch after chain commit"
        );

        // ── Lower to SpecStep: emit = committed[1..] (accepted drafts + bonus,
        // seed dropped), next_seed = bonus. emit.len() == accepted + 1 drives the
        // daemon's position += emit.len(). Pre-commit budget guarantees
        // emit.len() <= max_emit; cap_emit is defensive only.
        if let Some(t) = self.scratch.online.as_mut() {
            let committed: Vec<u32> = drafts[..accepted].iter().copied().chain([bonus]).collect();
            t.observe(position, &committed, accepted);
        }
        let emit = drafts[..accepted]
            .iter()
            .copied()
            .chain(std::iter::once(bonus));
        let step = SpecStep::new(emit, bonus, drafts.len(), accepted);
        debug_assert!(
            step.emit.len() <= max_emit && !step.emit.is_empty(),
            "chain emit len {} vs max_emit {}",
            step.emit.len(),
            max_emit
        );
        Ok(step.cap_emit(max_emit))
    }
}

/// Pure (testable) truncation math for the partial-accept host-buffer update.
///
/// After a cycle that committed `committed_len` tokens (= `accepted + 2`:
/// `[seed, accepted drafts…, bonus]`), `draft_forward`'s incremental contract
/// grows the cached prefix by exactly `accepted + 1` rows. So the host buffer
/// must hold `(position + accepted + 1) × row_stride` floats, where
/// `row_stride = num_extract × hidden`. The verify produced `block_hidden` for
/// all `b+1 = drafts+1` positions; we keep the first `accepted + 1` rows and
/// discard the rejected tail.
///
/// Returns the number of f32 elements the host buffer should have after the
/// append. Factored out so the truncation invariant is unit-testable without a
/// GPU.
fn committed_host_len(
    position: usize,
    accepted: usize,
    num_extract: usize,
    hidden: usize,
) -> usize {
    (position + accepted + 1) * num_extract * hidden
}

/// Number of leading f32 elements of a verify `block_hidden` buffer that belong
/// to the committed prefix (the first `accepted + 1` rows). The rejected tail
/// (rows `accepted+1 ..= drafts`) is discarded.
fn committed_block_hidden_elems(accepted: usize, num_extract: usize, hidden: usize) -> usize {
    (accepted + 1) * num_extract * hidden
}

/// Clamp an accept/bonus pair so `emit_len = accepted + 1` ≤ `max_emit`.
///
/// When the unclamped accept already fits, returns it unchanged. When it must
/// shrink, `boundary_token` becomes the new bonus — callers pass:
/// - greedy chain: `target_pick[max_accept]` (argmax at the boundary row);
/// - sampled chain: `drafts[max_accept]` (the walk's matched draw at that row);
///
/// Never substitutes argmax for a sampled boundary draw.
fn clamp_accept_to_budget(
    accepted: usize,
    bonus: u32,
    max_emit: usize,
    boundary_token: u32,
) -> (usize, u32) {
    let max_accept = max_emit.saturating_sub(1);
    if accepted > max_accept {
        (max_accept, boundary_token)
    } else {
        (accepted, bonus)
    }
}

impl Speculator for GenericDflashSpeculator {
    fn name(&self) -> &'static str {
        "dflash"
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
        // Mirror the qwen35 prefill cache-hit/miss split, minus the DeltaNet /
        // hidden-ring machinery: on a miss we re-seed the whole prompt and reset
        // the draft's incremental-upload cursor; on a hit we advance only the
        // suffix from `prefill_start` and keep the cached prefix projections.
        let (fill_tokens, start_pos): (&[u32], usize) = if cache_hit {
            (prefill_tokens, prefill_start)
        } else {
            (prompt_tokens, 0)
        };

        if let Some(t) = self.scratch.online.as_mut() {
            t.seed_prompt(prompt_tokens);
        }
        if !cache_hit {
            // Cold start: drop both the cumulative host shadow and the draft's
            // upload/projection tracking so the first step re-uploads from row 0.
            self.target_hidden_host.clear();
            self.scratch.reset_upload_tracking();
        }

        // Advance the target AND capture its residual hidden into the cumulative
        // host buffer in one pass. `reset = !cache_hit` zeroes the target's KV
        // (and recurrent, for arches that have it) on a miss. Capture only fires
        // when the target's `dflash_extract_layers()` is `Some` (set at build).
        let adv = target.spec_advance(
            gpu,
            fill_tokens,
            start_pos,
            !cache_hit,
            abort,
            Some(&mut self.target_hidden_host),
        )?;
        // First emit = target draw at the final prompt position.
        // temp≈0 keeps historical `last_argmax` (byte-identical greedy).
        // temp>0 reuses the empty-draft `naive_sample_chain` pipeline that
        // `step_one_token` / later chain verify use — one RNG advance, no
        // second distribution path. Requires host last-row logits from the
        // target (LLaMA/generic path populates `SpecAdvance::Ready.last_logits`).
        let first_token = match adv {
            SpecAdvance::Aborted => return Ok(PrefillOutcome::Aborted),
            SpecAdvance::Ready {
                last_argmax,
                last_logits,
            } => {
                if self.sample_temp <= 1e-6 {
                    last_argmax
                } else {
                    let logits = last_logits.ok_or_else(|| {
                        "GenericDflashSpeculator: temp>0 prefill needs last_logits \
                         from SpecTarget::spec_advance (target only exposed GPU argmax)"
                            .to_string()
                    })?;
                    let vocab = self.config.vocab_size;
                    if logits.len() != vocab {
                        return Err(format!(
                            "GenericDflashSpeculator: last_logits len {} != vocab {}",
                            logits.len(),
                            vocab
                        ));
                    }
                    let (accepted, bonus) = naive_sample_chain(
                        &logits,
                        &[],
                        vocab,
                        self.sample_temp,
                        self.sample_top_p,
                        self.sample_top_k,
                        &mut self.rng_state,
                    );
                    debug_assert_eq!(
                        accepted, 0,
                        "empty-draft naive_sample_chain accepts nothing"
                    );
                    bonus
                }
            }
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
        // emit ≤ max_emit: a window of block size b yields up to b tokens
        // (accepted drafts + bonus ≤ b). Size b so full-accept emit cannot
        // exceed the remaining budget; max_emit==1 takes the true one-token
        // path (no draft, verify [seed] only).
        let cfg_b = self.config.block_size.max(2);
        if max_emit == 0 {
            return Err("GenericDflashSpeculator: max_emit=0 (no remaining output budget)".into());
        }
        if max_emit == 1 {
            return self.step_one_token(gpu, target, position, seed, temp);
        }
        // max_emit >= 2: b ∈ [2, min(cfg, max_emit)] so emit ≤ b ≤ max_emit
        // on full accept without post-commit truncation.
        let b = cfg_b.min(max_emit).max(2);
        assert!(b >= 2, "dflash block size must be >= 2");
        let h = self.config.hidden;
        let ne = self.feature_rows();

        // ── 1. Build the masked block: [seed, MASK, …, MASK] ────────────────
        let mut block: Vec<u32> = vec![self.config.mask_token_id; b];
        block[0] = seed;

        // ── 2. Build the block-diffusion draft INPUT ────────────────────────
        // noise_embedding mirrors the z-lab reference `embed_tokens([seed, MASK,
        // …, MASK])` (dflash.py:235-237): position 0 carries the COMMITTED seed
        // token's embedding (it anchors the bidirectional block — every drafted
        // slot attends to it), positions 1..b are the mask-token embedding.
        // Broadcasting MASK to slot 0 too (the old behavior) stripped the anchor
        // and collapsed the drafts into degenerate repetition → low acceptance.
        let mask_row = target.embed_row(gpu, self.config.mask_token_id)?;
        let seed_row = target.embed_row(gpu, seed)?;
        debug_assert_eq!(mask_row.len(), h, "embed_row length != hidden");
        let mut noise_embedding: Vec<f32> = Vec::with_capacity(b * h);
        noise_embedding.extend_from_slice(&seed_row);
        for _ in 1..b {
            noise_embedding.extend_from_slice(&mask_row);
        }

        // positions_q: absolute positions of the block slots [position .. position+b).
        // positions_k: context positions [0 .. position) then block [position .. position+b).
        // (Positions are contiguous.)
        let positions_q: Vec<i32> = (position as i32..(position + b) as i32).collect();
        let positions_k: Vec<i32> = (0i32..(position + b) as i32).collect();

        // ── 3. Draft forward over the cumulative target-hidden prefix ───────
        // The host buffer is authoritative: hand draft_forward rows [0..position).
        // Its incremental-upload fast path keys off scratch.thlog.uploaded_rows().
        let ctx_elems = position * ne * h;
        assert_eq!(
            self.target_hidden_host.len(),
            ctx_elems,
            "target_hidden_host len {} != position {} * ne {} * h {}",
            self.target_hidden_host.len(),
            position,
            ne,
            h
        );
        draft_forward(
            gpu,
            &self.weights,
            &self.config,
            Some(&noise_embedding),
            Some(&self.target_hidden_host[..ctx_elems]),
            &positions_q,
            &positions_k,
            b,
            position,
            &mut self.scratch,
        )
        .map_err(|e| format!("draft_forward: {e}"))?;

        // ── 3b. Draft → tokens: apply the TARGET lm_head to draft hidden rows ─
        // The draft's final-hidden rows live in draft_scratch.x; row 0 is the seed
        // slot, rows 1..b are the drafted positions (mirrors qwen35's
        // draft_scratch.x.sub_offset(h, batch*h)). We argmax the target lm_head
        // over those b-1 rows to get the drafted tokens.
        let batch = b - 1;
        let draft_hidden = self.scratch.x.sub_offset(h, batch * h);
        let draft_logits = target.lm_head_logits(gpu, &draft_hidden, batch)?;
        debug_assert_eq!(draft_logits.len(), batch * self.config.vocab_size);
        let vocab = self.config.vocab_size;
        let drafts: Vec<u32> = match self.scratch.online.as_mut() {
            // Developer online draft tuner: re-rank the host top-K. Drafts stay
            // deterministic, which both the greedy accept and the temp>0 naive
            // sampling verify (exact for any deterministic draft) require.
            Some(t) => {
                let (ids, vals) = crate::dflash_online::top_k_rows(&draft_logits, vocab, batch);
                let picks = t.propose(position, seed, &ids, &vals, batch);
                if t.mode == crate::dflash_online::Mode::Sweep {
                    // Always rejected: every cycle commits exactly one target token.
                    vec![vocab as u32 - 1; batch]
                } else {
                    picks
                }
            }
            None => (0..batch)
                .map(|i| llama::argmax(&draft_logits[i * vocab..(i + 1) * vocab]))
                .collect(),
        };
        for (i, &d) in drafts.iter().enumerate() {
            block[i + 1] = d;
        }

        // ── 4t. temp>0 chain naive-sampling verify (distribution-exact) ─────
        // When the request temperature is >0,
        // verify the chain by drawing x ~ softmax(target_logits/temp) per position
        // and accepting the longest prefix where draft[i] == x_i (SpecInfer NAIVE
        // sampling — distribution-EXACT at any temperature; see
        // `spec_sampling::naive_sample_chain`). We need the FULL per-position target
        // logits, so this arm uses `verify_block_logits` instead of `verify_block`
        // (which only returns the argmax). The same H2 truncation applies: append
        // only the first accepted+1 rows of the captured hidden.
        if temp > 1e-6 {
            let vs = self
                .verify_scratch
                .as_mut()
                .ok_or("GenericDflashSpeculator: verify scratch already freed")?;
            let mut block_hidden: Vec<f32> = Vec::with_capacity(b * ne * h);
            let logits_per_pos = target.verify_block_logits(
                gpu,
                &block,
                position,
                vs.as_mut(),
                Some(&mut block_hidden),
            )?;
            debug_assert_eq!(logits_per_pos.len(), b * vocab);
            debug_assert_eq!(block_hidden.len(), b * ne * h);

            // `drafts` (len b-1) are block[1..b]; logits_per_pos has b rows (one
            // bonus row past the last draft). naive_sample_chain draws the target
            // token per row, accepts the matching draft prefix, and returns the
            // bonus at the divergence position. All emitted tokens are genuine
            // target draws → distribution-exact.
            let (accepted, bonus) = naive_sample_chain(
                &logits_per_pos,
                &drafts,
                vocab,
                temp,
                self.sample_top_p,
                self.sample_top_k,
                &mut self.rng_state,
            );
            // Pre-commit budget: emit = accepted + 1 ≤ max_emit. With b ≤ max_emit
            // this is defensive; if it binds, keep the walk's own boundary draw
            // (drafts[max_accept] — matched, since accepted > max_accept), never argmax.
            let boundary = if accepted > max_emit.saturating_sub(1) {
                drafts[max_emit.saturating_sub(1)]
            } else {
                bonus
            };
            let (accepted, bonus) = clamp_accept_to_budget(accepted, bonus, max_emit, boundary);

            target.commit_prefix(gpu, &block, accepted, position, vs.as_mut())?;
            return self.finish_chain(
                &block_hidden,
                &drafts,
                accepted,
                bonus,
                position,
                ne,
                h,
                max_emit,
            );
        }

        // ── 4. Verify + accept + truncation (review finding H2) ─────────────
        // Verify [seed, drafts…] through the target. Returns the per-position
        // greedy argmax (length b) AND fills block_hidden with the per-position
        // residual hidden (b rows × ne × h). The target leaves its state advanced
        // by `b`; commit_prefix fixes it to the committed prefix afterward.
        let vs = self
            .verify_scratch
            .as_mut()
            .ok_or("GenericDflashSpeculator: verify scratch already freed")?;
        let mut block_hidden: Vec<f32> = Vec::with_capacity(b * ne * h);
        let target_pick =
            target.verify_block(gpu, &block, position, vs.as_mut(), Some(&mut block_hidden))?;
        debug_assert_eq!(target_pick.len(), b, "verify_block returned != b argmax");
        debug_assert_eq!(
            block_hidden.len(),
            b * ne * h,
            "verify_block hidden != b*ne*h"
        );

        // Greedy accept: drafts = block[1..b], target_pick is the verifier's argmax
        // after each of the b positions. eos=None — DFlash never early-stops on
        // EOS here (the daemon handles EOS downstream). committed = accepted prefix
        // + bonus (= target_pick[accepted]).
        let acc = accept_greedy_prefix(&drafts, &target_pick, None);
        let accepted = acc.accepted;
        let bonus = *acc.committed.last().expect("eos=None yields a bonus");
        // Pre-commit budget: emit = accepted + 1 ≤ max_emit. Greedy boundary
        // re-pick is argmax at the clamped row (temp-0 correct).
        let boundary = if accepted > max_emit.saturating_sub(1) {
            target_pick[max_emit.saturating_sub(1)]
        } else {
            bonus
        };
        let (accepted, bonus) = clamp_accept_to_budget(accepted, bonus, max_emit, boundary);

        // Fix the target state to the committed prefix [seed, accepted…, bonus].
        // For a stateless target this is a no-op; for a recurrent one it restores
        // the snapshot verify_block stashed in `vs`. accept_len passed is the
        // number of accepted DRAFTS (block[1..=accept_len] accepted).
        target.commit_prefix(gpu, &block, accepted, position, vs.as_mut())?;
        self.finish_chain(
            &block_hidden,
            &drafts,
            accepted,
            bonus,
            position,
            ne,
            h,
            max_emit,
        )
    }

    /// Forced tokens (think-budget force-close) must land in the drafter's
    /// cumulative `target_hidden_host`, not just the target's KV. The default
    /// trait hook returns `false`, so the daemon's plain `spec_advance(..., None)`
    /// would leave a hole: the next `draft_forward` asserts host len == position
    /// and would either panic or condition on a stale/short prefix.
    ///
    /// Same one-row ownership as qwen35 `DflashSpeculator::on_forced_advance`:
    /// advance the target WITH hidden extraction into our host buffer and return
    /// `true` so the caller does not double-advance. Fail closed before mutation
    /// when extract layers are missing (capture would be a silent no-op).
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
        let extract = target.dflash_extract_layers().unwrap_or(&[]);
        if extract.is_empty() {
            // No capture possible → would leave host misaligned. Refuse before
            // any target mutation so the daemon fail-closed path can roll back.
            return Err(
                "GenericDflashSpeculator: forced advance requires dflash extract \
                 layers for target-hidden ownership"
                    .into(),
            );
        }
        let ne = self.feature_rows();
        let h = self.config.hidden;
        let expected_len = self.target_hidden_host.len() + tokens.len() * ne * h;
        // Host is authoritative at `start_pos` rows going in (same invariant as step).
        let expected_prefix = start_pos * ne * h;
        if self.target_hidden_host.len() != expected_prefix {
            return Err(format!(
                "GenericDflashSpeculator: forced advance host len {} != start_pos {} * ne {} * h {}",
                self.target_hidden_host.len(),
                start_pos,
                ne,
                h
            ));
        }
        match target.spec_advance(
            gpu,
            tokens,
            start_pos,
            false,
            abort,
            Some(&mut self.target_hidden_host),
        )? {
            SpecAdvance::Aborted => {
                // Caller tears the request down; leaving a partial append is fine
                // because the drafter state is reset on the way out.
                Ok(true)
            }
            SpecAdvance::Ready { .. } => {
                if self.target_hidden_host.len() != expected_len {
                    return Err(format!(
                        "GenericDflashSpeculator: forced advance captured {} floats, expected {}",
                        self.target_hidden_host.len(),
                        expected_len
                    ));
                }
                Ok(true)
            }
        }
    }

    fn supports_ar_step(&self) -> bool {
        true
    }

    fn ar_step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        _emitted: &[u32],
    ) -> Result<u32, String> {
        self.step_one_token(gpu, target, position, seed, 0.0)
            .map(|step| step.next_seed)
    }

    fn reset(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        // Drafter-local reset: clear the cumulative host shadow + the draft's
        // upload/projection tracking. The target's KV/recurrent reset is the
        // daemon's job (it owns the bundle).
        self.target_hidden_host.clear();
        self.scratch.reset_upload_tracking();
        Ok(())
    }

    fn reset_state_evidence(&self) -> Option<crate::spec::SpecResetEvidence> {
        let th = &self.scratch.thlog;
        Some(crate::spec::SpecResetEvidence {
            drafter_reset: self.target_hidden_host.is_empty()
                && th.uploaded_rows() == 0
                && th.proj_cached_rows() == 0
                && th.full_cached_rows() == 0,
            // Generic path has no divergent-render checkpoint ring.
            checkpoint_empty: true,
        })
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn supports_temp_verify(&self) -> bool {
        // The chain path (SpecInfer naive sampling) is distribution-EXACT at
        // temp>0, so temp>0 always routes through spec.
        true
    }

    fn requires_greedy(&self) -> bool {
        // The chain honors temp>0 distribution-correctly.
        false
    }

    fn configure_request(&mut self, cfg: SpecRequestConfig) {
        // Store temp + truncation policy for the post-prefill first-token draw and
        // every subsequent chain verify / bonus draw. The chain verify now applies
        // the same temp -> top-k -> nucleus truncation as the AR sampler
        // (`sample_host_nucleus`), via `naive_sample_chain`'s `top_p`/`top_k`
        // arguments. Previously `top_p`/`top_k` were discarded here, so a caller
        // requesting top_p/top_k with generic DFlash on was silently sampling the
        // UNTRUNCATED softmax while believing truncation was applied.
        // Blast radius: generic-DFlash targets loaded via LlamaCarrier (arch ids
        // 0/1). Qwen35's own `DflashSpeculator` (`hipfire-arch-qwen35`) stores and
        // plumbs all three correctly and is not affected.
        // Re-seed RNG to the same fixed value qwen35 uses per request so sampled
        // runs are deterministic given seed; greedy does not consume it.
        // New SpecRequestConfig fields (min_p / rng_seed / ngram) are ignored.
        self.sample_temp = cfg.temp;
        self.sample_top_p = cfg.top_p;
        self.sample_top_k = cfg.top_k_cut();
        self.rng_state = crate::spec::request_rng_state(cfg.rng_seed);
        if let Some(t) = self.scratch.online.as_mut() {
            t.reset();
        }
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        let GenericDflashSpeculator {
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

/// Build a target-generic chain DFlash speculator from a converted draft HFQ.
///
/// The draft must be the F16 product of `dflash_convert` (`has_mq = false`), so
/// the scratch is built with [`DflashScratch::new`] (the `new_with_mq` path is
/// only for an MQ-quantized draft — review finding L3; qwen35 regressed exactly
/// this once at `dflash_spec.rs:91`).
///
/// The verify scratch is minted by the TARGET (`new_spec_scratch`) so no arch
/// type leaks here, and the target is told the draft's extract layers via
/// [`SpecTarget::set_dflash_extract_layers`] so hidden capture matches the
/// drafter's `fc` input layout.
pub fn build_generic_dflash_speculator(
    gpu: &mut Gpu,
    draft_hfq_path: &str,
    target: &mut dyn SpecTarget,
    ctx_capacity: usize,
) -> Result<Box<dyn Speculator>, String> {
    let draft_hfq = HfqFile::open(Path::new(draft_hfq_path)).map_err(|e| format!("{e}"))?;
    let config = DflashConfig::from_hfq(&draft_hfq)
        .ok_or_else(|| "draft: failed to parse DflashConfig from HFQ metadata".to_string())?;
    let target_streams = target.dflash_streams_per_layer();
    if target_streams != config.target_streams_per_layer {
        return Err(format!(
            "draft '{draft_hfq_path}' expects target_streams_per_layer {} but the target \
             captures {target_streams} stream(s) per extract layer",
            config.target_streams_per_layer
        ));
    }
    let weights = DflashWeights::load(gpu, &draft_hfq, &config).map_err(|e| format!("{e}"))?;
    let block_size = config.block_size;
    // L3: F16 drafts (dflash_convert) → has_mq=false → DflashScratch::new.
    // new_with_mq only for an MQ-quantized draft.
    // Transactional: `weights` and `scratch` own GPU memory with no `Drop`,
    // so each fallible step below frees what is already owned before
    // returning Err (same class as the `or_free!` chain in
    // `hipfire-arch-qwen35`'s `load_dflash_state`).
    let scratch = match if weights.has_mq {
        DflashScratch::new_with_mq(gpu, &config, block_size, ctx_capacity, true)
            .map_err(|e| format!("{e}"))
    } else {
        DflashScratch::new(gpu, &config, block_size, ctx_capacity).map_err(|e| format!("{e}"))
    } {
        Ok(s) => s,
        Err(e) => {
            weights.free_gpu(gpu);
            return Err(e);
        }
    };
    let _ = draft_hfq;

    // Tell the target which residual-hidden layers to capture (the drafter's
    // target_layer_ids), and mint the per-target verify scratch.
    target.set_dflash_extract_layers(config.target_layer_ids.clone());
    let verify_scratch = match target.new_spec_scratch(gpu, block_size) {
        Ok(v) => v,
        Err(e) => {
            weights.free_gpu(gpu);
            scratch.free_gpu(gpu);
            return Err(e);
        }
    };

    Ok(Box::new(GenericDflashSpeculator {
        weights,
        scratch,
        config,
        target_hidden_host: Vec::new(),
        verify_scratch: Some(verify_scratch),
        block_size,
        ctx_capacity,
        // Fixed deterministic seed (matches the qwen35 dflash_spec default).
        rng_state: 0x13579BDF,
        // Greedy until a request calls `set_sampling`. Truncation defaults
        // disabled (top_p=1.0, top_k=0) so behaviour is byte-identical until
        // the caller opts in — matches the `1.0, 0` callers pass in tests.
        sample_temp: 0.0,
        sample_top_p: 1.0,
        sample_top_k: 0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::SpecStep;
    use smallvec::SmallVec;

    // The SpecStep lowering the chain step produces: emit = accepted drafts +
    // bonus (seed dropped), next_seed = bonus = emit.last(). Mirrors
    // `spec.rs::emit_len_drives_advance_not_accepted` — pins the load-bearing
    // loop contract that the daemon advances `position` by emit.len(), not by
    // `accepted`.
    fn lower_chain(drafts: &[u32], accepted: usize, bonus: u32) -> SpecStep {
        let emit = drafts[..accepted]
            .iter()
            .copied()
            .chain(std::iter::once(bonus));
        SpecStep::new(emit, bonus, drafts.len(), accepted)
    }

    #[test]
    fn chain_lowering_emit_len_is_accepted_plus_one() {
        // 4 drafts, accepted 2, bonus 99 → emit = [d0, d1, 99] (len 3).
        let drafts = [10u32, 11, 12, 13];
        let step = lower_chain(&drafts, 2, 99);
        assert_eq!(step.emit.as_slice(), &[10, 11, 99]);
        assert_eq!(step.emit.len(), step.accepted + 1);
        assert_eq!(*step.emit.last().unwrap(), step.next_seed);
        assert_eq!(step.next_seed, 99);
        assert_eq!(step.proposed, 4);
        assert_eq!(step.accepted, 2);
    }

    #[test]
    fn chain_lowering_full_accept() {
        // all 3 drafts accepted, bonus 77 → emit = [d0,d1,d2,77] (len 4).
        let drafts = [10u32, 11, 12];
        let step = lower_chain(&drafts, 3, 77);
        assert_eq!(step.emit.as_slice(), &[10, 11, 12, 77]);
        assert_eq!(step.emit.len(), step.accepted + 1);
        assert_eq!(step.next_seed, 77);
    }

    #[test]
    fn chain_lowering_zero_accept() {
        // 0 accepted → emit = [bonus] only (still non-empty, forward progress).
        let drafts = [10u32, 11];
        let step = lower_chain(&drafts, 0, 42);
        assert_eq!(step.emit.as_slice(), &[42]);
        assert_eq!(step.emit.len(), 1);
        assert_eq!(step.next_seed, 42);
        assert_eq!(step.accepted, 0);
    }

    // The lowering is byte-equivalent to building a SmallVec by hand (guards the
    // SpecStep::new IntoIterator path against an accidental reorder).
    #[test]
    fn chain_lowering_matches_manual_smallvec() {
        let drafts = [5u32, 6, 7, 8];
        let step = lower_chain(&drafts, 1, 99);
        let manual: SmallVec<[u32; 8]> = SmallVec::from_slice(&[5, 99]);
        assert_eq!(step.emit, manual);
    }

    // Partial-accept truncation math (review finding H2): after a step that
    // accepted `accepted` drafts at absolute `position`, the cumulative host
    // buffer must hold (position + accepted + 1) committed rows × ne × h, and we
    // keep exactly (accepted + 1) rows of the verify's block_hidden.
    #[test]
    fn truncation_keeps_accept_plus_one_rows() {
        let ne = 5usize;
        let h = 4096usize;
        // Simulate appending the committed-prefix hidden onto a host buffer that
        // already holds `position` rows.
        for &(position, drafts, accepted) in
            &[(0usize, 16usize, 0usize), (10, 16, 7), (100, 16, 16)]
        {
            let row_stride = ne * h;
            // verify produced b = drafts+1 rows of hidden.
            let b = drafts + 1;
            let block_hidden = vec![1.0f32; b * row_stride];
            let keep = committed_block_hidden_elems(accepted, ne, h);
            assert_eq!(keep, (accepted + 1) * row_stride);
            // Host buffer pre-step holds `position` rows.
            let mut host = vec![0.0f32; position * row_stride];
            host.extend_from_slice(&block_hidden[..keep]);
            assert_eq!(host.len(), committed_host_len(position, accepted, ne, h));
            assert_eq!(host.len(), (position + accepted + 1) * row_stride);
        }
    }

    // Full-accept and zero-accept boundaries of the truncation math.
    #[test]
    fn truncation_boundaries() {
        let ne = 2usize;
        let h = 8usize;
        // zero accept → keep exactly 1 row (the seed position's hidden).
        assert_eq!(committed_block_hidden_elems(0, ne, h), ne * h);
        // full accept of 15 drafts → keep 16 rows.
        assert_eq!(committed_block_hidden_elems(15, ne, h), 16 * ne * h);
        // host length advances by accept+1 rows from position.
        assert_eq!(committed_host_len(50, 3, ne, h), (50 + 4) * ne * h);
    }

    // Pre-commit budget clamp: emit = accepted + 1 must fit max_emit, and the
    // boundary token must be the caller's supplied draw (never silently swapped).
    #[test]
    fn clamp_accept_preserves_fitting_window() {
        assert_eq!(clamp_accept_to_budget(3, 99, 4, 7), (3, 99));
        assert_eq!(clamp_accept_to_budget(0, 42, 1, 7), (0, 42));
        assert_eq!(clamp_accept_to_budget(5, 11, 6, 7), (5, 11));
    }

    #[test]
    fn clamp_accept_uses_boundary_token_not_old_bonus() {
        // accepted=4 → emit would be 5; max_emit=3 → keep 2 drafts + boundary.
        assert_eq!(clamp_accept_to_budget(4, 999, 3, 55), (2, 55));
        // max_emit=1 → zero drafts + boundary as the sole emit token.
        assert_eq!(clamp_accept_to_budget(3, 999, 1, 77), (0, 77));
    }

    #[test]
    fn precommit_budget_emit_never_exceeds_max_emit() {
        // Mirrors finish_chain lowering after clamp_accept_to_budget.
        let drafts = [10u32, 11, 12, 13, 14];
        for max_emit in 1usize..=6 {
            let raw_accepted = 4usize; // would emit 5 without clamp
            let raw_bonus = 99u32;
            let boundary = drafts[max_emit.saturating_sub(1).min(drafts.len() - 1)];
            let (accepted, bonus) =
                clamp_accept_to_budget(raw_accepted, raw_bonus, max_emit, boundary);
            let step = lower_chain(&drafts, accepted, bonus);
            assert!(
                !step.emit.is_empty() && step.emit.len() <= max_emit,
                "emit {:?} vs max_emit {}",
                step.emit.as_slice(),
                max_emit
            );
            assert_eq!(step.emit.len(), accepted + 1);
            assert_eq!(step.next_seed, bonus);
            // Defensive cap_emit is a no-op when pre-commit clamp held.
            let capped = step.clone().cap_emit(max_emit);
            assert_eq!(capped.emit, step.emit);
            assert_eq!(capped.accepted, step.accepted);
            assert_eq!(capped.next_seed, step.next_seed);
        }
    }

    #[test]
    fn sampled_boundary_keeps_matched_draft_not_argmax() {
        // Simulate temp>0 clamp: walk accepted past budget; boundary = drafts[max_accept]
        // (the matched draw), never a fresh argmax substitute.
        let drafts = [10u32, 20, 30, 40];
        let walk_accepted = 3usize;
        let walk_bonus = 999u32; // would-be full-accept bonus
        let max_emit = 2usize;
        let max_accept = max_emit - 1;
        let boundary = drafts[max_accept]; // matched draft at clamped row
        let (accepted, bonus) =
            clamp_accept_to_budget(walk_accepted, walk_bonus, max_emit, boundary);
        assert_eq!(accepted, 1);
        assert_eq!(bonus, 20); // drafts[1], not argmax, not walk_bonus
        let step = lower_chain(&drafts, accepted, bonus);
        assert_eq!(step.emit.as_slice(), &[10, 20]);
        assert_eq!(step.cap_emit(max_emit).emit.as_slice(), &[10, 20]);
    }

    #[test]
    fn one_token_emit_shape() {
        // max_emit==1 path: proposed=0, accepted=0, emit=[token], next_seed=token.
        let step = SpecStep::new(std::iter::once(42u32), 42, 0, 0);
        assert_eq!(step.emit.as_slice(), &[42]);
        assert_eq!(step.accepted, 0);
        assert_eq!(step.proposed, 0);
        assert_eq!(step.next_seed, 42);
        let capped = step.cap_emit(1);
        assert_eq!(capped.emit.as_slice(), &[42]);
        assert_eq!(capped.accepted, 0);
    }

    // ── Task 4: generic first-token sampling seams ─────────────────────────
    //
    // Prefill first token (Speculator::prefill after set_sampling) and
    // step_one_token share the same already-configured sampling semantics:
    //   sample_temp > 1e-6 → naive_sample_chain empty-draft bonus draw
    //   sample_temp ≤ 1e-6 → last_argmax / llama::argmax (byte-identical)
    // set_sampling reseeds rng_state to 0x13579BDF before the first draw.

    /// temp≈0 first token is the seed-row argmax — byte-identical to greedy
    /// `naive_sample_chain(..., temp=0, empty drafts)` and `llama::argmax`.
    /// Mirrors prefill `sample_temp <= 1e-6 → last_argmax` and step_one_token
    /// greedy branch.
    #[test]
    fn generic_first_token_temp_zero_is_argmax_byte_identical() {
        let vocab = 5usize;
        // Distinct logits; argmax must be index 3 (highest).
        let logits = [1.0f32, -2.0, 0.5, 4.25, 4.0];
        assert_eq!(logits.len(), vocab);

        // Prefill greedy keeps last_argmax; chain temp=0 is the same fold.
        let last_argmax = crate::llama::argmax(&logits);
        assert_eq!(last_argmax, 3);

        let mut rng = 0xDEAD_BEEF_u64;
        let (accepted, bonus) =
            crate::spec_sampling::naive_sample_chain(&logits, &[], vocab, 0.0, 1.0, 0, &mut rng);
        assert_eq!(accepted, 0, "empty-draft chain accepts nothing");
        assert_eq!(
            bonus, last_argmax,
            "temp=0 empty-draft bonus == last_argmax"
        );
        // temp=0 does not consume RNG (argmax fold only) — prefill greedy path
        // likewise leaves rng_state untouched.
        assert_eq!(rng, 0xDEAD_BEEF_u64);

        // Tie-break: first strict-max wins (matches GPU kernel `>` fold).
        let tied = [1.0f32, 5.0, 5.0, 0.0];
        assert_eq!(crate::llama::argmax(&tied), 1);
        let mut rng_tie = 1u64;
        let (_, tbonus) = crate::spec_sampling::naive_sample_chain(
            &tied,
            &[],
            tied.len(),
            0.0,
            1.0,
            0,
            &mut rng_tie,
        );
        assert_eq!(tbonus, 1);
        assert_eq!(rng_tie, 1);

        // Threshold gate used by prefill / step_one_token: ≤ 1e-6 is greedy.
        // (naive_sample_chain itself treats any temp>0 as softmax — the gate
        // lives in the caller, which we pin by asserting argmax equivalence.)
        const GREEDY_TEMP: f32 = 1e-6;
        assert!(GREEDY_TEMP <= 1e-6);
        assert!(0.0f32 <= 1e-6);
    }

    /// temp>0 first token uses the same empty-draft `naive_sample_chain` draw
    /// as later target tokens (`step_one_token` / prefill after set_sampling).
    /// set_sampling reseeds to 0x13579BDF; fixed seed ⇒ deterministic selection
    /// and in-place RNG advancement shared with subsequent steps.
    #[test]
    fn generic_first_token_temp_gt_zero_seeded_sample_advances_rng() {
        let vocab = 4usize;
        // Non-degenerate logits under temp sampling (would be last_logits from
        // SpecAdvance::Ready on the LLaMA/generic prefill path).
        let logits = [0.0f32, 2.0, 0.5, -1.0];
        let temp = 0.8f32;
        assert!(temp > 1e-6, "must take the sample branch, not last_argmax");
        // Matches GenericDflashSpeculator::set_sampling reseed + construction default.
        let seed = 0x13579BDF_u64;

        let mut rng_a = seed;
        let (acc_a, tok_a) =
            crate::spec_sampling::naive_sample_chain(&logits, &[], vocab, temp, 1.0, 0, &mut rng_a);
        assert_eq!(acc_a, 0, "first-token empty-draft never accepts drafts");
        assert!(
            (tok_a as usize) < vocab,
            "sampled token {tok_a} out of vocab {vocab}"
        );
        // RNG must advance (xorshift_unit mutates state on every draw).
        assert_ne!(rng_a, seed, "empty-draft sample must advance rng_state");

        // Same seed → identical draw and identical post-state (deterministic).
        let mut rng_b = seed;
        let (acc_b, tok_b) =
            crate::spec_sampling::naive_sample_chain(&logits, &[], vocab, temp, 1.0, 0, &mut rng_b);
        assert_eq!(acc_b, 0);
        assert_eq!(tok_b, tok_a);
        assert_eq!(rng_b, rng_a);

        // Stream progresses from the advanced state (not a constant of logits).
        // Later step_one_token / chain verify continue from this rng_state.
        let mut rng_c = rng_a;
        let (_acc_c, _tok_c) =
            crate::spec_sampling::naive_sample_chain(&logits, &[], vocab, temp, 1.0, 0, &mut rng_c);
        assert_ne!(rng_c, rng_a, "second draw continues the RNG stream");

        // temp≈0 branch remains last_argmax / llama::argmax (byte-identical).
        let last_argmax = crate::llama::argmax(&logits);
        assert_eq!(last_argmax, 1);
        let mut rng_g = seed;
        let (_, gbonus) =
            crate::spec_sampling::naive_sample_chain(&logits, &[], vocab, 0.0, 1.0, 0, &mut rng_g);
        assert_eq!(gbonus, last_argmax);
        assert_eq!(rng_g, seed, "greedy first token must not advance rng");
    }
}
