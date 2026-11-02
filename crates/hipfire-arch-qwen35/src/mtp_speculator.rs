// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Qwen3.5/3.6 MTP `MtpDrafter` impl — the arch-specific half of the unified
//! MTP spec-decode core ([`hipfire_runtime::spec::MtpDrafter`] +
//! [`MtpSpeculator`]).
//!
//! This owns the loaded [`Qwen35MtpHead`] + a lazily-allocated [`MtpSpecState`]
//! and drives the full-vocab compressed-serial spec step
//! ([`spec_step_mtp_compressed_serial`]) per acceptance window. It downcasts the
//! generic `&mut dyn SpecTarget` to the concrete [`ModelSlot`] — exactly as
//! `DflashSpeculator` does — so the daemon never sees a qwen35 type. The
//! arch-INvariant adaptation (prefill→`PrefillOutcome`, window→`SpecStep`) lives
//! in `MtpSpeculator<A>`; here we only implement the four fused operations.

use crate::mtp_head::{MtpKvMode, Qwen35MtpHead};
use crate::mtp_spec::{
    prefill_trunk_and_mtp_cache, prefill_trunk_and_mtp_cache_with_boundary, sample_from_logits,
    spec_step_mtp_compressed_serial_with_k,
    spec_step_mtp_compressed_serial_with_takeover_candidates, MtpPromptRoute, MtpSamplingConfig,
    MtpSpecResult, MtpSpecState, PENALTY_REPEAT_CAP,
};
use crate::speculative::{take_dn_checkpoint, DeltaNetSnapshot, ModelSlot};
use hipfire_runtime::ngram_mod::{MtpNgramContext, NgramModConfig};
use hipfire_runtime::spec::{
    terminal_prefix_replay, MtpDrafter, MtpRequestStats, MtpSpeculator, MtpWindow, PrefillOutcome,
    SpecGrammar, SpecRequestConfig, SpecTarget, Speculator,
};
use hipfire_runtime::spec_sampling::PenaltyHistory;
use rdna_compute::Gpu;

/// `HIPFIRE_NGRAM_MOD_{N_MATCH,N_MIN,N_MAX}` with production defaults.
/// `None` when the triple is invalid (max>64, zero match/max, or min>max).
/// Values resolve through the config-owned snapshot
/// ([`hipfire_config::ngram_mod_triple`]); validation stays here.
fn ngram_mod_env_config() -> Option<NgramModConfig> {
    let (n_match, n_min, n_max) = hipfire_config::ngram_mod_triple();
    if n_max <= 64 && n_max >= 1 && n_match >= 1 && n_min <= n_max {
        Some(NgramModConfig {
            capacity: 1 << 22,
            n_match,
            n_min,
            n_max,
        })
    } else {
        None
    }
}

/// Verify-batch ceiling [`Qwen35MtpDrafter::ensure_state`] builds its
/// `MtpSpecState` with for chain depth `max_n`: widened to the configured
/// n-gram ceiling only when the process explicitly enables the modifier.
pub fn mtp_verify_capacity(max_n: usize) -> usize {
    if hipfire_config::mtp_ngram_enabled() {
        ngram_mod_env_config()
            .map(|cfg| max_n.max(cfg.n_max))
            .unwrap_or(max_n)
    } else {
        max_n
    }
}

/// Chain depth the drafter actually runs with for a requested depth
/// (`Qwen35MtpDrafter::new` clamps to 1..=8).
pub fn mtp_drafter_max_n(requested: usize) -> usize {
    requested.clamp(1, 8)
}

/// qwen35 MTP drafter. `state` is allocated on the first `mtp_prefill` (it needs
/// `&mut Gpu` + the concrete `ModelSlot`, neither available at load-time
/// construction). Sampling is installed once per request via
/// [`MtpDrafter::configure_request`] and persisted across warm LCP hits.
///
/// Optional n-gram-mod composition is owned here (no Arc/Mutex): the pool may
/// survive requests; request context and counters reset per `configure_request`.
pub struct Qwen35MtpDrafter {
    head: Qwen35MtpHead,
    state: Option<MtpSpecState>,
    /// Rank-0 drafter state + per-rank verify scratch when the target is a
    /// dense-TP mesh; `state` stays unused on that path.
    tp: Option<crate::mtp_dense_tp::MtpTpRuntime>,
    /// A mesh reset cannot run from the daemon's `Gpu`, so it defers to the
    /// next prefill, which owns rank 0.
    tp_reset_pending: bool,
    max_n: usize,
    ctx_capacity: usize,
    request: SpecRequestConfig,
    /// Capacity (tokens) of the AR producer's `repeat_buf` for the loaded
    /// model: single-GPU = `ModelSlot.scratch.repeat_buf`, dense TP = rank-0
    /// `Qwen35DenseTpTarget.scratches[0].repeat_buf` (the scratch the TP AR
    /// sampler uses). It bounds the penalty window exactly as AR's
    /// `repeat_buf_cap = (repeat_buf bytes / 4).min(request repeat_window)`.
    /// Starts at [`PENALTY_REPEAT_CAP`] until a slot/mesh is first seen.
    ar_repeat_cap: usize,
    ngram: Option<MtpNgramContext>,
    ngram_active: bool,
    /// Reusable copy of the current window's n-gram candidates (borrow split
    /// from `ngram`; no per-window allocation).
    ngram_candidates: Vec<u32>,
    /// Request-local wire counters (reset on configure_request).
    stats: MtpRequestStats,
    /// Identity of the window whose pre-verify state is still in `trunk_snap`.
    last_window: Option<(usize, u32)>,
    /// Bounded recurrent-state snapshots used to resume a divergent re-render.
    checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    checkpoint_resume: bool,
    checkpoint_interval: usize,
    checkpoint_cap: usize,
    /// Strict-prefix terminal-repair enablement, resolved once at construction
    /// from [`hipfire_config::mtp_cache_policy`] (config-owned). The repair
    /// path performs no policy reads of its own.
    window_rollback: bool,
    /// Trunk route for prompt fill, forced advance and terminal-repair replay,
    /// resolved once at construction from [`hipfire_config::mtp_own_prefill`].
    prompt_route: MtpPromptRoute,
}

impl Qwen35MtpDrafter {
    pub fn new(head: Qwen35MtpHead, max_n: usize, ctx_capacity: usize) -> Self {
        // Cache policy is config-owned: resolved once here from the process
        // snapshot via [`hipfire_config::mtp_cache_policy`] (never per-repair
        // or per-step), mirroring `build_dflash_speculator`'s one-shot
        // resolution for the DFlash ring.
        let policy = hipfire_config::mtp_cache_policy();
        let prompt_route = MtpPromptRoute::from_own_prefill(hipfire_config::mtp_own_prefill());
        eprintln!(
            "  qwen35 MTP prompt fill route: {}",
            match prompt_route {
                MtpPromptRoute::ArRoute => "ar (AR prefill route)",
                MtpPromptRoute::Own => "own (HIPFIRE_MTP_OWN_PREFILL=1)",
            }
        );
        Self {
            head,
            state: None,
            tp: None,
            tp_reset_pending: false,
            max_n: mtp_drafter_max_n(max_n),
            ctx_capacity,
            request: SpecRequestConfig::default(),
            ar_repeat_cap: PENALTY_REPEAT_CAP,
            ngram: None,
            ngram_active: false,
            ngram_candidates: Vec::new(),
            stats: MtpRequestStats::default(),
            last_window: None,
            checkpoints: Vec::new(),
            checkpoint_resume: policy.checkpoint_resume,
            checkpoint_interval: policy.checkpoint_interval,
            checkpoint_cap: policy.checkpoint_cap,
            window_rollback: policy.window_rollback,
            prompt_route,
        }
    }
    /// Borrow the live MTP spec state (`None` before the first `mtp_prefill`
    /// allocates it). Attestation hook for hardware byte-identity proofs:
    /// lets a test fingerprint the MTP-head KV / `prev_hidden` resident in
    /// the drafter without re-implementing the prefill/step/repair paths.
    pub fn mtp_live_state(&self) -> Option<&MtpSpecState> {
        self.state.as_ref()
    }

    fn tp_target(
        target: &mut dyn SpecTarget,
    ) -> Option<&mut crate::mtp_dense_tp::Qwen35DenseTpTarget> {
        target
            .as_any_mut()
            .downcast_mut::<crate::mtp_dense_tp::Qwen35DenseTpTarget>()
    }

    /// The head, the mesh target and its lazily built rank-0 runtime; `what`
    /// names the caller in the error when `target` is not a dense-TP mesh.
    fn tp_runtime<'a>(
        &'a mut self,
        target: &'a mut dyn SpecTarget,
        what: &str,
    ) -> Result<
        (
            &'a Qwen35MtpHead,
            &'a mut crate::mtp_dense_tp::Qwen35DenseTpTarget,
            &'a mut crate::mtp_dense_tp::MtpTpRuntime,
        ),
        String,
    > {
        let tp = Self::tp_target(target)
            .ok_or_else(|| format!("dense TP {what} without a TP target"))?;
        let Self {
            head,
            tp: runtime,
            max_n,
            request,
            ar_repeat_cap,
            ..
        } = self;
        if runtime.is_none() {
            // Dense TP has no single-GPU AR scratch; rank 0's scratch is the
            // one the TP AR sampler penalizes through.
            *ar_repeat_cap = tp
                .scratches
                .first()
                .map(|s| s.repeat_buf.buf.size() / 4)
                .unwrap_or(PENALTY_REPEAT_CAP);
            let mut rt = crate::mtp_dense_tp::MtpTpRuntime::new(tp, head, *max_n, MtpKvMode::Q8)?;
            Self::apply_request(&mut rt.state, *request, *ar_repeat_cap);
            *runtime = Some(rt);
        }
        Ok((head, tp, runtime.as_mut().expect("built above")))
    }

    /// Downcast the generic target to a qwen35 `ModelSlot` (same as DflashSpeculator).
    fn slot(target: &mut dyn SpecTarget) -> Result<&mut ModelSlot, String> {
        target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or_else(|| "Qwen35MtpDrafter: target is not a Qwen3.5 ModelSlot".to_string())
    }

    /// Qwen3.x AR penalty window for `cfg`: the GPU sampler's history is the
    /// generated tokens only, its kernel gate is `repeat_penalty > 1.0` /
    /// `presence > 0` / `frequency > 0`, and the window is
    /// `min(AR repeat_buf capacity, request repeat_window.max(1))`. `0` ⇒ no
    /// penalty (neutral request: no history work, no GPU prepass).
    fn penalty_window(cfg: &SpecRequestConfig, ar_cap: usize) -> usize {
        if cfg.repeat_penalty > 1.0 || cfg.presence_penalty > 0.0 || cfg.frequency_penalty > 0.0 {
            ar_cap.min(cfg.repeat_window.max(1)).min(PENALTY_REPEAT_CAP)
        } else {
            0
        }
    }

    /// Re-bind only the penalty controls + a fresh [`PenaltyHistory`] (used
    /// when the AR capacity becomes known after the request was installed;
    /// leaves the RNG and the rest of the sampling untouched).
    fn install_penalty(state: &mut MtpSpecState, cfg: &SpecRequestConfig, ar_cap: usize) {
        let w = Self::penalty_window(cfg, ar_cap);
        state.sampling.repeat_penalty = cfg.repeat_penalty;
        state.sampling.repeat_window = w;
        state.sampling.presence_penalty = cfg.presence_penalty;
        state.sampling.frequency_penalty = cfg.frequency_penalty;
        state.penalty = PenaltyHistory::new(w);
    }

    /// Install sampling without changing the independent MTP draft-confidence
    /// cutoff initialized by `MtpSpecState` from its arch/env default.
    ///
    /// Penalties follow the Qwen3.x AR window rule ([`Self::penalty_window`]).
    /// The fresh [`PenaltyHistory`] is generation-only (`set_prompt` is never
    /// called): each window rebuilds it from the loop's `emitted`, so the
    /// `emitted` repeat-penalty contract holds and the prompt never enters it.
    fn apply_request(state: &mut MtpSpecState, cfg: SpecRequestConfig, ar_cap: usize) {
        let top_p = if cfg.top_p > 0.0 {
            cfg.top_p.min(1.0)
        } else {
            1.0
        };
        let w = Self::penalty_window(&cfg, ar_cap);
        state.set_sampling(
            MtpSamplingConfig {
                temp: cfg.temp,
                top_k: cfg.ar_candidate_cap(),
                top_p,
                min_p: cfg.min_p,
                repeat_penalty: cfg.repeat_penalty,
                repeat_window: w,
                presence_penalty: cfg.presence_penalty,
                frequency_penalty: cfg.frequency_penalty,
            },
            cfg.rng_seed,
        );
        state.penalty = PenaltyHistory::new(w);
    }

    /// Allocate `state` against `slot` on first use, plus both compressed
    /// sidecar scratches when the head ships a compressed vocab, then apply
    /// the current request. Subsequent calls are a no-op (warm hits persist).
    ///
    /// Verify capacity expands to the configured n-gram ceiling only when the
    /// process explicitly enables the modifier. This preserves normal MTP's
    /// smaller allocation while allowing a later eligible request to arm
    /// n-gram-mod without reallocating/destroying warm prefix state.
    fn ensure_state(&mut self, gpu: &mut Gpu, slot: &ModelSlot) -> Result<(), String> {
        // AR's penalty window is bounded by this slot's `repeat_buf`; refresh
        // it and re-bind the penalty only if the capacity changed.
        let slot_cap = slot.scratch.repeat_buf.buf.size() / 4;
        if slot_cap != self.ar_repeat_cap {
            self.ar_repeat_cap = slot_cap;
            if let Some(st) = self.state.as_mut() {
                Self::install_penalty(st, &self.request, slot_cap);
            }
        }
        if self.state.is_none() {
            let verify_capacity = mtp_verify_capacity(self.max_n);
            let mut st = MtpSpecState::new_for_slot_with_kv_mode_and_verify_capacity(
                gpu,
                slot,
                &self.head,
                self.max_n,
                verify_capacity,
                MtpKvMode::Q8,
            )
            .map_err(|e| format!("alloc MtpSpecState: {e}"))?;
            if let Some(cvs) = self.head.weights.compressed_vocab_size {
                let compressed_result = st
                    .mtp_scratch
                    .ensure_compressed_logits(gpu, cvs)
                    .map_err(|e| format!("alloc logits_compressed: {e}"))
                    .and_then(|_| {
                        st.ensure_compressed_lm_logits(gpu, cvs)
                            .map_err(|e| format!("alloc mtp_lm_logits_compressed: {e}"))
                    });
                if let Err(error) = compressed_result {
                    st.free_gpu(gpu);
                    return Err(error);
                }
            }
            Self::apply_request(&mut st, self.request, self.ar_repeat_cap);
            self.state = Some(st);
        }
        Ok(())
    }

    fn begin_ngram_request(&mut self, prompt_tokens: &[u32]) {
        // Begin request-local n-gram history exactly once. Prefix realignment
        // calls prefill again inside the same request; preserving this context
        // keeps counters and already-emitted history intact. The first token
        // reaches the context through `emitted` in `mtp_step`.
        if !self.ngram_active {
            return;
        }
        if let Some(ctx) = self.ngram.as_mut() {
            if !ctx.is_begun() {
                ctx.begin_request(prompt_tokens);
            }
        }
    }

    fn tp_prefill(
        &mut self,
        target: &mut dyn SpecTarget,
        fill_tokens: &[u32],
        start_pos: usize,
        cache_hit: bool,
    ) -> Result<u32, String> {
        let reset_pending = std::mem::take(&mut self.tp_reset_pending);
        let (head, tp, rt) = self.tp_runtime(target, "prefill")?;
        if !cache_hit || reset_pending {
            let dev0 = &mut tp.gpus.devices[0];
            dev0.bind_thread().map_err(|e| e.to_string())?;
            rt.state
                .reset(dev0)
                .map_err(|e| format!("dense TP MTP state reset: {e}"))?;
        }
        crate::mtp_dense_tp::prefill_trunk_and_mtp_cache_dense_tp(
            tp,
            head,
            rt,
            fill_tokens,
            start_pos,
        )?;
        let logits = tp.last_logits()?;
        let state = &mut rt.state;
        let first_token = if state.sampling.is_greedy() {
            logits
                .iter()
                .enumerate()
                .fold((0u32, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i as u32, v)
                    } else {
                        (bi, bv)
                    }
                })
                .0
        } else {
            // The first token is drawn unpenalized: AR's first token has an
            // empty generation history (the prompt is never penalized).
            sample_from_logits(&logits, &state.sampling, &mut state.rng).0
        };
        Ok(first_token)
    }

    fn tp_prefill_tokens(
        &mut self,
        target: &mut dyn SpecTarget,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<(), String> {
        let (head, tp, rt) = self.tp_runtime(target, "advance")?;
        crate::mtp_dense_tp::prefill_trunk_and_mtp_cache_dense_tp(tp, head, rt, tokens, start_pos)
    }

    fn tp_step(
        &mut self,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        eos: u32,
        k: usize,
    ) -> Result<MtpSpecResult, String> {
        let (head, tp, rt) = self.tp_runtime(target, "step")?;
        // `rt.state` is the very state `spec_step_mtp_dense_tp` verifies
        // with, so row i sees suffix_W(emitted ‖ drafts[..i]).
        rt.state.penalty.begin_window(emitted);
        crate::mtp_dense_tp::spec_step_mtp_dense_tp(tp, head, rt, position, seed, eos, k)
    }

    fn clear_checkpoints(&mut self, gpu: &mut Gpu) {
        for (_, snapshot) in self.checkpoints.drain(..) {
            snapshot.free_gpu(gpu);
        }
    }
}

impl MtpDrafter for Qwen35MtpDrafter {
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
        self.last_window = None;
        if abort() {
            return Ok(PrefillOutcome::Aborted);
        }
        // Cold start: reset the trunk recurrent state (target owns it) BEFORE
        // borrowing the concrete slot, so positions start at 0. Warm hit
        // preserves target + MTP cache for honest LCP suffix prefill.
        if !cache_hit {
            target.reset_recurrent(gpu)?;
        }
        if Self::tp_target(target).is_some() {
            let first_token = self.tp_prefill(target, fill_tokens, start_pos, cache_hit)?;
            if abort() {
                return Ok(PrefillOutcome::Aborted);
            }
            self.begin_ngram_request(prompt_tokens);
            return Ok(PrefillOutcome::Ready { first_token });
        }
        let slot = Self::slot(target)?;
        self.ensure_state(gpu, slot)?;
        let state = self.state.as_mut().expect("ensure_state set it");
        if !cache_hit {
            // Clear the head KV so its absolute positions start clean too.
            state
                .reset(gpu)
                .map_err(|e| format!("mtp state reset: {e}"))?;
        }

        let mut checkpoints = std::mem::take(&mut self.checkpoints);
        if !cache_hit {
            for (_, snapshot) in checkpoints.drain(..) {
                snapshot.free_gpu(gpu);
            }
        }
        let checkpoint_resume = self.checkpoint_resume;
        let prompt_route = self.prompt_route;
        let checkpoint_interval = self.checkpoint_interval;
        let checkpoint_cap = self.checkpoint_cap;
        let prefill = prefill_trunk_and_mtp_cache_with_boundary(
            gpu,
            slot,
            &self.head,
            state,
            fill_tokens,
            start_pos,
            prompt_route,
            |gpu, dn_state, kv, position| {
                if checkpoint_resume {
                    take_dn_checkpoint(
                        &mut checkpoints,
                        dn_state,
                        kv,
                        gpu,
                        position,
                        checkpoint_interval,
                        checkpoint_cap,
                    );
                }
                Ok(())
            },
        );
        self.checkpoints = checkpoints;
        prefill.map_err(|e| format!("mtp prefill: {e}"))?;
        if abort() {
            return Ok(PrefillOutcome::Aborted);
        }

        // Seed from the trunk logits at the last prefilled position
        // (`prefill_trunk_and_mtp_cache` leaves them in scratch.logits).
        // Greedy = argmax; temp>0 draws from the state's continuous RNG/config.
        let logits = gpu
            .download_f32(&slot.scratch.logits)
            .map_err(|e| format!("download seed logits: {e}"))?;
        let first_token = if state.sampling.is_greedy() {
            logits
                .iter()
                .enumerate()
                .fold((0u32, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i as u32, v)
                    } else {
                        (bi, bv)
                    }
                })
                .0
        } else {
            // First token is unpenalized: AR's first token has empty history.
            sample_from_logits(&logits, &state.sampling, &mut state.rng).0
        };

        self.begin_ngram_request(prompt_tokens);
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
        // qwen35 enforces tool-call grammar post-hoc in the emission layer; the
        // in-step grammar handle is unused here. `k` is the per-call budget from
        // MtpSpeculator (already clamped by remaining max_emit); the fixed-window
        // compressed-serial core must honor it — never draft state.max_n blindly.
        // Sampling lives on `state` from configure_request; no per-step reconfig.
        self.ngram_candidates.clear();
        if self.ngram_active {
            if let Some(ctx) = self.ngram.as_mut() {
                if Self::tp_target(target).is_some() {
                    // The mesh has no takeover verify: learn emitted history only.
                    ctx.sync_emitted(emitted);
                } else if let Some(c) = ctx.propose(emitted, k) {
                    self.ngram_candidates.extend_from_slice(c);
                }
                if !emitted.is_empty() {
                    debug_assert!(
                        ctx.history().ends_with(emitted),
                        "qwen35 MTP ngram context suffix must match emitted (ctx={}, emitted={})",
                        ctx.history().len(),
                        emitted.len()
                    );
                }
            }
        }
        let used_ngram = !self.ngram_candidates.is_empty();
        let modifier = self.ngram_active;
        let native_k = k.min(self.max_n);

        if Self::tp_target(target).is_some() {
            // The mesh has no takeover verify, so an n-gram-mod hit is not
            // used there: the window drafts natively.
            let r = self.tp_step(target, position, seed, emitted, eos, native_k)?;
            self.last_window = Some((position, seed));
            self.stats.mtp_windows += 1;
            return Ok(MtpWindow {
                committed: r.committed,
                accepted: r.accept_count,
                drafts_generated: r.drafts_generated,
            });
        }
        let slot = Self::slot(target)?;
        let window_start =
            crate::mtp_spec::mtp_phase_timing_enabled().then(std::time::Instant::now);
        let (r, pen) = {
            let state = self
                .state
                .as_mut()
                .ok_or("Qwen35MtpDrafter: mtp_step before mtp_prefill")?;
            // Row i of the verify sees suffix_W(emitted ‖ drafts[..i]); `emitted`
            // already holds the pending seed. Rebuilt every window (also for
            // the n-gram takeover), so rejected/pruned drafts never leak.
            if window_start.is_some() {
                state.penalty_timing = Default::default();
            }
            state.penalty.begin_window(emitted);
            let r = if used_ngram {
                // The takeover fills the head KV for every row it commits, so
                // native MTP stays live for the next pool miss.
                spec_step_mtp_compressed_serial_with_takeover_candidates(
                    gpu,
                    slot,
                    &self.head,
                    state,
                    position,
                    seed,
                    eos,
                    &self.ngram_candidates,
                )
            } else {
                spec_step_mtp_compressed_serial_with_k(
                    gpu, slot, &self.head, state, position, seed, eos, native_k,
                )
            }
            .map_err(|e| e.to_string())?;
            (r, state.penalty_timing)
        };
        // `QWEN35_MTP_PHASE` is the Qwen3.x counterpart of Qwen4's
        // `QWEN4_MTP_PHASE`: one synced per-window line, only under
        // `HIPFIRE_MTP_PHASE_TIMING=1`.
        if let Some(window_start) = window_start {
            let _ = gpu.hip.device_synchronize();
            let window_us = window_start.elapsed().as_secs_f64() * 1e6;
            eprintln!(
                "QWEN35_MTP_PHASE {{\"event\":\"mtp_phase\",\"source\":\"{}\",\"position\":{},\"k\":{},\"accepted\":{},\"emitted\":{},\"window_us\":{:.1},\"penalty_host_us\":{:.1},\"penalty_launches\":{},\"penalty_uploads\":{},\"t_end\":{}}}",
                if used_ngram { "ngram" } else { "mtp" },
                position,
                r.drafts_generated,
                r.accept_count,
                r.committed.len(),
                window_us,
                pen.host_us,
                pen.launches,
                pen.uploads,
                rdna_compute::profile::unix_micros(),
            );
        }
        self.last_window = Some((position, seed));
        let budget = if used_ngram { k } else { native_k };
        debug_assert!(
            r.committed.len() <= budget + 1,
            "qwen35 MTP committed past k budget"
        );

        if used_ngram {
            self.stats.ngram_mod_windows += 1;
            self.stats.ngram_mod_drafts += r.drafts_generated;
            self.stats.ngram_mod_accepted += r.accept_count;
            if let Some(ctx) = self.ngram.as_mut() {
                ctx.observe_result(r.drafts_generated, r.accept_count);
            }
        } else {
            self.stats.mtp_windows += 1;
        }
        if modifier {
            // `retired=0` keeps the historical line format; nothing retires MTP.
            eprintln!(
                "[mtp-ngram] used={} drafted={} accepted={} retired=0",
                u8::from(used_ngram),
                r.drafts_generated,
                r.accept_count,
            );
        }

        Ok(MtpWindow {
            committed: r.committed,
            accepted: r.accept_count,
            drafts_generated: r.drafts_generated,
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
        self.last_window = None;
        // Forced tokens must land in BOTH the trunk and the MTP head KV /
        // prev_hidden. Plain spec_advance only moves trunk state, leaving an
        // unwritten hole in the head that poisons later draft steps.
        if tokens.is_empty() {
            return Ok(true);
        }
        if abort() {
            return Ok(true);
        }
        if Self::tp_target(target).is_some() {
            self.tp_prefill_tokens(target, tokens, start_pos)
                .map_err(|e| format!("qwen35 MTP forced advance (dense TP): {e}"))?;
            return Ok(true);
        }
        let slot = Self::slot(target)?;
        self.ensure_state(gpu, slot)?;
        let state = self.state.as_mut().expect("ensure_state set it");
        let route = self.prompt_route;
        prefill_trunk_and_mtp_cache(gpu, slot, &self.head, state, tokens, start_pos, route)
            .map_err(|e| format!("qwen35 MTP forced advance: {e}"))?;
        Ok(true)
    }

    fn mtp_repair_terminal_prefix(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        window_start: usize,
        window_seed: u32,
        consumed: &[u32],
    ) -> Result<bool, String> {
        if !self.window_rollback {
            return Ok(false);
        }
        let Some((saved_start, saved_seed)) = self.last_window.take() else {
            return Ok(false);
        };
        if (saved_start, saved_seed) != (window_start, window_seed) {
            return Err(format!(
                "qwen35 MTP terminal repair window mismatch (saved pos={saved_start} seed={saved_seed}, requested pos={window_start} seed={window_seed})"
            ));
        }

        if Self::tp_target(target).is_some() {
            let replay = terminal_prefix_replay(window_seed, consumed);
            {
                let tp = Self::tp_target(target).expect("checked above");
                let Some(rt) = self.tp.as_ref() else {
                    return Ok(false);
                };
                rt.restore_last_window(tp)?;
            }
            if replay.is_empty() {
                return Ok(true);
            }
            self.tp_prefill_tokens(target, &replay, window_start)
                .map_err(|e| format!("qwen35 MTP terminal repair replay (dense TP): {e}"))?;
            return Ok(true);
        }
        let slot = Self::slot(target)?;
        let Some(state) = self.state.as_mut() else {
            return Ok(false);
        };
        state
            .trunk_snap
            .restore_to(&mut slot.dn_state, gpu)
            .map_err(|e| format!("qwen35 MTP terminal repair restore: {e}"))?;

        let replay = terminal_prefix_replay(window_seed, consumed);
        if replay.is_empty() {
            return Ok(true);
        }
        let route = self.prompt_route;
        prefill_trunk_and_mtp_cache(gpu, slot, &self.head, state, &replay, window_start, route)
            .map_err(|e| format!("qwen35 MTP terminal repair replay: {e}"))?;
        Ok(true)
    }

    fn mtp_reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.last_window = None;
        self.tp_reset_pending = self.tp.is_some();
        self.stats = MtpRequestStats::default();
        if let Some(ctx) = self.ngram.as_mut() {
            ctx.reset_request();
        }
        self.clear_checkpoints(gpu);
        if let Some(state) = self.state.as_mut() {
            state
                .reset(gpu)
                .map_err(|e| format!("qwen35-mtp drafter reset: {e}"))?;
        }
        Ok(())
    }

    fn mtp_reset_for_realign(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.last_window = None;
        self.tp_reset_pending = self.tp.is_some();
        self.clear_checkpoints(gpu);
        if let Some(state) = self.state.as_mut() {
            state
                .reset(gpu)
                .map_err(|e| format!("qwen35-mtp drafter realign reset: {e}"))?;
        }
        Ok(())
    }

    fn mtp_checkpoint_positions(&self) -> Vec<usize> {
        if self.checkpoint_resume {
            self.checkpoints
                .iter()
                .map(|(position, _)| *position)
                .collect()
        } else {
            Vec::new()
        }
    }

    fn mtp_rewind_to(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
    ) -> Result<usize, String> {
        self.last_window = None;
        let Some(index) = self
            .checkpoints
            .iter()
            .rposition(|(checkpoint, _)| *checkpoint == position)
        else {
            return Err(format!(
                "qwen35 MTP checkpoint {position} was advertised but is no longer resident"
            ));
        };
        let slot = Self::slot(target)?;
        self.checkpoints[index]
            .1
            .restore_to(&mut slot.dn_state, gpu)
            .map_err(|e| format!("qwen35 MTP checkpoint restore at {position}: {e}"))?;
        for (_, snapshot) in self.checkpoints.drain(index + 1..) {
            snapshot.free_gpu(gpu);
        }
        Ok(position)
    }

    fn mtp_free(self: Box<Self>, gpu: &mut Gpu) {
        let mut this = *self;
        if this.tp.is_some() {
            // The per-rank verify scratch is unreachable from one `Gpu`; a
            // mesh drafter must go through `mtp_free_multi`, so say so loudly
            // instead of dropping those buffers on the floor.
            debug_assert!(
                false,
                "qwen35 MTP drafter with a dense-TP runtime freed via mtp_free"
            );
            eprintln!(
                "[mtp] BUG: dense-TP drafter freed via mtp_free; per-rank verify scratch leaked"
            );
        }
        this.clear_checkpoints(gpu);
        if let Some(state) = this.state {
            state.free_gpu(gpu);
        }
        this.head.free_gpu(gpu);
    }

    fn mtp_free_multi(self: Box<Self>, gpus: &mut hipfire_runtime::multi_gpu::Gpus) {
        let mut this = *self;
        if let Some(tp) = this.tp.take() {
            tp.free_on_ranks(gpus);
        }
        if let Some(dev) = gpus.devices.first_mut() {
            let _ = dev.bind_thread();
            this.clear_checkpoints(dev);
            if let Some(state) = this.state {
                state.free_gpu(dev);
            }
            this.head.free_gpu(dev);
        }
    }

    fn k(&self) -> usize {
        self.max_n
    }

    fn proposal_capacity(&self) -> usize {
        if self.ngram_active {
            self.ngram
                .as_ref()
                .map(|c| self.max_n.max(c.config().n_max))
                .unwrap_or(self.max_n)
        } else {
            self.max_n
        }
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn requires_greedy(&self) -> bool {
        false
    }

    fn supports_temp_verify(&self) -> bool {
        true
    }

    fn configure_request(&mut self, cfg: SpecRequestConfig) {
        self.request = cfg;
        if let Some(state) = self.state.as_mut() {
            Self::apply_request(state, cfg, self.ar_repeat_cap);
        }
        if let Some(rt) = self.tp.as_mut() {
            Self::apply_request(&mut rt.state, cfg, self.ar_repeat_cap);
        }
        self.stats = MtpRequestStats::default();
        if let Some(ctx) = self.ngram.as_mut() {
            ctx.reset_request();
        }
        self.ngram_active = false;
        // Sampled requests arm the pool too: takeover windows verify with the
        // deterministic draft's point-mass q (see
        // `spec_step_mtp_compressed_serial_with_takeover_candidates`).
        if !cfg.allow_ngram_modifier {
            return;
        }
        let Some(new_cfg) = ngram_mod_env_config() else {
            return;
        };
        let reuse = self.ngram.as_ref().is_some_and(|c| c.config() == &new_cfg);
        if !reuse {
            match MtpNgramContext::new(new_cfg) {
                Ok(ctx) => self.ngram = Some(ctx),
                Err(_) => return,
            }
        }
        self.ngram_active = self.ngram.is_some();
        self.stats.mtp_ngram = self.ngram_active;
    }

    fn request_stats(&self) -> MtpRequestStats {
        let mut s = self.stats;
        s.mtp_ngram = self.ngram_active;
        s.ngram_mod_accept_rate = if s.ngram_mod_drafts > 0 {
            let rate = s.ngram_mod_accepted as f64 / s.ngram_mod_drafts as f64;
            (rate * 1000.0).round() / 1000.0
        } else {
            0.0
        };
        s
    }
}

/// Build the qwen35 MTP speculator (the boxed `dyn Speculator` the loader's
/// `build_speculator` returns). The `MtpSpecState` is allocated lazily on the
/// first `prefill`.
pub fn build_qwen35_mtp_speculator(
    head: Qwen35MtpHead,
    max_n: usize,
    ctx_capacity: usize,
) -> Box<dyn Speculator> {
    Box::new(MtpSpeculator::new(Qwen35MtpDrafter::new(
        head,
        max_n,
        ctx_capacity,
    )))
}

/// A continuous-batching spec lane's MTP state, allocated and configured
/// exactly as [`Qwen35MtpDrafter`] does for a request on the singleton
/// route (`ensure_state` + `configure_request`, no n-gram modifier): verify
/// capacity `max_n`, Q8 head KV, the compressed-vocab scratches when the
/// head ships them, then the request's sampling and penalty window against
/// the AR repeat capacity `ar_cap`. `dn` sizes the trunk snapshot.
pub fn new_qwen35_mtp_lane_state(
    gpu: &mut Gpu,
    config: &crate::qwen35::Qwen35Config,
    dn: &crate::qwen35::DeltaNetState,
    head: &Qwen35MtpHead,
    max_n: usize,
    request: SpecRequestConfig,
    ar_cap: usize,
) -> Result<MtpSpecState, String> {
    let max_n = max_n.clamp(1, 8);
    let mut st = MtpSpecState::new_for_components_with_verify_capacity(
        gpu,
        config,
        dn,
        head,
        max_n,
        max_n,
        MtpKvMode::Q8,
    )
    .map_err(|e| format!("alloc MtpSpecState: {e}"))?;
    if let Some(cvs) = head.weights.compressed_vocab_size {
        let compressed = st
            .mtp_scratch
            .ensure_compressed_logits(gpu, cvs)
            .map_err(|e| format!("alloc logits_compressed: {e}"))
            .and_then(|_| {
                st.ensure_compressed_lm_logits(gpu, cvs)
                    .map_err(|e| format!("alloc mtp_lm_logits_compressed: {e}"))
            });
        if let Err(error) = compressed {
            st.free_gpu(gpu);
            return Err(error);
        }
    }
    Qwen35MtpDrafter::apply_request(&mut st, request, ar_cap);
    Ok(st)
}
