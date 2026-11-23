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
    prefill_trunk_and_mtp_cache, prefill_trunk_and_mtp_cache_with_boundary_deferred,
    sample_from_logits, spec_step_mtp_compressed_serial_with_k,
    spec_step_mtp_compressed_serial_with_takeover_candidates, MtpPromptRoute, MtpSamplingConfig,
    MtpSpecResult, MtpSpecState, PendingMtpFill, PENALTY_REPEAT_CAP,
};
use crate::speculative::{
    checkpoint_fits_admitted_context, take_dn_boundary_checkpoint, take_turn_checkpoint,
    DeltaNetSnapshot, ModelSlot, TurnCheckpoint,
};
use hipfire_runtime::checkpoint_pool::prefix_fingerprint;
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

/// MTP chain depth requested by the load policy (before the drafter's own
/// 1..=8 clamp): `spec.mtp_k`, else the process default, clamped to 1..=10.
pub fn mtp_requested_k(spec: &hipfire_runtime::loader_api::SpecLoadCfg) -> usize {
    spec.mtp_k
        .unwrap_or(
            hipfire_runtime::config::get()
                .mtp_k
                .unwrap_or(hipfire_config::MTP_K_DEFAULT),
        )
        .clamp(1, 10)
}

/// Head fill a deferring prefill left owed, and whether that prefill added a
/// prompt-end checkpoint to the ring (which the owed fill guards).
struct OwedFill {
    fill: PendingMtpFill,
    ring_added: bool,
}

/// Seed a prompt-end payload carries besides its DeltaNet image: the final
/// trunk logits row and the head's `prev_hidden`, copied after the owed head
/// fill (if any) is known, plus the prompt it describes.
struct PromptEndSeed {
    len: usize,
    /// [`prefix_fingerprint`] of the whole prompt `[0, len)`.
    fingerprint: u64,
    logits: Vec<f32>,
    prev_hidden: Vec<f32>,
    /// The head KV covers `[0, len)`: no owed fill remains. Only a settled
    /// payload is advertised or restorable.
    head_settled: bool,
}

/// One bounded, drafter-owned prompt-end payload, separate from the ordinary
/// stride / turn checkpoint ring. `snapshot` is the sealed DeltaNet image and
/// stays allocated across requests; `seed` is `None` while the payload is
/// stale (cold prefill, rewritten prefix) and is refilled by the next capture.
struct PromptEndPayload {
    snapshot: DeltaNetSnapshot,
    seed: Option<PromptEndSeed>,
}

fn prompt_end_trace() -> bool {
    hipfire_config::developer_var("HIPFIRE_QWEN_CACHE_TRACE")
        .ok()
        .as_deref()
        == Some("1")
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: any f32 is plain bytes; length is exactly the slice's byte size.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

/// First token from the trunk's last-row logits: greedy strict-`>` fold, else
/// a draw from the state's continuous RNG/config with an empty penalty history
/// (AR's first token has none). Shared by a fresh prefill and a prompt-end
/// restore so both produce the token from the same code.
fn seed_first_token(logits: &[f32], state: &mut MtpSpecState) -> u32 {
    if state.sampling.is_greedy() {
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
        sample_from_logits(logits, &state.sampling, &mut state.rng).0
    }
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
    checkpoint_interval: usize,
    checkpoint_cap: usize,
    /// This request's user-turn boundary (`Speculator::set_turn_boundary`), taken by
    /// the prompt prefill.
    turn_boundary: Option<usize>,
    /// Position of the newest user-turn snapshot in `checkpoints`, protected from
    /// recycling until the next user turn replaces it.
    pinned_boundary: Option<usize>,
    /// Trunk route for prompt fill, forced advance and terminal-repair replay,
    /// resolved once at construction from [`hipfire_config::mtp_own_prefill`].
    prompt_route: MtpPromptRoute,
    /// One-shot opt-in of the next prefill to leave its tail fill owed.
    defer_prefill_tail: bool,
    /// Head fill the last deferring prefill left owed (see [`OwedFill`]).
    owed_fill: Option<OwedFill>,
    /// Sealed prompt-end payload of the newest prefill that reached its prompt
    /// end (see [`PromptEndPayload`]).
    prompt_end: Option<PromptEndPayload>,
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
            checkpoint_interval: policy.checkpoint_interval,
            checkpoint_cap: policy.checkpoint_cap,
            turn_boundary: None,
            pinned_boundary: None,
            prompt_route,
            defer_prefill_tail: false,
            owed_fill: None,
            prompt_end: None,
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

    /// Head-fill rows a deferring prefill still owes (0 when settled).
    /// Attestation hook for the deferred-tail oracle and the daemon's
    /// "no debt at a request terminal" invariant.
    pub fn owed_fill_rows(&self) -> usize {
        self.owed_fill.as_ref().map_or(0, |owed| owed.fill.rows())
    }

    /// Checkpoint position the owed fill withholds, if its prompt-end snapshot
    /// was taken by the deferring prefill.
    fn withheld_checkpoint(&self) -> Option<usize> {
        self.owed_fill
            .as_ref()
            .filter(|owed| owed.ring_added)
            .map(|owed| owed.fill.committed_pos())
    }

    /// Drop the ring entry the deferring prefill added at the prompt end.
    fn drop_owed_checkpoint(&mut self, gpu: &mut Gpu, position: usize) {
        if let Some(index) = self.checkpoints.iter().rposition(|(p, _)| *p == position) {
            let (_, snapshot) = self.checkpoints.remove(index);
            snapshot.free_gpu(gpu);
        }
    }

    /// Discard owed head fill without running it (new cold prefill, reset,
    /// unload, abort, failure): the prompt-end checkpoint it guarded goes too.
    fn discard_owed_fill(&mut self, gpu: &mut Gpu) {
        if let Some(owed) = self.owed_fill.take() {
            self.invalidate_prompt_end();
            let position = owed.fill.committed_pos();
            let ring_added = owed.ring_added;
            owed.fill.discard(gpu);
            if ring_added {
                self.drop_owed_checkpoint(gpu, position);
            }
        }
    }

    /// Run owed head fill (idempotent). On failure the debt and the checkpoint
    /// it guarded are dropped and the error returned; the caller fails closed.
    fn settle_owed_fill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
    ) -> Result<(), String> {
        if self.owed_fill.is_none() {
            return Ok(());
        }
        let slot = match Self::slot(target) {
            Ok(slot) => slot,
            Err(error) => {
                self.discard_owed_fill(gpu);
                return Err(error);
            }
        };
        let Some(state) = self.state.as_mut() else {
            self.discard_owed_fill(gpu);
            return Err("qwen35 MTP: owed prompt fill without drafter state".into());
        };
        let owed = self.owed_fill.take().expect("checked above");
        let position = owed.fill.committed_pos();
        let ring_added = owed.ring_added;
        if let Err(error) = owed.fill.complete(gpu, &self.head, state, &slot.weights) {
            if ring_added {
                self.drop_owed_checkpoint(gpu, position);
            }
            self.invalidate_prompt_end();
            return Err(format!("qwen35 MTP deferred prompt fill: {error}"));
        }
        if let Some(seed) = self.prompt_end.as_mut().and_then(|p| p.seed.as_mut()) {
            if seed.len == position {
                seed.head_settled = true;
            }
        }
        Ok(())
    }

    /// Length of the resident (non-stale) prompt-end seed.
    fn prompt_end_len(&self) -> Option<usize> {
        self.prompt_end.as_ref()?.seed.as_ref().map(|seed| seed.len)
    }

    /// Mark the prompt-end payload stale; its snapshot buffers stay allocated
    /// so the next capture reuses them.
    fn invalidate_prompt_end(&mut self) {
        if let Some(payload) = self.prompt_end.as_mut() {
            payload.seed = None;
        }
    }

    /// Release the prompt-end payload's device memory (reset / unload).
    fn free_prompt_end(&mut self, gpu: &mut Gpu) {
        if let Some(payload) = self.prompt_end.take() {
            payload.snapshot.free_gpu(gpu);
        }
    }

    /// Capture the sealed prompt end after a prefill whose fill reached the end
    /// of `prompt_tokens`: a DeltaNet image of the live (already sealed) state,
    /// the last-row logits and `prev_hidden`. The target KV prefix and the
    /// head KV prefix are not copied: both are absolute-position storage that
    /// append-only decode never rewrites below the prompt end. Skipped (the
    /// route keeps its checkpoint behavior) when the snapshot does not fit the
    /// admitted context, the KV is compacted, or any copy fails.
    #[allow(clippy::too_many_arguments)]
    fn capture_prompt_end(
        payload: &mut Option<PromptEndPayload>,
        gpu: &mut Gpu,
        slot: &ModelSlot,
        state: &MtpSpecState,
        prompt_tokens: &[u32],
        logits: &[f32],
        head_settled: bool,
    ) {
        if let Some(existing) = payload.as_mut() {
            existing.seed = None;
        }
        if prompt_tokens.is_empty() || slot.kv_cache.compact_offset != 0 {
            return;
        }
        let started = std::time::Instant::now();
        if payload.is_none() {
            if !checkpoint_fits_admitted_context(gpu, &slot.kv_cache, &slot.dn_state) {
                return;
            }
            match DeltaNetSnapshot::new_for(gpu, &slot.dn_state) {
                Ok(snapshot) => {
                    *payload = Some(PromptEndPayload {
                        snapshot,
                        seed: None,
                    })
                }
                Err(e) => {
                    eprintln!(
                        "qwen35 MTP prompt-end capture skipped: snapshot allocation failed: {e}"
                    );
                    return;
                }
            }
        }
        let payload = payload.as_mut().expect("allocated above");
        let prev_hidden = match gpu.download_f32(&state.prev_hidden) {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "qwen35 MTP prompt-end capture skipped: prev_hidden download failed: {e}"
                );
                return;
            }
        };
        if let Err(e) = payload.snapshot.save_from(&slot.dn_state, gpu) {
            eprintln!("qwen35 MTP prompt-end capture skipped: snapshot save failed: {e}");
            return;
        }
        if prompt_end_trace() {
            eprintln!(
                "[qwen-cache mtp prompt-end capture] len={} snapshot_bytes={} logits_bytes={} hidden_bytes={} head_settled={} capture_ms={:.2}",
                prompt_tokens.len(),
                payload.snapshot.bytes_len(),
                logits.len() * 4,
                prev_hidden.len() * 4,
                head_settled,
                started.elapsed().as_secs_f64() * 1e3,
            );
        }
        payload.seed = Some(PromptEndSeed {
            len: prompt_tokens.len(),
            fingerprint: prefix_fingerprint(prompt_tokens),
            logits: logits.to_vec(),
            prev_hidden,
            head_settled,
        });
    }

    /// Whole-prompt reuse: put the sealed prompt end back and return the first
    /// token through [`seed_first_token`], the same helper a fresh prefill
    /// uses (never a saved token; the request's RNG is untouched until here).
    /// The planner only plans this against an advertised descriptor, so any
    /// mismatch is an invariant failure and fails closed.
    fn restore_prompt_end(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        prompt_tokens: &[u32],
        start_pos: usize,
        abort: &dyn Fn() -> bool,
    ) -> Result<PrefillOutcome, String> {
        if self.owed_fill.is_some() {
            return Err("qwen35 MTP prompt-end restore with head fill still owed".into());
        }
        self.defer_prefill_tail = false;
        self.last_window = None;
        if abort() {
            return Ok(PrefillOutcome::Aborted);
        }
        if Self::tp_target(target).is_some() {
            return Err(
                "qwen35 MTP prompt-end restore is not available on a dense-TP target".into(),
            );
        }
        let slot = Self::slot(target)?;
        self.ensure_state(gpu, slot)?;
        let state = self.state.as_mut().expect("ensure_state set it");
        let payload = self
            .prompt_end
            .as_ref()
            .ok_or("qwen35 MTP prompt-end restore without a resident payload")?;
        let seed = payload
            .seed
            .as_ref()
            .filter(|seed| seed.head_settled)
            .ok_or("qwen35 MTP prompt-end payload is stale or incomplete")?;
        if seed.len != start_pos
            || start_pos != prompt_tokens.len()
            || seed.fingerprint != prefix_fingerprint(prompt_tokens)
        {
            return Err(format!(
                "qwen35 MTP prompt-end payload (len {}) does not describe this prompt (len {}, start {start_pos})",
                seed.len,
                prompt_tokens.len()
            ));
        }
        if slot.kv_cache.compact_offset != 0 {
            return Err("qwen35 MTP prompt-end restore on a compacted KV cache".into());
        }
        let logits_bytes = f32_bytes(&seed.logits);
        let hidden_bytes = f32_bytes(&seed.prev_hidden);
        if logits_bytes.len() > slot.scratch.logits.buf.size()
            || hidden_bytes.len() > state.prev_hidden.buf.size()
        {
            return Err("qwen35 MTP prompt-end seed rows do not fit this slot".into());
        }
        let started = std::time::Instant::now();
        payload
            .snapshot
            .restore_to(&mut slot.dn_state, gpu)
            .map_err(|e| format!("qwen35 MTP prompt-end DeltaNet restore: {e}"))?;
        gpu.hip
            .memcpy_htod(&slot.scratch.logits.buf, logits_bytes)
            .map_err(|e| format!("qwen35 MTP prompt-end logits restore: {e}"))?;
        gpu.hip
            .memcpy_htod(&state.prev_hidden.buf, hidden_bytes)
            .map_err(|e| format!("qwen35 MTP prompt-end prev_hidden restore: {e}"))?;
        state.prev_hidden_pos = Some(start_pos - 1);
        // The user-turn boundary was snapshotted by the capturing prefill and
        // is still pinned in the ring; it is one-shot, so it never leaks into
        // a later prefill.
        self.turn_boundary = None;
        let first_token = seed_first_token(&seed.logits, state);
        if prompt_end_trace() {
            eprintln!(
                "[qwen-cache mtp prompt-end restore] len={start_pos} restore_ms={:.2}",
                started.elapsed().as_secs_f64() * 1e3,
            );
        }
        self.begin_ngram_request(prompt_tokens);
        Ok(PrefillOutcome::Ready { first_token })
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
        // Whole-prompt reuse (no rows to prefill): restore the prompt-end
        // payload instead of running a prefill.
        if cache_hit && fill_tokens.is_empty() {
            return self.restore_prompt_end(gpu, target, prompt_tokens, start_pos, abort);
        }
        // A new prefill supersedes any owed fill of a dead request (cold), or
        // continues the trunk state it belongs to (warm): settle first.
        if cache_hit {
            self.settle_owed_fill(gpu, target)?;
        } else {
            self.discard_owed_fill(gpu);
        }
        let defer_tail = std::mem::take(&mut self.defer_prefill_tail);
        // Rows below the prompt-end payload's length are rewritten by anything
        // but a pure extension, so the payload stops describing the live head
        // and target KV prefix. Its snapshot buffers stay allocated for reuse.
        if !cache_hit || self.prompt_end_len().is_some_and(|len| start_pos < len) {
            self.invalidate_prompt_end();
        }
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
        let prompt_route = self.prompt_route;
        let checkpoint_interval = self.checkpoint_interval;
        let checkpoint_cap = self.checkpoint_cap;
        // This request's user-turn boundary: the prefill snapshots at its newest
        // chunk end at or before it; that snapshot stays pinned through the
        // following tool-result requests (`pinned_boundary`) until the next user
        // turn. One-shot: a realign re-prefill inside the request does not see it.
        let mut turn = TurnCheckpoint {
            boundary: self.turn_boundary.take(),
            pin: self.pinned_boundary,
        };
        let end_pos = start_pos + fill_tokens.len();
        let ring_had_end = checkpoints.iter().any(|(position, _)| *position == end_pos);
        let prefill = prefill_trunk_and_mtp_cache_with_boundary_deferred(
            gpu,
            slot,
            &self.head,
            state,
            fill_tokens,
            start_pos,
            prompt_route,
            turn.boundary,
            defer_tail,
            |gpu, dn_state, kv, position, at_turn| {
                // The ring covers prompt rows only: `[0..position]` of the
                // canonical prompt this request resumed or cold-seeded.
                let prefix = (prompt_tokens.get(..position).unwrap_or(&[]), &[][..]);
                if at_turn {
                    take_dn_boundary_checkpoint(
                        &mut checkpoints,
                        dn_state,
                        kv,
                        gpu,
                        position,
                        checkpoint_cap,
                        prefix,
                    );
                    if checkpoints.iter().any(|(p, _)| *p == position) {
                        turn.pin = Some(position);
                    }
                } else {
                    take_turn_checkpoint(
                        &mut checkpoints,
                        dn_state,
                        kv,
                        gpu,
                        position,
                        checkpoint_interval,
                        checkpoint_cap,
                        turn,
                        prefix,
                    );
                }
                Ok(())
            },
        );
        self.checkpoints = checkpoints;
        self.pinned_boundary = turn.pin;
        let (_timings, owed) = prefill.map_err(|e| format!("mtp prefill: {e}"))?;
        if let Some(fill) = owed {
            let ring_added = !ring_had_end
                && self
                    .checkpoints
                    .iter()
                    .any(|(position, _)| *position == fill.committed_pos());
            self.owed_fill = Some(OwedFill { fill, ring_added });
        }
        if abort() {
            self.discard_owed_fill(gpu);
            return Ok(PrefillOutcome::Aborted);
        }

        // Seed from the trunk logits at the last prefilled position
        // (`prefill_trunk_and_mtp_cache` leaves them in scratch.logits).
        // Greedy = argmax; temp>0 draws from the state's continuous RNG/config.
        let logits = gpu
            .download_f32(&slot.scratch.logits)
            .map_err(|e| format!("download seed logits: {e}"))?;
        // The sealed prompt end (DeltaNet image, this row, `prev_hidden`) is
        // captured BEFORE the first token is sampled, so a restored retry draws
        // its first token from identical bytes through the same helper below.
        if start_pos + fill_tokens.len() == prompt_tokens.len() {
            Self::capture_prompt_end(
                &mut self.prompt_end,
                gpu,
                &*slot,
                &*state,
                prompt_tokens,
                &logits,
                self.owed_fill.is_none(),
            );
        }
        let first_token = seed_first_token(&logits, state);

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
        self.settle_owed_fill(gpu, target)?;
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
        self.settle_owed_fill(gpu, target)?;
        if self.prompt_end_len().is_some_and(|len| start_pos < len) {
            self.invalidate_prompt_end();
        }
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
        self.settle_owed_fill(gpu, target)?;
        if self.prompt_end_len().is_some_and(|len| window_start < len) {
            self.invalidate_prompt_end();
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
        self.discard_owed_fill(gpu);
        self.last_window = None;
        self.tp_reset_pending = self.tp.is_some();
        self.stats = MtpRequestStats::default();
        if let Some(ctx) = self.ngram.as_mut() {
            ctx.reset_request();
        }
        self.clear_checkpoints(gpu);
        self.free_prompt_end(gpu);
        if let Some(state) = self.state.as_mut() {
            state
                .reset(gpu)
                .map_err(|e| format!("qwen35-mtp drafter reset: {e}"))?;
        }
        Ok(())
    }

    fn mtp_reset_for_realign(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.discard_owed_fill(gpu);
        self.last_window = None;
        self.tp_reset_pending = self.tp.is_some();
        self.clear_checkpoints(gpu);
        self.free_prompt_end(gpu);
        if let Some(state) = self.state.as_mut() {
            state
                .reset(gpu)
                .map_err(|e| format!("qwen35-mtp drafter realign reset: {e}"))?;
        }
        Ok(())
    }

    fn mtp_set_turn_boundary(&mut self, boundary: Option<usize>) {
        self.turn_boundary = boundary;
    }

    fn mtp_defer_prefill_tail(&mut self, on: bool) {
        self.defer_prefill_tail = on;
    }

    fn mtp_finish_prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
    ) -> Result<(), String> {
        self.settle_owed_fill(gpu, target)
    }

    // While head fill is owed, a checkpoint at the prompt end exists in the ring
    // (its DeltaNet snapshot had to precede the seal) but must not be advertised:
    // its head frontier is not valid until the fill lands.
    fn mtp_checkpoint_positions(&self) -> Vec<usize> {
        let withheld = self.withheld_checkpoint();
        self.checkpoints
            .iter()
            .map(|(position, _)| *position)
            .filter(|position| Some(*position) != withheld)
            .collect()
    }

    fn mtp_checkpoint_prefix_hashes(&self) -> Vec<(usize, u64)> {
        let withheld = self.withheld_checkpoint();
        self.checkpoints
            .iter()
            .filter(|(position, _)| Some(*position) != withheld)
            .map(|(position, snapshot)| (*position, snapshot.prefix_hash))
            .collect()
    }

    // Advertised only once the owed head fill has landed: before that the head
    // KV does not yet cover `[0, len)`, so the payload is not a complete seed.
    fn mtp_prompt_end_checkpoint(&self) -> Option<(usize, u64)> {
        self.prompt_end
            .as_ref()?
            .seed
            .as_ref()
            .filter(|seed| seed.head_settled)
            .map(|seed| (seed.len, seed.fingerprint))
    }

    fn mtp_rewind_to(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
    ) -> Result<usize, String> {
        self.settle_owed_fill(gpu, target)?;
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
        this.discard_owed_fill(gpu);
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
        this.free_prompt_end(gpu);
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
            this.discard_owed_fill(dev);
            this.clear_checkpoints(dev);
            this.free_prompt_end(dev);
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
