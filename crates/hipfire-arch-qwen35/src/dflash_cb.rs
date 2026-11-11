// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched DFlash lanes of the VMM continuous-batching store.
//!
//! [`DflashVmmLaneState`] is the persistent private draft/verify state of one
//! continuous-batching lane: the singleton [`DflashState`] owners plus the
//! request-local cursors the singleton [`DflashSpeculator`] keeps beside it.
//! It is built by consuming a [`DflashLaneSnapshot`] (`take_vmm_lane`, which
//! transfers every owner and cursor losslessly) or cold, from a lane
//! `DflashState` (`new_dflash_lane_state` with `DflashOwnerKind::VmmLane`)
//! after the lane prompt prefill.
//!
//! A VMM lane owns NO FWHT rotation scratch (`mq_x_rot`/`mq_x_rot_f16`) and
//! no `VerifyScratch::prefill_batch` (the batched verify runs on the shared
//! trunk scratch; the lane never runs a target forward through its own
//! scratch). The rotation buffers are a per-GEMM-call temporary of the draft
//! forward, identical in size for every draft scratch, so the engine's ONE
//! shared draft scratch ([`DflashCbScratch::reserve_draft`]) owns them: the
//! batched draft borrows them directly (`draft_forward_lanes`), and every
//! other lane-draft path (the singleton-call fallback, the lane prompt
//! prefill backfill) lends them to the lane for the duration of the call
//! ([`DflashCbScratch::lend_mq`] / [`DflashCbScratch::reclaim_mq`]). All
//! calls run on the single stream, so the borrowed temporary is never live
//! across two calls. A lane promoted from the singleton arrives with the
//! singleton's own buffers, which `DflashState::shed_vmm_lane_scratch` frees.
//!
//! The in-crate halves of the shared DFlash verify:
//!
//! * [`dflash_lane_draft`] — the singleton greedy draft call of
//!   `spec_step_dflash` (`draft_dflash_block_rank`) on a lane's own draft
//!   state; returns `[seed, candidates..]`.
//! * [`dflash_cb_head_argmax`] — the singleton DFlash verify head
//!   (`dflash_enqueue_verify_lm_head_argmax` + one packed i32 D2H) over the
//!   first `rows` rows of a shared [`VerifyScratch`], for greedy picks only.
//!   Admitted sampled CB chunks share the explicit exact F16 head; otherwise
//!   each sampled lane retains its private singleton-b Product dispatch.
//! * [`dflash_cb_verify`] — whole lanes packed into trunk chunks of at most
//!   the effective cap (`<= 63` rows; up to 128 with
//!   `HIPFIRE_CB_VERIFY_CHUNK128` (default on for gfx1201; `=0` restores 63) on
//!   an admitted dense MQ4G256V2 gfx1201 target) (`forward_prefill_batch_multi`, ChainVerify fusion, each lane's
//!   own hidden ring and tape at offset 0). Greedy lanes share a packed head;
//!   sampled lanes retain actual selector q and private FastTargetRows until
//!   their shared singleton accept/commit consumer runs.

use crate::dflash_spec::{
    DflashCheckpointPolicy, DflashLaneSnapshot, DflashState, DflashWindowMark,
};
use crate::qwen35::prefill::fp8_verify_open;
use crate::qwen35::prefill::multi::{
    cb_dn_save_free_enabled, dflash_iu4_verify_admitted, forward_prefill_batch_multi,
    lane_gdn_batch_ready, lane_gdn_varrows_enabled, multi_chunk_pack_cap, multi_chunk_scratch_rows,
    pack_whole_lanes, MultiChunkRequest, MultiChunkScratch, MULTI_CHUNK_MAX_LANE_ROWS,
    MULTI_CHUNK_PRODUCT_MAX_ROWS,
};
use crate::qwen35::{self, DeltaNetState, DflashFusionCtx};
use crate::qwen35::{Qwen35Config, Qwen35Weights};
use crate::speculative::{
    dflash_device_accept_eligible, dflash_device_accept_enabled, dflash_download_verify_argmax,
    dflash_draft_batch_eligible, dflash_draft_batch_max_rows, dflash_enqueue_verify_lm_head_argmax,
    dflash_sampled_accept_device, dflash_sampled_selector_draft_into, dflash_sparse_target_enabled,
    dflash_target_rows_into, draft_dflash_block_rank, draft_dflash_blocks_batched,
    draft_dflash_sampled_blocks_batched, DeltaNetSnapshot, DflashCbDraft, DflashDraftLane,
    DflashSampleCfg, DflashSampledDraft, DflashSampledDraftLane, FastTargetRows,
    SampledWindowDecision, TargetRowsScratch, VerifyScratch,
};
use hip_bridge::{HipError, HipResult};
use hipfire_dispatch::pipeline::batched::{verify_dense_math, DenseBatchMath};
use hipfire_runtime::dflash::DflashScratch;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::spec::{request_rng_state, SpecRequestConfig};
use rdna_compute::Gpu;
use std::ops::Range;

static CB_SAMPLED_TARGET_BATCH: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_CB_SAMPLED_TARGET_BATCH")
        .ok()
        .as_deref()
        != Some("0")
});

static CB_SAMPLED_PACKED_HEAD: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_CB_SAMPLED_PACKED_HEAD")
        .ok()
        .as_deref()
        != Some("0")
});

static CB_DFLASH_DRAFT_BATCH_MASTER: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_CB_DFLASH_DRAFT_BATCH")
        .ok()
        .as_deref()
        != Some("0")
});

static CB_SAMPLED_DRAFT_BATCH: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_CB_SAMPLED_DRAFT_BATCH")
        .ok()
        .as_deref()
        != Some("0")
});

/// The singleton's verify block for remaining output budget `max_emit`
/// (`E > 0`): `B = min(block_size, max(E, 2))`. Even `E = 1` verifies two rows
/// (the accept clamp `max_accept = E - 1` then keeps zero drafts).
pub fn dflash_block_for_emit(block_size: usize, max_emit: usize) -> usize {
    block_size.min(max_emit.max(2))
}

/// Private DFlash state of one lane (see the module docs).
pub struct DflashVmmLaneState {
    /// The singleton owners: draft scratch/weights, hidden ring, verify
    /// scratch, target DeltaNet snapshot, GDN tape.
    pub df: DflashState,
    /// Committed target position at capture; the store keeps it equal to the
    /// request's position after every commit.
    pub position: usize,
    pub rng_state: u64,
    pub sample_temp: f32,
    pub sample_top_p: f32,
    pub sample_top_k: usize,
    pub sample_cactus: f32,
    /// Divergent-render checkpoints `(position, snapshot)`.
    pub checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    pub policy: DflashCheckpointPolicy,
    /// Pre-window mark of the last committed window (terminal-prefix repair).
    pub last_window: Option<DflashWindowMark>,
    /// Target picks of this lane's verified window (`B` ids), kept from the
    /// shared head until the greedy accept consumer runs.
    pub picks: Vec<u32>,
    /// Request-owned normalized target rows, consumed before chunk scratch reuse.
    pub sampled_target: FastTargetRows,
    /// Verify-time sampled decision; consumed by commit without another RNG draw.
    pub device_decision: Option<SampledWindowDecision>,
    /// Shared singleton first-token / acceptance workspace.
    pub sample_scratch: Vec<f32>,
    /// Actual selector proposal and q, retained beside the planned verify.
    pub sampled_draft: Option<DflashSampledDraft>,
    /// Remaining output budget `E` installed by the driver for the next
    /// window (`0` = not installed; a window needs `E >= 1`).
    pub max_emit: usize,
    /// `picks` hold the verify outcome of the lane's planned window.
    pub verified: bool,
    /// Tokens the last committed window emitted (`committed[1..]`).
    pub last_emit: Vec<u32>,
    /// The request's history length before the last window's emit was
    /// appended (terminal-prefix repair truncates back to it).
    pub last_history_len: usize,
}

// SAFETY: a lane lives inside the store (`Qwen35Bundle::vmm_store`, which must
// be `Send` to ride the model across threads) and, like every GPU owner there
// (`DeviceBuffer` is `Send`), is used only by the thread holding the model.
// The one non-`Send` part of a `DflashState` is its retained-PM4 route, which a
// lane keeps explicitly disabled; the shared draft weights `Arc` is a counted
// reference to the one resident set, read only through GPU calls under that
// same single owner.
unsafe impl Send for DflashVmmLaneState {}

impl DflashVmmLaneState {
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        df: DflashState,
        position: usize,
        rng_state: u64,
        sample_temp: f32,
        sample_top_p: f32,
        sample_top_k: usize,
        sample_cactus: f32,
        checkpoints: Vec<(usize, DeltaNetSnapshot)>,
        policy: DflashCheckpointPolicy,
        last_window: Option<DflashWindowMark>,
    ) -> Self {
        let block = df.block_size;
        Self {
            df,
            position,
            rng_state,
            sample_temp,
            sample_top_p,
            sample_top_k,
            sample_cactus,
            checkpoints,
            policy,
            last_window,
            picks: Vec::with_capacity(block),
            sampled_target: FastTargetRows::default(),
            device_decision: None,
            sample_scratch: Vec::new(),
            sampled_draft: None,
            max_emit: 0,
            verified: false,
            last_emit: Vec::with_capacity(block),
            last_history_len: 0,
        }
    }

    /// Consume `snapshot`, moving every owner and cursor into the lane.
    /// Read `snapshot.rows()` before calling.
    pub fn from_snapshot(snapshot: DflashLaneSnapshot) -> Self {
        let DflashLaneSnapshot {
            df,
            position,
            rng_state,
            sample_temp,
            sample_top_p,
            sample_top_k,
            sample_cactus,
            checkpoints,
            policy,
            last_window,
        } = snapshot;
        Self::assemble(
            df,
            position,
            rng_state,
            sample_temp,
            sample_top_p,
            sample_top_k,
            sample_cactus,
            checkpoints,
            policy,
            last_window,
        )
    }

    /// A lane that just finished its cold prompt prefill (`position` prompt
    /// rows committed): request sampling configured exactly as
    /// `DflashSpeculator::configure_request`, no completed window yet.
    pub fn cold(
        df: DflashState,
        checkpoints: Vec<(usize, DeltaNetSnapshot)>,
        policy: DflashCheckpointPolicy,
        position: usize,
        request: &SpecRequestConfig,
    ) -> Self {
        Self::assemble(
            df,
            position,
            request_rng_state(request.rng_seed),
            request.temp,
            request.top_p,
            request.ar_candidate_cap(),
            request.cactus_delta,
            checkpoints,
            policy,
            None,
        )
    }

    pub fn sample_cfg(&self) -> DflashSampleCfg {
        DflashSampleCfg {
            temp: self.sample_temp,
            top_p: self.sample_top_p,
            top_k: self.sample_top_k,
            cactus_delta: self.sample_cactus,
        }
    }

    /// Free every owned GPU object exactly once (shared draft weights go
    /// through `DflashState::free_gpu`'s final-owner release).
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let DflashVmmLaneState {
            df,
            position: _,
            rng_state: _,
            sample_temp: _,
            sample_top_p: _,
            sample_top_k: _,
            sample_cactus: _,
            checkpoints,
            policy: _,
            last_window: _,
            picks: _,
            sampled_target: _,
            device_decision: _,
            sample_scratch: _,
            sampled_draft: _,
            max_emit: _,
            verified: _,
            last_emit: _,
            last_history_len: _,
        } = self;
        df.free_gpu(gpu);
        for (_, snap) in checkpoints {
            snap.free_gpu(gpu);
        }
    }
}

/// Greedy chain draft of one lane window of `b` rows at `position` with
/// pending `seed`: exactly the singleton `spec_step_dflash` greedy fast-path
/// call (`ctx_slice=None`, dense target so no draft FFN graph). Mutates the
/// lane's draft scratch (projection watermarks, draft K/V) as the singleton
/// draft does. Returns `[seed, candidates..]` (`b` tokens).
pub fn dflash_lane_draft(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    compact_offset: i32,
    df: &mut DflashState,
    position: usize,
    seed: u32,
    b: usize,
) -> HipResult<Vec<u32>> {
    if config.num_experts > 0 {
        return Err(hip_bridge::HipError::new(
            0,
            "dflash_lane_draft: MoE targets use the draft FFN graph route (not reproduced)",
        ));
    }
    if b < 2 {
        return Err(hip_bridge::HipError::new(
            0,
            "dflash_lane_draft: block size must be >= 2",
        ));
    }
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
    let mut block: Vec<u32> = vec![df.draft_config.mask_token_id; b];
    block[0] = seed;
    draft_dflash_block_rank(
        gpu,
        weights,
        &df.draft_weights,
        &df.draft_config,
        &mut df.draft_scratch,
        &df.verify_scratch,
        &[],
        &block,
        position,
        compact_offset,
        None,
        false,
        seed,
        config.vocab_size,
    )
}

/// Shared verify head: the singleton DFlash dispatcher (rotate + batched
/// lm-head + GPU argmax) over `vs.final_hidden[0..rows]`, then one packed i32
/// D2H. Logits land in `vs.logits[0..rows]`, picks in `vs.argmax[0..rows]`.
/// A head over 64+ rows (only a wide chunk reaches it) uses the exact
/// singleton-WMMA lm-head, whose rows equal the singleton head's; below 64
/// rows the product head is the singleton's own, except inside the PM FP8
/// verify scope ([`verify_dense_math`]) where every row count takes the
/// `*VerifyExact` head (FP8-routed). Call it inside the same scope as the
/// trunk forward it follows. Admitted sampled packing instead explicitly
/// requests SingletonWmma at every row count, never aggregate Product.
pub fn dflash_cb_head_argmax(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    vs: &VerifyScratch,
    rows: usize,
) -> HipResult<Vec<u32>> {
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
    let final_hidden = vs.final_hidden.sub_offset(0, rows * config.dim);
    let math = if rows > MULTI_CHUNK_PRODUCT_MAX_ROWS {
        DenseBatchMath::SingletonWmma
    } else {
        verify_dense_math(gpu)
    };
    dflash_enqueue_verify_lm_head_argmax(
        gpu,
        &weights.output,
        &final_hidden,
        vs,
        rows,
        config.vocab_size,
        math,
    )?;
    dflash_download_verify_argmax(gpu, vs, rows)
}

/// Packing sampled logits is legal only on the singleton's normal F16
/// residual chain. Refusal leaves its original Product-b route untouched.
fn cb_sampled_packed_head_admitted(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    lanes: &[DflashCbVerifyLane<'_>],
    rows: usize,
) -> bool {
    let head = &weights.output;
    *CB_SAMPLED_PACKED_HEAD
        && lanes.len() > 1
        && lanes.iter().any(|l| l.state.sample_temp > 1e-6)
        && gpu.mq4_verify_singleton_chain()
        && head.gpu_dtype == rdna_compute::DType::MQ4G256V2
        && head.k > 0
        && head.k % 256 == 0
        && head.k == config.dim
        && head.m == config.vocab_size
        && head.m > 0
        && (1..=128).contains(&rows)
        && !gpu.fp8_verify_scope
        && !gpu.iu4_verify_scope
        && !gpu.graphs.capture_mode
        && !gpu.replay.is_recording()
        && !gpu.replay.is_g0_observing()
        && lanes.iter().all(|l| {
            let b = l.draft.verify_tokens.len();
            (2..=16).contains(&b) && !gpu.a8_prefill_active(b, head.k)
        })
}

/// Both admitted sampled head sources reproduce the singleton Product-b rows.
/// `device_ok` is `Gpu::dflash_device_accept_supported()`: the production
/// kernel is a gfx1201 builder module with no hipcc fallback.
fn cb_sampled_device_accept_eligible(lane: &DflashCbVerifyLane<'_>, device_ok: bool) -> bool {
    lane.state.sample_temp > 1e-6
        && lane.state.sampled_draft.as_ref().is_some_and(|receipt| {
            dflash_device_accept_eligible(
                dflash_device_accept_enabled() && device_ok,
                &lane.state.sample_cfg(),
                receipt,
                lane.draft.verify_tokens.len(),
            )
        })
}

/// Number of consecutive lanes at the head of `lanes` that can share ONE
/// sampled target-rows pass: sampled lanes whose `DflashSampleCfg`
/// (temperature, top_p, top_k, CACTUS delta, compared bitwise) is identical
/// and for which the sparse representation is active
/// ([`dflash_sparse_target_enabled`]; CACTUS and the sparse kill switch stay
/// on the per-lane dense route). `1` (a singleton) never batches.
fn cb_sampled_target_run_len(lanes: &[DflashCbVerifyLane<'_>], device_ok: bool) -> usize {
    let Some(first) = lanes.first() else {
        return 0;
    };
    if first.state.sample_temp <= 1e-6 || cb_sampled_device_accept_eligible(first, device_ok) {
        return 1;
    }
    let c0 = first.state.sample_cfg();
    if !dflash_sparse_target_enabled(&c0) {
        return 1;
    }
    let same = |c: &DflashSampleCfg| {
        c.temp.to_bits() == c0.temp.to_bits()
            && c.top_p.to_bits() == c0.top_p.to_bits()
            && c.top_k == c0.top_k
            && c.cactus_delta.to_bits() == c0.cactus_delta.to_bits()
    };
    lanes
        .iter()
        .take_while(|l| {
            l.state.sample_temp > 1e-6
                && !cb_sampled_device_accept_eligible(l, device_ok)
                && same(&l.state.sample_cfg())
        })
        .count()
}

/// Shared verify buffers of one DFlash CB engine: the trunk row scratch, ONE
/// head scratch (`final_hidden`/`logits`/`rot`/`argmax`, `draft_rows` rows,
/// `hidden_k = dim.next_power_of_two()` like a singleton lane's), and the
/// reusable chunk tables. Allocated once; a cycle allocates no GPU memory.
/// `max_rows` is the EFFECTIVE cap (63, or up to 128 on the wide verify route).
pub struct DflashCbScratch {
    max_rows: usize,
    /// Head-scratch rows (`>= max_rows`): see [`Self::draft_rows`].
    draft_rows: usize,
    trunk: MultiChunkScratch,
    head: VerifyScratch,
    rows: Vec<usize>,
    ranges: Vec<Range<usize>>,
    /// Diagnostic: also copy each lane's head rows (post-norm hidden, logits,
    /// argmax) into its own `df.verify_scratch` (the oracle's proof arrays).
    /// Greedy production leaves this off.
    pub keep_lane_rows: bool,
    /// Row scratch of the batched draft forward, allocated by
    /// [`Self::reserve_draft`] (or on the first draft call that needs it): the
    /// shared draft activations (lane draft state stays in each lane) and the
    /// ONE set of FWHT rotation buffers every VMM lane borrows
    /// ([`Self::lend_mq`]).
    draft: Option<DflashScratch>,
    /// Fewest lanes a draft chunk needs to take the batched forward (a lone
    /// lane runs the singleton call; the oracle sets 1 to prove the batched
    /// forward at one lane).
    pub draft_min_lanes: usize,
    /// `(lanes, chunks)` the last [`dflash_cb_draft`] call drafted through the
    /// batched forward (the rest ran the singleton call); evidence for the
    /// oracle that the batched path, not its fallback, produced the drafts.
    pub draft_stats: (usize, usize),
    /// Persistent grow-on-demand device buffers and run-level host rows of the
    /// batched sampled target path ([`dflash_cb_verify`]); freed in
    /// [`Self::free_gpu`].
    target_rows: TargetRowsScratch,
}

impl DflashCbScratch {
    /// `max_rows` is the planned rows; the scratch allocates this target's
    /// effective cap ([`multi_chunk_scratch_rows`]: `2..=63`, or up to 128
    /// only when the wide verify route is admitted for `weights`/`config`).
    /// The scratch stores that cap ([`Self::max_rows`]). The head scratch holds
    /// `max(cap, min(planned, dflash_draft_batch_max_rows(gpu)))` rows so a
    /// 64-row batched draft chunk fits even when the verify cap is 63
    /// ([`Self::draft_rows`]); with the 48-row draft cap this equals the cap.
    pub fn new(
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        planned_rows: usize,
    ) -> HipResult<Self> {
        let max_rows = multi_chunk_scratch_rows(gpu, weights, config, planned_rows);
        let draft_rows = max_rows.max(planned_rows.min(dflash_draft_batch_max_rows(gpu)));
        let trunk = MultiChunkScratch::new(gpu, config, max_rows)?;
        let head = match VerifyScratch::new(
            gpu,
            draft_rows,
            config.dim,
            config.vocab_size,
            config.dim.next_power_of_two(),
        ) {
            Ok(h) => h,
            Err(e) => {
                let _ = trunk.free_gpu(gpu);
                return Err(e);
            }
        };
        Ok(Self {
            max_rows,
            draft_rows,
            trunk,
            head,
            rows: Vec::with_capacity(max_rows),
            ranges: Vec::with_capacity(max_rows),
            keep_lane_rows: false,
            draft: None,
            draft_min_lanes: 2,
            target_rows: TargetRowsScratch::default(),
            draft_stats: (0, 0),
        })
    }

    /// Allocate the batched-draft row scratch now (idempotent), so a serving
    /// engine never allocates inside a step; `dflash_cb_draft` otherwise
    /// allocates it on the first batched chunk. The scratch also OWNS the
    /// FWHT rotation buffers every lane borrows ([`Self::lend_mq`]).
    pub fn reserve_draft(
        &mut self,
        gpu: &mut Gpu,
        draft_config: &hipfire_runtime::dflash::DflashConfig,
    ) -> HipResult<()> {
        if self.draft.is_none() {
            let mut ds = DflashScratch::new_with_mq(
                gpu,
                draft_config,
                dflash_draft_batch_max_rows(gpu),
                1,
                true,
            )?;
            if let Err(e) = ds.enable_lane_workspace(gpu, draft_config) {
                ds.free_gpu(gpu);
                return Err(e);
            }
            self.draft = Some(ds);
        }
        Ok(())
    }

    /// Lend the shared FWHT rotation buffers to `lane` (a lane draft scratch
    /// allocated without them) so a lane-local draft call can run its
    /// `gemm_dispatch`es. `Ok(false)`: the lane owns its own buffers (a
    /// singleton-origin scratch), nothing moved. `Ok(true)`: both buffers
    /// moved from the shared scratch into the lane; the caller MUST call
    /// [`Self::reclaim_mq`] with the returned flag on every path (success and
    /// error) before the lane or the engine is used again. `Err` when no
    /// shared draft scratch (or no rotation buffers in it) exists.
    ///
    /// Exact: the buffers have the same element count as a lane's own would
    /// (`MQ_X_ROT_CHUNK_ROWS × max(inter, q_dim, num_extract × hidden)`), so
    /// `gemm_dispatch` chunks identically, and each call rotates into them
    /// before its GEMM reads them (a per-call temporary, single stream).
    pub fn lend_mq(&mut self, lane: &mut DflashScratch) -> HipResult<bool> {
        if lane.mq_x_rot.is_some() {
            return Ok(false);
        }
        let shared = self.draft.as_mut().ok_or_else(|| {
            HipError::new(
                0,
                "lend_mq: no shared draft scratch to lend rotation buffers from",
            )
        })?;
        if shared.mq_x_rot.is_none() || shared.mq_x_rot_f16.is_none() {
            return Err(HipError::new(
                0,
                "lend_mq: the shared draft scratch has no FWHT rotation buffers",
            ));
        }
        lane.mq_x_rot = shared.mq_x_rot.take();
        lane.mq_x_rot_f16 = shared.mq_x_rot_f16.take();
        Ok(true)
    }

    /// Return buffers lent by [`Self::lend_mq`] (`lent` is its `Ok` value) to
    /// the shared scratch. No-op when nothing was lent.
    pub fn reclaim_mq(&mut self, lane: &mut DflashScratch, lent: bool) {
        if !lent {
            return;
        }
        if let Some(shared) = self.draft.as_mut() {
            shared.mq_x_rot = lane.mq_x_rot.take();
            shared.mq_x_rot_f16 = lane.mq_x_rot_f16.take();
        }
    }

    /// Rows one trunk chunk can hold (the effective cap: `<= 63`, or up to
    /// 128 on the wide verify route).
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    /// Rows one batched draft chunk may hold: the head scratch's row count.
    /// `max_rows` unless the planned rows exceed it and the 64-row draft
    /// chunk ([`dflash_draft_batch_max_rows`]) is admitted, in which case
    /// the head (and so this cap) is widened to hold it.
    pub fn draft_rows(&self) -> usize {
        self.draft_rows
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> HipResult<()> {
        let DflashCbScratch {
            max_rows: _,
            draft_rows: _,
            trunk,
            head,
            rows: _,
            ranges: _,
            keep_lane_rows: _,
            draft,
            draft_min_lanes: _,
            draft_stats: _,
            target_rows,
        } = self;
        target_rows.free_gpu(gpu);
        if let Some(d) = draft {
            d.free_gpu(gpu);
        }
        let r = trunk.free_gpu(gpu);
        head.free_gpu(gpu);
        r
    }
}

/// One lane of [`dflash_cb_draft`]: its private state and the window to draft
/// (`b` rows at `position` with pending `seed`).
pub struct DflashCbDraftLane<'a> {
    pub state: &'a mut DflashVmmLaneState,
    pub position: usize,
    pub seed: u32,
    pub b: usize,
    pub compact_offset: i32,
}

/// Whether one sampled lane may join a batched sampled-draft run led by
/// `first`'s shared draft assets (host-only; touches no device state).
fn cb_sampled_draft_lane_eligible(
    l: &DflashCbDraftLane<'_>,
    first: &DflashCbDraftLane<'_>,
    cap: usize,
) -> bool {
    let (df, f) = (&l.state.df, &first.state.df);
    l.state.sample_temp > 1e-6
        && l.state.sampled_draft.is_some()
        && l.b >= 3
        && l.b <= df.block_size
        && l.b <= cap
        && l.b <= df.draft_scratch.max_block_size
        && df.draft_weights.has_candidate_selector()
        && std::sync::Arc::ptr_eq(&df.draft_weights, &f.draft_weights)
        && df.block_size == f.block_size
        && df.draft_config.n_layers == f.draft_config.n_layers
        && df.draft_config.hidden == f.draft_config.hidden
        && df.draft_config.intermediate == f.draft_config.intermediate
        && df.draft_config.vocab_size == f.draft_config.vocab_size
        && df.draft_config.block_size == f.draft_config.block_size
        && df.draft_config.mask_token_id == f.draft_config.mask_token_id
}

/// Strict normal-F16 overwrite admission of a sampled run on exact gfx1201:
/// every MQ4G256V2 draft weight and the target head must admit the backend
/// overwrite route at the packed row count (inside the 64-row scope when that
/// is the run size) and at each lane's original `b` and `b - 1` row counts
/// (the singleton forward and Product head classes). The backend predicate
/// excludes FP8/IU4 scopes, A8, LDS-staged, capture/recording/G0 and the
/// disabled-F16/WMMA flags. Decided before any device write.
fn cb_sampled_draft_run_admitted(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    run: &[DflashCbDraftLane<'_>],
    rows: usize,
    cap: usize,
) -> bool {
    use rdna_compute::DType::MQ4G256V2;
    if gpu.arch != "gfx1201" || rows == 0 || rows > cap || weights.output.gpu_dtype != MQ4G256V2 {
        return false;
    }
    let dw = std::sync::Arc::clone(&run[0].state.df.draft_weights);
    let mut ks: Vec<usize> = vec![weights.output.k];
    let mut push = |w: &hipfire_runtime::llama::WeightTensor| {
        if w.gpu_dtype == MQ4G256V2 {
            ks.push(w.k);
        }
    };
    push(&dw.fc);
    for l in &dw.layers {
        for w in [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down] {
            push(w);
        }
        if let Some(w) = l.attn_conv_proj.as_ref() {
            push(w);
        }
        if let Some(w) = l.mlp_conv_proj.as_ref() {
            push(w);
        }
    }
    ks.sort_unstable();
    ks.dedup();
    let prev = gpu.draft_rows64_enter(rows == 64 && cap == 64);
    let ok = ks.iter().all(|&k| {
        gpu.draft_overwrite_gfx12_admitted(k, rows)
            && run.iter().all(|l| {
                gpu.draft_overwrite_gfx12_admitted(k, l.b)
                    && gpu.draft_overwrite_gfx12_admitted(k, l.b - 1)
            })
    });
    gpu.draft_rows64_exit(prev);
    ok
}

/// Draft phase over lanes. With `batched` (and the exact-gfx1201 MQ4 v2
/// eligibility of [`dflash_draft_batch_eligible`]), greedy full-block lanes are
/// packed in order into chunks of at most
/// `min(dflash_draft_batch_max_rows(gpu), cb.draft_rows())` rows (48, or 64
/// on exact gfx1201 where the scoped 64-row draft route is exact); each chunk of at
/// least `cb.draft_min_lanes` lanes is ONE draft-model forward and ONE shared
/// lm-head ([`draft_dflash_blocks_batched`]) with every lane keeping its own
/// context rings, positions and hidden context. Every other lane (budget
/// tails, lone lanes, ineligible machines/drafts, `batched == false`) runs
/// the singleton call [`dflash_lane_draft`]. Sampled lanes always use the shared
/// singleton selector draft unless the batched-draft master switch
/// (`HIPFIRE_CB_DFLASH_DRAFT_BATCH`, default on) and `HIPFIRE_CB_SAMPLED_DRAFT_BATCH`
/// (default on) both stay enabled: then each contiguous run of at least
/// `max(cb.draft_min_lanes, 2)` eligible sampled lanes (positive temperature,
/// `3 <= b <= block_size`, shared draft weights/config/layout, selector and
/// receipt present, normal F16 gfx1201 overwrite admission for every MQ4
/// draft/head weight at the run, lane `b` and `b - 1` row counts) shares one
/// draft forward and head ([`draft_dflash_sampled_blocks_batched`]) while each
/// lane keeps its own selector, RNG and receipt in the original order. An
/// ineligible sampled lane or any greedy lane ends a run; refused or short runs
/// use the singleton selector draft, retaining its genuine q receipt on the lane.
/// Returns each lane's `[seed, candidates..]`.
pub fn dflash_cb_draft(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    cb: &mut DflashCbScratch,
    lanes: &mut [DflashCbDraftLane<'_>],
    batched: bool,
) -> HipResult<Vec<Vec<u32>>> {
    let n = lanes.len();
    let mut out: Vec<Option<Vec<u32>>> = vec![None; n];
    // chunk id per lane (None = singleton draft).
    let mut chunk_of: Vec<Option<usize>> = vec![None; n];
    let mut n_chunks = 0usize;
    if batched
        && n > 0
        && dflash_draft_batch_eligible(gpu, weights, config, &lanes[0].state.df.draft_weights)
    {
        let cap = dflash_draft_batch_max_rows(gpu).min(cb.draft_rows);
        let mut rows = 0usize;
        let mut members: Vec<Vec<usize>> = Vec::new();
        for (i, l) in lanes.iter().enumerate() {
            if l.state.sample_temp > 1e-6 || l.b < 3 || l.b != l.state.df.block_size || l.b > cap {
                continue;
            }
            if members.last().is_none() || rows + l.b > cap {
                members.push(Vec::new());
                rows = 0;
            }
            members.last_mut().unwrap().push(i);
            rows += l.b;
        }
        for m in members {
            if m.len() >= cb.draft_min_lanes.max(1) {
                for &i in &m {
                    chunk_of[i] = Some(n_chunks);
                }
                n_chunks += 1;
            }
        }
    }
    cb.draft_stats = (chunk_of.iter().flatten().count(), n_chunks);
    if n_chunks > 0 && cb.draft.is_none() {
        let cfg = lanes
            .iter()
            .zip(&chunk_of)
            .find(|(_, c)| c.is_some())
            .map(|(l, _)| l.state.df.draft_config.clone());
        if let Some(cfg) = cfg {
            let mut ds =
                DflashScratch::new_with_mq(gpu, &cfg, dflash_draft_batch_max_rows(gpu), 1, true)?;
            if let Err(e) = ds.enable_lane_workspace(gpu, &cfg) {
                ds.free_gpu(gpu);
                return Err(e);
            }
            cb.draft = Some(ds);
        }
    }
    for c in 0..n_chunks {
        let mut dl: Vec<DflashDraftLane<'_>> = Vec::new();
        let mut idx: Vec<usize> = Vec::new();
        for (i, l) in lanes.iter_mut().enumerate() {
            if chunk_of[i] == Some(c) {
                idx.push(i);
                dl.push(DflashDraftLane {
                    df: &mut l.state.df,
                    position: l.position,
                    seed: l.seed,
                    b: l.b,
                    compact_offset: l.compact_offset,
                });
            }
        }
        let DflashCbScratch { draft, head, .. } = &mut *cb;
        let shared = draft
            .as_mut()
            .ok_or_else(|| HipError::new(0, "dflash_cb_draft: no draft batch scratch"))?;
        let toks =
            draft_dflash_blocks_batched(gpu, weights, config.vocab_size, shared, head, &mut dl)?;
        for (i, t) in idx.into_iter().zip(toks) {
            out[i] = Some(t);
        }
    }
    let sampled_batch = batched
        && n > 0
        && *CB_DFLASH_DRAFT_BATCH_MASTER
        && *CB_SAMPLED_DRAFT_BATCH
        && dflash_draft_batch_eligible(gpu, weights, config, &lanes[0].state.df.draft_weights);
    let sampled_cap = dflash_draft_batch_max_rows(gpu).min(cb.draft_rows);
    let sampled_min_lanes = cb.draft_min_lanes.max(2);
    let mut i = 0usize;
    while i < n {
        if out[i].is_some() {
            i += 1;
            continue;
        }
        if sampled_batch {
            // Contiguous run of eligible sampled lanes in the original
            // traversal; any ineligible sampled lane or greedy lane ends it, so
            // the per-lane RNG draw order matches the singleton fallback.
            let mut j = i;
            let mut rows = 0usize;
            while j < n
                && out[j].is_none()
                && rows + lanes[j].b <= sampled_cap
                && cb_sampled_draft_lane_eligible(&lanes[j], &lanes[0], sampled_cap)
            {
                rows += lanes[j].b;
                j += 1;
            }
            if j - i >= sampled_min_lanes
                && cb_sampled_draft_run_admitted(gpu, weights, &lanes[i..j], rows, sampled_cap)
            {
                if cb.draft.is_none() {
                    cb.reserve_draft(gpu, &lanes[i].state.df.draft_config)?;
                }
                let mut dl: Vec<DflashSampledDraftLane<'_>> = Vec::with_capacity(j - i);
                for l in lanes[i..j].iter_mut() {
                    let st = &mut *l.state;
                    let cfg = st.sample_cfg();
                    let Some(receipt) = st.sampled_draft.as_mut() else {
                        return Err(HipError::new(
                            0,
                            "dflash_cb_draft: sampled receipt not reserved",
                        ));
                    };
                    dl.push(DflashSampledDraftLane {
                        draft: DflashDraftLane {
                            df: &mut st.df,
                            position: l.position,
                            seed: l.seed,
                            b: l.b,
                            compact_offset: l.compact_offset,
                        },
                        cfg,
                        rng: &mut st.rng_state,
                        out: receipt,
                    });
                }
                let DflashCbScratch { draft, head, .. } = &mut *cb;
                let shared = draft
                    .as_mut()
                    .ok_or_else(|| HipError::new(0, "dflash_cb_draft: no draft batch scratch"))?;
                let toks = draft_dflash_sampled_blocks_batched(
                    gpu,
                    weights,
                    config.vocab_size,
                    shared,
                    head,
                    &mut dl,
                )?;
                drop(dl);
                for (k, t) in toks.into_iter().enumerate() {
                    out[i + k] = Some(t);
                }
                cb.draft_stats.0 += j - i;
                cb.draft_stats.1 += 1;
                i = j;
                continue;
            }
        }
        out[i] = Some(cb_draft_lane_singleton(
            gpu,
            weights,
            config,
            cb,
            &mut lanes[i],
        )?);
        i += 1;
    }
    Ok(out
        .into_iter()
        .map(|t| t.expect("every lane drafted"))
        .collect())
}

/// The singleton draft of one lane (sampled selector draft or greedy
/// [`dflash_lane_draft`]) with the shared rotation buffers lent for the call.
fn cb_draft_lane_singleton(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    cb: &mut DflashCbScratch,
    l: &mut DflashCbDraftLane<'_>,
) -> HipResult<Vec<u32>> {
    // A VMM lane has no rotation buffers of its own: borrow the
    // shared scratch's for this call and hand them back on EVERY
    // path (no `?` between lend and reclaim).
    if cb.draft.is_none() && l.state.df.draft_scratch.mq_x_rot.is_none() {
        cb.reserve_draft(gpu, &l.state.df.draft_config)?;
    }
    let lent = cb.lend_mq(&mut l.state.df.draft_scratch)?;
    let r = if l.state.sample_temp > 1e-6 {
        let cfg = l.state.sample_cfg();
        match l.state.sampled_draft.as_mut() {
            None => Err(HipError::new(
                0,
                "dflash_cb_draft: sampled receipt not reserved",
            )),
            Some(receipt) => dflash_sampled_selector_draft_into(
                gpu,
                weights,
                &mut l.state.df,
                l.position,
                l.compact_offset,
                l.seed,
                l.b,
                config.vocab_size,
                cfg,
                &mut l.state.rng_state,
                receipt,
            )
            .map(|()| receipt.block.clone()),
        }
    } else {
        dflash_lane_draft(
            gpu,
            weights,
            config,
            l.compact_offset,
            &mut l.state.df,
            l.position,
            l.seed,
            l.b,
        )
    };
    cb.reclaim_mq(&mut l.state.df.draft_scratch, lent);
    r
}

/// One drafted lane between verify and accept: its trunk owners, its private
/// DFlash state and the window it drafted.
pub struct DflashCbVerifyLane<'a> {
    pub kv_cache: &'a mut KvCache,
    pub dn_state: &'a mut DeltaNetState,
    pub state: &'a mut DflashVmmLaneState,
    pub draft: &'a DflashCbDraft,
}

/// Verify phase over drafted lanes: the pre-verify DeltaNet state of each
/// lane is preserved (right after its draft, as the singleton) either by a
/// SAVE copy into its `target_snap` or, on an admitted chunk, by the
/// save-free owner exchange ([`cb_dn_save_free_admitted`]); then whole lanes
/// in order
/// through `forward_prefill_batch_multi` chunks of at most the packing cap
/// ([`multi_chunk_pack_cap`] of `cb.max_rows()`: C1 `[16]`, C3 `[48]`, C4
/// `[48,16]`, C8 `[48,48,32]` at 63 rows; C8 `[128]` with the wide route on a
/// dense MQ4G256V2 gfx1201 target) with each lane's own ChainVerify fusion,
/// hidden ring and tape (offset 0). Greedy lanes share one head/argmax/D2H;
/// admitted sampled chunks share the exact F16 head, otherwise each sampled
/// lane retains its singleton-b Product head. Target rows remain lane-owned
/// and are materialized in original lane order before scratch reuse.
///
/// Everything refusable is checked before the first snapshot is saved (or
/// owner exchanged) or row launched (capture/recording, block sizes,
/// ring/tape capacity, batched-body eligibility). A later failure leaves the
/// lanes of executed chunks advanced (after an exchange, also the failing
/// chunk's lanes with their live bank unwritten): the caller poisons the
/// participants, and nothing here undoes an exchange.
pub fn dflash_cb_verify(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    scratch: &qwen35::Qwen35Scratch,
    cb: &mut DflashCbScratch,
    lanes: &mut [DflashCbVerifyLane<'_>],
) -> HipResult<()> {
    let refuse = |why: String| Err(HipError::new(0, &format!("dflash_cb_verify: {why}")));
    if lanes.is_empty() {
        return Ok(());
    }
    if gpu.graphs.capture_mode || gpu.replay.is_recording() {
        return refuse("graph capture / replay recording is not supported".into());
    }
    if config.num_experts > 0 {
        return refuse("MoE targets are not admitted".into());
    }
    // Effective packing cap: the scratch capacity, held to the product cap
    // unless the wide exact route is admitted for this target.
    let pack_cap = multi_chunk_pack_cap(gpu, weights, config, cb.max_rows);
    cb.rows.clear();
    for (i, lane) in lanes.iter().enumerate() {
        let b = lane.draft.verify_tokens.len();
        if lane.draft.verify_tokens.first() != Some(&lane.draft.seed) {
            return refuse(format!("lane {i}: window does not start with its seed"));
        }
        if b > pack_cap.min(MULTI_CHUNK_MAX_LANE_ROWS) {
            return refuse(format!("lane {i}: {b} rows > chunk capacity {pack_cap}"));
        }
        let df = &lane.state.df;
        if b > df.gdn_tape.max_n || b > df.hidden_rb.max_batch || b > df.hidden_rb.max_positions {
            return refuse(format!("lane {i}: {b} rows exceed its tape/ring staging"));
        }
        if !qwen35::prefill_batch_pbs_eligible(
            weights,
            config,
            lane.dn_state,
            b,
            gpu.arch.as_str(),
            true,
        ) {
            return refuse(format!(
                "lane {i}: {b} rows not eligible for the batched prefill body"
            ));
        }
        cb.rows.push(b);
    }
    pack_whole_lanes(&cb.rows, pack_cap, &mut cb.ranges)
        .map_err(|e| HipError::new(0, &format!("dflash_cb_verify: {e}")))?;
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
    let (dim, vocab) = (config.dim, config.vocab_size);
    for ci in 0..cb.ranges.len() {
        let range = cb.ranges[ci].clone();
        let rows: usize = cb.rows[range.clone()].iter().sum();
        // One decision per packed chunk, never per lane (no half admission).
        let save_free = cb_dn_save_free_admitted(gpu, config, &cb.trunk, &lanes[range.clone()]);
        if save_free {
            // Save-free DeltaNet verify (`HIPFIRE_CB_DN_SAVE_FREE`). Instead of
            // copying every lane's live state A into its snapshot B, exchange
            // the owners of the four state families, so the canonical live
            // `dn_state` owns the old snapshot bank and the canonical
            // `target_snap` owns A, the pre-window state. Verify then reads A
            // (the request's `dn_state_src`) and writes only the live bank:
            // the conv ring is copied A -> B (pure bytes) and the in-place conv
            // kernel runs on B, and `gated_delta_net_q8_fast_lanes_varrows_from` is the
            // in-place recurrence with only the first-use LOAD addresses
            // changed, so every byte the live bank ends with is the byte the
            // in-place verify of a SAVEd lane leaves. The existing commit then
            // runs unchanged on the canonical fields: full accept keeps the
            // live bank, partial / zero accept replay snapshot A -> live bank
            // (D8), and A stays the pre-window state terminal repair restores.
            // One stream drain before the exchange (everything queued on the
            // active stream, incl. the previous step's commit, has finished
            // with the old owners). The exchange itself runs below, after the
            // last fallible step that precedes the forward, and is host-only
            // and infallible after the whole-chunk preflight.
            if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip.stream_synchronize(stream)?;
            }
        } else {
            for lane in lanes[range.clone()].iter_mut() {
                lane.state.df.target_snap.save_from(lane.dn_state, gpu)?;
            }
        }
        // PM FP8 verify scope (`HIPFIRE_CB_VERIFY_FP8`): trunk + head run the one
        // E4M3 verify arithmetic at every row count, so this batched chunk equals
        // each lane's singleton verify (same admission: the lanes share target
        // and KV tier); closed on every path below.
        let prev = fp8_verify_open(gpu, weights, config, &*lanes[range.start].kv_cache)?;
        if save_free {
            for lane in lanes[range.clone()].iter_mut() {
                lane.state
                    .df
                    .target_snap
                    .exchange_buffers(&mut *lane.dn_state);
            }
        }
        let fwd = {
            let mut reqs: Vec<MultiChunkRequest<'_>> = Vec::with_capacity(range.len());
            for lane in lanes[range.clone()].iter_mut() {
                let DflashCbVerifyLane {
                    kv_cache,
                    dn_state,
                    state,
                    draft,
                } = lane;
                let df = &mut state.df;
                reqs.push(MultiChunkRequest {
                    tokens: &draft.verify_tokens,
                    start_pos: draft.position,
                    kv_cache: &mut **kv_cache,
                    dn_state: &mut **dn_state,
                    gdn_tape: Some(&df.gdn_tape),
                    fusion: DflashFusionCtx::ChainVerify,
                    hidden_rb: Some(&mut df.hidden_rb),
                    dn_state_src: if save_free {
                        Some(&df.target_snap)
                    } else {
                        None
                    },
                });
            }
            // Trunk row-byte identity at aggregates 63/64/65/96/128, chunk128
            // off/HIP/PM: cb/chunk128/STAGE4.md:39-58 (the aggregate probe
            // freezes verify.final_hidden as well as logits against singleton).
            // DFlash IU4 verify (this load's decision): the trunk runs inside the IU4 verify
            // scope, the same A4 producers + IU4 GEMMs as the singleton verify;
            // the heads below stay F16.
            let iu4 = gpu.iu4_verify_enter(dflash_iu4_verify_admitted(gpu, weights, config));
            let r = forward_prefill_batch_multi(
                gpu,
                weights,
                config,
                scratch,
                &cb.trunk,
                &mut reqs,
                Some(&cb.head.final_hidden),
            );
            gpu.iu4_verify_exit(iu4);
            r
        };
        // The admitted exact F16 family preserves every singleton Product-b
        // logit; all other sampled routes retain their original private head.
        let packed_sampled = rows <= cb.head.max_n
            && rows * weights.output.k <= cb.head.max_n * cb.head.hidden_k
            && cb_sampled_packed_head_admitted(gpu, weights, config, &lanes[range.clone()], rows);
        let any_greedy = lanes[range.clone()]
            .iter()
            .any(|l| l.state.sample_temp <= 1e-6);
        let picks = match fwd {
            Ok(()) if packed_sampled => {
                let hidden = cb.head.final_hidden.sub_offset(0, rows * dim);
                dflash_enqueue_verify_lm_head_argmax(
                    gpu,
                    &weights.output,
                    &hidden,
                    &cb.head,
                    rows,
                    vocab,
                    DenseBatchMath::SingletonWmma,
                )
                .and_then(|()| dflash_download_verify_argmax(gpu, &cb.head, rows))
                .map(Some)
            }
            Ok(()) if any_greedy => {
                dflash_cb_head_argmax(gpu, weights, config, &cb.head, rows).map(Some)
            }
            Ok(()) => Ok(None),
            Err(e) => Err(e),
        };
        // Per-lane heads run inside the FP8 scope too (the singleton's verify
        // head does); the scope is closed on every path.
        let heads = picks.and_then(|picks| {
            let mut row = 0usize;
            let chunk = &mut lanes[range];
            // Lanes `< batched_until` already hold their target rows from one
            // batched run (always the lane the run started at or later).
            let mut batched_until = 0usize;
            for li in 0..chunk.len() {
                if packed_sampled && li >= batched_until && *CB_SAMPLED_TARGET_BATCH {
                    let n = cb_sampled_target_run_len(
                        &chunk[li..],
                        gpu.dflash_device_accept_supported(),
                    );
                    if n >= 2 {
                        let end = li + n;
                        let run_rows: usize = chunk[li..end]
                            .iter()
                            .map(|l| l.draft.verify_tokens.len())
                            .sum();
                        let cfg = chunk[li].state.sample_cfg();
                        let logits = cb.head.logits.sub_offset(row * vocab, run_rows * vocab);
                        cb.target_rows
                            .run_sparse(gpu, &logits, run_rows, vocab, cfg)?;
                        let mut r0 = 0usize;
                        for l in chunk[li..end].iter_mut() {
                            let b = l.draft.verify_tokens.len();
                            cb.target_rows.split_lane_into(
                                r0,
                                b,
                                vocab,
                                &mut l.state.sampled_target,
                            )?;
                            r0 += b;
                        }
                        batched_until = end;
                    }
                }
                let in_run = li < batched_until;
                let device_accept = cb_sampled_device_accept_eligible(
                    &chunk[li],
                    gpu.dflash_device_accept_supported(),
                );
                let lane = &mut chunk[li];
                let b = lane.draft.verify_tokens.len();
                let st = &mut *lane.state;
                st.picks.clear();
                st.device_decision = None;
                if st.sample_temp > 1e-6 {
                    let cfg = st.sample_cfg();
                    let v = &st.df.verify_scratch;
                    gpu.memcpy_dtod_at_auto(
                        &v.final_hidden.buf,
                        0,
                        &cb.head.final_hidden.buf,
                        row * dim * 4,
                        b * dim * 4,
                    )?;
                    if packed_sampled {
                        st.picks.extend_from_slice(
                            &picks.as_ref().expect("packed sampled head")[row..row + b],
                        );
                        if device_accept {
                            let logits = cb.head.logits.sub_offset(row * vocab, b * vocab);
                            st.device_decision = Some(dflash_sampled_accept_device(
                                gpu,
                                &logits,
                                &lane.draft.verify_tokens,
                                st.sampled_draft.as_ref().expect("eligible sampled receipt"),
                                vocab,
                                cfg,
                                lane.draft.max_accept,
                                &mut st.rng_state,
                                &mut *v.device_accept.lock().unwrap_or_else(|e| e.into_inner()),
                            )?);
                        } else if !in_run {
                            let logits = cb.head.logits.sub_offset(row * vocab, b * vocab);
                            // Keep normalization at the original lane point/order,
                            // including SparseRows' request-owned representation.
                            dflash_target_rows_into(
                                gpu,
                                &logits,
                                b,
                                vocab,
                                cfg,
                                &mut st.sampled_target,
                            )?;
                        }
                        if cb.keep_lane_rows {
                            gpu.memcpy_dtod_at_auto(
                                &v.logits.buf,
                                0,
                                &cb.head.logits.buf,
                                row * vocab * 4,
                                b * vocab * 4,
                            )?;
                            gpu.memcpy_dtod_at_auto(
                                &v.argmax.buf,
                                0,
                                &cb.head.argmax.buf,
                                row * 4,
                                b * 4,
                            )?;
                        }
                    } else {
                        let hidden = v.final_hidden.sub_offset(0, b * dim);
                        dflash_enqueue_verify_lm_head_argmax(
                            gpu,
                            &weights.output,
                            &hidden,
                            v,
                            b,
                            vocab,
                            DenseBatchMath::Product,
                        )?;
                        st.picks = dflash_download_verify_argmax(gpu, v, b)?;
                        let logits = v.logits.sub_offset(0, b * vocab);
                        // Decide/materialize before shared scratch or v.logits reuse.
                        if device_accept {
                            st.device_decision = Some(dflash_sampled_accept_device(
                                gpu,
                                &logits,
                                &lane.draft.verify_tokens,
                                st.sampled_draft.as_ref().expect("eligible sampled receipt"),
                                vocab,
                                cfg,
                                lane.draft.max_accept,
                                &mut st.rng_state,
                                &mut *v.device_accept.lock().unwrap_or_else(|e| e.into_inner()),
                            )?);
                        } else {
                            dflash_target_rows_into(
                                gpu,
                                &logits,
                                b,
                                vocab,
                                cfg,
                                &mut st.sampled_target,
                            )?;
                        }
                    }
                } else {
                    st.picks
                        .extend_from_slice(&picks.as_ref().expect("greedy head")[row..row + b]);
                    if cb.keep_lane_rows {
                        let v = &st.df.verify_scratch;
                        gpu.memcpy_dtod_at_auto(
                            &v.final_hidden.buf,
                            0,
                            &cb.head.final_hidden.buf,
                            row * dim * 4,
                            b * dim * 4,
                        )?;
                        gpu.memcpy_dtod_at_auto(
                            &v.logits.buf,
                            0,
                            &cb.head.logits.buf,
                            row * vocab * 4,
                            b * vocab * 4,
                        )?;
                        gpu.memcpy_dtod_at_auto(
                            &v.argmax.buf,
                            0,
                            &cb.head.argmax.buf,
                            row * 4,
                            b * 4,
                        )?;
                    }
                }
                st.verified = true;
                row += b;
            }
            Ok(())
        });
        gpu.fp8_verify_exit(prev);
        heads?;
    }
    Ok(())
}

/// `HIPFIRE_CB_DN_SAVE_FREE_DIAG=1` (developer var, default off, process
/// snapshot): [`cb_dn_save_free_admitted`] writes one stderr line per packed
/// chunk naming the FIRST failing predicate, for
/// attributing why a chunk keeps the SAVE route. Investigative only: it never
/// changes the verdict, and when off nothing below runs (no formatting, no
/// allocation, no extra family walk).
static CB_DN_SAVE_FREE_DIAG: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_bool("HIPFIRE_CB_DN_SAVE_FREE_DIAG", false)
});

/// Early-return refusal of [`cb_dn_save_free_admitted`]. The detail arguments
/// are only evaluated when `$diag` is set.
macro_rules! dn_sf_refuse {
    ($diag:expr, $gpu:expr, $lanes:expr, $lane:expr, $reason:literal, $($detail:tt)+) => {{
        if $diag {
            dn_sf_log($gpu, $lanes, $lane, concat!("refuse=", $reason), format_args!($($detail)+));
        }
        return false;
    }};
}

/// One diagnostic line: chunk lane count / rows / per-lane `b` and sample
/// temperature, the offending lane's `b`, tape capacity and KV
/// `compact_offset` when `lane` is given, the process VARROWS switch and the
/// replay-graph / Redline flags, then the reason-specific `detail`.
fn dn_sf_log(
    gpu: &Gpu,
    lanes: &[DflashCbVerifyLane<'_>],
    lane: Option<usize>,
    verdict: &str,
    detail: std::fmt::Arguments<'_>,
) {
    use std::fmt::Write as _;
    let mut per_lane = String::new();
    let mut rows = 0usize;
    let _ = per_lane.write_str(" b=[");
    for (i, l) in lanes.iter().enumerate() {
        let b = l.draft.verify_tokens.len();
        rows += b;
        let _ = write!(per_lane, "{}{b}", if i == 0 { "" } else { "," });
    }
    let _ = per_lane.write_str("] sample_temp=[");
    for (i, l) in lanes.iter().enumerate() {
        let _ = write!(
            per_lane,
            "{}{}",
            if i == 0 { "" } else { "," },
            l.state.sample_temp
        );
    }
    per_lane.push(']');
    if let Some((i, l)) = lane.and_then(|i| lanes.get(i).map(|l| (i, l))) {
        let _ = write!(
            per_lane,
            " lane={i} b_i={} tape_max_n={} compact_offset={}",
            l.draft.verify_tokens.len(),
            l.state.df.gdn_tape.max_n,
            l.kv_cache.compact_offset,
        );
    }
    eprintln!(
        "[cb-dn-save-free-diag] {verdict} lanes={} rows={rows} b0={} varrows={} replay_enabled={} replay_graphs={}{per_lane} {detail}",
        lanes.len(),
        lanes.first().map_or(0, |l| l.draft.verify_tokens.len()),
        lane_gdn_varrows_enabled(),
        gpu.replay.is_enabled(),
        gpu.graphs.replay_graph_count(),
    );
}

/// Components of [`GdnTape::replay_from_snapshot_admits`] (its private
/// `replay_ml_admits` rebuilt from public fields), reported separately.
fn dn_sf_replay_detail(
    gpu: &Gpu,
    state: &DeltaNetState,
    tape: &crate::speculative::GdnTape,
    snap: &DeltaNetSnapshot,
    b: usize,
) -> String {
    use rdna_compute::dflash_gdn_replay::DFLASH_GDN_REPLAY_MAX_STEPS;
    let n_la = tape.qkv_bufs.len();
    format!(
        "dn_snapshot_flip={} q8={} n_la={n_la} steps_le_tape_max_n={} s_len_eq_n_la={} conv_len_eq_n_la={} \
         backend_replay_eligible={} (supported={} recording={} steps_in_1..={}={}) mirrors={} \
         live_ef_len={} snap_ef_len={}",
        gpu.flags.dn_snapshot_flip,
        state.quant == qwen35::StateQuant::Q8,
        b <= tape.max_n,
        state.s_matrices.len() == n_la,
        state.conv_states.len() == n_la,
        gpu.gdn_replay_ml_eligible(tape.n_v_heads, tape.n_key_heads, tape.key_head_dim, tape.value_head_dim, b),
        gpu.gdn_replay_ml_supported(tape.n_v_heads, tape.n_key_heads, tape.key_head_dim, tape.value_head_dim),
        gpu.replay.is_recording(),
        DFLASH_GDN_REPLAY_MAX_STEPS,
        (1..=DFLASH_GDN_REPLAY_MAX_STEPS).contains(&b),
        snap.mirrors(state),
        state.s_ef_residual.len(),
        snap.s_ef_len(),
    )
}

/// First failing family/layer of [`DeltaNetSnapshot::cb_verify_owned_exact`],
/// walked in its own order (geometry, layer count, Q8, S, scales, conv, EF
/// extents, then HIP ownership). It runs only after `mirrors` passed, so each
/// snapshot buffer has the live buffer's allocated size; the snapshot's
/// buffers are private, so its ownership is reported by elimination.
/// The logical extents repeat `DnCbExtents::for_config` (private to
/// `speculative.rs`).
fn dn_sf_owned_detail(
    state: &DeltaNetState,
    snap: &DeltaNetSnapshot,
    config: &Qwen35Config,
) -> String {
    if config.conv_kernel_dim != 4
        || config.linear_key_head_dim != 128
        || config.linear_value_head_dim != 128
    {
        return format!(
            "geometry conv_kernel_dim={} key_head_dim={} value_head_dim={} (exact kernels need 4/128/128)",
            config.conv_kernel_dim, config.linear_key_head_dim, config.linear_value_head_dim
        );
    }
    let (hv, hk) = (config.linear_num_value_heads, config.linear_num_key_heads);
    let n = config
        .layer_types
        .iter()
        .filter(|t| **t == qwen35::LayerType::LinearAttention)
        .count();
    if n == 0 {
        return "no_deltanet_layers".to_string();
    }
    if state.quant != qwen35::StateQuant::Q8 {
        return format!("state_quant={:?} (need Q8)", state.quant);
    }
    let conv_channels = 2 * hk * 128 + hv * 128;
    let fams: [(&str, &[rdna_compute::GpuTensor], usize, bool); 4] = [
        ("s", state.s_matrices.as_slice(), hv * 128 * 128, true),
        ("scales", state.s_scales.as_slice(), hv * 128 * 4, false),
        (
            "conv",
            state.conv_states.as_slice(),
            conv_channels * 3 * 4,
            false,
        ),
        (
            "ef",
            state.s_ef_residual.as_slice(),
            hv * 128 * 128 * 2,
            true,
        ),
    ];
    for (name, live, logical, exact) in fams {
        if live.len() != n {
            return format!("family={name} live_len={} expected_layers={n}", live.len());
        }
        for (la, t) in live.iter().enumerate() {
            let sz = t.buf.size();
            if if exact { sz != logical } else { sz < logical } {
                return format!(
                    "family={name} layer={la} live_alloc={sz} snap_alloc_inferred={sz}(from mirrors==true; not read) \
                     logical={logical} exact_required={exact} hip_owned_live={}",
                    t.buf.is_hip_allocation()
                );
            }
        }
    }
    for (name, live, _, _) in fams {
        for (la, t) in live.iter().enumerate() {
            if !t.buf.is_hip_allocation() {
                return format!(
                    "family={name} layer={la} live_not_hip_owned vmm={} borrowed={}",
                    t.buf.is_vmm_owner(),
                    t.buf.is_borrowed()
                );
            }
        }
    }
    format!(
        "snapshot_hip_ownership=INFERRED_not_observed (candidate: snapshot buffers not HIP-owned, by elimination only: \
         geometry, extents and live ownership passed; snapshot buffers are private and were not inspected) \
         snap_bytes={} live_bytes={}",
        snap.bytes_len(),
        DeltaNetSnapshot::bytes_for(state)
    )
}

/// The first duplicated owner pointer of the sorted chunk vector, how often it
/// occurs, and every `lane:bank:family[layer]` slot holding it
/// (`push_owner_ptrs` order: live S/scales/conv/EF then the snapshot's).
fn dn_sf_alias_detail(lanes: &[DflashCbVerifyLane<'_>], sorted: &[u64]) -> String {
    use std::fmt::Write as _;
    let Some(dup) = sorted.windows(2).find(|w| w[0] == w[1]).map(|w| w[0]) else {
        return String::new();
    };
    let mut out = format!(
        "dup_ptr={dup:#x} occurrences={} slots:",
        sorted.iter().filter(|&&p| p == dup).count()
    );
    let mut tmp: Vec<u64> = Vec::new();
    for (li, lane) in lanes.iter().enumerate() {
        let state: &DeltaNetState = &*lane.dn_state;
        tmp.clear();
        lane.state.df.target_snap.push_owner_ptrs(state, &mut tmp);
        let lens = [
            state.s_matrices.len(),
            state.s_scales.len(),
            state.conv_states.len(),
            state.s_ef_residual.len(),
        ];
        let live_len: usize = lens.iter().sum();
        for (idx, _) in tmp.iter().enumerate().filter(|(_, p)| **p == dup) {
            let (bank, mut rel) = if idx < live_len {
                ("live", idx)
            } else {
                ("snap", idx - live_len)
            };
            let mut fam = 0;
            while fam < 3 && rel >= lens[fam] {
                rel -= lens[fam];
                fam += 1;
            }
            let _ = write!(
                out,
                " {li}:{bank}:{}[{rel}]",
                ["s", "scales", "conv", "ef"][fam]
            );
        }
    }
    out
}

/// Whole-chunk admission of the save-free DeltaNet verify for one packed chunk
/// (`lanes` = the chunk's lanes), checked BEFORE any owner is exchanged. All
/// of the following must hold, otherwise the chunk takes today's SAVE route
/// unchanged:
///
/// * `HIPFIRE_CB_DN_SAVE_FREE` on (`cb_dn_save_free_enabled`), exact gfx1201;
/// * the shared scratch armed its snapshot-source lane table
///   ([`MultiChunkScratch::gdn_from_ready`]);
/// * the lane-batched GDN admission's process half
///   ([`lane_gdn_batch_ready`]: `HIPFIRE_CB_LANE_GDN_BATCH` /
///   `HIPFIRE_CB_TAPE_COPY_BATCH`, `1..=CONV_LANES_MAX` lanes, exact gfx1201
///   head geometry, no capture / recording / G0 observation) and its chunk
///   half (Sequential, ChainVerify, tape per lane — all structural here —
///   and either EQUAL row counts or, unless `HIPFIRE_CB_LANE_GDN_VARROWS=0`,
///   a per-lane count, each lane within its own tape);
/// * per lane `2 <= b_i <= 16` (the D8 replay's step bound, checked against
///   the lane's own count, so every accept class can replay from the snapshot);
/// * no enabled retained Redline controller and no existing replay graph
///   (`replay_graph_count() == 0`: a graph's fast launch bypasses the
///   table re-fingerprinting an owner exchange relies on);
/// * per lane: D8 on + the replay-ML route admitted for `b` steps + the
///   snapshot mirrors the state ([`GdnTape::replay_from_snapshot_admits`]),
///   the tape's snapshot-source tables armed
///   ([`GdnTape::snapshot_source_armed`]), all four families exact: Q8 + EF,
///   conv kernel dimension 4, the config's DeltaNet layer count, per-layer
///   equal-size banks each at least the exact logical extent (pool rounding
///   allowed) and HIP-owned buffers on both banks
///   ([`DeltaNetSnapshot::cb_verify_owned_exact`]), and an uncompacted Q8/fp8
///   KV cache (the batched forward's own admission, so it cannot refuse after
///   the exchange);
/// * the chunk's `2 * 4 * n_layers` owner pointers per lane are pairwise
///   distinct across all lanes and both banks.
///
/// Host-only: one small pointer vector, no device access.
pub(crate) fn cb_dn_save_free_admitted(
    gpu: &Gpu,
    config: &Qwen35Config,
    trunk: &MultiChunkScratch,
    lanes: &[DflashCbVerifyLane<'_>],
) -> bool {
    use rdna_compute::dflash_gdn_replay::DFLASH_GDN_REPLAY_MAX_STEPS;
    let diag = *CB_DN_SAVE_FREE_DIAG;
    let Some(first) = lanes.first() else {
        dn_sf_refuse!(diag, gpu, lanes, None, "empty", "")
    };
    let b0 = first.draft.verify_tokens.len();
    let varrows = lane_gdn_varrows_enabled();
    // Same order and short-circuit as the single `||` chain this replaces.
    if !cb_dn_save_free_enabled() {
        dn_sf_refuse!(diag, gpu, lanes, None, "disabled", "")
    }
    if !gpu.arch_caps.is_gfx1201() {
        dn_sf_refuse!(diag, gpu, lanes, None, "arch", "")
    }
    if !trunk.gdn_from_ready() {
        dn_sf_refuse!(diag, gpu, lanes, None, "trunk_from_unready", "")
    }
    if !lane_gdn_batch_ready(gpu, config, lanes.len()) {
        dn_sf_refuse!(
            diag,
            gpu,
            lanes,
            None,
            "lane_gdn_unready",
            "conv_lanes_max={} backend_lane_gdn_admitted={} recording={} capture={} g0_observing={} \
             env_LANE_GDN_BATCH={:?} env_TAPE_COPY_BATCH={:?}",
            rdna_compute::lane_gdn::CONV_LANES_MAX,
            gpu.lane_gdn_batch_admitted(
                config.linear_num_value_heads,
                config.linear_num_key_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
            ),
            gpu.replay.is_recording(),
            gpu.graphs.capture_mode,
            gpu.replay.is_g0_observing(),
            hipfire_config::developer_var("HIPFIRE_CB_LANE_GDN_BATCH").ok(),
            hipfire_config::developer_var("HIPFIRE_CB_TAPE_COPY_BATCH").ok(),
        )
    }
    if gpu.replay.is_enabled() {
        dn_sf_refuse!(diag, gpu, lanes, None, "redline_replay_enabled", "")
    }
    if gpu.graphs.replay_graph_count() != 0 {
        dn_sf_refuse!(diag, gpu, lanes, None, "replay_graphs", "")
    }
    let mut ptrs: Vec<u64> = Vec::with_capacity(lanes.len() * 8 * config.n_layers);
    for (li, lane) in lanes.iter().enumerate() {
        let df = &lane.state.df;
        let state: &DeltaNetState = &*lane.dn_state;
        let kv: &KvCache = &*lane.kv_cache;
        let b_i = lane.draft.verify_tokens.len();
        if !(2..=DFLASH_GDN_REPLAY_MAX_STEPS).contains(&b_i) {
            dn_sf_refuse!(
                diag,
                gpu,
                lanes,
                Some(li),
                "b_range",
                "allowed=2..={}",
                DFLASH_GDN_REPLAY_MAX_STEPS
            )
        }
        if !(varrows || b_i == b0) {
            dn_sf_refuse!(diag, gpu, lanes, Some(li), "b_mixed_no_varrows", "")
        }
        if b_i > df.gdn_tape.max_n {
            dn_sf_refuse!(diag, gpu, lanes, Some(li), "tape_capacity", "")
        }
        if !df.gdn_tape.snapshot_source_armed() {
            dn_sf_refuse!(diag, gpu, lanes, Some(li), "source_unarmed", "")
        }
        if !df
            .gdn_tape
            .replay_from_snapshot_admits(gpu, state, &df.target_snap, b_i)
        {
            dn_sf_refuse!(
                diag,
                gpu,
                lanes,
                Some(li),
                "replay_from_refused",
                "{}",
                dn_sf_replay_detail(gpu, state, &df.gdn_tape, &df.target_snap, b_i)
            )
        }
        if !df.target_snap.cb_verify_owned_exact(state, config) {
            dn_sf_refuse!(
                diag,
                gpu,
                lanes,
                Some(li),
                "owned_pair_refused",
                "{}",
                dn_sf_owned_detail(state, &df.target_snap, config)
            )
        }
        if !(kv.quant_q8 || kv.quant_fp8) {
            dn_sf_refuse!(
                diag,
                gpu,
                lanes,
                Some(li),
                "kv_mode",
                "quant_q8={} quant_fp8={}",
                kv.quant_q8,
                kv.quant_fp8
            )
        }
        if kv.compact_offset != 0 {
            dn_sf_refuse!(diag, gpu, lanes, Some(li), "compact_offset", "")
        }
        df.target_snap.push_owner_ptrs(state, &mut ptrs);
    }
    ptrs.sort_unstable();
    if !ptrs.windows(2).all(|w| w[0] != w[1]) {
        dn_sf_refuse!(
            diag,
            gpu,
            lanes,
            None,
            "owner_alias",
            "{}",
            dn_sf_alias_detail(lanes, &ptrs)
        )
    }
    true
}

#[cfg(test)]
mod tests {
    use super::dflash_block_for_emit;

    #[test]
    fn block_follows_the_singleton_budget_rule() {
        // B = min(16, max(E, 2)); E = 1 still verifies two rows.
        let b: Vec<usize> = [1, 2, 3, 15, 16, 17, 1000]
            .iter()
            .map(|&e| dflash_block_for_emit(16, e))
            .collect();
        assert_eq!(b, vec![2, 2, 3, 15, 16, 16, 16]);
    }
}
