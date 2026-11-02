// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cross-request batched MTP speculative decode (continuous batching).
//!
//! One cycle over several requests, each its own singleton MTP window
//! (`spec_step_mtp_compressed_serial_inner`, the `Qwen35MtpDrafter::mtp_step`
//! route) except that the verify trunk and the verify head are shared:
//!
//! 1. draft — per request, unchanged ([`mtp_draft_phase_inner`]); the greedy
//!    device-chain lanes and the exact sampled lanes (matching softmax
//!    controls; per-lane penalties replayed exactly) additionally share the lm_head (and, sampled, the
//!    softmax kernel) per draft step ([`mtp_cb_draft_batched`]);
//! 2. verify trunk — every request's `[seed, drafts..]` rows through ONE
//!    [`forward_prefill_batch_multi`] (per request byte-identical to the
//!    singleton verify forward, rollback tape included);
//! 3. verify head — one row-independent lm_head over all rows, each
//!    request's logits/hidden rows copied into its own `MtpSpecState`;
//! 4. accept/repair — per request, the singleton consumer
//!    ([`mtp_accept_and_rollback`] with the head precomputed).
//!
//! A request whose window cannot take the shared trunk (a single verify row,
//! or a batched-prefill-ineligible row count) runs the singleton verify.

use super::*;
use crate::qwen35::prefill::fp8_verify_open;
use crate::qwen35::prefill::multi::{
    forward_prefill_batch_multi, MultiChunkRequest, MultiChunkScratch, MULTI_CHUNK_MAX_LANE_ROWS,
    MULTI_CHUNK_PRODUCT_MAX_ROWS,
};

/// Rows one MTP CB trunk chunk packs: the product cap. The wide 64..=128-row
/// verify route is DFlash-only until a wide MTP state-oracle receipt exists,
/// so MTP never takes the exact wide head; inside the PM FP8 verify scope its
/// shared stages run `DenseBatchMath::SingletonWmma` at these <= 63 rows.
const MTP_CB_MAX_ROWS: usize = MULTI_CHUNK_PRODUCT_MAX_ROWS;

/// Shared verify buffers for one CB engine: trunk scratch, post-norm hidden
/// rows, head rotation scratch and head logits, all `max_rows` rows
/// (`<= MTP_CB_MAX_ROWS`, 63), plus the sampled-draft softmax outputs the
/// batched sampled drafter reuses: `sampled_probs` (`max_rows * vocab`),
/// `sampled_tau` and `sampled_z` (`max_rows` each). The host staging for
/// those three tensors is NOT held here: it is allocated once per
/// [`mtp_cb_draft_batched`] call, sized for the largest sampled cohort.
pub struct MtpCbScratch {
    pub max_rows: usize,
    trunk: MultiChunkScratch,
    hidden: GpuTensor,
    rot: GpuTensor,
    logits: GpuTensor,
    sampled_probs: GpuTensor,
    sampled_tau: GpuTensor,
    sampled_z: GpuTensor,
}

impl MtpCbScratch {
    /// `max_rows` is clamped to `2..=63` (the MTP effective cap, whatever the
    /// wide verify route admits); a cycle with more rows runs in several
    /// shared chunks.
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, max_rows: usize) -> HipResult<Self> {
        let max_rows = max_rows.clamp(2, MTP_CB_MAX_ROWS);
        let trunk = MultiChunkScratch::new(gpu, config, max_rows)?;
        let alloc = |gpu: &mut Gpu, n: usize| gpu.zeros(&[n], DType::F32);
        let hidden = alloc(gpu, max_rows * config.dim)?;
        let rot = alloc(gpu, max_rows * config.dim)?;
        let logits = alloc(gpu, max_rows * config.vocab_size)?;
        let sampled_probs = alloc(gpu, max_rows * config.vocab_size)?;
        let sampled_tau = alloc(gpu, max_rows)?;
        let sampled_z = alloc(gpu, max_rows)?;
        Ok(Self {
            max_rows,
            trunk,
            hidden,
            rot,
            logits,
            sampled_probs,
            sampled_tau,
            sampled_z,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> HipResult<()> {
        self.trunk.free_gpu(gpu)?;
        gpu.free_tensor(self.hidden)?;
        gpu.free_tensor(self.rot)?;
        gpu.free_tensor(self.logits)?;
        gpu.free_tensor(self.sampled_probs)?;
        gpu.free_tensor(self.sampled_tau)?;
        gpu.free_tensor(self.sampled_z)
    }
}

/// One lane's draft request.
pub struct MtpCbDraftLane<'a> {
    pub state: &'a mut MtpSpecState,
    pub cur_pos: usize,
    pub last_committed: u32,
    /// Emitted history ending in `last_committed` (penalty window).
    pub emitted: &'a [u32],
    pub k: usize,
}

/// Does this lane's singleton draft take the greedy full-vocab
/// device-token-chain loop (no proposal graph) whose head the batched
/// drafter shares?
fn draft_batchable(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    head: &Qwen35MtpHead,
    lane: &MtpCbDraftLane<'_>,
) -> bool {
    let st = &*lane.state;
    lane.k.min(st.max_n) > 0
        && head.weights.lm_head_draft.is_none()
        && st.sampling.is_greedy()
        && st.p_min <= 0.0
        && mtp_device_token_chain_enabled_from_env()
        && mtp_device_token_chain_eligible_for(weights.embd_format, false, false)
        && weights.output.gpu_dtype == DType::MQ4G256V2
        && gpu.arch_caps.is_gfx1201()
}

/// Does this lane's singleton draft take the exact full-vocab SAMPLED loop
/// (host token embedding, no device token chain, no proposal graph; any
/// per-lane penalties are replayed per row by the batched drafter) whose head the batched sampled drafter shares? Its lm_head is
/// the same PM `multirow_r2` the greedy group uses, so the PM launcher's
/// shape/32-bit-offset limits for the largest chunk
/// (`min(cb.max_rows, PM_XBATCH_MAX)` columns) are part of eligibility:
/// an unsupported shape drafts as a singleton instead of erroring. The
/// batched PM column is bit-identical to the singleton `gemv_mq4g256v2`
/// only for the 2-row default (`gemv_rows_default() == 2`).
fn sampled_batchable(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    head: &Qwen35MtpHead,
    cb: &MtpCbScratch,
    lane: &MtpCbDraftLane<'_>,
) -> bool {
    let st = &*lane.state;
    let out_w = &weights.output;
    let (m, k) = (out_w.m, out_w.k);
    let b = cb.max_rows.min(rdna_compute::pm_xbatch::PM_XBATCH_MAX);
    let fits = |n: Option<usize>| n.is_some_and(|n| n <= u32::MAX as usize);
    lane.k.min(st.max_n) > 0
        && head.weights.lm_head_draft.is_none()
        && !st.sampling.is_greedy()
        && out_w.gpu_dtype == DType::MQ4G256V2
        && gpu.arch_caps.is_gfx1201()
        && gpu.arch_caps.gemv_rows_default() == 2
        && m == config.vocab_size
        && k == config.dim
        && m > 0
        && k > 0
        && k % 256 == 0
        && m <= i32::MAX as usize
        && k <= i32::MAX as usize
        && fits(m.checked_mul(b * 4))
        && fits(k.checked_mul(b * 4))
        && fits((k / 256).checked_mul(m.saturating_mul(136)))
}

/// Draft phase of many lanes, step-synchronous: at each draft step every
/// lane runs its own singleton head block, norm and lm_head input rotate,
/// then ONE multi-column lm_head (the PM twin of the singleton
/// `gemv_mq4g256v2_multirow_r2`, column-identical) over all lanes, then
/// either each lane's own argmax into its device token chain (greedy group)
/// or, for exact sampled lanes ([`sampled_batchable`]: same softmax
/// controls, per-lane penalties replayed on each lane's own row), ONE batched softmax+nucleus kernel over the cohort
/// with three dense D2H copies, then each lane's own host top_p cut and
/// RNG draw. Per lane this is the singleton greedy full-vocab device-chain
/// draft or the singleton sampled draft; other lanes draft alone
/// ([`mtp_cb_draft`]). `cb.rot`/`cb.logits`/`cb.sampled_*` are free before
/// verify and stage the head columns.
#[allow(clippy::too_many_arguments)]
pub fn mtp_cb_draft_batched(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    head: &Qwen35MtpHead,
    cb: &MtpCbScratch,
    lanes: &mut [MtpCbDraftLane<'_>],
) -> HipResult<Vec<MtpDraftOutput>> {
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
    let dim = config.dim;
    let vocab = config.vocab_size;
    let batched: Vec<bool> = lanes
        .iter()
        .map(|l| draft_batchable(gpu, weights, head, l))
        .collect();
    let group: Vec<usize> = (0..lanes.len()).filter(|&i| batched[i]).collect();
    let mut out: Vec<Option<MtpDraftOutput>> = (0..lanes.len()).map(|_| None).collect();
    if group.len() >= 2 && group.len() <= cb.max_rows {
        let n_k: Vec<usize> = group
            .iter()
            .map(|&i| lanes[i].k.min(lanes[i].state.max_n))
            .collect();
        for &i in &group {
            let l = &mut lanes[i];
            l.state.penalty.begin_window(l.emitted);
            let seed = l.last_committed as i32;
            gpu.hip
                .memcpy_htod(&l.state.mtp_token_chain.buf, &seed.to_ne_bytes())?;
        }
        let out_w = &weights.output;
        let k_max = n_k.iter().copied().max().unwrap_or(0);
        for k in 0..k_max {
            let act: Vec<usize> = (0..group.len()).filter(|&g| n_k[g] > k).collect();
            for (col, &g) in act.iter().enumerate() {
                let cur_pos = lanes[group[g]].cur_pos;
                let st = &mut *lanes[group[g]].state;
                let token_slot = st.mtp_token_chain.sub_offset(k, 1);
                embed_device_token_into(gpu, weights, &st.mtp_token_embed, &token_slot, dim)?;
                let prev = if k == 0 {
                    st.prev_hidden.sub_offset(0, dim)
                } else {
                    st.mtp_t_outs.sub_offset((k - 1) * dim, dim)
                };
                mtp_head::mtp_head_forward_block_only(
                    gpu,
                    head,
                    &st.mtp_scratch,
                    &mut st.mtp_kv,
                    0,
                    &prev,
                    Some(&st.mtp_token_embed),
                    cur_pos + k,
                    weights,
                )?;
                gpu.rmsnorm_f32(
                    &st.mtp_scratch.t_mtp_out,
                    &head.weights.shared_head_norm,
                    &st.mtp_scratch.tmp,
                    head.config.rms_norm_eps,
                )?;
                // weight_gemv's MQ4 input rotate, into this lane's column.
                let col_x = cb.rot.sub_offset(col * dim, dim);
                llama::rotate_x_mq_for(gpu, out_w, &st.mtp_scratch.tmp, &col_x, out_w.k)?;
            }
            let mut c0 = 0usize;
            while c0 < act.len() {
                let c1 = (c0 + rdna_compute::pm_xbatch::PM_XBATCH_MAX).min(act.len());
                hipfire_dispatch::ops::pm_xbatch::multirow_r2(
                    gpu,
                    &out_w.buf,
                    &cb.rot.sub_offset(c0 * dim, (c1 - c0) * dim),
                    &cb.logits.sub_offset(c0 * vocab, (c1 - c0) * vocab),
                    vocab,
                    dim,
                    c1 - c0,
                )?;
                c0 = c1;
            }
            for (col, &g) in act.iter().enumerate() {
                let st = &mut *lanes[group[g]].state;
                let argmax_view = st.mtp_lm_argmax.sub_offset(0, 1);
                let logits = cb.logits.sub_offset(col * vocab, vocab);
                gpu.argmax_token_chain_f32(
                    &logits,
                    &argmax_view,
                    &st.mtp_token_chain,
                    None,
                    vocab,
                    k + 1,
                )?;
                if k + 1 < n_k[g] {
                    gpu.memcpy_dtod_at_auto(
                        &st.mtp_t_outs.buf,
                        k * dim * 4,
                        &st.mtp_scratch.t_mtp_out.buf,
                        0,
                        dim * 4,
                    )?;
                }
            }
        }
        for (g, &i) in group.iter().enumerate() {
            let l = &mut lanes[i];
            let n = n_k[g];
            let mut host = vec![0i32; n];
            {
                let bytes: &mut [u8] =
                    unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut u8, n * 4) };
                gpu.hip
                    .memcpy_dtoh(bytes, &l.state.mtp_token_chain.sub_offset(1, n).buf)?;
            }
            out[i] = Some(MtpDraftOutput {
                candidates: host.into_iter().map(|t| t as u32).collect(),
                drafts_generated: n,
                chain_truncated: false,
                use_sampling: false,
                sampling: l.state.sampling,
                draft_probs: Vec::new(),
                draft_softmaxes: Vec::new(),
                use_device_token_chain: true,
                cur_pos: l.cur_pos,
                last_committed: l.last_committed,
            });
        }
    }
    // ── Exact sampled lanes ──────────────────────────────────────────────
    // Each eligible sampled lane is the singleton full-vocab sampled draft
    // (host token embedding, per-lane head block + norm + rotate, host RNG,
    // `apply_topp_trunc` / `sample_categorical`) with the lm_head GEMV and
    // the softmax+nucleus kernel shared across the cohort. A cohort must be
    // launch-uniform in the softmax controls (raw-bit-equal temp/top_p/min_p, equal top_k; penalty
    // controls may differ per lane and are applied per lane row before the
    // shared softmax), kept in lane order and split at `cb.max_rows`; cohorts of
    // fewer than two lanes fall through to the singleton drafter below.
    let mut cohorts: Vec<Vec<usize>> = Vec::new();
    for i in 0..lanes.len() {
        if out[i].is_some() || !sampled_batchable(gpu, weights, config, head, cb, &lanes[i]) {
            continue;
        }
        let s = lanes[i].state.sampling;
        let same = |j: usize| {
            let t = lanes[j].state.sampling;
            s.temp.to_bits() == t.temp.to_bits()
                && s.top_p.to_bits() == t.top_p.to_bits()
                && s.min_p.to_bits() == t.min_p.to_bits()
                && s.top_k == t.top_k
        };
        match cohorts
            .iter_mut()
            .find(|c| c.len() < cb.max_rows && same(c[0]))
        {
            Some(c) => c.push(i),
            None => cohorts.push(vec![i]),
        }
    }
    // Host staging for the three dense D2H copies per depth: allocated ONCE
    // per call for the largest cohort that will batch, reused across depths
    // and cohorts (`download_f32_into` resizes within this capacity).
    let stage_rows = cohorts
        .iter()
        .map(Vec::len)
        .filter(|&n| n >= 2)
        .max()
        .unwrap_or(0);
    let mut probs_host: Vec<f32> = Vec::with_capacity(stage_rows * vocab);
    let mut tau_host: Vec<f32> = Vec::with_capacity(stage_rows);
    let mut z_host: Vec<f32> = Vec::with_capacity(stage_rows);
    let mut act: Vec<usize> = Vec::with_capacity(stage_rows);
    for group in cohorts.iter().filter(|c| c.len() >= 2) {
        let samp = lanes[group[0]].state.sampling;
        let out_w = &weights.output;
        let n_k: Vec<usize> = group
            .iter()
            .map(|&i| lanes[i].k.min(lanes[i].state.max_n))
            .collect();
        let mut outs: Vec<MtpDraftOutput> = Vec::with_capacity(group.len());
        for (g, &i) in group.iter().enumerate() {
            let l = &mut lanes[i];
            l.state.penalty.begin_window(l.emitted);
            outs.push(MtpDraftOutput {
                candidates: Vec::with_capacity(n_k[g]),
                drafts_generated: 0,
                chain_truncated: false,
                use_sampling: true,
                sampling: l.state.sampling,
                draft_probs: Vec::with_capacity(n_k[g]),
                draft_softmaxes: Vec::with_capacity(n_k[g]),
                use_device_token_chain: false,
                cur_pos: l.cur_pos,
                last_committed: l.last_committed,
            });
        }
        let k_max = n_k.iter().copied().max().unwrap_or(0);
        for k in 0..k_max {
            act.clear();
            act.extend((0..group.len()).filter(|&g| k < n_k[g] && !outs[g].chain_truncated));
            if act.is_empty() {
                break;
            }
            for (col, &g) in act.iter().enumerate() {
                let cur_pos = lanes[group[g]].cur_pos;
                let st = &mut *lanes[group[g]].state;
                let next_tok = if k == 0 {
                    outs[g].last_committed
                } else {
                    outs[g].candidates[k - 1]
                };
                if k == 0 {
                    mtp_head::mtp_head_forward_block_only(
                        gpu,
                        head,
                        &st.mtp_scratch,
                        &mut st.mtp_kv,
                        next_tok,
                        &st.prev_hidden,
                        None,
                        cur_pos + k,
                        weights,
                    )?;
                } else {
                    let prev_row = st.mtp_t_outs.sub_offset((k - 1) * dim, dim);
                    mtp_head::mtp_head_forward_block_only(
                        gpu,
                        head,
                        &st.mtp_scratch,
                        &mut st.mtp_kv,
                        next_tok,
                        &prev_row,
                        None,
                        cur_pos + k,
                        weights,
                    )?;
                }
                gpu.rmsnorm_f32(
                    &st.mtp_scratch.t_mtp_out,
                    &head.weights.shared_head_norm,
                    &st.mtp_scratch.tmp,
                    head.config.rms_norm_eps,
                )?;
                let col_x = cb.rot.sub_offset(col * dim, dim);
                llama::rotate_x_mq_for(gpu, out_w, &st.mtp_scratch.tmp, &col_x, out_w.k)?;
            }
            let n_act = act.len();
            let mut c0 = 0usize;
            while c0 < n_act {
                let c1 = (c0 + rdna_compute::pm_xbatch::PM_XBATCH_MAX).min(n_act);
                hipfire_dispatch::ops::pm_xbatch::multirow_r2(
                    gpu,
                    &out_w.buf,
                    &cb.rot.sub_offset(c0 * dim, (c1 - c0) * dim),
                    &cb.logits.sub_offset(c0 * vocab, (c1 - c0) * vocab),
                    vocab,
                    dim,
                    c1 - c0,
                )?;
                c0 = c1;
            }
            // Per-lane exact-AR penalties on the lane's own full-vocab row
            // (`emitted ‖ candidates[..k]` history, lane's own controls),
            // mirroring the singleton drafter: the row's penalty depth is the
            // logical draft depth `k`, not the packed column `col`.
            for (col, &g) in act.iter().enumerate() {
                let st = &mut *lanes[group[g]].state;
                if !st.sampling.penalized() {
                    continue;
                }
                let sampling = st.sampling;
                let t0 = MtpPenaltyTiming::start();
                let mut uploads = 0;
                if k == 0 {
                    mtp_penalty_begin_drafts(gpu, &mut st.penalty, &st.mtp_sample_repeat_buf)?;
                    uploads = u32::from(!st.penalty.tokens().is_empty());
                }
                mtp_penalty_apply_row(
                    gpu,
                    &st.penalty,
                    &st.mtp_sample_repeat_buf,
                    &cb.logits.sub_offset(col * vocab, vocab),
                    k,
                    vocab,
                    &sampling,
                )?;
                let launches = u32::from(!st.penalty.row_range(k).is_empty());
                st.penalty_timing.finish(t0, uploads, launches);
            }
            let probs_view = cb.sampled_probs.sub_offset(0, n_act * vocab);
            let tau_view = cb.sampled_tau.sub_offset(0, n_act);
            let z_view = cb.sampled_z.sub_offset(0, n_act);
            gpu.softmax_temp_topp_batched_into_f32(
                &cb.logits.sub_offset(0, n_act * vocab),
                &probs_view,
                &tau_view,
                &z_view,
                vocab,
                n_act,
                samp.temp,
                samp.top_p,
                samp.top_k,
                samp.min_p,
            )?;
            gpu.download_f32_into(&probs_view, &mut probs_host)?;
            gpu.download_f32_into(&tau_view, &mut tau_host)?;
            gpu.download_f32_into(&z_view, &mut z_host)?;
            for (col, &g) in act.iter().enumerate() {
                let st = &mut *lanes[group[g]].state;
                // The retained q row outlives the staging buffer: copy it.
                let mut probs = probs_host[col * vocab..(col + 1) * vocab].to_vec();
                // Raw (untruncated) top prob, read only for the p_min cutoff.
                let top_prob = if st.p_min > 0.0 {
                    probs.iter().cloned().fold(0.0f32, f32::max)
                } else {
                    0.0
                };
                apply_topp_trunc(&mut probs, tau_host[col], z_host[col]);
                let u = st.rng.next_uniform_f32();
                let draft_idx = sample_categorical(&probs, u) as usize;
                assert!(
                    draft_idx < vocab,
                    "draft sample {draft_idx} out of draft vocab {vocab}"
                );
                let o = &mut outs[g];
                o.draft_probs.push(probs[draft_idx]);
                o.candidates.push(draft_idx as u32);
                if o.sampling.penalized() {
                    // Singleton `mtp_draft_phase_inner` appends every actual
                    // pick (even one that truncates the chain below), never a
                    // future candidate.
                    let t0 = MtpPenaltyTiming::start();
                    mtp_penalty_append_draft(
                        gpu,
                        &mut st.penalty,
                        &st.mtp_sample_repeat_buf,
                        draft_idx as u32,
                    )?;
                    st.penalty_timing.finish(t0, 1, 0);
                }
                o.draft_softmaxes.push(probs);
                o.drafts_generated = o.candidates.len();
                if st.p_min > 0.0 && top_prob < st.p_min {
                    // Keep this candidate, stop the chain (singleton breaks
                    // before the t_outs copy).
                    o.chain_truncated = true;
                    continue;
                }
                if k + 1 < n_k[g] {
                    gpu.memcpy_dtod_at_auto(
                        &st.mtp_t_outs.buf,
                        k * dim * 4,
                        &st.mtp_scratch.t_mtp_out.buf,
                        0,
                        dim * 4,
                    )?;
                }
            }
        }
        for (&i, o) in group.iter().zip(outs) {
            out[i] = Some(o);
        }
    }
    for (i, l) in lanes.iter_mut().enumerate() {
        if out[i].is_none() {
            out[i] = Some(mtp_cb_draft(
                gpu,
                weights,
                config,
                head,
                l.state,
                l.cur_pos,
                l.last_committed,
                l.emitted,
                l.k,
            )?);
        }
    }
    Ok(out
        .into_iter()
        .map(|d| d.expect("every lane drafted"))
        .collect())
}

/// One request's window in a CB cycle: its trunk owners, its MTP drafter
/// state, and the `mtp_step` arguments.
pub struct MtpCbLane<'a> {
    pub kv_cache: &'a mut KvCache,
    pub dn_state: &'a mut DeltaNetState,
    pub state: &'a mut MtpSpecState,
    pub cur_pos: usize,
    pub last_committed: u32,
    /// Emitted history ending in `last_committed` (penalty window).
    pub emitted: &'a [u32],
    pub eos_token_id: u32,
    /// Per-window draft budget (`MtpSpeculator` k, already clamped).
    pub k: usize,
}

/// Synced phase wall times of one cycle (zero unless
/// `HIPFIRE_MTP_PHASE_TIMING=1`).
#[derive(Clone, Copy, Debug, Default)]
pub struct MtpCbTiming {
    pub draft_us: f64,
    pub verify_us: f64,
    pub accept_us: f64,
    /// Lanes that took the shared verify / the singleton fallback.
    pub shared_lanes: usize,
    pub singleton_lanes: usize,
}

/// Draft phase of one lane's window: the singleton drafter (penalty window
/// rebuilt from `emitted`, then `mtp_draft_phase_inner`). Touches only the
/// lane's MTP state (head KV rows at and past `cur_pos`, token chain), so a
/// draft that is dropped before verify leaves committed state unchanged.
#[allow(clippy::too_many_arguments)]
pub fn mtp_cb_draft(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    emitted: &[u32],
    k: usize,
) -> HipResult<MtpDraftOutput> {
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
    state.penalty.begin_window(emitted);
    mtp_draft_phase_inner(
        gpu,
        weights,
        config,
        head,
        state,
        cur_pos,
        last_committed,
        k,
        false,
    )
}

/// One drafted lane between verify and accept.
pub struct MtpCbVerifyLane<'a> {
    pub kv_cache: &'a mut KvCache,
    pub dn_state: &'a mut DeltaNetState,
    pub state: &'a mut MtpSpecState,
    pub draft: &'a MtpDraftOutput,
    pub eos_token_id: u32,
}

/// A lane after the verify phase.
pub enum MtpCbVerified {
    /// Trunk verified on the shared body, head logits/hidden in the lane's
    /// state: [`mtp_cb_accept`] commits it.
    Pending,
    /// A window the shared trunk cannot take (single verify row) ran the
    /// whole singleton verify/accept/repair; its result.
    Done(MtpSpecResult),
}

/// Can the shared trunk verify this drafted window exactly (the singleton
/// verifies it on the batched-prefill body with a rollback tape)? `pack_cap`
/// is the chunk's packing cap (`<= 63`); a window never exceeds one
/// request's 63 rows.
fn shared_verify_ok(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pack_cap: usize,
    lane: &MtpCbVerifyLane<'_>,
) -> bool {
    let n = lane.draft.n_verify();
    n >= 2
        && n <= pack_cap.min(MULTI_CHUNK_MAX_LANE_ROWS)
        && n <= lane.state.trunk_gdn_tape.max_n
        && qwen35::prefill_batch_pbs_eligible(
            weights,
            config,
            lane.dn_state,
            n,
            gpu.arch.as_str(),
            true,
        )
}

/// Verify phase over drafted lanes: DeltaNet snapshot, ONE shared trunk
/// forward (chunks of whole lanes within `cb.max_rows`, never above 63: the
/// product route) and ONE verify head per chunk; each lane's hidden/logit
/// rows land in its own state. Writes trunk KV/DeltaNet (a failure here
/// leaves the lanes untrusted).
pub fn mtp_cb_verify(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    scratch: &Qwen35Scratch,
    cb: &MtpCbScratch,
    lanes: &mut [MtpCbVerifyLane<'_>],
) -> HipResult<Vec<MtpCbVerified>> {
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
    let pack_cap = cb.max_rows.min(MTP_CB_MAX_ROWS);
    let shared: Vec<bool> = lanes
        .iter()
        .map(|l| shared_verify_ok(gpu, weights, config, pack_cap, l))
        .collect();
    let mut out: Vec<MtpCbVerified> = (0..lanes.len()).map(|_| MtpCbVerified::Pending).collect();
    let dim = config.dim;
    let vocab = config.vocab_size;
    let mut idx: Vec<usize> = (0..lanes.len()).filter(|&i| shared[i]).collect();
    while !idx.is_empty() {
        let mut rows = 0usize;
        let mut take = 0usize;
        while take < idx.len() && rows + lanes[idx[take]].draft.n_verify() <= pack_cap {
            rows += lanes[idx[take]].draft.n_verify();
            take += 1;
        }
        let chunk: Vec<usize> = idx.drain(..take).collect();
        let verify_tokens: Vec<Vec<u32>> = chunk
            .iter()
            .map(|&i| lanes[i].draft.verify_tokens())
            .collect();
        for &i in &chunk {
            let lane = &mut lanes[i];
            lane.state.trunk_snap.save_from(lane.dn_state, gpu)?;
        }
        // PM FP8 verify scope (`HIPFIRE_CB_VERIFY_FP8`): trunk + verify head run the
        // one E4M3 verify arithmetic at every row count, so this shared chunk equals
        // each lane's singleton verify (same admission: the lanes share target and
        // KV tier); closed on every path below.
        let prev = fp8_verify_open(gpu, weights, config, &*lanes[chunk[0]].kv_cache)?;
        let fwd = {
            let mut reqs: Vec<MultiChunkRequest<'_>> = Vec::with_capacity(chunk.len());
            let mut rest: &mut [MtpCbVerifyLane<'_>] = lanes;
            let mut base = 0usize;
            for (j, &i) in chunk.iter().enumerate() {
                let (_, tail) = std::mem::take(&mut rest).split_at_mut(i - base);
                let (lane, after) = tail.split_first_mut().expect("lane index in range");
                rest = after;
                base = i + 1;
                reqs.push(MultiChunkRequest {
                    tokens: &verify_tokens[j],
                    start_pos: lane.draft.cur_pos,
                    kv_cache: &mut *lane.kv_cache,
                    dn_state: &mut *lane.dn_state,
                    gdn_tape: Some(&lane.state.trunk_gdn_tape),
                    fusion: crate::qwen35::DflashFusionCtx::Off,
                    hidden_rb: None,
                    dn_state_src: None,
                });
            }
            forward_prefill_batch_multi(
                gpu,
                weights,
                config,
                scratch,
                &cb.trunk,
                &mut reqs,
                Some(&cb.hidden),
            )
        };
        let logits = cb.logits.sub_offset(0, rows * vocab);
        let head = fwd.and_then(|()| {
            mtp_trunk_verify_lm_head(
                gpu,
                &weights.output,
                &cb.hidden,
                &cb.rot,
                &logits,
                rows,
                dim,
                vocab,
            )
        });
        gpu.fp8_verify_exit(prev);
        head?;
        let mut row = 0usize;
        for &i in &chunk {
            let n = lanes[i].draft.n_verify();
            let st = &mut *lanes[i].state;
            gpu.memcpy_dtod_at_auto(
                &st.verify_hidden.buf,
                0,
                &cb.hidden.buf,
                row * dim * 4,
                n * dim * 4,
            )?;
            gpu.memcpy_dtod_at_auto(
                &st.verify_logits.buf,
                0,
                &cb.logits.buf,
                row * vocab * 4,
                n * vocab * 4,
            )?;
            row += n;
        }
    }
    for i in (0..lanes.len()).filter(|&i| !shared[i]) {
        let lane = &mut lanes[i];
        let d = lane.draft;
        out[i] = MtpCbVerified::Done(mtp_shared_verify_accept_rollback_inner(
            gpu,
            weights,
            config,
            lane.kv_cache,
            lane.dn_state,
            scratch,
            lane.state,
            d.cur_pos,
            d.last_committed,
            lane.eos_token_id,
            &d.candidates,
            d.drafts_generated,
            d.chain_truncated,
            false,
            d.use_sampling,
            d.sampling,
            &d.draft_probs,
            &d.draft_softmaxes,
            false,
            d.use_device_token_chain,
        )?);
    }
    Ok(out)
}

/// Accept/repair phase of one [`MtpCbVerified::Pending`] lane: the
/// singleton consumer over its precomputed head logits (greedy/sampled
/// accept, prev_hidden capture, DeltaNet restore + tape replay on a
/// partial accept).
pub fn mtp_cb_accept(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    scratch: &Qwen35Scratch,
    lane: &mut MtpCbVerifyLane<'_>,
) -> HipResult<MtpSpecResult> {
    let d = lane.draft;
    let verify_tokens = d.verify_tokens();
    mtp_accept_and_rollback(
        gpu,
        weights,
        config,
        lane.kv_cache,
        lane.dn_state,
        scratch,
        lane.state,
        d.n_verify(),
        &verify_tokens,
        &d.candidates,
        d.drafts_generated,
        d.chain_truncated,
        d.use_sampling,
        d.sampling,
        &d.draft_probs,
        &d.draft_softmaxes,
        false,
        d.use_device_token_chain,
        true,
        d.cur_pos,
        lane.eos_token_id,
        true,
    )
}

/// One MTP window per lane, verify shared; results in lane order. Each
/// lane's committed tokens and state equal its singleton window's.
#[allow(clippy::too_many_arguments)]
pub fn mtp_cb_cycle(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    scratch: &Qwen35Scratch,
    head: &Qwen35MtpHead,
    cb: &MtpCbScratch,
    lanes: &mut [MtpCbLane<'_>],
) -> HipResult<(Vec<MtpSpecResult>, MtpCbTiming)> {
    let timed = mtp_phase_timing_enabled();
    let mut timing = MtpCbTiming::default();
    let mark = |gpu: &Gpu, slot: &mut f64, t: &mut Instant| -> HipResult<()> {
        if timed {
            gpu.hip.device_synchronize()?;
            *slot += t.elapsed().as_secs_f64() * 1e6;
            *t = Instant::now();
        }
        Ok(())
    };
    let mut t = Instant::now();
    let drafts = {
        let mut dl: Vec<MtpCbDraftLane<'_>> = lanes
            .iter_mut()
            .map(|l| MtpCbDraftLane {
                state: &mut *l.state,
                cur_pos: l.cur_pos,
                last_committed: l.last_committed,
                emitted: l.emitted,
                k: l.k,
            })
            .collect();
        mtp_cb_draft_batched(gpu, weights, config, head, cb, &mut dl)?
    };
    mark(gpu, &mut timing.draft_us, &mut t)?;
    let mut vlanes: Vec<MtpCbVerifyLane<'_>> = lanes
        .iter_mut()
        .zip(&drafts)
        .map(|(l, d)| MtpCbVerifyLane {
            kv_cache: &mut *l.kv_cache,
            dn_state: &mut *l.dn_state,
            state: &mut *l.state,
            draft: d,
            eos_token_id: l.eos_token_id,
        })
        .collect();
    let verified = mtp_cb_verify(gpu, weights, config, scratch, cb, &mut vlanes)?;
    mark(gpu, &mut timing.verify_us, &mut t)?;
    let mut results = Vec::with_capacity(vlanes.len());
    for (lane, v) in vlanes.iter_mut().zip(verified) {
        results.push(match v {
            MtpCbVerified::Done(r) => {
                timing.singleton_lanes += 1;
                r
            }
            MtpCbVerified::Pending => {
                timing.shared_lanes += 1;
                mtp_cb_accept(gpu, weights, config, scratch, lane)?
            }
        });
    }
    mark(gpu, &mut timing.accept_us, &mut t)?;
    Ok((results, timing))
}
