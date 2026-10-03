// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MTP speculative decode on a dense tensor-parallel trunk.
//!
//! The drafter is rank-0 work: the MTP head, the trunk embedding table and the
//! trunk lm_head are replicated, so the K-step proposal is the same
//! [`mtp_spec::mtp_draft_phase_inner`] the single-slot and slot-engine paths
//! run, and the accept rule plus `prev_hidden` capture is the same
//! [`mtp_spec::mtp_verify_accept`]. Only the trunk verify forward, the
//! DeltaNet snapshot and the rollback fan out across ranks. Verify runs through
//! the persistent-PBS capture entry point the dense-TP DFlash mesh uses
//! ([`qwen35::forward::forward_prefill_dense_tp_verify_capture`]).

use crate::mtp_head::{self, MtpKvMode, Qwen35MtpHead, Qwen35MtpHeadBatchedScratch};
use crate::mtp_spec::{
    mtp_draft_phase_inner, mtp_verify_accept, MtpSpecResult, MtpSpecState, MtpVerifyAccepted,
};
use crate::qwen35::forward::DenseTpSpecCapture;
use crate::qwen35::{self, DeltaNetState, Qwen35Config, Qwen35Scratch, Qwen35Weights};
use crate::speculative::{DeltaNetSnapshot, GdnTape, HiddenStateRingBuffer};
use hip_bridge::{HipError, HipResult};
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::spec::{SpecAdvance, SpecScratch, SpecTarget};
use hipfire_runtime::tp_shard::ShardConfig;
use rdna_compute::{DType, Gpu, GpuTensor};

const BLOCK_VERIFY_UNSUPPORTED: &str =
    "dense TP: block-verify speculators (n-gram, DFlash, DSpark) are not wired; only the MTP drafter runs on dense TP";

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// The dense-TP trunk as a spec target. Owns its ranks for the request because
/// `SpecTarget: Any` forces `'static`: the loader guard moves the mesh in and
/// takes it back, and the `&mut Gpu` every trait method carries (the daemon's
/// single device handle) is not one of these ranks and stays untouched.
pub struct Qwen35DenseTpTarget {
    pub gpus: Gpus,
    pub shard: ShardConfig,
    pub weights: Vec<Qwen35Weights>,
    pub configs: Vec<Qwen35Config>,
    pub kv_caches: Vec<KvCache>,
    pub dn_states: Vec<DeltaNetState>,
    pub scratches: Vec<Qwen35Scratch>,
    pub eos_token: u32,
    pub ctx_capacity: usize,
}

impl Qwen35DenseTpTarget {
    fn tp(&self) -> usize {
        self.gpus.devices.len()
    }

    fn prefill_chunk(&self) -> usize {
        qwen35::prefill_max_batch_tp(&self.gpus.devices[0], self.tp()).max(1)
    }

    /// Rank-0 logits of the last forwarded position.
    pub fn last_logits(&mut self) -> Result<Vec<f32>, String> {
        let dev = &mut self.gpus.devices[0];
        dev.bind_thread().map_err(|e| e.to_string())?;
        dev.download_f32(&self.scratches[0].logits)
            .map_err(|e| e.to_string())
    }
}

impl SpecTarget for Qwen35DenseTpTarget {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        for rank in 0..self.tp() {
            let dev = &mut self.gpus.devices[rank];
            dev.bind_thread()
                .map_err(|e| format!("dense TP reset bind rank {rank}: {e}"))?;
            self.dn_states[rank]
                .reset(dev)
                .map_err(|e| format!("dense TP reset_recurrent rank {rank}: {e}"))?;
            self.kv_caches[rank].compact_offset = 0;
        }
        Ok(())
    }

    fn retry_reset_eligible(&self) -> bool {
        true
    }

    fn eos_token(&self) -> u32 {
        self.eos_token
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn new_spec_scratch(
        &mut self,
        _gpu: &mut Gpu,
        _block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        Err(BLOCK_VERIFY_UNSUPPORTED.to_string())
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
        let chunk_max = self.prefill_chunk();
        let mut off = 0usize;
        while off < tokens.len() {
            if abort() {
                let _ = self.reset_recurrent(gpu);
                return Ok(SpecAdvance::Aborted);
            }
            let end = (off + chunk_max).min(tokens.len());
            qwen35::forward_prefill_dense_tp(
                &mut self.gpus,
                &self.shard,
                &self.weights,
                &self.configs,
                &tokens[off..end],
                start_pos + off,
                &mut self.kv_caches,
                &mut self.dn_states,
                &self.scratches,
            )
            .map_err(|e| e.to_string())?;
            off = end;
        }
        let logits = self.last_logits()?;
        let last_argmax = argmax(&logits);
        Ok(SpecAdvance::Ready {
            last_argmax,
            last_logits: Some(logits),
        })
    }

    fn verify_block(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        _position: usize,
        _scratch: &mut dyn SpecScratch,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        Err(BLOCK_VERIFY_UNSUPPORTED.to_string())
    }

    fn commit_prefix(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        _accept_len: usize,
        _position: usize,
        _scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        Err(BLOCK_VERIFY_UNSUPPORTED.to_string())
    }
}

/// Verify scratch rank `r > 0` owns. Rank 0 uses `MtpSpecState::trunk_pbs`,
/// `trunk_gdn_tape` and `trunk_snap`, which the shared state already allocates
/// on the drafter's device against rank 0's local config.
struct RankVerifyScratch {
    pbs: qwen35::PrefillBatchScratch,
    tape: GdnTape,
    snap: DeltaNetSnapshot,
}

/// One rank's verify buffers. Mirrors `DenseTpDflashRankState` minus the
/// draft replica: the MTP drafter exists once, on rank 0.
struct DenseTpMtpRank {
    partial: GpuTensor,
    /// The capture entry point takes one ring per rank; MTP extracts no
    /// intermediate layers, so an empty ring allocates nothing.
    hidden_sink: HiddenStateRingBuffer,
    /// `None` on rank 0 (see `RankVerifyScratch`).
    own: Option<RankVerifyScratch>,
    moe_router_logits: bool,
}

/// Rank-0 drafter state plus the per-rank verify scratch.
pub struct MtpTpRuntime {
    pub state: MtpSpecState,
    ranks: Vec<DenseTpMtpRank>,
    /// Unnormed rank-0 residual rows the capture forward copies out; the
    /// post-norm rows the accept rule reads land in `state.verify_hidden`.
    rank0_residual: GpuTensor,
    verify_rows: usize,
}

impl MtpTpRuntime {
    pub fn new(
        target: &mut Qwen35DenseTpTarget,
        head: &Qwen35MtpHead,
        max_n: usize,
        kv_mode: MtpKvMode,
    ) -> Result<Self, String> {
        let tp = target.tp();
        if tp < 2 {
            return Err(format!("dense TP MTP requires at least 2 ranks, got {tp}"));
        }
        let verify_rows = max_n + 1;
        let dim = target.configs[0].dim;

        let state = {
            let dev0 = &mut target.gpus.devices[0];
            dev0.bind_thread().map_err(|e| e.to_string())?;
            MtpSpecState::new_for_components_with_verify_capacity(
                dev0,
                &target.configs[0],
                &target.dn_states[0],
                head,
                max_n,
                max_n,
                kv_mode,
            )
            .map_err(|e| format!("dense TP MTP state: {e}"))?
        };

        let mut ranks: Vec<DenseTpMtpRank> = Vec::with_capacity(tp);
        let mut rank0_residual: Option<GpuTensor> = None;
        let build = (|| -> Result<(), String> {
            for rank in 0..tp {
                let dev = &mut target.gpus.devices[rank];
                dev.bind_thread()
                    .map_err(|e| format!("dense TP MTP bind rank {rank}: {e}"))?;
                let partial = dev
                    .alloc_tensor(&[verify_rows * dim], DType::F32)
                    .map_err(|e| format!("dense TP MTP partial rank {rank}: {e}"))?;
                let hidden_sink = match HiddenStateRingBuffer::new_for_layers(
                    dev,
                    &[],
                    dim,
                    verify_rows,
                    verify_rows,
                ) {
                    Ok(h) => h,
                    Err(e) => {
                        let _ = dev.free_tensor(partial);
                        return Err(format!("dense TP MTP hidden sink rank {rank}: {e}"));
                    }
                };
                let (own, moe_router_logits) = if rank == 0 {
                    (None, state.trunk_pbs.moe_router_logits_batch.is_some())
                } else {
                    let scratch = (|| -> Result<RankVerifyScratch, String> {
                        let pbs = qwen35::PrefillBatchScratch::new(
                            dev,
                            &target.configs[rank],
                            verify_rows,
                        )
                        .map_err(|e| format!("dense TP MTP PBS rank {rank}: {e}"))?;
                        let tape = match GdnTape::new_for_config(
                            dev,
                            &target.configs[rank],
                            verify_rows,
                        ) {
                            Ok(t) => t,
                            Err(e) => {
                                let _ = pbs.free_gpu(dev);
                                return Err(format!("dense TP MTP tape rank {rank}: {e}"));
                            }
                        };
                        let snap = match DeltaNetSnapshot::new_for(dev, &target.dn_states[rank]) {
                            Ok(s) => s,
                            Err(e) => {
                                let _ = pbs.free_gpu(dev);
                                tape.free_gpu(dev);
                                return Err(format!("dense TP MTP snapshot rank {rank}: {e}"));
                            }
                        };
                        Ok(RankVerifyScratch { pbs, tape, snap })
                    })();
                    match scratch {
                        Ok(s) => {
                            let moe = s.pbs.moe_router_logits_batch.is_some();
                            (Some(s), moe)
                        }
                        Err(e) => {
                            let _ = dev.free_tensor(partial);
                            hidden_sink.free_gpu(dev);
                            return Err(e);
                        }
                    }
                };
                ranks.push(DenseTpMtpRank {
                    partial,
                    hidden_sink,
                    own,
                    moe_router_logits,
                });
                if rank == 0 {
                    rank0_residual = Some(
                        dev.alloc_tensor(&[verify_rows * dim], DType::F32)
                            .map_err(|e| format!("dense TP MTP residual: {e}"))?,
                    );
                }
            }
            Ok(())
        })();
        if let Err(e) = build {
            free_ranks(&mut target.gpus, ranks);
            let dev0 = &mut target.gpus.devices[0];
            let _ = dev0.bind_thread();
            if let Some(res) = rank0_residual {
                let _ = dev0.free_tensor(res);
            }
            state.free_gpu(dev0);
            return Err(e);
        }

        Ok(Self {
            state,
            ranks,
            rank0_residual: rank0_residual.expect("rank 0 residual built above"),
            verify_rows,
        })
    }

    /// Per-rank twin of the single-slot `state.trunk_snap` restore for the
    /// terminal-prefix repair.
    pub fn restore_last_window(&self, target: &mut Qwen35DenseTpTarget) -> Result<(), String> {
        for (rank, r) in self.ranks.iter().enumerate() {
            let dev = &mut target.gpus.devices[rank];
            dev.bind_thread()
                .map_err(|e| format!("dense TP MTP repair bind rank {rank}: {e}"))?;
            rank_snap(&self.state, r)
                .restore_to(&mut target.dn_states[rank], dev)
                .map_err(|e| format!("dense TP MTP repair restore rank {rank}: {e}"))?;
        }
        Ok(())
    }

    pub fn free_on_ranks(self, gpus: &mut Gpus) {
        let Self {
            state,
            ranks,
            rank0_residual,
            verify_rows: _,
        } = self;
        free_ranks(gpus, ranks);
        if let Some(dev0) = gpus.devices.first_mut() {
            let _ = dev0.bind_thread();
            let _ = dev0.free_tensor(rank0_residual);
            state.free_gpu(dev0);
        }
    }
}

fn rank_snap<'a>(state: &'a MtpSpecState, r: &'a DenseTpMtpRank) -> &'a DeltaNetSnapshot {
    match r.own.as_ref() {
        Some(own) => &own.snap,
        None => &state.trunk_snap,
    }
}

fn free_ranks(gpus: &mut Gpus, ranks: Vec<DenseTpMtpRank>) {
    for (rank, r) in ranks.into_iter().enumerate() {
        let Some(dev) = gpus.devices.get_mut(rank) else {
            continue;
        };
        let _ = dev.bind_thread();
        let DenseTpMtpRank {
            partial,
            hidden_sink,
            own,
            moe_router_logits: _,
        } = r;
        let _ = dev.free_tensor(partial);
        hidden_sink.free_gpu(dev);
        if let Some(RankVerifyScratch { pbs, tape, snap }) = own {
            let _ = pbs.free_gpu(dev);
            tape.free_gpu(dev);
            snap.free_gpu(dev);
        }
    }
}

/// Save every rank's DeltaNet state before the verify forward advances it.
fn snapshot_ranks(
    target: &mut Qwen35DenseTpTarget,
    state: &mut MtpSpecState,
    ranks: &mut [DenseTpMtpRank],
) -> HipResult<()> {
    for (rank, r) in ranks.iter_mut().enumerate() {
        let dev = &mut target.gpus.devices[rank];
        dev.bind_thread()?;
        let snap = match r.own.as_mut() {
            Some(own) => &mut own.snap,
            None => &mut state.trunk_snap,
        };
        snap.save_from(&target.dn_states[rank], dev)?;
    }
    Ok(())
}

/// Trunk verify across ranks. Writes the post-norm hidden of every row into
/// `state.verify_hidden` (rank 0) and, when `capture_tape`, each rank's
/// DeltaNet innovations into its tape.
fn verify_forward_ranks(
    target: &mut Qwen35DenseTpTarget,
    state: &mut MtpSpecState,
    ranks: &mut [DenseTpMtpRank],
    rank0_residual: &GpuTensor,
    tokens: &[u32],
    cur_pos: usize,
    capture_tape: bool,
) -> HipResult<()> {
    let tp = target.tp();
    let n = tokens.len();
    let dim = target.configs[0].dim;
    let required = cur_pos
        .checked_add(n)
        .ok_or_else(|| HipError::new(0, "dense TP MTP verify KV range overflow"))?;
    for rank in 0..tp {
        target.gpus.devices[rank].bind_thread()?;
        target.kv_caches[rank].ensure_mapped_capacity(&mut target.gpus.devices[rank], required)?;
    }
    let post_norm = state.verify_hidden.sub_offset(0, n * dim);
    let rank0_pbs = &state.trunk_pbs;
    let mut rank0_tape = Some(&mut state.trunk_gdn_tape);
    let mut pbs_refs: Vec<&qwen35::PrefillBatchScratch> = Vec::with_capacity(tp);
    let mut partial_refs: Vec<&GpuTensor> = Vec::with_capacity(tp);
    let mut caps: Vec<DenseTpSpecCapture<'_>> = Vec::with_capacity(tp);
    for r in ranks.iter_mut() {
        let DenseTpMtpRank {
            partial,
            hidden_sink,
            own,
            moe_router_logits: _,
        } = r;
        partial_refs.push(&*partial);
        let tape = match own.as_mut() {
            Some(own) => {
                pbs_refs.push(&own.pbs);
                Some(&mut own.tape)
            }
            None => {
                pbs_refs.push(rank0_pbs);
                rank0_tape.take()
            }
        };
        caps.push(DenseTpSpecCapture {
            hidden: hidden_sink,
            tape: if capture_tape { tape } else { None },
        });
    }
    qwen35::forward::forward_prefill_dense_tp_verify_capture(
        &mut target.gpus,
        &target.shard,
        &target.weights,
        &target.configs,
        tokens,
        cur_pos,
        &mut target.kv_caches,
        &mut target.dn_states,
        &target.scratches,
        &pbs_refs,
        &partial_refs,
        &mut caps,
        rank0_residual,
        Some(&post_norm),
    )
}

/// Rewind every rank to the accepted prefix: restore the snapshot, then replay
/// the first `advance` rows from the tape, or re-forward them when the verify
/// ran without one.
fn rollback_ranks(
    target: &mut Qwen35DenseTpTarget,
    state: &mut MtpSpecState,
    ranks: &mut [DenseTpMtpRank],
    verify_tokens: &[u32],
    advance: usize,
    cur_pos: usize,
    tape_captured: bool,
) -> HipResult<()> {
    for (rank, r) in ranks.iter().enumerate() {
        let dev = &mut target.gpus.devices[rank];
        dev.bind_thread()?;
        rank_snap(state, r).restore_to(&mut target.dn_states[rank], dev)?;
    }
    if tape_captured {
        for (rank, r) in ranks.iter_mut().enumerate() {
            let dev = &mut target.gpus.devices[rank];
            dev.bind_thread()?;
            let tape = match r.own.as_mut() {
                Some(own) => &mut own.tape,
                None => &mut state.trunk_gdn_tape,
            };
            tape.replay_gdn(
                dev,
                &target.weights[rank],
                &target.configs[rank],
                &mut target.dn_states[rank],
                advance,
            )?;
        }
        Ok(())
    } else {
        qwen35::forward_prefill_dense_tp(
            &mut target.gpus,
            &target.shard,
            &target.weights,
            &target.configs,
            &verify_tokens[..advance],
            cur_pos,
            &mut target.kv_caches,
            &mut target.dn_states,
            &target.scratches,
        )
    }
}

/// One MTP window on dense TP: draft on rank 0 ([`mtp_draft_phase_inner`]),
/// verify across ranks, accept on rank 0 ([`mtp_verify_accept`]), then rewind
/// every rank to the accepted prefix.
pub fn spec_step_mtp_dense_tp(
    target: &mut Qwen35DenseTpTarget,
    head: &Qwen35MtpHead,
    rt: &mut MtpTpRuntime,
    cur_pos: usize,
    last_committed: u32,
    eos_token_id: u32,
    k: usize,
) -> Result<MtpSpecResult, String> {
    let max_n = k.min(rt.state.max_n);
    if max_n + 1 > rt.verify_rows {
        return Err(format!(
            "dense TP MTP verify window {} exceeds the allocated {} rows",
            max_n + 1,
            rt.verify_rows
        ));
    }
    let dim = target.configs[0].dim;
    let vocab = target.configs[0].vocab_size;
    // Peer-direct reduction needs a live stream on every rank; without one the
    // rooted fallback runs instead (slower, same arithmetic order).
    for dev in target.gpus.devices.iter_mut() {
        dev.bind_thread().map_err(|e| e.to_string())?;
        if dev.active_stream.is_none() {
            dev.active_stream = Some(dev.hip.stream_create().map_err(|e| e.to_string())?);
        }
    }

    let MtpTpRuntime {
        state,
        ranks,
        rank0_residual,
        ..
    } = rt;
    let draft = {
        let dev0 = &mut target.gpus.devices[0];
        dev0.bind_thread().map_err(|e| e.to_string())?;
        // No proposal graph: a rank-0 capture would race the dense-TP graph
        // segments this process also captures; the mesh stays eager.
        mtp_draft_phase_inner(
            dev0,
            &target.weights[0],
            &target.configs[0],
            head,
            state,
            cur_pos,
            last_committed,
            max_n,
            /* skip_proposal_graph */ true,
        )
        .map_err(|e| format!("dense TP MTP draft: {e}"))?
    };

    let verify_tokens = draft.verify_tokens();
    let n_verify = verify_tokens.len();
    debug_assert_eq!(n_verify, draft.n_verify());
    let step = (|| -> HipResult<MtpSpecResult> {
        snapshot_ranks(target, state, ranks)?;
        let tape_captured = (0..ranks.len()).all(|rank| {
            qwen35::prefill_batch_pbs_eligible(
                &target.weights[rank],
                &target.configs[rank],
                &target.dn_states[rank],
                n_verify,
                target.gpus.devices[rank].arch.as_str(),
                ranks[rank].moe_router_logits,
            )
        });
        verify_forward_ranks(
            target,
            state,
            ranks,
            rank0_residual,
            &verify_tokens,
            cur_pos,
            tape_captured,
        )?;

        let MtpVerifyAccepted {
            committed,
            accept_count,
            hit_eos,
        } = {
            let dev0 = &mut target.gpus.devices[0];
            dev0.bind_thread()?;
            mtp_verify_accept(
                dev0,
                &target.weights[0].output,
                dim,
                vocab,
                state,
                n_verify,
                &draft.candidates,
                draft.drafts_generated,
                draft.use_sampling,
                draft.sampling,
                &draft.draft_probs,
                &draft.draft_softmaxes,
                false,
                draft.use_device_token_chain,
                cur_pos,
                eos_token_id,
            )?
        };
        let advance = committed.len();
        let full_accept_no_eos = advance == draft.drafts_generated + 1 && !hit_eos;
        if !full_accept_no_eos {
            rollback_ranks(
                target,
                state,
                ranks,
                &verify_tokens,
                advance,
                cur_pos,
                tape_captured,
            )?;
        }
        Ok(MtpSpecResult {
            committed,
            accept_count,
            hit_eos,
            advance,
            drafts_generated: draft.drafts_generated,
            chain_truncated: draft.chain_truncated,
            replay_skipped: full_accept_no_eos,
        })
    })();
    step.map_err(|e| format!("dense TP MTP verify: {e}"))
}

/// Trunk prefill across ranks with the MTP head's private KV filled over the
/// same rows, mirroring `mtp_spec::prefill_trunk_and_mtp_cache`: the head pairs
/// token `p` with the trunk hidden of position `p - 1`, row 0 is
/// `state.prev_hidden` when it holds `start_pos - 1` and zero otherwise, and
/// `prev_hidden` ends on the last prompt row.
pub fn prefill_trunk_and_mtp_cache_dense_tp(
    target: &mut Qwen35DenseTpTarget,
    head: &Qwen35MtpHead,
    rt: &mut MtpTpRuntime,
    prompt_tokens: &[u32],
    start_pos: usize,
) -> Result<(), String> {
    if prompt_tokens.is_empty() {
        return Ok(());
    }
    let dim = target.configs[0].dim;
    let dim_bytes = dim * 4;
    let chunk_max = target.prefill_chunk().min(prompt_tokens.len()).max(1);
    let state = &mut rt.state;

    let use_batched_fill = mtp_head::mtp_prompt_fill_uses_batched(
        state.mtp_kv.kv_mode,
        &[
            head.weights.eh_proj.gpu_dtype,
            head.weights.wq.gpu_dtype,
            head.weights.wk.gpu_dtype,
            head.weights.wv.gpu_dtype,
        ],
    );
    let (prompt_hidden, mut fill_scratch, fill_rot) = {
        let dev = &mut target.gpus.devices[0];
        dev.bind_thread().map_err(|e| e.to_string())?;
        let hidden = dev
            .alloc_tensor(&[(prompt_tokens.len() + 1) * dim], DType::F32)
            .map_err(|e| format!("dense TP MTP prompt hidden: {e}"))?;
        if use_batched_fill {
            let scratch = match Qwen35MtpHeadBatchedScratch::new(dev, &head.config, chunk_max) {
                Ok(s) => s,
                Err(e) => {
                    let _ = dev.free_tensor(hidden);
                    return Err(format!("dense TP MTP fill scratch: {e}"));
                }
            };
            let widest_k = (2 * dim)
                .max(head.config.n_ff)
                .max(dim)
                .max(head.config.n_head * head.config.head_dim);
            match dev.alloc_tensor(&[chunk_max * widest_k], DType::F32) {
                Ok(rot) => (hidden, Some(scratch), Some(rot)),
                Err(e) => {
                    scratch.free_gpu(dev);
                    let _ = dev.free_tensor(hidden);
                    return Err(format!("dense TP MTP fill rot: {e}"));
                }
            }
        } else {
            (hidden, None, None)
        }
    };

    let run = (|| -> Result<(), String> {
        {
            let dev = &mut target.gpus.devices[0];
            let row0 = if start_pos > 0 && state.prev_hidden_pos == Some(start_pos - 1) {
                dev.hip
                    .memcpy_dtod_at(&prompt_hidden.buf, 0, &state.prev_hidden.buf, 0, dim_bytes)
            } else {
                dev.hip.memcpy_htod(&prompt_hidden.buf, &vec![0u8; dim_bytes])
            };
            row0.map_err(|e| format!("dense TP MTP partner row: {e}"))?;
        }
        let mut off = 0usize;
        while off < prompt_tokens.len() {
            let end = (off + chunk_max).min(prompt_tokens.len());
            let chunk = &prompt_tokens[off..end];
            let chunk_start = start_pos + off;
            // Trunk rows land one below their fill row: row `j + 1` is the
            // hidden of position `start_pos + j`.
            let chunk_hidden = prompt_hidden.sub_offset((off + 1) * dim, chunk.len() * dim);
            qwen35::forward_prefill_dense_tp_with_hidden(
                &mut target.gpus,
                &target.shard,
                &target.weights,
                &target.configs,
                chunk,
                chunk_start,
                &mut target.kv_caches,
                &mut target.dn_states,
                &target.scratches,
                Some(&chunk_hidden),
            )
            .map_err(|e| format!("dense TP MTP prompt trunk: {e}"))?;

            let dev = &mut target.gpus.devices[0];
            dev.bind_thread().map_err(|e| e.to_string())?;
            let fill_hidden = prompt_hidden.sub_offset(off * dim, chunk.len() * dim);
            match (fill_scratch.as_mut(), fill_rot.as_ref()) {
                (Some(scratch), Some(rot)) => {
                    let positions: Vec<i32> = (chunk_start..chunk_start + chunk.len())
                        .map(|p| p as i32)
                        .collect();
                    mtp_head::mtp_head_forward_block_batched(
                        dev,
                        head,
                        scratch,
                        &mut state.mtp_kv,
                        chunk,
                        &fill_hidden,
                        &positions,
                        chunk.len(),
                        &target.weights[0],
                        Some(rot),
                        true,
                    )
                    .map_err(|e| format!("dense TP MTP batched head fill: {e}"))?;
                }
                _ => {
                    for (i, &token) in chunk.iter().enumerate() {
                        let hidden_row = prompt_hidden.sub_offset((off + i) * dim, dim);
                        mtp_head::mtp_head_forward_block_only(
                            dev,
                            head,
                            &state.mtp_scratch,
                            &mut state.mtp_kv,
                            token,
                            &hidden_row,
                            None,
                            chunk_start + i,
                            &target.weights[0],
                        )
                        .map_err(|e| format!("dense TP MTP head fill: {e}"))?;
                    }
                }
            }
            off = end;
        }
        let dev = &mut target.gpus.devices[0];
        dev.bind_thread().map_err(|e| e.to_string())?;
        let last = prompt_tokens.len() - 1;
        dev.hip
            .memcpy_dtod_at(
                &state.prev_hidden.buf,
                0,
                &prompt_hidden.buf,
                (last + 1) * dim_bytes,
                dim_bytes,
            )
            .map_err(|e| format!("dense TP MTP prev_hidden: {e}"))?;
        state.prev_hidden_pos = Some(start_pos + last);
        Ok(())
    })();

    let dev = &mut target.gpus.devices[0];
    let _ = dev.bind_thread();
    let _ = dev.free_tensor(prompt_hidden);
    if let Some(scratch) = fill_scratch.take() {
        scratch.free_gpu(dev);
    }
    if let Some(rot) = fill_rot {
        let _ = dev.free_tensor(rot);
    }
    run
}
