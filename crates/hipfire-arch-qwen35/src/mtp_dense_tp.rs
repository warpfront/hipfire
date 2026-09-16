// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MTP speculative decode on a dense tensor-parallel trunk.
//!
//! The drafter is rank-0 work — the MTP head, the trunk embedding table and
//! the trunk lm_head are all replicated — so the K-step proposal is the shared
//! [`mtp_spec::mtp_draft_chain`] and only the verify, the accept-prefix
//! rollback and the DeltaNet replay fan out across ranks. Verify runs through
//! the same persistent-PBS capture entry point the dense-TP DFlash mesh uses
//! ([`qwen35::forward_prefill_dense_tp_verify_capture`]), and the accept rule
//! is [`mtp_spec::mtp_shared_verify_accept_rollback`], unchanged — a forked
//! accept path is exactly how this route lost its acceptance rate before.

use crate::mtp_head::{self, MtpKvMode, Qwen35MtpHead, Qwen35MtpHeadBatchedScratch};
use crate::mtp_spec::{
    mtp_draft_chain, mtp_shared_verify_accept_rollback, mtp_trunk_verify_lm_head, MtpSpecResult,
    MtpSpecState, MtpTrunkBackend,
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

    fn last_logits(&mut self) -> Result<Vec<f32>, String> {
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

/// Verify scratch a rank owns outright. Rank 0 has none of this: its PBS,
/// GDN tape and DeltaNet snapshot are `MtpSpecState::trunk_*`, which the
/// shared state already allocates on the drafter's device.
struct RankVerifyScratch {
    pbs: qwen35::PrefillBatchScratch,
    tape: GdnTape,
    snap: DeltaNetSnapshot,
}

/// One rank's verify scratch. Mirrors `DenseTpDflashRankState` minus the draft
/// replica: the MTP drafter exists once, on rank 0.
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
        if !(2..=5).contains(&tp) {
            return Err(format!("dense TP MTP requires 2..=5 ranks, got {tp}"));
        }
        let verify_rows = max_n + 1;
        let dim = target.configs[0].dim;

        let state = {
            let dev0 = &mut target.gpus.devices[0];
            dev0.bind_thread().map_err(|e| e.to_string())?;
            MtpSpecState::new_for_parts_with_verify_capacity(
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

    /// Per-rank twin of the single-GPU `state.trunk_snap` restore for the
    /// strict-prefix repair.
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

fn rank_snap_mut<'a>(
    state: &'a mut MtpSpecState,
    r: &'a mut DenseTpMtpRank,
) -> &'a mut DeltaNetSnapshot {
    match r.own.as_mut() {
        Some(own) => &mut own.snap,
        None => &mut state.trunk_snap,
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

#[derive(Default)]
struct MeshPhases {
    on: bool,
    snapshot_us: u128,
    verify_us: u128,
    lm_head_us: u128,
    restore_us: u128,
    tape_us: u128,
    replay_us: u128,
    tape_used: bool,
    replayed: bool,
    n_verify: usize,
    multirow: bool,
}

struct DenseTpTrunk<'a> {
    target: &'a mut Qwen35DenseTpTarget,
    ranks: &'a mut [DenseTpMtpRank],
    rank0_residual: &'a GpuTensor,
    phases: MeshPhases,
}

impl DenseTpTrunk<'_> {
    /// Phase timing only means anything if every rank has actually finished.
    fn sync_ranks(&mut self) {
        if !self.phases.on {
            return;
        }
        for dev in self.target.gpus.devices.iter() {
            let _ = dev.bind_thread();
            let _ = dev.hip.device_synchronize();
        }
    }

    /// Runs `f` between two rank syncs and charges its wall time to the phase
    /// counter `pick` selects; the syncs are no-ops unless phases are on.
    fn timed<R>(
        &mut self,
        pick: fn(&mut MeshPhases) -> &mut u128,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.sync_ranks();
        let t0 = std::time::Instant::now();
        let r = f(self);
        self.sync_ranks();
        *pick(&mut self.phases) += t0.elapsed().as_micros();
        r
    }
}

impl MtpTrunkBackend for DenseTpTrunk<'_> {
    fn dim(&self) -> usize {
        self.target.configs[0].dim
    }

    fn vocab(&self) -> usize {
        self.target.configs[0].vocab_size
    }

    fn gpu0(&mut self) -> &mut Gpu {
        &mut self.target.gpus.devices[0]
    }

    fn ensure_active_stream(&mut self) -> HipResult<()> {
        // Peer-direct reduction needs a live stream on every rank; without one
        // the rooted fallback runs instead (slower, same arithmetic order).
        for dev in self.target.gpus.devices.iter_mut() {
            dev.bind_thread()?;
            if dev.active_stream.is_none() {
                dev.active_stream = Some(dev.hip.stream_create()?);
            }
        }
        Ok(())
    }

    fn snapshot_sync(&mut self, state: &mut MtpSpecState) -> HipResult<()> {
        self.timed(
            |p| &mut p.snapshot_us,
            |t| {
                for (rank, r) in t.ranks.iter_mut().enumerate() {
                    let dev = &mut t.target.gpus.devices[rank];
                    dev.bind_thread()?;
                    rank_snap_mut(state, r).save_from(&t.target.dn_states[rank], dev)?;
                }
                Ok(())
            },
        )
    }

    fn await_async_snapshot(&mut self, _state: &mut MtpSpecState) -> HipResult<()> {
        Err(HipError::new(
            0,
            "dense TP MTP takes synchronous per-rank snapshots; overlap is single-GPU only",
        ))
    }

    fn restore_snapshot(&mut self, state: &mut MtpSpecState) -> HipResult<()> {
        self.timed(
            |p| &mut p.restore_us,
            |t| {
                for (rank, r) in t.ranks.iter().enumerate() {
                    let dev = &mut t.target.gpus.devices[rank];
                    dev.bind_thread()?;
                    rank_snap(state, r).restore_to(&mut t.target.dn_states[rank], dev)?;
                }
                Ok(())
            },
        )
    }

    fn tape_eligible(&self, n_verify: usize) -> bool {
        (0..self.ranks.len()).all(|rank| {
            qwen35::prefill_batch_pbs_eligible(
                &self.target.weights[rank],
                &self.target.configs[rank],
                &self.target.dn_states[rank],
                n_verify,
                self.target.gpus.devices[rank].arch.as_str(),
                self.ranks[rank].moe_router_logits,
            )
        })
    }

    fn verify_forward(
        &mut self,
        state: &mut MtpSpecState,
        tokens: &[u32],
        cur_pos: usize,
        capture_tape: bool,
    ) -> HipResult<()> {
        let tp = self.target.gpus.devices.len();
        let n = tokens.len();
        let dim = self.target.configs[0].dim;
        let required = cur_pos
            .checked_add(n)
            .ok_or_else(|| HipError::new(0, "dense TP MTP verify KV range overflow"))?;
        for rank in 0..tp {
            self.target.gpus.devices[rank].bind_thread()?;
            self.target.kv_caches[rank]
                .ensure_mapped_capacity(&mut self.target.gpus.devices[rank], required)?;
        }
        self.phases.n_verify = n;
        self.phases.multirow = crate::qwen35::prefill::q8_multirow_attn_admitted(
            self.target.gpus.devices[0].arch_caps.arch(),
            self.target.kv_caches[0].quant_q8,
            self.target.configs[0].head_dim,
            n,
            cur_pos + n,
            crate::qwen35::prefill::fa_pertoken_min_ctx(),
            false,
            false,
            self.target.gpus.devices[0].graphs.capture_mode,
            self.target.gpus.devices[0].replay.is_recording(),
        );
        let post_norm = state.verify_hidden.sub_offset(0, n * dim);
        let rank0_pbs = &state.trunk_pbs;
        let mut rank0_tape = Some(&mut state.trunk_gdn_tape);
        self.timed(
            |p| &mut p.verify_us,
            |t| {
                let mut pbs_refs: Vec<&qwen35::PrefillBatchScratch> = Vec::with_capacity(tp);
                let mut partial_refs: Vec<&GpuTensor> = Vec::with_capacity(tp);
                let mut caps: Vec<DenseTpSpecCapture<'_>> = Vec::with_capacity(tp);
                for r in t.ranks.iter_mut() {
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
                crate::qwen35::forward::forward_prefill_dense_tp_verify_capture(
                    &mut t.target.gpus,
                    &t.target.shard,
                    &t.target.weights,
                    &t.target.configs,
                    tokens,
                    cur_pos,
                    &mut t.target.kv_caches,
                    &mut t.target.dn_states,
                    &t.target.scratches,
                    &pbs_refs,
                    &partial_refs,
                    &mut caps,
                    t.rank0_residual,
                    Some(&post_norm),
                )
            },
        )
    }

    fn verify_lm_head(&mut self, state: &mut MtpSpecState, n_verify: usize) -> HipResult<()> {
        let dim = self.target.configs[0].dim;
        let vocab = self.target.configs[0].vocab_size;
        let logits_view = state.verify_logits.sub_offset(0, n_verify * vocab);
        self.timed(
            |p| &mut p.lm_head_us,
            |t| {
                let dev0 = &mut t.target.gpus.devices[0];
                dev0.bind_thread()?;
                mtp_trunk_verify_lm_head(
                    dev0,
                    &t.target.weights[0].output,
                    &state.verify_hidden,
                    &state.verify_rot,
                    &logits_view,
                    n_verify,
                    dim,
                    vocab,
                )
            },
        )
    }

    fn replay_from_tape(&mut self, state: &mut MtpSpecState, advance: usize) -> HipResult<()> {
        self.phases.tape_used = true;
        self.timed(
            |p| &mut p.tape_us,
            |t| {
                for (rank, r) in t.ranks.iter_mut().enumerate() {
                    let dev = &mut t.target.gpus.devices[rank];
                    dev.bind_thread()?;
                    let tape = match r.own.as_mut() {
                        Some(own) => &mut own.tape,
                        None => &mut state.trunk_gdn_tape,
                    };
                    tape.replay_gdn(
                        dev,
                        &t.target.weights[rank],
                        &t.target.configs[rank],
                        &mut t.target.dn_states[rank],
                        advance,
                    )?;
                }
                Ok(())
            },
        )
    }

    fn replay_tokens(
        &mut self,
        _state: &mut MtpSpecState,
        tokens: &[u32],
        cur_pos: usize,
    ) -> HipResult<()> {
        self.phases.replayed = true;
        self.timed(
            |p| &mut p.replay_us,
            |t| {
                qwen35::forward_prefill_dense_tp(
                    &mut t.target.gpus,
                    &t.target.shard,
                    &t.target.weights,
                    &t.target.configs,
                    tokens,
                    cur_pos,
                    &mut t.target.kv_caches,
                    &mut t.target.dn_states,
                    &t.target.scratches,
                )
            },
        )
    }
}

/// One MTP cycle on dense TP: draft on rank 0, verify across ranks, accept and
/// rewind through the shared rule.
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

    let t_draft = std::time::Instant::now();
    let draft = {
        let dev0 = &mut target.gpus.devices[0];
        dev0.bind_thread().map_err(|e| e.to_string())?;
        mtp_draft_chain(
            dev0,
            &target.weights[0],
            head,
            &mut rt.state,
            cur_pos,
            last_committed,
            max_n,
            dim,
            vocab,
            // Rank-0 proposal capture would race the dense-TP AR graph segments
            // this process also captures; the mesh stays eager, like DFlash-TP2.
            false,
        )
        .map_err(|e| format!("dense TP MTP draft: {e}"))?
    };
    let draft_us = t_draft.elapsed().as_micros();

    let MtpTpRuntime {
        state,
        ranks,
        rank0_residual,
        ..
    } = rt;
    let phases_on = hipfire_config::developer_var("HIPFIRE_SPEC_PHASES")
        .ok()
        .as_deref()
        == Some("1");
    let mut trunk = DenseTpTrunk {
        target,
        ranks,
        rank0_residual,
        phases: MeshPhases {
            on: phases_on,
            ..Default::default()
        },
    };
    let out = mtp_shared_verify_accept_rollback(
        &mut trunk,
        state,
        cur_pos,
        last_committed,
        eos_token_id,
        &draft.candidates,
        draft.drafts_generated,
        draft.chain_truncated,
        false,
        draft.use_sampling,
        draft.sampling,
        &draft.draft_probs,
        &draft.draft_softmaxes,
        false,
        draft.use_device_token_chain,
    )
    .map_err(|e| format!("dense TP MTP verify: {e}"));
    if trunk.phases.on {
        let p = &trunk.phases;
        eprintln!(
            "[phase-mtp-tp] K={max_n} n={} multirow={} draft={draft_us}us snapshot={}us verify={}us lm_head={}us restore={}us tape={}us({}) replay={}us({})",
            p.n_verify,
            u8::from(p.multirow),
            p.snapshot_us,
            p.verify_us,
            p.lm_head_us,
            p.restore_us,
            p.tape_us,
            u8::from(p.tape_used),
            p.replay_us,
            u8::from(p.replayed),
        );
    }
    out
}

/// Trunk prefill across ranks with the MTP head's private KV warmed over the
/// same prompt rows, mirroring `mtp_spec::prefill_trunk_and_mtp_cache`.
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

    let use_batched_fill = mtp_head::mtp_prompt_fill_uses_batched(
        rt.state.mtp_kv.kv_mode,
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
            .alloc_tensor(&[prompt_tokens.len() * dim], DType::F32)
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
        let mut off = 0usize;
        while off < prompt_tokens.len() {
            let end = (off + chunk_max).min(prompt_tokens.len());
            let chunk = &prompt_tokens[off..end];
            let chunk_start = start_pos + off;
            let chunk_hidden = prompt_hidden.sub_offset(off * dim, chunk.len() * dim);
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
            match (fill_scratch.as_mut(), fill_rot.as_ref()) {
                (Some(scratch), Some(rot)) => {
                    let positions: Vec<i32> = (chunk_start..chunk_start + chunk.len())
                        .map(|p| p as i32)
                        .collect();
                    mtp_head::mtp_head_forward_block_batched(
                        dev,
                        head,
                        scratch,
                        &mut rt.state.mtp_kv,
                        chunk,
                        &chunk_hidden,
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
                            &rt.state.mtp_scratch,
                            &mut rt.state.mtp_kv,
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
        let last = (prompt_tokens.len() - 1) * dim_bytes;
        dev.hip
            .memcpy_dtod_at(
                &rt.state.prev_hidden.buf,
                0,
                &prompt_hidden.buf,
                last,
                dim_bytes,
            )
            .map_err(|e| format!("dense TP MTP prev_hidden: {e}"))?;
        Ok(())
    })();

    let dev = &mut target.gpus.devices[0];
    let _ = dev.free_tensor(prompt_hidden);
    if let Some(scratch) = fill_scratch.take() {
        scratch.free_gpu(dev);
    }
    if let Some(rot) = fill_rot {
        let _ = dev.free_tensor(rot);
    }
    run
}

/// Rank-0 logits of the last forwarded position.
pub fn rank0_logits(target: &mut Qwen35DenseTpTarget) -> Result<Vec<f32>, String> {
    target.last_logits()
}
