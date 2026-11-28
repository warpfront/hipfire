// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4's engine lanes ([`MeshLanes`]): one extra KV set per request that
//! decodes beside others. Gemma 4 has no batched-rows decode, so a lane epoch
//! runs each row as its own one-token forward on that lane's KV (whole-step
//! interleaving): the eager decode the legacy loop's captured graph replays
//! byte-identically, with the lane's caches swapped in for the step. The
//! resident caches keep their buffers, so the captured decode graph stays
//! bound to them.
//!
//! A lane is never a conversation record ([`MeshLanes::keeps_records`]):
//! decoded rows are not reused (`bundle.rs`), so a finished lane is freed.
//!
//! On a load with the EAGLE drafter the lanes run speculative windows
//! ([`MeshLanes::windows`]): each lane holds a fork of the drafter (shared
//! weights, its own conditioning hidden), and a window epoch runs each lane's
//! greedy window, the singleton's `mtp_step`, on its swapped-in caches.

use super::mesh_family::device;
use super::{q8_row_bytes, Gemma4Bundle};
use crate::forward;
use crate::mtp::Gemma4Drafter;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::mesh_family::{DecodeOut, LaneWindow, MeshLanes};
use hipfire_runtime::multi_gpu::executor::Result;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::spec::{MtpDrafter, Speculator};
use rdna_compute::{DType, Gpu, GpuTensor};

struct Lane {
    kv_sliding: KvCache,
    kv_full: KvCache,
    n_tokens: usize,
    /// The request holding the lane; `None` = free.
    tag: Option<u64>,
    /// The window lane's drafter fork; `None` on AR lanes.
    drafter: Option<Gemma4Drafter>,
    /// A window lane's pending seed: emitted, not in the KV.
    seed: u32,
}

/// The staged lanes and the device argmax slot their rows share.
pub(super) struct Gemma4Lanes {
    lanes: Vec<Lane>,
    top1: GpuTensor,
    /// Sampled requests may take a lane (`serve.batch_sampled`).
    sampled: bool,
}

impl Gemma4Lanes {
    pub(super) fn free_gpu(self, gpu: &mut Gpu) {
        for lane in self.lanes {
            let _ = lane.kv_sliding.free_gpu(gpu);
            let _ = lane.kv_full.free_gpu(gpu);
            if let Some(drafter) = lane.drafter {
                drafter.free_gpu(gpu);
            }
        }
        let _ = gpu.free_tensor(self.top1);
    }
}

/// A KV set shaped like `kv` (Q8 caches only).
fn alloc_like(gpu: &mut Gpu, kv: &KvCache) -> Result<KvCache> {
    KvCache::new_gpu_q8(gpu, kv.k_gpu.len(), kv.n_kv_heads, kv.head_dim, kv.max_seq)
        .map_err(|e| format!("gemma4 lane KV: {e:?}"))
}

/// Copy the first `rows` positions of every K/V slot of `src` into `dst`.
fn copy_rows(gpu: &mut Gpu, dst: &KvCache, src: &KvCache, rows: usize) -> Result<()> {
    let bytes = rows * q8_row_bytes(src);
    for (d, s) in dst
        .k_gpu
        .iter()
        .chain(&dst.v_gpu)
        .zip(src.k_gpu.iter().chain(&src.v_gpu))
    {
        gpu.memcpy_dtod_at_auto(&d.buf, 0, &s.buf, 0, bytes)
            .map_err(|e| format!("gemma4 lane KV copy: {e:?}"))?;
    }
    Ok(())
}

impl Gemma4Bundle {
    /// Stage `n` engine lanes, each a KV set shaped like the resident caches;
    /// `sampled` admits sampled requests to them. With the load's EAGLE
    /// `drafter` the lanes run its windows. Q8 caches only (a park copies Q8
    /// rows).
    pub fn stage_engine_lanes(
        &mut self,
        gpu: &mut Gpu,
        n: usize,
        sampled: bool,
        drafter: Option<&mut dyn Speculator>,
    ) -> Result<()> {
        if self.lanes.is_some() {
            return Err("gemma4 engine lanes are already staged".into());
        }
        let drafter = match drafter {
            None => None,
            Some(spec) => Some(
                spec.drafter_any_mut()
                    .and_then(|any| any.downcast_mut::<Gemma4Drafter>())
                    .ok_or("gemma4 window lanes take the Gemma 4 EAGLE drafter")?,
            ),
        };
        let (sliding, full) = (&self.state.kv_sliding, &self.state.kv_full);
        if !(sliding.quant_q8 && full.quant_q8) {
            return Err("gemma4 engine lanes need Q8 KV caches".into());
        }
        let top1 = gpu
            .zeros(&[std::mem::size_of::<u32>()], DType::Raw)
            .map_err(|e| format!("gemma4 lane argmax slot: {e:?}"))?;
        let mut staged = Gemma4Lanes {
            lanes: Vec::with_capacity(n),
            top1,
            sampled,
        };
        for _ in 0..n {
            let lane = alloc_like(gpu, sliding).and_then(|kv_sliding| {
                let mut lane = Lane {
                    kv_full: match alloc_like(gpu, full) {
                        Ok(kv_full) => kv_full,
                        Err(error) => {
                            let _ = kv_sliding.free_gpu(gpu);
                            return Err(error);
                        }
                    },
                    kv_sliding,
                    n_tokens: 0,
                    tag: None,
                    drafter: None,
                    seed: 0,
                };
                if let Some(drafter) = drafter.as_deref() {
                    match drafter.fork(gpu) {
                        Ok(fork) => lane.drafter = Some(fork),
                        Err(error) => {
                            let _ = lane.kv_sliding.free_gpu(gpu);
                            let _ = lane.kv_full.free_gpu(gpu);
                            return Err(error);
                        }
                    }
                }
                Ok(lane)
            });
            match lane {
                Ok(lane) => staged.lanes.push(lane),
                Err(error) => {
                    staged.free_gpu(gpu);
                    return Err(error);
                }
            }
        }
        self.lanes = Some(staged);
        Ok(())
    }

    /// Whether [`Self::stage_engine_lanes`] staged lanes on this load.
    pub fn engine_lanes_staged(&self) -> bool {
        self.lanes.is_some()
    }

    /// The staged lanes' `sampled` flag; `None` without lanes.
    pub fn engine_lanes_sampled(&self) -> Option<bool> {
        self.lanes.as_ref().map(|staged| staged.sampled)
    }

    /// Move the resident request, `position` rows in the KV, into a free lane
    /// as `tag`: its first `rows` KV rows are copied, and a window lane's
    /// drafter fork adopts `from`'s window state. The prompt's pool snapshots
    /// publish now (as on a client cancel), and the resident state resets.
    fn park_rows(
        &mut self,
        gpu: &mut Gpu,
        tag: u64,
        position: usize,
        rows: usize,
        seed: u32,
        from: Option<&Gemma4Drafter>,
    ) -> Result<()> {
        if self.state.n_tokens != position {
            return Err(format!(
                "park: the resident state sits at {}, not after its {position} tokens",
                self.state.n_tokens
            ));
        }
        let staged = self.lanes.as_mut().ok_or("no gemma4 engine lanes staged")?;
        let lane = staged
            .lanes
            .iter_mut()
            .find(|lane| lane.tag.is_none())
            .ok_or("no free gemma4 engine lane")?;
        copy_rows(gpu, &lane.kv_sliding, &self.state.kv_sliding, rows)?;
        copy_rows(gpu, &lane.kv_full, &self.state.kv_full, rows)?;
        if let Some(from) = from {
            lane.drafter
                .as_mut()
                .ok_or("these gemma4 lanes run no speculative windows")?
                .adopt_window(gpu, from)?;
        }
        lane.n_tokens = position;
        lane.seed = seed;
        lane.tag = Some(tag);
        if let Some(session) = self.session.as_mut() {
            session.publish_captures();
            session.forget_live();
        }
        self.state.reset();
        Ok(())
    }

    /// [`MeshLanes::step_windows`] with the lanes taken out of the bundle,
    /// which each window borrows as its target.
    fn lane_windows(
        &mut self,
        gpu: &mut Gpu,
        staged: &mut Gemma4Lanes,
        rows: &[(u64, usize)],
    ) -> Result<Vec<LaneWindow>> {
        let resident = self.state.n_tokens;
        let eos = self.eos_tok;
        let mut windows = Vec::with_capacity(rows.len());
        for &(tag, drafts) in rows {
            let lane = staged
                .lanes
                .iter_mut()
                .find(|lane| lane.tag == Some(tag))
                .ok_or_else(|| format!("no gemma4 lane holds request {tag}"))?;
            let drafter = lane
                .drafter
                .as_mut()
                .ok_or("these gemma4 lanes run no speculative windows")?;
            // The singleton's draft count: `MtpSpeculator::step` clamps it
            // to the drafter's K the same way.
            let k = drafts.min(drafter.k());
            std::mem::swap(&mut self.state.kv_sliding, &mut lane.kv_sliding);
            std::mem::swap(&mut self.state.kv_full, &mut lane.kv_full);
            self.state.n_tokens = lane.n_tokens;
            let window = drafter.mtp_step(gpu, self, lane.n_tokens, lane.seed, &[], k, eos, None);
            let advanced = self.state.n_tokens;
            std::mem::swap(&mut self.state.kv_sliding, &mut lane.kv_sliding);
            std::mem::swap(&mut self.state.kv_full, &mut lane.kv_full);
            self.state.n_tokens = resident;
            let window = window?;
            lane.seed = *window
                .committed
                .last()
                .ok_or("a gemma4 lane window committed no token")?;
            lane.n_tokens = advanced;
            windows.push(LaneWindow {
                committed: window.committed,
                drafted: window.drafts_generated,
                accepted: window.accepted,
            });
        }
        Ok(windows)
    }
}

impl MeshLanes for Gemma4Bundle {
    fn lane_capacity(&self) -> usize {
        self.lanes.as_ref().map_or(0, |staged| staged.lanes.len())
    }

    fn free_lanes(&self) -> usize {
        self.lanes.as_ref().map_or(0, |staged| {
            staged
                .lanes
                .iter()
                .filter(|lane| lane.tag.is_none())
                .count()
        })
    }

    /// Every row is its own forward: one epoch steps every decoding lane.
    fn lane_rows(&self) -> usize {
        self.lane_capacity()
    }

    fn keeps_records(&self) -> bool {
        false
    }

    fn park(
        &mut self,
        gpus: &mut Gpus,
        tag: u64,
        prompt: &[u32],
        ran: &[u32],
        seed: u32,
        _remaining: usize,
    ) -> Result<()> {
        let position = prompt.len() + ran.len();
        self.park_rows(device(gpus)?, tag, position, position, seed, None)
    }

    fn windows(&self) -> bool {
        self.lanes
            .as_ref()
            .is_some_and(|staged| staged.lanes.iter().all(|lane| lane.drafter.is_some()))
    }

    /// The request's window state moves into the lane's drafter fork; the
    /// pending seed's row moves with the KV when the next round attends it.
    fn park_window(
        &mut self,
        gpus: &mut Gpus,
        tag: u64,
        prompt: &[u32],
        emitted: &[u32],
        drafter: &mut dyn Speculator,
        _remaining: usize,
    ) -> Result<()> {
        let from = drafter
            .drafter_any_mut()
            .and_then(|any| any.downcast_mut::<Gemma4Drafter>())
            .ok_or("a gemma4 window lane takes the Gemma 4 EAGLE drafter")?;
        let seed = *emitted.last().ok_or("a window lane parks with its seed")?;
        let position = prompt.len() + emitted.len() - 1;
        let rows = position + usize::from(from.seed_row_cached());
        self.park_rows(device(gpus)?, tag, position, rows, seed, Some(from))
    }

    fn step_windows(&mut self, gpus: &mut Gpus, rows: &[(u64, usize)]) -> Result<Vec<LaneWindow>> {
        let gpu = device(gpus)?;
        let mut staged = self.lanes.take().ok_or("no gemma4 engine lanes staged")?;
        let windows = self.lane_windows(gpu, &mut staged, rows);
        self.lanes = Some(staged);
        windows
    }

    fn unpark(
        &mut self,
        _gpus: &mut Gpus,
        tag: u64,
        _tokens: &[u32],
        _drafter: Option<&mut dyn Speculator>,
    ) -> Result<()> {
        Err(format!("gemma4 lane {tag} is not a conversation record"))
    }

    fn step_lanes(&mut self, gpus: &mut Gpus, rows: &[(u64, u32, bool)]) -> Result<Vec<DecodeOut>> {
        let gpu = device(gpus)?;
        let Gemma4Bundle {
            config,
            weights,
            state,
            lanes,
            ..
        } = self;
        let staged = lanes.as_mut().ok_or("no gemma4 engine lanes staged")?;
        let resident = state.n_tokens;
        let mut outs = Vec::with_capacity(rows.len());
        for &(tag, token, logits) in rows {
            let lane = staged
                .lanes
                .iter_mut()
                .find(|lane| lane.tag == Some(tag))
                .ok_or_else(|| format!("no gemma4 lane holds request {tag}"))?;
            std::mem::swap(&mut state.kv_sliding, &mut lane.kv_sliding);
            std::mem::swap(&mut state.kv_full, &mut lane.kv_full);
            let position = lane.n_tokens as u32;
            let out = forward::decode_step_device(config, weights, state, gpu, token, position)
                .and_then(|()| {
                    if logits {
                        return gpu
                            .download_f32(&state.logits)
                            .map(DecodeOut::Logits)
                            .map_err(|e| format!("gemma4 lane logits: {e:?}"));
                    }
                    gpu.argmax_f32_batched(&state.logits, &staged.top1, config.vocab_size, 1)
                        .and_then(|()| gpu.download_raw_bytes(&staged.top1))
                        .map_err(|e| format!("gemma4 lane argmax: {e:?}"))
                        .and_then(|bytes| {
                            bytes
                                .get(..4)
                                .map(|b| {
                                    DecodeOut::Top1(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                                })
                                .ok_or_else(|| "gemma4 lane argmax readback is short".to_string())
                        })
                });
            std::mem::swap(&mut state.kv_sliding, &mut lane.kv_sliding);
            std::mem::swap(&mut state.kv_full, &mut lane.kv_full);
            state.n_tokens = resident;
            outs.push(out?);
            lane.n_tokens += 1;
        }
        Ok(outs)
    }

    fn free_lane(&mut self, tag: u64) {
        if let Some(lane) = self
            .lanes
            .as_mut()
            .and_then(|staged| staged.lanes.iter_mut().find(|lane| lane.tag == Some(tag)))
        {
            lane.tag = None;
            lane.n_tokens = 0;
        }
    }
}
