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

use super::mesh_family::device;
use super::{q8_row_bytes, Gemma4Bundle};
use crate::forward;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::mesh_family::{DecodeOut, MeshLanes};
use hipfire_runtime::multi_gpu::executor::Result;
use hipfire_runtime::multi_gpu::Gpus;
use rdna_compute::{DType, Gpu, GpuTensor};

struct Lane {
    kv_sliding: KvCache,
    kv_full: KvCache,
    n_tokens: usize,
    /// The request holding the lane; `None` = free.
    tag: Option<u64>,
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
    /// `sampled` admits sampled requests to them. Q8 caches only (a park
    /// copies Q8 rows).
    pub fn stage_engine_lanes(&mut self, gpu: &mut Gpu, n: usize, sampled: bool) -> Result<()> {
        if self.lanes.is_some() {
            return Err("gemma4 engine lanes are already staged".into());
        }
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
            let lane =
                alloc_like(gpu, sliding).and_then(|kv_sliding| match alloc_like(gpu, full) {
                    Ok(kv_full) => Ok(Lane {
                        kv_sliding,
                        kv_full,
                        n_tokens: 0,
                        tag: None,
                    }),
                    Err(error) => {
                        let _ = kv_sliding.free_gpu(gpu);
                        Err(error)
                    }
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

    /// Copy the resident rows into a free lane; the prompt's pool snapshots
    /// publish now (as on a client cancel), and the resident state resets.
    fn park(
        &mut self,
        gpus: &mut Gpus,
        tag: u64,
        prompt: &[u32],
        ran: &[u32],
        _seed: u32,
        _remaining: usize,
    ) -> Result<()> {
        let gpu = device(gpus)?;
        let position = prompt.len() + ran.len();
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
        copy_rows(gpu, &lane.kv_sliding, &self.state.kv_sliding, position)?;
        copy_rows(gpu, &lane.kv_full, &self.state.kv_full, position)?;
        lane.n_tokens = position;
        lane.tag = Some(tag);
        if let Some(session) = self.session.as_mut() {
            session.publish_captures();
            session.forget_live();
        }
        self.state.reset();
        Ok(())
    }

    fn unpark(&mut self, _gpus: &mut Gpus, tag: u64, _tokens: &[u32]) -> Result<()> {
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
