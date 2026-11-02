// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched, host-sync-free twin of [`crate::sampler::sample`].
//!
//! A batched serve step picks one token per lane from `K` independent logits
//! rows. Calling [`crate::sampler::sample`] serially costs, per lane, one
//! sync H2D for the penalty window, one H2D per blocked token and one sync
//! 8-byte D2H. [`sample_batch`] enqueues the same kernels (same penalty
//! prepass, same reducer, same grids/args — see
//! [`Gpu::sample_top_p_pf_enqueue`]) for every lane back to back on one
//! stream and reads every `(token, rng)` result back with ONE D2H. The
//! result is byte-identical per lane to the serial call: token, advanced
//! RNG, and the post-sample logits bytes (penalty prepass / blocked tokens
//! write into the logits row in place).
//!
//! # Exactness argument
//!
//! * Penalty window: the serial path uploads the trailing
//!   `min(repeat_window, repeat_cap)` history tokens and passes
//!   `scope.len()` as the kernel window. The kernel reads the window only
//!   when a penalty is active (`repeat_penalty > 1.0 || presence_penalty >
//!   0.0 || frequency_penalty > 0.0`) and `scope.len() > 0`, so only those
//!   lanes' scopes are uploaded (packed into one pinned stage, one async
//!   H2D); inactive lanes never read theirs.
//! * Blocked tokens: a stream-ordered 4-byte D2D copy of `-INF` per token,
//!   issued before that lane's sampler launches, writes the same bytes the
//!   serial 4-byte H2D wrote.
//! * Fast-stable reducer sentinel: a lane whose fast reducer reports
//!   `token_id == u32::MAX` is re-run on the already-penalized logits with
//!   the exact reducer and its ORIGINAL RNG — the singleton's own fallback.
//! * Shared partials scratch: every lane's partial/finalize pair reuses one
//!   device scratch in stream order, so lanes are serialized by the stream.
//!   Growth (`ensure_sample_partials`) device-synchronizes before freeing
//!   the old buffer, hence it is safe mid-batch and happens at most once per
//!   reducer width.

use crate::sampler::{self, SamplerConfig};
use hip_bridge::{HipError, HipResult};
use rdna_compute::sampling::SampleEnqueued;
use rdna_compute::{Gpu, GpuTensor};
use std::ffi::c_void;

/// One lane to sample: its logits row (mutated in place by blocked tokens
/// and the penalty prepass, exactly like the serial call), its token
/// history, policy and RNG state. The row's vocab is `logits.numel()`.
pub struct BatchSampleJob<'a> {
    pub logits: &'a GpuTensor,
    pub history: &'a [u32],
    pub cfg: &'a SamplerConfig,
    pub rng_state: u32,
}

/// A lane's pick and its advanced RNG (the caller's `&mut rng_state` of the
/// serial [`crate::sampler::sample`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchSamplePick {
    pub token_id: u32,
    pub advanced_rng: u32,
}

/// Per-lane scope bookkeeping: `(offset in packed stage, scope len, uploaded)`.
type ScopeSlot = (usize, usize, bool);

/// Preallocated sampler state: device result slots, a packed repeat slab,
/// the `-INF` constant for blocked tokens and the pinned host stage. Owned
/// by the batch store; freed with [`BatchSamplerArena::free_gpu`].
pub struct BatchSamplerArena {
    max_lanes: usize,
    repeat_cap: usize,
    /// `[2 * max_lanes]` u32 `(token, rng)` slots.
    results: GpuTensor,
    /// `[max_lanes * max(repeat_cap, 1)]` u32 packed penalty scopes.
    slab: GpuTensor,
    /// One f32 `-INF`.
    neg_inf: GpuTensor,
    /// Pinned host stage: `slab_words` scope words, then `2 * max_lanes`
    /// result words.
    host: *mut c_void,
    slab_words: usize,
    scopes: Vec<ScopeSlot>,
    kinds: Vec<Option<SampleEnqueued>>,
    picks: Vec<BatchSamplePick>,
}

// SAFETY: `host` is the arena's exclusively owned pinned allocation, touched
// only through `&mut BatchSamplerArena`; the tensors are device handles.
unsafe impl Send for BatchSamplerArena {}

impl BatchSamplerArena {
    /// `max_lanes` bounds one batch; `repeat_cap` is the serial singleton's
    /// penalty-window capacity in tokens (`repeat_buf.buf.size() / 4`).
    pub fn new(gpu: &mut Gpu, max_lanes: usize, repeat_cap: usize) -> HipResult<Self> {
        if max_lanes == 0 {
            return Err(HipError::new(0, "BatchSamplerArena: max_lanes must be > 0"));
        }
        gpu.bind_thread()?;
        let slab_words = max_lanes * repeat_cap.max(1);
        let host_words = slab_words + 2 * max_lanes;
        let results = gpu.zeros(&[2 * max_lanes], rdna_compute::DType::F32)?;
        let slab = match gpu.zeros(&[slab_words], rdna_compute::DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(results);
                return Err(e);
            }
        };
        let neg_inf = match gpu.upload_f32(&[f32::NEG_INFINITY], &[1]) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(results);
                let _ = gpu.free_tensor(slab);
                return Err(e);
            }
        };
        let host = match gpu.hip.host_malloc(host_words * 4, 0) {
            Ok(p) => p,
            Err(e) => {
                let _ = gpu.free_tensor(results);
                let _ = gpu.free_tensor(slab);
                let _ = gpu.free_tensor(neg_inf);
                return Err(e);
            }
        };
        Ok(Self {
            max_lanes,
            repeat_cap,
            results,
            slab,
            neg_inf,
            host,
            slab_words,
            scopes: Vec::with_capacity(max_lanes),
            kinds: Vec::with_capacity(max_lanes),
            picks: Vec::with_capacity(max_lanes),
        })
    }

    pub fn max_lanes(&self) -> usize {
        self.max_lanes
    }

    pub fn repeat_cap(&self) -> usize {
        self.repeat_cap
    }

    /// Free the device and pinned allocations. No work may be in flight
    /// (every [`sample_batch`] ends with a stream sync).
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.results);
        let _ = gpu.free_tensor(self.slab);
        let _ = gpu.free_tensor(self.neg_inf);
        let _ = gpu.hip.host_free(self.host);
    }
}

#[inline]
fn penalty_active(cfg: &SamplerConfig) -> bool {
    cfg.repeat_penalty > 1.0 || cfg.presence_penalty > 0.0 || cfg.frequency_penalty > 0.0
}

/// Host-asynchronous H2D on the stream the sampler kernels run on.
fn h2d_async(gpu: &Gpu, dst: &GpuTensor, src: &[u8]) -> HipResult<()> {
    match gpu.active_stream.as_ref() {
        Some(stream) => gpu.hip.memcpy_htod_async(&dst.buf, src, stream),
        None => gpu.hip.memcpy_htod_async_default(&dst.buf, src),
    }
}

/// D2H of `dst.len()` bytes from the start of `src`, ordered after every
/// queued kernel (stream sync on the active stream, blocking copy on the
/// legacy stream otherwise). Like `Gpu::download_*`, it then fails if a
/// bounded device wait timed out during the step.
fn d2h_sync(gpu: &Gpu, dst: &mut [u8], src: &GpuTensor) -> HipResult<()> {
    match gpu.active_stream.as_ref() {
        Some(stream) => {
            gpu.hip.memcpy_dtoh_async(dst, &src.buf, Some(stream))?;
            gpu.hip.stream_synchronize(stream)?;
        }
        None => gpu.hip.memcpy_dtoh(dst, &src.buf)?,
    }
    gpu.take_device_wait_error()
}

/// Sample every job. Byte-identical (token, advanced RNG, post-sample logits
/// bytes) to calling [`crate::sampler::sample`] serially per job with a
/// `repeat_buf` of `arena.repeat_cap()` tokens. One H2D (only if some lane
/// has an active penalty scope) and one D2H (plus one more only when a
/// fast-reducer sentinel fallback ran) per call. The returned slice is in
/// job order.
pub fn sample_batch<'a>(
    gpu: &mut Gpu,
    arena: &'a mut BatchSamplerArena,
    jobs: &[BatchSampleJob<'_>],
) -> HipResult<&'a [BatchSamplePick]> {
    let n = jobs.len();
    if n > arena.max_lanes {
        return Err(HipError::new(
            0,
            &format!(
                "sample_batch: {n} jobs exceed the arena's {} lanes",
                arena.max_lanes
            ),
        ));
    }
    arena.picks.clear();
    arena.scopes.clear();
    arena.kinds.clear();
    if n == 0 {
        return Ok(&arena.picks);
    }
    gpu.bind_thread()?;

    // (a) Pack the scopes of lanes whose penalty is active into the pinned
    // stage, then ONE async H2D into the device slab.
    let host_words = arena.host as *mut u32;
    let mut used = 0usize;
    for job in jobs {
        let window = job.cfg.repeat_window.min(arena.repeat_cap);
        let start = job.history.len().saturating_sub(window);
        let scope = &job.history[start..];
        let uploaded = penalty_active(job.cfg) && !scope.is_empty();
        if uploaded {
            // SAFETY: `used + scope.len() <= n * repeat_cap <= slab_words`
            // (each scope is at most `repeat_cap` tokens); the stage is the
            // arena's pinned allocation and the previous call's final sync
            // guarantees no queued copy still reads it.
            unsafe {
                std::ptr::copy_nonoverlapping(scope.as_ptr(), host_words.add(used), scope.len());
            }
            arena.scopes.push((used, scope.len(), true));
            used += scope.len();
        } else {
            arena.scopes.push((0, scope.len(), false));
        }
    }
    if used > 0 {
        // SAFETY: `used` words were just initialized in the live stage.
        let bytes = unsafe { std::slice::from_raw_parts(host_words as *const u8, used * 4) };
        h2d_async(gpu, &arena.slab, bytes)?;
    }

    // (b)+(c) Per lane: blocked tokens, then the enqueue-only sampler into
    // its result slot.
    for (i, job) in jobs.iter().enumerate() {
        let vocab = job.logits.numel();
        for &tok in &job.cfg.blocked_tokens {
            if (tok as usize) < vocab {
                gpu.memcpy_dtod_at_auto(
                    &job.logits.buf,
                    (tok as usize) * 4,
                    &arena.neg_inf.buf,
                    0,
                    4,
                )?;
            }
        }
        let (off, len, uploaded) = arena.scopes[i];
        let repeat_view = if uploaded {
            arena.slab.sub_offset(off, len)
        } else {
            arena.slab.sub_offset(0, 1)
        };
        let slot = arena.results.sub_offset(2 * i, 2);
        let cfg = job.cfg;
        let kind = gpu.sample_top_p_pf_enqueue(
            job.logits,
            &slot,
            &repeat_view,
            vocab,
            cfg.temperature,
            cfg.top_p,
            job.rng_state,
            len,
            cfg.repeat_penalty,
            cfg.presence_penalty,
            cfg.frequency_penalty,
            cfg.top_k,
            cfg.min_p,
        )?;
        match kind {
            Some(k) => arena.kinds.push(Some(k)),
            None => {
                // The parallel sampler is disabled process-wide, so this can
                // only be the first lane: run the whole batch through the
                // serial sampler (exact by construction).
                if i != 0 {
                    return Err(HipError::new(
                        0,
                        "sample_batch: parallel sampler disabled mid-batch",
                    ));
                }
                return sample_serial(gpu, arena, jobs);
            }
        }
    }

    // (d) ONE D2H of every (token, rng) pair.
    let res_off = arena.slab_words;
    // SAFETY: the result region `[res_off, res_off + 2 * max_lanes)` lies in
    // the live pinned stage; `2 * n <= 2 * max_lanes`.
    let res_ptr = unsafe { host_words.add(res_off) };
    {
        // SAFETY: exclusive access to the stage through `&mut arena`.
        let dst = unsafe { std::slice::from_raw_parts_mut(res_ptr as *mut u8, 2 * n * 4) };
        d2h_sync(gpu, dst, &arena.results)?;
    }
    // SAFETY: `2 * n` initialized words just written by the D2H.
    let res = unsafe { std::slice::from_raw_parts(res_ptr, 2 * n) };

    // (e) Sentinel fallback: lanes whose fast reducer could not prove an
    // unambiguous order re-run the exact reducer on the same (already
    // penalized) logits with the ORIGINAL rng.
    let mut lo = usize::MAX;
    let mut hi = 0usize;
    for (i, job) in jobs.iter().enumerate() {
        if arena.kinds[i] == Some(SampleEnqueued::Fast) && res[2 * i] == u32::MAX {
            let slot = arena.results.sub_offset(2 * i, 2);
            gpu.sample_top_p_pf_exact_enqueue(
                job.logits,
                &slot,
                job.logits.numel(),
                job.cfg.temperature,
                job.cfg.top_p,
                job.rng_state,
                job.cfg.top_k,
                job.cfg.min_p,
            )?;
            lo = lo.min(i);
            hi = hi.max(i);
        }
    }
    if lo != usize::MAX {
        let view = arena.results.sub_offset(2 * lo, 2 * (hi - lo + 1));
        // SAFETY: slots `lo..=hi` lie within the stage's result region.
        let dst = unsafe {
            std::slice::from_raw_parts_mut(res_ptr.add(2 * lo) as *mut u8, 2 * (hi - lo + 1) * 4)
        };
        d2h_sync(gpu, dst, &view)?;
    }
    // SAFETY: as above; the slots were refreshed by the second D2H.
    let res = unsafe { std::slice::from_raw_parts(res_ptr, 2 * n) };

    // (f) Picks in job order.
    for i in 0..n {
        arena.picks.push(BatchSamplePick {
            token_id: res[2 * i],
            advanced_rng: res[2 * i + 1],
        });
    }
    Ok(&arena.picks)
}

/// Serial reference path (`HIPFIRE_SAMPLE_PARALLEL=0`): the singleton's own
/// [`sampler::sample`] per job, on the arena's slab/result slot standing in
/// for the scratch `repeat_buf` / `sample_buf`.
fn sample_serial<'a>(
    gpu: &mut Gpu,
    arena: &'a mut BatchSamplerArena,
    jobs: &[BatchSampleJob<'_>],
) -> HipResult<&'a [BatchSamplePick]> {
    let sample_buf = arena.results.sub_offset(0, 2);
    let repeat_buf = arena.slab.sub_offset(0, arena.repeat_cap);
    arena.picks.clear();
    for job in jobs {
        let mut rng = job.rng_state;
        let token_id = sampler::sample(
            gpu,
            job.logits,
            &sample_buf,
            &repeat_buf,
            job.logits.numel(),
            job.history,
            job.cfg,
            &mut rng,
        );
        arena.picks.push(BatchSamplePick {
            token_id,
            advanced_rng: rng,
        });
    }
    Ok(&arena.picks)
}
