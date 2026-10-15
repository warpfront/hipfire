// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Strided KV compressor of one row (main or indexer flavor).
//!
//! `CompressorProject` writes `kv = wkv · x` and `score = wgate · x`;
//! `KvCompress` then adds `ape[pos % ratio]` to `score` and writes both into a
//! ring of the last `ratio` rows (two windows when the ratio-4 compressor
//! overlaps). On a window boundary the ring is softmax-pooled over `score`
//! into one compressed cache row, RMS-normed, tail-rotated, stored (F32, or F16
//! through an F32 staging row) and the overlap ring shifts by one window.
//!
//! Decode reads every slot (ring slot, commit row or -1, shift) from device
//! state, so a captured graph replays every kernel and the commit kernels
//! no-op off the boundary. The host-gated route (batched prefill fallback)
//! decides the boundary on the host and addresses rows directly.

use rdna_compute::{DType, Gpu, GpuTensor};

use super::{project_prepared, E8SoaKernel};
use crate::types::DispatchError;

fn hip_err(what: impl std::fmt::Display, e: hip_bridge::HipError) -> DispatchError {
    DispatchError::Hip(format!("{what}: {e:?}"))
}

/// This row's `kv` / `score` projections: row `row` of `[rows, proj_dim]`
/// projections a [`CompressorProjectOp`] (or a packed projection) wrote.
pub struct CompressorInput<'a> {
    pub kv: &'a GpuTensor,
    pub score: &'a GpuTensor,
    pub row: usize,
}

/// One compressor projection pair (`wkv`, `wgate`) into `kv` / `score`.
pub struct CompressorProjection<'a> {
    pub wkv: &'a GpuTensor,
    pub wgate: &'a GpuTensor,
    pub kv: &'a GpuTensor,
    pub score: &'a GpuTensor,
    pub proj_dim: usize,
}

/// The compressor projections of the main and (ratio 4) indexer flavor over
/// the prepared input (`tmp` rotated, `tmp_plain` plain).
pub struct CompressorProjectOp<'a> {
    pub main: CompressorProjection<'a>,
    pub indexer: Option<CompressorProjection<'a>>,
    pub tmp: &'a GpuTensor,
    pub tmp_plain: &'a GpuTensor,
    pub hidden: usize,
    pub rows: usize,
    pub layer: usize,
    pub route: CompressorProjectRoute,
}

pub enum CompressorProjectRoute {
    /// One row: `wkv` then `wgate` as prepared projections. The flavors share
    /// the `kv` / `score` scratch, so decode runs one step per flavor.
    Decode { e8_soa: E8SoaKernel },
}

impl CompressorProjectOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let decode = matches!(self.route, CompressorProjectRoute::Decode { .. });
        super::attention::check_rows(self.rows, decode, "compressor projection", self.layer)
    }
}

pub fn execute_compressor_project(
    gpu: &mut Gpu,
    op: &CompressorProjectOp<'_>,
) -> Result<(), DispatchError> {
    let flavors = std::iter::once(&op.main).chain(op.indexer.as_ref());
    match op.route {
        CompressorProjectRoute::Decode { e8_soa } => {
            for p in flavors {
                for (w, y) in [(p.wkv, p.kv), (p.wgate, p.score)] {
                    project_prepared(
                        gpu,
                        w,
                        op.tmp,
                        op.tmp_plain,
                        y,
                        p.proj_dim,
                        op.hidden,
                        e8_soa,
                    )?;
                }
            }
        }
    }
    Ok(())
}

/// How the window boundary is gated.
#[derive(Clone, Copy, Debug)]
pub enum CompressorCommit {
    /// Device slots in `attn_state` (ring, commit or -1, shift). `ring_only`
    /// skips the commit stage on a known non-boundary (gfx942 host gate,
    /// never during capture).
    Device {
        ring_slot: usize,
        commit_slot: usize,
        shift_slot: usize,
        ring_only: bool,
    },
    /// Host-decided: ring row `ring_row`; `commit` is `None` off a window
    /// boundary or past the cache capacity (nothing after the ring write),
    /// else `Some(row)` with the rank-local cache row (`None` when another
    /// rank owns it; the replicated ring still shifts).
    Host {
        ring_row: usize,
        commit: Option<Option<usize>>,
    },
}

/// Tail-RoPE of a committed row.
#[derive(Clone, Copy, Debug)]
pub struct CompressorRope {
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub corr_low: f32,
    pub corr_high: f32,
    pub rope_dim: usize,
}

/// One compressor step of one row.
pub struct KvCompressOp<'a> {
    pub input: CompressorInput<'a>,
    /// `[ratio, proj_dim]` absolute position bias.
    pub ape: &'a GpuTensor,
    pub norm: &'a GpuTensor,
    /// `[state_rows, proj_dim]` rings.
    pub kv_state: &'a GpuTensor,
    pub score_state: &'a GpuTensor,
    /// Compressed cache `[rows, head_dim]` (F32 or F16).
    pub kv_cache: &'a GpuTensor,
    /// F32 staging row for an F16 cache.
    pub staging: Option<&'a GpuTensor>,
    /// `[2 * ratio, head_dim]` concat scratch (overlap only).
    pub concat: Option<(&'a GpuTensor, &'a GpuTensor)>,
    /// Device slot state (`attn_state_buf`); the device route reads its
    /// slots, the ring/APE slot is also read by the device APE add.
    pub attn_state: Option<&'a GpuTensor>,
    /// RoPE position: element `pos_index` of `pos_array`.
    pub pos_array: &'a GpuTensor,
    pub pos_index: usize,
    pub rope: CompressorRope,
    pub commit: CompressorCommit,
    /// APE row on the host route (`pos % ratio`); `None` reads the ring slot
    /// on device.
    pub host_ape_row: Option<usize>,
    pub ratio: usize,
    pub head_dim: usize,
    pub eps: f32,
    /// Indexer flavor (selects the at-slot RoPE symbol on the host route).
    pub indexer: bool,
    pub gfx1151_route: bool,
    /// `HIPFIRE_COMP_DUMP` stage fingerprints at this (position, layer).
    pub debug: Option<CompressorDebug>,
    pub layer: usize,
}

/// Stage fingerprint dump context (diagnostic).
#[derive(Clone, Copy, Debug)]
pub struct CompressorDebug {
    pub position: u32,
    /// Rank-local cache row this position commits into, when it exists.
    pub commit_row: Option<usize>,
}

impl KvCompressOp<'_> {
    fn overlap(&self) -> bool {
        self.ratio == 4
    }

    fn proj_dim(&self) -> usize {
        if self.overlap() {
            2 * self.head_dim
        } else {
            self.head_dim
        }
    }

    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.ratio == 0 || self.head_dim == 0 {
            return Err(DispatchError::Hip("kv compress has empty geometry".into()));
        }
        if self.overlap() && self.concat.is_none() {
            return Err(DispatchError::Hip(format!(
                "comp l{}: overlap concat scratch missing",
                self.layer
            )));
        }
        if self.kv_cache.dtype == DType::F16 && self.staging.is_none() {
            return Err(DispatchError::Hip(format!(
                "comp F16 staging row missing l{}",
                self.layer
            )));
        }
        let needs_state =
            matches!(self.commit, CompressorCommit::Device { .. }) || self.host_ape_row.is_none();
        if needs_state && self.attn_state.is_none() {
            return Err(DispatchError::Hip(format!(
                "comp l{}: attn_state_buf missing (precompute_attn_state must run first)",
                self.layer
            )));
        }
        Ok(())
    }
}

/// Prints one stage fingerprint (L2 norm and head) when debugging.
fn comp_dbg(gpu: &Gpu, op: &KvCompressOp<'_>, name: &str, t: &GpuTensor, n: usize) {
    let Some(debug) = op.debug else {
        return;
    };
    let _ = gpu.hip.device_synchronize();
    if let Ok(v) = gpu.download_f32(t) {
        let l2: f64 = v
            .iter()
            .take(n)
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        let head: Vec<String> = v.iter().take(6).map(|x| format!("{x:.6e}")).collect();
        eprintln!(
            "COMPDUMP l{} pos={} idx={} {name}: l2={l2:.9e} head={}",
            op.layer,
            debug.position,
            op.indexer,
            head.join(",")
        );
    }
}

pub fn execute_kv_compress(gpu: &mut Gpu, op: &KvCompressOp<'_>) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    let layer = op.layer;
    let (ratio, proj_dim) = (op.ratio, op.proj_dim());
    let state_rows = if op.overlap() { 2 * ratio } else { ratio };

    // 1. This row of the `kv = wkv · x`, `score = wgate · x` projections.
    let input = &op.input;
    let owned = (
        input.kv.sub_offset(input.row * proj_dim, proj_dim),
        input.score.sub_offset(input.row * proj_dim, proj_dim),
    );
    let (kv_buf, score_buf) = (&owned.0, &owned.1);

    // 2. score += ape[pos % ratio]: the slot's position bias, which lets the
    //    pool tell window slots apart. The score scratch may be wider than
    //    this flavor's proj_dim; only proj_dim is live.
    let score_view = score_buf.sub_offset(0, proj_dim);
    match op.host_ape_row {
        Some(row) => {
            let ape_row = op.ape.sub_offset(row * proj_dim, proj_dim);
            gpu.add_inplace_f32(&score_view, &ape_row)
                .map_err(|e| hip_err(format!("comp ape add l{layer}"), e))?;
        }
        None => {
            let ring_slot = op
                .attn_state
                .unwrap()
                .sub_offset(if ratio == 4 { 6 } else { 8 }, 1);
            gpu.compressor_add_ape_f32_buf(
                &score_view,
                op.ape,
                &ring_slot,
                proj_dim as i32,
                ratio as i32,
            )
            .map_err(|e| hip_err(format!("comp ape add buf l{layer}"), e))?;
        }
    }
    comp_dbg(gpu, op, "kv_buf(gemv)", kv_buf, proj_dim);
    comp_dbg(gpu, op, "score_buf(gemv+ape)", score_buf, proj_dim);

    let pos = op.pos_array.sub_offset(op.pos_index, 1);
    match op.commit {
        CompressorCommit::Host { ring_row, commit } => execute_host(
            gpu, op, kv_buf, score_buf, &pos, ring_row, commit, state_rows,
        ),
        CompressorCommit::Device {
            ring_slot,
            commit_slot,
            shift_slot,
            ring_only,
        } => {
            let state = op.attn_state.unwrap();
            let ring_slot = state.sub_offset(ring_slot, 1);
            let commit_slot = state.sub_offset(commit_slot, 1);
            let shift_slot = state.sub_offset(shift_slot, 1);
            execute_device(
                gpu,
                op,
                kv_buf,
                score_buf,
                &pos,
                [&ring_slot, &commit_slot, &shift_slot],
                ring_only,
                state_rows,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_host(
    gpu: &mut Gpu,
    op: &KvCompressOp<'_>,
    kv_buf: &GpuTensor,
    score_buf: &GpuTensor,
    pos: &GpuTensor,
    ring_row: usize,
    commit: Option<Option<usize>>,
    state_rows: usize,
) -> Result<(), DispatchError> {
    let layer = op.layer;
    let (ratio, head_dim, proj_dim) = (op.ratio, op.head_dim, op.proj_dim());
    let (kv_state, score_state) = (op.kv_state, op.score_state);
    let kv_dst = kv_state.sub_offset(ring_row * proj_dim, proj_dim);
    let score_dst = score_state.sub_offset(ring_row * proj_dim, proj_dim);
    gpu.memcpy_dtod_auto(&kv_dst.buf, &kv_buf.buf, proj_dim * 4)
        .map_err(|e| hip_err(format!("comp kv-store l{layer}"), e))?;
    gpu.memcpy_dtod_auto(&score_dst.buf, &score_buf.buf, proj_dim * 4)
        .map_err(|e| hip_err(format!("comp score-store l{layer}"), e))?;
    comp_dbg(gpu, op, "kv_state(ring)", kv_state, state_rows * proj_dim);
    comp_dbg(
        gpu,
        op,
        "score_state(ring)",
        score_state,
        state_rows * proj_dim,
    );
    let Some(local_row) = commit else {
        return Ok(());
    };
    // Long-lived cache storage is owner-only on the exact gfx1201 TP route.
    // The rings stay replicated and shift on every rank.
    if let Some(row) = local_row {
        let kv_cache_slot = op.kv_cache.sub_offset(row * head_dim, head_dim);
        let cache_is_f16 = op.kv_cache.dtype == DType::F16;
        let staged;
        let commit_f32 = if cache_is_f16 {
            staged = op.staging.unwrap().sub_offset(0, head_dim);
            &staged
        } else {
            &kv_cache_slot
        };
        if let Some((concat_kv, concat_score)) = op.concat {
            gpu.compressor_overlap_concat_f32(kv_state, concat_kv, ratio as i32, head_dim as i32)
                .map_err(|e| hip_err(format!("comp concat_kv l{layer}"), e))?;
            gpu.compressor_overlap_concat_f32(
                score_state,
                concat_score,
                ratio as i32,
                head_dim as i32,
            )
            .map_err(|e| hip_err(format!("comp concat_score l{layer}"), e))?;
            comp_dbg(gpu, op, "concat_kv", concat_kv, 2 * ratio * head_dim);
            comp_dbg(gpu, op, "concat_score", concat_score, 2 * ratio * head_dim);
            gpu.compressor_softmax_pool_f32(
                concat_kv,
                concat_score,
                commit_f32,
                (2 * ratio) as i32,
                head_dim as i32,
            )
            .map_err(|e| hip_err(format!("comp pool l{layer}"), e))?;
        } else {
            gpu.compressor_softmax_pool_f32(
                kv_state,
                score_state,
                commit_f32,
                ratio as i32,
                head_dim as i32,
            )
            .map_err(|e| hip_err(format!("comp pool no-overlap l{layer}"), e))?;
        }
        comp_dbg(gpu, op, "kv_cache(pool)", commit_f32, head_dim);
        gpu.rmsnorm_f32(commit_f32, op.norm, commit_f32, op.eps)
            .map_err(|e| hip_err(format!("comp rmsnorm l{layer}"), e))?;
        comp_dbg(gpu, op, "kv_cache(rmsnorm)", commit_f32, head_dim);
        let r = op.rope;
        if cache_is_f16 {
            let commit_slot = op
                .attn_state
                .ok_or_else(|| {
                    DispatchError::Hip(format!("comp l{layer}: attn_state_buf missing"))
                })?
                .sub_offset(if ratio == 4 { 7 } else { 9 }, 1);
            gpu.rope_tail_yarn_interleaved_staged_buf(
                commit_f32,
                pos,
                &commit_slot,
                head_dim as i32,
                r.rope_dim as i32,
                r.freq_base,
                r.freq_scale,
                r.ext_factor,
                r.attn_factor,
                r.corr_low,
                r.corr_high,
            )
            .map_err(|e| hip_err(format!("comp staged rope l{layer}"), e))?;
        } else if op.indexer {
            // The device-slot symbol of capture/replay: the plain-RoPE symbol
            // rounds differently on gfx1151 despite equivalent algebra,
            // poisoning later indexer state after the first boundary.
            let commit_slot = op
                .attn_state
                .ok_or_else(|| {
                    DispatchError::Hip(format!("comp l{layer}: attn_state_buf missing"))
                })?
                .sub_offset(7, 1);
            gpu.rope_tail_yarn_interleaved_at_slot_buf(
                op.kv_cache,
                pos,
                &commit_slot,
                head_dim as i32,
                r.rope_dim as i32,
                r.freq_base,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
            )
            .map_err(|e| hip_err(format!("comp rope slot l{layer}"), e))?;
        } else {
            gpu.rope_tail_yarn_interleaved(
                op.gfx1151_route,
                &kv_cache_slot,
                &kv_cache_slot,
                pos,
                1,
                0,
                head_dim as i32,
                r.rope_dim as i32,
                r.freq_base,
                r.freq_scale,
                r.ext_factor,
                r.attn_factor,
                r.corr_low,
                r.corr_high,
                /*inverse=*/ 0,
            )
            .map_err(|e| hip_err(format!("comp main rope l{layer}"), e))?;
        }
        comp_dbg(gpu, op, "kv_cache(rope)", commit_f32, head_dim);
        if cache_is_f16 {
            gpu.cast_f32_to_f16(commit_f32, &kv_cache_slot)
                .map_err(|e| hip_err(format!("comp cache f16 store l{layer}"), e))?;
        }
    }
    if op.overlap() {
        let shift_bytes = ratio * proj_dim * 4;
        for (ring, what) in [(kv_state, "kv_state"), (score_state, "score_state")] {
            let src = ring.sub_offset(ratio * proj_dim, ratio * proj_dim);
            let dst = ring.sub_offset(0, ratio * proj_dim);
            gpu.memcpy_dtod_auto(&dst.buf, &src.buf, shift_bytes)
                .map_err(|e| hip_err(format!("comp {what} shift l{layer}"), e))?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_device(
    gpu: &mut Gpu,
    op: &KvCompressOp<'_>,
    kv_buf: &GpuTensor,
    score_buf: &GpuTensor,
    pos: &GpuTensor,
    [ring_slot, commit_slot, shift_slot]: [&GpuTensor; 3],
    ring_only: bool,
    state_rows: usize,
) -> Result<(), DispatchError> {
    let layer = op.layer;
    let (ratio, head_dim, proj_dim) = (op.ratio, op.head_dim, op.proj_dim());
    let (kv_state, score_state, kv_cache) = (op.kv_state, op.score_state, op.kv_cache);
    // Ring write: unconditional, a no-op on the -1 sentinel.
    gpu.state_ring_write_f32_buf(kv_buf, kv_state, ring_slot, proj_dim as i32)
        .map_err(|e| hip_err(format!("comp ring write kv l{layer}"), e))?;
    gpu.state_ring_write_f32_buf(score_buf, score_state, ring_slot, proj_dim as i32)
        .map_err(|e| hip_err(format!("comp ring write score l{layer}"), e))?;
    comp_dbg(gpu, op, "kv_state(ring)", kv_state, state_rows * proj_dim);
    comp_dbg(
        gpu,
        op,
        "score_state(ring)",
        score_state,
        state_rows * proj_dim,
    );
    if ring_only {
        return Ok(());
    }
    // Commit stage: concat (overlap) is unconditional; pool / norm / rope /
    // store / shift no-op on the -1 commit sentinel.
    let cache_is_f16 = kv_cache.dtype == DType::F16;
    let staged;
    let commit_f32 = if cache_is_f16 {
        staged = op.staging.unwrap().sub_offset(0, head_dim);
        &staged
    } else {
        kv_cache
    };
    let (pool_kv, pool_score, pool_rows, what) = match op.concat {
        Some((concat_kv, concat_score)) => {
            gpu.compressor_overlap_concat_f32(kv_state, concat_kv, ratio as i32, head_dim as i32)
                .map_err(|e| hip_err(format!("comp concat_kv l{layer}"), e))?;
            gpu.compressor_overlap_concat_f32(
                score_state,
                concat_score,
                ratio as i32,
                head_dim as i32,
            )
            .map_err(|e| hip_err(format!("comp concat_score l{layer}"), e))?;
            comp_dbg(gpu, op, "concat_kv", concat_kv, 2 * ratio * head_dim);
            comp_dbg(gpu, op, "concat_score", concat_score, 2 * ratio * head_dim);
            (concat_kv, concat_score, 2 * ratio, "")
        }
        None => (kv_state, score_state, ratio, " no-overlap"),
    };
    if cache_is_f16 {
        gpu.compressor_softmax_pool_f32_staged_buf(
            pool_kv,
            pool_score,
            commit_f32,
            commit_slot,
            pool_rows as i32,
            head_dim as i32,
        )
        .map_err(|e| hip_err(format!("comp pool staged buf{what} l{layer}"), e))?;
    } else {
        gpu.compressor_softmax_pool_f32_buf(
            pool_kv,
            pool_score,
            kv_cache,
            commit_slot,
            pool_rows as i32,
            head_dim as i32,
        )
        .map_err(|e| hip_err(format!("comp pool buf{what} l{layer}"), e))?;
    }
    // Debug view of the committed row; past the cache capacity the row does
    // not exist (the kernels no-op on the sentinel), so nothing is printed.
    let dbg_commit_row = |gpu: &Gpu, name: &str| {
        let Some(row) = op.debug.and_then(|d| d.commit_row) else {
            return;
        };
        if cache_is_f16 {
            comp_dbg(gpu, op, name, commit_f32, head_dim);
        } else {
            comp_dbg(
                gpu,
                op,
                name,
                &kv_cache.sub_offset(row * head_dim, head_dim),
                head_dim,
            );
        }
    };
    dbg_commit_row(gpu, "kv_cache(pool)");
    if cache_is_f16 {
        gpu.rmsnorm_f32_staged_buf(commit_f32, op.norm, commit_slot, head_dim as i32, op.eps)
            .map_err(|e| hip_err(format!("comp rmsnorm staged buf l{layer}"), e))?;
    } else {
        gpu.rmsnorm_f32_at_slot_buf(kv_cache, op.norm, commit_slot, head_dim as i32, op.eps)
            .map_err(|e| hip_err(format!("comp rmsnorm buf l{layer}"), e))?;
    }
    dbg_commit_row(gpu, "kv_cache(rmsnorm)");
    let r = op.rope;
    if cache_is_f16 {
        gpu.rope_tail_yarn_interleaved_staged_buf(
            commit_f32,
            pos,
            commit_slot,
            head_dim as i32,
            r.rope_dim as i32,
            r.freq_base,
            r.freq_scale,
            r.ext_factor,
            r.attn_factor,
            r.corr_low,
            r.corr_high,
        )
        .map_err(|e| hip_err(format!("comp rope staged buf l{layer}"), e))?;
    } else {
        gpu.rope_tail_yarn_interleaved_at_slot_buf(
            kv_cache,
            pos,
            commit_slot,
            head_dim as i32,
            r.rope_dim as i32,
            r.freq_base,
            r.freq_scale,
            r.ext_factor,
            r.attn_factor,
            r.corr_low,
            r.corr_high,
        )
        .map_err(|e| hip_err(format!("comp rope buf l{layer}"), e))?;
    }
    dbg_commit_row(gpu, "kv_cache(rope)");
    if cache_is_f16 {
        gpu.cast_f32_to_f16_at_slot_buf(commit_f32, kv_cache, commit_slot, head_dim as i32)
            .map_err(|e| hip_err(format!("comp cache f16 store buf l{layer}"), e))?;
    }
    if op.overlap() {
        for (ring, what) in [(kv_state, "kv_state"), (score_state, "score_state")] {
            gpu.state_overlap_shift_f32_buf(ring, shift_slot, ratio as i32, proj_dim as i32)
                .map_err(|e| hip_err(format!("comp {what} shift buf l{layer}"), e))?;
        }
    }
    Ok(())
}
