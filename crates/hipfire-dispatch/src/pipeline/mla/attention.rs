// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Sparse latent attention, one operation per kernel chain: the ratio-4
//! indexer (query projection and RoPE, head weights, relu-weighted scores
//! over the compressed indexer cache, top-k), the SWA ring write, the
//! compressed-row gather, the attention (one joint softmax over the SWA
//! window, the gathered compressed rows and the sink) and the grouped
//! O-LoRA; the inverse tail RoPE between them is an
//! [`super::query_kv::ApplyTailRopeOp`]. Every operation serves one decode
//! row or a window of batched rows on an explicit route. Decode reads
//! positions, window fill and compressed counts from device state, as does a
//! `capture_safe` batched window, so a captured graph replays them.

use rdna_compute::{DType, Gpu, GpuTensor};

use super::rows::htod_i32;
use super::{project_prepared, project_prepared_rows, E8SoaKernel, RowsProjection};
use crate::types::DispatchError;

fn hip_err(what: impl std::fmt::Display, e: hip_bridge::HipError) -> DispatchError {
    DispatchError::Hip(format!("{what}: {e:?}"))
}

/// Model-owned exact-device kernels the attention ops may route to (the
/// gfx942 MQ2R backend). Portable owners refuse or decline.
pub trait LatentAttentionBackend {
    /// Grouped O-LoRA A projection: `y[g] = a[g] · x[g]` for `groups` groups.
    #[allow(clippy::too_many_arguments)]
    fn grouped_olora_e8(
        &self,
        gpu: &mut Gpu,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        groups: usize,
        m: usize,
        k: usize,
    ) -> Result<(), String>;

    /// Parallel indexer top-k; `Ok(false)` declines (the caller then runs the
    /// portable kernel).
    #[allow(clippy::too_many_arguments)]
    fn try_indexer_top_k_buf_parallel(
        &self,
        gpu: &mut Gpu,
        scores: &GpuTensor,
        top_indices: &GpuTensor,
        n_compressed_buf: &GpuTensor,
        k_buf: &GpuTensor,
        n_idx_heads: i32,
        max_k: i32,
        bounded: bool,
    ) -> Result<bool, String>;
}

/// Storage of a compressed cache (main or indexer).
#[derive(Clone, Copy)]
pub enum CompressedCache<'a> {
    F32(&'a GpuTensor),
    F16(&'a GpuTensor),
    /// F32 rows block-cyclic across `world` devices (exact gfx1201 TP).
    Sharded {
        shards: &'a [usize; 4],
        world: usize,
        block_rows: usize,
    },
}

/// Top-k kernel of the indexer.
#[derive(Clone, Copy)]
pub enum IndexTopK<'a> {
    /// Two-stage merge tree with its workspace.
    TwoStage {
        ws_scores: &'a GpuTensor,
        ws_indices: &'a GpuTensor,
    },
    /// The owner's parallel kernel, else the portable one.
    BackendParallel { bounded: bool, wide: bool },
    /// The portable one-launch kernel (`wide` = gfx1151/gfx1201 variant).
    Portable { wide: bool },
}

/// Ratio-4 sparse indexer of `rows` rows (`rows == 1` is one decode row): the
/// indexer query and head weights, ReLU scores over the indexer cache, then
/// top-k into `topk_indices`. `route` picks the decode or the batched chain.
pub struct SparseIndexOp<'a> {
    pub rows: usize,
    pub layer: usize,
    pub wq_b: &'a GpuTensor,
    pub weights_proj: &'a GpuTensor,
    pub q_lat: &'a GpuTensor,
    pub q_lat_rot: &'a GpuTensor,
    pub tmp: &'a GpuTensor,
    pub tmp_plain: &'a GpuTensor,
    pub q_idx: &'a GpuTensor,
    pub idx_weights: &'a GpuTensor,
    pub scores: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub pos: &'a GpuTensor,
    /// The head weights are already in `idx_weights` (packed projection).
    pub weights_preprojected: bool,
    pub heads: usize,
    pub head_dim: usize,
    pub top_k_max: usize,
    pub q_lora_rank: usize,
    pub hidden: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub route: SparseIndexRoute<'a>,
}

/// Decode chain (one row, `Step::SparseIndex`) or batched chain (a `rows` window).
/// Explicit, so a one-row verify window still takes the batched chain.
// One per compressed layer of a window; boxing the indexer buys nothing.
#[allow(clippy::large_enum_variant)]
pub enum SparseIndexRoute<'a> {
    Decode {
        cache: CompressedCache<'a>,
        /// `[.., n_compressed_4 @2, .., k_active_4 @4, ..]` device state.
        attn_state: &'a GpuTensor,
        top_k: IndexTopK<'a>,
        backend: &'a dyn LatentAttentionBackend,
        /// One head per wave (exact gfx1201, H=64 D=128 R=64).
        head_parallel_rope: bool,
        max_compressed: usize,
        e8_soa: E8SoaKernel,
    },
    Batched {
        cache: CompressedCache<'a>,
        wmma: bool,
        x_f16: &'a GpuTensor,
        /// Per-row compressed counts: the device array of the capture route, or
        /// the scratch array the direct route uploads `n_per_row` into.
        n_compressed: &'a GpuTensor,
        n_per_row: Vec<i32>,
        n_max: usize,
        /// Score row stride (the compressed-capacity bucket).
        capacity: usize,
        /// The attention reads the main cache through the top-k indices.
        direct: bool,
        capture_safe: bool,
        projection: RowsProjection,
        main_cache: CompressedCache<'a>,
        /// The LDS-transpose gather of the replicated F32 cache.
        tiled_gather: bool,
        staged: &'a GpuTensor,
        gather_head_dim: usize,
    },
}

impl SparseIndexOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let decode = matches!(self.route, SparseIndexRoute::Decode { .. });
        check_rows(self.rows, decode, || {
            format!("sparse index l{}", self.layer)
        })?;
        if let SparseIndexRoute::Decode { max_compressed, .. } = &self.route {
            if self.heads == 0 || self.head_dim == 0 || self.top_k_max == 0 || *max_compressed == 0
            {
                return Err(DispatchError::Hip("sparse index has empty geometry".into()));
            }
        }
        Ok(())
    }
}

pub fn execute_sparse_index(gpu: &mut Gpu, op: &SparseIndexOp<'_>) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    match &op.route {
        SparseIndexRoute::Decode { .. } => execute_decode_indexer(gpu, op),
        SparseIndexRoute::Batched { .. } => execute_batched_indexer(gpu, op),
    }
}

fn execute_decode_indexer(gpu: &mut Gpu, op: &SparseIndexOp<'_>) -> Result<(), DispatchError> {
    let &SparseIndexRoute::Decode {
        cache,
        attn_state,
        top_k,
        backend,
        head_parallel_rope,
        max_compressed,
        e8_soa,
    } = &op.route
    else {
        return Err(DispatchError::Hip(
            "sparse index decode route expected".into(),
        ));
    };
    let (layer, h, d, k) = (op.layer, op.heads, op.head_dim, op.top_k_max);
    let max_compressed = max_compressed as i32;
    // 1. q_idx = wq_b · q_lat (M = H*D, K = q_lora_rank).
    project_prepared(
        gpu,
        op.wq_b,
        op.q_lat_rot,
        op.q_lat,
        op.q_idx,
        h * d,
        op.q_lora_rank,
        e8_soa,
    )?;
    // 2. Tail RoPE on q_idx with the compressor's theta, at the current
    //    position (the K side was rotated by the indexer compressor).
    if head_parallel_rope {
        gpu.rope_tail_interleaved_h64d128r64_gfx1201(op.q_idx, op.pos, op.rope_theta, 32)
            .map_err(|e| hip_err(format!("idx rope gfx1201 l{layer}"), e))?;
    } else {
        gpu.rope_tail_interleaved(
            op.q_idx,
            op.q_idx,
            op.pos,
            h as i32,
            0,
            d as i32,
            op.rope_dim as i32,
            op.rope_theta,
        )
        .map_err(|e| hip_err(format!("idx rope l{layer}"), e))?;
    }
    // 3. idx_w = weights_proj · x → [H]
    if !op.weights_preprojected {
        project_prepared(
            gpu,
            op.weights_proj,
            op.tmp,
            op.tmp_plain,
            op.idx_weights,
            h,
            op.hidden,
            e8_soa,
        )?;
    }
    // 4. Relu-weighted scores over the first n_compressed_4 rows; rows past
    //    it score -inf so the top-k ignores them.
    let n_buf = attn_state.sub_offset(2, 1);
    let k_buf = attn_state.sub_offset(4, 1);
    match cache {
        CompressedCache::Sharded {
            shards,
            world,
            block_rows,
        } => gpu
            .indexer_relu_score_f32_buf_sharded_gfx1201(
                op.q_idx,
                shards,
                op.idx_weights,
                op.scores,
                &n_buf,
                max_compressed,
                h as i32,
                d as i32,
                world as i32,
                block_rows as i32,
            )
            .map_err(|e| hip_err(format!("idx score sharded buf l{layer}"), e))?,
        CompressedCache::F16(cache) => gpu
            .indexer_relu_score_f16_buf(
                op.q_idx,
                cache,
                op.idx_weights,
                op.scores,
                &n_buf,
                max_compressed,
                h as i32,
                d as i32,
            )
            .map_err(|e| hip_err(format!("idx score f16 buf l{layer}"), e))?,
        CompressedCache::F32(cache) => gpu
            .indexer_relu_score_f32_buf(
                op.q_idx,
                cache,
                op.idx_weights,
                op.scores,
                &n_buf,
                max_compressed,
                h as i32,
                d as i32,
            )
            .map_err(|e| hip_err(format!("idx score buf l{layer}"), e))?,
    }
    // 5. Top-k over the device-side N and K.
    let portable_wide = match top_k {
        IndexTopK::TwoStage {
            ws_scores,
            ws_indices,
        } => {
            return gpu
                .indexer_top_k_two_stage(
                    op.scores,
                    ws_scores,
                    ws_indices,
                    op.topk_indices,
                    &n_buf,
                    &k_buf,
                    max_compressed,
                    k as i32,
                )
                .map_err(|e| hip_err(format!("idx top_k two-stage l{layer}"), e));
        }
        IndexTopK::BackendParallel { bounded, wide } => {
            if backend
                .try_indexer_top_k_buf_parallel(
                    gpu,
                    op.scores,
                    op.topk_indices,
                    &n_buf,
                    &k_buf,
                    /*n_idx_heads=*/ 1,
                    k as i32,
                    bounded,
                )
                .map_err(DispatchError::Hip)?
            {
                return Ok(());
            }
            wide
        }
        IndexTopK::Portable { wide } => wide,
    };
    gpu.indexer_top_k_buf(
        portable_wide,
        op.scores,
        op.topk_indices,
        &n_buf,
        &k_buf,
        /*n_idx_heads=*/ 1,
        max_compressed,
        k as i32,
    )
    .map_err(|e| hip_err(format!("idx top_k buf l{layer}"), e))
}

fn execute_batched_indexer(gpu: &mut Gpu, op: &SparseIndexOp<'_>) -> Result<(), DispatchError> {
    let &SparseIndexRoute::Batched {
        n_max,
        capacity,
        direct,
        capture_safe,
        main_cache,
        tiled_gather,
        staged,
        gather_head_dim,
        ..
    } = &op.route
    else {
        return Err(DispatchError::Hip(
            "sparse index batched route expected".into(),
        ));
    };
    let (l, rows) = (op.layer, op.rows);
    if n_max == 0 {
        return Ok(());
    }
    batched_indexer_scores(gpu, op)?;
    if direct {
        return Ok(());
    }
    let n_gather = if capture_safe { capacity } else { n_max };
    let (hd, k, b, n) = (
        gather_head_dim as i32,
        op.top_k_max as i32,
        rows as i32,
        n_gather as i32,
    );
    match main_cache {
        CompressedCache::Sharded {
            shards,
            world,
            block_rows,
        } => gpu
            .deepseek4_topk_kv_gather_batched_tiled_sharded_gfx1201(
                shards,
                op.topk_indices,
                staged,
                k,
                hd,
                n,
                k,
                0,
                1.0,
                b,
                world as i32,
                block_rows as i32,
            )
            .map_err(|e| {
                hip_err(
                    format_args!("deepseek4_topk_kv_gather_batched_tiled_sharded_gfx1201 l{l}"),
                    e,
                )
            }),
        CompressedCache::F16(cache) => gpu
            .deepseek4_topk_kv_gather_batched_tiled_f16(
                cache,
                op.topk_indices,
                staged,
                k,
                hd,
                n,
                k,
                0,
                1.0,
                b,
            )
            .map_err(|e| {
                hip_err(
                    format_args!("deepseek4_topk_kv_gather_batched_tiled_f16 l{l}"),
                    e,
                )
            }),
        // The LDS-transpose gather (coalesced stores).
        CompressedCache::F32(cache) if tiled_gather => gpu
            .deepseek4_topk_kv_gather_batched_tiled_gfx1201(
                cache,
                op.topk_indices,
                staged,
                k,
                hd,
                n,
                k,
                0,
                1.0,
                b,
            )
            .map_err(|e| {
                hip_err(
                    format_args!("deepseek4_topk_kv_gather_batched_tiled_gfx1201 l{l}"),
                    e,
                )
            }),
        CompressedCache::F32(cache) => gpu
            .deepseek4_topk_kv_gather_batched_f32(
                cache,
                op.topk_indices,
                staged,
                k,
                hd,
                n,
                k,
                0,
                1.0,
                b,
            )
            .map_err(|e| hip_err(format_args!("deepseek4_topk_kv_gather_batched l{l}"), e)),
    }
}

fn batched_indexer_scores(gpu: &mut Gpu, op: &SparseIndexOp<'_>) -> Result<(), DispatchError> {
    let &SparseIndexRoute::Batched {
        cache,
        wmma,
        x_f16,
        n_compressed,
        ref n_per_row,
        n_max,
        capacity,
        capture_safe,
        projection,
        ..
    } = &op.route
    else {
        return Err(DispatchError::Hip(
            "sparse index batched route expected".into(),
        ));
    };
    let (l, rows) = (op.layer, op.rows);
    if !capture_safe {
        htod_i32(gpu, n_compressed, n_per_row, "htod n_per_batch")?;
    }
    project_prepared_rows(
        gpu,
        op.wq_b,
        op.q_lat_rot,
        op.q_lat,
        op.q_idx,
        op.heads * op.head_dim,
        op.q_lora_rank,
        rows,
        Some(x_f16),
        &projection,
    )?;
    gpu.rope_tail_interleaved_batched(
        op.q_idx,
        op.q_idx,
        op.pos,
        op.heads as i32,
        0,
        op.head_dim as i32,
        op.rope_dim as i32,
        op.rope_theta,
        rows as i32,
    )
    .map_err(|e| hip_err(format_args!("rope_tail_batched idx l{l}"), e))?;
    if !op.weights_preprojected {
        project_prepared_rows(
            gpu,
            op.weights_proj,
            op.tmp,
            op.tmp_plain,
            op.idx_weights,
            op.heads,
            op.hidden,
            rows,
            Some(x_f16),
            &projection,
        )?;
    }
    // Scores are written at the allocated row stride; slots past a row's
    // count read stale cache rows but score -inf.
    let (h, d, cap, b) = (
        op.heads as i32,
        op.head_dim as i32,
        capacity as i32,
        rows as i32,
    );
    match cache {
        CompressedCache::Sharded {
            shards,
            world,
            block_rows,
        } => gpu
            .indexer_relu_score_wmma_batched_sharded_gfx1201(
                op.q_idx,
                shards,
                op.idx_weights,
                n_compressed,
                op.scores,
                h,
                d,
                cap,
                b,
                world as i32,
                block_rows as i32,
            )
            .map_err(|e| {
                hip_err(
                    format_args!("indexer_relu_score_wmma_batched_sharded l{l}"),
                    e,
                )
            })?,
        CompressedCache::F16(cache) if wmma => gpu
            .indexer_relu_score_wmma_batched_f16(
                op.q_idx,
                cache,
                op.idx_weights,
                n_compressed,
                op.scores,
                h,
                d,
                cap,
                b,
            )
            .map_err(|e| hip_err(format_args!("indexer_relu_score_wmma_batched_f16 l{l}"), e))?,
        CompressedCache::F16(cache) => gpu
            .indexer_relu_score_batched_f16(
                op.q_idx,
                cache,
                op.idx_weights,
                n_compressed,
                op.scores,
                h,
                d,
                cap,
                b,
            )
            .map_err(|e| hip_err(format_args!("indexer_relu_score_batched_f16 l{l}"), e))?,
        // All four WMMA warps consume the same 16x128 K tile: gfx1151 stages
        // it once in LDS.
        CompressedCache::F32(cache) if wmma && gpu.arch.eq_ignore_ascii_case("gfx1151") => gpu
            .indexer_relu_score_wmma_batched_klds_gfx1151(
                op.q_idx,
                cache,
                op.idx_weights,
                n_compressed,
                op.scores,
                h,
                d,
                cap,
                b,
            )
            .map_err(|e| hip_err(format_args!("indexer_relu_score_wmma_batched_klds l{l}"), e))?,
        CompressedCache::F32(cache) if wmma => gpu
            .indexer_relu_score_wmma_batched_f32(
                op.q_idx,
                cache,
                op.idx_weights,
                n_compressed,
                op.scores,
                h,
                d,
                cap,
                b,
            )
            .map_err(|e| hip_err(format_args!("indexer_relu_score_wmma_batched l{l}"), e))?,
        CompressedCache::F32(cache) => gpu
            .indexer_relu_score_batched_f32(
                op.q_idx,
                cache,
                op.idx_weights,
                n_compressed,
                op.scores,
                h,
                d,
                cap,
                b,
            )
            .map_err(|e| hip_err(format_args!("indexer_relu_score_batched l{l}"), e))?,
    }
    // Top-k over the window's live range (`n_max`), not the capacity stride.
    let k_fill = op.top_k_max.min(n_max) as i32;
    let (n_max, k) = (n_max as i32, op.top_k_max as i32);
    if capture_safe {
        gpu.indexer_top_k_batched_buf(op.scores, op.topk_indices, n_compressed, 1, cap, k, b)
            .map_err(|e| hip_err(format_args!("indexer_top_k_batched_buf l{l}"), e))
    } else if gpu.arch.eq_ignore_ascii_case("gfx1151") {
        // One bounded exact merge per row: prefill has hundreds of rows.
        gpu.indexer_top_k_batched_bounded_gfx1151(
            op.scores,
            op.topk_indices,
            1,
            cap,
            n_max,
            k,
            k_fill,
            b,
        )
        .map_err(|e| {
            hip_err(
                format_args!("indexer_top_k_batched_bounded_gfx1151 l{l}"),
                e,
            )
        })
    } else if gpu.arch.eq_ignore_ascii_case("gfx1201") {
        // The portable rank count is O(N^2) past K=512 compressed rows.
        gpu.indexer_top_k_batched_bounded_gfx1201(
            op.scores,
            op.topk_indices,
            1,
            cap,
            n_max,
            k,
            k_fill,
            b,
        )
        .map_err(|e| {
            hip_err(
                format_args!("indexer_top_k_batched_bounded_gfx1201 l{l}"),
                e,
            )
        })
    } else {
        gpu.indexer_top_k_batched(op.scores, op.topk_indices, 1, cap, n_max, k, k_fill, b)
            .map_err(|e| hip_err(format_args!("indexer_top_k_batched l{l}"), e))
    }
}

/// A decode route serves exactly one row; every route at least one.
pub(super) fn check_rows(
    rows: usize,
    decode: bool,
    what: impl FnOnce() -> String,
) -> Result<(), DispatchError> {
    if rows == 0 || (decode && rows != 1) {
        return Err(DispatchError::Hip(format!(
            "{}: {rows} rows on the {} route",
            what(),
            if decode { "decode" } else { "batched" }
        )));
    }
    Ok(())
}

/// Write the step's KV rows into both SWA rings (`K = V` tied).
pub struct SwaRingWriteOp<'a> {
    pub kv: &'a GpuTensor,
    pub swa_k: &'a GpuTensor,
    pub swa_v: &'a GpuTensor,
    pub n_kv: usize,
    pub head_dim: usize,
    pub win: usize,
    pub rows: usize,
    pub layer: usize,
    pub route: SwaRingWriteRoute<'a>,
}

pub enum SwaRingWriteRoute<'a> {
    /// One row into the ring slot `attn_state[0]`. Decode attention reads the
    /// rings, so this write precedes it.
    Decode { attn_state: &'a GpuTensor },
    /// The window's rows, after its attention (which reads them staged).
    /// `capture_safe` reads each row's slot from `positions`.
    Batched {
        positions: &'a GpuTensor,
        start_pos: u32,
        capture_safe: bool,
    },
}

impl SwaRingWriteOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let decode = matches!(self.route, SwaRingWriteRoute::Decode { .. });
        check_rows(self.rows, decode, || {
            format!("SWA ring write l{}", self.layer)
        })
    }
}

pub fn execute_swa_ring_write(gpu: &mut Gpu, op: &SwaRingWriteOp<'_>) -> Result<(), DispatchError> {
    let (n_kv, hd, win) = (op.n_kv as i32, op.head_dim as i32, op.win as i32);
    match op.route {
        SwaRingWriteRoute::Decode { attn_state } => {
            // The `_buf` kernels read the ring slot from device state, so a
            // captured launch picks up each new position on replay.
            let slot = attn_state.sub_offset(0, 1);
            for (ring, what) in [(op.swa_k, "swa_k write"), (op.swa_v, "swa_v write")] {
                gpu.swa_ring_write_f32_buf(op.kv, ring, &slot, n_kv, hd, win)
                    .map_err(|e| hip_err(what, e))?;
            }
        }
        SwaRingWriteRoute::Batched {
            positions,
            start_pos,
            capture_safe,
        } => {
            let (l, rows) = (op.layer, op.rows as i32);
            for (ring, which) in [(op.swa_k, "k"), (op.swa_v, "v")] {
                if capture_safe {
                    gpu.swa_ring_write_batched_pos_f32(op.kv, ring, positions, n_kv, hd, win, rows)
                        .map_err(|e| {
                            hip_err(format_args!("swa_ring_write_batched_pos ({which}) l{l}"), e)
                        })?;
                } else {
                    gpu.swa_ring_write_batched_f32(
                        op.kv,
                        ring,
                        n_kv,
                        hd,
                        win,
                        start_pos as i32,
                        rows,
                    )
                    .map_err(|e| {
                        hip_err(format_args!("swa_ring_write_batched ({which}) l{l}"), e)
                    })?;
                }
            }
        }
    }
    Ok(())
}

/// Sliding-window attention (compress ratio 0): one softmax over the window
/// and the sink into `out`.
pub struct SwaAttentionOp<'a> {
    pub q: &'a GpuTensor,
    pub sink: &'a GpuTensor,
    pub out: &'a GpuTensor,
    pub n_heads: usize,
    pub head_dim: usize,
    pub n_groups: usize,
    pub win: usize,
    pub rows: usize,
    pub layer: usize,
    pub route: SwaAttentionRoute<'a>,
}

pub enum SwaAttentionRoute<'a> {
    /// One row over the SWA rings; the window fill is `attn_state[1]`.
    Decode {
        swa_k: &'a GpuTensor,
        swa_v: &'a GpuTensor,
        attn_state: &'a GpuTensor,
    },
    /// Developer diagnostic (`HIPFIRE_DEEPSEEK4_ATTN=pos0`): each head attends
    /// only the current row's `kv`, on every layer.
    Pos0 { kv: &'a GpuTensor },
    /// The window's rows over its staged SWA visibility
    /// ([`super::rows::SwaStageOp`]); `n_valid` holds each row's fill.
    Batched {
        staged: &'a GpuTensor,
        n_valid: &'a GpuTensor,
        start_pos: u32,
        /// Developer: one `deepseek4_attn_swa` launch per row.
        per_position: bool,
    },
}

impl SwaAttentionOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let decode = matches!(
            self.route,
            SwaAttentionRoute::Decode { .. } | SwaAttentionRoute::Pos0 { .. }
        );
        check_rows(self.rows, decode, || {
            format!("SWA attention l{}", self.layer)
        })
    }
}

pub fn execute_swa_attention(gpu: &mut Gpu, op: &SwaAttentionOp<'_>) -> Result<(), DispatchError> {
    let (heads, hd, groups) = (op.n_heads as i32, op.head_dim as i32, op.n_groups as i32);
    match op.route {
        SwaAttentionRoute::Decode {
            swa_k,
            swa_v,
            attn_state,
        } => {
            let n_valid = attn_state.sub_offset(1, 1);
            gpu.deepseek4_attn_swa_buf(
                op.q,
                swa_k,
                swa_v,
                op.sink,
                op.out,
                &n_valid,
                heads,
                hd,
                groups,
                op.win as i32,
            )
            .map_err(|e| hip_err("deepseek4_attn_swa_buf", e))
        }
        SwaAttentionRoute::Pos0 { kv } => gpu
            .deepseek4_attn_pos0(op.q, kv, op.sink, op.out, heads, hd, groups)
            .map_err(|e| hip_err("deepseek4_attn_pos0", e)),
        SwaAttentionRoute::Batched {
            staged,
            n_valid,
            start_pos,
            per_position,
        } => {
            let l = op.layer;
            if per_position {
                let q_per = op.n_heads * op.head_dim;
                let kv_per = op.head_dim * op.win;
                for b in 0..op.rows {
                    let q = op.q.sub_offset(b * q_per, q_per);
                    let kv = staged.sub_offset(b * kv_per, kv_per);
                    let out = op.out.sub_offset(b * q_per, q_per);
                    let n_valid = ((start_pos as usize + b + 1).min(op.win)) as i32;
                    gpu.deepseek4_attn_swa(
                        &q,
                        &kv,
                        &kv,
                        op.sink,
                        &out,
                        heads,
                        hd,
                        groups,
                        n_valid,
                        op.win as i32,
                    )
                    .map_err(|e| {
                        hip_err(format_args!("deepseek4_attn_swa per-pos b={b} l{l}"), e)
                    })?;
                }
                return Ok(());
            }
            gpu.deepseek4_attn_swa_batched(
                op.q,
                staged,
                staged,
                op.sink,
                n_valid,
                op.out,
                heads,
                hd,
                groups,
                op.win as i32,
                op.rows as i32,
            )
            .map_err(|e| hip_err(format_args!("deepseek4_attn_swa_batched l{l}"), e))
        }
    }
}

/// Identity top-k gather (compress ratio 128: no indexer): the first active
/// compressed rows, in order, into `staged`.
pub struct IdentityGatherOp<'a> {
    pub cache: CompressedCache<'a>,
    pub staged: &'a GpuTensor,
    pub head_dim: usize,
    pub top_k_max: usize,
    pub rows: usize,
    pub layer: usize,
    pub route: IdentityGatherRoute<'a>,
}

pub enum IdentityGatherRoute<'a> {
    /// One row; the active count is `attn_state[k_active_slot]`.
    Decode {
        attn_state: &'a GpuTensor,
        k_active_slot: usize,
    },
    /// The window's rows; `n` compressed rows (`0` skips).
    Batched { n: usize },
}

impl IdentityGatherOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let decode = matches!(self.route, IdentityGatherRoute::Decode { .. });
        check_rows(self.rows, decode, || {
            format!("identity gather l{}", self.layer)
        })
    }
}

pub fn execute_identity_gather(
    gpu: &mut Gpu,
    op: &IdentityGatherOp<'_>,
) -> Result<(), DispatchError> {
    let (layer, hd, top) = (op.layer, op.head_dim as i32, op.top_k_max as i32);
    match op.route {
        IdentityGatherRoute::Decode {
            attn_state,
            k_active_slot,
        } => {
            let k_active = attn_state.sub_offset(k_active_slot, 1);
            match op.cache {
                CompressedCache::Sharded {
                    shards,
                    world,
                    block_rows,
                } => gpu
                    .deepseek4_topk_kv_gather_identity_f32_buf_sharded_gfx1201(
                        shards,
                        op.staged,
                        &k_active,
                        top,
                        hd,
                        top,
                        world as i32,
                        block_rows as i32,
                    )
                    .map_err(|e| hip_err(format!("mixed gather sharded (all,buf) l{layer}"), e)),
                CompressedCache::F16(cache) => gpu
                    .deepseek4_topk_kv_gather_identity_f16_buf(
                        cache, op.staged, &k_active, top, hd, top,
                    )
                    .map_err(|e| hip_err(format!("mixed gather f16 (all,buf) l{layer}"), e)),
                CompressedCache::F32(cache) => gpu
                    .deepseek4_topk_kv_gather_identity_f32_buf(
                        cache, op.staged, &k_active, top, hd, top,
                    )
                    .map_err(|e| hip_err(format!("mixed gather (all,buf) l{layer}"), e)),
            }
        }
        IdentityGatherRoute::Batched { n } => {
            if n == 0 {
                return Ok(());
            }
            let (n, b) = (n as i32, op.rows as i32);
            match op.cache {
                CompressedCache::Sharded {
                    shards,
                    world,
                    block_rows,
                } => gpu
                    .deepseek4_topk_kv_gather_identity_batched_sharded_gfx1201(
                        shards,
                        op.staged,
                        n,
                        hd,
                        top,
                        b,
                        world as i32,
                        block_rows as i32,
                    )
                    .map_err(|e| {
                        hip_err(
                            format_args!(
                                "deepseek4_topk_kv_gather_identity_batched_sharded l{layer}"
                            ),
                            e,
                        )
                    }),
                CompressedCache::F16(cache) => gpu
                    .deepseek4_topk_kv_gather_identity_batched_f16(cache, op.staged, n, hd, top, b)
                    .map_err(|e| {
                        hip_err(
                            format_args!("deepseek4_topk_kv_gather_identity_batched_f16 l{layer}"),
                            e,
                        )
                    }),
                CompressedCache::F32(cache) => gpu
                    .deepseek4_topk_kv_gather_identity_batched_f32(cache, op.staged, n, hd, top, b)
                    .map_err(|e| {
                        hip_err(
                            format_args!("deepseek4_topk_kv_gather_identity_batched l{layer}"),
                            e,
                        )
                    }),
            }
        }
    }
}

/// The indexer's top-k rows (ratio 4) a decode row gathers from the main
/// compressed cache before attending.
pub struct TopkGather<'a> {
    pub cache: CompressedCache<'a>,
    pub indices: &'a GpuTensor,
    /// `attn_state` slot of the compressed row count.
    pub n_compressed_slot: usize,
}

/// One softmax over the SWA window, the gathered compressed rows (K = V
/// tied) and the sink into `out` (compressed layers).
pub struct SparseAttentionOp<'a> {
    pub q: &'a GpuTensor,
    pub sink: &'a GpuTensor,
    pub out: &'a GpuTensor,
    pub n_heads: usize,
    pub head_dim: usize,
    pub win: usize,
    pub topk: usize,
    pub rows: usize,
    pub layer: usize,
    pub route: SparseAttentionRoute<'a>,
}

pub enum SparseAttentionRoute<'a> {
    /// One row over the SWA rings and `gathered`; the window fill is
    /// `attn_state[1]`, the active compressed count `attn_state[k_active_slot]`.
    /// `gather` fills `gathered` from the indexer's top-k first (ratio 4);
    /// ratio 128 gathers in a preceding [`IdentityGatherOp`].
    Decode {
        swa_k: &'a GpuTensor,
        swa_v: &'a GpuTensor,
        gathered: &'a GpuTensor,
        attn_state: &'a GpuTensor,
        k_active_slot: usize,
        gather: Option<TopkGather<'a>>,
        gfx1151_route: bool,
    },
    /// The window's rows over the staged SWA visibility and the staged top-k
    /// rows, or (`direct`) the main cache read through the top-k indices.
    Batched {
        staged_swa: &'a GpuTensor,
        staged_topk: &'a GpuTensor,
        n_valid: &'a GpuTensor,
        /// Per-row active top-k counts: the capture route's device array, or
        /// the array the direct route uploads `n_active_host` into.
        n_active: &'a GpuTensor,
        n_active_host: Vec<i32>,
        direct: Option<DirectTopk<'a>>,
        /// F16 WMMA attention admitted (falls back to F32 on a refused launch).
        wmma: bool,
        capture_safe: bool,
    },
}

/// The main compressed cache a batched [`SparseAttentionRoute`] reads through
/// the indexer's top-k indices.
pub struct DirectTopk<'a> {
    pub cache: &'a GpuTensor,
    pub indices: &'a GpuTensor,
    pub n_compressed: usize,
}

impl SparseAttentionOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let decode = matches!(self.route, SparseAttentionRoute::Decode { .. });
        check_rows(self.rows, decode, || {
            format!("sparse attention l{}", self.layer)
        })
    }
}

pub fn execute_sparse_attention(
    gpu: &mut Gpu,
    op: &SparseAttentionOp<'_>,
) -> Result<(), DispatchError> {
    let layer = op.layer;
    let (heads, hd, top) = (op.n_heads as i32, op.head_dim as i32, op.topk as i32);
    match &op.route {
        SparseAttentionRoute::Decode {
            swa_k,
            swa_v,
            gathered,
            attn_state,
            k_active_slot,
            gather,
            gfx1151_route,
        } => {
            let n_valid = attn_state.sub_offset(1, 1);
            let k_active = attn_state.sub_offset(*k_active_slot, 1);
            if let Some(g) = gather {
                let n_compressed = attn_state.sub_offset(g.n_compressed_slot, 1);
                let idx = g.indices;
                match g.cache {
                    CompressedCache::Sharded {
                        shards,
                        world,
                        block_rows,
                    } => gpu
                        .deepseek4_topk_kv_gather_f32_buf_sharded_gfx1201(
                            shards,
                            idx,
                            gathered,
                            &k_active,
                            &n_compressed,
                            top,
                            hd,
                            top,
                            0,
                            1.0,
                            world as i32,
                            block_rows as i32,
                        )
                        .map_err(|e| {
                            hip_err(format!("mixed gather sharded (idx,buf) l{layer}"), e)
                        })?,
                    CompressedCache::F16(cache) => gpu
                        .deepseek4_topk_kv_gather_f16_buf(
                            cache,
                            idx,
                            gathered,
                            &k_active,
                            &n_compressed,
                            top,
                            hd,
                            top,
                            0,
                            1.0,
                        )
                        .map_err(|e| hip_err(format!("mixed gather f16 (idx,buf) l{layer}"), e))?,
                    CompressedCache::F32(cache) => gpu
                        .deepseek4_topk_kv_gather_f32_buf(
                            cache,
                            idx,
                            gathered,
                            &k_active,
                            &n_compressed,
                            top,
                            hd,
                            top,
                            0,
                            1.0,
                        )
                        .map_err(|e| hip_err(format!("mixed gather (idx,buf) l{layer}"), e))?,
                }
            }
            gpu.deepseek4_attn_swa_topk_f32_buf(
                *gfx1151_route,
                op.q,
                swa_k,
                swa_v,
                gathered,
                gathered,
                op.sink,
                op.out,
                &n_valid,
                &k_active,
                heads,
                hd,
                op.win as i32,
                top,
            )
            .map_err(|e| hip_err(format!("deepseek4_attn_swa_topk_buf l{layer}"), e))
        }
        SparseAttentionRoute::Batched {
            staged_swa,
            staged_topk,
            n_valid,
            n_active,
            n_active_host,
            direct,
            wmma,
            capture_safe,
        } => {
            if !capture_safe {
                htod_i32(gpu, n_active, n_active_host, "htod n_active_topk_arr")?;
            }
            // Dynamic LDS is part of a captured launch: size it for the route
            // maximum under capture, not the first window's count.
            let max_n_total = if *capture_safe {
                (op.win + op.topk) as i32
            } else {
                op.win as i32 + n_active_host.iter().copied().max().unwrap_or(0)
            };
            let (win, b) = (op.win as i32, op.rows as i32);
            if let Some(d) = direct {
                let nc = d.n_compressed as i32;
                if *wmma
                    && gpu
                        .deepseek4_attn_swa_topk_direct_wmma(
                            op.q,
                            staged_swa,
                            d.cache,
                            d.indices,
                            op.sink,
                            n_valid,
                            n_active,
                            op.out,
                            heads,
                            hd,
                            win,
                            top,
                            nc,
                            b,
                            max_n_total,
                        )
                        .is_ok()
                {
                    return Ok(());
                }
                return gpu
                    .deepseek4_attn_swa_topk_direct_batched_f32(
                        op.q, staged_swa, staged_swa, d.cache, d.indices, op.sink, n_valid,
                        n_active, op.out, heads, hd, win, top, nc, b,
                    )
                    .map_err(|e| {
                        hip_err(
                            format_args!("deepseek4_attn_swa_topk_direct_batched l{layer}"),
                            e,
                        )
                    });
            }
            if *wmma {
                let result = if gpu.arch_caps.is_gfx1201() {
                    gpu.deepseek4_attn_swa_topk_batched_wmma_gfx12(
                        op.q,
                        staged_swa,
                        staged_topk,
                        op.sink,
                        n_valid,
                        n_active,
                        op.out,
                        heads,
                        hd,
                        win,
                        top,
                        b,
                        max_n_total,
                    )
                } else {
                    gpu.deepseek4_attn_swa_topk_batched_wmma(
                        op.q,
                        staged_swa,
                        staged_topk,
                        op.sink,
                        n_valid,
                        n_active,
                        op.out,
                        heads,
                        hd,
                        win,
                        top,
                        b,
                        max_n_total,
                    )
                };
                if result.is_ok() {
                    return Ok(());
                }
            }
            gpu.deepseek4_attn_swa_topk_batched_f32(
                op.q,
                staged_swa,
                staged_swa,
                staged_topk,
                staged_topk,
                op.sink,
                n_valid,
                n_active,
                op.out,
                heads,
                hd,
                win,
                top,
                b,
            )
            .map_err(|e| hip_err(format_args!("deepseek4_attn_swa_topk_batched l{layer}"), e))
        }
    }
}

/// Grouped O-LoRA A projection route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OloraA {
    /// One-launch grouped MQ4 (MQ2RXT).
    Mq4Grouped,
    /// One-launch grouped E8 SoA, per arch.
    E8Gfx1151,
    E8Gfx1201,
    E8Gfx1100,
    /// The owner's exact gfx942 backend.
    E8Backend,
    /// One projection per group; `group_bytes` is one group's weight slice
    /// (elements for F32 weights).
    PerGroup {
        group_bytes: usize,
    },
}

/// Grouped O-LoRA of the attention output `raw`: (rotate) → per-group
/// `wo_a` → (rotate) → `wo_b` into `out`.
pub struct OLoraOp<'a> {
    pub wo_a: &'a GpuTensor,
    pub wo_b: &'a GpuTensor,
    pub raw: &'a GpuTensor,
    pub raw_rot: &'a GpuTensor,
    pub wo_a_out: &'a GpuTensor,
    pub wo_a_out_rot: &'a GpuTensor,
    pub out: &'a GpuTensor,
    pub n_groups: usize,
    pub o_lora_rank: usize,
    pub per_group_in: usize,
    pub hidden: usize,
    /// FWHT-rotate the attention output / the `wo_a` output before the next
    /// projection.
    pub rotate_raw: bool,
    pub rotate_mid: bool,
    pub rows: usize,
    pub layer: usize,
    pub route: OLoraRoute<'a>,
}

pub enum OLoraRoute<'a> {
    /// One row: `wo_a` on `olora_a`, `wo_b` a prepared projection.
    Decode {
        olora_a: OloraA,
        backend: &'a dyn LatentAttentionBackend,
        e8_soa: E8SoaKernel,
    },
    /// The window's rows: grouped `wo_a` per format, `wo_b` a prepared
    /// rows projection.
    Batched {
        x_f16: &'a GpuTensor,
        /// Developer Q8 `wo_a` multi-row variant (0, 2 or 4).
        q8_multirow: i32,
        projection: RowsProjection,
    },
}

impl OLoraOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.n_groups == 0 || self.o_lora_rank == 0 || self.per_group_in == 0 {
            return Err(DispatchError::Hip(format!(
                "O-LoRA l{}: empty geometry",
                self.layer
            )));
        }
        let decode = matches!(self.route, OLoraRoute::Decode { .. });
        if !decode
            && !matches!(
                self.wo_a.dtype,
                DType::F32
                    | DType::Q8_0
                    | DType::Raw
                    | DType::MQ4G256
                    | DType::MFP4G32E8
                    | DType::MFP4G32E8SOA
                    | DType::MFP3G32E8
            )
        {
            return Err(DispatchError::Hip(format!(
                "batched O-LoRA l{}: unsupported wo_a dtype {:?}",
                self.layer, self.wo_a.dtype
            )));
        }
        check_rows(self.rows, decode, || format!("O-LoRA l{}", self.layer))
    }
}

pub fn execute_o_lora(gpu: &mut Gpu, op: &OLoraOp<'_>) -> Result<(), DispatchError> {
    let layer = op.layer;
    let groups_rank = op.n_groups * op.o_lora_rank;
    match &op.route {
        &OLoraRoute::Decode {
            olora_a,
            backend,
            e8_soa,
        } => {
            if op.rotate_raw {
                gpu.rotate_x_mq_batched(op.raw, op.raw_rot, op.per_group_in, op.n_groups)
                    .map_err(|e| hip_err(format!("rotate attn_out batched l{layer}"), e))?;
            }
            decode_olora_a(gpu, op, olora_a, backend, e8_soa)?;
            if op.rotate_mid {
                gpu.rotate_x_mq(op.wo_a_out, op.wo_a_out_rot, groups_rank)
                    .map_err(|e| hip_err(format!("rotate wo_a_out l{layer}"), e))?;
            }
            project_prepared(
                gpu,
                op.wo_b,
                op.wo_a_out_rot,
                op.wo_a_out,
                op.out,
                op.hidden,
                groups_rank,
                e8_soa,
            )
        }
        OLoraRoute::Batched {
            x_f16,
            q8_multirow,
            projection,
        } => {
            let rows = op.rows;
            let (g, r, pin, b) = (
                op.n_groups as i32,
                op.o_lora_rank as i32,
                op.per_group_in as i32,
                rows as i32,
            );
            if op.rotate_raw {
                gpu.rotate_x_mq_batched(op.raw, op.raw_rot, op.n_groups * op.per_group_in, rows)
                    .map_err(|e| hip_err(format_args!("rotate attn_out_raw l{layer}"), e))?;
            }
            match op.wo_a.dtype {
                DType::F32 => gpu
                    .wo_per_group_batched_f32(op.wo_a, op.raw, op.wo_a_out, g, r, pin, b)
                    .map_err(|e| hip_err(format_args!("wo_per_group_batched_f32 l{layer}"), e))?,
                // Q8_0 reads the plain attention output.
                DType::Q8_0 if *q8_multirow == 0 => gpu
                    .wo_per_group_batched_q8_0(op.wo_a, op.raw, op.wo_a_out, g, r, pin, b)
                    .map_err(|e| hip_err(format_args!("wo_per_group_batched_q8_0 l{layer}"), e))?,
                DType::Q8_0 => gpu
                    .wo_per_group_batched_q8_0_multirow(
                        op.wo_a,
                        op.raw,
                        op.wo_a_out,
                        g,
                        r,
                        pin,
                        b,
                        *q8_multirow,
                    )
                    .map_err(|e| {
                        hip_err(
                            format_args!("wo_per_group_batched_q8_0_multirow l{layer}"),
                            e,
                        )
                    })?,
                DType::Raw | DType::MQ4G256 => gpu
                    .wo_per_group_batched_hfq4g256(op.wo_a, op.raw_rot, op.wo_a_out, g, r, pin, b)
                    .map_err(|e| {
                        hip_err(format_args!("wo_per_group_batched_hfq4g256 l{layer}"), e)
                    })?,
                _ => wo_a_rows_e8(gpu, op, x_f16, projection)?,
            }
            if op.rotate_mid {
                gpu.rotate_x_mq_batched(op.wo_a_out, op.wo_a_out_rot, groups_rank, rows)
                    .map_err(|e| hip_err(format_args!("rotate wo_a_out l{layer}"), e))?;
            }
            project_prepared_rows(
                gpu,
                op.wo_b,
                op.wo_a_out_rot,
                op.wo_a_out,
                op.out,
                op.hidden,
                groups_rank,
                rows,
                Some(x_f16),
                projection,
            )
        }
    }
}

/// Grouped E8 `wo_a` of batched rows: one block-diagonal launch where
/// admitted, else one projection per (row, group).
fn wo_a_rows_e8(
    gpu: &mut Gpu,
    op: &OLoraOp<'_>,
    x_f16: &GpuTensor,
    route: &RowsProjection,
) -> Result<(), DispatchError> {
    let (wo_a, rows, n_groups, rank, per_group_in) = (
        op.wo_a,
        op.rows,
        op.n_groups,
        op.o_lora_rank,
        op.per_group_in,
    );
    let e = |what: &'static str| {
        let l = op.layer;
        move |err: hip_bridge::HipError| {
            hip_err(format_args!("wo_per_group_batched_e8 l{l}: {what}"), err)
        }
    };
    // At B=1 the grouped WMMA cost 14.61 ms against the decode grouped GEMV's
    // 4.23 ms: the largest term between a batched forward and an AR step.
    if wo_a.dtype == DType::MFP4G32E8SOA
        && route.gfx1201_e8
        && gpu.arch_caps.is_gfx1201()
        && per_group_in % 256 == 0
    {
        gpu.deepseek4_convert_f32_to_f16(
            op.raw_rot,
            x_f16,
            (rows * n_groups * per_group_in) as i64,
        )
        .map_err(e("convert_f32_to_f16 (gfx1201 grouped E8 WMMA)"))?;
        return gpu
            .gemm_mfp4g32_e8_soa_grouped_wmma_gfx1201_f16(
                wo_a,
                x_f16,
                op.wo_a_out,
                n_groups,
                rank,
                per_group_in,
                rows,
            )
            .map_err(e("grouped gfx1201 E8 O-LoRA A"));
    }
    if wo_a.dtype == DType::MFP4G32E8SOA && route.e8_batched_gemv(&gpu.arch, rows, per_group_in) {
        return gpu
            .gemv_mfp4g32_e8_soa_grouped_batched_gfx1151(
                wo_a,
                op.raw_rot,
                op.wo_a_out,
                rows,
                n_groups,
                rank,
                per_group_in,
            )
            .map_err(e("grouped batched E8 O-LoRA A"));
    }
    if wo_a.dtype == DType::MFP4G32E8SOA && rows > 16 && route.e8_b2 {
        return gpu
            .gemm_mfp4g32_e8_soa_grouped_wmma_b2(
                wo_a,
                op.raw_rot,
                op.wo_a_out,
                n_groups,
                rank,
                per_group_in,
                rows,
            )
            .map_err(e("grouped batched E8 O-LoRA A B2"));
    }
    if wo_a.dtype == DType::MFP4G32E8SOA && gpu.arch_caps.is_gfx1151() {
        return gpu
            .gemm_mfp4g32_e8_soa_grouped_wmma(
                wo_a,
                op.raw_rot,
                op.wo_a_out,
                n_groups,
                rank,
                per_group_in,
                rows,
            )
            .map_err(e("grouped batched E8 O-LoRA A"));
    }
    let group_weight_bytes = rank
        * wo_a
            .dtype
            .row_bytes(per_group_in)
            .expect("E8 row geometry is defined for K > 0");
    let input_stride = n_groups * per_group_in;
    let output_stride = n_groups * rank;
    for b in 0..rows {
        for g in 0..n_groups {
            let w = wo_a.sub_offset(g * group_weight_bytes, group_weight_bytes);
            let x_off = b * input_stride + g * per_group_in;
            let x_rot = op.raw_rot.sub_offset(x_off, per_group_in);
            let x_plain = op.raw.sub_offset(x_off, per_group_in);
            let y = op.wo_a_out.sub_offset(b * output_stride + g * rank, rank);
            project_prepared(
                gpu,
                &w,
                &x_rot,
                &x_plain,
                &y,
                rank,
                per_group_in,
                route.e8_soa,
            )?;
        }
    }
    Ok(())
}

fn decode_olora_a(
    gpu: &mut Gpu,
    op: &OLoraOp<'_>,
    olora_a: OloraA,
    backend: &dyn LatentAttentionBackend,
    e8_soa: E8SoaKernel,
) -> Result<(), DispatchError> {
    let layer = op.layer;
    let (groups, rank, per_group_in) = (op.n_groups, op.o_lora_rank, op.per_group_in);
    match olora_a {
        OloraA::Mq4Grouped => gpu
            .wo_per_group_batched_hfq4g256(
                op.wo_a,
                op.raw_rot,
                op.wo_a_out,
                groups as i32,
                rank as i32,
                per_group_in as i32,
                1,
            )
            .map_err(|e| hip_err(format!("grouped MQ4 wo_a l{layer}"), e)),
        OloraA::E8Gfx1151 => gpu
            .gemv_mfp4g32_e8_soa_grouped_gfx1151(
                op.wo_a,
                op.raw_rot,
                op.wo_a_out,
                groups,
                rank,
                per_group_in,
            )
            .map_err(|e| hip_err(format!("grouped E8 wo_a l{layer}"), e)),
        OloraA::E8Gfx1201 => gpu
            .gemv_mfp4g32_e8_soa_grouped_gfx1201(
                op.wo_a,
                op.raw_rot,
                op.wo_a_out,
                groups,
                rank,
                per_group_in,
            )
            .map_err(|e| hip_err(format!("grouped gfx1201 E8 wo_a l{layer}"), e)),
        OloraA::E8Gfx1100 => gpu
            .gemv_mfp4g32_e8_soa_grouped_gfx1100(
                op.wo_a,
                op.raw_rot,
                op.wo_a_out,
                groups,
                rank,
                per_group_in,
            )
            .map_err(|e| hip_err(format!("grouped gfx1100 E8 wo_a l{layer}"), e)),
        OloraA::E8Backend => backend
            .grouped_olora_e8(
                gpu,
                op.wo_a,
                op.raw_rot,
                op.wo_a_out,
                groups,
                rank,
                per_group_in,
            )
            .map_err(DispatchError::Hip),
        OloraA::PerGroup { group_bytes } => {
            for g in 0..groups {
                let raw_view = op.raw.sub_offset(g * per_group_in, per_group_in);
                let rot_view = op.raw_rot.sub_offset(g * per_group_in, per_group_in);
                let wo_a_view = if op.wo_a.dtype == DType::F32 {
                    // sub_offset scales F32 by element; gemv_f32 wants 2D [m, k].
                    let mut view = op.wo_a.sub_offset(g * group_bytes, group_bytes);
                    view.shape = vec![rank, per_group_in];
                    view
                } else {
                    op.wo_a.sub_offset(g * group_bytes, group_bytes)
                };
                let out_view = op.wo_a_out.sub_offset(g * rank, rank);
                project_prepared(
                    gpu,
                    &wo_a_view,
                    &rot_view,
                    &raw_view,
                    &out_view,
                    rank,
                    per_group_in,
                    e8_soa,
                )?;
            }
            Ok(())
        }
    }
}
