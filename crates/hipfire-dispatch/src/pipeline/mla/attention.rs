// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Sparse latent attention, one operation per kernel chain: the ratio-4
//! indexer (query projection and RoPE, head weights, relu-weighted scores
//! over the compressed indexer cache, top-k), the SWA ring write, the
//! compressed-row gather, the attention (one joint softmax over the SWA
//! window, the gathered compressed rows and the sink) and the grouped
//! O-LoRA; the inverse tail RoPE between them is an
//! [`super::query_kv::ApplyTailRopeOp`]. Decode reads positions, window fill
//! and compressed counts from device state so a captured graph replays them.

use rdna_compute::{DType, Gpu, GpuTensor};

use super::{project_prepared, E8SoaKernel};
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

/// Ratio-4 indexer of one decode row.
pub struct SparseIndexOp<'a> {
    pub wq_b: &'a GpuTensor,
    pub weights_proj: &'a GpuTensor,
    pub q_lat: &'a GpuTensor,
    pub q_lat_rot: &'a GpuTensor,
    pub tmp: &'a GpuTensor,
    pub tmp_plain: &'a GpuTensor,
    pub q_idx: &'a GpuTensor,
    pub idx_weights: &'a GpuTensor,
    pub cache: CompressedCache<'a>,
    pub scores: &'a GpuTensor,
    pub topk: &'a GpuTensor,
    /// `[.., n_compressed_4 @2, .., k_active_4 @4, ..]` device state.
    pub attn_state: &'a GpuTensor,
    pub pos: &'a GpuTensor,
    pub top_k: IndexTopK<'a>,
    pub backend: &'a dyn LatentAttentionBackend,
    /// One head per wave (exact gfx1201, H=64 D=128 R=64).
    pub head_parallel_rope: bool,
    /// The head weights are already in `idx_weights` (packed projection).
    pub weights_preprojected: bool,
    pub heads: usize,
    pub head_dim: usize,
    pub top_k_max: usize,
    pub q_lora_rank: usize,
    pub hidden: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub max_compressed: usize,
    pub e8_soa: E8SoaKernel,
    pub layer: usize,
}

impl SparseIndexOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.heads == 0 || self.head_dim == 0 || self.top_k_max == 0 || self.max_compressed == 0
        {
            return Err(DispatchError::Hip("sparse index has empty geometry".into()));
        }
        Ok(())
    }
}

pub fn execute_sparse_index(gpu: &mut Gpu, op: &SparseIndexOp<'_>) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    let (layer, h, d, k) = (op.layer, op.heads, op.head_dim, op.top_k_max);
    let max_compressed = op.max_compressed as i32;
    // 1. q_idx = wq_b · q_lat (M = H*D, K = q_lora_rank).
    project_prepared(
        gpu,
        op.wq_b,
        op.q_lat_rot,
        op.q_lat,
        op.q_idx,
        h * d,
        op.q_lora_rank,
        op.e8_soa,
    )?;
    // 2. Tail RoPE on q_idx with the compressor's theta, at the current
    //    position (the K side was rotated by the indexer compressor).
    if op.head_parallel_rope {
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
            op.e8_soa,
        )?;
    }
    // 4. Relu-weighted scores over the first n_compressed_4 rows; rows past
    //    it score -inf so the top-k ignores them.
    let n_buf = op.attn_state.sub_offset(2, 1);
    let k_buf = op.attn_state.sub_offset(4, 1);
    match op.cache {
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
    let portable_wide = match op.top_k {
        IndexTopK::TwoStage {
            ws_scores,
            ws_indices,
        } => {
            return gpu
                .indexer_top_k_two_stage(
                    op.scores,
                    ws_scores,
                    ws_indices,
                    op.topk,
                    &n_buf,
                    &k_buf,
                    max_compressed,
                    k as i32,
                )
                .map_err(|e| hip_err(format!("idx top_k two-stage l{layer}"), e));
        }
        IndexTopK::BackendParallel { bounded, wide } => {
            if op
                .backend
                .try_indexer_top_k_buf_parallel(
                    gpu, op.scores, op.topk, &n_buf, &k_buf, /*n_idx_heads=*/ 1, k as i32,
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
        op.topk,
        &n_buf,
        &k_buf,
        /*n_idx_heads=*/ 1,
        max_compressed,
        k as i32,
    )
    .map_err(|e| hip_err(format!("idx top_k buf l{layer}"), e))
}

/// A decode route serves exactly one row; every route at least one.
pub(super) fn check_rows(
    rows: usize,
    decode: bool,
    what: &str,
    layer: usize,
) -> Result<(), DispatchError> {
    if rows == 0 || (decode && rows != 1) {
        return Err(DispatchError::Hip(format!(
            "{what} l{layer}: {rows} rows on the {} route",
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
}

impl SwaRingWriteOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let decode = matches!(self.route, SwaRingWriteRoute::Decode { .. });
        check_rows(self.rows, decode, "SWA ring write", self.layer)
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
}

impl SwaAttentionOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let decode = matches!(
            self.route,
            SwaAttentionRoute::Decode { .. } | SwaAttentionRoute::Pos0 { .. }
        );
        check_rows(self.rows, decode, "SWA attention", self.layer)
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
}

impl IdentityGatherOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let decode = matches!(self.route, IdentityGatherRoute::Decode { .. });
        check_rows(self.rows, decode, "identity gather", self.layer)
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
}

impl SparseAttentionOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let decode = matches!(self.route, SparseAttentionRoute::Decode { .. });
        check_rows(self.rows, decode, "sparse attention", self.layer)
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
        check_rows(self.rows, decode, "O-LoRA", self.layer)
    }
}

pub fn execute_o_lora(gpu: &mut Gpu, op: &OLoraOp<'_>) -> Result<(), DispatchError> {
    let layer = op.layer;
    let groups_rank = op.n_groups * op.o_lora_rank;
    match op.route {
        OLoraRoute::Decode {
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
    }
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
