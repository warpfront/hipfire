// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sealed expert-slab route (`MoeRouteCapability::SlabCodecGrouped`) stage
//! helpers shared by the decode and prefill lowerings.
//!
//! The layer's experts live in one slab arena and carry a codec each: MQ4
//! symmetric plus at most one cold codec per layer (MQ3-I4 W3 on gfx1201 and gfx1151,
//! or E8P-I4 on gfx1201 and gfx1100, and on gfx1151 once
//! `rdna_compute::slab_admission::SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED` is true; W2 nowhere).
//! Two kernel families serve them:
//!
//! * `route_rows <= Gpu::few_row_limit()` (decode, MTP verify, a one-row forward): the K1 lane-union
//!   GEMV over codec buckets ([`rdna_compute::slab_moe`]). The layer sits in the
//!   path-1 stage shape (gate/up F32 rows, SwiGLU activation + 128-wide
//!   rotation, unweighted F32 down rows, top-10 combine); only the two GEMVs
//!   are K1 launches.
//! * longer prefill: the K2 grouped IU4 kernels, MQ4 and the cold codec, over
//!   the same grouped buffers with codec-masked tile ids (disjoint tiles).
//!
//! A layer whose experts all share one codec carries no codec table; a mixed
//! layer's table is the layer's dtype-tag tensor: `[512]` u8 codec bytes
//! (cold-codec membership only: `0` = MQ4, `1` = cold) followed by the same
//! codes as `[512]` i32 (the table-driven tile mask's owner table). The
//! tensor's logical shape is the u8 table; its allocation is
//! [`CODEC_TAG_ALLOC_BYTES`]. Buckets: cold = 0, MQ4 = 1.

use super::layer_ops::hip;
use crate::families::moe::MoeDtypes;
use crate::types::DispatchError;
use rdna_compute::slab_moe::{
    k1_rows8_cold_codec, SlabBuckets, SLAB_EXPERTS, SLAB_NARROW_MAX_LANES,
};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Lanes of one unchanged narrow K1 launch. Wider admitted calls use fixed
/// eight-row tile descriptors for the whole window, never host launch groups.
pub const K1_MAX_ROWS: usize = SLAB_NARROW_MAX_LANES;
/// Bytes of the u8 codec table.
pub const CODEC_U8_BYTES: usize = SLAB_EXPERTS;
/// Bytes of the whole codec-tag allocation: u8 table, then i32 table.
pub const CODEC_TAG_ALLOC_BYTES: usize = CODEC_U8_BYTES + 4 * SLAB_EXPERTS;

fn refuse(what: &str) -> DispatchError {
    DispatchError::Hip(format!("sealed expert-slab route: {what}"))
}

/// Whether `arch` hosts the sealed expert-slab route (the arch admits MQ4).
pub(crate) fn arch_supported(arch: &str) -> bool {
    rdna_compute::qwen4_slab_codec_admitted(arch, 0)
}

/// A layer's K1 / K2 codec layout: the bucket map and the raw codec id of its
/// cold bucket (the `cold_codec` the runtime entries take).
#[derive(Clone, Copy)]
pub(crate) struct SlabLayer<'a> {
    pub buckets: SlabBuckets<'a>,
    pub cold_codec: u8,
}

/// The codec layout of a layer on `arch`: its codec table when mixed, else the
/// one codec its (uniform) routed dtype names, which `arch` must admit. Two
/// buckets: cold = 0, MQ4 = 1. The cold codec is the policy's declaration,
/// validated against the dtype tables by route selection; nothing here scans
/// a table.
pub(crate) fn layer<'a>(
    arch: &str,
    policy: Option<&crate::families::moe::MoeRoutePolicy>,
    dtypes: &MoeDtypes<'_>,
    tags: Option<&'a GpuTensor>,
) -> Result<SlabLayer<'a>, DispatchError> {
    let policy = policy
        .filter(|policy| policy.capability.is_slab())
        .ok_or_else(|| refuse("the layer declares no slab route"))?;
    let cold_codec = crate::families::moe::slab_launch_cold_codec(arch, policy.slab_cold_codec)
        .ok_or_else(|| {
            refuse(&format!(
                "{arch} does not admit the layer's cold codec {:?}",
                policy.slab_cold_codec
            ))
        })?;
    let buckets = match tags {
        Some(table) => {
            if table.buf.size() < CODEC_TAG_ALLOC_BYTES {
                return Err(refuse(
                    "codec tag allocation is smaller than the u8 + i32 tables",
                ));
            }
            SlabBuckets::Table(table)
        }
        None => crate::families::moe::slab_bucket_of_pair(
            arch,
            dtypes.routed_gate_up,
            dtypes.routed_down,
        )
        .map(SlabBuckets::Uniform)
        .ok_or_else(|| {
            refuse(&format!(
                "uniform layer dtypes {:?}/{:?} are not a slab codec of {arch}",
                dtypes.routed_gate_up, dtypes.routed_down
            ))
        })?,
    };
    Ok(SlabLayer {
        buckets,
        cold_codec,
    })
}

/// The `[512]` i32 codec table of a mixed layer's tag tensor (the table-driven
/// tile mask's owner table).
pub(crate) fn codec_i32(tags: &GpuTensor) -> Result<GpuTensor, DispatchError> {
    if tags.buf.size() < CODEC_TAG_ALLOC_BYTES {
        return Err(refuse(
            "codec tag allocation is smaller than the u8 + i32 tables",
        ));
    }
    Ok(GpuTensor {
        buf: tags.buf.byte_view(CODEC_U8_BYTES, 4 * SLAB_EXPERTS),
        shape: vec![SLAB_EXPERTS],
        dtype: DType::F32,
    })
}

fn k1_check_lanes(gpu: &Gpu, what: &str, lanes: usize) -> Result<(), DispatchError> {
    let limit = gpu
        .few_row_limit()
        .min(rdna_compute::slab_moe::SLAB_MAX_LANES);
    if lanes == 0 || lanes > limit {
        return Err(refuse(&format!("{what}: lanes must be 1..={limit}")));
    }
    Ok(())
}

/// K1 gate/up of `lanes` lanes (up to the shared per-architecture exact cap):
/// union of `topk` and launches every non-empty bucket. `xr` `[lanes, 2560]`
/// F32 rotated activations; `gate_out`/`up_out` `[lanes * 10, 640]` F32.
///
/// Wider calls require a tile-capable K1 twin before any planner mutation.
/// One union-planner launch writes all eight-row tile descriptors; each codec
/// projection consumes that whole window in one launch. The descriptors remain
/// immutable across SwiGLU and are reused by [`k1_down`].
///
/// One lane of an MQ4 or W3 layer on gfx1151 (unless `HIPFIRE_QWEN4_SLAB_K1_C1=0`) takes K1-C1
/// instead: one launch for both codecs straight from `topk`, no union plan (bitwise the K1 result).
/// E8P layers keep the K1 bucket entries.
#[allow(clippy::too_many_arguments)]
pub(crate) fn k1_gate_up(
    gpu: &mut Gpu,
    layer: SlabLayer<'_>,
    gate_up_ptrs: &GpuTensor,
    topk: &GpuTensor,
    xr: &GpuTensor,
    gate_out: &GpuTensor,
    up_out: &GpuTensor,
    lanes: usize,
) -> Result<(), DispatchError> {
    k1_check_lanes(gpu, "k1_gate_up", lanes)?;
    if lanes == 1 && gpu.slab_k1_c1() && k1_rows8_cold_codec(layer.cold_codec) {
        return hip(gpu.mq_i4_moe_gemv_slab_c1(
            false,
            layer.buckets,
            gate_up_ptrs,
            topk,
            xr,
            gate_out,
            Some(up_out),
            layer.cold_codec,
        ));
    }
    if lanes > K1_MAX_ROWS && !gpu.qwen4_slab_wide_twin(lanes, layer.cold_codec) {
        return Err(refuse(
            "k1_gate_up: wide lanes require an admitted tile-capable K1 twin",
        ));
    }
    hip(gpu.qwen4_slab_union_plan(topk, layer.buckets, lanes))?;
    hip(gpu.mq_i4_moe_gemv_slab(
        false,
        layer.buckets,
        gate_up_ptrs,
        xr,
        gate_out,
        Some(up_out),
        lanes,
        layer.cold_codec,
    ))
}

/// K1 down of `lanes` lanes (up to the shared per-architecture exact cap): `xrot`
/// `[lanes * 10, 640]` F32, `y0` `[lanes * 10, 2560]` F32 unweighted.
///
/// Narrow and wide tiled calls consume the immutable union descriptors from
/// gate/up, without replanning after SwiGLU. K1-C1 at one lane reads `topk`
/// instead and needs no union.
#[allow(clippy::too_many_arguments)]
pub(crate) fn k1_down(
    gpu: &mut Gpu,
    layer: SlabLayer<'_>,
    topk: &GpuTensor,
    down_ptrs: &GpuTensor,
    xrot: &GpuTensor,
    y0: &GpuTensor,
    lanes: usize,
) -> Result<(), DispatchError> {
    k1_check_lanes(gpu, "k1_down", lanes)?;
    if lanes == 1 && gpu.slab_k1_c1() && k1_rows8_cold_codec(layer.cold_codec) {
        return hip(gpu.mq_i4_moe_gemv_slab_c1(
            true,
            layer.buckets,
            down_ptrs,
            topk,
            xrot,
            y0,
            None,
            layer.cold_codec,
        ));
    }
    if lanes > K1_MAX_ROWS && !gpu.qwen4_slab_wide_twin(lanes, layer.cold_codec) {
        return Err(refuse(
            "k1_down: wide lanes require an admitted tile-capable K1 twin",
        ));
    }
    hip(gpu.mq_i4_moe_gemv_slab(
        true,
        layer.buckets,
        down_ptrs,
        xrot,
        y0,
        None,
        lanes,
        layer.cold_codec,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::slab_moe_decode::{self as k1, BucketKey, Codec, Kind};
    use super::K1_MAX_ROWS;
    use rdna_compute::slab_moe as gpu;

    #[test]
    fn runtime_capacity_keeps_the_narrow_oracle_separate() {
        assert_eq!(gpu::SLAB_MAX_LANES, 64);
        assert_eq!(K1_MAX_ROWS, 8);
        // This old CPU planner models one non-tiled union, not a 64-row window.
        assert_eq!(k1::MAX_LANES, 32);
        assert!(K1_MAX_ROWS <= k1::MAX_LANES);
    }

    /// The launcher's constants are the certified K1 contract's: symbols in
    /// bucket order (W3 first), geometry, lanes, and the codec byte mapping.
    #[test]
    fn launcher_contract_is_the_k1_contract() {
        for (bucket, codec) in [(0usize, Codec::Mq3I4), (1usize, Codec::Mq4Sym)] {
            assert_eq!(Codec::ORDER[bucket], codec);
            for (down, kind) in [(0usize, Kind::GateUp), (1usize, Kind::Down)] {
                assert_eq!(
                    gpu::MQ_I4_MOE_GEMV_SYMBOLS[bucket][down],
                    BucketKey::new(codec, kind).symbol()
                );
            }
        }
        for bucket in 0..2 {
            for kind in 0..2 {
                assert_eq!(
                    gpu::MQ_I4_MOE_GEMV_ROWTILE_SYMBOLS[bucket][kind],
                    format!("{}_t8", gpu::MQ_I4_MOE_GEMV_SYMBOLS[bucket][kind])
                );
                assert_eq!(
                    gpu::MQ_I4_MOE_GEMV_CN_ROWTILE_SYMBOLS[bucket][kind],
                    format!("{}_t8", gpu::MQ_I4_MOE_GEMV_CN_SYMBOLS[bucket][kind])
                );
            }
        }
        assert_eq!(gpu::SLAB_TOP_K, k1::TOP_K);
        assert_eq!(gpu::SLAB_EXPERTS, k1::MAX_EXPERTS);
        assert_eq!(
            (gpu::SLAB_GATE_UP_M, gpu::SLAB_GATE_UP_K),
            (k1::GATE_UP_M, k1::GATE_UP_K)
        );
        assert_eq!(
            (gpu::SLAB_DOWN_M, gpu::SLAB_DOWN_K),
            (k1::DOWN_M, k1::DOWN_K)
        );
        // Codec bytes of the per-layer table: 1 = W3 -> bucket 0, 0 = MQ4 -> bucket 1.
        assert_eq!(gpu::slab_bucket_of_codec(1), Codec::Mq3I4.order_index());
        assert_eq!(gpu::slab_bucket_of_codec(0), Codec::Mq4Sym.order_index());
    }

    /// The K1-C1 launcher's constants: its two symbols, grid `[M / 8, 10, 1]`, block
    /// `[256, 1, 1]`, the 80-byte bounded kernarg (`n_experts@72`, zero `reserved@76`),
    /// and the shapes it shares with K1.
    #[test]
    fn c1_launcher_contract() {
        assert_eq!(
            gpu::MQ_I4_MOE_GEMV_C1_SYMBOLS,
            [
                "mq_i4_moe_gemv_c1_r256_gate_up",
                "mq_i4_moe_gemv_c1_r128_down"
            ]
        );
        assert_eq!(gpu::MQ_I4_MOE_GEMV_C1_KERNARG_BYTES, 80);
        assert_eq!(
            (
                gpu::MQ_I4_MOE_GEMV_C1_ROWS_PER_WG,
                gpu::MQ_I4_MOE_GEMV_C1_BLOCK
            ),
            (8, 256)
        );
        let gate = gpu::slab_c1_launch(false);
        assert_eq!(gate.grid, [1280 / 8, 10, 1]);
        assert_eq!(gate.block, [256, 1, 1]);
        assert_eq!(
            (
                gate.m as usize,
                gate.k as usize,
                gate.x_row_div as usize,
                gate.y_slot_stride as usize
            ),
            (k1::GATE_UP_M, k1::GATE_UP_K, k1::TOP_K, k1::GATE_UP_M / 2)
        );
        let down = gpu::slab_c1_launch(true);
        assert_eq!(down.grid, [2560 / 8, 10, 1]);
        assert_eq!(down.block, [256, 1, 1]);
        assert_eq!(
            (
                down.m as usize,
                down.k as usize,
                down.x_row_div as usize,
                down.y_slot_stride as usize
            ),
            (k1::DOWN_M, k1::DOWN_K, 1, k1::DOWN_M)
        );
    }

    /// The unchanged non-tiled CPU oracle retains its own 32-lane capacity.
    #[test]
    fn narrow_union_scratch_words_match_the_descriptor_layout() {
        for lanes in 1..=k1::MAX_LANES {
            let s = 10 * lanes;
            assert_eq!(
                rdna_compute::scratch::qwen4_slab_union_words(lanes),
                2 * s + 2 * (s + 1) + 2 * s
            );
        }
    }

    /// Fixed physical descriptors cover a whole window, including a partial
    /// final tile. These are CPU address/extent contracts, not a device receipt.
    #[test]
    fn rowtile_descriptor_geometry_and_tail_guards() {
        use rdna_compute::scratch::{qwen4_slab_rowtile_bytes, qwen4_slab_rowtile_words_checked};

        const S: usize = 8 * k1::TOP_K;
        const WORDS: usize = 6 * S + 2;
        const GUARD: u32 = 0xdead_beef;
        assert_eq!((S, WORDS), (80, 482));
        for lanes in 1..=gpu::SLAB_MAX_LANES {
            let tiles = lanes.div_ceil(K1_MAX_ROWS);
            let words = qwen4_slab_rowtile_words_checked(lanes).unwrap();
            assert_eq!(words, tiles * WORDS);
            assert_eq!(qwen4_slab_rowtile_bytes(lanes), Some(tiles * 1928));
            let mut guarded = vec![GUARD; words + 2];
            let mut visits = vec![0; lanes * k1::TOP_K];
            for tile in 0..tiles {
                let first = tile * K1_MAX_ROWS;
                let live_rows = K1_MAX_ROWS.min(lanes - first);
                let live_slots = live_rows * k1::TOP_K;
                let descriptor = &mut guarded[1 + tile * WORDS..1 + (tile + 1) * WORDS];
                descriptor.fill(0);
                for bucket in 0..2 {
                    let ids = bucket * S;
                    let offsets = 2 * S + bucket * (S + 1);
                    let slots = 4 * S + 2 + bucket * S;
                    assert!(ids + S <= 2 * S);
                    assert!(offsets + S + 1 <= 4 * S + 2);
                    assert!(slots + S <= WORDS);
                }
                // A uniform bucket with ten distinct experts gives every
                // expert <=8 slots. Store absolute slot ids even in the tail.
                let ids = 0;
                let offsets = 2 * S;
                let slots = 4 * S + 2;
                for rank in 0..k1::TOP_K {
                    descriptor[ids + rank] = rank as u32;
                    descriptor[offsets + rank] = (rank * live_rows) as u32;
                    for row in 0..live_rows {
                        let slot = (first + row) * k1::TOP_K + rank;
                        descriptor[slots + rank * live_rows + row] = slot as u32;
                    }
                }
                descriptor[offsets + k1::TOP_K..offsets + S + 1].fill(live_slots as u32);
                assert!(descriptor[ids + k1::TOP_K..ids + S]
                    .iter()
                    .all(|&id| id == 0));
                assert!(descriptor[slots + live_slots..slots + S]
                    .iter()
                    .all(|&slot| slot == 0));
                for rank in 0..k1::TOP_K {
                    let start = descriptor[offsets + rank] as usize;
                    let end = descriptor[offsets + rank + 1] as usize;
                    assert_eq!(end - start, live_rows);
                    assert!(end - start <= K1_MAX_ROWS);
                    assert!(end <= S);
                    for &slot in &descriptor[slots + start..slots + end] {
                        let slot = slot as usize;
                        assert!(slot < lanes * k1::TOP_K);
                        // Full-window X/Y bases: no second tile offset.
                        assert_eq!(slot / k1::TOP_K / K1_MAX_ROWS, tile);
                        for kind in [Kind::GateUp, Kind::Down] {
                            assert!(slot / kind.x_row_div() < kind.x_src_rows(lanes));
                            assert!(
                                slot * kind.y_slot_stride() + kind.y_slot_stride()
                                    <= lanes * k1::TOP_K * kind.y_slot_stride()
                            );
                        }
                        visits[slot] += 1;
                    }
                }
                // The unused codec bucket and its offset sentinel stay empty.
                assert!(descriptor[S..2 * S].iter().all(|&v| v == 0));
                assert!(descriptor[2 * S + S + 1..4 * S + 2].iter().all(|&v| v == 0));
                assert!(descriptor[4 * S + 2 + S..].iter().all(|&v| v == 0));
            }
            assert!(visits.iter().all(|&count| count == 1));
            assert_eq!(guarded[0], GUARD);
            assert_eq!(guarded[words + 1], GUARD);
        }
        assert_eq!(qwen4_slab_rowtile_words_checked(64), Some(3856));
        assert_eq!(qwen4_slab_rowtile_bytes(64), Some(15_424));
    }
}
