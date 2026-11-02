// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Gpu` launchers for the lane-batched FullAttention verify chain of the
//! DFlash continuous-batching trunk (exact gfx1201, native fp8-E4M3 KV).
//!
//! The multi-request verify chunk keeps every lane's rows in ONE shared
//! scratch (lane `i` owns rows `[row0_i, row0_i + n_i)`), so the per-lane
//! `fp8 KV write (K, V) + VerifyAttn tile + reduce` sequence (4 launches per
//! lane per FullAttention layer) collapses to one launch per stage per layer:
//!
//! * [`Gpu::kv_cache_write_fp8_e4m3_lanes`] — K, then V (two launches for the
//!   whole chunk), each lane's cache / source / positions in a by-value
//!   kernarg lane table (`kernels/src/kv_cache_write_fp8_lanes.hip`);
//! * [`Gpu::attention_verify_gqa_fp8_lanes`] +
//!   [`Gpu::attention_verify_reduce_lanes`] — lane-table twins of
//!   `attention_verify_gqa_fp8_gfx1201` / `attention_verify_reduce_gfx1201`
//!   (`kernels/src/attention_verify_lanes.gfx1201.hip`, compiled in one
//!   module with the original so `va_verify_gqa` is reused, not copied).
//!
//! Every launcher is byte-identical to the per-lane singleton launches over
//! the same views (see the kernel headers). Launched through
//! `launch_maybe_blob` so the kernarg bytes are retained under capture.
//! Self-contained via `include_str!` (no shared-registry edits).

use crate::dispatch::{Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult, KernargBlob};
use std::ffi::c_void;

/// Lanes per launch (`VA_LANES_MAX` / `KVW_LANES_MAX` in the kernels).
pub const LANE_FA_MAX: usize = 16;
/// Largest lane row count the singleton VerifyAttn route admits
/// (`VERIFY_GQA_MAX_ROWS` in `attention.rs`).
pub const LANE_FA_MAX_ROWS: usize = 32;
/// f32 elements per `[row][head][tile]` partials slot (`2 + head_dim` at the
/// only admitted head dim, 256).
pub const LANE_FA_PARTIALS_STRIDE: usize = 258;
/// Keys per VerifyAttn tile (`VA_TILE`).
const LANE_FA_TILE: usize = 128;
/// Query rows one VerifyAttn workgroup serves (`VERIFY_GQA_ROWS`).
const LANE_FA_ROWS_PER_WG: usize = 8;
/// Workgroup target of the gfx1201 split count (`VerifySplitTarget::Workgroups`).
const LANE_FA_SPLIT_TARGET: usize = 256;

/// Module key of the verify + reduce lane kernels.
pub const ATTENTION_VERIFY_LANES_MODULE: &str = "attention_verify_lanes_gfx1201";
/// Verify lane kernel symbol.
pub const ATTENTION_VERIFY_GQA_FP8_LANES_SYMBOL: &str = "attention_verify_gqa_fp8_lanes_gfx1201";
/// Reduce lane kernel symbol.
pub const ATTENTION_VERIFY_REDUCE_LANES_SYMBOL: &str = "attention_verify_reduce_lanes_gfx1201";
/// Verify + reduce lane kernel module source: the original gfx1201 VerifyAttn
/// source (`va_verify_gqa` template, `VA_*` constants) followed by the lane
/// entries, so the lane kernel instantiates the identical template.
pub const ATTENTION_VERIFY_LANES_SRC: &str = concat!(
    include_str!("../../../kernels/src/attention_verify_gqa.gfx1201.hip"),
    "\n",
    include_str!("../../../kernels/src/attention_verify_lanes.gfx1201.hip")
);
/// fp8 KV write lane kernel source / module key / symbol.
pub const KV_CACHE_WRITE_FP8_LANES_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_fp8_lanes.hip");
pub const KV_CACHE_WRITE_FP8_LANES_SYMBOL: &str = "kv_cache_write_fp8_e4m3_lanes";

/// `#[repr(C)]` mirror of `struct VaLaneTable` (the by-value kernel argument):
/// 5 x 16 x u64 pointers, 3 x 16 x i32 (640 + 192 = 832 B, 8-aligned).
#[repr(C)]
struct VaLaneTableArgs {
    q: [u64; LANE_FA_MAX],
    k_cache: [u64; LANE_FA_MAX],
    v_cache: [u64; LANE_FA_MAX],
    partials: [u64; LANE_FA_MAX],
    positions: [u64; LANE_FA_MAX],
    n_rows: [i32; LANE_FA_MAX],
    max_tiles: [i32; LANE_FA_MAX],
    batch_offset: [i32; LANE_FA_MAX],
}

/// `#[repr(C)]` mirror of `struct VaReduceLaneTable`: 3 x 16 x u64, 3 x 16 x
/// i32 (384 + 192 = 576 B).
#[repr(C)]
struct VaReduceLaneTableArgs {
    partials: [u64; LANE_FA_MAX],
    out: [u64; LANE_FA_MAX],
    positions: [u64; LANE_FA_MAX],
    n_rows: [i32; LANE_FA_MAX],
    max_tiles: [i32; LANE_FA_MAX],
    batch_offset: [i32; LANE_FA_MAX],
}

/// `#[repr(C)]` mirror of `struct KvWriteLaneTable`: 3 x 16 x u64, 16 x i32
/// (384 + 64 = 448 B).
#[repr(C)]
struct KvWriteLaneTableArgs {
    dst: [u64; LANE_FA_MAX],
    src: [u64; LANE_FA_MAX],
    positions: [u64; LANE_FA_MAX],
    n: [i32; LANE_FA_MAX],
}

/// One lane of a lane-batched fp8 KV write: `rows` rows of `src` (`[rows,
/// n_kv_heads * head_dim]` f32) into `cache` at `positions[0..rows]`.
pub struct LaneFaWrite<'a> {
    pub cache: &'a GpuTensor,
    pub src: &'a GpuTensor,
    pub positions: &'a GpuTensor,
    pub rows: usize,
}

/// One lane of a lane-batched VerifyAttn attend. `max_ctx_len` /
/// `physical_cap` are exactly the singleton `attention_flash_fp8_e4m3_tile_batched`
/// arguments of that lane (`max_ctx_len = start_pos + rows`).
#[derive(Clone, Copy)]
pub struct LaneFaAttend<'a> {
    pub q: &'a GpuTensor,
    pub k_cache: &'a GpuTensor,
    pub v_cache: &'a GpuTensor,
    pub out: &'a GpuTensor,
    pub positions: &'a GpuTensor,
    pub rows: usize,
    pub max_ctx_len: usize,
    pub physical_cap: usize,
}

/// The singleton's native-KV capacity guard (`check_native_kv_capacity`):
/// VMM owners are skipped, others must hold `tokens * row_bytes` bytes.
fn check_native_kv_capacity(
    cache: &GpuTensor,
    tokens: usize,
    row_bytes: usize,
    what: &'static str,
) -> HipResult<()> {
    if cache.buf.is_vmm_owner() {
        return Ok(());
    }
    let need = tokens.saturating_mul(row_bytes);
    let have = cache.buf.size();
    if have < need {
        return Err(HipError::new(
            0,
            &format!(
                "{what}: cache holds {have} bytes, need {need} (tokens={tokens} row={row_bytes})"
            ),
        ));
    }
    Ok(())
}

/// `max_tiles` of a lane's singleton launch.
fn lane_max_tiles(max_ctx_len: usize) -> usize {
    max_ctx_len.div_ceil(LANE_FA_TILE)
}

/// The singleton's split count (`verify_gqa_splits`, gfx1201
/// `Workgroups(256)`) for a lane of `rows` rows. Output bytes never depend on
/// it; the lane launch uses the max over lanes as grid.y.
fn lane_splits(max_tiles: usize, rows: usize, n_kv_heads: usize) -> usize {
    let per_split = rows.div_ceil(LANE_FA_ROWS_PER_WG) * n_kv_heads;
    let cap = LANE_FA_SPLIT_TARGET.div_ceil(per_split.max(1)).max(1);
    max_tiles.clamp(1, cap)
}

impl Gpu {
    /// f32 elements of partials the lane-batched attend of these lanes needs:
    /// lane `i` owns a disjoint region of `rows_i * n_heads * max_tiles_i *
    /// 258` floats (its singleton's own partials extent), laid out one after
    /// the other in lane order. `lanes[i] = (rows, max_ctx_len)`.
    pub fn lane_fa_partials_len(n_heads: usize, lanes: &[(usize, usize)]) -> usize {
        lanes
            .iter()
            .map(|&(rows, ctx)| rows * n_heads * lane_max_tiles(ctx) * LANE_FA_PARTIALS_STRIDE)
            .sum()
    }

    /// May the lane-batched FullAttention fp8 verify chain run on this device
    /// and process state for lanes of `lane_rows[i]` rows each? Exact gfx1201;
    /// not graph capture, replay recording or G0 observation; and every
    /// device-level condition of the singleton `try_attention_verify_gqa`
    /// fp8 route (`kernel.verify_attn`, head_dim 256, GQA group 6, tile 128,
    /// a gfx1201 fp8 verify kernel, each lane 1..=32 rows). The caller adds
    /// the per-chunk conditions (every lane ChainVerify on native fp8 KV, no
    /// fused prep, no FA2 arm, ...).
    pub fn lane_fa_batch_admitted(
        &self,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        lane_rows: &[usize],
    ) -> bool {
        self.arch_caps.is_gfx1201()
            && self.flags.verify_attn
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
            && !self.replay.is_g0_observing()
            && self.attn_tile_size() == LANE_FA_TILE
            && head_dim == 256
            && n_kv_heads > 0
            && n_heads == 6 * n_kv_heads
            && (1..=LANE_FA_MAX).contains(&lane_rows.len())
            && lane_rows
                .iter()
                .all(|&r| (1..=LANE_FA_MAX_ROWS).contains(&r))
    }

    /// Per-lane `(partials base in f32 elements, max_tiles)`; errors when
    /// `partials` cannot hold every lane's region.
    fn lane_fa_partials_layout(
        n_heads: usize,
        lanes: &[(usize, usize)],
        partials: &GpuTensor,
        what: &'static str,
    ) -> HipResult<Vec<(usize, usize)>> {
        let need = Self::lane_fa_partials_len(n_heads, lanes);
        if partials.numel() < need {
            return Err(HipError::new(
                0,
                &format!(
                    "{what}: partials hold {} floats, lane layout needs {need}",
                    partials.numel()
                ),
            ));
        }
        let mut base = 0usize;
        let mut out = Vec::with_capacity(lanes.len());
        for &(rows, ctx) in lanes {
            let mt = lane_max_tiles(ctx);
            out.push((base, mt));
            base += rows * n_heads * mt * LANE_FA_PARTIALS_STRIDE;
        }
        Ok(out)
    }

    /// Native fp8-E4M3 KV write of `lanes.len()` (1..=16) lanes in ONE launch:
    /// byte-identical to one `kv_cache_write_fp8_e4m3_batched` per lane (see
    /// the kernel header). Exact gfx1201 only (the kernel is).
    pub fn kv_cache_write_fp8_e4m3_lanes(
        &mut self,
        lanes: &[LaneFaWrite<'_>],
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        assert!(
            (1..=LANE_FA_MAX).contains(&lanes.len()),
            "kv_cache_write_fp8_e4m3_lanes: {} lanes outside 1..={LANE_FA_MAX}",
            lanes.len()
        );
        self.bind_thread()?;
        let row = crate::attention::fp8_e4m3_row_bytes(n_kv_heads, head_dim);
        for l in lanes {
            check_native_kv_capacity(l.cache, 1, row, "kv_cache_write_fp8_e4m3_lanes")?;
        }
        self.ensure_kernel(
            KV_CACHE_WRITE_FP8_LANES_SYMBOL,
            KV_CACHE_WRITE_FP8_LANES_SRC,
            KV_CACHE_WRITE_FP8_LANES_SYMBOL,
        )?;
        let mut args = KvWriteLaneTableArgs {
            dst: [0; LANE_FA_MAX],
            src: [0; LANE_FA_MAX],
            positions: [0; LANE_FA_MAX],
            n: [0; LANE_FA_MAX],
        };
        let mut max_n = 0usize;
        for (i, l) in lanes.iter().enumerate() {
            args.dst[i] = l.cache.buf.as_ptr() as u64;
            args.src[i] = l.src.buf.as_ptr() as u64;
            args.positions[i] = l.positions.buf.as_ptr() as u64;
            args.n[i] = l.rows as i32;
            max_n = max_n.max(l.rows);
        }
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut args as *mut _ as *mut c_void,
            &nkv as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
        ];
        let blob_args = &args;
        self.launch_maybe_blob(
            KV_CACHE_WRITE_FP8_LANES_SYMBOL,
            [n_kv_heads as u32, max_n as u32, lanes.len() as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                for arr in [&blob_args.dst, &blob_args.src, &blob_args.positions] {
                    for v in arr {
                        b.push_u64(*v);
                    }
                }
                for v in &blob_args.n {
                    b.push_i32(*v);
                }
                b.push_i32(nkv);
                b.push_i32(hd);
                b
            },
        )
    }

    /// VerifyAttn fp8 tile of `lanes.len()` (1..=16) lanes in ONE launch:
    /// byte-identical partials to one `attention_verify_gqa_fp8_gfx1201` per
    /// lane with that lane's `max_tiles`, written to the lane's disjoint region
    /// of `partials` (layout: [`Gpu::lane_fa_partials_len`]). Run
    /// [`Gpu::attention_verify_reduce_lanes`] with the same lanes and
    /// `partials` afterwards. The caller guarantees
    /// [`Gpu::lane_fa_batch_admitted`].
    pub fn attention_verify_gqa_fp8_lanes(
        &mut self,
        lanes: &[LaneFaAttend<'_>],
        partials: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        assert!(
            (1..=LANE_FA_MAX).contains(&lanes.len()),
            "attention_verify_gqa_fp8_lanes: {} lanes outside 1..={LANE_FA_MAX}",
            lanes.len()
        );
        self.bind_thread()?;
        let row = crate::attention::fp8_e4m3_row_bytes(n_kv_heads, head_dim);
        let shape: Vec<(usize, usize)> = lanes.iter().map(|l| (l.rows, l.max_ctx_len)).collect();
        for l in lanes {
            if l.max_ctx_len == 0 || !(1..=LANE_FA_MAX_ROWS).contains(&l.rows) {
                return Err(HipError::new(
                    0,
                    "attention_verify_gqa_fp8_lanes: lane rows / context outside the VerifyAttn route",
                ));
            }
            check_native_kv_capacity(
                l.k_cache,
                l.physical_cap,
                row,
                "attention_verify_gqa_fp8_lanes",
            )?;
            check_native_kv_capacity(
                l.v_cache,
                l.physical_cap,
                row,
                "attention_verify_gqa_fp8_lanes",
            )?;
        }
        let layout = Self::lane_fa_partials_layout(
            n_heads,
            &shape,
            partials,
            "attention_verify_gqa_fp8_lanes",
        )?;
        self.ensure_kernel(
            ATTENTION_VERIFY_LANES_MODULE,
            ATTENTION_VERIFY_LANES_SRC,
            ATTENTION_VERIFY_GQA_FP8_LANES_SYMBOL,
        )?;
        let mut args = VaLaneTableArgs {
            q: [0; LANE_FA_MAX],
            k_cache: [0; LANE_FA_MAX],
            v_cache: [0; LANE_FA_MAX],
            partials: [0; LANE_FA_MAX],
            positions: [0; LANE_FA_MAX],
            n_rows: [0; LANE_FA_MAX],
            max_tiles: [0; LANE_FA_MAX],
            batch_offset: [0; LANE_FA_MAX],
        };
        let p_base = partials.buf.as_ptr() as *mut u8;
        let (mut grid_x, mut grid_y) = (1usize, 1usize);
        for (i, (l, &(base, mt))) in lanes.iter().zip(&layout).enumerate() {
            args.q[i] = l.q.buf.as_ptr() as u64;
            args.k_cache[i] = l.k_cache.buf.as_ptr() as u64;
            args.v_cache[i] = l.v_cache.buf.as_ptr() as u64;
            args.partials[i] = unsafe { p_base.add(base * 4) } as u64;
            args.positions[i] = l.positions.buf.as_ptr() as u64;
            args.n_rows[i] = l.rows as i32;
            args.max_tiles[i] = mt as i32;
            args.batch_offset[i] = 0;
            grid_x = grid_x.max(l.rows.div_ceil(LANE_FA_ROWS_PER_WG) * n_kv_heads);
            grid_y = grid_y.max(lane_splits(mt, l.rows, n_kv_heads));
        }
        let nkv = n_kv_heads as i32;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut params: Vec<*mut c_void> = vec![
            &mut args as *mut _ as *mut c_void,
            &nkv as *const _ as *mut c_void,
            &scale as *const _ as *mut c_void,
        ];
        let blob_args = &args;
        self.launch_maybe_blob(
            ATTENTION_VERIFY_GQA_FP8_LANES_SYMBOL,
            [grid_x as u32, grid_y as u32, lanes.len() as u32],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                for arr in [
                    &blob_args.q,
                    &blob_args.k_cache,
                    &blob_args.v_cache,
                    &blob_args.partials,
                    &blob_args.positions,
                ] {
                    for v in arr {
                        b.push_u64(*v);
                    }
                }
                for arr in [
                    &blob_args.n_rows,
                    &blob_args.max_tiles,
                    &blob_args.batch_offset,
                ] {
                    for v in arr {
                        b.push_i32(*v);
                    }
                }
                b.push_i32(nkv);
                b.push_f32(scale);
                b
            },
        )
    }

    /// VerifyAttn reduce of `lanes.len()` (1..=16) lanes in ONE launch
    /// (`attention_verify_reduce_gfx1201` per lane): each lane's attention
    /// output rows land in its `out` exactly where the singleton writes them.
    pub fn attention_verify_reduce_lanes(
        &mut self,
        lanes: &[LaneFaAttend<'_>],
        partials: &GpuTensor,
        n_heads: usize,
    ) -> HipResult<()> {
        assert!(
            (1..=LANE_FA_MAX).contains(&lanes.len()),
            "attention_verify_reduce_lanes: {} lanes outside 1..={LANE_FA_MAX}",
            lanes.len()
        );
        self.bind_thread()?;
        let shape: Vec<(usize, usize)> = lanes.iter().map(|l| (l.rows, l.max_ctx_len)).collect();
        let layout = Self::lane_fa_partials_layout(
            n_heads,
            &shape,
            partials,
            "attention_verify_reduce_lanes",
        )?;
        self.ensure_kernel(
            ATTENTION_VERIFY_LANES_MODULE,
            ATTENTION_VERIFY_LANES_SRC,
            ATTENTION_VERIFY_REDUCE_LANES_SYMBOL,
        )?;
        let mut args = VaReduceLaneTableArgs {
            partials: [0; LANE_FA_MAX],
            out: [0; LANE_FA_MAX],
            positions: [0; LANE_FA_MAX],
            n_rows: [0; LANE_FA_MAX],
            max_tiles: [0; LANE_FA_MAX],
            batch_offset: [0; LANE_FA_MAX],
        };
        let p_base = partials.buf.as_ptr() as *mut u8;
        let mut max_rows = 1usize;
        for (i, (l, &(base, mt))) in lanes.iter().zip(&layout).enumerate() {
            args.partials[i] = unsafe { p_base.add(base * 4) } as u64;
            args.out[i] = l.out.buf.as_ptr() as u64;
            args.positions[i] = l.positions.buf.as_ptr() as u64;
            args.n_rows[i] = l.rows as i32;
            args.max_tiles[i] = mt as i32;
            args.batch_offset[i] = 0;
            max_rows = max_rows.max(l.rows);
        }
        let nh = n_heads as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut args as *mut _ as *mut c_void,
            &nh as *const _ as *mut c_void,
        ];
        let blob_args = &args;
        self.launch_maybe_blob(
            ATTENTION_VERIFY_REDUCE_LANES_SYMBOL,
            [n_heads as u32, max_rows as u32, lanes.len() as u32],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                for arr in [&blob_args.partials, &blob_args.out, &blob_args.positions] {
                    for v in arr {
                        b.push_u64(*v);
                    }
                }
                for arr in [
                    &blob_args.n_rows,
                    &blob_args.max_tiles,
                    &blob_args.batch_offset,
                ] {
                    for v in arr {
                        b.push_i32(*v);
                    }
                }
                b.push_i32(nh);
                b
            },
        )
    }

    /// Which lanes of `lanes` the singleton attend serves with the native-fp8
    /// split-KV verifier ([`Gpu::fp8_fa2_split_verify_admitted`] at that
    /// lane's context over the singleton's `single_partials`): those lanes
    /// keep their own singleton launch so batched == singleton holds.
    pub fn lane_fa_split_lanes(
        &self,
        lanes: &[LaneFaAttend<'_>],
        single_partials: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Vec<bool> {
        lanes
            .iter()
            .map(|l| {
                self.fp8_fa2_split_verify_admitted(
                    l.max_ctx_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    l.rows,
                    single_partials.numel(),
                )
                .is_some()
            })
            .collect()
    }

    /// The attend of a lane-batched chunk, equal to one singleton
    /// `attention_flash_fp8_e4m3_tile_batched` per lane. Lanes the singleton
    /// serves with the native-fp8 split-KV verifier
    /// ([`Self::lane_fa_split_lanes`]) run exactly that singleton launch over
    /// `single_partials` (the model scratch's `flash_partials`, the
    /// singleton's own); every other lane shares ONE VerifyAttn tile launch
    /// and one reduce over `lane_partials`, laid out over those lanes only,
    /// in lane order ([`Gpu::lane_fa_partials_len`]). The caller guarantees
    /// [`Gpu::lane_fa_batch_admitted`].
    pub fn attention_verify_fp8_lanes_routed(
        &mut self,
        lanes: &[LaneFaAttend<'_>],
        lane_partials: &GpuTensor,
        single_partials: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        let split = self.lane_fa_split_lanes(lanes, single_partials, n_heads, n_kv_heads, head_dim);
        let mut plain: Vec<LaneFaAttend<'_>> = Vec::with_capacity(lanes.len());
        for (l, &is_split) in lanes.iter().zip(&split) {
            if is_split {
                self.attention_flash_fp8_e4m3_tile_batched(
                    l.q,
                    l.k_cache,
                    l.v_cache,
                    l.out,
                    l.positions,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    l.physical_cap,
                    l.max_ctx_len,
                    l.rows,
                    single_partials,
                    None,
                    0,
                    0,
                    None,
                )?;
            } else {
                plain.push(*l);
            }
        }
        if plain.is_empty() {
            return Ok(());
        }
        self.attention_verify_gqa_fp8_lanes(&plain, lane_partials, n_heads, n_kv_heads, head_dim)?;
        self.attention_verify_reduce_lanes(&plain, lane_partials, n_heads)
    }
}
