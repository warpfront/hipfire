// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Gpu` launchers for the lane-batched DeltaNet (GDN) verify preamble of the
//! DFlash continuous-batching trunk (exact gfx1201).
//!
//! The multi-request verify chunk keeps every lane's rows in ONE shared
//! scratch (lane `i` owns rows `[row0_i, row0_i + n_i)`), so the per-lane
//! `sigmoid/alpha-gate + conv1d + QK-norm + GDN` sequence (4 launches per lane
//! per DeltaNet layer) collapses to one launch per stage per layer:
//!
//! * the two row-local kernels (`fused_sigmoid_alpha_gate_f32_batched`,
//!   `fused_qk_l2_norm_scale_interleave_f32_batched`) run once over all rows
//!   (existing launchers, unchanged);
//! * [`Gpu::conv1d_silu_split_f32_lanes`] — the per-lane conv rings ride in a
//!   by-value kernarg lane table (`kernels/src/conv1d_silu_split_lanes.hip`);
//! * `Gpu::gated_delta_net_q8_fast_lanes_varrows` over a device
//!   [`GdnLaneTable`] (one row per lane carrying its own token count and its
//!   frame prefix; uploaded by [`Gpu::upload_gdn_lane_table`] only when it
//!   changes). Lane `i` reserves frames `[F + P_i, F + P_i + n_i)` with
//!   `P_i = sum_{j<i} n_j`; the single-end fast kernel seeds its final requant
//!   with `F + P_i` (no per-token seed), and EF consumes no RNG but still
//!   reserves all `n_i` frames.
//!
//! Self-contained via `include_str!` (no shared-registry edits), launched
//! through `launch_maybe_blob` so the kernarg bytes are retained under capture.

use crate::dflash_gdn_replay::{GdnLayerTable, GdnLayerTableFrom};
use crate::dispatch::{Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult, KernargBlob};
use std::ffi::c_void;

/// Kernel source for [`Gpu::conv1d_silu_split_f32_lanes`].
pub const CONV1D_SILU_SPLIT_LANES_SRC: &str =
    include_str!("../../../kernels/src/conv1d_silu_split_lanes.hip");
/// Module key and device symbol.
pub const CONV1D_SILU_SPLIT_LANES_SYMBOL: &str = "conv1d_silu_split_f32_lanes";
/// Lanes per launch (`CONV_LANES_MAX` in the kernel); also the row padding of
/// the per-layer device GDN lane table.
pub const CONV_LANES_MAX: usize = 16;
/// GDN head dim the variable-row lane recurrence supports.
const GDN_LANE_HEAD_DIM: usize = 128;
/// GDN tile rows per block (the shipping `HIPFIRE_GDN_TILE_ROWS`).
const GDN_LANE_TILE_ROWS: usize = 4;

/// Variable-row lane recurrence source: the shipping `gated_delta_net_q8_fast`
/// defines (DPP reduction + prefetch, same as the replay ML kernel) plus the
/// layer-table entry in its `HIPFIRE_GDN_LANE_VARROWS` variant. The variant
/// lives in its own copy, `gated_delta_net_q8_fast_lanes_varrows.hip`: the
/// shipping `gated_delta_net_q8_fast.hip` is a P0 sha-pinned source (and the
/// rows-batched twins rewrite its text), so it must stay byte-identical. The
/// copy is that file verbatim plus the macro-isolated entry/table variant.
pub const GATED_DELTA_NET_Q8_FAST_LANES_VARROWS_SRC: &str = concat!(
    "#define HIPFIRE_GDN_DPP_REDUCE 1\n#define HIPFIRE_GDN_PREFETCH 1\n",
    "#define HIPFIRE_GDN_LAYER_TABLE 1\n#define HIPFIRE_GDN_LANE_VARROWS 1\n",
    "#define HIPFIRE_GDN_KERNEL gated_delta_net_q8_fast_lanes_varrows\n",
    include_str!("../../../kernels/src/gated_delta_net_q8_fast_lanes_varrows.hip")
);
/// Compiled-module key and device symbol of [`GATED_DELTA_NET_Q8_FAST_LANES_VARROWS_SRC`].
pub const GATED_DELTA_NET_Q8_FAST_LANES_VARROWS_SYMBOL: &str =
    "gated_delta_net_q8_fast_lanes_varrows";
/// [`GATED_DELTA_NET_Q8_FAST_LANES_VARROWS_SRC`] plus `HIPFIRE_GDN_STATE_SRC`:
/// reads S / scales / EF residual from the snapshot columns of
/// [`GdnLaneTableFrom`] and writes the live state.
pub const GATED_DELTA_NET_Q8_FAST_LANES_VARROWS_FROM_SRC: &str = concat!(
    "#define HIPFIRE_GDN_DPP_REDUCE 1\n#define HIPFIRE_GDN_PREFETCH 1\n",
    "#define HIPFIRE_GDN_LAYER_TABLE 1\n#define HIPFIRE_GDN_STATE_SRC 1\n",
    "#define HIPFIRE_GDN_LANE_VARROWS 1\n",
    "#define HIPFIRE_GDN_KERNEL gated_delta_net_q8_fast_lanes_varrows_from\n",
    include_str!("../../../kernels/src/gated_delta_net_q8_fast_lanes_varrows.hip")
);
/// Compiled-module key and device symbol of
/// [`GATED_DELTA_NET_Q8_FAST_LANES_VARROWS_FROM_SRC`].
pub const GATED_DELTA_NET_Q8_FAST_LANES_VARROWS_FROM_SYMBOL: &str =
    "gated_delta_net_q8_fast_lanes_varrows_from";

/// `#[repr(C)]` mirror of `struct ConvLaneTable` (the by-value kernel
/// argument): 16 x u64 state pointers, 16 x i32 row0, 16 x i32 n_tokens
/// (128 + 64 + 64 = 256 B, 8-aligned).
#[repr(C)]
struct ConvLaneTableArgs {
    state: [u64; CONV_LANES_MAX],
    row0: [i32; CONV_LANES_MAX],
    n_tokens: [i32; CONV_LANES_MAX],
}

/// One recurrence row per LANE of the variable-row lane GDN: the replay
/// [`GdnLayerTable`] pointer row plus that lane's own token count and its
/// frame offset. `#[repr(C)]` layout matches `GdnLayerTable` compiled with
/// `HIPFIRE_GDN_LANE_VARROWS` in `kernels/src/gated_delta_net_q8_fast.hip`
/// (72 B pointer row + 2 x u32 = 80 B, 8-aligned, no padding).
/// `frame_off` is RELATIVE: the sum of the preceding lanes' `n_tokens` within
/// this chunk (never the absolute frame counter), so identical shapes can hit
/// the host table cache.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnLaneTable {
    pub base: GdnLayerTable,
    pub n_tokens: u32,
    pub frame_off: u32,
}

/// [`GdnLaneTable`] over [`GdnLayerTableFrom`] (snapshot-source state
/// columns): 96 B + 2 x u32 = 104 B, 8-aligned, no padding.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnLaneTableFrom {
    pub base: GdnLayerTableFrom,
    pub n_tokens: u32,
    pub frame_off: u32,
}

const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(size_of::<GdnLayerTable>() == 72);
    assert!(size_of::<GdnLayerTableFrom>() == 96);
    assert!(size_of::<GdnLaneTable>() == 80);
    assert!(align_of::<GdnLaneTable>() == 8);
    assert!(offset_of!(GdnLaneTable, base) == 0);
    assert!(offset_of!(GdnLaneTable, n_tokens) == 72);
    assert!(offset_of!(GdnLaneTable, frame_off) == 76);
    assert!(size_of::<GdnLaneTableFrom>() == 104);
    assert!(align_of::<GdnLaneTableFrom>() == 8);
    assert!(offset_of!(GdnLaneTableFrom, base) == 0);
    assert!(offset_of!(GdnLaneTableFrom, n_tokens) == 96);
    assert!(offset_of!(GdnLaneTableFrom, frame_off) == 100);
};

impl Gpu {
    /// Conv1d + SiLU + Q/K/V split of `lanes.len()` (1..=16) lanes in ONE
    /// launch. `lanes[i] = (conv ring device pointer, first row, rows)`; the
    /// buffers are the FULL shared row-major scratch buffers. Byte-identical
    /// to one `conv1d_silu_split_f32_n` per lane over that lane's row view
    /// (see the kernel header).
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_silu_split_f32_lanes(
        &mut self,
        q_out: &GpuTensor,
        k_out: &GpuTensor,
        v_out: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        lanes: &[(u64, usize, usize)],
        k_dim: usize,
        v_dim: usize,
    ) -> HipResult<()> {
        assert!(
            (1..=CONV_LANES_MAX).contains(&lanes.len()),
            "conv1d_silu_split_f32_lanes: {} lanes outside 1..={CONV_LANES_MAX}",
            lanes.len()
        );
        self.bind_thread()?;
        self.ensure_kernel(
            CONV1D_SILU_SPLIT_LANES_SYMBOL,
            CONV1D_SILU_SPLIT_LANES_SRC,
            CONV1D_SILU_SPLIT_LANES_SYMBOL,
        )?;
        let mut args = ConvLaneTableArgs {
            state: [0; CONV_LANES_MAX],
            row0: [0; CONV_LANES_MAX],
            n_tokens: [0; CONV_LANES_MAX],
        };
        for (i, &(state, row0, n)) in lanes.iter().enumerate() {
            args.state[i] = state;
            args.row0[i] = row0 as i32;
            args.n_tokens[i] = n as i32;
        }
        let qp = q_out.buf.as_ptr();
        let kp = k_out.buf.as_ptr();
        let vp = v_out.buf.as_ptr();
        let ip = input.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let kd = k_dim as i32;
        let vd = v_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &mut args as *mut _ as *mut c_void,
            &kd as *const _ as *mut c_void,
            &vd as *const _ as *mut c_void,
        ];
        let n_channels = 2 * k_dim + v_dim;
        // Same block / channel grid as `conv1d_silu_split_f32_n`; grid.y = lane.
        let block = 256u32;
        let grid = ((n_channels as u32) + block - 1) / block;
        let blob_args = &args;
        self.launch_maybe_blob(
            CONV1D_SILU_SPLIT_LANES_SYMBOL,
            [grid, lanes.len() as u32, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                // The 256 B struct is 8-aligned: it starts at offset 40, right
                // after the five pointers.
                for v in &blob_args.state {
                    b.push_u64(*v);
                }
                for v in blob_args.row0.iter().chain(&blob_args.n_tokens) {
                    b.push_i32(*v);
                }
                b.push_i32(kd);
                b.push_i32(vd);
                b
            },
        )
    }

    /// JIT the variable-row lane GDN recurrence
    /// ([`GATED_DELTA_NET_Q8_FAST_LANES_VARROWS_SYMBOL`]); idempotent. Only
    /// this recurrence: no replay preamble.
    pub fn ensure_lane_gdn_varrows(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            GATED_DELTA_NET_Q8_FAST_LANES_VARROWS_SYMBOL,
            GATED_DELTA_NET_Q8_FAST_LANES_VARROWS_SRC,
            GATED_DELTA_NET_Q8_FAST_LANES_VARROWS_SYMBOL,
        )
    }

    /// JIT the snapshot-source twin
    /// ([`GATED_DELTA_NET_Q8_FAST_LANES_VARROWS_FROM_SYMBOL`]); idempotent.
    /// Run while arming the sourced scratch, outside graph capture.
    pub fn ensure_lane_gdn_varrows_from(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            GATED_DELTA_NET_Q8_FAST_LANES_VARROWS_FROM_SYMBOL,
            GATED_DELTA_NET_Q8_FAST_LANES_VARROWS_FROM_SRC,
            GATED_DELTA_NET_Q8_FAST_LANES_VARROWS_FROM_SYMBOL,
        )
    }

    /// GDN Q8 fast recurrence for `n_lanes` lanes with PER-LANE token counts
    /// in ONE launch. `table` is a device array of [`GdnLaneTable`] rows
    /// (row z = lane z: pointers, its own `n_tokens`, and `frame_off` = the
    /// prefix sum of the preceding lanes' token counts).
    ///
    /// Reserves `total_rows` (= sum of every lane's `n_tokens`) stochastic-
    /// rounding frames ONCE, so the host counter advances exactly as the
    /// per-lane sequential launches would. Lane z's final requant seeds from
    /// `frame_base + frame_off_z`; with error feedback no RNG is consumed but
    /// the same `total_rows` frames are still reserved.
    ///
    /// Caller contract: 1..=16 lanes; contiguous prefix metadata;
    /// `total_rows == sum(n_tokens)`; head_dim 128; disjoint valid lane
    /// pointers; exact gfx1201 fast body. No GPU readback validates this.
    pub fn gated_delta_net_q8_fast_lanes_varrows(
        &mut self,
        table: *const c_void,
        n_lanes: usize,
        total_rows: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.launch_gdn_q8_fast_lanes_varrows(
            GATED_DELTA_NET_Q8_FAST_LANES_VARROWS_SYMBOL,
            table,
            n_lanes,
            total_rows,
            n_heads,
            head_dim,
        )
    }

    /// [`Self::gated_delta_net_q8_fast_lanes_varrows`] over
    /// [`GdnLaneTableFrom`] rows (state read from the snapshot columns,
    /// written to the live buffers). Same frames, grid and arithmetic.
    pub fn gated_delta_net_q8_fast_lanes_varrows_from(
        &mut self,
        table: *const c_void,
        n_lanes: usize,
        total_rows: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.launch_gdn_q8_fast_lanes_varrows(
            GATED_DELTA_NET_Q8_FAST_LANES_VARROWS_FROM_SYMBOL,
            table,
            n_lanes,
            total_rows,
            n_heads,
            head_dim,
        )
    }

    /// Common launcher: bind, ensure the symbol's own module BEFORE
    /// reserving frames, reserve `total_rows` once, launch
    /// `[n_heads, 128/4, n_lanes]` x `[32, 1, 1]`. Kernarg order (offset):
    /// table ptr (0), i32 n_heads (8), i32 head_dim (12), u32 frame_base (16).
    fn launch_gdn_q8_fast_lanes_varrows(
        &mut self,
        symbol: &str,
        table: *const c_void,
        n_lanes: usize,
        total_rows: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if symbol == GATED_DELTA_NET_Q8_FAST_LANES_VARROWS_FROM_SYMBOL {
            self.ensure_lane_gdn_varrows_from()?;
        } else {
            self.ensure_lane_gdn_varrows()?;
        }
        let tp = table as *mut c_void;
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let fr = crate::norm::reserve_gdn_requant_frames(total_rows as u32);
        let mut params: [*mut c_void; 4] = [
            &tp as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &fr as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            symbol,
            [
                n_heads as u32,
                (GDN_LANE_HEAD_DIM / GDN_LANE_TILE_ROWS) as u32,
                n_lanes as u32,
            ],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(tp);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_u32(fr);
                b
            },
        )
    }

    /// May the lane-batched GDN verify route run for a chunk of DeltaNet
    /// layers with these head counts / dims on this device and process state?
    /// Exact gfx1201; the fast (single-end requant) GDN kernel; head dims 128;
    /// a GQA layout (`n_key_heads < n_v_heads`, exact division); no graph
    /// capture, replay recording or G0 observation (the multi-lane GDN kernel
    /// reads its pointers from device memory, which a recording could not
    /// attribute). The caller adds the per-chunk conditions (every lane
    /// carries a tape, ChainVerify fusion, per-lane row counts in range, ...).
    pub fn lane_gdn_batch_admitted(
        &self,
        n_v_heads: usize,
        n_key_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> bool {
        self.arch_caps.is_gfx1201()
            && !crate::norm::dn_requant_per_token()
            && key_head_dim == 128
            && value_head_dim == 128
            && n_key_heads > 0
            && n_key_heads < n_v_heads
            && n_v_heads % n_key_heads == 0
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
            && !self.replay.is_g0_observing()
    }

    /// Stage `bytes` into the device GDN lane table `dst` on the active
    /// stream (stream-ordered after every kernel already queued, so the
    /// previous step's recurrence launches finish reading the old rows
    /// first). The copy is followed by a stream synchronize so the host
    /// source may be dropped or rewritten immediately (a pageable-source
    /// async copy gives no earlier guarantee); callers upload only when the
    /// table contents change. Refused during graph capture.
    pub fn upload_gdn_lane_table(&mut self, dst: &GpuTensor, bytes: &[u8]) -> HipResult<()> {
        self.bind_thread()?;
        if self.graphs.capture_mode {
            return Err(HipError::new(
                0,
                "upload_gdn_lane_table: table staging during graph capture",
            ));
        }
        if bytes.len() > dst.buf.size() {
            return Err(HipError::new(
                0,
                &format!(
                    "upload_gdn_lane_table: table holds {} B, needs {} B",
                    dst.buf.size(),
                    bytes.len()
                ),
            ));
        }
        match self.active_stream.as_ref() {
            Some(stream) => {
                self.hip.memcpy_htod_async(&dst.buf, bytes, stream)?;
                self.hip.stream_synchronize(stream)
            }
            None => self.hip.memcpy_htod(&dst.buf, bytes),
        }
    }
}
