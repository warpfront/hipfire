// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Railgun E0 / L6c: `Gpu` launchers for the multi-layer DFlash/MTP GDN
//! replay (gfx1201).
//!
//! `GdnTape::replay_gdn` advances every LinearAttention layer's conv ring and
//! recurrent state by `n_steps` tokens from the verify tape. Layers are
//! independent, so the per-layer launch sequence (conv1d + QK norm +
//! interleave + GDN: 4 launches x 48 layers on gfx1201) collapses to two
//! launches over a device pointer table with one row per layer:
//!
//! 1. [`Gpu::dflash_gdn_replay_pre_ml`] — the `dflash_gdn_pre_replay_gfx1100`
//!    preamble body per (block, layer), grid.y = layer.
//! 2. [`Gpu::gated_delta_net_q8_fast_ml`] — `gated_delta_net_q8_fast` built
//!    with the shipping defines plus `HIPFIRE_GDN_LAYER_TABLE`, grid.z =
//!    layer, frames reserved exactly as the per-layer launches would.
//!
//! Railgun D8 adds `_from` twins of both kernels (`DFLASH_GDN_REPLAY_STATE_SRC`
//! / `HIPFIRE_GDN_STATE_SRC`): the same bodies with the recurrent-state loads
//! redirected to a second table column, so the DFlash rollback replays from
//! the snapshot buffers into the live ones instead of restoring first.
//!
//! Both kernels read pointers from device memory, so a Redline tape could
//! not attribute their effects; the launchers therefore decline (return
//! `Ok(false)`) while a Redline recording is open. HipGraph capture is fine
//! (`launch_maybe_blob`). The pointer tables are owned by the caller.

use crate::dispatch::Gpu;
use hip_bridge::{DeviceBuffer, HipError, HipResult, KernargBlob};
use std::ffi::c_void;

/// Preamble kernel source.
pub const DFLASH_GDN_REPLAY_PRE_ML_SRC: &str =
    include_str!("../../../kernels/src/dflash_gdn_replay_pre_ml.hip");
/// Compiled-module key and device symbol for the preamble kernel.
pub const DFLASH_GDN_REPLAY_PRE_ML_SYMBOL: &str = "dflash_gdn_replay_pre_ml";
/// Recurrence kernel source: the shipping `gated_delta_net_q8_fast` defines
/// (`GATED_DELTA_NET_Q8_FAST_SRC`) plus the layer-table entry.
pub const GATED_DELTA_NET_Q8_FAST_ML_SRC: &str = concat!(
    "#define HIPFIRE_GDN_DPP_REDUCE 1\n#define HIPFIRE_GDN_PREFETCH 1\n",
    "#define HIPFIRE_GDN_LAYER_TABLE 1\n#define HIPFIRE_GDN_KERNEL gated_delta_net_q8_fast_ml\n",
    include_str!("../../../kernels/src/gated_delta_net_q8_fast.hip")
);
/// Compiled-module key and device symbol for the recurrence kernel.
pub const GATED_DELTA_NET_Q8_FAST_ML_SYMBOL: &str = "gated_delta_net_q8_fast_ml";

/// Railgun D8: the preamble built with `DFLASH_GDN_REPLAY_STATE_SRC` — its
/// first step reads the conv ring from the snapshot row
/// ([`DflashReplayPreLayerFrom::conv_state_src`]) and every step writes the
/// live ring, so a rollback needs no restore copy of the conv state.
pub const DFLASH_GDN_REPLAY_PRE_ML_FROM_SRC: &str = concat!(
    "#define DFLASH_GDN_REPLAY_STATE_SRC 1\n",
    "#define DFLASH_GDN_REPLAY_PRE_KERNEL dflash_gdn_replay_pre_ml_from\n",
    include_str!("../../../kernels/src/dflash_gdn_replay_pre_ml.hip")
);
/// Compiled-module key and device symbol of [`DFLASH_GDN_REPLAY_PRE_ML_FROM_SRC`].
pub const DFLASH_GDN_REPLAY_PRE_ML_FROM_SYMBOL: &str = "dflash_gdn_replay_pre_ml_from";
/// Railgun D8: the recurrence of [`GATED_DELTA_NET_Q8_FAST_ML_SRC`] plus
/// `HIPFIRE_GDN_STATE_SRC` — it reads S / scales / EF residual from the
/// snapshot row ([`GdnLayerTableFrom`]) and writes the live state.
pub const GATED_DELTA_NET_Q8_FAST_ML_FROM_SRC: &str = concat!(
    "#define HIPFIRE_GDN_DPP_REDUCE 1\n#define HIPFIRE_GDN_PREFETCH 1\n",
    "#define HIPFIRE_GDN_LAYER_TABLE 1\n#define HIPFIRE_GDN_STATE_SRC 1\n",
    "#define HIPFIRE_GDN_KERNEL gated_delta_net_q8_fast_ml_from\n",
    include_str!("../../../kernels/src/gated_delta_net_q8_fast.hip")
);
/// Compiled-module key and device symbol of [`GATED_DELTA_NET_Q8_FAST_ML_FROM_SRC`].
pub const GATED_DELTA_NET_Q8_FAST_ML_FROM_SYMBOL: &str = "gated_delta_net_q8_fast_ml_from";

/// Preamble threads per block (matches `GDN_PRE_BLOCK`).
pub const DFLASH_GDN_REPLAY_PRE_BLOCK: u32 = 256;
/// Only head_dim == 128 is supported (matches `GDN_PRE_HD` / the GDN `HD`).
pub const DFLASH_GDN_REPLAY_HEAD_DIM: usize = 128;
/// Replay step ceiling (matches `GDN_PRE_MAXN` LDS staging).
pub const DFLASH_GDN_REPLAY_MAX_STEPS: usize = 16;
/// GDN tile rows per block (the shipping `HIPFIRE_GDN_TILE_ROWS`).
const GDN_TILE_ROWS: u32 = 4;

/// One preamble row per LinearAttention layer. `#[repr(C)]` layout matches
/// `DflashReplayPreLayer` in `kernels/src/dflash_gdn_replay_pre_ml.hip`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DflashReplayPreLayer {
    pub qkv_tape: u64,
    pub conv_w: u64,
    pub conv_state: u64,
    pub v_out: u64,
    pub q_dst: u64,
    pub k_dst: u64,
}

/// One recurrence row per LinearAttention layer. `#[repr(C)]` layout matches
/// `GdnLayerTable` in `kernels/src/gated_delta_net_q8_fast.hip`. `ef` is 0
/// when the layer has no error-feedback residual.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnLayerTable {
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub gate: u64,
    pub beta: u64,
    pub s_q8: u64,
    pub s_scales: u64,
    pub output: u64,
    pub ef: u64,
}

/// Railgun D8 preamble row: [`DflashReplayPreLayer`] (whose `conv_state` is
/// the live ring, written) plus the snapshot ring the first step reads.
/// `#[repr(C)]` layout matches the `DFLASH_GDN_REPLAY_STATE_SRC` struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DflashReplayPreLayerFrom {
    pub base: DflashReplayPreLayer,
    pub conv_state_src: u64,
}

/// Railgun D8 recurrence row: [`GdnLayerTable`] (whose `s_q8` / `s_scales` /
/// `ef` are the live state, written) plus the snapshot state it reads.
/// `#[repr(C)]` layout matches the `HIPFIRE_GDN_STATE_SRC` struct. `ef_src` is
/// 0 exactly when `base.ef` is.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnLayerTableFrom {
    pub base: GdnLayerTable,
    pub s_q8_src: u64,
    pub s_scales_src: u64,
    pub ef_src: u64,
}

/// Byte view of a padding-free `#[repr(C)]` table of plain integers for one
/// `memcpy_htod`: the all-`u64` replay rows above and the lane rows
/// (`GdnLaneTable` / `GdnLaneTableFrom` in `lane_gdn.rs`: those rows plus two
/// trailing `u32`, 80 B / 104 B, 8-aligned, no padding bytes).
pub fn table_bytes<T: Copy>(rows: &[T]) -> &[u8] {
    // SAFETY: callers pass the padding-free repr(C) row types above.
    unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, std::mem::size_of_val(rows)) }
}

/// Shape of one multi-layer replay scratch: four F32 regions (q, k, v and the
/// dead recurrence output) of `n_layers * rows * v_dim` elements each. It is
/// the key of `Gpu::dflash_gdn_replay_scratch`: every armed `GdnTape` with the
/// same shape shares one buffer; different shapes (another model, a different
/// `rows` cap) get their own entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GdnReplayScratchShape {
    pub n_layers: usize,
    pub rows: usize,
    pub v_dim: usize,
}

impl GdnReplayScratchShape {
    /// Bytes of one region.
    pub fn region_bytes(&self) -> usize {
        self.n_layers * self.rows * self.v_dim * std::mem::size_of::<f32>()
    }
}

/// Railgun E0: one entry of `Gpu::dflash_gdn_replay_scratch`, the
/// multi-layer replay's q/k/v/out scratch shared by every armed `GdnTape` of
/// its [`GdnReplayScratchShape`] on this `Gpu` (each tape arms at
/// construction, so a per-tape copy would cost every lane ~75 MB on the 27B
/// whether or not it ever replays). Drained by `Gpu::invalidate_weight_caches`
/// (model unload), like `fp16_shadow_cache`. Sharing is sound because of
/// three invariants, each held per entry:
///
/// 1. **One stream at a time.** Every replay launches on `Gpu::active_stream`:
///    `launch_replay_pre_ml` / `launch_gdn_q8_fast_ml` (this file, `:438`,
///    `:543`) go through `Gpu::launch_maybe_blob` (`dispatch.rs:2898`), which
///    launches on `self.active_stream`. The DFlash, MTP and VMM
///    continuous-batching paths create that stream once per `Gpu` and never
///    swap it (`hipfire-arch-qwen35/src/dflash_cb.rs:235`, `mtp_cb.rs:100`,
///    `speculative.rs:4120`); the only swaps are the per-rank devices of
///    `hipfire-runtime/src/multi_gpu.rs:3300`/`:3359`, each a separate `Gpu`
///    with its own scratch, and the MTP snapshot-overlap side stream
///    (`mtp_spec.rs:518`) only copies snapshots. Belt and braces:
///    [`Gpu::dflash_gdn_replay_scratch_order`] runs before every replay and
///    replay-graph launch and, when the stream differs from the entry's
///    previous replay, drains the device first (refused inside a capture).
/// 2. **Pointer lifetime.** The address is baked into each tape's tables and,
///    through them, into any `HIPFIRE_REPLAY_GRAPH` capture
///    (`graph.rs:487` `begin_replay_graph_capture`, from
///    `hipfire-arch-qwen35/src/speculative.rs:2260` `GdnTape::replay_gdn`).
///    An entry is released only when no armed tape holds it (`refs == 0`)
///    AND `GraphState::replay_graph_count() == 0` (`graph.rs:550`), and its
///    buffer never moves while it exists. Unload drains every entry; a tape
///    that outlives it fails loudly at its next replay
///    ([`Gpu::dflash_gdn_replay_scratch_order`] checks the entry still holds
///    the tape's base) instead of writing freed memory; unload also destroys
///    every replay graph (`Gpu::invalidate_graph_state`). Redline / railgun
///    PM4 recordings never contain the route: [`Gpu::gdn_replay_ml_eligible`]
///    (`:231`) requires `!replay.is_recording()` (`:240`).
/// 3. **No cross-replay reads.** Within one replay the preamble writes q/k/v
///    for every (layer, step < n_steps)
///    (`kernels/src/dflash_gdn_replay_pre_ml.hip:154`, `:155`, `:171`) before
///    the recurrence reads exactly those rows, and the recurrence's `out` is
///    never read; nothing outside the two launches touches the buffer. So no
///    replay observes another tape's (or its own earlier) contents.
pub(crate) struct GdnReplaySharedScratch {
    buf: DeviceBuffer,
    refs: usize,
    /// Raw handle of the stream of the last ordered replay (null = default).
    last_stream: Option<usize>,
}

impl Gpu {
    /// Process-static part of [`Self::gdn_replay_ml_eligible`]: exact
    /// gfx1201, the `HIPFIRE_GDN_REPLAY_ML_OFF` opt-out, the fast
    /// (single-end requant) GDN kernel, head_dim 128 and GQA divisibility.
    /// A replay tape arms its multi-layer tables at construction iff this
    /// holds, so the replay route never depends on later memory pressure.
    pub fn gdn_replay_ml_supported(
        &self,
        n_v_heads: usize,
        n_key_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> bool {
        self.arch_caps.is_gfx1201()
            && !self.flags.gdn_replay_ml_off
            && !crate::norm::dn_requant_per_token()
            && key_head_dim == DFLASH_GDN_REPLAY_HEAD_DIM
            && value_head_dim == DFLASH_GDN_REPLAY_HEAD_DIM
            && n_key_heads > 0
            && n_v_heads % n_key_heads == 0
    }

    /// Host-side eligibility shared by both launchers:
    /// [`Self::gdn_replay_ml_supported`], no open Redline recording, and
    /// 1 <= n_steps <= 16.
    pub fn gdn_replay_ml_eligible(
        &self,
        n_v_heads: usize,
        n_key_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        n_steps: usize,
    ) -> bool {
        self.gdn_replay_ml_supported(n_v_heads, n_key_heads, key_head_dim, value_head_dim)
            && !self.replay.is_recording()
            && (1..=DFLASH_GDN_REPLAY_MAX_STEPS).contains(&n_steps)
    }

    /// JIT both kernels (idempotent). Called when the tape arms its tables,
    /// before any capture.
    pub fn ensure_dflash_gdn_replay_ml(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            DFLASH_GDN_REPLAY_PRE_ML_SYMBOL,
            DFLASH_GDN_REPLAY_PRE_ML_SRC,
            DFLASH_GDN_REPLAY_PRE_ML_SYMBOL,
        )?;
        self.ensure_kernel(
            GATED_DELTA_NET_Q8_FAST_ML_SYMBOL,
            GATED_DELTA_NET_Q8_FAST_ML_SRC,
            GATED_DELTA_NET_Q8_FAST_ML_SYMBOL,
        )
    }

    /// Railgun D8: JIT the snapshot-source twins of both kernels (idempotent).
    pub fn ensure_dflash_gdn_replay_ml_from(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            DFLASH_GDN_REPLAY_PRE_ML_FROM_SYMBOL,
            DFLASH_GDN_REPLAY_PRE_ML_FROM_SRC,
            DFLASH_GDN_REPLAY_PRE_ML_FROM_SYMBOL,
        )?;
        self.ensure_kernel(
            GATED_DELTA_NET_Q8_FAST_ML_FROM_SYMBOL,
            GATED_DELTA_NET_Q8_FAST_ML_FROM_SRC,
            GATED_DELTA_NET_Q8_FAST_ML_FROM_SYMBOL,
        )
    }

    /// Acquire the shared multi-layer replay scratch of `shape`
    /// ([`GdnReplaySharedScratch`]) for one armed tape and return its base:
    /// the existing entry (one more holder) or a new one. Allocation failure
    /// is an error, so the tape is refused at construction.
    pub fn dflash_gdn_replay_scratch_acquire(
        &mut self,
        shape: GdnReplayScratchShape,
    ) -> HipResult<*mut c_void> {
        self.bind_thread()?;
        if let Some(s) = self.dflash_gdn_replay_scratch.get_mut(&shape) {
            s.refs += 1;
            return Ok(s.buf.as_ptr());
        }
        let buf = self.pool.alloc(&self.hip, 4 * shape.region_bytes())?;
        let ptr = buf.as_ptr();
        self.dflash_gdn_replay_scratch.insert(
            shape,
            GdnReplaySharedScratch {
                buf,
                refs: 1,
                last_stream: None,
            },
        );
        Ok(ptr)
    }

    /// Drop one holder of the `shape` entry acquired at `base`. The entry
    /// returns to the pool once no tape holds it and no replay graph can
    /// reference it; otherwise it stays at its address until unload. A
    /// holder whose entry was already drained (unload) is a no-op.
    pub fn dflash_gdn_replay_scratch_release(
        &mut self,
        shape: GdnReplayScratchShape,
        base: *mut c_void,
    ) {
        // bind_thread: skip — bookkeeping; the buffer goes to the pool, no device call.
        let Some(s) = self.dflash_gdn_replay_scratch.get_mut(&shape) else {
            return;
        };
        if s.buf.as_ptr() != base {
            return;
        }
        s.refs = s.refs.saturating_sub(1);
        if s.refs == 0 && self.graphs.replay_graph_count() == 0 {
            if let Some(old) = self.dflash_gdn_replay_scratch.remove(&shape) {
                self.pool.free(old.buf);
            }
        }
    }

    /// Check and stream-order the `shape` entry before a replay that uses
    /// it at `base` (invariants 1 and 2 of [`GdnReplaySharedScratch`]): the
    /// entry must still hold `base` (a tape that outlived an unload fails
    /// here, before any launch), and a replay on a different stream than the
    /// entry's previous one first drains the device, which is refused while
    /// a graph capture is open.
    pub fn dflash_gdn_replay_scratch_order(
        &mut self,
        shape: GdnReplayScratchShape,
        base: *mut c_void,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let stream = self
            .active_stream
            .as_ref()
            .map_or(0, |s| s.as_raw() as usize);
        let capturing = self.graphs.capture_mode;
        let last = match self.dflash_gdn_replay_scratch.get(&shape) {
            Some(s) if s.buf.as_ptr() == base => s.last_stream,
            _ => {
                return Err(HipError::new(
                    0,
                    "shared GDN replay scratch was drained under a live tape (tape outlived unload)",
                ));
            }
        };
        match last {
            Some(last) if last == stream => return Ok(()),
            Some(_) if capturing => {
                return Err(HipError::new(
                    0,
                    "shared GDN replay scratch: stream changed inside a graph capture",
                ));
            }
            Some(_) => self.hip.device_synchronize()?,
            None => {}
        }
        if let Some(s) = self.dflash_gdn_replay_scratch.get_mut(&shape) {
            s.last_stream = Some(stream);
        }
        Ok(())
    }

    /// Return every shared replay scratch entry to the pool (model unload,
    /// via `invalidate_weight_caches`). Tapes still holding one fail loudly
    /// at their next replay.
    pub(crate) fn drain_dflash_gdn_replay_scratch(&mut self) {
        let entries: Vec<GdnReplaySharedScratch> = self
            .dflash_gdn_replay_scratch
            .drain()
            .map(|(_, s)| s)
            .collect();
        for s in entries {
            self.pool.free(s.buf);
        }
    }

    /// Conv1d + QK norm + interleave for `n_layers` table rows in one launch.
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_gdn_replay_pre_ml(
        &mut self,
        table: *const c_void,
        n_layers: usize,
        n_v_heads: usize,
        n_key_heads: usize,
        k_dim: usize,
        v_dim: usize,
        qkv_dim: usize,
        n_steps: usize,
        q_scale: f32,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_dflash_gdn_replay_ml()?;
        self.launch_replay_pre_ml(
            DFLASH_GDN_REPLAY_PRE_ML_SYMBOL,
            table,
            n_layers,
            n_v_heads,
            n_key_heads,
            k_dim,
            v_dim,
            qkv_dim,
            n_steps,
            q_scale,
            eps,
        )
    }

    /// Railgun D8: [`Self::dflash_gdn_replay_pre_ml`] over
    /// [`DflashReplayPreLayerFrom`] rows (conv ring read from the snapshot).
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_gdn_replay_pre_ml_from(
        &mut self,
        table: *const c_void,
        n_layers: usize,
        n_v_heads: usize,
        n_key_heads: usize,
        k_dim: usize,
        v_dim: usize,
        qkv_dim: usize,
        n_steps: usize,
        q_scale: f32,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_dflash_gdn_replay_ml_from()?;
        self.launch_replay_pre_ml(
            DFLASH_GDN_REPLAY_PRE_ML_FROM_SYMBOL,
            table,
            n_layers,
            n_v_heads,
            n_key_heads,
            k_dim,
            v_dim,
            qkv_dim,
            n_steps,
            q_scale,
            eps,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_replay_pre_ml(
        &mut self,
        symbol: &str,
        table: *const c_void,
        n_layers: usize,
        n_v_heads: usize,
        n_key_heads: usize,
        k_dim: usize,
        v_dim: usize,
        qkv_dim: usize,
        n_steps: usize,
        q_scale: f32,
        eps: f32,
    ) -> HipResult<()> {
        let tp = table as *mut c_void;
        let nvh = n_v_heads as i32;
        let nkh = n_key_heads as i32;
        let ratio = (n_v_heads / n_key_heads) as i32;
        let kd = k_dim as i32;
        let vd = v_dim as i32;
        let qd = qkv_dim as i32;
        let ns = n_steps as i32;
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &nvh as *const _ as *mut c_void,
            &nkh as *const _ as *mut c_void,
            &ratio as *const _ as *mut c_void,
            &kd as *const _ as *mut c_void,
            &vd as *const _ as *mut c_void,
            &qd as *const _ as *mut c_void,
            &ns as *const _ as *mut c_void,
            &q_scale as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
        ];
        let v_blocks = v_dim.div_ceil(DFLASH_GDN_REPLAY_PRE_BLOCK as usize);
        self.launch_maybe_blob(
            symbol,
            [(n_key_heads + v_blocks) as u32, n_layers as u32, 1],
            [DFLASH_GDN_REPLAY_PRE_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(tp);
                b.push_i32(nvh);
                b.push_i32(nkh);
                b.push_i32(ratio);
                b.push_i32(kd);
                b.push_i32(vd);
                b.push_i32(qd);
                b.push_i32(ns);
                b.push_f32(q_scale);
                b.push_f32(eps);
                b
            },
        )
    }

    /// GDN Q8 recurrence (fast, EF-capable) for `n_layers` table rows in one
    /// launch. Reserves `n_layers * n_steps` stochastic-rounding frames —
    /// the same total, in the same per-layer order, as `n_layers` sequential
    /// `gated_delta_net_q8_batch_seq` calls.
    pub fn gated_delta_net_q8_fast_ml(
        &mut self,
        table: *const c_void,
        n_layers: usize,
        n_steps: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_dflash_gdn_replay_ml()?;
        self.launch_gdn_q8_fast_ml(
            GATED_DELTA_NET_Q8_FAST_ML_SYMBOL,
            table,
            n_layers,
            n_steps,
            n_heads,
            head_dim,
        )
    }

    /// Railgun D8: [`Self::gated_delta_net_q8_fast_ml`] over
    /// [`GdnLayerTableFrom`] rows (state read from the snapshot, written to
    /// the live buffers). Same frames, grid and arithmetic.
    pub fn gated_delta_net_q8_fast_ml_from(
        &mut self,
        table: *const c_void,
        n_layers: usize,
        n_steps: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_dflash_gdn_replay_ml_from()?;
        self.launch_gdn_q8_fast_ml(
            GATED_DELTA_NET_Q8_FAST_ML_FROM_SYMBOL,
            table,
            n_layers,
            n_steps,
            n_heads,
            head_dim,
        )
    }

    fn launch_gdn_q8_fast_ml(
        &mut self,
        symbol: &str,
        table: *const c_void,
        n_layers: usize,
        n_steps: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        let tp = table as *mut c_void;
        let nt = n_steps as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let fr = crate::norm::reserve_gdn_requant_frames((n_layers * n_steps) as u32);
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &nt as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &fr as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            symbol,
            [
                n_heads as u32,
                DFLASH_GDN_REPLAY_HEAD_DIM as u32 / GDN_TILE_ROWS,
                n_layers as u32,
            ],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(tp);
                b.push_i32(nt);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_u32(fr);
                b
            },
        )
    }
}
