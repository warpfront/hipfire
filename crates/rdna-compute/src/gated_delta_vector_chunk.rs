// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Chunked (C64) generic gated delta rule with per-key-channel diagonal log decay
//! (`HIPFIRE_GATED_DELTA_VECTOR_CHUNKED`, developer, default ON on gfx1201; `=0` kills).
//!
//! [`Gpu::gated_delta_net_vector_chunked_f32`] takes the same [`GatedDeltaVectorLaunch`] as the
//! serial [`Gpu::gated_delta_net_vector_serial_f32`] and produces the same outputs and final state
//! (reassociated, not bit-identical): `y_t = S_t q_t` read after the write, state `[H,V,K]` FP32
//! value-major, per-key-channel (or per-head) log decay `a_t <= 0`, optional BF16 output rounding.
//! The algebra and numerics are documented at the top of
//! `kernels/src/gated_delta_vector_chunk.gfx1201.hip`: a log-space cumulative decay reset per C64
//! chunk, every exponent a causal relative one (`<= 0`, the diagonal 16-row sub-blocks are bounded
//! by `exp(8 * 10)` through a per-token clamp of `a_t` to `[-10, 0]`), a chunk-parallel prepare
//! kernel (pair matrices, `(I + diag(beta) G)^-1`, `W'`, `Q'`, `U'`, `Kend`) and a one-wave-per
//! (head, 16 value rows) scan that carries the FP32 state through registers.
//!
//! No atomics, no per-launch allocation (the workspace is a caller-owned
//! [`GatedDeltaVectorChunkScratch`]), deterministic bytes for a fixed shape. Rows beyond the
//! workspace capacity run as consecutive capacity-sized segments chained through the state tensor.
//!
//! Admission (everything else keeps the serial kernels): exact `gfx1201`, `head_dim == 128`, host
//! `slot` (no `slot_idx`/`n_dev` device indirection, so no graph-captured launch), and at least
//! [`GATED_DELTA_VECTOR_CHUNK_MIN_ROWS`] rows (ordinary prefill only; decode and verify rows stay
//! on the PM serial kernel).

use crate::dispatch::{Gpu, GpuTensor};
use crate::gated_delta_vector::{
    i32_of, need, refuse, GatedDeltaVectorLaunch, GATED_DELTA_VECTOR_HEAD_DIM,
};
use crate::profile;
use crate::DType;
use hip_bridge::{HipResult, KernargBlob};
use std::ffi::c_void;
use std::sync::LazyLock;

pub const GATED_DELTA_VECTOR_CHUNK_SRC: &str =
    include_str!("../../../kernels/src/gated_delta_vector_chunk.gfx1201.hip");
const MODULE: &str = "gated_delta_vector_chunk_gfx1201";
const PREP_SYMBOL: &str = "gated_delta_vector_chunk_prep_gfx12";
const SCAN_SYMBOL: &str = "gated_delta_vector_chunk_scan_gfx12";

/// Tokens per chunk.
pub const GATED_DELTA_VECTOR_CHUNK_ROWS: usize = 64;
/// Smallest row count the chunked route is admitted for.
// selector-receipt: benchmarks/selector-receipts/gated_delta_vector_chunk_min_rows.json
pub const GATED_DELTA_VECTOR_CHUNK_MIN_ROWS: usize = 64;
/// Scratch bytes per (chunk, head): `-W'` hi and lo, `Q'`, `Kend^T` hi, `Kend^T` lo (16 KiB f16 each),
/// `U'^T` (32 KiB f32), `M` (8 KiB f16) and `exp(ell_T)` (512 B f32); see the kernel header.
pub const GATED_DELTA_VECTOR_CHUNK_BLOCK_BYTES: usize = 123392;
/// Value rows per scan wave (the scan grid is `n_heads * 8` waves).
const SCAN_VTILES: u32 = 8;
/// Resident prepare workgroups per multiprocessor (LDS 56.6 KiB, 256 threads: two co-reside on a WGP;
/// the earlier 320-thread block did not, measured).
const PREP_WGS_PER_MP: u32 = 2;

static FLAG: LazyLock<bool> =
    LazyLock::new(|| hipfire_config::developer_bool("HIPFIRE_GATED_DELTA_VECTOR_CHUNKED", true));

/// `HIPFIRE_GATED_DELTA_VECTOR_CHUNKED` (default on; `0` kills).
pub fn chunked_enabled() -> bool {
    *FLAG
}

/// Why a launch keeps the serial kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatedDeltaVectorChunkRefusal {
    FlagOff,
    Arch,
    HeadDim,
    DeviceIndirection,
    TooFewRows,
}

impl std::fmt::Display for GatedDeltaVectorChunkRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::FlagOff => "HIPFIRE_GATED_DELTA_VECTOR_CHUNKED is off",
            Self::Arch => "no chunked RDA kernel for this architecture (exact gfx1201 only)",
            Self::HeadDim => "chunked RDA is specialized for head_dim 128",
            Self::DeviceIndirection => {
                "chunked RDA needs a host slot and token count (no slot_idx/n_dev, no graph capture)"
            }
            Self::TooFewRows => "chunked RDA is admitted for prefill rows >= 64 only",
        })
    }
}

/// Pure admission of the chunked route (the flag, the exact arch, the launch shape).
pub fn chunked_admission(
    flag: bool,
    arch: &str,
    head_dim: usize,
    n_tokens: usize,
    device_indirection: bool,
) -> Result<(), GatedDeltaVectorChunkRefusal> {
    if !flag {
        return Err(GatedDeltaVectorChunkRefusal::FlagOff);
    }
    chunked_supported(arch, head_dim, n_tokens, device_indirection)
}

/// [`chunked_admission`] without the flag: whether the kernel itself supports the launch.
pub fn chunked_supported(
    arch: &str,
    head_dim: usize,
    n_tokens: usize,
    device_indirection: bool,
) -> Result<(), GatedDeltaVectorChunkRefusal> {
    if arch != "gfx1201" {
        return Err(GatedDeltaVectorChunkRefusal::Arch);
    }
    if head_dim != GATED_DELTA_VECTOR_HEAD_DIM {
        return Err(GatedDeltaVectorChunkRefusal::HeadDim);
    }
    if device_indirection {
        return Err(GatedDeltaVectorChunkRefusal::DeviceIndirection);
    }
    if n_tokens < GATED_DELTA_VECTOR_CHUNK_MIN_ROWS {
        return Err(GatedDeltaVectorChunkRefusal::TooFewRows);
    }
    Ok(())
}

/// Preallocated C64 workspace of the chunked RDA route, sized from (max rows, heads) with
/// `head_dim == 128`. Owned by the caller, never grown or reallocated by a launch.
pub struct GatedDeltaVectorChunkScratch {
    buf: GpuTensor,
    max_rows: usize,
    n_heads: usize,
}

impl GatedDeltaVectorChunkScratch {
    /// Device bytes [`Gpu::alloc_gated_delta_vector_chunk_scratch`] allocates (the pool's 256 B
    /// minimum aside): `ceil(max_rows / 64) * n_heads * 123392`.
    pub fn required_bytes(max_rows: usize, n_heads: usize) -> usize {
        max_rows.div_ceil(GATED_DELTA_VECTOR_CHUNK_ROWS)
            * n_heads
            * GATED_DELTA_VECTOR_CHUNK_BLOCK_BYTES
    }

    /// Rows one segment covers (a whole number of chunks); a longer launch runs as several
    /// segments chained through the state tensor.
    pub fn segment_rows(&self) -> usize {
        (self.max_rows / GATED_DELTA_VECTOR_CHUNK_ROWS).max(1) * GATED_DELTA_VECTOR_CHUNK_ROWS
    }

    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    pub fn bytes(&self) -> usize {
        self.buf.buf.size()
    }
}

impl Gpu {
    /// Allocate the workspace for launches of up to `max_rows` rows (per segment) and `n_heads`
    /// heads.
    pub fn alloc_gated_delta_vector_chunk_scratch(
        &mut self,
        max_rows: usize,
        n_heads: usize,
    ) -> HipResult<GatedDeltaVectorChunkScratch> {
        if max_rows == 0 || n_heads == 0 {
            return Err(refuse(
                "alloc_gated_delta_vector_chunk_scratch: max_rows and n_heads must be nonzero"
                    .into(),
            ));
        }
        let bytes = GatedDeltaVectorChunkScratch::required_bytes(max_rows, n_heads);
        let buf = self.alloc_tensor(&[bytes / 4], DType::F32)?;
        Ok(GatedDeltaVectorChunkScratch {
            buf,
            max_rows,
            n_heads,
        })
    }

    /// Return the workspace to the pool.
    pub fn free_gated_delta_vector_chunk_scratch(
        &mut self,
        scratch: GatedDeltaVectorChunkScratch,
    ) -> HipResult<()> {
        self.free_tensor(scratch.buf)
    }

    /// Whether this device has the chunked kernels (exact gfx1201).
    pub fn gated_delta_vector_chunked_arch_supported(&self) -> bool {
        self.arch == "gfx1201"
    }

    /// Chunked equivalent of [`Gpu::gated_delta_net_vector_serial_f32`] for ordinary prefill:
    /// same operands, same outputs and final state to the reassociation tolerance. Refuses
    /// (never silently falls back) a launch outside the admission in the module docs.
    pub fn gated_delta_net_vector_chunked_f32(
        &mut self,
        p: &GatedDeltaVectorLaunch<'_>,
        scratch: &GatedDeltaVectorChunkScratch,
    ) -> HipResult<()> {
        self.gated_delta_net_vector_chunked_stages(p, scratch, true, true)
    }

    /// [`Self::gated_delta_net_vector_chunked_f32`] with the two kernels selectable
    /// (`prep`, `scan`): node benchmarking only (a scan without its prep reads stale scratch).
    #[doc(hidden)]
    pub fn gated_delta_net_vector_chunked_stages(
        &mut self,
        p: &GatedDeltaVectorLaunch<'_>,
        scratch: &GatedDeltaVectorChunkScratch,
        prep: bool,
        scan: bool,
    ) -> HipResult<()> {
        self.gated_delta_net_vector_chunked_stages_prof(p, scratch, prep, scan, None)
    }

    /// [`Self::gated_delta_net_vector_chunked_stages`] with optional debug timestamps: `prof` is
    /// `[chunks * n_heads, 16]` f32 words (8 u64 stamps per chunk-head, 10 ns ticks), written by the
    /// prepare kernel; only meaningful for a single-segment launch.
    #[doc(hidden)]
    pub fn gated_delta_net_vector_chunked_stages_prof(
        &mut self,
        p: &GatedDeltaVectorLaunch<'_>,
        scratch: &GatedDeltaVectorChunkScratch,
        prep: bool,
        scan: bool,
        prof: Option<&GpuTensor>,
    ) -> HipResult<()> {
        const NAME: &str = "gated_delta_net_vector_chunked_f32";
        let d = GATED_DELTA_VECTOR_HEAD_DIM;
        if let Err(r) = chunked_supported(
            &self.arch,
            p.head_dim,
            p.n_tokens,
            p.slot_idx.is_some() || p.n_dev.is_some(),
        ) {
            return Err(refuse(format!("{NAME}: {r}")));
        }
        if p.n_heads == 0 || p.qk_group == 0 || p.n_heads % p.qk_group != 0 {
            return Err(refuse(format!(
                "{NAME}: n_heads {} not divisible by qk_group {}",
                p.n_heads, p.qk_group
            )));
        }
        if p.n_heads > scratch.n_heads {
            return Err(refuse(format!(
                "{NAME}: scratch holds {} heads, launch has {}",
                scratch.n_heads, p.n_heads
            )));
        }
        if p.state_row_stride < d || p.state_head_stride < d * p.state_row_stride {
            return Err(refuse(format!("{NAME}: state strides overlap rows/heads")));
        }
        if p.decay_per_key && p.decay_row_stride % 4 != 0 {
            return Err(refuse(format!(
                "{NAME}: per-key log decay row stride must be a multiple of 4 f32 (16-byte loads)"
            )));
        }
        if p.q_row_stride % 4 != 0 || p.k_row_stride % 4 != 0 {
            return Err(refuse(format!(
                "{NAME}: q/k row strides must be multiples of 4 f32 (16-byte loads)"
            )));
        }
        if p.state_row_stride % 4 != 0
            || p.state_head_stride % 4 != 0
            || p.state_slot_stride % 4 != 0
        {
            return Err(refuse(format!(
                "{NAME}: state strides must be multiples of 4 f32 (16-byte state tiles)"
            )));
        }
        let n = p.n_tokens;
        let hq = p.n_heads / p.qk_group;
        need(p.query, "query", n, p.q_row_stride, hq * d)?;
        need(p.key, "key", n, p.k_row_stride, hq * d)?;
        need(p.value, "value", n, p.v_row_stride, p.n_heads * d)?;
        let decay_extent = if p.decay_per_key {
            p.n_heads * d
        } else {
            p.n_heads
        };
        need(
            p.log_decay,
            "log_decay",
            n,
            p.decay_row_stride,
            decay_extent,
        )?;
        need(p.beta, "beta", n, p.beta_row_stride, p.n_heads)?;
        need(p.output, "output", n, p.out_row_stride, p.n_heads * d)?;
        need(
            p.state,
            "state",
            p.slot + 1,
            p.state_slot_stride,
            (p.n_heads - 1) * p.state_head_stride + (d - 1) * p.state_row_stride + d,
        )?;

        self.bind_thread()?;
        self.ensure_kernel(MODULE, GATED_DELTA_VECTOR_CHUNK_SRC, PREP_SYMBOL)?;
        self.ensure_kernel(MODULE, GATED_DELTA_VECTOR_CHUNK_SRC, SCAN_SYMBOL)?;

        let seg_cap = scratch.segment_rows();
        let n_heads = i32_of(p.n_heads, "n_heads")?;
        let qk_group = i32_of(p.qk_group, "qk_group")?;
        let per_key = p.decay_per_key as i32;
        let round = p.round_out_bf16 as i32;
        let qrs = i32_of(p.q_row_stride, "q_row_stride")?;
        let krs = i32_of(p.k_row_stride, "k_row_stride")?;
        let vrs = i32_of(p.v_row_stride, "v_row_stride")?;
        let drs = i32_of(p.decay_row_stride, "decay_row_stride")?;
        let brs = i32_of(p.beta_row_stride, "beta_row_stride")?;
        let ors = i32_of(p.out_row_stride, "out_row_stride")?;
        let shs = i32_of(p.state_head_stride, "state_head_stride")?;
        let srs = i32_of(p.state_row_stride, "state_row_stride")?;

        let at = |t: &GpuTensor, elems: usize| -> *const c_void {
            // SAFETY: `need()` proved the tensor holds every element a segment touches, and the
            // offset (in f32 elements) stays inside that extent.
            unsafe { (t.buf.as_ptr() as *const u8).add(elems * 4) as *const c_void }
        };
        let scr = scratch.buf.buf.as_ptr() as *const c_void;
        let state = at(p.state, p.slot * p.state_slot_stride);
        // Persistent prepare grid: PREP_WGS_PER_MP resident workgroups per multiprocessor (WGP on
        // RDNA); the kernel strides over the chunk-heads.
        let mp = self
            .hip
            .get_device_attribute(
                crate::profiler::HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                0,
            )
            .ok()
            .filter(|&v| v > 0)
            .unwrap_or(32) as u32;
        let prep_grid = mp * PREP_WGS_PER_MP;
        let prof_ptr: *const c_void =
            prof.map_or(std::ptr::null(), |t| t.buf.as_ptr() as *const c_void);

        let mut row0 = 0usize;
        while row0 < n {
            let seg = (n - row0).min(seg_cap);
            let seg_i = i32_of(seg, "segment rows")?;
            let qp = at(p.query, row0 * p.q_row_stride);
            let kp = at(p.key, row0 * p.k_row_stride);
            let vp = at(p.value, row0 * p.v_row_stride);
            let lp = at(p.log_decay, row0 * p.decay_row_stride);
            let bp = at(p.beta, row0 * p.beta_row_stride);
            let op = at(p.output, row0 * p.out_row_stride);
            let nchunks = seg.div_ceil(GATED_DELTA_VECTOR_CHUNK_ROWS);

            if prep {
                let bytes = seg * p.n_heads * d * 4 * 4
                    + nchunks * p.n_heads * GATED_DELTA_VECTOR_CHUNK_BLOCK_BYTES;
                let timer = profile::begin_timer(&self.hip, "gated_delta", PREP_SYMBOL, bytes);
                let mut params: Vec<*mut c_void> = vec![
                    &qp as *const _ as *mut c_void,
                    &kp as *const _ as *mut c_void,
                    &vp as *const _ as *mut c_void,
                    &lp as *const _ as *mut c_void,
                    &bp as *const _ as *mut c_void,
                    &scr as *const _ as *mut c_void,
                    &seg_i as *const _ as *mut c_void,
                    &n_heads as *const _ as *mut c_void,
                    &qk_group as *const _ as *mut c_void,
                    &per_key as *const _ as *mut c_void,
                    &qrs as *const _ as *mut c_void,
                    &krs as *const _ as *mut c_void,
                    &vrs as *const _ as *mut c_void,
                    &drs as *const _ as *mut c_void,
                    &brs as *const _ as *mut c_void,
                    &prof_ptr as *const _ as *mut c_void,
                ];
                let r = self.launch_maybe_blob(
                    PREP_SYMBOL,
                    [prep_grid.min((nchunks * p.n_heads) as u32).max(1), 1, 1],
                    [256, 1, 1],
                    0,
                    &mut params,
                    || {
                        let mut b = KernargBlob::new();
                        b.push_ptr(qp);
                        b.push_ptr(kp);
                        b.push_ptr(vp);
                        b.push_ptr(lp);
                        b.push_ptr(bp);
                        b.push_ptr(scr);
                        b.push_i32(seg_i);
                        b.push_i32(n_heads);
                        b.push_i32(qk_group);
                        b.push_i32(per_key);
                        b.push_i32(qrs);
                        b.push_i32(krs);
                        b.push_i32(vrs);
                        b.push_i32(drs);
                        b.push_i32(brs);
                        b.push_ptr(prof_ptr);
                        b
                    },
                );
                if let Some(t) = timer {
                    t.finish(&self.hip);
                }
                r?;
            }

            if scan {
                let bytes = nchunks * p.n_heads * GATED_DELTA_VECTOR_CHUNK_BLOCK_BYTES
                    + seg * p.n_heads * d * 4
                    + p.n_heads * d * d * 4 * 2;
                let timer = profile::begin_timer(&self.hip, "gated_delta", SCAN_SYMBOL, bytes);
                let mut params: Vec<*mut c_void> = vec![
                    &scr as *const _ as *mut c_void,
                    &state as *const _ as *mut c_void,
                    &op as *const _ as *mut c_void,
                    &seg_i as *const _ as *mut c_void,
                    &n_heads as *const _ as *mut c_void,
                    &round as *const _ as *mut c_void,
                    &ors as *const _ as *mut c_void,
                    &shs as *const _ as *mut c_void,
                    &srs as *const _ as *mut c_void,
                ];
                let r = self.launch_maybe_blob(
                    SCAN_SYMBOL,
                    [p.n_heads as u32 * SCAN_VTILES, 1, 1],
                    [32, 1, 1],
                    0,
                    &mut params,
                    || {
                        let mut b = KernargBlob::new();
                        b.push_ptr(scr);
                        b.push_ptr(state);
                        b.push_ptr(op);
                        b.push_i32(seg_i);
                        b.push_i32(n_heads);
                        b.push_i32(round);
                        b.push_i32(ors);
                        b.push_i32(shs);
                        b.push_i32(srs);
                        b
                    },
                );
                if let Some(t) = timer {
                    t.finish(&self.hip);
                }
                r?;
            }
            row0 += seg;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_is_gfx1201_head128_host_slot_prefill_rows() {
        let ok = |arch, hd, n, ind| chunked_admission(true, arch, hd, n, ind);
        assert_eq!(ok("gfx1201", 128, 64, false), Ok(()));
        assert_eq!(ok("gfx1201", 128, 4096, false), Ok(()));
        assert_eq!(
            ok("gfx1201", 128, 63, false),
            Err(GatedDeltaVectorChunkRefusal::TooFewRows)
        );
        assert_eq!(
            ok("gfx1201", 128, 1, false),
            Err(GatedDeltaVectorChunkRefusal::TooFewRows)
        );
        assert_eq!(
            ok("gfx1201", 64, 4096, false),
            Err(GatedDeltaVectorChunkRefusal::HeadDim)
        );
        assert_eq!(
            ok("gfx1100", 128, 4096, false),
            Err(GatedDeltaVectorChunkRefusal::Arch)
        );
        assert_eq!(
            ok("gfx1151", 128, 4096, false),
            Err(GatedDeltaVectorChunkRefusal::Arch)
        );
        assert_eq!(
            ok("gfx1201", 128, 4096, true),
            Err(GatedDeltaVectorChunkRefusal::DeviceIndirection)
        );
        assert_eq!(
            chunked_admission(false, "gfx1201", 128, 4096, false),
            Err(GatedDeltaVectorChunkRefusal::FlagOff)
        );
    }

    #[test]
    fn scratch_bytes_follow_chunks_heads_and_block() {
        assert_eq!(
            GatedDeltaVectorChunkScratch::required_bytes(4096, 40),
            64 * 40 * GATED_DELTA_VECTOR_CHUNK_BLOCK_BYTES
        );
        assert_eq!(
            GatedDeltaVectorChunkScratch::required_bytes(100, 40),
            2 * 40 * GATED_DELTA_VECTOR_CHUNK_BLOCK_BYTES
        );
        assert_eq!(GATED_DELTA_VECTOR_CHUNK_BLOCK_BYTES % 256, 0);
    }
}
