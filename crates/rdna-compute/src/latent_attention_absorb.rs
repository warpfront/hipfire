// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Absorbed-latent support (see latent_attention_tiled.rs for the shared contract).
//!
//! For head `h` the expansion weight `W [n_heads*(nope+v), latent]` has key rows
//! `Ak_h = W[h*(nope+v) + d]` (`d < nope`) and value rows
//! `Av_h = W[h*(nope+v) + nope + e]` (`e < v`). With the query nope part `qn_h`
//! and the shared tail `qt`:
//!
//! ```text
//! qabs_h[c] = sum_d Ak_h[d][c] * qn_h[d]          (c < latent)      absorb_q
//! Q'_h      = [qabs_h || qt]                      (latent + tail)  -> tiled attention
//! y_h[e]    = sum_c Av_h[e][c] * z_h[c]           (e < v)          absorb_v
//! ```
//!
//! so that `Q'_h . [c || kpe] == qn_h . (Ak_h c) + qt . kpe` and the attention
//! kernels can run on the latent cache directly (`z_h = sum_t p_t c_t`).
//!
//! Operands: weights are BF16 (RNE from the caller's F32 column plane, same
//! routine as the serial reference), activations are F32 planes, accumulation is
//! FP32 FMA in a fixed order, no atomics, no allocation, no host readback; every
//! launch goes through `bind_thread` + `launch_maybe_blob` with an owned kernarg
//! blob and grid dims that depend only on the call's geometry.
//!
//! Packed weight layout (private to this module; built by
//! [`Gpu::latent_attention_absorb_pack`], read by absorb_q / absorb_v): head `h`
//! occupies elements `[h*latent*(nope+v), (h+1)*latent*(nope+v))` holding
//! `AkT_h [nope][latent]` (`AkT_h[d][c] = W[h*(nope+v)+d][c]`) followed by
//! `AvT_h [latent][v]` (`AvT_h[c][e] = W[h*(nope+v)+nope+e][c]`). Both halves are
//! "reduction index major, output index contiguous", the single operand shape of
//! the gemm kernel. Total size: [`latent_attention_absorb_packed_elems`].
//!
//! Kernel: `kernels/src/latent_attention_absorb.hip` (see its header for the
//! tiling). Summation order: with `rows > 8` each output is one ascending-`k`
//! FMA chain; with `rows <= 8` (decode) the `k` range is split in up to 8
//! contiguous segments, each summed ascending from zero, then added in
//! ascending segment order. The configuration is a pure function of the launch
//! geometry, so results are byte-deterministic per geometry, and the two
//! configurations differ only at the F32 summation-order level.
//!
//! Limits (explicit errors otherwise): `nope`, `v`, `latent` nonzero multiples
//! of 8; `n_heads` in `1..=65535`; `rows` in `1..=65535*8*rgn` (grid y limit);
//! packed weight base pointer 16-byte aligned; every extent fits `i32`.

use std::ffi::c_void;

use hip_bridge::{HipError, HipResult, KernargBlob};

use crate::dispatch::{DType, Gpu, GpuTensor};

pub const LATENT_ATTENTION_ABSORB_SRC: &str =
    include_str!("../../../kernels/src/latent_attention_absorb.hip");

const MODULE: &str = "latent_attention_absorb";
const PACK_KERNEL: &str = "latent_attention_absorb_pack_bf16";
const GEMM_KERNEL: &str = "latent_attention_absorb_gemm";

/// Threads per block of the pack kernel.
const PACK_BLOCK: u32 = 256;
/// Rows per thread (fixed by the kernel, `LAB_RPT`).
const ROWS_PER_THREAD: usize = 8;
/// Largest row count served by the split-`k` decode configuration.
pub const LATENT_ATTENTION_ABSORB_DECODE_ROWS: usize = 8;
/// Row groups per block of the tiled configuration (32-row tiles).
const TILE_ROW_GROUPS: usize = 4;
/// Largest number of `k` segments of the decode configuration.
const MAX_KSPLIT: usize = 8;
/// LDS `k` chunk (elements) of the tiled / decode configurations.
const TILE_KCHUNK: usize = 128;
const DECODE_KCHUNK: usize = 64;
/// Grid y/z hardware limit.
const MAX_GRID_YZ: usize = 65535;

fn err(msg: impl AsRef<str>) -> HipError {
    HipError::new(0, msg.as_ref())
}

fn i32_of(v: usize, what: &str) -> HipResult<i32> {
    i32::try_from(v).map_err(|_| err(format!("latent_attention_absorb: {what} exceeds i32")))
}

fn need_dtype(t: &GpuTensor, d: DType, what: &str) -> HipResult<()> {
    if t.dtype == d {
        Ok(())
    } else {
        Err(err(format!(
            "latent_attention_absorb: {what} must be {d:?}, got {:?}",
            t.dtype
        )))
    }
}

fn need_numel(t: &GpuTensor, min: usize, what: &str) -> HipResult<()> {
    if t.numel() >= min {
        Ok(())
    } else {
        Err(err(format!(
            "latent_attention_absorb: {what} holds {} elements, needs {min}",
            t.numel()
        )))
    }
}

fn mul(a: usize, b: usize, what: &str) -> HipResult<usize> {
    a.checked_mul(b)
        .ok_or_else(|| err(format!("latent_attention_absorb: {what} overflows usize")))
}

fn need_multiple_of_8(v: usize, what: &str) -> HipResult<()> {
    if v != 0 && v % 8 == 0 {
        Ok(())
    } else {
        Err(err(format!(
            "latent_attention_absorb: {what} must be a nonzero multiple of 8, got {v}"
        )))
    }
}

/// Geometry shared by pack / q / v: validated extents and the packed size.
fn check_geometry(n_heads: usize, nope: usize, v: usize, latent: usize) -> HipResult<usize> {
    if n_heads == 0 || n_heads > MAX_GRID_YZ {
        return Err(err(format!(
            "latent_attention_absorb: n_heads must be in 1..={MAX_GRID_YZ}, got {n_heads}"
        )));
    }
    need_multiple_of_8(nope, "nope")?;
    need_multiple_of_8(v, "v")?;
    need_multiple_of_8(latent, "latent")?;
    let per_head = mul(latent, nope + v, "latent * (nope + v)")?;
    let total = mul(n_heads, per_head, "packed element count")?;
    i32_of(total, "packed element count")?;
    Ok(total)
}

fn need_aligned16(p: *mut c_void, what: &str) -> HipResult<()> {
    if (p as usize) % 16 == 0 {
        Ok(())
    } else {
        Err(err(format!(
            "latent_attention_absorb: {what} base pointer must be 16-byte aligned"
        )))
    }
}

/// Launch configuration of the gemm kernel (pure function of the geometry).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GemmPlan {
    nl: usize,
    rgn: usize,
    ks: usize,
    kl: usize,
    dc: usize,
    grid: [u32; 3],
    block: u32,
    shared: u32,
}

fn plan_gemm(rows: usize, n_heads: usize, k: usize, n: usize) -> HipResult<GemmPlan> {
    let (nl, rgn, ks, kchunk) = if rows <= LATENT_ATTENTION_ABSORB_DECODE_ROWS {
        // Decode: narrow n tile (more blocks), reduction split across ks segments.
        let mut ks = 1usize;
        while ks * 2 <= MAX_KSPLIT && ks * 2 * 16 <= k {
            ks *= 2;
        }
        (8usize, 1usize, ks, DECODE_KCHUNK)
    } else {
        let lanes = n.div_ceil(8).next_power_of_two().min(32);
        let rgn = rows.div_ceil(ROWS_PER_THREAD).min(TILE_ROW_GROUPS);
        (lanes, rgn, 1usize, TILE_KCHUNK)
    };
    let kl = k.div_ceil(ks);
    let dc = kl.min(kchunk).max(1);
    let tr = rgn * ROWS_PER_THREAD;
    let nt = nl * 8;
    let gx = n.div_ceil(nt);
    let gy = rows.div_ceil(tr);
    if gy > MAX_GRID_YZ {
        return Err(err(format!(
            "latent_attention_absorb: {rows} rows need grid y {gy} > {MAX_GRID_YZ}"
        )));
    }
    let block = nl * rgn * ks;
    let q_floats = ks * dc * tr;
    let red_floats = if ks > 1 { ks * tr * nt } else { 0 };
    let shared = q_floats.max(red_floats) * 4;
    Ok(GemmPlan {
        nl,
        rgn,
        ks,
        kl,
        dc,
        grid: [gx as u32, gy as u32, n_heads as u32],
        block: block as u32,
        shared: shared as u32,
    })
}

/// BF16 element count of the packed expansion weight for this geometry.
pub fn latent_attention_absorb_packed_elems(
    n_heads: usize,
    nope: usize,
    v: usize,
    latent: usize,
) -> usize {
    n_heads * latent * (nope + v)
}

impl Gpu {
    /// Convert the F32 column plane `w_t [latent, n_heads*(nope+v)]` (row `c`
    /// holds column `c` of `W`) to the private packed BF16 layout (RNE).
    /// Intended to run once at load/admission.
    pub fn latent_attention_absorb_pack(
        &mut self,
        w_t: &GpuTensor,
        packed: &GpuTensor,
        n_heads: usize,
        nope: usize,
        v: usize,
        latent: usize,
    ) -> HipResult<()> {
        let total = check_geometry(n_heads, nope, v, latent)?;
        need_dtype(w_t, DType::F32, "w_t plane")?;
        need_numel(w_t, total, "w_t plane")?;
        need_dtype(packed, DType::BF16, "packed weight")?;
        need_numel(packed, total, "packed weight")?;
        let blocks = total.div_ceil(PACK_BLOCK as usize);
        let blocks = u32::try_from(blocks)
            .ok()
            .filter(|b| *b <= i32::MAX as u32)
            .ok_or_else(|| err("latent_attention_absorb: pack grid too large"))?;
        let (nh, nopei, vi, lat) = (
            i32_of(n_heads, "n_heads")?,
            i32_of(nope, "nope")?,
            i32_of(v, "v")?,
            i32_of(latent, "latent")?,
        );
        self.bind_thread()?;
        self.ensure_kernel(MODULE, LATENT_ATTENTION_ABSORB_SRC, PACK_KERNEL)?;
        let (wp, pp) = (w_t.buf.as_ptr(), packed.buf.as_ptr());
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &nopei as *const _ as *mut c_void,
            &vi as *const _ as *mut c_void,
            &lat as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            PACK_KERNEL,
            [blocks, 1, 1],
            [PACK_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(wp);
                b.push_ptr(pp);
                b.push_i32(nh);
                b.push_i32(nopei);
                b.push_i32(vi);
                b.push_i32(lat);
                b
            },
        )
    }

    /// `q_abs[r][h] = [ sum_d Ak_h[d][:] q[r][h][d]  ||  q[r][h][nope..nope+tail] ]`.
    ///
    /// `q` is F32 `[rows, n_heads, nope + tail]`, `q_abs` F32
    /// `[rows, n_heads, latent + tail]`; `packed` is the tensor filled by
    /// [`Self::latent_attention_absorb_pack`] for the same `(n_heads, nope, v,
    /// latent)`.
    #[allow(clippy::too_many_arguments)]
    pub fn latent_attention_absorb_q(
        &mut self,
        q: &GpuTensor,
        packed: &GpuTensor,
        q_abs: &GpuTensor,
        rows: usize,
        n_heads: usize,
        nope: usize,
        tail: usize,
        v: usize,
        latent: usize,
    ) -> HipResult<()> {
        let total = check_geometry(n_heads, nope, v, latent)?;
        need_dtype(q, DType::F32, "q plane")?;
        need_dtype(packed, DType::BF16, "packed weight")?;
        need_dtype(q_abs, DType::F32, "q_abs plane")?;
        need_numel(packed, total, "packed weight")?;
        let x_hs = nope + tail;
        let o_hs = latent + tail;
        need_numel(q, mul(mul(rows, n_heads, "q")?, x_hs, "q")?, "q plane")?;
        need_numel(
            q_abs,
            mul(mul(rows, n_heads, "q_abs")?, o_hs, "q_abs")?,
            "q_abs plane",
        )?;
        if rows == 0 {
            return Ok(());
        }
        self.absorb_gemm(
            q.buf.as_ptr(),
            packed.buf.as_ptr(),
            q_abs.buf.as_ptr(),
            rows,
            n_heads,
            nope,
            latent,
            x_hs,
            o_hs,
            latent * (nope + v),
            false,
            tail,
        )
    }

    /// `y[r][h][e] = sum_c Av_h[e][c] z[r][h][c]`, optionally rounded once to BF16.
    ///
    /// `z` is F32 `[rows, n_heads, latent]`, `y` F32 `[rows, n_heads, v]`.
    #[allow(clippy::too_many_arguments)]
    pub fn latent_attention_absorb_v(
        &mut self,
        z: &GpuTensor,
        packed: &GpuTensor,
        y: &GpuTensor,
        rows: usize,
        n_heads: usize,
        nope: usize,
        v: usize,
        latent: usize,
        round_y_bf16: bool,
    ) -> HipResult<()> {
        let total = check_geometry(n_heads, nope, v, latent)?;
        need_dtype(z, DType::F32, "z plane")?;
        need_dtype(packed, DType::BF16, "packed weight")?;
        need_dtype(y, DType::F32, "y plane")?;
        need_numel(packed, total, "packed weight")?;
        need_numel(z, mul(mul(rows, n_heads, "z")?, latent, "z")?, "z plane")?;
        need_numel(y, mul(mul(rows, n_heads, "y")?, v, "y")?, "y plane")?;
        if rows == 0 {
            return Ok(());
        }
        // AvT_h starts `nope * latent` elements into each head block (16-byte
        // aligned because latent is a multiple of 8).
        let w_ptr =
            unsafe { (packed.buf.as_ptr() as *mut u8).add(nope * latent * 2) } as *mut c_void;
        self.absorb_gemm(
            z.buf.as_ptr(),
            w_ptr,
            y.buf.as_ptr(),
            rows,
            n_heads,
            latent,
            v,
            latent,
            v,
            latent * (nope + v),
            round_y_bf16,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn absorb_gemm(
        &mut self,
        x_ptr: *mut c_void,
        w_ptr: *mut c_void,
        out_ptr: *mut c_void,
        rows: usize,
        n_heads: usize,
        k: usize,
        n: usize,
        x_hs: usize,
        o_hs: usize,
        w_hs: usize,
        round_bf16: bool,
        tail: usize,
    ) -> HipResult<()> {
        need_aligned16(w_ptr, "packed weight")?;
        let plan = plan_gemm(rows, n_heads, k, n)?;
        let (rows_i, nh_i, k_i, n_i, xhs_i, ohs_i) = (
            i32_of(rows, "rows")?,
            i32_of(n_heads, "n_heads")?,
            i32_of(k, "reduction width")?,
            i32_of(n, "output width")?,
            i32_of(x_hs, "input head stride")?,
            i32_of(o_hs, "output head stride")?,
        );
        i32_of(
            mul(mul(rows, n_heads, "input")?, x_hs.max(o_hs), "plane")?,
            "plane element count",
        )?;
        let w_hs_u = w_hs as u64;
        let (nl_i, ks_i, kl_i, dc_i, tail_i) = (
            plan.nl as i32,
            plan.ks as i32,
            plan.kl as i32,
            plan.dc as i32,
            i32_of(tail, "tail")?,
        );
        let ovec = i32::from((out_ptr as usize) % 16 == 0 && o_hs % 4 == 0);
        let round_i = i32::from(round_bf16);
        self.bind_thread()?;
        self.ensure_kernel(MODULE, LATENT_ATTENTION_ABSORB_SRC, GEMM_KERNEL)?;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &w_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &rows_i as *const _ as *mut c_void,
            &nh_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
            &n_i as *const _ as *mut c_void,
            &xhs_i as *const _ as *mut c_void,
            &ohs_i as *const _ as *mut c_void,
            &w_hs_u as *const _ as *mut c_void,
            &nl_i as *const _ as *mut c_void,
            &ks_i as *const _ as *mut c_void,
            &kl_i as *const _ as *mut c_void,
            &dc_i as *const _ as *mut c_void,
            &ovec as *const _ as *mut c_void,
            &round_i as *const _ as *mut c_void,
            &tail_i as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            GEMM_KERNEL,
            plan.grid,
            [plan.block, 1, 1],
            plan.shared,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(x_ptr);
                b.push_ptr(w_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(rows_i);
                b.push_i32(nh_i);
                b.push_i32(k_i);
                b.push_i32(n_i);
                b.push_i32(xhs_i);
                b.push_i32(ohs_i);
                b.push_u64(w_hs_u);
                b.push_i32(nl_i);
                b.push_i32(ks_i);
                b.push_i32(kl_i);
                b.push_i32(dc_i);
                b.push_i32(ovec);
                b.push_i32(round_i);
                b.push_i32(tail_i);
                b
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_decode_and_tile() {
        // Raon-like absorbed q: K = nope 128, N = latent 512.
        let d = plan_gemm(1, 40, 128, 512).unwrap();
        assert_eq!((d.nl, d.rgn, d.ks, d.kl, d.dc), (8, 1, 8, 16, 16));
        assert_eq!(d.grid, [8, 1, 40]);
        assert_eq!(d.block, 64);
        // absorbed v decode: K = latent 512, N = v 128.
        let v = plan_gemm(1, 40, 512, 128).unwrap();
        assert_eq!((v.nl, v.ks, v.kl, v.dc), (8, 8, 64, 64));
        assert_eq!(v.grid, [2, 1, 40]);
        // prefill tile.
        let t = plan_gemm(4096, 40, 128, 512).unwrap();
        assert_eq!((t.nl, t.rgn, t.ks, t.kl, t.dc), (32, 4, 1, 128, 128));
        assert_eq!(t.grid, [2, 128, 40]);
        assert_eq!(t.block, 128);
        assert_eq!(t.shared, 128 * 32 * 4);
        // v prefill: N = 128 -> 16 lanes.
        let tv = plan_gemm(64, 40, 512, 128).unwrap();
        assert_eq!((tv.nl, tv.rgn, tv.grid), (16, 4, [1, 2, 40]));
    }

    #[test]
    fn geometry_rejects_misaligned() {
        assert!(check_geometry(40, 128, 128, 512).is_ok());
        assert!(check_geometry(40, 126, 128, 512).is_err());
        assert!(check_geometry(40, 128, 120 + 1, 512).is_err());
        assert!(check_geometry(0, 128, 128, 512).is_err());
        assert!(check_geometry(40, 128, 128, 0).is_err());
    }
}
