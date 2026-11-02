// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generic latent attention: split latent/tail, unequal QK/V, causal offsets.
//!
//! Geometry (all independent, none Raon-specific): `latent` cached channels
//! that are RMS-normalized, `tail` extra channels cached verbatim and shared by
//! every head, `n_heads` heads with `nope` per-head key channels expanded from
//! the latent through a `[(nope+v) * n_heads, latent]` projection and `v`
//! per-head value channels. Query/key width is `nope + tail`, value width `v`,
//! score scale is supplied by the caller. No positional rotation is applied.
//!
//! Storage contracts (frozen for callers):
//! * latent cache: BF16, row-major `[cache_rows, latent + tail]`, one row per
//!   token, normalized latent followed by the raw tail.
//! * expanded mirrors (BF16, token-major, so they grow by appending rows):
//!   `knope[t][h][nope]` and `vmir[t][h][v]`; `mirror_cap` is the number of
//!   rows the caller has backed (a bound, not a stride). The tail is not
//!   duplicated per head; attention reads it from the latent cache.
//! * activations (`kv_a`, `q`, `y`, `g`, …) are F32 planes holding
//!   BF16-rounded values; kernels apply the BF16 rounding at the same points a
//!   BF16 reference does.
//! * positions/lengths live on the device (a 4-byte i32 scalar holding the
//!   number of rows already cached), never read back on the host, so every
//!   launcher is graph-capture safe. Launchers never mutate that scalar.
//!
//! Kernels: `kernels/src/latent_attention_{split,decompress,serial,gate}.hip`.

use std::ffi::c_void;

use hip_bridge::{HipError, HipResult, KernargBlob};

use crate::dispatch::{DType, Gpu, GpuTensor};

pub const LATENT_ATTENTION_SPLIT_SRC: &str =
    include_str!("../../../kernels/src/latent_attention_split.hip");
pub const LATENT_ATTENTION_DECOMPRESS_SRC: &str =
    include_str!("../../../kernels/src/latent_attention_decompress.hip");
pub const LATENT_ATTENTION_SERIAL_SRC: &str =
    include_str!("../../../kernels/src/latent_attention_serial.hip");
pub const LATENT_ATTENTION_GATE_SRC: &str =
    include_str!("../../../kernels/src/latent_attention_gate.hip");

const SPLIT_MODULE: &str = "latent_attention_split";
const DECOMPRESS_MODULE: &str = "latent_attention_decompress";
const SERIAL_MODULE: &str = "latent_attention_serial";
const GATE_MODULE: &str = "latent_attention_gate";

/// Threads per block of the split / row-norm kernels (fixed by the kernels).
const SPLIT_BLOCK: u32 = 256;
/// Keys per tile of the serial attention kernel (fixed by the kernel).
const SERIAL_KEY_TILE: usize = 128;
/// Largest value width the serial kernel supports (`LA_KT * LA_DVQ`).
pub const LATENT_ATTENTION_SERIAL_MAX_V: usize = 512;
const ELEM_BLOCK: u32 = 256;

fn err(msg: impl AsRef<str>) -> HipError {
    HipError::new(0, msg.as_ref())
}

fn i32_of(v: usize, what: &str) -> HipResult<i32> {
    i32::try_from(v).map_err(|_| err(format!("latent_attention: {what} exceeds i32")))
}

fn need_dtype(t: &GpuTensor, d: DType, what: &str) -> HipResult<()> {
    if t.dtype == d {
        Ok(())
    } else {
        Err(err(format!(
            "latent_attention: {what} must be {d:?}, got {:?}",
            t.dtype
        )))
    }
}

fn need_numel(t: &GpuTensor, min: usize, what: &str) -> HipResult<()> {
    if t.numel() >= min {
        Ok(())
    } else {
        Err(err(format!(
            "latent_attention: {what} holds {} elements, needs {min}",
            t.numel()
        )))
    }
}

fn need_scalar_i32(t: &GpuTensor, what: &str) -> HipResult<()> {
    if t.byte_size() >= 4 {
        Ok(())
    } else {
        Err(err(format!(
            "latent_attention: {what} must hold a 4-byte device scalar"
        )))
    }
}

/// Gamma may be F32 or BF16; returns the kernel flag.
fn gamma_flag(g: &GpuTensor, n: usize, what: &str) -> HipResult<i32> {
    need_numel(g, n, what)?;
    match g.dtype {
        DType::F32 => Ok(0),
        DType::BF16 => Ok(1),
        d => Err(err(format!(
            "latent_attention: {what} must be F32 or BF16, got {d:?}"
        ))),
    }
}

fn product(a: usize, b: usize, what: &str) -> HipResult<usize> {
    a.checked_mul(b)
        .ok_or_else(|| err(format!("latent_attention: {what} overflows usize")))
}

impl Gpu {
    /// Split a projected `[n_rows, latent + tail]` row into the latent cache.
    ///
    /// RMS over the `latent` channels only (`eps`, `gamma[latent]` plain, no
    /// `1+gamma`), BF16 round, tail copied verbatim (BF16), written to cache
    /// row `*pos + i`. Rows landing outside `[0, cache_rows)` are skipped.
    #[allow(clippy::too_many_arguments)]
    pub fn latent_attention_split_kv(
        &mut self,
        kv_a: &GpuTensor,
        gamma: &GpuTensor,
        cache: &GpuTensor,
        pos: &GpuTensor,
        n_rows: usize,
        latent: usize,
        tail: usize,
        eps: f32,
        cache_rows: usize,
    ) -> HipResult<()> {
        if n_rows == 0 {
            return Ok(());
        }
        if latent == 0 {
            return Err(err("latent_attention: latent width must be nonzero"));
        }
        let width = latent + tail;
        need_dtype(kv_a, DType::F32, "kv_a plane")?;
        need_numel(kv_a, product(n_rows, width, "kv_a")?, "kv_a plane")?;
        need_dtype(cache, DType::BF16, "latent cache")?;
        need_numel(cache, product(cache_rows, width, "cache")?, "latent cache")?;
        need_scalar_i32(pos, "position")?;
        let gamma_bf16 = gamma_flag(gamma, latent, "latent gamma")?;
        let (nr, lat, tl, cr) = (
            i32_of(n_rows, "rows")?,
            i32_of(latent, "latent")?,
            i32_of(tail, "tail")?,
            i32_of(cache_rows, "cache rows")?,
        );
        self.bind_thread()?;
        self.ensure_kernel(
            SPLIT_MODULE,
            LATENT_ATTENTION_SPLIT_SRC,
            "latent_attention_split_kv_bf16",
        )?;
        let (kp, gp, cp, pp) = (
            kv_a.buf.as_ptr(),
            gamma.buf.as_ptr(),
            cache.buf.as_ptr(),
            pos.buf.as_ptr(),
        );
        let mut params: Vec<*mut c_void> = vec![
            &kp as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
            &lat as *const _ as *mut c_void,
            &tl as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
            &gamma_bf16 as *const _ as *mut c_void,
            &cr as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "latent_attention_split_kv_bf16",
            [n_rows as u32, 1, 1],
            [SPLIT_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(kp);
                b.push_ptr(gp);
                b.push_ptr(cp);
                b.push_ptr(pp);
                b.push_i32(nr);
                b.push_i32(lat);
                b.push_i32(tl);
                b.push_f32(eps);
                b.push_i32(gamma_bf16);
                b.push_i32(cr);
                b
            },
        )
    }

    /// Row RMSNorm over the full `width` (`eps`, plain `gamma`), result rounded
    /// to BF16 and kept as an F32 plane. `out` may be the same tensor as `x`.
    pub fn latent_attention_rms_rows(
        &mut self,
        x: &GpuTensor,
        gamma: &GpuTensor,
        out: &GpuTensor,
        n_rows: usize,
        width: usize,
        eps: f32,
    ) -> HipResult<()> {
        if n_rows == 0 {
            return Ok(());
        }
        if width == 0 {
            return Err(err("latent_attention: row width must be nonzero"));
        }
        let total = product(n_rows, width, "rms rows")?;
        need_dtype(x, DType::F32, "rms input plane")?;
        need_dtype(out, DType::F32, "rms output plane")?;
        need_numel(x, total, "rms input plane")?;
        need_numel(out, total, "rms output plane")?;
        let gamma_bf16 = gamma_flag(gamma, width, "rms gamma")?;
        let (nr, w) = (i32_of(n_rows, "rows")?, i32_of(width, "width")?);
        self.bind_thread()?;
        self.ensure_kernel(
            SPLIT_MODULE,
            LATENT_ATTENTION_SPLIT_SRC,
            "latent_attention_rms_rows_bf16",
        )?;
        let (xp, gp, op) = (x.buf.as_ptr(), gamma.buf.as_ptr(), out.buf.as_ptr());
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
            &w as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
            &gamma_bf16 as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "latent_attention_rms_rows_bf16",
            [n_rows as u32, 1, 1],
            [SPLIT_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(gp);
                b.push_ptr(op);
                b.push_i32(nr);
                b.push_i32(w);
                b.push_f32(eps);
                b.push_i32(gamma_bf16);
                b
            },
        )
    }

    /// Cache rows `[base + row0, base + row0 + n_rows)` → F32 plane
    /// `[n_rows, latent]` (the latent channels only), the operand of an external
    /// (e.g. quantized) latent→expanded projector. `base` is an optional device
    /// scalar added to `row0`.
    #[allow(clippy::too_many_arguments)]
    pub fn latent_attention_gather_latent(
        &mut self,
        cache: &GpuTensor,
        base: Option<&GpuTensor>,
        out: &GpuTensor,
        row0: usize,
        n_rows: usize,
        latent: usize,
        tail: usize,
        cap: usize,
    ) -> HipResult<()> {
        if n_rows == 0 {
            return Ok(());
        }
        let width = latent + tail;
        need_dtype(cache, DType::BF16, "latent cache")?;
        need_numel(cache, product(cap, width, "cache")?, "latent cache")?;
        need_dtype(out, DType::F32, "gathered latent plane")?;
        let total = product(n_rows, latent, "gather")?;
        need_numel(out, total, "gathered latent plane")?;
        if let Some(b) = base {
            need_scalar_i32(b, "row base")?;
        }
        let (r0, nr, lat, tl, cp_rows) = (
            i32_of(row0, "row0")?,
            i32_of(n_rows, "rows")?,
            i32_of(latent, "latent")?,
            i32_of(tail, "tail")?,
            i32_of(cap, "cap")?,
        );
        self.bind_thread()?;
        self.ensure_kernel(
            DECOMPRESS_MODULE,
            LATENT_ATTENTION_DECOMPRESS_SRC,
            "latent_attention_gather_latent_f32",
        )?;
        let (cp, bp, op) = (
            cache.buf.as_ptr(),
            base.map_or(std::ptr::null_mut(), |b| b.buf.as_ptr()),
            out.buf.as_ptr(),
        );
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &r0 as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
            &lat as *const _ as *mut c_void,
            &tl as *const _ as *mut c_void,
            &cp_rows as *const _ as *mut c_void,
        ];
        let grid = total.div_ceil(ELEM_BLOCK as usize) as u32;
        self.launch_maybe_blob(
            "latent_attention_gather_latent_f32",
            [grid, 1, 1],
            [ELEM_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(cp);
                b.push_ptr(bp);
                b.push_ptr(op);
                b.push_i32(r0);
                b.push_i32(nr);
                b.push_i32(lat);
                b.push_i32(tl);
                b.push_i32(cp_rows);
                b
            },
        )
    }

    /// External projector output `[n_rows, n_heads*(nope+v)]` (F32) → BF16
    /// head-major `knope`/`vmir` mirrors at rows `base + row0 + i`.
    #[allow(clippy::too_many_arguments)]
    pub fn latent_attention_scatter_expanded(
        &mut self,
        proj: &GpuTensor,
        knope: &GpuTensor,
        vmir: &GpuTensor,
        base: Option<&GpuTensor>,
        row0: usize,
        n_rows: usize,
        n_heads: usize,
        nope: usize,
        v: usize,
        cap: usize,
    ) -> HipResult<()> {
        if n_rows == 0 {
            return Ok(());
        }
        let per = nope + v;
        let total = product(product(n_rows, n_heads, "scatter")?, per, "scatter")?;
        need_dtype(proj, DType::F32, "projector output")?;
        need_numel(proj, total, "projector output")?;
        self.check_mirrors(knope, vmir, n_heads, nope, v, cap)?;
        if let Some(b) = base {
            need_scalar_i32(b, "row base")?;
        }
        let (r0, nr, nh, np, vv, cp_rows) = (
            i32_of(row0, "row0")?,
            i32_of(n_rows, "rows")?,
            i32_of(n_heads, "heads")?,
            i32_of(nope, "nope")?,
            i32_of(v, "v")?,
            i32_of(cap, "cap")?,
        );
        self.bind_thread()?;
        self.ensure_kernel(
            DECOMPRESS_MODULE,
            LATENT_ATTENTION_DECOMPRESS_SRC,
            "latent_attention_scatter_expanded_bf16",
        )?;
        let (pp, kp, vp, bp) = (
            proj.buf.as_ptr(),
            knope.buf.as_ptr(),
            vmir.buf.as_ptr(),
            base.map_or(std::ptr::null_mut(), |b| b.buf.as_ptr()),
        );
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &r0 as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &np as *const _ as *mut c_void,
            &vv as *const _ as *mut c_void,
            &cp_rows as *const _ as *mut c_void,
        ];
        let grid = total.div_ceil(ELEM_BLOCK as usize) as u32;
        self.launch_maybe_blob(
            "latent_attention_scatter_expanded_bf16",
            [grid, 1, 1],
            [ELEM_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(bp);
                b.push_i32(r0);
                b.push_i32(nr);
                b.push_i32(nh);
                b.push_i32(np);
                b.push_i32(vv);
                b.push_i32(cp_rows);
                b
            },
        )
    }

    /// External projector output `[n_rows, n_heads*(nope+v)]` (F32) → Q8
    /// expanded mirror planes at rows `base + row0 + i`: per (row, head, plane)
    /// INT8 codes with one F16 scale (`F16_rne(max|x| / 127)`), laid out as rows of
    /// `n_heads * (width + 2)` bytes, codes then scales (see
    /// `kernels/src/latent_attention_decompress.hip` and
    /// [`crate::latent_attention_tiled::q8_plane_row_bytes`]). `knope` / `vmir`
    /// are `Raw` byte tensors of at least `cap` plane rows.
    #[allow(clippy::too_many_arguments)]
    pub fn latent_attention_scatter_expanded_q8(
        &mut self,
        proj: &GpuTensor,
        knope: &GpuTensor,
        vmir: &GpuTensor,
        base: Option<&GpuTensor>,
        row0: usize,
        n_rows: usize,
        n_heads: usize,
        nope: usize,
        v: usize,
        cap: usize,
    ) -> HipResult<()> {
        if n_rows == 0 {
            return Ok(());
        }
        let per = nope + v;
        let total = product(product(n_rows, n_heads, "scatter q8")?, per, "scatter q8")?;
        need_dtype(proj, DType::F32, "projector output")?;
        need_numel(proj, total, "projector output")?;
        need_dtype(knope, DType::Raw, "Q8 K_nope mirror")?;
        need_dtype(vmir, DType::Raw, "Q8 V mirror")?;
        for (t, w, what) in [(knope, nope, "Q8 K_nope mirror"), (vmir, v, "Q8 V mirror")] {
            let need = product(
                cap,
                crate::latent_attention_tiled::q8_plane_row_bytes(n_heads, w),
                what,
            )?;
            if t.byte_size() < need {
                return Err(err(format!(
                    "latent_attention: {what} holds {} bytes, needs {need} for {cap} rows",
                    t.byte_size()
                )));
            }
        }
        if let Some(b) = base {
            need_scalar_i32(b, "row base")?;
        }
        let (r0, nr, nh, np, vv, cp_rows) = (
            i32_of(row0, "row0")?,
            i32_of(n_rows, "rows")?,
            i32_of(n_heads, "heads")?,
            i32_of(nope, "nope")?,
            i32_of(v, "v")?,
            i32_of(cap, "cap")?,
        );
        let blocks = product(n_rows, n_heads, "scatter q8")?;
        if blocks > i32::MAX as usize {
            return Err(err("latent_attention: scatter q8 grid exceeds i32"));
        }
        // Kernel contract: 16-byte vector loads of the F32 projector rows, 4-byte code stores.
        if nope % 4 != 0
            || v % 4 != 0
            || (n_heads * (nope + 2)) % 4 != 0
            || (n_heads * (v + 2)) % 4 != 0
            || (proj.buf.as_ptr() as usize) % 16 != 0
            || (knope.buf.as_ptr() as usize) % 4 != 0
            || (vmir.buf.as_ptr() as usize) % 4 != 0
        {
            return Err(err(
                "latent_attention: scatter q8 needs nope and v multiples of 4, plane rows \
                 multiples of 4 bytes and a 16-byte aligned projector plane",
            ));
        }
        self.bind_thread()?;
        self.ensure_kernel(
            DECOMPRESS_MODULE,
            LATENT_ATTENTION_DECOMPRESS_SRC,
            "latent_attention_scatter_expanded_q8",
        )?;
        let (pp, kp, vp, bp) = (
            proj.buf.as_ptr(),
            knope.buf.as_ptr(),
            vmir.buf.as_ptr(),
            base.map_or(std::ptr::null_mut(), |b| b.buf.as_ptr()),
        );
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &r0 as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &np as *const _ as *mut c_void,
            &vv as *const _ as *mut c_void,
            &cp_rows as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "latent_attention_scatter_expanded_q8",
            [blocks as u32, 1, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(bp);
                b.push_i32(r0);
                b.push_i32(nr);
                b.push_i32(nh);
                b.push_i32(np);
                b.push_i32(vv);
                b.push_i32(cp_rows);
                b
            },
        )
    }

    /// Fused dense decompression: cache latent rows × `w[(nope+v)*n_heads,
    /// latent]` (BF16 or F32 weights, FP32 accumulation in ascending latent
    /// order) → BF16 `knope`/`vmir` mirrors at rows `base + row0 + i`.
    #[allow(clippy::too_many_arguments)]
    pub fn latent_attention_decompress_dense(
        &mut self,
        cache: &GpuTensor,
        base: Option<&GpuTensor>,
        w: &GpuTensor,
        knope: &GpuTensor,
        vmir: &GpuTensor,
        row0: usize,
        n_rows: usize,
        n_heads: usize,
        nope: usize,
        v: usize,
        latent: usize,
        tail: usize,
        cap: usize,
        mirror_cap: usize,
    ) -> HipResult<()> {
        if n_rows == 0 {
            return Ok(());
        }
        let width = latent + tail;
        let n_out = product(n_heads, nope + v, "kv_b rows")?;
        need_dtype(cache, DType::BF16, "latent cache")?;
        need_numel(cache, product(cap, width, "cache")?, "latent cache")?;
        let w_bf16 = gamma_flag(w, product(n_out, latent, "kv_b")?, "expansion weight")?;
        self.check_mirrors(knope, vmir, n_heads, nope, v, mirror_cap)?;
        let mcap = i32_of(mirror_cap, "mirror capacity")?;
        if let Some(b) = base {
            need_scalar_i32(b, "row base")?;
        }
        let row_tiles = n_rows.div_ceil(16);
        if row_tiles > 65535 {
            return Err(err("latent_attention: decompress row tiles exceed grid.y"));
        }
        let (r0, nr, nh, np, vv, lat, tl, cp_rows) = (
            i32_of(row0, "row0")?,
            i32_of(n_rows, "rows")?,
            i32_of(n_heads, "heads")?,
            i32_of(nope, "nope")?,
            i32_of(v, "v")?,
            i32_of(latent, "latent")?,
            i32_of(tail, "tail")?,
            i32_of(cap, "cap")?,
        );
        self.bind_thread()?;
        self.ensure_kernel(
            DECOMPRESS_MODULE,
            LATENT_ATTENTION_DECOMPRESS_SRC,
            "latent_attention_decompress_dense",
        )?;
        let (cp, bp, wp, kp, vp) = (
            cache.buf.as_ptr(),
            base.map_or(std::ptr::null_mut(), |b| b.buf.as_ptr()),
            w.buf.as_ptr(),
            knope.buf.as_ptr(),
            vmir.buf.as_ptr(),
        );
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &r0 as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &np as *const _ as *mut c_void,
            &vv as *const _ as *mut c_void,
            &lat as *const _ as *mut c_void,
            &tl as *const _ as *mut c_void,
            &cp_rows as *const _ as *mut c_void,
            &w_bf16 as *const _ as *mut c_void,
            &mcap as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "latent_attention_decompress_dense",
            [n_out.div_ceil(16) as u32, row_tiles as u32, 1],
            [16, 16, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(cp);
                b.push_ptr(bp);
                b.push_ptr(wp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_i32(r0);
                b.push_i32(nr);
                b.push_i32(nh);
                b.push_i32(np);
                b.push_i32(vv);
                b.push_i32(lat);
                b.push_i32(tl);
                b.push_i32(cp_rows);
                b.push_i32(w_bf16);
                b.push_i32(mcap);
                b
            },
        )
    }

    fn check_mirrors(
        &self,
        knope: &GpuTensor,
        vmir: &GpuTensor,
        n_heads: usize,
        nope: usize,
        v: usize,
        cap: usize,
    ) -> HipResult<()> {
        need_dtype(knope, DType::BF16, "K_nope mirror")?;
        need_dtype(vmir, DType::BF16, "V mirror")?;
        need_numel(
            knope,
            product(product(n_heads, cap, "K mirror")?, nope, "K mirror")?,
            "K_nope mirror",
        )?;
        need_numel(
            vmir,
            product(product(n_heads, cap, "V mirror")?, v, "V mirror")?,
            "V mirror",
        )
    }

    /// Serial reference attention over the latent cache + expanded mirrors.
    ///
    /// Query row `i` sees keys `j <= *pos + i` (device scalar `pos` = rows cached
    /// before this call, the new rows `*pos..*pos+n_rows` included in the
    /// mirrors/cache). `q` is `[n_rows, n_heads, nope+tail]`, `y` is
    /// `[n_rows, n_heads, v]`, both BF16-valued F32 planes. `scale` multiplies
    /// every dot product (caller passes `(nope+tail)^-0.5`).
    #[allow(clippy::too_many_arguments)]
    pub fn latent_attention_serial(
        &mut self,
        q: &GpuTensor,
        cache: &GpuTensor,
        knope: &GpuTensor,
        vmir: &GpuTensor,
        y: &GpuTensor,
        pos: &GpuTensor,
        n_rows: usize,
        n_heads: usize,
        nope: usize,
        tail: usize,
        v: usize,
        latent: usize,
        cap: usize,
        mirror_cap: usize,
        scale: f32,
    ) -> HipResult<()> {
        if n_rows == 0 {
            return Ok(());
        }
        if n_heads == 0 || nope + tail == 0 || v == 0 {
            return Err(err(
                "latent_attention: heads/query width/value width must be nonzero",
            ));
        }
        if v > LATENT_ATTENTION_SERIAL_MAX_V {
            return Err(err(format!(
                "latent_attention: serial route supports value width <= {LATENT_ATTENTION_SERIAL_MAX_V}, got {v}"
            )));
        }
        if n_rows > 65535 || n_heads > 65535 {
            return Err(err(
                "latent_attention: serial grid (heads, rows) exceeds 65535",
            ));
        }
        let dqk = nope + tail;
        let width = latent + tail;
        need_dtype(q, DType::F32, "query plane")?;
        need_numel(
            q,
            product(product(n_rows, n_heads, "q")?, dqk, "q")?,
            "query plane",
        )?;
        need_dtype(y, DType::F32, "attention output plane")?;
        need_numel(
            y,
            product(product(n_rows, n_heads, "y")?, v, "y")?,
            "attention output plane",
        )?;
        need_dtype(cache, DType::BF16, "latent cache")?;
        need_numel(cache, product(cap, width, "cache")?, "latent cache")?;
        self.check_mirrors(knope, vmir, n_heads, nope, v, mirror_cap)?;
        let mcap = i32_of(mirror_cap, "mirror capacity")?;
        need_scalar_i32(pos, "position")?;
        // 16-byte vector loads need every row start 16-byte aligned.
        let vec8 = nope % 8 == 0
            && tail % 8 == 0
            && latent % 8 == 0
            && (cache.buf.as_ptr() as usize) % 16 == 0
            && (knope.buf.as_ptr() as usize) % 16 == 0;
        let vec8_i = vec8 as i32;
        let (nr, nh, np, tl, vv, lat, cp_rows) = (
            i32_of(n_rows, "rows")?,
            i32_of(n_heads, "heads")?,
            i32_of(nope, "nope")?,
            i32_of(tail, "tail")?,
            i32_of(v, "v")?,
            i32_of(latent, "latent")?,
            i32_of(cap, "cap")?,
        );
        i32_of(product(cap, width, "cache")?, "cache elements")?;
        self.bind_thread()?;
        self.ensure_kernel(
            SERIAL_MODULE,
            LATENT_ATTENTION_SERIAL_SRC,
            "latent_attention_serial_bf16",
        )?;
        let (qp, cp, kp, vp, yp, pp) = (
            q.buf.as_ptr(),
            cache.buf.as_ptr(),
            knope.buf.as_ptr(),
            vmir.buf.as_ptr(),
            y.buf.as_ptr(),
            pos.buf.as_ptr(),
        );
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &np as *const _ as *mut c_void,
            &tl as *const _ as *mut c_void,
            &vv as *const _ as *mut c_void,
            &lat as *const _ as *mut c_void,
            &cp_rows as *const _ as *mut c_void,
            &scale as *const _ as *mut c_void,
            &vec8_i as *const _ as *mut c_void,
            &mcap as *const _ as *mut c_void,
        ];
        let shared = ((dqk + 2 * SERIAL_KEY_TILE) * 4) as u32;
        self.launch_maybe_blob(
            "latent_attention_serial_bf16",
            [n_heads as u32, n_rows as u32, 1],
            [SERIAL_KEY_TILE as u32, 1, 1],
            shared,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(cp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(yp);
                b.push_ptr(pp);
                b.push_i32(nr);
                b.push_i32(nh);
                b.push_i32(np);
                b.push_i32(tl);
                b.push_i32(vv);
                b.push_i32(lat);
                b.push_i32(cp_rows);
                b.push_f32(scale);
                b.push_i32(vec8_i);
                b.push_i32(mcap);
                b
            },
        )
    }

    /// `out = bf16(y * bf16(sigmoid(g)))` over `n` elements (BF16-valued F32
    /// planes). `out` may be the same tensor as `y`.
    pub fn latent_attention_gate(
        &mut self,
        y: &GpuTensor,
        g: &GpuTensor,
        out: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        if n == 0 {
            return Ok(());
        }
        for (t, what) in [(y, "gated input"), (g, "gate"), (out, "gated output")] {
            need_dtype(t, DType::F32, what)?;
            need_numel(t, n, what)?;
        }
        let nn = i32_of(n, "elements")?;
        self.bind_thread()?;
        self.ensure_kernel(
            GATE_MODULE,
            LATENT_ATTENTION_GATE_SRC,
            "latent_attention_gate_bf16",
        )?;
        let (yp, gp, op) = (y.buf.as_ptr(), g.buf.as_ptr(), out.buf.as_ptr());
        let mut params: Vec<*mut c_void> = vec![
            &yp as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &nn as *const _ as *mut c_void,
        ];
        let grid = n.div_ceil(ELEM_BLOCK as usize) as u32;
        self.launch_maybe_blob(
            "latent_attention_gate_bf16",
            [grid, 1, 1],
            [ELEM_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(yp);
                b.push_ptr(gp);
                b.push_ptr(op);
                b.push_i32(nn);
                b
            },
        )
    }
}
