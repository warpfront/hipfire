// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! latent attention flash (see latent_attention_tiled.rs for the shared contract).
//!
//! Tiled causal prefill: online softmax over 64-key tiles, BF16 matrix-unit
//! score and value products, no `T x T` storage, no atomics. One
//! `latent_attention_flash.hip` source serves gfx11 (wave32 WMMA, 16-wide
//! operands) and gfx12 (8-wide operands); the arch split is compile-time.
//!
//! Decomposition (value width <= 128, or a width the wide kernel cannot take):
//! grid `[ceil(rows / (16 * waves)), heads, ceil(v.width / 128)]`, block
//! `32 * waves` threads (one wave per 16 query rows); every `z` block of a head
//! recomputes the scores for its 128-column value chunk. Query tiles are
//! launched heaviest (latest) first. `waves` is the largest of 4, 2, 1 whose LDS
//! fits [`LATENT_ATTENTION_FLASH_LDS_LIMIT`].
//!
//! Wide values (`v.width` a multiple of 64 up to 512, the absorbed latent route
//! has 512): one 4-wave block owns a 16-query tile of a head, the four waves
//! split the score contraction and the value columns, and every score and value
//! product is computed once (`latent_attention_flash_prefill_wide`; grid
//! `[ceil(rows / 16), heads, 1]`, block 128).
//!
//! Numerics: queries are rounded once to BF16 at LDS staging; scores, softmax
//! state and accumulation are FP32; probabilities are rounded to BF16 for the
//! value product and the normalizer is summed from those rounded values;
//! exponentials use the hardware base-2 exponential.

use std::ffi::c_void;

use hip_bridge::{HipResult, KernargBlob};

use crate::dispatch::Gpu;
use crate::latent_attention_tiled::{err, LatentRows, TiledAttention};

pub const LATENT_ATTENTION_FLASH_SRC: &str =
    include_str!("../../../kernels/src/latent_attention_flash.hip");

const FLASH_MODULE: &str = "latent_attention_flash";
const FLASH_FUNC: &str = "latent_attention_flash_prefill";
const FLASH_Q8_FUNC: &str = "latent_attention_flash_prefill_q8";

/// Largest query/key width (`k_a.width + k_b.width`) the kernel supports.
pub const LATENT_ATTENTION_FLASH_MAX_DQ: usize = 1024;
/// Dynamic LDS budget per block (the per-workgroup limit of every target arch).
pub const LATENT_ATTENTION_FLASH_LDS_LIMIT: usize = 65536;
/// Value columns per `z` block (fixed by the kernel).
const FLASH_DV_CHUNK: usize = 128;
/// Words per transposed-value LDS row (fixed by the kernel).
const FLASH_VP_WORDS: usize = 36;

fn i32_of(v: usize, what: &str) -> HipResult<i32> {
    i32::try_from(v).map_err(|_| err(format!("latent_attention flash: {what} exceeds i32")))
}

/// Element-offset-adjusted base pointer of a view, 16-byte alignment checked.
fn view_ptr(v: &LatentRows<'_>, what: &str) -> HipResult<*mut c_void> {
    for (x, n) in [
        (v.offset, "offset"),
        (v.tok_stride, "token stride"),
        (v.head_stride, "head stride"),
    ] {
        if x % 8 != 0 {
            return Err(err(format!(
                "latent_attention flash: {what} {n} {x} is not a multiple of 8 elements"
            )));
        }
    }
    if v.width % 16 != 0 {
        return Err(err(format!(
            "latent_attention flash: {what} width {} is not a multiple of 16",
            v.width
        )));
    }
    let p = (v.buf.buf.as_ptr() as usize)
        .checked_add(v.offset * 2)
        .ok_or_else(|| err(format!("latent_attention flash: {what} pointer overflows")))?;
    if p % 16 != 0 {
        return Err(err(format!(
            "latent_attention flash: {what} base pointer is not 16-byte aligned"
        )));
    }
    Ok(p as *mut c_void)
}

/// Words per transposed-value LDS row of the wide kernel.
const WIDE_VP_WORDS: usize = 20;
/// Score-partial exchange area of the wide kernel (4 waves x 8 x 32 lanes x 4 B).
const WIDE_SP_BYTES: usize = 4096;
/// Largest value width the wide kernel takes (4 waves x 128 columns).
const WIDE_MAX_DV: usize = 512;

fn wide_lds_bytes(dq: usize, dv: usize) -> usize {
    16 * (dq + 8) * 2 + dv * WIDE_VP_WORDS * 4 + WIDE_SP_BYTES
}

fn flash_lds_bytes(waves: usize, dq: usize, dv: usize) -> usize {
    waves * 16 * (dq + 8) * 2 + dv.min(FLASH_DV_CHUNK) * FLASH_VP_WORDS * 4
}

/// LDS bytes of the dequantized K_nope tile of the Q8 kernel (64 keys).
fn q8_k_tile_bytes(wa: usize) -> usize {
    64 * (wa + 8) * 2
}

/// Base pointer of a Q8 plane view (byte units): 16-byte aligned base, whole-row
/// byte strides that are multiples of 16, width a multiple of 16.
fn q8_view_ptr(v: &LatentRows<'_>, what: &str) -> HipResult<*mut c_void> {
    if v.tok_stride % 16 != 0 {
        return Err(err(format!(
            "latent_attention flash: Q8 {what} token stride {} is not a multiple of 16 bytes",
            v.tok_stride
        )));
    }
    if v.width % 16 != 0 {
        return Err(err(format!(
            "latent_attention flash: {what} width {} is not a multiple of 16",
            v.width
        )));
    }
    let p = v.buf.buf.as_ptr() as usize;
    if p % 16 != 0 {
        return Err(err(format!(
            "latent_attention flash: Q8 {what} base pointer is not 16-byte aligned"
        )));
    }
    Ok(p as *mut c_void)
}

impl Gpu {
    /// Causal flash-style prefill attention for `a.n_rows` query rows (`>= 1`;
    /// intended for `> 1`, see `latent_attention_decode` for single rows).
    ///
    /// Requires `k_a.width`, `k_b.width`, `v.width` multiples of 16, every
    /// view offset/stride a multiple of 8 elements, 16-byte aligned view and
    /// `q`/`y` base pointers, `dq <= LATENT_ATTENTION_FLASH_MAX_DQ`, and
    /// `n_heads <= 65535`. Deterministic: fixed reduction orders, no atomics,
    /// bytes independent of `max_rows` as long as it does not truncate.
    pub fn latent_attention_flash_prefill(&mut self, a: &TiledAttention<'_>) -> HipResult<()> {
        self.flash_prefill_impl(
            a,
            FLASH_MODULE,
            FLASH_FUNC,
            LATENT_ATTENTION_FLASH_SRC,
            false,
        )
    }

    /// [`Self::latent_attention_flash_prefill`] over a Q8 expanded mirror:
    /// `a.k_a` and `a.v` are INT8 mirror planes (`Raw` bytes, rows of
    /// [`crate::latent_attention_tiled::q8_plane_row_bytes`], codes then F16
    /// per-head scales; see `kernels/src/latent_attention_flash.hip`), `a.k_b`
    /// the BF16 latent cache tail. The value route is the chunked kernel (no wide
    /// variant). Same determinism contract; the K_nope tile is dequantized once
    /// per block into LDS, so the LDS budget includes `64 * (k_a.width + 8) * 2`
    /// bytes. Requires `k_a.width / 8` to be a power of two.
    pub fn latent_attention_flash_prefill_q8(&mut self, a: &TiledAttention<'_>) -> HipResult<()> {
        self.flash_prefill_impl(
            a,
            FLASH_MODULE,
            FLASH_Q8_FUNC,
            LATENT_ATTENTION_FLASH_SRC,
            true,
        )
    }

    /// [`Self::latent_attention_flash_prefill`] with an explicit kernel source
    /// and module/function names (the shipped source plus a leading `#define`
    /// for fixture builds; see the kernel header).
    #[doc(hidden)]
    pub fn latent_attention_flash_prefill_with_source(
        &mut self,
        a: &TiledAttention<'_>,
        module: &str,
        func: &str,
        src: &str,
    ) -> HipResult<()> {
        self.flash_prefill_impl(a, module, func, src, false)
    }

    fn flash_prefill_impl(
        &mut self,
        a: &TiledAttention<'_>,
        module: &str,
        func: &str,
        src: &str,
        q8: bool,
    ) -> HipResult<()> {
        if q8 {
            a.validate_q8()?;
            if !(a.k_a.width / 8).is_power_of_two() {
                return Err(err(format!(
                    "latent_attention flash: Q8 K tile staging needs k_a width / 8 to be a power \
                     of two, got width {}",
                    a.k_a.width
                )));
            }
        } else {
            a.validate()?;
        }
        let wide_func = format!("{func}_wide");
        let (wa, wb, dv) = (a.k_a.width, a.k_b.width, a.v.width);
        let dq = wa + wb;
        if dq > LATENT_ATTENTION_FLASH_MAX_DQ {
            return Err(err(format!(
                "latent_attention flash: query width {dq} exceeds {LATENT_ATTENTION_FLASH_MAX_DQ}"
            )));
        }
        if dv % 16 != 0 {
            return Err(err(format!(
                "latent_attention flash: value width {dv} is not a multiple of 16"
            )));
        }
        if a.n_heads > 65535 {
            return Err(err(
                "latent_attention flash: n_heads exceeds 65535 (grid.y)",
            ));
        }
        let (ka_p, v_p) = if q8 {
            (q8_view_ptr(&a.k_a, "k_a")?, q8_view_ptr(&a.v, "v")?)
        } else {
            (view_ptr(&a.k_a, "k_a")?, view_ptr(&a.v, "v")?)
        };
        let kb_p = view_ptr(&a.k_b, "k_b")?;
        let (qp, yp, pp) = (a.q.buf.as_ptr(), a.y.buf.as_ptr(), a.pos.buf.as_ptr());
        if (qp as usize) % 16 != 0 || (yp as usize) % 16 != 0 {
            return Err(err(
                "latent_attention flash: q/y base pointers must be 16-byte aligned",
            ));
        }
        let wide = !q8
            && dv > FLASH_DV_CHUNK
            && dv % 64 == 0
            && dv <= WIDE_MAX_DV
            && wide_lds_bytes(dq, dv) <= LATENT_ATTENTION_FLASH_LDS_LIMIT;
        let (waves, shared, func, grid_z) = if wide {
            (
                4usize,
                wide_lds_bytes(dq, dv) as u32,
                wide_func.as_str(),
                1usize,
            )
        } else {
            let lds =
                |w: usize| flash_lds_bytes(w, dq, dv) + if q8 { q8_k_tile_bytes(wa) } else { 0 };
            let waves = [4usize, 2, 1]
                .into_iter()
                .find(|&w| lds(w) <= LATENT_ATTENTION_FLASH_LDS_LIMIT)
                .ok_or_else(|| {
                    err(format!(
                        "latent_attention flash: query width {dq} needs more than \
                         {LATENT_ATTENTION_FLASH_LDS_LIMIT} bytes of LDS"
                    ))
                })?;
            (waves, lds(waves) as u32, func, dv.div_ceil(FLASH_DV_CHUNK))
        };
        let (nr, nh, wai, wbi, dvi, po, mr) = (
            i32_of(a.n_rows, "rows")?,
            i32_of(a.n_heads, "heads")?,
            i32_of(wa, "k_a width")?,
            i32_of(wb, "k_b width")?,
            i32_of(dv, "value width")?,
            i32_of(a.pos_offset, "pos_offset")?,
            i32_of(a.max_rows, "max_rows")?,
        );
        let (kat, kah, kbt, kbh, vt, vh) = (
            i32_of(a.k_a.tok_stride, "k_a token stride")?,
            i32_of(a.k_a.head_stride, "k_a head stride")?,
            i32_of(a.k_b.tok_stride, "k_b token stride")?,
            i32_of(a.k_b.head_stride, "k_b head stride")?,
            i32_of(a.v.tok_stride, "v token stride")?,
            i32_of(a.v.head_stride, "v head stride")?,
        );
        let scale = a.scale;
        let round_y = a.round_y_bf16 as i32;

        self.bind_thread()?;
        self.ensure_kernel(module, src, func)?;

        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &ka_p as *const _ as *mut c_void,
            &kb_p as *const _ as *mut c_void,
            &v_p as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &wai as *const _ as *mut c_void,
            &wbi as *const _ as *mut c_void,
            &dvi as *const _ as *mut c_void,
            &po as *const _ as *mut c_void,
            &mr as *const _ as *mut c_void,
            &kat as *const _ as *mut c_void,
            &kah as *const _ as *mut c_void,
            &kbt as *const _ as *mut c_void,
            &kbh as *const _ as *mut c_void,
            &vt as *const _ as *mut c_void,
            &vh as *const _ as *mut c_void,
            &scale as *const _ as *mut c_void,
            &round_y as *const _ as *mut c_void,
        ];
        let tiles = a.n_rows.div_ceil(if wide { 16 } else { 16 * waves });
        let grid = [tiles as u32, a.n_heads as u32, grid_z as u32];
        self.launch_maybe_blob(
            func,
            grid,
            [32 * waves as u32, 1, 1],
            shared,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(yp);
                b.push_ptr(ka_p);
                b.push_ptr(kb_p);
                b.push_ptr(v_p);
                b.push_ptr(pp);
                for x in [nr, nh, wai, wbi, dvi, po, mr, kat, kah, kbt, kbh, vt, vh] {
                    b.push_i32(x);
                }
                b.push_f32(scale);
                b.push_i32(round_y);
                b
            },
        )
    }
}
