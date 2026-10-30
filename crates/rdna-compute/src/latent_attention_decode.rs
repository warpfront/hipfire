// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! latent attention decode (see latent_attention_tiled.rs for the shared contract).
//!
//! Split-KV decode attention for a single query row (`n_rows == 1`), FP32
//! throughout (BF16 keys/values are widened exactly; the query is used as F32),
//! deterministic, no atomics, graph-capture safe.
//!
//! Two kernels (`kernels/src/latent_attention_decode.hip`) are launched back to
//! back by [`Gpu::latent_attention_decode_split`]:
//!
//! 1. *partials*, grid `(LATENT_DECODE_SPLITS, n_heads)`: the visible key count
//!    `len = min(*pos + pos_offset + 1, max_rows)` is read from the device
//!    scalar inside the kernel; split `s` owns the contiguous key range
//!    `[s * per, min((s + 1) * per, len))` with
//!    `per = round_up(ceil(len / SPLITS), 16)`, a function of `len` only, so
//!    the result for a given `len` does not depend on `max_rows`, on the
//!    allocation capacity, or on whether the launch is eager or a graph replay.
//!    Empty splits exit before touching any memory.
//! 2. *merge*, one block per head: fixed ascending split order over the
//!    non-empty splits; `len == 0` writes zeros.
//!
//! Both grids are launch constants; only the device scalar decides how much
//! work the blocks do. `pos` must not be modified between the two kernels (they
//! are consecutive launches on the same stream).
//!
//! Workspace: an F32 tensor of at least
//! [`latent_attention_decode_workspace_elems`] elements holding
//! `[n_heads][LATENT_DECODE_SPLITS][v_width + 2]` records `(m, l, acc[..])`.
//! Its contents are scratch; it is fully rewritten (for the non-empty splits)
//! by every call and may be reused by any other decode call on the stream.
//!
//! Limits (refused with an explicit error): `n_rows == 1`; every view width,
//! offset and stride a multiple of 8 BF16 elements and every view base pointer
//! 16-byte aligned (16-byte vector loads); `k_a.width + k_b.width <=`
//! [`LATENT_DECODE_MAX_DQ`], `v.width <=` [`LATENT_DECODE_MAX_DV`];
//! `n_heads <= 65535`; `max_rows <= 2^30`.

use std::ffi::c_void;

use hip_bridge::{HipResult, KernargBlob};

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::latent_attention_tiled::{err, LatentRows, TiledAttention};

pub const LATENT_ATTENTION_DECODE_SRC: &str =
    include_str!("../../../kernels/src/latent_attention_decode.hip");

const DECODE_MODULE: &str = "latent_attention_decode";
const PARTIAL_KERNEL: &str = "latent_attention_decode_partial";
const MERGE_KERNEL: &str = "latent_attention_decode_merge";
const PARTIAL_Q8_KERNEL: &str = "latent_attention_decode_partial_q8";

/// Number of key splits (grid `x` of the partials kernel). Fixed: the merge
/// order, and therefore the output bytes for a given `len`, depend on it.
pub const LATENT_DECODE_SPLITS: usize = 32;
/// Largest split count the merge kernel's LDS tables hold (`LA_DEC_MAX_SPLITS`).
const MAX_SPLITS: usize = 64;
const _: () = assert!(LATENT_DECODE_SPLITS >= 1 && LATENT_DECODE_SPLITS <= MAX_SPLITS);

/// Largest query/key width `k_a.width + k_b.width` (`LA_DEC_MAX_D`).
pub const LATENT_DECODE_MAX_DQ: usize = 2048;
/// Largest value width (`LA_DEC_MAX_D`).
pub const LATENT_DECODE_MAX_DV: usize = 2048;
/// Threads per block of both kernels (`LA_DEC_THREADS`).
const DECODE_BLOCK: u32 = 256;
/// Elements per 16-byte vector load.
const VEC: usize = 8;
/// Visible-key bound keeping the kernels' 32-bit split arithmetic exact.
const MAX_ROWS_LIMIT: usize = 1 << 30;

/// F32 elements the workspace tensor must hold for `n_heads` heads and value
/// width `v_width`.
pub const fn latent_attention_decode_workspace_elems(n_heads: usize, v_width: usize) -> usize {
    n_heads * LATENT_DECODE_SPLITS * (v_width + 2)
}

fn check_view(v: &LatentRows<'_>, what: &str) -> HipResult<()> {
    for (value, name) in [
        (v.width, "width"),
        (v.offset, "offset"),
        (v.tok_stride, "tok_stride"),
        (v.head_stride, "head_stride"),
    ] {
        if value % VEC != 0 {
            return Err(err(format!(
                "latent_attention decode: {what} {name} {value} is not a multiple of {VEC} BF16 elements"
            )));
        }
    }
    if (v.buf.buf.as_ptr() as usize) % 16 != 0 {
        return Err(err(format!(
            "latent_attention decode: {what} base pointer is not 16-byte aligned"
        )));
    }
    Ok(())
}

/// Device pointer of element `view.offset` of the view's buffer.
fn view_base(v: &LatentRows<'_>) -> *mut c_void {
    (v.buf.buf.as_ptr() as usize + v.offset * 2) as *mut c_void
}

fn to_i32(v: usize, what: &str) -> HipResult<i32> {
    i32::try_from(v).map_err(|_| err(format!("latent_attention decode: {what} exceeds i32")))
}

impl Gpu {
    /// Split-KV decode attention for one query row; see the module docs.
    ///
    /// `a.q` is F32 `[1, n_heads, dq]`, `a.y` F32 `[1, n_heads, dv]`,
    /// `workspace` an F32 tensor of at least
    /// [`latent_attention_decode_workspace_elems`]`(a.n_heads, a.dv())`
    /// elements. Visible keys are `j < min(*pos + pos_offset + 1, max_rows)`;
    /// a head with no visible key writes zeros.
    pub fn latent_attention_decode_split(
        &mut self,
        a: &TiledAttention<'_>,
        workspace: &GpuTensor,
    ) -> HipResult<()> {
        self.decode_split_impl(a, workspace, false)
    }

    /// [`Self::latent_attention_decode_split`] over a Q8 expanded mirror:
    /// `a.k_a` and `a.v` are INT8 mirror planes (`Raw` bytes, rows of
    /// [`crate::latent_attention_tiled::q8_plane_row_bytes`]: codes then F16
    /// per-head scales; see `kernels/src/latent_attention_decode.hip`), `a.k_b`
    /// the BF16 latent cache tail. Same split layout, merge kernel and
    /// determinism contract.
    pub fn latent_attention_decode_split_q8(
        &mut self,
        a: &TiledAttention<'_>,
        workspace: &GpuTensor,
    ) -> HipResult<()> {
        self.decode_split_impl(a, workspace, true)
    }

    fn decode_split_impl(
        &mut self,
        a: &TiledAttention<'_>,
        workspace: &GpuTensor,
        q8: bool,
    ) -> HipResult<()> {
        if a.n_rows != 1 {
            return Err(err(format!(
                "latent_attention decode: supports exactly one query row, got n_rows = {}",
                a.n_rows
            )));
        }
        if q8 {
            a.validate_q8()?;
        } else {
            a.validate()?;
        }
        let (wa, wb, dv) = (a.k_a.width, a.k_b.width, a.dv());
        let dq = wa + wb;
        if dq > LATENT_DECODE_MAX_DQ {
            return Err(err(format!(
                "latent_attention decode: query/key width {dq} exceeds the supported maximum {LATENT_DECODE_MAX_DQ}"
            )));
        }
        if dv > LATENT_DECODE_MAX_DV {
            return Err(err(format!(
                "latent_attention decode: value width {dv} exceeds the supported maximum {LATENT_DECODE_MAX_DV}"
            )));
        }
        if a.n_heads > 65535 {
            return Err(err(format!(
                "latent_attention decode: n_heads {} exceeds the grid limit 65535",
                a.n_heads
            )));
        }
        if a.max_rows > MAX_ROWS_LIMIT {
            return Err(err(format!(
                "latent_attention decode: max_rows {} exceeds {MAX_ROWS_LIMIT}",
                a.max_rows
            )));
        }
        check_view(&a.k_a, "k_a")?;
        check_view(&a.k_b, "k_b")?;
        check_view(&a.v, "v")?;
        let ws_need = latent_attention_decode_workspace_elems(a.n_heads, dv);
        if workspace.dtype != DType::F32 {
            return Err(err(format!(
                "latent_attention decode: workspace must be F32, got {:?}",
                workspace.dtype
            )));
        }
        if workspace.numel() < ws_need {
            return Err(err(format!(
                "latent_attention decode: workspace holds {} elements, needs {ws_need} \
                 ([{}][{LATENT_DECODE_SPLITS}][{dv} + 2])",
                workspace.numel(),
                a.n_heads
            )));
        }

        let (wa_i, wb_i, dv_i) = (
            to_i32(wa, "k_a width")?,
            to_i32(wb, "k_b width")?,
            to_i32(dv, "value width")?,
        );
        let pos_offset = to_i32(a.pos_offset, "pos_offset")?;
        let max_rows = to_i32(a.max_rows, "max_rows")?;
        let splits = LATENT_DECODE_SPLITS as i32;
        let round_y = a.round_y_bf16 as i32;
        let scale = a.scale;

        let (qp, kap, kbp, vp) = (
            a.q.buf.as_ptr(),
            view_base(&a.k_a),
            view_base(&a.k_b),
            view_base(&a.v),
        );
        let (wsp, yp, pp) = (workspace.buf.as_ptr(), a.y.buf.as_ptr(), a.pos.buf.as_ptr());
        let (ka_tok, ka_head) = (a.k_a.tok_stride as i64, a.k_a.head_stride as i64);
        let (kb_tok, kb_head) = (a.k_b.tok_stride as i64, a.k_b.head_stride as i64);
        let (v_tok, v_head) = (a.v.tok_stride as i64, a.v.head_stride as i64);

        let partial_kernel = if q8 {
            PARTIAL_Q8_KERNEL
        } else {
            PARTIAL_KERNEL
        };
        let nh = to_i32(a.n_heads, "n_heads")?;
        self.bind_thread()?;
        self.ensure_kernel(DECODE_MODULE, LATENT_ATTENTION_DECODE_SRC, partial_kernel)?;
        self.ensure_kernel(DECODE_MODULE, LATENT_ATTENTION_DECODE_SRC, MERGE_KERNEL)?;

        let mut partial_params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kap as *const _ as *mut c_void,
            &kbp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &wsp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &ka_tok as *const _ as *mut c_void,
            &ka_head as *const _ as *mut c_void,
            &kb_tok as *const _ as *mut c_void,
            &kb_head as *const _ as *mut c_void,
            &v_tok as *const _ as *mut c_void,
            &v_head as *const _ as *mut c_void,
            &wa_i as *const _ as *mut c_void,
            &wb_i as *const _ as *mut c_void,
            &dv_i as *const _ as *mut c_void,
            &pos_offset as *const _ as *mut c_void,
            &max_rows as *const _ as *mut c_void,
            &splits as *const _ as *mut c_void,
            &scale as *const _ as *mut c_void,
        ];
        if q8 {
            partial_params.push(&nh as *const _ as *mut c_void);
        }
        self.launch_maybe_blob(
            partial_kernel,
            [LATENT_DECODE_SPLITS as u32, a.n_heads as u32, 1],
            [DECODE_BLOCK, 1, 1],
            0,
            &mut partial_params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kap);
                b.push_ptr(kbp);
                b.push_ptr(vp);
                b.push_ptr(wsp);
                b.push_ptr(pp);
                b.push_u64(ka_tok as u64);
                b.push_u64(ka_head as u64);
                b.push_u64(kb_tok as u64);
                b.push_u64(kb_head as u64);
                b.push_u64(v_tok as u64);
                b.push_u64(v_head as u64);
                b.push_i32(wa_i);
                b.push_i32(wb_i);
                b.push_i32(dv_i);
                b.push_i32(pos_offset);
                b.push_i32(max_rows);
                b.push_i32(splits);
                b.push_f32(scale);
                if q8 {
                    b.push_i32(nh);
                }
                b
            },
        )?;

        let mut merge_params: Vec<*mut c_void> = vec![
            &wsp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &dv_i as *const _ as *mut c_void,
            &pos_offset as *const _ as *mut c_void,
            &max_rows as *const _ as *mut c_void,
            &splits as *const _ as *mut c_void,
            &round_y as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            MERGE_KERNEL,
            [a.n_heads as u32, 1, 1],
            [DECODE_BLOCK, 1, 1],
            0,
            &mut merge_params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(wsp);
                b.push_ptr(yp);
                b.push_ptr(pp);
                b.push_i32(dv_i);
                b.push_i32(pos_offset);
                b.push_i32(max_rows);
                b.push_i32(splits);
                b.push_i32(round_y);
                b
            },
        )
    }
}
