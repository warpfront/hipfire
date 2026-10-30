// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Operand contract shared by the tiled latent-attention kernels
//! (`latent_attention_flash.rs` prefill, `latent_attention_decode.rs` split-KV
//! decode, `latent_attention_absorb.rs` absorbed-latent support).
//!
//! Nothing here is checkpoint-specific. Keys, values and queries are described
//! by *strided views*, so the same kernels serve
//!
//! * the decompressed route: `k_a` = expanded `K_nope` mirror rows
//!   (`[t][h][nope]`), `k_b` = the shared latent-cache tail, `v` = expanded V
//!   mirror rows (`[t][h][v]`);
//! * the absorbed route: `k_a` = the normalized latent channels of the cache,
//!   `k_b` = the cache tail, `v` = the latent channels again (value width =
//!   latent width), `q` = `[q_nope^T Wk || q_tail]` (see
//!   `latent_attention_absorb.rs`).
//!
//! Semantics (both routes): for query row `i` of this call, head `h`,
//!
//! ```text
//! key(j)   = k_a(j, h)[0..wa] ++ k_b(j, h)[0..wb]        (width dq = wa + wb)
//! score(j) = scale * dot(q[i][h][0..dq], key(j))
//! p        = softmax over the visible keys
//! y[i][h]  = sum_j p[j] * v(j, h)[0..wv]
//! ```
//!
//! visible keys are `j <= *pos + pos_offset + i` and `j < max_rows`; a query
//! row with no visible key writes zeros. No positional rotation.
//!
//! Numerics: scores/softmax/accumulation in FP32; K, V are BF16 (exact in the
//! products); the queries are F32 planes that the kernels may round to BF16
//! when they feed matrix units (documented per kernel). Outputs are F32 planes;
//! with `round_y_bf16` each element is rounded once to BF16 (and stored as the
//! F32 value of that BF16), as the reference's BF16 attention output.
//!
//! Determinism: fixed reduction trees/orders, no atomics, no dependence on the
//! launch's occupancy or on the host-side allocation capacity (a given
//! `(*pos + pos_offset + i)` produces the same bytes whatever `max_rows` is,
//! as long as `max_rows` does not truncate the visible range).
//!
//! Capture safety: `pos` is a device i32 scalar read by the kernels; every
//! other quantity is a launch-constant kernarg, all launches go through
//! `bind_thread` + `launch_maybe_blob`, there is no host readback and no
//! allocation inside a launcher.

use hip_bridge::{HipError, HipResult};

use crate::dispatch::{DType, GpuTensor};

/// A BF16 row source viewed as `[token][head][width]`: element `(t, h, d)` is
/// `buf[offset + t * tok_stride + h * head_stride + d]` (strides in BF16
/// elements; `head_stride == 0` means every head shares the row).
#[derive(Clone, Copy)]
pub struct LatentRows<'a> {
    pub buf: &'a GpuTensor,
    pub offset: usize,
    pub tok_stride: usize,
    pub head_stride: usize,
    pub width: usize,
}

/// One tiled attention call (prefill rows `> 1` or decode rows `== 1`).
#[derive(Clone, Copy)]
pub struct TiledAttention<'a> {
    /// F32 `[n_rows, n_heads, k_a.width + k_b.width]`.
    pub q: &'a GpuTensor,
    /// F32 `[n_rows, n_heads, v.width]`.
    pub y: &'a GpuTensor,
    pub k_a: LatentRows<'a>,
    pub k_b: LatentRows<'a>,
    pub v: LatentRows<'a>,
    /// Device i32 scalar: rows already cached before this call.
    pub pos: &'a GpuTensor,
    /// Host-constant rows added to `*pos` (lets a caller split a long prefill
    /// into sub-calls that share one device position scalar).
    pub pos_offset: usize,
    pub n_rows: usize,
    pub n_heads: usize,
    pub scale: f32,
    /// Hard bound: keys with `j >= max_rows` are never read (the smaller of the
    /// latent cache's and the expanded mirror's mapped rows).
    pub max_rows: usize,
    pub round_y_bf16: bool,
}

pub(crate) fn err(msg: impl AsRef<str>) -> HipError {
    HipError::new(0, msg.as_ref())
}

fn bf16_view_end(v: &LatentRows<'_>, n_heads: usize, max_rows: usize, what: &str) -> HipResult<()> {
    if v.width == 0 {
        return Err(err(format!("latent_attention tiled: {what} width is zero")));
    }
    if v.buf.dtype != DType::BF16 {
        return Err(err(format!(
            "latent_attention tiled: {what} must be BF16, got {:?}",
            v.buf.dtype
        )));
    }
    // Last element touched: offset + (max_rows-1)*tok + (heads-1)*head + width.
    let end = max_rows
        .checked_sub(1)
        .and_then(|r| r.checked_mul(v.tok_stride))
        .and_then(|a| {
            n_heads
                .checked_sub(1)
                .and_then(|h| h.checked_mul(v.head_stride))
                .and_then(|b| a.checked_add(b))
        })
        .and_then(|a| a.checked_add(v.offset))
        .and_then(|a| a.checked_add(v.width))
        .ok_or_else(|| {
            err(format!(
                "latent_attention tiled: {what} extent overflows usize"
            ))
        })?;
    if end > v.buf.numel() {
        return Err(err(format!(
            "latent_attention tiled: {what} view needs {end} elements, tensor holds {}",
            v.buf.numel()
        )));
    }
    Ok(())
}

/// Bytes of one row of a Q8 expanded-mirror plane holding `n_heads` heads of
/// `width` channels: `n_heads * width` int8 codes followed by `n_heads` F16
/// scales (one per head). Head `h`'s codes start at byte `h * width`, its scale
/// at byte `n_heads * width + 2 * h`.
pub fn q8_plane_row_bytes(n_heads: usize, width: usize) -> usize {
    n_heads * (width + 2)
}

/// Q8 plane view contract: a `Raw` byte tensor whose view has `offset == 0`,
/// `head_stride == width` and `tok_stride == q8_plane_row_bytes(n_heads, width)`
/// (all in BYTES, the unit of `LatentRows` for these views).
fn q8_plane_end(v: &LatentRows<'_>, n_heads: usize, max_rows: usize, what: &str) -> HipResult<()> {
    if v.width == 0 {
        return Err(err(format!("latent_attention tiled: {what} width is zero")));
    }
    if v.buf.dtype != DType::Raw {
        return Err(err(format!(
            "latent_attention tiled: Q8 {what} plane must be Raw bytes, got {:?}",
            v.buf.dtype
        )));
    }
    if v.offset != 0
        || v.head_stride != v.width
        || v.tok_stride != q8_plane_row_bytes(n_heads, v.width)
    {
        return Err(err(format!(
            "latent_attention tiled: Q8 {what} plane view must have offset 0, head stride {} and \
             token stride {} bytes (got offset {}, head stride {}, token stride {})",
            v.width,
            q8_plane_row_bytes(n_heads, v.width),
            v.offset,
            v.head_stride,
            v.tok_stride
        )));
    }
    let need = max_rows.checked_mul(v.tok_stride).ok_or_else(|| {
        err(format!(
            "latent_attention tiled: {what} extent overflows usize"
        ))
    })?;
    if need > v.buf.byte_size() {
        return Err(err(format!(
            "latent_attention tiled: Q8 {what} plane needs {need} bytes, tensor holds {}",
            v.buf.byte_size()
        )));
    }
    Ok(())
}

impl TiledAttention<'_> {
    /// Query/key width per head.
    pub fn dq(&self) -> usize {
        self.k_a.width + self.k_b.width
    }

    /// Value width per head.
    pub fn dv(&self) -> usize {
        self.v.width
    }

    /// Geometry/extent validation shared by every tiled launcher. Does not
    /// check kernel-specific alignment; launchers add their own.
    pub fn validate(&self) -> HipResult<()> {
        self.validate_with(false)
    }

    /// [`Self::validate`] for the Q8 expanded-mirror launchers: `k_a` and `v`
    /// are INT8 mirror planes (see [`q8_plane_row_bytes`]), `k_b` stays BF16.
    pub fn validate_q8(&self) -> HipResult<()> {
        self.validate_with(true)
    }

    fn validate_with(&self, q8: bool) -> HipResult<()> {
        if self.n_rows == 0 || self.n_heads == 0 {
            return Err(err(
                "latent_attention tiled: rows and heads must be nonzero",
            ));
        }
        if self.max_rows == 0 {
            return Err(err("latent_attention tiled: max_rows must be nonzero"));
        }
        let (dq, dv) = (self.dq(), self.dv());
        for (t, d, what, n) in [
            (
                self.q,
                DType::F32,
                "query plane",
                self.n_rows * self.n_heads * dq,
            ),
            (
                self.y,
                DType::F32,
                "output plane",
                self.n_rows * self.n_heads * dv,
            ),
        ] {
            if t.dtype != d {
                return Err(err(format!(
                    "latent_attention tiled: {what} must be {d:?}, got {:?}",
                    t.dtype
                )));
            }
            if t.numel() < n {
                return Err(err(format!(
                    "latent_attention tiled: {what} holds {} elements, needs {n}",
                    t.numel()
                )));
            }
        }
        if self.pos.byte_size() < 4 {
            return Err(err(
                "latent_attention tiled: pos must hold a 4-byte device scalar",
            ));
        }
        if q8 {
            q8_plane_end(&self.k_a, self.n_heads, self.max_rows, "k_a")?;
        } else {
            bf16_view_end(&self.k_a, self.n_heads, self.max_rows, "k_a")?;
        }
        bf16_view_end(&self.k_b, self.n_heads, self.max_rows, "k_b")?;
        if q8 {
            q8_plane_end(&self.v, self.n_heads, self.max_rows, "v")?;
        } else {
            bf16_view_end(&self.v, self.n_heads, self.max_rows, "v")?;
        }
        if i32::try_from(self.max_rows + self.pos_offset + self.n_rows).is_err() {
            return Err(err("latent_attention tiled: row counts exceed i32"));
        }
        Ok(())
    }
}
