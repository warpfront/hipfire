// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `LatentAttentionOp` / `execute_latent_attention`: the generic latent
//! attention pipeline (split latent/tail, latent row cache, bounded expanded
//! K/V mirrors, unequal QK/V attention with causal offsets, BF16 output gate).
//!
//! One call executes one stage; the caller sequences the stages around its own
//! projections (the query projection and any quantized expansion projector are
//! ordinary projections and stay outside this op):
//!
//! ```text
//! NormRows(q_a out) -> [caller: q = q_b * qlat]
//! SplitKv(kv_a out) -> latent cache row pos+i
//! Decompress(NewRows) | GatherLatent -> [caller projector] -> Decompress(External)
//! Attention(q, mirror) -> Gate(y, g)  -> [caller: o_proj]
//! ```
//!
//! The op borrows persistent buffers and owns nothing. Positions/lengths are a
//! device-resident i32 scalar (rows already cached) that no stage mutates, so
//! every stage is graph-capture safe and performs no host readback/allocation.
//! Decode is the `rows == 1` instance of the same route.

use rdna_compute::latent_attention_tiled::{q8_plane_row_bytes, LatentRows, TiledAttention};
use rdna_compute::{Gpu, GpuTensor};

use crate::types::DispatchError;

/// Independent attention geometry. Nothing here is checkpoint-specific.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatentGeometry {
    pub n_heads: usize,
    /// Per-head key channels expanded from the latent.
    pub nope: usize,
    /// Cached channels shared by all heads (plain extra QK channels).
    pub tail: usize,
    /// Per-head value channels (independent of `nope + tail`).
    pub v_dim: usize,
    /// Normalized cached latent channels.
    pub latent: usize,
}

impl LatentGeometry {
    /// Query/key width per head.
    pub fn qk_dim(&self) -> usize {
        self.nope + self.tail
    }
    /// Cache row width (`latent + tail`).
    pub fn row_width(&self) -> usize {
        self.latent + self.tail
    }
    /// Expansion projection output width (`n_heads * (nope + v_dim)`).
    pub fn expanded_width(&self) -> usize {
        self.n_heads * (self.nope + self.v_dim)
    }
    /// Default score scale `(nope + tail)^-0.5`.
    pub fn default_scale(&self) -> f32 {
        (self.qk_dim() as f32).powf(-0.5)
    }
}

/// BF16 latent row cache for one layer/slot, row-major `[capacity_rows, latent+tail]`.
#[derive(Clone, Copy)]
pub struct LatentCache<'a> {
    pub rows: &'a GpuTensor,
    pub capacity_rows: usize,
    /// Device scalar: number of rows already cached before this call.
    pub pos: &'a GpuTensor,
    /// Optional host-known copy of `*pos` for eager callers; enables a
    /// capacity refusal before launch. Capture callers pass `None` and own the
    /// length accounting.
    pub host_pos: Option<usize>,
}

/// Bounded expanded mirrors, token-major, valid for `t < pos`.
/// `capacity_rows` is the number of rows the caller has physically backed
/// (independent of the latent cache's mapped capacity, and a bound rather than
/// a stride, so a mirror can grow by mapping more rows in place). Sized by the
/// caller; never an unconditional max-sequence allocation.
///
/// * `q8 == false`: BF16 planes `knope[t][h][nope]`, `vmir[t][h][v_dim]`.
/// * `q8 == true`: `Raw` byte planes of INT8 codes with one F16 scale per
///   (row, head); a plane row is `n_heads * (width + 2)` bytes, codes then
///   scales (see `rdna_compute::latent_attention_tiled::q8_plane_row_bytes`).
///   Served by the tiled routes only; written by the `External` expansion only.
#[derive(Clone, Copy)]
pub struct ExpandedMirror<'a> {
    pub knope: &'a GpuTensor,
    pub vmir: &'a GpuTensor,
    pub capacity_rows: usize,
    pub q8: bool,
}

/// Which cache rows a stage covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowRange {
    /// Rows `*pos + 0 .. *pos + count` (the rows written this call).
    NewRows { count: usize },
    /// Absolute rows `start .. start + count`.
    Absolute { start: usize, count: usize },
}

/// How cached latent rows become expanded K_nope/V rows.
#[derive(Clone, Copy)]
pub enum KvExpansion<'a> {
    /// Built-in deterministic dense projector. `weight` is
    /// `[n_heads*(nope+v_dim), latent]` BF16 or F32 (row `h*(nope+v)+e`).
    Dense { weight: &'a GpuTensor },
    /// The caller already ran its own projector on a `GatherLatent` plane;
    /// `projected` is `[count, n_heads*(nope+v_dim)]` F32. This stage only
    /// rounds to BF16 and scatters into the mirrors.
    External { projected: &'a GpuTensor },
}

/// Operands of the absorbed-latent route (PLAN §1.6). The expansion weight is
/// absorbed into the query and the value instead of being applied to every
/// cached row, so attention reads only the latent cache (`latent + tail`
/// BF16 per token) and needs no expanded mirror. Not bit-exact to the
/// decompressed route (different reduction order, no intermediate BF16 K/V
/// rounding, BF16 query operand in the matrix-unit prefill kernel).
#[derive(Clone, Copy)]
pub struct AbsorbOperands<'a> {
    /// BF16 packed expansion weight from `Gpu::latent_attention_absorb_pack`.
    pub packed: &'a GpuTensor,
    /// F32 `[rows, n_heads, latent + tail]` scratch: `[q_nope^T Wk || q_tail]`.
    pub q_abs: &'a GpuTensor,
    /// F32 `[rows, n_heads, latent]` scratch: latent-space context `z`.
    pub z: &'a GpuTensor,
}

/// Which attention kernels serve an [`LatentAttentionOp::Attention`] stage.
#[derive(Clone, Copy)]
pub enum AttentionRoute<'a> {
    /// Deterministic per-head serial kernel over the expanded mirror: the
    /// oracle route.
    Reference,
    /// Tiled causal flash-style prefill (`rows > 1`) / split-KV decode
    /// (`rows == 1`, needs `workspace`) over the expanded mirror. Reordered
    /// arithmetic: inside the BF16 band, not bit-exact to `Reference`.
    Tiled { workspace: Option<&'a GpuTensor> },
    /// Absorbed latent attention (no mirror needed); the tiled kernels run in
    /// latent space and the value expansion is applied to `z`.
    Absorbed {
        ops: AbsorbOperands<'a>,
        workspace: Option<&'a GpuTensor>,
    },
}

/// One stage of the latent attention pipeline.
pub enum LatentAttentionOp<'a> {
    /// RMS over the full row, BF16-rounded F32 result (`out` may alias `x`).
    NormRows {
        x: &'a GpuTensor,
        gamma: &'a GpuTensor,
        out: &'a GpuTensor,
        rows: usize,
        width: usize,
        eps: f32,
    },
    /// Normalize the latent channels only, append the raw tail, write BF16 row
    /// `*pos + i` of the cache.
    SplitKv {
        geometry: LatentGeometry,
        cache: LatentCache<'a>,
        kv_a: &'a GpuTensor,
        gamma: &'a GpuTensor,
        rows: usize,
        eps: f32,
    },
    /// Cache latent rows → F32 `[count, latent]` plane for an external projector.
    GatherLatent {
        geometry: LatentGeometry,
        cache: LatentCache<'a>,
        range: RowRange,
        out: &'a GpuTensor,
    },
    /// Cache latent rows → expanded BF16 K_nope/V mirror rows.
    Decompress {
        geometry: LatentGeometry,
        cache: LatentCache<'a>,
        mirror: ExpandedMirror<'a>,
        expansion: KvExpansion<'a>,
        range: RowRange,
    },
    /// Attention (query row `i` sees keys `j <= *pos + i`) on the selected
    /// [`AttentionRoute`]. `mirror` is required by `Reference` and `Tiled`.
    Attention {
        geometry: LatentGeometry,
        cache: LatentCache<'a>,
        mirror: Option<ExpandedMirror<'a>>,
        /// `[rows, n_heads, nope+tail]` F32 (BF16-valued).
        q: &'a GpuTensor,
        /// `[rows, n_heads, v_dim]` F32 (BF16-valued).
        y: &'a GpuTensor,
        rows: usize,
        scale: f32,
        route: AttentionRoute<'a>,
    },
    /// `out = bf16(y * bf16(sigmoid(g)))` (`out` may alias `y`).
    Gate {
        y: &'a GpuTensor,
        g: &'a GpuTensor,
        out: &'a GpuTensor,
        elements: usize,
    },
}

fn refuse(msg: String) -> DispatchError {
    DispatchError::Hip(format!("latent_attention: {msg}"))
}

fn hip(e: hip_bridge::HipError) -> DispatchError {
    DispatchError::Hip(e.to_string())
}

fn check_geometry(g: &LatentGeometry) -> Result<(), DispatchError> {
    if g.n_heads == 0 || g.latent == 0 || g.v_dim == 0 || g.nope + g.tail == 0 {
        return Err(refuse(format!("degenerate geometry {g:?}")));
    }
    Ok(())
}

/// Refuse when the host-known live length exceeds the mirror's admitted rows
/// (device-side kernels additionally clamp, never read/write past the stride).
fn check_mirror(m: &ExpandedMirror<'_>, end: Option<usize>) -> Result<(), DispatchError> {
    if m.capacity_rows == 0 {
        return Err(refuse("expanded mirror capacity is zero".into()));
    }
    if let Some(e) = end {
        if e > m.capacity_rows {
            return Err(refuse(format!(
                "expanded mirror holds {} rows, live length needs {e}",
                m.capacity_rows
            )));
        }
    }
    Ok(())
}

fn check_cache(c: &LatentCache<'_>, rows: usize) -> Result<(), DispatchError> {
    if c.capacity_rows == 0 {
        return Err(refuse("latent cache capacity is zero".into()));
    }
    if let Some(p) = c.host_pos {
        if p.checked_add(rows).map_or(true, |e| e > c.capacity_rows) {
            return Err(refuse(format!(
                "latent cache overflow: pos {p} + {rows} rows > capacity {}",
                c.capacity_rows
            )));
        }
    }
    Ok(())
}

/// Resolve a [`RowRange`] to `(use_device_base, row0, count)`.
fn resolve_range(
    c: &LatentCache<'_>,
    range: RowRange,
) -> Result<(bool, usize, usize), DispatchError> {
    match range {
        RowRange::NewRows { count } => {
            check_cache(c, count)?;
            Ok((true, 0, count))
        }
        RowRange::Absolute { start, count } => {
            if start
                .checked_add(count)
                .map_or(true, |e| e > c.capacity_rows)
            {
                return Err(refuse(format!(
                    "row range {start}+{count} exceeds cache capacity {}",
                    c.capacity_rows
                )));
            }
            Ok((false, start, count))
        }
    }
}

/// Execute one latent-attention stage.
pub fn execute_latent_attention(
    gpu: &mut Gpu,
    op: &LatentAttentionOp<'_>,
) -> Result<(), DispatchError> {
    match op {
        LatentAttentionOp::NormRows {
            x,
            gamma,
            out,
            rows,
            width,
            eps,
        } => gpu
            .latent_attention_rms_rows(x, gamma, out, *rows, *width, *eps)
            .map_err(hip),
        LatentAttentionOp::SplitKv {
            geometry,
            cache,
            kv_a,
            gamma,
            rows,
            eps,
        } => {
            check_geometry(geometry)?;
            check_cache(cache, *rows)?;
            gpu.latent_attention_split_kv(
                kv_a,
                gamma,
                cache.rows,
                cache.pos,
                *rows,
                geometry.latent,
                geometry.tail,
                *eps,
                cache.capacity_rows,
            )
            .map_err(hip)
        }
        LatentAttentionOp::GatherLatent {
            geometry,
            cache,
            range,
            out,
        } => {
            check_geometry(geometry)?;
            let (dev_base, row0, count) = resolve_range(cache, *range)?;
            gpu.latent_attention_gather_latent(
                cache.rows,
                dev_base.then_some(cache.pos),
                out,
                row0,
                count,
                geometry.latent,
                geometry.tail,
                cache.capacity_rows,
            )
            .map_err(hip)
        }
        LatentAttentionOp::Decompress {
            geometry,
            cache,
            mirror,
            expansion,
            range,
        } => {
            check_geometry(geometry)?;
            let mirror_end = match range {
                RowRange::Absolute { start, count } => start.checked_add(*count),
                RowRange::NewRows { count } => cache.host_pos.map(|p| p + *count),
            };
            check_mirror(mirror, mirror_end)?;
            let (dev_base, row0, count) = resolve_range(cache, *range)?;
            let base = dev_base.then_some(cache.pos);
            if mirror.q8 && matches!(expansion, KvExpansion::Dense { .. }) {
                return Err(refuse(
                    "the dense expansion writes BF16 mirrors only; a Q8 mirror takes the external \
                     expansion"
                        .into(),
                ));
            }
            match expansion {
                KvExpansion::Dense { weight } => gpu
                    .latent_attention_decompress_dense(
                        cache.rows,
                        base,
                        weight,
                        mirror.knope,
                        mirror.vmir,
                        row0,
                        count,
                        geometry.n_heads,
                        geometry.nope,
                        geometry.v_dim,
                        geometry.latent,
                        geometry.tail,
                        cache.capacity_rows,
                        mirror.capacity_rows,
                    )
                    .map_err(hip),
                KvExpansion::External { projected } if mirror.q8 => gpu
                    .latent_attention_scatter_expanded_q8(
                        projected,
                        mirror.knope,
                        mirror.vmir,
                        base,
                        row0,
                        count,
                        geometry.n_heads,
                        geometry.nope,
                        geometry.v_dim,
                        mirror.capacity_rows,
                    )
                    .map_err(hip),
                KvExpansion::External { projected } => gpu
                    .latent_attention_scatter_expanded(
                        projected,
                        mirror.knope,
                        mirror.vmir,
                        base,
                        row0,
                        count,
                        geometry.n_heads,
                        geometry.nope,
                        geometry.v_dim,
                        mirror.capacity_rows,
                    )
                    .map_err(hip),
            }
        }
        LatentAttentionOp::Attention {
            geometry,
            cache,
            mirror,
            q,
            y,
            rows,
            scale,
            route,
        } => {
            check_geometry(geometry)?;
            check_cache(cache, *rows)?;
            attention(
                gpu,
                geometry,
                cache,
                mirror.as_ref(),
                q,
                y,
                *rows,
                *scale,
                route,
            )
        }
        LatentAttentionOp::Gate {
            y,
            g,
            out,
            elements,
        } => gpu.latent_attention_gate(y, g, out, *elements).map_err(hip),
    }
}

fn need_mirror<'m, 'a>(
    mirror: Option<&'m ExpandedMirror<'a>>,
    cache: &LatentCache<'_>,
    rows: usize,
    route: &str,
) -> Result<&'m ExpandedMirror<'a>, DispatchError> {
    let m = mirror.ok_or_else(|| {
        refuse(format!(
            "the {route} attention route needs the expanded mirror"
        ))
    })?;
    check_mirror(m, cache.host_pos.map(|p| p + rows))?;
    Ok(m)
}

/// Run the selected attention route for one stage.
#[allow(clippy::too_many_arguments)]
fn attention(
    gpu: &mut Gpu,
    g: &LatentGeometry,
    cache: &LatentCache<'_>,
    mirror: Option<&ExpandedMirror<'_>>,
    q: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    scale: f32,
    route: &AttentionRoute<'_>,
) -> Result<(), DispatchError> {
    let width = g.row_width();
    match route {
        AttentionRoute::Reference => {
            let m = need_mirror(mirror, cache, rows, "reference")?;
            if m.q8 {
                return Err(refuse(
                    "the reference attention route does not read a Q8 expanded mirror".into(),
                ));
            }
            gpu.latent_attention_serial(
                q,
                cache.rows,
                m.knope,
                m.vmir,
                y,
                cache.pos,
                rows,
                g.n_heads,
                g.nope,
                g.tail,
                g.v_dim,
                g.latent,
                cache.capacity_rows,
                m.capacity_rows,
                scale,
            )
            .map_err(hip)
        }
        AttentionRoute::Tiled { workspace } => {
            let m = need_mirror(mirror, cache, rows, "tiled")?;
            // BF16 mirror: element-unit views; Q8 mirror: byte-unit plane views.
            let (k_a, v) = if m.q8 {
                (
                    LatentRows {
                        buf: m.knope,
                        offset: 0,
                        tok_stride: q8_plane_row_bytes(g.n_heads, g.nope),
                        head_stride: g.nope,
                        width: g.nope,
                    },
                    LatentRows {
                        buf: m.vmir,
                        offset: 0,
                        tok_stride: q8_plane_row_bytes(g.n_heads, g.v_dim),
                        head_stride: g.v_dim,
                        width: g.v_dim,
                    },
                )
            } else {
                (
                    LatentRows {
                        buf: m.knope,
                        offset: 0,
                        tok_stride: g.n_heads * g.nope,
                        head_stride: g.nope,
                        width: g.nope,
                    },
                    LatentRows {
                        buf: m.vmir,
                        offset: 0,
                        tok_stride: g.n_heads * g.v_dim,
                        head_stride: g.v_dim,
                        width: g.v_dim,
                    },
                )
            };
            let a = TiledAttention {
                q,
                y,
                k_a,
                k_b: LatentRows {
                    buf: cache.rows,
                    offset: g.latent,
                    tok_stride: width,
                    head_stride: 0,
                    width: g.tail,
                },
                v,
                pos: cache.pos,
                pos_offset: 0,
                n_rows: rows,
                n_heads: g.n_heads,
                scale,
                max_rows: cache.capacity_rows.min(m.capacity_rows),
                round_y_bf16: true,
            };
            run_tiled(gpu, &a, *workspace, m.q8)
        }
        AttentionRoute::Absorbed { ops, workspace } => {
            let packed_ok = ops.packed.dtype == rdna_compute::DType::BF16
                && ops.packed.numel()
                    >= rdna_compute::latent_attention_absorb::latent_attention_absorb_packed_elems(
                        g.n_heads, g.nope, g.v_dim, g.latent,
                    );
            if !packed_ok {
                return Err(refuse(
                    "absorbed route: packed expansion weight is missing or undersized".into(),
                ));
            }
            gpu.latent_attention_absorb_q(
                q, ops.packed, ops.q_abs, rows, g.n_heads, g.nope, g.tail, g.v_dim, g.latent,
            )
            .map_err(hip)?;
            let a = TiledAttention {
                q: ops.q_abs,
                y: ops.z,
                k_a: LatentRows {
                    buf: cache.rows,
                    offset: 0,
                    tok_stride: width,
                    head_stride: 0,
                    width: g.latent,
                },
                k_b: LatentRows {
                    buf: cache.rows,
                    offset: g.latent,
                    tok_stride: width,
                    head_stride: 0,
                    width: g.tail,
                },
                v: LatentRows {
                    buf: cache.rows,
                    offset: 0,
                    tok_stride: width,
                    head_stride: 0,
                    width: g.latent,
                },
                pos: cache.pos,
                pos_offset: 0,
                n_rows: rows,
                n_heads: g.n_heads,
                scale,
                max_rows: cache.capacity_rows,
                round_y_bf16: false,
            };
            run_tiled(gpu, &a, *workspace, false)?;
            gpu.latent_attention_absorb_v(
                ops.z, ops.packed, y, rows, g.n_heads, g.nope, g.v_dim, g.latent, true,
            )
            .map_err(hip)
        }
    }
}

/// Prefill kernel for `rows > 1`, split-KV decode for `rows == 1`; `q8`: the
/// K_nope and V views are Q8 mirror planes.
fn run_tiled(
    gpu: &mut Gpu,
    a: &TiledAttention<'_>,
    workspace: Option<&GpuTensor>,
    q8: bool,
) -> Result<(), DispatchError> {
    if a.n_rows == 1 {
        let ws =
            workspace.ok_or_else(|| refuse("tiled decode needs the split-KV workspace".into()))?;
        if q8 {
            gpu.latent_attention_decode_split_q8(a, ws).map_err(hip)
        } else {
            gpu.latent_attention_decode_split(a, ws).map_err(hip)
        }
    } else if q8 {
        gpu.latent_attention_flash_prefill_q8(a).map_err(hip)
    } else {
        gpu.latent_attention_flash_prefill(a).map_err(hip)
    }
}
