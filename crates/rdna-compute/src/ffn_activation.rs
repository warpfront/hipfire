// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Generic FFN activation descriptor (SwiGLU, SiTU) kernels. Owner: slice F.
//!
//! The compute layer of a grouped MoE / dense FFN dispatch:
//!
//! * [`Gpu::ffn_activation_rows_f32`] and [`Gpu::ffn_activation_grouped_f32`]:
//!   gate/up activation (SwiGLU, clamped SwiGLU, SiTU-GLU) with optional BF16
//!   input/output rounding, zero padding of the physical columns and an
//!   optional AWQ divide (`kernels/src/ffn_situ.hip`,
//!   `kernels/src/ffn_grouped_activation_pad.hip`).
//! * [`Gpu::ffn_moe_router_topk_f32`]: softmax- or sigmoid+bias-routed top-k
//!   with renormalised weights (`kernels/src/ffn_router_renorm.hip`).
//! * [`Gpu::ffn_moe_combine_f32`] and [`Gpu::ffn_round_bf16_f32`]: the routed
//!   combine with an explicit rounding policy and an in-place BF16 rounding
//!   (`kernels/src/ffn_moe_combine_policy.hip`).
//!
//! Every launcher binds the thread, validates byte extents of its operands,
//! and launches through `launch_maybe_blob` with by-value kernarg copies, so
//! the launches are hipGraph-capture safe (no sync, no device-to-host copy).
//! The host references at the bottom of the module (`bf16_round_f32`,
//! `ffn_activation_scalar`, `router_select_reference`, `combine_reference`)
//! mirror the kernels' operation order.

use std::ffi::c_void;

use crate::dispatch::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult, KernargBlob};

const SITU_SRC: &str = include_str!("../../../kernels/src/ffn_situ.hip");
const GROUPED_SRC: &str = include_str!("../../../kernels/src/ffn_grouped_activation_pad.hip");
const ROUTER_SRC: &str = include_str!("../../../kernels/src/ffn_router_renorm.hip");
const COMBINE_SRC: &str = include_str!("../../../kernels/src/ffn_moe_combine_policy.hip");

const SITU_MODULE: &str = "ffn_situ";
const GROUPED_MODULE: &str = "ffn_grouped_activation_pad";
const GROUPED_ROTATE_I8_SRC: &str = crate::kernels::FFN_GROUPED_ACTIVATION_ROTATE_I8_SRC;
const GROUPED_ROTATE_I8_MODULE: &str = "ffn_grouped_activation_rotate_i8_gfx1201";
const ROUTER_MODULE: &str = "ffn_router_renorm";
const COMBINE_MODULE: &str = "ffn_moe_combine_policy";

const ENTRY_ROUTER: &str = "ffn_moe_router_topk_f32";
const ENTRY_COMBINE: &str = "ffn_moe_combine_f32";
const ENTRY_ROUND_BF16: &str = "ffn_round_bf16_f32";

/// Router kernel limits: LDS holds 1024 scores per row and eight picks.
pub const FFN_ROUTER_MAX_EXPERTS: usize = 1024;
pub const FFN_ROUTER_MAX_TOP_K: usize = 8;

/// Column tiles of 256 ride grid.y, which caps at 65535 blocks.
const MAX_COL_TILES: usize = 65535;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FfnActivation {
    /// silu(g) * u  == g/(1+exp(-g)) * u
    SwiGlu,
    /// DeepSeek-style asymmetric clamp then SwiGLU: g=min(g,limit);
    /// u=clamp(u,-limit,limit); silu(g)*u.
    SwiGluClamp { limit: f32 },
    /// SiTU-GLU: `[gate_beta*tanh(g/gate_beta)*sigmoid(g)] * [value_beta*tanh(u/value_beta)]`,
    /// evaluated as `((gb*tanhf(g/gb))*sigmoid(g)) * (vb*tanhf(u/vb))` with
    /// `sigmoid(g) = 1/(1+expf(-g))`.
    Situ { gate_beta: f32, value_beta: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RouterKind {
    /// softmax over logits, top-k on probs (tie -> lower expert id), weights =
    /// chosen / sum(chosen) (sequential sum from 0.0f in rank order, direct
    /// divide, no epsilon).
    SoftmaxRenorm,
    /// s=sigmoid(logit) (FP32); select top-k on s + bias[e] (bias optional;
    /// None == 0); weights = s[idx] / (sum(s[idx]) + eps) * scale, where the
    /// sum is sequential from 0.0f in rank (selection) order; selection order
    /// is descending biased score, tie -> LOWER expert id.
    SigmoidBiasRenorm { eps: f32, scale: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineOrder {
    Rank,
    ExpertIdAscending,
}

/// Per-launch AWQ divide applied to the activation result (after optional
/// BF16 output rounding and the zero pad), before the caller's FWHT rotate.
#[derive(Clone, Copy)]
pub enum FfnAwq<'a> {
    None,
    /// one F32 vector `[phys_n]` shared by every row
    Vector(&'a GpuTensor),
    /// `table`: `[n_experts]` u64 device pointer table, entry e -> `F32[phys_n]`
    /// for expert e; `row_expert`: i32 `[rows]`, expert of each row. Rows whose
    /// expert id is out of `[0, n_experts)` are written as zeros.
    PerExpert {
        table: &'a GpuTensor,
        row_expert: &'a GpuTensor,
        n_experts: usize,
    },
}

pub struct FfnActivationRows<'a> {
    pub activation: FfnActivation,
    /// F32; element offset of row 0 col 0 and elements per row.
    pub gate: &'a GpuTensor,
    pub gate_offset: usize,
    pub gate_ld: usize,
    pub up: &'a GpuTensor,
    pub up_offset: usize,
    pub up_ld: usize,
    /// F32 `[rows, phys_n]` row-major.
    pub out: &'a GpuTensor,
    pub rows: usize,
    /// activation columns computed
    pub logical_n: usize,
    /// `>= logical_n`; columns `logical_n..phys_n` are written as exactly +0.0
    /// (zero pad after the activation and output cast, before the AWQ divide).
    pub phys_n: usize,
    /// round g and u to BF16 (RNE) before the activation
    pub round_inputs: bool,
    /// round the activation result to BF16 (RNE) after computing in FP32
    pub round_output: bool,
    pub awq: FfnAwq<'a>,
}

pub struct FfnActivationGrouped<'a> {
    pub activation: FfnActivation,
    /// F32 `[m_total, 2*logical_n]`: gate in cols `0..logical_n`, up in cols
    /// `logical_n..2*logical_n`; row g is a grouped (expert-sorted) row.
    pub y_grouped: &'a GpuTensor,
    /// i32 `[m_total]`; flat slot (`token*top_k+rank`) of grouped row g, or -1
    /// for padding rows (skipped, nothing written).
    pub sorted_slot_index: &'a GpuTensor,
    /// i32 `[m_total / block_m]`; expert of tile `g/block_m` (only read when
    /// `awq_table` is Some).
    pub expert_tile_ids: &'a GpuTensor,
    /// F32 `[total_slots, phys_n]`: row = flat slot (un-scatter fused here),
    /// same pad / AWQ semantics as the rows kernel. `total_slots` is taken from
    /// the tensor (`shape[0]` for a `[total_slots, phys_n]` tensor, else the
    /// byte extent); a flat slot outside it is skipped.
    pub out: &'a GpuTensor,
    pub m_total: usize,
    pub block_m: usize,
    pub logical_n: usize,
    pub phys_n: usize,
    pub round_inputs: bool,
    pub round_output: bool,
    /// `[n_experts]` u64 ptr table -> `F32[phys_n]`
    pub awq_table: Option<&'a GpuTensor>,
    pub n_experts: usize,
}

/// Operands of [`Gpu::ffn_activation_grouped_rotate_i8_f32`]: [`FfnActivationGrouped`] with the F32
/// activation rows replaced by the A8 sidecar of their rotation (and an optional F32 copy of the
/// rotated rows).
pub struct FfnActivationGroupedI8<'a> {
    pub activation: FfnActivation,
    /// F32 `[m_total, 2*logical_n]`, as [`FfnActivationGrouped::y_grouped`].
    pub y_grouped: &'a GpuTensor,
    /// i32 `[m_total]`; flat slot of grouped row g, or -1 for padding rows (skipped).
    pub sorted_slot_index: &'a GpuTensor,
    /// i32 `[m_total / block_m]`; read only when `awq_table` is Some.
    pub expert_tile_ids: &'a GpuTensor,
    /// Optional F32 `[total_slots, phys_n]` receiving the rotated rows (what
    /// `rotate_x_mq_batched` would store); `None` when no F32 consumer remains.
    pub x_out: Option<&'a GpuTensor>,
    pub m_total: usize,
    pub block_m: usize,
    pub logical_n: usize,
    pub phys_n: usize,
    /// Flat slots (`rows * top_k`): the sidecar's batch extent; a flat slot outside it is skipped.
    pub total_slots: usize,
    pub round_inputs: bool,
    pub round_output: bool,
    /// `[n_experts]` u64 ptr table -> `F32[phys_n]`, as [`FfnActivationGrouped::awq_table`].
    pub awq_table: Option<&'a GpuTensor>,
    pub n_experts: usize,
}

/// Operands of [`Gpu::ffn_activation_rows_rotate_i8_f32`]: [`FfnActivationRows`] (no per-expert
/// AWQ) with the F32 activation rows replaced by the A8 sidecar of their rotation (and an optional
/// F32 copy of the rotated rows).
pub struct FfnActivationRowsI8<'a> {
    pub activation: FfnActivation,
    /// F32; element offset of row 0 col 0 and elements per row, as [`FfnActivationRows`].
    pub gate: &'a GpuTensor,
    pub gate_offset: usize,
    pub gate_ld: usize,
    pub up: &'a GpuTensor,
    pub up_offset: usize,
    pub up_ld: usize,
    /// Optional F32 `[rows, phys_n]` receiving the rotated rows; `None` when no F32 consumer
    /// remains.
    pub x_out: Option<&'a GpuTensor>,
    pub rows: usize,
    pub logical_n: usize,
    pub phys_n: usize,
    pub round_inputs: bool,
    pub round_output: bool,
    /// One F32 `[phys_n]` AWQ divide shared by every row ([`FfnAwq::Vector`]); `None` = no divide.
    pub awq: Option<&'a GpuTensor>,
}

pub struct MoeRouterTopK<'a> {
    pub kind: RouterKind,
    /// F32 `[rows, n_experts]` raw router linear output
    pub logits: &'a GpuTensor,
    /// F32 `[n_experts]`, selection only; ignored by `SoftmaxRenorm`
    pub correction_bias: Option<&'a GpuTensor>,
    /// i32 `[rows, top_k]` out
    pub topk_indices: &'a GpuTensor,
    /// F32 `[rows, top_k]` out
    pub topk_weights: &'a GpuTensor,
    pub rows: usize,
    /// `1 <= top_k <= FFN_ROUTER_MAX_TOP_K`, `top_k <= n_experts <= FFN_ROUTER_MAX_EXPERTS`
    pub n_experts: usize,
    pub top_k: usize,
}

pub struct MoeCombine<'a> {
    pub order: CombineOrder,
    /// BF16-round each expert row value before weighting
    pub round_expert_outputs: bool,
    /// BF16-round the accumulated routed sum (the "cast once")
    pub round_routed_sum: bool,
    /// BF16-round the (routed + shared) result
    pub round_final: bool,
    /// F32 `[n_down_rows, hidden]`
    pub down_rows: &'a GpuTensor,
    /// i32 `[rows*top_k]`: flat slot -> row of `down_rows` (<0 = unrouted,
    /// skipped); None => row = flat slot (expanded layout)
    pub row_of_slot: Option<&'a GpuTensor>,
    /// `[rows, top_k]`
    pub topk_indices: &'a GpuTensor,
    /// `[rows, top_k]`
    pub topk_weights: &'a GpuTensor,
    /// F32 `[rows, hidden]`, already cast per policy; added after the routed
    /// sum is rounded
    pub shared: Option<&'a GpuTensor>,
    /// optional diagnostic: routed sum after `round_routed_sum`, F32 `[rows, hidden]`
    pub routed_out: Option<&'a GpuTensor>,
    /// F32 `[rows, hidden]`
    pub out: &'a GpuTensor,
    pub rows: usize,
    pub hidden: usize,
    pub top_k: usize,
    pub n_experts: usize,
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn refuse(msg: &str) -> HipError {
    HipError::new(1, msg)
}

/// `v` as an i32 kernarg, or a named refusal.
fn as_i32(v: usize, what: &str) -> HipResult<i32> {
    i32::try_from(v).map_err(|_| refuse(&format!("{what}: {v} exceeds i32::MAX")))
}

/// A float tensor operand: F32, or raw bytes the caller typed as F32 words.
fn require_f32(t: &GpuTensor, what: &str) -> HipResult<()> {
    match t.dtype {
        DType::F32 | DType::Raw => Ok(()),
        _ => Err(refuse(&format!("{what}: tensor dtype must be F32"))),
    }
}

/// `t` holds at least `elems` 4-byte words.
fn require_words(t: &GpuTensor, elems: usize, what: &str) -> HipResult<()> {
    let need = elems
        .checked_mul(4)
        .ok_or_else(|| refuse(&format!("{what}: extent overflows usize")))?;
    if t.buf.size() < need {
        return Err(refuse(&format!(
            "{what}: buffer holds {} bytes, needs {need}",
            t.buf.size()
        )));
    }
    Ok(())
}

/// `rows * cols` without overflow.
fn mul(rows: usize, cols: usize, what: &str) -> HipResult<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| refuse(&format!("{what}: extent overflows usize")))
}

/// Elements covered by a strided view: `offset + (rows-1)*ld + cols`.
fn strided_extent(
    offset: usize,
    ld: usize,
    rows: usize,
    cols: usize,
    what: &str,
) -> HipResult<usize> {
    let span = mul(rows - 1, ld, what)?;
    offset
        .checked_add(span)
        .and_then(|v| v.checked_add(cols))
        .ok_or_else(|| refuse(&format!("{what}: extent overflows usize")))
}

/// Rows of width `width` a tensor holds: its leading dimension when it is a
/// `[rows, width]` tensor (bounded by the byte extent), else the byte extent.
fn rows_in(t: &GpuTensor, width: usize) -> usize {
    let by_bytes = t.buf.size() / (width * 4);
    if t.shape.len() == 2 && t.shape[1] == width {
        t.shape[0].min(by_bytes)
    } else {
        by_bytes
    }
}

/// Raw device pointer `elems` F32 words past the tensor base.
fn offset_ptr(t: &GpuTensor, elems: usize) -> *mut c_void {
    (t.buf.as_ptr() as *mut u8)
        .wrapping_add(elems * 4)
        .cast::<c_void>()
}

/// The two activation parameters (clamp limit; or gate/value betas), validated.
fn activation_params(act: FfnActivation) -> HipResult<(f32, f32)> {
    match act {
        FfnActivation::SwiGlu => Ok((0.0, 0.0)),
        FfnActivation::SwiGluClamp { limit } => {
            if !limit.is_finite() || limit < 0.0 {
                return Err(refuse(
                    "ffn activation: clamp limit must be finite and >= 0",
                ));
            }
            Ok((limit, 0.0))
        }
        FfnActivation::Situ {
            gate_beta,
            value_beta,
        } => {
            if !gate_beta.is_finite() || gate_beta == 0.0 {
                return Err(refuse(
                    "ffn activation: SiTU gate_beta must be finite, nonzero",
                ));
            }
            if !value_beta.is_finite() || value_beta == 0.0 {
                return Err(refuse(
                    "ffn activation: SiTU value_beta must be finite, nonzero",
                ));
            }
            Ok((gate_beta, value_beta))
        }
    }
}

fn rows_entry(act: FfnActivation) -> &'static str {
    match act {
        FfnActivation::SwiGlu => "ffn_swiglu_rows_f32",
        FfnActivation::SwiGluClamp { .. } => "ffn_swiglu_clamp_rows_f32",
        FfnActivation::Situ { .. } => "ffn_situ_rows_f32",
    }
}

fn grouped_entry(act: FfnActivation) -> &'static str {
    match act {
        FfnActivation::SwiGlu => "ffn_swiglu_grouped_f32",
        FfnActivation::SwiGluClamp { .. } => "ffn_swiglu_clamp_grouped_f32",
        FfnActivation::Situ { .. } => "ffn_situ_grouped_f32",
    }
}

fn grouped_rotate_i8_entry(act: FfnActivation) -> &'static str {
    match act {
        FfnActivation::SwiGlu => "ffn_swiglu_grouped_rotate_i8_gfx1201",
        FfnActivation::SwiGluClamp { .. } => "ffn_swiglu_clamp_grouped_rotate_i8_gfx1201",
        FfnActivation::Situ { .. } => "ffn_situ_grouped_rotate_i8_gfx1201",
    }
}

fn rows_rotate_i8_entry(act: FfnActivation) -> &'static str {
    match act {
        FfnActivation::SwiGlu => "ffn_swiglu_rows_rotate_i8_gfx1201",
        FfnActivation::SwiGluClamp { .. } => "ffn_swiglu_clamp_rows_rotate_i8_gfx1201",
        FfnActivation::Situ { .. } => "ffn_situ_rows_rotate_i8_gfx1201",
    }
}

impl Gpu {
    /// Gate/up activation over dense rows (`ffn_situ.hip`).
    ///
    /// `out[r, c] = awq_div(pad(cast_out(act(cast_in(g), cast_in(u)))))` for
    /// `c < logical_n`; columns `logical_n..phys_n` are +0.0 before the AWQ
    /// divide. Grid is `(rows, ceil(phys_n / 256))`: rows ride grid.x so row
    /// counts beyond 65535 launch.
    ///
    /// Kernarg order: `gate*, up*, out*, awq_a*, row_expert*, rows, logical_n,
    /// phys_n, gate_ld, up_ld, awq_mode, n_experts, round_inputs,
    /// round_output, p0, p1`.
    pub fn ffn_activation_rows_f32(&mut self, a: &FfnActivationRows<'_>) -> HipResult<()> {
        self.bind_thread()?;
        const WHAT: &str = "ffn_activation_rows_f32";
        let (p0, p1) = activation_params(a.activation)?;
        if a.logical_n > a.phys_n {
            return Err(refuse("ffn_activation_rows_f32: logical_n exceeds phys_n"));
        }
        if a.phys_n.div_ceil(256) > MAX_COL_TILES {
            return Err(refuse(
                "ffn_activation_rows_f32: phys_n exceeds 65535 column tiles",
            ));
        }
        let rows_i = as_i32(a.rows, "ffn_activation_rows_f32 rows")?;
        let logical_i = as_i32(a.logical_n, "ffn_activation_rows_f32 logical_n")?;
        let phys_i = as_i32(a.phys_n, "ffn_activation_rows_f32 phys_n")?;
        let gate_ld_i = as_i32(a.gate_ld, "ffn_activation_rows_f32 gate_ld")?;
        let up_ld_i = as_i32(a.up_ld, "ffn_activation_rows_f32 up_ld")?;
        require_f32(a.gate, "ffn_activation_rows_f32 gate")?;
        require_f32(a.up, "ffn_activation_rows_f32 up")?;
        require_f32(a.out, "ffn_activation_rows_f32 out")?;
        if a.rows == 0 || a.phys_n == 0 {
            return Ok(());
        }
        if a.logical_n > 0 {
            let g = strided_extent(a.gate_offset, a.gate_ld, a.rows, a.logical_n, WHAT)?;
            require_words(a.gate, g, "ffn_activation_rows_f32 gate extent")?;
            let u = strided_extent(a.up_offset, a.up_ld, a.rows, a.logical_n, WHAT)?;
            require_words(a.up, u, "ffn_activation_rows_f32 up extent")?;
        }
        require_words(
            a.out,
            mul(a.rows, a.phys_n, WHAT)?,
            "ffn_activation_rows_f32 out extent",
        )?;

        let (awq_mode, awq_ptr, row_expert_ptr, n_experts_i) = match a.awq {
            FfnAwq::None => (
                0i32,
                std::ptr::null_mut::<c_void>(),
                std::ptr::null_mut(),
                0i32,
            ),
            FfnAwq::Vector(v) => {
                require_f32(v, "ffn_activation_rows_f32 awq vector")?;
                require_words(v, a.phys_n, "ffn_activation_rows_f32 awq vector extent")?;
                (1, v.buf.as_ptr(), std::ptr::null_mut(), 0)
            }
            FfnAwq::PerExpert {
                table,
                row_expert,
                n_experts,
            } => {
                if n_experts == 0 {
                    return Err(refuse(
                        "ffn_activation_rows_f32: per-expert AWQ needs n_experts > 0",
                    ));
                }
                let ne = as_i32(n_experts, "ffn_activation_rows_f32 n_experts")?;
                require_words(
                    table,
                    mul(n_experts, 2, WHAT)?,
                    "ffn_activation_rows_f32 awq pointer table extent",
                )?;
                require_words(
                    row_expert,
                    a.rows,
                    "ffn_activation_rows_f32 row_expert extent",
                )?;
                (2, table.buf.as_ptr(), row_expert.buf.as_ptr(), ne)
            }
        };

        let entry = rows_entry(a.activation);
        self.ensure_kernel(SITU_MODULE, SITU_SRC, entry)?;
        let gp = offset_ptr(a.gate, a.gate_offset);
        let up = offset_ptr(a.up, a.up_offset);
        let op = a.out.buf.as_ptr();
        let ri = a.round_inputs as i32;
        let ro = a.round_output as i32;
        let mut params = [
            &gp as *const _ as *mut c_void,
            &up as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &awq_ptr as *const _ as *mut c_void,
            &row_expert_ptr as *const _ as *mut c_void,
            &rows_i as *const _ as *mut c_void,
            &logical_i as *const _ as *mut c_void,
            &phys_i as *const _ as *mut c_void,
            &gate_ld_i as *const _ as *mut c_void,
            &up_ld_i as *const _ as *mut c_void,
            &awq_mode as *const _ as *mut c_void,
            &n_experts_i as *const _ as *mut c_void,
            &ri as *const _ as *mut c_void,
            &ro as *const _ as *mut c_void,
            &p0 as *const _ as *mut c_void,
            &p1 as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            entry,
            [a.rows as u32, a.phys_n.div_ceil(256) as u32, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(gp);
                b.push_ptr(up);
                b.push_ptr(op);
                b.push_ptr(awq_ptr);
                b.push_ptr(row_expert_ptr);
                b.push_i32(rows_i);
                b.push_i32(logical_i);
                b.push_i32(phys_i);
                b.push_i32(gate_ld_i);
                b.push_i32(up_ld_i);
                b.push_i32(awq_mode);
                b.push_i32(n_experts_i);
                b.push_i32(ri);
                b.push_i32(ro);
                b.push_f32(p0);
                b.push_f32(p1);
                b
            },
        )
    }

    /// Grouped gate/up activation with the un-scatter fused in
    /// (`ffn_grouped_activation_pad.hip`). Grid `(m_total, ceil(phys_n/256))`.
    ///
    /// Kernarg order: `y_grouped*, sorted_slot_index*, expert_tile_ids*,
    /// awq_table*, out*, m_total, block_m, logical_n, phys_n, n_experts,
    /// total_slots, round_inputs, round_output, p0, p1`.
    pub fn ffn_activation_grouped_f32(&mut self, a: &FfnActivationGrouped<'_>) -> HipResult<()> {
        self.bind_thread()?;
        const WHAT: &str = "ffn_activation_grouped_f32";
        let (p0, p1) = activation_params(a.activation)?;
        if a.logical_n > a.phys_n {
            return Err(refuse(
                "ffn_activation_grouped_f32: logical_n exceeds phys_n",
            ));
        }
        if a.block_m == 0 {
            return Err(refuse("ffn_activation_grouped_f32: block_m must be > 0"));
        }
        if a.phys_n.div_ceil(256) > MAX_COL_TILES {
            return Err(refuse(
                "ffn_activation_grouped_f32: phys_n exceeds 65535 column tiles",
            ));
        }
        let m_total_i = as_i32(a.m_total, "ffn_activation_grouped_f32 m_total")?;
        let block_m_i = as_i32(a.block_m, "ffn_activation_grouped_f32 block_m")?;
        let logical_i = as_i32(a.logical_n, "ffn_activation_grouped_f32 logical_n")?;
        let phys_i = as_i32(a.phys_n, "ffn_activation_grouped_f32 phys_n")?;
        require_f32(a.y_grouped, "ffn_activation_grouped_f32 y_grouped")?;
        require_f32(a.out, "ffn_activation_grouped_f32 out")?;
        if a.m_total == 0 || a.phys_n == 0 {
            return Ok(());
        }
        if a.logical_n > 0 {
            let y = mul(a.m_total, mul(a.logical_n, 2, WHAT)?, WHAT)?;
            require_words(
                a.y_grouped,
                y,
                "ffn_activation_grouped_f32 y_grouped extent",
            )?;
        }
        require_words(
            a.sorted_slot_index,
            a.m_total,
            "ffn_activation_grouped_f32 sorted_slot_index extent",
        )?;
        require_words(a.out, a.phys_n, "ffn_activation_grouped_f32 out extent")?;
        let total_slots_i = rows_in(a.out, a.phys_n).min(i32::MAX as usize) as i32;

        let (table_ptr, n_experts_i) = match a.awq_table {
            None => (std::ptr::null_mut::<c_void>(), 0i32),
            Some(t) => {
                if a.n_experts == 0 {
                    return Err(refuse(
                        "ffn_activation_grouped_f32: awq_table needs n_experts > 0",
                    ));
                }
                let ne = as_i32(a.n_experts, "ffn_activation_grouped_f32 n_experts")?;
                require_words(
                    t,
                    mul(a.n_experts, 2, WHAT)?,
                    "ffn_activation_grouped_f32 awq pointer table extent",
                )?;
                require_words(
                    a.expert_tile_ids,
                    a.m_total.div_ceil(a.block_m),
                    "ffn_activation_grouped_f32 expert_tile_ids extent",
                )?;
                (t.buf.as_ptr(), ne)
            }
        };

        let entry = grouped_entry(a.activation);
        self.ensure_kernel(GROUPED_MODULE, GROUPED_SRC, entry)?;
        let yp = a.y_grouped.buf.as_ptr();
        let sp = a.sorted_slot_index.buf.as_ptr();
        let tp = a.expert_tile_ids.buf.as_ptr();
        let op = a.out.buf.as_ptr();
        let ri = a.round_inputs as i32;
        let ro = a.round_output as i32;
        let mut params = [
            &yp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &table_ptr as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &m_total_i as *const _ as *mut c_void,
            &block_m_i as *const _ as *mut c_void,
            &logical_i as *const _ as *mut c_void,
            &phys_i as *const _ as *mut c_void,
            &n_experts_i as *const _ as *mut c_void,
            &total_slots_i as *const _ as *mut c_void,
            &ri as *const _ as *mut c_void,
            &ro as *const _ as *mut c_void,
            &p0 as *const _ as *mut c_void,
            &p1 as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            entry,
            [a.m_total as u32, a.phys_n.div_ceil(256) as u32, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(yp);
                b.push_ptr(sp);
                b.push_ptr(tp);
                b.push_ptr(table_ptr);
                b.push_ptr(op);
                b.push_i32(m_total_i);
                b.push_i32(block_m_i);
                b.push_i32(logical_i);
                b.push_i32(phys_i);
                b.push_i32(n_experts_i);
                b.push_i32(total_slots_i);
                b.push_i32(ri);
                b.push_i32(ro);
                b.push_f32(p0);
                b.push_f32(p1);
                b
            },
        )
    }

    /// Grouped gate/up activation + FWHT rotation + A8 `block_i8_128` quantization in one launch
    /// (`ffn_grouped_activation_rotate_i8.gfx1201.hip`), exact gfx1201 only.
    ///
    /// The blocks written into `reservation` are byte-identical to
    /// [`Self::ffn_activation_grouped_f32`] into an F32 `[total_slots, phys_n]` tensor, then
    /// [`Self::rotate_x_mq_batched`], then [`Self::ensure_int8_mmq_x`] (`k = phys_n`,
    /// `n = total_slots`); `x_out = Some` additionally stores the rotated F32 rows the middle step
    /// leaves. Seals `reservation` after a successful launch; allocates nothing and never calls
    /// `ensure_int8_mmq_x`. Every `total_slots` row must be named by exactly one
    /// `sorted_slot_index` entry (the MoE scatter's permutation); an unnamed row's blocks are not
    /// written. Refusals (named, nothing launched): another arch, `logical_n % 8 != 0`,
    /// `phys_n % 256 != 0`, `logical_n > phys_n`, a reservation not `(phys_n, total_slots)`.
    pub fn ffn_activation_grouped_rotate_i8_f32(
        &mut self,
        a: &FfnActivationGroupedI8<'_>,
        reservation: crate::scratch::Int8MmqReservation,
    ) -> HipResult<crate::scratch::Int8MmqPrepared> {
        self.bind_thread()?;
        const WHAT: &str = "ffn_activation_grouped_rotate_i8_f32";
        if self.arch != "gfx1201" {
            return Err(refuse(
                "ffn_activation_grouped_rotate_i8_f32: exact gfx1201 only",
            ));
        }
        let (p0, p1) = activation_params(a.activation)?;
        if a.block_m == 0 {
            return Err(refuse(
                "ffn_activation_grouped_rotate_i8_f32: block_m must be > 0",
            ));
        }
        if a.logical_n == 0 || a.logical_n % 8 != 0 || a.logical_n > a.phys_n || a.phys_n % 256 != 0
        {
            return Err(refuse(
                "ffn_activation_grouped_rotate_i8_f32: needs 0 < logical_n <= phys_n, \
                 logical_n % 8 == 0 and phys_n % 256 == 0",
            ));
        }
        if reservation.k() != a.phys_n || reservation.n() != a.total_slots {
            return Err(refuse(
                "ffn_activation_grouped_rotate_i8_f32: reservation (k,n) mismatch",
            ));
        }
        if a.m_total == 0 || a.total_slots == 0 {
            return Err(refuse(
                "ffn_activation_grouped_rotate_i8_f32: m_total and total_slots must be > 0",
            ));
        }
        let groups = a.phys_n / 256;
        let blocks = mul(a.m_total, groups, WHAT)?;
        if blocks > u32::MAX as usize {
            return Err(refuse(
                "ffn_activation_grouped_rotate_i8_f32: grid exceeds u32::MAX blocks",
            ));
        }
        let m_total_i = as_i32(a.m_total, "ffn_activation_grouped_rotate_i8_f32 m_total")?;
        let block_m_i = as_i32(a.block_m, "ffn_activation_grouped_rotate_i8_f32 block_m")?;
        let logical_i = as_i32(
            a.logical_n,
            "ffn_activation_grouped_rotate_i8_f32 logical_n",
        )?;
        let phys_i = as_i32(a.phys_n, "ffn_activation_grouped_rotate_i8_f32 phys_n")?;
        let total_slots_i = as_i32(
            a.total_slots,
            "ffn_activation_grouped_rotate_i8_f32 total_slots",
        )?;
        require_f32(
            a.y_grouped,
            "ffn_activation_grouped_rotate_i8_f32 y_grouped",
        )?;
        require_words(
            a.y_grouped,
            mul(a.m_total, mul(a.logical_n, 2, WHAT)?, WHAT)?,
            "ffn_activation_grouped_rotate_i8_f32 y_grouped extent",
        )?;
        require_words(
            a.sorted_slot_index,
            a.m_total,
            "ffn_activation_grouped_rotate_i8_f32 sorted_slot_index extent",
        )?;
        if let Some(x) = a.x_out {
            require_f32(x, "ffn_activation_grouped_rotate_i8_f32 x_out")?;
            require_words(
                x,
                mul(a.total_slots, a.phys_n, WHAT)?,
                "ffn_activation_grouped_rotate_i8_f32 x_out extent",
            )?;
        }
        let (table_ptr, n_experts_i) = match a.awq_table {
            None => (std::ptr::null_mut::<c_void>(), 0i32),
            Some(t) => {
                if a.n_experts == 0 {
                    return Err(refuse(
                        "ffn_activation_grouped_rotate_i8_f32: awq_table needs n_experts > 0",
                    ));
                }
                let ne = as_i32(
                    a.n_experts,
                    "ffn_activation_grouped_rotate_i8_f32 n_experts",
                )?;
                require_words(
                    t,
                    mul(a.n_experts, 2, WHAT)?,
                    "ffn_activation_grouped_rotate_i8_f32 awq pointer table extent",
                )?;
                require_words(
                    a.expert_tile_ids,
                    a.m_total.div_ceil(a.block_m),
                    "ffn_activation_grouped_rotate_i8_f32 expert_tile_ids extent",
                )?;
                (t.buf.as_ptr(), ne)
            }
        };

        self.ensure_mq_signs()?;
        let entry = grouped_rotate_i8_entry(a.activation);
        self.ensure_kernel(GROUPED_ROTATE_I8_MODULE, GROUPED_ROTATE_I8_SRC, entry)?;
        let yp = a.y_grouped.buf.as_ptr();
        let sp = a.sorted_slot_index.buf.as_ptr();
        let tp = a.expert_tile_ids.buf.as_ptr();
        let xo = a
            .x_out
            .map_or(std::ptr::null_mut::<c_void>(), |t| t.buf.as_ptr());
        let s1 = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let i8p = reservation.ptr();
        let ri = a.round_inputs as i32;
        let ro = a.round_output as i32;
        let mut params = [
            &yp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &table_ptr as *const _ as *mut c_void,
            &xo as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &i8p as *const _ as *mut c_void,
            &m_total_i as *const _ as *mut c_void,
            &block_m_i as *const _ as *mut c_void,
            &logical_i as *const _ as *mut c_void,
            &phys_i as *const _ as *mut c_void,
            &n_experts_i as *const _ as *mut c_void,
            &total_slots_i as *const _ as *mut c_void,
            &ri as *const _ as *mut c_void,
            &ro as *const _ as *mut c_void,
            &p0 as *const _ as *mut c_void,
            &p1 as *const _ as *mut c_void,
        ];
        let result = self.launch_maybe_blob(
            entry,
            [blocks as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(yp);
                b.push_ptr(sp);
                b.push_ptr(tp);
                b.push_ptr(table_ptr);
                b.push_ptr(xo);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(i8p);
                b.push_i32(m_total_i);
                b.push_i32(block_m_i);
                b.push_i32(logical_i);
                b.push_i32(phys_i);
                b.push_i32(n_experts_i);
                b.push_i32(total_slots_i);
                b.push_i32(ri);
                b.push_i32(ro);
                b.push_f32(p0);
                b.push_f32(p1);
                b
            },
        );
        if !xo.is_null() {
            self.invalidate_x_caches_for(xo);
        }
        result?;
        Ok(crate::scratch::Int8MmqPrepared::from_reservation(
            reservation,
        ))
    }

    /// Dense-rows gate/up activation + FWHT rotation + A8 `block_i8_128` quantization in one launch
    /// (the rows entries of `ffn_grouped_activation_rotate_i8.gfx1201.hip`), exact gfx1201 only.
    ///
    /// Byte-identical to [`Self::ffn_activation_rows_f32`] (`FfnAwq::None` / `FfnAwq::Vector`) into
    /// an F32 `[rows, phys_n]` tensor, then [`Self::rotate_x_mq_batched`], then
    /// [`Self::ensure_int8_mmq_x`] (`k = phys_n`, `n = rows`); `x_out = Some` additionally stores
    /// the rotated F32 rows. Seals `reservation`; allocates nothing. Refusals (named, nothing
    /// launched): another arch, `logical_n % 8 != 0`, `phys_n % 256 != 0`, `logical_n > phys_n`,
    /// `gate_ld` / `up_ld` / offsets not multiples of 4, a reservation not `(phys_n, rows)`.
    pub fn ffn_activation_rows_rotate_i8_f32(
        &mut self,
        a: &FfnActivationRowsI8<'_>,
        reservation: crate::scratch::Int8MmqReservation,
    ) -> HipResult<crate::scratch::Int8MmqPrepared> {
        self.bind_thread()?;
        const WHAT: &str = "ffn_activation_rows_rotate_i8_f32";
        if self.arch != "gfx1201" {
            return Err(refuse(
                "ffn_activation_rows_rotate_i8_f32: exact gfx1201 only",
            ));
        }
        let (p0, p1) = activation_params(a.activation)?;
        if a.logical_n == 0 || a.logical_n % 8 != 0 || a.logical_n > a.phys_n || a.phys_n % 256 != 0
        {
            return Err(refuse(
                "ffn_activation_rows_rotate_i8_f32: needs 0 < logical_n <= phys_n, \
                 logical_n % 8 == 0 and phys_n % 256 == 0",
            ));
        }
        if a.gate_ld % 4 != 0 || a.up_ld % 4 != 0 || a.gate_offset % 4 != 0 || a.up_offset % 4 != 0
        {
            return Err(refuse(
                "ffn_activation_rows_rotate_i8_f32: gate/up ld and offset must be multiples of 4",
            ));
        }
        if a.rows == 0 {
            return Err(refuse(
                "ffn_activation_rows_rotate_i8_f32: rows must be > 0",
            ));
        }
        if reservation.k() != a.phys_n || reservation.n() != a.rows {
            return Err(refuse(
                "ffn_activation_rows_rotate_i8_f32: reservation (k,n) mismatch",
            ));
        }
        let blocks = mul(a.rows, a.phys_n / 256, WHAT)?;
        if blocks > u32::MAX as usize {
            return Err(refuse(
                "ffn_activation_rows_rotate_i8_f32: grid exceeds u32::MAX blocks",
            ));
        }
        let rows_i = as_i32(a.rows, "ffn_activation_rows_rotate_i8_f32 rows")?;
        let logical_i = as_i32(a.logical_n, "ffn_activation_rows_rotate_i8_f32 logical_n")?;
        let phys_i = as_i32(a.phys_n, "ffn_activation_rows_rotate_i8_f32 phys_n")?;
        let gate_ld_i = as_i32(a.gate_ld, "ffn_activation_rows_rotate_i8_f32 gate_ld")?;
        let up_ld_i = as_i32(a.up_ld, "ffn_activation_rows_rotate_i8_f32 up_ld")?;
        require_f32(a.gate, "ffn_activation_rows_rotate_i8_f32 gate")?;
        require_f32(a.up, "ffn_activation_rows_rotate_i8_f32 up")?;
        let g = strided_extent(a.gate_offset, a.gate_ld, a.rows, a.logical_n, WHAT)?;
        require_words(a.gate, g, "ffn_activation_rows_rotate_i8_f32 gate extent")?;
        let u = strided_extent(a.up_offset, a.up_ld, a.rows, a.logical_n, WHAT)?;
        require_words(a.up, u, "ffn_activation_rows_rotate_i8_f32 up extent")?;
        if let Some(x) = a.x_out {
            require_f32(x, "ffn_activation_rows_rotate_i8_f32 x_out")?;
            require_words(
                x,
                mul(a.rows, a.phys_n, WHAT)?,
                "ffn_activation_rows_rotate_i8_f32 x_out extent",
            )?;
        }
        let awq_ptr = match a.awq {
            None => std::ptr::null_mut::<c_void>(),
            Some(v) => {
                require_f32(v, "ffn_activation_rows_rotate_i8_f32 awq vector")?;
                require_words(
                    v,
                    a.phys_n,
                    "ffn_activation_rows_rotate_i8_f32 awq vector extent",
                )?;
                v.buf.as_ptr()
            }
        };

        self.ensure_mq_signs()?;
        let entry = rows_rotate_i8_entry(a.activation);
        self.ensure_kernel(GROUPED_ROTATE_I8_MODULE, GROUPED_ROTATE_I8_SRC, entry)?;
        let gp = offset_ptr(a.gate, a.gate_offset);
        let up = offset_ptr(a.up, a.up_offset);
        let xo = a
            .x_out
            .map_or(std::ptr::null_mut::<c_void>(), |t| t.buf.as_ptr());
        let s1 = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let i8p = reservation.ptr();
        let ri = a.round_inputs as i32;
        let ro = a.round_output as i32;
        let mut params = [
            &gp as *const _ as *mut c_void,
            &up as *const _ as *mut c_void,
            &xo as *const _ as *mut c_void,
            &awq_ptr as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &i8p as *const _ as *mut c_void,
            &rows_i as *const _ as *mut c_void,
            &logical_i as *const _ as *mut c_void,
            &phys_i as *const _ as *mut c_void,
            &gate_ld_i as *const _ as *mut c_void,
            &up_ld_i as *const _ as *mut c_void,
            &ri as *const _ as *mut c_void,
            &ro as *const _ as *mut c_void,
            &p0 as *const _ as *mut c_void,
            &p1 as *const _ as *mut c_void,
        ];
        let result = self.launch_maybe_blob(
            entry,
            [blocks as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(gp);
                b.push_ptr(up);
                b.push_ptr(xo);
                b.push_ptr(awq_ptr);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(i8p);
                b.push_i32(rows_i);
                b.push_i32(logical_i);
                b.push_i32(phys_i);
                b.push_i32(gate_ld_i);
                b.push_i32(up_ld_i);
                b.push_i32(ri);
                b.push_i32(ro);
                b.push_f32(p0);
                b.push_f32(p1);
                b
            },
        );
        if !xo.is_null() {
            self.invalidate_x_caches_for(xo);
        }
        result?;
        Ok(crate::scratch::Int8MmqPrepared::from_reservation(
            reservation,
        ))
    }

    /// Router top-k with renormalised weights (`ffn_router_renorm.hip`): one
    /// 256-thread block per row.
    ///
    /// Kernarg order: `logits*, bias*, idx_out*, w_out*, n_experts, top_k,
    /// kind, eps, scale` (kind 0 = softmax renorm, 1 = sigmoid + bias renorm).
    pub fn ffn_moe_router_topk_f32(&mut self, a: &MoeRouterTopK<'_>) -> HipResult<()> {
        self.bind_thread()?;
        const WHAT: &str = "ffn_moe_router_topk_f32";
        if a.top_k == 0 || a.top_k > FFN_ROUTER_MAX_TOP_K {
            return Err(refuse("ffn_moe_router_topk_f32: top_k outside 1..=8"));
        }
        if a.n_experts == 0 || a.n_experts > FFN_ROUTER_MAX_EXPERTS {
            return Err(refuse(
                "ffn_moe_router_topk_f32: n_experts outside 1..=1024",
            ));
        }
        if a.top_k > a.n_experts {
            return Err(refuse("ffn_moe_router_topk_f32: top_k exceeds n_experts"));
        }
        let rows_i = as_i32(a.rows, "ffn_moe_router_topk_f32 rows")?;
        require_f32(a.logits, "ffn_moe_router_topk_f32 logits")?;
        require_f32(a.topk_weights, "ffn_moe_router_topk_f32 topk_weights")?;
        if a.rows == 0 {
            return Ok(());
        }
        require_words(
            a.logits,
            mul(a.rows, a.n_experts, WHAT)?,
            "ffn_moe_router_topk_f32 logits extent",
        )?;
        let out_words = mul(a.rows, a.top_k, WHAT)?;
        require_words(
            a.topk_indices,
            out_words,
            "ffn_moe_router_topk_f32 topk_indices extent",
        )?;
        require_words(
            a.topk_weights,
            out_words,
            "ffn_moe_router_topk_f32 topk_weights extent",
        )?;

        let (kind, eps, scale, bias_ptr) = match a.kind {
            RouterKind::SoftmaxRenorm => (0i32, 0.0f32, 1.0f32, std::ptr::null_mut::<c_void>()),
            RouterKind::SigmoidBiasRenorm { eps, scale } => {
                let bp = match a.correction_bias {
                    None => std::ptr::null_mut(),
                    Some(b) => {
                        require_f32(b, "ffn_moe_router_topk_f32 correction_bias")?;
                        require_words(
                            b,
                            a.n_experts,
                            "ffn_moe_router_topk_f32 correction_bias extent",
                        )?;
                        b.buf.as_ptr()
                    }
                };
                (1, eps, scale, bp)
            }
        };

        self.ensure_kernel(ROUTER_MODULE, ROUTER_SRC, ENTRY_ROUTER)?;
        let lp = a.logits.buf.as_ptr();
        let ip = a.topk_indices.buf.as_ptr();
        let wp = a.topk_weights.buf.as_ptr();
        let ne = a.n_experts as i32;
        let tk = a.top_k as i32;
        let mut params = [
            &lp as *const _ as *mut c_void,
            &bias_ptr as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &ne as *const _ as *mut c_void,
            &tk as *const _ as *mut c_void,
            &kind as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
            &scale as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            ENTRY_ROUTER,
            [rows_i as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(lp);
                b.push_ptr(bias_ptr);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_i32(ne);
                b.push_i32(tk);
                b.push_i32(kind);
                b.push_f32(eps);
                b.push_f32(scale);
                b
            },
        )
    }

    /// Routed combine with an explicit rounding policy
    /// (`ffn_moe_combine_policy.hip`): one thread per output column, grid
    /// `(rows, ceil(hidden/256))`.
    ///
    /// Kernarg order: `down_rows*, row_of_slot*, topk_idx*, topk_w*, shared*,
    /// routed_out*, out*, rows, hidden, top_k, n_experts, order,
    /// round_expert_outputs, round_routed_sum, round_final, n_down_rows`
    /// (`n_down_rows`, derived from `down_rows`, bounds the gathered row).
    pub fn ffn_moe_combine_f32(&mut self, a: &MoeCombine<'_>) -> HipResult<()> {
        self.bind_thread()?;
        const WHAT: &str = "ffn_moe_combine_f32";
        if a.top_k == 0 || a.top_k > FFN_ROUTER_MAX_TOP_K {
            return Err(refuse("ffn_moe_combine_f32: top_k outside 1..=8"));
        }
        if a.n_experts == 0 {
            return Err(refuse("ffn_moe_combine_f32: n_experts must be > 0"));
        }
        if a.hidden.div_ceil(256) > MAX_COL_TILES {
            return Err(refuse(
                "ffn_moe_combine_f32: hidden exceeds 65535 column tiles",
            ));
        }
        let rows_i = as_i32(a.rows, "ffn_moe_combine_f32 rows")?;
        let hidden_i = as_i32(a.hidden, "ffn_moe_combine_f32 hidden")?;
        let ne_i = as_i32(a.n_experts, "ffn_moe_combine_f32 n_experts")?;
        for (t, what) in [
            (a.down_rows, "ffn_moe_combine_f32 down_rows"),
            (a.topk_weights, "ffn_moe_combine_f32 topk_weights"),
            (a.out, "ffn_moe_combine_f32 out"),
        ] {
            require_f32(t, what)?;
        }
        if a.rows == 0 || a.hidden == 0 {
            return Ok(());
        }
        let slots = mul(a.rows, a.top_k, WHAT)?;
        let out_words = mul(a.rows, a.hidden, WHAT)?;
        require_words(
            a.topk_indices,
            slots,
            "ffn_moe_combine_f32 topk_indices extent",
        )?;
        require_words(
            a.topk_weights,
            slots,
            "ffn_moe_combine_f32 topk_weights extent",
        )?;
        require_words(a.out, out_words, "ffn_moe_combine_f32 out extent")?;
        let n_down_rows = rows_in(a.down_rows, a.hidden);
        let row_ptr = match a.row_of_slot {
            Some(r) => {
                require_words(r, slots, "ffn_moe_combine_f32 row_of_slot extent")?;
                r.buf.as_ptr()
            }
            None => {
                if n_down_rows < slots {
                    return Err(refuse(
                        "ffn_moe_combine_f32: down_rows holds fewer than rows*top_k rows",
                    ));
                }
                std::ptr::null_mut()
            }
        };
        let shared_ptr = match a.shared {
            Some(s) => {
                require_f32(s, "ffn_moe_combine_f32 shared")?;
                require_words(s, out_words, "ffn_moe_combine_f32 shared extent")?;
                s.buf.as_ptr()
            }
            None => std::ptr::null_mut(),
        };
        let routed_ptr = match a.routed_out {
            Some(r) => {
                require_f32(r, "ffn_moe_combine_f32 routed_out")?;
                require_words(r, out_words, "ffn_moe_combine_f32 routed_out extent")?;
                r.buf.as_ptr()
            }
            None => std::ptr::null_mut(),
        };

        self.ensure_kernel(COMBINE_MODULE, COMBINE_SRC, ENTRY_COMBINE)?;
        let dp = a.down_rows.buf.as_ptr();
        let ip = a.topk_indices.buf.as_ptr();
        let wp = a.topk_weights.buf.as_ptr();
        let op = a.out.buf.as_ptr();
        let tk = a.top_k as i32;
        let order = match a.order {
            CombineOrder::Rank => 0i32,
            CombineOrder::ExpertIdAscending => 1,
        };
        let reo = a.round_expert_outputs as i32;
        let rrs = a.round_routed_sum as i32;
        let rf = a.round_final as i32;
        let ndr = n_down_rows.min(i32::MAX as usize) as i32;
        let mut params = [
            &dp as *const _ as *mut c_void,
            &row_ptr as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &shared_ptr as *const _ as *mut c_void,
            &routed_ptr as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &rows_i as *const _ as *mut c_void,
            &hidden_i as *const _ as *mut c_void,
            &tk as *const _ as *mut c_void,
            &ne_i as *const _ as *mut c_void,
            &order as *const _ as *mut c_void,
            &reo as *const _ as *mut c_void,
            &rrs as *const _ as *mut c_void,
            &rf as *const _ as *mut c_void,
            &ndr as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            ENTRY_COMBINE,
            [a.rows as u32, a.hidden.div_ceil(256) as u32, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(dp);
                b.push_ptr(row_ptr);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(shared_ptr);
                b.push_ptr(routed_ptr);
                b.push_ptr(op);
                b.push_i32(rows_i);
                b.push_i32(hidden_i);
                b.push_i32(tk);
                b.push_i32(ne_i);
                b.push_i32(order);
                b.push_i32(reo);
                b.push_i32(rrs);
                b.push_i32(rf);
                b.push_i32(ndr);
                b
            },
        )
    }

    /// In-place BF16 round-to-nearest-even of the first `n_elems` F32 elements
    /// of `x`. Grid `ceil(n_elems / 256)`, block 256.
    pub fn ffn_round_bf16_f32(&mut self, x: &GpuTensor, n_elems: usize) -> HipResult<()> {
        self.bind_thread()?;
        let n_i = as_i32(n_elems, "ffn_round_bf16_f32 n_elems")?;
        require_f32(x, "ffn_round_bf16_f32 x")?;
        require_words(x, n_elems, "ffn_round_bf16_f32 x extent")?;
        if n_elems == 0 {
            return Ok(());
        }
        self.ensure_kernel(COMBINE_MODULE, COMBINE_SRC, ENTRY_ROUND_BF16)?;
        let xp = x.buf.as_ptr();
        let mut params = [
            &xp as *const _ as *mut c_void,
            &n_i as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            ENTRY_ROUND_BF16,
            [n_elems.div_ceil(256) as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_i32(n_i);
                b
            },
        )
    }

    /// Fill the single-expert grouping tables of `tables`
    /// (`single_expert_table_words(rows)` F32/i32 words, layout in
    /// `ffn_grouped_activation_pad.hip`) for the three dense weights `w`.
    pub fn ffn_single_expert_tables(
        &mut self,
        tables: &GpuTensor,
        rows: usize,
        w: [&GpuTensor; 3],
    ) -> HipResult<()> {
        self.bind_thread()?;
        require_f32(tables, "ffn_single_expert_tables tables")?;
        require_words(
            tables,
            single_expert_table_words(rows),
            "ffn_single_expert_tables extent",
        )?;
        if rows == 0 {
            return Ok(());
        }
        let rows_pad = rows.div_ceil(16) * 16;
        let rows_i = as_i32(rows, "ffn_single_expert_tables rows")?;
        let pad_i = as_i32(rows_pad, "ffn_single_expert_tables rows_pad")?;
        self.ensure_kernel(GROUPED_MODULE, GROUPED_SRC, ENTRY_SINGLE_TABLES)?;
        let tp = tables.buf.as_ptr();
        let (w0, w1, w2) = (w[0].buf.as_ptr(), w[1].buf.as_ptr(), w[2].buf.as_ptr());
        let mut params = [
            &tp as *const _ as *mut c_void,
            &rows_i as *const _ as *mut c_void,
            &pad_i as *const _ as *mut c_void,
            &w0 as *const _ as *mut c_void,
            &w1 as *const _ as *mut c_void,
            &w2 as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            ENTRY_SINGLE_TABLES,
            [rows_pad.div_ceil(256).max(1) as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(tp);
                b.push_i32(rows_i);
                b.push_i32(pad_i);
                b.push_ptr(w0);
                b.push_ptr(w1);
                b.push_ptr(w2);
                b
            },
        )
    }

    /// Views of the single-expert tables of [`Self::ffn_single_expert_tables`]
    /// for `rows` rows: `(ptr tables [gate, up, down] one entry each, expert
    /// index [rows_pad], sorted_slot_index [rows_pad], expert_tile_ids
    /// [rows_pad / 16])`.
    pub fn single_expert_views(tables: &GpuTensor, rows: usize) -> SingleExpertViews {
        let rows_pad = rows.div_ceil(16) * 16;
        SingleExpertViews {
            gate_ptr: tables.sub_offset(0, 2),
            up_ptr: tables.sub_offset(2, 2),
            down_ptr: tables.sub_offset(4, 2),
            index: tables.sub_offset(6, rows_pad),
            sorted: tables.sub_offset(6 + rows_pad, rows_pad),
            tiles: tables.sub_offset(6 + 2 * rows_pad, rows_pad / 16),
            rows_pad,
        }
    }
}

const ENTRY_SINGLE_TABLES: &str = "ffn_single_expert_tables";

/// Words of the single-expert grouping tables for `rows` rows.
pub fn single_expert_table_words(rows: usize) -> usize {
    let rows_pad = rows.div_ceil(16) * 16;
    6 + 2 * rows_pad + rows_pad / 16
}

/// Borrow-free views into the tables of [`Gpu::ffn_single_expert_tables`].
pub struct SingleExpertViews {
    pub gate_ptr: GpuTensor,
    pub up_ptr: GpuTensor,
    pub down_ptr: GpuTensor,
    pub index: GpuTensor,
    pub sorted: GpuTensor,
    pub tiles: GpuTensor,
    pub rows_pad: usize,
}

// ---------------------------------------------------------------------------
// Host references (the kernels' operation order, on the CPU)
// ---------------------------------------------------------------------------

/// BF16 round-to-nearest-even of an F32 value, returned as F32. NaN and the
/// infinities pass through; a finite value past the BF16 range rounds to inf.
pub fn bf16_round_f32(x: f32) -> f32 {
    let u = x.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 {
        return x;
    }
    let r = u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(r)
}

/// One activation element, in the kernels' operation order: optional BF16
/// rounding of `g` and `u`, the activation in FP32, optional BF16 rounding of
/// the result.
pub fn ffn_activation_scalar(
    act: FfnActivation,
    g: f32,
    u: f32,
    round_inputs: bool,
    round_output: bool,
) -> f32 {
    let (mut g, mut u) = if round_inputs {
        (bf16_round_f32(g), bf16_round_f32(u))
    } else {
        (g, u)
    };
    let silu_mul = |g: f32, u: f32| g / (1.0 + (-g).exp()) * u;
    let r = match act {
        FfnActivation::SwiGlu => silu_mul(g, u),
        FfnActivation::SwiGluClamp { limit } => {
            if g > limit {
                g = limit;
            }
            if u > limit {
                u = limit;
            }
            if u < -limit {
                u = -limit;
            }
            silu_mul(g, u)
        }
        FfnActivation::Situ {
            gate_beta,
            value_beta,
        } => {
            let sig = 1.0 / (1.0 + (-g).exp());
            let gate = (gate_beta * (g / gate_beta).tanh()) * sig;
            let value = value_beta * (u / value_beta).tanh();
            gate * value
        }
    };
    if round_output {
        bf16_round_f32(r)
    } else {
        r
    }
}

/// Softmax probabilities in the kernel's reduction order: the max is exact; the
/// exponent sum is 256 per-thread strided partial sums folded by a tree
/// (stride 128, 64, ..., 1); every probability is a direct divide.
fn softmax_probs_block256(logits: &[f32]) -> Vec<f32> {
    let mut m = f32::NEG_INFINITY;
    for &v in logits {
        if v > m {
            m = v;
        }
    }
    let ex: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
    let mut part = [0.0f32; 256];
    for (t, p) in part.iter_mut().enumerate() {
        let mut e = t;
        while e < ex.len() {
            *p += ex[e];
            e += 256;
        }
    }
    let mut s = 128;
    while s > 0 {
        for t in 0..s {
            part[t] += part[t + s];
        }
        s >>= 1;
    }
    let total = part[0];
    ex.iter().map(|&v| v / total).collect()
}

/// Router top-k on the host, with the kernel's tie / order / sum semantics:
/// descending selection score, tie -> lower expert id, NaN or -inf selection
/// scores ranked lowest but still selectable, chosen sum sequential from 0.0
/// in selection order. Returns `(indices, weights)` of length `top_k`.
///
/// `exp` is the host libm; the device `expf` may differ in the last ulp.
///
/// # Panics
/// If `top_k > logits.len()` or a given `bias` is shorter than `logits`.
pub fn router_select_reference(
    kind: RouterKind,
    logits: &[f32],
    bias: Option<&[f32]>,
    top_k: usize,
) -> (Vec<i32>, Vec<f32>) {
    let n = logits.len();
    assert!(top_k <= n, "top_k exceeds n_experts");
    let (scores, mut sel): (Vec<f32>, Vec<f32>) = match kind {
        RouterKind::SoftmaxRenorm => {
            let p = softmax_probs_block256(logits);
            (p.clone(), p)
        }
        RouterKind::SigmoidBiasRenorm { .. } => {
            let s: Vec<f32> = logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
            let sel = match bias {
                Some(b) => {
                    assert!(b.len() >= n, "bias shorter than n_experts");
                    s.iter().zip(b).map(|(&sg, &bv)| sg + bv).collect()
                }
                None => s.clone(),
            };
            (s, sel)
        }
    };
    for v in sel.iter_mut() {
        if v.is_nan() || *v < f32::MIN {
            *v = f32::MIN;
        }
    }
    let mut idx = Vec::with_capacity(top_k);
    let mut chosen = Vec::with_capacity(top_k);
    for _ in 0..top_k {
        let mut best_v = f32::NEG_INFINITY;
        let mut best_i = usize::MAX;
        for (e, &v) in sel.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = e;
            }
        }
        idx.push(best_i as i32);
        chosen.push(scores[best_i]);
        sel[best_i] = f32::NEG_INFINITY;
    }
    let mut sum = 0.0f32;
    for &s in &chosen {
        sum += s;
    }
    let weights = match kind {
        RouterKind::SoftmaxRenorm => chosen.iter().map(|&s| s / sum).collect(),
        RouterKind::SigmoidBiasRenorm { eps, scale } => {
            let denom = sum + eps;
            chosen.iter().map(|&s| (s / denom) * scale).collect()
        }
    };
    (idx, weights)
}

/// Routed combine of ONE token on the host, in the kernel's operation order.
///
/// * `topk_indices` / `topk_weights`: the token's `top_k` expert ids and
///   weights, in rank order.
/// * `expert_rows[rank]`: the `hidden`-long down row of that rank's slot, or
///   `None` for an unrouted slot (skipped).
/// * `shared`: the token's already-cast shared row, added after the routed sum
///   is rounded.
///
/// Ranks whose expert id is outside `[0, n_experts)` are skipped; with
/// `ExpertIdAscending` they sort last and equal ids keep rank order. Each
/// contribution is a separately rounded multiply then add (no fma).
/// Returns `(routed, out)`: the routed sum after `round_routed_sum`, and the
/// final row after the shared add and `round_final`.
///
/// # Panics
/// If the three per-rank slices differ in length or a row is not `hidden` long.
#[allow(clippy::too_many_arguments)]
pub fn combine_reference(
    order: CombineOrder,
    round_expert_outputs: bool,
    round_routed_sum: bool,
    round_final: bool,
    n_experts: usize,
    hidden: usize,
    topk_indices: &[i32],
    topk_weights: &[f32],
    expert_rows: &[Option<&[f32]>],
    shared: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    let k = topk_indices.len();
    assert!(
        topk_weights.len() == k && expert_rows.len() == k,
        "per-rank slices must agree in length"
    );
    let valid = |e: i32| e >= 0 && (e as usize) < n_experts;
    let mut ord: Vec<usize> = (0..k).collect();
    if order == CombineOrder::ExpertIdAscending {
        // stable: equal keys keep rank order; invalid ids sort last
        ord.sort_by_key(|&j| {
            if valid(topk_indices[j]) {
                topk_indices[j]
            } else {
                i32::MAX
            }
        });
    }
    let mut routed = vec![0.0f32; hidden];
    for &j in &ord {
        if !valid(topk_indices[j]) {
            continue;
        }
        let Some(row) = expert_rows[j] else { continue };
        assert_eq!(row.len(), hidden, "expert row length");
        for (acc, &v) in routed.iter_mut().zip(row) {
            let v = if round_expert_outputs {
                bf16_round_f32(v)
            } else {
                v
            };
            let prod = v * topk_weights[j];
            *acc += prod;
        }
    }
    if round_routed_sum {
        for v in routed.iter_mut() {
            *v = bf16_round_f32(*v);
        }
    }
    let mut out = routed.clone();
    if let Some(s) = shared {
        assert_eq!(s.len(), hidden, "shared row length");
        for (o, &sv) in out.iter_mut().zip(s) {
            *o += sv;
        }
    }
    if round_final {
        for v in out.iter_mut() {
            *v = bf16_round_f32(*v);
        }
    }
    (routed, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(x: f32) -> u32 {
        x.to_bits()
    }

    #[test]
    fn bf16_rounds_ties_to_even() {
        // low half exactly 0x8000: tie; the even (bit 16 == 0) neighbour wins.
        assert_eq!(
            bits(bf16_round_f32(f32::from_bits(0x3f80_8000))),
            0x3f80_0000
        );
        assert_eq!(
            bits(bf16_round_f32(f32::from_bits(0x3f81_8000))),
            0x3f82_0000
        );
        // just past the tie rounds up, just below rounds down
        assert_eq!(
            bits(bf16_round_f32(f32::from_bits(0x3f80_8001))),
            0x3f81_0000
        );
        assert_eq!(
            bits(bf16_round_f32(f32::from_bits(0x3f80_7fff))),
            0x3f80_0000
        );
        // sign is symmetric
        assert_eq!(
            bits(bf16_round_f32(f32::from_bits(0xbf80_8000))),
            0xbf80_0000
        );
        assert_eq!(
            bits(bf16_round_f32(f32::from_bits(0xbf81_8000))),
            0xbf82_0000
        );
        // already representable values are unchanged
        assert_eq!(bits(bf16_round_f32(1.5)), bits(1.5));
        assert_eq!(bits(bf16_round_f32(-0.0)), bits(-0.0));
    }

    #[test]
    fn bf16_handles_inf_nan_overflow_subnormal() {
        assert_eq!(bf16_round_f32(f32::INFINITY), f32::INFINITY);
        assert_eq!(bf16_round_f32(f32::NEG_INFINITY), f32::NEG_INFINITY);
        assert!(bf16_round_f32(f32::NAN).is_nan());
        assert!(bf16_round_f32(f32::from_bits(0xffc0_0001)).is_nan());
        // the largest finite F32 rounds up to infinity
        assert_eq!(bf16_round_f32(f32::MAX), f32::INFINITY);
        assert_eq!(bf16_round_f32(f32::MIN), f32::NEG_INFINITY);
        // subnormals: tie to even (0), above the tie rounds up, tiny rounds to 0
        assert_eq!(bits(bf16_round_f32(f32::from_bits(0x0000_8000))), 0);
        assert_eq!(
            bits(bf16_round_f32(f32::from_bits(0x0001_8000))),
            0x0002_0000
        );
        assert_eq!(
            bits(bf16_round_f32(f32::from_bits(0x0000_8001))),
            0x0001_0000
        );
        assert_eq!(bits(bf16_round_f32(f32::from_bits(1))), 0);
    }

    #[test]
    fn situ_saturates_without_overflow() {
        let act = FfnActivation::Situ {
            gate_beta: 10.0,
            value_beta: 25.0,
        };
        for g in [80.0f32, -80.0, 0.0, 1e4, -1e4] {
            for u in [1e4f32, -1e4, 0.0] {
                let r = ffn_activation_scalar(act, g, u, false, false);
                assert!(r.is_finite(), "g={g} u={u} -> {r}");
            }
        }
        // g = +80: sigmoid = 1, gate = 10*tanh(8) ~ 10; u = +-1e4: value = +-25.
        let hi = ffn_activation_scalar(act, 80.0, 1e4, false, false);
        let lo = ffn_activation_scalar(act, 80.0, -1e4, false, false);
        assert!((hi - 250.0).abs() < 1e-2, "{hi}");
        assert!((lo + 250.0).abs() < 1e-2, "{lo}");
        // g = -80: sigmoid ~ 1.8e-35, result tiny but finite and non-negative for u < 0.
        let neg = ffn_activation_scalar(act, -80.0, -1e4, false, false);
        assert!(neg.is_finite() && neg >= 0.0 && neg < 1e-30, "{neg}");
    }

    #[test]
    fn swiglu_clamp_matches_unclamped_inside_limit() {
        let clamp = FfnActivation::SwiGluClamp { limit: 10.0 };
        for &g in &[-9.0f32, -3.5, 0.0, 0.25, 4.0, 9.9] {
            for &u in &[-9.9f32, -1.0, 0.0, 2.0, 9.0] {
                let a = ffn_activation_scalar(clamp, g, u, false, false);
                let b = ffn_activation_scalar(FfnActivation::SwiGlu, g, u, false, false);
                assert_eq!(bits(a), bits(b), "g={g} u={u}");
            }
        }
        // outside the limit: gate capped from above only, up capped both ways
        let capped = ffn_activation_scalar(clamp, 20.0, 30.0, false, false);
        let reference = ffn_activation_scalar(FfnActivation::SwiGlu, 10.0, 10.0, false, false);
        assert_eq!(bits(capped), bits(reference));
        let neg_up = ffn_activation_scalar(clamp, -20.0, -30.0, false, false);
        let reference = ffn_activation_scalar(FfnActivation::SwiGlu, -20.0, -10.0, false, false);
        assert_eq!(bits(neg_up), bits(reference));
    }

    #[test]
    fn activation_rounding_flags_apply_in_order() {
        let act = FfnActivation::SwiGlu;
        let (g, u) = (0.3337f32, -1.7771f32);
        let plain = ffn_activation_scalar(act, g, u, false, false);
        let in_r = ffn_activation_scalar(act, g, u, true, false);
        let both = ffn_activation_scalar(act, g, u, true, true);
        assert_eq!(
            bits(in_r),
            bits(ffn_activation_scalar(
                act,
                bf16_round_f32(g),
                bf16_round_f32(u),
                false,
                false
            ))
        );
        assert_eq!(bits(both), bits(bf16_round_f32(in_r)));
        assert_eq!(
            bits(ffn_activation_scalar(act, g, u, false, true)),
            bits(bf16_round_f32(plain))
        );
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[test]
    fn router_ties_go_to_lower_expert_id() {
        let logits = [0.5f32; 8];
        let (idx, w) = router_select_reference(RouterKind::SoftmaxRenorm, &logits, None, 3);
        assert_eq!(idx, vec![0, 1, 2]);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let kind = RouterKind::SigmoidBiasRenorm {
            eps: 1e-20,
            scale: 1.0,
        };
        let (idx, _) = router_select_reference(kind, &logits, Some(&[0.0; 8]), 3);
        assert_eq!(idx, vec![0, 1, 2]);
        // a tie that is not at the start: experts 1 and 4 share the top score
        let mut l = [0.0f32; 6];
        l[1] = 2.0;
        l[4] = 2.0;
        let (idx, _) = router_select_reference(kind, &l, None, 2);
        assert_eq!(idx, vec![1, 4]);
    }

    #[test]
    fn router_bias_changes_selection_but_not_weights() {
        let logits = [2.0f32, 1.0, 0.0, -1.0];
        let bias = [0.0f32, 0.0, 5.0, 0.0];
        let kind = RouterKind::SigmoidBiasRenorm {
            eps: 1e-20,
            scale: 2.5,
        };
        let (plain_idx, _) = router_select_reference(kind, &logits, None, 2);
        let (idx, w) = router_select_reference(kind, &logits, Some(&bias), 2);
        assert_eq!(plain_idx, vec![0, 1]);
        assert_eq!(idx, vec![2, 0]);
        // weights come from the unbiased scores, sum in rank order, eps 1e-20
        let (s2, s0) = (sigmoid(0.0), sigmoid(2.0));
        let sum = (0.0f32 + s2) + s0;
        assert_eq!(bits(w[0]), bits((s2 / (sum + 1e-20)) * 2.5));
        assert_eq!(bits(w[1]), bits((s0 / (sum + 1e-20)) * 2.5));
        assert!((w.iter().sum::<f32>() - 2.5).abs() < 1e-5);
    }

    #[test]
    fn softmax_renorm_weights_sum_to_one_in_descending_order() {
        let logits: Vec<f32> = (0..300)
            .map(|i| (((i * 37 + 11) % 101) as f32 - 50.0) * 0.07)
            .collect();
        let (idx, w) = router_select_reference(RouterKind::SoftmaxRenorm, &logits, None, 8);
        assert_eq!(idx.len(), 8);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        for pair in idx.windows(2) {
            assert!(logits[pair[0] as usize] >= logits[pair[1] as usize]);
        }
        for pair in w.windows(2) {
            assert!(pair[0] >= pair[1]);
        }
        let mut seen = idx.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 8);
    }

    #[test]
    fn router_selects_nan_and_neg_inf_last_but_distinct() {
        let logits = [f32::NAN, 1.0, f32::NEG_INFINITY, 0.5];
        let kind = RouterKind::SigmoidBiasRenorm {
            eps: 0.0,
            scale: 1.0,
        };
        let (idx, _) = router_select_reference(kind, &logits, None, 4);
        assert_eq!(&idx[..2], &[1, 3]);
        let mut rest = idx[2..].to_vec();
        rest.sort_unstable();
        assert_eq!(rest, vec![0, 2]);
    }

    #[test]
    fn combine_orders_are_equal_for_one_rank() {
        let row = [0.3f32, -1.25, 7.0];
        for order in [CombineOrder::Rank, CombineOrder::ExpertIdAscending] {
            let (routed, out) = combine_reference(
                order,
                true,
                true,
                true,
                16,
                3,
                &[5],
                &[0.7],
                &[Some(&row[..])],
                None,
            );
            let expect: Vec<f32> = row
                .iter()
                .map(|&v| bf16_round_f32(bf16_round_f32(v) * 0.7))
                .collect();
            assert_eq!(routed, expect);
            assert_eq!(out, expect);
        }
    }

    #[test]
    fn combine_ascending_order_differs_from_rank_order_in_the_last_bit() {
        // ranks hold [1.0, 1e8, -1e8] under ids [3, 1, 2]; ascending id visits
        // 1e8, -1e8, 1.0.
        let rows: [&[f32]; 3] = [&[1.0], &[1e8], &[-1e8]];
        let ids = [3, 1, 2];
        let w = [1.0f32; 3];
        let r: Vec<Option<&[f32]>> = rows.iter().map(|&s| Some(s)).collect();
        let (rank, _) = combine_reference(
            CombineOrder::Rank,
            false,
            false,
            false,
            8,
            1,
            &ids,
            &w,
            &r,
            None,
        );
        let (asc, _) = combine_reference(
            CombineOrder::ExpertIdAscending,
            false,
            false,
            false,
            8,
            1,
            &ids,
            &w,
            &r,
            None,
        );
        assert_eq!(rank, vec![0.0]);
        assert_eq!(asc, vec![1.0]);
    }

    #[test]
    fn combine_ascending_order_is_permutation_invariant() {
        let rows: [&[f32]; 4] = [&[0.1, 5.0], &[1e7, -3.0], &[-1e7, 0.25], &[0.3, 9.5]];
        let ids = [9i32, 4, 11, 2];
        let w = [0.4f32, 0.3, 0.2, 0.1];
        let base: Vec<Option<&[f32]>> = rows.iter().map(|&s| Some(s)).collect();
        let (expect, _) = combine_reference(
            CombineOrder::ExpertIdAscending,
            true,
            true,
            false,
            16,
            2,
            &ids,
            &w,
            &base,
            None,
        );
        let perm = [2usize, 0, 3, 1];
        let pids: Vec<i32> = perm.iter().map(|&p| ids[p]).collect();
        let pw: Vec<f32> = perm.iter().map(|&p| w[p]).collect();
        let prows: Vec<Option<&[f32]>> = perm.iter().map(|&p| Some(rows[p])).collect();
        let (got, _) = combine_reference(
            CombineOrder::ExpertIdAscending,
            true,
            true,
            false,
            16,
            2,
            &pids,
            &pw,
            &prows,
            None,
        );
        assert_eq!(
            expect.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn combine_skips_invalid_and_unrouted_ranks_and_adds_shared_after_rounding() {
        let a: &[f32] = &[2.0];
        let b: &[f32] = &[100.0];
        // rank 1 has an out-of-range id, rank 2 is unrouted (None)
        let (routed, out) = combine_reference(
            CombineOrder::ExpertIdAscending,
            false,
            true,
            true,
            8,
            1,
            &[1, 99, 2],
            &[0.5, 1.0, 1.0],
            &[Some(a), Some(b), None],
            Some(&[0.00390625f32]),
        );
        assert_eq!(routed, vec![1.0]);
        // 1.0 + 2^-8 is a BF16 tie that rounds to even (1.0)
        assert_eq!(out, vec![1.0]);
    }
}
