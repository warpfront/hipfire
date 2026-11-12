// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Absolute-position WY (C16) chunk-scan stream for Qwen4 on gfx11
//! (CHUNK-CANONICAL-DESIGN §3.3, §3.5, §4.3).
//!
//! The chunk kernel (`gated_delta_chunk_gate_wmma`, or its sealed PM twin: the
//! launches below reuse its kernarg ABI unchanged) scans 16-row tiles from an
//! F32 state and stores the F32 state at the end of its last tile. Called per
//! outer chunk it therefore (a) restarts its tile grid at every call start,
//! (b) ends on a short tile whose end state is not the state obtained by
//! completing that tile, and (c) used to requantize a Q8-family state at every
//! call end. This executor removes all three:
//!
//! * tile boundaries are absolute multiples of [`WY_TILE_ROWS`] (the first tile
//!   of a phase that begins off-grid is clipped to the next boundary);
//! * the state at the entry of the incomplete tile (`tile_entry`, F32) and the
//!   tile's post-convolution scan inputs (`pending_*`) are persistent carry. A
//!   later append replays the incomplete tile from them; the short tile's end
//!   state is transient (computed on a scratch copy);
//! * a Q8/Q8Ef state is quantized only at absolute [`GDN_COMMIT_ROWS`] seams (and
//!   at an explicit [`gdn_wy_stream_seal`]), and the F32 accumulator continues
//!   from the *reconstructed* state, exactly as a launch that begins at the
//!   seam would.
//!
//! No kernel is changed: every scan is the incumbent launch on a pointer-offset
//! row range, so the F32 results and the PM twin are the same machine code as
//! before. Partitioning the same stream into any sequence of appends is
//! byte-exact against one append of all its rows (oracle
//! `examples/gdn_wy_stream_oracle.rs`).
//!
//! Q8Ef is never treated as Q8: its seams write fresh codes, scales and the
//! real F16 residual (`gdn_state_f32_to_q8ef`) and reconstruct
//! `code * scale + residual` (`gdn_state_q8ef_to_f32`).

use super::{
    checked_i32, checked_product, checked_u32, copy_regions, ensure_f32, gdn_chunk_kernargs,
    gdn_state_from_f32, gdn_state_to_f32, validate_gdn_state, CopyRegion, GatedDeltaGateBatched,
    GatedDeltaStepBatched, GdnChunkKernargs, GdnChunkState, GdnStateFormat, TENSOR_OPS_SRC,
};
use crate::gdn_stream::{
    GdnStreamMeta, GdnStreamPhase, GDN_COMMIT_ROWS, GDN_STREAM_SCHEMA_VERSION,
};
use crate::prefill_invariant::{qwen4_gdn_prefill, Qwen4GdnPrefill};
use crate::{DType, Gpu, GpuTensor};
use hip_bridge::{DeviceBuffer, HipError, HipResult, KernargBlob};
use std::cell::Cell;
use std::ffi::c_void;

/// Rows of one WY tile (the chunk kernel's chunk length).
pub const WY_TILE_ROWS: usize = 16;

/// The persistent carry of one GDN layer's stream (borrowed).
#[derive(Clone, Copy)]
pub struct GdnWyCarry<'a> {
    /// F32 recurrent state `[value_heads, 128, 128]` at `meta.tile_base`; the
    /// authoritative state while the stream is [`GdnStreamPhase::Open`].
    pub tile_entry: &'a GpuTensor,
    /// BF16 convolution output `[WY_TILE_ROWS, qkv]` of the incomplete tile's
    /// pending rows (the scan input, exactly as the kernel read it).
    pub pending_conv: &'a GpuTensor,
    /// F32 `[2, WY_TILE_ROWS, value_heads]`: gate rows then beta rows of the
    /// pending rows.
    pub pending_params: &'a GpuTensor,
    /// Host stream cursor; updated by a successful append / seal.
    pub meta: &'a Cell<GdnStreamMeta>,
    /// Identity of the numerical recipe the stream was opened under.
    pub recipe_id: u64,
}

/// Shared per-stream staging for the output/Z rows of a replayed tile
/// (sequential layers reuse one).
#[derive(Clone, Copy)]
pub struct GdnWyEdge<'a> {
    /// F32 `[WY_TILE_ROWS, value_width]`.
    pub z: &'a GpuTensor,
    /// `[WY_TILE_ROWS, value_width]` elements of 4 bytes (holds BF16 or F32
    /// rows).
    pub out: &'a GpuTensor,
}

/// Bytes of the per-layer tile-entry carry, the pending scan inputs and the
/// shared edge staging for a head geometry (what the state allocates).
pub const fn wy_carry_bytes(
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> (usize, usize, usize, usize) {
    let qkv = (2 * key_heads * key_dim) + value_heads * value_dim;
    (
        4 * value_heads * key_dim * value_dim,
        2 * WY_TILE_ROWS * qkv,
        4 * 2 * WY_TILE_ROWS * value_heads,
        4 * WY_TILE_ROWS * value_heads * value_dim,
    )
}

/// Static (extent-free) admission of the WY stream: the same architecture,
/// flag and geometry conditions as `gated_delta_chunk_route`, minus the row
/// count and the recorder/capture state (which the executor checks itself).
pub fn wy_stream_admitted(
    gpu: &Gpu,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> bool {
    wy_stream_admitted_arch(&gpu.arch, key_heads, value_heads, key_dim, value_dim)
}

/// [`wy_stream_admitted`] for an architecture name: the byte accounting of an admission census
/// has no device handle but must make the same decision.
pub fn wy_stream_admitted_arch(
    arch: &str,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> bool {
    matches!(
        arch,
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" | "gfx1150" | "gfx1151" | "gfx1152"
    ) && qwen4_gdn_prefill() == Qwen4GdnPrefill::Wy
        && *crate::gemm::QWEN4_F16_WMMA
        && key_dim == 128
        && value_dim == 128
        && key_heads > 0
        && value_heads % key_heads == 0
}

/// Next absolute tile boundary strictly after `x`.
const fn tile_end(x: u64) -> u64 {
    (x / WY_TILE_ROWS as u64 + 1) * WY_TILE_ROWS as u64
}

/// Next absolute commit seam strictly after `x`.
const fn seam_end(x: u64) -> u64 {
    (x / GDN_COMMIT_ROWS as u64 + 1) * GDN_COMMIT_ROWS as u64
}

fn off(ptr: *mut c_void, bytes: usize) -> *mut c_void {
    (ptr as usize + bytes) as *mut c_void
}

fn view(ptr: *mut c_void, bytes: usize) -> DeviceBuffer {
    // SAFETY: a non-owning window over a live allocation the caller keeps
    // alive for the call; `DeviceBuffer::from_raw` never frees.
    unsafe { DeviceBuffer::from_raw(ptr, bytes) }
}

/// One append of `step.rows` rows at absolute `step.position` into a stream.
pub struct GdnWyAppend<'a> {
    /// `projection` is the BF16 convolution output of the call's rows;
    /// `gate`/`beta` its parameters; `state`/`format`/`residual` the live
    /// slot; `position` the absolute position of the first row.
    pub step: &'a GatedDeltaStepBatched<'a>,
    /// Z source and gated output of the call's rows (rows published: exactly
    /// the new rows, never replayed ones).
    pub gate: &'a GatedDeltaGateBatched<'a>,
    pub carry: GdnWyCarry<'a>,
    pub edge: &'a GdnWyEdge<'a>,
}

struct Ctx {
    format: GdnStateFormat,
    value_heads: usize,
    key_heads: usize,
    qkv: usize,
    output_bf16: bool,
    kernel: &'static str,
    query_scale: f32,
    norm: *mut c_void,
}

impl Ctx {
    fn out_elem(&self) -> usize {
        if self.output_bf16 {
            2
        } else {
            4
        }
    }
}

fn qk_norm(
    gpu: &mut Gpu,
    c: &Ctx,
    projection: *mut c_void,
    qn: *mut c_void,
    kn: *mut c_void,
    rows: usize,
) -> HipResult<()> {
    let mut args = KernargBlob::new();
    args.push_ptr(projection);
    args.push_ptr(qn);
    args.push_ptr(kn);
    args.push_i32(checked_i32(rows, "WY stream rows")?);
    args.push_i32(checked_i32(c.qkv, "WY stream qkv width")?);
    args.push_i32(checked_i32(c.key_heads, "WY stream key heads")?);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "gated_delta_qk_norm_bf16_batched",
        [checked_u32(rows, "WY stream row grid")?, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

#[allow(clippy::too_many_arguments)]
fn scan(
    gpu: &mut Gpu,
    c: &Ctx,
    projection: *mut c_void,
    qn: *mut c_void,
    kn: *mut c_void,
    gate: *mut c_void,
    beta: *mut c_void,
    state: *mut c_void,
    z: *mut c_void,
    output: *mut c_void,
    rows: usize,
) -> HipResult<()> {
    let mut args = gdn_chunk_kernargs(&GdnChunkKernargs {
        projection,
        qn,
        kn,
        gate,
        beta,
        state: GdnChunkState::F32 { state },
        z,
        norm: c.norm,
        output,
        rows: checked_i32(rows, "WY stream rows")?,
        qkv_width: checked_i32(c.qkv, "WY stream qkv width")?,
        key_heads: checked_i32(c.key_heads, "WY stream key heads")?,
        value_heads: checked_i32(c.value_heads, "WY stream value heads")?,
        query_scale: c.query_scale,
        output_bf16: i32::from(c.output_bf16),
    });
    args.pad_to(16);
    gpu.launch_blob_recorded(
        c.kernel,
        [checked_u32(c.value_heads, "WY stream head grid")?, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Copy `bytes` between raw device addresses (stream-ordered).
fn copy_raw(
    gpu: &mut Gpu,
    dst: *mut c_void,
    dst_bytes: usize,
    src: *mut c_void,
    src_bytes: usize,
    bytes: usize,
) -> HipResult<()> {
    if bytes == 0 {
        return Ok(());
    }
    let (dst, src) = (view(dst, dst_bytes), view(src, src_bytes));
    copy_regions(
        gpu,
        &[CopyRegion {
            dst: &dst,
            dst_offset: 0,
            src: &src,
            src_offset: 0,
            bytes,
        }],
    )
}

fn state_ptr(t: &GpuTensor) -> *mut c_void {
    t.buf.as_ptr()
}

fn wrong_shape() -> HipError {
    HipError::new(0, &super::ComputeError::WrongShape.to_string())
}

fn quantize_and_reconstruct(
    gpu: &mut Gpu,
    c: &Ctx,
    tile_entry: *mut c_void,
    slot: *mut c_void,
    residual: Option<*mut c_void>,
    seam: u64,
) -> HipResult<()> {
    // A seam `s` quantizes with the absolute position of its last row.
    let frame = checked_u32((seam - 1) as usize, "WY stream Q8 frame")?;
    gdn_state_from_f32(
        gpu,
        c.format,
        tile_entry,
        slot,
        residual,
        c.value_heads,
        frame,
    )?;
    gdn_state_to_f32(gpu, c.format, slot, residual, tile_entry, c.value_heads)
}

fn sealed_meta(prev: GdnStreamMeta, position: u64) -> GdnStreamMeta {
    GdnStreamMeta {
        schema_version: GDN_STREAM_SCHEMA_VERSION,
        recipe_id: prev.recipe_id,
        phase: GdnStreamPhase::Sealed,
        position,
        phase_start: position,
        frame_base: position,
        tile_base: position,
        pending_rows: 0,
        owner_generation: prev.owner_generation,
    }
}

fn validate_carry(
    carry: &GdnWyCarry<'_>,
    edge: &GdnWyEdge<'_>,
    value_heads: usize,
    qkv: usize,
    value: usize,
) -> HipResult<()> {
    let (state, conv, params, stage) = (
        value_heads * 128 * 128,
        WY_TILE_ROWS * qkv,
        2 * WY_TILE_ROWS * value_heads,
        WY_TILE_ROWS * value,
    );
    ensure_f32(carry.tile_entry)?;
    ensure_f32(carry.pending_params)?;
    ensure_f32(edge.z)?;
    if carry.tile_entry.numel() != state
        || carry.pending_conv.dtype != DType::BF16
        || carry.pending_conv.numel() < conv
        || carry.pending_params.numel() < params
        || edge.z.numel() < stage
        || edge.out.buf.size() < 4 * stage
    {
        return Err(wrong_shape());
    }
    Ok(())
}

fn validate_open_meta(meta: &GdnStreamMeta, carry: &GdnWyCarry<'_>, start: u64) -> HipResult<()> {
    if meta.schema_version != GDN_STREAM_SCHEMA_VERSION
        || meta.recipe_id != carry.recipe_id
        || meta.position != start
        || meta.tile_base > start
        || meta.tile_base + meta.pending_rows as u64 != start
        || meta.pending_rows as usize >= WY_TILE_ROWS
    {
        return Err(HipError::new(
            0,
            "GDN WY stream: append position, recipe or pending rows disagree with the stream cursor",
        ));
    }
    Ok(())
}

/// Append `step.rows` rows at absolute `step.position` to the stream and
/// publish their gated outputs; returns the new stream cursor (also stored in
/// `carry.meta`). A `Sealed` stream is opened first: its F32 state is the
/// reconstruction of the live slot (the F32 state itself, or `code * scale`
/// plus the real residual for Q8Ef).
///
/// Everything is validated before the first device write; a failure after
/// that point leaves the layer for the owner's transaction (restore the
/// ticket or reset), never for a retry on another recipe.
pub fn gdn_wy_stream_append(gpu: &mut Gpu, a: &GdnWyAppend<'_>) -> HipResult<GdnStreamMeta> {
    let (p, gate, carry, edge) = (a.step, a.gate, &a.carry, a.edge);
    if !wy_stream_admitted(gpu, p.key_heads, p.value_heads, p.key_dim, p.value_dim)
        || gpu.replay.is_recording()
        || gpu.graphs.capture_mode
        || p.projection.dtype != DType::BF16
        || p.row_states.is_some()
        || p.rows == 0
    {
        return Err(wrong_shape());
    }
    let format = p.format;
    validate_gdn_state(
        format,
        p.state,
        p.residual,
        None,
        1,
        p.value_heads,
        p.key_dim,
        p.value_dim,
    )?;
    for tensor in [p.gate, p.beta, gate.z] {
        ensure_f32(tensor)?;
    }
    let output_bf16 = gate.output.dtype == DType::BF16;
    if !output_bf16 {
        ensure_f32(gate.output)?;
    }
    let qk = checked_product(p.key_heads, p.key_dim, "WY stream qk extent")?;
    let value = checked_product(p.value_heads, p.value_dim, "WY stream value extent")?;
    let qkv = 2 * qk + value;
    let n = p.rows;
    if p.qkv_width != qkv
        || p.projection.numel() < n * qkv
        || p.gate.numel() < n * p.value_heads
        || p.beta.numel() < n * p.value_heads
        || gate.rows != n
        || gate.value_heads != p.value_heads
        || gate.value_dim != p.value_dim
        || gate.norm.dtype != DType::BF16
        || gate.norm.numel() != p.value_dim
        || gate.z.numel() < n * value
        || gate.output.numel() < n * value
    {
        return Err(wrong_shape());
    }
    validate_carry(carry, edge, p.value_heads, qkv, value)?;
    let start = p.position as u64;
    let end = start + n as u64;
    let carry_prev = carry.meta.get();
    let mut meta = carry_prev;
    match meta.phase {
        GdnStreamPhase::Open => validate_open_meta(&meta, carry, start)?,
        GdnStreamPhase::Sealed => {
            if meta.schema_version != GDN_STREAM_SCHEMA_VERSION {
                return Err(HipError::new(0, "GDN WY stream: schema mismatch"));
            }
        }
    }
    // Every module, scratch and extent is resolved before any state moves.
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "gated_delta_qk_norm_bf16_batched",
    )?;
    let kernel = gpu.gdn_chunk_ensure(GdnStateFormat::F32, false, false)?;
    // Scratch regions: [head tile Q|K][call Q|K] (BF16 each).
    let qn_head_elems = WY_TILE_ROWS * qk;
    let call_elems = n * qk;
    let scratch = gpu.qwen4_f16_x_scratch(2 * qn_head_elems + 2 * call_elems)?;
    let sp = scratch.buf.as_ptr();
    let at = |elems: usize| off(sp, elems * 2);
    let (qn_head, kn_head) = (at(0), at(qn_head_elems));
    let (qn_call, kn_call) = (at(2 * qn_head_elems), at(2 * qn_head_elems + call_elems));
    let f32_state_bytes = GdnStateFormat::F32.state_bytes(p.value_heads, p.key_dim, p.value_dim);
    let work = gpu.gdn_state_f32_scratch(f32_state_bytes)?;
    let c = Ctx {
        format,
        value_heads: p.value_heads,
        key_heads: p.key_heads,
        qkv,
        output_bf16,
        kernel,
        query_scale: (p.key_dim as f32).sqrt().recip(),
        norm: gate.norm.buf.as_ptr(),
    };
    let slot = p.state.buf.as_ptr();
    let residual = p.residual.map(|r| r.buf.as_ptr());
    let tile_entry = state_ptr(carry.tile_entry);
    let entry_bytes = carry.tile_entry.byte_size();

    // ---- open a sealed stream: F32 tile entry from the live slot ----
    if meta.phase == GdnStreamPhase::Sealed {
        match format {
            GdnStateFormat::F32 => {
                copy_raw(
                    gpu,
                    tile_entry,
                    entry_bytes,
                    slot,
                    f32_state_bytes,
                    f32_state_bytes,
                )?;
            }
            _ => gdn_state_to_f32(gpu, format, slot, residual, tile_entry, p.value_heads)?,
        }
        meta = GdnStreamMeta {
            schema_version: GDN_STREAM_SCHEMA_VERSION,
            recipe_id: carry.recipe_id,
            phase: GdnStreamPhase::Open,
            position: start,
            phase_start: start,
            frame_base: start,
            tile_base: start,
            pending_rows: 0,
            owner_generation: meta.owner_generation,
        };
    }

    let proj = p.projection.buf.as_ptr();
    let gate_p = p.gate.buf.as_ptr();
    let beta_p = p.beta.buf.as_ptr();
    let z_p = gate.z.buf.as_ptr();
    let out_p = gate.output.buf.as_ptr();
    let proj_row = qkv * 2;
    let hv_row = p.value_heads * 4;
    let z_row = value * 4;
    let out_row = value * c.out_elem();
    let pend_conv = state_ptr(carry.pending_conv);
    let pend_gate = state_ptr(carry.pending_params);
    let pend_beta = off(pend_gate, WY_TILE_ROWS * hv_row);
    let conv_bytes = carry.pending_conv.buf.size();
    let params_bytes = carry.pending_params.buf.size();
    let stage_z = state_ptr(edge.z);
    let stage_out = state_ptr(edge.out);
    let stage_z_bytes = edge.z.buf.size();
    let stage_out_bytes = edge.out.buf.size();

    // Append rows [r0, r0 + k) of the call to the pending buffers at row `at_row`.
    let append_pending = |gpu: &mut Gpu, at_row: usize, r0: usize, k: usize| -> HipResult<()> {
        copy_raw(
            gpu,
            off(pend_conv, at_row * proj_row),
            conv_bytes - at_row * proj_row,
            off(proj, r0 * proj_row),
            p.projection.buf.size() - r0 * proj_row,
            k * proj_row,
        )?;
        copy_raw(
            gpu,
            off(pend_gate, at_row * hv_row),
            params_bytes - at_row * hv_row,
            off(gate_p, r0 * hv_row),
            p.gate.buf.size() - r0 * hv_row,
            k * hv_row,
        )?;
        copy_raw(
            gpu,
            off(pend_beta, at_row * hv_row),
            params_bytes - (WY_TILE_ROWS + at_row) * hv_row,
            off(beta_p, r0 * hv_row),
            p.beta.buf.size() - r0 * hv_row,
            k * hv_row,
        )
    };

    let tile_base = meta.tile_base;
    let m = meta.pending_rows as usize;
    let mut x = start; // next absolute row to consume
    let mut frame_base = meta.frame_base;
    let mut cur_tile_base = tile_base;
    let mut cur_pending = m;

    // One normalization pass over the call's rows serves every direct launch.
    qk_norm(gpu, &c, proj, qn_call, kn_call, n)?;

    // ---- head: finish (or extend) the pending tile, staged ----
    if m > 0 {
        let t_end = tile_end(tile_base);
        let upto = t_end.min(end);
        let k = (upto - start) as usize;
        append_pending(gpu, m, 0, k)?;
        let rows = m + k;
        // Z of the replayed rows is never published; only rows `m..` are read back.
        copy_raw(
            gpu,
            off(stage_z, m * z_row),
            stage_z_bytes - m * z_row,
            z_p,
            gate.z.buf.size(),
            k * z_row,
        )?;
        qk_norm(gpu, &c, pend_conv, qn_head, kn_head, rows)?;
        let complete = upto == t_end;
        let state = if complete {
            tile_entry
        } else {
            copy_raw(
                gpu,
                work,
                f32_state_bytes,
                tile_entry,
                entry_bytes,
                f32_state_bytes,
            )?;
            work
        };
        scan(
            gpu, &c, pend_conv, qn_head, kn_head, pend_gate, pend_beta, state, stage_z, stage_out,
            rows,
        )?;
        copy_raw(
            gpu,
            out_p,
            gate.output.buf.size(),
            off(stage_out, m * out_row),
            stage_out_bytes - m * out_row,
            k * out_row,
        )?;
        x += k as u64;
        if complete {
            cur_tile_base = t_end;
            cur_pending = 0;
            if c.format != GdnStateFormat::F32 && t_end % GDN_COMMIT_ROWS as u64 == 0 {
                quantize_and_reconstruct_if_q8(gpu, &c, tile_entry, slot, residual, t_end)?;
                frame_base = t_end;
            }
        } else {
            cur_pending = rows;
        }
    } else if x % WY_TILE_ROWS as u64 != 0 {
        // Off-grid phase start: the first tile is clipped to the next boundary.
        let t_end = tile_end(x);
        let upto = t_end.min(end);
        let k = (upto - x) as usize;
        let r0 = (x - start) as usize;
        let complete = upto == t_end;
        let state = if complete {
            tile_entry
        } else {
            copy_raw(
                gpu,
                work,
                f32_state_bytes,
                tile_entry,
                entry_bytes,
                f32_state_bytes,
            )?;
            work
        };
        scan(
            gpu,
            &c,
            off(proj, r0 * proj_row),
            off(qn_call, r0 * qk * 2),
            off(kn_call, r0 * qk * 2),
            off(gate_p, r0 * hv_row),
            off(beta_p, r0 * hv_row),
            state,
            off(z_p, r0 * z_row),
            off(out_p, r0 * out_row),
            k,
        )?;
        if complete {
            cur_tile_base = t_end;
            cur_pending = 0;
            if c.format != GdnStateFormat::F32 && t_end % GDN_COMMIT_ROWS as u64 == 0 {
                quantize_and_reconstruct_if_q8(gpu, &c, tile_entry, slot, residual, t_end)?;
                frame_base = t_end;
            }
        } else {
            append_pending(gpu, 0, r0, k)?;
            cur_pending = k;
        }
        x = upto;
    }

    // ---- bulk: whole aligned tiles, split at absolute seams ----
    let bulk_end = end - end % WY_TILE_ROWS as u64;
    if cur_pending == 0 && x < bulk_end {
        let mut a_pos = x;
        while a_pos < bulk_end {
            let b_pos = seam_end(a_pos).min(bulk_end);
            let r0 = (a_pos - start) as usize;
            let k = (b_pos - a_pos) as usize;
            scan(
                gpu,
                &c,
                off(proj, r0 * proj_row),
                off(qn_call, r0 * qk * 2),
                off(kn_call, r0 * qk * 2),
                off(gate_p, r0 * hv_row),
                off(beta_p, r0 * hv_row),
                tile_entry,
                off(z_p, r0 * z_row),
                off(out_p, r0 * out_row),
                k,
            )?;
            if c.format != GdnStateFormat::F32 && b_pos % GDN_COMMIT_ROWS as u64 == 0 {
                quantize_and_reconstruct_if_q8(gpu, &c, tile_entry, slot, residual, b_pos)?;
                frame_base = b_pos;
            }
            a_pos = b_pos;
        }
        x = bulk_end;
        cur_tile_base = bulk_end;
    }

    // ---- tail: the incomplete tile, on a scratch copy; its inputs are carry ----
    if cur_pending == 0 && x < end {
        let r0 = (x - start) as usize;
        let k = (end - x) as usize;
        copy_raw(
            gpu,
            work,
            f32_state_bytes,
            tile_entry,
            entry_bytes,
            f32_state_bytes,
        )?;
        scan(
            gpu,
            &c,
            off(proj, r0 * proj_row),
            off(qn_call, r0 * qk * 2),
            off(kn_call, r0 * qk * 2),
            off(gate_p, r0 * hv_row),
            off(beta_p, r0 * hv_row),
            work,
            off(z_p, r0 * z_row),
            off(out_p, r0 * out_row),
            k,
        )?;
        append_pending(gpu, 0, r0, k)?;
        cur_tile_base = x;
        cur_pending = k;
    }

    let meta = GdnStreamMeta {
        position: end,
        frame_base,
        tile_base: cur_tile_base,
        pending_rows: cur_pending as u32,
        ..meta
    };
    debug_assert_eq!(
        meta,
        wy_advance_meta(
            carry_prev,
            carry.recipe_id,
            start,
            n as u64,
            format != GdnStateFormat::F32,
        )
    );
    carry.meta.set(meta);
    Ok(meta)
}

/// Pure image of the stream cursor after appending `rows` rows at absolute
/// `start` to a stream in state `prev` (a `Sealed` stream opens at `start`).
/// `quantized` is true for a Q8/Q8Ef state, whose commit seams move
/// `frame_base`; an F32 state has none. The executor stores exactly this
/// value; owners use it to describe a cut inside a window without running the
/// scan.
pub fn wy_advance_meta(
    prev: GdnStreamMeta,
    recipe_id: u64,
    start: u64,
    rows: u64,
    quantized: bool,
) -> GdnStreamMeta {
    let mut meta = match prev.phase {
        GdnStreamPhase::Open => prev,
        GdnStreamPhase::Sealed => GdnStreamMeta {
            schema_version: GDN_STREAM_SCHEMA_VERSION,
            recipe_id,
            phase: GdnStreamPhase::Open,
            position: start,
            phase_start: start,
            frame_base: start,
            tile_base: start,
            pending_rows: 0,
            owner_generation: prev.owner_generation,
        },
    };
    let end = start + rows;
    let tile = WY_TILE_ROWS as u64;
    let seam = GDN_COMMIT_ROWS as u64;
    let mut x = start;
    let mut tile_base = meta.tile_base;
    let mut pending = meta.pending_rows as u64;
    let mut frame_base = meta.frame_base;
    if pending > 0 || x % tile != 0 {
        let t_end = tile_end(tile_base);
        let upto = t_end.min(end);
        if upto == t_end {
            tile_base = t_end;
            pending = 0;
            if quantized && t_end % seam == 0 {
                frame_base = t_end;
            }
        } else {
            pending += upto - x;
        }
        x = upto;
    }
    let bulk_end = end - end % tile;
    if pending == 0 && x < bulk_end {
        let mut a = x;
        while a < bulk_end {
            let b = seam_end(a).min(bulk_end);
            if quantized && b % seam == 0 {
                frame_base = b;
            }
            a = b;
        }
        x = bulk_end;
        tile_base = bulk_end;
    }
    if pending == 0 && x < end {
        tile_base = x;
        pending = end - x;
    }
    meta.position = end;
    meta.frame_base = frame_base;
    meta.tile_base = tile_base;
    meta.pending_rows = pending as u32;
    meta
}

fn quantize_and_reconstruct_if_q8(
    gpu: &mut Gpu,
    c: &Ctx,
    tile_entry: *mut c_void,
    slot: *mut c_void,
    residual: Option<*mut c_void>,
    seam: u64,
) -> HipResult<()> {
    match c.format {
        // An F32 state has no artificial commit seam.
        GdnStateFormat::F32 => Ok(()),
        GdnStateFormat::Q8 | GdnStateFormat::Q8Ef => {
            quantize_and_reconstruct(gpu, c, tile_entry, slot, residual, seam)
        }
    }
}

/// Everything [`gdn_wy_stream_seal`] touches.
pub struct GdnWySeal<'a> {
    /// The live slot (Raw Q8 bytes, or the F32 state) and its format/residual.
    pub state: &'a GpuTensor,
    pub format: GdnStateFormat,
    pub residual: Option<&'a GpuTensor>,
    pub carry: GdnWyCarry<'a>,
    pub edge: &'a GdnWyEdge<'a>,
    pub key_heads: usize,
    pub value_heads: usize,
    /// The model's gated-norm weight (BF16 `[128]`); only its address is
    /// passed to the scan kernel, whose output here is discarded.
    pub norm: &'a GpuTensor,
}

/// Terminal fence of an ordinary-prefill phase (before decode/verify reads
/// the state): finish the incomplete tile's end state from `tile_entry` and the
/// pending rows, quantize it once at the absolute end (a Q8/Q8Ef state; the F32
/// state is stored), retain the live convolution and transition to `Sealed`.
/// A no-op on an already sealed stream. Never called for an outer chunk
/// boundary, checkpoint, restore or lane exchange.
pub fn gdn_wy_stream_seal(gpu: &mut Gpu, s: &GdnWySeal<'_>) -> HipResult<GdnStreamMeta> {
    let carry = &s.carry;
    let meta = carry.meta.get();
    if meta.phase == GdnStreamPhase::Sealed {
        return Ok(meta);
    }
    if gpu.replay.is_recording() || gpu.graphs.capture_mode {
        return Err(wrong_shape());
    }
    let (qk, value) = (s.key_heads * 128, s.value_heads * 128);
    let qkv = 2 * qk + value;
    validate_gdn_state(
        s.format,
        s.state,
        s.residual,
        None,
        1,
        s.value_heads,
        128,
        128,
    )?;
    validate_carry(carry, s.edge, s.value_heads, qkv, value)?;
    if meta.schema_version != GDN_STREAM_SCHEMA_VERSION
        || meta.recipe_id != carry.recipe_id
        || meta.tile_base + meta.pending_rows as u64 != meta.position
        || meta.pending_rows as usize >= WY_TILE_ROWS
    {
        return Err(HipError::new(
            0,
            "GDN WY stream seal: stream cursor disagrees with the carry",
        ));
    }
    let position = meta.position;
    let f32_state_bytes = GdnStateFormat::F32.state_bytes(s.value_heads, 128, 128);
    let tile_entry = state_ptr(carry.tile_entry);
    let entry_bytes = carry.tile_entry.byte_size();
    let slot = s.state.buf.as_ptr();
    let residual = s.residual.map(|r| r.buf.as_ptr());
    let r = meta.pending_rows as usize;
    // The slot is already the stream's state exactly when nothing was
    // consumed since it was last committed (open, or a seam).
    let slot_current = r == 0 && meta.tile_base == meta.frame_base;
    let mut final_state = tile_entry;
    if r > 0 {
        gpu.ensure_kernel_public(
            "tensor_ops",
            TENSOR_OPS_SRC,
            "gated_delta_qk_norm_bf16_batched",
        )?;
        let kernel = gpu.gdn_chunk_ensure(GdnStateFormat::F32, false, false)?;
        let scratch = gpu.qwen4_f16_x_scratch(2 * WY_TILE_ROWS * qk)?;
        let sp = scratch.buf.as_ptr();
        let work = gpu.gdn_state_f32_scratch(f32_state_bytes)?;
        let c = Ctx {
            format: s.format,
            value_heads: s.value_heads,
            key_heads: s.key_heads,
            qkv,
            output_bf16: false,
            kernel,
            query_scale: 128f32.sqrt().recip(),
            norm: s.norm.buf.as_ptr(),
        };
        let pend_conv = state_ptr(carry.pending_conv);
        let pend_gate = state_ptr(carry.pending_params);
        let pend_beta = off(pend_gate, WY_TILE_ROWS * s.value_heads * 4);
        let (qn, kn) = (sp, off(sp, WY_TILE_ROWS * qk * 2));
        copy_raw(
            gpu,
            work,
            f32_state_bytes,
            tile_entry,
            entry_bytes,
            f32_state_bytes,
        )?;
        qk_norm(gpu, &c, pend_conv, qn, kn, r)?;
        // Outputs/Z of the replayed rows are staging scratch: nothing reads them.
        scan(
            gpu,
            &c,
            pend_conv,
            qn,
            kn,
            pend_gate,
            pend_beta,
            work,
            state_ptr(s.edge.z),
            state_ptr(s.edge.out),
            r,
        )?;
        final_state = work;
    }
    if !slot_current {
        match s.format {
            GdnStateFormat::F32 => {
                copy_raw(
                    gpu,
                    slot,
                    f32_state_bytes,
                    final_state,
                    f32_state_bytes,
                    f32_state_bytes,
                )?;
            }
            GdnStateFormat::Q8 | GdnStateFormat::Q8Ef => {
                let frame = checked_u32((position - 1) as usize, "WY stream Q8 frame")?;
                gdn_state_from_f32(
                    gpu,
                    s.format,
                    final_state,
                    slot,
                    residual,
                    s.value_heads,
                    frame,
                )?;
            }
        }
    }
    let sealed = sealed_meta(meta, position);
    carry.meta.set(sealed);
    Ok(sealed)
}
