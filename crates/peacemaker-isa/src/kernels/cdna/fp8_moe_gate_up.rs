// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! FP8 × FP8 grouped MoE GEMM tile for gfx942 (MI300X, wave64) on
//! `v_mfma_f32_16x16x32_fp8_fp8`: the Qwen4 Flash-Next routed-expert gate/up
//! projection (and, with `x_row_div = 1`, the same tile for down) over qt=55
//! `FP8E4M3B128` expert weights and FNUZ activation codes from the FP8
//! activation producer (`kernels::cdna::fp8_act_producer`). Its CPU
//! reference is [`reference`]; its HIP twin is
//! `kernels/src/pm_cdna/fp8_moe_gate_up.gfx942.hip` (same ABI, layout and
//! math; compiled, never run here). Design: `pm-cdna3-ref/route/R2-prefill-moe.md`
//! "Proposed qt55 native-FP8 grouped tile contract"; GFX942-FP8-FN-PLAN §3.3
//! path C and §13.2.
//!
//! # ABI (kernargs, 72 bytes; [`KERNARG_OFFSETS`])
//!
//! | offset | name | |
//! |---|---|---|
//! | 0 | `W` | qt=55 expert blocks: expert `e` at `W + e*expert_stride`, OCP e4m3fn codes `u8[R, K]` row-major, then BF16 `s_inv[R/128, K/128]` (a multiplier) at `+R*K` |
//! | 8 | `expert_stride` | u64 (the PM metadata declares it as two by-value u32 halves, lo then hi: same bytes) |
//! | 16 | `tile_desc` | `i32x4[n_tiles]` `{expert, row0, nrows, 0}`; `expert < 0` or `expert >= n_experts` skips the tile |
//! | 24 | `sorted_rows` | `i32[n_sorted]`: flat slot or `-1` (padding) |
//! | 32 | `X` | FNUZ e4m3 codes `u8[x_rows, K]` from the producer (`cvt_pk_fp8_f32((x/a)*0.5)`, canonical `+0`, never `0x80`) |
//! | 40 | `x_scale` | `f32[x_rows]`: the producer's per-row power-of-two scale `a` |
//! | 48 | `Y` | `f32[n_sorted, R]` output |
//! | 56, 60, 64 | `R`, `K`, `x_row_div` | u32 |
//! | 68 | `n_experts` | u32: experts in `W`; a tile whose `expert` is outside `[0, n_experts)` is skipped (fail closed, before `W + expert*expert_stride` is formed) |
//!
//! This is the tile interface of the BF16-MFMA baseline launcher
//! (`gemm_fp8b128_moe_grouped_bf16mfma.gfx942.hip` in the gfx942-fp8 tree:
//! `tile_desc` from `gemm_fp8b128_moe_tile_desc_pad16`, the stable scatter's
//! `sorted_rows`, 16-row M tiles) with the activation argument replaced by
//! the producer's codes plus their row scales, so a route swaps the kernel
//! key and the activation buffers only. Gate/up: `R = 1280` (rows 0..639
//! gate, 640..1279 up, both in the one `Y` row as the unscatter expects),
//! `K = 2560`, `x_row_div = 10`; down: `R = 2560`, `K = 640`, `x_row_div = 1`.
//!
//! Launch: block 256 (four wave64), grid `[R/64, n_tiles, 1]`, no dynamic
//! LDS. Tile `y` covers sorted rows `row0 .. row0+nrows` (`nrows` 1..=16):
//! for every such row `i` the output row `Y[i]` columns `64*x .. 64*x+64`
//! are overwritten (`+0` for a `-1` row); nothing else is written (no
//! atomics). X row of sorted row `i` = `sorted_rows[i] / x_row_div`.
//!
//! Contract (not checked by the kernel; [`reference`] checks it): `K % 128 ==
//! 0`, `K >= 128`, `R % 64 == 0`, `expert_stride % 8 == 0`, `1 <= nrows <=
//! 16`, `0 <= sorted_rows[i] < 2^21` or `-1`, `1 <= x_row_div < 2^21`; weight
//! codes never the OCP NaNs `0x7f`/`0xff`; finite `s_inv`; activation codes
//! never `0x80`; `x_scale` finite powers of two (zero rows use `1`).
//!
//! # Numerical order (fixed for every row count and tile placement)
//!
//! Per output (slot `n`, column `o`) two F32 values, `total = +0` once and
//! `partial = +0` at each 128-K block. For `kb = 0 .. K/128` ascending: four
//! MFMAs `t = 0..3`, `partial = mfma(W[o, 128kb+32t ..+32], X[n, same], partial)`
//! (the first with `C = 0`), then `scale = (4.0 * a_row) * s_w[o/128, kb]`
//! and `total = fma(partial, scale, total)` (`v_fmac_f32`). No split-K.
//! `4·a_row` is exact (power of two); the FNUZ operands are `x/(2a)` and
//! `w/2`, so each MFMA product is `x·w/(4a)` exactly in F32.
//!
//! **Weight `0x80` (OCP −0) is handled in-kernel**, on the stored artifact
//! bytes: every weight dword is rewritten in registers before the MFMA with
//! `w & ((((w & 0x7f7f7f7f) + 0x7f7f7f7f) | 0x7f7f7f7f))`, which clears the
//! sign bit of exactly the bytes whose low seven bits are zero (`0x80 →
//! 0x00`, value-exact) and leaves every other byte unchanged; reinterpreted
//! as FNUZ, every remaining finite OCP byte `b` is `fn(b)/2`
//! (`docs/quant-formats/fp8-e4m3.md` "FNUZ fold"). No remapped weight copy
//! is required, and the BF16 teacher decoder (which keeps −0) is untouched.
//!
//! # Tile mapping (CDNA3 ISA §7.1.4 p.46-48)
//!
//! For `v_mfma_f32_16x16x32_fp8_fp8`, `K_L = 8`: lane `l` holds A item `i`
//! = `A[l%16, 8(l/16)+i]`, B item `i` = `B[8(l/16)+i, l%16]` (8 bytes, a
//! VGPR pair, little-endian), and D item `r` = `D[4(l/16)+r, l%16]`. As in
//! the BF16 baseline the multiplication is transposed: hardware A = weights
//! (16 output columns × 32 K), B = activations (32 K × 16 slots). Wave `w`
//! of workgroup `x` owns output columns `ob = 64x + 16w .. +16`; lane `l`
//! (`n = l%16`, `s = l/16`) loads its A pair for step `t` directly from
//! global memory, `W[e, ob+n, 128kb+32t+8s .. +8]`, and reads its B pair from
//! LDS, row `n`, bytes `32t+8s .. +8`; it stores `Y[row0+n, ob+4s .. +4]`
//! (one `global_store_dwordx4`). A 16-column wave and a 64-column workgroup
//! never straddle a 128-row scale block.
//!
//! LDS: two 128-K activation buffers `[2][16 rows][144 B]` (128 code bytes +
//! 16 B pad per row), 4608 B. Thread `tid` stages row `tid/16`, bytes
//! `8(tid%16) .. +8` of block `kb` into buffer `kb & 1` (rows past `nrows`
//! or with slot `-1` stage zeros); one barrier per block (the double buffer
//! makes the barrier of block `kb+1` also the WAR fence for block `kb`'s
//! buffer). Each block's operand loads (activation pair, four weight pairs,
//! the BF16 scale) are issued one block ahead into registers (the last block
//! reloads itself, keeping the loop's pending-load shape invariant).
//!
//! # Hardware assumptions
//!
//! - `[UNVERIFIED: MFMA accumulation order on MI300X]`
//!   The ISA gives only `D = C + Σ_k A·B`. The MFMA-internal sum is the one thing
//!   [`reference`] shares with the emulator: it calls
//!   `peacemaker_cdna::sim::mfma::accumulate` with the caller's
//!   `AccumModel` (the same model the emulator run uses; default
//!   `Cdna3Fused`). Everything else in the reference (OCP/FNUZ operand
//!   decode, the `0x80` remap, lane layout, block scale DAG) is written
//!   independently of the emulator.
//! - `[UNVERIFIED: confirm on MI300X]` FNUZ subnormal operands are honoured
//!   (not flushed) by the FP8 MFMA; the reference honours them.
//! - `[UNVERIFIED: confirm on MI300X]` LDS bank behaviour of the 144-byte
//!   row pitch, occupancy and issue spacing (performance only).
//! - The `x_row_div` division uses an `f32` reciprocal estimate followed by
//!   exact integer corrections, so its result does not depend on the
//!   precision of `v_rcp_f32` for the contracted ranges.
use crate::cdna::builder::Builder;
use crate::cdna::ops::{self as o};
use crate::cdna::Kernel;
use crate::Arch;
use peacemaker_cdna::enc::ds::DsOp;
use peacemaker_cdna::enc::flat::FlatOp;
use peacemaker_cdna::enc::smem::SmemOp;
use peacemaker_cdna::enc::sop::{Sop1Op, Sop2Op, SopcOp, SoppOp};
use peacemaker_cdna::enc::vop::{Vop1Op, Vop2Op, VopcOp};
use peacemaker_cdna::enc::vop3::Vop3OnlyOp;
use peacemaker_cdna::enc::vop3p::MaiOp;
use peacemaker_cdna::enc::vop3p::MfmaType;
use peacemaker_cdna::operand::InlineFloat;
use peacemaker_cdna::sim::mfma::{accumulate, AccumModel};
use peacemaker_cdna::Src;

/// PeaceMaker kernel symbol.
pub const SYMBOL: &str = "gemm_fp8b128_moe_grouped_fp8mfma_pm";
/// The HIP twin's symbol (same ABI).
pub const HIP_SYMBOL: &str = "gemm_fp8b128_moe_grouped_fp8mfma";
/// Workgroup size (four wave64).
pub const WORKGROUP: u32 = 256;
/// Output columns per workgroup.
pub const COLS_PER_WG: u32 = 64;
/// Sorted rows per M tile.
pub const TILE_ROWS: u32 = 16;
/// Kernarg byte offsets: W, expert_stride (lo, hi), tile_desc, sorted_rows,
/// X, x_scale, Y, R, K, x_row_div, n_experts.
pub const KERNARG_OFFSETS: [u32; 12] = [0, 8, 12, 16, 24, 32, 40, 48, 56, 60, 64, 68];
/// Kernarg segment bytes.
pub const KERNARG_BYTES: u32 = 72;
/// Bytes of one staged activation row in LDS (128 codes + 16 pad).
pub const LDS_ROW_BYTES: u32 = 144;
/// Bytes of one 128-K activation buffer.
pub const LDS_BUF_BYTES: u32 = TILE_ROWS * LDS_ROW_BYTES;

/// Host-side shape admission. R and K remain runtime arguments, so the
/// resulting code object also handles other admitted shapes unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub r: u32,
    pub k: u32,
}

/// The workgroup grid for `r` output columns and `n_tiles` tiles.
pub fn grid(r: u32, n_tiles: u32) -> [u32; 3] {
    [r / COLS_PER_WG, n_tiles, 1]
}

/// qt=55 expert block stride: `round_up(R*K + 2*ceil(R/128)*ceil(K/128), 256)`.
pub fn expert_stride(r: usize, k: usize) -> usize {
    (r * k + 2 * r.div_ceil(128) * k.div_ceil(128)).div_ceil(256) * 256
}

fn reg(src: Src, i: u8) -> Src {
    match src {
        Src::Sgpr(n) => Src::Sgpr(n + i),
        Src::Vgpr(n) => Src::Vgpr(n + i),
        Src::Agpr(n) => Src::Agpr(n + i),
        other => other,
    }
}

/// Registers live across the K loop.
struct Loop {
    w_base: Src,
    x_base: Src,
    exec_save: Src,
    m_stage: Src,
    vx: Src,
    wr: Src,
    vswr: Src,
    v_xoff: Src,
    v_woff: Src,
    v_swoff: Src,
}

/// Issue block `kb`'s operand loads (offsets already pointing at `kb`), in
/// the fixed order activation pair, weight pairs `t = 0..3`, scale.
fn issue_loads(b: &mut Builder, l: &Loop) -> Result<(), String> {
    b.push(o::s1(Sop1Op::AndSaveexecB64, l.exec_save, l.m_stage))?;
    b.push(o::global_load(
        FlatOp::LoadDwordx2,
        l.vx,
        l.v_xoff,
        Some(l.x_base),
        0,
    ))?;
    b.push(o::s1(Sop1Op::MovB64, Src::ExecLo, l.exec_save))?;
    for t in 0..4u8 {
        b.push(o::global_load(
            FlatOp::LoadDwordx2,
            reg(l.wr, 2 * t),
            l.v_woff,
            Some(l.w_base),
            32 * i16::from(t),
        ))?;
    }
    b.push(o::global_load(
        FlatOp::LoadUshort,
        l.vswr,
        l.v_swoff,
        Some(l.w_base),
        0,
    ))
}

/// `dst = num / div` for `num < 2^21`, `1 <= div < 2^21`: `f32` estimate
/// (within ±1 of the quotient for these ranges whatever the `v_rcp_f32`
/// rounding), then one exact downward and one exact upward correction.
fn udiv(
    b: &mut Builder,
    dst: Src,
    num: Src,
    sdiv: Src,
    v_rcp: Src,
    t: Src,
    t2: Src,
) -> Result<(), String> {
    b.push(o::v1(Vop1Op::CvtF32U32, t, num))?;
    b.push(o::v2(Vop2Op::MulF32, t, v_rcp, t))?;
    b.push(o::v1(Vop1Op::CvtU32F32, dst, t))?;
    b.push(o::v3(Vop3OnlyOp::MulLoU32, t, &[dst, sdiv]))?;
    b.push(o::v2(Vop2Op::SubU32, t, num, t))?;
    // r < 0: q -= 1, r += div.
    b.push(o::vc(VopcOp::CmpGtI32, Src::VccLo, Src::Int(0), t))?;
    b.push(o::v2(Vop2Op::AddU32, t2, Src::Int(-1), dst))?;
    b.push(o::v2ci(Vop2Op::CndmaskB32, dst, None, dst, t2, Src::VccLo))?;
    b.push(o::v2(Vop2Op::AddU32, t2, sdiv, t))?;
    b.push(o::v2ci(Vop2Op::CndmaskB32, t, None, t, t2, Src::VccLo))?;
    // r >= div: q += 1.
    b.push(o::vc(VopcOp::CmpLeU32, Src::VccLo, sdiv, t))?;
    b.push(o::v2(Vop2Op::AddU32, t2, Src::Int(1), dst))?;
    b.push(o::v2ci(Vop2Op::CndmaskB32, dst, None, dst, t2, Src::VccLo))
}

/// Admit `shape` and build the kernel for `arch` (gfx942). R and K remain
/// kernel arguments for the baseline grouped launcher: code is deliberately
/// identical for all admitted shapes and does not specialize the K order.
pub fn build(arch: Arch, shape: Shape) -> Result<Kernel, String> {
    if arch.cdna().is_none() {
        return Err(format!("{SYMBOL}: {} is not a CDNA target", arch.name()));
    }
    if shape.r == 0 || shape.r % COLS_PER_WG != 0 || shape.k == 0 || shape.k % 128 != 0 {
        return Err(format!("{SYMBOL}: R must be a positive multiple of 64 and K a positive multiple of 128: {shape:?}"));
    }
    let bytes = u64::from(shape.r) * u64::from(shape.k)
        + 2 * u64::from(shape.r.div_ceil(128)) * u64::from(shape.k / 128);
    if bytes > u64::from(u32::MAX) {
        return Err(format!(
            "{SYMBOL}: shape exceeds the 32-bit per-expert offset range: {shape:?}"
        ));
    }
    let mut b = Builder::new(arch, SYMBOL, WORKGROUP)?;
    let offsets = [
        b.arg_ptr("W", true)?,
        b.arg_u32("expert_stride_lo")?,
        b.arg_u32("expert_stride_hi")?,
        b.arg_ptr("tile_desc", true)?,
        b.arg_ptr("sorted_rows", true)?,
        b.arg_ptr("X", true)?,
        b.arg_ptr("x_scale", true)?,
        b.arg_ptr("Y", false)?,
        b.arg_u32("R")?,
        b.arg_u32("K")?,
        b.arg_u32("x_row_div")?,
        b.arg_u32("n_experts")?,
    ];
    if offsets != KERNARG_OFFSETS {
        return Err(format!(
            "{SYMBOL}: kernarg offsets {offsets:?} differ from the ABI {KERNARG_OFFSETS:?}"
        ));
    }
    let wg_x = b.workgroup_id(0)?;
    let wg_y = b.workgroup_id(1)?;
    let tid = b.workitem_ids(1)?;
    let lds = b.lds(2 * LDS_BUF_BYTES, 16)?;

    // ---- scalars
    let ka = b.sgprs(16, 4)?; // W, stride, tile_desc, sorted, X, x_scale, Y, R, K
    let (s_w, s_stride_lo, s_stride_hi, s_td, s_sorted, s_x, s_xs, s_y, s_r, s_k) = (
        ka,
        reg(ka, 2),
        reg(ka, 3),
        reg(ka, 4),
        reg(ka, 6),
        reg(ka, 8),
        reg(ka, 10),
        reg(ka, 12),
        reg(ka, 14),
        reg(ka, 15),
    );
    let s_div = b.sgprs(1, 1)?;
    let s_nexp = b.sgprs(1, 1)?;
    let td = b.sgprs(4, 4)?;
    let (s_expert, s_row0, s_nrows) = (td, reg(td, 1), reg(td, 2));
    let s_tda = b.sgprs(2, 2)?;
    let s_wexp = b.sgprs(2, 2)?;
    let s_t = b.sgprs(2, 2)?;
    let s_wave = b.sgprs(1, 1)?;
    let s_ob = b.sgprs(1, 1)?;
    let s_nkb = b.sgprs(1, 1)?;
    let s_kb = b.sgprs(1, 1)?;
    let s_buf = b.sgprs(1, 1)?;
    let s_step = b.sgprs(1, 1)?;
    let s_step2 = b.sgprs(1, 1)?;
    let s_c7f = b.sgprs(1, 1)?;
    let m_n = b.sgprs(2, 2)?;
    let m_slot = b.sgprs(2, 2)?;
    let m_xn = b.sgprs(2, 2)?;
    let m_xs = b.sgprs(2, 2)?;
    let exec_save = b.sgprs(2, 2)?;

    // ---- vectors
    let vn = b.vgprs(1, 1)?;
    let vs = b.vgprs(1, 1)?;
    let vslot = b.vgprs(1, 1)?;
    let vt = b.vgprs(1, 1)?;
    let vt2 = b.vgprs(1, 1)?;
    let vt3 = b.vgprs(1, 1)?;
    let v_flat_n = b.vgprs(1, 1)?;
    let v_flat_s = b.vgprs(1, 1)?;
    let v_rcp = b.vgprs(1, 1)?;
    let v_xrow_n = b.vgprs(1, 1)?;
    let v_xrow_s = b.vgprs(1, 1)?;
    let v_a = b.vgprs(1, 1)?;
    let v_fa = b.vgprs(1, 1)?;
    let v_lw = b.vgprs(1, 1)?;
    let v_lr = b.vgprs(1, 1)?;
    let v_addr = b.vgprs(1, 1)?;
    let v_xoff = b.vgprs(1, 1)?;
    let v_woff = b.vgprs(1, 1)?;
    let v_swoff = b.vgprs(1, 1)?;
    let vswr = b.vgprs(1, 1)?;
    let v_scale = b.vgprs(1, 1)?;
    let vx = b.vgprs(2, 2)?;
    let wr = b.vgprs(8, 2)?;
    let wa = b.vgprs(8, 2)?;
    let vb = b.vgprs(8, 2)?;
    let p = b.vgprs(4, 4)?;
    let total = b.vgprs(4, 4)?;

    // ---- kernargs and the tile descriptor
    let kernarg = b.kernarg_base();
    b.push(o::s_load(SmemOp::LoadDwordx16, ka, kernarg, 0))?;
    b.push(o::s_load(SmemOp::LoadDword, s_div, kernarg, 64))?;
    b.push(o::s_load(SmemOp::LoadDword, s_nexp, kernarg, 68))?;
    b.push(o::s2(Sop2Op::LshlB32, s_t, wg_y, Src::Int(4)))?;
    b.push(o::s2(Sop2Op::AddU32, s_tda, s_td, s_t))?;
    b.push(o::s2(
        Sop2Op::AddcU32,
        reg(s_tda, 1),
        reg(s_td, 1),
        Src::Int(0),
    ))?;
    b.push(o::s_load(SmemOp::LoadDwordx4, td, s_tda, 0))?;
    // Fail closed: one unsigned compare skips the -1 sentinel and every id >= n_experts.
    b.push(o::sc(SopcOp::CmpGeU32, s_expert, s_nexp))?;
    b.branch(SoppOp::CbranchScc1, "done")?;

    // Expert block base: W + expert * expert_stride (64-bit).
    b.push(o::s2(Sop2Op::MulI32, s_wexp, s_expert, s_stride_lo))?;
    b.push(o::s2(Sop2Op::MulHiU32, s_t, s_expert, s_stride_lo))?;
    b.push(o::s2(Sop2Op::MulI32, reg(s_t, 1), s_expert, s_stride_hi))?;
    b.push(o::s2(Sop2Op::AddU32, reg(s_wexp, 1), s_t, reg(s_t, 1)))?;
    b.push(o::s2(Sop2Op::AddU32, s_wexp, s_wexp, s_w))?;
    b.push(o::s2(
        Sop2Op::AddcU32,
        reg(s_wexp, 1),
        reg(s_wexp, 1),
        reg(s_w, 1),
    ))?;

    // Lane geometry: n = tid & 15, s = (tid & 63) >> 4, slot = tid >> 4, wave = tid >> 6.
    b.push(o::v2(Vop2Op::AndB32, vn, Src::Int(15), tid))?;
    b.push(o::v2(Vop2Op::AndB32, vs, Src::Int(63), tid))?;
    b.push(o::v2(Vop2Op::LshrrevB32, vs, Src::Int(4), vs))?;
    b.push(o::v2(Vop2Op::LshrrevB32, vslot, Src::Int(4), tid))?;
    b.push(o::v2(Vop2Op::LshrrevB32, vt, Src::Int(6), tid))?;
    b.push(o::v1(Vop1Op::ReadfirstlaneB32, s_wave, vt))?;
    // ob = 64 * wg_x + 16 * wave.
    b.push(o::s2(Sop2Op::LshlB32, s_ob, wg_x, Src::Int(6)))?;
    b.push(o::s2(Sop2Op::LshlB32, s_t, s_wave, Src::Int(4)))?;
    b.push(o::s2(Sop2Op::AddU32, s_ob, s_ob, s_t))?;
    b.push(o::s2(Sop2Op::LshrB32, s_nkb, s_k, Src::Int(7)))?;

    // Sorted-row lookups for the B column n and the staged row slot (rows
    // past nrows stay -1 and are never read).
    b.push(o::vc(VopcOp::CmpGtU32, m_n, s_nrows, vn))?;
    b.push(o::vc(VopcOp::CmpGtU32, m_slot, s_nrows, vslot))?;
    b.push(o::v1(Vop1Op::MovB32, v_flat_n, Src::Int(-1)))?;
    b.push(o::v1(Vop1Op::MovB32, v_flat_s, Src::Int(-1)))?;
    b.push(o::v2(Vop2Op::AddU32, vt, s_row0, vn))?;
    b.push(o::v2(Vop2Op::LshlrevB32, vt, Src::Int(2), vt))?;
    b.push(o::v2(Vop2Op::AddU32, vt2, s_row0, vslot))?;
    b.push(o::v2(Vop2Op::LshlrevB32, vt2, Src::Int(2), vt2))?;
    b.push(o::s1(Sop1Op::AndSaveexecB64, exec_save, m_n))?;
    b.push(o::global_load(
        FlatOp::LoadDword,
        v_flat_n,
        vt,
        Some(s_sorted),
        0,
    ))?;
    b.push(o::s1(Sop1Op::MovB64, Src::ExecLo, exec_save))?;
    b.push(o::s1(Sop1Op::AndSaveexecB64, exec_save, m_slot))?;
    b.push(o::global_load(
        FlatOp::LoadDword,
        v_flat_s,
        vt2,
        Some(s_sorted),
        0,
    ))?;
    b.push(o::s1(Sop1Op::MovB64, Src::ExecLo, exec_save))?;

    // X rows: flat / x_row_div where flat >= 0.
    b.push(o::v1(Vop1Op::CvtF32U32, v_rcp, s_div))?;
    b.push(o::v1(Vop1Op::RcpF32, v_rcp, v_rcp))?;
    b.push(o::vc(VopcOp::CmpLeI32, m_xn, Src::Int(0), v_flat_n))?;
    b.push(o::vc(VopcOp::CmpLeI32, m_xs, Src::Int(0), v_flat_s))?;
    udiv(&mut b, v_xrow_n, v_flat_n, s_div, v_rcp, vt, vt2)?;
    udiv(&mut b, v_xrow_s, v_flat_s, s_div, v_rcp, vt, vt2)?;

    // 4 * a_row for column n (1 for padding columns).
    b.push(o::v1(Vop1Op::MovB32, v_a, Src::Float(InlineFloat::One)))?;
    b.push(o::v2(Vop2Op::LshlrevB32, vt, Src::Int(2), v_xrow_n))?;
    b.push(o::s1(Sop1Op::AndSaveexecB64, exec_save, m_xn))?;
    b.push(o::global_load(FlatOp::LoadDword, v_a, vt, Some(s_xs), 0))?;
    b.push(o::s1(Sop1Op::MovB64, Src::ExecLo, exec_save))?;
    b.push(o::v2(
        Vop2Op::MulF32,
        v_fa,
        Src::Float(InlineFloat::Four),
        v_a,
    ))?;

    // Global byte offsets of block 0.
    b.push(o::v3(Vop3OnlyOp::MulLoU32, vt, &[v_xrow_s, s_k]))?;
    b.push(o::v3(
        Vop3OnlyOp::LshlAddU32,
        v_xoff,
        &[vn, Src::Int(3), vt],
    ))?;
    b.push(o::v2(Vop2Op::AddU32, vt, s_ob, vn))?;
    b.push(o::v3(Vop3OnlyOp::MulLoU32, vt, &[vt, s_k]))?;
    b.push(o::v3(
        Vop3OnlyOp::LshlAddU32,
        v_woff,
        &[vs, Src::Int(3), vt],
    ))?;
    // Scale offset: R*K + 2 * ((ob >> 7) * nkb).
    b.push(o::s2(Sop2Op::LshrB32, s_t, s_ob, Src::Int(7)))?;
    b.push(o::s2(Sop2Op::MulI32, s_t, s_t, s_nkb))?;
    b.push(o::s2(Sop2Op::LshlB32, s_t, s_t, Src::Int(1)))?;
    b.push(o::s2(Sop2Op::MulI32, reg(s_t, 1), s_r, s_k))?;
    b.push(o::s2(Sop2Op::AddU32, s_t, s_t, reg(s_t, 1)))?;
    b.push(o::v1(Vop1Op::MovB32, v_swoff, s_t))?;

    // LDS addresses: staging row slot bytes 8n; operand row n bytes 8s.
    b.push(o::v2(
        Vop2Op::MulU32U24,
        vt3,
        Src::Lit(LDS_ROW_BYTES),
        vslot,
    ))?;
    b.push(o::v3(Vop3OnlyOp::LshlAddU32, v_lw, &[vn, Src::Int(3), vt3]))?;
    b.push(o::v2(Vop2Op::MulU32U24, vt3, Src::Lit(LDS_ROW_BYTES), vn))?;
    b.push(o::v3(Vop3OnlyOp::LshlAddU32, v_lr, &[vs, Src::Int(3), vt3]))?;
    if lds.offset != 0 {
        b.push(o::v2(Vop2Op::AddU32, v_lw, Src::Lit(lds.offset), v_lw))?;
        b.push(o::v2(Vop2Op::AddU32, v_lr, Src::Lit(lds.offset), v_lr))?;
    }

    // Accumulators, zero activation pair for rows that stage nothing, loop state.
    for i in 0..4 {
        b.push(o::v1(Vop1Op::MovB32, reg(total, i), Src::Int(0)))?
    }
    for i in 0..2 {
        b.push(o::v1(Vop1Op::MovB32, reg(vx, i), Src::Int(0)))?
    }
    b.push(o::s1(Sop1Op::MovB32, s_kb, Src::Int(0)))?;
    b.push(o::s1(Sop1Op::MovB32, s_buf, Src::Int(0)))?;
    b.push(o::s1(Sop1Op::MovB32, s_c7f, Src::Lit(0x7f7f_7f7f)))?;

    let lp = Loop {
        w_base: s_wexp,
        x_base: s_x,
        exec_save,
        m_stage: m_xs,
        vx,
        wr,
        vswr,
        v_xoff,
        v_woff,
        v_swoff,
    };
    issue_loads(&mut b, &lp)?;

    // ---- K loop, one 128-K block per iteration
    b.label("kloop")?;
    b.push(o::v2(Vop2Op::AddU32, v_addr, s_buf, v_lw))?;
    b.push(o::ds_write(DsOp::WriteB64, v_addr, vx, 0))?;
    // Weight 0x80 -> 0x00 in registers (see the module doc).
    for i in 0..8 {
        let (w, d) = (reg(wr, i), reg(wa, i));
        b.push(o::v2(Vop2Op::AndB32, d, s_c7f, w))?;
        b.push(o::v2(Vop2Op::AddU32, d, s_c7f, d))?;
        b.push(o::v2(Vop2Op::OrB32, d, s_c7f, d))?;
        b.push(o::v2(Vop2Op::AndB32, d, d, w))?;
    }
    // scale = (4 * a_row) * f32(s_inv bf16).
    b.push(o::v2(Vop2Op::LshlrevB32, vt, Src::Int(16), vswr))?;
    b.push(o::v2(Vop2Op::MulF32, v_scale, v_fa, vt))?;
    // Next block's loads (the last block reloads itself).
    b.push(o::s2(Sop2Op::AddU32, s_kb, s_kb, Src::Int(1)))?;
    b.push(o::sc(SopcOp::CmpLtU32, s_kb, s_nkb))?;
    b.push(o::s2(
        Sop2Op::CselectB32,
        s_step,
        Src::Lit(128),
        Src::Int(0),
    ))?;
    b.push(o::s2(Sop2Op::CselectB32, s_step2, Src::Int(2), Src::Int(0)))?;
    b.push(o::v2(Vop2Op::AddU32, v_xoff, s_step, v_xoff))?;
    b.push(o::v2(Vop2Op::AddU32, v_woff, s_step, v_woff))?;
    b.push(o::v2(Vop2Op::AddU32, v_swoff, s_step2, v_swoff))?;
    issue_loads(&mut b, &lp)?;
    b.barrier()?;
    b.push(o::v2(Vop2Op::AddU32, v_addr, s_buf, v_lr))?;
    for t in 0..4u8 {
        b.push(o::ds_read(
            DsOp::ReadB64,
            reg(vb, 2 * t),
            v_addr,
            32 * u16::from(t),
        ))?
    }
    for t in 0..4u8 {
        let c = if t == 0 { Src::Int(0) } else { p };
        b.push(o::mfma(
            MaiOp::MfmaF32_16x16x32Fp8Fp8,
            p,
            reg(wa, 2 * t),
            reg(vb, 2 * t),
            c,
        ))?;
    }
    for i in 0..4 {
        b.push(o::v2(Vop2Op::FmacF32, reg(total, i), reg(p, i), v_scale))?
    }
    b.push(o::s2(Sop2Op::XorB32, s_buf, s_buf, Src::Lit(LDS_BUF_BYTES)))?;
    b.push(o::sc(SopcOp::CmpLtU32, s_kb, s_nkb))?;
    b.branch(SoppOp::CbranchScc1, "kloop")?;

    // ---- store Y[row0 + n, ob + 4s .. +4] for n < nrows
    // Padding is +0 even when a finite weight scale overflows the scale DAG.
    for i in 0..4 {
        b.push(o::v2ci(
            Vop2Op::CndmaskB32,
            reg(total, i),
            None,
            Src::Int(0),
            reg(total, i),
            m_xn,
        ))?;
    }
    b.push(o::v2(Vop2Op::AddU32, vt, s_row0, vn))?;
    b.push(o::v3(Vop3OnlyOp::MulLoU32, vt, &[vt, s_r]))?;
    b.push(o::v2(Vop2Op::AddU32, vt, s_ob, vt))?;
    b.push(o::v3(Vop3OnlyOp::LshlAddU32, vt, &[vs, Src::Int(2), vt]))?;
    b.push(o::v2(Vop2Op::LshlrevB32, vt, Src::Int(2), vt))?;
    b.push(o::s1(Sop1Op::AndSaveexecB64, exec_save, m_n))?;
    b.push(o::global_store(
        FlatOp::StoreDwordx4,
        vt,
        total,
        Some(s_y),
        0,
    ))?;
    b.label("done")?;
    b.finish()
}

// ----------------------------------------------------------------------------
// CPU reference

/// Inputs of one launch, in the kernel's memory layout.
#[derive(Clone, Copy, Debug)]
pub struct Problem<'a> {
    /// Expert blocks (`n_experts * expert_stride` bytes).
    pub w: &'a [u8],
    pub expert_stride: usize,
    pub tile_desc: &'a [[i32; 4]],
    pub sorted_rows: &'a [i32],
    /// FNUZ codes `[x_rows, K]`.
    pub x: &'a [u8],
    /// Row scales `[x_rows]`.
    pub x_scale: &'a [f32],
    pub r: usize,
    pub k: usize,
    pub x_row_div: usize,
    /// Experts in `w`; a tile whose expert is outside `[0, n_experts)` is skipped (nothing written), as in the kernel.
    pub n_experts: usize,
}

/// OCP e4m3fn code as the FNUZ operand the kernel feeds the MFMA: `fn(b)/2`,
/// with `0x80` (−0) → `+0` (the in-kernel remap). Decoded from the OCP
/// definition (bias 7, subnormal `m·2^-9`), not from the FNUZ table.
fn weight_operand(b: u8) -> Result<f32, String> {
    if b & 0x7f == 0x7f {
        return Err(format!("weight code {b:#04x} is an OCP NaN"));
    }
    if b == 0x80 {
        return Ok(0.0);
    }
    let (e, m) = (i32::from(b >> 3 & 15), f32::from(b & 7));
    let mag = if e == 0 {
        m * 2f32.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    Ok(if b & 0x80 != 0 { -mag } else { mag } / 2.0)
}

/// FNUZ e4m3 activation code (bias 8, subnormal `m·2^-10`, `0x80` NaN).
fn act_operand(b: u8) -> Result<f32, String> {
    if b == 0x80 {
        return Err("activation code 0x80 is FNUZ NaN".into());
    }
    let (e, m) = (i32::from(b >> 3 & 15), f32::from(b & 7));
    let mag = if e == 0 {
        m * 2f32.powi(-10)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 8)
    };
    Ok(if b & 0x80 != 0 { -mag } else { mag })
}

/// The kernel's result on the CPU: overwrites exactly the `Y` elements the
/// kernel writes (`y` is `[n_sorted, R]`), leaves the rest untouched.
/// Validates the ABI contract of the module doc.
pub fn reference(pb: &Problem, model: AccumModel, y: &mut [f32]) -> Result<(), String> {
    let (r, k) = (pb.r, pb.k);
    if k == 0 || k % 128 != 0 {
        return Err(format!("K {k} is not a positive multiple of 128"));
    }
    if r == 0 || r % 64 != 0 {
        return Err(format!("R {r} is not a positive multiple of 64"));
    }
    let nkb = k / 128;
    let block_bytes = r * k + 2 * r.div_ceil(128) * nkb;
    if pb.expert_stride % 8 != 0 || pb.expert_stride < block_bytes {
        return Err(format!(
            "expert_stride {} does not hold an R={r} K={k} qt=55 block",
            pb.expert_stride
        ));
    }
    if pb.x_row_div == 0 || pb.x_row_div >= 1 << 21 {
        return Err(format!("x_row_div {} outside 1..2^21", pb.x_row_div));
    }
    if y.len() != pb.sorted_rows.len() * r {
        return Err(format!(
            "Y has {} elements, expected {} x {r}",
            y.len(),
            pb.sorted_rows.len()
        ));
    }
    let x_rows = pb.x_scale.len();
    if pb.x.len() != x_rows * k {
        return Err(format!(
            "X has {} bytes, expected {x_rows} x {k}",
            pb.x.len()
        ));
    }
    for (i, &a) in pb.x_scale.iter().enumerate() {
        if !a.is_finite() || a <= 0.0 || a.to_bits() & 0x007f_ffff != 0 {
            return Err(format!("x_scale[{i}] = {a} is not a finite power of two"));
        }
    }
    // Operand tables (NaN codes → Err when used).
    let act: Vec<Result<f32, String>> = (0..=255u8).map(act_operand).collect();
    let wgt: Vec<Result<f32, String>> = (0..=255u8).map(weight_operand).collect();
    let (mut wa, mut xb) = ([0f32; 32], [0f32; 32]);
    for (tile, &[expert, row0, nrows, _]) in pb.tile_desc.iter().enumerate() {
        if expert < 0 || expert as usize >= pb.n_experts {
            continue;
        }
        if !(1..=16).contains(&nrows) || row0 < 0 {
            return Err(format!("tile {tile}: row0 {row0} nrows {nrows}"));
        }
        let base = expert as usize * pb.expert_stride;
        let blk =
            pb.w.get(base..base + block_bytes)
                .ok_or_else(|| format!("tile {tile}: expert {expert} outside W"))?;
        let (codes, scales) = blk.split_at(r * k);
        for n in 0..nrows as usize {
            let i = row0 as usize + n;
            let flat = *pb
                .sorted_rows
                .get(i)
                .ok_or_else(|| format!("tile {tile}: sorted row {i} outside sorted_rows"))?;
            let (xrow, a) = if flat >= 0 {
                if flat >= 1 << 21 {
                    return Err(format!("sorted_rows[{i}] = {flat} >= 2^21"));
                }
                let xr = flat as usize / pb.x_row_div;
                if xr >= x_rows {
                    return Err(format!("sorted_rows[{i}] = {flat}: X row {xr} >= {x_rows}"));
                }
                (Some(&pb.x[xr * k..xr * k + k]), pb.x_scale[xr])
            } else if flat == -1 {
                (None, 1.0)
            } else {
                return Err(format!("sorted_rows[{i}] = {flat}"));
            };
            let fa = 4.0f32 * a;
            for col in 0..r {
                let wrow = &codes[col * k..col * k + k];
                let mut total = 0f32;
                for kb in 0..nkb {
                    let mut partial = 0f32;
                    for t in 0..4 {
                        let k0 = 128 * kb + 32 * t;
                        for j in 0..32 {
                            xb[j] = match xrow {
                                Some(x) => act[usize::from(x[k0 + j])].clone()?,
                                None => 0.0,
                            };
                            wa[j] = wgt[usize::from(wrow[k0 + j])].clone()?;
                        }
                        // Hardware A = weights, B = activations, k ascending.
                        partial = accumulate(model, MfmaType::Fp8, partial, &wa, &xb);
                    }
                    let si = (col / 128) * nkb + kb;
                    let s_w = f32::from_bits(
                        u32::from(u16::from_le_bytes([scales[2 * si], scales[2 * si + 1]])) << 16,
                    );
                    if !s_w.is_finite() {
                        return Err(format!(
                            "expert {expert} scale ({}, {kb}) is not finite",
                            col / 128
                        ));
                    }
                    total = partial.mul_add(fa * s_w, total);
                }
                y[i * r + col] = if flat < 0 { 0.0 } else { total };
            }
        }
    }
    Ok(())
}
