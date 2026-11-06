// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PM-only prefill decompression of the swizzled HC8 weight: exports `hc8_dequant_bf16` and
//! `hc8_dequant_f16`, appended to module `hc8_wmma_pm_gfx1201`.
//!
//! # ABI
//! 32-byte kernarg: `q@0` (swizzled E4M3 `[M*K]` bytes, see `swizzle_hc8`), `s@8` (f32 `[M]`
//! power-of-two row scales), `out@16` (row-major u16 `[M*K]`), `M@24`, `K@28` (u32). Block 256,
//! grid `ceil(M*K / 256)`. Host guarantees `M % 16 == 0`, `K % 32 == 0`, `M*K < 2^31`.
//!
//! # Mapping
//! Thread `i = 256*wg + tid` owns the swizzled byte `q[i]` (contiguous loads) and inverts the
//! swizzle to its row-major output slot: `tile = i >> 9`, lane `l = (i >> 4) & 31`
//! (`h = l >> 4`, `rr = l & 15`), `byte = i & 15` (`step = byte >> 3`, `j = byte & 7`),
//! `mt, kt = divmod(tile, K/32)`, `r = 16*mt + rr`,
//! `c = 32*kt + 16*step + 4*h + (j & 3) + 8*(j >> 2)`; it stores at `out[r*K + c]`.
//! Lanes with `i >= M*K` are exec-masked off before any memory access.
//!
//! # Arithmetic
//! `v = decode_e4m3(q) * s[r]` in f32 (`v_cvt_pk_f32_fp8_e32`, one `v_mul_f32`; exact because
//! `s` is a power of two). bf16: the high 16 bits of `v` (every finite product is exactly a
//! BF16). f16: `v_cvt_f16_f32` (RNE, gradual subnormals, overflow to inf).
//! The tile divmod is the f32 reciprocal estimate with the two-sided integer correction of
//! `common::gather_offset` (`tile < 2^22`, quotient off by at most one).
//!
//! Registers: kernarg ptr s[0:1]; loaded kernarg s[8:15]; s16 `K/32`, s17 `M*K`, s18 workgroup
//! id, s19 lane mask (`ttmp9` is not a plan register). VGPRs v0..v17. Every range is declared
//! in widths 1/2/4/8 (`add_split`), so the emitted SGPR/VGPR counts are exactly these.
use super::{add_split, branch, R};
use crate::insn::MemoryClass;
use crate::kernels::common::{mem, op, s, smem, sop, sr, v, vr};
use crate::reg::RegRef;
use crate::plan::Access;
use crate::reg::Kind;
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

/// Output element format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DequantOut {
    Bf16,
    F16,
}

/// Both dequant exports, in module order.
pub const DEQUANT_OUTS: [DequantOut; 2] = [DequantOut::Bf16, DequantOut::F16];
/// Kernarg bytes.
pub const DEQUANT_KERNARG_BYTES: u32 = 32;
/// Threads per workgroup.
pub const DEQUANT_BLOCK: u32 = 256;

impl DequantOut {
    pub fn symbol(self) -> &'static str {
        match self {
            DequantOut::Bf16 => "hc8_dequant_bf16",
            DequantOut::F16 => "hc8_dequant_f16",
        }
    }
}

// SGPRs.
const KARG: u8 = 0;
const QP: u8 = 8;
const SP: u8 = 10;
const OP: u8 = 12;
const SM: u8 = 14;
const SK: u8 = 15;
const KT: u8 = 16;
const TOT: u8 = 17;
const WG: u8 = 18;
const MSK: u8 = 19;
// VGPRs.
const TID: u8 = 0;
const I: u8 = 1;
const T: u8 = 2;
const Q: u8 = 3;
const R_: u8 = 4;
const W: u8 = 5;
const F: u8 = 6;
const STEP: u8 = 7;
const PK: u8 = 8; // pair 8..9
const QB: u8 = 10;
const SC: u8 = 11;
const ROW: u8 = 12;
const COL: u8 = 13;
const OFF: u8 = 14;
const SOFF: u8 = 15;
const T1: u8 = 16;
const T2: u8 = 17;
const V_TOP: u16 = 18;

fn plan() -> Result<RegPlan, String> {
    let mut regs = RegPlan::new(V_TOP, 24)?;
    for (name, base, len) in [("kernarg", KARG, 2), ("args", QP, 8), ("scalars", KT, 4)] {
        add_split(&mut regs, name, Kind::S, base, len)?;
    }
    add_split(&mut regs, "lanes", Kind::V, 0, V_TOP as u8)?;
    Ok(regs)
}

fn gload(b: &mut Builder, mnemonic: &str, dst: u8, voff: u8, base: u8) -> R {
    mem(
        b,
        format!("{mnemonic} v{dst}, v{voff}, s[{base}:{}]", base + 1),
        &[v(dst)],
        &[v(voff), sr(base, 2)],
        MemoryClass::VmemLoad,
    )
}

/// `v{dst} = f(src regs)` for one VALU op over VGPR operands, `uses` listing every VGPR/SGPR read.
fn alu(b: &mut Builder, text: String, dst: u8, uses: &[RegRef]) -> R {
    op(b, text, &[v(dst)], uses)
}

/// `(Q, R_) = divmod(T, s{KT})`, `T < 2^22`: f32 reciprocal estimate, then the two-sided
/// integer correction of `common::gather_offset` (the estimate is off by at most one).
fn divmod_tile(b: &mut Builder) -> R {
    alu(b, format!("v_cvt_f32_u32_e32 v{F}, s{KT}"), F, &[s(KT)])?;
    alu(b, format!("v_rcp_iflag_f32_e32 v{F}, v{F}"), F, &[v(F)])?;
    alu(b, format!("v_cvt_f32_u32_e32 v{Q}, v{T}"), Q, &[v(T)])?;
    alu(b, format!("v_mul_f32_e32 v{Q}, v{Q}, v{F}"), Q, &[v(Q), v(F)])?;
    alu(b, format!("v_cvt_u32_f32_e32 v{Q}, v{Q}"), Q, &[v(Q)])?;
    alu(b, format!("v_mul_lo_u32 v{R_}, v{Q}, s{KT}"), R_, &[v(Q), s(KT)])?;
    alu(b, format!("v_sub_nc_u32_e32 v{R_}, v{T}, v{R_}"), R_, &[v(T), v(R_)])?;
    // remainder < 0: q -= 1, r += d.
    op(b, format!("v_cmp_gt_i32_e64 s{MSK}, 0, v{R_}"), &[s(MSK)], &[v(R_)])?;
    alu(b, format!("v_add_nc_u32_e32 v{W}, -1, v{Q}"), W, &[v(Q)])?;
    alu(
        b,
        format!("v_cndmask_b32_e64 v{Q}, v{Q}, v{W}, s{MSK}"),
        Q,
        &[v(Q), v(W), s(MSK)],
    )?;
    alu(b, format!("v_add_nc_u32_e32 v{W}, s{KT}, v{R_}"), W, &[s(KT), v(R_)])?;
    alu(
        b,
        format!("v_cndmask_b32_e64 v{R_}, v{R_}, v{W}, s{MSK}"),
        R_,
        &[v(R_), v(W), s(MSK)],
    )?;
    // remainder >= d: q += 1, r -= d.
    op(b, format!("v_cmp_le_u32_e64 s{MSK}, s{KT}, v{R_}"), &[s(MSK)], &[s(KT), v(R_)])?;
    alu(b, format!("v_add_nc_u32_e32 v{W}, 1, v{Q}"), W, &[v(Q)])?;
    alu(
        b,
        format!("v_cndmask_b32_e64 v{Q}, v{Q}, v{W}, s{MSK}"),
        Q,
        &[v(Q), v(W), s(MSK)],
    )?;
    alu(b, format!("v_sub_nc_u32_e64 v{W}, v{R_}, s{KT}"), W, &[s(KT), v(R_)])?;
    alu(
        b,
        format!("v_cndmask_b32_e64 v{R_}, v{R_}, v{W}, s{MSK}"),
        R_,
        &[v(R_), v(W), s(MSK)],
    )
}

fn bfe(b: &mut Builder, dst: u8, off: u32, width: u32) -> R {
    alu(b, format!("v_bfe_u32 v{dst}, v{I}, {off}, {width}"), dst, &[v(I)])
}

fn emit_one(out: DequantOut) -> Result<Emitted, String> {
    let kernargs = KernargLayout::new(DEQUANT_KERNARG_BYTES)
        .pointer_access("q", 0, Access::ReadOnly)
        .pointer_access("s", 8, Access::ReadOnly)
        .pointer_access("out", 16, Access::WriteOnly)
        .hidden("m", 24, 4, "by_value")
        .hidden("k", 28, 4, "by_value");
    let spec = KernelSpec {
        kernel_id: "hc8_dequant".into(),
        variant: out.symbol().strip_prefix("hc8_dequant_").unwrap_or("").into(),
        arch: Arch::Gfx1201,
        symbol: out.symbol().into(),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: DEQUANT_BLOCK as u16,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut bld = Builder::new(spec, plan()?);
    bld.enable_delay_alu();
    let b = &mut bld;
    smem(b, QP, 8, KARG, 0)?;
    sop(b, format!("s_mov_b32 s{WG}, ttmp9"), &[WG], &[])?;
    // i = 256*wg + tid.
    op(
        b,
        format!("v_lshl_add_u32 v{I}, s{WG}, 8, v{TID}"),
        &[v(I)],
        &[s(WG), v(TID)],
    )?;
    b.wait_all()?;
    sop(b, format!("s_lshr_b32 s{KT}, s{SK}, 5"), &[KT], &[SK])?;
    sop(b, format!("s_mul_i32 s{TOT}, s{SM}, s{SK}"), &[TOT], &[SM, SK])?;
    // Lanes with i >= M*K touch nothing.
    op(
        b,
        format!("v_cmp_gt_u32_e64 s{MSK}, s{TOT}, v{I}"),
        &[s(MSK)],
        &[s(TOT), v(I)],
    )?;
    op(b, format!("s_and_b32 exec_lo, exec_lo, s{MSK}"), &[], &[s(MSK)])?;
    gload(b, "global_load_u8", QB, I, QP)?;
    alu(b, format!("v_lshrrev_b32_e32 v{T}, 9, v{I}"), T, &[v(I)])?;
    divmod_tile(b)?; // Q = mt, R_ = kt
    // Swizzle inverse: rr = (i>>4)&15, h = (i>>8)&1, step = (i>>3)&1, j = i&7.
    bfe(b, W, 4, 4)?; // rr
    bfe(b, T1, 8, 1)?; // h
    bfe(b, STEP, 3, 1)?;
    bfe(b, T2, 0, 2)?; // j & 3
    alu(b, format!("v_lshlrev_b32_e32 v{STEP}, 4, v{STEP}"), STEP, &[v(STEP)])?;
    alu(b, format!("v_add_nc_u32_e32 v{STEP}, v{STEP}, v{T2}"), STEP, &[v(STEP), v(T2)])?;
    alu(b, format!("v_lshl_add_u32 v{STEP}, v{T1}, 2, v{STEP}"), STEP, &[v(T1), v(STEP)])?;
    bfe(b, T2, 2, 1)?; // j >> 2
    alu(b, format!("v_lshl_add_u32 v{STEP}, v{T2}, 3, v{STEP}"), STEP, &[v(T2), v(STEP)])?; // kk
    alu(b, format!("v_lshl_add_u32 v{ROW}, v{Q}, 4, v{W}"), ROW, &[v(Q), v(W)])?; // r
    alu(b, format!("v_lshl_add_u32 v{COL}, v{R_}, 5, v{STEP}"), COL, &[v(R_), v(STEP)])?; // c
    alu(b, format!("v_lshlrev_b32_e32 v{SOFF}, 2, v{ROW}"), SOFF, &[v(ROW)])?;
    gload(b, "global_load_b32", SC, SOFF, SP)?;
    alu(b, format!("v_mul_lo_u32 v{OFF}, v{ROW}, s{SK}"), OFF, &[v(ROW), s(SK)])?;
    alu(b, format!("v_add_nc_u32_e32 v{OFF}, v{OFF}, v{COL}"), OFF, &[v(OFF), v(COL)])?;
    alu(b, format!("v_lshlrev_b32_e32 v{OFF}, 1, v{OFF}"), OFF, &[v(OFF)])?;
    // Decode: byte 0 of QB as E4M3 -> f32 pair (lo, byte 1); lo * s[r] is exact.
    op(
        b,
        format!("v_cvt_pk_f32_fp8_e32 {}, v{QB}.l", vr(PK, 2)),
        &[vr(PK, 2)],
        &[v(QB)],
    )?;
    alu(b, format!("v_mul_f32_e32 v{PK}, v{PK}, v{SC}"), PK, &[v(PK), v(SC)])?;
    match out {
        DequantOut::Bf16 => {
            alu(b, format!("v_lshrrev_b32_e32 v{PK}, 16, v{PK}"), PK, &[v(PK)])?;
        }
        DequantOut::F16 => {
            op(
                b,
                format!("v_cvt_f16_f32_e32 v{PK}.l, v{PK}"),
                &[v(PK)],
                &[v(PK)],
            )?;
        }
    }
    mem(
        b,
        format!("global_store_b16 v{OFF}, v{PK}, s[{OP}:{}]", OP + 1),
        &[],
        &[v(OFF), v(PK), sr(OP, 2)],
        MemoryClass::VmemStore,
    )?;
    b.wait_all()?;
    branch(b, "s_endpgm")?;
    bld.finish()
}

/// Build one dequant export (the public test/CLI emitter).
pub fn emit_dequant(out: DequantOut) -> Result<Emitted, String> {
    emit_one(out).map_err(|e| format!("{}: {}: {e}", super::MODULE, out.symbol()))
}

/// Both dequant exports, in [`DEQUANT_OUTS`] order.
pub fn emit_dequants() -> Result<Vec<Emitted>, String> {
    DEQUANT_OUTS.into_iter().map(emit_dequant).collect()
}
