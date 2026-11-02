// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Row-local R128 / K640 `block_i4_128` (A4) producer for the sub-4-bit down
//! projection (K3), emitted with the checked ISA builder for gfx1201 only:
//! the single symbol `mq_i4_a4_r128_k640` in the module `mq_i4_a4_pm_gfx1201`
//! (bundle `kernels/mq_i4_a4_pm_gfx1201.hxaco`). It is the PM form of
//! `kernels/src/qwen4_moe_rotate128_i4.hip` for the FN down profile (rotation
//! width 128, `K = 640` = five K128 groups, MQ-I4 qt132/qt134), emitting the
//! same bytes as that kernel's C2 `{5,7}` quantizer under the standalone lane
//! ownership.
//!
//! ABI (48-byte kernarg segment): `h@0` (read-only, grouped BF16 SwiGLU rows
//! `[P][640]` as `u16` bf16 bits), `sorted@8` (read-only, `P` i32 flat-slot
//! ids, `-1` = pad row), `xq@16` (write-only, the down sidecar: `5 * L` records
//! of 72 bytes), `signs1@24` / `signs2@32` (read-only, 128 f32 each, the
//! profile's R128 sign tables, seeds 43 / 1043), `P@40` (u32 rows), `L@44`
//! (u32 total flat slots).
//!
//! Launch: grid `[5 * P, 1, 1]`, block `[32, 1, 1]` (one wave32 per
//! (row, K128 group), no LDS, no barrier). Workgroup `x` is `(p, g)` with
//! `p = x / 5`, `g = x % 5`; `x >= 5 * P` returns. The row count enters only
//! through that grid and guard: the kernel is one straight-line function of the
//! single (row, group) it owns.
//!
//! Row-locality: a wave reads exactly `sorted[p]`, the 256 bytes of `h` row `p`
//! group `g` (`h + p*1280 + g*256`, lane `l` the 8 bytes at `8l`) and the 128-entry
//! sign tables; nothing of any other row. It writes exactly record `g*L + sorted[p]`
//! of `xq` (`xq + 72*(g*L + slot)`): `qs` bytes `2l..2l+1` per lane, and
//! `(d, s)` from lane 0. A pad row (`sorted[p] < 0`) returns before any load of
//! `h` and writes nothing; `slot < L` is the host's contract (each valid slot
//! appears once).
//!
//! Domain (host-validated): `P > 0`, `L > 0`, `5 * P` and `h`'s `P * 1280` bytes
//! addressable; `h` finite bf16 whose rotation does not overflow f32.
//!
//! Numerics (W3-GPU-PLAN §1.0 A4 block: repr(C) 72 B `{f32 d, i32 s, 64 qs}`,
//! element `j` in byte `j/2`, **low nibble first**, no MQ4 XOR rebias). Lane `l`
//! owns the four contiguous group elements `4l..4l+3`:
//!  * widen: element `4l` / `4l+1` are the low / high bf16 of dword 0, `4l+2` /
//!    `4l+3` of dword 1, each shifted into the high half of an f32 (lossless);
//!  * rotation, as `mq_rotate_x_128_v2`: `v = x * signs1`; in-lane butterflies
//!    strides 1 then 2 (`lo' = lo + hi`, `hi' = lo - hi`); lane butterflies at
//!    lane xor 1, 2, 4, 8, 16 (DPP `row_xmask` / `v_permlanex16_b32`;
//!    `v = fma(sign, v, partner)`, sign `-1` on lanes with the stride bit, so
//!    `partner - v` there and `v + partner` elsewhere, each one rounding);
//!    `x = (v * 0.0883883476f) * signs2`;
//!  * quantizer = `fake_quantize_block_i4_128(candidates = 2)`
//!    (`calibration.rs`, CPU twin of `quantize_block_i4_128_wave<true>` with
//!    `IU4_A4_CANDIDATES=2`) and `pilot_solve.a4_emulate`: wave `amax` of
//!    `|x|` (xor tree 16,8,4,2,1); `base = (amax / 7) * 0.5` (IEEE divide);
//!    candidates `d0 = base * 0x1.b6db6ep+0`, `d1 = base * 2`; per candidate and
//!    element `q = clamp(rint(x / d), -8, 7)` (IEEE divide, every divide is the
//!    full `v_div_scale` / `v_rcp` / fma / `v_div_fmas` / `v_div_fixup`
//!    expansion; there is no reciprocal shortcut and no fallback branch),
//!    `err = fma(-q, d, x)`, per-lane `mse` chain `err0*err0` then `fma(err_i,
//!    err_i, mse)`, summed across lanes by the same xor tree 16,8,4,2,1 (so
//!    every lane holds lane 0's value); candidate 0 wins iff `mse0 < 1e30`,
//!    candidate 1 iff `mse1 < (mse0 < 1e30 ? mse0 : 1e30)` (strict, in that
//!    order); `amax == 0` or no winner is the unit block `d = 1`, `q = 0` /
//!    `clamp(rint(x))`; `s` = exact integer sum of the 128 codes.
//!
//! Resources: no LDS, no barrier, no scratch; 72 VGPRs, wave32.

use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{lit, mem, op, s, s_add_i32, smem, sop, sr, v, vr};
use crate::plan::Access;
use crate::reg::{Kind, Live, RegRef};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// The kernarg segment size.
pub const KERNARG_BYTES: u32 = 48;
/// Workgroup size (threads): one wave32.
pub const WORKGROUP_SIZE: u32 = 32;
/// Rotation width and A4 group width in columns.
pub const GROUP: u32 = 128;
/// The row width this entry is specialised to (FN down, `K = 640`).
pub const K: u32 = 640;
/// K128 groups per row; workgroups per row.
pub const GROUPS: u32 = K / GROUP;
/// Bytes of one `block_i4_128` record.
pub const RECORD_BYTES: u32 = 72;
/// BF16 bytes of one `h` row.
pub const ROW_BYTES: u32 = K * 2;
/// Sign-table seeds of the R128 profile (`signs1`, `signs2`).
pub const SIGN_SEEDS: (u32, u32) = (43, 1043);
/// Bits of the f32 rotation scale `0.0883883476f` = `1/sqrt(128)` rounded to f32.
pub const ROTATION_SCALE_BITS: u32 = 0x3db5_04f3;
/// Candidate 0's multiplier `0x1.b6db6ep+0f`.
pub const CAND0_MULT_BITS: u32 = 0x3fdb_6db7;

/// The only entry: `(arch)`; gfx1201 only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
}
impl Spec {
    /// The exported symbol.
    pub fn symbol(&self) -> String {
        "mq_i4_a4_r128_k640".into()
    }
    /// The `--epi` name, the label prefix and the contract file tag.
    pub fn tag(&self) -> &'static str {
        "r128_k640"
    }
    pub fn module(arch: Arch) -> String {
        format!("mq_i4_a4_pm_{}", arch.name())
    }
    pub fn validate(&self) -> R {
        if self.arch != Arch::Gfx1201 {
            return Err("mq_i4_a4 builder is exact-gfx1201 only".into());
        }
        Ok(())
    }
}

/// The module's entries in order (one).
pub fn module_specs(arch: Arch) -> Vec<Spec> {
    vec![Spec { arch }]
}

// ----------------------------------------------------------------------- SGPRs
// s0:1 kernarg pointer; the workgroup id is ttmp9 (gfx12).
pub(super) const KARG: u8 = 0;
/// Kernargs: `s_load_b256` s[8:15] = h, sorted, xq, signs1; `s_load_b128` s[16:19] = signs2, P, L.
pub(super) const ARG_H: u8 = 8;
pub(super) const ARG_SORTED: u8 = 10;
pub(super) const ARG_XQ: u8 = 12;
pub(super) const ARG_S1: u8 = 14;
pub(super) const ARG_S2: u8 = 16;
const ARG_P: u8 = 18;
const ARG_L: u8 = 19;
const WGX: u8 = 20;
const ROWS5: u8 = 21;
pub(super) const ROW: u8 = 22;
pub(super) const GRP: u8 = 23;
pub(super) const T0: u8 = 24;
pub(super) const T1: u8 = 25;
pub(super) const SORT_PTR: u8 = 26;
pub(super) const SLOT: u8 = 28;
pub(super) const H_PTR: u8 = 30;
pub(super) const XQ_PTR: u8 = 32;
const C7: u8 = 34;
const CM8: u8 = 35;
const ZERO: u8 = 36;
const TAKE0: u8 = 37;
const USE1: u8 = 38;
const UNIT: u8 = 39;
const MT: u8 = 40;
const PSEL: u8 = 41;
pub(super) const KIMM: u8 = 42;

// ----------------------------------------------------------------------- VGPRs
// v0 = lane (1D block, no packed ids).
const LANE: u8 = 0;
const VH: u8 = 1;
const VSG: u8 = 2;
const VQS: u8 = 3;
const VZERO: u8 = 4;
/// `global_load_b64` destination: the lane's four bf16 as two dwords.
const RAW: u8 = 6;
/// signs1[4l..4l+3] and signs2[4l..4l+3].
const SG1: u8 = 8;
const SG2: u8 = 12;
/// The lane's four elements, then four butterfly spares.
const E: u8 = 16;
/// Lane-butterfly signs (`-1.0` where lane bit `i` is set) of strides 1, 2, 4, 8, 16.
const SGN: u8 = 24;
/// Lane-butterfly partners (4), `amax`, scratch, `base`, `d0`.
const PART: u8 = 32;
const AM: u8 = 36;
const T: u8 = 37;
const BASE: u8 = 38;
const D0: u8 = 39;
const D1: u8 = 40;
const MSE0: u8 = 41;
const MSE1: u8 = 42;
const WORD: u8 = 43;
/// Candidate codes q0 / q1, error buffers y0 / y1, integer codes.
const Q0: u8 = 44;
const Q1: u8 = 48;
const Y0: u8 = 52;
const Y1: u8 = 56;
const QI: u8 = 60;
/// Divide temporaries (5), then the record's `d` and `s` (consecutive for the b64 store).
const DIVT: u8 = 64;
const DOUT: u8 = 69;
const SOUT: u8 = 70;
/// VGPRs reserved (whole 8-register granules).
pub const VGPRS: u16 = 72;

const EXIT: &str = ".Lmqi4a4_exit";

pub(super) fn plan() -> Result<RegPlan, String> {
    let mut regs = RegPlan::new(VGPRS, 64)?;
    for (name, kind, base, len) in [
        ("kernarg", Kind::S, KARG, 2),
        ("kernel_arguments", Kind::S, ARG_H, 8),
        ("kernel_arguments2", Kind::S, ARG_S2, 4),
        ("workgroup_x", Kind::S, WGX, 1),
        ("rows5", Kind::S, ROWS5, 1),
        ("row", Kind::S, ROW, 1),
        ("group", Kind::S, GRP, 1),
        ("temp0", Kind::S, T0, 1),
        ("temp1", Kind::S, T1, 1),
        ("sorted_pointer", Kind::S, SORT_PTR, 2),
        ("slot", Kind::S, SLOT, 1),
        ("h_pointer", Kind::S, H_PTR, 2),
        ("xq_pointer", Kind::S, XQ_PTR, 2),
        ("const_7", Kind::S, C7, 1),
        ("const_m8", Kind::S, CM8, 1),
        ("zero_mask", Kind::S, ZERO, 1),
        ("take0_mask", Kind::S, TAKE0, 1),
        ("use1_mask", Kind::S, USE1, 1),
        ("unit_mask", Kind::S, UNIT, 1),
        ("none_mask", Kind::S, MT, 1),
        ("permlane_sel", Kind::S, PSEL, 1),
        ("imm_const", Kind::S, KIMM, 1),
        ("lane_and_raw", Kind::V, 0, 8),
        ("signs", Kind::V, 8, 8),
        ("elements_and_spares", Kind::V, E, 8),
        ("lane_signs", Kind::V, SGN, 8),
        ("partners_and_scalars", Kind::V, PART, 8),
        ("candidate_scalars", Kind::V, D1, 8),
        ("q1_y0", Kind::V, Q1, 8),
        ("y1_codes", Kind::V, Y1, 8),
        ("divide_and_record", Kind::V, DIVT, 8),
    ] {
        regs.add_range(name, kind, base, len, Live::Whole)?;
    }
    // v44..47 (q0) sit inside `candidate_scalars` (v40..47).
    Ok(regs)
}

pub(super) fn lbl() -> &'static str {
    EXIT
}

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    let kernargs = KernargLayout::new(KERNARG_BYTES)
        .pointer_access("h", 0, Access::ReadOnly)
        .pointer_access("sorted", 8, Access::ReadOnly)
        .pointer_access("xq", 16, Access::WriteOnly)
        .pointer_access("signs1", 24, Access::ReadOnly)
        .pointer_access("signs2", 32, Access::ReadOnly)
        .hidden("P", 40, 4, "by_value")
        .hidden("L", 44, 4, "by_value");
    let kspec = KernelSpec {
        kernel_id: "mq_i4_a4".into(),
        variant: spec.tag().into(),
        arch: spec.arch,
        symbol: spec.symbol(),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: WORKGROUP_SIZE as u16,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan()?);
    b.enable_delay_alu();
    prologue(&mut b)?;
    let e = rotate(&mut b)?;
    quantize(&mut b, e)?;
    store(&mut b)?;
    b.finish()
}

/// The module as one code object (`Spec::module`).
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = module_specs(arch)
        .into_iter()
        .map(emit)
        .collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, &Spec::module(arch))?;
    Ok((emitted, text, proof))
}

/// `dst = base + x * imm` over a 64-bit SGPR pair (`x` an unsigned SGPR, `imm` a literal);
/// clobbers `T0`, `T1`.
pub(super) fn mad64(b: &mut Builder, dst: u8, base: u8, x: u8, imm: u32) -> R {
    // SOP2 literals other than inline constants are not encodable here: the multiplier goes through an SGPR.
    sop(b, format!("s_mov_b32 s{KIMM}, {}", lit(imm)), &[KIMM], &[])?;
    mad64_reg(b, dst, base, x, KIMM)
}

/// `dst = base + x * s[mult]` over a 64-bit SGPR pair (`x`, `mult` unsigned SGPRs); clobbers `T0`, `T1`.
pub(super) fn mad64_reg(b: &mut Builder, dst: u8, base: u8, x: u8, mult: u8) -> R {
    // `T0` is written before `s_mul_hi_u32` reads `x`: an index in T0/T1 would be clobbered.
    if x == T0 || x == T1 {
        return Err(format!("mad64: index s{x} aliases the T0/T1 scratch pair"));
    }
    let a = b.spec.arch;
    sop(
        b,
        format!("s_mul_i32 s{T0}, s{x}, s{mult}"),
        &[T0],
        &[x, mult],
    )?;
    sop(
        b,
        format!("s_mul_hi_u32 s{T1}, s{x}, s{mult}"),
        &[T1],
        &[x, mult],
    )?;
    sop(
        b,
        format!("{} s{dst}, s{base}, s{T0}", super::common::s_add_u32(a)),
        &[dst],
        &[base, T0],
    )?;
    sop(
        b,
        format!(
            "{} s{}, s{}, s{T1}",
            super::common::s_addc_u32(a),
            dst + 1,
            base + 1
        ),
        &[dst + 1],
        &[base + 1, T1],
    )
}

/// Kernel arguments, the (row, group) of the workgroup, the row's slot (exit on a pad row),
/// the h / xq pointers, per-lane constants and the three loads.
fn prologue(b: &mut Builder) -> R {
    let a = b.spec.arch;
    smem(b, ARG_H, 8, KARG, 0)?;
    smem(b, ARG_S2, 4, KARG, 32)?;
    sop(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[WGX], &[])?;
    sop(
        b,
        format!("s_mul_i32 s{ROWS5}, s{ARG_P}, {GROUPS}"),
        &[ROWS5],
        &[ARG_P],
    )?;
    sop(
        b,
        format!("s_cmp_ge_u32 s{WGX}, s{ROWS5}"),
        &[],
        &[WGX, ROWS5],
    )?;
    branch(b, &format!("s_cbranch_scc1 {}", lbl()))?;
    // p = x / 5 (x * 0xcccccccd >> 34 is exact for every u32), g = x - 5p.
    sop(
        b,
        format!("s_mov_b32 s{KIMM}, {}", lit(0xcccc_cccd)),
        &[KIMM],
        &[],
    )?;
    sop(
        b,
        format!("s_mul_hi_u32 s{ROW}, s{WGX}, s{KIMM}"),
        &[ROW],
        &[WGX, KIMM],
    )?;
    sop(b, format!("s_lshr_b32 s{ROW}, s{ROW}, 2"), &[ROW], &[ROW])?;
    sop(
        b,
        format!("s_mul_i32 s{T0}, s{ROW}, {GROUPS}"),
        &[T0],
        &[ROW],
    )?;
    sop(
        b,
        format!("s_sub_co_u32 s{GRP}, s{WGX}, s{T0}"),
        &[GRP],
        &[WGX, T0],
    )?;
    // slot = sorted[p]; a pad row writes nothing.
    mad64(b, SORT_PTR, ARG_SORTED, ROW, 4)?;
    smem(b, SLOT, 1, SORT_PTR, 0)?;
    sop(b, format!("s_cmp_lt_i32 s{SLOT}, 0"), &[], &[SLOT])?;
    branch(b, &format!("s_cbranch_scc1 {}", lbl()))?;
    // h + p*1280 + g*256.
    mad64(b, H_PTR, ARG_H, ROW, ROW_BYTES)?;
    sop(b, format!("s_lshl_b32 s{T0}, s{GRP}, 8"), &[T0], &[GRP])?;
    sop(
        b,
        format!("{} s{H_PTR}, s{H_PTR}, s{T0}", super::common::s_add_u32(a)),
        &[H_PTR],
        &[H_PTR, T0],
    )?;
    sop(
        b,
        format!(
            "{} s{}, s{}, 0",
            super::common::s_addc_u32(a),
            H_PTR + 1,
            H_PTR + 1
        ),
        &[H_PTR + 1],
        &[H_PTR + 1],
    )?;
    // xq + 72 * (g*L + slot).
    // The record index goes to ROWS5 (dead after the pad-row test), not T0: `mad64` clobbers T0, and
    // `s_mul_hi_u32` must read the index, not the low product.
    sop(
        b,
        format!("s_mul_i32 s{ROWS5}, s{GRP}, s{ARG_L}"),
        &[ROWS5],
        &[GRP, ARG_L],
    )?;
    sop(
        b,
        format!("{} s{ROWS5}, s{ROWS5}, s{SLOT}", s_add_i32(a)),
        &[ROWS5],
        &[ROWS5, SLOT],
    )?;
    mad64(b, XQ_PTR, ARG_XQ, ROWS5, RECORD_BYTES)?;
    lane_setup(b)
}

/// The pointer-independent tail of every producer prologue: the clamp / permlane constants, the per-lane
/// offsets, the three loads (this lane's bf16 pair from `H_PTR`, the two sign quads) and the lane-butterfly signs.
pub(super) fn lane_setup(b: &mut Builder) -> R {
    sop(
        b,
        format!("s_mov_b32 s{C7}, {}", lit(7.0f32.to_bits())),
        &[C7],
        &[],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{CM8}, {}", lit((-8.0f32).to_bits())),
        &[CM8],
        &[],
    )?;
    sop(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[PSEL], &[])?;
    // Per-lane constants: h byte offset 8l, signs byte offset 16l, qs store offset 8 + 2l.
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VH}, 3, v{LANE}"),
        &[v(VH)],
        &[v(LANE)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VSG}, 4, v{LANE}"),
        &[v(VSG)],
        &[v(LANE)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VQS}, 1, v{LANE}"),
        &[v(VQS)],
        &[v(LANE)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{VQS}, 8, v{VQS}"),
        &[v(VQS)],
        &[v(VQS)],
    )?;
    op(b, format!("v_mov_b32_e32 v{VZERO}, 0"), &[v(VZERO)], &[])?;
    // The row group's bf16 pair of dwords and the two sign quads (exactly this lane's elements).
    mem(
        b,
        format!(
            "global_load_b64 v[{RAW}:{}], v{VH}, s[{H_PTR}:{}]",
            RAW + 1,
            H_PTR + 1
        ),
        &[vr(RAW, 2)],
        &[v(VH), sr(H_PTR, 2)],
        MemoryClass::VmemLoad,
    )?;
    mem(
        b,
        format!(
            "global_load_b128 v[{SG1}:{}], v{VSG}, s[{ARG_S1}:{}]",
            SG1 + 3,
            ARG_S1 + 1
        ),
        &[vr(SG1, 4)],
        &[v(VSG), sr(ARG_S1, 2)],
        MemoryClass::VmemLoad,
    )?;
    mem(
        b,
        format!(
            "global_load_b128 v[{SG2}:{}], v{VSG}, s[{ARG_S2}:{}]",
            SG2 + 3,
            ARG_S2 + 1
        ),
        &[vr(SG2, 4)],
        &[v(VSG), sr(ARG_S2, 2)],
        MemoryClass::VmemLoad,
    )?;
    // Lane-butterfly signs: -1.0 where lane bit i is set (stride 2^i): bit i to bit 31, mask, or 1.0.
    for i in 0..5u8 {
        op(
            b,
            format!("v_lshlrev_b32_e32 v{T}, {}, v{LANE}", 31 - i),
            &[v(T)],
            &[v(LANE)],
        )?;
        op(
            b,
            format!("v_and_b32_e32 v{T}, {}, v{T}", lit(0x8000_0000)),
            &[v(T)],
            &[v(T)],
        )?;
        // 1.0 (0x3f800000) is an inline constant: the oracle assembler encodes it inline, so the source must spell it so.
        op(
            b,
            format!("v_or_b32_e32 v{}, 1.0, v{T}", SGN + i),
            &[v(SGN + i)],
            &[v(T)],
        )?;
    }
    Ok(())
}

/// `dst = src` of lane `l ^ m` for `m` in 1, 2, 4, 8 (DPP `row_xmask`) and 16 (`v_permlanex16_b32`).
fn xch(b: &mut Builder, dst: u8, src: u8, m: u32) -> R {
    if m < 16 {
        op(
            b,
            format!("v_mov_b32_dpp v{dst}, v{src} row_xmask:{m} row_mask:0xf bank_mask:0xf"),
            &[v(dst)],
            &[v(src)],
        )
    } else {
        op(
            b,
            format!("v_permlanex16_b32 v{dst}, v{src}, s{PSEL}, 0xfedcba98"),
            &[v(dst)],
            &[v(src), s(PSEL)],
        )
    }
}

/// The lane's four rotated elements (registers returned in element order).
pub(super) fn rotate(b: &mut Builder) -> Result<[u8; 4], String> {
    // Widen: element 0 / 1 the low / high bf16 of dword 0, element 2 / 3 of dword 1.
    op(
        b,
        format!("v_lshlrev_b32_e32 v{E}, 16, v{RAW}"),
        &[v(E)],
        &[v(RAW)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{}, {}, v{RAW}", E + 1, lit(0xffff_0000)),
        &[v(E + 1)],
        &[v(RAW)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{}, 16, v{}", E + 2, RAW + 1),
        &[v(E + 2)],
        &[v(RAW + 1)],
    )?;
    op(
        b,
        format!(
            "v_and_b32_e32 v{}, {}, v{}",
            E + 3,
            lit(0xffff_0000),
            RAW + 1
        ),
        &[v(E + 3)],
        &[v(RAW + 1)],
    )?;
    // signs1.
    for k in 0..4u8 {
        op(
            b,
            format!("v_mul_f32_e32 v{0}, v{0}, v{1}", E + k, SG1 + k),
            &[v(E + k)],
            &[v(E + k), v(SG1 + k)],
        )?;
    }
    // In-lane butterflies, strides 1 then 2: lo' = lo + hi, hi' = lo - hi (the difference lands in a spare).
    let mut e = [E, E + 1, E + 2, E + 3];
    let mut spare: Vec<u8> = (E + 4..E + 8).collect();
    for stride in [1usize, 2] {
        for lo in (0..4).filter(|i| i & stride == 0) {
            let (hi, t) = (lo + stride, spare.pop().ok_or("butterfly spares")?);
            op(
                b,
                format!("v_sub_f32_e32 v{t}, v{}, v{}", e[lo], e[hi]),
                &[v(t)],
                &[v(e[lo]), v(e[hi])],
            )?;
            op(
                b,
                format!("v_add_f32_e32 v{0}, v{0}, v{1}", e[lo], e[hi]),
                &[v(e[lo])],
                &[v(e[lo]), v(e[hi])],
            )?;
            spare.push(e[hi]);
            e[hi] = t;
        }
    }
    // Lane butterflies, strides 1..16: v = fma(sign, v, partner) (partner - v on lanes with the stride bit).
    for (i, m) in [1u32, 2, 4, 8, 16].into_iter().enumerate() {
        for k in 0..4u8 {
            xch(b, PART + k, e[usize::from(k)], m)?;
        }
        let sign = SGN + i as u8;
        for k in 0..4u8 {
            let x = e[usize::from(k)];
            op(
                b,
                format!("v_fma_f32 v{x}, v{sign}, v{x}, v{}", PART + k),
                &[v(x)],
                &[v(sign), v(x), v(PART + k)],
            )?;
        }
    }
    // x = (v * 0.0883883476) * signs2.
    for k in 0..4u8 {
        let x = e[usize::from(k)];
        op(
            b,
            format!("v_mul_f32_e32 v{x}, {}, v{x}", lit(ROTATION_SCALE_BITS)),
            &[v(x)],
            &[v(x)],
        )?;
    }
    for k in 0..4u8 {
        let x = e[usize::from(k)];
        op(
            b,
            format!("v_mul_f32_e32 v{x}, v{x}, v{}", SG2 + k),
            &[v(x)],
            &[v(x), v(SG2 + k)],
        )?;
    }
    Ok(e)
}

#[derive(Clone, Copy)]
enum Opd {
    V(u8),
    S(u8),
}
impl Opd {
    fn text(self) -> String {
        match self {
            Self::V(n) => format!("v{n}"),
            Self::S(n) => format!("s{n}"),
        }
    }
    fn reg(self) -> RegRef {
        match self {
            Self::V(n) => v(n),
            Self::S(n) => s(n),
        }
    }
}

/// IEEE f32 `dst = num / den`: the `v_div_scale` / `v_rcp` / fma refinement /
/// `v_div_fmas` / `v_div_fixup` expansion (f32 denormals on, as `Epilogue::silu_dense`),
/// the scaled numerator's carry in VCC. Uses `DIVT..DIVT+5`; `dst` may alias `num`.
fn fdiv(b: &mut Builder, dst: u8, num: u8, den: Opd) -> R {
    let t = |i: u8| DIVT + i;
    let d = den.text();
    op(
        b,
        format!("v_div_scale_f32 v{}, null, {d}, {d}, v{num}", t(0)),
        &[v(t(0))],
        &[den.reg(), v(num)],
    )?;
    op(
        b,
        format!("v_div_scale_f32 v{}, vcc_lo, v{num}, {d}, v{num}", t(1)),
        &[v(t(1))],
        &[den.reg(), v(num)],
    )?;
    op(
        b,
        format!("v_rcp_f32_e32 v{}, v{}", t(2), t(0)),
        &[v(t(2))],
        &[v(t(0))],
    )?;
    op(
        b,
        format!("v_fma_f32 v{}, -v{}, v{}, 1.0", t(3), t(0), t(2)),
        &[v(t(3))],
        &[v(t(0)), v(t(2))],
    )?;
    op(
        b,
        format!("v_fmac_f32_e32 v{0}, v{1}, v{0}", t(2), t(3)),
        &[v(t(2))],
        &[v(t(2)), v(t(3))],
    )?;
    op(
        b,
        format!("v_mul_f32_e32 v{}, v{}, v{}", t(3), t(1), t(2)),
        &[v(t(3))],
        &[v(t(1)), v(t(2))],
    )?;
    op(
        b,
        format!("v_fma_f32 v{}, -v{}, v{}, v{}", t(4), t(0), t(3), t(1)),
        &[v(t(4))],
        &[v(t(0)), v(t(3)), v(t(1))],
    )?;
    op(
        b,
        format!("v_fmac_f32_e32 v{}, v{}, v{}", t(3), t(4), t(2)),
        &[v(t(3))],
        &[v(t(3)), v(t(4)), v(t(2))],
    )?;
    op(
        b,
        format!("v_fma_f32 v{}, -v{}, v{}, v{}", t(4), t(0), t(3), t(1)),
        &[v(t(4))],
        &[v(t(0)), v(t(3)), v(t(1))],
    )?;
    op(
        b,
        format!("v_div_fmas_f32 v{0}, v{0}, v{1}, v{2}", t(4), t(2), t(3)),
        &[v(t(4))],
        &[v(t(4)), v(t(2)), v(t(3))],
    )?;
    op(
        b,
        format!("v_div_fixup_f32 v{dst}, v{}, {d}, v{num}", t(4)),
        &[v(dst)],
        &[v(t(4)), den.reg(), v(num)],
    )
}

/// `x = tree(x)` over the wave by lane xor 16, 8, 4, 2, 1 with `combine` (`v_max_num_f32`,
/// `v_add_f32` or `v_add_nc_u32`); every lane ends with the same value.
fn reduce(b: &mut Builder, x: u8, combine: &str) -> R {
    for m in [16u32, 8, 4, 2, 1] {
        xch(b, T, x, m)?;
        op(
            b,
            format!("{combine} v{x}, v{x}, v{T}"),
            &[v(x)],
            &[v(x), v(T)],
        )?;
    }
    Ok(())
}

/// The `{5,7}` block quantizer on the lane's elements `e`, leaving the packed `qs` word in
/// `WORD`, the record's `(d, s)` in `DOUT`, `SOUT`.
pub(super) fn quantize(b: &mut Builder, e: [u8; 4]) -> R {
    // amax: lane max of |x|, then the wave.
    op(
        b,
        format!("v_max_num_f32_e64 v{Y0}, |v{}|, |v{}|", e[0], e[1]),
        &[v(Y0)],
        &[v(e[0]), v(e[1])],
    )?;
    op(
        b,
        format!("v_max_num_f32_e64 v{}, |v{}|, |v{}|", Y0 + 1, e[2], e[3]),
        &[v(Y0 + 1)],
        &[v(e[2]), v(e[3])],
    )?;
    op(
        b,
        format!("v_max_num_f32_e32 v{AM}, v{Y0}, v{}", Y0 + 1),
        &[v(AM)],
        &[v(Y0), v(Y0 + 1)],
    )?;
    reduce(b, AM, "v_max_num_f32_e32")?;
    op(
        b,
        format!("v_cmp_eq_f32_e64 s{ZERO}, 0, v{AM}"),
        &[s(ZERO)],
        &[v(AM)],
    )?;
    // base = (amax / 7) * 0.5; d0 = base * 0x1.b6db6ep+0; d1 = base * 2.
    fdiv(b, BASE, AM, Opd::S(C7))?;
    op(
        b,
        format!("v_mul_f32_e32 v{BASE}, 0.5, v{BASE}"),
        &[v(BASE)],
        &[v(BASE)],
    )?;
    op(
        b,
        format!("v_mul_f32_e32 v{D0}, {}, v{BASE}", lit(CAND0_MULT_BITS)),
        &[v(D0)],
        &[v(BASE)],
    )?;
    op(
        b,
        format!("v_mul_f32_e32 v{D1}, 2.0, v{BASE}"),
        &[v(D1)],
        &[v(BASE)],
    )?;
    // Per candidate: q = clamp(rint(x / d), -8, 7); err = fma(-q, d, x); mse chain; wave sum.
    for (q, y, d, mse) in [(Q0, Y0, D0, MSE0), (Q1, Y1, D1, MSE1)] {
        for k in 0..4u8 {
            fdiv(b, q + k, e[usize::from(k)], Opd::V(d))?;
            op(
                b,
                format!("v_rndne_f32_e32 v{0}, v{0}", q + k),
                &[v(q + k)],
                &[v(q + k)],
            )?;
            op(
                b,
                format!("v_maxmin_num_f32 v{0}, v{0}, s{CM8}, s{C7}", q + k),
                &[v(q + k)],
                &[v(q + k), s(CM8), s(C7)],
            )?;
        }
        for k in 0..4u8 {
            op(
                b,
                format!(
                    "v_fma_f32 v{}, -v{}, v{d}, v{}",
                    y + k,
                    q + k,
                    e[usize::from(k)]
                ),
                &[v(y + k)],
                &[v(q + k), v(d), v(e[usize::from(k)])],
            )?;
        }
        op(
            b,
            format!("v_mul_f32_e32 v{mse}, v{y}, v{y}"),
            &[v(mse)],
            &[v(y)],
        )?;
        for k in 1..4u8 {
            op(
                b,
                format!("v_fmac_f32_e32 v{mse}, v{0}, v{0}", y + k),
                &[v(mse)],
                &[v(mse), v(y + k)],
            )?;
        }
        reduce(b, mse, "v_add_f32_e32")?;
    }
    // Winner: strict `<` against 1e30 for d0, then d1 against the d0 result.
    let big = lit(1.0e30f32.to_bits());
    op(
        b,
        format!("v_cmp_gt_f32_e64 s{TAKE0}, {big}, v{MSE0}"),
        &[s(TAKE0)],
        &[v(MSE0)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{T}, {big}, v{MSE0}, s{TAKE0}"),
        &[v(T)],
        &[v(MSE0), s(TAKE0)],
    )?;
    op(
        b,
        format!("v_cmp_gt_f32_e64 s{USE1}, v{T}, v{MSE1}"),
        &[s(USE1)],
        &[v(T), v(MSE1)],
    )?;
    sop(
        b,
        format!("s_or_b32 s{MT}, s{TAKE0}, s{USE1}"),
        &[MT],
        &[TAKE0, USE1],
    )?;
    sop(b, format!("s_not_b32 s{MT}, s{MT}"), &[MT], &[MT])?;
    sop(
        b,
        format!("s_or_b32 s{UNIT}, s{ZERO}, s{MT}"),
        &[UNIT],
        &[ZERO, MT],
    )?;
    for k in 0..4u8 {
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{0}, v{0}, v{1}, s{USE1}",
                Q0 + k,
                Q1 + k
            ),
            &[v(Q0 + k)],
            &[v(Q0 + k), v(Q1 + k), s(USE1)],
        )?;
    }
    // Unit block (amax == 0 or no winner): q = zero ? 0 : clamp(rint(x / 1)), d = 1.
    for k in 0..4u8 {
        let (u, q) = (Y0 + k, Q0 + k);
        op(
            b,
            format!("v_rndne_f32_e32 v{u}, v{}", e[usize::from(k)]),
            &[v(u)],
            &[v(e[usize::from(k)])],
        )?;
        op(
            b,
            format!("v_maxmin_num_f32 v{u}, v{u}, s{CM8}, s{C7}"),
            &[v(u)],
            &[v(u), s(CM8), s(C7)],
        )?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{u}, v{u}, 0, s{ZERO}"),
            &[v(u)],
            &[v(u), s(ZERO)],
        )?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{q}, v{q}, v{u}, s{UNIT}"),
            &[v(q)],
            &[v(q), v(u), s(UNIT)],
        )?;
    }
    op(
        b,
        format!("v_cndmask_b32_e64 v{DOUT}, v{D0}, v{D1}, s{USE1}"),
        &[v(DOUT)],
        &[v(D0), v(D1), s(USE1)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{0}, v{0}, 1.0, s{UNIT}", DOUT),
        &[v(DOUT)],
        &[v(DOUT), s(UNIT)],
    )?;
    // Codes, their exact wave sum, and the lane's 16-bit qs word (nibble i = element 4l+i, low first).
    for k in 0..4u8 {
        op(
            b,
            format!("v_cvt_i32_f32_e32 v{}, v{}", QI + k, Q0 + k),
            &[v(QI + k)],
            &[v(Q0 + k)],
        )?;
    }
    op(
        b,
        format!("v_add_nc_u32_e32 v{SOUT}, v{QI}, v{}", QI + 1),
        &[v(SOUT)],
        &[v(QI), v(QI + 1)],
    )?;
    for k in 2..4u8 {
        op(
            b,
            format!("v_add_nc_u32_e32 v{SOUT}, v{}, v{SOUT}", QI + k),
            &[v(SOUT)],
            &[v(QI + k), v(SOUT)],
        )?;
    }
    reduce(b, SOUT, "v_add_nc_u32_e32")?;
    op(
        b,
        format!("v_and_b32_e32 v{WORD}, 15, v{QI}"),
        &[v(WORD)],
        &[v(QI)],
    )?;
    for k in 1..4u8 {
        op(
            b,
            format!("v_lshlrev_b32_e32 v{T}, {}, v{}", 4 * k, QI + k),
            &[v(T)],
            &[v(QI + k)],
        )?;
        op(
            b,
            format!(
                "v_and_or_b32 v{WORD}, v{T}, {}, v{WORD}",
                lit(0xf << (4 * k))
            ),
            &[v(WORD)],
            &[v(T), v(WORD)],
        )?;
    }
    Ok(())
}

/// Every lane stores its 16-bit qs word at record byte `8 + 2l`; lane 0 stores `(d, s)` at 0.
pub(super) fn store(b: &mut Builder) -> R {
    mem(
        b,
        format!(
            "global_store_b16 v{VQS}, v{WORD}, s[{XQ_PTR}:{}]",
            XQ_PTR + 1
        ),
        &[],
        &[v(VQS), v(WORD), sr(XQ_PTR, 2)],
        MemoryClass::VmemStore,
    )?;
    op(b, format!("v_cmpx_eq_u32_e32 0, v{LANE}"), &[], &[v(LANE)])?;
    mem(
        b,
        format!(
            "global_store_b64 v{VZERO}, v[{DOUT}:{SOUT}], s[{XQ_PTR}:{}]",
            XQ_PTR + 1
        ),
        &[],
        &[v(VZERO), vr(DOUT, 2), sr(XQ_PTR, 2)],
        MemoryClass::VmemStore,
    )?;
    b.wait_all()?;
    b.label(lbl())?;
    branch(b, "s_endpgm")
}

/// The shape contract of the emitted entry: counts of its characteristic
/// instructions, the forbidden families, register ceilings, wave32, no spills,
/// no private or LDS memory. Committed as
/// `kernels/mq_i4_a4.gfx1201.r128_k640.contract.json`.
pub fn contract(spec: Spec, e: &Emitted) -> serde_json::Value {
    contract_json(&spec.symbol(), e)
}

/// The shape contract of `e` under `symbol`; shared with `moe_rotate_i4_pm`, which is the same instruction set under
/// the same resource ceilings.
pub(super) fn contract_json(symbol: &str, e: &Emitted) -> serde_json::Value {
    let mut counts = serde_json::Map::new();
    for name in COUNTED {
        let n = e
            .s_text
            .lines()
            .filter(|l| l.split_whitespace().next() == Some(name))
            .count();
        if n > 0 {
            counts.insert(name.into(), n.into());
        }
    }
    serde_json::json!({
        "symbol": symbol, "counts": counts,
        "forbidden": ["scratch_*", "flat_*", "buffer_*", "ds_*", "s_barrier*", "v_readlane*", "v_writelane*", "v_wmma*", "v_swmmac*",
                      "v_fma_mix*", "s_waitcnt*", "s_setpc*", "s_swappc*", "s_branch*", "s_cbranch_vcc*", "s_cbranch_execz*", "s_cbranch_execnz*", "s_cbranch_scc0*"],
        "vgpr_max": VGPRS, "sgpr_max": 64,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0, "group_segment_fixed_bytes": 0,
    })
}
const COUNTED: [&str; 24] = [
    "s_load_b256",
    "s_load_b128",
    "s_load_b32",
    "global_load_b64",
    "global_load_b128",
    "global_store_b16",
    "global_store_b64",
    "v_div_scale_f32",
    "v_div_fmas_f32",
    "v_div_fixup_f32",
    "v_rcp_f32_e32",
    "v_mov_b32_dpp",
    "v_permlanex16_b32",
    "v_rndne_f32_e32",
    "v_maxmin_num_f32",
    "v_max_num_f32_e64",
    "v_max_num_f32_e32",
    "v_cvt_i32_f32_e32",
    "v_cndmask_b32_e64",
    "v_cmp_eq_f32_e64",
    "v_cmp_gt_f32_e64",
    "v_cmpx_eq_u32_e32",
    "s_cbranch_scc1",
    "s_endpgm",
];

pub(super) fn branch(b: &mut Builder, text: &str) -> R {
    b.control(Instruction::new(text, vec![], vec![]))
}
