// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1151 MTP-append DET Q8_0 -> F16 weight converter, emitted with the checked
//! ISA builder. Entry `mtp_q8_0_to_f16_pm_gfx1151`, module `mtp_q8_det_pm_gfx1151`.
//!
//! One elementwise pass turns a contiguous Q8_0 matrix into the F16 matrix the DET
//! dense GEMM (`mtp_dense_det_pm_gfx1151_w8`) consumes, in the same `[M, K]` layout:
//! element `idx` (row-major, `idx = row * K + col`) lives in Q8_0 block `idx / 32`
//! at quant slot `idx % 32`. A Q8_0 block is 34 bytes: an F16 scale `d` (little
//! endian, byte offset `34 * block`) then 32 signed int8 quants (byte offset
//! `34 * block + 2 + slot`). The gfx11 table has no byte load, so a quant is read as the
//! aligned (even-offset) 16-bit word `34 * block + 2 + (slot & 30)` and its byte selected by
//! `idx & 1` (the word never crosses the extent: blocks are 34 bytes). The shadow is immutable and cached per Q8 weight for
//! the model's lifetime by the runtime (`ensure_q8_0_f16_shadow`).
//!
//! Kernargs (20 bytes): `src` (Q8_0 blocks, offset 0), `dst` (F16, offset 8) and
//! `elems` (u32, offset 16: `M * K`). Grid `[ceil(elems / 256), 1, 1]`, block 256,
//! no LDS, one element per lane. The host (not the kernel) guarantees `elems % 32
//! == 0` (the route needs `K % 64 == 0`), 4-byte-aligned non-overlapping buffers
//! whose extents (`elems / 32 * 34` source bytes, `elems * 2` destination bytes)
//! are below `2^31`. Both are bounded raw buffer descriptors: no lane reads or writes
//! past the extents, `elems == 0` writes nothing.
//!
//! Numerics (explicit; DET, defined by this recipe, not by any incumbent kernel).
//! Per element: `d32 = f32(d)` (`v_cvt_f32_f16`, exact widening), `q32 = f32(int8)`
//! (the byte biased by 128, `v_cvt_f32_u32`, then `- 128.0`: exact), `p = d32 * q32` (one F32
//! multiply; for a finite `d` the product has at most 19 significant bits and an
//! exponent far inside the F32 range, so it is exact in F32), then `F16(p)`
//! (`v_cvt_f16_f32`, round to nearest even, single rounding of the exact product):
//!  * overflow: a finite product above 65504 rounds to +-Inf iff `|p| >= 65520`
//!    (RNE; e.g. `d = 65504, q = 2`) else saturates to the largest finite F16;
//!  * underflow: F16 subnormals are preserved (the module keeps the default
//!    `float_denorm_mode_16_64 = 3`); a product below half the smallest subnormal
//!    (`2^-25`) rounds to a signed zero (the sign of `d * q`, including `-0.0` for
//!    `q = 0` with a negative `d`); ties round to even;
//!  * non-finite: `d = +-Inf` gives `+-Inf * sign(q)` for `q != 0` and the default
//!    quiet NaN (`0xffc00000` -> F16 `0xfe00`) for `q = 0`; a NaN scale gives
//!    the NaN converted by `v_cvt_f16_f32` (`sign | 0x7c00 | 0x200 | payload >> 13`),
//!    for every `q`. A real Q8_0 model has finite scales, so none of this occurs.
use crate::insn::MemoryClass;
use crate::kernels::common::{lit, mem, op, s, smem, sop, sr, srd_tail, v};
use crate::{reg::Live, reg::RegRef, Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};
use peacemaker_author::{Gfx1151, Target, Workgroup};

type R = Result<(), String>;

/// The one entry (exact gfx1151).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
}
impl Spec {
    pub fn symbol(&self) -> String {
        format!("mtp_q8_0_to_f16_pm_{}", self.arch.name())
    }
    pub fn module(arch: Arch) -> String {
        format!("mtp_q8_det_pm_{}", arch.name())
    }
    pub fn validate(&self) -> R {
        if self.arch != Arch::Gfx1151 {
            return Err("mtp_q8_det_gfx11 builder is exact-gfx1151 only".into());
        }
        Ok(())
    }
}

/// Lanes per workgroup (one element each) and kernarg block bytes.
pub const THREADS: u16 = 256;
pub const KERNARG_BYTES: u32 = 20;
/// Q8_0 block geometry: 34 bytes = F16 scale + 32 int8, 32 elements per block.
pub const BLOCK_BYTES: u32 = 34;
pub const BLOCK_ELEMS: u32 = 32;

// SGPRs: s0:1 kernarg pointer, s2 workgroup id x.
const KARG: u8 = 0;
const WGX: u8 = 2;
const PTRS: u8 = 4; // src 4:5, dst 6:7
const SN: u8 = 8;
const SBASE: u8 = 9;
const MASK: u8 = 10; // pair 10:11
const SRD_SRC: u8 = 12;
const SRD_DST: u8 = 16;
const SRC_BYTES: u8 = 20;
const DST_BYTES: u8 = 21;
const SGPR_BUDGET: u16 = 24;
// VGPRs: v0 tid, v1 element, v2 block byte offset, v3 even slot (slot & 30), v4 quant word byte offset,
// v5 scale -> product, v6 quant word -> quant, v7 destination byte offset, v8 F16 result,
// v9 byte shift; v15 pads the granule.
const TID: u8 = 0;
const IDX: u8 = 1;
const BOFF: u8 = 2;
const SLOT: u8 = 3;
const QOFF: u8 = 4;
const SCALE: u8 = 5;
const QUANT: u8 = 6;
const DOFF: u8 = 7;
const HALF: u8 = 8;
const SHIFT: u8 = 9;
const PAD: u8 = 15;
const VGPRS: u16 = 16;

fn lbl(name: &str) -> String {
    format!(".Lmtpq8_{name}")
}

fn plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(VGPRS, SGPR_BUDGET)?;
    let w = || Live::Whole;
    p.s::<2>("kernarg_ptr", KARG, w())?;
    p.s::<1>("wg_x", WGX, w())?;
    p.s::<4>("pointers", PTRS, w())?;
    for (name, r) in [
        ("elems", SN),
        ("wg_base", SBASE),
        ("src_bytes", SRC_BYTES),
        ("dst_bytes", DST_BYTES),
    ] {
        p.s::<1>(name, r, w())?;
    }
    p.s::<2>("lane_mask", MASK, w())?;
    p.s::<4>("srd_src", SRD_SRC, w())?;
    p.s::<4>("srd_dst", SRD_DST, w())?;
    for (name, r) in [
        ("tid", TID),
        ("element", IDX),
        ("block_off", BOFF),
        ("slot", SLOT),
        ("quant_off", QOFF),
        ("scale", SCALE),
        ("quant", QUANT),
        ("dst_off", DOFF),
        ("half", HALF),
        ("byte_shift", SHIFT),
        ("granule_pad", PAD),
    ] {
        p.v::<1>(name, r, w())?;
    }
    Ok(p)
}

fn mask() -> [RegRef; 1] {
    [sr(MASK, 2)]
}

/// Raw buffer load `u16` at the even byte offset `v{voff}` (zero-extended into the whole VGPR). The
/// gfx11 table has no `buffer_load_u8`: a quant byte is read as its aligned 16-bit word.
fn bload(b: &mut Builder, name: &str, dst: u8, voff: u8, srd: u8) -> R {
    mem(
        b,
        format!(
            "buffer_load_{name} v{dst}, v{voff}, s[{srd}:{}], 0 offen",
            srd + 3
        ),
        &[v(dst)],
        &[v(voff), sr(srd, 4)],
        MemoryClass::VmemLoad,
    )
}
/// Raw buffer store of the low 16 bits at `v{voff}`.
fn bstore_b16(b: &mut Builder, data: u8, voff: u8, srd: u8) -> R {
    mem(
        b,
        format!(
            "buffer_store_b16 v{data}, v{voff}, s[{srd}:{}], 0 offen",
            srd + 3
        ),
        &[],
        &[v(voff), v(data), sr(srd, 4)],
        MemoryClass::VmemStore,
    )
}
/// Descriptor over `s[ptr:ptr+1]` bounded at `s{records}` bytes.
fn descriptor(b: &mut Builder, srd: u8, ptr: u8, records: u8) -> R {
    sop(b, format!("s_mov_b32 s{srd}, s{ptr}"), &[srd], &[ptr])?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", srd + 1, ptr + 1),
        &[srd + 1],
        &[ptr + 1],
    )?;
    srd_tail(b, srd, Some(records))
}

fn kernel<T: Target>(wg: &mut Workgroup<T, Builder>) -> R {
    let end = wg.exit(&lbl("end"))?;
    let b = wg.isa();
    smem(b, PTRS, 2, KARG, 0)?;
    smem(b, PTRS + 2, 2, KARG, 8)?;
    smem(b, SN, 1, KARG, 16)?;
    // Extents: ceil(elems / 32) * 34 source bytes, elems * 2 destination bytes.
    sop(
        b,
        format!("s_add_i32 s{SRC_BYTES}, s{SN}, {}", BLOCK_ELEMS - 1),
        &[SRC_BYTES],
        &[SN],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{SRC_BYTES}, s{SRC_BYTES}, 5"),
        &[SRC_BYTES],
        &[SRC_BYTES],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{SRC_BYTES}, s{SRC_BYTES}, {BLOCK_BYTES}"),
        &[SRC_BYTES],
        &[SRC_BYTES],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{DST_BYTES}, s{SN}, 1"),
        &[DST_BYTES],
        &[SN],
    )?;
    descriptor(b, SRD_SRC, PTRS, SRC_BYTES)?;
    descriptor(b, SRD_DST, PTRS + 2, DST_BYTES)?;
    // element = wg * 256 + tid; lanes with element >= elems are off for every memory access.
    sop(
        b,
        format!("s_lshl_b32 s{SBASE}, s{WGX}, 8"),
        &[SBASE],
        &[WGX],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{IDX}, s{SBASE}, v{TID}"),
        &[v(IDX)],
        &[s(SBASE), v(TID)],
    )?;
    op(
        b,
        format!("v_cmp_gt_u32_e64 s{MASK}, s{SN}, v{IDX}"),
        &mask(),
        &[s(SN), v(IDX)],
    )?;
    op(
        b,
        format!("s_and_b32 exec_lo, exec_lo, s{MASK}"),
        &[],
        &mask(),
    )?;
    // Block byte offset 34 * (element / 32); the quant byte `34 * block + 2 + slot` is read as the
    // aligned (even) 16-bit word at `34 * block + 2 + (slot & 30)`, its byte chosen by `element & 1`.
    op(
        b,
        format!("v_lshrrev_b32_e32 v{BOFF}, 5, v{IDX}"),
        &[v(BOFF)],
        &[v(IDX)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{SLOT}, {}, v{IDX}", lit(BLOCK_ELEMS - 2)),
        &[v(SLOT)],
        &[v(IDX)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{SHIFT}, 1, v{IDX}"),
        &[v(SHIFT)],
        &[v(IDX)],
    )?;
    op(
        b,
        format!("v_mul_lo_u32 v{BOFF}, v{BOFF}, {}", lit(BLOCK_BYTES)),
        &[v(BOFF)],
        &[v(BOFF)],
    )?;
    op(
        b,
        format!("v_add3_u32 v{QOFF}, v{BOFF}, v{SLOT}, 2"),
        &[v(QOFF)],
        &[v(BOFF), v(SLOT)],
    )?;
    bload(b, "u16", SCALE, BOFF, SRD_SRC)?;
    bload(b, "u16", QUANT, QOFF, SRD_SRC)?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{DOFF}, 1, v{IDX}"),
        &[v(DOFF)],
        &[v(IDX)],
    )?;
    // d32 = f32(d); q32 = f32(int8); p = d32 * q32 (one F32 multiply); out = F16 RNE(p).
    op(
        b,
        format!("v_cvt_f32_f16_e64 v{SCALE}, v{SCALE}.l"),
        &[v(SCALE)],
        &[v(SCALE)],
    )?;
    // The quant byte of its word (bit offset 8 * (element & 1)), biased by 128 (`xor 0x80`: 0..=255),
    // widened by the exact unsigned convert and unbiased by an exact F32 add of -128.0: `q32` is the
    // exact signed value (`+0.0` for q = 0). Only mnemonics with a gfx11 table row and a semantics row.
    op(
        b,
        format!("v_lshlrev_b32_e32 v{SHIFT}, 3, v{SHIFT}"),
        &[v(SHIFT)],
        &[v(SHIFT)],
    )?;
    op(
        b,
        format!("v_bfe_u32 v{QUANT}, v{QUANT}, v{SHIFT}, 8"),
        &[v(QUANT)],
        &[v(QUANT), v(SHIFT)],
    )?;
    op(
        b,
        format!("v_xor_b32_e32 v{QUANT}, {}, v{QUANT}", lit(0x80)),
        &[v(QUANT)],
        &[v(QUANT)],
    )?;
    op(
        b,
        format!("v_cvt_f32_u32_e32 v{QUANT}, v{QUANT}"),
        &[v(QUANT)],
        &[v(QUANT)],
    )?;
    op(
        b,
        format!("v_add_f32_e32 v{QUANT}, {}, v{QUANT}", lit(0xc300_0000)),
        &[v(QUANT)],
        &[v(QUANT)],
    )?;
    op(
        b,
        format!("v_mul_f32_e32 v{SCALE}, v{SCALE}, v{QUANT}"),
        &[v(SCALE)],
        &[v(SCALE), v(QUANT)],
    )?;
    // Both halves defined: the store reads the whole VGPR.
    op(b, format!("v_mov_b32_e32 v{HALF}, 0"), &[v(HALF)], &[])?;
    op(
        b,
        format!("v_cvt_f16_f32_e32 v{HALF}.l, v{SCALE}"),
        &[v(HALF)],
        &[v(SCALE), v(HALF)],
    )?;
    bstore_b16(b, HALF, DOFF, SRD_DST)?;
    wg.end(end)
}

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    let kernargs = KernargLayout::new(KERNARG_BYTES)
        .pointer("src", 0)
        .pointer("dst", 8)
        .hidden("elems", 16, 4, "by_value");
    let kspec = KernelSpec {
        kernel_id: "mtp_q8_det".into(),
        variant: "q8_f16".into(),
        arch: spec.arch,
        symbol: spec.symbol(),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: THREADS,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan()?);
    b.enable_delay_alu();
    kernel(&mut Workgroup::<Gfx1151, Builder>::new(&mut b)?)?;
    b.finish()
}

/// The entry as its code object (`Spec::module`).
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = vec![emit(Spec { arch })?];
    let (text, proof) = super::iu4_gemm::module(&emitted, &Spec::module(arch))?;
    Ok((emitted, text, proof))
}
