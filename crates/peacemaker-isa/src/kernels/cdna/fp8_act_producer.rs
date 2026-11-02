// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Plain F32 activation pack, mirroring `pack_f32_to_fp8_mq4v2_gfx12`.
//!
//! ABI: x_f32, x_fp8, half_sums, row_scales (four pointers), K, N,
//! scale_mode, row_scale_shift (four i32/u32 values), 48 bytes total.
//! Launch one wave64 workgroup per row. K must be positive and divisible by
//! 256; inputs must be finite; the source route admits shifts -1 through 3.
//! Codes are row-major [N,K], sums [N,K/256,2], scales F32 [N].
//! Mode 1 stores a=2^e, including the optional shift; every other mode stores
//! one and ignores the shift. A zero row uses e=0 before the shift. The
//! source computes ceil-log2 from bits of RN(amax * RN(1/224)), NOT an exact
//! real division by 224. Both exponent clamps are reproduced.
//!
//! The FNUZ conversion uses ldexp(x,-e-1), never a reciprocal. Within the
//! source's OCP-finite range its bytes equal OCP encoding of x/a, except
//! negative zero becomes positive zero. Upper-clamp overflow rows do not
//! satisfy that byte-identity claim. Half sums decode the *emitted bytes*
//! as OCP, preserving the original sidecar domain and XOR reduction tree.
//! FP16_OVFL is zero, as for the raw HIP builtin (overflow emits FNUZ NaN).
//! [UNVERIFIED: confirm on MI300X] FNUZ negative-zero canonicalization,
//! pre-round overflow threshold, and NaN propagation/sign in half sums.

use crate::cdna::{builder::Builder, ops, Kernel};
use crate::Arch;
use peacemaker_cdna::enc::{
    ds::DsOp,
    flat::FlatOp,
    smem::SmemOp,
    sop::{Sop1Op as S1, Sop2Op as S2, SopcOp as SC, SoppOp as SP},
    vop::{Vop1Op as V1, Vop2Op as V2, VopcOp as VC},
    vop3::{Vop3OnlyOp as V3, Vop3Op},
};
use peacemaker_cdna::Src;

pub const SYMBOL: &str = "pm_cdna_fp8_act_producer";

fn reg(base: Src, delta: u8) -> Src {
    match base {
        Src::Sgpr(n) => Src::Sgpr(n + delta),
        Src::Vgpr(n) => Src::Vgpr(n + delta),
        _ => unreachable!("allocated register"),
    }
}

// Full 64-bit scalar pointer addition, including a carry from the byte offset.
fn add_ptr(b: &mut Builder, dst: Src, base: Src, lo: Src, hi: Src) -> Result<(), String> {
    b.push(ops::s2(S2::AddU32, dst, base, lo))?;
    b.push(ops::s2(S2::AddcU32, reg(dst, 1), reg(base, 1), hi))
}

fn address(b: &mut Builder, dst: Src, base: Src, offset: Src) -> Result<(), String> {
    // Carry-in already consumes the gfx9 scalar bus; stage the high word.
    b.push(ops::v1(V1::MovB32, reg(dst, 1), reg(base, 1)))?;
    b.push(ops::v2c(V2::AddCoU32, dst, Src::VccLo, base, offset))?;
    b.push(ops::v2ci(
        V2::AddcCoU32,
        reg(dst, 1),
        Some(Src::VccLo),
        Src::Int(0),
        reg(dst, 1),
        Src::VccLo,
    ))
}

// OCP decode by integer exponent construction; deliberately not a FNUZ decode.
fn decode_ocp(b: &mut Builder, dst: Src, byte: Src, tmp: Src) -> Result<(), String> {
    let exponent = tmp;
    let mantissa = reg(tmp, 1);
    let subnormal = reg(tmp, 2);
    let sign = reg(tmp, 3);
    b.push(ops::v2(V2::LshrrevB32, exponent, Src::Int(3), byte))?;
    b.push(ops::v2(V2::AndB32, exponent, Src::Int(15), exponent))?;
    b.push(ops::v2(V2::AndB32, mantissa, Src::Int(7), byte))?;
    b.push(ops::v1(V1::CvtF32U32, subnormal, mantissa))?;
    b.push(ops::v1(
        V1::MovB32,
        sign,
        Src::Lit((1.0f32 / 512.0).to_bits()),
    ))?;
    b.push(ops::v2(V2::MulF32, subnormal, sign, subnormal))?;
    b.push(ops::v2(V2::AddU32, dst, Src::Lit(120), exponent))?;
    b.push(ops::v2(V2::LshlrevB32, dst, Src::Int(23), dst))?;
    b.push(ops::v2(V2::LshlrevB32, mantissa, Src::Int(20), mantissa))?;
    b.push(ops::v2(V2::OrB32, dst, mantissa, dst))?;
    b.push(ops::vc(VC::CmpEqU32, Src::VccLo, Src::Int(0), exponent))?;
    b.push(ops::v2ci(
        V2::CndmaskB32,
        dst,
        None,
        dst,
        subnormal,
        Src::VccLo,
    ))?;
    b.push(ops::v2(V2::AndB32, mantissa, Src::Lit(127), byte))?;
    b.push(ops::v1(V1::MovB32, subnormal, Src::Lit(0x7fc0_0000)))?;
    b.push(ops::vc(VC::CmpEqU32, Src::VccLo, Src::Lit(127), mantissa))?;
    b.push(ops::v2ci(
        V2::CndmaskB32,
        dst,
        None,
        dst,
        subnormal,
        Src::VccLo,
    ))?;
    b.push(ops::v2(V2::AndB32, sign, Src::Lit(128), byte))?;
    b.push(ops::v2(V2::LshlrevB32, sign, Src::Int(24), sign))?;
    b.push(ops::v2(V2::OrB32, dst, sign, dst))
}

/// Build a gfx942 wave64 producer. The kernarg ABI is identical to the HIP twin.
pub fn build(arch: Arch) -> Result<Kernel, String> {
    let mut b = Builder::new(arch, SYMBOL, 64)?;
    for (name, readonly) in [
        ("x_f32", true),
        ("x_fp8", false),
        ("half_sums", false),
        ("row_scales", false),
    ] {
        b.arg_ptr(name, readonly)?;
    }
    for name in ["K", "N", "scale_mode", "row_scale_shift"] {
        b.arg_u32(name)?;
    }
    let row = b.workgroup_id(0)?;
    let lane = b.workitem_ids(1)?;
    let args = b.sgprs(12, 4)?;
    let x = args;
    let out = reg(args, 2);
    let sums = reg(args, 4);
    let scales = reg(args, 6);
    let k = reg(args, 8);
    let n = reg(args, 9);
    let mode = reg(args, 10);
    let shift = reg(args, 11);
    let ptrs = b.sgprs(10, 2)?;
    let xp = ptrs;
    let op = reg(ptrs, 2);
    let hp = reg(ptrs, 4);
    let scan = reg(ptrs, 6);
    let scale_ptr = reg(ptrs, 8);
    let scratch = b.sgprs(8, 2)?;
    let count = scratch;
    let exponent = reg(scratch, 1);
    let bits = reg(scratch, 2);
    let temp = reg(scratch, 3);
    let product = reg(scratch, 4);
    let saved_exec = reg(scratch, 6);
    let va = b.vgprs(2, 2)?;
    let values = b.vgprs(4, 4)?;
    let vmax = b.vgprs(1, 1)?;
    let other = b.vgprs(1, 1)?;
    let offset = b.vgprs(1, 1)?;
    let shuffle = b.vgprs(1, 1)?;
    let packed = b.vgprs(1, 1)?;
    let byte = b.vgprs(1, 1)?;
    let decoded = b.vgprs(4, 4)?;
    let decode_tmp = b.vgprs(4, 1)?;
    let sum = b.vgprs(1, 1)?;
    b.push(ops::s_load(SmemOp::LoadDwordx8, args, b.kernarg_base(), 0))?;
    b.push(ops::s_load(SmemOp::LoadDwordx4, k, b.kernarg_base(), 32))?;
    b.push(ops::sc(SC::CmpGeU32, row, n))?;
    b.branch(SP::CbranchScc1, "done")?;
    b.push(ops::sc(SC::CmpLeI32, k, Src::Int(0)))?;
    b.branch(SP::CbranchScc1, "done")?;
    // n*K uses both halves; byte addresses remain valid above 4 GiB.
    b.push(ops::s2(S2::MulI32, product, row, k))?;
    b.push(ops::s2(S2::MulHiU32, reg(product, 1), row, k))?;
    add_ptr(&mut b, op, out, product, reg(product, 1))?;
    b.push(ops::s2(S2::LshrB64, product, product, Src::Int(5)))?;
    add_ptr(&mut b, hp, sums, product, reg(product, 1))?;
    b.push(ops::s2(S2::LshlB64, product, product, Src::Int(7)))?;
    add_ptr(&mut b, xp, x, product, reg(product, 1))?;
    b.push(ops::s1(S1::MovB32, product, row))?;
    b.push(ops::s1(S1::MovB32, reg(product, 1), Src::Int(0)))?;
    b.push(ops::s2(S2::LshlB64, product, product, Src::Int(2)))?;
    add_ptr(&mut b, scale_ptr, scales, product, reg(product, 1))?;
    b.push(ops::s1(S1::MovB32, exponent, Src::Int(0)))?;
    b.push(ops::sc(SC::CmpEqU32, mode, Src::Int(1)))?;
    b.branch(SP::CbranchScc0, "publish")?;
    b.push(ops::s1(S1::MovB64, scan, xp))?;
    b.push(ops::s1(S1::MovB32, count, Src::Int(0)))?;
    b.push(ops::v1(V1::MovB32, vmax, Src::Int(0)))?;
    b.push(ops::v2(V2::LshlrevB32, offset, Src::Int(2), lane))?;
    b.label("amax_loop")?;
    address(&mut b, va, scan, offset)?;
    b.push(ops::global_load(FlatOp::LoadDword, values, va, None, 0))?;
    b.push(ops::v2(V2::AndB32, values, Src::Lit(0x7fff_ffff), values))?;
    b.push(ops::v2(V2::MaxF32, vmax, values, vmax))?;
    add_ptr(&mut b, scan, scan, Src::Lit(256), Src::Int(0))?;
    b.push(ops::s2(S2::AddU32, count, count, Src::Int(64)))?;
    b.push(ops::sc(SC::CmpLtU32, count, k))?;
    b.wait_all()?;
    b.branch(SP::CbranchScc1, "amax_loop")?;
    for distance in [32, 16, 8, 4, 2, 1] {
        b.push(ops::v2(V2::XorB32, shuffle, Src::Int(distance), lane))?;
        b.push(ops::v2(V2::LshlrevB32, shuffle, Src::Int(2), shuffle))?;
        b.push(ops::ds_xlane(DsOp::BpermuteB32, other, shuffle, vmax, 0))?;
        b.push(ops::v2(V2::MaxF32, vmax, other, vmax))?;
    }
    b.push(ops::v1(V1::ReadfirstlaneB32, bits, vmax))?;
    b.push(ops::sc(SC::CmpEqU32, bits, Src::Int(0)))?;
    b.branch(SP::CbranchScc1, "shift")?;
    b.push(ops::v1(
        V1::MovB32,
        other,
        Src::Lit((1.0f32 / 224.0).to_bits()),
    ))?;
    b.push(ops::v2(V2::MulF32, other, vmax, other))?;
    b.push(ops::v1(V1::ReadfirstlaneB32, bits, other))?;
    b.push(ops::s2(S2::LshrB32, exponent, bits, Src::Int(23)))?;
    b.push(ops::s2(S2::SubI32, exponent, exponent, Src::Lit(127)))?;
    b.push(ops::s2(S2::AndB32, temp, bits, Src::Lit(0x7fffff)))?;
    b.push(ops::sc(SC::CmpLgU32, temp, Src::Int(0)))?;
    b.push(ops::s2(S2::CselectB32, temp, Src::Int(1), Src::Int(0)))?;
    b.push(ops::s2(S2::AddI32, exponent, exponent, temp))?;
    b.push(ops::s2(
        S2::MaxI32,
        exponent,
        exponent,
        Src::Lit((-31i32) as u32),
    ))?;
    b.push(ops::s2(S2::MinI32, exponent, exponent, Src::Int(9)))?;
    b.label("shift")?;
    b.push(ops::s2(S2::AddI32, exponent, exponent, shift))?;
    b.label("publish")?;
    b.push(ops::s2(S2::AddI32, bits, exponent, Src::Lit(127)))?;
    b.push(ops::s2(S2::LshlB32, bits, bits, Src::Int(23)))?;
    b.push(ops::v1(V1::MovB32, other, bits))?;
    b.push(ops::v1(V1::MovB32, offset, Src::Int(0)))?;
    address(&mut b, va, scale_ptr, offset)?;
    b.push(ops::vc(VC::CmpEqU32, Src::VccLo, Src::Int(0), lane))?;
    b.push(ops::s1(S1::AndSaveexecB64, saved_exec, Src::VccLo))?;
    b.push(ops::global_store(FlatOp::StoreDword, va, other, None, 0))?;
    b.wait_all()?;
    b.push(ops::s1(S1::MovB64, Src::ExecLo, saved_exec))?;
    b.push(ops::s2(S2::SubI32, exponent, Src::Int(-1), exponent))?;
    b.push(ops::s1(S1::MovB32, count, Src::Int(0)))?;
    b.label("pack_loop")?;
    b.push(ops::v2(V2::LshlrevB32, offset, Src::Int(4), lane))?;
    address(&mut b, va, xp, offset)?;
    b.push(ops::global_load(FlatOp::LoadDwordx4, values, va, None, 0))?;
    for i in 0..4 {
        b.push(ops::v3(
            V3::LdexpF32,
            reg(values, i),
            &[reg(values, i), exponent],
        ))?;
    }
    b.push(ops::v1(V1::MovB32, packed, Src::Int(0)))?;
    for i in 0..2 {
        b.push(ops::v3m(
            Vop3Op::Only(V3::CvtPkFp8F32),
            packed,
            &[reg(values, i * 2), reg(values, i * 2 + 1)],
            ops::Mods {
                op_sel: i * 8,
                ..Default::default()
            },
        ))?;
    }
    b.push(ops::v2(V2::LshlrevB32, offset, Src::Int(2), lane))?;
    address(&mut b, va, op, offset)?;
    b.push(ops::global_store(FlatOp::StoreDword, va, packed, None, 0))?;
    for i in 0..4 {
        b.push(ops::v2(V2::LshrrevB32, byte, Src::Int(i * 8), packed))?;
        b.push(ops::v2(V2::AndB32, byte, Src::Lit(255), byte))?;
        decode_ocp(&mut b, reg(decoded, i as u8), byte, decode_tmp)?;
    }
    b.push(ops::v2(V2::AddF32, sum, decoded, reg(decoded, 1)))?;
    b.push(ops::v2(V2::AddF32, sum, sum, reg(decoded, 2)))?;
    b.push(ops::v2(V2::AddF32, sum, sum, reg(decoded, 3)))?;
    for distance in [16, 8, 4, 2, 1] {
        b.push(ops::v2(V2::XorB32, shuffle, Src::Int(distance), lane))?;
        b.push(ops::v2(V2::LshlrevB32, shuffle, Src::Int(2), shuffle))?;
        b.push(ops::ds_xlane(DsOp::BpermuteB32, other, shuffle, sum, 0))?;
        b.push(ops::v2(V2::AddF32, sum, sum, other))?;
    }
    b.push(ops::v2(V2::LshrrevB32, offset, Src::Int(5), lane))?;
    b.push(ops::v2(V2::LshlrevB32, offset, Src::Int(2), offset))?;
    address(&mut b, va, hp, offset)?;
    b.push(ops::v2(V2::AndB32, other, Src::Int(31), lane))?;
    b.push(ops::vc(VC::CmpEqU32, Src::VccLo, Src::Int(0), other))?;
    b.push(ops::s1(S1::AndSaveexecB64, saved_exec, Src::VccLo))?;
    b.push(ops::global_store(FlatOp::StoreDword, va, sum, None, 0))?;
    b.wait_all()?;
    b.push(ops::s1(S1::MovB64, Src::ExecLo, saved_exec))?;
    add_ptr(&mut b, xp, xp, Src::Lit(1024), Src::Int(0))?;
    add_ptr(&mut b, op, op, Src::Lit(256), Src::Int(0))?;
    add_ptr(&mut b, hp, hp, Src::Int(8), Src::Int(0))?;
    b.push(ops::s2(S2::AddU32, count, count, Src::Lit(256)))?;
    b.push(ops::sc(SC::CmpLtU32, count, k))?;
    b.wait_all()?;
    b.branch(SP::CbranchScc1, "pack_loop")?;
    b.label("done")?;
    b.finish()
}

/// Independent CPU outputs; no emulator or `peacemaker_cdna::float` dependency.
#[derive(Clone, Debug)]
pub struct Reference {
    pub codes: Vec<u8>,
    pub half_sums: Vec<f32>,
    pub row_scales: Vec<f32>,
}

// Exact f64 image of the positive OCP grid, extended with code 127's FNUZ
// finite value times two (480). Every value here is a small integer × 2^n.
fn ocp_image(code: u8) -> f64 {
    let e = i32::from(code >> 3);
    let m = i32::from(code & 7);
    if e == 0 {
        f64::from(m) / 512.0
    } else {
        f64::from(8 + m) * 2.0f64.powi(e - 10)
    }
}

fn encode_image(value: f64) -> u8 {
    let x = value.abs();
    // MODE.FP16_OVFL=0 tests the unrounded source magnitude (ISA §7.2).
    if x > 480.0 {
        return 0x80;
    }
    let (mut lo, mut hi) = (0u8, 127u8);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if ocp_image(mid) <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let lower_error = x - ocp_image(lo);
    let upper_error = ocp_image(hi) - x;
    let best = if upper_error < lower_error || (upper_error == lower_error && hi & 1 == 0) {
        hi
    } else {
        lo
    };
    if best == 0 {
        0
    } else {
        best | if value.is_sign_negative() { 128 } else { 0 }
    }
}

fn decode_reference(code: u8) -> f32 {
    let magnitude = code & 127;
    let value = if magnitude == 127 {
        f32::from_bits(0x7fc0_0000)
    } else {
        ocp_image(magnitude) as f32
    };
    if code & 128 != 0 {
        -value
    } else {
        value
    }
}

/// Source-exact scale selection and exact rational/f64 OCP→FNUZ image rounding.
/// `x.len()/k` rows; supported source-route shifts are -1..=3.
pub fn reference(
    x: &[f32],
    k: usize,
    scale_mode: u32,
    row_scale_shift: i32,
) -> Result<Reference, String> {
    if k == 0 || k % 256 != 0 || x.len() % k != 0 {
        return Err("K must be a positive multiple of 256 and divide the input length".into());
    }
    if !(-1..=3).contains(&row_scale_shift) || x.iter().any(|v| !v.is_finite()) {
        return Err("finite inputs and a source-route shift in -1..=3 are required".into());
    }
    let mut result = Reference {
        codes: Vec::with_capacity(x.len()),
        half_sums: Vec::with_capacity(x.len() / 128),
        row_scales: Vec::with_capacity(x.len() / k),
    };
    for row in x.chunks_exact(k) {
        let mut e = 0;
        if scale_mode == 1 {
            let amax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            if amax > 0.0 {
                // f64 product is exact for two f32 significands; cast rounds
                // once to f32, exactly as the source's v_mul_f32.
                let t = (f64::from(amax) * f64::from(1.0f32 / 224.0)) as f32;
                let bits = t.to_bits();
                e = (((bits >> 23) as i32 - 127) + i32::from(bits & 0x7fffff != 0)).clamp(-31, 9);
            }
            e += row_scale_shift;
        }
        result
            .row_scales
            .push(f32::from_bits(((e + 127) as u32) << 23));
        let start = result.codes.len();
        let inverse_scale = 2.0f64.powi(-e);
        for &v in row {
            // Division by a and the FNUZ half fold are exact in f64 for all
            // finite f32 inputs in this exponent range. Encoding uses the
            // twice-FNUZ (OCP) image, so the explicit half cancels here.
            result
                .codes
                .push(encode_image(f64::from(v) * inverse_scale));
        }
        for half in result.codes[start..].chunks_exact(128) {
            let mut lanes = [0.0f32; 32];
            for (lane, four) in half.chunks_exact(4).enumerate() {
                lanes[lane] = ((decode_reference(four[0]) + decode_reference(four[1]))
                    + decode_reference(four[2]))
                    + decode_reference(four[3]);
            }
            for off in [16, 8, 4, 2, 1] {
                let old = lanes;
                for lane in 0..32 {
                    lanes[lane] = old[lane] + old[lane ^ off];
                }
            }
            result.half_sums.push(lanes[0]);
        }
    }
    Ok(result)
}
