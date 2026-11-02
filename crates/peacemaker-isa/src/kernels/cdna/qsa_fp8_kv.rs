// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 QSA FP8 KV-cache row codec for gfx942 (MI300X, CDNA3, wave64): the
//! PeaceMaker encoder and decoder, their independent CPU references and the
//! contract of the HIP twins.
//!
//! Source of truth: `qsa_kv_store<QSA_KV_FP8>` (`kernels/src/tensor_ops.hip`,
//! "E1" of route R4), which is `gfx1201`-only (OCP E4M3 hardware converts).
//! gfx942 has FNUZ E4M3 converts instead; this module keeps the stored bytes
//! of the source format and moves only the arithmetic.
//!
//! # Stored row (unchanged from gfx1201)
//!
//! One K *or* V row of `kv_heads` (2 or 4) heads of `head_dim` = 256 channels
//! is `kv_heads * 256` code bytes (head-major, channel-minor) followed by
//! `kv_heads` little-endian F16 scales, one per head: `258 * kv_heads` bytes
//! ([`row_bytes`]). A code byte is the **OCP E4M3** byte of `x / scale`
//! (bias 7, max 448); the value stored for a channel is
//! `ocp(code) * f32(scale)`.
//!
//! # Encoder numerics (per head, identical to the source)
//!
//! 1. `amax` = max `|x|` over the head's 256 channels (exact: max commutes, so
//!    the reduction tree is free; the source's eight width-32 groups and this
//!    kernel's DPP/LDS tree give the same bits for finite input).
//! 2. `s16 = amax == 0 ? F16(1) : F16_RNE(amax / 448.0f)` with an IEEE F32
//!    division; positive `amax` whose scale rounds to zero gets bits `0x0001`;
//!    if `448 * f32(s16) < amax` (the product is exact in F32) the F16 bits
//!    are incremented **once** (an `if`, not a loop), capped at `0x7bff`.
//! 3. `z = x / f32(s16)`: IEEE F32 division (not `x * (1/s)`).
//! 4. FNUZ conversion of the **half** value: `code = cvt_pk_fp8_f32(z * 0.5)`
//!    (RNE). FNUZ E4M3 has bias 8 and max 240, so the FNUZ byte of `z/2` has
//!    the same bits as the OCP byte of `z` for every finite nonzero `z` with
//!    `|z| <= 448` (normals and subnormals: the 2x exponent shift is exact and
//!    RNE ties agree). `z * 0.5` is exact in that range.
//! 5. The only byte that differs from the OCP source is negative zero: OCP
//!    `0x80` (input `-0`, or a negative value rounding to zero) is stored as
//!    `0x00`, because FNUZ `0x80` is NaN. The kernel therefore folds a `0x80`
//!    result of step 4 to `0x00` (a no-op if the hardware already returns
//!    `0x00` for -0). Consequently a stored `0x80` is never produced.
//!
//! The F32 divisions are emitted exactly as gfx942 clang lowers an IEEE
//! `fdiv` (`v_div_scale_f32` x2, `v_rcp_f32`, `v_fma_f32` x5, `v_mul_f32`,
//! `v_div_fmas_f32`, `v_div_fixup_f32`; probed with `hipcc -O3
//! --offload-arch=gfx942` on the source expressions); that sequence is
//! correctly rounded, which is what the reference's native `f32 /` is.
//!
//! # Decoder numerics
//!
//! `y = cvt_f32_fp8(code) * (2 * f32(s16))` where `cvt_f32_fp8` is the FNUZ
//! decode. FNUZ decodes the OCP byte to half its OCP value, so `y` equals the
//! OCP reconstruction `ocp(code) * f32(s16)`; `2 * f32(s16)` is formed in
//! **F32** (twice 65504 is a legal F32). Every product is exact (4-bit by
//! 11-bit significands), so the result is independent of rounding mode. Byte
//! `0x80` decodes to NaN (FNUZ NaN); the encoder never emits it.
//!
//! # Domain
//!
//! Finite inputs with `amax <= 448 * 65504`: above that the capped single
//! bump no longer guarantees `|z| <= 448` (the source has the same limit).
//! Non-finite input and the ceiling are errors in the CPU reference and
//! undefined (unspecified bytes) in the kernels.
//!
//! # Kernel ABI (PM kernels; the HIP twins use the same layout)
//!
//! Block 256 (four wave64s), grid `(kv_heads, rows, 1)`: `workgroup_id.x` is
//! the head, `workgroup_id.y` the row. One tensor (K or V) per launch, rows
//! contiguous.
//!
//! | kernel | kernarg 0 (offset 0) | kernarg 1 (offset 8) |
//! |---|---|---|
//! | [`encode_symbol`] | `dst`: `u8*` stored rows, row stride [`row_bytes`] | `src`: `const f32*`, row stride `kv_heads * 256` floats |
//! | [`decode_symbol`] | `dst`: `f32*`, row stride `kv_heads * 256` floats | `src`: `const u8*` stored rows |
//!
//! The encoder uses 64 bytes of LDS (sixteen 16-lane row maxima) and one
//! barrier. Per head: global load; `|x|`; DPP max over quad, 8-lane half row
//! and 16-lane row (`quad_perm` x2, `row_half_mirror`, `row_mirror`); each
//! 16-lane row publishes its maximum to LDS slot `tid >> 4`; barrier; every
//! thread maxes the sixteen slots; scale, division, convert, byte stores; the
//! scale is stored by thread 0 only (EXEC masked).
//!
//! # Required mode state (kernel descriptor)
//!
//! F32 and F16/F64 denormals preserved (`float_denorm_mode_32 = 3`,
//! `float_denorm_mode_16_64 = 3`), round-to-nearest-even, `ieee_mode = 1`: the
//! values `hipcc` gives these kernels by default. `FP16_OVFL` is not used
//! (finite domain; the converts never overflow there).
//!
//! # Unverified on hardware
//!
//! `[UNVERIFIED: confirm on MI300X]`
//! - `v_cvt_pk_fp8_f32` of a value that rounds to (negative) zero: whether the
//!   byte is `0x00` or `0x80`. The kernel is correct either way (it folds
//!   `0x80` to `0x00`); the CPU reference states the intent.
//! - The FNUZ-half identity of step 4 is a derivation, checked here by the
//!   reference against an independent bit-level FNUZ encoder, not yet
//!   exercised on a device against `cvt_pk_fp8_f32` / the gfx1201 bytes.
//! - `v_cvt_f32_fp8` without SDWA selecting byte 0 (the assembler accepts the
//!   `_e32` form; the byte-0 semantics are taken from the SDWA default).
//! - Out-of-domain behavior (NaN/inf inputs, `amax` above the ceiling) is not
//!   modeled by the hardware path.
//! - Division. The CPU reference divides with native IEEE `f32 /` (correctly
//!   rounded). The kernels reproduce it only if the hardware sequence is
//!   correctly rounded for every operand pair used here, which relies on
//!   (a) `v_rcp_f32` accuracy (the sequence's two Newton steps assume a
//!   reciprocal within 1 ulp; the emulator models a correctly rounded rcp that
//!   flushes denormal input/output per the ISA, CDNA3 ISA p.188) and (b) the
//!   `v_div_scale_f32` / `v_div_fmas_f32` exponent scaling at extreme exponent
//!   differences (the emulator models a 2^±64 post-scale of `v_div_fmas_f32`
//!   from the scaled-operand exponent; the ISA text says 2^32). Operand pairs
//!   with exponent difference >= 96 or a denormal quotient are the exposed
//!   cases; here `x / s16` with `|x| <= amax <= 448 * s16` has a quotient in
//!   `[2^-149 .., 448]`, so denormal quotients (|x| tiny against the head's
//!   scale: they round to code 0 either way, `-0` folded) and the exponent
//!   scaling path are reachable only for inputs below roughly `amax * 2^-126 / 448`.
//!   Byte equality with the reference is therefore certain for quotients in
//!   the normal F32 range and is `[UNVERIFIED]` outside it; the emitted code
//!   byte is unaffected there because such quotients round to zero in FP8.
use crate::cdna::builder::Builder;
use crate::cdna::ops::{self, Mods};
use crate::cdna::Kernel;
use crate::Arch;
use peacemaker_cdna::enc::{
    ds::DsOp,
    flat::FlatOp,
    smem::SmemOp,
    sop::{Sop1Op, Sop2Op},
    vop::{DppCtrl, Vop1Op, Vop2Op, VopcOp},
    vop3::{Vop3OnlyOp, Vop3Op},
};
use peacemaker_cdna::operand::InlineFloat;
use peacemaker_cdna::Src;

type R<T> = Result<T, String>;

/// Channels per head (the only supported head dimension).
pub const HEAD_DIM: usize = 256;
/// Threads per block: one per channel.
pub const BLOCK: u32 = 256;
/// LDS bytes the encoder uses: sixteen F32 row maxima.
pub const ENCODE_LDS_BYTES: u32 = 64;

/// Stored bytes of one K or V row: `kv_heads * 256` codes plus `kv_heads`
/// little-endian F16 scales.
pub fn row_bytes(kv_heads: usize) -> usize {
    kv_heads * (HEAD_DIM + 2)
}

/// Symbol of the encoder for `kv_heads` KV heads.
pub fn encode_symbol(kv_heads: usize) -> String {
    format!("qsa_kv_encode_fp8_h{kv_heads}_pm_gfx942")
}
/// Symbol of the decoder for `kv_heads` KV heads.
pub fn decode_symbol(kv_heads: usize) -> String {
    format!("qsa_kv_decode_fp8_h{kv_heads}_pm_gfx942")
}

/// Launch grid for `rows` rows: `(kv_heads, rows, 1)` workgroups of [`BLOCK`] threads.
pub fn grid(kv_heads: usize, rows: usize) -> [u32; 3] {
    [kv_heads as u32, rows as u32, 1]
}

fn check_geometry(kv_heads: usize) -> R<()> {
    if kv_heads == 2 || kv_heads == 4 {
        Ok(())
    } else {
        Err(format!(
            "qsa fp8 kv: kv_heads must be 2 or 4 (head_dim 256, even heads), got {kv_heads}"
        ))
    }
}

// ───────────────────────────── CPU reference ─────────────────────────────
//
// Independent of the builder, the emulator and `peacemaker_cdna::float`:
// plain f32/f64 arithmetic and integer bit manipulation only.

/// Exact F32 -> F16 conversion (RNE), the bits of `_Float16(x)`.
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x007f_ffff;
    if exp == 0xff {
        return sign | if man == 0 { 0x7c00 } else { 0x7e00 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Subnormal F16 result (F32 zero and subnormals land here with e <= -112).
        let shift = (14 - e) as u32;
        if shift >= 25 {
            return sign;
        }
        let m24 = man | 0x0080_0000;
        let q = m24 >> shift;
        let rem = m24 & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        let up = rem > half || (rem == half && (q & 1) == 1);
        return sign | (q + u32::from(up)) as u16;
    }
    let q = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let up = rem > 0x1000 || (rem == 0x1000 && (q & 1) == 1);
    sign | (q + u32::from(up)) as u16
}

/// Exact F16 -> F32 conversion.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits) & 0x8000) << 16;
    let exp = u32::from(bits >> 10) & 0x1f;
    let man = u32::from(bits) & 0x3ff;
    if exp == 0 {
        // man * 2^-24, exact in F32.
        let mag = man as f32 * f32::from_bits(0x3380_0000);
        return f32::from_bits(sign | mag.to_bits());
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (man << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (man << 13))
}

/// The F16 scale bits of a head whose largest `|x|` is `amax` (>= 0).
///
/// `amax == 0` gives `0x3c00`; otherwise RNE F16 of `amax / 448`, a rounded
/// zero becomes `0x0001`, and one conditional bump (cap `0x7bff`) makes
/// `448 * f32(scale) >= amax`.
pub fn reference_scale_bits(amax: f32) -> u16 {
    let mut bits = if amax == 0.0 {
        0x3c00
    } else {
        f32_to_f16_bits(amax / 448.0)
    };
    if amax > 0.0 && bits == 0 {
        bits = 0x0001;
    }
    let sf = f16_bits_to_f32(bits);
    if amax > 0.0 && 448.0 * sf < amax {
        bits = if bits >= 0x7bff { 0x7bff } else { bits + 1 };
    }
    bits
}

/// Round a nonnegative f64 to the nearest integer, ties to even.
fn round_ties_even(x: f64) -> f64 {
    let f = x.floor();
    let d = x - f;
    if d > 0.5 {
        f + 1.0
    } else if d < 0.5 {
        f
    } else if (f as u64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

/// FNUZ E4M3 (bias 8, max 240, `0x80` = NaN) encoding of a finite `h`, RNE,
/// with negative zero canonicalized to `0x00`. `Err` when the magnitude rounds
/// above 240 (hardware overflow behavior is not modeled).
fn fnuz_e4m3_encode(h: f32) -> R<u8> {
    let a = f64::from(h.abs());
    if a == 0.0 {
        return Ok(0x00);
    }
    let mag: u64 = if a < 2f64.powi(-7) {
        // Subnormal: multiples of 2^-10; 8 is the smallest normal (e=1, m=0).
        round_ties_even(a * 1024.0) as u64
    } else {
        let exp = ((a.to_bits() >> 52) & 0x7ff) as i64 - 1023; // a >= 2^-7: normal f64
        let n = round_ties_even(a * 2f64.powi((3 - exp) as i32)) as u64; // 8..=16
        (((exp + 8) as u64) << 3) + n - 8
    };
    if mag > 0x7f {
        return Err(format!("fp8 fnuz overflow: |{h}| rounds above 240"));
    }
    if mag == 0 {
        return Ok(0x00);
    }
    Ok((u8::from(h.is_sign_negative()) << 7) | mag as u8)
}

/// FNUZ E4M3 byte to F32 (exact). `0x80` is NaN.
fn fnuz_e4m3_decode(code: u8) -> f32 {
    if code == 0x80 {
        return f32::NAN;
    }
    let exp = i32::from((code >> 3) & 0xf);
    let man = f32::from(code & 7);
    let mag = if exp == 0 {
        man * f32::from_bits(0x3a80_0000) // 2^-10
    } else {
        (8.0 + man) * f32::from_bits(((127 + exp - 11) as u32) << 23)
    };
    if code & 0x80 != 0 {
        -mag
    } else {
        mag
    }
}

/// Encode one head: `src` is the head's 256 channels. Returns the OCP-valued
/// code bytes and the F16 scale bits. `Err` for non-finite input or an `amax`
/// above `448 * 65504`.
pub fn reference_encode_head(src: &[f32; HEAD_DIM]) -> R<([u8; HEAD_DIM], u16)> {
    if let Some(bad) = src.iter().find(|v| !v.is_finite()) {
        return Err(format!("qsa fp8 kv encode: non-finite input {bad}"));
    }
    let amax = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = reference_scale_bits(amax);
    let sf = f16_bits_to_f32(scale);
    if 448.0 * sf < amax {
        return Err(format!(
            "qsa fp8 kv encode: amax {amax} exceeds the scale ceiling 448*65504"
        ));
    }
    let mut codes = [0u8; HEAD_DIM];
    for (code, &x) in codes.iter_mut().zip(src.iter()) {
        let z = x / sf;
        *code = fnuz_e4m3_encode(z * 0.5)?;
    }
    Ok((codes, scale))
}

/// Decode one head: `code * (2 * f32(scale))` per channel, `0x80` decodes to NaN.
pub fn reference_decode_head(codes: &[u8; HEAD_DIM], scale_bits: u16) -> [f32; HEAD_DIM] {
    let s2 = 2.0 * f16_bits_to_f32(scale_bits);
    let mut out = [0.0f32; HEAD_DIM];
    for (y, &c) in out.iter_mut().zip(codes.iter()) {
        *y = fnuz_e4m3_decode(c) * s2;
    }
    out
}

/// Encode `src` (`rows * kv_heads * 256` F32, row-major) into stored rows
/// (`rows * row_bytes(kv_heads)` bytes: per row all codes then the F16 scales).
pub fn reference_encode(kv_heads: usize, src: &[f32]) -> R<Vec<u8>> {
    check_geometry(kv_heads)?;
    let per_row = kv_heads * HEAD_DIM;
    if src.len() % per_row != 0 {
        return Err(format!(
            "qsa fp8 kv encode: {} values is not a whole number of {per_row}-value rows",
            src.len()
        ));
    }
    let rows = src.len() / per_row;
    let rb = row_bytes(kv_heads);
    let mut out = vec![0u8; rows * rb];
    for row in 0..rows {
        for head in 0..kv_heads {
            let at = row * per_row + head * HEAD_DIM;
            let head_src: &[f32; HEAD_DIM] = src[at..at + HEAD_DIM]
                .try_into()
                .expect("slice of HEAD_DIM");
            let (codes, scale) = reference_encode_head(head_src)?;
            let o = row * rb;
            out[o + head * HEAD_DIM..o + (head + 1) * HEAD_DIM].copy_from_slice(&codes);
            let so = o + kv_heads * HEAD_DIM + 2 * head;
            out[so..so + 2].copy_from_slice(&scale.to_le_bytes());
        }
    }
    Ok(out)
}

/// Decode stored rows (`rows * row_bytes(kv_heads)` bytes) to F32
/// (`rows * kv_heads * 256`).
pub fn reference_decode(kv_heads: usize, stored: &[u8]) -> R<Vec<f32>> {
    check_geometry(kv_heads)?;
    let rb = row_bytes(kv_heads);
    if stored.len() % rb != 0 {
        return Err(format!(
            "qsa fp8 kv decode: {} bytes is not a whole number of {rb}-byte rows",
            stored.len()
        ));
    }
    let rows = stored.len() / rb;
    let mut out = Vec::with_capacity(rows * kv_heads * HEAD_DIM);
    for row in 0..rows {
        let o = row * rb;
        for head in 0..kv_heads {
            let codes: &[u8; HEAD_DIM] = stored[o + head * HEAD_DIM..o + (head + 1) * HEAD_DIM]
                .try_into()
                .expect("slice of HEAD_DIM");
            let so = o + kv_heads * HEAD_DIM + 2 * head;
            let scale = u16::from_le_bytes([stored[so], stored[so + 1]]);
            out.extend_from_slice(&reference_decode_head(codes, scale));
        }
    }
    Ok(out)
}

// ───────────────────────────── PM kernels ────────────────────────────────

const F32_448: u32 = 0x43e0_0000;
const F16_ONE: u32 = 0x3c00;
const F16_MAX_BUMP_FROM: u32 = 0x7bfe;
const MASK_16: u32 = 0xffff;

fn sg(base: Src, k: u8) -> Src {
    match base {
        Src::Sgpr(n) => Src::Sgpr(n + k),
        other => panic!("sg: {other:?} is not an SGPR"),
    }
}
fn vg(base: Src, k: u8) -> Src {
    match base {
        Src::Vgpr(n) => Src::Vgpr(n + k),
        other => panic!("vg: {other:?} is not a VGPR"),
    }
}
fn fma_neg0() -> Mods {
    Mods {
        neg: 0b001,
        ..Mods::default()
    }
}
fn fma() -> Vop3Op {
    Vop3Op::Only(Vop3OnlyOp::FmaF32)
}

/// `dst = base + off` (64-bit pointer plus 32-bit unsigned byte offset).
fn ptr_add(b: &mut Builder, dst: Src, base: Src, off: Src) -> R<()> {
    b.push(ops::s2(Sop2Op::AddU32, dst, base, off))?;
    b.push(ops::s2(
        Sop2Op::AddcU32,
        sg(dst, 1),
        sg(base, 1),
        Src::Int(0),
    ))
}

/// Scratch registers of one IEEE F32 division.
struct DivRegs {
    scaled_den: Src,
    recip: Src,
    scaled_num: Src,
    err: Src,
    quot: Src,
    dummy_sdst: Src,
}

/// `q = num / den` in IEEE F32, the sequence gfx942 clang emits for `fdiv`
/// with F32 denormals preserved. `den` and `num` are VGPRs or one SGPR/inline
/// (constant-bus limit: at most one distinct SGPR across `den`/`num`); the
/// result is written to `q`. `v_div_fmas_f32` reads VCC from the numerator
/// `v_div_scale_f32`.
fn emit_div(b: &mut Builder, q: Src, num: Src, den: Src, w: &DivRegs) -> R<()> {
    let one = Src::Float(InlineFloat::One);
    b.push(ops::v3b(
        Vop3OnlyOp::DivScaleF32,
        w.scaled_den,
        w.dummy_sdst,
        &[den, den, num],
    ))?;
    b.push(ops::v1(Vop1Op::RcpF32, w.recip, w.scaled_den))?;
    b.push(ops::v3b(
        Vop3OnlyOp::DivScaleF32,
        w.scaled_num,
        Src::VccLo,
        &[num, den, num],
    ))?;
    b.push(ops::v3m(
        fma(),
        w.err,
        &[w.scaled_den, w.recip, one],
        fma_neg0(),
    ))?; //  e = fma(-d, r, 1)
    b.push(ops::v3m(
        fma(),
        w.recip,
        &[w.err, w.recip, w.recip],
        Mods::default(),
    ))?; //  r = fma(e, r, r)
    b.push(ops::v2(Vop2Op::MulF32, w.quot, w.scaled_num, w.recip))?; //  y = n * r
    b.push(ops::v3m(
        fma(),
        w.err,
        &[w.scaled_den, w.quot, w.scaled_num],
        fma_neg0(),
    ))?; //  e = fma(-d, y, n)
    b.push(ops::v3m(
        fma(),
        w.quot,
        &[w.err, w.recip, w.quot],
        Mods::default(),
    ))?; //  y = fma(e, r, y)
    b.push(ops::v3m(
        fma(),
        w.err,
        &[w.scaled_den, w.quot, w.scaled_num],
        fma_neg0(),
    ))?; //  e = fma(-d, y, n)
    b.push(ops::v3(
        Vop3OnlyOp::DivFmasF32,
        w.quot,
        &[w.err, w.recip, w.quot],
    ))?;
    b.push(ops::v3(Vop3OnlyOp::DivFixupF32, q, &[w.quot, den, num]))
}

/// The encoder: one head of one row per workgroup, see the module docs.
pub fn build_encode(arch: Arch, kv_heads: usize) -> R<Kernel> {
    check_geometry(kv_heads)?;
    let heads = kv_heads as u32;
    let rb = row_bytes(kv_heads) as u32;
    let scale_imm = i16::try_from(kv_heads * HEAD_DIM)
        .map_err(|_| "scale offset exceeds the immediate".to_string())?;
    let mut b = Builder::new(arch, &encode_symbol(kv_heads), BLOCK)?;
    let a_dst = b.arg_ptr("dst", false)?;
    let a_src = b.arg_ptr("src", true)?;
    let wg_head = b.workgroup_id(0)?;
    let wg_row = b.workgroup_id(1)?;
    let v0 = b.workitem_ids(1)?;
    let lds = b.lds(ENCODE_LDS_BYTES, 16)?;
    let lds_off = |k: u32| {
        u16::try_from(lds.offset + k).map_err(|_| "LDS offset exceeds the DS immediate".to_string())
    };

    // SGPRs.
    let dstp = b.sgprs(2, 2)?;
    let srcp = b.sgprs(2, 2)?;
    let cptr = b.sgprs(2, 2)?; // dst + row*rb + head*256        (code bytes)
    let sptr = b.sgprs(2, 2)?; // dst + row*rb + head*2          (+ heads*256 immediate: scale)
    let xptr = b.sgprs(2, 2)?; // src + (row*heads + head)*1024  (f32 values)
    let dummy = b.sgprs(2, 2)?; // discarded `v_div_scale` carry flag
    let m_gt = b.sgprs(2, 2)?; // amax > 0
    let m_eq = b.sgprs(2, 2)?; // amax == 0
    let m_t0 = b.sgprs(2, 2)?;
    let m_t1 = b.sgprs(2, 2)?;
    let t = b.sgprs(4, 4)?;
    let k448 = b.sgprs(1, 1)?;
    // VGPRs.
    let tid = b.vgprs(1, 1)?;
    let zero = b.vgprs(1, 1)?;
    let off4 = b.vgprs(1, 1)?;
    let slot = b.vgprs(1, 1)?;
    let x = b.vgprs(1, 1)?;
    let amax = b.vgprs(1, 1)?;
    let rows16 = b.vgprs(16, 4)?;
    let bits = b.vgprs(1, 1)?;
    let alt = b.vgprs(1, 1)?;
    let one16 = b.vgprs(1, 1)?;
    let sf = b.vgprs(1, 1)?;
    let prod = b.vgprs(1, 1)?;
    let z = b.vgprs(1, 1)?;
    let word = b.vgprs(1, 1)?;
    let w = DivRegs {
        scaled_den: b.vgprs(1, 1)?,
        recip: b.vgprs(1, 1)?,
        scaled_num: b.vgprs(1, 1)?,
        err: b.vgprs(1, 1)?,
        quot: b.vgprs(1, 1)?,
        dummy_sdst: dummy,
    };

    // Kernel arguments and pointers.
    let kbase = b.kernarg_base();
    b.push(ops::s_load(SmemOp::LoadDwordx2, dstp, kbase, a_dst as i32))?;
    b.push(ops::s_load(SmemOp::LoadDwordx2, srcp, kbase, a_src as i32))?;
    b.push(ops::s1(Sop1Op::MovB32, k448, Src::Lit(F32_448)))?;
    // dst + row*rb + head*256 and + head*2.
    b.push(ops::s2(Sop2Op::MulI32, t, wg_row, Src::Lit(rb)))?;
    b.push(ops::s2(Sop2Op::LshlB32, sg(t, 1), wg_head, Src::Int(8)))?;
    b.push(ops::s2(Sop2Op::AddU32, sg(t, 1), t, sg(t, 1)))?; //  row*rb + head*256
    b.push(ops::s2(Sop2Op::LshlB32, sg(t, 2), wg_head, Src::Int(1)))?;
    b.push(ops::s2(Sop2Op::AddU32, sg(t, 2), t, sg(t, 2)))?; //  row*rb + head*2
    ptr_add(&mut b, cptr, dstp, sg(t, 1))?;
    ptr_add(&mut b, sptr, dstp, sg(t, 2))?;
    // src + (row*heads + head)*1024.
    b.push(ops::s2(Sop2Op::MulI32, t, wg_row, Src::Lit(heads * 1024)))?;
    b.push(ops::s2(Sop2Op::LshlB32, sg(t, 1), wg_head, Src::Int(10)))?;
    b.push(ops::s2(Sop2Op::AddU32, sg(t, 1), t, sg(t, 1)))?;
    ptr_add(&mut b, xptr, srcp, sg(t, 1))?;

    // Thread ids and constants.
    b.push(ops::v2(Vop2Op::AndB32, tid, Src::Lit(0x3ff), v0))?;
    b.push(ops::v1(Vop1Op::MovB32, zero, Src::Int(0)))?;
    b.push(ops::v1(Vop1Op::MovB32, one16, Src::Lit(F16_ONE)))?;
    b.push(ops::v2(Vop2Op::LshlrevB32, off4, Src::Int(2), tid))?;
    b.push(ops::v2(Vop2Op::LshrrevB32, slot, Src::Int(4), tid))?;
    b.push(ops::v2(Vop2Op::LshlrevB32, slot, Src::Int(2), slot))?; //  LDS byte address of the 16-lane row slot

    // x = src[head][tid]; amax of the head.
    b.push(ops::global_load(FlatOp::LoadDword, x, off4, Some(xptr), 0))?;
    b.push(ops::v3m(
        Vop3Op::V2(Vop2Op::MaxF32),
        amax,
        &[x, x],
        Mods {
            abs: 0b011,
            ..Mods::default()
        },
    ))?; //  |x|
    for ctrl in [
        DppCtrl::QuadPerm([1, 0, 3, 2]),
        DppCtrl::QuadPerm([2, 3, 0, 1]),
        DppCtrl::RowHalfMirror,
        DppCtrl::RowMirror,
    ] {
        b.push(ops::dpp2(
            Vop2Op::MaxF32,
            amax,
            amax,
            amax,
            ctrl,
            0xf,
            0xf,
            false,
        ))?;
    }
    // Each 16-lane row holds its maximum in every lane; lanes of a row write the same value to one slot.
    b.push(ops::ds_write(DsOp::WriteB32, slot, amax, lds_off(0)?))?;
    b.barrier()?;
    for i in 0..4u8 {
        b.push(ops::ds_read(
            DsOp::ReadB128,
            vg(rows16, 4 * i),
            zero,
            lds_off(16 * u32::from(i))?,
        ))?;
    }
    b.push(ops::v2(Vop2Op::MaxF32, amax, rows16, vg(rows16, 1)))?;
    for i in 2..16u8 {
        b.push(ops::v2(Vop2Op::MaxF32, amax, vg(rows16, i), amax))?;
    }

    // s16 bits.
    emit_div(&mut b, z, amax, k448, &w)?; //  amax / 448
    b.push(ops::v1(Vop1Op::CvtF16F32, bits, z))?;
    b.push(ops::v2(Vop2Op::AndB32, bits, Src::Lit(MASK_16), bits))?;
    b.push(ops::vc(VopcOp::CmpLtF32, m_gt, Src::Int(0), amax))?; //  amax > 0
    b.push(ops::vc(VopcOp::CmpEqF32, m_eq, Src::Int(0), amax))?; //  amax == 0
    b.push(ops::vc(VopcOp::CmpEqU32, m_t0, Src::Int(0), bits))?; //  scale rounded to zero
    b.push(ops::s2(Sop2Op::AndB64, m_t0, m_t0, m_gt))?;
    b.push(ops::v2ci(
        Vop2Op::CndmaskB32,
        bits,
        None,
        bits,
        Src::Int(1),
        m_t0,
    ))?; //  0 -> 0x0001 (amax > 0)
    b.push(ops::v2ci(Vop2Op::CndmaskB32, bits, None, bits, one16, m_eq))?; //  amax == 0 -> F16(1)
    b.push(ops::v1(Vop1Op::CvtF32F16, sf, bits))?;
    b.push(ops::v2(Vop2Op::MulF32, prod, Src::Lit(F32_448), sf))?; //  448 * sf, exact
    b.push(ops::vc(VopcOp::CmpLtF32, m_t1, prod, amax))?;
    b.push(ops::s2(Sop2Op::AndB64, m_t1, m_t1, m_gt))?;
    b.push(ops::v2(
        Vop2Op::MinU32,
        alt,
        Src::Lit(F16_MAX_BUMP_FROM),
        bits,
    ))?;
    b.push(ops::v2(Vop2Op::AddU32, alt, Src::Int(1), alt))?; //  bits >= 0x7bff ? 0x7bff : bits + 1
    b.push(ops::v2ci(Vop2Op::CndmaskB32, bits, None, bits, alt, m_t1))?; //  single conditional ulp bump
    b.push(ops::v1(Vop1Op::CvtF32F16, sf, bits))?;

    // code = FNUZ(x / sf * 0.5), -0 folded to +0.
    emit_div(&mut b, z, x, sf, &w)?;
    b.push(ops::v2(Vop2Op::MulF32, z, Src::Float(InlineFloat::Half), z))?;
    b.push(ops::v1(Vop1Op::MovB32, word, Src::Int(0)))?;
    b.push(ops::v3(Vop3OnlyOp::CvtPkFp8F32, word, &[z, zero]))?; //  src1 must be a VGPR: byte 1 = fp8(0) = 0
    b.push(ops::vc(VopcOp::CmpEqU32, Src::VccLo, Src::Lit(0x80), word))?;
    b.push(ops::v2ci(
        Vop2Op::CndmaskB32,
        word,
        None,
        word,
        zero,
        Src::VccLo,
    ))?;
    b.push(ops::global_store(
        FlatOp::StoreByte,
        tid,
        word,
        Some(cptr),
        0,
    ))?;

    // Thread 0 stores the scale (EXEC = lane 0 of wave 0; the kernel ends after the store).
    b.push(ops::vc(VopcOp::CmpEqU32, Src::VccLo, Src::Int(0), tid))?;
    b.push(ops::s1(Sop1Op::MovB64, Src::ExecLo, Src::VccLo))?;
    b.push(ops::global_store(
        FlatOp::StoreShort,
        zero,
        bits,
        Some(sptr),
        scale_imm,
    ))?;
    b.finish()
}

/// The decoder: one head of one row per workgroup, see the module docs.
pub fn build_decode(arch: Arch, kv_heads: usize) -> R<Kernel> {
    check_geometry(kv_heads)?;
    let heads = kv_heads as u32;
    let rb = row_bytes(kv_heads) as u32;
    let scale_imm = i16::try_from(kv_heads * HEAD_DIM)
        .map_err(|_| "scale offset exceeds the immediate".to_string())?;
    let mut b = Builder::new(arch, &decode_symbol(kv_heads), BLOCK)?;
    let a_dst = b.arg_ptr("dst", false)?;
    let a_src = b.arg_ptr("src", true)?;
    let wg_head = b.workgroup_id(0)?;
    let wg_row = b.workgroup_id(1)?;
    let v0 = b.workitem_ids(1)?;

    let dstp = b.sgprs(2, 2)?;
    let srcp = b.sgprs(2, 2)?;
    let cptr = b.sgprs(2, 2)?; // src + row*rb + head*256 (codes)
    let sptr = b.sgprs(2, 2)?; // src + row*rb + head*2   (+ heads*256: scale)
    let xptr = b.sgprs(2, 2)?; // dst + (row*heads + head)*1024
    let t = b.sgprs(4, 4)?;
    let tid = b.vgprs(1, 1)?;
    let zero = b.vgprs(1, 1)?;
    let off4 = b.vgprs(1, 1)?;
    let code = b.vgprs(1, 1)?;
    let sbits = b.vgprs(1, 1)?;
    let sf = b.vgprs(1, 1)?;
    let s2 = b.vgprs(1, 1)?;
    let val = b.vgprs(1, 1)?;
    let out = b.vgprs(1, 1)?;

    let kbase = b.kernarg_base();
    b.push(ops::s_load(SmemOp::LoadDwordx2, dstp, kbase, a_dst as i32))?;
    b.push(ops::s_load(SmemOp::LoadDwordx2, srcp, kbase, a_src as i32))?;
    b.push(ops::s2(Sop2Op::MulI32, t, wg_row, Src::Lit(rb)))?;
    b.push(ops::s2(Sop2Op::LshlB32, sg(t, 1), wg_head, Src::Int(8)))?;
    b.push(ops::s2(Sop2Op::AddU32, sg(t, 1), t, sg(t, 1)))?;
    b.push(ops::s2(Sop2Op::LshlB32, sg(t, 2), wg_head, Src::Int(1)))?;
    b.push(ops::s2(Sop2Op::AddU32, sg(t, 2), t, sg(t, 2)))?;
    ptr_add(&mut b, cptr, srcp, sg(t, 1))?;
    ptr_add(&mut b, sptr, srcp, sg(t, 2))?;
    b.push(ops::s2(Sop2Op::MulI32, t, wg_row, Src::Lit(heads * 1024)))?;
    b.push(ops::s2(Sop2Op::LshlB32, sg(t, 1), wg_head, Src::Int(10)))?;
    b.push(ops::s2(Sop2Op::AddU32, sg(t, 1), t, sg(t, 1)))?;
    ptr_add(&mut b, xptr, dstp, sg(t, 1))?;

    b.push(ops::v2(Vop2Op::AndB32, tid, Src::Lit(0x3ff), v0))?;
    b.push(ops::v1(Vop1Op::MovB32, zero, Src::Int(0)))?;
    b.push(ops::v2(Vop2Op::LshlrevB32, off4, Src::Int(2), tid))?;
    b.push(ops::global_load(
        FlatOp::LoadUbyte,
        code,
        tid,
        Some(cptr),
        0,
    ))?;
    b.push(ops::global_load(
        FlatOp::LoadUshort,
        sbits,
        zero,
        Some(sptr),
        scale_imm,
    ))?;
    b.push(ops::v1(Vop1Op::CvtF32F16, sf, sbits))?;
    b.push(ops::v2(
        Vop2Op::MulF32,
        s2,
        Src::Float(InlineFloat::Two),
        sf,
    ))?; //  2 * f32(s16), in F32
    b.push(ops::v1(Vop1Op::CvtF32Fp8, val, code))?; //  FNUZ decode of the OCP byte (= half its OCP value)
    b.push(ops::v2(Vop2Op::MulF32, out, s2, val))?;
    b.push(ops::global_store(
        FlatOp::StoreDword,
        off4,
        out,
        Some(xptr),
        0,
    ))?;
    b.finish()
}
