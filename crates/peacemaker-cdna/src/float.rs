// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Small-float conversions, bit-exact: FP8 (E4M3) and BF8 (E5M2) in the FNUZ
//! encoding of CDNA3 (bias 8 / 16, `0x80` = NaN, no -0, no E4M3 infinity;
//! CDNA3 ISA §7.2 Table 30 p.50) and the OCP encoding (E4M3FN bias 7, max 448,
//! NaN `0x7f`/`0xff`; E5M2 bias 15 with infinities), F16 and BF16.
//!
//! F32 → FP8/BF8 follows CDNA3 ISA §7.2 p.50-51: round to nearest even
//! (`v_cvt_pk_*`) or stochastic (`v_cvt_sr_*`, see [`f32_to_f8_sr`]),
//! clamp/omod ignored, and the FP16_OVFL table (§7.2 p.51): with FP16_OVFL=1
//! ±Inf and values of magnitude greater than the format's max saturate to
//! ±max, with FP16_OVFL=0 they become NaN (E4M3) / ±Inf (E5M2; FNUZ has no
//! Inf → NaN `0x80`). NaN → NaN.
//!
//! Policies the hardware may implement differently (each is
//! `[UNVERIFIED: confirm on MI300X]`):
//! - **Overflow threshold.** The FP16_OVFL rule applies to the *source*
//!   magnitude, before rounding: `|x| > max` (the table's row "Greater than
//!   max FP8 magnitude", §7.2 p.51, and AMD's reference `cast_to_f8`,
//!   `/opt/rocm/include/hip/amd_detail/amd_hip_fp8.h:281`). So
//!   `x ∈ (max, max + ½ulp)` overflows here, where IEEE RNE would give `max`.
//! - **Signed zero in FNUZ.** FNUZ has no -0 (`0x80` is NaN): `-0.0` and every
//!   negative value that rounds to zero encode as `0x00` (+0), as in
//!   `amd_hip_fp8.h:379`. OCP keeps the sign (`0x80`).
//! - **NaN codes produced.** FNUZ: `0x80`. OCP: `sign | 0x7f` for both E4M3
//!   and E5M2 (`amd_hip_fp8.h:239`); OCP is not a gfx942 encoding, so this is
//!   only exercised by the CDNA4/RDNA4 variants.
//! - **NaN values decoded.** FNUZ `0x80` → `0x7fc0_0000`; OCP NaN codes →
//!   `sign | 0x7fc0_0000` (payload dropped).
//!
//! F16/BF16 are IEEE binary16 / bfloat16 with subnormals kept (§7.3 p.52:
//! sub-32-bit float types ignore MODE.denorm and do not flush).
use crate::arch::Fp8Encoding;

/// FP8 (E4M3) or BF8 (E5M2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum F8 {
    Fp8,
    Bf8,
}

/// Geometry of one 8-bit format.
#[derive(Clone, Copy)]
struct Fmt {
    /// Stored mantissa bits.
    mbits: u32,
    /// Exponent bias.
    bias: i32,
    /// Largest finite magnitude code (sign bit clear).
    max_code: u8,
}

impl Fmt {
    const fn of(enc: Fp8Encoding, kind: F8) -> Self {
        match (enc, kind) {
            (Fp8Encoding::Fnuz, F8::Fp8) => Fmt {
                mbits: 3,
                bias: 8,
                max_code: 0x7f,
            },
            (Fp8Encoding::Fnuz, F8::Bf8) => Fmt {
                mbits: 2,
                bias: 16,
                max_code: 0x7f,
            },
            (Fp8Encoding::Ocp, F8::Fp8) => Fmt {
                mbits: 3,
                bias: 7,
                max_code: 0x7e,
            },
            (Fp8Encoding::Ocp, F8::Bf8) => Fmt {
                mbits: 2,
                bias: 15,
                max_code: 0x7b,
            },
        }
    }

    /// Minimum normal exponent (unbiased).
    const fn emin(self) -> i32 {
        1 - self.bias
    }

    /// Exact value of the finite magnitude code `mag` (`mag <= max_code`).
    fn magnitude(self, mag: u8) -> f32 {
        let e = i32::from(mag >> self.mbits);
        let m = u32::from(mag) & ((1 << self.mbits) - 1);
        // Both factors and the product are exact in f32: at most 4 significant
        // bits times a power of two in [2^-17, 2^15].
        if e == 0 {
            m as f32 * pow2(self.emin() - self.mbits as i32)
        } else {
            ((1 << self.mbits) | m) as f32 * pow2(e - self.bias - self.mbits as i32)
        }
    }

    /// f32 bits of the largest finite magnitude.
    fn max_bits(self) -> u32 {
        self.magnitude(self.max_code).to_bits()
    }
}

/// `2^k` for a normal-range exponent (`-126 <= k <= 127`).
fn pow2(k: i32) -> f32 {
    debug_assert!((-126..=127).contains(&k));
    f32::from_bits(((k + 127) as u32) << 23)
}

const QNAN32: u32 = 0x7fc0_0000;

/// The value of `code` in `enc` (NaN codes → a quiet f32 NaN).
///
/// Bit layouts per CDNA3 ISA §7.2 Table 30 p.50 (FNUZ) and the OCP 8-bit
/// floating point specification (OCP: E4M3FN has no Inf, `S.1111.111` is
/// NaN; E5M2 is IEEE-like with `S.11111.00` = ±Inf and `S.11111.xx` = NaN).
/// NaN results are `0x7fc0_0000` (FNUZ) and `sign | 0x7fc0_0000` (OCP)
/// `[UNVERIFIED: confirm on MI300X]` (`v_cvt_f32_fp8` NaN bits).
pub fn f8_to_f32(enc: Fp8Encoding, kind: F8, code: u8) -> f32 {
    let f = Fmt::of(enc, kind);
    let sign = u32::from(code >> 7) << 31;
    let mag = code & 0x7f;
    match enc {
        Fp8Encoding::Fnuz => {
            if code == 0x80 {
                return f32::from_bits(QNAN32);
            }
        }
        Fp8Encoding::Ocp => {
            if mag > f.max_code {
                // E4M3FN: only 0x7f; E5M2: 0x7c = Inf, 0x7d..0x7f = NaN.
                return f32::from_bits(match kind {
                    F8::Bf8 if mag == 0x7c => sign | 0x7f80_0000,
                    _ => sign | QNAN32,
                });
            }
        }
    }
    f32::from_bits(sign | f.magnitude(mag).to_bits())
}

/// Rounding direction of [`round_mag`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Round {
    /// Round to nearest, ties to even.
    Rne,
    /// Round toward zero (truncate).
    Rtz,
}

/// Rounds the finite, non-negative f32 `abs` (bits, sign clear) onto the grid
/// of a binary format with `mbits` stored mantissa bits and minimum normal
/// exponent `emin` (gradual underflow, unbounded exponent above). Returns the
/// magnitude code `(E << mbits) | m`; a carry out of the mantissa increments
/// `E`, so callers detect overflow by comparing against their largest code.
fn round_mag(abs: u32, mbits: u32, emin: i32, mode: Round) -> u32 {
    let (e, sig) = match abs >> 23 {
        0 => (-126, abs & 0x7f_ffff),
        be => (be as i32 - 127, (abs & 0x7f_ffff) | 0x80_0000),
    };
    if sig == 0 {
        return 0;
    }
    let eff = e.max(emin);
    // Bits of `sig` below the result's LSB: always >= 23 - mbits >= 13 here.
    let shift = (23 - mbits as i32 + (eff - e)) as u32;
    if shift >= 25 {
        // sig < 2^24 <= half an LSB: rounds to zero in both modes.
        return 0;
    }
    let mut q = sig >> shift;
    if mode == Round::Rne {
        let rem = sig & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if rem > half || (rem == half && q & 1 == 1) {
            q += 1;
        }
    }
    // Normal: q carries the implicit bit at `mbits`, and
    // ((e - emin) << mbits) + q == ((e + bias) << mbits) | m.
    // Subnormal: eff == emin, so the code is q itself.
    (((eff - emin) as u32) << mbits) + q
}

/// NaN code of `enc` for a value of sign bit `sign`.
fn f8_nan(enc: Fp8Encoding, sign: u8) -> u8 {
    match enc {
        Fp8Encoding::Fnuz => 0x80,
        Fp8Encoding::Ocp => (sign << 7) | 0x7f,
    }
}

/// FP16_OVFL table rows "±Inf" and "Greater than max FP8 magnitude"
/// (CDNA3 ISA §7.2 p.51).
fn f8_overflow(enc: Fp8Encoding, kind: F8, sign: u8, fp16_ovfl: bool) -> u8 {
    let f = Fmt::of(enc, kind);
    match (fp16_ovfl, kind, enc) {
        (true, _, _) => (sign << 7) | f.max_code,
        (false, F8::Fp8, _) | (false, F8::Bf8, Fp8Encoding::Fnuz) => f8_nan(enc, sign),
        (false, F8::Bf8, Fp8Encoding::Ocp) => (sign << 7) | 0x7c,
    }
}

/// Attaches the sign to a rounded magnitude code; zero is `0x00` in FNUZ
/// (no -0) and keeps its sign in OCP.
fn f8_finish(enc: Fp8Encoding, sign: u8, mag: u8) -> u8 {
    if mag == 0 && enc == Fp8Encoding::Fnuz {
        0
    } else {
        (sign << 7) | mag
    }
}

/// `v_cvt_pk_{fp8,bf8}_f32` on one element: RNE with the FP16_OVFL rule.
///
/// CDNA3 ISA §7.2 Table 31 p.50 (RNE, clamp/omod ignored) and the FP16_OVFL
/// table p.51. Gradual underflow (no denorm flushing, §7.3 p.52). The
/// overflow test is `|x| > max` before rounding, and FNUZ maps `-0.0` and
/// negative values rounding to zero to `0x00` — both
/// `[UNVERIFIED: confirm on MI300X]` (module doc).
pub fn f32_to_f8_rne(enc: Fp8Encoding, kind: F8, x: f32, fp16_ovfl: bool) -> u8 {
    let f = Fmt::of(enc, kind);
    let bits = x.to_bits();
    let sign = (bits >> 31) as u8;
    let abs = bits & 0x7fff_ffff;
    if abs > 0x7f80_0000 {
        return f8_nan(enc, sign);
    }
    if abs > f.max_bits() {
        return f8_overflow(enc, kind, sign, fp16_ovfl);
    }
    // |x| <= max and max is on the grid, so the rounded code is <= max_code.
    let mag = round_mag(abs, f.mbits, f.emin(), Round::Rne) as u8;
    f8_finish(enc, sign, mag)
}

/// `v_cvt_sr_{fp8,bf8}_f32`: stochastic rounding with random dword `rnd`
/// (CDNA3 ISA §7.2 p.51; the exact bit-use of `rnd` is `[UNVERIFIED]` until
/// confirmed on MI300X and must be documented as such).
///
/// Bit-use model implemented here `[UNVERIFIED: confirm on MI300X]`:
/// 1. NaN, ±Inf and `|x| > max` follow the FP16_OVFL table exactly as in
///    [`f32_to_f8_rne`] (the random value is not added to them).
/// 2. The top 20 (FP8: `rnd[31:12]`) or 21 (BF8: `rnd[31:11]`) bits of `rnd`
///    are added as an unsigned integer to `|x|`'s f32 bit pattern, i.e. into
///    mantissa bits `[19:0]` / `[20:0]` — the bits below a normal result's LSB
///    (pseudocode of V_CVT_SR_FP8_F32 / V_CVT_SR_BF8_F32, CDNA3 ISA §12.11
///    p.362-363: `m = mantissa(S0) + S1[31:12]` resp. `S1[31:11]`).
/// 3. The sum is truncated toward zero to the 8-bit grid (§7.2 p.51: "add a
///    random value … and then truncate"). The sum is below the grid point
///    after `|x|`'s rounded-down value, so it never truncates past `max`.
///    FNUZ zero is `0x00`.
///
/// Deliberate departures from the literal §12 pseudocode, which (a) drops the
/// mantissa carry (`23'U(...)`, "wrap around on overflow") so inputs near the
/// top of a binade would wrap down, and (b) converts the sum with RNE, which
/// biases the rounding: here the carry propagates into the exponent and the
/// sum is truncated. For results in the 8-bit subnormal range the random bits
/// cover only the low part of the dropped bits. AMD's software twin
/// (`amd_hip_fp8.h:351`) instead adds `rnd`'s *low* bits below the LSB after
/// the subnormal shift; hardware may match either.
pub fn f32_to_f8_sr(enc: Fp8Encoding, kind: F8, x: f32, rnd: u32, fp16_ovfl: bool) -> u8 {
    let f = Fmt::of(enc, kind);
    let bits = x.to_bits();
    let sign = (bits >> 31) as u8;
    let abs = bits & 0x7fff_ffff;
    if abs > 0x7f80_0000 {
        return f8_nan(enc, sign);
    }
    if abs > f.max_bits() {
        return f8_overflow(enc, kind, sign, fp16_ovfl);
    }
    let r = match kind {
        F8::Fp8 => rnd >> 12,
        F8::Bf8 => rnd >> 11,
    };
    // abs <= 57344's bits, so the sum stays far below the f32 Inf pattern.
    // r < one 8-bit LSB of |x|'s binade (2^20 / 2^21 f32 units for a normal
    // result, more for a subnormal one), so the truncation is |x|'s grid
    // point or the next one up, never beyond max (max is on the grid).
    let mag = round_mag(abs + r, f.mbits, f.emin(), Round::Rtz);
    debug_assert!(mag <= u32::from(f.max_code));
    f8_finish(enc, sign, mag as u8)
}

/// OCP E4M3FN byte → its FNUZ E4M3 image with value `ocp/2`: the same byte
/// for every finite code except `0x80` (OCP -0) → `0x00`; `None` for the OCP
/// NaN codes `0x7f`/`0xff` (GFX942-FP8-FN-PLAN §3.2).
///
/// Same bits, bias 8 instead of 7: normals `2^(E-8)(1+m/8)` vs
/// `2^(E-7)(1+m/8)`, subnormals `m·2^-10` vs `m·2^-9`.
pub fn ocp_e4m3_to_fnuz_half(code: u8) -> Option<u8> {
    match code {
        0x7f | 0xff => None,
        0x80 => Some(0x00),
        c => Some(c),
    }
}

/// IEEE binary16 bits → f32 (exact).
///
/// NaN is quieted (bit 22 set), payload kept in the top mantissa bits
/// `[UNVERIFIED: confirm on MI300X]` (`v_cvt_f32_f16` on a signalling NaN).
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let e = u32::from(bits >> 10) & 0x1f;
    let m = u32::from(bits) & 0x3ff;
    let mag = match e {
        // m · 2^-24, exact.
        0 => (m as f32 * pow2(-24)).to_bits(),
        0x1f if m == 0 => 0x7f80_0000,
        0x1f => QNAN32 | (m << 13),
        _ => ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(sign | mag)
}

/// f32 → binary16 with the given rounding; RNE overflows to ±Inf, RTZ
/// saturates finite values to ±65504 (IEEE 754 §7.4 overflow by mode).
fn f32_to_f16(x: f32, mode: Round) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let abs = bits & 0x7fff_ffff;
    if abs > 0x7f80_0000 {
        // Quiet NaN, top 9 payload bits kept.
        return sign | 0x7e00 | ((abs >> 13) & 0x1ff) as u16;
    }
    if abs == 0x7f80_0000 {
        return sign | 0x7c00;
    }
    let mag = round_mag(abs, 10, -14, mode);
    let mag = if mag >= 0x7c00 {
        match mode {
            Round::Rne => 0x7c00,
            Round::Rtz => 0x7bff,
        }
    } else {
        mag as u16
    };
    sign | mag
}

/// f32 → binary16, round to nearest even (`v_cvt_f16_f32`, MODE RNE).
///
/// Subnormals kept, overflow → ±Inf (with MODE.FP16_OVFL=0; the instruction's
/// FP16_OVFL clamp is the emulator's to apply), NaN → quiet NaN keeping the
/// top payload bits `[UNVERIFIED: confirm on MI300X]` (NaN payload).
pub fn f32_to_f16_rne(x: f32) -> u16 {
    f32_to_f16(x, Round::Rne)
}

/// f32 → binary16, round toward zero (`v_cvt_pkrtz_f16_f32`).
///
/// Finite values beyond 65504 give ±65504, ±Inf stays ±Inf, NaN as in
/// [`f32_to_f16_rne`] `[UNVERIFIED: confirm on MI300X]` (NaN payload).
pub fn f32_to_f16_rtz(x: f32) -> u16 {
    f32_to_f16(x, Round::Rtz)
}

/// BF16 bits → f32 (exact).
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// f32 → BF16, round to nearest even; NaN stays NaN (quiet).
///
/// Subnormals kept; overflow → ±Inf; NaN → `(bits >> 16) | 0x0040` (sign and
/// top payload kept, quiet bit set) `[UNVERIFIED: confirm on MI300X]`.
pub fn f32_to_bf16_rne(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let abs = bits & 0x7fff_ffff;
    if abs > 0x7f80_0000 {
        return (bits >> 16) as u16 | 0x0040;
    }
    if abs == 0x7f80_0000 {
        return sign | 0x7f80;
    }
    // A carry past the largest finite value lands exactly on 0x7f80 (Inf).
    sign | round_mag(abs, 7, -126, Round::Rne) as u16
}
