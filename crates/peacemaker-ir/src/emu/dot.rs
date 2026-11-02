// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! VOP3P dot-product opcodes (wave32): `v_dot4_i32_iu8`, `v_dot8_i32_iu4`, `v_dot2_f32_f16`, `v_dot2_f32_bf16`.
//!
//! The integer forms follow the RDNA3/RDNA4 pseudocode: every nibble/byte is zero- or sign-extended according to
//! its operand's NEG_LO bit (0 = unsigned, 1 = signed), the products and `S2.i32` add in a signed accumulator and
//! the dword wraps. NEG_HI is undefined for the IU forms and OP_SEL has no modeled meaning: both are errors.
//! Floating-point arithmetic lives in [`super::mma`], which owns every measured FP32 accumulation model.
use super::{mma, Result, State};
use crate::{
    operand::{Omod, Operand},
    Arch, Inst,
};

pub(super) fn execute(arch: Arch, name: &str, i: &Inst, s: &mut State) -> Result<()> {
    let m = &i.mods;
    let o = &i.operands;
    if o.len() != 4 {
        return Err(format!(
            "{name}: expected vdst,S0,S1,S2 operands, got {}",
            o.len()
        ));
    }
    if m.omod != Omod::None || m.dpp.is_some() || m.sdwa.is_some() {
        return Err(format!("{name}: omod/DPP/SDWA are not part of a VOP3P dot"));
    }
    // The integer dots model no op_sel at all; the float dots interpret op_sel/op_sel_hi themselves below.
    let integer = matches!(name, "v_dot4_i32_iu8" | "v_dot8_i32_iu4");
    if integer && (m.op_sel != 0 || !matches!(m.op_sel_hi, 0 | 7)) {
        return Err(format!(
            "{name}: op_sel={:#x} op_sel_hi={:#x} has no modeled dot meaning",
            m.op_sel, m.op_sel_hi
        ));
    }
    let mut out = [None::<u32>; 64];
    match name {
        "v_dot4_i32_iu8" | "v_dot8_i32_iu4" => {
            // GFX1201 probe: NEG[2] and NEG_HI have no effect on the integer dots; unmeasured elsewhere.
            if arch != Arch::Gfx1201 && (m.neg_lo & 4 != 0 || m.neg_hi != 0) {
                return Err(format!("{name}: NEG[2]/NEG_HI are undefined for integer dots and unmeasured on {arch:?}"));
            }
            let (bits, count) = if name == "v_dot4_i32_iu8" {
                (8, 4)
            } else {
                (4, 8)
            };
            let signed = [m.neg_lo & 1 != 0, m.neg_lo & 2 != 0];
            let field = |word: u32, n: usize, signed: bool| -> i32 {
                let v = ((word >> (bits * n)) & ((1 << bits) - 1)) as i32;
                if signed {
                    (v << (32 - bits)) >> (32 - bits)
                } else {
                    v
                }
            };
            for lane in (0..s.lanes).filter(|l| s.exec >> l & 1 != 0) {
                let (a, b, c) = (
                    s.read(&o[1], lane, 0)?,
                    s.read(&o[2], lane, 0)?,
                    s.read(&o[3], lane, 0)? as i32,
                );
                let mut a_f = [0i32; 8];
                let mut b_f = [0i32; 8];
                for n in 0..count {
                    a_f[n] = field(a, n, signed[0]);
                    b_f[n] = field(b, n, signed[1]);
                }
                out[lane] =
                    Some(mma::dot_i32(arch, &a_f[..count], &b_f[..count], c, m.clamp)? as u32);
            }
        }
        "v_dot2_f32_f16" | "v_dot2_f32_bf16" => {
            // Only gfx1201 has a probed model (mma::dot2_f32 documents the receipt); gfx1151/gfx1100 refuse.
            if arch != Arch::Gfx1201 {
                return Err(format!("{name} on {arch:?}: no probed floating-point dot model; refusing to approximate"));
            }
            // op_sel[2] / op_sel_hi[2] (the f32 C operand) were never varied: only the defaults are modeled.
            if m.op_sel & 4 != 0 || m.op_sel_hi & 4 == 0 {
                return Err(format!(
                    "{name}: op_sel[2]/op_sel_hi[2] on the f32 accumulator are not measured"
                ));
            }
            if [&o[1], &o[2]]
                .iter()
                .any(|op| matches!(op, Operand::Inline(_)))
            {
                return Err(format!(
                    "{name}: an inline constant as a packed half source is not measured"
                ));
            }
            let bf16 = name == "v_dot2_f32_bf16";
            // neg_lo[k]/neg_hi[k] flip the sign of operand k's lo/hi element; op_sel[k] picks which half is the lo
            // element and op_sel_hi[k] which half is the hi element (default: lo, hi). CLAMP and MODE have no effect.
            let pick = |word: u32, k: u8| -> [u16; 2] {
                let half = |hi: bool| (if hi { word >> 16 } else { word }) as u16;
                let mut e = [
                    half(m.op_sel >> k & 1 != 0),
                    half(m.op_sel_hi >> k & 1 != 0),
                ];
                if m.neg_lo >> k & 1 != 0 {
                    e[0] ^= 0x8000;
                }
                if m.neg_hi >> k & 1 != 0 {
                    e[1] ^= 0x8000;
                }
                e
            };
            for lane in (0..s.lanes).filter(|l| s.exec >> l & 1 != 0) {
                let (a, b, mut c) = (
                    s.read(&o[1], lane, 0)?,
                    s.read(&o[2], lane, 0)?,
                    s.read(&o[3], lane, 0)?,
                );
                if m.neg_hi & 4 != 0 {
                    c &= 0x7fff_ffff;
                }
                if m.neg_lo & 4 != 0 {
                    c ^= 0x8000_0000;
                }
                out[lane] = Some(mma::dot2_f32(arch, pick(a, 0), pick(b, 1), c, bf16)?);
            }
        }
        _ => return Err(format!("unsupported dot opcode {name}")),
    }
    for (lane, v) in out.into_iter().enumerate() {
        if let Some(v) = v {
            s.put(&o[0], lane, 0, v)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        operand::{Modifiers, Operand},
        reg::{Kind, RegRef},
    };

    fn v(base: u16) -> Operand {
        Operand::Reg(RegRef {
            kind: Kind::V,
            base,
            len: 1,
        })
    }
    fn run(
        arch: Arch,
        name: &str,
        s0: u32,
        s1: u32,
        c: u32,
        neg_lo: u8,
        neg_hi: u8,
        clamp: bool,
    ) -> std::result::Result<u32, String> {
        let table = if arch == Arch::Gfx1201 {
            crate::isa::gfx12()
        } else {
            crate::isa::gfx1151()
        };
        let row = table
            .iter()
            .find(|r| r.name == name)
            .unwrap_or_else(|| panic!("{name} missing"));
        let mods = Modifiers {
            neg_lo,
            neg_hi,
            clamp,
            op_sel_hi: 7,
            ..Default::default()
        };
        let i = Inst::from_parts(
            arch,
            row.op,
            row.form,
            Default::default(),
            vec![v(0), v(1), v(2), v(3)].into_iter().collect(),
            mods,
            None,
            Default::default(),
        )
        .unwrap();
        let mut st = State {
            s: [0; 106],
            t: [0; 16],
            v: Box::new([[0; 64]; 256]),
            lanes: 32,
            nan_loose: false,
            exec: 1,
            vcc: 0,
            scc: false,
            m0: 0,
            pc: 0,
            ended: false,
            barrier: None,
            signals: 0,
            scratch: Vec::new(),
        };
        (st.v[1][0], st.v[2][0], st.v[3][0]) = (s0, s1, c);
        st.v[0][1] = 0x1234; // inactive lane keeps its destination
        execute(arch, name, &i, &mut st).map(|_| {
            assert_eq!(st.v[0][1], 0x1234);
            st.v[0][0]
        })
    }

    // NEG_LO bit n = operand n signed; products and S2 add exactly; the dword wraps.
    #[test]
    fn dot4_iu8_signedness_and_wrap() {
        let (a, b) = (0x7f80_02ff, 0xff02_fe03);
        for arch in [Arch::Gfx1151, Arch::Gfx1201] {
            let n = "v_dot4_i32_iu8";
            assert_eq!(run(arch, n, a, b, 10, 3, 0, false), Ok((-380i32) as u32));
            assert_eq!(run(arch, n, a, b, 10, 0, 0, false), Ok(33924));
            assert_eq!(run(arch, n, a, b, 10, 1, 0, false), Ok(32644));
            let all = 0x7f7f_7f7f;
            assert_eq!(
                run(arch, n, all, all, i32::MAX as u32, 3, 0, false),
                Ok((i64::from(i32::MAX) + 4 * 127 * 127) as u32)
            );
        }
    }

    #[test]
    fn dot8_iu4_signedness() {
        for arch in [Arch::Gfx1151, Arch::Gfx1201] {
            let n = "v_dot8_i32_iu4";
            let a = 0xf123_4567;
            assert_eq!(run(arch, n, a, 0x1111_1111, 100, 0, 0, false), Ok(143));
            assert_eq!(run(arch, n, a, 0x1111_1111, 100, 1, 0, false), Ok(127));
            assert_eq!(run(arch, n, a, 0xffff_ffff, 100, 3, 0, false), Ok(73));
            assert_eq!(
                run(arch, n, a, 0xffff_ffff, 100, 0, 0, false),
                Ok(100 + 15 * 43)
            );
        }
    }

    // GFX1201 probe: CLAMP is one final signed saturation; NEG[2]/NEG_HI are ignored. Unmeasured on gfx1151.
    #[test]
    fn gfx1201_clamp_saturates_once_and_ignores_neg2_and_neg_hi() {
        let all = 0x7f7f_7f7f;
        let n = "v_dot4_i32_iu8";
        assert_eq!(
            run(Arch::Gfx1201, n, all, all, i32::MAX as u32, 3, 0, true),
            Ok(i32::MAX as u32)
        );
        assert_eq!(
            run(
                Arch::Gfx1201,
                n,
                0x8080_8080,
                all,
                i32::MIN as u32,
                3,
                0,
                true
            ),
            Ok(i32::MIN as u32)
        );
        // One final saturation, not one per step: C = MAX-100 plus 127*127 would clip, then -128*127 brings it back.
        assert_eq!(
            run(
                Arch::Gfx1201,
                n,
                0x0000_807f,
                all,
                (i32::MAX - 100) as u32,
                3,
                0,
                true
            ),
            Ok((i64::from(i32::MAX) - 100 + 127 * 127 - 128 * 127) as u32)
        );
        assert_eq!(
            run(Arch::Gfx1201, n, 0x7f80_02ff, 0xff02_fe03, 10, 7, 7, false),
            run(Arch::Gfx1201, n, 0x7f80_02ff, 0xff02_fe03, 10, 3, 0, false)
        );
        assert!(run(Arch::Gfx1151, n, all, all, 0, 3, 0, true).is_err());
        assert!(run(Arch::Gfx1151, n, all, all, 0, 7, 0, false).is_err());
        assert!(run(Arch::Gfx1151, n, all, all, 0, 3, 1, false).is_err());
    }

    fn dot2(
        arch: Arch,
        name: &str,
        (neg_lo, neg_hi, op_sel, op_sel_hi, clamp): (u8, u8, u8, u8, bool),
        s0: u32,
        s1: u32,
        c: u32,
    ) -> std::result::Result<u32, String> {
        let table = if arch == Arch::Gfx1201 {
            crate::isa::gfx12()
        } else {
            crate::isa::gfx1151()
        };
        let row = table
            .iter()
            .find(|r| r.name == name)
            .unwrap_or_else(|| panic!("{name} missing"));
        let mods = Modifiers {
            neg_lo,
            neg_hi,
            op_sel,
            op_sel_hi,
            clamp,
            ..Default::default()
        };
        let i = Inst::from_parts(
            arch,
            row.op,
            row.form,
            Default::default(),
            vec![v(0), v(1), v(2), v(3)].into_iter().collect(),
            mods,
            None,
            Default::default(),
        )
        .unwrap();
        let mut st = State {
            s: [0; 106],
            t: [0; 16],
            v: Box::new([[0; 64]; 256]),
            lanes: 32,
            nan_loose: false,
            exec: 1,
            vcc: 0,
            scc: false,
            m0: 0,
            pc: 0,
            ended: false,
            barrier: None,
            signals: 0,
            scratch: Vec::new(),
        };
        (st.v[1][0], st.v[2][0], st.v[3][0]) = (s0, s1, c);
        execute(arch, name, &i, &mut st).map(|_| st.v[0][0])
    }

    // Raw gfx1201 hardware words (cards B and C, receipt rcpt/gfx1201-C-r1, set `red`, 99,742 cases per variant).
    // Tuple: (neg_lo, neg_hi, op_sel, op_sel_hi, clamp), A, B, C, hardware D. Cases were picked where the variant
    // changes the result relative to the unmodified opcode.
    #[test]
    fn gfx1201_dot2_matches_the_hardware_receipt_for_every_modifier() {
        let cases: &[(&str, (u8, u8, u8, u8, bool), u32, u32, u32, u32)] = &[
            (
                "v_dot2_f32_f16",
                (0, 0, 0, 7, false),
                0x2c01b614,
                0xbd454200,
                0x3e800000,
                0xbf78d945,
            ),
            (
                "v_dot2_f32_f16",
                (0, 0, 0, 7, true),
                0x58084d87,
                0xe3ef4400,
                0xc531a800,
                0xc8029e64,
            ),
            (
                "v_dot2_f32_f16",
                (1, 0, 0, 7, false),
                0xc22e4008,
                0x4020439c,
                0xc010d610,
                0xc1827002,
            ),
            (
                "v_dot2_f32_f16",
                (4, 0, 0, 7, false),
                0xc1e4c000,
                0xc2ddc000,
                0x40c307d0,
                0x41003368,
            ),
            (
                "v_dot2_f32_f16",
                (0, 1, 0, 7, false),
                0x7b9155ee,
                0xefe0bc00,
                0x4affffff,
                0x4df2577d,
            ),
            (
                "v_dot2_f32_f16",
                (0, 4, 0, 7, false),
                0x48002cd1,
                0x2c002c00,
                0xbc800000,
                0x3f053440,
            ),
            (
                "v_dot2_f32_f16",
                (0, 0, 1, 7, false),
                0x03ff27c0,
                0xa4a2a7ff,
                0xb8c723d3,
                0xb8cd72bf,
            ),
            (
                "v_dot2_f32_f16",
                (0, 0, 2, 7, false),
                0x31204200,
                0x3c010401,
                0xb9ffffff,
                0x404a6a90,
            ),
            (
                "v_dot2_f32_f16",
                (0, 0, 3, 7, false),
                0xe804dcc0,
                0xe000e5e5,
                0xc289024f,
                0x4a007eee,
            ),
            (
                "v_dot2_f32_f16",
                (0, 0, 0, 6, false),
                0xe99058f4,
                0xd45b54ba,
                0xc6dd0969,
                0xc6d5af39,
            ),
            (
                "v_dot2_f32_f16",
                (0, 0, 0, 5, false),
                0x31204200,
                0x3c010401,
                0xb9ffffff,
                0xb99ac6b7,
            ),
            (
                "v_dot2_f32_f16",
                (0, 0, 0, 4, false),
                0x3db8bc00,
                0x3d51bc80,
                0xc0800000,
                0xbfe00000,
            ),
            (
                "v_dot2_f32_f16",
                (7, 7, 0, 7, false),
                0x14008643,
                0x06308a6e,
                0x34ab6a36,
                0xb45fb307,
            ),
            (
                "v_dot2_f32_bf16",
                (0, 0, 0, 7, false),
                0x3e20be68,
                0xbe01be40,
                0x3cffffff,
                0x3d5d6000,
            ),
            (
                "v_dot2_f32_bf16",
                (0, 0, 0, 7, true),
                0xc210c27f,
                0x0040c240,
                0xc4b6ee9c,
                0x44c79164,
            ),
            (
                "v_dot2_f32_bf16",
                (1, 0, 0, 7, false),
                0x4564c57f,
                0xc455c400,
                0xc8000000,
                0xcaa29a00,
            ),
            (
                "v_dot2_f32_bf16",
                (4, 0, 0, 7, false),
                0x3f763f66,
                0x3f403f00,
                0x3e32928f,
                0x3f7edb5c,
            ),
            (
                "v_dot2_f32_bf16",
                (0, 1, 0, 7, false),
                0x437f4000,
                0x47bcc501,
                0xccc2cff2,
                0xccf1a2f6,
            ),
            (
                "v_dot2_f32_bf16",
                (0, 4, 0, 7, false),
                0x41c0c5fa,
                0x448241e0,
                0xc5004000,
                0xc8405f00,
            ),
            (
                "v_dot2_f32_bf16",
                (0, 0, 1, 7, false),
                0x4297bf60,
                0xc152c400,
                0x44800010,
                0xc716def0,
            ),
            (
                "v_dot2_f32_bf16",
                (0, 0, 2, 7, false),
                0xc0ff3b88,
                0x3bfb7fff,
                0xbc670d2b,
                0xbd99d37a,
            ),
            (
                "v_dot2_f32_bf16",
                (0, 0, 3, 7, false),
                0x3f043f04,
                0x3f01bf16,
                0x403db000,
                0x405ef200,
            ),
            (
                "v_dot2_f32_bf16",
                (0, 0, 0, 6, false),
                0xc508c400,
                0x414a4438,
                0xc6800000,
                0xc8c32800,
            ),
            (
                "v_dot2_f32_bf16",
                (0, 0, 0, 5, false),
                0xbf403687,
                0x3a01bc7f,
                0xb6800000,
                0x3c3f2fbd,
            ),
            (
                "v_dot2_f32_bf16",
                (0, 0, 0, 4, false),
                0xbf50c301,
                0x0080c295,
                0x4450399d,
                0x469cabcd,
            ),
            (
                "v_dot2_f32_bf16",
                (7, 7, 0, 7, false),
                0xbccfbba1,
                0x3de9bc80,
                0x39800000,
                0xbb475f00,
            ),
            // neg_lo[1] / neg_hi[1] negate the B operand's elements (same hardware pick as variants 3 and 6).
            (
                "v_dot2_f32_f16",
                (2, 0, 0, 7, false),
                0xc22e4008,
                0x4020439c,
                0xc010d610,
                0xc1827002,
            ),
            (
                "v_dot2_f32_f16",
                (0, 2, 0, 7, false),
                0x7b9155ee,
                0xefe0bc00,
                0x4affffff,
                0x4df2577d,
            ),
            (
                "v_dot2_f32_bf16",
                (2, 0, 0, 7, false),
                0x4564c57f,
                0xc455c400,
                0xc8000000,
                0xcaa29a00,
            ),
        ];
        for &(name, mods, a, b, c, want) in cases {
            assert_eq!(
                dot2(Arch::Gfx1201, name, mods, a, b, c),
                Ok(want),
                "{name} {mods:?} a={a:#x} b={b:#x} c={c:#x}"
            );
        }
        // Factor NaN diagnostic payload and invalid-product NaN come from the shared measured step.
        assert_eq!(
            dot2(
                Arch::Gfx1201,
                "v_dot2_f32_f16",
                (0, 0, 0, 7, false),
                0xfe00_0001,
                0x0400_0400,
                0x8080_0000
            ),
            Ok(0xffc0_0a3d)
        );
    }

    // Unmeasured corners and architectures refuse instead of approximating.
    #[test]
    fn float_dot2_refuses_unmeasured_architectures_and_modes() {
        let plain = (0, 0, 0, 7, false);
        for name in ["v_dot2_f32_f16", "v_dot2_f32_bf16"] {
            assert!(
                dot2(Arch::Gfx1151, name, plain, 0x3c00_3c00, 0x3c00_3c00, 0).is_err(),
                "{name} gfx1151"
            );
            assert!(
                dot2(Arch::Gfx1201, name, (0, 0, 4, 7, false), 0, 0, 0).is_err(),
                "op_sel[2]"
            );
            assert!(
                dot2(Arch::Gfx1201, name, (0, 0, 0, 3, false), 0, 0, 0).is_err(),
                "op_sel_hi[2] clear"
            );
        }
        // 1.0 * 1.0 + 1.0 * 1.0 + 0.5, exact.
        assert_eq!(
            dot2(
                Arch::Gfx1201,
                "v_dot2_f32_f16",
                plain,
                0x3c00_3c00,
                0x3c00_3c00,
                0.5f32.to_bits()
            ),
            Ok(2.5f32.to_bits())
        );
    }
}
