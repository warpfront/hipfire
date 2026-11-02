// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! RDNA3/RDNA3.5 codec: separate target opcode tables, shared typed field codec.
pub use super::gfx12::DecodeError;
use crate::inst::{Arch, Inst};
use smallvec::SmallVec;

pub fn decode(arch: Arch, words: &[u32]) -> Result<(Inst, usize), DecodeError> {
    assert!(matches!(arch, Arch::Gfx1100 | Arch::Gfx1151));
    super::gfx12::decode_for(arch, words)
}
pub fn encode(arch: Arch, inst: &Inst) -> Result<SmallVec<[u32; 3]>, DecodeError> {
    assert!(matches!(arch, Arch::Gfx1100 | Arch::Gfx1151));
    super::gfx12::encode_for(arch, inst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{isa, wait::Counter};

    #[test]
    fn target_rows_reencode_from_typed_fields() {
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            let mut failures = Vec::new();
            for row in isa::table(arch) {
                let words: Vec<u32> = row
                    .encoding
                    .split_whitespace()
                    .map(|word| u32::from_str_radix(word, 16).unwrap())
                    .collect();
                match decode(arch, &words).and_then(|(inst, used)| {
                    if used != words.len() {
                        return Err(super::super::gfx12::DecodeError::Rejected {
                            offset: 0,
                            reason: "width mismatch".into(),
                        });
                    }
                    let output = encode(arch, &inst)?;
                    if output.as_slice() != words {
                        return Err(super::super::gfx12::DecodeError::Rejected {
                            offset: 0,
                            reason: "word mismatch".into(),
                        });
                    }
                    Ok(())
                }) {
                    Ok(()) => {}
                    Err(error) => failures.push(format!("{} {words:08x?}: {error}", row.name)),
                }
            }
            assert!(failures.is_empty(), "{arch:?}:\n{}", failures.join("\n"));
        }
    }

    #[test]
    fn gfx11_waits_and_scalar_float_are_arch_specific() {
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            let (inst, _) = decode(arch, &[0xbf89_0432]).unwrap();
            let wait = inst.mods.wait.as_ref().unwrap();
            assert_eq!(wait.per_counter[Counter::Vm as usize], Some(1));
            assert_eq!(wait.per_counter[Counter::Exp as usize], Some(2));
            assert_eq!(wait.per_counter[Counter::Lgkm as usize], Some(3));
            assert_eq!(encode(arch, &inst).unwrap().as_slice(), &[0xbf89_0432]);
            let (mut variant, _) = decode(arch, &[0xbf89_043a]).unwrap();
            assert_eq!(variant.mods.wait, inst.mods.wait);
            variant.prov.bytes = None;
            assert_eq!(encode(arch, &variant).unwrap().as_slice(), &[0xbf89_043a]);
        }
        assert!(decode(Arch::Gfx1100, &[0xa000_0201]).is_err());
        let (add, _) = decode(Arch::Gfx1151, &[0xa000_0201]).unwrap();
        assert_eq!(add.op.name(Arch::Gfx1151), Some("s_add_f32"));
    }

    /// Every low/high-half conversion in the historical gfx1151 Down C1 ELF,
    /// disassembled by LLVM for both gfx1100 and gfx1151. The first word is the
    /// original lift failure at 0x1aa0, not a current-source gfx1151 artifact.
    #[test]
    fn historical_down_c1_u32_u16_conversions_roundtrip_with_llvm() {
        use crate::{
            inst::Form,
            operand::{Half, Operand},
            reg::{Kind, RegRef},
        };
        use std::{
            io::Write,
            process::{Command, Stdio},
        };
        let cases: [(u32, u16, u16, Half); 7] = [
            (0x7e12_d789, 9, 9, Half::Hi),
            (0x7e5a_d71b, 45, 27, Half::Lo),
            (0x7e36_d79b, 27, 27, Half::Hi),
            (0x7e3a_d71d, 29, 29, Half::Lo),
            (0x7e5c_d71c, 46, 28, Half::Lo),
            (0x7e38_d79c, 28, 28, Half::Hi),
            (0x7e14_d70a, 10, 10, Half::Lo),
        ];
        let mc = crate::rocm_llvm_mc();
        let mc = if crate::rocm::tests_disabled() {
            None
        } else {
            assert!(
                mc.is_file(),
                "required ROCm assembler missing: {}",
                mc.display()
            );
            Some(mc)
        };
        let v = |base| RegRef {
            kind: Kind::V,
            base,
            len: 1,
        };
        for (arch, cpu) in [(Arch::Gfx1100, "gfx1100"), (Arch::Gfx1151, "gfx1151")] {
            for (word, dst, src, half) in cases {
                let suffix = if half == Half::Hi { "h" } else { "l" };
                let text = format!("v_cvt_u32_u16_e32 v{dst}, v{src}.{suffix}");
                if let Some(mc) = &mc {
                    let mut child = Command::new(mc)
                        .args([
                            "-triple=amdgcn-amd-amdhsa",
                            &format!("-mcpu={cpu}"),
                            "-show-encoding",
                        ])
                        .stdin(Stdio::piped())
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .spawn()
                        .expect("pinned llvm-mc");
                    child
                        .stdin
                        .take()
                        .unwrap()
                        .write_all(format!("{text}\n").as_bytes())
                        .unwrap();
                    let out = child.wait_with_output().unwrap();
                    assert!(
                        out.status.success(),
                        "{cpu} {text}: {}",
                        String::from_utf8_lossy(&out.stderr)
                    );
                    let expected = word
                        .to_le_bytes()
                        .iter()
                        .map(|b| format!("0x{b:02x}"))
                        .collect::<Vec<_>>()
                        .join(",");
                    assert!(
                        String::from_utf8_lossy(&out.stdout)
                            .contains(&format!("encoding: [{expected}]")),
                        "{cpu} {text}"
                    );
                }
                let (mut inst, used) =
                    decode(arch, &[word]).unwrap_or_else(|e| panic!("{cpu} {text}: {e}"));
                assert_eq!(used, 1);
                assert_eq!(inst.form, Form::Vop1);
                assert_eq!(inst.op.id, 0x6b);
                assert_eq!(inst.op.name(arch), Some("v_cvt_u32_u16_e32"));
                assert_eq!(
                    inst.operands.as_slice(),
                    &[Operand::Reg(v(dst)), Operand::Half(v(src), half)]
                );
                assert_eq!(inst.effects.defs.as_slice(), &[v(dst)]);
                assert_eq!(inst.effects.uses.as_slice(), &[v(src)]);
                assert_eq!(encode(arch, &inst).unwrap().as_slice(), &[word]);
                inst.prov = crate::provenance::Provenance::default();
                assert_eq!(encode(arch, &inst).unwrap().as_slice(), &[word]);
            }
            // A nearby undefined VOP1 opcode must remain rejected; adding the
            // conversion row must not turn into generic unknown-form acceptance.
            let error = decode(arch, &[0x7e12_ff89]).unwrap_err().to_string();
            assert!(error.contains("undefined"), "{cpu}: {error}");
        }
    }

    /// GLOBAL atomics: GLC selects the returning form. Words 1-7 are every
    /// atomic in the DS4 gfx1151 objects; 8-10 are the opposite forms.
    #[test]
    fn global_atomic_return_form_is_the_glc_bit() {
        use crate::{
            effects::MemClass,
            operand::Operand,
            reg::{Kind, RegRef},
        };
        use std::{
            io::Write,
            process::{Command, Stdio},
        };
        let cases: [([u32; 2], &str, bool); 10] = [
            (
                [0xdcd2_4000, 0x0000_0002],
                "global_atomic_cmpswap_b32 v0, v2, v[0:1], s[0:1] glc",
                true,
            ),
            (
                [0xdcd2_4004, 0x0000_0002],
                "global_atomic_cmpswap_b32 v0, v2, v[0:1], s[0:1] offset:4 glc",
                true,
            ),
            (
                [0xdcd2_4000, 0x0000_0004],
                "global_atomic_cmpswap_b32 v0, v4, v[0:1], s[0:1] glc",
                true,
            ),
            (
                [0xdcd6_4000, 0x0204_020a],
                "global_atomic_add_u32 v2, v10, v2, s[4:5] glc",
                true,
            ),
            (
                [0xdcce_0004, 0x0006_0c0d],
                "global_atomic_swap_b32 v13, v12, s[6:7] offset:4",
                false,
            ),
            (
                [0xdcce_0004, 0x0000_0000],
                "global_atomic_swap_b32 v0, v0, s[0:1] offset:4",
                false,
            ),
            (
                [0xdcce_0000, 0x0000_0000],
                "global_atomic_swap_b32 v0, v0, s[0:1]",
                false,
            ),
            (
                [0xdcd2_0000, 0x0000_0002],
                "global_atomic_cmpswap_b32 v2, v[0:1], s[0:1]",
                false,
            ),
            (
                [0xdcd6_0000, 0x0004_020a],
                "global_atomic_add_u32 v10, v2, s[4:5]",
                false,
            ),
            (
                [0xdcce_4004, 0x0106_0c0d],
                "global_atomic_swap_b32 v1, v13, v12, s[6:7] offset:4 glc",
                true,
            ),
        ];
        // The encoding cross-check needs the pinned llvm-mc (resolved like
        // the gfx12 table gate); the no-GPU CI runner still checks decode.
        let mc = crate::rocm_llvm_mc();
        let mc = if crate::rocm::tests_disabled() {
            None
        } else {
            assert!(
                mc.is_file(),
                "required ROCm assembler missing: {}",
                mc.display()
            );
            Some(mc)
        };
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            let cpu = if arch == Arch::Gfx1100 {
                "gfx1100"
            } else {
                "gfx1151"
            };
            for (words, text, returns) in cases {
                if let Some(mc) = &mc {
                    let mut child = Command::new(mc)
                        .args([
                            "-triple=amdgcn-amd-amdhsa",
                            &format!("-mcpu={cpu}"),
                            "-show-encoding",
                        ])
                        .stdin(Stdio::piped())
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .spawn()
                        .expect("pinned llvm-mc");
                    child
                        .stdin
                        .take()
                        .unwrap()
                        .write_all(format!("{text}\n").as_bytes())
                        .unwrap();
                    let out = child.wait_with_output().unwrap();
                    assert!(
                        out.status.success(),
                        "{text}: {}",
                        String::from_utf8_lossy(&out.stderr)
                    );
                    let expected: String = words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .map(|b| format!("0x{b:02x}"))
                        .collect::<Vec<_>>()
                        .join(",");
                    assert!(
                        String::from_utf8_lossy(&out.stdout)
                            .contains(&format!("encoding: [{expected}]")),
                        "{text}"
                    );
                }

                let (mut inst, used) =
                    decode(arch, &words).unwrap_or_else(|e| panic!("{text}: {e}"));
                assert_eq!(used, 2);
                assert_eq!(inst.mods.cpol.glc, returns);
                let mem = inst.effects.mem.expect("atomic memory effect");
                assert_eq!(mem.class, MemClass::VmemAtomic { returns }, "{text}");
                let (counted, idle) = if returns {
                    (Counter::Vm, Counter::Vs)
                } else {
                    (Counter::Vs, Counter::Vm)
                };
                assert!(
                    mem.counters.contains(counted) && !mem.counters.contains(idle),
                    "{text}: {:?}",
                    mem.counters
                );
                let vdst = RegRef {
                    kind: Kind::V,
                    base: (words[1] >> 24) as u16,
                    len: 1,
                };
                assert_eq!(
                    inst.effects.defs.to_vec(),
                    if returns { vec![vdst] } else { vec![] },
                    "{text}"
                );
                let data_len = if text.contains("cmpswap") { 2 } else { 1 };
                let data = RegRef {
                    kind: Kind::V,
                    base: (words[1] >> 8 & 0xff) as u16,
                    len: data_len,
                };
                assert!(
                    inst.effects.uses.contains(&data),
                    "{text}: data {data:?} not read"
                );
                inst.prov.bytes = None;
                assert_eq!(encode(arch, &inst).unwrap().as_slice(), &words, "{text}");
            }
            // Without GLC there is no destination, so a nonzero VDST byte is unmodeled.
            assert!(decode(arch, &[0xdcce_0004, 0x0106_0c0d]).is_err());
        }
    }

    /// Rows added by the pm-opcode-fill pass: the 59 emitted gfx11 identities of
    /// ACO-LLVM-OPCODE-DIFF.md that the tables lacked (minus the three gfx1151-only scalar
    /// ops below) plus the useful WMMA/dot identities 0x13/0x16/0x18/0x1a/0x41/0x43. Each
    /// entry is `(row name, opcode id, words)`; the words are the pinned ROCm
    /// `llvm-mc -mcpu=gfx1100/gfx1151 -show-encoding` output of the commented text (identical
    /// on both targets), and the id is the Mesa `aco_opcodes.py` gfx11 column / MR-ISA
    /// `amdgpu_isa_rdna3.xml` opcode. They live in gfx1100.tbl and gfx1151 inherits them.
    const PM_FILL_BASE: &[(&str, u16, &[u32])] = &[
        ("ds_add_u32", 0x00, &[0xd800_0000, 0x0000_0407]), // ds_add_u32 v7, v4
        ("ds_or_b32", 0x0a, &[0xd828_0000, 0x0000_0100]),  // ds_or_b32 v0, v1
        ("ds_add_f32", 0x15, &[0xd854_0010, 0x0000_0201]), // ds_add_f32 v1, v2 offset:16
        ("ds_add_rtn_u32", 0x20, &[0xd880_0000, 0x0100_0701]), // ds_add_rtn_u32 v1, v1, v7
        ("ds_load_u16", 0x3c, &[0xd8f0_0000, 0x8800_0015]), // ds_load_u16 v136, v21
        ("ds_store_b96", 0xde, &[0xdb78_0010, 0x0000_0201]), // ds_store_b96 v1, v[2:4] offset:16
        ("flat_store_b8", 0x18, &[0xdc60_0000, 0x007c_0200]), // flat_store_b8 v[0:1], v2
        ("flat_store_b32", 0x1a, &[0xdc68_0004, 0x007c_0200]), // flat_store_b32 v[0:1], v2 offset:4
        ("flat_store_b128", 0x1d, &[0xdc74_0010, 0x007c_0218]), // flat_store_b128 v[24:25], v[2:5] offset:16
        ("flat_load_d16_u8", 0x1e, &[0xdc78_0004, 0x017c_0007]), // flat_load_d16_u8 v1, v[7:8] offset:4
        ("flat_load_d16_hi_b16", 0x23, &[0xdc8c_0004, 0x017c_0007]), // flat_load_d16_hi_b16 v1, v[7:8] offset:4
        ("s_sext_i32_i8", 0x0e, &[0xbe81_0e02]),                     // s_sext_i32_i8 s1, s2
        ("s_bfe_i32", 0x27, &[0x9380_ff75, 0x0001_001d]),            // s_bfe_i32 s0, ttmp9, 0x1001d
        ("s_pack_lh_b32_b16", 0x33, &[0x9980_2780]),                 // s_pack_lh_b32_b16 s0, 0, s39
        ("v_cvt_i32_f64_e32", 0x03, &[0x7e08_0704]),                 // v_cvt_i32_f64_e32 v4, v[4:5]
        ("v_cvt_f64_i32_e32", 0x04, &[0x7e08_0904]),                 // v_cvt_f64_i32_e32 v[4:5], v4
        ("v_cvt_f32_f64_e32", 0x0f, &[0x7e42_1f21]), // v_cvt_f32_f64_e32 v33, v[33:34]
        ("v_cvt_f64_f32_e32", 0x10, &[0x7e4a_2125]), // v_cvt_f64_f32_e32 v[37:38], v37
        ("v_cvt_f32_ubyte1_e32", 0x12, &[0x7e4c_251d]), // v_cvt_f32_ubyte1_e32 v38, v29
        ("v_cvt_f32_ubyte2_e32", 0x13, &[0x7e4e_271d]), // v_cvt_f32_ubyte2_e32 v39, v29
        ("v_cvt_f32_ubyte3_e32", 0x14, &[0x7e14_290a]), // v_cvt_f32_ubyte3_e32 v10, v10
        ("v_cvt_f64_u32_e32", 0x16, &[0x7e08_2d05]), // v_cvt_f64_u32_e32 v[4:5], v5
        ("v_cls_i32_e32", 0x3b, &[0x7e02_7702]),     // v_cls_i32_e32 v1, v2
        ("v_min_f32_e32", 0x0f, &[0x1e10_0905]),     // v_min_f32_e32 v8, v5, v4
        ("v_cvt_pk_rtz_f16_f32_e32", 0x2f, &[0x5e02_0702]), // v_cvt_pk_rtz_f16_f32_e32 v1, v2, v3
        ("v_cmp_neq_f16_e64", 0x0d, &[0xd40d_0000, 0x0202_1680]), // v_cmp_neq_f16_e64 s0, 0, v11.l
        ("v_cmp_nlg_f32_e64", 0x1a, &[0xd41a_0000, 0x0202_0501]), // v_cmp_nlg_f32_e64 s0, v1, v2
        (
            "v_cmp_nle_f32_e64",
            0x1c,
            &[0xd41c_0002, 0x0202_22ff, 0x0d80_0000],
        ), // v_cmp_nle_f32_e64 s2, 0xd800000, v17
        ("v_cmp_eq_u64_e64", 0x5a, &[0xd45a_0000, 0x0202_3480]), // v_cmp_eq_u64_e64 s0, 0, v[26:27]
        ("v_cmpx_lt_i32_e64", 0xc1, &[0xd4c1_007e, 0x0202_2915]), // v_cmpx_lt_i32_e64 v21, v20
        ("v_cmpx_eq_u32_e64", 0xca, &[0xd4ca_007e, 0x0202_1204]), // v_cmpx_eq_u32_e64 s4, v9
        ("v_cmpx_gt_i64_e64", 0xd4, &[0xd4d4_007e, 0x0202_0610]), // v_cmpx_gt_i64_e64 s[16:17], v[3:4]
        ("v_cmpx_le_u64_e64", 0xdb, &[0xd4db_007e, 0x0202_0802]), // v_cmpx_le_u64_e64 s[2:3], v[4:5]
        ("v_cmpx_gt_u64_e64", 0xdc, &[0xd4dc_007e, 0x0202_0802]), // v_cmpx_gt_u64_e64 s[2:3], v[4:5]
        ("v_add_f32_e64", 0x103, &[0xd503_0001, 0x0202_0702]),    // v_add_f32_e64 v1, v2, v3
        ("v_min3_f32", 0x219, &[0xd619_0006, 0x0432_1706]),       // v_min3_f32 v6, v6, v11, v12
        ("v_xad_u32", 0x245, &[0xd645_0005, 0x0005_8300]),        // v_xad_u32 v5, v0, -1, s1
        ("v_subrev_co_u32", 0x302, &[0xd702_6a02, 0x0202_0002]), // v_subrev_co_u32 v2, vcc_lo, s2, v0
        ("v_sub_nc_u16", 0x304, &[0xd704_0004, 0x0202_2b16]),    // v_sub_nc_u16 v4.l, v22.l, v21.l
        ("v_mul_lo_u16", 0x305, &[0xd705_0002, 0x0202_0702]),    // v_mul_lo_u16 v2.l, v2.l, v3.l
        ("v_add_f64", 0x327, &[0xd727_0002, 0x0202_0d04]),       // v_add_f64 v[2:3], v[4:5], v[6:7]
        ("v_min_f64", 0x329, &[0xd729_0002, 0x0202_0d04]),       // v_min_f64 v[2:3], v[4:5], v[6:7]
        ("v_lshrrev_b64", 0x33d, &[0xd73d_0002, 0x0202_0d04]),   // v_lshrrev_b64 v[2:3], v4, v[6:7]
        ("v_dot2_f32_f16", 0x13, &[0xcc13_4000, 0x1c0e_0501]),   // v_dot2_f32_f16 v0, v1, v2, v3
        ("v_dot4_i32_iu8", 0x16, &[0xcc16_4000, 0x3c0e_0501]), // v_dot4_i32_iu8 v0, v1, v2, v3 neg_lo:[1,0,0]
        ("v_dot8_i32_iu4", 0x18, &[0xcc18_4000, 0x3c0e_0501]), // v_dot8_i32_iu4 v0, v1, v2, v3 neg_lo:[1,0,0]
        ("v_dot2_f32_bf16", 0x1a, &[0xcc1a_4000, 0x1c0e_0501]), // v_dot2_f32_bf16 v0, v1, v2, v3
        (
            "v_wmma_f32_16x16x16_bf16",
            0x41,
            &[0xcc41_4000, 0x1c02_2108],
        ), // v_wmma_f32_16x16x16_bf16 v[0:7], v[8:15], v[16:23], v[0:7]
        (
            "v_wmma_bf16_16x16x16_bf16",
            0x43,
            &[0xcc43_4000, 0x1c02_2108],
        ), // v_wmma_bf16_16x16x16_bf16 v[0:7], v[8:15], v[16:23], v[0:7]
        ("v_cmp_nle_f32_e32", 0x1c, &[0x7c38_32ff, 0x2680_0000]), // v_cmp_nle_f32_e32 vcc_lo, 0x26800000, v25
        ("v_cmp_lt_i64_e32", 0x51, &[0x7ca2_104a]), // v_cmp_lt_i64_e32 vcc_lo, s[74:75], v[8:9]
        ("v_cmp_eq_u64_e32", 0x5a, &[0x7cb4_0e80]), // v_cmp_eq_u64_e32 vcc_lo, 0, v[7:8]
        ("v_cmp_le_u64_e32", 0x5b, &[0x7cb6_0802]), // v_cmp_le_u64_e32 vcc_lo, s[2:3], v[4:5]
        ("v_cmp_gt_u64_e32", 0x5c, &[0x7cb8_0802]), // v_cmp_gt_u64_e32 vcc_lo, s[2:3], v[4:5]
        ("v_cmpx_le_f32_e32", 0x93, &[0x7d26_0880]), // v_cmpx_le_f32_e32 0, v4
        ("v_cmpx_nlg_f32_e32", 0x9a, &[0x7d34_02ff, 0xff80_0000]), // v_cmpx_nlg_f32_e32 0xff800000, v1
        ("v_cmpx_ngt_f32_e32", 0x9b, &[0x7d36_0104]),              // v_cmpx_ngt_f32_e32 v4, v0
        ("v_cmpx_eq_u16_e32", 0xba, &[0x7d74_1281]),               // v_cmpx_eq_u16_e32 1, v9.l
        ("v_cmpx_ne_u16_e32", 0xbd, &[0x7d7a_0280]),               // v_cmpx_ne_u16_e32 0, v1.l
        ("v_cmpx_lt_i64_e32", 0xd1, &[0x7da2_0802]), // v_cmpx_lt_i64_e32 s[2:3], v[4:5]
        ("v_cmpx_eq_u64_e32", 0xda, &[0x7db4_0802]), // v_cmpx_eq_u64_e32 s[2:3], v[4:5]
    ];
    /// Scalar fp rows that `llvm-mc -mcpu=gfx1100` rejects and only the RDNA3.5 XML defines:
    /// gfx1151.tbl deltas, absent from the gfx1100 base.
    const PM_FILL_GFX1151_ONLY: &[(&str, u16, &[u32])] = &[
        ("s_cvt_f32_f16", 0x69, &[0xbe90_6903]), // s_cvt_f32_f16 s16, s3
        ("s_cvt_hi_f32_f16", 0x6a, &[0xbe80_6a00]), // s_cvt_hi_f32_f16 s0, s0
        ("s_fmaak_f32", 0x45, &[0xa283_0503, 0x3586_37bd]), // s_fmaak_f32 s3, s3, s5, 0x358637bd
    ];

    fn fill_words(name: &str) -> &'static [u32] {
        for &(row, _, words) in PM_FILL_BASE.iter().chain(PM_FILL_GFX1151_ONLY) {
            if row == name {
                return words;
            }
        }
        panic!("no pm-opcode-fill row {name}");
    }

    /// Decode `words` on `arch`, require the named table row and opcode id, then re-encode
    /// byte-exact both from the decoded instruction and from its typed fields alone.
    fn pinned_row(arch: Arch, name: &str, opcode: u16, words: &[u32]) -> Inst {
        let (inst, used) =
            decode(arch, words).unwrap_or_else(|e| panic!("{arch:?} {name} {words:08x?}: {e}"));
        assert_eq!(used, words.len(), "{arch:?} {name}: width");
        assert_eq!(inst.op.id, opcode, "{arch:?} {name}: opcode id");
        let row = isa::lookup(arch, inst.op, inst.form).expect("decoded opcode has a table row");
        assert_eq!(row.name, name, "{arch:?} {words:08x?}");
        let table_words: Vec<u32> = row
            .encoding
            .split_whitespace()
            .map(|word| u32::from_str_radix(word, 16).unwrap())
            .collect();
        assert_eq!(
            table_words, words,
            "{arch:?} {name}: table sample is not the pinned llvm-mc encoding"
        );
        assert_eq!(
            encode(arch, &inst).unwrap().as_slice(),
            words,
            "{arch:?} {name}: re-encode"
        );
        let rebuilt = Inst::from_parts(
            arch,
            inst.op,
            inst.form,
            inst.fields.clone(),
            inst.operands.clone(),
            inst.mods.clone(),
            inst.literal,
            crate::provenance::Provenance::default(),
        )
        .unwrap_or_else(|e| panic!("{arch:?} {name}: {e}"));
        assert_eq!(
            encode(arch, &rebuilt).unwrap().as_slice(),
            words,
            "{arch:?} {name}: typed-field-only encode"
        );
        inst
    }

    #[test]
    fn pm_opcode_fill_rows_roundtrip_their_llvm_samples() {
        assert_eq!((PM_FILL_BASE.len(), PM_FILL_GFX1151_ONLY.len()), (61, 3));
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            for &(name, opcode, words) in PM_FILL_BASE {
                pinned_row(arch, name, opcode, words);
            }
        }
        for &(name, _, _) in PM_FILL_BASE {
            assert!(
                isa::gfx1100().iter().any(|row| row.name == name),
                "{name} missing from the gfx1100 base table"
            );
        }
        for &(name, opcode, words) in PM_FILL_GFX1151_ONLY {
            pinned_row(Arch::Gfx1151, name, opcode, words);
            let error = decode(Arch::Gfx1100, words).unwrap_err().to_string();
            assert!(error.contains("undefined"), "{name}: {error}");
            assert!(
                !isa::gfx1100().iter().any(|row| row.name == name),
                "{name} leaked into the gfx1100 base table"
            );
        }
    }

    /// `v_wmma_i32_16x16x16_iu8` (VOP3P/0x44) is a required useful identity; it must be in the
    /// gfx1100 base table itself, not only reachable through a gfx1151 delta.
    #[test]
    fn wmma_iu8_is_in_the_gfx1100_base_table() {
        assert!(isa::gfx1100()
            .iter()
            .any(|row| row.name == "v_wmma_i32_16x16x16_iu8" && row.op.id == 0x44));
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            pinned_row(
                arch,
                "v_wmma_i32_16x16x16_iu8",
                0x44,
                &[0xcc44_c08c, 0x7c02_4110],
            );
        }
    }

    /// The unsigned/signed split of the iu dot products is `neg_lo`; LLVM prints the
    /// `neg_lo:[1,1,0]` form as `v_dot4_i32_i8` / `v_dot8_i32_i4`, but it is the same row.
    #[test]
    fn dot_iu_rows_carry_signedness_in_neg_lo() {
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            for (name, opcode, mixed, signed) in [
                (
                    "v_dot4_i32_iu8",
                    0x16,
                    [0xcc16_4000, 0x3c0e_0501],
                    [0xcc16_4000, 0x7c0e_0501],
                ), // v_dot4_i32_i8 v0, v1, v2, v3
                (
                    "v_dot8_i32_iu4",
                    0x18,
                    [0xcc18_4000, 0x3c0e_0501],
                    [0xcc18_4000, 0x7c0e_0501],
                ), // v_dot8_i32_i4 v0, v1, v2, v3
            ] {
                assert_eq!(
                    pinned_row(arch, name, opcode, &mixed).mods.neg_lo,
                    0b001,
                    "{name}"
                );
                assert_eq!(
                    decode(arch, &signed).unwrap().0.mods.neg_lo,
                    0b011,
                    "{name}"
                );
                let (inst, _) = decode(arch, &signed).unwrap();
                assert_eq!(isa::lookup(arch, inst.op, inst.form).unwrap().name, name);
                assert_eq!(encode(arch, &inst).unwrap().as_slice(), &signed);
            }
        }
    }

    /// Defs/uses, implicit SCC/VCC/EXEC and wait-counter membership of the new rows, against
    /// the operand lists of `amdgpu_isa_rdna3{,_5}.xml` and the RDNA3 ISA guide (FLAT counts
    /// LGKMcnt in addition to VMcnt for loads / VScnt for stores).
    #[test]
    fn pm_opcode_fill_effects_follow_the_isa_xml() {
        use crate::{
            effects::{ImplicitSet, MemClass},
            reg::{Kind, RegRef},
        };
        let v = |base: u16, len: u8| RegRef {
            kind: Kind::V,
            base,
            len,
        };
        let s = |base: u16, len: u8| RegRef {
            kind: Kind::S,
            base,
            len,
        };
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            let at = |name: &str| {
                let opcode = PM_FILL_BASE
                    .iter()
                    .chain(PM_FILL_GFX1151_ONLY)
                    .find(|row| row.0 == name)
                    .map(|row| row.1)
                    .unwrap_or_else(|| panic!("no pm-opcode-fill row {name}"));
                pinned_row(arch, name, opcode, fill_words(name))
            };
            // DS: LGKMcnt only; returning atomics define VDST, non-returning ones define nothing.
            let add = at("ds_add_u32");
            let mem = add.effects.mem.expect("ds atomic memory effect");
            assert_eq!(mem.class, MemClass::DsAtomic { returns: false });
            assert!(mem.counters.contains(Counter::Lgkm) && !mem.counters.contains(Counter::Ds));
            assert!(add.effects.defs.is_empty());
            assert_eq!(add.effects.uses.as_slice(), &[v(7, 1), v(4, 1)]);
            let rtn = at("ds_add_rtn_u32");
            assert_eq!(
                rtn.effects.mem.unwrap().class,
                MemClass::DsAtomic { returns: true }
            );
            assert_eq!(rtn.effects.defs.as_slice(), &[v(1, 1)]);
            assert_eq!(rtn.effects.uses.as_slice(), &[v(1, 1), v(7, 1)]);
            for name in ["ds_or_b32", "ds_add_f32"] {
                assert_eq!(
                    at(name).effects.mem.unwrap().class,
                    MemClass::DsAtomic { returns: false },
                    "{name}"
                );
            }
            let load = at("ds_load_u16");
            assert_eq!(load.effects.mem.unwrap().class, MemClass::DsLoad);
            assert_eq!(
                (load.effects.defs.as_slice(), load.effects.uses.as_slice()),
                (&[v(136, 1)][..], &[v(21, 1)][..])
            );
            let store = at("ds_store_b96");
            assert_eq!(store.effects.mem.unwrap().class, MemClass::DsStore);
            assert!(store.effects.defs.is_empty());
            assert_eq!(store.effects.uses.as_slice(), &[v(1, 1), v(2, 3)]);
            // FLAT: both LGKMcnt and VScnt (stores) / VMcnt (loads).
            for name in ["flat_store_b8", "flat_store_b32", "flat_store_b128"] {
                let store = at(name);
                let mem = store.effects.mem.expect("flat store memory effect");
                assert_eq!(mem.class, MemClass::FlatStore, "{name}");
                assert!(
                    mem.counters.contains(Counter::Vs) && mem.counters.contains(Counter::Lgkm),
                    "{name}"
                );
                assert!(!mem.counters.contains(Counter::Vm), "{name}");
                assert!(store.effects.defs.is_empty(), "{name}");
            }
            assert_eq!(
                at("flat_store_b32").effects.uses.as_slice(),
                &[v(0, 2), v(2, 1)]
            );
            assert_eq!(
                at("flat_store_b128").effects.uses.as_slice(),
                &[v(24, 2), v(2, 4)]
            );
            for name in ["flat_load_d16_u8", "flat_load_d16_hi_b16"] {
                let load = at(name);
                let mem = load.effects.mem.expect("flat load memory effect");
                assert_eq!(mem.class, MemClass::FlatLoad, "{name}");
                assert!(
                    mem.counters.contains(Counter::Vm) && mem.counters.contains(Counter::Lgkm),
                    "{name}"
                );
                assert!(!mem.counters.contains(Counter::Vs), "{name}");
                assert_eq!(load.effects.defs.as_slice(), &[v(1, 1)], "{name}");
                // D16 writes half of VDST, so the old value is read.
                assert!(
                    load.effects.uses.contains(&v(1, 1)) && load.effects.uses.contains(&v(7, 2)),
                    "{name}"
                );
            }
            // SOP: s_bfe_i32 writes SCC (XML SSRC_SPECIAL_SCC output); s_pack_lh and s_sext do not.
            let bfe = at("s_bfe_i32");
            assert_eq!(
                (bfe.effects.implicit.writes, bfe.effects.implicit.reads),
                (ImplicitSet::SCC, 0)
            );
            assert_eq!(bfe.effects.defs.as_slice(), &[s(0, 1)]);
            for name in ["s_pack_lh_b32_b16", "s_sext_i32_i8"] {
                let inst = at(name);
                assert_eq!(
                    (inst.effects.implicit.writes, inst.effects.implicit.reads),
                    (0, 0),
                    "{name}"
                );
                assert_eq!(inst.effects.defs.len(), 1, "{name}");
            }
            // VOPC e32: v_cmp writes VCC, v_cmpx writes EXEC and defines no register.
            let cmp = at("v_cmp_lt_i64_e32");
            assert_eq!(
                (cmp.effects.implicit.writes, cmp.effects.implicit.reads),
                (ImplicitSet::VCC, 0)
            );
            assert!(cmp.effects.defs.is_empty());
            assert_eq!(cmp.effects.uses.as_slice(), &[s(74, 2), v(8, 2)]);
            for name in [
                "v_cmpx_le_f32_e32",
                "v_cmpx_nlg_f32_e32",
                "v_cmpx_ngt_f32_e32",
                "v_cmpx_eq_u16_e32",
                "v_cmpx_ne_u16_e32",
                "v_cmpx_lt_i64_e32",
                "v_cmpx_eq_u64_e32",
                "v_cmpx_gt_i64_e64",
                "v_cmpx_le_u64_e64",
                "v_cmpx_gt_u64_e64",
                "v_cmpx_lt_i32_e64",
                "v_cmpx_eq_u32_e64",
            ] {
                let inst = at(name);
                assert_eq!(
                    (inst.effects.implicit.writes, inst.effects.implicit.reads),
                    (ImplicitSet::EXEC, 0),
                    "{name}"
                );
                assert!(inst.effects.defs.is_empty(), "{name}");
            }
            // VOP3 compare and carry-out: the 64-bit SDST/VDST lane-mask slot is a real def.
            let cmp = at("v_cmp_eq_u64_e64");
            assert_eq!(cmp.effects.defs.as_slice(), &[s(0, 2)]);
            assert_eq!(cmp.effects.implicit.writes, 0);
            let sub = at("v_subrev_co_u32");
            assert_eq!(sub.effects.defs.as_slice(), &[v(2, 1)]);
            assert_eq!(
                (sub.effects.implicit.writes, sub.effects.implicit.reads),
                (ImplicitSet::VCC, 0)
            );
            assert_eq!(sub.effects.uses.as_slice(), &[s(2, 1), v(0, 1)]);
            // Mixed-width converts and shifts.
            let cvt = at("v_cvt_f64_i32_e32");
            assert_eq!(
                (cvt.effects.defs.as_slice(), cvt.effects.uses.as_slice()),
                (&[v(4, 2)][..], &[v(4, 1)][..])
            );
            let cvt = at("v_cvt_i32_f64_e32");
            assert_eq!(
                (cvt.effects.defs.as_slice(), cvt.effects.uses.as_slice()),
                (&[v(4, 1)][..], &[v(4, 2)][..])
            );
            let shift = at("v_lshrrev_b64");
            assert_eq!(
                (shift.effects.defs.as_slice(), shift.effects.uses.as_slice()),
                (&[v(2, 2)][..], &[v(4, 1), v(6, 2)][..])
            );
            // Wave32 WMMA bf16 operands are eight VGPRs each (XML OPR_VGPR/256).
            for name in ["v_wmma_f32_16x16x16_bf16", "v_wmma_bf16_16x16x16_bf16"] {
                let inst = at(name);
                assert_eq!(inst.effects.defs.as_slice(), &[v(0, 8)], "{name}");
                assert_eq!(
                    inst.effects.uses.as_slice(),
                    &[v(8, 8), v(16, 8), v(0, 8)],
                    "{name}"
                );
            }
            for name in [
                "v_dot2_f32_f16",
                "v_dot2_f32_bf16",
                "v_dot4_i32_iu8",
                "v_dot8_i32_iu4",
            ] {
                let inst = at(name);
                assert_eq!(inst.effects.defs.as_slice(), &[v(0, 1)], "{name}");
                assert_eq!(
                    inst.effects.uses.as_slice(),
                    &[v(1, 1), v(2, 1), v(3, 1)],
                    "{name}"
                );
            }
        }
        // gfx1151-only scalar fp.
        let cvt = pinned_row(
            Arch::Gfx1151,
            "s_cvt_f32_f16",
            0x69,
            fill_words("s_cvt_f32_f16"),
        );
        assert_eq!(
            (cvt.effects.defs.as_slice(), cvt.effects.uses.as_slice()),
            (&[s(16, 1)][..], &[s(3, 1)][..])
        );
        let fmaak = pinned_row(
            Arch::Gfx1151,
            "s_fmaak_f32",
            0x45,
            fill_words("s_fmaak_f32"),
        );
        assert_eq!(
            (fmaak.effects.defs.as_slice(), fmaak.effects.uses.as_slice()),
            (&[s(3, 1)][..], &[s(3, 1), s(5, 1)][..])
        );
        assert_eq!(
            (fmaak.effects.implicit.reads, fmaak.effects.implicit.writes),
            (0, 0)
        );
    }
}
