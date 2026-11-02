// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Wmma::swmmac`: the eleven gfx12 SWMMAC kinds. Texts and words are pinned
//! against `crates/peacemaker-ir/isa/gfx12.tbl` rows 502-511 and 135 and
//! against ROCm `llvm-mc -mcpu=gfx1201` (`-show-encoding`).

use peacemaker_ir::{
    codec::gfx12,
    inst::{Arch as IrArch, Inst},
};
use peacemaker_isa::{
    insn::{Instruction, Swmmac, SwmmacMods, Wmma},
    reg::{Kind, Live, RegRef, V},
    Arch, Builder, KernargLayout, KernelSpec, RegPlan,
};

fn vr(base: u8, len: u8) -> RegRef {
    RegRef {
        kind: Kind::V,
        base,
        len,
    }
}

/// `(kind, default sample text, default sample words)` from the gfx12.tbl sample column.
const DEFAULTS: [(Swmmac, &str, [u32; 2]); 11] = [
    (
        Swmmac::F32F16,
        "v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20",
        [0xCC504000, 0x1C521908],
    ),
    (
        Swmmac::F32Bf16,
        "v_swmmac_f32_16x16x32_bf16 v[0:7], v[8:11], v[12:19], v20",
        [0xCC514000, 0x1C521908],
    ),
    (
        Swmmac::F16F16,
        "v_swmmac_f16_16x16x32_f16 v[0:3], v[4:7], v[8:15], v16",
        [0xCC524000, 0x1C421104],
    ),
    (
        Swmmac::Bf16Bf16,
        "v_swmmac_bf16_16x16x32_bf16 v[0:3], v[4:7], v[8:15], v16",
        [0xCC534000, 0x1C421104],
    ),
    (
        Swmmac::I32Iu8,
        "v_swmmac_i32_16x16x32_iu8 v[0:7], v[8:9], v[10:13], v14",
        [0xCC544000, 0x1C3A1508],
    ),
    (
        Swmmac::I32Iu4,
        "v_swmmac_i32_16x16x32_iu4 v[0:7], v8, v[9:10], v11",
        [0xCC554000, 0x1C2E1308],
    ),
    (
        Swmmac::I32Iu4X64,
        "v_swmmac_i32_16x16x64_iu4 v[0:7], v[8:9], v[10:13], v14",
        [0xCC564000, 0x1C3A1508],
    ),
    (
        Swmmac::F32Fp8Fp8,
        "v_swmmac_f32_16x16x32_fp8_fp8 v[0:7], v[8:9], v[10:13], v14",
        [0xCC574000, 0x1C3A1508],
    ),
    (
        Swmmac::F32Fp8Bf8,
        "v_swmmac_f32_16x16x32_fp8_bf8 v[0:7], v[8:9], v[10:13], v14",
        [0xCC584000, 0x1C3A1508],
    ),
    (
        Swmmac::F32Bf8Fp8,
        "v_swmmac_f32_16x16x32_bf8_fp8 v[0:7], v[8:9], v[10:13], v14",
        [0xCC594000, 0x1C3A1508],
    ),
    (
        Swmmac::F32Bf8Bf8,
        "v_swmmac_f32_16x16x32_bf8_bf8 v[0:7], v[8:9], v[10:13], v14",
        [0xCC5A4000, 0x1C3A1508],
    ),
];

type Regs = (RegRef, RegRef, RegRef, V<1>);

/// The registers of the table sample: D at v0, then A, B and the index packed back to back.
fn sample_regs(kind: Swmmac) -> Regs {
    let (d, a, b) = kind.regs();
    (vr(0, d), vr(d, a), vr(d + a, b), V(d + a + b))
}

fn swmmac(kind: Swmmac, mods: SwmmacMods) -> Result<Instruction, String> {
    let (d, a, b, idx) = sample_regs(kind);
    Wmma::swmmac(Arch::Gfx1201, kind, d, a, b, idx, mods)
}

fn encode(text: &str) -> Vec<u32> {
    let inst = peacemaker_lift::text::parse_line(text, IrArch::Gfx1201)
        .unwrap_or_else(|e| panic!("{text}: {e}"));
    gfx12::encode(&inst)
        .unwrap_or_else(|e| panic!("{text}: {e}"))
        .to_vec()
}

fn decode_text(words: &[u32]) -> String {
    let (inst, n): (Inst, usize) = gfx12::decode(words).expect("decodes");
    assert_eq!(n, 2);
    inst.text(IrArch::Gfx1201).expect("renders")
}

#[test]
fn contract_tables() {
    assert_eq!(Swmmac::ALL.len(), 11);
    for (i, (kind, text, _)) in DEFAULTS.iter().enumerate() {
        assert_eq!(Swmmac::ALL[i], *kind);
        assert!(text.starts_with(kind.mnemonic()));
        let (d, a, b) = kind.regs();
        let (rd, ra, rb, _) = sample_regs(*kind);
        assert_eq!((rd.len, ra.len, rb.len), (d, a, b));
    }
}

#[test]
fn default_modifiers_emit_table_text_and_words() {
    for (kind, text, words) in DEFAULTS {
        let insn = swmmac(kind, SwmmacMods::default()).unwrap();
        assert_eq!(insn.text, text, "{kind:?}");
        assert_eq!(encode(&insn.text), words, "{kind:?}");
        assert_eq!(decode_text(&words), text, "{kind:?} round trip");
    }
}

#[test]
fn effects_read_the_destination_accumulator_in_place() {
    for kind in Swmmac::ALL {
        let (d, a, b, idx) = sample_regs(kind);
        let insn = swmmac(kind, SwmmacMods::default()).unwrap();
        assert_eq!(insn.defs, vec![d]);
        assert_eq!(insn.uses, vec![a, b, idx.reg(), d], "{kind:?}");
        // The IR table row agrees: SRC0, SRC1, SRC2 and VDST are all read.
        let parsed = peacemaker_lift::text::parse_line(&insn.text, IrArch::Gfx1201).unwrap();
        let reads_dst = parsed
            .effects
            .uses
            .iter()
            .any(|u| u.kind == peacemaker_ir::reg::Kind::V && u.base == 0 && u.len == d.len);
        assert!(
            reads_dst,
            "{kind:?}: table effects must read the accumulator"
        );
    }
}

fn pinned(kind: Swmmac, mods: SwmmacMods, text: &str, words: [u32; 2]) {
    let insn = swmmac(kind, mods).unwrap();
    assert_eq!(insn.text, text);
    assert_eq!(encode(text), words, "{text}");
    assert_eq!(decode_text(&words), text, "{text} round trip");
}

#[test]
fn llvm_mc_pinned_modifier_encodings() {
    pinned(
        Swmmac::F32F16,
        SwmmacMods {
            index_key: 1,
            neg_lo: 3,
            neg_hi: 3,
            clamp: false,
        },
        "v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20 index_key:1 neg_lo:[1,1,0] neg_hi:[1,1,0]",
        [0xcc504b00, 0x7c521908],
    );
    pinned(
        Swmmac::I32Iu8,
        SwmmacMods {
            index_key: 1,
            neg_lo: 3,
            neg_hi: 0,
            clamp: true,
        },
        "v_swmmac_i32_16x16x32_iu8 v[0:7], v[8:9], v[10:13], v14 index_key:1 neg_lo:[1,1,0] clamp",
        [0xcc54c800, 0x7c3a1508],
    );
    pinned(
        Swmmac::F32Fp8Fp8,
        SwmmacMods {
            index_key: 1,
            ..Default::default()
        },
        "v_swmmac_f32_16x16x32_fp8_fp8 v[0:7], v[8:9], v[10:13], v14 index_key:1",
        [0xcc574800, 0x1c3a1508],
    );
    pinned(
        Swmmac::Bf16Bf16,
        SwmmacMods {
            index_key: 1,
            ..Default::default()
        },
        "v_swmmac_bf16_16x16x32_bf16 v[0:3], v[4:7], v[8:15], v16 index_key:1",
        [0xcc534800, 0x1c421104],
    );
    pinned(
        Swmmac::F16F16,
        SwmmacMods {
            index_key: 1,
            ..Default::default()
        },
        "v_swmmac_f16_16x16x32_f16 v[0:3], v[4:7], v[8:15], v16 index_key:1",
        [0xcc524800, 0x1c421104],
    );
    pinned(
        Swmmac::F32F16,
        SwmmacMods {
            index_key: 1,
            ..Default::default()
        },
        "v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20 index_key:1",
        [0xcc504800, 0x1c521908],
    );
    // Further llvm-mc-probed words: x64 iu4 (no index_key), x32 iu4 with clamp,
    // bf16 neg_lo/neg_hi, bf8_bf8 index_key.
    pinned(
        Swmmac::I32Iu4X64,
        SwmmacMods {
            neg_lo: 1,
            clamp: true,
            ..Default::default()
        },
        "v_swmmac_i32_16x16x64_iu4 v[0:7], v[8:9], v[10:13], v14 neg_lo:[1,0,0] clamp",
        [0xcc56c000, 0x3c3a1508],
    );
    pinned(
        Swmmac::I32Iu4,
        SwmmacMods {
            index_key: 1,
            neg_lo: 2,
            clamp: true,
            ..Default::default()
        },
        "v_swmmac_i32_16x16x32_iu4 v[0:7], v8, v[9:10], v11 index_key:1 neg_lo:[0,1,0] clamp",
        [0xcc55c800, 0x5c2e1308],
    );
    pinned(
        Swmmac::Bf16Bf16,
        SwmmacMods {
            neg_lo: 1,
            neg_hi: 2,
            ..Default::default()
        },
        "v_swmmac_bf16_16x16x32_bf16 v[0:3], v[4:7], v[8:15], v16 neg_lo:[1,0,0] neg_hi:[0,1,0]",
        [0xcc534200, 0x3c421104],
    );
    pinned(
        Swmmac::F32Bf8Bf8,
        SwmmacMods {
            index_key: 1,
            ..Default::default()
        },
        "v_swmmac_f32_16x16x32_bf8_bf8 v[0:7], v[8:9], v[10:13], v14 index_key:1",
        [0xcc5a4800, 0x1c3a1508],
    );
}

#[test]
fn every_legal_modifier_combination_round_trips_through_the_codec() {
    let mut seen = 0;
    for kind in Swmmac::ALL {
        for index_key in 0..=1 {
            for neg_lo in 0..=3 {
                for neg_hi in 0..=3 {
                    for clamp in [false, true] {
                        let mods = SwmmacMods {
                            index_key,
                            neg_lo,
                            neg_hi,
                            clamp,
                        };
                        let Ok(insn) = swmmac(kind, mods) else {
                            continue;
                        };
                        seen += 1;
                        let words = encode(&insn.text);
                        assert_eq!(decode_text(&words), insn.text, "{kind:?} {mods:?}");
                    }
                }
            }
        }
    }
    // float16 x4: 2*4*4 = 32 each; int x32 x2: 2*4*2 = 16 each; x64: 4*2 = 8; fp8 x4: 2 each.
    assert_eq!(seen, 4 * 32 + 2 * 16 + 8 + 4 * 2);
}

#[test]
fn illegal_arch_width_and_modifiers_refuse() {
    for arch in [Arch::Gfx1100, Arch::Gfx1151] {
        let (d, a, b, idx) = sample_regs(Swmmac::F32F16);
        let err =
            Wmma::swmmac(arch, Swmmac::F32F16, d, a, b, idx, SwmmacMods::default()).unwrap_err();
        assert_eq!(err, "SWMMAC requires gfx12");
    }
    for kind in Swmmac::ALL {
        let name = kind.mnemonic();
        let (d, a, b, idx) = sample_regs(kind);
        let m = SwmmacMods::default();
        // Wrong width in each role, and a non-VGPR range.
        for (role, bad) in [("D", d), ("A", a), ("B", b)] {
            let with = |reg: RegRef| match role {
                "D" => (reg, a, b),
                "A" => (d, reg, b),
                _ => (d, a, reg),
            };
            let wrong = RegRef {
                len: bad.len + 1,
                ..bad
            };
            let (rd, ra, rb) = with(wrong);
            let err = Wmma::swmmac(Arch::Gfx1201, kind, rd, ra, rb, idx, m).unwrap_err();
            assert_eq!(
                err,
                format!("{name}: {role} requires {} VGPRs, found {wrong}", bad.len)
            );
            let scalar = RegRef {
                kind: Kind::S,
                ..bad
            };
            let (rd, ra, rb) = with(scalar);
            let err = Wmma::swmmac(Arch::Gfx1201, kind, rd, ra, rb, idx, m).unwrap_err();
            assert_eq!(
                err,
                format!("{name}: {role} requires {} VGPRs, found {scalar}", bad.len)
            );
        }
        let refuse = |mods: SwmmacMods, expect: String| {
            let err = swmmac(kind, mods).unwrap_err();
            assert_eq!(err, expect, "{kind:?} {mods:?}");
        };
        let float16 = matches!(
            kind,
            Swmmac::F32F16 | Swmmac::F32Bf16 | Swmmac::F16F16 | Swmmac::Bf16Bf16
        );
        let int = matches!(kind, Swmmac::I32Iu8 | Swmmac::I32Iu4 | Swmmac::I32Iu4X64);
        let fp8 = !float16 && !int;
        if kind == Swmmac::I32Iu4X64 {
            refuse(
                SwmmacMods { index_key: 1, ..m },
                format!("{name} has no index_key"),
            );
        } else {
            refuse(
                SwmmacMods { index_key: 2, ..m },
                format!("{name}: index_key must be 0 or 1"),
            );
        }
        for bits in [4, 5, 7] {
            for mods in [
                SwmmacMods { neg_lo: bits, ..m },
                SwmmacMods { neg_hi: bits, ..m },
            ] {
                refuse(
                    mods,
                    format!("{name}: neg_lo/neg_hi bit 2 (C) is not encodable"),
                );
            }
        }
        if fp8 {
            refuse(
                SwmmacMods { neg_lo: 1, ..m },
                format!("{name} does not accept neg_lo"),
            );
        }
        if !float16 {
            refuse(
                SwmmacMods { neg_hi: 1, ..m },
                format!("{name} does not accept neg_hi"),
            );
        }
        if !int {
            refuse(
                SwmmacMods { clamp: true, ..m },
                format!("{name} does not accept clamp"),
            );
        }
    }
}

fn spec() -> KernelSpec {
    KernelSpec {
        kernel_id: "swmmac_probe".into(),
        variant: "default".into(),
        arch: Arch::Gfx1201,
        symbol: "swmmac_probe".into(),
        kernargs: KernargLayout::new(8),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 32,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    }
}

/// The original defect: the constructor omitted the in-place accumulator read
/// of `dst` from `uses`. A builder driving the corrected constructor
/// finishes; the table row's effects are a strict superset of the old
/// declaration (the front end's `validate::instruction` refuses that one; see
/// `peacemaker-rip/tests/vocabulary.rs`).
#[test]
fn builder_accepts_swmmac_with_the_accumulator_in_uses() {
    let mut plan = RegPlan::new(32, 8).unwrap();
    let d = plan.v::<8>("d", 0, Live::Whole).unwrap();
    let a = plan.v::<4>("a", 8, Live::Whole).unwrap();
    let b = plan.v::<8>("b", 12, Live::Whole).unwrap();
    let idx = plan.v::<1>("idx", 20, Live::Whole).unwrap();
    let mut builder = Builder::new(spec(), plan);
    // Inputs and the accumulator are defined first.
    for reg in 0..21u8 {
        builder
            .push(Instruction::new(
                format!("v_mov_b32 v{reg}, 0"),
                vec![vr(reg, 1)],
                vec![],
            ))
            .unwrap();
    }
    let insn = Wmma::swmmac(
        Arch::Gfx1201,
        Swmmac::F32F16,
        d.reg(),
        a.reg(),
        b.reg(),
        idx,
        SwmmacMods::default(),
    )
    .unwrap();
    assert!(insn.uses.contains(&d.reg()));
    builder.push(insn).unwrap();
    builder
        .control(Instruction::new("s_endpgm", vec![], vec![]))
        .unwrap();
    let emitted = builder.finish().unwrap();
    assert!(emitted
        .s_text
        .contains("v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20"));
    let parsed = peacemaker_lift::text::parse_line(
        "v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20",
        IrArch::Gfx1201,
    )
    .unwrap();
    let declared_old = [a.reg(), b.reg(), idx.reg()];
    assert!(
        parsed.effects.uses.iter().any(|u| {
            let u = vr(u.base as u8, u.len);
            !declared_old.iter().any(|r| r.overlaps(u))
        }),
        "table effects must read a register the old declaration omitted"
    );
}
