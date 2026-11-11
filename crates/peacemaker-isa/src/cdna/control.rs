// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Universal, block-local scalar facts for strict typed CDNA certification.
//!
//! CDNA3 ISA §12.1–12.5: MOV and CSELECT preserve SCC; AND writes SCC from
//! the entire result; CMP_EQ/LG_U32 writes its comparison result. These are
//! the only precise transfers here. Every other definition is forgotten via
//! the full typed footprint, including conditional writes and VALU SGPR defs.
//! Special registers remain unknown even after writes. No facts cross a CFG
//! block boundary and no fact belongs to a pending-memory/replay state.

use crate::ledger_control::{EdgeChoice, Facts, Truth, Word};
use peacemaker_cdna::enc::sop::{Sop1Op, Sop2Op, SopcOp, SoppOp};
use peacemaker_cdna::inst::{Footprint, RegFile};
use peacemaker_cdna::operand::{constant_bits, Src};
use peacemaker_cdna::Inst;

fn read32(src: Src, facts: &Facts) -> Word {
    if let Src::Sgpr(index) = src {
        facts.read32(usize::from(index))
    } else {
        constant_bits(src).map_or(Word::Unknown, Word::Const)
    }
}

fn read64(src: Src, facts: &Facts) -> [Word; 2] {
    match src {
        Src::Sgpr(index) => facts.read64(usize::from(index)),
        // Architectural inline integers sign-extend, including -1 in both dwords.
        Src::Int(value) => {
            let bits = i64::from(value) as u64;
            [Word::Const(bits as u32), Word::Const((bits >> 32) as u32)]
        }
        // General SALU raw-literal extension is explicitly unverified in the codec.
        // Do not learn either dword from it, a float inline, or a special register.
        _ => [Word::Unknown; 2],
    }
}

enum Write {
    One(Src, Word),
    Pair(Src, [Word; 2]),
}

/// Read all inputs before invalidating definitions; then commit only proven results.
/// `footprint` must be the complete typed footprint, before ledger bank filtering.
/// This function neither allocates nor retains a copy of the instruction or footprint.
pub(crate) fn transfer(inst: &Inst, footprint: &Footprint, facts: &mut Facts) {
    let (write, scc) = match inst {
        Inst::Sop1(i) if i.op == Sop1Op::MovB32 => (
            i.sdst
                .zip(i.ssrc0)
                .map(|(dst, src)| Write::One(dst, read32(src, facts))),
            None,
        ),
        Inst::Sop1(i) if i.op == Sop1Op::MovB64 => (
            i.sdst
                .zip(i.ssrc0)
                .map(|(dst, src)| Write::Pair(dst, read64(src, facts))),
            None,
        ),
        Inst::Sop2(i) if i.op == Sop2Op::AndB32 => {
            let result = read32(i.ssrc0, facts).and(read32(i.ssrc1, facts));
            (
                Some(Write::One(i.sdst, result)),
                Some(Truth::nonzero(&[result])),
            )
        }
        Inst::Sop2(i) if i.op == Sop2Op::AndB64 => {
            let a = read64(i.ssrc0, facts);
            let b = read64(i.ssrc1, facts);
            let result = [a[0].and(b[0]), a[1].and(b[1])];
            (
                Some(Write::Pair(i.sdst, result)),
                Some(Truth::nonzero(&result)),
            )
        }
        Inst::Sop2(i) if i.op == Sop2Op::CselectB32 => (
            Some(Write::One(
                i.sdst,
                Word::select(facts.scc, read32(i.ssrc0, facts), read32(i.ssrc1, facts)),
            )),
            None,
        ),
        Inst::Sop2(i) if i.op == Sop2Op::CselectB64 => {
            let a = read64(i.ssrc0, facts);
            let b = read64(i.ssrc1, facts);
            let result = [
                Word::select(facts.scc, a[0], b[0]),
                Word::select(facts.scc, a[1], b[1]),
            ];
            (Some(Write::Pair(i.sdst, result)), None)
        }
        Inst::Sopc(i) if matches!(i.op, SopcOp::CmpEqU32 | SopcOp::CmpLgU32) => {
            let truth = match (read32(i.ssrc0, facts), read32(i.ssrc1, facts)) {
                (Word::Const(a), Word::Const(b)) => {
                    if (a == b) == (i.op == SopcOp::CmpEqU32) {
                        Truth::True
                    } else {
                        Truth::False
                    }
                }
                _ => Truth::Unknown,
            };
            (None, Some(truth))
        }
        _ => (None, None),
    };

    facts.kill_defs(
        footprint
            .defs
            .iter()
            .filter(|span| span.file == RegFile::Sgpr)
            .map(|span| {
                let start = usize::from(span.base);
                start..start + usize::from(span.len)
            }),
        footprint.defs.iter().any(|span| span.file == RegFile::Scc),
    );
    match write {
        Some(Write::One(Src::Sgpr(index), value)) => facts.write32(usize::from(index), value),
        Some(Write::Pair(Src::Sgpr(index), value)) => facts.write64(usize::from(index), value),
        _ => {}
    }
    if let Some(value) = scc {
        facts.scc = value;
    }
}

/// Only SCC branches with universally known conditions admit a single edge.
pub(crate) fn branch_choice(inst: &Inst, facts: &Facts) -> EdgeChoice {
    match (inst, facts.scc) {
        (Inst::Sopp(i), Truth::True) if i.op == SoppOp::CbranchScc1 => EdgeChoice::Taken,
        (Inst::Sopp(i), Truth::False) if i.op == SoppOp::CbranchScc0 => EdgeChoice::Taken,
        (Inst::Sopp(i), Truth::False) if i.op == SoppOp::CbranchScc1 => EdgeChoice::Fallthrough,
        (Inst::Sopp(i), Truth::True) if i.op == SoppOp::CbranchScc0 => EdgeChoice::Fallthrough,
        _ => EdgeChoice::Both,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use peacemaker_cdna::enc::sop::{Sop1, Sop2, Sopc, Sopp};
    use peacemaker_cdna::enc::vop::Vop1Op;
    use peacemaker_cdna::operand::InlineFloat;
    use peacemaker_cdna::sim::{exec_one, Config, Flow, Memory, Wave};
    use peacemaker_cdna::Arch;

    fn mov32(dst: Src, src: Src) -> Inst {
        Inst::Sop1(Sop1 {
            op: Sop1Op::MovB32,
            sdst: Some(dst),
            ssrc0: Some(src),
        })
    }
    fn mov64(dst: Src, src: Src) -> Inst {
        Inst::Sop1(Sop1 {
            op: Sop1Op::MovB64,
            sdst: Some(dst),
            ssrc0: Some(src),
        })
    }
    fn binary(op: Sop2Op, dst: Src, a: Src, b: Src) -> Inst {
        Inst::Sop2(Sop2 {
            op,
            sdst: dst,
            ssrc0: a,
            ssrc1: b,
        })
    }
    fn compare(op: SopcOp, a: Src, b: Src) -> Inst {
        Inst::Sopc(Sopc {
            op,
            ssrc0: a,
            ssrc1: b,
        })
    }
    fn branch(op: SoppOp) -> Inst {
        Inst::Sopp(Sopp { op, simm16: 7 })
    }
    fn step(facts: &mut Facts, inst: Inst) {
        transfer(&inst, &inst.footprint(), facts);
    }

    #[test]
    fn zero_mask_certificate_agrees_with_concrete_scalar_execution() {
        let chain = [
            mov64(Src::Sgpr(0), Src::Int(0)),
            binary(Sop2Op::AndB64, Src::Sgpr(0), Src::Sgpr(0), Src::ExecLo),
            binary(Sop2Op::CselectB32, Src::Sgpr(0), Src::Int(1), Src::Int(0)),
            compare(SopcOp::CmpLgU32, Src::Sgpr(0), Src::Int(1)),
        ];
        let footprints: Vec<_> = chain.iter().map(Inst::footprint).collect();
        let jump = branch(SoppOp::CbranchScc1);
        let mut masks = vec![0, u64::MAX, 0xaaaa_aaaa_aaaa_aaaa, 0x5555_5555_5555_5555];
        masks.extend((0..64).map(|bit| 1u64 << bit));
        let mut rng = 0x92c4_8675_1234_0001u64;
        for _ in 0..512 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            masks.push(rng);
        }
        let mut wave = Wave::new(Arch::Gfx942, 0, 0);
        let mut facts = Facts::new(wave.s.len());
        let mut memory = Memory::default();
        let config = Config::default();
        for initial_scc in [false, true] {
            for poison in [0, 0xdead_beef, u32::MAX] {
                for &exec in &masks {
                    facts.reset_block();
                    wave.s.fill(poison);
                    wave.scc = initial_scc;
                    wave.exec = exec;
                    for (inst, footprint) in chain.iter().zip(&footprints) {
                        transfer(inst, footprint, &mut facts);
                        assert_eq!(
                            exec_one(&mut wave, inst, &mut [], &mut memory, &config).unwrap(),
                            Flow::Next
                        );
                        for index in 0..wave.s.len() {
                            if let Word::Const(expected) = facts.read32(index) {
                                assert_eq!(wave.s[index], expected, "{inst}, EXEC={exec:016x}");
                            }
                        }
                        match facts.scc {
                            Truth::True => assert!(wave.scc, "{inst}"),
                            Truth::False => assert!(!wave.scc, "{inst}"),
                            Truth::Unknown => {}
                        }
                    }
                    // Independent integer expectation: zero & EXEC = zero, select 0,
                    // then 0 != 1, for every EXEC and either incoming SCC.
                    assert_eq!(wave.s[0], 0);
                    assert!(wave.scc);
                    assert_eq!(facts.scc, Truth::True);
                    assert_eq!(branch_choice(&jump, &facts), EdgeChoice::Taken);
                    assert_eq!(
                        exec_one(&mut wave, &jump, &mut [], &mut memory, &config).unwrap(),
                        Flow::Branch(7)
                    );
                    assert_eq!(
                        branch_choice(&branch(SoppOp::CbranchScc0), &facts),
                        EdgeChoice::Fallthrough
                    );
                }
            }
        }
    }

    #[test]
    fn pair_aliases_inline_integer_widths_and_in_place_reads_are_exact() {
        let mut facts = Facts::new(102);
        facts.scc = Truth::True;
        step(&mut facts, mov64(Src::Sgpr(0), Src::Int(-1)));
        assert_eq!(facts.read64(0), [Word::Const(u32::MAX); 2]);
        assert_eq!(facts.scc, Truth::True);
        step(&mut facts, mov32(Src::Sgpr(1), Src::Lit(0x1234_5678)));
        assert_eq!(
            facts.read64(0),
            [Word::Const(u32::MAX), Word::Const(0x1234_5678)]
        );
        step(&mut facts, mov64(Src::Sgpr(2), Src::Sgpr(0)));
        assert_eq!(facts.read64(2), facts.read64(0));
        step(
            &mut facts,
            binary(Sop2Op::AndB64, Src::Sgpr(0), Src::Sgpr(0), Src::Int(-1)),
        );
        assert_eq!(
            facts.read64(0),
            [Word::Const(u32::MAX), Word::Const(0x1234_5678)]
        );
        assert_eq!(facts.scc, Truth::True);
        step(
            &mut facts,
            binary(Sop2Op::AndB64, Src::Sgpr(0), Src::Sgpr(0), Src::Int(0)),
        );
        assert_eq!(facts.read64(0), [Word::Const(0); 2]);
        assert_eq!(facts.scc, Truth::False);
        step(&mut facts, mov32(Src::Sgpr(1), Src::Sgpr(2)));
        assert_eq!(facts.read64(0), [Word::Const(0), Word::Const(u32::MAX)]);
    }

    #[test]
    fn unknown_selects_retain_only_equal_constant_dwords_and_preserve_scc() {
        let mut facts = Facts::new(102);
        facts.write64(0, [Word::Const(7), Word::Const(3)]);
        facts.write64(2, [Word::Const(7), Word::Const(4)]);
        step(
            &mut facts,
            binary(Sop2Op::CselectB64, Src::Sgpr(0), Src::Sgpr(0), Src::Sgpr(2)),
        );
        assert_eq!(facts.read64(0), [Word::Const(7), Word::Unknown]);
        assert_eq!(facts.scc, Truth::Unknown);
        step(
            &mut facts,
            binary(Sop2Op::CselectB32, Src::Sgpr(4), Src::Int(1), Src::Int(0)),
        );
        assert_eq!(facts.read32(4), Word::Unknown);
        step(
            &mut facts,
            compare(SopcOp::CmpEqU32, Src::Sgpr(4), Src::Int(1)),
        );
        assert_eq!(
            branch_choice(&branch(SoppOp::CbranchScc1), &facts),
            EdgeChoice::Both
        );
        facts.scc = Truth::False;
        step(
            &mut facts,
            binary(Sop2Op::CselectB32, Src::Sgpr(4), Src::Int(1), Src::Int(-1)),
        );
        assert_eq!(facts.read32(4), Word::Const(u32::MAX));
        assert_eq!(facts.scc, Truth::False);
    }

    #[test]
    fn ambiguous_wide_literals_and_all_special_sources_remain_unknown() {
        let mut facts = Facts::new(102);
        for src in [
            Src::Lit(0),
            Src::Lit(0xffff_ffff),
            Src::Float(InlineFloat::One),
            Src::ExecLo,
            Src::VccLo,
            Src::Scc,
            Src::Execz,
            Src::M0,
        ] {
            facts.write64(0, [Word::Const(0); 2]);
            step(&mut facts, mov64(Src::Sgpr(0), src));
            assert_eq!(facts.read64(0), [Word::Unknown; 2], "{src:?}");
            step(
                &mut facts,
                compare(SopcOp::CmpEqU32, Src::Sgpr(0), Src::Int(0)),
            );
            assert_eq!(facts.scc, Truth::Unknown);
        }
        step(&mut facts, mov32(Src::ExecLo, Src::Int(0)));
        step(&mut facts, mov32(Src::Sgpr(0), Src::ExecLo));
        assert_eq!(facts.read32(0), Word::Unknown);
        step(&mut facts, mov32(Src::Sgpr(0), Src::Lit(0x8000_0000)));
        assert_eq!(facts.read32(0), Word::Const(0x8000_0000));
        step(
            &mut facts,
            compare(SopcOp::CmpLgU32, Src::Sgpr(0), Src::Lit(0x8000_0000)),
        );
        assert_eq!(facts.scc, Truth::False);
    }

    #[test]
    fn unsupported_conditional_vector_and_scc_definitions_forget_exact_spans() {
        let mut facts = Facts::new(102);
        facts.write64(0, [Word::Const(0); 2]);
        facts.write32(2, Word::Const(9));
        facts.scc = Truth::False;
        step(
            &mut facts,
            Inst::Sop1(Sop1 {
                op: Sop1Op::CmovB32,
                sdst: Some(Src::Sgpr(0)),
                ssrc0: Some(Src::Int(1)),
            }),
        );
        assert_eq!(facts.read64(0), [Word::Unknown, Word::Const(0)]);
        assert_eq!(facts.scc, Truth::False);
        facts.write32(0, Word::Const(0));
        let vector = crate::cdna::ops::v1(Vop1Op::ReadfirstlaneB32, Src::Sgpr(1), Src::Vgpr(0));
        assert!(vector
            .footprint()
            .defs
            .iter()
            .any(|d| d.file == RegFile::Sgpr && d.base == 1));
        step(&mut facts, vector);
        assert_eq!(facts.read64(0), [Word::Const(0), Word::Unknown]);
        step(
            &mut facts,
            binary(Sop2Op::AndB64, Src::Sgpr(0), Src::Sgpr(0), Src::ExecLo),
        );
        assert_eq!(
            facts.scc,
            Truth::Unknown,
            "an unknown high half may be nonzero"
        );
        assert_eq!(facts.read32(2), Word::Const(9));
        facts.scc = Truth::True;
        step(
            &mut facts,
            binary(Sop2Op::AddU32, Src::Sgpr(8), Src::Int(0), Src::Int(0)),
        );
        assert_eq!(
            facts.scc,
            Truth::Unknown,
            "unsupported SCC writer is not a no-op"
        );
        assert_eq!(
            branch_choice(&branch(SoppOp::CbranchScc1), &facts),
            EdgeChoice::Both
        );
    }

    #[test]
    fn branch_polarity_is_exact_but_exec_vcc_and_unknown_scc_keep_both_edges() {
        let mut facts = Facts::new(102);
        for (truth, scc0, scc1) in [
            (Truth::True, EdgeChoice::Fallthrough, EdgeChoice::Taken),
            (Truth::False, EdgeChoice::Taken, EdgeChoice::Fallthrough),
            (Truth::Unknown, EdgeChoice::Both, EdgeChoice::Both),
        ] {
            facts.scc = truth;
            assert_eq!(branch_choice(&branch(SoppOp::CbranchScc0), &facts), scc0);
            assert_eq!(branch_choice(&branch(SoppOp::CbranchScc1), &facts), scc1);
            for op in [
                SoppOp::CbranchExecz,
                SoppOp::CbranchExecnz,
                SoppOp::CbranchVccz,
                SoppOp::CbranchVccnz,
                SoppOp::Branch,
            ] {
                assert_eq!(branch_choice(&branch(op), &facts), EdgeChoice::Both);
            }
        }
    }
}
