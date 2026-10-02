//! C5: per-kernel resource summary over typed streams.
//!
//! Scans explicit `Reg`/`Half` operands plus `Effects` for the
//! [`ResourceSummary`]: highest VGPR/SGPR dword touched, VCC and flat
//! scratch use, and the fixed LDS size handed in from the kernel
//! descriptor (KT48: 0 — the 49,936 B launch allocation is dynamic and
//! comes from the dispatch packet, never the object). Dynamic LDS is
//! always `None` for lifted kernels. Comparing the summary against the
//! descriptor granules and metadata is the certifier's check (§5.3); this
//! pass only measures.

use crate::cfg::Body;
use crate::effects::ImplicitSet;
use crate::operand::{Operand, Special};
use crate::reg::Kind;
use crate::state::ResourceSummary;

/// Measure VGPR/SGPR high-water marks and special-register use.
///
/// `lds_fixed` is `KernelDescriptor.group_segment_fixed_size`; it is a
/// parameter (not re-read here) so the measurement stays a pure function
/// of the instruction stream.
pub fn summarize(body: &Body, arch: crate::inst::Arch, wave: crate::inst::Wave, lds_fixed: u32) -> ResourceSummary {
    let mut max_vgpr = 0u16;
    let mut max_sgpr = 0u16;
    let mut uses_vcc = false;
    let mut uses_flat_scratch = false;
    for id in &body.layout {
        let Some(inst) = body.insts.get(*id) else {
            continue;
        };
        for (index, operand) in inst.operands.iter().enumerate() {
            match operand {
                Operand::Reg(reg) | Operand::Half(reg, _) => match reg.kind {
                    Kind::V => max_vgpr = max_vgpr.max(reg.base + u16::from(reg.len)),
                    Kind::S => {
                        let reg = crate::codec::gfx12::operand_register(arch, wave, inst, index).expect("register operand");
                        max_sgpr = max_sgpr.max(reg.base + u16::from(reg.len));
                    }
                    Kind::Ttmp => {}
                },
                Operand::Special(
                    Special::Vcc | Special::VccLo | Special::VccHi,
                ) => uses_vcc = true,
                Operand::Special(Special::FlatScratch) => uses_flat_scratch = true,
                _ => {}
            }
        }
        if inst.effects.implicit.reads & ImplicitSet::VCC != 0
            || inst.effects.implicit.writes & ImplicitSet::VCC != 0
        {
            uses_vcc = true;
        }
    }
    ResourceSummary {
        max_vgpr,
        max_sgpr,
        uses_vcc,
        uses_flat_scratch,
        lds_fixed,
        lds_dynamic_max: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::SmallVec;

    use crate::inst::{Arch, Form, Inst};
    use crate::operand::Modifiers;
    use crate::provenance::Provenance;

    fn inst(name: &str, operands: Vec<Operand>) -> Inst {
        let row = crate::isa::gfx12().iter().find(|row| row.name == name).expect(name);
        let form: Form = row.form;
        Inst::from_parts(
            Arch::Gfx1201,
            row.op,
            form,
            crate::inst::FormFields::None,
            SmallVec::from_vec(operands),
            Modifiers::default(),
            None,
            Provenance::default(),
        )
        .unwrap_or_else(|error| panic!("{name}: {error:?}"))
    }

    #[test]
    fn metadata_wave_width_controls_scalar_masks_not_data_pairs() {
        let s = |base, len| Operand::Reg(crate::reg::RegRef { kind: Kind::S, base, len });
        let mut body = Body::default();
        for i in [
            inst("v_cmp_nlt_f32_e64", vec![s(87, 2), v(0, 1), v(1, 1)]),
            inst("v_cndmask_b32_e64", vec![v(2, 1), v(0, 1), v(1, 1), s(87, 2)]),
        ] { let id = body.insts.insert(i); body.layout.push(id); }
        for (wave, expected) in [(crate::inst::Wave::Wave32, 88), (crate::inst::Wave::Wave64, 89)] {
            assert_eq!(summarize(&body, Arch::Gfx1201, wave, 0).max_sgpr, expected);
        }
        let pair = inst("s_mov_b64", vec![s(87, 2), s(0, 2)]);
        let id = body.insts.insert(pair); body.layout.push(id);
        assert_eq!(summarize(&body, Arch::Gfx1201, crate::inst::Wave::Wave32, 0).max_sgpr, 89);
    }

    fn v(base: u16, len: u8) -> Operand {
        Operand::Reg(crate::reg::RegRef { kind: Kind::V, base, len })
    }

    #[test]
    fn summary_measures_high_water_marks() {
        let mov = inst(
            "v_mov_b32_e32",
            vec![v(200, 1), Operand::Inline(crate::operand::InlineConst::Integer(1))],
        );
        let mut body = Body::default();
        let id = body.insts.insert(mov);
        body.layout.push(id);
        let summary = summarize(&body, Arch::Gfx1201, crate::inst::Wave::Wave32, 0);
        assert_eq!(summary.max_vgpr, 201);
        assert_eq!(summary.max_sgpr, 0);
        assert!(!summary.uses_vcc);
        assert_eq!(summary.lds_dynamic_max, None);
    }
}

#[cfg(test)]
mod kt48_tests {
    use crate::cfg::Body;
    use crate::inst::Arch;
    use crate::passes::cfg::build_blocks;

    fn kt48_body() -> Body {
        const IMAGE: &[u8] =
            include_bytes!("../../../peacemaker-lift/tests/fixtures/kt48/hipcc.co");
        const START: usize = 0x6f00;
        const SIZE: usize = 10_604;
        let words: Vec<u32> = IMAGE[START..START + SIZE]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let mut body = Body::default();
        let mut index = 0;
        while index < words.len() {
            let (inst, count) = crate::codec::gfx12::decode(&words[index..]).expect("KT48 decodes");
            let id = body.insts.insert(inst);
            body.layout.push(id);
            index += count;
        }
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).expect("CFG");
        body
    }

    /// Resource high-water marks stay inside the descriptor/metadata budget
    /// (`.vgpr_count` 238, `.sgpr_count` 30, fixed LDS 0).
    #[test]
    fn kt48_resource_summary() {
        let body = kt48_body();
        let summary = super::summarize(&body, Arch::Gfx1201, crate::inst::Wave::Wave32, 0);
        assert!(summary.max_vgpr <= 238, "vgpr {}", summary.max_vgpr);
        assert!(summary.max_sgpr <= 30, "sgpr {}", summary.max_sgpr);
        assert!(summary.uses_vcc);
        assert_eq!(summary.lds_fixed, 0);
        assert_eq!(summary.lds_dynamic_max, None);
    }
}
