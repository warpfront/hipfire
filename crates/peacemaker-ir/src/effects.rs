// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::{
    cfg::{BarrierKind, Cond},
    inst::{Arch, Form, Inst, Opcode, ValidateError, Wave},
    lds::LdsAccess,
    operand::{CachePolicy, Operand, Special},
    reg::{Kind, RegRef},
    wait::{Counter, CounterSet},
};
use smallvec::SmallVec;
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ImplicitSet {
    pub reads: u8,
    pub writes: u8,
}
impl ImplicitSet {
    pub const SCC: u8 = 1 << 0;
    pub const VCC: u8 = 1 << 1;
    pub const EXEC: u8 = 1 << 2;
    pub const M0: u8 = 1 << 3;
    pub const MODE: u8 = 1 << 4;
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemClass {
    VmemLoad,
    VmemStore,
    VmemAtomic { returns: bool },
    DsLoad,
    DsStore,
    DsAtomic { returns: bool },
    SmemLoad,
    SmemStore,
    Export,
    LdsDma,
    FlatLoad,
    FlatStore,
    FlatAtomic { returns: bool },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrderType {
    Load,
    Store,
    Ds,
    Smem,
    Sample,
    Bvh,
    Exp,
    Mixed,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SrcRead {
    AtIssue,
    Deferred,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemEffect {
    pub class: MemClass,
    pub counters: CounterSet,
    pub in_order_type: OrderType,
    pub src_read: SrcRead,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Control {
    #[default]
    None,
    Branch {
        cond: Cond,
    },
    Jump,
    EndPgm,
    Barrier(BarrierKind),
    Wait,
    Clause,
    Delay,
    Halt,
    Trap,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Effects {
    pub defs: SmallVec<[RegRef; 2]>,
    pub uses: SmallVec<[RegRef; 4]>,
    pub implicit: ImplicitSet,
    pub mem: Option<MemEffect>,
    pub control: Control,
    pub lds: Option<LdsAccess>,
}
/// Direction of an explicit VALU lane-mask operand.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaneMask {
    /// A VOP3 `sdst` the instruction writes: compare result, carry/borrow out,
    /// `v_div_scale` flag, `v_mad_*64_*32` overflow (VOP3B).
    Out,
    /// The trailing lane-mask source: `v_cndmask_*` select, carry/borrow in.
    In,
}
/// Whether explicit operand `index` of `inst` is a VALU lane mask. The opcode
/// tables reserve a 64-bit slot for every such operand, but the hardware
/// width is one SGPR in wave32 and an SGPR pair in wave64; see [`wave_effects`].
/// Only VOP3 carries the mask as an explicit SGPR: the VOP2/VOPC encodings name
/// `vcc`, and `v_cmpx_*_e64` writes EXEC (its `src0` is not a mask).
pub fn lane_mask_role(arch: Arch, inst: &Inst, index: usize) -> Option<LaneMask> {
    if inst.form != Form::Vop3 {
        return None;
    }
    let name = inst.op.name(arch)?;
    let out = (name.starts_with("v_cmp_") && index == 0)
        || (index == 1
            && (name.starts_with("v_div_scale_")
                || name.starts_with("v_") && name.contains("_co_")
                || matches!(name, "v_mad_u64_u32" | "v_mad_i64_i32")));
    if out {
        return Some(LaneMask::Out);
    }
    let input = (name.starts_with("v_cndmask_")
        || name.starts_with("v_") && name.contains("_co_ci_"))
        && index + 1 == inst.operands.len();
    input.then_some(LaneMask::In)
}
/// `inst.effects` with every explicit SGPR lane mask narrowed to the kernel's
/// wave width (one SGPR in wave32, a pair in wave64). The table-derived
/// `inst.effects` stays wave-neutral (64-bit mask slots, the conservative
/// superset); a consumer that knows the wave size MUST use this instead, or a
/// wave32 compare into `s5` also "writes" `s6`.
pub fn wave_effects(arch: Arch, wave: Wave, inst: &Inst) -> Effects {
    let mut out = inst.effects.clone();
    let len = match wave {
        Wave::Wave32 => 1,
        Wave::Wave64 => 2,
    };
    for (index, operand) in inst.operands.iter().enumerate() {
        let Operand::Reg(mask) = operand else {
            continue;
        };
        if mask.kind != Kind::S {
            continue;
        }
        let list: &mut [RegRef] = match lane_mask_role(arch, inst, index) {
            Some(LaneMask::Out) => &mut out.defs,
            Some(LaneMask::In) => &mut out.uses,
            None => continue,
        };
        for reg in list.iter_mut().filter(|reg| *reg == mask) {
            reg.len = len;
        }
    }
    out
}
/// The old VGPR value preserved by a lane/half write.  This is not a
/// full-register definition: without lane-sensitive proof, an undefined old
/// destination must remain an open obligation rather than a proved read.
pub fn preserved_vgpr_destination(arch: Arch, inst: &Inst) -> Option<RegRef> {
    let writelane = inst.op.name(arch) == Some("v_writelane_b32");
    inst.operands.iter().enumerate().find_map(|(i, operand)| {
        let reg = match operand {
            Operand::Half(reg, _) => reg,
            Operand::Reg(reg) if writelane && i == 0 => reg,
            _ => return None,
        };
        if reg.kind != Kind::V || !inst.effects.defs.contains(reg) {
            return None;
        }
        // An explicit source overlapping the old destination remains a real
        // undefined read, not merely the value preserved by the partial write.
        let overlap = inst.operands.iter().enumerate().any(|(j, src)| {
            if j == i {
                return false;
            }
            let other = match src {
                Operand::Reg(r) | Operand::Half(r, _) => r,
                _ => return false,
            };
            other.kind == reg.kind
                && other.base < reg.base + u16::from(reg.len)
                && reg.base < other.base + u16::from(other.len)
        });
        (!overlap).then_some(*reg)
    })
}

impl Effects {
    /// Single table source for authoring and lifted instruction effects.
    /// Physical register widths come from decoded operands; implicit reads and
    /// writes, counter membership and opcode-specific accumulator uses come
    /// from the target's opcode table. `cpol` selects a VMEM atomic's returning
    /// or non-returning form (its `@rtn` destination slot and counter rules).
    pub fn from_table(
        arch: Arch,
        op: Opcode,
        form: Form,
        operands: &[Operand],
        cpol: &CachePolicy,
    ) -> Result<Self, ValidateError> {
        let row =
            crate::isa::lookup(arch, op, form).ok_or(ValidateError::UnknownOpcode { op, form })?;
        let returns = crate::isa::atomic_returns(arch, cpol);
        let mut out = Self::default();
        let mut field_names = row.slots(returns).map(|(name, _)| name);
        // CMPX encodes the fixed EXEC destination as VDST, but it is not an operand.
        if form == Form::Vop3 && row.name.starts_with("v_cmpx_") {
            field_names.next();
        }
        // VOPC's explicit printed `vcc_lo` destination is fieldless in XML.
        let prefix = usize::from(
            form == Form::Vopc
                && row.name.starts_with("v_cmp_")
                && matches!(
                    operands.first(),
                    Some(Operand::Special(Special::Vcc | Special::VccLo))
                ),
        );
        for operand in operands.iter().skip(prefix) {
            let name = field_names.next().unwrap_or("");
            let def = row.defs.split(',').any(|s| s == name);
            // A half-register destination leaves the other half intact.
            let used = row.uses.split(',').any(|s| s == name)
                || def && matches!(operand, Operand::Half(..));
            match operand {
                Operand::Reg(reg) | Operand::Half(reg, _) => {
                    if def {
                        out.defs.push(*reg);
                    }
                    if used {
                        out.uses.push(*reg);
                    }
                }
                Operand::Special(special) => {
                    let bit = match special {
                        Special::Scc => ImplicitSet::SCC,
                        Special::Vcc | Special::VccLo | Special::VccHi => ImplicitSet::VCC,
                        Special::Exec | Special::ExecLo | Special::ExecHi => ImplicitSet::EXEC,
                        Special::M0 => ImplicitSet::M0,
                        _ => 0,
                    };
                    if def {
                        out.implicit.writes |= bit;
                    }
                    if used {
                        out.implicit.reads |= bit;
                    }
                }
                _ => {}
            }
        }
        for effect in row.implicit.split(',') {
            let (reg, mode) = effect.split_once(':').unwrap_or(("", ""));
            let bit = match reg {
                "scc" => ImplicitSet::SCC,
                "vcc" => ImplicitSet::VCC,
                "exec" => ImplicitSet::EXEC,
                "m0" => ImplicitSet::M0,
                "mode" => ImplicitSet::MODE,
                _ => continue,
            };
            if mode.contains('r') {
                out.implicit.reads |= bit;
            }
            if mode.contains('w') {
                out.implicit.writes |= bit;
            }
        }
        let name = row.name;
        out.control = if name == "s_endpgm" {
            Control::EndPgm
        } else if name == "s_branch" {
            Control::Jump
        } else if name.starts_with("s_cbranch") {
            let cond = if name.ends_with("scc0") {
                Cond::Scc0
            } else if name.ends_with("scc1") {
                Cond::Scc1
            } else if name.ends_with("vccz") {
                Cond::Vccz
            } else if name.ends_with("vccnz") {
                Cond::Vccnz
            } else if name.ends_with("execnz") {
                Cond::Execnz
            } else {
                Cond::Execz
            };
            Control::Branch { cond }
        } else if name == "s_barrier" {
            Control::Barrier(BarrierKind::Full)
        } else if name == "s_barrier_signal" {
            Control::Barrier(BarrierKind::Signal(crate::cfg::BarrierId(0)))
        } else if name == "s_barrier_wait" {
            Control::Barrier(BarrierKind::Wait)
        } else if name.starts_with("s_wait") {
            Control::Wait
        } else if name == "s_clause" {
            Control::Clause
        } else if name == "s_delay_alu" {
            Control::Delay
        } else {
            Control::None
        };
        for (counter, units) in row.counter_rules(returns) {
            let counter = match counter {
                "Km" => Counter::Km,
                "Ds" => Counter::Ds,
                "Load" => Counter::Load,
                "Store" => Counter::Store,
                "Sample" => Counter::Sample,
                "Bvh" => Counter::Bvh,
                "Exp" => Counter::Exp,
                "Vm" => Counter::Vm,
                "Vs" => Counter::Vs,
                "Lgkm" => Counter::Lgkm,
                _ => return Err(ValidateError::Operand("unknown counter rule".into())),
            };
            if !matches!(units, "1" | "2") {
                return Err(ValidateError::Operand(
                    "unsupported counter unit weight".into(),
                ));
            }
            let (class, order) = if name.starts_with("s_load") {
                (MemClass::SmemLoad, OrderType::Smem)
            } else if name.starts_with("s_sendmsg_rtn") {
                (MemClass::SmemLoad, OrderType::Smem)
            } else if name.starts_with("ds_load") {
                (MemClass::DsLoad, OrderType::Ds)
            } else if name.starts_with("ds_store") {
                (MemClass::DsStore, OrderType::Ds)
            } else if name.starts_with("ds_") {
                (
                    MemClass::DsAtomic {
                        returns: !out.defs.is_empty(),
                    },
                    OrderType::Ds,
                )
            } else if name.starts_with("flat_load") {
                (MemClass::FlatLoad, OrderType::Load)
            } else if name.starts_with("flat_store") {
                (MemClass::FlatStore, OrderType::Store)
            } else if name.starts_with("global_load")
                || name.starts_with("buffer_load")
                || name.starts_with("scratch_load")
            {
                (MemClass::VmemLoad, OrderType::Load)
            } else if name.starts_with("global_store")
                || name.starts_with("buffer_store")
                || name.starts_with("scratch_store")
            {
                (MemClass::VmemStore, OrderType::Store)
            } else if name.starts_with("global_atomic") || name.starts_with("buffer_atomic") {
                (
                    MemClass::VmemAtomic { returns },
                    if returns {
                        OrderType::Load
                    } else {
                        OrderType::Store
                    },
                )
            } else if name.starts_with("flat_atomic") {
                (
                    MemClass::FlatAtomic { returns },
                    if returns {
                        OrderType::Load
                    } else {
                        OrderType::Store
                    },
                )
            } else {
                return Err(ValidateError::Operand(format!(
                    "unclassified memory rule for {name}"
                )));
            };
            let mem = out.mem.get_or_insert_with(|| MemEffect {
                class,
                counters: CounterSet::default(),
                in_order_type: order,
                src_read: if matches!(class, MemClass::DsStore) {
                    SrcRead::Deferred
                } else {
                    SrcRead::AtIssue
                },
            });
            mem.counters.insert(counter);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        codec::gfx12,
        inst::FormFields,
        operand::{InlineConst, Modifiers},
        provenance::Provenance,
        reg::Kind,
    };

    fn s(base: u16, len: u8) -> RegRef {
        RegRef {
            kind: Kind::S,
            base,
            len,
        }
    }
    fn v(base: u16, len: u8) -> RegRef {
        RegRef {
            kind: Kind::V,
            base,
            len,
        }
    }
    /// A VOP3 instruction built from the shipped row; operands in printed order.
    fn vop3(arch: Arch, name: &str, operands: &[RegRef]) -> Inst {
        let row = crate::isa::table(arch)
            .iter()
            .find(|row| row.name == name && row.form == Form::Vop3)
            .unwrap_or_else(|| panic!("{arch:?} {name}"));
        Inst::from_parts(
            arch,
            row.op,
            row.form,
            FormFields::None,
            operands.iter().map(|r| Operand::Reg(*r)).collect(),
            Modifiers::default(),
            None,
            Provenance::default(),
        )
        .unwrap_or_else(|e| panic!("{arch:?} {name}: {e}"))
    }
    fn sgprs(regs: &[RegRef]) -> Vec<RegRef> {
        regs.iter().copied().filter(|r| r.kind == Kind::S).collect()
    }
    /// Lane-mask def at `s5`: wave32 writes exactly `s5`, wave64 `s[5:6]`.
    fn assert_mask_def(arch: Arch, name: &str, operands: &[RegRef]) {
        let inst = vop3(arch, name, operands);
        let w32 = wave_effects(arch, Wave::Wave32, &inst);
        let w64 = wave_effects(arch, Wave::Wave64, &inst);
        assert_eq!(
            sgprs(&w32.defs),
            vec![s(5, 1)],
            "{arch:?} {name} wave32 defs"
        );
        assert_eq!(
            sgprs(&w64.defs),
            vec![s(5, 2)],
            "{arch:?} {name} wave64 defs"
        );
        // the wave-neutral table effect keeps the 64-bit slot
        assert_eq!(
            sgprs(&inst.effects.defs),
            vec![s(5, 2)],
            "{arch:?} {name} table defs"
        );
        // wave64 is the table's own 64-bit slot, unchanged
        assert_eq!(w64.uses, inst.effects.uses, "{arch:?} {name} wave64 uses");
    }
    /// Lane-mask use at `s7`: wave32 reads exactly `s7`, wave64 `s[7:8]`.
    fn assert_mask_use(arch: Arch, name: &str, operands: &[RegRef]) {
        let inst = vop3(arch, name, operands);
        let w32 = wave_effects(arch, Wave::Wave32, &inst);
        let w64 = wave_effects(arch, Wave::Wave64, &inst);
        assert!(
            w32.uses.contains(&s(7, 1)) && !w32.uses.contains(&s(7, 2)),
            "{arch:?} {name} wave32 uses {:?}",
            w32.uses
        );
        assert!(
            w64.uses.contains(&s(7, 2)) && !w64.uses.contains(&s(7, 1)),
            "{arch:?} {name} wave64 uses {:?}",
            w64.uses
        );
    }

    #[test]
    fn vop3_compare_sdst_is_one_sgpr_in_wave32() {
        for arch in [Arch::Gfx1100, Arch::Gfx1201] {
            for name in ["v_cmp_lt_i32_e64", "v_cmp_gt_f32_e64", "v_cmp_eq_u32_e64"] {
                if crate::isa::table(arch)
                    .iter()
                    .any(|r| r.name == name && r.form == Form::Vop3)
                {
                    assert_mask_def(arch, name, &[s(5, 2), v(0, 1), v(1, 1)]);
                }
            }
        }
    }
    #[test]
    fn vop3_64bit_compare_narrows_the_sdst_not_the_data_pair() {
        let mut found = 0;
        for arch in [Arch::Gfx1100, Arch::Gfx1201] {
            for name in ["v_cmp_lt_i64_e64", "v_cmp_gt_u64_e64", "v_cmp_eq_u64_e64"] {
                if crate::isa::table(arch)
                    .iter()
                    .any(|r| r.name == name && r.form == Form::Vop3)
                {
                    // src0 is a genuine scalar pair at s[2:3]; it is a use, never narrowed
                    let inst = vop3(arch, name, &[s(5, 2), s(2, 2), v(0, 2)]);
                    let w32 = wave_effects(arch, Wave::Wave32, &inst);
                    assert_eq!(sgprs(&w32.defs), vec![s(5, 1)], "{arch:?} {name}");
                    assert!(w32.uses.contains(&s(2, 2)), "{arch:?} {name}");
                    found += 1;
                }
            }
        }
        assert!(found > 0, "no 64-bit VOP3 compare row shipped");
    }
    #[test]
    fn vop3_cmpx_has_no_sdst_and_keeps_a_scalar_pair_src0() {
        for arch in [Arch::Gfx1100, Arch::Gfx1201] {
            let name = "v_cmpx_gt_i64_e64";
            if !crate::isa::table(arch)
                .iter()
                .any(|r| r.name == name && r.form == Form::Vop3)
            {
                continue;
            }
            let inst = vop3(arch, name, &[s(2, 2), v(0, 2)]);
            assert_eq!(lane_mask_role(arch, &inst, 0), None, "{arch:?}");
            for wave in [Wave::Wave32, Wave::Wave64] {
                let fx = wave_effects(arch, wave, &inst);
                assert!(
                    fx.uses.contains(&s(2, 2)),
                    "{arch:?} {wave:?} src0 pair kept: {:?}",
                    fx.uses
                );
            }
        }
    }
    #[test]
    fn carry_out_sdst_is_one_sgpr_in_wave32() {
        for arch in [Arch::Gfx1100, Arch::Gfx1201] {
            for name in ["v_add_co_u32", "v_sub_co_u32", "v_subrev_co_u32"] {
                assert_mask_def(arch, name, &[v(0, 1), s(5, 2), v(1, 1), v(2, 1)]);
            }
            for name in [
                "v_add_co_ci_u32_e64",
                "v_sub_co_ci_u32_e64",
                "v_subrev_co_ci_u32_e64",
            ] {
                assert_mask_def(arch, name, &[v(0, 1), s(5, 2), v(1, 1), v(2, 1), s(7, 2)]);
            }
        }
    }
    #[test]
    fn carry_in_and_cndmask_src2_is_one_sgpr_in_wave32() {
        for arch in [Arch::Gfx1100, Arch::Gfx1201] {
            for name in [
                "v_add_co_ci_u32_e64",
                "v_sub_co_ci_u32_e64",
                "v_subrev_co_ci_u32_e64",
            ] {
                assert_mask_use(arch, name, &[v(0, 1), s(5, 2), v(1, 1), v(2, 1), s(7, 2)]);
            }
            assert_mask_use(
                arch,
                "v_cndmask_b32_e64",
                &[v(0, 1), v(1, 1), v(2, 1), s(7, 2)],
            );
        }
    }
    #[test]
    fn v_div_scale_sdst_is_one_sgpr_in_wave32() {
        for arch in [Arch::Gfx1100, Arch::Gfx1201] {
            assert_mask_def(
                arch,
                "v_div_scale_f32",
                &[v(0, 1), s(5, 2), v(1, 1), v(2, 1), v(3, 1)],
            );
        }
    }
    #[test]
    fn mad_64bit_overflow_sdst_is_one_sgpr_in_wave32() {
        // gfx12 names it v_mad_co_u64_u32 / v_mad_co_i64_i32, gfx11 v_mad_u64_u32 / v_mad_i64_i32.
        // The 64-bit addend `src2` is data, not a mask: it stays a pair in both wave sizes.
        for (arch, name) in [
            (Arch::Gfx1201, "v_mad_co_u64_u32"),
            (Arch::Gfx1100, "v_mad_u64_u32"),
            (Arch::Gfx1100, "v_mad_i64_i32"),
        ] {
            let inst = vop3(arch, name, &[v(0, 2), s(5, 2), v(4, 1), v(5, 1), s(8, 2)]);
            let w32 = wave_effects(arch, Wave::Wave32, &inst);
            let w64 = wave_effects(arch, Wave::Wave64, &inst);
            assert_eq!(
                sgprs(&w32.defs),
                vec![s(5, 1)],
                "{arch:?} {name} wave32 defs"
            );
            assert_eq!(
                sgprs(&w64.defs),
                vec![s(5, 2)],
                "{arch:?} {name} wave64 defs"
            );
            for fx in [&w32, &w64] {
                assert!(
                    fx.uses.contains(&s(8, 2)),
                    "{arch:?} {name} addend pair kept"
                );
            }
        }
    }
    #[test]
    fn scalar_data_pairs_and_vector_defs_are_never_narrowed() {
        let arch = Arch::Gfx1201;
        // a VOP3 64-bit shift reads a genuine s[0:1] pair
        let one = Operand::Inline(InlineConst::Integer(1));
        let row = crate::isa::table(arch)
            .iter()
            .find(|r| r.name == "v_lshlrev_b64_e64")
            .expect("row");
        let inst = Inst::from_parts(
            arch,
            row.op,
            row.form,
            FormFields::None,
            [Operand::Reg(v(4, 2)), one, Operand::Reg(s(0, 2))]
                .into_iter()
                .collect(),
            Modifiers::default(),
            None,
            Provenance::default(),
        )
        .unwrap();
        assert_eq!(wave_effects(arch, Wave::Wave32, &inst), inst.effects);
    }

    #[test]
    fn partial_writes_depend_on_the_previous_destination() {
        for words in [
            &[0x7e00_1500][..],              // v_cvt_f16_f32_e32 v0.l, v0
            &[0xda98_08c0, 0x3e00_0032][..], // ds_load_u16_d16 v62, v50
            &[0xcc22_0011, 0x244e_2f1a][..], // v_fma_mixhi_f16 v17, ...
            &[0xd761_0061, 0x0201_0000][..], // v_writelane_b32 v97, ...
        ] {
            let (inst, _) = gfx12::decode(words).unwrap();
            let dest = inst
                .effects
                .defs
                .iter()
                .find(|r| r.kind == Kind::V)
                .unwrap();
            assert!(
                inst.effects.uses.contains(dest),
                "{:?} fails to read its preserved destination",
                inst.op
            );
        }
    }
}
