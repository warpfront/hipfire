// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! RDNA3 / RDNA3.5 base fields, from isarefs MR-ISA `amdgpu_isa_rdna3{,_5}.xml`.
//! The two XMLs agree on bit layouts; opcode legality is selected by target.
use super::forms::Field;
use crate::inst::{Form, VmemForm};
macro_rules! f {
    ($n:literal, $b:literal, $w:literal) => {
        Field::new($n, $b, $w)
    };
}
const SMEM: &[Field] = &[
    f!("SDATA", 6, 7),
    f!("SBASE", 0, 6),
    f!("IOFFSET", 32, 21),
    f!("SOFFSET", 57, 7),
    f!("GLC", 14, 1),
    f!("DLC", 13, 1),
];
const GLOBAL: &[Field] = &[
    f!("VDST", 56, 8),
    f!("VSRC", 40, 8),
    f!("VADDR", 32, 8),
    f!("ADDR", 32, 8),
    f!("SADDR", 48, 7),
    f!("IOFFSET", 0, 13),
    f!("SVE", 55, 1),
    f!("GLC", 14, 1),
    f!("SLC", 15, 1),
    f!("DLC", 13, 1),
];
const BUFFER: &[Field] = &[
    f!("VDATA", 40, 8),
    f!("VADDR", 32, 8),
    f!("RSRC", 48, 5),
    f!("SOFFSET", 56, 8),
    f!("IOFFSET", 0, 12),
    f!("OFFEN", 54, 1),
    f!("IDXEN", 55, 1),
    f!("TFE", 53, 1),
    f!("GLC", 14, 1),
    f!("SLC", 12, 1),
    f!("DLC", 13, 1),
];
pub(super) fn field(form: Form, name: &str) -> Option<Field> {
    let layout = match form {
        Form::Smem => SMEM,
        Form::Vmem(VmemForm::Global | VmemForm::Scratch | VmemForm::Flat) => GLOBAL,
        Form::Vmem(VmemForm::Buffer) => BUFFER,
        _ => return super::forms::field(form, name),
    };
    layout.iter().copied().find(|f| f.name == name)
}
pub(super) fn opcode(form: Form) -> Option<Field> {
    Some(match form {
        Form::Smem => f!("OP", 18, 8),
        Form::Vop1 | Form::Vop1Dpp => f!("OP", 9, 8),
        Form::Vmem(VmemForm::Global | VmemForm::Scratch | VmemForm::Flat) => f!("OP", 18, 7),
        Form::Vmem(VmemForm::Buffer) => f!("OP", 18, 8),
        _ => return super::forms::opcode(form),
    })
}
pub(super) fn prefix(form: Form) -> Option<(u32, u32)> {
    Some(match form {
        Form::Smem => (0xfc00_0000, 0xf400_0000),
        Form::Vmem(VmemForm::Global) => (0xff03_0000, 0xdc02_0000),
        Form::Vmem(VmemForm::Scratch) => (0xff03_0000, 0xdc01_0000),
        Form::Vmem(VmemForm::Flat) => (0xff03_0000, 0xdc00_0000),
        Form::Vmem(VmemForm::Buffer) => (0xfc00_0000, 0xe000_0000),
        _ => return super::forms::prefix(form),
    })
}
