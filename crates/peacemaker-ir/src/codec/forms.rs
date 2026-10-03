//! RDNA4 encoding layouts (AMD RDNA4 MR-ISA XML, ENC_* bitmaps).
use crate::inst::{Arch, Form, VmemForm};

#[derive(Clone, Copy)]
pub(super) struct Field { pub name: &'static str, pub bit: u8, pub width: u8 }
impl Field {
    pub const fn new(name: &'static str, bit: u8, width: u8) -> Self { Self { name, bit, width } }
    pub fn value(self, words: &[u32]) -> u32 {
        let index = usize::from(self.bit / 32);
        (words[index] >> (self.bit % 32)) & ((1u64 << self.width) - 1) as u32
    }
    pub fn set(self, words: &mut [u32], value: u32) -> Result<(), &'static str> {
        if u64::from(value) >= 1u64 << self.width { return Err(self.name); }
        let index = usize::from(self.bit / 32);
        let mask = (((1u64 << self.width) - 1) as u32) << (self.bit % 32);
        words[index] = (words[index] & !mask) | (value << (self.bit % 32));
        Ok(())
    }
    pub fn mask(self) -> u32 { (((1u64 << self.width) - 1) as u32) << (self.bit % 32) }
}
macro_rules! f { ($n:literal, $b:literal, $w:literal) => { Field::new($n,$b,$w) } }
const SOP1: &[Field] = &[f!("SDST",16,7),f!("SSRC0",0,8)];
const SOP2: &[Field] = &[f!("SDST",16,7),f!("SSRC0",0,8),f!("SSRC1",8,8)];
const SOPC: &[Field] = &[f!("SSRC0",0,8),f!("SSRC1",8,8)];
const SOPK: &[Field] = &[f!("SDST",16,7),f!("SIMM16",0,16)];
const SOPP: &[Field] = &[f!("SIMM16",0,16)];
const SMEM: &[Field] = &[f!("SDATA",6,7),f!("SBASE",0,6),f!("IOFFSET",32,24),f!("SOFFSET",57,7),f!("NV",20,1),f!("SCOPE",21,2),f!("TH",23,2)];
const VOP1: &[Field] = &[f!("VDST",17,8),f!("SRC0",0,9)];
const VOP2: &[Field] = &[f!("VDST",17,8),f!("SRC0",0,9),f!("VSRC1",9,8)];
const VOPC: &[Field] = &[f!("SRC0",0,9),f!("VSRC1",9,8)];
const VOP1_DPP: &[Field] = &[f!("VDST",17,8),f!("SRC0",32,8),f!("DPP_CTRL",40,9),f!("FI",50,1),f!("BOUND_CTRL",51,1),f!("SRC0_NEG",52,1),f!("SRC0_ABS",53,1),f!("SRC1_NEG",54,1),f!("SRC1_ABS",55,1),f!("BANK_MASK",56,4),f!("ROW_MASK",60,4)];
const VOP2_DPP: &[Field] = &[f!("VDST",17,8),f!("SRC0",32,8),f!("VSRC1",9,8),f!("DPP_CTRL",40,9),f!("FI",50,1),f!("BOUND_CTRL",51,1),f!("SRC0_NEG",52,1),f!("SRC0_ABS",53,1),f!("SRC1_NEG",54,1),f!("SRC1_ABS",55,1),f!("BANK_MASK",56,4),f!("ROW_MASK",60,4)];
const VOP3: &[Field] = &[f!("VDST",0,8),f!("SRC0",32,9),f!("SRC1",41,9),f!("SRC2",50,9),f!("ABS",8,3),f!("OPSEL",11,4),f!("CLAMP",15,1),f!("OMOD",59,2),f!("NEG",61,3)];
const VOP3P: &[Field] = &[f!("VDST",0,8),f!("SRC0",32,9),f!("SRC1",41,9),f!("SRC2",50,9),f!("NEG_HI",8,3),f!("OPSEL",11,3),f!("OPSEL_HI_LO",59,2),f!("OPSEL_HI_2",14,1),f!("CLAMP",15,1),f!("NEG",61,3)];
// VOPD VDSTY stores bits [7:1]; bit 0 is implicit and opposite VDSTX bit 0.
const VOPD: &[Field] = &[f!("VDSTX",56,8),f!("SRCX0",0,9),f!("VSRCX1",9,8),f!("VDSTY",49,7),f!("SRCY0",32,9),f!("VSRCY1",41,8)];
const DS: &[Field] = &[f!("VDST",56,8),f!("ADDR",32,8),f!("DATA0",40,8),f!("DATA1",48,8),f!("OFFSET0",0,8),f!("OFFSET1",8,8)];
const GLOBAL: &[Field] = &[f!("VDST",32,8),f!("VSRC",55,8),f!("VADDR",64,8),f!("SADDR",0,7),f!("IOFFSET",72,24),f!("NV",7,1),f!("SVE",49,1),f!("SCOPE",50,2),f!("TH",52,3)];
const SCRATCH: &[Field] = &[f!("VDST",32,8),f!("VSRC",55,8),f!("VADDR",64,8),f!("SADDR",0,7),f!("IOFFSET",72,24),f!("NV",7,1),f!("SVE",49,1),f!("SCOPE",50,2),f!("TH",52,3)];
const BUFFER: &[Field] = &[f!("VDATA",32,8),f!("VADDR",64,8),f!("RSRC",41,9),f!("SOFFSET",0,7),f!("IOFFSET",72,24),f!("NV",7,1),f!("OFFEN",62,1),f!("IDXEN",63,1),f!("SCOPE",50,2),f!("TH",52,3),f!("FORMAT",55,7),f!("TFE",22,1)];

pub(super) fn layout(form: Form) -> &'static [Field] {
    match form {
        Form::Sop1 => SOP1, Form::Sop2 => SOP2, Form::Sopc => SOPC, Form::Sopk => SOPK,
        Form::Sopp => SOPP, Form::Smem => SMEM, Form::Vop1 => VOP1, Form::Vop1Dpp => VOP1_DPP,
        Form::Vop2 => VOP2, Form::Vop2Dpp => VOP2_DPP, Form::Vopc => VOPC, Form::Vop3 => VOP3, Form::Vop3p => VOP3P, Form::Vopd => VOPD,
        // VFLAT shares the VGLOBAL bit layout; only the segment prefix differs.
        Form::Ds => DS, Form::Vmem(VmemForm::Global | VmemForm::Flat) => GLOBAL,
        Form::Vmem(VmemForm::Buffer) => BUFFER, Form::Vmem(VmemForm::Scratch) => SCRATCH, _ => &[],
    }
}
pub(super) fn field(form: Form, name: &str) -> Option<Field> {
    layout(form).iter().copied().find(|f| f.name == name)
}
pub(super) fn field_for(arch: Arch, form: Form, name: &str) -> Option<Field> {
    if matches!(arch, Arch::Gfx1100 | Arch::Gfx1151) {
        super::forms_gfx11::field(form, name)
    } else { field(form, name) }
}
pub(super) fn opcode(form: Form) -> Option<Field> {
    Some(match form {
        Form::Sop1 => f!("OP",8,8), Form::Sop2 => f!("OP",23,7),
        Form::Sopc | Form::Sopp => f!("OP",16,7), Form::Sopk => f!("OP",23,5),
        Form::Smem => f!("OP",13,6), Form::Vop1 | Form::Vop1Dpp => f!("OP",9,7),
        Form::Vop2 | Form::Vop2Dpp => f!("OP",25,6), Form::Vopc => f!("OP",17,8),
        Form::Vop3 => f!("OP",16,10), Form::Vop3p => f!("OP",16,7),
        Form::Vopd => f!("OPX",22,4), Form::Ds => f!("OP",18,8),
        Form::Vmem(VmemForm::Global | VmemForm::Buffer | VmemForm::Scratch | VmemForm::Flat) => f!("OP",14,8),
        _ => return None,
    })
}
pub(super) fn opcode_for(arch: Arch, form: Form) -> Option<Field> {
    if matches!(arch, Arch::Gfx1100 | Arch::Gfx1151) {
        super::forms_gfx11::opcode(form)
    } else { opcode(form) }
}
pub(super) fn prefix(form: Form) -> Option<(u32,u32)> {
    Some(match form {
        Form::Sop1 => (0xff80_0000,0xbe80_0000), Form::Sop2 => (0xc000_0000,0x8000_0000),
        Form::Sopc => (0xff80_0000,0xbf00_0000), Form::Sopk => (0xf000_0000,0xb000_0000),
        Form::Sopp => (0xff80_0000,0xbf80_0000), Form::Smem => (0xfc00_0000,0xf400_0000),
        Form::Vop1 | Form::Vop1Dpp => (0xfe00_0000,0x7e00_0000), Form::Vop2 | Form::Vop2Dpp => (0x8000_0000,0),
        Form::Vopc => (0xfe00_0000,0x7c00_0000), Form::Vop3 => (0xfc00_0000,0xd400_0000),
        Form::Vop3p => (0xff00_0000,0xcc00_0000), Form::Vopd => (0xfc00_0000,0xc800_0000),
        Form::Ds => (0xfc00_0000,0xd800_0000), Form::Vmem(VmemForm::Global) => (0xff00_0000,0xee00_0000),
        Form::Vmem(VmemForm::Buffer) => (0xfc00_0000,0xc400_0000),
        Form::Vmem(VmemForm::Scratch) => (0xff00_0000,0xed00_0000),
        Form::Vmem(VmemForm::Flat) => (0xff00_0000,0xec00_0000), _ => return None,
    })
}
pub(super) fn prefix_for(arch: Arch, form: Form) -> Option<(u32,u32)> {
    if matches!(arch, Arch::Gfx1100 | Arch::Gfx1151) {
        super::forms_gfx11::prefix(form)
    } else { prefix(form) }
}
