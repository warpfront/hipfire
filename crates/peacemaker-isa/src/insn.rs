// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::{
    arch::Arch,
    reg::{Kind, RegRef, S, V},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryClass {
    VmemLoad,
    VmemStore,
    DsLoad,
    DsStore,
    SmemLoad,
    Export,
}
#[derive(Clone, Debug)]
pub struct Instruction {
    pub text: String,
    pub defs: Vec<RegRef>,
    pub uses: Vec<RegRef>,
    pub memory: Option<MemoryClass>,
}
#[derive(Clone, Debug)]
pub struct Program {
    pub arch: Arch,
    pub instructions: Vec<Instruction>,
}
impl Instruction {
    pub fn new(text: impl Into<String>, defs: Vec<RegRef>, uses: Vec<RegRef>) -> Self {
        Self {
            text: text.into(),
            defs,
            uses,
            memory: None,
        }
    }
    pub fn memory(mut self, class: MemoryClass) -> Self {
        self.memory = Some(class);
        self
    }
    pub fn mnemonic(&self) -> &str {
        self.text.split_whitespace().next().unwrap_or("")
    }
    pub fn validate(&self, arch: Arch) -> Result<(), String> {
        let mnemonic = self.mnemonic();
        arch.check_mnemonic(mnemonic)?;
        if mnemonic.starts_with("scratch_") {
            return Err("scratch instructions are forbidden".into());
        }
        fn imm(text: &str) -> Result<u32, String> {
            let value = text.trim();
            if let Some(hex) = value.strip_prefix("0x") {
                u32::from_str_radix(hex, 16).map_err(|_| format!("invalid immediate {text}"))
            } else {
                value
                    .parse()
                    .map_err(|_| format!("invalid immediate {text}"))
            }
        }
        let arg = self.text.split_once(' ').map(|(_, arg)| arg).unwrap_or("");
        let maximum = match mnemonic {
            "s_clause" => Some(31),
            "s_wait_kmcnt" => Some(31),
            "s_wait_loadcnt" | "s_wait_dscnt" | "s_wait_storecnt" => Some(63),
            _ => None,
        };
        if let Some(max) = maximum {
            let value = imm(arg)?;
            if value > max || (mnemonic == "s_clause" && value == 0) {
                return Err(format!(
                    "{mnemonic} immediate {value} exceeds policy/encoding"
                ));
            }
        }
        if mnemonic == "s_wait_loadcnt_dscnt" {
            let value = imm(arg)?;
            if value & !0x3f3f != 0 {
                return Err(format!(
                    "combined wait {value:#x} exceeds either 6-bit counter field"
                ));
            }
        }
        if mnemonic.starts_with("buffer_") {
            // The offset is one token; llvm-objdump's canonical order puts cache
            // modifiers after it (`offen offset:136 scope:SCOPE_DEV`).
            if let Some((_, offset)) = self.text.rsplit_once(" offset:") {
                let value = imm(offset.split_whitespace().next().unwrap_or(""))?;
                if value > arch.buffer_offset_max() {
                    return Err(format!(
                        "buffer offset {value} exceeds {} field",
                        arch.name()
                    ));
                }
            }
        }
        if mnemonic.starts_with("ds_") {
            for token in self.text.split_whitespace() {
                if let Some((_, value)) = token
                    .split_once("offset0:")
                    .or_else(|| token.split_once("offset1:"))
                {
                    if imm(value)? > 255 {
                        return Err(format!("DS offset {value} exceeds 8-bit field"));
                    }
                }
            }
        }
        Ok(())
    }
    /// One plain instruction for a raw `push`: a single assembler statement
    /// (no line break, no label, no directive, lower-case mnemonic) that
    /// neither changes control flow nor touches a barrier. Branches, the
    /// program end and barriers are emitted by the typed core (`Auth`) or
    /// the untyped builder entry points.
    pub fn check_raw(&self) -> Result<(), String> {
        if self.text.chars().any(|c| c.is_control() && c != '\t') {
            return Err(format!(
                "{:?} is not a single assembler statement",
                self.text
            ));
        }
        let mnemonic = self.mnemonic();
        if !mnemonic.starts_with(|c: char| c.is_ascii_lowercase())
            || !mnemonic
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        {
            return Err(format!("{mnemonic:?} is not an instruction mnemonic"));
        }
        if control_flow(mnemonic) {
            return Err(format!("{mnemonic} changes control flow: branches and the program end go through the typed core"));
        }
        if mnemonic.contains("barrier") {
            return Err(format!(
                "{mnemonic} is a barrier: barriers go through the typed core"
            ));
        }
        Ok(())
    }
}
/// Branches, calls, returns, traps and the program end: every SOPP/SOP1
/// instruction that moves the program counter other than to the next one.
pub fn control_flow(mnemonic: &str) -> bool {
    [
        "s_branch",
        "s_cbranch",
        "s_setpc",
        "s_swappc",
        "s_call",
        "s_rfe",
        "s_endpgm",
        "s_trap",
        "s_sethalt",
        "s_sendmsghalt",
        "s_subvector_loop",
        "s_code_end",
    ]
    .iter()
    .any(|p| mnemonic.starts_with(p))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sop {
    Clause(u8),
    WaitLoad(u8),
    WaitDs(u8),
    WaitKm(u8),
    WaitStore(u8),
    End,
    Dealloc,
}
impl Sop {
    pub fn encode(self, arch: Arch) -> Result<Instruction, String> {
        let t = match self {
            Self::Clause(n) if (2..=32).contains(&n) => format!("s_clause {:#x}", n - 1),
            Self::WaitLoad(n) if n <= 63 => {
                if arch.gfx12() {
                    format!("s_wait_loadcnt {n:#x}")
                } else {
                    format!("s_waitcnt vmcnt({n})")
                }
            }
            Self::WaitDs(n) if n <= 63 => {
                if arch.gfx12() {
                    format!("s_wait_dscnt {n:#x}")
                } else {
                    format!("s_waitcnt lgkmcnt({n})")
                }
            }
            Self::WaitKm(n) if arch.gfx12() && n <= 31 => format!("s_wait_kmcnt {n:#x}"),
            Self::WaitStore(n) if arch.gfx12() && n <= 63 => format!("s_wait_storecnt {n:#x}"),
            Self::End => "s_endpgm".into(),
            Self::Dealloc => "s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)".into(),
            _ => return Err("invalid SOP immediate or unsupported architecture".into()),
        };
        Ok(Instruction::new(t, vec![], vec![]))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vbuffer {
    LoadB32,
    LoadB64,
    StoreB32,
    StoreB64,
}
impl Vbuffer {
    pub fn emit(
        self,
        arch: Arch,
        data: RegRef,
        voffset: RegRef,
        srd: RegRef,
        soffset: Option<RegRef>,
        offset: u32,
    ) -> Result<Instruction, String> {
        if offset > arch.buffer_offset_max() {
            return Err(format!(
                "buffer offset {offset} exceeds {} field",
                arch.name()
            ));
        }
        let (name, load, width) = match self {
            Self::LoadB32 => ("buffer_load_b32", true, 1),
            Self::LoadB64 => ("buffer_load_b64", true, 2),
            Self::StoreB32 => ("buffer_store_b32", false, 1),
            Self::StoreB64 => ("buffer_store_b64", false, 2),
        };
        if data.kind != Kind::V
            || data.len != width
            || voffset.kind != Kind::V
            || voffset.len != 1
            || srd.kind != Kind::S
            || srd.len != 4
            || soffset.is_some_and(|s| s.kind != Kind::S || s.len != 1)
        {
            return Err("invalid buffer operand width/class".into());
        }
        let off = soffset.map_or_else(
            || {
                if arch.gfx12() {
                    "null".into()
                } else {
                    "0".into()
                }
            },
            |r| r.to_string(),
        );
        let text = format!(
            "{name} {data}, {voffset}, {srd}, {off} offen{}",
            if offset == 0 {
                String::new()
            } else {
                format!(" offset:{offset}")
            }
        );
        let mut uses = vec![voffset, srd];
        if let Some(s) = soffset {
            uses.push(s)
        }
        let defs = if load {
            vec![data]
        } else {
            uses.push(data);
            vec![]
        };
        Ok(Instruction::new(text, defs, uses).memory(if load {
            MemoryClass::VmemLoad
        } else {
            MemoryClass::VmemStore
        }))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wmma {
    Iu4,
    Fp8,
}
impl Wmma {
    pub fn mnemonic(self, arch: Arch) -> Result<&'static str, String> {
        match self {
            Self::Iu4 => Ok(arch.wmma_iu4()),
            Self::Fp8 if arch == Arch::Gfx1201 => Ok("v_wmma_f32_16x16x16_fp8_fp8"),
            _ => Err("WMMA opcode requires gfx1201".into()),
        }
    }
}
impl Wmma {
    pub fn iu4(arch: Arch, dst: V<8>, a: V<2>, b: V<2>, acc: Option<V<8>>) -> Instruction {
        let mut uses = vec![a.reg(), b.reg()];
        let c = if let Some(acc) = acc {
            uses.push(acc.reg());
            acc.reg().to_string()
        } else {
            "0".into()
        };
        Instruction::new(
            format!(
                "{} {}, {}, {}, {c} neg_lo:[1,1,0]",
                arch.wmma_iu4(),
                dst.reg(),
                a.reg(),
                b.reg()
            ),
            vec![dst.reg()],
            uses,
        )
    }
    pub fn fp8(
        arch: Arch,
        dst: V<8>,
        w: V<2>,
        x: V<2>,
        acc: Option<V<8>>,
    ) -> Result<Instruction, String> {
        let opcode = Self::Fp8.mnemonic(arch)?;
        let mut uses = vec![w.reg(), x.reg()];
        let c = if let Some(acc) = acc {
            uses.push(acc.reg());
            acc.reg().to_string()
        } else {
            "0".into()
        };
        Ok(Instruction::new(
            format!("{opcode} {}, {}, {}, {c}", dst.reg(), w.reg(), x.reg()),
            vec![dst.reg()],
            uses,
        ))
    }
    /// Sparse (2:4 structured) WMMA, gfx12 only.
    ///
    /// The pinned table rows are `crates/peacemaker-ir/isa/gfx12.tbl` 502-511
    /// (the ten `v_swmmac_*x32_*` kinds) and 135 (`i32_16x16x64_iu4`); their
    /// operand column is `SRC0,SRC1,SRC2,VDST`, i.e. SWMMAC has no separate C
    /// operand: `dst` is read in place as the accumulator, so `dst` is in
    /// both `defs` and `uses`. `a` is the compressed A operand, `b` the dense
    /// B operand, `index` the sparsity index VGPR.
    ///
    /// Modifier acceptance matrix (probed with ROCm `llvm-mc -mcpu=gfx1201`;
    /// OP_SEL bit 0 carries `index_key`, bit 2 of `neg_lo`/`neg_hi` is never
    /// accepted):
    ///
    /// | kinds | index_key | neg_lo | neg_hi | clamp |
    /// |---|---|---|---|---|
    /// | `F32F16` `F32Bf16` `F16F16` `Bf16Bf16` | 0\|1 | bits 0-1 | bits 0-1 | no |
    /// | `I32Iu8` `I32Iu4` | 0\|1 | bits 0-1 | no | yes |
    /// | `I32Iu4X64` | no | bits 0-1 | no | yes |
    /// | `F32Fp8Fp8` `F32Fp8Bf8` `F32Bf8Fp8` `F32Bf8Bf8` | 0\|1 | no | no | no |
    pub fn swmmac(
        arch: Arch,
        kind: Swmmac,
        dst: RegRef,
        a: RegRef,
        b: RegRef,
        index: V<1>,
        mods: SwmmacMods,
    ) -> Result<Instruction, String> {
        if !arch.gfx12() {
            return Err("SWMMAC requires gfx12".into());
        }
        let name = kind.mnemonic();
        let (nd, na, nb) = kind.regs();
        for (role, reg, n) in [("D", dst, nd), ("A", a, na), ("B", b, nb)] {
            if reg.kind != Kind::V || reg.len != n {
                return Err(format!("{name}: {role} requires {n} VGPRs, found {reg}"));
            }
        }
        let class = kind.class();
        if class == SwmmacClass::Int64 {
            if mods.index_key != 0 {
                return Err(format!("{name} has no index_key"));
            }
        } else if mods.index_key > 1 {
            return Err(format!("{name}: index_key must be 0 or 1"));
        }
        if mods.neg_lo > 3 || mods.neg_hi > 3 {
            return Err(format!("{name}: neg_lo/neg_hi bit 2 (C) is not encodable"));
        }
        if class == SwmmacClass::Fp8 && mods.neg_lo != 0 {
            return Err(format!("{name} does not accept neg_lo"));
        }
        if class != SwmmacClass::Float16 && mods.neg_hi != 0 {
            return Err(format!("{name} does not accept neg_hi"));
        }
        if class != SwmmacClass::Int32 && class != SwmmacClass::Int64 && mods.clamp {
            return Err(format!("{name} does not accept clamp"));
        }
        let mut text = format!("{name} {dst}, {a}, {b}, {}", index.reg());
        if mods.index_key == 1 {
            text.push_str(" index_key:1");
        }
        for (label, bits) in [("neg_lo", mods.neg_lo), ("neg_hi", mods.neg_hi)] {
            if bits != 0 {
                text.push_str(&format!(" {label}:[{},{},0]", bits & 1, (bits >> 1) & 1));
            }
        }
        if mods.clamp {
            text.push_str(" clamp");
        }
        Ok(Instruction::new(
            text,
            vec![dst],
            vec![a, b, index.reg(), dst],
        ))
    }
}
/// The SWMMAC opcodes of gfx12 (`v_swmmac_*`), named `{D}{A}{B}` after the
/// accumulator/A/B element types (`Iu4X64` is the K=64 iu4 row).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Swmmac {
    F32F16,
    F32Bf16,
    F16F16,
    Bf16Bf16,
    I32Iu8,
    I32Iu4,
    I32Iu4X64,
    F32Fp8Fp8,
    F32Fp8Bf8,
    F32Bf8Fp8,
    F32Bf8Bf8,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum SwmmacClass {
    Float16,
    Int32,
    Int64,
    Fp8,
}
impl Swmmac {
    pub const ALL: [Swmmac; 11] = [
        Self::F32F16,
        Self::F32Bf16,
        Self::F16F16,
        Self::Bf16Bf16,
        Self::I32Iu8,
        Self::I32Iu4,
        Self::I32Iu4X64,
        Self::F32Fp8Fp8,
        Self::F32Fp8Bf8,
        Self::F32Bf8Fp8,
        Self::F32Bf8Bf8,
    ];
    pub fn mnemonic(self) -> &'static str {
        match self {
            Self::F32F16 => "v_swmmac_f32_16x16x32_f16",
            Self::F32Bf16 => "v_swmmac_f32_16x16x32_bf16",
            Self::F16F16 => "v_swmmac_f16_16x16x32_f16",
            Self::Bf16Bf16 => "v_swmmac_bf16_16x16x32_bf16",
            Self::I32Iu8 => "v_swmmac_i32_16x16x32_iu8",
            Self::I32Iu4 => "v_swmmac_i32_16x16x32_iu4",
            Self::I32Iu4X64 => "v_swmmac_i32_16x16x64_iu4",
            Self::F32Fp8Fp8 => "v_swmmac_f32_16x16x32_fp8_fp8",
            Self::F32Fp8Bf8 => "v_swmmac_f32_16x16x32_fp8_bf8",
            Self::F32Bf8Fp8 => "v_swmmac_f32_16x16x32_bf8_fp8",
            Self::F32Bf8Bf8 => "v_swmmac_f32_16x16x32_bf8_bf8",
        }
    }
    /// VGPR counts `(dst, a, b)`; the sparse index is always one VGPR.
    pub fn regs(self) -> (u8, u8, u8) {
        match self {
            Self::F32F16 | Self::F32Bf16 => (8, 4, 8),
            Self::F16F16 | Self::Bf16Bf16 => (4, 4, 8),
            Self::I32Iu8 => (8, 2, 4),
            Self::I32Iu4 => (8, 1, 2),
            Self::I32Iu4X64 => (8, 2, 4),
            Self::F32Fp8Fp8 | Self::F32Fp8Bf8 | Self::F32Bf8Fp8 | Self::F32Bf8Bf8 => (8, 2, 4),
        }
    }
    fn class(self) -> SwmmacClass {
        match self {
            Self::F32F16 | Self::F32Bf16 | Self::F16F16 | Self::Bf16Bf16 => SwmmacClass::Float16,
            Self::I32Iu8 | Self::I32Iu4 => SwmmacClass::Int32,
            Self::I32Iu4X64 => SwmmacClass::Int64,
            Self::F32Fp8Fp8 | Self::F32Fp8Bf8 | Self::F32Bf8Fp8 | Self::F32Bf8Bf8 => {
                SwmmacClass::Fp8
            }
        }
    }
}
/// SWMMAC modifiers: `index_key` (0|1, OP_SEL bit 0), `neg_lo`/`neg_hi`
/// (bit 0 = A, bit 1 = B; bit 2 is never encodable) and `clamp`. Which are
/// legal depends on the kind; see [`Wmma::swmmac`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SwmmacMods {
    pub index_key: u8,
    pub neg_lo: u8,
    pub neg_hi: u8,
    pub clamp: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ds {
    Load2B64,
    Load2Stride64B64,
    Store2Stride64B64,
    Load2B32,
}
impl Ds {
    pub fn offset(self, offset0: u16, offset1: u16) -> Result<String, String> {
        if offset0 > 255 || offset1 > 255 {
            return Err("DS offset exceeds 8 bits".into());
        }
        Ok(format!(
            "{} offset0:{offset0} offset1:{offset1}",
            match self {
                Self::Load2B64 => "ds_load_2addr_b64",
                Self::Load2Stride64B64 => "ds_load_2addr_stride64_b64",
                Self::Store2Stride64B64 => "ds_store_2addr_stride64_b64",
                Self::Load2B32 => "ds_load_2addr_b32",
            }
        ))
    }
}
impl Ds {
    pub fn load(
        self,
        dst: RegRef,
        addr: V<1>,
        offset0: u16,
        offset1: u16,
    ) -> Result<Instruction, String> {
        let expected = match self {
            Self::Load2B32 => 2,
            Self::Load2B64 | Self::Load2Stride64B64 => 4,
            Self::Store2Stride64B64 => return Err("use Ds::store for a DS store".into()),
        };
        if dst.kind != Kind::V || dst.len != expected {
            return Err("DS load destination width differs from opcode".into());
        }
        let name = self.offset(offset0, offset1)?;
        let (mnemonic, offsets) = name.split_once(' ').ok_or("invalid DS offset encoding")?;
        Ok(Instruction::new(
            format!("{mnemonic} {dst}, {} {offsets}", addr.reg()),
            vec![dst],
            vec![addr.reg()],
        )
        .memory(MemoryClass::DsLoad))
    }
    pub fn store(
        self,
        addr: V<1>,
        first: V<2>,
        second: V<2>,
        offset0: u16,
        offset1: u16,
    ) -> Result<Instruction, String> {
        if self != Self::Store2Stride64B64 {
            return Err("DS store requires Store2Stride64B64".into());
        }
        let name = self.offset(offset0, offset1)?;
        let (mnemonic, offsets) = name.split_once(' ').ok_or("invalid DS offset encoding")?;
        Ok(Instruction::new(
            format!(
                "{mnemonic} {}, {}, {} {offsets}",
                addr.reg(),
                first.reg(),
                second.reg()
            ),
            vec![],
            vec![addr.reg(), first.reg(), second.reg()],
        )
        .memory(MemoryClass::DsStore))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vop1 {
    CvtF32I32,
    MovB32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vop2 {
    XorB32,
    MulF32,
    FmacF32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vop3 {
    FmaF32,
    AddNcU32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vop3p {
    Dot4I32Iu8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Global {
    LoadB32,
    LoadB64,
    StoreB32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Smem {
    LoadB256,
}
impl Smem {
    pub fn load_b256(self, dst: S<8>, base: S<2>, offset: u32) -> Instruction {
        Instruction::new(
            format!("s_load_b256 {}, {}, {offset:#x}", dst.reg(), base.reg()),
            vec![dst.reg()],
            vec![base.reg()],
        )
        .memory(MemoryClass::SmemLoad)
    }
}
impl Vop1 {
    pub fn emit(self, dst: V<1>, src: V<1>) -> Instruction {
        let mnemonic = match self {
            Self::CvtF32I32 => "v_cvt_f32_i32_e32",
            Self::MovB32 => "v_mov_b32_e32",
        };
        Instruction::new(
            format!("{mnemonic} {}, {}", dst.reg(), src.reg()),
            vec![dst.reg()],
            vec![src.reg()],
        )
    }
    pub fn mov_literal(dst: V<1>, literal: u32) -> Instruction {
        Instruction::new(
            format!("v_mov_b32_e32 {}, {literal:#x}", dst.reg()),
            vec![dst.reg()],
            vec![],
        )
    }
}
impl Vop2 {
    pub fn emit(self, dst: V<1>, src0: V<1>, src1: V<1>) -> Instruction {
        let mnemonic = match self {
            Self::XorB32 => "v_xor_b32_e32",
            Self::MulF32 => "v_mul_f32_e32",
            Self::FmacF32 => "v_fmac_f32_e32",
        };
        let mut uses = vec![src0.reg(), src1.reg()];
        if self == Self::FmacF32 {
            uses.push(dst.reg())
        }
        Instruction::new(
            format!("{mnemonic} {}, {}, {}", dst.reg(), src0.reg(), src1.reg()),
            vec![dst.reg()],
            uses,
        )
    }
    pub fn xor_literal(dst: V<1>, literal: u32, src: V<1>) -> Instruction {
        Instruction::new(
            format!("v_xor_b32_e32 {}, {literal:#x}, {}", dst.reg(), src.reg()),
            vec![dst.reg()],
            vec![src.reg()],
        )
    }
}
impl Vop3 {
    pub fn fma(dst: V<1>, a: V<1>, b: V<1>, c: V<1>) -> Instruction {
        Instruction::new(
            format!(
                "v_fma_f32 {}, {}, {}, {}",
                dst.reg(),
                a.reg(),
                b.reg(),
                c.reg()
            ),
            vec![dst.reg()],
            vec![a.reg(), b.reg(), c.reg()],
        )
    }
}
impl Global {
    pub fn load_b32(dst: V<1>, addr: V<1>, base: S<2>) -> Instruction {
        Instruction::new(
            format!(
                "global_load_b32 {}, {}, {}",
                dst.reg(),
                addr.reg(),
                base.reg()
            ),
            vec![dst.reg()],
            vec![addr.reg(), base.reg()],
        )
        .memory(MemoryClass::VmemLoad)
    }
    pub fn store_b32(addr: V<1>, data: V<1>, base: S<2>) -> Instruction {
        Instruction::new(
            format!(
                "global_store_b32 {}, {}, {}",
                addr.reg(),
                data.reg(),
                base.reg()
            ),
            vec![],
            vec![addr.reg(), data.reg(), base.reg()],
        )
        .memory(MemoryClass::VmemStore)
    }
}
impl Global {
    pub fn load_b64(dst: V<2>, addr: V<1>, base: S<2>) -> Instruction {
        Instruction::new(
            format!(
                "global_load_b64 {}, {}, {}",
                dst.reg(),
                addr.reg(),
                base.reg()
            ),
            vec![dst.reg()],
            vec![addr.reg(), base.reg()],
        )
        .memory(MemoryClass::VmemLoad)
    }
}

#[cfg(test)]
mod buffer_offset_tests {
    use super::*;
    #[test]
    fn buffer_offset_is_one_token_before_cache_modifiers() {
        let ok = Instruction::new(
            "buffer_load_b32 v1, v2, s[4:7], null offen offset:136 scope:SCOPE_DEV",
            vec![],
            vec![],
        );
        assert!(ok.validate(Arch::Gfx1201).is_ok());
        let big = Instruction::new(
            "buffer_load_b32 v1, v2, s[4:7], null offen offset:8388608 scope:SCOPE_DEV",
            vec![],
            vec![],
        );
        assert!(big.validate(Arch::Gfx1201).is_err());
        let bad = Instruction::new(
            "buffer_load_b32 v1, v2, s[4:7], null offen offset:x scope:SCOPE_DEV",
            vec![],
            vec![],
        );
        assert!(bad.validate(Arch::Gfx1201).is_err());
    }
}

pub const GFX1201_GOLDEN: &[&str] = &[
    "s_wait_loadcnt 0x3",
    "s_wait_dscnt 0x1",
    "s_wait_loadcnt_dscnt 0x205",
    "s_wait_kmcnt 0x0",
    "s_wait_alu depctr_sa_sdst(0)",
    "s_wait_alu depctr_va_sdst(0)",
    "s_barrier_signal -1",
    "s_barrier_wait 0xffff",
    "global_inv scope:SCOPE_SE",
    "v_wmma_i32_16x16x32_iu4 v[57:64], v[71:72], v[1:2], 0 neg_lo:[1,1,0]",
    "v_swmmac_i32_16x16x64_iu4 v[0:7], v[8:9], v[10:13], v14",
    "v_dual_mul_f32 v83, v97, v81 :: v_dual_mul_f32 v84, v98, v81",
    "v_dual_fmac_f32 v163, v83, v57 :: v_dual_fmac_f32 v166, v84, v58",
    "buffer_load_b64 v[0:1], v2, s[4:7], s8 offen",
    "ds_load_2addr_stride64_b64 v[71:74], v81 offset1:1",
    "s_clause 0x1",
    "v_cvt_f32_i32_e32 v57, v57",
    "s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)",
    "s_endpgm",
];
pub const GFX1100_GOLDEN: &[&str] = &[
    "s_waitcnt vmcnt(3)",
    "s_waitcnt lgkmcnt(1)",
    "s_waitcnt_depctr depctr_va_sdst(0)",
    "s_barrier",
    "buffer_gl0_inv",
    "v_wmma_i32_16x16x16_iu4 v[0:7], v[8:9], v[10:11], 0 neg_lo:[1,1,0]",
    "v_dual_mul_f32 v83, v97, v81 :: v_dual_mul_f32 v84, v98, v82",
    "v_dual_fmac_f32 v163, v83, v57 :: v_dual_fmac_f32 v166, v84, v58",
    "buffer_load_b64 v[0:1], v2, s[4:7], s8 offen",
    "ds_load_2addr_stride64_b64 v[71:74], v81 offset1:1",
    "s_clause 0x1",
    "s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)",
    "s_endpgm",
];
