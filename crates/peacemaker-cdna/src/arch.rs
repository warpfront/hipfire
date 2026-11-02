// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CDNA variants and their capability table.
//!
//! Every encoder, decoder and emulator path that differs between CDNA
//! generations takes an [`Arch`] and reads its [`Caps`] row. Only gfx942 is
//! populated. A later variant (gfx950 / CDNA4: OCP FP8, FP4/FP6 MFMA, larger
//! LDS) adds an enum variant, a `Caps` row, and per-variant opcode arms in the
//! `enc` tables (`*Op::opcode(arch)` returns `None` where a variant lacks an
//! instruction); nothing else is forked.
use crate::enc::vop3p::MaiOp;
use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Arch {
    /// CDNA3, AMD Instinct MI300X/MI300A/MI325X.
    Gfx942,
}

/// How FP8/BF8 bytes are interpreted by the variant's conversion and MFMA
/// instructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fp8Encoding {
    /// E4M3 bias 8 / E5M2 bias 16, `0x80` = NaN, no -0, no E4M3 infinity
    /// (CDNA3 ISA §7.2 Table 30 p.50).
    Fnuz,
    /// OCP 8-bit floating point: E4M3FN bias 7 / E5M2 bias 15 (RDNA4, CDNA4).
    Ocp,
}

/// One variant's capabilities. Counts are per wave unless stated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Caps {
    pub arch: Arch,
    /// `.amdgcn_target` processor name.
    pub name: &'static str,
    pub generation: &'static str,
    /// `EF_AMDGPU_MACH_*` value of the ELF header (LLVM `AMDGPUUsage`).
    pub elf_mach: u32,
    pub fp8: Fp8Encoding,
    /// LDS bytes one workgroup may allocate.
    pub lds_bytes: u32,
    /// Addressable SGPRs (s0..s101; VCC above them).
    pub sgprs: u16,
    /// Architectural VGPRs (v0..v255) and AccVGPRs (a0..a255).
    pub vgprs: u16,
    pub agprs: u16,
    /// VGPR+AGPR file shared by one wave, split by `accum_offset`
    /// (CDNA3 ISA §7 p.40; unified since gfx90a).
    pub unified_vgprs: u16,
    /// VGPR allocation granule (registers) of `GRANULATED_WORKITEM_VGPR_COUNT`.
    pub vgpr_granule: u16,
    /// Work-item IDs X/Y/Z packed into v0 bits [9:0]/[19:10]/[29:20]
    /// (gfx90a+; LLVM `AMDGPUUsage` "Initial Kernel Execution State").
    pub packed_tid: bool,
    /// `s_waitcnt` field maxima (CDNA3 ISA §4.4 p.19, SOPP S_WAITCNT p.?):
    /// VM_CNT 6 bits, LGKM_CNT 4 bits, EXP_CNT 3 bits; no VS_CNT.
    pub vmcnt_max: u8,
    pub lgkmcnt_max: u8,
    pub expcnt_max: u8,
    /// `s_barrier` does not wait for outstanding memory (back-off barrier,
    /// gfx90a+): waits before a barrier are the program's job.
    pub back_off_barrier: bool,
    /// DPP `row_bcast:15/31` and `wave_shl/shr/rol/ror` exist (gfx9 DPP).
    pub dpp_wave_ops: bool,
    /// `v_permlane16_b32` / `v_permlanex16_b32` (gfx10+ only).
    pub permlane16: bool,
    /// `v_permlane16_swap` / `v_permlane32_swap` (gfx950).
    pub permlane_swap: bool,
    /// MFMA/accvgpr opcodes the variant implements.
    pub mai: &'static [MaiOp],
    /// FP4/FP6 MFMA operand types (gfx950).
    pub fp4_fp6_mfma: bool,
}

const GFX942_MAI: &[MaiOp] = MaiOp::ALL;

const GFX942: Caps = Caps {
    arch: Arch::Gfx942,
    name: "gfx942",
    generation: "CDNA3",
    elf_mach: 0x04c,
    fp8: Fp8Encoding::Fnuz,
    lds_bytes: 64 * 1024,
    sgprs: 102,
    vgprs: 256,
    agprs: 256,
    unified_vgprs: 512,
    vgpr_granule: 8,
    packed_tid: true,
    vmcnt_max: 63,
    lgkmcnt_max: 15,
    expcnt_max: 7,
    back_off_barrier: true,
    dpp_wave_ops: true,
    permlane16: false,
    permlane_swap: false,
    mai: GFX942_MAI,
    fp4_fp6_mfma: false,
};

impl Arch {
    pub const ALL: &'static [Arch] = &[Arch::Gfx942];
    pub fn caps(self) -> &'static Caps {
        match self {
            Arch::Gfx942 => &GFX942,
        }
    }
    pub fn name(self) -> &'static str {
        self.caps().name
    }
    pub fn has_mai(self, op: MaiOp) -> bool {
        self.caps().mai.contains(&op)
    }
}

impl std::str::FromStr for Arch {
    type Err = String;
    fn from_str(s: &str) -> Result<Self> {
        Arch::ALL
            .iter()
            .copied()
            .find(|a| a.name() == s)
            .ok_or_else(|| format!("unsupported CDNA architecture {s}"))
    }
}

impl std::fmt::Display for Arch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
