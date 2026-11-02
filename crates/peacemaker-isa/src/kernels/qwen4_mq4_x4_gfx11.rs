// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1151 MQ4G256V2 trunk GEMM: the `Fmt::Mq4` instance of the MQ6 PM builder
//! (`qwen4_mq6_x4_gfx11`), exact twins of the current hipcc MQ4 route
//! (`gemm_mq4g256v2_residual_wmma_gfx11_bt4`): the same A operands (f16 FMA
//! dequant, single rounding), the same B operands (F16 rotated X) and the same
//! WMMA chain. Entries: `qwen4_mq4_x4_pm_gfx1151_{w8, w4, w8_bf16out,
//! w4_bf16out, w8_regions, w8_hcw}` in the module `qwen4_mq4_x4_pm_gfx1151`,
//! with the kernarg ABIs of the MQ6 entries. Only the weight format differs
//! from the MQ6 builder (136-byte groups, one b32 of eight nibbles per lane per
//! 16-K step, 4-bit `v_bfe_u32`); see its module documentation.
use crate::kernels::qwen4_mq6_x4_gfx11::{self as base, Cfg, Fmt};
pub use crate::kernels::qwen4_mq6_x4_gfx11::{Kind, GROUP_BYTES, SLOT_BYTES, TOK};
use crate::{Arch, Emitted};

type R = Result<(), String>;

/// `wr` waves (16 rows each) per workgroup, of one MQ4 entry kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
    pub wr: u8,
    pub kind: Kind,
}
impl Spec {
    /// The six PM entries.
    pub const ALL: [(u8, Kind); 6] = Cfg::ALL;
    pub fn cfg(self) -> Cfg {
        Cfg {
            arch: self.arch,
            wr: self.wr,
            kind: self.kind,
            fmt: Fmt::Mq4,
        }
    }
    pub fn symbol(&self) -> String {
        self.cfg().symbol()
    }
    pub fn module(arch: Arch) -> String {
        Cfg::module(Fmt::Mq4, arch)
    }
    pub fn validate(&self) -> R {
        self.cfg().validate()
    }
    /// Rows per workgroup.
    pub fn rows(&self) -> u32 {
        self.cfg().rows()
    }
}

/// One MQ4 entry.
pub fn emit(spec: Spec) -> Result<Emitted, String> {
    base::emit_cfg(spec.cfg())
}

/// Every MQ4 entry as one code object (`Spec::module`).
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    base::emit_module_fmt(Fmt::Mq4, arch)
}
