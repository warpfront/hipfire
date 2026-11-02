// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Arch {
    Gfx1201,
    Gfx1100,
    Gfx1151,
    Gfx942,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WgIdSource {
    Ttmp7,
    Ttmp9,
    Sgpr(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WgIds {
    pub x: WgIdSource,
    pub y: WgIdSource,
}

impl Arch {
    pub fn name(self) -> &'static str {
        match self {
            Self::Gfx1201 => "gfx1201",
            Self::Gfx1100 => "gfx1100",
            Self::Gfx1151 => "gfx1151",
            Self::Gfx942 => "gfx942",
        }
    }
    /// The CDNA instruction-model variant (wave64, `peacemaker_cdna`), `None` on RDNA.
    pub fn cdna(self) -> Option<peacemaker_cdna::Arch> {
        match self {
            Self::Gfx942 => Some(peacemaker_cdna::Arch::Gfx942),
            _ => None,
        }
    }
    pub fn gfx12(self) -> bool {
        matches!(self, Self::Gfx1201)
    }
    /// Whether this architecture uses the CDNA wave64 instruction model.
    pub fn wave64(self) -> bool {
        self.cdna().is_some()
    }
    pub fn buffer_offset_max(self) -> u32 {
        if self.gfx12() {
            (1 << 23) - 1
        } else {
            4095
        }
    }
    /// RDNA-only WMMA mnemonic; CDNA callers must use `peacemaker_isa::cdna`.
    pub fn wmma_iu4(self) -> &'static str {
        if self.gfx12() {
            "v_wmma_i32_16x16x32_iu4"
        } else {
            "v_wmma_i32_16x16x16_iu4"
        }
    }
    /// RDNA-only dependency wait; CDNA callers must use `peacemaker_isa::cdna`.
    pub fn dependency_wait(self) -> &'static str {
        if self.gfx12() {
            "s_wait_alu"
        } else {
            "s_waitcnt_depctr"
        }
    }
    pub fn barrier(self) -> &'static [&'static str] {
        if self.gfx12() {
            &["s_barrier_signal -1", "s_barrier_wait 0xffff"]
        } else {
            &["s_barrier"]
        }
    }
    /// Minimum number of initialised user + system (workgroup-id/info) SGPRs a wave32 compute kernel
    /// must enable: LLVM `FeatureUserSGPRInit16Bug` ("Bug requiring at least 16 user+system SGPRs to
    /// be enabled"), set for gfx1100 (`FeatureISAVersion11_0_0`), gfx1102 and generic gfx11, not for
    /// gfx1101/gfx1103/gfx115x/gfx12. `SITargetLowering::allocateSystemSGPRs` satisfies it by padding
    /// the user SGPRs (`addReservedUserSGPR`) until `user + system >= 16`; the private-segment wave
    /// offset is not counted. Only gfx1100 is a builder target of those processors.
    pub const fn min_user_system_sgprs(self) -> u8 {
        match self {
            Self::Gfx1100 => 16,
            _ => 0,
        }
    }
    /// The `.amdhsa_user_sgpr_count` to declare: `user` real user SGPRs (kernarg pointer, ...)
    /// padded with dead user SGPRs, as LLVM does, so that `user + system >= 16` where `system` is
    /// the number of enabled workgroup-id/info SGPRs. The system SGPRs follow the declared count,
    /// so the padding moves them: `workgroup_ids(declared)` is where the hardware now puts them.
    pub const fn declared_user_sgprs(self, user: u8, system: u8) -> u8 {
        let pad = self.min_user_system_sgprs().saturating_sub(system);
        if user > pad {
            user
        } else {
            pad
        }
    }
    /// The `FeatureUserSGPRInit16Bug` rule on a kernel descriptor: `COMPUTE_PGM_RSRC2.USER_SGPR_COUNT`
    /// (bits 1..5) plus the enabled `ENABLE_SGPR_WORKGROUP_ID_X/Y/Z/INFO` (bits 7..10) must reach
    /// [`Arch::min_user_system_sgprs`] (LLVM asserts `getNumPreloadedSGPRs() >= 16`; the private-segment
    /// wave offset is not counted). Wave32 compute kernels only: every builder object is wave32.
    pub fn check_user_system_sgprs(self, symbol: &str, rsrc2: u32) -> Result<(), String> {
        let user = (rsrc2 >> 1) & 0x1f;
        let system = (rsrc2 >> 7 & 1) + (rsrc2 >> 8 & 1) + (rsrc2 >> 9 & 1) + (rsrc2 >> 10 & 1);
        let min = u32::from(self.min_user_system_sgprs());
        if user + system < min {
            return Err(format!(
                "{symbol}: {} needs at least {min} user+system SGPRs (LLVM FeatureUserSGPRInit16Bug), \
                 descriptor has {user} user + {system} system = {} (compute_pgm_rsrc2 {rsrc2:#x})",
                self.name(),
                user + system
            ));
        }
        Ok(())
    }
    /// Workgroup ids at `user_sgpr_count` DECLARED user SGPRs (see [`Arch::declared_user_sgprs`]; use
    /// `KernelSpec::workgroup_ids` for a kernel).
    pub fn workgroup_ids(self, user_sgpr_count: u8) -> Result<WgIds, String> {
        if self.gfx12() {
            Ok(WgIds {
                x: WgIdSource::Ttmp9,
                y: WgIdSource::Ttmp7,
            })
        } else if user_sgpr_count <= 102 {
            Ok(WgIds {
                x: WgIdSource::Sgpr(user_sgpr_count),
                y: WgIdSource::Sgpr(user_sgpr_count + 1),
            })
        } else {
            Err("workgroup ID SGPRs exceed architectural range".into())
        }
    }
    pub fn check_mnemonic(self, mnemonic: &str) -> Result<(), String> {
        if self.wave64() {
            return Err(format!("{mnemonic} cannot be emitted by the RDNA text builder on {}; use peacemaker_isa::cdna", self.name()));
        }
        if (self.gfx12() && (mnemonic == "s_waitcnt" || mnemonic == "s_waitcnt_vscnt"))
            || (!self.gfx12()
                && (mnemonic.starts_with("s_wait_loadcnt")
                    || mnemonic.starts_with("s_wait_storecnt")
                    || mnemonic.starts_with("s_wait_dscnt")
                    || mnemonic.starts_with("s_wait_kmcnt")
                    || mnemonic == "s_wait_alu"
                    || mnemonic.starts_with("v_swmmac")
                    || mnemonic == "v_wmma_f32_16x16x16_fp8_fp8"
                    || mnemonic == "v_wmma_i32_16x16x32_iu4"
                    || mnemonic == "global_inv"
                    || mnemonic == "s_barrier_signal"
                    || mnemonic == "s_barrier_wait"
                    || (self != Self::Gfx1151
                        && (mnemonic == "s_add_f32" || mnemonic == "s_mul_f32"))
                    || mnemonic == "v_s_rcp_f32"))
            || (self.gfx12()
                && (mnemonic == "s_barrier"
                    || mnemonic.starts_with("buffer_gl")
                    || mnemonic == "s_waitcnt_depctr"))
        {
            return Err(format!("{mnemonic} is unsupported on {}", self.name()));
        }
        Ok(())
    }
}
impl std::str::FromStr for Arch {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gfx1201" => Ok(Self::Gfx1201),
            "gfx1100" => Ok(Self::Gfx1100),
            "gfx1151" => Ok(Self::Gfx1151),
            "gfx942" => Ok(Self::Gfx942),
            _ => Err(format!("unsupported architecture {s}")),
        }
    }
}

#[cfg(test)]
mod arch_tests {
    use super::*;

    #[test]
    fn gfx942_helpers() {
        let arch = Arch::Gfx942;
        assert!(arch.wave64());
        assert_eq!(arch.buffer_offset_max(), 4095);
        assert_eq!(arch.barrier(), &["s_barrier"]);
        assert_eq!(
            arch.workgroup_ids(8).unwrap(),
            WgIds {
                x: WgIdSource::Sgpr(8),
                y: WgIdSource::Sgpr(9)
            }
        );
        assert!(arch.workgroup_ids(103).is_err());
        for mnemonic in [
            "",
            "s_nop",
            "s_barrier",
            "s_waitcnt",
            "v_add_f32",
            "unknown",
        ] {
            assert!(arch
                .check_mnemonic(mnemonic)
                .unwrap_err()
                .contains("peacemaker_isa::cdna"));
        }
    }

    #[test]
    fn rdna_remains_wave32() {
        for arch in [Arch::Gfx1201, Arch::Gfx1100, Arch::Gfx1151] {
            assert!(!arch.wave64());
            assert!(arch.check_mnemonic("s_nop").is_ok());
        }
    }

    #[test]
    fn gfx1151_scalar_float_add_mul() {
        // RDNA3.5's scalar FP fold/scale instructions are pinned in gfx1151.tbl.
        for mnemonic in ["s_add_f32", "s_mul_f32"] {
            assert!(Arch::Gfx1151.check_mnemonic(mnemonic).is_ok());
            assert!(Arch::Gfx1201.check_mnemonic(mnemonic).is_ok());
            assert!(Arch::Gfx1100.check_mnemonic(mnemonic).is_err());
            assert!(Arch::Gfx942.check_mnemonic(mnemonic).is_err());
        }
    }
}
