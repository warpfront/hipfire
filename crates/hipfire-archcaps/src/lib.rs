// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `GfxArch`: the one GPU architecture identity. Identity only: no
//! capabilities and no family groupings (those belong with the facts table).
//!
//! Names are exact. Near neighbours (`gfx1200` vs `gfx1201`) are never
//! aliases, and target suffixes such as `gfx1201:xnack-` do not parse.

/// Exact GPU architecture (one LLVM processor name each).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GfxArch {
    Gfx906,
    Gfx908,
    Gfx940,
    Gfx941,
    Gfx942,
    Gfx950,
    Gfx1010,
    Gfx1011,
    Gfx1012,
    Gfx1030,
    Gfx1031,
    Gfx1032,
    Gfx1033,
    Gfx1034,
    Gfx1035,
    Gfx1036,
    Gfx1100,
    Gfx1101,
    Gfx1102,
    Gfx1103,
    Gfx1150,
    Gfx1151,
    Gfx1152,
    Gfx1200,
    Gfx1201,
}

impl GfxArch {
    /// Every variant, in declaration order.
    pub const ALL: [GfxArch; 25] = [
        GfxArch::Gfx906,
        GfxArch::Gfx908,
        GfxArch::Gfx940,
        GfxArch::Gfx941,
        GfxArch::Gfx942,
        GfxArch::Gfx950,
        GfxArch::Gfx1010,
        GfxArch::Gfx1011,
        GfxArch::Gfx1012,
        GfxArch::Gfx1030,
        GfxArch::Gfx1031,
        GfxArch::Gfx1032,
        GfxArch::Gfx1033,
        GfxArch::Gfx1034,
        GfxArch::Gfx1035,
        GfxArch::Gfx1036,
        GfxArch::Gfx1100,
        GfxArch::Gfx1101,
        GfxArch::Gfx1102,
        GfxArch::Gfx1103,
        GfxArch::Gfx1150,
        GfxArch::Gfx1151,
        GfxArch::Gfx1152,
        GfxArch::Gfx1200,
        GfxArch::Gfx1201,
    ];

    /// The LLVM processor name, e.g. `"gfx1201"`.
    pub const fn name(self) -> &'static str {
        match self {
            GfxArch::Gfx906 => "gfx906",
            GfxArch::Gfx908 => "gfx908",
            GfxArch::Gfx940 => "gfx940",
            GfxArch::Gfx941 => "gfx941",
            GfxArch::Gfx942 => "gfx942",
            GfxArch::Gfx950 => "gfx950",
            GfxArch::Gfx1010 => "gfx1010",
            GfxArch::Gfx1011 => "gfx1011",
            GfxArch::Gfx1012 => "gfx1012",
            GfxArch::Gfx1030 => "gfx1030",
            GfxArch::Gfx1031 => "gfx1031",
            GfxArch::Gfx1032 => "gfx1032",
            GfxArch::Gfx1033 => "gfx1033",
            GfxArch::Gfx1034 => "gfx1034",
            GfxArch::Gfx1035 => "gfx1035",
            GfxArch::Gfx1036 => "gfx1036",
            GfxArch::Gfx1100 => "gfx1100",
            GfxArch::Gfx1101 => "gfx1101",
            GfxArch::Gfx1102 => "gfx1102",
            GfxArch::Gfx1103 => "gfx1103",
            GfxArch::Gfx1150 => "gfx1150",
            GfxArch::Gfx1151 => "gfx1151",
            GfxArch::Gfx1152 => "gfx1152",
            GfxArch::Gfx1200 => "gfx1200",
            GfxArch::Gfx1201 => "gfx1201",
        }
    }

    /// Exact-name parse; anything else (including `gfx1201:xnack-`) is `None`.
    pub fn parse(name: &str) -> Option<Self> {
        GfxArch::ALL.into_iter().find(|a| a.name() == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_round_trips() {
        for a in GfxArch::ALL {
            assert_eq!(GfxArch::parse(a.name()), Some(a));
        }
    }

    #[test]
    fn names_are_unique() {
        for (i, a) in GfxArch::ALL.iter().enumerate() {
            for b in &GfxArch::ALL[i + 1..] {
                assert_ne!(a.name(), b.name());
            }
        }
    }

    #[test]
    fn rejects_suffixed_and_unknown() {
        for n in [
            "gfx1201:xnack-",
            "gfx1201:sramecc+",
            "gfx9999",
            "GFX1201",
            "",
            " gfx1100",
            "gfx11",
        ] {
            assert_eq!(GfxArch::parse(n), None, "{n:?}");
        }
    }

    /// Literal name lists copied from the existing sources; each must parse.
    #[test]
    fn existing_sources_all_parse() {
        // crates/rdna-compute/src/arch_caps.rs:24-43 (the 20 atoms)
        let atoms = [
            "gfx906", "gfx908", "gfx1010", "gfx1011", "gfx1012", "gfx1030", "gfx1031", "gfx1032",
            "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200",
            "gfx1201", "gfx940", "gfx941", "gfx942",
        ];
        // crates/hipfire-kernel-abi/src/lib.rs:58
        let kernel_abi = ["gfx1201", "gfx1200", "gfx1100", "gfx1151", "gfx942"];
        // crates/peacemaker-isa/src/arch.rs:9
        let isa = ["gfx1100", "gfx1151", "gfx1201"];
        // crates/peacemaker-ir/src/inst.rs:21
        let ir = [
            "gfx1010", "gfx1030", "gfx1100", "gfx1151", "gfx1200", "gfx1201",
        ];
        // crates/peacemaker-cdna/src/arch.rs:18
        let cdna = ["gfx942", "gfx950"];
        for list in [&atoms[..], &kernel_abi, &isa, &ir, &cdna] {
            for n in list {
                assert!(GfxArch::parse(n).is_some(), "{n} has no GfxArch");
            }
        }
        assert_eq!(atoms.len(), 20);
        // crates/rdna-compute/src/arch_caps.rs:131-136 (RDNA2 APUs ArchCaps admits, HIPF-354)
        for n in ["gfx1033", "gfx1034", "gfx1035", "gfx1036"] {
            assert!(GfxArch::parse(n).is_some(), "{n} has no GfxArch");
        }
    }
}
