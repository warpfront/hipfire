// SPDX-License-Identifier: Apache-2.0
//! Op-level stage map of a co-processor (NPU) stage beside the GPU.
//!
//! [`crate::mesh_placement::BandPlacement`] places whole layer bands on interchangeable
//! devices. A co-processor stage is not that: it owns *operations inside* every layer.
//! A [`StageMap`] declares, for a hybrid recurrent + attention layer stack, which
//! projection families the co-processor may own; everything else of a layer (mixer core,
//! recurrent/KV state, residual, FFN, head) stays on the GPU stage. Which of the eligible
//! families the co-processor actually owns is a [`FamilySet`], the EP-like ownership
//! table the calibrated planner (`coproc::plan`) chooses per load or per request.
//!
//! A map is a pure architecture shape description: no device handle, calibration or
//! schedule lives here. Topology lives in [`crate::device_mesh`], ownership in
//! [`FamilySet`], scheduling in `coproc::schedule`, costs in `coproc::calibration`.

use std::fmt;

/// Layer kinds of the hybrid stack. Kind 0 is the recurrent mixer (GDN), kind 1 the attention mixer.
pub const KINDS: usize = 2;
/// Ownership groups: the smallest family sets the planner moves as a unit.
pub const GROUPS: usize = 4;

/// Device kind that executes a stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StageDevice {
    Gpu,
    Npu,
}

/// Output numerics of a co-processor stage. The GPU stage is always the native kernels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum EpilogueContract {
    /// W8A8: exact integer accumulation across the whole reduction, grid15-up weight and activation scale
    /// sidecars, one IEEE BF16 rounding (`peacemaker_npu::kernels::w8a8_epi::bf16_direct`). No int32 partial,
    /// combine or fold exists anywhere, on either device.
    #[default]
    Bf16Direct,
}

impl EpilogueContract {
    /// Spelling used in load logs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16Direct => "bf16_direct",
        }
    }

    /// Spelling stored in calibration records (`npu.epilogue`, `key.epilogue`).
    pub const fn calibration_name(self) -> &'static str {
        match self {
            Self::Bf16Direct => "bf16",
        }
    }
}

impl fmt::Display for EpilogueContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where an operation of a layer belongs in the op-level map.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpSlot {
    /// One input projection family of a mixer (`StageMap::families` index).
    InProj(usize),
    /// The out projection of a mixer of this kind.
    OutProj(usize),
    /// Mixer core (recurrence, attention), residual, norms, FFN and head: always GPU.
    Gpu,
}

/// One layer kind of the stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerKindDecl {
    /// Calibration spelling of the kind (`families.<f>.layer`).
    pub name: &'static str,
    /// Layers of this kind in the stack.
    pub layers: usize,
    /// Reduction dimension of the kind's input projections (the model width).
    pub in_k: usize,
    /// Calibration name of the kind's out-projection family.
    pub out_family: &'static str,
}

/// One input-projection family: a weight matrix whose output columns the co-processor can produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FamilyDecl {
    pub name: &'static str,
    /// Index into [`StageMap::kinds`].
    pub kind: usize,
    /// Output features (matrix rows).
    pub cols: usize,
}

/// An ownership group: families that move together (one calibrated GPU cost).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupDecl {
    /// Calibration name (`gpu.trunk_in_ms_per_8192.<name>`, `families.<name>`).
    pub name: &'static str,
    /// Index into [`StageMap::kinds`].
    pub kind: usize,
    /// Member families, as a [`FamilySet`] bit mask.
    pub families: u32,
}

/// Op-level map of the co-processor stage: which families it may own and the numerics it owes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageMap {
    /// Log tag of the route (`<tag>: mode=...`).
    pub tag: &'static str,
    pub kinds: [LayerKindDecl; KINDS],
    /// Input-projection families, grouped by kind in fused order (one fused NPU shape per kind).
    pub families: &'static [FamilyDecl],
    pub groups: [GroupDecl; GROUPS],
    /// Numerics of every co-processor-owned output.
    pub contract: EpilogueContract,
}

impl StageMap {
    /// Structural invariants: kinds name distinct layers, every family belongs to exactly one group of its kind,
    /// and each group's families are adjacent in fused order (so an owned group is one contiguous column range).
    pub fn validate(&self) -> Result<(), String> {
        if self.families.is_empty() || self.families.len() > 32 {
            return Err(format!("{}: 1..=32 families required", self.tag));
        }
        if self.kinds[0].name == self.kinds[1].name || self.kinds.iter().any(|k| k.in_k == 0) {
            return Err(format!("{}: invalid layer kinds", self.tag));
        }
        for (i, f) in self.families.iter().enumerate() {
            if f.kind >= KINDS || f.cols == 0 {
                return Err(format!("{}: invalid family {}", self.tag, f.name));
            }
            if i > 0 && self.families[i - 1].kind > f.kind {
                return Err(format!("{}: families must be grouped by kind", self.tag));
            }
        }
        let mut covered = 0u32;
        for g in &self.groups {
            let members: Vec<usize> = (0..self.families.len())
                .filter(|i| g.families & (1 << i) != 0)
                .collect();
            let contiguous = members.windows(2).all(|w| w[1] == w[0] + 1);
            if g.kind >= KINDS
                || members.is_empty()
                || g.families >> self.families.len() != 0
                || covered & g.families != 0
                || !contiguous
                || members.iter().any(|&i| self.families[i].kind != g.kind)
            {
                return Err(format!("{}: invalid group {}", self.tag, g.name));
            }
            covered |= g.families;
        }
        if covered != self.all().0 {
            return Err(format!("{}: every family needs a group", self.tag));
        }
        Ok(())
    }

    /// Every family.
    pub fn all(&self) -> FamilySet {
        FamilySet(u32::MAX >> (32 - self.families.len()))
    }

    /// Union of the groups whose bit is set in `bits` (bit `i` = `groups[i]`).
    pub fn subset(&self, bits: u8) -> FamilySet {
        FamilySet(
            self.groups
                .iter()
                .enumerate()
                .filter(|(i, _)| bits & (1 << i) != 0)
                .fold(0, |mask, (_, g)| mask | g.families),
        )
    }

    /// Groups wholly owned by `set`.
    pub fn owned_groups(&self, set: FamilySet) -> [bool; GROUPS] {
        self.groups.map(|g| set.0 & g.families == g.families)
    }

    /// Whether the planner can move `set`: nonempty and no group partially owned.
    pub fn supports(&self, set: FamilySet) -> bool {
        set.0 != 0
            && self
                .groups
                .iter()
                .all(|g| set.0 & g.families == 0 || set.0 & g.families == g.families)
    }

    /// Group names sorted, the selection tie-break order.
    pub fn sorted_group_names(&self) -> [&'static str; GROUPS] {
        let mut names = self.groups.map(|g| g.name);
        names.sort_unstable();
        names
    }

    /// Device that runs `slot` under ownership `set` with out families `out` moved.
    pub fn owner(&self, set: FamilySet, out: [bool; KINDS], slot: OpSlot) -> StageDevice {
        let npu = match slot {
            OpSlot::InProj(family) => set.contains(family),
            OpSlot::OutProj(kind) => out[kind],
            OpSlot::Gpu => false,
        };
        if npu {
            StageDevice::Npu
        } else {
            StageDevice::Gpu
        }
    }
}

/// Families owned by the co-processor stage (bit `i` = `StageMap::families[i]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FamilySet(pub u32);

impl FamilySet {
    pub const NONE: FamilySet = FamilySet(0);

    pub const fn contains(self, family: usize) -> bool {
        family < 32 && self.0 & (1 << family) != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn is_subset_of(self, other: FamilySet) -> bool {
        self.0 & !other.0 == 0
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    const FAMILIES: [FamilyDecl; 7] = [
        FamilyDecl {
            name: "qkv",
            kind: 0,
            cols: 10240,
        },
        FamilyDecl {
            name: "z",
            kind: 0,
            cols: 6144,
        },
        FamilyDecl {
            name: "a",
            kind: 0,
            cols: 48,
        },
        FamilyDecl {
            name: "b",
            kind: 0,
            cols: 48,
        },
        FamilyDecl {
            name: "q",
            kind: 1,
            cols: 12288,
        },
        FamilyDecl {
            name: "k",
            kind: 1,
            cols: 1024,
        },
        FamilyDecl {
            name: "v",
            kind: 1,
            cols: 1024,
        },
    ];

    /// A dense-27B-shaped map (48 GDN + 16 FA layers, K = 5120, FA k/v 1024 wide, no index head).
    pub(crate) const DENSE: StageMap = StageMap {
        tag: "dense",
        kinds: [
            LayerKindDecl {
                name: "gdn",
                layers: 48,
                in_k: 5120,
                out_family: "gdn.out",
            },
            LayerKindDecl {
                name: "fa",
                layers: 16,
                in_k: 5120,
                out_family: "fa.o",
            },
        ],
        families: &FAMILIES,
        groups: [
            GroupDecl {
                name: "qkv",
                kind: 0,
                families: 0x01,
            },
            GroupDecl {
                name: "zab",
                kind: 0,
                families: 0x0e,
            },
            GroupDecl {
                name: "q",
                kind: 1,
                families: 0x10,
            },
            GroupDecl {
                name: "kv",
                kind: 1,
                families: 0x60,
            },
        ],
        contract: EpilogueContract::Bf16Direct,
    };

    #[test]
    fn groups_partition_families_into_contiguous_ranges() {
        DENSE.validate().unwrap();
        let mut gap = DENSE;
        gap.groups[1].families = 0x0a; // z, b without a: not adjacent in fused order
        gap.groups[0].families = 0x05;
        assert!(gap.validate().is_err());
        let mut cross = DENSE;
        cross.groups[2].families = 0x18; // b (gdn) with q (fa)
        cross.groups[1].families = 0x06;
        assert!(cross.validate().is_err());
        let mut orphan = DENSE;
        orphan.groups[3].families = 0x20; // v in no group
        assert!(orphan.validate().is_err());
    }

    #[test]
    fn ownership_is_whole_groups_and_gpu_keeps_the_rest() {
        let qkv_zab = DENSE.subset(0b0011);
        assert_eq!(qkv_zab, FamilySet(0x0f));
        assert!(DENSE.supports(qkv_zab));
        assert_eq!(DENSE.owned_groups(qkv_zab), [true, true, false, false]);
        assert!(
            !DENSE.supports(FamilySet(0x02)),
            "z without a, b splits a group"
        );
        assert!(!DENSE.supports(FamilySet::NONE));
        assert_eq!(
            DENSE.owner(qkv_zab, [false; 2], OpSlot::InProj(0)),
            StageDevice::Npu
        );
        assert_eq!(
            DENSE.owner(qkv_zab, [false; 2], OpSlot::InProj(4)),
            StageDevice::Gpu
        );
        assert_eq!(
            DENSE.owner(DENSE.all(), [true; 2], OpSlot::Gpu),
            StageDevice::Gpu
        );
        assert_eq!(
            DENSE.owner(FamilySet::NONE, [false, true], OpSlot::OutProj(1)),
            StageDevice::Npu
        );
        assert_eq!(DENSE.sorted_group_names(), ["kv", "q", "qkv", "zab"]);
    }
}
