// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Scratch (private-segment) ring planning for GFX10/GFX11 PM4 dispatch.
//!
//! A PM4 indirect buffer submitted through AMD's vendor AQL packet bypasses
//! the command processor's AQL dispatch path, which is the part that reads
//! `amd_queue_t` and programs the scratch registers for ordinary kernel
//! dispatches. A PM4 stream therefore owns the whole scratch contract:
//!
//! 1. a backing allocation of `slots * bytes_per_wave` bytes;
//! 2. `COMPUTE_TMPRING_SIZE` = `WAVES | WAVESIZE << 12`;
//! 3. gfx11: the ring base in `COMPUTE_DISPATCH_SCRATCH_BASE_LO/HI`
//!    (`>> 8` / `>> 40`); gfx10: a buffer V# in the private-segment user SGPRs
//!    plus the absolute ring base in the `FLAT_SCRATCH_INIT` user SGPRs;
//! 4. `COMPUTE_PGM_RSRC2.SCRATCH_EN` (already in the kernel descriptor), which
//!    makes the SPI hand each wave its slot offset.
//!
//! Register offsets (SET_SH_REG dword offset from 0x2c00 = byte offset 0xB000
//! `>> 2`): `COMPUTE_DISPATCH_SCRATCH_BASE_LO` 0xB840 -> 0x210,
//! `COMPUTE_DISPATCH_SCRATCH_BASE_HI` 0xB844 -> 0x211, `COMPUTE_TMPRING_SIZE`
//! 0xB860 -> 0x218 (Mesa `radv_queue.c: radv_emit_compute_scratch`:
//! `R_00B840_COMPUTE_DISPATCH_SCRATCH_BASE_LO`, `R_00B860_COMPUTE_TMPRING_SIZE`).
//! `COMPUTE_TMPRING_SIZE` is `WAVES` [11:0] and `WAVESIZE` [24:12] on gfx10.x
//! / [26:12] on gfx11 (Linux `gc_10_3_0_sh_mask.h` mask `0x01FFF000`,
//! `gc_11_0_0_sh_mask.h` mask `0x07FFF000`; Mesa `registers/gfx10.json`,
//! `gfx11.json`).

use std::fmt;

/// `COMPUTE_TMPRING_SIZE.WAVESIZE` starts at bit 12 on every GFX10/GFX11 part.
pub(crate) const TMPRING_WAVESIZE_SHIFT: u32 = 12;
/// GFX10 `WAVESIZE` unit: 1024 bytes per wave (Mesa `ac_gpu_info.c`:
/// `scratch_wavesize_granularity_shift = gfx_level >= GFX11 ? 8 : 10`).
const GFX10_WAVESIZE_GRANULE: u32 = 1 << 10;
/// GFX11 `WAVESIZE` unit: 256 bytes per wave.
const GFX11_WAVESIZE_GRANULE: u32 = 1 << 8;
/// Fewest wave slots per shader engine a ring may be capped to: a 1024-thread
/// workgroup is 32 wave32 waves and all of them need slots to make progress.
pub const MIN_SCRATCH_SLOTS_PER_ENGINE: u32 = 32;

/// `SQ_BUF_RSRC_WORD1.SWIZZLE_ENABLE` (bit 31; Mesa `registers/gfx10-rsrc.json`).
const SQ_BUF_RSRC_WORD1_SWIZZLE_ENABLE: u32 = 1 << 31;
/// ROCr `FillBufRsrcWord3_Gfx10` with `INDEX_STRIDE` left for the wave size:
/// `DST_SEL_{X,Y,Z,W}` = `SQ_SEL_{X,Y,Z,W}` (4,5,6,7 at bits 2:0/5:3/8:6/11:9),
/// `FORMAT` = `BUF_FORMAT_32_UINT` (20, bits 18:12), `ADD_TID_ENABLE` (bit 23),
/// `RESOURCE_LEVEL` = 1 (bit 24, "must be 1"), `OOB_SELECT` = 2 "disabled"
/// (bits 29:28, Mesa `SQ_BUF_RSRC_WORD3__OOB_SELECT`), `TYPE` = 0 (buffer).
const SQ_BUF_RSRC_WORD3_GFX10_SCRATCH: u32 =
    4 | (5 << 3) | (6 << 6) | (7 << 9) | (20 << 12) | (1 << 23) | (1 << 24) | (2 << 28);
/// `SQ_BUF_RSRC_WORD3.INDEX_STRIDE` (bits 22:21): 2 = 32, 3 = 64 elements.
const SQ_BUF_INDEX_STRIDE_SHIFT: u32 = 21;
const SQ_BUF_INDEX_STRIDE_32: u32 = 2;
const SQ_BUF_INDEX_STRIDE_64: u32 = 3;

/// Chip family selecting how a PM4 dispatch reaches its scratch ring.
///
/// * `Gfx10` (gfx1010..gfx1036, LLVM "absolute flat scratch"): the scratch
///   ring is described to the shader by a buffer V# in the private-segment
///   user SGPRs and, for flat access, the absolute ring address in the
///   `FLAT_SCRATCH_INIT` user SGPRs. `COMPUTE_TMPRING_SIZE.WAVES` counts all
///   wave slots and `WAVESIZE` is in 1024-byte units.
/// * `Gfx11` (gfx1100..gfx1153, LLVM "architected flat scratch"): the ring
///   base is the `COMPUTE_DISPATCH_SCRATCH_BASE_LO/HI` register pair and the
///   hardware initializes `FLAT_SCRATCH` itself, so no scratch user SGPRs
///   exist. `WAVES` is per shader engine and `WAVESIZE` is in 256-byte units.
///
/// Primary sources: Mesa `radv_queue.c: radv_emit_compute_scratch` and
/// `ac_shader_util.c: ac_get_scratch_tmpring_size` (register choice, per-SE
/// `WAVES` on GFX11, 256/1024-byte granule), ROCr `amd_aql_queue.cpp:
/// FillComputeTmpRingSize{,_Gfx11}` (`WAVES`/`WAVESIZE` derivation) and
/// `FillBufRsrcWord{0,1,2}`/`FillBufRsrcWord3_Gfx10` (the gfx10 V#), LLVM
/// `AMDGPUUsage.md` "Absolute/Architected flat scratch" and "Private Segment
/// Buffer", and the Linux `gc_{10_3,11_0}_0_sh_mask.h` `COMPUTE_TMPRING_SIZE`
/// and `COMPUTE_DISPATCH_SCRATCH_BASE_{LO,HI}` field masks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gfx10ScratchFamily {
    Gfx10,
    Gfx11,
}

impl Gfx10ScratchFamily {
    /// `COMPUTE_TMPRING_SIZE.WAVESIZE` unit in bytes (per wave).
    const fn wavesize_granule(self) -> u32 {
        match self {
            Self::Gfx10 => GFX10_WAVESIZE_GRANULE,
            Self::Gfx11 => GFX11_WAVESIZE_GRANULE,
        }
    }

    /// Largest encodable `WAVESIZE` value: 13 bits on gfx10, 15 on gfx11.
    const fn max_wavesize_units(self) -> u32 {
        match self {
            Self::Gfx10 => 0x1fff,
            Self::Gfx11 => 0x7fff,
        }
    }
}

/// Shader topology the scratch ring is sized from. ROCr derives the same three
/// numbers for `compute_tmpring_size`: compute units
/// (`NumFComputeCores / NumSIMDPerCU`), `NumShaderBanks` shader engines and
/// `MaxSlotsScratchCU` wave slots per CU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gfx10ScratchTopology {
    pub compute_units: u32,
    pub shader_engines: u32,
    pub max_waves_per_cu: u32,
}

/// Why a scratch ring could not be planned or bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gfx10ScratchError {
    /// A topology count is zero.
    EmptyTopology(Gfx10ScratchTopology),
    /// `required_bytes_per_wave` is zero (nothing to provision).
    NothingRequired,
    /// The per-wave size does not fit `COMPUTE_TMPRING_SIZE.WAVESIZE`.
    WaveSizeTooLarge { required: u32, max: u32 },
    /// Even [`MIN_SCRATCH_SLOTS_PER_ENGINE`] wave slots per shader engine
    /// exceed the backing-size cap.
    BackingTooLarge { required: u64, cap: u64 },
    /// The ring base is null, not 4 KiB aligned, or above the 48-bit VA range.
    InvalidBase(u64),
}

impl fmt::Display for Gfx10ScratchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTopology(topology) => write!(
                formatter,
                "scratch topology has a zero count (compute_units={}, shader_engines={}, max_waves_per_cu={})",
                topology.compute_units, topology.shader_engines, topology.max_waves_per_cu
            ),
            Self::NothingRequired => write!(formatter, "no scratch bytes per wave requested"),
            Self::WaveSizeTooLarge { required, max } => write!(
                formatter,
                "scratch needs {required} bytes per wave but COMPUTE_TMPRING_SIZE.WAVESIZE encodes at most {max}"
            ),
            Self::BackingTooLarge { required, cap } => write!(
                formatter,
                "scratch ring needs at least {required} bytes of backing, above the {cap}-byte cap"
            ),
            Self::InvalidBase(address) => write!(
                formatter,
                "scratch ring base {address:#x} is null, not 4 KiB aligned, or beyond 48 bits"
            ),
        }
    }
}

impl std::error::Error for Gfx10ScratchError {}

/// A scratch ring before it has memory: the constant `COMPUTE_TMPRING_SIZE`
/// (`WAVES`, `WAVESIZE`) and the backing size that goes with it.
///
/// Both fields must stay constant for as long as any wave can use the ring
/// (Mesa `ac_get_scratch_tmpring_size`: "WAVESIZE must be constant while the
/// scratch buffer is being used by the GPU"), so a command buffer pins one plan
/// for its whole lifetime and a larger ring means a new ring, never a rewrite.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gfx10ScratchPlan {
    family: Gfx10ScratchFamily,
    bytes_per_wave: u32,
    waves: u32,
    slots: u32,
}

impl Gfx10ScratchPlan {
    /// Plan a ring giving every wave `required_bytes_per_wave` bytes
    /// (`private_segment_size` rounded to a dword, times the lanes per wave).
    ///
    /// The wave slot count is the device's full wave capacity
    /// (`compute_units * max_waves_per_cu`, ROCr's `max_scratch_waves`) so
    /// scratch kernels are never throttled below hardware occupancy; it is
    /// reduced only to honour `max_backing_bytes`, and never below
    /// [`MIN_SCRATCH_SLOTS_PER_ENGINE`] per shader engine so a full
    /// 1024-thread workgroup always fits.
    pub fn new(
        family: Gfx10ScratchFamily,
        topology: Gfx10ScratchTopology,
        required_bytes_per_wave: u32,
        max_backing_bytes: u64,
    ) -> Result<Self, Gfx10ScratchError> {
        if topology.compute_units == 0
            || topology.shader_engines == 0
            || topology.max_waves_per_cu == 0
        {
            return Err(Gfx10ScratchError::EmptyTopology(topology));
        }
        if required_bytes_per_wave == 0 {
            return Err(Gfx10ScratchError::NothingRequired);
        }
        let granule = family.wavesize_granule();
        // Mesa `ac_compute_scratch_wavesize`: make the unit count odd so
        // consecutive waves spread over memory channels ("exactly
        // `|= AMD_MEMCHANNEL_INTERLEAVE_BYTES`" on GFX11).
        let units = (u64::from(required_bytes_per_wave).div_ceil(u64::from(granule))) | 1;
        let max_units = u64::from(family.max_wavesize_units());
        if units > max_units {
            return Err(Gfx10ScratchError::WaveSizeTooLarge {
                required: required_bytes_per_wave,
                max: (max_units * u64::from(granule)) as u32,
            });
        }
        let bytes_per_wave = (units * u64::from(granule)) as u32;
        // The V# NUM_RECORDS dword (gfx10) and the 32-bit tooling around it
        // bound the ring below 4 GiB.
        let cap = max_backing_bytes.min(u64::from(u32::MAX));
        let engines = u64::from(topology.shader_engines);
        let target = u64::from(topology.compute_units) * u64::from(topology.max_waves_per_cu);
        // WAVES is 12 bits; gfx11 counts per shader engine, gfx10 in total
        // (and ROCr asserts it divisible by the engine count).
        let mut per_engine = target.div_ceil(engines).clamp(1, 0xfff);
        let by_cap = cap / (u64::from(bytes_per_wave) * engines);
        per_engine = per_engine.min(by_cap);
        let floor = u64::from(MIN_SCRATCH_SLOTS_PER_ENGINE);
        if per_engine < floor {
            return Err(Gfx10ScratchError::BackingTooLarge {
                required: floor * engines * u64::from(bytes_per_wave),
                cap,
            });
        }
        let slots = (per_engine * engines) as u32;
        let waves = match family {
            Gfx10ScratchFamily::Gfx11 => per_engine as u32,
            Gfx10ScratchFamily::Gfx10 => {
                slots.min(0xfff) / topology.shader_engines * topology.shader_engines
            }
        };
        let slots = match family {
            Gfx10ScratchFamily::Gfx11 => slots,
            Gfx10ScratchFamily::Gfx10 => waves,
        };
        Ok(Self {
            family,
            bytes_per_wave,
            waves,
            slots,
        })
    }

    pub fn family(&self) -> Gfx10ScratchFamily {
        self.family
    }

    /// Scratch bytes every wave owns (the constant `WAVESIZE`, in bytes).
    pub fn bytes_per_wave(&self) -> u32 {
        self.bytes_per_wave
    }

    /// Wave slots the ring holds in total (all shader engines).
    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// Bytes of device memory the ring needs behind its base address.
    pub fn backing_bytes(&self) -> u64 {
        u64::from(self.slots) * u64::from(self.bytes_per_wave)
    }

    /// The `COMPUTE_TMPRING_SIZE` value: `WAVES` in bits 11:0, `WAVESIZE` from
    /// bit 12 (`gc_*_sh_mask.h`).
    pub fn tmpring_size(&self) -> u32 {
        self.waves
            | ((self.bytes_per_wave / self.family.wavesize_granule()) << TMPRING_WAVESIZE_SHIFT)
    }

    /// Attach the ring's memory: `base_address` must be the GPU VA of at least
    /// [`Self::backing_bytes`] bytes, 4 KiB aligned.
    pub fn bind(self, base_address: u64) -> Result<Gfx10Scratch, Gfx10ScratchError> {
        if base_address == 0 || base_address & 0xfff != 0 || base_address >> 48 != 0 {
            return Err(Gfx10ScratchError::InvalidBase(base_address));
        }
        Ok(Gfx10Scratch {
            plan: self,
            base_address,
        })
    }
}

/// A scratch ring bound to memory. `Copy` and plain data: whoever creates it
/// owns the backing allocation and must keep it alive (and unshared with any
/// concurrently running queue) for as long as a stream built with it can run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gfx10Scratch {
    plan: Gfx10ScratchPlan,
    base_address: u64,
}

impl Gfx10Scratch {
    pub fn plan(&self) -> Gfx10ScratchPlan {
        self.plan
    }

    pub fn family(&self) -> Gfx10ScratchFamily {
        self.plan.family
    }

    pub fn base_address(&self) -> u64 {
        self.base_address
    }

    pub fn bytes_per_wave(&self) -> u32 {
        self.plan.bytes_per_wave
    }

    pub fn tmpring_size(&self) -> u32 {
        self.plan.tmpring_size()
    }

    /// The gfx10 private-segment buffer V# the command processor would copy
    /// from `amd_queue_t.scratch_resource_descriptor` into the first four user
    /// SGPRs: ring base, `SWIZZLE_ENABLE`, ring size as `NUM_RECORDS`, and a
    /// swizzled (`ADD_TID_ENABLE`) `GFX10_FORMAT_32_UINT` word whose
    /// `INDEX_STRIDE` is the wave size (ROCr leaves it "filled in by CP").
    pub fn gfx10_private_segment_buffer(&self, wave32: bool) -> [u32; 4] {
        let base = self.base_address;
        let index_stride = if wave32 {
            SQ_BUF_INDEX_STRIDE_32
        } else {
            SQ_BUF_INDEX_STRIDE_64
        };
        [
            base as u32,
            ((base >> 32) as u32 & 0xffff) | SQ_BUF_RSRC_WORD1_SWIZZLE_ENABLE,
            self.plan.backing_bytes() as u32,
            SQ_BUF_RSRC_WORD3_GFX10_SCRATCH | (index_stride << SQ_BUF_INDEX_STRIDE_SHIFT),
        ]
    }

    /// The gfx10 `FLAT_SCRATCH_INIT` user SGPRs for "absolute flat scratch":
    /// the 64-bit address of the ring base (LLVM `AMDGPUUsage.md`,
    /// "Flat Scratch Init").
    pub fn gfx10_flat_scratch_init(&self) -> [u32; 2] {
        [self.base_address as u32, (self.base_address >> 32) as u32]
    }

    /// The gfx11 `COMPUTE_DISPATCH_SCRATCH_BASE_{LO,HI}` values: ring base
    /// bits 39:8 and 47:40 (Mesa `radv_emit_compute_scratch`).
    pub(crate) fn gfx11_dispatch_scratch_base(&self) -> [u32; 2] {
        [
            (self.base_address >> 8) as u32,
            (self.base_address >> 40) as u32 & 0xff,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const XTX: Gfx10ScratchTopology = Gfx10ScratchTopology {
        compute_units: 96,
        shader_engines: 6,
        max_waves_per_cu: 32,
    };
    const HALO: Gfx10ScratchTopology = Gfx10ScratchTopology {
        compute_units: 40,
        shader_engines: 2,
        max_waves_per_cu: 32,
    };
    const CAP: u64 = 256 << 20;

    #[test]
    fn gfx11_plan_counts_waves_per_engine_and_makes_wavesize_odd() {
        // private12 on wave32: 12 B * 32 lanes = 384 B -> 2 units of 256 B,
        // made odd -> 3 units = 768 B; 96 CUs * 32 slots / 6 SEs = 512 per SE.
        let plan = Gfx10ScratchPlan::new(Gfx10ScratchFamily::Gfx11, XTX, 12 * 32, CAP).unwrap();
        assert_eq!(plan.bytes_per_wave(), 768);
        assert_eq!(plan.slots(), 3072);
        assert_eq!(plan.backing_bytes(), 3072 * 768);
        assert_eq!(plan.tmpring_size(), 512 | (3 << 12));
        let halo = Gfx10ScratchPlan::new(Gfx10ScratchFamily::Gfx11, HALO, 12 * 32, CAP).unwrap();
        assert_eq!(halo.slots(), 1280);
        assert_eq!(halo.tmpring_size(), 640 | (3 << 12));
    }

    #[test]
    fn gfx10_plan_counts_total_waves_in_1k_units() {
        let navi21 = Gfx10ScratchTopology {
            compute_units: 80,
            shader_engines: 4,
            max_waves_per_cu: 32,
        };
        // 12 B * 32 = 384 B -> 1 unit of 1 KiB (already odd).
        let plan = Gfx10ScratchPlan::new(Gfx10ScratchFamily::Gfx10, navi21, 12 * 32, CAP).unwrap();
        assert_eq!(plan.bytes_per_wave(), 1024);
        assert_eq!(plan.slots(), 2560);
        assert_eq!(plan.tmpring_size(), 2560 | (1 << 12));
        // 2 KiB -> 2 units -> odd 3 units.
        let plan = Gfx10ScratchPlan::new(Gfx10ScratchFamily::Gfx10, navi21, 2048, CAP).unwrap();
        assert_eq!(plan.bytes_per_wave(), 3072);
    }

    #[test]
    fn plan_is_capped_by_backing_but_never_below_one_workgroup() {
        // 544 B/lane * 64 lanes = 34816 B -> 137 units (odd) = 35072 B per wave.
        let plan =
            Gfx10ScratchPlan::new(Gfx10ScratchFamily::Gfx11, XTX, 544 * 64, 64 << 20).unwrap();
        assert_eq!(plan.bytes_per_wave(), 137 * 256);
        assert!(plan.backing_bytes() <= 64 << 20);
        assert_eq!(plan.slots() % XTX.shader_engines, 0);
        assert!(plan.slots() >= MIN_SCRATCH_SLOTS_PER_ENGINE * XTX.shader_engines);
        assert_eq!(
            Gfx10ScratchPlan::new(Gfx10ScratchFamily::Gfx11, XTX, 544 * 64, 1 << 20),
            Err(Gfx10ScratchError::BackingTooLarge {
                required: 32 * 6 * 137 * 256,
                cap: 1 << 20,
            })
        );
    }

    #[test]
    fn plan_rejects_unencodable_and_empty_inputs() {
        assert_eq!(
            Gfx10ScratchPlan::new(Gfx10ScratchFamily::Gfx10, XTX, 0, CAP),
            Err(Gfx10ScratchError::NothingRequired)
        );
        let empty = Gfx10ScratchTopology {
            compute_units: 0,
            ..XTX
        };
        assert_eq!(
            Gfx10ScratchPlan::new(Gfx10ScratchFamily::Gfx11, empty, 4, CAP),
            Err(Gfx10ScratchError::EmptyTopology(empty))
        );
        // 0x1fff 1 KiB units is the gfx10 ceiling.
        assert!(matches!(
            Gfx10ScratchPlan::new(Gfx10ScratchFamily::Gfx10, XTX, 0x1fff * 1024 + 1, u64::MAX),
            Err(Gfx10ScratchError::WaveSizeTooLarge { .. })
        ));
    }

    #[test]
    fn bind_validates_the_ring_base() {
        let plan = Gfx10ScratchPlan::new(Gfx10ScratchFamily::Gfx11, HALO, 64, CAP).unwrap();
        assert!(plan.bind(0).is_err());
        assert!(plan.bind(0x1100).is_err());
        assert!(plan.bind(1 << 48).is_err());
        let ring = plan.bind(0x7f12_3456_7000).unwrap();
        assert_eq!(ring.gfx11_dispatch_scratch_base(), [0x1234_5670, 0x7f]);
    }

    #[test]
    fn gfx10_private_segment_buffer_matches_rocr_v_sharp() {
        let navi21 = Gfx10ScratchTopology {
            compute_units: 80,
            shader_engines: 4,
            max_waves_per_cu: 32,
        };
        let plan = Gfx10ScratchPlan::new(Gfx10ScratchFamily::Gfx10, navi21, 384, CAP).unwrap();
        let ring = plan.bind(0x0000_7f12_3456_7000).unwrap();
        let wave32 = ring.gfx10_private_segment_buffer(true);
        let wave64 = ring.gfx10_private_segment_buffer(false);
        assert_eq!(wave32[0], 0x3456_7000);
        assert_eq!(wave32[1], 0x8000_7f12);
        assert_eq!(wave32[2], 2560 * 1024);
        assert_eq!(wave32[3], 0x21c1_4fac);
        assert_eq!(wave64[3], 0x21e1_4fac);
        assert_eq!(ring.gfx10_flat_scratch_init(), [0x3456_7000, 0x7f12]);
    }
}
