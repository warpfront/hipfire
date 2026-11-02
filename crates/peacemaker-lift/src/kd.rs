// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! C2: 64-byte HSA kernel descriptor codec.
//!
//! Layout and bit meanings are LLVM's "Code Object V3 Kernel Descriptor" table
//! (AMDGPUUsage.rst:6153-6320) and the `compute_pgm_rsrc{1,2,3}` tables that follow it.
//! `from_bytes` is structural (length, reserved bytes, reserved `kernel_code_properties`
//! bits) and keeps every other bit as encoded; `validate` applies the per-subfamily
//! lifter rejects of core.md §5.2.

use peacemaker_ir::descriptor::{
    DescriptorError, KernargPreload, KernelCodeProperties, KernelDescriptor, Rsrc1, Rsrc2, Rsrc3,
};
use peacemaker_ir::inst::Arch;

pub const KD_SIZE: usize = 64;
/// Reserved byte ranges, in the order `KernelDescriptor::reserved` stores them (4 + 20 + 4).
const RESERVED: [(usize, usize); 3] = [(12, 16), (24, 44), (60, 64)];
/// `kernel_code_properties` bits 7–9 and 12–15: "Reserved, must be 0" on every target.
pub const KCP_RESERVED: u16 = 0xf380;
const KCP_PRIVATE_SEGMENT_BUFFER: u16 = 1 << 0;
const KCP_FLAT_SCRATCH_INIT: u16 = 1 << 5;
const KCP_USES_DYNAMIC_STACK: u16 = 1 << 11;
/// GFX120* rsrc1 bits that must be 0: SGPR granule (9:6, reserved on GFX10–12), PRIORITY
/// (11:10), PRIV (20), DEBUG_MODE (22), DISABLE_PERF (23), BULKY (24), CDBG_USER (25), 27, 28.
pub const GFX120_RSRC1_MUST_BE_ZERO: u32 = 0x1bd0_0fc0;
/// GFX11 bit 23 is ENABLE_IEEE_MODE rather than reserved DISABLE_PERF.
/// LLVM AMDGPUUsage.rst compute_pgm_rsrc1-gfx6-gfx12-table, bit 23.
pub const GFX11_RSRC1_MUST_BE_ZERO: u32 = GFX120_RSRC1_MUST_BE_ZERO & !(1 << 23);
/// GFX120* rsrc2 bit 6 is ENABLE_DYNAMIC_VGPR (the trap-handler bit on GFX6–11).
pub const GFX120_RSRC2_DYNAMIC_VGPR: u32 = 1 << 6;
/// rsrc2 23:15 GRANULATED_LDS_SIZE: CP takes LDS from the dispatch packet; must be 0.
pub const RSRC2_LDS_SIZE: u32 = 0x1ff << 15;
/// rsrc2 bits that must be 0: EXCEPTION_ADDRESS_WATCH (13), EXCEPTION_MEMORY (14), 31.
pub const RSRC2_MUST_BE_ZERO: u32 = (1 << 13) | (1 << 14) | (1 << 31);
/// GFX120* rsrc3: everything but INST_PREF_SIZE (11:4), GLG_EN (13) and IMAGE_OP (31) is
/// reserved; this includes bit 17 (ENABLE_DYNAMIC_VGPR only on GFX125*).
pub const GFX120_RSRC3_MUST_BE_ZERO: u32 = 0x7fff_d00f;
/// LLVM AMDGPUUsage.rst compute_pgm_rsrc3-gfx10-gfx11: shared VGPRs [3:0],
/// gfx11 prefetch [9:4], traps [11:10] must be zero, bits [30:12] reserved,
/// and IMAGE_OP [31] must be zero for compute.
pub const GFX11_RSRC3_MUST_BE_ZERO: u32 = 0xffff_fc00;

/// Byte codec for the kernel descriptor, so callers can write `KernelDescriptor::from_bytes`.
pub trait DescriptorCodec: Sized {
    /// Decodes exactly 64 bytes; rejects nonzero reserved bytes and reserved
    /// `kernel_code_properties` bits, keeps every other bit.
    fn from_bytes(bytes: &[u8]) -> Result<Self, DescriptorError>;
    /// Encodes every field, reserved bytes included; total.
    fn to_bytes(&self) -> [u8; KD_SIZE];
}

impl DescriptorCodec for KernelDescriptor {
    fn from_bytes(bytes: &[u8]) -> Result<Self, DescriptorError> {
        let bytes: &[u8; KD_SIZE] = bytes.try_into().map_err(|_| DescriptorError::Length)?;
        let mut reserved = [0u8; 28];
        let mut at = 0;
        for (lo, hi) in RESERVED {
            if let Some(i) = bytes[lo..hi].iter().position(|&b| b != 0) {
                return Err(DescriptorError::Reserved { offset: lo + i });
            }
            reserved[at..at + hi - lo].copy_from_slice(&bytes[lo..hi]);
            at += hi - lo;
        }
        let u16_at = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let properties = u16_at(56);
        if properties & KCP_RESERVED != 0 {
            let bit = (properties & KCP_RESERVED).trailing_zeros() as usize;
            return Err(DescriptorError::Reserved {
                offset: 56 + bit / 8,
            });
        }
        Ok(KernelDescriptor {
            group_segment_fixed_size: u32_at(0),
            private_segment_fixed_size: u32_at(4),
            kernarg_size: u32_at(8),
            kernel_code_entry_byte_offset: i64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            compute_pgm_rsrc3: Rsrc3(u32_at(44)),
            compute_pgm_rsrc1: Rsrc1(u32_at(48)),
            compute_pgm_rsrc2: Rsrc2(u32_at(52)),
            kernel_code_properties: KernelCodeProperties(properties),
            kernarg_preload: KernargPreload(u16_at(58)),
            reserved,
        })
    }

    fn to_bytes(&self) -> [u8; KD_SIZE] {
        let mut out = [0u8; KD_SIZE];
        out[0..4].copy_from_slice(&self.group_segment_fixed_size.to_le_bytes());
        out[4..8].copy_from_slice(&self.private_segment_fixed_size.to_le_bytes());
        out[8..12].copy_from_slice(&self.kernarg_size.to_le_bytes());
        out[16..24].copy_from_slice(&self.kernel_code_entry_byte_offset.to_le_bytes());
        out[44..48].copy_from_slice(&self.compute_pgm_rsrc3.0.to_le_bytes());
        out[48..52].copy_from_slice(&self.compute_pgm_rsrc1.0.to_le_bytes());
        out[52..56].copy_from_slice(&self.compute_pgm_rsrc2.0.to_le_bytes());
        out[56..58].copy_from_slice(&self.kernel_code_properties.0.to_le_bytes());
        out[58..60].copy_from_slice(&self.kernarg_preload.0.to_le_bytes());
        let mut at = 0;
        for (lo, hi) in RESERVED {
            out[lo..hi].copy_from_slice(&self.reserved[at..at + hi - lo]);
            at += hi - lo;
        }
        out
    }
}

/// Why a structurally valid descriptor cannot be lifted for a target.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum DescriptorReject {
    #[error(transparent)]
    Descriptor(#[from] DescriptorError),
    #[error("{field} bits {bits:#x} must be 0 on {arch:?}")]
    ReservedBits {
        field: &'static str,
        bits: u32,
        arch: Arch,
    },
    #[error("USES_DYNAMIC_STACK is set")]
    DynamicStack,
    #[error("{0} is unsupported with architected flat scratch")]
    ArchitectedFlatScratch(&'static str),
    #[error("GRANULATED_LDS_SIZE is {0}; it must be 0 (LDS comes from the dispatch packet)")]
    LdsGranule(u32),
    #[error("descriptor validity is not tabulated for {0:?}")]
    UnsupportedArch(Arch),
}

/// Applies LLVM's per-architecture kernel-descriptor tables (AMDGPUUsage.rst,
/// compute_pgm_rsrc3-gfx10-gfx11 and compute_pgm_rsrc3-gfx12).
pub fn validate(kd: &KernelDescriptor, arch: Arch) -> Result<(), DescriptorReject> {
    if !matches!(arch, Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1201) {
        return Err(DescriptorReject::UnsupportedArch(arch));
    }
    let reserved_bytes = RESERVED
        .iter()
        .flat_map(|&(lo, hi)| lo..hi)
        .zip(kd.reserved);
    if let Some((offset, _)) = reserved_bytes.into_iter().find(|&(_, b)| b != 0) {
        return Err(DescriptorError::Reserved { offset }.into());
    }
    let reserved = |field, bits: u32| {
        if bits == 0 {
            Ok(())
        } else {
            Err(DescriptorReject::ReservedBits { field, bits, arch })
        }
    };
    let properties = kd.kernel_code_properties.0;
    reserved(
        "kernel_code_properties",
        u32::from(properties & KCP_RESERVED),
    )?;
    if properties & KCP_USES_DYNAMIC_STACK != 0 {
        return Err(DescriptorReject::DynamicStack);
    }
    if properties & KCP_PRIVATE_SEGMENT_BUFFER != 0 {
        return Err(DescriptorReject::ArchitectedFlatScratch(
            "ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER",
        ));
    }
    if properties & KCP_FLAT_SCRATCH_INIT != 0 {
        return Err(DescriptorReject::ArchitectedFlatScratch(
            "ENABLE_SGPR_FLAT_SCRATCH_INIT",
        ));
    }
    // The KD table defines KERNARG_PRELOAD only for GFX90A/GFX942; GFX12 has no row, so a
    // nonzero value would ask CP for preloaded SGPRs it does not provide.
    reserved("kernarg_preload", u32::from(kd.kernarg_preload.0))?;
    reserved(
        "compute_pgm_rsrc1",
        kd.compute_pgm_rsrc1.0
            & if arch == Arch::Gfx1201 {
                GFX120_RSRC1_MUST_BE_ZERO
            } else {
                GFX11_RSRC1_MUST_BE_ZERO
            },
    )?;
    let rsrc2 = kd.compute_pgm_rsrc2.0;
    if rsrc2 & GFX120_RSRC2_DYNAMIC_VGPR != 0 {
        return Err(DescriptorError::DynamicVgpr.into());
    }
    if rsrc2 & RSRC2_LDS_SIZE != 0 {
        return Err(DescriptorReject::LdsGranule((rsrc2 & RSRC2_LDS_SIZE) >> 15));
    }
    reserved("compute_pgm_rsrc2", rsrc2 & RSRC2_MUST_BE_ZERO)?;
    let rsrc3 = kd.compute_pgm_rsrc3.0;
    let mask = if arch == Arch::Gfx1201 {
        GFX120_RSRC3_MUST_BE_ZERO
    } else {
        GFX11_RSRC3_MUST_BE_ZERO
            | if kd.kernel_code_properties.wave32() {
                0xf
            } else {
                0
            }
    };
    reserved("compute_pgm_rsrc3", rsrc3 & mask)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elf::fixtures::kt48_co;

    /// The selected KT48 kernel's descriptor at file offset 0x2240 (core.md §0/§8 T5).
    fn kt48_selected() -> (KernelDescriptor, [u8; KD_SIZE]) {
        let co = kt48_co();
        let bytes: [u8; KD_SIZE] = co[0x2240..0x2240 + KD_SIZE].try_into().unwrap();
        (KernelDescriptor::from_bytes(&bytes).unwrap(), bytes)
    }

    #[test]
    fn kt48_selected_descriptor_fields_and_bytes() {
        let (kd, original) = kt48_selected();
        assert_eq!(kd.group_segment_fixed_size, 0);
        assert_eq!(kd.private_segment_fixed_size, 0);
        assert_eq!(kd.kernarg_size, 328);
        assert_eq!(kd.kernel_code_entry_byte_offset, 23_744);
        assert_eq!(0x2240 + kd.kernel_code_entry_byte_offset, 0x7f00);
        assert_eq!(kd.compute_pgm_rsrc3, Rsrc3(0x530));
        assert_eq!(kd.compute_pgm_rsrc1, Rsrc1(0xe00f_001d));
        assert_eq!(kd.compute_pgm_rsrc2, Rsrc2(0x384));
        assert_eq!(kd.kernel_code_properties, KernelCodeProperties(0x408));
        assert!(kd.kernel_code_properties.wave32());
        assert_ne!(kd.compute_pgm_rsrc1.0 & (1 << 29), 0, "WGP_MODE");
        assert_eq!(kd.compute_pgm_rsrc1.next_free_vgpr(true), 240);
        assert_eq!(kd.to_bytes(), original);
        validate(&kd, Arch::Gfx1201).unwrap();
    }

    #[test]
    fn from_bytes_rejects_reserved_and_bad_length() {
        let (_, original) = kt48_selected();
        for offset in [12, 15, 24, 43, 60, 63] {
            let mut bytes = original;
            bytes[offset] = 0x10;
            assert_eq!(
                KernelDescriptor::from_bytes(&bytes),
                Err(DescriptorError::Reserved { offset })
            );
        }
        for (bit, offset) in [(7, 56), (9, 57), (12, 57), (15, 57)] {
            let mut bytes = original;
            let properties = u16::from_le_bytes([bytes[56], bytes[57]]) | (1 << bit);
            bytes[56..58].copy_from_slice(&properties.to_le_bytes());
            assert_eq!(
                KernelDescriptor::from_bytes(&bytes),
                Err(DescriptorError::Reserved { offset }),
                "bit {bit}"
            );
        }
        assert_eq!(
            KernelDescriptor::from_bytes(&original[..63]),
            Err(DescriptorError::Length)
        );
    }

    #[test]
    fn validate_applies_gfx120_rejects_and_keeps_defined_bits() {
        let (kd, _) = kt48_selected();
        let with = |edit: fn(&mut KernelDescriptor)| {
            let mut kd = kd.clone();
            edit(&mut kd);
            validate(&kd, Arch::Gfx1201)
        };
        assert_eq!(
            with(|kd| kd.compute_pgm_rsrc2.0 |= 1 << 6),
            Err(DescriptorError::DynamicVgpr.into())
        );
        assert_eq!(
            with(|kd| kd.compute_pgm_rsrc2.0 |= 3 << 15),
            Err(DescriptorReject::LdsGranule(3))
        );
        assert_eq!(
            with(|kd| kd.compute_pgm_rsrc3.0 |= 1 << 17),
            Err(DescriptorReject::ReservedBits {
                field: "compute_pgm_rsrc3",
                bits: 1 << 17,
                arch: Arch::Gfx1201
            })
        );
        assert_eq!(
            with(|kd| kd.compute_pgm_rsrc1.0 |= 1 << 6),
            Err(DescriptorReject::ReservedBits {
                field: "compute_pgm_rsrc1",
                bits: 1 << 6,
                arch: Arch::Gfx1201
            })
        );
        assert_eq!(with(|kd| kd.private_segment_fixed_size = 16), Ok(()));
        assert_eq!(
            with(|kd| kd.kernel_code_properties.0 |= 1 << 11),
            Err(DescriptorReject::DynamicStack)
        );
        assert!(matches!(
            with(|kd| kd.kernel_code_properties.0 |= 1 << 5),
            Err(DescriptorReject::ArchitectedFlatScratch(_))
        ));
        assert!(matches!(
            with(|kd| kd.kernarg_preload.0 = 1),
            Err(DescriptorReject::ReservedBits {
                field: "kernarg_preload",
                ..
            })
        ));
        assert_eq!(
            with(|kd| kd.reserved[4] = 1),
            Err(DescriptorError::Reserved { offset: 24 }.into())
        );
        // GLG_EN (rsrc3 bit 13) and WG_RR_EN (rsrc1 bit 21) are defined on GFX12: kept, not rejected.
        assert_eq!(
            with(|kd| {
                kd.compute_pgm_rsrc3.0 |= 1 << 13;
                kd.compute_pgm_rsrc1.0 |= 1 << 21
            }),
            Ok(())
        );
        assert_eq!(
            validate(&kd, Arch::Gfx1030),
            Err(DescriptorReject::UnsupportedArch(Arch::Gfx1030))
        );
    }
}
