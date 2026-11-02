// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The 64-byte AMDHSA kernel descriptor, gfx90a+ layout (LLVM `AMDGPUUsage`,
//! "Kernel Descriptor" for code object V3+; COMPUTE_PGM_RSRC1/2/3 fields for
//! GFX9/GFX90A/GFX942). Raw words plus field accessors: the writer
//! (`peacemaker-isa` `cdna::emit`) computes the values and proves them against
//! `llvm-mc`'s `.amdhsa_kernel` assembly; the emulator reads them.
use crate::Result;

pub const KD_BYTES: usize = 64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KernelDescriptor {
    /// Bytes 0..4.
    pub group_segment_fixed_size: u32,
    /// Bytes 4..8.
    pub private_segment_fixed_size: u32,
    /// Bytes 8..12.
    pub kernarg_size: u32,
    /// Bytes 16..24: entry address minus descriptor address.
    pub kernel_code_entry_byte_offset: i64,
    /// Bytes 44..48.
    pub compute_pgm_rsrc3: u32,
    /// Bytes 48..52.
    pub compute_pgm_rsrc1: u32,
    /// Bytes 52..56.
    pub compute_pgm_rsrc2: u32,
    /// Bytes 56..58.
    pub kernel_code_properties: u16,
    /// Bytes 58..60 (KERNARG_PRELOAD_SPEC_LENGTH [6:0], OFFSET [15:7]).
    pub kernarg_preload: u16,
}

fn bits(word: u32, lo: u32, width: u32) -> u32 {
    (word >> lo) & ((1u32 << width) - 1)
}

impl KernelDescriptor {
    pub fn to_bytes(&self) -> [u8; KD_BYTES] {
        let mut b = [0u8; KD_BYTES];
        b[0..4].copy_from_slice(&self.group_segment_fixed_size.to_le_bytes());
        b[4..8].copy_from_slice(&self.private_segment_fixed_size.to_le_bytes());
        b[8..12].copy_from_slice(&self.kernarg_size.to_le_bytes());
        b[16..24].copy_from_slice(&self.kernel_code_entry_byte_offset.to_le_bytes());
        b[44..48].copy_from_slice(&self.compute_pgm_rsrc3.to_le_bytes());
        b[48..52].copy_from_slice(&self.compute_pgm_rsrc1.to_le_bytes());
        b[52..56].copy_from_slice(&self.compute_pgm_rsrc2.to_le_bytes());
        b[56..58].copy_from_slice(&self.kernel_code_properties.to_le_bytes());
        b[58..60].copy_from_slice(&self.kernarg_preload.to_le_bytes());
        b
    }

    /// Parse; reserved bytes must be zero.
    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        if b.len() != KD_BYTES {
            return Err(format!(
                "kernel descriptor is {} bytes, not {KD_BYTES}",
                b.len()
            ));
        }
        if b[12..16]
            .iter()
            .chain(&b[24..44])
            .chain(&b[60..64])
            .any(|&x| x != 0)
        {
            return Err("kernel descriptor reserved bytes are not zero".into());
        }
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        Ok(Self {
            group_segment_fixed_size: u32_at(0),
            private_segment_fixed_size: u32_at(4),
            kernarg_size: u32_at(8),
            kernel_code_entry_byte_offset: i64::from_le_bytes(b[16..24].try_into().unwrap()),
            compute_pgm_rsrc3: u32_at(44),
            compute_pgm_rsrc1: u32_at(48),
            compute_pgm_rsrc2: u32_at(52),
            kernel_code_properties: u16::from_le_bytes([b[56], b[57]]),
            kernarg_preload: u16::from_le_bytes([b[58], b[59]]),
        })
    }

    // COMPUTE_PGM_RSRC1 (GFX9 layout).
    pub fn granulated_workitem_vgpr_count(&self) -> u32 {
        bits(self.compute_pgm_rsrc1, 0, 6)
    }
    pub fn granulated_wavefront_sgpr_count(&self) -> u32 {
        bits(self.compute_pgm_rsrc1, 6, 4)
    }
    pub fn float_round_mode_32(&self) -> u32 {
        bits(self.compute_pgm_rsrc1, 12, 2)
    }
    pub fn float_round_mode_16_64(&self) -> u32 {
        bits(self.compute_pgm_rsrc1, 14, 2)
    }
    pub fn float_denorm_mode_32(&self) -> u32 {
        bits(self.compute_pgm_rsrc1, 16, 2)
    }
    pub fn float_denorm_mode_16_64(&self) -> u32 {
        bits(self.compute_pgm_rsrc1, 18, 2)
    }
    pub fn dx10_clamp(&self) -> bool {
        bits(self.compute_pgm_rsrc1, 21, 1) != 0
    }
    pub fn ieee_mode(&self) -> bool {
        bits(self.compute_pgm_rsrc1, 23, 1) != 0
    }
    /// FP16_OVFL (GFX9+): also governs F32 → FP8/BF8 overflow (CDNA3 ISA §7.2 p.51).
    pub fn fp16_overflow(&self) -> bool {
        bits(self.compute_pgm_rsrc1, 26, 1) != 0
    }

    // COMPUTE_PGM_RSRC2.
    pub fn enable_private_segment(&self) -> bool {
        bits(self.compute_pgm_rsrc2, 0, 1) != 0
    }
    pub fn user_sgpr_count(&self) -> u32 {
        bits(self.compute_pgm_rsrc2, 1, 5)
    }
    pub fn enable_trap_handler(&self) -> bool {
        bits(self.compute_pgm_rsrc2, 6, 1) != 0
    }
    /// Workgroup ID X/Y/Z system SGPRs (`dim` 0..=2).
    pub fn enable_workgroup_id(&self, dim: u32) -> bool {
        bits(self.compute_pgm_rsrc2, 7 + dim, 1) != 0
    }
    pub fn enable_workgroup_info(&self) -> bool {
        bits(self.compute_pgm_rsrc2, 10, 1) != 0
    }
    /// 0: X, 1: X and Y, 2: X, Y and Z work-item IDs.
    pub fn enable_vgpr_workitem_id(&self) -> u32 {
        bits(self.compute_pgm_rsrc2, 11, 2)
    }

    // COMPUTE_PGM_RSRC3 (GFX90A/GFX942).
    /// ACCUM_OFFSET field: first AGPR at `4 * (field + 1)` unified registers.
    pub fn accum_offset_field(&self) -> u32 {
        bits(self.compute_pgm_rsrc3, 0, 6)
    }
    pub fn tg_split(&self) -> bool {
        bits(self.compute_pgm_rsrc3, 16, 1) != 0
    }

    // KERNEL_CODE_PROPERTIES.
    pub fn enable_sgpr_private_segment_buffer(&self) -> bool {
        self.kernel_code_properties & 1 != 0
    }
    pub fn enable_sgpr_dispatch_ptr(&self) -> bool {
        self.kernel_code_properties & (1 << 1) != 0
    }
    pub fn enable_sgpr_queue_ptr(&self) -> bool {
        self.kernel_code_properties & (1 << 2) != 0
    }
    pub fn enable_sgpr_kernarg_segment_ptr(&self) -> bool {
        self.kernel_code_properties & (1 << 3) != 0
    }
    pub fn enable_sgpr_dispatch_id(&self) -> bool {
        self.kernel_code_properties & (1 << 4) != 0
    }
    pub fn enable_sgpr_flat_scratch_init(&self) -> bool {
        self.kernel_code_properties & (1 << 5) != 0
    }
    pub fn enable_sgpr_private_segment_size(&self) -> bool {
        self.kernel_code_properties & (1 << 6) != 0
    }
    pub fn enable_wavefront_size32(&self) -> bool {
        self.kernel_code_properties & (1 << 10) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_trip() {
        let kd = KernelDescriptor {
            group_segment_fixed_size: 4096,
            kernarg_size: 24,
            kernel_code_entry_byte_offset: 192,
            compute_pgm_rsrc1: 0x00af_0042,
            compute_pgm_rsrc2: 0x0000_0784,
            compute_pgm_rsrc3: 3,
            kernel_code_properties: 8,
            ..Default::default()
        };
        let b = kd.to_bytes();
        assert_eq!(KernelDescriptor::from_bytes(&b).unwrap(), kd);
        assert_eq!(kd.user_sgpr_count(), 2);
        assert!(
            kd.enable_workgroup_id(0) && kd.enable_workgroup_id(1) && kd.enable_workgroup_id(2)
        );
        assert!(kd.enable_sgpr_kernarg_segment_ptr());
    }
}
