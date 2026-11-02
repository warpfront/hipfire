// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rsrc1(pub u32);
impl Rsrc1 {
    pub fn vgpr_granules(self) -> u32 {
        self.0 & 0x3f
    }
    pub fn next_free_vgpr(self, wave32: bool) -> u32 {
        (self.vgpr_granules() + 1) * if wave32 { 8 } else { 4 }
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rsrc2(pub u32);
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rsrc3(pub u32);
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KernelCodeProperties(pub u16);
impl KernelCodeProperties {
    pub fn wave32(self) -> bool {
        self.0 & (1 << 10) != 0
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KernargPreload(pub u16);
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelDescriptor {
    pub group_segment_fixed_size: u32,
    pub private_segment_fixed_size: u32,
    pub kernarg_size: u32,
    pub kernel_code_entry_byte_offset: i64,
    pub compute_pgm_rsrc3: Rsrc3,
    pub compute_pgm_rsrc1: Rsrc1,
    pub compute_pgm_rsrc2: Rsrc2,
    pub kernel_code_properties: KernelCodeProperties,
    pub kernarg_preload: KernargPreload,
    pub reserved: [u8; 28],
}
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum DescriptorError {
    #[error("kernel descriptor must be exactly 64 bytes")]
    Length,
    #[error("reserved descriptor field at byte {offset} is nonzero")]
    Reserved { offset: usize },
    #[error("dynamic VGPRs are unsupported on gfx1201")]
    DynamicVgpr,
}
