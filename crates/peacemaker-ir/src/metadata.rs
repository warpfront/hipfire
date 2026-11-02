// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HsaKernelMetadata {
    pub raw_msgpack: Vec<u8>,
    pub parsed: KernelMeta,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KernelMeta {
    pub name: String,
    pub symbol: String,
    pub args: Vec<Kernarg>,
    pub kernarg_segment_size: u32,
    pub kernarg_segment_align: u32,
    pub group_segment_fixed_size: u32,
    pub private_segment_fixed_size: u32,
    pub vgpr_count: u32,
    pub sgpr_count: u32,
    pub wavefront_size: u32,
    pub max_flat_workgroup_size: u32,
    pub workgroup_processor_mode: bool,
    pub uniform_work_group_size: bool,
    pub uses_dynamic_stack: bool,
    pub sgpr_spill_count: u32,
    pub vgpr_spill_count: u32,
}
/// One `.args[]` entry. `address_space` is the pointer's `.address_space` (`global`,
/// `generic`, …) when the map carries one.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Kernarg {
    pub name: String,
    pub size: u32,
    pub offset: u32,
    pub value_kind: String,
    pub address_space: Option<String>,
}
