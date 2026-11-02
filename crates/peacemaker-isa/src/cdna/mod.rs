// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The CDNA arm of PeaceMaker: wave64 kernels for CDNA3 and later, gfx942
//! (MI300X) first.
//!
//! The RDNA builder (`crate::Builder`) is a text builder over the gfx11/gfx12
//! codec in `peacemaker-ir`; this arm is typed end to end over
//! [`peacemaker_cdna`]: kernels push [`peacemaker_cdna::Inst`] values, the
//! [`builder::Builder`] inserts the `s_waitcnt`s the shared wait ledger
//! (`crate::ledger`, CDNA3 rules for `crate::Arch::Gfx942`) proves necessary
//! and the barrier drains, [`hazard`] inserts the wait states CDNA3 requires
//! (CDNA3 ISA §4.5 Table 11, §7.5), and [`emit`] links the encoded words into
//! an HSA code object (v5/v6) through the native ELF writer — no ROCm tool in
//! the path. `peacemaker_cdna::sim` executes the result on the CPU.
//!
//! - [`builder`]: checked wave64 builder (registers, kernargs, LDS, waits,
//!   branches, barriers).
//! - [`ops`]: instruction constructors with assembler-style e32/e64 choice.
//! - [`hazard`]: CDNA3 wait-state insertion and checking.
//! - [`emit`]: descriptor directives, metadata, code object, `.s` oracle text.
pub mod builder;
pub mod emit;
pub mod hazard;
pub mod ops;

pub use peacemaker_cdna as model;
use peacemaker_cdna::enc::sop::SoppOp;
use peacemaker_cdna::Inst;

/// One program item before branch resolution.
#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    Inst(Inst),
    /// A branch target (local label, no `.L` prefix required).
    Label(String),
    /// `s_branch` / `s_cbranch_*` to a label; the offset is resolved after
    /// hazard NOP insertion.
    Branch {
        op: SoppOp,
        target: String,
    },
}

/// What the descriptor and metadata need to know about a kernel.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Resources {
    /// One past the highest SGPR used (s0..).
    pub next_free_sgpr: u16,
    /// One past the highest architectural VGPR used (v0..).
    pub next_free_vgpr: u16,
    /// AccVGPRs used (a0..).
    pub agprs: u16,
    /// First unified register of the AGPRs: `next_free_vgpr` rounded up to 4
    /// (at least 4); `.amdhsa_accum_offset`.
    pub accum_offset: u16,
    pub uses_vcc: bool,
    pub group_segment_fixed_size: u32,
    pub kernarg_size: u32,
    /// `.amdhsa_system_sgpr_workgroup_id_{x,y,z}`.
    pub workgroup_id: [bool; 3],
    /// `.amdhsa_system_vgpr_workitem_id` (0 = X, 1 = XY, 2 = XYZ).
    pub workitem_id: u8,
    /// Flat work-items per workgroup (`.max_flat_workgroup_size`).
    pub workgroup_size: u32,
    /// `.amdhsa_fp16_overflow`.
    pub fp16_ovfl: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernArgKind {
    /// A global pointer (`.value_kind: global_buffer`); `readonly` adds
    /// `.is_const`/`.actual_access: read_only` metadata.
    GlobalBuffer { readonly: bool },
    /// A by-value scalar (`.value_kind: by_value`).
    ByValue,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernArg {
    pub name: String,
    pub offset: u32,
    pub size: u32,
    pub kind: KernArgKind,
}

/// A finished, certified kernel.
#[derive(Clone, Debug)]
pub struct Kernel {
    pub arch: peacemaker_cdna::Arch,
    pub name: String,
    /// Final instructions in order (waits and hazard NOPs included,
    /// branches resolved to offsets).
    pub insts: Vec<Inst>,
    /// Their encoding.
    pub words: Vec<u32>,
    /// Label → dword offset in `words`.
    pub labels: Vec<(String, usize)>,
    pub resources: Resources,
    pub kernargs: Vec<KernArg>,
    pub waits: Vec<crate::ledger::WaitProof>,
    pub hazards: Vec<crate::hazard::HazardProof>,
}
