// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::{arch::Arch, hazard::HazardProof, lds::LdsSlot, ledger::WaitProof, reg::Range};
use serde::Serialize;

/// `actual_access` is the pointer's `.actual_access` metadata (`read_only`, `write_only`,
/// `read_write`) when the twinned hipcc kernel declares one; absent otherwise.
#[derive(Clone, Debug, Serialize)]
pub struct Kernarg {
    pub name: String,
    pub offset: u32,
    pub size: u32,
    pub value_kind: String,
    pub address_space: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual_access: Option<String>,
}
/// A global pointer's `.actual_access`, as clang derives it from the kernel's uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}
impl Access {
    pub fn name(self) -> &'static str {
        match self {
            Self::ReadOnly => "read_only",
            Self::WriteOnly => "write_only",
            Self::ReadWrite => "read_write",
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct KernargLayout {
    pub size: u32,
    pub args: Vec<Kernarg>,
}
impl KernargLayout {
    pub fn new(size: u32) -> Self {
        Self { size, args: vec![] }
    }
    pub fn pointer(mut self, name: &str, offset: u32) -> Self {
        self.args.push(Kernarg {
            name: name.into(),
            offset,
            size: 8,
            value_kind: "global_buffer".into(),
            address_space: Some("global".into()),
            actual_access: None,
        });
        self
    }
    /// A global pointer carrying the twinned kernel's frozen `.actual_access`.
    pub fn pointer_access(mut self, name: &str, offset: u32, access: Access) -> Self {
        self.args.push(Kernarg {
            name: name.into(),
            offset,
            size: 8,
            value_kind: "global_buffer".into(),
            address_space: Some("global".into()),
            actual_access: Some(access.name().into()),
        });
        self
    }
    pub fn hidden(mut self, name: &str, offset: u32, size: u32, value_kind: &str) -> Self {
        self.args.push(Kernarg {
            name: name.into(),
            offset,
            size,
            value_kind: value_kind.into(),
            address_space: None,
            actual_access: None,
        });
        self
    }
    pub fn validate(&self) -> Result<(), String> {
        let mut end = 0;
        for a in &self.args {
            if a.offset < end || a.offset.checked_add(a.size).is_none_or(|n| n > self.size) {
                return Err(format!("overlapping/out-of-bounds kernarg {}", a.name));
            }
            end = a.offset + a.size
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct KernelSpec {
    pub kernel_id: String,
    pub variant: String,
    pub arch: Arch,
    pub symbol: String,
    pub kernargs: KernargLayout,
    pub user_sgpr_count: u8,
    pub system_sgpr_workgroup_id_y: bool,
    pub workgroup_size: u16,
    pub group_segment_fixed_size: u32,
    pub wave32: bool,
    /// CU mode (`workgroup_processor_mode 0`): every wave of a workgroup on one CU.
    pub cu_mode: bool,
}
impl KernelSpec {
    /// Enabled system SGPRs: workgroup id X, plus Y and (when `workgroup_id_z`) Z.
    pub fn system_sgprs(&self, workgroup_id_z: bool) -> u8 {
        1 + u8::from(self.system_sgpr_workgroup_id_y) + u8::from(workgroup_id_z)
    }
    /// The `.amdhsa_user_sgpr_count` the descriptor declares: `user_sgpr_count` real user SGPRs
    /// padded per [`Arch::declared_user_sgprs`] (gfx1100: user + system >= 16, as LLVM).
    pub fn declared_user_sgprs(&self, workgroup_id_z: bool) -> u8 {
        self.arch
            .declared_user_sgprs(self.user_sgpr_count, self.system_sgprs(workgroup_id_z))
    }
    /// Where the hardware puts the workgroup ids of this kernel (after the DECLARED user SGPRs).
    pub fn workgroup_ids(&self, workgroup_id_z: bool) -> Result<crate::arch::WgIds, String> {
        self.arch
            .workgroup_ids(self.declared_user_sgprs(workgroup_id_z))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryScope {
    LdsOnly,
    Workgroup,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct IsaShape {
    pub instructions: usize,
    pub valu_slots: usize,
    pub vopd_pairs: usize,
    pub wmma: usize,
    pub ds: usize,
    pub vmem: usize,
    pub waits: usize,
    pub barriers: usize,
    pub next_free_vgpr: u16,
    pub next_free_sgpr: u16,
}
#[derive(Clone, Debug, Serialize)]
pub struct ClauseProof {
    pub start: usize,
    pub len: usize,
    pub class: String,
}
#[derive(Clone, Debug, Serialize)]
pub struct BarrierProof {
    pub pc_index: usize,
    pub transitions: Vec<String>,
}
#[derive(Clone, Debug, Serialize)]
pub struct LoopFixpoint {
    pub head: String,
    pub iterations: u8,
}
#[derive(Clone, Debug, Serialize)]
pub struct BuilderProof {
    pub kernel_id: String,
    pub variant: String,
    pub arch: Arch,
    pub builder_crate_version: String,
    pub builder_git_sha: String,
    pub reg_plan: Vec<Range>,
    pub next_free_vgpr: u16,
    pub next_free_sgpr: u16,
    /// The descriptor enables workgroup Z (gfx11 SGPR or gfx12 `ttmp7[31:16]`).
    #[serde(skip_serializing_if = "is_false")]
    pub system_sgpr_workgroup_id_z: bool,
    pub waits: Vec<WaitProof>,
    pub hazards: Vec<HazardProof>,
    pub clauses: Vec<ClauseProof>,
    pub vopd_pairs: usize,
    pub lds_slots: Vec<LdsSlot>,
    pub barriers: Vec<BarrierProof>,
    pub loop_fixpoints: Vec<LoopFixpoint>,
    pub forbidden_mnemonics_checked: Vec<String>,
    pub s_text_sha256: String,
    /// Declared release signals, event polls and fences (`sync::verify`); absent when none.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub syncs: Vec<crate::sync::SyncProof>,
    /// Task-body regions emitted inside this program (`Builder::region`); absent when none.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub regions: Vec<RegionProof>,
}
fn is_false(value: &bool) -> bool {
    !*value
}
/// Declared entry state of a task-body region ([`crate::Builder::region`]): the
/// caller-owned registers the body reads (argument block pointer, lane id,
/// virtual workgroup coordinate, ...), the LDS bytes it may use and the
/// physical workgroup size it assumes.
#[derive(Clone, Debug)]
pub struct RegionAbi {
    /// Labels `.L{name}_entry` / `.L{name}_exit` (local labels: not kernel symbols).
    pub name: String,
    /// `(meaning, register)`: read-only inputs, owned by the caller's register plan.
    pub inputs: Vec<(String, crate::reg::RegRef)>,
    pub lds_bytes: u32,
    pub workgroup_size: u16,
}
/// The entry label of region `name`.
pub fn region_entry_label(name: &str) -> String {
    format!(".L{name}_entry")
}
/// The exit label of region `name`.
pub fn region_exit_label(name: &str) -> String {
    format!(".L{name}_exit")
}
/// What `Builder::region` proved about one emitted region.
#[derive(Clone, Debug, Serialize)]
pub struct RegionProof {
    pub name: String,
    /// Instruction indices of the entry and exit labels.
    pub entry_pc: usize,
    pub exit_pc: usize,
    /// `(meaning, register)` of every declared input, live and unwritten across the region.
    pub inputs: Vec<(String, String)>,
    /// Register ranges live only inside the region (`Live::Between(entry, exit)`): free for
    /// the caller outside it, disjoint from every caller range inside it (`verify_lifetimes`).
    pub locals: Vec<String>,
    pub lds_bytes: u32,
    pub workgroup_size: u16,
    /// The body writes EXEC (e.g. a lane-0 store `v_cmpx`): the caller restores it at the exit.
    pub exec_clobbered: bool,
    /// Global writes of the body not yet covered by a release at the exit: the caller
    /// must release before publishing completion.
    pub unreleased_writes_at_exit: bool,
}
#[derive(Clone, Debug)]
pub struct Emitted {
    pub s_text: String,
    pub proof: BuilderProof,
    pub shape: IsaShape,
    /// The instruction list the proofs' `pc_index` fields index (labels included).
    pub program: crate::insn::Program,
    /// What `Builder::record_vgpr_stream` recorded (pass 1 of `regalloc::compact_vgprs`).
    pub vgpr_record: Option<crate::regalloc::Recording>,
    /// What the VGPR compaction measured (set by `regalloc::compact_vgprs`).
    pub vgpr_stats: Option<crate::regalloc::Stats>,
}
