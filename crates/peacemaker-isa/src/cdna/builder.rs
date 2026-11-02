// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The checked wave64 builder.
//!
//! Every pushed instruction goes through the shared wait ledger
//! (`crate::ledger::Ledger` with `crate::Arch::Gfx942` rules): the builder
//! inserts one combined `vmcnt/expcnt/lgkmcnt` wait when required, recording
//! a `WaitProof` for each counter. Forward branches join their ledgers at
//! the target; a backward branch must preserve the target's pending shape.
//! An unconditional branch has no fallthrough: a subsequent label must
//! have an incoming forward edge to resume construction.
//!
//! `barrier()` drains LGKM_CNT before `s_barrier`, including outstanding DS
//! reads (LDS WAR), writes (LDS RAW), and SMEM. All DS reads share one
//! conservative ledger slot because this API does not describe LDS regions.
//! `finish()` emits `s_endpgm` (which implicitly drains counters, CDNA3 ISA
//! §12.5 p.130), runs [`super::hazard::resolve`], resolves signed dword branch
//! offsets, and encodes the resulting [`super::Kernel`]. Proof indexes in
//! the finished kernel index `Kernel::insts`, not labels in the item stream.
//!
//! Register allocation is a bump allocator per file with alignment. System
//! registers come first: s[0:1] is the kernarg pointer, followed by enabled
//! workgroup IDs in X/Y/Z order; newly enabled dimensions must therefore be
//! requested in ascending order before the first [`Builder::sgprs`] call.
//! v0 is always reserved for packed work-item IDs. SGPRs are limited to
//! s0..s101, with two of the allocation reserved when VCC is used (CDNA3 ISA
//! §3.6.2 p.12); the VGPR/AGPR split uses an at-least-four, four-aligned
//! `accum_offset` and the architecture's unified-register limit.

use super::{Item, KernArg, KernArgKind, Kernel, Resources};
use crate::insn::{Instruction, MemoryClass};
use crate::ledger::{Counter, Ledger, Reason, WaitProof};
use crate::reg::{Kind, RegRef};
use crate::Arch;
use peacemaker_cdna::enc::flat::FlatSeg;
use peacemaker_cdna::enc::sop::{waitcnt_imm, Sop1Op, Sopp, SoppOp};
use peacemaker_cdna::inst::{MemKind, RegFile, RegSpan};
use peacemaker_cdna::{Inst, Src};
use std::collections::BTreeMap;

/// An LDS allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lds {
    pub offset: u32,
    pub bytes: u32,
}

pub struct Builder {
    arch: peacemaker_cdna::Arch,
    name: String,
    resources: Resources,
    kernargs: Vec<KernArg>,
    items: Vec<Item>,
    ledger: Ledger,
    waits: Vec<WaitProof>,
    labels: BTreeMap<String, Ledger>,
    incoming: BTreeMap<String, Ledger>,
    reachable: bool,
    allocated_sgprs: bool,
}

impl Builder {
    /// `arch` must be a CDNA arch (`Arch::Gfx942`); `workgroup_size` is the
    /// flat work-item count (multiple of 64, ≤ 1024).
    pub fn new(arch: Arch, name: &str, workgroup_size: u32) -> Result<Self, String> {
        let arch = arch
            .cdna()
            .ok_or("the CDNA builder requires a CDNA architecture")?;
        if workgroup_size == 0 || workgroup_size > 1024 || workgroup_size % 64 != 0 {
            return Err("workgroup size must be a multiple of 64 in 64..=1024".into());
        }
        if name.is_empty() {
            return Err("kernel name must not be empty".into());
        }
        Ok(Self {
            arch,
            name: name.into(),
            resources: Resources {
                next_free_sgpr: 2,
                next_free_vgpr: 1,
                accum_offset: 4,
                workgroup_size,
                ..Resources::default()
            },
            kernargs: Vec::new(),
            items: Vec::new(),
            ledger: Ledger::default(),
            waits: Vec::new(),
            labels: BTreeMap::new(),
            incoming: BTreeMap::new(),
            reachable: true,
            allocated_sgprs: false,
        })
    }

    /// Declare a global-pointer kernel argument (8 bytes, 8-aligned); returns its byte offset.
    pub fn arg_ptr(&mut self, name: &str, readonly: bool) -> Result<u32, String> {
        self.arg(name, 8, 8, KernArgKind::GlobalBuffer { readonly })
    }

    /// Declare a 4-byte by-value kernel argument; returns its byte offset.
    pub fn arg_u32(&mut self, name: &str) -> Result<u32, String> {
        self.arg(name, 4, 4, KernArgKind::ByValue)
    }

    fn arg(&mut self, name: &str, size: u32, align: u32, kind: KernArgKind) -> Result<u32, String> {
        if name.is_empty() || self.kernargs.iter().any(|arg| arg.name == name) {
            return Err(format!("empty or duplicate kernel argument name {name:?}"));
        }
        let offset = align_up(self.resources.kernarg_size, align)?;
        let end = offset
            .checked_add(size)
            .ok_or("kernel argument layout overflow")?;
        self.kernargs.push(KernArg {
            name: name.into(),
            offset,
            size,
            kind,
        });
        self.resources.kernarg_size = end;
        Ok(offset)
    }

    /// s[0:1]: the kernarg segment pointer.
    pub fn kernarg_base(&self) -> Src {
        Src::Sgpr(0)
    }

    /// The SGPR holding workgroup ID `dim` (0..=2); enables it. Newly enabled
    /// dimensions must be requested in ascending order before `sgprs`.
    pub fn workgroup_id(&mut self, dim: u8) -> Result<Src, String> {
        if dim > 2 {
            return Err("workgroup ID dimension must be 0..=2".into());
        }
        if self.allocated_sgprs {
            return Err("workgroup_id must precede sgprs".into());
        }
        let dim = usize::from(dim);
        if !self.resources.workgroup_id[dim] {
            if self.resources.workgroup_id[dim + 1..]
                .iter()
                .any(|enabled| *enabled)
            {
                return Err("new workgroup ID dimensions must be enabled in X/Y/Z order".into());
            }
            self.resources.workgroup_id[dim] = true;
            self.resources.next_free_sgpr += 1;
        }
        let before = self.resources.workgroup_id[..dim]
            .iter()
            .filter(|enabled| **enabled)
            .count();
        Ok(Src::Sgpr(2 + before as u8))
    }

    /// v0, the packed work-item IDs (X[9:0], Y[19:10], Z[29:20]); enables `dims` (1..=3) dimensions.
    pub fn workitem_ids(&mut self, dims: u8) -> Result<Src, String> {
        if !(1..=3).contains(&dims) {
            return Err("work-item ID dimensions must be 1..=3".into());
        }
        self.resources.workitem_id = self.resources.workitem_id.max(dims - 1);
        Ok(Src::Vgpr(0))
    }

    /// `count` consecutive SGPRs starting at a multiple of nonzero `align`.
    pub fn sgprs(&mut self, count: u8, align: u8) -> Result<Src, String> {
        let (base, end) = register_range(
            self.resources.next_free_sgpr,
            count,
            align,
            self.sgpr_limit(),
        )?;
        self.resources.next_free_sgpr = end;
        self.allocated_sgprs = true;
        Ok(Src::Sgpr(base))
    }

    /// `count` consecutive VGPRs starting at a multiple of nonzero `align`.
    pub fn vgprs(&mut self, count: u8, align: u8) -> Result<Src, String> {
        let (base, end) = register_range(
            self.resources.next_free_vgpr,
            count,
            align,
            self.arch.caps().vgprs,
        )?;
        let accum_offset = (end + 3) / 4 * 4;
        if accum_offset + self.resources.agprs > self.arch.caps().unified_vgprs {
            return Err("VGPR allocation exceeds the unified VGPR/AGPR limit".into());
        }
        self.resources.next_free_vgpr = end;
        self.resources.accum_offset = accum_offset.max(4);
        Ok(Src::Vgpr(base))
    }

    /// `count` consecutive AccVGPRs starting at a multiple of nonzero `align`.
    pub fn agprs(&mut self, count: u8, align: u8) -> Result<Src, String> {
        let (base, end) =
            register_range(self.resources.agprs, count, align, self.arch.caps().agprs)?;
        if self.resources.accum_offset + end > self.arch.caps().unified_vgprs {
            return Err("AGPR allocation exceeds the unified VGPR/AGPR limit".into());
        }
        self.resources.agprs = end;
        Ok(Src::Agpr(base))
    }

    /// Static LDS bytes (aligned); total ≤ the arch's LDS size.
    pub fn lds(&mut self, bytes: u32, align: u32) -> Result<Lds, String> {
        let offset = align_up(self.resources.group_segment_fixed_size, align)?;
        let end = offset.checked_add(bytes).ok_or("LDS allocation overflow")?;
        if end > self.arch.caps().lds_bytes {
            return Err("LDS allocation exceeds the architecture's LDS size".into());
        }
        self.resources.group_segment_fixed_size = end;
        Ok(Lds { offset, bytes })
    }

    /// MODE.FP16_OVFL for the kernel (descriptor bit).
    pub fn set_fp16_ovfl(&mut self, on: bool) {
        self.resources.fp16_ovfl = on;
    }

    /// Any instruction except `s_waitcnt`, `s_barrier`, `s_endpgm` and
    /// branches; waits are inserted before it as the ledger requires.
    /// Memory data/destinations must be plain SGPRs or VGPRs: the shared
    /// ledger cannot track AGPRs or special-register files. Copy through
    /// plain registers before using those files.
    /// Of the FLAT-format segments, only GLOBAL is supported: FLAT needs
    /// dual VM/LGKM accounting, and the descriptor has no private segment
    /// for SCRATCH (CDNA3 ISA §10.3).
    pub fn push(&mut self, inst: Inst) -> Result<(), String> {
        self.require_reachable()?;
        if matches!(&inst, Inst::Sopp(p) if matches!(p.op, SoppOp::Waitcnt | SoppOp::Barrier | SoppOp::Endpgm))
            || inst.branch_offset().is_some()
            || matches!(&inst, Inst::Sop1(p) if p.op == Sop1Op::SetpcB64)
        {
            return Err(
                "waits, barriers, program end and branches must use builder methods".into(),
            );
        }
        if matches!(&inst, Inst::Flat(i) if i.seg != FlatSeg::Global) {
            return Err("checked builder supports GLOBAL only: FLAT counts on VM+LGKM (CDNA3 ISA §10.3); SCRATCH requires a private segment".into());
        }
        if !plain_memory_data(&inst) {
            return Err("checked memory data/destinations must use plain SGPR/VGPR registers, not AGPRs or special-register files".into());
        }
        let footprint = inst.footprint();
        let memory = inst.memory();
        let uses_vcc = footprint
            .defs
            .iter()
            .chain(&footprint.uses)
            .any(|span| span.file == RegFile::Vcc);
        if uses_vcc && self.resources.next_free_sgpr > self.arch.caps().sgprs - 2 {
            return Err("VCC requires two SGPRs beyond the allocated SGPRs".into());
        }
        for span in footprint.defs.iter().chain(&footprint.uses) {
            let limit = match span.file {
                RegFile::Sgpr => Some(
                    self.arch.caps().sgprs
                        - if self.resources.uses_vcc || uses_vcc {
                            2
                        } else {
                            0
                        },
                ),
                RegFile::Vgpr => Some(self.arch.caps().vgprs),
                RegFile::Agpr => Some(self.arch.caps().agprs),
                _ => None,
            };
            if limit.is_some_and(|limit| u16::from(span.base) + u16::from(span.len) > limit) {
                return Err(format!(
                    "instruction register span {span:?} exceeds the architecture's limit"
                ));
            }
        }
        let shadow = Instruction {
            text: inst.to_string(),
            defs: footprint.defs.iter().filter_map(ledger_reg).collect(),
            uses: footprint.uses.iter().filter_map(ledger_reg).collect(),
            memory: memory.map(memory_class),
        };
        self.emit_waits(self.ledger.required(&shadow))?;
        self.ledger.record(Arch::Gfx942, &shadow);
        if matches!(
            memory,
            Some(MemKind::DsLoad | MemKind::DsAtomicRtn | MemKind::DsXlane)
        ) {
            self.ledger.tag_lds_load(0)?;
        }
        self.resources.uses_vcc |= uses_vcc;
        self.items.push(Item::Inst(inst));
        Ok(())
    }

    /// Place a branch target and join every incoming forward edge with the
    /// reachable fallthrough. A label with no predecessor cannot resume an
    /// unreachable instruction stream.
    pub fn label(&mut self, name: &str) -> Result<(), String> {
        if name.is_empty() || self.labels.contains_key(name) {
            return Err(format!("empty or duplicate label {name:?}"));
        }
        if !self.reachable && !self.incoming.contains_key(name) {
            return Err(format!("unreachable label {name} has no incoming branch"));
        }
        if let Some(incoming) = self.incoming.remove(name) {
            if self.reachable {
                self.ledger.join(&incoming);
            } else {
                self.ledger.resume(incoming);
            }
            self.reachable = true;
        }
        self.labels.insert(name.into(), self.ledger.clone());
        self.items.push(Item::Label(name.into()));
        Ok(())
    }

    /// `s_branch` / `s_cbranch_*` to `target`. Back edges must preserve the
    /// label's recorded pending ledger shape exactly.
    pub fn branch(&mut self, op: SoppOp, target: &str) -> Result<(), String> {
        self.require_reachable()?;
        let branch = Sopp { op, simm16: 0 };
        if branch.branch_offset().is_none() {
            return Err(format!("{op:?} is not a supported branch opcode"));
        }
        if target.is_empty() {
            return Err("branch target must not be empty".into());
        }
        let uses_vcc = matches!(op, SoppOp::CbranchVccz | SoppOp::CbranchVccnz);
        if uses_vcc && self.resources.next_free_sgpr > self.arch.caps().sgprs - 2 {
            return Err("VCC requires two SGPRs beyond the allocated SGPRs".into());
        }
        if let Some(at) = self.labels.get(target) {
            if self.ledger.shape() != at.shape() {
                return Err(format!(
                    "backward branch to {target} violates the pending-memory loop invariant"
                ));
            }
        } else if let Some(incoming) = self.incoming.get_mut(target) {
            incoming.join(&self.ledger);
        } else {
            self.incoming.insert(target.into(), self.ledger.clone());
        }
        self.resources.uses_vcc |= uses_vcc;
        self.items.push(Item::Branch {
            op,
            target: target.into(),
        });
        if op == SoppOp::Branch {
            self.reachable = false;
        }
        Ok(())
    }

    /// Drain LGKM_CNT, then `s_barrier`. No pending DS read or write may
    /// cross the barrier, even when its result/source is not used later.
    pub fn barrier(&mut self) -> Result<(), String> {
        self.require_reachable()?;
        let pending = self.ledger.retire_loads_of(&[0]).is_some()
            || self.ledger.pending_stores()
            || self
                .ledger
                .shape()
                .iter()
                .any(|entry| entry.0 == Counter::Lgkm);
        if pending {
            self.emit_waits(vec![(Counter::Lgkm, 0, Reason::Barrier)])?;
        }
        self.items.push(Item::Inst(Inst::Sopp(Sopp {
            op: SoppOp::Barrier,
            simm16: 0,
        })));
        Ok(())
    }

    /// Drain every counter (`s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)` as needed).
    pub fn wait_all(&mut self) -> Result<(), String> {
        self.require_reachable()?;
        let waits = self.ledger.drain();
        self.emit_waits(waits)
    }

    /// The items so far (read-only).
    pub fn items(&self) -> &[Item] {
        &self.items
    }

    /// `s_endpgm`, hazard pass, branch resolution, encoding.
    pub fn finish(mut self) -> Result<Kernel, String> {
        if let Some((target, _)) = self.incoming.first_key_value() {
            return Err(format!("undefined branch target {target}"));
        }
        self.items.push(Item::Inst(Inst::Sopp(Sopp {
            op: SoppOp::Endpgm,
            simm16: 0,
        })));
        let (items, mut hazards) = super::hazard::resolve(self.arch, self.items)?;
        let mut insts = Vec::with_capacity(items.len());
        let mut words = Vec::with_capacity(items.len() * 2);
        let mut labels = Vec::with_capacity(self.labels.len());
        let mut branches = Vec::new();
        let mut item_to_inst = Vec::with_capacity(items.len());
        let mut wait_index = 0;
        for item in items {
            let inst_index = insts.len();
            item_to_inst.push(inst_index);
            let mut is_branch = false;
            let inst = match item {
                Item::Label(name) => {
                    labels.push((name, words.len()));
                    continue;
                }
                Item::Inst(inst) => inst,
                Item::Branch { op, target } => {
                    is_branch = true;
                    branches.push((inst_index, words.len(), target));
                    Inst::Sopp(Sopp { op, simm16: 0 })
                }
            };
            if matches!(&inst, Inst::Sopp(p) if p.op == SoppOp::Waitcnt) {
                let original = self
                    .waits
                    .get(wait_index)
                    .ok_or("hazard pass added an unproved wait")?
                    .pc_index;
                while let Some(proof) = self.waits.get_mut(wait_index) {
                    if proof.pc_index != original {
                        break;
                    }
                    proof.pc_index = inst_index;
                    wait_index += 1;
                }
            }
            if is_branch {
                // SOPP branches occupy one dword; encode only after the
                // final target offset is known.
                words.push(0);
            } else {
                inst.encode(self.arch, &mut words)
                    .map_err(|error| format!("instruction {inst_index}: {error}"))?;
            }
            insts.push(inst);
        }
        if wait_index != self.waits.len() {
            return Err("hazard pass removed a proved wait".into());
        }
        for (inst_index, pc, target) in branches {
            let target_pc = labels
                .iter()
                .find(|(name, _)| name == &target)
                .map(|(_, pc)| *pc)
                .ok_or_else(|| format!("undefined branch target {target}"))?;
            let delta = target_pc as i128 - (pc as i128 + 1);
            let delta = i16::try_from(delta).map_err(|_| {
                format!("branch to {target} exceeds the signed 16-bit dword offset range")
            })?;
            let Inst::Sopp(branch) = &mut insts[inst_index] else {
                unreachable!("resolved branch is SOPP")
            };
            branch.simm16 = delta as u16;
            // Encode the resolved branch through the same codec, reusing
            // the output's spare capacity instead of allocating a buffer.
            branch.encode(self.arch, &mut words)?;
            words[pc] = words.pop().ok_or("branch encoder returned no word")?;
        }
        for proof in &mut hazards {
            proof.pc_index = *item_to_inst
                .get(proof.pc_index)
                .ok_or("hazard proof points outside the item stream")?;
        }
        Ok(Kernel {
            arch: self.arch,
            name: self.name,
            insts,
            words,
            labels,
            resources: self.resources,
            kernargs: self.kernargs,
            waits: self.waits,
            hazards,
        })
    }

    fn require_reachable(&self) -> Result<(), String> {
        if self.reachable {
            Ok(())
        } else {
            Err(
                "instruction stream is unreachable after s_branch; place an incoming label first"
                    .into(),
            )
        }
    }

    fn sgpr_limit(&self) -> u16 {
        self.arch.caps().sgprs - if self.resources.uses_vcc { 2 } else { 0 }
    }

    fn emit_waits(&mut self, waits: Vec<(Counter, u8, Reason)>) -> Result<(), String> {
        if waits.is_empty() {
            return Ok(());
        }
        let caps = self.arch.caps();
        let (mut vm, mut exp, mut lgkm) = (caps.vmcnt_max, caps.expcnt_max, caps.lgkmcnt_max);
        for (counter, count, _) in &waits {
            match counter {
                Counter::Vm => vm = vm.min(*count),
                Counter::Lgkm => lgkm = lgkm.min(*count),
                Counter::Exp => exp = exp.min(*count),
                _ => return Err(format!("{counter:?} is not a CDNA wait counter")),
            }
        }
        let inst = Inst::Sopp(Sopp {
            op: SoppOp::Waitcnt,
            simm16: waitcnt_imm(self.arch, vm, exp, lgkm)?,
        });
        let text = inst.to_string();
        let pc_index = self.items.len();
        for (counter, count, reason) in waits {
            let count = match counter {
                Counter::Vm => vm,
                Counter::Lgkm => lgkm,
                Counter::Exp => exp,
                _ => count,
            };
            self.waits.push(WaitProof {
                pc_index,
                insn: text.clone(),
                counter,
                count,
                reason,
            });
            self.ledger.wait(counter, count);
        }
        self.items.push(Item::Inst(inst));
        Ok(())
    }
}

fn align_up(value: u32, align: u32) -> Result<u32, String> {
    if align == 0 {
        return Err("allocation alignment must not be zero".into());
    }
    let padding = (align - value % align) % align;
    value
        .checked_add(padding)
        .ok_or_else(|| "allocation alignment overflow".into())
}

fn register_range(next: u16, count: u8, align: u8, limit: u16) -> Result<(u8, u16), String> {
    if count == 0 {
        return Err("register allocation count must not be zero".into());
    }
    let base = align_up(u32::from(next), u32::from(align))?;
    let end = base + u32::from(count);
    if end > u32::from(limit) {
        return Err(format!(
            "register allocation ending at {end} exceeds limit {limit}"
        ));
    }
    Ok((base as u8, end as u16))
}

fn ledger_reg(span: &RegSpan) -> Option<RegRef> {
    let kind = match span.file {
        RegFile::Sgpr => Kind::S,
        RegFile::Vgpr => Kind::V,
        _ => return None,
    };
    Some(RegRef {
        kind,
        base: span.base,
        len: span.len,
    })
}

fn memory_class(memory: MemKind) -> MemoryClass {
    match memory {
        MemKind::VmemLoad | MemKind::VmemAtomicRtn => MemoryClass::VmemLoad,
        MemKind::VmemStore | MemKind::VmemAtomic => MemoryClass::VmemStore,
        MemKind::DsLoad | MemKind::DsAtomicRtn | MemKind::DsXlane => MemoryClass::DsLoad,
        MemKind::DsStore | MemKind::DsAtomic => MemoryClass::DsStore,
        MemKind::SmemLoad => MemoryClass::SmemLoad,
    }
}

fn plain_memory_data(inst: &Inst) -> bool {
    let plain = |src: Option<Src>| src.is_none_or(|src| matches!(src, Src::Sgpr(_) | Src::Vgpr(_)));
    match inst {
        Inst::Smem(i) => plain(i.sdata),
        Inst::Flat(i) => plain(i.vdst) && plain(i.data),
        Inst::Ds(i) => plain(i.vdst) && plain(i.data0) && plain(i.data1),
        Inst::Mubuf(i) => plain(Some(i.vdata)),
        _ => true,
    }
}
