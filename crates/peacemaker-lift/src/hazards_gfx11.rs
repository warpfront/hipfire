// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Static hardware-hazard checker over lifted gfx1100 (and, where the rule exists, gfx1151 /
//! gfx1201, wave32 and wave64) kernels.
//!
//! The emulator executes instructions atomically and in order, so it cannot see the hazards
//! the real pipeline has. LLVM's `GCNHazardRecognizer` only protects code the compiler
//! schedules itself: inline `asm` is opaque to it (`getWaitStatesSince` skips inline asm,
//! `hasHazard` does not look into it), so hand-placed sequences and PM-authored streams are
//! unprotected. This module re-implements, over the lifted instruction stream and its CFG,
//! every `GCNHazardRecognizer` rule whose feature predicate is enabled for the target, with
//! the *same* search semantics LLVM uses: a backwards walk from the consumer over all
//! predecessor paths (a hazard on any path is a finding), expiring on the rule's own
//! mitigation set.
//!
//! gfx1201's SGPR rule (`gfx12-valu-read-sgpr`) is evaluated by `rule_gfx12_sgpr`, which runs
//! `peacemaker_ir::passes::hazards::analyze_with(.., SgprModel::LlvmExact)` (PM's port of
//! `AMDGPUWaitSGPRHazards`) over every gfx1201 kernel and turns each demanded-but-absent
//! `s_wait_alu` into a Required finding; the `hazard_check` example additionally prints the
//! pipeline-default `analyze` replay for cross-checking.
//!
//! Source of truth: ROCm LLVM `d6f6cb691863e9258034ab48c1dfce21689d5606`
//! (`llvm/lib/Target/AMDGPU/{GCNHazardRecognizer.cpp, GCNSubtarget.h, AMDGPU.td,
//! SIISelLowering.cpp, VOP3Instructions.td, VOP1Instructions.td, SIInsertWaitcnts.cpp,
//! AMDGPUWaitSGPRHazards.cpp}`), the same compiler `/opt/rocm` ships. See [`RULES`] for the
//! rule table: which rules apply to which target, the LLVM function that defines each, the
//! mitigation it needs, and the rules that exist in LLVM but are *not* enabled for gfx1100
//! (with the `AMDGPU.td` feature line that excludes them).
//!
//! ## Rules applied to gfx1201 (wave32 and wave64)
//!
//! Derived from the same commit's `AMDGPU.td` `FeatureISAVersion12` / `FeatureGFX12` sets and
//! `GCNHazardRecognizer::fixHazards` / `PreEmitNoopsCommon` (local copy of the cited sources:
//! `pm-krsi-1100/scratch/HazardGate/src/`). Enabled, and implemented here:
//! * `gfx12-valu-read-sgpr`: `AMDGPUWaitSGPRHazards` (`hasVALUReadSGPRHazard()`), the only
//!   gfx12 SGPR/VCC hazard. It tracks 64-bit SGPR pairs ever read by a VALU; a SALU write
//!   later read (SALU or VALU) needs `s_wait_alu sa_sdst(0)`, a VALU write read by a VALU
//!   needs `va_sdst(0)` (`va_vcc(0)` for VCC), a VALU write read by a SALU needs nothing.
//!   VCC is the pair `vcc_lo/vcc_hi` (explicit operand or implicit by descriptor), EXEC is
//!   never tracked, wave64 lane masks are SGPR pairs. SMRD and non-FLAT VMEM retire all
//!   pending writes; FLAT/global/scratch do not; `s_setpc_b64`/`s_swappc_b64` flush. The
//!   checker *verifies* the `s_wait_alu` the incumbent objects carry (the
//!   `gfx1201_incumbents_*` tests neutralise them and watch the findings appear).
//! * `gfx11-vcmpx-permlane` (`FeatureVcmpxPermlaneHazard` is in `FeatureISAVersion12`):
//!   `v_cmpx` writing EXEC then `v_permlane*` needs a non-nop VALU between.
//! * `gfx11-wmma-chain` (`fixWMMAHazards`, ungated; gfx12 also checks the SWMMAC index).
//!
//! Not enabled for gfx12, so deliberately not checked (see [`RULES`] for the gate of each):
//! `FeatureGFX12` carries `NoDataDepHazard`, so `PreEmitNoopsCommon` returns before the VMEM /
//! VALU / DPP / v_div_fmas / readlane-writelane / s_getreg-s_setreg / RFE / M0 / MAI checks:
//! DPP reads of VALU-written VGPRs, `v_readlane`/`v_s_*` SGPR writers feeding VMEM, `s_getreg`,
//! `ds_swizzle`/`ds_bpermute`, `global_inv`, `s_barrier_signal`/`s_barrier_wait` have no
//! recognizer rule. `fixVALUTransUseHazard` is gfx1100-only (TRANS->VALU use is
//! hardware-interlocked on gfx12; `s_delay_alu` TRANS32_DEP is a performance hint),
//! `fixVALUPartialForwardingHazard` / `fixVALUMaskWriteHazard` are `generation == GFX11`
//! (gfx12 wave64 uses AMDGPUWaitSGPRHazards instead), `fixVcmpxExecWARHazard`,
//! `fixLdsBranchVmemWARHazard`, `fixVMEMtoScalarWriteHazards`, `fixSMEMtoVectorWriteHazards`,
//! `checkNSAtoVMEMHazard` are gfx10.1 features. LDSDIR hazards cannot occur (not in the ISA
//! table). Counter waits (`s_wait_loadcnt`/`dscnt`/`kmcnt`) are `SIInsertWaitcnts`, not
//! hazards, and are verified by `peacemaker_ir::passes::waits`.
//!
//! Severity:
//! * [`Severity::Required`]: LLVM enables the rule for this target; a hit is a real hardware
//!   hazard (or a codegen constraint the hardware bug demands) and must be mitigated.
//! * [`Severity::Advisory`]: the rule exists in LLVM only for *other* targets (gfx10.1) or is
//!   documented only by the ISA guide as hardware-interlocked; reported so the exact site and
//!   separation are visible, but NOT a violation on gfx1100 by the vendor's rules.
//! * [`Severity::Unchecked`]: a rule that applies but cannot be evaluated by this checker
//!   (fail closed; never silently passes).

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::hash::Hash;

use peacemaker_ir::cfg::{Block, Body};
use peacemaker_ir::effects::MemClass;
use peacemaker_ir::inst::{Abi, Arch, Form, Inst, Kernel, Program, Wave};
use peacemaker_ir::operand::{ImmField, Operand, Special};
use peacemaker_ir::passes::hazards as pm_hazards;
use peacemaker_ir::reg::{Kind, RegRef};
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub enum Severity {
    Required,
    Advisory,
    Unchecked,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Required => "required",
            Severity::Advisory => "advisory",
            Severity::Unchecked => "unchecked",
        }
    }
}

/// One entry of the rule table.
#[derive(Clone, Copy, Debug)]
pub struct RuleInfo {
    pub id: &'static str,
    /// Targets (of the five lifted architectures) where LLVM enables the rule.
    pub applies: &'static str,
    pub severity: Severity,
    /// `GCNHazardRecognizer.cpp` / other LLVM function that defines it.
    pub llvm: &'static str,
    /// The `AMDGPU.td` / `GCNSubtarget.h` predicate that gates it.
    pub gate: &'static str,
    pub mitigation: &'static str,
    /// Whether this checker evaluates the rule.
    pub implemented: bool,
}

pub const RULES: &[RuleInfo] = &[
    RuleInfo {
        id: "gfx11-vcmpx-permlane",
        applies: "gfx1100 gfx1151 gfx1201 (every FeatureISAVersion11_* and FeatureISAVersion12 set)",
        severity: Severity::Required,
        llvm: "GCNHazardRecognizer::fixVcmpxPermlaneHazards",
        gate: "FeatureVcmpxPermlaneHazard in FeatureISAVersion11_Common (AMDGPU.td:2129) and FeatureISAVersion12 (AMDGPU.td:2260)",
        mitigation: "a non-v_nop VALU between the v_cmpx that writes EXEC and v_permlane*/v_permlane64 (LLVM inserts v_mov_b32 vX, vX; v_nop is discarded by SQ and does not count)",
        implemented: true,
    },
    RuleInfo {
        id: "gfx1100-valu-trans-use",
        applies: "gfx1100 only (not gfx1151)",
        severity: Severity::Required,
        llvm: "GCNHazardRecognizer::fixVALUTransUseHazard",
        gate: "FeatureVALUTransUseHazard in FeatureISAVersion11_0_Common / 11_Generic (absent from 11_5_Common)",
        mitigation: "s_waitcnt_depctr depctr_va_vdst(0) before the consumer, or >5 VALUs / >1 TRANS between, or a VMEM/DS/EXP between (they expire the hazard)",
        implemented: true,
    },
    RuleInfo {
        id: "gfx11-wmma-chain",
        applies: "gfx1100 gfx1151 gfx1201",
        severity: Severity::Required,
        llvm: "GCNHazardRecognizer::fixWMMAHazards",
        gate: "unconditional for isWMMA/isSWMMAC instructions (no feature gate)",
        mitigation: "any VALU (LLVM inserts v_nop) between a WMMA/SWMMAC and a following one whose A/B (gfx12 SWMMAC: also index) source overlaps the first one's destination",
        implemented: true,
    },
    RuleInfo {
        id: "gfx1100-mad64-intra-fwd",
        applies: "gfx1100 (FeatureMADIntraFwdBug: gfx11.0.x only)",
        severity: Severity::Required,
        llvm: "VOP3Instructions.td V_MAD_U64_U32_gfx11 / V_MAD_I64_I32_gfx11: Constraints = \"@earlyclobber $vdst\" under HasMADIntraFwdBug (codegen constraint, not a recognizer pass)",
        gate: "FeatureMADIntraFwdBug in FeatureISAVersion11_0_Common / 11_Generic (AMDGPU.td:2146,2158)",
        mitigation: "v_mad_u64_u32 / v_mad_i64_i32 vdst pair must not overlap any VGPR source (src0/src1/src2)",
        implemented: true,
    },
    RuleInfo {
        id: "gfx1100-user-sgpr-init16",
        applies: "gfx1100 gfx1102 (FeatureUserSGPRInit16Bug: 11.0.0, 11.0.2), wave32 compute",
        severity: Severity::Required,
        llvm: "SIISelLowering.cpp SITargetLowering::allocateSystemSGPRs (pads user SGPRs until user+system >= 16; asserts getNumPreloadedSGPRs() >= 16)",
        gate: "GCNSubtarget::hasUserSGPRInit16BugInWave32() = HasUserSGPRInit16Bug && isWave32()",
        mitigation: "kernel descriptor: USER_SGPR_COUNT + ENABLE_SGPR_WORKGROUP_ID_X/Y/Z + ENABLE_SGPR_WORKGROUP_INFO >= 16 (pad with dead user SGPRs). LLVM does not count the private-segment wave offset",
        implemented: true,
    },
    RuleInfo {
        id: "gfx12-valu-read-sgpr",
        applies: "gfx1201 only (HasGFX12Insts && !HasGFX1250Insts), wave32 and wave64",
        severity: Severity::Required,
        llvm: "AMDGPUWaitSGPRHazards.cpp (AMDGPUWaitSGPRHazards::runOnMachineBasicBlock / run: s_wait_alu insertion), replayed by peacemaker_ir::passes::hazards::analyze_with(SgprModel::LlvmExact)",
        gate: "GCNSubtarget::hasVALUReadSGPRHazard() = HasGFX12Insts && !HasGFX1250Insts (GCNSubtarget.h:625). Wave64 differs from wave32 only in 64-bit lane-mask / VCC SGPR pairs and DsNopCount (8 vs 4; ds_nop culls are off by default and absent from the ISA table)",
        mitigation: "s_wait_alu depctr_sa_sdst(0) when a tracked pair written by a SALU (or VCC written by a SALU) is read; depctr_va_sdst(0) when a pair written by a VALU is read by a VALU; depctr_va_vcc(0) when VCC written by a VALU is read by a VALU; a VALU write read by a SALU needs nothing; buffer_*/image_* and SMRD retire all pending writes, global_*/scratch_*/flat_* do not",
        implemented: true,
    },
    RuleInfo {
        id: "gfx11-wave64-valu-partial-forwarding",
        applies: "gfx1100 gfx1151, wave64 only",
        severity: Severity::Required,
        llvm: "GCNHazardRecognizer::fixVALUPartialForwardingHazard (early-outs `!ST.isWave64()`)",
        gate: "GCNSubtarget::hasVALUPartialForwardingHazard() = (generation == GFX11)",
        mitigation: "s_waitcnt_depctr (va_vdst/va_sdst) per LLVM's window search",
        implemented: false,
    },
    RuleInfo {
        id: "gfx11-wave64-valu-mask-write",
        applies: "gfx1100 gfx1151, wave64 only",
        severity: Severity::Required,
        llvm: "GCNHazardRecognizer::fixVALUMaskWriteHazard (early-outs `!ST.isWave64()`); AMDGPUWaitSGPRHazards::run returns false for wave32 gfx11",
        gate: "GCNSubtarget::hasVALUMaskWriteHazard() = (generation == GFX11)",
        mitigation: "s_waitcnt_depctr sa_sdst(0) / va_sdst(0) around SGPR mask writes",
        implemented: false,
    },
    RuleInfo {
        id: "n/a-gfx12-no-data-dep-hazard",
        applies: "gfx1201 (NoDataDepHazard in FeatureGFX12, AMDGPU.td:1729): none of the rules below exist",
        severity: Severity::Advisory,
        llvm: "GCNHazardRecognizer::PreEmitNoopsCommon returns before checkVMEMHazards (VALU-written SGPR -> VMEM), checkVALUHazards, checkDPPHazards (VALU VGPR write -> DPP read), checkDivFMasHazards (VCC write -> v_div_fmas), checkRWLaneHazards (v_readlane/v_writelane lane-select SGPR), checkGetRegHazards / checkSetRegHazards (s_getreg/s_setreg), checkRFEHazards, checkReadM0*Hazards, checkMAI*Hazards; checkSMRDHazards runs first but only acts with XNACK (not supported on gfx12) or on SI",
        gate: "GCNSubtarget::hasNoDataDepHazard() (FeatureNoDataDepHazard is in FeatureGFX11 and FeatureGFX12). global_inv, ds_swizzle_b32, ds_bpermute_b32, s_barrier_signal / s_barrier_wait, s_sendmsg, v_s_*_f32 (SALU-float pseudo-scalar VALU, a VALU SGPR writer covered by gfx12-valu-read-sgpr) and the SALU-float s_*_f32 have no GCNHazardRecognizer rule; their ordering is SIInsertWaitcnts (passes::waits) territory",
        mitigation: "not applicable: the hardware interlocks these dependencies on gfx12",
        implemented: false,
    },
    RuleInfo {
        id: "n/a-gfx12-trans-use",
        applies: "gfx1100 only (not gfx1151, not gfx12)",
        severity: Severity::Advisory,
        llvm: "GCNHazardRecognizer::fixVALUTransUseHazard / fixVALUTransCoexecutionHazards",
        gate: "FeatureVALUTransUseHazard only in FeatureISAVersion11_0_Common / 11_Generic (AMDGPU.td:2144,2157); FeatureTransCoexecutionHazard only in gfx1250 sets (AMDGPU.td:2401+); absent from FeatureISAVersion12",
        mitigation: "none on gfx12: a TRANS result is hardware-interlocked. s_delay_alu instid TRANS32_DEP (AMDGPUInsertDelayAlu) is a pure performance hint, not a correctness requirement, and is validated by peacemaker's own s_delay_alu window checks, not here",
        implemented: false,
    },
    RuleInfo {
        id: "adv-mad64-overlap-dead-hi",
        applies: "gfx1100 (same gate as gfx1100-mad64-intra-fwd)",
        severity: Severity::Advisory,
        llvm: "shape LLVM itself emits under HasMADIntraFwdBug: `v_mad_u64_u32 v[2:3], null, s8, s15, v[1:2]` (dst low over src2 high, whose value is undef in the SSA); measured on /opt/rocm hipcc d6f6cb69",
        gate: "@earlyclobber $vdst only constrains live-range interference; an undef operand has none",
        mitigation: "none required for this shape; an overlap that is not this shape is gfx1100-mad64-intra-fwd",
        implemented: true,
    },
    RuleInfo {
        id: "adv-vmem-to-scalar-write",
        applies: "NOT gfx11. gfx1010/1011/1012 only",
        severity: Severity::Advisory,
        llvm: "GCNHazardRecognizer::fixVMEMtoScalarWriteHazards (SALU/SMEM writing an SGPR (or EXEC) that an earlier VMEM/DS reads; expires on any VALU, s_waitcnt 0, depctr vm_vsrc(0))",
        gate: "FeatureVMEMtoScalarWriteHazard only in FeatureISAVersion10_1_Common (AMDGPU.td:2050); gfx11 has FeatureNoDataDepHazard (FeatureGFX11)",
        mitigation: "none required on gfx1100; gfx10.1 mitigation would be s_waitcnt_depctr depctr_vm_vsrc(0)",
        implemented: true,
    },
    RuleInfo {
        id: "adv-vgpr-war-async-write",
        applies: "no LLVM rule for gfx11 DS/VMEM-load returns (LDSDIR-only analogue: fixLdsDirectVMEMHazard)",
        severity: Severity::Advisory,
        llvm: "GCNHazardRecognizer::fixLdsDirectVMEMHazard shape (VMEM/DS reads a VGPR, a later async writer overwrites it; expires on VALU / EXP / s_waitcnt 0 / depctr vm_vsrc(0)), applied to VMEM/DS load returns",
        gate: "none for gfx11 compute: LDSDIR does not exist in the lifted ISA table; RDNA3 ISA guide 10.8 only documents the *ALU*-write case (hardware stalls it); SIInsertWaitcnts::vmemWriteNeedsExpWaitcnt() is generation < SEA_ISLANDS",
        mitigation: "none required by vendor rules; a VALU/EXP, s_waitcnt lgkmcnt(0) (DS source) or vscnt(0) (VMEM store source) between would satisfy the conservative LDSDIR-shaped reading",
        implemented: true,
    },
    RuleInfo {
        id: "n/a-smem-to-vector-write",
        applies: "gfx1010 gfx1011 gfx1012 only",
        severity: Severity::Advisory,
        llvm: "GCNHazardRecognizer::fixSMEMtoVectorWriteHazards",
        gate: "FeatureSMEMtoVectorWriteHazard only in FeatureISAVersion10_1_Common (AMDGPU.td:2051)",
        mitigation: "not applicable to gfx11",
        implemented: false,
    },
    RuleInfo {
        id: "n/a-vcmpx-exec-war",
        applies: "gfx1010 gfx1011 gfx1012 only",
        severity: Severity::Advisory,
        llvm: "GCNHazardRecognizer::fixVcmpxExecWARHazard",
        gate: "FeatureVcmpxExecWARHazard only in FeatureISAVersion10_1_Common (AMDGPU.td:2053)",
        mitigation: "not applicable to gfx11",
        implemented: false,
    },
    RuleInfo {
        id: "n/a-lds-branch-vmem-war",
        applies: "gfx1010 gfx1011 gfx1012 only",
        severity: Severity::Advisory,
        llvm: "GCNHazardRecognizer::fixLdsBranchVmemWARHazard",
        gate: "FeatureLdsBranchVmemWARHazard only in FeatureISAVersion10_1_Common (AMDGPU.td:2054)",
        mitigation: "not applicable to gfx11",
        implemented: false,
    },
    RuleInfo {
        id: "n/a-nsa-to-vmem",
        applies: "gfx1010 gfx1011 gfx1012 only",
        severity: Severity::Advisory,
        llvm: "GCNHazardRecognizer::checkNSAtoVMEMHazard (hasNSAtoVMEMBug)",
        gate: "FeatureNSAtoVMEMBug only in FeatureISAVersion10_1_Common (AMDGPU.td:2055)",
        mitigation: "not applicable to gfx11 (image NSA only)",
        implemented: false,
    },
    RuleInfo {
        id: "n/a-lds-direct",
        applies: "gfx11+ (hasLdsDirect: generation >= GFX11) — pixel shaders only",
        severity: Severity::Advisory,
        llvm: "GCNHazardRecognizer::fixLdsDirectVALUHazard / fixLdsDirectVMEMHazard",
        gate: "LDSDIR (lds_param_load / lds_direct_load) is absent from the lifted ISA table (peacemaker-ir isa/gfx1100.tbl): `lift_object` rejects an object containing it (Rule::Decode), so no lifted kernel can contain one",
        mitigation: "n/a for compute",
        implemented: false,
    },
    RuleInfo {
        id: "n/a-others",
        applies: "no gfx1100 / gfx1201",
        severity: Severity::Advisory,
        llvm: "hasShift64HighRegBug (HasGFX90AInsts), hasDstSelForwardingHazard / hasVDecCoExecHazard / hasTransForwardingHazard (HasGFX940Insts), hasLdsWaitVMSRC (>= GFX12; only used by fixLdsDirectVMEMHazard, LDSDIR absent), hasFPAtomicToDenormModeHazard (== GFX10), hasReadM0*Hazard (GFX9), checkSoftClauseHazards (needs XNACK), hasDOTOpSelHazard (HasGFX940Insts||HasGFX11Insts: SIInstrInfo/codegen only, no recognizer rule), checkWMMACoexecutionHazards / fixVALUTransCoexecutionHazards / fixGetRegWaitIdle / fixSetRegMode / fixTDM / fixDsAtomicAsyncBarrierArriveB64 / fixScratchBaseForwardingHazard (gfx1250), fixRequiredExportPriority (gfx11.5/generic: exports only), hasFlatScratchSVSSwizzleBug (gfx11 scratch SVS: not a recognizer rule), fixVALUPartialForwardingHazard / fixVALUMaskWriteHazard (generation == GFX11 only; on gfx12 AMDGPUWaitSGPRHazards replaces them)",
        gate: "NoDataDepHazard (FeatureGFX11, FeatureGFX12) makes GCNHazardRecognizer::PreEmitNoopsCommon return before the VMEM/VALU/DPP/DivFMas/RWLane/SetReg/GetReg/RFE/ReadM0/MAI checks",
        mitigation: "not applicable",
        implemented: false,
    },
];

/// A located instruction in a finding.
#[derive(Clone, Debug)]
pub struct InstRef {
    pub role: &'static str,
    pub pc: u32,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct Finding {
    pub rule: &'static str,
    pub severity: Severity,
    pub kernel: String,
    /// The consumer / offending instruction (code-object virtual address).
    pub pc: u32,
    pub insts: Vec<InstRef>,
    pub required: String,
    pub present: String,
    pub detail: String,
    /// Layout-order neighbourhood of the consumer (up to 6 before, 2 after) for review.
    pub context: Vec<InstRef>,
}

impl Finding {
    pub fn to_json(&self) -> Value {
        json!({
            "rule": self.rule,
            "severity": self.severity.as_str(),
            "kernel": self.kernel,
            "pc": format!("{:#x}", self.pc),
            "instructions": self.insts.iter().map(|i| json!({
                "role": i.role,
                "pc": format!("{:#x}", i.pc),
                "text": i.text,
            })).collect::<Vec<_>>(),
            "required": self.required,
            "present": self.present,
            "detail": self.detail,
            "context": self.context.iter().map(|i| json!({
                "pc": format!("{:#x}", i.pc),
                "text": i.text,
            })).collect::<Vec<_>>(),
        })
    }
}

// ---------------------------------------------------------------- classification

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Class {
    Salu,
    Smem,
    Valu,
    Vmem,
    Ds,
    Export,
}

fn class_of(inst: &Inst) -> Class {
    match inst.form {
        Form::Sop1 | Form::Sop2 | Form::Sopc | Form::Sopk | Form::Sopp => Class::Salu,
        Form::Smem => Class::Smem,
        Form::Vop1
        | Form::Vop1Dpp
        | Form::Vop2
        | Form::Vop2Dpp
        | Form::Vopc
        | Form::Vop3
        | Form::Vop3p
        | Form::Vopd
        | Form::Vinterp => Class::Valu,
        Form::Ds => Class::Ds,
        Form::Vmem(_) => Class::Vmem,
        Form::Export => Class::Export,
    }
}

/// Mnemonic without the encoding suffix (`_e32`, `_e64`, `_dpp`, `_sdwa`).
fn base_name(name: &str) -> &str {
    let mut n = name;
    loop {
        let before = n;
        for suffix in ["_e32", "_e64", "_dpp", "_sdwa"] {
            if let Some(s) = n.strip_suffix(suffix) {
                n = s;
            }
        }
        if n == before {
            return n;
        }
    }
}

/// `SIInstrFlags::TRANS` on gfx11 (`VOP1Instructions.td`: the `let TRANS = 1` groups that
/// exist on GFX11: exp/log/rcp/rcp_iflag/rsq/sqrt f32, rcp/rsq/sqrt f64, sin/cos f32 and the
/// f16 forms).
fn is_trans_name(base: &str) -> bool {
    matches!(
        base,
        "v_exp_f32"
            | "v_log_f32"
            | "v_rcp_f32"
            | "v_rcp_iflag_f32"
            | "v_rsq_f32"
            | "v_sqrt_f32"
            | "v_rcp_f64"
            | "v_rsq_f64"
            | "v_sqrt_f64"
            | "v_sin_f32"
            | "v_cos_f32"
            | "v_rcp_f16"
            | "v_sqrt_f16"
            | "v_rsq_f16"
            | "v_log_f16"
            | "v_exp_f16"
            | "v_sin_f16"
            | "v_cos_f16"
    )
}

/// `isPermlane` restricted to what exists on gfx11/gfx12 (`GCNHazardRecognizer.cpp:519`).
fn is_permlane_name(base: &str) -> bool {
    matches!(
        base,
        "v_permlane16_b32"
            | "v_permlanex16_b32"
            | "v_permlane64_b32"
            | "v_permlane16_var_b32"
            | "v_permlanex16_var_b32"
    )
}

fn sopp_imm(inst: &Inst) -> Option<u16> {
    inst.operands.iter().find_map(|op| match op {
        Operand::Imm(ImmField::Sopp(n) | ImmField::Sopk(n)) => Some(*n as u16),
        _ => None,
    })
}

fn v_regs(regs: &[RegRef]) -> impl Iterator<Item = RegRef> + '_ {
    regs.iter().copied().filter(|r| r.kind == Kind::V)
}

fn s_regs(regs: &[RegRef]) -> impl Iterator<Item = RegRef> + '_ {
    regs.iter().copied().filter(|r| r.kind == Kind::S)
}

fn any_overlap(a: &[RegRef], b: &[RegRef]) -> bool {
    a.iter().any(|x| b.iter().any(|y| x.overlaps(*y)))
}

/// Decoded per-instruction facts the rules query.
struct Facts {
    name: &'static str,
    base: &'static str,
    class: Class,
    trans: bool,
    wmma: bool,
    swmmac: bool,
    nop_v: bool,
    /// `s_waitcnt_depctr` immediate.
    depctr: Option<u16>,
    /// `s_waitcnt` family: (vm, exp, lgkm, vs) fields, `None` = field not waited (max).
    wait: Option<WaitFields>,
    v_defs: Vec<RegRef>,
    v_uses: Vec<RegRef>,
    /// `v_uses`, except that a 64-bit MAD's src2 contributes only its low register: the high
    /// half of src2 is the undef operand of compiler-scheduled integer index math.
    v_uses_sans_mad_hi: Vec<RegRef>,
    s_defs: Vec<RegRef>,
    s_uses: Vec<RegRef>,
    writes_exec: bool,
    reads_exec: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct WaitFields {
    /// Raw `s_waitcnt` immediate when the instruction is `s_waitcnt`.
    raw_waitcnt: Option<u16>,
    lgkm0: bool,
    vscnt0: bool,
}

fn facts_of(inst: &Inst, arch: Arch) -> Facts {
    let name = inst.op.name(arch).unwrap_or("?");
    let base = base_name(name);
    let class = class_of(inst);
    let imm = sopp_imm(inst);
    let mut wait = None;
    let mut depctr = None;
    match name {
        "s_waitcnt" => {
            let raw = imm.unwrap_or(0xffff);
            wait = Some(WaitFields {
                raw_waitcnt: Some(raw),
                lgkm0: (raw >> 4) & 0x3f == 0,
                vscnt0: false,
            });
        }
        "s_waitcnt_lgkmcnt" => {
            wait = Some(WaitFields {
                raw_waitcnt: None,
                lgkm0: imm == Some(0),
                vscnt0: false,
            });
        }
        "s_waitcnt_vscnt" => {
            wait = Some(WaitFields {
                raw_waitcnt: None,
                lgkm0: false,
                vscnt0: imm == Some(0),
            });
        }
        "s_waitcnt_vmcnt" => wait = Some(WaitFields::default()),
        "s_waitcnt_depctr" => depctr = Some(imm.unwrap_or(0xffff)),
        _ => {}
    }
    let wmma = base.starts_with("v_wmma_");
    let swmmac = base.starts_with("v_swmmac_");
    let special_exec = |ops: &[Operand]| {
        ops.iter().any(|o| {
            matches!(
                o,
                Operand::Special(Special::Exec | Special::ExecLo | Special::ExecHi)
            )
        })
    };
    let writes_exec = inst.effects.implicit.writes & peacemaker_ir::effects::ImplicitSet::EXEC != 0;
    // Every VMEM/DS/FLAT instruction implicitly reads EXEC.
    let reads_exec = matches!(class, Class::Vmem | Class::Ds | Class::Export)
        || inst.effects.implicit.reads & peacemaker_ir::effects::ImplicitSet::EXEC != 0
        || special_exec(&inst.operands) && !writes_exec;
    let v_uses: Vec<RegRef> = v_regs(&inst.effects.uses).collect();
    let v_uses_sans_mad_hi = if matches!(base, "v_mad_u64_u32" | "v_mad_i64_i32") {
        inst.operands
            .iter()
            .enumerate()
            .filter_map(|(idx, op)| match op {
                Operand::Reg(r) if r.kind == Kind::V && idx >= 2 => Some(if idx == 4 {
                    RegRef { len: 1, ..*r }
                } else {
                    *r
                }),
                _ => None,
            })
            .collect()
    } else {
        v_uses.clone()
    };
    Facts {
        name,
        base,
        class,
        trans: class == Class::Valu && is_trans_name(base),
        wmma,
        swmmac,
        nop_v: base == "v_nop",
        depctr,
        wait,
        v_defs: v_regs(&inst.effects.defs).collect(),
        v_uses,
        v_uses_sans_mad_hi,
        s_defs: s_regs(&inst.effects.defs).collect(),
        s_uses: s_regs(&inst.effects.uses).collect(),
        writes_exec,
        reads_exec,
    }
}

// ---------------------------------------------------------------- instruction stream

enum Step {
    Found,
    Expired,
    Next,
}

struct Hit<S> {
    producer: usize,
    /// Instructions strictly between producer and consumer on the shortest path found.
    dist: u32,
    state: S,
}

pub struct Stream<'a> {
    symbol: &'a str,
    arch: Arch,
    insts: Vec<&'a Inst>,
    facts: Vec<Facts>,
    pcs: Vec<u32>,
    block_of: Vec<usize>,
    blocks: &'a [Block],
}

impl<'a> Stream<'a> {
    pub fn new(symbol: &'a str, body: &'a Body, arch: Arch) -> Self {
        let insts: Vec<&Inst> = body
            .layout
            .iter()
            .map(|id| {
                body.insts
                    .get(*id)
                    .expect("layout names a live instruction")
            })
            .collect();
        let facts = insts.iter().map(|i| facts_of(i, arch)).collect();
        let pcs = insts
            .iter()
            .enumerate()
            .map(|(i, inst)| inst.prov.pc.unwrap_or(4 * i as u32))
            .collect();
        let mut block_of = vec![0; insts.len()];
        for (b, block) in body.blocks.iter().enumerate() {
            for slot in &mut block_of[block.range.0..block.range.1] {
                *slot = b;
            }
        }
        Self {
            symbol,
            arch,
            insts,
            facts,
            pcs,
            block_of,
            blocks: &body.blocks,
        }
    }

    pub fn len(&self) -> usize {
        self.insts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.insts.is_empty()
    }

    fn text(&self, i: usize) -> String {
        self.insts[i]
            .text(self.arch)
            .unwrap_or_else(|_| self.facts[i].name.to_owned())
    }

    fn at(&self, role: &'static str, i: usize) -> InstRef {
        InstRef {
            role,
            pc: self.pcs[i],
            text: self.text(i),
        }
    }

    /// Indices of the instructions executed immediately before `i`.
    fn pred_ends(&self, i: usize, out: &mut Vec<usize>) {
        let b = self.block_of[i];
        let block = &self.blocks[b];
        if i > block.range.0 {
            out.push(i - 1);
            return;
        }
        // Block entry: every predecessor's last instruction. An empty predecessor block is
        // transparent.
        let mut stack: Vec<usize> = block.preds.iter().map(|p| p.0).collect();
        let mut seen = HashSet::new();
        while let Some(p) = stack.pop() {
            if !seen.insert(p) {
                continue;
            }
            let pb = &self.blocks[p];
            if pb.range.1 > pb.range.0 {
                out.push(pb.range.1 - 1);
            } else {
                stack.extend(pb.preds.iter().map(|q| q.0));
            }
        }
    }

    /// LLVM `hasHazard` / `getWaitStatesSince`: walk backwards from `consumer` over every
    /// predecessor path. `step` sees each earlier instruction (nearest first) and the path's
    /// rule state; it may declare the hazard found, the path expired, or continue. States are
    /// memoised per block entry, so loops terminate. All distinct producers are returned.
    fn search_back<S: Clone + Eq + Hash>(
        &self,
        consumer: usize,
        init: S,
        step: impl FnMut(&mut S, usize) -> Step,
    ) -> Vec<Hit<S>> {
        self.search_back_ex(consumer, init, step).0
    }

    /// [`Self::search_back`], plus whether some path reached the kernel entry without being
    /// found or expired.
    fn search_back_ex<S: Clone + Eq + Hash>(
        &self,
        consumer: usize,
        init: S,
        mut step: impl FnMut(&mut S, usize) -> Step,
    ) -> (Vec<Hit<S>>, bool) {
        let mut entry_reached = false;
        let mut hits: BTreeMap<usize, Hit<S>> = BTreeMap::new();
        let mut visited: HashSet<(usize, S)> = HashSet::new();
        let mut work: VecDeque<(usize, S, u32)> = VecDeque::new();
        let mut seeds = Vec::new();
        self.pred_ends(consumer, &mut seeds);
        if seeds.is_empty() {
            entry_reached = true;
        }
        for p in seeds {
            work.push_back((p, init.clone(), 0));
        }
        while let Some((cursor, mut state, mut dist)) = work.pop_front() {
            let b = self.block_of[cursor];
            let block = &self.blocks[b];
            if cursor + 1 == block.range.1 && !visited.insert((b, state.clone())) {
                continue;
            }
            let mut i = cursor;
            let mut ended = false;
            loop {
                match step(&mut state, i) {
                    Step::Found => {
                        let hit = Hit {
                            producer: i,
                            dist,
                            state: state.clone(),
                        };
                        match hits.get(&i) {
                            Some(old) if old.dist <= dist => {}
                            _ => {
                                hits.insert(i, hit);
                            }
                        }
                        ended = true;
                        break;
                    }
                    Step::Expired => {
                        ended = true;
                        break;
                    }
                    Step::Next => dist = dist.saturating_add(1),
                }
                if i == block.range.0 {
                    break;
                }
                i -= 1;
            }
            if !ended {
                let mut preds = Vec::new();
                self.pred_ends(block.range.0, &mut preds);
                if preds.is_empty() {
                    entry_reached = true;
                }
                for p in preds {
                    work.push_back((p, state.clone(), dist));
                }
            }
        }
        (hits.into_values().collect(), entry_reached)
    }

    /// First instructions executed after `i`: the next instruction, or the first instruction
    /// of each successor block (empty successor blocks are transparent).
    fn succ_starts(&self, i: usize, out: &mut Vec<usize>) {
        let b = self.block_of[i];
        let block = &self.blocks[b];
        if i + 1 < block.range.1 {
            out.push(i + 1);
            return;
        }
        let mut stack: Vec<usize> = block.succs.iter().map(|p| p.0).collect();
        let mut seen = HashSet::new();
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            let nb = &self.blocks[n];
            if nb.range.1 > nb.range.0 {
                out.push(nb.range.0);
            } else {
                stack.extend(nb.succs.iter().map(|q| q.0));
            }
        }
    }

    /// Is VGPR `n` read, on some path, after instruction `i` and before it is rewritten?
    fn vgpr_read_after(&self, i: usize, n: u16) -> bool {
        let reg = RegRef {
            kind: Kind::V,
            base: n,
            len: 1,
        };
        let mut visited: HashSet<usize> = HashSet::new();
        let mut work = Vec::new();
        self.succ_starts(i, &mut work);
        while let Some(start) = work.pop() {
            if !visited.insert(start) {
                continue;
            }
            let b = self.block_of[start];
            let mut j = start;
            let mut killed = false;
            loop {
                let g = &self.facts[j];
                if any_overlap(&g.v_uses_sans_mad_hi, &[reg]) {
                    return true;
                }
                if any_overlap(&g.v_defs, &[reg]) {
                    killed = true;
                    break;
                }
                j += 1;
                if j >= self.blocks[b].range.1 {
                    break;
                }
            }
            if !killed {
                let mut next = Vec::new();
                self.succ_starts(self.blocks[b].range.1 - 1, &mut next);
                work.extend(next);
            }
        }
        false
    }

    fn finding(
        &self,
        rule: &'static str,
        severity: Severity,
        consumer: usize,
        producer: Option<usize>,
        required: String,
        present: String,
        detail: String,
    ) -> Finding {
        let mut insts = Vec::new();
        if let Some(p) = producer {
            insts.push(self.at("producer", p));
        }
        insts.push(self.at("consumer", consumer));
        let lo = consumer.saturating_sub(6);
        let hi = (consumer + 3).min(self.len());
        let context = (lo..hi).map(|i| self.at("ctx", i)).collect();
        Finding {
            rule,
            severity,
            kernel: self.symbol.to_owned(),
            pc: self.pcs[consumer],
            insts,
            required,
            present,
            detail,
            context,
        }
    }
}

// ---------------------------------------------------------------- rules

fn regs_text(regs: &[RegRef]) -> String {
    regs.iter()
        .map(|r| r.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// `fixVcmpxPermlaneHazards`.
fn rule_vcmpx_permlane(s: &Stream, out: &mut Vec<Finding>) {
    for i in 0..s.len() {
        let f = &s.facts[i];
        if f.class != Class::Valu || !is_permlane_name(f.base) {
            continue;
        }
        let hits = s.search_back(i, (), |_, j| {
            let g = &s.facts[j];
            if g.class == Class::Valu && g.writes_exec && g.base.starts_with("v_cmpx_") {
                Step::Found
            } else if g.class == Class::Valu && !g.nop_v {
                Step::Expired
            } else {
                Step::Next
            }
        });
        for h in hits {
            out.push(s.finding(
                "gfx11-vcmpx-permlane",
                Severity::Required,
                i,
                Some(h.producer),
                "at least one VALU other than v_nop between the v_cmpx (writes EXEC) and the permlane".into(),
                format!("{} instruction(s) between, none a non-nop VALU", h.dist),
                "GCNHazardRecognizer::fixVcmpxPermlaneHazards".into(),
            ));
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
struct TransState {
    valus: u8,
    trans: u8,
}

/// `fixVALUTransUseHazard` (gfx1100 only).
fn rule_trans_use(s: &Stream, out: &mut Vec<Finding>) {
    const INTV_MAX_VALUS: u8 = 5;
    const INTV_MAX_TRANS: u8 = 1;
    for i in 0..s.len() {
        let f = &s.facts[i];
        if f.class != Class::Valu || f.v_uses.is_empty() {
            continue;
        }
        let srcs = &f.v_uses;
        let hits = s.search_back(i, TransState::default(), |st, j| {
            if st.valus > INTV_MAX_VALUS || st.trans > INTV_MAX_TRANS {
                return Step::Expired;
            }
            let g = &s.facts[j];
            if matches!(g.class, Class::Vmem | Class::Ds | Class::Export)
                || g.depctr.is_some_and(|d| (d >> 12) & 0xf == 0)
            {
                return Step::Expired;
            }
            if g.trans && any_overlap(&g.v_defs, srcs) {
                return Step::Found;
            }
            if g.class == Class::Valu {
                st.valus = st.valus.saturating_add(1);
            }
            if g.trans {
                st.trans = st.trans.saturating_add(1);
            }
            Step::Next
        });
        for h in hits {
            out.push(s.finding(
                "gfx1100-valu-trans-use",
                Severity::Required,
                i,
                Some(h.producer),
                "s_waitcnt_depctr depctr_va_vdst(0) before the consumer (or >5 VALUs / >1 TRANS between, or a VMEM/DS/EXP between)".into(),
                format!(
                    "{} VALU(s) and {} TRANS between producer and consumer, no va_vdst(0) wait",
                    h.state.valus, h.state.trans
                ),
                format!(
                    "TRANS result {} read by a VALU within the forwarding window; GCNHazardRecognizer::fixVALUTransUseHazard",
                    regs_text(&s.facts[h.producer].v_defs)
                ),
            ));
        }
    }
}

/// `fixWMMAHazards`.
fn rule_wmma_chain(s: &Stream, out: &mut Vec<Finding>) {
    for i in 0..s.len() {
        let f = &s.facts[i];
        if !(f.wmma || f.swmmac) {
            continue;
        }
        let mut srcs: Vec<RegRef> = f.v_uses.iter().take(2).copied().collect();
        if s.arch == Arch::Gfx1201 && f.swmmac {
            if let Some(idx) = f.v_uses.get(2) {
                srcs.push(*idx);
            }
        }
        let hits = s.search_back(i, (), |_, j| {
            let g = &s.facts[j];
            if (g.wmma || g.swmmac) && g.v_defs.first().is_some_and(|d| any_overlap(&[*d], &srcs)) {
                Step::Found
            } else if g.class == Class::Valu {
                Step::Expired
            } else {
                Step::Next
            }
        });
        for h in hits {
            out.push(s.finding(
                "gfx11-wmma-chain",
                Severity::Required,
                i,
                Some(h.producer),
                "at least one VALU (v_nop is enough) between the two matrix instructions".into(),
                format!("{} instruction(s) between, none a VALU", h.dist),
                format!(
                    "source {} overlaps the previous matrix destination {}; GCNHazardRecognizer::fixWMMAHazards",
                    regs_text(&srcs),
                    regs_text(&s.facts[h.producer].v_defs[..1])
                ),
            ));
        }
    }
}

/// `V_MAD_*64_*32_gfx11` `@earlyclobber $vdst` under `HasMADIntraFwdBug`: the destination pair
/// may not overlap a *live* VGPR source.
///
/// LLVM does not treat an `undef` operand as interfering, and its integer index/stride math
/// builds a 64-bit src2 from one real register and one arbitrary register. Measured on
/// `/opt/rocm` hipcc (the compiler with the workaround enabled, see AMDGPU.td:2146): with the
/// workaround on it never overlaps src0/src1 or src2's low register (`-mattr=-mad-intra-fwd-bug`
/// emits `v[0:1]` over `v[0:1]`), but it does emit dst-low over src2-high
/// (`v_mad_u64_u32 v[2:3], null, s8, s15, v[1:2]`, 2 of 2 inspected families, across the whole
/// hipcc decode set). That shape is reported as the advisory `adv-mad64-overlap-dead-hi`;
/// every other overlap (src0, src1, src2 low half) is `gfx1100-mad64-intra-fwd`.
fn rule_mad64(s: &Stream, out: &mut Vec<Finding>) {
    for i in 0..s.len() {
        let f = &s.facts[i];
        if !matches!(f.base, "v_mad_u64_u32" | "v_mad_i64_i32") {
            continue;
        }
        let Some(dst) = f.v_defs.first().copied() else {
            continue;
        };
        // Operands are [vdst, sdst, src0, src1, src2].
        let reg_of = |idx: usize| match s.insts[i].operands.get(idx) {
            Some(Operand::Reg(r)) if r.kind == Kind::V => Some(*r),
            _ => None,
        };
        let roles = [
            ("src0", reg_of(2)),
            ("src1", reg_of(3)),
            ("src2", reg_of(4)),
        ];
        for (role, src) in roles {
            let Some(src) = src else { continue };
            if !src.overlaps(dst) {
                continue;
            }
            let lo = dst.base.max(src.base);
            let hi = (dst.base + u16::from(dst.len)).min(src.base + u16::from(src.len));
            let overlap: Vec<u16> = (lo..hi).collect();
            let src2_hi_only = role == "src2" && overlap == [src.base + 1];
            let dst_hi_dead = !s.vgpr_read_after(i, dst.base + 1);
            let (rule, severity, required) = if src2_hi_only {
                (
                    "adv-mad64-overlap-dead-hi",
                    Severity::Advisory,
                    "(shape LLVM itself emits with the workaround enabled: dst low over src2 high) no vendor rule forbids it",
                )
            } else {
                (
                    "gfx1100-mad64-intra-fwd",
                    Severity::Required,
                    "vdst must not overlap any live VGPR source (@earlyclobber $vdst)",
                )
            };
            out.push(s.finding(
                rule,
                severity,
                i,
                None,
                required.into(),
                format!(
                    "vdst {dst} overlaps {role} {src} on {}; destination high half {}",
                    overlap
                        .iter()
                        .map(|n| format!("v{n}"))
                        .collect::<Vec<_>>()
                        .join(","),
                    if dst_hi_dead { "never read" } else { "is read" }
                ),
                "MAD_U64/I64 intra instruction forwarding bug (FeatureMADIntraFwdBug)".into(),
            ));
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
struct ScalarWriteState;

/// `fixVMEMtoScalarWriteHazards` semantics (gfx10.1 rule), reported as advisory on gfx11.
fn rule_vmem_to_scalar_write(s: &Stream, out: &mut Vec<Finding>) {
    for i in 0..s.len() {
        let f = &s.facts[i];
        if !matches!(f.class, Class::Salu | Class::Smem) {
            continue;
        }
        if f.s_defs.is_empty() && !f.writes_exec {
            continue;
        }
        let defs = f.s_defs.clone();
        let hits = s.search_back(i, ScalarWriteState, |_, j| {
            let g = &s.facts[j];
            if matches!(g.class, Class::Vmem | Class::Ds)
                && (any_overlap(&g.s_uses, &defs) || (f.writes_exec && g.reads_exec))
            {
                return Step::Found;
            }
            let full_wait = g.wait.is_some_and(|w| w.raw_waitcnt == Some(0));
            if g.class == Class::Valu || full_wait || g.depctr.is_some_and(|d| (d >> 2) & 0x7 == 0)
            {
                return Step::Expired;
            }
            Step::Next
        });
        for h in hits {
            let what = if f.writes_exec
                && s.facts[h.producer].reads_exec
                && !any_overlap(&s.facts[h.producer].s_uses, &defs)
            {
                "EXEC".to_owned()
            } else {
                regs_text(&defs)
            };
            out.push(s.finding(
                "adv-vmem-to-scalar-write",
                Severity::Advisory,
                i,
                Some(h.producer),
                "(gfx10.1 rule only; not required on gfx1100) a VALU, s_waitcnt 0 or depctr vm_vsrc(0) between".into(),
                format!("{} instruction(s) between, no VALU / full wait / vm_vsrc(0)", h.dist),
                format!(
                    "{} is read by the earlier VMEM/DS and overwritten by a scalar instruction; vendor rule exists only for gfx1010/1011/1012 (FeatureVMEMtoScalarWriteHazard)",
                    what
                ),
            ));
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
struct WarState {
    lgkm0: bool,
    vscnt0: bool,
}

/// Data (store-source) VGPR operands of a DS/VMEM instruction, by table slot name.
fn data_vgprs(inst: &Inst, arch: Arch) -> Vec<RegRef> {
    let Some(row) = peacemaker_ir::isa::lookup(arch, inst.op, inst.form) else {
        return Vec::new();
    };
    let returns = peacemaker_ir::isa::atomic_returns(arch, &inst.mods.cpol);
    row.slots(returns)
        .zip(inst.operands.iter())
        .filter_map(|((slot, _), op)| {
            let data = slot.starts_with("DATA") || slot == "VDATA" || slot == "VSRC";
            match op {
                Operand::Reg(r) if data && r.kind == Kind::V => Some(*r),
                _ => None,
            }
        })
        .collect()
}

/// LDSDIR-shaped WAR (`fixLdsDirectVMEMHazard`) applied to asynchronous VGPR writers
/// (VMEM / DS / FLAT load and returning-atomic destinations): a store's *data* VGPRs
/// overwritten with no VALU, s_waitcnt 0, vm_vsrc(0), or the store's own completion wait.
fn rule_vgpr_war_async(s: &Stream, out: &mut Vec<Finding>) {
    for i in 0..s.len() {
        let f = &s.facts[i];
        let Some(mem) = s.insts[i].effects.mem else {
            continue;
        };
        let async_writer = matches!(
            mem.class,
            MemClass::VmemLoad
                | MemClass::VmemAtomic { returns: true }
                | MemClass::DsLoad
                | MemClass::DsAtomic { returns: true }
                | MemClass::FlatLoad
                | MemClass::FlatAtomic { returns: true }
        );
        if !async_writer || f.v_defs.is_empty() {
            continue;
        }
        let defs = f.v_defs.clone();
        let hits = s.search_back(i, WarState::default(), |st, j| {
            let g = &s.facts[j];
            if matches!(g.class, Class::Vmem | Class::Ds) {
                if let Some(m) = s.insts[j].effects.mem {
                    let store_like = matches!(
                        m.class,
                        MemClass::DsStore
                            | MemClass::DsAtomic { .. }
                            | MemClass::VmemStore
                            | MemClass::VmemAtomic { .. }
                            | MemClass::FlatStore
                            | MemClass::FlatAtomic { .. }
                    );
                    if store_like {
                        let data = data_vgprs(s.insts[j], s.arch);
                        let ds = g.class == Class::Ds;
                        let completed = if ds { st.lgkm0 } else { st.vscnt0 };
                        if !completed && any_overlap(&data, &defs) {
                            return Step::Found;
                        }
                    }
                }
            }
            let full_wait = g.wait.is_some_and(|w| w.raw_waitcnt == Some(0));
            if g.class == Class::Valu
                || g.class == Class::Export
                || full_wait
                || g.depctr.is_some_and(|d| (d >> 2) & 0x7 == 0)
            {
                return Step::Expired;
            }
            if let Some(w) = g.wait {
                st.lgkm0 |= w.lgkm0;
                st.vscnt0 |= w.vscnt0;
            }
            Step::Next
        });
        for h in hits {
            let producer = &s.facts[h.producer];
            let data = data_vgprs(s.insts[h.producer], s.arch);
            let overlap: Vec<RegRef> = defs
                .iter()
                .copied()
                .filter(|d| data.iter().any(|x| x.overlaps(*d)))
                .collect();
            let wait = if producer.class == Class::Ds {
                "s_waitcnt lgkmcnt(0)"
            } else {
                "s_waitcnt_vscnt null, 0"
            };
            out.push(s.finding(
                "adv-vgpr-war-async-write",
                Severity::Advisory,
                i,
                Some(h.producer),
                format!("(no vendor rule on gfx11) conservative: a VALU/EXP, depctr vm_vsrc(0), s_waitcnt 0 or {wait} between"),
                format!("{} instruction(s) between, no VALU/EXP/vm_vsrc(0)/full wait/{wait}", h.dist),
                format!(
                    "{} is a store-data source of the earlier {} and the destination of this load; if the store's source read is not complete when the load returns, the store would carry the load's data",
                    regs_text(&overlap),
                    if producer.class == Class::Ds { "DS" } else { "VMEM" }
                ),
            ));
        }
    }
}

/// `allocateSystemSGPRs` / `FeatureUserSGPRInit16Bug`: user + system SGPRs must be >= 16.
/// `rsrc2` is `COMPUTE_PGM_RSRC2` (bits 1..5 USER_SGPR_COUNT, 7/8/9/10 WORKGROUP_ID_X/Y/Z/INFO).
pub fn user_sgpr_init16(rsrc2: u32) -> (u32, u32, bool) {
    let user = (rsrc2 >> 1) & 0x1f;
    let system = (rsrc2 >> 7 & 1) + (rsrc2 >> 8 & 1) + (rsrc2 >> 9 & 1) + (rsrc2 >> 10 & 1);
    (user, system, user + system < 16)
}

fn rule_user_sgpr(
    symbol: &str,
    arch: Arch,
    wave: Wave,
    rsrc2: Option<u32>,
    out: &mut Vec<Finding>,
) {
    let Some(rsrc2) = rsrc2 else {
        return;
    };
    if arch != Arch::Gfx1100 || wave != Wave::Wave32 {
        return;
    }
    let (user, system, bad) = user_sgpr_init16(rsrc2);
    if bad {
        out.push(Finding {
            rule: "gfx1100-user-sgpr-init16",
            severity: Severity::Required,
            kernel: symbol.to_owned(),
            pc: 0,
            insts: Vec::new(),
            required: "USER_SGPR_COUNT + enabled workgroup-id/info SGPRs >= 16".into(),
            present: format!(
                "user {user} + system {system} = {} (compute_pgm_rsrc2 = {rsrc2:#x}); private-segment wave offset not counted, as LLVM",
                user + system
            ),
            detail: "FeatureUserSGPRInit16Bug (gfx1100 = gfx11.0.0): hardware requires >= 16 initialised user+system SGPRs in wave32; SITargetLowering::allocateSystemSGPRs pads to 16".into(),
            context: Vec::new(),
        });
    }
}

fn rule_wave64(symbol: &str, arch: Arch, wave: Wave, out: &mut Vec<Finding>) {
    if wave == Wave::Wave64 && matches!(arch, Arch::Gfx1100 | Arch::Gfx1151) {
        for rule in [
            "gfx11-wave64-valu-partial-forwarding",
            "gfx11-wave64-valu-mask-write",
        ] {
            out.push(Finding {
                rule,
                severity: Severity::Unchecked,
                kernel: symbol.to_owned(),
                pc: 0,
                insts: Vec::new(),
                required: "wave64 kernel: LLVM fixVALUPartialForwardingHazard / fixVALUMaskWriteHazard apply".into(),
                present: "rule not implemented in this checker".into(),
                detail: "kernel is wave64; the checker fails closed instead of reporting it clean".into(),
                context: Vec::new(),
            });
        }
    }
}

/// gfx1201 `AMDGPUWaitSGPRHazards`: every demanded `s_wait_alu` that no reaching path
/// carries is a Required finding. The replay is PM's port in its LLVM-exact model (VCC as a
/// tracked pair, 64-bit lane-mask pairs in wave64, FLAT does not retire writes, `IsSetPC`
/// flushes), applied to wave32 too, so the checker never inherits the pipeline's historic
/// wave32 builder-parity boundary. A replay that cannot run (no CFG, no fixpoint, arch
/// outside the tables) is Unchecked: the checker fails closed.
fn rule_gfx12_sgpr(s: &Stream, body: &Body, arch: Arch, wave: Wave, out: &mut Vec<Finding>) {
    match pm_hazards::analyze_with(body, arch, wave, pm_hazards::SgprModel::LlvmExact) {
        Err(e) => out.push(Finding {
            rule: "gfx12-valu-read-sgpr",
            severity: Severity::Unchecked,
            kernel: s.symbol.to_owned(),
            pc: 0,
            insts: Vec::new(),
            required: "AMDGPUWaitSGPRHazards replay over the kernel CFG".into(),
            present: format!("replay failed: {e}"),
            detail: "the SGPR rule applies to every gfx1201 kernel; failing closed".into(),
            context: Vec::new(),
        }),
        Ok(found) => {
            for missing in &found.missing {
                let Some(i) = body.layout.iter().position(|id| *id == missing.inst) else {
                    continue;
                };
                out.push(s.finding(
                    "gfx12-valu-read-sgpr",
                    Severity::Required,
                    i,
                    None,
                    format!("s_wait_alu {} before this instruction", missing.wait),
                    "no s_wait_alu on some reaching path retires the pending SGPR/VCC write".into(),
                    "AMDGPUWaitSGPRHazards::runOnMachineBasicBlock (WA_SALU sa_sdst / WA_VALU va_sdst / WA_VCC va_vcc)".into(),
                ));
            }
        }
    }
}

/// Check one instruction body. `rsrc2` is the kernel descriptor's `COMPUTE_PGM_RSRC2` when
/// known. `advisory` enables the non-required rules.
pub fn check_body(
    symbol: &str,
    body: &Body,
    arch: Arch,
    wave: Wave,
    rsrc2: Option<u32>,
    advisory: bool,
) -> Vec<Finding> {
    let mut out = Vec::new();
    let stream = Stream::new(symbol, body, arch);
    rule_user_sgpr(symbol, arch, wave, rsrc2, &mut out);
    rule_wave64(symbol, arch, wave, &mut out);
    if arch == Arch::Gfx1201 {
        rule_gfx12_sgpr(&stream, body, arch, wave, &mut out);
    }
    rule_vcmpx_permlane(&stream, &mut out);
    rule_wmma_chain(&stream, &mut out);
    if arch == Arch::Gfx1100 {
        rule_trans_use(&stream, &mut out);
        rule_mad64(&stream, &mut out);
    }
    if advisory && matches!(arch, Arch::Gfx1100 | Arch::Gfx1151) {
        rule_vmem_to_scalar_write(&stream, &mut out);
        rule_vgpr_war_async(&stream, &mut out);
    }
    out.retain(|f| advisory || f.severity != Severity::Advisory);
    out.sort_by_key(|f| (f.severity, f.pc, f.rule));
    out
}

/// Check one lifted kernel.
pub fn check_kernel(kernel: &Kernel, arch: Arch, advisory: bool) -> Vec<Finding> {
    let rsrc2 = match &kernel.abi {
        Abi::Hsa { descriptor, .. } => Some(descriptor.compute_pgm_rsrc2.0),
        Abi::Raw(_) => None,
    };
    check_body(
        &kernel.symbol.0,
        &kernel.body,
        arch,
        kernel.wave,
        rsrc2,
        advisory,
    )
}

/// Check every kernel of a lifted program.
pub fn check_program(program: &Program, advisory: bool) -> Vec<Finding> {
    program
        .kernels
        .iter()
        .flat_map(|k| check_kernel(k, program.target.arch, advisory))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use peacemaker_ir::codec::gfx11;
    use peacemaker_ir::passes::cfg::build_blocks;
    use peacemaker_ir::provenance::Provenance;

    const EXP: &[u32] = &[0x7e024b00]; // v_exp_f32 v1, v0
    const ADD_V1: &[u32] = &[0x6040301]; // v_add_f32 v2, v1, v1
    const ADD_V7: &[u32] = &[0x6100f07]; // v_add_f32 v8, v7, v7
    const MOV_V9: &[u32] = &[0x7e120309]; // v_mov_b32 v9, v9
    const VNOP: &[u32] = &[0x7e000000]; // v_nop
    const DEPCTR_VA_VDST0: &[u32] = &[0xbf880f9f]; // s_waitcnt_depctr depctr_va_vdst(0)
    const DEPCTR_VM_VSRC0: &[u32] = &[0xbf88ff83]; // s_waitcnt_depctr depctr_vm_vsrc(0)
    const CMPX: &[u32] = &[0x7d980300]; // v_cmpx_gt_u32_e32 v0, v1
    const PERM: &[u32] = &[0xd65b0002, 0x40103]; // v_permlane16_b32 v2, v3, s0, s1
    const WMMA_A: &[u32] = &[0xcc404010, 0x1c421100]; // v_wmma_f32_16x16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23]
    const WMMA_B_OVL: &[u32] = &[0xcc404018, 0x1c621110]; // v_wmma_f32_16x16x16_f16 v[24:31], v[16:23], v[8:15], v[24:31]
    const WMMA_B_OK: &[u32] = &[0xcc404018, 0x1c621100]; // v_wmma_f32_16x16x16_f16 v[24:31], v[0:7], v[8:15], v[24:31]
    const MAD_BAD: &[u32] = &[0xd6fe7c00, 0x400040f]; // v_mad_u64_u32 v[0:1], null, s15, s2, v[0:1]
    const MAD_BAD_PARTIAL: &[u32] = &[0xd6fe7c01, 0x400040f]; // v_mad_u64_u32 v[1:2], null, s15, s2, v[0:1]
    const MAD_OK: &[u32] = &[0xd6fe7c04, 0x400040f]; // v_mad_u64_u32 v[4:5], null, s15, s2, v[0:1]
    const MAD_I_BAD_SRC0: &[u32] = &[0xd6ff7c04, 0x4000504]; // v_mad_i64_i32 v[4:5], null, v4, s2, v[0:1]
    const BUF_LOAD_SRD: &[u32] = &[0xe0540000, 0x80460201]; // buffer_load_b64 v[2:3], v1, s[24:27], 0 offen
    const BUF_LOAD_SRD_SOFF: &[u32] = &[0xe0540000, 0x19420201]; // buffer_load_b64 v[2:3], v1, s[8:11], s25 offen
    const S_MOV_S25: &[u32] = &[0xbe990080]; // s_mov_b32 s25, 0
    const S_MOV_S30: &[u32] = &[0xbe9e0080]; // s_mov_b32 s30, 0
    const G_STORE: &[u32] = &[0xdc6a0000, 0x7c0200]; // global_store_b32 v[0:1], v2, off
    const S_OR_EXEC: &[u32] = &[0x8c7e027e]; // s_or_b32 exec_lo, exec_lo, s2
    const DS_STORE128: &[u32] = &[0xdb7c1820, 0x4852]; // ds_store_b128 v82, v[72:75] offset:6176
    const DS_STORE128B: &[u32] = &[0xdb7c182c, 0x4c52]; // ds_store_b128 v82, v[76:79] offset:6188
    const BUF_LOAD72: &[u32] = &[0xe05c0000, 0x80434850]; // buffer_load_b128 v[72:75], v80, s[12:15], 0 offen
    const BUF_LOAD76: &[u32] = &[0xe05c0000, 0x80434c51]; // buffer_load_b128 v[76:79], v81, s[12:15], 0 offen
    const LGKM0: &[u32] = &[0xbf89fc07]; // s_waitcnt lgkmcnt(0)
    const WAIT0: &[u32] = &[0xbf890000]; // s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)
    const VSCNT0: &[u32] = &[0xbc7c0000]; // s_waitcnt_vscnt null, 0x0
    const S_NOP: &[u32] = &[0xbf800000]; // s_nop 0
    const ENDPGM: &[u32] = &[0xbfb00000]; // s_endpgm
    const CBR1: &[u32] = &[0xbfa20001]; // s_cbranch_scc1 +1
    const CBR_BACK3: &[u32] = &[0xbfa2fffd]; // s_cbranch_scc1 -3

    fn body_of(seq: &[&[u32]]) -> Body {
        let mut body = Body::default();
        let mut pc = 0u32;
        for words in seq.iter().chain(std::iter::once(&ENDPGM)) {
            let (mut inst, n) = gfx11::decode(Arch::Gfx1100, words).expect("decode");
            assert_eq!(n, words.len());
            inst.prov = Provenance {
                pc: Some(pc),
                ..Provenance::default()
            };
            pc += 4 * n as u32;
            let id = body.insts.insert(inst);
            body.layout.push(id);
        }
        build_blocks(&mut body, Arch::Gfx1100).expect("cfg");
        body
    }

    fn findings(seq: &[&[u32]], advisory: bool) -> Vec<Finding> {
        check_body(
            "k",
            &body_of(seq),
            Arch::Gfx1100,
            Wave::Wave32,
            None,
            advisory,
        )
    }

    fn rules(seq: &[&[u32]]) -> Vec<(&'static str, u32)> {
        findings(seq, true)
            .into_iter()
            .map(|f| (f.rule, f.pc))
            .collect()
    }

    fn has(seq: &[&[u32]], rule: &str) -> bool {
        rules(seq).iter().any(|(r, _)| *r == rule)
    }

    #[test]
    fn trans_use_positive_and_negative() {
        // v_exp_f32 v1; v_add_f32 v2, v1, v1
        assert!(has(&[EXP, ADD_V1], "gfx1100-valu-trans-use"));
        // up to five VALUs between still hazardous
        assert!(has(
            &[EXP, MOV_V9, MOV_V9, MOV_V9, MOV_V9, MOV_V9, ADD_V1],
            "gfx1100-valu-trans-use"
        ));
        // six VALUs expire it
        assert!(!has(
            &[EXP, MOV_V9, MOV_V9, MOV_V9, MOV_V9, MOV_V9, MOV_V9, ADD_V1],
            "gfx1100-valu-trans-use"
        ));
        // depctr va_vdst(0) mitigates; vm_vsrc(0) does not
        assert!(!has(
            &[EXP, DEPCTR_VA_VDST0, ADD_V1],
            "gfx1100-valu-trans-use"
        ));
        assert!(has(
            &[EXP, DEPCTR_VM_VSRC0, ADD_V1],
            "gfx1100-valu-trans-use"
        ));
        // a DS/VMEM between expires it
        assert!(!has(&[EXP, DS_STORE128, ADD_V1], "gfx1100-valu-trans-use"));
        // unrelated consumer register
        assert!(!has(&[EXP, ADD_V7], "gfx1100-valu-trans-use"));
        // a second TRANS between expires it (intervening TRANS > 1 needs two)
        assert!(has(&[EXP, EXP, ADD_V1], "gfx1100-valu-trans-use"));
    }

    #[test]
    fn trans_use_across_branch_join() {
        // The taken path of the branch skips the v_exp; the fall-through path runs it and
        // reaches the consumer: a hazard on any path is a finding.
        assert!(has(&[CBR1, EXP, ADD_V1], "gfx1100-valu-trans-use"));
        // The depctr wait is skipped by the taken path: still a finding.
        assert!(has(
            &[EXP, CBR1, DEPCTR_VA_VDST0, ADD_V1],
            "gfx1100-valu-trans-use"
        ));
        // Both arms of the diamond pass through the depctr: clean.
        assert!(!has(
            &[EXP, CBR1, MOV_V9, DEPCTR_VA_VDST0, ADD_V1],
            "gfx1100-valu-trans-use"
        ));
    }

    #[test]
    fn trans_use_loop_back_edge() {
        // consumer at the loop top reads the TRANS result of the previous iteration
        // 0: add v2,v1,v1 ; 1: exp v1,v0 ; 2: s_cbranch_scc1 -3 (to 0)
        assert!(has(&[ADD_V1, EXP, CBR_BACK3], "gfx1100-valu-trans-use"));
    }

    #[test]
    fn vcmpx_permlane() {
        assert!(has(&[CMPX, PERM], "gfx11-vcmpx-permlane"));
        // v_nop is discarded by SQ and does not mitigate
        assert!(has(&[CMPX, VNOP, PERM], "gfx11-vcmpx-permlane"));
        // a real VALU does
        assert!(!has(&[CMPX, MOV_V9, PERM], "gfx11-vcmpx-permlane"));
        // SALU / s_nop in between do not
        assert!(has(&[CMPX, S_NOP, PERM], "gfx11-vcmpx-permlane"));
        // no cmpx: clean
        assert!(!has(&[MOV_V9, PERM], "gfx11-vcmpx-permlane"));
    }

    #[test]
    fn wmma_chain() {
        assert!(has(&[WMMA_A, WMMA_B_OVL], "gfx11-wmma-chain"));
        assert!(!has(&[WMMA_A, WMMA_B_OK], "gfx11-wmma-chain"));
        // v_nop (any VALU) between mitigates
        assert!(!has(&[WMMA_A, VNOP, WMMA_B_OVL], "gfx11-wmma-chain"));
        // s_nop does not
        assert!(has(&[WMMA_A, S_NOP, WMMA_B_OVL], "gfx11-wmma-chain"));
    }

    #[test]
    fn mad64_earlyclobber() {
        assert!(has(&[MAD_BAD], "gfx1100-mad64-intra-fwd"));
        assert!(has(&[MAD_I_BAD_SRC0], "gfx1100-mad64-intra-fwd"));
        assert!(!has(&[MAD_OK], "gfx1100-mad64-intra-fwd"));
    }

    #[test]
    fn mad64_dst_low_over_src2_high_shape_is_advisory() {
        // dst v[1:2] over src2 v[0:1] touches only src2's high half (v1): the shape hipcc emits.
        let f = findings(&[MAD_BAD_PARTIAL], true);
        assert!(f.iter().any(|f| f.rule == "adv-mad64-overlap-dead-hi"));
        assert!(!f.iter().any(|f| f.rule == "gfx1100-mad64-intra-fwd"));
        // full overlap (src2 low half) is never that shape
        let f = findings(&[MAD_BAD], true);
        assert!(f.iter().any(|f| f.rule == "gfx1100-mad64-intra-fwd"));
        // advisory shape is dropped with the other advisory rules
        assert!(findings(&[MAD_BAD_PARTIAL], false).is_empty());
    }

    #[test]
    fn user_sgpr_init16_descriptor_rule() {
        // R3Mq6Wp-02: rsrc2 = 0x19c: 14 user + X + Y = 16 -> fine
        assert_eq!(user_sgpr_init16(0x19c), (14, 2, false));
        // 6 user SGPRs + X: 7 < 16
        assert_eq!(user_sgpr_init16((6 << 1) | (1 << 7)), (6, 1, true));
        assert_eq!(user_sgpr_init16((15 << 1) | (1 << 7)), (15, 1, false));
        assert_eq!(user_sgpr_init16(15 << 1), (15, 0, true));
        let body = body_of(&[MOV_V9]);
        let run = |wave, rsrc2| check_body("k", &body, Arch::Gfx1100, wave, Some(rsrc2), false);
        assert!(run(Wave::Wave32, (6 << 1) | (1 << 7))
            .iter()
            .any(|f| f.rule == "gfx1100-user-sgpr-init16" && f.severity == Severity::Required));
        assert!(run(Wave::Wave32, 0x19c).is_empty());
        // wave64 is exempt from this rule but fails closed on the wave64 rules
        let f = run(Wave::Wave64, (6 << 1) | (1 << 7));
        assert!(!f.iter().any(|f| f.rule == "gfx1100-user-sgpr-init16"));
        assert!(f.iter().any(|f| f.severity == Severity::Unchecked));
    }

    #[test]
    fn vmem_to_scalar_write_advisory() {
        // SRD s[24:27] read by the load, s25 rewritten with no VALU in between
        assert!(has(&[BUF_LOAD_SRD, S_MOV_S25], "adv-vmem-to-scalar-write"));
        // one VALU in between expires it (gfx10.1 rule)
        assert!(!has(
            &[BUF_LOAD_SRD, MOV_V9, S_MOV_S25],
            "adv-vmem-to-scalar-write"
        ));
        // soffset s25 read, rewritten
        assert!(has(
            &[BUF_LOAD_SRD_SOFF, S_MOV_S25],
            "adv-vmem-to-scalar-write"
        ));
        // unrelated SGPR
        assert!(!has(&[BUF_LOAD_SRD, S_MOV_S30], "adv-vmem-to-scalar-write"));
        // EXEC rewrite right after a store
        assert!(has(&[G_STORE, S_OR_EXEC], "adv-vmem-to-scalar-write"));
        assert!(!has(
            &[G_STORE, MOV_V9, S_OR_EXEC],
            "adv-vmem-to-scalar-write"
        ));
        // full wait mitigates
        assert!(!has(
            &[BUF_LOAD_SRD, WAIT0, S_MOV_S25],
            "adv-vmem-to-scalar-write"
        ));
        // vm_vsrc(0) mitigates
        assert!(!has(
            &[BUF_LOAD_SRD, DEPCTR_VM_VSRC0, S_MOV_S25],
            "adv-vmem-to-scalar-write"
        ));
        // advisory findings carry the advisory severity
        let f = findings(&[BUF_LOAD_SRD, S_MOV_S25], true);
        assert!(!f.is_empty() && f.iter().all(|f| f.severity == Severity::Advisory));
        // and are omitted when advisory rules are off
        assert!(findings(&[BUF_LOAD_SRD, S_MOV_S25], false).is_empty());
    }

    #[test]
    fn vgpr_war_async_advisory() {
        // ds_store reads v[72:75]; buffer_load rewrites v[72:75] with nothing in between
        assert!(has(&[DS_STORE128, BUF_LOAD72], "adv-vgpr-war-async-write"));
        // two stores then two loads (the lag stub shape)
        assert!(has(
            &[DS_STORE128, DS_STORE128B, BUF_LOAD72, BUF_LOAD76],
            "adv-vgpr-war-async-write"
        ));
        // any VALU between, a full wait, or the DS store's lgkmcnt(0) satisfies it
        assert!(!has(
            &[DS_STORE128, MOV_V9, BUF_LOAD72],
            "adv-vgpr-war-async-write"
        ));
        assert!(!has(
            &[DS_STORE128, LGKM0, BUF_LOAD72],
            "adv-vgpr-war-async-write"
        ));
        assert!(!has(
            &[DS_STORE128, WAIT0, BUF_LOAD72],
            "adv-vgpr-war-async-write"
        ));
        // vscnt(0) does not satisfy a DS source
        assert!(has(
            &[DS_STORE128, VSCNT0, BUF_LOAD72],
            "adv-vgpr-war-async-write"
        ));
        // different registers: clean
        assert!(!has(&[DS_STORE128, BUF_LOAD76], "adv-vgpr-war-async-write"));
    }

    #[test]
    fn finding_json_names_the_instructions() {
        let f = findings(&[EXP, ADD_V1], false);
        assert_eq!(f.len(), 1);
        let j = f[0].to_json();
        assert_eq!(j["rule"], "gfx1100-valu-trans-use");
        assert_eq!(j["pc"], "0x4");
        assert_eq!(j["instructions"][0]["role"], "producer");
        assert_eq!(j["instructions"][0]["pc"], "0x0");
    }

    #[test]
    fn rule_table_ids_are_unique() {
        let mut ids: Vec<_> = RULES.iter().map(|r| r.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), RULES.len());
    }

    // gfx1201 encodings (llvm-mc -mcpu=gfx1201 -mattr=+wavefrontsize64 / +wavefrontsize32).
    const G_CNDMASK_VCC: &[u32] = &[0x0204_0903]; // v_cndmask_b32_e32 v2, v3, v4, vcc
    const G_CMP_VCC: &[u32] = &[0x7c82_0300]; // v_cmp_lt_i32_e32 vcc, v0, v1
    const G_CBR_VCCZ_1: &[u32] = &[0xbfa3_0001]; // s_cbranch_vccz +1
    const G_SAVEEXEC_VCC: &[u32] = &[0xbe82_216a]; // s_and_saveexec_b64 s[2:3], vcc
    const G_CNDMASK_S23_A: &[u32] = &[0xd501_0000, 0x000a_0501]; // v_cndmask_b32_e64 v0, v1, v2, s[2:3]
    const G_CMP64_S23: &[u32] = &[0xd441_0002, 0x0202_0300]; // v_cmp_lt_i32_e64 s[2:3], v0, v1
    const G_CNDMASK_S23_B: &[u32] = &[0xd501_0003, 0x000a_0501]; // v_cndmask_b32_e64 v3, v1, v2, s[2:3]
    const G_WAIT_VA_SDST0: &[u32] = &[0xbf88_f19f]; // s_wait_alu depctr_va_sdst(0)
    const G_AND_VCC_EXEC_S45: &[u32] = &[0x8bea_047e]; // s_and_b64 vcc, exec, s[4:5]
    const G_AND_VCCLO_EXECLO_S4: &[u32] = &[0x8b6a_047e]; // s_and_b32 vcc_lo, exec_lo, s4
    const G_WAIT_SA_SDST0: &[u32] = &[0xbf88_ff9e]; // s_wait_alu depctr_sa_sdst(0)

    fn gfx12_body(seq: &[&[u32]]) -> Body {
        let mut body = Body::default();
        let mut pc = 0u32;
        for words in seq.iter().chain(std::iter::once(&ENDPGM)) {
            let (mut inst, n) =
                peacemaker_ir::codec::gfx12::decode_for(Arch::Gfx1201, words).expect("decode");
            assert_eq!(n, words.len());
            inst.prov = Provenance {
                pc: Some(pc),
                ..Provenance::default()
            };
            pc += 4 * n as u32;
            let id = body.insts.insert(inst);
            body.layout.push(id);
        }
        build_blocks(&mut body, Arch::Gfx1201).expect("cfg");
        body
    }

    fn gfx12_findings(wave: Wave, seq: &[&[u32]]) -> Vec<Finding> {
        check_body("k", &gfx12_body(seq), Arch::Gfx1201, wave, None, true)
    }

    /// A wave64 gfx1201 kernel with VCC pair / saveexec / lane-mask traffic that carries every
    /// guard LLVM demands is clean: no Required, Unchecked or Advisory finding at all.
    #[test]
    fn gfx1201_wave64_guarded_kernel_is_clean() {
        let seq: &[&[u32]] = &[
            G_CNDMASK_VCC,
            G_CMP_VCC,
            G_CBR_VCCZ_1,
            G_SAVEEXEC_VCC,
            G_CNDMASK_S23_A,
            G_CMP64_S23,
            G_WAIT_VA_SDST0,
            G_CNDMASK_S23_B,
        ];
        let f = gfx12_findings(Wave::Wave64, seq);
        assert!(f.is_empty(), "{f:?}");
    }

    /// The synthetic wave64 violation: a VALU-written lane-mask pair read by a VALU with no
    /// `s_wait_alu depctr_va_sdst(0)` is a Required `gfx12-valu-read-sgpr` finding at the reader.
    #[test]
    fn gfx1201_wave64_missing_va_sdst_is_required() {
        let f = gfx12_findings(
            Wave::Wave64,
            &[G_CNDMASK_S23_A, G_CMP64_S23, G_CNDMASK_S23_B],
        );
        assert_eq!(f.len(), 1, "{f:?}");
        assert_eq!(f[0].rule, "gfx12-valu-read-sgpr");
        assert_eq!(f[0].severity, Severity::Required);
        assert_eq!(f[0].pc, 16, "third instruction: 8 + 8 bytes in");
        assert!(f[0].required.contains("depctr_va_sdst(0)"), "{:?}", f[0]);
        assert!(f[0].insts.iter().any(|i| i.text.contains("s[2:3]")));
        // The guard on the path removes it, a guard of the wrong kind does not.
        assert!(gfx12_findings(
            Wave::Wave64,
            &[
                G_CNDMASK_S23_A,
                G_CMP64_S23,
                G_WAIT_VA_SDST0,
                G_CNDMASK_S23_B
            ],
        )
        .is_empty());
        assert_eq!(
            gfx12_findings(
                Wave::Wave64,
                &[
                    G_CNDMASK_S23_A,
                    G_CMP64_S23,
                    G_WAIT_SA_SDST0,
                    G_CNDMASK_S23_B
                ],
            )
            .len(),
            1
        );
    }

    /// The checker runs the LLVM-exact VCC model in wave32 too, which the pipeline-default
    /// `analyze` (builder parity) does not: a SALU `vcc_lo` write reread by a VALU needs
    /// `sa_sdst(0)`.
    #[test]
    fn gfx1201_wave32_checker_tracks_vcc_like_llvm() {
        let seq: &[&[u32]] = &[G_CNDMASK_VCC, G_AND_VCCLO_EXECLO_S4, G_CNDMASK_VCC];
        let f = gfx12_findings(Wave::Wave32, seq);
        assert_eq!(f.len(), 1, "{f:?}");
        assert!(f[0].required.contains("depctr_sa_sdst(0)"));
        let body = gfx12_body(seq);
        let default =
            peacemaker_ir::passes::hazards::analyze(&body, Arch::Gfx1201, Wave::Wave32).unwrap();
        assert!(default.missing.is_empty(), "{default:?}");
        assert!(gfx12_findings(
            Wave::Wave32,
            &[
                G_CNDMASK_VCC,
                G_AND_VCCLO_EXECLO_S4,
                G_WAIT_SA_SDST0,
                G_CNDMASK_VCC
            ],
        )
        .is_empty());
        // wave64 spelling of the same hazard.
        let f = gfx12_findings(
            Wave::Wave64,
            &[G_CNDMASK_VCC, G_AND_VCC_EXEC_S45, G_CNDMASK_VCC],
        );
        assert_eq!(f.len(), 1, "{f:?}");
    }

    #[test]
    fn gfx12_sgpr_rule_is_implemented_and_documented() {
        let r = RULES
            .iter()
            .find(|r| r.id == "gfx12-valu-read-sgpr")
            .unwrap();
        assert!(r.implemented && r.severity == Severity::Required);
        assert!(RULES.iter().any(|r| r.id == "n/a-gfx12-no-data-dep-hazard"));
        assert!(RULES.iter().any(|r| r.id == "n/a-gfx12-trans-use"));
    }
}
