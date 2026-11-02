// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! C5: per-arch SGPR, VALU, and WMMA hazards.
//!
//! Whole-program hazard replay over typed streams. The gfx12 SGPR tracker
//! ports `peacemaker-isa/src/hazard.rs`; gfx11 rules are grounded in ROCm LLVM
//! `GCNHazardRecognizer.cpp` / `AMDGPU.td` at compiler commit
//! `8f497e0992fb7513f7f78a6f6b6f1056c375e961`:
//! * `gfx12_sgpr`: SALU-def/VALU-use SGPR tracking. The state machine is
//!   `hazard.rs` verbatim (memory clears, tracked-pair marks, demand
//!   reporting with self-clear). Where the builder *emits* the demanded
//!   `s_wait_alu`, the whole-program replay *verifies* it. Actual decoded
//!   guards retire pending writes on their reaching paths. Block inputs
//!   join every predecessor and loop back edge at a fixed point; a guard
//!   on another arm, or before a newer write, cannot satisfy a demand.
//! * `gfx11_wave32`: gfx1100 TRANS→VALU forwarding (`s_waitcnt_depctr
//!   va_vdst(0)`), both gfx11 targets' VCMPX→PERMLANE interlock, and WMMA
//!   chaining, using LLVM `AMDGPU.td` target feature sets and
//!   `GCNHazardRecognizer.cpp` (`fixVALUTransUseHazard`,
//!   `fixVcmpxPermlaneHazards`, `fixWMMAHazards`). The gfx11 feature sets
//!   include `NoDataDepHazard`: the older VMEM/VALU-SGPR NOP rule does not
//!   apply, and gfx1151 does not have `VALUTransUseHazard`.
//! * WMMA chaining: a `v_wmma_*`/`v_swmmac_*` whose A/B (or SWMMAC index)
//!   operands overlap the previous matrix instruction's destination needs an
//!   independent VALU or `v_nop` between them. Replay joins all predecessor
//!   states at CFG boundaries, including loop backedges.
//!
//! Depctr field positions are pinned by `llvm-mc -mcpu=gfx1201`: `sa_sdst`
//! is bit 0, `va_vcc` is bit 1, `va_sdst` is bits [11:9] of the `s_wait_alu`
//! simm16.
//!
//! gfx1201 wave64 (`AMDGPUWaitSGPRHazards.cpp`, gate
//! `hasVALUReadSGPRHazard() = HasGFX12Insts && !HasGFX1250Insts`; the pass has
//! no other wave-size dependence than `DsNopCount`, 4 vs 8, which only
//! matters for `ds_nop` hazard culls that the hipcc incumbents never carry):
//! * lane masks are SGPR *pairs*: explicit mask slots of `v_cmp*`,
//!   `v_cndmask`, `v_*_co_*`, `v_*_co_ci_*`, `v_div_scale` are two SGPRs
//!   wide (`codec::gfx12::operand_register`); `s_*_b64` mask operands are
//!   genuine 64-bit SGPR pairs; EXEC is never tracked (`sgprNumber`), VCC is
//!   tracked as the one pair `vcc_lo/vcc_hi` (s106/s107 → pair 53) with its
//!   own SALU/VALU source (`VCCHazard`, waited by `sa_sdst` / `va_vcc`).
//! * implicit VCC accesses (`HasImplicitVCC`: `v_cndmask_*_e32`,
//!   `v_*_co_ci_*_e32`, `v_cmp_*_e32`, `v_div_fmas`, `s_cbranch_vcc*`) and
//!   explicit `vcc`/`vcc_lo`/`vcc_hi` operands are replayed; a VALU write
//!   reread by a SALU needs nothing (the SALU read clears it), a SALU write
//!   reread by anything needs `sa_sdst(0)`, a VALU write reread by a VALU
//!   needs `va_sdst(0)` (`va_vcc(0)` for VCC).
//! * `SIInstrInfo::isVMEM && !isFLAT || isSMRD` clear the state; FLAT
//!   encodings (`global_*`, `scratch_*`, `flat_*`) are `isFLAT` and do NOT
//!   clear it, and `s_sendmsg_rtn_*` is SALU, not SMRD. `s_setpc_b64` /
//!   `s_swappc_b64` (`IsSetPC`) must flush every pending write.
//! * wave32 keeps the historic builder-parity model (VCC untracked, every
//!   VMEM/FLAT/SMEM clears) so wave32 results are unchanged.
//! Deliberate over-approximation shared with the wave32 port: a wide SGPR
//! tuple marks every pair it spans tracked, LLVM marks only the base pair.

use crate::cfg::{Body, InstId};
use crate::effects::{Control, MemClass};
use crate::inst::{Arch, Inst, Wave};
use crate::operand::{ImmField, Operand};
use crate::reg::{Kind, RegRef};
use crate::state::{Obligation, ObligationKind};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum HazardError {
    #[error(
        "hazard tables cover gfx1100/gfx1151 (gfx11 rules) and gfx1201 (SGPR model, wave32 and wave64) only (got {0:?} {1:?})"
    )]
    Unsupported(Arch, Wave),
    #[error("SGPR hazard replay needs a built CFG")]
    MissingCfg,
    #[error("SGPR hazard replay did not converge in {walks} CFG sweeps")]
    NoFixpoint { walks: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pipeline {
    Salu,
    Valu,
    Vmem,
    Smem,
    Ds,
}

fn pipe_of(inst: &Inst, arch: Arch) -> Pipeline {
    match &inst.effects.mem {
        Some(mem) => match mem.class {
            MemClass::VmemLoad
            | MemClass::VmemStore
            | MemClass::VmemAtomic { .. }
            | MemClass::FlatLoad
            | MemClass::FlatStore
            | MemClass::FlatAtomic { .. } => Pipeline::Vmem,
            MemClass::SmemLoad => Pipeline::Smem,
            _ => pipe_by_name(inst, arch),
        },
        None => pipe_by_name(inst, arch),
    }
}

fn pipe_by_name(inst: &Inst, arch: Arch) -> Pipeline {
    match inst.op.name(arch).unwrap_or("") {
        name if name.starts_with('s') => Pipeline::Salu,
        name if name.starts_with('v') => Pipeline::Valu,
        _ => Pipeline::Ds,
    }
}

/// `AMDGPUWaitSGPRHazards::runOnMachineBasicBlock` instruction classes:
/// `isVMEM && !isFLAT` and `isSMRD` clear the state, FLAT-encoded accesses
/// (`global_*`, `scratch_*`, `flat_*`) are ignored, `s_sendmsg_rtn_*` is SOP1.
fn llvm_pipe(inst: &Inst, arch: Arch) -> Pipeline {
    let name = inst.op.name(arch).unwrap_or("");
    // Every SMEM-encoded instruction is `isSMRD` in LLVM, including the effect-less hints (`s_prefetch_data`): it
    // clears the pending state instead of consuming an SGPR (it is not a SALU/VALU in the pass's classification).
    if inst.form == crate::inst::Form::Smem && !name.starts_with("s_sendmsg_rtn") {
        return Pipeline::Smem;
    }
    match &inst.effects.mem {
        Some(mem) => match mem.class {
            MemClass::SmemLoad if name.starts_with("s_sendmsg_rtn") => Pipeline::Salu,
            MemClass::SmemLoad | MemClass::SmemStore => Pipeline::Smem,
            MemClass::FlatLoad | MemClass::FlatStore | MemClass::FlatAtomic { .. } => Pipeline::Ds,
            MemClass::VmemLoad | MemClass::VmemStore | MemClass::VmemAtomic { .. }
                if name.starts_with("global_") || name.starts_with("scratch_") =>
            {
                Pipeline::Ds
            }
            MemClass::VmemLoad | MemClass::VmemStore | MemClass::VmemAtomic { .. } => {
                Pipeline::Vmem
            }
            _ => pipe_by_name(inst, arch),
        },
        None => pipe_by_name(inst, arch),
    }
}

/// VCC reads/writes the way `AMDGPUWaitSGPRHazards` sees them: explicit VCC
/// operands plus the instruction descriptor's implicit VCC. The ISA table's
/// implicit column omits the VOP2/VOPC `_e32` VCC slot, so name it here.
fn vcc_access(inst: &Inst, name: &str) -> (bool, bool) {
    use crate::effects::ImplicitSet;
    let e32 = name.ends_with("_e32");
    let used = inst.effects.implicit.reads & ImplicitSet::VCC != 0
        || e32 && (name.starts_with("v_cndmask_") || name.contains("_co_ci_"));
    let defined = inst.effects.implicit.writes & ImplicitSet::VCC != 0
        || e32 && (name.starts_with("v_cmp_") || name.contains("_co_ci_"));
    (used, defined)
}

/// `vcc_lo`/`vcc_hi` are s106/s107: pair 53 in `AMDGPUWaitSGPRHazards::Tracked`.
const VCC_PAIR: usize = 53;

/// Port of `peacemaker-isa` `Gfx12Sgpr`: tracks SGPR pairs read by VALU and
/// written by SALU/VALU, demanding `depctr_*_sdst(0)` / `depctr_va_vcc(0)`
/// before a cross-pipe reread. Clears on vector/scalar memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gfx12Sgpr {
    tracked: [bool; 64],
    salu: [bool; 128],
    valu: [bool; 128],
    vcc_salu: bool,
    vcc_valu: bool,
}

impl Default for Gfx12Sgpr {
    fn default() -> Self {
        Self {
            tracked: [false; 64],
            salu: [false; 128],
            valu: [false; 128],
            vcc_salu: false,
            vcc_valu: false,
        }
    }
}

impl Gfx12Sgpr {
    fn join(&mut self, other: &Self) {
        for (a, b) in self.tracked.iter_mut().zip(other.tracked) {
            *a |= b;
        }
        for (a, b) in self.salu.iter_mut().zip(other.salu) {
            *a |= b;
        }
        for (a, b) in self.valu.iter_mut().zip(other.valu) {
            *a |= b;
        }
        self.vcc_salu |= other.vcc_salu;
        self.vcc_valu |= other.vcc_valu;
    }
    fn wait(&mut self, bits: u16) {
        if bits & 1 == 0 {
            self.salu.fill(false);
            self.vcc_salu = false;
        }
        if bits >> 1 & 1 == 0 {
            self.vcc_valu = false;
        }
        if bits >> 9 & 7 == 0 {
            self.valu.fill(false);
        }
    }
    pub fn clear_on_memory(&mut self) {
        self.salu.fill(false);
        self.valu.fill(false);
        self.vcc_salu = false;
        self.vcc_valu = false;
    }

    /// `IsSetPC` (`s_setpc_b64`, `s_swappc_b64`, calls, returns): every pending
    /// write must be flushed because the target will not see the hazard chain.
    pub fn flush(&mut self) -> Vec<&'static str> {
        let mut waits = Vec::new();
        if self.salu.iter().any(|b| *b) || self.vcc_salu {
            waits.push("depctr_sa_sdst(0)");
        }
        if self.valu.iter().any(|b| *b) {
            waits.push("depctr_va_sdst(0)");
        }
        if self.vcc_valu {
            waits.push("depctr_va_vcc(0)");
        }
        self.salu.fill(false);
        self.valu.fill(false);
        self.vcc_salu = false;
        self.vcc_valu = false;
        waits
    }

    pub fn step(
        &mut self,
        pipe: Pipeline,
        uses: &[RegRef],
        defs: &[RegRef],
        vcc_use: bool,
        vcc_def: bool,
    ) -> Vec<&'static str> {
        if matches!(pipe, Pipeline::Vmem | Pipeline::Smem) {
            self.clear_on_memory();
            return vec![];
        }
        if !matches!(pipe, Pipeline::Salu | Pipeline::Valu) {
            return vec![];
        }
        let mut sa = false;
        let mut va = false;
        let mut vcc = false;
        let mut salu_read_tracked = false;
        for reg in uses {
            if reg.kind != Kind::S {
                continue;
            }
            for n in reg.base as usize..reg.base as usize + reg.len as usize {
                if self.tracked[n / 2] {
                    sa |= self.salu[n];
                    if pipe == Pipeline::Valu {
                        va |= self.valu[n];
                    } else {
                        salu_read_tracked = true;
                    }
                }
            }
        }
        if vcc_use && self.tracked[VCC_PAIR] {
            sa |= self.vcc_salu;
            if pipe == Pipeline::Valu {
                vcc |= self.vcc_valu;
            } else {
                self.vcc_valu = false;
            }
        }
        let mut waits = Vec::new();
        if sa {
            waits.push("depctr_sa_sdst(0)");
            self.salu.fill(false);
            self.vcc_salu = false;
        }
        if va {
            waits.push("depctr_va_sdst(0)");
            self.valu.fill(false);
        }
        if vcc {
            waits.push("depctr_va_vcc(0)");
            self.vcc_valu = false;
        }
        if salu_read_tracked {
            self.valu.fill(false);
        }
        for reg in uses {
            if reg.kind == Kind::S && pipe == Pipeline::Valu {
                for n in reg.base as usize..reg.base as usize + reg.len as usize {
                    self.tracked[n / 2] = true;
                }
            }
        }
        if vcc_use && pipe == Pipeline::Valu {
            self.tracked[VCC_PAIR] = true;
        }
        for reg in defs {
            if reg.kind == Kind::S {
                for n in reg.base as usize..reg.base as usize + reg.len as usize {
                    if self.tracked[n / 2] {
                        if pipe == Pipeline::Salu {
                            self.salu[n] = true;
                        } else {
                            self.valu[n] = true;
                        }
                    }
                }
            }
        }
        if vcc_def && self.tracked[VCC_PAIR] {
            self.vcc_salu = pipe == Pipeline::Salu;
            self.vcc_valu = pipe == Pipeline::Valu;
        }
        waits
    }
}

/// gfx1100 wave32 TRANS-use hazard. Primary evidence: LLVM release/22.x
/// `llvm/lib/Target/AMDGPU/AMDGPU.td` FeatureISAVersion11_0_Common
/// enables VALUTransUseHazard, unlike FeatureISAVersion11_5_Common.
/// `GCNHazardRecognizer.cpp::fixVALUTransUseHazard` tracks five intervening
/// VALUs / one TRANS and expires on VMEM, DS, EXP or depctr va_vdst(0).
/// Neither gfx11 target needs the older VALU-SGPR→VMEM five-NOP rule:
/// FeatureISAVersion11_Common enables NoDataDepHazard, which bypasses
/// `checkVMEMHazards` in GCNHazardRecognizer::PreEmitNoopsCommon.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Gfx11Hazards {
    trans_defs: Vec<(RegRef, u8, u8)>,
}
impl Gfx11Hazards {
    const TRANS: [&'static str; 7] = [
        "v_rcp_", "v_sqrt_", "v_rsq_", "v_sin_", "v_cos_", "v_exp_", "v_log_",
    ];

    pub fn step(
        &mut self,
        pipe: Pipeline,
        mnemonic: &str,
        uses: &[RegRef],
        defs: &[RegRef],
    ) -> Vec<String> {
        let mut waits = Vec::new();
        let trans = pipe == Pipeline::Valu
            && Self::TRANS
                .iter()
                .any(|prefix| mnemonic.starts_with(prefix));
        if matches!(pipe, Pipeline::Vmem | Pipeline::Ds) {
            self.trans_defs.clear();
        }
        if pipe == Pipeline::Valu
            && self.trans_defs.iter().any(|(def, valu_age, trans_age)| {
                *valu_age <= 5 && *trans_age <= 1 && uses.iter().any(|reg| reg.overlaps(*def))
            })
        {
            waits.push("s_waitcnt_depctr depctr_va_vdst(0)".into());
            self.trans_defs.clear();
        }
        if pipe == Pipeline::Valu {
            for (_, valu_age, trans_age) in &mut self.trans_defs {
                *valu_age = valu_age.saturating_add(1);
                if trans {
                    *trans_age = trans_age.saturating_add(1);
                }
            }
            self.trans_defs
                .retain(|(_, valu_age, trans_age)| *valu_age <= 5 && *trans_age <= 1);
            if trans {
                self.trans_defs.extend(
                    defs.iter()
                        .filter(|reg| reg.kind == Kind::V)
                        .map(|reg| (*reg, 0, 0)),
                );
            }
        }
        waits
    }

    /// A decoded depctr va_vdst(0) drains a preceding TRANS.
    pub fn wait_va_vdst(&mut self) {
        self.trans_defs.clear();
    }

    /// At a CFG join any reaching, unresolved TRANS must remain tracked.
    fn join(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for &(reg, va, trans) in &other.trans_defs {
            if let Some((_, old_va, old_trans)) =
                self.trans_defs.iter_mut().find(|(r, _, _)| *r == reg)
            {
                if va < *old_va {
                    *old_va = va;
                    changed = true;
                }
                if trans < *old_trans {
                    *old_trans = trans;
                    changed = true;
                }
            } else {
                self.trans_defs.push((reg, va, trans));
                changed = true;
            }
        }
        if changed {
            self.trans_defs
                .sort_unstable_by_key(|(reg, va, trans)| (*reg, *va, *trans));
        }
        changed
    }
}

/// One demanded-but-absent `s_wait_alu`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MissingWait {
    pub inst: InstId,
    pub wait: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HazardAnalysis {
    pub missing: Vec<MissingWait>,
    pub obligations: Vec<Obligation>,
}

/// Which SGPR-hazard replay a gfx1201 analysis runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SgprModel {
    /// The builder-parity boundary the wave32 pipeline has always used: VCC is
    /// not producer-tracked and every VMEM/FLAT/SMEM access clears the state.
    BuilderParity,
    /// `AMDGPUWaitSGPRHazards.cpp` as LLVM runs it: VCC tracked as a pair, FLAT
    /// (`global_*`/`scratch_*`/`flat_*`) does not clear, `IsSetPC` flushes.
    LlvmExact,
}

/// Whole-program hazard replay for the supported ISA tables. RDNA3 gfx11
/// admits wave32 and wave64; gfx1201 replays the `AMDGPUWaitSGPRHazards` SGPR
/// model in both wave sizes (lane masks are SGPR pairs in wave64, see the
/// module docs). wave32 uses [`SgprModel::BuilderParity`] (unchanged), wave64
/// uses [`SgprModel::LlvmExact`].
pub fn analyze(body: &Body, arch: Arch, wave: Wave) -> Result<HazardAnalysis, HazardError> {
    let model = match wave {
        Wave::Wave32 => SgprModel::BuilderParity,
        Wave::Wave64 => SgprModel::LlvmExact,
    };
    analyze_with(body, arch, wave, model)
}

/// [`analyze`] with an explicit gfx1201 SGPR model (ignored on gfx11).
pub fn analyze_with(
    body: &Body,
    arch: Arch,
    wave: Wave,
    model: SgprModel,
) -> Result<HazardAnalysis, HazardError> {
    let mut analysis = HazardAnalysis::default();
    match arch {
        Arch::Gfx1201 => run_sgpr_model(body, arch, wave, model, &mut analysis)?,
        Arch::Gfx1100 | Arch::Gfx1151 => run_gfx11(body, arch, &mut analysis),
        _ => return Err(HazardError::Unsupported(arch, wave)),
    }
    run_wmma(body, arch, &mut analysis);
    Ok(analysis)
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Gfx11State {
    trans: Gfx11Hazards,
    cmpx_pending: bool,
}

impl Gfx11State {
    fn join(&mut self, other: &Self) -> bool {
        let trans_changed = self.trans.join(&other.trans);
        let cmpx_changed = !self.cmpx_pending && other.cmpx_pending;
        self.cmpx_pending |= other.cmpx_pending;
        trans_changed || cmpx_changed
    }
}

fn gfx11_block(
    body: &Body,
    arch: Arch,
    block: &crate::cfg::Block,
    state: &mut Gfx11State,
    mut analysis: Option<&mut HazardAnalysis>,
) {
    for &id in &body.layout[block.range.0..block.range.1] {
        let Some(inst) = body.insts.get(id) else {
            continue;
        };
        if matches!(inst.effects.control, Control::EndPgm) {
            *state = Gfx11State::default();
            continue;
        }
        let name = inst.op.name(arch).unwrap_or("");
        // ROCm LLVM GCNHazardRecognizer.cpp fixVcmpxPermlaneHazards and
        // AMDGPU.td FeatureISAVersion11_Common enable this on both targets.
        // fixVcmpxPermlaneHazards: V_NOP does NOT break the chain; a real
        // intervening VALU (e.g. v_mov_b32) does.
        let perm_lane = name == "v_permlane16_b32" || name == "v_permlanex16_b32";
        if perm_lane && state.cmpx_pending {
            if let Some(analysis) = &mut analysis {
                analysis.missing.push(MissingWait {
                    inst: id,
                    wait: "v_mov_b32 <live vgpr>, <same vgpr>".into(),
                });
                analysis.obligations.push(Obligation {
                    kind: ObligationKind::Hazard, insts: vec![id],
                    rule_id: "gfx11-vcmpx-permlane".into(),
                    text: "VCMPX→PERMLANE needs a real intervening VALU (LLVM GCNHazardRecognizer.cpp fixVcmpxPermlaneHazards)".into(),
                });
            }
        }
        if pipe_of(inst, arch) == Pipeline::Valu && name != "v_nop_e32" && name != "v_nop_e64" {
            state.cmpx_pending = name.starts_with("v_cmpx_");
        }
        // gfx11 depctr encoding: va_vdst is SIMM16[15:12]; zero drains
        // preceding TRANS (LLVM AMDGPU gfx11_waitcnt_depctr.rst).
        if name == "s_waitcnt_depctr"
            && inst.operands.iter().any(
                |op| matches!(op, Operand::Imm(ImmField::Sopp(n)) if (*n as u16 >> 12) & 15 == 0),
            )
        {
            state.trans.wait_va_vdst();
        }
        if arch == Arch::Gfx1100 {
            for demand in state.trans.step(
                pipe_of(inst, arch),
                name,
                &inst.effects.uses,
                &inst.effects.defs,
            ) {
                if let Some(analysis) = &mut analysis {
                    analysis.missing.push(MissingWait {
                        inst: id,
                        wait: demand.clone(),
                    });
                    analysis.obligations.push(Obligation {
                        kind: ObligationKind::Hazard, insts: vec![id],
                        rule_id: "gfx1100-trans-valu".into(),
                        text: format!("{name} needs {demand} (LLVM GCNHazardRecognizer.cpp fixVALUTransUseHazard)"),
                    });
                }
            }
        }
    }
}

fn run_gfx11(body: &Body, arch: Arch, analysis: &mut HazardAnalysis) {
    // LLVM GCNHazardRecognizer::hasHazard walks predecessor blocks. Join all
    // reaching unresolved facts and revisit loops until their state settles;
    // replay once more from the fixed-point inputs to emit each site once.
    let mut inputs = vec![None; body.blocks.len()];
    if inputs.is_empty() {
        return;
    }
    inputs[0] = Some(Gfx11State::default());
    let mut pending = std::collections::BTreeSet::from([0]);
    while let Some(index) = pending.pop_first() {
        let Some(mut state) = inputs[index].clone() else {
            continue;
        };
        let block = &body.blocks[index];
        gfx11_block(body, arch, block, &mut state, None);
        for successor in &block.succs {
            let next = &mut inputs[successor.0];
            let changed = match next {
                Some(existing) => existing.join(&state),
                None => {
                    *next = Some(state.clone());
                    true
                }
            };
            if changed {
                pending.insert(successor.0);
            }
        }
    }
    for (block, input) in body.blocks.iter().zip(inputs) {
        if let Some(mut state) = input {
            gfx11_block(body, arch, block, &mut state, Some(analysis));
        }
    }
}

fn sgpr_block(
    body: &Body,
    arch: Arch,
    wave: Wave,
    model: SgprModel,
    block: &crate::cfg::Block,
    machine: &mut Gfx12Sgpr,
    mut analysis: Option<&mut HazardAnalysis>,
) {
    for &id in &body.layout[block.range.0..block.range.1] {
        let Some(inst) = body.insts.get(id) else {
            continue;
        };
        if matches!(inst.effects.control, Control::EndPgm) {
            *machine = Gfx12Sgpr::default();
            continue;
        }
        if inst.op.name(arch) == Some("s_wait_alu") {
            for operand in &inst.operands {
                if let Operand::Imm(ImmField::Sopp(bits) | ImmField::Sopk(bits)) = operand {
                    machine.wait(*bits as u16);
                }
            }
            continue;
        }
        // Encoding rows use 64-bit slots for lane masks in both wave sizes.
        // Use the kernel's metadata-selected wave width, unlike genuine
        // 64-bit scalar data operands, which remain two registers.
        let mask_len = match wave {
            Wave::Wave32 => 1,
            Wave::Wave64 => 2,
        };
        let name = inst.op.name(arch).unwrap_or("");
        let mut uses = smallvec::SmallVec::<[RegRef; 4]>::new();
        let mut defs = smallvec::SmallVec::<[RegRef; 2]>::new();
        let mask_input =
            name.starts_with("v_cndmask_") || (name.starts_with("v_") && name.contains("_co_ci_"));
        let mask_output = name.starts_with("v_cmp")
            || name.starts_with("v_div_scale_")
            || (name.starts_with("v_") && name.contains("_co_"));
        let use_refs = if mask_input {
            let mask = inst.operands.last().and_then(|operand| match operand {
                Operand::Reg(reg) => Some(*reg),
                _ => None,
            });
            uses.extend(inst.effects.uses.iter().map(|reg| {
                let mut reg = *reg;
                if Some(reg) == mask && reg.kind == Kind::S {
                    reg.len = mask_len;
                }
                reg
            }));
            uses.as_slice()
        } else {
            inst.effects.uses.as_slice()
        };
        let def_refs = if mask_output {
            defs.extend(inst.effects.defs.iter().map(|reg| {
                let mut reg = *reg;
                if reg.kind == Kind::S {
                    reg.len = mask_len;
                }
                reg
            }));
            defs.as_slice()
        } else {
            inst.effects.defs.as_slice()
        };
        // LlvmExact replays the LLVM pass exactly (VCC tracked, FLAT does not
        // clear, IsSetPC flushes). BuilderParity keeps the historic boundary:
        // VCC is not producer-tracked and every memory access clears.
        let exact = model == SgprModel::LlvmExact;
        let prior = machine.clone();
        let demands = if exact && matches!(name, "s_setpc_b64" | "s_swappc_b64") {
            machine.flush()
        } else if exact {
            let (vcc_use, vcc_def) = vcc_access(inst, name);
            machine.step(llvm_pipe(inst, arch), use_refs, def_refs, vcc_use, vcc_def)
        } else {
            machine.step(pipe_of(inst, arch), use_refs, def_refs, false, false)
        };
        for demand in demands {
            // A missing guard is not an imaginary wait. Keep unresolved
            // writes pending so later consumers also see the real state.
            match demand {
                "depctr_sa_sdst(0)" => {
                    for (a, b) in machine.salu.iter_mut().zip(prior.salu) {
                        *a |= b;
                    }
                    machine.vcc_salu |= prior.vcc_salu;
                }
                "depctr_va_sdst(0)" => {
                    for (a, b) in machine.valu.iter_mut().zip(prior.valu) {
                        *a |= b;
                    }
                }
                "depctr_va_vcc(0)" => machine.vcc_valu |= prior.vcc_valu,
                _ => unreachable!("closed SGPR demand set"),
            }
            if let Some(analysis) = &mut analysis {
                analysis.missing.push(MissingWait {
                    inst: id,
                    wait: demand.into(),
                });
                analysis.obligations.push(Obligation {
                    kind: ObligationKind::Hazard,
                    insts: vec![id],
                    rule_id: format!("sgpr-{}", demand.replace(['(', ')'], "")),
                    text: format!("SGPR read needs {demand} on a reaching CFG path"),
                });
            }
        }
    }
}

/// Default-model replay (wave32 builder parity, wave64 exact); unit tests.
#[cfg(test)]
fn run_sgpr(
    body: &Body,
    arch: Arch,
    wave: Wave,
    analysis: &mut HazardAnalysis,
) -> Result<(), HazardError> {
    let model = match wave {
        Wave::Wave32 => SgprModel::BuilderParity,
        Wave::Wave64 => SgprModel::LlvmExact,
    };
    run_sgpr_model(body, arch, wave, model, analysis)
}

fn run_sgpr_model(
    body: &Body,
    arch: Arch,
    wave: Wave,
    model: SgprModel,
    analysis: &mut HazardAnalysis,
) -> Result<(), HazardError> {
    if body.layout.is_empty() {
        return Ok(());
    }
    if body.blocks.is_empty() {
        return Err(HazardError::MissingCfg);
    }
    let mut inputs = vec![None; body.blocks.len()];
    let mut outputs = inputs.clone();
    let mut fixed = false;
    for _ in 0..256 {
        let mut changed = false;
        for (index, block) in body.blocks.iter().enumerate() {
            let mut at = (index == 0).then(Gfx12Sgpr::default);
            for from in &block.preds {
                if let Some(s) = &outputs[from.0] {
                    if let Some(at) = &mut at {
                        at.join(s);
                    } else {
                        at = Some(s.clone());
                    }
                }
            }
            let next = at.clone().map(|mut s| {
                sgpr_block(body, arch, wave, model, block, &mut s, None);
                s
            });
            changed |= next != outputs[index];
            inputs[index] = at;
            outputs[index] = next;
        }
        if !changed {
            fixed = true;
            break;
        }
    }
    if !fixed {
        return Err(HazardError::NoFixpoint { walks: 256 });
    }
    for (block, input) in body.blocks.iter().zip(inputs) {
        if let Some(mut state) = input {
            sgpr_block(body, arch, wave, model, block, &mut state, Some(analysis));
        }
    }
    Ok(())
}

/// ROCm LLVM GCNHazardRecognizer.cpp::fixWMMAHazards searches across
/// predecessor edges, retaining every reaching prior WMMA until an independent
/// VALU (including V_NOP) breaks the chain.
fn wmma_block(
    body: &Body,
    arch: Arch,
    block: &crate::cfg::Block,
    previous: &mut Vec<RegRef>,
    mut analysis: Option<&mut HazardAnalysis>,
) {
    for &id in &body.layout[block.range.0..block.range.1] {
        let Some(inst) = body.insts.get(id) else {
            continue;
        };
        let name = inst.op.name(arch).unwrap_or("");
        if name.starts_with("v_wmma_") || name.starts_with("v_swmmac_") {
            let chained = previous.iter().any(|old| {
                inst.effects
                    .uses
                    .iter()
                    .take(2)
                    .any(|src| old.overlaps(*src))
                    || (arch == Arch::Gfx1201
                        && name.starts_with("v_swmmac_")
                        && inst
                            .effects
                            .uses
                            .get(2)
                            .is_some_and(|src| old.overlaps(*src)))
            });
            if chained {
                if let Some(analysis) = &mut analysis {
                    analysis.obligations.push(Obligation {
                        kind: ObligationKind::Hazard, insts: vec![id],
                        rule_id: "wmma-chain".into(),
                        text: format!("{name} reuses a reaching matrix destination without an intervening VALU or v_nop"),
                    });
                }
            }
            previous.clear();
            if let Some(dst) = inst.effects.defs.first() {
                previous.push(*dst);
            }
        } else if pipe_of(inst, arch) == Pipeline::Valu {
            previous.clear();
        }
    }
}

fn run_wmma(body: &Body, arch: Arch, analysis: &mut HazardAnalysis) {
    let mut inputs: Vec<Option<Vec<RegRef>>> = vec![None; body.blocks.len()];
    if inputs.is_empty() {
        return;
    }
    inputs[0] = Some(Vec::new());
    let mut pending = std::collections::BTreeSet::from([0]);
    while let Some(index) = pending.pop_first() {
        let Some(mut state) = inputs[index].clone() else {
            continue;
        };
        let block = &body.blocks[index];
        wmma_block(body, arch, block, &mut state, None);
        for successor in &block.succs {
            let next = &mut inputs[successor.0];
            let changed = match next {
                Some(existing) => {
                    let old_len = existing.len();
                    for &reg in &state {
                        if !existing.contains(&reg) {
                            existing.push(reg);
                        }
                    }
                    existing.len() != old_len
                }
                None => {
                    *next = Some(state.clone());
                    true
                }
            };
            if changed {
                pending.insert(successor.0);
            }
        }
    }
    for (block, input) in body.blocks.iter().zip(inputs) {
        if let Some(mut state) = input {
            wmma_block(body, arch, block, &mut state, Some(analysis));
        }
    }
}

/// Block-local SGPR replay used by unit tests: returns demands in order.
#[cfg(test)]
pub fn sgpr_demands(body: &Body, arch: Arch) -> Vec<(InstId, Vec<&'static str>)> {
    let mut machine = Gfx12Sgpr::default();
    let mut out = Vec::new();
    for id in &body.layout {
        let Some(inst) = body.insts.get(*id) else {
            continue;
        };
        let pipe = pipe_of(inst, arch);
        let demands = machine.step(pipe, &inst.effects.uses, &inst.effects.defs, false, false);
        if !demands.is_empty() {
            out.push((*id, demands));
        }
    }
    out
}

/// Ported rule self-checks plus WMMA chaining on hand-built streams.
#[cfg(test)]
mod tests {
    use super::*;

    fn sgpr_probe(ops: &[(&str, Vec<Operand>)]) -> (Body, HazardAnalysis) {
        use crate::{inst::FormFields, operand::Modifiers, provenance::Provenance};
        let arch = Arch::Gfx1201;
        let mut body = Body::default();
        for (name, operands) in ops {
            let row = crate::isa::table(arch)
                .iter()
                .find(|row| row.name == *name)
                .expect(name);
            let inst = Inst::from_parts(
                arch,
                row.op,
                row.form,
                FormFields::None,
                smallvec::SmallVec::from_vec(operands.clone()),
                Modifiers::default(),
                None,
                Provenance::default(),
            )
            .unwrap();
            body.layout.push(body.insts.insert(inst));
        }
        crate::passes::cfg::build_blocks(&mut body, arch).unwrap();
        let found = analyze(&body, arch, Wave::Wave32).unwrap();
        (body, found)
    }
    fn sgpr_read() -> (&'static str, Vec<Operand>) {
        (
            "v_add_nc_u32_e32",
            vec![
                Operand::Reg(RegRef {
                    kind: Kind::V,
                    base: 0,
                    len: 1,
                }),
                Operand::Reg(RegRef {
                    kind: Kind::S,
                    base: 4,
                    len: 1,
                }),
                Operand::Reg(RegRef {
                    kind: Kind::V,
                    base: 1,
                    len: 1,
                }),
            ],
        )
    }
    fn sgpr_write() -> (&'static str, Vec<Operand>) {
        (
            "s_mov_b32",
            vec![
                Operand::Reg(RegRef {
                    kind: Kind::S,
                    base: 4,
                    len: 1,
                }),
                Operand::Inline(crate::operand::InlineConst::Integer(1)),
            ],
        )
    }
    fn sa_wait() -> (&'static str, Vec<Operand>) {
        (
            "s_wait_alu",
            vec![Operand::Imm(ImmField::Sopp(0xff9eu16 as i16))],
        )
    }
    fn branch(name: &'static str, offset: i16) -> (&'static str, Vec<Operand>) {
        (name, vec![Operand::Imm(ImmField::Sopp(offset))])
    }

    #[test]
    fn explicit_lane_mask_width_preserves_wave64_and_real_scalar_pairs() {
        let s = |base, len| {
            Operand::Reg(RegRef {
                kind: Kind::S,
                base,
                len,
            })
        };
        let v = |base, len| {
            Operand::Reg(RegRef {
                kind: Kind::V,
                base,
                len,
            })
        };
        let one = Operand::Inline(crate::operand::InlineConst::Integer(1));
        let (body, _) = sgpr_probe(&[
            ("v_add_nc_u32_e32", vec![v(0, 1), s(1, 1), v(1, 1)]),
            ("s_mov_b32", vec![s(1, 1), one.clone()]),
            (
                "v_cndmask_b32_e64",
                vec![v(2, 1), v(0, 1), v(1, 1), s(0, 2)],
            ),
            ("v_lshlrev_b64_e64", vec![v(4, 2), one, s(0, 2)]),
            ("s_endpgm", vec![]),
        ]);
        for (wave, expected) in [
            (Wave::Wave32, vec![body.layout[3]]),
            (Wave::Wave64, vec![body.layout[2], body.layout[3]]),
        ] {
            let mut found = HazardAnalysis::default();
            run_sgpr(&body, Arch::Gfx1201, wave, &mut found).unwrap();
            assert_eq!(
                found.missing.iter().map(|m| m.inst).collect::<Vec<_>>(),
                expected
            );
        }
    }
    #[test]
    fn gfx12_sgpr_loop_head_keeps_backedge_write() {
        let (body, found) = sgpr_probe(&[
            sgpr_read(),
            sgpr_write(),
            branch("s_cbranch_scc1", -3),
            ("s_endpgm", vec![]),
        ]);
        assert!(
            found
                .missing
                .iter()
                .any(|m| m.inst == body.layout[0] && m.wait == "depctr_sa_sdst(0)"),
            "{found:?}"
        );
        let (_, guarded) = sgpr_probe(&[
            sa_wait(),
            sgpr_read(),
            sgpr_write(),
            branch("s_cbranch_scc1", -4),
            ("s_endpgm", vec![]),
        ]);
        assert!(guarded.obligations.is_empty(), "{guarded:?}");
    }

    #[test]
    fn gfx12_guard_on_one_arm_does_not_cover_the_other() {
        let (body, found) = sgpr_probe(&[
            sgpr_read(),
            sgpr_write(),
            branch("s_cbranch_scc1", 2),
            sa_wait(),
            branch("s_branch", 1),
            sgpr_read(),
            ("s_endpgm", vec![]),
        ]);
        assert!(
            found.missing.iter().any(|m| m.inst == body.layout[5]),
            "{found:?}"
        );
        let (_, guarded) = sgpr_probe(&[
            sgpr_read(),
            sgpr_write(),
            branch("s_cbranch_scc1", 2),
            sa_wait(),
            branch("s_branch", 2),
            sa_wait(),
            sgpr_read(),
            ("s_endpgm", vec![]),
        ]);
        assert!(guarded.obligations.is_empty(), "{guarded:?}");
    }

    #[test]
    fn gfx12_unreachable_write_does_not_infect_join() {
        let (_, found) = sgpr_probe(&[
            sgpr_read(),
            branch("s_branch", 2),
            sgpr_write(),
            ("s_endpgm", vec![]),
            sgpr_read(),
            ("s_endpgm", vec![]),
        ]);
        assert!(found.obligations.is_empty(), "{found:?}");
    }

    #[test]
    fn gfx12_guard_before_new_write_is_not_reusable() {
        let (body, found) = sgpr_probe(&[
            sgpr_read(),
            sa_wait(),
            sgpr_write(),
            sgpr_read(),
            ("s_endpgm", vec![]),
        ]);
        assert!(
            found.missing.iter().any(|m| m.inst == body.layout[3]),
            "{found:?}"
        );
    }

    // gfx1201 wave64 encodings (llvm-mc -mcpu=gfx1201 -mattr=+wavefrontsize64).
    const CNDMASK_VCC: &[u32] = &[0x0204_0903]; // v_cndmask_b32_e32 v2, v3, v4, vcc
    const CMP_VCC: &[u32] = &[0x7c82_0300]; // v_cmp_lt_i32_e32 vcc, v0, v1
    const CBR_VCCZ_1: &[u32] = &[0xbfa3_0001]; // s_cbranch_vccz +1
    const SAVEEXEC_VCC: &[u32] = &[0xbe82_216a]; // s_and_saveexec_b64 s[2:3], vcc
    const CNDMASK_S23_A: &[u32] = &[0xd501_0000, 0x000a_0501]; // v_cndmask_b32_e64 v0, v1, v2, s[2:3]
    const CMP64_S23: &[u32] = &[0xd441_0002, 0x0202_0300]; // v_cmp_lt_i32_e64 s[2:3], v0, v1
    const CNDMASK_S23_B: &[u32] = &[0xd501_0003, 0x000a_0501]; // v_cndmask_b32_e64 v3, v1, v2, s[2:3]
    const MOV_V0_S3: &[u32] = &[0x7e00_0203]; // v_mov_b32_e32 v0, s3
    const MOV_V1_S3: &[u32] = &[0x7e02_0203]; // v_mov_b32_e32 v1, s3
    const MOV_S23_0: &[u32] = &[0xbe82_0180]; // s_mov_b64 s[2:3], 0
    const AND_VCC_EXEC_S45: &[u32] = &[0x8bea_047e]; // s_and_b64 vcc, exec, s[4:5]
    const WAIT_VA_SDST0: &[u32] = &[0xbf88_f19f]; // s_wait_alu depctr_va_sdst(0)
    const WAIT_SA_SDST0: &[u32] = &[0xbf88_ff9e]; // s_wait_alu depctr_sa_sdst(0)
    const WAIT_VA_VCC0: &[u32] = &[0xbf88_ff9d]; // s_wait_alu depctr_va_vcc(0)
    const GLOBAL_LOAD: &[u32] = &[0xee05_007c, 0, 2]; // global_load_b32 v0, v[2:3], off
    const BUFFER_LOAD: &[u32] = &[0xc405_000c, 0x4080_1000, 1]; // buffer_load_b32 v0, v1, s[8:11], s12 offen
    const S_LOAD: &[u32] = &[0xf400_0000, 0xf800_0000]; // s_load_b32 s0, s[0:1], 0x0
    const S_PREFETCH: &[u32] = &[0xf404_c00b, 0xf800_0000]; // s_prefetch_data s[22:23], 0x0, null, 0
    const SETPC_S45: &[u32] = &[0xbe80_4804]; // s_setpc_b64 s[4:5]
    const ENDPGM: &[u32] = &[0xbfb0_0000]; // s_endpgm

    fn decode_body(arch: Arch, seq: &[&[u32]]) -> Body {
        let mut body = Body::default();
        for words in seq.iter().chain(std::iter::once(&ENDPGM)) {
            let (inst, used) = crate::codec::gfx12::decode_for(arch, words).expect("decode");
            assert_eq!(used, words.len(), "{words:x?}");
            body.layout.push(body.insts.insert(inst));
        }
        crate::passes::cfg::build_blocks(&mut body, arch).expect("cfg");
        body
    }

    /// `(body, missing waits as (layout index, wait))` of one gfx1201 stream.
    fn gfx1201_missing(wave: Wave, seq: &[&[u32]]) -> Vec<(usize, String)> {
        let body = decode_body(Arch::Gfx1201, seq);
        let found = analyze(&body, Arch::Gfx1201, wave).expect("gfx1201 analysis");
        found
            .missing
            .iter()
            .map(|m| {
                let at = body.layout.iter().position(|id| *id == m.inst).unwrap();
                (at, m.wait.clone())
            })
            .collect()
    }

    /// AMDGPUWaitSGPRHazards, wave64: VCC is the tracked pair vcc_lo/vcc_hi
    /// and EXEC is never tracked, so a VALU-written VCC read by the SALU
    /// branch / saveexec needs nothing (the SALU read retires it).
    #[test]
    fn gfx1201_wave64_accepts_vcc_branch_and_saveexec() {
        let seq: &[&[u32]] = &[CNDMASK_VCC, CMP_VCC, CBR_VCCZ_1, SAVEEXEC_VCC, CNDMASK_VCC];
        // The last v_cndmask rereads the VALU-written VCC after the SALU read
        // retired it: nothing pending, so the stream is accepted.
        assert_eq!(gfx1201_missing(Wave::Wave64, seq), vec![]);
        // Same stream without the SALU consumer: VALU VCC write, VALU reread.
        assert_eq!(
            gfx1201_missing(Wave::Wave64, &[CNDMASK_VCC, CMP_VCC, CNDMASK_VCC]),
            vec![(2, "depctr_va_vcc(0)".to_string())]
        );
    }

    #[test]
    fn gfx1201_wave64_vcc_waits_are_verified() {
        // SALU write of VCC reread by a VALU needs sa_sdst(0).
        let hazard: &[&[u32]] = &[CNDMASK_VCC, AND_VCC_EXEC_S45, CNDMASK_VCC];
        assert_eq!(
            gfx1201_missing(Wave::Wave64, hazard),
            vec![(2, "depctr_sa_sdst(0)".to_string())]
        );
        assert_eq!(
            gfx1201_missing(
                Wave::Wave64,
                &[CNDMASK_VCC, AND_VCC_EXEC_S45, WAIT_SA_SDST0, CNDMASK_VCC]
            ),
            vec![]
        );
        // va_vcc(0) retires a VALU VCC write; sa_sdst(0) does not.
        let valu: &[&[u32]] = &[CNDMASK_VCC, CMP_VCC, WAIT_SA_SDST0, CNDMASK_VCC];
        assert_eq!(
            gfx1201_missing(Wave::Wave64, valu),
            vec![(3, "depctr_va_vcc(0)".to_string())]
        );
        assert_eq!(
            gfx1201_missing(
                Wave::Wave64,
                &[CNDMASK_VCC, CMP_VCC, WAIT_VA_VCC0, CNDMASK_VCC]
            ),
            vec![]
        );
        // VCC that no VALU ever read is not tracked (LLVM `Tracked`).
        assert_eq!(
            gfx1201_missing(Wave::Wave64, &[AND_VCC_EXEC_S45, CNDMASK_VCC]),
            vec![]
        );
    }

    /// The synthetic wave64 violation: a VALU-written lane-mask pair read by
    /// a VALU without `s_wait_alu depctr_va_sdst(0)`.
    #[test]
    fn gfx1201_wave64_missing_va_sdst_is_caught() {
        let hazard: &[&[u32]] = &[CNDMASK_S23_A, CMP64_S23, CNDMASK_S23_B];
        assert_eq!(
            gfx1201_missing(Wave::Wave64, hazard),
            vec![(2, "depctr_va_sdst(0)".to_string())]
        );
        assert_eq!(
            gfx1201_missing(
                Wave::Wave64,
                &[CNDMASK_S23_A, CMP64_S23, WAIT_VA_SDST0, CNDMASK_S23_B]
            ),
            vec![]
        );
        // sa_sdst(0) is not the guard for a VALU write.
        assert_eq!(
            gfx1201_missing(
                Wave::Wave64,
                &[CNDMASK_S23_A, CMP64_S23, WAIT_SA_SDST0, CNDMASK_S23_B]
            ),
            vec![(3, "depctr_va_sdst(0)".to_string())]
        );
    }

    /// Mask pairs are two SGPRs wide in wave64 only: the VALU read of the high
    /// half s3 hits the pair written by the 64-bit `v_cmp`, but in wave32 the
    /// mask is s2 alone.
    #[test]
    fn gfx1201_mask_pair_high_half_is_wave64_only() {
        let seq: &[&[u32]] = &[MOV_V0_S3, CMP64_S23, MOV_V1_S3];
        assert_eq!(
            gfx1201_missing(Wave::Wave64, seq),
            vec![(2, "depctr_va_sdst(0)".to_string())]
        );
        assert_eq!(gfx1201_missing(Wave::Wave32, seq), vec![]);
    }

    /// s_and_saveexec_b64 writes a genuine SGPR pair (both halves) and EXEC
    /// itself is never a tracked register.
    #[test]
    fn gfx1201_wave64_saveexec_defines_the_whole_pair() {
        let mov_v1_s2: &[u32] = &[0x7e02_0202]; // v_mov_b32_e32 v1, s2
        assert_eq!(
            gfx1201_missing(Wave::Wave64, &[MOV_V0_S3, SAVEEXEC_VCC, mov_v1_s2]),
            vec![(2, "depctr_sa_sdst(0)".to_string())]
        );
    }

    /// `isVMEM && !isFLAT` and `isSMRD` retire pending writes, FLAT
    /// (`global_*`) does not; `s_setpc_b64` flushes everything.
    #[test]
    fn gfx1201_wave64_memory_clears_and_setpc_flushes() {
        let tail = |mem: &'static [u32]| -> Vec<(usize, String)> {
            gfx1201_missing(Wave::Wave64, &[MOV_V0_S3, MOV_S23_0, mem, MOV_V1_S3])
        };
        assert_eq!(
            tail(GLOBAL_LOAD),
            vec![(3, "depctr_sa_sdst(0)".to_string())]
        );
        assert_eq!(tail(BUFFER_LOAD), vec![]);
        assert_eq!(tail(S_LOAD), vec![]);
        // SMEM hints are `isSMRD` too: a prefetch consumes no pending SGPR write and clears the state.
        assert_eq!(tail(S_PREFETCH), vec![]);
        // wave32 keeps the historic model where every VMEM/FLAT access clears.
        assert_eq!(
            gfx1201_missing(
                Wave::Wave32,
                &[MOV_V0_S3, MOV_S23_0, GLOBAL_LOAD, MOV_V1_S3]
            ),
            vec![]
        );
        assert_eq!(
            gfx1201_missing(Wave::Wave64, &[MOV_V0_S3, MOV_S23_0, SETPC_S45]),
            vec![(2, "depctr_sa_sdst(0)".to_string())]
        );
        assert_eq!(
            gfx1201_missing(
                Wave::Wave64,
                &[MOV_V0_S3, MOV_S23_0, WAIT_SA_SDST0, SETPC_S45]
            ),
            vec![]
        );
    }

    /// `analyze_with` picks the model explicitly: LLVM-exact in wave32 sees the
    /// FLAT write that builder parity retires, and wave64 builder parity
    /// retires it again.
    #[test]
    fn gfx1201_model_is_selectable_per_wave() {
        let body = decode_body(
            Arch::Gfx1201,
            &[MOV_V0_S3, MOV_S23_0, GLOBAL_LOAD, MOV_V1_S3],
        );
        let missing = |wave, model| {
            analyze_with(&body, Arch::Gfx1201, wave, model)
                .unwrap()
                .missing
                .len()
        };
        assert_eq!(missing(Wave::Wave32, SgprModel::BuilderParity), 0);
        assert_eq!(missing(Wave::Wave32, SgprModel::LlvmExact), 1);
        assert_eq!(missing(Wave::Wave64, SgprModel::BuilderParity), 0);
        assert_eq!(missing(Wave::Wave64, SgprModel::LlvmExact), 1);
    }

    /// Gfx12Sgpr tracks the builder's exact demand sequence.
    #[test]
    fn sgpr_machine_demands_depctr_on_cross_pipe_reread() {
        let mut machine = Gfx12Sgpr::default();
        let s = |base: u16, len: u8| RegRef {
            kind: Kind::S,
            base,
            len,
        };
        // VALU reads s4 (track pair 2), SALU writes s4, VALU rereads.
        assert!(machine
            .step(Pipeline::Valu, &[s(4, 1)], &[], false, false)
            .is_empty());
        assert!(machine
            .step(Pipeline::Salu, &[], &[s(4, 1)], false, false)
            .is_empty());
        assert_eq!(
            machine.step(Pipeline::Valu, &[s(4, 1)], &[], false, false),
            vec!["depctr_sa_sdst(0)"]
        );
        // VALU-written SGPR reread by VALU needs va_sdst.
        let mut machine = Gfx12Sgpr::default();
        assert!(machine
            .step(Pipeline::Valu, &[s(8, 1)], &[], false, false)
            .is_empty());
        assert!(machine
            .step(Pipeline::Valu, &[], &[s(8, 1)], false, false)
            .is_empty());
        assert_eq!(
            machine.step(Pipeline::Valu, &[s(8, 1)], &[], false, false),
            vec!["depctr_va_sdst(0)"]
        );
        // Vector memory clears the tracker.
        let mut machine = Gfx12Sgpr::default();
        assert!(machine
            .step(Pipeline::Valu, &[s(4, 1)], &[], false, false)
            .is_empty());
        assert!(machine
            .step(Pipeline::Salu, &[], &[s(4, 1)], false, false)
            .is_empty());
        assert!(machine
            .step(Pipeline::Vmem, &[], &[], false, false)
            .is_empty());
        assert!(machine
            .step(Pipeline::Valu, &[s(4, 1)], &[], false, false)
            .is_empty());
    }

    /// LLVM AMDGPU.td gfx11 NoDataDepHazard excludes VALU-SGPR→VMEM NOPs;
    /// gfx1100's separately gated TRANS-use rule needs depctr va_vdst(0).
    #[test]
    fn gfx11_trans_forwarding_but_no_vmem_sgpr_nop() {
        let v = |base: u16| RegRef {
            kind: Kind::V,
            base,
            len: 1,
        };
        let s = |base: u16| RegRef {
            kind: Kind::S,
            base,
            len: 1,
        };
        let mut machine = Gfx11Hazards::default();
        assert!(machine
            .step(Pipeline::Valu, "v_rcp_f32", &[], &[v(10)])
            .is_empty());
        assert_eq!(
            machine.step(Pipeline::Valu, "v_add_f32", &[v(10)], &[v(11)]),
            vec!["s_waitcnt_depctr depctr_va_vdst(0)"]
        );
        machine.wait_va_vdst();
        assert!(machine
            .step(Pipeline::Valu, "v_add_f32", &[v(10)], &[v(11)])
            .is_empty());
        assert!(machine
            .step(Pipeline::Valu, "v_add_f32", &[], &[s(4)])
            .is_empty());
        assert!(machine
            .step(Pipeline::Vmem, "buffer_load_b32", &[s(4)], &[v(0)])
            .is_empty());
    }

    /// A TRANS at the loop tail reaches a VALU use at the head only through
    /// the backedge; a linear layout scan misses this obligation.
    #[test]
    fn gfx1100_trans_use_at_loop_head_tracks_backedge() {
        use crate::cfg::Body;
        use crate::inst::FormFields;
        use crate::operand::Modifiers;
        use crate::provenance::Provenance;
        use smallvec::SmallVec;
        let arch = Arch::Gfx1100;
        let v = |base| {
            Operand::Reg(RegRef {
                kind: Kind::V,
                base,
                len: 1,
            })
        };
        let inst = |name: &str, operands: Vec<Operand>| {
            let row = crate::isa::table(arch)
                .iter()
                .find(|row| row.name == name)
                .expect(name);
            Inst::from_parts(
                arch,
                row.op,
                row.form,
                FormFields::None,
                SmallVec::from_vec(operands),
                Modifiers::default(),
                None,
                Provenance::default(),
            )
            .unwrap_or_else(|error| panic!("{name}: {error:?}"))
        };
        let mut body = Body::default();
        for instruction in [
            inst("v_add_f32_e32", vec![v(12), v(10), v(13)]),
            inst("v_rcp_f32_e32", vec![v(10), v(11)]),
            inst("s_cbranch_scc1", vec![Operand::Imm(ImmField::Sopp(-3))]),
            inst("s_endpgm", vec![]),
        ] {
            let id = body.insts.insert(instruction);
            body.layout.push(id);
        }
        crate::passes::cfg::build_blocks(&mut body, arch).unwrap();
        assert!(body.blocks[0].succs.contains(&crate::cfg::BlockId(0)));
        let found = analyze(&body, arch, Wave::Wave32).unwrap();
        assert!(
            found
                .obligations
                .iter()
                .any(|obligation| obligation.rule_id == "gfx1100-trans-valu"
                    && obligation.insts == vec![body.layout[0]]),
            "{found:?}"
        );
        assert!(!analyze(&body, Arch::Gfx1151, Wave::Wave32)
            .unwrap()
            .obligations
            .iter()
            .any(|obligation| obligation.rule_id == "gfx1100-trans-valu"));
    }

    /// Non-runtime LLVM-MC conformance probe: gfx11 VCMPX→V_NOP→PERMLANE16
    /// still needs a real VALU, while VCMPX→V_MOV→PERMLANEX16 is safe.
    /// These are assembled machine words, not a claim about runtime JIT sites.
    #[test]
    fn gfx11_cmpx_permlane_probe_in_both_wave_sizes() {
        use crate::cfg::Body;
        let words = [
            0x7d28_0300, // v_cmpx_gt_f32_e32 v0, v1
            0x7e00_0000, // v_nop: LLVM does not count this as a separator
            0xd65b_0002,
            0x0004_0101, // v_permlane16_b32 v2, v1, s0, s1
            0x7d28_0300, // another VCMPX
            0x7e00_0300, // v_mov_b32_e32 v0, v0: real VALU separator
            0xd65c_0002,
            0x0004_0101, // v_permlanex16_b32 v2, v1, s0, s1
            0xbfb0_0000, // s_endpgm
        ];
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            let mut body = Body::default();
            let mut offset = 0;
            while offset < words.len() {
                let (inst, width) =
                    crate::codec::gfx12::decode_for(arch, &words[offset..]).unwrap();
                assert_eq!(
                    crate::codec::gfx12::encode_for(arch, &inst)
                        .unwrap()
                        .as_slice(),
                    &words[offset..offset + width]
                );
                body.layout.push(body.insts.insert(inst));
                offset += width;
            }
            crate::passes::cfg::build_blocks(&mut body, arch).unwrap();
            for wave in [Wave::Wave32, Wave::Wave64] {
                let analysis = analyze(&body, arch, wave).unwrap();
                let sites = analysis
                    .obligations
                    .iter()
                    .filter(|o| o.rule_id == "gfx11-vcmpx-permlane")
                    .map(|o| o.insts.as_slice())
                    .collect::<Vec<_>>();
                assert_eq!(
                    sites,
                    vec![&body.layout[2..3]],
                    "{arch:?}/{wave:?}: {analysis:?}"
                );
            }
        }
    }
}

#[cfg(test)]
mod kt48_tests {
    use crate::cfg::Body;
    use crate::inst::{Arch, Wave};
    use crate::passes::cfg::build_blocks;

    fn kt48_body() -> Body {
        const IMAGE: &[u8] =
            include_bytes!("../../../peacemaker-lift/tests/fixtures/kt48/hipcc.co");
        const START: usize = 0x6f00;
        const SIZE: usize = 10_604;
        let words: Vec<u32> = IMAGE[START..START + SIZE]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let mut body = Body::default();
        let mut index = 0;
        while index < words.len() {
            let (inst, count) = crate::codec::gfx12::decode(&words[index..]).expect("KT48 decodes");
            let id = body.insts.insert(inst);
            body.layout.push(id);
            index += count;
        }
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).expect("CFG");
        body
    }

    /// KT48 carries no unsatisfied SGPR/WMMA hazard: hipcc emitted every
    /// depctr wait the machine demands, and no WMMA chain is unguarded.
    #[test]
    fn kt48_hazards_are_clean() {
        let body = kt48_body();
        let analysis = super::analyze(&body, Arch::Gfx1201, Wave::Wave32).unwrap();
        assert!(analysis.missing.is_empty(), "{:?}", analysis.missing);
        assert!(
            analysis.obligations.is_empty(),
            "{:?}",
            analysis.obligations
        );
    }
}
