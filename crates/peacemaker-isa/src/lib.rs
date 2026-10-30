// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#![forbid(unsafe_code)]
//! An imported `Region<In, Out>` must take ownership of typed live-in `V<N>`/`S<N>`
//! ranges and LDS slot tokens, then return its live-out handles. The seam carries
//! `Ledger` plus EXEC/VCC/SCC/M0 capabilities; it reconciles pending counters
//! and slot states before returning ownership. Foreign bytes become a region
//! only after disassembly parse-back and independent wait-ledger replay.
// The gfx11 K2 module (`kernels/qwen4_moe_sym_gfx1100/`) names this crate by its public path so a standalone
// example or test can still include it with `#[path]`; inside the crate that path is the crate itself.
extern crate self as peacemaker_isa;
pub mod aco;
pub mod aco_cert;
pub mod aco_hsa;
pub mod arch;
pub mod author;
/// The CDNA (wave64, gfx942 first) arm: checked builder, hazard pass, code-object emission.
pub mod cdna;
pub mod emit;
pub mod hazard;
pub mod hip_twin;
pub mod insn;
pub mod lds;
pub mod ledger;
pub mod native;
#[path = "kernels/npu_trunk.rs"]
pub mod npu_trunk;
#[path = "kernels/npu_trunk_arena.rs"]
pub mod npu_trunk_arena;
pub mod plan;
pub mod profile;
#[path = "kernels/railgun_guard.rs"]
pub mod railgun_guard;
pub mod reg;
pub mod vopd;
pub mod kernels {
    pub mod bf16;
    pub mod common;
    pub mod dflash_accept;
    pub mod e8p_i4_moe_gemv;
    pub mod ep_gather_grouped_rows_by_tile_owner;
    pub mod ep_tile_mask;
    pub mod fa_prefill;
    pub mod fa_qprep;
    pub mod fp8_gemm;
    pub mod gdn_chunk_prep;
    pub mod gdn_scan;
    pub mod gemm_uk;
    pub mod iu4_c8_producer;
    pub mod iu4_fold;
    pub mod iu4_gemm;
    pub mod iu4_k1;
    pub mod iu4_v2b;
    pub mod iu4_v2b_a4;
    pub mod iu4_v2c;
    pub mod lane_rows_bf16_gfx1201;
    pub mod lane_rows_mq6_gfx1201;
    pub mod legacy_mq4_rowtiles;
    pub mod moe_rotate_i4_pm;
    pub mod moe_slot_ops;
    pub mod mq4_fp8_verify;
    pub mod mq4_rows_x4;
    pub mod mq4_verify;
    pub mod mq6_i8_gemm;
    pub mod mq_i4_a4;
    pub mod mq_i4_moe_gemv;
    pub mod mq_i4_moe_gemv_c1;
    pub mod mq_i4_moe_gemv_cn;
    pub mod mq_i4_moe_gemv_gfx1100;
    pub mod mq_i4_rows;
    pub mod mtp_dense_det_gfx11;
    pub mod mtp_q8_det_gfx11;
    pub mod pm_decode;
    #[path = "qsa_gather.rip.rs"]
    pub mod qsa_gather;
    #[path = "qsa_score.rip.rs"]
    pub mod qsa_score;
    pub mod qsa_select;
    #[path = "qsa_topk.rip.rs"]
    pub mod qsa_topk;
    /// Shared HC epilogue emission helpers of the BF16 wide k4 entries.
    pub mod qwen4_bf16_hc_epilogue;
    pub mod qwen4_bf16_rows_gfx1100;
    /// BF16 singleton-DAG verify twins: fixed eight-row physical tiles, logical rows 1..64, ordinary x4 and k4 HC entries.
    pub mod qwen4_bf16_rows_wide;
    pub mod qwen4_ep_owner_assign_pm;
    pub mod qwen4_lane_rows;
    pub mod qwen4_lane_rows_mq6;
    pub mod qwen4_lane_rows_q8;
    pub mod qwen4_lane_rows_q8_gfx1100;
    pub mod qwen4_moe_down_gp;
    pub mod qwen4_moe_sym;
    /// gfx11 (gfx1100, gfx1151) NT1 grouped symmetric IU4 MoE GEMMs; the E8P-I4 module is the gfx1151 Halo port (`--kernel qwen4_moe_sym_gfx11_e8p`).
    #[allow(dead_code, unused_imports)]
    #[path = "qwen4_moe_sym_gfx1100/mod.rs"]
    pub mod qwen4_moe_sym_gfx11;
    pub mod qwen4_mq4_x4_gfx11;
    pub mod qwen4_mq6_x4;
    pub mod qwen4_mq6_x4_gfx11;
    pub mod qwen4_mq6_x4_gfx11_hcw;
    pub mod qwen4_shared_gu_gfx11;
    pub mod qwen4_slab_union_plan;
    pub mod spec_accept_pm;
    pub mod spec_sampling_rows_pm;
    /// gfx942 (CDNA3) kernels built with `crate::cdna::builder`.
    pub mod cdna {
        pub mod fp8_act_producer;
        pub mod fp8_moe_gate_up;
        pub mod qsa_fp8_kv;
    }
}
#[cfg(feature = "toolchain")]
pub mod audit;
#[cfg(feature = "toolchain")]
mod cert_cache;
#[cfg(feature = "toolchain")]
pub mod cost_lint;
#[cfg(feature = "toolchain")]
pub mod ledger_replay;
#[cfg(feature = "toolchain")]
pub mod pm_check;
#[cfg(feature = "toolchain")]
pub mod toolchain;
pub use arch::Arch;
use hazard::{DelayAlu, Gfx11Hazards, Gfx12Sgpr, HazardProof, Pipeline};
use insn::{Instruction, MemoryClass, Program};
use lds::{Lds, Transition};
use ledger::{Counter, Ledger, Reason, WaitProof};
pub use plan::{BuilderProof, Emitted, IsaShape, KernargLayout, KernelSpec, MemoryScope};
pub use reg::{RegPlan, S, V};
use sha2::{Digest, Sha256};

/// The checked builder. The program, its wait ledger and its LDS slot
/// machine are reachable only through the checked entry points: raw
/// `push` takes plain instructions, branches and the program end go
/// through the typed core (`author`, behind an `Auth`) or the untyped
/// `control`, and the program and ledger are read-only from outside.
#[derive(Clone)]
pub struct Builder {
    pub spec: KernelSpec,
    pub regs: RegPlan,
    program: Program,
    ledger: Ledger,
    lds: Lds,
    pub waits: Vec<WaitProof>,
    pub hazards: Vec<HazardProof>,
    pub clauses: Vec<plan::ClauseProof>,
    pub barriers: Vec<plan::BarrierProof>,
    pub loop_fixpoints: Vec<plan::LoopFixpoint>,
    hazard: Gfx12Sgpr,
    gfx11_hazard: Gfx11Hazards,
    delay_alu: Option<DelayAlu>,
    labels: Vec<String>,
    current_label: String,
    lds_access_allowed: bool,
    previous_wmma_dst: Vec<reg::RegRef>,
    workgroup_id_z: bool,
    pending_barrier: Option<Vec<Transition>>,
    /// Driven by `peacemaker_author::Workgroup`: LDS, barriers, branches and
    /// loops go through the typed core (`author`), and the untyped entry
    /// points refuse.
    seal: peacemaker_author::Seal,
    /// Kernel exits reserved by the typed core: placed only with `s_endpgm` after them.
    exits: Vec<String>,
    /// Code falls through to the current point: false after a typed `s_branch`
    /// until a join brings in the state of a branch to this point.
    reachable: bool,
}
/// Loop bodies emitted before the back-edge state must have converged.
const MAX_LOOP_RUNS: u8 = 8;
// RDNA4 ISA §5.7.1: bits 15:8 are LOADcnt, bits 7:0 are DScnt.
fn load_ds_wait_imm(load_count: u8, ds_count: u8) -> u16 {
    (u16::from(load_count) << 8) | u16::from(ds_count)
}
/// The two counters one combined wait retires: gfx12 `s_wait_loadcnt_dscnt`
/// (LOADcnt, DScnt) or the gfx11 `s_waitcnt` fields (VMcnt, LGKMcnt; RDNA3
/// ISA §16.5: VM[15:10], LGKM[9:4], EXP[2:0]).
fn combined_wait(arch: Arch, first: u8, second: u8) -> (Counter, Counter, String) {
    if arch.gfx12() {
        (
            Counter::Load,
            Counter::Ds,
            format!(
                "s_wait_loadcnt_dscnt {:#x}",
                load_ds_wait_imm(first, second)
            ),
        )
    } else {
        (
            Counter::Vm,
            Counter::Lgkm,
            format!("s_waitcnt vmcnt({first}) lgkmcnt({second})"),
        )
    }
}
impl Builder {
    pub fn new(spec: KernelSpec, regs: RegPlan) -> Self {
        let arch = spec.arch;
        Self {
            spec,
            regs,
            program: Program {
                arch,
                instructions: vec![],
            },
            ledger: Ledger::default(),
            lds: Lds::default(),
            waits: vec![],
            hazards: vec![],
            clauses: vec![],
            barriers: vec![],
            loop_fixpoints: vec![],
            hazard: Gfx12Sgpr::default(),
            gfx11_hazard: Gfx11Hazards::for_arch(arch),
            delay_alu: None,
            labels: vec!["entry".into()],
            current_label: "entry".into(),
            lds_access_allowed: false,
            workgroup_id_z: false,
            previous_wmma_dst: vec![],
            pending_barrier: None,
            seal: Default::default(),
            exits: vec![],
            reachable: true,
        }
    }
    /// The program so far (read-only).
    pub fn program(&self) -> &Program {
        &self.program
    }
    /// The wait ledger at the current point (read-only).
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }
    /// Emit `s_delay_alu` issue hints before dependent VALU instructions (see
    /// `hazard::DelayAlu`). Opt-in, so existing kernels keep their bytes.
    pub fn enable_delay_alu(&mut self) {
        self.delay_alu = Some(DelayAlu::default())
    }
    /// Enable the descriptor's workgroup Z input without changing instructions.
    ///
    /// On gfx11 it follows the enabled X/Y system SGPRs: with two user
    /// SGPRs and Y enabled, Z is `s4`. Declare that live input in the register
    /// plan. On gfx1201 Z is the upper 16 bits of `ttmp7`, not an ordinary
    /// SGPR. The emitter must extract it and mask Y to the lower 16 bits.
    /// Disabled by default so existing kernel images retain their bytes.
    pub fn enable_workgroup_id_z(&mut self) {
        self.workgroup_id_z = true;
    }
    pub fn label(&mut self, name: &str) -> Result<(), String> {
        if self.exits.iter().any(|e| e == name) {
            return Err(format!(
                "{name} is a kernel exit: only Workgroup::end places it"
            ));
        }
        self.place_label(name)
    }
    fn place_label(&mut self, name: &str) -> Result<(), String> {
        if self.labels.iter().any(|l| l == name) {
            return Err(format!("duplicate label {name}"));
        }
        self.labels.push(name.into());
        self.current_label = name.into();
        if let Some(d) = &mut self.delay_alu {
            d.label()
        }
        self.program
            .instructions
            .push(Instruction::new(format!("{name}:"), vec![], vec![]));
        Ok(())
    }
    fn emit_wait(&mut self, c: Counter, n: u8, reason: Reason) -> Result<(), String> {
        let entries = Ledger::wait_instruction(self.spec.arch, &[(c, n, reason.clone())])?;
        for (_, _, text, _) in entries {
            let pc_index = self.program.instructions.len();
            self.program
                .instructions
                .push(Instruction::new(text.clone(), vec![], vec![]));
            self.waits.push(WaitProof {
                pc_index,
                insn: text,
                counter: c,
                count: n,
                reason: reason.clone(),
            })
        }
        self.ledger.wait(c, n);
        Ok(())
    }
    pub fn wait(&mut self, c: Counter, n: u8) -> Result<(), String> {
        self.emit_wait(c, n, Reason::Barrier)
    }
    fn emit_required(&mut self, mut required: Vec<(Counter, u8, Reason)>) -> Result<(), String> {
        let (first_counter, second_counter, _) = combined_wait(self.spec.arch, 0, 0);
        let first = required
            .iter()
            .position(|(counter, _, _)| *counter == first_counter);
        let second = required
            .iter()
            .position(|(counter, _, _)| *counter == second_counter);
        if let (Some(first), Some(second)) = (first, second) {
            let (first_count, first_reason) = (required[first].1, required[first].2.clone());
            let (second_count, second_reason) = (required[second].1, required[second].2.clone());
            let (_, _, text) = combined_wait(self.spec.arch, first_count, second_count);
            let pc_index = self.program.instructions.len();
            self.program
                .instructions
                .push(Instruction::new(text.clone(), vec![], vec![]));
            self.waits.push(WaitProof {
                pc_index,
                insn: text.clone(),
                counter: first_counter,
                count: first_count,
                reason: first_reason,
            });
            self.waits.push(WaitProof {
                pc_index,
                insn: text,
                counter: second_counter,
                count: second_count,
                reason: second_reason,
            });
            self.ledger.wait(first_counter, first_count);
            self.ledger.wait(second_counter, second_count);
            required
                .retain(|(counter, _, _)| *counter != first_counter && *counter != second_counter);
        }
        for (counter, count, reason) in required {
            self.emit_wait(counter, count, reason)?
        }
        Ok(())
    }
    pub fn wait_all(&mut self) -> Result<(), String> {
        let required = self.ledger.clone().drain();
        self.emit_required(required)
    }
    /// Emit now the waits an instruction defining `regs` would need (the hazards of loads still
    /// in flight into them, of stores still reading them): the minimal counter waits of the
    /// ledger, nothing when there is none. A caller about to open a clause settles its members'
    /// destinations first, since a clause takes no waits inside.
    pub fn settle_defs(&mut self, regs: &[reg::RegRef]) -> Result<(), String> {
        let probe = Instruction::new("", regs.to_vec(), Vec::new());
        let required = self.ledger.required(&probe);
        self.emit_required(required)
    }
    /// gfx12 `s_wait_alu depctr_vm_vsrc(0)`: every issued VMEM store has read its
    /// sources, so they may be redefined without waiting for store completion.
    pub fn release_store_sources(&mut self) -> Result<(), String> {
        if !self.spec.arch.gfx12() {
            return Err("depctr_vm_vsrc requires gfx12".into());
        }
        let pc_index = self.program.instructions.len();
        let text = "s_wait_alu depctr_vm_vsrc(0)".to_string();
        self.program
            .instructions
            .push(Instruction::new(text.clone(), vec![], vec![]));
        self.hazards.push(HazardProof {
            pc_index,
            insn: text,
            rule: "VMEM store sources read (store-source WAR)".into(),
        });
        self.ledger.release_store_sources();
        Ok(())
    }
    /// A plain instruction (`Instruction::check_raw`): no branch, program
    /// end, barrier, clause or global visibility, and DS access only through
    /// `ds_load`/`ds_store`. The builder inserts its waits and hazard guards.
    pub fn push(&mut self, insn: Instruction) -> Result<(), String> {
        insn.check_raw()?;
        if matches!(
            insn.mnemonic(),
            "s_clause" | "global_inv" | "buffer_gl0_inv" | "buffer_gl1_inv"
        ) {
            return Err("clauses and global visibility require their builder methods".into());
        }
        self.emit(insn)
    }
    /// Untyped kernels' branches and program end (`s_branch`, `s_cbranch_*`,
    /// `s_endpgm`): the typed core emits its own, so a sealed builder refuses.
    pub fn control(&mut self, insn: Instruction) -> Result<(), String> {
        self.untyped()?;
        let m = insn.mnemonic();
        if !(m == "s_branch" || m.starts_with("s_cbranch_") || m == "s_endpgm")
            || insn.text.contains('\n')
        {
            return Err(format!("{:?} is not a branch or s_endpgm", insn.text));
        }
        self.emit(insn)
    }
    /// Every instruction goes through here: waits, hazard guards, ledger.
    fn emit(&mut self, insn: Instruction) -> Result<(), String> {
        insn.validate(self.spec.arch)?;
        if self
            .program
            .instructions
            .last()
            .is_some_and(|i| i.mnemonic() == "s_endpgm")
        {
            return Err("instruction after s_endpgm".into());
        }
        if insn.mnemonic().starts_with("ds_") && !self.lds_access_allowed {
            return Err("DS access must carry an LDS slot through ds_load/ds_store".into());
        }
        for r in insn.defs.iter().chain(&insn.uses) {
            self.regs
                .verify_access(*r, &self.current_label, &self.labels)?
        }
        self.emit_required(self.ledger.required(&insn))?;
        let mnemonic = insn.mnemonic();
        let pipe = if matches!(
            insn.memory,
            Some(MemoryClass::VmemLoad | MemoryClass::VmemStore)
        ) {
            Pipeline::Vmem
        } else if insn.memory == Some(MemoryClass::SmemLoad) {
            Pipeline::Smem
        } else if mnemonic.starts_with('s') {
            Pipeline::Salu
        } else if mnemonic.starts_with('v') {
            Pipeline::Valu
        } else {
            Pipeline::Ds
        };
        if self.spec.arch.gfx12() {
            let waits = self.hazard.step(pipe, &insn.uses, &insn.defs, false, false);
            if !waits.is_empty() {
                let text = format!("s_wait_alu {}", waits.join(" | "));
                let pc_index = self.program.instructions.len();
                self.program
                    .instructions
                    .push(Instruction::new(text.clone(), vec![], vec![]));
                self.hazards.push(HazardProof {
                    pc_index,
                    insn: text,
                    rule: "gfx12 SGPR read-after-write".into(),
                })
            }
        }
        if !self.spec.arch.gfx12() {
            for text in self
                .gfx11_hazard
                .step(pipe, mnemonic, &insn.uses, &insn.defs)
            {
                let pc_index = self.program.instructions.len();
                self.program
                    .instructions
                    .push(Instruction::new(text.clone(), vec![], vec![]));
                self.hazards.push(HazardProof {
                    pc_index,
                    insn: text,
                    rule: "gfx1100 wave32 TRANS-use (VALUTransUseHazard)".into(),
                });
            }
        }
        if mnemonic.starts_with("v_wmma_") || mnemonic.starts_with("v_swmmac_") {
            let dst = *insn
                .defs
                .first()
                .ok_or("WMMA destination is missing from instruction defs")?;
            if insn.uses.len() < 2 {
                return Err("WMMA A/B operands are missing from instruction uses".into());
            }
            if self.previous_wmma_dst.iter().any(|&old| {
                insn.uses[..2].iter().any(|source| old.overlaps(*source))
                    || (mnemonic.starts_with("v_swmmac_")
                        && insn.uses.get(2).is_some_and(|index| old.overlaps(*index)))
            }) {
                let pc_index = self.program.instructions.len();
                self.program
                    .instructions
                    .push(Instruction::new("v_nop", vec![], vec![]));
                self.hazards.push(HazardProof {
                    pc_index,
                    insn: "v_nop".into(),
                    rule: "WMMA destination feeds next WMMA A/B or SWMMAC index".into(),
                });
                if let Some(d) = &mut self.delay_alu {
                    d.step("v_nop", &[], &[]);
                }
            }
            self.previous_wmma_dst = vec![dst];
        } else if pipe == Pipeline::Valu {
            self.previous_wmma_dst.clear()
        }
        // gfx11 WMMA chains accumulate in the matrix core, where LLVM emits no
        // `s_delay_alu`: a WMMA counts as an issued VALU but neither takes a hint
        // nor becomes a hint producer.
        let matrix = mnemonic.starts_with("v_wmma_") || mnemonic.starts_with("v_swmmac_");
        if pipe == Pipeline::Valu && matrix && !self.spec.arch.gfx12() {
            if let Some(d) = &mut self.delay_alu {
                d.step(mnemonic, &[], &[]);
            }
        } else if pipe == Pipeline::Valu {
            if let Some(text) = self
                .delay_alu
                .as_mut()
                .and_then(|d| d.step(mnemonic, &insn.uses, &insn.defs))
            {
                let pc_index = self.program.instructions.len();
                self.program
                    .instructions
                    .push(Instruction::new(text.clone(), vec![], vec![]));
                self.hazards.push(HazardProof {
                    pc_index,
                    insn: text,
                    rule: "VALU dependency issue hint".into(),
                });
            }
        }
        self.ledger.record(self.spec.arch, &insn);
        if mnemonic == "s_endpgm" {
            self.ledger = Ledger::default()
        }
        self.program.instructions.push(insn);
        Ok(())
    }
    fn untyped(&self) -> Result<(), String> {
        if self.seal.is_sealed() {
            Err(
                "typed kernel: LDS, barriers, branches and loops go through peacemaker-author"
                    .into(),
            )
        } else {
            Ok(())
        }
    }
    /// Declare one LDS slot of an untyped kernel (ids follow declaration order).
    pub fn lds_slot(&mut self, name: &str, base: u32, len: u32) -> Result<usize, String> {
        self.untyped()?;
        self.lds.add(name, base, len)
    }
    /// End an untyped kernel's slot layout (`Lds::relayout`).
    pub fn lds_relayout(&mut self) -> Result<(), String> {
        self.untyped()?;
        self.lds.relayout()
    }
    pub fn ds_store(&mut self, slot: usize, insn: Instruction) -> Result<(), String> {
        self.untyped()?;
        self.ds_store_slot(slot, insn)
    }
    fn ds_store_slot(&mut self, slot: usize, insn: Instruction) -> Result<(), String> {
        if insn.memory != Some(MemoryClass::DsStore) || !insn.mnemonic().starts_with("ds_store") {
            return Err("ds_store requires an LDS store instruction".into());
        }
        let old = self.lds.clone();
        self.lds.store(slot)?;
        self.lds_access_allowed = true;
        let result = self.push(insn);
        self.lds_access_allowed = false;
        if result.is_err() {
            self.lds = old
        }
        result
    }
    pub fn ds_load(&mut self, slot: usize, insn: Instruction) -> Result<(), String> {
        self.untyped()?;
        self.ds_load_slot(slot, insn)
    }
    fn ds_load_slot(&mut self, slot: usize, insn: Instruction) -> Result<(), String> {
        if insn.memory != Some(MemoryClass::DsLoad) || !insn.mnemonic().starts_with("ds_load") {
            return Err("ds_load requires an LDS load instruction".into());
        }
        let old = self.lds.clone();
        self.lds.load(slot)?;
        self.lds_access_allowed = true;
        let result = self
            .push(insn)
            .and_then(|()| self.ledger.tag_lds_load(slot));
        self.lds_access_allowed = false;
        if result.is_err() {
            self.lds = old
        }
        result
    }
    /// Before a barrier: drain LDS stores (RAW, the barrier publishes them),
    /// else retire the pending DS loads of every slot the barrier retires
    /// (WAR: after the barrier another wave may store to that slot, and on a
    /// WGP the other CU's store can land before this wave's queued load reads).
    /// `s_barrier` waits for no memory counter of its own.
    fn barrier_waits(&mut self, transitions: &[Transition], ds: Counter) -> Result<(), String> {
        if self.ledger.pending_stores() {
            return self.emit_wait(ds, 0, Reason::Barrier);
        }
        let retired: Vec<usize> = transitions
            .iter()
            .filter_map(|t| match t {
                Transition::Retire(id) => Some(*id),
                Transition::Ready(_) => None,
            })
            .collect();
        match self.ledger.retire_loads_of(&retired) {
            Some((counter, count)) => self.emit_wait(counter, count, Reason::Barrier),
            None => Ok(()),
        }
    }
    /// Cross-lane exchange on the LDS crossbar (`ds_swizzle_b32`, or `ds_bpermute_b32`
    /// with its lane-byte address and data VGPR uses): it counts on LGKM like a DS load
    /// but reads and writes no LDS memory, so it carries no slot.
    pub fn ds_crosslane(&mut self, insn: Instruction) -> Result<(), String> {
        if insn.memory != Some(MemoryClass::DsLoad)
            || !matches!(insn.mnemonic(), "ds_swizzle_b32" | "ds_bpermute_b32")
        {
            return Err("ds_crosslane requires a ds_swizzle_b32 or ds_bpermute_b32 DS-load-class instruction".into());
        }
        self.lds_access_allowed = true;
        let result = self.push(insn);
        self.lds_access_allowed = false;
        result
    }
    pub fn vopd(&mut self, x: vopd::VopdOp, y: vopd::VopdOp) -> Result<(), String> {
        let insn = vopd::packet(self.spec.arch, x, y)?;
        self.push(insn)
    }
    pub fn vopd_ff<
        const DX: u8,
        const DY: u8,
        const AX: u8,
        const AY: u8,
        const BX: u8,
        const BY: u8,
    >(
        &mut self,
        x: vopd::VopdF32Op<reg::Vp<DX>, vopd::Src0<AX>, reg::Vb<BX>>,
        y: vopd::VopdF32Op<reg::Vp<DY>, vopd::Src0<AY>, reg::Vb<BY>>,
    ) -> Result<(), String>
    where
        reg::Vp<DX>: reg::OppositeParity<DY>,
        reg::Vb<AX>: reg::DistinctBanks<AY>,
        reg::Vb<BX>: reg::DistinctBanks<BY>,
    {
        self.push(vopd::typed(self.spec.arch, x, y)?)
    }
    pub fn vopd_ff_shared_src1<const DX: u8, const DY: u8, const AX: u8, const AY: u8>(
        &mut self,
        x: vopd::VopdF32Op<reg::Vp<DX>, vopd::Src0<AX>, V<1>>,
        y: vopd::VopdF32Op<reg::Vp<DY>, vopd::Src0<AY>, V<1>>,
    ) -> Result<(), String>
    where
        reg::Vp<DX>: reg::OppositeParity<DY>,
        reg::Vb<AX>: reg::DistinctBanks<AY>,
    {
        self.push(vopd::shared_src1(self.spec.arch, x, y)?)
    }
    pub fn clause(
        &mut self,
        body: impl FnOnce(&mut Builder) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut draft = self.clone();
        let start = draft.program.instructions.len();
        body(&mut draft)?;
        let members = &draft.program.instructions[start..];
        let valid = members.len() >= 2
            && members.len() <= 32
            && members
                .iter()
                .all(|i| i.memory == members[0].memory && i.memory.is_some())
            && members.iter().enumerate().all(|(i, insn)| {
                !members[..i].iter().any(|prev| {
                    prev.defs
                        .iter()
                        .any(|r| insn.uses.iter().chain(&insn.defs).any(|u| r.overlaps(*u)))
                })
            });
        if !valid {
            return Err(
                "clause requires 2..=32 independent instructions of one memory class without waits"
                    .into(),
            );
        }
        let len = members.len();
        let class = format!("{:?}", members[0].memory.unwrap());
        draft
            .program
            .instructions
            .insert(start, insn::Sop::Clause(len as u8).encode(self.spec.arch)?);
        draft.clauses.push(plan::ClauseProof { start, len, class });
        *self = draft;
        Ok(())
    }
    pub fn barrier(&mut self, transitions: &[Transition]) -> Result<(), String> {
        self.barrier_with_scope(transitions, MemoryScope::LdsOnly)
    }
    pub fn barrier_signal(&mut self, transitions: &[Transition]) -> Result<(), String> {
        self.untyped()?;
        self.signal(transitions)
    }
    fn signal(&mut self, transitions: &[Transition]) -> Result<(), String> {
        if !self.spec.arch.gfx12() {
            return Err("split barriers require gfx12".into());
        }
        self.barrier_waits(transitions, Counter::Ds)?;
        self.lds
            .barrier_signal(transitions, !self.ledger.pending_stores())?;
        self.pending_barrier = Some(transitions.to_vec());
        self.program
            .instructions
            .push(Instruction::new("s_barrier_signal -1", vec![], vec![]));
        Ok(())
    }
    pub fn barrier_wait(&mut self) -> Result<(), String> {
        self.untyped()?;
        self.arrive()
    }
    fn arrive(&mut self) -> Result<(), String> {
        let transitions = self
            .pending_barrier
            .take()
            .ok_or("barrier wait without signal")?;
        self.lds.barrier_wait()?;
        let pc_index = self.program.instructions.len();
        self.program
            .instructions
            .push(Instruction::new("s_barrier_wait 0xffff", vec![], vec![]));
        self.barriers.push(plan::BarrierProof {
            pc_index,
            transitions: transitions.iter().map(|t| format!("{t:?}")).collect(),
        });
        Ok(())
    }
    pub fn barrier_with_scope(
        &mut self,
        transitions: &[Transition],
        scope: MemoryScope,
    ) -> Result<(), String> {
        self.untyped()?;
        self.full_barrier(transitions, scope)
    }
    fn full_barrier(
        &mut self,
        transitions: &[Transition],
        scope: MemoryScope,
    ) -> Result<(), String> {
        if self.spec.arch.gfx12() {
            self.signal(transitions)?;
            self.arrive()?
        } else {
            self.barrier_waits(transitions, Counter::Lgkm)?;
            self.lds
                .barrier(transitions, !self.ledger.pending_stores())?;
            let pc_index = self.program.instructions.len();
            self.program
                .instructions
                .push(Instruction::new("s_barrier", vec![], vec![]));
            self.barriers.push(plan::BarrierProof {
                pc_index,
                transitions: transitions.iter().map(|t| format!("{t:?}")).collect(),
            });
        }
        if scope == MemoryScope::Workgroup {
            let invalidate = if self.spec.arch.gfx12() {
                "global_inv scope:SCOPE_SE"
            } else {
                "buffer_gl0_inv"
            };
            self.program
                .instructions
                .push(Instruction::new(invalidate, vec![], vec![]));
        }
        Ok(())
    }
    /// The body is emitted once, so its waits and guards must hold on every
    /// iteration. The back-edge ledger must equal the entry ledger (the same
    /// pending operations in the same order: loads issued for the next
    /// iteration are allowed when the code before the loop issues the same
    /// ones). The hazard trackers and LDS slots at the head are the entry's
    /// joined with the back edge's: the body is re-emitted from that join
    /// until its back edge adds nothing, and that last emission is kept.
    pub fn loop_(
        &mut self,
        head: &str,
        body: impl Fn(&mut Builder) -> Result<(), String>,
    ) -> Result<(), String> {
        self.untyped()?;
        self.fixpoint_loop(head, body)
    }
    fn fixpoint_loop(
        &mut self,
        head: &str,
        body: impl Fn(&mut Builder) -> Result<(), String>,
    ) -> Result<(), String> {
        let entry = self.ledger.shape();
        let mut start = self.clone();
        for runs in 1..=MAX_LOOP_RUNS {
            let mut run = start.clone();
            run.label(head)?;
            body(&mut run)?;
            if run.ledger.shape() != entry {
                return Err(format!(
                    "loop {head}: back-edge ledger must equal the entry ledger"
                ));
            }
            // The head of the next run: this run's back-edge ledger (equal to the
            // entry's up to ids), everything else joined with its back edge.
            let mut next = start.clone();
            next.ledger = run.ledger.clone();
            let grew = next
                .join_back_edge(&run)
                .map_err(|e| format!("loop {head} back edge: {e}"))?;
            // At least two runs: the second runs from the back-edge ledger.
            if runs >= 2 && !grew {
                *self = run;
                self.loop_fixpoints.push(plan::LoopFixpoint {
                    head: head.into(),
                    iterations: runs,
                });
                return Ok(());
            }
            start = next;
        }
        Err(format!("loop {head}: hazard and LDS state did not reach a fixed point in {MAX_LOOP_RUNS} emissions"))
    }
    /// Join a loop's back-edge state into this head state; true when the
    /// head changed (the body must be emitted again from it).
    fn join_back_edge(&mut self, back: &Builder) -> Result<bool, String> {
        let before = (
            self.lds.clone(),
            self.hazard.clone(),
            self.gfx11_hazard.clone(),
            self.previous_wmma_dst.len(),
        );
        self.lds.join(&back.lds)?;
        self.join_hazards(&back.hazard, &back.gfx11_hazard, &back.previous_wmma_dst);
        Ok(self.lds != before.0
            || self.hazard != before.1
            || self.gfx11_hazard != before.2
            || self.previous_wmma_dst.len() != before.3)
    }
    /// Join another control path's hazard trackers into this point (a write
    /// pending a guard on either path is pending here).
    pub(crate) fn join_hazards(
        &mut self,
        sgpr: &Gfx12Sgpr,
        trans: &Gfx11Hazards,
        wmma: &[reg::RegRef],
    ) {
        self.hazard.join(sgpr);
        self.gfx11_hazard.join(trans);
        for r in wmma {
            if !self.previous_wmma_dst.contains(r) {
                self.previous_wmma_dst.push(*r)
            }
        }
    }
    /// Finish only after every declared lifetime has both endpoints and all live aliases are disjoint.
    pub fn finish(self) -> Result<Emitted, String> {
        if self.pending_barrier.is_some() {
            return Err("unmatched barrier signal".into());
        }
        if let Some(e) = self.exits.iter().find(|e| !self.labels.contains(e)) {
            return Err(format!("kernel exit {e} is never placed"));
        }
        if self
            .program
            .instructions
            .last()
            .is_none_or(|i| i.mnemonic() != "s_endpgm")
        {
            return Err("kernel must end with s_endpgm".into());
        }
        if !self.ledger.is_empty() {
            return Err(
                "outstanding memory operations: wait or end the kernel before finish".into(),
            );
        }
        self.regs.verify_lifetimes(&self.labels)?;
        let next_free_sgpr =
            emit::sgpr_high_water(&self.spec, &self.regs, &self.program, self.workgroup_id_z);
        if self.regs.next_free_vgpr() > self.regs.vgpr_budget
            || next_free_sgpr > self.regs.sgpr_budget
        {
            return Err("register plan exceeds budget".into());
        }
        let text = emit::assembly_with_workgroup_id_z(
            &self.spec,
            &self.regs,
            &self.program,
            self.workgroup_id_z,
            next_free_sgpr,
        )?;
        let hash = format!("{:x}", Sha256::digest(text.as_bytes()));
        let mut shape = IsaShape {
            instructions: self.program.instructions.len(),
            next_free_vgpr: self.regs.next_free_vgpr(),
            next_free_sgpr,
            waits: self.waits.len() + self.hazards.len(),
            barriers: self.barriers.len(),
            ..Default::default()
        };
        for i in &self.program.instructions {
            let name = i.mnemonic();
            if name.starts_with("v_dual_") {
                shape.vopd_pairs += 1;
                shape.valu_slots += 1
            } else if name.starts_with("v_wmma_") {
                shape.wmma += 1
            } else if name.starts_with('v') {
                shape.valu_slots += 1
            }
            if name.starts_with("ds_") {
                shape.ds += 1
            }
            if name.starts_with("buffer_") || name.starts_with("global_") {
                shape.vmem += 1
            }
        }
        let proof = BuilderProof {
            kernel_id: self.spec.kernel_id,
            variant: self.spec.variant,
            arch: self.spec.arch,
            builder_crate_version: env!("CARGO_PKG_VERSION").into(),
            builder_git_sha: option_env!("HIPFIRE_BUILDER_GIT_SHA")
                .unwrap_or("unknown")
                .into(),
            reg_plan: self.regs.ranges,
            next_free_vgpr: shape.next_free_vgpr,
            next_free_sgpr: shape.next_free_sgpr,
            system_sgpr_workgroup_id_z: self.workgroup_id_z,
            waits: self.waits,
            hazards: self.hazards,
            clauses: self.clauses,
            vopd_pairs: shape.vopd_pairs,
            lds_slots: self.lds.all_slots(),
            barriers: self.barriers,
            loop_fixpoints: self.loop_fixpoints,
            forbidden_mnemonics_checked: vec!["s_waitcnt (gfx12)".into(), "scratch_*".into()],
            s_text_sha256: hash,
        };
        Ok(Emitted {
            s_text: text,
            proof,
            shape,
        })
    }
}

#[cfg(test)]
mod wait_encoding_tests {
    use super::load_ds_wait_imm;

    #[test]
    fn asymmetric_load_ds_wait_fields() {
        assert_eq!(load_ds_wait_imm(7, 3), 0x703);
    }
}
