// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Certification of a lifted gfx942 kernel (`peacemaker_lift::cdna`): the typed instruction
//! stream of a hipcc-compiled kernel is replayed through the same two checkers the CDNA
//! builder's output goes through, without inserting or changing anything.
//!
//! * **Hazards**: [`super::hazard::check`] over the typed instructions exactly as decoded
//!   (branches keep their encoded offsets; no `s_nop` is added, no instruction rewritten).
//!   Every reported violation is an error.
//! * **Wait ledger**: strict CFG replay over a [`Program`] derived from the
//!   typed stream, path-sensitive over the kernel's CFG. The ledger metadata is not parsed
//!   from text: defs/uses come from [`Inst::footprint`], the memory class from
//!   [`Inst::memory`]. Labels are placed at the (validated) branch targets and branches
//!   name them; `s_waitcnt` is rendered from the codec's decoded fields (`waitcnt_fields`),
//!   so nonzero `vmcnt`/`lgkmcnt` waits retire by position under the ledger's own
//!   in-order rules (gfx942: loads, stores and returning atomics share one in-order
//!   VM_CNT stream; DS operations are in order on LGKM_CNT and a SMEM operation in flight
//!   there makes every nonzero wait retire nothing). A CDNA `s_barrier` requires every DS
//!   operation of the wave to have retired (LDS RAW for pending writes, LDS WAR for
//!   pending reads; `ledger_cfg` module docs). `s_endpgm` ends a path. A pending store's
//!   data/address registers are not held until its counter retires (`Ledger::required_replay`
//!   and its ISA basis in the `ledger` module docs): overwriting them needs no counter
//!   wait, while the store still occupies its counter slot and blocks a barrier, and the
//!   ISA's wait-state rule for wide VMEM store data is enforced by the hazard pass.
//!
//! **Static branch proofs** (strict only): `ledger_program` also derives, per ledger
//! instruction, an `EdgeChoice` from block-local typed scalar facts
//! (`control::transfer` / `control::branch_choice`; `ledger_control` module). The facts start
//! unknown at instruction 0, at every branch target of the original CFG (reachable or not)
//! and after every branch or `s_endpgm`; nothing flows between blocks, so a choice holds for
//! every state entering its block (the P1b `s_mov_b64 s[0:1],0; s_and_b64 s[0:1],s[0:1],exec;
//! s_cselect_b32 s0,1,0; s_cmp_lg_u32 s0,1; s_cbranch_scc1` gate always takes its branch).
//! `certify_stream` hands that exact table to `ledger_cfg::verify_with_branch_proofs`, which
//! prunes only the impossible edge; the layout, target, fall-off and hazard validation still
//! run over the full unchanged typed stream first, and inserted labels map to `Both`. No
//! public API accepts caller-supplied proofs.
//!
//! Fail-closed, never skipped: an instruction the ledger cannot model is an error, not a
//! pass. That covers `s_setpc_b64`, `s_trap`, `s_sendmsg`, a branch off an instruction
//! boundary or past the code, a reachable fall-through off the end, FLAT-segment and
//! SCRATCH accesses (FLAT counts on VM_CNT and LGKM_CNT and may address LDS; SCRATCH needs
//! a private segment), GDS operations, and memory operations whose data/destination
//! registers are special-register files (the ledger tracks SGPRs, VGPRs and AGPRs; AGPRs
//! are a separate bank, `Kind::A`, with the same wait/hazard accounting as VGPRs).
//!
//! Not modelled (reported, not hidden): global-memory visibility across a barrier,
//! unreachable instructions are hazard-checked but not ledger-walked (their count is in
//! the evidence), and every wait-state/counter rule is the shared checkers' and carries
//! their `[UNVERIFIED: confirm on MI300X]` caveats.

use super::{control, hazard};
use crate::arch::Arch;
use crate::insn::{Instruction, MemoryClass, Program};
use crate::ledger_cfg;
use crate::ledger_control::{EdgeChoice, Facts};
use crate::reg::{Kind, RegRef};
use peacemaker_cdna::enc::flat::FlatSeg;
use peacemaker_cdna::enc::sop::{waitcnt_fields, Sop1Op, SoppOp};
use peacemaker_cdna::inst::{Footprint, MemKind, RegFile, RegSpan};
use peacemaker_cdna::Inst;
use peacemaker_lift::cdna::{Instruction as Typed, Kernel};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};

const CDNA: peacemaker_cdna::Arch = peacemaker_cdna::Arch::Gfx942;
/// Violations quoted in an error message (the count is always complete).
const SHOWN: usize = 8;

/// Certify one lifted kernel: hazard check and CFG wait-ledger replay over its typed
/// instructions. Returns the evidence (`checked`, `blocking: 0`, hazard and ledger
/// statistics) or the first class of failure.
pub fn certify(kernel: &Kernel) -> Result<Value, String> {
    let evidence = certify_stream(&kernel.image.name, &kernel.instructions)?;
    let dwords = evidence["dwords"].as_u64().unwrap_or(0) as usize;
    if dwords * 4 != kernel.image.code.len() {
        return Err(format!(
            "{}: the typed instructions cover {} bytes but the kernel code is {} bytes",
            kernel.image.name,
            dwords * 4,
            kernel.image.code.len()
        ));
    }
    Ok(evidence)
}

/// [`certify`] for a bare typed stream: `instructions` must be the whole code of one
/// kernel in order with their true dword offsets.
pub fn certify_stream(name: &str, instructions: &[Typed]) -> Result<Value, String> {
    for typed in instructions {
        super::require_verified(CDNA, &typed.inst)?;
        if let Some(reason) = typed.inst.unsupported() {
            return Err(format!("{name}: dword {}: {reason}", typed.dword_offset));
        }
    }
    let layout = layout(name, instructions)?;
    let insts: Vec<Inst> = instructions.iter().map(|t| t.inst.clone()).collect();
    let cfg = cfg(name, &insts, &layout)?;

    let violations = hazard::check(CDNA, &insts);
    if !violations.is_empty() {
        return Err(format!(
            "{name}: {} CDNA hazard violation(s): {}{}",
            violations.len(),
            violations
                .iter()
                .take(SHOWN)
                .cloned()
                .collect::<Vec<_>>()
                .join(" | "),
            if violations.len() > SHOWN {
                " | ..."
            } else {
                ""
            }
        ));
    }

    let (program, counts, choices) = ledger_program(name, &insts, &cfg)?;
    let stats = ledger_cfg::verify_with_branch_proofs(&program, &[], &choices)
        .map_err(|e| format!("{name}: wait ledger replay: {e}"))?;

    let mut nops = 0u64;
    let mut nop_wait_states = 0u64;
    let mut mfma = 0u64;
    for inst in &insts {
        match inst {
            Inst::Sopp(p) if p.op == SoppOp::Nop => {
                nops += 1;
                nop_wait_states += u64::from(p.simm16 & 0xf) + 1;
            }
            Inst::Mai(m) if m.op.shape().is_some() => mfma += 1,
            _ => {}
        }
    }
    let unreachable = cfg.reachable.iter().filter(|r| !**r).count();
    Ok(json!({
        "checked": true,
        "findings": 0,
        "blocking": 0,
        "advisory": 0,
        "arch": "gfx942",
        "kernel": name,
        "instructions": insts.len(),
        "dwords": layout.dwords,
        "unreachable_instructions": unreachable,
        "bypasses": 0,
        "hazard": {
            "checker": "peacemaker_isa::cdna::hazard::check",
            "violations": 0,
            "inserted_nops": 0,
            "rewritten_instructions": 0,
            "s_nop": nops,
            "s_nop_wait_states": nop_wait_states,
            "mfma": mfma,
        },
        "ledger": {
            "verifier": "peacemaker_isa::ledger_cfg::verify_with_branch_proofs",
            // `steps` and `label_states` are informational exploration diagnostics
            // (`hip_twin::LEDGER_DIAGNOSTIC_FIELDS`), excluded from the pinned claims by
            // `hip_twin::pinned_report`; every other field here is pinned.
            "steps": stats.steps,
            "label_states": stats.label_states,
            "labels": cfg.targets.len(),
            "branches": counts.branches,
            "barriers": counts.barriers,
            "waitcnt": counts.waitcnt,
            "waitcnt_vmcnt_partial": counts.vm_partial,
            "waitcnt_lgkmcnt_partial": counts.lgkm_partial,
            "memory": {
                "vmem_load": counts.vmem_load,
                "vmem_store": counts.vmem_store,
                "vmem_atomic": counts.vmem_atomic,
                "ds_load": counts.ds_load,
                "ds_store": counts.ds_store,
                "smem": counts.smem,
            },
        },
    }))
}

/// Dword layout of the stream: offsets must be exactly those the encoder produces.
struct Layout {
    dwords: usize,
    /// Dword offset -> instruction index.
    index: HashMap<usize, usize>,
    /// Dword offset after instruction `i`.
    next: Vec<usize>,
}

fn layout(name: &str, stream: &[Typed]) -> Result<Layout, String> {
    if stream.is_empty() {
        return Err(format!("{name}: no instructions"));
    }
    let mut at = 0usize;
    let mut index = HashMap::with_capacity(stream.len());
    let mut next = Vec::with_capacity(stream.len());
    for (i, typed) in stream.iter().enumerate() {
        if typed.dword_offset != at {
            return Err(format!(
                "{name}: instruction {i} is at dword {} but the preceding instructions end at dword {at}",
                typed.dword_offset
            ));
        }
        let words = typed
            .inst
            .words(CDNA)
            .map_err(|e| format!("{name}: instruction {i} (dword {at}) does not encode: {e}"))?;
        index.insert(at, i);
        at += words.len();
        next.push(at);
    }
    Ok(Layout {
        dwords: at,
        index,
        next,
    })
}

struct Cfg {
    /// Branch target of each instruction (instruction index).
    branch: Vec<Option<usize>>,
    /// Instruction indexes that carry a label.
    targets: BTreeSet<usize>,
    reachable: Vec<bool>,
}

fn cfg(name: &str, insts: &[Inst], layout: &Layout) -> Result<Cfg, String> {
    let n = insts.len();
    let mut branch = vec![None; n];
    let mut targets = BTreeSet::new();
    for (i, inst) in insts.iter().enumerate() {
        if matches!(inst, Inst::Sop1(s) if s.op == Sop1Op::SetpcB64) {
            return Err(format!(
                "{name}: instruction {i}: s_setpc_b64 (indirect branch) cannot be followed"
            ));
        }
        if let Some(off) = inst.branch_offset() {
            let target = layout.next[i] as i64 + i64::from(off);
            let at = usize::try_from(target)
                .ok()
                .and_then(|t| layout.index.get(&t).copied())
                .ok_or_else(|| {
                    format!(
                        "{name}: instruction {i}: branch target dword {target} is not an instruction boundary inside the {}-dword code",
                        layout.dwords
                    )
                })?;
            branch[i] = Some(at);
            targets.insert(at);
        }
    }
    let mut reachable = vec![false; n];
    let mut stack = vec![0usize];
    while let Some(i) = stack.pop() {
        if std::mem::replace(&mut reachable[i], true) {
            continue;
        }
        let (falls, jump) = match &insts[i] {
            Inst::Sopp(p) if p.op == SoppOp::Endpgm => (false, None),
            Inst::Sopp(p) if p.op == SoppOp::Branch => (false, branch[i]),
            _ => (true, branch[i]),
        };
        if let Some(t) = jump {
            stack.push(t);
        }
        if falls {
            if i + 1 >= n {
                return Err(format!(
                    "{name}: instruction {i}: execution can fall off the end of the code"
                ));
            }
            stack.push(i + 1);
        }
    }
    Ok(Cfg {
        branch,
        targets,
        reachable,
    })
}

#[derive(Default)]
struct Counts {
    branches: u64,
    barriers: u64,
    waitcnt: u64,
    vm_partial: u64,
    lgkm_partial: u64,
    vmem_load: u64,
    vmem_store: u64,
    vmem_atomic: u64,
    ds_load: u64,
    ds_store: u64,
    smem: u64,
}

/// The ledger [`Program`] of the typed stream, its counts, and the static branch proofs
/// (`choices[pc]`, one per ledger instruction, labels included).
///
/// Each instruction's full typed [`Footprint`] is computed once and reused for both the
/// ledger defs/uses (`convert`, which then drops the special registers) and the scalar
/// control analysis (`control::transfer`, whose invalidation must see SCC/EXEC/VCC and every
/// SGPR a scalar or vector instruction may write). The scalar `Facts` are block-local: they
/// are reset at instruction 0, at every branch target (original CFG, reachable or not) and
/// after every branch or `s_endpgm`, so nothing is carried between blocks, along any edge or
/// around any loop. A conditional branch's choice is read from the facts of its own block
/// before its transfer; inserted labels and every other instruction are `Both`.
fn ledger_program(
    name: &str,
    insts: &[Inst],
    cfg: &Cfg,
) -> Result<(Program, Counts, Vec<EdgeChoice>), String> {
    let mut counts = Counts::default();
    let capacity = insts.len() + cfg.targets.len();
    let mut out = Vec::with_capacity(capacity);
    let mut choices = Vec::with_capacity(capacity);
    // One allocation for the stream; the register count is the architecture's (s0..s101).
    let mut facts = Facts::new(usize::from(CDNA.caps().sgprs));
    let mut block_start = true;
    for (i, inst) in insts.iter().enumerate() {
        let target = cfg.targets.contains(&i);
        if target {
            out.push(Instruction::new(format!("L{i}:"), vec![], vec![]));
            choices.push(EdgeChoice::Both);
        }
        if block_start || target {
            facts.reset_block();
        }
        let footprint = inst.footprint();
        let branch = cfg.branch[i];
        let conditional =
            branch.is_some() && !matches!(inst, Inst::Sopp(p) if p.op == SoppOp::Branch);
        let choice = if conditional {
            control::branch_choice(inst, &facts)
        } else {
            EdgeChoice::Both
        };
        control::transfer(inst, &footprint, &mut facts);
        out.push(
            convert(inst, &footprint, branch, &mut counts)
                .map_err(|e| format!("{name}: instruction {i} `{inst}`: {e}"))?,
        );
        choices.push(choice);
        block_start = branch.is_some() || matches!(inst, Inst::Sopp(p) if p.op == SoppOp::Endpgm);
    }
    Ok((
        Program {
            arch: Arch::Gfx942,
            instructions: out,
        },
        counts,
        choices,
    ))
}

fn convert(
    inst: &Inst,
    footprint: &Footprint,
    branch: Option<usize>,
    counts: &mut Counts,
) -> Result<Instruction, String> {
    if let Some(reason) = inst.unsupported() {
        return Err(reason);
    }
    let control = |text: String| -> Result<Instruction, String> {
        Ok(Instruction::new(text, vec![], vec![]))
    };
    if let Inst::Sopp(p) = inst {
        if let Some(target) = branch {
            counts.branches += 1;
            return control(format!("{} L{target}", p.op.name()));
        }
        return match p.op {
            SoppOp::Waitcnt => {
                let (vm, exp, lgkm) = waitcnt_fields(CDNA, p.simm16);
                let caps = CDNA.caps();
                counts.waitcnt += 1;
                if vm != 0 && vm < caps.vmcnt_max {
                    counts.vm_partial += 1;
                }
                if lgkm != 0 && lgkm < caps.lgkmcnt_max {
                    counts.lgkm_partial += 1;
                }
                control(format!(
                    "s_waitcnt vmcnt({vm}) expcnt({exp}) lgkmcnt({lgkm})"
                ))
            }
            SoppOp::Barrier => {
                counts.barriers += 1;
                control("s_barrier".into())
            }
            SoppOp::Endpgm => control("s_endpgm".into()),
            SoppOp::Trap | SoppOp::Sendmsg => Err(format!(
                "{} is outside the wait ledger's model",
                p.op.name()
            )),
            // s_nop, s_sleep, s_setprio, s_icache_inv: no counter or register effect.
            _ => control(inst.to_string()),
        };
    }
    let kind = inst.memory();
    let Some(kind) = kind else {
        return Ok(Instruction::new(
            inst.to_string(),
            tracked(&footprint.defs),
            tracked(&footprint.uses),
        ));
    };
    match inst {
        Inst::Flat(f) if f.seg != FlatSeg::Global => {
            return Err(format!(
                "{:?} segment access: FLAT counts on VM_CNT and LGKM_CNT and may address LDS, SCRATCH needs a private segment; only GLOBAL is modelled",
                f.seg
            ))
        }
        Inst::Ds(d) if d.gds => return Err("GDS operations are not modelled".into()),
        _ => {}
    }
    if let Some(bad) = footprint
        .defs
        .iter()
        .find(|d| !matches!(d.file, RegFile::Sgpr | RegFile::Vgpr | RegFile::Agpr))
    {
        return Err(format!(
            "memory destination {bad:?} is not a plain SGPR/VGPR/AGPR; the ledger cannot track it"
        ));
    }
    if let Some(bad) = footprint.uses.iter().find(|u| {
        !matches!(
            u.file,
            RegFile::Sgpr
                | RegFile::Vgpr
                | RegFile::Agpr
                | RegFile::Exec
                | RegFile::M0
                | RegFile::Scc
                | RegFile::Vcc
        )
    }) {
        return Err(format!(
            "memory source {bad:?} is not a plain SGPR/VGPR/AGPR; the ledger cannot track it"
        ));
    }
    let class = match kind {
        MemKind::VmemLoad | MemKind::VmemAtomicRtn => {
            if kind == MemKind::VmemLoad {
                counts.vmem_load += 1;
            } else {
                counts.vmem_atomic += 1;
            }
            MemoryClass::VmemLoad
        }
        MemKind::VmemStore | MemKind::VmemAtomic => {
            if kind == MemKind::VmemStore {
                counts.vmem_store += 1;
            } else {
                counts.vmem_atomic += 1;
            }
            MemoryClass::VmemStore
        }
        MemKind::DsLoad | MemKind::DsAtomicRtn | MemKind::DsXlane => {
            counts.ds_load += 1;
            MemoryClass::DsLoad
        }
        MemKind::DsStore | MemKind::DsAtomic => {
            counts.ds_store += 1;
            MemoryClass::DsStore
        }
        MemKind::SmemLoad => {
            counts.smem += 1;
            MemoryClass::SmemLoad
        }
        MemKind::SmemStore | MemKind::SmemAtomic | MemKind::SmemAtomicRtn => {
            return Err(format!("{inst}: {kind:?} has no CDNA ledger memory class"));
        }
    };
    Ok(Instruction::new(
        inst.to_string(),
        tracked(&footprint.defs),
        tracked(&footprint.uses),
    )
    .memory(class))
}

/// The SGPR/VGPR/AGPR spans, as ledger registers (AGPRs are their own bank, `Kind::A`).
fn tracked(spans: &[RegSpan]) -> Vec<RegRef> {
    spans
        .iter()
        .filter_map(|span| {
            let kind = match span.file {
                RegFile::Sgpr => Kind::S,
                RegFile::Vgpr => Kind::V,
                RegFile::Agpr => Kind::A,
                _ => return None,
            };
            Some(RegRef {
                kind,
                base: span.base,
                len: span.len,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdna::ops;
    use peacemaker_cdna::enc::sop::{Sop2Op, SopcOp};
    use peacemaker_cdna::enc::vop::Vop1Op;
    use peacemaker_cdna::Src;
    use peacemaker_lift::bundle::BundleCodec;
    use EdgeChoice::{Both, Fallthrough, Taken};
    use Src::{Int, Sgpr as s, Vgpr as v};

    /// The final P1b anchor (bytes identical to the pinned owner bundle).
    const P1B: &[u8] = include_bytes!("../../tests/fixtures/cdna/gfx942_p1b_wide.co");

    fn typed(insts: &[Inst]) -> Vec<Typed> {
        let mut at = 0;
        insts
            .iter()
            .map(|inst| {
                let it = Typed {
                    dword_offset: at,
                    inst: inst.clone(),
                };
                at += inst.words(CDNA).expect("encodes").len();
                it
            })
            .collect()
    }

    /// The same front half as `certify_stream` (layout, CFG), then the ledger conversion.
    fn analyze(insts: &[Inst]) -> Result<(Program, Counts, Vec<EdgeChoice>), String> {
        let layout = layout("t", &typed(insts))?;
        let cfg = cfg("t", insts, &layout)?;
        ledger_program("t", insts, &cfg)
    }
    fn convert_ok(insts: &[Inst]) -> (Program, Vec<EdgeChoice>) {
        let (program, _, choices) = analyze(insts).expect("converts");
        assert_eq!(choices.len(), program.instructions.len());
        (program, choices)
    }

    /// A branch at instruction `from` to instruction `to`; every instruction in these streams
    /// is one dword.
    fn br(op: SoppOp, from: usize, to: usize) -> Inst {
        ops::sp(op, (to as i64 - from as i64 - 1) as i16 as u16)
    }
    fn end() -> Inst {
        ops::sp(SoppOp::Endpgm, 0)
    }
    fn nop() -> Inst {
        ops::sp(SoppOp::Nop, 0)
    }
    /// `s_mov_b64 s[0:1], 0`
    fn zero_pair() -> Inst {
        ops::s1(Sop1Op::MovB64, s(0), Int(0))
    }
    /// `s_and_b64 s[0:1], s[0:1], exec`
    fn and_exec() -> Inst {
        ops::s2(Sop2Op::AndB64, s(0), s(0), Src::ExecLo)
    }
    /// `s_cselect_b32 s0, 1, 0`
    fn select() -> Inst {
        ops::s2(Sop2Op::CselectB32, s(0), Int(1), Int(0))
    }
    /// `s_cmp_lg_u32 s0, 1`
    fn compare() -> Inst {
        ops::sc(SopcOp::CmpLgU32, s(0), Int(1))
    }
    /// The five-instruction certificate of the P1b exit gate, first four instructions: leaves
    /// SCC provably 1.
    fn certificate() -> Vec<Inst> {
        vec![zero_pair(), and_exec(), select(), compare()]
    }

    #[test]
    fn zero_mask_certificate_proves_the_branch_taken() {
        let mut insts = certificate();
        insts.extend([br(SoppOp::CbranchScc1, 4, 6), nop(), end()]);
        let (program, choices) = convert_ok(&insts);
        // Ledger pcs: 0..=3 certificate, 4 branch, 5 nop, 6 `L6:`, 7 endpgm.
        assert_eq!(program.instructions[4].text, "s_cbranch_scc1 L6");
        assert_eq!(program.instructions[6].text, "L6:");
        assert_eq!(
            choices,
            [Both, Both, Both, Both, Taken, Both, Both, Both],
            "labels and non-conditionals are Both"
        );
        ledger_cfg::verify_with_branch_proofs(&program, &[], &choices).unwrap();

        // The opposite polarity falls through.
        let mut insts = certificate();
        insts.extend([br(SoppOp::CbranchScc0, 4, 6), nop(), end()]);
        let (_, choices) = convert_ok(&insts);
        assert_eq!(choices[4], Fallthrough);
    }

    #[test]
    fn facts_reset_after_every_branch_and_terminator_including_unreachable_code() {
        // The second branch needs the same SCC but starts a new block: Both.
        let mut insts = certificate();
        insts.extend([
            br(SoppOp::CbranchScc1, 4, 7),
            br(SoppOp::CbranchScc1, 5, 7),
            nop(),
            end(),
        ]);
        let (_, choices) = convert_ok(&insts);
        assert_eq!(choices[4], Taken);
        assert_eq!(choices[5], Both, "{choices:?}");

        // After an unconditional branch the textually following (unreachable) block starts
        // from unknown facts as well.
        let mut insts = certificate();
        insts.extend([
            br(SoppOp::Branch, 4, 6),
            br(SoppOp::CbranchScc1, 5, 6),
            end(),
        ]);
        let layout = layout("t", &typed(&insts)).unwrap();
        let cfg = cfg("t", &insts, &layout).unwrap();
        assert!(!cfg.reachable[5], "the conditional follows an s_branch");
        let (_, choices) = convert_ok(&insts);
        assert_eq!(choices[4], Both, "an unconditional branch carries no proof");
        assert_eq!(choices[5], Both, "{choices:?}");

        // After s_endpgm too.
        let mut insts = vec![end()];
        insts.extend(certificate());
        insts.extend([br(SoppOp::CbranchScc1, 5, 0), end()]);
        let (_, choices) = convert_ok(&insts);
        // pcs: 0 `L0:`, 1 endpgm, 2..=5 certificate, 6 branch, 7 endpgm; the certificate
        // restarts from unknown facts only at the endpgm boundary, so it still proves.
        assert_eq!(choices[6], Taken, "{choices:?}");
    }

    #[test]
    fn a_branch_target_in_the_middle_of_the_constant_sequence_resets_the_facts() {
        // Instruction 4 (the compare) is a branch target: without a reset the preceding
        // constant s0 would prove its branch, but the early branch reaches it with any s0.
        let insts = [
            br(SoppOp::CbranchScc0, 0, 4),
            zero_pair(),
            and_exec(),
            select(),
            compare(),
            br(SoppOp::CbranchScc1, 5, 7),
            nop(),
            end(),
        ];
        let (program, choices) = convert_ok(&insts);
        // pcs: 0 br, 1..=3 mov/and/select, 4 `L4:`, 5 compare, 6 br, 7 nop, 8 `L7:`, 9 endpgm.
        assert_eq!(program.instructions[4].text, "L4:");
        assert_eq!(program.instructions[6].text, "s_cbranch_scc1 L7");
        assert_eq!(choices[4], Both, "inserted labels map to Both");
        assert_eq!(choices[6], Both, "{choices:?}");
        assert!(choices.iter().all(|c| *c == Both));
    }

    #[test]
    fn a_vector_instruction_writing_an_sgpr_kills_the_scalar_fact() {
        // v_readfirstlane_b32 s0 between the select and the compare: s0 is unknown.
        let insts = [
            zero_pair(),
            and_exec(),
            select(),
            ops::v1(Vop1Op::ReadfirstlaneB32, s(0), v(1)),
            compare(),
            br(SoppOp::CbranchScc1, 5, 7),
            nop(),
            end(),
        ];
        let (_, choices) = convert_ok(&insts);
        assert!(choices.iter().all(|c| *c == Both), "{choices:?}");
    }

    #[test]
    fn loop_back_edges_prove_and_replay_with_inserted_labels() {
        let mut insts = certificate();
        insts.extend([br(SoppOp::CbranchScc1, 4, 0), end()]);
        let (program, choices) = convert_ok(&insts);
        // pcs: 0 `L0:`, 1..=4 certificate, 5 branch, 6 endpgm.
        assert_eq!(program.instructions[0].text, "L0:");
        assert_eq!(program.instructions[5].text, "s_cbranch_scc1 L0");
        assert_eq!(choices, [Both, Both, Both, Both, Both, Taken, Both]);
        let stats = ledger_cfg::verify_with_branch_proofs(&program, &[], &choices).unwrap();
        assert_eq!(stats.label_states, 1, "{stats:?}");
    }

    #[test]
    fn an_invalid_branch_target_is_rejected_whatever_the_proof_says() {
        let mut insts = certificate();
        insts.extend([ops::sp(SoppOp::CbranchScc1, 100), end()]);
        let e = analyze(&insts).map(|_| ()).unwrap_err();
        assert!(e.contains("not an instruction boundary"), "{e}");
        // Falling off the end stays an error as well.
        let mut insts = certificate();
        insts.push(br(SoppOp::CbranchScc1, 4, 0));
        let e = analyze(&insts).map(|_| ()).unwrap_err();
        assert!(e.contains("fall off the end"), "{e}");
    }

    #[test]
    fn certify_stream_hands_the_proof_table_to_the_strict_verifier() {
        // The unwaited load/use pair sits only on the impossible fall-through of the
        // zero-mask gate; every real execution jumps to the end.
        let mut insts = certificate();
        insts.extend([
            br(SoppOp::CbranchScc1, 4, 7),
            ops::global_load(
                peacemaker_cdna::enc::flat::FlatOp::LoadDword,
                v(1),
                v(2),
                None,
                0,
            ),
            ops::v1(Vop1Op::MovB32, v(200), v(1)),
            end(),
        ]);
        let (program, choices) = convert_ok(&insts);
        assert_eq!(choices[4], Taken);
        // Without the proof the same Program is refused for the RAW...
        let e = ledger_cfg::verify(&program, &[]).unwrap_err();
        assert!(e.contains("needs Vm wait 0"), "{e}");
        // ...and certification of the typed stream, which carries it, passes.
        let evidence = certify_stream("k", &typed(&insts)).expect("certifies");
        assert_eq!(evidence["checked"], true);
    }

    /// The exact P1b wide64 anchor: pc 893 is the zero-mask exit gate and is `Taken`; the
    /// all-`Both` reference walk (the unadapted verifier) still reports the baseline's pc 975
    /// VMEM WAW, which is reachable only through the impossible fall-through of pc 893.
    #[test]
    fn p1b_wide64_pc893_is_taken_and_the_all_both_reference_keeps_its_waw() {
        let (bundle, payloads) = peacemaker_ir::envelope::Bundle::read(P1B).expect("bundle");
        let entry = bundle
            .entries
            .iter()
            .position(|e| e.triple == "hipv4-amdgcn-amd-amdhsa--gfx942")
            .expect("one gfx942 entry");
        let lifted = peacemaker_lift::cdna::lift_object(payloads[entry]).expect("typed image");
        let kernel = lifted
            .kernels
            .iter()
            .find(|k| k.image.name == "gemm_bf16_mfma_gfx942_wide64")
            .expect("wide64 symbol");
        let insts: Vec<Inst> = kernel.instructions.iter().map(|t| t.inst.clone()).collect();
        let layout = layout("wide64", &kernel.instructions).unwrap();
        let cfg = cfg("wide64", &insts, &layout).unwrap();
        assert!(
            hazard::check(CDNA, &insts).is_empty(),
            "hazard pass is unchanged"
        );
        let (program, _, choices) = ledger_program("wide64", &insts, &cfg).unwrap();
        assert_eq!(choices.len(), program.instructions.len());

        let text = |pc: usize| program.instructions[pc].text.as_str();
        assert!(text(889).starts_with("s_mov_b64"), "{}", text(889));
        assert!(text(890).starts_with("s_and_b64"), "{}", text(890));
        assert!(text(891).starts_with("s_cselect_b32"), "{}", text(891));
        assert!(text(892).starts_with("s_cmp_lg_u32"), "{}", text(892));
        assert!(text(893).starts_with("s_cbranch_scc1 "), "{}", text(893));
        assert_eq!(choices[893], Taken);
        for (pc, insn) in program.instructions.iter().enumerate() {
            if is_label_text(&insn.text) {
                assert_eq!(choices[pc], Both, "label pc {pc}");
            }
        }

        // Reference: no proofs. `verify` and an all-Both table are the same walk.
        let reference = ledger_cfg::verify(&program, &[]).unwrap_err();
        for required in [
            "pc 975",
            "global_load_dwordx4 v[34:37], v[4:5], off",
            "L872",
            "needs Vm wait 0",
            "Waw { reg: \"v[34:37]\" }",
        ] {
            assert!(
                reference.contains(required),
                "missing {required:?}: {reference}"
            );
        }
        let both = vec![Both; program.instructions.len()];
        assert_eq!(
            ledger_cfg::verify_with_branch_proofs(&program, &[], &both),
            Err(reference.clone())
        );

        // Refined: either certifies, or reports a finding that is not the infeasible edge
        // into pc 975 (a distinct later error is fine; nothing is hidden).
        if let Err(refined) = ledger_cfg::verify_with_branch_proofs(&program, &[], &choices) {
            let impossible = format!("pc 893 `{}` not taken", text(893));
            assert!(!refined.contains(&impossible), "{refined}");
        }
    }

    fn is_label_text(text: &str) -> bool {
        text.trim().ends_with(':')
    }
}

#[cfg(test)]
mod generated_encoding_tests {
    use super::*;

    #[test]
    fn verified_generated_samples_refuse_simulation_and_certification_by_name() {
        use model::sim::{self, Config, Memory, Wave};
        use model::xml::{self, Isa};
        let mut generated = 0;
        for form in xml::forms(Isa::Cdna3) {
            if !form.verification.is_verified() {
                continue;
            }
            let (inst, _) = Inst::decode(CDNA, form.sample).expect(form.name);
            let Inst::Xml(x) = &inst else { continue };
            generated += 1;
            let mnemonic = x.decoded().form.mnemonic;
            let mut wave = Wave::new(CDNA, 0, 0);
            let error = sim::exec_one(
                &mut wave, &inst, &mut [], &mut Memory::default(), &Config::default(),
            ).unwrap_err().to_string();
            assert!(error.contains(mnemonic), "{error}");
            let error = certify_stream(
                "generated", &[Typed { dword_offset: 0, inst }],
            ).unwrap_err();
            assert!(error.contains(mnemonic), "{error}");
        }
        assert!(generated >= 88, "missing generated form coverage: {generated}");
    }

    #[test]
    fn seal_refuses_unverified_encoding_by_name() {
        let (inst, _) = Inst::decode(CDNA, &[0xbf9f_0000]).unwrap();
        let error = certify_stream(
            "unverified",
            &[Typed {
                dword_offset: 0,
                inst,
            }],
        )
        .unwrap_err();
        assert!(error.contains("s_set_valu_coexec_mode"), "{error}");
        assert!(error.contains("unverified encoding"), "{error}");
    }

    #[test]
    fn seal_refuses_unmodelled_scalar_memory_by_name() {
        for (word, mnemonic) in [
            (0xc040_0000, "s_store_dword"),
            (0xc208_0000, "s_atomic_add"),
        ] {
            let (inst, _) = Inst::decode(CDNA, &[word, 0]).unwrap();
            let error = certify_stream(
                "unmodelled",
                &[Typed {
                    dword_offset: 0,
                    inst,
                }],
            )
            .unwrap_err();
            assert!(error.contains(mnemonic), "{error}");
        }
    }
}
