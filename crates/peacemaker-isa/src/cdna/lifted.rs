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
//! * **Wait ledger**: [`crate::ledger_cfg::verify`] over a [`Program`] derived from the
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
//! Fail-closed, never skipped: an instruction the ledger cannot model is an error, not a
//! pass. That covers `s_setpc_b64`, `s_trap`, `s_sendmsg`, a branch off an instruction
//! boundary or past the code, a reachable fall-through off the end, FLAT-segment and
//! SCRATCH accesses (FLAT counts on VM_CNT and LGKM_CNT and may address LDS; SCRATCH needs
//! a private segment), GDS operations, and memory operations whose data/destination
//! registers are AGPRs or special-register files (the ledger tracks only SGPRs and VGPRs).
//!
//! Not modelled (reported, not hidden): global-memory visibility across a barrier,
//! unreachable instructions are hazard-checked but not ledger-walked (their count is in
//! the evidence), and every wait-state/counter rule is the shared checkers' and carries
//! their `[UNVERIFIED: confirm on MI300X]` caveats.

use super::hazard;
use crate::arch::Arch;
use crate::insn::{Instruction, MemoryClass, Program};
use crate::ledger_cfg;
use crate::reg::{Kind, RegRef};
use peacemaker_cdna::enc::flat::FlatSeg;
use peacemaker_cdna::enc::sop::{waitcnt_fields, Sop1Op, SoppOp};
use peacemaker_cdna::inst::{MemKind, RegFile, RegSpan};
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
            if violations.len() > SHOWN { " | ..." } else { "" }
        ));
    }

    let (program, counts) = ledger_program(name, &insts, &cfg)?;
    let stats = ledger_cfg::verify(&program, &[])
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
            "verifier": "peacemaker_isa::ledger_cfg::verify",
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

fn ledger_program(name: &str, insts: &[Inst], cfg: &Cfg) -> Result<(Program, Counts), String> {
    let mut counts = Counts::default();
    let mut out = Vec::with_capacity(insts.len() + cfg.targets.len());
    for (i, inst) in insts.iter().enumerate() {
        if cfg.targets.contains(&i) {
            out.push(Instruction::new(format!("L{i}:"), vec![], vec![]));
        }
        out.push(
            convert(inst, cfg.branch[i], &mut counts)
                .map_err(|e| format!("{name}: instruction {i} `{inst}`: {e}"))?,
        );
    }
    Ok((
        Program {
            arch: Arch::Gfx942,
            instructions: out,
        },
        counts,
    ))
}

fn convert(inst: &Inst, branch: Option<usize>, counts: &mut Counts) -> Result<Instruction, String> {
    let control =
        |text: String| -> Result<Instruction, String> { Ok(Instruction::new(text, vec![], vec![])) };
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
                control(format!("s_waitcnt vmcnt({vm}) expcnt({exp}) lgkmcnt({lgkm})"))
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
    let footprint = inst.footprint();
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
        .find(|d| !matches!(d.file, RegFile::Sgpr | RegFile::Vgpr))
    {
        return Err(format!(
            "memory destination {bad:?} is not a plain SGPR/VGPR; the ledger cannot track it"
        ));
    }
    if let Some(bad) = footprint.uses.iter().find(|u| {
        !matches!(
            u.file,
            RegFile::Sgpr | RegFile::Vgpr | RegFile::Exec | RegFile::M0 | RegFile::Scc | RegFile::Vcc
        )
    }) {
        return Err(format!(
            "memory source {bad:?} is not a plain SGPR/VGPR; the ledger cannot track it"
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
    };
    Ok(Instruction::new(
        inst.to_string(),
        tracked(&footprint.defs),
        tracked(&footprint.uses),
    )
    .memory(class))
}

/// The SGPR/VGPR spans, as ledger registers.
fn tracked(spans: &[RegSpan]) -> Vec<RegRef> {
    spans
        .iter()
        .filter_map(|span| {
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
        })
        .collect()
}
