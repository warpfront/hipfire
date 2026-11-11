// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lowering of the generic (text) [`crate::Builder`] onto the checked CDNA
//! [`super::builder::Builder`].
//!
//! On `Arch::Gfx942` the generic builder is a source-ownership and control-flow
//! front end: it validates register ownership/lifetimes, LDS slot ownership,
//! release/acquire obligations and label structure over the source text, but
//! it neither decides the final waits nor the hazard wait states. At `finish`
//! the source program is parsed (`peacemaker_cdna::text::parse_line`) and
//! replayed onto the typed CDNA builder, whose ledger inserts the counter
//! waits, whose hazard pass inserts the CDNA3 wait states and whose encoder and
//! emitter produce the code. The waits the generic builder inserted while
//! building are dropped: only explicit requests (`Builder::wait`,
//! `wait_all`, `drain_all_counters`) are replayed.

use super::builder::{ledger_reg, memory_class, Builder as CdnaBuilder};
use super::{KernArgKind, Kernel};
use crate::insn::{Instruction, MemoryClass};
use crate::ledger::Counter;
use peacemaker_cdna::enc::sop::{Sopp, SoppOp};
use peacemaker_cdna::inst::RegFile;
use peacemaker_cdna::Inst;

/// What one source line of a gfx942 program is.
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    /// `name:`
    Label(String),
    /// `s_branch` / `s_cbranch_*` to a symbolic label.
    Branch { op: SoppOp, target: String },
    /// `s_barrier`
    Barrier,
    /// `s_endpgm`
    End,
    /// `s_waitcnt`: the generic builder's own wait (re-derived by the CDNA
    /// ledger) or a raw one, which `Builder::push` refuses.
    Wait,
    /// Any other instruction, fully typed.
    Inst(Inst),
}

fn mnemonic(text: &str) -> &str {
    text.split_whitespace().next().unwrap_or("")
}

/// Parse one source line of a gfx942 program.
pub fn classify(text: &str, arch: peacemaker_cdna::Arch) -> Result<Source, String> {
    let text = text.trim();
    if let Some(name) = text.strip_suffix(':') {
        if name.is_empty() || name.contains(char::is_whitespace) {
            return Err(format!("{text:?} is not a label"));
        }
        return Ok(Source::Label(name.to_owned()));
    }
    let mn = mnemonic(text);
    let symbolic = |operand: &str| {
        operand
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '.' || c == '_')
    };
    if mn == "s_branch" || mn.starts_with("s_cbranch_") {
        let operand = text[mn.len()..].trim();
        if symbolic(operand) {
            if operand.contains(char::is_whitespace) {
                return Err(format!("{text:?}: a branch takes one label"));
            }
            let probe = peacemaker_cdna::text::parse_line(&format!("{mn} 0"), arch)
                .map_err(|e| format!("{text}: {e}"))?;
            return match probe {
                Inst::Sopp(p) if p.branch_offset().is_some() => Ok(Source::Branch {
                    op: p.op,
                    target: operand.to_owned(),
                }),
                _ => Err(format!("{text}: not a branch")),
            };
        }
    }
    let inst = peacemaker_cdna::text::parse_line(text, arch).map_err(|e| format!("{text}: {e}"))?;
    super::require_verified(arch, &inst)?;
    if let Some(reason) = inst.unsupported() {
        return Err(reason);
    }
    Ok(match inst {
        Inst::Sopp(Sopp {
            op: SoppOp::Waitcnt,
            ..
        }) => Source::Wait,
        Inst::Sopp(Sopp {
            op: SoppOp::Barrier,
            ..
        }) => Source::Barrier,
        Inst::Sopp(Sopp {
            op: SoppOp::Endpgm, ..
        }) => Source::End,
        inst => Source::Inst(inst),
    })
}

/// The ledger view of a typed instruction: its register footprint (SGPR, VGPR
/// and AGPR spans; special registers are tracked by the hazard pass, not the
/// ledger) and memory class.
/// Generated instructions without a hand-written ledger/hazard model refuse
/// by name instead of being assigned an existing instruction's memory class.
pub fn shadow(inst: &Inst) -> Result<Instruction, String> {
    super::require_verified(peacemaker_cdna::Arch::Gfx942, inst)?;
    if let Some(reason) = inst.unsupported() {
        return Err(reason);
    }
    let footprint = inst.footprint();
    let mut shadow = Instruction::new(
        format!("{}", inst.assembly()),
        footprint.defs.iter().filter_map(ledger_reg).collect(),
        footprint.uses.iter().filter_map(ledger_reg).collect(),
    );
    shadow.memory = inst.memory().map(memory_class).transpose()?;
    Ok(shadow)
}

/// An explicit wait request recorded by the generic builder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Wait {
    /// `Builder::wait(counter, count)` and `drain_all_counters`.
    Counters(Vec<(Counter, u8)>),
    /// `Builder::wait_all`.
    All,
}

/// The kernel-level inputs of the lowering (the generic `KernelSpec` and the
/// register plan's high water marks).
pub struct Abi<'a> {
    pub name: &'a str,
    pub workgroup_size: u32,
    pub user_sgpr_count: u8,
    pub workgroup_id_y: bool,
    pub workgroup_id_z: bool,
    pub group_segment_fixed_size: u32,
    pub kernargs: &'a [(String, u32, u32, String, Option<String>)],
    pub kernarg_size: u32,
    pub plan_sgprs: u16,
    pub plan_vgprs: u16,
    pub plan_agprs: u16,
}

/// Replay `program` (the generic builder's source program) onto a fresh
/// checked CDNA builder and finish it. `waits` are the explicit wait requests
/// keyed by the source program index they precede.
pub fn lower(
    arch: crate::Arch,
    abi: &Abi<'_>,
    program: &[Instruction],
    waits: &[(usize, Wait)],
) -> Result<Kernel, String> {
    let cdna = arch.cdna().ok_or("lowering requires a CDNA architecture")?;
    if abi.user_sgpr_count != 2 {
        return Err(format!(
            "gfx942 kernels have exactly the kernarg pointer s[0:1] as user SGPRs, KernelSpec declares {}",
            abi.user_sgpr_count
        ));
    }
    if abi.workgroup_id_z && !abi.workgroup_id_y {
        return Err("workgroup id Z requires workgroup id Y".into());
    }
    let sources = program
        .iter()
        .map(|i| classify(&i.text, cdna))
        .collect::<Result<Vec<_>, _>>()?;
    // Registers the source touches beyond the plan (imported text): the
    // descriptor must cover every one of them.
    let (mut sgprs, mut vgprs, mut agprs) = (abi.plan_sgprs, abi.plan_vgprs, abi.plan_agprs);
    for source in &sources {
        let Source::Inst(inst) = source else { continue };
        let footprint = inst.footprint();
        for span in footprint.defs.iter().chain(&footprint.uses) {
            let end = u16::from(span.base) + u16::from(span.len);
            match span.file {
                RegFile::Sgpr => sgprs = sgprs.max(end),
                RegFile::Vgpr => vgprs = vgprs.max(end),
                RegFile::Agpr => agprs = agprs.max(end),
                _ => {}
            }
        }
    }
    let mut b = CdnaBuilder::new(arch, abi.name, abi.workgroup_size)?;
    for (name, offset, size, value_kind, _) in abi.kernargs {
        let kind = match value_kind.as_str() {
            "global_buffer" => KernArgKind::GlobalBuffer { readonly: false },
            "by_value" => KernArgKind::ByValue,
            other => {
                return Err(format!(
                    "kernel argument {name}: value kind {other:?} has no gfx942 lowering"
                ))
            }
        };
        b.arg_at(name, *offset, *size, kind)?;
    }
    b.kernarg_segment_size(abi.kernarg_size)?;
    b.workgroup_id(0)?;
    if abi.workgroup_id_y {
        b.workgroup_id(1)?;
    }
    if abi.workgroup_id_z {
        b.workgroup_id(2)?;
    }
    b.workitem_ids(1)?;
    b.reserve_sgprs(sgprs)?;
    b.reserve_vgprs(vgprs)?;
    b.reserve_agprs(agprs)?;
    if abi.group_segment_fixed_size > 0 {
        b.lds(abi.group_segment_fixed_size, 1)?;
    }
    let mut pending = waits.iter().peekable();
    for (index, source) in sources.into_iter().enumerate() {
        while let Some((_, wait)) = pending.next_if(|(at, _)| *at <= index) {
            match wait {
                Wait::Counters(counters) => b.wait_counters(counters)?,
                Wait::All => b.wait_all()?,
            }
        }
        match source {
            Source::Label(name) => b.label(&format!("rip_{name}"))?,
            Source::Branch { op, target } => b.branch(op, &format!("rip_{target}"))?,
            Source::Barrier => b.barrier()?,
            Source::End => b.end()?,
            // Re-derived by the CDNA ledger from the instructions that need them.
            Source::Wait => {}
            Source::Inst(inst) => b.push(inst)?,
        }
    }
    for (_, wait) in pending {
        match wait {
            Wait::Counters(counters) => b.wait_counters(counters)?,
            Wait::All => b.wait_all()?,
        }
    }
    b.finish()
}

/// The final program of a lowered kernel as generic [`Instruction`]s with
/// symbolic labels and branches (what `Emitted::program` carries), plus, per
/// `kernel.insts` index, the program index of that instruction (labels shift
/// them). Proof `pc_index` fields of `kernel` index `kernel.insts`; remap them
/// through the second value.
pub fn final_program(kernel: &Kernel) -> Result<(Vec<Instruction>, Vec<usize>), String> {
    let mut starts = Vec::with_capacity(kernel.insts.len());
    let mut words = Vec::with_capacity(kernel.words.len());
    for inst in &kernel.insts {
        starts.push(words.len());
        inst.encode(kernel.arch, &mut words)?;
    }
    let label_at = |at: usize| kernel.labels.iter().find(|(_, o)| *o == at).map(|(n, _)| n);
    let mut program = Vec::with_capacity(kernel.insts.len() + kernel.labels.len());
    let mut index = Vec::with_capacity(kernel.insts.len());
    for (i, inst) in kernel.insts.iter().enumerate() {
        for (name, _) in kernel.labels.iter().filter(|(_, o)| *o == starts[i]) {
            program.push(Instruction::new(format!("{name}:"), vec![], vec![]));
        }
        index.push(program.len());
        let branch = match inst {
            Inst::Sopp(p) => p.branch_offset().and_then(|offset| {
                let target = starts[i] as i64 + 1 + i64::from(offset);
                usize::try_from(target)
                    .ok()
                    .and_then(label_at)
                    .map(|name| format!("{} {name}", p.op.name()))
            }),
            _ => None,
        };
        match branch {
            Some(text) => program.push(Instruction::new(text, vec![], vec![])),
            None => program.push(shadow(inst)?),
        }
    }
    for (name, _) in kernel.labels.iter().filter(|(_, o)| *o == words.len()) {
        program.push(Instruction::new(format!("{name}:"), vec![], vec![]));
    }
    Ok((program, index))
}

/// The memory class a final-program instruction carries (for shape counting).
pub fn is_memory(insn: &Instruction) -> bool {
    matches!(
        insn.memory,
        Some(MemoryClass::VmemLoad | MemoryClass::VmemStore | MemoryClass::VmemAtomic { .. })
    )
}

#[cfg(test)]
mod xml_refusal_tests {
    use super::*;

    #[test]
    fn scalar_store_and_atomic_refuse_by_name() {
        for (words, name) in [
            ([0xc040_0000, 0], "s_store_dword"),
            ([0xc208_0000, 0], "s_atomic_add"),
        ] {
            for arch in [peacemaker_cdna::Arch::Gfx942, peacemaker_cdna::Arch::Gfx950] {
                let (inst, n) = Inst::decode(arch, &words).unwrap();
                assert_eq!(n, 2);
                let error = shadow(&inst).unwrap_err();
                assert!(error.contains(name), "{error}");
                assert!(error.contains("no hand-written model"), "{error}");
                let hazards = super::super::hazard::check(arch, &[inst]);
                assert!(hazards.iter().any(|error| error.contains(name)), "{hazards:?}");
            }
        }
    }

    #[test]
    fn unverified_encoding_refuses_emission_and_shadow_by_name() {
        let (inst, _) =
            Inst::decode(peacemaker_cdna::Arch::Gfx942, &[0xbf9f_0000]).unwrap();
        let error = inst.words(peacemaker_cdna::Arch::Gfx942).unwrap_err();
        assert!(error.contains("s_set_valu_coexec_mode"), "{error}");
        assert!(error.contains("unverified encoding"), "{error}");
        let error = shadow(&inst).unwrap_err();
        assert!(error.contains("s_set_valu_coexec_mode"), "{error}");
        assert!(error.contains("unverified encoding"), "{error}");
    }
}

#[cfg(test)]
mod typed_unverified_tests {
    use super::*;

    #[test]
    fn typed_decoder_cannot_bypass_unverified_encoding_gate() {
        let (inst, _) =
            Inst::decode(peacemaker_cdna::Arch::Gfx942, &[0xd140_0000, 0]).unwrap();
        assert!(inst.xml().is_none(), "the legacy typed path is the regression");
        let error = shadow(&inst).unwrap_err();
        assert!(error.contains("v_nop"), "{error}");
        assert!(error.contains("unverified encoding"), "{error}");
    }
}
