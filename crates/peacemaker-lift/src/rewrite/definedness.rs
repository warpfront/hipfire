// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Check 11 `definedness` (architecture.md §3): every register an instruction reads —
//! implicit EXEC, VCC, SCC and M0 included — must be defined on every incoming CFG path,
//! by the code or by the kernel's ABI entry live-ins.
//!
//! This is the certifier's own replay, separate from core's `edit` precondition: a
//! forward must-defined dataflow over the basic blocks of a kernel lifted from the final
//! bytes, dword-granular (a true16 half write does not define its register), seeded from
//! the descriptor as the hardware loads it: enabled user SGPRs, work-group ids,
//! `v0` and EXEC. Partial VGPR destination reads remain open core obligations
//! until lane/half coverage can be proved; other newly undefined reads fail.

use std::collections::BTreeSet;

use peacemaker_ir::cfg::{Body, InstId};
use peacemaker_ir::effects::{preserved_vgpr_destination, ImplicitSet};
use peacemaker_ir::inst::{Abi, EntryCell, Form, Inst, Kernel, Wave};
use peacemaker_ir::operand::Operand;
use peacemaker_ir::reg::{Kind, RegRef};

const V0: usize = 0;
const S0: usize = 256;
const T0: usize = S0 + 106;
const VCC_LO: usize = T0 + 16;
const VCC_HI: usize = VCC_LO + 1;
const EXEC_LO: usize = VCC_LO + 2;
const EXEC_HI: usize = VCC_LO + 3;
const SCC: usize = VCC_LO + 4;
const M0: usize = VCC_LO + 5;
const LOCATIONS: usize = M0 + 1;
const WORDS: usize = LOCATIONS.div_ceil(64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Set([u64; WORDS]);

impl Set {
    const EMPTY: Set = Set([0; WORDS]);
    const ALL: Set = Set([u64::MAX; WORDS]);
    fn insert(&mut self, i: usize) {
        self.0[i / 64] |= 1 << (i % 64);
    }
    fn contains(&self, i: usize) -> bool {
        self.0[i / 64] >> (i % 64) & 1 != 0
    }
    fn meet(&mut self, other: &Set) {
        for (a, b) in self.0.iter_mut().zip(other.0) {
            *a &= b;
        }
    }
}

fn locations(reg: RegRef) -> impl Iterator<Item = usize> {
    let base = match reg.kind {
        Kind::V => V0,
        Kind::S => S0,
        Kind::Ttmp => T0,
    };
    (usize::from(reg.base)..usize::from(reg.base) + usize::from(reg.len)).map(move |r| base + r)
}

fn implicit(bits: u8, wave: Wave) -> Vec<usize> {
    let mut out = Vec::new();
    let wide = wave == Wave::Wave64;
    if bits & ImplicitSet::SCC != 0 {
        out.push(SCC);
    }
    if bits & ImplicitSet::VCC != 0 {
        out.push(VCC_LO);
        if wide {
            out.push(VCC_HI);
        }
    }
    if bits & ImplicitSet::EXEC != 0 {
        out.push(EXEC_LO);
        if wide {
            out.push(EXEC_HI);
        }
    }
    if bits & ImplicitSet::M0 != 0 {
        out.push(M0);
    }
    out
}

fn name(i: usize) -> String {
    match i {
        _ if i < S0 => format!("v{i}"),
        _ if i < T0 => format!("s{}", i - S0),
        _ if i < VCC_LO => format!("ttmp{}", i - T0),
        VCC_LO => "vcc_lo".into(),
        VCC_HI => "vcc_hi".into(),
        EXEC_LO => "exec_lo".into(),
        EXEC_HI => "exec_hi".into(),
        SCC => "scc".into(),
        _ => "m0".into(),
    }
}

fn is_vector(form: Form) -> bool {
    matches!(
        form,
        Form::Vop1
            | Form::Vop2
            | Form::Vop3
            | Form::Vop3p
            | Form::Vopc
            | Form::Vopd
            | Form::Vinterp
            | Form::Ds
            | Form::Vmem(_)
            | Form::Export
    )
}

/// (reads, full definitions) of one instruction.
fn access(inst: &Inst, wave: Wave) -> (Vec<usize>, Vec<usize>) {
    let half = |reg: &RegRef| {
        inst.operands
            .iter()
            .any(|op| matches!(op, Operand::Half(r, _) if r == reg))
    };
    let mut reads: Vec<usize> = inst
        .effects
        .uses
        .iter()
        .flat_map(|r| locations(*r))
        .collect();
    reads.extend(implicit(inst.effects.implicit.reads, wave));
    if is_vector(inst.form) {
        reads.extend(implicit(ImplicitSet::EXEC, wave));
    }
    let mut defs: Vec<usize> = inst
        .effects
        .defs
        .iter()
        .filter(|r| !half(r))
        .flat_map(|r| locations(*r))
        .collect();
    defs.extend(implicit(inst.effects.implicit.writes, wave));
    (reads, defs)
}

/// Locations defined at the kernel entry: the hardware-initialised state of an HSA
/// descriptor, or exactly the declared entry facts of a raw kernel.
fn entry(kernel: &Kernel) -> Set {
    let mut set = Set::EMPTY;
    let (user, wg_x, wg_yz) = match &kernel.abi {
        Abi::Hsa { descriptor, .. } => {
            let rsrc2 = descriptor.compute_pgm_rsrc2.0;
            (
                (rsrc2 >> 1 & 0x1f) as usize,
                rsrc2 >> 7 & 1 != 0,
                rsrc2 >> 8 & 3 != 0,
            )
        }
        Abi::Raw(raw) => {
            // Facts that do not validate define nothing.
            for (cell, _) in raw.entry_cells().unwrap_or_default() {
                match cell {
                    EntryCell::Vgpr(n) => set.insert(V0 + usize::from(n)),
                    EntryCell::Sgpr(n) => set.insert(S0 + usize::from(n)),
                    EntryCell::Ttmp(n) => set.insert(T0 + usize::from(n)),
                    EntryCell::VccLo => set.insert(VCC_LO),
                    EntryCell::VccHi => set.insert(VCC_HI),
                    EntryCell::ExecLo => set.insert(EXEC_LO),
                    EntryCell::ExecHi => set.insert(EXEC_HI),
                    EntryCell::Scc => set.insert(SCC),
                    EntryCell::M0 => set.insert(M0),
                    _ => {}
                }
            }
            return set;
        }
    };
    for s in 0..user {
        set.insert(S0 + s);
    }
    if wg_x {
        set.insert(T0 + 9);
    }
    if wg_yz {
        set.insert(T0 + 7);
    }
    set.insert(V0);
    for e in implicit(ImplicitSet::EXEC, kernel.wave) {
        set.insert(e);
    }
    set
}

/// One instruction's reads that are not defined on every path, by layout index.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Undefined {
    pub index: usize,
    pub inst: InstId,
    pub locations: BTreeSet<String>,
}

/// Every read of `kernel` that some path from the entry reaches undefined.
pub fn undefined_reads(kernel: &Kernel) -> Vec<Undefined> {
    let body: &Body = &kernel.body;
    let n = body.blocks.len();
    let inst = |p: usize| body.insts.get(body.layout[p]).expect("laid out");
    let transfer = |b: usize, mut set: Set| {
        for p in body.blocks[b].range.0..body.blocks[b].range.1 {
            for d in access(inst(p), kernel.wave).1 {
                set.insert(d);
            }
        }
        set
    };
    let seed = entry(kernel);
    let mut out_sets = vec![Set::ALL; n];
    let mut in_sets = vec![Set::ALL; n];
    let mut changed = true;
    while changed {
        changed = false;
        for b in 0..n {
            let mut set = if b == 0 { seed } else { Set::ALL };
            for p in &body.blocks[b].preds {
                set.meet(&out_sets[p.0]);
            }
            if b != 0 && body.blocks[b].preds.is_empty() {
                continue;
            }
            let out = transfer(b, set);
            if set != in_sets[b] || out != out_sets[b] {
                changed = true;
            }
            in_sets[b] = set;
            out_sets[b] = out;
        }
    }
    let mut found = Vec::new();
    for b in 0..n {
        if b != 0 && body.blocks[b].preds.is_empty() {
            continue;
        }
        let mut set = in_sets[b];
        for p in body.blocks[b].range.0..body.blocks[b].range.1 {
            let (reads, defs) = access(inst(p), kernel.wave);
            let missing: BTreeSet<String> = reads
                .into_iter()
                .filter(|&r| !set.contains(r))
                .map(name)
                .collect();
            if !missing.is_empty() {
                found.push(Undefined {
                    index: p,
                    inst: body.layout[p],
                    locations: missing,
                });
            }
            for d in defs {
                set.insert(d);
            }
        }
    }
    found
}

/// Check 11 on a rewrite. `edited` is the kernel the script produced (its instructions
/// carry edit provenance, original ones keep their base ids); `emitted` is the same kernel
/// lifted back from the final bytes, instruction for instruction.
pub fn check(
    arch: peacemaker_ir::inst::Arch,
    base: &Kernel,
    edited: &Kernel,
    emitted: &Kernel,
) -> Result<(), String> {
    if emitted.body.layout.len() != edited.body.layout.len() {
        return Err(format!(
            "the final bytes hold {} instructions, the script produced {}",
            emitted.body.layout.len(),
            edited.body.layout.len()
        ));
    }
    let before: std::collections::HashMap<InstId, BTreeSet<String>> = undefined_reads(base)
        .into_iter()
        .map(|u| (u.inst, u.locations))
        .collect();
    let mut failures = Vec::new();
    for u in undefined_reads(emitted) {
        let id = edited.body.layout[u.index];
        let inst = edited.body.insts.get(id).expect("laid out");
        let mut missing = u.locations;
        if let Some(reg) = preserved_vgpr_destination(arch, inst) {
            for d in locations(reg) {
                missing.remove(&name(d));
            }
        }
        if missing.is_empty() {
            continue;
        }
        let text = inst.text(arch).unwrap_or_default();
        if inst.prov.edit.is_some() {
            failures.push(format!(
                "inserted `{text}` reads {missing:?} undefined on some path"
            ));
        } else if !missing.is_subset(before.get(&id).unwrap_or(&BTreeSet::new())) {
            failures.push(format!(
                "original `{text}` now reads {missing:?} undefined on some path"
            ));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use peacemaker_ir::cfg::Arena;
    use peacemaker_ir::descriptor::{
        KernargPreload, KernelCodeProperties, KernelDescriptor, Rsrc1, Rsrc2, Rsrc3,
    };
    use peacemaker_ir::inst::{KernelOrigin, SymbolId};
    use peacemaker_ir::metadata::{HsaKernelMetadata, KernelMeta};
    use peacemaker_ir::passes::cfg::build_blocks;
    use peacemaker_ir::provenance::EditId;

    /// `words` decoded in order; `inserted` marks edit provenance by layout index.
    fn kernel(words: &[u32], inserted: &[usize]) -> Kernel {
        let mut insts: Arena<Inst> = Arena::new();
        let mut layout = Vec::new();
        for (k, w) in words.iter().enumerate() {
            let (mut inst, _) = peacemaker_ir::codec::gfx12::decode(&[*w]).unwrap();
            if inserted.contains(&k) {
                inst.prov.edit = Some(EditId(1));
            }
            layout.push(insts.insert(inst));
        }
        let mut body = Body {
            insts,
            blocks: Vec::new(),
            layout,
        };
        build_blocks(&mut body, peacemaker_ir::inst::Arch::Gfx1201).unwrap();
        let descriptor = KernelDescriptor {
            group_segment_fixed_size: 0,
            private_segment_fixed_size: 0,
            kernarg_size: 0,
            kernel_code_entry_byte_offset: 0,
            compute_pgm_rsrc3: Rsrc3(0),
            compute_pgm_rsrc1: Rsrc1(0),
            compute_pgm_rsrc2: Rsrc2(2 << 1),
            kernel_code_properties: KernelCodeProperties(1 << 10),
            kernarg_preload: KernargPreload(0),
            reserved: [0; 28],
        };
        Kernel {
            symbol: SymbolId("k".into()),
            wave: Wave::Wave32,
            abi: Abi::Hsa {
                descriptor,
                metadata: HsaKernelMetadata {
                    raw_msgpack: Vec::new(),
                    parsed: KernelMeta::default(),
                },
            },
            body,
            origin: KernelOrigin::Authored {
                builder_crate: "test".into(),
                version: "0".into(),
                git: "0".into(),
            },
        }
    }

    const CMP: u32 = 0xbf06_8000; // s_cmp_eq_u32 s0, 0
    const SKIP1: u32 = 0xbfa2_0001; // s_cbranch_scc1 +1
    const DEF_V200: u32 = 0x7f90_0280; // v_mov_b32_e32 v200, 0
    const READ_V200: u32 = 0x7e02_03c8; // v_mov_b32_e32 v1, v200
    const ENDPGM: u32 = 0xbfb0_0000;

    /// M3 check-11 regression: a diamond where only the fall-through predecessor defines
    /// `v200`; an inserted read of it at the join fails, although `v200` is dead there.
    #[test]
    fn inserted_read_of_a_register_one_predecessor_leaves_undefined_fails() {
        let base = kernel(&[CMP, SKIP1, DEF_V200, ENDPGM], &[]);
        let edited = kernel(&[CMP, SKIP1, DEF_V200, READ_V200, ENDPGM], &[3]);
        assert_eq!(
            edited.body.blocks.len(),
            3,
            "diamond: [cmp, branch] [def v200] [join]"
        );
        let err = check(peacemaker_ir::inst::Arch::Gfx1201, &base, &edited, &edited).unwrap_err();
        assert!(err.contains("v200"), "{err}");
    }

    /// Straight-line insertion reading what dominates it passes; so does an original
    /// read that was already undefined in the base (not a regression of the edit).
    #[test]
    fn straight_line_insertion_passes() {
        let base = kernel(&[DEF_V200, ENDPGM], &[]);
        let edited = kernel(&[DEF_V200, READ_V200, ENDPGM], &[1]);
        assert_eq!(
            check(peacemaker_ir::inst::Arch::Gfx1201, &base, &edited, &edited),
            Ok(())
        );
        let base = kernel(&[CMP, SKIP1, DEF_V200, READ_V200, ENDPGM], &[]);
        assert_eq!(
            undefined_reads(&base).len(),
            1,
            "the original join read is undefined on the taken path"
        );
        assert_eq!(
            check(peacemaker_ir::inst::Arch::Gfx1201, &base, &base, &base),
            Ok(())
        );
    }
}
