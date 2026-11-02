// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! M3: certified rewrites (architecture.md §4, §7 M3).
//!
//! A [`Rewrite`] is a script of core [`Edit`]s on one kernel of a lifted object plus the
//! equivalence it claims. [`apply`] runs every step as its own checked transaction
//! (`Analyzed::edit`: preconditions, validation, re-analysis), emits the module (re-laid
//! out when it grows, `crate::layout`), lifts the bytes back and requires the same kernels
//! (`emit::checked`), then runs check 11 ([`definedness`]) on the final bytes. The
//! [`Receipt`] carries the script, what core reported (claims, `s_delay_alu` rewrites,
//! re-encoded branches, obligations the edit opened) and the equivalence claim, which is
//! admitted only where its rule holds: `ByteIdentical` when the bytes equal the base,
//! `ProvedCommutation` for insert-only scripts whose inserted code writes no memory other
//! than vector stores, touches no LDS, barrier or control flow; `Unproved` always.

pub mod build;
pub mod definedness;
pub mod profile;

use std::collections::{BTreeSet, HashMap};

use peacemaker_ir::cfg::{BlockId, InstId};
use peacemaker_ir::edit::{analyze, DelayRewrite, Edit, EditError};
use peacemaker_ir::effects::{Control, MemClass};
use peacemaker_ir::inst::{Arch, Kernel, Program, SymbolId};
use peacemaker_ir::reg::RegClaim;
use peacemaker_ir::state::{Analyzed, Obligation};

use crate::text::{parse_source, SourceKind};
use crate::{emit, lift_object, sha256, LiftError, Options};

/// What a rewrite claims about the behaviour of the edited kernel against its baseline.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EquivalenceKind {
    ByteIdentical,
    ProvedCommutation,
    GpuBitExact { oracle_receipt: String },
    Unproved,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Equivalence {
    pub baseline_object_sha256: [u8; 32],
    pub kind: EquivalenceKind,
}

/// A script over one kernel (named by its symbol before the script) and its claim.
#[derive(Clone, Debug)]
pub struct Rewrite {
    pub kernel: SymbolId,
    pub script: Vec<Edit>,
    pub claim: EquivalenceKind,
}

#[derive(Clone, Debug)]
pub struct Receipt {
    pub base_object_sha256: [u8; 32],
    pub object_sha256: [u8; 32],
    pub kernel_before: String,
    pub kernel: String,
    /// One line per script step (`Debug` of the core edit).
    pub script: Vec<String>,
    pub claims: Vec<RegClaim>,
    pub delay_rewrites: Vec<DelayRewrite>,
    pub reencoded_branches: usize,
    /// Obligations of the edited kernel that the base kernel did not have.
    pub new_obligations: Vec<Obligation>,
    /// Check 11 on the final bytes.
    pub definedness: Result<(), String>,
    pub equivalence: Equivalence,
}

/// The emitted module, its receipt and the analysed edited program.
pub struct Certified {
    pub bytes: Vec<u8>,
    pub receipt: Receipt,
    pub program: Analyzed<Program>,
}

#[derive(Debug, thiserror::Error)]
pub enum RewriteError {
    #[error("lift: {0}")]
    Lift(#[from] LiftError),
    #[error("step {step}: {error}")]
    Edit {
        step: usize,
        #[source]
        error: EditError,
    },
    #[error("analysis: {0}")]
    Analysis(EditError),
    #[error(transparent)]
    Emit(#[from] emit::EmitError),
    #[error("claim {claim:?} is not admissible: {reason}")]
    Claim {
        claim: EquivalenceKind,
        reason: String,
    },
    #[error("anchor: {0}")]
    Anchor(String),
    #[error("build: {0}")]
    Build(#[from] build::BuildError),
    #[error("client: {0}")]
    Client(String),
}

fn kernel<'a>(program: &'a Program, symbol: &SymbolId) -> &'a Kernel {
    program
        .kernels
        .iter()
        .find(|k| &k.symbol == symbol)
        .expect("kernel present")
}

fn key(o: &Obligation) -> (String, Vec<InstId>) {
    (o.rule_id.clone(), o.insts.clone())
}

/// Apply `rewrite` to the lifted `base` object (see the module docs).
pub fn apply(base: &[u8], options: Options, rewrite: Rewrite) -> Result<Certified, RewriteError> {
    let lifted = lift_object(base, options)?;
    let base_sha = sha256(base);
    let before = analyze(lifted.program, &rewrite.kernel).map_err(RewriteError::Analysis)?;
    let base_kernel = kernel(&before.program, &rewrite.kernel).clone();
    let base_obligations: BTreeSet<(String, Vec<InstId>)> =
        before.obligations.iter().map(key).collect();
    let mut current = before;
    let mut symbol = rewrite.kernel.clone();
    let (mut claims, mut delay_rewrites, mut reencoded) = (Vec::new(), Vec::new(), BTreeSet::new());
    for (step, edit) in rewrite.script.iter().enumerate() {
        let (next, delta) = current
            .edit(&symbol, edit.clone())
            .map_err(|error| RewriteError::Edit { step, error })?;
        for claim in delta.claims {
            if !claims.contains(&claim) {
                claims.push(claim);
            }
        }
        delay_rewrites.extend(delta.delay_rewrites);
        reencoded.extend(delta.reencoded_branches);
        symbol = delta.kernel;
        current = next;
    }
    let (bytes, relifted) = emit::checked(&current.program, options)?;
    let edited = kernel(&current.program, &symbol);
    let emitted = kernel(&relifted.program, &symbol);
    let definedness =
        definedness::check(current.program.target.arch, &base_kernel, edited, emitted);
    let new_obligations: Vec<Obligation> = current
        .obligations
        .iter()
        .filter(|o| !base_obligations.contains(&key(o)))
        .cloned()
        .collect();
    admit(&rewrite, current.program.target.arch, edited, base, &bytes)?;
    let receipt = Receipt {
        base_object_sha256: base_sha,
        object_sha256: sha256(&bytes),
        kernel_before: rewrite.kernel.0.clone(),
        kernel: symbol.0.clone(),
        script: rewrite.script.iter().map(|e| format!("{e:?}")).collect(),
        claims,
        delay_rewrites,
        reencoded_branches: reencoded.len(),
        new_obligations,
        definedness,
        equivalence: Equivalence {
            baseline_object_sha256: base_sha,
            kind: rewrite.claim.clone(),
        },
    };
    Ok(Certified {
        bytes,
        receipt,
        program: current,
    })
}

/// The equivalence rule (architecture.md §3 "two claims, never conflated").
fn admit(
    rewrite: &Rewrite,
    arch: Arch,
    edited: &Kernel,
    base: &[u8],
    bytes: &[u8],
) -> Result<(), RewriteError> {
    let refuse = |reason: String| {
        Err(RewriteError::Claim {
            claim: rewrite.claim.clone(),
            reason,
        })
    };
    match &rewrite.claim {
        EquivalenceKind::Unproved => Ok(()),
        EquivalenceKind::ByteIdentical if bytes == base => Ok(()),
        EquivalenceKind::ByteIdentical => refuse("the emitted bytes differ from the base".into()),
        EquivalenceKind::GpuBitExact { .. } => {
            refuse("no path-specific GPU oracle receipt is checked in M3".into())
        }
        EquivalenceKind::ProvedCommutation => {
            for (step, edit) in rewrite.script.iter().enumerate() {
                if !matches!(
                    edit,
                    Edit::Insert { .. } | Edit::Descriptor { .. } | Edit::Metadata { .. }
                ) {
                    return refuse(format!("step {step} is not an insertion or an ABI change"));
                }
            }
            for &id in &edited.body.layout {
                let inst = edited.body.insts.get(id).expect("laid out");
                if inst.prov.edit.is_none() {
                    continue;
                }
                let text = inst.text(arch).unwrap_or_default();
                if !matches!(inst.effects.control, Control::None | Control::Wait) {
                    return refuse(format!(
                        "inserted `{text}` is control flow, a barrier or a window"
                    ));
                }
                if inst.effects.lds.is_some()
                    || inst.effects.mem.is_some_and(|m| {
                        !matches!(m.class, MemClass::VmemStore | MemClass::SmemLoad)
                    })
                {
                    return refuse(format!("inserted `{text}` accesses memory other than a vector store or a scalar load"));
                }
            }
            Ok(())
        }
    }
}

/// A kernel's assembly source as the anchors of a script: its instruction lines, one per
/// layout instruction (checked by mnemonic), their `.s` line numbers, and its labels as
/// layout indices.
#[derive(Clone, Debug)]
pub struct SourceMap {
    pub lines: Vec<String>,
    pub line_no: Vec<usize>,
    pub labels: HashMap<String, usize>,
}

impl SourceMap {
    /// Map `kernel` onto `source`, whose instructions from the label named after the
    /// kernel on must be the kernel's layout.
    pub fn new(kernel: &Kernel, source: &str) -> Result<Self, RewriteError> {
        let fail = |reason: String| RewriteError::Anchor(reason);
        let file = parse_source(source).map_err(|e| fail(e.to_string()))?;
        let insts: Vec<(usize, &str)> = file
            .lines
            .iter()
            .filter_map(|l| match &l.kind {
                SourceKind::Inst(text) => Some((l.line_no, text.as_str())),
                _ => None,
            })
            .collect();
        let first = file
            .labels
            .iter()
            .find(|(l, _)| *l == kernel.symbol.0)
            .map(|(_, at)| *at)
            .ok_or_else(|| fail(format!("{} is not a label in the source", kernel.symbol.0)))?;
        let n = kernel.body.layout.len();
        let own = insts
            .get(first..first + n)
            .ok_or_else(|| fail("the source holds fewer instructions than the kernel".into()))?;
        for (k, ((_, text), &id)) in own.iter().zip(&kernel.body.layout).enumerate() {
            let name = mnemonic(kernel.body.insts.get(id).expect("laid out"));
            if text.split_whitespace().next() != Some(name) {
                return Err(fail(format!(
                    "source instruction {k} {text:?} is not the kernel's {name}"
                )));
            }
        }
        Ok(Self {
            lines: own.iter().map(|(_, t)| (*t).to_owned()).collect(),
            line_no: own.iter().map(|(n, _)| *n).collect(),
            labels: file
                .labels
                .iter()
                .filter(|(_, at)| (first..first + n).contains(at))
                .map(|(l, at)| (l.clone(), at - first))
                .collect(),
        })
    }

    /// Layout index of the instruction on `.s` line `line`.
    pub fn at_line(&self, line: usize) -> Result<usize, RewriteError> {
        self.line_no.iter().position(|&n| n == line).ok_or_else(|| {
            RewriteError::Anchor(format!(
                ".s line {line} is not an instruction of the kernel"
            ))
        })
    }

    /// Layout index of the instruction `label` precedes.
    pub fn at_label(&self, label: &str) -> Result<usize, RewriteError> {
        self.labels
            .get(label)
            .copied()
            .ok_or_else(|| RewriteError::Anchor(format!("label {label} is not in the kernel")))
    }
}

/// The table mnemonic of `inst` (by opcode and form: VMEM forms share opcode ids, so
/// `Opcode::name` alone can name the wrong form).
pub fn mnemonic(inst: &peacemaker_ir::inst::Inst) -> &'static str {
    peacemaker_ir::isa::lookup(Arch::Gfx1201, inst.op, inst.form).map_or("", |row| row.name)
}

/// Block and instruction id at layout index `at` of `kernel`.
pub fn position(kernel: &Kernel, at: usize) -> (BlockId, InstId) {
    let block = kernel
        .body
        .blocks
        .iter()
        .position(|b| b.range.0 <= at && at < b.range.1)
        .expect("laid out");
    (BlockId(block), kernel.body.layout[at])
}
