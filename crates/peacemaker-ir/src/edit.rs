//! C6: checked edit transactions over `Analyzed<Program>` (core.md §6).
//!
//! `Analyzed::edit` = validate preconditions → apply to a clone →
//! `Program::validate` → re-analyse → `(Analyzed{revision+1}, EditDelta)`.
//! Facts are never patched: the candidate is re-analysed from scratch by
//! [`analyze`], which is also the `Lifted → Analyzed` entry point.
//!
//! Precondition facts are computed here on an instruction-level flow graph
//! (every branch is an exit of its own position), with dword-granular
//! register locations split into `.l`/`.h` halves:
//!
//! * liveness (backward, may): explicit operand roles come from the C1 table
//!   (`Effects::from_table`, probed per operand), implicit reads/writes from
//!   the row's implicit column, and every vector-issuing form additionally
//!   reads EXEC. Scalar writes kill; a VGPR write kills only reads made under
//!   the same EXEC, i.e. not across an EXEC write (lanes outside EXEC keep
//!   their old value; divergence itself is not modelled).
//! * definedness (forward, must): seeded with the gfx12 ABI entry table (user
//!   SGPRs the descriptor enables, `ttmp9`/`ttmp7` work-group ids, `v0`,
//!   EXEC, hardware state); everything else (M0, VCC, SCC, other SGPRs,
//!   VGPRs, TTMPs) starts undefined. A half write defines only its half.
//!
//! Inverses: every edit returns a natural inverse, and the delta's `inverse`
//! wraps it in [`Edit::Revert`], which is admitted only when the kernel is
//! exactly the post-edit state and must land exactly on the base state
//! (digest-checked both ways), so undo never needs to re-prove semantic
//! preconditions and restores instruction ids, hints and ABI bytes.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::{self, Write as _};
use std::hash::{DefaultHasher, Hasher};

use smallvec::SmallVec;
use thiserror::Error;

use crate::cfg::{Arena, Block, BlockId, Body, InstId, Terminator};
use crate::effects::{preserved_vgpr_destination, Control, Effects, ImplicitSet};
use crate::inst::{Abi, Arch, Form, FormFields, Inst, Kernel, Program, SymbolId, UserSgprRole, ValidateError, Wave};
use crate::lds::AddrFact;
use crate::metadata::Kernarg;
use crate::operand::{DelayAluHint, ImmField, Msg, Operand, Special};
use crate::passes;
use crate::provenance::{EditId, Provenance, Source};
use crate::reg::{Kind, RegClaim, RegRef, RegSet, Scope};
use crate::state::{Analyzed, Facts, Obligation, ObligationKind};
use crate::wait::{Counter, WaitImm, N};

#[path = "edit/lane_definedness.rs"]
mod lane_definedness;

// ---------------------------------------------------------------------------
// Public edit vocabulary
// ---------------------------------------------------------------------------

/// What a cursor does when it lands inside an `s_clause` window or between an
/// `s_delay_alu` and its last target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowPolicy { Reject, MoveUp, MoveDown }

/// An insertion gap: before `before` inside `block`, or at the end of `block`
/// (`None`, only for blocks that fall through).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cursor { pub block: BlockId, pub before: Option<InstId>, pub window_policy: WindowPolicy }
impl Cursor {
    pub fn before(block: BlockId, inst: InstId) -> Self { Self { block, before: Some(inst), window_policy: WindowPolicy::Reject } }
    pub fn end(block: BlockId) -> Self { Self { block, before: None, window_policy: WindowPolicy::Reject } }
    pub fn with_policy(self, window_policy: WindowPolicy) -> Self { Self { window_policy, ..self } }
}

/// A contiguous layout run `first..=last` inside one block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstRange { pub first: InstId, pub last: InstId }
impl InstRange {
    pub fn single(id: InstId) -> Self { Self { first: id, last: id } }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DescriptorChange {
    KernargSize(u32),
    /// Sets `rsrc1` GRANULATED_WORKITEM_VGPR_COUNT to `max(0, ceil(n/granule)-1)`.
    NextFreeVgpr(u32),
    /// Refused on GFX10–12: the SGPR granule is reserved there (128 always allocated).
    NextFreeSgpr(u32),
    GroupSegmentFixedSize(u32),
    /// Renames the symbol, its `.kd`, metadata `.name`/`.symbol` and the ELF slot together.
    Rename(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MetaChange {
    /// Appends after the last argument; `.kernarg_segment_size` grows to cover them.
    AppendArgs(Vec<Kernarg>),
    /// Replaces the argument list and segment size (the inverse form of `AppendArgs`).
    SetArgs { args: Vec<Kernarg>, kernarg_segment_size: u32 },
    SetVgprCount(u32),
    SetSgprCount(u32),
    Rename(String),
}

/// Digest of one kernel's live state (symbol, ABI, layout, blocks, live
/// instructions, ELF slot name); tombstoned arena slots do not participate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct StateDigest(pub u64);

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Edit {
    Insert { at: Cursor, insts: Vec<Inst>, claims: Vec<RegClaim> },
    Remove { range: InstRange },
    Replace { range: InstRange, insts: Vec<Inst> },
    /// Identity-preserving remove+insert; ids and `prov.bytes` are kept.
    Move { range: InstRange, to: Cursor },
    ReRegister { scope: Scope, map: Vec<(RegRef, RegRef)>, claims: Vec<RegClaim> },
    /// Materialises a leader at `at.before`.
    SplitBlock { at: Cursor },
    /// Only for an edit-inserted branch (an original branch needs an oracle).
    Retarget { branch: InstId, to: BlockId },
    SetWait { inst: InstId, imm: WaitImm },
    SetDelayHint { inst: InstId, hint: DelayAluHint },
    Descriptor { kernel: SymbolId, change: DescriptorChange },
    Metadata { kernel: SymbolId, change: MetaChange },
    /// Inverse of `Remove`/`Replace`: re-inserts tombstoned instructions under their ids.
    Restore { at: Cursor, insts: Vec<(InstId, Inst)> },
    /// Inverse of `SplitBlock`: joins `block` to its fall-through predecessor.
    MergeBlock { block: BlockId },
    /// Steps in order, each checked against the state left by the previous
    /// one; one validation and re-analysis at the end.
    Batch(Vec<Edit>),
    /// Undo by identity: admitted only when the kernel digest is `from`, the
    /// `undo` steps are applied without semantic preconditions, and the
    /// result must digest to `to`.
    Revert { from: StateDigest, to: StateDigest, undo: Box<Edit> },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DelayRewrite { pub hint: InstId, pub before: DelayAluHint, pub after: DelayAluHint }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MoveDirection { Down, Up }

/// The total commutation proof of one `Move`: the run is control-equivalent
/// to its destination and commutes with every skipped instruction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommutationProof {
    pub moved: Vec<InstId>,
    pub direction: MoveDirection,
    pub skipped: Vec<InstId>,
    /// DS pairs reordered only because their constant LDS footprints are disjoint.
    pub lds_disjoint: Vec<(InstId, InstId)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditDelta {
    pub id: EditId,
    pub base_revision: u32,
    /// The kernel's symbol after the edit: apply `inverse` against this name.
    pub kernel: SymbolId,
    pub applied: Edit,
    pub inverse: Edit,
    /// Registers read or written by inserted, removed, moved or renamed instructions.
    pub touched: RegSet,
    /// Every live instruction's byte offset from the kernel entry in the new
    /// layout; `None` for instructions this edit removed.
    pub inst_map: Vec<(InstId, Option<u32>)>,
    pub claims: Vec<RegClaim>,
    /// `s_delay_alu` hints rewritten automatically (producer distance changed).
    pub delay_rewrites: Vec<DelayRewrite>,
    /// Original branches whose simm16 now differs from `prov.bytes` (distance changed).
    pub reencoded_branches: Vec<InstId>,
    pub commutations: Vec<CommutationProof>,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum EditError {
    #[error("no kernel named {0}")]
    UnknownKernel(String),
    #[error("edit names kernel {edit} but the transaction targets {target}")]
    KernelMismatch { edit: String, target: String },
    #[error("kernel has no HSA descriptor/metadata")]
    NotHsa,
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("candidate program is invalid: {0}")]
    Invalid(#[from] ValidateError),
    #[error("re-analysis failed: {0}")]
    Analysis(String),
    #[error("instruction {0:?} is not laid out in this kernel")]
    UnknownInst(InstId),
    #[error("block {0:?} does not exist")]
    UnknownBlock(BlockId),
    #[error("cursor instruction {inst:?} is not in block {block:?}")]
    CursorNotInBlock { block: BlockId, inst: InstId },
    #[error("cursor at the end of block {0:?}, after its control instruction")]
    CursorAfterTerminator(BlockId),
    #[error("cursor before layout position {pos} is inside the window opened by {opener:?}")]
    CursorInWindow { pos: usize, opener: InstId },
    #[error("cursor before layout position {pos} is reachable after s_sendmsg(MSG_DEALLOC_VGPRS)")]
    AfterDealloc { pos: usize },
    #[error("inserted instruction {index}: {reason}")]
    InvalidInst { index: usize, reason: String },
    #[error("inserted instruction {index}: {reason}")]
    ControlInsert { index: usize, reason: String },
    #[error("writes {reg}, which is live at the cursor and not covered by a sound claim")]
    WriteLive { reg: String },
    #[error("{inst:?} reads {regs} without a definition on every incoming path")]
    Undefined { inst: InstId, regs: String },
    #[error("claim {claim}: {reason}")]
    ClaimUnsound { claim: String, reason: String },
    #[error("range {0:?} is not one contiguous run inside one block")]
    BadRange(InstRange),
    #[error("range {0:?} would leave its block empty")]
    RangeEmptiesBlock(InstRange),
    #[error("{inst:?} is a control-flow instruction that this edit may not remove or move")]
    ControlRemove { inst: InstId },
    #[error("removes the definition of {reg} by {inst:?}, which is live after the range")]
    RemoveLiveDef { inst: InstId, reg: String },
    #[error("wait {wait:?} satisfies the still-pending event issued by {event:?}")]
    WaitInUse { wait: InstId, event: InstId },
    #[error("range cuts the window opened by {opener:?}")]
    WindowBroken { opener: InstId },
    #[error("move is empty or overlaps its destination")]
    NoOpMove,
    #[error("move destination is not control-equivalent to the run: {0}")]
    NotControlEquivalent(String),
    #[error("moved {moved:?} does not commute with {skipped:?}: {reason}")]
    NotCommutable { moved: InstId, skipped: InstId, reason: String },
    #[error("re-register: {0}")]
    ReRegister(String),
    #[error("split: {0}")]
    Split(String),
    #[error("merge: {0}")]
    Merge(String),
    #[error("{0:?} is an original branch: retargeting it changes which instructions execute (oracle required)")]
    RetargetOriginal(InstId),
    #[error("retarget: {0}")]
    Retarget(String),
    #[error("barrier pairing regresses: {0}")]
    BarrierPairing(String),
    #[error("{0:?} is not a counter wait")]
    NotAWait(InstId),
    #[error("wait immediate: {0}")]
    WaitImm(String),
    #[error("loosening {wait:?} stops it retiring the event issued by {event:?}")]
    WaitLoosened { wait: InstId, event: InstId },
    #[error("{0:?} is not an s_delay_alu")]
    NotADelay(InstId),
    #[error("delay hint: {0}")]
    DelayHint(String),
    #[error("descriptor: {0}")]
    Descriptor(String),
    #[error("metadata: {0}")]
    Metadata(String),
    #[error("rename: {0}")]
    Rename(String),
    #[error("revert expected kernel state {expected:?}, found {found:?}")]
    RevertStale { expected: StateDigest, found: StateDigest },
    #[error("revert landed on {found:?} instead of {expected:?}")]
    RevertDiverged { expected: StateDigest, found: StateDigest },
    #[error("label: {0}")]
    Label(String),
    #[error("encode {inst:?}: {reason}")]
    Encode { inst: InstId, reason: String },
}

// ---------------------------------------------------------------------------
// Analysis driver (Lifted → Analyzed; also the re-analysis after every edit)
// ---------------------------------------------------------------------------

/// Analyse one kernel of `program` (revision 0). Blocks must be built
/// (`passes::cfg::build_blocks`). Obligations are the union of the wait
/// replay, hazard, barrier, definedness and resource/ABI findings;
/// `facts.resources` holds this kernel's summary.
pub fn analyze(program: Program, kernel: &SymbolId) -> Result<Analyzed<Program>, EditError> {
    analyze_at(program, kernel, 0)
}

fn analyze_at(program: Program, kernel: &SymbolId, revision: u32) -> Result<Analyzed<Program>, EditError> {
    program.validate()?;
    let k = kernel_index(&program, kernel)?;
    let arch = program.target.arch;
    let kern = &program.kernels[k];
    let body = &kern.body;
    let fail = |what: &str, e: &dyn fmt::Display| EditError::Analysis(format!("{what}: {e}"));
    passes::windows::check_windows(body).map_err(|e| fail("windows", &e))?;
    let replay = passes::waits::replay(body, arch).map_err(|e| fail("waits", &e))?;
    let hazards = passes::hazards::analyze(body, arch, kern.wave).map_err(|e| fail("hazards", &e))?;
    let lds = passes::lds::analyze(body, arch).map_err(|e| fail("lds", &e))?;
    let barriers = passes::barriers::analyze(body, arch).map_err(|e| fail("barriers", &e))?;
    let lds_fixed = match &kern.abi { Abi::Hsa { descriptor, .. } => descriptor.group_segment_fixed_size, Abi::Raw { lds_bytes, .. } => *lds_bytes };
    let summary = passes::resources::summarize(body, arch, kern.wave, lds_fixed);
    let mut obligations = replay.obligations;
    obligations.extend(hazards.obligations);
    obligations.extend(barriers.obligations);
    let flow = Flow::new(body, arch, kern.wave)?;
    let (def_in, _) = flow.defined(entry_seed(kern, arch)?);
    let partial_sites: Vec<_> = (0..flow.n).filter(|&p| {
        let inst = body.insts.get(flow.ids[p]).expect("laid out");
        !partial_undefined(arch, inst, &flow.acc[p].reads.minus(&def_in[p])).is_empty()
    }).collect();
    let lane_proofs = if partial_sites.is_empty() { Vec::new() } else {
        lane_definedness::prove(body, &flow, arch, kern.wave, &partial_sites)
    };
    for p in 0..flow.n {
        let id = flow.ids[p];
        let inst = body.insts.get(id).ok_or(EditError::UnknownInst(id))?;
        let missing = flow.acc[p].reads.minus(&def_in[p]);
        let partial = partial_undefined(arch, inst, &missing);
        if !partial.is_empty() && !lane_proofs.contains(&p) {
            obligations.push(Obligation {
                kind: ObligationKind::Definedness, insts: vec![id], rule_id: "definedness-partial-write".into(),
                text: format!("{} preserves {} without a definition on every path; lane coverage remains unproved",
                    inst.op.name(arch).unwrap_or("unknown opcode"), bits_names(&partial)),
            });
        }
        let missing = missing.minus(&partial);
        if !missing.is_empty() {
            obligations.push(Obligation {
                kind: ObligationKind::Definedness, insts: vec![id], rule_id: "definedness-entry".into(),
                text: format!("reads {} without a definition on every path from the kernel entry (ABI entry table)", bits_names(&missing)),
            });
        }
    }
    obligations.extend(abi_obligations(kern, &summary));
    let kernargs = passes::kernargs::analyze(kern, arch);
    Ok(Analyzed { program, facts: Facts { waits: replay.facts, lds: lds.facts, resources: vec![summary], kernargs }, obligations, revision })
}

fn abi_obligations(kern: &Kernel, summary: &crate::state::ResourceSummary) -> Vec<Obligation> {
    let Abi::Hsa { descriptor, metadata } = &kern.abi else { return Vec::new() };
    let meta = &metadata.parsed;
    let wave32 = descriptor.kernel_code_properties.wave32();
    let mut out = Vec::new();
    let mut push = |rule: &str, text: String| out.push(Obligation { kind: ObligationKind::Unknown, insts: Vec::new(), rule_id: rule.into(), text });
    let max_vgpr = u32::from(summary.max_vgpr);
    let next_free = descriptor.compute_pgm_rsrc1.next_free_vgpr(wave32);
    if max_vgpr > next_free { push("resource-vgpr-descriptor", format!("code uses {max_vgpr} VGPRs, descriptor allocates {next_free}")); }
    if max_vgpr > meta.vgpr_count { push("resource-vgpr-metadata", format!("code uses {max_vgpr} VGPRs, .vgpr_count is {}", meta.vgpr_count)); }
    if meta.vgpr_count > next_free {
        push("abi-vgpr-allocation", format!(".vgpr_count {} exceeds the descriptor allocation {next_free}", meta.vgpr_count));
    }
    if descriptor.private_segment_fixed_size != 0 {
        push("scratch-in-use", format!("kernel allocates {} bytes of architected flat scratch per work-item; scratch address bounds are not certified", descriptor.private_segment_fixed_size));
    }
    let sgprs = u32::from(summary.max_sgpr) + if summary.uses_vcc { 2 } else { 0 };
    if sgprs > meta.sgpr_count { push("resource-sgpr-metadata", format!("code needs {sgprs} SGPRs (VCC included), .sgpr_count is {}", meta.sgpr_count)); }
    if meta.kernarg_segment_size != descriptor.kernarg_size {
        push("abi-kernarg-size", format!(".kernarg_segment_size {} but descriptor kernarg_size {}", meta.kernarg_segment_size, descriptor.kernarg_size));
    }
    if let Some(arg) = meta.args.iter().find(|a| u64::from(a.offset) + u64::from(a.size) > u64::from(meta.kernarg_segment_size)) {
        push("abi-kernarg-args", format!("argument {} ends past .kernarg_segment_size", arg.name));
    }
    if meta.group_segment_fixed_size != descriptor.group_segment_fixed_size {
        push("abi-group-segment", format!(".group_segment_fixed_size {} but descriptor {}", meta.group_segment_fixed_size, descriptor.group_segment_fixed_size));
    }
    if meta.private_segment_fixed_size != descriptor.private_segment_fixed_size {
        push("abi-private-segment", format!(".private_segment_fixed_size {} but descriptor {}", meta.private_segment_fixed_size, descriptor.private_segment_fixed_size));
    }
    out
}

impl Analyzed<Program> {
    /// The only mutation path (core.md §6.1). `kernel` names the kernel whose
    /// body the edit addresses, by its symbol before the edit.
    pub fn edit(&self, kernel: &SymbolId, edit: Edit) -> Result<(Analyzed<Program>, EditDelta), EditError> {
        let k = kernel_index(&self.program, kernel)?;
        let id = EditId(u64::from(self.revision) + 1);
        let base = digest(&self.program, k);
        let mut tx = Tx::new(self.program.clone(), k, id);
        let natural = tx.step(&edit, true)?;
        let Tx { program, removed, touched, claims, delay_rewrites, commutations, .. } = tx;
        program.validate()?;
        let post = digest(&program, k);
        let inverse = match natural {
            revert @ Edit::Revert { .. } => revert,
            undo => Edit::Revert { from: post, to: base, undo: Box::new(undo) },
        };
        let symbol = program.kernels[k].symbol.clone();
        let body = &program.kernels[k].body;
        let lowered = lower_labels(body, program.target.arch)?;
        let mut inst_map = Vec::with_capacity(body.layout.len() + removed.len());
        let mut offset = 0u32;
        let mut reencoded_branches = Vec::new();
        for (&id, low) in body.layout.iter().zip(&lowered) {
            inst_map.push((id, Some(offset)));
            offset += 4 * passes::cfg::dwords_of(low, program.target.arch) as u32;
            let inst = body.insts.get(id).expect("laid out");
            if matches!(inst.effects.control, Control::Branch { .. } | Control::Jump) {
                if let (Some(bytes), Ok(words)) = (inst.prov.bytes, crate::codec::gfx12::encode_for(program.target.arch, low)) {
                    if words.first() != Some(&bytes[0]) { reencoded_branches.push(id); }
                }
            }
        }
        inst_map.extend(removed.iter().filter(|id| body.insts.get(**id).is_none()).map(|&id| (id, None)));
        let delta = EditDelta {
            id, base_revision: self.revision, kernel: symbol.clone(), applied: edit, inverse,
            touched: bits_regset(&touched), inst_map, claims, delay_rewrites, reencoded_branches, commutations,
        };
        let analyzed = analyze_at(program, &symbol, self.revision + 1)?;
        Ok((analyzed, delta))
    }

    /// Apply `delta.inverse` (an identity-checked [`Edit::Revert`]).
    pub fn undo(&self, delta: &EditDelta) -> Result<(Analyzed<Program>, EditDelta), EditError> {
        self.edit(&delta.kernel, delta.inverse.clone())
    }
}

fn kernel_index(program: &Program, kernel: &SymbolId) -> Result<usize, EditError> {
    program.kernels.iter().position(|k| &k.symbol == kernel).ok_or_else(|| EditError::UnknownKernel(kernel.0.clone()))
}

/// Digest of kernel `k`'s live state (see [`StateDigest`]).
pub fn digest(program: &Program, k: usize) -> StateDigest {
    struct Feed<'a>(&'a mut DefaultHasher);
    impl fmt::Write for Feed<'_> {
        fn write_str(&mut self, s: &str) -> fmt::Result { self.0.write(s.as_bytes()); Ok(()) }
    }
    let kern = &program.kernels[k];
    let mut hasher = DefaultHasher::new();
    let mut feed = Feed(&mut hasher);
    let _ = write!(feed, "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}", kern.symbol, kern.wave, kern.abi, kern.origin, kern.body.layout, kern.body.blocks);
    for &id in &kern.body.layout {
        let _ = write!(feed, "|{}:{:?}", id.0, kern.body.insts.get(id));
    }
    if let Some(source) = &program.source {
        let _ = write!(feed, "|{:?}", source.elf.kernels.get(k).map(|slot| &slot.name));
    }
    StateDigest(hasher.finish())
}

// ---------------------------------------------------------------------------
// Label lowering and stream encoding
// ---------------------------------------------------------------------------

/// Layout-ordered instructions with every `Operand::Label(b)` lowered to the
/// SOPP simm16 dword offset from the next PC, computed from the current layout.
pub fn lower_labels(body: &Body, arch: Arch) -> Result<Vec<Inst>, EditError> {
    let mut pcs = Vec::with_capacity(body.layout.len() + 1);
    let mut pc = 0usize;
    for &id in &body.layout {
        pcs.push(pc);
        pc += passes::cfg::dwords_of(body.insts.get(id).ok_or(EditError::UnknownInst(id))?, arch);
    }
    let mut out = Vec::with_capacity(body.layout.len());
    for (index, &id) in body.layout.iter().enumerate() {
        let mut inst = body.insts.get(id).ok_or(EditError::UnknownInst(id))?.clone();
        let next = pcs[index] + passes::cfg::dwords_of(&inst, arch);
        for operand in inst.operands.iter_mut() {
            if let Operand::Label(target) = operand {
                let block = body.blocks.get(target.0).ok_or_else(|| EditError::Label(format!("{id:?} targets missing block {target:?}")))?;
                let target_pc = *pcs.get(block.range.0).ok_or_else(|| EditError::Label(format!("block {target:?} is empty")))?;
                let offset = target_pc as i64 - next as i64;
                let offset = i16::try_from(offset).map_err(|_| EditError::Label(format!("{id:?} branch offset {offset} does not fit simm16")))?;
                *operand = Operand::Imm(ImmField::Sopp(offset));
            }
        }
        out.push(inst);
    }
    Ok(out)
}

/// Encode a kernel body (labels lowered from its current layout) with the
/// selected target codec. Never reads `prov.bytes`.
pub fn encode_stream(body: &Body, arch: Arch) -> Result<Vec<u32>, EditError> {
    if !matches!(arch, Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1201) { return Err(EditError::Unsupported(format!("no codec for {arch:?}"))); }
    let mut words = Vec::new();
    for (inst, &id) in lower_labels(body, arch)?.iter().zip(&body.layout) {
        words.extend(crate::codec::gfx12::encode_for(arch, inst).map_err(|e| EditError::Encode { inst: id, reason: e.to_string() })?);
    }
    Ok(words)
}

// ---------------------------------------------------------------------------
// Register locations (dword × half) and per-instruction access sets
// ---------------------------------------------------------------------------

const S0: usize = 256;
const T0: usize = 362;
const VCC_LO: usize = 378;
const VCC_HI: usize = 379;
const EXEC_LO: usize = 380;
const EXEC_HI: usize = 381;
const SCC: usize = 382;
const M0: usize = 383;
const MODE: usize = 384;
/// flat_scratch, src_shared_base/limit, src_private_base/limit, pc, hwreg.
const OTHER0: usize = 385;
const NDW: usize = 392;
const WORDS: usize = (NDW * 2).div_ceil(64);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Bits([u64; WORDS]);
impl Bits {
    const FULL: Bits = Bits([u64::MAX; WORDS]);
    fn bit(&mut self, b: usize) { self.0[b / 64] |= 1 << (b % 64); }
    fn has(&self, b: usize) -> bool { self.0[b / 64] >> (b % 64) & 1 != 0 }
    fn dword(&mut self, d: usize) { self.bit(2 * d); self.bit(2 * d + 1); }
    fn half(&mut self, d: usize, hi: bool) { self.bit(2 * d + usize::from(hi)); }
    fn or(&mut self, other: &Bits) { for (a, b) in self.0.iter_mut().zip(other.0) { *a |= b; } }
    fn union(&self, other: &Bits) -> Bits { let mut out = *self; out.or(other); out }
    fn inter(&self, other: &Bits) -> Bits { let mut out = *self; for (a, b) in out.0.iter_mut().zip(other.0) { *a &= b; } out }
    fn minus(&self, other: &Bits) -> Bits { let mut out = *self; for (a, b) in out.0.iter_mut().zip(other.0) { *a &= !b; } out }
    fn is_empty(&self) -> bool { self.0.iter().all(|w| *w == 0) }
    fn ones(&self) -> impl Iterator<Item = usize> + '_ { (0..NDW * 2).filter(|&b| self.has(b)) }
    fn dwords(&self) -> BTreeSet<usize> { self.ones().map(|b| b / 2).collect() }
    fn of_reg(reg: RegRef) -> Bits { let mut out = Bits::default(); for d in reg_dwords(reg) { out.dword(d); } out }
}

fn reg_dwords(reg: RegRef) -> std::ops::Range<usize> {
    let (base, limit) = match reg.kind { Kind::V => (0, 256), Kind::S => (S0, 106), Kind::Ttmp => (T0, 16) };
    let lo = usize::from(reg.base).min(limit);
    let hi = (usize::from(reg.base) + usize::from(reg.len)).min(limit);
    base + lo..base + hi
}

fn special_dwords(special: Special) -> SmallVec<[usize; 2]> {
    match special {
        Special::Vcc => smallvec::smallvec![VCC_LO, VCC_HI],
        Special::VccLo => smallvec::smallvec![VCC_LO],
        Special::VccHi => smallvec::smallvec![VCC_HI],
        Special::Exec => smallvec::smallvec![EXEC_LO, EXEC_HI],
        Special::ExecLo => smallvec::smallvec![EXEC_LO],
        Special::ExecHi => smallvec::smallvec![EXEC_HI],
        Special::Scc => smallvec::smallvec![SCC],
        Special::M0 => smallvec::smallvec![M0],
        Special::Null => SmallVec::new(),
        Special::Ttmp(n) => smallvec::smallvec![T0 + usize::from(n).min(15)],
        Special::FlatScratch => smallvec::smallvec![OTHER0],
        Special::SrcSharedBase => smallvec::smallvec![OTHER0 + 1],
        Special::SrcSharedLimit => smallvec::smallvec![OTHER0 + 2],
        Special::SrcPrivateBase => smallvec::smallvec![OTHER0 + 3],
        Special::SrcPrivateLimit => smallvec::smallvec![OTHER0 + 4],
        Special::Pc => smallvec::smallvec![OTHER0 + 5],
    }
}

fn implicit_dwords(bits: u8, wave: Wave) -> SmallVec<[usize; 6]> {
    let mut out = SmallVec::new();
    let wide = wave == Wave::Wave64;
    if bits & ImplicitSet::SCC != 0 { out.push(SCC); }
    if bits & ImplicitSet::VCC != 0 { out.push(VCC_LO); if wide { out.push(VCC_HI); } }
    if bits & ImplicitSet::EXEC != 0 { out.push(EXEC_LO); if wide { out.push(EXEC_HI); } }
    if bits & ImplicitSet::M0 != 0 { out.push(M0); }
    if bits & ImplicitSet::MODE != 0 { out.push(MODE); }
    out
}

fn dword_name(d: usize) -> String {
    match d {
        0..=255 => format!("v{d}"),
        S0..=361 => format!("s{}", d - S0),
        T0..=377 => format!("ttmp{}", d - T0),
        VCC_LO => "vcc_lo".into(), VCC_HI => "vcc_hi".into(), EXEC_LO => "exec_lo".into(), EXEC_HI => "exec_hi".into(),
        SCC => "scc".into(), M0 => "m0".into(), MODE => "mode".into(),
        _ => ["flat_scratch", "src_shared_base", "src_shared_limit", "src_private_base", "src_private_limit", "pc", "hwreg"]
            .get(d - OTHER0).copied().unwrap_or("?").into(),
    }
}

fn bits_names(bits: &Bits) -> String {
    bits.ones().map(|b| {
        let partial = !bits.has(b ^ 1);
        let name = dword_name(b / 2);
        if partial { format!("{name}.{}", if b % 2 == 0 { "l" } else { "h" }) } else { name }
    }).collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>().join(", ")
}

fn dword_reg(d: usize) -> Option<RegRef> {
    match d {
        0..=255 => Some(RegRef { kind: Kind::V, base: d as u16, len: 1 }),
        S0..=361 => Some(RegRef { kind: Kind::S, base: (d - S0) as u16, len: 1 }),
        T0..=377 => Some(RegRef { kind: Kind::Ttmp, base: (d - T0) as u16, len: 1 }),
        _ => None,
    }
}

/// Coalesced V/S/Ttmp ranges of a location set (specials are not `RegRef`s).
fn bits_regset(bits: &Bits) -> RegSet {
    let mut out: Vec<RegRef> = Vec::new();
    for reg in bits.dwords().into_iter().filter_map(dword_reg) {
        match out.last_mut() {
            Some(last) if last.kind == reg.kind && last.base + u16::from(last.len) == reg.base && last.len < u8::MAX => last.len += 1,
            _ => out.push(reg),
        }
    }
    RegSet(out)
}

/// What one instruction reads and writes, by location half.
#[derive(Clone, Copy, Debug, Default)]
struct Access {
    /// Values the instruction consumes (definedness and liveness gens).
    reads: Bits,
    writes: Bits,
    /// Every location the table names as a use, including the preserved half
    /// of a partial write (commutation).
    uses: Bits,
}

fn row_implicit(arch: Arch, op: crate::inst::Opcode, form: Form) -> (u8, u8) {
    let Some(row) = crate::isa::lookup(arch, op, form) else { return (0, 0) };
    let (mut reads, mut writes) = (0, 0);
    for effect in row.implicit.split(',') {
        let (reg, mode) = effect.split_once(':').unwrap_or(("", ""));
        let bit = match reg { "scc" => ImplicitSet::SCC, "vcc" => ImplicitSet::VCC, "exec" => ImplicitSet::EXEC, "m0" => ImplicitSet::M0, "mode" => ImplicitSet::MODE, _ => continue };
        if mode.contains('r') { reads |= bit; }
        if mode.contains('w') { writes |= bit; }
    }
    (reads, writes)
}

/// (def, use) role of every explicit operand, taken from the C1 table itself
/// by probing `Effects::from_table` with one marker register per operand.
fn operand_roles(arch: Arch, inst: &Inst) -> Result<Vec<(bool, bool)>, EditError> {
    let row = crate::isa::lookup(arch, inst.op, inst.form)
        .ok_or_else(|| EditError::Unsupported(format!("no {arch:?} row for {:?}/{:?}", inst.op, inst.form)))?;
    let marker = |k: usize| RegRef { kind: Kind::V, base: 0x4000 + k as u16, len: 1 };
    let vopc_prefix = inst.form == Form::Vopc && row.name.starts_with("v_cmp_")
        && matches!(inst.operands.first(), Some(Operand::Special(Special::Vcc | Special::VccLo)));
    let probe: SmallVec<[Operand; 6]> = inst.operands.iter().enumerate().map(|(k, op)| match op {
        Operand::Special(_) if k == 0 && vopc_prefix => op.clone(),
        Operand::Reg(_) | Operand::Special(_) => Operand::Reg(marker(k)),
        Operand::Half(_, half) => Operand::Half(marker(k), *half),
        _ => op.clone(),
    }).collect();
    let table = |e: ValidateError| EditError::Unsupported(e.to_string());
    let effects = match &inst.fields {
        FormFields::Vopd { y_op, x_operands } => {
            let split = usize::from(*x_operands).min(probe.len());
            let mut x = Effects::from_table(arch, inst.op, inst.form, &probe[..split], &inst.mods.cpol).map_err(table)?;
            let y = Effects::from_table(arch, *y_op, Form::Vopd, &probe[split..], &inst.mods.cpol).map_err(table)?;
            x.defs.extend(y.defs);
            x.uses.extend(y.uses);
            x
        }
        _ => Effects::from_table(arch, inst.op, inst.form, &probe, &inst.mods.cpol).map_err(table)?,
    };
    Ok((0..inst.operands.len()).map(|k| {
        if k == 0 && vopc_prefix { return (true, false); }
        (effects.defs.contains(&marker(k)), effects.uses.contains(&marker(k)))
    }).collect())
}

fn is_vector_form(form: Form) -> bool {
    matches!(form, Form::Vop1 | Form::Vop2 | Form::Vop3 | Form::Vop3p | Form::Vopc | Form::Vopd | Form::Vinterp | Form::Ds | Form::Vmem(_) | Form::Export)
}

fn access(arch: Arch, wave: Wave, inst: &Inst) -> Result<Access, EditError> {
    let roles = operand_roles(arch, inst)?;
    let mut acc = Access::default();
    let name = inst.op.name(arch).unwrap_or("");
    for (index, (operand, &(def, used))) in inst.operands.iter().zip(&roles).enumerate() {
        match operand {
            Operand::Reg(reg) => {
                let bits = Bits::of_reg(crate::codec::gfx12::operand_register(arch, wave, inst, index).unwrap_or(*reg));
                if def { acc.writes.or(&bits); }
                if used { acc.reads.or(&bits); acc.uses.or(&bits); }
            }
            Operand::Half(reg, half) => {
                let Some(d) = reg_dwords(*reg).next() else { continue };
                let hi = matches!(half, crate::operand::Half::Hi);
                if def { acc.writes.half(d, hi); }
                if used {
                    acc.uses.dword(d);
                    // A half destination listed as a use is the preserved
                    // other half, not a value this instruction consumes: the
                    // half-granular state keeps that half's own definedness,
                    // so any later reader of it is checked where it reads.
                    if !def { acc.reads.half(d, hi); }
                }
            }
            Operand::Special(special) => {
                for d in special_dwords(*special) {
                    if def { acc.writes.dword(d); }
                    if used { acc.reads.dword(d); acc.uses.dword(d); }
                }
            }
            Operand::Hwreg(_) => {
                if name.starts_with("s_setreg") { acc.writes.dword(OTHER0 + 6); } else { acc.reads.dword(OTHER0 + 6); acc.uses.dword(OTHER0 + 6); }
            }
            _ => {}
        }
    }
    let (mut reads, mut writes) = row_implicit(arch, inst.op, inst.form);
    if let FormFields::Vopd { y_op, .. } = inst.fields {
        let (r, w) = row_implicit(arch, y_op, Form::Vopd);
        reads |= r;
        writes |= w;
    }
    if is_vector_form(inst.form) { reads |= ImplicitSet::EXEC; }
    for d in implicit_dwords(reads, wave) { acc.reads.dword(d); acc.uses.dword(d); }
    for d in implicit_dwords(writes, wave) { acc.writes.dword(d); }
    Ok(acc)
}
/// Only the old value preserved by a lane write may remain open: the
/// dword-granular state marks the whole VGPR defined after `v_writelane`,
/// so its undefined preserved lanes are recorded at the write. A half write
/// contributes nothing: `access` does not read the preserved half, whose own
/// definedness is checked at its readers. Explicit
/// undefined sources, even when they overlap the destination, still fail
/// the checked edit.
fn partial_undefined(arch: Arch, inst: &Inst, missing: &Bits) -> Bits {
    let Some(reg) = preserved_vgpr_destination(arch, inst) else { return Bits::default(); };
    missing.inter(&Bits::of_reg(reg))
}


/// HSA ABI entry table (core.md §6.2): the only locations defined at entry.
fn entry_seed(kern: &Kernel, arch: Arch) -> Result<Bits, EditError> {
    if !matches!(arch, Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1201) { return Err(EditError::Unsupported(format!("no ABI entry table for {arch:?}"))); }
    let mut seed = Bits::default();
    let (user, wg_x, wg_yz) = match &kern.abi {
        // gfx11 (RDNA3 ISA §3.3 / LLVM AMDHSA): the system SGPRs follow the
        // COMPUTE_PGM_RSRC2.USER_SGPR_COUNT user SGPRs in order TGID_X
        // (bit 7), TGID_Y (bit 8), TGID_Z (bit 9), TG_SIZE (bit 10); there
        // are no workgroup-id TTMPs.
        Abi::Hsa { descriptor, .. } if arch != Arch::Gfx1201 => {
            let rsrc2 = descriptor.compute_pgm_rsrc2.0;
            let user = (rsrc2 >> 1 & 0x1f) as usize;
            let system = (7..=10).filter(|bit| rsrc2 >> bit & 1 != 0).count();
            for s in 0..(user + system).min(106) { seed.dword(S0 + s); }
            (0, false, false)
        }
        Abi::Hsa { descriptor, .. } => {
            let props = descriptor.kernel_code_properties.0;
            let sizes = [4usize, 2, 2, 2, 2, 2, 1];
            let user: usize = sizes.iter().enumerate().filter(|(bit, _)| props >> bit & 1 != 0).map(|(_, n)| n).sum();
            let rsrc2 = descriptor.compute_pgm_rsrc2.0;
            (user, rsrc2 >> 7 & 1 != 0, rsrc2 >> 8 & 3 != 0)
        }
        Abi::Raw { user_sgprs, .. } => {
            let user = user_sgprs.iter().map(|role| match role {
                UserSgprRole::PrivateSegmentBuffer => 4, UserSgprRole::PrivateSegmentSize => 1, _ => 2,
            }).sum();
            (user, true, true)
        }
    };
    for s in 0..user.min(106) { seed.dword(S0 + s); }
    if wg_x { seed.dword(T0 + 9); }
    if wg_yz { seed.dword(T0 + 7); }
    seed.dword(0);
    seed.dword(EXEC_LO);
    if kern.wave == Wave::Wave64 { seed.dword(EXEC_HI); }
    seed.dword(MODE);
    for k in 0..7 { seed.dword(OTHER0 + k); }
    Ok(seed)
}

// ---------------------------------------------------------------------------
// Instruction-level flow graph and dataflow
// ---------------------------------------------------------------------------

struct Flow {
    n: usize,
    ids: Vec<InstId>,
    pos: HashMap<InstId, usize>,
    block_of: Vec<usize>,
    succ: Vec<SmallVec<[usize; 2]>>,
    pred: Vec<SmallVec<[usize; 2]>>,
    acc: Vec<Access>,
    predicates: Vec<HashMap<InstId, bool>>,
}

fn label_of(inst: &Inst) -> Option<BlockId> {
    inst.operands.iter().find_map(|op| match op { Operand::Label(b) => Some(*b), _ => None })
}

impl Flow {
    fn new(body: &Body, arch: Arch, wave: Wave) -> Result<Self, EditError> {
        let n = body.layout.len();
        let mut block_of = vec![0; n];
        for block in &body.blocks {
            for p in block.range.0..block.range.1.min(n) { block_of[p] = block.id.0; }
        }
        let mut succ = vec![SmallVec::new(); n];
        let mut pred = vec![SmallVec::new(); n];
        let mut acc = Vec::with_capacity(n);
        let mut pos = HashMap::with_capacity(n);
        for (p, &id) in body.layout.iter().enumerate() {
            let inst = body.insts.get(id).ok_or(EditError::UnknownInst(id))?;
            pos.insert(id, p);
            acc.push(access(arch, wave, inst)?);
            let target = |b: BlockId| body.blocks.get(b.0).map(|blk| blk.range.0).filter(|&t| t < n)
                .ok_or_else(|| EditError::Label(format!("{id:?} targets missing block {b:?}")));
            let out: &mut SmallVec<[usize; 2]> = &mut succ[p];
            match inst.effects.control {
                Control::Jump => out.push(target(label_of(inst).ok_or_else(|| EditError::Label(format!("{id:?} has no label")))?)?),
                Control::Branch { .. } => {
                    if p + 1 < n { out.push(p + 1); }
                    let t = target(label_of(inst).ok_or_else(|| EditError::Label(format!("{id:?} has no label")))?)?;
                    if !out.contains(&t) { out.push(t); }
                }
                Control::EndPgm | Control::Halt | Control::Trap => {}
                _ => if p + 1 < n { out.push(p + 1); },
            }
        }
        for p in 0..n {
            for &s in succ[p].clone().iter() { pred[s].push(p); }
        }
        Ok(Self { n, ids: body.layout.clone(), pos, block_of, succ, pred, acc,
            predicates: passes::predicates::partitions(body, arch) })
    }

    /// Live-in per position. A VGPR write kills only lanes-alike reads: a
    /// value read after an EXEC write may come from lanes a write before it
    /// did not cover, so crossing an EXEC write (backward) moves VGPR
    /// liveness into a set that VGPR writes no longer kill. Scalar writes
    /// always kill. With `only`, positions it rejects are transparent.
    fn live(&self, only: Option<&dyn Fn(usize) -> bool>) -> Vec<Bits> {
        let mut vgprs = Bits::default();
        for d in 0..256 { vgprs.dword(d); }
        let mut exec = Bits::default();
        exec.dword(EXEC_LO);
        exec.dword(EXEC_HI);
        let mut state = vec![(Bits::default(), Bits::default()); self.n];
        let mut work: Vec<usize> = (0..self.n).collect();
        let mut queued = vec![true; self.n];
        while let Some(p) = work.pop() {
            queued[p] = false;
            let (mut same, mut other) = (Bits::default(), Bits::default());
            for &s in &self.succ[p] { same.or(&state[s].0); other.or(&state[s].1); }
            if only.is_none_or(|f| f(p)) {
                let acc = &self.acc[p];
                if !acc.writes.inter(&exec).is_empty() {
                    other.or(&same.inter(&vgprs));
                    same = same.minus(&vgprs);
                }
                same = same.minus(&acc.writes).union(&acc.reads);
                other = other.minus(&acc.writes.minus(&vgprs));
            }
            if (same, other) != state[p] {
                state[p] = (same, other);
                for &q in &self.pred[p] { if !queued[q] { queued[q] = true; work.push(q); } }
            }
        }
        state.into_iter().map(|(same, other)| same.union(&other)).collect()
    }

    fn live_out(&self, live_in: &[Bits], p: usize) -> Bits {
        let mut out = Bits::default();
        for &s in &self.succ[p] { out.or(&live_in[s]); }
        out
    }

    /// Must-defined (in, out) per position from the entry seed.
    fn defined(&self, seed: Bits) -> (Vec<Bits>, Vec<Bits>) {
        let coarse = self.defined_on(seed, &self.succ, &self.pred);
        // Must-defined without guard correlation is conservative. Refine only
        // when a consumer lacks a definition, not merely because guards exist.
        if (self.predicates.len() == 1 && self.predicates[0].is_empty())
            || self.acc.iter().zip(&coarse.0).all(|(acc, inn)| acc.reads.minus(inn).is_empty()) {
            return coarse;
        }
        let mut joined_in = vec![Bits::FULL; self.n];
        let mut joined_out = vec![Bits::FULL; self.n];
        for choices in &self.predicates {
            let mut succ = self.succ.clone();
            for (p, id) in self.ids.iter().enumerate() {
                if let Some(&taken) = choices.get(id) {
                    if succ[p].len() > 1 { succ[p].retain(|q| (*q != p + 1) == taken); }
                }
            }
            let reachable = self.reach(&[0], &|_| false, &|p| succ[p].clone(), self.n);
            let mut pred = vec![SmallVec::<[usize; 2]>::new(); self.n];
            for p in 0..self.n {
                if reachable[p] { for &q in &succ[p] { pred[q].push(p); } }
            }
            let (inn, out) = self.defined_on(seed, &succ, &pred);
            for p in 0..self.n {
                if reachable[p] {
                    joined_in[p] = joined_in[p].inter(&inn[p]);
                    joined_out[p] = joined_out[p].inter(&out[p]);
                }
            }
        }
        (joined_in, joined_out)
    }

    fn defined_on(&self, seed: Bits, succ: &[SmallVec<[usize; 2]>], pred: &[SmallVec<[usize; 2]>]) -> (Vec<Bits>, Vec<Bits>) {
        let mut def_in = vec![Bits::FULL; self.n];
        let mut def_out = vec![Bits::FULL; self.n];
        let mut work: Vec<usize> = (0..self.n).rev().collect();
        let mut queued = vec![true; self.n];
        while let Some(p) = work.pop() {
            queued[p] = false;
            let mut inn = if p == 0 { seed } else { Bits::FULL };
            for &q in &pred[p] { inn = inn.inter(&def_out[q]); }
            let out = inn.union(&self.acc[p].writes);
            def_in[p] = inn;
            if out != def_out[p] {
                def_out[p] = out;
                for &s in &succ[p] { if !queued[s] { queued[s] = true; work.push(s); } }
            }
        }
        (def_in, def_out)
    }

    /// May-analysis: true before `p` when some path from entry executed a
    /// `s_sendmsg(MSG_DEALLOC_VGPRS)`.
    fn after_dealloc(&self, body: &Body) -> Vec<bool> {
        let dealloc: Vec<bool> = self.ids.iter().map(|id| body.insts.get(*id).is_some_and(is_dealloc)).collect();
        let mut before = vec![false; self.n];
        let mut work: Vec<usize> = (0..self.n).filter(|&p| dealloc[p]).collect();
        while let Some(p) = work.pop() {
            for &s in &self.succ[p] { if !before[s] { before[s] = true; work.push(s); } }
        }
        before
    }

    fn reach(&self, starts: &[usize], avoid: &dyn Fn(usize) -> bool, next: &dyn Fn(usize) -> SmallVec<[usize; 2]>, size: usize) -> Vec<bool> {
        let mut seen = vec![false; size];
        let mut work: Vec<usize> = Vec::new();
        for &s in starts { if !avoid(s) && !seen[s] { seen[s] = true; work.push(s); } }
        while let Some(p) = work.pop() {
            for q in next(p) { if !avoid(q) && !seen[q] { seen[q] = true; work.push(q); } }
        }
        seen
    }
}

fn is_dealloc(inst: &Inst) -> bool {
    inst.operands.iter().any(|op| matches!(op, Operand::SendMsg(Msg { id: 3, .. })))
}

fn is_edit_inserted(inst: &Inst) -> bool { inst.prov.edit.is_some() }

// ---------------------------------------------------------------------------
// Windows (clause members, delay targets) as forbidden gap intervals
// ---------------------------------------------------------------------------

/// `(opener_pos, last_pos, opener)`: gaps `opener_pos+1 ..= last_pos` are inside.
fn windows(body: &Body) -> Vec<(usize, usize, InstId, bool)> {
    let n = body.layout.len();
    let mut out = Vec::new();
    for (p, &id) in body.layout.iter().enumerate() {
        let Some(inst) = body.insts.get(id) else { continue };
        match inst.effects.control {
            Control::Clause => if let Some(len) = inst.mods.clause {
                out.push((p, (p + usize::from(len) + 1).min(n - 1), id, true));
            },
            Control::Delay => if let Some(hint) = inst.mods.delay {
                let last = if hint.instid1 != 0 { p + 1 + usize::from(hint.instskip) } else { p + 1 };
                out.push((p, last.min(n - 1), id, false));
            },
            _ => {}
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
struct Gap { block: usize, pos: usize, at_end: bool }

fn resolve_cursor(body: &Body, cursor: &Cursor, checked: bool) -> Result<Gap, EditError> {
    let block = body.blocks.get(cursor.block.0).ok_or(EditError::UnknownBlock(cursor.block))?;
    let (start, end) = block.range;
    let mut pos = match cursor.before {
        Some(id) => (start..end).find(|&p| body.layout[p] == id).ok_or(EditError::CursorNotInBlock { block: cursor.block, inst: id })?,
        None => end,
    };
    if checked {
        let wins = windows(body);
        while let Some(&(lo, hi, opener, _)) = wins.iter().find(|(lo, hi, ..)| *lo < pos && pos <= *hi) {
            pos = match cursor.window_policy {
                WindowPolicy::Reject => return Err(EditError::CursorInWindow { pos, opener }),
                WindowPolicy::MoveUp => lo,
                WindowPolicy::MoveDown => hi + 1,
            };
        }
    }
    let pos = pos.clamp(start, end);
    if pos == end {
        let last = body.insts.get(body.layout[end - 1]).ok_or(EditError::UnknownInst(body.layout[end - 1]))?;
        if !matches!(last.effects.control, Control::None | Control::Barrier(_) | Control::Wait | Control::Clause | Control::Delay) {
            return Err(EditError::CursorAfterTerminator(cursor.block));
        }
    }
    Ok(Gap { block: cursor.block.0, pos, at_end: pos == end })
}

/// Positions `lo..=hi` of a range, which must lie inside one block.
fn resolve_range(body: &Body, range: InstRange) -> Result<(usize, usize, usize), EditError> {
    let find = |id: InstId| body.layout.iter().position(|x| *x == id).ok_or(EditError::UnknownInst(id));
    let (lo, hi) = (find(range.first)?, find(range.last)?);
    let block = body.blocks.iter().find(|b| b.range.0 <= lo && lo < b.range.1).ok_or(EditError::BadRange(range))?;
    if hi < lo || hi >= block.range.1 { return Err(EditError::BadRange(range)); }
    if lo == block.range.0 && hi + 1 == block.range.1 { return Err(EditError::RangeEmptiesBlock(range)); }
    Ok((block.id.0, lo, hi))
}

/// A range must contain each window it touches, opener included; a removed
/// `s_delay_alu` may leave its targets behind (hints are performance-only).
fn check_window_integrity(body: &Body, lo: usize, hi: usize, allow_orphan_targets: bool) -> Result<(), EditError> {
    for (w_lo, w_hi, opener, clause) in windows(body) {
        let has_opener = (lo..=hi).contains(&w_lo);
        let cuts_members = lo <= w_hi && hi > w_lo;
        if has_opener {
            if w_hi > hi && (clause || !allow_orphan_targets) { return Err(EditError::WindowBroken { opener }); }
        } else if cuts_members && lo > w_lo {
            return Err(EditError::WindowBroken { opener });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Block-structured splicing and CFG reinstallation
// ---------------------------------------------------------------------------

fn seqs_of(body: &Body) -> Vec<Vec<InstId>> {
    body.blocks.iter().map(|b| body.layout[b.range.0..b.range.1].to_vec()).collect()
}

/// Rebuild `layout` and `blocks` from per-block sequences, with C4's
/// conventions (true basic blocks; unreachable blocks keep no edges).
fn install(body: &mut Body, seqs: Vec<Vec<InstId>>) -> Result<(), EditError> {
    let count = seqs.len();
    let mut layout = Vec::with_capacity(seqs.iter().map(Vec::len).sum());
    let mut ranges = Vec::with_capacity(count);
    for seq in &seqs {
        if seq.is_empty() { return Err(EditError::Split("a block would become empty".into())); }
        ranges.push((layout.len(), layout.len() + seq.len()));
        layout.extend_from_slice(seq);
    }
    enum Term { Branch(usize, crate::cfg::Cond), Jump(usize), End, Fall }
    let mut terms = Vec::with_capacity(count);
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (b, seq) in seqs.iter().enumerate() {
        for (i, &id) in seq.iter().enumerate() {
            let inst = body.insts.get(id).ok_or(EditError::UnknownInst(id))?;
            if i + 1 < seq.len() && !matches!(inst.effects.control, Control::None | Control::Barrier(_) | Control::Wait | Control::Clause | Control::Delay) {
                return Err(EditError::Label(format!("control instruction {id:?} would sit inside block {b}")));
            }
        }
        let last_id = *seq.last().expect("non-empty");
        let last = body.insts.get(last_id).ok_or(EditError::UnknownInst(last_id))?;
        let label = || -> Result<usize, EditError> {
            let t = label_of(last).ok_or_else(|| EditError::Label(format!("{last_id:?} has no label")))?;
            if t.0 >= count { return Err(EditError::Label(format!("{last_id:?} targets missing block {t:?}"))); }
            Ok(t.0)
        };
        let term = match last.effects.control {
            Control::Jump => { let t = label()?; succs[b].push(t); Term::Jump(t) }
            Control::Branch { cond } => {
                let t = label()?;
                if b + 1 >= count { return Err(EditError::Label("conditional branch in the last block".into())); }
                succs[b].extend([t, b + 1]);
                Term::Branch(t, cond)
            }
            Control::EndPgm | Control::Halt | Control::Trap => Term::End,
            _ => { if b + 1 < count { succs[b].push(b + 1); } Term::Fall }
        };
        succs[b].sort_unstable();
        succs[b].dedup();
        terms.push(term);
    }
    let mut reachable = vec![false; count];
    let mut work = vec![0usize];
    reachable[0] = true;
    while let Some(b) = work.pop() {
        for &s in &succs[b] { if !reachable[s] { reachable[s] = true; work.push(s); } }
    }
    let mut blocks: Vec<Block> = (0..count).map(|b| Block {
        id: BlockId(b), range: ranges[b],
        term: if !reachable[b] { Terminator::Unreachable } else {
            match terms[b] {
                Term::Branch(t, cond) => Terminator::Branch { cond, taken: BlockId(t), fallthrough: BlockId(b + 1) },
                Term::Jump(t) => Terminator::Jump(BlockId(t)),
                Term::End => Terminator::EndPgm,
                Term::Fall => Terminator::FallThrough,
            }
        },
        preds: SmallVec::new(), succs: SmallVec::new(),
    }).collect();
    for b in 0..count {
        if !reachable[b] { continue; }
        for &s in &succs[b] {
            if reachable[s] { blocks[b].succs.push(BlockId(s)); blocks[s].preds.push(BlockId(b)); }
        }
    }
    body.layout = layout;
    body.blocks = blocks;
    Ok(())
}

/// Rewrite every label `> at` by `delta` (block renumbering after split/merge).
fn relabel(body: &mut Body, at: usize, delta: isize) {
    let ids: Vec<InstId> = body.insts.iter().map(|(id, _)| id).collect();
    for id in ids {
        let inst = body.insts.get_mut(id).expect("live");
        for op in inst.operands.iter_mut() {
            if let Operand::Label(b) = op {
                if b.0 > at { *b = BlockId((b.0 as isize + delta) as usize); }
            }
        }
    }
}

/// Re-occupy tombstoned slots under their original ids.
fn revive(arena: &mut Arena<Inst>, items: &[(InstId, Inst)]) -> Result<(), EditError> {
    let wanted: HashMap<usize, &Inst> = items.iter().map(|(id, inst)| (id.0, inst)).collect();
    for (id, _) in items {
        if id.0 >= arena.len() || arena.get(*id).is_some() {
            return Err(EditError::Merge(format!("{id:?} is not a tombstone of this kernel")));
        }
    }
    let Some(filler) = items.first().map(|(_, inst)| inst.clone()) else { return Ok(()) };
    let old = std::mem::take(arena);
    let mut dead = Vec::new();
    for slot in 0..old.len() {
        let id = InstId(slot);
        match (old.get(id), wanted.get(&slot)) {
            (Some(inst), _) => { arena.insert(inst.clone()); }
            (None, Some(inst)) => { arena.insert((*inst).clone()); }
            (None, None) => { arena.insert(filler.clone()); dead.push(id); }
        }
    }
    for id in dead { arena.remove(id); }
    Ok(())
}

// ---------------------------------------------------------------------------
// s_delay_alu maintenance
// ---------------------------------------------------------------------------

struct HintSlot { hint: InstId, slot: u8, n: u8, producers: BTreeSet<Option<InstId>> }

/// Producers `n` non-TRANS VALUs back (`VALU_DEP_n`) from `consumer` on every path (`None` = a path
/// reaches the entry first).
fn producers(flow: &Flow, body: &Body, consumer: usize, n: u8) -> BTreeSet<Option<InstId>> {
    let mut out = BTreeSet::new();
    let mut seen = HashSet::new();
    let mut work: Vec<(usize, u8)> = flow.pred[consumer].iter().map(|&p| (p, 0)).collect();
    if work.is_empty() { out.insert(None); }
    while let Some((p, count)) = work.pop() {
        if !seen.insert((p, count)) { continue; }
        let valu = body.insts.get(flow.ids[p]).is_some_and(passes::windows::counts_for_valu_dep);
        let count = count + u8::from(valu);
        if valu && count == n { out.insert(Some(flow.ids[p])); continue; }
        if flow.pred[p].is_empty() { out.insert(None); }
        for &q in &flow.pred[p] { work.push((q, count)); }
    }
    out
}

fn hint_consumer(flow: &Flow, hint_pos: usize, hint: DelayAluHint, slot: u8) -> Option<usize> {
    let c = if slot == 0 { hint_pos + 1 } else { hint_pos + 1 + usize::from(hint.instskip) };
    (c < flow.n).then_some(c)
}

fn hint_snapshot(flow: &Flow, body: &Body) -> Vec<HintSlot> {
    let mut out = Vec::new();
    for (p, &id) in flow.ids.iter().enumerate() {
        let Some(hint) = body.insts.get(id).and_then(|i| i.mods.delay.filter(|_| i.effects.control == Control::Delay)) else { continue };
        for (slot, n) in [(0u8, hint.instid0), (1u8, hint.instid1)] {
            if !(1..=4).contains(&n) { continue; }
            if let Some(c) = hint_consumer(flow, p, hint, slot) {
                out.push(HintSlot { hint: id, slot, n, producers: producers(flow, body, c, n) });
            }
        }
    }
    out
}

fn set_delay(inst: &mut Inst, hint: DelayAluHint) {
    let raw = u16::from(hint.instid0) | u16::from(hint.instskip) << 4 | u16::from(hint.instid1) << 7;
    for op in inst.operands.iter_mut() {
        if let Operand::Imm(ImmField::Sopp(v)) = op { *v = raw as i16; }
    }
    inst.mods.delay = Some(hint);
}

// ---------------------------------------------------------------------------
// The transaction
// ---------------------------------------------------------------------------

struct Tx {
    program: Program,
    k: usize,
    arch: Arch,
    id: EditId,
    removed: Vec<InstId>,
    touched: Bits,
    claims: Vec<RegClaim>,
    delay_rewrites: Vec<DelayRewrite>,
    commutations: Vec<CommutationProof>,
}

fn changes_paths(edit: &Edit) -> bool {
    matches!(edit, Edit::Insert { .. } | Edit::Remove { .. } | Edit::Replace { .. } | Edit::Move { .. } | Edit::Restore { .. } | Edit::Retarget { .. })
}

impl Tx {
    fn new(program: Program, k: usize, id: EditId) -> Self {
        let arch = program.target.arch;
        Self { program, k, arch, id, removed: Vec::new(), touched: Bits::default(), claims: Vec::new(), delay_rewrites: Vec::new(), commutations: Vec::new() }
    }
    fn kernel(&self) -> &Kernel { &self.program.kernels[self.k] }
    fn body(&self) -> &Body { &self.program.kernels[self.k].body }
    fn body_mut(&mut self) -> &mut Body { &mut self.program.kernels[self.k].body }
    fn wave(&self) -> Wave { self.kernel().wave }
    fn flow(&self) -> Result<Flow, EditError> { Flow::new(self.body(), self.arch, self.wave()) }

    fn step(&mut self, edit: &Edit, checked: bool) -> Result<Edit, EditError> {
        match edit {
            Edit::Batch(steps) => {
                let mut inverses = Vec::with_capacity(steps.len());
                for step in steps { inverses.push(self.step(step, checked)?); }
                inverses.reverse();
                Ok(Edit::Batch(inverses))
            }
            Edit::Revert { from, to, undo } => {
                let found = digest(&self.program, self.k);
                if found != *from { return Err(EditError::RevertStale { expected: *from, found }); }
                let redo = self.step(undo, false)?;
                let found = digest(&self.program, self.k);
                if found != *to { return Err(EditError::RevertDiverged { expected: *to, found }); }
                Ok(Edit::Revert { from: *to, to: *from, undo: Box::new(redo) })
            }
            other => {
                let snapshot = if checked && changes_paths(other) { let flow = self.flow()?; Some(hint_snapshot(&flow, self.body())) } else { None };
                let inverse = self.step_one(other, checked)?;
                let Some(snapshot) = snapshot else { return Ok(inverse) };
                let restores = self.maintain_hints(snapshot)?;
                if restores.is_empty() { return Ok(inverse); }
                let mut steps = vec![inverse];
                steps.extend(restores);
                Ok(Edit::Batch(steps))
            }
        }
    }

    fn step_one(&mut self, edit: &Edit, checked: bool) -> Result<Edit, EditError> {
        match edit {
            Edit::Insert { at, insts, claims } => self.insert(at, insts, claims, checked),
            Edit::Restore { at, insts } => self.restore(at, insts, checked),
            Edit::Remove { range } => self.remove(*range, checked),
            Edit::Replace { range, insts } => self.replace(*range, insts, checked),
            Edit::Move { range, to } => self.move_run(*range, to, checked),
            Edit::ReRegister { scope, map, claims } => self.reregister(scope, map, claims, checked),
            Edit::SplitBlock { at } => self.split(at, checked),
            Edit::MergeBlock { block } => self.merge(*block),
            Edit::Retarget { branch, to } => self.retarget(*branch, *to, checked),
            Edit::SetWait { inst, imm } => self.set_wait(*inst, imm, checked),
            Edit::SetDelayHint { inst, hint } => self.set_delay_hint(*inst, *hint, checked),
            Edit::Descriptor { kernel, change } => self.descriptor(kernel, change, checked),
            Edit::Metadata { kernel, change } => self.metadata(kernel, change, checked),
            Edit::Batch(_) | Edit::Revert { .. } => self.step(edit, checked),
        }
    }

    // ---- shared pieces ----

    /// Re-derive table effects and stamp edit provenance on a new instruction.
    fn stamp(&self, inst: &Inst, index: usize) -> Result<Inst, EditError> {
        let bad = |reason: String| EditError::InvalidInst { index, reason };
        let derived = Inst::from_parts(self.arch, inst.op, inst.form, inst.fields.clone(), inst.operands.clone(), inst.mods.clone(), inst.literal, inst.prov.clone())
            .map_err(|e| bad(e.to_string()))?;
        if derived.effects != inst.effects { return Err(bad("effects are not the table-derived effects".into())); }
        let mut probe = derived.clone();
        for op in probe.operands.iter_mut() { if matches!(op, Operand::Label(_)) { *op = Operand::Imm(ImmField::Sopp(0)); } }
        crate::codec::gfx12::encode_for(self.arch, &probe).map_err(|e| bad(format!("not encodable: {e}")))?;
        let mut out = derived;
        out.prov = Provenance { source: Source::Edit(self.id), pc: None, bytes: None, line: inst.prov.line, edit: Some(self.id) };
        Ok(out)
    }

    fn touch(&mut self, flow_acc: &Access) { self.touched.or(&flow_acc.reads.union(&flow_acc.writes)); }

    /// Splice `items` into block `gap.block` at `gap.pos`; `Some(id)` revives a tombstone.
    fn splice_in(&mut self, gap: Gap, items: Vec<(Option<InstId>, Inst)>) -> Result<Vec<InstId>, EditError> {
        let (arch, wave) = (self.arch, self.wave());
        let mut ids = Vec::with_capacity(items.len());
        let revived: Vec<(InstId, Inst)> = items.iter().filter_map(|(id, inst)| id.map(|id| (id, inst.clone()))).collect();
        revive(&mut self.body_mut().insts, &revived)?;
        for (id, inst) in items {
            let acc = access(arch, wave, &inst)?;
            self.touch(&acc);
            ids.push(match id { Some(id) => id, None => self.body_mut().insts.insert(inst) });
        }
        let start = self.body().blocks[gap.block].range.0;
        let mut seqs = seqs_of(self.body());
        let at = gap.pos - start;
        seqs[gap.block].splice(at..at, ids.iter().copied());
        install(self.body_mut(), seqs)?;
        Ok(ids)
    }

    fn splice_out(&mut self, block: usize, lo: usize, hi: usize) -> Result<Vec<(InstId, Inst)>, EditError> {
        let (arch, wave) = (self.arch, self.wave());
        let start = self.body().blocks[block].range.0;
        let mut seqs = seqs_of(self.body());
        let ids: Vec<InstId> = seqs[block].drain(lo - start..=hi - start).collect();
        install(self.body_mut(), seqs)?;
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let inst = self.body_mut().insts.remove(id).ok_or(EditError::UnknownInst(id))?;
            let acc = access(arch, wave, &inst)?;
            self.touch(&acc);
            self.removed.push(id);
            out.push((id, inst));
        }
        Ok(out)
    }

    /// Cursor at the gap a range leaves behind (before its successor in the block, or the block end).
    fn cursor_after(&self, block: usize, hi: usize) -> Cursor {
        let end = self.body().blocks[block].range.1;
        let before = (hi + 1 < end).then(|| self.body().layout[hi + 1]);
        Cursor { block: BlockId(block), before, window_policy: WindowPolicy::Reject }
    }

    fn check_defined(&self, ids: &[InstId]) -> Result<(), EditError> {
        let flow = self.flow()?;
        let (def_in, _) = flow.defined(entry_seed(self.kernel(), self.arch)?);
        for id in ids {
            let p = flow.pos[id];
            let inst = self.body().insts.get(*id).ok_or(EditError::UnknownInst(*id))?;
            let missing = flow.acc[p].reads.minus(&def_in[p]);
            let partial = partial_undefined(self.arch, inst, &missing);
            let missing = missing.minus(&partial);
            if !missing.is_empty() { return Err(EditError::Undefined { inst: *id, regs: bits_names(&missing) }); }
        }
        Ok(())
    }

    /// Net locations a straight-line sequence changes: scalar `s_mov_b32`
    /// copies are tracked so a saved-and-restored register (EXEC around a
    /// store) is not a write.
    fn net_writes(&self, insts: &[&Inst]) -> Result<Bits, EditError> {
        let mut value: HashMap<usize, Option<usize>> = HashMap::new();
        let mut written = Bits::default();
        for inst in insts {
            let acc = access(self.arch, self.wave(), inst)?;
            written.or(&acc.writes);
            if let Some((dst, src)) = scalar_copy(self.arch, inst) {
                let v = value.get(&src).copied().unwrap_or(Some(src));
                value.insert(dst, v);
                continue;
            }
            for d in acc.writes.dwords() { value.insert(d, None); }
        }
        let mut net = written;
        for (d, v) in value {
            if v == Some(d) { net = net.minus(&{ let mut b = Bits::default(); b.dword(d); b }); }
        }
        Ok(net)
    }

    fn check_writes(&self, net: &Bits, live: &Bits, gap: Gap, claims: &[RegClaim], licensed: &Bits) -> Result<(), EditError> {
        let conflict = net.inter(live).minus(licensed);
        for d in conflict.dwords() {
            let reg = dword_reg(d).ok_or_else(|| EditError::WriteLive { reg: dword_name(d) })?;
            let covered = claims.iter().any(|c| c.reg.overlaps(reg) && reg_dwords(c.reg).contains(&d) && self.scope_covers(&c.scope, gap));
            if !covered { return Err(EditError::WriteLive { reg: dword_name(d) }); }
        }
        Ok(())
    }

    fn scope_positions(&self, scope: &Scope) -> Result<Vec<usize>, EditError> {
        let body = self.body();
        Ok(match scope {
            Scope::Whole => (0..body.layout.len()).collect(),
            Scope::Blocks(blocks) => {
                let mut out = Vec::new();
                for b in blocks {
                    let block = body.blocks.get(b.0).ok_or(EditError::UnknownBlock(*b))?;
                    out.extend(block.range.0..block.range.1);
                }
                out
            }
            Scope::Between(a, b) => {
                let find = |id: &InstId| body.layout.iter().position(|x| x == id).ok_or(EditError::UnknownInst(*id));
                let (pa, pb) = (find(a)?, find(b)?);
                if pa > pb { return Err(EditError::ClaimUnsound { claim: format!("{scope:?}"), reason: "scope ends before it starts".into() }); }
                (pa..=pb).collect()
            }
        })
    }

    fn scope_covers(&self, scope: &Scope, gap: Gap) -> bool {
        match scope {
            Scope::Whole => true,
            Scope::Blocks(blocks) => blocks.iter().any(|b| b.0 == gap.block),
            Scope::Between(..) => self.scope_positions(scope).is_ok_and(|ps| {
                ps.first().is_some_and(|&a| a <= gap.pos) && ps.last().is_some_and(|&b| gap.pos <= b + 1)
            }),
        }
    }

    /// A claim is sound iff no original instruction in scope names the
    /// register and no original value of it is live anywhere in scope.
    fn check_claims(&self, flow: &Flow, claims: &[RegClaim]) -> Result<(), EditError> {
        if claims.is_empty() { return Ok(()); }
        let body = self.body();
        let original = |p: usize| body.insts.get(flow.ids[p]).is_some_and(|i| !is_edit_inserted(i));
        let live = flow.live(Some(&original));
        for claim in claims {
            let unsound = |reason: String| EditError::ClaimUnsound { claim: format!("{} {}", claim.name, claim.reg), reason };
            if claim.reg.kind == Kind::Ttmp { return Err(unsound("trap temporaries are entry facts, not claimable".into())); }
            claim.reg.validate().map_err(unsound)?;
            let bits = Bits::of_reg(claim.reg);
            for p in self.scope_positions(&claim.scope)? {
                let acc = &flow.acc[p];
                if original(p) && !acc.uses.union(&acc.writes).inter(&bits).is_empty() {
                    return Err(unsound(format!("original instruction {:?} names it", flow.ids[p])));
                }
                if !live[p].inter(&bits).is_empty() || !flow.live_out(&live, p).inter(&bits).is_empty() {
                    return Err(unsound(format!("an original value is live at {:?}", flow.ids[p])));
                }
            }
        }
        Ok(())
    }

    fn check_no_dealloc(&self, flow: &Flow, gap: Gap) -> Result<(), EditError> {
        let after = flow.after_dealloc(self.body());
        let hit = if gap.at_end { after[gap.pos - 1] || self.body().insts.get(flow.ids[gap.pos - 1]).is_some_and(is_dealloc) } else { after[gap.pos] };
        if hit { return Err(EditError::AfterDealloc { pos: gap.pos }); }
        Ok(())
    }

    /// Only a branch or jump to the same next block it would fall through to
    /// may be inserted (semantically neutral); `Retarget` then gives it meaning.
    fn check_inserted_control(&self, insts: &[Inst], gap: Gap) -> Result<(), EditError> {
        for (index, inst) in insts.iter().enumerate() {
            let bad = |reason: &str| EditError::ControlInsert { index, reason: reason.into() };
            match inst.effects.control {
                Control::EndPgm | Control::Halt | Control::Trap =>
                    return Err(bad("terminators are never inserted")),
                Control::Branch { .. } | Control::Jump => {
                    let neutral = index + 1 == insts.len() && gap.at_end && label_of(inst) == Some(BlockId(gap.block + 1));
                    if !neutral { return Err(bad("a branch or jump may only be inserted last, at a block end, targeting its own fall-through (then Retarget)")); }
                }
                Control::Clause | Control::Delay => {
                    let reach = if inst.effects.control == Control::Clause {
                        inst.mods.clause.map(|l| usize::from(l) + 1)
                    } else {
                        inst.mods.delay.map(|h| if h.instid1 != 0 { 1 + usize::from(h.instskip) } else { 1 })
                    };
                    if reach.is_none_or(|r| index + r >= insts.len()) { return Err(bad("an inserted window must end inside the inserted sequence")); }
                }
                _ => {}
            }
        }
        Ok(())
    }

    // ---- Insert / Restore ----

    fn insert(&mut self, at: &Cursor, insts: &[Inst], claims: &[RegClaim], checked: bool) -> Result<Edit, EditError> {
        if insts.is_empty() { return Err(EditError::InvalidInst { index: 0, reason: "empty insertion".into() }); }
        let stamped = insts.iter().enumerate().map(|(i, inst)| self.stamp(inst, i)).collect::<Result<Vec<_>, _>>()?;
        let gap = resolve_cursor(self.body(), at, checked)?;
        if checked { self.check_insert(gap, &stamped, claims)?; }
        let ids = self.splice_in(gap, stamped.into_iter().map(|i| (None, i)).collect())?;
        if checked { self.check_defined(&ids)?; }
        self.claims.extend(claims.iter().cloned());
        Ok(Edit::Remove { range: InstRange { first: ids[0], last: *ids.last().expect("non-empty") } })
    }

    fn check_insert(&self, gap: Gap, insts: &[Inst], claims: &[RegClaim]) -> Result<(), EditError> {
        let flow = self.flow()?;
        self.check_no_dealloc(&flow, gap)?;
        self.check_inserted_control(insts, gap)?;
        let live = flow.live(None);
        let live_gap = if gap.pos < flow.n { live[gap.pos] } else { Bits::default() };
        let refs: Vec<&Inst> = insts.iter().collect();
        let net = self.net_writes(&refs)?;
        self.check_claims(&flow, claims)?;
        self.check_writes(&net, &live_gap, gap, claims, &Bits::default())
    }

    fn restore(&mut self, at: &Cursor, insts: &[(InstId, Inst)], checked: bool) -> Result<Edit, EditError> {
        if insts.is_empty() { return Err(EditError::InvalidInst { index: 0, reason: "empty restore".into() }); }
        let gap = resolve_cursor(self.body(), at, checked)?;
        if checked {
            let plain: Vec<Inst> = insts.iter().map(|(_, i)| i.clone()).collect();
            for (index, inst) in plain.iter().enumerate() {
                Inst::from_parts(self.arch, inst.op, inst.form, inst.fields.clone(), inst.operands.clone(), inst.mods.clone(), inst.literal, inst.prov.clone())
                    .map_err(|e| EditError::InvalidInst { index, reason: e.to_string() })?;
            }
            self.check_insert(gap, &plain, &[])?;
        }
        let ids = self.splice_in(gap, insts.iter().map(|(id, inst)| (Some(*id), inst.clone())).collect())?;
        self.removed.retain(|id| !ids.contains(id));
        if checked { self.check_defined(&ids)?; }
        Ok(Edit::Remove { range: InstRange { first: ids[0], last: *ids.last().expect("non-empty") } })
    }

    // ---- Remove / Replace ----

    fn check_removable(&self, flow: &Flow, lo: usize, hi: usize) -> Result<(), EditError> {
        let body = self.body();
        for p in lo..=hi {
            let id = flow.ids[p];
            let inst = body.insts.get(id).ok_or(EditError::UnknownInst(id))?;
            match inst.effects.control {
                Control::EndPgm | Control::Halt | Control::Trap => return Err(EditError::ControlRemove { inst: id }),
                Control::Branch { .. } | Control::Jump => {
                    let neutral = is_edit_inserted(inst) && p + 1 == body.blocks[flow.block_of[p]].range.1
                        && label_of(inst) == Some(BlockId(flow.block_of[p] + 1));
                    if !neutral { return Err(EditError::ControlRemove { inst: id }); }
                }
                _ => {}
            }
        }
        check_window_integrity(body, lo, hi, true)?;
        let replay = passes::waits::replay(body, self.arch).map_err(|e| EditError::Analysis(e.to_string()))?;
        let range: HashSet<InstId> = flow.ids[lo..=hi].iter().copied().collect();
        let issuer: HashMap<_, _> = replay.events.iter().map(|e| (e.id, e.inst)).collect();
        for fact in replay.facts.iter().filter(|f| range.contains(&f.wait)) {
            for event in &fact.satisfies {
                let inst = issuer[event];
                if !range.contains(&inst) { return Err(EditError::WaitInUse { wait: fact.wait, event: inst }); }
            }
        }
        Ok(())
    }

    fn remove(&mut self, range: InstRange, checked: bool) -> Result<Edit, EditError> {
        let (block, lo, hi) = resolve_range(self.body(), range)?;
        if checked {
            let flow = self.flow()?;
            self.check_removable(&flow, lo, hi)?;
            let live = flow.live(None);
            let after = flow.live_out(&live, hi);
            let insts: Vec<&Inst> = flow.ids[lo..=hi].iter().map(|id| self.body().insts.get(*id).expect("live")).collect();
            let net = self.net_writes(&insts)?;
            let hit = net.inter(&after);
            if let Some(d) = hit.dwords().into_iter().next() {
                let inst = flow.ids[lo..=hi].iter().rev().copied()
                    .find(|id| access(self.arch, self.wave(), self.body().insts.get(*id).expect("live")).is_ok_and(|a| a.writes.dwords().contains(&d)))
                    .unwrap_or(range.last);
                return Err(EditError::RemoveLiveDef { inst, reg: dword_name(d) });
            }
        }
        let at = self.cursor_after(block, hi);
        let removed = self.splice_out(block, lo, hi)?;
        Ok(Edit::Restore { at, insts: removed })
    }

    fn replace(&mut self, range: InstRange, insts: &[Inst], checked: bool) -> Result<Edit, EditError> {
        if insts.is_empty() { return Err(EditError::InvalidInst { index: 0, reason: "empty replacement (use Remove)".into() }); }
        let stamped = insts.iter().enumerate().map(|(i, inst)| self.stamp(inst, i)).collect::<Result<Vec<_>, _>>()?;
        let (block, lo, hi) = resolve_range(self.body(), range)?;
        let at_end = hi + 1 == self.body().blocks[block].range.1;
        let gap = Gap { block, pos: lo, at_end };
        if checked {
            let flow = self.flow()?;
            self.check_removable(&flow, lo, hi)?;
            let live = flow.live(None);
            let after = flow.live_out(&live, hi);
            let old: Vec<&Inst> = flow.ids[lo..=hi].iter().map(|id| self.body().insts.get(*id).expect("live")).collect();
            let old_net = self.net_writes(&old)?;
            let new_refs: Vec<&Inst> = stamped.iter().collect();
            let new_net = self.net_writes(&new_refs)?;
            let orphaned = old_net.inter(&after).minus(&new_net);
            if let Some(d) = orphaned.dwords().into_iter().next() {
                return Err(EditError::RemoveLiveDef { inst: range.last, reg: dword_name(d) });
            }
            // The gap after removal: dealloc and control rules at the replaced position.
            let insert_gap = Gap { block, pos: lo, at_end };
            if flow.after_dealloc(self.body())[lo] { return Err(EditError::AfterDealloc { pos: lo }); }
            self.check_inserted_control(&stamped, insert_gap)?;
            self.check_writes(&new_net, &after, gap, &[], &old_net)?;
        }
        let at = self.cursor_after(block, hi);
        let removed = self.splice_out(block, lo, hi)?;
        let ids = self.splice_in(gap, stamped.into_iter().map(|i| (None, i)).collect())?;
        if checked {
            let body = self.body();
            let start = body.layout.iter().position(|x| *x == ids[0]).expect("spliced");
            let wins = windows(body);
            if let Some(&(_, _, opener, _)) = wins.iter().find(|(w_lo, w_hi, ..)| *w_lo < start && start <= *w_hi && !ids.contains(&body.layout[*w_lo])) {
                return Err(EditError::CursorInWindow { pos: start, opener });
            }
            self.check_defined(&ids)?;
        }
        let range = InstRange { first: ids[0], last: *ids.last().expect("non-empty") };
        Ok(Edit::Batch(vec![Edit::Remove { range }, Edit::Restore { at, insts: removed }]))
    }

    // ---- Move ----

    fn move_run(&mut self, range: InstRange, to: &Cursor, checked: bool) -> Result<Edit, EditError> {
        let (block, lo, hi) = resolve_range(self.body(), range)?;
        let back = self.cursor_after(block, hi);
        let gap = resolve_cursor(self.body(), to, checked)?;
        if gap.block == block && (lo..=hi + 1).contains(&gap.pos) { return Err(EditError::NoOpMove); }
        let ids: Vec<InstId> = self.body().layout[lo..=hi].to_vec();
        if checked {
            let flow = self.flow()?;
            for &id in &ids {
                let inst = self.body().insts.get(id).expect("live");
                if matches!(inst.effects.control, Control::Branch { .. } | Control::Jump | Control::EndPgm | Control::Halt | Control::Trap) {
                    return Err(EditError::ControlRemove { inst: id });
                }
            }
            check_window_integrity(self.body(), lo, hi, false)?;
            self.check_no_dealloc(&flow, gap)?;
            let proof = self.commutation(&flow, lo, hi, gap)?;
            self.commutations.push(proof);
        }
        let anchor = (gap.pos < self.body().blocks[gap.block].range.1).then(|| self.body().layout[gap.pos]);
        let barrier_run = ids.iter().any(|id| matches!(self.body().insts.get(*id).map(|i| i.effects.control), Some(Control::Barrier(_))));
        let base_barriers = if checked && barrier_run { Some(self.barrier_findings()?) } else { None };
        let mut seqs = seqs_of(self.body());
        let start = self.body().blocks[block].range.0;
        seqs[block].drain(lo - start..=hi - start);
        let at = match anchor { Some(a) => seqs[gap.block].iter().position(|x| *x == a).expect("anchor survives"), None => seqs[gap.block].len() };
        seqs[gap.block].splice(at..at, ids.iter().copied());
        install(self.body_mut(), seqs)?;
        for id in &ids {
            let acc = access(self.arch, self.wave(), self.body().insts.get(*id).expect("live"))?;
            self.touch(&acc);
        }
        if checked {
            self.check_defined(&ids)?;
            if let Some(base) = base_barriers { self.check_barriers_not_worse(&base)?; }
        }
        Ok(Edit::Move { range, to: back })
    }

    fn barrier_findings(&self) -> Result<Vec<(String, Vec<InstId>)>, EditError> {
        let analysis = passes::barriers::analyze(self.body(), self.arch).map_err(|e| EditError::Analysis(e.to_string()))?;
        Ok(analysis.obligations.into_iter().map(|o| (o.rule_id, o.insts)).collect())
    }

    fn check_barriers_not_worse(&self, base: &[(String, Vec<InstId>)]) -> Result<(), EditError> {
        for finding in self.barrier_findings()? {
            if !base.contains(&finding) { return Err(EditError::BarrierPairing(format!("{} at {:?}", finding.0, finding.1))); }
        }
        Ok(())
    }

    /// Control equivalence of run and destination plus pairwise commutation
    /// with every instruction between them on any path (core.md §6.2 Move).
    fn commutation(&self, flow: &Flow, lo: usize, hi: usize, gap: Gap) -> Result<CommutationProof, EditError> {
        let body = self.body();
        let n = flow.n;
        let g = n;
        let gap_edge = |x: usize, y: usize| y == gap.pos && (!gap.at_end || x + 1 == gap.pos);
        let succs = |x: usize| -> SmallVec<[usize; 2]> {
            if x == g { return smallvec::smallvec![gap.pos]; }
            flow.succ[x].iter().map(|&y| if gap_edge(x, y) { g } else { y }).collect()
        };
        let preds = |y: usize| -> SmallVec<[usize; 2]> {
            if y == g { return flow.pred[gap.pos].iter().copied().filter(|&x| gap_edge(x, gap.pos)).collect(); }
            flow.pred[y].iter().map(|&x| if gap_edge(x, y) { g } else { x }).collect()
        };
        let entry = if gap.pos == 0 && !gap.at_end { g } else { 0 };
        let size = n + 1;
        let is_exit = |x: usize| x < n && flow.succ[x].is_empty();
        let in_run = |x: usize| (lo..=hi).contains(&x);
        let from_entry_avoiding = |avoid: usize| flow.reach(&[entry], &|x| x == avoid, &succs, size);
        let dom_rg = entry == lo || !from_entry_avoiding(lo)[g];
        let dom_gr = entry == g || !from_entry_avoiding(g)[lo];
        let after_run: Vec<usize> = succs(hi).to_vec();
        let post_g = flow.reach(&after_run, &|x| x == g, &succs, size);
        let postdom_gr = !(0..n).any(|x| post_g[x] && is_exit(x));
        let post_r = flow.reach(&[gap.pos], &|x| x == lo, &succs, size);
        let postdom_rg = !(0..n).any(|x| post_r[x] && is_exit(x));
        let direction = if dom_rg && postdom_gr { MoveDirection::Down } else if dom_gr && postdom_rg { MoveDirection::Up } else {
            return Err(EditError::NotControlEquivalent("the run and the destination do not dominate/post-dominate each other".into()));
        };
        if flow.reach(&[gap.pos], &|x| x == lo, &succs, size)[g] {
            return Err(EditError::NotControlEquivalent("the destination repeats on a cycle that skips the run".into()));
        }
        if post_g[lo] {
            return Err(EditError::NotControlEquivalent("the run repeats on a cycle that skips the destination".into()));
        }
        let (fwd, bwd) = match direction {
            MoveDirection::Down => (
                flow.reach(&after_run, &|x| x == g || in_run(x), &succs, size),
                flow.reach(&preds(g), &|x| x == g || in_run(x), &preds, size),
            ),
            MoveDirection::Up => (
                flow.reach(&[gap.pos], &|x| x == g || in_run(x), &succs, size),
                flow.reach(&preds(lo), &|x| x == g || in_run(x), &preds, size),
            ),
        };
        let skipped: Vec<usize> = (0..n).filter(|&x| fwd[x] && bwd[x] && !in_run(x)).collect();
        let lds = passes::lds::analyze(body, self.arch).map_err(|e| EditError::Analysis(e.to_string()))?;
        let footprint: HashMap<InstId, (Option<(u64, u64)>, bool)> = lds.facts.accesses.iter().map(|(a, _)| {
            let name = body.insts.get(a.inst).and_then(|i| i.op.name(self.arch)).unwrap_or("");
            let range = match a.addr {
                AddrFact::Const(c) => Some((u64::from(c), u64::from(c) + u64::from(a.bytes))),
                AddrFact::Bounded { lo, .. } => Some((u64::from(lo), u64::MAX)),
                _ => None,
            };
            (a.inst, (range, name.contains("2addr") || a.bytes == 0))
        }).collect();
        let mut lds_disjoint = Vec::new();
        let run_has_wait = (lo..=hi).any(|r| body.insts.get(flow.ids[r]).is_some_and(|i| i.effects.control == Control::Wait));
        for &k in &skipped {
            let kid = flow.ids[k];
            let kin = body.insts.get(kid).expect("live");
            for r in lo..=hi {
                let rid = flow.ids[r];
                let rin = body.insts.get(rid).expect("live");
                let refuse = |reason: String| EditError::NotCommutable { moved: rid, skipped: kid, reason };
                if matches!(kin.effects.control, Control::Wait | Control::Barrier(_)) { return Err(refuse("crosses a wait or barrier".into())); }
                if is_sendmsg(kin, self.arch) || is_sendmsg(rin, self.arch) { return Err(refuse("crosses a message".into())); }
                let (ra, ka) = (&flow.acc[r], &flow.acc[k]);
                let clash = ra.writes.inter(&ka.uses.union(&ka.writes)).union(&ra.uses.inter(&ka.writes));
                if !clash.is_empty() { return Err(refuse(format!("register dependence on {}", bits_names(&clash)))); }
                let (rm, km) = (is_memory(rin, self.arch), is_memory(kin, self.arch));
                if run_has_wait && km { return Err(refuse("a moved wait would cross a memory instruction".into())); }
                if rm && km {
                    let fp = |id: InstId| footprint.get(&id).and_then(|(r, sketchy)| if *sketchy { None } else { *r });
                    match (fp(rid), fp(kid)) {
                        (Some(a), Some(b)) if a.1 <= b.0 || b.1 <= a.0 => lds_disjoint.push((rid, kid)),
                        _ => return Err(refuse("memory accesses without disjoint LDS address facts".into())),
                    }
                }
            }
        }
        Ok(CommutationProof { moved: flow.ids[lo..=hi].to_vec(), direction, skipped: skipped.iter().map(|&x| flow.ids[x]).collect(), lds_disjoint })
    }

    // ---- ReRegister ----

    fn reregister(&mut self, scope: &Scope, map: &[(RegRef, RegRef)], claims: &[RegClaim], checked: bool) -> Result<Edit, EditError> {
        let positions = self.scope_positions(scope)?;
        let bad = |m: String| EditError::ReRegister(m);
        if map.is_empty() { return Err(bad("empty map".into())); }
        let flow = self.flow()?;
        if checked {
            for (i, (old, new)) in map.iter().enumerate() {
                if old.kind != new.kind || old.len != new.len { return Err(bad(format!("{old} and {new} differ in kind or width"))); }
                if old.kind == Kind::Ttmp { return Err(bad("trap temporaries are never remapped".into())); }
                old.validate().map_err(bad)?;
                new.validate().map_err(bad)?;
                if old == new { return Err(bad(format!("{old} maps to itself"))); }
                if new.kind == Kind::S && new.len >= 2 && new.base % 2 != 0 { return Err(bad(format!("{new} is a misaligned SGPR tuple"))); }
                if let Some(c) = self.claims.iter().find(|c| c.reg.overlaps(*new)) { return Err(bad(format!("{new} is claimed by {}", c.name))); }
                for (j, (o2, n2)) in map.iter().enumerate() {
                    if i != j && (new.overlaps(*n2) || old.overlaps(*o2)) { return Err(bad("mapped ranges overlap".into())); }
                    if new.overlaps(*o2) { return Err(bad(format!("{new} is also renamed away"))); }
                }
            }
            let live = flow.live(None);
            let in_scope: HashSet<usize> = positions.iter().copied().collect();
            for (old, new) in map {
                let (ob, nb) = (Bits::of_reg(*old), Bits::of_reg(*new));
                for &p in &positions {
                    let acc = &flow.acc[p];
                    if !acc.uses.union(&acc.writes).inter(&nb).is_empty() { return Err(bad(format!("{new} is named in scope at {:?}", flow.ids[p]))); }
                    if !live[p].inter(&nb).is_empty() || !flow.live_out(&live, p).inter(&nb).is_empty() {
                        return Err(bad(format!("{new} is live in scope at {:?}", flow.ids[p])));
                    }
                    for x in &flow.pred[p] {
                        if !in_scope.contains(x) && !live[p].inter(&ob).is_empty() { return Err(bad(format!("{old} is live into the scope at {:?}", flow.ids[p]))); }
                    }
                    if p == 0 && !live[0].inter(&ob).is_empty() { return Err(bad(format!("{old} is live at the kernel entry"))); }
                    for &s in &flow.succ[p] {
                        if !in_scope.contains(&s) && !live[s].inter(&ob).is_empty() { return Err(bad(format!("{old} is live out of the scope at {:?}", flow.ids[s]))); }
                    }
                }
            }
            self.check_claims(&flow, claims)?;
        }
        let mut renamed = Vec::new();
        for &p in &positions {
            let id = flow.ids[p];
            let inst = self.body().insts.get(id).expect("live").clone();
            let mut operands = inst.operands.clone();
            let mut changed = false;
            for op in operands.iter_mut() {
                let (Operand::Reg(r) | Operand::Half(r, _)) = op else { continue };
                for (old, new) in map {
                    if r.overlaps(*old) {
                        if *r != *old { return Err(bad(format!("{id:?} names {r}, which only partly overlaps {old}"))); }
                        *r = *new;
                        changed = true;
                        break;
                    }
                }
            }
            if !changed { continue; }
            let rebuilt = Inst::from_parts(self.arch, inst.op, inst.form, inst.fields.clone(), operands, inst.mods.clone(), inst.literal, inst.prov.clone())
                .map_err(|e| bad(format!("{id:?}: {e}")))?;
            if checked {
                let mut probe = rebuilt.clone();
                for op in probe.operands.iter_mut() { if matches!(op, Operand::Label(_)) { *op = Operand::Imm(ImmField::Sopp(0)); } }
                crate::codec::gfx12::encode_for(self.arch, &probe).map_err(|e| bad(format!("{id:?} not encodable: {e}")))?;
                if vopd_banks_ok(&inst) && !vopd_banks_ok(&rebuilt) { return Err(bad(format!("{id:?} breaks the VOPD bank rule"))); }
            }
            let acc = access(self.arch, self.wave(), &rebuilt)?;
            self.touch(&acc);
            *self.body_mut().insts.get_mut(id).expect("live") = rebuilt;
            renamed.push(id);
        }
        if checked { self.check_defined(&renamed)?; }
        self.claims.extend(claims.iter().cloned());
        Ok(Edit::ReRegister { scope: scope.clone(), map: map.iter().map(|(o, n)| (*n, *o)).collect(), claims: Vec::new() })
    }

    // ---- SplitBlock / MergeBlock / Retarget ----

    fn split(&mut self, at: &Cursor, checked: bool) -> Result<Edit, EditError> {
        if at.before.is_none() { return Err(EditError::Split("a split needs an instruction to lead the new block".into())); }
        let gap = resolve_cursor(self.body(), at, checked)?;
        let (start, _) = self.body().blocks[gap.block].range;
        if gap.pos == start || gap.at_end { return Err(EditError::Split("the split point is already a block boundary".into())); }
        let mut seqs = seqs_of(self.body());
        let tail = seqs[gap.block].split_off(gap.pos - start);
        seqs.insert(gap.block + 1, tail);
        relabel(self.body_mut(), gap.block, 1);
        install(self.body_mut(), seqs)?;
        Ok(Edit::MergeBlock { block: BlockId(gap.block + 1) })
    }

    fn merge(&mut self, block: BlockId) -> Result<Edit, EditError> {
        let body = self.body();
        let b = block.0;
        if b == 0 || b >= body.blocks.len() { return Err(EditError::Merge(format!("{block:?} has no layout predecessor"))); }
        let prev_last = body.insts.get(body.layout[body.blocks[b - 1].range.1 - 1]).expect("live");
        if !matches!(prev_last.effects.control, Control::None | Control::Barrier(_) | Control::Wait | Control::Clause | Control::Delay) {
            return Err(EditError::Merge(format!("{:?} ends in control flow", BlockId(b - 1))));
        }
        if body.insts.iter().any(|(_, i)| label_of(i) == Some(block)) { return Err(EditError::Merge(format!("{block:?} is a branch target"))); }
        let leader = body.layout[body.blocks[b].range.0];
        let mut seqs = seqs_of(body);
        let tail = seqs.remove(b);
        seqs[b - 1].extend(tail);
        relabel(self.body_mut(), b, -1);
        install(self.body_mut(), seqs)?;
        Ok(Edit::SplitBlock { at: Cursor::before(BlockId(b - 1), leader) })
    }

    /// An inserted branch or jump may target (a) its own fall-through
    /// (neutral), (b) a block it alone reaches (dedicated code), or (c) a later
    /// block when every instruction it skips — the layout blocks between its
    /// own and the target — is edit-inserted, so the original instruction
    /// sequence of every path is unchanged and only inserted code is chosen.
    /// New paths are re-checked for barrier pairing and for the definedness
    /// of every inserted read.
    fn retarget(&mut self, branch: InstId, to: BlockId, checked: bool) -> Result<Edit, EditError> {
        let body = self.body();
        let p = body.layout.iter().position(|x| *x == branch).ok_or(EditError::UnknownInst(branch))?;
        let inst = body.insts.get(branch).expect("live");
        if !matches!(inst.effects.control, Control::Branch { .. } | Control::Jump) { return Err(EditError::Retarget(format!("{branch:?} is not a branch"))); }
        if to.0 >= body.blocks.len() { return Err(EditError::UnknownBlock(to)); }
        let old = label_of(inst).ok_or_else(|| EditError::Retarget(format!("{branch:?} has no label")))?;
        let from_block = body.blocks.iter().position(|b| b.range.0 <= p && p < b.range.1).expect("laid out");
        let neutralising = to.0 == from_block + 1;
        let skips_inserted_only = to.0 > from_block + 1
            && body.layout[body.blocks[from_block + 1].range.0..body.blocks[to.0].range.0].iter()
                .all(|id| body.insts.get(*id).is_some_and(is_edit_inserted));
        if checked && !is_edit_inserted(inst) { return Err(EditError::RetargetOriginal(branch)); }
        let base_barriers = if checked { Some(self.barrier_findings()?) } else { None };
        let inst = self.body_mut().insts.get_mut(branch).expect("live");
        for op in inst.operands.iter_mut() { if let Operand::Label(b) = op { *b = to; } }
        let seqs = seqs_of(self.body());
        install(self.body_mut(), seqs)?;
        if checked {
            let preds = &self.body().blocks[to.0].preds;
            // The entry block always has the kernel entry as an extra predecessor.
            let dedicated = to.0 != 0 && preds.len() == 1 && preds[0].0 == from_block;
            if !neutralising && !dedicated && !skips_inserted_only {
                return Err(EditError::Retarget(format!("{to:?} must be reached only by the inserted branch, or lie after it past inserted code only")));
            }
            if let Some(base) = base_barriers { self.check_barriers_not_worse(&base)?; }
            let inserted: Vec<InstId> = self.body().layout.iter().copied()
                .filter(|id| self.body().insts.get(*id).is_some_and(is_edit_inserted)).collect();
            self.check_defined(&inserted)?;
        }
        Ok(Edit::Retarget { branch, to: old })
    }

    // ---- SetWait / SetDelayHint ----

    fn set_wait(&mut self, id: InstId, imm: &WaitImm, checked: bool) -> Result<Edit, EditError> {
        let arch = self.arch;
        let inst = self.body().insts.get(id).filter(|_| self.body().layout.contains(&id)).ok_or(EditError::UnknownInst(id))?.clone();
        let name = inst.op.name(arch).unwrap_or("");
        let old = inst.mods.wait.clone().filter(|_| inst.effects.control == Control::Wait && name != "s_wait_alu").ok_or(EditError::NotAWait(id))?;
        let raw_old = inst.operands.iter().find_map(|op| match op { Operand::Imm(ImmField::Sopp(v)) => Some(*v as u16), _ => None })
            .ok_or_else(|| EditError::WaitImm("wait has no immediate".into()))?;
        let mut raw = raw_old;
        for c in 0..N {
            match (old.per_counter[c], imm.per_counter[c]) {
                (None, None) => {}
                (Some(_), Some(n)) => {
                    let (shift, mask) = wait_field(name, c);
                    if u16::from(n) > mask { return Err(EditError::WaitImm(format!("{n} exceeds the {name} field"))); }
                    raw = raw & !(mask << shift) | u16::from(n) << shift;
                }
                _ => return Err(EditError::WaitImm(format!("{name} cannot change which counters it waits on"))),
            }
        }
        let mut operands = inst.operands.clone();
        for op in operands.iter_mut() { if let Operand::Imm(ImmField::Sopp(v)) = op { *v = raw as i16; } }
        let mut mods = inst.mods.clone();
        mods.wait = Some(imm.clone());
        let rebuilt = Inst::from_parts(arch, inst.op, inst.form, inst.fields.clone(), operands, mods, inst.literal, inst.prov.clone())
            .map_err(|e| EditError::WaitImm(e.to_string()))?;
        let loosened = (0..N).any(|c| imm.per_counter[c] > old.per_counter[c]);
        let before = if checked && loosened { Some(self.wait_retires(id)?) } else { None };
        *self.body_mut().insts.get_mut(id).expect("live") = rebuilt;
        if let Some(before) = before {
            let after = self.wait_retires(id)?;
            if let Some(event) = before.iter().find(|e| !after.contains(e)) { return Err(EditError::WaitLoosened { wait: id, event: *event }); }
        }
        Ok(Edit::SetWait { inst: id, imm: old })
    }

    /// Issuing instructions of the events `wait` retires in the whole-CFG replay.
    fn wait_retires(&self, wait: InstId) -> Result<BTreeSet<InstId>, EditError> {
        let replay = passes::waits::replay(self.body(), self.arch).map_err(|e| EditError::Analysis(e.to_string()))?;
        let issuer: HashMap<_, _> = replay.events.iter().map(|e| (e.id, e.inst)).collect();
        Ok(replay.facts.iter().filter(|f| f.wait == wait).flat_map(|f| f.satisfies.iter().map(|e| issuer[e])).collect())
    }

    fn set_delay_hint(&mut self, id: InstId, hint: DelayAluHint, checked: bool) -> Result<Edit, EditError> {
        let body = self.body();
        let p = body.layout.iter().position(|x| *x == id).ok_or(EditError::UnknownInst(id))?;
        let inst = body.insts.get(id).expect("live");
        let old = inst.mods.delay.filter(|_| inst.effects.control == Control::Delay).ok_or(EditError::NotADelay(id))?;
        if checked {
            if hint.instid0 > 11 || hint.instid1 > 11 || hint.instskip > 5 { return Err(EditError::DelayHint(format!("{hint:?} has an invalid field"))); }
            let end = body.blocks.iter().find(|b| b.range.0 <= p && p < b.range.1).expect("laid out").range.1;
            let last = if hint.instid1 != 0 { p + 1 + usize::from(hint.instskip) } else { p + 1 };
            if last >= end { return Err(EditError::DelayHint("the hint's last target is past its block".into())); }
        }
        set_delay(self.body_mut().insts.get_mut(id).expect("live"), hint);
        Ok(Edit::SetDelayHint { inst: id, hint: old })
    }

    /// Re-point `VALU_DEP_n` hints at the producers they named before the
    /// step; a hint whose producer cannot be named any more becomes NO_DEP.
    fn maintain_hints(&mut self, before: Vec<HintSlot>) -> Result<Vec<Edit>, EditError> {
        let flow = self.flow()?;
        let mut originals: Vec<(InstId, DelayAluHint)> = Vec::new();
        for slot in before {
            let Some(&p) = flow.pos.get(&slot.hint) else { continue };
            let Some(hint) = self.body().insts.get(slot.hint).and_then(|i| i.mods.delay) else { continue };
            let Some(c) = hint_consumer(&flow, p, hint, slot.slot) else { continue };
            if producers(&flow, self.body(), c, slot.n) == slot.producers { continue; }
            let target = match slot.producers.iter().next() {
                Some(Some(producer)) if slot.producers.len() == 1 && flow.pos.contains_key(producer) =>
                    (1..=4u8).find(|&d| producers(&flow, self.body(), c, d) == slot.producers),
                _ => None,
            };
            let mut new = hint;
            if slot.slot == 0 { new.instid0 = target.unwrap_or(0); } else { new.instid1 = target.unwrap_or(0); }
            // Without a second dependency INSTSKIP names nothing; the assembler's
            // spelling (`s_delay_alu instid0(..)` / `s_delay_alu 0`) encodes it as 0.
            if new.instid1 == 0 { new.instskip = 0; }
            if !originals.iter().any(|(id, _)| *id == slot.hint) { originals.push((slot.hint, hint)); }
            set_delay(self.body_mut().insts.get_mut(slot.hint).expect("live"), new);
        }
        let mut restores = Vec::new();
        for (id, original) in originals {
            let after = self.body().insts.get(id).and_then(|i| i.mods.delay).expect("hint");
            if after != original {
                self.delay_rewrites.push(DelayRewrite { hint: id, before: original, after });
                restores.push(Edit::SetDelayHint { inst: id, hint: original });
            }
        }
        Ok(restores)
    }

    // ---- Descriptor / Metadata ----

    fn check_kernel(&self, kernel: &SymbolId) -> Result<(), EditError> {
        let current = &self.kernel().symbol;
        if kernel != current { return Err(EditError::KernelMismatch { edit: kernel.0.clone(), target: current.0.clone() }); }
        Ok(())
    }

    fn summary(&self) -> crate::state::ResourceSummary { passes::resources::summarize(self.body(), self.arch, self.wave(), 0) }

    fn rename(&mut self, name: &str, checked: bool) -> Result<String, EditError> {
        if checked {
            if name.is_empty() || !name.bytes().all(|b| b.is_ascii_graphic()) { return Err(EditError::Rename(format!("{name:?} is not a valid symbol"))); }
            if name.ends_with(".kd") { return Err(EditError::Rename("the .kd suffix is derived".into())); }
            if self.program.kernels.iter().enumerate().any(|(i, k)| i != self.k && k.symbol.0 == name) {
                return Err(EditError::Rename(format!("{name} is already a kernel")));
            }
        }
        let k = self.k;
        let old = std::mem::replace(&mut self.program.kernels[k].symbol, SymbolId(name.to_owned())).0;
        if let Abi::Hsa { metadata, .. } = &mut self.program.kernels[k].abi {
            metadata.parsed.name = name.to_owned();
            metadata.parsed.symbol = format!("{name}.kd");
        }
        if let Some(slot) = self.program.source.as_mut().and_then(|s| s.elf.kernels.get_mut(k)) { slot.name = name.to_owned(); }
        Ok(old)
    }

    fn descriptor(&mut self, kernel: &SymbolId, change: &DescriptorChange, checked: bool) -> Result<Edit, EditError> {
        self.check_kernel(kernel)?;
        let summary = self.summary();
        let arch = self.arch;
        let k = self.k;
        if let DescriptorChange::Rename(name) = change {
            if !matches!(self.kernel().abi, Abi::Hsa { .. }) { return Err(EditError::NotHsa); }
            let old = self.rename(name, checked)?;
            return Ok(Edit::Descriptor { kernel: SymbolId(name.clone()), change: DescriptorChange::Rename(old) });
        }
        let Abi::Hsa { descriptor, .. } = &mut self.program.kernels[k].abi else { return Err(EditError::NotHsa) };
        let bad = |m: String| EditError::Descriptor(m);
        let inverse = match change {
            DescriptorChange::KernargSize(n) => {
                if checked && *n > 0 && descriptor.kernel_code_properties.0 & (1 << 3) == 0 {
                    return Err(bad("kernarg segment pointer is not enabled".into()));
                }
                DescriptorChange::KernargSize(std::mem::replace(&mut descriptor.kernarg_size, *n))
            }
            DescriptorChange::NextFreeVgpr(n) => {
                let wave32 = descriptor.kernel_code_properties.wave32();
                let granule = if wave32 { 8 } else { 4 };
                if checked {
                    if *n == 0 || *n > 256 { return Err(bad(format!("next_free_vgpr {n} outside 1..=256"))); }
                    if *n < u32::from(summary.max_vgpr) { return Err(bad(format!("next_free_vgpr {n} is below the {} VGPRs the code uses", summary.max_vgpr))); }
                }
                let old = descriptor.compute_pgm_rsrc1.next_free_vgpr(wave32);
                let g = n.div_ceil(granule).saturating_sub(1);
                if g > 0x3f { return Err(bad(format!("granule {g} does not fit rsrc1[5:0]"))); }
                descriptor.compute_pgm_rsrc1.0 = descriptor.compute_pgm_rsrc1.0 & !0x3f | g;
                DescriptorChange::NextFreeVgpr(old)
            }
            DescriptorChange::NextFreeSgpr(_) => {
                return Err(bad(format!("{arch:?}: the SGPR granule is reserved on GFX10–12 (128 always allocated); use Metadata SetSgprCount")));
            }
            DescriptorChange::GroupSegmentFixedSize(n) => {
                if checked && *n > 65_536 { return Err(bad(format!("group segment {n} exceeds 64 KiB"))); }
                DescriptorChange::GroupSegmentFixedSize(std::mem::replace(&mut descriptor.group_segment_fixed_size, *n))
            }
            DescriptorChange::Rename(_) => unreachable!("handled above"),
        };
        Ok(Edit::Descriptor { kernel: kernel.clone(), change: inverse })
    }

    fn metadata(&mut self, kernel: &SymbolId, change: &MetaChange, checked: bool) -> Result<Edit, EditError> {
        self.check_kernel(kernel)?;
        let summary = self.summary();
        let k = self.k;
        if let MetaChange::Rename(name) = change {
            if !matches!(self.kernel().abi, Abi::Hsa { .. }) { return Err(EditError::NotHsa); }
            let old = self.rename(name, checked)?;
            return Ok(Edit::Metadata { kernel: SymbolId(name.clone()), change: MetaChange::Rename(old) });
        }
        let Abi::Hsa { metadata, .. } = &mut self.program.kernels[k].abi else { return Err(EditError::NotHsa) };
        let meta = &mut metadata.parsed;
        let bad = |m: String| EditError::Metadata(m);
        let inverse = match change {
            MetaChange::AppendArgs(args) => {
                if checked {
                    if args.is_empty() { return Err(bad("no arguments to append".into())); }
                    let mut end = meta.args.iter().map(|a| u64::from(a.offset) + u64::from(a.size)).max().unwrap_or(0);
                    for arg in args {
                        let align = u64::from(arg.size.clamp(1, 8).next_power_of_two());
                        if arg.size == 0 || arg.value_kind.is_empty() { return Err(bad(format!("argument {} needs a size and a value kind", arg.name))); }
                        if u64::from(arg.offset) < end { return Err(bad(format!("argument {} at {} overlaps the arguments before it (end {end})", arg.name, arg.offset))); }
                        if u64::from(arg.offset) % align != 0 { return Err(bad(format!("argument {} at {} is not {align}-aligned", arg.name, arg.offset))); }
                        end = u64::from(arg.offset) + u64::from(arg.size);
                    }
                }
                let old = MetaChange::SetArgs { args: meta.args.clone(), kernarg_segment_size: meta.kernarg_segment_size };
                let end = args.iter().map(|a| a.offset.saturating_add(a.size)).max().unwrap_or(0);
                meta.args.extend(args.iter().cloned());
                meta.kernarg_segment_size = meta.kernarg_segment_size.max(end);
                old
            }
            MetaChange::SetArgs { args, kernarg_segment_size } => {
                if checked {
                    let mut end = 0u64;
                    for arg in args {
                        if u64::from(arg.offset) < end { return Err(bad(format!("argument {} overlaps its predecessor", arg.name))); }
                        end = u64::from(arg.offset) + u64::from(arg.size);
                    }
                    if end > u64::from(*kernarg_segment_size) { return Err(bad("arguments end past the segment size".into())); }
                }
                let old = MetaChange::SetArgs { args: std::mem::replace(&mut meta.args, args.clone()), kernarg_segment_size: meta.kernarg_segment_size };
                meta.kernarg_segment_size = *kernarg_segment_size;
                old
            }
            MetaChange::SetVgprCount(n) => {
                if checked && (*n > 256 || *n < u32::from(summary.max_vgpr)) { return Err(bad(format!(".vgpr_count {n} outside {}..=256", summary.max_vgpr))); }
                MetaChange::SetVgprCount(std::mem::replace(&mut meta.vgpr_count, *n))
            }
            MetaChange::SetSgprCount(n) => {
                let need = u32::from(summary.max_sgpr) + if summary.uses_vcc { 2 } else { 0 };
                if checked && (*n > 108 || *n < need) { return Err(bad(format!(".sgpr_count {n} outside {need}..=108"))); }
                MetaChange::SetSgprCount(std::mem::replace(&mut meta.sgpr_count, *n))
            }
            MetaChange::Rename(_) => unreachable!("handled above"),
        };
        Ok(Edit::Metadata { kernel: kernel.clone(), change: inverse })
    }
}

/// `(shift, mask)` of counter `c` inside a gfx12 `s_wait_*` simm16.
fn wait_field(name: &str, c: usize) -> (u16, u16) {
    let combined = name.ends_with("loadcnt_dscnt") || name.ends_with("storecnt_dscnt");
    if combined { return if c == Counter::Ds as usize { (0, 0x3f) } else { (8, 0x3f) }; }
    match c {
        c if c == Counter::Km as usize => (0, 0x1f),
        c if c == Counter::Bvh as usize || c == Counter::Exp as usize => (0, 0x7),
        _ => (0, 0x3f),
    }
}

fn scalar_copy(arch: Arch, inst: &Inst) -> Option<(usize, usize)> {
    if inst.op.name(arch) != Some("s_mov_b32") { return None; }
    let dword = |op: &Operand| match op {
        Operand::Reg(r) if r.len == 1 && r.kind != Kind::V => reg_dwords(*r).next(),
        Operand::Special(s) => { let d = special_dwords(*s); (d.len() == 1).then(|| d[0]) }
        _ => None,
    };
    Some((dword(inst.operands.first()?)?, dword(inst.operands.get(1)?)?))
}

fn is_sendmsg(inst: &Inst, arch: Arch) -> bool { inst.op.name(arch).is_some_and(|n| n.starts_with("s_sendmsg")) }

fn is_memory(inst: &Inst, arch: Arch) -> bool {
    inst.effects.mem.is_some() || inst.op.name(arch).is_some_and(|n| n.starts_with("global_inv") || n.starts_with("buffer_gl") || n.starts_with("global_wb") || n.contains("_inv"))
}

/// VOPD: X/Y sources in different VGPR banks (`v % 4`) per source slot.
fn vopd_banks_ok(inst: &Inst) -> bool {
    let FormFields::Vopd { x_operands, .. } = inst.fields else { return true };
    let split = usize::from(x_operands);
    let bank = |op: Option<&Operand>| match op { Some(Operand::Reg(r)) if r.kind == Kind::V => Some(r.base % 4), _ => None };
    (1..3).all(|slot| match (bank(inst.operands.get(slot)), bank(inst.operands.get(split + slot))) {
        (Some(a), Some(b)) => a != b,
        _ => true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{KernargPreload, KernelCodeProperties, KernelDescriptor, Rsrc1, Rsrc2, Rsrc3};
    use crate::inst::{Frontend, KernelOrigin, Setting, Target};
    use crate::metadata::{HsaKernelMetadata, KernelMeta};
    use crate::operand::{InlineConst, Modifiers, VmemToken};
    use crate::passes::cfg::build_blocks;
    use crate::effects::MemClass;
    use crate::reg::ClaimOwner;
    use proptest::prelude::*;

    const ARCH: Arch = Arch::Gfx1201;
    const KT48: &str = "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_gfx1201";
    const KT48_TEXT: std::ops::Range<usize> = 0x6f00..0x6f00 + 10_604;
    const IMAGE: &[u8] = include_bytes!("../../peacemaker-lift/tests/fixtures/kt48/hipcc.co");

    // ---- instruction and program builders ----

    fn row(name: &str) -> &'static crate::isa::OpRow {
        crate::isa::gfx12().iter().find(|r| r.name == name).unwrap_or_else(|| panic!("no row {name}"))
    }
    /// The don't-care fields a zero-filled encoding decodes to (the codec
    /// refuses an `Inst` whose fields its own words would not reproduce).
    fn canonical_fields(r: &crate::isa::OpRow) -> FormFields {
        use crate::inst::NamedField;
        let mut ignored: SmallVec<[NamedField; 4]> = SmallVec::new();
        let mut honored: SmallVec<[NamedField; 4]> = SmallVec::new();
        if r.form == Form::Vop3 && !r.grammar.contains("SRC2:") && !r.benign_src2.is_empty() {
            if r.fields.len() == 1 { return FormFields::Vop3b { src2_unused: r.benign_src2[0] }; }
            ignored.push(NamedField { name: "src2_unused", value: u32::from(r.benign_src2[0]) });
        }
        for rule in r.fields.iter().filter(|rule| rule.name != "src2_unused") {
            let item = NamedField { name: rule.name, value: rule.allowed.first().copied().unwrap_or(0) };
            if rule.class == crate::isa::FieldClass::Honored { honored.push(item) } else { ignored.push(item) }
        }
        if ignored.is_empty() && honored.is_empty() { FormFields::None } else { FormFields::Bits { ignored, honored } }
    }
    fn mkm(name: &str, operands: Vec<Operand>, mods: Modifiers) -> Inst {
        let r = row(name);
        let inst = Inst::from_parts(ARCH, r.op, r.form, canonical_fields(r), SmallVec::from_vec(operands), mods, None, Provenance::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        if !inst.operands.iter().any(|op| matches!(op, Operand::Label(_))) {
            crate::codec::gfx12::encode(&inst).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
        inst
    }
    fn mk(name: &str, operands: Vec<Operand>) -> Inst { mkm(name, operands, Modifiers::default()) }
    fn raw(word: u32) -> Inst { crate::codec::gfx12::decode(&[word]).unwrap().0 }
    fn v(base: u16) -> Operand { vr(base, 1) }
    fn vr(base: u16, len: u8) -> Operand { Operand::Reg(RegRef { kind: Kind::V, base, len }) }
    fn s(base: u16) -> Operand { sr(base, 1) }
    fn sr(base: u16, len: u8) -> Operand { Operand::Reg(RegRef { kind: Kind::S, base, len }) }
    fn ttmp(base: u16) -> Operand { Operand::Reg(RegRef { kind: Kind::Ttmp, base, len: 1 }) }
    fn int(n: i8) -> Operand { Operand::Inline(InlineConst::Integer(n)) }
    fn vreg(base: u16, len: u8) -> RegRef { RegRef { kind: Kind::V, base, len } }
    fn sreg(base: u16, len: u8) -> RegRef { RegRef { kind: Kind::S, base, len } }
    fn wait(name: &str, counter: Counter, n: u8) -> Inst {
        let mut w = WaitImm::default();
        w.per_counter[counter as usize] = Some(n);
        mkm(name, vec![Operand::Imm(ImmField::Sopp(i16::from(n)))], Modifiers { wait: Some(w), ..Modifiers::default() })
    }
    fn dscnt(n: u8) -> Inst { wait("s_wait_dscnt", Counter::Ds, n) }
    fn delay(instid0: u8, instskip: u8, instid1: u8) -> Inst {
        raw(0xbf87_0000 | u32::from(instid0) | u32::from(instskip) << 4 | u32::from(instid1) << 7)
    }
    fn clause(n: u8) -> Inst { raw(0xbf85_0000 | u32::from(n)) }
    fn endpgm() -> Inst { mk("s_endpgm", vec![]) }
    fn nop() -> Inst { mk("s_nop", vec![Operand::Imm(ImmField::Sopp(0))]) }
    fn vmov(dst: u16, src: Operand) -> Inst { mk("v_mov_b32_e32", vec![v(dst), src]) }
    fn vadd(dst: u16, a: u16, b: u16) -> Inst { mk("v_add_f32_e32", vec![v(dst), v(a), v(b)]) }
    fn smov(dst: Operand, src: Operand) -> Inst { mk("s_mov_b32", vec![dst, src]) }
    fn ds_store(addr: u16, data: u16) -> Inst { mk("ds_store_b32", vec![v(addr), v(data)]) }
    fn ds_load(dst: u16, addr: u16) -> Inst { mk("ds_load_b32", vec![v(dst), v(addr)]) }
    fn gload(dst: u16, addr: u16) -> Inst { mk("global_load_b32", vec![v(dst), vr(addr, 2), Operand::Vmem(VmemToken::Off)]) }
    fn dealloc() -> Inst { raw(0xbfb6_0003) }
    fn wait_alu_sa() -> Inst { raw(0xbf88_ff9e) }

    enum It { I(Inst), L(&'static str), B(&'static str, &'static str) }
    use It::{B, I, L};

    /// Assemble items (labels, branches by label) into a body with built blocks.
    fn body_of(items: Vec<It>) -> Body {
        let mut insts = Vec::new();
        let mut labels = HashMap::new();
        let mut fixups = Vec::new();
        for item in items {
            match item {
                I(inst) => insts.push(inst),
                L(name) => { labels.insert(name, insts.len()); }
                B(op, target) => { fixups.push((insts.len(), target)); insts.push(mk(op, vec![Operand::Imm(ImmField::Sopp(0))])); }
            }
        }
        let mut pcs = vec![0usize];
        for inst in &insts { pcs.push(pcs.last().unwrap() + passes::cfg::dwords_of(inst, crate::inst::Arch::Gfx1201)); }
        for (at, target) in fixups {
            let off = pcs[labels[target]] as i64 - pcs[at + 1] as i64;
            insts[at].operands[0] = Operand::Imm(ImmField::Sopp(off as i16));
        }
        let mut body = Body::default();
        for inst in insts { let id = body.insts.insert(inst); body.layout.push(id); }
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        body
    }

    fn sym() -> SymbolId { SymbolId("k".into()) }

    fn program_of(body: Body) -> Program {
        let descriptor = KernelDescriptor { group_segment_fixed_size: 0, private_segment_fixed_size: 0, kernarg_size: 64,
            kernel_code_entry_byte_offset: 0, compute_pgm_rsrc3: Rsrc3(0), compute_pgm_rsrc1: Rsrc1(0x1f), compute_pgm_rsrc2: Rsrc2(0x384),
            kernel_code_properties: KernelCodeProperties(0x408), kernarg_preload: KernargPreload(0), reserved: [0; 28] };
        let parsed = KernelMeta { name: "k".into(), symbol: "k.kd".into(), kernarg_segment_size: 64, kernarg_segment_align: 8,
            vgpr_count: 256, sgpr_count: 108, wavefront_size: 32, max_flat_workgroup_size: 256, ..Default::default() };
        Program {
            target: Target { arch: ARCH, xnack: Setting::Any, sramecc: Setting::Any, abi_version: 4 },
            kernels: vec![Kernel { symbol: sym(), wave: Wave::Wave32,
                abi: Abi::Hsa { descriptor, metadata: HsaKernelMetadata { raw_msgpack: Vec::new(), parsed } }, body,
                origin: KernelOrigin::Authored { builder_crate: "c6".into(), version: "0".into(), git: "0".into() } }],
            source: None,
        }
    }

    fn analyzed(items: Vec<It>) -> Analyzed<Program> { analyze(program_of(body_of(items)), &sym()).unwrap() }

    #[test]
    fn immutable_guards_correlate_definitions_and_waits_but_mutable_guards_do_not() {
        let run = |mutate: bool, omit_wait: bool, omit_def: bool| {
            let mut items = vec![I(smov(s(32), s(0))), I(mk("s_cmp_ge_u32", vec![s(32), int(2)])),
                B("s_cbranch_scc1", "skip_def")];
            if !omit_def { items.push(I(ds_load(150, 1))); }
            items.push(L("skip_def"));
            if mutate { items.push(I(smov(s(32), s(1)))); }
            items.extend([I(mk("s_cmp_ge_u32", vec![s(32), int(2)])), B("s_cbranch_scc1", "done")]);
            if !omit_wait { items.push(I(dscnt(0))); }
            items.extend([I(vadd(151, 150, 150)), L("done"), I(endpgm())]);
            let a = analyzed(items);
            a.obligations.into_iter().filter(|o| o.insts == [a.program.kernels[0].body.layout[a.program.kernels[0].body.layout.len() - 2]]).collect::<Vec<_>>()
        };
        assert!(run(false, false, false).is_empty());
        assert!(run(true, false, false).iter().any(|o| o.kind == ObligationKind::Definedness));
        assert!(run(false, true, false).iter().any(|o| o.rule_id == "wait-raw-ds-load"));
        assert!(run(false, false, true).iter().any(|o| o.kind == ObligationKind::Definedness));
    }

    #[test]
    fn immutable_guard_definedness_survives_a_long_instruction_chain() {
        let mut items = vec![I(smov(s(32), s(0))), I(mk("s_cmp_ge_u32", vec![s(32), int(2)])),
            B("s_cbranch_scc1", "skip_def"), I(vmov(150, int(1))), L("skip_def")];
        items.extend((0..40_000).map(|_| I(nop())));
        items.extend([I(mk("s_cmp_ge_u32", vec![s(32), int(2)])),
            B("s_cbranch_scc1", "done"), I(vadd(151, 150, 150)), L("done"), I(endpgm())]);
        let program = program_of(body_of(items));
        let kernel = &program.kernels[0];
        let flow = Flow::new(&kernel.body, ARCH, kernel.wave).unwrap();
        let (defined, _) = flow.defined(entry_seed(kernel, ARCH).unwrap());
        let consumer = kernel.body.layout.len() - 2;
        assert!(flow.acc[consumer].reads.minus(&defined[consumer]).is_empty(),
            "both reaching immutable-guard outcomes must retain the guarded definition");
    }

    #[test]
    fn loop_carried_load_order_is_position_correct_at_a_join() {
        for threshold in [1, 2] {
            let a = analyzed(vec![I(gload(150, 0)), L("loop"), I(gload(151, 0)),
                I(wait("s_wait_loadcnt", Counter::Load, threshold)), I(vadd(152, 150, 150)),
                I(wait("s_wait_loadcnt", Counter::Load, 0)), I(gload(150, 0)),
                I(mk("s_cmp_eq_u32", vec![s(0), int(0)])), B("s_cbranch_scc1", "loop"),
                I(endpgm())]);
            let consumer = a.program.kernels[0].body.layout[3];
            assert_eq!(a.obligations.iter().any(|o| o.rule_id == "wait-raw-vmem-load" && o.insts == [consumer]),
                threshold == 2);
        }
    }

    #[test]
    fn unlike_join_sequences_do_not_count_mutually_exclusive_loads_as_younger() {
        let a = analyzed(vec![I(mk("s_cmp_eq_u32", vec![s(0), int(0)])), B("s_cbranch_scc1", "other"),
            I(gload(150, 0)), B("s_branch", "join"), L("other"),
            I(gload(151, 0)), I(gload(152, 0)), L("join"),
            I(wait("s_wait_loadcnt", Counter::Load, 1)), I(vadd(153, 150, 150)), I(endpgm())]);
        let consumer = a.program.kernels[0].body.layout[a.program.kernels[0].body.layout.len() - 2];
        assert!(a.obligations.iter().any(|o| o.rule_id == "wait-raw-vmem-load" && o.insts == [consumer]));
    }

    fn kt48_program() -> Program {
        let words: Vec<u32> = IMAGE[KT48_TEXT].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut body = Body::default();
        let mut index = 0;
        while index < words.len() {
            let (mut inst, count) = crate::codec::gfx12::decode(&words[index..]).unwrap();
            let mut bytes = [0u32; 3];
            bytes[..count].copy_from_slice(&words[index..index + count]);
            inst.prov = Provenance { source: Source::Object(crate::provenance::ObjectSha([0; 32])), pc: Some(0x7f00 + 4 * index as u32), bytes: Some(bytes), line: None, edit: None };
            let id = body.insts.insert(inst);
            body.layout.push(id);
            index += count;
        }
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        let descriptor = KernelDescriptor { group_segment_fixed_size: 0, private_segment_fixed_size: 0, kernarg_size: 328,
            kernel_code_entry_byte_offset: 23_744, compute_pgm_rsrc3: Rsrc3(0x530), compute_pgm_rsrc1: Rsrc1(0xe00f_001d),
            compute_pgm_rsrc2: Rsrc2(0x384), kernel_code_properties: KernelCodeProperties(0x408), kernarg_preload: KernargPreload(0), reserved: [0; 28] };
        // Scalar fields as pinned in core.md §0; the argument list is the lift layer's (C2/C7).
        let parsed = KernelMeta { name: KT48.into(), symbol: format!("{KT48}.kd"), kernarg_segment_size: 328, kernarg_segment_align: 8,
            vgpr_count: 238, sgpr_count: 30, wavefront_size: 32, max_flat_workgroup_size: 768, workgroup_processor_mode: true, ..Default::default() };
        Program {
            target: Target { arch: ARCH, xnack: Setting::Any, sramecc: Setting::Any, abi_version: 4 },
            kernels: vec![Kernel { symbol: SymbolId(KT48.into()), wave: Wave::Wave32,
                abi: Abi::Hsa { descriptor, metadata: HsaKernelMetadata { raw_msgpack: Vec::new(), parsed } }, body,
                origin: KernelOrigin::Frontend { kind: Frontend::Hipcc, object_sha256: [0; 32], entry_va: 0x7f00, size: 10_604 } }],
            source: None,
        }
    }

    /// Everything but tombstoned arena slots (ids are never reused).
    fn live_view(p: &Program) -> (SymbolId, Abi, Vec<InstId>, Vec<Block>, Vec<(InstId, Inst)>) {
        let k = &p.kernels[0];
        (k.symbol.clone(), k.abi.clone(), k.body.layout.clone(), k.body.blocks.clone(),
            k.body.layout.iter().map(|id| (*id, k.body.insts.get(*id).unwrap().clone())).collect())
    }

    fn stream(p: &Program) -> Vec<u32> { encode_stream(&p.kernels[0].body, ARCH).unwrap() }
    fn ids(a: &Analyzed<Program>) -> Vec<InstId> { a.program.kernels[0].body.layout.clone() }
    fn inst<'a>(a: &'a Analyzed<Program>, id: InstId) -> &'a Inst { a.program.kernels[0].body.insts.get(id).unwrap() }
    fn edit(a: &Analyzed<Program>, e: Edit) -> Result<(Analyzed<Program>, EditDelta), EditError> { a.edit(&a.program.kernels[0].symbol.clone(), e) }
    fn obligations(a: &Analyzed<Program>) -> Vec<(String, Vec<InstId>)> {
        let mut out: Vec<_> = a.obligations.iter().map(|o| (o.rule_id.clone(), o.insts.clone())).collect();
        out.sort();
        out
    }
    fn at(block: usize, id: InstId) -> Cursor { Cursor::before(BlockId(block), id) }
    fn claim(name: &str, reg: RegRef, scope: Scope) -> RegClaim { RegClaim { name: name.into(), reg, scope, owner: ClaimOwner::Builder } }
    fn insert(at: Cursor, insts: Vec<Inst>) -> Edit { Edit::Insert { at, insts, claims: Vec::new() } }

    /// Apply, check the forward candidate, undo, and require exact restoration.
    fn round_trip(a: &Analyzed<Program>, e: Edit) -> (Analyzed<Program>, EditDelta) {
        let (b, delta) = edit(a, e).unwrap();
        let (c, back) = b.undo(&delta).unwrap();
        assert_eq!(live_view(&c.program), live_view(&a.program), "undo restores the kernel");
        assert_eq!(stream(&c.program), stream(&a.program), "undo restores the stream");
        assert_eq!(obligations(&c), obligations(a), "undo restores the obligations");
        assert_eq!(c.revision, a.revision + 2);
        let (d, _) = c.undo(&back).unwrap();
        assert_eq!(live_view(&d.program), live_view(&b.program), "redo reproduces the candidate exactly");
        (b, delta)
    }

    // ---- codec / fact plumbing on KT48 ----

    #[test]
    fn kt48_stream_encodes_from_labels_and_cfg_reinstalls_exactly() {
        let program = kt48_program();
        let body = &program.kernels[0].body;
        let bytes: Vec<u8> = encode_stream(body, ARCH).unwrap().iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(bytes, IMAGE[KT48_TEXT]);
        let mut copy = body.clone();
        install(&mut copy, seqs_of(body)).unwrap();
        assert_eq!(&copy, body, "block reinstallation reproduces C4's CFG");
    }

    #[test]
    fn kt48_operand_roles_agree_with_table_effects() {
        let program = kt48_program();
        let body = &program.kernels[0].body;
        for &id in &body.layout {
            let inst = body.insts.get(id).unwrap();
            let roles = operand_roles(ARCH, inst).unwrap();
            let (mut defs, mut uses) = (Vec::new(), Vec::new());
            for (op, (def, used)) in inst.operands.iter().zip(roles) {
                if let Operand::Reg(r) | Operand::Half(r, _) = op {
                    if def { defs.push(*r); }
                    if used { uses.push(*r); }
                }
            }
            assert_eq!(defs.as_slice(), inst.effects.defs.as_slice(), "{}", inst.text(ARCH).unwrap());
            assert_eq!(uses.as_slice(), inst.effects.uses.as_slice(), "{}", inst.text(ARCH).unwrap());
        }
    }

    #[test]
    fn kt48_definedness_obligations_are_entry_facts() {
        let a = analyze(kt48_program(), &SymbolId(KT48.into())).unwrap();
        let undefined: Vec<&Obligation> = a.obligations.iter().filter(|o| o.kind == ObligationKind::Definedness).collect();
        // s_sendmsg's table row reads M0, which the ABI never initialises and
        // KT48 never writes; the VGPR findings are loads skipped on an
        // s_cbranch_execz path (lanes are not modelled) and true16 halves
        // written on one path only. The true16 pairs `v_mov_b16_e64 v198.l`
        // then `.h` (and v199) preserve a half the next instruction
        // overwrites; the preserved half is not read, so they add no
        // `definedness-partial-write` obligation (that class is for lane
        // writes, which the dword-granular state cannot follow).
        assert_eq!(undefined.len(), 27);
        let dealloc = undefined.iter().filter(|o| o.text.starts_with("reads m0 ")).count();
        assert_eq!(dealloc, 1);
        assert!(undefined.iter().all(|o| o.rule_id == "definedness-entry" && (o.text.contains("reads m0 ") || o.text.contains("reads v"))));
        assert_eq!(a.obligations.iter().filter(|o| o.kind == ObligationKind::SrcReadTiming(MemClass::DsStore)).count(), 124);
        assert!(!a.obligations.iter().any(|o| o.kind == ObligationKind::Unknown), "descriptor/metadata agree with the code");
    }

    #[test]
    fn lane_definedness_proves_only_the_lanes_observed_on_every_path() {
        let write = |lane| I(mk("v_writelane_b32", vec![v(200), int(7), int(lane)]));
        let exec = || Operand::Special(Special::ExecLo);
        let store = || I(ds_store(0, 200));
        let findings = |items| {
            let analyzed = analyzed(items);
            analyzed.obligations.iter().filter(|o| o.rule_id == "definedness-partial-write").count()
        };
        let narrow = vec![write(0), write(1), I(smov(s(20), exec())),
            I(smov(exec(), int(3))), store(), I(smov(exec(), s(20))), I(endpgm())];
        assert_eq!(findings(narrow), 0, "the two written lanes cover EXEC=0b11");
        assert_eq!(findings(vec![write(0), write(1), I(smov(exec(), int(7))), store(), I(endpgm())]), 1,
            "a store of lane 2 cannot read an unwritten lane");
        assert_eq!(findings(vec![write(0), write(1), store(), I(endpgm())]), 1,
            "the ABI's unknown initial EXEC may include unwritten lanes");
        assert_eq!(findings(vec![write(0), write(1),
            I(mk("s_and_b32", vec![exec(), exec(), int(3)])), store(), I(endpgm())]), 0,
            "AND with a constant bounds the active lanes even when initial EXEC is unknown");
        assert_eq!(findings(vec![write(0), I(mk("s_cmp_eq_u32", vec![s(0), int(0)])),
            B("s_cbranch_scc1", "skip"), write(1), B("s_branch", "join"),
            L("skip"), I(nop()), L("join"), I(smov(exec(), int(3))), store(), I(endpgm())]), 1,
            "a lane written on just one branch cannot cover the join");
        assert_eq!(findings(vec![write(0), I(mk("v_readlane_b32", vec![s(22), v(200), int(0)])), I(endpgm())]), 0,
            "a lane read observes its selected lane, not the current unknown EXEC");
        assert_eq!(findings(vec![write(0), I(mk("v_readlane_b32", vec![s(22), v(200), int(1)])), I(endpgm())]), 1,
            "a lane read of another lane is unsafe even if EXEC has no known mask");
        assert_eq!(findings(vec![I(smov(s(21), int(1))),
            I(mk("v_writelane_b32", vec![v(200), int(7), s(21)])), write(0),
            I(smov(exec(), int(3))), store(), I(endpgm())]), 0,
            "a lane index copied through an SGPR defines that specific lane");
        assert_eq!(findings(vec![
            I(mk("v_writelane_b32", vec![v(200), int(7), s(21)])),
            I(smov(exec(), int(1))), store(), I(endpgm())]), 1,
            "an unknown scalar lane index cannot establish coverage of lane 0");
        assert_eq!(findings(vec![write(0), I(smov(exec(), int(2))),
            I(vmov(200, int(9))), I(smov(exec(), int(3))), store(), I(endpgm())]), 0,
            "an EXEC-masked full VGPR write defines its definitely active lanes");
        assert_eq!(findings(vec![write(0), I(vmov(200, int(9))),
            I(smov(exec(), int(3))), store(), I(endpgm())]), 1,
            "a full VGPR write under unknown EXEC cannot prove lane 1");
        assert_eq!(findings(vec![write(0), I(smov(s(20), int(1))),
            I(mk("s_and_saveexec_b32", vec![s(21), s(20)])), store(), I(endpgm())]), 0,
            "saveexec AND bounds the lanes even when it saves an unknown old EXEC");
        assert_eq!(findings(vec![write(0), I(smov(s(20), int(1))),
            I(mk("s_or_saveexec_b32", vec![s(21), s(20)])), store(), I(endpgm())]), 1,
            "saveexec OR cannot bound an unknown old EXEC");
    }

    // ---- T10: the profiler entry script and its inverse on KT48 ----

    /// The profiler's entry record (core.md §6.4, `profile.rs` `entry`) in the
    /// gfx1201 rows the table carries: kernarg-extension loads, the wave's
    /// record pointer into the claimed s[28:29], EXEC saved in s31, one
    /// record store under a one-lane EXEC, EXEC restored. Temporaries s4..s12
    /// are written by KT48 before it reads them.
    fn profiler_entry() -> Vec<Inst> {
        let off = |n: i32| Operand::Imm(ImmField::SmemOffset(n));
        let exec = || Operand::Special(Special::ExecLo);
        vec![
            mk("s_load_b128", vec![sr(4, 4), sr(0, 2), off(0x148)]),
            mk("s_load_b32", vec![s(8), sr(0, 2), off(0x158)]),
            mk("s_load_b32", vec![s(9), sr(0, 2), off(0x15c)]),
            mk("v_readfirstlane_b32", vec![s(10), v(0)]),
            wait("s_wait_kmcnt", Counter::Km, 0),
            raw(0xbf88_f19f),
            mk("s_lshl_b32", vec![s(10), s(10), int(22)]),
            mk("s_lshr_b32", vec![s(10), s(10), int(27)]),
            mk("s_lshr_b32", vec![s(11), ttmp(7), int(16)]),
            mk("s_mul_i32", vec![s(11), s(11), s(9)]),
            mk("s_lshl_b32", vec![s(12), ttmp(7), int(16)]),
            mk("s_lshr_b32", vec![s(12), s(12), int(16)]),
            mk("s_add_co_i32", vec![s(11), s(11), s(12)]),
            mk("s_mul_i32", vec![s(11), s(11), s(8)]),
            mk("s_add_co_i32", vec![s(11), s(11), ttmp(9)]),
            mk("s_mul_i32", vec![s(12), s(11), s(7)]),
            mk("s_add_co_i32", vec![s(12), s(12), s(10)]),
            mk("s_mul_i32", vec![s(28), s(12), s(6)]),
            mk("s_mul_hi_u32", vec![s(29), s(12), s(6)]),
            mk("s_add_co_u32", vec![s(28), s(28), s(4)]),
            mk("s_add_co_ci_u32", vec![s(29), s(29), s(5)]),
            wait_alu_sa(),
            vmov(240, s(28)), vmov(241, s(29)),
            vmov(244, s(10)), vmov(245, s(11)), vmov(246, s(12)), vmov(247, int(0)),
            smov(s(31), exec()),
            smov(exec(), int(1)),
            mk("global_store_b128", vec![vr(240, 2), vr(244, 4), Operand::Vmem(VmemToken::Off)]),
            smov(exec(), s(31)),
        ]
    }

    #[test]
    fn t10_profiler_entry_script_then_inverse_restores_kt48() {
        let kernel = SymbolId(KT48.into());
        let base = analyze(kt48_program(), &kernel).unwrap();
        let base_stream = stream(&base.program);
        let profiled = format!("{KT48}__pm_profile");
        let arg = |name: &str, offset: u32, size: u32, kind: &str| Kernarg { name: name.into(), size, offset, value_kind: kind.into(), address_space: None };
        let args = vec![arg("pm_profile_records", 328, 8, "global_buffer"), arg("pm_slot_bytes", 336, 4, "by_value"),
            arg("pm_waves_per_wg", 340, 4, "by_value"), arg("pm_grid_x", 344, 4, "by_value"), arg("pm_grid_y", 348, 4, "by_value")];
        let claims = vec![claim("pm_pointer", sreg(28, 2), Scope::Whole), claim("pm_exec_save", sreg(31, 1), Scope::Whole),
            claim("pm_record", vreg(240, 8), Scope::Whole)];
        let record = profiler_entry();
        let record_dwords: usize = record.iter().map(|inst| passes::cfg::dwords_of(inst, Arch::Gfx1201)).sum();
        let script = Edit::Batch(vec![
            Edit::Descriptor { kernel: kernel.clone(), change: DescriptorChange::KernargSize(352) },
            Edit::Metadata { kernel: kernel.clone(), change: MetaChange::AppendArgs(args.clone()) },
            Edit::Descriptor { kernel: kernel.clone(), change: DescriptorChange::NextFreeVgpr(248) },
            Edit::Metadata { kernel: kernel.clone(), change: MetaChange::SetVgprCount(248) },
            Edit::Metadata { kernel: kernel.clone(), change: MetaChange::SetSgprCount(34) },
            Edit::Insert { at: at(0, ids(&base)[0]), insts: record, claims: claims.clone() },
            Edit::Metadata { kernel: kernel.clone(), change: MetaChange::Rename(profiled.clone()) },
        ]);
        let (edited, delta) = base.edit(&kernel, script).unwrap();

        assert_eq!((edited.revision, delta.base_revision), (1, 0));
        assert_eq!(delta.kernel.0, profiled);
        let k = &edited.program.kernels[0];
        let Abi::Hsa { descriptor, metadata } = &k.abi else { unreachable!() };
        assert_eq!((descriptor.kernarg_size, descriptor.compute_pgm_rsrc1.0), (352, 0xe00f_001e), "granule 29 -> 30");
        assert_eq!((metadata.parsed.name.as_str(), metadata.parsed.symbol.clone()), (profiled.as_str(), format!("{profiled}.kd")));
        assert_eq!((metadata.parsed.kernarg_segment_size, metadata.parsed.vgpr_count, metadata.parsed.sgpr_count), (352, 248, 34));
        assert_eq!(metadata.parsed.args, args);
        assert!(metadata.raw_msgpack.is_empty(), "typed view mutated; the raw slice stays the serialisation template");
        let edited_stream = stream(&edited.program);
        assert_eq!(edited_stream.len(), base_stream.len() + record_dwords);
        assert_eq!(&edited_stream[record_dwords..], &base_stream[..], "the original stream follows the record unchanged");
        assert!(delta.reencoded_branches.is_empty(), "entry insertion moves no branch relative to its target");
        assert_eq!(delta.claims, claims);
        assert_eq!(delta.inst_map.iter().find(|(id, _)| *id == ids(&base)[0]), Some(&(ids(&base)[0], Some(4 * record_dwords as u32))));
        assert!(delta.touched.0.contains(&sreg(28, 2)) && delta.touched.0.iter().any(|r| r.overlaps(vreg(247, 1))));
        assert_eq!(obligations(&edited), obligations(&base), "no new obligation anywhere (waits, hazards, barriers, definedness, ABI)");
        // Facts are re-derived for the new revision, never carried over.
        let inserted_wait = ids(&edited)[4];
        let fact = edited.facts.waits.iter().find(|f| f.wait == inserted_wait).expect("the inserted s_wait_kmcnt has a fact");
        assert_eq!(fact.satisfies.len(), 3, "it retires the three inserted SMEM loads");
        assert_eq!((edited.facts.resources[0].max_vgpr, edited.facts.resources[0].max_sgpr), (248, 32));
        assert_eq!((base.facts.resources[0].max_vgpr, base.facts.resources[0].max_sgpr), (238, 28));
        assert!(matches!(delta.inverse, Edit::Revert { .. }));

        let (restored, _) = edited.undo(&delta).unwrap();
        let bytes: Vec<u8> = stream(&restored.program).iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(bytes, IMAGE[KT48_TEXT], "stream identity with the original object bytes");
        assert_eq!(live_view(&restored.program), live_view(&base.program), "descriptor, metadata, symbol, CFG and instructions restored");
        assert_eq!(obligations(&restored), obligations(&base));
    }

    // ---- Insert preconditions ----

    fn clause_kernel() -> Analyzed<Program> {
        analyzed(vec![
            I(vmov(1, int(0))), I(vmov(2, int(0))), I(vadd(3, 1, 2)), I(delay(1, 0, 0)), I(vadd(4, 3, 3)),
            I(clause(1)), I(gload(10, 1)), I(gload(11, 1)),
            I(wait("s_wait_loadcnt", Counter::Load, 0)), I(vadd(12, 10, 11)), I(endpgm()),
        ])
    }

    #[test]
    fn insert_inside_a_window_is_refused_or_moved_by_policy() {
        let a = clause_kernel();
        let l = ids(&a);
        let (hint, consumer, opener, member1) = (l[3], l[4], l[5], l[7]);
        let err = edit(&a, insert(at(0, member1), vec![nop()])).unwrap_err();
        assert_eq!(err, EditError::CursorInWindow { pos: 7, opener });
        let err = edit(&a, insert(at(0, consumer), vec![nop()])).unwrap_err();
        assert_eq!(err, EditError::CursorInWindow { pos: 4, opener: hint });
        let (up, _) = edit(&a, insert(at(0, member1).with_policy(WindowPolicy::MoveUp), vec![nop()])).unwrap();
        assert_eq!(ids(&up)[6], opener, "moved up to just before the s_clause");
        let (down, _) = edit(&a, insert(at(0, member1).with_policy(WindowPolicy::MoveDown), vec![nop()])).unwrap();
        assert_eq!(ids(&down)[7], member1, "moved down past the last clause member");
        assert!(is_edit_inserted(inst(&down, ids(&down)[8])));
    }

    #[test]
    fn insert_writing_a_live_register_is_refused() {
        let a = analyzed(vec![
            I(mk("s_cmp_eq_u32", vec![s(0), int(0)])), I(vmov(3, int(1))), B("s_cbranch_scc1", "out"),
            I(vadd(4, 3, 3)), L("out"), I(endpgm()),
        ]);
        let l = ids(&a);
        let err = edit(&a, insert(at(0, l[2]), vec![mk("s_and_b32", vec![s(20), s(0), s(0)])])).unwrap_err();
        assert_eq!(err, EditError::WriteLive { reg: "scc".into() }, "SCC is live between s_cmp and s_cbranch_scc1");
        let err = edit(&a, insert(at(0, l[2]), vec![vmov(3, int(2))])).unwrap_err();
        assert_eq!(err, EditError::WriteLive { reg: "v3".into() });
        // A claim cannot license it: the original code names v3.
        let err = edit(&a, Edit::Insert { at: at(0, l[2]), insts: vec![vmov(3, int(2))], claims: vec![claim("x", vreg(3, 1), Scope::Whole)] }).unwrap_err();
        assert!(matches!(err, EditError::ClaimUnsound { .. }), "{err:?}");
        // A dead register is fine, and so is a live one under a sound claim:
        // v60 is set at entry and read before s_endpgm by inserted code only,
        // so rewriting it in between needs the claim that owns it.
        round_trip(&a, insert(at(0, l[2]), vec![vmov(50, v(0))]));
        let (b, _) = edit(&a, insert(at(0, l[0]), vec![vmov(60, v(0))])).unwrap();
        let (c, _) = edit(&b, insert(at(2, l[4]), vec![vmov(61, v(60))])).unwrap();
        let rewrite = vec![vmov(60, v(0))];
        assert_eq!(edit(&c, insert(at(1, l[3]), rewrite.clone())).unwrap_err(), EditError::WriteLive { reg: "v60".into() });
        round_trip(&c, Edit::Insert { at: at(1, l[3]), insts: rewrite, claims: vec![claim("pm", vreg(60, 2), Scope::Whole)] });
    }

    #[test]
    fn saved_and_restored_exec_is_not_a_write() {
        let a = analyzed(vec![I(vmov(3, int(1))), I(vadd(4, 3, 3)), I(endpgm())]);
        let l = ids(&a);
        let exec = || Operand::Special(Special::ExecLo);
        let guarded = vec![smov(s(20), exec()), smov(exec(), int(1)), vmov(200, v(0)), smov(exec(), s(20))];
        round_trip(&a, insert(at(0, l[1]), guarded));
        let clobber = vec![smov(exec(), int(1)), vmov(200, v(0))];
        assert_eq!(edit(&a, insert(at(0, l[1]), clobber)).unwrap_err(), EditError::WriteLive { reg: "exec_lo".into() });
    }

    /// A VGPR write under a narrowed EXEC leaves the other lanes' old value
    /// live: writing v5 before it would clobber what the full-EXEC read sees.
    #[test]
    fn vgpr_writes_under_a_narrowed_exec_do_not_end_liveness() {
        let exec = || Operand::Special(Special::ExecLo);
        let items = |narrow: bool| {
            let (save, set, restore) = if narrow {
                (smov(s(20), exec()), smov(exec(), int(1)), smov(exec(), s(20)))
            } else {
                (nop(), nop(), nop())
            };
            vec![I(vmov(5, int(0))), I(save), I(set), I(vmov(5, int(1))), I(restore), I(vadd(6, 5, 5)), I(endpgm())]
        };
        let narrowed = analyzed(items(true));
        let l = ids(&narrowed);
        assert_eq!(edit(&narrowed, insert(at(0, l[3]), vec![vmov(5, int(2))])).unwrap_err(), EditError::WriteLive { reg: "v5".into() });
        let same = analyzed(items(false));
        round_trip(&same, insert(at(0, ids(&same)[3]), vec![vmov(5, int(2))]));
    }

    /// core.md §6.2: one predecessor initialises v200, the other does not; a
    /// store of v200 at the join is refused although v200 is dead there, the
    /// claim on it is sound, and replay/hazards/resources would all pass.
    #[test]
    fn insert_reading_a_register_undefined_on_one_path_is_refused() {
        let diamond = |then_def: u16| analyzed(vec![
            I(mk("s_cmp_eq_u32", vec![s(0), int(0)])), B("s_cbranch_scc1", "else"),
            I(vmov(then_def, int(1))), B("s_branch", "join"),
            L("else"), I(vmov(200, int(2))),
            L("join"), I(endpgm()),
        ]);
        let store = |a: &Analyzed<Program>| {
            let join = a.program.kernels[0].body.blocks.len() - 1;
            let at_join = at(join, *ids(a).last().unwrap());
            Edit::Insert { at: at_join, insts: vec![ds_store(0, 200)], claims: vec![claim("v200", vreg(200, 1), Scope::Blocks(vec![BlockId(join)]))] }
        };
        let a = diamond(201);
        let err = edit(&a, store(&a)).unwrap_err();
        assert!(matches!(&err, EditError::Undefined { regs, .. } if regs == "v200"), "{err:?}");
        // Applied without preconditions, every other analysis is clean: the
        // only new obligation is the definedness finding at the store.
        let mut tx = Tx::new(a.program.clone(), 0, EditId(99));
        tx.step(&store(&a), false).unwrap();
        let candidate = analyze(tx.program, &sym()).unwrap();
        let new: Vec<_> = candidate.obligations.iter().filter(|o| !a.obligations.contains(o)).collect();
        assert_eq!(new.len(), 1);
        assert_eq!((new[0].kind.clone(), new[0].text.starts_with("reads v200 ")), (ObligationKind::Definedness, true));
        let b = diamond(200);
        round_trip(&b, store(&b));
        // Implicit reads count too: s_cselect reads SCC, which nothing defined.
        let c = analyzed(vec![I(vmov(3, int(1))), I(endpgm())]);
        let err = edit(&c, insert(at(0, ids(&c)[1]), vec![mk("s_cselect_b32", vec![s(20), int(1), int(0)])])).unwrap_err();
        assert!(matches!(&err, EditError::Undefined { regs, .. } if regs == "scc"), "{err:?}");
    }

    #[test]
    fn inserted_control_flow_must_be_a_neutral_branch() {
        // B0 = [s_cmp, s_cbranch], B1 = [v_mov v3] falls through into B2 = [v_mov v4, s_endpgm].
        let a = analyzed(vec![I(mk("s_cmp_eq_u32", vec![s(0), int(0)])), B("s_cbranch_scc1", "mid"), I(vmov(3, int(1))),
            L("mid"), I(vmov(4, int(2))), I(endpgm())]);
        let l = ids(&a);
        let control = |e: Result<(Analyzed<Program>, EditDelta), EditError>| matches!(e.unwrap_err(), EditError::ControlInsert { .. });
        assert!(control(edit(&a, insert(at(2, l[3]), vec![endpgm()]))), "terminators are never inserted");
        assert!(control(edit(&a, insert(Cursor::end(BlockId(1)), vec![mk("s_branch", vec![Operand::Label(BlockId(0))])]))), "nor non-neutral jumps");
        let branch_to = |b| mk("s_cbranch_scc1", vec![Operand::Label(BlockId(b))]);
        assert!(control(edit(&a, insert(at(1, l[2]), vec![branch_to(2)]))), "mid-block");
        assert!(control(edit(&a, insert(Cursor::end(BlockId(1)), vec![branch_to(0)]))), "not its fall-through");
        assert!(control(edit(&a, insert(Cursor::end(BlockId(1)), vec![branch_to(2), nop()]))), "not last");
        assert_eq!(edit(&a, insert(Cursor::end(BlockId(0)), vec![nop()])).unwrap_err(), EditError::CursorAfterTerminator(BlockId(0)));
        // M1 C6 finding: a neutral branch (taken == fall-through) inserted as its own
        // edit is a valid program with one successor edge; so is a neutral jump. Both
        // revert through their inverse.
        for neutral in [branch_to(2), mk("s_branch", vec![Operand::Label(BlockId(2))])] {
            let (b, _) = round_trip(&a, insert(Cursor::end(BlockId(1)), vec![neutral]));
            b.program.validate().unwrap();
            assert_eq!(b.program.kernels[0].body.blocks[1].succs.as_slice(), &[BlockId(2)]);
        }
    }

    #[test]
    fn nothing_is_inserted_after_msg_dealloc_vgprs() {
        let a = analyzed(vec![I(vmov(3, int(1))), I(nop()), I(dealloc()), I(endpgm())]);
        let l = ids(&a);
        assert_eq!(edit(&a, insert(at(0, l[3]), vec![nop()])).unwrap_err(), EditError::AfterDealloc { pos: 3 });
        round_trip(&a, insert(at(0, l[1]), vec![vmov(9, v(0))]));
    }

    #[test]
    fn inserted_effects_must_be_the_table_effects() {
        let a = analyzed(vec![I(vmov(3, int(1))), I(endpgm())]);
        let mut forged = vmov(9, v(0));
        forged.effects.defs.clear();
        let err = edit(&a, insert(at(0, ids(&a)[1]), vec![forged])).unwrap_err();
        assert!(matches!(err, EditError::InvalidInst { index: 0, .. }), "{err:?}");
    }

    // ---- s_delay_alu maintenance ----

    #[test]
    fn valu_dep_hints_follow_their_producer_and_undo_restores_them() {
        let a = analyzed(vec![I(vmov(1, int(1))), I(delay(1, 0, 0)), I(vadd(2, 1, 1)), I(endpgm())]);
        let l = ids(&a);
        let (b, delta) = round_trip(&a, insert(at(0, l[1]), vec![vmov(20, v(0))]));
        assert_eq!(inst(&b, l[1]).mods.delay.unwrap().instid0, 2, "VALU_DEP_1 -> VALU_DEP_2");
        assert_eq!(delta.delay_rewrites, vec![DelayRewrite { hint: l[1], before: DelayAluHint { instid0: 1, instskip: 0, instid1: 0 }, after: DelayAluHint { instid0: 2, instskip: 0, instid1: 0 } }]);
        let four: Vec<Inst> = (20..24).map(|r| vmov(r, v(0))).collect();
        let (c, _) = round_trip(&a, insert(at(0, l[1]), four));
        assert_eq!(inst(&c, l[1]).mods.delay.unwrap().instid0, 0, "beyond VALU_DEP_4 the hint becomes NO_DEP");
        // Removing a VALU between producer and hint shortens the distance.
        let e = analyzed(vec![I(vmov(1, int(1))), I(vmov(5, int(2))), I(delay(2, 0, 0)), I(vadd(2, 1, 1)), I(endpgm())]);
        let m = ids(&e);
        let (f, _) = round_trip(&e, Edit::Remove { range: InstRange::single(m[1]) });
        assert_eq!(inst(&f, m[2]).mods.delay.unwrap().instid0, 1, "VALU_DEP_2 -> VALU_DEP_1");
        // Removing the producer itself leaves nothing to name: NO_DEP.
        let g = analyzed(vec![I(vmov(1, int(1))), I(vmov(5, int(2))), I(delay(1, 0, 0)), I(vadd(2, 1, 1)), I(endpgm())]);
        let (h, _) = round_trip(&g, Edit::Remove { range: InstRange::single(ids(&g)[1]) });
        assert_eq!(inst(&h, ids(&g)[2]).mods.delay.unwrap().instid0, 0);
    }


    // ---- Remove / Replace ----

    #[test]
    fn remove_preconditions() {
        let a = analyzed(vec![
            I(vmov(1, int(0))), I(vmov(3, int(1))), I(vmov(5, int(2))), I(vadd(4, 3, 3)),
            I(ds_store(1, 4)), I(dscnt(0)), I(mk("s_cmp_eq_u32", vec![s(0), int(0)])), B("s_cbranch_scc1", "out"),
            I(nop()), L("out"), I(endpgm()),
        ]);
        let l = ids(&a);
        let one = |id| Edit::Remove { range: InstRange::single(id) };
        assert_eq!(edit(&a, one(l[1])).unwrap_err(), EditError::RemoveLiveDef { inst: l[1], reg: "v3".into() });
        assert_eq!(edit(&a, one(l[5])).unwrap_err(), EditError::WaitInUse { wait: l[5], event: l[4] });
        assert_eq!(edit(&a, one(l[7])).unwrap_err(), EditError::ControlRemove { inst: l[7] });
        assert_eq!(edit(&a, one(l[8])).unwrap_err(), EditError::RangeEmptiesBlock(InstRange::single(l[8])));
        assert_eq!(edit(&a, Edit::Remove { range: InstRange { first: l[3], last: l[1] } }).unwrap_err(), EditError::BadRange(InstRange { first: l[3], last: l[1] }));
        round_trip(&a, one(l[2]));
        // The store and the wait that retires it leave together.
        round_trip(&a, Edit::Remove { range: InstRange { first: l[4], last: l[5] } });

        let c = clause_kernel();
        let m = ids(&c);
        assert_eq!(edit(&c, one(m[5])).unwrap_err(), EditError::WindowBroken { opener: m[5] }, "opener with members left");
        assert_eq!(edit(&c, one(m[7])).unwrap_err(), EditError::WindowBroken { opener: m[5] }, "member without its opener");
        assert_eq!(edit(&c, one(m[4])).unwrap_err(), EditError::WindowBroken { opener: m[3] }, "a hint's target");
    }

    #[test]
    fn replace_preconditions() {
        let a = analyzed(vec![I(vmov(5, int(1))), I(vmov(3, int(1))), I(vadd(4, 3, 5)), I(endpgm())]);
        let l = ids(&a);
        let replace = |insts| Edit::Replace { range: InstRange::single(l[1]), insts };
        let (b, _) = round_trip(&a, replace(vec![vmov(3, int(2))]));
        assert_eq!(inst(&b, ids(&b)[1]).operands[1], int(2));
        assert_eq!(edit(&a, replace(vec![nop()])).unwrap_err(), EditError::RemoveLiveDef { inst: l[1], reg: "v3".into() });
        assert_eq!(edit(&a, replace(vec![vmov(3, int(2)), vmov(5, int(7))])).unwrap_err(), EditError::WriteLive { reg: "v5".into() });
        let err = edit(&a, replace(vec![vmov(3, v(90))])).unwrap_err();
        assert!(matches!(&err, EditError::Undefined { regs, .. } if regs == "v90"), "{err:?}");
    }

    // ---- Move ----

    #[test]
    fn move_requires_commutation_with_everything_it_crosses() {
        let a = analyzed(vec![I(vmov(3, int(1))), I(vmov(5, int(2))), I(vadd(6, 5, 5)), I(vadd(4, 3, 3)), I(endpgm())]);
        let l = ids(&a);
        let mv = |id, to| Edit::Move { range: InstRange::single(id), to: at(0, to) };
        let (b, delta) = round_trip(&a, mv(l[0], l[3]));
        assert_eq!(ids(&b), vec![l[1], l[2], l[0], l[3], l[4]]);
        assert_eq!(delta.commutations, vec![CommutationProof { moved: vec![l[0]], direction: MoveDirection::Down, skipped: vec![l[1], l[2]], lds_disjoint: vec![] }]);
        assert_eq!(inst(&b, l[0]).prov, inst(&a, l[0]).prov, "moves keep identity and provenance");
        let (_, up) = round_trip(&a, mv(l[3], l[1]));
        assert_eq!(up.commutations[0].direction, MoveDirection::Up);
        let err = edit(&a, mv(l[1], l[3])).unwrap_err();
        assert!(matches!(&err, EditError::NotCommutable { moved, skipped, .. } if *moved == l[1] && *skipped == l[2]), "{err:?}");
        assert_eq!(edit(&a, mv(l[1], l[2])).unwrap_err(), EditError::NoOpMove);

        let w = analyzed(vec![I(vmov(1, int(0))), I(ds_load(3, 1)), I(dscnt(0)), I(vmov(7, int(1))), I(vadd(4, 3, 3)), I(endpgm())]);
        let m = ids(&w);
        let err = edit(&w, Edit::Move { range: InstRange::single(m[3]), to: at(0, m[2]) }).unwrap_err();
        assert!(matches!(&err, EditError::NotCommutable { reason, .. } if reason.contains("wait")), "{err:?}");
    }

    #[test]
    fn move_across_memory_needs_disjoint_lds_facts() {
        let a = analyzed(vec![I(vmov(1, int(0))), I(vmov(2, int(0))), I(gload(3, 1)), I(gload(4, 1)),
            I(wait("s_wait_loadcnt", Counter::Load, 0)), I(vadd(5, 3, 4)), I(endpgm())]);
        let l = ids(&a);
        let err = edit(&a, Edit::Move { range: InstRange::single(l[2]), to: at(0, l[4]) }).unwrap_err();
        assert!(matches!(&err, EditError::NotCommutable { reason, .. } if reason.contains("memory")), "{err:?}");
        let lds = |b: i8| analyzed(vec![I(vmov(1, int(0))), I(vmov(2, int(b))), I(vmov(5, int(3))), I(ds_store(1, 5)), I(ds_load(6, 2)),
            I(dscnt(0)), I(vadd(7, 6, 6)), I(endpgm())]);
        let d = lds(16);
        let m = ids(&d);
        let (_, delta) = round_trip(&d, Edit::Move { range: InstRange::single(m[3]), to: at(0, m[5]) });
        assert_eq!(delta.commutations[0].lds_disjoint, vec![(m[3], m[4])]);
        let o = lds(0);
        let n = ids(&o);
        let err = edit(&o, Edit::Move { range: InstRange::single(n[3]), to: at(0, n[5]) }).unwrap_err();
        assert!(matches!(&err, EditError::NotCommutable { .. }), "overlapping LDS footprints: {err:?}");
    }

    #[test]
    fn move_destination_must_be_control_equivalent() {
        let a = analyzed(vec![
            I(mk("s_cmp_eq_u32", vec![s(0), int(0)])), I(vmov(3, int(1))), B("s_cbranch_scc1", "else"),
            I(vmov(4, int(2))), B("s_branch", "join"),
            L("else"), I(vmov(5, int(3))),
            L("join"), I(vadd(6, 3, 3)), I(endpgm()),
        ]);
        let l = ids(&a);
        let err = edit(&a, Edit::Move { range: InstRange::single(l[1]), to: at(1, l[3]) }).unwrap_err();
        assert!(matches!(err, EditError::NotControlEquivalent(_)), "{err:?}");
        let (_, delta) = round_trip(&a, Edit::Move { range: InstRange::single(l[1]), to: at(3, l[6]) });
        let proof = &delta.commutations[0];
        assert_eq!(proof.direction, MoveDirection::Down);
        let skipped: BTreeSet<_> = proof.skipped.iter().copied().collect();
        assert_eq!(skipped, BTreeSet::from([l[2], l[3], l[4], l[5]]), "both arms are crossed");

        let looped = analyzed(vec![
            I(vmov(3, int(1))), I(smov(s(5), int(3))),
            L("loop"), I(mk("s_add_co_i32", vec![s(5), s(5), int(-1)])), I(mk("s_cmp_eq_u32", vec![s(5), int(0)])), B("s_cbranch_scc0", "loop"),
            I(vadd(4, 3, 3)), I(endpgm()),
        ]);
        let m = ids(&looped);
        let err = edit(&looped, Edit::Move { range: InstRange::single(m[0]), to: at(1, m[2]) }).unwrap_err();
        assert!(matches!(err, EditError::NotControlEquivalent(_)), "into a loop body: {err:?}");
    }

    // ---- ReRegister ----

    #[test]
    fn reregister_preconditions() {
        let a = analyzed(vec![I(vmov(3, int(1))), I(vadd(4, 3, 3)), I(vmov(7, v(4))), I(endpgm())]);
        let l = ids(&a);
        let rename = |scope, old, new| Edit::ReRegister { scope, map: vec![(vreg(old, 1), vreg(new, 1))], claims: vec![] };
        let (b, _) = round_trip(&a, rename(Scope::Between(l[0], l[1]), 3, 50));
        assert_eq!(inst(&b, l[1]).operands.as_slice(), &[v(4), v(50), v(50)]);
        let bad = |e: Edit| matches!(edit(&a, e).unwrap_err(), EditError::ReRegister(_));
        assert!(bad(rename(Scope::Between(l[0], l[1]), 3, 4)), "the new name is used in scope");
        assert!(bad(rename(Scope::Between(l[1], l[1]), 3, 50)), "the old value flows into the scope");
        assert!(bad(rename(Scope::Between(l[0], l[0]), 3, 50)), "the old value flows out of the scope");
        assert!(bad(Edit::ReRegister { scope: Scope::Whole, map: vec![(vreg(3, 1), vreg(50, 2))], claims: vec![] }), "width");
        assert!(bad(Edit::ReRegister { scope: Scope::Whole, map: vec![(RegRef { kind: Kind::Ttmp, base: 7, len: 1 }, RegRef { kind: Kind::Ttmp, base: 3, len: 1 })], claims: vec![] }));
        let t = analyzed(vec![I(vmov(2, int(0))), I(vmov(3, int(0))), I(gload(5, 2)), I(wait("s_wait_loadcnt", Counter::Load, 0)), I(vadd(6, 5, 5)), I(endpgm())]);
        assert!(matches!(edit(&t, rename(Scope::Whole, 3, 50)).unwrap_err(), EditError::ReRegister(m) if m.contains("partly")));
    }

    // ---- SplitBlock / MergeBlock / Retarget ----

    #[test]
    fn split_and_merge_blocks() {
        let a = clause_kernel();
        let l = ids(&a);
        let (b, delta) = round_trip(&a, Edit::SplitBlock { at: at(0, l[1]) });
        assert_eq!(b.program.kernels[0].body.blocks.len(), 2);
        assert!(matches!(delta.inverse, Edit::Revert { ref undo, .. } if **undo == Edit::MergeBlock { block: BlockId(1) }));
        assert!(matches!(edit(&a, Edit::SplitBlock { at: at(0, l[0]) }).unwrap_err(), EditError::Split(_)), "already a leader");
        assert!(matches!(edit(&a, Edit::SplitBlock { at: at(0, l[6]) }).unwrap_err(), EditError::CursorInWindow { .. }));
        let (c, _) = round_trip(&a, Edit::SplitBlock { at: at(0, l[6]).with_policy(WindowPolicy::MoveUp) });
        assert_eq!(c.program.kernels[0].body.blocks[1].range.0, 5, "the split moved up to the s_clause");
        let t = analyzed(vec![I(mk("s_cmp_eq_u32", vec![s(0), int(0)])), B("s_cbranch_scc1", "mid"), I(nop()), L("mid"), I(endpgm())]);
        assert!(matches!(edit(&t, Edit::MergeBlock { block: BlockId(2) }).unwrap_err(), EditError::Merge(_)), "a branch target");
    }

    fn retarget_kernel(barrier: bool) -> Analyzed<Program> {
        let head = if barrier { mk("s_barrier_signal", vec![int(-1)]) } else { vmov(3, int(1)) };
        let tail = if barrier { raw(0xbf94_ffff) } else { vmov(4, int(2)) };
        analyzed(vec![I(mk("s_cmp_eq_u32", vec![s(0), int(0)])), I(head), I(tail), I(endpgm()), L("dead"), I(vmov(5, int(3))), I(endpgm())])
    }

    /// SplitBlock + neutral branch + Retarget (core.md §6.2): the only way a
    /// branch enters a kernel. Inserted ids are the arena's next slots.
    fn branch_script(a: &Analyzed<Program>, to: usize) -> Edit {
        let l = ids(a);
        let branch = InstId(a.program.kernels[0].body.insts.len());
        Edit::Batch(vec![
            Edit::SplitBlock { at: at(0, l[2]) },
            insert(Cursor::end(BlockId(0)), vec![mk("s_cbranch_scc1", vec![Operand::Label(BlockId(1))])]),
            Edit::Retarget { branch, to: BlockId(to) },
        ])
    }

    #[test]
    fn inserted_branch_retargets_to_a_block_it_alone_reaches() {
        let a = retarget_kernel(false);
        let (b, _) = round_trip(&a, branch_script(&a, 2));
        let blocks = &b.program.kernels[0].body.blocks;
        assert!(matches!(blocks[0].term, Terminator::Branch { taken: BlockId(2), fallthrough: BlockId(1), .. }));
        assert_eq!(blocks[2].preds.as_slice(), &[BlockId(0)], "the dead block is now reached only by the inserted branch");
        let err = edit(&a, branch_script(&a, 0)).unwrap_err();
        assert!(matches!(err, EditError::Retarget(_)), "{err:?}");
        let original = analyzed(vec![I(mk("s_cmp_eq_u32", vec![s(0), int(0)])), B("s_cbranch_scc1", "out"), I(nop()), L("out"), I(endpgm())]);
        let o = ids(&original);
        assert_eq!(edit(&original, Edit::Retarget { branch: o[1], to: BlockId(1) }).unwrap_err(), EditError::RetargetOriginal(o[1]));
        let s = retarget_kernel(true);
        let err = edit(&s, branch_script(&s, 2)).unwrap_err();
        assert!(matches!(err, EditError::BarrierPairing(_)), "the new edge skips the s_barrier_wait: {err:?}");
    }

    /// Inserted control may choose among inserted code: a forward Retarget is
    /// admitted when every instruction it skips is edit-inserted (prio-entry's
    /// priority switch, attn-sync's guarded signal), and refused when it would
    /// skip an original instruction.
    #[test]
    fn inserted_branch_may_skip_inserted_code_only() {
        let a = analyzed(vec![I(vmov(3, int(1))), I(vmov(4, int(2))), I(vmov(5, int(3))), I(endpgm())]);
        let l = ids(&a);
        let next = a.program.kernels[0].body.insts.len();
        let (cmp, alt, branch, jump) = (InstId(next), InstId(next + 1), InstId(next + 2), InstId(next + 3));
        // [v3, s_cmp, cbr->B1] [s_nop 0 ; s_branch->B2] [v4, v5, s_endpgm]
        let script = Edit::Batch(vec![
            insert(at(0, l[1]), vec![mk("s_cmp_eq_u32", vec![s(0), int(0)]), nop()]),
            Edit::SplitBlock { at: at(0, alt) },
            Edit::SplitBlock { at: at(1, l[1]) },
            insert(Cursor::end(BlockId(0)), vec![mk("s_cbranch_scc1", vec![Operand::Label(BlockId(1))])]),
            insert(Cursor::end(BlockId(1)), vec![mk("s_branch", vec![Operand::Label(BlockId(2))])]),
            Edit::Retarget { branch, to: BlockId(2) },
        ]);
        let (b, _) = round_trip(&a, script);
        let body = &b.program.kernels[0].body;
        assert_eq!(body.layout, vec![l[0], cmp, branch, alt, jump, l[1], l[2], l[3]]);
        assert!(matches!(body.blocks[0].term, Terminator::Branch { taken: BlockId(2), fallthrough: BlockId(1), .. }));
        assert_eq!(body.blocks[2].preds.as_slice(), &[BlockId(0), BlockId(1)], "a join after inserted code");
        // Skipping the original v_mov v4 is refused.
        let skip_original = Edit::Batch(vec![
            insert(at(0, l[1]), vec![mk("s_cmp_eq_u32", vec![s(0), int(0)])]),
            Edit::SplitBlock { at: at(0, l[1]) },
            Edit::SplitBlock { at: at(1, l[2]) },
            insert(Cursor::end(BlockId(0)), vec![mk("s_cbranch_scc1", vec![Operand::Label(BlockId(1))])]),
            Edit::Retarget { branch: InstId(next + 1), to: BlockId(2) },
        ]);
        assert!(matches!(edit(&a, skip_original).unwrap_err(), EditError::Retarget(_)));
    }

    // ---- SetWait / SetDelayHint ----

    #[test]
    fn set_wait_only_tightens_unless_the_replay_agrees() {
        let a = analyzed(vec![I(dscnt(0)), I(vmov(1, int(0))), I(ds_load(3, 1)), I(ds_load(4, 1)), I(dscnt(1)), I(vadd(5, 3, 3)),
            I(dscnt(0)), I(vadd(6, 4, 4)), I(endpgm())]);
        let l = ids(&a);
        let imm = |c: Counter, n: u8| { let mut w = WaitImm::default(); w.per_counter[c as usize] = Some(n); w };
        let (b, _) = round_trip(&a, Edit::SetWait { inst: l[4], imm: imm(Counter::Ds, 0) });
        assert_eq!(inst(&b, l[4]).operands[0], Operand::Imm(ImmField::Sopp(0)));
        assert_eq!(edit(&a, Edit::SetWait { inst: l[4], imm: imm(Counter::Ds, 2) }).unwrap_err(), EditError::WaitLoosened { wait: l[4], event: l[2] });
        assert_eq!(edit(&a, Edit::SetWait { inst: l[6], imm: imm(Counter::Ds, 1) }).unwrap_err(), EditError::WaitLoosened { wait: l[6], event: l[3] });
        round_trip(&a, Edit::SetWait { inst: l[0], imm: imm(Counter::Ds, 5) });
        assert!(matches!(edit(&a, Edit::SetWait { inst: l[4], imm: imm(Counter::Load, 0) }).unwrap_err(), EditError::WaitImm(_)));
        assert!(matches!(edit(&a, Edit::SetWait { inst: l[4], imm: imm(Counter::Ds, 64) }).unwrap_err(), EditError::WaitImm(_)));
        assert_eq!(edit(&a, Edit::SetWait { inst: l[1], imm: imm(Counter::Ds, 0) }).unwrap_err(), EditError::NotAWait(l[1]));
    }

    #[test]
    fn set_delay_hint_preconditions() {
        let a = analyzed(vec![I(vmov(1, int(1))), I(delay(1, 0, 0)), I(vadd(2, 1, 1)), I(endpgm())]);
        let l = ids(&a);
        let hint = |instid0, instskip, instid1| Edit::SetDelayHint { inst: l[1], hint: DelayAluHint { instid0, instskip, instid1 } };
        let (b, _) = round_trip(&a, hint(0, 0, 0));
        assert_eq!(inst(&b, l[1]).mods.delay, Some(DelayAluHint::default()));
        assert!(matches!(edit(&a, hint(12, 0, 0)).unwrap_err(), EditError::DelayHint(_)));
        assert!(matches!(edit(&a, hint(1, 3, 1)).unwrap_err(), EditError::DelayHint(_)), "second target past the block");
        assert_eq!(edit(&a, Edit::SetDelayHint { inst: l[0], hint: DelayAluHint::default() }).unwrap_err(), EditError::NotADelay(l[0]));
    }

    // ---- Descriptor / Metadata ----

    #[test]
    fn descriptor_and_metadata_preconditions() {
        let a = analyzed(vec![I(vmov(100, int(1))), I(endpgm())]);
        let k = sym();
        let desc = |change| Edit::Descriptor { kernel: sym(), change };
        let meta = |change| Edit::Metadata { kernel: sym(), change };
        let bad_desc = |e: Edit| matches!(edit(&a, e).unwrap_err(), EditError::Descriptor(_));
        let bad_meta = |e: Edit| matches!(edit(&a, e).unwrap_err(), EditError::Metadata(_));
        assert!(bad_desc(desc(DescriptorChange::NextFreeVgpr(64))), "below the 101 VGPRs in use");
        assert!(bad_desc(desc(DescriptorChange::NextFreeSgpr(40))), "no SGPR granule on gfx12");
        assert!(bad_desc(desc(DescriptorChange::GroupSegmentFixedSize(70_000))));
        let (b, _) = round_trip(&a, desc(DescriptorChange::NextFreeVgpr(104)));
        let Abi::Hsa { descriptor, .. } = &b.program.kernels[0].abi else { unreachable!() };
        assert_eq!(descriptor.compute_pgm_rsrc1.vgpr_granules(), 12);
        round_trip(&a, desc(DescriptorChange::KernargSize(96)));
        round_trip(&a, desc(DescriptorChange::GroupSegmentFixedSize(4096)));
        let arg = |name: &str, offset, size| Kernarg { name: name.into(), size, offset, value_kind: "by_value".into(), address_space: None };
        assert!(bad_meta(meta(MetaChange::AppendArgs(vec![arg("a", 64, 8), arg("b", 68, 4)]))), "overlap");
        assert!(bad_meta(meta(MetaChange::AppendArgs(vec![arg("a", 66, 4)]))), "misaligned");
        assert!(bad_meta(meta(MetaChange::SetVgprCount(64))));
        assert!(bad_meta(meta(MetaChange::SetSgprCount(200))));
        let (c, _) = round_trip(&a, meta(MetaChange::AppendArgs(vec![arg("a", 64, 8), arg("b", 72, 4)])));
        let Abi::Hsa { metadata, .. } = &c.program.kernels[0].abi else { unreachable!() };
        assert_eq!(metadata.parsed.kernarg_segment_size, 76);
        assert!(c.obligations.iter().any(|o| o.rule_id == "abi-kernarg-size"), "metadata now disagrees with the descriptor");
        assert!(matches!(edit(&a, meta(MetaChange::Rename("a b".into()))).unwrap_err(), EditError::Rename(_)));
        assert!(matches!(edit(&a, desc(DescriptorChange::Rename("x.kd".into()))).unwrap_err(), EditError::Rename(_)));
        let (r, delta) = round_trip(&a, desc(DescriptorChange::Rename("k2".into())));
        assert_eq!(delta.kernel, SymbolId("k2".into()));
        let Abi::Hsa { metadata, .. } = &r.program.kernels[0].abi else { unreachable!() };
        assert_eq!((metadata.parsed.name.as_str(), metadata.parsed.symbol.as_str()), ("k2", "k2.kd"));
        assert_eq!(r.edit(&k, desc(DescriptorChange::KernargSize(8))).unwrap_err(), EditError::UnknownKernel("k".into()));
        let other = Edit::Descriptor { kernel: SymbolId("other".into()), change: DescriptorChange::KernargSize(8) };
        assert!(matches!(a.edit(&k, other).unwrap_err(), EditError::KernelMismatch { .. }));
    }

    #[test]
    fn revert_is_admitted_only_from_the_post_edit_state() {
        let a = analyzed(vec![I(vmov(3, int(1))), I(endpgm())]);
        let (b, delta) = edit(&a, insert(at(0, ids(&a)[1]), vec![nop()])).unwrap();
        let (c, _) = b.undo(&delta).unwrap();
        assert!(matches!(c.undo(&delta).unwrap_err(), EditError::RevertStale { .. }));
        let (d, _) = edit(&b, insert(at(0, ids(&b)[0]), vec![nop()])).unwrap();
        assert!(matches!(d.undo(&delta).unwrap_err(), EditError::RevertStale { .. }), "a later edit must be undone first");
    }

    // ---- property: random insert/remove scripts undo to identity ----

    fn prop_items() -> Vec<It> {
        vec![
            I(vmov(1, int(0))), I(vmov(2, int(0))),
            I(mk("s_cmp_eq_u32", vec![s(0), int(0)])), B("s_cbranch_scc1", "else"),
            I(vadd(3, 1, 2)), I(delay(1, 0, 0)), I(vadd(4, 3, 3)), B("s_branch", "join"),
            L("else"), I(vadd(3, 2, 1)), I(vadd(4, 3, 1)),
            L("join"), I(smov(s(5), int(3))),
            L("loop"), I(vadd(4, 4, 3)), I(mk("s_add_co_i32", vec![s(5), s(5), int(-1)])), I(mk("s_cmp_eq_u32", vec![s(5), int(0)])),
            B("s_cbranch_scc0", "loop"),
            I(ds_store(1, 4)), I(dscnt(0)), I(endpgm()),
        ]
    }

    #[derive(Clone, Debug)]
    enum Op { Insert { at: usize, kind: u8, up: bool }, Remove { which: usize } }

    fn op() -> impl Strategy<Value = Op> {
        prop_oneof![
            3 => (any::<usize>(), 0u8..4, any::<bool>()).prop_map(|(at, kind, up)| Op::Insert { at, kind, up }),
            2 => any::<usize>().prop_map(|which| Op::Remove { which }),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(96))]
        #[test]
        fn random_insert_remove_scripts_undo_to_identity(ops in prop::collection::vec(op(), 1..10)) {
            let base = analyzed(prop_items());
            let mut cur = base.clone();
            let mut deltas = Vec::new();
            let mut groups: Vec<Vec<InstId>> = Vec::new();
            let mut fresh = 0u16;
            for op in ops {
                match op {
                    Op::Insert { at: seed, kind, up } => {
                        let layout = ids(&cur);
                        let pos = seed % layout.len();
                        let block = cur.program.kernels[0].body.blocks.iter().position(|b| b.range.0 <= pos && pos < b.range.1).unwrap();
                        let (r, sr) = (200 + fresh, 40 + fresh);
                        fresh += 2;
                        let insts = match kind {
                            0 => vec![nop()],
                            1 => vec![vmov(r, v(0))],
                            2 => vec![smov(s(sr), s(0))],
                            _ => vec![vmov(r, v(0)), vadd(r + 1, r, r)],
                        };
                        let policy = if up { WindowPolicy::MoveUp } else { WindowPolicy::MoveDown };
                        let cursor = Cursor { block: BlockId(block), before: Some(layout[pos]), window_policy: policy };
                        let (next, delta) = edit(&cur, insert(cursor, insts)).unwrap();
                        let before: HashSet<InstId> = layout.into_iter().collect();
                        groups.push(ids(&next).into_iter().filter(|id| !before.contains(id)).collect());
                        deltas.push(delta);
                        cur = next;
                    }
                    Op::Remove { which } => {
                        if groups.is_empty() { continue; }
                        let group = groups.remove(which % groups.len());
                        for id in group.into_iter().rev() {
                            let (next, delta) = edit(&cur, Edit::Remove { range: InstRange::single(id) }).unwrap();
                            prop_assert!(delta.inst_map.contains(&(id, None)));
                            deltas.push(delta);
                            cur = next;
                        }
                    }
                }
            }
            for delta in deltas.iter().rev() {
                cur = cur.undo(delta).unwrap().0;
            }
            prop_assert_eq!(live_view(&cur.program), live_view(&base.program));
            prop_assert_eq!(stream(&cur.program), stream(&base.program));
            prop_assert_eq!(obligations(&cur), obligations(&base));
        }
    }
}
