// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The byte-level half of the task-graph certificate for PeaceMaker persistent
//! pool objects (`railgun::task_graph` is the host half; plan §8.2).
//!
//! [`certify_pool`] is [`crate::certify`] for an object whose kernels carry a
//! `.note.pm_regions` table (`peacemaker_lift::regions_note`): the dispatcher
//! (pool) kernel plus the hosted task bodies it enters as *regions*. A tagged
//! kernel is upgraded from A1 `unknown` to `conditional` (witness
//! `descriptor_arena` + a [`TaskProtocolRecord`]) when, and only when, all of
//! these are proven from the bytes, and each failure stays an explicit
//! `task_graph-*` unresolved rule on the unchanged record:
//!
//! 1. **Regions.** Every region of the note is re-derived. A lifted region
//!    (non-zero source digest): the standalone object is looked up by digest
//!    (its SHA-256 is checked), converted with `lift_region` from the note's
//!    recorded inputs, assembled, and its words must equal the pool kernel's
//!    bytes in `[entry_pc, exit_pc)` exactly. A native region (zero digest):
//!    the region's own instructions are analysed by A1 in a region-scoped
//!    mode (the region cut out of the lifted CFG as a kernel whose `s[0:1]` is
//!    the record base, with no declared arguments, so every record pointer is
//!    a synthetic slot at its record offset).
//! 2. **Per-template arena facts.** A lifted region's standalone object must
//!    have a `proven` A1 record; its explicit pointer arguments (offset →
//!    classes) are the template's `pointer_loads` (inside the region
//!    `s[0:1]` is the record base, so a kernarg offset is a record offset).
//!    The same facts are re-derived by the region-scoped A1 of the region's
//!    bytes in the pool and must agree. They are emitted as one
//!    [`DescriptorArenaRecord`] whose arena argument is the dispatcher's
//!    `kernargs` pointer. The hidden arguments a body loads live in the same
//!    record at their standalone offsets (reported in the `task-graph` check).
//! 3. **Dispatcher shell.** A1 on the pool kernel with the regions excluded
//!    from attribution ([`peacemaker_ir::passes::kernargs::analyze_excluding`])
//!    must be `Proven`: every access outside the regions resolves to a pointer
//!    argument of the dispatcher's params block, the event cells are touched
//!    only by `atomic_release` and `load_acquire`, and no region instruction
//!    writes an SGPR of the dispatcher range or a trap temporary, or reaches
//!    outside the region's own LDS and control flow.
//! 4. **Protocol.** [`protocol`] runs a symbolic dataflow over the lifted
//!    pool CFG and proves the six [`TASK_PROTO_REQUIRED`] properties, which
//!    the host reads from the [`TaskProtocolRecord`], and, when it holds,
//!    `region_wg_from_header` with the header offsets of the virtual grid.
//!
//! # Complexity and work bounds
//!
//! The certificate is a fixed pipeline over finite objects, so it terminates; what this
//! section bounds is *how long*. Notation: `I`, `B` are the instructions and basic blocks
//! of the pool kernel, `R` its regions, `Is`, `Bs` those of a hosted standalone kernel,
//! `O` the instructions of a whole standalone object (all its kernels). Every iterative
//! phase carries an explicit work bound; one that is exceeded stops the phase and refuses
//! with a `task_graph-*-budget` rule naming the bound and the measured counts, the
//! kernel stays `unknown`, and nothing the stopped phase had computed is used, so a
//! refusal never admits anything. The bounds are [`PoolBudget`]'s fields (the same
//! figures are the defaults of [`certify_pool`]; [`certify_pool_with`] takes others).
//!
//! | phase | cost | bound (default) | refusal |
//! |---|---|---|---|
//! | note and kernel size | `O(1)` | `R <= 64`, `I <= 65536` (`max_regions`, `max_kernel_insts`) | `task_graph-size-budget` |
//! | whole-object receipt (`certify`, the unchanged baseline record) | lift `O(I)`; A1 and its guarded/sealed variants as below; peacemaker-ir passes: wait replay (`NoFixpoint` past 1000 walks per instruction), SGPR hazards (256 sweeps), LDS (widening after 3 updates per block), definedness (bitset dataflow) | the default [`A1Budget`] on its A1 runs | the record's own `a1-fixpoint-budget` unresolved (the whole-object record is replaced or refused by the obligations below) |
//! | standalone object | SHA-256 and `lift_object` `O(O)`, once per digest per call | none needed (linear) | `task_graph-region-source-*`, `task_graph-region-lift` |
//! | region re-derivation (`lift_region_in`) | liveness plus barrier-participation fixpoint over the (block, EXEC) product graph, `O(Is)` states | 200 000 rounds (`uniformity::explore`) | `task_graph-region-lift` |
//! | byte compare | `O(region words)` | linear | `task_graph-region-bytes` |
//! | standalone body A1 and sealed/guarded variants | one forward dataflow per `(digest, symbol)`: at most `work_per_inst * Is` units, a unit being one replayed instruction or block visit; the sealed entry/guard replays are bounded by 4096 paths and 400 000 steps | `A1Budget::fixpoint_work_per_inst` = 512 | `task_graph-a1-budget` (A1 rule `a1-fixpoint-budget`); `sealed-guard-path-budget` stays an A1 `sealed-*` rule |
//! | workgroup footprints of a body | one A1 run with compare facts per `(digest, symbol, block)` plus per-lane evaluation `O(block * forms)` | the same A1 bound; the footprints are withheld (never refused: an absent footprint is no claim) and the evidence line says why | evidence only |
//! | region-scoped A1 (native region; lifted cross-check) | `O(region)` cut plus one A1 run | `A1Budget` | `task_graph-a1-budget` (a cross-check only reports it) |
//! | dispatcher shell A1 (`analyze_excluding`, once) | one A1 run, `O(512 * I)` worst case | `A1Budget` | `task_graph-a1-budget` |
//! | protocol walk (`Walk`/`Fix`) | worklist over `(block, facts)` path partitions: a partition is joined monotonically and visited once per change of its state, a visit costs `O(block * alts)`; at most `2^18` fact sets per block in principle | state visits `<= 256 * B`, held partitions `<= 64 * B` per pass | `task_graph-protocol-budget` |
//! | protocol LDS-slot fixpoint | the walk repeated until the slot contents are stable (one extra pass for a multi-wave pool) | 5 passes, and the last pass must have been run on the contents it produced | `task_graph-protocol-budget` |
//! | protocol prelude (`region_exit_waited`, `region_defs`, `find_step_loops`, `on_inner_cycle`) | monotone must-dataflow on 4-bit masks (`<= 5` visits per block), linear scans, `O(claim-loop blocks * (B + E))` | linear or by lattice height | none |
//!
//! Measured (release, `railgun-cert pool --timings`; the table of every phase is printed per
//! pool): the A1 runs of the five fixture pools, their bodies and their mutations use 1.0 to
//! 21 units of work per instruction (the largest of 1414 fixpoints over every fixture of this
//! crate's tests is 63, in the Whole16 fixtures); the protocol walks
//! make 1.4 to 6.1 state visits and hold 0.8 to 5.1 partitions per block (b32 172 visits and
//! 94 partitions over 124 blocks, b64 177 and 149 over 29, b128 393 and 224 over 163, b256
//! 734 and 324 over 391, b320 177 and 149 over 59), in two passes for a multi-wave pool and
//! one for a single wave.
//! The A1, walk and region-count defaults are at least 8 times the largest measured value;
//! the pool-size bound (65 536 instructions against 14 315 measured) and the 5 LDS passes
//! (2 needed) keep smaller margins on purpose: they cap the worst case.
//!
//! Caching: a standalone object is looked up, hashed, lifted and radiowave-certified once
//! per [`certify_pool_with`] call, its kernel analysed once per symbol and its footprints
//! once per `(symbol, block)`; each cache entry is a function of exactly those keys, the
//! object's SHA-256 (checked) and the call's budget, so a hit is the value a recomputation
//! would give. The pool kernel is lifted and certified once and its shell A1 run once.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use peacemaker_ir::cfg::{BlockId, Body, Terminator};
use peacemaker_ir::codec::gfx12;
use peacemaker_ir::effects::Control;
use peacemaker_ir::inst::{Abi, Arch, Frontend, Kernel, Program};
use peacemaker_ir::kernarg::{KernargFacts, KernelStatus};
use peacemaker_ir::passes::kernargs::{self, A1Budget};
use peacemaker_ir::reg::Kind;
use peacemaker_ir::state::Lifted;
use peacemaker_lift::regions_note::{read_regions, KernelRegions, Region};
use peacemaker_lift::rewrite::region::{lift_region_in, RegionInputs};
use peacemaker_lift::{lift_object, Options};
use railgun_corpus::receipt::{
    ArenaPointerRecord, ArenaTemplateRecord, DescriptorArenaRecord, HiddenGridRecord,
    SealedTableRecord, TaskProtocolRecord, WgFootprintRecord, SEALED_BOUND_KERNARG,
    STATUS_CONDITIONAL, STATUS_PROVEN, STATUS_UNKNOWN, WITNESS_DESCRIPTOR_ARENA,
    WITNESS_SEALED_TABLE,
};
use sha2::{Digest, Sha256};

use crate::{
    arg_record, case_digest, kernel_record, metadata_effects, radiowave_cert, CertError, Check,
    KernelRecord, Receipt, UnresolvedRecord,
};

pub mod protocol;

/// The standalone-object resolver of [`certify_pool`]: the object whose SHA-256
/// is the argument, if the caller has it.
pub type Standalone<'a> = &'a dyn Fn(&[u8; 32]) -> Option<Vec<u8>>;

/// The work bounds of one [`certify_pool_with`] call (module docs, "Complexity and work
/// bounds"). A bound that is exceeded refuses with a `task_graph-*-budget` rule naming it and
/// the measured counts; the kernel then stays `unknown`. A refusal is never an admission:
/// every proof obligation still has to hold on the work that was not cut short, and a
/// refused analysis contributes no fact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolBudget {
    /// Regions one pool kernel's note may list (`task_graph-size-budget`).
    pub max_regions: usize,
    /// Instructions of one analysed kernel: the pool kernel, or a hosted standalone body
    /// (`task_graph-size-budget`).
    pub max_kernel_insts: usize,
    /// Every A1 analysis the certificate runs on a body, a region cut or the dispatcher
    /// shell (`task_graph-a1-budget`; the standalone footprint analysis reports it as
    /// evidence and withholds the footprints).
    pub a1: A1Budget,
    /// The protocol dataflow (`task_graph-protocol-budget`).
    pub protocol: protocol::Budget,
}

impl Default for PoolBudget {
    fn default() -> Self {
        PoolBudget {
            max_regions: 64,
            max_kernel_insts: 1 << 16,
            a1: A1Budget::default(),
            protocol: protocol::Budget::default(),
        }
    }
}

/// Obligation names a conditional pool record adds to the receipt.
pub const OBLIGATION_ARENA: &str = "DescriptorArenaWitness";
pub const OBLIGATION_PROTOCOL: &str = "TaskProtocol";

/// A failed obligation: stays on the record as an `Unresolved` with this rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub rule: &'static str,
    pub detail: String,
    pub pc: Option<u32>,
}

pub(crate) fn refuse(rule: &'static str, detail: impl Into<String>) -> Refusal {
    Refusal {
        rule,
        detail: detail.into(),
        pc: None,
    }
}

/// What the certificate established about one pool kernel (conditional upgrade).
#[derive(Clone, Debug)]
pub struct PoolProof {
    pub arena: DescriptorArenaRecord,
    /// The span table (`spans` argument) as a binding arena, when the dispatcher has span
    /// mode and it is proved: its pointer fields are loaded from descriptor records.
    pub span_arena: Option<DescriptorArenaRecord>,
    pub protocol: TaskProtocolRecord,
    /// Human-readable evidence lines (the `task-graph` check detail).
    pub evidence: Vec<String>,
}

// ---------------------------------------------------------------------------
// object bytes
// ---------------------------------------------------------------------------

/// The code bytes of kernel `symbol` in the ELF `elf`.
fn kernel_text<'a>(elf: &'a [u8], symbol: &str) -> Result<&'a [u8], String> {
    use object::{Object, ObjectSection, ObjectSymbol};
    let file = object::File::parse(elf).map_err(|e| format!("ELF parse: {e}"))?;
    let sym = file
        .symbols()
        .find(|s| s.name() == Ok(symbol) && !s.is_undefined())
        .ok_or_else(|| format!("kernel symbol {symbol:?} not found"))?;
    let index = sym
        .section_index()
        .ok_or_else(|| format!("{symbol}: no defining section"))?;
    let section = file
        .section_by_index(index)
        .map_err(|e| format!("{symbol}: {e}"))?;
    let data = section.data().map_err(|e| format!("{symbol}: {e}"))?;
    let at = sym
        .address()
        .checked_sub(section.address())
        .ok_or_else(|| format!("{symbol}: below its section"))? as usize;
    data.get(at..at + sym.size() as usize)
        .ok_or_else(|| format!("{symbol}: outside its section"))
}

// ---------------------------------------------------------------------------
// region assembly (the native writer's text path: `parse_line` + the codec,
// branch labels resolved to word offsets)
// ---------------------------------------------------------------------------

fn encode_line(line: &str, arch: Arch) -> Result<Vec<u32>, String> {
    let inst = peacemaker_lift::text::parse_line(line, arch).map_err(|e| format!("{line}: {e}"))?;
    gfx12::encode(&inst)
        .map(|w| w.to_vec())
        .map_err(|e| format!("{line}: {e}"))
}

/// The words `Builder::foreign_region` emits between a region's entry and exit
/// labels for a converted body: `s_wait_alu 0x0`, then its lines with every
/// label branch resolved.
pub fn region_words(lines: &[String], arch: Arch) -> Result<Vec<u32>, String> {
    let mut words: Vec<u32> = Vec::new();
    let mut labels: HashMap<&str, usize> = HashMap::new();
    let mut branches: Vec<(usize, &str, &str)> = Vec::new();
    words.extend(encode_line("s_wait_alu 0x0", arch)?);
    for line in lines {
        if let Some(label) = line.strip_suffix(':') {
            if labels.insert(label, words.len()).is_some() {
                return Err(format!("duplicate label {label}"));
            }
            continue;
        }
        let (head, rest) = line
            .split_once(char::is_whitespace)
            .map_or((line.as_str(), ""), |(h, r)| (h, r.trim()));
        if (head == "s_branch" || head.starts_with("s_cbranch_")) && rest.starts_with(".L") {
            branches.push((words.len(), head, rest));
            words.push(0);
        } else {
            words.extend(encode_line(line, arch)?);
        }
    }
    for (at, mnemonic, label) in branches {
        let target = *labels
            .get(label)
            .ok_or_else(|| format!("undefined label {label}"))?;
        let offset = target as i64 - (at as i64 + 1);
        let words_of = encode_line(&format!("{mnemonic} {offset}"), arch)?;
        let [word] = words_of[..] else {
            return Err(format!("{mnemonic}: branch is not one dword"));
        };
        words[at] = word;
    }
    Ok(words)
}

// ---------------------------------------------------------------------------
// the pool kernel as the checks see it
// ---------------------------------------------------------------------------

/// One region of the note located in the lifted kernel's layout.
#[derive(Clone, Debug)]
pub struct Placed {
    pub region: Region,
    /// Layout position of the instruction at `entry_pc`, and of the first
    /// instruction at or after `exit_pc` (the layout length when none).
    pub p_entry: usize,
    pub p_exit: usize,
}

/// The address of the kernel's first instruction: `Provenance::pc` of a lifted
/// instruction is an address in the object's text, the region note's offsets are
/// relative to the kernel symbol.
pub(crate) fn text_base(kernel: &Kernel) -> u32 {
    match &kernel.origin {
        peacemaker_ir::inst::KernelOrigin::Frontend { entry_va, .. } => *entry_va as u32,
        _ => 0,
    }
}

/// `pc` relative to the kernel symbol.
pub(crate) fn rel_pc(kernel: &Kernel, pc: Option<u32>) -> Option<u32> {
    pc.map(|p| p.wrapping_sub(text_base(kernel)))
}

/// Layout position of each instruction's byte offset from the kernel symbol.
pub(crate) fn pc_positions(kernel: &Kernel) -> BTreeMap<u32, usize> {
    let body = &kernel.body;
    body.layout
        .iter()
        .enumerate()
        .filter_map(|(pos, &id)| Some((rel_pc(kernel, body.insts.get(id)?.prov.pc)?, pos)))
        .collect()
}

fn place_regions(kernel: &Kernel, notes: &KernelRegions) -> Result<Vec<Placed>, Refusal> {
    let by_pc = pc_positions(kernel);
    let mut out = Vec::new();
    for r in &notes.regions {
        let (entry, exit) = (r.entry_pc as u32, r.exit_pc as u32);
        let p_entry = *by_pc.get(&entry).ok_or_else(|| {
            refuse(
                "task_graph-region-structure",
                format!(
                    "template {}: entry {entry:#x} is not an instruction boundary",
                    r.template_id
                ),
            )
        })?;
        let p_exit = by_pc
            .range(exit..)
            .next()
            .map_or(kernel.body.layout.len(), |(_, &p)| p);
        if by_pc.get(&exit).is_none() && p_exit != kernel.body.layout.len() {
            return Err(refuse(
                "task_graph-region-structure",
                format!(
                    "template {}: exit {exit:#x} is not an instruction boundary",
                    r.template_id
                ),
            ));
        }
        out.push(Placed {
            region: r.clone(),
            p_entry,
            p_exit,
        });
    }
    Ok(out)
}

fn mnemonic(kernel: &Kernel, arch: Arch, pos: usize) -> &'static str {
    kernel
        .body
        .insts
        .get(kernel.body.layout[pos])
        .and_then(|i| i.op.name(arch))
        .unwrap_or("")
}

/// Structural checks of one placed region over the lifted CFG: the region is
/// entered only at its entry and left only through its exit, no instruction
/// names a trap temporary, none ends or traps the wave, and a native region touches
/// no LDS (its barriers are admitted by [`derive_native`] only with the barrier-participation
/// proof: every wave of the pool's workgroup reaches each one the same number of times).
fn check_region_structure(kernel: &Kernel, arch: Arch, p: &Placed) -> Vec<Refusal> {
    let body = &kernel.body;
    let t = p.region.template_id;
    let mut out = Vec::new();
    let inside = |pos: usize| (p.p_entry..p.p_exit).contains(&pos);
    for block in &body.blocks {
        let (s, e) = block.range;
        if s >= e {
            continue;
        }
        let last_inside = inside(e - 1);
        for succ in &block.succs {
            let target = body.blocks[succ.0].range.0;
            let enters_interior = target > p.p_entry && target < p.p_exit;
            if enters_interior && !last_inside {
                out.push(refuse(
                    "task_graph-region-structure",
                    format!("template {t}: a branch from outside enters the middle of the region (block {} -> {})", block.id.0, succ.0),
                ));
            }
            if last_inside && !(inside(target) || target == p.p_exit) {
                out.push(refuse(
                    "task_graph-region-structure",
                    format!(
                        "template {t}: the region branches out of itself (block {} -> {})",
                        block.id.0, succ.0
                    ),
                ));
            }
        }
        if last_inside && matches!(block.term, Terminator::EndPgm | Terminator::Unreachable) {
            out.push(refuse(
                "task_graph-region-structure",
                format!(
                    "template {t}: block {} ends the wave inside the region",
                    block.id.0
                ),
            ));
        }
    }
    for pos in p.p_entry..p.p_exit.min(body.layout.len()) {
        let Some(inst) = body.insts.get(body.layout[pos]) else {
            continue;
        };
        let name = mnemonic(kernel, arch, pos);
        let text = format!("{name} @{:#x}", rel_pc(kernel, inst.prov.pc).unwrap_or(0));
        // (a write of the dispatcher's SGPR range is the protocol check's business: it is
        // admitted only when parked and restored, `protocol::enter_region`)
        for r in inst.effects.defs.iter().chain(&inst.effects.uses) {
            if r.kind == Kind::Ttmp {
                out.push(refuse(
                    "task_graph-region-structure",
                    format!("template {t}: `{text}` names a trap temporary"),
                ));
            }
        }
        if matches!(
            inst.effects.control,
            Control::EndPgm | Control::Halt | Control::Trap
        ) || name.starts_with("s_setreg")
            || matches!(
                name,
                "s_setpc_b64"
                    | "s_swappc_b64"
                    | "s_getpc_b64"
                    | "s_rfe_b64"
                    | "s_round_mode"
                    | "s_denorm_mode"
                    | "s_sethalt"
                    | "s_trap"
            )
        {
            out.push(refuse(
                "task_graph-region-structure",
                format!("template {t}: `{text}` ends the wave, traps or changes MODE"),
            ));
        }
        if p.region.is_native() {
            // (a barrier is admitted by `derive_native` only with its participation proof)
            let lds = inst.effects.lds.is_some()
                || matches!(
                    inst.effects.mem.map(|m| m.class),
                    Some(
                        peacemaker_ir::effects::MemClass::DsLoad
                            | peacemaker_ir::effects::MemClass::DsStore
                            | peacemaker_ir::effects::MemClass::DsAtomic { .. }
                            | peacemaker_ir::effects::MemClass::LdsDma
                    )
                );
            if lds {
                out.push(refuse(
                    "task_graph-region-native",
                    format!("template {t}: native region `{text}` uses LDS, which its declared geometry does not cover"),
                ));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// region arena facts
// ---------------------------------------------------------------------------

/// What one region contributes to the arena and the receipt.
#[derive(Clone, Debug)]
pub struct RegionFacts {
    pub template: ArenaTemplateRecord,
    pub lds_bytes: u32,
    pub evidence: String,
}

fn pointer_loads(facts: &KernargFacts) -> Vec<ArenaPointerRecord> {
    let mut out: Vec<ArenaPointerRecord> = facts
        .args
        .iter()
        .filter(|a| !a.accesses.is_empty())
        .map(|a| {
            let r = arg_record(a);
            ArenaPointerRecord {
                record_offset: r.offset,
                classes: r.classes,
            }
        })
        .collect();
    out.sort_by_key(|p| p.record_offset);
    out
}

/// The region `[p_entry, p_exit)` of `kernel` cut out as a kernel of its own:
/// the region's blocks (clipped at the region), `s[0:1]` the kernarg-segment
/// pointer of the entry state (the record base), no declared arguments and no
/// kernarg size, so every pointer loaded from the record is a synthetic slot
/// at its record offset.
fn region_kernel(kernel: &Kernel, p: &Placed) -> Result<Kernel, String> {
    let body = &kernel.body;
    let mut map: HashMap<usize, usize> = HashMap::new();
    let mut blocks = Vec::new();
    for b in &body.blocks {
        let (s, e) = (b.range.0.max(p.p_entry), b.range.1.min(p.p_exit));
        if s < e {
            map.insert(b.id.0, blocks.len());
            blocks.push(b);
        }
    }
    if blocks.is_empty() {
        return Err("no block overlaps the region".into());
    }
    let first = blocks[0];
    if first.range.0.max(p.p_entry) != p.p_entry {
        return Err("the region does not start its first block".into());
    }
    let mut out_blocks = Vec::with_capacity(blocks.len());
    for (i, b) in blocks.iter().enumerate() {
        let (s, e) = (b.range.0.max(p.p_entry), b.range.1.min(p.p_exit));
        let clipped_end = b.range.1 > p.p_exit;
        let remap = |id: &BlockId| map.get(&id.0).map(|&n| BlockId(n));
        let succs: smallvec::SmallVec<[BlockId; 2]> = b.succs.iter().filter_map(remap).collect();
        let preds: smallvec::SmallVec<[BlockId; 2]> = b.preds.iter().filter_map(remap).collect();
        let term = match (&b.term, clipped_end) {
            (_, true) => Terminator::EndPgm,
            (Terminator::Jump(t), false) => match remap(t) {
                Some(t) => Terminator::Jump(t),
                None => Terminator::EndPgm,
            },
            (
                Terminator::Branch {
                    cond,
                    taken,
                    fallthrough,
                },
                false,
            ) => match (remap(taken), remap(fallthrough)) {
                (Some(taken), Some(fallthrough)) => Terminator::Branch {
                    cond: *cond,
                    taken,
                    fallthrough,
                },
                _ => return Err("a conditional branch leaves the region".into()),
            },
            (Terminator::FallThrough, false) => match succs.first() {
                Some(&n) => {
                    let _ = n;
                    Terminator::FallThrough
                }
                None => Terminator::EndPgm,
            },
            (t, false) => t.clone(),
        };
        out_blocks.push(peacemaker_ir::cfg::Block {
            id: BlockId(i),
            range: (s - p.p_entry, e - p.p_entry),
            term,
            preds,
            succs,
        });
    }
    let mut abi = kernel.abi.clone();
    if let Abi::Hsa {
        descriptor,
        metadata,
    } = &mut abi
    {
        metadata.parsed.args.clear();
        // a record is not bounded by the pool's own kernarg size: every dword of the
        // analysed window is the region's
        metadata.parsed.kernarg_segment_size = 512;
        descriptor.kernarg_size = 512;
    }
    Ok(Kernel {
        symbol: kernel.symbol.clone(),
        wave: kernel.wave,
        abi,
        body: Body {
            insts: body.insts.clone(),
            layout: body.layout[p.p_entry..p.p_exit].to_vec(),
            blocks: out_blocks,
        },
        origin: kernel.origin.clone(),
    })
}

/// The `task_graph-a1-budget` refusal when an A1 analysis (`what`) stopped at its work bound:
/// `unresolved` are its `(rule, detail)` reasons.
fn a1_budget<'a>(
    what: &str,
    mut unresolved: impl Iterator<Item = (&'a str, &'a str)>,
) -> Option<Refusal> {
    unresolved
        .find(|(rule, _)| *rule == kernargs::RULE_FIXPOINT_BUDGET)
        .map(|(rule, detail)| {
            refuse(
                "task_graph-a1-budget",
                format!(
                    "{what} ran past its work bound ({rule}: {detail}); the kernel stays unknown"
                ),
            )
        })
}

/// Region-scoped A1: the region's own instructions, `s[0:1]` the record base.
fn region_scoped(
    kernel: &Kernel,
    arch: Arch,
    p: &Placed,
    budget: &PoolBudget,
) -> Result<KernargFacts, Refusal> {
    let t = p.region.template_id;
    let sub = region_kernel(kernel, p).map_err(|e| {
        refuse(
            "task_graph-region-native",
            format!("template {t}: region cannot be cut out: {e}"),
        )
    })?;
    let facts = kernargs::analyze_budgeted(&sub, arch, &budget.a1);
    if facts.status != KernelStatus::Proven {
        if let Some(r) = a1_budget(
            &format!("template {t}: region-scoped A1"),
            facts.unresolved.iter().map(|u| (u.rule, u.detail.as_str())),
        ) {
            return Err(r);
        }
        let why: Vec<String> = facts
            .unresolved
            .iter()
            .take(4)
            .map(|u| format!("{}: {}", u.rule, u.detail))
            .collect();
        return Err(refuse(
            "task_graph-region-native",
            format!(
                "template {t}: region-scoped A1 is {:?}: {}",
                facts.status,
                why.join("; ")
            ),
        ));
    }
    Ok(facts)
}

/// What one standalone kernel's receipt record and A1 facts are, derived once.
struct BodyCert {
    record: KernelRecord,
    /// A1 as `kernargs::analyze` returned it (the key of the workgroup-footprint analysis).
    raw: KernargFacts,
}

/// One standalone object of a pool certification: looked up and hashed once, lifted once,
/// and each kernel it hosts analysed once, however many regions (templates) name it. Every
/// cached value is a function of the object's bytes (the map key is their SHA-256, checked
/// when the entry is made), the kernel symbol, the region's block size (footprints) and the
/// work bounds, which are fixed for the lifetime of the [`Sources`] that owns it.
struct StandaloneObject {
    sha256_hex: String,
    lifted: Result<Lifted<Program>, String>,
    source: String,
    cert: Option<radiowave::CodeObjectCertification>,
    bodies: HashMap<String, BodyCert>,
    footprints: HashMap<(String, u32), (Vec<WgFootprintRecord>, Option<String>)>,
}

impl StandaloneObject {
    /// The record and facts of kernel `symbol` (the body half of [`crate::certify_kernel`],
    /// run at most once per symbol).
    fn body(&mut self, symbol: &str, a1: &A1Budget) -> Option<&BodyCert> {
        if !self.bodies.contains_key(symbol) {
            let lifted = self.lifted.as_ref().ok()?;
            let index = lifted
                .program
                .kernels
                .iter()
                .position(|k| k.symbol.0 == symbol)?;
            let c = crate::certify_kernel(
                lifted,
                index,
                &self.sha256_hex,
                &self.source,
                self.cert.as_ref(),
                crate::KernelDetail::Body,
                a1,
            );
            self.bodies.insert(
                symbol.to_owned(),
                BodyCert {
                    record: c.record,
                    raw: c.raw,
                },
            );
        }
        self.bodies.get(symbol)
    }

    /// The workgroup footprints of kernel `symbol`'s pointer arguments for a region of
    /// `block` threads (`record_offset` = the standalone kernarg offset = the record
    /// offset), and why none were computed when the analysis did not apply. A body that
    /// cannot be re-analysed has none: an absent footprint is never a claim.
    fn footprints(
        &mut self,
        symbol: &str,
        block: u32,
        a1: &A1Budget,
    ) -> (Vec<WgFootprintRecord>, Option<String>) {
        let key = (symbol.to_owned(), block);
        if let Some(f) = self.footprints.get(&key) {
            return f.clone();
        }
        let out = (|| {
            let raw = self.bodies.get(symbol)?.raw.clone();
            let program = &self.lifted.as_ref().ok()?.program;
            let kernel = program.kernels.iter().find(|k| k.symbol.0 == symbol)?;
            Some(crate::wg_footprint_outcome(
                kernel,
                program.target.arch,
                &raw,
                Some(block),
                a1,
            ))
        })()
        .unwrap_or_default();
        self.footprints.insert(key, out.clone());
        out
    }
}

/// The standalone objects of one [`certify_pool_with`] call, by SHA-256.
struct Sources<'a> {
    lookup: Standalone<'a>,
    budget: PoolBudget,
    objects: HashMap<[u8; 32], Result<StandaloneObject, Refusal>>,
}

impl<'a> Sources<'a> {
    fn new(lookup: Standalone<'a>, budget: PoolBudget) -> Self {
        Sources {
            lookup,
            budget,
            objects: HashMap::new(),
        }
    }

    fn load(&self, digest: &[u8; 32]) -> Result<StandaloneObject, Refusal> {
        let bytes = (self.lookup)(digest).ok_or_else(|| {
            refuse(
                "task_graph-region-source-missing",
                format!("no standalone object with SHA-256 {}", hex32(digest)),
            )
        })?;
        let actual: [u8; 32] = Sha256::digest(&bytes).into();
        if &actual != digest {
            return Err(refuse(
                "task_graph-region-source-digest",
                format!(
                    "the object offered for {} hashes to {}",
                    hex32(digest),
                    hex32(&actual)
                ),
            ));
        }
        let lifted = crate::timing::timed("lift standalone (once per digest)", || {
            lift_object(
                &bytes,
                Options {
                    frontend: Frontend::Hipcc,
                },
            )
        })
        .map_err(|e| e.to_string());
        let (source, cert) = match &lifted {
            Ok(l) => radiowave_cert(&bytes, &l.program, None),
            Err(_) => (String::new(), None),
        };
        Ok(StandaloneObject {
            sha256_hex: hex32(digest),
            lifted,
            source,
            cert,
            bodies: HashMap::new(),
            footprints: HashMap::new(),
        })
    }

    /// The object with SHA-256 `digest`, loaded on first use.
    fn object(&mut self, digest: &[u8; 32]) -> Result<&mut StandaloneObject, Refusal> {
        if !self.objects.contains_key(digest) {
            let loaded = self.load(digest);
            self.objects.insert(*digest, loaded);
        }
        match self.objects.get_mut(digest) {
            Some(Ok(o)) => Ok(o),
            Some(Err(r)) => Err(r.clone()),
            None => unreachable!("inserted above"),
        }
    }
}

fn hex32(d: &[u8; 32]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// Item 1 and 2 of one lifted region.
fn derive_lifted(
    kernel: &Kernel,
    arch: Arch,
    pool_text: &[u8],
    pool_waves: u32,
    pool_rsrc1_modes: Option<u32>,
    p: &Placed,
    sources: &mut Sources<'_>,
) -> Result<RegionFacts, Refusal> {
    let r = &p.region;
    let t = r.template_id;
    let budget = sources.budget;
    let object = crate::timing::timed("read + sha256 standalone", || sources.object(&r.digest))?;
    let lifted = {
        let program = match &object.lifted {
            Ok(l) => &l.program,
            Err(e) => {
                return Err(refuse(
                    "task_graph-region-lift",
                    format!("template {t}: lift: {e}"),
                ))
            }
        };
        crate::timing::timed("lift_region (convert)", || {
            lift_region_in(
                program,
                r.digest,
                &r.source_symbol,
                &RegionInputs {
                    prefix: r.prefix.clone(),
                    workgroup_x: r.wg_x,
                    workgroup_yz: r.wg_yz,
                    block: r.block,
                    dynamic_lds: r.dynamic_lds,
                },
            )
        })
        .map_err(|e| {
            refuse(
                "task_graph-region-lift",
                format!("template {t}: {}", e.reason),
            )
        })?
    };
    if lifted.receipt.waves != pool_waves {
        return Err(refuse(
            "task_graph-region-geometry",
            format!(
                "template {t}: the body runs {} waves, the pool kernel {pool_waves}",
                lifted.receipt.waves
            ),
        ));
    }
    if let Some(modes) = pool_rsrc1_modes {
        if modes != lifted.receipt.rsrc1_modes {
            return Err(refuse(
                "task_graph-region-rsrc1",
                format!(
                    "template {t}: descriptor RSRC1 modes {modes:#x} differ from the body's {:#x}",
                    lifted.receipt.rsrc1_modes
                ),
            ));
        }
    }
    let words = crate::timing::timed("region_words (assemble)", || {
        region_words(&lifted.lines, arch)
    })
    .map_err(|e| {
        refuse(
            "task_graph-region-lift",
            format!("template {t}: assemble: {e}"),
        )
    })?;
    let have = pool_text
        .get(r.entry_pc as usize..r.exit_pc as usize)
        .ok_or_else(|| {
            refuse(
                "task_graph-region-bytes",
                format!("template {t}: range outside the kernel"),
            )
        })?;
    let want: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    if have != &want[..] {
        let at = have
            .chunks(4)
            .zip(want.chunks(4))
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| have.len().min(want.len()) / 4);
        return Err(refuse(
            "task_graph-region-bytes",
            format!(
                "template {t}: the pool's bytes [{:#x}, {:#x}) ({} words) differ from the body re-derived from {} ({} words), first at word {at} (pc {:#x})",
                r.entry_pc,
                r.exit_pc,
                have.len() / 4,
                r.source_symbol,
                want.len() / 4,
                r.entry_pc + 4 * at as u64
            ),
        ));
    }
    // item 2: the standalone body's own A1 record (its size is bounded by the pool
    // kernel's: the body's words are the region's, compared equal above)
    let record = crate::timing::timed("standalone body A1 record (once per kernel)", || {
        object
            .body(&r.source_symbol, &budget.a1)
            .map(|b| b.record.clone())
    })
    .ok_or_else(|| {
        refuse(
            "task_graph-region-arena",
            format!(
                "template {t}: {} is not in the standalone object",
                r.source_symbol
            ),
        )
    })?;
    let record = &record;
    if let Some(refusal) = a1_budget(
        &format!("template {t}: the standalone body {}'s A1", r.source_symbol),
        record
            .unresolved
            .iter()
            .map(|u| (u.rule.as_str(), u.detail.as_str())),
    ) {
        return Err(refusal);
    }
    let (loads, sealed_tables) = template_from_standalone(t, record)?;
    if record.dynamic_kernarg_reads
        || record.taint_overflow
        || record.side.module_global != 0
        || record.side.abi_packet != 0
        || record.side.scratch != 0
        || record.side.kernarg_segment != 0
    {
        return Err(refuse(
            "task_graph-region-arena",
            format!("template {t}: the body reads outside its own record (side reads or dynamic kernarg reads)"),
        ));
    }
    // The same facts from the pool's own bytes (region-scoped A1) are reported, not
    // required: the bytes are proven identical to the standalone body, whose own
    // proven A1 record is the source of the template.
    let cross = match crate::timing::timed("region-scoped A1 cross-check", || {
        region_scoped(kernel, arch, p, &budget)
    }) {
        Ok(s) if pointer_loads(&s) == loads => "agrees".to_owned(),
        Ok(s) => format!("differs ({:?})", pointer_loads(&s)),
        Err(r) => format!("not analysable ({})", r.rule),
    };
    let hidden: Vec<String> = record
        .implicit
        .hidden_loaded
        .iter()
        .map(|k| {
            let off = record
                .args
                .iter()
                .find(|a| a.value_kind == *k && a.loaded)
                .map(|a| (a.offset, a.size));
            match off {
                Some((o, s)) => format!("{k}@{o}+{s}"),
                None => k.clone(),
            }
        })
        .collect();
    for h in record
        .args
        .iter()
        .filter(|a| a.value_kind.starts_with("hidden_") && a.loaded)
    {
        if h.offset + h.size > record.kernarg_size {
            return Err(refuse(
                "task_graph-region-arena",
                format!(
                    "template {t}: hidden argument {} lies outside the {}-byte record",
                    h.value_kind, record.kernarg_size
                ),
            ));
        }
    }
    let sealed_note = if sealed_tables.is_empty() {
        String::new()
    } else {
        format!(
            "; sealed tables [{}]",
            sealed_tables
                .iter()
                .map(|s| format!(
                    "@{} bound {}={}",
                    s.table_arg_offset, s.bound_kind, s.bound_value
                ))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let (wg_footprints, wg_why) =
        crate::timing::timed("standalone footprints (once per kernel and block)", || {
            object.footprints(&r.source_symbol, r.block, &budget.a1)
        });
    let wg_note = match wg_why {
        Some(why) if why.contains(kernargs::RULE_FIXPOINT_BUDGET) => {
            format!("; workgroup footprints not computed ({why})")
        }
        _ => String::new(),
    };
    Ok(RegionFacts {
        template: ArenaTemplateRecord {
            template: t,
            pointer_loads: loads,
            sealed_tables,
            wg_footprints,
            hidden_grid: hidden_grid_from_standalone(record),
        },
        lds_bytes: lifted.lds_bytes,
        evidence: format!(
            "t{t}: {} lifted from {} ({}) = pool bytes [{:#x},{:#x}); record {}B, hidden loaded [{}]; region-scoped A1 cross-check of the pool's bytes: {cross} (informational: the standalone record is the proof)",
            r.source_symbol,
            hex32(&r.digest).get(..12).unwrap_or(""),
            record.status,
            r.entry_pc,
            r.exit_pc,
            record.kernarg_size,
            hidden.join(", ")
        ) + &sealed_note
            + &wg_note,
    })
}

/// The loaded hidden grid arguments of a standalone body (`hidden_block_count_*`,
/// `hidden_group_size_*`, `hidden_remainder_*`, `hidden_grid_dims`), at their standalone
/// kernarg offsets: inside the pool region the record is the body's kernarg block, so
/// the host patches/checks exactly these words. `kind` is the value kind without `hidden_`.
fn hidden_grid_from_standalone(record: &KernelRecord) -> Vec<HiddenGridRecord> {
    let mut out: Vec<HiddenGridRecord> = record
        .args
        .iter()
        .filter(|a| {
            a.loaded && peacemaker_ir::kernarg::HIDDEN_GRID_KINDS.contains(&a.value_kind.as_str())
        })
        .map(|a| HiddenGridRecord {
            record_offset: a.offset,
            kind: a
                .value_kind
                .strip_prefix("hidden_")
                .unwrap_or(&a.value_kind)
                .to_owned(),
        })
        .collect();
    out.sort_by_key(|h| h.record_offset);
    out
}

/// The template facts of a standalone body's A1 record: its pointer loads and
/// sealed tables. A proven record yields no sealed tables; a conditional record is
/// admitted only when its sole condition is sealed tables (see the rules below).
fn template_from_standalone(
    t: u32,
    record: &KernelRecord,
) -> Result<(Vec<ArenaPointerRecord>, Vec<SealedTableRecord>), Refusal> {
    let no = |why: String| {
        refuse(
            "task_graph-region-arena",
            format!(
                "template {t}: the standalone body's A1 record is {}, not admitted: {why}",
                record.status
            ),
        )
    };
    let sealed_only = match record.status.as_str() {
        STATUS_PROVEN => false,
        STATUS_CONDITIONAL => {
            if !record.indexed_effects.is_empty() {
                return Err(no("it has indexed effects (indexed_bind)".into()));
            }
            if record.sealed_tables.is_empty() {
                return Err(no("it is conditional without sealed tables".into()));
            }
            if !record.guards.is_empty()
                || !record.descriptor_arenas.is_empty()
                || record.task_protocol.is_some()
            {
                return Err(no(
                    "it has guards, descriptor arenas or a task protocol".into()
                ));
            }
            if record.required_witnesses != [WITNESS_SEALED_TABLE] {
                return Err(no(format!(
                    "its witnesses are {:?}, not exactly [sealed_table]",
                    record.required_witnesses
                )));
            }
            true
        }
        _ => return Err(no("it is not proven".into())),
    };
    let mut sealed: Vec<SealedTableRecord> = Vec::new();
    if sealed_only {
        for s in &record.sealed_tables {
            s.validate(&record.symbol).map_err(no)?;
            let arg = record
                .args
                .iter()
                .find(|a| a.offset == s.table_arg_offset)
                .filter(|a| a.role == "pointer" && a.size == 8)
                .ok_or_else(|| {
                    no(format!(
                        "sealed table arg +{} is not an 8-byte pointer argument",
                        s.table_arg_offset
                    ))
                })?;
            if arg
                .classes
                .iter()
                .any(|c| !matches!(c.as_str(), "smem_load" | "vmem_load"))
                || matches!(arg.mode.as_deref(), Some("write" | "read_write"))
            {
                return Err(no(format!(
                    "sealed table arg +{} is written or accessed as {:?}",
                    s.table_arg_offset, arg.classes
                )));
            }
            if s.bound_kind == SEALED_BOUND_KERNARG {
                let b = s.bound_value;
                if b % 4 != 0 || b + 4 > u64::from(record.kernarg_size) {
                    return Err(no(format!(
                        "kernarg bound +{b} is not a 4-aligned scalar inside the {}-byte record",
                        record.kernarg_size
                    )));
                }
                if record.args.iter().any(|a| {
                    a.role == "pointer"
                        && b < u64::from(a.offset) + u64::from(a.size)
                        && u64::from(a.offset) < b + 4
                }) {
                    return Err(no(format!(
                        "kernarg bound +{b} overlaps a pointer argument"
                    )));
                }
            }
            sealed.push(s.clone());
        }
        sealed.sort_by(|a, b| {
            (a.table_arg_offset, &a.bound_kind, a.bound_value).cmp(&(
                b.table_arg_offset,
                &b.bound_kind,
                b.bound_value,
            ))
        });
    }
    let mut loads: Vec<ArenaPointerRecord> = record
        .args
        .iter()
        .filter(|a| a.role == "pointer" && a.accesses > 0)
        .map(|a| ArenaPointerRecord {
            record_offset: a.offset,
            classes: a.classes.clone(),
        })
        .collect();
    for s in &sealed {
        if !loads.iter().any(|l| l.record_offset == s.table_arg_offset) {
            let mut classes: Vec<String> = s.entries.iter().map(|e| e.class.clone()).collect();
            classes.sort();
            classes.dedup();
            loads.push(ArenaPointerRecord {
                record_offset: s.table_arg_offset,
                classes,
            });
        }
    }
    loads.sort_by_key(|p| p.record_offset);
    Ok((loads, sealed))
}

fn derive_native(
    kernel: &Kernel,
    arch: Arch,
    p: &Placed,
    block: u32,
    budget: &PoolBudget,
) -> Result<RegionFacts, Refusal> {
    let t = p.region.template_id;
    let scoped = crate::timing::timed("region-scoped A1", || {
        region_scoped(kernel, arch, p, budget)
    })?;
    let loads = pointer_loads(&scoped);
    if loads.is_empty() {
        return Err(refuse(
            "task_graph-region-native",
            format!("template {t}: the native region dereferences no record pointer"),
        ));
    }
    // A barrier in a native region must be executed by every wave of the pool's workgroup
    // the same number of times (a wave that skipped it would pair its next dispatcher
    // barrier with another wave's): the lifted bodies' barrier-participation proof, over
    // the region cut out of the pool.
    let barriers = (p.p_entry..p.p_exit.min(kernel.body.layout.len()))
        .filter(|&pos| {
            kernel
                .body
                .insts
                .get(kernel.body.layout[pos])
                .is_some_and(|i| matches!(i.effects.control, Control::Barrier(_)))
        })
        .count();
    let mut participation = String::new();
    if barriers > 0 {
        let sub = region_kernel(kernel, p).map_err(|e| {
            refuse(
                "task_graph-region-native",
                format!("template {t}: region cannot be cut out: {e}"),
            )
        })?;
        let (uniform, divergent, verdict) =
            peacemaker_lift::rewrite::uniformity::barrier_participation(&sub.body, block)
                .map_err(|reason| {
                    refuse(
                        "task_graph-region-native",
                        format!("template {t}: native region barrier is not reached by every wave: {reason}"),
                    )
                })?;
        participation = format!(
            "; {barriers} barrier instruction(s) reached by every wave of the {block}-thread workgroup ({uniform} uniform / {divergent} per-wave branch nodes: {verdict})"
        );
    }
    let access = loads
        .iter()
        .map(|l| format!("+{} {}", l.record_offset, l.classes.join("|")))
        .collect::<Vec<_>>()
        .join(", ");
    Ok(RegionFacts {
        template: ArenaTemplateRecord {
            template: t,
            pointer_loads: loads,
            sealed_tables: Vec::new(),
            wg_footprints: Vec::new(),
            // A native region's cut-out kernel has no declared arguments, so no record
            // dword is a hidden grid value; the grid it reads (COPY: the dispatcher's s72 =
            // header gx) is tied by `region_wg_from_header`, not by a record offset.
            hidden_grid: Vec::new(),
        },
        lds_bytes: 0,
        evidence: format!(
            "t{t}: native region [{:#x},{:#x}) analysed by region-scoped A1 (s[0:1] = record base): record pointer fields {access}{participation}",
            p.region.entry_pc, p.region.exit_pc
        ),
    })
}

// ---------------------------------------------------------------------------
// one pool kernel
// ---------------------------------------------------------------------------

/// Run every obligation for one tagged kernel under the default [`PoolBudget`]; all refusals
/// are returned.
pub fn prove_pool(
    program: &Program,
    kernel_index: usize,
    pool_elf: &[u8],
    notes: &KernelRegions,
    standalone: Standalone<'_>,
    partial: &mut Option<PoolProof>,
) -> Result<PoolProof, Vec<Refusal>> {
    prove_pool_budgeted(
        program,
        kernel_index,
        pool_elf,
        notes,
        standalone,
        &PoolBudget::default(),
        partial,
    )
}

/// [`prove_pool`] under an explicit [`PoolBudget`].
pub fn prove_pool_budgeted(
    program: &Program,
    kernel_index: usize,
    pool_elf: &[u8],
    notes: &KernelRegions,
    standalone: Standalone<'_>,
    budget: &PoolBudget,
    partial: &mut Option<PoolProof>,
) -> Result<PoolProof, Vec<Refusal>> {
    let mut sources = Sources::new(standalone, *budget);
    prove_pool_in(
        program,
        kernel_index,
        pool_elf,
        notes,
        &mut sources,
        partial,
    )
    .map(|(p, _)| p)
}

/// [`prove_pool`] over the standalone objects `sources` already holds, also returning the
/// dispatcher shell's A1 facts (the record of a proven pool is built from them).
fn prove_pool_in(
    program: &Program,
    kernel_index: usize,
    pool_elf: &[u8],
    notes: &KernelRegions,
    sources: &mut Sources<'_>,
    partial: &mut Option<PoolProof>,
) -> Result<(PoolProof, KernargFacts), Vec<Refusal>> {
    let budget = sources.budget;
    let kernel = &program.kernels[kernel_index];
    let arch = program.target.arch;
    let mut refusals = Vec::new();
    if arch != Arch::Gfx1201 {
        return Err(vec![refuse(
            "task_graph-arch",
            format!("{arch:?}: persistent pools are gfx1201 only"),
        )]);
    }
    let Abi::Hsa {
        descriptor,
        metadata,
    } = &kernel.abi
    else {
        return Err(vec![refuse("task_graph-abi", "not an HSA kernel")]);
    };
    let block = metadata.parsed.max_flat_workgroup_size;
    if block == 0 || block % 32 != 0 || block > 1024 {
        return Err(vec![refuse(
            "task_graph-geometry",
            format!("workgroup size {block} is not a whole number of waves"),
        )]);
    }
    let waves = block / 32;
    let rsrc1_modes = descriptor.compute_pgm_rsrc1.0 & !(0x3f | (0xf << 6));
    if notes.regions.len() > budget.max_regions {
        return Err(vec![refuse(
            "task_graph-size-budget",
            format!(
                "the note lists {} regions, past the bound of {}",
                notes.regions.len(),
                budget.max_regions
            ),
        )]);
    }
    if kernel.body.layout.len() > budget.max_kernel_insts {
        return Err(vec![refuse(
            "task_graph-size-budget",
            format!(
                "the pool kernel has {} instructions, past the bound of {}",
                kernel.body.layout.len(),
                budget.max_kernel_insts
            ),
        )]);
    }
    let placed = match place_regions(kernel, notes) {
        Ok(p) => p,
        Err(r) => return Err(vec![r]),
    };
    let text = match kernel_text(pool_elf, &kernel.symbol.0) {
        Ok(t) => t,
        Err(e) => return Err(vec![refuse("task_graph-note", e)]),
    };
    // 1+2: regions
    let mut facts = Vec::new();
    for p in &placed {
        let label = format!("region t{} {}", p.region.template_id, {
            if p.region.is_native() {
                "(native)"
            } else {
                p.region.source_symbol.as_str()
            }
        });
        crate::timing::timed(&label, || {
            let before = refusals.len();
            refusals.extend(crate::timing::timed("check_region_structure", || {
                check_region_structure(kernel, arch, p)
            }));
            if refusals.len() != before {
                return;
            }
            let result = if p.region.is_native() {
                derive_native(kernel, arch, p, block, &budget)
            } else {
                derive_lifted(kernel, arch, text, waves, Some(rsrc1_modes), p, sources)
            };
            match result {
                Ok(f) => facts.push(f),
                Err(r) => refusals.push(r),
            }
        });
    }
    // 3: the dispatcher shell
    let base = text_base(kernel);
    let ranges: Vec<(u32, u32)> = placed
        .iter()
        .map(|p| {
            (
                p.region.entry_pc as u32 + base,
                p.region.exit_pc as u32 + base,
            )
        })
        .collect();
    let mut shell = crate::timing::timed("shell A1 (analyze_excluding)", || {
        kernargs::analyze_excluding_budgeted(kernel, arch, &ranges, &budget.a1)
    });
    let mut shell_refusal: Option<Refusal> = None;
    if shell.status != KernelStatus::Proven {
        if let Some(r) = a1_budget(
            "the dispatcher shell's A1",
            shell.unresolved.iter().map(|u| (u.rule, u.detail.as_str())),
        ) {
            // the protocol check reads the shell's facts: there are none to read
            refusals.push(r);
            return Err(refusals);
        }
        // refused unless the protocol proves span mode and attributes every unresolved
        // access to the span table's arena (below)
        shell_refusal = Some(refuse(
            "task_graph-shell-unresolved",
            format!(
                "the dispatcher shell's A1 is {:?}: {}",
                shell.status,
                shell
                    .unresolved
                    .iter()
                    .take(4)
                    .map(|u| format!("{}: {}", u.rule, u.detail))
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
        ));
    }
    // 4: the protocol (also derives which pointer argument plays which role)
    let proto = match crate::timing::timed("protocol::check", || {
        protocol::check(
            kernel,
            arch,
            notes,
            &placed,
            waves,
            &shell,
            &facts,
            &budget.protocol,
        )
    }) {
        Ok(p) => p,
        Err(rs) => {
            refusals.extend(shell_refusal);
            refusals.extend(rs);
            return Err(refusals);
        }
    };
    if let Some(r) = shell_refusal {
        let attributed = proto
            .span
            .as_ref()
            .is_some_and(|s| attribute_span_shell(&mut shell, s));
        if !attributed {
            refusals.push(r);
        }
    }
    // LDS: the dispatcher's claim slot lies above every body's LDS
    if let Some(slot) = proto.lds_slot_base {
        let top = facts.iter().map(|f| f.lds_bytes).max().unwrap_or(0);
        if slot < top {
            refusals.push(refuse(
                "task_graph-lds",
                format!("the claim slot at LDS {slot} overlaps a body's LDS [0, {top})"),
            ));
        }
        if descriptor.group_segment_fixed_size < slot + 64 {
            refusals.push(refuse(
                "task_graph-lds",
                format!(
                    "the claim slot [{slot}, {}) exceeds the {}-byte LDS allocation",
                    slot + 64,
                    descriptor.group_segment_fixed_size
                ),
            ));
        }
    }
    let mut templates: Vec<ArenaTemplateRecord> =
        facts.iter().map(|f| f.template.clone()).collect();
    templates.sort_by_key(|t| t.template);
    let mut evidence: Vec<String> = facts.iter().map(|f| f.evidence.clone()).collect();
    evidence.extend(proto.evidence.iter().cloned());
    let span_arena = proto.span.as_ref().map(span_arena);
    let proof = PoolProof {
        arena: DescriptorArenaRecord {
            arena_arg_offset: proto.kernargs_arg_offset,
            templates,
        },
        span_arena,
        protocol: proto.record.clone(),
        evidence,
    };
    if let Err(rs) = check_shell_args(&shell, &proof, &proto) {
        refusals.extend(rs);
    }
    // the poison contract: only a monotonic update (set-only OR with release) may
    // write the poison cell; a plain store or a swap is refused
    for (pc, off) in &proto.poison_plain_stores {
        refusals.push(Refusal {
            rule: "task_graph-poison-plain-store",
            detail: format!(
                "a plain 32-bit store through params +{off}: the watchdog poison must be a set-only atomic OR with release, not a store"
            ),
            pc: *pc,
        });
    }
    if refusals.is_empty() {
        Ok((proof, shell))
    } else {
        *partial = Some(proof);
        Err(refusals)
    }
}

/// The span table's descriptor records (the arena fact of the `spans` argument).
pub const SPAN_DESC_TEMPLATE: u32 = 0xFFFF_FFFE;
/// The span table's header record (`n_spans`): no pointer is loaded from it.
pub const SPAN_HEADER_TEMPLATE: u32 = 0xFFFF_FFFD;

/// The dispatcher shell's accesses through span-table-loaded pointers (the live list, the
/// count cell, the claim bitmap) that A1 cannot resolve itself (`base-memory-derived`) are
/// exactly the sites the protocol check attributed to descriptor fields of the span table;
/// they are then the arena's pointer loads and the shell is attributed. False when the shell
/// has any other unresolved access.
fn attribute_span_shell(shell: &mut KernargFacts, span: &protocol::SpanProof) -> bool {
    let sites: BTreeSet<peacemaker_ir::InstId> = span.sites.iter().map(|s| s.inst).collect();
    let covered = !shell.unresolved.is_empty()
        && shell
            .unresolved
            .iter()
            .all(|u| u.rule == "base-memory-derived" && u.inst.is_some_and(|i| sites.contains(&i)));
    if covered {
        shell.unresolved.clear();
        shell.status = KernelStatus::Proven;
    }
    covered
}

/// The `spans` argument as a binding arena: a descriptor record (template
/// [`SPAN_DESC_TEMPLATE`]) holds the pointers at +0 (live list), +8 (count cell), +16 (claim
/// bitmap) with the classes the dispatcher accesses them with, and the table header record
/// ([`SPAN_HEADER_TEMPLATE`], `n_spans`) holds none.
fn span_arena(span: &protocol::SpanProof) -> DescriptorArenaRecord {
    let o = &span.offsets;
    let loads = [o.live_list_field, o.count_cell_field, o.bitmap_field]
        .into_iter()
        .map(|field| {
            let mut classes: Vec<String> = span
                .sites
                .iter()
                .filter(|s| s.field == field)
                .map(|s| s.class.to_owned())
                .collect();
            classes.sort();
            classes.dedup();
            ArenaPointerRecord {
                record_offset: field,
                classes,
            }
        })
        .collect();
    let record = |template, pointer_loads| ArenaTemplateRecord {
        template,
        pointer_loads,
        sealed_tables: Vec::new(),
        wg_footprints: Vec::new(),
        hidden_grid: Vec::new(),
    };
    DescriptorArenaRecord {
        arena_arg_offset: o.spans_arg_offset,
        templates: vec![
            record(SPAN_HEADER_TEMPLATE, Vec::new()),
            record(SPAN_DESC_TEMPLATE, loads),
        ],
    }
}

/// The shell's argument accesses must be exactly the roles the protocol found.
fn check_shell_args(
    shell: &KernargFacts,
    proof: &PoolProof,
    proto: &protocol::ProtocolProof,
) -> Result<(), Vec<Refusal>> {
    let mut out = Vec::new();
    let t = &proof.protocol;
    for a in shell.args.iter().filter(|a| !a.accesses.is_empty()) {
        let r = arg_record(a);
        let classes: BTreeSet<&str> = r.classes.iter().map(String::as_str).collect();
        if proto.output_args.contains(&a.offset) {
            // trace and poison: stored to, or monotonically updated (set-only OR)
            if let Some(bad) = classes
                .iter()
                .find(|c| !matches!(**c, "vmem_store" | "atomic_release" | "atomic_commutative"))
            {
                out.push(refuse(
                    "task_graph-shell-access",
                    format!("output argument +{} is accessed as {bad}", a.offset),
                ));
            }
            continue;
        }
        let want: Option<&[&str]> = if a.offset == t.tasks_arg_offset {
            Some(&["vmem_load"])
        } else if a.offset == t.ticket_arg_offset {
            Some(&["atomic_commutative"])
        } else if a.offset == t.events_arg_offset {
            Some(&["atomic_release", "load_acquire"])
        } else if Some(a.offset) == proof.protocol.span.map(|s| s.spans_arg_offset) {
            // the span table: n_spans and descriptors are scalar-loaded (the pointers they
            // hold are the span arena's)
            Some(&["smem_load"])
        } else if a.offset == proof.arena.arena_arg_offset {
            None
        } else if proto.output_args.contains(&a.offset) {
            Some(&["vmem_store"])
        } else {
            out.push(refuse(
                "task_graph-shell-access",
                format!(
                    "the shell accesses argument +{} ({}) that is none of the protocol's roles",
                    a.offset, a.name
                ),
            ));
            continue;
        };
        if let Some(want) = want {
            let want: BTreeSet<&str> = want.iter().copied().collect();
            if classes != want {
                out.push(refuse(
                    "task_graph-shell-access",
                    format!(
                        "argument +{} ({}) is accessed as {classes:?}, the protocol needs exactly {want:?}",
                        a.offset, a.name
                    ),
                ));
            }
        } else if a.mode.is_some_and(|m| m.writes()) {
            out.push(refuse(
                "task_graph-shell-access",
                format!("a region writes the argument block +{}", a.offset),
            ));
        }
    }
    let side = &shell.side;
    if side.kernarg_segment != 0
        || side.module_global != 0
        || side.abi_packet != 0
        || side.scratch != 0
        || shell.dynamic_kernarg_reads
        || shell.taint_overflow
    {
        out.push(refuse(
            "task_graph-shell-access",
            "the shell reads the kernarg segment dynamically, a module global, the AQL packet or scratch",
        ));
    }
    if out.is_empty() {
        Ok(())
    } else {
        Err(out)
    }
}

// ---------------------------------------------------------------------------
// receipt assembly
// ---------------------------------------------------------------------------

/// [`crate::certify`] for a pool object: every kernel tagged by the object's
/// `.note.pm_regions` is upgraded to a conditional record carrying its
/// descriptor arena and task protocol when all four obligations (module docs)
/// are proven from the bytes; otherwise its A1 record is unchanged and gains
/// one `task_graph-*` unresolved rule per failed obligation. `standalone`
/// resolves a region's source object by its SHA-256. Runs under the default
/// [`PoolBudget`].
pub fn certify_pool(pool_elf: &[u8], standalone: Standalone<'_>) -> Result<Receipt, CertError> {
    certify_pool_with(pool_elf, standalone, &PoolBudget::default())
}

/// [`certify_pool`] under an explicit [`PoolBudget`]: a work bound that is exceeded refuses
/// (the kernel stays `unknown`, the refusal names the bound and the measured counts) and
/// never admits anything.
pub fn certify_pool_with(
    pool_elf: &[u8],
    standalone: Standalone<'_>,
    budget: &PoolBudget,
) -> Result<Receipt, CertError> {
    let lifted = crate::timing::timed("lift_object (pool)", || {
        lift_object(
            pool_elf,
            Options {
                frontend: Frontend::Hipcc,
            },
        )
    })
    .map_err(|e| CertError::Lift(e.to_string()))?;
    let mut receipt = crate::timing::timed("certify (whole pool object)", || {
        crate::certify_lifted(pool_elf, &lifted, None)
    });
    let notes =
        read_regions(pool_elf).map_err(|e| CertError::Lift(format!("regions note: {e}")))?;
    if notes.is_empty() {
        return Ok(receipt);
    }
    let program = &lifted.program;
    let (source, cert) = radiowave_cert(pool_elf, program, None);
    // one standalone cache for every tagged kernel of the object: an object that hosts
    // several templates is lifted and analysed once
    let mut sources = Sources::new(standalone, *budget);
    let mut details = Vec::new();
    for note in &notes {
        let Some(index) = program
            .kernels
            .iter()
            .position(|k| k.symbol.0 == note.kernel)
        else {
            return Err(CertError::Lift(format!(
                "regions note names {:?}, which is not a kernel of the object",
                note.kernel
            )));
        };
        let Some(slot) = receipt
            .railgun
            .kernels
            .iter()
            .position(|k| k.symbol == note.kernel)
        else {
            return Err(CertError::Lift(format!("{}: no record", note.kernel)));
        };
        let mut partial = None;
        match crate::timing::timed("prove_pool", || {
            prove_pool_in(program, index, pool_elf, note, &mut sources, &mut partial)
        }) {
            Ok((proof, shell)) => {
                let kernel = &program.kernels[index];
                let meta = metadata_effects(&source, cert.as_ref(), &kernel.symbol.0);
                let old = &receipt.railgun.kernels[slot];
                let mut record: KernelRecord = kernel_record(kernel, &shell, &meta);
                record.status = STATUS_CONDITIONAL.into();
                record.unguarded_unresolved = old.unresolved.clone();
                record.pm_obligations = old.pm_obligations.clone();
                record.pm_analyze_error = old.pm_analyze_error.clone();
                record.guard_flag = old.guard_flag;
                record.differential.guarded = true;
                record.differential.guards = Vec::new();
                record.descriptor_arenas = std::iter::once(proof.arena.clone())
                    .chain(proof.span_arena.clone())
                    .collect();
                record.task_protocol = Some(proof.protocol.clone());
                record.required_witnesses = record.implied_witnesses();
                let sealed = proof
                    .arena
                    .templates
                    .iter()
                    .any(|t| !t.sealed_tables.is_empty());
                debug_assert_eq!(
                    record.required_witnesses,
                    if sealed {
                        vec![WITNESS_DESCRIPTOR_ARENA, WITNESS_SEALED_TABLE]
                    } else {
                        vec![WITNESS_DESCRIPTOR_ARENA]
                    }
                );
                record.case_digest = case_digest(
                    &receipt.object_sha256,
                    &receipt.device_elf_sha256,
                    &receipt.arch,
                    &record,
                );
                details.push(format!(
                    "{}: conditional ({} templates; protocol {})",
                    note.kernel,
                    proof.arena.templates.len(),
                    proof.protocol.proved.join(",")
                ));
                details.extend(proof.evidence.iter().map(|e| format!("  {e}")));
                receipt.railgun.kernels[slot] = record;
            }
            Err(refusals) => {
                let record = &mut receipt.railgun.kernels[slot];
                details.push(format!(
                    "{}: unknown ({} refusals)",
                    note.kernel,
                    refusals.len()
                ));
                for r in refusals {
                    details.push(format!("  {}: {}", r.rule, r.detail));
                    record.unresolved.push(UnresolvedRecord {
                        pc: r.pc,
                        rule: r.rule.into(),
                        detail: r.detail,
                    });
                }
                record.status = STATUS_UNKNOWN.into();
            }
        }
    }
    let records = &receipt.railgun.kernels;
    let proven = records.iter().filter(|r| r.status == STATUS_PROVEN).count();
    let conditional = records
        .iter()
        .filter(|r| r.status == STATUS_CONDITIONAL)
        .count();
    let proofs = records.iter().filter(|r| r.task_protocol.is_some()).count();
    let total = records.len();
    for check in &mut receipt.checks {
        if check.id == "railgun-a1" {
            check.status = if proven == total {
                "proven"
            } else if proven + conditional == total {
                "conditional"
            } else {
                "unknown"
            }
            .into();
            check.detail = format!(
                "{proven}/{total} kernels Proven, {conditional} Conditional (witness required)"
            );
        }
    }
    receipt.checks.push(Check {
        id: "task-graph".into(),
        status: if details.iter().any(|d| d.contains(": unknown (")) {
            "unknown"
        } else {
            "conditional"
        }
        .into(),
        detail: details.join("\n"),
    });
    if proofs > 0 {
        for o in [OBLIGATION_ARENA, OBLIGATION_PROTOCOL] {
            if !receipt.obligations.iter().any(|x| x == o) {
                receipt.obligations.push(o.to_owned());
            }
        }
        receipt.obligations.sort();
    }
    Ok(receipt)
}

// ---------------------------------------------------------------------------
// command line
// ---------------------------------------------------------------------------

fn collect_objects(dir: &std::path::Path, out: &mut BTreeMap<[u8; 32], std::path::PathBuf>) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in read.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_objects(&path, out);
        } else if let Ok(bytes) = std::fs::read(&path) {
            out.entry(Sha256::digest(&bytes).into()).or_insert(path);
        }
    }
}

/// `railgun-cert pool [--out DIR] [--standalone DIR]... [--timings] POOL.co...` writes the
/// [`certify_pool`] receipt of each pool object to
/// `DIR/receipts/<arch>/<object_sha256>.receipt.json` (standalone objects are
/// found by SHA-256 anywhere under the `--standalone` directories; `--timings`
/// prints the per-phase wall times of each pool object to stderr, see
/// [`crate::timing`]);
/// `railgun-cert pool-dump POOL.co` prints each tagged kernel's region table and
/// the canonical text of its lifted instructions.
pub fn cli(command: &str, args: &[String]) -> Result<(), String> {
    let mut out = std::path::PathBuf::from("railgun-cert-out");
    let mut dirs = Vec::new();
    let mut objects = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out = it.next().ok_or("--out needs a value")?.into(),
            "--timings" => crate::timing::enable(),
            "--standalone" => dirs.push(std::path::PathBuf::from(
                it.next().ok_or("--standalone needs a value")?,
            )),
            other => objects.push(std::path::PathBuf::from(other)),
        }
    }
    match command {
        "pool" => {
            let mut index = BTreeMap::new();
            for d in &dirs {
                collect_objects(d, &mut index);
            }
            let lookup = |digest: &[u8; 32]| index.get(digest).and_then(|p| std::fs::read(p).ok());
            for path in objects {
                let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                let started = std::time::Instant::now();
                let result = crate::timing::timed("certify_pool", || certify_pool(&bytes, &lookup));
                if crate::timing::enabled() {
                    eprintln!(
                        "timings {} ({:.2} s wall)\n{}",
                        path.display(),
                        started.elapsed().as_secs_f64(),
                        crate::timing::take_report()
                    );
                }
                let receipt = result.map_err(|e| format!("{}: {e}", path.display()))?;
                let dir = out.join("receipts").join(&receipt.arch);
                std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                let file = dir.join(format!("{}.receipt.json", receipt.object_sha256));
                std::fs::write(
                    &file,
                    serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                for k in &receipt.railgun.kernels {
                    println!(
                        "{}\t{}\t{}\t{}",
                        receipt.arch,
                        k.symbol,
                        k.status,
                        file.display()
                    );
                    for u in k
                        .unresolved
                        .iter()
                        .filter(|u| u.rule.starts_with("task_graph-"))
                    {
                        println!("  {}: {}", u.rule, u.detail);
                    }
                }
                if let Some(c) = receipt.checks.iter().find(|c| c.id == "task-graph") {
                    println!("{}", c.detail);
                }
            }
            Ok(())
        }
        "pool-dump" => {
            for path in objects {
                let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                let notes = read_regions(&bytes)?;
                let lifted = lift_object(
                    &bytes,
                    Options {
                        frontend: Frontend::Hipcc,
                    },
                )
                .map_err(|e| e.to_string())?;
                for n in &notes {
                    println!("kernel {} dispatcher {:?}", n.kernel, n.dispatcher);
                    for r in &n.regions {
                        println!(
                            "  region t{} [{:#x},{:#x}) native={} src={} prefix={} wg=({},{}) block={} lds={}",
                            r.template_id, r.entry_pc, r.exit_pc, r.is_native(), r.source_symbol,
                            r.prefix, r.wg_x, r.wg_yz, r.block, r.dynamic_lds
                        );
                    }
                    let Some(k) = lifted
                        .program
                        .kernels
                        .iter()
                        .find(|k| k.symbol.0 == n.kernel)
                    else {
                        continue;
                    };
                    let arch = lifted.program.target.arch;
                    for (b, block) in k.body.blocks.iter().enumerate() {
                        println!("block {b}: {:?}", block.term);
                        for pos in block.range.0..block.range.1 {
                            let inst = k.body.insts.get(k.body.layout[pos]).expect("laid out");
                            let pc = rel_pc(k, inst.prov.pc).unwrap_or(0);
                            let tag = n
                                .regions
                                .iter()
                                .find(|r| (r.entry_pc..r.exit_pc).contains(&u64::from(pc)))
                                .map_or(String::new(), |r| format!(" [t{}]", r.template_id));
                            println!(
                                "  {pc:#06x} {}{tag}",
                                peacemaker_lift::text::print::canonical(inst, arch)
                                    .unwrap_or_else(|e| format!("<{e}>"))
                            );
                        }
                    }
                }
            }
            Ok(())
        }
        other => Err(format!("unknown pool command {other}")),
    }
}

#[cfg(test)]
mod sealed_tests {
    use super::*;
    use railgun_corpus::receipt::{
        case_digest, ArgRecord, Differential, SealedAccessRecord, SealedEntryRecord,
    };
    use std::collections::BTreeMap;

    fn arg(offset: u32, size: u32, role: &str, accesses: usize, classes: &[&str]) -> ArgRecord {
        ArgRecord {
            offset,
            size,
            index: None,
            name: format!("a{offset}"),
            value_kind: "global_buffer".into(),
            role: role.into(),
            mode: (accesses > 0).then(|| "read".to_owned()),
            bound: None,
            classes: classes.iter().map(|c| (*c).to_owned()).collect(),
            paths: vec![],
            cache_bits: vec![],
            accesses,
            shared: false,
            read_cache: "not_read".into(),
            sinks: vec![],
            loaded: true,
        }
    }

    fn table(arg_off: u32, bound: u64) -> SealedTableRecord {
        SealedTableRecord {
            table_arg_offset: arg_off,
            bound_kind: SEALED_BOUND_KERNARG.into(),
            bound_value: bound,
            entry_bytes: 8,
            entries: vec![SealedEntryRecord {
                pc: 0x40,
                class: "smem_load".into(),
                path: "scalar".into(),
                cache: "th=0 scope=0".into(),
                width: 8,
                index_expr: "load@0x20".into(),
                guard_pcs: vec![0x30],
            }],
            accesses: vec![SealedAccessRecord {
                pc: 0x50,
                class: "vmem_load".into(),
                path: "vector".into(),
                cache: "th=0 scope=0".into(),
                width: 16,
                entry_pcs: vec![0x40],
            }],
            null_guarded: false,
        }
    }

    /// Table arg +0 (no counted accesses), output pointer +8, bound scalar +16.
    fn record() -> KernelRecord {
        KernelRecord {
            symbol: "body".into(),
            status: STATUS_CONDITIONAL.into(),
            unresolved: vec![],
            kernarg_size: 24,
            read_cache: "not_read".into(),
            dynamic_kernarg_reads: false,
            taint_overflow: false,
            side: Default::default(),
            implicit: Default::default(),
            args: vec![
                arg(0, 8, "pointer", 0, &[]),
                arg(8, 8, "pointer", 1, &["vmem_store"]),
                arg(16, 4, "integer", 0, &[]),
            ],
            dwords: vec![],
            differential: Differential {
                source: "none".into(),
                status: "pass".into(),
                args: vec![],
                radiowave_mutable_read_cache: None,
                guards: vec![],
                guarded: true,
            },
            pm_obligations: BTreeMap::new(),
            pm_analyze_error: None,
            unguarded_unresolved: vec![],
            guards: vec![],
            indexed_effects: vec![],
            sealed_tables: vec![table(0, 16)],
            descriptor_arenas: vec![],
            required_witnesses: vec![WITNESS_SEALED_TABLE.to_owned()],
            case_digest: String::new(),
            guard_flag: None,
            task_protocol: None,
            wg_footprints: Vec::new(),
        }
    }

    fn refused(r: &KernelRecord) -> Refusal {
        template_from_standalone(7, r).expect_err("must refuse")
    }

    #[test]
    fn sealed_only_conditional_is_admitted_with_unchanged_offsets() {
        let (loads, sealed) = template_from_standalone(7, &record()).expect("admitted");
        assert_eq!(sealed, vec![table(0, 16)]);
        let offs: Vec<u32> = loads.iter().map(|l| l.record_offset).collect();
        assert_eq!(offs, [0, 8]);
        assert_eq!(loads[0].classes, ["smem_load"]);
        assert_eq!(loads[1].classes, ["vmem_store"]);
    }

    #[test]
    fn other_conditional_shapes_are_refused() {
        let mut indexed = record();
        indexed.indexed_effects = vec![railgun_corpus::receipt::IndexedEffectRecord {
            pc: 0,
            base_arg_offset: 8,
            mode: "read".into(),
            class: "vmem_load".into(),
            path: "vector".into(),
            cache: "".into(),
            width: 4,
            address_expr: "x".into(),
            index_loads: vec![],
            guards: vec![],
        }];
        indexed.required_witnesses = vec!["indexed_bind".into(), WITNESS_SEALED_TABLE.into()];
        let r = refused(&indexed);
        assert_eq!(r.rule, "task_graph-region-arena");
        assert!(r.detail.contains("indexed"), "{}", r.detail);

        let mut guards = record();
        guards.guards = vec![railgun_corpus::receipt::GuardRecord {
            kind: "k".into(),
            arg_offset: 0,
            size: 4,
            value: "1".into(),
        }];
        assert_eq!(refused(&guards).rule, "task_graph-region-arena");

        let mut wit = record();
        wit.required_witnesses = vec!["indexed_bind".into(), WITNESS_SEALED_TABLE.into()];
        assert_eq!(refused(&wit).rule, "task_graph-region-arena");

        let mut overlap = record();
        overlap.sealed_tables = vec![table(0, 8)];
        assert_eq!(refused(&overlap).rule, "task_graph-region-arena");

        let mut outside = record();
        outside.sealed_tables = vec![table(0, 24)];
        assert_eq!(refused(&outside).rule, "task_graph-region-arena");

        let mut scalar_table = record();
        scalar_table.sealed_tables = vec![table(16, 20)];
        assert_eq!(refused(&scalar_table).rule, "task_graph-region-arena");

        let mut written = record();
        written.args[0] = arg(0, 8, "pointer", 1, &["vmem_store"]);
        assert_eq!(refused(&written).rule, "task_graph-region-arena");
    }

    #[test]
    fn pool_record_with_sealed_templates_validates_with_both_witnesses() {
        let (loads, sealed) = template_from_standalone(7, &record()).unwrap();
        let mut pool = record();
        pool.args.clear();
        pool.sealed_tables.clear();
        pool.required_witnesses.clear();
        pool.descriptor_arenas = vec![DescriptorArenaRecord {
            arena_arg_offset: 0,
            templates: vec![ArenaTemplateRecord {
                template: 7,
                pointer_loads: loads,
                sealed_tables: sealed,
                wg_footprints: Vec::new(),
                hidden_grid: Vec::new(),
            }],
        }];
        pool.required_witnesses = pool.implied_witnesses();
        assert_eq!(
            pool.required_witnesses,
            [WITNESS_DESCRIPTOR_ARENA, WITNESS_SEALED_TABLE]
        );
        pool.case_digest = case_digest("obj", "elf", "gfx1201", &pool);
        pool.validate_case("obj", "elf", "gfx1201").expect("valid");
    }
}
