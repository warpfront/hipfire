// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Liveness-based VGPR renaming by re-emission (opt-in: `compact_vgprs`).
//!
//! An emitter lays its registers out statically (every register `Live::Whole`), so a kernel whose live
//! ranges would fit in far fewer registers still declares the sum of its buffers. This pass compacts the layout
//! without touching the instruction DAG:
//!
//! 1. **Pass 1** builds the kernel with the checked `Builder`, recording the program as it reaches the final
//!    order (after the VALU window scheduler and VOPD pairing), which submission ids (the n-th `Builder::emit`
//!    call) each instruction or packet carries, and the scheduler's decision per VALU window.
//! 2. [`vgpr_compaction`] builds the CFG from labels and branches, solves per-unit reaching definitions
//!    (def-use chains), splits each physical register into *webs* (def-use connected components, unioned at
//!    joins), and renames the webs under these hard constraints:
//!    * `new == old (mod 4)` for every unit (VOPD bank/parity legality, tuple alignment); a group that carries
//!      a multi-register operand keeps its position inside the aligned 2/4/8-block that the original operand
//!      lay in, so the new plan's aligned granules still contain every access;
//!    * `v0` and every live-in register keep their identity; no load's destination is placed in `v0` (a buffer
//!      op without `offen` encodes its unused vaddr as `v0`, and the independent disassembly replay reads it);
//!    * a multi-register operand `v[a:b]` forces the webs of its units into one rigid group (the group moves as a
//!      whole by a multiple of the group alignment);
//!    * no two interfering webs share a unit; a def interferes with every web live after it, with the other
//!      defs of its instruction and, unless the instruction is a plain single-register VALU op, with its sources.
//!    A def that is not a plain full-register write (`.h/.l`, `_d16`, DPP, permlane, `v_writelane`, 16-bit
//!    results) or whose destination is also a source is part of the web of the value it updates; so is every def
//!    reachable from a write of EXEC (inactive lanes keep the old value).
//!    A web only holds a value where some path wrote it (or the dispatch did): a read that may see garbage
//!    does not keep its web live on the path that never wrote it.
//! 3. The renaming is proven before it is used: reaching definitions are recomputed on the renamed operand
//!    lists and must equal the original def-use chains instruction for instruction (a read of garbage may see
//!    any value).
//! 4. **Pass 2** re-emits the kernel through the checked `Builder` with the map installed
//!    (`Builder::set_vgpr_map`): `emit` rewrites each submitted instruction's VGPR operands (text, defs, uses)
//!    by submission id, `flush_valu` replays the VALU window schedule (and VOPD pairing) pass 1 chose, because
//!    the interference was computed on that final order, and the wait ledger and hazard trackers see the
//!    renamed program, so every wait for the new reuse pattern (async VMEM/DS writes, store-source WAR) is
//!    recomputed.
use crate::insn::{control_flow, Instruction, MemoryClass};
use crate::reg::{Kind, RegRef};
use crate::Emitted;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// One item of the final-order program stream of pass 1.
#[derive(Clone, Debug)]
pub enum StreamItem {
    /// An instruction or VOPD packet as it reached the program, with the submission ids it carries
    /// (`usize::MAX` for none).
    Insn {
        insn: Instruction,
        subs: [usize; 2],
    },
    Label(String),
}

/// One step of a recorded VALU window schedule (local indices of the window's submissions).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// One instruction.
    One(usize),
    /// A VOPD packet of two instructions.
    Pair(usize, usize),
    /// One merged wait for the strongest requirement of these (ready) instructions, before any of them issues.
    Wait(Vec<usize>),
}

/// What pass 1 records: the program in final order (after the VALU window scheduler), the hash of every
/// submitted instruction's text (by submission id) and the scheduler's decision per window (in issue order).
#[derive(Clone, Debug, Default)]
pub struct Recording {
    pub stream: Vec<StreamItem>,
    pub hashes: Vec<u64>,
    pub windows: Vec<Vec<Step>>,
    /// Every submitted instruction in submission order (the program the emitter wrote, labels in place), as it
    /// reached `Builder::emit` after the pass's renaming.
    pub submitted: Vec<StreamItem>,
}

/// What an emitter's build function does with the VGPR map.
#[derive(Clone, Copy)]
pub enum Pass<'a> {
    /// First pass: build with the emitter's own layout and record the submission stream.
    Record,
    /// Intermediate pass: rename by the (pre-scheduling) map, schedule afresh and record the resulting stream.
    Spread(&'a VgprMap),
    /// Last pass: build with the compacted layout.
    Apply(&'a VgprMap),
}
impl Pass<'_> {
    /// The VGPR count of the renamed plan (`None` when the emitter keeps its own layout).
    pub fn vgprs(&self) -> Option<u16> {
        match self {
            Pass::Record => None,
            Pass::Spread(m) | Pass::Apply(m) => (m.vgprs != 0).then_some(m.vgprs),
        }
    }
}

/// Diagnostics of one compaction.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// Registers of the uncompacted plan (highest unit used + 1).
    pub before: u16,
    /// Maximum simultaneously live webs (a lower bound before the mod-4/rigidity constraints).
    pub max_live: u16,
    /// `4 x` the peak live webs of the busiest bank (unit mod 4): the bound once banks are kept.
    pub bank_bound: u16,
    pub webs: usize,
    pub groups: usize,
    pub pinned_webs: usize,
    pub order: &'static str,
    /// VGPRs of the renamed plan the scheduler saw (0: the emitter's own layout).
    pub pre_vgprs: u16,
    /// Independent VOPD candidate pairs the bank assignment looked at, and how many it made legal.
    pub cand_edges: usize,
    pub cand_ok: usize,
    /// Register groups that changed bank.
    pub rotated: usize,
    /// Covering definitions tied to the register of the web they cover.
    pub ties: usize,
    /// The same in the compaction of the scheduled program (spread renamings).
    pub ties_post: usize,
}

#[derive(Clone, Debug)]
struct Inner {
    /// Per submitted instruction: `(hash of its original text, [(old unit, new unit)] sorted by old unit)`.
    per_insn: Vec<(u64, Box<[(u8, u8)]>)>,
    /// The VALU window schedules of pass 1, replayed by pass 2 (empty when `replay` is off).
    windows: Vec<Vec<Step>>,
    /// Pass 2 replays the recorded window schedules; off for the renaming that precedes scheduling.
    replay: bool,
    /// The covering definitions of the renaming (spread only: the compaction of the scheduled program inherits them).
    covers: Vec<Cover>,
}
/// The renaming of one kernel, keyed by submission id (the n-th `Builder::emit` call), not by final pc.
#[derive(Clone, Debug)]
pub struct VgprMap {
    /// VGPRs of the compacted plan (multiple of 8).
    pub vgprs: u16,
    pub stats: Stats,
    inner: Arc<Inner>,
}

pub(crate) fn text_hash(text: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

/// A VGPR token of an instruction text: byte span, first unit, unit count.
struct Tok {
    start: usize,
    end: usize,
    base: u8,
    len: u8,
    range: bool,
}
/// VGPR operands named in `text` after the mnemonic (`v5`, `v[4:7]`), in textual order.
fn vtokens(text: &str) -> Vec<Tok> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = match text.find(char::is_whitespace) {
        Some(k) => k,
        None => return out,
    };
    while i < b.len() {
        let prev_word =
            i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_' || b[i - 1] == b'.');
        if b[i] == b'v' && !prev_word {
            let j = i + 1;
            if j < b.len() && b[j] == b'[' {
                if let Some(close) = text[j..].find(']').map(|k| k + j) {
                    if let Some((lo, hi)) = text[j + 1..close].split_once(':') {
                        if let (Ok(lo), Ok(hi)) = (lo.trim().parse::<u8>(), hi.trim().parse::<u8>())
                        {
                            if hi >= lo {
                                out.push(Tok {
                                    start: i,
                                    end: close + 1,
                                    base: lo,
                                    len: hi - lo + 1,
                                    range: true,
                                });
                                i = close + 1;
                                continue;
                            }
                        }
                    }
                }
            } else {
                let mut k = j;
                while k < b.len() && b[k].is_ascii_digit() {
                    k += 1;
                }
                let next_word = k < b.len() && (b[k].is_ascii_alphanumeric() || b[k] == b'_');
                if k > j && !next_word {
                    if let Ok(n) = text[j..k].parse::<u8>() {
                        out.push(Tok {
                            start: i,
                            end: k,
                            base: n,
                            len: 1,
                            range: false,
                        });
                        i = k;
                        continue;
                    }
                }
            }
        }
        i += 1;
    }
    out
}

/// The renamed unit of old unit `u` in the first of `items` (submission renamings) that names it.
fn renamed_in<'a>(
    mut items: impl Iterator<Item = &'a (u64, Box<[(u8, u8)]>)>,
    u: u8,
) -> Option<u8> {
    items.find_map(|(_, pairs)| {
        pairs
            .binary_search_by_key(&u, |p| p.0)
            .ok()
            .map(|k| pairs[k].1)
    })
}

impl VgprMap {
    /// Number of submitted instructions the map covers.
    pub fn len(&self) -> usize {
        self.inner.per_insn.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.per_insn.is_empty()
    }
    /// Number of VALU window schedules the map replays.
    pub fn windows(&self) -> usize {
        self.inner.windows.len()
    }
    /// Pass 2 replays the recorded window schedules (off for the renaming that precedes scheduling).
    pub fn replays(&self) -> bool {
        self.inner.replay
    }
    /// The schedule of the `n`-th non-empty VALU window, which must hold `len` instructions.
    pub(crate) fn window(&self, n: usize, len: usize) -> Result<&[Step], String> {
        let w =
            self.inner.windows.get(n).ok_or_else(|| {
                format!("vgpr map: VALU window {n} is past the recorded schedule")
            })?;
        let held: usize = w
            .iter()
            .map(|s| match s {
                Step::One(_) => 1,
                Step::Pair(..) => 2,
                Step::Wait(_) => 0,
            })
            .sum();
        if held != len {
            return Err(format!("vgpr map: VALU window {n} holds {len} instructions, the recorded schedule {held} (emission is not deterministic)"));
        }
        Ok(w)
    }
    /// Registers `regs` of the emitter's numbering as the program holds them just before submission `at`: the
    /// renamed unit of the last earlier submission that names the unit (the value pass 1's wait ledger saw
    /// there), or, for a unit not named yet, of the first submission that names it. A unit no submission
    /// names is dropped: nothing ever writes or reads it.
    pub(crate) fn translate_defs(&self, at: usize, regs: &[RegRef]) -> Vec<RegRef> {
        let at = at.min(self.len());
        let mut out = vec![];
        for r in regs {
            if r.kind != Kind::V {
                out.push(*r);
                continue;
            }
            for u in r.base..r.base + r.len {
                let held = renamed_in(self.inner.per_insn[..at].iter().rev(), u)
                    .or_else(|| renamed_in(self.inner.per_insn[at..].iter(), u));
                if let Some(n) = held {
                    out.push(RegRef {
                        kind: Kind::V,
                        base: n,
                        len: 1,
                    });
                }
            }
        }
        out
    }
    /// The instruction submitted as the `n`-th `emit` call, with every VGPR operand renamed.
    pub(crate) fn rewrite(&self, n: usize, mut insn: Instruction) -> Result<Instruction, String> {
        let (hash, pairs) = self.inner.per_insn.get(n).ok_or_else(|| {
            format!(
                "vgpr map: submission {n} is past the recorded stream ({})",
                self.len()
            )
        })?;
        if *hash != text_hash(&insn.text) {
            return Err(format!(
                "vgpr map: submission {n} `{}` differs from the recorded pass (emission is not deterministic)",
                insn.text
            ));
        }
        if pairs.is_empty() {
            return Ok(insn);
        }
        let f = |u: u8| -> Result<u8, String> {
            pairs
                .binary_search_by_key(&u, |p| p.0)
                .map(|k| pairs[k].1)
                .map_err(|_| {
                    format!(
                        "vgpr map: v{u} of `{}` (submission {n}) is unmapped",
                        insn.text
                    )
                })
        };
        let reg = |base: u8, len: u8| -> Result<u8, String> {
            let nb = f(base)?;
            for k in 1..len {
                if f(base + k)? != nb + k {
                    return Err(format!(
                        "vgpr map: v[{base}:{}] of `{}` is not contiguous after renaming",
                        base + len - 1,
                        insn.text
                    ));
                }
            }
            Ok(nb)
        };
        let mut text = String::with_capacity(insn.text.len());
        let mut at = 0;
        for t in vtokens(&insn.text) {
            text.push_str(&insn.text[at..t.start]);
            let nb = reg(t.base, t.len)?;
            if t.range {
                text.push_str(&format!("v[{nb}:{}]", u16::from(nb) + u16::from(t.len) - 1));
            } else {
                text.push_str(&format!("v{nb}"));
            }
            at = t.end;
        }
        text.push_str(&insn.text[at..]);
        for r in insn.defs.iter_mut().chain(insn.uses.iter_mut()) {
            if r.kind == Kind::V {
                r.base = reg(r.base, r.len)?;
            }
        }
        insn.text = text;
        Ok(insn)
    }
}

// ------------------------------------------------------------------------------------------------ operands

pub(crate) const UNITS: usize = 256;
/// The reaching value of a register no instruction wrote and the dispatch did not define (garbage).
pub(crate) const UNDEF: u32 = u32::MAX;

/// Operand units of one instruction.
pub(crate) struct Op {
    /// Units written, in operand order.
    pub(crate) defs: Vec<u8>,
    /// Units read: the declared sources plus, for a def that is not a plain full-register write, the def itself.
    pub(crate) uses: Vec<u8>,
    /// `uses[k]` is also a def of this instruction (the def continues the web it updates).
    /// Multi-register operand ranges of the text: `(base, len)`.
    ranges: Vec<(u8, u8)>,
    /// A def and a source of this instruction may share a register (plain single-register VALU op).
    overlap_ok: bool,
}

fn push_units(out: &mut Vec<u8>, r: &RegRef) {
    for k in 0..r.len {
        let u = r.base + k;
        if !out.contains(&u) {
            out.push(u);
        }
    }
}

/// The def of this instruction is not a plain full-register write (it merges with the old value).
fn non_plain(insn: &Instruction, exec_written: bool) -> bool {
    if exec_written {
        return true;
    }
    let m = insn.mnemonic();
    // `op_sel` alone selects source halves (v_fma_mix_f32 writes a whole f32): only a 16-bit result is partial.
    const TEXT: [&str; 9] = [
        ".h",
        ".l",
        "_d16",
        "dpp",
        "row_",
        "quad_perm",
        "bank_mask",
        "fi:",
        "sdwa",
    ];
    if TEXT.iter().any(|t| insn.text.contains(t)) || m.contains("_d16") {
        return true;
    }
    if m.starts_with("v_permlane") || m.starts_with("v_writelane") || m.starts_with("v_movrel") {
        return true;
    }
    const WIDE16: [&str; 5] = [
        "v_cvt_f32_f16",
        "v_cvt_f32_bf16",
        "v_fma_mix_f32",
        "v_pk_",
        "v_dot2",
    ];
    m.starts_with("v_") && m.contains("16") && !WIDE16.iter().any(|p| m.starts_with(p))
}

fn plain_overlap(insn: &Instruction, non_plain: bool) -> bool {
    let m = insn.mnemonic();
    if non_plain || !m.starts_with("v_") || insn.memory.is_some() {
        return false;
    }
    const BAD: [&str; 8] = [
        "v_wmma",
        "v_swmmac",
        "v_mad_u64",
        "v_mad_i64",
        "v_div_",
        "v_permlane",
        "v_readlane",
        "v_cmpx",
    ];
    if BAD.iter().any(|p| m.starts_with(p)) {
        return false;
    }
    insn.defs
        .iter()
        .chain(&insn.uses)
        .all(|r| r.kind != Kind::V || r.len == 1)
}

/// The smallest aligned block (2, 4 or 8 units) that contains `[base, base + len)`.
fn granule(base: u8, len: u8) -> Option<u8> {
    [2u8, 4, 8]
        .into_iter()
        .find(|&g| g >= len && base / g == (base + len - 1) / g)
}

pub(crate) fn operands(insn: &Instruction, exec_written: bool) -> Result<Op, String> {
    let mut defs = vec![];
    let mut uses = vec![];
    let mut declared = vec![];
    for r in insn.defs.iter().filter(|r| r.kind == Kind::V) {
        push_units(&mut defs, r);
        push_units(&mut declared, r);
    }
    for r in insn.uses.iter().filter(|r| r.kind == Kind::V) {
        push_units(&mut uses, r);
        push_units(&mut declared, r);
    }
    let np = non_plain(insn, exec_written) && !defs.is_empty();
    if np {
        for &d in &defs {
            if !uses.contains(&d) {
                uses.push(d);
            }
        }
    }
    let mut ranges = vec![];
    for t in vtokens(&insn.text) {
        for k in 0..t.len {
            if !declared.contains(&(t.base + k)) {
                return Err(format!(
                    "`{}`: v{} is named in the text but not declared as a def or use",
                    insn.text,
                    t.base + k
                ));
            }
        }
        if t.len > 1 {
            if granule(t.base, t.len).is_none() {
                return Err(format!(
                    "`{}`: v[{}:{}] straddles the aligned granules of the plan",
                    insn.text,
                    t.base,
                    t.base + t.len - 1
                ));
            }
            ranges.push((t.base, t.len));
        }
    }
    for r in insn
        .defs
        .iter()
        .chain(&insn.uses)
        .filter(|r| r.kind == Kind::V && r.len > 1)
    {
        if !ranges.contains(&(r.base, r.len)) {
            ranges.push((r.base, r.len));
        }
    }
    Ok(Op {
        defs,
        uses,
        ranges,
        overlap_ok: plain_overlap(insn, np),
    })
}

// ------------------------------------------------------------------------------------------------ CFG and reaching definitions

pub(crate) enum Item {
    Label(String),
    Insn(usize),
}
pub(crate) struct Block {
    pub(crate) first: usize,
    pub(crate) last: usize,
    pub(crate) succ: Vec<usize>,
}

pub(crate) fn build_cfg(items: &[Item], insns: &[&Instruction]) -> Result<Vec<Block>, String> {
    let terminator = |it: &Item| match it {
        Item::Insn(i) => {
            let m = insns[*i].mnemonic();
            m == "s_branch" || m.starts_with("s_cbranch_") || m == "s_endpgm"
        }
        Item::Label(_) => false,
    };
    let mut leader = vec![false; items.len() + 1];
    leader[0] = true;
    for (k, it) in items.iter().enumerate() {
        if matches!(it, Item::Label(_)) {
            leader[k] = true;
        }
        if terminator(it) {
            leader[k + 1] = true;
        }
    }
    let mut blocks: Vec<Block> = vec![];
    let mut start = 0;
    for k in 1..=items.len() {
        if leader[k] {
            if k > start {
                blocks.push(Block {
                    first: start,
                    last: k - 1,
                    succ: vec![],
                });
            }
            start = k;
        }
    }
    let mut by_label = std::collections::HashMap::new();
    for (b, blk) in blocks.iter().enumerate() {
        if let Item::Label(name) = &items[blk.first] {
            by_label.insert(name.clone(), b);
        }
    }
    let n = blocks.len();
    for b in 0..n {
        let last = &items[blocks[b].last];
        let mut succ = vec![];
        let mut falls = true;
        if let Item::Insn(i) = last {
            let ins = insns[*i];
            let m = ins.mnemonic();
            if m == "s_endpgm" {
                falls = false;
            } else if m == "s_branch" || m.starts_with("s_cbranch_") {
                let target = crate::sync::branch_target(&ins.text)
                    .ok_or_else(|| format!("branch without a target: `{}`", ins.text))?;
                let t = *by_label
                    .get(target)
                    .ok_or_else(|| format!("branch to unplaced label {target}"))?;
                succ.push(t);
                falls = m != "s_branch";
            } else if control_flow(m) {
                return Err(format!("`{}`: unsupported control flow", ins.text));
            }
        }
        if falls && b + 1 < n {
            succ.push(b + 1);
        }
        blocks[b].succ = succ;
    }
    Ok(blocks)
}

type State = Vec<Vec<u32>>;

fn merge(into: &mut Vec<u32>, from: &[u32]) -> bool {
    let mut changed = false;
    for &x in from {
        if let Err(p) = into.binary_search(&x) {
            into.insert(p, x);
            changed = true;
        }
    }
    changed
}

/// Reaching definitions per used unit of every reachable instruction.
pub(crate) struct Reaching {
    /// `use_rd[insn][k]`: the dense def ids reaching `ops[insn].uses[k]` (sorted; ids below `UNITS` are the dispatch-defined entry values, `UNDEF` garbage).
    pub(crate) use_rd: Vec<Vec<Vec<u32>>>,
    /// `def_id[insn][j]`: the dense id of `ops[insn].defs[j]`.
    pub(crate) def_id: Vec<Vec<u32>>,
    pub(crate) reachable: Vec<bool>,
    /// `use_rd` before the correlated-branch refinement (`reaching_refined`); empty otherwise.
    pub(crate) base_use_rd: Vec<Vec<Vec<u32>>>,
}

/// Reaching definitions with the correlated branches of [`crate::corr`] taken into account.
pub(crate) fn reaching_refined(
    items: &[Item],
    blocks: &[Block],
    insns: &[&Instruction],
    ops: &[Op],
    live_in: &[bool],
) -> Reaching {
    let mut rd = reaching(items, blocks, ops, live_in);
    let base = rd.use_rd.clone();
    crate::corr::refine(items, blocks, insns, ops, live_in, &mut rd);
    rd.base_use_rd = base;
    rd
}

pub(crate) fn reaching(items: &[Item], blocks: &[Block], ops: &[Op], live_in: &[bool]) -> Reaching {
    let mut next = UNITS as u32;
    let mut def_id = Vec::with_capacity(ops.len());
    for op in ops {
        def_id.push(
            op.defs
                .iter()
                .map(|_| {
                    next += 1;
                    next - 1
                })
                .collect::<Vec<u32>>(),
        );
    }
    let transfer = |state: &mut State, blk: &Block, rec: &mut dyn FnMut(usize, usize, &[u32])| {
        for item in &items[blk.first..=blk.last] {
            if let Item::Insn(i) = item {
                for (k, &u) in ops[*i].uses.iter().enumerate() {
                    rec(*i, k, &state[u as usize]);
                }
                for (j, &u) in ops[*i].defs.iter().enumerate() {
                    state[u as usize].clear();
                    state[u as usize].push(def_id[*i][j]);
                }
            }
        }
    };
    let mut ins: Vec<Option<State>> = vec![None; blocks.len()];
    if blocks.is_empty() {
        return Reaching {
            use_rd: vec![],
            def_id,
            reachable: vec![],
            base_use_rd: vec![],
        };
    }
    ins[0] = Some(
        (0..UNITS as u32)
            .map(|u| vec![if live_in[u as usize] { u } else { UNDEF }])
            .collect(),
    );
    let mut work: VecDeque<usize> = VecDeque::from([0]);
    let mut queued = vec![false; blocks.len()];
    queued[0] = true;
    while let Some(b) = work.pop_front() {
        queued[b] = false;
        let mut st = ins[b].clone().unwrap();
        transfer(&mut st, &blocks[b], &mut |_, _, _| {});
        for &s in &blocks[b].succ {
            let changed = match &mut ins[s] {
                slot @ None => {
                    *slot = Some(st.clone());
                    true
                }
                Some(into) => {
                    let mut c = false;
                    for u in 0..UNITS {
                        c |= merge(&mut into[u], &st[u]);
                    }
                    c
                }
            };
            if changed && !queued[s] {
                queued[s] = true;
                work.push_back(s);
            }
        }
    }
    let mut use_rd: Vec<Vec<Vec<u32>>> = ops.iter().map(|o| vec![vec![]; o.uses.len()]).collect();
    let mut reachable = vec![false; ops.len()];
    for (b, blk) in blocks.iter().enumerate() {
        if let Some(st) = &ins[b] {
            let mut st = st.clone();
            for item in &items[blk.first..=blk.last] {
                if let Item::Insn(i) = item {
                    reachable[*i] = true;
                }
            }
            transfer(&mut st, blk, &mut |i, k, r| use_rd[i][k] = r.to_vec());
        }
    }
    Reaching {
        use_rd,
        def_id,
        reachable,
        base_use_rd: vec![],
    }
}

// ------------------------------------------------------------------------------------------------ union-find

/// The def id naming the web of a use: its first real reaching def; a use that only reads garbage is
/// an entry value (pinned).
fn rep_of(r: &[u32], unit: u8) -> u32 {
    r.iter()
        .copied()
        .find(|&x| x != UNDEF)
        .unwrap_or(u32::from(unit))
}

struct Uf(Vec<u32>);
impl Uf {
    fn new(n: usize) -> Self {
        Self((0..n as u32).collect())
    }
    fn find(&mut self, mut x: u32) -> u32 {
        while self.0[x as usize] != x {
            let p = self.0[x as usize];
            self.0[x as usize] = self.0[p as usize];
            x = self.0[x as usize];
        }
        x
    }
    fn union(&mut self, a: u32, b: u32) -> u32 {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            let (lo, hi) = (a.min(b), a.max(b));
            self.0[hi as usize] = lo;
            lo
        } else {
            a
        }
    }
}

// ------------------------------------------------------------------------------------------------ bitsets

#[derive(Clone)]
struct Bits(Vec<u64>);
impl Bits {
    fn new(n: usize) -> Self {
        Self(vec![0; n.div_ceil(64)])
    }
    fn set(&mut self, i: usize) {
        self.0[i / 64] |= 1 << (i % 64);
    }
    fn clear(&mut self, i: usize) {
        self.0[i / 64] &= !(1 << (i % 64));
    }
    fn or(&mut self, o: &Bits) -> bool {
        let mut c = false;
        for (a, b) in self.0.iter_mut().zip(&o.0) {
            let n = *a | *b;
            c |= n != *a;
            *a = n;
        }
        c
    }
    fn count(&self) -> usize {
        self.0.iter().map(|w| w.count_ones() as usize).sum()
    }
    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().flat_map(|(w, &bits)| {
            let mut b = bits;
            std::iter::from_fn(move || {
                if b == 0 {
                    None
                } else {
                    let t = b.trailing_zeros() as usize;
                    b &= b - 1;
                    Some(w * 64 + t)
                }
            })
        })
    }
}

// ------------------------------------------------------------------------------------------------ the pass

/// The instruction writes EXEC (its first operand, or a `saveexec` form).
fn writes_exec(i: &Instruction) -> bool {
    let m = i.mnemonic();
    if m.starts_with("v_cmpx") || m.contains("saveexec") || m.contains("wrexec") {
        return true;
    }
    !m.starts_with("v_")
        && !m.starts_with("s_cbranch")
        && i.text
            .split_once(char::is_whitespace)
            .is_some_and(|(_, rest)| rest.trim_start().starts_with("exec"))
}

/// Instructions that may run with a partial EXEC (reachable from a write of it) do not write whole registers.
pub(crate) fn exec_dirty_of(items: &[Item], blocks: &[Block], insns: &[&Instruction]) -> Vec<bool> {
    let mut exec_dirty = vec![false; insns.len()];
    let mut dirty_in = vec![false; blocks.len()];
    loop {
        let mut changed = false;
        for (b, blk) in blocks.iter().enumerate() {
            let mut dirty = dirty_in[b];
            for item in &items[blk.first..=blk.last] {
                if let Item::Insn(i) = item {
                    dirty |= writes_exec(insns[*i]);
                    exec_dirty[*i] |= dirty;
                }
            }
            if dirty {
                for &s in &blk.succ {
                    if !dirty_in[s] {
                        dirty_in[s] = true;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    exec_dirty
}

/// How the allocator places webs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Pack the scheduled program: every web keeps its bank (`unit mod 4`), first-fit.
    Compact,
    /// Rename the program in submission order, before it is scheduled, so the scheduler sees only true
    /// dependences and VOPD-legal banks: webs may change bank (single-register webs freely), reuse goes to the
    /// register freed longest ago, and the banks may be picked to make independent VOPD candidates legal.
    Spread(Spread),
}

/// The choices of the [`Mode::Spread`] renaming.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spread {
    /// Pick the banks that make the independent same-op VOPD candidates legal (never past the register count the
    /// original banks reach).
    pub banks: bool,
    /// Keep every web defined inside a VALU window off the registers of all the webs the window touches, so the
    /// scheduler meets no false dependence there (as far as 256 registers allow).
    pub windows: bool,
}

/// Compact the VGPRs of the recorded pass-1 stream.
///
/// `live_in` lists the registers defined by the dispatch (their identity is kept); `v0` (the workitem id) is
/// always live-in.
pub fn vgpr_compaction(rec: &Recording, live_in: &[RegRef]) -> Result<VgprMap, String> {
    allocate(rec, live_in, Mode::Compact, None)
}

/// A definition that covers a read for definedness only (see `allocate`): the submissions of the definition and of
/// the read, and the unit the read names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cover {
    def_sub: usize,
    use_sub: usize,
    unit: u8,
}

/// `inherit`: the renaming this program was recorded after; its covering definitions bind here too (the scheduled
/// program no longer shows the control correlation that made them dead, but the definedness they provide must
/// survive).
pub(crate) fn allocate(
    rec: &Recording,
    live_in: &[RegRef],
    mode: Mode,
    inherit: Option<&VgprMap>,
) -> Result<VgprMap, String> {
    let spread = matches!(mode, Mode::Spread(_));
    let spread_opts = match mode {
        Mode::Spread(o) => Some(o),
        Mode::Compact => None,
    };
    let mut insns: Vec<&Instruction> = vec![];
    let mut subs_of: Vec<[usize; 2]> = vec![];
    let mut items: Vec<Item> = vec![];
    for s in &rec.stream {
        match s {
            StreamItem::Insn { insn, subs } => {
                items.push(Item::Insn(insns.len()));
                insns.push(insn);
                subs_of.push(*subs);
            }
            StreamItem::Label(l) => items.push(Item::Label(l.clone())),
        }
    }
    let blocks = build_cfg(&items, &insns)?;
    let exec_dirty = exec_dirty_of(&items, &blocks, &insns);
    let ops: Vec<Op> = insns
        .iter()
        .enumerate()
        .map(|(i, ins)| operands(ins, exec_dirty[i]))
        .collect::<Result<_, _>>()?;
    let mut live_units = vec![false; UNITS];
    live_units[0] = true;
    for r in live_in.iter().filter(|r| r.kind == Kind::V) {
        for k in 0..r.len {
            live_units[(r.base + k) as usize] = true;
        }
    }
    let rd = reaching_refined(&items, &blocks, &insns, &ops, &live_units);

    // Webs: def-use connected components.
    let n_ids = UNITS + rd.def_id.iter().map(Vec::len).sum::<usize>();
    let mut def_unit = vec![0u8; n_ids];
    let mut def_insn = vec![u32::MAX; n_ids];
    for u in 0..UNITS {
        def_unit[u] = u as u8;
    }
    for (i, ids) in rd.def_id.iter().enumerate() {
        for (j, &id) in ids.iter().enumerate() {
            def_unit[id as usize] = ops[i].defs[j];
            def_insn[id as usize] = i as u32;
        }
    }
    let mut uf = Uf::new(n_ids);
    for (i, op) in ops.iter().enumerate() {
        if !rd.reachable[i] {
            continue;
        }
        for (k, r) in rd.use_rd[i].iter().enumerate() {
            let rep = rep_of(r, op.uses[k]);
            for &x in r.iter().filter(|&&x| x != UNDEF) {
                uf.union(rep, x);
            }
            // A def that is also a source continues the web of the value it updates.
            if let Some(j) = op.defs.iter().position(|d| *d == op.uses[k]) {
                uf.union(rep, rd.def_id[i][j]);
            }
        }
    }
    // Dense web ids over the roots of ids that take part in the program.
    let mut web_of_root: std::collections::HashMap<u32, u32> = Default::default();
    let mut web_unit: Vec<u8> = vec![];
    let mut firsts: Vec<(u32, u32)> = vec![];
    let mut use_web: Vec<Vec<u32>> = vec![vec![]; ops.len()];
    let mut def_web: Vec<Vec<u32>> = vec![vec![]; ops.len()];
    for (i, op) in ops.iter().enumerate() {
        if !rd.reachable[i] {
            continue;
        }
        let mut web = |id: u32| -> u32 {
            let root = uf.find(id);
            *web_of_root.entry(root).or_insert_with(|| {
                web_unit.push(def_unit[id as usize]);
                (web_unit.len() - 1) as u32
            })
        };
        for (k, r) in rd.use_rd[i].iter().enumerate() {
            if r.is_empty() {
                return Err("use without a reaching definition".into());
            }
            use_web[i].push(web(rep_of(r, op.uses[k])));
        }
        for j in 0..op.defs.len() {
            let w = web(rd.def_id[i][j]);
            def_web[i].push(w);
            firsts.push((w, i as u32));
        }
    }
    let n_webs = web_unit.len();
    let mut web_first = vec![u32::MAX; n_webs];
    for (w, i) in firsts {
        web_first[w as usize] = web_first[w as usize].min(i);
    }
    // Pinned webs hold a dispatch-defined value (`v0`, `live_in`) or a register only ever read as garbage.
    let mut web_pinned = vec![false; n_webs];
    // Webs holding an entry value keep their identity.
    for u in 0..UNITS {
        if let Some(&w) = web_of_root.get(&uf.find(u as u32)) {
            web_pinned[w as usize] = true;
            web_first[w as usize] = 0;
        }
    }
    let pinned_webs = web_pinned.iter().filter(|p| **p).count();

    // Rigid groups of webs: the units of a multi-register operand.
    let mut guf = Uf::new(n_webs);
    let mut ties = 0usize;
    let mut tied: std::collections::HashSet<u32> = Default::default();
    let mut covers: Vec<Cover> = vec![];
    // Definedness: the checked program (and the independent definedness replay) asks every read to have a
    // definition of its register on every path from the entry, whichever path can really run. A definition the
    // correlated branches show can never reach a read (the prologue zero of an accumulator) is such a covering
    // definition: its web keeps the register of the web it covers, so the renamed program is defined wherever the
    // original was.
    {
        let mut web_used = vec![false; n_webs];
        for uw in &use_web {
            for &w in uw {
                web_used[w as usize] = true;
            }
        }
        for (i, op) in ops.iter().enumerate() {
            if !rd.reachable[i] || rd.base_use_rd.is_empty() {
                continue;
            }
            for k in 0..op.uses.len() {
                for &d in &rd.base_use_rd[i][k] {
                    if d == UNDEF || (d as usize) < UNITS || rd.use_rd[i][k].contains(&d) {
                        continue;
                    }
                    if let Some(&dw) = web_of_root.get(&uf.find(d)) {
                        if !web_used[dw as usize] {
                            guf.union(dw, use_web[i][k]);
                            tied.insert(dw);
                            ties += 1;
                            let di = def_insn[d as usize] as usize;
                            if subs_of[di][0] != usize::MAX && subs_of[i][0] != usize::MAX {
                                covers.push(Cover {
                                    def_sub: subs_of[di][0],
                                    use_sub: subs_of[i][0],
                                    unit: op.uses[k],
                                });
                            }
                        }
                    }
                }
            }
        }
        // the covers of the renaming this program came from, in this program's names
        if let Some(pre) = inherit {
            let mut idx_of_sub: std::collections::HashMap<usize, usize> = Default::default();
            for (i, sb) in subs_of.iter().enumerate() {
                for &x in sb.iter().filter(|x| **x != usize::MAX) {
                    idx_of_sub.insert(x, i);
                }
            }
            for c in &pre.inner.covers {
                let renamed = pre.inner.per_insn[c.use_sub]
                    .1
                    .binary_search_by_key(&c.unit, |p| p.0)
                    .map(|k| pre.inner.per_insn[c.use_sub].1[k].1);
                let (Ok(unit), Some(&di), Some(&ui)) =
                    (renamed, idx_of_sub.get(&c.def_sub), idx_of_sub.get(&c.use_sub))
                else {
                    continue;
                };
                if !rd.reachable[di] || !rd.reachable[ui] || def_web[di].is_empty() {
                    continue;
                }
                if let Some(k) = ops[ui].uses.iter().position(|u| *u == unit) {
                    guf.union(def_web[di][0], use_web[ui][k]);
                    ties += 1;
                }
            }
        }
    }
    let web_at =
        |i: usize, u: u8, op: &Op, use_web: &[Vec<u32>], def_web: &[Vec<u32>]| -> Option<u32> {
            if let Some(j) = op.defs.iter().position(|d| *d == u) {
                Some(def_web[i][j])
            } else {
                op.uses.iter().position(|d| *d == u).map(|k| use_web[i][k])
            }
        };
    let mut galign = vec![if spread { 1u8 } else { 4u8 }; n_webs];
    for (i, op) in ops.iter().enumerate() {
        if !rd.reachable[i] {
            continue;
        }
        for &(base, len) in &op.ranges {
            let first = web_at(i, base, op, &use_web, &def_web).ok_or("range without a web")?;
            let g = if spread { granule(base, len).unwrap() } else { granule(base, len).unwrap().max(4) };
            for k in 0..len {
                let w = web_at(i, base + k, op, &use_web, &def_web).ok_or("range without a web")?;
                guf.union(first, w);
            }
            for k in 0..len {
                let w = web_at(i, base + k, op, &use_web, &def_web).unwrap();
                let a = &mut galign[w as usize];
                *a = (*a).max(g);
            }
        }
    }

    // Liveness over webs.
    let n_blocks = blocks.len();
    let ops_uw = |i: usize| -> &[u32] { &use_web[i] };
    let mut gen: Vec<Bits> = vec![Bits::new(n_webs); n_blocks];
    let mut kill: Vec<Bits> = vec![Bits::new(n_webs); n_blocks];
    for (b, blk) in blocks.iter().enumerate() {
        for item in items[blk.first..=blk.last].iter().rev() {
            if let Item::Insn(i) = item {
                if !rd.reachable[*i] {
                    continue;
                }
                for (j, &d) in def_web[*i].iter().enumerate() {
                    // a def whose unit is also a source continues the web: no kill
                    let cont = ops[*i].uses.contains(&ops[*i].defs[j]);
                    if !cont {
                        gen[b].clear(d as usize);
                        kill[b].set(d as usize);
                    }
                }
                for &u in ops_uw(*i) {
                    gen[b].set(u as usize);
                }
            }
        }
    }
    let mut live_in_b: Vec<Bits> = vec![Bits::new(n_webs); n_blocks];
    let mut live_out_b: Vec<Bits> = vec![Bits::new(n_webs); n_blocks];
    let mut preds: Vec<Vec<usize>> = vec![vec![]; n_blocks];
    for (b, blk) in blocks.iter().enumerate() {
        for &s in &blk.succ {
            preds[s].push(b);
        }
    }
    let mut work: VecDeque<usize> = (0..n_blocks).rev().collect();
    let mut queued = vec![true; n_blocks];
    while let Some(b) = work.pop_front() {
        queued[b] = false;
        let mut out = Bits::new(n_webs);
        for &s in &blocks[b].succ {
            out.or(&live_in_b[s]);
        }
        let mut inn = out.clone();
        for (w, k) in inn.0.iter_mut().zip(&kill[b].0) {
            *w &= !*k;
        }
        inn.or(&gen[b]);
        live_out_b[b] = out;
        if live_in_b[b].or(&inn) {
            for &p in &preds[b] {
                if !queued[p] {
                    queued[p] = true;
                    work.push_back(p);
                }
            }
        }
    }

    // A web holds a value only where some path wrote it (or the dispatch did): forward "maybe defined" webs.
    // A read of a register no path has written yet is a read of garbage, so such a point does not keep the web live.
    let mut block_defs: Vec<Bits> = vec![Bits::new(n_webs); n_blocks];
    for (b, blk) in blocks.iter().enumerate() {
        for item in &items[blk.first..=blk.last] {
            if let Item::Insn(i) = item {
                for &d in &def_web[*i] {
                    block_defs[b].set(d as usize);
                }
            }
        }
    }
    let mut dfn_in: Vec<Bits> = vec![Bits::new(n_webs); n_blocks];
    let mut dfn_out: Vec<Bits> = vec![Bits::new(n_webs); n_blocks];
    for w in (0..n_webs).filter(|&w| web_pinned[w]) {
        dfn_in[0].set(w);
    }
    loop {
        let mut changed = false;
        for b in 0..n_blocks {
            let mut inn = dfn_in[b].clone();
            for &p in &preds[b] {
                inn.or(&dfn_out[p]);
            }
            let mut out = inn.clone();
            out.or(&block_defs[b]);
            changed |= dfn_in[b].or(&inn);
            changed |= dfn_out[b].or(&out);
        }
        if !changed {
            break;
        }
    }

    // Interference.
    let mut adj: Vec<Vec<u32>> = vec![vec![]; n_webs];
    let mut max_live = 0usize;
    let mut bank_peak = [0usize; 4];
    let mut peak: Option<(usize, usize, Vec<usize>)> = None;
    let want_hot = spread_opts.is_some_and(|o| o.banks);
    let mut held_at: Vec<(usize, Vec<u32>)> = vec![];
    let mut web_last = vec![0u32; n_webs];
    let add = |adj: &mut Vec<Vec<u32>>, a: u32, b: u32| {
        if a != b {
            adj[a as usize].push(b);
            adj[b as usize].push(a);
        }
    };
    for (b, blk) in blocks.iter().enumerate() {
        // maybe-defined webs after each instruction of the block
        let mut after: Vec<Bits> = vec![];
        let mut cur = dfn_in[b].clone();
        for item in &items[blk.first..=blk.last] {
            if let Item::Insn(i) = item {
                for &d in &def_web[*i] {
                    cur.set(d as usize);
                }
                after.push(cur.clone());
            }
        }
        let mut live = live_out_b[b].clone();
        let mut pos = after.len();
        for item in items[blk.first..=blk.last].iter().rev() {
            let Item::Insn(i) = item else { continue };
            let i = *i;
            pos -= 1;
            if !rd.reachable[i] {
                continue;
            }
            let mut held = live.clone();
            for (h, a) in held.0.iter_mut().zip(&after[pos].0) {
                *h &= *a;
            }
            for w in held.iter().chain(use_web[i].iter().map(|w| *w as usize)) {
                web_last[w] = web_last[w].max(i as u32);
            }
            let count = held.count();
            if want_hot {
                held_at.push((count, held.iter().map(|w| w as u32).collect()));
            }
            if count > max_live {
                peak = Some((count, i, held.iter().collect()));
            }
            max_live = max_live.max(count);
            let mut banks = [0usize; 4];
            for w in held.iter() {
                banks[web_unit[w] as usize % 4] += 1;
            }
            for (peak, n) in bank_peak.iter_mut().zip(banks) {
                *peak = (*peak).max(n);
            }
            let defs = &def_web[i];
            for (x, &d1) in defs.iter().enumerate() {
                for &d2 in &defs[x + 1..] {
                    add(&mut adj, d1, d2);
                }
                for w in held.iter() {
                    add(&mut adj, d1, w as u32);
                }
                if !ops[i].overlap_ok {
                    for &u in &use_web[i] {
                        add(&mut adj, d1, u);
                    }
                }
            }
            for (j, &d) in defs.iter().enumerate() {
                if !ops[i].uses.contains(&ops[i].defs[j]) {
                    live.clear(d as usize);
                }
            }
            for &u in &use_web[i] {
                live.set(u as usize);
            }
        }
    }
    // Spread: the scheduler reorders every window, so a web defined inside one must not share a register with any
    // web the window touches (no false dependence for it to respect), as far as the 256 registers allow.
    if spread_opts.is_some_and(|o| o.windows) {
        for w in valu_windows(&items, &insns, &rd.reachable) {
            let mut touched: Vec<u32> = w
                .iter()
                .flat_map(|&i| def_web[i].iter().chain(&use_web[i]).copied())
                .collect();
            touched.sort_unstable();
            touched.dedup();
            if touched.len() > 384 {
                continue;
            }
            let mut defd: Vec<u32> = w.iter().flat_map(|&i| def_web[i].iter().copied()).collect();
            defd.sort_unstable();
            defd.dedup();
            for &d in &defd {
                for &t in &touched {
                    add(&mut adj, d, t);
                }
            }
        }
    }
    for a in adj.iter_mut() {
        a.sort_unstable();
        a.dedup();
    }

    // Webs defined by an asynchronous memory load (their destination is pending after the issue).
    let mut async_def = vec![false; n_webs];
    for (i, ins) in insns.iter().enumerate() {
        if rd.reachable[i]
            && matches!(
                ins.memory,
                Some(
                    MemoryClass::VmemLoad
                        | MemoryClass::DsLoad
                        | MemoryClass::VmemAtomic { returns: true }
                )
            )
        {
            for &d in &def_web[i] {
                async_def[d as usize] = true;
            }
        }
    }
    // Groups.
    let mut group_of_root: std::collections::HashMap<u32, usize> = Default::default();
    let mut group_members: Vec<Vec<u32>> = vec![];
    let mut web_group = vec![0usize; n_webs];
    for w in 0..n_webs as u32 {
        let r = guf.find(w);
        let g = *group_of_root.entry(r).or_insert_with(|| {
            group_members.push(vec![]);
            group_members.len() - 1
        });
        group_members[g].push(w);
        web_group[w as usize] = g;
    }
    let n_groups = group_members.len();
    struct Group {
        umin: i32,
        umax: i32,
        align: i32,
        pinned: bool,
        first: u32,
        degree: usize,
        size: usize,
    }
    let groups: Vec<Group> = group_members
        .iter()
        .map(|m| Group {
            umin: m
                .iter()
                .map(|&w| i32::from(web_unit[w as usize]))
                .min()
                .unwrap(),
            umax: m
                .iter()
                .map(|&w| i32::from(web_unit[w as usize]))
                .max()
                .unwrap(),
            align: m
                .iter()
                .map(|&w| i32::from(galign[w as usize]))
                .max()
                .unwrap(),
            pinned: m.iter().any(|&w| web_pinned[w as usize]),
            first: m.iter().map(|&w| web_first[w as usize]).min().unwrap(),
            degree: m.iter().map(|&w| adj[w as usize].len()).sum(),
            size: m.len(),
        })
        .collect();

    // Banks (Spread): the shift mod 4 of every group so that the independent VOPD candidates are legal.
    let mut shift = vec![0u8; n_groups];
    let (mut cand_edges, mut cand_ok, mut rotated) = (0usize, 0usize, 0usize);
    if want_hot {
        let edges = bank_edges(&items, &insns, &ops, &def_web, &use_web, &rd.reachable);
        let allowed: Vec<u8> = groups
            .iter()
            .map(|g| match (g.pinned, g.align) {
                (true, _) => 0b0001,
                (false, 1) => 0b1111,
                (false, 2) => 0b0101,
                _ => 0b0001,
            })
            .collect();
        // the positions where a bank could exceed the original bank peak: those with nearly that many live webs
        let cap = bank_peak.iter().copied().max().unwrap_or(0);
        let hot: Vec<Vec<u32>> = held_at
            .into_iter()
            .filter(|(c, _)| *c * 4 > cap * 3 && *c > cap)
            .map(|(_, h)| h)
            .collect();
        let r = solve_banks(&edges, &web_unit, &web_group, &group_members, &allowed, &hot, cap);
        cand_edges = edges.len();
        cand_ok = r.1;
        shift = r.0;
        rotated = shift.iter().filter(|s| **s != 0).count();
    }
    // Placement: lowest feasible shift per group, several orders, keep the smallest.
    let mut unreachable_top = 0i32;
    for (i, op) in ops.iter().enumerate() {
        if !rd.reachable[i] {
            for &u in op.defs.iter().chain(&op.uses) {
                unreachable_top = unreachable_top.max(i32::from(u) + 1);
            }
        }
    }
    let place = |order: &[usize]| -> Result<(Vec<i32>, i32), String> {
        let mut delta = vec![i32::MIN; n_groups];
        let mut placed = vec![-1i32; n_webs];
        let mut top = unreachable_top;
        for (g, grp) in groups.iter().enumerate() {
            if grp.pinned {
                delta[g] = 0;
                for &w in &group_members[g] {
                    placed[w as usize] = i32::from(web_unit[w as usize]);
                }
                top = top.max(grp.umax + 1);
            }
        }
        let mut forbidden = vec![false; 2 * UNITS + 1];
        let mut last_end = vec![-1i64; UNITS];
        for &g in order {
            let grp = &groups[g];
            if grp.pinned {
                continue;
            }
            forbidden.iter_mut().for_each(|f| *f = false);
            for &w in &group_members[g] {
                let uw = i32::from(web_unit[w as usize]);
                // A buffer op without `offen` encodes its unused vaddr as v0, which the independent disassembly
                // replay reads as a use: never leave a load's pending destination in v0.
                if async_def[w as usize] {
                    forbidden[(-uw + UNITS as i32) as usize] = true;
                }
                for &x in &adj[w as usize] {
                    let t = placed[x as usize];
                    if t >= 0 {
                        forbidden[(t - uw + UNITS as i32) as usize] = true;
                    }
                }
            }
            let step = grp.align.max(4);
            let off = i32::from(shift[g]);
            let mut d = -grp.umin;
            while d.rem_euclid(step) != off {
                d += 1;
            }
            // `Compact`: the lowest feasible shift. `Spread`: among the feasible registers below the current top,
            // the one whose last occupant is the oldest (no false dependence between near uses), else the lowest.
            let mut chosen: Option<(i64, i32)> = None;
            loop {
                if grp.umax + d >= UNITS as i32 {
                    break;
                }
                if !forbidden[(d + UNITS as i32) as usize] {
                    if !spread {
                        chosen = Some((0, d));
                        break;
                    }
                    if grp.umax + d < top {
                        let key = group_members[g]
                            .iter()
                            .map(|&w| last_end[(i32::from(web_unit[w as usize]) + d) as usize])
                            .max()
                            .unwrap();
                        if chosen.map_or(true, |c| key < c.0) {
                            chosen = Some((key, d));
                        }
                    } else {
                        if chosen.is_none() {
                            chosen = Some((0, d));
                        }
                        break;
                    }
                }
                d += step;
            }
            let Some((_, d)) = chosen else {
                return Err("vgpr compaction: no feasible placement within 256 registers".into());
            };
            delta[g] = d;
            for &w in &group_members[g] {
                let t = i32::from(web_unit[w as usize]) + d;
                placed[w as usize] = t;
                let e = &mut last_end[t as usize];
                *e = (*e).max(i64::from(web_last[w as usize]));
            }
            top = top.max(grp.umax + d + 1);
        }
        Ok((delta, top))
    };
    let base: Vec<usize> = (0..n_groups).collect();
    let mut orders: Vec<(&'static str, Vec<usize>)> = vec![];
    let by = |key: &dyn Fn(usize) -> (i64, i64, i64)| {
        let mut o = base.clone();
        o.sort_by_key(|&g| key(g));
        o
    };
    orders.push((
        "first-def",
        by(&|g| {
            (
                i64::from(groups[g].first),
                -(groups[g].size as i64),
                g as i64,
            )
        }),
    ));
    orders.push((
        "degree",
        by(&|g| {
            (
                -(groups[g].degree as i64),
                i64::from(groups[g].first),
                g as i64,
            )
        }),
    ));
    orders.push((
        "size-first",
        by(&|g| {
            (
                -(groups[g].size as i64),
                i64::from(groups[g].first),
                g as i64,
            )
        }),
    ));
    orders.push((
        "degree-per-web",
        by(&|g| {
            (
                -((groups[g].degree / groups[g].size.max(1)) as i64),
                i64::from(groups[g].first),
                g as i64,
            )
        }),
    ));
    let mut best: Option<(Vec<i32>, i32, &'static str)> = None;
    for (name, order) in &orders {
        if let Ok((d, top)) = place(order) {
            if best.as_ref().map_or(true, |b| top < b.1) {
                best = Some((d, top, name));
            }
        }
    }
    // Seeded perturbations of the best linear-scan order.
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    // more perturbations for the small kernels, where one more register is a larger share and a trial is cheap
    let rounds = (150_000 / n_groups.max(1)).clamp(24, 600);
    for round in 0..rounds {
        let mut o = orders[0].1.clone();
        for _ in 0..(o.len() / 16 + 1) {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let a = (seed % o.len().max(1) as u64) as usize;
            let span = 1 + (seed >> 40) as usize % (8 + round);
            let b = (a + span).min(o.len().saturating_sub(1));
            o.swap(a, b);
        }
        if let Ok((d, top)) = place(&o) {
            if best.as_ref().map_or(true, |b| top < b.1) {
                best = Some((d, top, "perturbed"));
            }
        }
    }
    let (delta, top, order_name) = best.ok_or("vgpr compaction: no feasible placement")?;
    let vgprs = (top.max(1) as u16).div_ceil(8) * 8;

    // The per-submission map.
    let mut per_insn: Vec<Option<(u64, Box<[(u8, u8)]>)>> = vec![None; rec.hashes.len()];
    let new_unit =
        |w: u32| -> u8 { (i32::from(web_unit[w as usize]) + delta[web_group[w as usize]]) as u8 };
    let before = insns
        .iter()
        .flat_map(|i| i.defs.iter().chain(&i.uses))
        .filter(|r| r.kind == Kind::V)
        .map(|r| u16::from(r.base) + u16::from(r.len))
        .max()
        .unwrap_or(0);
    let mut new_ops: Vec<Op> = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        let mut pairs: Vec<(u8, u8)> = vec![];
        if rd.reachable[i] {
            for (j, &u) in op.defs.iter().enumerate() {
                pairs.push((u, new_unit(def_web[i][j])));
            }
            for (k, &u) in op.uses.iter().enumerate() {
                let n = new_unit(use_web[i][k]);
                match pairs.iter().find(|p| p.0 == u) {
                    Some(p) if p.1 != n => {
                        return Err(format!(
                        "vgpr compaction: `{}` v{u} is renamed to v{} as a def and v{n} as a use",
                        insns[i].text, p.1
                    ))
                    }
                    Some(_) => {}
                    None => pairs.push((u, n)),
                }
            }
        } else {
            for &u in op.defs.iter().chain(&op.uses) {
                if !pairs.iter().any(|p| p.0 == u) {
                    pairs.push((u, u));
                }
            }
        }
        pairs.sort_unstable();
        // Checks of the renaming of this instruction.
        for (x, p) in pairs.iter().enumerate() {
            if (!spread && p.1 % 4 != p.0 % 4) || u16::from(p.1) >= vgprs {
                return Err(format!(
                    "vgpr compaction: `{}`: v{} -> v{} violates the bank or plan constraint",
                    insns[i].text, p.0, p.1
                ));
            }
            for q in &pairs[..x] {
                let both = |s: &[u8]| s.contains(&p.0) && s.contains(&q.0);
                let mixed = !both(&op.defs) && !both(&op.uses);
                if q.1 == p.1 && (!mixed || !op.overlap_ok) {
                    return Err(format!(
                        "vgpr compaction: `{}`: v{} and v{} are renamed to v{}",
                        insns[i].text, q.0, p.0, p.1
                    ));
                }
            }
        }
        let f = |u: u8| pairs.iter().find(|p| p.0 == u).map(|p| p.1).unwrap();
        for &(b, l) in &op.ranges {
            for k in 1..l {
                if f(b + k) != f(b) + k {
                    return Err(format!(
                        "vgpr compaction: `{}`: v[{b}:{}] is not contiguous after renaming",
                        insns[i].text,
                        b + l - 1
                    ));
                }
            }
            let g = granule(b, l).unwrap();
            if f(b) % g != b % g {
                return Err(format!(
                    "vgpr compaction: `{}`: v[{b}:{}] leaves its aligned granule",
                    insns[i].text,
                    b + l - 1
                ));
            }
        }
        new_ops.push(Op {
            defs: op.defs.iter().map(|&u| f(u)).collect(),
            uses: op.uses.iter().map(|&u| f(u)).collect(),
            ranges: vec![],
            overlap_ok: op.overlap_ok,
        });
        for sub in subs_of[i].into_iter().filter(|s| *s != usize::MAX) {
            let slot = per_insn
                .get_mut(sub)
                .ok_or("vgpr compaction: submission id out of range")?;
            if slot
                .replace((rec.hashes[sub], pairs.clone().into_boxed_slice()))
                .is_some()
            {
                return Err(format!(
                    "vgpr compaction: submission {sub} reaches the program twice"
                ));
            }
        }
    }
    let per_insn: Vec<(u64, Box<[(u8, u8)]>)> = per_insn
        .into_iter()
        .enumerate()
        .map(|(n, p)| {
            p.ok_or_else(|| format!("vgpr compaction: submission {n} never reached the program"))
        })
        .collect::<Result<_, _>>()?;
    // Proof: the renamed program has the def-use chains of the original, instruction for instruction. A read
    // that may see garbage (`UNDEF`) may see any other value in the renamed program.
    let rd2 = reaching_refined(&items, &blocks, &insns, &new_ops, &live_units);
    for i in 0..ops.len() {
        if !rd.reachable[i] {
            continue;
        }
        for (k, (a, b)) in rd.use_rd[i].iter().zip(&rd2.use_rd[i]).enumerate() {
            let garbage = a.contains(&UNDEF);
            let real: Vec<u32> = a.iter().copied().filter(|&x| x != UNDEF).collect();
            let ok = if garbage {
                real.iter().all(|x| b.contains(x))
            } else {
                a == b
            };
            if !ok {
                return Err(format!(
                    "vgpr compaction: renaming changed the def-use chains of `{}` (instruction {i}, v{}): {a:?} -> {b:?}",
                    insns[i].text, ops[i].uses[k]
                ));
            }
        }
    }
    Ok(VgprMap {
        vgprs,
        stats: Stats {
            before,
            max_live: max_live as u16,
            bank_bound: (4 * bank_peak.iter().copied().max().unwrap_or(0)) as u16,
            webs: n_webs,
            groups: n_groups,
            pinned_webs,
            order: order_name,
            cand_edges,
            cand_ok,
            rotated,
            ties,
            ..Stats::default()
        },
        inner: Arc::new(Inner {
            per_insn,
            windows: if spread { vec![] } else { rec.windows.clone() },
            replay: !spread,
            covers,
        }),
    })
}

/// Declare VGPRs `[0, n)` as aligned width-8/4/2/1 granules (every multi-register access of the compacted
/// kernel lies inside one granule: see `vgpr_compaction`).
pub fn declare_vgprs(regs: &mut crate::RegPlan, n: u16) -> Result<(), String> {
    let mut at = 0u16;
    while at < n {
        let width = [8u16, 4, 2, 1]
            .into_iter()
            .find(|&w| at + w <= n && at % w == 0)
            .unwrap_or(1);
        regs.add_range(
            "vgpr",
            Kind::V,
            at as u8,
            width as u8,
            crate::reg::Live::Whole,
        )?;
        at += width;
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------ banks

/// The maximal runs of schedulable VALU instructions of a submission-order program: the windows the Builder's
/// scheduler reorders.
fn valu_windows(items: &[Item], insns: &[&Instruction], reachable: &[bool]) -> Vec<Vec<usize>> {
    let mut windows: Vec<Vec<usize>> = vec![];
    let mut cur: Vec<usize> = vec![];
    for it in items {
        match it {
            Item::Insn(i) if reachable[*i] && crate::valu_sched::schedulable(insns[*i]) => {
                cur.push(*i)
            }
            _ => windows.push(std::mem::take(&mut cur)),
        }
    }
    windows.push(cur);
    windows.retain(|w| !w.is_empty());
    windows
}

/// Two independent VOPD-able instructions the scheduler may pair: the webs behind the packet's operands.
struct BankEdge {
    dx: u32,
    dy: u32,
    s0: Option<(u32, u32)>,
    s1: (u32, u32),
}

/// The candidate pairs of a program in submission order: per maximal run of schedulable VALU instructions (the
/// Builder's VALU window), two f32 add/sub/mul/fmac instructions of nearly the same dependence depth, neither
/// reading what the other writes (through the webs), of the same op, that the packet rules could join (one SGPR
/// at most).
fn bank_edges(
    items: &[Item],
    insns: &[&Instruction],
    ops: &[Op],
    def_web: &[Vec<u32>],
    use_web: &[Vec<u32>],
    reachable: &[bool],
) -> Vec<BankEdge> {
    use crate::vopd::Operand;
    const MAX_WINDOW: usize = 4096;
    const REACH: usize = 192;
    const PER_NODE: usize = 8;
    let windows = valu_windows(items, insns, reachable);
    let mut edges = vec![];
    for w in windows.iter().filter(|w| w.len() > 1 && w.len() <= MAX_WINDOW) {
        let n = w.len();
        let words = n.div_ceil(64);
        let mut anc = vec![0u64; n * words];
        let mut level = vec![0usize; n];
        let mut last_def: std::collections::HashMap<u32, usize> = Default::default();
        for (k, &i) in w.iter().enumerate() {
            let mut preds: Vec<usize> = use_web[i]
                .iter()
                .filter_map(|u| last_def.get(u).copied())
                .collect();
            preds.sort_unstable();
            preds.dedup();
            for &p in &preds {
                level[k] = level[k].max(level[p] + 1);
                for x in 0..words {
                    anc[k * words + x] |= anc[p * words + x];
                }
                anc[k * words + p / 64] |= 1 << (p % 64);
            }
            for &d in &def_web[i] {
                last_def.insert(d, k);
            }
        }
        let half: Vec<_> = w
            .iter()
            .map(|&i| crate::valu_sched::vopd_half(insns[i]))
            .collect();
        let web_of = |i: usize, unit: u8| -> Option<u32> {
            ops[i]
                .uses
                .iter()
                .position(|u| *u == unit)
                .map(|k| use_web[i][k])
        };
        for k in 0..n {
            let Some(x) = half[k] else { continue };
            let i = w[k];
            let mut taken = 0;
            for l in k + 1..n.min(k + REACH) {
                let Some(y) = half[l] else { continue };
                let j = w[l];
                if taken >= PER_NODE
                    || level[k].abs_diff(level[l]) > 1
                    || x.op != y.op
                    || anc[l * words + k / 64] >> (k % 64) & 1 == 1
                {
                    continue;
                }
                let sgpr = [x.src0, y.src0]
                    .iter()
                    .filter(|s| matches!(s, Operand::S(_)))
                    .count();
                if sgpr > 1 {
                    continue;
                }
                let (Some(&dx), Some(&dy)) = (def_web[i].first(), def_web[j].first()) else {
                    continue;
                };
                if dx == dy {
                    continue;
                }
                let (Some(bx), Some(by)) = (web_of(i, x.src1), web_of(j, y.src1)) else {
                    continue;
                };
                let s0 = match (x.src0, y.src0) {
                    (Operand::V(a), Operand::V(b)) => match (web_of(i, a), web_of(j, b)) {
                        (Some(a), Some(b)) if a != b => Some((a, b)),
                        (Some(_), Some(_)) => continue, // the same source can never pair (src0 banks)
                        _ => continue,
                    },
                    _ => None,
                };
                edges.push(BankEdge {
                    dx,
                    dy,
                    s0,
                    s1: (bx, by),
                });
                taken += 1;
            }
        }
    }
    edges
}

/// Banks (shift mod 4 of every group) that make the most candidate pairs legal without letting any bank hold
/// more than `cap` live webs at the `hot` positions (the lists of live webs where the pressure is highest): the
/// register count the original banks reach stays reachable. Coordinate descent from the original banks with
/// seeded visiting orders. `allowed[g]`: bit `s` set when group `g` may shift by `s`.
fn solve_banks(
    edges: &[BankEdge],
    web_unit: &[u8],
    web_group: &[usize],
    group_members: &[Vec<u32>],
    allowed: &[u8],
    hot: &[Vec<u32>],
    cap: usize,
) -> (Vec<u8>, usize) {
    let n_groups = group_members.len();
    let mut inc: Vec<Vec<u32>> = vec![vec![]; n_groups];
    for (e, edge) in edges.iter().enumerate() {
        let mut ws = vec![edge.dx, edge.dy, edge.s1.0, edge.s1.1];
        if let Some((a, b)) = edge.s0 {
            ws.push(a);
            ws.push(b);
        }
        let mut gs: Vec<usize> = ws.iter().map(|w| web_group[*w as usize]).collect();
        gs.sort_unstable();
        gs.dedup();
        for g in gs {
            if allowed[g].count_ones() > 1 {
                inc[g].push(e as u32);
            }
        }
    }
    // The three packet rules of an edge: destination parity, src0 banks, src1 banks (the same register is fine).
    let rules = |shift: &[u8], e: &BankEdge| -> (usize, usize) {
        let r = |w: u32| (web_unit[w as usize] + shift[web_group[w as usize]]) & 3;
        let mut met = [
            (r(e.dx) ^ r(e.dy)) & 1 == 1,
            e.s1.0 == e.s1.1 || r(e.s1.0) != r(e.s1.1),
        ]
        .iter()
        .filter(|b| **b)
        .count();
        let mut n = 2;
        if let Some((a, b)) = e.s0 {
            n += 1;
            met += usize::from(r(a) != r(b));
        }
        (met, n)
    };
    let ok = |shift: &[u8], e: &BankEdge| -> bool {
        let (met, n) = rules(shift, e);
        met == n
    };
    // what the descent maximises: the rules met, and the whole edge again when all are (a plateau-free ladder)
    let score = |shift: &[u8], e: &BankEdge| -> usize {
        let (met, n) = rules(shift, e);
        met + if met == n { n } else { 0 }
    };
    let total = |shift: &[u8]| edges.iter().filter(|e| ok(shift, e)).count();
    // hot positions each web is live at
    let n_webs = web_unit.len();
    let mut web_hot: Vec<Vec<u32>> = vec![vec![]; n_webs];
    for (p, h) in hot.iter().enumerate() {
        for &w in h {
            web_hot[w as usize].push(p as u32);
        }
    }
    let movable: Vec<usize> = (0..n_groups).filter(|&g| !inc[g].is_empty()).collect();
    let descend = |seed: &mut u64| -> Vec<u8> {
        let mut shift = vec![0u8; n_groups];
        let mut counts: Vec<[u16; 4]> = hot
            .iter()
            .map(|h| {
                let mut c = [0u16; 4];
                for &w in h {
                    c[(web_unit[w as usize] & 3) as usize] += 1;
                }
                c
            })
            .collect();
        for _ in 0..24 {
            let mut order = movable.clone();
            for k in (1..order.len()).rev() {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                order.swap(k, (*seed % (k as u64 + 1)) as usize);
            }
            let mut improved = false;
            for &g in &order {
                let keep = shift[g];
                let mut scored: Vec<(usize, u8)> = vec![];
                for s in 0..4u8 {
                    if s == keep || allowed[g] >> s & 1 == 1 {
                        shift[g] = s;
                        let sc = inc[g]
                            .iter()
                            .map(|e| score(&shift, &edges[*e as usize]))
                            .sum::<usize>();
                        scored.push((sc, s));
                    }
                }
                shift[g] = keep;
                let cur = scored.iter().find(|x| x.1 == keep).unwrap().0;
                scored.retain(|x| x.0 > cur);
                scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                for (_, s) in scored {
                    // move the group's webs between banks at the hot positions, keep it when no bank exceeds `cap`
                    let mut moved: Vec<(u32, usize, usize)> = vec![];
                    let mut fits = true;
                    'apply: for &w in &group_members[g] {
                        let (from, to) = (
                            ((web_unit[w as usize] + keep) & 3) as usize,
                            ((web_unit[w as usize] + s) & 3) as usize,
                        );
                        if from == to {
                            continue;
                        }
                        for &p in &web_hot[w as usize] {
                            let c = &mut counts[p as usize];
                            c[from] -= 1;
                            c[to] += 1;
                            if c[to] as usize > cap {
                                fits = false;
                            }
                        }
                        moved.push((w, from, to));
                        if !fits {
                            break 'apply;
                        }
                    }
                    if fits {
                        shift[g] = s;
                        improved = true;
                        break;
                    }
                    for (w, from, to) in moved {
                        for &p in &web_hot[w as usize] {
                            let c = &mut counts[p as usize];
                            c[from] += 1;
                            c[to] -= 1;
                        }
                    }
                }
            }
            if !improved {
                break;
            }
        }
        shift
    };
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut best_shift = vec![0u8; n_groups];
    let mut best_total = total(&best_shift);
    for _ in 0..8 {
        let shift = descend(&mut seed);
        let t = total(&shift);
        if t > best_total {
            best_total = t;
            best_shift = shift;
        }
    }
    (best_shift, best_total)
}

// ------------------------------------------------------------------------------------------------ driver

/// The renaming of the recorded scheduled program (`post`, over the names `pre` gave) as one map over the
/// emitter's names.
fn compose(pre: &VgprMap, post: &VgprMap, hashes0: &[u64]) -> Result<VgprMap, String> {
    if pre.len() != post.len() || post.len() != hashes0.len() {
        return Err("vgpr compose: the passes submitted different numbers of instructions".into());
    }
    let mut per_insn = Vec::with_capacity(pre.len());
    for n in 0..pre.len() {
        let a = &pre.inner.per_insn[n].1;
        let b = &post.inner.per_insn[n].1;
        let mut out = Vec::with_capacity(a.len());
        for &(u, v) in a.iter() {
            let w = b
                .binary_search_by_key(&v, |p| p.0)
                .map(|k| b[k].1)
                .map_err(|_| {
                    format!("vgpr compose: submission {n} names v{v} the second pass does not map")
                })?;
            out.push((u, w));
        }
        per_insn.push((hashes0[n], out.into_boxed_slice()));
    }
    Ok(VgprMap {
        vgprs: post.vgprs,
        stats: Stats {
            before: pre.stats.before,
            pre_vgprs: pre.vgprs,
            cand_edges: pre.stats.cand_edges,
            cand_ok: pre.stats.cand_ok,
            rotated: pre.stats.rotated,
            ties: pre.stats.ties,
            ties_post: post.stats.ties,
            ..post.stats.clone()
        },
        inner: Arc::new(Inner {
            per_insn,
            windows: post.inner.windows.clone(),
            replay: true,
            covers: vec![],
        }),
    })
}

/// How [`compact_vgprs_as`] renames the program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Renaming {
    /// Spread renaming with bank assignment and window-wide separation, then compaction of its schedule.
    Full,
    /// As `Full` without the window-wide separation.
    Spread,
    /// Compaction of the emitter's own schedule.
    Plain,
}

/// Three-pass driver for one [`Renaming`]: pass 0 builds and records the program as the emitter wrote it; the
/// spread renaming is computed on it; pass 1 re-emits through that renaming, scheduling afresh, and records the
/// scheduled program; the compaction of that stream is the final renaming that pass 2 applies (replaying pass 1's
/// schedule). `Plain` skips pass 1.
///
/// `build(Pass::Record)` must build the kernel with `Builder::vgpr_pass(Pass::Record)` and its own layout;
/// `build(Pass::Spread(map))` and `build(Pass::Apply(map))` must declare a plan of `map.vgprs` registers and call
/// `Builder::vgpr_pass` with the pass. The emitter must be deterministic in the instruction sequence it submits.
pub fn compact_vgprs_as<T>(
    how: Renaming,
    build: &impl Fn(Pass) -> Result<(T, Emitted), String>,
) -> Result<(T, Emitted), String> {
    let (_, first) = build(Pass::Record)?;
    let rec0 = first
        .vgpr_record
        .ok_or("compact_vgprs: pass 1 did not record (call Builder::vgpr_pass)")?;
    let map = match how {
        Renaming::Plain => vgpr_compaction(&rec0, &[])?,
        Renaming::Full | Renaming::Spread => {
            let rec = Recording {
                stream: rec0.submitted.clone(),
                hashes: rec0.hashes.clone(),
                ..Default::default()
            };
            let pre = allocate(
                &rec,
                &[],
                Mode::Spread(Spread {
                    banks: true,
                    windows: how == Renaming::Full,
                }),
                None,
            )?;
            let (_, second) = build(Pass::Spread(&pre))?;
            let rec1 = second.vgpr_record.ok_or(
                "compact_vgprs: the spread pass did not record (call Builder::vgpr_pass)",
            )?;
            compose(&pre, &allocate(&rec1, &[], Mode::Compact, Some(&pre))?, &rec0.hashes)?
        }
    };
    let (t, mut emitted) = build(Pass::Apply(&map))?;
    emitted.vgpr_stats = Some(map.stats.clone());
    Ok((t, emitted))
}

/// Occupancy (waves per SIMD) of a kernel that declares `vgprs` registers: a 1536-register file per SIMD in
/// 8-register granules, at most 16 waves.
fn waves(vgprs: u16) -> u16 {
    (1536 / (vgprs.max(1).div_ceil(8) * 8)).min(16)
}

/// Compact the VGPRs of the kernel `build` emits: every [`Renaming`] is tried and the one with the most waves,
/// then the fewest instructions, wins (a spread that costs registers never costs occupancy).
pub fn compact_vgprs<T>(
    build: impl Fn(Pass) -> Result<(T, Emitted), String>,
) -> Result<(T, Emitted), String> {
    let mut best: Option<(T, Emitted, (u16, usize, u16))> = None;
    let mut first_err = None;
    // on a tie the plain compaction, which leaves the emitter's own schedule alone, wins
    for how in [Renaming::Plain, Renaming::Spread, Renaming::Full] {
        match compact_vgprs_as(how, &build) {
            Ok((t, emitted)) => {
                let key = (
                    u16::MAX - waves(emitted.shape.next_free_vgpr),
                    emitted.program.instructions.len(),
                    emitted.shape.next_free_vgpr,
                );
                if best.as_ref().map_or(true, |b| key < b.2) {
                    best = Some((t, emitted, key));
                }
            }
            Err(e) => {
                first_err.get_or_insert(e);
            }
        }
    }
    match best {
        Some((t, emitted, _)) => Ok((t, emitted)),
        None => Err(first_err.unwrap_or_else(|| "vgpr compaction: no renaming".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::insn::MemoryClass;

    fn v(base: u8, len: u8) -> RegRef {
        RegRef {
            kind: Kind::V,
            base,
            len,
        }
    }
    fn i(text: &str, defs: &[RegRef], uses: &[RegRef]) -> StreamItem {
        StreamItem::Insn {
            insn: Instruction::new(text, defs.to_vec(), uses.to_vec()),
            subs: [0, usize::MAX],
        }
    }
    /// A recording of `items` as emitted in order, one submission per instruction (no VALU windows).
    fn recording(items: Vec<StreamItem>) -> Recording {
        let mut rec = Recording::default();
        for it in items {
            match it {
                StreamItem::Insn { insn, .. } => {
                    rec.hashes.push(text_hash(&insn.text));
                    rec.stream.push(StreamItem::Insn {
                        insn,
                        subs: [rec.hashes.len() - 1, usize::MAX],
                    });
                }
                l => rec.stream.push(l),
            }
        }
        rec
    }
    fn lbl(n: &str) -> StreamItem {
        StreamItem::Label(n.into())
    }
    fn renamed(m: &VgprMap, n: usize, text: &str, defs: &[RegRef], uses: &[RegRef]) -> Instruction {
        m.rewrite(n, Instruction::new(text, defs.to_vec(), uses.to_vec()))
            .unwrap()
    }

    #[test]
    fn disjoint_lifetimes_share_a_register_and_banks_are_kept() {
        let s = vec![
            i("v_mov_b32_e32 v1, 1", &[v(1, 1)], &[]),
            i("v_add_f32_e32 v2, v1, v0", &[v(2, 1)], &[v(1, 1), v(0, 1)]),
            i("v_mov_b32_e32 v33, 2", &[v(33, 1)], &[]),
            i(
                "v_add_f32_e32 v34, v33, v2",
                &[v(34, 1)],
                &[v(33, 1), v(2, 1)],
            ),
            i("s_endpgm", &[], &[]),
        ];
        let m = vgpr_compaction(&recording(s), &[]).unwrap();
        // v33 is dead-before-live of v1 (bank 1): it lands on v1 again, v34 on v3 or lower of bank 2
        let a = renamed(&m, 2, "v_mov_b32_e32 v33, 2", &[v(33, 1)], &[]);
        assert_eq!(a.text, "v_mov_b32_e32 v1, 2");
        let b = renamed(
            &m,
            3,
            "v_add_f32_e32 v34, v33, v2",
            &[v(34, 1)],
            &[v(33, 1), v(2, 1)],
        );
        assert_eq!(
            b.text,
            "v_add_f32_e32 v6, v1, v2".replace("v6", &format!("v{}", b.defs[0].base))
        );
        assert_eq!(b.defs[0].base % 4, 2);
        assert!(m.vgprs <= 8, "{}", m.vgprs);
    }

    #[test]
    fn live_in_v0_keeps_identity_and_interfering_webs_stay_apart() {
        let s = vec![
            i("v_mov_b32_e32 v8, 1", &[v(8, 1)], &[]),
            i("v_mov_b32_e32 v9, 2", &[v(9, 1)], &[]),
            i(
                "v_add_f32_e32 v10, v8, v0",
                &[v(10, 1)],
                &[v(8, 1), v(0, 1)],
            ),
            i(
                "v_add_f32_e32 v11, v10, v9",
                &[v(11, 1)],
                &[v(10, 1), v(9, 1)],
            ),
            i("s_endpgm", &[], &[]),
        ];
        let m = vgpr_compaction(&recording(s), &[]).unwrap();
        let ins = |n: usize, t: &str, d: u8, u: &[u8]| {
            renamed(
                &m,
                n,
                t,
                &[v(d, 1)],
                &u.iter().map(|&x| v(x, 1)).collect::<Vec<_>>(),
            )
        };
        let r2 = ins(2, "v_add_f32_e32 v10, v8, v0", 10, &[8, 0]);
        assert!(r2.text.ends_with(", v0"), "{}", r2.text);
        let r0 = ins(0, "v_mov_b32_e32 v8, 1", 8, &[]);
        let r1 = ins(1, "v_mov_b32_e32 v9, 2", 9, &[]);
        assert_ne!(r0.defs[0].base, r1.defs[0].base);
    }

    #[test]
    fn multi_register_operands_move_as_a_rigid_aligned_group() {
        let load = |n: u8| {
            Instruction::new(
                format!("buffer_load_b128 v[{n}:{}], v0, s[0:3], null offen", n + 3),
                vec![v(n, 4)],
                vec![
                    v(0, 1),
                    RegRef {
                        kind: Kind::S,
                        base: 0,
                        len: 4,
                    },
                ],
            )
            .memory(MemoryClass::VmemLoad)
        };
        let s = vec![
            StreamItem::Insn {
                insn: load(40),
                subs: [0, usize::MAX],
            },
            i(
                "v_add_f32_e32 v1, v41, v42",
                &[v(1, 1)],
                &[v(41, 1), v(42, 1)],
            ),
            i("v_mov_b32_e32 v2, v1", &[v(2, 1)], &[v(1, 1)]),
            StreamItem::Insn {
                insn: load(88),
                subs: [0, usize::MAX],
            },
            i(
                "v_add_f32_e32 v3, v89, v90",
                &[v(3, 1)],
                &[v(89, 1), v(90, 1)],
            ),
            i("s_endpgm", &[], &[]),
        ];
        let m = vgpr_compaction(&recording(s), &[]).unwrap();
        for (n, base) in [(0usize, 40u8), (3, 88)] {
            let r = m.rewrite(n, load(base)).unwrap();
            assert_eq!(r.defs[0].len, 4);
            assert_eq!(r.defs[0].base % 4, 0);
            assert!(
                r.text
                    .contains(&format!("v[{}:{}]", r.defs[0].base, r.defs[0].base + 3)),
                "{}",
                r.text
            );
        }
        let a = m.rewrite(0, load(40)).unwrap().defs[0].base;
        let b = m.rewrite(3, load(88)).unwrap().defs[0].base;
        assert_eq!(a, b, "second group reuses the dead first group");
        assert!(m.vgprs <= 16);
    }

    #[test]
    fn a_def_that_is_also_a_source_and_joins_stay_in_one_web() {
        // v5 = 1; if (scc) v5 += 2 (fmac-style read-modify-write); use v5 after the join; v21 is a long-lived temporary.
        let s = vec![
            i("v_mov_b32_e32 v21, 3", &[v(21, 1)], &[]),
            i("v_mov_b32_e32 v5, 1", &[v(5, 1)], &[]),
            i("s_cbranch_scc1 .Lskip", &[], &[]),
            i(
                "v_fmac_f32_e32 v5, v21, v21",
                &[v(5, 1)],
                &[v(21, 1), v(5, 1)],
            ),
            lbl(".Lskip"),
            i(
                "v_add_f32_e32 v6, v5, v21",
                &[v(6, 1)],
                &[v(5, 1), v(21, 1)],
            ),
            i("s_endpgm", &[], &[]),
        ];
        let m = vgpr_compaction(&recording(s), &[]).unwrap();
        let a = renamed(&m, 1, "v_mov_b32_e32 v5, 1", &[v(5, 1)], &[]);
        let b = renamed(
            &m,
            3,
            "v_fmac_f32_e32 v5, v21, v21",
            &[v(5, 1)],
            &[v(21, 1), v(5, 1)],
        );
        let t = renamed(&m, 0, "v_mov_b32_e32 v21, 3", &[v(21, 1)], &[]);
        assert_eq!(a.defs[0].base, b.defs[0].base);
        assert_ne!(a.defs[0].base, t.defs[0].base);
    }

    fn w1(base: u8) -> RegRef {
        v(base, 1)
    }
    fn sg(base: u8) -> RegRef {
        RegRef {
            kind: Kind::S,
            base,
            len: 1,
        }
    }
    fn alloc_spread(stream: Vec<StreamItem>, opts: Spread) -> VgprMap {
        let mut rec = recording(stream);
        rec.hashes = (0..rec.hashes.len()).map(|k| rec.hashes[k]).collect();
        allocate(&rec, &[], Mode::Spread(opts), None).unwrap()
    }

    #[test]
    fn a_prologue_zero_under_repeated_guards_is_a_web_of_its_own() {
        // v5 = 0 (prologue); if (bit B of s1) v5 = 0 + v6; v9 = 1; if (bit 3 of s1) v8 = v5 + v6; use v9.
        // With B = 3 the zero never reaches the guarded read: it is a lone definition, the accumulator another web.
        let prog = |bit: u32| {
            vec![
                i("v_mov_b32_e32 v5, 0", &[w1(5)], &[]),
                i("v_mov_b32_e32 v6, 2", &[w1(6)], &[]),
                i(&format!("s_bitcmp0_b32 s1, {bit}"), &[], &[sg(1)]),
                i("s_cbranch_scc1 .L1", &[], &[]),
                i("v_add_f32_e32 v5, 0, v6", &[w1(5)], &[w1(6)]),
                lbl(".L1"),
                i("v_mov_b32_e32 v9, 1", &[w1(9)], &[]),
                i("s_bitcmp0_b32 s1, 3", &[], &[sg(1)]),
                i("s_cbranch_scc1 .L2", &[], &[]),
                i("v_add_f32_e32 v8, v5, v6", &[w1(8)], &[w1(5), w1(6)]),
                lbl(".L2"),
                i("v_add_f32_e32 v10, v9, v9", &[w1(10)], &[w1(9), w1(9)]),
                i("s_endpgm", &[], &[]),
            ]
        };
        let correlated = vgpr_compaction(&recording(prog(3)), &[]).unwrap();
        let unrelated = vgpr_compaction(&recording(prog(4)), &[]).unwrap();
        assert_eq!(correlated.stats.webs, unrelated.stats.webs + 1);
        // the guarded accumulator stays live across v9's definition, whichever web the zero is in
        let acc = renamed(&correlated, 4, "v_add_f32_e32 v5, 0, v6", &[w1(5)], &[w1(6)]);
        let nine = renamed(&correlated, 5, "v_mov_b32_e32 v9, 1", &[w1(9)], &[]);
        assert_ne!(acc.defs[0].base, nine.defs[0].base);
    }

    #[test]
    fn a_covering_zero_keeps_the_register_of_the_accumulator_it_covers() {
        let prog = vec![
            i("v_mov_b32_e32 v5, 0", &[w1(5)], &[]),
            i("v_mov_b32_e32 v6, 2", &[w1(6)], &[]),
            i("s_bitcmp0_b32 s1, 3", &[], &[sg(1)]),
            i("s_cbranch_scc1 .L1", &[], &[]),
            i("v_add_f32_e32 v5, 0, v6", &[w1(5)], &[w1(6)]),
            lbl(".L1"),
            i("v_mov_b32_e32 v9, 1", &[w1(9)], &[]),
            i("s_bitcmp0_b32 s1, 3", &[], &[sg(1)]),
            i("s_cbranch_scc1 .L2", &[], &[]),
            i("v_add_f32_e32 v8, v5, v6", &[w1(8)], &[w1(5), w1(6)]),
            lbl(".L2"),
            i("v_add_f32_e32 v10, v9, v9", &[w1(10)], &[w1(9), w1(9)]),
            i("s_endpgm", &[], &[]),
        ];
        let reg = |m: &VgprMap, n: usize, text: &str, d: u8, u: &[u8]| {
            let uses: Vec<RegRef> = u.iter().map(|&x| w1(x)).collect();
            renamed(m, n, text, &[w1(d)], &uses).defs[0].base
        };
        let check = |m: &VgprMap| {
            let zero = reg(m, 0, "v_mov_b32_e32 v5, 0", 5, &[]);
            let acc = reg(m, 4, "v_add_f32_e32 v5, 0, v6", 5, &[6]);
            assert_eq!(zero, acc, "the zero defines the register the guarded read reads");
        };
        check(&vgpr_compaction(&recording(prog.clone()), &[]).unwrap());
        for windows in [false, true] {
            check(&alloc_spread(prog.clone(), Spread { banks: true, windows }));
            // both passes, as `compact_vgprs_as` runs them
            let rec = recording(prog.clone());
            let pre = allocate(&rec, &[], Mode::Spread(Spread { banks: true, windows }), None).unwrap();
            let mut second = Recording::default();
            for it in &rec.stream {
                match it {
                    StreamItem::Insn { insn, subs } => {
                        let r = pre.rewrite(subs[0], insn.clone()).unwrap();
                        second.hashes.push(text_hash(&r.text));
                        second.stream.push(StreamItem::Insn { insn: r, subs: *subs });
                    }
                    l => second.stream.push(l.clone()),
                }
            }
            let post = allocate(&second, &[], Mode::Compact, Some(&pre)).unwrap();
            check(&compose(&pre, &post, &rec.hashes).unwrap());
        }
    }

    #[test]
    fn spread_gives_each_reuse_of_a_temporary_its_own_register_inside_a_window() {
        // two rows decoded through one temporary v10: the scheduler would see a WAR/WAW chain between the rows
        let s = vec![
            i("v_mov_b32_e32 v1, s2", &[w1(1)], &[sg(2)]),
            i("v_mov_b32_e32 v2, s3", &[w1(2)], &[sg(3)]),
            i("v_add_f32_e32 v10, v1, v2", &[w1(10)], &[w1(1), w1(2)]),
            i("v_mul_f32_e32 v11, v10, v1", &[w1(11)], &[w1(10), w1(1)]),
            i("v_add_f32_e32 v10, v2, v2", &[w1(10)], &[w1(2), w1(2)]),
            i("v_mul_f32_e32 v12, v10, v2", &[w1(12)], &[w1(10), w1(2)]),
            i("s_endpgm", &[], &[]),
        ];
        let reg = |m: &VgprMap, n: usize, text: &str, d: u8, u: &[u8]| {
            let uses: Vec<RegRef> = u.iter().map(|&x| w1(x)).collect();
            renamed(m, n, text, &[w1(d)], &uses).defs[0].base
        };
        for (opts, apart) in [
            (Spread { banks: false, windows: true }, true),
            (Spread { banks: false, windows: false }, false),
        ] {
            let m = alloc_spread(s.clone(), opts);
            let a = reg(&m, 2, "v_add_f32_e32 v10, v1, v2", 10, &[1, 2]);
            let b = reg(&m, 4, "v_add_f32_e32 v10, v2, v2", 10, &[2, 2]);
            if apart {
                assert_ne!(a, b, "{opts:?}");
            }
            assert!(m.vgprs <= 32);
            assert!(!m.replays());
        }
    }

    #[test]
    fn spread_picks_banks_so_that_independent_adds_can_dual_issue() {
        use crate::arch::Arch;
        // both rows have the same banks: equal destination parity, equal src0 and src1 banks
        let s = vec![
            i("v_mov_b32_e32 v16, s2", &[w1(16)], &[sg(2)]),
            i("v_mov_b32_e32 v20, s3", &[w1(20)], &[sg(3)]),
            i("v_mov_b32_e32 v24, s4", &[w1(24)], &[sg(4)]),
            i("v_mov_b32_e32 v28, s5", &[w1(28)], &[sg(5)]),
            i("v_add_f32_e32 v4, v16, v24", &[w1(4)], &[w1(16), w1(24)]),
            i("v_add_f32_e32 v8, v20, v28", &[w1(8)], &[w1(20), w1(28)]),
            i("v_mov_b32_e32 v1, v4", &[w1(1)], &[w1(4)]),
            i("v_mov_b32_e32 v2, v8", &[w1(2)], &[w1(8)]),
            i("s_endpgm", &[], &[]),
        ];
        let m = alloc_spread(s, Spread { banks: true, windows: false });
        let a = renamed(&m, 4, "v_add_f32_e32 v4, v16, v24", &[w1(4)], &[w1(16), w1(24)]);
        let b = renamed(&m, 5, "v_add_f32_e32 v8, v20, v28", &[w1(8)], &[w1(20), w1(28)]);
        assert!(
            crate::valu_sched::pair(Arch::Gfx1201, &a, &b).is_some(),
            "`{}` and `{}` should pair: {:?}",
            a.text,
            b.text,
            m.stats
        );
        assert!(m.stats.rotated > 0 && m.stats.cand_ok == m.stats.cand_edges, "{:?}", m.stats);
    }

    #[test]
    fn composed_renaming_maps_the_emitters_names_to_the_final_ones() {
        let s = vec![
            i("v_mov_b32_e32 v16, s2", &[w1(16)], &[sg(2)]),
            i("v_add_f32_e32 v17, v16, v16", &[w1(17)], &[w1(16), w1(16)]),
            i("s_endpgm", &[], &[]),
        ];
        let rec = recording(s);
        let pre = allocate(&rec, &[], Mode::Spread(Spread { banks: false, windows: false }), None).unwrap();
        // the program as the second pass records it: renamed by `pre`
        let mut second = Recording::default();
        for it in &rec.stream {
            match it {
                StreamItem::Insn { insn, subs } => {
                    let r = pre.rewrite(subs[0], insn.clone()).unwrap();
                    second.hashes.push(text_hash(&r.text));
                    second.stream.push(StreamItem::Insn { insn: r, subs: *subs });
                }
                l => second.stream.push(l.clone()),
            }
        }
        let post = vgpr_compaction(&second, &[]).unwrap();
        let both = compose(&pre, &post, &rec.hashes).unwrap();
        assert!(both.replays());
        assert_eq!(both.vgprs, post.vgprs);
        for (n, ins) in rec.stream.iter().enumerate().filter_map(|(n, it)| match it {
            StreamItem::Insn { insn, .. } => Some((n, insn)),
            _ => None,
        }) {
            let via = post.rewrite(n, pre.rewrite(n, ins.clone()).unwrap()).unwrap();
            assert_eq!(both.rewrite(n, ins.clone()).unwrap().text, via.text);
        }
    }

    #[test]
    fn rewrite_refuses_a_different_instruction() {
        let s = vec![
            i("v_mov_b32_e32 v1, 1", &[v(1, 1)], &[]),
            i("s_endpgm", &[], &[]),
        ];
        let m = vgpr_compaction(&recording(s), &[]).unwrap();
        assert!(m
            .rewrite(
                0,
                Instruction::new("v_mov_b32_e32 v1, 2", vec![v(1, 1)], vec![])
            )
            .is_err());
    }
}
