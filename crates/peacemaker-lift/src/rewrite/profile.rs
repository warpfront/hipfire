// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Profiler client (architecture.md §4 `insert_before/after`, §6 "Profiler"): the
//! `peacemaker profile` instrumentation (`peacemaker-isa/src/profile.rs`, `f537e3fd1`) as a
//! script of core edits on a lifted kernel.
//!
//! The rules, register choice, site numbering and instruction sequences are the text
//! pass's; everything else is core's. Points are `Insert`s at cursors resolved on the
//! lifted kernel (the assembly source only names the anchors: rule prefixes and labels
//! match its instruction lines, which must be the kernel's instructions one for one);
//! windows are settled here exactly as the text pass does so site ids match, and core
//! refuses any cursor it still finds inside a window. The kernarg extension, VGPR/SGPR
//! counts and the `__pm_profile` rename are `Descriptor`/`Metadata` edits. `s_delay_alu`
//! hints whose producer a point pushes back are rewritten by core's hint maintenance,
//! not by the client.

use std::collections::BTreeSet;

use peacemaker_ir::cfg::BlockId;
use peacemaker_ir::edit::{Cursor, DescriptorChange, Edit, MetaChange};
use peacemaker_ir::effects::Control;
use peacemaker_ir::inst::{Abi, Inst, Kernel, Program, SymbolId};
use peacemaker_ir::metadata::Kernarg;
use peacemaker_ir::operand::{CachePolicy, Modifiers, Msg, Operand};
use peacemaker_ir::reg::{ClaimOwner, Kind, RegClaim, RegRef, Scope};
use peacemaker_ir::wait::Counter;

use super::build::{
    self, depctr, hw, imm, inst, op, s, s2, s4, simm16, smem_offset, ttmp, v, vmem_offset,
};
use super::{RewriteError, SourceMap};

pub const PROFILED_SUFFIX: &str = "__pm_profile";
pub const MAGIC: u32 = 0x504d_5046;
pub const HEADER_BYTES: u32 = 32;
pub const RECORD_BYTES: u32 = 8;
pub const EXIT_RECORD_BYTES: u32 = 16;
pub const EXIT_FLAG: u32 = 0x8000_0000;
pub const KERNARG_EXT_BYTES: u32 = 24;
const SGPR_LIMIT: u16 = 106;
const VGPR_GRANULE: u32 = 24;

/// One named point family (the text pass's `points.json` rule). Exactly one of `label`,
/// `before`, `after`; `first_after` keeps the first match after each match of that
/// prefix; `occurrences` keeps the listed 0-based matches.
#[derive(Clone, Debug, Default)]
pub struct Rule {
    pub name: String,
    pub label: Option<String>,
    pub before: Option<String>,
    pub after: Option<String>,
    pub first_after: Option<String>,
    pub occurrences: Option<Vec<usize>>,
}

/// The profile's registers: scratch VGPR, record pointer pair, timestamp, EXEC save, and
/// the entry-only temporaries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Registers {
    pub scratch_vgpr: u16,
    pub pointer: u16,
    pub timestamp: u16,
    pub exec_save: u16,
    pub entry_quad: u16,
    pub entry_pair: u16,
    pub realtime_pair: u16,
    pub entry_singles: [u16; 6],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Site {
    pub id: u32,
    pub rule: String,
    /// Kernel layout index the point lands before (the entry is 0).
    pub at: usize,
    pub moved: bool,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub profiled: String,
    pub registers: Registers,
    pub sites: Vec<Site>,
    pub script: Vec<Edit>,
}

/// A point position: before layout index `at`, and before a label there (end of the
/// previous block) when `before_label`.
#[derive(Clone, Copy, Debug)]
struct Gap {
    at: usize,
    before_label: bool,
}

fn client(reason: impl Into<String>) -> RewriteError {
    RewriteError::Client(reason.into())
}

fn name(i: &Inst) -> &'static str {
    super::mnemonic(i)
}

/// `(first, last)` layout positions of every `s_clause` / `s_delay_alu` window, as the
/// text pass reads them (a delay skip does not count other `s_delay_alu` lines).
fn windows(insts: &[&Inst]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (i, inst) in insts.iter().enumerate() {
        let last = match inst.effects.control {
            Control::Clause => i + 1 + usize::from(inst.mods.clause.unwrap_or(0)),
            Control::Delay => {
                let hint = inst.mods.delay.unwrap_or_default();
                let mut left = if hint.instid1 != 0 {
                    usize::from(hint.instskip)
                } else {
                    0
                };
                let mut t = i + 1;
                let mut j = t + 1;
                while left > 0 && j < insts.len() {
                    if insts[j].effects.control != Control::Delay {
                        t = j;
                        left -= 1;
                    }
                    j += 1;
                }
                t
            }
            _ => continue,
        };
        out.push((i, last.min(insts.len() - 1)));
    }
    out
}

/// Move a gap out of any window: up to the opener (`before`), down past the last member
/// or target (`after`, labels).
fn settle(
    mut gap: Gap,
    down: bool,
    windows: &[(usize, usize)],
    labels: &BTreeSet<usize>,
) -> (Gap, bool) {
    let mut moved = false;
    while let Some(&(s, e)) = windows.iter().find(|&&(s, e)| gap.at > s && gap.at <= e) {
        gap = if down {
            Gap {
                at: e + 1,
                before_label: labels.contains(&(e + 1)),
                ..gap
            }
        } else {
            Gap {
                at: s,
                before_label: false,
                ..gap
            }
        };
        moved = true;
    }
    (gap, moved)
}

/// The text pass's register choice: the highest VGPR below the occupancy ceiling (the
/// 24-VGPR wave32 granule above `.vgpr_count`) the kernel never names; SGPRs past the
/// kernel inputs, persistent ones never named by the kernel.
fn pick(kernel: &Kernel, vgpr_count: u32, inputs: u16) -> Result<Registers, RewriteError> {
    let mut used_v = BTreeSet::new();
    let mut used_s = BTreeSet::new();
    for &id in &kernel.body.layout {
        for operand in &kernel.body.insts.get(id).expect("laid out").operands {
            let (Operand::Reg(r) | Operand::Half(r, _)) = operand else {
                continue;
            };
            let set = match r.kind {
                Kind::V => &mut used_v,
                Kind::S => &mut used_s,
                Kind::Ttmp => continue,
            };
            set.extend(r.base..r.base + u16::from(r.len));
        }
    }
    let limit = (vgpr_count.div_ceil(VGPR_GRANULE) * VGPR_GRANULE).min(256) as u16;
    let scratch_vgpr = (0..limit)
        .rev()
        .find(|n| !used_v.contains(n))
        .ok_or_else(|| client(format!("no unreferenced VGPR below v{limit}")))?;
    let mut taken: BTreeSet<u16> = (0..inputs).collect();
    let mut take = |n: u16, align: u16, unused_only: bool| -> Result<u16, RewriteError> {
        let free = |unused: bool, taken: &BTreeSet<u16>| {
            (0..SGPR_LIMIT).step_by(usize::from(align)).find(|&b| {
                b + n <= SGPR_LIMIT
                    && (b..b + n).all(|r| !taken.contains(&r) && (!unused || !used_s.contains(&r)))
            })
        };
        let b = free(unused_only, &taken)
            .or_else(|| {
                if unused_only {
                    None
                } else {
                    free(true, &taken)
                }
            })
            .ok_or_else(|| client("not enough SGPRs for the profile record"))?;
        taken.extend(b..b + n);
        Ok(b)
    };
    let pointer = take(2, 2, true)?;
    let timestamp = take(1, 1, true)?;
    let exec_save = take(1, 1, true)?;
    let entry_quad = take(4, 4, false)?;
    let entry_pair = take(2, 2, false)?;
    let realtime_pair = take(2, 2, false)?;
    let mut entry_singles = [0; 6];
    for s in &mut entry_singles {
        *s = take(1, 1, false)?;
    }
    Ok(Registers {
        scratch_vgpr,
        pointer,
        timestamp,
        exec_save,
        entry_quad,
        entry_pair,
        realtime_pair,
        entry_singles,
    })
}

fn nt_store() -> Modifiers {
    Modifiers {
        cpol: CachePolicy {
            th: 1,
            ..CachePolicy::default()
        },
        ..Modifiers::default()
    }
}

fn wait_alu(raw: u16) -> Result<Inst, RewriteError> {
    Ok(op("s_wait_alu", vec![simm16(raw)])?)
}

fn getreg(dst: u16, id: u16) -> Result<Inst, RewriteError> {
    Ok(op("s_getreg_b32", vec![s(dst), build::hwreg(id, 0, 32)])?)
}

fn writelane(r: &Registers, src: Operand, lane: i64) -> Result<Inst, RewriteError> {
    Ok(op(
        "v_writelane_b32",
        vec![v(r.scratch_vgpr), src, imm(lane)],
    )?)
}

fn realtime(r: &Registers) -> Result<Inst, RewriteError> {
    Ok(op(
        "s_sendmsg_rtn_b64",
        vec![
            s2(r.realtime_pair),
            Operand::SendMsg(Msg {
                id: build::MSG_RTN_GET_REALTIME,
                op: 0,
            }),
        ],
    )?)
}

fn kmcnt0() -> Result<Inst, RewriteError> {
    Ok(inst(
        "s_wait_kmcnt",
        vec![simm16(0)],
        build::wait_mods(Counter::Km, 0),
    )?)
}

/// EXEC to `lanes` lanes, bump the pointer, store the scratch VGPR's lanes, restore EXEC.
fn store(r: &Registers, lanes: u32, bytes: u32, va_sdst: bool) -> Result<Vec<Inst>, RewriteError> {
    let mut out = Vec::new();
    if va_sdst {
        out.push(wait_alu(depctr::VA_SDST_0)?);
    }
    out.push(op("s_mov_b32", vec![s(r.exec_save), build::exec_lo()])?);
    out.push(op(
        "s_mov_b32",
        vec![build::exec_lo(), imm((1i64 << lanes) - 1)],
    )?);
    out.push(op(
        "s_add_nc_u64",
        vec![s2(r.pointer), s2(r.pointer), imm(i64::from(bytes))],
    )?);
    out.push(inst(
        "global_store_addtid_b32",
        vec![
            v(r.scratch_vgpr),
            s2(r.pointer),
            vmem_offset(-(bytes as i32)),
        ],
        nt_store(),
    )?);
    out.push(op("s_mov_b32", vec![build::exec_lo(), s(r.exec_save)])?);
    Ok(out)
}

fn point(r: &Registers, id: u32, va_sdst: bool) -> Result<Vec<Inst>, RewriteError> {
    let mut out = vec![
        getreg(r.timestamp, hw::SHADER_CYCLES_LO)?,
        wait_alu(depctr::SA_SDST_0_VM_VSRC_0)?,
        writelane(r, s(r.timestamp), 0)?,
        writelane(r, imm(i64::from(id)), 1)?,
    ];
    out.extend(store(r, 2, RECORD_BYTES, va_sdst)?);
    Ok(out)
}

fn entry(r: &Registers, ext: u32, va_sdst: bool) -> Result<Vec<Inst>, RewriteError> {
    let [w, h1, h2, ch, z, l] = r.entry_singles;
    let (q, rr, rt, p) = (r.entry_quad, r.entry_pair, r.realtime_pair, r.pointer);
    let sop2 = |name: &str, d: u16, a: Operand, b: Operand| -> Result<Inst, RewriteError> {
        Ok(op(name, vec![s(d), a, b])?)
    };
    let mut out = vec![
        op("s_load_b128", vec![s4(q), s2(0), smem_offset(ext as i32)])?,
        op(
            "s_load_b64",
            vec![s2(rr), s2(0), smem_offset((ext + 16) as i32)],
        )?,
        getreg(h1, hw::HW_ID1)?,
        getreg(h2, hw::HW_ID2)?,
        getreg(ch, hw::SHADER_CYCLES_HI)?,
        op("v_readfirstlane_b32", vec![s(w), v(0)])?,
        realtime(r)?,
        kmcnt0()?,
        wait_alu(depctr::VA_SDST_0)?,
        sop2("s_and_b32", w, s(w), imm(0x3ff))?,
        sop2("s_lshr_b32", w, s(w), imm(5))?,
        sop2("s_lshr_b32", z, ttmp(7), imm(16))?,
        sop2("s_mul_i32", z, s(z), s(rr + 1))?,
        sop2("s_and_b32", l, ttmp(7), imm(0xffff))?,
        sop2("s_add_co_i32", z, s(z), s(l))?,
        sop2("s_mul_i32", z, s(z), s(rr))?,
        sop2("s_add_co_i32", z, s(z), ttmp(9))?,
        sop2("s_mul_i32", l, s(z), s(q + 3))?,
        sop2("s_add_co_i32", l, s(l), s(w))?,
        sop2("s_mul_i32", p, s(l), s(q + 2))?,
        sop2("s_mul_hi_u32", p + 1, s(l), s(q + 2))?,
        sop2("s_add_co_u32", p, s(p), s(q))?,
        sop2("s_add_co_ci_u32", p + 1, s(p + 1), s(q + 1))?,
        wait_alu(depctr::SA_SDST_0)?,
        writelane(r, imm(i64::from(MAGIC)), 0)?,
    ];
    for (lane, src) in [h1, h2, z, w, rt, rt + 1, ch].into_iter().enumerate() {
        out.push(writelane(r, s(src), lane as i64 + 1)?);
    }
    out.extend(store(r, 8, HEADER_BYTES, va_sdst)?);
    out.extend(point(r, 0, va_sdst)?);
    Ok(out)
}

fn exit(r: &Registers, id: u32, va_sdst: bool) -> Result<Vec<Inst>, RewriteError> {
    let mut out = vec![
        getreg(r.timestamp, hw::SHADER_CYCLES_LO)?,
        realtime(r)?,
        kmcnt0()?,
        wait_alu(depctr::SA_SDST_0_VM_VSRC_0)?,
        writelane(r, s(r.timestamp), 0)?,
        writelane(r, imm(i64::from(id | EXIT_FLAG)), 1)?,
        writelane(r, s(r.realtime_pair), 2)?,
        writelane(r, s(r.realtime_pair + 1), 3)?,
    ];
    out.extend(store(r, 4, EXIT_RECORD_BYTES, va_sdst)?);
    Ok(out)
}

/// Instrument `symbol` of `program` (lifted from the object assembled from `source`) with
/// the point `rules`; returns the core script, the sites and the registers.
pub fn plan(
    program: &Program,
    symbol: &SymbolId,
    source: &str,
    rules: &[Rule],
) -> Result<Plan, RewriteError> {
    let kernel = program
        .kernels
        .iter()
        .find(|k| &k.symbol == symbol)
        .ok_or_else(|| client(format!("no kernel {}", symbol.0)))?;
    let Abi::Hsa {
        descriptor,
        metadata,
    } = &kernel.abi
    else {
        return Err(client("the profiler needs an HSA kernel"));
    };
    if !descriptor.kernel_code_properties.wave32() {
        return Err(client("the profiler supports wave32 kernels only"));
    }
    let body = &kernel.body;
    let insts: Vec<&Inst> = body
        .layout
        .iter()
        .map(|&id| body.insts.get(id).expect("laid out"))
        .collect();

    // The source names the anchors: its instruction lines of this kernel are the
    // kernel's instructions one for one.
    let map = SourceMap::new(kernel, source)?;
    let lines = &map.lines;
    let label_at: BTreeSet<usize> = map.labels.values().copied().filter(|&at| at > 0).collect();

    // Kernel facts the text pass reads from the source.
    let va_sdst = insts.iter().any(|i| name(i).starts_with("v_cmpx"));
    for (k, i) in insts.iter().enumerate() {
        let store_count = i
            .mods
            .wait
            .as_ref()
            .and_then(|w| w.per_counter[Counter::Store as usize]);
        if store_count.is_some_and(|n| n != 0) {
            return Err(client(format!(
                "instruction {k}: a nonzero store wait would count profile stores"
            )));
        }
    }
    let rsrc2 = descriptor.compute_pgm_rsrc2.0;
    let inputs = (rsrc2 >> 1 & 0x1f)
        + (rsrc2 >> 7 & 1)
        + (rsrc2 >> 8 & 1)
        + (rsrc2 >> 9 & 1)
        + (rsrc2 >> 10 & 1)
        + (rsrc2 & 1);
    let meta = &metadata.parsed;
    let registers = pick(kernel, meta.vgpr_count, inputs as u16)?;

    // Gaps: rules, then every s_endpgm (above LLVM's `s_nop 0` + VGPR-dealloc epilogue).
    let windows = windows(&insts);
    let mut gaps: Vec<(Gap, String, bool)> = Vec::new();
    for rule in rules {
        let (prefix, down) = match (&rule.label, &rule.before, &rule.after) {
            (Some(label), None, None) => {
                let at = map.at_label(label)?;
                let (gap, moved) = settle(
                    Gap {
                        at,
                        before_label: false,
                    },
                    true,
                    &windows,
                    &label_at,
                );
                gaps.push((gap, rule.name.clone(), moved));
                continue;
            }
            (None, Some(p), None) => (p.as_str(), false),
            (None, None, Some(p)) => (p.as_str(), true),
            _ => {
                return Err(client(format!(
                    "rule {}: give exactly one of label/before/after",
                    rule.name
                )))
            }
        };
        let mut hits: Vec<usize> = (0..lines.len())
            .filter(|&k| lines[k].starts_with(prefix))
            .collect();
        if let Some(mark) = &rule.first_after {
            let marks: Vec<usize> = (0..lines.len())
                .filter(|&k| lines[k].starts_with(mark.as_str()))
                .collect();
            hits = marks
                .iter()
                .enumerate()
                .filter_map(|(n, &m)| {
                    let limit = marks.get(n + 1).copied().unwrap_or(lines.len());
                    hits.iter().copied().find(|&h| h > m && h < limit)
                })
                .collect();
        }
        if let Some(keep) = &rule.occurrences {
            if keep.iter().any(|&k| k >= hits.len()) {
                return Err(client(format!(
                    "rule {}: occurrence out of range",
                    rule.name
                )));
            }
            hits = keep.iter().map(|&k| hits[k]).collect();
        }
        if hits.is_empty() {
            return Err(client(format!(
                "rule {}: no instruction starts with {prefix:?}",
                rule.name
            )));
        }
        for h in hits {
            let gap = if down {
                Gap {
                    at: h + 1,
                    before_label: label_at.contains(&(h + 1)),
                }
            } else {
                Gap {
                    at: h,
                    before_label: false,
                }
            };
            let (gap, moved) = settle(gap, down, &windows, &label_at);
            gaps.push((gap, rule.name.clone(), moved));
        }
    }
    for k in (0..insts.len()).filter(|&k| name(insts[k]) == "s_endpgm") {
        if windows.iter().any(|&(s, e)| k > s && k <= e) {
            return Err(client("s_endpgm inside a scheduling window"));
        }
        let mut at = k;
        let is_dealloc = |j: usize| {
            name(insts[j]) == "s_sendmsg"
                && insts[j].operands.first() == Some(&Operand::SendMsg(Msg { id: 3, op: 0 }))
        };
        if at > 0 && is_dealloc(at - 1) {
            at -= 1;
            if at > 0
                && name(insts[at - 1]) == "s_nop"
                && insts[at - 1].operands.first() == Some(&simm16(0))
            {
                at -= 1;
            }
        }
        if (at + 1..=k).any(|j| label_at.contains(&j)) {
            return Err(client("label between the exit record and s_endpgm"));
        }
        gaps.push((
            Gap {
                at,
                before_label: false,
            },
            "exit".into(),
            at != k,
        ));
    }
    gaps.sort_by_key(|(g, _, _)| (g.at, !g.before_label));

    // Descriptor/metadata edits, then the points in site order, then the rename.
    let kernel_sym = symbol.clone();
    let ext = descriptor.kernarg_size.div_ceil(8) * 8;
    let r = &registers;
    let new_vgpr = meta.vgpr_count.max(u32::from(r.scratch_vgpr) + 1);
    let ours = [
        r.pointer + 1,
        r.timestamp,
        r.exec_save,
        r.entry_quad + 3,
        r.entry_pair + 1,
        r.realtime_pair + 1,
    ]
    .into_iter()
    .chain(r.entry_singles)
    .max()
    .unwrap_or(0);
    let new_sgpr = meta.sgpr_count.max(u32::from(ours) + 1);
    let arg = |name: &str, offset: u32, size: u32, kind: &str, space: Option<&str>| Kernarg {
        name: name.into(),
        size,
        offset,
        value_kind: kind.into(),
        address_space: space.map(Into::into),
    };
    let args = vec![
        arg(
            "pm_profile_records",
            ext,
            8,
            "global_buffer",
            Some("global"),
        ),
        arg("pm_slot_bytes", ext + 8, 4, "by_value", None),
        arg("pm_waves_per_wg", ext + 12, 4, "by_value", None),
        arg("pm_grid_x", ext + 16, 4, "by_value", None),
        arg("pm_grid_y", ext + 20, 4, "by_value", None),
    ];
    let claim = |name: &str, reg: RegRef| RegClaim {
        name: name.into(),
        reg,
        scope: Scope::Whole,
        owner: ClaimOwner::Builder,
    };
    let sreg = |base: u16, len: u8| RegRef {
        kind: Kind::S,
        base,
        len,
    };
    let claims = vec![
        claim("pm_pointer", sreg(r.pointer, 2)),
        claim("pm_timestamp", sreg(r.timestamp, 1)),
        claim("pm_exec_save", sreg(r.exec_save, 1)),
        claim(
            "pm_record",
            RegRef {
                kind: Kind::V,
                base: r.scratch_vgpr,
                len: 1,
            },
        ),
    ];
    let mut script = vec![
        Edit::Descriptor {
            kernel: kernel_sym.clone(),
            change: DescriptorChange::KernargSize(ext + KERNARG_EXT_BYTES),
        },
        Edit::Metadata {
            kernel: kernel_sym.clone(),
            change: MetaChange::AppendArgs(args),
        },
    ];
    // Counts the text pass leaves unchanged (`max(old, new)`) are not edited.
    if new_vgpr != meta.vgpr_count {
        script.push(Edit::Descriptor {
            kernel: kernel_sym.clone(),
            change: DescriptorChange::NextFreeVgpr(new_vgpr),
        });
        script.push(Edit::Metadata {
            kernel: kernel_sym.clone(),
            change: MetaChange::SetVgprCount(new_vgpr),
        });
    }
    if new_sgpr != meta.sgpr_count {
        script.push(Edit::Metadata {
            kernel: kernel_sym.clone(),
            change: MetaChange::SetSgprCount(new_sgpr),
        });
    }
    // Every insertion carries the claims: each step is its own transaction.
    script.push(Edit::Insert {
        at: Cursor::before(BlockId(0), body.layout[0]),
        insts: entry(r, ext, va_sdst)?,
        claims: claims.clone(),
    });
    let block_of = |at: usize| {
        body.blocks
            .iter()
            .position(|b| b.range.0 <= at && at < b.range.1)
            .expect("laid out")
    };
    let mut sites = vec![Site {
        id: 0,
        rule: "entry".into(),
        at: 0,
        moved: false,
    }];
    for (k, (gap, rule, moved)) in gaps.iter().enumerate() {
        let id = k as u32 + 1;
        if gap.at >= body.layout.len() {
            return Err(client(format!(
                "rule {rule} lands after the last instruction"
            )));
        }
        let block = block_of(gap.at);
        let cursor = if gap.before_label && body.blocks[block].range.0 == gap.at {
            if block == 0 {
                return Err(client(format!(
                    "rule {rule}: nothing precedes the entry label"
                )));
            }
            Cursor::end(BlockId(block - 1))
        } else {
            Cursor::before(BlockId(block), body.layout[gap.at])
        };
        let seq = if rule == "exit" {
            exit(r, id, va_sdst)?
        } else {
            point(r, id, va_sdst)?
        };
        script.push(Edit::Insert {
            at: cursor,
            insts: seq,
            claims: claims.clone(),
        });
        sites.push(Site {
            id,
            rule: rule.clone(),
            at: gap.at,
            moved: *moved,
        });
    }
    let profiled = format!("{}{PROFILED_SUFFIX}", symbol.0);
    script.push(Edit::Metadata {
        kernel: kernel_sym,
        change: MetaChange::Rename(profiled.clone()),
    });
    Ok(Plan {
        profiled,
        registers,
        sites,
        script,
    })
}
