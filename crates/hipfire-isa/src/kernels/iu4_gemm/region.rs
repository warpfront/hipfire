//! Imported hipcc region: the SiLU epilogue value `h = g / (1 + expf(-g)) * u`.
//!
//! `expf` and the IEEE division are lowered by hipcc into a fixed VALU DAG
//! (range-reduced `v_exp_f32`, overflow/underflow selects, the
//! `v_div_scale`/`v_div_fmas`/`v_div_fixup` sequence). Bit-exactness needs
//! that DAG verbatim, so it is not re-derived: it is sliced out of hipcc's own
//! disassembly (`slice_silu`), committed as `kernels/iu4_gemm.silu.region.s`,
//! parsed back into a register-renamed template (`Region::parse`) and
//! instantiated per element. Only the registers, the interleaving of
//! independent elements, the VOPD packing of their VOP2 operations and hazard
//! waits are chosen here; every element keeps its ops in golden order.
use super::op;
use crate::{Builder, reg::{Kind, RegRef}, vopd::{self, VopdF32, VopdOp}};
use std::collections::{BTreeMap, BTreeSet};

pub const SILU_GOLDEN: &str = include_str!("../../../kernels/iu4_gemm.silu.region.s");
/// Temporaries and lane-mask SGPRs one element needs.
pub const SILU_TEMPS: usize = 7;
pub const SILU_MASKS: usize = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Operand { G, U, Temp(usize), Mask(usize), Fixed(String) }
#[derive(Clone, Debug, PartialEq, Eq)]
struct Op { mnemonic: String, operands: Vec<(bool, Operand)>, vcc_def: bool, vcc_use: bool, dual: Option<VopdF32> }

/// A parsed, register-renamed region: two VGPR inputs, one VGPR output
/// (the last definition), `temps` VGPR and `masks` SGPR temporaries, and
/// VCC as an implicit capability.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region { ops: Vec<Op>, pub temps: usize, pub masks: usize }

/// VOPD halves issue as their VOP2 forms; the operation is identical.
fn standalone(mnemonic: &str) -> String {
    match mnemonic.strip_prefix("v_dual_") {
        Some(rest) => format!("v_{rest}_e32"),
        None => mnemonic.to_owned(),
    }
}
/// The VOPD opcode a VOP2 region op issues as when its operands fit a VOPD
/// half: VGPR destination, unmodified sources, VGPR src1. VOP1/VOP3 forms,
/// compares, selects and the division sequence always issue alone.
fn dual_form(mnemonic: &str, operands: &[(bool, Operand)]) -> Option<VopdF32> {
    let op = match mnemonic {
        "v_add_f32_e32" => VopdF32::Add,
        "v_sub_f32_e32" => VopdF32::Sub,
        "v_subrev_f32_e32" => VopdF32::Subrev,
        "v_mul_f32_e32" => VopdF32::Mul,
        "v_fmac_f32_e32" => VopdF32::Fmac,
        _ => return None,
    };
    let vgpr = |o: &Operand| matches!(o, Operand::G | Operand::U | Operand::Temp(_));
    match operands {
        [(false, Operand::Temp(_)), (false, _), (false, src1)] if vgpr(src1) => Some(op),
        _ => None,
    }
}
fn register(token: &str) -> Option<(bool, char, u16)> {
    let (neg, t) = token.strip_prefix('-').map_or((false, token), |t| (true, t));
    let kind = t.chars().next()?;
    if kind != 'v' && kind != 's' { return None }
    t[1..].parse().ok().map(|n| (neg, kind, n))
}
fn instruction(line: &str) -> Option<(&str, &str)> {
    let code = line.split(';').next()?.split("//").next()?.trim();
    if code.is_empty() { return None }
    Some(code.split_once(' ').unwrap_or((code, "")))
}

impl Region {
    /// Parse a sliced hipcc sequence. Inputs are the two VGPRs read before any
    /// definition: `g` (read first) and `u`. Every other register maps
    /// one-to-one to a temporary, preserving hipcc's own reuse pattern.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut ops = Vec::new();
        let (mut vmap, mut smap) = (BTreeMap::new(), BTreeMap::new());
        let mut inputs: Vec<u16> = Vec::new();
        let mut defined = BTreeSet::new();
        for line in text.lines() {
            let Some((mnemonic, args)) = instruction(line) else { continue };
            if mnemonic == "s_wait_alu" {
                ops.push(Op { mnemonic: mnemonic.into(), operands: vec![(false, Operand::Fixed(args.into()))], vcc_def: false, vcc_use: false, dual: None });
                continue;
            }
            let tokens: Vec<_> = args.split(',').map(str::trim).collect();
            let mut operands = Vec::new();
            for (i, t) in tokens.iter().enumerate() {
                let Some((neg, kind, n)) = register(t) else { operands.push((false, Operand::Fixed((*t).into()))); continue };
                let is_def = i == 0;
                let slot = if kind == 's' {
                    if !is_def && !smap.contains_key(&n) { return Err(format!("mask s{n} read before definition")) }
                    let next = smap.len();
                    Operand::Mask(*smap.entry(n).or_insert(next))
                } else if !is_def && !defined.contains(&n) {
                    if !inputs.contains(&n) { inputs.push(n) }
                    if inputs.len() > 2 { return Err(format!("region reads a third live-in v{n}")) }
                    if inputs[0] == n { Operand::G } else { Operand::U }
                } else {
                    if is_def { defined.insert(n); }
                    let next = vmap.len();
                    Operand::Temp(*vmap.entry(n).or_insert(next))
                };
                operands.push((neg, slot));
            }
            let vcc_def = tokens.get(1) == Some(&"vcc_lo");
            let vcc_use = mnemonic == "v_div_fmas_f32";
            let mnemonic = standalone(mnemonic);
            let dual = dual_form(&mnemonic, &operands);
            ops.push(Op { mnemonic, operands, vcc_def, vcc_use, dual });
        }
        if inputs.len() != 2 { return Err("region must read exactly two live-ins".into()) }
        match ops.last() { Some(op) if matches!(op.operands.first(), Some((false, Operand::Temp(_)))) => {}, _ => return Err("region must end in a VGPR definition".into()) }
        Ok(Self { ops, temps: vmap.len(), masks: smap.len() })
    }

    pub fn silu() -> Result<Self, String> { Self::parse(SILU_GOLDEN) }
    pub fn len(&self) -> usize { self.ops.len() }
    pub fn is_empty(&self) -> bool { self.ops.is_empty() }
    /// Mnemonic sequence (for census and tests).
    pub fn mnemonics(&self) -> Vec<&str> { self.ops.iter().map(|o| o.mnemonic.as_str()).collect() }
}

/// Registers bound to one instance of the region.
#[derive(Clone, Debug)]
pub struct Binding { pub g: u8, pub u: u8, pub out: u8, pub temps: Vec<u8>, pub masks: Vec<u8> }

/// One rendered instance op: its standalone text, register effects, and its
/// typed VOPD half (with the half's own text) when it has one.
struct Issue { text: String, defs: Vec<RegRef>, uses: Vec<RegRef>, dual: Option<(VopdOp, String)> }

/// A src0 constant as the VOPD literal rule sees it. Hex and non-inline
/// integers are literals, small integers are inline; inline floats (`1.0`)
/// are counted as literals of their bits, which can only reject a legal
/// pair, never admit an illegal one.
fn constant(text: &str) -> Option<vopd::Operand> {
    if let Some(hex) = text.strip_prefix("0x") { return u32::from_str_radix(hex, 16).ok().map(vopd::Operand::Lit) }
    if let Ok(n) = text.parse::<i64>() {
        return Some(if (-16..=64).contains(&n) { vopd::Operand::Inline(n as i8) } else { vopd::Operand::Lit(n as u32) });
    }
    text.parse::<f32>().ok().map(|x| vopd::Operand::Lit(x.to_bits()))
}

fn render(op: &Op, bind: &Binding, last: bool) -> Issue {
    let mut text = op.mnemonic.clone();
    let (mut defs, mut uses) = (Vec::new(), Vec::new());
    let mut spelled = Vec::new();
    let mut src0 = None;
    for (i, (neg, operand)) in op.operands.iter().enumerate() {
        let vreg = |n: u8| (format!("v{n}"), Some(RegRef { kind: Kind::V, base: n, len: 1 }));
        let (name, reg) = match operand {
            Operand::G => vreg(bind.g),
            Operand::U => vreg(bind.u),
            Operand::Temp(n) => vreg(if last && i == 0 { bind.out } else { bind.temps[*n] }),
            Operand::Mask(n) => (format!("s{}", bind.masks[*n]), Some(RegRef { kind: Kind::S, base: bind.masks[*n], len: 1 })),
            Operand::Fixed(t) => (t.clone(), None),
        };
        if i == 1 {
            src0 = match (operand, reg) {
                (Operand::Mask(_), Some(r)) => Some(vopd::Operand::S(r.base)),
                (_, Some(r)) => Some(vopd::Operand::V(r.base)),
                (_, None) => constant(&name),
            };
        }
        text.push_str(if i == 0 { " " } else { ", " });
        if *neg { text.push('-') }
        text.push_str(&name);
        spelled.push(name);
        if let Some(r) = reg {
            if i == 0 && op.mnemonic != "s_wait_alu" { defs.push(r) } else { uses.push(r) }
        }
    }
    if op.mnemonic.starts_with("v_fmac") { uses.extend(defs.iter().copied()) }
    let dual = match (op.dual, src0, defs.first()) {
        (Some(kind), Some(src0), Some(dst)) => {
            let base = op.mnemonic.strip_prefix("v_").and_then(|m| m.strip_suffix("_e32")).unwrap_or(&op.mnemonic);
            let src1 = match &op.operands[2].1 { Operand::G => bind.g, Operand::U => bind.u, Operand::Temp(n) => bind.temps[*n], _ => unreachable!("dual_form admits VGPR src1 only") };
            Some((VopdOp { op: kind, dst: dst.base, src0, src1 }, format!("v_dual_{base} {}", spelled.join(", "))))
        }
        _ => None,
    };
    Issue { text, defs, uses, dual }
}

/// The typed VOPD packet of two instance ops, when legal: both have VOPD
/// halves, `vopd::validate_pair` accepts their parity/bank/literal use, and
/// neither half reads or writes the other's destination.
fn packet(arch: crate::Arch, a: &Issue, b: &Issue) -> Option<(String, Vec<RegRef>, Vec<RegRef>)> {
    let ((x, x_text), (y, y_text)) = (a.dual.as_ref()?, b.dual.as_ref()?);
    vopd::validate_pair(arch, *x, *y).ok()?;
    let touches = |defs: &[RegRef], other: &Issue| defs.iter().any(|d| other.uses.iter().chain(&other.defs).any(|r| r.overlaps(*d)));
    if touches(&a.defs, b) || touches(&b.defs, a) { return None }
    Some((format!("{x_text} :: {y_text}"), [a.defs.clone(), b.defs.clone()].concat(), [a.uses.clone(), b.uses.clone()].concat()))
}

/// Issue-model latencies, in issue slots, used only to rank ready work: the
/// hardware interlocks VGPR dependences, and SGPR/VCC hazards have explicit
/// waits, so these numbers never decide correctness.
const LATENCY: u32 = 5;
const TRANS_LATENCY: u32 = 10;
fn latency(mnemonic: &str) -> u32 {
    let trans = ["v_exp_", "v_log_", "v_rcp_", "v_rsq_", "v_sqrt_", "v_sin_", "v_cos_"].iter().any(|p| mnemonic.starts_with(p));
    if trans { TRANS_LATENCY } else { LATENCY }
}

/// Emit instances of `region` for independent elements; every instance
/// issues its own ops in golden order.
///
/// - Instances `2k` and `2k + 1` are VOPD partners. When both stand at the
///   same op and `packet` accepts it, the two issue as one VOPD packet; an
///   instance waits for a partner that is behind it, unless it holds the VCC
///   window.
/// - An instance starts only after every earlier instance sharing one of its
///   registers has issued its last op, so register slots are reused in order.
/// - An instance may not define VCC while another instance's VCC window (its
///   `v_div_scale` to its `v_div_fmas`) is open.
/// - Before an instruction reads a lane mask written by VALU since the last
///   fence, a `va_sdst` wait is issued.
/// - Each slot takes the first-listed issuable op (or packet) whose sources
///   are ready under the fixed latency model, else the one ready soonest.
pub fn emit_interleaved(b: &mut Builder, region: &Region, binds: &[Binding]) -> Result<(), String> {
    let (n, len, arch) = (binds.len(), region.ops.len(), b.spec.arch);
    let owned: Vec<BTreeSet<(bool, u8)>> = binds.iter().map(|bind| {
        [bind.g, bind.u, bind.out].into_iter().chain(bind.temps.iter().copied()).map(|r| (false, r))
            .chain(bind.masks.iter().map(|&m| (true, m))).collect()
    }).collect();
    let after: Vec<Vec<usize>> = (0..n).map(|k| (0..k).filter(|&e| !owned[e].is_disjoint(&owned[k])).collect()).collect();
    let issue = |s: usize, i: usize| render(&region.ops[i], &binds[s], i + 1 == len);
    let mut next = vec![0usize; n];
    let mut open: Option<usize> = None;
    let mut unfenced: BTreeSet<u8> = BTreeSet::new();
    let mut ready: BTreeMap<(bool, u8), u32> = BTreeMap::new();
    let mut vcc_ready = 0u32;
    let mut clock = 0u32;
    while next.iter().any(|&i| i < len) {
        let live = |s: usize| next[s] < len && (next[s] > 0 || after[s].iter().all(|&e| next[e] == len));
        let ready_at = |s: usize| {
            let op = &region.ops[next[s]];
            let regs = issue(s, next[s]).uses.iter().flat_map(|r| (0..r.len).map(move |k| (r.kind == Kind::S, r.base + k)))
                .map(|key| ready.get(&key).copied().unwrap_or(0)).max().unwrap_or(0);
            if op.mnemonic == "s_wait_alu" { regs.max(vcc_ready) } else { regs }
        };
        let mut groups: Vec<(Vec<usize>, u32)> = Vec::new();
        for s in (0..n).filter(|&s| live(s)) {
            let i = next[s];
            let op = &region.ops[i];
            if op.vcc_def && open.is_some_and(|o| o != s) { continue }
            let p = s ^ 1;
            if op.dual.is_some() && open != Some(s) && p < n && next[p] <= i && packet(arch, &issue(s, i), &issue(p, i)).is_some() {
                // Same op for both partners: one packet, listed once. A
                // partner that is behind (or not yet started) is waited for.
                if next[p] == i && live(p) && s < p { groups.push((vec![s, p], ready_at(s).max(ready_at(p)))) }
                continue;
            }
            groups.push((vec![s], ready_at(s)));
        }
        let (members, at) = groups.iter().find(|(_, t)| *t <= clock).or_else(|| groups.iter().min_by_key(|(_, t)| *t))
            .cloned().ok_or("region interleave deadlock")?;
        clock = clock.max(at);
        let issues: Vec<Issue> = members.iter().map(|&s| issue(s, next[s])).collect();
        let (text, defs, uses) = match issues.as_slice() {
            [a, b] => packet(arch, a, b).ok_or("VOPD packet rejected after selection")?,
            [a] => (a.text.clone(), a.defs.clone(), a.uses.clone()),
            _ => unreachable!("groups hold one op or one packet"),
        };
        if uses.iter().any(|r| r.kind == Kind::S && unfenced.contains(&r.base)) {
            op_fence(b)?;
            unfenced.clear();
        }
        op(b, text, &defs, &uses)?;
        for r in defs.iter().filter(|r| r.kind == Kind::S) { unfenced.insert(r.base); }
        let lat = if members.len() == 2 { LATENCY } else { latency(&region.ops[next[members[0]]].mnemonic) };
        for r in &defs { for k in 0..r.len { ready.insert((r.kind == Kind::S, r.base + k), clock + lat); } }
        for &s in &members {
            let op = &region.ops[next[s]];
            if op.vcc_def { open = Some(s); vcc_ready = clock + lat }
            if op.vcc_use { open = None }
            next[s] += 1;
        }
        clock += 1;
    }
    Ok(())
}
fn op_fence(b: &mut Builder) -> Result<(), String> { op(b, "s_wait_alu depctr_va_sdst(0)", &[], &[]) }

/// Slice the SiLU region out of an llvm-objdump listing of hipcc's gate/up
/// symbol: the backward dataflow slice of the first value stored after the
/// first `v_div_fixup_f32`, stopping at the two accumulator live-ins, with
/// the `va_vcc` hazard wait hipcc placed inside it.
pub fn slice_silu(disassembly: &str, symbol: &str) -> Result<String, String> {
    let mut stream: Vec<(String, String)> = Vec::new();
    let mut inside = false;
    for line in disassembly.lines() {
        let t = line.trim();
        if t.ends_with(">:") && t.contains(" <") {
            if inside { break }
            inside = t.ends_with(&format!("<{symbol}>:"));
            continue;
        }
        if !inside { continue }
        let Some((code, _)) = line.split_once("//") else { continue };
        for part in code.trim().split(" :: ") {
            let (m, a) = part.trim().split_once(' ').unwrap_or((part.trim(), ""));
            if !m.is_empty() { stream.push((m.into(), a.trim().into())) }
        }
    }
    let regs = |s: &str| -> Vec<String> {
        s.split(',').filter_map(|t| register(t.trim()).map(|(_, k, n)| format!("{k}{n}"))).collect()
    };
    let du = |m: &str, a: &str| -> (Vec<String>, Vec<String>) {
        if m.starts_with("s_wait") || m.starts_with("s_delay") || m.starts_with("s_cbranch") || m.starts_with("s_branch") || a.is_empty() { return (vec![], vec![]) }
        let (first, rest) = a.split_once(',').unwrap_or((a, ""));
        let mut d = regs(first);
        let mut u = regs(rest);
        if m.contains("fmac") { u.extend(d.clone()) }
        if rest.trim_start().starts_with("vcc_lo") { d.push("vcc".into()) }
        if m == "v_div_fmas_f32" { u.push("vcc".into()) }
        (d, u)
    };
    let fixup = stream.iter().position(|(m, _)| m == "v_div_fixup_f32").ok_or("no v_div_fixup_f32 in symbol")?;
    let q = du(&stream[fixup].0, &stream[fixup].1).0[0].clone();
    let g = regs(&stream[fixup].1).last().cloned().ok_or("fixup operands")?;
    let consumer = (fixup + 1..stream.len()).find(|&i| du(&stream[i].0, &stream[i].1).1.contains(&q)).ok_or("no consumer of the quotient")?;
    let u = du(&stream[consumer].0, &stream[consumer].1).1.into_iter().find(|r| *r != q).ok_or("no u operand")?;
    let inputs = [g, u];
    let mut need: BTreeSet<String> = du(&stream[consumer].0, &stream[consumer].1).1.into_iter().filter(|r| !inputs.contains(r)).collect();
    let mut slice = vec![consumer];
    let mut i = consumer;
    while !need.is_empty() {
        if i == 0 { return Err(format!("unresolved region live-ins {need:?}")) }
        i -= 1;
        let (d, u) = du(&stream[i].0, &stream[i].1);
        if d.iter().any(|r| need.contains(r)) {
            for r in &d { need.remove(r); }
            need.extend(u.into_iter().filter(|r| !inputs.contains(r)));
            slice.push(i);
        }
    }
    slice.sort_unstable();
    let mut out = String::new();
    for &j in &slice {
        let mut k = j;
        let mut waits = Vec::new();
        while k > 0 && matches!(stream[k - 1].0.as_str(), "s_wait_alu" | "s_delay_alu") {
            k -= 1;
            if stream[k].0 == "s_wait_alu" && stream[k].1.contains("va_vcc") { waits.push(k) }
        }
        for &w in waits.iter().rev() { out.push_str(&format!("{} {}\n", stream[w].0, stream[w].1)) }
        out.push_str(&format!("{} {}\n", stream[j].0, stream[j].1));
    }
    Ok(out)
}

/// The committed golden without its provenance comments.
pub fn golden_body() -> String {
    SILU_GOLDEN.lines().filter(|l| !l.trim_start().starts_with(';') && !l.trim().is_empty()).map(|l| format!("{l}\n")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Arch, KernargLayout, KernelSpec, RegPlan, reg::Live};

    /// `elements` instances over `slots` register slots, laid out the way the
    /// gate/up epilogue binds them (partners an odd distance apart).
    fn binds(elements: u8, slots: u8, out: u8) -> Vec<Binding> {
        let masks = [85u8, 86, 87, 63, 56, 57, 58, 59];
        (0..elements).map(|k| {
            let q = k % slots;
            Binding { g: 64 + k, u: 72 + k, out: out + k, temps: (0..7).map(|t| 7 * q + t).collect(),
                masks: vec![masks[2 * q as usize], masks[2 * q as usize + 1]] }
        }).collect()
    }

    fn emitted(binds: &[Binding]) -> Vec<String> {
        let mut p = RegPlan::new(192, 104).unwrap();
        for i in 0..32u8 { p.v::<4>("v", 4 * i, Live::Whole).unwrap(); }
        for s in [56u8, 57, 58, 59, 63, 85, 86, 87] { p.s::<1>("mask", s, Live::Whole).unwrap(); }
        let spec = KernelSpec { kernel_id: "silu".into(), variant: "test".into(), arch: Arch::Gfx1201, symbol: "silu".into(),
            kernargs: KernargLayout::new(8), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false, workgroup_size: 256,
            group_segment_fixed_size: 0, wave32: true };
        let mut b = Builder::new(spec, p);
        emit_interleaved(&mut b, &Region::silu().unwrap(), binds).unwrap();
        b.program.instructions.iter().map(|i| i.text.clone()).collect()
    }

    /// Interleaving and VOPD packing only: the stream is exactly each
    /// element's golden region, in golden order; packets join partners; one
    /// VCC window at a time, closed after its `va_vcc` wait; every mask read
    /// is fenced after the compare that wrote it.
    #[test]
    fn schedule_is_each_elements_golden_region() {
        let region = Region::silu().unwrap();
        for (elements, slots, out) in [(2u8, 2u8, 16u8), (8, 4, 32)] {
            let bs = binds(elements, slots, out);
            let lines = emitted(&bs);
            let mut halves: Vec<(usize, String, Option<usize>)> = Vec::new();
            for (i, l) in lines.iter().enumerate().filter(|(_, l)| l.starts_with("v_")) {
                for h in l.split(" :: ") {
                    let (m, a) = h.split_once(' ').unwrap();
                    halves.push((i, format!("{} {a}", standalone(m)), None));
                }
            }
            for (k, bind) in bs.iter().enumerate() {
                let mut at = 0;
                for (i, op) in region.ops.iter().enumerate().filter(|(_, o)| o.mnemonic != "s_wait_alu") {
                    let text = render(op, bind, i + 1 == region.ops.len()).text;
                    let j = (at..halves.len()).find(|&j| halves[j].2.is_none() && halves[j].1 == text)
                        .unwrap_or_else(|| panic!("element {k} op {i} `{text}` missing or out of order"));
                    halves[j].2 = Some(k);
                    at = j + 1;
                }
            }
            assert!(halves.iter().all(|h| h.2.is_some()), "ops outside the golden elements");
            assert!(halves.len() > lines.iter().filter(|l| l.starts_with("v_")).count(), "no VOPD packet formed");
            for w in halves.windows(2).filter(|w| w[0].0 == w[1].0) { assert_eq!(w[0].2.unwrap() ^ 1, w[1].2.unwrap(), "packet joins non-partners") }
            let (mut open, mut waited, mut unfenced) = (false, false, BTreeSet::new());
            for l in &lines {
                if l.starts_with("v_div_scale_f32") && l.contains("vcc_lo") { assert!(!open, "overlapping VCC windows"); open = true; waited = false }
                if l == "s_wait_alu depctr_va_vcc(0)" { waited = true }
                if l.starts_with("v_div_fmas_f32") { assert!(open && waited, "div_fmas without its window/wait"); open = false }
                if l == "s_wait_alu depctr_va_sdst(0)" { unfenced.clear() }
                if l.starts_with("v_cmp_") { unfenced.insert(l.split([' ', ',']).nth(1).unwrap().to_owned()); }
                if l.starts_with("v_cndmask_b32_e64") { assert!(!unfenced.contains(l.rsplit(", ").next().unwrap()), "unfenced mask read: {l}") }
            }
        }
    }
}
