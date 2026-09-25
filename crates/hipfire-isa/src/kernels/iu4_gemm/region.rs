//! Imported hipcc region: the SiLU epilogue value `h = g / (1 + expf(-g)) * u`.
//!
//! `expf` and the IEEE division are lowered by hipcc into a fixed VALU DAG
//! (range-reduced `v_exp_f32`, overflow/underflow selects, the
//! `v_div_scale`/`v_div_fmas`/`v_div_fixup` sequence). Bit-exactness needs
//! that DAG verbatim, so it is not re-derived: it is sliced out of hipcc's own
//! disassembly (`slice_silu`), committed as `kernels/iu4_gemm.silu.region.s`,
//! parsed back into a register-renamed template (`Region::parse`) and
//! instantiated per element. Only the registers, the interleaving of two
//! independent elements and hazard waits are chosen here.
use super::op;
use crate::{Builder, reg::{Kind, RegRef}};
use std::collections::{BTreeMap, BTreeSet};

pub const SILU_GOLDEN: &str = include_str!("../../../kernels/iu4_gemm.silu.region.s");
/// Temporaries and lane-mask SGPRs one element needs.
pub const SILU_TEMPS: usize = 7;
pub const SILU_MASKS: usize = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Operand { G, U, Temp(usize), Mask(usize), Fixed(String) }
#[derive(Clone, Debug, PartialEq, Eq)]
struct Op { mnemonic: String, operands: Vec<(bool, Operand)>, vcc_def: bool, vcc_use: bool }

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
                ops.push(Op { mnemonic: mnemonic.into(), operands: vec![(false, Operand::Fixed(args.into()))], vcc_def: false, vcc_use: false });
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
            ops.push(Op { mnemonic: standalone(mnemonic), operands, vcc_def, vcc_use });
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

fn render(op: &Op, bind: &Binding, last: bool) -> (String, Vec<RegRef>, Vec<RegRef>) {
    let mut text = op.mnemonic.clone();
    let (mut defs, mut uses) = (Vec::new(), Vec::new());
    let arity = op.operands.len();
    for (i, (neg, operand)) in op.operands.iter().enumerate() {
        let (spelled, reg) = match operand {
            Operand::G => (format!("v{}", bind.g), Some(RegRef { kind: Kind::V, base: bind.g, len: 1 })),
            Operand::U => (format!("v{}", bind.u), Some(RegRef { kind: Kind::V, base: bind.u, len: 1 })),
            Operand::Temp(n) => {
                let r = if last && i == 0 { bind.out } else { bind.temps[*n] };
                (format!("v{r}"), Some(RegRef { kind: Kind::V, base: r, len: 1 }))
            }
            Operand::Mask(n) => (format!("s{}", bind.masks[*n]), Some(RegRef { kind: Kind::S, base: bind.masks[*n], len: 1 })),
            Operand::Fixed(t) => (t.clone(), None),
        };
        text.push_str(if i == 0 { " " } else { ", " });
        if *neg { text.push('-') }
        text.push_str(&spelled);
        if let Some(r) = reg {
            if i == 0 && op.mnemonic != "s_wait_alu" { defs.push(r) } else { uses.push(r) }
        }
        let _ = arity;
    }
    if op.mnemonic.starts_with("v_fmac") { uses.extend(defs.iter().copied()) }
    (text, defs, uses)
}

/// Emit instances of `region` for independent elements, round-robin. An
/// instance may not define VCC while another instance's VCC window (its
/// `v_div_scale` to its `v_div_fmas`) is open. Before an instruction reads a
/// lane mask written by VALU since the last fence, a `va_sdst` wait is issued.
pub fn emit_interleaved(b: &mut Builder, region: &Region, binds: &[Binding]) -> Result<(), String> {
    let mut next = vec![0usize; binds.len()];
    let mut open: Option<usize> = None;
    let mut unfenced: BTreeSet<u8> = BTreeSet::new();
    let mut turn = 0usize;
    while next.iter().any(|&i| i < region.ops.len()) {
        let mut progressed = false;
        for step in 0..binds.len() {
            let s = (turn + step) % binds.len();
            let Some(op) = region.ops.get(next[s]) else { continue };
            if op.vcc_def && open.is_some_and(|o| o != s) { continue }
            let last = next[s] + 1 == region.ops.len();
            let (text, defs, uses) = render(op, &binds[s], last);
            if uses.iter().any(|r| r.kind == Kind::S && unfenced.contains(&r.base)) {
                op_fence(b)?;
                unfenced.clear();
            }
            super::op(b, text, &defs, &uses)?;
            for r in defs.iter().filter(|r| r.kind == Kind::S) { unfenced.insert(r.base); }
            if op.vcc_def { open = Some(s) }
            if op.vcc_use { open = None }
            next[s] += 1;
            turn = s + 1;
            progressed = true;
            break;
        }
        if !progressed { return Err("region interleave deadlock".into()) }
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
