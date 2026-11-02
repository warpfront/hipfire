// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Imported hipcc regions of `gdn_chunk_prep` for the F2 QKVZA+GDN epilogue
//! (and the A4 `_b1` fused projection, `iu4_gemm::gdn_epilogue`).
//!
//! The per-token arithmetic of the GDN preparation (width-4 causal conv1d,
//! SiLU, the q/k head RMS norm with its xor butterfly, the q scale and the
//! FP16 conversions) is not re-derived. It is sliced out of the lifted hipcc
//! code object (`peacemaker-lift` round-trips it byte for byte;
//! [`slice_gdn`] reads its canonical text), committed as
//! `kernels/fp8_gemm.gdn.*.region.s`, parsed back into register-renamed
//! templates ([`Region::parse`]) and instantiated per channel / per token.
//! Only register names, the interleaving of independent instances, VOPD
//! packing and hazard waits are chosen here; every instance issues its ops in
//! golden order. Two value-preserving renamings are applied while parsing:
//! lane masks that hipcc kept in VCC (`v_cmp_*_e32` feeding
//! `v_cndmask_b32_e32`) move to SGPR masks, and the `__shfl_xor` lane-index
//! idiom (`lane ^ d`, its always-true `< 32` guard, `<< 2`, `ds_bpermute_b32`)
//! becomes a register exchange (`v_mov_b32_dpp row_xmask:d`, or
//! `v_permlanex16_b32` for d = 16). Both move identical bits.
//!
//! The epilogue instantiates one region that is not a slice:
//! `fp8_gemm.gdn.conv_silu_lean.region.s` keeps the golden's four conv MACs
//! verbatim and replaces its SiLU with a shorter sequence (clamped exp range
//! reduction, one-correction division ending in `v_div_fixup_f32`) that
//! returns the golden's bits for all 2^32 f32 conv results
//! (`tools/gdn/silu_exhaust.py` runs both regions exhaustively on gfx1201).
use crate::{
    reg::{Kind, RegRef},
    vopd::{self, VopdF32, VopdOp},
    Builder,
};
use std::collections::{BTreeMap, BTreeSet};

pub const CONV_SILU_GOLDEN: &str = include_str!("../../../kernels/fp8_gemm.gdn.conv_silu.region.s");
pub const NORM_Q_GOLDEN: &str = include_str!("../../../kernels/fp8_gemm.gdn.norm_q.region.s");
pub const NORM_K_GOLDEN: &str = include_str!("../../../kernels/fp8_gemm.gdn.norm_k.region.s");
pub const CVT_V_GOLDEN: &str = include_str!("../../../kernels/fp8_gemm.gdn.cvt_v.region.s");
pub const CONV_SILU_LEAN: &str =
    include_str!("../../../kernels/fp8_gemm.gdn.conv_silu_lean.region.s");

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Half {
    Full,
    Lo,
    Hi,
}
impl Half {
    fn suffix(self) -> &'static str {
        match self {
            Self::Full => "",
            Self::Lo => ".l",
            Self::Hi => ".h",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Operand {
    In(usize),
    SIn(usize),
    Temp(usize, Half),
    Mask(usize),
    Vcc,
    Fixed(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Op {
    mnemonic: String,
    operands: Vec<(bool, Operand)>,
    vcc_def: bool,
    vcc_use: bool,
    dual: Option<VopdF32>,
    /// Lane exchange `dst = src[lane ^ xor]`.
    exchange: Option<u8>,
    /// Index into `Region::outputs` when this op's destination is that output's final value.
    output: Option<usize>,
}

/// A parsed, register-renamed region with named VGPR inputs, SGPR inputs and
/// outputs (a register or a 16-bit half of one).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    ops: Vec<Op>,
    pub inputs: Vec<String>,
    pub sinputs: Vec<String>,
    pub outputs: Vec<String>,
    pub temps: usize,
    pub masks: usize,
}

/// Registers bound to one region instance. `outputs` gives the register and
/// half each declared output is written to; `lane_select` is an SGPR holding
/// 0x76543210 when the region exchanges across the two 16-lane rows.
#[derive(Clone, Debug)]
pub struct Binding {
    pub inputs: Vec<u8>,
    pub sinputs: Vec<u8>,
    pub temps: Vec<u8>,
    pub masks: Vec<u8>,
    pub outputs: Vec<(u8, Half)>,
    pub lane_select: Option<u8>,
}

fn instruction(line: &str) -> Option<(&str, &str)> {
    let code = line.split(';').next()?.split("//").next()?.trim();
    if code.is_empty() {
        return None;
    }
    Some(code.split_once(' ').unwrap_or((code, "")))
}
fn standalone(mnemonic: &str) -> String {
    match mnemonic.strip_prefix("v_dual_") {
        Some("cndmask_b32") => "v_cndmask_b32_e32".into(),
        Some(rest) => format!("v_{rest}_e32"),
        None => mnemonic.to_owned(),
    }
}
/// `(negated, kind, number, half)` of a single-register token.
fn register(token: &str) -> Option<(bool, char, u16, Half)> {
    let (neg, t) = token
        .strip_prefix('-')
        .map_or((false, token), |t| (true, t));
    let (t, half) = if let Some(r) = t.strip_suffix(".l") {
        (r, Half::Lo)
    } else if let Some(r) = t.strip_suffix(".h") {
        (r, Half::Hi)
    } else {
        (t, Half::Full)
    };
    let kind = t.chars().next()?;
    if kind != 'v' && kind != 's' {
        return None;
    }
    t[1..].parse().ok().map(|n| (neg, kind, n, half))
}
fn dual_form(mnemonic: &str, operands: &[(bool, Operand)]) -> Option<VopdF32> {
    let op = match mnemonic {
        "v_add_f32_e32" => VopdF32::Add,
        "v_sub_f32_e32" => VopdF32::Sub,
        "v_subrev_f32_e32" => VopdF32::Subrev,
        "v_mul_f32_e32" => VopdF32::Mul,
        "v_fmac_f32_e32" => VopdF32::Fmac,
        _ => return None,
    };
    let vgpr = |o: &Operand| matches!(o, Operand::In(_) | Operand::Temp(_, Half::Full));
    match operands {
        [(false, Operand::Temp(_, Half::Full)), (false, src0), (false, src1)]
            if vgpr(src1) && !matches!(src0, Operand::Temp(_, Half::Lo | Half::Hi)) =>
        {
            Some(op)
        }
        _ => None,
    }
}
fn header<'a>(text: &'a str, key: &str) -> Vec<(&'a str, &'a str)> {
    text.lines()
        .filter_map(|l| {
            l.trim()
                .strip_prefix(';')?
                .trim()
                .strip_prefix(key)?
                .strip_prefix(':')
        })
        .flat_map(|rest| rest.split_whitespace())
        .filter_map(|pair| pair.split_once('='))
        .collect()
}

/// Index-lane symbolic values of the `__shfl_xor` idiom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    Xor(u8),
    Guard(u8),
    Bytes(u8),
}

impl Region {
    /// Parse a golden. Header comments declare the interface:
    /// `; inputs: name=vN ...`, `; sinputs: name=sN ...`, `; lane: vN` (the
    /// lane id register of the exchange idiom) and `; outputs: name=vN[.l|.h]`.
    pub fn parse(text: &str) -> Result<Self, String> {
        let inputs = header(text, "inputs");
        let sinputs = header(text, "sinputs");
        let outputs = header(text, "outputs");
        let lane = text
            .lines()
            .find_map(|l| l.trim().strip_prefix(';')?.trim().strip_prefix("lane:"))
            .map(|s| s.trim().to_owned());
        let in_reg: BTreeMap<u16, usize> = inputs
            .iter()
            .enumerate()
            .map(|(k, (_, r))| {
                register(r)
                    .filter(|x| x.1 == 'v' && x.3 == Half::Full)
                    .map(|x| (x.2, k))
                    .ok_or_else(|| format!("bad input {r}"))
            })
            .collect::<Result<_, _>>()?;
        let sin_reg: BTreeMap<u16, usize> = sinputs
            .iter()
            .enumerate()
            .map(|(k, (_, r))| {
                register(r)
                    .filter(|x| x.1 == 's')
                    .map(|x| (x.2, k))
                    .ok_or_else(|| format!("bad sinput {r}"))
            })
            .collect::<Result<_, _>>()?;
        let lane_reg = match &lane {
            Some(l) => Some(
                register(l)
                    .filter(|x| x.1 == 'v')
                    .ok_or("bad lane register")?
                    .2,
            ),
            None => None,
        };
        let mut ops: Vec<Op> = Vec::new();
        let mut vmap: BTreeMap<u16, usize> = BTreeMap::new();
        let mut defined: BTreeSet<u16> = BTreeSet::new();
        let mut symbolic: BTreeMap<u16, Lane> = BTreeMap::new();
        let mut vcc_lane: Option<Lane> = None;
        // Current producer of VCC: a compare (renamed to a mask) or a division scale.
        let mut vcc_mask: Option<usize> = None;
        let mut masks = 0usize;
        // Per golden register: index of the op holding its last definition (per half).
        let mut last_def: BTreeMap<(u16, Half), usize> = BTreeMap::new();
        let temp = |n: u16, vmap: &mut BTreeMap<u16, usize>| {
            let next = vmap.len();
            *vmap.entry(n).or_insert(next)
        };
        for line in text.lines() {
            let Some((raw, args)) = instruction(line) else {
                continue;
            };
            let mnemonic = standalone(raw);
            let tokens: Vec<&str> = if args.is_empty() {
                vec![]
            } else {
                args.split(',').map(str::trim).collect()
            };
            if mnemonic == "s_wait_alu" {
                ops.push(Op {
                    mnemonic,
                    operands: vec![(false, Operand::Fixed(args.into()))],
                    vcc_def: false,
                    vcc_use: false,
                    dual: None,
                    exchange: None,
                    output: None,
                });
                continue;
            }
            // The lane-index idiom is evaluated symbolically and never emitted.
            let src_lane = |t: &str, symbolic: &BTreeMap<u16, Lane>| {
                register(t).and_then(|r| {
                    if r.1 == 'v' {
                        symbolic.get(&r.2).copied()
                    } else {
                        None
                    }
                })
            };
            let is_lane = |t: &str| {
                register(t)
                    .is_some_and(|r| r.1 == 'v' && Some(r.2) == lane_reg && !defined.contains(&r.2))
            };
            match (mnemonic.as_str(), tokens.as_slice()) {
                ("v_xor_b32_e32", [d, k, l]) if is_lane(l) => {
                    let xor: u8 = k.parse().map_err(|_| format!("lane xor constant {k}"))?;
                    if !(1..=16).contains(&xor) || !xor.is_power_of_two() {
                        return Err(format!("unsupported lane xor {xor}"));
                    }
                    symbolic.insert(register(d).ok_or("xor dst")?.2, Lane::Xor(xor));
                    continue;
                }
                ("v_cmp_gt_u32_e32", ["vcc_lo", "32", i])
                    if matches!(src_lane(i, &symbolic), Some(Lane::Xor(_))) =>
                {
                    let Some(Lane::Xor(x)) = src_lane(i, &symbolic) else {
                        unreachable!()
                    };
                    vcc_lane = Some(Lane::Guard(x));
                    continue;
                }
                ("v_cndmask_b32_e32", [d, l, i, "vcc_lo"])
                    if is_lane(l) && matches!(src_lane(i, &symbolic), Some(Lane::Xor(_))) =>
                {
                    let Some(Lane::Xor(x)) = src_lane(i, &symbolic) else {
                        unreachable!()
                    };
                    // lane < 32 and x <= 16 keep lane ^ x < 32: the guard always selects the xor.
                    if vcc_lane != Some(Lane::Guard(x)) {
                        return Err("lane guard does not match its xor".into());
                    }
                    symbolic.insert(register(d).ok_or("cndmask dst")?.2, Lane::Xor(x));
                    continue;
                }
                ("v_lshlrev_b32_e32", [d, "2", i])
                    if matches!(src_lane(i, &symbolic), Some(Lane::Xor(_))) =>
                {
                    let Some(Lane::Xor(x)) = src_lane(i, &symbolic) else {
                        unreachable!()
                    };
                    symbolic.insert(register(d).ok_or("lshl dst")?.2, Lane::Bytes(x));
                    continue;
                }
                ("ds_bpermute_b32", [d, i, s]) => {
                    let Some(Lane::Bytes(x)) = src_lane(i, &symbolic) else {
                        return Err("ds_bpermute outside the lane-xor idiom".into());
                    };
                    let dst = register(d).ok_or("bpermute dst")?.2;
                    let src = register(s).ok_or("bpermute src")?.2;
                    let src_op = if !defined.contains(&src) {
                        Operand::In(
                            *in_reg
                                .get(&src)
                                .ok_or("exchange of an undeclared live-in")?,
                        )
                    } else {
                        Operand::Temp(temp(src, &mut vmap), Half::Full)
                    };
                    defined.insert(dst);
                    symbolic.remove(&dst);
                    let t = temp(dst, &mut vmap);
                    last_def.insert((dst, Half::Full), ops.len());
                    ops.push(Op {
                        mnemonic: "exchange".into(),
                        operands: vec![(false, Operand::Temp(t, Half::Full)), (false, src_op)],
                        vcc_def: false,
                        vcc_use: false,
                        dual: None,
                        exchange: Some(x),
                        output: None,
                    });
                    continue;
                }
                _ => {}
            }
            let mut operands: Vec<(bool, Operand)> = Vec::with_capacity(tokens.len());
            let mut out_mnemonic = mnemonic.clone();
            let mut vcc_def = false;
            let mut vcc_use = mnemonic == "v_div_fmas_f32";
            // Sources are classified against the state before this instruction.
            let mut def_slot: Option<(u16, Half)> = None;
            for (i, t) in tokens.iter().enumerate() {
                let is_def = i == 0 || (i == 1 && mnemonic == "v_div_scale_f32");
                if *t == "vcc_lo" {
                    if is_def && mnemonic.starts_with("v_cmp_") && mnemonic.ends_with("_e32") {
                        vcc_mask = Some(masks);
                        operands.push((false, Operand::Mask(masks)));
                        masks += 1;
                        out_mnemonic =
                            format!("{}_e64", mnemonic.strip_suffix("_e32").unwrap_or(&mnemonic));
                    } else if is_def {
                        vcc_mask = None;
                        vcc_def = true;
                        operands.push((false, Operand::Vcc));
                    } else if mnemonic == "v_cndmask_b32_e32" && i == tokens.len() - 1 {
                        let m = vcc_mask.ok_or("VCC select without a compare producer")?;
                        operands.push((false, Operand::Mask(m)));
                        out_mnemonic = "v_cndmask_b32_e64".into();
                    } else {
                        vcc_use = true;
                        operands.push((false, Operand::Vcc));
                    }
                    continue;
                }
                let Some((neg, kind, n, half)) = register(t) else {
                    operands.push((false, Operand::Fixed((*t).into())));
                    continue;
                };
                if kind == 's' {
                    if is_def {
                        return Err(format!("region defines SGPR s{n}"));
                    }
                    operands.push((
                        neg,
                        Operand::SIn(
                            *sin_reg
                                .get(&n)
                                .ok_or_else(|| format!("undeclared SGPR input s{n}"))?,
                        ),
                    ));
                    continue;
                }
                if i == 0 {
                    def_slot = Some((n, half));
                    operands.push((neg, Operand::Fixed(String::new())));
                    continue;
                }
                if symbolic.contains_key(&n) {
                    return Err(format!("lane-index value v{n} escapes the exchange idiom"));
                }
                let slot = if !defined.contains(&n) {
                    if half != Half::Full {
                        return Err(format!("half read of live-in v{n}"));
                    }
                    Operand::In(
                        *in_reg
                            .get(&n)
                            .ok_or_else(|| format!("undeclared VGPR live-in v{n}"))?,
                    )
                } else {
                    Operand::Temp(temp(n, &mut vmap), half)
                };
                operands.push((neg, slot));
            }
            if let Some((n, half)) = def_slot {
                defined.insert(n);
                symbolic.remove(&n);
                last_def.insert((n, half), ops.len());
                if half == Half::Full {
                    last_def.insert((n, Half::Lo), ops.len());
                    last_def.insert((n, Half::Hi), ops.len());
                }
                operands[0] = (false, Operand::Temp(temp(n, &mut vmap), half));
            }
            let dual = dual_form(&out_mnemonic, &operands);
            ops.push(Op {
                mnemonic: out_mnemonic,
                operands,
                vcc_def,
                vcc_use,
                dual,
                exchange: None,
                output: None,
            });
        }
        if !symbolic.is_empty()
            && symbolic
                .values()
                .any(|v| !matches!(v, Lane::Bytes(_) | Lane::Xor(_)))
        {
            return Err("dangling lane-index value".into());
        }
        for (k, (name, r)) in outputs.iter().enumerate() {
            let (_, kind, n, half) = register(r).ok_or_else(|| format!("bad output {r}"))?;
            if kind != 'v' {
                return Err(format!("output {name} is not a VGPR"));
            }
            let at = *last_def
                .get(&(n, half))
                .ok_or_else(|| format!("output {name}={r} is never defined"))?;
            if ops[at].output.is_some() {
                return Err(format!("output {name} shares its final op"));
            }
            ops[at].output = Some(k);
        }
        Ok(Self {
            ops,
            inputs: inputs.iter().map(|(n, _)| (*n).into()).collect(),
            sinputs: sinputs.iter().map(|(n, _)| (*n).into()).collect(),
            outputs: outputs.iter().map(|(n, _)| (*n).into()).collect(),
            temps: vmap.len(),
            masks,
        })
    }
    pub fn conv_silu() -> Result<Self, String> {
        Self::parse(CONV_SILU_GOLDEN)
    }
    /// The exhaustively proven lean conv+SiLU (SGPR input `clamp` = 128.0).
    pub fn conv_silu_lean() -> Result<Self, String> {
        Self::parse(CONV_SILU_LEAN)
    }
    pub fn norm_q() -> Result<Self, String> {
        Self::parse(NORM_Q_GOLDEN)
    }
    pub fn norm_k() -> Result<Self, String> {
        Self::parse(NORM_K_GOLDEN)
    }
    pub fn cvt_v() -> Result<Self, String> {
        Self::parse(CVT_V_GOLDEN)
    }
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
    pub fn input(&self, name: &str) -> Result<usize, String> {
        self.inputs
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| format!("region has no input {name}"))
    }
    pub fn sinput(&self, name: &str) -> Result<usize, String> {
        self.sinputs
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| format!("region has no SGPR input {name}"))
    }
    pub fn output(&self, name: &str) -> Result<usize, String> {
        self.outputs
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| format!("region has no output {name}"))
    }
    pub fn mnemonics(&self) -> Vec<&str> {
        self.ops.iter().map(|o| o.mnemonic.as_str()).collect()
    }
}

struct Issue {
    texts: Vec<String>,
    defs: Vec<RegRef>,
    uses: Vec<RegRef>,
    dual: Option<(VopdOp, String)>,
}

fn constant(text: &str) -> Option<vopd::Operand> {
    if let Some(hex) = text.strip_prefix("0x") {
        return u32::from_str_radix(hex, 16).ok().map(vopd::Operand::Lit);
    }
    if let Ok(n) = text.parse::<i64>() {
        return Some(if (-16..=64).contains(&n) {
            vopd::Operand::Inline(n as i8)
        } else {
            vopd::Operand::Lit(n as u32)
        });
    }
    text.parse::<f32>()
        .ok()
        .map(|x| vopd::Operand::Lit(x.to_bits()))
}
fn vreg(n: u8) -> RegRef {
    RegRef {
        kind: Kind::V,
        base: n,
        len: 1,
    }
}
fn sreg(n: u8) -> RegRef {
    RegRef {
        kind: Kind::S,
        base: n,
        len: 1,
    }
}

fn render(op: &Op, bind: &Binding) -> Result<Issue, String> {
    let reg_of = |o: &Operand, def: bool| -> Result<(String, Option<RegRef>), String> {
        Ok(match o {
            Operand::In(k) => {
                let r = *bind.inputs.get(*k).ok_or("unbound input")?;
                (format!("v{r}"), Some(vreg(r)))
            }
            Operand::SIn(k) => {
                let r = *bind.sinputs.get(*k).ok_or("unbound SGPR input")?;
                (format!("s{r}"), Some(sreg(r)))
            }
            Operand::Temp(t, half) => {
                let (r, h) = match (def, op.output) {
                    (true, Some(k)) => *bind.outputs.get(k).ok_or("unbound output")?,
                    _ => (*bind.temps.get(*t).ok_or("unbound temp")?, *half),
                };
                (format!("v{r}{}", h.suffix()), Some(vreg(r)))
            }
            Operand::Mask(m) => {
                let r = *bind.masks.get(*m).ok_or("unbound mask")?;
                (format!("s{r}"), Some(sreg(r)))
            }
            Operand::Vcc => ("vcc_lo".into(), None),
            Operand::Fixed(t) => (t.clone(), None),
        })
    };
    if let Some(x) = op.exchange {
        let (dst, dr) = reg_of(&op.operands[0].1, true)?;
        let (src, sr) = reg_of(&op.operands[1].1, false)?;
        let (dr, sr) = (dr.ok_or("exchange dst")?, sr.ok_or("exchange src")?);
        return Ok(if x == 16 {
            let sel = bind
                .lane_select
                .ok_or("row exchange needs a lane-select SGPR")?;
            // permlanex16's destination is also its old value; define it first.
            Issue {
                texts: vec![
                    format!("v_mov_b32_e32 {dst}, {src}"),
                    format!("v_permlanex16_b32 {dst}, {src}, s{sel}, 0xfedcba98"),
                ],
                defs: vec![dr],
                uses: vec![sr, sreg(sel)],
                dual: None,
            }
        } else {
            Issue {
                texts: vec![format!(
                    "v_mov_b32_dpp {dst}, {src} row_xmask:{x} row_mask:0xf bank_mask:0xf"
                )],
                defs: vec![dr],
                uses: vec![sr],
                dual: None,
            }
        });
    }
    let mut text = op.mnemonic.clone();
    let (mut defs, mut uses) = (Vec::new(), Vec::new());
    let mut spelled = Vec::new();
    let mut src0 = None;
    for (i, (neg, operand)) in op.operands.iter().enumerate() {
        let def =
            i == 0 && op.mnemonic != "s_wait_alu" || (i == 1 && op.mnemonic == "v_div_scale_f32");
        let (name, reg) = reg_of(operand, i == 0 && op.mnemonic != "s_wait_alu")?;
        if i == 1 {
            src0 = match (operand, reg) {
                (Operand::Mask(_) | Operand::SIn(_), Some(r)) => Some(vopd::Operand::S(r.base)),
                (_, Some(r)) => Some(vopd::Operand::V(r.base)),
                (_, None) => constant(&name),
            };
        }
        text.push_str(if i == 0 { " " } else { ", " });
        if *neg {
            text.push('-')
        }
        text.push_str(&name);
        spelled.push(name);
        if let Some(r) = reg {
            if def {
                defs.push(r)
            } else {
                uses.push(r)
            }
            // A 16-bit half write keeps the other half.
            if def && matches!(operand, Operand::Temp(_, Half::Lo | Half::Hi))
                || def && op.output.is_some() && bind.outputs[op.output.unwrap()].1 != Half::Full
            {
                uses.push(r)
            }
        }
    }
    if op.mnemonic.starts_with("v_fmac") {
        uses.extend(defs.iter().copied())
    }
    let dual = match (op.dual, src0, defs.first()) {
        (Some(kind), Some(src0), Some(dst))
            if op.output.is_none() || bind.outputs[op.output.unwrap()].1 == Half::Full =>
        {
            let base = op
                .mnemonic
                .strip_prefix("v_")
                .and_then(|m| m.strip_suffix("_e32"))
                .unwrap_or(&op.mnemonic);
            let src1 = match &op.operands[2].1 {
                Operand::In(k) => bind.inputs[*k],
                Operand::Temp(t, _) => bind.temps[*t],
                _ => return Err("dual form admits VGPR src1 only".into()),
            };
            Some((
                VopdOp {
                    op: kind,
                    dst: dst.base,
                    src0,
                    src1,
                },
                format!("v_dual_{base} {}", spelled.join(", ")),
            ))
        }
        _ => None,
    };
    Ok(Issue {
        texts: vec![text],
        defs,
        uses,
        dual,
    })
}

fn packet(arch: crate::Arch, a: &Issue, b: &Issue) -> Option<(String, Vec<RegRef>, Vec<RegRef>)> {
    let ((x, x_text), (y, y_text)) = (a.dual.as_ref()?, b.dual.as_ref()?);
    vopd::validate_pair(arch, *x, *y).ok()?;
    let touches = |defs: &[RegRef], other: &Issue| {
        defs.iter()
            .any(|d| other.uses.iter().chain(&other.defs).any(|r| r.overlaps(*d)))
    };
    if touches(&a.defs, b) || touches(&b.defs, a) {
        return None;
    }
    Some((
        format!("{x_text} :: {y_text}"),
        [a.defs.clone(), b.defs.clone()].concat(),
        [a.uses.clone(), b.uses.clone()].concat(),
    ))
}

const LATENCY: u32 = 5;
const TRANS_LATENCY: u32 = 10;
fn latency(mnemonic: &str) -> u32 {
    if ["v_exp_", "v_log_", "v_rcp_", "v_rsq_", "v_sqrt_"]
        .iter()
        .any(|p| mnemonic.starts_with(p))
    {
        TRANS_LATENCY
    } else {
        LATENCY
    }
}

/// Emit instances of `region` for independent elements with the same rules
/// as the imported SiLU region (`iu4_gemm::region::emit_interleaved`): each
/// instance issues its ops in golden order; instances `2k`/`2k+1` pair into
/// VOPD packets when legal; one VCC window (`v_div_scale` .. `v_div_fmas`) at
/// a time. Lane-mask (`va_sdst`) waits come from the builder's gfx12 SGPR
/// hazard model on every mask read.
pub fn emit_interleaved(b: &mut Builder, region: &Region, binds: &[Binding]) -> Result<(), String> {
    let (n, len, arch) = (binds.len(), region.ops.len(), b.spec.arch);
    let owned: Vec<BTreeSet<(bool, u8)>> = binds
        .iter()
        .map(|bind| {
            bind.temps
                .iter()
                .chain(bind.outputs.iter().map(|(r, _)| r))
                .map(|&r| (false, r))
                .chain(bind.masks.iter().map(|&m| (true, m)))
                .collect()
        })
        .collect();
    let after: Vec<Vec<usize>> = (0..n)
        .map(|k| {
            (0..k)
                .filter(|&e| !owned[e].is_disjoint(&owned[k]))
                .collect()
        })
        .collect();
    let issue = |s: usize, i: usize| render(&region.ops[i], &binds[s]);
    let mut next = vec![0usize; n];
    let mut open: Option<usize> = None;
    let mut ready: BTreeMap<(bool, u8), u32> = BTreeMap::new();
    let mut vcc_ready = 0u32;
    let mut clock = 0u32;
    while next.iter().any(|&i| i < len) {
        let live =
            |s: usize| next[s] < len && (next[s] > 0 || after[s].iter().all(|&e| next[e] == len));
        let ready_at = |s: usize| -> Result<u32, String> {
            let op = &region.ops[next[s]];
            let regs = issue(s, next[s])?
                .uses
                .iter()
                .flat_map(|r| (0..r.len).map(move |k| (r.kind == Kind::S, r.base + k)))
                .map(|key| ready.get(&key).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            Ok(if op.mnemonic == "s_wait_alu" {
                regs.max(vcc_ready)
            } else {
                regs
            })
        };
        let mut groups: Vec<(Vec<usize>, u32)> = Vec::new();
        for s in (0..n).filter(|&s| live(s)) {
            let i = next[s];
            let op = &region.ops[i];
            if op.vcc_def && open.is_some_and(|o| o != s) {
                continue;
            }
            let p = s ^ 1;
            if op.dual.is_some()
                && open != Some(s)
                && p < n
                && next[p] <= i
                && packet(arch, &issue(s, i)?, &issue(p, i)?).is_some()
            {
                if next[p] == i && live(p) && s < p {
                    groups.push((vec![s, p], ready_at(s)?.max(ready_at(p)?)))
                }
                continue;
            }
            groups.push((vec![s], ready_at(s)?));
        }
        let (members, at) = groups
            .iter()
            .find(|(_, t)| *t <= clock)
            .or_else(|| groups.iter().min_by_key(|(_, t)| *t))
            .cloned()
            .ok_or("region interleave deadlock")?;
        clock = clock.max(at);
        let issues: Vec<Issue> = members
            .iter()
            .map(|&s| issue(s, next[s]))
            .collect::<Result<_, _>>()?;
        let (texts, defs, uses) = match issues.as_slice() {
            [a, b] => {
                let (t, d, u) = packet(arch, a, b).ok_or("VOPD packet rejected after selection")?;
                (vec![t], d, u)
            }
            [a] => (a.texts.clone(), a.defs.clone(), a.uses.clone()),
            _ => unreachable!("groups hold one op or one packet"),
        };
        let count = texts.len();
        for (k, text) in texts.into_iter().enumerate() {
            // A two-instruction exchange defines its destination in both.
            let (d, u) = if count == 2 && k == 0 {
                (defs.clone(), uses[..1].to_vec())
            } else if count == 2 {
                (defs.clone(), [uses.clone(), defs.clone()].concat())
            } else {
                (defs.clone(), uses.clone())
            };
            b.push(crate::insn::Instruction::new(text, d, u))?;
        }
        let lat = if members.len() == 2 {
            LATENCY
        } else {
            latency(&region.ops[next[members[0]]].mnemonic)
        };
        for r in &defs {
            for k in 0..r.len {
                ready.insert((r.kind == Kind::S, r.base + k), clock + lat);
            }
        }
        for &s in &members {
            let op = &region.ops[next[s]];
            if op.vcc_def {
                open = Some(s);
                vcc_ready = clock + lat
            }
            if op.vcc_use {
                open = None
            }
            next[s] += 1;
        }
        clock += 1;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Slicer: canonical lifted text of `gdn_chunk_prep` -> the four goldens.
// ---------------------------------------------------------------------------

/// One instruction (VOPD halves split into their VOP2 forms).
#[derive(Clone, Debug)]
struct Line {
    index: usize,
    mnemonic: String,
    args: String,
}

fn split(lines: &[String]) -> Vec<Line> {
    let mut out = Vec::new();
    for (index, l) in lines.iter().enumerate() {
        for part in l.split(" :: ") {
            let part = part.trim();
            let (m, a) = part.split_once(' ').unwrap_or((part, ""));
            if m.is_empty() {
                continue;
            }
            let (m, a) = match m.strip_prefix("v_dual_") {
                Some("cndmask_b32") => (
                    "v_cndmask_b32_e32".to_owned(),
                    format!("{}, vcc_lo", a.trim()),
                ),
                Some(rest) => (format!("v_{rest}_e32"), a.trim().to_owned()),
                None => (m.to_owned(), a.trim().to_owned()),
            };
            out.push(Line {
                index,
                mnemonic: m,
                args: a,
            });
        }
    }
    out
}
fn regs_of(token: &str) -> Vec<String> {
    let t = token.trim();
    if t == "vcc_lo" {
        return vec!["vcc".into()];
    }
    if let Some((kind, rest)) = t
        .strip_prefix('v')
        .map(|r| ('v', r))
        .or_else(|| t.strip_prefix('s').map(|r| ('s', r)))
    {
        if let Some(range) = rest.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
            if let Some((a, b)) = range.split_once(':') {
                if let (Ok(a), Ok(b)) = (a.parse::<u16>(), b.parse::<u16>()) {
                    return (a..=b).map(|n| format!("{kind}{n}")).collect();
                }
            }
            return vec![];
        }
    }
    register(t)
        .map(|(_, k, n, _)| vec![format!("{k}{n}")])
        .unwrap_or_default()
}
/// Definitions (a 16-bit half write is spelled `vN.l`/`vN.h`) and full-register uses.
fn def_use(l: &Line) -> (Vec<String>, Vec<String>) {
    let m = l.mnemonic.as_str();
    if l.args.is_empty()
        || m.starts_with("s_wait")
        || m.starts_with("s_delay")
        || m.starts_with("s_cbranch")
        || m.starts_with("s_branch")
        || m.starts_with("s_clause")
    {
        return (vec![], vec![]);
    }
    let toks: Vec<&str> = l.args.split(',').map(str::trim).collect();
    if m.starts_with("global_store") || m.starts_with("buffer_store") || m.starts_with("ds_store") {
        return (vec![], toks.iter().flat_map(|t| regs_of(t)).collect());
    }
    let half_def = |t: &str| {
        if t.ends_with(".l") || t.ends_with(".h") {
            vec![t.to_owned()]
        } else {
            regs_of(t)
        }
    };
    let mut d = half_def(toks[0]);
    let mut u: Vec<String> = toks[1..].iter().flat_map(|t| regs_of(t)).collect();
    if m == "v_div_scale_f32" {
        d = [regs_of(toks[0]), regs_of(toks[1])].concat();
        u = toks[2..].iter().flat_map(|t| regs_of(t)).collect();
    }
    if m.contains("fmac") {
        u.extend(regs_of(toks[0]))
    }
    if m == "v_div_fmas_f32" || (m.starts_with("v_cndmask") && m.ends_with("_e32")) {
        u.push("vcc".into())
    }
    (d, u)
}
/// Backward dataflow slice of `targets` (full registers or `vN.l`/`vN.h`)
/// within `seq`, tracked per 16-bit half so a half write satisfies only its
/// half. Returns the slice and its live-in registers.
fn backward(seq: &[Line], targets: &[&str]) -> (Vec<Line>, BTreeSet<String>) {
    let halves = |r: &str| -> Vec<String> {
        if r.ends_with(".l") || r.ends_with(".h") || !r.starts_with('v') {
            vec![r.to_owned()]
        } else {
            vec![format!("{r}.l"), format!("{r}.h")]
        }
    };
    let mut need: BTreeSet<String> = targets.iter().flat_map(|s| halves(s)).collect();
    let mut keep = Vec::new();
    for k in (0..seq.len()).rev() {
        let (d, u) = def_use(&seq[k]);
        let written: Vec<String> = d.iter().flat_map(|r| halves(r)).collect();
        if written.iter().any(|r| need.contains(r)) {
            keep.push(k);
            for r in &written {
                need.remove(r);
            }
            need.extend(u.iter().flat_map(|r| halves(r)));
        }
    }
    keep.sort_unstable();
    let live = need
        .iter()
        .map(|r| r.trim_end_matches(".l").trim_end_matches(".h").to_owned())
        .collect();
    (keep.into_iter().map(|k| seq[k].clone()).collect(), live)
}
/// hipcc keeps `s_wait_alu depctr_va_vcc(0)` between each division's VCC
/// definition and its `v_div_fmas_f32`; the golden keeps it in golden order.
/// Lane-mask hazards are re-derived by the instantiation after mask renaming.
fn with_vcc_waits(slice: &[Line]) -> Vec<Line> {
    let mut out = Vec::new();
    for l in slice {
        if l.mnemonic == "v_div_fmas_f32" {
            out.push(Line {
                index: l.index,
                mnemonic: "s_wait_alu".into(),
                args: "depctr_va_vcc(0)".into(),
            })
        }
        out.push(l.clone());
    }
    out
}
fn text(lines: &[Line]) -> String {
    lines
        .iter()
        .map(|l| {
            if l.args.is_empty() {
                format!("{}\n", l.mnemonic)
            } else {
                format!("{} {}\n", l.mnemonic, l.args)
            }
        })
        .collect()
}

/// Value numbering of a straight-line slice: two slices compute the same
/// function when their outputs number equal after live-ins are bound to role
/// names. Register names, schedule and VCC routing do not matter; operation,
/// operand order, modifiers and constants do. Half writes are numbered per
/// half; `v_mov_b32` and `s_mov_b32` are copies.
fn value_numbers(slice: &[Line], roles: &BTreeMap<String, String>, outs: &[&str]) -> Vec<String> {
    let key = |t: &str| {
        if t == "vcc_lo" {
            "vcc".to_owned()
        } else {
            t.to_owned()
        }
    };
    let mut val: BTreeMap<String, String> = BTreeMap::new();
    let get = |k: &str, val: &BTreeMap<String, String>| {
        val.get(k)
            .cloned()
            .or_else(|| roles.get(k).cloned())
            .unwrap_or_else(|| format!("?{k}"))
    };
    for l in slice {
        let m = l.mnemonic.as_str();
        if m.starts_with("s_wait") {
            continue;
        }
        let toks: Vec<&str> = l.args.split(',').map(str::trim).collect();
        let src = |t: &str, val: &BTreeMap<String, String>| -> String {
            let (neg, body) = t.strip_prefix('-').map_or((false, t), |b| (true, b));
            let v = if body == "vcc_lo" || register(body).is_some() {
                get(&key(body), val)
            } else {
                body.to_owned()
            };
            if neg {
                format!("-{v}")
            } else {
                v
            }
        };
        if m == "s_mov_b32" || m == "v_mov_b32_e32" {
            let v = src(toks[1], &val);
            val.insert(key(toks[0]), v);
            continue;
        }
        let op = m.trim_end_matches("_e32").trim_end_matches("_e64");
        let (dsts, srcs): (Vec<String>, Vec<String>) = if m == "v_div_scale_f32" {
            (
                vec![key(toks[0]), key(toks[1])],
                toks[2..].iter().map(|t| src(t, &val)).collect(),
            )
        } else if m.starts_with("v_cmp") && m.ends_with("_e32") {
            (
                vec!["vcc".into()],
                toks[1..].iter().map(|t| src(t, &val)).collect(),
            )
        } else {
            let mut s: Vec<String> = toks[1..].iter().map(|t| src(t, &val)).collect();
            if m.contains("fmac") {
                s.push(src(toks[0], &val))
            }
            if m == "v_div_fmas_f32" {
                s.push(get("vcc", &val))
            }
            (vec![key(toks[0])], s)
        };
        let v = format!("{op}({})", srcs.join(","));
        for (k, d) in dsts.iter().enumerate() {
            if d == "null" {
                continue;
            }
            if !d.ends_with(".l") && !d.ends_with(".h") {
                val.remove(&format!("{d}.l"));
                val.remove(&format!("{d}.h"));
            }
            val.insert(
                d.clone(),
                if dsts.len() > 1 {
                    format!("{v}#{k}")
                } else {
                    v.clone()
                },
            );
        }
    }
    outs.iter().map(|o| get(o, &val)).collect()
}

/// The four goldens (conv+SiLU per channel, q norm, k norm, v conversion)
/// with their interface headers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Goldens {
    pub conv_silu: String,
    pub norm_q: String,
    pub norm_k: String,
    pub cvt_v: String,
}

/// Slice `gdn_chunk_prep`'s per-token loop out of its canonical lifted text.
/// The loop head is the b128 load of the current token's four channels right
/// after the latch that shifts the conv window; the body ends at the four
/// `v_div_fixup_f32`. The v path (plain FP16) and the q/k path (head norm)
/// follow; the fourth channel's FP16 conversion sits in the latch both paths
/// rejoin (a software-pipelined store).
pub fn slice_gdn(lines: &[String]) -> Result<Goldens, String> {
    let ops = split(lines);
    let find = |from: usize, pred: &dyn Fn(&Line) -> bool| {
        (from..ops.len())
            .find(|&k| pred(&ops[k]))
            .ok_or("slice anchor not found")
    };
    let load = (0..ops.len())
        .find(|&k| {
            ops[k].mnemonic == "global_load_b128"
                && ops[k.saturating_sub(24)..k]
                    .iter()
                    .filter(|o| o.mnemonic == "v_mov_b32_e32")
                    .count()
                    >= 12
        })
        .ok_or("per-token load not found")?;
    let cur = regs_of(ops[load].args.split(',').next().ok_or("load dst")?);
    if cur.len() != 4 {
        return Err("current-token load is not four channels".into());
    }
    let latch = &ops[load.saturating_sub(24)..load];
    let copy: BTreeMap<String, String> = latch
        .iter()
        .filter(|o| o.mnemonic == "v_mov_b32_e32")
        .filter_map(|l| {
            let t: Vec<&str> = l.args.split(',').map(str::trim).collect();
            Some((
                regs_of(t.first()?).first()?.clone(),
                regs_of(t.get(1)?).first()?.clone(),
            ))
        })
        .collect();
    let back = |v: &str| {
        copy.iter()
            .find(|(_, s)| s.as_str() == v)
            .map(|(d, _)| d.clone())
            .ok_or_else(|| format!("no window copy of {v}"))
    };
    let fixups: Vec<usize> = (load..ops.len())
        .filter(|&k| ops[k].mnemonic == "v_div_fixup_f32")
        .take(4)
        .collect();
    if fixups.len() != 4 {
        return Err("expected four channel quotients".into());
    }
    let body = &ops[load + 1..=fixups[3]];
    let mut channel_out = vec![String::new(); 4];
    let mut per_channel: Vec<(Vec<Line>, BTreeMap<String, String>)> =
        vec![(vec![], BTreeMap::new()); 4];
    for &f in &fixups {
        let out = regs_of(ops[f].args.split(',').next().ok_or("fixup dst")?)[0].clone();
        let upto: Vec<Line> = body
            .iter()
            .filter(|l| l.index <= ops[f].index)
            .cloned()
            .collect();
        let (slice, live) = backward(&upto, &[out.as_str()]);
        let c = cur
            .iter()
            .position(|r| live.contains(r))
            .ok_or("channel slice reads no current value")?;
        if !channel_out[c].is_empty() {
            return Err(format!("two quotients for channel {c}"));
        }
        let win2 = back(&cur[c])?;
        let win1 = back(&win2)?;
        let win0 = back(&win1)?;
        let mut roles: BTreeMap<String, String> = BTreeMap::new();
        for (r, n) in [
            (&cur[c], "cur"),
            (&win2, "win2"),
            (&win1, "win1"),
            (&win0, "win0"),
        ] {
            roles.insert(r.clone(), n.into());
        }
        // y = fmaf(w0, win0, fmaf(w1, win1, fmaf(w3, cur, w2 * win2))): one mul and three fmacs on one register.
        if slice.len() < 4 {
            return Err(format!("channel {c} slice too short"));
        }
        let conv = &slice[..4];
        let acc = regs_of(conv[0].args.split(',').next().ok_or("conv dst")?);
        for (l, (mnemonic, tap, win)) in conv.iter().zip([
            ("v_mul_f32_e32", "w2", &win2),
            ("v_fmac_f32_e32", "w3", &cur[c]),
            ("v_fmac_f32_e32", "w1", &win1),
            ("v_fmac_f32_e32", "w0", &win0),
        ]) {
            let t: Vec<&str> = l.args.split(',').map(str::trim).collect();
            if l.mnemonic != mnemonic || t.len() != 3 || regs_of(t[0]) != acc {
                return Err(format!(
                    "channel {c} conv op `{} {}` out of shape",
                    l.mnemonic, l.args
                ));
            }
            let (a, b) = (regs_of(t[1]), regs_of(t[2]));
            let weight = if a.first() == Some(win) {
                b
            } else if b.first() == Some(win) {
                a
            } else {
                return Err(format!("channel {c} tap {tap} misses its window value"));
            };
            roles.insert(weight.first().ok_or("tap weight")?.clone(), tap.into());
        }
        if let Some(r) = live.iter().find(|r| !roles.contains_key(*r)) {
            return Err(format!("channel {c} slice reads unnamed live-in {r}"));
        }
        channel_out[c] = out;
        per_channel[c] = (slice, roles);
    }
    // All four channels compute the same function of their roles.
    let numbered: Vec<Vec<String>> = (0..4)
        .map(|c| {
            value_numbers(
                &per_channel[c].0,
                &per_channel[c].1,
                &[channel_out[c].as_str()],
            )
        })
        .collect();
    if numbered.iter().any(|n| *n != numbered[0]) {
        return Err("channel slices differ".into());
    }
    // Golden: the channel whose division scale writes VCC directly (no `s_mov_b32 vcc_lo, sN`).
    let pick = (0..4)
        .rev()
        .find(|&c| !per_channel[c].0.iter().any(|l| l.mnemonic == "s_mov_b32"))
        .ok_or("no direct-VCC channel")?;
    let (slice, roles) = &per_channel[pick];
    let named = |role: &str| {
        roles
            .iter()
            .find(|(_, n)| n.as_str() == role)
            .map(|(r, _)| r.clone())
            .unwrap_or_default()
    };
    let conv_silu = format!(
        "; inputs: w2={} win2={} w3={} cur={} w1={} win1={} w0={} win0={}\n; outputs: out={}\n{}",
        named("w2"),
        named("win2"),
        named("w3"),
        named("cur"),
        named("w1"),
        named("win1"),
        named("w0"),
        named("win0"),
        channel_out[pick],
        text(&with_vcc_waits(slice))
    );

    let latch_cvt = latch
        .iter()
        .find(|o| o.mnemonic == "v_cvt_f16_f32_e32")
        .ok_or("latch conversion missing")?
        .clone();
    let branch = find(fixups[3] + 1, &|l| l.mnemonic == "s_cbranch_execz")?;
    let v_end = find(branch, &|l| l.mnemonic == "global_store_b32")?;
    let qk_start = find(v_end, &|l| l.mnemonic == "s_cbranch_execz")? + 1;
    let qk_end = find(qk_start, &|l| l.mnemonic == "global_store_b32")?;
    let path = |from: usize, to: usize| -> Vec<Line> {
        ops[from..=to]
            .iter()
            .cloned()
            .chain(std::iter::once(latch_cvt.clone()))
            .collect()
    };
    let (v_seq, qk_seq) = (path(branch + 1, v_end), path(qk_start, qk_end));
    if v_seq.iter().any(|l| l.mnemonic == "ds_bpermute_b32")
        || !qk_seq.iter().any(|l| l.mnemonic == "ds_bpermute_b32")
    {
        return Err("v and q/k paths are not in the expected order".into());
    }
    // Stored registers: b32 = halves 0/1, b16 = half 2, latch conversion = half 3 (checked by value below).
    let stored = |seq: &[Line]| -> Result<[String; 3], String> {
        let data = |m: &str| {
            seq.iter()
                .rev()
                .find(|l| l.mnemonic == m)
                .and_then(|l| l.args.split(',').nth(1))
                .and_then(|t| regs_of(t).first().cloned())
                .ok_or(format!("{m} data"))
        };
        let latch_dst = regs_of(latch_cvt.args.split(',').next().unwrap_or(""))
            .first()
            .cloned()
            .ok_or("latch dst")?;
        Ok([
            data("global_store_b32")?,
            data("global_store_b16")?,
            latch_dst,
        ])
    };
    let outs = |r: &[String; 3]| {
        [
            format!("{}.l", r[0]),
            format!("{}.h", r[0]),
            format!("{}.l", r[1]),
            format!("{}.l", r[2]),
        ]
    };
    let ch_roles: BTreeMap<String, String> = (0..4)
        .map(|c| (channel_out[c].clone(), format!("o{c}")))
        .collect();
    let ins = format!(
        "; inputs: o0={} o1={} o2={} o3={}\n",
        channel_out[0], channel_out[1], channel_out[2], channel_out[3]
    );

    // b32 stores both halves; the b16 stores and the latch conversion carry low halves.
    let store_targets =
        |r: &[String; 3]| [r[0].clone(), format!("{}.l", r[1]), format!("{}.l", r[2])];
    let rv = stored(&v_seq)?;
    let vts = store_targets(&rv);
    let vt: Vec<&str> = vts.iter().map(String::as_str).collect();
    let (v_slice, _) = backward(&v_seq, &vt);
    let ov = outs(&rv);
    let vv = value_numbers(
        &v_slice,
        &ch_roles,
        &ov.iter().map(String::as_str).collect::<Vec<_>>(),
    );
    for (c, v) in vv.iter().enumerate() {
        if *v != format!("v_cvt_f16_f32(o{c})") {
            return Err(format!(
                "v half {c} is {v}, not the conversion of channel {c}"
            ));
        }
    }
    let cvt_v = format!(
        "{ins}; outputs: h0={} h1={} h2={} h3={}\n{}",
        ov[0],
        ov[1],
        ov[2],
        ov[3],
        text(&v_slice)
    );

    let rq = stored(&qk_seq)?;
    let qts = store_targets(&rq);
    let qt: Vec<&str> = qts.iter().map(String::as_str).collect();
    let (qk_slice, qk_live) = backward(&qk_seq, &qt);
    let lane = qk_live
        .iter()
        .find(|r| r.starts_with('v') && !channel_out.contains(*r))
        .cloned()
        .ok_or("lane id live-in")?;
    let sgpr_operand = |m: &str| {
        qk_slice
            .iter()
            .find(|l| {
                l.mnemonic == m
                    && l.args
                        .split(',')
                        .nth(1)
                        .is_some_and(|t| t.trim().starts_with('s'))
            })
            .and_then(|l| regs_of(l.args.split(',').nth(1)?).first().cloned())
            .ok_or(format!("SGPR operand of {m}"))
    };
    let eps = sgpr_operand("v_add_f32_e32")?;
    let qscale = sgpr_operand("v_mul_f32_e32")?;
    let select = qk_live
        .iter()
        .find(|r| r.starts_with('s') && **r != eps && **r != qscale)
        .cloned()
        .ok_or("unit-class select mask")?;
    // `v_cndmask_b32_e64 d, scaled, plain, select` picks by unit class (select = unit > 15):
    // q keeps the q_scale product (as a copy), k the plain normalised value.
    let specialise = |is_q: bool| -> Result<String, String> {
        let mut kept = Vec::new();
        for l in &qk_slice {
            let t: Vec<&str> = l.args.split(',').map(str::trim).collect();
            if l.mnemonic == "v_cndmask_b32_e64" && t.len() == 4 && t[3] == select {
                let chosen = if is_q { t[1] } else { t[2] };
                if chosen != t[0] {
                    kept.push(Line {
                        index: l.index,
                        mnemonic: "v_mov_b32_e32".into(),
                        args: format!("{}, {chosen}", t[0]),
                    })
                }
                continue;
            }
            kept.push(l.clone());
        }
        let (live, _) = backward(&kept, &qt);
        let o = outs(&rq);
        let mut sroles = ch_roles.clone();
        sroles.insert(eps.clone(), "eps".into());
        sroles.insert(qscale.clone(), "qscale".into());
        sroles.insert(lane.clone(), "lane".into());
        let vals = value_numbers(
            &live,
            &sroles,
            &o.iter().map(String::as_str).collect::<Vec<_>>(),
        );
        // Every half is FP16(z_c) with z_c = [qscale *] (inv * o_c) and one shared inv.
        let mut inv: Option<String> = None;
        for (c, v) in vals.iter().enumerate() {
            let z = v
                .strip_prefix("v_cvt_f16_f32(")
                .and_then(|s| s.strip_suffix(')'))
                .ok_or_else(|| format!("half {c} is not an FP16 conversion: {v}"))?;
            let z = if is_q {
                z.strip_prefix("v_mul_f32(qscale,")
                    .and_then(|s| s.strip_suffix(')'))
                    .ok_or_else(|| format!("q half {c} lacks q_scale: {z}"))?
            } else {
                z
            };
            let i = z
                .strip_prefix("v_mul_f32(")
                .and_then(|s| s.strip_suffix(&format!(",o{c})")))
                .ok_or_else(|| format!("half {c} is not inv * o{c}: {z}"))?;
            if inv.get_or_insert_with(|| i.to_owned()) != i {
                return Err("halves disagree on the norm".into());
            }
        }
        let sin = if is_q {
            format!("; sinputs: eps={eps} qscale={qscale}\n")
        } else {
            format!("; sinputs: eps={eps}\n")
        };
        Ok(format!(
            "{ins}{sin}; lane: {lane}\n; outputs: h0={} h1={} h2={} h3={}\n{}",
            o[0],
            o[1],
            o[2],
            o[3],
            text(&with_vcc_waits(&live))
        ))
    };
    Ok(Goldens {
        conv_silu,
        norm_q: specialise(true)?,
        norm_k: specialise(false)?,
        cvt_v,
    })
}

/// The committed golden with its provenance comments removed (headers kept).
pub fn golden_body(golden: &str) -> String {
    golden
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            !t.is_empty()
                && (!t.starts_with(';')
                    || [";inputs", "; inputs", "; sinputs", "; lane", "; outputs"]
                        .iter()
                        .any(|p| t.starts_with(p)))
        })
        .map(|l| format!("{l}\n"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goldens_parse_with_their_interfaces() {
        let conv = Region::conv_silu().unwrap();
        assert_eq!(
            conv.inputs,
            ["w2", "win2", "w3", "cur", "w1", "win1", "w0", "win0"]
        );
        assert_eq!(conv.outputs, ["out"]);
        assert_eq!(conv.masks, 2);
        for (region, sin) in [
            (Region::norm_q().unwrap(), 2),
            (Region::norm_k().unwrap(), 1),
        ] {
            assert_eq!(region.inputs, ["o0", "o1", "o2", "o3"]);
            assert_eq!(region.sinputs.len(), sin);
            assert_eq!(region.outputs, ["h0", "h1", "h2", "h3"]);
            assert_eq!(
                region
                    .ops
                    .iter()
                    .filter(|o| o.exchange.is_some())
                    .map(|o| o.exchange.unwrap())
                    .collect::<Vec<_>>(),
                [16, 8, 4, 2, 1]
            );
        }
        let v = Region::cvt_v().unwrap();
        assert_eq!(v.mnemonics(), ["v_cvt_f16_f32_e32"; 4]);
    }

    /// The lean region keeps the imported conv MACs and interface; only the
    /// SiLU after them is replaced (its equivalence is the exhaustive run).
    #[test]
    fn lean_conv_silu_keeps_the_golden_conv_and_interface() {
        let body = |g: &str| {
            golden_body(g)
                .lines()
                .filter(|l| !l.starts_with(';'))
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        let (gold, lean) = (body(CONV_SILU_GOLDEN), body(CONV_SILU_LEAN));
        assert_eq!(gold[..4], lean[..4]);
        let (g, l) = (
            Region::conv_silu().unwrap(),
            Region::conv_silu_lean().unwrap(),
        );
        assert_eq!(
            (l.inputs, l.outputs, l.sinputs),
            (g.inputs, g.outputs, vec!["clamp".to_owned()])
        );
    }
}
