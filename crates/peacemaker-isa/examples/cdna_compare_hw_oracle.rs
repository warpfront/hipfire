// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx942 hardware-vs-interpreter oracle of the complete floating compare family:
//! all 64 `v_cmp_*` / `v_cmpx_*` F16/F32 operations (16 predicates x CMP/CMPX x 2 widths) plus the extra
//! `v_cmp_ge_u64` row. Every operation is measured in its real opcode and encoding:
//!
//! * VOPC E32 (implicit VCC), VOP3 E64 with a VCC and with a non-VCC even SGPR-pair destination, every
//!   legal abs/neg combination the codec admits for the opcode, and VOPC SDWA (the only legal F16 half
//!   selects on gfx942: WORD_0/WORD_1 for both sources, with and without neg/abs) with VCC and SGPR-pair
//!   destinations. VOP3 op_sel is only exercised if the codec claims it legal for the opcode.
//!   VOP3 clamp=1 is also run for every legal neg/abs combination (mask/EXEC are expected to ignore it).
//! * Each instruction is a separate inline-asm block on fixed physical registers (v250.., s[20:21], vcc)
//!   so the instruction text printed by the codec is exactly what LLVM assembles; the code object is
//!   searched for the codec encoding of every instruction (exactly one hit) before anything runs.
//! * The block loads an explicit EXEC, incoming VCC and incoming SGPR pair, runs the compare, snapshots VCC,
//!   the SGPR pair and the post-instruction EXEC (also for EXEC = 0 and for CMPX) and only then restores EXEC
//!   for the global stores. Lane l <-> mask bit l (bit 63 included) is proven by walking one-hot waves.
//! * Each (case, variant) record = {VCC, s[20:21], EXEC}. Hardware is compared bit-exactly against (a) the
//!   `peacemaker-cdna` interpreter and (b) an independent per-lane truth model written here from the
//!   predicate names (ordered LT/EQ/LE/GT/LG/GE, O, U, N* complements true on NaN, F, TRU) with explicit
//!   input-denormal flush when the width's FP_DENORM bit0 is 0.
//! * MODE is written with s_setreg and read back before and after the instructions: FP_DENORM 0..3 and
//!   FP_ROUND {0,1,3} for the operation's width, while the other width's fields are driven through a
//!   different combination.
//! * Output allocations are guarded, run under two poison patterns, and planted corruption controls of
//!   mask and EXEC records are run through the same comparator (each MUST be detected).
//!
//! No dry-run, no tolerance. Hardware/interpreter/truth mismatches are reported with exact evidence and
//! make the oracle FAIL; hipcc/launch/guard/mode-readback problems are immediate errors.
//! Run only in a fabric job:
//! `cargo run --release --locked -p peacemaker-isa --features lab --example cdna_compare_hw_oracle -- OUT_DIR`.
use hip_bridge::{DeviceBuffer, Function, HipRuntime, Module};
use peacemaker_cdna::enc::vop::{Sdwa, SdwaSel, SdwaUnused, VopExt, Vopc, VopcOp};
use peacemaker_cdna::enc::vop3::{Vop3, Vop3Op};
use peacemaker_cdna::sim::{exec_one, Config, Memory, Mode, Wave};
use peacemaker_cdna::{Arch, Inst, Src};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const GUARD: usize = 64;
const POISONS: [u8; 2] = [0x5a, 0xa5];
const SEED: u64 = 0x2c1d_7a95_e0b3_4f68;
const DEFAULT_MODE: u32 = 0x3f0;
const MAX_SAMPLES: usize = 64;
const PRED_NAMES: [&str; 16] = [
    "f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt",
    "tru",
];

const E32: [u32; 32] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x807f_ffff,
    0x0080_0000,
    0x8080_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x3f80_0001,
    0xbf80_0001,
    0x3f7f_ffff,
    0x4000_0000,
    0xc000_0000,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0xffc0_0000,
    0x7f80_0001,
    0xff80_0001,
    0x7fff_ffff,
    0xffff_ffff,
    0x7fa0_0000,
    0xffbf_ffff,
    0x7fc0_0001,
    0x0040_0000,
    0x8040_0000,
    0x3f00_0000,
    0xc2f6_e979,
];
const E16: [u16; 32] = [
    0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x3c00, 0xbc00, 0x3c01, 0xbc01,
    0x3bff, 0x4000, 0xc000, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0xfe00, 0x7c01, 0xfc01, 0x7fff,
    0xffff, 0x7d00, 0xfdff, 0x7e01, 0x0200, 0x8200, 0x3800, 0x5640,
];
const E64: [u64; 32] = [
    0,
    1,
    2,
    3,
    u64::MAX,
    u64::MAX - 1,
    u64::MAX - 2,
    1 << 63,
    (1 << 63) + 1,
    (1 << 63) - 1,
    0xffff_ffff,
    0x1_0000_0000,
    0x1_ffff_ffff,
    0xffff_ffff_0000_0000,
    0xffff_fffe_ffff_ffff,
    0x0000_0001_0000_0001,
    0x8000_0000,
    0x7fff_ffff,
    0x8000_0001,
    0x1_8000_0000,
    0x8000_0000_8000_0000,
    0x7fff_ffff_ffff_fffe,
    0x0123_4567_89ab_cdef,
    0xfedc_ba98_7654_3210,
    0x0000_ffff_0000_ffff,
    0xffff_0000_ffff_0000,
    0x5555_5555_5555_5555,
    0xaaaa_aaaa_aaaa_aaaa,
    0x00ff_00ff_00ff_00ff,
    0x1_0000,
    0xffff,
    0x2_0000_0000,
];

fn sha(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn h64(x: u64) -> String {
    format!("{x:#018x}")
}
fn h32(x: u32) -> String {
    format!("{x:#010x}")
}
fn words_bytes(w: &[u64]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_u64(b: &[u8]) -> Vec<u64> {
    b.chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().expect("8-byte chunk")))
        .collect()
}
fn bytes_u32(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("4-byte chunk")))
        .collect()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn u32(&mut self) -> u32 {
        (self.next() >> 32) as u32
    }
}

// ----------------------------------------------------------------------------
// Operations and instruction variants

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    F16,
    F32,
    U64,
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dst {
    Vcc,
    S20,
}
struct OpInfo {
    op: VopcOp,
    name: &'static str,
    kind: Kind,
    cmpx: bool,
    pred: usize,
    tok: String,
}
struct Var {
    label: String,
    inst: Inst,
    dst: Dst,
    /// Source k reads the high 16-bit half (F16 only).
    half: [bool; 2],
    neg: u8,
    abs: u8,
    text: String,
    words: Vec<u32>,
}

fn ops() -> Result<Vec<OpInfo>, String> {
    let mut v = Vec::new();
    let mut seen = BTreeSet::new();
    for &op in VopcOp::ALL {
        let Some((width, pred)) = op.float_compare() else {
            continue;
        };
        let name = op.name();
        let cmpx = op.is_cmpx();
        let rest = name
            .strip_prefix(if cmpx { "v_cmpx_" } else { "v_cmp_" })
            .ok_or_else(|| format!("{name}: prefix does not match is_cmpx() = {cmpx}"))?;
        let (tok, w) = if let Some(t) = rest.strip_suffix("_f16") {
            (t, 16)
        } else if let Some(t) = rest.strip_suffix("_f32") {
            (t, 32)
        } else {
            return Err(format!("{name}: not an f16/f32 name"));
        };
        if w != width {
            return Err(format!(
                "{name}: width {width} from float_compare vs {w} from the name"
            ));
        }
        if PRED_NAMES.get(usize::from(pred)).copied() != Some(tok) {
            return Err(format!(
                "{name}: predicate nibble {pred} does not match predicate token `{tok}`"
            ));
        }
        let opcode = op
            .opcode(Arch::Gfx942)
            .ok_or_else(|| format!("{name}: no gfx942 opcode"))?;
        let want = (if width == 16 { 32u8 } else { 64u8 }) + (if cmpx { 16u8 } else { 0u8 }) + pred;
        if opcode != want {
            return Err(format!("{name}: opcode {opcode} != {want}"));
        }
        if op.src_dwords() != 1 {
            return Err(format!("{name}: src_dwords != 1"));
        }
        if !seen.insert((width, cmpx, pred)) {
            return Err(format!("{name}: duplicate (width, cmpx, predicate)"));
        }
        v.push(OpInfo {
            op,
            name,
            kind: if width == 16 { Kind::F16 } else { Kind::F32 },
            cmpx,
            pred: usize::from(pred),
            tok: tok.to_string(),
        });
    }
    if v.len() != 64 {
        return Err(format!(
            "expected 64 float compare ops, VopcOp::ALL has {}",
            v.len()
        ));
    }
    let ge = VopcOp::CmpGeU64;
    if !VopcOp::ALL.contains(&ge)
        || ge.name() != "v_cmp_ge_u64"
        || ge.src_dwords() != 2
        || ge.is_cmpx()
    {
        return Err("VopcOp::CmpGeU64 is not the 64-bit v_cmp_ge_u64".into());
    }
    v.push(OpInfo {
        op: ge,
        name: ge.name(),
        kind: Kind::U64,
        cmpx: false,
        pred: 6,
        tok: "ge".into(),
    });
    Ok(v)
}

fn dst_src(d: Dst) -> Src {
    match d {
        Dst::Vcc => Src::VccLo,
        Dst::S20 => Src::Sgpr(20),
    }
}

fn mk_var(
    label: String,
    inst: Inst,
    dst: Dst,
    half: [bool; 2],
    neg: u8,
    abs: u8,
) -> Result<Var, String> {
    let mut words = Vec::new();
    inst.encode(Arch::Gfx942, &mut words)
        .map_err(|e| format!("{label}: codec encode: {e:?}"))?;
    let (back, n) =
        Inst::decode(Arch::Gfx942, &words).map_err(|e| format!("{label}: codec decode: {e:?}"))?;
    let text = inst.to_string();
    if n != words.len() || back.to_string() != text {
        return Err(format!(
            "{label}: codec decode round trip `{}` != `{text}`",
            back
        ));
    }
    Ok(Var {
        label,
        inst,
        dst,
        half,
        neg,
        abs,
        text,
        words,
    })
}

fn variants(o: &OpInfo) -> Result<Vec<Var>, String> {
    let s0 = Src::Vgpr(250);
    let s1 = if o.kind == Kind::U64 {
        Src::Vgpr(252)
    } else {
        Src::Vgpr(251)
    };
    let mut v = Vec::new();
    let vop3 = |dst: Dst, abs: u8, neg: u8, op_sel: u8, clamp: bool| {
        Inst::Vop3(Vop3 {
            op: Vop3Op::C(o.op),
            dst: dst_src(dst),
            sdst: None,
            src: [Some(s0), Some(s1), None],
            abs,
            neg,
            clamp,
            omod: 0,
            op_sel,
        })
    };
    v.push(mk_var(
        "e32".into(),
        Inst::Vopc(Vopc {
            op: o.op,
            src0: s0,
            src1: s1,
            ext: VopExt::None,
        }),
        Dst::Vcc,
        [false; 2],
        0,
        0,
    )?);
    v.push(mk_var(
        "e64_vcc".into(),
        vop3(Dst::Vcc, 0, 0, 0, false),
        Dst::Vcc,
        [false; 2],
        0,
        0,
    )?);
    v.push(mk_var(
        "e64_s20".into(),
        vop3(Dst::S20, 0, 0, 0, false),
        Dst::S20,
        [false; 2],
        0,
        0,
    )?);
    let m = Vop3Op::C(o.op).mods();
    let mut alt = 0;
    for neg in 0..=3u8 {
        for abs in 0..=3u8 {
            if (neg | abs) == 0 || neg & !m.neg != 0 || abs & !m.abs != 0 {
                continue;
            }
            let dst = if alt % 2 == 0 { Dst::Vcc } else { Dst::S20 };
            alt += 1;
            v.push(mk_var(
                format!(
                    "e64_{}_neg{neg}_abs{abs}",
                    if dst == Dst::Vcc { "vcc" } else { "s20" }
                ),
                vop3(dst, abs, neg, 0, false),
                dst,
                [false; 2],
                neg,
                abs,
            )?);
        }
    }
    // VOP3 clamp=1 (the codec advertises it for the float compares; the interpreter ignores it for
    // masks): every legal neg/abs combination, both destination kinds for the plain form.
    if m.clamp {
        let mut alt = 0;
        for neg in 0..=3u8 {
            for abs in 0..=3u8 {
                if neg & !m.neg != 0 || abs & !m.abs != 0 {
                    continue;
                }
                let dsts: &[Dst] = if (neg | abs) == 0 {
                    &[Dst::Vcc, Dst::S20]
                } else {
                    alt += 1;
                    if alt % 2 == 1 {
                        &[Dst::Vcc]
                    } else {
                        &[Dst::S20]
                    }
                };
                for &dst in dsts {
                    v.push(mk_var(
                        format!(
                            "e64_{}_neg{neg}_abs{abs}_clamp",
                            if dst == Dst::Vcc { "vcc" } else { "s20" }
                        ),
                        vop3(dst, abs, neg, 0, true),
                        dst,
                        [false; 2],
                        neg,
                        abs,
                    )?);
                }
            }
        }
    }
    // VOP3 op_sel only where the codec claims it legal for this opcode (gfx942 F16/F32 compares: none).
    for op_sel in 1..16u8 {
        if op_sel & !m.op_sel != 0 || o.kind != Kind::F16 {
            continue;
        }
        v.push(mk_var(
            format!("e64_vcc_opsel{op_sel}"),
            vop3(Dst::Vcc, 0, 0, op_sel, false),
            Dst::Vcc,
            [op_sel & 1 != 0, op_sel & 2 != 0],
            0,
            0,
        )?);
    }
    let sdwa = |s0sel: SdwaSel, s1sel: SdwaSel, neg: u8, abs: u8, dst: Dst| {
        Inst::Vopc(Vopc {
            op: o.op,
            src0: s0,
            src1: s1,
            ext: VopExt::Sdwa(Sdwa {
                dst_sel: SdwaSel::Dword,
                dst_unused: SdwaUnused::Pad,
                clamp: false,
                omod: 0,
                src0_sel: s0sel,
                src0_sext: false,
                src0_neg: neg & 1 != 0,
                src0_abs: abs & 1 != 0,
                src1_sel: s1sel,
                src1_sext: false,
                src1_neg: neg & 2 != 0,
                src1_abs: abs & 2 != 0,
                sdst: match dst {
                    Dst::Vcc => None,
                    Dst::S20 => Some(Src::Sgpr(20)),
                },
            }),
        })
    };
    let dname = |d: Dst| if d == Dst::Vcc { "vcc" } else { "s20" };
    match o.kind {
        Kind::F16 => {
            let mut list = Vec::new();
            for (a, b) in [(false, false), (false, true), (true, false), (true, true)] {
                for d in [Dst::Vcc, Dst::S20] {
                    list.push((a, b, 0u8, 0u8, d));
                }
            }
            list.push((true, true, 3, 0, Dst::Vcc));
            list.push((false, true, 1, 2, Dst::S20));
            list.push((true, false, 2, 3, Dst::Vcc));
            for (a, b, neg, abs, d) in list {
                let sel = |h: bool| if h { SdwaSel::Word1 } else { SdwaSel::Word0 };
                v.push(mk_var(
                    format!(
                        "sdwa_{}{}_{}_neg{neg}_abs{abs}",
                        if a { "w1" } else { "w0" },
                        if b { "w1" } else { "w0" },
                        dname(d)
                    ),
                    sdwa(sel(a), sel(b), neg, abs, d),
                    d,
                    [a, b],
                    neg,
                    abs,
                )?);
            }
        }
        Kind::F32 => {
            for (neg, abs, d) in [
                (0u8, 0u8, Dst::Vcc),
                (0, 0, Dst::S20),
                (3, 0, Dst::Vcc),
                (1, 2, Dst::S20),
            ] {
                v.push(mk_var(
                    format!("sdwa_dw_{}_neg{neg}_abs{abs}", dname(d)),
                    sdwa(SdwaSel::Dword, SdwaSel::Dword, neg, abs, d),
                    d,
                    [false; 2],
                    neg,
                    abs,
                )?);
            }
        }
        Kind::U64 => {}
    }
    Ok(v)
}

// ----------------------------------------------------------------------------
// Independent truth model

fn f32_operand(raw: u32, neg: bool, abs: bool, flush: bool) -> Option<f64> {
    let mut b = raw;
    if abs {
        b &= 0x7fff_ffff;
    }
    if neg {
        b ^= 0x8000_0000;
    }
    let (e, m) = ((b >> 23) & 0xff, b & 0x7f_ffff);
    if e == 0xff && m != 0 {
        return None;
    }
    if flush && e == 0 && m != 0 {
        return Some(0.0);
    }
    Some(f64::from(f32::from_bits(b)))
}
fn f16_operand(raw: u16, neg: bool, abs: bool, flush: bool) -> Option<f64> {
    let mut h = raw;
    if abs {
        h &= 0x7fff;
    }
    if neg {
        h ^= 0x8000;
    }
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (i32::from((h >> 10) & 31), f64::from(h & 0x3ff));
    Some(if e == 31 {
        if m != 0.0 {
            return None;
        }
        s * f64::INFINITY
    } else if e == 0 {
        if flush {
            s * 0.0
        } else {
            s * m * 2f64.powi(-24)
        }
    } else {
        s * (1.0 + m / 1024.0) * 2f64.powi(e - 15)
    })
}
fn eval_pred(tok: &str, x: Option<f64>, y: Option<f64>) -> bool {
    let ord = |t: &str| -> bool {
        match (x, y) {
            (Some(a), Some(b)) => match t {
                "lt" => a < b,
                "eq" => a == b,
                "le" => a <= b,
                "gt" => a > b,
                "lg" => a < b || a > b,
                "ge" => a >= b,
                _ => false,
            },
            _ => false,
        }
    };
    match tok {
        "f" => false,
        "tru" => true,
        "o" => x.is_some() && y.is_some(),
        "u" => x.is_none() || y.is_none(),
        "nge" | "nlg" | "ngt" | "nle" | "neq" | "nlt" => !ord(&tok[1..]),
        t => ord(t),
    }
}
fn lane_truth(o: &OpInfo, half: [bool; 2], neg: u8, abs: u8, mode: Mode, a: u64, b: u64) -> bool {
    match o.kind {
        Kind::U64 => a >= b,
        Kind::F32 => {
            let flush = mode.denorm32 & 1 == 0;
            eval_pred(
                &o.tok,
                f32_operand(a as u32, neg & 1 != 0, abs & 1 != 0, flush),
                f32_operand(b as u32, neg & 2 != 0, abs & 2 != 0, flush),
            )
        }
        Kind::F16 => {
            let flush = mode.denorm16_64 & 1 == 0;
            let sel = |x: u64, h: bool| (x >> if h { 16 } else { 0 }) as u16;
            eval_pred(
                &o.tok,
                f16_operand(sel(a, half[0]), neg & 1 != 0, abs & 1 != 0, flush),
                f16_operand(sel(b, half[1]), neg & 2 != 0, abs & 2 != 0, flush),
            )
        }
    }
}

// ----------------------------------------------------------------------------
// Cases

struct CaseSet {
    /// Per lane operand words (F32: the dword; F16: lo | hi << 16; U64: the quadword).
    a: Vec<u64>,
    b: Vec<u64>,
    ctx: Vec<[u64; 3]>,
    label: Vec<&'static str>,
}
impl CaseSet {
    fn n(&self) -> usize {
        self.ctx.len()
    }
    fn push(&mut self, lanes: &[(u64, u64)], ex: u64, label: &'static str) {
        let c = self.n() as u64;
        for &(a, b) in lanes {
            self.a.push(a);
            self.b.push(b);
        }
        let vin = match c % 4 {
            0 => u64::MAX,
            1 => 0,
            _ => c.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x5bd1_e995_a5a5_5a5a,
        };
        let sin = match (c / 4) % 4 {
            0 => 0,
            1 => u64::MAX,
            _ => c.wrapping_mul(0xd6e8_feb8_6659_fd93) ^ 0x1234_5678_9abc_def0,
        };
        self.ctx.push([ex, vin, sin]);
        self.label.push(label);
    }
}

fn pair32(r: &mut Rng) -> (u32, u32) {
    match r.below(8) {
        0..=2 => (r.u32(), r.u32()),
        3 => {
            let x = r.u32();
            (x, x)
        }
        4 => {
            let x = r.u32();
            (x, x ^ 0x8000_0000)
        }
        5 => {
            let x = r.u32();
            (x, x.wrapping_add(1 + r.below(3) as u32))
        }
        6 => (E32[r.below(32) as usize], r.u32()),
        _ => (E32[r.below(32) as usize], E32[r.below(32) as usize]),
    }
}
fn pair16(r: &mut Rng) -> (u16, u16) {
    match r.below(8) {
        0..=2 => (r.u32() as u16, r.u32() as u16),
        3 => {
            let x = r.u32() as u16;
            (x, x)
        }
        4 => {
            let x = r.u32() as u16;
            (x, x ^ 0x8000)
        }
        5 => {
            let x = r.u32() as u16;
            (x, x.wrapping_add(1 + r.below(3) as u16))
        }
        6 => (E16[r.below(32) as usize], r.u32() as u16),
        _ => (E16[r.below(32) as usize], E16[r.below(32) as usize]),
    }
}
fn pair64(r: &mut Rng) -> (u64, u64) {
    match r.below(8) {
        0 => (r.next(), r.next()),
        1 => {
            let x = r.next();
            (x, x)
        }
        2 => {
            let x = r.next();
            (x, (x & !0xffff_ffff) | (r.next() & 0xffff_ffff))
        }
        3 => {
            let x = r.next();
            (x, (x & 0xffff_ffff) | (r.next() & !0xffff_ffff))
        }
        4 => {
            let x = r.next();
            (x, x.wrapping_add(1).wrapping_sub(2 * r.below(2)))
        }
        5 => (E64[r.below(32) as usize], r.next()),
        6 => (E64[r.below(32) as usize], E64[r.below(32) as usize]),
        _ => {
            let x = r.next();
            (x, (x & !0xffff_ffff) | (x.rotate_left(17) & 0xffff_ffff))
        }
    }
}
fn rand_lane(kind: Kind, r: &mut Rng) -> (u64, u64) {
    match kind {
        Kind::F32 => {
            let (a, b) = pair32(r);
            (u64::from(a), u64::from(b))
        }
        Kind::F16 => {
            let (la, lb) = pair16(r);
            let (ha, hb) = pair16(r);
            (
                u64::from(la) | u64::from(ha) << 16,
                u64::from(lb) | u64::from(hb) << 16,
            )
        }
        Kind::U64 => pair64(r),
    }
}
/// Representative operand pair per relation: LT, EQ, GT, UN (U64: no UN).
fn rel_rep(kind: Kind, rel: usize, hi: bool) -> (u64, u64) {
    match kind {
        Kind::F32 => [
            (0x3f80_0000u64, 0x4000_0000u64),
            (0x3fc0_0000, 0x3fc0_0000),
            (0x4000_0000, 0x3f80_0000),
            (0x7fc0_0000, 0x3f80_0000),
        ][rel],
        Kind::F16 => {
            let reps = [
                (0x3c00u64, 0x4000u64),
                (0x3e00, 0x3e00),
                (0x4000, 0x3c00),
                (0x7e00, 0x3c00),
            ];
            let lo = reps[rel];
            let h = reps[(rel + 1) % 4];
            if hi {
                (h.0 << 16 | lo.0, h.1 << 16 | lo.1)
            } else {
                lo
            }
        }
        Kind::U64 => [(1, 2), (5, 5), (2, 1)][rel],
    }
}

fn exec_list(r: &mut Rng) -> [u64; 16] {
    [
        0,
        u64::MAX,
        1,
        1 << 63,
        (1 << 63) | 1,
        0xffff_ffff,
        0xffff_ffff << 32,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x0f0f_0f0f_0f0f_0f0f,
        0xf0f0_f0f0_f0f0_f0f0,
        0x00ff_ff00_ff00_00ff,
        r.next(),
        r.next(),
        r.next(),
        r.next(),
    ]
}

fn gen_cases(o: &OpInfo, idx: usize) -> CaseSet {
    let mut r = Rng(SEED ^ (idx as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    for _ in 0..8 {
        r.next();
    }
    let ex = exec_list(&mut r);
    let mut cs = CaseSet {
        a: Vec::new(),
        b: Vec::new(),
        ctx: Vec::new(),
        label: Vec::new(),
    };
    // Full cartesian product of the 32 edge values (high halves of F16 lanes: a shifted bijection so every
    // SDWA half-select pairing also sees the full product).
    for w in 0..16 {
        let lanes: Vec<(u64, u64)> = (0..64)
            .map(|l| {
                let p = w * 64 + l;
                let (ia, ib) = (p / 32, p % 32);
                match o.kind {
                    Kind::F32 => (u64::from(E32[ia]), u64::from(E32[ib])),
                    Kind::F16 => (
                        u64::from(E16[ia]) | u64::from(E16[(ia + 13) % 32]) << 16,
                        u64::from(E16[ib]) | u64::from(E16[(ib + 7) % 32]) << 16,
                    ),
                    Kind::U64 => (E64[ia], E64[ib]),
                }
            })
            .collect();
        cs.push(&lanes, u64::MAX, "edge_product");
    }
    let mut base: Vec<Vec<(u64, u64)>> = Vec::new();
    for w in 0..48 {
        let lanes: Vec<(u64, u64)> = (0..64).map(|_| rand_lane(o.kind, &mut r)).collect();
        if w < 4 {
            base.push(lanes.clone());
        }
        cs.push(&lanes, ex[w % 16], "random");
    }
    for lanes in &base {
        for &e in &ex {
            cs.push(lanes, e, "exec_sweep");
        }
    }
    // One-hot / one-cold lane walks (skipped when the predicate is constant over the relations).
    let nrel = if o.kind == Kind::U64 { 3 } else { 4 };
    let mode = Mode::default();
    let truth: Vec<bool> = (0..nrel)
        .map(|rel| {
            let (a, b) = rel_rep(o.kind, rel, false);
            lane_truth(o, [false; 2], 0, 0, mode, a, b)
        })
        .collect();
    let rin = truth.iter().position(|&t| t);
    let rout = truth.iter().position(|&t| !t);
    if let (Some(rin), Some(rout)) = (rin, rout) {
        for k in 0..64 {
            for polarity in [false, true] {
                let (hot, cold) = if polarity { (rout, rin) } else { (rin, rout) };
                let lanes: Vec<(u64, u64)> = (0..64)
                    .map(|l| rel_rep(o.kind, if l == k { hot } else { cold }, l % 2 == 0))
                    .collect();
                cs.push(&lanes, u64::MAX, "lane_walk");
            }
        }
    }
    cs
}

// ----------------------------------------------------------------------------
// Modes

fn mode_bits(d32: u32, r32: u32, d16: u32, r16: u32) -> u32 {
    r32 | r16 << 2 | d32 << 4 | d16 << 6 | 1 << 8 | 1 << 9
}
fn sim_mode(bits: u32) -> Mode {
    Mode {
        round32: (bits & 3) as u8,
        round16_64: (bits >> 2 & 3) as u8,
        denorm32: (bits >> 4 & 3) as u8,
        denorm16_64: (bits >> 6 & 3) as u8,
        dx10_clamp: bits >> 8 & 1 != 0,
        ieee: bits >> 9 & 1 != 0,
        fp16_ovfl: bits >> 23 & 1 != 0,
    }
}
fn mode_json(bits: u32) -> Value {
    json!({"bits": h32(bits), "fp_round32": bits & 3, "fp_round16_64": bits >> 2 & 3,
        "fp_denorm32": bits >> 4 & 3, "fp_denorm16_64": bits >> 6 & 3,
        "dx10_clamp": bits >> 8 & 1, "ieee": bits >> 9 & 1, "fp16_ovfl": bits >> 23 & 1})
}
/// The operation's own (denorm 0..3) x (round 0,1,3) = 12 combinations; the other width's fields run
/// through a different combination of the same list.
fn modes_for(kind: Kind) -> Vec<u32> {
    if kind == Kind::U64 {
        return vec![DEFAULT_MODE];
    }
    let mut combos = Vec::new();
    for d in 0..4u32 {
        for r in [0u32, 1, 3] {
            combos.push((d, r));
        }
    }
    (0..combos.len())
        .map(|i| {
            let (d, r) = combos[i];
            let (od, or) = combos[(i + 5) % combos.len()];
            if kind == Kind::F32 {
                mode_bits(d, r, od, or)
            } else {
                mode_bits(od, or, d, r)
            }
        })
        .collect()
}

// ----------------------------------------------------------------------------
// Expected values

fn expectations(
    o: &OpInfo,
    vars: &[Var],
    cs: &CaseSet,
    mode: Mode,
) -> Result<(Vec<u64>, Vec<u64>), String> {
    let n = cs.n();
    let nv = vars.len();
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16);
    type Part = Vec<(usize, Vec<u64>, Vec<u64>)>;
    let parts: Vec<Result<Part, String>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || -> Result<Part, String> {
                    let mut w = Wave::new(Arch::Gfx942, 0, 0xbaad_f00d);
                    if w.v.len() < 254 {
                        return Err(format!("interpreter wave has only {} VGPRs", w.v.len()));
                    }
                    let cfg = Config::default();
                    let mut mem = Memory::default();
                    let mut out = Vec::new();
                    for c in (t..n).step_by(threads) {
                        let [ex, vin, sin] = cs.ctx[c];
                        let (a, b) = (&cs.a[c * 64..c * 64 + 64], &cs.b[c * 64..c * 64 + 64]);
                        for l in 0..64 {
                            if o.kind == Kind::U64 {
                                w.v[250][l] = a[l] as u32;
                                w.v[251][l] = (a[l] >> 32) as u32;
                                w.v[252][l] = b[l] as u32;
                                w.v[253][l] = (b[l] >> 32) as u32;
                            } else {
                                w.v[250][l] = a[l] as u32;
                                w.v[251][l] = b[l] as u32;
                            }
                        }
                        let (mut sim, mut truth) =
                            (Vec::with_capacity(nv * 3), Vec::with_capacity(nv * 3));
                        for var in vars {
                            w.mode = mode;
                            w.exec = ex;
                            w.vcc = vin;
                            w.s[20] = sin as u32;
                            w.s[21] = (sin >> 32) as u32;
                            w.pending.clear();
                            exec_one(&mut w, &var.inst, &mut [], &mut mem, &cfg)
                                .map_err(|e| format!("interpreter {} case {c}: {e:?}", var.text))?;
                            sim.extend([
                                w.vcc,
                                u64::from(w.s[20]) | u64::from(w.s[21]) << 32,
                                w.exec,
                            ]);
                            let mut mask = 0u64;
                            for l in 0..64 {
                                if ex >> l & 1 != 0
                                    && lane_truth(o, var.half, var.neg, var.abs, mode, a[l], b[l])
                                {
                                    mask |= 1 << l;
                                }
                            }
                            let (v, s) = if var.dst == Dst::Vcc {
                                (mask, sin)
                            } else {
                                (vin, mask)
                            };
                            truth.extend([v, s, if o.cmpx { mask } else { ex }]);
                        }
                        out.push((c, sim, truth));
                    }
                    Ok(out)
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| {
                h.join().unwrap_or_else(|p| {
                    let m = p
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_default();
                    Err(format!("interpreter thread panic: {m}"))
                })
            })
            .collect()
    });
    let (mut sim, mut truth) = (vec![0u64; n * nv * 3], vec![0u64; n * nv * 3]);
    for p in parts {
        for (c, s, t) in p? {
            sim[c * nv * 3..(c + 1) * nv * 3].copy_from_slice(&s);
            truth[c * nv * 3..(c + 1) * nv * 3].copy_from_slice(&t);
        }
    }
    Ok((sim, truth))
}

fn bad_records(a: &[u64], b: &[u64]) -> u64 {
    a.chunks_exact(3)
        .zip(b.chunks_exact(3))
        .filter(|(x, y)| x != y)
        .count() as u64
}

// ----------------------------------------------------------------------------
// HIP source

const SRC_HEADER: &str = r#"// Generated by cdna_compare_hw_oracle (peacemaker-isa example). One kernel per compare operation; one wave
// per case; every variant is an asm block on fixed registers v250.., s[20:21], vcc that loads EXEC / VCC /
// s[20:21], runs the real instruction and snapshots VCC, s[20:21] and the post-instruction EXEC.
#include <hip/hip_runtime.h>

__device__ __forceinline__ void cmp_set_mode(unsigned lo10, unsigned ovfl) {
  asm volatile("s_setreg_b32 hwreg(HW_REG_MODE, 0, 10), %0\n\ts_setreg_b32 hwreg(HW_REG_MODE, 23, 1), %1\n\ts_nop 7\n\ts_nop 7"
               :: "s"(lo10), "s"(ovfl) : "memory");
}
__device__ __forceinline__ unsigned cmp_get_mode() {
  unsigned lo, hi;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 0, 10)" : "=s"(lo));
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 23, 1)" : "=s"(hi));
  return lo | (hi << 23);
}
"#;

fn kernel_name(o: &OpInfo) -> String {
    format!("cmp_{}", o.name.trim_start_matches("v_"))
}

fn gen_source(ops: &[OpInfo], vars: &[Vec<Var>]) -> String {
    let mut s = String::from(SRC_HEADER);
    for (o, vs) in ops.iter().zip(vars) {
        let u64k = o.kind == Kind::U64;
        s.push_str(&format!(
            "\nextern \"C\" __global__ void __launch_bounds__(64) {}(\n    const unsigned* __restrict__ A, const unsigned* __restrict__ B, const unsigned long long* __restrict__ CTX,\n    unsigned long long* __restrict__ OUT, unsigned* __restrict__ MRB, unsigned ncases, unsigned mode_lo, unsigned mode_hi) {{\n  const unsigned c = blockIdx.x;\n  const unsigned lane = threadIdx.x;\n  cmp_set_mode(mode_lo, mode_hi);\n  const unsigned pre = cmp_get_mode();\n",
            kernel_name(o)
        ));
        if u64k {
            s.push_str("  const unsigned i2 = 2u * (c * 64u + lane);\n  const unsigned alo = A[i2], ahi = A[i2 + 1u], blo = B[i2], bhi = B[i2 + 1u];\n");
        } else {
            s.push_str("  const unsigned a = A[c * 64u + lane], b = B[c * 64u + lane];\n");
        }
        s.push_str("  const unsigned long long ex = CTX[3u * c], vin = CTX[3u * c + 1u], sin_ = CTX[3u * c + 2u];\n  unsigned long long vo, so, eo, sv;\n");
        for (vi, var) in vs.iter().enumerate() {
            let mut lines: Vec<String> = Vec::new();
            if u64k {
                lines.push("v_mov_b32 v250, %[alo]".into());
                lines.push("v_mov_b32 v251, %[ahi]".into());
                lines.push("v_mov_b32 v252, %[blo]".into());
                lines.push("v_mov_b32 v253, %[bhi]".into());
            } else {
                lines.push("v_mov_b32 v250, %[a]".into());
                lines.push("v_mov_b32 v251, %[b]".into());
            }
            lines.extend(
                [
                    "s_mov_b64 %[sv], exec",
                    "s_mov_b64 s[20:21], %[sin]",
                    "s_mov_b64 vcc, %[vin]",
                    "s_mov_b64 exec, %[ex]",
                    "s_nop 4",
                ]
                .map(String::from),
            );
            lines.push(var.text.clone());
            lines.extend(
                [
                    "s_nop 4",
                    "s_mov_b64 %[eo], exec",
                    "s_mov_b64 exec, %[sv]",
                    "s_mov_b64 %[vo], vcc",
                    "s_mov_b64 %[so], s[20:21]",
                ]
                .map(String::from),
            );
            let ins = if u64k {
                "[alo] \"v\"(alo), [ahi] \"v\"(ahi), [blo] \"v\"(blo), [bhi] \"v\"(bhi)"
            } else {
                "[a] \"v\"(a), [b] \"v\"(b)"
            };
            let clob = if u64k {
                "\"vcc\", \"v250\", \"v251\", \"v252\", \"v253\", \"s20\", \"s21\""
            } else {
                "\"vcc\", \"v250\", \"v251\", \"s20\", \"s21\""
            };
            s.push_str(&format!(
                "  // variant {vi}: {}\n  asm volatile(\"{}\"\n    : [sv] \"=&s\"(sv), [eo] \"=&s\"(eo), [vo] \"=&s\"(vo), [so] \"=&s\"(so)\n    : {ins}, [ex] \"s\"(ex), [vin] \"s\"(vin), [sin] \"s\"(sin_)\n    : {clob});\n  if (lane == 0u) {{ OUT[(c * {}u + {vi}u) * 3u] = vo; OUT[(c * {}u + {vi}u) * 3u + 1u] = so; OUT[(c * {}u + {vi}u) * 3u + 2u] = eo; }}\n",
                var.label,
                lines.join("\\n\\t"),
                vs.len(),
                vs.len(),
                vs.len()
            ));
        }
        s.push_str("  const unsigned post = cmp_get_mode();\n  if (lane == 0u) { MRB[2u * c] = pre; MRB[2u * c + 1u] = post; }\n}\n");
    }
    s
}

// ----------------------------------------------------------------------------
// Hardware

fn hip_blob(explicit: &[u8], grid: u32, block: u32) -> Vec<u8> {
    let mut v = explicit.to_vec();
    while v.len() % 8 != 0 {
        v.push(0);
    }
    for x in [grid, 1, 1] {
        v.extend(x.to_le_bytes());
    }
    for x in [block as u16, 1, 1] {
        v.extend(x.to_le_bytes());
    }
    v.resize(v.len().max(256), 0);
    v
}

struct Launched {
    payload: Vec<u64>,
    guards_ok: bool,
    mrb: Vec<u32>,
}

struct Bufs {
    a: DeviceBuffer,
    b: DeviceBuffer,
    ctx: DeviceBuffer,
    out: DeviceBuffer,
    mrb: DeviceBuffer,
    out_bytes: usize,
    mrb_bytes: usize,
    ncases: usize,
}

fn launch(
    rt: &HipRuntime,
    f: &Function,
    bu: &Bufs,
    bits: u32,
    poison: u8,
) -> Result<Launched, String> {
    let total = 2 * GUARD + bu.out_bytes;
    rt.memset(&bu.out, i32::from(poison), total)
        .map_err(|e| format!("memset(out): {e}"))?;
    rt.memset(&bu.mrb, i32::from(poison), bu.mrb_bytes)
        .map_err(|e| format!("memset(mrb): {e}"))?;
    let p = |d: &DeviceBuffer| d.as_ptr() as usize as u64;
    let mut e = Vec::new();
    for x in [
        p(&bu.a),
        p(&bu.b),
        p(&bu.ctx),
        p(&bu.out) + GUARD as u64,
        p(&bu.mrb),
    ] {
        e.extend(x.to_le_bytes());
    }
    e.extend((bu.ncases as u32).to_le_bytes());
    e.extend((bits & 0x3ff).to_le_bytes());
    e.extend((bits >> 23 & 1).to_le_bytes());
    let mut blob = hip_blob(&e, bu.ncases as u32, 64);
    // SAFETY: the blob is the kernel's explicit kernarg image (5 pointers to live allocations sized for the
    // launch, then three u32) followed by the zeroed hidden arguments; one 64-thread block per case.
    unsafe { rt.launch_kernel_blob(f, [bu.ncases as u32, 1, 1], [64, 1, 1], 0, None, &mut blob) }
        .map_err(|e| format!("launch: {e}"))?;
    rt.device_synchronize()
        .map_err(|e| format!("synchronize: {e}"))?;
    let mut ob = vec![0u8; total];
    rt.memcpy_dtoh(&mut ob, &bu.out)
        .map_err(|e| format!("readback(out): {e}"))?;
    let mut mb = vec![0u8; bu.mrb_bytes];
    rt.memcpy_dtoh(&mut mb, &bu.mrb)
        .map_err(|e| format!("readback(mrb): {e}"))?;
    let guards_ok = ob[..GUARD].iter().all(|&x| x == poison)
        && ob[total - GUARD..].iter().all(|&x| x == poison);
    Ok(Launched {
        payload: bytes_u64(&ob[GUARD..total - GUARD]),
        guards_ok,
        mrb: bytes_u32(&mb),
    })
}

fn free_all(rt: &HipRuntime, bu: Bufs) -> Result<(), String> {
    for d in [bu.a, bu.b, bu.ctx, bu.out, bu.mrb] {
        rt.free(d).map_err(|e| format!("hipFree: {e}"))?;
    }
    Ok(())
}

// ----------------------------------------------------------------------------
// Controls

/// Planted corruption of hardware records; each MUST be seen by BOTH comparisons (hardware vs interpreter
/// and hardware vs truth) as exactly one additional bad record.
fn controls(
    hw: &[u64],
    sim: &[u64],
    truth: &[u64],
    cs: &CaseSet,
    vars: &[Var],
) -> Result<Vec<Value>, String> {
    let nv = vars.len();
    let clean = |r: usize| {
        hw[3 * r..3 * r + 3] == sim[3 * r..3 * r + 3]
            && hw[3 * r..3 * r + 3] == truth[3 * r..3 * r + 3]
    };
    let find = |pred: &dyn Fn(usize, usize) -> bool| -> Option<usize> {
        (0..cs.n() * nv).find(|&r| clean(r) && pred(r / nv, r % nv))
    };
    let dstf = |r: usize| if vars[r % nv].dst == Dst::Vcc { 0 } else { 1 };
    type Plan<'a> = (
        &'static str,
        Option<usize>,
        Box<dyn Fn(usize) -> (usize, u64) + 'a>,
    );
    let plans: Vec<Plan<'_>> = vec![
        (
            "dst_mask_bit63_flip",
            find(&|_, _| true),
            Box::new(|r| (3 * r + dstf(r), 1u64 << 63)),
        ),
        (
            "dst_mask_bit0_flip",
            find(&|c, _| cs.ctx[c][0] != 0),
            Box::new(|r| (3 * r + dstf(r), 1)),
        ),
        (
            "exec_out_bit0_flip",
            find(&|c, _| cs.ctx[c][0] != 0),
            Box::new(|r| (3 * r + 2, 1)),
        ),
        (
            "exec_out_bit63_flip",
            find(&|_, _| true),
            Box::new(|r| (3 * r + 2, 1u64 << 63)),
        ),
        (
            "exec_zero_case_exec_out_flip",
            find(&|c, _| cs.ctx[c][0] == 0),
            Box::new(|r| (3 * r + 2, 1u64 << 63)),
        ),
        (
            "preserved_register_flip",
            find(&|_, _| true),
            Box::new(|r| (3 * r + 1 - dstf(r), 1u64 << 62)),
        ),
    ];
    let mut out = Vec::new();
    for (name, rec, f) in plans {
        let r = rec.ok_or_else(|| format!("control {name}: no clean record to corrupt"))?;
        let (idx, bit) = f(r);
        let mut bad = hw.to_vec();
        bad[idx] ^= bit;
        let (db, dt) = (bad_records(&bad, sim), bad_records(&bad, truth));
        let (b0, t0) = (bad_records(hw, sim), bad_records(hw, truth));
        if db != b0 + 1 || dt != t0 + 1 {
            return Err(format!(
                "control {name} (record {r}) missed: vs interpreter {b0}->{db}, vs truth {t0}->{dt}"
            ));
        }
        out.push(json!({"control": name, "record": r, "word": idx, "xor": h64(bit), "detected_vs_interpreter": true, "detected_vs_truth": true}));
    }
    Ok(out)
}

// ----------------------------------------------------------------------------
// Run

#[derive(Default)]
struct State {
    rows: Vec<Value>,
    ops: Vec<Value>,
    samples: Vec<Value>,
    cases: u64,
    lane_compares: u64,
    records: u64,
    launches: u64,
    bad_hw_sim: u64,
    bad_hw_truth: u64,
    bad_sim_truth: u64,
    controls: u64,
    meta: serde_json::Map<String, Value>,
}

fn run_cmd_version(path: &str, arg: &str) -> Value {
    match Command::new(path).arg(arg).output() {
        Ok(o) => Value::String(
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .take(4)
                .collect::<Vec<_>>()
                .join(" | "),
        ),
        Err(e) => Value::String(format!("unavailable: {e}")),
    }
}

fn run(dir: &Path, st: &mut State) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let all = ops()?;
    let mut vars = Vec::new();
    for o in &all {
        let vs = variants(o)?;
        if o.kind != Kind::U64 && !vs.iter().any(|v| v.label.ends_with("_clamp")) {
            return Err(format!(
                "{}: no VOP3 clamp=1 variant (codec must advertise clamp for every float compare)",
                o.name
            ));
        }
        vars.push(vs);
    }
    let src = gen_source(&all, &vars);
    let src_path = dir.join("cmp_family.hip");
    let co_path = dir.join("cmp_family.hsaco");
    std::fs::write(&src_path, &src).map_err(|e| e.to_string())?;
    let cc = Command::new("/opt/rocm/bin/hipcc")
        .args([
            "--genco",
            "--offload-arch=gfx942",
            "-O3",
            "--no-offload-compress",
            "-fuse-cuid=none",
            "-o",
            co_path.to_str().ok_or("non-UTF8 output path")?,
            src_path.to_str().ok_or("non-UTF8 source path")?,
        ])
        .output()
        .map_err(|e| e.to_string())?;
    if !cc.status.success() {
        return Err(format!("hipcc: {}", String::from_utf8_lossy(&cc.stderr)));
    }
    let code = std::fs::read(&co_path).map_err(|e| e.to_string())?;
    // Every variant's codec encoding must appear exactly once in the generated code object: the hardware
    // executes the requested opcode and encoding, not compiler-substituted arithmetic.
    let dw: Vec<u32> = bytes_u32(&code[..code.len() / 4 * 4]);
    let mut enc_hash = Sha256::new();
    let mut n_variants = 0usize;
    for (o, vs) in all.iter().zip(&vars) {
        for v in vs {
            let hits = dw
                .windows(v.words.len())
                .filter(|w| *w == v.words.as_slice())
                .count();
            if hits != 1 {
                return Err(format!(
                    "{} / {}: encoding of `{}` ({:08x?}) occurs {hits} times in the code object (expected exactly 1)",
                    o.name, v.label, v.text, v.words
                ));
            }
            enc_hash.update(v.text.as_bytes());
            for w in &v.words {
                enc_hash.update(w.to_le_bytes());
            }
            n_variants += 1;
        }
    }
    st.meta.insert("variants_total".into(), json!(n_variants));
    st.meta.insert(
        "variant_encodings_sha256".into(),
        json!(format!("{:x}", enc_hash.finalize())),
    );
    st.meta
        .insert("hip_source_sha256".into(), json!(sha(src.as_bytes())));
    st.meta
        .insert("code_object_sha256".into(), json!(sha(&code)));
    st.meta.insert(
        "hipcc_version".into(),
        run_cmd_version("/opt/rocm/bin/hipcc", "--version"),
    );

    let rt = HipRuntime::load().map_err(|e| e.to_string())?;
    rt.set_device(0).map_err(|e| e.to_string())?;
    let arch = rt.get_arch(0).map_err(|e| e.to_string())?;
    if arch != "gfx942" {
        return Err(format!("requires gfx942, got {arch}"));
    }
    st.meta.insert("arch".into(), json!(arch));
    let module: Module = rt.module_load_data(&code).map_err(|e| e.to_string())?;

    let mut inputs_hash = Sha256::new();
    let mut hw_hash = Sha256::new();
    for (idx, (o, vs)) in all.iter().zip(&vars).enumerate() {
        let f = rt
            .module_get_function(&module, &kernel_name(o))
            .map_err(|e| format!("hipModuleGetFunction({}): {e}", kernel_name(o)))?;
        let cs = gen_cases(o, idx);
        let n = cs.n();
        let nv = vs.len();
        let stride = if o.kind == Kind::U64 { 2 } else { 1 };
        let (mut ab, mut bb) = (Vec::new(), Vec::new());
        for i in 0..n * 64 {
            if stride == 2 {
                ab.extend(cs.a[i].to_le_bytes());
                bb.extend(cs.b[i].to_le_bytes());
            } else {
                ab.extend((cs.a[i] as u32).to_le_bytes());
                bb.extend((cs.b[i] as u32).to_le_bytes());
            }
        }
        let cb = words_bytes(&cs.ctx.concat());
        for x in [&ab, &bb, &cb] {
            inputs_hash.update(x.as_slice());
        }
        let out_bytes = n * nv * 24;
        let mrb_bytes = 2 * n * 4;
        let mal =
            |sz: usize, what: &str| rt.malloc(sz).map_err(|e| format!("hipMalloc({what}): {e}"));
        let bu = Bufs {
            a: mal(ab.len(), "A")?,
            b: mal(bb.len(), "B")?,
            ctx: mal(cb.len(), "CTX")?,
            out: mal(2 * GUARD + out_bytes, "OUT")?,
            mrb: mal(mrb_bytes, "MRB")?,
            out_bytes,
            mrb_bytes,
            ncases: n,
        };
        rt.memcpy_htod(&bu.a, &ab)
            .map_err(|e| format!("upload A: {e}"))?;
        rt.memcpy_htod(&bu.b, &bb)
            .map_err(|e| format!("upload B: {e}"))?;
        rt.memcpy_htod(&bu.ctx, &cb)
            .map_err(|e| format!("upload CTX: {e}"))?;
        let modes = modes_for(o.kind);
        let (mut op_bad_sim, mut op_bad_truth, mut op_bad_st) = (0u64, 0u64, 0u64);
        let mut op_controls = Vec::new();
        for (mi, &bits) in modes.iter().enumerate() {
            let mut hw: Option<Vec<u64>> = None;
            let (mut guard_bad, mut poison_dep, mut rb_bad) = (0u64, 0u64, 0u64);
            for &poison in &POISONS {
                let l = launch(&rt, &f, &bu, bits, poison)
                    .map_err(|e| format!("{} mode {}: {e}", o.name, h32(bits)))?;
                st.launches += 1;
                guard_bad += u64::from(!l.guards_ok);
                rb_bad += l.mrb.iter().filter(|&&x| x != bits).count() as u64;
                if let Some(p0) = &hw {
                    poison_dep += p0.iter().zip(&l.payload).filter(|(a, b)| a != b).count() as u64;
                } else {
                    hw = Some(l.payload);
                }
            }
            if guard_bad > 0 {
                return Err(format!(
                    "{} mode {}: {guard_bad} output guard canaries corrupted",
                    o.name,
                    h32(bits)
                ));
            }
            if poison_dep > 0 {
                return Err(format!(
                    "{} mode {}: {poison_dep} output words depend on the poison pattern (unwritten/nondeterministic)",
                    o.name,
                    h32(bits)
                ));
            }
            if rb_bad > 0 {
                return Err(format!(
                    "{} mode {}: {rb_bad} MODE readbacks differ from the requested bits",
                    o.name,
                    h32(bits)
                ));
            }
            let hw = hw.expect("two launches");
            let mode = sim_mode(bits);
            let (sim, truth) = expectations(o, vs, &cs, mode)?;
            let (b1, b2, b3) = (
                bad_records(&hw, &sim),
                bad_records(&hw, &truth),
                bad_records(&sim, &truth),
            );
            op_bad_sim += b1;
            op_bad_truth += b2;
            op_bad_st += b3;
            hw_hash.update(words_bytes(&hw));
            if b1 + b2 + b3 > 0 {
                let mut per_row = 0;
                for r in 0..n * nv {
                    let (h, s, t) = (
                        &hw[3 * r..3 * r + 3],
                        &sim[3 * r..3 * r + 3],
                        &truth[3 * r..3 * r + 3],
                    );
                    if (h != s || h != t || s != t) && per_row < 4 && st.samples.len() < MAX_SAMPLES
                    {
                        per_row += 1;
                        let (c, v) = (r / nv, r % nv);
                        let var = &vs[v];
                        let fld = if var.dst == Dst::Vcc { 0 } else { 1 };
                        let diff = (h[fld] ^ t[fld]) | (h[fld] ^ s[fld]);
                        let lane = (diff != 0).then(|| diff.trailing_zeros() as usize);
                        let lane_data = lane.map(|l| json!({"lane": l, "a": h64(cs.a[c * 64 + l]), "b": h64(cs.b[c * 64 + l])}));
                        st.samples.push(json!({
                            "op": o.name, "mode": h32(bits), "variant": var.label, "instruction": var.text,
                            "case": c, "case_kind": cs.label[c],
                            "exec_in": h64(cs.ctx[c][0]), "vcc_in": h64(cs.ctx[c][1]), "s20_in": h64(cs.ctx[c][2]),
                            "hardware": h.iter().map(|&x| h64(x)).collect::<Vec<_>>(),
                            "interpreter": s.iter().map(|&x| h64(x)).collect::<Vec<_>>(),
                            "truth": t.iter().map(|&x| h64(x)).collect::<Vec<_>>(),
                            "first_differing_lane": lane_data,
                            "record_layout": ["vcc", "s20_s21", "exec_after"],
                        }));
                    }
                }
            }
            if mi == 0 {
                op_controls = controls(&hw, &sim, &truth, &cs, vs)?;
                st.controls += op_controls.len() as u64;
            }
            st.cases += n as u64;
            st.records += (n * nv) as u64;
            st.lane_compares += (n * nv * 64) as u64;
            st.rows.push(json!({
                "op": o.name, "mode": mode_json(bits), "cases": n, "variants": nv, "records": n * nv,
                "guards_ok": true, "poison_dependent_words": 0, "mode_readbacks_ok": true,
                "bad_records_hardware_vs_interpreter": b1, "bad_records_hardware_vs_truth": b2,
                "bad_records_interpreter_vs_truth": b3,
                "hardware_sha256": sha(&words_bytes(&hw)), "interpreter_sha256": sha(&words_bytes(&sim)),
                "truth_sha256": sha(&words_bytes(&truth)),
            }));
        }
        st.bad_hw_sim += op_bad_sim;
        st.bad_hw_truth += op_bad_truth;
        st.bad_sim_truth += op_bad_st;
        let case_kinds = {
            let mut m = std::collections::BTreeMap::<&str, usize>::new();
            for k in &cs.label {
                *m.entry(*k).or_default() += 1;
            }
            json!(m)
        };
        st.ops.push(json!({
            "op": o.name, "opcode": o.op.opcode(Arch::Gfx942), "kind": format!("{:?}", o.kind), "cmpx": o.cmpx,
            "predicate_nibble": o.pred, "predicate_token": o.tok, "kernel": kernel_name(o),
            "cases": n, "case_kinds": case_kinds, "modes": modes.len(),
            "variants": vs.iter().map(|v| json!({"label": v.label, "text": v.text,
                "words": v.words.iter().map(|&w| h32(w)).collect::<Vec<_>>()})).collect::<Vec<_>>(),
            "controls": op_controls,
            "vop3_clamp1_variants": vs.iter().filter(|v| v.label.ends_with("_clamp")).count(),
            "vop3_clamp0_variants": vs.iter().filter(|v| v.label.starts_with("e64_") && !v.label.ends_with("_clamp")).count(),
            "bad_records_hardware_vs_interpreter": op_bad_sim, "bad_records_hardware_vs_truth": op_bad_truth,
            "bad_records_interpreter_vs_truth": op_bad_st,
        }));
        eprintln!(
            "{:>3}/{} {}: {} cases x {} variants x {} modes; bad vs interpreter {op_bad_sim}, vs truth {op_bad_truth}, interpreter vs truth {op_bad_st}",
            idx + 1,
            all.len(),
            o.name,
            n,
            nv,
            modes.len()
        );
        free_all(&rt, bu)?;
    }
    st.meta.insert(
        "inputs_sha256".into(),
        json!(format!("{:x}", inputs_hash.finalize())),
    );
    st.meta.insert(
        "hardware_outputs_sha256".into(),
        json!(format!("{:x}", hw_hash.finalize())),
    );
    Ok(())
}

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let Some(dir) = args.next().map(PathBuf::from) else {
        eprintln!("usage: cdna_compare_hw_oracle OUT_DIR");
        return ExitCode::FAILURE;
    };
    if args.next().is_some() {
        eprintln!("unexpected argument");
        return ExitCode::FAILURE;
    }
    let mut st = State::default();
    let result = run(&dir, &mut st);
    let mismatches = st.bad_hw_sim + st.bad_hw_truth + st.bad_sim_truth;
    let pass = result.is_ok() && mismatches == 0;
    let mut v = json!({
        "status": if pass { "PASS" } else { "FAIL" },
        "oracle": "cdna_compare_hw_oracle",
        "error": result.as_ref().err(),
        "complete": result.is_ok(),
        "operations": st.ops.len(), "rows": st.rows.len(), "cases": st.cases, "records": st.records,
        "lane_compares": st.lane_compares, "launches": st.launches, "controls_detected": st.controls,
        "bad_records_hardware_vs_interpreter": st.bad_hw_sim,
        "bad_records_hardware_vs_truth": st.bad_hw_truth,
        "bad_records_interpreter_vs_truth": st.bad_sim_truth,
        "poison_bytes": POISONS.iter().map(|p| format!("{p:#04x}")).collect::<Vec<_>>(),
        "record_layout": "per (case, variant): u64 VCC, u64 s[20:21], u64 EXEC after the instruction (snapshotted before EXEC is restored)",
        "seed": SEED,
        "oracle_source_sha256": sha(include_bytes!("cdna_compare_hw_oracle.rs")),
        "interpreter_valu_source_sha256": sha(include_bytes!("../../peacemaker-cdna/src/sim/valu.rs")),
        "interpreter_mod_source_sha256": sha(include_bytes!("../../peacemaker-cdna/src/sim/mod.rs")),
        "codec_vop_source_sha256": sha(include_bytes!("../../peacemaker-cdna/src/enc/vop.rs")),
        "codec_vop3_source_sha256": sha(include_bytes!("../../peacemaker-cdna/src/enc/vop3.rs")),
        "codec_inst_source_sha256": sha(include_bytes!("../../peacemaker-cdna/src/inst.rs")),
        "mismatch_samples": st.samples.clone(),
        "operation_table": st.ops.clone(),
        "mode_rows": st.rows.clone(),
    });
    for (k, x) in &st.meta {
        v[k] = x.clone();
    }
    let path = dir.join("receipt.json");
    let saved = serde_json::to_vec_pretty(&v)
        .map_err(|e| e.to_string())
        .and_then(|b| std::fs::write(&path, b).map_err(|e| e.to_string()));
    if pass && saved.is_ok() {
        println!(
            "PASS cdna_compare_hw_oracle: {} operations, {} rows, {} cases, {} records, {} lane compares, {} controls detected -> {}",
            st.ops.len(),
            st.rows.len(),
            st.cases,
            st.records,
            st.lane_compares,
            st.controls,
            path.display()
        );
        ExitCode::SUCCESS
    } else {
        println!(
            "FAIL cdna_compare_hw_oracle: error={:?} bad(hw/interp)={} bad(hw/truth)={} bad(interp/truth)={}; receipt {} ({:?})",
            result.err(),
            st.bad_hw_sim,
            st.bad_hw_truth,
            st.bad_sim_truth,
            path.display(),
            saved
        );
        ExitCode::FAILURE
    }
}
