// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact libm lowerings for the checked ISA builder: `sinf`, `cosf`, `sincosf` and `powf`, bit-identical to the
//! hipcc / ROCm 7.16 OCML results (`__ocml_sin_f32`, `__ocml_cos_f32`, `__ocml_pow_f32`) for every input,
//! on gfx1100, gfx1151 and gfx1201.
//!
//! Provenance. The arithmetic DAGs are the ones hipcc emits for
//! `y[i] = sinf(x[i])` / `cosf` / `powf(x[i], e[i])` with the product flags of `rdna-compute`'s
//! `core_hipcc_flags` (`-O3 --genco -fuse-cuid=none -mcode-object-version=6`), transcribed op for op from the
//! gfx1201 disassembly (`/home/kaden/qcal/release-0.4.5/pmrevive/lower/probe/lm-hip-amdgcn-amd-amdhsa-gfx1201.s`):
//! the VOPD pairs are split into two plain VALU instructions (a pair reads before it writes: where the second
//! half reads the first half's destination the second is emitted first), every `vcc` is an explicit SGPR lane
//! mask, and the arch spelling differs only in `v_mad_co_u64_u32` (gfx12) / `v_mad_u64_u32` (gfx11). The three
//! architectures' hipcc objects have the same DAG (they differ only in scheduling, register assignment and VOPD
//! packing); the exhaustive (sinf, cosf: all 2^32 inputs) and sampled (powf) GPU comparisons against hipcc are in
//! `examples/libm_exact.rs`.
//!
//! * `sinf`/`cosf`: `|x|` (+inf replaced by a quiet NaN) is reduced by Cody-Waite for `|x| < 2^17`
//!   (`0x48000000`) and by Payne-Hanek (`__ocml` `trigred_large`, 192-bit 2/pi table in `v_mad_u64_u32`
//!   immediates) above; both reductions yield `(r, q & 3)` and the shared kernel polynomials give `sin r` and
//!   `cos r`; the quadrant, parity and the sign of `x` select and sign the result. hipcc's `sinf(a)` and
//!   `cosf(a)` of the same angle reduce twice; [`sincosf`] reduces once (the reduction is a pure function of
//!   `|x|`, so each result is bit-identical to the standalone function).
//! * `powf(x, y)`: the OCML double-single `log2` / `exp2` pipeline with its special-case ladder, as emitted.
//!
//! Register discipline. The caller supplies the input register(s), the output register(s), a block of VGPR
//! temporaries and a block of SGPR temporaries (even-aligned; lane masks are 32-bit in wave32 and sit on pair
//! boundaries), exactly like the per-kernel `Exp` structs: nothing here assumes a kernel. Inputs are never
//! written (outputs may not alias them). No `vcc`, no `scc`-dependence is left behind; `exec` is saved and
//! restored by the sin/cos large-argument split. Hazard guards (SGPR/exec write -> read waits) are the
//! builder's.
//!
//! Large-argument handling ([`LargeArg`]): `Branch` skips the Payne-Hanek block with `s_cbranch_execz` when no
//! lane needs it (the hipcc shape; needs an untyped builder); `Masked` always issues both reduction blocks
//! under complementary exec masks (no branch: usable in typed kernels and where labels are unwelcome). The
//! pieces ([`trig_prelude`], [`trig_large`], [`trig_small`], [`trig_finish`]) are public for callers that own
//! the control flow (typed `Workgroup` scopes).
use crate::kernels::common::op;
use crate::reg::{Kind, RegRef};
use crate::{Arch, Builder};

type R = Result<(), String>;

// ------------------------------------------------------------------ register environment

/// Where the template names live: `X`/`Y` inputs, `O`/`SO`/`CO` outputs, `T<n>` VGPR temporaries from `v`,
/// `M<n>` lane-mask SGPRs (pair `n` of the SGPR block at `s`), `K<n>` scalar SGPRs after the `nm` mask pairs.
#[derive(Clone, Copy)]
struct Env {
    arch: Arch,
    x: u8,
    y: u8,
    o: u8,
    so: u8,
    co: u8,
    v: u8,
    s: u8,
    nm: u8,
}

fn vreg(n: u16) -> Result<String, String> {
    if n > 255 {
        return Err(format!("libm: VGPR v{n} out of range"));
    }
    Ok(format!("v{n}"))
}

/// Substitute the template names of `t` by the physical registers of `e`.
fn subst(t: &str, e: &Env) -> Result<String, String> {
    let b = t.as_bytes();
    let mut out = String::with_capacity(t.len() + 8);
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let boundary = i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
        if !(c.is_ascii_uppercase() && boundary) {
            out.push(c as char);
            i += 1;
            continue;
        }
        let start = i;
        while i < b.len() && b[i].is_ascii_uppercase() {
            i += 1;
        }
        let name = &t[start..i];
        if i < b.len() && b[i] == b'[' && name == "T" {
            let close = t[i..].find(']').ok_or("libm: unterminated T[a:b]")? + i;
            let (a, z) = t[i + 1..close].split_once(':').ok_or("libm: bad T[a:b]")?;
            let (a, z): (u16, u16) = (
                a.parse().map_err(|_| "libm: bad T[a:b]")?,
                z.parse().map_err(|_| "libm: bad T[a:b]")?,
            );
            out.push_str(&format!(
                "v[{}:{}]",
                &vreg(u16::from(e.v) + a)?[1..],
                &vreg(u16::from(e.v) + z)?[1..]
            ));
            i = close + 1;
            continue;
        }
        let ds = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        let n: u16 = t[ds..i].parse().unwrap_or(0);
        match name {
            "X" => out.push_str(&vreg(e.x.into())?),
            "Y" => out.push_str(&vreg(e.y.into())?),
            "O" => out.push_str(&vreg(e.o.into())?),
            "SO" => out.push_str(&vreg(e.so.into())?),
            "CO" => out.push_str(&vreg(e.co.into())?),
            "T" => out.push_str(&vreg(u16::from(e.v) + n)?),
            "M" => out.push_str(&format!("s{}", u16::from(e.s) + 2 * n)),
            "K" => out.push_str(&format!("s{}", u16::from(e.s) + 2 * u16::from(e.nm) + n)),
            _ => return Err(format!("libm: unknown template name {name}")),
        }
    }
    // gfx12 spells the 64-bit multiply-add `_co_`.
    if !e.arch.gfx12() {
        out = out.replace("v_mad_co_u64_u32", "v_mad_u64_u32");
    }
    Ok(out)
}

// ------------------------------------------------------------------ defs / uses

fn operands(text: &str) -> Vec<&str> {
    let Some((_, rest)) = text.split_once(' ') else {
        return vec![];
    };
    let mut out = Vec::new();
    let (mut depth, mut from) = (0i32, 0usize);
    for (i, c) in rest.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(rest[from..i].trim());
                from = i + 1;
            }
            _ => {}
        }
    }
    out.push(rest[from..].trim());
    out
}

/// The registers of a VGPR / SGPR (or `[a:b]` range) operand with its source modifiers stripped, one
/// single-register `RegRef` each (a register plan owns registers by 8-aligned block, so a range that straddles
/// two declared blocks is tracked as its members); empty for everything else (literals, `null`, `exec_lo`, ...).
fn reg_operand(o: &str) -> Vec<RegRef> {
    let mut o = o.trim();
    o = o.strip_prefix('-').unwrap_or(o);
    if let Some(inner) = o.strip_prefix('|').and_then(|r| r.strip_suffix('|')) {
        o = inner;
    }
    o = o
        .strip_suffix("")
        .or_else(|| o.strip_suffix(".h"))
        .unwrap_or(o);
    let kind = match o.as_bytes().first() {
        Some(b'v') => Kind::V,
        Some(b's') => Kind::S,
        _ => return vec![],
    };
    let rest = &o[1..];
    let (first, last) =
        if let Some(range) = rest.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
            let Some((a, z)) = range.split_once(':') else {
                return vec![];
            };
            let (Ok(a), Ok(z)) = (a.parse::<u8>(), z.parse::<u8>()) else {
                return vec![];
            };
            (a, z)
        } else {
            let Ok(n) = rest.parse::<u8>() else {
                return vec![];
            };
            (n, n)
        };
    (first..=last)
        .map(|base| RegRef { kind, base, len: 1 })
        .collect()
}

/// `(defs, uses)` of one emitted template instruction: the first operand is the destination (a `v_cmp`
/// writes its SGPR lane mask, a `v_cmpx` and an exec write define no tracked register), `v_fmac` also reads
/// it, every other register operand is a use. Registers outside the V/S files (`exec_lo`, `null`) are not
/// tracked by the builder.
fn derive(text: &str) -> (Vec<RegRef>, Vec<RegRef>) {
    let m = text.split_whitespace().next().unwrap_or("");
    let ops = operands(text);
    let (mut defs, mut uses) = (Vec::new(), Vec::new());
    let has_dst = !m.starts_with("v_cmpx_");
    for (i, o) in ops.iter().enumerate() {
        let regs = reg_operand(o);
        if i == 0 && has_dst {
            if m.starts_with("v_fmac_") {
                uses.extend(&regs);
            }
            defs.extend(regs);
        } else {
            uses.extend(regs);
        }
    }
    (defs, uses)
}

fn run(b: &mut Builder, seq: &[&str], e: &Env) -> R {
    for t in seq {
        let text = subst(t, e)?;
        let (defs, uses) = derive(&text);
        op(b, text, &defs, &uses)?;
    }
    Ok(())
}

fn one(b: &mut Builder, t: &str, e: &Env) -> R {
    run(b, &[t], e)
}

// ------------------------------------------------------------------ sinf / cosf / sincosf

/// VGPR temporaries of [`sincosf`] / [`sinf`] / [`cosf`] (a contiguous block starting at `Trig::v`).
pub const TRIG_VGPRS: u8 = 13;
/// SGPR temporaries of the trig family (a contiguous, even-aligned block starting at `Trig::s`): four lane
/// masks (`s..s+8`), then the saved exec, then the two polynomial coefficients.
pub const TRIG_SGPRS: u8 = 11;
const TRIG_MASKS: u8 = 4;

/// One `sinf` / `cosf` / `sincosf` instance. `x` is read, never written; `sin` / `cos` are the requested
/// outputs (at least one; they may alias `x` but not each other or a temporary).
#[derive(Clone, Copy, Debug)]
pub struct Trig {
    pub x: u8,
    pub sin: Option<u8>,
    pub cos: Option<u8>,
    /// First of [`TRIG_VGPRS`] contiguous scratch VGPRs.
    pub v: u8,
    /// First of [`TRIG_SGPRS`] contiguous scratch SGPRs (even).
    pub s: u8,
}

/// How the Payne-Hanek (`|x| >= 2^17`) reduction block is entered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LargeArg {
    /// `s_cbranch_execz` over the block when no lane needs it (hipcc's shape). Requires an untyped builder;
    /// `id` makes the label unique within the kernel.
    Branch { id: u32 },
    /// No branch: the block always issues, under an exec mask selecting the large lanes.
    Masked,
}

impl Trig {
    fn check(&self) -> R {
        if self.sin.is_none() && self.cos.is_none() {
            return Err("libm: Trig wants neither sin nor cos".into());
        }
        if self.s % 2 != 0 {
            return Err("libm: Trig SGPR block must be even-aligned".into());
        }
        if self.sin.is_some() && self.sin == self.cos {
            return Err("libm: sin and cos outputs alias".into());
        }
        let temps = self.v as u16..self.v as u16 + TRIG_VGPRS as u16;
        for r in [Some(self.x), self.sin, self.cos].into_iter().flatten() {
            if temps.contains(&(r as u16)) {
                return Err(format!("libm: v{r} lies inside the Trig temporaries"));
            }
        }
        Ok(())
    }
    fn env(&self, arch: Arch) -> Env {
        Env {
            arch,
            x: self.x,
            y: 0,
            o: 0,
            so: self.sin.unwrap_or(0),
            co: self.cos.unwrap_or(0),
            v: self.v,
            s: self.s,
            nm: TRIG_MASKS,
        }
    }
    /// The lane mask (SGPR) of the lanes that take the Payne-Hanek block after [`trig_prelude`].
    pub fn large_mask(&self) -> u8 {
        self.s
    }
    /// The SGPR [`trig_exec_split`] saves exec to.
    pub fn saved_exec(&self) -> u8 {
        self.s + 2 * TRIG_MASKS
    }
}

const TRIG_PRE: &[&str] = &[
    "v_cmp_neq_f32_e64 M1, 0x7f800000, |X|",
    "v_cndmask_b32_e64 T1, 0x7fc00000, |X|, M1",
];
const TRIG_PRE_SIGN: &[&str] = &[
    "v_and_b32_e32 T12, 0x80000000, X",
    "v_cndmask_b32_e64 T12, 0, T12, M1",
];
const TRIG_PRE_MASK: &[&str] = &["v_cmp_le_f32_e64 M0, 0x48000000, T1"];
const TRIG_TAIL: &[&str] = &[
    "v_mul_f32_e32 T1, T0, T0",
    "s_mov_b32 K1, 0xb94c1982",
    "s_mov_b32 K2, 0x37d75334",
    "v_cmp_lt_u32_e64 M1, 1, T2",
    "v_and_b32_e32 T3, 1, T2",
    "v_fmaak_f32 T4, K1, T1, 0x3c0881c4",
    "v_cndmask_b32_e64 T2, 0, 0x80000000, M1",
    "v_cmp_eq_u32_e64 M1, 0, T3",
    "v_fmaak_f32 T4, T1, T4, 0xbe2aaa9d",
    "v_mul_f32_e32 T4, T1, T4",
    "v_fmac_f32_e32 T0, T0, T4",
    "v_fmaak_f32 T5, K2, T1, 0xbab64f3b",
    "v_fmaak_f32 T5, T1, T5, 0x3d2aabf7",
    "v_fmaak_f32 T5, T1, T5, 0xbf000004",
    "v_fma_f32 T1, T1, T5, 1.0",
];
const TRIG_TAIL_COS: &[&str] = &[
    "v_cndmask_b32_e64 T3, -T0, T1, M1",
    "v_xor_b32_e32 CO, T2, T3",
];
const TRIG_TAIL_SIN: &[&str] = &[
    "v_cndmask_b32_e64 T4, T1, T0, M1",
    "v_xor3_b32 SO, T12, T2, T4",
];

/// `|x|` with `+-inf` replaced by a quiet NaN (`T1`), the sign of `x` (`T12`, sin only) and the large-argument
/// lane mask ([`Trig::large_mask`]). Reads `x`.
pub fn trig_prelude(b: &mut Builder, t: &Trig) -> R {
    t.check()?;
    let e = t.env(b.spec.arch);
    run(b, TRIG_PRE, &e)?;
    if t.sin.is_some() {
        run(b, TRIG_PRE_SIGN, &e)?;
    }
    run(b, TRIG_PRE_MASK, &e)
}

/// `saved_exec = exec; exec &= large_mask` (clobbers SCC).
pub fn trig_exec_split(b: &mut Builder, t: &Trig) -> R {
    let e = t.env(b.spec.arch);
    run(
        b,
        &["s_mov_b32 K0, exec_lo", "s_and_b32 exec_lo, exec_lo, M0"],
        &e,
    )
}

/// The Payne-Hanek block: run under `exec = lanes with |x| >= 2^17`; leaves `(r, q & 3)` in `T0`, `T2`.
pub fn trig_large(b: &mut Builder, t: &Trig) -> R {
    run(b, LARGE, &t.env(b.spec.arch))
}

/// `exec = saved_exec & ~large_mask` (the small lanes).
pub fn trig_small_exec(b: &mut Builder, t: &Trig) -> R {
    run(b, &["s_and_not1_b32 exec_lo, K0, M0"], &t.env(b.spec.arch))
}

/// The Cody-Waite block: run under the small lanes' exec; leaves `(r, q & 3)` in `T0`, `T2`.
pub fn trig_small(b: &mut Builder, t: &Trig) -> R {
    run(b, SMALL, &t.env(b.spec.arch))
}

/// `exec = saved_exec`.
pub fn trig_exec_restore(b: &mut Builder, t: &Trig) -> R {
    run(b, &["s_mov_b32 exec_lo, K0"], &t.env(b.spec.arch))
}

/// The polynomial, quadrant and sign selection (full exec): writes the requested outputs.
pub fn trig_finish(b: &mut Builder, t: &Trig) -> R {
    let e = t.env(b.spec.arch);
    run(b, TRIG_TAIL, &e)?;
    if t.cos.is_some() {
        run(b, TRIG_TAIL_COS, &e)?;
    }
    if t.sin.is_some() {
        run(b, TRIG_TAIL_SIN, &e)?;
    }
    Ok(())
}

fn trig_label(id: u32) -> String {
    format!(".Llibm_trig_{id}")
}

/// `sinf` and/or `cosf` of `t.x` (bit-identical to hipcc's `sinf` / `cosf`; both outputs share one reduction).
pub fn sincosf(b: &mut Builder, t: &Trig, large: LargeArg) -> R {
    trig_prelude(b, t)?;
    trig_exec_split(b, t)?;
    match large {
        LargeArg::Branch { id } => {
            let label = trig_label(id);
            op(b, format!("s_cbranch_execz {label}"), &[], &[])?;
            trig_large(b, t)?;
            b.label(&label)?;
        }
        LargeArg::Masked => trig_large(b, t)?,
    }
    trig_small_exec(b, t)?;
    trig_small(b, t)?;
    trig_exec_restore(b, t)?;
    trig_finish(b, t)
}

/// `dst = sinf(x)`.
pub fn sinf(b: &mut Builder, dst: u8, x: u8, v: u8, s: u8, large: LargeArg) -> R {
    sincosf(
        b,
        &Trig {
            x,
            sin: Some(dst),
            cos: None,
            v,
            s,
        },
        large,
    )
}

/// `dst = cosf(x)`.
pub fn cosf(b: &mut Builder, dst: u8, x: u8, v: u8, s: u8, large: LargeArg) -> R {
    sincosf(
        b,
        &Trig {
            x,
            sin: None,
            cos: Some(dst),
            v,
            s,
        },
        large,
    )
}

// ------------------------------------------------------------------ powf

/// VGPR temporaries of [`powf`] (a contiguous block starting at `Pow::v`).
pub const POW_VGPRS: u8 = 13;
/// SGPR temporaries of [`powf`] (contiguous, even-aligned from `Pow::s`): eight lane masks and one constant.
pub const POW_SGPRS: u8 = 17;
const POW_MASKS: u8 = 8;

/// One `powf(x, y)` instance: `dst = powf(x, y)`. `x` and `y` are read, never written; `dst` may alias either
/// (it is written last) but not a temporary.
#[derive(Clone, Copy, Debug)]
pub struct Pow {
    pub dst: u8,
    pub x: u8,
    pub y: u8,
    /// First of [`POW_VGPRS`] contiguous scratch VGPRs.
    pub v: u8,
    /// First of [`POW_SGPRS`] contiguous scratch SGPRs (even).
    pub s: u8,
}

/// `dst = powf(x, y)`, bit-identical to hipcc's `powf` (straight-line, no `exec` change, no branch).
pub fn powf(b: &mut Builder, p: &Pow) -> R {
    if p.s % 2 != 0 {
        return Err("libm: Pow SGPR block must be even-aligned".into());
    }
    let temps = p.v as u16..p.v as u16 + POW_VGPRS as u16;
    for r in [p.dst, p.x, p.y] {
        if temps.contains(&(r as u16)) {
            return Err(format!("libm: v{r} lies inside the Pow temporaries"));
        }
    }
    let e = Env {
        arch: b.spec.arch,
        x: p.x,
        y: p.y,
        o: p.dst,
        so: 0,
        co: 0,
        v: p.v,
        s: p.s,
        nm: POW_MASKS,
    };
    one(b, "s_mov_b32 K0, 0x3e76c4e1", &e)?;
    run(b, POWF, &e)
}

/// The instruction templates of every sequence (tests and tooling).
pub fn templates() -> Vec<(&'static str, &'static [&'static str])> {
    vec![
        ("trig_pre", TRIG_PRE),
        ("trig_pre_sign", TRIG_PRE_SIGN),
        ("trig_pre_mask", TRIG_PRE_MASK),
        ("trig_large", LARGE),
        ("trig_small", SMALL),
        ("trig_tail", TRIG_TAIL),
        ("trig_tail_cos", TRIG_TAIL_COS),
        ("trig_tail_sin", TRIG_TAIL_SIN),
        ("powf", POWF),
    ]
}

// ------------------------------------------------------------------ transcribed DAGs

/// Payne-Hanek reduction of `AX = T1` (`|x| >= 2^17`): `T0` = reduced argument, `T2` = quadrant & 3.
const LARGE: &[&str] = &[
    "v_frexp_mant_f32_e32 T0, T1",
    "v_frexp_exp_i32_f32_e32 T1, T1",
    "v_mov_b32_e32 T4, 0",
    "v_add_nc_u32_e32 T1, 6, T1",
    "v_mul_f32_e32 T0, 0x4b800000, T0",
    "v_mov_b32_e32 T7, T4",
    "v_mov_b32_e32 T8, T4",
    "v_mov_b32_e32 T9, T4",
    "v_cmp_lt_u32_e64 M3, 63, T1",
    "v_cvt_u32_f32_e32 T0, T0",
    "v_mad_co_u64_u32 T[2:3], null, 0xfe5163ab, T0, 0",
    "v_mad_co_u64_u32 T[5:6], null, 0x3c439041, T0, T[3:4]",
    "v_cndmask_b32_e64 T3, 0, 0xffffffc0, M3",
    "v_mov_b32_e32 T10, T4",
    "v_add_nc_u32_e32 T1, T3, T1",
    "v_mad_co_u64_u32 T[6:7], null, 0xdb629599, T0, T[6:7]",
    "v_cmp_lt_u32_e64 M1, 31, T1",
    "v_cndmask_b32_e64 T3, 0, 0xffffffe0, M1",
    "v_mad_co_u64_u32 T[7:8], null, 0xf534ddc0, T0, T[7:8]",
    "v_mad_co_u64_u32 T[8:9], null, 0xfc2757d1, T0, T[8:9]",
    "v_add_nc_u32_e32 T3, T3, T1",
    "v_mad_co_u64_u32 T[9:10], null, 0x4e441529, T0, T[9:10]",
    "v_mov_b32_e32 T11, T4",
    "v_mad_co_u64_u32 T[0:1], null, 0xa2f9836e, T0, T[10:11]",
    "v_cndmask_b32_e64 T10, T9, T7, M3",
    "v_cndmask_b32_e64 T0, T0, T8, M3",
    "v_cmp_lt_u32_e64 M2, 31, T3",
    "v_cndmask_b32_e64 T1, T1, T9, M3",
    "v_cndmask_b32_e64 T8, T8, T6, M3",
    "v_cndmask_b32_e64 T4, 0, 0xffffffe0, M2",
    "v_add_nc_u32_e32 T3, T4, T3",
    "v_cndmask_b32_e64 T4, T7, T5, M3",
    "v_cndmask_b32_e64 T5, T0, T10, M1",
    "v_cndmask_b32_e64 T0, T1, T0, M1",
    "v_cndmask_b32_e64 T1, T10, T8, M1",
    "v_sub_nc_u32_e32 T7, 32, T3",
    "v_cndmask_b32_e64 T8, T8, T4, M1",
    "v_cndmask_b32_e64 T0, T0, T5, M2",
    "v_cndmask_b32_e64 T5, T5, T1, M2",
    "v_cndmask_b32_e64 T2, T6, T2, M3",
    "v_cndmask_b32_e64 T1, T1, T8, M2",
    "v_cmp_eq_u32_e64 M3, 0, T3",
    "v_alignbit_b32 T9, T0, T5, T7",
    "v_alignbit_b32 T6, T5, T1, T7",
    "v_cndmask_b32_e64 T3, T6, T5, M3",
    "v_cndmask_b32_e64 T2, T4, T2, M1",
    "v_cndmask_b32_e64 T2, T8, T2, M2",
    "v_alignbit_b32 T7, T1, T2, T7",
    "v_cndmask_b32_e64 T0, T9, T0, M3",
    "v_cndmask_b32_e64 T1, T7, T1, M3",
    "v_bfe_u32 T4, T0, 29, 1",
    "v_alignbit_b32 T5, T0, T3, 30",
    "v_alignbit_b32 T3, T3, T1, 30",
    "v_sub_nc_u32_e32 T6, 0, T4",
    "v_alignbit_b32 T1, T1, T2, 30",
    "v_xor_b32_e32 T5, T5, T6",
    "v_xor_b32_e32 T2, T3, T6",
    "v_xor_b32_e32 T1, T1, T6",
    "v_lshrrev_b32_e32 T6, 29, T0",
    "v_lshrrev_b32_e32 T0, 30, T0",
    "v_clz_i32_u32_e32 T7, T5",
    "v_min_u32_e32 T7, 32, T7",
    "v_sub_nc_u32_e32 T3, 31, T7",
    "v_lshlrev_b32_e32 T8, 23, T7",
    "v_alignbit_b32 T5, T5, T2, T3",
    "v_alignbit_b32 T1, T2, T1, T3",
    "v_lshlrev_b32_e32 T2, 31, T6",
    "v_alignbit_b32 T3, T5, T1, 9",
    "v_or_b32_e32 T6, 0.5, T2",
    "v_lshrrev_b32_e32 T5, 9, T5",
    "v_clz_i32_u32_e32 T9, T3",
    "v_sub_nc_u32_e32 T6, T6, T8",
    "v_min_u32_e32 T8, 32, T9",
    "v_or_b32_e32 T5, T5, T6",
    "v_not_b32_e32 T6, T8",
    "v_mul_f32_e32 T9, 0x3fc90fda, T5",
    "v_or_b32_e32 T2, 0x33000000, T2",
    "v_add_lshl_u32 T7, T8, T7, 23",
    "v_alignbit_b32 T1, T3, T1, T6",
    "v_fma_f32 T3, 0x3fc90fda, T5, -T9",
    "v_sub_nc_u32_e32 T2, T2, T7",
    "v_lshrrev_b32_e32 T1, 9, T1",
    "v_fmamk_f32 T3, T5, 0x33a22168, T3",
    "v_or_b32_e32 T1, T2, T1",
    "v_fmac_f32_e32 T3, 0x3fc90fda, T1",
    "v_add_nc_u32_e32 T1, T4, T0",
    "v_and_b32_e32 T2, 3, T1",
    "v_add_f32_e32 T0, T9, T3",
];

/// Cody-Waite reduction of `AX = T1`: `T0` = reduced argument, `T2` = quadrant & 3.
const SMALL: &[&str] = &[
    "v_mul_f32_e32 T0, 0x3f22f983, T1",
    "v_rndne_f32_e32 T0, T0",
    "v_cvt_i32_f32_e32 T2, T0",
    "v_fmamk_f32 T1, T0, 0xbfc90fda, T1",
    "v_and_b32_e32 T2, 3, T2",
    "v_fmamk_f32 T1, T0, 0xb3a22168, T1",
    "v_fmamk_f32 T0, T0, 0xa7c234c4, T1",
];

/// `powf(X, Y)` into `O`, straight-line.
const POWF: &[&str] = &[
    "v_cmp_neq_f32_e64 M0, 1.0, X",
    "v_cndmask_b32_e64 T1, 1.0, Y, M0",
    "v_cmp_neq_f32_e64 M0, 0, T1",
    "v_cndmask_b32_e64 T0, 1.0, X, M0",
    "v_cmp_neq_f32_e64 M1, T1, |T1|",
    "v_frexp_mant_f32_e64 T2, |T0|",
    "v_cmp_lt_f32_e64 M2, |T0|, 1.0",
    "v_cmp_eq_f32_e64 M3, 0, T0",
    "v_cmp_ngt_f32_e64 M4, 0, T0",
    "v_cmp_gt_f32_e64 M0, 0x3f2aaaab, T2",
    "v_cndmask_b32_e64 T3, 1.0, 2.0, M0",
    "v_mul_f32_e32 T2, T2, T3",
    "v_add_f32_e32 T3, 1.0, T2",
    "v_rcp_f32_e32 T4, T3",
    "v_add_f32_e32 T7, -1.0, T3",
    "v_add_f32_e32 T5, -1.0, T2",
    "v_sub_f32_e32 T2, T2, T7",
    "v_mul_f32_e32 T6, T5, T4",
    "v_mul_f32_e32 T8, T3, T6",
    "v_fma_f32 T3, T6, T3, -T8",
    "v_fmac_f32_e32 T3, T6, T2",
    "v_add_f32_e32 T2, T8, T3",
    "v_sub_f32_e32 T7, T5, T2",
    "v_sub_f32_e32 T5, T5, T7",
    "v_sub_f32_e32 T8, T2, T8",
    "v_sub_f32_e32 T2, T5, T2",
    "v_sub_f32_e32 T3, T8, T3",
    "v_add_f32_e32 T2, T3, T2",
    "v_add_f32_e32 T2, T7, T2",
    "v_mul_f32_e32 T2, T4, T2",
    "v_add_f32_e32 T3, T6, T2",
    "v_sub_f32_e32 T4, T3, T6",
    "v_mul_f32_e32 T5, T3, T3",
    "v_sub_f32_e32 T2, T2, T4",
    "v_fma_f32 T4, T3, T3, -T5",
    "v_add_f32_e32 T6, T2, T2",
    "v_fmac_f32_e32 T4, T3, T6",
    "v_add_f32_e32 T7, T5, T4",
    "v_fmaak_f32 T8, K0, T7, 0x3e91f4c4",
    "v_mul_f32_e32 T11, T3, T7",
    "v_sub_f32_e32 T5, T7, T5",
    "v_fmaak_f32 T8, T7, T8, 0x3ecccdef",
    "v_fma_f32 T12, T7, T3, -T11",
    "v_sub_f32_e32 T4, T4, T5",
    "v_mul_f32_e32 T9, T7, T8",
    "v_fma_f32 T5, T7, T8, -T9",
    "v_fmac_f32_e32 T5, T4, T8",
    "v_add_f32_e32 T8, T9, T5",
    "v_sub_f32_e32 T9, T8, T9",
    "v_add_f32_e32 T10, 0x3f2aaaaa, T8",
    "v_fmac_f32_e32 T12, T7, T2",
    "v_sub_f32_e32 T5, T5, T9",
    "v_add_f32_e32 T9, 0xbf2aaaaa, T10",
    "v_fmac_f32_e32 T12, T4, T3",
    "v_add_f32_e32 T3, T3, T3",
    "v_add_f32_e32 T5, 0x31739010, T5",
    "v_sub_f32_e32 T8, T8, T9",
    "v_add_f32_e32 T2, T5, T8",
    "v_add_f32_e32 T5, T11, T12",
    "v_sub_f32_e32 T9, T5, T11",
    "v_sub_f32_e32 T9, T12, T9",
    "v_add_f32_e32 T4, T10, T2",
    "v_sub_f32_e32 T7, T10, T4",
    "v_mul_f32_e32 T8, T5, T4",
    "v_add_f32_e32 T2, T2, T7",
    "v_fma_f32 T7, T5, T4, -T8",
    "v_fmac_f32_e32 T7, T5, T2",
    "v_frexp_exp_i32_f32_e32 T2, T0",
    "v_fmac_f32_e32 T7, T9, T4",
    "v_subrev_co_ci_u32_e64 T2, null, 0, T2, M0",
    "v_cvt_f32_i32_e32 T2, T2",
    "v_add_f32_e32 T4, T8, T7",
    "v_mul_f32_e32 T9, 0x3f317218, T2",
    "v_add_f32_e32 T5, T3, T4",
    "v_sub_f32_e32 T8, T4, T8",
    "v_sub_f32_e32 T3, T5, T3",
    "v_sub_f32_e32 T7, T7, T8",
    "v_fma_f32 T8, 0x3f317218, T2, -T9",
    "v_fmamk_f32 T2, T2, 0xb102e308, T8",
    "v_sub_f32_e32 T3, T4, T3",
    "v_add_f32_e32 T4, T6, T7",
    "v_add_f32_e32 T3, T4, T3",
    "v_add_f32_e32 T4, T9, T2",
    "v_add_f32_e32 T6, T5, T3",
    "v_sub_f32_e32 T9, T4, T9",
    "v_add_f32_e32 T7, T4, T6",
    "v_sub_f32_e32 T2, T2, T9",
    "v_sub_f32_e32 T5, T6, T5",
    "v_sub_f32_e32 T8, T7, T4",
    "v_sub_f32_e32 T10, T7, T8",
    "v_sub_f32_e32 T3, T3, T5",
    "v_sub_f32_e32 T5, T6, T8",
    "v_sub_f32_e32 T4, T4, T10",
    "v_add_f32_e32 T6, T2, T3",
    "v_add_f32_e32 T4, T5, T4",
    "v_sub_f32_e32 T5, T6, T2",
    "v_sub_f32_e32 T3, T3, T5",
    "v_add_f32_e32 T4, T6, T4",
    "v_sub_f32_e32 T6, T6, T5",
    "v_add_f32_e32 T8, T7, T4",
    "v_sub_f32_e32 T2, T2, T6",
    "v_sub_f32_e32 T5, T8, T7",
    "v_add_f32_e32 T2, T3, T2",
    "v_sub_f32_e32 T3, T4, T5",
    "v_add_f32_e32 T2, T2, T3",
    "v_add_f32_e32 T3, T8, T2",
    "v_mul_f32_e32 T5, T1, T3",
    "v_sub_f32_e32 T4, T3, T8",
    "v_cmp_class_f32_e64 M0, T5, 0x204",
    "v_sub_f32_e32 T2, T2, T4",
    "v_fma_f32 T3, T1, T3, -T5",
    "v_fmac_f32_e32 T3, T1, T2",
    "v_add_f32_e32 T2, T5, T3",
    "v_cndmask_b32_e64 T4, T2, T5, M0",
    "v_sub_f32_e32 T2, T2, T5",
    "v_cmp_eq_f32_e64 M0, 0x42b17218, T4",
    "v_sub_f32_e32 T2, T3, T2",
    "v_mul_f32_e32 T3, 0.5, T1",
    "v_cndmask_b32_e64 T6, 0, 0x37000000, M0",
    "v_cmp_neq_f32_e64 M0, 0x7f800000, |T4|",
    "v_sub_f32_e32 T7, T4, T6",
    "v_cndmask_b32_e64 T2, 0, T2, M0",
    "v_mul_f32_e32 T8, 0x3fb8aa3b, T7",
    "v_cmp_ngt_f32_e64 M0, 0xc2ce8ed0, T7",
    "v_cmp_nlt_f32_e64 M5, 0x42b17218, T7",
    "v_add_f32_e32 T2, T6, T2",
    "v_fma_f32 T9, 0x3fb8aa3b, T7, -T8",
    "v_rndne_f32_e32 T10, T8",
    "v_fmamk_f32 T9, T7, 0x32a5705f, T9",
    "v_sub_f32_e32 T8, T8, T10",
    "v_cvt_i32_f32_e32 T5, T10",
    "v_add_f32_e32 T8, T8, T9",
    "v_trunc_f32_e32 T9, T1",
    "v_exp_f32_e32 T8, T8",
    "v_ldexp_f32 T5, T8, T5",
    "v_cndmask_b32_e64 T4, 0, T5, M0",
    "v_trunc_f32_e32 T5, T3",
    "v_cmp_gt_f32_e64 M0, 0, T1",
    "v_cndmask_b32_e64 T4, 0x7f800000, T4, M5",
    "v_cmp_eq_f32_e64 M5, T9, T1",
    "v_cmp_neq_f32_e64 M6, T5, T3",
    "v_fma_f32 T2, T4, T2, T4",
    "v_cmp_class_f32_e64 M7, T4, 0x204",
    "s_and_b32 M6, M5, M6",
    "v_cndmask_b32_e64 T3, 1.0, T0, M6",
    "v_cndmask_b32_e64 T2, T2, T4, M7",
    "s_xor_b32 M7, M1, M2",
    "v_cndmask_b32_e64 T4, 0x7f800000, 0, M7",
    "s_xor_b32 M7, M0, M3",
    "v_cmp_neq_f32_e64 M0, |T0|, 1.0",
    "v_bfi_b32 T2, 0x7fffffff, T2, T3",
    "v_cndmask_b32_e64 T5, 0x7f800000, 0, M7",
    "v_cndmask_b32_e64 T3, 1.0, T4, M0",
    "s_or_b32 M0, M5, M4",
    "v_cndmask_b32_e64 T4, 0, T0, M6",
    "v_cndmask_b32_e64 T2, 0x7fc00000, T2, M0",
    "v_cmp_class_f32_e64 M0, T1, 0x204",
    "v_cmp_class_f32_e64 M5, T0, 0x204",
    "v_bfi_b32 T4, 0x7fffffff, T5, T4",
    "v_cndmask_b32_e64 T2, T2, T3, M0",
    "s_or_b32 M0, M3, M5",
    "v_cndmask_b32_e64 T2, T2, T4, M0",
    "v_cmp_o_f32_e64 M0, T0, T1",
    "v_cndmask_b32_e64 O, 0x7fc00000, T2, M0",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn env(arch: Arch) -> Env {
        Env {
            arch,
            x: 20,
            y: 21,
            o: 22,
            so: 23,
            co: 24,
            v: 40,
            s: 16,
            nm: 4,
        }
    }

    #[test]
    fn subst_names() {
        let e = env(Arch::Gfx1201);
        assert_eq!(
            subst("v_mad_co_u64_u32 T[2:3], null, 0xfe5163ab, T0, 0", &e).unwrap(),
            "v_mad_co_u64_u32 v[42:43], null, 0xfe5163ab, v40, 0"
        );
        assert_eq!(
            subst(
                "v_mad_co_u64_u32 T[2:3], null, 0xfe5163ab, T0, 0",
                &env(Arch::Gfx1100)
            )
            .unwrap(),
            "v_mad_u64_u32 v[42:43], null, 0xfe5163ab, v40, 0"
        );
        assert_eq!(
            subst("v_cndmask_b32_e64 SO, T1, T0, M3", &e).unwrap(),
            "v_cndmask_b32_e64 v23, v41, v40, s22"
        );
        assert_eq!(
            subst("s_mov_b32 K2, 0x37d75334", &e).unwrap(),
            "s_mov_b32 s26, 0x37d75334"
        );
    }

    #[test]
    fn derive_defs_uses() {
        let (d, u) = derive("v_fmac_f32_e32 v1, 0x3fc90fda, v3");
        assert_eq!((d.len(), u.len()), (1, 2));
        let (d, u) = derive("v_cmp_neq_f32_e64 s4, 0x7f800000, |v2|");
        assert_eq!((d[0].kind, d[0].base, u.len()), (Kind::S, 4, 1));
        let (d, u) = derive("v_mad_u64_u32 v[3:4], null, 0xfe5163ab, v13, v[4:5]");
        assert_eq!((d.len(), u.len()), (2, 3));
        let (d, u) = derive("s_and_b32 exec_lo, exec_lo, s2");
        assert_eq!((d.len(), u.len()), (0, 1));
        let (d, _) = derive("v_cmpx_le_f32_e64 0x48000000, |v2|");
        assert!(d.is_empty());
        let (d, u) = derive("v_alignbit_b32 v12, v4, v8, v11");
        assert_eq!((d[0].base, u.len(), u[2].base), (12, 3, 11));
    }

    #[test]
    fn templates_are_clean() {
        for (name, seq) in templates() {
            for t in seq {
                assert!(!t.contains("vcc"), "{name}: {t}");
                assert!(!t.contains("_dual_"), "{name}: {t}");
                for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201] {
                    let text = subst(t, &env(arch)).unwrap_or_else(|e| panic!("{name}: {t}: {e}"));
                    assert!(
                        !text.chars().any(|c| c.is_ascii_uppercase()),
                        "{name}: unresolved name in {text}"
                    );
                    let _ = derive(&text);
                }
            }
        }
    }

    #[test]
    fn vgpr_temps_fit() {
        // the highest T<n> any sequence names must fit the advertised block
        let mut hi_trig = 0;
        let mut hi_pow = 0;
        for (name, seq) in templates() {
            for t in seq {
                let b = t.as_bytes();
                let mut i = 0;
                while i + 1 < b.len() {
                    if b[i] == b'T' && (i == 0 || !b[i - 1].is_ascii_alphanumeric()) {
                        let mut j = i + 1;
                        if b[j] == b'[' {
                            j += 1;
                        }
                        let mut k = j;
                        while k < b.len() && (b[k].is_ascii_digit() || b[k] == b':') {
                            k += 1;
                        }
                        for n in t[j..k].split(':').filter_map(|n| n.parse::<u8>().ok()) {
                            if name == "powf" {
                                hi_pow = hi_pow.max(n)
                            } else {
                                hi_trig = hi_trig.max(n)
                            }
                        }
                        i = k;
                    } else {
                        i += 1;
                    }
                }
            }
        }
        assert_eq!(hi_trig + 1, TRIG_VGPRS);
        assert_eq!(hi_pow + 1, POW_VGPRS);
    }
}
