// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201 Qwen GDN chunk preparation for the A3B geometry (HK = 16 key heads, HV = 32 value heads, head dim 128),
//! emitted with the checked ISA builder. One export, `gdn_chunk_prep_pm_gfx1201`, in its own module of the same
//! name: the PM twin of the incumbent HIP kernel `kernels/src/gdn_chunk_scan_prep.gfx1201.hip` (`gdn_chunk_prep`,
//! compiled `-O3 -ffp-contract=off`; its disassembly is the reference for every operand DAG below).
//!
//! ABI (100 bytes, the incumbent's): `x` f32 `[T, 8192]` @0, `conv_w` f32 `[8192, 4]` @8, `conv_state` f32
//! `[8192, 3]` @16 (read/write), `a` f32 `[T, 32]` @24 (in place), `b` f32 `[T, 32]` @32 (in place), `dt_bias`
//! f32 `[32]` @40, `a_log` f32 `[32]` @48, `q` f16 `[T, 16, 128]` @56, `k` f16 `[T, 16, 128]` @64, `v` f16
//! `[T, 32, 128]` @72, `T` i32 @80, `q_scale` f32 @84, `eps` f32 @88, `hk` u32 @92, `hv` u32 @96. Grid
//! `(ceil(T / 64), 8)`, block 256 = eight waves, no LDS. Every launch with `T` outside `1..=8192`, `hk != 16` or
//! `hv != 32` exits before any access.
//!
//! One block owns a 64-row chunk; wave `w` of block `(bx, by)` owns unit `u = 8 by + w` of the 64 conv units
//! (`u < 16`: q head `u`; `u < 32`: k head `u - 16`; else v head `u - 32`), four channels per lane
//! (`dbase = 128 u + 4 lane`). Blocks with `by == 0` first run the gates (each wave: heads `w, w + 8, w + 16,
//! w + 24`; two adjacent rows per lane; the inclusive 64-row prefix is wave-local).
//!
//! Arithmetic (per element and per reduction exactly the incumbent's instruction DAG; the compiler's exec-mask
//! branches become `v_cndmask` selects over the same operands, which are pure):
//!
//!  * **expf** - `x * C1`, `fma(C1, x, -hi)`, `rndne`, `fma-acc C2 * x`, `hi - rint`, `+ lo`, `cvt_i32`,
//!    `v_exp_f32`, `ldexp`, then the under/overflow selects (`x >= -103.97 ? e : 0`, `x <= 88.72 ? e : inf`); the
//!    negated form (`C1 = -log2e`, `C2 = -lo`) with the mirrored clamps on `x` is the sigmoid / SiLU exponent.
//!  * **softplus** - `x > 20 ? x : (x < -20 ? expf(x) : logf(1 + expf(x)))`, `logf` as the OCML sequence
//!    (denormal prescale by `2^32` through `v_ldexp`, `v_log_f32`, `ln2` hi/lo split product with the finite
//!    select, minus the prescale correction).
//!  * **gate** - `g = softplus(a + dt_bias[h]) * (-expf(a_log[h]))`; `beta = 1 / (1 + expf(-b))` with the full
//!    division graph; the prefix is the Hillis-Steele scan `d = 1, 2, 4, 8, 16` over `pair = g0 + g1` through
//!    `ds_bpermute_b32` (lane `l - d`, guard `l >= d`), `before = (pair - g0) - g1`, `a0 = g0 + before`,
//!    `a1 = g1 + a0`; out-of-range rows contribute `+0`.
//!  * **conv + SiLU** - `y = fma(w0, win0, fma(w1, win1, fma(w3, cur, w2 * win2)))`, `out = y / (1 + expf(-y))`
//!    (full division graph, one per channel).
//!  * **q/k norm** - `sum = ((c0^2 + c1^2) + c2^2) + c3^2` (separate multiplies and adds), lane XOR butterfly
//!    16 / 8 / 4 / 2 / 1 (`v_permlanex16_b32` + DPP `row_xmask`; add is commutative, so every lane ends with the
//!    same root), `inv = rsq(sum + eps)` through the vector denormal guard (`x < 2^-126` scales by `2^24`,
//!    `v_rsq_f32`, scale back by `2^12`), `z = out * inv`, q only `z * q_scale` (a separate multiply).
//!  * **f16** - `v_cvt_f16_f32` (RNE) pairs, one `global_store_b64` per lane (the four channels).
//!  * **conv_state** - the chunk-0 blocks advance each channel's three-deep ring after their rows (new0 =
//!    `x[T-1]`, new1 = `T >= 2 ? x[T-2] : old0`, new2 = `T >= 3 ? x[T-3] : (T == 2 ? old0 : old1)`).
//!
//! Lane exchange uses `ds_bpermute_b32` (no LDS allocation) and DPP / `v_permlanex16_b32`; all DS results are
//! drained by the builder's DScnt ledger.

use super::fa_prefill::{cvt_half, wait_alu};
use crate::insn::{Instruction, MemoryClass, Sop};
use crate::kernels::common::{add64, lit, op, s, smem, sop, sr, v, vr};
use crate::ledger::Counter;
use crate::plan::Access;
use crate::reg::{Live, RegRef};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// CLI kernel id.
pub const KERNEL_ID: &str = "gdn_chunk_prep_pm";
/// Module / cache key.
pub const MODULE: &str = "gdn_chunk_prep_pm_gfx1201";
/// The export symbol.
pub const SYMBOL: &str = "gdn_chunk_prep_pm_gfx1201";
/// Block size: eight waves.
pub const BLOCK: u32 = 256;
/// Kernarg segment bytes.
pub const KERNARG_BYTES: u32 = 100;
/// Key heads.
pub const HK: u32 = 16;
/// Value heads.
pub const HV: u32 = 32;
/// Head dimension.
pub const HEAD_DIM: u32 = 128;
/// Conv units (`2 * HK + HV`), one wave each.
pub const UNITS: u32 = 64;
/// Rows per block.
pub const CHUNK: u32 = 64;
/// Grid blocks in y (`UNITS / 8`).
pub const GRID_Y: u32 = 8;
/// Largest launch row count of the authoring domain.
pub const MAX_ROWS: usize = 8192;

/// The one supported geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GdnChunkPrepSpec {
    pub hk: u32,
    pub hv: u32,
    pub head_dim: u32,
}

impl GdnChunkPrepSpec {
    /// A3B: 16 key heads, 32 value heads, head dim 128.
    pub const A3B: GdnChunkPrepSpec = GdnChunkPrepSpec {
        hk: HK,
        hv: HV,
        head_dim: HEAD_DIM,
    };

    /// Exactly `(16, 32, 128)`.
    pub fn validate(&self) -> Result<(), String> {
        if *self != Self::A3B {
            return Err(format!(
                "gdn_chunk_prep: only hk 16, hv 32, head_dim 128 are supported, got {}/{}/{}",
                self.hk, self.hv, self.head_dim
            ));
        }
        Ok(())
    }
}

/// Pure launch validation: valid spec and `1 <= rows <= 8192`.
pub fn validate(spec: &GdnChunkPrepSpec, rows: usize) -> Result<(), String> {
    spec.validate()?;
    if rows < 1 || rows > MAX_ROWS {
        return Err(format!(
            "gdn_chunk_prep: rows {rows} is outside 1..={MAX_ROWS}"
        ));
    }
    Ok(())
}

fn admit(arch: Arch, spec: &GdnChunkPrepSpec) -> R {
    if arch != Arch::Gfx1201 {
        return Err("gdn_chunk_prep: built for gfx1201 only".into());
    }
    spec.validate()
}

// ------------------------------------------------------------------ labels
const END: &str = ".Lgdcp_end";
const GATE_LOOP: &str = ".Lgdcp_gate";
const GATE_DONE: &str = ".Lgdcp_gate_done";
const STATE_WIN: &str = ".Lgdcp_state_win";
const WIN_DONE: &str = ".Lgdcp_win_done";
const K_START: &str = ".Lgdcp_k_start";
const V_START: &str = ".Lgdcp_v_start";
const EPI: &str = ".Lgdcp_epi";
const Q_LOOP: &str = ".Lgdcp_q";
const K_LOOP: &str = ".Lgdcp_k";
const V_LOOP: &str = ".Lgdcp_v";

// ------------------------------------------------------------------ VGPR map
const TID: u8 = 0;
const LANE: u8 = 1;
const GAD: u8 = 2; // five ds_bpermute byte addresses (shfl_up d = 1, 2, 4, 8, 16)
const VOFFG: u8 = 7; // (chunk0 + 2 lane) * 128
const VADDR: u8 = 8; // VOFFG + 4 h
const G0: u8 = 9;
const G1: u8 = 10;
const PAIR: u8 = 11;
const OTH: u8 = 12;
const BEF: u8 = 13;
const RA: u8 = 14;
const RB: u8 = 15;
const EX: u8 = 16;
const L1: u8 = 17;
const LG: u8 = 18;
const M1: u8 = 19;
const M2: u8 = 20;
const SC1: u8 = 21;
const SIG: u8 = 22;
const EA: u8 = 23;
const GT: u8 = 24;
const TA1: u8 = 25;
const TE1: u8 = 26;
const TN1: u8 = 27;
const GDIV: u8 = 28; // four fdiv temporaries
const SPV: u8 = 32;
const VOFFW: u8 = 33; // lane * 64 (conv_w)
const VST: u8 = 34; // lane * 48 (conv_state)
const VOFFX: u8 = 35; // lane * 16 (x)
const VOFFO: u8 = 36; // lane * 8 (f16 outputs)
const VT0: u8 = 37;
const VT1: u8 = 38;
const W: u8 = 40; // w[j][c] at 40 + 4 c + j
const WIN0: u8 = 56;
const WIN1: u8 = 60;
const WIN2: u8 = 64;
const XB: u8 = 68; // current row, four channels
const YB: u8 = 72; // prefetched next row
const Y: u8 = 76;
const OUT: u8 = 80;
const TA: u8 = 84;
const TE: u8 = 88;
const TN: u8 = 92;
const DSC: u8 = 96;
const DR: u8 = 100;
const DQ: u8 = 104;
const DE: u8 = 108;
const NSUM: u8 = 112;
const NPEER: u8 = 113;
const NINV: u8 = 114;
const NT: u8 = 115;
const HP: u8 = 116; // two packed f16x2 words
const SQ: u8 = 118; // four squares
const VGPRS: u16 = 128; // 122 used, rounded to the allocation granule

// ------------------------------------------------------------------ SGPR map
const KARG: u8 = 0;
const S_X: u8 = 8;
const S_CW: u8 = 10;
const S_ST: u8 = 12;
const S_A: u8 = 14;
const S_B: u8 = 16;
const S_DT: u8 = 18;
const S_ALOG: u8 = 20;
const S_Q: u8 = 22;
const S_K: u8 = 24;
const S_V: u8 = 26;
const S_T: u8 = 28;
const S_QS: u8 = 29;
const S_EPS: u8 = 30;
const S_HK: u8 = 31;
const S_HV: u8 = 32;
const S_WAVE: u8 = 33;
const S_C0: u8 = 34;
const S_ROWS: u8 = 35;
const S_UNIT: u8 = 36;
const S_TMP: u8 = 37;
const S_TMP2: u8 = 38;
const S_PSEL: u8 = 39;
const S_I: u8 = 40;
const S_H: u8 = 41;
const S_DTV: u8 = 42;
const S_ALV: u8 = 43;
const S_HB4: u8 = 44;
const GE: u8 = 45; // five lane >= d masks
const S_M0: u8 = 50;
const S_M1: u8 = 51;
const S_SAVE: u8 = 53;
const S_EM: u8 = 54;
const S_CM: u8 = 55;
const PGAT: u8 = 56;
const PW: u8 = 58;
const PS: u8 = 60;
const PXU: u8 = 62;
const PX: u8 = 64;
const PN: u8 = 66;
const PO: u8 = 68;
const S_OSTR: u8 = 70;
const CM4: u8 = 72;
const DM4: u8 = 76;
const S_M2: u8 = 80;
const S_M3: u8 = 81;
const S_MEQ: u8 = 82;
const P1: u8 = 84;
const P2: u8 = 86;
const P3: u8 = 88;
const PGA2: u8 = 90;

fn plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(VGPRS.next_multiple_of(8), 104)?;
    let w = || Live::Whole;
    for (name, base) in [
        ("tid", TID),
        ("lane", LANE),
        ("voff_gate", VOFFG),
        ("vaddr", VADDR),
        ("g0", G0),
        ("g1", G1),
        ("pair", PAIR),
        ("other", OTH),
        ("before", BEF),
        ("ra", RA),
        ("rb", RB),
        ("ex", EX),
        ("l1", L1),
        ("lg", LG),
        ("m1", M1),
        ("m2", M2),
        ("sc1", SC1),
        ("sig", SIG),
        ("ea", EA),
        ("gt", GT),
        ("ta1", TA1),
        ("te1", TE1),
        ("tn1", TN1),
        ("softplus", SPV),
        ("voff_w", VOFFW),
        ("voff_state", VST),
        ("voff_x", VOFFX),
        ("voff_out", VOFFO),
        ("vt0", VT0),
        ("vt1", VT1),
        ("sum", NSUM),
        ("peer", NPEER),
        ("inv", NINV),
        ("nt", NT),
    ] {
        p.v::<1>(name, base, w())?;
    }
    p.v::<4>("scan_addr", GAD, w())?;
    p.v::<1>("scan_addr16", GAD + 4, w())?;
    p.v::<4>("gate_div", GDIV, w())?;
    p.v::<8>("conv_w_lo", W, w())?;
    p.v::<8>("conv_w_hi", W + 8, w())?;
    for (name, base) in [
        ("win0", WIN0),
        ("win1", WIN1),
        ("win2", WIN2),
        ("x_cur", XB),
        ("x_next", YB),
        ("y", Y),
        ("out", OUT),
        ("exp_a", TA),
        ("exp_e", TE),
        ("exp_n", TN),
        ("div_sc", DSC),
        ("div_r", DR),
        ("div_q", DQ),
        ("div_e", DE),
        ("squares", SQ),
    ] {
        p.v::<4>(name, base, w())?;
    }
    p.v::<2>("packed", HP, w())?;
    // The descriptor reserves whole granules of eight (the certification requires the report to equal it).
    p.v::<4>("granule_pad", 122, w())?;
    p.v::<2>("granule_pad", 126, w())?;
    p.s::<2>("kernarg_ptr", KARG, w())?;
    p.s::<8>("ptrs0", S_X, w())?;
    p.s::<8>("ptrs1", S_B, w())?;
    p.s::<4>("ptrs2", S_K, w())?;
    p.s::<4>("tail", S_T, w())?;
    for n in [
        S_HV, S_WAVE, S_C0, S_ROWS, S_UNIT, S_TMP, S_TMP2, S_PSEL, S_I, S_H, S_DTV, S_ALV, S_HB4,
        S_M0, S_M1, S_SAVE, S_EM, S_CM, S_OSTR, S_M2, S_M3, S_MEQ,
    ] {
        p.s::<1>("scalar", n, w())?;
    }
    for n in GE..GE + 5 {
        p.s::<1>("scan_mask", n, w())?;
    }
    p.s::<4>("exp_masks", CM4, w())?;
    p.s::<4>("div_masks", DM4, w())?;
    for n in [PGAT, PW, PS, PXU, PX, PN, PO, P1, P2, P3, PGA2] {
        p.s::<2>("ptr_pair", n, w())?;
    }
    Ok(p)
}

// ------------------------------------------------------------------ helpers
fn off(imm: u32) -> String {
    if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    }
}

/// `global_load_<kind> dst, v{voff}, s[sbase:sbase+1] offset:imm` (SADDR form, 32-bit lane offset).
fn gload(b: &mut Builder, kind: &str, dst: RegRef, voff: u8, sbase: u8, imm: u32) -> R {
    b.push(
        Instruction::new(
            format!(
                "global_load_{kind} {dst}, v{voff}, s[{sbase}:{}]{}",
                sbase + 1,
                off(imm)
            ),
            vec![dst],
            vec![v(voff), sr(sbase, 2)],
        )
        .memory(MemoryClass::VmemLoad),
    )
}

/// `global_store_<kind> v{voff}, data, s[sbase:sbase+1] offset:imm`.
fn gstore(b: &mut Builder, kind: &str, voff: u8, data: RegRef, sbase: u8, imm: u32) -> R {
    b.push(
        Instruction::new(
            format!(
                "global_store_{kind} v{voff}, {data}, s[{sbase}:{}]{}",
                sbase + 1,
                off(imm)
            ),
            vec![],
            vec![v(voff), data, sr(sbase, 2)],
        )
        .memory(MemoryClass::VmemStore),
    )
}

/// VALU instruction over plain VGPRs.
fn vop(b: &mut Builder, text: String, defs: &[u8], uses: &[u8]) -> R {
    let d: Vec<RegRef> = defs.iter().map(|&n| v(n)).collect();
    let u: Vec<RegRef> = uses.iter().map(|&n| v(n)).collect();
    op(b, text, &d, &u)
}

/// `exec &= mask`, saving the old exec (restore with [`exec_restore`]).
fn exec_and(b: &mut Builder, mask: u8) -> R {
    op(
        b,
        format!("s_mov_b32 s{S_SAVE}, exec_lo"),
        &[s(S_SAVE)],
        &[],
    )?;
    op(
        b,
        format!("s_and_b32 exec_lo, exec_lo, s{mask}"),
        &[],
        &[s(mask)],
    )
}

fn exec_restore(b: &mut Builder) -> R {
    op(
        b,
        format!("s_or_b32 exec_lo, exec_lo, s{S_SAVE}"),
        &[],
        &[s(S_SAVE)],
    )
}

/// One `expf` instance of [`exp_many`]: `dst = expf(±x)` through temporaries `a`, `e`, `n` and mask SGPR `m`.
#[derive(Clone, Copy)]
struct Exp {
    dst: u8,
    x: u8,
    a: u8,
    e: u8,
    n: u8,
    m: u8,
}

/// hipcc's `expf(x)` (`neg`: `expf(-x)`), op for op, interleaved across the instances (`x` is untouched).
fn exp_many(b: &mut Builder, items: &[Exp], neg: bool) -> R {
    let (c1, c2) = if neg {
        ("0xbfb8aa3b", "0xb2a5705f")
    } else {
        ("0x3fb8aa3b", "0x32a5705f")
    };
    for i in items {
        vop(
            b,
            format!("v_mul_f32_e32 v{}, {c1}, v{}", i.a, i.x),
            &[i.a],
            &[i.x],
        )?;
    }
    for i in items {
        vop(
            b,
            format!("v_fma_f32 v{}, {c1}, v{}, -v{}", i.e, i.x, i.a),
            &[i.e],
            &[i.x, i.a],
        )?;
    }
    for i in items {
        vop(
            b,
            format!("v_rndne_f32_e32 v{}, v{}", i.n, i.a),
            &[i.n],
            &[i.a],
        )?;
    }
    for i in items {
        vop(
            b,
            format!("v_fmac_f32_e32 v{}, {c2}, v{}", i.e, i.x),
            &[i.e],
            &[i.e, i.x],
        )?;
    }
    for i in items {
        vop(
            b,
            format!("v_sub_f32_e32 v{0}, v{0}, v{1}", i.a, i.n),
            &[i.a],
            &[i.a, i.n],
        )?;
    }
    for i in items {
        vop(
            b,
            format!("v_add_f32_e32 v{0}, v{0}, v{1}", i.a, i.e),
            &[i.a],
            &[i.a, i.e],
        )?;
    }
    for i in items {
        vop(
            b,
            format!("v_cvt_i32_f32_e32 v{0}, v{0}", i.n),
            &[i.n],
            &[i.n],
        )?;
    }
    for i in items {
        vop(b, format!("v_exp_f32_e32 v{0}, v{0}", i.a), &[i.a], &[i.a])?;
    }
    for i in items {
        vop(
            b,
            format!("v_ldexp_f32 v{0}, v{0}, v{1}", i.a, i.n),
            &[i.a],
            &[i.a, i.n],
        )?;
    }
    let (lo, hi) = if neg {
        ("v_cmp_nlt_f32_e64", "v_cmp_ngt_f32_e64")
    } else {
        ("v_cmp_ngt_f32_e64", "v_cmp_nlt_f32_e64")
    };
    let (lo_c, hi_c) = if neg {
        ("0x42ce8ed0", "0xc2b17218")
    } else {
        ("0xc2ce8ed0", "0x42b17218")
    };
    for i in items {
        op(
            b,
            format!("{lo} s{}, {lo_c}, v{}", i.m, i.x),
            &[s(i.m)],
            &[v(i.x)],
        )?;
    }
    wait_alu(b, "depctr_va_sdst(0)")?;
    for i in items {
        op(
            b,
            format!("v_cndmask_b32_e64 v{0}, 0, v{0}, s{1}", i.a, i.m),
            &[v(i.a)],
            &[v(i.a), s(i.m)],
        )?;
    }
    for i in items {
        op(
            b,
            format!("{hi} s{}, {hi_c}, v{}", i.m, i.x),
            &[s(i.m)],
            &[v(i.x)],
        )?;
    }
    wait_alu(b, "depctr_va_sdst(0)")?;
    for i in items {
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{}, 0x7f800000, v{}, s{}",
                i.dst, i.a, i.m
            ),
            &[v(i.dst)],
            &[v(i.a), s(i.m)],
        )?;
    }
    Ok(())
}

/// One division instance of [`fdiv_many`]: `dst = num / den` (`num = None`: `1.0`).
#[derive(Clone, Copy)]
struct Div {
    dst: u8,
    num: Option<u8>,
    den: u8,
    sc: u8,
    r: u8,
    q: u8,
    e: u8,
    m: u8,
}

/// hipcc's correctly rounded f32 division (`v_div_scale` / `v_rcp` / fma Newton steps / `v_div_fmas` /
/// `v_div_fixup`), op for op, interleaved across the instances; the second scale's flag goes to a per-instance SGPR
/// and is copied to VCC for its `v_div_fmas`, as the incumbent does.
fn fdiv_many(b: &mut Builder, items: &[Div]) -> R {
    let num = |i: &Div| match i.num {
        Some(n) => format!("v{n}"),
        None => "1.0".to_string(),
    };
    let uses = |i: &Div| {
        let mut u = vec![v(i.den)];
        if let Some(n) = i.num {
            u.push(v(n));
        }
        u
    };
    for i in items {
        op(
            b,
            format!(
                "v_div_scale_f32 v{}, null, v{}, v{}, {}",
                i.sc,
                i.den,
                i.den,
                num(i)
            ),
            &[v(i.sc)],
            &uses(i),
        )?;
    }
    for i in items {
        vop(
            b,
            format!("v_rcp_f32_e32 v{}, v{}", i.r, i.sc),
            &[i.r],
            &[i.sc],
        )?;
    }
    for i in items {
        vop(
            b,
            format!("v_fma_f32 v{}, -v{}, v{}, 1.0", i.e, i.sc, i.r),
            &[i.e],
            &[i.sc, i.r],
        )?;
    }
    for i in items {
        vop(
            b,
            format!("v_fmac_f32_e32 v{0}, v{1}, v{0}", i.r, i.e),
            &[i.r],
            &[i.r, i.e],
        )?;
    }
    for i in items {
        let mut d = vec![v(i.e)];
        d.push(s(i.m));
        op(
            b,
            format!(
                "v_div_scale_f32 v{}, s{}, {}, v{}, {}",
                i.e,
                i.m,
                num(i),
                i.den,
                num(i)
            ),
            &d,
            &uses(i),
        )?;
    }
    for i in items {
        vop(
            b,
            format!("v_mul_f32_e32 v{}, v{}, v{}", i.q, i.e, i.r),
            &[i.q],
            &[i.e, i.r],
        )?;
    }
    for i in items {
        vop(
            b,
            format!("v_fma_f32 v{}, -v{}, v{}, v{}", i.dst, i.sc, i.q, i.e),
            &[i.dst],
            &[i.sc, i.q, i.e],
        )?;
    }
    for i in items {
        vop(
            b,
            format!("v_fmac_f32_e32 v{0}, v{1}, v{2}", i.q, i.dst, i.r),
            &[i.q],
            &[i.q, i.dst, i.r],
        )?;
    }
    for i in items {
        vop(
            b,
            format!("v_fma_f32 v{0}, -v{0}, v{1}, v{2}", i.sc, i.q, i.e),
            &[i.sc],
            &[i.sc, i.q, i.e],
        )?;
    }
    wait_alu(b, "depctr_va_sdst(0)")?;
    for i in items {
        op(b, format!("s_mov_b32 vcc_lo, s{}", i.m), &[], &[s(i.m)])?;
        wait_alu(b, "depctr_sa_sdst(0)")?;
        vop(
            b,
            format!("v_div_fmas_f32 v{0}, v{0}, v{1}, v{2}", i.sc, i.r, i.q),
            &[i.sc],
            &[i.sc, i.r, i.q],
        )?;
    }
    for i in items {
        let mut u = vec![v(i.sc)];
        u.extend(uses(i));
        op(
            b,
            format!(
                "v_div_fixup_f32 v{}, v{}, v{}, {}",
                i.dst,
                i.sc,
                i.den,
                num(i)
            ),
            &[v(i.dst)],
            &u,
        )?;
    }
    Ok(())
}

// ------------------------------------------------------------------ prologue
/// Kernarg loads, per-wave identity, shape / row / unit guards (all uniform, before any data access).
fn prologue(b: &mut Builder) -> R {
    smem(b, S_X, 8, KARG, 0x00)?;
    smem(b, S_B, 8, KARG, 0x20)?;
    smem(b, S_K, 4, KARG, 0x40)?;
    smem(b, S_T, 4, KARG, 0x50)?;
    smem(b, S_HV, 1, KARG, 0x60)?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{VT0}, 5, v{TID}"),
        &[v(VT0)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{S_WAVE}, v{VT0}"),
        &[s(S_WAVE)],
        &[v(VT0)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{LANE}, 31, v{TID}"),
        &[v(LANE)],
        &[v(TID)],
    )?;
    b.wait(Counter::Km, 0)?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    // T outside 1..=8192, hk != 16, hv != 32.
    sop(
        b,
        format!("s_add_co_i32 s{S_TMP}, s{S_T}, -1"),
        &[S_TMP],
        &[S_T],
    )?;
    sop(
        b,
        format!("s_cmp_ge_u32 s{S_TMP}, {}", lit(MAX_ROWS as u32)),
        &[],
        &[S_TMP],
    )?;
    op(b, format!("s_cbranch_scc1 {END}"), &[], &[])?;
    sop(b, format!("s_cmp_lg_u32 s{S_HK}, {HK}"), &[], &[S_HK])?;
    op(b, format!("s_cbranch_scc1 {END}"), &[], &[])?;
    sop(b, format!("s_cmp_lg_u32 s{S_HV}, {HV}"), &[], &[S_HV])?;
    op(b, format!("s_cbranch_scc1 {END}"), &[], &[])?;
    // Chunk rows.
    op(b, format!("s_lshl_b32 s{S_C0}, ttmp9, 6"), &[s(S_C0)], &[])?;
    sop(
        b,
        format!("s_sub_co_i32 s{S_ROWS}, s{S_T}, s{S_C0}"),
        &[S_ROWS],
        &[S_T, S_C0],
    )?;
    sop(b, format!("s_cmp_lt_i32 s{S_ROWS}, 1"), &[], &[S_ROWS])?;
    op(b, format!("s_cbranch_scc1 {END}"), &[], &[])?;
    sop(
        b,
        format!("s_min_u32 s{S_ROWS}, s{S_ROWS}, {CHUNK}"),
        &[S_ROWS],
        &[S_ROWS],
    )?;
    // Unit (blocks past the 64 units exit; by == 0 always has units 0..8).
    op(
        b,
        format!("s_lshl_b32 s{S_UNIT}, ttmp7, 3"),
        &[s(S_UNIT)],
        &[],
    )?;
    sop(
        b,
        format!("s_add_co_i32 s{S_UNIT}, s{S_UNIT}, s{S_WAVE}"),
        &[S_UNIT],
        &[S_UNIT, S_WAVE],
    )?;
    sop(
        b,
        format!("s_cmp_ge_u32 s{S_UNIT}, {UNITS}"),
        &[],
        &[S_UNIT],
    )?;
    op(b, format!("s_cbranch_scc1 {END}"), &[], &[])?;
    sop(
        b,
        format!("s_mov_b32 s{S_PSEL}, 0x76543210"),
        &[S_PSEL],
        &[],
    )?;
    wait_alu(b, "depctr_sa_sdst(0)")
}

// ------------------------------------------------------------------ gates
/// Lane tables of the scan, the row masks and the gate row offset of this lane.
fn gate_setup(b: &mut Builder) -> R {
    for (i, d) in [1u32, 2, 4, 8, 16].into_iter().enumerate() {
        let (ge, ad) = (GE + i as u8, GAD + i as u8);
        op(
            b,
            format!("v_cmp_le_u32_e64 s{ge}, {d}, v{LANE}"),
            &[s(ge)],
            &[v(LANE)],
        )?;
        op(
            b,
            format!(
                "v_add_nc_u32_e32 v{VT0}, {}, v{LANE}",
                lit(0u32.wrapping_sub(d))
            ),
            &[v(VT0)],
            &[v(LANE)],
        )?;
        wait_alu(b, "depctr_va_sdst(0)")?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{VT0}, v{LANE}, v{VT0}, s{ge}"),
            &[v(VT0)],
            &[v(LANE), v(VT0), s(ge)],
        )?;
        vop(
            b,
            format!("v_lshlrev_b32_e32 v{ad}, 2, v{VT0}"),
            &[ad],
            &[VT0],
        )?;
    }
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{VT1}, 1, v{LANE}"),
        &[VT1],
        &[LANE],
    )?;
    op(
        b,
        format!("v_cmp_gt_u32_e64 s{S_M0}, s{S_ROWS}, v{VT1}"),
        &[s(S_M0)],
        &[s(S_ROWS), v(VT1)],
    )?;
    vop(b, format!("v_or_b32_e32 v{VT1}, 1, v{VT1}"), &[VT1], &[VT1])?;
    op(
        b,
        format!("v_cmp_gt_u32_e64 s{S_M1}, s{S_ROWS}, v{VT1}"),
        &[s(S_M1)],
        &[s(S_ROWS), v(VT1)],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{S_TMP}, s{S_C0}, 7"),
        &[S_TMP],
        &[S_C0],
    )?;
    sop(b, format!("s_mov_b32 s{S_H}, s{S_WAVE}"), &[S_H], &[S_WAVE])?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{VT0}, 8, v{LANE}"),
        &[VT0],
        &[LANE],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{VOFFG}, s{S_TMP}, v{VT0}"),
        &[v(VOFFG)],
        &[s(S_TMP), v(VT0)],
    )?;
    wait_alu(b, "depctr_va_sdst(0)")
}

/// `softplus(RA)` into `SPV`: `x > 20 ? x : (x < -20 ? e : log1p-graph)` with `e = expf(x)`.
fn softplus(b: &mut Builder) -> R {
    exp_many(
        b,
        &[Exp {
            dst: EX,
            x: RA,
            a: TA1,
            e: TE1,
            n: TN1,
            m: S_EM,
        }],
        false,
    )?;
    vop(b, format!("v_add_f32_e32 v{L1}, 1.0, v{EX}"), &[L1], &[EX])?;
    op(
        b,
        format!("v_cmp_gt_f32_e32 vcc_lo, 0x800000, v{L1}"),
        &[],
        &[v(L1)],
    )?;
    wait_alu(b, "depctr_va_vcc(0)")?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{SC1}, 0, 32, vcc_lo"),
        &[v(SC1)],
        &[],
    )?;
    vop(
        b,
        format!("v_ldexp_f32 v{L1}, v{L1}, v{SC1}"),
        &[L1],
        &[L1, SC1],
    )?;
    vop(b, format!("v_log_f32_e32 v{LG}, v{L1}"), &[LG], &[L1])?;
    vop(
        b,
        format!("v_mul_f32_e32 v{M1}, 0x3f317217, v{LG}"),
        &[M1],
        &[LG],
    )?;
    op(
        b,
        format!("v_cmp_gt_f32_e64 s{S_CM}, 0x7f800000, |v{LG}|"),
        &[s(S_CM)],
        &[v(LG)],
    )?;
    vop(
        b,
        format!("v_fma_f32 v{M2}, 0x3f317217, v{LG}, -v{M1}"),
        &[M2],
        &[LG, M1],
    )?;
    vop(
        b,
        format!("v_fmac_f32_e32 v{M2}, 0x3377d1cf, v{LG}"),
        &[M2],
        &[M2, LG],
    )?;
    vop(
        b,
        format!("v_add_f32_e32 v{M1}, v{M1}, v{M2}"),
        &[M1],
        &[M1, M2],
    )?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{LG}, v{LG}, v{M1}, s{S_CM}"),
        &[v(LG)],
        &[v(LG), v(M1), s(S_CM)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{M2}, 0, 0x41b17218, vcc_lo"),
        &[v(M2)],
        &[],
    )?;
    vop(
        b,
        format!("v_sub_f32_e32 v{LG}, v{LG}, v{M2}"),
        &[LG],
        &[LG, M2],
    )?;
    op(
        b,
        format!("v_cmp_ngt_f32_e64 s{S_CM}, 0xc1a00000, v{RA}"),
        &[s(S_CM)],
        &[v(RA)],
    )?;
    op(
        b,
        format!("v_cmp_nlt_f32_e64 s{S_EM}, 0x41a00000, v{RA}"),
        &[s(S_EM)],
        &[v(RA)],
    )?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{EX}, v{EX}, v{LG}, s{S_CM}"),
        &[v(EX)],
        &[v(EX), v(LG), s(S_CM)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{SPV}, v{RA}, v{EX}, s{S_EM}"),
        &[v(SPV)],
        &[v(RA), v(EX), s(S_EM)],
    )
}

/// One gate head (`h = S_H`) over this lane's two rows.
fn gate_body(b: &mut Builder) -> R {
    sop(
        b,
        format!("s_lshl_b32 s{S_HB4}, s{S_H}, 2"),
        &[S_HB4],
        &[S_H],
    )?;
    add64(b, PGAT, S_DT, S_HB4)?;
    add64(b, PGA2, S_ALOG, S_HB4)?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    smem(b, S_DTV, 1, PGAT, 0)?;
    smem(b, S_ALV, 1, PGA2, 0)?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{VADDR}, s{S_HB4}, v{VOFFG}"),
        &[v(VADDR)],
        &[s(S_HB4), v(VOFFG)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{EX}, s{S_ALV}"),
        &[v(EX)],
        &[s(S_ALV)],
    )?;
    exp_many(
        b,
        &[Exp {
            dst: EA,
            x: EX,
            a: TA1,
            e: TE1,
            n: TN1,
            m: S_EM,
        }],
        false,
    )?;
    for rr in 0..2u32 {
        let mk = if rr == 0 { S_M0 } else { S_M1 };
        let g = if rr == 0 { G0 } else { G1 };
        let imm = 128 * rr;
        exec_and(b, mk)?;
        gload(b, "b32", v(RA), VADDR, S_A, imm)?;
        gload(b, "b32", v(RB), VADDR, S_B, imm)?;
        exec_restore(b)?;
        // a + dt_bias[h]
        op(
            b,
            format!("v_add_f32_e32 v{RA}, s{S_DTV}, v{RA}"),
            &[v(RA)],
            &[s(S_DTV), v(RA)],
        )?;
        softplus(b)?;
        // beta = 1 / (1 + expf(-b))
        exp_many(
            b,
            &[Exp {
                dst: EX,
                x: RB,
                a: TE1,
                e: TN1,
                n: L1,
                m: S_EM,
            }],
            true,
        )?;
        vop(b, format!("v_add_f32_e32 v{EX}, 1.0, v{EX}"), &[EX], &[EX])?;
        fdiv_many(
            b,
            &[Div {
                dst: SIG,
                num: None,
                den: EX,
                sc: GDIV,
                r: GDIV + 1,
                q: GDIV + 2,
                e: GDIV + 3,
                m: S_CM,
            }],
        )?;
        exec_and(b, mk)?;
        gstore(b, "b32", VADDR, v(SIG), S_B, imm)?;
        exec_restore(b)?;
        b.release_store_sources()?;
        // g = softplus * (-expf(a_log)); rows past the chunk contribute +0.
        vop(
            b,
            format!("v_mul_f32_e64 v{g}, v{SPV}, -v{EA}"),
            &[g],
            &[SPV, EA],
        )?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{g}, 0, v{g}, s{mk}"),
            &[v(g)],
            &[v(g), s(mk)],
        )?;
    }
    vop(
        b,
        format!("v_add_f32_e32 v{PAIR}, v{G0}, v{G1}"),
        &[PAIR],
        &[G0, G1],
    )?;
    for i in 0..5u8 {
        let (ad, ge) = (GAD + i, GE + i);
        b.ds_crosslane(
            Instruction::new(
                format!("ds_bpermute_b32 v{OTH}, v{ad}, v{PAIR}"),
                vec![v(OTH)],
                vec![v(ad), v(PAIR)],
            )
            .memory(MemoryClass::DsLoad),
        )?;
        vop(
            b,
            format!("v_add_f32_e32 v{BEF}, v{PAIR}, v{OTH}"),
            &[BEF],
            &[PAIR, OTH],
        )?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{PAIR}, v{PAIR}, v{BEF}, s{ge}"),
            &[v(PAIR)],
            &[v(PAIR), v(BEF), s(ge)],
        )?;
    }
    vop(
        b,
        format!("v_sub_f32_e32 v{BEF}, v{PAIR}, v{G0}"),
        &[BEF],
        &[PAIR, G0],
    )?;
    vop(
        b,
        format!("v_sub_f32_e32 v{BEF}, v{BEF}, v{G1}"),
        &[BEF],
        &[BEF, G1],
    )?;
    vop(
        b,
        format!("v_add_f32_e32 v{GT}, v{G0}, v{BEF}"),
        &[GT],
        &[G0, BEF],
    )?;
    exec_and(b, S_M0)?;
    gstore(b, "b32", VADDR, v(GT), S_A, 0)?;
    exec_restore(b)?;
    b.release_store_sources()?;
    vop(
        b,
        format!("v_add_f32_e32 v{GT}, v{G1}, v{GT}"),
        &[GT],
        &[G1, GT],
    )?;
    exec_and(b, S_M1)?;
    gstore(b, "b32", VADDR, v(GT), S_A, 128)?;
    exec_restore(b)?;
    b.release_store_sources()?;
    sop(b, format!("s_add_co_i32 s{S_H}, s{S_H}, 8"), &[S_H], &[S_H])?;
    sop(b, format!("s_cmp_lt_u32 s{S_H}, {HV}"), &[], &[S_H])?;
    op(b, format!("s_cbranch_scc1 {GATE_LOOP}"), &[], &[])
}

/// Only `by == 0` blocks run the gates.
/// Only the `by == 0` blocks run the gates: any other block branches to `GATE_DONE`.
fn gate_guard(b: &mut Builder) -> R {
    op(b, "s_cmp_lg_u32 ttmp7, 0", &[], &[])?;
    op(b, format!("s_cbranch_scc1 {GATE_DONE}"), &[], &[])
}

fn gate(b: &mut Builder) -> R {
    gate_guard(b)?;
    gate_setup(b)?;
    b.wait_all()?;
    b.loop_(GATE_LOOP, gate_body)?;
    b.wait_all()?;
    b.label(GATE_DONE)
}

// ------------------------------------------------------------------ conv
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Region {
    Q,
    K,
    V,
}

const X_ROW_BYTES: u32 = 8192 * 4;

/// Unit pointers, per-lane offsets, the conv weights, row 0 and the three-row window.
fn conv_setup(b: &mut Builder) -> R {
    sop(
        b,
        format!("s_lshl_b32 s{S_TMP}, s{S_UNIT}, 11"),
        &[S_TMP],
        &[S_UNIT],
    )?;
    add64(b, PW, S_CW, S_TMP)?;
    sop(
        b,
        format!("s_mul_i32 s{S_TMP}, s{S_UNIT}, {}", lit(1536)),
        &[S_TMP],
        &[S_UNIT],
    )?;
    add64(b, PS, S_ST, S_TMP)?;
    sop(
        b,
        format!("s_lshl_b32 s{S_TMP}, s{S_UNIT}, 9"),
        &[S_TMP],
        &[S_UNIT],
    )?;
    add64(b, PXU, S_X, S_TMP)?;
    sop(
        b,
        format!("s_lshl_b32 s{S_TMP2}, s{S_C0}, 15"),
        &[S_TMP2],
        &[S_C0],
    )?;
    add64(b, PX, PXU, S_TMP2)?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{VOFFW}, 6, v{LANE}"),
        &[VOFFW],
        &[LANE],
    )?;
    vop(
        b,
        format!("v_mul_u32_u24_e32 v{VST}, 48, v{LANE}"),
        &[VST],
        &[LANE],
    )?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{VOFFX}, 4, v{LANE}"),
        &[VOFFX],
        &[LANE],
    )?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{VOFFO}, 3, v{LANE}"),
        &[VOFFO],
        &[LANE],
    )?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    for c in 0..4u8 {
        gload(b, "b128", vr(W + 4 * c, 4), VOFFW, PW, 16 * u32::from(c))?;
    }
    gload(b, "b128", vr(XB, 4), VOFFX, PX, 0)?;
    sop(b, format!("s_cmp_eq_u32 s{S_C0}, 0"), &[], &[S_C0])?;
    op(b, format!("s_cbranch_scc1 {STATE_WIN}"), &[], &[])?;
    // Later chunks: the three rows before the chunk are in x.
    sop(
        b,
        format!("s_add_co_i32 s{S_TMP}, s{S_C0}, -3"),
        &[S_TMP],
        &[S_C0],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{S_TMP}, s{S_TMP}, 15"),
        &[S_TMP],
        &[S_TMP],
    )?;
    add64(b, PN, PXU, S_TMP)?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    for m in 0..3u8 {
        gload(
            b,
            "b128",
            vr(WIN0 + 4 * m, 4),
            VOFFX,
            PN,
            X_ROW_BYTES * u32::from(m),
        )?;
    }
    b.wait_all()?;
    op(b, format!("s_branch {WIN_DONE}"), &[], &[])?;
    // Chunk 0: the window is the conv_state ring (win2 = ring[0], win1 = ring[1], win0 = ring[2]).
    b.label(STATE_WIN)?;
    for c in 0..4u8 {
        for j in 0..3u8 {
            gload(
                b,
                "b32",
                v(WIN2 - 4 * j + c),
                VST,
                PS,
                12 * u32::from(c) + 4 * u32::from(j),
            )?;
        }
    }
    b.wait_all()?;
    b.label(WIN_DONE)
}

/// Output pointer, stride, next-row pointer and the row counter of one region.
fn region_setup(b: &mut Builder, region: Region) -> R {
    let (base, stride, head_shift_sub) = match region {
        Region::Q => (S_Q, 4096u32, 0i32),
        Region::K => (S_K, 4096, -16),
        Region::V => (S_V, 8192, -32),
    };
    // PO = base + chunk0 * stride + head * 256
    sop(
        b,
        format!("s_mul_i32 s{S_TMP}, s{S_C0}, {}", lit(stride)),
        &[S_TMP],
        &[S_C0],
    )?;
    add64(b, PO, base, S_TMP)?;
    if head_shift_sub == 0 {
        sop(
            b,
            format!("s_mov_b32 s{S_TMP2}, s{S_UNIT}"),
            &[S_TMP2],
            &[S_UNIT],
        )?;
    } else {
        sop(
            b,
            format!(
                "s_add_co_i32 s{S_TMP2}, s{S_UNIT}, {}",
                lit(head_shift_sub as u32)
            ),
            &[S_TMP2],
            &[S_UNIT],
        )?;
    }
    sop(
        b,
        format!("s_lshl_b32 s{S_TMP2}, s{S_TMP2}, 8"),
        &[S_TMP2],
        &[S_TMP2],
    )?;
    add64(b, PO, PO, S_TMP2)?;
    sop(
        b,
        format!("s_mov_b32 s{S_OSTR}, {}", lit(stride)),
        &[S_OSTR],
        &[],
    )?;
    // PN: the row after the first (or the first again when the chunk has one row).
    sop(b, format!("s_cmp_gt_u32 s{S_ROWS}, 1"), &[], &[S_ROWS])?;
    sop(
        b,
        format!("s_cselect_b32 s{S_TMP}, {}, 0", lit(X_ROW_BYTES)),
        &[S_TMP],
        &[],
    )?;
    add64(b, PN, PX, S_TMP)?;
    sop(b, format!("s_mov_b32 s{S_I}, s{S_ROWS}"), &[S_I], &[S_ROWS])?;
    wait_alu(b, "depctr_sa_sdst(0)")
}

/// One row. Entry: `x_cur` holds the row, the ledger is empty. The next row is prefetched into `x_next` and moved
/// to `x_cur` at the end.
fn row_body(b: &mut Builder, region: Region, head: &str) -> R {
    gload(b, "b128", vr(YB, 4), VOFFX, PN, 0)?;
    sop(b, format!("s_cmp_gt_u32 s{S_I}, 2"), &[], &[S_I])?;
    sop(
        b,
        format!("s_cselect_b32 s{S_TMP}, {}, 0", lit(X_ROW_BYTES)),
        &[S_TMP],
        &[],
    )?;
    add64(b, PN, PN, S_TMP)?;
    // y = fma(w0, win0, fma(w1, win1, fma(w3, cur, w2 * win2)))
    for c in 0..4u8 {
        vop(
            b,
            format!(
                "v_mul_f32_e32 v{}, v{}, v{}",
                Y + c,
                W + 4 * c + 2,
                WIN2 + c
            ),
            &[Y + c],
            &[W + 4 * c + 2, WIN2 + c],
        )?;
    }
    for c in 0..4u8 {
        vop(
            b,
            format!(
                "v_fmac_f32_e32 v{0}, v{1}, v{2}",
                Y + c,
                W + 4 * c + 3,
                XB + c
            ),
            &[Y + c],
            &[Y + c, W + 4 * c + 3, XB + c],
        )?;
    }
    for c in 0..4u8 {
        vop(
            b,
            format!(
                "v_fmac_f32_e32 v{0}, v{1}, v{2}",
                Y + c,
                W + 4 * c + 1,
                WIN1 + c
            ),
            &[Y + c],
            &[Y + c, W + 4 * c + 1, WIN1 + c],
        )?;
    }
    for c in 0..4u8 {
        vop(
            b,
            format!(
                "v_fmac_f32_e32 v{0}, v{1}, v{2}",
                Y + c,
                W + 4 * c,
                WIN0 + c
            ),
            &[Y + c],
            &[Y + c, W + 4 * c, WIN0 + c],
        )?;
    }
    // Window shift (the current row becomes win2).
    for c in 0..4u8 {
        vop(
            b,
            format!("v_mov_b32_e32 v{}, v{}", WIN0 + c, WIN1 + c),
            &[WIN0 + c],
            &[WIN1 + c],
        )?;
    }
    for c in 0..4u8 {
        vop(
            b,
            format!("v_mov_b32_e32 v{}, v{}", WIN1 + c, WIN2 + c),
            &[WIN1 + c],
            &[WIN2 + c],
        )?;
    }
    for c in 0..4u8 {
        vop(
            b,
            format!("v_mov_b32_e32 v{}, v{}", WIN2 + c, XB + c),
            &[WIN2 + c],
            &[XB + c],
        )?;
    }
    // out = y / (1 + expf(-y))
    let exps: Vec<Exp> = (0..4u8)
        .map(|c| Exp {
            dst: TA + c,
            x: Y + c,
            a: TE + c,
            e: TN + c,
            n: DE + c,
            m: CM4 + c,
        })
        .collect();
    exp_many(b, &exps, true)?;
    for c in 0..4u8 {
        vop(
            b,
            format!("v_add_f32_e32 v{0}, 1.0, v{0}", TA + c),
            &[TA + c],
            &[TA + c],
        )?;
    }
    let divs: Vec<Div> = (0..4u8)
        .map(|c| Div {
            dst: OUT + c,
            num: Some(Y + c),
            den: TA + c,
            sc: DSC + c,
            r: DR + c,
            q: DQ + c,
            e: DE + c,
            m: DM4 + c,
        })
        .collect();
    fdiv_many(b, &divs)?;
    if region != Region::V {
        for c in 0..4u8 {
            vop(
                b,
                format!("v_mul_f32_e32 v{0}, v{1}, v{1}", SQ + c, OUT + c),
                &[SQ + c],
                &[OUT + c],
            )?;
        }
        vop(
            b,
            format!("v_add_f32_e32 v{NSUM}, v{}, v{}", SQ, SQ + 1),
            &[NSUM],
            &[SQ, SQ + 1],
        )?;
        vop(
            b,
            format!("v_add_f32_e32 v{NSUM}, v{NSUM}, v{}", SQ + 2),
            &[NSUM],
            &[NSUM, SQ + 2],
        )?;
        vop(
            b,
            format!("v_add_f32_e32 v{NSUM}, v{NSUM}, v{}", SQ + 3),
            &[NSUM],
            &[NSUM, SQ + 3],
        )?;
        for k in [16u32, 8, 4, 2, 1] {
            if k == 16 {
                op(
                    b,
                    format!("v_permlanex16_b32 v{NPEER}, v{NSUM}, s{S_PSEL}, 0xfedcba98"),
                    &[v(NPEER)],
                    &[v(NSUM), s(S_PSEL)],
                )?;
            } else {
                vop(
                    b,
                    format!(
                        "v_mov_b32_dpp v{NPEER}, v{NSUM} row_xmask:{k} row_mask:0xf bank_mask:0xf"
                    ),
                    &[NPEER],
                    &[NSUM],
                )?;
            }
            vop(
                b,
                format!("v_add_f32_e32 v{NSUM}, v{NPEER}, v{NSUM}"),
                &[NSUM],
                &[NPEER, NSUM],
            )?;
        }
        // inv = rsqrtf(sum + eps) with the vector denormal guard.
        op(
            b,
            format!("v_add_f32_e32 v{NSUM}, s{S_EPS}, v{NSUM}"),
            &[v(NSUM)],
            &[s(S_EPS), v(NSUM)],
        )?;
        vop(
            b,
            format!("v_mul_f32_e32 v{NT}, 0x4b800000, v{NSUM}"),
            &[NT],
            &[NSUM],
        )?;
        op(
            b,
            format!("v_cmp_gt_f32_e32 vcc_lo, 0x800000, v{NSUM}"),
            &[],
            &[v(NSUM)],
        )?;
        wait_alu(b, "depctr_va_vcc(0)")?;
        op(
            b,
            format!("v_cndmask_b32_e32 v{NSUM}, v{NSUM}, v{NT}, vcc_lo"),
            &[v(NSUM)],
            &[v(NSUM), v(NT)],
        )?;
        vop(
            b,
            format!("v_rsq_f32_e32 v{NINV}, v{NSUM}"),
            &[NINV],
            &[NSUM],
        )?;
        vop(
            b,
            format!("v_mul_f32_e32 v{NT}, 0x45800000, v{NINV}"),
            &[NT],
            &[NINV],
        )?;
        op(
            b,
            format!("v_cndmask_b32_e32 v{NINV}, v{NINV}, v{NT}, vcc_lo"),
            &[v(NINV)],
            &[v(NINV), v(NT)],
        )?;
        for c in 0..4u8 {
            vop(
                b,
                format!("v_mul_f32_e32 v{0}, v{1}, v{0}", OUT + c, NINV),
                &[OUT + c],
                &[OUT + c, NINV],
            )?;
        }
        if region == Region::Q {
            for c in 0..4u8 {
                op(
                    b,
                    format!("v_mul_f32_e32 v{0}, s{S_QS}, v{0}", OUT + c),
                    &[v(OUT + c)],
                    &[s(S_QS), v(OUT + c)],
                )?;
            }
        }
    }
    cvt_half(b, HP, false, OUT)?;
    cvt_half(b, HP, true, OUT + 1)?;
    cvt_half(b, HP + 1, false, OUT + 2)?;
    cvt_half(b, HP + 1, true, OUT + 3)?;
    gstore(b, "b64", VOFFO, vr(HP, 2), PO, 0)?;
    b.release_store_sources()?;
    add64(b, PO, PO, S_OSTR)?;
    sop(
        b,
        format!("s_add_co_i32 s{S_I}, s{S_I}, -1"),
        &[S_I],
        &[S_I],
    )?;
    for c in 0..4u8 {
        vop(
            b,
            format!("v_mov_b32_e32 v{}, v{}", XB + c, YB + c),
            &[XB + c],
            &[YB + c],
        )?;
    }
    sop(b, format!("s_cmp_lg_u32 s{S_I}, 0"), &[], &[S_I])?;
    op(b, format!("s_cbranch_scc1 {head}"), &[], &[])
}

/// The three region loops behind a uniform dispatch; each falls to (or jumps to) the epilogue.
/// Unit dispatch: units `>= 2 * HK` (v heads) branch to `V_START`.
fn conv_to_v(b: &mut Builder) -> R {
    sop(
        b,
        format!("s_cmp_ge_u32 s{S_UNIT}, {}", 2 * HK),
        &[],
        &[S_UNIT],
    )?;
    op(b, format!("s_cbranch_scc1 {V_START}"), &[], &[])
}

/// Unit dispatch: units `>= HK` (k heads) branch to `K_START`.
fn conv_to_k(b: &mut Builder) -> R {
    sop(b, format!("s_cmp_ge_u32 s{S_UNIT}, {HK}"), &[], &[S_UNIT])?;
    op(b, format!("s_cbranch_scc1 {K_START}"), &[], &[])
}

/// End of a region: `s_branch EPI`.
fn to_epi(b: &mut Builder) -> R {
    op(b, format!("s_branch {EPI}"), &[], &[])
}

fn conv(b: &mut Builder) -> R {
    conv_setup(b)?;
    conv_to_v(b)?;
    conv_to_k(b)?;
    region_setup(b, Region::Q)?;
    b.loop_(Q_LOOP, |b| row_body(b, Region::Q, Q_LOOP))?;
    to_epi(b)?;
    b.label(K_START)?;
    region_setup(b, Region::K)?;
    b.loop_(K_LOOP, |b| row_body(b, Region::K, K_LOOP))?;
    to_epi(b)?;
    b.label(V_START)?;
    region_setup(b, Region::V)?;
    b.loop_(V_LOOP, |b| row_body(b, Region::V, V_LOOP))
}

/// Chunk-0 blocks advance their channels' three-deep conv_state ring after consuming the old values.
fn epilogue(b: &mut Builder) -> R {
    b.label(EPI)?;
    sop(b, format!("s_cmp_lg_u32 s{S_C0}, 0"), &[], &[S_C0])?;
    op(b, format!("s_cbranch_scc1 {END}"), &[], &[])?;
    for (ptr, back) in [(P1, 1i32), (P2, 2), (P3, 3)] {
        sop(
            b,
            format!("s_add_co_i32 s{S_TMP}, s{S_T}, {}", -back),
            &[S_TMP],
            &[S_T],
        )?;
        if back > 1 {
            sop(
                b,
                format!("s_max_i32 s{S_TMP}, s{S_TMP}, 0"),
                &[S_TMP],
                &[S_TMP],
            )?;
        }
        sop(
            b,
            format!("s_lshl_b32 s{S_TMP}, s{S_TMP}, 15"),
            &[S_TMP],
            &[S_TMP],
        )?;
        add64(b, ptr, PXU, S_TMP)?;
    }
    sop(b, format!("s_cmp_ge_u32 s{S_T}, 2"), &[], &[S_T])?;
    sop(b, format!("s_cselect_b32 s{S_M2}, -1, 0"), &[S_M2], &[])?;
    sop(b, format!("s_cmp_ge_u32 s{S_T}, 3"), &[], &[S_T])?;
    sop(b, format!("s_cselect_b32 s{S_M3}, -1, 0"), &[S_M3], &[])?;
    sop(b, format!("s_cmp_eq_u32 s{S_T}, 2"), &[], &[S_T])?;
    sop(b, format!("s_cselect_b32 s{S_MEQ}, -1, 0"), &[S_MEQ], &[])?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    // Rows T-1 (xb), T-2 (yb), T-3 (y) and the old ring (the dead weight registers).
    gload(b, "b128", vr(XB, 4), VOFFX, P1, 0)?;
    gload(b, "b128", vr(YB, 4), VOFFX, P2, 0)?;
    gload(b, "b128", vr(Y, 4), VOFFX, P3, 0)?;
    for i in 0..3u8 {
        gload(b, "b128", vr(W + 4 * i, 4), VST, PS, 16 * u32::from(i))?;
    }
    // new ring of channel c at NEW[3c..3c+3] (= win0..win2 registers, contiguous).
    let new = WIN0;
    for c in 0..4u8 {
        let old0 = W + 3 * c;
        let old1 = W + 3 * c + 1;
        vop(
            b,
            format!("v_mov_b32_e32 v{}, v{}", new + 3 * c, XB + c),
            &[new + 3 * c],
            &[XB + c],
        )?;
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{}, v{old0}, v{}, s{S_M2}",
                new + 3 * c + 1,
                YB + c
            ),
            &[v(new + 3 * c + 1)],
            &[v(old0), v(YB + c), s(S_M2)],
        )?;
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{}, v{old1}, v{old0}, s{S_MEQ}",
                new + 3 * c + 2
            ),
            &[v(new + 3 * c + 2)],
            &[v(old1), v(old0), s(S_MEQ)],
        )?;
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{0}, v{0}, v{1}, s{S_M3}",
                new + 3 * c + 2,
                Y + c
            ),
            &[v(new + 3 * c + 2)],
            &[v(new + 3 * c + 2), v(Y + c), s(S_M3)],
        )?;
    }
    for i in 0..3u8 {
        gstore(b, "b128", VST, vr(new + 4 * i, 4), PS, 16 * u32::from(i))?;
    }
    Ok(())
}

/// The chunk-prep entry `gdn_chunk_prep_pm_gfx1201`.
pub fn emit(arch: Arch, spec: &GdnChunkPrepSpec) -> Result<Emitted, String> {
    let g = blocks::Gen::new(arch, spec)?;
    let mut b = Builder::new(g.kernel_spec(), g.plan()?);
    b.enable_delay_alu();
    blocks::prologue(&mut b)?;
    gate(&mut b)?;
    conv(&mut b)?;
    blocks::epilogue(&mut b)?;
    b.wait_all()?;
    b.label(END)?;
    b.control(Sop::End.encode(Arch::Gfx1201)?)?;
    b.finish()
}

/// The emit's phases as public blocks: [`emit`] with `gate` and `conv` (the sequences below) is the composition
/// the `.rip` twin (`gdn_chunk_prep.rip`) writes, statement for statement, over the `gdn::*` natives. Every block
/// emits straight into the builder; the two gate and three conv loops are the script's `b.loop_` statements
/// over `gate_body` / `row_body`.
pub mod blocks {
    use super::*;

    /// The one geometry's launch contract and register plan.
    pub struct Gen {
        arch: Arch,
    }

    impl Gen {
        pub fn new(arch: Arch, spec: &GdnChunkPrepSpec) -> Result<Gen, String> {
            admit(arch, spec)?;
            Ok(Gen { arch })
        }
        pub fn plan(&self) -> Result<RegPlan, String> {
            super::plan()
        }
        pub fn kernel_spec(&self) -> KernelSpec {
            let kernargs = KernargLayout::new(KERNARG_BYTES)
                .pointer_access("x", 0, Access::ReadOnly)
                .pointer_access("conv_w", 8, Access::ReadOnly)
                .pointer_access("conv_state", 16, Access::ReadWrite)
                .pointer_access("a", 24, Access::ReadWrite)
                .pointer_access("b", 32, Access::ReadWrite)
                .pointer_access("dt_bias", 40, Access::ReadOnly)
                .pointer_access("a_log", 48, Access::ReadOnly)
                .pointer_access("q", 56, Access::WriteOnly)
                .pointer_access("k", 64, Access::WriteOnly)
                .pointer_access("v", 72, Access::WriteOnly)
                .hidden("T", 80, 4, "by_value")
                .hidden("q_scale", 84, 4, "by_value")
                .hidden("eps", 88, 4, "by_value")
                .hidden("hk", 92, 4, "by_value")
                .hidden("hv", 96, 4, "by_value");
            KernelSpec {
                kernel_id: KERNEL_ID.into(),
                variant: "chunk_prep".into(),
                arch: self.arch,
                symbol: SYMBOL.into(),
                kernargs,
                user_sgpr_count: 2,
                system_sgpr_workgroup_id_y: true,
                workgroup_size: BLOCK as u16,
                group_segment_fixed_size: 0,
                wave32: true,
                cu_mode: false,
            }
        }
    }

    /// A label of the kernel: `end`, `gate_loop`, `gate_done`, `k_start`, `v_start`, `epi`, or a conv loop head
    /// `q_loop` / `k_loop` / `v_loop`.
    pub fn label(name: &str) -> Result<&'static str, String> {
        Ok(match name {
            "end" => END,
            "gate_loop" => GATE_LOOP,
            "gate_done" => GATE_DONE,
            "k_start" => K_START,
            "v_start" => V_START,
            "epi" => EPI,
            "q_loop" => Q_LOOP,
            "k_loop" => K_LOOP,
            "v_loop" => V_LOOP,
            o => return Err(format!("gdn label `{o}`: expected end, gate_loop, gate_done, k_start, v_start, epi, q_loop, k_loop or v_loop")),
        })
    }

    /// One of the three conv regions by name (`q`, `k`, `v`).
    pub fn region(name: &str) -> Result<Region, String> {
        match name {
            "q" => Ok(Region::Q),
            "k" => Ok(Region::K),
            "v" => Ok(Region::V),
            o => Err(format!("gdn region `{o}`: expected q, k or v")),
        }
    }

    /// Kernel arguments, ids, bounds exits and the launch constants.
    pub fn prologue(b: &mut Builder) -> R {
        super::prologue(b)
    }
    /// The gate unit's set-up (heads, parameters, first rows).
    pub fn gate_setup(b: &mut Builder) -> R {
        super::gate_setup(b)
    }
    /// One trip of the gate loop: two rows per lane, softplus, beta, the prefix; includes the back branch.
    pub fn gate_body(b: &mut Builder) -> R {
        super::gate_body(b)
    }
    /// Unit pointers, per-lane offsets, the conv weights, row 0 and the three-row window.
    pub fn conv_setup(b: &mut Builder) -> R {
        super::conv_setup(b)
    }
    /// A region's output pointers and constants.
    pub fn region_setup(b: &mut Builder, region: Region) -> R {
        super::region_setup(b, region)
    }
    /// One trip of a region's row loop (`head` is its loop label); includes the back branch.
    pub fn row_body(b: &mut Builder, region: Region, head: &str) -> R {
        super::row_body(b, region, head)
    }
    /// `s_cmp_lg_u32 ttmp7, 0; s_cbranch_scc1 gate_done`: only the `by == 0` blocks run the gates.
    pub fn gate_guard(b: &mut Builder) -> R {
        super::gate_guard(b)
    }
    /// Unit dispatch: v-head units branch to `v_start`.
    pub fn conv_to_v(b: &mut Builder) -> R {
        super::conv_to_v(b)
    }
    /// Unit dispatch: k-head units branch to `k_start`.
    pub fn conv_to_k(b: &mut Builder) -> R {
        super::conv_to_k(b)
    }
    /// `s_branch epi`.
    pub fn to_epi(b: &mut Builder) -> R {
        super::to_epi(b)
    }
    /// The chunk-0 blocks' conv_state ring advance (and the `epi` label).
    pub fn epilogue(b: &mut Builder) -> R {
        super::epilogue(b)
    }
}

/// The single export as one code object (`gdn_chunk_prep_pm_gfx1201`).
pub fn emit_module(
    arch: Arch,
    spec: &GdnChunkPrepSpec,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    admit(arch, spec)?;
    let emitted = vec![emit(arch, spec)?];
    let (text, proof) = super::iu4_gemm::module(&emitted, MODULE)?;
    Ok((emitted, text, proof))
}

// ------------------------------------------------------------------ contract
const COUNTED: [&str; 34] = [
    "global_load_b32",
    "global_load_b128",
    "global_store_b32",
    "global_store_b64",
    "global_store_b128",
    "s_load_b256",
    "s_load_b128",
    "s_load_b32",
    "ds_bpermute_b32",
    "v_exp_f32_e32",
    "v_log_f32_e32",
    "v_rcp_f32_e32",
    "v_rsq_f32_e32",
    "v_rndne_f32_e32",
    "v_ldexp_f32",
    "v_cvt_i32_f32_e32",
    "v_div_scale_f32",
    "v_div_fmas_f32",
    "v_div_fixup_f32",
    "v_fma_f32",
    "v_fmac_f32_e32",
    "v_mul_f32_e32",
    "v_mul_f32_e64",
    "v_add_f32_e32",
    "v_sub_f32_e32",
    "v_permlanex16_b32",
    "v_mov_b32_dpp",
    "v_cvt_f16_f32_e32",
    "v_cndmask_b32_e64",
    "v_cndmask_b32_e32",
    "v_cmp_gt_f32_e32",
    "s_cbranch_scc1",
    "s_branch",
    "s_endpgm",
];

fn counts(e: &Emitted, names: &[&str]) -> serde_json::Map<String, serde_json::Value> {
    let mut counts = serde_json::Map::new();
    for name in names {
        let n = e
            .s_text
            .lines()
            .filter(|l| l.trim().split_whitespace().next() == Some(*name))
            .count();
        counts.insert((*name).into(), n.into());
    }
    counts
}

/// Shape contract of the export (`toolchain::IsaShapeContract`): counts are derived from the emission. No LDS
/// allocation, barrier, WMMA or DS memory access; `ds_bpermute_b32` is the only DS instruction (lane exchange).
pub fn contract(arch: Arch, emitted: &Emitted) -> serde_json::Value {
    let _ = arch;
    serde_json::json!({
        "symbol": SYMBOL, "counts": counts(emitted, &COUNTED),
        "forbidden": ["scratch_*", "flat_*", "buffer_*", "global_atomic*", "ds_load*", "ds_store*", "ds_swizzle*",
                      "s_barrier*", "v_wmma*", "v_swmmac*", "v_writelane*", "s_setpc*", "s_swappc*", "s_call*",
                      "s_waitcnt*"],
        "vgpr_max": VGPRS, "sgpr_max": 104,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0,
        "group_segment_fixed_bytes": 0,
    })
}
