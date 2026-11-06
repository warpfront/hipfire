// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact gfx1201 (R9700, wave32) compressed-weight W8A16 GEMM: module
//! `hc8_wmma_pm_gfx1201`, entries `hc8_gemm_{e4m3|i8}_mw{MW}_kw{KW}_n{NT}_{set|act}`.
//!
//! Every E4M3 / int8 weight is exactly a BF16, so decoding it to BF16 and running
//! `v_wmma_f32_16x16x16_bf16` (f32 accumulate) is an exact-product W8A16 GEMM. The
//! activations `x` are f32 in memory but BF16-exact (hyper_norm / hc_act outputs), so
//! the B fragment is the high 16 bits of each f32. The per-output-row scale `s[m]` is
//! applied once, in the epilogue.
//!
//! # Weight layout (`swizzle_hc8`)
//! `q` is `[M, K]` 8-bit, pre-swizzled into WMMA A-fragment tiles. Tile `(mt, kt)` covers
//! 16 rows x 32 K = 512 B at byte offset `(mt * (K/32) + kt) * 512`. Inside a tile lane
//! `l = h*16 + r` (`r` = row, `h` = K half selector) owns the 16 bytes at `l*16`, read
//! with ONE `global_load_b128`. Byte `i` (`step = i>>3`, `j = i&7`):
//! `W[mt*16 + r][kt*32 + 16*step + 4*h + (j&3) + 8*(j>>2)]`.
//! Those 8 bytes of a step are exactly lane `l`'s A operand (4 VGPRs of packed bf16) of
//! the K16 step in the gfx1201 16-bit layout (`peacemaker-ir/src/emu/wmma.rs::f16_loc`):
//! `A[r,k] -> lane ((k>>2)&1)*16 + r, VGPR (k>>3)*2 + ((k>>1)&1), half k&1`. The B
//! operand `B[k,n]` uses the same map with `idx = n`; `D[row,col] -> lane (row>>3)*16 +
//! col, VGPR row&7`.
//!
//! # ABI
//! 48-byte kernarg: `q@0 s@8 x@16 y@24` (u64), `M@32 K@36 rows@40` (u32), `act_scale@44`
//! (f32, read only by `act`). `s` is f32 `[M]`, `x` f32 `[rows][K]`, `y` f32 `[rows][M]`.
//! Host guarantees `M % (16*MW) == 0`, `K % (32*KW) == 0`, `M*K < 2^32`. Grid x =
//! `M / (16*MW)`; block = `32*MW*KW`. `rows` outside `1..=16*NT` returns before any
//! barrier and any store.
//!
//! # Schedule / DAG (identical for every `rows`)
//! * A workgroup is `MW*KW` wave32 waves; wave `w` has `mi = w / KW`, `ki = w % KW`. It
//!   owns M-tile `wg*MW + mi` and the contiguous K-tile range
//!   `[ki*KT, (ki+1)*KT)`, `KT = K / (32*KW)`.
//! * Per wave: accumulators start at `+0`; for each K32 tile in ascending order, K16
//!   step 0 then step 1, `acc[nt] = wmma(A, B[nt], acc[nt])` for each column tile
//!   `nt < NT`. The weight fragment is decoded once per tile and shared by all column
//!   tiles. Columns `n >= rows` are never loaded (exec-masked loads into zeroed
//!   registers => `B = 0`) and never stored.
//! * The `KW` per-wave partials fold through LDS in ONE fixed order, independent of
//!   rows: `f = p0; f = f + p1; f = f + p2; ... + p[KW-1]` (sequential ascending in `ki`,
//!   `v_add_f32 f, f, p`). Partial `j` (accumulator VGPR index `nt*8 + (row&7)`) of
//!   wave `w` sits at LDS `((w*NR + j)*32 + lane)*4`, `NR = 8*NT`. The fold of
//!   accumulator index `j` is done by the wave with `ki == j % KW`. `KW == 1` has no LDS.
//! * Epilogue per output `(row m, col n)`: `v = f * s[m]` (one `v_mul_f32`); `set`:
//!   `y[n][m] = v`; `act`: `y[n][m] = hc_act(v, act_scale)` with the SAME DAG as the
//!   lane_rows HC entries (`lane_rows_bf16_gfx1201::hc_act_tail`: bf16 round trip,
//!   `act_scale * x`, round trip, `x / (1 + exp(-x))` with the full division sequence,
//!   round trip).
//! * Weight decode: E4M3 via `v_cvt_pk_f32_fp8_e64` (`op_sel:[0,0]` / `[1,0]`
//!   for the low / high byte pairs; raw weights occupy VGPRs above the e32 true16
//!   limit), then `v_perm_b32 0x07060302` packs the f32 high halves; int8 via
//!   `v_bfe_i32` + `v_cvt_f32_i32` + the same pack. Both are exact.
//! * K loop: batches of `U` tiles (`U = 8/4/2/2` for `NT = 1/2/3/4`), then halving
//!   batches `U/2 .. 1` for the remainder; all loads of a batch are issued before its
//!   first use so the builder's ledger waits only for what is needed. No scratch, no
//!   spills.
mod dequant;
use super::iu4_gemm::ModuleProof;
use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{add64, add64_imm, lit, op, s, s_add_i32, smem, sop, v, vr};
use crate::kernels::lane_rows_bf16_gfx1201::hc_act_tail;
use crate::lds::Transition;
use crate::plan::Access;
use crate::reg::{Kind, Live};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};
pub use dequant::{
    emit_dequant, emit_dequants, DequantOut, DEQUANT_BLOCK, DEQUANT_KERNARG_BYTES, DEQUANT_OUTS,
};

type R = Result<(), String>;

/// Module name; the embedded image is `kernels/hc8_wmma_pm_gfx1201.hxaco`.
pub const MODULE: &str = "hc8_wmma_pm_gfx1201";

/// Weight element format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fmt {
    E4M3,
    I8,
}

impl Fmt {
    fn tag(self) -> &'static str {
        match self {
            Fmt::E4M3 => "e4m3",
            Fmt::I8 => "i8",
        }
    }
}

/// One export: format, `(MW, KW)` workgroup geometry, column tiles, epilogue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geo {
    pub fmt: Fmt,
    pub mw: u32,
    pub kw: u32,
    pub nt: u32,
    pub act: bool,
}

/// The `(MW, KW)` geometries.
pub const GEOMETRIES: [(u32, u32); 5] = [(1, 8), (1, 16), (4, 1), (8, 1), (2, 4)];

/// Every export in module order: fmt x geometry x NT x epilogue (`set`, `act`).
pub fn all_geos() -> Vec<Geo> {
    let mut out = Vec::new();
    for fmt in [Fmt::E4M3, Fmt::I8] {
        for (mw, kw) in GEOMETRIES {
            for nt in 1..=4 {
                for act in [false, true] {
                    out.push(Geo {
                        fmt,
                        mw,
                        kw,
                        nt,
                        act,
                    });
                }
            }
        }
    }
    out
}

/// The first-priority subset: e4m3 `mw1kw16` and `mw4kw1`, NT 1, both epilogues.
pub fn priority_geos() -> Vec<Geo> {
    let mut out = Vec::new();
    for (mw, kw) in [(1, 16), (4, 1)] {
        for act in [false, true] {
            out.push(Geo {
                fmt: Fmt::E4M3,
                mw,
                kw,
                nt: 1,
                act,
            });
        }
    }
    out
}

/// Exported symbols of the full module, in module order.
pub fn symbols() -> Vec<String> {
    let mut out: Vec<String> = all_geos().iter().map(Geo::symbol).collect();
    out.extend(DEQUANT_OUTS.iter().map(|d| d.symbol().to_string()));
    out
}

impl Geo {
    pub fn symbol(&self) -> String {
        format!(
            "hc8_gemm_{}_mw{}_kw{}_n{}_{}",
            self.fmt.tag(),
            self.mw,
            self.kw,
            self.nt,
            if self.act { "act" } else { "set" }
        )
    }
    /// Workgroup threads: `32 * MW * KW`.
    pub fn block(&self) -> u32 {
        32 * self.mw * self.kw
    }
    /// Workgroups (x) for `m` output rows.
    pub fn grid(&self, m: u32) -> u32 {
        m / (16 * self.mw)
    }
    /// Most live activation rows.
    pub fn max_rows(&self) -> u32 {
        16 * self.nt
    }
    /// Accumulator VGPRs per lane, `8 * NT`.
    fn nr(&self) -> u32 {
        8 * self.nt
    }
    /// Static LDS bytes (partials of every wave; none for `KW == 1`).
    pub fn lds_bytes(&self) -> u32 {
        if self.kw == 1 {
            0
        } else {
            self.mw * self.kw * self.nr() * 128
        }
    }
    /// Tiles per batch of the main K loop.
    fn unroll(&self) -> u32 {
        match self.nt {
            1 => 8,
            2 => 4,
            _ => 2,
        }
    }
    fn batches(&self) -> Vec<u32> {
        let mut out = Vec::new();
        let mut u = self.unroll();
        while u >= 1 {
            out.push(u);
            u /= 2;
        }
        out
    }
    fn validate(&self) -> R {
        if !matches!(self.mw * self.kw, 4 | 8 | 16) || !self.kw.is_power_of_two() {
            return Err(format!("{}: MW*KW must be 4, 8 or 16", self.symbol()));
        }
        if !(1..=4).contains(&self.nt) {
            return Err(format!("{}: NT must be 1..=4", self.symbol()));
        }
        Ok(())
    }
}

// ------------------------------------------------------------ host swizzle
/// Swizzle a row-major `[m, k]` 8-bit matrix into the A-fragment tile order described
/// in the module doc. `m % 16 == 0` and `k % 32 == 0` (panics otherwise).
pub fn swizzle_hc8(w: &[u8], m: usize, k: usize) -> Vec<u8> {
    assert!(
        m % 16 == 0 && k % 32 == 0 && w.len() == m * k,
        "bad hc8 dims"
    );
    let mut out = vec![0u8; m * k];
    for_each_slot(m, k, |dst, src| out[dst] = w[src]);
    out
}

/// Inverse of [`swizzle_hc8`].
pub fn unswizzle_hc8(q: &[u8], m: usize, k: usize) -> Vec<u8> {
    assert!(
        m % 16 == 0 && k % 32 == 0 && q.len() == m * k,
        "bad hc8 dims"
    );
    let mut out = vec![0u8; m * k];
    for_each_slot(m, k, |dst, src| out[src] = q[dst]);
    out
}

/// Calls `f(swizzled_offset, rowmajor_offset)` for every element.
fn for_each_slot(m: usize, k: usize, mut f: impl FnMut(usize, usize)) {
    for mt in 0..m / 16 {
        for kt in 0..k / 32 {
            for l in 0..32usize {
                let (r, h) = (l & 15, l >> 4);
                for step in 0..2usize {
                    for j in 0..8usize {
                        let kk = 16 * step + 4 * h + (j & 3) + 8 * (j >> 2);
                        f(
                            (mt * (k / 32) + kt) * 512 + l * 16 + step * 8 + j,
                            (mt * 16 + r) * k + kt * 32 + kk,
                        );
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------- SGPRs
const KARG: u8 = 0;
/// Kernarg words 0..8: `q s x y` pointer pairs.
const KQ: u8 = 8;
const QP: u8 = 8;
const SP: u8 = 10;
const XQ: u8 = 12;
const YP: u8 = 14;
/// Kernarg words 8..12: `M K rows act_scale`.
const MK: u8 = 16;
const SM: u8 = 16;
const SK: u8 = 17;
const ROWS: u8 = 18;
const ASC: u8 = 19;
/// Weight pointer (advanced per tile) and X pointer.
const WP: u8 = 20;
const XP: u8 = 22;
const WG: u8 = 26;
const WAVE: u8 = 27;
const MI: u8 = 28;
const KI: u8 = 29;
const KT: u8 = 30;
const REM: u8 = 31;
const TMP: u8 = 32;
const TMP2: u8 = 33;
const MT: u8 = 34;
const KB: u8 = 35;
/// Live-lane masks of the column tiles.
const MASK: u8 = 36;
/// `nt*16*M*4` byte offsets of the column tiles' output rows.
const YN: u8 = 40;
const TMP3: u8 = 44;
const TMP4: u8 = 45;
const SGPR_BUDGET: u16 = 48;

// ----------------------------------------------------------------------- VGPRs
const TID: u8 = 0;
const LANE: u8 = 1;
const WOFF: u8 = 2;
/// X byte offsets per column tile (`3 + nt`) and output byte offsets (`7 + nt`).
const XOFF: u8 = 3;
const YA: u8 = 7;
/// LDS partial store / fold-read bases.
const LB: u8 = 11;
const LR: u8 = 12;
const VT: u8 = 13;
/// Scale load byte offset.
const SO: u8 = 14;
const VT2: u8 = 15;
/// Decode temporaries (two sets of four).
const DEC: u8 = 16;
const ACC: u8 = 24;

/// Resource layout of one geometry.
#[derive(Clone, Copy)]
struct Lay {
    u: u32,
    nt: u32,
    /// Double-buffered decoded A fragments (2 x 8 VGPRs).
    da: u8,
    /// Raw x registers (`16` per tile per column tile).
    xb: u8,
    /// Raw weight dwords (`4` per tile).
    wr: u8,
    top: u32,
}

impl Lay {
    fn new(g: &Geo) -> Result<Lay, String> {
        let (u, nt) = (g.unroll(), g.nt);
        let da = 24 + 8 * nt;
        let xb = da + 16;
        let wr = xb + 16 * u * nt;
        let top = wr + 4 * u;
        if top > 256 || wr > 255 {
            return Err(format!("{}: needs {top} VGPRs (> 256)", g.symbol()));
        }
        Ok(Lay {
            u,
            nt,
            da: da as u8,
            xb: xb as u8,
            wr: wr as u8,
            top,
        })
    }
    fn acc(&self, nt: u32) -> u8 {
        ACC + 8 * nt as u8
    }
    /// Raw x registers (8) of tile `t`, column tile `nt`, K16 step `st`.
    fn x(&self, t: u32, nt: u32, st: u32) -> u8 {
        self.xb + (((t * self.nt + nt) * 2 + st) * 8) as u8
    }
    fn w(&self, t: u32) -> u8 {
        self.wr + 4 * t as u8
    }
}

fn lbl(g: &Geo, name: &str) -> String {
    format!(".Lhc8_{}_{name}", g.symbol())
}

/// Declare `[base, base+len)` as the fewest aligned ranges of legal widths (1, 2, 4, 8).
/// No operand of this kernel spans two chunks: every multi-register operand (an 8-wide
/// accumulator, a 4-wide fragment, a 2-wide decode pair, an SGPR pair or quad) sits
/// inside one 8-aligned or naturally aligned chunk.
fn add_split(regs: &mut RegPlan, name: &str, kind: Kind, base: u8, len: u8) -> R {
    let (mut at, end) = (u32::from(base), u32::from(base) + u32::from(len));
    while at < end {
        let w = [8u32, 4, 2, 1]
            .into_iter()
            .find(|w| at % w == 0 && at + w <= end)
            .expect("width 1 always fits");
        regs.add_range(
            &format!("{name}@{at}"),
            kind,
            at as u8,
            w as u8,
            Live::Whole,
        )?;
        at += w;
    }
    Ok(())
}

fn plan(g: &Geo, l: &Lay) -> Result<RegPlan, String> {
    let mut regs = RegPlan::new(l.top as u16, SGPR_BUDGET)?;
    let nt = g.nt as u8;
    let sr_: Vec<(&str, u8, u8)> = vec![
        ("kernarg", KARG, 2),
        ("kernarg_ptrs", KQ, 8),
        ("kernarg_mkra", MK, 4),
        ("weight_ptr", WP, 2),
        ("x_ptr", XP, 2),
        ("workgroup", WG, 1),
        ("wave", WAVE, 1),
        ("m_tile_in_wg", MI, 1),
        ("k_slice", KI, 1),
        ("k_tiles", KT, 1),
        ("k_remaining", REM, 1),
        ("temp", TMP, 1),
        ("temp2", TMP2, 1),
        ("m_tile", MT, 1),
        ("k_bytes", KB, 1),
        ("col_masks", MASK, nt),
        ("y_col_offsets", YN, nt),
        ("temp3", TMP3, 1),
        ("temp4", TMP4, 1),
    ];
    for (name, base, len) in sr_ {
        add_split(&mut regs, name, Kind::S, base, len)?;
    }
    let mut vr_: Vec<(&str, u8, u8)> = vec![
        ("tid", TID, 1),
        ("lane", LANE, 1),
        ("weight_offset", WOFF, 1),
        ("x_offsets", XOFF, nt),
        ("y_offsets", YA, nt),
        ("lds_store", LB, 1),
        ("lds_read", LR, 1),
        ("temp", VT, 1),
        ("scale_offset", SO, 1),
        ("temp2", VT2, 1),
        ("decode_temps", DEC, 8),
        ("acc", ACC, 8 * nt),
        ("a_frags", l.da, 16),
    ];
    vr_.push(("x_raw", l.xb, (16 * l.u * g.nt) as u8));
    vr_.push(("w_raw", l.wr, (4 * l.u) as u8));
    for (name, base, len) in vr_ {
        add_split(&mut regs, name, Kind::V, base, len)?;
    }
    Ok(regs)
}

// ------------------------------------------------------------- instruction glue
fn branch(b: &mut Builder, text: &str) -> R {
    b.control(Instruction::new(text, vec![], vec![]))
}
fn exec_mask(b: &mut Builder, nt: u32) -> R {
    op(
        b,
        format!("s_mov_b32 exec_lo, s{}", MASK + nt as u8),
        &[],
        &[s(MASK + nt as u8)],
    )
}
fn exec_all(b: &mut Builder) -> R {
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])
}
fn vload128(b: &mut Builder, dst: u8, voff: u8, ptr: u8, imm: u32) -> R {
    let off = if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    };
    b.push(
        Instruction::new(
            format!(
                "global_load_b128 v[{dst}:{}], v{voff}, s[{ptr}:{}]{off}",
                dst + 3,
                ptr + 1
            ),
            (0..4).map(|i| v(dst + i)).collect(),
            vec![v(voff), s(ptr), s(ptr + 1)],
        )
        .memory(MemoryClass::VmemLoad),
    )
}
fn perm_pack(b: &mut Builder, dst: u8, hi: u8, lo: u8) -> R {
    // {hi16(hi) : hi16(lo)}; S0 = hi (selectors 4..7), S1 = lo (selectors 0..3).
    let mut uses = vec![v(hi), v(lo)];
    uses.dedup();
    op(
        b,
        format!("v_perm_b32 v{dst}, v{hi}, v{lo}, 0x07060302"),
        &[v(dst)],
        &uses,
    )
}

// -------------------------------------------------------------------- prologue
fn prologue(b: &mut Builder, g: &Geo, end: &str) -> R {
    let lg = g.kw.trailing_zeros();
    let nt = g.nt;
    smem(b, KQ, 8, KARG, 0)?;
    smem(b, MK, 4, KARG, 32)?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{VT}, 5, v{TID}"),
        &[v(VT)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{WAVE}, v{VT}"),
        &[s(WAVE)],
        &[v(VT)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{LANE}, 31, v{TID}"),
        &[v(LANE)],
        &[v(TID)],
    )?;
    sop(b, format!("s_mov_b32 s{WG}, ttmp9"), &[WG], &[])?;
    b.wait_all()?;
    // Out-of-range rows: return before any barrier or store.
    sop(b, format!("s_cmp_eq_u32 s{ROWS}, 0"), &[], &[ROWS])?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    sop(
        b,
        format!("s_cmp_gt_u32 s{ROWS}, {}", 16 * nt),
        &[],
        &[ROWS],
    )?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    // Wave roles.
    if lg == 0 {
        sop(b, format!("s_mov_b32 s{KI}, 0"), &[KI], &[])?;
        sop(b, format!("s_mov_b32 s{MI}, s{WAVE}"), &[MI], &[WAVE])?;
    } else {
        sop(
            b,
            format!("s_and_b32 s{KI}, s{WAVE}, {}", g.kw - 1),
            &[KI],
            &[WAVE],
        )?;
        sop(
            b,
            format!("s_lshr_b32 s{MI}, s{WAVE}, {lg}"),
            &[MI],
            &[WAVE],
        )?;
    }
    sop(
        b,
        format!("s_lshr_b32 s{KT}, s{SK}, {}", 5 + lg),
        &[KT],
        &[SK],
    )?;
    sop(b, format!("s_mov_b32 s{REM}, s{KT}"), &[REM], &[KT])?;
    sop(b, format!("s_lshl_b32 s{KB}, s{SK}, 2"), &[KB], &[SK])?;
    // M tile = wg*MW + mi.
    if g.mw == 1 {
        let a = b.spec.arch;
        sop(
            b,
            format!("{} s{MT}, s{WG}, s{MI}", s_add_i32(a)),
            &[MT],
            &[WG, MI],
        )?;
    } else {
        sop(b, format!("s_mul_i32 s{MT}, s{WG}, {}", g.mw), &[MT], &[WG])?;
        let a = b.spec.arch;
        sop(
            b,
            format!("{} s{MT}, s{MT}, s{MI}", s_add_i32(a)),
            &[MT],
            &[MT, MI],
        )?;
    }
    // Weight pointer: q + ((mt*(K/32) + ki*KT) * 512).
    sop(b, format!("s_lshr_b32 s{TMP}, s{SK}, 5"), &[TMP], &[SK])?;
    sop(
        b,
        format!("s_mul_i32 s{TMP2}, s{MT}, s{TMP}"),
        &[TMP2],
        &[MT, TMP],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{TMP3}, s{KI}, s{KT}"),
        &[TMP3],
        &[KI, KT],
    )?;
    let a = b.spec.arch;
    sop(
        b,
        format!("{} s{TMP2}, s{TMP2}, s{TMP3}", s_add_i32(a)),
        &[TMP2],
        &[TMP2, TMP3],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{TMP2}, s{TMP2}, 9"),
        &[TMP2],
        &[TMP2],
    )?;
    add64(b, WP, QP, TMP2)?;
    // X pointer: x + ki*KT*128.
    sop(
        b,
        format!("s_lshl_b32 s{TMP4}, s{TMP3}, 7"),
        &[TMP4],
        &[TMP3],
    )?;
    add64(b, XP, XQ, TMP4)?;
    // Column live-lane masks: ((1 << clamp(rows-16nt, 0, 16)) - 1) * 0x10001.
    for c in 0..nt {
        let m = MASK + c as u8;
        if c == 0 {
            sop(b, format!("s_mov_b32 s{TMP}, s{ROWS}"), &[TMP], &[ROWS])?;
        } else {
            sop(
                b,
                format!("s_sub_co_i32 s{TMP}, s{ROWS}, {}", 16 * c),
                &[TMP],
                &[ROWS],
            )?;
        }
        sop(b, format!("s_max_i32 s{TMP}, s{TMP}, 0"), &[TMP], &[TMP])?;
        sop(b, format!("s_min_u32 s{TMP}, s{TMP}, 16"), &[TMP], &[TMP])?;
        sop(b, format!("s_bfm_b32 s{m}, s{TMP}, 0"), &[m], &[TMP])?;
        sop(b, format!("s_lshl_b32 s{TMP}, s{m}, 16"), &[TMP], &[m])?;
        sop(b, format!("s_or_b32 s{m}, s{m}, s{TMP}"), &[m], &[m, TMP])?;
    }
    // Per-lane vectors. r = lane & 15 (v13), h = lane >> 4 (v14).
    op(
        b,
        format!("v_and_b32_e32 v{VT}, 15, v{LANE}"),
        &[v(VT)],
        &[v(LANE)],
    )?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{SO}, 4, v{LANE}"),
        &[v(SO)],
        &[v(LANE)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{WOFF}, 4, v{LANE}"),
        &[v(WOFF)],
        &[v(LANE)],
    )?;
    // x offset of column tile 0: r*K*4 + h*16.
    op(
        b,
        format!("v_mul_lo_u32 v{XOFF}, s{KB}, v{VT}"),
        &[v(XOFF)],
        &[v(VT), s(KB)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VT2}, 4, v{SO}"),
        &[v(VT2)],
        &[v(SO)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{XOFF}, v{XOFF}, v{VT2}"),
        &[v(XOFF)],
        &[v(XOFF), v(VT2)],
    )?;
    if nt > 1 {
        sop(b, format!("s_lshl_b32 s{TMP}, s{KB}, 4"), &[TMP], &[KB])?;
        for c in 1..nt {
            sop(
                b,
                format!("s_mul_i32 s{TMP2}, s{TMP}, {c}"),
                &[TMP2],
                &[TMP],
            )?;
            op(
                b,
                format!("v_add_nc_u32_e32 v{}, s{TMP2}, v{XOFF}", XOFF + c as u8),
                &[v(XOFF + c as u8)],
                &[v(XOFF), s(TMP2)],
            )?;
        }
    }
    // y offset of column tile 0: r*M*4 + (mt*16 + h*8)*4.
    sop(b, format!("s_lshl_b32 s{TMP}, s{SM}, 2"), &[TMP], &[SM])?;
    op(
        b,
        format!("v_mul_lo_u32 v{YA}, s{TMP}, v{VT}"),
        &[v(YA)],
        &[v(VT), s(TMP)],
    )?;
    sop(b, format!("s_lshl_b32 s{TMP2}, s{MT}, 6"), &[TMP2], &[MT])?;
    // h*32 (same offset indexes the scale vector `s`).
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VT2}, 1, v{VT2}"),
        &[v(VT2)],
        &[v(VT2)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{YA}, v{YA}, v{VT2}"),
        &[v(YA)],
        &[v(YA), v(VT2)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{YA}, s{TMP2}, v{YA}"),
        &[v(YA)],
        &[v(YA), s(TMP2)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{SO}, s{TMP2}, v{VT2}"),
        &[v(SO)],
        &[v(VT2), s(TMP2)],
    )?;
    for c in 1..nt {
        sop(
            b,
            format!("s_mul_i32 s{}, s{TMP}, {}", YN + c as u8, 16 * c),
            &[YN + c as u8],
            &[TMP],
        )?;
        op(
            b,
            format!(
                "v_add_nc_u32_e32 v{}, s{}, v{YA}",
                YA + c as u8,
                YN + c as u8
            ),
            &[v(YA + c as u8)],
            &[v(YA), s(YN + c as u8)],
        )?;
    }
    Ok(())
}

fn zero_regs(b: &mut Builder, base: u8, count: u32) -> R {
    for i in 0..count {
        op(
            b,
            format!("v_mov_b32_e32 v{}, 0", base as u32 + i),
            &[v(base + i as u8)],
            &[],
        )?;
    }
    Ok(())
}

// -------------------------------------------------------------------- K loop
/// Decode the raw weight dwords of tile `t` into the (double-buffered) A fragments.
fn decode(b: &mut Builder, g: &Geo, l: &Lay, t: u32) -> R {
    let da = l.da + 8 * (t % 2) as u8;
    for d in 0..4u8 {
        let w = l.w(t) + d;
        let tb = DEC + 4 * (d & 1);
        match g.fmt {
            Fmt::E4M3 => {
                op(
                    b,
                    format!("v_cvt_pk_f32_fp8_e64 {}, v{w}.l op_sel:[0,0]", vr(tb, 2)),
                    &[vr(tb, 2)],
                    &[v(w)],
                )?;
                op(
                    b,
                    format!(
                        "v_cvt_pk_f32_fp8_e64 {}, v{w}.h op_sel:[1,0]",
                        vr(tb + 2, 2)
                    ),
                    &[vr(tb + 2, 2)],
                    &[v(w)],
                )?;
            }
            Fmt::I8 => {
                for i in 0..4u8 {
                    op(
                        b,
                        format!("v_bfe_i32 v{}, v{w}, {}, 8", tb + i, 8 * i),
                        &[v(tb + i)],
                        &[v(w)],
                    )?;
                }
                for i in 0..4u8 {
                    op(
                        b,
                        format!("v_cvt_f32_i32_e32 v{}, v{}", tb + i, tb + i),
                        &[v(tb + i)],
                        &[v(tb + i)],
                    )?;
                }
            }
        }
        let dst = da + (d >> 1) * 4 + (d & 1) * 2;
        perm_pack(b, dst, tb + 1, tb)?;
        perm_pack(b, dst + 1, tb + 3, tb + 2)?;
    }
    Ok(())
}

/// Pack the raw f32 x of tile `t` / column tile `c` / step `st` into its B fragment
/// (VGPRs 0..3 of the 8-register block, in place; each destination is consumed first).
fn pack_b(b: &mut Builder, l: &Lay, t: u32, c: u32, st: u32) -> R {
    let x = l.x(t, c, st);
    for i in 0..4u8 {
        perm_pack(b, x + i, x + 2 * i + 1, x + 2 * i)?;
    }
    Ok(())
}

fn wmma(b: &mut Builder, l: &Lay, t: u32, c: u32, st: u32) -> R {
    let da = l.da + 8 * (t % 2) as u8 + 4 * st as u8;
    let x = l.x(t, c, st);
    let acc = l.acc(c);
    b.push(Instruction::new(
        format!(
            "v_wmma_f32_16x16x16_bf16 {}, {}, {}, {}",
            vr(acc, 8),
            vr(da, 4),
            vr(x, 4),
            vr(acc, 8)
        ),
        vec![vr(acc, 8)],
        vec![vr(da, 4), vr(x, 4), vr(acc, 8)],
    ))
}

/// One batch of `u` K32 tiles: issue every load, advance the pointers, then consume.
fn batch(b: &mut Builder, g: &Geo, l: &Lay, u: u32) -> R {
    for t in 0..u {
        vload128(b, l.w(t), WOFF, WP, 512 * t)?;
        for c in 0..g.nt {
            exec_mask(b, c)?;
            for st in 0..2u32 {
                for h in 0..2u32 {
                    vload128(
                        b,
                        l.x(t, c, st) + 4 * h as u8,
                        XOFF + c as u8,
                        XP,
                        128 * t + 64 * st + 32 * h,
                    )?;
                }
            }
        }
        exec_all(b)?;
    }
    add64_imm(b, WP, 512 * u)?;
    add64_imm(b, XP, 128 * u)?;
    for t in 0..u {
        decode(b, g, l, t)?;
        for c in 0..g.nt {
            for st in 0..2 {
                pack_b(b, l, t, c, st)?;
            }
        }
        for st in 0..2 {
            for c in 0..g.nt {
                wmma(b, l, t, c, st)?;
            }
        }
    }
    Ok(())
}

fn k_loops(b: &mut Builder, g: &Geo, l: &Lay) -> R {
    for u in g.batches() {
        let skip = lbl(g, &format!("skip{u}"));
        let head = lbl(g, &format!("loop{u}"));
        sop(b, format!("s_cmp_lt_u32 s{REM}, {u}"), &[], &[REM])?;
        branch(b, &format!("s_cbranch_scc1 {skip}"))?;
        b.loop_(&head, |b| {
            batch(b, g, l, u)?;
            sop(
                b,
                format!("s_sub_co_i32 s{REM}, s{REM}, {u}"),
                &[REM],
                &[REM],
            )?;
            sop(b, format!("s_cmp_ge_u32 s{REM}, {u}"), &[], &[REM])?;
            branch(b, &format!("s_cbranch_scc1 {head}"))
        })?;
        b.label(&skip)?;
    }
    Ok(())
}

// -------------------------------------------------------------------- epilogue
/// `y[col][m] = epi(val * s[m])` for accumulator index `j` (column tile `j/8`, row
/// `j%8`) of the value in `val`.
fn emit_store(b: &mut Builder, g: &Geo, l: &Lay, val: u8, j: u32) -> R {
    let (c, r) = (j / 8, (j % 8) as u8);
    let sv = l.xb;
    let vout = l.xb + 14;
    let t: [u8; 6] = std::array::from_fn(|i| l.xb + 8 + i as u8);
    op(
        b,
        format!("v_mul_f32_e32 v{vout}, v{val}, v{}", sv + r),
        &[v(vout)],
        &[v(val), v(sv + r)],
    )?;
    if g.act {
        hc_act_tail(b, vout, t, ASC)?;
    }
    exec_mask(b, c)?;
    b.push(
        Instruction::new(
            format!(
                "global_store_b32 v{}, v{vout}, s[{YP}:{}] offset:{}",
                YA + c as u8,
                YP + 1,
                4 * r
            ),
            vec![],
            vec![v(YA + c as u8), v(vout), s(YP), s(YP + 1)],
        )
        .memory(MemoryClass::VmemStore),
    )?;
    exec_all(b)?;
    b.release_store_sources()
}

fn epilogue(b: &mut Builder, g: &Geo, l: &Lay, slot: Option<usize>, end: &str) -> R {
    let nr = g.nr();
    // Scales of this lane's eight rows: s[mt*16 + h*8 .. +8].
    vload128(b, l.xb, SO, SP, 0)?;
    vload128(b, l.xb + 4, SO, SP, 16)?;
    let Some(slot) = slot else {
        for j in 0..nr {
            emit_store(b, g, l, ACC + j as u8, j)?;
        }
        return b.wait_all();
    };
    // Partials to LDS: ((wave*NR + j)*32 + lane)*4.
    op(
        b,
        format!("v_lshlrev_b32_e32 v{LB}, 2, v{LANE}"),
        &[v(LB)],
        &[v(LANE)],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{TMP}, s{WAVE}, {}", lit(nr * 128)),
        &[TMP],
        &[WAVE],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{LB}, s{TMP}, v{LB}"),
        &[v(LB)],
        &[v(LB), s(TMP)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{LR}, 2, v{LANE}"),
        &[v(LR)],
        &[v(LANE)],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{TMP}, s{MI}, {}", lit(g.kw * nr * 128)),
        &[TMP],
        &[MI],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{LR}, s{TMP}, v{LR}"),
        &[v(LR)],
        &[v(LR), s(TMP)],
    )?;
    for j in 0..nr {
        let a = ACC + j as u8;
        b.ds_store(
            slot,
            Instruction::new(
                format!("ds_store_b32 v{LB}, v{a} offset:{}", j * 128),
                vec![],
                vec![v(LB), v(a)],
            )
            .memory(MemoryClass::DsStore),
        )?;
    }
    b.wait_all()?;
    b.barrier(&[Transition::Ready(slot)])?;
    // Wave `ki` folds accumulator indices j with j % KW == ki, partials in ascending ki.
    let p = l.xb + 16;
    for ki in 0..g.kw {
        let js: Vec<u32> = (0..nr).filter(|j| j % g.kw == ki).collect();
        if js.is_empty() {
            continue;
        }
        let next = lbl(g, &format!("fold{ki}_next"));
        sop(b, format!("s_cmp_eq_u32 s{KI}, {ki}"), &[], &[KI])?;
        branch(b, &format!("s_cbranch_scc0 {next}"))?;
        for j in js {
            for k2 in 0..g.kw {
                b.ds_load(
                    slot,
                    Instruction::new(
                        format!(
                            "ds_load_b32 v{}, v{LR} offset:{}",
                            p + k2 as u8,
                            (k2 * nr + j) * 128
                        ),
                        vec![v(p + k2 as u8)],
                        vec![v(LR)],
                    )
                    .memory(MemoryClass::DsLoad),
                )?;
            }
            for k2 in 1..g.kw {
                op(
                    b,
                    format!("v_add_f32_e32 v{p}, v{p}, v{}", p + k2 as u8),
                    &[v(p)],
                    &[v(p), v(p + k2 as u8)],
                )?;
            }
            emit_store(b, g, l, p, j)?;
        }
        b.wait_all()?;
        branch(b, &format!("s_branch {end}"))?;
        b.label(&next)?;
    }
    b.wait_all()
}

fn emit_geo(g: &Geo) -> Result<Emitted, String> {
    g.validate()?;
    let l = Lay::new(g)?;
    let kernargs = KernargLayout::new(48)
        .pointer_access("q", 0, Access::ReadOnly)
        .pointer_access("s", 8, Access::ReadOnly)
        .pointer_access("x", 16, Access::ReadOnly)
        .pointer_access("y", 24, Access::WriteOnly)
        .hidden("M", 32, 4, "by_value")
        .hidden("K", 36, 4, "by_value")
        .hidden("rows", 40, 4, "by_value")
        .hidden("act_scale", 44, 4, "by_value");
    let spec = KernelSpec {
        kernel_id: "hc8_wmma".into(),
        variant: g.symbol().strip_prefix("hc8_gemm_").unwrap_or("").into(),
        arch: Arch::Gfx1201,
        symbol: g.symbol(),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: u16::try_from(g.block()).map_err(|e| e.to_string())?,
        group_segment_fixed_size: g.lds_bytes(),
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(spec, plan(g, &l)?);
    b.enable_delay_alu();
    let slot = if g.lds_bytes() > 0 {
        Some(b.lds_slot("partials", 0, g.lds_bytes())?)
    } else {
        None
    };
    let end = lbl(g, "end");
    prologue(&mut b, g, &end)?;
    zero_regs(&mut b, ACC, g.nr())?;
    zero_regs(&mut b, l.xb, 16 * l.u * g.nt)?;
    k_loops(&mut b, g, &l)?;
    epilogue(&mut b, g, &l, slot, &end)?;
    b.wait_all()?;
    b.label(&end)?;
    branch(&mut b, "s_endpgm")?;
    b.finish()
}

fn exported(e: &Emitted) -> Option<&str> {
    e.s_text
        .lines()
        .find_map(|l| l.trim().strip_prefix(".amdhsa_kernel "))
        .map(str::trim)
}

/// Link `emitted` (each already checked against `symbols`) into one module source and proof.
fn link(
    emitted: Vec<Emitted>,
    symbols: &[String],
) -> Result<(Vec<Emitted>, String, ModuleProof), String> {
    for (sym, e) in symbols.iter().zip(&emitted) {
        if exported(e) != Some(sym.as_str()) {
            return Err(format!(
                "{MODULE}: builder exported {:?}, expected {sym}",
                exported(e)
            ));
        }
        if e.proof.arch != Arch::Gfx1201 {
            return Err(format!("{MODULE}: {sym} not built for exact gfx1201"));
        }
    }
    let (text, proof) = super::iu4_gemm::module(&emitted, MODULE)?;
    Ok((emitted, text, proof))
}

fn emit_gemms(geos: &[Geo]) -> Result<Vec<Emitted>, String> {
    geos.iter()
        .map(|g| emit_geo(g).map_err(|e| format!("{MODULE}: {}: {e}", g.symbol())))
        .collect()
}

/// Build the given GEMM exports (no dequant entries) into one module source and proof.
pub fn emit_geos(geos: &[Geo]) -> Result<(Vec<Emitted>, String, ModuleProof), String> {
    let symbols: Vec<String> = geos.iter().map(Geo::symbol).collect();
    link(emit_gemms(geos)?, &symbols)
}

/// Build the full module: every export of [`all_geos`] then `hc8_dequant_{bf16,f16}`
/// (82 exports, [`symbols`] order).
pub fn emit_module() -> Result<(Vec<Emitted>, String, ModuleProof), String> {
    let mut emitted = emit_gemms(&all_geos())?;
    emitted.extend(emit_dequants()?);
    link(emitted, &symbols())
}
