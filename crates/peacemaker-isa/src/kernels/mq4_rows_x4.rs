// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Few-row (B = 1..8 activation rows) exact MQ4G256V2 GEMV over up to four
//! matrices per launch, emitted with the checked ISA builder for gfx1151 and
//! gfx1201: `gemv_mq4g256v2_rows_x4_pm_<arch>`, module `mq4_rows_x4_pm_<arch>`,
//! which also exports the fixed-eight-row grid-y twin from [`emit_rowtiles`].
//! The original narrow entry remains unchanged.
//!
//! ABI (96-byte kernarg segment): `A0@0 A1@8 A2@16 A3@24` (weights, read-only
//! pointers), `x@32` (read-only), `y0@40 y1@48 y2@56 y3@64` (write-only),
//! `M0@72 M1@76 M2@80 M3@84 K@88 B@92` (i32 by value). Unused matrices have
//! `M = 0`.
//!
//! Grid `[P0+P1+P2+P3, 1, 1]` with `Pi = ceil(Mi/2)`, block `[32,1,1]`, no LDS.
//! Workgroup `w` routes to the first matrix `i` with `w < P0+..+Pi` and the
//! local pair `p = w - (P0+..+P(i-1))`; its wave owns rows `2p` (even) and
//! `2p+1` (odd) of matrix `i`. An absent odd row (`2p+1 >= Mi`) clamps its
//! weight loads to the even row and is never stored. Workgroups past the sum
//! return.
//!
//! Domain: `B` in 1..=8 (runtime dispatch to a per-B section), `K % 256 == 0`,
//! `K > 0`; MQ4G256V2 rows of `K/256` groups of 136 bytes (header dwords hA
//! for K 0..127 and hB for K 128..255 of a group, then 128 bytes of nibbles,
//! element `e` at byte `8 + e/2`, low nibble even). `x` is `[B][K]` f32,
//! `y_i` is `[B][M_i]` f32 (`y_i[b*M_i + row]`). Every matrix, `x` and `y`
//! byte span is at most `u32::MAX` (32-bit buffer/global offsets).
//!
//! Rowtile contract: `gemv_mq4g256v2_rows_x4_t8_pm_<arch>` takes total
//! `R = 1..64` at offset 92 with the same other kernargs. Grid is
//! `[sum_i ceil(M_i/2), ceil(R/8), 1]`, block32, LDS0. Workgroup y owns
//! `[8*y, min(8*y+8,R))`; both X and the selected Y retain their full-window
//! bases/strides, with at most eight rows' accumulators. Invalid domains and
//! dead workgroups exit uniformly without X/weight loads or output stores.
//! gfx1100 uses the scalar singleton recipe for both outputs; gfx1151 uses it
//! for the forced-scalar LM matrix M248320/K2048. These scalar sections start
//! every dot with terms 1,0,2..7, keep the object's main/tail operand orientation,
//! and fold `(a0+a1)+(a3+a2)`. Other gfx1151/gfx1201 matrices use the paired
//! recipe below. Nondefault singleton-route flags require separate admission
//! and object parity; the emitter does not grant runtime qualification.
//!
//! Exactness: each output (matrix `i`, column `b`, row `r`) is bitwise the
//! 1-row decode GEMV's (`gemv_mq4g256v2` = `gemv_mq4g256v2_multirow_r2`) for
//! activation row `b`. Its exact DAG:
//!  * a wave (32 lanes) owns the row pair (row0 = 2p even, row1 = 2p+1 odd;
//!    row1 >= M clamps its loads to row0 and is never stored);
//!  * lane `l` covers elements `g*256 + 8l .. 8l+7` of group `g`; the header
//!    dword is hA if `l < 16` else hB; the packed dword is group byte `8 + 4l`;
//!  * `w_i = v_fma_mix_f32(sc_f16, f32(nibble_i), zp_f16)` (one f32 rounding),
//!    `nibble_i` = bits `4i..4i+3`;
//!  * row0 dot: `d = x0*w0` (`v_mul_f32`), then `d = fma(x1,w1,d)`, then fma
//!    terms 2..7 in order; row1 dot: `d = x1*w1`, then `d = fma(x0,w0,d)`,
//!    then fma terms 2..7;
//!  * stream `s = g % 4` (groups strictly ascending, tails included):
//!    `acc_row[s] = acc_row[s] + d`;
//!  * combine `(acc0+acc1)+(acc2+acc3)`, then the shfl_down 16/8/4/2/1 tree;
//!    lane 0 stores.
//!
//! Freedom used (scheduling and layout only): one group is loaded and
//! dequantized once and reused for all `B` columns (the dequant is shared
//! across columns but never between the two output rows); column tiles of `x`
//! stream through a two-deep register ring, with explicit per-column waits so
//! at most the current and the next tile are in flight; a runtime `B` dispatch
//! picks the unrolled 1..8-column section; the even and odd rows' dots share
//! VOPD packets (the even row `x0*w0 ... x7*w7`, the odd row `x1*w1, x0*w0,
//! x2*w2 ...`, each in its own order); no accumulator is shared between rows
//! or columns. On gfx1151 a VOPD packet may not take both halves' `src1` from
//! one VGPR bank, so terms 2..7 (where both rows read the same `x`) are
//! skewed by one slot between the halves (the even row's term `k` pairs with
//! the odd row's term `k-1`; the two unpaired terms are plain `v_fmac_f32`):
//! both rows still accumulate their terms in the DAG's order.
//!
//! Reference and divergence: gfx1151's reference is the hipcc
//! `gemv_mq4g256v2_multirow_r2` object (its DAG checked by disassembly);
//! gfx1201's is the PM multirow twin
//! (`pm_decode/gemv_hfq4g256_multirow_default_mq4v2.rs`) whose multi-column
//! form this builder extends. The gfx1201 body is that twin's instruction
//! sequence; the gfx1151 body differs only where gfx11 differs: the workgroup
//! id is the first system SGPR (`s2`), `s_add_i32`/`s_sub_i32`/`s_add_u32`
//! spell without `co`, no `s_add_nc_u64`, plain buffer/global loads (no
//! `scope:`), no `s_wait_alu`, `null` SMEM offsets, vmcnt/lgkmcnt waits, and
//! the weight loads retire with an explicit vmcnt (the builder's ledger does
//! not order buffer against global loads on gfx11).
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{add64, lit, op, s_add_i32, smem, sop, sr, SRD_WORD3};
use crate::ledger::Counter;
use crate::plan::Access;
use crate::reg::{Kind, Live, RegRef};
use crate::vopd::{Operand, VopdF32, VopdOp};

type R = Result<(), String>;

/// The kernarg segment size.
pub const KERNARG_BYTES: u32 = 96;
/// Most activation rows (columns of `x`) one launch takes.
pub const MAX_COLUMNS: u8 = 8;
/// Physical activation-row tile; independent of the logical window width.
pub const ROW_TILE: u8 = 8;
/// Maximum logical row count of the rowtile entry (not the narrow entry).
pub const MAX_ROWS: u8 = 64;
/// Weight rows one wave owns.
pub const ROWS_PER_WORKGROUP: u8 = 2;

/// The single entry, per arch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
}
impl Spec {
    pub fn symbol(&self) -> String {
        format!("gemv_mq4g256v2_rows_x4_pm_{}", self.arch.name())
    }
    pub fn module(arch: Arch) -> String {
        format!("mq4_rows_x4_pm_{}", arch.name())
    }
    pub fn validate(&self) -> R {
        if !matches!(self.arch, Arch::Gfx1151 | Arch::Gfx1201) {
            return Err("mq4_rows_x4 builder is exact-gfx1151 / gfx1201 only".into());
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------- SGPRs
// s0:1 kernarg pointer; the workgroup ids (gfx11) are the system SGPRs after the declared user
// SGPRs: s2/s3 on gfx1151, s14/s15 on gfx1100 (`FeatureUserSGPRInit16Bug`: user + system >= 16, so
// the kernarg pointer is padded to 14 user SGPRs); gfx12 reads them from ttmp9/ttmp7.
const KARG: u8 = 0;
const USER_SGPRS: u8 = 2;
/// Workgroup X and Y (rowtile entry) system SGPRs.
const SYSTEM_SGPRS: u8 = 2;
/// First workgroup-id SGPR: the declared user SGPR count of the arch.
const fn wgin(arch: Arch) -> u8 {
    arch.declared_user_sgprs(USER_SGPRS, SYSTEM_SGPRS)
}
/// First scalar of the `A0..A3` weight-pointer block: s[8:15] on gfx1151; the gfx1100 workgroup ids
/// occupy s14/s15, so the block moves to s[80:87].
const fn pa(arch: Arch) -> u8 {
    if matches!(arch, Arch::Gfx1100) {
        80
    } else {
        8
    }
}
/// Weight buffer resource (base of the selected matrix).
const SRD: u8 = 4;
/// Kernargs: s[pa:pa+7] A0..A3 (s[8:15], gfx1100 s[80:87]), s[16:23] x y0 y1 y2, s[24:27] y3 M0 M1, s[28:31] M2 M3 K B.
const PXY: u8 = 16;
const Y3M: u8 = 24;
const M2KB: u8 = 28;
const XP: u8 = 16;
const Y0: u8 = 18;
const Y1: u8 = 20;
const Y2: u8 = 22;
const Y3: u8 = 24;
const M0: u8 = 26;
const M1: u8 = 27;
const M2: u8 = 28;
const M3: u8 = 29;
const SK: u8 = 30;
const SB: u8 = 31;
// Routing: the selected y pointer, M and the first pair of its matrix; the
// cumulative pair counts P0, P0+P1, P0+..+P2, P0+..+P3.
const SY: u8 = 32;
const SM: u8 = 34;
const SBASE: u8 = 35;
const CP0: u8 = 36;
const CP1: u8 = 37;
const CP2: u8 = 38;
const CP3: u8 = 39;
// The reference body's scalars.
const SW: u8 = 40;
const ROW0: u8 = 41;
const ROW1: u8 = 42;
const GROUPS: u8 = 43;
const STRIDE: u8 = 44;
const TMP: u8 = 45;
const RB0: u8 = 46;
const RB1: u8 = 47;
const GOFF: u8 = 48;
const XG: u8 = 49;
const GI: u8 = 50;
const STREAM: u8 = 51;
const KB: u8 = 52;
const MB: u8 = 53;
const R0B: u8 = 54;
const STMP: u8 = 55;
// Only the rowtile entry reserves these otherwise-unused scalar slots.
const ROW_BASE: u8 = 56;
const TOTAL_ROWS: u8 = 57;
const SCALAR_RECIPE: u8 = 58;
const MAIN_GROUPS: u8 = 59;
/// Column `c`'s `x` base pointer pair.
const XCOL: u8 = 64;
fn xbase(c: u8) -> u8 {
    XCOL + 2 * c
}

// ----------------------------------------------------------------------- VGPRs
const W: [u8; 2] = [16, 26];
const DOT: [u8; 2] = [24, 25];
fn acc(c: u8, stream: u8, row: u8) -> u8 {
    40 + 8 * c + 2 * stream + row
}
// x occupies the bank between the two rows' weights (n, n+2).
fn ring(c: u8) -> u8 {
    105 + 8 * (c % 2)
}
fn red_tmp(c: u8, row: u8) -> u8 {
    16 + 2 * c + row
}

fn end_label(rowtiles: bool) -> &'static str {
    if rowtiles {
        ".Lmq4rxt8_end"
    } else {
        ".Lmq4rx_end"
    }
}
fn lbl(nb: u8, name: &str, rowtiles: bool) -> String {
    let prefix = if rowtiles { ".Lmq4rxt8" } else { ".Lmq4rx" };
    format!("{prefix}_b{nb}_{name}")
}

fn plan(arch: Arch, rowtiles: bool) -> Result<RegPlan, String> {
    let mut regs = RegPlan::new(128, if arch == Arch::Gfx1100 { 88 } else { 80 })?;
    let mut s_ranges: Vec<(&str, u8, u8)> = vec![
        ("kernarg", KARG, 2),
        ("weight_resource", SRD, 4),
        ("weight_pointers", pa(arch), 8),
        ("x_y_pointers", PXY, 8),
        ("y3_m01", Y3M, 4),
        ("m23_k_b", M2KB, 4),
        ("y_selected", SY, 2),
        ("m_selected", SM, 1),
        ("pair_base", SBASE, 1),
        ("pairs_0", CP0, 1),
        ("pairs_01", CP1, 1),
        ("pairs_012", CP2, 1),
        ("pairs_0123", CP3, 1),
        ("workgroup", SW, 1),
        ("row0", ROW0, 1),
        ("row1", ROW1, 1),
        ("groups", GROUPS, 1),
        ("stride", STRIDE, 1),
        ("temp", TMP, 1),
        ("row0_offset", RB0, 1),
        ("row1_offset", RB1, 1),
        ("group_offset", GOFF, 1),
        ("x_group_offset", XG, 1),
        ("group_index", GI, 1),
        ("stream", STREAM, 1),
        ("k_bytes", KB, 1),
        ("m_bytes", MB, 1),
        ("row0_bytes", R0B, 1),
        ("store_temp", STMP, 1),
        ("column_bases0", XCOL, 8),
        ("column_bases1", XCOL + 8, 8),
    ];
    if !arch.gfx12() {
        s_ranges.push(("workgroup_id", wgin(arch), 1));
        if rowtiles {
            s_ranges.push(("workgroup_id_y", wgin(arch) + 1, 1));
        }
    }
    if rowtiles {
        s_ranges.push(("activation_row_base", ROW_BASE, 1));
        s_ranges.push(("total_rows", TOTAL_ROWS, 1));
        s_ranges.push(("scalar_recipe", SCALAR_RECIPE, 1));
        s_ranges.push(("main_groups", MAIN_GROUPS, 1));
    }
    for (name, base, len) in s_ranges {
        regs.add_range(name, Kind::S, base, len, Live::Whole)?;
    }
    for (name, base, len) in [
        ("lane", 0, 1),
        ("header_offset", 1, 1),
        ("packed_offset", 2, 1),
        ("x_offset", 3, 1),
        ("shuffle_addr", 4, 1),
        ("header_packed", 8, 4),
        ("output_address", 12, 4),
        ("weights0", 16, 8),
        ("dots", 24, 2),
        ("weights1_lo", 26, 4),
        ("weights1_hi", 30, 4),
        ("acc0", 40, 8),
        ("acc1", 48, 8),
        ("acc2", 56, 8),
        ("acc3", 64, 8),
        ("acc4", 72, 8),
        ("acc5", 80, 8),
        ("acc6", 88, 8),
        ("acc7", 96, 8),
        ("x_ring0", 105, 8),
        ("x_ring1", 113, 8),
    ] {
        regs.add_range(name, Kind::V, base, len, Live::Whole)?;
    }
    Ok(regs)
}

/// The vector-load counter: gfx12 LOADcnt, gfx11 VMcnt.
fn load_counter(arch: Arch) -> Counter {
    if arch.gfx12() {
        Counter::Load
    } else {
        Counter::Vm
    }
}
/// `s_sub_i32` (gfx11) / `s_sub_co_i32` (gfx12).
fn s_sub_i32(arch: Arch) -> &'static str {
    if arch.gfx12() {
        "s_sub_co_i32"
    } else {
        "s_sub_i32"
    }
}

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    emit_inner(spec, false)
}

/// Emit the actual default singleton recipes over fixed-eight-row tiles.
/// gfx1100 uses its scalar recipe; gfx1151 selects that recipe for the forced
/// scalar LM shape. Other gfx1151/gfx1201 matrices use their paired recipe.
pub fn emit_rowtiles(spec: Spec) -> Result<Emitted, String> {
    emit_inner(spec, true)
}

fn emit_inner(spec: Spec, rowtiles: bool) -> Result<Emitted, String> {
    if !(rowtiles && spec.arch == Arch::Gfx1100) {
        spec.validate()?;
    }
    let arch = spec.arch;
    let kernargs = KernargLayout::new(KERNARG_BYTES)
        .pointer_access("A0", 0, Access::ReadOnly)
        .pointer_access("A1", 8, Access::ReadOnly)
        .pointer_access("A2", 16, Access::ReadOnly)
        .pointer_access("A3", 24, Access::ReadOnly)
        .pointer_access("x", 32, Access::ReadOnly)
        .pointer_access("y0", 40, Access::WriteOnly)
        .pointer_access("y1", 48, Access::WriteOnly)
        .pointer_access("y2", 56, Access::WriteOnly)
        .pointer_access("y3", 64, Access::WriteOnly)
        .hidden("M0", 72, 4, "by_value")
        .hidden("M1", 76, 4, "by_value")
        .hidden("M2", 80, 4, "by_value")
        .hidden("M3", 84, 4, "by_value")
        .hidden("K", 88, 4, "by_value")
        .hidden(if rowtiles { "R" } else { "B" }, 92, 4, "by_value");
    let kspec = KernelSpec {
        kernel_id: "mq4_rows_x4".into(),
        variant: if rowtiles {
            "exact_default_rowtiles_t8_x4"
        } else {
            "exact_columns_r2_b8_x4"
        }
        .into(),
        arch,
        symbol: if rowtiles {
            format!("gemv_mq4g256v2_rows_x4_t8_pm_{}", arch.name())
        } else {
            spec.symbol()
        },
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: rowtiles,
        workgroup_size: 32,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan(arch, rowtiles)?);
    b.enable_delay_alu();
    prologue(&mut b, rowtiles)?;
    if arch != Arch::Gfx1100 {
        for nb in 1..=MAX_COLUMNS {
            section(&mut b, nb, false)?;
        }
    }
    if rowtiles && !arch.gfx12() {
        for nb in 1..=MAX_COLUMNS {
            section(&mut b, nb, true)?;
        }
    }
    b.label(end_label(rowtiles))?;
    branch(&mut b, "s_endpgm")?;
    b.finish()
}

/// The production module: the unchanged narrow entry, then the rowtile entry.
/// gfx1100 has no legacy narrow entry, so its image contains only the rowtile.
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = if arch == Arch::Gfx1100 {
        vec![emit_rowtiles(Spec { arch })?]
    } else {
        vec![emit(Spec { arch })?, emit_rowtiles(Spec { arch })?]
    };
    let (text, proof) = super::iu4_gemm::module(&emitted, &Spec::module(arch))?;
    Ok((emitted, text, proof))
}

/// Kernarg loads, matrix routing, the selected matrix's resource and row
/// pair, the per-column `x` bases, and the runtime dispatch on `B`.
fn prologue(b: &mut Builder, rowtiles: bool) -> R {
    let a = b.spec.arch;
    let end = end_label(rowtiles);
    let pa0 = pa(a);
    let wg0 = wgin(a);
    smem(b, pa0, 8, KARG, 0)?;
    smem(b, PXY, 8, KARG, 32)?;
    smem(b, Y3M, 4, KARG, 64)?;
    smem(b, M2KB, 4, KARG, 80)?;
    if a.gfx12() {
        sop(b, format!("s_mov_b32 s{SW}, ttmp9"), &[SW], &[])?;
    } else {
        sop(b, format!("s_mov_b32 s{SW}, s{wg0}"), &[SW], &[wg0])?;
    }
    if rowtiles {
        rowtile_domain(b)?;
    }
    // Cumulative pair counts: Pi = ceil(Mi / 2).
    sop(
        b,
        format!("{} s{CP0}, s{M0}, 1", s_add_i32(a)),
        &[CP0],
        &[M0],
    )?;
    sop(b, format!("s_lshr_b32 s{CP0}, s{CP0}, 1"), &[CP0], &[CP0])?;
    for (dst, prev, m) in [(CP1, CP0, M1), (CP2, CP1, M2), (CP3, CP2, M3)] {
        sop(b, format!("{} s{TMP}, s{m}, 1", s_add_i32(a)), &[TMP], &[m])?;
        sop(b, format!("s_lshr_b32 s{TMP}, s{TMP}, 1"), &[TMP], &[TMP])?;
        sop(
            b,
            format!("{} s{dst}, s{prev}, s{TMP}", s_add_i32(a)),
            &[dst],
            &[prev, TMP],
        )?;
    }
    sop(b, format!("s_cmp_ge_u32 s{SW}, s{CP3}"), &[], &[SW, CP3])?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    // First matrix i with w < P0+..+Pi: matrix 2 or 3, then 1, then 0.
    let pair = |n: u8| sr(n, 2);
    sop(b, format!("s_cmp_lt_u32 s{SW}, s{CP2}"), &[], &[SW, CP2])?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{SRD}:{}], s[{}:{}], s[{}:{}]",
            SRD + 1,
            pa0 + 4,
            pa0 + 5,
            pa0 + 6,
            pa0 + 7
        ),
        &[pair(SRD)],
        &[pair(pa0 + 4), pair(pa0 + 6)],
    )?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{SY}:{}], s[{Y2}:{}], s[{Y3}:{}]",
            SY + 1,
            Y2 + 1,
            Y3 + 1
        ),
        &[pair(SY)],
        &[pair(Y2), pair(Y3)],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SM}, s{M2}, s{M3}"),
        &[SM],
        &[M2, M3],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SBASE}, s{CP1}, s{CP2}"),
        &[SBASE],
        &[CP1, CP2],
    )?;
    sop(b, format!("s_cmp_lt_u32 s{SW}, s{CP1}"), &[], &[SW, CP1])?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{SRD}:{}], s[{}:{}], s[{SRD}:{}]",
            SRD + 1,
            pa0 + 2,
            pa0 + 3,
            SRD + 1
        ),
        &[pair(SRD)],
        &[pair(pa0 + 2), pair(SRD)],
    )?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{SY}:{}], s[{Y1}:{}], s[{SY}:{}]",
            SY + 1,
            Y1 + 1,
            SY + 1
        ),
        &[pair(SY)],
        &[pair(Y1), pair(SY)],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SM}, s{M1}, s{SM}"),
        &[SM],
        &[M1, SM],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SBASE}, s{CP0}, s{SBASE}"),
        &[SBASE],
        &[CP0, SBASE],
    )?;
    sop(b, format!("s_cmp_lt_u32 s{SW}, s{CP0}"), &[], &[SW, CP0])?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{SRD}:{}], s[{pa0}:{}], s[{SRD}:{}]",
            SRD + 1,
            pa0 + 1,
            SRD + 1
        ),
        &[pair(SRD)],
        &[pair(pa0), pair(SRD)],
    )?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{SY}:{}], s[{Y0}:{}], s[{SY}:{}]",
            SY + 1,
            Y0 + 1,
            SY + 1
        ),
        &[pair(SY)],
        &[pair(Y0), pair(SY)],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SM}, s{M0}, s{SM}"),
        &[SM],
        &[M0, SM],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SBASE}, 0, s{SBASE}"),
        &[SBASE],
        &[SBASE],
    )?;
    // Rows 2p and 2p+1 (clamped to 2p past M) of the selected matrix; `GROUPS`
    // groups of 136 bytes per row.
    sop(
        b,
        format!("{} s{ROW0}, s{SW}, s{SBASE}", s_sub_i32(a)),
        &[ROW0],
        &[SW, SBASE],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{ROW0}, s{ROW0}, 1"),
        &[ROW0],
        &[ROW0],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{GROUPS}, s{SK}, 8"),
        &[GROUPS],
        &[SK],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{STRIDE}, s{GROUPS}, {}", lit(0x88)),
        &[STRIDE],
        &[GROUPS],
    )?;
    sop(
        b,
        format!("{} s{ROW1}, s{ROW0}, 1", s_add_i32(a)),
        &[ROW1],
        &[ROW0],
    )?;
    sop(b, format!("s_cmp_lt_i32 s{ROW1}, s{SM}"), &[], &[ROW1, SM])?;
    sop(
        b,
        format!("s_cselect_b32 s{ROW1}, s{ROW1}, s{ROW0}"),
        &[ROW1],
        &[ROW1, ROW0],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{RB0}, s{ROW0}, s{STRIDE}"),
        &[RB0],
        &[ROW0, STRIDE],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{RB1}, s{ROW1}, s{STRIDE}"),
        &[RB1],
        &[ROW1, STRIDE],
    )?;
    // Weight buffer resource over the selected matrix: base, no stride,
    // unbounded records, raw 32-bit buffer.
    sop(
        b,
        format!("s_and_b32 s{}, s{}, 0xffff", SRD + 1, SRD + 1),
        &[SRD + 1],
        &[SRD + 1],
    )?;
    sop(b, format!("s_mov_b32 s{}, -1", SRD + 2), &[SRD + 2], &[])?;
    sop(
        b,
        format!("s_mov_b32 s{}, {}", SRD + 3, lit(SRD_WORD3)),
        &[SRD + 3],
        &[],
    )?;
    sop(b, format!("s_lshl_b32 s{KB}, s{SK}, 2"), &[KB], &[SK])?;
    sop(b, format!("s_lshl_b32 s{MB}, s{SM}, 2"), &[MB], &[SM])?;
    sop(b, format!("s_lshl_b32 s{R0B}, s{ROW0}, 2"), &[R0B], &[ROW0])?;
    if rowtiles {
        // Scalar singleton: all outputs use the odd-start dot, a reversed
        // second fold pair, and the object's distinct tail operand order.
        sop(
            b,
            format!("s_and_b32 s{MAIN_GROUPS}, s{GROUPS}, -4"),
            &[MAIN_GROUPS],
            &[GROUPS],
        )?;
        if a == Arch::Gfx1100 {
            sop(
                b,
                format!("s_mov_b32 s{SCALAR_RECIPE}, 1"),
                &[SCALAR_RECIPE],
                &[],
            )?;
        } else if a == Arch::Gfx1151 {
            sop(b, format!("s_cmp_eq_u32 s{SM}, 0x3ca00"), &[], &[SM])?;
            sop(
                b,
                format!("s_cselect_b32 s{SCALAR_RECIPE}, 1, 0"),
                &[SCALAR_RECIPE],
                &[],
            )?;
            sop(b, format!("s_cmp_eq_u32 s{SK}, 0x800"), &[], &[SK])?;
            sop(
                b,
                format!("s_cselect_b32 s{SCALAR_RECIPE}, s{SCALAR_RECIPE}, 0"),
                &[SCALAR_RECIPE],
                &[SCALAR_RECIPE],
            )?;
        }
        // Rebase once by the absolute token row, not by the local tail count.
        // The narrow body's c*K and c*M offsets remain unchanged.
        sop(
            b,
            format!("s_mul_i32 s{STMP}, s{ROW_BASE}, s{KB}"),
            &[STMP],
            &[ROW_BASE, KB],
        )?;
        add64(b, XP, XP, STMP)?;
        sop(
            b,
            format!("s_mul_i32 s{STMP}, s{ROW_BASE}, s{MB}"),
            &[STMP],
            &[ROW_BASE, MB],
        )?;
        add64(b, SY, SY, STMP)?;
    }
    for c in 0..MAX_COLUMNS {
        let p = xbase(c);
        sop(b, format!("s_mul_i32 s{p}, s{KB}, {c}"), &[p], &[KB])?;
        if a.gfx12() {
            sop(b, format!("s_mov_b32 s{}, 0", p + 1), &[p + 1], &[])?;
            op(
                b,
                format!(
                    "s_add_nc_u64 s[{p}:{}], s[{XP}:{}], s[{p}:{}]",
                    p + 1,
                    XP + 1,
                    p + 1
                ),
                &[sr(p, 2)],
                &[sr(XP, 2), sr(p, 2)],
            )?;
        } else {
            add64(b, p, XP, p)?;
        }
    }
    b.wait_all()?;
    if rowtiles && !a.gfx12() {
        sop(
            b,
            format!("s_cmp_lg_u32 s{SCALAR_RECIPE}, 0"),
            &[],
            &[SCALAR_RECIPE],
        )?;
        branch(b, "s_cbranch_scc1 .Lmq4rxt8_scalar_dispatch")?;
    }
    if a != Arch::Gfx1100 {
        for nb in 1..=MAX_COLUMNS {
            sop(b, format!("s_cmp_eq_u32 s{SB}, {nb}"), &[], &[SB])?;
            branch(b, &format!("s_cbranch_scc1 {}", lbl(nb, "start", rowtiles)))?;
        }
        branch(b, &format!("s_branch {end}"))?;
    }
    if rowtiles && !a.gfx12() {
        b.label(".Lmq4rxt8_scalar_dispatch")?;
        for nb in 1..=MAX_COLUMNS {
            sop(b, format!("s_cmp_eq_u32 s{SB}, {nb}"), &[], &[SB])?;
            branch(
                b,
                &format!("s_cbranch_scc1 {}", lbl(nb + MAX_COLUMNS, "start", true)),
            )?;
        }
        branch(b, &format!("s_branch {end}"))?;
    }
    Ok(())
}

/// Uniform validation and local row count, before any data-memory access.
/// Grid-y is read with the architecture's actual system-register convention.
fn rowtile_domain(b: &mut Builder) -> R {
    let a = b.spec.arch;
    let end = end_label(true);
    sop(
        b,
        format!("s_mov_b32 s{TOTAL_ROWS}, s{SB}"),
        &[TOTAL_ROWS],
        &[SB],
    )?;
    sop(
        b,
        format!("s_cmp_eq_u32 s{TOTAL_ROWS}, 0"),
        &[],
        &[TOTAL_ROWS],
    )?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    sop(
        b,
        format!("s_cmp_gt_u32 s{TOTAL_ROWS}, {MAX_ROWS}"),
        &[],
        &[TOTAL_ROWS],
    )?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    sop(b, format!("s_cmp_le_i32 s{SK}, 0"), &[], &[SK])?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    sop(
        b,
        format!("s_and_b32 s{TMP}, s{SK}, {}", lit(255)),
        &[TMP],
        &[SK],
    )?;
    sop(b, format!("s_cmp_lg_u32 s{TMP}, 0"), &[], &[TMP])?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    for m in [M0, M1, M2, M3] {
        sop(b, format!("s_cmp_lt_i32 s{m}, 0"), &[], &[m])?;
        branch(b, &format!("s_cbranch_scc1 {end}"))?;
    }
    if a.gfx12() {
        sop(
            b,
            format!("s_and_b32 s{ROW_BASE}, ttmp7, 0xffff"),
            &[ROW_BASE],
            &[],
        )?;
    } else {
        sop(
            b,
            format!("s_mov_b32 s{ROW_BASE}, s{}", wgin(a) + 1),
            &[ROW_BASE],
            &[wgin(a) + 1],
        )?;
    }
    // Reject before shifting, so an excess grid cannot wrap onto a live tile.
    sop(b, format!("s_cmp_ge_u32 s{ROW_BASE}, 8"), &[], &[ROW_BASE])?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    sop(
        b,
        format!("s_lshl_b32 s{ROW_BASE}, s{ROW_BASE}, 3"),
        &[ROW_BASE],
        &[ROW_BASE],
    )?;
    sop(
        b,
        format!("s_cmp_ge_u32 s{ROW_BASE}, s{TOTAL_ROWS}"),
        &[],
        &[ROW_BASE, TOTAL_ROWS],
    )?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    sop(
        b,
        format!("{} s{SB}, s{TOTAL_ROWS}, s{ROW_BASE}", s_sub_i32(a)),
        &[SB],
        &[TOTAL_ROWS, ROW_BASE],
    )?;
    sop(b, format!("s_cmp_gt_u32 s{SB}, {ROW_TILE}"), &[], &[SB])?;
    sop(
        b,
        format!("s_cselect_b32 s{SB}, {ROW_TILE}, s{SB}"),
        &[SB],
        &[SB],
    )
}

fn section(b: &mut Builder, nb: u8, scalar: bool) -> R {
    let a = b.spec.arch;
    let rowtiles = b.spec.system_sgpr_workgroup_id_y;
    let label_nb = if scalar { nb + MAX_COLUMNS } else { nb };
    b.label(&lbl(label_nb, "start", rowtiles))?;
    for c in 0..nb {
        for s in 0..4 {
            for r in 0..2 {
                let n = acc(c, s, r);
                valu(b, &format!("v_mov_b32_e32 v{n}, 0"), &[n], &[], &[])?;
            }
        }
    }
    sop(b, format!("s_mov_b32 s{GI}, 0"), &[GI], &[])?;
    sop(b, format!("s_cmp_eq_u32 s{GROUPS}, 0"), &[], &[GROUPS])?;
    branch(
        b,
        &format!("s_cbranch_scc1 {}", lbl(label_nb, "fold", rowtiles)),
    )?;
    b.loop_(&lbl(label_nb, "group", rowtiles), |b| {
        sop(
            b,
            format!("s_mul_i32 s{GOFF}, s{GI}, {}", lit(0x88)),
            &[GOFF],
            &[GI],
        )?;
        sop(b, format!("s_lshl_b32 s{XG}, s{GI}, 10"), &[XG], &[GI])?;
        weight_loads(b)?;
        valu(b, "v_lshlrev_b32_e32 v3, 5, v0", &[3], &[0], &[])?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v3, s{XG}, v3"),
            &[3],
            &[3],
            &[XG],
        )?;
        x_load(b, 0)?;
        if nb > 1 {
            x_load(b, 1)?;
        }
        // gfx11's ledger orders buffer against global loads as unrelated, so
        // it would drain every x tile with the weights: retire the four
        // weight loads (oldest) explicitly, leaving the x tiles in flight.
        if !a.gfx12() {
            b.wait(load_counter(a), 2 * nb.min(2))?;
        }
        dequant(b)?;
        sop(
            b,
            format!("s_and_b32 s{STREAM}, s{GI}, 3"),
            &[STREAM],
            &[GI],
        )?;
        for s in 0..3 {
            sop(b, format!("s_cmp_eq_u32 s{STREAM}, {s}"), &[], &[STREAM])?;
            branch(
                b,
                &format!(
                    "s_cbranch_scc1 {}",
                    lbl(label_nb, &format!("s{s}"), rowtiles)
                ),
            )?;
        }
        branch(b, &format!("s_branch {}", lbl(label_nb, "s3", rowtiles)))?;
        for s in 0..4 {
            b.label(&lbl(label_nb, &format!("s{s}"), rowtiles))?;
            if scalar {
                let tail = lbl(label_nb, &format!("s{s}_tail"), rowtiles);
                let done = lbl(label_nb, &format!("s{s}_dot_done"), rowtiles);
                sop(
                    b,
                    format!("s_cmp_ge_u32 s{GI}, s{MAIN_GROUPS}"),
                    &[],
                    &[GI, MAIN_GROUPS],
                )?;
                branch(b, &format!("s_cbranch_scc1 {tail}"))?;
                columns(b, nb, s, Some(false))?;
                b.wait_all()?;
                branch(b, &format!("s_branch {done}"))?;
                b.label(&tail)?;
                columns(b, nb, s, Some(true))?;
                b.label(&done)?;
            } else {
                columns(b, nb, s, None)?;
            }
            b.wait_all()?;
            if s < 3 {
                branch(
                    b,
                    &format!("s_branch {}", lbl(label_nb, "group_done", rowtiles)),
                )?;
            }
        }
        b.label(&lbl(label_nb, "group_done", rowtiles))?;
        sop(b, format!("{} s{GI}, s{GI}, 1", s_add_i32(a)), &[GI], &[GI])?;
        sop(
            b,
            format!("s_cmp_lt_u32 s{GI}, s{GROUPS}"),
            &[],
            &[GI, GROUPS],
        )?;
        branch(
            b,
            &format!("s_cbranch_scc1 {}", lbl(label_nb, "group", rowtiles)),
        )
    })?;
    b.label(&lbl(label_nb, "fold", rowtiles))?;
    for c in 0..nb {
        let fold = if scalar {
            [(0, 1), (3, 2), (0, 3)]
        } else {
            [(0, 1), (2, 3), (0, 2)]
        };
        for (lhs, rhs) in fold {
            b.vopd(
                VopdOp {
                    op: VopdF32::Add,
                    dst: acc(c, lhs, 0),
                    src0: Operand::V(acc(c, lhs, 0)),
                    src1: acc(c, rhs, 0),
                },
                VopdOp {
                    op: VopdF32::Add,
                    dst: acc(c, lhs, 1),
                    src0: Operand::V(acc(c, lhs, 1)),
                    src1: acc(c, rhs, 1),
                },
            )?;
        }
    }
    reduce(b, nb)?;
    store(b, nb, label_nb)
}

fn weight_loads(b: &mut Builder) -> R {
    let a = b.spec.arch;
    // gfx12 buffer loads carry the device-scope cache policy; gfx11 spells
    // plain loads (the hipcc singleton's own).
    let policy = if a.gfx12() { " scope:SCOPE_DEV" } else { "" };
    valu(b, "v_lshrrev_b32_e32 v1, 4, v0", &[1], &[0], &[])?;
    valu(b, "v_lshlrev_b32_e32 v1, 2, v1", &[1], &[1], &[])?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v1, s{GOFF}, v1"),
        &[1],
        &[1],
        &[GOFF],
    )?;
    valu(b, "v_lshlrev_b32_e32 v2, 2, v0", &[2], &[0], &[])?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v2, s{GOFF}, v2"),
        &[2],
        &[2],
        &[GOFF],
    )?;
    let res: Vec<u8> = (SRD..SRD + 4).collect();
    for r in 0..2u8 {
        let soff = [RB0, RB1][r as usize];
        let mut scalar = res.clone();
        scalar.push(soff);
        vmem(
            b,
            &format!(
                "buffer_load_b32 v{}, v1, s[{SRD}:{}], s{soff} offen{policy}",
                8 + r,
                SRD + 3
            ),
            &[8 + r],
            &[1],
            &scalar,
            false,
        )?;
        vmem(
            b,
            &format!(
                "buffer_load_b32 v{}, v2, s[{SRD}:{}], s{soff} offen offset:8{policy}",
                10 + r,
                SRD + 3
            ),
            &[10 + r],
            &[2],
            &scalar,
            false,
        )?;
    }
    Ok(())
}

fn x_load(b: &mut Builder, c: u8) -> R {
    let x = ring(c);
    let p = xbase(c);
    for (j, suffix) in [(0, ""), (4, " offset:16")] {
        vmem(
            b,
            &format!(
                "global_load_b128 v[{}:{}], v3, s[{p}:{}]{suffix}",
                x + j,
                x + j + 3,
                p + 1
            ),
            &[x + j, x + j + 1, x + j + 2, x + j + 3],
            &[3],
            &[p, p + 1],
            false,
        )?;
    }
    Ok(())
}

fn dequant(b: &mut Builder) -> R {
    // The nibble integers and f32 conversion are exact; dequant is shared
    // across every B column, but never shared between the two output rows.
    for n in 0..8 {
        for r in 0..2u8 {
            let w = W[r as usize] + n;
            valu(
                b,
                &format!("v_bfe_u32 v{w}, v{}, {}, 4", 10 + r, n * 4),
                &[w],
                &[10 + r],
                &[],
            )?;
            valu(
                b,
                &format!("v_cvt_f32_ubyte0_e32 v{w}, v{w}"),
                &[w],
                &[w],
                &[],
            )?;
            valu(
                b,
                &format!(
                    "v_fma_mix_f32 v{w}, v{h}, v{w}, v{h} op_sel:[0,0,1] op_sel_hi:[1,0,1]",
                    h = 8 + r
                ),
                &[w],
                &[8 + r, w],
                &[],
            )?;
        }
    }
    Ok(())
}

/// One column's two dot chains into `DOT[0]` (row 0: x0*w0, then fma terms
/// 1..7) and `DOT[1]` (row 1: x1*w1, x0*w0, then fma terms 2..7).
fn dots(b: &mut Builder, x: u8) -> R {
    let a = b.spec.arch;
    let term = |op: VopdF32, row: usize, n: u8| VopdOp {
        op,
        dst: DOT[row],
        src0: Operand::V(W[row] + n),
        src1: x + n,
    };
    let kind = |t: u8| if t == 0 { VopdF32::Mul } else { VopdF32::Fmac };
    // Row 1 reads x1, x0, x2, ..., x7.
    let n1 = |t: u8| match t {
        0 => 1,
        1 => 0,
        _ => t,
    };
    if a.gfx12() {
        for t in 0..8u8 {
            b.vopd(term(kind(t), 0, t), term(kind(t), 1, n1(t)))?;
        }
        return Ok(());
    }
    // gfx11: a packet's two src1 operands must sit in different VGPR banks, so
    // terms 2..7 (both rows read the same x) pair row 0's term k with row 1's
    // term k-1.
    for t in 0..2u8 {
        b.vopd(term(kind(t), 0, t), term(kind(t), 1, n1(t)))?;
    }
    fmac(b, DOT[0], W[0] + 2, x + 2)?;
    for t in 3..8u8 {
        b.vopd(term(VopdF32::Fmac, 0, t), term(VopdF32::Fmac, 1, t - 1))?;
    }
    fmac(b, DOT[1], W[1] + 7, x + 7)
}
fn fmac(b: &mut Builder, dst: u8, w: u8, x: u8) -> R {
    valu(
        b,
        &format!("v_fmac_f32_e32 v{dst}, v{w}, v{x}"),
        &[dst],
        &[dst, w, x],
        &[],
    )
}

/// Source-matched HIP scalar objects on gfx1151/gfx1100 start every output
/// with term 1, then term 0. Full quads multiply (weight, X); residual groups
/// multiply (X, weight). Keep both operand orders, even for commutative inputs.
/// Pinned source-matched objects are recorded in PLAN's singleton-reference
/// packet: Halo cache `gemv_mq4g256v2_mq4v2.92ecf66b777f981f` and XTX cache
/// `gemv_mq4g256v2_rdna3_mq4v2.37cf66a5274c0003`. Their scalar fold is 01+32.
fn scalar_dots(b: &mut Builder, x: u8, tail: bool) -> R {
    for (r, &w) in W.iter().enumerate() {
        let dst = DOT[r];
        for (term, n) in [1, 0, 2, 3, 4, 5, 6, 7].into_iter().enumerate() {
            let (lhs, rhs) = if tail { (x + n, w + n) } else { (w + n, x + n) };
            if term == 0 {
                valu(
                    b,
                    &format!("v_mul_f32_e32 v{dst}, v{lhs}, v{rhs}"),
                    &[dst],
                    &[lhs, rhs],
                    &[],
                )?;
            } else {
                fmac(b, dst, lhs, rhs)?;
            }
        }
    }
    Ok(())
}

fn columns(b: &mut Builder, nb: u8, s: u8, scalar_tail: Option<bool>) -> R {
    let a = b.spec.arch;
    for c in 0..nb {
        let x = ring(c);
        // At most current+next x tiles are pending, two loads each. These
        // explicit per-arm waits are needed even when the untyped ledger's
        // textual predecessor is a different forward-dispatch arm. Retire
        // current while leaving next in flight; the last column drains.
        b.wait(load_counter(a), if c + 1 < nb { 2 } else { 0 })?;
        if let Some(tail) = scalar_tail {
            scalar_dots(b, x, tail)?;
        } else {
            dots(b, x)?;
        }
        b.vopd(
            VopdOp {
                op: VopdF32::Add,
                dst: acc(c, s, 0),
                src0: Operand::V(if scalar_tail.is_some() {
                    acc(c, s, 0)
                } else {
                    DOT[0]
                }),
                src1: if scalar_tail.is_some() {
                    DOT[0]
                } else {
                    acc(c, s, 0)
                },
            },
            VopdOp {
                op: VopdF32::Add,
                dst: acc(c, s, 1),
                src0: Operand::V(acc(c, s, 1)),
                src1: DOT[1],
            },
        )?;
        if c + 2 < nb {
            x_load(b, c + 2)?;
        }
    }
    Ok(())
}

fn reduce(b: &mut Builder, nb: u8) -> R {
    let a = b.spec.arch;
    for c in 0..nb {
        for r in 0..2 {
            let (v, t) = (acc(c, 0, r), red_tmp(c, r));
            b.ds_crosslane(
                Instruction::new(
                    format!("ds_swizzle_b32 v{t}, v{v} offset:swizzle(BITMASK_PERM,\"1pppp\")"),
                    refs(Kind::V, &[t]),
                    refs(Kind::V, &[v]),
                )
                .memory(MemoryClass::DsLoad),
            )?;
        }
    }
    for c in 0..nb {
        for r in 0..2 {
            let (v, t) = (acc(c, 0, r), red_tmp(c, r));
            valu(
                b,
                &format!("v_add_f32_e32 v{v}, v{v}, v{t}"),
                &[v],
                &[v, t],
                &[],
            )?;
        }
    }
    for off in [8u32, 4, 2, 1] {
        valu(
            b,
            &format!("v_cmp_gt_u32_e32 vcc_lo, {}, v0", 32 - off),
            &[],
            &[0],
            &[],
        )?;
        if a.gfx12() {
            b.push(Instruction::new(
                "s_wait_alu depctr_va_vcc(0)",
                vec![],
                vec![],
            ))?;
        }
        valu(
            b,
            &format!("v_cndmask_b32_e64 v4, 0, {off}, vcc_lo"),
            &[4],
            &[],
            &[],
        )?;
        valu(b, "v_add_lshl_u32 v4, v4, v0, 2", &[4], &[4, 0], &[])?;
        for c in 0..nb {
            for r in 0..2 {
                let (v, t) = (acc(c, 0, r), red_tmp(c, r));
                b.ds_crosslane(
                    Instruction::new(
                        format!("ds_bpermute_b32 v{t}, v4, v{v}"),
                        refs(Kind::V, &[t]),
                        refs(Kind::V, &[4, v]),
                    )
                    .memory(MemoryClass::DsLoad),
                )?;
            }
        }
        for c in 0..nb {
            for r in 0..2 {
                let (v, t) = (acc(c, 0, r), red_tmp(c, r));
                valu(
                    b,
                    &format!("v_add_f32_e32 v{v}, v{v}, v{t}"),
                    &[v],
                    &[v, t],
                    &[],
                )?;
            }
        }
    }
    Ok(())
}

/// Lane 0 stores `y[c*M + row0]` (and `+4` for the odd row when it exists).
fn store(b: &mut Builder, nb: u8, label_nb: u8) -> R {
    let a = b.spec.arch;
    let rowtiles = b.spec.system_sgpr_workgroup_id_y;
    let end = end_label(rowtiles);
    valu(b, "v_cmpx_eq_u32_e32 0, v0", &[], &[0], &[])?;
    for c in 0..nb {
        sop(b, format!("s_mul_i32 s{STMP}, s{MB}, {c}"), &[STMP], &[MB])?;
        sop(
            b,
            format!("{} s{STMP}, s{STMP}, s{R0B}", s_add_i32(a)),
            &[STMP],
            &[STMP, R0B],
        )?;
        valu(
            b,
            &format!("v_mov_b32_e32 v{}, s{STMP}", 8 + c),
            &[8 + c],
            &[],
            &[STMP],
        )?;
    }
    let ys: Vec<u8> = vec![SY, SY + 1];
    for c in 0..nb {
        vmem(
            b,
            &format!(
                "global_store_b32 v{}, v{}, s[{SY}:{}]",
                8 + c,
                acc(c, 0, 0),
                SY + 1
            ),
            &[],
            &[8 + c, acc(c, 0, 0)],
            &ys,
            true,
        )?;
    }
    sop(
        b,
        format!("s_cmp_eq_u32 s{ROW1}, s{ROW0}"),
        &[],
        &[ROW1, ROW0],
    )?;
    branch(
        b,
        &format!("s_cbranch_scc1 {}", lbl(label_nb, "stored", rowtiles)),
    )?;
    for c in 0..nb {
        vmem(
            b,
            &format!(
                "global_store_b32 v{}, v{}, s[{SY}:{}] offset:4",
                8 + c,
                acc(c, 0, 1),
                SY + 1
            ),
            &[],
            &[8 + c, acc(c, 0, 1)],
            &ys,
            true,
        )?;
    }
    b.label(&lbl(label_nb, "stored", rowtiles))?;
    b.wait_all()?;
    branch(b, &format!("s_branch {end}"))
}

fn refs(kind: Kind, indices: &[u8]) -> Vec<RegRef> {
    indices
        .iter()
        .map(|&base| RegRef { kind, base, len: 1 })
        .collect()
}
fn valu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8]) -> R {
    let mut inputs = refs(Kind::V, uses);
    inputs.extend(refs(Kind::S, scalar));
    b.push(Instruction::new(text, refs(Kind::V, defs), inputs))
}
fn vmem(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8], store: bool) -> R {
    let mut inputs = refs(Kind::V, uses);
    inputs.extend(refs(Kind::S, scalar));
    b.push(
        Instruction::new(text, refs(Kind::V, defs), inputs).memory(if store {
            MemoryClass::VmemStore
        } else {
            MemoryClass::VmemLoad
        }),
    )
}
fn branch(b: &mut Builder, text: &str) -> R {
    b.control(Instruction::new(text, vec![], vec![]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rowtile_k_mask_uses_canonical_hex_literal() {
        for arch in [Arch::Gfx1151, Arch::Gfx1201, Arch::Gfx1100] {
            let e = emit_rowtiles(Spec { arch }).expect("rowtile emission");
            assert!(
                e.s_text.contains(&format!("s_and_b32 s{TMP}, s{SK}, 0xff")),
                "{arch:?}: missing canonical K mask"
            );
            assert!(
                !e.s_text.contains(&format!("s_and_b32 s{TMP}, s{SK}, 255")),
                "{arch:?}: raw decimal K mask"
            );
        }
    }
}
