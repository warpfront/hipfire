// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1151 Q8_0 row kernels of the Flash-Next lane step: exact multi-column
//! (1..=8 activation columns) twins of the HIP incumbents `gemv_q8_0_k8` /
//! `gemv_q8_0_k320_staged` that read each activation once for several weight
//! rows instead of once per weight row.
//!
//! Every column `c` of every output row keeps the incumbent's arithmetic DAG
//! (`fn-batch/s7/DAG.md` §2, §3), so `y[c*M + row]` is byte-identical to the
//! 1-row incumbent on `x[c*K..]`:
//!  * a Q8_0 block is 34 bytes (f16 `d`, then 32 int8 `q`); lane `l` of a
//!    block takes `q[l]` and `x[32*b + l]`; `dq = v_fma_mix_f32(d_f16,
//!    f32(q), -0.0)` (`op_sel_hi:[1,0,0]`), `sum = fma(x, dq, sum)` from `+0`
//!    in increasing block order;
//!  * lane reduction: `ds_swizzle_b32 "1pppp"` (offset 16), then
//!    `ds_bpermute_b32` 8/4/2/1 with own-lane addresses for the out-of-range
//!    lanes, each level `own + other`; lane 0 holds the wave's value;
//!  * k8 only: lane 0 of wave `v` parks `part[row][col][v]` in LDS, one thread
//!    per (row, column) folds `((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7))` and, for
//!    `hc_act != 0`, runs the BF16 round trip, `scale * B`, a second round
//!    trip, the incumbent's exact exp/division expansion (SiLU) and a final
//!    round trip, all as VALU (the HIP object's `s_mul_f32` / `s_add_f32` /
//!    exec-masked sequences become the same operations lane-wise; the
//!    BF16 exceptional-exponent path is a `v_cndmask` select that yields the
//!    same bits). The F32 bits are stored.
//!
//! # `gemv_q8_0_k8_xbatch_pm`
//! Kernargs `A@0, x@8, y@16, M@24, K@28, B@32, scale@36 (f32), hc_act@40`
//! (44 B). Grid `ceil(M/4)`, block 256 (8 waves), static LDS 1024 B
//! (`part[4][8][8]` f32). Workgroup `w` owns weight rows `4w..4w+3` (loads
//! clamped to `M-1`, stores only for `row < M`); wave `v` runs K-slice `v`
//! (`K/8` elements, `K/256` blocks, `K % 2048 == 0`) of the four rows for
//! every column, and each activation load feeds the four rows. The slice is
//! processed as `K/2048` iterations of 8 unrolled blocks. Loads run in a
//! four-block software pipeline (a ring of four block slots; the first four
//! blocks of the next iteration load while the last four compute; the last
//! iteration re-reads its own first blocks instead of running off the slice).
//! A and the per-row offsets are 32-bit VGPR offsets from the kernarg bases:
//! `M * (K/32) * 34` must stay below 2^32 bytes and `B * K * 4` below 2^32.
//!
//! # `gemv_q8_0_k320_xbatch_pm`
//! Kernargs `A@0, x@8, y@16, M@24, B@28` (32 B), `K == 320` (10 blocks, 340
//! bytes per row). Grid `ceil(M/16)`, block 64 (2 waves), static LDS 1536 B.
//! Wave `v` of workgroup `w` owns rows `16w + 8v .. +7` (clamped to `M-1`),
//! keeps its `B x 10` activations in registers, and per row stages the 340
//! byte row in wave-local LDS (85 dwords: lane, lane+32, lane+64 for lane <
//! 21; lanes past 21 re-read dword 84 into a 44-byte pad of the 384-byte wave
//! region), then reads `d` / `q` back from LDS. Rows are double-buffered over
//! two LDS slots with one workgroup barrier per row; the global loads of a
//! row run four rows ahead through a register ring.
//!
//! B is specialized (one straight-line path per column count, selected once
//! from the kernarg); `B` outside `1..=8` exits without storing.
//!
//! Historical r16/r32 wide entries (`gemv_q8_0_{k8,k320,k2560,k2560_x2,wide}_xbatch_wide_r{16,32}_pm`,
//! `build_wide_gfx1151`/`WIDE_EXPORTS`/`WIDE_CAPS`) were removed; the runtime/registry is
//! t8-only. The fixed-eight `build_rowtiles_*` twins below are the sole wide path and keep
//! the selected decode-object arithmetic, staging and barrier discipline unchanged.
//!
//! # Fixed-eight row tiles ([`build_rowtiles_gfx1151`])
//! Five gfx1151-only exports use the same recipe bodies with eight physical
//! activation rows and total logical `R = 1..=64`. Grid y is `ceil(R/8)`;
//! x/y bases advance by `8*workgroup_id_y*K` / `8*workgroup_id_y*M` elements
//! after x2 matrix selection, never by the live tail count. Inactive output
//! waves still stage X and meet every barrier. Tail rows do not load, reduce
//! or store. The HC partial array is bounded to `partial[2][8][8]`.
use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{lit, mem, op, s, smem, sop, sr, v, vr};
use crate::lds::Transition;
use crate::ledger::Counter;
use crate::plan::Access;
use crate::reg::{Kind, Live, RegRef};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

pub const K8_SYMBOL: &str = "gemv_q8_0_k8_xbatch_pm";
pub const K320_SYMBOL: &str = "gemv_q8_0_k320_xbatch_pm";
pub const MAX_COLUMNS: u8 = 8;

/// Static LDS of k8: `part[4 rows][8 columns][8 waves]` f32.
pub const K8_LDS_BYTES: u32 = 4 * 8 * 8 * 4;
/// Static LDS of k320: two row slots of two 384-byte wave regions.
pub const K320_LDS_BYTES: u32 = 2 * 2 * K320_WAVE_REGION;
/// One wave's staging region of k320: 340 row bytes plus a 44-byte pad that
/// absorbs the third staging store of the lanes past 20.
const K320_WAVE_REGION: u32 = 384;
/// One k320 slot (both waves' regions).
const K320_SLOT: u32 = 2 * K320_WAVE_REGION;

/// Values per lane-tree batch (temporaries).
const TREE_CHUNK: usize = 8;

// ------------------------------------------------------------------ helpers

/// Declare `base..base+len` as the largest aligned pieces the allocator admits.
fn declare(p: &mut RegPlan, name: &str, kind: Kind, base: u8, len: u8) -> R {
    let mut at = 0u8;
    while at < len {
        let rest = len - at;
        let piece = [8u8, 4, 2, 1]
            .into_iter()
            .find(|w| *w <= rest && (kind == Kind::V || (base + at) % *w == 0))
            .ok_or("unalignable register range")?;
        p.add_range(&format!("{name}{at}"), kind, base + at, piece, Live::Whole)?;
        at += piece;
    }
    Ok(())
}

/// One VALU instruction over VGPR defs/uses plus SGPR uses.
fn vop(b: &mut Builder, text: impl Into<String>, defs: &[u8], uses: &[u8], suses: &[u8]) -> R {
    let d: Vec<RegRef> = defs.iter().map(|&n| v(n)).collect();
    let mut u: Vec<RegRef> = uses.iter().map(|&n| v(n)).collect();
    u.extend(suses.iter().map(|&n| s(n)));
    op(b, text, &d, &u)
}

fn off(imm: u32) -> String {
    if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    }
}

/// `global_load_*` of one dword-or-narrower value at `s[base:base+1] + v{voff} + imm`.
fn gload(b: &mut Builder, mnemonic: &str, dst: u8, voff: u8, base: u8, imm: u32) -> R {
    mem(
        b,
        format!(
            "{mnemonic} v{dst}, v{voff}, s[{base}:{}]{}",
            base + 1,
            off(imm)
        ),
        &[v(dst)],
        &[v(voff), sr(base, 2)],
        MemoryClass::VmemLoad,
    )
}

fn ds_st(b: &mut Builder, slot: usize, addr: u8, data: u8, imm: u32) -> R {
    b.ds_store(
        slot,
        Instruction::new(
            format!("ds_store_b32 v{addr}, v{data}{}", off(imm)),
            vec![],
            vec![v(addr), v(data)],
        )
        .memory(MemoryClass::DsStore),
    )
}

/// `ds_load_u16_d16` (low half, high half kept) and `ds_load_i8` (sign extended).
fn ds_ld(
    b: &mut Builder,
    slot: usize,
    mnemonic: &str,
    dst: u8,
    addr: u8,
    imm: u32,
    keeps_dst: bool,
) -> R {
    let mut uses = vec![v(addr)];
    if keeps_dst {
        uses.push(v(dst));
    }
    b.ds_load(
        slot,
        Instruction::new(
            format!("{mnemonic} v{dst}, v{addr}{}", off(imm)),
            vec![v(dst)],
            uses,
        )
        .memory(MemoryClass::DsLoad),
    )
}

/// The wave-wide lane reduction of every value in `values` (VGPRs), batched
/// by `TREE_CHUNK` through `tmp..tmp+TREE_CHUNK`: swizzle offset 16, then the
/// own-lane-clamped bpermutes 8/4/2/1 whose byte addresses sit in
/// `addr..addr+4`; each level is `own + other`, lane 0 ends with the sum.
fn lane_tree(b: &mut Builder, values: &[u8], tmp: u8, addr: u8) -> R {
    let add = |b: &mut Builder, chunk: &[u8]| -> R {
        for (i, &val) in chunk.iter().enumerate() {
            let t = tmp + i as u8;
            vop(
                b,
                format!("v_add_f32_e32 v{val}, v{val}, v{t}"),
                &[val],
                &[val, t],
                &[],
            )?;
        }
        Ok(())
    };
    for chunk in values.chunks(TREE_CHUNK) {
        for (i, &val) in chunk.iter().enumerate() {
            let t = tmp + i as u8;
            b.ds_crosslane(
                Instruction::new(
                    format!("ds_swizzle_b32 v{t}, v{val} offset:swizzle(BITMASK_PERM,\"1pppp\")"),
                    vec![v(t)],
                    vec![v(val)],
                )
                .memory(MemoryClass::DsLoad),
            )?;
        }
        add(b, chunk)?;
        for level in 0..4u8 {
            for (i, &val) in chunk.iter().enumerate() {
                let t = tmp + i as u8;
                b.ds_crosslane(
                    Instruction::new(
                        format!("ds_bpermute_b32 v{t}, v{}, v{val}", addr + level),
                        vec![v(t)],
                        vec![v(addr + level), v(val)],
                    )
                    .memory(MemoryClass::DsLoad),
                )?;
            }
            add(b, chunk)?;
        }
    }
    Ok(())
}

/// Byte addresses of the shfl_down 8/4/2/1 bpermutes into `addr..addr+4`:
/// `(lane + (lane < 32 - o ? o : 0)) * 4`, i.e. the own lane past the end.
fn bpermute_addresses(b: &mut Builder, addr: u8, lane: u8) -> R {
    for (i, o) in [8u32, 4, 2, 1].into_iter().enumerate() {
        let a = addr + i as u8;
        vop(
            b,
            format!("v_cmp_gt_u32_e32 vcc_lo, {}, v{lane}", 32 - o),
            &[],
            &[lane],
            &[],
        )?;
        vop(
            b,
            format!("v_cndmask_b32_e64 v{a}, 0, {o}, vcc_lo"),
            &[a],
            &[],
            &[],
        )?;
        vop(
            b,
            format!("v_add_lshl_u32 v{a}, v{a}, v{lane}, 2"),
            &[a],
            &[a, lane],
            &[],
        )?;
    }
    Ok(())
}

/// The incumbent's BF16 round trip of `x` in place (RNE, exceptional exponent
/// preserved; `DAG.md` §2 / §6): the finite result
/// `(u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000`, and for an all-ones
/// exponent `u` (low half zero) or `u | 0x10000` (low half non-zero), selected
/// by `v_cndmask` instead of the HIP object's exec-masked branches. `t` are
/// four scratch VGPRs.
fn bf16_round(b: &mut Builder, x: u8, t: [u8; 4]) -> R {
    vop(
        b,
        format!("v_and_b32_e32 v{}, 0x7f800000, v{x}", t[0]),
        &[t[0]],
        &[x],
        &[],
    )?;
    vop(
        b,
        format!("v_bfe_u32 v{}, v{x}, 16, 1", t[1]),
        &[t[1]],
        &[x],
        &[],
    )?;
    vop(
        b,
        format!("v_add3_u32 v{0}, v{x}, v{0}, 0x7fff", t[1]),
        &[t[1]],
        &[x, t[1]],
        &[],
    )?;
    vop(
        b,
        format!("v_and_b32_e32 v{}, 0xffff, v{x}", t[2]),
        &[t[2]],
        &[x],
        &[],
    )?;
    vop(
        b,
        format!("v_or_b32_e32 v{}, 0x10000, v{x}", t[3]),
        &[t[3]],
        &[x],
        &[],
    )?;
    vop(
        b,
        format!("v_cmp_eq_u32_e32 vcc_lo, 0, v{}", t[2]),
        &[],
        &[t[2]],
        &[],
    )?;
    vop(
        b,
        format!("v_cndmask_b32_e32 v{0}, v{1}, v{x}, vcc_lo", t[2], t[3]),
        &[t[2]],
        &[t[3], x],
        &[],
    )?;
    vop(
        b,
        format!("v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v{}", t[0]),
        &[],
        &[t[0]],
        &[],
    )?;
    vop(
        b,
        format!("v_cndmask_b32_e32 v{0}, v{1}, v{0}, vcc_lo", t[1], t[2]),
        &[t[1]],
        &[t[2], t[1]],
        &[],
    )?;
    vop(
        b,
        format!("v_and_b32_e32 v{x}, 0xffff0000, v{}", t[1]),
        &[x],
        &[t[1]],
        &[],
    )
}

/// SiLU of `x` in place: the r8 incumbent's expansion of `x / (1 + expf(-x))`
/// instruction for instruction (`DAG.md` §2, lines 10645-10679), registers
/// `t[0..6]` standing for the listing's v2..v7 and `x` for v1.
fn silu(b: &mut Builder, x: u8, t: [u8; 6]) -> R {
    let [t0, t1, t2, t3, t4, t5] = t;
    vop(
        b,
        format!("v_mul_f32_e32 v{t0}, 0xbfb8aa3b, v{x}"),
        &[t0],
        &[x],
        &[],
    )?;
    vop(
        b,
        format!("v_fma_f32 v{t1}, 0xbfb8aa3b, v{x}, -v{t0}"),
        &[t1],
        &[x, t0],
        &[],
    )?;
    vop(
        b,
        format!("v_rndne_f32_e32 v{t2}, v{t0}"),
        &[t2],
        &[t0],
        &[],
    )?;
    vop(
        b,
        format!("v_sub_f32_e32 v{t0}, v{t0}, v{t2}"),
        &[t0],
        &[t0, t2],
        &[],
    )?;
    vop(
        b,
        format!("v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v{x}"),
        &[],
        &[x],
        &[],
    )?;
    vop(
        b,
        format!("v_fmamk_f32 v{t1}, v{x}, 0xb2a5705f, v{t1}"),
        &[t1],
        &[x, t1],
        &[],
    )?;
    vop(
        b,
        format!("v_add_f32_e32 v{t0}, v{t0}, v{t1}"),
        &[t0],
        &[t0, t1],
        &[],
    )?;
    vop(
        b,
        format!("v_cvt_i32_f32_e32 v{t1}, v{t2}"),
        &[t1],
        &[t2],
        &[],
    )?;
    vop(b, format!("v_exp_f32_e32 v{t0}, v{t0}"), &[t0], &[t0], &[])?;
    vop(
        b,
        format!("v_ldexp_f32 v{t0}, v{t0}, v{t1}"),
        &[t0],
        &[t0, t1],
        &[],
    )?;
    vop(
        b,
        format!("v_cndmask_b32_e32 v{t0}, 0, v{t0}, vcc_lo"),
        &[t0],
        &[t0],
        &[],
    )?;
    vop(
        b,
        format!("v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v{x}"),
        &[],
        &[x],
        &[],
    )?;
    vop(
        b,
        format!("v_cndmask_b32_e32 v{t0}, 0x7f800000, v{t0}, vcc_lo"),
        &[t0],
        &[t0],
        &[],
    )?;
    vop(
        b,
        format!("v_add_f32_e32 v{t0}, 1.0, v{t0}"),
        &[t0],
        &[t0],
        &[],
    )?;
    vop(
        b,
        format!("v_div_scale_f32 v{t1}, null, v{t0}, v{t0}, v{x}"),
        &[t1],
        &[t0, x],
        &[],
    )?;
    vop(b, format!("v_rcp_f32_e32 v{t2}, v{t1}"), &[t2], &[t1], &[])?;
    vop(
        b,
        format!("v_fma_f32 v{t3}, -v{t1}, v{t2}, 1.0"),
        &[t3],
        &[t1, t2],
        &[],
    )?;
    vop(
        b,
        format!("v_fmac_f32_e32 v{t2}, v{t3}, v{t2}"),
        &[t2],
        &[t2, t3],
        &[],
    )?;
    vop(
        b,
        format!("v_div_scale_f32 v{t3}, vcc_lo, v{x}, v{t0}, v{x}"),
        &[t3],
        &[x, t0],
        &[],
    )?;
    vop(
        b,
        format!("v_mul_f32_e32 v{t4}, v{t3}, v{t2}"),
        &[t4],
        &[t3, t2],
        &[],
    )?;
    vop(
        b,
        format!("v_fma_f32 v{t5}, -v{t1}, v{t4}, v{t3}"),
        &[t5],
        &[t1, t4, t3],
        &[],
    )?;
    vop(
        b,
        format!("v_fmac_f32_e32 v{t4}, v{t5}, v{t2}"),
        &[t4],
        &[t4, t5, t2],
        &[],
    )?;
    vop(
        b,
        format!("v_fma_f32 v{t1}, -v{t1}, v{t4}, v{t3}"),
        &[t1],
        &[t1, t4, t3],
        &[],
    )?;
    vop(
        b,
        format!("v_div_fmas_f32 v{t1}, v{t1}, v{t2}, v{t4}"),
        &[t1],
        &[t1, t2, t4],
        &[],
    )?;
    vop(
        b,
        format!("v_div_fixup_f32 v{x}, v{t1}, v{t0}, v{x}"),
        &[x],
        &[t1, t0, x],
        &[],
    )
}

/// `dq = fma_mix(d_f16, q, -0.0)` in place over the block's `d` register and
/// `q = f32(int8)` in place (`DAG.md` §2: `neg(0) op_sel_hi:[1,0,0]`).
fn dequant(b: &mut Builder, d: u8, q: u8) -> R {
    vop(b, format!("v_cvt_f32_i32_e32 v{q}, v{q}"), &[q], &[q], &[])?;
    vop(
        b,
        format!("v_fma_mix_f32 v{d}, v{d}, v{q}, neg(0) op_sel_hi:[1,0,0]"),
        &[d],
        &[d, q],
        &[],
    )
}

// ===================================================================== k8
// VGPRs.
const V_TID: u8 = 0;
/// `tid & 31`.
const V_LANE: u8 = 1;
/// `(tid >> 5) * 4`: the wave's partial slot byte offset.
const V_WAVE4: u8 = 2;
/// Byte addresses of the shfl_down 8/4/2/1 bpermutes.
const V_ADDR: u8 = 3;
/// Activation byte offsets of the columns (`c*K*4 + wave*K/2 + lane*4`, +iteration).
const V_XOFF: u8 = 8;
/// Block-start byte offsets of the four rows (scale loads).
const V_WOFFD: u8 = 16;
/// The same plus the lane (int8 loads).
const V_WOFFQ: u8 = 20;
/// Four block slots of 16 VGPRs: q[4], d[4], x[8].
const V_SLOT: u8 = 32;
/// `sum[row][col]` at `+ 8*row + col`.
const V_SUM: u8 = 96;
const V_TMP: u8 = 128;
/// The eight wave partials of this thread's (row, column).
const V_PART: u8 = 136;
/// Epilogue scratch (six for SiLU, four for BF16).
const V_EP: u8 = 144;
/// Epilogue indices: row-in-group, column, LDS read address, global row, y byte offset.
const V_IR: u8 = 152;
const V_IC: u8 = 153;
const V_ILDS: u8 = 154;
const V_IROW: u8 = 155;
const V_IY: u8 = 156;
/// VGPR count of the k8 plan (`V_IY + 1`).
pub const K8_VGPRS: u16 = 157;

// SGPRs.
const S_KARG: u8 = 0;
const S_WG: u8 = 2;
/// A at s[4:5], x at s[6:7], y at s[8:9].
const S_A: u8 = 4;
const S_X: u8 = 6;
const S_Y: u8 = 8;
const S_M: u8 = 10;
const S_K: u8 = 11;
const S_B: u8 = 12;
const S_SCALE: u8 = 13;
const S_HC: u8 = 14;
const S_WAVE: u8 = 15;
const S_ROW0: u8 = 16;
const S_ROWBYTES: u8 = 17;
const S_SLICE: u8 = 18;
const S_XWAVE: u8 = 19;
/// `rowoff[r] = min(row0 + r, M-1) * rowbytes + slice` of the four rows.
const S_ROWOFF: u8 = 20;
const S_KX4: u8 = 24;
const S_ITERS: u8 = 25;
const S_WSTEP: u8 = 26;
const S_XSTEP: u8 = 27;
const S_T0: u8 = 28;
const S_MM1: u8 = 29;
/// Lane-0 exec mask.
const S_LANE0: u8 = 30;
/// Bytes of one 8-block iteration: weights 8*34, activations 8*128.
const K8_WSTEP: u32 = 8 * 34;
const K8_XSTEP: u32 = 8 * 128;

fn k8_q(slot: u8, r: u8) -> u8 {
    V_SLOT + 16 * slot + r
}
fn k8_d(slot: u8, r: u8) -> u8 {
    V_SLOT + 16 * slot + 4 + r
}
fn k8_x(slot: u8, c: u8) -> u8 {
    V_SLOT + 16 * slot + 8 + c
}
fn k8_sum(r: u8, c: u8) -> u8 {
    V_SUM + 8 * r + c
}

fn k8_plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(K8_VGPRS, 40)?;
    for (name, base, len) in [
        ("kernarg_ptr", S_KARG, 2),
        ("wg_x", S_WG, 1),
        ("a_x_ptrs", S_A, 4),
        ("y_ptr", S_Y, 2),
        ("m_k", S_M, 2),
        ("b_scale", S_B, 2),
        ("hc_act", S_HC, 1),
    ] {
        declare(&mut p, name, Kind::S, base, len)?;
    }
    for n in S_WAVE..=S_LANE0 {
        declare(&mut p, "scalar", Kind::S, n, 1)?;
    }
    for (name, base, len) in [
        ("tid", V_TID, 1),
        ("lane", V_LANE, 1),
        ("wave4", V_WAVE4, 1),
        ("bpermute_addr", V_ADDR, 4),
        ("x_offset", V_XOFF, 8),
        ("w_offset_d", V_WOFFD, 4),
        ("w_offset_q", V_WOFFQ, 4),
        ("block_slots", V_SLOT, 64),
        ("sums", V_SUM, 32),
        ("tree_tmp", V_TMP, 8),
        ("partials", V_PART, 8),
        ("epilogue_tmp", V_EP, 8),
        ("epilogue_index", V_IR, 5),
    ] {
        declare(&mut p, name, Kind::V, base, len)?;
    }
    Ok(p)
}

fn k8_spec() -> KernelSpec {
    KernelSpec {
        kernel_id: "qwen4_lane_rows_q8".into(),
        variant: "k8_xbatch".into(),
        arch: Arch::Gfx1151,
        symbol: K8_SYMBOL.into(),
        kernargs: KernargLayout::new(44)
            .pointer_access("A", 0, Access::ReadOnly)
            .pointer_access("x", 8, Access::ReadOnly)
            .pointer_access("y", 16, Access::WriteOnly)
            .hidden("M", 24, 4, "by_value")
            .hidden("K", 28, 4, "by_value")
            .hidden("B", 32, 4, "by_value")
            .hidden("scale", 36, 4, "by_value")
            .hidden("hc_act", 40, 4, "by_value"),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 256,
        group_segment_fixed_size: K8_LDS_BYTES,
        wave32: true,
        cu_mode: false,
    }
}

fn k8() -> Result<Emitted, String> {
    let mut b = Builder::new(k8_spec(), k8_plan()?);
    b.enable_delay_alu();
    let slot = b.lds_slot("part", 0, K8_LDS_BYTES)?;
    k8_prologue(&mut b)?;
    for nb in 1..=MAX_COLUMNS {
        sop(&mut b, format!("s_cmp_eq_u32 s{S_B}, {nb}"), &[], &[S_B])?;
        op(&mut b, format!("s_cbranch_scc1 .Lq8k8_b{nb}"), &[], &[])?;
    }
    op(&mut b, "s_branch .Lq8k8_end", &[], &[])?;
    for nb in 1..=MAX_COLUMNS {
        k8_section(&mut b, nb, slot)?;
    }
    b.label(".Lq8k8_end")?;
    op(&mut b, "s_endpgm", &[], &[])?;
    b.finish()
}

/// Kernarg loads, lane/wave constants, row and slice offsets; exits when the
/// workgroup has no row or `K < 2048`.
fn k8_prologue(b: &mut Builder) -> R {
    smem(b, S_A, 4, S_KARG, 0)?;
    smem(b, S_Y, 2, S_KARG, 0x10)?;
    smem(b, S_M, 2, S_KARG, 0x18)?;
    smem(b, S_B, 2, S_KARG, 0x20)?;
    smem(b, S_HC, 1, S_KARG, 0x28)?;
    vop(
        b,
        format!("v_and_b32_e32 v{V_LANE}, 31, v{V_TID}"),
        &[V_LANE],
        &[V_TID],
        &[],
    )?;
    vop(
        b,
        format!("v_lshrrev_b32_e32 v{V_WAVE4}, 5, v{V_TID}"),
        &[V_WAVE4],
        &[V_TID],
        &[],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{S_WAVE}, v{V_WAVE4}"),
        &[s(S_WAVE)],
        &[v(V_WAVE4)],
    )?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{V_WAVE4}, 2, v{V_WAVE4}"),
        &[V_WAVE4],
        &[V_WAVE4],
        &[],
    )?;
    bpermute_addresses(b, V_ADDR, V_LANE)?;
    vop(
        b,
        format!("v_cmp_eq_u32_e32 vcc_lo, 0, v{V_LANE}"),
        &[],
        &[V_LANE],
        &[],
    )?;
    op(
        b,
        format!("s_mov_b32 s{S_LANE0}, vcc_lo"),
        &[s(S_LANE0)],
        &[],
    )?;
    b.wait_all()?;
    sop(
        b,
        format!("s_lshl_b32 s{S_ROW0}, s{S_WG}, 2"),
        &[S_ROW0],
        &[S_WG],
    )?;
    sop(
        b,
        format!("s_cmp_ge_i32 s{S_ROW0}, s{S_M}"),
        &[],
        &[S_ROW0, S_M],
    )?;
    op(b, "s_cbranch_scc1 .Lq8k8_end", &[], &[])?;
    sop(
        b,
        format!("s_lshr_b32 s{S_ITERS}, s{S_K}, 11"),
        &[S_ITERS],
        &[S_K],
    )?;
    sop(b, format!("s_cmp_eq_u32 s{S_ITERS}, 0"), &[], &[S_ITERS])?;
    op(b, "s_cbranch_scc1 .Lq8k8_end", &[], &[])?;
    // rowbytes = (K/32)*34; slice = wave*(K/256)*34; xwave = wave*K/2 (activation bytes); K*4.
    sop(
        b,
        format!("s_lshr_b32 s{S_ROWBYTES}, s{S_K}, 5"),
        &[S_ROWBYTES],
        &[S_K],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{S_ROWBYTES}, s{S_ROWBYTES}, 34"),
        &[S_ROWBYTES],
        &[S_ROWBYTES],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{S_SLICE}, s{S_K}, 8"),
        &[S_SLICE],
        &[S_K],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{S_SLICE}, s{S_SLICE}, 34"),
        &[S_SLICE],
        &[S_SLICE],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{S_SLICE}, s{S_SLICE}, s{S_WAVE}"),
        &[S_SLICE],
        &[S_SLICE, S_WAVE],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{S_XWAVE}, s{S_K}, 1"),
        &[S_XWAVE],
        &[S_K],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{S_XWAVE}, s{S_XWAVE}, s{S_WAVE}"),
        &[S_XWAVE],
        &[S_XWAVE, S_WAVE],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{S_KX4}, s{S_K}, 2"),
        &[S_KX4],
        &[S_K],
    )?;
    sop(
        b,
        format!("s_add_i32 s{S_MM1}, s{S_M}, -1"),
        &[S_MM1],
        &[S_M],
    )?;
    for r in 0..4u8 {
        let ro = S_ROWOFF + r;
        sop(
            b,
            format!("s_add_i32 s{S_T0}, s{S_ROW0}, {r}"),
            &[S_T0],
            &[S_ROW0],
        )?;
        sop(
            b,
            format!("s_min_i32 s{S_T0}, s{S_T0}, s{S_MM1}"),
            &[S_T0],
            &[S_T0, S_MM1],
        )?;
        sop(
            b,
            format!("s_mul_i32 s{ro}, s{S_T0}, s{S_ROWBYTES}"),
            &[ro],
            &[S_T0, S_ROWBYTES],
        )?;
        sop(
            b,
            format!("s_add_i32 s{ro}, s{ro}, s{S_SLICE}"),
            &[ro],
            &[ro, S_SLICE],
        )?;
        vop(
            b,
            format!("v_mov_b32_e32 v{}, s{ro}", V_WOFFD + r),
            &[V_WOFFD + r],
            &[],
            &[ro],
        )?;
        vop(
            b,
            format!("v_add_nc_u32_e32 v{}, s{ro}, v{V_LANE}", V_WOFFQ + r),
            &[V_WOFFQ + r],
            &[V_LANE],
            &[ro],
        )?;
    }
    Ok(())
}

/// Loads of block `jb` (0..8 of the current base) into ring slot `slot`:
/// the four rows' int8 `q` and f16 `d`, then the columns' activations.
fn k8_issue(b: &mut Builder, nb: u8, slot: u8, jb: u32) -> R {
    for r in 0..4u8 {
        gload(
            b,
            "global_load_i8",
            k8_q(slot, r),
            V_WOFFQ + r,
            S_A,
            34 * jb + 2,
        )?;
    }
    for r in 0..4u8 {
        gload(
            b,
            "global_load_u16",
            k8_d(slot, r),
            V_WOFFD + r,
            S_A,
            34 * jb,
        )?;
    }
    for c in 0..nb {
        gload(
            b,
            "global_load_b32",
            k8_x(slot, c),
            V_XOFF + c,
            S_X,
            128 * jb,
        )?;
    }
    Ok(())
}

/// One block of all four rows against every column from ring slot `slot`:
/// `dq` once per row, then `sum[r][c] = fma(x[c], dq[r], sum[r][c])`.
fn k8_compute(b: &mut Builder, nb: u8, slot: u8) -> R {
    // At nb = 8 a slot's first int8 load has 63 younger loads in flight (three
    // 16-load blocks plus 15 of its own), which `vmcnt` (6 bits) cannot
    // express; drain the oldest two so the slot's waits fit the counter.
    if nb == 8 {
        b.wait(Counter::Vm, 62)?;
    }
    for r in 0..4u8 {
        vop(
            b,
            format!("v_cvt_f32_i32_e32 v{0}, v{0}", k8_q(slot, r)),
            &[k8_q(slot, r)],
            &[k8_q(slot, r)],
            &[],
        )?;
    }
    for r in 0..4u8 {
        let (d, q) = (k8_d(slot, r), k8_q(slot, r));
        vop(
            b,
            format!("v_fma_mix_f32 v{d}, v{d}, v{q}, neg(0) op_sel_hi:[1,0,0]"),
            &[d],
            &[d, q],
            &[],
        )?;
    }
    for c in 0..nb {
        for r in 0..4u8 {
            let (sum, x, d) = (k8_sum(r, c), k8_x(slot, c), k8_d(slot, r));
            vop(
                b,
                format!("v_fmac_f32_e32 v{sum}, v{x}, v{d}"),
                &[sum],
                &[sum, x, d],
                &[],
            )?;
        }
    }
    Ok(())
}

fn k8_section(b: &mut Builder, nb: u8, slot: usize) -> R {
    let tag = format!(".Lq8k8_b{nb}");
    b.label(&tag)?;
    for c in 0..nb {
        sop(
            b,
            format!("s_mul_i32 s{S_T0}, s{S_KX4}, {c}"),
            &[S_T0],
            &[S_KX4],
        )?;
        sop(
            b,
            format!("s_add_i32 s{S_T0}, s{S_T0}, s{S_XWAVE}"),
            &[S_T0],
            &[S_T0, S_XWAVE],
        )?;
        op(
            b,
            format!("v_lshl_add_u32 v{}, v{V_LANE}, 2, s{S_T0}", V_XOFF + c),
            &[v(V_XOFF + c)],
            &[v(V_LANE), s(S_T0)],
        )?;
    }
    for r in 0..4u8 {
        for c in 0..nb {
            vop(
                b,
                format!("v_mov_b32_e32 v{}, 0", k8_sum(r, c)),
                &[k8_sum(r, c)],
                &[],
                &[],
            )?;
        }
    }
    for slot_i in 0..4u8 {
        k8_issue(b, nb, slot_i, u32::from(slot_i))?;
    }
    let head = format!("{tag}_loop");
    b.loop_(&head, |b| {
        // The loads of the next iteration's first four blocks advance by a full
        // iteration; in the last iteration they re-read the current blocks.
        sop(b, format!("s_cmp_gt_u32 s{S_ITERS}, 1"), &[], &[S_ITERS])?;
        sop(
            b,
            format!("s_cselect_b32 s{S_WSTEP}, {}, 0", lit(K8_WSTEP)),
            &[S_WSTEP],
            &[],
        )?;
        sop(
            b,
            format!("s_cselect_b32 s{S_XSTEP}, {}, 0", lit(K8_XSTEP)),
            &[S_XSTEP],
            &[],
        )?;
        for j in 0..8u8 {
            let ring = j % 4;
            k8_compute(b, nb, ring)?;
            if j < 4 {
                k8_issue(b, nb, ring, u32::from(j) + 4)?;
                if j == 3 {
                    for r in 0..4u8 {
                        vop(
                            b,
                            format!("v_add_nc_u32_e32 v{0}, s{S_WSTEP}, v{0}", V_WOFFD + r),
                            &[V_WOFFD + r],
                            &[V_WOFFD + r],
                            &[S_WSTEP],
                        )?;
                        vop(
                            b,
                            format!("v_add_nc_u32_e32 v{0}, s{S_WSTEP}, v{0}", V_WOFFQ + r),
                            &[V_WOFFQ + r],
                            &[V_WOFFQ + r],
                            &[S_WSTEP],
                        )?;
                    }
                    for c in 0..nb {
                        vop(
                            b,
                            format!("v_add_nc_u32_e32 v{0}, s{S_XSTEP}, v{0}", V_XOFF + c),
                            &[V_XOFF + c],
                            &[V_XOFF + c],
                            &[S_XSTEP],
                        )?;
                    }
                }
            } else {
                k8_issue(b, nb, ring, u32::from(j) - 4)?;
            }
        }
        sop(
            b,
            format!("s_add_i32 s{S_ITERS}, s{S_ITERS}, -1"),
            &[S_ITERS],
            &[S_ITERS],
        )?;
        sop(b, format!("s_cmp_lg_u32 s{S_ITERS}, 0"), &[], &[S_ITERS])?;
        op(b, format!("s_cbranch_scc1 {head}"), &[], &[])
    })?;
    b.wait_all()?;
    // Wave reduction of every (row, column) value, then lane 0 parks the wave's
    // partial: part[(row*8 + col)*8 + wave].
    let values: Vec<u8> = (0..4u8)
        .flat_map(|r| (0..nb).map(move |c| k8_sum(r, c)))
        .collect();
    lane_tree(b, &values, V_TMP, V_ADDR)?;
    op(
        b,
        format!("s_mov_b32 exec_lo, s{S_LANE0}"),
        &[],
        &[s(S_LANE0)],
    )?;
    for r in 0..4u8 {
        for c in 0..nb {
            ds_st(
                b,
                slot,
                V_WAVE4,
                k8_sum(r, c),
                (u32::from(r) * 8 + u32::from(c)) * 32,
            )?;
        }
    }
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
    // Thread t (< 4*nb) folds (row t / nb, column t % nb): the quotient is
    // (t * ceil(2^16 / nb)) >> 16, exact for t < 32.
    let magic = (65536 + u32::from(nb) - 1) / u32::from(nb);
    vop(
        b,
        format!("v_mul_u32_u24_e32 v{V_IR}, {}, v{V_TID}", lit(magic)),
        &[V_IR],
        &[V_TID],
        &[],
    )?;
    vop(
        b,
        format!("v_lshrrev_b32_e32 v{V_IR}, 16, v{V_IR}"),
        &[V_IR],
        &[V_IR],
        &[],
    )?;
    vop(
        b,
        format!("v_mul_u32_u24_e32 v{V_IC}, {nb}, v{V_IR}"),
        &[V_IC],
        &[V_IR],
        &[],
    )?;
    vop(
        b,
        format!("v_sub_nc_u32_e32 v{V_IC}, v{V_TID}, v{V_IC}"),
        &[V_IC],
        &[V_TID, V_IC],
        &[],
    )?;
    vop(
        b,
        format!("v_lshl_add_u32 v{V_ILDS}, v{V_IR}, 3, v{V_IC}"),
        &[V_ILDS],
        &[V_IR, V_IC],
        &[],
    )?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{V_ILDS}, 5, v{V_ILDS}"),
        &[V_ILDS],
        &[V_ILDS],
        &[],
    )?;
    vop(
        b,
        format!("v_add_nc_u32_e32 v{V_IROW}, s{S_ROW0}, v{V_IR}"),
        &[V_IROW],
        &[V_IR],
        &[S_ROW0],
    )?;
    op(
        b,
        format!("v_mul_lo_u32 v{V_IY}, v{V_IC}, s{S_M}"),
        &[v(V_IY)],
        &[v(V_IC), s(S_M)],
    )?;
    vop(
        b,
        format!("v_add_nc_u32_e32 v{V_IY}, v{V_IROW}, v{V_IY}"),
        &[V_IY],
        &[V_IROW, V_IY],
        &[],
    )?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{V_IY}, 2, v{V_IY}"),
        &[V_IY],
        &[V_IY],
        &[],
    )?;
    b.barrier(&[Transition::Ready(slot)])?;
    vop(
        b,
        format!("v_cmp_gt_u32_e32 vcc_lo, {}, v{V_TID}", 4 * u32::from(nb)),
        &[],
        &[V_TID],
        &[],
    )?;
    op(b, "s_mov_b32 exec_lo, vcc_lo", &[], &[])?;
    for half in 0..2u8 {
        b.ds_load(
            slot,
            Instruction::new(
                format!(
                    "ds_load_b128 {}, v{V_ILDS}{}",
                    vrange(V_PART + 4 * half, 4),
                    off(16 * u32::from(half))
                ),
                vec![vr(V_PART + 4 * half, 4)],
                vec![v(V_ILDS)],
            )
            .memory(MemoryClass::DsLoad),
        )?;
    }
    b.wait_all()?;
    b.barrier(&[Transition::Retire(slot)])?;
    // Only wave 0 holds threads t < 4*nb.
    op(b, "s_cbranch_execz .Lq8k8_end", &[], &[])?;
    let p = |n: u8| V_PART + n;
    // ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7))
    for (dst, a, c) in [
        (0u8, 0u8, 1u8),
        (2, 2, 3),
        (4, 4, 5),
        (6, 6, 7),
        (0, 0, 2),
        (4, 4, 6),
        (0, 0, 4),
    ] {
        vop(
            b,
            format!("v_add_f32_e32 v{}, v{}, v{}", p(dst), p(a), p(c)),
            &[p(dst)],
            &[p(a), p(c)],
            &[],
        )?;
    }
    let x = p(0);
    let store = format!("{tag}_store");
    sop(b, format!("s_cmp_eq_u32 s{S_HC}, 0"), &[], &[S_HC])?;
    op(b, format!("s_cbranch_scc1 {store}"), &[], &[])?;
    let e = |n: u8| V_EP + n;
    bf16_round(b, x, [e(0), e(1), e(2), e(3)])?;
    vop(
        b,
        format!("v_mul_f32_e32 v{x}, s{S_SCALE}, v{x}"),
        &[x],
        &[x],
        &[S_SCALE],
    )?;
    bf16_round(b, x, [e(0), e(1), e(2), e(3)])?;
    silu(b, x, [e(0), e(1), e(2), e(3), e(4), e(5)])?;
    bf16_round(b, x, [e(0), e(1), e(2), e(3)])?;
    b.label(&store)?;
    vop(
        b,
        format!("v_cmp_gt_i32_e32 vcc_lo, s{S_M}, v{V_IROW}"),
        &[],
        &[V_IROW],
        &[S_M],
    )?;
    op(b, "s_and_b32 exec_lo, exec_lo, vcc_lo", &[], &[])?;
    mem(
        b,
        format!("global_store_b32 v{V_IY}, v{x}, s[{S_Y}:{}]", S_Y + 1),
        &[],
        &[v(V_IY), v(x), sr(S_Y, 2)],
        MemoryClass::VmemStore,
    )?;
    b.wait_all()?;
    op(b, "s_branch .Lq8k8_end", &[], &[])
}

fn vrange(base: u8, len: u8) -> String {
    format!("v[{base}:{}]", u16::from(base) + u16::from(len) - 1)
}

// ================================================================== k320
// VGPRs.
/// `(tid >> 5) * 384`: the wave's LDS region (scale reads).
const W_BASED: u8 = 2;
/// `+ lane`: the int8 reads.
const W_BASEQ: u8 = 3;
/// `+ lane*4`: the staging stores.
const W_STA: u8 = 4;
const W_LANE4: u8 = 5;
/// `min(lane, 20) * 4`: the third staging load.
const W_LANE4C: u8 = 6;
const W_ADDR: u8 = 7;
const W_XOFF: u8 = 12;
const W_YOFF: u8 = 20;
/// Global row ring: four rows of three dwords.
const W_G: u8 = 28;
const W_OFFA: u8 = 40;
const W_OFFB: u8 = 41;
const W_D: u8 = 44;
const W_Q: u8 = 54;
const W_TMP: u8 = 64;
/// Four `sum[col]` sets of eight, one per row in flight.
const W_SUM: u8 = 72;
/// `x[block][col]` at `+ 8*block + col`.
const W_X: u8 = 104;
/// VGPR count of the k320 plan.
pub const K320_VGPRS: u16 = 184;

// SGPRs (A at s[4:5], x at s[6:7], y at s[8:9]).
const T_M: u8 = 10;
const T_B: u8 = 11;
const T_WAVE: u8 = 12;
const T_ROW0: u8 = 13;
const T_MM1: u8 = 14;
const T_T0: u8 = 15;
const T_LANE0: u8 = 16;
/// `rowoff[q] = min(row0 + q, M-1) * 340` of the wave's eight rows.
const T_ROWOFF: u8 = 17;

fn k320_x(blk: u8, c: u8) -> u8 {
    W_X + 8 * blk + c
}
fn k320_sum(set: u8, c: u8) -> u8 {
    W_SUM + 8 * set + c
}

fn k320_plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(K320_VGPRS, 40)?;
    for (name, base, len) in [
        ("kernarg_ptr", S_KARG, 2),
        ("wg_x", S_WG, 1),
        ("a_x_ptrs", S_A, 4),
        ("y_ptr", S_Y, 2),
        ("m_b", T_M, 2),
    ] {
        declare(&mut p, name, Kind::S, base, len)?;
    }
    for n in T_WAVE..T_ROWOFF + 8 {
        declare(&mut p, "scalar", Kind::S, n, 1)?;
    }
    for (name, base, len) in [
        ("tid", V_TID, 1),
        ("lane", V_LANE, 1),
        ("lds_base_d", W_BASED, 1),
        ("lds_base_q", W_BASEQ, 1),
        ("lds_stage_addr", W_STA, 1),
        ("lane4", W_LANE4, 1),
        ("lane4_clamped", W_LANE4C, 1),
        ("bpermute_addr", W_ADDR, 4),
        ("x_offset", W_XOFF, 8),
        ("y_offset", W_YOFF, 8),
        ("row_ring", W_G, 12),
        ("row_offset_a", W_OFFA, 1),
        ("row_offset_b", W_OFFB, 1),
        ("scales", W_D, 10),
        ("weights", W_Q, 10),
        ("tree_tmp", W_TMP, 8),
        ("sums", W_SUM, 32),
        ("activations", W_X, 80),
    ] {
        declare(&mut p, name, Kind::V, base, len)?;
    }
    Ok(p)
}

fn k320_spec() -> KernelSpec {
    KernelSpec {
        kernel_id: "qwen4_lane_rows_q8".into(),
        variant: "k320_xbatch".into(),
        arch: Arch::Gfx1151,
        symbol: K320_SYMBOL.into(),
        kernargs: KernargLayout::new(32)
            .pointer_access("A", 0, Access::ReadOnly)
            .pointer_access("x", 8, Access::ReadOnly)
            .pointer_access("y", 16, Access::WriteOnly)
            .hidden("M", 24, 4, "by_value")
            .hidden("B", 28, 4, "by_value"),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 64,
        group_segment_fixed_size: K320_LDS_BYTES,
        wave32: true,
        cu_mode: false,
    }
}

fn k320() -> Result<Emitted, String> {
    let mut b = Builder::new(k320_spec(), k320_plan()?);
    b.enable_delay_alu();
    let slots = [
        b.lds_slot("rows0", 0, K320_SLOT)?,
        b.lds_slot("rows1", K320_SLOT, K320_SLOT)?,
    ];
    k320_prologue(&mut b)?;
    for nb in 1..=MAX_COLUMNS {
        sop(&mut b, format!("s_cmp_eq_u32 s{T_B}, {nb}"), &[], &[T_B])?;
        op(&mut b, format!("s_cbranch_scc1 .Lq8k320_b{nb}"), &[], &[])?;
    }
    op(&mut b, "s_branch .Lq8k320_end", &[], &[])?;
    for nb in 1..=MAX_COLUMNS {
        k320_section(&mut b, nb, slots)?;
    }
    b.label(".Lq8k320_end")?;
    op(&mut b, "s_endpgm", &[], &[])?;
    b.finish()
}

fn k320_prologue(b: &mut Builder) -> R {
    smem(b, S_A, 4, S_KARG, 0)?;
    smem(b, S_Y, 2, S_KARG, 0x10)?;
    smem(b, T_M, 2, S_KARG, 0x18)?;
    vop(
        b,
        format!("v_and_b32_e32 v{V_LANE}, 31, v{V_TID}"),
        &[V_LANE],
        &[V_TID],
        &[],
    )?;
    vop(
        b,
        format!("v_lshrrev_b32_e32 v{W_BASED}, 5, v{V_TID}"),
        &[W_BASED],
        &[V_TID],
        &[],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{T_WAVE}, v{W_BASED}"),
        &[s(T_WAVE)],
        &[v(W_BASED)],
    )?;
    vop(
        b,
        format!(
            "v_mul_u32_u24_e32 v{W_BASED}, {}, v{W_BASED}",
            lit(K320_WAVE_REGION)
        ),
        &[W_BASED],
        &[W_BASED],
        &[],
    )?;
    vop(
        b,
        format!("v_add_nc_u32_e32 v{W_BASEQ}, v{W_BASED}, v{V_LANE}"),
        &[W_BASEQ],
        &[W_BASED, V_LANE],
        &[],
    )?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{W_LANE4}, 2, v{V_LANE}"),
        &[W_LANE4],
        &[V_LANE],
        &[],
    )?;
    vop(
        b,
        format!("v_add_nc_u32_e32 v{W_STA}, v{W_BASED}, v{W_LANE4}"),
        &[W_STA],
        &[W_BASED, W_LANE4],
        &[],
    )?;
    vop(
        b,
        format!("v_min_u32_e32 v{W_LANE4C}, 20, v{V_LANE}"),
        &[W_LANE4C],
        &[V_LANE],
        &[],
    )?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{W_LANE4C}, 2, v{W_LANE4C}"),
        &[W_LANE4C],
        &[W_LANE4C],
        &[],
    )?;
    bpermute_addresses(b, W_ADDR, V_LANE)?;
    vop(
        b,
        format!("v_cmp_eq_u32_e32 vcc_lo, 0, v{V_LANE}"),
        &[],
        &[V_LANE],
        &[],
    )?;
    op(
        b,
        format!("s_mov_b32 s{T_LANE0}, vcc_lo"),
        &[s(T_LANE0)],
        &[],
    )?;
    // `ds_load_u16_d16` keeps the high half of its destination.
    for blk in 0..10u8 {
        vop(
            b,
            format!("v_mov_b32_e32 v{}, 0", W_D + blk),
            &[W_D + blk],
            &[],
            &[],
        )?;
    }
    b.wait_all()?;
    // The workgroup's first row must exist.
    sop(
        b,
        format!("s_lshl_b32 s{T_T0}, s{S_WG}, 4"),
        &[T_T0],
        &[S_WG],
    )?;
    sop(
        b,
        format!("s_cmp_ge_i32 s{T_T0}, s{T_M}"),
        &[],
        &[T_T0, T_M],
    )?;
    op(b, "s_cbranch_scc1 .Lq8k320_end", &[], &[])?;
    sop(
        b,
        format!("s_lshl_b32 s{T_ROW0}, s{T_WAVE}, 3"),
        &[T_ROW0],
        &[T_WAVE],
    )?;
    sop(
        b,
        format!("s_add_i32 s{T_ROW0}, s{T_ROW0}, s{T_T0}"),
        &[T_ROW0],
        &[T_ROW0, T_T0],
    )?;
    sop(
        b,
        format!("s_add_i32 s{T_MM1}, s{T_M}, -1"),
        &[T_MM1],
        &[T_M],
    )?;
    for q in 0..8u8 {
        let ro = T_ROWOFF + q;
        sop(
            b,
            format!("s_add_i32 s{ro}, s{T_ROW0}, {q}"),
            &[ro],
            &[T_ROW0],
        )?;
        sop(
            b,
            format!("s_min_i32 s{ro}, s{ro}, s{T_MM1}"),
            &[ro],
            &[ro, T_MM1],
        )?;
        sop(
            b,
            format!("s_mul_i32 s{ro}, s{ro}, {}", lit(340)),
            &[ro],
            &[ro],
        )?;
    }
    Ok(())
}

/// Global loads of row `q` (clamped) into ring slot `q % 4`: dwords lane,
/// lane+32 and min(lane, 20)+64.
fn k320_row_load(b: &mut Builder, q: u8) -> R {
    let g = W_G + 3 * (q % 4);
    let ro = T_ROWOFF + q;
    vop(
        b,
        format!("v_add_nc_u32_e32 v{W_OFFA}, s{ro}, v{W_LANE4}"),
        &[W_OFFA],
        &[W_LANE4],
        &[ro],
    )?;
    vop(
        b,
        format!("v_add_nc_u32_e32 v{W_OFFB}, s{ro}, v{W_LANE4C}"),
        &[W_OFFB],
        &[W_LANE4C],
        &[ro],
    )?;
    gload(b, "global_load_b32", g, W_OFFA, S_A, 0)?;
    gload(b, "global_load_b32", g + 1, W_OFFA, S_A, 128)?;
    gload(b, "global_load_b32", g + 2, W_OFFB, S_A, 256)
}

/// Stage row `q` from its ring slot into LDS slot `q % 2` (wave-local region).
fn k320_stage(b: &mut Builder, slots: [usize; 2], q: u8) -> R {
    let g = W_G + 3 * (q % 4);
    let base = u32::from(q % 2) * K320_SLOT;
    for dword in 0..3u8 {
        ds_st(
            b,
            slots[usize::from(q % 2)],
            W_STA,
            g + dword,
            base + 128 * u32::from(dword),
        )?;
    }
    Ok(())
}

/// Row `r` from LDS slot `r % 2` against every column into sum set `r % 4`;
/// then the lane tree and lane 0's stores of valid rows.
fn k320_row(b: &mut Builder, nb: u8, slot: usize, r: u8) -> R {
    let base = u32::from(r % 2) * K320_SLOT;
    let set = r % 4;
    for blk in 0..10u8 {
        ds_ld(
            b,
            slot,
            "ds_load_u16_d16",
            W_D + blk,
            W_BASED,
            base + 34 * u32::from(blk),
            true,
        )?;
        ds_ld(
            b,
            slot,
            "ds_load_i8",
            W_Q + blk,
            W_BASEQ,
            base + 34 * u32::from(blk) + 2,
            false,
        )?;
    }
    for blk in 0..10u8 {
        dequant(b, W_D + blk, W_Q + blk)?;
        for c in 0..nb {
            let (sum, x, d) = (k320_sum(set, c), k320_x(blk, c), W_D + blk);
            if blk == 0 {
                vop(
                    b,
                    format!("v_fma_f32 v{sum}, v{x}, v{d}, 0"),
                    &[sum],
                    &[x, d],
                    &[],
                )?;
            } else {
                vop(
                    b,
                    format!("v_fmac_f32_e32 v{sum}, v{x}, v{d}"),
                    &[sum],
                    &[sum, x, d],
                    &[],
                )?;
            }
        }
    }
    Ok(())
}

fn k320_section(b: &mut Builder, nb: u8, slots: [usize; 2]) -> R {
    let tag = format!(".Lq8k320_b{nb}");
    b.label(&tag)?;
    for c in 0..nb {
        if c == 0 {
            vop(
                b,
                format!("v_mov_b32_e32 v{W_XOFF}, v{W_LANE4}"),
                &[W_XOFF],
                &[W_LANE4],
                &[],
            )?;
        } else {
            vop(
                b,
                format!(
                    "v_add_nc_u32_e32 v{}, {}, v{W_LANE4}",
                    W_XOFF + c,
                    lit(1280 * u32::from(c))
                ),
                &[W_XOFF + c],
                &[W_LANE4],
                &[],
            )?;
        }
        // y byte offset of (column c, this wave's first row).
        sop(
            b,
            format!("s_mul_i32 s{T_T0}, s{T_M}, {c}"),
            &[T_T0],
            &[T_M],
        )?;
        sop(
            b,
            format!("s_add_i32 s{T_T0}, s{T_T0}, s{T_ROW0}"),
            &[T_T0],
            &[T_T0, T_ROW0],
        )?;
        sop(
            b,
            format!("s_lshl_b32 s{T_T0}, s{T_T0}, 2"),
            &[T_T0],
            &[T_T0],
        )?;
        vop(
            b,
            format!("v_mov_b32_e32 v{}, s{T_T0}", W_YOFF + c),
            &[W_YOFF + c],
            &[],
            &[T_T0],
        )?;
    }
    for q in 0..4u8 {
        k320_row_load(b, q)?;
    }
    for blk in 0..10u8 {
        for c in 0..nb {
            gload(
                b,
                "global_load_b32",
                k320_x(blk, c),
                W_XOFF + c,
                S_X,
                128 * u32::from(blk),
            )?;
        }
    }
    k320_stage(b, slots, 0)?;
    b.barrier(&[Transition::Ready(slots[0])])?;
    for r in 0..8u8 {
        let par = usize::from(r % 2);
        if r + 1 < 8 {
            k320_stage(b, slots, r + 1)?;
        }
        if r + 4 < 8 {
            k320_row_load(b, r + 4)?;
        }
        // Row r reuses the sum set that row r-4 stored. Those stores may be
        // skipped (row >= M) and retire out of order, so only vscnt(0)
        // releases their sources; four rows later they have long retired.
        if r >= 4 {
            b.wait(Counter::Vs, 0)?;
        }
        k320_row(b, nb, slots[par], r)?;
        let mut transitions = vec![Transition::Retire(slots[par])];
        if r + 1 < 8 {
            transitions.push(Transition::Ready(slots[1 - par]));
        }
        b.barrier(&transitions)?;
        let set = r % 4;
        let values: Vec<u8> = (0..nb).map(|c| k320_sum(set, c)).collect();
        lane_tree(b, &values, W_TMP, W_ADDR)?;
        let skip = format!("{tag}_r{r}_skip");
        sop(
            b,
            format!("s_add_i32 s{T_T0}, s{T_ROW0}, {r}"),
            &[T_T0],
            &[T_ROW0],
        )?;
        sop(
            b,
            format!("s_cmp_ge_i32 s{T_T0}, s{T_M}"),
            &[],
            &[T_T0, T_M],
        )?;
        op(b, format!("s_cbranch_scc1 {skip}"), &[], &[])?;
        op(
            b,
            format!("s_mov_b32 exec_lo, s{T_LANE0}"),
            &[],
            &[s(T_LANE0)],
        )?;
        for c in 0..nb {
            mem(
                b,
                format!(
                    "global_store_b32 v{}, v{}, s[{S_Y}:{}]{}",
                    W_YOFF + c,
                    k320_sum(set, c),
                    S_Y + 1,
                    off(4 * u32::from(r))
                ),
                &[],
                &[v(W_YOFF + c), v(k320_sum(set, c)), sr(S_Y, 2)],
                MemoryClass::VmemStore,
            )?;
        }
        op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
        b.label(&skip)?;
    }
    b.wait_all()?;
    op(b, "s_branch .Lq8k320_end", &[], &[])
}

/// Both Q8 row exports of the lane step, in table order.
pub fn build_gfx1151() -> Result<Vec<Emitted>, String> {
    Ok(vec![k8()?, k320()?])
}

// ============================================================ fixed-eight row tiles
// Historical r16/r32 wide twins are removed (see the module docs). The fixed-eight
// schedule below is the shared t8 row-tile path only; narrow exports keep their
// bytes and their register allocation.

/// SGPR budget of every row-tile entry (the plan's high water is 49).
pub const WIDE_SGPRS: u16 = 56;
/// Ordinary row tiles: eight output columns (waves) x one 8-row x 256-K
/// X tile in LDS. HC k8: two columns x eight partitions, a 2-row x 8-part x
/// 256-K tile.
pub const WIDE_LDS_BYTES: u32 = 8 * 8 * 32 * 4;
pub const WIDE_K8_LDS_BYTES: u32 = 2 * 8 * 8 * 32 * 4;

/// Which incumbent recipe (and ABI) a row-tile entry reproduces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WideKind {
    /// HC read down `gemv_q8_0_k8`: eight single-chain K/8 partitions, fixed fold, HC epilogue.
    K8,
    /// Single-chain `gemv_q8_0_row` recipe, bound for K = 320 (`gemv_q8_0_k320_staged`).
    K320,
    /// Single-chain `gemv_q8_0_row` recipe, bound for K = 2560 (`gemv_q8_0_k2560_staged`).
    K2560,
    /// Two K = 2560 single-chain matrices of one activation (shared gate/up).
    K2560X2,
    /// Four-chain small-K `gemv_q8_0_wide` (and, at K = 640, `gemv_q8_0_wide_k640_staged`).
    Wide,
}

/// One fixed-eight row-tile entry: recipe with physical capacity 8. Total logical
/// rows live in the `R` kernarg (1..=64); `cap` is frozen at 8.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WideSpec {
    pub kind: WideKind,
    pub arch: Arch,
}

impl WideSpec {
    /// Workgroup id x SGPR: right after the DECLARED user SGPRs (s2; gfx1100 pads them so that
    /// user + system >= 16, LLVM `FeatureUserSGPRInit16Bug`, which moves the ids to s14/s15).
    pub const fn s_wg(&self) -> u8 {
        self.arch
            .declared_user_sgprs(WIDE_USER_SGPRS, WIDE_SYSTEM_SGPRS)
    }
    /// System workgroup id y, the SGPR after [`WideSpec::s_wg`].
    pub const fn s_wg_y(&self) -> u8 {
        self.s_wg() + 1
    }
    /// True when the padded ids sit on the gfx1151 `WS_HC`/`WS_WAVE` numbers (s14/s15).
    const fn ids_moved(&self) -> bool {
        self.s_wg() != WIDE_USER_SGPRS
    }
    /// This wave's index (s15 on gfx1151, a free SGPR above `WS_ROWOFF` on gfx1100).
    pub const fn s_wave(&self) -> u8 {
        if self.ids_moved() {
            WS_WAVE_PADDED
        } else {
            WS_WAVE
        }
    }
    /// HC activation flag (s14 on gfx1151, a free SGPR above `WS_ROWOFF` on gfx1100).
    pub const fn s_hc(&self) -> u8 {
        if self.ids_moved() {
            WS_HC_PADDED
        } else {
            WS_HC
        }
    }
    /// Physical tile rows.
    pub const fn cap(&self) -> u8 {
        8
    }
    fn tag(&self) -> &'static str {
        match self.kind {
            WideKind::K8 => "k8",
            WideKind::K320 => "k320",
            WideKind::K2560 => "k2560",
            WideKind::K2560X2 => "k2560_x2",
            WideKind::Wide => "wide",
        }
    }
    fn hc(&self) -> bool {
        self.kind == WideKind::K8
    }
    fn four(&self) -> bool {
        self.kind == WideKind::Wide
    }
    fn x2(&self) -> bool {
        self.kind == WideKind::K2560X2
    }
    pub fn kernarg_bytes(&self) -> u32 {
        match self.kind {
            WideKind::K8 => 44,
            WideKind::K2560X2 => 56,
            _ => 36,
        }
    }
    pub fn workgroup_size(&self) -> u16 {
        if self.hc() {
            512
        } else {
            256
        }
    }
    pub fn lds_bytes(&self) -> u32 {
        if self.hc() {
            WIDE_K8_LDS_BYTES
        } else {
            WIDE_LDS_BYTES
        }
    }
    /// Declared VGPR budget (fixed-eight design ceiling): 80 for one chain per
    /// row (ordinary and HC), 80 for four chains per row at cap 8.
    pub fn vgpr_budget(&self) -> u16 {
        80
    }
    /// Persistent accumulator VGPRs per lane.
    pub fn accumulators(&self) -> usize {
        usize::from(self.cap()) * if self.four() { 4 } else { 1 }
    }
    /// VGPR high water (`WV_ACC + accumulators`), in usize arithmetic.
    pub fn vgpr_high_water(&self) -> usize {
        usize::from(WV_ACC) + self.accumulators()
    }
    /// Activation rows per X stage.
    fn rows_per_tile(&self) -> u32 {
        if self.hc() {
            2
        } else {
            8
        }
    }
    fn tiles(&self) -> u32 {
        u32::from(self.cap()) / self.rows_per_tile()
    }
    fn label(&self) -> String {
        format!(".Lq8w_{}t8", self.tag())
    }
}

/// Operand spelling of the four-chain recipe, pinned to the decode object
/// that `Gpu::gemv_q8_0` selects on gfx1151 (one-chain kernels have one
/// spelling, `Generic` is passed for them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    /// `gemv_q8_0_wide` (every K but 640): quad blocks `fmac(acc, dq, x)`;
    /// remainder blocks `p = x * dq`, `acc = acc + p`; fold
    /// `(a3 + a2) + (a1 + a0)` spelled `t1 = a1 + a0`, `t2 = a3 + a2`, `t2 + t1`.
    Generic,
    /// `gemv_q8_0_wide_k640_staged` (K = 640, no remainder blocks): quad
    /// blocks `fmac(acc, x, dq)`; fold `(a0 + a1) + (a2 + a3)`.
    K640,
}

// SGPRs (`S_KARG` = 0, `S_WG` = 2 as the narrow entries).
/// `A` at s[4:5], `x` at s[6:7] (one b128 load), `y` at s[8:9]; `M`, `K`, `R`.
/// For the x2 entry these are the workgroup's selected matrix.
const WS_A: u8 = 4;
const WS_X: u8 = 6;
const WS_Y: u8 = 8;
const WS_M: u8 = 10;
const WS_K: u8 = 11;
const WS_R: u8 = 12;
const WS_SCALE: u8 = 13;
const WS_HC: u8 = 14;
const WS_WAVE: u8 = 15;
/// gfx1100 relocations of `WS_WAVE`/`WS_HC`: s14/s15 are the padded workgroup ids there.
const WS_WAVE_PADDED: u8 = 49;
const WS_HC_PADDED: u8 = 50;
const WIDE_USER_SGPRS: u8 = 2;
const WIDE_SYSTEM_SGPRS: u8 = 2;
/// First matrix-local output column of the workgroup.
const WS_MBASE: u8 = 16;
/// This wave's output column, and whether it exists (`COL < M`).
const WS_COL: u8 = 17;
const WS_LIVE: u8 = 18;
/// Weight bytes of one row (ordinary) or one K/8 partition (HC).
const WS_ROWB: u8 = 19;
/// Q8 blocks per row (ordinary) or partition (HC); its full 8-block tiles and the tail tile's blocks.
const WS_NBLK: u8 = 20;
const WS_NFULL: u8 = 21;
const WS_TAIL: u8 = 22;
const WS_IT: u8 = 23;
/// x2 raw kernargs: A0/A1, x/y0, y1, M0/M1, K/R.
const WS_RA: u8 = 24;
const WS_RXY: u8 = 28;
const WS_RY1: u8 = 32;
const WS_RM: u8 = 34;
const WS_RKR: u8 = 36;
const WS_T0: u8 = 38;
const WS_T1: u8 = 39;
const WS_LANE0: u8 = 40;
/// `M * 4` and `COL * 4` (ordinary y offsets).
const WS_MB: u8 = 41;
const WS_COL4: u8 = 42;
const WS_KX4: u8 = 43;
/// Producer X byte offset of this wave's row/partition in row tile 0.
const WS_XTB: u8 = 44;
/// Producer row within a tile: the wave (ordinary) or the column half `ml` (HC).
const WS_PROW: u8 = 45;
const WS_MASK: u8 = 46;
/// HC partition (`wave & 7`).
const WS_PART: u8 = 47;
/// Byte offset of this wave's weight row (partition) from `A`.
const WS_ROWOFF: u8 = 48;

// VGPRs (`V_TID` = 0, `V_LANE` = 1).
const WV_ADDR: u8 = 2;
/// LDS producer store address `wave*1024 + lane*4`, consumer read base.
const WV_LDST: u8 = 6;
const WV_LDRD: u8 = 7;
/// `lane*4 + k0*4`: the current K tile's lane offset into an activation row (partition).
const WV_XOFF: u8 = 8;
const WV_XADDR: u8 = 9;
const WV_WOFFD: u8 = 10;
const WV_WOFFQ: u8 = 11;
/// The remainder blocks' rounded product (four chains).
const WV_TMP: u8 = 12;
const WV_VT: u8 = 13;
/// HC: lane 0's partial store base `(ml*cap*8 + part)*4`.
const WV_PST: u8 = 14;
const WV_EY: u8 = 15;
/// `dq[j]` of the current K tile (HC epilogue: the eight partials).
const WV_DQ: u8 = 16;
/// `q[j]` (ordinary epilogue: store addresses; HC epilogue: scratch).
const WV_Q: u8 = 24;
/// X values (producer staging, consumer row vector, lane-tree temporaries).
const WV_XS: u8 = 32;
const WV_ACC: u8 = 40;

/// Accumulator of activation row `row`, chain `chain` (`0` for one chain).
fn wacc(w: WideSpec, row: u32, chain: u8) -> u8 {
    let n = usize::from(WV_ACC)
        + if w.four() {
            4 * row as usize + usize::from(chain)
        } else {
            row as usize
        };
    u8::try_from(n).expect("accumulator index validated by wide_plan")
}

fn wide_plan(w: WideSpec) -> Result<RegPlan, String> {
    let hw = w.vgpr_high_water();
    if hw > 256 {
        return Err(format!(
            "qwen4 rowtile {} needs {hw} VGPRs (> 256)",
            w.tag()
        ));
    }
    if hw > usize::from(w.vgpr_budget()) {
        return Err(format!(
            "qwen4 rowtile {} needs {hw} VGPRs (> budget {})",
            w.tag(),
            w.vgpr_budget()
        ));
    }
    let mut p = RegPlan::new(w.vgpr_budget(), WIDE_SGPRS)?;
    for (name, base, len) in [
        ("kernarg_ptr", S_KARG, 2),
        ("wg_x", w.s_wg(), 1),
        ("a_x_ptrs", WS_A, 4),
        ("y_ptr", WS_Y, 2),
        ("m_k", WS_M, 2),
    ] {
        declare(&mut p, name, Kind::S, base, len)?;
    }
    declare(&mut p, "wg_y", Kind::S, w.s_wg_y(), 1)?;
    if w.hc() {
        declare(&mut p, "r_scale", Kind::S, WS_R, 2)?;
        declare(&mut p, "hc_act", Kind::S, w.s_hc(), 1)?;
    } else {
        declare(&mut p, "r", Kind::S, WS_R, 1)?;
    }
    if w.ids_moved() {
        declare(&mut p, "scalar", Kind::S, w.s_wave(), 1)?;
    }
    for n in WS_WAVE..=WS_IT {
        if !w.ids_moved() || n > w.s_wg_y() {
            declare(&mut p, "scalar", Kind::S, n, 1)?;
        }
    }
    if w.x2() {
        for (name, base, len) in [
            ("x2_a", WS_RA, 4),
            ("x2_x_y0", WS_RXY, 4),
            ("x2_y1", WS_RY1, 2),
            ("x2_m", WS_RM, 2),
            ("x2_k_r", WS_RKR, 2),
        ] {
            declare(&mut p, name, Kind::S, base, len)?;
        }
    }
    for n in WS_T0..=WS_ROWOFF {
        declare(&mut p, "scalar", Kind::S, n, 1)?;
    }
    let acc =
        u8::try_from(w.accumulators()).map_err(|_| "accumulator count past 255".to_string())?;
    for (name, base, len) in [
        ("tid", V_TID, 1),
        ("lane", V_LANE, 1),
        ("bpermute_addr", WV_ADDR, 4),
        ("lds_store", WV_LDST, 1),
        ("lds_read", WV_LDRD, 1),
        ("x_offset", WV_XOFF, 1),
        ("x_addr", WV_XADDR, 1),
        ("w_offset_d", WV_WOFFD, 1),
        ("w_offset_q", WV_WOFFQ, 1),
        ("product", WV_TMP, 1),
        ("temp", WV_VT, 1),
        ("partial_store", WV_PST, 1),
        ("y_offset", WV_EY, 1),
        ("dq", WV_DQ, 8),
        ("q", WV_Q, 8),
        ("x_slots", WV_XS, 8),
        ("accumulators", WV_ACC, acc),
    ] {
        declare(&mut p, name, Kind::V, base, len)?;
    }
    Ok(p)
}

fn wide_kspec(w: WideSpec) -> KernelSpec {
    let kernargs = match w.kind {
        WideKind::K8 => KernargLayout::new(44)
            .pointer_access("A", 0, Access::ReadOnly)
            .pointer_access("x", 8, Access::ReadOnly)
            .pointer_access("y", 16, Access::WriteOnly)
            .hidden("M", 24, 4, "by_value")
            .hidden("K", 28, 4, "by_value")
            .hidden("R", 32, 4, "by_value")
            .hidden("scale", 36, 4, "by_value")
            .hidden("hc_act", 40, 4, "by_value"),
        WideKind::K2560X2 => KernargLayout::new(56)
            .pointer_access("A0", 0, Access::ReadOnly)
            .pointer_access("A1", 8, Access::ReadOnly)
            .pointer_access("x", 16, Access::ReadOnly)
            .pointer_access("y0", 24, Access::WriteOnly)
            .pointer_access("y1", 32, Access::WriteOnly)
            .hidden("M0", 40, 4, "by_value")
            .hidden("M1", 44, 4, "by_value")
            .hidden("K", 48, 4, "by_value")
            .hidden("R", 52, 4, "by_value"),
        _ => KernargLayout::new(36)
            .pointer_access("A", 0, Access::ReadOnly)
            .pointer_access("x", 8, Access::ReadOnly)
            .pointer_access("y", 16, Access::WriteOnly)
            .hidden("M", 24, 4, "by_value")
            .hidden("K", 28, 4, "by_value")
            .hidden("R", 32, 4, "by_value"),
    };
    KernelSpec {
        kernel_id: "qwen4_lane_rows_q8".into(),
        variant: format!("rowtiles_{}_t8", w.tag()),
        arch: w.arch,
        symbol: rowtile_symbol(w.kind).into(),
        kernargs,
        user_sgpr_count: WIDE_USER_SGPRS,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: w.workgroup_size(),
        group_segment_fixed_size: w.lds_bytes(),
        wave32: true,
        cu_mode: false,
    }
}

/// Explicit object-derived arithmetic selection for a sibling architecture.
/// `Small` always uses the generic four-chain DAG, including K=640.
#[derive(Clone, Copy)]
pub(super) enum RowtileRecipe {
    Hc,
    Plain,
    PlainX2,
    Small,
}

/// Runtime K domain, independent of the export's historical shape label.
#[derive(Clone, Copy)]
pub(super) enum RowtileDomain {
    EightPartitions,
    Exact(u32),
    Small,
    Large,
}

/// A sibling owns its architecture, recipe, domain and external identity.
#[derive(Clone, Copy)]
pub(super) struct RowtileTarget {
    pub arch: Arch,
    pub symbol: &'static str,
    pub recipe: RowtileRecipe,
    pub domain: RowtileDomain,
}

/// Reuse the fixed8 scheduler, not Halo's architecture/shape dispatch.
pub(super) fn emit_rowtiles_for(target: RowtileTarget) -> Result<Emitted, String> {
    let kind = match target.recipe {
        RowtileRecipe::Hc => WideKind::K8,
        RowtileRecipe::Plain => WideKind::K2560,
        RowtileRecipe::PlainX2 => WideKind::K2560X2,
        RowtileRecipe::Small => WideKind::Wide,
    };
    wide_impl_target(
        WideSpec {
            kind,
            arch: target.arch,
        },
        Some(target),
    )
}

fn wide_impl(w: WideSpec) -> Result<Emitted, String> {
    wide_impl_target(w, None)
}

fn wide_impl_target(w: WideSpec, target: Option<RowtileTarget>) -> Result<Emitted, String> {
    let mut spec = wide_kspec(w);
    if let Some(target) = target {
        debug_assert_eq!(w.arch, target.arch);
        spec.kernel_id = "qwen4_lane_rows_q8_gfx1100".into();
        spec.variant = format!("rowtiles_{}", target.symbol);
        spec.arch = target.arch;
        spec.symbol = target.symbol.into();
    }
    let ids = spec.workgroup_ids(false)?;
    if ids.x != crate::arch::WgIdSource::Sgpr(w.s_wg())
        || ids.y != crate::arch::WgIdSource::Sgpr(w.s_wg_y())
    {
        return Err(format!("qwen4 rowtile workgroup ids moved: {ids:?}"));
    }
    let mut b = Builder::new(spec, wide_plan(w)?);
    b.enable_delay_alu();
    let slot = b.lds_slot("x_tile", 0, w.lds_bytes())?;
    let lbl = target.map_or_else(|| w.label(), |t| format!(".L{}_body", t.symbol));
    let end = format!("{lbl}_end");
    wide_prologue(&mut b, w, target.map(|t| t.domain), &end)?;
    // +0 accumulators (the incumbents' `0.0f`); rows >= R are never stored.
    for i in 0..w.accumulators() {
        let a = WV_ACC + i as u8;
        vop(&mut b, format!("v_mov_b32_e32 v{a}, 0"), &[a], &[], &[])?;
    }
    let tails: Vec<u8> = (1..8).collect();
    match w.kind {
        WideKind::Wide if target.is_none() => {
            let k640 = format!("{lbl}_k640");
            sop(
                &mut b,
                format!("s_cmp_eq_u32 s{WS_K}, {}", lit(640)),
                &[],
                &[WS_K],
            )?;
            op(&mut b, format!("s_cbranch_scc1 {k640}"), &[], &[])?;
            let g = format!("{lbl}_g");
            wide_traverse(&mut b, w, slot, Order::Generic, &tails, &g)?;
            wide_epilogue(&mut b, w, Order::Generic, &g, &end)?;
            op(&mut b, format!("s_branch {end}"), &[], &[])?;
            // K = 640: 20 blocks, two full tiles and a 4-block (all-quad) tail tile.
            b.label(&k640)?;
            let k = format!("{lbl}_k");
            wide_traverse(&mut b, w, slot, Order::K640, &[4], &k)?;
            wide_epilogue(&mut b, w, Order::K640, &k, &end)?;
        }
        WideKind::K8 => {
            wide_traverse(&mut b, w, slot, Order::Generic, &tails, &lbl)?;
            wide_hc_epilogue(&mut b, w, &lbl, &end)?;
        }
        _ => {
            wide_traverse(&mut b, w, slot, Order::Generic, &tails, &lbl)?;
            wide_epilogue(&mut b, w, Order::Generic, &lbl, &end)?;
        }
    }
    b.label(&end)?;
    op(&mut b, "s_endpgm", &[], &[])?;
    b.finish()
}

/// Kernarg loads, lane/wave constants; the uniform refusals (before any
/// barrier); the x2 matrix selection; the wave's column, weight and X
/// offsets, the tile counts and the LDS addresses.
fn wide_prologue(b: &mut Builder, w: WideSpec, domain: Option<RowtileDomain>, end: &str) -> R {
    let (wg_x, wg_y, wave, hc_act) = (w.s_wg(), w.s_wg_y(), w.s_wave(), w.s_hc());
    match w.kind {
        WideKind::K2560X2 => {
            smem(b, WS_RA, 4, S_KARG, 0)?;
            smem(b, WS_RXY, 4, S_KARG, 0x10)?;
            smem(b, WS_RY1, 2, S_KARG, 0x20)?;
            smem(b, WS_RM, 2, S_KARG, 0x28)?;
            smem(b, WS_RKR, 2, S_KARG, 0x30)?;
        }
        WideKind::K8 => {
            smem(b, WS_A, 4, S_KARG, 0)?;
            smem(b, WS_Y, 2, S_KARG, 0x10)?;
            smem(b, WS_M, 2, S_KARG, 0x18)?;
            smem(b, WS_R, 2, S_KARG, 0x20)?;
            smem(b, hc_act, 1, S_KARG, 0x28)?;
        }
        _ => {
            smem(b, WS_A, 4, S_KARG, 0)?;
            smem(b, WS_Y, 2, S_KARG, 0x10)?;
            smem(b, WS_M, 2, S_KARG, 0x18)?;
            smem(b, WS_R, 1, S_KARG, 0x20)?;
        }
    }
    vop(
        b,
        format!("v_and_b32_e32 v{V_LANE}, 31, v{V_TID}"),
        &[V_LANE],
        &[V_TID],
        &[],
    )?;
    vop(
        b,
        format!("v_lshrrev_b32_e32 v{WV_VT}, 5, v{V_TID}"),
        &[WV_VT],
        &[V_TID],
        &[],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{wave}, v{WV_VT}"),
        &[s(wave)],
        &[v(WV_VT)],
    )?;
    bpermute_addresses(b, WV_ADDR, V_LANE)?;
    vop(
        b,
        format!("v_cmp_eq_u32_e32 vcc_lo, 0, v{V_LANE}"),
        &[],
        &[V_LANE],
        &[],
    )?;
    op(
        b,
        format!("s_mov_b32 s{WS_LANE0}, vcc_lo"),
        &[s(WS_LANE0)],
        &[],
    )?;
    b.wait_all()?;
    let wg = if w.x2() {
        // Blocks [0, ceil(M0/8)) run matrix 0, the rest matrix 1 with the
        // matrix-local block index; M0 = 0 routes every block to matrix 1.
        sop(
            b,
            format!("s_add_i32 s{WS_T1}, s{WS_RM}, 7"),
            &[WS_T1],
            &[WS_RM],
        )?;
        sop(
            b,
            format!("s_lshr_b32 s{WS_T1}, s{WS_T1}, 3"),
            &[WS_T1],
            &[WS_T1],
        )?;
        sop(
            b,
            format!("s_cmp_ge_u32 s{wg_x}, s{WS_T1}"),
            &[],
            &[wg_x, WS_T1],
        )?;
        op(
            b,
            format!(
                "s_cselect_b64 s[{WS_A}:{}], s[{}:{}], s[{WS_RA}:{}]",
                WS_A + 1,
                WS_RA + 2,
                WS_RA + 3,
                WS_RA + 1
            ),
            &[sr(WS_A, 2)],
            &[sr(WS_RA + 2, 2), sr(WS_RA, 2)],
        )?;
        op(
            b,
            format!(
                "s_cselect_b64 s[{WS_Y}:{}], s[{WS_RY1}:{}], s[{}:{}]",
                WS_Y + 1,
                WS_RY1 + 1,
                WS_RXY + 2,
                WS_RXY + 3
            ),
            &[sr(WS_Y, 2)],
            &[sr(WS_RY1, 2), sr(WS_RXY + 2, 2)],
        )?;
        sop(
            b,
            format!("s_cselect_b32 s{WS_M}, s{}, s{WS_RM}", WS_RM + 1),
            &[WS_M],
            &[WS_RM + 1, WS_RM],
        )?;
        sop(
            b,
            format!("s_cselect_b32 s{WS_T1}, s{WS_T1}, 0"),
            &[WS_T1],
            &[WS_T1],
        )?;
        sop(
            b,
            format!("s_sub_i32 s{WS_T0}, s{wg_x}, s{WS_T1}"),
            &[WS_T0],
            &[wg_x, WS_T1],
        )?;
        op(
            b,
            format!(
                "s_mov_b64 s[{WS_X}:{}], s[{WS_RXY}:{}]",
                WS_X + 1,
                WS_RXY + 1
            ),
            &[sr(WS_X, 2)],
            &[sr(WS_RXY, 2)],
        )?;
        sop(
            b,
            format!("s_mov_b32 s{WS_K}, s{WS_RKR}"),
            &[WS_K],
            &[WS_RKR],
        )?;
        sop(
            b,
            format!("s_mov_b32 s{WS_R}, s{}", WS_RKR + 1),
            &[WS_R],
            &[WS_RKR + 1],
        )?;
        WS_T0
    } else {
        wg_x
    };
    // Uniform refusals precede all loads/stores/barriers: total R in 1..=64 for
    // the row-tile ABI.
    sop(b, format!("s_cmp_eq_u32 s{WS_R}, 0"), &[], &[WS_R])?;
    op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
    sop(b, format!("s_cmp_gt_u32 s{WS_R}, 64"), &[], &[WS_R])?;
    op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
    sop(
        b,
        format!(
            "s_and_b32 s{WS_T1}, s{WS_K}, {}",
            lit(if w.hc() { 255 } else { 31 })
        ),
        &[WS_T1],
        &[WS_K],
    )?;
    sop(b, format!("s_cmp_lg_u32 s{WS_T1}, 0"), &[], &[WS_T1])?;
    op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
    if w.hc() {
        sop(b, format!("s_cmp_eq_u32 s{WS_K}, 0"), &[], &[WS_K])?;
        op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
    }
    if let Some(domain) = domain {
        match domain {
            RowtileDomain::EightPartitions => {
                // K%256 is checked above; signed positivity also rejects
                // negative i32 kernargs before any producer/barrier phase.
                sop(b, format!("s_cmp_le_i32 s{WS_K}, 0"), &[], &[WS_K])?;
                op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
            }
            RowtileDomain::Exact(k) => {
                sop(b, format!("s_cmp_lg_u32 s{WS_K}, {}", lit(k)), &[], &[WS_K])?;
                op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
            }
            RowtileDomain::Small => {
                sop(b, format!("s_cmp_eq_u32 s{WS_K}, 0"), &[], &[WS_K])?;
                op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
                sop(
                    b,
                    format!("s_cmp_gt_u32 s{WS_K}, {}", lit(1536)),
                    &[],
                    &[WS_K],
                )?;
                op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
            }
            RowtileDomain::Large => {
                sop(
                    b,
                    format!("s_cmp_le_i32 s{WS_K}, {}", lit(1536)),
                    &[],
                    &[WS_K],
                )?;
                op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
            }
        }
    } else {
        match w.kind {
            WideKind::K320 | WideKind::K2560 | WideKind::K2560X2 => {
                let k = if w.kind == WideKind::K320 { 320 } else { 2560 };
                sop(b, format!("s_cmp_lg_u32 s{WS_K}, {}", lit(k)), &[], &[WS_K])?;
                op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
            }
            WideKind::Wide => {
                sop(b, format!("s_cmp_eq_u32 s{WS_K}, 0"), &[], &[WS_K])?;
                op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
                sop(
                    b,
                    format!("s_cmp_gt_u32 s{WS_K}, {}", lit(1536)),
                    &[],
                    &[WS_K],
                )?;
                op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
            }
            WideKind::K8 => {}
        }
    }
    sop(
        b,
        format!(
            "s_lshl_b32 s{WS_MBASE}, s{wg}, {}",
            if w.hc() { 1 } else { 3 }
        ),
        &[WS_MBASE],
        &[wg],
    )?;
    sop(
        b,
        format!("s_cmp_ge_i32 s{WS_MBASE}, s{WS_M}"),
        &[],
        &[WS_MBASE, WS_M],
    )?;
    op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
    // `wg_y` is the system workgroup-id-y SGPR (s3; s15 on gfx1100). Rebase the selected matrix's
    // pointers once, retaining K/M as the full-window row strides.
    sop(
        b,
        format!("s_lshl_b32 s{wg_y}, s{wg_y}, 3"),
        &[wg_y],
        &[wg_y],
    )?;
    sop(
        b,
        format!("s_cmp_ge_u32 s{wg_y}, s{WS_R}"),
        &[],
        &[wg_y, WS_R],
    )?;
    op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
    sop(
        b,
        format!("s_sub_u32 s{WS_R}, s{WS_R}, s{wg_y}"),
        &[WS_R],
        &[WS_R, wg_y],
    )?;
    sop(
        b,
        format!("s_min_u32 s{WS_R}, s{WS_R}, 8"),
        &[WS_R],
        &[WS_R],
    )?;
    for (base, stride) in [(WS_X, WS_K), (WS_Y, WS_M)] {
        sop(
            b,
            format!("s_mul_i32 s{WS_T0}, s{wg_y}, s{stride}"),
            &[WS_T0],
            &[wg_y, stride],
        )?;
        sop(
            b,
            format!("s_lshl_b32 s{WS_T0}, s{WS_T0}, 2"),
            &[WS_T0],
            &[WS_T0],
        )?;
        sop(
            b,
            format!("s_add_u32 s{base}, s{base}, s{WS_T0}"),
            &[base],
            &[base, WS_T0],
        )?;
        sop(
            b,
            format!("s_addc_u32 s{}, s{}, 0", base + 1, base + 1),
            &[base + 1],
            &[base + 1],
        )?;
    }
    // Column, liveness, weight row (HC: row-major partitions `(m*8 + part)`).
    if w.hc() {
        sop(
            b,
            format!("s_lshr_b32 s{WS_PROW}, s{wave}, 3"),
            &[WS_PROW],
            &[wave],
        )?;
        sop(
            b,
            format!("s_and_b32 s{WS_PART}, s{wave}, 7"),
            &[WS_PART],
            &[wave],
        )?;
    } else {
        sop(
            b,
            format!("s_mov_b32 s{WS_PROW}, s{wave}"),
            &[WS_PROW],
            &[wave],
        )?;
    }
    sop(
        b,
        format!("s_add_i32 s{WS_COL}, s{WS_MBASE}, s{WS_PROW}"),
        &[WS_COL],
        &[WS_MBASE, WS_PROW],
    )?;
    sop(
        b,
        format!("s_cmp_lt_i32 s{WS_COL}, s{WS_M}"),
        &[],
        &[WS_COL, WS_M],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{WS_LIVE}, 1, 0"),
        &[WS_LIVE],
        &[],
    )?;
    sop(
        b,
        format!(
            "s_lshr_b32 s{WS_NBLK}, s{WS_K}, {}",
            if w.hc() { 8 } else { 5 }
        ),
        &[WS_NBLK],
        &[WS_K],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{WS_ROWB}, s{WS_NBLK}, 34"),
        &[WS_ROWB],
        &[WS_NBLK],
    )?;
    if w.hc() {
        sop(
            b,
            format!("s_lshl_b32 s{WS_ROWOFF}, s{WS_COL}, 3"),
            &[WS_ROWOFF],
            &[WS_COL],
        )?;
        sop(
            b,
            format!("s_add_i32 s{WS_ROWOFF}, s{WS_ROWOFF}, s{WS_PART}"),
            &[WS_ROWOFF],
            &[WS_ROWOFF, WS_PART],
        )?;
        sop(
            b,
            format!("s_mul_i32 s{WS_ROWOFF}, s{WS_ROWOFF}, s{WS_ROWB}"),
            &[WS_ROWOFF],
            &[WS_ROWOFF, WS_ROWB],
        )?;
    } else {
        sop(
            b,
            format!("s_mul_i32 s{WS_ROWOFF}, s{WS_COL}, s{WS_ROWB}"),
            &[WS_ROWOFF],
            &[WS_COL, WS_ROWB],
        )?;
    }
    vop(
        b,
        format!("v_mov_b32_e32 v{WV_WOFFD}, s{WS_ROWOFF}"),
        &[WV_WOFFD],
        &[],
        &[WS_ROWOFF],
    )?;
    vop(
        b,
        format!("v_add_nc_u32_e32 v{WV_WOFFQ}, s{WS_ROWOFF}, v{V_LANE}"),
        &[WV_WOFFQ],
        &[V_LANE],
        &[WS_ROWOFF],
    )?;
    // Producer row of tile 0: x row `PROW` (HC: plus partition `part*K/8`).
    sop(
        b,
        format!("s_lshl_b32 s{WS_KX4}, s{WS_K}, 2"),
        &[WS_KX4],
        &[WS_K],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{WS_XTB}, s{WS_PROW}, s{WS_KX4}"),
        &[WS_XTB],
        &[WS_PROW, WS_KX4],
    )?;
    if w.hc() {
        sop(
            b,
            format!("s_lshr_b32 s{WS_T1}, s{WS_K}, 1"),
            &[WS_T1],
            &[WS_K],
        )?;
        sop(
            b,
            format!("s_mul_i32 s{WS_T1}, s{WS_T1}, s{WS_PART}"),
            &[WS_T1],
            &[WS_T1, WS_PART],
        )?;
        sop(
            b,
            format!("s_add_i32 s{WS_XTB}, s{WS_XTB}, s{WS_T1}"),
            &[WS_XTB],
            &[WS_XTB, WS_T1],
        )?;
    }
    sop(
        b,
        format!("s_lshr_b32 s{WS_NFULL}, s{WS_NBLK}, 3"),
        &[WS_NFULL],
        &[WS_NBLK],
    )?;
    sop(
        b,
        format!("s_and_b32 s{WS_TAIL}, s{WS_NBLK}, 7"),
        &[WS_TAIL],
        &[WS_NBLK],
    )?;
    // LDS word (stage row v, block j, lane l) = (v*8 + j)*32 + l; the
    // producer wave (HC: (ml, part) = wave) owns stage row `wave`.
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{WV_XOFF}, 2, v{V_LANE}"),
        &[WV_XOFF],
        &[V_LANE],
        &[],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{WS_T1}, s{wave}, 10"),
        &[WS_T1],
        &[wave],
    )?;
    vop(
        b,
        format!("v_add_nc_u32_e32 v{WV_LDST}, s{WS_T1}, v{WV_XOFF}"),
        &[WV_LDST],
        &[WV_XOFF],
        &[WS_T1],
    )?;
    if w.hc() {
        // Consumer (ml, part) reads stage rows (r2, part), r2 = 0, 1; lane 0
        // parks partial[ml][r][part] at ((ml*cap + r)*8 + part)*4.
        sop(
            b,
            format!("s_lshl_b32 s{WS_T1}, s{WS_PART}, 10"),
            &[WS_T1],
            &[WS_PART],
        )?;
        vop(
            b,
            format!("v_add_nc_u32_e32 v{WV_LDRD}, s{WS_T1}, v{WV_XOFF}"),
            &[WV_LDRD],
            &[WV_XOFF],
            &[WS_T1],
        )?;
        sop(
            b,
            format!(
                "s_mul_i32 s{WS_T1}, s{WS_PROW}, {}",
                lit(32 * u32::from(w.cap())),
            ),
            &[WS_T1],
            &[WS_PROW],
        )?;
        sop(
            b,
            format!("s_lshl_b32 s{WS_T0}, s{WS_PART}, 2"),
            &[WS_T0],
            &[WS_PART],
        )?;
        sop(
            b,
            format!("s_add_i32 s{WS_T1}, s{WS_T1}, s{WS_T0}"),
            &[WS_T1],
            &[WS_T1, WS_T0],
        )?;
        vop(
            b,
            format!("v_mov_b32_e32 v{WV_PST}, s{WS_T1}"),
            &[WV_PST],
            &[],
            &[WS_T1],
        )
    } else {
        vop(
            b,
            format!("v_mov_b32_e32 v{WV_LDRD}, v{WV_XOFF}"),
            &[WV_LDRD],
            &[WV_XOFF],
            &[],
        )?;
        sop(
            b,
            format!("s_lshl_b32 s{WS_MB}, s{WS_M}, 2"),
            &[WS_MB],
            &[WS_M],
        )?;
        sop(
            b,
            format!("s_lshl_b32 s{WS_COL4}, s{WS_COL}, 2"),
            &[WS_COL4],
            &[WS_COL],
        )
    }
}

/// The whole K sweep: the full 8-block tiles in a loop, then the tail tile
/// of `TAIL` (1..=7) blocks through a compile-time variant per listed
/// count, so no block past the row (partition) is loaded or consumed.
fn wide_traverse(
    b: &mut Builder,
    w: WideSpec,
    slot: usize,
    order: Order,
    tails: &[u8],
    lbl: &str,
) -> R {
    let tail_lbl = format!("{lbl}_tails");
    let kdone = format!("{lbl}_kdone");
    sop(
        b,
        format!("s_mov_b32 s{WS_IT}, s{WS_NFULL}"),
        &[WS_IT],
        &[WS_NFULL],
    )?;
    sop(b, format!("s_cmp_eq_u32 s{WS_IT}, 0"), &[], &[WS_IT])?;
    op(b, format!("s_cbranch_scc1 {tail_lbl}"), &[], &[])?;
    let head = format!("{lbl}_kloop");
    let full = format!("{lbl}_full");
    b.loop_(&head, |b| {
        wide_pass(b, w, slot, order, 8, &full)?;
        vop(
            b,
            format!("v_add_nc_u32_e32 v{WV_XOFF}, {}, v{WV_XOFF}", lit(8 * 128)),
            &[WV_XOFF],
            &[WV_XOFF],
            &[],
        )?;
        for r in [WV_WOFFD, WV_WOFFQ] {
            vop(
                b,
                format!("v_add_nc_u32_e32 v{r}, {}, v{r}", lit(8 * 34)),
                &[r],
                &[r],
                &[],
            )?;
        }
        sop(
            b,
            format!("s_add_i32 s{WS_IT}, s{WS_IT}, -1"),
            &[WS_IT],
            &[WS_IT],
        )?;
        sop(b, format!("s_cmp_lg_u32 s{WS_IT}, 0"), &[], &[WS_IT])?;
        op(b, format!("s_cbranch_scc1 {head}"), &[], &[])
    })?;
    b.label(&tail_lbl)?;
    for &nb in tails {
        sop(b, format!("s_cmp_eq_u32 s{WS_TAIL}, {nb}"), &[], &[WS_TAIL])?;
        op(b, format!("s_cbranch_scc1 {lbl}_tail{nb}"), &[], &[])?;
    }
    op(b, format!("s_branch {kdone}"), &[], &[])?;
    let part = format!("{lbl}_part");
    for &nb in tails {
        b.label(&format!("{lbl}_tail{nb}"))?;
        wide_pass(b, w, slot, order, nb, &part)?;
        op(b, format!("s_branch {kdone}"), &[], &[])?;
    }
    b.label(&kdone)
}

/// One K tile of `nb` blocks: the live wave's weights once, then every row tile.
fn wide_pass(b: &mut Builder, w: WideSpec, slot: usize, order: Order, nb: u8, lbl: &str) -> R {
    let lbl = format!("{lbl}{nb}");
    wide_weights(b, nb, &lbl)?;
    for t in 0..w.tiles() {
        wide_tile(b, w, slot, order, nb, t, &lbl)?;
    }
    Ok(())
}

/// Live waves: `q[j]` / `d[j]` of the tile's `nb` blocks and
/// `dq[j] = fma_mix(d_f16, f32(q), -0.0)`, held until every row consumed them.
fn wide_weights(b: &mut Builder, nb: u8, lbl: &str) -> R {
    let done = format!("{lbl}_wdone");
    sop(b, format!("s_cmp_eq_u32 s{WS_LIVE}, 0"), &[], &[WS_LIVE])?;
    op(b, format!("s_cbranch_scc1 {done}"), &[], &[])?;
    for j in 0..nb {
        gload(
            b,
            "global_load_i8",
            WV_Q + j,
            WV_WOFFQ,
            WS_A,
            34 * u32::from(j) + 2,
        )?;
        gload(
            b,
            "global_load_u16",
            WV_DQ + j,
            WV_WOFFD,
            WS_A,
            34 * u32::from(j),
        )?;
    }
    for j in 0..nb {
        dequant(b, WV_DQ + j, WV_Q + j)?;
    }
    b.label(&done)
}

/// Row tile `t` of one K tile: producer fill, barrier, every live wave
/// consumes the tile's rows `< R`, barrier. Rows past `R` are neither loaded
/// nor consumed; dead waves produce and keep every barrier.
fn wide_tile(
    b: &mut Builder,
    w: WideSpec,
    slot: usize,
    order: Order,
    nb: u8,
    t: u32,
    lbl: &str,
) -> R {
    let rows = w.rows_per_tile();
    let rb = rows * t;
    let tag = format!("{lbl}_t{t}");
    let done = format!("{tag}_done");
    if t > 0 {
        sop(b, format!("s_cmp_le_u32 s{WS_R}, {rb}"), &[], &[WS_R])?;
        op(b, format!("s_cbranch_scc1 {done}"), &[], &[])?;
    }
    let nofill = format!("{tag}_nofill");
    if t == 0 {
        sop(
            b,
            format!("s_cmp_ge_u32 s{WS_PROW}, s{WS_R}"),
            &[],
            &[WS_PROW, WS_R],
        )?;
    } else {
        sop(
            b,
            format!("s_add_i32 s{WS_T0}, s{WS_PROW}, {rb}"),
            &[WS_T0],
            &[WS_PROW],
        )?;
        sop(
            b,
            format!("s_cmp_ge_u32 s{WS_T0}, s{WS_R}"),
            &[],
            &[WS_T0, WS_R],
        )?;
    }
    op(b, format!("s_cbranch_scc1 {nofill}"), &[], &[])?;
    if t == 0 {
        vop(
            b,
            format!("v_add_nc_u32_e32 v{WV_XADDR}, s{WS_XTB}, v{WV_XOFF}"),
            &[WV_XADDR],
            &[WV_XOFF],
            &[WS_XTB],
        )?;
    } else {
        sop(
            b,
            format!("s_mul_i32 s{WS_T0}, s{WS_KX4}, {}", lit(rb)),
            &[WS_T0],
            &[WS_KX4],
        )?;
        sop(
            b,
            format!("s_add_i32 s{WS_T0}, s{WS_T0}, s{WS_XTB}"),
            &[WS_T0],
            &[WS_T0, WS_XTB],
        )?;
        vop(
            b,
            format!("v_add_nc_u32_e32 v{WV_XADDR}, s{WS_T0}, v{WV_XOFF}"),
            &[WV_XADDR],
            &[WV_XOFF],
            &[WS_T0],
        )?;
    }
    for j in 0..nb {
        gload(
            b,
            "global_load_b32",
            WV_XS + j,
            WV_XADDR,
            WS_X,
            128 * u32::from(j),
        )?;
    }
    for j in 0..nb {
        ds_st(b, slot, WV_LDST, WV_XS + j, 128 * u32::from(j))?;
    }
    b.label(&nofill)?;
    b.barrier(&[Transition::Ready(slot)])?;
    let cdone = format!("{tag}_cdone");
    sop(b, format!("s_cmp_eq_u32 s{WS_LIVE}, 0"), &[], &[WS_LIVE])?;
    op(b, format!("s_cbranch_scc1 {cdone}"), &[], &[])?;
    for r in 0..rows {
        if r > 0 {
            sop(b, format!("s_cmp_le_u32 s{WS_R}, {}", rb + r), &[], &[WS_R])?;
            op(b, format!("s_cbranch_scc1 {cdone}"), &[], &[])?;
        }
        wide_consume(b, w, slot, order, nb, r, rb + r)?;
    }
    b.label(&cdone)?;
    b.barrier(&[Transition::Retire(slot)])?;
    b.label(&done)
}

/// Stage row `r_local` of the tile into activation row `row`'s chain(s):
/// blocks ascending. One chain (ordinary, HC): `sum = fma(x, dq, sum)`
/// (`v_fmac_f32 sum, x, dq`). Four chains: block `j` (global block
/// `8k + j`, chain `j & 3`) is a quad block when `j < 4*floor(nb/4)` (the
/// whole tile when `nb = 8`): `Order` spelling of the fused update; else
/// a remainder block: `p = RN(x * dq)`, `acc = acc + p`.
fn wide_consume(
    b: &mut Builder,
    w: WideSpec,
    slot: usize,
    order: Order,
    nb: u8,
    r_local: u32,
    row: u32,
) -> R {
    let base = r_local * if w.hc() { 8 * 1024 } else { 1024 };
    for j in 0..nb {
        ds_ld(
            b,
            slot,
            "ds_load_b32",
            WV_XS + j,
            WV_LDRD,
            base + 128 * u32::from(j),
            false,
        )?;
    }
    let quad_left = if nb == 8 { 8 } else { nb & !3 };
    for j in 0..nb {
        let (x, d) = (WV_XS + j, WV_DQ + j);
        if !w.four() {
            let a = wacc(w, row, 0);
            vop(
                b,
                format!("v_fmac_f32_e32 v{a}, v{x}, v{d}"),
                &[a],
                &[a, x, d],
                &[],
            )?;
            continue;
        }
        let a = wacc(w, row, j & 3);
        if j < quad_left {
            let (s0, s1) = match order {
                Order::Generic => (d, x),
                Order::K640 => (x, d),
            };
            vop(
                b,
                format!("v_fmac_f32_e32 v{a}, v{s0}, v{s1}"),
                &[a],
                &[a, s0, s1],
                &[],
            )?;
        } else {
            vop(
                b,
                format!("v_mul_f32_e32 v{WV_TMP}, v{x}, v{d}"),
                &[WV_TMP],
                &[x, d],
                &[],
            )?;
            vop(
                b,
                format!("v_add_f32_e32 v{a}, v{a}, v{WV_TMP}"),
                &[a],
                &[a, WV_TMP],
                &[],
            )?;
        }
    }
    Ok(())
}

/// Four-chain fold of row `row` into chain 0 in the selected object's operand order.
fn wide_fold(b: &mut Builder, w: WideSpec, order: Order, row: u32) -> R {
    let a = |c: u8| wacc(w, row, c);
    let steps = match order {
        Order::Generic => [(a(1), a(1), a(0)), (a(3), a(3), a(2)), (a(0), a(3), a(1))],
        Order::K640 => [(a(0), a(0), a(1)), (a(2), a(2), a(3)), (a(0), a(0), a(2))],
    };
    for (d, x, y) in steps {
        vop(
            b,
            format!("v_add_f32_e32 v{d}, v{x}, v{y}"),
            &[d],
            &[x, y],
            &[],
        )?;
    }
    Ok(())
}

/// Ordinary epilogue: dead waves leave (no barrier follows); per batch of
/// eight rows `< R`: (four chains) the fold, the lane tree, lane 0 stores
/// `y[r*M + m]`.
fn wide_epilogue(b: &mut Builder, w: WideSpec, order: Order, lbl: &str, end: &str) -> R {
    sop(b, format!("s_cmp_eq_u32 s{WS_LIVE}, 0"), &[], &[WS_LIVE])?;
    op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
    for batch in 0..u32::from(w.cap()) / 8 {
        let r0 = 8 * batch;
        if batch > 0 {
            sop(b, format!("s_cmp_le_u32 s{WS_R}, {r0}"), &[], &[WS_R])?;
            op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
        }
        let rows: Vec<u32> = (r0..r0 + 8).collect();
        for &r in &rows {
            let skip = format!("{lbl}_reduce_skip{r}");
            sop(b, format!("s_cmp_le_u32 s{WS_R}, {r}"), &[], &[WS_R])?;
            op(b, format!("s_cbranch_scc1 {skip}"), &[], &[])?;
            if w.four() {
                wide_fold(b, w, order, r)?;
            }
            lane_tree(b, &[wacc(w, r, 0)], WV_XS, WV_ADDR)?;
            b.label(&skip)?;
        }
        for (i, &r) in rows.iter().enumerate() {
            let a = WV_Q + i as u8;
            if r == 0 {
                vop(
                    b,
                    format!("v_mov_b32_e32 v{a}, s{WS_COL4}"),
                    &[a],
                    &[],
                    &[WS_COL4],
                )?;
            } else {
                sop(
                    b,
                    format!("s_mul_i32 s{WS_T0}, s{WS_MB}, {r}"),
                    &[WS_T0],
                    &[WS_MB],
                )?;
                sop(
                    b,
                    format!("s_add_i32 s{WS_T0}, s{WS_T0}, s{WS_COL4}"),
                    &[WS_T0],
                    &[WS_T0, WS_COL4],
                )?;
                vop(
                    b,
                    format!("v_mov_b32_e32 v{a}, s{WS_T0}"),
                    &[a],
                    &[],
                    &[WS_T0],
                )?;
            }
        }
        op(
            b,
            format!("s_mov_b32 exec_lo, s{WS_LANE0}"),
            &[],
            &[s(WS_LANE0)],
        )?;
        let stored = format!("{lbl}_stored{batch}");
        for (i, &r) in rows.iter().enumerate() {
            if i > 0 {
                sop(b, format!("s_cmp_le_u32 s{WS_R}, {r}"), &[], &[WS_R])?;
                op(b, format!("s_cbranch_scc1 {stored}"), &[], &[])?;
            }
            let (a, val) = (WV_Q + i as u8, wacc(w, r, 0));
            mem(
                b,
                format!("global_store_b32 v{a}, v{val}, s[{WS_Y}:{}]", WS_Y + 1),
                &[],
                &[v(a), v(val), sr(WS_Y, 2)],
                MemoryClass::VmemStore,
            )?;
        }
        b.label(&stored)?;
        op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
        b.wait_all()?;
    }
    Ok(())
}

/// HC epilogue: the X tile is dead (its last readers retired by the final
/// tile barrier) and its bytes become `partial[2][cap][8]`. Lane 0 of each
/// live wave parks its rows' partition sums; barrier; thread
/// `t < 2*cap` folds column `t >> log2(cap)`, row `t & (cap-1)` (when the row
/// is `< R` and the column `< M`) exactly as the narrow k8 entry does.
fn wide_hc_epilogue(b: &mut Builder, w: WideSpec, lbl: &str, end: &str) -> R {
    let cap = u32::from(w.cap());
    b.lds_relayout()?;
    let pslot = b.lds_slot("partials", 0, 2 * cap * 8 * 4)?;
    let parked = format!("{lbl}_parked");
    sop(b, format!("s_cmp_eq_u32 s{WS_LIVE}, 0"), &[], &[WS_LIVE])?;
    op(b, format!("s_cbranch_scc1 {parked}"), &[], &[])?;
    for batch in 0..cap / 8 {
        let r0 = 8 * batch;
        if batch > 0 {
            sop(b, format!("s_cmp_le_u32 s{WS_R}, {r0}"), &[], &[WS_R])?;
            op(b, format!("s_cbranch_scc1 {parked}"), &[], &[])?;
        }
        for r in r0..r0 + 8 {
            let skip = format!("{lbl}_partial_skip{r}");
            sop(b, format!("s_cmp_le_u32 s{WS_R}, {r}"), &[], &[WS_R])?;
            op(b, format!("s_cbranch_scc1 {skip}"), &[], &[])?;
            lane_tree(b, &[wacc(w, r, 0)], WV_XS, WV_ADDR)?;
            op(
                b,
                format!("s_mov_b32 exec_lo, s{WS_LANE0}"),
                &[],
                &[s(WS_LANE0)],
            )?;
            ds_st(b, pslot, WV_PST, wacc(w, r, 0), 32 * r)?;
            op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
            b.label(&skip)?;
        }
    }
    b.label(&parked)?;
    let (e_addr, e_r, e_m2) = (WV_XADDR, WV_TMP, WV_VT);
    vop(
        b,
        format!("v_and_b32_e32 v{e_r}, {}, v{V_TID}", cap - 1),
        &[e_r],
        &[V_TID],
        &[],
    )?;
    vop(
        b,
        format!(
            "v_lshrrev_b32_e32 v{e_m2}, {}, v{V_TID}",
            cap.trailing_zeros()
        ),
        &[e_m2],
        &[V_TID],
        &[],
    )?;
    vop(
        b,
        format!("v_add_nc_u32_e32 v{e_m2}, s{WS_MBASE}, v{e_m2}"),
        &[e_m2],
        &[e_m2],
        &[WS_MBASE],
    )?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{e_addr}, 5, v{V_TID}"),
        &[e_addr],
        &[V_TID],
        &[],
    )?;
    op(
        b,
        format!("v_mul_lo_u32 v{WV_EY}, v{e_r}, s{WS_M}"),
        &[v(WV_EY)],
        &[v(e_r), s(WS_M)],
    )?;
    vop(
        b,
        format!("v_add_nc_u32_e32 v{WV_EY}, v{e_m2}, v{WV_EY}"),
        &[WV_EY],
        &[e_m2, WV_EY],
        &[],
    )?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{WV_EY}, 2, v{WV_EY}"),
        &[WV_EY],
        &[WV_EY],
        &[],
    )?;
    b.barrier(&[Transition::Ready(pslot)])?;
    vop(
        b,
        format!("v_cmp_gt_u32_e32 vcc_lo, {}, v{V_TID}", 2 * cap),
        &[],
        &[V_TID],
        &[],
    )?;
    op(
        b,
        format!("s_mov_b32 s{WS_MASK}, vcc_lo"),
        &[s(WS_MASK)],
        &[],
    )?;
    vop(
        b,
        format!("v_cmp_gt_u32_e32 vcc_lo, s{WS_R}, v{e_r}"),
        &[],
        &[e_r],
        &[WS_R],
    )?;
    sop(
        b,
        format!("s_and_b32 s{WS_MASK}, s{WS_MASK}, vcc_lo"),
        &[WS_MASK],
        &[WS_MASK],
    )?;
    vop(
        b,
        format!("v_cmp_gt_i32_e32 vcc_lo, s{WS_M}, v{e_m2}"),
        &[],
        &[e_m2],
        &[WS_M],
    )?;
    op(
        b,
        format!("s_and_b32 exec_lo, s{WS_MASK}, vcc_lo"),
        &[],
        &[s(WS_MASK)],
    )?;
    op(b, format!("s_cbranch_execz {end}"), &[], &[])?;
    for half in 0..2u8 {
        b.ds_load(
            pslot,
            Instruction::new(
                format!(
                    "ds_load_b128 {}, v{e_addr}{}",
                    vrange(WV_DQ + 4 * half, 4),
                    off(16 * u32::from(half))
                ),
                vec![vr(WV_DQ + 4 * half, 4)],
                vec![v(e_addr)],
            )
            .memory(MemoryClass::DsLoad),
        )?;
    }
    let p = |n: u8| WV_DQ + n;
    // ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7)), the narrow k8 spelling.
    for (dst, a, c) in [
        (0u8, 0u8, 1u8),
        (2, 2, 3),
        (4, 4, 5),
        (6, 6, 7),
        (0, 0, 2),
        (4, 4, 6),
        (0, 0, 4),
    ] {
        vop(
            b,
            format!("v_add_f32_e32 v{}, v{}, v{}", p(dst), p(a), p(c)),
            &[p(dst)],
            &[p(a), p(c)],
            &[],
        )?;
    }
    let x = p(0);
    let store = format!("{lbl}_store");
    let hc_act = w.s_hc();
    sop(b, format!("s_cmp_eq_u32 s{hc_act}, 0"), &[], &[hc_act])?;
    op(b, format!("s_cbranch_scc1 {store}"), &[], &[])?;
    let e = |n: u8| WV_Q + n;
    bf16_round(b, x, [e(0), e(1), e(2), e(3)])?;
    vop(
        b,
        format!("v_mul_f32_e32 v{x}, s{WS_SCALE}, v{x}"),
        &[x],
        &[x],
        &[WS_SCALE],
    )?;
    bf16_round(b, x, [e(0), e(1), e(2), e(3)])?;
    silu(b, x, [e(0), e(1), e(2), e(3), e(4), e(5)])?;
    bf16_round(b, x, [e(0), e(1), e(2), e(3)])?;
    b.label(&store)?;
    mem(
        b,
        format!("global_store_b32 v{WV_EY}, v{x}, s[{WS_Y}:{}]", WS_Y + 1),
        &[],
        &[v(WV_EY), v(x), sr(WS_Y, 2)],
        MemoryClass::VmemStore,
    )?;
    b.wait_all()
}

/// Historical r16/r32 wide exports were removed with `WIDE_EXPORTS`; row tiles call
/// [`wide_impl`] directly. Narrow [`build_gfx1151`] exports and text are unchanged.

fn rowtile_symbol(kind: WideKind) -> &'static str {
    match kind {
        WideKind::K8 => "gemv_q8_0_k8_rows_t8_pm_gfx1151",
        WideKind::K320 => "gemv_q8_0_k320_rows_t8_pm_gfx1151",
        WideKind::K2560 => "gemv_q8_0_k2560_rows_t8_pm_gfx1151",
        WideKind::K2560X2 => "gemv_q8_0_k2560_rows_x2_t8_pm_gfx1151",
        WideKind::Wide => "gemv_q8_0_wide_rows_t8_pm_gfx1151",
    }
}

/// Fixed-eight physical tiles for total logical R=1..64 on gfx1151 only.
/// Frozen order: HC k8, k320, k2560, k2560 x2, small-K wide. Kernargs are
/// unchanged from the wide entries; grid y is ceil(R/8), block sizes 512/256.
/// This is not a gfx1201 recipe or a capability/admission declaration.
pub fn build_rowtiles_gfx1151() -> Result<Vec<Emitted>, String> {
    [
        WideKind::K8,
        WideKind::K320,
        WideKind::K2560,
        WideKind::K2560X2,
        WideKind::Wide,
    ]
    .into_iter()
    .map(|kind| {
        wide_impl(WideSpec {
            kind,
            arch: Arch::Gfx1151,
        })
        .map_err(|e| format!("{}: {e}", rowtile_symbol(kind)))
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_plans_fit_the_ring_layout() {
        for slot in 0..4u8 {
            assert_eq!(k8_q(slot, 3) + 1, k8_d(slot, 0));
            assert_eq!(k8_d(slot, 3) + 1, k8_x(slot, 0));
            assert!(u16::from(k8_x(slot, 7)) < u16::from(V_SUM));
        }
        assert_eq!(k8_sum(3, 7) + 1, V_TMP);
        assert_eq!(u16::from(V_IY) + 1, K8_VGPRS);
        assert!(u16::from(k320_x(9, 7)) + 1 == K320_VGPRS);
        assert_eq!(W_Q, W_D + 10);
        assert_eq!(W_SUM, W_TMP + 8);
        assert_eq!(W_X, k320_sum(3, 7) + 1);
        // Eight k8 columns keep 64 loads in flight, the vmcnt field's reach.
        assert_eq!(4 * (4 + 4 + 8), 64);
        // k320: the third staging store of lane 31 still ends inside the wave region.
        assert!(256 + 4 * 31 + 4 <= K320_WAVE_REGION);
        assert_eq!(K320_LDS_BYTES, 1536);
        assert_eq!(K8_LDS_BYTES, 1024);
    }

    #[test]
    fn exports_build_with_the_frozen_budgets() {
        let emitted = build_gfx1151().expect("q8 lane rows");
        assert_eq!(emitted.len(), 2);
        let (k8, k320) = (&emitted[0], &emitted[1]);
        assert_eq!(k8.shape.next_free_vgpr, K8_VGPRS);
        assert_eq!(k320.shape.next_free_vgpr, K320_VGPRS);
        assert!(k8.shape.next_free_sgpr <= 40 && k320.shape.next_free_sgpr <= 40);
        assert!(k8
            .s_text
            .contains(&format!(".amdhsa_group_segment_fixed_size {K8_LDS_BYTES}")));
        assert!(k320.s_text.contains(&format!(
            ".amdhsa_group_segment_fixed_size {K320_LDS_BYTES}"
        )));
        assert_eq!(k8.proof.lds_slots.len(), 1);
        assert_eq!(k320.proof.lds_slots.len(), 2);
        for (e, symbol) in [(k8, K8_SYMBOL), (k320, K320_SYMBOL)] {
            assert!(e.s_text.contains(&format!("{symbol}:")));
            // The DAG nodes: dq = fma_mix(d, q, -0.0); sum = fma(x, dq, sum); the lane tree.
            assert!(
                e.s_text.contains("v_fma_mix_f32") && e.s_text.contains("neg(0) op_sel_hi:[1,0,0]")
            );
            assert!(e.s_text.contains("v_fmac_f32_e32") && e.s_text.contains("v_cvt_f32_i32_e32"));
            assert!(e.s_text.contains("ds_swizzle_b32") && e.s_text.contains("ds_bpermute_b32"));
            assert!(e.s_text.contains("s_barrier"));
        }
        // Only k8 carries the epilogue.
        assert!(
            k8.s_text.contains("v_div_fixup_f32")
                && k8.s_text.contains("v_div_fmas_f32")
                && k8.s_text.contains("v_exp_f32_e32")
        );
        assert!(!k320.s_text.contains("v_div_fixup_f32"));
        assert_eq!(k8.proof.loop_fixpoints.len(), usize::from(MAX_COLUMNS));
        assert!(k320.proof.loop_fixpoints.is_empty());
    }

    #[test]
    fn rowtiles_build_in_frozen_order_with_declared_budgets() {
        let kinds = [
            WideKind::K8,
            WideKind::K320,
            WideKind::K2560,
            WideKind::K2560X2,
            WideKind::Wide,
        ];
        let emitted = build_rowtiles_gfx1151().expect("q8 rowtiles");
        assert_eq!(emitted.len(), kinds.len());
        for (e, kind) in emitted.iter().zip(kinds) {
            let symbol = rowtile_symbol(kind);
            assert!(
                e.s_text.contains(&format!(".amdhsa_kernel {symbol}")),
                "{symbol}"
            );
            assert!(!e.s_text.contains("xbatch_wide_r16"));
            assert!(!e.s_text.contains("xbatch_wide_r32"));
            let w = WideSpec {
                kind,
                arch: Arch::Gfx1151,
            };
            assert_eq!(
                e.shape.next_free_vgpr,
                w.vgpr_high_water() as u16,
                "{symbol}"
            );
            assert!(e.shape.next_free_vgpr <= w.vgpr_budget(), "{symbol}");
            assert!(e.shape.next_free_sgpr <= WIDE_SGPRS, "{symbol}");
            assert!(
                e.s_text.contains(&format!(
                    ".amdhsa_group_segment_fixed_size {}",
                    w.lds_bytes()
                )),
                "{symbol}"
            );
            assert!(!e.s_text.contains("scratch_"), "{symbol}");
            assert!(e.s_text.contains("s_barrier"), "{symbol}");
            // dq = fma_mix(d, q, -0.0) once per block; the swizzle/bpermute tree.
            assert!(
                e.s_text.contains("v_fma_mix_f32") && e.s_text.contains("neg(0) op_sel_hi:[1,0,0]"),
                "{symbol}"
            );
            assert!(
                e.s_text.contains("ds_swizzle_b32") && e.s_text.contains("ds_bpermute_b32"),
                "{symbol}"
            );
            // The full-tile K loop converged.
            assert!(!e.proof.loop_fixpoints.is_empty(), "{symbol}");
        }
        // HC: the X tile retired, then the partial layout.
        assert_eq!(emitted[0].proof.lds_slots.len(), 2);
        assert_eq!(emitted[1].proof.lds_slots.len(), 1);
        assert_eq!(
            WideSpec {
                kind: WideKind::K8,
                arch: Arch::Gfx1151
            }
            .kernarg_bytes(),
            44
        );
        assert_eq!(
            WideSpec {
                kind: WideKind::K320,
                arch: Arch::Gfx1151,
            }
            .kernarg_bytes(),
            36
        );
        assert_eq!(
            WideSpec {
                kind: WideKind::K2560X2,
                arch: Arch::Gfx1151,
            }
            .kernarg_bytes(),
            56
        );
        assert_eq!(
            WideSpec {
                kind: WideKind::K8,
                arch: Arch::Gfx1151
            }
            .workgroup_size(),
            512
        );
        assert_eq!(
            WideSpec {
                kind: WideKind::Wide,
                arch: Arch::Gfx1151,
            }
            .workgroup_size(),
            256
        );
        // The narrow exports are a separate builder, untouched by the row tiles.
        assert_eq!(build_gfx1151().expect("narrow").len(), 2);
    }

    /// Weight loads are emitted once per K tile shape (full loop body plus the
    /// tail variants): physical capacity is frozen at eight.
    #[test]
    fn rowtile_weight_loads_match_the_k_tile_shapes() {
        let kinds = [
            WideKind::K8,
            WideKind::K320,
            WideKind::K2560,
            WideKind::K2560X2,
            WideKind::Wide,
        ];
        let emitted = build_rowtiles_gfx1151().expect("q8 rowtiles");
        let count = |e: &Emitted, m: &str| {
            e.s_text
                .lines()
                .filter(|l| l.trim_start().starts_with(m))
                .count()
        };
        // 8 (full tile) + 1 + .. + 7 (tail variants); four chains add the K = 640 path (8 + 4).
        for (e, kind) in emitted.iter().zip(kinds) {
            let want = if kind == WideKind::Wide { 36 + 12 } else { 36 };
            assert_eq!(
                count(e, "global_load_i8 "),
                want,
                "{}",
                rowtile_symbol(kind)
            );
        }
    }

    /// Each entry spells the selected decode object's operations (`DAG.md`
    /// receipts; asm of `gemv_q8_0_k320_staged` / `_k2560_staged` / `_k8`,
    /// `gemv_q8_0_wide`, `gemv_q8_0_wide_k640_staged` on gfx1151).
    #[test]
    fn rowtile_entries_pin_the_selected_decode_dags() {
        let kinds = [
            WideKind::K8,
            WideKind::K320,
            WideKind::K2560,
            WideKind::K2560X2,
            WideKind::Wide,
        ];
        let emitted = build_rowtiles_gfx1151().expect("q8 rowtiles");
        let (acc0, x0, dq0) = (WV_ACC, WV_XS, WV_DQ);
        for (e, kind) in emitted.iter().zip(kinds) {
            let w = WideSpec {
                kind,
                arch: Arch::Gfx1151,
            };
            let t = &e.s_text;
            if w.four() {
                // Generic quad: fmac(acc, dq, x); K = 640: fmac(acc, x, dq).
                assert!(t.contains(&format!("v_fmac_f32_e32 v{acc0}, v{dq0}, v{x0}")));
                assert!(t.contains(&format!("v_fmac_f32_e32 v{acc0}, v{x0}, v{dq0}")));
                // Remainder block: RN(x * dq), then acc + p (no contraction).
                assert!(t.contains(&format!("v_mul_f32_e32 v{WV_TMP}, v{x0}, v{dq0}")));
                assert!(t.contains(&format!("v_add_f32_e32 v{acc0}, v{acc0}, v{WV_TMP}")));
                // Folds: generic (a3 + a2) + (a1 + a0); K = 640 (a0 + a1) + (a2 + a3).
                let a = |c: u8| acc0 + c;
                for line in [
                    format!("v_add_f32_e32 v{}, v{}, v{}", a(1), a(1), a(0)),
                    format!("v_add_f32_e32 v{}, v{}, v{}", a(3), a(3), a(2)),
                    format!("v_add_f32_e32 v{}, v{}, v{}", a(0), a(3), a(1)),
                    format!("v_add_f32_e32 v{}, v{}, v{}", a(0), a(0), a(1)),
                    format!("v_add_f32_e32 v{}, v{}, v{}", a(2), a(2), a(3)),
                    format!("v_add_f32_e32 v{}, v{}, v{}", a(0), a(0), a(2)),
                ] {
                    assert!(t.contains(&line), "{line}");
                }
                assert!(t.contains("s_cmp_eq_u32 s11, 0x280"));
            } else {
                // One chain: every block fused, fmac(sum, x, dq), no unfused product.
                assert!(t.contains(&format!("v_fmac_f32_e32 v{acc0}, v{x0}, v{dq0}")));
                assert!(!t.contains(&format!("v_fmac_f32_e32 v{acc0}, v{dq0}, v{x0}")));
                assert!(!t.contains(&format!("v_mul_f32_e32 v{WV_TMP},")));
            }
            if w.hc() {
                assert!(
                    t.contains("v_div_fixup_f32")
                        && t.contains("v_div_fmas_f32")
                        && t.contains("v_exp_f32_e32")
                );
                assert!(t.contains(&format!("v_mul_f32_e32 v{dq0}, s{WS_SCALE}, v{dq0}")));
            } else {
                assert!(!t.contains("v_div_fixup_f32"));
            }
        }
    }

    #[cfg(feature = "toolchain")]
    #[test]
    fn q8_lane_rows_m7() {
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-lane-rows-q8-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for (emitted, symbol) in build_gfx1151()
            .expect("q8 lane rows")
            .into_iter()
            .zip([K8_SYMBOL, K320_SYMBOL])
        {
            let elf =
                crate::native::assemble(&emitted.s_text, Arch::Gfx1151).expect("native assemble");
            std::fs::write(dir.join(format!("{symbol}.test.s")), &emitted.s_text).unwrap();
            let co = dir.join(format!("{symbol}.test.co"));
            std::fs::write(&co, &elf).unwrap();
            let report = crate::pm_check::m7(&co, "gfx1151", symbol).expect("M7");
            println!("{symbol}: {report}");
            assert_eq!(report["obligations"], serde_json::json!({}), "{report}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(feature = "toolchain")]
    #[test]
    fn q8_rowtiles_m7() {
        let kinds = [
            WideKind::K8,
            WideKind::K320,
            WideKind::K2560,
            WideKind::K2560X2,
            WideKind::Wide,
        ];
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-lane-rows-q8-rowtiles-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for (emitted, kind) in build_rowtiles_gfx1151()
            .expect("q8 rowtiles")
            .into_iter()
            .zip(kinds)
        {
            let symbol = rowtile_symbol(kind);
            assert!(!symbol.contains("xbatch_wide_r"));
            let elf =
                crate::native::assemble(&emitted.s_text, Arch::Gfx1151).expect("native assemble");
            std::fs::write(dir.join(format!("{symbol}.test.s")), &emitted.s_text).unwrap();
            let co = dir.join(format!("{symbol}.test.co"));
            std::fs::write(&co, &elf).unwrap();
            let report = crate::pm_check::m7(&co, "gfx1151", symbol).expect("M7");
            println!("{symbol}: {report}");
            assert_eq!(report["obligations"], serde_json::json!({}), "{report}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rowtiles_freeze_abi_resources_and_selected_dags() {
        let emitted = build_rowtiles_gfx1151().expect("q8 rowtiles");
        let kinds = [
            WideKind::K8,
            WideKind::K320,
            WideKind::K2560,
            WideKind::K2560X2,
            WideKind::Wide,
        ];
        assert_eq!(emitted.len(), kinds.len());
        for (e, kind) in emitted.iter().zip(kinds) {
            let w = WideSpec {
                kind,
                arch: Arch::Gfx1151,
            };
            let spec = wide_kspec(w);
            assert_eq!(spec.arch, Arch::Gfx1151);
            assert_eq!(spec.symbol, rowtile_symbol(kind));
            assert!(spec.system_sgpr_workgroup_id_y);
            assert_eq!(spec.workgroup_size, if w.hc() { 512 } else { 256 });
            assert_eq!(
                w.kernarg_bytes(),
                if w.hc() {
                    44
                } else if w.x2() {
                    56
                } else {
                    36
                }
            );
            assert_eq!(e.shape.next_free_vgpr, if w.four() { 72 } else { 48 });
            assert!(e.shape.next_free_sgpr <= WIDE_SGPRS);
            assert!(!e.s_text.contains("scratch_"));
            assert!(e.s_text.contains("s_lshl_b32 s3, s3, 3"));
            assert!(e.s_text.contains("s_min_u32 s12, s12, 8"));
            assert!(e.s_text.contains("s_addc_u32 s7, s7, 0"));
            assert!(e.s_text.contains("s_addc_u32 s9, s9, 0"));
            assert!(e.s_text.contains("ds_swizzle_b32"));
            assert!(e.s_text.contains("ds_bpermute_b32"));
            assert!(!e.proof.loop_fixpoints.is_empty());
            #[cfg(feature = "toolchain")]
            crate::ledger_replay::replay_waits(&e.s_text, Arch::Gfx1151).unwrap();
            let (a, x, d) = (WV_ACC, WV_XS, WV_DQ);
            assert!(e
                .s_text
                .contains(&format!("v_fmac_f32_e32 v{a}, v{x}, v{d}")));
            if w.four() {
                assert!(e
                    .s_text
                    .contains(&format!("v_fmac_f32_e32 v{a}, v{d}, v{x}")));
                assert!(e
                    .s_text
                    .contains(&format!("v_mul_f32_e32 v{WV_TMP}, v{x}, v{d}")));
                assert!(e
                    .s_text
                    .contains(&format!("v_add_f32_e32 v{a}, v{a}, v{WV_TMP}")));
                for (dst, left, right) in [
                    (1, 1, 0),
                    (3, 3, 2),
                    (0, 3, 1),
                    (0, 0, 1),
                    (2, 2, 3),
                    (0, 0, 2),
                ] {
                    assert!(e.s_text.contains(&format!(
                        "v_add_f32_e32 v{}, v{}, v{}",
                        a + dst,
                        a + left,
                        a + right
                    )));
                }
            }
            if w.hc() {
                assert_eq!(e.proof.lds_slots.len(), 2);
                assert!(e.s_text.contains("v_exp_f32_e32"));
                assert!(e.s_text.contains("v_div_fmas_f32"));
                assert!(e.s_text.contains("v_div_fixup_f32"));
            }
        }
    }

    // CPU row/address/DAG oracle, not an actual hipcc singleton-object or
    // hardware HC exceptional-domain certificate. Unsupported emulator
    // instructions fail explicitly rather than replacing execution by a mock.
    mod rowtile_oracle {
        use super::*;
        use peacemaker_ir::emu::{Emulator, Launch, Memory};

        const KA: u64 = 0x10000;
        const A: [u64; 2] = [0x100000, 0x200000];
        // Exercise low-word pointer carries when a tile advances X and Y.
        const X: u64 = 0xffff_ff00;
        const Y: [u64; 2] = [0x2_ffff_ffe0, 0x3_ffff_ffe0];

        fn run(
            emu: &Emulator<'_>,
            kind: WideKind,
            k: usize,
            ms: [usize; 2],
            weights: &[Vec<u8>; 2],
            x: &[f32],
            hc: bool,
        ) -> [Vec<u32>; 2] {
            let rows = x.len() / k;
            assert_eq!(x.len(), rows * k);
            let w = WideSpec {
                kind,
                arch: Arch::Gfx1151,
            };
            let mut args = Vec::new();
            let ptrs = if w.x2() {
                vec![A[0], A[1], X, Y[0] + 16, Y[1] + 16]
            } else {
                vec![A[0], X, Y[0] + 16]
            };
            for ptr in ptrs {
                args.extend(ptr.to_le_bytes());
            }
            let dims = if w.x2() {
                vec![ms[0] as u32, ms[1] as u32, k as u32, rows as u32]
            } else {
                vec![ms[0] as u32, k as u32, rows as u32]
            };
            for dim in dims {
                args.extend(dim.to_le_bytes());
            }
            if w.hc() {
                args.extend(0.375f32.to_le_bytes());
                args.extend(u32::from(hc).to_le_bytes());
            }
            assert_eq!(args.len(), w.kernarg_bytes() as usize);
            let mut mem = Memory::default();
            mem.map("kernarg".into(), KA, args.clone()).unwrap();
            let xbytes: Vec<u8> = x.iter().flat_map(|f| f.to_le_bytes()).collect();
            mem.map("x".into(), X, xbytes.clone()).unwrap();
            for i in 0..2 {
                mem.map(format!("a{i}"), A[i], weights[i].clone()).unwrap();
                mem.map(
                    format!("y{i}"),
                    Y[i],
                    vec![0xa5; 32 + rows.div_ceil(8) * 8 * ms[i] * 4],
                )
                .unwrap();
            }
            let gx = if w.x2() {
                ms[0].div_ceil(8) + ms[1].div_ceil(8)
            } else {
                ms[0].div_ceil(if w.hc() { 2 } else { 8 })
            };
            // One extra dead workgroup in both dimensions must be harmless.
            for gy in 0..=rows.div_ceil(8) {
                for gx in 0..=gx {
                    emu.run_workgroup(
                        &Launch {
                            kernarg_ptr: KA,
                            workgroup: [gx as u32, gy as u32, 0],
                            block: [u32::from(w.workgroup_size()), 1, 1],
                            dynamic_lds: 0,
                        },
                        &mut mem,
                    )
                    .unwrap_or_else(|e| panic!("{kind:?} K{k} R{rows} wg({gx},{gy}): {e}"));
                }
            }
            let region = |name: &str| &mem.regions.iter().find(|r| r.name == name).unwrap().bytes;
            assert_eq!(region("kernarg"), &args);
            assert_eq!(region("x"), &xbytes);
            std::array::from_fn(|i| {
                assert_eq!(region(&format!("a{i}")), &weights[i]);
                let out = region(&format!("y{i}"));
                let len = if i == 0 || w.x2() {
                    rows * ms[i] * 4
                } else {
                    0
                };
                assert!(out[..16].iter().all(|&b| b == 0xa5));
                assert!(out[16 + len..].iter().all(|&b| b == 0xa5));
                out[16..16 + len]
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            })
        }

        fn weights(m: usize, k: usize, seed: usize) -> Vec<u8> {
            let mut out = Vec::with_capacity(m * k / 32 * 34);
            for block in 0..m * k / 32 {
                let d = 0x3000u16 | ((block * 73 + seed) as u16 & 1023);
                out.extend(d.to_le_bytes());
                for lane in 0..32 {
                    out.push((block * 37 + lane * 29 + seed * 11) as u8);
                }
            }
            // Empty x2 slots still have a mapped, never-read byte.
            if out.is_empty() {
                out.push(0x5a);
            }
            out
        }

        fn dot(a: &[u8], x: &[f32], four: bool, k640: bool) -> f32 {
            let nb = x.len() / 32;
            let mut lanes = [0.0f32; 32];
            for lane in 0..32 {
                let mut acc = [0.0f32; 4];
                for block in 0..nb {
                    let d = u16::from_le_bytes([a[34 * block], a[34 * block + 1]]);
                    let scale = f32::from_bits(
                        (((u32::from(d) >> 10) + 112) << 23) | ((u32::from(d) & 1023) << 13),
                    );
                    let dq = scale * f32::from(a[34 * block + 2 + lane] as i8);
                    let value = x[32 * block + lane];
                    let chain = if four { block & 3 } else { 0 };
                    acc[chain] = if four && block >= nb / 4 * 4 {
                        acc[chain] + value * dq
                    } else if four && !k640 {
                        dq.mul_add(value, acc[chain])
                    } else {
                        value.mul_add(dq, acc[chain])
                    };
                }
                lanes[lane] = if !four {
                    acc[0]
                } else if k640 {
                    (acc[0] + acc[1]) + (acc[2] + acc[3])
                } else {
                    (acc[3] + acc[2]) + (acc[1] + acc[0])
                };
            }
            for offset in [16, 8, 4, 2, 1] {
                let prev = lanes;
                for lane in 0..32 {
                    lanes[lane] = prev[lane]
                        + prev[if lane + offset < 32 {
                            lane + offset
                        } else {
                            lane
                        }];
                }
            }
            lanes[0]
        }

        #[test]
        fn every_logical_width_matches_c1_and_independent_raw_dag() {
            for (kind, ks, ms_cases) in [
                (
                    WideKind::K8,
                    vec![256, 768, 2048, 2304],
                    vec![[1, 0], [3, 0]],
                ),
                (WideKind::K320, vec![320], vec![[2, 0], [9, 0]]),
                (WideKind::K2560, vec![2560], vec![[7, 0], [8, 0]]),
                (
                    WideKind::K2560X2,
                    vec![2560],
                    vec![[0, 3], [3, 0], [1, 9], [9, 1], [0, 0]],
                ),
                (
                    WideKind::Wide,
                    (1..=8).map(|b| b * 32).chain([640, 1536]).collect(),
                    vec![[3, 0], [9, 0]],
                ),
            ] {
                let e = wide_impl(WideSpec {
                    kind,
                    arch: Arch::Gfx1151,
                })
                .unwrap();
                let elf = crate::native::assemble(&e.s_text, Arch::Gfx1151).unwrap();
                let lifted = peacemaker_lift::lift_object(
                    &elf,
                    peacemaker_lift::Options {
                        frontend: peacemaker_ir::inst::Frontend::Builder,
                    },
                )
                .unwrap();
                let emu = Emulator::new(&lifted.program, rowtile_symbol(kind)).unwrap();
                for k in ks {
                    let x: Vec<f32> = (0..64 * k)
                        .map(|i| {
                            let bits = (if i & 1 == 0 { 0 } else { 1 << 31 })
                                | (((117 + (i * 13 % 21)) as u32) << 23)
                                | ((i as u32).wrapping_mul(0x174321) & 0x7fffff);
                            f32::from_bits(bits)
                        })
                        .collect();
                    for ms in ms_cases.iter().copied() {
                        let a = [weights(ms[0], k, 1), weights(ms[1], k, 2)];
                        for hc in
                            [false, true]
                                .into_iter()
                                .take(if kind == WideKind::K8 { 2 } else { 1 })
                        {
                            let mut singleton: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
                            for row in 0..64 {
                                let one =
                                    run(&emu, kind, k, ms, &a, &x[row * k..(row + 1) * k], hc);
                                for i in 0..2 {
                                    singleton[i].extend_from_slice(&one[i]);
                                }
                                if !hc {
                                    for i in 0..if kind == WideKind::K2560X2 { 2 } else { 1 } {
                                        for m in 0..ms[i] {
                                            let wa = &a[i][m * k / 32 * 34..];
                                            let xr = &x[row * k..(row + 1) * k];
                                            let want = if kind == WideKind::K8 {
                                                let p: [f32; 8] = std::array::from_fn(|part| {
                                                    dot(
                                                        &wa[part * (k / 256) * 34..],
                                                        &xr[part * (k / 8)..(part + 1) * (k / 8)],
                                                        false,
                                                        false,
                                                    )
                                                });
                                                ((p[0] + p[1]) + (p[2] + p[3]))
                                                    + ((p[4] + p[5]) + (p[6] + p[7]))
                                            } else {
                                                dot(wa, xr, kind == WideKind::Wide, k == 640)
                                            };
                                            assert_eq!(
                                                one[i][m],
                                                want.to_bits(),
                                                "{kind:?} K{k} row{row} matrix{i} m{m}"
                                            );
                                        }
                                    }
                                }
                            }
                            for rows in 1..=64 {
                                let got = run(&emu, kind, k, ms, &a, &x[..rows * k], hc);
                                for i in 0..2 {
                                    assert_eq!(
                                        got[i],
                                        singleton[i][..got[i].len()],
                                        "{kind:?} K{k} M{ms:?} R{rows} hc{hc} matrix{i}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        #[test]
        fn invalid_domains_return_before_touching_unmapped_operands() {
            for kind in [
                WideKind::K8,
                WideKind::K320,
                WideKind::K2560,
                WideKind::K2560X2,
                WideKind::Wide,
            ] {
                let w = WideSpec {
                    kind,
                    arch: Arch::Gfx1151,
                };
                let e = wide_impl(w).unwrap();
                let elf = crate::native::assemble(&e.s_text, Arch::Gfx1151).unwrap();
                let lifted = peacemaker_lift::lift_object(
                    &elf,
                    peacemaker_lift::Options {
                        frontend: peacemaker_ir::inst::Frontend::Builder,
                    },
                )
                .unwrap();
                let emu = Emulator::new(&lifted.program, rowtile_symbol(kind)).unwrap();
                let good_k: u32 = match kind {
                    WideKind::K8 => 256,
                    WideKind::K320 => 320,
                    WideKind::K2560 | WideKind::K2560X2 => 2560,
                    WideKind::Wide => 640,
                };
                let mut cases = vec![
                    (0, good_k, 0),
                    (65, good_k, 0),
                    (u32::MAX, good_k, 0),
                    (1, 1, 0),
                    (1, 0, 0),
                    (1, good_k, 1),
                ];
                if kind != WideKind::K8 {
                    cases.push((1, if kind == WideKind::Wide { 1568 } else { 64 }, 0));
                }
                for (rows, k, gy) in cases {
                    let mut args = vec![0u8; w.kernarg_bytes() as usize];
                    let dims: Vec<(usize, u32)> = if w.x2() {
                        vec![(40, 3), (44, 5), (48, k), (52, rows)]
                    } else {
                        vec![(24, 3), (28, k), (32, rows)]
                    };
                    for (off, value) in dims {
                        args[off..off + 4].copy_from_slice(&value.to_le_bytes());
                    }
                    let mut mem = Memory::default();
                    mem.map("kernarg".into(), KA, args.clone()).unwrap();
                    emu.run_workgroup(&Launch {
                        kernarg_ptr: KA, workgroup: [0, gy, 0],
                        block: [u32::from(w.workgroup_size()), 1, 1], dynamic_lds: 0,
                    }, &mut mem).unwrap_or_else(|e| {
                        panic!("{kind:?} R{rows} K{k} tile{gy} must refuse before any operand access: {e}")
                    });
                    assert_eq!(mem.regions.len(), 1);
                    assert_eq!(mem.regions[0].bytes, args);
                }
            }
        }
    }
}
