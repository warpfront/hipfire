// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Prologue: kernel arguments, CTA raster, buffer descriptors, hoisted lane offsets, block-0
//! staging (decode, publish) and the first rendezvous. Workgroup ids on gfx1201 are `ttmp9` (x)
//! and `ttmp7[15:0]` (y).
use super::{publish, Gen, Lds, MetaD, MetaS, PayA, PayW, Raster, Tile, Wg};
use crate::kernels::common::{mem, op, s, sop, sr, v};
use crate::{
    insn::{Instruction, MemoryClass},
    Builder,
};
use peacemaker_author::{prime, End, Free, Ring};

/// SALU unsigned 32-bit division, LLVM's AMDGPU expansion: a float
/// reciprocal refined once, then two quotient corrections. Exact for all
/// operands. `x`, `y` are scratch.
fn udiv(b: &mut Builder, num: u8, den: u8, q: u8, r: u8, x: u8, y: u8) -> Result<(), String> {
    let o = |b: &mut Builder, t: String, d: &[u8], u: &[u8]| {
        op(
            b,
            t,
            &d.iter().map(|&n| s(n)).collect::<Vec<_>>(),
            &u.iter().map(|&n| s(n)).collect::<Vec<_>>(),
        )
    };
    o(b, format!("s_cvt_f32_u32 s{x}, s{den}"), &[x], &[den])?;
    o(b, format!("v_s_rcp_f32 s{x}, s{x}"), &[x], &[x])?;
    o(b, format!("s_mul_f32 s{x}, s{x}, 0x4f7ffffe"), &[x], &[x])?;
    o(b, format!("s_cvt_u32_f32 s{x}, s{x}"), &[x], &[x])?;
    o(b, format!("s_sub_co_i32 s{y}, 0, s{den}"), &[y], &[den])?;
    o(b, format!("s_mul_i32 s{y}, s{y}, s{x}"), &[y], &[y, x])?;
    o(b, format!("s_mul_hi_u32 s{y}, s{x}, s{y}"), &[y], &[x, y])?;
    o(b, format!("s_add_co_i32 s{x}, s{x}, s{y}"), &[x], &[x, y])?;
    o(
        b,
        format!("s_mul_hi_u32 s{q}, s{num}, s{x}"),
        &[q],
        &[num, x],
    )?;
    o(b, format!("s_mul_i32 s{y}, s{q}, s{den}"), &[y], &[q, den])?;
    o(
        b,
        format!("s_sub_co_i32 s{r}, s{num}, s{y}"),
        &[r],
        &[num, y],
    )?;
    for _ in 0..2 {
        o(b, format!("s_add_co_i32 s{x}, s{q}, 1"), &[x], &[q])?;
        o(
            b,
            format!("s_sub_co_i32 s{y}, s{r}, s{den}"),
            &[y],
            &[r, den],
        )?;
        o(b, format!("s_cmp_ge_u32 s{r}, s{den}"), &[], &[r, den])?;
        o(b, format!("s_cselect_b32 s{q}, s{x}, s{q}"), &[q], &[x, q])?;
        o(b, format!("s_cselect_b32 s{r}, s{y}, s{r}"), &[r], &[y, r])?;
    }
    Ok(())
}

/// VALU helper: `vdst` defined, `vuse`/`suse` read.
fn vop(b: &mut Builder, text: String, vdst: u8, vuse: &[u8], suse: &[u8]) -> Result<(), String> {
    let uses = vuse
        .iter()
        .map(|&n| v(n))
        .chain(suse.iter().map(|&n| s(n)))
        .collect::<Vec<_>>();
    op(b, text, &[v(vdst)], &uses)
}

/// Kernel arguments (`s_load_b256` x2) and the CTA raster: the tile origin `rs` (weight rows) and
/// `bs` (tokens). Ends before the out-of-range exits ([`outside_insn`]).
pub(crate) fn front(b: &mut Builder, g: &Gen) -> Result<(), String> {
    let a = g.args;
    // The tile origin is the tile index times the tile's rows / tokens.
    let (row_shift, tok_shift) = (
        g.spec.tile.rows().trailing_zeros(),
        g.spec.tile.tokens().trailing_zeros(),
    );
    let t: [u8; 8] = std::array::from_fn(|i| g.tmp + i as u8);
    mem(
        b,
        "s_load_b256 s[8:15], s[0:1], 0x0",
        &[sr(8, 8)],
        &[sr(0, 2)],
        MemoryClass::SmemLoad,
    )?;
    mem(
        b,
        "s_load_b256 s[16:23], s[0:1], 0x20",
        &[sr(16, 8)],
        &[sr(0, 2)],
        MemoryClass::SmemLoad,
    )?;
    match g.spec.raster {
        Raster::Banded => {
            // Banded raster: lin = y * gridDim.x + x; bands of 8 row tiles, rows fastest within
            // a band, then token tiles (bijective, partial last band).
            sop(
                b,
                format!("s_and_b32 s{}, ttmp7, 0xffff", t[0]),
                &[t[0]],
                &[],
            )?;
            sop(
                b,
                format!("s_mul_i32 s{}, s{}, s{}", t[1], t[0], a.bcx),
                &[t[1]],
                &[t[0], a.bcx],
            )?;
            sop(
                b,
                format!("s_add_co_i32 s{0}, s{0}, ttmp9", t[1]),
                &[t[1]],
                &[t[1]],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{}, s{}, 3", t[2], a.bcy),
                &[t[2]],
                &[a.bcy],
            )?;
            udiv(b, t[1], t[2], t[3], t[4], t[5], t[6])?; // band, in_band
            sop(
                b,
                format!("s_lshl_b32 s{}, s{}, 3", t[5], t[3]),
                &[t[5]],
                &[t[3]],
            )?; // band_row0
            sop(
                b,
                format!("s_sub_co_i32 s{}, s{}, s{}", t[6], a.bcx, t[5]),
                &[t[6]],
                &[a.bcx, t[5]],
            )?;
            sop(
                b,
                format!("s_min_u32 s{0}, s{0}, 8", t[6]),
                &[t[6]],
                &[t[6]],
            )?; // band_rows
            udiv(b, t[4], t[6], t[7], t[0], t[1], t[2])?; // token tile, row within band
            sop(
                b,
                format!("s_add_co_i32 s{0}, s{1}, s{0}", t[0], t[5]),
                &[t[0]],
                &[t[0], t[5]],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{}, s{}, {}", g.rs, t[0], row_shift),
                &[g.rs],
                &[t[0]],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{}, s{}, {}", g.bs, t[7], tok_shift),
                &[g.bs],
                &[t[7]],
            )?;
        }
        Raster::Linear => {
            sop(
                b,
                format!("s_lshl_b32 s48, ttmp9, {row_shift}"),
                &[g.rs],
                &[],
            )?;
            sop(
                b,
                format!("s_and_b32 s{}, ttmp7, 0xffff", t[0]),
                &[t[0]],
                &[],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{}, s{}, {}", g.bs, t[0], tok_shift),
                &[g.bs],
                &[t[0]],
            )?;
        }
    }
    Ok(())
}

/// The scalar compare of the tile-outside exits: `0` tile rows start at/past `M`, `1` tile tokens
/// start at/past `N` (none for the host grid; kept for safety).
pub(crate) fn outside_insn(g: &Gen, which: usize) -> Result<Instruction, String> {
    let a = g.args;
    match which {
        0 => Ok(Instruction::new(
            format!("s_cmp_ge_u32 s{}, s{}", g.rs, a.m),
            vec![],
            vec![s(g.rs), s(a.m)],
        )),
        1 => Ok(Instruction::new(
            format!("s_cmp_ge_u32 s{}, s{}", g.bs, a.n),
            vec![],
            vec![s(g.bs), s(a.n)],
        )),
        _ => Err(format!(
            "outside_insn: {which} is not 0 (rows) or 1 (tokens)"
        )),
    }
}

/// Scalars (200 * K/256 row stride, loop trips, group offsets, wave index, odd-lane mask) and the
/// buffer descriptors.
pub(crate) fn scalars(b: &mut Builder, g: &Gen) -> Result<(), String> {
    let a = g.args;
    let t: [u8; 8] = std::array::from_fn(|i| g.tmp + i as u8);
    // Scalars: 200 * K/256 (row stride), loop trips, group offsets, wave index, odd-lane mask.
    sop(
        b,
        format!("s_lshr_b32 s{}, s{}, 8", t[0], a.k),
        &[t[0]],
        &[a.k],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{}, s{}, 0xc8", g.gpr200, t[0]),
        &[g.gpr200],
        &[t[0]],
    )?;
    sop(
        b,
        format!("s_add_co_i32 s{}, s{}, -1", g.trips, t[0]),
        &[g.trips],
        &[t[0]],
    )?;
    sop(b, format!("s_mov_b32 s{}, 0", g.goff), &[g.goff], &[])?;
    sop(
        b,
        format!(
            "s_lshl_b32 s{}, s{}, {}",
            g.goff1,
            g.gpr200,
            g.spec.tile.round_rows().trailing_zeros()
        ),
        &[g.goff1],
        &[g.gpr200],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, 0xaaaaaaaa", g.omask),
        &[g.omask],
        &[],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{}, v0", g.wave),
        &[s(g.wave)],
        &[v(0)],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{0}, s{0}, 5", g.wave),
        &[g.wave],
        &[g.wave],
    )?;

    // Descriptors: Xq per 128-K block (N * 136 records, OOB tokens read 0); weights
    // `M * gpr200` records (OOB rows read 0), word3 0x31004000.
    let [a0, a1] = g.srd_a;
    sop(b, format!("s_mov_b32 s{}, s{}", a0, a.xq), &[a0], &[a.xq])?;
    sop(
        b,
        format!("s_and_b32 s{}, s{}, 0xffff", a0 + 1, a.xq + 1),
        &[a0 + 1],
        &[a.xq + 1],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{}, s{}, 0x88", a0 + 2, a.n),
        &[a0 + 2],
        &[a.n],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, 0x31004000", a0 + 3),
        &[a0 + 3],
        &[],
    )?;
    sop(
        b,
        format!("s_add_co_u32 s{}, s{}, s{}", a1, a0, a0 + 2),
        &[a1],
        &[a0, a0 + 2],
    )?;
    sop(
        b,
        format!("s_add_co_ci_u32 s{}, s{}, 0", a1 + 1, a0 + 1),
        &[a1 + 1],
        &[a0 + 1],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", a1 + 2, a0 + 2),
        &[a1 + 2],
        &[a0 + 2],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, 0x31004000", a1 + 3),
        &[a1 + 3],
        &[],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{}, s{}, 1", g.step2, a0 + 2),
        &[g.step2],
        &[a0 + 2],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, 0", g.step2 + 1),
        &[g.step2 + 1],
        &[],
    )?;
    let w = g.srd_w;
    sop(b, format!("s_mov_b32 s{}, s{}", w, a.a), &[w], &[a.a])?;
    sop(
        b,
        format!("s_and_b32 s{}, s{}, 0xffff", w + 1, a.a + 1),
        &[w + 1],
        &[a.a + 1],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{}, s{}, s{}", w + 2, a.m, g.gpr200),
        &[w + 2],
        &[a.m, g.gpr200],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, 0x31004000", w + 3),
        &[w + 3],
        &[],
    )?;
    Ok(())
}

/// Hoisted lane offsets and fold addresses (`v0` = tid).
pub(crate) fn lanes(b: &mut Builder, g: &Gen) -> Result<(), String> {
    let t: [u8; 8] = std::array::from_fn(|i| g.tmp + i as u8);
    // Hoisted lane offsets (v0 = tid). Temporaries live in v1.. until K_BEGIN.
    let (ml, lane, x4, tt0, c, c0, c1, m, half, gp0) =
        (1u8, 2u8, 3u8, 5u8, 6u8, 7u8, 8u8, 9u8, 10u8, 11u8);
    let (p, q, r) = (12u8, 13u8, 14u8);
    vop(b, format!("v_and_b32_e32 v{ml}, 15, v0"), ml, &[0], &[])?;
    vop(b, format!("v_and_b32_e32 v{lane}, 31, v0"), lane, &[0], &[])?;
    vop(b, format!("v_lshrrev_b32_e32 v{x4}, 2, v0"), x4, &[0], &[])?;
    vop(b, format!("v_and_b32_e32 v{x4}, 4, v{x4}"), x4, &[x4], &[])?;
    // reader lane position p = lane ^ (kg << 2)
    vop(
        b,
        format!("v_xor_b32_e32 v{lane}, v{lane}, v{x4}"),
        lane,
        &[lane, x4],
        &[],
    )?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{x4}, 4, v{lane}"),
        x4,
        &[lane],
        &[],
    )?;
    let tile = g.spec.tile;
    let (th, wp, x) = (t[0], t[1], t[2]);
    // Wave `w` owns token half `w % tok_halves` and row pair `w / tok_halves`.
    let tok_halves = tile.tok_halves();
    sop(
        b,
        format!("s_and_b32 s{th}, s{}, {}", g.wave, tok_halves - 1),
        &[th],
        &[g.wave],
    )?;
    sop(
        b,
        format!(
            "s_lshr_b32 s{wp}, s{}, {}",
            g.wave,
            tok_halves.trailing_zeros()
        ),
        &[wp],
        &[g.wave],
    )?;
    sop(b, format!("s_lshl_b32 s{x}, s{th}, 12"), &[x], &[th])?;
    vop(
        b,
        format!("v_add_nc_u32_e32 v{}, s{x}, v{x4}", g.fr_a),
        g.fr_a,
        &[x4],
        &[x],
    )?;
    sop(b, format!("s_lshl_b32 s{x}, s{wp}, 11"), &[x], &[wp])?;
    vop(
        b,
        format!("v_add_nc_u32_e32 v{}, s{x}, v{x4}", g.fr_w),
        g.fr_w,
        &[x4],
        &[x],
    )?;
    // Staging coordinates.
    vop(
        b,
        format!("v_lshrrev_b32_e32 v{tt0}, 2, v0"),
        tt0,
        &[0],
        &[],
    )?;
    vop(b, format!("v_and_b32_e32 v{c}, 3, v0"), c, &[0], &[])?;
    vop(b, format!("v_and_b32_e32 v{c0}, 1, v{c}"), c0, &[c], &[])?;
    vop(
        b,
        format!("v_lshrrev_b32_e32 v{c1}, 1, v{c}"),
        c1,
        &[c],
        &[],
    )?;
    vop(b, format!("v_and_b32_e32 v{m}, 15, v{tt0}"), m, &[tt0], &[])?;
    vop(
        b,
        format!("v_lshrrev_b32_e32 v{half}, 6, v0"),
        half,
        &[0],
        &[],
    )?;
    vop(
        b,
        format!("v_and_b32_e32 v{half}, 1, v{half}"),
        half,
        &[half],
        &[],
    )?;
    vop(
        b,
        format!("v_lshrrev_b32_e32 v{gp0}, 7, v0"),
        gp0,
        &[0],
        &[],
    )?;
    // Activation slot: gp0*2048 + c0*512 + p_s*16 + 8*(half ^ c0), p_s = (m | c1<<4) ^ (c1<<2).
    for (pos, sel, dst, base) in [(c1, c0, g.st_alds, 9u32), (c0, c1, g.st_wlds, 10)] {
        // `pos` is the lane-half bit (kg), `sel` the slot-select / plane bit.
        vop(
            b,
            format!("v_lshlrev_b32_e32 v{p}, 4, v{pos}"),
            p,
            &[pos],
            &[],
        )?;
        vop(b, format!("v_or_b32_e32 v{p}, v{p}, v{m}"), p, &[p, m], &[])?;
        vop(
            b,
            format!("v_lshlrev_b32_e32 v{q}, 2, v{pos}"),
            q,
            &[pos],
            &[],
        )?;
        vop(
            b,
            format!("v_xor_b32_e32 v{p}, v{p}, v{q}"),
            p,
            &[p, q],
            &[],
        )?;
        vop(b, format!("v_lshlrev_b32_e32 v{p}, 4, v{p}"), p, &[p], &[])?;
        vop(
            b,
            format!("v_xor_b32_e32 v{q}, v{half}, v{sel}"),
            q,
            &[half, sel],
            &[],
        )?;
        vop(b, format!("v_lshlrev_b32_e32 v{q}, 3, v{q}"), q, &[q], &[])?;
        vop(
            b,
            format!("v_add_nc_u32_e32 v{p}, v{p}, v{q}"),
            p,
            &[p, q],
            &[],
        )?;
        // plane term: A: e*512 (c0 << 9); W: sb*1024 (c1 << 10)
        vop(
            b,
            format!("v_lshlrev_b32_e32 v{q}, {base}, v{sel}"),
            q,
            &[sel],
            &[],
        )?;
        vop(
            b,
            format!("v_add_nc_u32_e32 v{p}, v{p}, v{q}"),
            p,
            &[p, q],
            &[],
        )?;
        vop(
            b,
            format!("v_lshlrev_b32_e32 v{q}, 11, v{gp0}"),
            q,
            &[gp0],
            &[],
        )?;
        vop(
            b,
            format!("v_add_nc_u32_e32 v{dst}, v{p}, v{q}"),
            dst,
            &[p, q],
            &[],
        )?;
    }
    // Global offsets: activation token (bs + R) * 136 + c * 8; weight row (rs + R) * gpr200 + c * 12.
    vop(b, format!("v_lshlrev_b32_e32 v{r}, 3, v{c}"), r, &[c], &[])?;
    vop(
        b,
        format!("v_add_nc_u32_e32 v{q}, s{}, v{tt0}", g.bs),
        q,
        &[tt0],
        &[g.bs],
    )?;
    vop(
        b,
        format!("v_mad_u32_u24 v{}, v{q}, 0x88, v{r}", g.st_a),
        g.st_a,
        &[q, r],
        &[],
    )?;
    vop(
        b,
        format!("v_add_nc_u32_e32 v{q}, s{}, v{tt0}", g.rs),
        q,
        &[tt0],
        &[g.rs],
    )?;
    vop(
        b,
        format!("v_mul_lo_u32 v{q}, v{q}, s{}", g.gpr200),
        q,
        &[q],
        &[g.gpr200],
    )?;
    vop(
        b,
        format!("v_mad_u32_u24 v{}, v{c}, 12, v{q}", g.st_w),
        g.st_w,
        &[c, q],
        &[],
    )?;
    // Metadata lanes: slot tok = tid >> 1 of plane w = tid & 1.
    let (tok, wl) = (15u8, 16u8);
    vop(
        b,
        format!("v_lshrrev_b32_e32 v{tok}, 1, v0"),
        tok,
        &[0],
        &[],
    )?;
    vop(b, format!("v_and_b32_e32 v{wl}, 1, v0"), wl, &[0], &[])?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{q}, 9, v{wl}"),
        q,
        &[wl],
        &[],
    )?;
    if tile == Tile::T128x128 {
        vop(
            b,
            format!("v_lshl_add_u32 v{}, v{tok}, 2, v{q}", g.meta_lds),
            g.meta_lds,
            &[tok, q],
            &[],
        )?;
        vop(
            b,
            format!("v_add_nc_u32_e32 v{q}, s{}, v{tok}", g.bs),
            q,
            &[tok],
            &[g.bs],
        )?;
        vop(
            b,
            format!("v_mul_u32_u24_e32 v{q}, 0x88, v{q}"),
            q,
            &[q],
            &[],
        )?;
        vop(
            b,
            format!("v_lshl_add_u32 v{}, v{wl}, 2, v{q}", g.ds_voff),
            g.ds_voff,
            &[wl, q],
            &[],
        )?;
        vop(
            b,
            format!("v_add_nc_u32_e32 v{q}, s{}, v{tok}", g.rs),
            q,
            &[tok],
            &[g.rs],
        )?;
        vop(
            b,
            format!("v_mul_lo_u32 v{}, v{q}, s{}", g.sz_voff, g.gpr200),
            g.sz_voff,
            &[q],
            &[g.gpr200],
        )?;
    } else {
        // 512 threads: lane `tid` serves slot `tok = tid >> 1` (0..256) of plane `wl`. A plane of
        // 256 slots is chunked per 128 (`1024 * chunk` bytes; `d` / `sc` then `float(s)` / `z'`
        // 512 bytes on); a plane of 128 slots is served twice, by `tok` and `tok + 128`, which
        // store the same words to the same addresses.
        let (tm, ch) = (17u8, 18u8);
        vop(
            b,
            format!("v_and_b32_e32 v{tm}, 0x7f, v{tok}"),
            tm,
            &[tok],
            &[],
        )?;
        let mut plane_addr = |dst: u8, slots: u32| -> Result<(), String> {
            if slots == 256 {
                vop(
                    b,
                    format!("v_lshrrev_b32_e32 v{ch}, 7, v{tok}"),
                    ch,
                    &[tok],
                    &[],
                )?;
                vop(
                    b,
                    format!("v_lshlrev_b32_e32 v{ch}, 10, v{ch}"),
                    ch,
                    &[ch],
                    &[],
                )?;
                vop(
                    b,
                    format!("v_add_nc_u32_e32 v{ch}, v{ch}, v{q}"),
                    ch,
                    &[ch, q],
                    &[],
                )?;
                vop(
                    b,
                    format!("v_lshl_add_u32 v{dst}, v{tm}, 2, v{ch}"),
                    dst,
                    &[tm, ch],
                    &[],
                )
            } else {
                vop(
                    b,
                    format!("v_lshl_add_u32 v{dst}, v{tm}, 2, v{q}"),
                    dst,
                    &[tm, q],
                    &[],
                )
            }
        };
        plane_addr(g.meta_lds, tile.tokens())?;
        plane_addr(g.meta_lds_sz, tile.rows())?;
        let tok_idx = if tile.tokens() == 256 { tok } else { tm };
        let row_idx = if tile.rows() == 256 { tok } else { tm };
        vop(
            b,
            format!("v_add_nc_u32_e32 v{q}, s{}, v{tok_idx}", g.bs),
            q,
            &[tok_idx],
            &[g.bs],
        )?;
        vop(
            b,
            format!("v_mul_u32_u24_e32 v{q}, 0x88, v{q}"),
            q,
            &[q],
            &[],
        )?;
        vop(
            b,
            format!("v_lshl_add_u32 v{}, v{wl}, 2, v{q}", g.ds_voff),
            g.ds_voff,
            &[wl, q],
            &[],
        )?;
        vop(
            b,
            format!("v_add_nc_u32_e32 v{q}, s{}, v{row_idx}", g.rs),
            q,
            &[row_idx],
            &[g.rs],
        )?;
        vop(
            b,
            format!("v_mul_lo_u32 v{}, v{q}, s{}", g.sz_voff, g.gpr200),
            g.sz_voff,
            &[q],
            &[g.gpr200],
        )?;
    }
    // Fold addresses: row planes wt_pair*128 + kg*32, token planes tok_half*256 + m_lane*4 (plus
    // the 128-chunk term of the 256-wide planes).
    vop(b, format!("v_lshrrev_b32_e32 v{q}, 4, v0"), q, &[0], &[])?;
    vop(b, format!("v_and_b32_e32 v{q}, 1, v{q}"), q, &[q], &[])?;
    vop(b, format!("v_lshlrev_b32_e32 v{q}, 5, v{q}"), q, &[q], &[])?;
    sop(b, format!("s_lshl_b32 s{x}, s{wp}, 7"), &[x], &[wp])?;
    if tile.rows() == 256 {
        // Row pairs 4..8 sit in the second 128-row chunk of the plane (1024 bytes on).
        let y = t[3];
        sop(b, format!("s_lshr_b32 s{y}, s{wp}, 2"), &[y], &[wp])?;
        sop(b, format!("s_lshl_b32 s{y}, s{y}, 9"), &[y], &[y])?;
        sop(b, format!("s_add_co_i32 s{x}, s{x}, s{y}"), &[x], &[x, y])?;
    }
    vop(
        b,
        format!("v_add_nc_u32_e32 v{}, s{x}, v{q}", g.sc_addr),
        g.sc_addr,
        &[q],
        &[x],
    )?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{q}, 2, v{ml}"),
        q,
        &[ml],
        &[],
    )?;
    sop(b, format!("s_lshl_b32 s{x}, s{th}, 8"), &[x], &[th])?;
    if tile.tokens() == 256 {
        // Token halves 2..4 sit in the second 128-token chunk of the plane (1024 bytes on).
        let y = t[3];
        sop(b, format!("s_lshr_b32 s{y}, s{th}, 1"), &[y], &[th])?;
        sop(b, format!("s_lshl_b32 s{y}, s{y}, 9"), &[y], &[y])?;
        sop(b, format!("s_add_co_i32 s{x}, s{x}, s{y}"), &[x], &[x, y])?;
    }
    vop(
        b,
        format!("v_add_nc_u32_e32 v{}, s{x}, v{q}", g.d_addr),
        g.d_addr,
        &[q],
        &[x],
    )?;
    Ok(())
}

/// Block 0's raw fetches: slab 0 K32 half 0, the weight dwords, the token word and row header.
pub(crate) fn fetch0(b: &mut Builder, g: &Gen) -> Result<(), String> {
    let a0r = g.srd_a[0];
    publish::fetch_a(b, g, 0, 0, a0r)?;
    publish::fetch_w(b, g, 8)?;
    b.clause(|b| {
        publish::vload(b, g.ds_nx, 1, g.ds_voff, a0r, None, 0)?;
        publish::vload(b, g.sz_nx, 1, g.sz_voff, g.srd_w, Some(g.goff), 0)
    })
    .map_err(|e| format!("prologue meta clause: {e}"))
}

/// The float accumulators to zero (32 dual moves).
pub(crate) fn zero_acc(b: &mut Builder, g: &Gen) -> Result<(), String> {
    for i in (0..64u8).step_by(2) {
        let r = g.acc + i;
        op(
            b,
            format!("v_dual_mov_b32 v{r}, 0 :: v_dual_mov_b32 v{}, 0", r + 1),
            &[v(r), v(r + 1)],
            &[],
        )?;
    }
    Ok(())
}

/// Kernel arguments, CTA raster, out-of-range exits, scalars and hoisted lane offsets: everything
/// before block 0's staging.
pub(crate) fn pre(wg: &mut Wg, g: &Gen, end: &End) -> Result<(), String> {
    front(wg.isa(), g)?;
    // Tiles past the problem exit (none for the host grid; kept for safety).
    let outside = wg.scmp_wg_uniform(outside_insn(g, 0)?)?;
    wg.exit_if(outside, end)?;
    let outside = wg.scmp_wg_uniform(outside_insn(g, 1)?)?;
    wg.exit_if(outside, end)?;
    scalars(wg.isa(), g)?;
    lanes(wg.isa(), g)
}

pub(crate) fn emit(
    wg: &mut Wg,
    g: &Gen,
    end: &End,
    (ra, rw, rd, rs): (
        Ring<PayA, Free, Free>,
        Ring<PayW, Free, Free>,
        Ring<MetaD, Free, Free>,
        Ring<MetaS, Free, Free>,
    ),
) -> Result<Lds, String> {
    pre(wg, g, end)?;

    // Block 0: slab 0 (K32 half 0 now, half 1 after half 0 is published), token plane 0, row plane 0.
    let a0r = g.srd_a[0];
    fetch0(wg.isa(), g)?;
    zero_acc(wg.isa(), g)?;
    let (mut sa, mut sw) = (wg.begin_write(ra), wg.begin_write(rw));
    for rnd in 0..g.w_pf.len() {
        publish::decode_round(wg.isa(), g, rnd)?;
        sw = publish::store_w_round(wg, g, sw, 0, rnd)?;
    }
    sa = publish::store_a_half(wg, g, sa, 0, 0)?;
    // The raw staging registers are still sources of the sb 0 stores: retire them first.
    wg.isa().wait_all()?;
    publish::fetch_a(wg.isa(), g, 0, 1, a0r)?;
    sa = publish::store_a_half(wg, g, sa, 0, 1)?;
    let ((rd, pd), (rs, ps)) = publish::publish_meta(wg, g, rd, rs)?;
    let (pa, pw) = (sa.1, sw.1);
    let (ra, rw) = (sa.0, sw.0);
    let (da, dw, dd, ds) = wg.wait_all((pa, pw, pd, ps))?;
    let arrived = wg.signal((prime(ra, da), prime(rw, dw), prime(rd, dd), prime(rs, ds)))?;
    // Seed the ring: slab 1 of block 0 (K32 half 0 and the weight codes).
    let b = wg.isa();
    publish::fetch_a(b, g, 1, 0, a0r)?;
    publish::fetch_w(b, g, 8 + 48)?;
    super::kloop::seed_fill(b, g)?;
    wg.wait_arrived(arrived)
}
