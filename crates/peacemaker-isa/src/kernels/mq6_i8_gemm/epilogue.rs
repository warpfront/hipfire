// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Masked store path. Lane (m_lane, k_grp) of wave (wt_pair, tok_half) owns, for token block `nb`
//! and row group `rg`, rows `wt_pair*32 + rg*16 + 8*k_grp + j` (j = 0..7) of token
//! `tok_half*64 + nb*16 + m_lane`: two b128 stores for F32, or per 4-row quad one b64 store of
//! packed BF16 (`isfinite(v) ? RNE : u >> 16`, `v = 0 + y`). EXEC selects quads with row < M
//! (`M % 4 == 0`) and tokens < N.
use super::{Epi, Gen, Raster, Wg, EPI};
use crate::kernels::common::{lit, mem, op, s, sop, sr, v, vr};
use crate::{
    insn::{Instruction, MemoryClass},
    Builder,
};
use peacemaker_author::End;

pub(crate) struct Epilogue {
    pub(crate) tok: u8,
    pub(crate) row: u8,
    pub(crate) off: u8,
    lim: u8,
    lim_off: [u8; 4],
    ntok: [u8; 4],
    rm: [u8; 4],
    tm: [u8; 4],
    nbo: [u8; 4],
    fin: u8,
}

pub(crate) fn layout(g: &Gen) -> Epilogue {
    let e = g.epi_s;
    Epilogue {
        tok: g.f[0],
        row: g.f[0] + 1,
        off: g.f[0] + 2,
        lim: e,
        lim_off: [e + 1, e + 2, e + 3, e + 4],
        ntok: [e + 5, e + 6, e + 7, e + 8],
        rm: [e + 9, e + 10, e + 11, e + 12],
        tm: [e + 13, e + 14, e + 15, e + 16],
        nbo: [e + 17, e + 18, e + 19, e + 20],
        fin: e + 21,
    }
}

fn vop(
    b: &mut Builder,
    text: String,
    defs: &[u8],
    uses: &[crate::reg::RegRef],
) -> Result<(), String> {
    op(
        b,
        text,
        &defs.iter().map(|&n| v(n)).collect::<Vec<_>>(),
        uses,
    )
}

pub(crate) fn emit(wg: &mut Wg, g: &Gen, _end: &End) -> Result<(), String> {
    wg.label(EPI)?;
    body(wg.isa(), g)
}

/// Everything after the [`EPI`] label: the tile-relative output descriptor, the lane masks and the
/// masked stores.
pub(crate) fn body(b: &mut Builder, g: &Gen) -> Result<(), String> {
    if g.spec.epi == Epi::SetNr {
        return Err("set_nr is emitted by nr_epilogue".into());
    }
    // Below the K-loop's priority 1: co-resident K-loops issue first.
    if g.spec.raster == Raster::Banded {
        op(b, "s_setprio 0", &[], &[])?;
    }
    let e = setup(b, g)?;
    stores(b, g, &e)
}

/// The tile-relative output descriptor, the lane's token and row (`e.tok`, `e.row`), its store
/// offset and the row / token masks.
pub(crate) fn setup(b: &mut Builder, g: &Gen) -> Result<Epilogue, String> {
    let a = g.args;
    let bf16 = g.spec.epi == Epi::SetBf16;
    let e = layout(g);
    let y = g.srd_y;
    let eb = if bf16 { 1 } else { 2 }; // log2(element bytes)
                                       // Tile-relative descriptor: Y + (bs * M + first row) * element size.
    sop(
        b,
        format!("s_mul_i32 s{y}, s{}, s{}", g.bs, a.m),
        &[y],
        &[g.bs, a.m],
    )?;
    sop(
        b,
        format!("s_mul_hi_u32 s{}, s{}, s{}", y + 1, g.bs, a.m),
        &[y + 1],
        &[g.bs, a.m],
    )?;
    sop(
        b,
        format!("s_add_co_u32 s{y}, s{y}, s{}", g.rs),
        &[y],
        &[y, g.rs],
    )?;
    sop(
        b,
        format!("s_add_co_ci_u32 s{0}, s{0}, 0", y + 1),
        &[y + 1],
        &[y + 1],
    )?;
    op(
        b,
        format!("s_lshl_b64 s[{y}:{}], s[{y}:{}], {eb}", y + 1, y + 1),
        &[sr(y, 2)],
        &[sr(y, 2)],
    )?;
    sop(
        b,
        format!("s_add_co_u32 s{y}, s{y}, s{}", a.y),
        &[y],
        &[y, a.y],
    )?;
    sop(
        b,
        format!("s_add_co_ci_u32 s{0}, s{0}, s{1}", y + 1, a.y + 1),
        &[y + 1],
        &[y + 1, a.y + 1],
    )?;
    sop(
        b,
        format!("s_and_b32 s{0}, s{0}, 0xffff", y + 1),
        &[y + 1],
        &[y + 1],
    )?;
    sop(b, format!("s_mov_b32 s{}, -1", y + 2), &[y + 2], &[])?;
    sop(
        b,
        format!("s_mov_b32 s{}, 0x31004000", y + 3),
        &[y + 3],
        &[],
    )?;
    // Lane coordinates from the fold addresses: token tok_half*64 + m_lane, row wt_pair*32 + 8*k_grp.
    op(
        b,
        format!("v_lshrrev_b32_e32 v{}, 2, v{}", e.tok, g.d_addr),
        &[v(e.tok)],
        &[v(g.d_addr)],
    )?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{}, 2, v{}", e.row, g.sc_addr),
        &[v(e.row)],
        &[v(g.sc_addr)],
    )?;
    // A 256-wide plane is chunked per 128 (1024 bytes): drop the 128 slots the chunk gap adds.
    let t = g.f[0] + 3;
    for (reg, addr, wide) in [
        (e.tok, g.d_addr, g.spec.tile.tokens() == 256),
        (e.row, g.sc_addr, g.spec.tile.rows() == 256),
    ] {
        if wide {
            op(
                b,
                format!("v_lshrrev_b32_e32 v{t}, 10, v{addr}"),
                &[v(t)],
                &[v(addr)],
            )?;
            op(
                b,
                format!("v_lshlrev_b32_e32 v{t}, 7, v{t}"),
                &[v(t)],
                &[v(t)],
            )?;
            op(
                b,
                format!("v_sub_nc_u32_e32 v{reg}, v{reg}, v{t}"),
                &[v(reg)],
                &[v(reg), v(t)],
            )?;
        }
    }
    op(
        b,
        format!("v_mul_lo_u32 v{}, v{}, s{}", e.off, e.tok, a.m),
        &[v(e.off)],
        &[v(e.tok), s(a.m)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{0}, v{0}, v{1}", e.off, e.row),
        &[v(e.off)],
        &[v(e.off), v(e.row)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{0}, {eb}, v{0}", e.off),
        &[v(e.off)],
        &[v(e.off)],
    )?;
    // Masks: quad (rg, c) valid iff row + rg*16 + 4c < M - first row; token block nb valid iff
    // token + 16*nb < N - bs.
    sop(
        b,
        format!("s_sub_co_i32 s{}, s{}, s{}", e.lim, a.m, g.rs),
        &[e.lim],
        &[a.m, g.rs],
    )?;
    for rg in 0..2 {
        for c in 0..2 {
            let i = rg * 2 + c;
            sop(
                b,
                format!(
                    "s_sub_co_i32 s{}, s{}, {}",
                    e.lim_off[i],
                    e.lim,
                    lit((16 * rg + 4 * c) as u32)
                ),
                &[e.lim_off[i]],
                &[e.lim],
            )?;
            op(
                b,
                format!(
                    "v_cmp_gt_i32_e64 s{}, s{}, v{}",
                    e.rm[i], e.lim_off[i], e.row
                ),
                &[s(e.rm[i])],
                &[s(e.lim_off[i]), v(e.row)],
            )?;
        }
    }
    sop(
        b,
        format!("s_sub_co_i32 s{}, s{}, s{}", e.ntok[0], a.n, g.bs),
        &[e.ntok[0]],
        &[a.n, g.bs],
    )?;
    for nb in 1..4 {
        sop(
            b,
            format!("s_sub_co_i32 s{}, s{}, 16", e.ntok[nb], e.ntok[nb - 1]),
            &[e.ntok[nb]],
            &[e.ntok[nb - 1]],
        )?;
    }
    for nb in 0..4 {
        op(
            b,
            format!(
                "v_cmp_gt_i32_e64 s{}, s{}, v{}",
                e.tm[nb], e.ntok[nb], e.tok
            ),
            &[s(e.tm[nb])],
            &[s(e.ntok[nb]), v(e.tok)],
        )?;
    }
    // Per token block soffset: nb * 16 * M * element bytes.
    sop(b, format!("s_mov_b32 s{}, 0", e.nbo[0]), &[e.nbo[0]], &[])?;
    sop(
        b,
        format!("s_lshl_b32 s{}, s{}, {}", e.nbo[1], a.m, 4 + eb),
        &[e.nbo[1]],
        &[a.m],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{}, s{}, 1", e.nbo[2], e.nbo[1]),
        &[e.nbo[2]],
        &[e.nbo[1]],
    )?;
    sop(
        b,
        format!("s_add_co_i32 s{}, s{}, s{}", e.nbo[3], e.nbo[2], e.nbo[1]),
        &[e.nbo[3]],
        &[e.nbo[2], e.nbo[1]],
    )?;
    op(b, "s_wait_alu depctr_va_sdst(0)", &[], &[])?;
    Ok(e)
}

/// The masked stores of the eight accumulator fragments (F32 `buffer_store_b128` per row quad, or
/// packed BF16).
pub(crate) fn stores(b: &mut Builder, g: &Gen, e: &Epilogue) -> Result<(), String> {
    let bf16 = g.spec.epi == Epi::SetBf16;
    let y = g.srd_y;
    let acc = |nb: usize, rg: usize, c: usize| g.acc + (8 * (nb * 2 + rg) + 4 * c) as u8;
    for nb in 0..4usize {
        for rg in 0..2usize {
            for c in 0..2usize {
                sop(
                    b,
                    format!("s_and_b32 exec_lo, s{}, s{}", e.rm[rg * 2 + c], e.tm[nb]),
                    &[],
                    &[e.rm[rg * 2 + c], e.tm[nb]],
                )?;
                if !bf16 {
                    let data = acc(nb, rg, c);
                    let imm = (64 * rg + 16 * c) as u32;
                    mem(
                        b,
                        format!(
                            "buffer_store_b128 v[{}:{}], v{}, s[{y}:{}], s{} offen{}",
                            data,
                            data + 3,
                            e.off,
                            y + 3,
                            e.nbo[nb],
                            if imm == 0 {
                                String::new()
                            } else {
                                format!(" offset:{imm}")
                            }
                        ),
                        &[],
                        &[v(e.off), sr(y, 4), s(e.nbo[nb]), vr(data, 4)],
                        MemoryClass::VmemStore,
                    )?;
                } else {
                    // Output dwords of every quad live in v0.. (dead int32 accumulators): one
                    // register pair per quad, so no store source is ever rewritten.
                    let q = (nb * 2 + rg) * 2 + c;
                    let out = [g.cacc + 2 * q as u8, g.cacc + 2 * q as u8 + 1];
                    for p in 0..2usize {
                        let (lo, hi) = (
                            acc(nb, rg, c) + 2 * p as u8,
                            acc(nb, rg, c) + 2 * p as u8 + 1,
                        );
                        bf16_pair(b, &e, g, lo, hi, out[p])?;
                    }
                    // One 8-byte store per quad (`y` 8-byte aligned, `M % 4 == 0`).
                    let imm = (32 * rg + 8 * c) as u32;
                    mem(
                        b,
                        format!(
                            "buffer_store_b64 v[{}:{}], v{}, s[{y}:{}], s{} offen{}",
                            out[0],
                            out[1],
                            e.off,
                            y + 3,
                            e.nbo[nb],
                            if imm == 0 {
                                String::new()
                            } else {
                                format!(" offset:{imm}")
                            }
                        ),
                        &[],
                        &[v(e.off), sr(y, 4), s(e.nbo[nb]), vr(out[0], 2)],
                        MemoryClass::VmemStore,
                    )?;
                }
            }
        }
    }
    Ok(())
}

/// `out = bf16(lo) | bf16(hi) << 16` with the incumbent's rounding:
/// `u = as_uint(0 + x)`, finite ? `(u + 0x7fff + ((u >> 16) & 1)) >> 16` : `u >> 16`.
/// Temporaries: the dead fragment registers `v{f0 + 4 ..}`; `fin` SGPR holds the finite mask.
fn bf16_pair(
    b: &mut Builder,
    e: &Epilogue,
    g: &Gen,
    lo: u8,
    hi: u8,
    out: u8,
) -> Result<(), String> {
    let t = g.f[0] + 4;
    // Two values, each into (r = t, t + 1 for lo / hi): u, r, hi16, m.
    let mut res = [0u8; 2];
    for (i, x) in [lo, hi].into_iter().enumerate() {
        let (u, r, h16, m) = (
            t + 4 * i as u8,
            t + 4 * i as u8 + 1,
            t + 4 * i as u8 + 2,
            t + 4 * i as u8 + 3,
        );
        vop(b, format!("v_add_f32_e32 v{u}, 0, v{x}"), &[u], &[v(x)])?;
        vop(b, format!("v_bfe_u32 v{r}, v{u}, 16, 1"), &[r], &[v(u)])?;
        vop(
            b,
            format!("v_add3_u32 v{r}, v{u}, 0x7fff, v{r}"),
            &[r],
            &[v(u), v(r)],
        )?;
        vop(
            b,
            format!("v_lshrrev_b32_e32 v{r}, 16, v{r}"),
            &[r],
            &[v(r)],
        )?;
        vop(
            b,
            format!("v_lshrrev_b32_e32 v{h16}, 16, v{u}"),
            &[h16],
            &[v(u)],
        )?;
        vop(
            b,
            format!("v_and_b32_e32 v{m}, 0x7f800000, v{u}"),
            &[m],
            &[v(u)],
        )?;
        op(
            b,
            format!("v_cmp_ne_u32_e64 s{}, 0x7f800000, v{m}", e.fin),
            &[s(e.fin)],
            &[v(m)],
        )?;
        vop(
            b,
            format!("v_cndmask_b32_e64 v{r}, v{h16}, v{r}, s{}", e.fin),
            &[r],
            &[v(h16), v(r), s(e.fin)],
        )?;
        res[i] = r;
    }
    vop(
        b,
        format!("v_lshl_or_b32 v{out}, v{}, 16, v{}", res[1], res[0]),
        &[out],
        &[v(res[1]), v(res[0])],
    )?;
    let _ = Instruction::new("", vec![], vec![]);
    Ok(())
}
