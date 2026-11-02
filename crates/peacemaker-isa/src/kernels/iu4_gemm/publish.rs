// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Global fetch and LDS publication of one 64-K slab and of the next block's
//! token/row metadata. Staging map (hipcc K1's, bank-swizzled): lane `tid`
//! moves 8 bytes of slab row `R = r * round_rows + tid / 4`, quad `tid % 4`,
//! to fragment-order slot `st_ldsoff + r * round_rows * 32`.
use super::{ds_offset, ds_offsets, FetchPlace, Gen, MetaD, MetaS, PayA, PayW, Staged, Tile, Wv};
use crate::kernels::{
    common::{mem, op, s, sr, v, vr},
    iu4_fold::{GROUP_BYTES, XBLK_BYTES},
};
use crate::{
    insn::{Instruction, MemoryClass},
    Builder,
};
use peacemaker_author::{Free, Ring, State};

/// `buffer_load_b{32,64}` with an optional scalar offset (objdump spelling).
pub(crate) fn vload(
    b: &mut Builder,
    dst: u8,
    width: u8,
    voff: u8,
    srd: u8,
    soff: Option<u8>,
    offset: u32,
) -> Result<(), String> {
    let name = if width == 2 {
        "buffer_load_b64"
    } else {
        "buffer_load_b32"
    };
    let d = if width == 2 {
        format!("v[{}:{}]", dst, dst + 1)
    } else {
        format!("v{dst}")
    };
    let so = soff.map_or("null".to_string(), |r| format!("s{r}"));
    let off = if offset == 0 {
        String::new()
    } else {
        format!(" offset:{offset}")
    };
    let mut uses = vec![v(voff), sr(srd, 4)];
    if let Some(r) = soff {
        uses.push(s(r))
    }
    mem(
        b,
        format!(
            "{name} {d}, v{voff}, s[{}:{}], {so} offen{off}",
            srd,
            srd + 3
        ),
        &[vr(dst, width)],
        &uses,
        MemoryClass::VmemLoad,
    )
}

fn ds_store(text: String, uses: Vec<crate::reg::RegRef>) -> Instruction {
    Instruction::new(text, vec![], uses).memory(MemoryClass::DsStore)
}
pub(crate) fn ds_load(text: String, dst: crate::reg::RegRef, addr: u8) -> Instruction {
    Instruction::new(text, vec![dst], vec![v(addr)]).memory(MemoryClass::DsLoad)
}

/// Scalar offset and immediate of staging round `r` of activation slab
/// `slab` (0: K bytes 0..31, 1: K bytes 32..63) relative to `st_a`.
pub(crate) fn a_offset(g: &Gen, slab: u32, r: usize) -> (Option<u8>, u32) {
    let rows = r as u32 * g.tile.round_rows();
    match g.a_slab1 {
        None => (None, 8 + 32 * slab + rows * XBLK_BYTES),
        Some(s1) => ((slab == 1).then_some(s1), rows * 32),
    }
}

/// Fetch slab 1 of block `h` (K bytes 32..63), using the group-relative W offset.
pub(crate) fn fetch_slab1(b: &mut Builder, g: &Gen, h: usize) -> Result<(), String> {
    fetch_slab1_at(b, g, h, h as u32 * 64 + 40)
}

/// Block 0's slab 1 (the prologue's seed of the K-loop's fetch chain). Under
/// `FetchPlace::PairW` the weights go to `W_x` ahead of the activation slab, the order the
/// loop's own blocks leave them in (`fetch_next`, then `fetch_slab1`), so the loop's
/// entry and back-edge load ledgers agree.
pub(crate) fn fetch_slab1_prologue(b: &mut Builder, g: &Gen) -> Result<(), String> {
    if g.spec.tune.fetch == FetchPlace::PairW {
        b.clause(|b| {
            for (r, &dst) in g.w_x.iter().enumerate() {
                vload(b, dst, 2, g.st_w[r], g.srd_w, Some(g.goff), 8 + 32)?;
            }
            Ok(())
        })?;
    }
    fetch_slab1(b, g, 0)
}

/// Fetch the next group's even block before `goff` advances at trip end.
pub(crate) fn fetch_slab1_next_group(b: &mut Builder, g: &Gen) -> Result<(), String> {
    fetch_slab1_at(b, g, 0, GROUP_BYTES + 40)
}

fn fetch_slab1_at(b: &mut Builder, g: &Gen, h: usize, weight_offset: u32) -> Result<(), String> {
    b.clause(|b| {
        for (r, &dst) in g.a_pf.iter().enumerate() {
            let (soff, offset) = a_offset(g, 1, r);
            vload(b, dst, 2, g.st_a, g.srd_a[h], soff, offset)?;
        }
        // `FetchPlace::PairW`: the block's slab-1 weights were fetched with its slab 0
        // (`fetch_next` / the prologue); only the activation slab is fetched here.
        if g.spec.tune.fetch != FetchPlace::PairW {
            for (r, &dst) in g.w_pf.iter().enumerate() {
                vload(b, dst, 2, g.st_w[r], g.srd_w, Some(g.goff), weight_offset)?;
            }
        }
        Ok(())
    })
}

/// The prefetch registers of set `set` (A rounds, W rounds): set 0 is the
/// only set at ring depth 2; depth 3 adds set 1 (a block's slab 1).
pub(crate) fn pf_set(g: &Gen, set: usize) -> Result<(&[u8], &[u8]), String> {
    match set {
        0 => Ok((&g.a_pf, &g.w_pf)),
        1 if !g.a_pf1.is_empty() => Ok((&g.a_pf1, &g.w_pf1)),
        1 => Err("prefetch set 1 exists at ring depth 3 only".into()),
        _ => Err(format!("prefetch set {set} is not 0 or 1")),
    }
}

/// Ring depth 3: fetch slab `slab` (0 or 1) of the block one group ahead on
/// plane `h` (the block after next) into prefetch set `slab`. The even/odd
/// block's activation descriptor has already moved to that group (the depth-3
/// K-loop steps it at the head of the block); the weight offset is the
/// group-relative one plus the group stride.
pub(crate) fn fetch_ahead(b: &mut Builder, g: &Gen, h: usize, slab: u32) -> Result<(), String> {
    if h > 1 {
        return Err(format!("fetch_ahead: plane {h} is not 0 or 1"));
    }
    fetch_slab_at(
        b,
        g,
        slab as usize,
        slab,
        h,
        GROUP_BYTES + h as u32 * 64 + 8 + 32 * slab,
    )
}

/// Fetch slab `slab` of the block whose activations descriptor is
/// `srd_a[desc]` and whose weight offset (relative to `goff`) is
/// `weight_offset`, into prefetch set `set`.
pub(crate) fn fetch_slab_at(
    b: &mut Builder,
    g: &Gen,
    set: usize,
    slab: u32,
    desc: usize,
    weight_offset: u32,
) -> Result<(), String> {
    if slab > 1 || desc > 1 {
        return Err(format!(
            "fetch_slab_at: slab {slab} / descriptor {desc} is not 0 or 1"
        ));
    }
    let (a_pf, w_pf) = pf_set(g, set)?;
    b.clause(|b| {
        for (r, &dst) in a_pf.iter().enumerate() {
            let (soff, offset) = a_offset(g, slab, r);
            vload(b, dst, 2, g.st_a, g.srd_a[desc], soff, offset)?;
        }
        for (r, &dst) in w_pf.iter().enumerate() {
            vload(b, dst, 2, g.st_w[r], g.srd_w, Some(g.goff), weight_offset)?;
        }
        Ok(())
    })
}

/// Issue next-block token scale and packed row header before B1: neither
/// payload register is still live after the preceding block's B2 publication.
pub(crate) fn fetch_next_meta(b: &mut Builder, g: &Gen, h: usize) -> Result<(), String> {
    let header = if h == 0 { 4 } else { GROUP_BYTES };
    b.clause(|b| {
        vload(b, g.ds_nx, 1, g.ds_voff, g.srd_a[h ^ 1], None, 0)?;
        vload(b, g.sz_nx, 1, g.sz_voff, g.srd_z, Some(g.goff), header)
    })
}

/// Next-block slab-0 fetch epoch after B1 of block `h`.
pub(crate) fn fetch_next(b: &mut Builder, g: &Gen, h: usize) -> Result<(), String> {
    // Next (group, half) relative to the current group offset in `goff`.
    let next = if h == 0 { 64 } else { GROUP_BYTES };
    b.clause(|b| {
        for (r, &dst) in g.a_pf.iter().enumerate() {
            let (soff, offset) = a_offset(g, 0, r);
            vload(b, dst, 2, g.st_a, g.srd_a[h ^ 1], soff, offset)?;
        }
        for (r, &dst) in g.w_pf.iter().enumerate() {
            vload(b, dst, 2, g.st_w[r], g.srd_w, Some(g.goff), next + 8)?;
        }
        // `FetchPlace::PairW`: slab 1's weights go out back to back with slab 0's.
        for (r, &dst) in g.w_x.iter().enumerate() {
            vload(b, dst, 2, g.st_w[r], g.srd_w, Some(g.goff), next + 8 + 32)?;
        }
        Ok(())
    })
}

/// Flip each uint4 code's sign bit so the signed WMMA consumes `q - 8`, on
/// prefetch set `set`.
pub(crate) fn flip_weights_set(b: &mut Builder, g: &Gen, set: usize) -> Result<(), String> {
    let (_, w_pf) = pf_set(g, set)?;
    flip_weights(b, w_pf)
}

fn flip_weights(b: &mut Builder, w_pf: &[u8]) -> Result<(), String> {
    for &w in w_pf {
        for r in [w, w + 1] {
            op(
                b,
                format!("v_xor_b32_e32 v{r}, 0x88888888, v{r}"),
                &[v(r)],
                &[v(r)],
            )?;
        }
    }
    Ok(())
}

/// Publish the staged slab in `A_pf/W_pf` to the payload rings' next slot.
pub(crate) fn publish_slab<C: State, const D: usize>(
    w: &mut Wv,
    g: &Gen,
    ra: Ring<PayA, C, Free, D>,
    rw: Ring<PayW, C, Free, D>,
) -> Result<(Staged<PayA, C, D>, Staged<PayW, C, D>), String> {
    publish_slab_set(w, g, 0, ra, rw)
}

/// [`publish_slab`] of prefetch set `set` (ring depth 3 publishes a block's
/// slab 0 from set 0 and its slab 1 from set 1).
pub(crate) fn publish_slab_set<C: State, const D: usize>(
    w: &mut Wv,
    g: &Gen,
    set: usize,
    ra: Ring<PayA, C, Free, D>,
    rw: Ring<PayW, C, Free, D>,
) -> Result<(Staged<PayA, C, D>, Staged<PayW, C, D>), String> {
    publish_slab_regs(w, g, set, false, ra, rw)
}

/// Publish a block's slab 1 (ring depth 2): the slab staged in `A_pf` and, under
/// `FetchPlace::PairW`, the weights its slab 0 fetch brought along in `W_x`.
pub(crate) fn publish_slab1<C: State, const D: usize>(
    w: &mut Wv,
    g: &Gen,
    ra: Ring<PayA, C, Free, D>,
    rw: Ring<PayW, C, Free, D>,
) -> Result<(Staged<PayA, C, D>, Staged<PayW, C, D>), String> {
    publish_slab_regs(w, g, 0, g.spec.tune.fetch == FetchPlace::PairW, ra, rw)
}

fn publish_slab_regs<C: State, const D: usize>(
    w: &mut Wv,
    g: &Gen,
    set: usize,
    w_from_x: bool,
    ra: Ring<PayA, C, Free, D>,
    rw: Ring<PayW, C, Free, D>,
) -> Result<(Staged<PayA, C, D>, Staged<PayW, C, D>), String> {
    let slot = ra.next_index();
    let (a_pf, w_pf) = pf_set(g, set)?;
    let w_pf: &[u8] = if w_from_x { &g.w_x } else { w_pf };
    flip_weights(w.isa(), w_pf)?;
    let l = g.layout;
    let step = g.tile.round_rows() / 16; // 512-byte units per staging round
    let (a, wo) = (l.a[slot] / 512, l.w[slot] / 512);
    let (mut sa, mut sw) = (w.begin_write(ra), w.begin_write(rw));
    match g.tile {
        Tile::T128x128x8 => {
            for r in 0..a_pf.len() {
                let (ar, wr) = (a_pf[r], w_pf[r]);
                (sa, sw) = w.ds_store2(
                    sa,
                    sw,
                    ds_store(
                        format!(
                            "ds_store_2addr_stride64_b64 v{}, v[{}:{}], v[{}:{}]{}",
                            g.st_lds,
                            ar,
                            ar + 1,
                            wr,
                            wr + 1,
                            ds_offsets(a + step * r as u32, wo + step * r as u32)
                        ),
                        vec![v(g.st_lds), vr(ar, 2), vr(wr, 2)],
                    ),
                )?;
            }
        }
        Tile::T256x128x16 => {
            let ar = a_pf[0];
            sa = w.ds_store(
                sa,
                ds_store(
                    format!(
                        "ds_store_b64 v{}, v[{}:{}]{}",
                        g.st_lds,
                        ar,
                        ar + 1,
                        ds_offset(l.a[slot])
                    ),
                    vec![v(g.st_lds), vr(ar, 2)],
                ),
            )?;
            let (w0, w1) = (w_pf[0], w_pf[1]);
            sw = w.ds_store(
                sw,
                ds_store(
                    format!(
                        "ds_store_2addr_stride64_b64 v{}, v[{}:{}], v[{}:{}]{}",
                        g.st_lds,
                        w0,
                        w0 + 1,
                        w1,
                        w1 + 1,
                        ds_offsets(wo, wo + step)
                    ),
                    vec![v(g.st_lds), vr(w0, 2), vr(w1, 2)],
                ),
            )?;
        }
    }
    Ok((sa, sw))
}

/// Publish the next block's `d` (token plane) and converted `sc` (row plane).
/// Rows past M and tokens past N carry harmless values: they only reach
/// outputs the epilogue never stores (tokens read 0 through the descriptor).
pub(crate) fn publish_meta<C: State>(
    w: &mut Wv,
    g: &Gen,
    rd: Ring<MetaD, C, Free>,
    rs: Ring<MetaS, C, Free>,
) -> Result<(Staged<MetaD, C>, Staged<MetaS, C>), String> {
    let plane = rd.next_index();
    op(
        w.isa(),
        format!("v_cvt_f32_f16_e64 v{0}, v{0}.l", g.sz_nx),
        &[v(g.sz_nx)],
        &[v(g.sz_nx)],
    )?;
    let l = g.layout;
    let (sd, sz) = (w.begin_write(rd), w.begin_write(rs));
    if g.meta_ds == g.meta_sz {
        w.ds_store2(
            sd,
            sz,
            ds_store(
                format!(
                    "ds_store_2addr_stride64_b32 v{}, v{}, v{}{}",
                    g.meta_ds,
                    g.ds_nx,
                    g.sz_nx,
                    ds_offsets(l.ds[plane] / 256, l.sz[plane] / 256)
                ),
                vec![v(g.meta_ds), v(g.ds_nx), v(g.sz_nx)],
            ),
        )
    } else {
        let sd = w.ds_store(
            sd,
            ds_store(
                format!(
                    "ds_store_b32 v{}, v{}{}",
                    g.meta_ds,
                    g.ds_nx,
                    ds_offset(l.ds[plane])
                ),
                vec![v(g.meta_ds), v(g.ds_nx)],
            ),
        )?;
        let sz = w.ds_store(
            sz,
            ds_store(
                format!(
                    "ds_store_b32 v{}, v{}{}",
                    g.meta_sz,
                    g.sz_nx,
                    ds_offset(l.sz[plane])
                ),
                vec![v(g.meta_sz), v(g.sz_nx)],
            ),
        )?;
        Ok((sd, sz))
    }
}
