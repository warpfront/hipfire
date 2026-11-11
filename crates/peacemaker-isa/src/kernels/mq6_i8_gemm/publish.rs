// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Global fetch and LDS publication of one 64-K slab (activation K32 halves, weight codes
//! decoded once per CTA) and of the next block's token / row metadata.
//!
//! Staging map (the hipcc twin's): lane `tid` owns slab row `R = r * 64 + tid / 4`, quad
//! `c = tid % 4`. Activations: token `R`, 8-byte chunk `c` of each K32 half `sb`
//! (`kg = c >> 1`, `e = c & 1`). Weights: row `R`, K16 step `ks = c` (12 bytes = 16 codes =
//! two 8-code operands `e = 0, 1`; `sb = ks >> 1`, `kg = ks & 1`).
//!
//! LDS fragment pairs: 512 B = 32 lanes x (two 8-byte operands of adjacent 16-token / 16-row
//! groups), lane position `p = (m | kg << 4) ^ (kg << 2)`; the operand of data half `h` sits in
//! 8-byte slot `h ^ x` (`x = e` for activations, `x = sb` for weights) so a wave's 8x4-lane
//! staging store hits every bank pair exactly twice. A K16 step is read as one `ds_load_b128`
//! per pair.
use super::{ds_offset, spec, Gen, MetaD, MetaS, PayA, PayW, Staged, Wv};
use crate::kernels::common::{lit, mem, op, s, sr, v, vr};
use crate::{
    insn::{Instruction, MemoryClass},
    reg::RegRef,
    Builder,
};
use peacemaker_author::State;

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

pub(crate) fn ds_store(text: String, uses: Vec<RegRef>) -> Instruction {
    Instruction::new(text, vec![], uses).memory(MemoryClass::DsStore)
}
pub(crate) fn ds_load(text: String, dst: RegRef, addr: u8) -> Instruction {
    Instruction::new(text, vec![dst], vec![v(addr)]).memory(MemoryClass::DsLoad)
}

/// Raw activation chunks of K32 half `sb` of slab `slab` of the block whose descriptor is
/// `srd`: one b64 per staging round into `a_pf[r]`.
pub(crate) fn fetch_a(b: &mut Builder, g: &Gen, slab: u32, sb: u32, srd: u8) -> Result<(), String> {
    // A clause takes no waits inside: settle the destinations' hazards first (at depth 3 the
    // half-0 stores still read the registers; depth 2 has drained them, nothing is emitted).
    let dsts: Vec<RegRef> = g.a_pf.iter().map(|&r| vr(r, 2)).collect();
    b.settle_defs(&dsts)
        .map_err(|e| format!("fetch_a(slab {slab}, sb {sb}): {e}"))?;
    let round_rows = g.spec.tile.round_rows();
    let mut load = |b: &mut Builder| {
        for r in 0..g.a_pf.len() as u32 {
            vload(
                b,
                g.a_pf[r as usize],
                2,
                g.st_a,
                srd,
                None,
                8 + 64 * slab + 32 * sb + r * round_rows * spec::XBLK_BYTES,
            )?;
        }
        Ok(())
    };
    // A clause needs two loads; a tile with one activation round has one.
    if g.a_pf.len() == 1 {
        load(b)
    } else {
        b.clause(load)
    }
    .map_err(|e| format!("fetch_a(slab {slab}, sb {sb}): {e}"))
}

/// Raw weight dwords (both rounds) at group-relative byte offset `base`
/// (`8 + 96 h + 48 slab`, plus the group stride when fetching across a group boundary).
pub(crate) fn fetch_w(b: &mut Builder, g: &Gen, base: u32) -> Result<(), String> {
    // A clause takes no waits inside: settle the destinations' hazards first.
    let dsts: Vec<RegRef> = g.w_pf.iter().map(|&r| vr(r, 3)).collect();
    b.settle_defs(&dsts)
        .map_err(|e| format!("fetch_w({base}): {e}"))?;
    b.clause(|b| {
        for r in 0..g.w_pf.len() {
            let soff = if r == 0 { g.goff } else { g.goff1 };
            for i in 0..3u32 {
                vload(
                    b,
                    g.w_pf[r] + i as u8,
                    1,
                    g.st_w,
                    g.srd_w,
                    Some(soff),
                    base + 4 * i,
                )?;
            }
        }
        Ok(())
    })
    .map_err(|e| format!("fetch_w({base}): {e}"))
}

/// Next block's token word (`d` on even lanes, `s` on odd lanes) and packed row header.
/// `h`: the parity of the CURRENT block; `hdr`: header byte offset inside the group of the next
/// block (`4 * h_next`, plus the group stride across a group boundary).
pub(crate) fn fetch_meta(b: &mut Builder, g: &Gen, h: usize, hdr: u32) -> Result<(), String> {
    b.clause(|b| {
        vload(b, g.ds_nx, 1, g.ds_voff, g.srd_a[h ^ 1], None, 0)?;
        vload(b, g.sz_nx, 1, g.sz_voff, g.srd_w, Some(g.goff), hdr)
    })
    .map_err(|e| format!("fetch_meta: {e}"))
}

/// Instruction sink: either the builder, or a list for the K-loop's interleaver.
pub(crate) trait Sink {
    fn put(&mut self, text: String, defs: &[u8], uses: &[RegRef]) -> Result<(), String>;
}
impl Sink for Builder {
    fn put(&mut self, text: String, defs: &[u8], uses: &[RegRef]) -> Result<(), String> {
        op(
            self,
            text,
            &defs.iter().map(|&n| v(n)).collect::<Vec<_>>(),
            uses,
        )
    }
}
impl Sink for Vec<Instruction> {
    fn put(&mut self, text: String, defs: &[u8], uses: &[RegRef]) -> Result<(), String> {
        self.push(Instruction::new(
            text,
            defs.iter().map(|&n| v(n)).collect::<Vec<_>>(),
            uses.to_vec(),
        ));
        Ok(())
    }
}
fn vop<S: Sink>(b: &mut S, text: String, defs: &[u8], uses: &[RegRef]) -> Result<(), String> {
    b.put(text, defs, uses)
}

/// `sp4` in place on `f`, `y` scratch: the four 6-bit codes of the 24-bit field `f` spread into the
/// four bytes of the dword (byte i = code i).
fn sp4<S: Sink>(b: &mut S, f: u8, y: u8) -> Result<(), String> {
    vop(
        b,
        format!("v_and_b32_e32 v{y}, 0xfff000, v{f}"),
        &[y],
        &[v(f)],
    )?;
    vop(b, format!("v_and_b32_e32 v{f}, 0xfff, v{f}"), &[f], &[v(f)])?;
    vop(
        b,
        format!("v_lshl_or_b32 v{f}, v{y}, 4, v{f}"),
        &[f],
        &[v(y), v(f)],
    )?;
    vop(
        b,
        format!("v_and_b32_e32 v{y}, 0xfc00fc0, v{f}"),
        &[y],
        &[v(f)],
    )?;
    vop(
        b,
        format!("v_and_b32_e32 v{f}, 0x3f003f, v{f}"),
        &[f],
        &[v(f)],
    )?;
    vop(
        b,
        format!("v_lshl_or_b32 v{f}, v{y}, 2, v{f}"),
        &[f],
        &[v(y), v(f)],
    )
}

/// Decode the weight round `r` raw dwords (16 six-bit codes in `w0 | w1 << 32 | w2 << 64`, code i
/// = bits `[6i, 6i + 6)`) into its two iu8 operands `e0` (codes 0..7) and `e1` (codes 8..15),
/// each `{c0,c2,c4,c6 | c1,c3,c5,c7}` as raw unsigned bytes. `e0` is left in the scratch pair
/// `o`, `e1` in the scratch pair `p`.
pub(crate) fn decode_round<S: Sink>(b: &mut S, g: &Gen, r: usize) -> Result<(), String> {
    let (w0, w1, w2) = (g.w_pf[r], g.w_pf[r] + 1, g.w_pf[r] + 2);
    let (y, p0, p1, o0, o1) = (
        g.scratch,
        g.scratch + 2,
        g.scratch + 3,
        g.scratch + 4,
        g.scratch + 5,
    );
    // f1 = (w0 >> 24) | (w1 << 8), f2 = (w1 >> 16) | (w2 << 16): byte funnels.
    vop(
        b,
        format!("v_perm_b32 v{p0}, v{w1}, v{w0}, {}", lit(0x0605_0403)),
        &[p0],
        &[v(w1), v(w0)],
    )?;
    vop(
        b,
        format!("v_perm_b32 v{p1}, v{w2}, v{w1}, {}", lit(0x0504_0302)),
        &[p1],
        &[v(w2), v(w1)],
    )?;
    // f3 = w2 >> 8
    vop(
        b,
        format!("v_lshrrev_b32_e32 v{w2}, 8, v{w2}"),
        &[w2],
        &[v(w2)],
    )?;
    for f in [w0, p0, p1, w2] {
        sp4(b, f, y)?;
    }
    // b0 = w0, b1 = p0, b2 = p1, b3 = w2. e0 -> (o0, o1); e1 -> (p0, p1) in place.
    for (dst, a, bb, sel) in [
        (o0, p0, w0, 0x0604_0200u32),
        (o1, p0, w0, 0x0705_0301),
        (p0, w2, p1, 0x0604_0200),
        (p1, w2, p1, 0x0705_0301),
    ] {
        vop(
            b,
            format!("v_perm_b32 v{dst}, v{a}, v{bb}, {}", lit(sel)),
            &[dst],
            &[v(a), v(bb)],
        )?;
    }
    Ok(())
}

/// The two LDS stores of the decoded operands of weight round `r` into slot `slot`: `e0` (data
/// in `o`) at `+0`, `e1` (data in `p`) at `+512`.
pub(crate) fn store_w_insns(g: &Gen, slot: usize, r: usize) -> Result<[Instruction; 2], String> {
    if slot >= g.depth() || r >= g.w_pf.len() {
        return Err(format!(
            "store_w: slot {slot} / round {r} out of range ({} / {})",
            g.depth(),
            g.w_pf.len()
        ));
    }
    let tile = g.spec.tile;
    let base = tile.layout().w_slot(slot)? + tile.round_lds_stride() * r as u32;
    let (o, p) = (g.scratch + 4, g.scratch + 2);
    Ok([
        ds_store(
            format!(
                "ds_store_b64 v{}, v[{}:{}]{}",
                g.st_wlds,
                o,
                o + 1,
                ds_offset(base)
            ),
            vec![v(g.st_wlds), vr(o, 2)],
        ),
        ds_store(
            format!(
                "ds_store_b64 v{}, v[{}:{}]{}",
                g.st_wlds,
                p,
                p + 1,
                ds_offset(base + 512)
            ),
            vec![v(g.st_wlds), vr(p, 2)],
        ),
    ])
}

/// The two decoded operands of round `r` to LDS slot `slot`.
pub(crate) fn store_w_round<C: State, const D: usize>(
    w: &mut Wv,
    g: &Gen,
    sw: Staged<PayW, C, D>,
    slot: usize,
    r: usize,
) -> Result<Staged<PayW, C, D>, String> {
    let [e0, e1] = store_w_insns(g, slot, r)?;
    let sw = w.ds_store(sw, e0)?;
    w.ds_store(sw, e1)
}

/// The two LDS stores of the raw activation chunks of K32 half `sb` into slot `slot`.
pub(crate) fn store_a_insns(g: &Gen, slot: usize, sb: u32) -> Result<Vec<Instruction>, String> {
    if slot >= g.depth() || sb > 1 {
        return Err(format!(
            "store_a: slot {slot} / half {sb} out of range (0..2)"
        ));
    }
    let tile = g.spec.tile;
    let a_base = tile.layout().a_slot(slot)?;
    let one = |r: u32| {
        let ar = g.a_pf[r as usize];
        ds_store(
            format!(
                "ds_store_b64 v{}, v[{}:{}]{}",
                g.st_alds,
                ar,
                ar + 1,
                ds_offset(a_base + tile.round_lds_stride() * r + 1024 * sb)
            ),
            vec![v(g.st_alds), vr(ar, 2)],
        )
    };
    Ok((0..g.a_pf.len() as u32).map(one).collect())
}

/// The raw activation chunks of K32 half `sb` to LDS slot `slot`.
pub(crate) fn store_a_half<C: State, const D: usize>(
    w: &mut Wv,
    g: &Gen,
    mut sa: Staged<PayA, C, D>,
    slot: usize,
    sb: u32,
) -> Result<Staged<PayA, C, D>, String> {
    for insn in store_a_insns(g, slot, sb)? {
        sa = w.ds_store(sa, insn)?;
    }
    Ok(sa)
}

/// The VALU part of [`publish_meta`]: `x` (scratch+6) receives the token-plane word, `wr`
/// (scratch+7) the row-plane word.
pub(crate) fn meta_insns<S: Sink>(b: &mut S, g: &Gen) -> Result<(), String> {
    let (x, wr, z) = (g.scratch + 6, g.scratch + 7, g.scratch + 1);
    // Token word: even lanes `d` (raw f32 bits), odd lanes `float(s)`.
    vop(
        b,
        format!("v_cvt_f32_i32_e32 v{wr}, v{}", g.ds_nx),
        &[wr],
        &[v(g.ds_nx)],
    )?;
    vop(
        b,
        format!("v_cndmask_b32_e64 v{x}, v{}, v{wr}, s{}", g.ds_nx, g.omask),
        &[x],
        &[v(g.ds_nx), v(wr), s(g.omask)],
    )?;
    // Row header word: lanes of parity 0 store `sc`, parity 1 store `z' = fma(32, sc, z)`.
    vop(
        b,
        format!("v_cvt_f32_f16_e64 v{wr}, v{}.l", g.sz_nx),
        &[wr],
        &[v(g.sz_nx)],
    )?;
    vop(
        b,
        format!("v_lshrrev_b32_e32 v{z}, 16, v{}", g.sz_nx),
        &[z],
        &[v(g.sz_nx)],
    )?;
    vop(b, format!("v_cvt_f32_f16_e64 v{z}, v{z}.l"), &[z], &[v(z)])?;
    vop(
        b,
        format!("v_fma_f32 v{z}, v{wr}, {}, v{z}", lit(0x4200_0000)),
        &[z],
        &[v(wr), v(z)],
    )?;
    vop(
        b,
        format!("v_cndmask_b32_e64 v{wr}, v{wr}, v{z}, s{}", g.omask),
        &[wr],
        &[v(wr), v(z), s(g.omask)],
    )?;
    Ok(())
}

/// The two LDS stores of [`meta_insns`]' words into plane `plane`: the token word (`x`) into the
/// `DS` plane, the row word (`wr`) into the `SZ` plane.
pub(crate) fn meta_store_insns(g: &Gen, plane: usize) -> Result<[Instruction; 2], String> {
    if plane > 1 {
        return Err(format!("meta plane {plane} out of range (0..2)"));
    }
    let l = g.spec.tile.layout();
    let (x, wr) = (g.scratch + 6, g.scratch + 7);
    Ok([
        ds_store(
            format!(
                "ds_store_b32 v{}, v{x}{}",
                g.meta_lds,
                ds_offset(l.ds[plane])
            ),
            vec![v(g.meta_lds), v(x)],
        ),
        ds_store(
            format!(
                "ds_store_b32 v{}, v{wr}{}",
                g.meta_lds_sz,
                ds_offset(l.sz[plane])
            ),
            vec![v(g.meta_lds_sz), v(wr)],
        ),
    ])
}

/// Publish the next block's `d` / `float(s)` token plane and the `sc` / `z' =
/// z + 32 s` row planes (planar per chunk of 128: `sc` / `d` at `+0`, `z'` / `float(s)` at
/// `+512`; lane `tid` writes slot `tid >> 1` of plane `tid & 1`; on the 512-thread tiles the
/// lanes past the shorter plane's extent store its words a second time, the same values). Rows past `M` and tokens past `N` read 0 through
/// the descriptors.
pub(crate) fn publish_meta<C: State>(
    w: &mut Wv,
    g: &Gen,
    rd: peacemaker_author::Ring<MetaD, C, peacemaker_author::Free>,
    rs: peacemaker_author::Ring<MetaS, C, peacemaker_author::Free>,
) -> Result<(Staged<MetaD, C>, Staged<MetaS, C>), String> {
    let plane = rd.next_index();
    meta_insns(w.isa(), g)?;
    let [td, tz] = meta_store_insns(g, plane)?;
    let (sd, sz) = (w.begin_write(rd), w.begin_write(rs));
    let sd = w.ds_store(sd, td)?;
    let sz = w.ds_store(sz, tz)?;
    Ok((sd, sz))
}
