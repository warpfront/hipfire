//! Global fetch and LDS publication of one 64-K slab and of the next block's
//! token/row metadata. Staging map (hipcc K1's, bank-swizzled): lane `tid`
//! moves 8 bytes of slab row `R = r * round_rows + tid / 4`, quad `tid % 4`,
//! to fragment-order slot `st_ldsoff + r * round_rows * 32`.
use super::{Gen, MetaD, MetaS, PayA, PayW, Staged, Tile, Wv, ds_offset, ds_offsets};
use crate::kernels::{common::{mem, op, s, sr, v, vr}, iu4_fold::{GROUP_BYTES, XBLK_BYTES}};
use crate::{Builder, insn::{Instruction, MemoryClass}};
use peacemaker_author::{Free, Ring, State};

/// `buffer_load_b{32,64}` with an optional scalar offset (objdump spelling).
pub(crate) fn vload(b: &mut Builder, dst: u8, width: u8, voff: u8, srd: u8, soff: Option<u8>, offset: u32) -> Result<(), String> {
    let name = if width == 2 { "buffer_load_b64" } else { "buffer_load_b32" };
    let d = if width == 2 { format!("v[{}:{}]", dst, dst + 1) } else { format!("v{dst}") };
    let so = soff.map_or("null".to_string(), |r| format!("s{r}"));
    let off = if offset == 0 { String::new() } else { format!(" offset:{offset}") };
    let mut uses = vec![v(voff), sr(srd, 4)];
    if let Some(r) = soff { uses.push(s(r)) }
    mem(b, format!("{name} {d}, v{voff}, s[{}:{}], {so} offen{off}", srd, srd + 3), &[vr(dst, width)], &uses, MemoryClass::VmemLoad)
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
        for (r, &dst) in g.w_pf.iter().enumerate() {
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
        Ok(())
    })
}

/// Flip each uint4 code's sign bit so the signed WMMA consumes `q - 8`.
pub(crate) fn flip_weights(b: &mut Builder, g: &Gen) -> Result<(), String> {
    for &w in &g.w_pf {
        for r in [w, w + 1] { op(b, format!("v_xor_b32_e32 v{r}, 0x88888888, v{r}"), &[v(r)], &[v(r)])?; }
    }
    Ok(())
}

/// Publish the staged slab in `A_pf/W_pf` to the payload rings' next slot.
pub(crate) fn publish_slab<C: State>(w: &mut Wv, g: &Gen, ra: Ring<PayA, C, Free>, rw: Ring<PayW, C, Free>) -> Result<(Staged<PayA, C>, Staged<PayW, C>), String> {
    let slot = ra.next_index();
    flip_weights(w.isa(), g)?;
    let l = g.layout;
    let step = g.tile.round_rows() / 16; // 512-byte units per staging round
    let (a, wo) = (l.a[slot] / 512, l.w[slot] / 512);
    let (mut sa, mut sw) = (w.begin_write(ra), w.begin_write(rw));
    match g.tile {
        Tile::T128x128x8 => for r in 0..g.a_pf.len() {
            let (ar, wr) = (g.a_pf[r], g.w_pf[r]);
            (sa, sw) = w.ds_store2(sa, sw, ds_store(format!("ds_store_2addr_stride64_b64 v{}, v[{}:{}], v[{}:{}]{}", g.st_lds, ar, ar + 1, wr, wr + 1,
                ds_offsets(a + step * r as u32, wo + step * r as u32)), vec![v(g.st_lds), vr(ar, 2), vr(wr, 2)]))?;
        },
        Tile::T256x128x16 => {
            let ar = g.a_pf[0];
            sa = w.ds_store(sa, ds_store(format!("ds_store_b64 v{}, v[{}:{}]{}", g.st_lds, ar, ar + 1, ds_offset(l.a[slot])), vec![v(g.st_lds), vr(ar, 2)]))?;
            let (w0, w1) = (g.w_pf[0], g.w_pf[1]);
            sw = w.ds_store(sw, ds_store(format!("ds_store_2addr_stride64_b64 v{}, v[{}:{}], v[{}:{}]{}", g.st_lds, w0, w0 + 1, w1, w1 + 1,
                ds_offsets(wo, wo + step)), vec![v(g.st_lds), vr(w0, 2), vr(w1, 2)]))?;
        }
    }
    Ok((sa, sw))
}

/// Publish the next block's `d` (token plane) and converted `sc` (row plane).
/// Rows past M and tokens past N carry harmless values: they only reach
/// outputs the epilogue never stores (tokens read 0 through the descriptor).
pub(crate) fn publish_meta<C: State>(w: &mut Wv, g: &Gen, rd: Ring<MetaD, C, Free>, rs: Ring<MetaS, C, Free>) -> Result<(Staged<MetaD, C>, Staged<MetaS, C>), String> {
    let plane = rd.next_index();
    op(w.isa(), format!("v_cvt_f32_f16_e64 v{0}, v{0}.l", g.sz_nx), &[v(g.sz_nx)], &[v(g.sz_nx)])?;
    let l = g.layout;
    let (sd, sz) = (w.begin_write(rd), w.begin_write(rs));
    if g.meta_ds == g.meta_sz {
        w.ds_store2(sd, sz, ds_store(format!("ds_store_2addr_stride64_b32 v{}, v{}, v{}{}", g.meta_ds, g.ds_nx, g.sz_nx,
            ds_offsets(l.ds[plane] / 256, l.sz[plane] / 256)), vec![v(g.meta_ds), v(g.ds_nx), v(g.sz_nx)]))
    } else {
        let sd = w.ds_store(sd, ds_store(format!("ds_store_b32 v{}, v{}{}", g.meta_ds, g.ds_nx, ds_offset(l.ds[plane])), vec![v(g.meta_ds), v(g.ds_nx)]))?;
        let sz = w.ds_store(sz, ds_store(format!("ds_store_b32 v{}, v{}{}", g.meta_sz, g.sz_nx, ds_offset(l.sz[plane])), vec![v(g.meta_sz), v(g.sz_nx)]))?;
        Ok((sd, sz))
    }
}

