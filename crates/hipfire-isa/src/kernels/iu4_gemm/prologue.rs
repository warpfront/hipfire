//! Prologue: kernel arguments, banded raster (hipcc `IU4_G12_RASTER`, band 8),
//! buffer descriptors, hoisted lane offsets, block-0 staging and the first
//! rendezvous. Workgroup ids on gfx1201 are `ttmp9` (x) and `ttmp7[15:0]` (y).
use super::{END, Epi, Gen, Tile, lit, mem, op, publish, s, sr, v};
use crate::{Builder, insn::MemoryClass, lds::Transition};

/// SALU unsigned 32-bit division, LLVM's AMDGPU expansion: a float
/// reciprocal refined once, then two quotient corrections. Exact for all
/// operands. `x`, `y` are scratch.
fn udiv(b: &mut Builder, num: u8, den: u8, q: u8, r: u8, x: u8, y: u8) -> Result<(), String> {
    let o = |b: &mut Builder, t: String, d: &[u8], u: &[u8]| op(b, t, &d.iter().map(|&n| s(n)).collect::<Vec<_>>(), &u.iter().map(|&n| s(n)).collect::<Vec<_>>());
    o(b, format!("s_cvt_f32_u32 s{x}, s{den}"), &[x], &[den])?;
    o(b, format!("v_s_rcp_f32 s{x}, s{x}"), &[x], &[x])?;
    o(b, format!("s_mul_f32 s{x}, s{x}, 0x4f7ffffe"), &[x], &[x])?;
    o(b, format!("s_cvt_u32_f32 s{x}, s{x}"), &[x], &[x])?;
    o(b, format!("s_sub_co_i32 s{y}, 0, s{den}"), &[y], &[den])?;
    o(b, format!("s_mul_i32 s{y}, s{y}, s{x}"), &[y], &[y, x])?;
    o(b, format!("s_mul_hi_u32 s{y}, s{x}, s{y}"), &[y], &[x, y])?;
    o(b, format!("s_add_co_i32 s{x}, s{x}, s{y}"), &[x], &[x, y])?;
    o(b, format!("s_mul_hi_u32 s{q}, s{num}, s{x}"), &[q], &[num, x])?;
    o(b, format!("s_mul_i32 s{y}, s{q}, s{den}"), &[y], &[q, den])?;
    o(b, format!("s_sub_co_i32 s{r}, s{num}, s{y}"), &[r], &[num, y])?;
    for _ in 0..2 {
        o(b, format!("s_add_co_i32 s{x}, s{q}, 1"), &[x], &[q])?;
        o(b, format!("s_sub_co_i32 s{y}, s{r}, s{den}"), &[y], &[r, den])?;
        o(b, format!("s_cmp_ge_u32 s{r}, s{den}"), &[], &[r, den])?;
        o(b, format!("s_cselect_b32 s{q}, s{x}, s{q}"), &[q], &[x, q])?;
        o(b, format!("s_cselect_b32 s{r}, s{y}, s{r}"), &[r], &[y, r])?;
    }
    Ok(())
}

fn sop(b: &mut Builder, text: String, d: &[u8], u: &[u8]) -> Result<(), String> {
    op(b, text, &d.iter().map(|&n| s(n)).collect::<Vec<_>>(), &u.iter().map(|&n| s(n)).collect::<Vec<_>>())
}
/// VALU helper: `vdst` defined, `vuse`/`suse` read.
fn vop(b: &mut Builder, text: String, vdst: u8, vuse: &[u8], suse: &[u8]) -> Result<(), String> {
    let uses = vuse.iter().map(|&n| v(n)).chain(suse.iter().map(|&n| s(n))).collect::<Vec<_>>();
    op(b, text, &[v(vdst)], &uses)
}

pub(crate) fn emit(b: &mut Builder, g: &Gen) -> Result<(), String> {
    let a = g.args;
    let silu = g.spec.epi == Epi::GateUpSilu;
    let tile = g.tile;
    let t: [u8; 8] = std::array::from_fn(|i| g.tmp + i as u8);
    mem(b, "s_load_b256 s[8:15], s[0:1], 0x0", &[sr(8, 8)], &[sr(0, 2)], MemoryClass::SmemLoad)?;
    mem(b, "s_load_b256 s[16:23], s[0:1], 0x20", &[sr(16, 8)], &[sr(0, 2)], MemoryClass::SmemLoad)?;

    // Banded raster: lin = y * gridDim.x + x; bands of 8 row tiles, rows
    // fastest within a band, then token tiles (bijective, partial last band).
    sop(b, format!("s_and_b32 s{}, ttmp7, 0xffff", t[0]), &[t[0]], &[])?;
    sop(b, format!("s_mul_i32 s{}, s{}, s{}", t[1], t[0], a.bcx), &[t[1]], &[t[0], a.bcx])?;
    sop(b, format!("s_add_co_i32 s{0}, s{0}, ttmp9", t[1]), &[t[1]], &[t[1]])?;
    sop(b, format!("s_lshl_b32 s{}, s{}, 3", t[2], a.bcy), &[t[2]], &[a.bcy])?;
    udiv(b, t[1], t[2], t[3], t[4], t[5], t[6])?; // band, in_band
    sop(b, format!("s_lshl_b32 s{}, s{}, 3", t[5], t[3]), &[t[5]], &[t[3]])?; // band_row0
    sop(b, format!("s_sub_co_i32 s{}, s{}, s{}", t[6], a.bcx, t[5]), &[t[6]], &[a.bcx, t[5]])?;
    sop(b, format!("s_min_u32 s{0}, s{0}, 8", t[6]), &[t[6]], &[t[6]])?; // band_rows
    udiv(b, t[4], t[6], t[7], t[0], t[1], t[2])?; // token tile, row within band
    sop(b, format!("s_add_co_i32 s{0}, s{1}, s{0}", t[0], t[5]), &[t[0]], &[t[0], t[5]])?;
    sop(b, format!("s_lshl_b32 s{}, s{}, {}", g.rs, t[0], tile.rows().trailing_zeros()), &[g.rs], &[t[0]])?;
    sop(b, format!("s_lshl_b32 s{}, s{}, 7", g.bs, t[7]), &[g.bs], &[t[7]])?;
    // Tiles past the problem exit (none for the host grid; kept for safety).
    sop(b, format!("s_lshl_b32 s54, s{}, {}", a.m, u32::from(silu)), &[54], &[a.m])?;
    sop(b, format!("s_cmp_ge_u32 s{}, s54", g.rs), &[], &[g.rs, 54])?;
    op(b, format!("s_cbranch_scc1 {END}"), &[], &[])?;
    sop(b, format!("s_cmp_ge_u32 s{}, s{}", g.bs, a.n), &[], &[g.bs, a.n])?;
    op(b, format!("s_cbranch_scc1 {END}"), &[], &[])?;

    // Scalars: M-1 (row clamp), 136 * K/256 (row stride), loop trips, h-rows.
    sop(b, format!("s_add_co_i32 s{}, s{}, -1", g.mm1, a.m), &[g.mm1], &[a.m])?;
    sop(b, format!("s_lshr_b32 s{}, s{}, 8", t[0], a.k), &[t[0]], &[a.k])?;
    sop(b, format!("s_mul_i32 s{}, s{}, 0x88", g.gpr136, t[0]), &[g.gpr136], &[t[0]])?;
    sop(b, format!("s_add_co_i32 s{}, s{}, -1", g.trips, t[0]), &[g.trips], &[t[0]])?;
    sop(b, format!("s_mov_b32 s{}, 0", g.goff), &[g.goff], &[])?;
    if silu { sop(b, format!("s_lshr_b32 s{}, s{}, 1", g.hs, g.rs), &[g.hs], &[g.rs])?; }
    op(b, format!("v_readfirstlane_b32 s{}, v0", g.wave), &[s(g.wave)], &[v(0)])?;
    sop(b, format!("s_lshr_b32 s{0}, s{0}, 5", g.wave), &[g.wave], &[g.wave])?;

    // Descriptors: Xq per 128-K block (N*72 records, OOB tokens read 0),
    // weights/headers unbounded (rows are clamped), word3 0x31004000.
    let [a0, a1] = g.srd_a;
    sop(b, format!("s_mov_b32 s{}, s{}", a0, a.xq), &[a0], &[a.xq])?;
    sop(b, format!("s_and_b32 s{}, s{}, 0xffff", a0 + 1, a.xq + 1), &[a0 + 1], &[a.xq + 1])?;
    sop(b, format!("s_mul_i32 s{}, s{}, 0x48", a0 + 2, a.n), &[a0 + 2], &[a.n])?;
    sop(b, format!("s_mov_b32 s{}, 0x31004000", a0 + 3), &[a0 + 3], &[])?;
    sop(b, format!("s_add_co_u32 s{}, s{}, s{}", a1, a0, a0 + 2), &[a1], &[a0, a0 + 2])?;
    sop(b, format!("s_add_co_ci_u32 s{}, s{}, 0", a1 + 1, a0 + 1), &[a1 + 1], &[a0 + 1])?;
    sop(b, format!("s_mov_b32 s{}, s{}", a1 + 2, a0 + 2), &[a1 + 2], &[a0 + 2])?;
    sop(b, format!("s_mov_b32 s{}, 0x31004000", a1 + 3), &[a1 + 3], &[])?;
    sop(b, format!("s_lshl_b32 s{}, s{}, 1", g.step2, a0 + 2), &[g.step2], &[a0 + 2])?;
    sop(b, format!("s_mov_b32 s{}, 0", g.step2 + 1), &[g.step2 + 1], &[])?;
    let weight_srd = |b: &mut Builder, srd: u8, bit: u32| -> Result<(), String> {
        match a.u {
            Some(u) => {
                sop(b, format!("s_bitcmp1_b32 s{}, {bit}", g.wave), &[], &[g.wave])?;
                op(b, format!("s_cselect_b64 s[{}:{}], s[{}:{}], s[{}:{}]", srd, srd + 1, u, u + 1, a.a, a.a + 1), &[sr(srd, 2)], &[sr(u, 2), sr(a.a, 2)])?;
                sop(b, format!("s_and_b32 s{0}, s{0}, 0xffff", srd + 1), &[srd + 1], &[srd + 1])?;
            }
            None => {
                sop(b, format!("s_mov_b32 s{}, s{}", srd, a.a), &[srd], &[a.a])?;
                sop(b, format!("s_and_b32 s{}, s{}, 0xffff", srd + 1, a.a + 1), &[srd + 1], &[a.a + 1])?;
            }
        }
        sop(b, format!("s_mov_b32 s{}, -1", srd + 2), &[srd + 2], &[])?;
        sop(b, format!("s_mov_b32 s{}, 0x31004000", srd + 3), &[srd + 3], &[])
    };
    // Gate/up: W staging picks gate or up by tid bit 6, the header row by bit 5.
    weight_srd(b, g.srd_w, 1)?;
    if g.srd_z != g.srd_w { weight_srd(b, g.srd_z, 0)?; }

    // Hoisted lane offsets (v0 = tid). Temporaries live in v1.. until K_BEGIN.
    let (ml, half, even, odd, r0, q) = (1u8, 2u8, 3u8, 4u8, 5u8, 6u8);
    vop(b, format!("v_and_b32_e32 v{ml}, 15, v0"), ml, &[0], &[])?;
    vop(b, format!("v_and_b32_e32 v{half}, 16, v0"), half, &[0], &[])?;
    vop(b, format!("v_lshrrev_b32_e32 v{even}, 2, v{half}"), even, &[half], &[])?;
    vop(b, format!("v_xor_b32_e32 v{even}, v{even}, v{ml}"), even, &[even, ml], &[])?;
    vop(b, format!("v_or_b32_e32 v{even}, v{even}, v{half}"), even, &[even, half], &[])?;
    vop(b, format!("v_lshlrev_b32_e32 v{even}, 3, v{even}"), even, &[even], &[])?;
    vop(b, format!("v_xor_b32_e32 v{odd}, 64, v{even}"), odd, &[even], &[])?;
    // Fragment bases: A frag (tok_half*4+nb)*2+sb, W frag (wt_pair*2+rg)*2+sb.
    let (th, wp, x) = (t[0], t[1], t[2]);
    sop(b, format!("s_and_b32 s{th}, s{}, 1", g.wave), &[th], &[g.wave])?;
    sop(b, format!("s_lshr_b32 s{wp}, s{}, 1", g.wave), &[wp], &[g.wave])?;
    for (base, shift, dst) in [(th, 11u32, g.fr_a), (wp, 10, g.fr_w)] {
        sop(b, format!("s_lshl_b32 s{x}, s{base}, {shift}"), &[x], &[base])?;
        vop(b, format!("v_add_nc_u32_e32 v{}, s{x}, v{even}", dst[0]), dst[0], &[even], &[x])?;
        sop(b, format!("s_addk_co_i32 s{x}, 0x100"), &[x], &[x])?;
        vop(b, format!("v_add_nc_u32_e32 v{}, s{x}, v{odd}", dst[1]), dst[1], &[odd], &[x])?;
    }
    // Staging slot: frag (R>>4)*2+(q>>1), lane (R&15)+16*(q&1), XOR-swizzled.
    let (fr, q2, lo, qa, qb) = (7u8, 8u8, 9u8, 10u8, 11u8);
    vop(b, format!("v_lshrrev_b32_e32 v{r0}, 2, v0"), r0, &[0], &[])?;
    vop(b, format!("v_and_b32_e32 v{q}, 3, v0"), q, &[0], &[])?;
    vop(b, format!("v_lshrrev_b32_e32 v{fr}, 4, v{r0}"), fr, &[r0], &[])?;
    vop(b, format!("v_lshlrev_b32_e32 v{fr}, 9, v{fr}"), fr, &[fr], &[])?;
    vop(b, format!("v_and_b32_e32 v{q2}, 2, v{q}"), q2, &[q], &[])?;
    vop(b, format!("v_lshlrev_b32_e32 v{qa}, 7, v{q2}"), qa, &[q2], &[])?;
    vop(b, format!("v_add_nc_u32_e32 v{fr}, v{fr}, v{qa}"), fr, &[fr, qa], &[])?;
    vop(b, format!("v_and_b32_e32 v{lo}, 15, v{r0}"), lo, &[r0], &[])?;
    vop(b, format!("v_lshlrev_b32_e32 v{qa}, 2, v{q2}"), qa, &[q2], &[])?;
    vop(b, format!("v_xor_b32_e32 v{lo}, v{lo}, v{qa}"), lo, &[lo, qa], &[])?;
    vop(b, format!("v_and_b32_e32 v{qb}, 1, v{q}"), qb, &[q], &[])?;
    vop(b, format!("v_lshlrev_b32_e32 v{qa}, 2, v{qb}"), qa, &[qb], &[])?;
    vop(b, format!("v_xor_b32_e32 v{lo}, v{lo}, v{qa}"), lo, &[lo, qa], &[])?;
    vop(b, format!("v_lshlrev_b32_e32 v{qb}, 4, v{qb}"), qb, &[qb], &[])?;
    vop(b, format!("v_or_b32_e32 v{lo}, v{lo}, v{qb}"), lo, &[lo, qb], &[])?;
    vop(b, format!("v_lshlrev_b32_e32 v{lo}, 3, v{lo}"), lo, &[lo], &[])?;
    vop(b, format!("v_add_nc_u32_e32 v{}, v{fr}, v{lo}", g.st_lds), g.st_lds, &[fr, lo], &[])?;
    // Global slab offsets: token (bs + R)*72 + q*8; weight row clamped to M-1.
    let (q8, row, hr) = (12u8, 13u8, 14u8);
    vop(b, format!("v_lshlrev_b32_e32 v{q8}, 3, v{q}"), q8, &[q], &[])?;
    vop(b, format!("v_add_nc_u32_e32 v{row}, s{}, v{r0}", g.bs), row, &[r0], &[g.bs])?;
    vop(b, format!("v_mul_u32_u24_e32 v{row}, 0x48, v{row}"), row, &[row], &[])?;
    vop(b, format!("v_add_nc_u32_e32 v{}, v{row}, v{q8}", g.st_a), g.st_a, &[row, q8], &[])?;
    // Gate/up virtual row R -> h row hs + 16*(R>>5) + (R&15).
    let hrow_of = |b: &mut Builder, src: u8, dst: u8| -> Result<(), String> {
        vop(b, format!("v_lshrrev_b32_e32 v{dst}, 5, v{src}"), dst, &[src], &[])?;
        vop(b, format!("v_lshlrev_b32_e32 v{dst}, 4, v{dst}"), dst, &[dst], &[])?;
        vop(b, format!("v_and_b32_e32 v{x}, 15, v{src}", x = 15u8), 15, &[src], &[])?;
        vop(b, format!("v_add_nc_u32_e32 v{dst}, v{dst}, v15"), dst, &[dst, 15], &[])
    };
    if silu { hrow_of(b, r0, hr)?; }
    for (r, &dst) in g.st_w.iter().enumerate() {
        let (base, rel) = if silu { (g.hs, r as u32 * tile.round_rows() / 2) } else { (g.rs, r as u32 * tile.round_rows()) };
        if rel == 0 { sop(b, format!("s_mov_b32 s{x}, s{base}"), &[x], &[base])?; }
        else { sop(b, format!("s_add_co_i32 s{x}, s{base}, {}", lit(rel)), &[x], &[base])?; }
        vop(b, format!("v_add_nc_u32_e32 v{row}, s{x}, v{}", if silu { hr } else { r0 }), row, &[if silu { hr } else { r0 }], &[x])?;
        vop(b, format!("v_min_u32_e32 v{row}, s{}, v{row}", g.mm1), row, &[row], &[g.mm1])?;
        vop(b, format!("v_mul_lo_u32 v{row}, v{row}, s{}", g.gpr136), row, &[row], &[g.gpr136])?;
        vop(b, format!("v_add_nc_u32_e32 v{dst}, v{row}, v{q8}"), dst, &[row, q8], &[])?;
    }
    // Metadata lanes: token/row V = tid>>1 (T256 tokens wrap at 128).
    let (vv, tok) = (16u8, 17u8);
    vop(b, format!("v_lshrrev_b32_e32 v{vv}, 1, v0"), vv, &[0], &[])?;
    if tile == Tile::T128x128x8 {
        vop(b, format!("v_mov_b32_e32 v{tok}, v{vv}"), tok, &[vv], &[])?;
    } else {
        vop(b, format!("v_and_b32_e32 v{tok}, 0x7f, v{vv}"), tok, &[vv], &[])?;
        vop(b, format!("v_lshlrev_b32_e32 v{}, 2, v{vv}", g.meta_sz), g.meta_sz, &[vv], &[])?;
    }
    vop(b, format!("v_lshlrev_b32_e32 v{}, 2, v{tok}", g.meta_ds), g.meta_ds, &[tok], &[])?;
    vop(b, format!("v_add_nc_u32_e32 v{row}, s{}, v{tok}", g.bs), row, &[tok], &[g.bs])?;
    vop(b, format!("v_mul_u32_u24_e32 v{}, 0x48, v{row}", g.ds_voff), g.ds_voff, &[row], &[])?;
    if silu {
        hrow_of(b, vv, hr)?;
        vop(b, format!("v_add_nc_u32_e32 v{row}, s{}, v{hr}", g.hs), row, &[hr], &[g.hs])?;
    } else {
        vop(b, format!("v_add_nc_u32_e32 v{row}, s{}, v{vv}", g.rs), row, &[vv], &[g.rs])?;
    }
    vop(b, format!("v_min_u32_e32 v{row}, s{}, v{row}", g.mm1), row, &[row], &[g.mm1])?;
    vop(b, format!("v_mul_lo_u32 v{}, v{row}, s{}", g.sz_voff, g.gpr136), g.sz_voff, &[row], &[g.gpr136])?;
    // Fold addresses: row scales (wt_pair*32 + 8*k_grp)*4, token scale
    // DS0 + (tok_half*64 + m_lane)*4.
    sop(b, format!("s_lshl_b32 s{x}, s{wp}, 7"), &[x], &[wp])?;
    vop(b, format!("v_lshlrev_b32_e32 v{row}, 1, v{half}"), row, &[half], &[])?;
    vop(b, format!("v_add_nc_u32_e32 v{}, s{x}, v{row}", g.sc_addr), g.sc_addr, &[row], &[x])?;
    sop(b, format!("s_lshl_b32 s{x}, s{th}, 8"), &[x], &[th])?;
    sop(b, format!("s_addk_co_i32 s{x}, {}", lit(g.layout.ds[0])), &[x], &[x])?;
    vop(b, format!("v_lshlrev_b32_e32 v{row}, 2, v{ml}"), row, &[ml], &[])?;
    vop(b, format!("v_add_nc_u32_e32 v{}, s{x}, v{row}", g.d_addr), g.d_addr, &[row], &[x])?;

    // Block 0: slab 0, d plane 0, sc plane 0.
    b.clause(|b| {
        for (r, &dst) in g.a_pf.iter().enumerate() {
            publish::vload(b, dst, 2, g.st_a, a0, None, 8 + r as u32 * tile.round_rows() * super::spec::BLOCK_I4_128)?;
        }
        for (r, &dst) in g.w_pf.iter().enumerate() { publish::vload(b, dst, 2, g.st_w[r], g.srd_w, Some(g.goff), 8)?; }
        publish::vload(b, g.ds_nx, 1, g.ds_voff, a0, None, 0)?;
        publish::vload(b, g.sz_nx, 1, g.sz_voff, g.srd_z, Some(g.goff), 0)
    })?;
    for i in (0..64u8).step_by(2) {
        let r = g.acc + i;
        op(b, format!("v_dual_mov_b32 v{r}, 0 :: v_dual_mov_b32 v{}, 0", r + 1), &[v(r), v(r + 1)], &[])?;
    }
    for i in (0..8u8).step_by(2) {
        let r = g.magic + i;
        op(b, format!("v_dual_mov_b32 v{r}, {m} :: v_dual_mov_b32 v{}, {m}", r + 1, m = lit(super::spec::MAGIC)), &[v(r), v(r + 1)], &[])?;
    }
    publish::publish_slab(b, g, 0)?;
    publish::publish_meta(b, g, 0)?;
    b.barrier(&[Transition::Ready(g.slot_a[0]), Transition::Ready(g.slot_w[0]), Transition::Ready(g.slot_ds[0]), Transition::Ready(g.slot_sz[0])])
}
