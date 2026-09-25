//! One masked store path per epilogue family. Lane (m_lane, k_grp) of wave
//! (wt_pair, tok_half) owns, for column block nb and row group rg, rows
//! `wt_pair*32 + rg*16 + 8*k_grp + j` (j = 0..7) of token
//! `tok_half*64 + nb*16 + m_lane`: 32 contiguous bytes, written as two b128
//! stores straight from the accumulators. EXEC selects quads with row < M
//! (M % 4 == 0, so a quad is all in or all out) and tokens < N. Values are
//! verbatim: SET `Y = acc`, ADD `Y = RN(Y + acc)`, gate/up the imported
//! hipcc SiLU region `h = g / (1 + expf(-g)) * u`.
use super::{EPI, Epi, Gen, lit, mem, op, region::{self, Binding, Region}, s, sr, v, vr};
use crate::{Builder, insn::MemoryClass};

struct Epilogue { tok: u8, row: u8, off: u8, lim: u8, lim_off: [u8; 4], ntok: [u8; 4], rm: [u8; 4], tm: [u8; 4], nbo: [u8; 4], masks: [u8; 4] }

fn layout(g: &Gen) -> Epilogue {
    let e = g.epi_s;
    Epilogue { tok: g.f[0], row: g.f[0] + 1, off: g.f[0] + 2, lim: e,
        lim_off: [e + 1, e + 2, e + 3, e + 4], ntok: [e + 5, e + 6, e + 7, e + 8], rm: [e + 9, e + 10, e + 11, e + 12],
        tm: [e + 13, e + 14, e + 15, e + 16], nbo: [e + 17, e + 18, e + 19, e + 20], masks: [e + 21, e + 22, e + 23, g.tmp + 7] }
}

fn sop(b: &mut Builder, text: String, d: &[u8], u: &[u8]) -> Result<(), String> {
    op(b, text, &d.iter().map(|&n| s(n)).collect::<Vec<_>>(), &u.iter().map(|&n| s(n)).collect::<Vec<_>>())
}

/// `rg` row groups x `quads` quads per lane; SET/ADD: 2 x 2, gate/up: 1 x 2.
pub(crate) fn emit(b: &mut Builder, g: &Gen) -> Result<(), String> {
    b.label(EPI)?;
    let a = g.args;
    let silu = g.spec.epi == Epi::GateUpSilu;
    let e = layout(g);
    let rbase = if silu { g.hs } else { g.rs };
    let y = g.srd_y;
    // Tile-relative descriptor: Y + (bs * M + first row) * 4.
    sop(b, format!("s_mul_i32 s{y}, s{}, s{}", g.bs, a.m), &[y], &[g.bs, a.m])?;
    sop(b, format!("s_mul_hi_u32 s{}, s{}, s{}", y + 1, g.bs, a.m), &[y + 1], &[g.bs, a.m])?;
    sop(b, format!("s_add_co_u32 s{y}, s{y}, s{rbase}"), &[y], &[y, rbase])?;
    sop(b, format!("s_add_co_ci_u32 s{0}, s{0}, 0", y + 1), &[y + 1], &[y + 1])?;
    op(b, format!("s_lshl_b64 s[{y}:{}], s[{y}:{}], 2", y + 1, y + 1), &[sr(y, 2)], &[sr(y, 2)])?;
    sop(b, format!("s_add_co_u32 s{y}, s{y}, s{}", a.y), &[y], &[y, a.y])?;
    sop(b, format!("s_add_co_ci_u32 s{0}, s{0}, s{1}", y + 1, a.y + 1), &[y + 1], &[y + 1, a.y + 1])?;
    sop(b, format!("s_and_b32 s{0}, s{0}, 0xffff", y + 1), &[y + 1], &[y + 1])?;
    sop(b, format!("s_mov_b32 s{}, -1", y + 2), &[y + 2], &[])?;
    sop(b, format!("s_mov_b32 s{}, 0x31004000", y + 3), &[y + 3], &[])?;
    // Lane coordinates from the fold addresses: token tok_half*64 + m_lane,
    // row wt_pair*32 + 8*k_grp (gate/up: h row wt_pair*16 + 8*k_grp).
    op(b, format!("v_lshrrev_b32_e32 v{}, 2, v{}", e.tok, g.d_addr), &[v(e.tok)], &[v(g.d_addr)])?;
    op(b, format!("v_subrev_nc_u32_e32 v{0}, {1}, v{0}", e.tok, lit(g.layout.ds[0] / 4)), &[v(e.tok)], &[v(e.tok)])?;
    op(b, format!("v_lshrrev_b32_e32 v{}, 2, v{}", e.row, g.sc_addr), &[v(e.row)], &[v(g.sc_addr)])?;
    if silu {
        let x = g.tmp;
        sop(b, format!("s_lshr_b32 s{x}, s{}, 1", g.wave), &[x], &[g.wave])?;
        sop(b, format!("s_lshl_b32 s{x}, s{x}, 4"), &[x], &[x])?;
        op(b, format!("v_subrev_nc_u32_e32 v{0}, s{x}, v{0}", e.row), &[v(e.row)], &[v(e.row), s(x)])?;
    }
    op(b, format!("v_mul_lo_u32 v{}, v{}, s{}", e.off, e.tok, a.m), &[v(e.off)], &[v(e.tok), s(a.m)])?;
    op(b, format!("v_add_nc_u32_e32 v{0}, v{0}, v{1}", e.off, e.row), &[v(e.off)], &[v(e.off), v(e.row)])?;
    op(b, format!("v_lshlrev_b32_e32 v{0}, 2, v{0}", e.off), &[v(e.off)], &[v(e.off)])?;
    // Masks: quad (rg, c) valid iff row + rg*16 + 4c < M - first row;
    // column block nb valid iff token + 16*nb < N - bs.
    let rgs = if silu { 1 } else { 2 };
    sop(b, format!("s_sub_co_i32 s{}, s{}, s{rbase}", e.lim, a.m), &[e.lim], &[a.m, rbase])?;
    for rg in 0..rgs {
        for c in 0..2 {
            let i = rg * 2 + c;
            sop(b, format!("s_sub_co_i32 s{}, s{}, {}", e.lim_off[i], e.lim, lit((16 * rg + 4 * c) as u32)), &[e.lim_off[i]], &[e.lim])?;
            op(b, format!("v_cmp_gt_i32_e64 s{}, s{}, v{}", e.rm[i], e.lim_off[i], e.row), &[s(e.rm[i])], &[s(e.lim_off[i]), v(e.row)])?;
        }
    }
    sop(b, format!("s_sub_co_i32 s{}, s{}, s{}", e.ntok[0], a.n, g.bs), &[e.ntok[0]], &[a.n, g.bs])?;
    for nb in 1..4 { sop(b, format!("s_sub_co_i32 s{}, s{}, 16", e.ntok[nb], e.ntok[nb - 1]), &[e.ntok[nb]], &[e.ntok[nb - 1]])?; }
    for nb in 0..4 { op(b, format!("v_cmp_gt_i32_e64 s{}, s{}, v{}", e.tm[nb], e.ntok[nb], e.tok), &[s(e.tm[nb])], &[s(e.ntok[nb]), v(e.tok)])?; }
    sop(b, format!("s_mov_b32 s{}, 0", e.nbo[0]), &[e.nbo[0]], &[])?;
    sop(b, format!("s_lshl_b32 s{}, s{}, 6", e.nbo[1], a.m), &[e.nbo[1]], &[a.m])?;
    sop(b, format!("s_lshl_b32 s{}, s{}, 1", e.nbo[2], e.nbo[1]), &[e.nbo[2]], &[e.nbo[1]])?;
    sop(b, format!("s_add_co_i32 s{}, s{}, s{}", e.nbo[3], e.nbo[2], e.nbo[1]), &[e.nbo[3]], &[e.nbo[2], e.nbo[1]])?;
    op(b, "s_wait_alu depctr_va_sdst(0)", &[], &[])?;

    let quad = |b: &mut Builder, text: &str, data: u8, nb: usize, rg: usize, c: usize, load: bool| -> Result<(), String> {
        sop(b, format!("s_and_b32 exec_lo, s{}, s{}", e.rm[rg * 2 + c], e.tm[nb]), &[], &[e.rm[rg * 2 + c], e.tm[nb]])?;
        let imm = (64 * rg + 16 * c) as u32;
        let t = format!("{text} v[{}:{}], v{}, s[{y}:{}], s{} offen{}", data, data + 3, e.off, y + 3, e.nbo[nb], if imm == 0 { String::new() } else { format!(" offset:{imm}") });
        let uses = vec![v(e.off), sr(y, 4), s(e.nbo[nb])];
        if load { mem(b, t, &[vr(data, 4)], &uses, MemoryClass::VmemLoad) }
        else { let mut u = uses; u.push(vr(data, 4)); mem(b, t, &[], &u, MemoryClass::VmemStore) }
    };
    let acc = |nb: usize, rg: usize, c: usize| g.acc + (8 * (nb * 2 + rg) + 4 * c) as u8;
    match g.spec.epi {
        Epi::Set => {
            for nb in 0..4 { for rg in 0..2 { for c in 0..2 { quad(b, "buffer_store_b128", acc(nb, rg, c), nb, rg, c, false)?; } } }
        }
        Epi::Add => {
            // Old Y into the dead int32 accumulators, one quad per (nb, rg, c).
            let tmp = |i: usize| g.cacc + 4 * i as u8;
            let order: Vec<_> = (0..4).flat_map(|nb| (0..2).flat_map(move |rg| (0..2).map(move |c| (nb, rg, c)))).collect();
            for (i, &(nb, rg, c)) in order.iter().enumerate() { quad(b, "buffer_load_b128", tmp(i), nb, rg, c, true)?; }
            op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
            for (i, &(nb, rg, c)) in order.iter().enumerate() {
                for p in [0u8, 2] {
                    let (t, x) = (tmp(i) + p, acc(nb, rg, c) + p);
                    op(b, format!("v_dual_add_f32 v{t}, v{t}, v{x} :: v_dual_add_f32 v{}, v{}, v{}", t + 1, t + 1, x + 1),
                        &[v(t), v(t + 1)], &[v(t), v(t + 1), v(x), v(x + 1)])?;
                }
            }
            for (i, &(nb, rg, c)) in order.iter().enumerate() { quad(b, "buffer_store_b128", tmp(i), nb, rg, c, false)?; }
        }
        Epi::GateUpSilu => {
            let silu_region = Region::silu()?;
            if silu_region.temps > region::SILU_TEMPS || silu_region.masks > region::SILU_MASKS { return Err("SiLU region needs more temporaries than planned".into()) }
            for nb in 0..4 {
                if nb > 0 { op(b, "s_mov_b32 exec_lo, -1", &[], &[])?; }
                // One output octet per column block (v16..v47): no register
                // is rewritten while a store reads it, so no store-counter
                // wait (whose completion order the ledger replay rightly
                // does not assume) is ever needed.
                let out = g.cacc + 16 + 8 * nb as u8;
                for j in (0..8u8).step_by(2) {
                    let binds: Vec<_> = (0..2u8).map(|s| Binding {
                        g: acc(nb, 0, 0) + j + s, u: acc(nb, 1, 0) + j + s, out: out + j + s,
                        temps: (0..region::SILU_TEMPS as u8).map(|t| g.cacc + 7 * s + t).collect(),
                        masks: vec![e.masks[2 * s as usize], e.masks[2 * s as usize + 1]],
                    }).collect();
                    region::emit_interleaved(b, &silu_region, &binds)?;
                }
                for c in 0..2 { quad(b, "buffer_store_b128", out + 4 * c as u8, nb, 0, c, false)?; }
            }
        }
    }
    Ok(())
}
