// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! One masked store path per epilogue family. Lane (m_lane, k_grp) of wave
//! (wt_pair, tok_half) owns, for column block nb and row group rg, rows
//! `wt_pair*32 + rg*16 + 8*k_grp + j` (j = 0..7) of token
//! `tok_half*64 + nb*16 + m_lane`: two b128 stores for f32, four b32 stores
//! for paired bf16. EXEC selects quads with row < M (M % 4 == 0) and
//! tokens < N. SET writes acc, ADD writes RN(Y + acc), and gate/up uses the
//! imported hipcc SiLU region `h = g / (1 + expf(-g)) * u`; the packed
//! variant rounds each h to bf16 RNE before storing.
use super::{
    region::{self, Binding, Region},
    Epi, Gen, IdleHalf, QkvStore, Wg, END, EPI,
};
use crate::kernels::{
    bf16::Bf16,
    common::{lit, mem, op, s, sop, sr, v, vr},
};
use crate::{
    insn::{Instruction, MemoryClass},
    reg::Live,
    Builder,
};
use peacemaker_author::End;

struct Epilogue {
    tok: u8,
    row: u8,
    off: u8,
    lim: u8,
    lim_off: [u8; 4],
    ntok: [u8; 4],
    rm: [u8; 4],
    tm: [u8; 4],
    nbo: [u8; 4],
    masks: [u8; 4],
}
/// Concurrently live SiLU region instances (register slots).
const SILU_SLOTS: u8 = 4;
/// `Epi::AddSigmoid`: per-nb gate logit / sigmoid value (one register each,
/// the region reads its input only in its first op and writes its output in
/// its last), the gate byte offset, and the four sigmoid slots' temporaries:
/// slot q owns `SIG_TEMPS + 6q .. +6` (v139..v162; v136..v138 are tok/row/off).
const SIG_VALUE: u8 = 128;
const SIG_OFF: u8 = 132;
const SIG_TEMPS: u8 = 139;
/// First SGPR of the gate buffer descriptor (past `epi_s`, s64..s87).
const SIG_SRD: u8 = 88;

/// Registers the sigmoid residual ADD epilogue needs beyond the common plan:
/// v128..v135 (dead `magic8`), v160..v162 (dead `d_x`/prefetch) and s88..s91.
pub(crate) fn plan_sigmoid(p: &mut crate::RegPlan) -> Result<(), String> {
    let epi = || Live::Between(EPI.into(), END.into());
    p.v::<4>("sig_value", SIG_VALUE, epi())?;
    p.v::<1>("sig_gate_off", SIG_OFF, epi())?;
    for r in 160..=162u8 {
        p.v::<1>("sig_tmp", r, epi())?;
    }
    p.s::<4>("srd_gate", SIG_SRD, epi())?;
    Ok(())
}

fn layout(g: &Gen) -> Epilogue {
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
        masks: [e + 21, e + 22, e + 23, g.tmp + 7],
    }
}

pub(crate) fn emit(wg: &mut Wg, g: &Gen, end: &End) -> Result<(), String> {
    wg.label(EPI)?;
    if g.spec.epi == Epi::QkvzaGdn {
        return fused_stores(wg, g, end);
    }
    let b = wg.isa();
    // Below the K-loop's priority 1: co-resident K-loops issue first.
    op(b, "s_setprio 0", &[], &[])?;
    let a = g.args;
    let rbase = if g.spec.epi.is_silu() { g.hs } else { g.rs };
    if g.spec.tune.idle == IdleHalf::Off {
        return store(b, g, g.spec.epi, a.y, a.m, rbase, None);
    }
    // The token-half-1 waves of a tile whose tokens all precede it (`N - bs <= 64`)
    // compute nothing and store nothing (every column block is masked off): they
    // leave without the store path. It holds no barrier, so skipping it is local.
    let live = wg.scmp(Instruction::new(
        format!("s_cmp_lg_u32 s{}, 0", g.idle),
        vec![],
        vec![s(g.idle)],
    ))?;
    wg.skip_if(live, &format!("{EPI}_live"), (), |w, ()| {
        store(w.isa(), g, g.spec.epi, a.y, a.m, rbase, None)
    })
}

/// Fused projection, non-GDN tiles: Z tiles store like SET into Yz (stride
/// Mz); the beta/alpha tile stores rows 0..47 (waves 0-3) into Ybeta and rows
/// 64..111 (waves 4-7) into Yalpha, both with stride 48; both then leave the
/// kernel. QKV tiles continue at the GDN preparation (`gdn_epilogue`).
fn fused_stores(wg: &mut Wg, g: &Gen, end: &End) -> Result<(), String> {
    let (mz, seg, yz, yb, ya) = (21u8, 31u8, 14u8, 16u8, 18u8);
    let (adj, mba, zero) = (g.srd_w, g.srd_w + 1, g.srd_w + 2);
    // The segment comes from the row tile (workgroup ids and kernel arguments).
    let qkv = wg.scmp_wg_uniform(Instruction::new(
        format!("s_cmp_eq_u32 s{seg}, 0"),
        vec![],
        vec![s(seg)],
    ))?;
    // `QkvStore::Gdn`: QKV tiles continue at the GDN preparation. `QkvStore::Set`: they continue at
    // `.Liu4_epilogue_qkv` below, store every row like SET and fall into the kernel exit.
    let qkv_target = if g.spec.tune.qkv == QkvStore::Set {
        format!("{EPI}_qkv")
    } else {
        super::GDN.to_string()
    };
    wg.exit_unless(qkv, &qkv_target, end, |wg| {
        let (ba, stored) = (format!("{EPI}_ba"), format!("{EPI}_stored"));
        wg.forward(|w, f| {
            let beta_alpha = w.scmp(Instruction::new(
                format!("s_cmp_ge_u32 s{}, s{mz}", g.rs),
                vec![],
                vec![s(g.rs), s(mz)],
            ))?;
            f.branch_if(w, beta_alpha, &ba)?;
            let b = w.isa();
            store(b, g, Epi::Set, yz, mz, g.rs, None)?;
            b.wait_all()?;
            f.goto(w, &stored)?;
            f.place(w, &ba)?;
            let b = w.isa();
            // Row pair (`wt_pair`) bit 1 picks the alpha rows: wave bit 2, or bit 1 once
            // `IdleHalf::SkipHi` remapped `wt_pair = wave & 3`.
            let alpha_bit = if g.spec.tune.idle == IdleHalf::SkipHi {
                1
            } else {
                2
            };
            sop(
                b,
                format!("s_bitcmp1_b32 s{}, {alpha_bit}", g.wave),
                &[],
                &[g.wave],
            )?;
            op(
                b,
                format!(
                    "s_cselect_b64 s[{yb}:{}], s[{ya}:{}], s[{yb}:{}]",
                    yb + 1,
                    ya + 1,
                    yb + 1
                ),
                &[sr(yb, 2)],
                &[sr(ya, 2), sr(yb, 2)],
            )?;
            sop(
                b,
                format!(
                    "s_cselect_b32 s{adj}, {}, 0",
                    lit(super::spec::GDN_FOLD_ALPHA_ROW)
                ),
                &[adj],
                &[],
            )?;
            sop(b, format!("s_mov_b32 s{mba}, 48"), &[mba], &[])?;
            sop(b, format!("s_mov_b32 s{zero}, 0"), &[zero], &[])?;
            store(b, g, Epi::Set, yb, mba, zero, Some(adj))?;
            b.wait_all()?;
            f.place(w, &stored)
        })?;
        op(wg.isa(), "s_mov_b32 exec_lo, -1", &[], &[])
    })?;
    if g.spec.tune.qkv == QkvStore::Set {
        // QKV tile: `X` (kernarg offset 104) is the plain QKV output, row stride M0 (s20).
        let (m0, yq) = (20u8, g.step2);
        let b = wg.isa();
        mem(
            b,
            format!(
                "s_load_b64 s[{yq}:{}], s[{}:{}], 0x68",
                yq + 1,
                g.karg,
                g.karg + 1
            ),
            &[sr(yq, 2)],
            &[sr(g.karg, 2)],
            MemoryClass::SmemLoad,
        )?;
        store(b, g, Epi::Set, yq, m0, g.rs, None)?;
        b.wait_all()?;
    }
    Ok(())
}

/// `rg` row groups x `quads` quads per lane; SET/ADD: 2 x 2, gate/up: 1 x 2.
/// Stores `epi` into the tile of `Y` (pointer SGPRs `yp`) with row stride
/// `s{m}` from row `s{rbase}`; `row_adjust` subtracts a wave-uniform SGPR
/// from the lane row (the beta/alpha tile's alpha half).
fn store(
    b: &mut Builder,
    g: &Gen,
    epi: Epi,
    yp: u8,
    m: u8,
    rbase: u8,
    row_adjust: Option<u8>,
) -> Result<(), String> {
    let a = g.args;
    let silu = epi.is_silu();
    let e = layout(g);
    let y = g.srd_y;
    // Tile-relative descriptor: Y + (bs * M + first row) * element size.
    sop(
        b,
        format!("s_mul_i32 s{y}, s{}, s{}", g.bs, m),
        &[y],
        &[g.bs, m],
    )?;
    sop(
        b,
        format!("s_mul_hi_u32 s{}, s{}, s{}", y + 1, g.bs, m),
        &[y + 1],
        &[g.bs, m],
    )?;
    sop(
        b,
        format!("s_add_co_u32 s{y}, s{y}, s{rbase}"),
        &[y],
        &[y, rbase],
    )?;
    sop(
        b,
        format!("s_add_co_ci_u32 s{0}, s{0}, 0", y + 1),
        &[y + 1],
        &[y + 1],
    )?;
    op(
        b,
        format!(
            "s_lshl_b64 s[{y}:{}], s[{y}:{}], {}",
            y + 1,
            y + 1,
            if epi == Epi::GateUpSiluBf16 { 1 } else { 2 }
        ),
        &[sr(y, 2)],
        &[sr(y, 2)],
    )?;
    sop(b, format!("s_add_co_u32 s{y}, s{y}, s{yp}"), &[y], &[y, yp])?;
    sop(
        b,
        format!("s_add_co_ci_u32 s{0}, s{0}, s{1}", y + 1, yp + 1),
        &[y + 1],
        &[y + 1, yp + 1],
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
    // Lane coordinates from the fold addresses: token tok_half*64 + m_lane,
    // row wt_pair*32 + 8*k_grp (gate/up: h row wt_pair*16 + 8*k_grp).
    op(
        b,
        format!("v_lshrrev_b32_e32 v{}, 2, v{}", e.tok, g.d_addr),
        &[v(e.tok)],
        &[v(g.d_addr)],
    )?;
    op(
        b,
        format!(
            "v_subrev_nc_u32_e32 v{0}, {1}, v{0}",
            e.tok,
            lit(g.layout.ds[0] / 4)
        ),
        &[v(e.tok)],
        &[v(e.tok)],
    )?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{}, 2, v{}", e.row, g.sc_addr),
        &[v(e.row)],
        &[v(g.sc_addr)],
    )?;
    if silu {
        let x = g.tmp;
        sop(
            b,
            if g.spec.tune.idle == IdleHalf::SkipHi {
                format!("s_and_b32 s{x}, s{}, 3", g.wave)
            } else {
                format!("s_lshr_b32 s{x}, s{}, 1", g.wave)
            },
            &[x],
            &[g.wave],
        )?;
        sop(b, format!("s_lshl_b32 s{x}, s{x}, 4"), &[x], &[x])?;
        op(
            b,
            format!("v_subrev_nc_u32_e32 v{0}, s{x}, v{0}", e.row),
            &[v(e.row)],
            &[v(e.row), s(x)],
        )?;
    }
    if let Some(x) = row_adjust {
        op(
            b,
            format!("v_subrev_nc_u32_e32 v{0}, s{x}, v{0}", e.row),
            &[v(e.row)],
            &[v(e.row), s(x)],
        )?;
    }
    op(
        b,
        format!("v_mul_lo_u32 v{}, v{}, s{m}", e.off, e.tok),
        &[v(e.off)],
        &[v(e.tok), s(m)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{0}, v{0}, v{1}", e.off, e.row),
        &[v(e.off)],
        &[v(e.off), v(e.row)],
    )?;
    op(
        b,
        format!(
            "v_lshlrev_b32_e32 v{0}, {1}, v{0}",
            e.off,
            if epi == Epi::GateUpSiluBf16 { 1 } else { 2 }
        ),
        &[v(e.off)],
        &[v(e.off)],
    )?;
    // Masks: quad (rg, c) valid iff row + rg*16 + 4c < M - first row;
    // column block nb valid iff token + 16*nb < N - bs.
    let rgs = if silu { 1 } else { 2 };
    sop(
        b,
        format!("s_sub_co_i32 s{}, s{m}, s{rbase}", e.lim),
        &[e.lim],
        &[m, rbase],
    )?;
    for rg in 0..rgs {
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
    sop(b, format!("s_mov_b32 s{}, 0", e.nbo[0]), &[e.nbo[0]], &[])?;
    sop(
        b,
        format!(
            "s_lshl_b32 s{}, s{m}, {}",
            e.nbo[1],
            if epi == Epi::GateUpSiluBf16 { 5 } else { 6 }
        ),
        &[e.nbo[1]],
        &[m],
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

    let quad = |b: &mut Builder,
                text: &str,
                data: u8,
                nb: usize,
                rg: usize,
                c: usize,
                load: bool|
     -> Result<(), String> {
        sop(
            b,
            format!("s_and_b32 exec_lo, s{}, s{}", e.rm[rg * 2 + c], e.tm[nb]),
            &[],
            &[e.rm[rg * 2 + c], e.tm[nb]],
        )?;
        let imm = (64 * rg + 16 * c) as u32;
        let t = format!(
            "{text} v[{}:{}], v{}, s[{y}:{}], s{} offen{}",
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
        );
        let uses = vec![v(e.off), sr(y, 4), s(e.nbo[nb])];
        if load {
            mem(b, t, &[vr(data, 4)], &uses, MemoryClass::VmemLoad)
        } else {
            let mut u = uses;
            u.push(vr(data, 4));
            mem(b, t, &[], &u, MemoryClass::VmemStore)
        }
    };
    let acc = |nb: usize, rg: usize, c: usize| g.acc + (8 * (nb * 2 + rg) + 4 * c) as u8;
    match epi {
        Epi::QkvzaGdn => return Err("the fused projection stores through fused_stores".into()),
        Epi::Set => {
            for nb in 0..4 {
                for rg in 0..2 {
                    for c in 0..2 {
                        quad(b, "buffer_store_b128", acc(nb, rg, c), nb, rg, c, false)?;
                    }
                }
            }
        }
        Epi::Add => {
            // Old Y into the dead int32 accumulators, one quad per (nb, rg, c).
            let tmp = |i: usize| g.cacc + 4 * i as u8;
            let order: Vec<_> = (0..4)
                .flat_map(|nb| (0..2).flat_map(move |rg| (0..2).map(move |c| (nb, rg, c))))
                .collect();
            for (i, &(nb, rg, c)) in order.iter().enumerate() {
                quad(b, "buffer_load_b128", tmp(i), nb, rg, c, true)?;
            }
            op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
            for (i, &(nb, rg, c)) in order.iter().enumerate() {
                for p in [0u8, 2] {
                    let (t, x) = (tmp(i) + p, acc(nb, rg, c) + p);
                    op(
                        b,
                        format!(
                            "v_dual_add_f32 v{t}, v{t}, v{x} :: v_dual_add_f32 v{}, v{}, v{}",
                            t + 1,
                            t + 1,
                            x + 1
                        ),
                        &[v(t), v(t + 1)],
                        &[v(t), v(t + 1), v(x), v(x + 1)],
                    )?;
                }
            }
            for (i, &(nb, rg, c)) in order.iter().enumerate() {
                quad(b, "buffer_store_b128", tmp(i), nb, rg, c, false)?;
            }
        }
        Epi::AddSigmoid => {
            // Y = fma(sigmoid(gate[token]), 0.0f + acc, Y), hipcc `full_add_v3`
            // `add == 2` (`iu4_sigmoid_residual`): the `0.0f +` turns an acc of -0
            // into +0 and the product is one fused rounding, like the incumbent
            // `sigmoid_scaled_residual_add_batched_f32`'s v_fmac_f32.
            let gate = g
                .args
                .gate
                .ok_or("the sigmoid residual ADD needs a shared_gate argument")?;
            let sigmoid = Region::sigmoid()?;
            if sigmoid.temps > region::SIGMOID_TEMPS {
                return Err("sigmoid region needs more temporaries than planned".into());
            }
            // Gate descriptor: base = shared_gate, N * 4 bytes (tokens >= N read 0
            // and are never stored: their `tm` lane mask is clear).
            let d = SIG_SRD;
            sop(b, format!("s_mov_b32 s{d}, s{gate}"), &[d], &[gate])?;
            sop(
                b,
                format!("s_and_b32 s{}, s{}, 0xffff", d + 1, gate + 1),
                &[d + 1],
                &[gate + 1],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{}, s{}, 2", d + 2, a.n),
                &[d + 2],
                &[a.n],
            )?;
            sop(
                b,
                format!("s_mov_b32 s{}, 0x31004000", d + 3),
                &[d + 3],
                &[],
            )?;
            // Byte offset of this lane's first token (bs + tok); column block nb adds 64 B.
            op(
                b,
                format!("v_add_nc_u32_e32 v{SIG_OFF}, s{}, v{}", g.bs, e.tok),
                &[v(SIG_OFF)],
                &[s(g.bs), v(e.tok)],
            )?;
            op(
                b,
                format!("v_lshlrev_b32_e32 v{0}, 2, v{0}", SIG_OFF),
                &[v(SIG_OFF)],
                &[v(SIG_OFF)],
            )?;
            for nb in 0..4u8 {
                let val = SIG_VALUE + nb;
                mem(
                    b,
                    format!(
                        "buffer_load_b32 v{val}, v{SIG_OFF}, s[{d}:{}], null offen{}",
                        d + 3,
                        if nb == 0 {
                            String::new()
                        } else {
                            format!(" offset:{}", 64 * u32::from(nb))
                        }
                    ),
                    &[v(val)],
                    &[v(SIG_OFF), sr(d, 4)],
                    MemoryClass::VmemLoad,
                )?;
            }
            // Old Y into the dead int32 accumulators, one quad per (nb, rg, c).
            let tmp = |i: usize| g.cacc + 4 * i as u8;
            let order: Vec<_> = (0..4)
                .flat_map(|nb| (0..2).flat_map(move |rg| (0..2).map(move |c| (nb, rg, c))))
                .collect();
            for (i, &(nb, rg, c)) in order.iter().enumerate() {
                quad(b, "buffer_load_b128", tmp(i), nb, rg, c, true)?;
            }
            op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
            // Four live sigmoid instances: the logit of column block nb in
            // v{128+nb}, replaced in place by its sigmoid.
            let binds: Vec<_> = (0..4u8)
                .map(|nb| Binding {
                    g: SIG_VALUE + nb,
                    u: SIG_VALUE + nb,
                    out: SIG_VALUE + nb,
                    temps: (0..region::SIGMOID_TEMPS as u8)
                        .map(|t| SIG_TEMPS + 6 * nb + t)
                        .collect(),
                    masks: vec![],
                })
                .collect();
            region::emit_interleaved(b, &sigmoid, &binds)?;
            for (i, &(nb, rg, c)) in order.iter().enumerate() {
                let sig = SIG_VALUE + nb as u8;
                for p in [0u8, 2] {
                    let (t, x) = (tmp(i) + p, acc(nb, rg, c) + p);
                    // x = 0.0f + x (a -0 acc becomes +0), then t = fma(sig, x, t).
                    op(
                        b,
                        format!(
                            "v_dual_add_f32 v{x}, 0, v{x} :: v_dual_add_f32 v{}, 0, v{}",
                            x + 1,
                            x + 1
                        ),
                        &[v(x), v(x + 1)],
                        &[v(x), v(x + 1)],
                    )?;
                    op(
                        b,
                        format!(
                            "v_dual_fmac_f32 v{t}, v{x}, v{sig} :: v_dual_fmac_f32 v{}, v{}, v{sig}",
                            t + 1,
                            x + 1
                        ),
                        &[v(t), v(t + 1)],
                        &[v(t), v(t + 1), v(x), v(x + 1), v(sig)],
                    )?;
                }
            }
            for (i, &(nb, rg, c)) in order.iter().enumerate() {
                quad(b, "buffer_store_b128", tmp(i), nb, rg, c, false)?;
            }
        }
        Epi::GateUpSilu | Epi::GateUpSiluBf16 => {
            let silu_region = Region::silu()?;
            if silu_region.temps > region::SILU_TEMPS || silu_region.masks > region::SILU_MASKS {
                return Err("SiLU region needs more temporaries than planned".into());
            }
            // Four live instances: register slot q owns temporaries
            // v[7q : 7q + 6] (v0..v27) and two lane-mask SGPRs, the four
            // `e.masks` then s56..s59 (prologue temporaries, dead once the row
            // offset above has read s56). Element k of a column block uses
            // slot k % 4, so k + 4 starts once k has issued its last op; VOPD
            // partners (2i, 2i + 1) sit an odd distance apart in every operand
            // (7 temporaries, one accumulator or output), which keeps each
            // same-op packet parity- and bank-legal.
            let masks = [
                e.masks[0],
                e.masks[1],
                e.masks[2],
                e.masks[3],
                g.tmp,
                g.tmp + 1,
                g.tmp + 2,
                g.tmp + 3,
            ];
            for nb in 0..4 {
                if nb > 0 {
                    op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
                }
                // One output octet per column block (v32..v63): no register
                // is rewritten while a store reads it, so no store-counter
                // wait (whose completion order the ledger replay rightly
                // does not assume) is ever needed.
                let out = g.cacc + 32 + 8 * nb as u8;
                let binds: Vec<_> = (0..8u8)
                    .map(|k| {
                        let q = k % SILU_SLOTS;
                        Binding {
                            g: acc(nb, 0, 0) + k,
                            u: acc(nb, 1, 0) + k,
                            out: out + k,
                            temps: (0..region::SILU_TEMPS as u8)
                                .map(|t| g.cacc + 7 * q + t)
                                .collect(),
                            masks: vec![masks[2 * q as usize], masks[2 * q as usize + 1]],
                        }
                    })
                    .collect();
                region::emit_interleaved(b, &silu_region, &binds)?;
                if epi == Epi::GateUpSiluBf16 {
                    // One pair of consecutive f32 h values per packed dword.
                    // In-place packing is safe: each source is consumed before
                    // that register is reused as a packed destination.
                    for k in 0..4u8 {
                        let dst = out + k;
                        let src0 = out + 2 * k;
                        // v0/v1 are packed-lane results; dead SiLU temporaries v2..v5 are scratch.
                        Bf16::hip_bfloat16(b, src0, 0, [2, 3, 4, 5])?;
                        Bf16::hip_bfloat16(b, src0 + 1, 1, [2, 3, 4, 5])?;
                        op(b, "v_lshlrev_b32_e32 v1, 16, v1", &[v(1)], &[v(1)])?;
                        op(
                            b,
                            format!("v_or_b32_e32 v{dst}, v0, v1"),
                            &[v(dst)],
                            &[v(0), v(1)],
                        )?;
                    }
                    for c in 0..2 {
                        sop(
                            b,
                            format!("s_and_b32 exec_lo, s{}, s{}", e.rm[c], e.tm[nb]),
                            &[],
                            &[e.rm[c], e.tm[nb]],
                        )?;
                        for p in 0..2u8 {
                            let data = out + 2 * c as u8 + p;
                            let imm = 8 * c as u32 + 4 * p as u32;
                            let t = format!(
                                "buffer_store_b32 v{data}, v{}, s[{y}:{}], s{} offen{}",
                                e.off,
                                y + 3,
                                e.nbo[nb],
                                if imm == 0 {
                                    String::new()
                                } else {
                                    format!(" offset:{imm}")
                                }
                            );
                            mem(
                                b,
                                t,
                                &[],
                                &[v(e.off), sr(y, 4), s(e.nbo[nb]), v(data)],
                                MemoryClass::VmemStore,
                            )?;
                        }
                    }
                } else {
                    for c in 0..2 {
                        quad(b, "buffer_store_b128", out + 4 * c as u8, nb, 0, c, false)?;
                    }
                }
            }
        }
    }
    Ok(())
}
