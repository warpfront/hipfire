// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Epi::SetNr`: the Flash-Next QSA projection GEMM (`T256x128`) with `qsa_norm_rope_head`
//! (`kernels/src/tensor_ops.hip`) in its epilogue, bit for bit.
//!
//! # What the tile owns
//! A 256-row tile at row start `rs` is one head's 256 channels when `(rs & head_mask) < 256`
//! (`head_mask` = head row stride - 1, kernel word at byte 36: 511 for the per-head
//! `[q(256) gate(256)]` q projection, whose gate tiles keep the plain store; 255 for a plain k
//! projection). Such a tile holds, for each of its 128 tokens, the whole 256-channel row the norm
//! reduces over: token `t` at lane `m` of wave `(wp, th)` for rows `32 wp + 16 rg + 8 kg + j`.
//!
//! # Arithmetic (the instruction sequence of the hipcc kernel, measured on its disassembly)
//! - `sum = 0; for i in 0..256 { sum = fma(x_i, x_i, sum) }` serially in channel order
//!   (`v_fmac_f32`), one lane per token. The tile's 256 channels reach that lane through LDS in
//!   four 64-row passes (`[token][64 + pad]` f32, 272-byte pitch); the sum stays in a register.
//! - `inv = rsq(sum * 2^-8 + 1e-6)`: the kernel divides by `(float)head_dim` (256), which is an
//!   exact power-of-two scaling, so `v_mul_f32` by 2^-8 is the same correctly rounded value.
//! - `s = (1 + bf16(w)) * (x * inv)`, three roundings in that order.
//! - Rows < 64 (`rotary_dim`, waves with `wp < 2`): HalfSplit RoPE with the partner row `+-32`
//!   (the wave `wp ^ 1`, same lane and register: exchanged through LDS) and the table's cosine and
//!   sine of that token's position and base (`rope`: `[token][64]` f32, `rope_yarn_sincos`
//!   already applied, so `sin`/`cos` carry no rounding here):
//!   `out = fma(s, cos, +-RN(s_partner * sin))`, minus for the low half (`v_mul` then `v_fma`,
//!   exactly `v_mul_f32` + `v_fmac_f32` of the hipcc kernel).
//!   The accumulators then go through the unchanged F32 store loop.
use super::{epilogue, spec, Gen, Lds, Raster, Wg, Wv, END, EPI};
use crate::kernels::common::{bload, lit, op, s, smem, sop, v, vr, SRD_WORD3};
use crate::{
    insn::{Instruction, MemoryClass},
    reg::{Live, RegRef},
    Builder, RegPlan,
};
use peacemaker_author::{ready, retire, retire_cur, Free, LdsRegion, Published};

/// LDS tag of the epilogue region (staging pass, inverse-RMS plane, RoPE partner exchange).
pub enum NrRegion {}

/// The whole `T256x128` launch allocation.
const LDS_BYTES: u32 = 55296;
/// Bytes per staged token row: 64 f32 plus 16 bytes of padding (bank skew, 16-byte alignment).
const STAGE_PITCH: u32 = 272;

// VGPRs (see `plan`): the accumulators `v64..v127` hold the tile's values throughout.
const T_SCRATCH: u8 = 131;
/// `1 + bf16(norm[row])` of the lane's 16 rows, index `rg * 8 + j` (v132..v147).
const W1: u8 = 132;
/// Per token block `nb` inverse RMS (v148..v151).
const INV: u8 = 148;
const SUM: u8 = 152;
const MT: u8 = 153;
/// LDS: staging store base, chain read base, exchange store base, exchange partner base.
const WR: u8 = 154;
const RD: u8 = 155;
const SW: u8 = 156;
const SR: u8 = 157;
/// Table (RoPE) byte offset and norm-weight byte offset of the lane.
const TV: u8 = 158;
const NV: u8 = 159;
/// Partner values of one token block (v160..v175).
const PART: u8 = 160;
const LANE: u8 = 176;
const TC: u8 = 177;
const TCI: u8 = 178;
const INVC: u8 = 179;
const INVA: u8 = 180;
/// Raw norm weights (v184..v191), then four RoPE temporaries (v184..v187).
const RAW: u8 = 184;
const SX: u8 = 184;
/// The VGPR pool v0..v63: the chain's 64 staged values, then two table buffers (cos, sin) of 32.
const POOL: u8 = 0;

/// Scalars: the descriptors reuse the dead K-loop ones, the rest `g.tmp`.
const T0: u8 = 0;
const SIGN: u8 = 1;
const T3: u8 = 3;

pub(crate) fn plan(p: &mut RegPlan, g: &Gen) -> Result<(), String> {
    let epi = || Live::Between(EPI.into(), END.into());
    for i in 0..16u8 {
        p.v::<4>("nr_pool", g.cacc + 4 * i, epi())?;
    }
    for r in 128..132u8 {
        p.v::<1>("nr_t", r, epi())?;
    }
    for q in 0..5u8 {
        p.v::<4>("nr_w", W1 + 4 * q, epi())?;
    }
    for r in 152..160u8 {
        p.v::<1>("nr_s", r, epi())?;
    }
    for q in 0..4u8 {
        p.v::<4>("nr_part", PART + 4 * q, epi())?;
    }
    for r in 176..182u8 {
        p.v::<1>("nr_a", r, epi())?;
    }
    for q in 0..2u8 {
        p.v::<4>("nr_raw", RAW + 4 * q, epi())?;
    }
    Ok(())
}

fn vop(b: &mut Builder, text: String, defs: &[u8], uses: &[RegRef]) -> Result<(), String> {
    op(
        b,
        text,
        &defs.iter().map(|&n| v(n)).collect::<Vec<_>>(),
        uses,
    )
}

fn off(imm: u32) -> String {
    if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    }
}
fn ds_store_b128(addr: u8, data: u8, imm: u32) -> Instruction {
    Instruction::new(
        format!("ds_store_b128 v{addr}, {}{}", vr(data, 4), off(imm)),
        vec![],
        vec![v(addr), vr(data, 4)],
    )
    .memory(MemoryClass::DsStore)
}
fn ds_store_b32(addr: u8, data: u8) -> Instruction {
    Instruction::new(
        format!("ds_store_b32 v{addr}, v{data}"),
        vec![],
        vec![v(addr), v(data)],
    )
    .memory(MemoryClass::DsStore)
}
fn ds_load_b128(dst: u8, addr: u8, imm: u32) -> Instruction {
    Instruction::new(
        format!("ds_load_b128 {}, v{addr}{}", vr(dst, 4), off(imm)),
        vec![vr(dst, 4)],
        vec![v(addr)],
    )
    .memory(MemoryClass::DsLoad)
}
fn ds_load_b32(dst: u8, addr: u8, imm: u32) -> Instruction {
    Instruction::new(
        format!("ds_load_b32 v{dst}, v{addr}{}", off(imm)),
        vec![v(dst)],
        vec![v(addr)],
    )
    .memory(MemoryClass::DsLoad)
}

/// Accumulator quad (rows `4c..4c+3` of row group `rg`, token block `nb`).
fn acc(g: &Gen, nb: usize, rg: usize, c: usize) -> u8 {
    g.acc + (8 * (nb * 2 + rg) + 4 * c) as u8
}

pub(crate) fn emit(wg: &mut Wg, g: &Gen, lds: Lds) -> Result<(), String> {
    let (ra, rw, rd, rs) = lds;
    wg.label(EPI)?;
    // Below the K-loop's priority 1: co-resident K-loops issue first.
    if g.spec.raster == Raster::Banded {
        op(wg.isa(), "s_setprio 0", &[], &[])?;
    }
    // Every wave has finished its last block's fragment and plane reads.
    let freed = wg.barrier((
        retire_cur(ra),
        retire_cur(rw),
        retire_cur(rd),
        retire_cur(rs),
    ))?;
    wg.relayout(freed)?;
    let region = wg.lds::<NrRegion>("nr", 0, LDS_BYTES)?;
    let e = epilogue::setup(wg.isa(), g)?;
    // The tile carries a head's q rows (or a whole k head) iff (rs & head_mask) < 256.
    let t0 = g.tmp + T0;
    let mask = g.args.n + 1;
    sop(
        wg.isa(),
        format!("s_and_b32 s{t0}, s{}, s{mask}", g.rs),
        &[t0],
        &[g.rs, mask],
    )?;
    let plain = wg.scmp_wg_uniform(Instruction::new(
        format!("s_cmp_ge_u32 s{t0}, 0x100"),
        vec![],
        vec![s(t0)],
    ))?;
    let _region = wg.wg_skip_if(plain, ".Lmq6i8_nr_plain", region, |wg, region| {
        fused(wg, g, &e, region)
    })?;
    epilogue::stores(wg.isa(), g, &e)
}

/// Scalar compare of `wave >> shift` or `wave` against `imm`, as a wave-uniform condition.
fn wave_is(
    wg: &mut Wg,
    g: &Gen,
    shift: u32,
    cmp: &str,
    imm: u32,
) -> Result<peacemaker_author::Uniform<peacemaker_author::Scc>, String> {
    let t = g.tmp + T3;
    let src = if shift == 0 {
        g.wave
    } else {
        sop(
            wg.isa(),
            format!("s_lshr_b32 s{t}, s{}, {shift}", g.wave),
            &[t],
            &[g.wave],
        )?;
        t
    };
    wg.scmp(Instruction::new(
        format!("{cmp} s{src}, {imm}"),
        vec![],
        vec![s(src)],
    ))
}

fn fused(
    wg: &mut Wg,
    g: &Gen,
    e: &epilogue::Epilogue,
    mut region: LdsRegion<NrRegion, Free>,
) -> Result<LdsRegion<NrRegion, Free>, String> {
    prelude(wg.isa(), g, e)?;
    // Four 64-row passes stage the tile's rows `[token][row]`; waves 0..3 run one serial
    // sum-of-squares chain per token.
    for p in 0..4u32 {
        let staged = wg.begin_write(region);
        let cond = wave_is(wg, g, 2, "s_cmp_eq_u32", p)?;
        let staged = wg.skip_unless(
            cond,
            &format!(".Lmq6i8_nr_stage{p}"),
            staged,
            |w, mut st| {
                for nb in 0..4usize {
                    for rg in 0..2usize {
                        for c in 0..2usize {
                            let imm = nb as u32 * 16 * STAGE_PITCH + rg as u32 * 64 + c as u32 * 16;
                            st = w.ds_store(st, ds_store_b128(WR, acc(g, nb, rg, c), imm))?;
                        }
                    }
                }
                Ok(st)
            },
        )?;
        let drained = wg.wait(staged.1)?;
        let (published,) = wg.barrier((ready(staged.0, drained),))?;
        let cond = wave_is(wg, g, 0, "s_cmp_lt_u32", 4)?;
        wg.skip_unless(cond, &format!(".Lmq6i8_nr_chain{p}"), (), |w, ()| {
            // Four batches of four b128 loads through two 16-register halves of the pool: batch
            // `k + 2` is issued once batch `k` has been consumed.
            chain_loads(w, &published, 0)?;
            chain_loads(w, &published, 1)?;
            for k in 0..4usize {
                let b = w.isa();
                for x in POOL + 16 * (k % 2) as u8..POOL + 16 * (k % 2) as u8 + 16 {
                    vop(
                        b,
                        format!("v_fmac_f32_e32 v{SUM}, v{x}, v{x}"),
                        &[SUM],
                        &[v(SUM), v(x)],
                    )?;
                }
                if k + 2 < 4 {
                    chain_loads(w, &published, k + 2)?;
                }
            }
            Ok(())
        })?;
        let (freed,) = wg.barrier((retire(published),))?;
        region = freed;
    }

    // inv = rsq(sum * 2^-8 + 1e-6) per token, published to every wave.
    let staged = wg.begin_write(region);
    let cond = wave_is(wg, g, 0, "s_cmp_lt_u32", 4)?;
    let staged = wg.skip_unless(cond, ".Lmq6i8_nr_inv", staged, |w, st| {
        let b = w.isa();
        vop(
            b,
            format!("v_mul_f32_e32 v{MT}, 0x3b800000, v{SUM}"),
            &[MT],
            &[v(SUM)],
        )?;
        vop(
            b,
            format!("v_add_f32_e32 v{MT}, 0x358637bd, v{MT}"),
            &[MT],
            &[v(MT)],
        )?;
        vop(
            b,
            format!("v_rsq_f32_e32 v{INVC}, v{MT}"),
            &[INVC],
            &[v(MT)],
        )?;
        w.ds_store(st, ds_store_b32(TCI, INVC))
    })?;
    let drained = wg.wait(staged.1)?;
    let (published,) = wg.barrier((ready(staged.0, drained),))?;
    for nb in 0..4u8 {
        wg.ds_load(&published, ds_load_b32(INV + nb, INVA, 64 * u32::from(nb)))?;
    }
    let (freed,) = wg.barrier((retire(published),))?;
    region = freed;

    expand_weights(wg.isa())?;
    // s = (1 + w) * (x * inv).
    {
        let b = wg.isa();
        for nb in 0..4usize {
            for rg in 0..2usize {
                for c in 0..2usize {
                    for i in 0..4usize {
                        let r = acc(g, nb, rg, c) + i as u8;
                        let w1 = W1 + (rg * 8 + 4 * c + i) as u8;
                        let inv = INV + nb as u8;
                        vop(
                            b,
                            format!("v_mul_f32_e32 v{r}, v{inv}, v{r}"),
                            &[r],
                            &[v(inv), v(r)],
                        )?;
                        vop(
                            b,
                            format!("v_mul_f32_e32 v{r}, v{w1}, v{r}"),
                            &[r],
                            &[v(w1), v(r)],
                        )?;
                    }
                }
            }
        }
    }

    // RoPE of rows 0..63: waves 0..3 swap their scaled values with the partner row's wave.
    let staged = wg.begin_write(region);
    let cond = wave_is(wg, g, 0, "s_cmp_lt_u32", 4)?;
    let staged = wg.skip_unless(cond, ".Lmq6i8_nr_swap", staged, |w, mut st| {
        for q in 0..16u32 {
            let (nb, rg, c) = ((q / 4) as usize, ((q / 2) % 2) as usize, (q % 2) as usize);
            st = w.ds_store(st, ds_store_b128(SW, acc(g, nb, rg, c), 16 * q))?;
        }
        Ok(st)
    })?;
    let drained = wg.wait(staged.1)?;
    let (published,) = wg.barrier((ready(staged.0, drained),))?;
    let cond = wave_is(wg, g, 0, "s_cmp_lt_u32", 4)?;
    wg.skip_unless(cond, ".Lmq6i8_nr_rope", (), |w, ()| {
        let rsrd = g.srd_a[0];
        for nb in 0..4usize {
            let cs = POOL + 32 * ((nb + 1) % 2) as u8;
            if nb + 1 < 4 {
                table_loads(w.isa(), nb + 1, 32 * (nb % 2) as u8, rsrd)?;
            }
            for rg in 0..2usize {
                for c in 0..2usize {
                    let q = ((nb * 2 + rg) * 2 + c) as u32;
                    w.ds_load(
                        &published,
                        ds_load_b128(PART + 4 * (rg * 2 + c) as u8, SR, 16 * q),
                    )?;
                }
            }
            let b = w.isa();
            let mut k = 0u8;
            for rg in 0..2usize {
                for c in 0..2usize {
                    for i in 0..4usize {
                        let a = acc(g, nb, rg, c) + i as u8;
                        let slot = (rg * 2 + c) as u8 * 4 + i as u8;
                        let (p, cos, sin) = (PART + slot, cs + slot, cs + 16 + slot);
                        let sx = SX + (k % 4);
                        k += 1;
                        let sign = g.tmp + SIGN;
                        vop(
                            b,
                            format!("v_xor_b32_e32 v{sx}, s{sign}, v{sin}"),
                            &[sx],
                            &[s(sign), v(sin)],
                        )?;
                        vop(
                            b,
                            format!("v_mul_f32_e32 v{sx}, v{p}, v{sx}"),
                            &[sx],
                            &[v(p), v(sx)],
                        )?;
                        vop(
                            b,
                            format!("v_fma_f32 v{a}, v{a}, v{cos}, v{sx}"),
                            &[a],
                            &[v(a), v(cos), v(sx)],
                        )?;
                    }
                }
            }
        }
        Ok(())
    })?;
    let (freed,) = wg.barrier((retire(published),))?;
    Ok(freed)
}

/// The cosines and sines of token block `nb`'s eight row quads into the 32-register buffer at
/// `buf`: cosines at `buf + 4 * (rg * 2 + c)`, sines 16 registers on.
fn table_loads(b: &mut Builder, nb: usize, buf: u8, rsrd: u8) -> Result<(), String> {
    for rg in 0..2usize {
        for c in 0..2usize {
            let q = (rg * 2 + c) as u8;
            let imm = nb as u32 * 4096 + rg as u32 * 64 + c as u32 * 16;
            bload(b, 4, buf + 4 * q, TV, rsrd, imm)?;
            bload(b, 4, buf + 16 + 4 * q, TV, rsrd, imm + 128)?;
        }
    }
    Ok(())
}

/// Kernel pointers to descriptors, the lane's addresses and `1 + bf16(norm)` of its 16 rows.
fn prelude(b: &mut Builder, g: &Gen, e: &epilogue::Epilogue) -> Result<(), String> {
    let (nd, rd) = (g.srd_w, g.srd_a[0]);
    let (t0, t1) = (g.tmp + T0, g.tmp + SIGN);
    let (tok, row) = (e.tok, e.row);
    smem(b, nd, 2, g.karg, spec::NR_NORM_OFFSET)?;
    smem(b, rd, 2, g.karg, spec::NR_ROPE_OFFSET)?;
    // Norm weights: 256 bf16 (512 records), OOB reads 0.
    sop(
        b,
        format!("s_and_b32 s{0}, s{0}, 0xffff", nd + 1),
        &[nd + 1],
        &[nd + 1],
    )?;
    sop(b, format!("s_mov_b32 s{}, 0x200", nd + 2), &[nd + 2], &[])?;
    sop(
        b,
        format!("s_mov_b32 s{}, {}", nd + 3, lit(SRD_WORD3)),
        &[nd + 3],
        &[],
    )?;
    // RoPE table: this tile's first token row on, `(N - bs) * 256` records (tokens past N read 0).
    sop(b, format!("s_lshl_b32 s{t0}, s{}, 8", g.bs), &[t0], &[g.bs])?;
    sop(
        b,
        format!("s_add_co_u32 s{rd}, s{rd}, s{t0}"),
        &[rd],
        &[rd, t0],
    )?;
    sop(
        b,
        format!("s_add_co_ci_u32 s{0}, s{0}, 0", rd + 1),
        &[rd + 1],
        &[rd + 1],
    )?;
    sop(
        b,
        format!("s_and_b32 s{0}, s{0}, 0xffff", rd + 1),
        &[rd + 1],
        &[rd + 1],
    )?;
    sop(
        b,
        format!("s_sub_co_i32 s{t0}, s{}, s{}", g.args.n, g.bs),
        &[t0],
        &[g.args.n, g.bs],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{}, s{t0}, 8", rd + 2),
        &[rd + 2],
        &[t0],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, {}", rd + 3, lit(SRD_WORD3)),
        &[rd + 3],
        &[],
    )?;
    // RoPE sign of the partner product: negative for the low half (wave pair 0), i.e. the sign
    // bit when `(wave >> 1) & 1 == 0`.
    sop(
        b,
        format!("s_lshr_b32 s{t0}, s{}, 1", g.wave),
        &[t0],
        &[g.wave],
    )?;
    sop(b, format!("s_and_b32 s{t0}, s{t0}, 1"), &[t0], &[t0])?;
    sop(b, format!("s_xor_b32 s{t0}, s{t0}, 1"), &[t0], &[t0])?;
    sop(b, format!("s_lshl_b32 s{t1}, s{t0}, 31"), &[t1], &[t0])?;

    // Norm weights of rows `row + 16 rg + 0..7` (16 bytes per row group).
    vop(
        b,
        format!("v_mbcnt_lo_u32_b32 v{LANE}, -1, 0"),
        &[LANE],
        &[],
    )?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{NV}, 1, v{row}"),
        &[NV],
        &[v(row)],
    )?;
    bload(b, 4, RAW, NV, nd, 0)?;
    bload(b, 4, RAW + 4, NV, nd, 32)?;
    // Staging store base: token * 272 + (row & 63) * 4.
    vop(
        b,
        format!("v_mul_u32_u24_e32 v{WR}, {}, v{tok}", lit(STAGE_PITCH)),
        &[WR],
        &[v(tok)],
    )?;
    vop(
        b,
        format!("v_and_b32_e32 v{T_SCRATCH}, 63, v{row}"),
        &[T_SCRATCH],
        &[v(row)],
    )?;
    vop(
        b,
        format!("v_lshl_add_u32 v{WR}, v{T_SCRATCH}, 2, v{WR}"),
        &[WR],
        &[v(T_SCRATCH), v(WR)],
    )?;
    // Chain token of waves 0..3: wave * 32 + lane (read base, inverse-RMS store address,
    // RoPE exchange slot of the wave and of its partner `wave ^ 2`).
    sop(
        b,
        format!("s_lshl_b32 s{t0}, s{}, 5", g.wave),
        &[t0],
        &[g.wave],
    )?;
    vop(
        b,
        format!("v_add_nc_u32_e32 v{TC}, s{t0}, v{LANE}"),
        &[TC],
        &[s(t0), v(LANE)],
    )?;
    vop(
        b,
        format!("v_mul_u32_u24_e32 v{RD}, {}, v{TC}", lit(STAGE_PITCH)),
        &[RD],
        &[v(TC)],
    )?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{TCI}, 2, v{TC}"),
        &[TCI],
        &[v(TC)],
    )?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{SW}, 8, v{TC}"),
        &[SW],
        &[v(TC)],
    )?;
    sop(
        b,
        format!("s_xor_b32 s{t0}, s{}, 2", g.wave),
        &[t0],
        &[g.wave],
    )?;
    sop(b, format!("s_lshl_b32 s{t0}, s{t0}, 5"), &[t0], &[t0])?;
    vop(
        b,
        format!("v_add_nc_u32_e32 v{SR}, s{t0}, v{LANE}"),
        &[SR],
        &[s(t0), v(LANE)],
    )?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{SR}, 8, v{SR}"),
        &[SR],
        &[v(SR)],
    )?;
    vop(b, format!("v_mov_b32_e32 v{SUM}, 0"), &[SUM], &[])?;
    // Table byte offset: token * 256 + (row & 8) * 4 (the lane's eight bases).
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{TV}, 8, v{tok}"),
        &[TV],
        &[v(tok)],
    )?;
    vop(
        b,
        format!("v_and_b32_e32 v{T_SCRATCH}, 8, v{row}"),
        &[T_SCRATCH],
        &[v(row)],
    )?;
    vop(
        b,
        format!("v_lshl_add_u32 v{TV}, v{T_SCRATCH}, 2, v{TV}"),
        &[TV],
        &[v(T_SCRATCH), v(TV)],
    )?;
    vop(
        b,
        format!("v_lshlrev_b32_e32 v{INVA}, 2, v{tok}"),
        &[INVA],
        &[v(tok)],
    )?;
    // The first token block's RoPE table (waves 0..3 consume it; it lands during the chain).
    table_loads(b, 0, 32, rd)?;
    Ok(())
}

/// Batch `k` (four registers' worth: pool offsets `16 * (k % 2) + 4 * j`) of the chain's 16
/// staged b128 reads of this lane's token row.
fn chain_loads(
    w: &mut Wv,
    published: &LdsRegion<NrRegion, Published>,
    k: usize,
) -> Result<(), String> {
    for j in 0..4usize {
        let i = 4 * k + j;
        w.ds_load(
            published,
            ds_load_b128(POOL + 16 * (k % 2) as u8 + 4 * j as u8, RD, 16 * i as u32),
        )?;
    }
    Ok(())
}

/// `1 + bf16(w)` of the lane's 16 rows: the low half of a dword is `w << 16`, the high half
/// `w & 0xffff0000`.
fn expand_weights(b: &mut Builder) -> Result<(), String> {
    for rg in 0..2u8 {
        for d in 0..4u8 {
            let raw = RAW + 4 * rg + d;
            let lo = W1 + 8 * rg + 2 * d;
            vop(
                b,
                format!("v_lshlrev_b32_e32 v{lo}, 16, v{raw}"),
                &[lo],
                &[v(raw)],
            )?;
            vop(
                b,
                format!("v_and_b32_e32 v{}, 0xffff0000, v{raw}", lo + 1),
                &[lo + 1],
                &[v(raw)],
            )?;
            for r in [lo, lo + 1] {
                vop(b, format!("v_add_f32_e32 v{r}, 1.0, v{r}"), &[r], &[v(r)])?;
            }
        }
    }
    Ok(())
}
