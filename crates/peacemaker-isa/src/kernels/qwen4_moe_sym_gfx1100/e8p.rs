// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1100 E8P-I4 (`Codec::E8p`) NT1 entries of the grouped symmetric IU4 MoE GEMMs: gate/up+SwiGLU
//! `qwen4_moe_gate_up_silu_iu4_sym_e8p_pm_gfx1100` (R256, K2560) and down
//! `qwen4_moe_down_iu4_sym_e8p_pm_gfx1100` (R128, K640), each with its `_acc` twin.
//!
//! Port of the reviewed gfx1201 K-E1 (`kernels/qwen4_moe_sym_e8p.rs`, K-E0 twin
//! `hipfire-runtime/src/e8p_i4_reference.rs`) onto the frozen gfx1100 NT1 skeleton of this module
//! (64-byte ABI, grid `(M/64, m_total/16)`, wave32, one K128 epoch per loop trip, native
//! workgroup ids, `(lr & 14) | hi` scale rows, `ds_swizzle` broadcast, the shared BF16 / `_acc`
//! epilogues). **Only the weight side and the fold differ from W2**; this is not an unpack-only
//! codec.
//!
//! # Wire layout
//!
//! `expert_weight_ptrs[e]` is the planar tensor `[M, K]`: `M*K/8` little-endian `u16` words
//! (row-major; row bytes `K/4`, ROWB), then `M*K/128` FP16 scales (SROW `K/64` bytes per row, plane
//! at `M*ROWB`): the W2 geometry. A word covers eight weights: `idx | sign7 << 8 | shift << 15`
//! (`idx` the abs-grid row, bit `k` of `sign7` negates lane `k < 7`, lane 7's sign follows from the row
//! parity, `shift` set = the QuIP# `-1/4`). One K128 epoch of a row is 16 words = 32 bytes at
//! `32 e` (two `b128` at `+0`, `+16`); the scale of epoch `e` is the `u16` at `2 e` of the scale
//! row. Activations are the A4 `block_i4_128` `[f32 d, i32 s, 64 nibble bytes]` of the other
//! entries: `d` at `+0` and **`s = sum(q)` at `+4`** (loaded with `d` as one `b64`), nibbles at `+8..+72`.
//!
//! # K assignment
//!
//! gfx11 `v_wmma_i32_16x16x16_iu4` lanes `l` and `l + 16` carry the same K16 operands, so (as in W2/W3)
//! every lane loads the whole epoch of its row. Raw dword `i` (words `2i`, `2i+1`) of the 8 raw dwords is
//! exactly K16 step `i`: word `2i` is operand dword `2i` (K elements `16i..16i+8`), word `2i+1` is
//! operand dword `2i+1`, matched by X dwords `2i`, `2i+1`.
//!
//! # LDS stage (1,536 B, once per CTA, [`STAGE_BYTES`])
//!
//! `LUT[256]` (offset 0, one u32 of eight nibbles `2|g|` per abs row, lane 7 stored two's-complement
//! negated on odd-parity rows) and `MASK[128]` (offset 1,024, `0xF` in the lanes `sign7` negates, lane 7 =
//! parity of `popcount(sign7)`). One typed `Tables` region is written by every wave, drained, published
//! at ONE workgroup barrier and then only read. MASK is computed arithmetically in every wave
//! (entries `tid + 32*waves*j`); the 8 LUT chunks of 32 dwords are split between the waves by wave-uniform
//! `skip_unless` blocks (wave `k` writes chunks `c % waves == k`) from SGPR literals combined into one VGPR with
//! `v_writelane_b32`. The stage runs after the CTA-uniform early exits of the prologue and before any
//! wave divergence; store addresses derive from the work-item id in the entry block (`v14 = 4*tid`,
//! `v15 = 4*lane`); table reads use `v_and_b32` immediate-masked addresses (`0x3fc`, `0x1fc`).
//!
//! # Unpack and the `2C + s` fold
//!
//! Per word `w` (lo word of the raw dword, then hi word): LUT address `(w << 2) & 0x3fc` /
//! `(w >> 14) & 0x3fc`, MASK address `(w >> 6) & 0x1fc` / `(w >> 22) & 0x1fc`, two `ds_load_b32`, then
//! `x = lut ^ mask`, `y = mask & 0x11111111`, `n = shift ? 0xeeeeeeef : 0`, operand `= x + y + n`
//! (`v_add3_u32`): `o = (v - 1) / 2` per nibble, no carry across nibbles ([`unpack_emitted_model`]).
//! `v = 2o + 1` so `sum(v q) = 2C + s`. The WMMA chain is seeded with MAGIC; after the epoch each
//! accumulator holds `a = MAGIC + C` and is folded as
//! `a' = v_lshl_add_u32(a, 1, s - MAGIC) = MAGIC + 2C + s`, `t = as_f32(a') + (-12582912.0)`,
//! `t *= 0.25` (exact) and `sum = fma(RN(d_act * f32(sc)), t, sum)` ([`epoch_model`]), ascending
//! epochs from `+0`. The existing D-only `iu4_fold::fold_pass` is *not* used: it has no `s`, no `x2` and
//! no `x0.25`. gfx11 VOPD cannot share a `src1` between the two halves, so the product pass pairs
//! `t_j = d * sc_j` (`src0 = d`) with the MAGIC add exactly like `fold_pass`.
//!
//! # Registers
//!
//! Over the MQ4/W2 layout (`SET0 = 92`): raw code dwords `92 + 8t + i`, X `x()` (16), scale words `x()+16+t`,
//! `d` at the next multiple of four (`s = d + 1`), A operands `a_out(t, i) = AOUT + 16 t + 2 i` (all eight
//! K16 steps live, no WAR reuse inside an epoch), unpack pool of five registers per word at `POOL`
//! (4 words GateUp, 2 Down). GateUp 184 VGPRs, Down 152. SGPR: eight literal staging registers at
//! `LITS = 88` (gfx1100: the dead padded-user SGPRs `s8..s13` and `s84,s85`, because the kernarg block takes `s88:95`; see `Gen::lit`).
use super::*;
use peacemaker_author::{ready, Free, LdsRegion, Published, Wave};
use peacemaker_isa::kernels::qwen4_moe_sym::e8p as twin;
use peacemaker_isa::vopd::{Operand, VopdF32, VopdOp};

/// `LUT[256]` bytes.
pub const LUT_BYTES: u32 = 256 * 4;
/// `MASK[128]` bytes (at `LUT_BYTES`).
pub const MASK_BYTES: u32 = 128 * 4;
/// Static LDS of every E8P entry: the per-CTA table stage.
pub const STAGE_BYTES: u32 = LUT_BYTES + MASK_BYTES;
/// Nibble ones: the `+1` of a two's-complement negation and the shift unit.
pub const ONES: u32 = 0x1111_1111;
/// `-ONES mod 2^32`: the shift subtraction as an addend.
pub const NEG_ONES: u32 = 0xeeee_eeef;
/// `-MAGIC mod 2^32`.
pub const MINUS_MAGIC_I32: u32 = 0xb4c0_0000;
/// `0.25f`.
pub const QUARTER: u32 = 0x3e80_0000;

/// The staged table region (one typed LDS region, LUT then MASK).
pub(super) enum Tables {}

/// The 256-word LUT as staged: the K-E0 table, or for [`Negative::WrongLut`] entry `i` holds row `i ^ 1`.
pub fn lut_table(neg: Negative) -> [u32; 256] {
    let lut = twin::build_lut256u32();
    if neg != Negative::WrongLut {
        return lut;
    }
    let mut w = lut;
    for (i, e) in w.iter_mut().enumerate() {
        *e = lut[i ^ 1];
    }
    w
}

/// The 128-word MASK as staged: the K-E0 table, or for [`Negative::WrongSignMask`] entry `s` holds `s ^ 1`.
pub fn mask_table(neg: Negative) -> [u32; 128] {
    let mask = twin::build_mask128u32();
    if neg != Negative::WrongSignMask {
        return mask;
    }
    let mut w = mask;
    for (s, e) in w.iter_mut().enumerate() {
        *e = mask[s ^ 1];
    }
    w
}

/// The emitted per-word network on one dword of two wire words (`hi` selects the high word): CPU
/// mirror of [`unpack_step`], over the tables as staged for `neg`.
pub fn unpack_emitted_model(neg: Negative, dword: u32, hi: bool) -> u32 {
    twin::unpack_emitted(&lut_table(neg), &mask_table(neg), dword, hi)
}

/// One epoch's accumulation as emitted: `a = MAGIC + C`, `a' = 2a + (s - MAGIC)`,
/// `t = as_f32(a') + (-12582912.0)`, `t * 0.25`, `sum = fma(RN(d_act * d_w), t, sum)`.
pub fn epoch_model(d_act: f32, d_w: f32, c: i32, s: i32, sum: f32) -> f32 {
    twin::epoch_model(d_act, d_w, c, s, sum)
}

/// [`epoch_model`] of the entry built for `neg`: [`Negative::WrongFoldEpoch`] replaces `s - MAGIC` by `-MAGIC`
/// (the activation sum is dropped), so the term is `2C/4` instead of `(2C + s)/4`.
pub fn epoch_model_for(neg: Negative, d_act: f32, d_w: f32, c: i32, s: i32, sum: f32) -> f32 {
    epoch_model(
        d_act,
        d_w,
        c,
        if neg == Negative::WrongFoldEpoch {
            0
        } else {
            s
        },
        sum,
    )
}

/// LUT chunks (32 dwords each, 8 in all) written by wave `wave` of `waves`.
pub fn stage_chunks(waves: u32, wave: u32) -> Vec<u32> {
    twin::stage_chunks(waves, wave)
}

// Prologue temporaries the stage owns (all inside the `prologue_tmp` range, dead after `k_begin`).
/// `4 * tid` (MASK stores) and `4 * lane` (LUT chunk stores), computed in the entry block.
pub(super) const LDS_AM: u8 = 14;
pub(super) const LDS_AL: u8 = 15;
/// Staging scratch: two MASK register sets (`e, parity, x, y`), four LUT chunk registers. `v2` (the `_acc`
/// lane id) is never touched.
const STG_MASK: [u8; 2] = [5, 9];
const STG_CHUNK: [u8; 4] = [1, 3, 4, 13];
/// Eight literal staging SGPRs.
pub(super) const LITS: u8 = 88;
/// gfx1100 literal staging SGPRs: `FeatureUserSGPRInit16Bug` pads the user SGPRs, so `s8..s13` are dead (the kernarg
/// block is at `s88:95`, see `Gen::args0`); two more from the unused `s83..s87` (an aligned pair). Eight in all, all below `s102`.
const LITS_PADDED: [u8; 8] = [8, 9, 10, 11, 12, 13, 84, 85];
const UNP_LEN: u8 = 5;

impl Gen {
    /// Literal staging SGPR `q` (0..8).
    pub(super) fn lit(&self, q: u8) -> u8 {
        if self.args0() == ARGS0 {
            LITS + q
        } else {
            LITS_PADDED[usize::from(q)]
        }
    }
    /// Weight dword `i` (words `2i`, `2i+1`) of tensor `t` of the epoch.
    pub(super) fn raw(&self, t: u8, i: u8) -> u8 {
        self.a(t) + i
    }
    /// First A-operand register (after `d`, `s`).
    pub(super) fn aout(&self) -> u8 {
        (self.d() + 2).div_ceil(4) * 4
    }
    /// A operand pair of K16 step `i` of tensor `t`.
    pub(super) fn a_out(&self, t: u8, i: u8) -> u8 {
        self.aout() + 16 * t + 2 * i
    }
    pub(super) fn pool(&self) -> u8 {
        self.aout() + 16 * self.tn()
    }
    /// Unpack pool register `k` of word `n` (`n = 2 t + which`): 0 LUT address, 1 MASK address, 2 LUT
    /// value, 3 MASK value, 4 shift addend.
    pub(super) fn unp(&self, n: u8, k: u8) -> u8 {
        self.pool() + UNP_LEN * n + k
    }
    /// Registered pool words: two per tensor.
    pub(super) fn unp_words(&self) -> u8 {
        2 * self.tn()
    }
}

/// Register the E8P operand set (mirrors the MQ4/W2 block of `Gen::plan`).
pub(super) fn plan_operands(
    g: &Gen,
    p: &mut RegPlan,
    kernel: impl Fn() -> Live,
) -> Result<(), String> {
    let tn = g.tn();
    for t in 0..tn {
        for h in 0..2u8 {
            p.v::<4>("code_words", g.a(t) + 4 * h, kernel())?;
        }
    }
    for q in 0..4u8 {
        p.v::<4>("x_nibbles", g.x() + 4 * q, kernel())?;
    }
    for t in 0..tn {
        p.v::<1>("scale_dword", g.hs(t), kernel())?;
    }
    p.v::<2>("act_d_s", g.d(), kernel())?;
    for t in 0..tn {
        for i in 0..8u8 {
            p.v::<2>("a_nibbles", g.a_out(t, i), kernel())?;
        }
    }
    for n in 0..g.unp_words() {
        for k in 0..UNP_LEN {
            p.v::<1>("unpack_tmp", g.unp(n, k), kernel())?;
        }
    }
    let live = || Live::Between("entry".into(), g.label("k_begin"));
    if g.args0() == ARGS0 {
        p.s::<8>("table_literals", LITS, live())?;
    } else {
        p.s::<4>("table_literals", LITS_PADDED[0], live())?;
        p.s::<2>("table_literals", LITS_PADDED[4], live())?;
        p.s::<2>("table_literals", LITS_PADDED[6], live())?;
    }
    Ok(())
}

fn ds_store_b32(addr: u8, data: u8, off: u32) -> Instruction {
    let o = if off == 0 {
        String::new()
    } else {
        format!(" offset:{off}")
    };
    Instruction::new(
        format!("ds_store_b32 v{addr}, v{data}{o}"),
        vec![],
        vec![v(addr), v(data)],
    )
    .memory(MemoryClass::DsStore)
}

fn ds_load_b32(dst: u8, addr: u8, off: u32) -> Instruction {
    let o = if off == 0 {
        String::new()
    } else {
        format!(" offset:{off}")
    };
    Instruction::new(
        format!("ds_load_b32 v{dst}, v{addr}{o}"),
        vec![v(dst)],
        vec![v(addr)],
    )
    .memory(MemoryClass::DsLoad)
}

/// Entry-block LDS store addresses: `4 * tid` and `4 * lane` (`v2` holds `tid & 31`).
pub(super) fn stage_addresses(b: &mut Builder) -> Result<(), String> {
    op(
        b,
        format!("v_lshlrev_b32_e32 v{LDS_AM}, 2, v0"),
        &[v(LDS_AM)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{LDS_AL}, 2, v2"),
        &[v(LDS_AL)],
        &[v(2)],
    )
}

/// Stage `LUT` and `MASK` once per CTA: every wave writes its share of the typed region, drains, and the
/// single workgroup barrier publishes it. All waves take the same path to the barrier (the early exits are
/// workgroup-uniform; the per-wave LUT blocks are `skip_unless` joins before it).
pub(super) fn stage<T: Gfx11>(
    wg: &mut Wg<'_, T>,
    g: &Gen,
    tables: LdsRegion<Tables, Free>,
) -> Result<LdsRegion<Tables, Published>, String> {
    let waves = g.spec.threads() / 32;
    let threads = 32 * waves;
    let neg = g.spec.negative;
    let lut = lut_table(neg);
    let mut out = wg.begin_write(tables);
    // MASK[tid + threads*j] arithmetically, in every wave (the unconditional store that puts the region
    // into the writing state on every path).
    for j in 0..128 / threads {
        let base = STG_MASK[(j % 2) as usize];
        let [e, pc, x, y] = [base, base + 1, base + 2, base + 3];
        let b = wg.isa();
        let mut src = TID;
        if j != 0 {
            op(
                b,
                format!("v_add_nc_u32_e32 v{e}, {}, v{TID}", lit(threads * j)),
                &[v(e)],
                &[v(TID)],
            )?;
            src = e;
        }
        if neg == Negative::WrongSignMask {
            // MASK[s] := MASK[s ^ 1].
            op(
                b,
                format!("v_xor_b32_e32 v{e}, 1, v{src}"),
                &[v(e)],
                &[v(src)],
            )?;
            src = e;
        }
        // x = sign7 | parity(sign7) << 7; parity by xor-shift reduction (sign7 < 128).
        op(
            b,
            format!("v_lshrrev_b32_e32 v{x}, 4, v{src}"),
            &[v(x)],
            &[v(src)],
        )?;
        op(
            b,
            format!("v_xor_b32_e32 v{pc}, v{src}, v{x}"),
            &[v(pc)],
            &[v(src), v(x)],
        )?;
        for sh in [2u8, 1] {
            op(
                b,
                format!("v_lshrrev_b32_e32 v{x}, {sh}, v{pc}"),
                &[v(x)],
                &[v(pc)],
            )?;
            op(
                b,
                format!("v_xor_b32_e32 v{pc}, v{pc}, v{x}"),
                &[v(pc)],
                &[v(pc), v(x)],
            )?;
        }
        op(
            b,
            format!("v_lshlrev_b32_e32 v{pc}, 7, v{pc}"),
            &[v(pc)],
            &[v(pc)],
        )?;
        op(
            b,
            format!("v_bfi_b32 v{x}, 0x80, v{pc}, v{src}"),
            &[v(x)],
            &[v(pc), v(src)],
        )?;
        // Spread bit k to nibble k.
        for (shift, mask) in [(12u32, 0x000f_000fu32), (6, 0x0303_0303), (3, ONES)] {
            op(
                b,
                format!("v_lshl_or_b32 v{x}, v{x}, {shift}, v{x}"),
                &[v(x)],
                &[v(x)],
            )?;
            op(
                b,
                format!("v_and_b32_e32 v{x}, {}, v{x}", lit(mask)),
                &[v(x)],
                &[v(x)],
            )?;
        }
        // 0xF per set bit: (x << 4) - x.
        op(
            b,
            format!("v_lshlrev_b32_e32 v{y}, 4, v{x}"),
            &[v(y)],
            &[v(x)],
        )?;
        op(
            b,
            format!("v_sub_nc_u32_e32 v{y}, v{y}, v{x}"),
            &[v(y)],
            &[v(y), v(x)],
        )?;
        out = wg.ds_store(out, ds_store_b32(LDS_AM, y, LUT_BYTES + 4 * threads * j))?;
    }
    // LUT chunks: wave k writes chunks c % waves == k.
    for k in 0..waves {
        let cond = wg.scmp(Instruction::new(
            format!("s_cmp_eq_u32 s{WAVE}, {k}"),
            vec![],
            vec![s(WAVE)],
        ))?;
        out = wg.skip_unless(cond, &g.label(&format!("stage_w{k}")), out, |w, mut out| {
            for (n, c) in stage_chunks(waves, k).into_iter().enumerate() {
                let vc = STG_CHUNK[n % 4];
                let b = w.isa();
                for batch in 0..4u32 {
                    for q in 0..8u32 {
                        let r = g.lit(q as u8);
                        sop(
                            b,
                            format!(
                                "s_mov_b32 s{r}, {}",
                                lit(lut[(32 * c + 8 * batch + q) as usize])
                            ),
                            &[r],
                            &[],
                        )?;
                    }
                    for q in 0..8u32 {
                        let r = g.lit(q as u8);
                        op(
                            b,
                            format!("v_writelane_b32 v{vc}, s{r}, {}", 8 * batch + q),
                            &[v(vc)],
                            &[v(vc), s(r)],
                        )?;
                    }
                }
                out = w.ds_store(out, ds_store_b32(LDS_AL, vc, 128 * c))?;
            }
            Ok(out)
        })?;
    }
    let (region, pending) = out;
    let drained = wg.wait(pending)?;
    let (published,) = wg.barrier((ready(region, drained),))?;
    Ok(published)
}

/// Issue exactly epoch `EIDX`'s loads: scales and the first tensor's codes, the second tensor's codes
/// (GateUp), then the activation block (`d` and `s` as one `b64`, then the 64 nibble bytes).
fn load_epoch(b: &mut Builder, g: &Gen) -> Result<(), String> {
    b.clause(|b| {
        for t in 0..g.tn() {
            kloop::scale_load(b, g.hs(t), g.scale_srd(t))?;
        }
        kloop::weight_loads(b, g, 0)
    })?;
    if g.tn() == 2 {
        b.clause(|b| kloop::weight_loads(b, g, 1))?;
    }
    b.clause(|b| {
        for q in 0..4u8 {
            bload(b, 4, g.x() + 4 * q, XE, SRD_X, 8 + 16 * u32::from(q))?;
        }
        bload(b, 2, g.d(), XE, SRD_X, 0)
    })
}

/// Unpack step `i` (every tensor's raw dword `i`, two wire words each) into `a_out(t, i)`: the table
/// addresses and lookups of all words are issued first, then the per-word network runs on the landed values.
fn unpack_step<T: Gfx11>(
    w: &mut Wave<'_, T, Builder>,
    g: &Gen,
    i: u8,
    tables: &LdsRegion<Tables, Published>,
) -> Result<(), String> {
    let tn = g.tn();
    for t in 0..tn {
        for which in 0..2u8 {
            let n = 2 * t + which;
            let word = g.raw(t, i);
            let (la, ma, sh) = (g.unp(n, 0), g.unp(n, 1), g.unp(n, 4));
            let b = w.isa();
            if which == 0 {
                op(
                    b,
                    format!("v_lshlrev_b32_e32 v{la}, 2, v{word}"),
                    &[v(la)],
                    &[v(word)],
                )?;
                op(
                    b,
                    format!("v_and_b32_e32 v{la}, {}, v{la}", lit(0x3fc)),
                    &[v(la)],
                    &[v(la)],
                )?;
                op(
                    b,
                    format!("v_lshrrev_b32_e32 v{ma}, 6, v{word}"),
                    &[v(ma)],
                    &[v(word)],
                )?;
                op(
                    b,
                    format!("v_and_b32_e32 v{ma}, {}, v{ma}", lit(0x1fc)),
                    &[v(ma)],
                    &[v(ma)],
                )?;
            } else {
                op(
                    b,
                    format!("v_lshrrev_b32_e32 v{la}, 14, v{word}"),
                    &[v(la)],
                    &[v(word)],
                )?;
                op(
                    b,
                    format!("v_and_b32_e32 v{la}, {}, v{la}", lit(0x3fc)),
                    &[v(la)],
                    &[v(la)],
                )?;
                op(
                    b,
                    format!("v_lshrrev_b32_e32 v{ma}, 22, v{word}"),
                    &[v(ma)],
                    &[v(word)],
                )?;
                op(
                    b,
                    format!("v_and_b32_e32 v{ma}, {}, v{ma}", lit(0x1fc)),
                    &[v(ma)],
                    &[v(ma)],
                )?;
            }
            w.ds_load(tables, ds_load_b32(g.unp(n, 2), la, 0))?;
            w.ds_load(tables, ds_load_b32(g.unp(n, 3), ma, LUT_BYTES))?;
            // Shift addend `-ones` from the shift bit (independent of the lookups).
            let b = w.isa();
            if which == 0 {
                op(
                    b,
                    format!("v_bfe_i32 v{sh}, v{word}, 15, 1"),
                    &[v(sh)],
                    &[v(word)],
                )?;
            } else {
                op(
                    b,
                    format!("v_ashrrev_i32_e32 v{sh}, 31, v{word}"),
                    &[v(sh)],
                    &[v(word)],
                )?;
            }
            op(
                b,
                format!("v_and_b32_e32 v{sh}, {}, v{sh}", lit(NEG_ONES)),
                &[v(sh)],
                &[v(sh)],
            )?;
        }
    }
    let b = w.isa();
    for t in 0..tn {
        for which in 0..2u8 {
            let n = 2 * t + which;
            let (lv, mv, sh) = (g.unp(n, 2), g.unp(n, 3), g.unp(n, 4));
            op(
                b,
                format!("v_xor_b32_e32 v{lv}, v{mv}, v{lv}"),
                &[v(lv)],
                &[v(mv), v(lv)],
            )?;
            op(
                b,
                format!("v_and_b32_e32 v{mv}, {}, v{mv}", lit(ONES)),
                &[v(mv)],
                &[v(mv)],
            )?;
            let dst = g.a_out(t, i) + which;
            op(
                b,
                format!("v_add3_u32 v{dst}, v{lv}, v{mv}, v{sh}"),
                &[v(dst)],
                &[v(lv), v(mv), v(sh)],
            )?;
        }
    }
    Ok(())
}

/// One tensor's fold of an accumulator fragment `c` (`a = MAGIC + C`): `a' = 2a + (s - MAGIC)`
/// (`v_lshl_add_u32`), `t_j = d_act * sc_j` paired with `a' + (-12582912.0)` (the `iu4_fold::fold_pass`
/// product pass, gfx11-legal VOPD pairs), `* 0.25` (exact), `sum_j = fma(t_j, c_j, sum_j)`.
fn fold(b: &mut Builder, c: u8, sum: u8, sc: u8, d: u8, sm: u8, t: u8) -> Result<(), String> {
    let ts = |j: u8| t + (j ^ 2);
    for j in 0..8u8 {
        op(
            b,
            format!("v_lshl_add_u32 v{0}, v{0}, 1, v{sm}", c + j),
            &[v(c + j)],
            &[v(c + j), v(sm)],
        )?;
    }
    for j in 0..8u8 {
        let mul = VopdOp {
            op: VopdF32::Mul,
            dst: ts(j),
            src0: Operand::V(d),
            src1: sc + j,
        };
        let cf = c + (j ^ 1);
        let add = VopdOp {
            op: VopdF32::Add,
            dst: cf,
            src0: Operand::Lit(iu4_fold::MAGIC_NEG),
            src1: cf,
        };
        b.vopd(mul, add)?;
    }
    for j in (0..8u8).step_by(2) {
        let x = VopdOp {
            op: VopdF32::Mul,
            dst: c + j,
            src0: Operand::Lit(QUARTER),
            src1: c + j,
        };
        let y = VopdOp {
            op: VopdF32::Mul,
            dst: c + j + 1,
            src0: Operand::Lit(QUARTER),
            src1: c + j + 1,
        };
        b.vopd(x, y)?;
    }
    for j in (0..8u8).step_by(2) {
        let x = VopdOp {
            op: VopdF32::Fmac,
            dst: sum + j,
            src0: Operand::V(ts(j)),
            src1: c + j,
        };
        let y = VopdOp {
            op: VopdF32::Fmac,
            dst: sum + j + 1,
            src0: Operand::V(ts(j + 1)),
            src1: c + j + 1,
        };
        b.vopd(x, y)?;
    }
    Ok(())
}

/// Fold the epoch whose operands are in the registers: scale broadcast, eight unpack+WMMA steps, `s - MAGIC`,
/// the `2C + s` / quarter fold.
fn compute<T: Gfx11>(
    w: &mut Wave<'_, T, Builder>,
    g: &Gen,
    tables: &LdsRegion<Tables, Published>,
) -> Result<(), String> {
    let a = g.spec.arch;
    let tn = g.tn();
    for t in 0..tn {
        let b = w.isa();
        op(
            b,
            format!("v_cvt_f32_f16_e64 v{}, v{}.l", HF + t, g.hs(t)),
            &[v(HF + t)],
            &[v(g.hs(t))],
        )?;
        for j in 0..8u8 {
            let text = format!(
                "ds_swizzle_b32 v{}, v{} offset:swizzle(BROADCAST,16,{})",
                SCF + 8 * t + j,
                HF + t,
                swizzle_lane(u32::from(j))
            );
            b.ds_crosslane(
                Instruction::new(text, vec![v(SCF + 8 * t + j)], vec![v(HF + t)])
                    .memory(MemoryClass::DsLoad),
            )?;
        }
    }
    for i in 0..8u8 {
        unpack_step(w, g, i, tables)?;
        let b = w.isa();
        for t in 0..tn {
            iu4_fold::wmma_step(
                b,
                a,
                V::<8>(CACC + 8 * t),
                V::<2>(g.a_out(t, i)),
                V::<2>(g.x() + 2 * i),
                i == 0,
                V::<8>(MAGIC8),
            )?;
        }
    }
    // s - MAGIC, in place (once per epoch, shared by the tensors). `WrongFoldEpoch` drops `s`: `-MAGIC` alone
    // (same encoding length; the wait on the in-flight `s` load is still required by the overwrite).
    let b = w.isa();
    let (d, sm) = (g.d(), g.d() + 1);
    if g.spec.negative == Negative::WrongFoldEpoch {
        op(
            b,
            format!("v_mov_b32_e32 v{sm}, {}", lit(MINUS_MAGIC_I32)),
            &[v(sm)],
            &[],
        )?;
    } else {
        op(
            b,
            format!("v_add_nc_u32_e32 v{sm}, {}, v{sm}", lit(MINUS_MAGIC_I32)),
            &[v(sm)],
            &[v(sm)],
        )?;
    }
    for t in 0..tn {
        fold(b, CACC + 8 * t, SUM + 8 * t, SCF + 8 * t, d, sm, TPROD)?;
    }
    Ok(())
}

fn cmp<T: Gfx11>(wg: &mut Wg<'_, T>, text: String, uses: &[u8]) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(Instruction::new(
        text,
        vec![],
        uses.iter().map(|&n| s(n)).collect(),
    ))
}

/// The epoch loop of an E8P entry (one K128 epoch per trip, like the MQ4/W2/W3 loop, with the LDS tables
/// carried through the loop). Places `k_begin`; leaves the builder at the fall-through into the
/// caller's `epilogue` label with an empty wait ledger.
pub(super) fn run_loop<T: Gfx11>(
    wg: &mut Wg<'_, T>,
    g: &Gen,
    tables: LdsRegion<Tables, Published>,
) -> Result<(), String> {
    wg.label(&g.label("k_begin"))?;
    wg.isa().wait_all()?;
    let (head, done) = (g.label("k_loop"), g.label("k_loop_end"));
    let entry = wg.isa().ledger().shape();
    let none = cmp(wg, format!("s_cmp_eq_u32 s{ECNT}, 0"), &[ECNT])?;
    let _tables = wg.wg_skip_if(none, &done, tables, |wg, tables| {
        let tables = wg.loop_carried(&head, tables, |wg, tables| {
            let b = wg.isa();
            kloop::epoch_offsets(b, g)?;
            load_epoch(b, g)?;
            compute(wg, g, &tables)?;
            // Drain before the back edge: the next iteration (and the epilogue aliases) start from an empty ledger.
            let b = wg.isa();
            b.wait_all()?;
            if g.spec.negative == Negative::ReversedEpoch {
                sop(
                    b,
                    format!("s_add_i32 s{EIDX}, s{EIDX}, -1"),
                    &[EIDX],
                    &[EIDX],
                )?;
                Ok((
                    tables,
                    cmp(wg, format!("s_cmp_lg_u32 s{EIDX}, -1"), &[EIDX])?,
                ))
            } else {
                sop(
                    b,
                    format!("s_add_i32 s{EIDX}, s{EIDX}, 1"),
                    &[EIDX],
                    &[EIDX],
                )?;
                Ok((
                    tables,
                    cmp(wg, format!("s_cmp_lg_u32 s{EIDX}, s{ECNT}"), &[EIDX, ECNT])?,
                ))
            }
        })?;
        if wg.isa().ledger().shape() != entry {
            return Err("k loop exit ledger differs from its entry".into());
        }
        Ok(tables)
    })?;
    wg.isa().wait_all()
}

/// Result of [`prove`]: named check counts and every failure message.
#[derive(Debug, Default)]
pub struct Proof {
    /// `(check, number of cases)`.
    pub checks: Vec<(&'static str, u64)>,
    pub failures: Vec<String>,
}

/// CPU proofs (no device, no toolchain) of the arithmetic, the table stage, the operand/K assignment, the
/// address geometry and the register layout of the E8P entries; the negatives must be effective.
pub fn prove() -> Proof {
    let mut p = Proof::default();
    let fail = |p: &mut Proof, m: String| p.failures.push(m);
    let (lut, mask) = (twin::build_lut256u32(), twin::build_mask128u32());
    if lut_table(Negative::None) != lut || mask_table(Negative::None) != mask {
        fail(
            &mut p,
            "lut_table/mask_table(None) differ from the K-E0 tables".into(),
        );
    }

    // 1. Emitted per-word network == per-lane reference for every (idx, sign7, shift), both halves of the dword, junk elsewhere.
    let mut junk = 0x9e37_79b9u32;
    let mut n = 0u64;
    for idx in 0..=255u8 {
        for sign7 in 0..128u8 {
            for shift in [false, true] {
                let want = twin::unpack_word_ref(idx, sign7, shift);
                let word = u32::from(idx) | u32::from(sign7) << 8 | u32::from(shift) << 15;
                junk = junk.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                if unpack_emitted_model(Negative::None, word | (junk & 0xffff_0000), false) != want
                {
                    fail(
                        &mut p,
                        format!("lo word idx {idx} sign7 {sign7} shift {shift}"),
                    );
                }
                if unpack_emitted_model(Negative::None, (word << 16) | (junk >> 16), true) != want {
                    fail(
                        &mut p,
                        format!("hi word idx {idx} sign7 {sign7} shift {shift}"),
                    );
                }
                let v = twin::decode8(idx, sign7, shift);
                for k in 0..8 {
                    let o = (((want >> (4 * k)) & 15) as i32) << 28 >> 28;
                    if 2 * o + 1 != i32::from(v[k]) || !(-6..=5).contains(&o) {
                        fail(
                            &mut p,
                            format!(
                                "operand nibble idx {idx} sign7 {sign7} shift {shift} lane {k}"
                            ),
                        );
                    }
                }
                n += 1;
            }
        }
    }
    p.checks.push(("unpack_every_word_both_halves", n * 2));

    // 2. Negatives are effective: every wrong table entry differs from the true one.
    let (wl, wm) = (
        lut_table(Negative::WrongLut),
        mask_table(Negative::WrongSignMask),
    );
    if (0..256).any(|i| wl[i] == lut[i]) {
        fail(&mut p, "wrong-lut leaves a LUT entry unchanged".into());
    }
    if (0..128).any(|s| wm[s] == mask[s]) {
        fail(
            &mut p,
            "wrong-sign-mask leaves a MASK entry unchanged".into(),
        );
    }
    if lut_table(Negative::WrongSignMask) != lut || mask_table(Negative::WrongLut) != mask {
        fail(&mut p, "a negative changed the other table".into());
    }
    p.checks
        .push(("negative_tables_differ_everywhere", 256 + 128));

    // 3. Operand / K assignment: A dword 2i+which is word 2i+which of raw dword i, nibble k = K element 16i + 8*which + k.
    // For random words and activations, 2*D + s == sum(v * q) with D the K16-step chain of the unpacked operands.
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut rnd = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as u32
    };
    let cases = 4000u64;
    for _ in 0..cases {
        let mut words = [0u16; 16];
        for w in &mut words {
            *w = (rnd() & 0xffff) as u16;
        }
        let mut q = [0i32; 128];
        for x in &mut q {
            *x = (rnd() & 15) as i32 - 8;
        }
        let s: i32 = q.iter().sum();
        let mut d = 0i32;
        let mut want = 0i32;
        for i in 0..8usize {
            let raw = u32::from(words[2 * i]) | u32::from(words[2 * i + 1]) << 16;
            for which in 0..2usize {
                let a = unpack_emitted_model(Negative::None, raw, which == 1);
                let w = u32::from(words[2 * i + which]);
                let v = twin::decode8((w & 0xff) as u8, ((w >> 8) & 0x7f) as u8, w >> 15 & 1 == 1);
                for k in 0..8usize {
                    let o = (((a >> (4 * k)) & 15) as i32) << 28 >> 28;
                    let e = 16 * i + 8 * which + k;
                    d += o * q[e];
                    want += i32::from(v[k]) * q[e];
                }
            }
        }
        if 2 * d + s != want {
            fail(&mut p, format!("2C+s {} != sum(v q) {want}", 2 * d + s));
        }
        if d.abs() > 128 * 6 * 8 || d.abs() >= 1 << 22 {
            fail(&mut p, format!("|C| {d} beyond the dot bound"));
        }
    }
    p.checks.push(("two_c_plus_s_equals_sum_vq", cases));

    // 4. Fold: the emitted network (2a + (s - MAGIC), + (-MAGIC), * 0.25, fma) is exactly fma(RN(d_act d), (2C + s)/4, sum).
    let mut sum = 0.37f32;
    let (d_act, d_w) = (0.0123f32, 0.8125f32);
    let mut folds = 0u64;
    for c in -6144..=5120i32 {
        for s in [
            -1024i32, -513, -255, -64, -7, -1, 0, 1, 5, 64, 127, 511, 1023,
        ] {
            let want = (d_act * d_w).mul_add(((2 * c + s) as f32) * 0.25, sum);
            if epoch_model(d_act, d_w, c, s, sum).to_bits() != want.to_bits() {
                fail(&mut p, format!("fold c {c} s {s}"));
            }
            if epoch_model_for(Negative::None, d_act, d_w, c, s, sum).to_bits() != want.to_bits() {
                fail(&mut p, format!("fold(None) c {c} s {s}"));
            }
            // wrong-fold-epoch: exactly fma(RN(d_act d), 2C/4, sum); from a small fixed sum it differs from the true fold whenever s != 0.
            let broken = (d_act * d_w).mul_add((2 * c) as f32 * 0.25, sum);
            let got = epoch_model_for(Negative::WrongFoldEpoch, d_act, d_w, c, s, sum);
            if got.to_bits() != broken.to_bits() {
                fail(&mut p, format!("wrong-fold-epoch model c {c} s {s}"));
            }
            if s != 0
                && epoch_model_for(Negative::WrongFoldEpoch, d_act, d_w, c, s, 0.37).to_bits()
                    == epoch_model(d_act, d_w, c, s, 0.37).to_bits()
            {
                fail(
                    &mut p,
                    format!("wrong-fold-epoch leaves the fold unchanged c {c} s {s}"),
                );
            }
            sum = want;
            folds += 1;
        }
    }
    if MINUS_MAGIC_I32 != iu4_fold::MAGIC.wrapping_neg()
        || NEG_ONES != ONES.wrapping_neg()
        || f32::from_bits(QUARTER) != 0.25
    {
        fail(&mut p, "fold constants".into());
    }
    if dot_bound(Codec::E8p) != 6144 || 2 * 6144 + 1024 >= 1 << 22 {
        fail(&mut p, "dot bound".into());
    }
    p.checks.push(("fold_epoch_exact", folds));

    // 5. Stage coverage: the chunks partition the 8 LUT chunks, the MASK entries cover all 128, every store ends inside the stage.
    let mut stores = 0u64;
    for waves in [2u32, 4] {
        let mut seen = [0u8; 8];
        for wave in 0..waves {
            for c in stage_chunks(waves, wave) {
                seen[c as usize] += 1;
                if 4 * 31 + 128 * c + 4 > LUT_BYTES {
                    fail(&mut p, format!("LUT chunk {c} store out of the table"));
                }
                stores += 1;
            }
        }
        if seen != [1u8; 8] {
            fail(
                &mut p,
                format!("LUT chunks not partitioned for {waves} waves: {seen:?}"),
            );
        }
        let threads = 32 * waves;
        let mut entries = [0u8; 128];
        for j in 0..128 / threads {
            for tid in 0..threads {
                entries[(tid + threads * j) as usize] += 1;
                if 4 * tid + LUT_BYTES + 4 * threads * j + 4 > STAGE_BYTES {
                    fail(
                        &mut p,
                        format!("MASK store tid {tid} j {j} out of the stage"),
                    );
                }
                stores += 1;
            }
        }
        if entries != [1u8; 128] {
            fail(
                &mut p,
                format!("MASK entries not covered once for {waves} waves"),
            );
        }
    }
    if STAGE_BYTES != 1536 {
        fail(&mut p, "STAGE_BYTES".into());
    }
    p.checks.push(("stage_partition_and_bounds", stores));

    // 6. Address geometry (W2 plane geometry, 32 code bytes and one u16 scale per epoch).
    let mut addr = 0u64;
    for (kind, m, k) in [(Kind::GateUp, 2560u32, 2560u32), (Kind::Down, 2560, 640)] {
        let rowb = row_code_bytes(Codec::E8p, k);
        let srow = row_scale_bytes(Codec::E8p, k);
        if rowb != k / 4
            || srow != k / 64
            || rowb != row_code_bytes(Codec::W2, k)
            || tensor_bytes(Codec::E8p, m, k) != u64::from(m) * u64::from(k / 4 + k / 64)
        {
            fail(&mut p, format!("row geometry K{k}"));
        }
        let loads = code_loads(Codec::E8p);
        if loads != vec![(0, 4), (16, 4)] {
            fail(
                &mut p,
                "E8P epoch loads are not two b128 over 32 bytes".into(),
            );
        }
        for e in 0..k / 128 {
            let c = code_offset(Codec::E8p, kind, e);
            let last = loads
                .iter()
                .map(|&(o, w)| c + o + 4 * u32::from(w))
                .max()
                .unwrap_or(0);
            if c != 32 * e || last != 32 * (e + 1) || last > rowb {
                fail(&mut p, format!("code window of epoch {e} K{k}"));
            }
            let so = scale_offset(Codec::E8p, kind, e);
            if so != 2 * e || so + 2 > srow {
                fail(&mut p, format!("scale offset of epoch {e} K{k}"));
            }
            addr += 1;
        }
    }
    p.checks.push(("epoch_address_windows", addr));

    // 7. Register layout of every entry: disjoint operand ranges inside 256 VGPRs, `d` aligned, `s = d + 1`.
    let mut layouts = 0u64;
    for kind in [Kind::GateUp, Kind::Down] {
        for acc in [false, true] {
            let g = Gen {
                // The register plan never reads the arch: one layout serves both gfx11 targets.
                spec: Spec {
                    arch: Arch::Gfx1100,
                    codec: Codec::E8p,
                    kind,
                    acc,
                    negative: Negative::None,
                },
            };
            let tn = g.tn();
            let mut ranges: Vec<(&str, u8, u8)> = Vec::new();
            for t in 0..tn {
                ranges.push(("code", g.a(t), 8));
            }
            ranges.push(("x", g.x(), 16));
            for t in 0..tn {
                ranges.push(("scale", g.hs(t), 1));
            }
            ranges.push(("d_s", g.d(), 2));
            for t in 0..tn {
                ranges.push(("a_out", g.a_out(t, 0), 16));
            }
            ranges.push(("pool", g.pool(), 5 * g.unp_words()));
            ranges.sort_by_key(|r| r.1);
            for w in ranges.windows(2) {
                if u16::from(w[0].1) + u16::from(w[0].2) > u16::from(w[1].1) {
                    fail(
                        &mut p,
                        format!("{kind:?} register ranges overlap: {} / {}", w[0].0, w[1].0),
                    );
                }
            }
            let top = ranges
                .last()
                .map(|r| u16::from(r.1) + u16::from(r.2))
                .unwrap_or(0);
            if top > 256 || g.d() % 4 != 0 || g.aout() % 4 != 0 {
                fail(&mut p, format!("{kind:?} layout top {top} or alignment"));
            }
            layouts += 1;
        }
    }
    p.checks.push(("register_layouts", layouts));
    p
}
