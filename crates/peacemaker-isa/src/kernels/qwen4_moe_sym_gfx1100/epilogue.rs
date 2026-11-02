// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1100 K2 epilogue emitter (Consumer24 B4-A): GateUp and Down BF16 epilogues and the
//! independent F32 `_acc` store, for the grouped NT1 IU4 MoE kernels
//! `qwen4_moe_{gate_up_silu,down}_iu4_sym{,_w2,_w3}_pm_gfx1100`.
//!
//! Design authority: `consumer24/SOL-B4-B5.md` §2.1 (register contract, scratch
//! ranges), §2.3 (scalar math = the existing `Epilogue::silu_dense` DAG and the
//! `Bf16::rne_finite_passthrough` helper, both unmodified), §2.4 (the gfx11
//! half-wave exchange of `qwen4_moe_sym.rs:578-603`, and the `_acc` scalar F32
//! store that deliberately shares nothing with that pack/exchange).
//!
//! # API
//!
//! `Kind` is `peacemaker_isa::kernels::qwen4_moe_sym::Kind` (`GateUp | Down`).
//! Every emitter takes the PeaceMaker `Builder`; the caller has already placed the
//! `epilogue` label (this file never places labels).
//!
//! ```ignore
//! pub fn reserve_bf16(p: &mut RegPlan, kind: Kind, epilogue_label: &str, end_label: &str) -> Result<(), String>;
//! pub fn reserve_acc (p: &mut RegPlan, kind: Kind, epilogue_label: &str, end_label: &str) -> Result<(), String>;
//! pub fn emit_bf16(b: &mut Builder, kind: Kind, sel: &PackSelectors) -> Result<(), String>;
//! pub fn emit_acc (b: &mut Builder, kind: Kind, regs: &AccRegs) -> Result<(), String>;
//! // pieces of emit_bf16, in order: round/SiLU (math), LIVE, exchange (routing), store
//! pub fn emit_round_silu(b: &mut Builder, kind: Kind) -> Result<(), String>;
//! pub fn emit_apply_live(b: &mut Builder) -> Result<(), String>;
//! pub fn emit_exchange(b: &mut Builder, sel: &PackSelectors) -> Result<(), String>;
//! pub fn emit_store(b: &mut Builder) -> Result<(), String>;
//! PackSelectors::frozen()  // default: the frozen selectors
//! PackSelectors::wrong_pack_selector() // negative control: exactly one final v_perm selector wrong
//! AccRegs::default()       // m = s18, tile16 = s47, rbase = s49, lane = v2
//! ```
//!
//! # Register contract at the K-loop exit (SOL §2.1, frozen)
//!
//! Registered by the CALLER (all `Live::Whole`, they survive the epilogue):
//! `SUM` v48.. (v48..v55 gate/Down sums, v56..v63 GateUp up sums; `v48+j` is row
//! `rbase+2*j+hi`), `YOFF` v20 (BF16 byte offset `2*(p*row_len+rbase+8*hi)`),
//! `LIVE` s52:53 (identical in lanes `lr` and `lr+16`), Y pointer s16:17, and the
//! two rounding lane-mask pairs `MASKT` s60:61, s62:63.
//!
//! Registered by [`reserve_bf16`] / [`reserve_acc`] with the real epilogue lifetime
//! `Live::Between(epilogue_label, end_label)` (they alias dead K-loop registers):
//! `RT_TMP` v32-39, `OWN` v64-67, `SEND` v68-69, `RECV` v70-71, `SEL` v72-73,
//! `OUT` v76-79, `PACK` v80-83, GateUp only SiLU temporaries v92-115 (`SET0`, six per
//! element, 2 groups of 4) and SiLU lane masks s24-47 (`SILU_MASK`, 12 pairs),
//! `HIMASK` s54:55, `PSEL` s56, `SRD_Y` s64:67. `_acc` reserves `RT_TMP`, `PSEL`
//! (up-row byte offset scratch) and `SRD_Y` only.
//!
//! EXEC stays full through the whole epilogue (no emitter here touches EXEC):
//! padding is masked as VALUES (`v_cndmask` by LIVE) so a partner lane is never
//! deactivated before `v_permlanex16_b32`.
//!
//! # `_acc` (qualification only, never dispatched)
//!
//! Stores every F32 `SUM[t*8+j]` (padding slots +0) with a scalar `buffer_store_b32`
//! at `4*(p*M + rbase + 2*j + hi + t*M/2)` (GateUp: gate rows, then up rows `M/2`
//! further; Down: `t = 0`). It does NOT use `YOFF` or the pack/exchange: it
//! recomputes `p = tile16 + lr`, `lr = lane & 15`, `hi = lane >> 4` from
//! [`AccRegs`] (`m`: kernarg M, `tile16`: `16*tile`, `rbase`: row base of this wave, all
//! SGPRs; `lane`: a VGPR holding `tid & 31`). The caller registers those four
//! `Live::Whole` in the acc plan.

use peacemaker_isa::{
    insn::MemoryClass,
    kernels::{
        bf16::Bf16,
        common::{bstore_b128, lit, mem, op, s, sop, sr, srd_tail, v},
        gemm_uk::Epilogue,
        qwen4_moe_sym::Kind,
    },
    reg::Live,
    Builder, RegPlan,
};

/// Frozen register assignment (SOL §2.1), mirrored from the parent kernel's private constants.
pub mod regs {
    pub const YOFF: u8 = 20;
    pub const SUM: u8 = 48;
    pub const RT_TMP: u8 = 32;
    pub const OWN: u8 = 64;
    pub const SEND: u8 = 68;
    pub const RECV: u8 = 70;
    pub const SEL: u8 = 72;
    pub const OUT: u8 = 76;
    pub const PACK: u8 = 80;
    pub const SET0: u8 = 92;
    pub const SILU_N: u8 = 4;
    pub const SILU_MASK: u8 = 24;
    pub const Y_PTR: u8 = 16;
    pub const LIVE: u8 = 52;
    pub const HIMASK: u8 = 54;
    pub const PSEL: u8 = 56;
    pub const MASKT: [u8; 2] = [60, 62];
    pub const SRD_Y: u8 = 64;
}
use regs::*;

/// The byte selectors of the gfx11 half-wave exchange. `frozen()` is SOL §2.4.
/// `v_perm_b32 d, src0, src1, sel` reads the 8-byte string `{src0, src1}` with `src1` as bytes 0..3.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackSelectors {
    /// Pack of two high halves, `v_perm(H_hi, H_lo, send)`: `H_lo | H_hi << 16`.
    pub send: u32,
    /// Final `v_perm(RECV, OWN, sel)` selector, hi = 0, `[even k, odd k]`.
    pub recv_h0: [u32; 2],
    /// Same for hi = 1.
    pub recv_h1: [u32; 2],
    /// `v_permlanex16_b32` selector dwords (src1: lanes 0..7, src2 literal: lanes 8..15).
    pub permlane_lo: u32,
    pub permlane_hi: u32,
}
impl PackSelectors {
    pub const fn frozen() -> Self {
        Self {
            send: 0x0706_0302,
            recv_h0: [0x0504_0302, 0x0706_0302],
            recv_h1: [0x0302_0504, 0x0302_0706],
            permlane_lo: 0x7654_3210,
            permlane_hi: 0xfedc_ba98,
        }
    }
    /// Negative control: exactly ONE final `v_perm` selector is wrong (hi = 0, odd k reads the
    /// received low half instead of the received high half). The BF16 entry only; `_acc` is unaffected.
    pub const fn wrong_pack_selector() -> Self {
        let mut s = Self::frozen();
        s.recv_h0[1] = s.recv_h0[0];
        s
    }
}
impl Default for PackSelectors {
    fn default() -> Self {
        Self::frozen()
    }
}

/// Scalar inputs of the `_acc` address recomputation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AccRegs {
    /// SGPR holding kernarg `M`.
    pub m: u8,
    /// SGPR holding `16 * tile` (the first grouped row `p` of the tile).
    pub tile16: u8,
    /// SGPR holding this wave's row base.
    pub rbase: u8,
    /// VGPR holding `tid & 31`.
    pub lane: u8,
}
impl Default for AccRegs {
    fn default() -> Self {
        Self {
            m: 18,
            tile16: 47,
            rbase: 49,
            lane: 2,
        }
    }
}

fn tensors(kind: Kind) -> u8 {
    match kind {
        Kind::GateUp => 2,
        Kind::Down => 1,
    }
}

/// Reserve the BF16 epilogue's scratch with its real lifetime.
pub fn reserve_bf16(
    p: &mut RegPlan,
    kind: Kind,
    epilogue_label: &str,
    end_label: &str,
) -> Result<(), String> {
    let epi = || Live::Between(epilogue_label.into(), end_label.into());
    for i in 0..8u8 {
        p.v::<1>("rt_tmp", RT_TMP + i, epi())?;
    }
    if kind == Kind::GateUp {
        for i in 0..(6 * SILU_N) {
            p.v::<1>("silu_tmp", SET0 + i, epi())?;
        }
        for i in 0..(3 * SILU_N) {
            p.s::<2>("silu_mask", SILU_MASK + 2 * i, epi())?;
        }
    }
    p.v::<4>("out", OUT, epi())?;
    for i in 0..4u8 {
        p.v::<1>("own", OWN + i, epi())?;
    }
    for i in 0..2u8 {
        p.v::<1>("send", SEND + i, epi())?;
        p.v::<1>("recv", RECV + i, epi())?;
        p.v::<1>("sel", SEL + i, epi())?;
    }
    for i in 0..4u8 {
        p.v::<1>("pack", PACK + i, epi())?;
    }
    p.s::<2>("hi_mask", HIMASK, epi())?;
    p.s::<1>("permlane_sel", PSEL, epi())?;
    p.s::<4>("srd_y", SRD_Y, epi())?;
    Ok(())
}

/// Reserve the `_acc` epilogue's scratch with its real lifetime.
pub fn reserve_acc(
    p: &mut RegPlan,
    _kind: Kind,
    epilogue_label: &str,
    end_label: &str,
) -> Result<(), String> {
    let epi = || Live::Between(epilogue_label.into(), end_label.into());
    for i in 0..4u8 {
        p.v::<1>("acc_tmp", RT_TMP + i, epi())?;
    }
    p.s::<1>("acc_up_off", PSEL, epi())?;
    p.s::<4>("srd_y", SRD_Y, epi())?;
    Ok(())
}

/// GateUp: round g and u to BF16 (clear the low half), run the existing SiLU DAG
/// (`g / (1 + expf(-g)) * u`, two groups of four), then round the result. Down: round the SUM.
/// After this, the high half of `SUM[j]` (GateUp: of `SUM[j]`, j < 8) holds the BF16 bits.
pub fn emit_round_silu(b: &mut Builder, kind: Kind) -> Result<(), String> {
    if kind == Kind::GateUp {
        for t in 0..2u8 {
            for j in 0..8u8 {
                Bf16::rne_finite_passthrough(
                    b,
                    SUM + 8 * t + j,
                    RT_TMP + j,
                    MASKT[usize::from(j % 2)],
                    true,
                )?;
            }
        }
        for grp in 0..2u8 {
            Epilogue::silu_dense(
                b,
                SUM + SILU_N * grp,
                SUM + 8 + SILU_N * grp,
                SET0,
                SILU_MASK,
                SILU_N,
            )?;
        }
    }
    for j in 0..8u8 {
        Bf16::rne_finite_passthrough(b, SUM + j, RT_TMP + j, MASKT[usize::from(j % 2)], false)?;
    }
    Ok(())
}

/// Padding slots store exactly +0, applied after the final rounding, as a value select (EXEC untouched).
pub fn emit_apply_live(b: &mut Builder) -> Result<(), String> {
    for j in 0..8u8 {
        op(
            b,
            format!("v_cndmask_b32_e64 v{0}, 0, v{0}, s{LIVE}", SUM + j),
            &[v(SUM + j)],
            &[v(SUM + j), s(LIVE)],
        )?;
    }
    Ok(())
}

fn perm(b: &mut Builder, dst: u8, hi_src: u8, lo_src: u8, sel: String) -> Result<(), String> {
    let mut uses = vec![v(hi_src), v(lo_src)];
    if sel.starts_with('v') {
        uses.push(v(sel[1..].parse::<u8>().map_err(|e| e.to_string())?));
    }
    op(
        b,
        format!("v_perm_b32 v{dst}, v{hi_src}, v{lo_src}, {sel}"),
        &[v(dst)],
        &uses,
    )
}

/// The gfx11 half-wave exchange (SOL §2.4): lane `(hi, lr)` holds rows `2j+hi` of token `lr` in the
/// high half of `SUM[j]`; afterwards `OUT[k]` = `bf16(row 8hi+2k) | bf16(row 8hi+2k+1) << 16`.
/// Full EXEC throughout.
pub fn emit_exchange(b: &mut Builder, sel: &PackSelectors) -> Result<(), String> {
    let vals = SUM;
    // A VOP3 lane-mask operand is a pair for M7; the high word is unused in wave32.
    op(
        b,
        format!("s_mov_b32 s{HIMASK}, 0xffff0000"),
        &[s(HIMASK)],
        &[],
    )?;
    op(
        b,
        format!("s_mov_b32 s{}, 0", HIMASK + 1),
        &[s(HIMASK + 1)],
        &[],
    )?;
    op(
        b,
        format!("s_mov_b32 s{PSEL}, {}", lit(sel.permlane_lo)),
        &[s(PSEL)],
        &[],
    )?;
    for k in 0..4u8 {
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{}, v{}, v{}, s{HIMASK}",
                OWN + k,
                vals + k,
                vals + 4 + k
            ),
            &[v(OWN + k)],
            &[v(vals + k), v(vals + 4 + k), s(HIMASK)],
        )?;
    }
    for m in 0..2u8 {
        // hi = 0 sends pack(H[4+2m], H[5+2m]); hi = 1 sends pack(H[2m], H[2m+1]).
        perm(
            b,
            PACK + 2 * m,
            vals + 5 + 2 * m,
            vals + 4 + 2 * m,
            lit(sel.send),
        )?;
        perm(
            b,
            PACK + 2 * m + 1,
            vals + 1 + 2 * m,
            vals + 2 * m,
            lit(sel.send),
        )?;
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{}, v{}, v{}, s{HIMASK}",
                SEND + m,
                PACK + 2 * m,
                PACK + 2 * m + 1
            ),
            &[v(SEND + m)],
            &[v(PACK + 2 * m), v(PACK + 2 * m + 1), s(HIMASK)],
        )?;
        op(
            b,
            format!("v_mov_b32_e32 v{}, v{}", RECV + m, SEND + m),
            &[v(RECV + m)],
            &[v(SEND + m)],
        )?;
        op(
            b,
            format!(
                "v_permlanex16_b32 v{}, v{}, s{PSEL}, {}",
                RECV + m,
                SEND + m,
                lit(sel.permlane_hi)
            ),
            &[v(RECV + m)],
            &[v(RECV + m), v(SEND + m), s(PSEL)],
        )?;
    }
    // Byte selectors (src0 = received word, src1 = own row): hi = 0 puts its own row low,
    // hi = 1 puts the received row low.
    for i in 0..2u8 {
        let r = SEL + i;
        let (lo, hi) = (sel.recv_h0[usize::from(i)], sel.recv_h1[usize::from(i)]);
        op(b, format!("v_mov_b32_e32 v{r}, {}", lit(hi)), &[v(r)], &[])?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{r}, {}, v{r}, s{HIMASK}", lit(lo)),
            &[v(r)],
            &[v(r), s(HIMASK)],
        )?;
    }
    for k in 0..4u8 {
        perm(
            b,
            OUT + k,
            RECV + k / 2,
            OWN + k,
            format!("v{}", SEL + k % 2),
        )?;
    }
    Ok(())
}

/// Raw buffer store of `OUT` (16 bytes per lane) at `YOFF` over Y.
pub fn emit_store(b: &mut Builder) -> Result<(), String> {
    sop(
        b,
        format!("s_mov_b32 s{SRD_Y}, s{Y_PTR}"),
        &[SRD_Y],
        &[Y_PTR],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_Y + 1, Y_PTR + 1),
        &[SRD_Y + 1],
        &[Y_PTR + 1],
    )?;
    srd_tail(b, SRD_Y, None)?;
    bstore_b128(b, OUT, regs::YOFF, SRD_Y, 0)
}

/// Complete BF16 epilogue: round/SiLU, LIVE, exchange, store.
pub fn emit_bf16(b: &mut Builder, kind: Kind, sel: &PackSelectors) -> Result<(), String> {
    emit_round_silu(b, kind)?;
    emit_apply_live(b)?;
    emit_exchange(b, sel)?;
    emit_store(b)
}

/// Qualification F32 epilogue: scalar stores of every accumulator (padding +0) at
/// `4*(p*M + rbase + 2*j + hi + t*M/2)`; independent of the BF16 `YOFF`, pack and exchange.
pub fn emit_acc(b: &mut Builder, kind: Kind, r: &AccRegs) -> Result<(), String> {
    let tn = tensors(kind);
    for t in 0..tn {
        for j in 0..8u8 {
            let x = SUM + 8 * t + j;
            op(
                b,
                format!("v_cndmask_b32_e64 v{x}, 0, v{x}, s{LIVE}"),
                &[v(x)],
                &[v(x), s(LIVE)],
            )?;
        }
    }
    let (lr, hi, g, u) = (RT_TMP, RT_TMP + 1, RT_TMP + 2, RT_TMP + 3);
    op(
        b,
        format!("v_and_b32_e32 v{lr}, 15, v{}", r.lane),
        &[v(lr)],
        &[v(r.lane)],
    )?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{hi}, 4, v{}", r.lane),
        &[v(hi)],
        &[v(r.lane)],
    )?;
    // Gate (or Down) byte base: 4 * (p * M + rbase + hi), p = 16 * tile + lr.
    op(
        b,
        format!("v_add_nc_u32_e32 v{g}, s{}, v{lr}", r.tile16),
        &[v(g)],
        &[s(r.tile16), v(lr)],
    )?;
    op(
        b,
        format!("v_mul_lo_u32 v{g}, v{g}, s{}", r.m),
        &[v(g)],
        &[v(g), s(r.m)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{g}, s{}, v{g}", r.rbase),
        &[v(g)],
        &[s(r.rbase), v(g)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{g}, v{hi}, v{g}"),
        &[v(g)],
        &[v(hi), v(g)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{g}, 2, v{g}"),
        &[v(g)],
        &[v(g)],
    )?;
    if tn == 2 {
        // Up rows are M/2 rows (2*M bytes) further.
        sop(
            b,
            format!("s_lshl_b32 s{PSEL}, s{}, 1", r.m),
            &[PSEL],
            &[r.m],
        )?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{u}, s{PSEL}, v{g}"),
            &[v(u)],
            &[s(PSEL), v(g)],
        )?;
    }
    sop(
        b,
        format!("s_mov_b32 s{SRD_Y}, s{Y_PTR}"),
        &[SRD_Y],
        &[Y_PTR],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_Y + 1, Y_PTR + 1),
        &[SRD_Y + 1],
        &[Y_PTR + 1],
    )?;
    srd_tail(b, SRD_Y, None)?;
    // Row of accumulator j is rbase + 2j + hi: 8 bytes apart per j.
    for (t, voff) in [(0u8, g), (1u8, u)].into_iter().take(usize::from(tn)) {
        for j in 0..8u8 {
            let off = 8 * u32::from(j);
            if off > b.spec.arch.buffer_offset_max() {
                return Err(format!(
                    "acc store offset {off} exceeds the buffer offset field"
                ));
            }
            let d = SUM + 8 * t + j;
            let offset = if off == 0 {
                String::new()
            } else {
                format!(" offset:{off}")
            };
            mem(
                b,
                format!(
                    "buffer_store_b32 v{d}, v{voff}, s[{SRD_Y}:{}], 0 offen{offset}",
                    SRD_Y + 3
                ),
                &[],
                &[v(voff), v(d), sr(SRD_Y, 4)],
                MemoryClass::VmemStore,
            )?;
        }
    }
    Ok(())
}
