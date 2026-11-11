// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Standalone gfx1100 epilogue fixture (B4-A, for the G1b oracle): the K2 BF16 epilogue of
//! [`super::epilogue`], unchanged, fed from F32 accumulator arrays in memory instead of a K loop.
//! The oracle supplies the F32 sums and the slot validity, and compares every output byte against
//! `mq_i4_reference::grouped_prefill_epilogue_with`; this kernel never produces expected values.
//!
//! Entries (one module, `qwen4_moe_epilogue_fixture_pm_gfx1100`; `emit_module`):
//! `qwen4_moe_epilogue_fixture_{gate_up,down}_gfx1100` (frozen selectors) and the negative controls
//! `..._wrongsel` (`PackSelectors::wrong_pack_selector()`: exactly one final `v_perm` selector wrong).
//!
//! ABI, 32 bytes (`KernargLayout`):
//!
//! | off | name        | meaning |
//! |-----|-------------|---------|
//! | 0   | `sums`      | `f32[P][M]`. Down: `[p][row]`, `row < M`. GateUp: `[p][0..M/2)` are the gate sums, `[p][M/2..M)` the paired up sums |
//! | 8   | `slots`     | `i32[P]`: grouped slot of token `p`; `< 0` is a padding slot (LIVE = false) |
//! | 16  | `Y_grouped` | BF16 out, `u16[P][row_len]`, `row_len = M/2` (GateUp) or `M` (Down) |
//! | 24  | `M`         | u32, as the K2 kernels (GateUp `M = 2*row_len`) |
//!
//! Launch: wave32; GateUp block 64, Down block 128; grid `(M/64, P/16)` with `P` a multiple of 16
//! and `M` a multiple of 64. Wave `w` of CTA `x`, tile `y` handles rows `rbase = 32x + 16w`
//! (GateUp; up rows `M/2 + rbase`) or `64x + 16w` (Down) of tokens `16y + lr`; lane `l` holds
//! `lr = l & 15`, `hi = l >> 4`, and `SUM[j]` = `sums[p][rbase + 2j + hi]` (GateUp also
//! `SUM[8+j]` = `sums[p][M/2 + rbase + 2j + hi]`), exactly the K2 register contract of SOL §2.1.
//! Output bytes are BF16 pairs written by the frozen exchange. Every output element is written
//! exactly once: padding tokens store `0x0000`.

use super::epilogue::{self, regs::*, PackSelectors};
use peacemaker_isa::{
    kernels::{
        common::{bload, op, s, smem, sop, srd_tail, v},
        qwen4_moe_sym::Kind,
    },
    reg::Live,
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};

pub const MODULE: &str = "qwen4_moe_epilogue_fixture_pm_gfx1100";
pub const KERNARG_BYTES: u32 = 32;

const KARG: u8 = 0;
const T0: u8 = 4;
const T1: u8 = 5;
const SUMS: u8 = 8; // s8:9 sums, s10:11 slots
const M: u8 = 18;
const SRD_SUM: u8 = 28;
const SRD_SLOT: u8 = 40;
const TILE16: u8 = 47;
const WAVE: u8 = 48;
const RBASE: u8 = 49;
/// Real user SGPRs (the kernarg pointer) and enabled system SGPRs (workgroup ids X, Y).
const USER_SGPRS: u8 = 2;
const SYSTEM_SGPRS: u8 = 2;
/// gfx1100 `FeatureUserSGPRInit16Bug`: the descriptor declares user SGPRs padded to user + system = 16, so the
/// native workgroup ids are s14 (x) and s15 (y) (`s8:11` and `s16:` are clear of them).
const DECLARED_USER_SGPRS: u8 = Arch::Gfx1100.declared_user_sgprs(USER_SGPRS, SYSTEM_SGPRS);
pub const WG_X_IN: u8 = DECLARED_USER_SGPRS;
pub const WG_Y_IN: u8 = DECLARED_USER_SGPRS + 1;

/// Entry symbol, e.g. `qwen4_moe_epilogue_fixture_gate_up_wrongsel_gfx1100`.
pub fn symbol(kind: Kind, wrong_pack_selector: bool) -> String {
    format!(
        "qwen4_moe_epilogue_fixture_{}{}_gfx1100",
        kind.tag(),
        if wrong_pack_selector { "_wrongsel" } else { "" }
    )
}

/// Local label unique per module entry (module assembly keeps `.L` names kernel-local).
fn label(kind: Kind, wrong: bool, name: &str) -> String {
    format!(
        ".Lq4fx_{}{}_{name}",
        kind.tag(),
        if wrong { "_wrongsel" } else { "" }
    )
}

fn plan(kind: Kind, wrong: bool) -> Result<RegPlan, String> {
    let mut p = RegPlan::new(256, 104)?;
    let kernel = || Live::Between("entry".into(), label(kind, wrong, "epilogue"));
    p.v::<1>("tid", 0, kernel())?;
    for i in 1..16u8 {
        p.v::<1>("prologue_tmp", i, kernel())?;
    }
    p.v::<1>("y_off", YOFF, Live::Whole)?;
    p.v::<8>("sum", SUM, Live::Whole)?;
    if kind == Kind::GateUp {
        p.v::<8>("sum_up", SUM + 8, Live::Whole)?;
    }
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    p.s::<1>("wg_x_in", WG_X_IN, Live::Whole)?;
    p.s::<1>("wg_y_in", WG_Y_IN, Live::Whole)?;
    p.s::<1>("s_tmp0", T0, kernel())?;
    p.s::<1>("s_tmp1", T1, kernel())?;
    p.s::<4>("args_ptrs", SUMS, kernel())?;
    p.s::<2>("y_base", Y_PTR, Live::Whole)?;
    p.s::<1>("m", M, kernel())?;
    p.s::<4>("srd_sums", SRD_SUM, kernel())?;
    p.s::<4>("srd_slots", SRD_SLOT, kernel())?;
    p.s::<1>("tile16", TILE16, kernel())?;
    p.s::<1>("wave", WAVE, kernel())?;
    p.s::<1>("row_base", RBASE, kernel())?;
    p.s::<2>("live", LIVE, Live::Whole)?;
    for m in MASKT {
        p.s::<2>("cmp_mask", m, Live::Whole)?;
    }
    epilogue::reserve_bf16(
        &mut p,
        kind,
        &label(kind, wrong, "epilogue"),
        &label(kind, wrong, "end"),
    )?;
    let top = p.next_free_vgpr();
    if top % 8 != 0 {
        p.v::<1>("granule_pad", (top.div_ceil(8) * 8 - 1) as u8, Live::Whole)?;
    }
    Ok(p)
}

fn builder(kind: Kind, wrong_pack_selector: bool) -> Result<Builder, String> {
    let spec = KernelSpec {
        kernel_id: "qwen4_moe_epilogue_fixture".into(),
        variant: format!(
            "{}{}",
            kind.tag(),
            if wrong_pack_selector { "_wrongsel" } else { "" }
        ),
        arch: Arch::Gfx1100,
        symbol: symbol(kind, wrong_pack_selector),
        kernargs: KernargLayout::new(KERNARG_BYTES)
            .pointer("sums", 0)
            .pointer("slots", 8)
            .pointer("Y_grouped", 16)
            .hidden("M", 24, 4, "by_value"),
        user_sgpr_count: USER_SGPRS,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: if kind == Kind::GateUp { 64 } else { 128 },
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(spec, plan(kind, wrong_pack_selector)?);
    b.enable_delay_alu();
    Ok(b)
}

/// Kernarg loads, lane/tile/row ids, LIVE from `slots`, YOFF, and the F32 sum loads into `SUM`.
/// Leaves the builder at the end of the prologue; the caller places the `epilogue` label.
pub fn emit_prologue(b: &mut Builder, kind: Kind) -> Result<(), String> {
    smem(b, SUMS, 4, KARG, 0)?;
    smem(b, Y_PTR, 2, KARG, 0x10)?;
    smem(b, M, 1, KARG, 0x18)?;
    // v1 = wave, v2 = lane, v3 = lr, v4 = hi (workgroup ids in s14/s15 on gfx1100, see WG_X_IN).
    op(b, "v_lshrrev_b32_e32 v1, 5, v0", &[v(1)], &[v(0)])?;
    op(b, "v_and_b32_e32 v2, 31, v0", &[v(2)], &[v(0)])?;
    op(
        b,
        format!("v_readfirstlane_b32 s{WAVE}, v1"),
        &[s(WAVE)],
        &[v(1)],
    )?;
    op(b, "v_and_b32_e32 v3, 15, v2", &[v(3)], &[v(2)])?;
    op(b, "v_lshrrev_b32_e32 v4, 4, v2", &[v(4)], &[v(2)])?;
    sop(
        b,
        format!("s_lshl_b32 s{TILE16}, s{WG_Y_IN}, 4"),
        &[TILE16],
        &[WG_Y_IN],
    )?;
    let shift = if kind == Kind::GateUp { 5 } else { 6 };
    sop(
        b,
        format!("s_lshl_b32 s{RBASE}, s{WG_X_IN}, {shift}"),
        &[RBASE],
        &[WG_X_IN],
    )?;
    sop(b, format!("s_lshl_b32 s{T0}, s{WAVE}, 4"), &[T0], &[WAVE])?;
    sop(
        b,
        format!("s_add_i32 s{RBASE}, s{RBASE}, s{T0}"),
        &[RBASE],
        &[RBASE, T0],
    )?;
    // LIVE: slot >= 0 of token p = 16*tile + lr.
    sop(
        b,
        format!("s_mov_b32 s{SRD_SLOT}, s{}", SUMS + 2),
        &[SRD_SLOT],
        &[SUMS + 2],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_SLOT + 1, SUMS + 3),
        &[SRD_SLOT + 1],
        &[SUMS + 3],
    )?;
    srd_tail(b, SRD_SLOT, None)?;
    op(
        b,
        format!("v_add_nc_u32_e32 v5, s{TILE16}, v3"),
        &[v(5)],
        &[s(TILE16), v(3)],
    )?;
    op(b, "v_lshlrev_b32_e32 v5, 2, v5", &[v(5)], &[v(5)])?;
    bload(b, 1, 6, 5, SRD_SLOT, 0)?;
    op(
        b,
        format!("v_cmp_le_i32_e64 s{LIVE}, 0, v6"),
        &[s(LIVE)],
        &[v(6)],
    )?;
    // Sums: byte offset 4 * (p*M + rbase + hi) (+ 2*M bytes for the up rows), + 8*j.
    sop(
        b,
        format!("s_mov_b32 s{SRD_SUM}, s{SUMS}"),
        &[SRD_SUM],
        &[SUMS],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_SUM + 1, SUMS + 1),
        &[SRD_SUM + 1],
        &[SUMS + 1],
    )?;
    srd_tail(b, SRD_SUM, None)?;
    op(
        b,
        format!("v_add_nc_u32_e32 v10, s{TILE16}, v3"),
        &[v(10)],
        &[s(TILE16), v(3)],
    )?;
    op(
        b,
        format!("v_mul_lo_u32 v10, v10, s{M}"),
        &[v(10)],
        &[v(10), s(M)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v10, s{RBASE}, v10"),
        &[v(10)],
        &[s(RBASE), v(10)],
    )?;
    op(b, "v_add_nc_u32_e32 v10, v4, v10", &[v(10)], &[v(4), v(10)])?;
    op(b, "v_lshlrev_b32_e32 v10, 2, v10", &[v(10)], &[v(10)])?;
    if kind == Kind::GateUp {
        sop(b, format!("s_lshl_b32 s{T0}, s{M}, 1"), &[T0], &[M])?;
        op(
            b,
            format!("v_add_nc_u32_e32 v11, s{T0}, v10"),
            &[v(11)],
            &[s(T0), v(10)],
        )?;
    }
    for j in 0..8u8 {
        bload(b, 1, SUM + j, 10, SRD_SUM, 8 * u32::from(j))?;
        if kind == Kind::GateUp {
            bload(b, 1, SUM + 8 + j, 11, SRD_SUM, 8 * u32::from(j))?;
        }
    }
    // YOFF = 2 * (p * row_len + rbase + 8*hi), row_len = M/2 (GateUp) or M (Down).
    match kind {
        Kind::GateUp => sop(b, format!("s_lshr_b32 s{T1}, s{M}, 1"), &[T1], &[M])?,
        Kind::Down => sop(b, format!("s_mov_b32 s{T1}, s{M}"), &[T1], &[M])?,
    }
    op(
        b,
        format!("v_add_nc_u32_e32 v8, s{TILE16}, v3"),
        &[v(8)],
        &[s(TILE16), v(3)],
    )?;
    op(
        b,
        format!("v_mul_lo_u32 v8, v8, s{T1}"),
        &[v(8)],
        &[v(8), s(T1)],
    )?;
    op(
        b,
        format!("v_lshl_add_u32 v9, v4, 3, s{RBASE}"),
        &[v(9)],
        &[v(4), s(RBASE)],
    )?;
    op(b, "v_add_nc_u32_e32 v8, v8, v9", &[v(8)], &[v(8), v(9)])?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{YOFF}, 1, v8"),
        &[v(YOFF)],
        &[v(8)],
    )?;
    Ok(())
}

/// The complete entry as a builder, before `finish` (the program is readable for audits).
pub fn emit_builder(kind: Kind, wrong_pack_selector: bool) -> Result<Builder, String> {
    let sel = if wrong_pack_selector {
        PackSelectors::wrong_pack_selector()
    } else {
        PackSelectors::frozen()
    };
    let mut b = builder(kind, wrong_pack_selector)?;
    emit_prologue(&mut b, kind)?;
    b.label(&label(kind, wrong_pack_selector, "epilogue"))?;
    epilogue::emit_bf16(&mut b, kind, &sel)?;
    b.label(&label(kind, wrong_pack_selector, "end"))?;
    op(&mut b, "s_endpgm", &[], &[])?;
    Ok(b)
}

/// The prologue alone (kernarg loads, ids, LIVE, YOFF, sum loads), for audits.
pub fn prologue_builder(kind: Kind) -> Result<Builder, String> {
    let mut b = builder(kind, false)?;
    emit_prologue(&mut b, kind)?;
    Ok(b)
}

/// One fixture entry.
pub fn emit(kind: Kind, wrong_pack_selector: bool) -> Result<Emitted, String> {
    emit_builder(kind, wrong_pack_selector)?.finish()
}

/// The four entries (GateUp, Down, each with its wrong-selector control) as one code-object source.
pub fn emit_module() -> Result<(Vec<Emitted>, String), String> {
    let mut emitted = Vec::new();
    for wrong in [false, true] {
        for kind in [Kind::GateUp, Kind::Down] {
            emitted.push(emit(kind, wrong)?);
        }
    }
    let (text, _proof) = peacemaker_isa::kernels::iu4_gemm::module(&emitted, MODULE)?;
    Ok((emitted, text))
}
