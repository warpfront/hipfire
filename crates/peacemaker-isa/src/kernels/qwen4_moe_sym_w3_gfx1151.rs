// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1151 `WeightCodec::Mq3I4` NT1 entries (gate/up `qwen4_moe_gate_up_silu_iu4_sym_w3_pm_gfx1151`,
//! down `qwen4_moe_down_iu4_sym_w3_pm_gfx1151`) and their F32 `_acc` twins. Sibling of the
//! gfx1201 body in `qwen4_moe_sym_w3.rs`, which is untouched: same 64-byte ABI (60 bytes plus a trailing `n_experts`), planar
//! qt131/qt132 tensors (`[codes: M*K*3/8 B][scales: M*(K/128) f16]`, weight `d * (2c - 7)`),
//! grid, A4 activations, exact int32 WMMA chain, magic-seeded fold and BF16 epilogue (SiLU for
//! gate/up) as the MQ4 gfx1151 NT1 entries; only the weight side differs.
//!
//! `v_wmma_i32_16x16x16_iu4`, eight K16 steps per K128 epoch: lanes `r` and `r + 16` both carry
//! the whole row's operand (16 dwords), so every lane loads all sixteen 8-code chunks of its
//! row's 48-byte epoch group, chunk `c` (K elements `8c..8c+8`) as one unaligned dword at byte
//! `3c` whose low 24 bits are the codes. K16 step `i` consumes dwords `2i`, `2i + 1`, the MQ4
//! gfx11 operand order. Each chunk is spread to eight 4-bit slots in place (three bit-field
//! inserts finished to the clean nibble `2c + 1` by one shift-or and one more insert, then the
//! `0x88888888` rebias XOR: the signed nibble `(2c - 7) & 15`, eight VALU per chunk); junk above
//! bit 23 never reaches an output.
//!
//! Registers: the MQ4 gfx11 operand set (`Gen::set_len`: 16 dwords per tensor, X, headers, `d`;
//! two sets) plus two unpack temporaries: 200 VGPRs for gate/up
//! (168 down). Row scales: C rows are `2j + hi`, so lane `(hi, k)` loads the scale of row
//! `(k & 14) | hi` and the `ds_swizzle` broadcast slot of accumulator `j` is `2j`; the odd epoch's
//! half of the trip's scale pair is moved down first. The BF16 epilogue is the MQ4 entries' own
//! (lane exchange so lane `hi` stores rows `8hi..8hi+8`); the `_acc` entry stores each accumulator
//! `j` as one dword at row `rbase + 2j + hi`. A loaded set is at most 32 loads per `s_clause`: the
//! gate/up second tensor's codes are a clause of their own. The down entry's odd epoch count tail
//! is the MQ4 down entry's `if_else`.
use super::*;

fn set_base(g: &Gen, p: usize) -> u8 {
    SET0 + p as u8 * g.set_len()
}
/// Code dword `c` of tensor `t` in set `p` (= chunk `c`; unpacked in place).
fn raw(g: &Gen, p: usize, t: u8, c: u8) -> u8 {
    set_base(g, p) + 16 * t + c
}
fn xr(g: &Gen, p: usize) -> u8 {
    set_base(g, p) + 16 * g.t()
}
fn hr(g: &Gen, p: usize, t: u8) -> u8 {
    xr(g, p) + 16 + t
}
fn dr(g: &Gen, p: usize) -> u8 {
    xr(g, p) + 16 + g.t()
}
/// Two unpack temporaries.
fn tmp(g: &Gen) -> u8 {
    set_base(g, 2)
}
/// Byte of chunk `c` in the epoch's 48-byte group.
fn chunk_off(c: u8) -> u32 {
    3 * u32::from(c)
}

pub(super) fn plan(g: &Gen) -> Result<RegPlan, String> {
    let (tn, acc) = (g.t(), g.acc);
    let gate_up = g.spec.kind == Kind::GateUp;
    let mut p = RegPlan::new(VGPR_CEILING, 104)?;
    let l = |n: &str| g.label(n);
    let kernel = || Live::Between("entry".into(), l("epilogue"));
    let pro = || Live::Between("entry".into(), l("k_begin"));
    let epi = || Live::Between(l("epilogue"), l("end"));
    p.v::<1>("tid", TID, pro())?;
    for i in 1..16u8 {
        p.v::<1>("prologue_tmp", i, pro())?;
    }
    for (name, r) in [("x_off", XOFF), ("w_off", WOFF), ("h_off", HOFF)] {
        p.v::<1>(name, r, kernel())?;
    }
    p.v::<1>("y_off", YOFF, Live::Whole)?;
    p.v::<8>("magic8", MAGIC8, kernel())?;
    for t in 0..tn {
        p.v::<8>("cacc", CACC + 8 * t, kernel())?;
        p.v::<8>("sum", SUM + 8 * t, Live::Whole)?;
        p.v::<8>("scale_rows", SCF + 8 * t, kernel())?;
        p.v::<1>("scale_f32", HF + t, kernel())?;
    }
    p.v::<8>("fold_t", TPROD, kernel())?;
    // Operands at their WMMA/load granularity: code-dword pairs (the K16 A operand) and X quads (b128).
    for set in 0..2usize {
        for t in 0..tn {
            for k in 0..8u8 {
                p.v::<2>("operand", raw(g, set, t, 2 * k), kernel())?;
            }
        }
        for chunk in 0..4u8 {
            p.v::<4>("operand", xr(g, set) + 4 * chunk, kernel())?;
        }
        for t in 0..tn {
            p.v::<1>("scale_pair", hr(g, set, t), kernel())?;
        }
        p.v::<1>("d", dr(g, set), kernel())?;
    }
    for i in 0..2u8 {
        p.v::<1>("unpack_tmp", tmp(g) + i, kernel())?;
    }
    // Epilogue registers over the dead K-loop ranges (the MQ4 gfx11 epilogue's).
    for i in 0..8u8 {
        p.v::<1>("rt_tmp", RT_TMP + i, epi())?;
    }
    if gate_up {
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
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    p.s::<4>("srd_y", SRD_Y, epi())?;
    p.s::<1>("wg_x_in", 2, Live::Whole)?;
    p.s::<1>("wg_y_in", 3, Live::Whole)?;
    p.s::<1>("s_tmp0", T0, kernel())?;
    p.s::<1>("s_tmp1", T1, kernel())?;
    p.s::<2>("s_tmp64", T64, kernel())?;
    p.s::<8>("kernargs_0x00", ARGS0, kernel())?;
    p.s::<2>("y_base", ARGS1, Live::Whole)?;
    p.s::<2>("kernargs_m_k", ARGS1 + 2, kernel())?;
    p.s::<2>("kernargs_0x30", ARGS2, kernel())?;
    p.s::<1>("x_src_rows", ARGS3, kernel())?;
    p.s::<1>("expert", EID, kernel())?;
    for t in 0..tn {
        p.s::<4>("srd_codes", srd_c(t), kernel())?;
        p.s::<4>("srd_scales", srd_s(t), kernel())?;
    }
    for r in SRD_X {
        p.s::<4>("srd_x", r, kernel())?;
    }
    p.s::<4>("srd_sorted", SRD_S, kernel())?;
    for (name, r) in [
        ("trips", TRIPS),
        ("x_stride2", XS2),
        ("row_bytes", ROWB),
        ("scale_row_bytes", SROW),
        ("tile16", TILE16),
        ("wave", WAVE),
        ("row_base", RBASE),
        ("wg_x", WGX),
        ("wg_y", WGY),
    ] {
        p.s::<1>(name, r, kernel())?;
    }
    p.s::<2>("w_ptr", WPTR, kernel())?;
    p.s::<2>("live", LIVE, Live::Whole)?;
    if acc && gate_up {
        p.s::<1>("up_offset", UPOFF, Live::Whole)?;
    }
    for m in MASKT {
        p.s::<2>("cmp_mask", m, Live::Whole)?;
    }
    let top = p.next_free_vgpr();
    if top % 8 != 0 {
        p.v::<1>("granule_pad", (top.div_ceil(8) * 8 - 1) as u8, Live::Whole)?;
    }
    Ok(p)
}

/// One gfx1151 entry (`acc`: the F32 qualification twin). The caller (`w3::emit_impl`) has
/// validated the spec.
pub(super) fn emit_gfx1151(spec: Spec, acc: bool) -> Result<Emitted, String> {
    if spec.arch != Arch::Gfx1151 {
        return Err("qwen4_moe_sym_w3_gfx1151: gfx1151 only".into());
    }
    let g = Gen { spec, acc };
    let sfx = if acc { "_acc" } else { "" };
    let kspec = KernelSpec {
        kernel_id: "qwen4_moe_sym".into(),
        variant: format!("{}{sfx}", spec.variant()),
        arch: spec.arch,
        symbol: format!("{}{sfx}", spec.symbol()),
        kernargs: spec.kernargs(),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: spec.threads() as u16,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan(&g)?);
    b.enable_delay_alu();
    body(&mut Workgroup::<Gfx1151, Builder>::new(&mut b)?, &g)?;
    b.finish()
}

fn body(wg: &mut Wg<Gfx1151>, g: &Gen) -> Result<(), String> {
    let end = wg.exit(&g.label("end"))?;
    prologue(wg, g, &end)?;
    kloop(wg, g)?;
    if g.acc {
        epilogue_acc(wg.isa(), g)?
    } else {
        epilogue(wg.isa(), g)?
    }
    wg.end(end)
}

fn prologue(wg: &mut Wg<Gfx1151>, g: &Gen, end: &End) -> Result<(), String> {
    let a = g.arch();
    let acc = g.acc;
    let gate_up = g.spec.kind == Kind::GateUp;
    let b = wg.isa();
    smem(b, ARGS0, 8, KARG, 0)?;
    smem(b, ARGS1, 2, KARG, 0x20)?;
    smem(b, ARGS1 + 2, 2, KARG, 0x28)?;
    smem(b, ARGS2, 2, KARG, 0x30)?;
    smem(b, ARGS3, 1, KARG, 0x38)?;
    // Workgroup ids arrive in s2 (x) and s3 (y).
    sop(b, format!("s_mov_b32 s{WGX}, s2"), &[WGX], &[2])?;
    sop(b, format!("s_mov_b32 s{WGY}, s3"), &[WGY], &[3])?;
    // v1 = wave, v2 = lane, v3 = lr, v4 = hi.
    op(b, "v_lshrrev_b32_e32 v1, 5, v0", &[v(1)], &[v(TID)])?;
    op(b, "v_and_b32_e32 v2, 31, v0", &[v(2)], &[v(TID)])?;
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
        format!("s_lshl_b32 s{TILE16}, s{WGY}, 4"),
        &[TILE16],
        &[WGY],
    )?;
    let outside = wg.scmp_wg_uniform(Instruction::new(
        format!("s_cmp_ge_i32 s{TILE16}, s{}", ARGS2 + 1),
        vec![],
        vec![s(TILE16), s(ARGS2 + 1)],
    ))?;
    wg.exit_if(outside, end)?;
    let b = wg.isa();
    sop(b, format!("s_lshl_b32 s{T0}, s{WGY}, 2"), &[T0], &[WGY])?;
    add64(b, T64, ARGS0 + 2, T0)?;
    smem(b, EID, 1, T64, 0)?;
    // Fail closed: an expert id outside [0, n_experts) (the -1 sentinel, or a corrupt / non-finite-router id) exits
    // before any `expert_weight_ptrs[id]` load. One unsigned compare covers both bounds (negative ids are huge).
    smem(b, T1, 1, KARG, 0x3c)?;
    let sentinel = wg.scmp_wg_uniform(Instruction::new(
        format!("s_cmp_ge_u32 s{EID}, s{T1}"),
        vec![],
        vec![s(EID), s(T1)],
    ))?;
    wg.exit_if(sentinel, end)?;
    let b = wg.isa();
    sop(b, format!("s_lshl_b32 s{T0}, s{EID}, 3"), &[T0], &[EID])?;
    add64(b, T64, ARGS0, T0)?;
    smem(b, WPTR, 2, T64, 0)?;
    // Codes row bytes K*3/8; scale row bytes (K/128)*2.
    let (m, k) = (ARGS1 + 2, ARGS1 + 3);
    sop(b, format!("s_mul_i32 s{ROWB}, s{k}, 3"), &[ROWB], &[k])?;
    sop(
        b,
        format!("s_lshr_b32 s{ROWB}, s{ROWB}, 3"),
        &[ROWB],
        &[ROWB],
    )?;
    sop(b, format!("s_lshr_b32 s{SROW}, s{k}, 6"), &[SROW], &[k])?;
    // Two-epoch trips before the tail: (K/128 - 1) / 2.
    sop(b, format!("s_lshr_b32 s{TRIPS}, s{k}, 7"), &[TRIPS], &[k])?;
    sop(
        b,
        format!("{} s{TRIPS}, s{TRIPS}, -1", s_add_i32(a)),
        &[TRIPS],
        &[TRIPS],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{TRIPS}, s{TRIPS}, 1"),
        &[TRIPS],
        &[TRIPS],
    )?;
    // Row base of this wave: x*cta_rows + 16*wave.
    sop(
        b,
        format!(
            "s_lshl_b32 s{RBASE}, s{WGX}, {}",
            if gate_up { 5 } else { 6 }
        ),
        &[RBASE],
        &[WGX],
    )?;
    sop(b, format!("s_lshl_b32 s{T0}, s{WAVE}, 4"), &[T0], &[WAVE])?;
    sop(
        b,
        format!("{} s{RBASE}, s{RBASE}, s{T0}", s_add_i32(a)),
        &[RBASE],
        &[RBASE, T0],
    )?;
    // Sorted slots of this tile.
    sop(b, format!("s_lshl_b32 s{T0}, s{WGY}, 6"), &[T0], &[WGY])?;
    add64(b, SRD_S, ARGS0 + 4, T0)?;
    sop(
        b,
        format!("s_mov_b32 s{}, 64", SRD_S + 2),
        &[SRD_S + 2],
        &[],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, {}", SRD_S + 3, lit(SRD_WORD3)),
        &[SRD_S + 3],
        &[],
    )?;
    op(b, "v_lshlrev_b32_e32 v5, 2, v3", &[v(5)], &[v(3)])?;
    bload(b, 1, 6, 5, SRD_S, 0)?;
    // Activation descriptors: epoch 0 (X0) and 1 (X1), x_src_rows*72 records.
    sop(
        b,
        format!("s_mul_i32 s{T0}, s{ARGS3}, 0x48"),
        &[T0],
        &[ARGS3],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_X[0], ARGS0 + 6),
        &[SRD_X[0]],
        &[ARGS0 + 6],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_X[0] + 1, ARGS0 + 7),
        &[SRD_X[0] + 1],
        &[ARGS0 + 7],
    )?;
    srd_tail(b, SRD_X[0], Some(T0))?;
    add64(b, SRD_X[1], ARGS0 + 6, T0)?;
    srd_tail(b, SRD_X[1], Some(T0))?;
    sop(b, format!("s_lshl_b32 s{XS2}, s{T0}, 1"), &[XS2], &[T0])?;
    // Code descriptors: rows rbase.. of the expert (gate/up: up rows M/2 further).
    sop(
        b,
        format!("s_mul_i32 s{T0}, s{RBASE}, s{ROWB}"),
        &[T0],
        &[RBASE, ROWB],
    )?;
    add64(b, SRD_W, WPTR, T0)?;
    srd_tail(b, SRD_W, None)?;
    if gate_up {
        sop(b, format!("s_lshr_b32 s{T1}, s{m}, 1"), &[T1], &[m])?;
        sop(
            b,
            format!("s_mul_i32 s{T1}, s{T1}, s{ROWB}"),
            &[T1],
            &[T1, ROWB],
        )?;
        add64(b, SRD_U, SRD_W, T1)?;
        srd_tail(b, SRD_U, None)?;
    }
    // Scale descriptors: the scale plane starts M*row_bytes after the tensor base.
    sop(
        b,
        format!("s_mul_i32 s{T0}, s{m}, s{ROWB}"),
        &[T0],
        &[m, ROWB],
    )?;
    add64(b, SRD_WS, WPTR, T0)?;
    sop(
        b,
        format!("s_mul_i32 s{T1}, s{RBASE}, s{SROW}"),
        &[T1],
        &[RBASE, SROW],
    )?;
    add64(b, SRD_WS, SRD_WS, T1)?;
    if gate_up {
        sop(b, format!("s_lshr_b32 s{T1}, s{m}, 1"), &[T1], &[m])?;
        sop(
            b,
            format!("s_mul_i32 s{T1}, s{T1}, s{SROW}"),
            &[T1],
            &[T1, SROW],
        )?;
        add64(b, SRD_US, SRD_WS, T1)?;
        srd_tail(b, SRD_US, None)?;
    }
    srd_tail(b, SRD_WS, None)?;
    // Lane offsets. Codes: lr*row_bytes (lanes r and r+16 read the same row).
    op(
        b,
        format!("v_mul_u32_u24_e32 v{WOFF}, s{ROWB}, v3"),
        &[v(WOFF)],
        &[s(ROWB), v(3)],
    )?;
    // Scale row: the row whose scale this lane's broadcast slot serves (C rows 2j + hi).
    op(b, "v_and_b32_e32 v7, 14, v3", &[v(7)], &[v(3)])?;
    op(b, "v_or_b32_e32 v7, v7, v4", &[v(7)], &[v(7), v(4)])?;
    op(
        b,
        format!("v_mul_u32_u24_e32 v{HOFF}, s{SROW}, v7"),
        &[v(HOFF)],
        &[s(SROW), v(7)],
    )?;
    // Y offset: ((16*tile + lr)*row_len + rbase + 8*hi)*2 (BF16, after the epilogue's lane
    // exchange) or ((16*tile + lr)*M + rbase + hi)*4 (F32 accumulators: lane (hi, lr) holds rows
    // 2j + hi, stored per j at `+ 8j`). row_len is M/2 for the BF16 gate/up H and M otherwise.
    if gate_up && !acc {
        sop(b, format!("s_lshr_b32 s{T1}, s{m}, 1"), &[T1], &[m])?;
    } else {
        sop(b, format!("s_mov_b32 s{T1}, s{m}"), &[T1], &[m])?;
    }
    if gate_up && acc {
        // Up rows sit M/2 columns (4 bytes each) further.
        sop(b, format!("s_lshl_b32 s{UPOFF}, s{m}, 1"), &[UPOFF], &[m])?;
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
    if acc {
        op(
            b,
            format!("v_add_nc_u32_e32 v9, s{RBASE}, v4"),
            &[v(9)],
            &[s(RBASE), v(4)],
        )?;
    } else {
        op(
            b,
            format!("v_lshl_add_u32 v9, v4, 3, s{RBASE}"),
            &[v(9)],
            &[v(4), s(RBASE)],
        )?;
    }
    op(b, "v_add_nc_u32_e32 v8, v8, v9", &[v(8)], &[v(8), v(9)])?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{YOFF}, {}, v8", if acc { 2 } else { 1 }),
        &[v(YOFF)],
        &[v(8)],
    )?;
    // Activation row offset (padding slots get the gather's out-of-range offset).
    common::gather_offset(
        b,
        XOFF,
        6,
        Some(ARGS2),
        iu4_fold::XBLK_BYTES,
        LIVE,
        GatherTemps {
            v: [11, 12, 13, 10],
            mask: MASKT[0],
        },
    )?;
    load_set(b, g, 0)?;
    for j in 0..8u8 {
        op(
            b,
            format!("v_mov_b32_e32 v{}, {}", MAGIC8 + j, lit(MAGIC)),
            &[v(MAGIC8 + j)],
            &[],
        )?;
    }
    for t in 0..g.t() {
        for j in 0..8u8 {
            op(
                b,
                format!("v_mov_b32_e32 v{}, 0", SUM + 8 * t + j),
                &[v(SUM + 8 * t + j)],
                &[],
            )?;
        }
    }
    Ok(())
}

/// Issue set `p`'s loads (epoch parity `p` of the current trip): the trip's scale pair of each
/// tensor's header row, `d`, the activation nibbles (b128), then one unaligned dword per chunk
/// (byte `3c` of the epoch's 48-byte group). At most 32 loads per `s_clause`: the head and the
/// first tensor's codes share one, each further tensor's codes are a clause of their own.
fn load_set(b: &mut Builder, g: &Gen, p: usize) -> Result<(), String> {
    let tn = g.t();
    let head = |b: &mut Builder| -> Result<(), String> {
        for t in 0..tn {
            bload(b, 1, hr(g, p, t), HOFF, srd_s(t), 0)?;
        }
        bload(b, 1, dr(g, p), XOFF, SRD_X[p], 0)?;
        for i in 0..4u8 {
            bload(
                b,
                4,
                xr(g, p) + 4 * i,
                XOFF,
                SRD_X[p],
                8 + 16 * u32::from(i),
            )?;
        }
        Ok(())
    };
    let codes = |b: &mut Builder, t: u8| -> Result<(), String> {
        for c in 0..16u8 {
            bload(
                b,
                1,
                raw(g, p, t, c),
                WOFF,
                srd_c(t),
                48 * p as u32 + chunk_off(c),
            )?;
        }
        Ok(())
    };
    b.clause(|b| {
        head(b)?;
        codes(b, 0)
    })?;
    for t in 1..tn {
        b.clause(|b| codes(b, t))?;
    }
    Ok(())
}

/// One chunk: the dword at the chunk's byte offset (its low 24 bits are the 8 codes), spread
/// and sign-finished into `dst` in eight VALU. `dst == code` is fine: `code` is last read by
/// the second instruction, `dst` first written by the third; `y` is the only temporary.
///
/// ```text
/// y   = code << 4;  y = bfi(0xfff, code, y)        fields 0..3 | fields 4..7 << 4
/// dst = y << 2;     y = bfi(0x003f003f, y, dst)    field pairs, junk between them
/// dst = y << 1;     y = (y << 2) | 0x11111111      odd fields at bits 5..7, bit 0 / 4 of each nibble = 1
/// dst = bfi(0x0e0e0e0e, dst, y)                    even fields at bits 1..3: nibble `2c + 1`, no junk
/// dst ^= 0x88888888                                signed nibble `2c - 7`
/// ```
fn unpack(b: &mut Builder, dst: u8, code: u8, y: u8) -> Result<(), String> {
    let bfi = |b: &mut Builder, d: u8, mask: u32, p: u8, q: u8| {
        op(
            b,
            format!("v_bfi_b32 v{d}, {}, v{p}, v{q}", lit(mask)),
            &[v(d)],
            &[v(p), v(q)],
        )
    };
    let shl = |b: &mut Builder, d: u8, n: u8, sr: u8| {
        op(
            b,
            format!("v_lshlrev_b32_e32 v{d}, {n}, v{sr}"),
            &[v(d)],
            &[v(sr)],
        )
    };
    shl(b, y, 4, code)?;
    bfi(b, y, 0xfff, code, y)?;
    shl(b, dst, 2, y)?;
    bfi(b, y, 0x003f_003f, y, dst)?;
    shl(b, dst, 1, y)?;
    op(
        b,
        format!("v_lshl_or_b32 v{y}, v{y}, 2, 0x11111111"),
        &[v(y)],
        &[v(y)],
    )?;
    bfi(b, dst, 0x0e0e_0e0e, dst, y)?;
    op(
        b,
        format!("v_xor_b32_e32 v{dst}, {}, v{dst}", lit(REBIAS)),
        &[v(dst)],
        &[v(dst)],
    )
}

/// Fold set `p` (one K128 epoch) into the sums.
fn compute(b: &mut Builder, g: &Gen, p: usize) -> Result<(), String> {
    let a = g.arch();
    let tn = g.t();
    // Row scales: epoch parity selects the low (even) or high (odd) half of the scale pair (the
    // odd half is moved down in place first; the pair is dead afterwards), then the ds_swizzle
    // broadcast to the accumulator rows (C rows 2j + hi: slot 2j of each 16-lane half).
    for t in 0..tn {
        let h = hr(g, p, t);
        if p == 1 {
            op(
                b,
                format!("v_lshrrev_b32_e32 v{h}, 16, v{h}"),
                &[v(h)],
                &[v(h)],
            )?;
        }
        op(
            b,
            format!("v_cvt_f32_f16_e64 v{}, v{h}.l", HF + t),
            &[v(HF + t)],
            &[v(h)],
        )?;
        for j in 0..8u8 {
            let text = format!(
                "ds_swizzle_b32 v{}, v{} offset:swizzle(BROADCAST,16,{})",
                SCF + 8 * t + j,
                HF + t,
                2 * j
            );
            b.ds_crosslane(
                Instruction::new(text, vec![v(SCF + 8 * t + j)], vec![v(HF + t)])
                    .memory(MemoryClass::DsLoad),
            )?;
        }
    }
    for t in 0..tn {
        for c in 0..16u8 {
            let n = (usize::from(t) * 16 + usize::from(c)) % 2;
            let r = raw(g, p, t, c);
            unpack(b, r, r, tmp(g) + n as u8)?;
        }
    }
    for i in 0..8u8 {
        for t in 0..tn {
            iu4_fold::wmma_step(
                b,
                a,
                V::<8>(CACC + 8 * t),
                V::<2>(raw(g, p, t, 2 * i)),
                V::<2>(xr(g, p) + 2 * i),
                i == 0,
                V::<8>(MAGIC8),
            )?;
        }
    }
    for t in 0..tn {
        iu4_fold::fold_pass(b, CACC + 8 * t, SUM + 8 * t, SCF + 8 * t, dr(g, p), TPROD)?;
    }
    Ok(())
}

fn kloop(wg: &mut Wg<Gfx1151>, g: &Gen) -> Result<(), String> {
    wg.label(&g.label("k_begin"))?;
    let (head, done) = (g.label("k_loop"), g.label("k_loop_end"));
    let entry = wg.isa().ledger.shape();
    let no_trip = trips_cmp(wg, "s_cmp_eq_u32", TRIPS)?;
    wg.wg_skip_if(no_trip, &done, (), |wg, ()| {
        wg.loop_carried(&head, (), |wg, ()| {
            let b = wg.isa();
            load_set(b, g, 1)?;
            compute(b, g, 0)?;
            advance(b, g)?;
            load_set(b, g, 0)?;
            compute(b, g, 1)?;
            op(
                b,
                format!("{} s{TRIPS}, s{TRIPS}, -1", s_add_i32(b.spec.arch)),
                &[s(TRIPS)],
                &[s(TRIPS)],
            )?;
            Ok(((), trips_cmp(wg, "s_cmp_lg_u32", TRIPS)?))
        })?;
        if wg.isa().ledger.shape() != entry {
            return Err("k loop exit ledger differs from its entry".into());
        }
        Ok(())
    })?;
    match g.spec.kind {
        // K % 256 == 0: an even epoch count, two epochs left.
        Kind::GateUp => {
            let b = wg.isa();
            load_set(b, g, 1)?;
            compute(b, g, 0)?;
            compute(b, g, 1)
        }
        // K % 128 == 0: one epoch left when K/128 is odd, else two (the MQ4 down entry's branch).
        Kind::Down => {
            let odd = wg.scmp(Instruction::new(
                format!("s_bitcmp1_b32 s{}, 7", ARGS1 + 3),
                vec![],
                vec![s(ARGS1 + 3)],
            ))?;
            wg.if_else(
                odd,
                &g.label("tail_odd"),
                &g.label("epilogue"),
                |w| {
                    let b = w.isa();
                    load_set(b, g, 1)?;
                    compute(b, g, 0)?;
                    compute(b, g, 1)
                },
                |w| compute(w.isa(), g, 0),
            )
        }
    }
}

/// Raw single-dword buffer store at `v{voff} + offset` (soffset zero; `common` has the b128 one only).
fn bstore_b32(b: &mut Builder, data: u8, voff: u8, srd: u8, offset: u32) -> Result<(), String> {
    if offset > b.spec.arch.buffer_offset_max() {
        return Err(format!("buffer offset {offset} exceeds the field"));
    }
    let imm = if offset == 0 {
        String::new()
    } else {
        format!(" offset:{offset}")
    };
    common::mem(
        b,
        format!(
            "buffer_store_b32 v{data}, v{voff}, s[{srd}:{}], 0 offen{imm}",
            srd + 3
        ),
        &[],
        &[v(voff), v(data), common::sr(srd, 4)],
        MemoryClass::VmemStore,
    )
}

/// F32 accumulators (`Y[P][M]` f32): lane (hi, lr) holds rows `rbase + 2j + hi` of token `lr` and
/// stores accumulator `j` as one dword at `+ 8j` for each tensor (gate/up: the up rows M/2 columns
/// further); padding stores +0.
fn epilogue_acc(b: &mut Builder, g: &Gen) -> Result<(), String> {
    let tn = g.t();
    b.label(&g.label("epilogue"))?;
    for t in 0..tn {
        for j in 0..8u8 {
            let r = SUM + 8 * t + j;
            op(
                b,
                format!("v_cndmask_b32_e64 v{r}, 0, v{r}, s{LIVE}"),
                &[v(r)],
                &[v(r), s(LIVE)],
            )?;
        }
    }
    sop(
        b,
        format!("s_mov_b32 s{SRD_Y}, s{ARGS1}"),
        &[SRD_Y],
        &[ARGS1],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_Y + 1, ARGS1 + 1),
        &[SRD_Y + 1],
        &[ARGS1 + 1],
    )?;
    srd_tail(b, SRD_Y, None)?;
    if tn == 2 {
        op(
            b,
            format!("v_add_nc_u32_e32 v{RT_TMP}, s{UPOFF}, v{YOFF}"),
            &[v(RT_TMP)],
            &[s(UPOFF), v(YOFF)],
        )?;
    }
    for (t, voff) in [(0u8, YOFF), (1u8, RT_TMP)]
        .into_iter()
        .take(usize::from(tn))
    {
        for j in 0..8u8 {
            bstore_b32(b, SUM + 8 * t + j, voff, SRD_Y, 8 * u32::from(j))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(arch: Arch, kind: Kind) -> Spec {
        Spec {
            arch,
            kind,
            nt: 1,
            rr: 1,
            codec: WeightCodec::Mq3I4,
        }
    }
    fn count(text: &str, needle: &str) -> usize {
        text.lines().filter(|l| l.contains(needle)).count()
    }

    /// The register file is the MQ4 gfx11 operand set plus two registers; the chunk offsets tile
    /// the 48-byte epoch group.
    #[test]
    fn layout_extends_the_mq4_gfx11_operand_set() {
        for (kind, vgprs) in [(Kind::GateUp, 200u16), (Kind::Down, 168)] {
            for acc in [false, true] {
                let g = Gen {
                    spec: spec(Arch::Gfx1151, kind),
                    acc,
                };
                assert_eq!(
                    plan(&g).unwrap().next_free_vgpr(),
                    vgprs,
                    "{kind:?} acc={acc}"
                );
                // Operand set `p` is the MQ4 gfx11 set: A per tensor, X, headers, d.
                for p in 0..2usize {
                    assert_eq!(raw(&g, p, 0, 0), g.a(p, 0));
                    assert_eq!(xr(&g, p), g.x(p));
                    for t in 0..g.t() {
                        assert_eq!(raw(&g, p, t, 0), g.a(p, t));
                        assert_eq!(hr(&g, p, t), g.h(p, t));
                    }
                    assert_eq!(dr(&g, p), g.d(p));
                }
                assert_eq!(tmp(&g), SET0 + 2 * g.set_len());
            }
        }
        assert_eq!(
            (0..16u8).map(chunk_off).collect::<Vec<_>>(),
            (0..16u32).map(|c| 3 * c).collect::<Vec<_>>()
        );
        assert!(chunk_off(15) + 4 <= 49);
    }

    /// Both entries and both twins emit, deterministically, with twice the gfx1201 body's K16
    /// WMMA steps and chunk unpacks (same epoch structure), no ttmp ids (s2/s3 carry them), the
    /// shared BF16 lane exchange only in the shipped entry and per-dword stores only in `_acc`.
    #[test]
    fn entries_emit_with_the_k16_chain_and_gfx11_conventions() {
        for kind in [Kind::GateUp, Kind::Down] {
            let tn = usize::from(kind.tensors());
            for acc in [false, true] {
                let e = emit_gfx1151(spec(Arch::Gfx1151, kind), acc).unwrap();
                let again = emit_gfx1151(spec(Arch::Gfx1151, kind), acc).unwrap();
                assert_eq!(e.proof.s_text_sha256, again.proof.s_text_sha256);
                let w12 = super::super::emit_impl(spec(Arch::Gfx1201, kind), acc).unwrap();
                let sfx = if acc { "_acc" } else { "" };
                assert!(e
                    .s_text
                    .contains(&format!("{}{sfx}", spec(Arch::Gfx1151, kind).symbol())));
                assert!(!e.s_text.contains("ttmp"));
                assert!(!e.s_text.contains("v_wmma_i32_16x16x32_iu4"));
                for (needle12, needle11) in [
                    ("v_wmma_i32_16x16x32_iu4", "v_wmma_i32_16x16x16_iu4"),
                    ("0x88888888", "0x88888888"),
                    ("0x88888888", "v_lshl_or_b32"),
                ] {
                    let (n12, n11) = (count(&w12.s_text, needle12), count(&e.s_text, needle11));
                    assert!(
                        n12 > 0 && n11 == 2 * n12,
                        "{kind:?} acc={acc} {needle11}: gfx1201 {n12}, gfx1151 {n11}"
                    );
                }
                assert_eq!(
                    count(&e.s_text, "buffer_store_b32"),
                    if acc { 8 * tn } else { 0 }
                );
                assert_eq!(e.s_text.contains("v_permlanex16_b32"), !acc);
            }
        }
    }
}
