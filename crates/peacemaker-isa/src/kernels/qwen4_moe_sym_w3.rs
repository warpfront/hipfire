// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `WeightCodec::Mq3I4` NT1 entries of the grouped symmetric IU4 MoE GEMM
//! (gfx1201; the gfx1151 body is `qwen4_moe_sym_w3_gfx1151.rs`, dispatched from `emit_impl`):
//! gate/up `qwen4_moe_gate_up_silu_iu4_sym_w3_pm_gfx1201` (R256, K2560)
//! and down `qwen4_moe_down_iu4_sym_w3_pm_gfx1201` (R128, K640).
//!
//! Same 64-byte ABI (60 bytes plus a trailing `n_experts`), grid `(M/cta_rows, m_total/16)`, A4 activations, WMMA chain,
//! fold and epilogue as the MQ4 NT1 entries. Only the weight side differs.
//! `expert_weight_ptrs[e]` is the planar qt131/qt132 tensor `[M, K]`:
//! `[codes: M*K*3/8 B][scales: M*(K/128) f16]`, code `j` of a row at bits
//! `3j..3j+3` of the row's little-endian bitstream, weight `d * (2c - 7)`.
//!
//! Per K128 epoch every wave loads its 16 rows' 48-byte code groups (and the
//! scale pair of the trip) straight into registers. A WMMA lane (`hi`, row
//! `lr`) owns, for step `i`, the 8-code chunks `4i + 2hi` and `4i + 2hi + 1`
//! (the same 16 elements the MQ4 entry feeds it). A chunk is 24 bits at byte
//! `12i + 6hi` (+3 for the second chunk) of the epoch's 48-byte group, so each
//! lane loads one (unaligned) dword per chunk and uses its low 24 bits: no funnel
//! shifts. The chunk is then spread to eight 4-bit slots with three bit-field
//! inserts and finished to the signed nibble `(2c - 7) & 15 = ((2c + 1) ^ 8)`;
//! the exact int32 WMMA chain, the magic-seeded fold
//! `sum = fma(RN(d_act * d_w), float(dot), sum)` and the BF16 epilogue (SiLU for
//! gate/up) are the MQ4 entries' own. Junk above bit 23 of a chunk never reaches
//! an output (see `spread_model` and its exhaustive test). The down entry's K640
//! has an odd epoch count: its tail is the MQ4 down entry's `if_else`.
use super::*;

/// gfx1151 sibling (own file); everything below is the gfx1201 body.
#[path = "qwen4_moe_sym_w3_gfx1151.rs"]
mod gfx1151;

/// The NT4 (four 16-slot tiles per weight stream) entries of this codec: a
/// separate module and code object, the NT1 entries here are unchanged.
#[path = "qwen4_moe_sym_w3_nt4.rs"]
pub mod nt4;
// VGPRs added over the MQ4 layout.
const SET_LEN: u8 = 28;
const AOUT: u8 = SET0 + 2 * SET_LEN; // 148: unpacked signed nibbles, 8 dwords per tensor
const TMPN: u8 = AOUT + 16; // 164: four unpack temporaries
const ONES: u8 = TMPN + 4; // 168: 0x11111111

// SGPRs added over the MQ4 layout.
const SRD_WS: u8 = 68;
const SRD_US: u8 = 72;
const SROW: u8 = 76;
const UPOFF: u8 = 78; // gate/up acc entry: (M/2)*4, the up-row byte offset

fn set_base(p: usize) -> u8 {
    SET0 + p as u8 * SET_LEN
}
/// Code dword of step `i` (`which` 0 = chunk 4i+2hi, 1 = chunk 4i+2hi+1) of tensor `t`.
fn raw(p: usize, t: u8, i: u8, which: u8) -> u8 {
    set_base(p) + 8 * t + 2 * i + which
}
fn xr(p: usize) -> u8 {
    set_base(p) + 16
}
fn hr(p: usize, t: u8) -> u8 {
    set_base(p) + 24 + t
}
fn dr(p: usize) -> u8 {
    set_base(p) + 26
}
fn a_out(t: u8, i: u8) -> u8 {
    AOUT + 8 * t + 2 * i
}
fn srd_c(t: u8) -> u8 {
    if t == 0 {
        SRD_W
    } else {
        SRD_U
    }
}
fn srd_s(t: u8) -> u8 {
    if t == 0 {
        SRD_WS
    } else {
        SRD_US
    }
}

/// CPU model of the chunk spread (`x` = the chunk's dword; bits above 23 are
/// junk). Returns eight signed nibbles `(2c - 7) & 15`, code `k` in nibble `k`.
pub fn spread_model(x: u32) -> u32 {
    let bfi = |m: u32, a: u32, b: u32| (a & m) | (b & !m);
    let t = bfi(0xfff, x, x << 4);
    let t = bfi(0x003f_003f, t, t << 2);
    let t = bfi(0x0707_0707, t, t << 1);
    let s = t << 1;
    bfi(0x1111_1111, 0x1111_1111, s) ^ 0x8888_8888
}

pub(super) fn plan(g: &Gen, acc: bool) -> Result<RegPlan, String> {
    let tn = g.t();
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
    for (name, r) in [
        ("x_off", XOFF),
        ("xq_off", XQOFF),
        ("w_off", WOFF),
        ("h_off", HOFF),
    ] {
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
    for set in 0..2usize {
        for t in 0..tn {
            for k in 0..8u8 {
                p.v::<1>("code_dword", raw(set, t, k / 2, k % 2), kernel())?;
            }
        }
        for chunk in 0..4u8 {
            p.v::<2>("operand", xr(set) + 2 * chunk, kernel())?;
        }
        for t in 0..tn {
            p.v::<1>("scale_pair", hr(set, t), kernel())?;
        }
        p.v::<1>("d", dr(set), kernel())?;
    }
    for t in 0..tn {
        for i in 0..4u8 {
            p.v::<2>("a_nibbles", a_out(t, i), kernel())?;
        }
    }
    for i in 0..4u8 {
        p.v::<1>("unpack_tmp", TMPN + i, kernel())?;
    }
    p.v::<1>("ones", ONES, kernel())?;
    // Epilogue registers over the dead K-loop ranges.
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
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    p.s::<4>("srd_y", SRD_Y, epi())?;
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

/// The shipped NT1 entry (BF16 epilogue).
pub(super) fn emit(spec: Spec) -> Result<Emitted, String> {
    emit_impl(spec, false)
}

/// Qualification entry `..._acc`: the same K loop, but stores the 16 rows' F32
/// accumulators (`Y[P][M]` f32; gate/up: gate rows then up rows; padding slots +0)
/// instead of the BF16 epilogue, so G1 compares the exact fold bytes against the
/// CPU twin. Never dispatched.
pub fn emit_acc(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    emit_impl(spec, true)
}

fn emit_impl(spec: Spec, acc: bool) -> Result<Emitted, String> {
    if spec.arch == Arch::Gfx1151 {
        return gfx1151::emit_gfx1151(spec, acc);
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
    let mut b = Builder::new(kspec, plan(&g, acc)?);
    b.enable_delay_alu();
    let mut wg = Workgroup::<Gfx1201, Builder>::new(&mut b)?;
    body(&mut wg, &g, acc)?;
    b.finish()
}

fn body(wg: &mut Wg<Gfx1201>, g: &Gen, acc: bool) -> Result<(), String> {
    let end = wg.exit(&g.label("end"))?;
    prologue(wg, g, &end, acc)?;
    kloop(wg, g)?;
    if acc {
        epilogue_acc(wg.isa(), g)?
    } else {
        epilogue(wg.isa(), g)?
    }
    wg.end(end)
}

fn prologue(wg: &mut Wg<Gfx1201>, g: &Gen, end: &End, acc: bool) -> Result<(), String> {
    let a = g.arch();
    let gate_up = g.spec.kind == Kind::GateUp;
    let b = wg.isa();
    smem(b, ARGS0, 8, KARG, 0)?;
    smem(b, ARGS1, 2, KARG, 0x20)?;
    smem(b, ARGS1 + 2, 2, KARG, 0x28)?;
    smem(b, ARGS2, 2, KARG, 0x30)?;
    smem(b, ARGS3, 1, KARG, 0x38)?;
    sop(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[WGX], &[])?;
    sop(b, format!("s_and_b32 s{WGY}, ttmp7, 0xffff"), &[WGY], &[])?;
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
    // Lane offsets. Codes: lr*row_bytes + 6*hi (the lane's first 8-code chunk of a K32 step).
    op(
        b,
        format!("v_mul_u32_u24_e32 v{WOFF}, s{ROWB}, v3"),
        &[v(WOFF)],
        &[s(ROWB), v(3)],
    )?;
    op(
        b,
        format!("v_mad_u32_u24 v{WOFF}, v4, 6, v{WOFF}"),
        &[v(WOFF)],
        &[v(4), v(WOFF)],
    )?;
    // Scale row: the row whose scale this lane's broadcast slot serves.
    op(b, "v_and_b32_e32 v7, 7, v3", &[v(7)], &[v(3)])?;
    op(b, "v_lshl_or_b32 v7, v4, 3, v7", &[v(7)], &[v(4), v(7)])?;
    op(
        b,
        format!("v_mul_u32_u24_e32 v{HOFF}, s{SROW}, v7"),
        &[v(HOFF)],
        &[s(SROW), v(7)],
    )?;
    // Y offset: ((16*tile + lr)*row_len + rbase + 8*hi)*2 (BF16) or *4 (F32 accumulators).
    // row_len is M/2 for the BF16 gate/up H and M otherwise.
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
    op(
        b,
        format!("v_lshl_add_u32 v9, v4, 3, s{RBASE}"),
        &[v(9)],
        &[v(4), s(RBASE)],
    )?;
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
    op(
        b,
        format!("v_lshl_add_u32 v{XQOFF}, v4, 3, v{XOFF}"),
        &[v(XQOFF)],
        &[v(4), v(XOFF)],
    )?;
    load_set(b, g, 0)?;
    op(
        b,
        format!("v_mov_b32_e32 v{ONES}, 0x11111111"),
        &[v(ONES)],
        &[],
    )?;
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

/// Issue set `p`'s loads (epoch parity `p` of the current trip): the trip's
/// scale pair of each tensor's header row, `d`, the activation nibbles, then
/// one unaligned dword per 8-code chunk (byte `12i + 6hi` and `+ 3` of the
/// epoch's 48-byte group), as one clause.
fn load_set(b: &mut Builder, g: &Gen, p: usize) -> Result<(), String> {
    let tn = g.t();
    b.clause(|b| {
        for t in 0..tn {
            bload(b, 1, hr(p, t), HOFF, srd_s(t), 0)?;
        }
        bload(b, 1, dr(p), XOFF, SRD_X[p], 0)?;
        for i in 0..4u8 {
            bload(b, 2, xr(p) + 2 * i, XQOFF, SRD_X[p], 8 + 16 * u32::from(i))?;
        }
        for t in 0..tn {
            for i in 0..4u8 {
                let off = 48 * p as u32 + 12 * u32::from(i);
                bload(b, 1, raw(p, t, i, 0), WOFF, srd_c(t), off)?;
                bload(b, 1, raw(p, t, i, 1), WOFF, srd_c(t), off + 3)?;
            }
        }
        Ok(())
    })
}

/// One chunk: the dword at the chunk's byte offset (its low 24 bits are the 8
/// codes), spread and sign-finished into `dst`.
fn unpack(b: &mut Builder, dst: u8, raw: u8, tmp: u8) -> Result<(), String> {
    let (x, y) = (tmp, tmp + 1);
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
    shl(b, y, 4, raw)?;
    bfi(b, y, 0xfff, raw, y)?;
    shl(b, x, 2, y)?;
    bfi(b, y, 0x003f_003f, y, x)?;
    shl(b, x, 1, y)?;
    bfi(b, y, 0x0707_0707, y, x)?;
    shl(b, x, 1, y)?;
    op(
        b,
        format!("v_bfi_b32 v{y}, 0x11111111, v{ONES}, v{x}"),
        &[v(y)],
        &[v(ONES), v(x)],
    )?;
    op(
        b,
        format!("v_xor_b32_e32 v{dst}, {}, v{y}", lit(REBIAS)),
        &[v(dst)],
        &[v(y)],
    )
}

/// Fold set `p` (one K128 epoch) into the sums.
fn compute(b: &mut Builder, g: &Gen, p: usize) -> Result<(), String> {
    let a = g.arch();
    let tn = g.t();
    // Row scales: epoch parity selects the low (even) or high (odd) half of
    // the scale pair, then the ds_swizzle broadcast to the accumulator rows.
    let half = if p == 0 { "l" } else { "h" };
    for t in 0..tn {
        op(
            b,
            format!("v_cvt_f32_f16_e64 v{}, v{}.{half}", HF + t, hr(p, t)),
            &[v(HF + t)],
            &[v(hr(p, t))],
        )?;
        for j in 0..8u8 {
            let text = format!(
                "ds_swizzle_b32 v{}, v{} offset:swizzle(BROADCAST,16,{j})",
                SCF + 8 * t + j,
                HF + t
            );
            b.ds_crosslane(
                Instruction::new(text, vec![v(SCF + 8 * t + j)], vec![v(HF + t)])
                    .memory(MemoryClass::DsLoad),
            )?;
        }
    }
    for t in 0..tn {
        for i in 0..4u8 {
            for which in 0..2u8 {
                let n = (usize::from(t) * 8 + usize::from(i) * 2 + usize::from(which)) % 2;
                unpack(
                    b,
                    a_out(t, i) + which,
                    raw(p, t, i, which),
                    TMPN + 2 * n as u8,
                )?;
            }
        }
    }
    for i in 0..4u8 {
        for t in 0..tn {
            iu4_fold::wmma_step(
                b,
                a,
                V::<8>(CACC + 8 * t),
                V::<2>(a_out(t, i)),
                V::<2>(xr(p) + 2 * i),
                i == 0,
                V::<8>(MAGIC8),
            )?;
        }
    }
    for t in 0..tn {
        iu4_fold::fold_pass(b, CACC + 8 * t, SUM + 8 * t, SCF + 8 * t, dr(p), TPROD)?;
    }
    Ok(())
}

fn advance(b: &mut Builder, g: &Gen) -> Result<(), String> {
    for t in 0..g.t() {
        add64_imm(b, srd_c(t), 96)?;
        add64_imm(b, srd_s(t), 4)?;
    }
    for r in SRD_X {
        add64(b, r, r, XS2)?;
    }
    Ok(())
}

fn kloop(wg: &mut Wg<Gfx1201>, g: &Gen) -> Result<(), String> {
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

/// F32 accumulators (`Y[P][M]` f32): lane (hi, lr) stores rows `rbase + 8hi..+8`
/// of token `lr` for each tensor (gate/up: the up rows M/2 columns further);
/// padding stores +0.
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
        bstore_b128(b, SUM + 8 * t, voff, SRD_Y, 0)?;
        bstore_b128(b, SUM + 8 * t + 4, voff, SRD_Y, 16)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::spread_model;

    /// Every 24-bit chunk, with junk above bit 23, spreads to the eight signed
    /// nibbles `(2c - 7) & 15`.
    #[test]
    fn spread_matches_the_signed_w3_grid_for_every_chunk() {
        let mut junk = 0x9e37_79b9u32;
        for x in 0..(1u32 << 24) {
            junk = junk.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let got = spread_model(x | (junk & 0xff00_0000));
            let want = (0..8).fold(0u32, |acc, k| {
                acc | ((2 * ((x >> (3 * k)) & 7)).wrapping_sub(7) & 15) << (4 * k)
            });
            assert_eq!(got, want, "chunk {x:#08x}");
        }
    }
}
