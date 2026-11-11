// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx11 prologue (B4-B): native workgroup ids (`s2`/`s3`; gfx1100 `s14`/`s15`, see `Gen::wg_in`), tile and expert
//! gating, exact-extent buffer descriptors, lane offsets, slot gather, and the
//! K-loop constants. Nothing is loaded from the weights here.
//!
//! Descriptor extents (every one is `tensor end - descriptor base`, 32-bit,
//! admission-checked to fit; every pointer addition is 64-bit with carry):
//!
//! | descriptor | base | `num_records` |
//! |---|---|---|
//! | codes, tensor 0 | `wptr + rbase*ROWB` | `TOTB - rbase*ROWB` |
//! | codes, up (GateUp) | codes0 base `+ (M/2)*ROWB` | codes0 records `- (M/2)*ROWB` |
//! | scales, tensor 0 (W2/W3) | `wptr + M*ROWB + rbase*SROW` | `TOTB - (M*ROWB + rbase*SROW)` |
//! | scales, up (W2/W3) | scales0 base `+ (M/2)*SROW` | scales0 records `- (M/2)*SROW` |
//! | X | `Xq` | `(K/128) * x_src_rows * 72` |
//! | sorted slots | `sorted + 64*tile` | 64 |
//!
//! `ROWB` is the physical code row bytes (MQ4 `K/128*68`, W3 `K*3/8`, W2
//! `K/4`), `SROW = K/64` (planar scale row, `K/128` f16), `TOTB = M*ROWB`
//! (MQ4) or `M*(ROWB + SROW)` (planar). MQ4 scales live in the code rows, so
//! its scale loads reuse the code descriptors. Because the code descriptor of
//! a planar tensor runs to the tensor end, W3's last unaligned dword
//! (`epoch bytes 45..49`) reads at most the first scale-plane byte of the same
//! tensor, and W2's `4i+2` dword at most two bytes past a row: both inside the
//! extent.
use super::*;

fn cmp<T: Gfx11>(wg: &mut Wg<'_, T>, text: String, uses: &[u8]) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(Instruction::new(
        text,
        vec![],
        uses.iter().map(|&n| s(n)).collect(),
    ))
}

pub(super) fn emit<T: Gfx11>(wg: &mut Wg<'_, T>, g: &Gen, end: &End) -> Result<(), String> {
    let a = g.spec.arch;
    let kind = g.spec.kind;
    let codec = g.spec.codec;
    let planar = codec != Codec::Mq4;
    let gate_up = kind == Kind::GateUp;
    let neg = g.spec.negative;
    let b = wg.isa();
    let args0 = g.args0();
    smem(b, args0, 8, KARG, 0)?;
    smem(b, ARGS1, 2, KARG, 0x20)?;
    smem(b, M, 2, KARG, 0x28)?;
    smem(b, ARGS2, 2, KARG, 0x30)?;
    smem(b, ARGS3, 1, KARG, 0x38)?;
    // gfx11 native workgroup ids (after the declared user SGPRs: s2/s3, gfx1100 s14/s15).
    let (wg_in_x, wg_in_y) = (g.wg_in(), g.wg_in() + 1);
    sop(
        b,
        format!("s_mov_b32 s{WGX}, s{wg_in_x}"),
        &[WGX],
        &[wg_in_x],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{WGY}, s{wg_in_y}"),
        &[WGY],
        &[wg_in_y],
    )?;
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
    // E8P: LDS stage store addresses are derived from the work-item id before the first branch.
    if g.spec.codec == Codec::E8p {
        e8p::stage_addresses(b)?;
    }
    // A tile at or past m_total, or a sentinel expert tile, leaves before any load or store.
    sop(
        b,
        format!("s_lshl_b32 s{TILE16}, s{WGY}, 4"),
        &[TILE16],
        &[WGY],
    )?;
    let outside = cmp(
        wg,
        format!("s_cmp_ge_i32 s{TILE16}, s{}", ARGS2 + 1),
        &[TILE16, ARGS2 + 1],
    )?;
    wg.exit_if(outside, end)?;
    let b = wg.isa();
    sop(b, format!("s_lshl_b32 s{T0}, s{WGY}, 2"), &[T0], &[WGY])?;
    add64(b, T64, args0 + 2, T0)?;
    smem(b, EID, 1, T64, 0)?;
    // Fail closed: an expert id outside [0, n_experts) (the -1 sentinel, or a corrupt / non-finite-router id) exits
    // before any `expert_weight_ptrs[id]` load or table prefetch. One unsigned compare covers both bounds (negative ids are huge).
    // The guard tests the routing id `EID` itself, so `WrongExpertPointer` (which only replaces the pointer index below) is guarded too.
    smem(b, T1, 1, KARG, 0x3c)?;
    let sentinel = cmp(wg, format!("s_cmp_ge_u32 s{EID}, s{T1}"), &[EID, T1])?;
    wg.exit_if(sentinel, end)?;
    let b = wg.isa();
    // Expert pointer: `expert_weight_ptrs[e]`, a 64-bit load and 64-bit add.
    // WrongExpertPointer deliberately reads entry 0 (always in range).
    if neg == Negative::WrongExpertPointer {
        sop(b, format!("s_mov_b32 s{T0}, 0"), &[T0], &[])?;
    } else {
        sop(b, format!("s_lshl_b32 s{T0}, s{EID}, 3"), &[T0], &[EID])?;
    }
    add64(b, T64, args0, T0)?;
    smem(b, WPTR, 2, T64, 0)?;
    // Row geometry.
    match codec {
        Codec::Mq4 => {
            sop(b, format!("s_lshr_b32 s{ROWB}, s{K}, 7"), &[ROWB], &[K])?;
            sop(
                b,
                format!("s_mul_i32 s{ROWB}, s{ROWB}, 0x44"),
                &[ROWB],
                &[ROWB],
            )?;
        }
        Codec::W2 | Codec::E8p => sop(b, format!("s_lshr_b32 s{ROWB}, s{K}, 2"), &[ROWB], &[K])?,
        Codec::W3 => {
            sop(b, format!("s_mul_i32 s{ROWB}, s{K}, 3"), &[ROWB], &[K])?;
            sop(
                b,
                format!("s_lshr_b32 s{ROWB}, s{ROWB}, 3"),
                &[ROWB],
                &[ROWB],
            )?;
        }
    }
    if planar {
        sop(b, format!("s_lshr_b32 s{SROW}, s{K}, 6"), &[SROW], &[K])?;
    }
    sop(
        b,
        format!("s_mul_i32 s{CBYTES}, s{M}, s{ROWB}"),
        &[CBYTES],
        &[M, ROWB],
    )?;
    if planar {
        sop(
            b,
            format!("s_mul_i32 s{T0}, s{M}, s{SROW}"),
            &[T0],
            &[M, SROW],
        )?;
        sop(
            b,
            format!("{} s{TOTB}, s{CBYTES}, s{T0}", s_add_u32(a)),
            &[TOTB],
            &[CBYTES, T0],
        )?;
    } else {
        sop(
            b,
            format!("s_mov_b32 s{TOTB}, s{CBYTES}"),
            &[TOTB],
            &[CBYTES],
        )?;
    }
    // Epoch count and first epoch index.
    sop(b, format!("s_lshr_b32 s{ECNT}, s{K}, 7"), &[ECNT], &[K])?;
    if neg == Negative::ReversedEpoch {
        sop(
            b,
            format!("s_add_i32 s{EIDX}, s{ECNT}, -1"),
            &[EIDX],
            &[ECNT],
        )?;
    } else {
        sop(b, format!("s_mov_b32 s{EIDX}, 0"), &[EIDX], &[])?;
    }
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
    add64(b, SRD_S, args0 + 4, T0)?;
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
    // Activation descriptor: all K/128 epochs, epoch stride x_src_rows*72.
    sop(
        b,
        format!("s_mul_i32 s{XSTR}, s{ARGS3}, 0x48"),
        &[XSTR],
        &[ARGS3],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{SRD_X}, s{}", args0 + 6),
        &[SRD_X],
        &[args0 + 6],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_X + 1, args0 + 7),
        &[SRD_X + 1],
        &[args0 + 7],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{T0}, s{ECNT}, s{XSTR}"),
        &[T0],
        &[ECNT, XSTR],
    )?;
    srd_tail(b, SRD_X, Some(T0))?;
    // Code descriptors: rows rbase.. of the expert; records run to the tensor end.
    sop(
        b,
        format!("s_mul_i32 s{T0}, s{RBASE}, s{ROWB}"),
        &[T0],
        &[RBASE, ROWB],
    )?;
    add64(b, SRD_W, WPTR, T0)?;
    sop(
        b,
        format!("s_sub_u32 s{}, s{TOTB}, s{T0}", SRD_W + 2),
        &[SRD_W + 2],
        &[TOTB, T0],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, {}", SRD_W + 3, lit(SRD_WORD3)),
        &[SRD_W + 3],
        &[],
    )?;
    if gate_up {
        sop(b, format!("s_lshr_b32 s{T1}, s{M}, 1"), &[T1], &[M])?;
        sop(
            b,
            format!("s_mul_i32 s{T1}, s{T1}, s{ROWB}"),
            &[T1],
            &[T1, ROWB],
        )?;
        add64(b, SRD_U, SRD_W, T1)?;
        sop(
            b,
            format!("s_sub_u32 s{}, s{}, s{T1}", SRD_U + 2, SRD_W + 2),
            &[SRD_U + 2],
            &[SRD_W + 2, T1],
        )?;
        sop(
            b,
            format!("s_mov_b32 s{}, {}", SRD_U + 3, lit(SRD_WORD3)),
            &[SRD_U + 3],
            &[],
        )?;
    }
    // Scale descriptors of the planar codecs: the scale plane starts M*ROWB after the tensor base.
    if planar {
        sop(
            b,
            format!("s_mul_i32 s{T0}, s{RBASE}, s{SROW}"),
            &[T0],
            &[RBASE, SROW],
        )?;
        sop(
            b,
            format!("{} s{T0}, s{T0}, s{CBYTES}", s_add_u32(a)),
            &[T0],
            &[T0, CBYTES],
        )?;
        add64(b, SRD_WS, WPTR, T0)?;
        sop(
            b,
            format!("s_sub_u32 s{}, s{TOTB}, s{T0}", SRD_WS + 2),
            &[SRD_WS + 2],
            &[TOTB, T0],
        )?;
        sop(
            b,
            format!("s_mov_b32 s{}, {}", SRD_WS + 3, lit(SRD_WORD3)),
            &[SRD_WS + 3],
            &[],
        )?;
        if gate_up {
            sop(b, format!("s_lshr_b32 s{T1}, s{M}, 1"), &[T1], &[M])?;
            sop(
                b,
                format!("s_mul_i32 s{T1}, s{T1}, s{SROW}"),
                &[T1],
                &[T1, SROW],
            )?;
            add64(b, SRD_US, SRD_WS, T1)?;
            sop(
                b,
                format!("s_sub_u32 s{}, s{}, s{T1}", SRD_US + 2, SRD_WS + 2),
                &[SRD_US + 2],
                &[SRD_WS + 2, T1],
            )?;
            sop(
                b,
                format!("s_mov_b32 s{}, {}", SRD_US + 3, lit(SRD_WORD3)),
                &[SRD_US + 3],
                &[],
            )?;
        }
    }
    // Lane offsets. Codes: lr*ROWB (no hi term: both half waves carry all 16 K16 operands).
    op(
        b,
        format!("v_mul_u32_u24_e32 v{WOFF}, s{ROWB}, v3"),
        &[v(WOFF)],
        &[s(ROWB), v(3)],
    )?;
    // Scale row of this lane's broadcast slot: gfx11 `(lr & 14) | hi`.
    // WrongScaleRow deliberately uses the gfx12 row `8*hi + (lr & 7)`.
    if neg == Negative::WrongScaleRow {
        op(b, "v_and_b32_e32 v7, 7, v3", &[v(7)], &[v(3)])?;
        op(b, "v_lshl_or_b32 v7, v4, 3, v7", &[v(7)], &[v(4), v(7)])?;
    } else {
        op(b, "v_and_b32_e32 v7, 14, v3", &[v(7)], &[v(3)])?;
        op(b, "v_or_b32_e32 v7, v7, v4", &[v(7)], &[v(7), v(4)])?;
    }
    let scale_row_bytes = if planar { SROW } else { ROWB };
    op(
        b,
        format!("v_mul_u32_u24_e32 v{HOFF}, s{scale_row_bytes}, v7"),
        &[v(HOFF)],
        &[s(scale_row_bytes), v(7)],
    )?;
    // BF16 Y offset: ((16*tile + lr)*row_len + rbase + 8*hi)*2, row_len = M/2 (GateUp) or M.
    if gate_up {
        sop(b, format!("s_lshr_b32 s{T1}, s{M}, 1"), &[T1], &[M])?;
    } else {
        sop(b, format!("s_mov_b32 s{T1}, s{M}"), &[T1], &[M])?;
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
    // Activation row offset (padding slots get the gather's out-of-range offset and LIVE).
    gather_offset(
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
    for j in 0..8u8 {
        op(
            b,
            format!("v_mov_b32_e32 v{}, {}", MAGIC8 + j, lit(MAGIC)),
            &[v(MAGIC8 + j)],
            &[],
        )?;
    }
    for t in 0..g.tn() {
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
