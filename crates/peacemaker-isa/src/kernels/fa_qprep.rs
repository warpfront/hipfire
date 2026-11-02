// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201 Q-only full-attention prep for the A3B FA2-GQA prefill route (H = 16, Hkv = 2, D = 256, rotary 64),
//! emitted with the checked ISA builder. One export, `attention_fa2_qprep_pm_gfx1201`, in its own module
//! `fa_qprep_pm_gfx1201` (the incumbent `fa_prefill_pm_gfx1201` bundle is untouched). One launch replaces the
//! unfused sequence
//!
//!   `deinterleave_q_rmsnorm_f32_batched` -> `rope_partial_halfsplit_batched_f32` (Q part) ->
//!   `attention_fa2_q_preconvert_pm_gfx1201`
//!
//! and never writes the intermediate f32 Q: per `(row, head)` the f32 interleaved projection row is read once, the
//! gate half is copied through verbatim as f32, and the final f16 (RNE) Q16 row is stored.
//!
//! ABI (48 bytes): `interleaved` f32 `[rows, 16, 512]` @0 (per head: 256 Q then 256 gate floats), `q16` f16
//! `[rows, 16, 256]` @8, `gate` f32 `[rows, 16, 256]` @16, `q_norm` f32 `[256]` @24, `angles` f32 `[rows, 64]` @32
//! (per row `cos[0..32]`, `sin[32..64]`, the exact table the incumbent K/Q RoPE shares), `rows` i32 @40,
//! `eps` f32 @44. Grid `x = 4 * rows` blocks of 128 threads = four waves; wave `w` of block `b` owns the
//! `(row, head)` row `r = 4 b + w` (`row = r >> 4`, `head = r & 15`). Waves past `rows * 16`, and every launch with
//! `rows` outside `0..=8192`, exit before any access (no barrier, no LDS, no atomics).
//!
//! Lane `l` of a wave owns dims `l + 32 k`, `k = 0..8`: no LDS, no barrier, no DS drain.
//!
//! Arithmetic (every rounding point equals the incumbent three-kernel graph, which the HIP objects realise as):
//!
//!  * **Q RMS sum** - incumbent (`deinterleave_q_rmsnorm_f32_batched`, block 256, one element per thread): leaf
//!    `fma(q, q, +0)` (`v_fmac_f32`, accumulator `+0`), then the LDS halving tree
//!    `sdata[t] = sdata[t + s] + sdata[t]` for `s = 128, 64, 32, 16, 8, 4, 2, 1` (add is commutative: both
//!    operand orders round identically). Lane `l` holds leaves `t = l + 32 k`, so levels 128/64/32 are the same
//!    pairs added in-lane (`k, k + 4`; then `k, k + 2`; then `k = 0, 1`) and levels 16/8/4/2/1 are the lane XOR
//!    butterfly (`v_permlanex16_b32` with the identity selects for XOR 16, DPP `row_xmask` 8/4/2/1). Every lane
//!    ends with the root node `sdata[0]` bit for bit (a butterfly of commutative adds).
//!  * **mean** - the full-precision `v_div_scale` / `v_rcp` / Newton / `v_div_fmas` / `v_div_fixup` division of the
//!    sum by `256.0f` (`fa_prefill::fdiv`, op for op the hipcc graph), computed in VALU, then `readfirstlane`.
//!  * **rsqrtf(mean + eps)** - the scalar sequence hipcc emitted for the incumbent (cached disassembly of
//!    `deinterleave_q_rmsnorm_f32_batched`): `s_add_f32 eps + mean`; if `x < 0x00800000` (float compare) then
//!    `x * 2^24` (`0x4b800000`) replaces `x`; `v_s_rsq_f32`; and in the same case the result is multiplied by
//!    `2^12` (`0x45800000`). The scalar forms are kept (not rewritten to the vector `v_rsq_f32`).
//!  * **output** - `(q * w) * rms` with `w = q_norm[l + 32 k]` (two separate roundings, `q * w` first).
//!  * **RoPE** (dims `0..64`, pair `(i, i + 32)` = `(k = 0, k = 1)` of lane `i`): angles come from the captured
//!    table, `cos = angles[row][i]`, `sin = angles[row][32 + i]`; `o0 = fma(x0, cos, -(x1 * sin))`,
//!    `o1 = fma(x0, sin, x1 * cos)`; dims `64..256` pass through.
//!  * **Q16** - `v_cvt_f16_f32` (round to nearest even), one `global_store_b16` per dim (a wave writes 64
//!    contiguous bytes per `k`).
//!
//! The gate is stored as the verbatim f32 bits (`global_store_b32`, 128 contiguous bytes per `k`).

use super::fa_prefill::{cvt_half, fdiv, gload, wait_alu, FaPrefillSpec, MAX_BATCH};
use crate::insn::{Instruction, MemoryClass, Sop};
use crate::kernels::common::{add64, lit, op, s, sop, sr, v};
use crate::ledger::Counter;
use crate::plan::Access;
use crate::reg::Live;
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// CLI kernel id.
pub const KERNEL_ID: &str = "fa_qprep_pm";
/// Module / cache key.
pub const MODULE: &str = "fa_qprep_pm_gfx1201";
/// The export symbol.
pub const SYMBOL: &str = "attention_fa2_qprep_pm_gfx1201";
/// Block size: four waves, one `(row, head)` each.
pub const BLOCK: u32 = 128;
/// Kernarg segment bytes.
pub const KERNARG_BYTES: u32 = 48;
/// Heads per row (the one supported geometry).
pub const HEADS: u32 = 16;
/// Head dimension.
pub const HEAD_DIM: u32 = 256;
/// Rotary dimension (half-split pairs `(i, i + 32)`).
pub const N_ROT: u32 = 64;
/// Grid blocks per row (`16` heads, four per block).
pub const BLOCKS_PER_ROW: u32 = 4;

/// Exact spec admission: the same H16 / KV2 / D256 geometry as the incumbent twin (`fa_prefill`).
fn admit(arch: Arch, spec: &FaPrefillSpec) -> R {
    if arch != Arch::Gfx1201 {
        return Err("fa_qprep: built for gfx1201 only".into());
    }
    spec.validate()
}

/// Pure length-level launch validation: valid spec and `1 <= rows <= 8192`.
pub fn validate(spec: &FaPrefillSpec, rows: usize) -> Result<(), String> {
    spec.validate()?;
    if rows < 1 || rows > MAX_BATCH {
        return Err(format!("fa_qprep: rows {rows} is outside 1..={MAX_BATCH}"));
    }
    Ok(())
}

const END: &str = ".Lfaqp_end";

// ------------------------------------------------------------------ VGPR map
const TID: u8 = 0;
const LANE: u8 = 1;
const OFF4: u8 = 2; // lane * 4 (f32 element offset)
const OFF2: u8 = 3; // lane * 2 (f16 element offset)
const WAVE: u8 = 4; // tid >> 5 (readfirstlane source)
const Q: u8 = 8; // eight f32 Q values, dims lane + 32k
const SQ: u8 = 16; // squares / reduction tree, then the eight f16 results (low halves)
const W: u8 = 24; // eight q_norm weights, then the normalised / rotated values
const COS: u8 = 32;
const SIN: u8 = 33;
const SUM: u8 = 34;
const PEER: u8 = 35;
const DIV: [u8; 4] = [36, 37, 38, 39]; // fdiv temporaries
const QUOT: u8 = 40; // sum / 256
const T1: u8 = 41; // x1 * sin
const T2: u8 = 42; // x1 * cos
const G: u8 = 48; // eight f32 gate values
const VGPRS: u16 = 56;

// ------------------------------------------------------------------ SGPR map
const KARG: u8 = 0;
const PTRS: u8 = 8; // interleaved (8), q16 (10), gate (12), q_norm (14)
const P_IL: u8 = 8;
const P_Q16: u8 = 10;
const P_GATE: u8 = 12;
const P_QN: u8 = 14;
const TAIL: u8 = 16; // angles (16), rows (18), eps (19)
const S_ANG: u8 = 16;
const S_ROWS: u8 = 18;
const S_EPS: u8 = 19;
const S_WAVE: u8 = 20;
const S_R: u8 = 21; // 4 * block + wave = row * 16 + head
const S_LIM: u8 = 22; // rows * 16
const S_PSEL: u8 = 23; // identity permlanex16 selects
const QB: u8 = 24; // interleaved head row base
const GB: u8 = 26; // gate head row base
const OB: u8 = 28; // q16 head row base
const AB: u8 = 30; // angle row base
const T0: u8 = 32;
const T1S: u8 = 33;
const DEN: u8 = 34; // 256.0f
const MEAN: u8 = 35;
const RMS: u8 = 36;

fn plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(VGPRS, 104)?;
    let w = || Live::Whole;
    for (name, base) in [
        ("tid", TID),
        ("lane", LANE),
        ("off_f32", OFF4),
        ("off_f16", OFF2),
        ("wave", WAVE),
        ("cos", COS),
        ("sin", SIN),
        ("sum", SUM),
        ("peer", PEER),
        ("quotient", QUOT),
        ("x1_sin", T1),
        ("x1_cos", T2),
    ] {
        p.v::<1>(name, base, w())?;
    }
    for base in DIV {
        p.v::<1>("div_temp", base, w())?;
    }
    p.v::<8>("q", Q, w())?;
    p.v::<8>("squares_f16", SQ, w())?;
    p.v::<8>("weights_out", W, w())?;
    p.v::<8>("gate", G, w())?;
    p.s::<2>("kernarg_ptr", KARG, w())?;
    p.s::<8>("kernarg_ptrs", PTRS, w())?;
    p.s::<4>("angles_rows_eps", TAIL, w())?;
    for n in [S_WAVE, S_R, S_LIM, S_PSEL, T0, T1S, DEN, MEAN, RMS] {
        p.s::<1>("scalar", n, w())?;
    }
    for n in [QB, GB, OB, AB] {
        p.s::<2>("base_pair", n, w())?;
    }
    Ok(p)
}

/// Store of one f16 / f32 value: `global_store_<kind> v{voff}, v{data}, s[base:base+1] offset:imm`.
fn gstore(b: &mut Builder, kind: &str, voff: u8, data: u8, sbase: u8, imm: u32) -> R {
    let off = if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    };
    b.push(
        Instruction::new(
            format!(
                "global_store_{kind} v{voff}, v{data}, s[{sbase}:{}]{off}",
                sbase + 1
            ),
            vec![],
            vec![v(voff), v(data), sr(sbase, 2)],
        )
        .memory(MemoryClass::VmemStore),
    )
}

/// Wave-uniform exits and the per-wave `(row, head)` addresses.
fn prologue(b: &mut Builder) -> R {
    crate::kernels::common::smem(b, PTRS, 8, KARG, 0x0)?;
    crate::kernels::common::smem(b, TAIL, 4, KARG, 0x20)?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{WAVE}, 5, v{TID}"),
        &[v(WAVE)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{S_WAVE}, v{WAVE}"),
        &[s(S_WAVE)],
        &[v(WAVE)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{LANE}, 31, v{TID}"),
        &[v(LANE)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{OFF4}, 2, v{LANE}"),
        &[v(OFF4)],
        &[v(LANE)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{OFF2}, 1, v{LANE}"),
        &[v(OFF2)],
        &[v(LANE)],
    )?;
    b.wait(Counter::Km, 0)?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    // Exits (uniform, before any access): rows outside 0..=8192 (the byte offsets below are 32-bit), then the
    // wave's row past rows * 16 (`rows <= 0` falls here as well).
    sop(
        b,
        format!("s_cmp_gt_u32 s{S_ROWS}, {}", lit(MAX_BATCH as u32)),
        &[],
        &[S_ROWS],
    )?;
    op(b, format!("s_cbranch_scc1 {END}"), &[], &[])?;
    sop(
        b,
        format!("s_lshl_b32 s{S_LIM}, s{S_ROWS}, 4"),
        &[S_LIM],
        &[S_ROWS],
    )?;
    op(b, format!("s_lshl_b32 s{S_R}, ttmp9, 2"), &[s(S_R)], &[])?;
    sop(
        b,
        format!("s_add_co_i32 s{S_R}, s{S_R}, s{S_WAVE}"),
        &[S_R],
        &[S_R, S_WAVE],
    )?;
    sop(
        b,
        format!("s_cmp_ge_i32 s{S_R}, s{S_LIM}"),
        &[],
        &[S_R, S_LIM],
    )?;
    op(b, format!("s_cbranch_scc1 {END}"), &[], &[])?;
    sop(
        b,
        format!("s_mov_b32 s{S_PSEL}, 0x76543210"),
        &[S_PSEL],
        &[],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{DEN}, {}", lit(0x4380_0000)),
        &[DEN],
        &[],
    )?;
    // Head row bases: interleaved r * 2048 B, gate r * 1024 B, q16 r * 512 B, angles (r >> 4) * 256 B.
    sop(b, format!("s_lshl_b32 s{T0}, s{S_R}, 11"), &[T0], &[S_R])?;
    add64(b, QB, P_IL, T0)?;
    sop(b, format!("s_lshl_b32 s{T0}, s{S_R}, 10"), &[T0], &[S_R])?;
    add64(b, GB, P_GATE, T0)?;
    sop(b, format!("s_lshl_b32 s{T0}, s{S_R}, 9"), &[T0], &[S_R])?;
    add64(b, OB, P_Q16, T0)?;
    sop(b, format!("s_lshr_b32 s{T1S}, s{S_R}, 4"), &[T1S], &[S_R])?;
    sop(b, format!("s_lshl_b32 s{T0}, s{T1S}, 8"), &[T0], &[T1S])?;
    add64(b, AB, S_ANG, T0)?;
    wait_alu(b, "depctr_sa_sdst(0)")
}

/// Loads, the one-wave Q norm, rotary and the f16 / gate stores.
fn body(b: &mut Builder) -> R {
    // Q (dims lane + 32k at +128k bytes), weights, the two angle lanes, then the gate half of the head row.
    for k in 0..8u8 {
        gload(b, "b32", v(Q + k), OFF4, QB, 128 * u32::from(k))?;
    }
    for k in 0..8u8 {
        gload(b, "b32", v(W + k), OFF4, P_QN, 128 * u32::from(k))?;
    }
    gload(b, "b32", v(COS), OFF4, AB, 0)?;
    gload(b, "b32", v(SIN), OFF4, AB, 128)?;
    for k in 0..8u8 {
        gload(b, "b32", v(G + k), OFF4, QB, 1024 + 128 * u32::from(k))?;
    }
    // Leaves fma(q, q, +0), then the in-lane levels 128 / 64 / 32 of the incumbent halving tree.
    for k in 0..8u8 {
        op(
            b,
            format!("v_fma_f32 v{}, v{}, v{}, 0", SQ + k, Q + k, Q + k),
            &[v(SQ + k)],
            &[v(Q + k)],
        )?;
    }
    for k in 0..4u8 {
        op(
            b,
            format!("v_add_f32_e32 v{0}, v{1}, v{0}", SQ + k, SQ + k + 4),
            &[v(SQ + k)],
            &[v(SQ + k + 4), v(SQ + k)],
        )?;
    }
    for k in 0..2u8 {
        op(
            b,
            format!("v_add_f32_e32 v{0}, v{1}, v{0}", SQ + k, SQ + k + 2),
            &[v(SQ + k)],
            &[v(SQ + k + 2), v(SQ + k)],
        )?;
    }
    op(
        b,
        format!("v_add_f32_e32 v{SUM}, v{}, v{SQ}", SQ + 1),
        &[v(SUM)],
        &[v(SQ + 1), v(SQ)],
    )?;
    // Levels 16 / 8 / 4 / 2 / 1: lane XOR butterfly.
    for k in [16u32, 8, 4, 2, 1] {
        if k == 16 {
            op(
                b,
                format!("v_permlanex16_b32 v{PEER}, v{SUM}, s{S_PSEL}, 0xfedcba98"),
                &[v(PEER)],
                &[v(SUM), s(S_PSEL)],
            )?;
        } else {
            op(
                b,
                format!("v_mov_b32_dpp v{PEER}, v{SUM} row_xmask:{k} row_mask:0xf bank_mask:0xf"),
                &[v(PEER)],
                &[v(SUM)],
            )?;
        }
        op(
            b,
            format!("v_add_f32_e32 v{SUM}, v{PEER}, v{SUM}"),
            &[v(SUM)],
            &[v(PEER), v(SUM)],
        )?;
    }
    // mean = sum / 256.0f (full division), then rsqrtf(eps + mean) with the incumbent's scalar denormal guard.
    fdiv(
        b,
        QUOT,
        &format!("v{SUM}"),
        &format!("s{DEN}"),
        &[v(SUM), s(DEN)],
        DIV,
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{MEAN}, v{QUOT}"),
        &[s(MEAN)],
        &[v(QUOT)],
    )?;
    sop(
        b,
        format!("s_add_f32 s{MEAN}, s{S_EPS}, s{MEAN}"),
        &[MEAN],
        &[S_EPS, MEAN],
    )?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    sop(b, format!("s_cmp_lt_f32 s{MEAN}, 0x800000"), &[], &[MEAN])?;
    sop(
        b,
        format!("s_mul_f32 s{T0}, s{MEAN}, 0x4b800000"),
        &[T0],
        &[MEAN],
    )?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    sop(
        b,
        format!("s_cselect_b32 s{MEAN}, s{T0}, s{MEAN}"),
        &[MEAN],
        &[T0, MEAN],
    )?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    sop(b, format!("v_s_rsq_f32 s{MEAN}, s{MEAN}"), &[MEAN], &[MEAN])?;
    sop(
        b,
        format!("s_mul_f32 s{T0}, s{MEAN}, 0x45800000"),
        &[T0],
        &[MEAN],
    )?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    sop(
        b,
        format!("s_cselect_b32 s{RMS}, s{T0}, s{MEAN}"),
        &[RMS],
        &[T0, MEAN],
    )?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    // (q * w) * rms, two roundings in that order.
    for k in 0..8u8 {
        op(
            b,
            format!("v_mul_f32_e32 v{0}, v{1}, v{0}", W + k, Q + k),
            &[v(W + k)],
            &[v(Q + k), v(W + k)],
        )?;
        op(
            b,
            format!("v_mul_f32_e32 v{0}, s{RMS}, v{0}", W + k),
            &[v(W + k)],
            &[s(RMS), v(W + k)],
        )?;
    }
    // Half-split RoPE on the lane's pair (dim lane, dim lane + 32) = (k = 0, k = 1).
    let (x0, x1) = (W, W + 1);
    op(
        b,
        format!("v_mul_f32_e32 v{T1}, v{x1}, v{SIN}"),
        &[v(T1)],
        &[v(x1), v(SIN)],
    )?;
    op(
        b,
        format!("v_mul_f32_e32 v{T2}, v{x1}, v{COS}"),
        &[v(T2)],
        &[v(x1), v(COS)],
    )?;
    op(
        b,
        format!("v_fma_f32 v{x1}, v{x0}, v{SIN}, v{T2}"),
        &[v(x1)],
        &[v(x0), v(SIN), v(T2)],
    )?;
    op(
        b,
        format!("v_fma_f32 v{x0}, v{x0}, v{COS}, -v{T1}"),
        &[v(x0)],
        &[v(x0), v(COS), v(T1)],
    )?;
    // f16 RNE and the scatter stores (dim lane + 32k at byte 64k of the head row).
    for k in 0..8u8 {
        cvt_half(b, SQ + k, false, W + k)?;
    }
    for k in 0..8u8 {
        gstore(b, "b16", OFF2, SQ + k, OB, 64 * u32::from(k))?;
    }
    // Gate passthrough (verbatim f32 bits, 128 B per k).
    for k in 0..8u8 {
        gstore(b, "b32", OFF4, G + k, GB, 128 * u32::from(k))?;
    }
    Ok(())
}

/// The Q prep entry `attention_fa2_qprep_pm_gfx1201`.
pub fn emit(arch: Arch, spec: &FaPrefillSpec) -> Result<Emitted, String> {
    admit(arch, spec)?;
    let kernargs = KernargLayout::new(KERNARG_BYTES)
        .pointer_access("interleaved", 0, Access::ReadOnly)
        .pointer_access("q16", 8, Access::WriteOnly)
        .pointer_access("gate", 16, Access::WriteOnly)
        .pointer_access("q_norm", 24, Access::ReadOnly)
        .pointer_access("angles", 32, Access::ReadOnly)
        .hidden("rows", 40, 4, "by_value")
        .hidden("eps", 44, 4, "by_value");
    let kspec = KernelSpec {
        kernel_id: KERNEL_ID.into(),
        variant: "qprep".into(),
        arch,
        symbol: SYMBOL.into(),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: BLOCK as u16,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan()?);
    b.enable_delay_alu();
    prologue(&mut b)?;
    body(&mut b)?;
    b.wait_all()?;
    b.label(END)?;
    b.control(Sop::End.encode(Arch::Gfx1201)?)?;
    b.finish()
}

/// The single export as one code object (`fa_qprep_pm_gfx1201`).
pub fn emit_module(
    arch: Arch,
    spec: &FaPrefillSpec,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    admit(arch, spec)?;
    let emitted = vec![emit(arch, spec)?];
    let (text, proof) = super::iu4_gemm::module(&emitted, MODULE)?;
    Ok((emitted, text, proof))
}

// ------------------------------------------------------------------ contract
const COUNTED: [&str; 22] = [
    "global_load_b32",
    "global_store_b16",
    "global_store_b32",
    "s_load_b256",
    "s_load_b128",
    "v_readfirstlane_b32",
    "v_fma_f32",
    "v_mul_f32_e32",
    "v_add_f32_e32",
    "v_permlanex16_b32",
    "v_mov_b32_dpp",
    "v_div_scale_f32",
    "v_div_fmas_f32",
    "v_div_fixup_f32",
    "v_rcp_f32_e32",
    "v_fmac_f32_e32",
    "v_cvt_f16_f32_e32",
    "v_s_rsq_f32",
    "s_add_f32",
    "s_mul_f32",
    "s_cmp_lt_f32",
    "s_endpgm",
];

fn counts(e: &Emitted, names: &[&str]) -> serde_json::Map<String, serde_json::Value> {
    let mut counts = serde_json::Map::new();
    for name in names {
        let n = e
            .s_text
            .lines()
            .filter(|l| l.trim().split_whitespace().next() == Some(*name))
            .count();
        counts.insert((*name).into(), n.into());
    }
    counts
}

/// Shape contract of the export (`toolchain::IsaShapeContract`): counts are derived from the emission. No DS,
/// barrier or WMMA; no LDS at all.
pub fn contract(arch: Arch, emitted: &Emitted) -> serde_json::Value {
    let _ = arch;
    serde_json::json!({
        "symbol": SYMBOL, "counts": counts(emitted, &COUNTED),
        "forbidden": ["scratch_*", "flat_*", "buffer_*", "global_atomic*", "ds_*", "s_barrier*", "v_wmma*", "v_swmmac*",
                      "v_writelane*", "s_setpc*", "s_swappc*", "s_call*", "s_waitcnt*"],
        "vgpr_max": VGPRS, "sgpr_max": 104,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0,
        "group_segment_fixed_bytes": 0,
    })
}
