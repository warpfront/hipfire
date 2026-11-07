// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted (PeaceMaker) gfx1201 draft-head partial top-8: `topk8_partial_f32_blocks_pm_gfx1201`,
//! the byte twin of the hipcc kernel `topk8_partial_f32_blocks` (`kernels/src/qwen4_gemv_q8_0.hip`),
//! which is `topk8_partial_f32` plus a trailing `block0`: workgroup `blockIdx.x` is group
//! `group = blockIdx.x + block0` of the 128-group launch (`block0 = 0` is `topk8_partial_f32` itself).
//!
//! ABI (nine arguments, 48 bytes, no padding): `v` `*f32` @0, `n` i32 @8, `chunk` i32 @12, `pv`
//! `*f32` @16, `pi` `*i32` @24, `front` i32 @32, `special` i32 @36, `tail` i32 @40, `block0` i32
//! @44. Grid `[blocks, 1, 1]`, block 256, 64 bytes of static LDS, no scratch, no atomics.
//!
//! Law (identical to the HIP kernel). `begin = group * chunk`, `end = min(n, begin + chunk)`.
//! Thread `t` holds the eight candidates `j = begin + t + 256 u` (`u` in 0..8; `j < end`, else the
//! sentinel `(-inf, INT_MAX)`). Eight rounds; each takes the block's best `(value, layout index)`
//! under `topk8_better` (`v > bv || (v == bv && i < bi)`: larger value, then smaller layout index;
//! NaN is never better, `+inf` is allowed, `-inf` beats the sentinel on a smaller index), removes
//! the candidate whose index equals the winner (value `-inf`, index `INT_MAX`), and thread 0 stores
//! `pv[group * 8 + round] = bits(winner value)` and `pi[group * 8 + round] = token(winner index)`
//! with `token(bi) = bi < front ? bi : (bi < front + tail ? bi - front + special : bi - tail)`.
//! A group without a qualifying candidate stores `(-inf, INT_MAX - tail)` for that round.
//!
//! Because layout indices are unique, the winner of a round is the unique maximum of the total
//! order above, so the reduction order is free: each thread folds its eight candidates, a wave
//! reduces with a five-level exchange butterfly (`row_xmask` DPP and `v_permlanex16_b32`), and the
//! eight wave winners meet through LDS (one workgroup barrier each way per round; every thread
//! folds the eight pairs). The value is only ever moved (never computed on), so its raw bits,
//! `+0` against `-0` included, are the stored bits.
//!
//! Loads go through a raw buffer descriptor based at `v + begin * 4` with `max(end - begin, 0) * 4`
//! records: a lane at or past `end` reads 0 without touching memory and is replaced by the
//! sentinel with a lane mask. `v` is indexed from layout row 0, so a vocabulary shard's rebased
//! pointer works exactly as with the hipcc kernel.
use super::common::{bload, lit, mem, op, s, smem, sop, sr, srd_tail, v};
use crate::{
    insn::{Instruction, MemoryClass},
    plan::Access,
    reg::{Kind, Live, RegRef},
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};
use peacemaker_author::{ready, retire, Free, Gfx1201, LdsRegion, Target, Workgroup};

type Wg<'b, T> = Workgroup<'b, T, Builder>;
type R = Result<(), String>;

/// Threads per workgroup.
pub const BLOCK: u32 = 256;
/// Kernarg bytes (the nine-argument HIP ABI; `block0` ends at byte 48).
pub const KERNARG_BYTES: u32 = 48;
/// Static LDS: eight 8-byte (value, index) wave winners.
pub const STATIC_LDS: u32 = 64;
/// Candidates per thread (`chunk <= 256 * CAND`) and rounds.
pub const CAND: u8 = 8;

pub enum Ex {}

pub fn symbol(arch: Arch) -> String {
    format!("topk8_partial_f32_blocks_pm_{}", arch.name())
}
pub fn module(arch: Arch) -> String {
    format!("topk8_partial_blocks_pm_{}", arch.name())
}

pub fn kernargs() -> KernargLayout {
    KernargLayout::new(KERNARG_BYTES)
        .pointer_access("v", 0, Access::ReadOnly)
        .hidden("n", 8, 4, "by_value")
        .hidden("chunk", 12, 4, "by_value")
        .pointer_access("pv", 16, Access::WriteOnly)
        .pointer_access("pi", 24, Access::WriteOnly)
        .hidden("front", 32, 4, "by_value")
        .hidden("special", 36, 4, "by_value")
        .hidden("tail", 40, 4, "by_value")
        .hidden("block0", 44, 4, "by_value")
}

// ---- SGPRs ------------------------------------------------------------------
const KARG: u8 = 0;
// kernarg: v 8:9, n 10, chunk 11, pv 12:13, pi 14:15 (one b256 load); front 16, special 17,
// tail 18, block0 19 (one b128 load).
const VP: u8 = 8;
const NIN: u8 = 10;
const CHUNK: u8 = 11;
const PVP: u8 = 12;
const PIP: u8 = 14;
const FRONT: u8 = 16;
const SPECIAL: u8 = 17;
const TAIL: u8 = 18;
const BLOCK0: u8 = 19;
const GROUP: u8 = 20;
const BEGIN: u8 = 21;
const CNT: u8 = 22;
const T0: u8 = 23;
const T1: u8 = 24;
const T2: u8 = 25;
/// `front + tail`.
const FT: u8 = 26;
/// `special - front`.
const D1: u8 = 27;
/// `-tail`.
const NT: u8 = 28;
const C4: u8 = 29;
const C32: u8 = 30;
const SRD_V: u8 = 32;
const SRD_PV: u8 = 36;
const SRD_PI: u8 = 40;
/// Lane-mask pairs (the high word is never written in wave32).
const MA: u8 = 44;
const MB: u8 = 46;
const MC: u8 = 48;
const MASKS: [u8; 3] = [MA, MB, MC];
const PSEL: u8 = 52;
const S_END: u8 = 56;

// ---- VGPRs ------------------------------------------------------------------
const TID: u8 = 0;
const VOFF: u8 = 1;
const LDSW: u8 = 2;
/// Always 0: the LDS read address and the buffer-store offset.
const VZ: u8 = 3;
const PV: u8 = 4;
const PK: u8 = 5;
/// Best (value, index) of the thread, then of the block.
const BV: u8 = 6;
const BI: u8 = 7;
/// Wave winner (value, index): a consecutive pair for the LDS store.
const WV: u8 = 8;
const WK: u8 = 9;
/// `-inf` and `INT_MAX` (the candidate sentinel).
const CNINF: u8 = 10;
const CIMAX: u8 = 11;
const VT0: u8 = 12;
const VT1: u8 = 13;
/// The eight candidate values / layout indices.
const VAL: u8 = 16;
const IDX: u8 = 24;
/// The eight (value, index) wave winners as `[v0, i0, v1, i1, ..]` (four b128 loads).
const FIN: u8 = 32;
/// The eight rounds' winner values and token ids, held until the final thread-0 stores.
const RV: u8 = 48;
const RI: u8 = 56;
const V_END: u8 = 64;

const NEG_INF: &str = "0xff800000";
const INT_MAX: &str = "0x7fffffff";

pub fn emit(arch: Arch) -> Result<Emitted, String> {
    if !matches!(arch, Arch::Gfx1201) {
        return Err(format!(
            "topk8_partial_blocks_pm: built for gfx1201 only, not {}",
            arch.name()
        ));
    }
    let kspec = KernelSpec {
        kernel_id: "topk8_partial_blocks_pm".into(),
        variant: "f32_blocks".into(),
        arch,
        symbol: symbol(arch),
        kernargs: kernargs(),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: BLOCK as u16,
        group_segment_fixed_size: STATIC_LDS,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan()?);
    b.enable_delay_alu();
    kernel(&mut Workgroup::<Gfx1201, Builder>::new(&mut b)?, arch)?;
    b.finish()
}

/// The single-symbol module `topk8_partial_blocks_pm_<arch>`.
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = vec![emit(arch)?];
    let (text, proof) = super::iu4_gemm::module(&emitted, &module(arch))?;
    Ok((emitted, text, proof))
}

fn plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(u16::from(V_END), 104)?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    for base in (8u8..S_END).step_by(8) {
        p.add_range(&format!("s_blk{base}"), Kind::S, base, 8, Live::Whole)?;
    }
    for base in (0u8..V_END).step_by(8) {
        p.add_range(&format!("v_blk{base}"), Kind::V, base, 8, Live::Whole)?;
    }
    Ok(p)
}

// ---- small emission helpers -------------------------------------------------
#[derive(Clone, Copy)]
enum O {
    V(u8),
    S(u8),
    M(u8),
    K(&'static str),
    N(i64),
}
fn ot(o: O) -> String {
    match o {
        O::V(n) => format!("v{n}"),
        O::S(n) | O::M(n) => format!("s{n}"),
        O::K(t) => t.to_string(),
        O::N(n) => lit(n as u32),
    }
}
fn oreg(o: O) -> Option<RegRef> {
    match o {
        O::V(n) => Some(v(n)),
        O::S(n) => Some(s(n)),
        O::M(n) => Some(sr(n, 2)),
        O::K(_) | O::N(_) => None,
    }
}
/// `mnemonic dst, src..` with the registers of the operands as defs and uses.
fn alu(b: &mut Builder, mn: &str, dst: O, srcs: &[O]) -> R {
    let text = format!(
        "{mn} {}, {}",
        ot(dst),
        srcs.iter().map(|o| ot(*o)).collect::<Vec<_>>().join(", ")
    );
    let uses: Vec<RegRef> = srcs.iter().filter_map(|o| oreg(*o)).collect();
    op(
        b,
        text,
        &[oreg(dst).ok_or("alu destination must be a register")?],
        &uses,
    )
}
fn mask(m: u8) -> RegRef {
    sr(m, 2)
}
fn vr(base: u8, len: u8) -> RegRef {
    super::common::vr(base, len)
}
fn ds_st(text: String, addr: u8, data: RegRef) -> Instruction {
    Instruction::new(text, vec![], vec![v(addr), data]).memory(MemoryClass::DsStore)
}
fn ds_ld(text: String, dst: RegRef, addr: u8) -> Instruction {
    Instruction::new(text, vec![dst], vec![v(addr)]).memory(MemoryClass::DsLoad)
}
/// `dst = a op b` over SGPRs.
fn s2(
    b: &mut Builder,
    mn: &str,
    dst: u8,
    a: &str,
    a_reg: Option<u8>,
    c: &str,
    c_reg: Option<u8>,
) -> R {
    let uses: Vec<u8> = [a_reg, c_reg].into_iter().flatten().collect();
    sop(b, format!("{mn} s{dst}, {a}, {c}"), &[dst], &uses)
}
/// `dst:dst+1 = src:src+1 + x * y` (64-bit product through `lo`, `hi`).
fn add64_mul(b: &mut Builder, dst: u8, src: u8, x: u8, y: u8, lo: u8, hi: u8) -> R {
    let a = b.spec.arch;
    sop(b, format!("s_mul_i32 s{lo}, s{x}, s{y}"), &[lo], &[x, y])?;
    sop(b, format!("s_mul_hi_u32 s{hi}, s{x}, s{y}"), &[hi], &[x, y])?;
    sop(
        b,
        format!("{} s{dst}, s{src}, s{lo}", super::common::s_add_u32(a)),
        &[dst],
        &[src, lo],
    )?;
    sop(
        b,
        format!(
            "{} s{}, s{}, s{hi}",
            super::common::s_addc_u32(a),
            dst + 1,
            src + 1
        ),
        &[dst + 1],
        &[src + 1, hi],
    )
}
/// Raw buffer store of one dword at `v{voff} + offset` (soffset `null` on gfx12).
fn bstore(b: &mut Builder, data: u8, voff: u8, srd: u8, offset: u32) -> R {
    if offset > b.spec.arch.buffer_offset_max() {
        return Err(format!("buffer offset {offset} exceeds the field"));
    }
    let off = if offset == 0 {
        String::new()
    } else {
        format!(" offset:{offset}")
    };
    mem(
        b,
        format!(
            "buffer_store_b32 v{data}, v{voff}, s[{srd}:{}], null offen{off}",
            srd + 3
        ),
        &[],
        &[v(voff), v(data), sr(srd, 4)],
        MemoryClass::VmemStore,
    )
}
/// `dst = src` of lane `l ^ m` for `m` in 1, 2, 4, 8 (DPP `row_xmask`) and 16 (`v_permlanex16_b32`).
fn xch(b: &mut Builder, dst: u8, src: u8, m: u32) -> R {
    if m < 16 {
        op(
            b,
            format!("v_mov_b32_dpp v{dst}, v{src} row_xmask:{m} row_mask:0xf bank_mask:0xf"),
            &[v(dst)],
            &[v(src)],
        )
    } else {
        op(
            b,
            format!("v_permlanex16_b32 v{dst}, v{src}, s{PSEL}, 0xfedcba98"),
            &[v(dst)],
            &[v(src), s(PSEL)],
        )
    }
}
/// `(wv, wk) = best((pv, pk), (wv, wk))` under `topk8_better`: larger value, equal values the
/// lower (non-negative) index; NaN is never better. Clobbers MA, MB, MC.
fn better(b: &mut Builder, pv: u8, pk: u8, wv: u8, wk: u8) -> R {
    alu(b, "v_cmp_gt_f32_e64", O::M(MA), &[O::V(pv), O::V(wv)])?;
    alu(b, "v_cmp_eq_f32_e64", O::M(MB), &[O::V(pv), O::V(wv)])?;
    alu(b, "v_cmp_lt_u32_e64", O::M(MC), &[O::V(pk), O::V(wk)])?;
    alu(b, "s_and_b32", O::M(MB), &[O::M(MB), O::M(MC)])?;
    alu(b, "s_or_b32", O::M(MA), &[O::M(MA), O::M(MB)])?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(wv),
        &[O::V(wv), O::V(pv), O::M(MA)],
    )?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(wk),
        &[O::V(wk), O::V(pk), O::M(MA)],
    )
}
/// Wave butterfly: every lane ends with the best `(wv, wk)` of the 32 lanes.
fn wave_best(b: &mut Builder, wv: u8, wk: u8) -> R {
    for m in [1u32, 2, 4, 8, 16] {
        xch(b, PV, wv, m)?;
        xch(b, PK, wk, m)?;
        better(b, PV, PK, wv, wk)?;
    }
    Ok(())
}

// ---- kernel -----------------------------------------------------------------
fn kernel<T: Target>(wg: &mut Wg<T>, arch: Arch) -> R {
    if !arch.gfx12() {
        return Err("topk8_partial_blocks_pm: gfx12 workgroup-id convention only".into());
    }
    let mut ex = wg.lds::<Ex>("ex", 0, STATIC_LDS)?;
    let end = wg.exit(".Ltopk8_end")?;
    prologue(wg)?;
    for round in 0..CAND {
        // The thread's best candidate.
        let b = wg.isa();
        alu(b, "v_mov_b32_e32", O::V(BV), &[O::V(CNINF)])?;
        alu(b, "v_mov_b32_e32", O::V(BI), &[O::V(CIMAX)])?;
        for u in 0..CAND {
            better(b, VAL + u, IDX + u, BV, BI)?;
        }
        // The wave's best, then the eight wave winners through LDS.
        alu(b, "v_mov_b32_e32", O::V(WV), &[O::V(BV)])?;
        alu(b, "v_mov_b32_e32", O::V(WK), &[O::V(BI)])?;
        wave_best(b, WV, WK)?;
        let st = wg.begin_write(ex);
        let (ew, pend) = wg.ds_store(
            st,
            ds_st(
                format!("ds_store_b64 v{LDSW}, v[{WV}:{WK}]"),
                LDSW,
                vr(WV, 2),
            ),
        )?;
        let d = wg.wait(pend)?;
        let (ep,) = wg.barrier((ready(ew, d),))?;
        for q in 0..4u8 {
            let off = if q == 0 {
                String::new()
            } else {
                format!(" offset:{}", 16 * u32::from(q))
            };
            wg.ds_load(
                &ep,
                ds_ld(
                    format!(
                        "ds_load_b128 v[{}:{}], v{VZ}{off}",
                        FIN + 4 * q,
                        FIN + 4 * q + 3
                    ),
                    vr(FIN + 4 * q, 4),
                    VZ,
                ),
            )?;
        }
        let b = wg.isa();
        alu(b, "v_mov_b32_e32", O::V(BV), &[O::V(FIN)])?;
        alu(b, "v_mov_b32_e32", O::V(BI), &[O::V(FIN + 1)])?;
        for w in 1..8u8 {
            better(b, FIN + 2 * w, FIN + 2 * w + 1, BV, BI)?;
        }
        let (ex2,) = wg.barrier((retire(ep),))?;
        ex = ex2;
        let b = wg.isa();
        // Remove the winner: the candidate whose index equals it (a sentinel winner removes
        // nothing that is not already the sentinel).
        if round + 1 < CAND {
            for u in 0..CAND {
                alu(b, "v_cmp_eq_u32_e64", O::M(MA), &[O::V(BI), O::V(IDX + u)])?;
                alu(
                    b,
                    "v_cndmask_b32_e64",
                    O::V(VAL + u),
                    &[O::V(VAL + u), O::V(CNINF), O::M(MA)],
                )?;
                alu(
                    b,
                    "v_cndmask_b32_e64",
                    O::V(IDX + u),
                    &[O::V(IDX + u), O::V(CIMAX), O::M(MA)],
                )?;
            }
        }
        // The round's output: the value bits and the token id
        // `bi < front ? bi : (bi < front + tail ? bi - front + special : bi - tail)`.
        alu(b, "v_mov_b32_e32", O::V(RV + round), &[O::V(BV)])?;
        alu(b, "v_add_nc_u32_e32", O::V(VT0), &[O::S(D1), O::V(BI)])?;
        alu(b, "v_add_nc_u32_e32", O::V(VT1), &[O::S(NT), O::V(BI)])?;
        alu(b, "v_cmp_gt_i32_e64", O::M(MA), &[O::S(FT), O::V(BI)])?;
        alu(
            b,
            "v_cndmask_b32_e64",
            O::V(RI + round),
            &[O::V(VT1), O::V(VT0), O::M(MA)],
        )?;
        alu(b, "v_cmp_gt_i32_e64", O::M(MB), &[O::S(FRONT), O::V(BI)])?;
        alu(
            b,
            "v_cndmask_b32_e64",
            O::V(RI + round),
            &[O::V(RI + round), O::V(BI), O::M(MB)],
        )?;
    }
    let _: LdsRegion<Ex, Free> = ex;
    let b = wg.isa();
    // Only thread 0 (wave 0, lane 0) stores the 8 + 8 results of its group.
    alu(b, "v_cmp_eq_u32_e64", O::M(MB), &[O::N(0), O::V(TID)])?;
    op(
        b,
        format!("s_and_b32 exec_lo, exec_lo, s{MB}"),
        &[],
        &[mask(MB)],
    )?;
    for r in 0..CAND {
        bstore(b, RV + r, VZ, SRD_PV, 4 * u32::from(r))?;
        bstore(b, RI + r, VZ, SRD_PI, 4 * u32::from(r))?;
    }
    b.wait_all()?;
    wg.end(end)
}

fn prologue<T: Target>(wg: &mut Wg<T>) -> R {
    let b = wg.isa();
    let arch = b.spec.arch;
    let add = super::common::s_add_i32(arch);
    smem(b, VP, 8, KARG, 0)?;
    smem(b, FRONT, 4, KARG, 32)?;
    alu(b, "v_lshlrev_b32_e32", O::V(VOFF), &[O::N(2), O::V(TID)])?;
    alu(b, "v_lshrrev_b32_e32", O::V(VT0), &[O::N(5), O::V(TID)])?;
    alu(b, "v_lshlrev_b32_e32", O::V(LDSW), &[O::N(3), O::V(VT0)])?;
    alu(b, "v_mov_b32_e32", O::V(VZ), &[O::N(0)])?;
    alu(b, "v_mov_b32_e32", O::V(CNINF), &[O::K(NEG_INF)])?;
    alu(b, "v_mov_b32_e32", O::V(CIMAX), &[O::K(INT_MAX)])?;
    for m in MASKS {
        sop(b, format!("s_mov_b32 s{}, 0", m + 1), &[m + 1], &[])?
    }
    sop(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[PSEL], &[])?;
    sop(b, format!("s_mov_b32 s{C4}, 4"), &[C4], &[])?;
    sop(b, format!("s_mov_b32 s{C32}, 32"), &[C32], &[])?;
    // group = workgroup id + block0; begin = group * chunk; cnt = max(min(n, begin + chunk) - begin, 0).
    s2(
        b,
        add,
        GROUP,
        "ttmp9",
        None,
        &format!("s{BLOCK0}"),
        Some(BLOCK0),
    )?;
    s2(
        b,
        "s_mul_i32",
        BEGIN,
        &format!("s{GROUP}"),
        Some(GROUP),
        &format!("s{CHUNK}"),
        Some(CHUNK),
    )?;
    s2(
        b,
        add,
        T0,
        &format!("s{BEGIN}"),
        Some(BEGIN),
        &format!("s{CHUNK}"),
        Some(CHUNK),
    )?;
    s2(
        b,
        "s_min_i32",
        T0,
        &format!("s{NIN}"),
        Some(NIN),
        &format!("s{T0}"),
        Some(T0),
    )?;
    s2(
        b,
        "s_sub_co_i32",
        CNT,
        &format!("s{T0}"),
        Some(T0),
        &format!("s{BEGIN}"),
        Some(BEGIN),
    )?;
    sop(b, format!("s_max_i32 s{CNT}, s{CNT}, 0"), &[CNT], &[CNT])?;
    // Token map constants: front + tail, special - front, -tail.
    s2(
        b,
        add,
        FT,
        &format!("s{FRONT}"),
        Some(FRONT),
        &format!("s{TAIL}"),
        Some(TAIL),
    )?;
    s2(
        b,
        "s_sub_co_i32",
        D1,
        &format!("s{SPECIAL}"),
        Some(SPECIAL),
        &format!("s{FRONT}"),
        Some(FRONT),
    )?;
    s2(
        b,
        "s_sub_co_i32",
        NT,
        "0",
        None,
        &format!("s{TAIL}"),
        Some(TAIL),
    )?;
    // Loads: v + begin * 4, cnt * 4 records.
    add64_mul(b, SRD_V, VP, BEGIN, C4, T1, T2)?;
    sop(b, format!("s_lshl_b32 s{T0}, s{CNT}, 2"), &[T0], &[CNT])?;
    srd_tail(b, SRD_V, Some(T0))?;
    // Stores: pv / pi + group * 32, 32 records.
    add64_mul(b, SRD_PV, PVP, GROUP, C32, T1, T2)?;
    srd_tail(b, SRD_PV, Some(C32))?;
    add64_mul(b, SRD_PI, PIP, GROUP, C32, T1, T2)?;
    srd_tail(b, SRD_PI, Some(C32))?;
    for u in 0..CAND {
        bload(b, 1, VAL + u, VOFF, SRD_V, 1024 * u32::from(u))?;
    }
    // Layout index begin + tid, and the in-range lane mask `tid + 256 u < cnt`.
    alu(b, "v_add_nc_u32_e32", O::V(VT0), &[O::S(BEGIN), O::V(TID)])?;
    for u in 0..CAND {
        let off = 256 * i64::from(u);
        if u == 0 {
            alu(b, "v_mov_b32_e32", O::V(IDX), &[O::V(VT0)])?;
            alu(b, "v_cmp_gt_i32_e64", O::M(MA), &[O::S(CNT), O::V(TID)])?;
        } else {
            alu(
                b,
                "v_add_nc_u32_e32",
                O::V(IDX + u),
                &[O::N(off), O::V(VT0)],
            )?;
            alu(b, "v_add_nc_u32_e32", O::V(VT1), &[O::N(off), O::V(TID)])?;
            alu(b, "v_cmp_gt_i32_e64", O::M(MA), &[O::S(CNT), O::V(VT1)])?;
        }
        alu(
            b,
            "v_cndmask_b32_e64",
            O::V(VAL + u),
            &[O::V(CNINF), O::V(VAL + u), O::M(MA)],
        )?;
        alu(
            b,
            "v_cndmask_b32_e64",
            O::V(IDX + u),
            &[O::V(CIMAX), O::V(IDX + u), O::M(MA)],
        )?;
    }
    Ok(())
}

/// Mnemonics whose exact counts the shape contract pins.
const COUNTED: [&str; 12] = [
    "s_load_b256",
    "s_load_b128",
    "buffer_load_b32",
    "buffer_store_b32",
    "ds_store_b64",
    "ds_load_b128",
    "s_barrier_signal",
    "s_barrier_wait",
    "v_mov_b32_dpp",
    "v_permlanex16_b32",
    "s_endpgm",
    "s_barrier",
];

/// Shape contract of the emitted kernel (`peacemaker-isa emit --co --shape`).
pub fn contract(arch: Arch, e: &Emitted) -> serde_json::Value {
    let mut counts = serde_json::Map::new();
    for name in COUNTED {
        let n = e
            .s_text
            .lines()
            .filter(|l| l.trim().split_whitespace().next() == Some(name))
            .count();
        counts.insert(name.into(), n.into());
    }
    serde_json::json!({
        "symbol": symbol(arch), "counts": counts,
        "forbidden": ["scratch_*", "flat_*", "global_*", "buffer_atomic_*", "ds_add_*", "ds_sub_*", "ds_rsub_*", "ds_inc_*", "ds_dec_*", "ds_min_*", "ds_max_*",
                      "ds_and_*", "ds_or_*", "ds_xor_*", "ds_mskor_*", "ds_cmpst*", "ds_cmpstore*", "ds_wrxchg*", "ds_append", "ds_consume", "ds_gws_*",
                      "ds_bpermute*", "ds_permute*", "s_setpc*", "s_swappc*"],
        "vgpr_max": V_END, "sgpr_max": 104,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0,
        "group_segment_fixed_bytes": STATIC_LDS,
    })
}
