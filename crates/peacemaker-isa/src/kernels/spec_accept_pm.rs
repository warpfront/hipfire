// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted gfx1151 / gfx1201 ordered sparse accept
//! (`spec_accept_ordered_u32_pm_<arch>`): the kernarg ABI, launch shape
//! (grid `[1,1,1]`, one thread) and result words of the hipcc twin
//! `spec_accept_ordered_u32` in `kernels/src/spec_sampling_cpu_ar.hip`
//! (`accept_sampled_prefix` over resident target / draft sparse rows).
//!
//! ```text
//! kernarg (80 bytes): tcounts@0 tids@8 tprobs@16 dcounts@24 dids@32 dprobs@40
//!                     proposals@48 uniforms@56 result@64, k i32@72, eos u32@76
//! for row in 0..k:
//!   qx = prob(draft row, proposals[row]);  px = prob(target row, proposals[row])
//!   if qx > 0 && RN(uniforms[row] * qx) < px:          // strict predicate, NaN loses
//!       if proposals[row] == eos: result = (row + 1, 0xFFFFFFFF); return
//!       continue
//!   residual (u2 = uniforms[row + 1]):                  // sample_residual(p, q, u2)
//!       r_i = max0(RN(p_i - q(id_i))) over the target row's entries in order
//!       mass = ((0 + r_0) + r_1) + ..
//!       mass > 0:  target = RN(u2 * mass); acc = 0
//!                  first entry with r_i > 0 and target < (acc += r_i) is the token;
//!                  none: the last entry with r_i > 0; none at all: sample(p, u2)
//!       else:      sample(p, u2)                        // inverse CDF over p
//!       result = (row, token); return
//! result = (k, sample(target row k, uniforms[k]))        // bonus
//! sample(p, u): acc = 0; first i with u < (acc += p_i) -> id_i; else n > 0 ? id_{n-1} : 0
//! ```
//!
//! `prob(row, token)` is the probability of the first entry `i < counts[row]`
//! (signed) whose id is `token`, else 0.
//!
//! Numerics are the twin's, instruction for instruction in value: every add,
//! subtract and multiply is one round-to-nearest `v_add_f32` / `v_sub_f32` /
//! `v_mul_f32` (no contraction, denormals preserved, the descriptor's
//! `float_denorm_mode_32 = 3`); `max0` is the single `v_max_f32` /
//! `v_max_num_f32` hipcc emits for `fmaxf(v, 0.0f)` (a NaN loses); every compare
//! is the ordered `v_cmp_{lt,..}_f32` (a NaN compares false). The skipped
//! `r <= 0` entries of the selection pass are folded into one branch-free
//! step: `acc + r` leaves `acc` bit-identical for `r = +-0`, and the hit mask
//! is `r > 0 && target < acc`, so the observable `acc`, `last` and token
//! sequence is the twin's.
//!
//! One thread: the kernel masks EXEC to lane 0 (the twin's `threadIdx.x != 0`
//! return) and returns when `blockIdx.x != 0`. There are no atomics, no LDS
//! and no scratch. All data comes in through scalar loads: the id search takes
//! the row eight ids at a time (`s_load_b256`, branch-free first-match
//! `s_min_u32` over the valid lanes of the group), and an entry's probability is
//! loaded only on a hit. A group may read up to seven ids past `counts[row]`,
//! always inside the row's 64 slots for `counts[row] <= 64`.
use super::common::{add64, mem, op, s, s_add_i32, smem, sop, sr, v};
use crate::{
    insn::{Instruction, MemoryClass},
    plan::Access,
    reg::Live,
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};
use peacemaker_author::{Gfx1151, Gfx1201, LoopExit, Scc, Target, Uniform, Wave, Workgroup};

pub fn symbol(arch: Arch) -> String {
    format!("spec_accept_ordered_u32_pm_{}", arch.name())
}
pub fn module(arch: Arch) -> String {
    format!("spec_accept_ordered_pm_{}", arch.name())
}

// SGPRs.
const KARG: u8 = 0; // kernarg pointer 0:1
const WGX_IN: u8 = 2; // gfx11 system SGPR: workgroup id x
const TC: u8 = 8; // tcounts 8:9, tids TI 10:11, tprobs TP 12:13, dcounts DC 14:15
const TI: u8 = 10;
const TP: u8 = 12;
const DC: u8 = 14;
const A1: u8 = 16; // dids DI 16:17, dprobs DP 18:19, proposals PR 20:21, uniforms UN 22:23
const DI: u8 = 16;
const DP: u8 = 18;
const PR: u8 = 20;
const UN: u8 = 22;
const A2: u8 = 24; // result RS 24:25, k K 26, eos EOS 27
const RS: u8 = 24;
const K: u8 = 26;
const EOS: u8 = 27;
const ROW: u8 = 28;
const DRAFT: u8 = 29;
const DONE: u8 = 30;
const RES0: u8 = 31;
const RES1: u8 = 32;
const WGX: u8 = 33;
const NT: u8 = 34;
const ND: u8 = 35;
const UB: u8 = 36;
const U2B: u8 = 37;
const QXB: u8 = 38;
const PXB: u8 = 39;
const T0: u8 = 40;
const TIR: u8 = 42;
const TPR: u8 = 44;
const DIR: u8 = 46;
const DPR: u8 = 48; // this row's id / probability bases
const PA: u8 = 50;
const PA2: u8 = 52; // address temporaries
const L_IDX: u8 = 54;
const L_BASE: u8 = 55;
const L_CNT: u8 = 56;
const L_T0: u8 = 57;
const M1: u8 = 58;
const M2: u8 = 60;
const M3: u8 = 62; // lane-mask pairs
const G: u8 = 64; // eight ids of a search group 64:71
const I: u8 = 72;
const PID: u8 = 73;
const PPB: u8 = 74;
const QQB: u8 = 75;
const LAST: u8 = 77;
const HAVE: u8 = 78;
const DONE2: u8 = 79;
const TOKEN: u8 = 80;
const SROW: u8 = 81;
const SU: u8 = 82;
const NS: u8 = 83;
const PB: u8 = 84;
const FOUND: u8 = 85;
const SI: u8 = 86;
const SIR: u8 = 88;
const SPR: u8 = 90;

// VGPRs (v0 is the hardware lane id, unused).
const VQ: u8 = 1;
const VPX: u8 = 2;
const VU: u8 = 3;
const VPROD: u8 = 4;
const VQ2: u8 = 5;
const VPP: u8 = 6;
const VR: u8 = 7;
const VMASS: u8 = 8;
const VTARG: u8 = 9;
const VACC: u8 = 10;
const VU2: u8 = 11;
const VSU: u8 = 12;
const VSACC: u8 = 13;
const VSP: u8 = 14;
const VOFF: u8 = 15;
const VD0: u8 = 16;
const VD1: u8 = 17;

fn kernargs() -> KernargLayout {
    let mut l = KernargLayout::new(80);
    for (i, name) in [
        "tcounts",
        "tids",
        "tprobs",
        "dcounts",
        "dids",
        "dprobs",
        "proposals",
        "uniforms",
    ]
    .into_iter()
    .enumerate()
    {
        l = l.pointer_access(name, 8 * i as u32, Access::ReadOnly);
    }
    l.pointer_access("result", 64, Access::WriteOnly)
        .hidden("k", 72, 4, "by_value")
        .hidden("eos", 76, 4, "by_value")
}

fn plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(64, 104)?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    p.s::<1>("wg_x_in", WGX_IN, Live::Whole)?;
    p.s::<8>("kernargs_0x00", TC, Live::Whole)?;
    p.s::<8>("kernargs_0x20", A1, Live::Whole)?;
    p.s::<4>("kernargs_0x40", A2, Live::Whole)?;
    for (name, r) in [
        ("row", ROW),
        ("draft", DRAFT),
        ("done", DONE),
        ("res0", RES0),
        ("res1", RES1),
        ("wg_x", WGX),
        ("target_n", NT),
        ("draft_n", ND),
        ("u_bits", UB),
        ("u2_bits", U2B),
        ("qx_bits", QXB),
        ("px_bits", PXB),
        ("t0", T0),
        ("l_idx", L_IDX),
        ("l_base", L_BASE),
        ("l_cnt", L_CNT),
        ("l_t0", L_T0),
        ("i", I),
        ("pid", PID),
        ("pp_bits", PPB),
        ("qq_bits", QQB),
        ("last", LAST),
        ("have", HAVE),
        ("done2", DONE2),
        ("token", TOKEN),
        ("s_row", SROW),
        ("s_u_bits", SU),
        ("s_n", NS),
        ("s_p_bits", PB),
        ("found", FOUND),
        ("s_i", SI),
    ] {
        p.s::<1>(name, r, Live::Whole)?;
    }
    for (name, r) in [
        ("target_ids", TIR),
        ("target_probs", TPR),
        ("draft_ids", DIR),
        ("draft_probs", DPR),
        ("addr", PA),
        ("addr2", PA2),
        ("mask_a", M1),
        ("mask_b", M2),
        ("mask_c", M3),
        ("s_ids", SIR),
        ("s_probs", SPR),
    ] {
        p.s::<2>(name, r, Live::Whole)?;
    }
    p.s::<8>("id_group", G, Live::Whole)?;
    for (name, r) in [
        ("vq", VQ),
        ("vpx", VPX),
        ("vu", VU),
        ("vprod", VPROD),
        ("vq2", VQ2),
        ("vpp", VPP),
        ("vr", VR),
        ("vmass", VMASS),
        ("vtarget", VTARG),
        ("vacc", VACC),
        ("vu2", VU2),
        ("vsu", VSU),
        ("vsacc", VSACC),
        ("vsp", VSP),
        ("voff", VOFF),
        ("vd0", VD0),
        ("vd1", VD1),
    ] {
        p.v::<1>(name, r, Live::Whole)?;
    }
    let top = p.next_free_vgpr();
    if top % 8 != 0 {
        p.v::<1>("granule_pad", (top.div_ceil(8) * 8 - 1) as u8, Live::Whole)?;
    }
    Ok(p)
}

/// The builder-emitted kernel (gfx1151 and gfx1201; every other target is refused).
pub fn emit(arch: Arch) -> Result<Emitted, String> {
    let spec = KernelSpec {
        kernel_id: "spec_accept_pm".into(),
        variant: "ordered_u32".into(),
        arch,
        symbol: symbol(arch),
        kernargs: kernargs(),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 32,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(spec, plan()?);
    match arch {
        Arch::Gfx1151 => author(&mut Workgroup::<Gfx1151, Builder>::new(&mut b)?, arch)?,
        Arch::Gfx1201 => author(&mut Workgroup::<Gfx1201, Builder>::new(&mut b)?, arch)?,
        a => {
            return Err(format!(
                "spec_accept_pm: built for gfx1151 and gfx1201 only, not {}",
                a.name()
            ))
        }
    }
    b.finish()
}

/// The single-symbol module `spec_accept_ordered_pm_<arch>`.
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = vec![emit(arch)?];
    let (text, proof) = super::iu4_gemm::module(&emitted, &module(arch))?;
    Ok((emitted, text, proof))
}

// ---- instruction helpers ---------------------------------------------------

fn s_sub_i32(a: Arch) -> &'static str {
    if a.gfx12() {
        "s_sub_co_i32"
    } else {
        "s_sub_i32"
    }
}

fn scmp<T: Target>(
    w: &mut Wave<'_, T, Builder>,
    text: String,
    uses: &[u8],
) -> Result<Uniform<Scc>, String> {
    w.scmp(Instruction::new(
        text,
        vec![],
        uses.iter().map(|&n| s(n)).collect(),
    ))
}

/// `v{dst} = bits of s{src}`.
fn vmov(b: &mut Builder, dst: u8, src: u8) -> Result<(), String> {
    op(
        b,
        format!("v_mov_b32_e32 v{dst}, s{src}"),
        &[v(dst)],
        &[s(src)],
    )
}

fn vzero(b: &mut Builder, dst: u8) -> Result<(), String> {
    op(b, format!("v_mov_b32_e32 v{dst}, 0"), &[v(dst)], &[])
}

/// `v{dst} = max(0, v{dst})`, hipcc's `fmaxf(x, 0.0f)`.
fn max0(b: &mut Builder, a: Arch, dst: u8) -> Result<(), String> {
    let m = if a.gfx12() {
        "v_max_num_f32_e32"
    } else {
        "v_max_f32_e32"
    };
    op(b, format!("{m} v{dst}, 0, v{dst}"), &[v(dst)], &[v(dst)])
}

/// `s{m} = lane mask of (lhs < rhs)` over two VGPR operands or an inline constant.
fn vlt(b: &mut Builder, m: u8, lhs: &str, rhs: &str, uses: &[u8]) -> Result<(), String> {
    let uses: Vec<_> = uses.iter().map(|&n| v(n)).collect();
    op(
        b,
        format!("v_cmp_lt_f32_e64 s{m}, {lhs}, {rhs}"),
        &[sr(m, 2)],
        &uses,
    )
}

// ---- the kernel ------------------------------------------------------------

fn author<T: Target>(wg: &mut Workgroup<'_, T, Builder>, a: Arch) -> Result<(), String> {
    let end = wg.exit(".Lacc_end")?;
    let b = wg.isa();
    smem(b, TC, 8, KARG, 0x00)?;
    smem(b, A1, 8, KARG, 0x20)?;
    smem(b, A2, 4, KARG, 0x40)?;
    // The twin's `threadIdx.x != 0` return: only lane 0 runs.
    op(b, "s_mov_b32 exec_lo, 1", &[], &[])?;
    if a.gfx12() {
        sop(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[WGX], &[])?;
    } else {
        sop(b, format!("s_mov_b32 s{WGX}, s{WGX_IN}"), &[WGX], &[WGX_IN])?;
    }
    // The twin's `blockIdx.x != 0` return.
    let off = wg.scmp_wg_uniform(Instruction::new(
        format!("s_cmp_lg_u32 s{WGX}, 0"),
        vec![],
        vec![s(WGX)],
    ))?;
    wg.exit_if(off, &end)?;
    let b = wg.isa();
    sop(b, format!("s_mov_b32 s{ROW}, 0"), &[ROW], &[])?;
    sop(b, format!("s_mov_b32 s{DONE}, 0"), &[DONE], &[])?;
    sop(b, format!("s_mov_b32 s{RES0}, 0"), &[RES0], &[])?;
    sop(b, format!("s_mov_b32 s{RES1}, 0"), &[RES1], &[])?;
    b.wait_all()?;
    wg.loop_until(".Lacc_row", ".Lacc_rows_done", |w, x| row_body(w, x, a))?;
    // No rejection and no accepted EOS: the bonus token from the target row k.
    let b = wg.isa();
    b.wait_all()?;
    let stop = wg.scmp(Instruction::new(
        format!("s_cmp_eq_u32 s{DONE}, 1"),
        vec![],
        vec![s(DONE)],
    ))?;
    wg.skip_if(stop, ".Lacc_bonus_done", (), |w, ()| bonus(w, a))?;
    // result[0], result[1]; lane 0 stores.
    let b = wg.isa();
    vzero(b, VOFF)?;
    vmov(b, VD0, RES0)?;
    vmov(b, VD1, RES1)?;
    mem(
        b,
        format!("global_store_b32 v{VOFF}, v{VD0}, s[{RS}:{}]", RS + 1),
        &[],
        &[v(VOFF), v(VD0), sr(RS, 2)],
        MemoryClass::VmemStore,
    )?;
    mem(
        b,
        format!(
            "global_store_b32 v{VOFF}, v{VD1}, s[{RS}:{}] offset:4",
            RS + 1
        ),
        &[],
        &[v(VOFF), v(VD1), sr(RS, 2)],
        MemoryClass::VmemStore,
    )?;
    b.wait_all()?;
    wg.end(end)
}

/// One row of the accept loop: leave when all `k` rows are accepted, else
/// verdict, then the accepted continuation or the residual terminal.
fn row_body<T: Target>(
    w: &mut Wave<'_, T, Builder>,
    x: &LoopExit<Builder>,
    a: Arch,
) -> Result<(), String> {
    let b = w.isa();
    sop(b, format!("s_mov_b32 s{DONE}, 0"), &[DONE], &[])?;
    let all = scmp(w, format!("s_cmp_ge_i32 s{ROW}, s{K}"), &[ROW, K])?;
    w.break_if(all, x)?;
    let b = w.isa();
    // This row's bases and scalars: ids / probs rows are 64 slots (256 bytes).
    sop(b, format!("s_lshl_b32 s{T0}, s{ROW}, 8"), &[T0], &[ROW])?;
    add64(b, TIR, TI, T0)?;
    add64(b, TPR, TP, T0)?;
    add64(b, DIR, DI, T0)?;
    add64(b, DPR, DP, T0)?;
    sop(b, format!("s_lshl_b32 s{T0}, s{ROW}, 2"), &[T0], &[ROW])?;
    add64(b, PA, TC, T0)?;
    smem(b, NT, 1, PA, 0)?;
    add64(b, PA2, DC, T0)?;
    smem(b, ND, 1, PA2, 0)?;
    add64(b, PA, PR, T0)?;
    smem(b, DRAFT, 1, PA, 0)?;
    add64(b, PA2, UN, T0)?;
    smem(b, UB, 1, PA2, 0)?;
    smem(b, U2B, 1, PA2, 4)?;
    b.wait_all()?;
    lookup(w, a, "qx", DIR, DPR, ND, DRAFT, QXB)?;
    lookup(w, a, "px", TIR, TPR, NT, DRAFT, PXB)?;
    // Verdict: qx > 0 && RN(u * qx) < px (a NaN compares false).
    let b = w.isa();
    vmov(b, VQ, QXB)?;
    vmov(b, VPX, PXB)?;
    vmov(b, VU, UB)?;
    vlt(b, M1, "0", &format!("v{VQ}"), &[VQ])?;
    op(
        b,
        format!("v_mul_f32_e32 v{VPROD}, v{VU}, v{VQ}"),
        &[v(VPROD)],
        &[v(VU), v(VQ)],
    )?;
    vlt(
        b,
        M2,
        &format!("v{VPROD}"),
        &format!("v{VPX}"),
        &[VPROD, VPX],
    )?;
    sop(
        b,
        format!("s_and_b32 s{M1}, s{M1}, s{M2}"),
        &[M1],
        &[M1, M2],
    )?;
    let rejected = scmp(w, format!("s_cmp_eq_u32 s{M1}, 0"), &[M1])?;
    w.if_else(
        rejected,
        ".Lacc_reject",
        ".Lacc_row_join",
        |w| {
            // Accepted: row + 1; an accepted EOS ends with (row + 1, 0xFFFFFFFF).
            let b = w.isa();
            sop(
                b,
                format!("{} s{T0}, s{ROW}, 1", s_add_i32(a)),
                &[T0],
                &[ROW],
            )?;
            sop(
                b,
                format!("s_cmp_eq_u32 s{DRAFT}, s{EOS}"),
                &[],
                &[DRAFT, EOS],
            )?;
            sop(b, format!("s_cselect_b32 s{DONE}, 1, 0"), &[DONE], &[])?;
            sop(
                b,
                format!("s_cselect_b32 s{RES0}, s{T0}, s{RES0}"),
                &[RES0],
                &[T0, RES0],
            )?;
            sop(
                b,
                format!("s_cselect_b32 s{RES1}, -1, s{RES1}"),
                &[RES1],
                &[RES1],
            )?;
            sop(b, format!("s_mov_b32 s{ROW}, s{T0}"), &[ROW], &[T0])
        },
        |w| residual(w, a),
    )?;
    w.label(".Lacc_row_join")?;
    let b = w.isa();
    b.wait_all()?;
    let fin = scmp(w, format!("s_cmp_eq_u32 s{DONE}, 1"), &[DONE])?;
    w.break_if(fin, x)
}

/// `out = prob(row, tok)`: the probability of the first entry `i < n` (signed)
/// of the row (`ids` / `probs` row bases) with `ids[i] == tok`, else 0.
/// The ids are searched eight at a time.
#[allow(clippy::too_many_arguments)]
fn lookup<T: Target>(
    w: &mut Wave<'_, T, Builder>,
    a: Arch,
    tag: &str,
    ids: u8,
    probs: u8,
    n: u8,
    tok: u8,
    out: u8,
) -> Result<(), String> {
    let b = w.isa();
    sop(b, format!("s_mov_b32 s{L_IDX}, -1"), &[L_IDX], &[])?;
    sop(b, format!("s_mov_b32 s{L_BASE}, 0"), &[L_BASE], &[])?;
    sop(b, format!("s_mov_b32 s{out}, 0"), &[out], &[])?;
    b.wait_all()?;
    w.loop_until(
        &format!(".Lacc_{tag}_scan"),
        &format!(".Lacc_{tag}_scanned"),
        |w, x| {
            let exhausted = scmp(w, format!("s_cmp_ge_i32 s{L_BASE}, s{n}"), &[L_BASE, n])?;
            w.break_if(exhausted, x)?;
            let b = w.isa();
            sop(
                b,
                format!("s_lshl_b32 s{L_T0}, s{L_BASE}, 2"),
                &[L_T0],
                &[L_BASE],
            )?;
            add64(b, PA, ids, L_T0)?;
            smem(b, G, 8, PA, 0)?;
            sop(
                b,
                format!("{} s{L_CNT}, s{n}, s{L_BASE}", s_sub_i32(a)),
                &[L_CNT],
                &[n, L_BASE],
            )?;
            sop(
                b,
                format!("s_min_i32 s{L_CNT}, s{L_CNT}, 8"),
                &[L_CNT],
                &[L_CNT],
            )?;
            for j in (0..8u8).rev() {
                // idx = min(idx, i + j) over the valid lanes whose id is the token.
                sop(
                    b,
                    format!("{} s{L_T0}, s{L_BASE}, {j}", s_add_i32(a)),
                    &[L_T0],
                    &[L_BASE],
                )?;
                sop(
                    b,
                    format!("s_cmp_eq_u32 s{}, s{tok}", G + j),
                    &[],
                    &[G + j, tok],
                )?;
                sop(
                    b,
                    format!("s_cselect_b32 s{L_T0}, s{L_T0}, -1"),
                    &[L_T0],
                    &[L_T0],
                )?;
                sop(b, format!("s_cmp_gt_i32 s{L_CNT}, {j}"), &[], &[L_CNT])?;
                sop(
                    b,
                    format!("s_cselect_b32 s{L_T0}, s{L_T0}, -1"),
                    &[L_T0],
                    &[L_T0],
                )?;
                sop(
                    b,
                    format!("s_min_u32 s{L_IDX}, s{L_IDX}, s{L_T0}"),
                    &[L_IDX],
                    &[L_IDX, L_T0],
                )?;
            }
            let found = scmp(w, format!("s_cmp_lg_u32 s{L_IDX}, -1"), &[L_IDX])?;
            w.break_if(found, x)?;
            let b = w.isa();
            sop(
                b,
                format!("{} s{L_BASE}, s{L_BASE}, 8", s_add_i32(a)),
                &[L_BASE],
                &[L_BASE],
            )
        },
    )?;
    let miss = scmp(w, format!("s_cmp_eq_u32 s{L_IDX}, -1"), &[L_IDX])?;
    w.skip_if(miss, &format!(".Lacc_{tag}_miss"), (), |w, ()| {
        let b = w.isa();
        sop(
            b,
            format!("s_lshl_b32 s{L_T0}, s{L_IDX}, 2"),
            &[L_T0],
            &[L_IDX],
        )?;
        add64(b, PA, probs, L_T0)?;
        smem(b, out, 1, PA, 0)?;
        b.wait_all()
    })
}

/// The target row's entry `I`: `PID` its id, `VR = max0(RN(p - q(PID)))` with
/// `q` the draft row's probability of that id.
fn entry<T: Target>(w: &mut Wave<'_, T, Builder>, a: Arch, tag: &str) -> Result<(), String> {
    let b = w.isa();
    sop(b, format!("s_lshl_b32 s{T0}, s{I}, 2"), &[T0], &[I])?;
    add64(b, PA, TIR, T0)?;
    smem(b, PID, 1, PA, 0)?;
    add64(b, PA2, TPR, T0)?;
    smem(b, PPB, 1, PA2, 0)?;
    b.wait_all()?;
    lookup(w, a, tag, DIR, DPR, ND, PID, QQB)?;
    let b = w.isa();
    vmov(b, VPP, PPB)?;
    vmov(b, VQ2, QQB)?;
    op(
        b,
        format!("v_sub_f32_e32 v{VR}, v{VPP}, v{VQ2}"),
        &[v(VR)],
        &[v(VPP), v(VQ2)],
    )?;
    max0(b, a, VR)
}

/// The rejection terminal of row `ROW`: `sample_residual(p, q, u2)`, then
/// `result = (ROW, token)` and `DONE = 1`.
fn residual<T: Target>(w: &mut Wave<'_, T, Builder>, a: Arch) -> Result<(), String> {
    let b = w.isa();
    vzero(b, VMASS)?;
    sop(b, format!("s_mov_b32 s{I}, 0"), &[I], &[])?;
    b.wait_all()?;
    // mass = ((0 + r_0) + r_1) + ..
    w.loop_until(".Lacc_mass", ".Lacc_mass_done", |w, x| {
        let end = scmp(w, format!("s_cmp_ge_i32 s{I}, s{NT}"), &[I, NT])?;
        w.break_if(end, x)?;
        entry(w, a, "ms")?;
        let b = w.isa();
        op(
            b,
            format!("v_add_f32_e32 v{VMASS}, v{VMASS}, v{VR}"),
            &[v(VMASS)],
            &[v(VMASS), v(VR)],
        )?;
        sop(b, format!("{} s{I}, s{I}, 1", s_add_i32(a)), &[I], &[I])
    })?;
    let b = w.isa();
    vmov(b, VU2, U2B)?;
    vlt(b, M1, "0", &format!("v{VMASS}"), &[VMASS])?;
    op(
        b,
        format!("v_mul_f32_e32 v{VTARG}, v{VU2}, v{VMASS}"),
        &[v(VTARG)],
        &[v(VU2), v(VMASS)],
    )?;
    vzero(b, VACC)?;
    for r in [I, PID, LAST, HAVE, DONE2] {
        sop(b, format!("s_mov_b32 s{r}, 0"), &[r], &[])?;
    }
    b.wait_all()?;
    // Selection pass, only when mass > 0 (a NaN mass takes the sample path).
    let flat = scmp(w, format!("s_cmp_eq_u32 s{M1}, 0"), &[M1])?;
    w.skip_if(flat, ".Lacc_select_skip", (), |w, ()| {
        w.loop_until(".Lacc_select", ".Lacc_select_done", |w, x| {
            let end = scmp(w, format!("s_cmp_ge_i32 s{I}, s{NT}"), &[I, NT])?;
            w.break_if(end, x)?;
            entry(w, a, "sl")?;
            let b = w.isa();
            // r > 0 (an entry with r <= 0 is skipped by the twin: acc is unchanged
            // bitwise by + (+-0), and it can neither hit nor become `last`).
            vlt(b, M3, "0", &format!("v{VR}"), &[VR])?;
            op(
                b,
                format!("v_add_f32_e32 v{VACC}, v{VACC}, v{VR}"),
                &[v(VACC)],
                &[v(VACC), v(VR)],
            )?;
            vlt(
                b,
                M2,
                &format!("v{VTARG}"),
                &format!("v{VACC}"),
                &[VTARG, VACC],
            )?;
            sop(
                b,
                format!("s_and_b32 s{M2}, s{M2}, s{M3}"),
                &[M2],
                &[M2, M3],
            )?;
            sop(b, format!("s_cmp_lg_u32 s{M3}, 0"), &[], &[M3])?;
            sop(
                b,
                format!("s_cselect_b32 s{LAST}, s{PID}, s{LAST}"),
                &[LAST],
                &[PID, LAST],
            )?;
            sop(
                b,
                format!("s_cselect_b32 s{HAVE}, 1, s{HAVE}"),
                &[HAVE],
                &[HAVE],
            )?;
            sop(b, format!("s_mov_b32 s{DONE2}, 1"), &[DONE2], &[])?;
            let hit = scmp(w, format!("s_cmp_lg_u32 s{M2}, 0"), &[M2])?;
            w.break_if(hit, x)?;
            let b = w.isa();
            sop(b, format!("s_mov_b32 s{DONE2}, 0"), &[DONE2], &[])?;
            sop(b, format!("{} s{I}, s{I}, 1", s_add_i32(a)), &[I], &[I])
        })
    })?;
    // token = the hit entry, else the last entry with r > 0, else sample(p, u2).
    let b = w.isa();
    sop(b, format!("s_cmp_eq_u32 s{DONE2}, 1"), &[], &[DONE2])?;
    sop(
        b,
        format!("s_cselect_b32 s{TOKEN}, s{PID}, s{LAST}"),
        &[TOKEN],
        &[PID, LAST],
    )?;
    sop(
        b,
        format!("s_or_b32 s{T0}, s{DONE2}, s{HAVE}"),
        &[T0],
        &[DONE2, HAVE],
    )?;
    let resolved = scmp(w, format!("s_cmp_lg_u32 s{T0}, 0"), &[T0])?;
    w.skip_if(resolved, ".Lacc_sample_skip", (), |w, ()| {
        let b = w.isa();
        sop(b, format!("s_mov_b32 s{SROW}, s{ROW}"), &[SROW], &[ROW])?;
        sop(b, format!("s_mov_b32 s{SU}, s{U2B}"), &[SU], &[U2B])?;
        sample(w, a, "rs")
    })?;
    let b = w.isa();
    sop(b, format!("s_mov_b32 s{DONE}, 1"), &[DONE], &[])?;
    sop(b, format!("s_mov_b32 s{RES0}, s{ROW}"), &[RES0], &[ROW])?;
    sop(b, format!("s_mov_b32 s{RES1}, s{TOKEN}"), &[RES1], &[TOKEN])
}

/// All rows accepted: `result = (k, sample(target row k, uniforms[k]))`.
fn bonus<T: Target>(w: &mut Wave<'_, T, Builder>, a: Arch) -> Result<(), String> {
    let b = w.isa();
    sop(b, format!("s_mov_b32 s{RES0}, s{K}"), &[RES0], &[K])?;
    sop(b, format!("s_mov_b32 s{SROW}, s{K}"), &[SROW], &[K])?;
    sop(b, format!("s_lshl_b32 s{T0}, s{K}, 2"), &[T0], &[K])?;
    add64(b, PA, UN, T0)?;
    smem(b, SU, 1, PA, 0)?;
    b.wait_all()?;
    sample(w, a, "bn")?;
    let b = w.isa();
    sop(b, format!("s_mov_b32 s{RES1}, s{TOKEN}"), &[RES1], &[TOKEN])
}

/// `TOKEN = SparseDist::sample(u)` over the target row `SROW` with the draw
/// bits `SU`: the first entry with `u < (acc += p_i)`, else the last entry,
/// else 0 for an empty row.
fn sample<T: Target>(w: &mut Wave<'_, T, Builder>, a: Arch, tag: &str) -> Result<(), String> {
    let b = w.isa();
    sop(b, format!("s_lshl_b32 s{T0}, s{SROW}, 2"), &[T0], &[SROW])?;
    add64(b, PA, TC, T0)?;
    smem(b, NS, 1, PA, 0)?;
    sop(b, format!("s_lshl_b32 s{T0}, s{SROW}, 8"), &[T0], &[SROW])?;
    add64(b, SIR, TI, T0)?;
    add64(b, SPR, TP, T0)?;
    vmov(b, VSU, SU)?;
    vzero(b, VSACC)?;
    sop(b, format!("s_mov_b32 s{SI}, 0"), &[SI], &[])?;
    sop(b, format!("s_mov_b32 s{FOUND}, 0"), &[FOUND], &[])?;
    b.wait_all()?;
    w.loop_until(
        &format!(".Lacc_{tag}_cdf"),
        &format!(".Lacc_{tag}_cdf_done"),
        |w, x| {
            let spent = scmp(w, format!("s_cmp_ge_i32 s{SI}, s{NS}"), &[SI, NS])?;
            w.break_if(spent, x)?;
            let b = w.isa();
            sop(b, format!("s_lshl_b32 s{T0}, s{SI}, 2"), &[T0], &[SI])?;
            add64(b, PA, SPR, T0)?;
            smem(b, PB, 1, PA, 0)?;
            vmov(b, VSP, PB)?;
            op(
                b,
                format!("v_add_f32_e32 v{VSACC}, v{VSACC}, v{VSP}"),
                &[v(VSACC)],
                &[v(VSACC), v(VSP)],
            )?;
            vlt(
                b,
                M1,
                &format!("v{VSU}"),
                &format!("v{VSACC}"),
                &[VSU, VSACC],
            )?;
            sop(b, format!("s_mov_b32 s{FOUND}, 1"), &[FOUND], &[])?;
            let hit = scmp(w, format!("s_cmp_lg_u32 s{M1}, 0"), &[M1])?;
            w.break_if(hit, x)?;
            let b = w.isa();
            sop(b, format!("s_mov_b32 s{FOUND}, 0"), &[FOUND], &[])?;
            sop(b, format!("{} s{SI}, s{SI}, 1", s_add_i32(a)), &[SI], &[SI])
        },
    )?;
    let b = w.isa();
    sop(b, format!("s_mov_b32 s{TOKEN}, 0"), &[TOKEN], &[])?;
    let any = scmp(w, format!("s_cmp_gt_i32 s{NS}, 0"), &[NS])?;
    w.skip_unless(any, &format!(".Lacc_{tag}_empty"), (), |w, ()| {
        let b = w.isa();
        sop(b, format!("{} s{T0}, s{NS}, 1", s_sub_i32(a)), &[T0], &[NS])?;
        sop(b, format!("s_cmp_eq_u32 s{FOUND}, 1"), &[], &[FOUND])?;
        sop(
            b,
            format!("s_cselect_b32 s{T0}, s{SI}, s{T0}"),
            &[T0],
            &[SI, T0],
        )?;
        sop(b, format!("s_lshl_b32 s{T0}, s{T0}, 2"), &[T0], &[T0])?;
        add64(b, PA, SIR, T0)?;
        smem(b, TOKEN, 1, PA, 0)?;
        b.wait_all()
    })
}

/// Shape contract of the emitted kernel (the `--shape` input of
/// `peacemaker-isa emit --co`): exact counts of the memory and control mnemonics
/// the kernel is allowed, the forbidden families and the resource limits.
pub fn contract(arch: Arch, e: &Emitted) -> serde_json::Value {
    let mut counts = serde_json::Map::new();
    for name in [
        "s_load_b32",
        "s_load_b128",
        "s_load_b256",
        "global_store_b32",
        "s_endpgm",
    ] {
        let n = e
            .s_text
            .lines()
            .filter(|l| l.trim().split_whitespace().next() == Some(name))
            .count();
        counts.insert(name.into(), n.into());
    }
    serde_json::json!({
        "symbol": symbol(arch), "counts": counts,
        "forbidden": ["scratch_*", "flat_*", "global_load*", "global_atomic*", "buffer_*", "ds_*", "s_barrier*", "s_setpc*", "s_swappc*", "v_writelane*"],
        "vgpr_max": 32, "sgpr_max": 104,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0,
        "group_segment_fixed_bytes": 0,
    })
}
