// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted (PeaceMaker) gfx1201 row argmax: `argmax_f32_batched_regrid_pm_gfx1201`,
//! the byte twin of the hipcc kernel `argmax_f32_batched_regrid`
//! (`kernels/src/select_regrid.hip`), which is itself byte-identical to the shipping
//! `argmax_f32_batched`.
//!
//! ABI (three arguments, 20 bytes): data `*f32` @0 (`[batch, n]` row-major), result `*i32` @8
//! (`[batch]`), n i32 @16. Grid `[batch, 1, 1]`, block 1024 (one row per workgroup), 256 bytes of
//! static LDS, no scratch, no atomics.
//!
//! Law (identical to the HIP twin, see its header): an element qualifies when `v > -1e30`
//! (NaN, -inf and everything at or below -1e30 never qualify). The winner is the maximum under
//! the total order "larger value first; on equal values (`-0 == +0`) the lower bit-reversed
//! stride-256 lane `rev8(i & 255)`, then the lower index `i`". A row with no qualifying element
//! writes 0.
//!
//! Algorithm. Thread `t` visits `i = t + 1024 j` (four loads in flight per trip) and keeps the
//! best `(value, key)` where `key = rev8(i & 255) << 24 | i >> 8`; `rev8(i & 255)` is injective
//! on the low byte, so equal leading bytes mean equal low bytes and `key` orders exactly like
//! `(rev8(i & 255), i)`, as one u32 compare. A wave takes the best of its 32 lanes with a
//! five-level exchange butterfly (`row_xmask` DPP and `v_permlanex16_b32`), the 32 wave winners
//! meet through 256 bytes of LDS (one workgroup barrier each way) and every wave takes the best
//! of the 32. Thread 0 stores `index = decode(key)` when the winning value is `> -1e30`, else 0.
use super::common::{add64, bload, lit, mem, op, s, smem, sop, sr, srd_tail, v};
use crate::{
    insn::{Instruction, MemoryClass},
    reg::{Kind, Live, RegRef},
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};
use peacemaker_author::{ready, retire, Free, Gfx1201, LdsRegion, Target, Workgroup};

type Wg<'b, T> = Workgroup<'b, T, Builder>;
type R = Result<(), String>;

/// Threads per row (one workgroup per row).
pub const BLOCK: u32 = 1024;
/// Kernarg bytes.
pub const KERNARG_BYTES: u32 = 20;
/// Static LDS: 32 eight-byte (value, key) wave winners.
pub const STATIC_LDS: u32 = 256;

pub enum Ex {}

pub fn symbol(arch: Arch) -> String {
    format!("argmax_f32_batched_regrid_pm_{}", arch.name())
}
pub fn module(arch: Arch) -> String {
    format!("argmax_regrid_pm_{}", arch.name())
}

pub fn kernargs() -> KernargLayout {
    KernargLayout::new(KERNARG_BYTES)
        .pointer("data", 0)
        .pointer("result", 8)
        .hidden("n", 16, 4, "by_value")
}

// ---- SGPRs ------------------------------------------------------------------
const KARG: u8 = 0;
// kernarg: data 8:9, result 10:11, n 12.
const DATA: u8 = 8;
const RES: u8 = 10;
const NIN: u8 = 12;
const ROW: u8 = 16;
const TRIPS: u8 = 17;
const TRIP: u8 = 18;
const T0: u8 = 19;
const T1: u8 = 20;
const T2: u8 = 21;
const N: u8 = 22;
const N4: u8 = 23;
const SRD_D: u8 = 32;
const SRD_R: u8 = 36;
/// Lane-mask pairs (the high word is never written in wave32).
const MA: u8 = 40;
const MB: u8 = 42;
const MC: u8 = 44;
const MD: u8 = 46;
const MASKS: [u8; 4] = [MA, MB, MC, MD];
const PSEL: u8 = 48;
const S_END: u8 = 56;

// ---- VGPRs ------------------------------------------------------------------
const TID: u8 = 0;
const LANE: u8 = 1;
const VIDX: u8 = 2;
const VOFF: u8 = 3;
const LDSW: u8 = 4;
const LDSR: u8 = 5;
const VT: u8 = 6;
const VK: u8 = 7;
const BV: u8 = 8;
const BK: u8 = 9;
const CI: u8 = 10;
const VZ: u8 = 11;
const PV: u8 = 12;
const PK: u8 = 13;
/// Wave winner (value, key): a consecutive pair for the LDS store.
const WV: u8 = 14;
const WK: u8 = 15;
const X: u8 = 16; // four loads 16..19
/// Final (value, key) after the LDS exchange: a consecutive pair for the LDS load.
const FV: u8 = 20;
const FK: u8 = 21;
const V_END: u8 = 32;

/// `-1e30f` bits (the HIP `SEL` empty value).
const NEG_1E30: &str = "0xf149f2ca";

pub fn emit(arch: Arch) -> Result<Emitted, String> {
    if !matches!(arch, Arch::Gfx1201) {
        return Err(format!(
            "argmax_regrid_pm: built for gfx1201 only, not {}",
            arch.name()
        ));
    }
    let kspec = KernelSpec {
        kernel_id: "argmax_regrid_pm".into(),
        variant: "f32_batched".into(),
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

/// The single-symbol module `argmax_regrid_pm_<arch>`.
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
fn vr2(base: u8) -> RegRef {
    super::common::vr(base, 2)
}
fn ds_st(text: String, addr: u8, data: RegRef) -> Instruction {
    Instruction::new(text, vec![], vec![v(addr), data]).memory(MemoryClass::DsStore)
}
fn ds_ld(text: String, dst: RegRef, addr: u8) -> Instruction {
    Instruction::new(text, vec![dst], vec![v(addr)]).memory(MemoryClass::DsLoad)
}
fn add_imm(b: &mut Builder, dst: u8, src: u8, imm: u32) -> R {
    let mn = super::common::s_add_i32(b.spec.arch);
    sop(
        b,
        format!("{mn} s{dst}, s{src}, {}", lit(imm)),
        &[dst],
        &[src],
    )
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
/// Raw buffer store of one dword at `v{voff}` (soffset `null` on gfx12).
fn bstore(b: &mut Builder, data: u8, voff: u8, srd: u8) -> R {
    mem(
        b,
        format!(
            "buffer_store_b32 v{data}, v{voff}, s[{srd}:{}], null offen",
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
/// `(wv, wk) = best((pv, pk), (wv, wk))`: larger value, equal values the lower key. Clobbers MA, MB, MC.
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
        return Err("argmax_regrid_pm: gfx12 workgroup-id convention only".into());
    }
    let ex = wg.lds::<Ex>("ex", 0, STATIC_LDS)?;
    let end = wg.exit(".Largmax_end")?;
    prologue(wg)?;
    scan(wg)?;
    let b = wg.isa();
    alu(b, "v_mov_b32_e32", O::V(WV), &[O::V(BV)])?;
    alu(b, "v_mov_b32_e32", O::V(WK), &[O::V(BK)])?;
    wave_best(b, WV, WK)?;
    // The 32 wave winners.
    let st = wg.begin_write(ex);
    let (ew, pend) = wg.ds_store(
        st,
        ds_st(format!("ds_store_b64 v{LDSW}, v[{WV}:{WK}]"), LDSW, vr2(WV)),
    )?;
    let d = wg.wait(pend)?;
    let (ep,) = wg.barrier((ready(ew, d),))?;
    wg.ds_load(
        &ep,
        ds_ld(format!("ds_load_b64 v[{FV}:{FK}], v{LDSR}"), vr2(FV), LDSR),
    )?;
    let b = wg.isa();
    wave_best(b, FV, FK)?;
    let (ex,) = wg.barrier((retire(ep),))?;
    let _: LdsRegion<Ex, Free> = ex;
    let b = wg.isa();
    // index = (value > -1e30) ? decode(key) : 0, with decode(key) = (key & 0xffffff) << 8 | rev8(key >> 24).
    alu(b, "v_lshrrev_b32_e32", O::V(VT), &[O::N(24), O::V(FK)])?;
    alu(b, "v_bfrev_b32_e32", O::V(VT), &[O::V(VT)])?;
    alu(b, "v_lshrrev_b32_e32", O::V(VT), &[O::N(24), O::V(VT)])?;
    alu(b, "v_and_b32_e32", O::V(VK), &[O::N(0xff_ffff), O::V(FK)])?;
    alu(b, "v_lshlrev_b32_e32", O::V(VK), &[O::N(8), O::V(VK)])?;
    alu(b, "v_or_b32_e32", O::V(VT), &[O::V(VT), O::V(VK)])?;
    alu(b, "v_cmp_gt_f32_e64", O::M(MA), &[O::V(FV), O::K(NEG_1E30)])?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(VT),
        &[O::N(0), O::V(VT), O::M(MA)],
    )?;
    // Only thread 0 (wave 0, lane 0) stores.
    alu(b, "v_cmp_eq_u32_e64", O::M(MB), &[O::N(0), O::V(TID)])?;
    op(
        b,
        format!("s_and_b32 exec_lo, exec_lo, s{MB}"),
        &[],
        &[mask(MB)],
    )?;
    bstore(b, VT, VZ, SRD_R)?;
    b.wait_all()?;
    wg.end(end)
}

fn prologue<T: Target>(wg: &mut Wg<T>) -> R {
    let b = wg.isa();
    smem(b, DATA, 4, KARG, 0)?;
    smem(b, NIN, 1, KARG, 0x10)?;
    alu(b, "v_and_b32_e32", O::V(LANE), &[O::N(31), O::V(TID)])?;
    alu(b, "v_lshlrev_b32_e32", O::V(LDSR), &[O::N(3), O::V(LANE)])?;
    alu(b, "v_lshrrev_b32_e32", O::V(VT), &[O::N(5), O::V(TID)])?;
    alu(b, "v_lshlrev_b32_e32", O::V(LDSW), &[O::N(3), O::V(VT)])?;
    alu(b, "v_lshlrev_b32_e32", O::V(VOFF), &[O::N(2), O::V(TID)])?;
    alu(b, "v_mov_b32_e32", O::V(VIDX), &[O::V(TID)])?;
    alu(b, "v_mov_b32_e32", O::V(VZ), &[O::N(0)])?;
    alu(b, "v_mov_b32_e32", O::V(BV), &[O::K(NEG_1E30)])?;
    alu(b, "v_mov_b32_e32", O::V(BK), &[O::N(-1)])?;
    sop(b, format!("s_mov_b32 s{ROW}, ttmp9"), &[ROW], &[])?;
    for m in MASKS {
        sop(b, format!("s_mov_b32 s{}, 0", m + 1), &[m + 1], &[])?
    }
    sop(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[PSEL], &[])?;
    // n = max(n, 0); the row as a raw descriptor of n * 4 bytes.
    sop(b, format!("s_max_i32 s{N}, s{NIN}, 0"), &[N], &[NIN])?;
    sop(b, format!("s_lshl_b32 s{N4}, s{N}, 2"), &[N4], &[N])?;
    add64_mul(b, SRD_D, DATA, ROW, N4, T1, T2)?;
    srd_tail(b, SRD_D, Some(N4))?;
    // result[row].
    sop(b, format!("s_lshl_b32 s{T0}, s{ROW}, 2"), &[T0], &[ROW])?;
    sop(b, format!("s_mov_b32 s{T1}, 4"), &[T1], &[])?;
    add64(b, SRD_R, RES, T0)?;
    srd_tail(b, SRD_R, Some(T1))?;
    // trips = ceil(n / 4096).
    add_imm(b, T0, N, 4095)?;
    sop(
        b,
        format!("s_lshr_b32 s{TRIPS}, s{T0}, 12"),
        &[TRIPS],
        &[T0],
    )?;
    sop(b, format!("s_mov_b32 s{TRIP}, 0"), &[TRIP], &[])
}

/// Four loads in flight per trip; element `i = tid + 1024 j`.
fn scan<T: Target>(wg: &mut Wg<T>) -> R {
    let none = wg.scmp_wg_uniform(Instruction::new(
        format!("s_cmp_eq_u32 s{TRIPS}, 0"),
        vec![],
        vec![s(TRIPS)],
    ))?;
    wg.wg_skip_if(none, ".Largmax_scan_done", (), |wg, ()| {
        wg.loop_carried(".Largmax_scan", (), |wg, ()| {
            let b = wg.isa();
            for j in 0..4u8 {
                bload(b, 1, X + j, VOFF, SRD_D, 4096 * u32::from(j))?;
            }
            for j in 0..4u8 {
                let b = wg.isa();
                let idx = if j == 0 {
                    VIDX
                } else {
                    alu(
                        b,
                        "v_add_nc_u32_e32",
                        O::V(CI),
                        &[O::N(1024 * i64::from(j)), O::V(VIDX)],
                    )?;
                    CI
                };
                // In range, and (value, key) beats the thread's best.
                alu(b, "v_cmp_gt_u32_e64", O::M(MA), &[O::S(N), O::V(idx)])?;
                alu(b, "v_and_b32_e32", O::V(VT), &[O::N(255), O::V(idx)])?;
                alu(b, "v_bfrev_b32_e32", O::V(VT), &[O::V(VT)])?;
                alu(b, "v_lshrrev_b32_e32", O::V(VK), &[O::N(8), O::V(idx)])?;
                alu(b, "v_or_b32_e32", O::V(VK), &[O::V(VK), O::V(VT)])?;
                alu(b, "v_cmp_gt_f32_e64", O::M(MB), &[O::V(X + j), O::V(BV)])?;
                alu(b, "v_cmp_eq_f32_e64", O::M(MC), &[O::V(X + j), O::V(BV)])?;
                alu(b, "v_cmp_lt_u32_e64", O::M(MD), &[O::V(VK), O::V(BK)])?;
                alu(b, "s_and_b32", O::M(MC), &[O::M(MC), O::M(MD)])?;
                alu(b, "s_or_b32", O::M(MB), &[O::M(MB), O::M(MC)])?;
                alu(b, "s_and_b32", O::M(MA), &[O::M(MA), O::M(MB)])?;
                alu(
                    b,
                    "v_cndmask_b32_e64",
                    O::V(BV),
                    &[O::V(BV), O::V(X + j), O::M(MA)],
                )?;
                alu(
                    b,
                    "v_cndmask_b32_e64",
                    O::V(BK),
                    &[O::V(BK), O::V(VK), O::M(MA)],
                )?;
            }
            let b = wg.isa();
            alu(
                b,
                "v_add_nc_u32_e32",
                O::V(VIDX),
                &[O::K("0x1000"), O::V(VIDX)],
            )?;
            alu(
                b,
                "v_add_nc_u32_e32",
                O::V(VOFF),
                &[O::K("0x4000"), O::V(VOFF)],
            )?;
            add_imm(b, TRIP, TRIP, 1)?;
            let more = wg.scmp_wg_uniform(Instruction::new(
                format!("s_cmp_lt_u32 s{TRIP}, s{TRIPS}"),
                vec![],
                vec![s(TRIP), s(TRIPS)],
            ))?;
            Ok(((), more))
        })
    })
}

/// Mnemonics whose exact counts the shape contract pins.
const COUNTED: [&str; 12] = [
    "s_load_b128",
    "s_load_b32",
    "buffer_load_b32",
    "buffer_store_b32",
    "ds_store_b64",
    "ds_load_b64",
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
