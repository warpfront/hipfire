// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted gfx1201 device accept / residual / bonus of a sampled
//! singleton DFlash window (`HIPFIRE_DFLASH_DEVICE_ACCEPT`), one module
//! `dflash_sampled_accept_pm_gfx1201` with the symbol
//! `dflash_sampled_accept_sparse_q_f32_pm_gfx1201`: the kernarg ABI (`probs`,
//! `tau_rows`, `z_rows`, `payload`, `out`, `n`, `vocab`; 48 bytes), grid `[1]`,
//! block 256, payload layout, decision words and fallback codes of the hipcc
//! oracle `kernels/src/dflash_sampled_accept_sparse_q.hip` (which stays the
//! retained oracle and the host twin `dflash_sampled_accept_reference`). Every
//! other target is refused.
//!
//! Law and draw order are the oracle's, bit for bit (no fp contraction; every
//! add, multiply, subtract and compare is the plain IEEE op the oracle spells,
//! `1 / max(z, FLT_MIN)` the correctly rounded `v_div_scale` sequence):
//! - Accept tests run one row per lane (`ceil(lim / 256)` trips); the first
//!   rejected row is the minimum over a per-wave `ctz` ballot published
//!   through LDS (exact, association free).
//! - The categorical walk is over ascending vocabulary ids with the oracle's
//!   thread-count independent association: lane `t` owns the contiguous
//!   segment `[t * seg, min((t + 1) * seg, vocab))`, `seg = ceil(vocab / 256)`;
//!   segment sums accumulate sequentially from 0 (16 raw buffer loads in
//!   flight per chunk, per-lane validity masks past the segment end), the 256
//!   sums are chained in segment order by wave 0 (`base[t + 1] = base[t] +
//!   seg[t]`, unrolled, from broadcast LDS reads) and published with the
//!   decision words (`u * total` or `u_fin`, code 1 on a vanishing residual).
//! - Residual `d = p - q` merges the ascending sparse q row per lane: the
//!   lane keeps the next q id in a register (binary-searched once), the
//!   rare match runs a wave-uniform slow path (q value and next-id loads).
//! - Only the claiming segment re-walks (exec = claiming lane, break at the
//!   first id whose CDF exceeds `u`); code 1 re-walks every segment for the
//!   first-maximum argmax; falling off the end takes the last support id
//!   (max over the per-lane last positive ids) with code 2.
//! - Wave 0 resolves the decision (sequential strict-greater argmax or exact
//!   integer maxima over broadcast LDS reads) and lane 0 stores the four
//!   output words with one `buffer_store_b128`.
//!
//! Resources: wave32, workgroup 256, static LDS 6208 bytes, no dynamic LDS,
//! no scratch, no atomics; four workgroup barriers, all uniform. Raw buffer
//! descriptors bound every read to its exact extent (the payload to its
//! `4n + 3 + 2 nnz` words), so a masked lane never touches memory. Host
//! contract (checked by `dflash_accept_pm_fits`): `n >= 1`, `vocab >= 1`,
//! `(n + 1) * vocab * 4 < 2^32`, a payload that passes `validate_payload`.
use super::common::{
    bload, lit, op, s, s_add_i32, s_add_u32, s_addc_u32, smem, sop, sr, srd_tail, v, vr,
};
use crate::{
    insn::{Instruction, MemoryClass},
    reg::{Kind, Live, RegRef},
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};
use peacemaker_author::{
    ready, End, Free, Gfx1201, LdsRegion, LoopExit, Published, Scc, Uniform, Wave, WgUniform,
    Workgroup,
};
use std::cell::Cell;

type Wg<'b> = Workgroup<'b, Gfx1201, Builder>;
type W<'a> = Wave<'a, Gfx1201, Builder>;
type R = Result<(), String>;

pub fn symbol(arch: Arch) -> String {
    format!("dflash_sampled_accept_sparse_q_f32_pm_{}", arch.name())
}
pub fn module(arch: Arch) -> String {
    format!("dflash_sampled_accept_pm_{}", arch.name())
}

// ---- LDS layout (all bases 16-byte aligned) ---------------------------------
const MINR_BASE: u32 = 0; // 8 words: first rejected row per wave
const MINR_BYTES: u32 = 32;
const SEG_BASE: u32 = 32; // 256 f32 segment sums
const LAST_BASE: u32 = 1056; // 256 i32 last positive ids
const SEGL_BASE: u32 = 32;
const SEGL_BYTES: u32 = 2048;
const BASE_BASE: u32 = 2080; // 257 f32 chained bases, then the decision words
const CTL_BASE: u32 = 3112; // [us bits, code]
const BASE_BYTES: u32 = 1056;
const WIN_BASE: u32 = 3136; // 256 i32 claimed bonus ids (-1: none)
const BV_BASE: u32 = 4160; // 256 f32 segment argmax values
const BI_BASE: u32 = 5184; // 256 i32 segment argmax ids
const FIN_BASE: u32 = 3136;
const FIN_BYTES: u32 = 3072;
pub const STATIC_LDS: u32 = 6208;

pub enum Minr {}
pub enum Segl {}
pub enum Base {}
pub enum Fin {}

// ---- SGPRs ------------------------------------------------------------------
const KARG: u8 = 0;
const PROBS: u8 = 8;
const TAUP: u8 = 10;
const ZP: u8 = 12;
const PAY: u8 = 14;
const OUT: u8 = 16;
const N: u8 = 18;
const VOCAB: u8 = 19;
const SRD_PAY: u8 = 20;
const SRD_TAU: u8 = 24;
const SRD_Z: u8 = 28;
const SRD_PR: u8 = 32;
const SRD_ROW: u8 = 36;
const SRD_OUT: u8 = 40;
const UFIN: u8 = 44;
const MAXACC: u8 = 45;
const LIM: u8 = 46;
const NNZ: u8 = 47;
const PAYB: u8 = 48;
const TRIP: u8 = 49;
const WAVE: u8 = 50;
const REJ: u8 = 51;
const ROW: u8 = 52;
const ROWMIN: u8 = 53;
const TAU: u8 = 54;
const ZB: u8 = 55;
const INVZ: u8 = 56;
const QLO: u8 = 57;
const QN: u8 = 58;
const QIB: u8 = 59;
const QVB: u8 = 60;
const SEG: u8 = 61;
const NCHUNK: u8 = 62;
const KCNT: u8 = 63;
const US: u8 = 64;
const CODE: u8 = 65;
const WMIN: u8 = 66;
const N4: u8 = 67;
const SQB: u8 = 68;
const WAVE32: u8 = 69;
const VM1: u8 = 71;
const T0: u8 = 72;
const T1: u8 = 73;
const T2: u8 = 74;
const T3: u8 = 75;
const T4: u8 = 76;
const T5: u8 = 77;
const PTR: u8 = 80;
const PTR2: u8 = 82;
/// Lane-mask pairs (the high word is never written in wave32; M7 reads VOP3 masks as pairs).
const MA: u8 = 84;
const MB: u8 = 86;
const MC: u8 = 88;
const MD: u8 = 90;
const MQ: u8 = 92;
const MTAU: u8 = 94;
const FOUND: u8 = 96;
const ACT: u8 = 98;
const CLM: u8 = 100;
const MASKS: [u8; 9] = [MA, MB, MC, MD, MQ, MTAU, FOUND, ACT, CLM];

// ---- VGPRs ------------------------------------------------------------------
const TID: u8 = 0;
const VZERO: u8 = 1;
const TID4: u8 = 2;
const VWA: u8 = 3;
const VCTL: u8 = 4;
const START: u8 = 5;
const END: u8 = 6;
const REM: u8 = 7;
const OFF: u8 = 8;
const VL: u8 = 9;
const LAST: u8 = 10;
const QP: u8 = 11;
const NEXTQ: u8 = 12;
const VCB: u8 = 13;
const VCJ: u8 = 14;
const D: u8 = 15;
const VT0: u8 = 16;
const VT1: u8 = 17;
const VT2: u8 = 18;
const VT3: u8 = 19;
const VT4: u8 = 20;
const BONUS: u8 = 21;
const BV: u8 = 22;
const BI: u8 = 23;
const QP0: u8 = 24;
const NEXTQ0: u8 = 25;
const LO: u8 = 26;
const HI: u8 = 27;
const VB0: u8 = 28;
const VB1: u8 = 29;
const VUS: u8 = 30;
const VCODE: u8 = 31;
const P: u8 = 32; // 16 probability loads 32..47
const AI: u8 = 48;
const ATOK: u8 = 49;
const AU: u8 = 50;
const ASQ: u8 = 51;
const ATAU: u8 = 52;
const AZ: u8 = 53;
const AP: u8 = 54;
const AOFF: u8 = 55;
const AINV: u8 = 56;
const AD: u8 = 57;
const APR: u8 = 58;
const AT: u8 = 59;
const RC: [u8; 5] = [60, 61, 62, 63, 64];
const AOFF2: u8 = 65;
const VAB: u8 = 66;
const GV0: u8 = 72;
const GV1: u8 = 88; // two 16-register broadcast groups
const ACC: u8 = 104;
const ABV: u8 = 105;
const ABI: u8 = 106;
const FW: u8 = 107;
const OUTV: u8 = 108;
const V_END: u8 = 112;

const FLT_MIN: &str = "0x800000";
const NEG_INF: &str = "0xff800000";
const CHUNK: u32 = 16;

#[derive(Default)]
struct Gen {
    n: Cell<u32>,
}
impl Gen {
    fn lbl(&self, base: &str) -> String {
        let n = self.n.get();
        self.n.set(n + 1);
        format!(".Ldsa_{base}{n}")
    }
}

pub fn emit(arch: Arch) -> Result<Emitted, String> {
    if arch != Arch::Gfx1201 {
        return Err("dflash_accept: built for gfx1201 only".into());
    }
    let kernargs = KernargLayout::new(48)
        .pointer("probs", 0)
        .pointer("tau_rows", 8)
        .pointer("z_rows", 16)
        .pointer("payload", 24)
        .pointer("out", 32)
        .hidden("n", 40, 4, "by_value")
        .hidden("vocab", 44, 4, "by_value");
    let kspec = KernelSpec {
        kernel_id: "dflash_accept".into(),
        variant: "sparse_q".into(),
        arch,
        symbol: symbol(arch),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 256,
        group_segment_fixed_size: STATIC_LDS,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan()?);
    b.enable_delay_alu();
    let g = Gen::default();
    kernel(&mut Workgroup::<Gfx1201, Builder>::new(&mut b)?, &g)?;
    b.finish()
}

/// The single-symbol module.
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = vec![emit(arch)?];
    let (text, proof) = super::iu4_gemm::module(&emitted, &module(arch))?;
    Ok((emitted, text, proof))
}

fn plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(V_END as u16, 104)?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    for base in (8u8..96).step_by(8) {
        p.add_range(&format!("s_blk{base}"), Kind::S, base, 8, Live::Whole)?;
    }
    // s96..s101: the FOUND / ACT / CLM mask pairs; VCC (2 SGPRs) completes the 104 budget.
    p.add_range("s_blk96", Kind::S, 96, 4, Live::Whole)?;
    p.add_range("s_blk100", Kind::S, 100, 2, Live::Whole)?;
    for base in (0u8..V_END).step_by(8) {
        p.add_range(&format!("v_blk{base}"), Kind::V, base, 8, Live::Whole)?;
    }
    Ok(p)
}

// ---- small emission helpers -------------------------------------------------
fn m(x: u8) -> RegRef {
    sr(x, 2)
}
fn vcmp(b: &mut Builder, text: String, mk: u8, uses: &[RegRef]) -> R {
    op(b, text, &[m(mk)], uses)
}
fn mand(b: &mut Builder, d: u8, x: u8, y: u8) -> R {
    op(
        b,
        format!("s_and_b32 s{d}, s{x}, s{y}"),
        &[m(d)],
        &[m(x), m(y)],
    )
}
fn cmp_wg(wg: &mut Wg, text: String, uses: &[RegRef]) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(Instruction::new(text, vec![], uses.to_vec()))
}
fn cmp_w(w: &mut W, text: String, uses: &[RegRef]) -> Result<Uniform<Scc>, String> {
    w.scmp(Instruction::new(text, vec![], uses.to_vec()))
}
fn ds_ld(text: String, dst: RegRef, addr: u8) -> Instruction {
    Instruction::new(text, vec![dst], vec![v(addr)]).memory(MemoryClass::DsLoad)
}
fn ds_st(text: String, addr: u8, data: RegRef) -> Instruction {
    Instruction::new(text, vec![], vec![v(addr), data]).memory(MemoryClass::DsStore)
}
fn sub_i32(b: &Builder) -> &'static str {
    if b.spec.arch.gfx12() {
        "s_sub_co_i32"
    } else {
        "s_sub_i32"
    }
}
fn add_i(b: &mut Builder, d: u8, x: u8, imm: &str) -> R {
    let name = s_add_i32(b.spec.arch);
    sop(b, format!("{name} s{d}, s{x}, {imm}"), &[d], &[x])
}
fn add_r(b: &mut Builder, d: u8, x: u8, y: u8) -> R {
    let name = s_add_i32(b.spec.arch);
    sop(b, format!("{name} s{d}, s{x}, s{y}"), &[d], &[x, y])
}
fn sub_i(b: &mut Builder, d: u8, x: u8, y: u8) -> R {
    let name = sub_i32(b);
    sop(b, format!("{name} s{d}, s{x}, s{y}"), &[d], &[x, y])
}
/// `dst[0..2] = src[0..2] + x` over a 64-bit SGPR pair (`x` an SGPR).
fn add64(b: &mut Builder, dst: u8, src: u8, x: u8) -> R {
    super::common::add64(b, dst, src, x)
}
/// `dst[0..2] = src[0..2] + lo + (hi << 32)` (64-bit).
fn add64_pair(b: &mut Builder, dst: u8, src: u8, lo: u8, hi: u8) -> R {
    let a = b.spec.arch;
    sop(
        b,
        format!("{} s{dst}, s{src}, s{lo}", s_add_u32(a)),
        &[dst],
        &[src, lo],
    )?;
    sop(
        b,
        format!("{} s{}, s{}, s{hi}", s_addc_u32(a), dst + 1, src + 1),
        &[dst + 1],
        &[src + 1, hi],
    )
}

/// IEEE `dst = 1 / den` (hipcc `__fdiv_rn(1.0f, den)`): the `v_div_scale` /
/// `v_rcp_f32` / fma / `v_div_fmas` / `v_div_fixup` expansion; VCC carries the scale flag.
fn rcp_rn(b: &mut Builder, dst: u8, den: u8, t: [u8; 5]) -> R {
    let [ns, ds, rc, e, q] = t;
    op(
        b,
        format!("v_div_scale_f32 v{ns}, vcc_lo, 1.0, v{den}, 1.0"),
        &[v(ns)],
        &[v(den)],
    )?;
    op(
        b,
        format!("v_div_scale_f32 v{ds}, null, v{den}, v{den}, 1.0"),
        &[v(ds)],
        &[v(den)],
    )?;
    op(b, format!("v_rcp_f32_e32 v{rc}, v{ds}"), &[v(rc)], &[v(ds)])?;
    op(
        b,
        format!("v_fma_f32 v{e}, -v{ds}, v{rc}, 1.0"),
        &[v(e)],
        &[v(ds), v(rc)],
    )?;
    op(
        b,
        format!("v_fmac_f32_e32 v{rc}, v{e}, v{rc}"),
        &[v(rc)],
        &[v(rc), v(e)],
    )?;
    op(
        b,
        format!("v_mul_f32_e32 v{q}, v{ns}, v{rc}"),
        &[v(q)],
        &[v(ns), v(rc)],
    )?;
    op(
        b,
        format!("v_fma_f32 v{e}, -v{ds}, v{q}, v{ns}"),
        &[v(e)],
        &[v(ds), v(q), v(ns)],
    )?;
    op(
        b,
        format!("v_fmac_f32_e32 v{q}, v{e}, v{rc}"),
        &[v(q)],
        &[v(q), v(e), v(rc)],
    )?;
    op(
        b,
        format!("v_fma_f32 v{e}, -v{ds}, v{q}, v{ns}"),
        &[v(e)],
        &[v(ds), v(q), v(ns)],
    )?;
    op(b, "s_wait_alu depctr_va_vcc(0)", &[], &[])?;
    op(
        b,
        format!("v_div_fmas_f32 v{e}, v{e}, v{rc}, v{q}"),
        &[v(e)],
        &[v(e), v(rc), v(q)],
    )?;
    op(
        b,
        format!("v_div_fixup_f32 v{dst}, v{e}, v{den}, 1.0"),
        &[v(dst)],
        &[v(e), v(den)],
    )
}

// ---- kernel -----------------------------------------------------------------
fn kernel(wg: &mut Wg, g: &Gen) -> R {
    let minr = wg.lds::<Minr>("minr", MINR_BASE, MINR_BYTES)?;
    let segl = wg.lds::<Segl>("segl", SEGL_BASE, SEGL_BYTES)?;
    let base = wg.lds::<Base>("base", BASE_BASE, BASE_BYTES)?;
    let fin = wg.lds::<Fin>("fin", FIN_BASE, FIN_BYTES)?;
    let end = wg.exit(".Ldsa_end")?;
    prologue(wg)?;
    accept(wg, g)?;
    let mp = publish_min(wg, minr)?;
    resolve_row(wg, &mp)?;
    row_setup(wg, g)?;
    let sp = pass_one(wg, g, segl)?;
    let bp = chain(wg, g, &sp, base)?;
    let fp = resolve(wg, g, &bp, fin)?;
    finish(wg, g, &sp, &fp, end)
}

fn prologue(wg: &mut Wg) -> R {
    let b = wg.isa();
    smem(b, PROBS, 8, KARG, 0)?;
    smem(b, OUT, 4, KARG, 0x20)?;
    op(b, format!("v_mov_b32_e32 v{VZERO}, 0"), &[v(VZERO)], &[])?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{TID4}, 2, v{TID}"),
        &[v(TID4)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{VAB}, {}, v{TID4}", lit(BASE_BASE)),
        &[v(VAB)],
        &[v(TID4)],
    )?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{VT0}, 5, v{TID}"),
        &[v(VT0)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{WAVE}, v{VT0}"),
        &[s(WAVE)],
        &[v(VT0)],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{WAVE32}, s{WAVE}, 5"),
        &[WAVE32],
        &[WAVE],
    )?;
    sop(b, format!("s_lshl_b32 s{T0}, s{WAVE}, 2"), &[T0], &[WAVE])?;
    op(
        b,
        format!("v_mov_b32_e32 v{VWA}, s{T0}"),
        &[v(VWA)],
        &[s(T0)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{VCTL}, {}", lit(CTL_BASE)),
        &[v(VCTL)],
        &[],
    )?;
    for mk in MASKS {
        sop(b, format!("s_mov_b32 s{}, 0", mk + 1), &[mk + 1], &[])?;
    }
    // Header scalars: u_fin and max_accept at word 2n, nnz at word 4n + 2.
    sop(b, format!("s_lshl_b32 s{N4}, s{N}, 2"), &[N4], &[N])?;
    sop(b, format!("s_lshl_b32 s{T0}, s{N}, 3"), &[T0], &[N])?;
    add64(b, PTR, PAY, T0)?;
    smem(b, UFIN, 2, PTR, 0)?;
    sop(b, format!("s_lshl_b32 s{T0}, s{N}, 4"), &[T0], &[N])?;
    add_i(b, T0, T0, "8")?;
    add64(b, PTR2, PAY, T0)?;
    smem(b, NNZ, 1, PTR2, 0)?;
    sop(
        b,
        format!("s_min_u32 s{LIM}, s{N}, s{MAXACC}"),
        &[LIM],
        &[N, MAXACC],
    )?;
    // Payload bytes (4n + 3 + 2 nnz) * 4, selected-q byte offset (2n + 2) * 4.
    sop(b, format!("s_lshl_b32 s{T1}, s{NNZ}, 3"), &[T1], &[NNZ])?;
    sop(b, format!("s_lshl_b32 s{T0}, s{N}, 4"), &[T0], &[N])?;
    add_i(b, T0, T0, "12")?;
    add_r(b, PAYB, T0, T1)?;
    sop(b, format!("s_lshl_b32 s{SQB}, s{N}, 3"), &[SQB], &[N])?;
    add_i(b, SQB, SQB, "8")?;
    sop(
        b,
        format!("s_sub_co_i32 s{VM1}, s{VOCAB}, 1"),
        &[VM1],
        &[VOCAB],
    )?;
    // Descriptors: payload (exact extent), tau / z rows, whole probs, output.
    sop(
        b,
        format!("s_mov_b32 s{SRD_PAY}, s{PAY}"),
        &[SRD_PAY],
        &[PAY],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_PAY + 1, PAY + 1),
        &[SRD_PAY + 1],
        &[PAY + 1],
    )?;
    srd_tail(b, SRD_PAY, Some(PAYB))?;
    add_i(b, T0, N4, "4")?;
    sop(
        b,
        format!("s_mov_b32 s{SRD_TAU}, s{TAUP}"),
        &[SRD_TAU],
        &[TAUP],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_TAU + 1, TAUP + 1),
        &[SRD_TAU + 1],
        &[TAUP + 1],
    )?;
    srd_tail(b, SRD_TAU, Some(T0))?;
    sop(b, format!("s_mov_b32 s{SRD_Z}, s{ZP}"), &[SRD_Z], &[ZP])?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_Z + 1, ZP + 1),
        &[SRD_Z + 1],
        &[ZP + 1],
    )?;
    srd_tail(b, SRD_Z, Some(T0))?;
    add_i(b, T0, N, "1")?;
    sop(
        b,
        format!("s_mul_i32 s{T0}, s{T0}, s{VOCAB}"),
        &[T0],
        &[T0, VOCAB],
    )?;
    sop(b, format!("s_lshl_b32 s{T0}, s{T0}, 2"), &[T0], &[T0])?;
    sop(
        b,
        format!("s_mov_b32 s{SRD_PR}, s{PROBS}"),
        &[SRD_PR],
        &[PROBS],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_PR + 1, PROBS + 1),
        &[SRD_PR + 1],
        &[PROBS + 1],
    )?;
    srd_tail(b, SRD_PR, Some(T0))?;
    sop(
        b,
        format!("s_mov_b32 s{SRD_OUT}, s{OUT}"),
        &[SRD_OUT],
        &[OUT],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_OUT + 1, OUT + 1),
        &[SRD_OUT + 1],
        &[OUT + 1],
    )?;
    sop(b, format!("s_mov_b32 s{T0}, 16"), &[T0], &[])?;
    srd_tail(b, SRD_OUT, Some(T0))?;
    sop(b, format!("s_mov_b32 s{WMIN}, 0x7fffffff"), &[WMIN], &[])
}

// ---- accept tests -----------------------------------------------------------
fn accept(wg: &mut Wg, g: &Gen) -> R {
    let none = cmp_wg(wg, format!("s_cmp_eq_u32 s{LIM}, 0"), &[s(LIM)])?;
    wg.wg_skip_if(none, &g.lbl("acc_skip"), (), |wg, ()| {
        sop(wg.isa(), format!("s_mov_b32 s{TRIP}, 0"), &[TRIP], &[])?;
        wg.loop_carried(&g.lbl("acc"), (), |wg, ()| {
            accept_trip(wg.isa())?;
            let b = wg.isa();
            add_i(b, TRIP, TRIP, "0x100")?;
            let more = cmp_wg(
                wg,
                format!("s_cmp_lt_u32 s{TRIP}, s{LIM}"),
                &[s(TRIP), s(LIM)],
            )?;
            Ok(((), more))
        })
    })
}

/// One trip: rows `TRIP + tid`, the oracle's `!(u * max(qsel, FLT_MIN) <= p_t)`.
fn accept_trip(b: &mut Builder) -> R {
    op(
        b,
        format!("v_add_nc_u32_e32 v{AI}, s{TRIP}, v{TID}"),
        &[v(AI)],
        &[s(TRIP), v(TID)],
    )?;
    vcmp(
        b,
        format!("v_cmp_gt_u32_e64 s{MA}, s{LIM}, v{AI}"),
        MA,
        &[s(LIM), v(AI)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{AOFF}, 2, v{AI}"),
        &[v(AOFF)],
        &[v(AI)],
    )?;
    bload(b, 1, ATOK, AOFF, SRD_PAY, 0)?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{AT}, s{N4}, v{AOFF}"),
        &[v(AT)],
        &[s(N4), v(AOFF)],
    )?;
    bload(b, 1, AU, AT, SRD_PAY, 0)?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{AOFF2}, s{SQB}, v{AOFF}"),
        &[v(AOFF2)],
        &[s(SQB), v(AOFF)],
    )?;
    bload(b, 1, ASQ, AOFF2, SRD_PAY, 0)?;
    bload(b, 1, ATAU, AOFF, SRD_TAU, 0)?;
    bload(b, 1, AZ, AOFF, SRD_Z, 0)?;
    op(
        b,
        format!("v_mul_lo_u32 v{AT}, v{AI}, s{VOCAB}"),
        &[v(AT)],
        &[v(AI), s(VOCAB)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{AT}, v{ATOK}, v{AT}"),
        &[v(AT)],
        &[v(ATOK), v(AT)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{AT}, 2, v{AT}"),
        &[v(AT)],
        &[v(AT)],
    )?;
    bload(b, 1, AP, AT, SRD_PR, 0)?;
    // inv_z = 1 / max(z, FLT_MIN)
    op(
        b,
        format!("v_max_num_f32_e32 v{AZ}, {FLT_MIN}, v{AZ}"),
        &[v(AZ)],
        &[v(AZ)],
    )?;
    rcp_rn(b, AINV, AZ, RC)?;
    // p_t = tau <= 0 ? p : (p >= tau ? p * inv_z : 0)
    vcmp(
        b,
        format!("v_cmp_ge_f32_e64 s{MB}, v{AP}, v{ATAU}"),
        MB,
        &[v(AP), v(ATAU)],
    )?;
    op(
        b,
        format!("v_mul_f32_e32 v{AD}, v{AP}, v{AINV}"),
        &[v(AD)],
        &[v(AP), v(AINV)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{AD}, 0, v{AD}, s{MB}"),
        &[v(AD)],
        &[v(AD), m(MB)],
    )?;
    vcmp(
        b,
        format!("v_cmp_ge_f32_e64 s{MC}, 0, v{ATAU}"),
        MC,
        &[v(ATAU)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{AD}, v{AD}, v{AP}, s{MC}"),
        &[v(AD)],
        &[v(AD), v(AP), m(MC)],
    )?;
    op(
        b,
        format!("v_max_num_f32_e32 v{ASQ}, {FLT_MIN}, v{ASQ}"),
        &[v(ASQ)],
        &[v(ASQ)],
    )?;
    op(
        b,
        format!("v_mul_f32_e32 v{APR}, v{AU}, v{ASQ}"),
        &[v(APR)],
        &[v(AU), v(ASQ)],
    )?;
    vcmp(
        b,
        format!("v_cmp_nle_f32_e64 s{MD}, v{APR}, v{AD}"),
        MD,
        &[v(APR), v(AD)],
    )?;
    mand(b, MD, MD, MA)?;
    // First rejected row of this wave: trip base + 32 wave + ctz(mask).
    op(b, format!("s_ctz_i32_b32 s{T1}, s{MD}"), &[s(T1)], &[m(MD)])?;
    sop(
        b,
        format!("s_add_co_i32 s{T2}, s{TRIP}, s{WAVE32}"),
        &[T2],
        &[TRIP, WAVE32],
    )?;
    sop(
        b,
        format!("s_add_co_i32 s{T1}, s{T1}, s{T2}"),
        &[T1],
        &[T1, T2],
    )?;
    op(b, format!("s_cmp_eq_u32 s{MD}, 0"), &[], &[m(MD)])?;
    sop(
        b,
        format!("s_cselect_b32 s{T1}, 0x7fffffff, s{T1}"),
        &[T1],
        &[T1],
    )?;
    sop(
        b,
        format!("s_min_u32 s{WMIN}, s{WMIN}, s{T1}"),
        &[WMIN],
        &[WMIN, T1],
    )
}

fn publish_min(
    wg: &mut Wg,
    minr: LdsRegion<Minr, Free>,
) -> Result<LdsRegion<Minr, Published>, String> {
    op(
        wg.isa(),
        format!("v_mov_b32_e32 v{VT0}, s{WMIN}"),
        &[v(VT0)],
        &[s(WMIN)],
    )?;
    let st = wg.begin_write(minr);
    let (tw, pend) = wg.ds_store(
        st,
        ds_st(format!("ds_store_b32 v{VWA}, v{VT0}"), VWA, v(VT0)),
    )?;
    let d = wg.wait(pend)?;
    let (tp,) = wg.barrier((ready(tw, d),))?;
    Ok(tp)
}

/// `accept_len = first rejected row` (min of the eight wave words), `row`, `rejected`.
fn resolve_row(wg: &mut Wg, mp: &LdsRegion<Minr, Published>) -> R {
    for k in 0..4u32 {
        let dst = GV0 + 2 * k as u8;
        let off0 = if k == 0 {
            String::new()
        } else {
            format!(" offset0:{}", 2 * k)
        };
        wg.ds_load(
            mp,
            ds_ld(
                format!(
                    "ds_load_2addr_b32 v[{dst}:{}], v{VZERO}{off0} offset1:{}",
                    dst + 1,
                    2 * k + 1
                ),
                vr(dst, 2),
                VZERO,
            ),
        )?;
    }
    let b = wg.isa();
    op(
        b,
        format!("v_min_u32_e32 v{ACC}, v{GV0}, v{}", GV0 + 1),
        &[v(ACC)],
        &[v(GV0), v(GV0 + 1)],
    )?;
    for i in 2..8u8 {
        op(
            b,
            format!("v_min_u32_e32 v{ACC}, v{ACC}, v{}", GV0 + i),
            &[v(ACC)],
            &[v(ACC), v(GV0 + i)],
        )?;
    }
    op(
        b,
        format!("v_readfirstlane_b32 s{ROWMIN}, v{ACC}"),
        &[s(ROWMIN)],
        &[v(ACC)],
    )?;
    sop(
        b,
        format!("s_cmp_lt_u32 s{ROWMIN}, s{LIM}"),
        &[],
        &[ROWMIN, LIM],
    )?;
    sop(b, format!("s_cselect_b32 s{REJ}, 1, 0"), &[REJ], &[])?;
    sop(
        b,
        format!("s_cselect_b32 s{ROW}, s{ROWMIN}, s{LIM}"),
        &[ROW],
        &[ROWMIN, LIM],
    )
}

// ---- row scalars, geometry, q cursor ----------------------------------------
fn row_setup(wg: &mut Wg, g: &Gen) -> R {
    let b = wg.isa();
    // tau, z of the row.
    sop(b, format!("s_lshl_b32 s{T0}, s{ROW}, 2"), &[T0], &[ROW])?;
    add64(b, PTR, TAUP, T0)?;
    smem(b, TAU, 1, PTR, 0)?;
    add64(b, PTR2, ZP, T0)?;
    smem(b, ZB, 1, PTR2, 0)?;
    // q row offsets (row 0 when nothing was rejected: always in bounds).
    sop(b, format!("s_cmp_eq_u32 s{REJ}, 0"), &[], &[REJ])?;
    sop(b, format!("s_cselect_b32 s{T1}, 0, s{ROW}"), &[T1], &[ROW])?;
    sop(b, format!("s_lshl_b32 s{T1}, s{T1}, 2"), &[T1], &[T1])?;
    sop(b, format!("s_mul_i32 s{T2}, s{N4}, 3"), &[T2], &[N4])?;
    add_i(b, T2, T2, "8")?;
    add_r(b, T1, T1, T2)?;
    add64(b, PTR, PAY, T1)?;
    smem(b, QLO, 1, PTR, 0)?;
    smem(b, T3, 1, PTR, 4)?;
    sub_i(b, QN, T3, QLO)?;
    sop(b, format!("s_cmp_eq_u32 s{REJ}, 0"), &[], &[REJ])?;
    sop(b, format!("s_cselect_b32 s{QN}, 0, s{QN}"), &[QN], &[QN])?;
    sop(
        b,
        format!("s_cselect_b32 s{QLO}, 0, s{QLO}"),
        &[QLO],
        &[QLO],
    )?;
    // Byte offsets of the row's q ids and values inside the payload.
    sop(b, format!("s_lshl_b32 s{T0}, s{N}, 4"), &[T0], &[N])?;
    add_i(b, T0, T0, "12")?;
    sop(b, format!("s_lshl_b32 s{T1}, s{QLO}, 2"), &[T1], &[QLO])?;
    add_r(b, QIB, T0, T1)?;
    sop(b, format!("s_lshl_b32 s{T1}, s{NNZ}, 2"), &[T1], &[NNZ])?;
    add_r(b, QVB, QIB, T1)?;
    // inv_z = 1 / max(z, FLT_MIN), uniform.
    op(
        b,
        format!("v_mov_b32_e32 v{VT0}, s{ZB}"),
        &[v(VT0)],
        &[s(ZB)],
    )?;
    op(
        b,
        format!("v_max_num_f32_e32 v{VT0}, {FLT_MIN}, v{VT0}"),
        &[v(VT0)],
        &[v(VT0)],
    )?;
    rcp_rn(b, VT1, VT0, RC)?;
    op(
        b,
        format!("v_readfirstlane_b32 s{INVZ}, v{VT1}"),
        &[s(INVZ)],
        &[v(VT1)],
    )?;
    // tau <= 0 lanes (uniform): the row is untruncated.
    vcmp(
        b,
        format!("v_cmp_ge_f32_e64 s{MTAU}, 0, s{TAU}"),
        MTAU,
        &[s(TAU)],
    )?;
    // Segment geometry.
    add_i(b, T0, VOCAB, "0xff")?;
    sop(b, format!("s_lshr_b32 s{SEG}, s{T0}, 8"), &[SEG], &[T0])?;
    add_i(b, T0, SEG, "15")?;
    sop(
        b,
        format!("s_lshr_b32 s{NCHUNK}, s{T0}, 4"),
        &[NCHUNK],
        &[T0],
    )?;
    // Row descriptor: probs + row * vocab * 4, records vocab * 4.
    sop(b, format!("s_lshl_b32 s{T0}, s{VOCAB}, 2"), &[T0], &[VOCAB])?;
    sop(
        b,
        format!("s_mul_i32 s{T1}, s{ROW}, s{T0}"),
        &[T1],
        &[ROW, T0],
    )?;
    sop(
        b,
        format!("s_mul_hi_u32 s{T2}, s{ROW}, s{T0}"),
        &[T2],
        &[ROW, T0],
    )?;
    add64_pair(b, SRD_ROW, PROBS, T1, T2)?;
    srd_tail(b, SRD_ROW, Some(T0))?;
    // Lane segment [start, end).
    op(
        b,
        format!("v_mul_u32_u24_e32 v{START}, s{SEG}, v{TID}"),
        &[v(START)],
        &[s(SEG), v(TID)],
    )?;
    op(
        b,
        format!("v_min_u32_e32 v{START}, s{VOCAB}, v{START}"),
        &[v(START)],
        &[s(VOCAB), v(START)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{END}, s{SEG}, v{START}"),
        &[v(END)],
        &[s(SEG), v(START)],
    )?;
    op(
        b,
        format!("v_min_u32_e32 v{END}, s{VOCAB}, v{END}"),
        &[v(END)],
        &[s(VOCAB), v(END)],
    )?;
    // q cursor: first q id >= start, and the id there (-1: exhausted).
    op(b, format!("v_mov_b32_e32 v{QP0}, 0"), &[v(QP0)], &[])?;
    op(b, format!("v_mov_b32_e32 v{NEXTQ0}, -1"), &[v(NEXTQ0)], &[])?;
    let none = cmp_wg(wg, format!("s_cmp_eq_u32 s{QN}, 0"), &[s(QN)])?;
    wg.wg_skip_if(none, &g.lbl("qs_skip"), (), |wg, ()| {
        let b = wg.isa();
        op(b, format!("v_mov_b32_e32 v{LO}, 0"), &[v(LO)], &[])?;
        op(b, format!("v_mov_b32_e32 v{HI}, s{QN}"), &[v(HI)], &[s(QN)])?;
        // Iterations = bit length of qn (every lane's range is empty after that many halvings):
        // the f32 exponent of qn, plus one (extra iterations are harmless).
        op(
            b,
            format!("v_cvt_f32_u32_e32 v{VT0}, s{QN}"),
            &[v(VT0)],
            &[s(QN)],
        )?;
        op(
            b,
            format!("v_lshrrev_b32_e32 v{VT0}, 23, v{VT0}"),
            &[v(VT0)],
            &[v(VT0)],
        )?;
        op(
            b,
            format!("v_readfirstlane_b32 s{T5}, v{VT0}"),
            &[s(T5)],
            &[v(VT0)],
        )?;
        sop(b, format!("s_sub_co_i32 s{T5}, s{T5}, 0x7e"), &[T5], &[T5])?;
        sop(b, format!("s_mov_b32 s{T4}, 0"), &[T4], &[])?;
        wg.loop_carried(&g.lbl("qs"), (), |wg, ()| {
            let b = wg.isa();
            vcmp(
                b,
                format!("v_cmp_lt_u32_e64 s{MA}, v{LO}, v{HI}"),
                MA,
                &[v(LO), v(HI)],
            )?;
            op(
                b,
                format!("v_add_nc_u32_e32 v{VT1}, v{LO}, v{HI}"),
                &[v(VT1)],
                &[v(LO), v(HI)],
            )?;
            op(
                b,
                format!("v_lshrrev_b32_e32 v{VT1}, 1, v{VT1}"),
                &[v(VT1)],
                &[v(VT1)],
            )?;
            op(
                b,
                format!("v_lshl_add_u32 v{VT2}, v{VT1}, 2, s{QIB}"),
                &[v(VT2)],
                &[v(VT1), s(QIB)],
            )?;
            bload(b, 1, VT3, VT2, SRD_PAY, 0)?;
            vcmp(
                b,
                format!("v_cmp_lt_u32_e64 s{MB}, v{VT3}, v{START}"),
                MB,
                &[v(VT3), v(START)],
            )?;
            mand(b, MC, MA, MB)?;
            op(
                b,
                format!("s_xor_b32 s{MD}, s{MA}, s{MC}"),
                &[m(MD)],
                &[m(MA), m(MC)],
            )?;
            op(
                b,
                format!("v_add_nc_u32_e32 v{VT4}, 1, v{VT1}"),
                &[v(VT4)],
                &[v(VT1)],
            )?;
            op(
                b,
                format!("v_cndmask_b32_e64 v{LO}, v{LO}, v{VT4}, s{MC}"),
                &[v(LO)],
                &[v(LO), v(VT4), m(MC)],
            )?;
            op(
                b,
                format!("v_cndmask_b32_e64 v{HI}, v{HI}, v{VT1}, s{MD}"),
                &[v(HI)],
                &[v(HI), v(VT1), m(MD)],
            )?;
            add_i(b, T4, T4, "1")?;
            let more = cmp_wg(wg, format!("s_cmp_lt_u32 s{T4}, s{T5}"), &[s(T4), s(T5)])?;
            Ok(((), more))
        })?;
        let b = wg.isa();
        op(
            b,
            format!("v_mov_b32_e32 v{QP0}, v{LO}"),
            &[v(QP0)],
            &[v(LO)],
        )?;
        op(
            b,
            format!("v_lshl_add_u32 v{VT2}, v{LO}, 2, s{QIB}"),
            &[v(VT2)],
            &[v(LO), s(QIB)],
        )?;
        bload(b, 1, VT3, VT2, SRD_PAY, 0)?;
        vcmp(
            b,
            format!("v_cmp_gt_u32_e64 s{MA}, s{QN}, v{LO}"),
            MA,
            &[s(QN), v(LO)],
        )?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{NEXTQ0}, -1, v{VT3}, s{MA}"),
            &[v(NEXTQ0)],
            &[v(VT3), m(MA)],
        )
    })
}

/// Restart a lane's segment walk: chunk cursor, accumulators and q cursor.
fn reset_chunk(b: &mut Builder) -> R {
    op(
        b,
        format!("v_lshlrev_b32_e32 v{OFF}, 2, v{START}"),
        &[v(OFF)],
        &[v(START)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{VCB}, v{START}"),
        &[v(VCB)],
        &[v(START)],
    )?;
    op(
        b,
        format!("v_sub_nc_u32_e32 v{REM}, v{END}, v{START}"),
        &[v(REM)],
        &[v(END), v(START)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{QP}, v{QP0}"),
        &[v(QP)],
        &[v(QP0)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{NEXTQ}, v{NEXTQ0}"),
        &[v(NEXTQ)],
        &[v(NEXTQ0)],
    )?;
    op(b, format!("v_mov_b32_e32 v{VL}, 0"), &[v(VL)], &[])?;
    op(b, format!("v_mov_b32_e32 v{LAST}, -1"), &[v(LAST)], &[])?;
    sop(b, format!("s_mov_b32 s{KCNT}, 0"), &[KCNT], &[])
}

// ---- segment walk ------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Sum,
    Walk,
    Argmax,
}

/// `d = dsa_eff(p)` of the row: `tau <= 0 ? p : (p >= tau ? p * inv_z : 0)`.
fn eff(b: &mut Builder, d: u8, p: u8, mk: u8) -> R {
    vcmp(
        b,
        format!("v_cmp_ge_f32_e64 s{mk}, v{p}, s{TAU}"),
        mk,
        &[v(p), s(TAU)],
    )?;
    op(
        b,
        format!("v_mul_f32_e32 v{VT0}, s{INVZ}, v{p}"),
        &[v(VT0)],
        &[s(INVZ), v(p)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{d}, 0, v{VT0}, s{mk}"),
        &[v(d)],
        &[v(VT0), m(mk)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{d}, v{d}, v{p}, s{MTAU}"),
        &[v(d)],
        &[v(d), v(p), m(MTAU)],
    )
}

/// One chunk of `CHUNK` ids per lane: the loads are issued together, the
/// elements are consumed strictly in ascending id order.
fn chunk(w: &mut W, g: &Gen, mode: Mode) -> R {
    let b = w.isa();
    for j in 0..CHUNK {
        bload(b, 1, P + j as u8, OFF, SRD_ROW, 4 * j)?;
    }
    for j in 0..CHUNK as u8 {
        element(w, g, mode, j)?;
    }
    let b = w.isa();
    op(
        b,
        format!("v_add_nc_u32_e32 v{OFF}, 64, v{OFF}"),
        &[v(OFF)],
        &[v(OFF)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{VCB}, 16, v{VCB}"),
        &[v(VCB)],
        &[v(VCB)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{REM}, -16, v{REM}"),
        &[v(REM)],
        &[v(REM)],
    )
}

fn element(w: &mut W, g: &Gen, mode: Mode, j: u8) -> R {
    let p = P + j;
    let b = w.isa();
    op(
        b,
        format!("v_add_nc_u32_e32 v{VCJ}, {j}, v{VCB}"),
        &[v(VCJ)],
        &[v(VCB)],
    )?;
    eff(b, D, p, MB)?;
    if mode == Mode::Argmax {
        vcmp(
            b,
            format!("v_cmp_lt_i32_e64 s{MC}, {j}, v{REM}"),
            MC,
            &[v(REM)],
        )?;
        vcmp(
            b,
            format!("v_cmp_gt_f32_e64 s{MA}, v{D}, v{BV}"),
            MA,
            &[v(D), v(BV)],
        )?;
        mand(b, MA, MA, MC)?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{BV}, v{BV}, v{D}, s{MA}"),
            &[v(BV)],
            &[v(BV), v(D), m(MA)],
        )?;
        return op(
            b,
            format!("v_cndmask_b32_e64 v{BI}, v{BI}, v{VCJ}, s{MA}"),
            &[v(BI)],
            &[v(BI), v(VCJ), m(MA)],
        );
    }
    // Residual: a lane whose next q id is this id subtracts it (rare, wave-uniform slow path).
    vcmp(
        b,
        format!("v_cmp_eq_u32_e64 s{MQ}, v{NEXTQ}, v{VCJ}"),
        MQ,
        &[v(NEXTQ), v(VCJ)],
    )?;
    let none = cmp_w(w, format!("s_cmp_eq_u32 s{MQ}, 0"), &[m(MQ)])?;
    w.skip_if(none, &g.lbl("mg"), (), |w, ()| merge_slow(w.isa()))?;
    let b = w.isa();
    vcmp(b, format!("v_cmp_lt_f32_e64 s{MA}, 0, v{D}"), MA, &[v(D)])?;
    vcmp(
        b,
        format!("v_cmp_lt_i32_e64 s{MC}, {j}, v{REM}"),
        MC,
        &[v(REM)],
    )?;
    mand(b, MA, MA, MC)?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{VT1}, 0, v{D}, s{MA}"),
        &[v(VT1)],
        &[v(D), m(MA)],
    )?;
    op(
        b,
        format!("v_add_f32_e32 v{VL}, v{VL}, v{VT1}"),
        &[v(VL)],
        &[v(VL), v(VT1)],
    )?;
    if mode == Mode::Sum {
        return op(
            b,
            format!("v_cndmask_b32_e64 v{LAST}, v{LAST}, v{VCJ}, s{MA}"),
            &[v(LAST)],
            &[v(LAST), v(VCJ), m(MA)],
        );
    }
    // Walk: the first id whose running CDF `base + L` exceeds `us`.
    op(
        b,
        format!("v_add_f32_e32 v{VT2}, v{VB0}, v{VL}"),
        &[v(VT2)],
        &[v(VB0), v(VL)],
    )?;
    vcmp(
        b,
        format!("v_cmp_lt_f32_e64 s{MD}, s{US}, v{VT2}"),
        MD,
        &[s(US), v(VT2)],
    )?;
    mand(b, MD, MD, MA)?;
    op(
        b,
        format!("s_not_b32 s{T0}, s{FOUND}"),
        &[s(T0)],
        &[m(FOUND)],
    )?;
    op(
        b,
        format!("s_and_b32 s{MD}, s{MD}, s{T0}"),
        &[m(MD)],
        &[m(MD), s(T0)],
    )?;
    op(
        b,
        format!("s_or_b32 s{FOUND}, s{FOUND}, s{MD}"),
        &[m(FOUND)],
        &[m(FOUND), m(MD)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{BONUS}, v{BONUS}, v{VCJ}, s{MD}"),
        &[v(BONUS)],
        &[v(BONUS), v(VCJ), m(MD)],
    )
}

/// The matched lanes take `d - q`, advance their q cursor and reload the next id.
fn merge_slow(b: &mut Builder) -> R {
    op(
        b,
        format!("v_lshl_add_u32 v{VT1}, v{QP}, 2, s{QVB}"),
        &[v(VT1)],
        &[v(QP), s(QVB)],
    )?;
    bload(b, 1, VT2, VT1, SRD_PAY, 0)?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{VT3}, 1, v{QP}"),
        &[v(VT3)],
        &[v(QP)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{QP}, v{QP}, v{VT3}, s{MQ}"),
        &[v(QP)],
        &[v(QP), v(VT3), m(MQ)],
    )?;
    op(
        b,
        format!("v_lshl_add_u32 v{VT1}, v{QP}, 2, s{QIB}"),
        &[v(VT1)],
        &[v(QP), s(QIB)],
    )?;
    bload(b, 1, VT4, VT1, SRD_PAY, 0)?;
    op(
        b,
        format!("v_sub_f32_e32 v{VT3}, v{D}, v{VT2}"),
        &[v(VT3)],
        &[v(D), v(VT2)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{D}, v{D}, v{VT3}, s{MQ}"),
        &[v(D)],
        &[v(D), v(VT3), m(MQ)],
    )?;
    vcmp(
        b,
        format!("v_cmp_gt_u32_e64 s{MA}, s{QN}, v{QP}"),
        MA,
        &[s(QN), v(QP)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{NEXTQ}, -1, v{VT4}, s{MA}"),
        &[v(NEXTQ)],
        &[v(VT4), m(MA)],
    )
}

/// A workgroup loop over all chunks of the segment walk.
fn chunk_loop(wg: &mut Wg, g: &Gen, head: &str, mode: Mode) -> R {
    wg.loop_carried(&g.lbl(head), (), |wg, ()| {
        chunk(wg, g, mode)?;
        let b = wg.isa();
        add_i(b, KCNT, KCNT, "1")?;
        let more = cmp_wg(
            wg,
            format!("s_cmp_lt_u32 s{KCNT}, s{NCHUNK}"),
            &[s(KCNT), s(NCHUNK)],
        )?;
        Ok(((), more))
    })
}

/// Pass 1: per-lane segment sum of the positive summands and last positive id.
fn pass_one(
    wg: &mut Wg,
    g: &Gen,
    segl: LdsRegion<Segl, Free>,
) -> Result<LdsRegion<Segl, Published>, String> {
    reset_chunk(wg.isa())?;
    chunk_loop(wg, g, "p1", Mode::Sum)?;
    let st = wg.begin_write(segl);
    let st = wg.ds_store(
        st,
        ds_st(
            format!("ds_store_b32 v{TID4}, v{VL} offset:{SEG_BASE}"),
            TID4,
            v(VL),
        ),
    )?;
    let (sw, pend) = wg.ds_store(
        st,
        ds_st(
            format!("ds_store_b32 v{TID4}, v{LAST} offset:{LAST_BASE}"),
            TID4,
            v(LAST),
        ),
    )?;
    let d = wg.wait(pend)?;
    let (sp,) = wg.barrier((ready(sw, d),))?;
    Ok(sp)
}

/// Broadcast `count` (a multiple of 4) consecutive words at `base_off` into
/// `dst..dst+count`: every lane reads the same words.
fn bcast_load<X>(w: &mut W, r: &LdsRegion<X, Published>, dst: u8, base_off: u32, count: u8) -> R {
    for q in 0..count / 4 {
        let d = dst + 4 * q;
        w.ds_load(
            r,
            ds_ld(
                format!(
                    "ds_load_b128 v[{d}:{}], v{VZERO} offset:{}",
                    d + 3,
                    base_off + 16 * u32::from(q)
                ),
                vr(d, 4),
                VZERO,
            ),
        )?;
    }
    Ok(())
}

// ---- chain, decision ---------------------------------------------------------
/// Wave 0 chains the 256 segment sums in segment order (the oracle's
/// `base[t + 1] = base[t] + seg[t]`), then derives `us` and the code.
fn chain(
    wg: &mut Wg,
    g: &Gen,
    sp: &LdsRegion<Segl, Published>,
    base: LdsRegion<Base, Free>,
) -> Result<LdsRegion<Base, Published>, String> {
    let st = wg.begin_write(base);
    let not_w0 = cmp_w(wg, format!("s_cmp_lg_u32 s{WAVE}, 0"), &[s(WAVE)])?;
    let (bw, pend) = wg.skip_if(not_w0, &g.lbl("chain"), st, |w, mut st| {
        op(w.isa(), format!("v_mov_b32_e32 v{ACC}, 0"), &[v(ACC)], &[])?;
        for grp in 0..16u32 {
            bcast_load(w, sp, GV0, SEG_BASE + 64 * grp, 16)?;
            for j in 0..16u32 {
                let t = 16 * grp + j;
                st = w.ds_store(
                    st,
                    ds_st(
                        format!("ds_store_b32 v{VZERO}, v{ACC} offset:{}", BASE_BASE + 4 * t),
                        VZERO,
                        v(ACC),
                    ),
                )?;
                let a = GV0 + j as u8;
                op(
                    w.isa(),
                    format!("v_add_f32_e32 v{ACC}, v{ACC}, v{a}"),
                    &[v(ACC)],
                    &[v(ACC), v(a)],
                )?;
            }
        }
        st = w.ds_store(
            st,
            ds_st(
                format!("ds_store_b32 v{VZERO}, v{ACC} offset:{}", BASE_BASE + 1024),
                VZERO,
                v(ACC),
            ),
        )?;
        let b = w.isa();
        // Vanishing residual: total <= 0 (NaN compares false) on a rejected window.
        vcmp(
            b,
            format!("v_cmp_ge_f32_e64 s{MA}, 0, v{ACC}"),
            MA,
            &[v(ACC)],
        )?;
        op(b, format!("s_cmp_lg_u32 s{MA}, 0"), &[], &[m(MA)])?;
        sop(b, format!("s_cselect_b32 s{T0}, 1, 0"), &[T0], &[])?;
        sop(
            b,
            format!("s_and_b32 s{CODE}, s{T0}, s{REJ}"),
            &[CODE],
            &[T0, REJ],
        )?;
        // us = rejected ? (code == 1 ? 0 : u_fin * total) : u_fin
        op(
            b,
            format!("v_mul_f32_e32 v{VT0}, s{UFIN}, v{ACC}"),
            &[v(VT0)],
            &[s(UFIN), v(ACC)],
        )?;
        op(
            b,
            format!("v_readfirstlane_b32 s{T1}, v{VT0}"),
            &[s(T1)],
            &[v(VT0)],
        )?;
        sop(b, format!("s_cmp_eq_u32 s{REJ}, 0"), &[], &[REJ])?;
        sop(
            b,
            format!("s_cselect_b32 s{US}, s{UFIN}, s{T1}"),
            &[US],
            &[UFIN, T1],
        )?;
        sop(b, format!("s_cmp_eq_u32 s{CODE}, 1"), &[], &[CODE])?;
        sop(b, format!("s_cselect_b32 s{US}, 0, s{US}"), &[US], &[US])?;
        op(
            b,
            format!("v_mov_b32_e32 v{VT0}, s{US}"),
            &[v(VT0)],
            &[s(US)],
        )?;
        op(
            b,
            format!("v_mov_b32_e32 v{VT1}, s{CODE}"),
            &[v(VT1)],
            &[s(CODE)],
        )?;
        st = w.ds_store(
            st,
            ds_st(
                format!("ds_store_b32 v{VZERO}, v{VT0} offset:{CTL_BASE}"),
                VZERO,
                v(VT0),
            ),
        )?;
        st = w.ds_store(
            st,
            ds_st(
                format!("ds_store_b32 v{VZERO}, v{VT1} offset:{}", CTL_BASE + 4),
                VZERO,
                v(VT1),
            ),
        )?;
        Ok(st)
    })?;
    let d = wg.wait(pend)?;
    let (bp,) = wg.barrier((ready(bw, d),))?;
    Ok(bp)
}

/// Every lane's claim of the CDF point (or its segment argmax on code 1).
fn resolve(
    wg: &mut Wg,
    g: &Gen,
    bp: &LdsRegion<Base, Published>,
    fin: LdsRegion<Fin, Free>,
) -> Result<LdsRegion<Fin, Published>, String> {
    let b = wg.isa();
    wg.ds_load(
        bp,
        ds_ld(
            format!("ds_load_2addr_b32 v[{VB0}:{VB1}], v{VAB} offset1:1"),
            vr(VB0, 2),
            VAB,
        ),
    )?;
    wg.ds_load(
        bp,
        ds_ld(
            format!("ds_load_2addr_b32 v[{VUS}:{VCODE}], v{VCTL} offset1:1"),
            vr(VUS, 2),
            VCTL,
        ),
    )?;
    let b = wg.isa();
    op(
        b,
        format!("v_readfirstlane_b32 s{US}, v{VUS}"),
        &[s(US)],
        &[v(VUS)],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{CODE}, v{VCODE}"),
        &[s(CODE)],
        &[v(VCODE)],
    )?;
    // claim: base[t] <= us < base[t + 1], never on the degenerate path.
    vcmp(
        b,
        format!("v_cmp_le_f32_e64 s{MA}, v{VB0}, s{US}"),
        MA,
        &[v(VB0), s(US)],
    )?;
    vcmp(
        b,
        format!("v_cmp_lt_f32_e64 s{MB}, s{US}, v{VB1}"),
        MB,
        &[s(US), v(VB1)],
    )?;
    mand(b, CLM, MA, MB)?;
    sop(b, format!("s_cmp_eq_u32 s{CODE}, 1"), &[], &[CODE])?;
    op(
        b,
        format!("s_cselect_b32 s{CLM}, 0, s{CLM}"),
        &[m(CLM)],
        &[m(CLM)],
    )?;
    op(b, format!("v_mov_b32_e32 v{BONUS}, -1"), &[v(BONUS)], &[])?;
    op(b, format!("v_mov_b32_e32 v{BV}, {NEG_INF}"), &[v(BV)], &[])?;
    op(b, format!("v_mov_b32_e32 v{BI}, -1"), &[v(BI)], &[])?;
    // Degenerate residual: every segment's first-maximum argmax of the truncated row.
    let not_degenerate = cmp_wg(wg, format!("s_cmp_lg_u32 s{CODE}, 1"), &[s(CODE)])?;
    wg.wg_skip_if(not_degenerate, &g.lbl("arg"), (), |wg, ()| {
        reset_chunk(wg.isa())?;
        chunk_loop(wg, g, "arg", Mode::Argmax)
    })?;
    // Claiming segment: re-walk until the CDF passes `us` (exec = the claiming lane).
    let unclaimed = cmp_w(wg, format!("s_cmp_eq_u32 s{CLM}, 0"), &[m(CLM)])?;
    wg.skip_if(unclaimed, &g.lbl("walk_skip"), (), |w, ()| {
        reset_chunk(w.isa())?;
        let b = w.isa();
        op(b, format!("s_mov_b32 s{FOUND}, 0"), &[m(FOUND)], &[])?;
        op(b, format!("s_mov_b32 exec_lo, s{CLM}"), &[], &[m(CLM)])?;
        let (head, exit) = (g.lbl("walk"), g.lbl("walk_exit"));
        w.loop_until(&head, &exit, |w, x| walk_body(w, x, g))?;
        op(w.isa(), "s_mov_b32 exec_lo, -1", &[], &[])
    })?;
    let st = wg.begin_write(fin);
    let st = wg.ds_store(
        st,
        ds_st(
            format!("ds_store_b32 v{TID4}, v{BONUS} offset:{WIN_BASE}"),
            TID4,
            v(BONUS),
        ),
    )?;
    let st = wg.ds_store(
        st,
        ds_st(
            format!("ds_store_b32 v{TID4}, v{BV} offset:{BV_BASE}"),
            TID4,
            v(BV),
        ),
    )?;
    let (fw, pend) = wg.ds_store(
        st,
        ds_st(
            format!("ds_store_b32 v{TID4}, v{BI} offset:{BI_BASE}"),
            TID4,
            v(BI),
        ),
    )?;
    let d = wg.wait(pend)?;
    let (fp,) = wg.barrier((ready(fw, d),))?;
    Ok(fp)
}

fn walk_body(w: &mut W, x: &LoopExit<Builder>, g: &Gen) -> R {
    chunk(w, g, Mode::Walk)?;
    let b = w.isa();
    // Lanes that found their id leave; the walk ends when none is left or the segment is done.
    op(
        b,
        format!("s_xor_b32 s{ACT}, s{CLM}, s{FOUND}"),
        &[m(ACT)],
        &[m(CLM), m(FOUND)],
    )?;
    op(b, format!("s_mov_b32 exec_lo, s{ACT}"), &[], &[m(ACT)])?;
    add_i(b, KCNT, KCNT, "1")?;
    sop(
        b,
        format!("s_cmp_ge_u32 s{KCNT}, s{NCHUNK}"),
        &[],
        &[KCNT, NCHUNK],
    )?;
    sop(b, format!("s_cselect_b32 s{T0}, 1, 0"), &[T0], &[])?;
    op(b, format!("s_cmp_eq_u32 s{ACT}, 0"), &[], &[m(ACT)])?;
    sop(b, format!("s_cselect_b32 s{T1}, 1, 0"), &[T1], &[])?;
    sop(b, format!("s_or_b32 s{T0}, s{T0}, s{T1}"), &[T0], &[T0, T1])?;
    let done = cmp_w(w, format!("s_cmp_lg_u32 s{T0}, 0"), &[s(T0)])?;
    w.break_if(done, x)
}

// ---- decision words -----------------------------------------------------------
fn finish(
    wg: &mut Wg,
    g: &Gen,
    sp: &LdsRegion<Segl, Published>,
    fp: &LdsRegion<Fin, Published>,
    end: End,
) -> R {
    {
        let w: &mut W = &mut **wg;
        let is_w0 = cmp_w(w, format!("s_cmp_eq_u32 s{WAVE}, 0"), &[s(WAVE)])?;
        w.exit_unless(is_w0, &end)?;
    }
    // Exact integer maxima: the claimed id, and the last support id (fall-off fallback).
    let b = wg.isa();
    op(b, format!("v_mov_b32_e32 v{FW}, -1"), &[v(FW)], &[])?;
    op(b, format!("v_mov_b32_e32 v{ACC}, -1"), &[v(ACC)], &[])?;
    for grp in 0..16u32 {
        bcast_load(wg, fp, GV0, WIN_BASE + 64 * grp, 16)?;
        bcast_load(wg, sp, GV1, LAST_BASE + 64 * grp, 16)?;
        let b = wg.isa();
        for j in 0..16u8 {
            op(
                b,
                format!("v_max_i32_e32 v{FW}, v{FW}, v{}", GV0 + j),
                &[v(FW)],
                &[v(FW), v(GV0 + j)],
            )?;
            op(
                b,
                format!("v_max_i32_e32 v{ACC}, v{ACC}, v{}", GV1 + j),
                &[v(ACC)],
                &[v(ACC), v(GV1 + j)],
            )?;
        }
    }
    let b = wg.isa();
    op(
        b,
        format!("v_mov_b32_e32 v{ABV}, {NEG_INF}"),
        &[v(ABV)],
        &[],
    )?;
    op(b, format!("v_mov_b32_e32 v{ABI}, 0"), &[v(ABI)], &[])?;
    // Degenerate: the oracle's sequential strict-greater argmax over the segments.
    let not_degenerate = cmp_w(wg, format!("s_cmp_lg_u32 s{CODE}, 1"), &[s(CODE)])?;
    wg.skip_if(not_degenerate, &g.lbl("fin_arg"), (), |w, ()| {
        for grp in 0..16u32 {
            bcast_load(w, fp, GV0, BV_BASE + 64 * grp, 16)?;
            bcast_load(w, fp, GV1, BI_BASE + 64 * grp, 16)?;
            let b = w.isa();
            for j in 0..16u8 {
                let (bv, bi) = (GV0 + j, GV1 + j);
                vcmp(
                    b,
                    format!("v_cmp_gt_f32_e64 s{MA}, v{bv}, v{ABV}"),
                    MA,
                    &[v(bv), v(ABV)],
                )?;
                vcmp(b, format!("v_cmp_le_i32_e64 s{MB}, 0, v{bi}"), MB, &[v(bi)])?;
                mand(b, MA, MA, MB)?;
                op(
                    b,
                    format!("v_cndmask_b32_e64 v{ABV}, v{ABV}, v{bv}, s{MA}"),
                    &[v(ABV)],
                    &[v(ABV), v(bv), m(MA)],
                )?;
                op(
                    b,
                    format!("v_cndmask_b32_e64 v{ABI}, v{ABI}, v{bi}, s{MA}"),
                    &[v(ABI)],
                    &[v(ABI), v(bi), m(MA)],
                )?;
            }
        }
        Ok(())
    })?;
    let b = wg.isa();
    op(
        b,
        format!("v_readfirstlane_b32 s{T0}, v{FW}"),
        &[s(T0)],
        &[v(FW)],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{T1}, v{ACC}"),
        &[s(T1)],
        &[v(ACC)],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{T2}, v{ABI}"),
        &[s(T2)],
        &[v(ABI)],
    )?;
    // Normal draw: claimed id, else the last support id (code 2), else vocab - 1 (code 2).
    sop(b, format!("s_cmp_ge_i32 s{T1}, 0"), &[], &[T1])?;
    sop(
        b,
        format!("s_cselect_b32 s{T4}, s{T1}, s{VM1}"),
        &[T4],
        &[T1, VM1],
    )?;
    sop(b, format!("s_cmp_ge_i32 s{T0}, 0"), &[], &[T0])?;
    sop(
        b,
        format!("s_cselect_b32 s{T3}, s{T0}, s{T4}"),
        &[T3],
        &[T0, T4],
    )?;
    sop(b, format!("s_cselect_b32 s{T5}, 0, 2"), &[T5], &[])?;
    sop(b, format!("s_cmp_eq_u32 s{CODE}, 1"), &[], &[CODE])?;
    sop(
        b,
        format!("s_cselect_b32 s{T3}, s{T2}, s{T3}"),
        &[T3],
        &[T2, T3],
    )?;
    sop(b, format!("s_cselect_b32 s{T5}, 1, s{T5}"), &[T5], &[T5])?;
    op(
        b,
        format!("v_mov_b32_e32 v{OUTV}, s{ROW}"),
        &[v(OUTV)],
        &[s(ROW)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{}, s{T3}", OUTV + 1),
        &[v(OUTV + 1)],
        &[s(T3)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{}, s{REJ}", OUTV + 2),
        &[v(OUTV + 2)],
        &[s(REJ)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{}, s{T5}", OUTV + 3),
        &[v(OUTV + 3)],
        &[s(T5)],
    )?;
    op(b, "s_mov_b32 exec_lo, 1", &[], &[])?;
    super::common::bstore_b128(b, OUTV, VZERO, SRD_OUT, 0)?;
    b.wait_all()?;
    wg.end(end)
}
