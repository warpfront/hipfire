// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted (PeaceMaker) canonical parallel target-row sampler for
//! gfx1151 and gfx1201: `spec_cpu_ar_rows_parallel_f32_pm_<arch>`, the byte
//! twin of the hipcc kernel `spec_cpu_ar_rows_parallel_f32`
//! (`kernels/src/spec_sampling_cpu_ar_parallel.hip`).
//!
//! ABI (ten arguments, 56 bytes): logits `*f32` @0, counts `*u32` @8, ids
//! `*u32` @16, probabilities `*f32` @24, vocab i32 @32, pool i32 @36, cap i32
//! @40, temperature f32 @44, top_p f32 @48, min_p f32 @52. Grid
//! `[rows, 1, 1]`, block 256, 64 bytes of static LDS, no scratch, no atomics.
//! Row `r` writes `counts[r]`, `ids[r * 64 + 0..64]`, `probs[r * 64 + 0..64]`
//! in descending-probability order with zero tails.
//!
//! Law (identical to the HIP twin): canonical top-`pool` under (logit desc,
//! id asc) over the finite logits (+0 and -0 tie on the lower id); the row
//! maximum is the canonical best record; `w = expf(RN(RN(l - max) * RN(1 /
//! temperature)))` through the builder's hipcc `expf` DAG with non-finite
//! results flushed to 0; the aggregate mass is the fixed 64-leaf tree
//! (strides 32, 16, 8, 4, 2, 1, left = lower index); stable descending sort
//! by weight; `min_p` floor `RN(min_p * w0)`; `top_p` threshold `RN(top_p *
//! mass)` over the serial cumulative of the (cut) pool, crossing token
//! included, residual correction when nothing crosses; `p > 0` retained in
//! order; argmax point mass when the full-pool mass is `<= 0` or non-finite.
//! Every add, subtract and multiply is its own non-contracted instruction and
//! every division is the IEEE `v_div_scale` / `v_div_fmas` / `v_div_fixup`
//! sequence.
//!
//! Algorithm (the output depends only on the canonical top-`pool` records, so
//! the structure below is free to differ from the HIP tile merge):
//! 1. Scan. Lane `t` of the workgroup walks tokens `t + 256 j` in ascending
//!    order and keeps its canonical top-64 (logit desc, id asc) in 128
//!    registers. A record is inserted by a branch-free carry chain that
//!    shifts the lane's list; a wave skips the chain when no lane's record
//!    beats its 64th entry. Ties between equal logits arrive in ascending id,
//!    and an equal logit never beats an earlier one, so every lane list is in
//!    canonical order.
//! 2. Extraction. `pool` rounds. Each round the wave takes the canonical best
//!    of its 32 lane heads with a five-level exchange butterfly (`row_xmask`
//!    DPP and `v_permlanex16_b32`), the eight wave winners meet through 64
//!    bytes of LDS (two workgroup barriers) and every wave takes the best of
//!    the eight. The round winner is written to the pool register pair at
//!    lane `round % 32` with `v_writelane_b32`, and the winning lane (the
//!    unique lane whose head carries that id) pops its list.
//! 3. Wave 0 finishes (the other waves exit): weights, the 64-leaf tree, the
//!    stable weight sort as a rank scatter, the `min_p` cut, the serial
//!    nucleus cumulative, normalization, and the order-preserving `p > 0`
//!    compaction (a prefix-count scatter), then the stores.
use super::common::{add64, bload, lit, mem, op, s, s_add_i32, smem, sop, sr, srd_tail, v, vr};
use crate::{
    insn::{Instruction, MemoryClass},
    reg::{Kind, Live, RegRef},
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};
use peacemaker_author::{
    ready, retire, End, Free, Gfx1151, Gfx1201, LdsRegion, Scc, Target, Uniform, Wave, WgUniform,
    Workgroup,
};
use std::cell::Cell;

type Wg<'b, T> = Workgroup<'b, T, Builder>;
type W<'a, T> = Wave<'a, T, Builder>;
type R = Result<(), String>;

/// Static LDS: eight 8-byte (logit, id) wave winners.
pub const STATIC_LDS: u32 = 64;

pub enum Ex {}

pub fn symbol(arch: Arch) -> String {
    format!("spec_cpu_ar_rows_parallel_f32_pm_{}", arch.name())
}
pub fn module(arch: Arch) -> String {
    format!("spec_sampling_rows_parallel_pm_{}", arch.name())
}

pub fn kernargs() -> KernargLayout {
    KernargLayout::new(56)
        .pointer("logits", 0)
        .pointer("counts", 8)
        .pointer("ids", 16)
        .pointer("probabilities", 24)
        .hidden("vocab", 32, 4, "by_value")
        .hidden("pool", 36, 4, "by_value")
        .hidden("cap", 40, 4, "by_value")
        .hidden("temperature", 44, 4, "by_value")
        .hidden("top_p", 48, 4, "by_value")
        .hidden("min_p", 52, 4, "by_value")
}

// ---- SGPRs ------------------------------------------------------------------
const KARG: u8 = 0;
const A0: u8 = 8;
const A1: u8 = 16;
// kernarg blob: logits 8:9, counts 10:11, ids 12:13, probs 14:15, vocab 16,
// pool 17, cap 18, temperature 19, top_p 20, min_p 21.
const LOGITS: u8 = 8;
const COUNTS: u8 = 10;
const IDS: u8 = 12;
const PROBS: u8 = 14;
const VOCAB: u8 = 16;
const POOLIN: u8 = 17;
const CAPIN: u8 = 18;
const TEMP: u8 = 19;
const TOPP: u8 = 20;
const MINP: u8 = 21;
const ROW: u8 = 24;
const VOCAB4: u8 = 25;
const TRIPS: u8 = 26;
const TRIP: u8 = 27;
const POOL: u8 = 28;
const WAVE: u8 = 29;
const T0: u8 = 30;
const T1: u8 = 31;
const T2: u8 = 32;
const T3: u8 = 33;
const SRD_L: u8 = 40;
const SRD_C: u8 = 44;
const SRD_I: u8 = 48;
const SRD_P: u8 = 52;
/// Lane-mask pairs (the high word is never written in wave32; M7 reads VOP3 masks as pairs).
const MA: u8 = 56;
const MB: u8 = 58;
const MC: u8 = 60;
const MD: u8 = 62;
const ME: u8 = 64;
const MKA: u8 = 66;
const MKB: u8 = 68;
const MASKS: [u8; 7] = [MA, MB, MC, MD, ME, MKA, MKB];
const PSEL: u8 = 70;
const RND: u8 = 71;
const REND: u8 = 72;
const S_WL: u8 = 73;
const S_WI: u8 = 74;
const S_LN: u8 = 75;
const S_TOPL: u8 = 76;
const S_TOPI: u8 = 77;
const S_HAVE: u8 = 78;
const S_MAXL: u8 = 79;
const S_SUM: u8 = 80;
const S_GOOD: u8 = 81;
const S_CAP: u8 = 82;
const S_CAPC: u8 = 83;
const S_SUMF: u8 = 84;
const S_BND: u8 = 85;
const S_I: u8 = 86;
const S_LIM: u8 = 87;
const S_I2: u8 = 88;
const S_LIM2: u8 = 89;
const S_CUM: u8 = 90;
const S_DEN: u8 = 91;
const S_N: u8 = 92;
const S_X: u8 = 93;
const S_Y: u8 = 94;
const S_Z: u8 = 95;
const S_NOUT: u8 = 34;
const S_PID: u8 = 35;
const S_CA: u8 = 36;
const S_W0: u8 = 37;
const S_END: u8 = 96;

// ---- VGPRs ------------------------------------------------------------------
const TID: u8 = 0;
const LANE: u8 = 1;
const LANE4: u8 = 2;
const LDSW: u8 = 3;
const LDSR: u8 = 4;
const VZ: u8 = 5;
const VOFF: u8 = 6;
const CL: u8 = 7;
const CI: u8 = 8;
const XL: u8 = 9;
const XI: u8 = 10;
const VT0: u8 = 11;
const X: u8 = 16; // four loads 16..19
/// Lane list, logit desc / id asc: logits at LG + i, ids at ID + i.
const LG: u8 = 24;
const ID: u8 = 88;
/// Pool records, lane `r % 32` of the A (r < 32) or B (r >= 32) register.
const PAL: u8 = 152;
const PAI: u8 = 153;
const PBL: u8 = 154;
const PBI: u8 = 155;
const WL: u8 = 156;
const WI: u8 = 157;
const PL: u8 = 158;
const PI: u8 = 159;
const EX: u8 = 160; // 160..175: eight (logit, id) wave winners
const INV: u8 = 176;
const VMAX: u8 = 177;
const VXA: u8 = 178;
const VXB: u8 = 179;
const E_T: u8 = 180;
const E_R: u8 = 181;
const E_LO: u8 = 182;
const VN: u8 = 183;
const WA: u8 = 184;
const WB: u8 = 185;
const IA: u8 = 186;
const IB: u8 = 187;
const VS: u8 = 188;
const VP: u8 = 189;
const VSB: u8 = 190;
const VQ: u8 = 191;
const RKA: u8 = 192;
const RKB: u8 = 193;
const VTMP: u8 = 194;
const VTMP2: u8 = 195;
const OWA: u8 = 196;
const OWB: u8 = 197;
const OIA: u8 = 198;
const OIB: u8 = 199;
const VF: u8 = 200;
const VTHR: u8 = 201;
const VCUM: u8 = 202;
const VDEN: u8 = 203;
const NWA: u8 = 204;
const NWB: u8 = 205;
const FWA: u8 = 206;
const FWB: u8 = 207;
const FIA: u8 = 208;
const FIB: u8 = 209;
const RKA2: u8 = 210;
const RKB2: u8 = 211;
const DT: u8 = 212; // five division temporaries 212..216
const VD: u8 = 217;
const V_END: u8 = 224;

#[derive(Default)]
struct Gen {
    n: Cell<u32>,
}
impl Gen {
    fn lbl(&self, base: &str) -> String {
        let n = self.n.get();
        self.n.set(n + 1);
        format!(".Lspr_{base}{n}")
    }
}

pub fn emit(arch: Arch) -> Result<Emitted, String> {
    if !matches!(arch, Arch::Gfx1151 | Arch::Gfx1201) {
        return Err("spec_sampling_rows_pm: built for gfx1151 and gfx1201 only".into());
    }
    let kspec = KernelSpec {
        kernel_id: "spec_sampling_rows_pm".into(),
        variant: "canonical_parallel_f32".into(),
        arch,
        symbol: symbol(arch),
        kernargs: kernargs(),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 256,
        group_segment_fixed_size: STATIC_LDS,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan(arch)?);
    b.enable_delay_alu();
    let g = Gen::default();
    match arch {
        Arch::Gfx1151 => kernel(&mut Workgroup::<Gfx1151, Builder>::new(&mut b)?, &g, arch)?,
        _ => kernel(&mut Workgroup::<Gfx1201, Builder>::new(&mut b)?, &g, arch)?,
    }
    b.finish()
}

/// The single-symbol module `spec_sampling_rows_parallel_pm_<arch>`.
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = vec![emit(arch)?];
    let (text, proof) = super::iu4_gemm::module(&emitted, &module(arch))?;
    Ok((emitted, text, proof))
}

fn plan(arch: Arch) -> Result<RegPlan, String> {
    let mut p = RegPlan::new(u16::from(V_END), 104)?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    if !arch.gfx12() {
        p.s::<1>("wg_x_in", 2, Live::Whole)?;
    }
    p.s::<8>("kernargs_0x00", A0, Live::Whole)?;
    p.s::<8>("kernargs_0x20", A1, Live::Whole)?;
    for base in (24u8..S_END).step_by(8) {
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
fn cmp_wg<T: Target>(
    wg: &mut Wg<T>,
    text: String,
    uses: &[RegRef],
) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(Instruction::new(text, vec![], uses.to_vec()))
}
fn cmp_w<T: Target>(w: &mut W<T>, text: String, uses: &[RegRef]) -> Result<Uniform<Scc>, String> {
    w.scmp(Instruction::new(text, vec![], uses.to_vec()))
}
fn ds_st(text: String, addr: u8, data: RegRef) -> Instruction {
    Instruction::new(text, vec![], vec![v(addr), data]).memory(MemoryClass::DsStore)
}
fn ds_ld(text: String, dst: RegRef, addr: u8) -> Instruction {
    Instruction::new(text, vec![dst], vec![v(addr)]).memory(MemoryClass::DsLoad)
}
fn off(o: u32) -> String {
    if o == 0 {
        String::new()
    } else {
        format!(" offset:{o}")
    }
}
fn soff(a: Arch) -> &'static str {
    if a.gfx12() {
        "null"
    } else {
        "0"
    }
}
fn s_sub_i32(a: Arch) -> &'static str {
    if a.gfx12() {
        "s_sub_co_i32"
    } else {
        "s_sub_i32"
    }
}
/// `s{dst} = s{src} - imm`.
fn sub_imm(b: &mut Builder, dst: u8, src: u8, imm: u32) -> R {
    sop(
        b,
        format!("{} s{dst}, s{src}, {}", s_sub_i32(b.spec.arch), lit(imm)),
        &[dst],
        &[src],
    )
}
/// `s{dst} = s{src} + imm`.
fn add_imm(b: &mut Builder, dst: u8, src: u8, imm: u32) -> R {
    sop(
        b,
        format!("{} s{dst}, s{src}, {}", s_add_i32(b.spec.arch), lit(imm)),
        &[dst],
        &[src],
    )
}
/// `s[dst:dst+1] = s[src:src+1] + x * y` (64-bit).
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
/// Raw buffer store of one dword at `v{voff} + offset`.
fn bstore(b: &mut Builder, data: u8, voff: u8, srd: u8, offset: u32) -> R {
    let a = b.spec.arch;
    if offset > a.buffer_offset_max() {
        return Err(format!("buffer offset {offset} exceeds the field"));
    }
    mem(
        b,
        format!(
            "buffer_store_b32 v{data}, v{voff}, s[{srd}:{}], {} offen{}",
            srd + 3,
            soff(a),
            off(offset)
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
/// `(wl, wi) = best((pl, pi), (wl, wi))` in the canonical order: higher logit,
/// equal logits the lower id (ties of identical padding keep `w`). Clobbers MA, MB, MC.
fn better(b: &mut Builder, pl: u8, pi: u8, wl: u8, wi: u8) -> R {
    alu(b, "v_cmp_gt_f32_e64", O::M(MA), &[O::V(pl), O::V(wl)])?;
    alu(b, "v_cmp_eq_f32_e64", O::M(MB), &[O::V(pl), O::V(wl)])?;
    alu(b, "v_cmp_lt_u32_e64", O::M(MC), &[O::V(pi), O::V(wi)])?;
    alu(b, "s_and_b32", O::M(MB), &[O::M(MB), O::M(MC)])?;
    alu(b, "s_or_b32", O::M(MA), &[O::M(MA), O::M(MB)])?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(wl),
        &[O::V(wl), O::V(pl), O::M(MA)],
    )?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(wi),
        &[O::V(wi), O::V(pi), O::M(MA)],
    )
}
/// Tree sum over the 64 leaves held as `VS = leaf[t] + leaf[t + 32]`: strides 16, 8, 4, 2, 1
/// with the lower lane on the left; lane 0 ends with the sum.
fn tree(b: &mut Builder) -> R {
    for m in [16u32, 8, 4, 2, 1] {
        xch(b, VP, VS, m)?;
        alu(b, "v_add_f32_e32", O::V(VS), &[O::V(VS), O::V(VP)])?;
    }
    Ok(())
}

// ---- kernel -----------------------------------------------------------------
fn kernel<T: Target>(wg: &mut Wg<T>, g: &Gen, arch: Arch) -> R {
    let ex = wg.lds::<Ex>("ex", 0, STATIC_LDS)?;
    let end = wg.exit(".Lspr_end")?;
    prologue(wg, arch)?;
    scan(wg, g)?;
    let ex = round_loop(wg, g, ex, false)?;
    // A pool above 32 continues in the B registers.
    let skip = cmp_wg(wg, format!("s_cmp_le_u32 s{POOL}, 32"), &[s(POOL)])?;
    let ex = wg.wg_skip_if(skip, &g.lbl("rounds_b"), ex, |wg, ex| {
        round_loop(wg, g, ex, true)
    })?;
    let _ = ex;
    tail(wg, g, &end)?;
    wg.end(end)
}

fn prologue<T: Target>(wg: &mut Wg<T>, arch: Arch) -> R {
    let b = wg.isa();
    smem(b, A0, 8, KARG, 0)?;
    smem(b, A1, 8, KARG, 0x20)?;
    // Lane constants and the dedicated LDS address registers (before the first branch).
    alu(b, "v_and_b32_e32", O::V(LANE), &[O::N(31), O::V(TID)])?;
    alu(b, "v_lshlrev_b32_e32", O::V(LANE4), &[O::N(2), O::V(LANE)])?;
    alu(b, "v_lshrrev_b32_e32", O::V(VT0), &[O::N(5), O::V(TID)])?;
    alu(b, "v_readfirstlane_b32", O::S(WAVE), &[O::V(VT0)])?;
    alu(b, "v_lshlrev_b32_e32", O::V(LDSW), &[O::N(3), O::V(VT0)])?;
    alu(b, "v_mov_b32_e32", O::V(LDSR), &[O::N(0)])?;
    alu(b, "v_mov_b32_e32", O::V(VZ), &[O::N(0)])?;
    if arch.gfx12() {
        sop(b, format!("s_mov_b32 s{ROW}, ttmp9"), &[ROW], &[])?;
    } else {
        sop(b, format!("s_mov_b32 s{ROW}, s2"), &[ROW], &[2])?;
    }
    for m in MASKS {
        sop(b, format!("s_mov_b32 s{}, 0", m + 1), &[m + 1], &[])?
    }
    sop(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[PSEL], &[])?;
    // pool = clamp(pool, 1, 64); the row's logits as a raw descriptor.
    sop(
        b,
        format!("s_max_i32 s{POOL}, s{POOLIN}, 1"),
        &[POOL],
        &[POOLIN],
    )?;
    sop(
        b,
        format!("s_min_i32 s{POOL}, s{POOL}, 64"),
        &[POOL],
        &[POOL],
    )?;
    sop(b, format!("s_max_i32 s{T0}, s{VOCAB}, 0"), &[T0], &[VOCAB])?;
    sop(
        b,
        format!("s_lshl_b32 s{VOCAB4}, s{T0}, 2"),
        &[VOCAB4],
        &[T0],
    )?;
    add64_mul(b, SRD_L, LOGITS, ROW, VOCAB4, T1, T2)?;
    srd_tail(b, SRD_L, Some(VOCAB4))?;
    // counts[row], ids[row * 64], probs[row * 64].
    sop(b, format!("s_lshl_b32 s{T0}, s{ROW}, 2"), &[T0], &[ROW])?;
    sop(b, format!("s_mov_b32 s{T3}, 4"), &[T3], &[])?;
    add64(b, SRD_C, COUNTS, T0)?;
    srd_tail(b, SRD_C, Some(T3))?;
    sop(b, format!("s_lshl_b32 s{T0}, s{ROW}, 8"), &[T0], &[ROW])?;
    sop(b, format!("s_mov_b32 s{T3}, {}", lit(256)), &[T3], &[])?;
    add64(b, SRD_I, IDS, T0)?;
    srd_tail(b, SRD_I, Some(T3))?;
    add64(b, SRD_P, PROBS, T0)?;
    srd_tail(b, SRD_P, Some(T3))
}

// ---- scan -------------------------------------------------------------------
/// Insert the carry `(CL, CI)` into this lane's sorted list; `MC` holds the
/// lane mask `CL > L[63]`. Slot `i` becomes `m_i ? (m_{i-1} ? L[i-1] : c) : L[i]`
/// with `m_i = c > L[i]` (monotone in `i`), walking from the last slot down so
/// `L[i-1]` is still the old value.
fn insert_chain(b: &mut Builder) -> R {
    let (mut mi, mut mp) = (MC, MD);
    for i in (1..64u8).rev() {
        alu(
            b,
            "v_cmp_gt_f32_e64",
            O::M(mp),
            &[O::V(CL), O::V(LG + i - 1)],
        )?;
        alu(
            b,
            "v_cndmask_b32_e64",
            O::V(XL),
            &[O::V(CL), O::V(LG + i - 1), O::M(mp)],
        )?;
        alu(
            b,
            "v_cndmask_b32_e64",
            O::V(XI),
            &[O::V(CI), O::V(ID + i - 1), O::M(mp)],
        )?;
        alu(
            b,
            "v_cndmask_b32_e64",
            O::V(LG + i),
            &[O::V(LG + i), O::V(XL), O::M(mi)],
        )?;
        alu(
            b,
            "v_cndmask_b32_e64",
            O::V(ID + i),
            &[O::V(ID + i), O::V(XI), O::M(mi)],
        )?;
        std::mem::swap(&mut mi, &mut mp);
    }
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(LG),
        &[O::V(LG), O::V(CL), O::M(mi)],
    )?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(ID),
        &[O::V(ID), O::V(CI), O::M(mi)],
    )
}

fn scan<T: Target>(wg: &mut Wg<T>, g: &Gen) -> R {
    let b = wg.isa();
    for i in 0..64u8 {
        alu(b, "v_mov_b32_e32", O::V(LG + i), &[O::K("0xff800000")])?;
        alu(b, "v_mov_b32_e32", O::V(ID + i), &[O::N(-1)])?;
    }
    for (pl, pi) in [(PAL, PAI), (PBL, PBI)] {
        alu(b, "v_mov_b32_e32", O::V(pl), &[O::K("0xff800000")])?;
        alu(b, "v_mov_b32_e32", O::V(pi), &[O::N(-1)])?;
    }
    alu(b, "v_lshlrev_b32_e32", O::V(VOFF), &[O::N(2), O::V(TID)])?;
    sop(b, format!("s_max_i32 s{T0}, s{VOCAB}, 0"), &[T0], &[VOCAB])?;
    add_imm(b, T0, T0, 1023)?;
    sop(
        b,
        format!("s_lshr_b32 s{TRIPS}, s{T0}, 10"),
        &[TRIPS],
        &[T0],
    )?;
    sop(b, format!("s_mov_b32 s{TRIP}, 0"), &[TRIP], &[])?;
    let none = cmp_wg(wg, format!("s_cmp_eq_u32 s{TRIPS}, 0"), &[s(TRIPS)])?;
    wg.wg_skip_if(none, &g.lbl("scan_done"), (), |wg, ()| {
        wg.loop_carried(&g.lbl("scan"), (), |wg, ()| {
            let b = wg.isa();
            for j in 0..4u8 {
                bload(b, 1, X + j, VOFF, SRD_L, 1024 * u32::from(j))?;
            }
            for j in 0..4u8 {
                let b = wg.isa();
                let byte = if j == 0 {
                    VOFF
                } else {
                    alu(
                        b,
                        "v_add_nc_u32_e32",
                        O::V(VT0),
                        &[O::N(1024 * i64::from(j)), O::V(VOFF)],
                    )?;
                    VT0
                };
                // Valid: the token is inside the row and its logit is finite.
                alu(b, "v_cmp_gt_u32_e64", O::M(MA), &[O::S(VOCAB4), O::V(byte)])?;
                alu(
                    b,
                    "v_cmp_class_f32_e64",
                    O::M(MB),
                    &[O::V(X + j), O::K("0x1f8")],
                )?;
                alu(b, "s_and_b32", O::M(MA), &[O::M(MA), O::M(MB)])?;
                alu(
                    b,
                    "v_cndmask_b32_e64",
                    O::V(CL),
                    &[O::K("0xff800000"), O::V(X + j), O::M(MA)],
                )?;
                alu(b, "v_lshrrev_b32_e32", O::V(CI), &[O::N(2), O::V(byte)])?;
                alu(b, "v_cmp_gt_f32_e64", O::M(MC), &[O::V(CL), O::V(LG + 63)])?;
                let quiet = cmp_w(wg, format!("s_cmp_eq_u32 s{MC}, 0"), &[mask(MC)])?;
                wg.skip_if(quiet, &g.lbl("ins"), (), |w, ()| insert_chain(w.isa()))?;
            }
            let b = wg.isa();
            alu(
                b,
                "v_add_nc_u32_e32",
                O::V(VOFF),
                &[O::K("0x1000"), O::V(VOFF)],
            )?;
            add_imm(b, TRIP, TRIP, 1)?;
            let more = cmp_wg(
                wg,
                format!("s_cmp_lt_u32 s{TRIP}, s{TRIPS}"),
                &[s(TRIP), s(TRIPS)],
            )?;
            Ok(((), more))
        })
    })
}

// ---- extraction rounds ------------------------------------------------------
/// Rounds `[0, min(pool, 32))` (record in the A registers) or `[32, pool)` (B).
fn round_loop<T: Target>(
    wg: &mut Wg<T>,
    g: &Gen,
    ex: LdsRegion<Ex, Free>,
    second: bool,
) -> Result<LdsRegion<Ex, Free>, String> {
    let b = wg.isa();
    if second {
        sop(b, format!("s_mov_b32 s{RND}, 32"), &[RND], &[])?;
        sop(b, format!("s_mov_b32 s{REND}, s{POOL}"), &[REND], &[POOL])?;
    } else {
        sop(b, format!("s_mov_b32 s{RND}, 0"), &[RND], &[])?;
        sop(
            b,
            format!("s_min_u32 s{REND}, s{POOL}, 32"),
            &[REND],
            &[POOL],
        )?;
    }
    let (pl, pi) = if second { (PBL, PBI) } else { (PAL, PAI) };
    wg.loop_carried(&g.lbl("round"), ex, |wg, ex| {
        let b = wg.isa();
        // The wave's canonical best head.
        alu(b, "v_mov_b32_e32", O::V(WL), &[O::V(LG)])?;
        alu(b, "v_mov_b32_e32", O::V(WI), &[O::V(ID)])?;
        for m in [1u32, 2, 4, 8, 16] {
            xch(b, PL, WL, m)?;
            xch(b, PI, WI, m)?;
            better(b, PL, PI, WL, WI)?;
        }
        // The eight wave winners.
        let st = wg.begin_write(ex);
        let (ew, pend) = wg.ds_store(
            st,
            ds_st(
                format!("ds_store_b64 v{LDSW}, v[{WL}:{}]", WL + 1),
                LDSW,
                vr(WL, 2),
            ),
        )?;
        let d = wg.wait(pend)?;
        let (ep,) = wg.barrier((ready(ew, d),))?;
        for k in 0..4u8 {
            wg.ds_load(
                &ep,
                ds_ld(
                    format!(
                        "ds_load_b128 v[{}:{}], v{LDSR}{}",
                        EX + 4 * k,
                        EX + 4 * k + 3,
                        off(16 * u32::from(k))
                    ),
                    vr(EX + 4 * k, 4),
                    LDSR,
                ),
            )?;
        }
        let b = wg.isa();
        for k in 1..8u8 {
            better(b, EX + 2 * k, EX + 2 * k + 1, EX, EX + 1)?;
        }
        let (ex,) = wg.barrier((retire(ep),))?;
        let b = wg.isa();
        // Record the round winner at lane `round % 32` of the pool registers.
        alu(b, "v_readfirstlane_b32", O::S(S_WL), &[O::V(EX)])?;
        alu(b, "v_readfirstlane_b32", O::S(S_WI), &[O::V(EX + 1)])?;
        sop(b, format!("s_and_b32 s{S_LN}, s{RND}, 31"), &[S_LN], &[RND])?;
        op(
            b,
            format!("v_writelane_b32 v{pl}, s{S_WL}, s{S_LN}"),
            &[v(pl)],
            &[v(pl), s(S_WL), s(S_LN)],
        )?;
        op(
            b,
            format!("v_writelane_b32 v{pi}, s{S_WI}, s{S_LN}"),
            &[v(pi)],
            &[v(pi), s(S_WI), s(S_LN)],
        )?;
        // The lane whose head carries the winner's id pops its list.
        alu(b, "v_cmp_eq_u32_e64", O::M(MA), &[O::S(S_WI), O::V(ID)])?;
        for i in 0..63u8 {
            alu(
                b,
                "v_cndmask_b32_e64",
                O::V(LG + i),
                &[O::V(LG + i), O::V(LG + i + 1), O::M(MA)],
            )?;
            alu(
                b,
                "v_cndmask_b32_e64",
                O::V(ID + i),
                &[O::V(ID + i), O::V(ID + i + 1), O::M(MA)],
            )?;
        }
        alu(
            b,
            "v_cndmask_b32_e64",
            O::V(LG + 63),
            &[O::V(LG + 63), O::K("0xff800000"), O::M(MA)],
        )?;
        alu(
            b,
            "v_cndmask_b32_e64",
            O::V(ID + 63),
            &[O::V(ID + 63), O::N(-1), O::M(MA)],
        )?;
        add_imm(b, RND, RND, 1)?;
        let more = cmp_wg(
            wg,
            format!("s_cmp_lt_u32 s{RND}, s{REND}"),
            &[s(RND), s(REND)],
        )?;
        Ok((ex, more))
    })
}

// ---- finish (wave 0) --------------------------------------------------------
/// Scatter lane-resident entries to the lane given by `idx`: for every source lane `j`
/// (A: `j`, B: `32 + j`) whose index is `p`, destination position `p` receives `(w, id)`.
/// Positions without a source stay zero; an index outside `0..64` drops its entry.
fn scatter(
    b: &mut Builder,
    srcw: [u8; 2],
    srcid: [u8; 2],
    idx: [u8; 2],
    dstw: [u8; 2],
    dstid: [u8; 2],
) -> R {
    for h in 0..2 {
        alu(b, "v_mov_b32_e32", O::V(dstw[h]), &[O::N(0)])?;
        alu(b, "v_mov_b32_e32", O::V(dstid[h]), &[O::N(0)])?;
    }
    for j in 0..64usize {
        let (h, lane) = (j / 32, (j % 32) as i64);
        alu(b, "v_readlane_b32", O::S(S_X), &[O::V(idx[h]), O::N(lane)])?;
        alu(b, "v_readlane_b32", O::S(S_Y), &[O::V(srcw[h]), O::N(lane)])?;
        alu(
            b,
            "v_readlane_b32",
            O::S(S_Z),
            &[O::V(srcid[h]), O::N(lane)],
        )?;
        sub_imm(b, T0, S_X, 32)?;
        alu(b, "v_cmp_eq_u32_e64", O::M(MA), &[O::S(S_X), O::V(LANE)])?;
        alu(b, "v_cmp_eq_u32_e64", O::M(MB), &[O::S(T0), O::V(LANE)])?;
        alu(
            b,
            "v_cndmask_b32_e64",
            O::V(dstw[0]),
            &[O::V(dstw[0]), O::S(S_Y), O::M(MA)],
        )?;
        alu(
            b,
            "v_cndmask_b32_e64",
            O::V(dstid[0]),
            &[O::V(dstid[0]), O::S(S_Z), O::M(MA)],
        )?;
        alu(
            b,
            "v_cndmask_b32_e64",
            O::V(dstw[1]),
            &[O::V(dstw[1]), O::S(S_Y), O::M(MB)],
        )?;
        alu(
            b,
            "v_cndmask_b32_e64",
            O::V(dstid[1]),
            &[O::V(dstid[1]), O::S(S_Z), O::M(MB)],
        )?;
    }
    Ok(())
}

/// Weight and id registers of one pool half (`h` 0: ranks 0..32, 1: 32..64).
fn weights(b: &mut Builder, h: usize) -> R {
    let (pl, pi, w, idr) = if h == 0 {
        (PAL, PAI, WA, IA)
    } else {
        (PBL, PBI, WB, IB)
    };
    let idx = if h == 0 {
        LANE
    } else {
        alu(b, "v_add_nc_u32_e32", O::V(VTMP), &[O::N(32), O::V(LANE)])?;
        VTMP
    };
    alu(b, "v_cmp_gt_u32_e64", O::M(MA), &[O::S(POOL), O::V(idx)])?;
    alu(b, "v_cmp_ne_u32_e64", O::M(MB), &[O::N(-1), O::V(pi)])?;
    alu(b, "s_and_b32", O::M(MB), &[O::M(MB), O::M(MA)])?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(idr),
        &[O::N(0), O::V(pi), O::M(MA)],
    )?;
    alu(b, "v_sub_f32_e32", O::V(VXA), &[O::V(pl), O::V(VMAX)])?;
    alu(b, "v_mul_f32_e32", O::V(VXA), &[O::V(VXA), O::V(INV)])?;
    super::qsa_gather::expf(b, w, VXA, false, E_T, E_R, E_LO, MC)?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(w),
        &[O::N(0), O::V(w), O::M(MB)],
    )?;
    alu(
        b,
        "v_cmp_class_f32_e64",
        O::M(MC),
        &[O::V(w), O::K("0x1f8")],
    )?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(w),
        &[O::N(0), O::V(w), O::M(MC)],
    )
}

/// Nucleus loop over `owx` positions `[0, lim)`: `cum += w[i]`, leave at the first
/// `cum >= threshold` with `S_I` at that position. `i`/`lim` are the SGPR counters.
fn nucleus_loop<T: Target>(wg: &mut W<T>, g: &Gen, owx: u8, i: u8, lim: u8) -> R {
    wg.loop_until(&g.lbl("nu"), &g.lbl("nux"), |w, x| {
        let c = cmp_w(w, format!("s_cmp_ge_u32 s{i}, s{lim}"), &[s(i), s(lim)])?;
        w.break_if(c, x)?;
        let b = w.isa();
        alu(b, "v_readlane_b32", O::S(S_X), &[O::V(owx), O::S(i)])?;
        alu(b, "v_add_f32_e32", O::V(VCUM), &[O::S(S_X), O::V(VCUM)])?;
        alu(b, "v_cmp_ge_f32_e64", O::M(MA), &[O::V(VCUM), O::V(VTHR)])?;
        let hit = cmp_w(w, format!("s_cmp_lg_u32 s{MA}, 0"), &[mask(MA)])?;
        w.break_if(hit, x)?;
        add_imm(w.isa(), i, i, 1)
    })
}

fn tail<T: Target>(wg: &mut Wg<T>, g: &Gen, end: &End) -> R {
    // Wave 0 alone finishes the row; no barrier follows.
    let wave0 = cmp_w(wg, format!("s_cmp_eq_u32 s{WAVE}, 0"), &[s(WAVE)])?;
    // The wave-scope exit (the workgroup scope has its own `exit_unless`).
    (&mut **wg).exit_unless(wave0, end)?;
    let b = wg.isa();
    let dt = [DT, DT + 1, DT + 2, DT + 3, DT + 4];
    // inv_temp = RN(1 / temperature).
    alu(b, "v_mov_b32_e32", O::V(VN), &[O::K("1.0")])?;
    alu(b, "v_mov_b32_e32", O::V(VD), &[O::S(TEMP)])?;
    super::qsa_gather::fdiv(b, INV, VN, VD, dt)?;
    // The canonical best record (round 0) is the row maximum and the point-mass token.
    alu(b, "v_readlane_b32", O::S(S_TOPL), &[O::V(PAL), O::N(0)])?;
    alu(b, "v_readlane_b32", O::S(S_TOPI), &[O::V(PAI), O::N(0)])?;
    op(b, format!("s_cmp_lg_u32 s{S_TOPI}, -1"), &[], &[s(S_TOPI)])?;
    sop(
        b,
        format!("s_cselect_b32 s{S_MAXL}, s{S_TOPL}, 0xff800000"),
        &[S_MAXL],
        &[S_TOPL],
    )?;
    sop(b, format!("s_cselect_b32 s{S_HAVE}, 1, 0"), &[S_HAVE], &[])?;
    sop(
        b,
        format!("s_cselect_b32 s{S_PID}, s{S_TOPI}, 0"),
        &[S_PID],
        &[S_TOPI],
    )?;
    alu(b, "v_mov_b32_e32", O::V(VMAX), &[O::S(S_MAXL)])?;
    // Weights, then the full-pool mass through the fixed tree.
    weights(b, 0)?;
    weights(b, 1)?;
    alu(b, "v_add_f32_e32", O::V(VS), &[O::V(WA), O::V(WB)])?;
    tree(b)?;
    alu(b, "v_readfirstlane_b32", O::S(S_SUM), &[O::V(VS)])?;
    // good = sum > 0 and finite.
    alu(b, "v_mov_b32_e32", O::V(VSB), &[O::S(S_SUM)])?;
    alu(b, "v_cmp_lt_f32_e64", O::M(MA), &[O::N(0), O::V(VSB)])?;
    alu(
        b,
        "v_cmp_class_f32_e64",
        O::M(MB),
        &[O::V(VSB), O::K("0x1f8")],
    )?;
    alu(b, "s_and_b32", O::M(MA), &[O::M(MA), O::M(MB)])?;
    op(b, format!("s_cmp_lg_u32 s{MA}, 0"), &[], &[mask(MA)])?;
    sop(b, format!("s_cselect_b32 s{S_GOOD}, 1, 0"), &[S_GOOD], &[])?;
    // Stable descending sort by weight: rank = #(w_j > w_i) + #(j < i, w_j == w_i).
    alu(b, "v_mov_b32_e32", O::V(RKA), &[O::N(0)])?;
    alu(b, "v_mov_b32_e32", O::V(RKB), &[O::N(0)])?;
    for j in 0..64i64 {
        let src = if j < 32 { WA } else { WB };
        alu(b, "v_readlane_b32", O::S(S_X), &[O::V(src), O::N(j % 32)])?;
        for h in 0..2usize {
            let (wh, rk) = if h == 0 { (WA, RKA) } else { (WB, RKB) };
            alu(b, "v_cmp_gt_f32_e64", O::M(MA), &[O::S(S_X), O::V(wh)])?;
            alu(b, "v_cmp_eq_f32_e64", O::M(MB), &[O::S(S_X), O::V(wh)])?;
            // j precedes position i = lane (h 0) or 32 + lane (h 1).
            if h == 0 || j >= 32 {
                alu(
                    b,
                    "v_cmp_lt_u32_e64",
                    O::M(MC),
                    &[O::N(j - 32 * h as i64), O::V(LANE)],
                )?;
                alu(b, "s_and_b32", O::M(MB), &[O::M(MB), O::M(MC)])?;
            }
            alu(b, "s_or_b32", O::M(MA), &[O::M(MA), O::M(MB)])?;
            alu(
                b,
                "v_cndmask_b32_e64",
                O::V(VTMP),
                &[O::N(0), O::N(1), O::M(MA)],
            )?;
            alu(b, "v_add_nc_u32_e32", O::V(rk), &[O::V(rk), O::V(VTMP)])?;
        }
    }
    scatter(b, [WA, WB], [IA, IB], [RKA, RKB], [OWA, OWB], [OIA, OIB])?;
    // cap = clamp(cap, 1, pool); the first rank i >= 1 below min_p * w0 cuts.
    sop(
        b,
        format!("s_max_i32 s{S_CAP}, s{CAPIN}, 1"),
        &[S_CAP],
        &[CAPIN],
    )?;
    sop(
        b,
        format!("s_min_i32 s{S_CAP}, s{S_CAP}, s{POOL}"),
        &[S_CAP],
        &[S_CAP, POOL],
    )?;
    alu(b, "v_readlane_b32", O::S(S_W0), &[O::V(OWA), O::N(0)])?;
    alu(b, "v_mov_b32_e32", O::V(VTMP), &[O::S(S_W0)])?;
    alu(b, "v_mul_f32_e32", O::V(VF), &[O::S(MINP), O::V(VTMP)])?;
    alu(b, "v_cmp_gt_f32_e64", O::M(MD), &[O::S(MINP), O::N(0)])?;
    alu(b, "v_cmp_lt_f32_e64", O::M(MA), &[O::V(OWA), O::V(VF)])?;
    alu(b, "v_cmp_lt_u32_e64", O::M(MB), &[O::N(0), O::V(LANE)])?;
    alu(b, "s_and_b32", O::M(MA), &[O::M(MA), O::M(MB)])?;
    alu(b, "v_cmp_gt_u32_e64", O::M(MB), &[O::S(S_CAP), O::V(LANE)])?;
    alu(b, "s_and_b32", O::M(MA), &[O::M(MA), O::M(MB)])?;
    alu(b, "s_and_b32", O::M(MA), &[O::M(MA), O::M(MD)])?;
    alu(b, "v_cmp_lt_f32_e64", O::M(MB), &[O::V(OWB), O::V(VF)])?;
    sub_imm(b, T0, S_CAP, 32)?;
    alu(b, "v_cmp_gt_i32_e64", O::M(MC), &[O::S(T0), O::V(LANE)])?;
    alu(b, "s_and_b32", O::M(MB), &[O::M(MB), O::M(MC)])?;
    alu(b, "s_and_b32", O::M(MB), &[O::M(MB), O::M(MD)])?;
    alu(b, "s_ctz_i32_b32", O::S(T0), &[O::M(MA)])?;
    alu(b, "s_ctz_i32_b32", O::S(T1), &[O::M(MB)])?;
    add_imm(b, T1, T1, 32)?;
    op(b, format!("s_cmp_lg_u32 s{MB}, 0"), &[], &[mask(MB)])?;
    sop(
        b,
        format!("s_cselect_b32 s{T2}, s{T1}, s{S_CAP}"),
        &[T2],
        &[T1, S_CAP],
    )?;
    op(b, format!("s_cmp_lg_u32 s{MA}, 0"), &[], &[mask(MA)])?;
    sop(
        b,
        format!("s_cselect_b32 s{S_CAPC}, s{T0}, s{T2}"),
        &[S_CAPC],
        &[T0, T2],
    )?;
    // A cut pool sums its prefix through the same tree.
    alu(b, "v_cmp_gt_u32_e64", O::M(MA), &[O::S(S_CAPC), O::V(LANE)])?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(VXA),
        &[O::N(0), O::V(OWA), O::M(MA)],
    )?;
    sub_imm(b, T0, S_CAPC, 32)?;
    alu(b, "v_cmp_gt_i32_e64", O::M(MB), &[O::S(T0), O::V(LANE)])?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(VXB),
        &[O::N(0), O::V(OWB), O::M(MB)],
    )?;
    alu(b, "v_add_f32_e32", O::V(VS), &[O::V(VXA), O::V(VXB)])?;
    tree(b)?;
    alu(b, "v_readfirstlane_b32", O::S(T1), &[O::V(VS)])?;
    op(
        b,
        format!("s_cmp_lt_i32 s{S_CAPC}, s{POOL}"),
        &[],
        &[s(S_CAPC), s(POOL)],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{S_SUMF}, s{T1}, s{S_SUM}"),
        &[S_SUMF],
        &[T1, S_SUM],
    )?;
    // Nucleus: serial cumulative over the cut pool, threshold RN(top_p * mass).
    alu(b, "v_mov_b32_e32", O::V(VTMP), &[O::S(S_SUMF)])?;
    alu(b, "v_mul_f32_e32", O::V(VTHR), &[O::S(TOPP), O::V(VTMP)])?;
    alu(b, "v_mov_b32_e32", O::V(VCUM), &[O::N(0)])?;
    sop(b, format!("s_mov_b32 s{S_I}, 0"), &[S_I], &[])?;
    sop(
        b,
        format!("s_min_u32 s{S_LIM}, s{S_CAPC}, 32"),
        &[S_LIM],
        &[S_CAPC],
    )?;
    nucleus_loop(wg, g, OWA, S_I, S_LIM)?;
    let b = wg.isa();
    add_imm(b, T0, S_I, 1)?;
    op(
        b,
        format!("s_cmp_lt_u32 s{S_I}, s{S_LIM}"),
        &[],
        &[s(S_I), s(S_LIM)],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{S_BND}, s{T0}, -1"),
        &[S_BND],
        &[T0],
    )?;
    // Positions 32.. only while nothing crossed.
    op(b, format!("s_cmp_lt_i32 s{S_BND}, 0"), &[], &[s(S_BND)])?;
    sop(b, format!("s_cselect_b32 s{T0}, 1, 0"), &[T0], &[])?;
    op(b, format!("s_cmp_gt_i32 s{S_CAPC}, 32"), &[], &[s(S_CAPC)])?;
    sop(b, format!("s_cselect_b32 s{T1}, 1, 0"), &[T1], &[])?;
    sop(
        b,
        format!("s_and_b32 s{T0}, s{T0}, s{T1}"),
        &[T0],
        &[T0, T1],
    )?;
    sop(b, format!("s_mov_b32 s{S_I2}, 0"), &[S_I2], &[])?;
    sop(b, format!("s_mov_b32 s{S_LIM2}, 0"), &[S_LIM2], &[])?;
    let no_b = cmp_w(wg, format!("s_cmp_eq_u32 s{T0}, 0"), &[s(T0)])?;
    wg.skip_if(no_b, &g.lbl("nu_b"), (), |w, ()| {
        sub_imm(w.isa(), S_LIM2, S_CAPC, 32)?;
        nucleus_loop(w, g, OWB, S_I2, S_LIM2)
    })?;
    let b = wg.isa();
    add_imm(b, T0, S_I2, 33)?;
    op(
        b,
        format!("s_cmp_lt_u32 s{S_I2}, s{S_LIM2}"),
        &[],
        &[s(S_I2), s(S_LIM2)],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{S_BND}, s{T0}, s{S_BND}"),
        &[S_BND],
        &[T0, S_BND],
    )?;
    // Normalize: crossing divides the first `boundary` weights by the cumulative, no
    // crossing divides the cut pool by its mass and corrects rank 0 by the residual.
    alu(b, "v_readfirstlane_b32", O::S(S_CUM), &[O::V(VCUM)])?;
    op(b, format!("s_cmp_ge_i32 s{S_BND}, 0"), &[], &[s(S_BND)])?;
    sop(
        b,
        format!("s_cselect_b32 s{S_DEN}, s{S_CUM}, s{S_SUMF}"),
        &[S_DEN],
        &[S_CUM, S_SUMF],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{S_N}, s{S_BND}, s{S_CAPC}"),
        &[S_N],
        &[S_BND, S_CAPC],
    )?;
    alu(b, "v_mov_b32_e32", O::V(VDEN), &[O::S(S_DEN)])?;
    super::qsa_gather::fdiv(b, NWA, OWA, VDEN, dt)?;
    super::qsa_gather::fdiv(b, NWB, OWB, VDEN, dt)?;
    alu(b, "v_mov_b32_e32", O::V(VTMP), &[O::S(S_SUMF)])?;
    super::qsa_gather::fdiv(b, VQ, VCUM, VTMP, dt)?;
    alu(b, "v_sub_f32_e32", O::V(VQ), &[O::K("1.0"), O::V(VQ)])?;
    alu(b, "v_cmp_lt_f32_e64", O::M(MA), &[O::N(0), O::V(VQ)])?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(VQ),
        &[O::N(0), O::V(VQ), O::M(MA)],
    )?;
    alu(b, "v_cmp_eq_u32_e64", O::M(MB), &[O::N(0), O::V(LANE)])?;
    op(b, format!("s_cmp_lt_i32 s{S_BND}, 0"), &[], &[s(S_BND)])?;
    sop(b, format!("s_cselect_b32 s{MC}, -1, 0"), &[MC], &[])?;
    alu(b, "s_and_b32", O::M(MB), &[O::M(MB), O::M(MC)])?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(VQ),
        &[O::N(0), O::V(VQ), O::M(MB)],
    )?;
    alu(b, "v_add_f32_e32", O::V(NWA), &[O::V(NWA), O::V(VQ)])?;
    // Retain p > 0 among the first n, order preserved: prefix-count scatter.
    alu(b, "v_cmp_gt_u32_e64", O::M(MKA), &[O::S(S_N), O::V(LANE)])?;
    alu(b, "v_cmp_lt_f32_e64", O::M(MA), &[O::N(0), O::V(NWA)])?;
    alu(b, "s_and_b32", O::M(MKA), &[O::M(MKA), O::M(MA)])?;
    sub_imm(b, T0, S_N, 32)?;
    alu(b, "v_cmp_gt_i32_e64", O::M(MKB), &[O::S(T0), O::V(LANE)])?;
    alu(b, "v_cmp_lt_f32_e64", O::M(MA), &[O::N(0), O::V(NWB)])?;
    alu(b, "s_and_b32", O::M(MKB), &[O::M(MKB), O::M(MA)])?;
    alu(b, "s_bcnt1_i32_b32", O::S(S_CA), &[O::M(MKA)])?;
    alu(b, "s_bcnt1_i32_b32", O::S(T1), &[O::M(MKB)])?;
    sop(
        b,
        format!("{} s{S_NOUT}, s{S_CA}, s{T1}", s_add_i32(b.spec.arch)),
        &[S_NOUT],
        &[S_CA, T1],
    )?;
    alu(b, "v_mbcnt_lo_u32_b32", O::V(RKA2), &[O::M(MKA), O::N(0)])?;
    alu(
        b,
        "v_mbcnt_lo_u32_b32",
        O::V(RKB2),
        &[O::M(MKB), O::S(S_CA)],
    )?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(RKA2),
        &[O::N(-1), O::V(RKA2), O::M(MKA)],
    )?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(RKB2),
        &[O::N(-1), O::V(RKB2), O::M(MKB)],
    )?;
    scatter(
        b,
        [NWA, NWB],
        [OIA, OIB],
        [RKA2, RKB2],
        [FWA, FWB],
        [FIA, FIB],
    )?;
    // A row whose full-pool mass is not positive and finite is the argmax point mass.
    op(b, format!("s_cmp_eq_u32 s{S_GOOD}, 0"), &[], &[s(S_GOOD)])?;
    sop(b, format!("s_cselect_b32 s{MD}, -1, 0"), &[MD], &[])?;
    sop(
        b,
        format!("s_cselect_b32 s{S_NOUT}, 1, s{S_NOUT}"),
        &[S_NOUT],
        &[S_NOUT],
    )?;
    alu(b, "v_cmp_eq_u32_e64", O::M(MA), &[O::N(0), O::V(LANE)])?;
    alu(b, "s_and_b32", O::M(MA), &[O::M(MA), O::M(MD)])?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(VTMP),
        &[O::N(0), O::K("1.0"), O::M(MA)],
    )?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(FWA),
        &[O::V(FWA), O::V(VTMP), O::M(MD)],
    )?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(FWB),
        &[O::V(FWB), O::N(0), O::M(MD)],
    )?;
    alu(b, "v_mov_b32_e32", O::V(VTMP), &[O::S(S_PID)])?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(VTMP2),
        &[O::N(0), O::V(VTMP), O::M(MA)],
    )?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(FIA),
        &[O::V(FIA), O::V(VTMP2), O::M(MD)],
    )?;
    alu(
        b,
        "v_cndmask_b32_e64",
        O::V(FIB),
        &[O::V(FIB), O::N(0), O::M(MD)],
    )?;
    // Stores: ids and probabilities fill all 64 slots; lane 0 writes the count.
    bstore(b, FIA, LANE4, SRD_I, 0)?;
    bstore(b, FIB, LANE4, SRD_I, 128)?;
    bstore(b, FWA, LANE4, SRD_P, 0)?;
    bstore(b, FWB, LANE4, SRD_P, 128)?;
    op(b, "s_mov_b32 exec_lo, 1", &[], &[])?;
    alu(b, "v_mov_b32_e32", O::V(VTMP), &[O::S(S_NOUT)])?;
    bstore(b, VTMP, VZ, SRD_C, 0)?;
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
    b.wait_all()
}

/// Mnemonics whose exact counts the shape contract pins.
const COUNTED: [&str; 12] = [
    "buffer_load_b32",
    "buffer_store_b32",
    "s_load_b256",
    "ds_store_b64",
    "ds_load_b128",
    "s_barrier",
    "s_barrier_signal",
    "s_barrier_wait",
    "v_readlane_b32",
    "v_writelane_b32",
    "v_permlanex16_b32",
    "s_endpgm",
];

/// The shape contract of the emitted kernel: pinned instruction counts, no scratch,
/// flat/global, atomics or LDS read-modify-write, the register ceilings, wave32,
/// the static LDS and no dynamic LDS.
pub fn contract(arch: Arch, e: &Emitted) -> serde_json::Value {
    let mut counts = serde_json::Map::new();
    for name in COUNTED {
        let n = e
            .s_text
            .lines()
            .filter(|l| l.trim().split_whitespace().next() == Some(name))
            .count();
        if n > 0 {
            counts.insert(name.into(), n.into());
        }
    }
    let dpp = e
        .s_text
        .lines()
        .filter(|l| l.trim().split_whitespace().next() == Some("v_mov_b32_dpp"))
        .count();
    counts.insert("v_mov_b32_dpp".into(), dpp.into());
    serde_json::json!({
        "symbol": symbol(arch), "counts": counts,
        "forbidden": ["scratch_*", "flat_*", "global_*", "buffer_atomic_*", "ds_add_*", "ds_sub_*", "ds_rsub_*", "ds_inc_*", "ds_dec_*", "ds_min_*", "ds_max_*",
                      "ds_and_*", "ds_or_*", "ds_xor_*", "ds_mskor_*", "ds_cmpst*", "ds_cmpstore*", "ds_wrxchg*", "ds_append", "ds_consume", "ds_gws_*",
                      "ds_bpermute*", "ds_permute*", "s_setpc*", "s_swappc*"],
        "vgpr_max": 224, "sgpr_max": 104,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0,
        "group_segment_fixed_bytes": STATIC_LDS,
    })
}
