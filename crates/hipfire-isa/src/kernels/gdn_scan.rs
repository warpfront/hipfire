//! gfx1201 Qwen3.5 DeltaNet chunk scan (`gdn_chunk_scan`), emitted with the
//! checked ISA builder.
//!
//! Semantics are those of `kernels/src/gdn_chunk_scan.gfx1201.hip` (its exact
//! gfx1201 `GS_G12_PIPE` loop): one 512-thread CU-mode workgroup per value head,
//! 64-row chunks, the same LDS image (score / state[2] / d[2] / ctrl), the same
//! WMMA operands and accumulation order, the same FP16/FP32 roundings (hipcc's
//! `expf` and IEEE-division DAGs are reproduced op for op) and the same Q8 /
//! scale / EF epilogue. Differences are scheduling only:
//!  * loop-carried loads: chunk c+1's V tile, G/beta rows, gl and first Q/K
//!    fragments are issued in chunk c's St-update phase (the builder loop
//!    accepts pending loads at the back-edge that equal those at entry);
//!  * the d decay is applied while forming the St-update B fragments (the same
//!    `v_fma_mix*_f16` per element), removing one barrier phase;
//!  * the score exponents are evaluated inside the QK loop, and only by the
//!    waves whose score tile is on or below the causal diagonal;
//!  * operands are double-buffered and waited by the builder ledger instead of
//!    hipcc's per-use `s_wait_*`, and dependent VALU instructions carry
//!    `s_delay_alu` hints.
//! LDS is dynamic: launch with `GROUP_BYTES` of dynamic LDS, grid [1,48,1],
//! block 512.
use crate::{Arch, Builder, Emitted, KernelSpec, KernargLayout, RegPlan};
use crate::insn::{Instruction, MemoryClass, Sop};
use crate::kernels::iu4_gemm::lit;
use crate::ledger::Counter;
use crate::lds::Transition;
use crate::reg::{Kind, Live, RegRef};

pub const SYMBOL: &str = "gdn_chunk_scan";
/// score[64][68] u16 + state[2][64][132] u16 + d[2][64][68] u16 + ctrl[4][64] f32.
pub const GROUP_BYTES: u32 = 60928;
const SCORE: u32 = 0;
const STATE: u32 = 8704;
const DB: u32 = 42496;
const CTRL: u32 = 59904;
const SP: u32 = 264; // state row pitch (bytes)
const AP: u32 = 136; // score / d row pitch (bytes)
const STATE_HALF: u32 = 16896;
const D_HALF: u32 = 8704;

const LOOP: &str = ".Lgs_loop";
const P2: &str = ".Lgs_p2";
const P3: &str = ".Lgs_p3";
const P4: &str = ".Lgs_p4";
const P5: &str = ".Lgs_p5";
const P7: &str = ".Lgs_p7";
const TAIL: &str = ".Lgs_tail";
const EPI: &str = ".Lgs_epi";
const END: &str = ".Lgs_end";

// LDS slots (ids are declaration order).
const L_SCORE: usize = 0;
const L_STATE: usize = 1;
const L_D: usize = 2;
const L_CTRL: usize = 3;

type R = Result<(), String>;
fn v(n: u8) -> RegRef { RegRef { kind: Kind::V, base: n, len: 1 } }
fn vr(n: u8, len: u8) -> RegRef { RegRef { kind: Kind::V, base: n, len } }
fn s(n: u8) -> RegRef { RegRef { kind: Kind::S, base: n, len: 1 } }
fn sr(n: u8, len: u8) -> RegRef { RegRef { kind: Kind::S, base: n, len } }
fn op(b: &mut Builder, text: impl Into<String>, defs: &[RegRef], uses: &[RegRef]) -> R {
    b.push(Instruction::new(text, defs.to_vec(), uses.to_vec()))
}
fn rs(base: u8, len: u8) -> String { if len == 1 { format!("v{base}") } else { format!("v[{base}:{}]", base + len - 1) } }
fn off(imm: u32) -> String { if imm == 0 { String::new() } else { format!(" offset:{imm}") } }
/// global_load_b{32,64,128} with a uniform SGPR base and a 32-bit lane offset.
fn gload(b: &mut Builder, dwords: u8, dst: u8, voff: u8, sbase: u8, imm: u32) -> R {
    let w = match dwords { 1 => 32, 2 => 64, 4 => 128, _ => return Err("gload width".into()) };
    b.push(Instruction::new(format!("global_load_b{w} {}, v{voff}, s[{sbase}:{}]{}", rs(dst, dwords), sbase + 1, off(imm)),
        vec![vr(dst, dwords)], vec![v(voff), sr(sbase, 2)]).memory(MemoryClass::VmemLoad))
}
fn gstore(b: &mut Builder, dwords: u8, voff: u8, data: u8, sbase: u8, imm: u32) -> R {
    let w = match dwords { 1 => 32, 2 => 64, 4 => 128, _ => return Err("gstore width".into()) };
    b.push(Instruction::new(format!("global_store_b{w} v{voff}, {}, s[{sbase}:{}]{}", rs(data, dwords), sbase + 1, off(imm)),
        vec![], vec![v(voff), vr(data, dwords), sr(sbase, 2)]).memory(MemoryClass::VmemStore))
}
fn dload(b: &mut Builder, slot: usize, dwords: u8, dst: u8, addr: u8, imm: u32) -> R {
    let w = match dwords { 1 => 32, 2 => 64, 4 => 128, _ => return Err("dload width".into()) };
    b.ds_load(slot, Instruction::new(format!("ds_load_b{w} {}, v{addr}{}", rs(dst, dwords), off(imm)),
        vec![vr(dst, dwords)], vec![v(addr)]).memory(MemoryClass::DsLoad))
}
fn dstore(b: &mut Builder, slot: usize, dwords: u8, addr: u8, data: u8, imm: u32) -> R {
    let w = match dwords { 1 => 32, 2 => 64, 4 => 128, _ => return Err("dstore width".into()) };
    b.ds_store(slot, Instruction::new(format!("ds_store_b{w} v{addr}, {}{}", rs(data, dwords), off(imm)),
        vec![], vec![v(addr), vr(data, dwords)]).memory(MemoryClass::DsStore))
}
fn dstore16(b: &mut Builder, slot: usize, addr: u8, data: u8, high: bool, imm: u32) -> R {
    let name = if high { "ds_store_b16_d16_hi" } else { "ds_store_b16" };
    b.ds_store(slot, Instruction::new(format!("{name} v{addr}, v{data}{}", off(imm)), vec![], vec![v(addr), v(data)])
        .memory(MemoryClass::DsStore))
}
fn wait_alu(b: &mut Builder, what: &str) -> R { op(b, format!("s_wait_alu {what}"), &[], &[]) }
/// v_wmma_f32_16x16x16_f16 dst, a, b, (c | 0)
fn wmma(b: &mut Builder, dst: u8, a: u8, bb: u8, acc: bool) -> R {
    let c = if acc { rs(dst, 8) } else { "0".into() };
    let mut uses = vec![vr(a, 4), vr(bb, 4)];
    if acc { uses.push(vr(dst, 8)); }
    op(b, format!("v_wmma_f32_16x16x16_f16 {}, {}, {}, {c}", rs(dst, 8), rs(a, 4), rs(bb, 4)), &[vr(dst, 8)], &uses)
}
/// hipcc's `expf(x)` (OCML f32, range-reduced v_exp_f32 with under/overflow
/// selects), op for op. `t` needs three temporaries; `m` one mask SGPR.
fn expf(b: &mut Builder, dst: u8, x: u8, t: [u8; 3], m: u8) -> R {
    let [a, e, n] = t;
    op(b, format!("v_mul_f32_e32 v{a}, 0x3fb8aa3b, v{x}"), &[v(a)], &[v(x)])?;
    op(b, format!("v_fma_f32 v{e}, 0x3fb8aa3b, v{x}, -v{a}"), &[v(e)], &[v(x), v(a)])?;
    op(b, format!("v_rndne_f32_e32 v{n}, v{a}"), &[v(n)], &[v(a)])?;
    op(b, format!("v_fmac_f32_e32 v{e}, 0x32a5705f, v{x}"), &[v(e)], &[v(e), v(x)])?;
    op(b, format!("v_sub_f32_e32 v{a}, v{a}, v{n}"), &[v(a)], &[v(a), v(n)])?;
    op(b, format!("v_add_f32_e32 v{a}, v{a}, v{e}"), &[v(a)], &[v(a), v(e)])?;
    op(b, format!("v_cvt_i32_f32_e32 v{n}, v{n}"), &[v(n)], &[v(n)])?;
    op(b, format!("v_exp_f32_e32 v{a}, v{a}"), &[v(a)], &[v(a)])?;
    op(b, format!("v_ldexp_f32 v{a}, v{a}, v{n}"), &[v(a)], &[v(a), v(n)])?;
    op(b, format!("v_cmp_ngt_f32_e64 s{m}, 0xc2ce8ed0, v{x}"), &[s(m)], &[v(x)])?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    op(b, format!("v_cndmask_b32_e64 v{a}, 0, v{a}, s{m}"), &[v(a)], &[v(a), s(m)])?;
    op(b, format!("v_cmp_nlt_f32_e64 s{m}, 0x42b17218, v{x}"), &[s(m)], &[v(x)])?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    op(b, format!("v_cndmask_b32_e64 v{dst}, 0x7f800000, v{a}, s{m}"), &[v(dst)], &[v(a), s(m)])
}
/// hipcc's correctly rounded f32 division num/den (v_div_scale / v_rcp / fma
/// Newton steps / v_div_fmas / v_div_fixup), op for op. Uses VCC.
fn fdiv(b: &mut Builder, dst: u8, num: &str, den: &str, uses: &[RegRef], t: [u8; 4]) -> R {
    let [sc, r, q, e] = t;
    op(b, format!("v_div_scale_f32 v{sc}, null, {den}, {den}, {num}"), &[v(sc)], uses)?;
    op(b, format!("v_rcp_f32_e32 v{r}, v{sc}"), &[v(r)], &[v(sc)])?;
    op(b, format!("v_fma_f32 v{e}, -v{sc}, v{r}, 1.0"), &[v(e)], &[v(sc), v(r)])?;
    op(b, format!("v_fmac_f32_e32 v{r}, v{e}, v{r}"), &[v(r)], &[v(r), v(e)])?;
    op(b, format!("v_div_scale_f32 v{e}, vcc_lo, {num}, {den}, {num}"), &[v(e)], uses)?;
    op(b, format!("v_mul_f32_e32 v{q}, v{e}, v{r}"), &[v(q)], &[v(e), v(r)])?;
    op(b, format!("v_fma_f32 v{dst}, -v{sc}, v{q}, v{e}"), &[v(dst)], &[v(sc), v(q), v(e)])?;
    op(b, format!("v_fmac_f32_e32 v{q}, v{dst}, v{r}"), &[v(q)], &[v(q), v(dst), v(r)])?;
    op(b, format!("v_fma_f32 v{sc}, -v{sc}, v{q}, v{e}"), &[v(sc)], &[v(sc), v(q), v(e)])?;
    wait_alu(b, "depctr_va_vcc(0)")?;
    op(b, format!("v_div_fmas_f32 v{sc}, v{sc}, v{r}, v{q}"), &[v(sc)], &[v(sc), v(r), v(q)])?;
    let mut fix = vec![v(sc)];
    fix.extend_from_slice(uses);
    op(b, format!("v_div_fixup_f32 v{dst}, v{sc}, {den}, {num}"), &[v(dst)], &fix)
}
/// Pack eight f32 VGPRs (src..src+7) into four f16x2 VGPRs (dst..dst+3), RNE.
fn pack8(b: &mut Builder, dst: u8, src: u8) -> R {
    for i in 0..4u8 {
        op(b, format!("v_cvt_f16_f32_e64 v{}.l, v{}", dst + i, src + 2 * i), &[v(dst + i)], &[v(src + 2 * i)])?;
        op(b, format!("v_cvt_f16_f32_e64 v{}.h, v{} op_sel:[0,1]", dst + i, src + 2 * i + 1), &[v(dst + i)], &[v(dst + i), v(src + 2 * i + 1)])?;
    }
    Ok(())
}

// ------------------------------------------------------------------ registers
// VGPR map. Whole: v0 tid, v1 lo, v2 hi, v3 8*hi, v4..v27 per-lane constant LDS
// addresses and indices, v28..v35 per-chunk offsets, v36/37 state offsets,
// v40..71 St (four 16x16 f32 tiles), v160..183 phase temporaries (v168..175
// carry the K tile from P3 to P5 and the next chunk's first Q/K fragments,
// v168..179, from P7 to P2).
// Phase aliases: v72..87 O (P2..P5), v88..103 U (P2..P3) / Dacc (P4),
// v104..111 P (P2), v112..143 fragments (P2) / temporaries (other phases),
// v144..159 next chunk's V tile and G/beta rows (P7..P1) / score operands (P2)
// / A fragments (P3..P4).
const TID: u8 = 0; const LO: u8 = 1; const HI: u8 = 2; const H8: u8 = 3;
const VST: u8 = 4; const VSTG: u8 = 5; const VVST: u8 = 6; const VCTL: u8 = 7; const VOGE: u8 = 8;
const VCJ: u8 = 9; const VSC: u8 = 10; const VDIFF: u8 = 11; const VCT: u8 = 12; const VDZ: u8 = 13;
const VZS: u8 = 14; const VDF: u8 = 15; const VSF: u8 = 16; const VOUT: u8 = 17; const VKS: u8 = 18;
const VDV: u8 = 19; const VC1: u8 = 20; const VKT: u8 = 21; const VMTL: u8 = 22; const VPNL: u8 = 23;
const VHXR: u8 = 24; const VT7: u8 = 25; const VT4: u8 = 26; const VT15: u8 = 27;
const VVLD0: u8 = 28; const VVLD1: u8 = 29; const VKM: u8 = 30; const VKN: u8 = 31; const VA: u8 = 32;
const VKLD0: u8 = 33; const VKLD1: u8 = 34; const VEC: u8 = 35; const VSOFF: u8 = 36; const VSOFF2: u8 = 37;
const VEP: u8 = 38; const VEL: u8 = 39;
const ST: u8 = 40; const O: u8 = 72; const U: u8 = 88; const PT: u8 = 104; const F: u8 = 112; const G: u8 = 144;
const T: u8 = 160; const KT: u8 = T + 8;
// SGPR map.
const SQ: u8 = 8; const SK: u8 = 10; const SV: u8 = 12; const SA: u8 = 14; const SG: u8 = 16; const SB: u8 = 18;
const SSQ: u8 = 20; const SSC: u8 = 22; const SEF: u8 = 24; const SOUT: u8 = 26; const SROW0: u8 = 28; const ST_: u8 = 29;
const HEAD: u8 = 30; const KEYH: u8 = 31; const WAVE: u8 = 32; const HALF: u8 = 33; const SWAVE: u8 = 34; const MT: u8 = 35;
const NTB: u8 = 36; const PN: u8 = 37; const KT0: u8 = 38; const VT0: u8 = 39;
const NCH: u8 = 40; const CHUNK: u8 = 41; const LOCAL0: u8 = 42; const PARENT0: u8 = 43; const ROWS: u8 = 44;
const LAST: u8 = 45; const DOP: u8 = 46; const STMP: u8 = 47;
const KB: u8 = 48; const QB: u8 = 50; const VB: u8 = 52; const AB: u8 = 54; const OB: u8 = 56; const GL: u8 = 58;
// PNL: K row tile of the P fragment loads, pn for waves with P and mt otherwise
// (their K fragment rows, so the unused loads hit lines this wave just read).
const STMP2: u8 = 59; const S64A: u8 = 60; const PNL: u8 = 62; const GB: u8 = 64; const BB: u8 = 66;
const MASK: u8 = 68; // s68..s79
const PSEL: u8 = 80; // s80,s81
const NEG128: u8 = 82; const EXS: u8 = 83; const SQH: u8 = 84; const EFH: u8 = 86; const SCH: u8 = 88;
// Chunk n = min(c+1, nch-1) of the loop-carried loads: index, base row, last row.
const NCHUNK: u8 = 90; const NPARENT0: u8 = 91; const NLAST: u8 = 92;
// Loop-carried load targets: kk=0 K/Q/KN fragments, V tile, G and beta rows.
const PF: u8 = T + 8; const VPF: u8 = G; const GPF: u8 = G + 8; const BPF: u8 = G + 9;

fn plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(184, 96)?;
    let w = || Live::Whole;
    let between = |a: &str, z: &str| Live::Between(a.into(), z.into());
    for n in 0..40u8 { p.v::<1>("lane_constant", n, w())?; }
    for i in 0..4u8 { p.v::<8>("st", ST + 8 * i, w())?; }
    for i in 0..3u8 { p.v::<8>("phase_temp", T + 8 * i, w())?; }
    // Prologue and epilogue temporaries (v72..v111 and v72..v143).
    for i in 0..5u8 { p.v::<8>("prologue_temp", O + 8 * i, between("entry", LOOP))?; }
    for i in 0..9u8 { p.v::<8>("epilogue_temp", O + 8 * i, between(EPI, END))?; }
    for i in 0..2u8 { p.v::<8>("o", O + 8 * i, between(P2, P7))?; }
    for i in 0..2u8 { p.v::<8>("u", U + 8 * i, between(P2, P4))?; p.v::<8>("dacc", U + 8 * i, between(P4, P5))?; }
    p.v::<8>("p", PT, between(P2, P3))?;
    for i in 0..4u8 {
        p.v::<8>("p1_temp", F + 8 * i, between(LOOP, P2))?;
        p.v::<8>("frag", F + 8 * i, between(P2, P3))?;
        p.v::<8>("p3_temp", F + 8 * i, between(P3, P4))?;
        p.v::<8>("p4_temp", F + 8 * i, between(P4, P5))?;
        p.v::<8>("p5_temp", F + 8 * i, between(P5, P7))?;
        p.v::<8>("p7_temp", F + 8 * i, between(P7, TAIL))?;
    }
    // v144..v159: next chunk's V tile and G/beta rows (P7 -> P1, across the
    // back-edge), score operands in P2, A fragments in P3..P4.
    for i in 0..2u8 {
        p.v::<8>("v_prefetch", G + 8 * i, between("entry", P2))?;
        p.v::<8>("p2_score", G + 8 * i, between(P2, P3))?;
        p.v::<8>("a_frag", G + 8 * i, between(P3, P5))?;
        p.v::<8>("v_prefetch", G + 8 * i, between(P7, EPI))?;
    }
    for i in [0u8, 2, 4, 6] { p.s::<2>("sgpr_pair", i, w())?; }
    p.s::<8>("kernargs0", 8, w())?; p.s::<8>("kernargs1", 16, w())?; p.s::<4>("kernargs2", 24, w())?;
    p.s::<2>("row0_t", 28, w())?;
    for n in 30..48u8 { p.s::<1>("scalar", n, w())?; }
    for n in (48..58u8).step_by(2) { p.s::<2>("chunk_base", n, w())?; }
    p.s::<1>("gl", GL, w())?; p.s::<1>("scalar", STMP2, w())?;
    for n in (60..68u8).step_by(2) { p.s::<2>("pair", n, w())?; }
    for n in 68..84u8 { p.s::<1>("mask", n, w())?; }
    for n in (84..90u8).step_by(2) { p.s::<2>("head_base", n, w())?; }
    for n in [NCHUNK, NPARENT0, NLAST] { p.s::<1>("next_chunk", n, w())?; }
    Ok(p)
}

fn declare_lds(b: &mut Builder) -> R {
    for (id, (name, base, len)) in [("score", SCORE, STATE - SCORE), ("state", STATE, DB - STATE), ("d", DB, CTRL - DB),
        ("ctrl", CTRL, GROUP_BYTES - CTRL)].into_iter().enumerate() {
        if b.lds.add(name, base, len)? != id { return Err("LDS slot order".into()) }
    }
    Ok(())
}

fn sop(b: &mut Builder, text: impl Into<String>, defs: &[u8], uses: &[u8]) -> R {
    op(b, text, &defs.iter().map(|&n| s(n)).collect::<Vec<_>>(), &uses.iter().map(|&n| s(n)).collect::<Vec<_>>())
}
/// base64 = base + (lo32 of `off`), zero-extended.
fn sadd64(b: &mut Builder, dst: u8, base: u8, off: u8) -> R {
    sop(b, format!("s_mov_b32 s{}, 0", S64A + 1), &[S64A + 1], &[])?;
    sop(b, format!("s_mov_b32 s{S64A}, s{off}"), &[S64A], &[off])?;
    op(b, format!("s_add_nc_u64 s[{dst}:{}], s[{base}:{}], s[{S64A}:{}]", dst + 1, base + 1, S64A + 1),
        &[sr(dst, 2)], &[sr(base, 2), sr(S64A, 2)])
}

// ------------------------------------------------------------------ prologue
fn prologue(b: &mut Builder) -> R {
    b.push(Instruction::new("s_load_b512 s[8:23], s[0:1], 0x0", vec![sr(8, 8), sr(16, 8)], vec![sr(0, 2)]).memory(MemoryClass::SmemLoad))?;
    b.push(Instruction::new("s_load_b128 s[24:27], s[0:1], 0x40", vec![sr(24, 4)], vec![sr(0, 2)]).memory(MemoryClass::SmemLoad))?;
    b.push(Instruction::new("s_load_b64 s[28:29], s[0:1], 0x50", vec![sr(28, 2)], vec![sr(0, 2)]).memory(MemoryClass::SmemLoad))?;
    b.wait(Counter::Km, 0)?;
    // T in 1..=512, x==0, head < 48, else exit.
    sop(b, format!("s_add_co_i32 s2, s{ST_}, -1"), &[2], &[ST_])?;
    sop(b, "s_cmp_gt_u32 s2, 0x1ff", &[], &[2])?;
    op(b, format!("s_cbranch_scc1 {END}"), &[], &[])?;
    op(b, "s_cmp_lg_u32 ttmp9, 0", &[], &[])?;
    op(b, format!("s_cbranch_scc1 {END}"), &[], &[])?;
    op(b, "s_cmp_gt_u32 ttmp7, 47", &[], &[])?;
    op(b, format!("s_cbranch_scc1 {END}"), &[], &[])?;
    op(b, format!("s_mov_b32 s{HEAD}, ttmp7"), &[s(HEAD)], &[])?;
    sop(b, format!("s_mul_hi_u32 s{KEYH}, s{HEAD}, 0xaaaaaaab"), &[KEYH], &[HEAD])?;
    sop(b, format!("s_lshr_b32 s{KEYH}, s{KEYH}, 1"), &[KEYH], &[KEYH])?;
    // Wave-uniform indices.
    op(b, format!("v_lshrrev_b32_e32 v{H8}, 5, v{TID}"), &[v(H8)], &[v(TID)])?;
    op(b, format!("v_readfirstlane_b32 s{WAVE}, v{H8}"), &[s(WAVE)], &[v(H8)])?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    sop(b, format!("s_lshr_b32 s{HALF}, s{WAVE}, 3"), &[HALF], &[WAVE])?;
    sop(b, format!("s_and_b32 s{SWAVE}, s{WAVE}, 7"), &[SWAVE], &[WAVE])?;
    sop(b, format!("s_lshr_b32 s{MT}, s{SWAVE}, 1"), &[MT], &[SWAVE])?;
    sop(b, format!("s_and_b32 s{NTB}, s{SWAVE}, 1"), &[NTB], &[SWAVE])?;
    sop(b, format!("s_lshl_b32 s{NTB}, s{NTB}, 1"), &[NTB], &[NTB])?;
    sop(b, format!("s_add_co_i32 s{PN}, s{NTB}, s{HALF}"), &[PN], &[NTB, HALF])?;
    sop(b, format!("s_lshl_b32 s{KT0}, s{MT}, 1"), &[KT0], &[MT])?;
    sop(b, format!("s_mov_b32 s{VT0}, s{NTB}"), &[VT0], &[NTB])?;
    sop(b, format!("s_cmp_le_u32 s{PN}, s{MT}"), &[], &[PN, MT])?;
    sop(b, format!("s_cselect_b32 s{DOP}, 1, 0"), &[DOP], &[])?;
    sop(b, format!("s_cselect_b32 s{PNL}, s{PN}, s{MT}"), &[PNL], &[PN, MT])?;
    sop(b, format!("s_add_co_i32 s{NCH}, s{ST_}, 63"), &[NCH], &[ST_])?;
    sop(b, format!("s_lshr_b32 s{NCH}, s{NCH}, 6"), &[NCH], &[NCH])?;
    // SGPR constants used by lane address math.
    sop(b, format!("s_mul_i32 s2, s{HALF}, {}", lit(STATE_HALF)), &[2], &[HALF])?;
    sop(b, format!("s_add_co_i32 s2, s2, {}", lit(STATE)), &[2], &[2])?;               // state + half
    sop(b, format!("s_mul_i32 s3, s{HALF}, {}", lit(D_HALF)), &[3], &[HALF])?;
    sop(b, format!("s_add_co_i32 s3, s3, {}", lit(DB)), &[3], &[3])?;                   // d + half
    sop(b, format!("s_lshl_b32 s4, s{MT}, 4"), &[4], &[MT])?;                     // mt*16
    sop(b, format!("s_lshl_b32 s5, s{PN}, 4"), &[5], &[PN])?;                     // pn*16
    sop(b, format!("s_lshl_b32 s6, s{NTB}, 4"), &[6], &[NTB])?;                   // ntb*16 (= vt0*16)
    sop(b, format!("s_lshl_b32 s7, s{KT0}, 4"), &[7], &[KT0])?;                   // kt0*16
    sop(b, format!("s_mov_b32 s{NEG128}, 0xc3000000"), &[NEG128], &[])?;
    // Head-scoped state bases.
    sop(b, format!("s_lshl_b32 s{STMP}, s{HEAD}, 14"), &[STMP], &[HEAD])?;
    sadd64(b, SQH, SSQ, STMP)?;
    sop(b, format!("s_lshl_b32 s{STMP}, s{HEAD}, 15"), &[STMP], &[HEAD])?;
    sadd64(b, EFH, SEF, STMP)?;
    sop(b, format!("s_lshl_b32 s{STMP}, s{HEAD}, 9"), &[STMP], &[HEAD])?;
    sadd64(b, SCH, SSC, STMP)?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    // Lane constants.
    let t0 = O; let t1 = O + 1; let t2 = O + 2;
    op(b, format!("v_and_b32_e32 v{LO}, 15, v{TID}"), &[v(LO)], &[v(TID)])?;
    op(b, format!("v_bfe_u32 v{HI}, v{TID}, 4, 1"), &[v(HI)], &[v(TID)])?;
    op(b, format!("v_lshlrev_b32_e32 v{H8}, 3, v{HI}"), &[v(H8)], &[v(HI)])?;
    op(b, format!("v_lshl_add_u32 v{VMTL}, s{MT}, 4, v{LO}"), &[v(VMTL)], &[s(MT), v(LO)])?;
    op(b, format!("v_lshl_add_u32 v{VPNL}, s{PNL}, 4, v{LO}"), &[v(VPNL)], &[s(PNL), v(LO)])?;
    op(b, format!("v_add_nc_u32_e32 v{t0}, s6, v{LO}"), &[v(t0)], &[s(6), v(LO)])?;            // ntb*16+lo
    // vST = state+half + (ntb*16+lo)*264 + 8hi
    op(b, format!("v_mad_u32_u24 v{VST}, v{t0}, {}, v{H8}", lit(SP)), &[v(VST)], &[v(t0), v(H8)])?;
    op(b, format!("v_add_nc_u32_e32 v{VST}, s2, v{VST}"), &[v(VST)], &[s(2), v(VST)])?;
    // vSTG = state+half + (vt0*16+lo)*264 + (kt0*16+8hi)*2
    op(b, format!("v_lshlrev_b32_e32 v{t1}, 4, v{HI}"), &[v(t1)], &[v(HI)])?;
    op(b, format!("v_lshl_add_u32 v{t1}, s7, 1, v{t1}"), &[v(t1)], &[s(7), v(t1)])?;
    op(b, format!("v_mad_u32_u24 v{VSTG}, v{t0}, {}, v{t1}", lit(SP)), &[v(VSTG)], &[v(t0), v(t1)])?;
    op(b, format!("v_add_nc_u32_e32 v{VSTG}, s2, v{VSTG}"), &[v(VSTG)], &[s(2), v(VSTG)])?;
    // vHXR=(tid&255)>>3, vT7=(tid&7)*16, vVST = d+half + (tid&7)*8*136 + vHXR*2
    op(b, format!("v_bfe_u32 v{VHXR}, v{TID}, 3, 5"), &[v(VHXR)], &[v(TID)])?;
    op(b, format!("v_and_b32_e32 v{t1}, 7, v{TID}"), &[v(t1)], &[v(TID)])?;
    op(b, format!("v_lshlrev_b32_e32 v{VT7}, 4, v{t1}"), &[v(VT7)], &[v(t1)])?;
    op(b, format!("v_mul_u32_u24_e32 v{t1}, {}, v{t1}", lit(8 * AP)), &[v(t1)], &[v(t1)])?;
    op(b, format!("v_lshl_add_u32 v{VVST}, v{VHXR}, 1, v{t1}"), &[v(VVST)], &[v(VHXR), v(t1)])?;
    op(b, format!("v_add_nc_u32_e32 v{VVST}, s3, v{VVST}"), &[v(VVST)], &[s(3), v(VVST)])?;
    op(b, format!("v_lshl_add_u32 v{VCTL}, v{TID}, 2, {}", lit(CTRL)), &[v(VCTL)], &[v(TID)])?;
    op(b, format!("v_lshl_add_u32 v{VOGE}, v{VMTL}, 2, {}", lit(CTRL)), &[v(VOGE)], &[v(VMTL)])?;
    op(b, format!("v_add_nc_u32_e32 v{t1}, s5, v{H8}"), &[v(t1)], &[s(5), v(H8)])?;             // pn*16+8hi
    op(b, format!("v_lshl_add_u32 v{VCJ}, v{t1}, 2, {}", lit(CTRL + 512)), &[v(VCJ)], &[v(t1)])?;
    op(b, format!("v_lshlrev_b32_e32 v{t2}, 1, v{t1}"), &[v(t2)], &[v(t1)])?;
    op(b, format!("v_mad_u32_u24 v{VSC}, v{VMTL}, {}, v{t2}", lit(AP)), &[v(VSC)], &[v(VMTL), v(t2)])?;
    op(b, format!("v_sub_nc_u32_e32 v{VDIFF}, v{VMTL}, v{t1}"), &[v(VDIFF)], &[v(VMTL), v(t1)])?;
    op(b, format!("v_add_nc_u32_e32 v{t1}, s4, v{H8}"), &[v(t1)], &[s(4), v(H8)])?;             // mt*16+8hi
    op(b, format!("v_lshl_add_u32 v{VCT}, v{t1}, 2, {}", lit(CTRL)), &[v(VCT)], &[v(t1)])?;
    op(b, format!("v_lshlrev_b32_e32 v{t2}, 1, v{t1}"), &[v(t2)], &[v(t1)])?;
    op(b, format!("v_mad_u32_u24 v{VDZ}, v{t0}, {}, v{t2}", lit(AP)), &[v(VDZ)], &[v(t0), v(t2)])?;
    op(b, format!("v_add_nc_u32_e32 v{VDZ}, s3, v{VDZ}"), &[v(VDZ)], &[s(3), v(VDZ)])?;
    op(b, format!("v_mad_u32_u24 v{VZS}, v{t0}, {}, v{t2}", lit(SP)), &[v(VZS)], &[v(t0), v(t2)])?;
    op(b, format!("v_add_nc_u32_e32 v{VZS}, s2, v{VZS}"), &[v(VZS)], &[s(2), v(VZS)])?;
    op(b, format!("v_mad_u32_u24 v{VDF}, v{t0}, {}, v{H8}", lit(AP)), &[v(VDF)], &[v(t0), v(H8)])?;
    op(b, format!("v_add_nc_u32_e32 v{VDF}, s3, v{VDF}"), &[v(VDF)], &[s(3), v(VDF)])?;
    op(b, format!("v_mad_u32_u24 v{VSF}, v{VMTL}, {}, v{H8}", lit(AP)), &[v(VSF)], &[v(VMTL), v(H8)])?;
    op(b, format!("v_add_nc_u32_e32 v{t1}, s6, v{H8}"), &[v(t1)], &[s(6), v(H8)])?;             // ntb*16+8hi
    op(b, format!("v_lshlrev_b32_e32 v{t1}, 2, v{t1}"), &[v(t1)], &[v(t1)])?;
    op(b, format!("v_mad_u32_u24 v{VOUT}, v{VMTL}, {}, v{t1}", lit(48 * 128 * 4)), &[v(VOUT)], &[v(VMTL), v(t1)])?;
    op(b, format!("v_lshrrev_b32_e32 v{VT4}, 4, v{TID}"), &[v(VT4)], &[v(TID)])?;
    op(b, format!("v_lshlrev_b32_e32 v{VT15}, 4, v{LO}"), &[v(VT15)], &[v(LO)])?;
    op(b, format!("v_mad_u32_u24 v{VKS}, v{VT4}, {}, v{VT15}", lit(SP)), &[v(VKS)], &[v(VT4), v(VT15)])?;
    op(b, format!("v_add_nc_u32_e32 v{VKS}, {}, v{VKS}", lit(STATE)), &[v(VKS)], &[v(VKS)])?;
    op(b, format!("v_mad_u32_u24 v{VDV}, v{t0}, {}, v{H8}", lit(AP)), &[v(VDV)], &[v(t0), v(H8)])?;     // vt0 == ntb
    op(b, format!("v_add_nc_u32_e32 v{VDV}, s3, v{VDV}"), &[v(VDV)], &[s(3), v(VDV)])?;
    op(b, format!("v_lshl_add_u32 v{VC1}, v{HI}, 4, {}", lit(CTRL + 256)), &[v(VC1)], &[v(HI)])?;
    op(b, format!("v_add_nc_u32_e32 v{t1}, s7, v{LO}"), &[v(t1)], &[s(7), v(LO)])?;             // kt0*16+lo
    op(b, format!("v_lshlrev_b32_e32 v{t1}, 1, v{t1}"), &[v(t1)], &[v(t1)])?;
    op(b, format!("v_mad_u32_u24 v{VKT}, v{HI}, {}, v{t1}", lit(4 * SP)), &[v(VKT)], &[v(HI), v(t1)])?;
    op(b, format!("v_add_nc_u32_e32 v{VKT}, {}, v{VKT}", lit(STATE)), &[v(VKT)], &[v(VKT)])?;
    // State offsets: ((half*64 + vt0*16 + lo)*128 + kt0*16 + 8hi) elements.
    sop(b, format!("s_lshl_b32 s{STMP}, s{HALF}, 6"), &[STMP], &[HALF])?;
    sop(b, format!("s_add_co_i32 s{STMP}, s{STMP}, s6"), &[STMP], &[STMP, 6])?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    op(b, format!("v_add_nc_u32_e32 v{t1}, s{STMP}, v{LO}"), &[v(t1)], &[s(STMP), v(LO)])?;     // vv - no, row index
    op(b, format!("v_lshlrev_b32_e32 v{t2}, 2, v{t1}"), &[v(t2)], &[v(t1)])?;                   // scale byte offset
    op(b, format!("v_lshlrev_b32_e32 v{t1}, 7, v{t1}"), &[v(t1)], &[v(t1)])?;
    op(b, format!("v_add3_u32 v{VSOFF}, v{t1}, s7, v{H8}"), &[v(VSOFF)], &[v(t1), s(7), v(H8)])?;
    op(b, format!("v_lshlrev_b32_e32 v{VSOFF2}, 1, v{VSOFF}"), &[v(VSOFF2)], &[v(VSOFF)])?;
    // Epilogue LDS scratch addresses (float view of state[half]).
    op(b, format!("v_lshl_add_u32 v{VEL}, v{t0}, 2, s2"), &[v(VEL)], &[v(t0), s(2)])?;          // + (vt0*16+lo)*4
    sop(b, format!("s_lshl_b32 s{STMP}, s{MT}, 8"), &[STMP], &[MT])?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    op(b, format!("v_add_nc_u32_e32 v{VEP}, s{STMP}, v{VEL}"), &[v(VEP)], &[s(STMP), v(VEL)])?;
    // Initial state: St = (float)q8 * scale + (float)EF, as hipcc.
    let (q8, ef, scv, tq) = (O + 8, O + 16, O + 32, O + 3);
    for vi in 0..2u8 { gload(b, 1, scv + vi, t2, SCH, u32::from(vi) * 64)?; }
    for ki in 0..2u8 { for vi in 0..2u8 {
        let t = ki * 2 + vi;
        let o = u32::from(vi) * 2048 + u32::from(ki) * 16;
        gload(b, 2, q8 + 2 * t, VSOFF, SQH, o)?;
        gload(b, 4, ef + 4 * t, VSOFF2, EFH, 2 * o)?;
    }}
    // Chunk 0's loop-carried loads (see the chunk loop), in flight with the
    // state loads; t0/t1 are dead.
    next_p1_setup(b, true, t0)?;
    prefetch_p1(b, ".Lgs_pf_ctrl0", t1)?;
    next_qk_setup(b, t0)?;
    qk_global(b, PF, 0)?;
    for ki in 0..2u8 { for vi in 0..2u8 {
        let t = ki * 2 + vi;
        for e in 0..8u8 {
            let qw = q8 + 2 * t + e / 4;
            op(b, format!("v_bfe_i32 v{tq}, v{qw}, {}, 8", 8 * (e % 4)), &[v(tq)], &[v(qw)])?;
            op(b, format!("v_cvt_f32_i32_e32 v{tq}, v{tq}"), &[v(tq)], &[v(tq)])?;
            op(b, format!("v_mul_f32_e32 v{tq}, v{}, v{tq}", scv + vi), &[v(tq)], &[v(scv + vi), v(tq)])?;
            let ew = ef + 4 * t + e / 2;
            let sel = if e % 2 == 1 { " op_sel:[1,0,0]" } else { "" };
            let dst = ST + 8 * t + e;
            op(b, format!("v_fma_mix_f32 v{dst}, v{ew}, 1.0, v{tq}{sel} op_sel_hi:[1,1,0]"), &[v(dst)], &[v(ew), v(tq)])?;
        }
    }}
    sop(b, format!("s_mov_b32 s{CHUNK}, 0"), &[CHUNK], &[])
}

// ------------------------------------------------------------------ chunk loop
// Loop-carried loads: chunk n = min(c+1, nch-1) has its V tile, G/beta rows,
// gl and kk=0 Q/K fragments issued in P7 of chunk c, so P1 and the
// first QK step find them in flight or landed. The prologue issues the same
// loads for chunk 0; the last chunk re-reads itself and nothing consumes it.
fn clamp(b: &mut Builder, dst: u8, idx: u8, add: Option<u32>, mul: u32, plus: u8, last: u8, t: u8) -> R {
    let src = if let Some(a) = add {
        op(b, format!("v_add_nc_u32_e32 v{t}, {a}, v{idx}"), &[v(t)], &[v(idx)])?; t
    } else { idx };
    op(b, format!("v_min_u32_e64 v{t}, s{last}, v{src}"), &[v(t)], &[s(last), v(src)])?;
    op(b, format!("v_mad_u32_u24 v{dst}, v{t}, {}, v{plus}", lit(mul)), &[v(dst)], &[v(t), v(plus)])
}

/// n (0 before the loop), its base row and last row, and the V / G / beta / gl
/// addresses. Rows are clamped to last, as the hipcc loop.
fn next_p1_setup(b: &mut Builder, first: bool, t: u8) -> R {
    if first {
        sop(b, format!("s_mov_b32 s{NCHUNK}, 0"), &[NCHUNK], &[])?;
    } else {
        sop(b, format!("s_add_co_i32 s{NCHUNK}, s{CHUNK}, 1"), &[NCHUNK], &[CHUNK])?;
        sop(b, format!("s_add_co_i32 s{STMP}, s{NCH}, -1"), &[STMP], &[NCH])?;
        sop(b, format!("s_min_u32 s{NCHUNK}, s{NCHUNK}, s{STMP}"), &[NCHUNK], &[NCHUNK, STMP])?;
    }
    sop(b, format!("s_lshl_b32 s{STMP}, s{NCHUNK}, 6"), &[STMP], &[NCHUNK])?;
    sop(b, format!("s_add_co_i32 s{NPARENT0}, s{SROW0}, s{STMP}"), &[NPARENT0], &[SROW0, STMP])?;
    sop(b, format!("s_sub_co_i32 s{NLAST}, s{ST_}, s{STMP}"), &[NLAST], &[ST_, STMP])?;
    sop(b, format!("s_min_u32 s{NLAST}, s{NLAST}, 64"), &[NLAST], &[NLAST])?;
    sop(b, format!("s_add_co_i32 s{NLAST}, s{NLAST}, -1"), &[NLAST], &[NLAST])?;
    // vb = v + parent0*12288 + head*256 + half*128
    sop(b, format!("s_mul_i32 s{STMP}, s{NPARENT0}, 0x3000"), &[STMP], &[NPARENT0])?;
    sop(b, format!("s_lshl_b32 s{STMP2}, s{HEAD}, 8"), &[STMP2], &[HEAD])?;
    sop(b, format!("s_add_co_i32 s{STMP}, s{STMP}, s{STMP2}"), &[STMP], &[STMP, STMP2])?;
    sop(b, format!("s_lshl_b32 s{STMP2}, s{HALF}, 7"), &[STMP2], &[HALF])?;
    sop(b, format!("s_add_co_i32 s{STMP}, s{STMP}, s{STMP2}"), &[STMP], &[STMP, STMP2])?;
    sadd64(b, VB, SV, STMP)?;
    // Gb/betab = G/beta + parent0*192 + head*4; gl = G[(parent0+last)*48+head]
    sop(b, format!("s_mul_i32 s{STMP}, s{NPARENT0}, 0xc0"), &[STMP], &[NPARENT0])?;
    sop(b, format!("s_lshl_b32 s{STMP2}, s{HEAD}, 2"), &[STMP2], &[HEAD])?;
    sop(b, format!("s_add_co_i32 s{STMP}, s{STMP}, s{STMP2}"), &[STMP], &[STMP, STMP2])?;
    sadd64(b, GB, SG, STMP)?;
    sadd64(b, BB, SB, STMP)?;
    sop(b, format!("s_mul_i32 s{STMP2}, s{NLAST}, 0xc0"), &[STMP2], &[NLAST])?;
    sop(b, format!("s_add_co_i32 s{STMP2}, s{STMP}, s{STMP2}"), &[STMP2], &[STMP, STMP2])?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    clamp(b, VVLD0, VHXR, None, 12288, VT7, NLAST, t)?;
    clamp(b, VVLD1, VHXR, Some(32), 12288, VT7, NLAST, t)
}

/// ctrl waves 0 and 1 (tid < 64) load gl. Every wave loads G/beta rows (only
/// ctrl waves use them; the others read row last) and its V tile slice, so the
/// load counter holds the same operations in every wave and each counted wait
/// is exact. `a` is a VGPR temporary.
fn prefetch_p1(b: &mut Builder, skip: &str, a: u8) -> R {
    sop(b, format!("s_cmp_ge_u32 s{WAVE}, 2"), &[], &[WAVE])?;
    op(b, format!("s_cbranch_scc1 {skip}"), &[], &[])?;
    b.push(Instruction::new(format!("s_load_b32 s{GL}, s[{SG}:{}], s{STMP2} offset:0x0", SG + 1), vec![s(GL)], vec![sr(SG, 2), s(STMP2)])
        .memory(MemoryClass::SmemLoad))?;
    b.label(skip)?;
    op(b, format!("v_min_u32_e64 v{a}, s{NLAST}, v{TID}"), &[v(a)], &[s(NLAST), v(TID)])?;
    op(b, format!("v_mul_u32_u24_e32 v{a}, 0xc0, v{a}"), &[v(a)], &[v(a)])?;
    gload(b, 1, GPF, a, GB, 0)?;
    gload(b, 1, BPF, a, BB, 0)?;
    gload(b, 4, VPF, VVLD0, VB, 0)?;
    gload(b, 4, VPF + 4, VVLD1, VB, 0)
}

/// K/Q bases of chunk n and the fragment row offsets (K rows mt, P rows pn).
fn next_qk_setup(b: &mut Builder, t: u8) -> R {
    // kb/qb = k/q + parent0*4096 + keyhead*256
    sop(b, format!("s_mul_i32 s{STMP}, s{NPARENT0}, 0x1000"), &[STMP], &[NPARENT0])?;
    sop(b, format!("s_lshl_b32 s{STMP2}, s{KEYH}, 8"), &[STMP2], &[KEYH])?;
    sop(b, format!("s_add_co_i32 s{STMP}, s{STMP}, s{STMP2}"), &[STMP], &[STMP, STMP2])?;
    sadd64(b, KB, SK, STMP)?;
    sadd64(b, QB, SQ, STMP)?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    clamp(b, VKM, VMTL, None, 4096, H8, NLAST, t)?;
    clamp(b, VKN, VPNL, None, 4096, H8, NLAST, t)
}

// Fragment register sets for the QK loop: kk=0 global fragments arrive in PF,
// later steps alternate FB/FA; LDS state fragments alternate STA/STB.
const FA: u8 = F; const FB: u8 = F + 12; const STA: u8 = F + 24; const STB: u8 = T;
fn qk_global(b: &mut Builder, set: u8, kk: u32) -> R {
    for (base, voff, sb) in [(set, VKM, KB), (set + 4, VKM, QB), (set + 8, VKN, KB)] {
        gload(b, 2, base, voff, sb, kk * 32)?;
        gload(b, 2, base + 2, voff, sb, kk * 32 + 16)?;
    }
    Ok(())
}
fn qk_lds(b: &mut Builder, st: u8, kk: u32) -> R {
    for j in 0..2u32 {
        dload(b, L_STATE, 2, st + 4 * j as u8, VST, j * 16 * SP + kk * 32)?;
        dload(b, L_STATE, 2, st + 4 * j as u8 + 2, VST, j * 16 * SP + kk * 32 + 16)?;
    }
    Ok(())
}

/// Chunk c's own offsets: rows, last, A and out bases, A and K-tile lanes.
fn chunk_setup(b: &mut Builder) -> R {
    sop(b, format!("s_lshl_b32 s{LOCAL0}, s{CHUNK}, 6"), &[LOCAL0], &[CHUNK])?;
    sop(b, format!("s_add_co_i32 s{PARENT0}, s{SROW0}, s{LOCAL0}"), &[PARENT0], &[SROW0, LOCAL0])?;
    sop(b, format!("s_sub_co_i32 s{ROWS}, s{ST_}, s{LOCAL0}"), &[ROWS], &[ST_, LOCAL0])?;
    sop(b, format!("s_min_u32 s{ROWS}, s{ROWS}, 64"), &[ROWS], &[ROWS])?;
    sop(b, format!("s_add_co_i32 s{LAST}, s{ROWS}, -1"), &[LAST], &[ROWS])?;
    // ab = A + local0*6144 + head*128
    sop(b, format!("s_mul_i32 s{STMP}, s{LOCAL0}, 0x1800"), &[STMP], &[LOCAL0])?;
    sop(b, format!("s_lshl_b32 s{STMP2}, s{HEAD}, 7"), &[STMP2], &[HEAD])?;
    sop(b, format!("s_add_co_i32 s{STMP}, s{STMP}, s{STMP2}"), &[STMP], &[STMP, STMP2])?;
    sadd64(b, AB, SA, STMP)?;
    // ob = out + parent0*24576 + head*512 + half*256
    sop(b, format!("s_mul_i32 s{STMP}, s{PARENT0}, 0x6000"), &[STMP], &[PARENT0])?;
    sop(b, format!("s_lshl_b32 s{STMP2}, s{HEAD}, 9"), &[STMP2], &[HEAD])?;
    sop(b, format!("s_add_co_i32 s{STMP}, s{STMP}, s{STMP2}"), &[STMP], &[STMP, STMP2])?;
    sop(b, format!("s_lshl_b32 s{STMP2}, s{HALF}, 8"), &[STMP2], &[HALF])?;
    sop(b, format!("s_add_co_i32 s{STMP}, s{STMP}, s{STMP2}"), &[STMP], &[STMP, STMP2])?;
    sadd64(b, OB, SOUT, STMP)?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    clamp(b, VA, VMTL, None, 6144, H8, LAST, F)?;
    clamp(b, VKLD0, VT4, None, 4096, VT15, LAST, F)?;
    clamp(b, VKLD1, VT4, Some(32), 4096, VT15, LAST, F)
}

fn phase1(b: &mut Builder) -> R {
    chunk_setup(b)?;
    // stage_state: St -> f16 state[half][vv][key].
    for ki in 0..2u8 { for vi in 0..2u8 {
        let t = ki * 2 + vi;
        let pk = F + 16 + 4 * t % 16;
        pack8(b, pk, ST + 8 * t)?;
        let o = u32::from(vi) * 16 * SP + u32::from(ki) * 32;
        dstore(b, L_STATE, 2, VSTG, pk, o)?;
        dstore(b, L_STATE, 2, VSTG, pk + 2, o + 8)?;
    }}
    // ctrl (waves 0 and 1) from the prefetched gl and G/beta rows.
    let (g, bt, x, e0, e1) = (GPF, BPF, F + 2, F + 3, F + 4);
    sop(b, format!("s_cmp_ge_u32 s{WAVE}, 2"), &[], &[WAVE])?;
    op(b, "s_cbranch_scc1 .Lgs_ctrl_done", &[], &[])?;
    op(b, format!("v_sub_f32_e32 v{x}, s{GL}, v{g}"), &[v(x)], &[s(GL), v(g)])?;
    expf(b, e0, g, [F + 6, F + 7, F + 8], MASK)?;
    expf(b, e1, x, [F + 6, F + 7, F + 8], MASK)?;
    op(b, format!("v_cmp_gt_u32_e64 s{}, s{ROWS}, v{TID}", MASK + 1), &[s(MASK + 1)], &[s(ROWS), v(TID)])?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    for (k, src) in [(0u32, e0), (1, e1), (2, g), (3, bt)] {
        let c = F + 9 + k as u8;
        op(b, format!("v_cndmask_b32_e64 v{c}, 0, v{src}, s{}", MASK + 1), &[v(c)], &[v(src), s(MASK + 1)])?;
        dstore(b, L_CTRL, 1, VCTL, c, k * 256)?;
    }
    b.label(".Lgs_ctrl_done")?;
    // V tile -> d[half][c][r] (transposed u16).
    for it in 0..2u8 { for e in 0..4u8 {
        let w = VPF + 4 * it + e;
        let base = u32::from(it) * 64 + u32::from(2 * e) * AP;
        dstore16(b, L_D, VVST, w, false, base)?;
        dstore16(b, L_D, VVST, w, true, base + AP)?;
    }}
    b.barrier(&[Transition::Ready(L_STATE), Transition::Ready(L_D), Transition::Ready(L_CTRL)])
}

fn phase2(b: &mut Builder) -> R {
    b.label(P2)?;
    // Score operands: ci = ctrl2[i] (i = mt*16+lo), cj[e] = ctrl2[pn*16+8hi+e].
    // Each QK step also evaluates one score exponent under its WMMAs;
    // em[e] = mask ? expf(ci - cj[e]) : +0 replaces cj[e].
    let (cj, ci, x, ex, pm) = (G, G + 8, G + 9, G + 13, G + 14);
    let temps = [G + 10, G + 11, G + 12];
    dload(b, L_CTRL, 1, ci, VOGE, 512)?;
    dload(b, L_CTRL, 4, cj, VCJ, 0)?;
    dload(b, L_CTRL, 4, cj + 4, VCJ, 16)?;
    op(b, format!("v_cmp_gt_i32_e64 s{MASK}, s{ROWS}, v{VMTL}"), &[s(MASK)], &[s(ROWS), v(VMTL)])?;
    for e in 0..8u8 { op(b, format!("v_cmp_le_i32_e64 s{}, {e}, v{VDIFF}", MASK + 1 + e), &[s(MASK + 1 + e)], &[v(VDIFF)])?; }
    wait_alu(b, "depctr_va_sdst(0)")?;
    for e in 0..8u8 { sop(b, format!("s_and_b32 s{0}, s{0}, s{MASK}", MASK + 1 + e), &[MASK + 1 + e], &[MASK + 1 + e, MASK])?; }
    wait_alu(b, "depctr_sa_sdst(0)")?;
    qk_lds(b, STA, 0)?;
    let global = |kk: u32| if kk == 0 { PF } else if kk % 2 == 1 { FB } else { FA };
    let state = |kk: u32| if kk % 2 == 0 { STA } else { STB };
    for kk in 0..8u32 {
        let (cur, st) = (global(kk), state(kk));
        if kk < 7 {
            qk_global(b, global(kk + 1), kk + 1)?;
            qk_lds(b, state(kk + 1), kk + 1)?;
        }
        let acc = kk > 0;
        for j in 0..2u8 {
            wmma(b, U + 8 * j, cur, st + 4 * j, acc)?;
            wmma(b, O + 8 * j, st + 4 * j, cur + 4, acc)?;
        }
        // Waves whose P tile lies above the causal diagonal (DOP = 0) store a
        // +0 score tile: they skip the P product and the exponents.
        sop(b, format!("s_cmp_eq_u32 s{DOP}, 0"), &[], &[DOP])?;
        op(b, format!("s_cbranch_scc1 .Lgs_np{kk}"), &[], &[])?;
        wmma(b, PT, cur + 8, cur + 4, acc)?;
        let e = kk as u8;
        op(b, format!("v_sub_f32_e32 v{x}, v{ci}, v{}", cj + e), &[v(x)], &[v(ci), v(cj + e)])?;
        expf(b, ex, x, temps, MASK + 9)?;
        op(b, format!("v_cndmask_b32_e64 v{}, 0, v{ex}, s{}", cj + e, MASK + 1 + e), &[v(cj + e)], &[v(ex), s(MASK + 1 + e)])?;
        b.label(&format!(".Lgs_np{kk}"))?;
    }
    // O *= exp(g_i); score tile (K rows pn, Q rows mt).
    let oge = T;
    dload(b, L_CTRL, 1, oge, VOGE, 0)?;
    for j in 0..16u8 { op(b, format!("v_mul_f32_e32 v{}, v{oge}, v{}", O + j, O + j), &[v(O + j)], &[v(oge), v(O + j)])?; }
    // hipcc forms each score half as f16(fma(Pt, e, +0)) (`v_fma_mix{lo,hi}_f16`,
    // one rounding) and stores +0 where masked; Pt is selected too, so a masked
    // lane gives fma(+0, +0, +0) = +0 whatever the product holds.
    let score = T + 16;
    sop(b, format!("s_cmp_eq_u32 s{DOP}, 0"), &[], &[DOP])?;
    op(b, "s_cbranch_scc1 .Lgs_score_zero", &[], &[])?;
    for e in 0..8u8 {
        op(b, format!("v_cndmask_b32_e64 v{pm}, 0, v{}, s{}", PT + e, MASK + 1 + e), &[v(pm)], &[v(PT + e), s(MASK + 1 + e)])?;
        let dst = score + e / 2;
        let name = if e % 2 == 1 { "v_fma_mixhi_f16" } else { "v_fma_mixlo_f16" };
        op(b, format!("{name} v{dst}, v{pm}, v{}, 0", cj + e), &[v(dst)], &[v(dst), v(pm), v(cj + e)])?;
    }
    op(b, "s_branch .Lgs_score_store", &[], &[])?;
    b.label(".Lgs_score_zero")?;
    for i in 0..4u8 { op(b, format!("v_mov_b32_e32 v{}, 0", score + i), &[v(score + i)], &[])?; }
    b.label(".Lgs_score_store")?;
    dstore(b, L_SCORE, 2, VSC, score, 0)?;
    dstore(b, L_SCORE, 2, VSC, score + 2, 8)?;
    b.barrier(&[Transition::Retire(L_STATE), Transition::Ready(L_SCORE)])
}

fn phase3(b: &mut Builder) -> R {
    b.label(P3)?;
    // A fragments for P4 and the K tile for P5's LDS store, in flight under
    // the z phase.
    for kk in 0..4u32 {
        gload(b, 2, G + 4 * kk as u8, VA, AB, kk * 32)?;
        gload(b, 2, G + 4 * kk as u8 + 2, VA, AB, kk * 32 + 16)?;
    }
    gload(b, 4, KT, VKLD0, KB, 0)?;
    gload(b, 4, KT + 4, VKLD1, KB, 0)?;
    let (c0, c3, dz, tt, w, zo) = (F, F + 8, F + 16, F + 20, F + 21, F + 24);
    dload(b, L_CTRL, 4, c0, VCT, 0)?;
    dload(b, L_CTRL, 4, c0 + 4, VCT, 16)?;
    dload(b, L_CTRL, 4, c3, VCT, 768)?;
    dload(b, L_CTRL, 4, c3 + 4, VCT, 784)?;
    for j in 0..2u8 {
        dload(b, L_D, 2, dz, VDZ, u32::from(j) * 16 * AP)?;
        dload(b, L_D, 2, dz + 2, VDZ, u32::from(j) * 16 * AP + 8)?;
        for e in 0..8u8 {
            op(b, format!("v_mul_f32_e32 v{tt}, v{}, v{}", U + 8 * j + e, c0 + e), &[v(tt)], &[v(U + 8 * j + e), v(c0 + e)])?;
            let dw = dz + e / 2;
            let sel = if e % 2 == 1 { "op_sel:[0,0,1] " } else { "" };
            op(b, format!("v_fma_mix_f32 v{w}, v{tt}, -1.0, v{dw} {sel}op_sel_hi:[0,1,1]"), &[v(w)], &[v(tt), v(dw)])?;
            let dst = zo + e / 2;
            let name = if e % 2 == 1 { "v_fma_mixhi_f16" } else { "v_fma_mixlo_f16" };
            op(b, format!("{name} v{dst}, v{}, v{w}, 0", c3 + e), &[v(dst)], &[v(dst), v(c3 + e), v(w)])?;
        }
        dstore(b, L_STATE, 2, VZS, zo, u32::from(j) * 16 * SP)?;
        dstore(b, L_STATE, 2, VZS, zo + 2, u32::from(j) * 16 * SP + 8)?;
    }
    b.barrier(&[Transition::Ready(L_STATE), Transition::Retire(L_D)])
}

fn phase4(b: &mut Builder) -> R {
    b.label(P4)?;
    let sets = [F, F + 8];
    let load = |b: &mut Builder, kk: u32| -> R {
        let st = sets[(kk % 2) as usize];
        for j in 0..2u32 {
            dload(b, L_STATE, 2, st + 4 * j as u8, VST, j * 16 * SP + kk * 32)?;
            dload(b, L_STATE, 2, st + 4 * j as u8 + 2, VST, j * 16 * SP + kk * 32 + 16)?;
        }
        Ok(())
    };
    load(b, 0)?;
    for kk in 0..4u32 {
        if kk < 3 { load(b, kk + 1)?; }
        let st = sets[(kk % 2) as usize];
        for j in 0..2u8 { wmma(b, U + 8 * j, G + 4 * kk as u8, st + 4 * j, kk > 0)?; }
    }
    for j in 0..2u8 {
        let pk = F + 16 + 4 * j;
        pack8(b, pk, U + 8 * j)?;
        dstore(b, L_D, 2, VDZ, pk, u32::from(j) * 16 * AP)?;
        dstore(b, L_D, 2, VDZ, pk + 2, u32::from(j) * 16 * AP + 8)?;
    }
    b.barrier(&[Transition::Retire(L_STATE), Transition::Ready(L_D)])
}

fn phase5(b: &mut Builder) -> R {
    b.label(P5)?;
    // ec = expf(gl) for the St update; gl is ctrl[2][last] (the loaded bits).
    sop(b, format!("s_lshl_b32 s{STMP}, s{LAST}, 2"), &[STMP], &[LAST])?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    op(b, format!("v_mov_b32_e32 v{}, s{STMP}", T), &[v(T)], &[s(STMP)])?;
    dload(b, L_CTRL, 1, T + 1, T, CTRL + 512)?;
    expf(b, VEC, T + 1, [T + 2, T + 3, T + 4], MASK)?;
    let sets = [F, F + 12];
    let load = |b: &mut Builder, kk: u32| -> R {
        let base = sets[(kk % 2) as usize];
        for j in 0..2u32 {
            dload(b, L_D, 2, base + 4 * j as u8, VDF, j * 16 * AP + kk * 32)?;
            dload(b, L_D, 2, base + 4 * j as u8 + 2, VDF, j * 16 * AP + kk * 32 + 16)?;
        }
        dload(b, L_SCORE, 2, base + 8, VSF, kk * 32)?;
        dload(b, L_SCORE, 2, base + 10, VSF, kk * 32 + 16)
    };
    load(b, 0)?;
    for kk in 0..4u32 {
        if kk < 3 { load(b, kk + 1)?; }
        let base = sets[(kk % 2) as usize];
        for j in 0..2u8 { wmma(b, O + 8 * j, base + 4 * j, base + 8, true)?; }
    }
    // Output rows tok < rows.
    op(b, format!("v_cmp_gt_i32_e64 s{MASK}, s{ROWS}, v{VMTL}"), &[s(MASK)], &[s(ROWS), v(VMTL)])?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    op(b, format!("s_and_saveexec_b32 s{EXS}, s{MASK}"), &[s(EXS)], &[s(MASK)])?;
    for j in 0..2u8 {
        gstore(b, 4, VOUT, O + 8 * j, OB, u32::from(j) * 64)?;
        gstore(b, 4, VOUT, O + 8 * j + 4, OB, u32::from(j) * 64 + 16)?;
    }
    op(b, format!("s_or_b32 exec_lo, exec_lo, s{EXS}"), &[], &[s(EXS)])?;
    // K tile into the dead z region of state[] (row-major, pitch 264).
    for it in 0..2u8 {
        dstore(b, L_STATE, 2, VKS, KT + 4 * it, u32::from(it) * 32 * SP)?;
        dstore(b, L_STATE, 2, VKS, KT + 4 * it + 2, u32::from(it) * 32 * SP + 8)?;
    }
    b.barrier(&[Transition::Retire(L_SCORE), Transition::Ready(L_STATE)])
}

fn phase7(b: &mut Builder) -> R {
    b.label(P7)?;
    // Chunk n's loop-carried loads: V tile and G/beta/gl into v144..v153,
    // kk=0 Q/K fragments into PF (the K tile is already in LDS).
    next_p1_setup(b, false, T)?;
    prefetch_p1(b, ".Lgs_pf_ctrl", T + 1)?;
    next_qk_setup(b, T)?;
    qk_global(b, PF, 0)?;
    for t in 0..32u8 { op(b, format!("v_mul_f32_e32 v{0}, v{0}, v{VEC}", ST + t), &[v(ST + t)], &[v(ST + t), v(VEC)])?; }
    let (c1, raw, dvd, kt) = (F, F + 8, F + 16, F + 24);
    for kk in 0..4u32 {
        dload(b, L_CTRL, 4, c1, VC1, kk * 64)?;
        dload(b, L_CTRL, 4, c1 + 4, VC1, kk * 64 + 32)?;
        for vi in 0..2u32 {
            dload(b, L_D, 2, raw + 4 * vi as u8, VDV, vi * 16 * AP + kk * 32)?;
            dload(b, L_D, 2, raw + 4 * vi as u8 + 2, VDV, vi * 16 * AP + kk * 32 + 16)?;
        }
        // K^T fragments: u16 loads of the even and odd k rows, packed as the
        // f16x2 fragment words (no two loads in flight write one VGPR).
        let odd = T;
        for ki in 0..2u32 { for q in 0..8u32 {
            let row = if q < 4 { q } else { 4 + q };     // k offsets 0..3, 8..11
            let i = 4 * ki as u8 + (q / 2) as u8;
            let dst = if q % 2 == 0 { kt + i } else { odd + i };
            let o = kk * 16 * SP + row * SP + ki * 32;
            b.ds_load(L_STATE, Instruction::new(format!("ds_load_u16 v{dst}, v{VKT}{}", off(o)), vec![v(dst)], vec![v(VKT)])
                .memory(MemoryClass::DsLoad))?;
        }}
        for i in 0..8u8 {
            op(b, format!("v_lshl_or_b32 v{0}, v{1}, 16, v{0}", kt + i, odd + i), &[v(kt + i)], &[v(kt + i), v(odd + i)])?;
        }
        // Decayed d: f16(ctrl1[t] * d[vv][t]), the per-element op of the hipcc decay.
        for vi in 0..2u8 { for q in 0..8u8 {
            let src = raw + 4 * vi + q / 2;
            let dst = dvd + 4 * vi + q / 2;
            let (name, sel) = if q % 2 == 1 { ("v_fma_mixhi_f16", "op_sel:[0,1,0] ") } else { ("v_fma_mixlo_f16", "") };
            op(b, format!("{name} v{dst}, v{}, v{src}, 0 {sel}op_sel_hi:[0,1,0]", c1 + q), &[v(dst)], &[v(dst), v(c1 + q), v(src)])?;
        }}
        for ki in 0..2u8 { for vi in 0..2u8 {
            wmma(b, ST + 8 * (ki * 2 + vi), kt + 4 * ki, dvd + 4 * vi, true)?;
        }}
    }
    b.barrier(&[Transition::Retire(L_D), Transition::Retire(L_STATE), Transition::Retire(L_CTRL)])?;
    b.label(TAIL)?;
    sop(b, format!("s_add_co_i32 s{CHUNK}, s{CHUNK}, 1"), &[CHUNK], &[CHUNK])?;
    // Out stores retire here; the prefetch loads stay in flight (the loop's
    // entry ledger holds the same ones, issued before the loop).
    b.wait(Counter::Store, 0)?;
    sop(b, format!("s_cmp_lt_u32 s{CHUNK}, s{NCH}"), &[], &[CHUNK, NCH])?;
    op(b, format!("s_cbranch_scc1 {LOOP}"), &[], &[])
}

// ------------------------------------------------------------------ epilogue
fn epilogue(b: &mut Builder) -> R {
    b.label(EPI)?;
    let (m, p, mx, a) = (O, O + 2, O + 4, O + 8);
    sop(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[PSEL], &[])?;
    sop(b, format!("s_mov_b32 s{}, 0xfedcba98", PSEL + 1), &[PSEL + 1], &[])?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    for vi in 0..2u8 {
        let xs: Vec<u8> = (0..2u8).flat_map(|ki| (0..8u8).map(move |e| ST + 8 * (ki * 2 + vi) + e)).collect();
        let mv = m + vi;
        op(b, format!("v_max3_num_f32 v{mv}, |v{}|, 0, |v{}|", xs[0], xs[1]), &[v(mv)], &[v(xs[0]), v(xs[1])])?;
        for pr in xs[2..].chunks(2) {
            op(b, format!("v_max3_num_f32 v{mv}, v{mv}, |v{}|, |v{}|", pr[0], pr[1]), &[v(mv)], &[v(mv), v(pr[0]), v(pr[1])])?;
        }
        let pv = p + vi;
        op(b, format!("v_permlanex16_b32 v{pv}, v{mv}, s{PSEL}, s{}", PSEL + 1), &[v(pv)], &[v(mv), s(PSEL), s(PSEL + 1)])?;
        op(b, format!("v_max_num_f32_e32 v{pv}, v{pv}, v{pv}"), &[v(pv)], &[v(pv)])?;
        op(b, format!("v_max_num_f32_e32 v{mv}, v{mv}, v{mv}"), &[v(mv)], &[v(mv)])?;
        op(b, format!("v_max_num_f32_e32 v{mv}, v{mv}, v{pv}"), &[v(mv)], &[v(mv), v(pv)])?;
        dstore(b, L_STATE, 1, VEP, mv, u32::from(vi) * 64)?;
    }
    b.barrier(&[Transition::Ready(L_STATE)])?;
    for vi in 0..2u8 {
        for g in 0..4u8 { dload(b, L_STATE, 1, a + g, VEL, u32::from(g) * 256 + u32::from(vi) * 64)?; }
        let x = mx + 2 * vi;
        op(b, format!("v_max3_num_f32 v{x}, v{}, 0, v{}", a, a + 1), &[v(x)], &[v(a), v(a + 1)])?;
        op(b, format!("v_max3_num_f32 v{x}, v{x}, v{}, v{}", a + 2, a + 3), &[v(x)], &[v(x), v(a + 2), v(a + 3)])?;
        // scale = mx>0 ? mx/127 : 1; inv = mx>0 ? 127/mx : 0.
        let (scale, inv) = (O + 16 + 2 * vi, O + 17 + 2 * vi);
        let tmp = [O + 24, O + 25, O + 26, O + 27];
        fdiv(b, scale, &format!("v{x}"), "0x42fe0000", &[v(x)], tmp)?;
        fdiv(b, inv, "0x42fe0000", &format!("v{x}"), &[v(x)], tmp)?;
        let gt = MASK + vi;
        op(b, format!("v_cmp_lt_f32_e64 s{gt}, 0, v{x}"), &[s(gt)], &[v(x)])?;
        wait_alu(b, "depctr_va_sdst(0)")?;
        op(b, format!("v_cndmask_b32_e64 v{scale}, 1.0, v{scale}, s{gt}"), &[v(scale)], &[v(scale), s(gt)])?;
        op(b, format!("v_cndmask_b32_e64 v{inv}, 0, v{inv}, s{gt}"), &[v(inv)], &[v(inv), s(gt)])?;
    }
    // Scales: swave < 2 and hi == 0.
    sop(b, format!("s_cmp_ge_u32 s{SWAVE}, 2"), &[], &[SWAVE])?;
    op(b, "s_cbranch_scc1 .Lgs_sc_done", &[], &[])?;
    op(b, format!("v_cmp_eq_u32_e64 s{}, 0, v{HI}", MASK + 2), &[s(MASK + 2)], &[v(HI)])?;
    // scale byte offset = (half*64 + vt0*16 + lo)*4 = VSOFF/128*4 recomputed.
    op(b, format!("v_lshrrev_b32_e32 v{}, 5, v{VSOFF}", O + 28), &[v(O + 28)], &[v(VSOFF)])?;
    op(b, format!("v_and_b32_e32 v{0}, -4, v{0}", O + 28), &[v(O + 28)], &[v(O + 28)])?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    op(b, format!("s_and_saveexec_b32 s{EXS}, s{}", MASK + 2), &[s(EXS)], &[s(MASK + 2)])?;
    for vi in 0..2u8 { gstore(b, 1, O + 28, O + 16 + 2 * vi, SCH, u32::from(vi) * 64)?; }
    op(b, format!("s_or_b32 exec_lo, exec_lo, s{EXS}"), &[], &[s(EXS)])?;
    b.label(".Lgs_sc_done")?;
    // Quantize: q = med3(rne(St*inv), -128, 127), EF = f16(St - q*scale).
    // Each tile's Q8 bytes and EF words get their own registers, so no store
    // source is redefined while its store is in flight.
    let (qi, qf, pr, df) = (O + 32, O + 40, O + 48, O + 56);
    for ki in 0..2u8 { for vi in 0..2u8 {
        let t = ki * 2 + vi;
        let bytes = O + 64 + 2 * t;
        let efp = [O + 12, O + 20, T, T + 4][t as usize];
        let (scale, inv) = (O + 16 + 2 * vi, O + 17 + 2 * vi);
        for e in 0..8u8 {
            let st = ST + 8 * t + e;
            op(b, format!("v_mul_f32_e32 v{}, v{inv}, v{st}", qf + e), &[v(qf + e)], &[v(inv), v(st)])?;
            op(b, format!("v_rndne_f32_e32 v{0}, v{0}", qf + e), &[v(qf + e)], &[v(qf + e)])?;
            op(b, format!("v_med3_num_f32 v{0}, v{0}, s{NEG128}, 0x42fe0000", qf + e), &[v(qf + e)], &[v(qf + e), s(NEG128)])?;
            op(b, format!("v_cvt_i32_f32_e32 v{}, v{}", qi + e, qf + e), &[v(qi + e)], &[v(qf + e)])?;
            op(b, format!("v_mul_f32_e32 v{}, v{scale}, v{}", pr + e, qf + e), &[v(pr + e)], &[v(scale), v(qf + e)])?;
            op(b, format!("v_sub_f32_e32 v{}, v{st}, v{}", df + e, pr + e), &[v(df + e)], &[v(st), v(pr + e)])?;
        }
        for d in 0..2u8 {
            let w = bytes + d;
            op(b, format!("v_and_b32_e32 v{w}, 0xff, v{}", qi + 4 * d), &[v(w)], &[v(qi + 4 * d)])?;
            for k in 1..4u8 {
                let x = qi + 4 * d + k;
                op(b, format!("v_and_b32_e32 v{x}, 0xff, v{x}"), &[v(x)], &[v(x)])?;
                op(b, format!("v_lshl_or_b32 v{w}, v{x}, {}, v{w}", 8 * k), &[v(w)], &[v(x), v(w)])?;
            }
        }
        pack8(b, efp, df)?;
        let o = u32::from(vi) * 2048 + u32::from(ki) * 16;
        gstore(b, 2, VSOFF, bytes, SQH, o)?;
        gstore(b, 4, VSOFF2, efp, EFH, 2 * o)?;
    }}
    b.wait_all()?;
    b.label(END)?;
    b.push(Sop::End.encode(Arch::Gfx1201)?)
}

pub fn emit(arch: Arch) -> Result<Emitted, String> {
    if arch != Arch::Gfx1201 { return Err("gdn_chunk_scan builder is exact-gfx1201 only".into()) }
    let mut kernargs = KernargLayout::new(88);
    for (i, name) in ["q", "k", "v", "A", "G", "beta", "sq", "sc", "ef", "out"].iter().enumerate() {
        kernargs = kernargs.pointer(name, 8 * i as u32);
    }
    let kernargs = kernargs.hidden("row0", 80, 4, "by_value").hidden("T", 84, 4, "by_value");
    let spec = KernelSpec { kernel_id: "gdn_chunk_scan".into(), variant: "b1".into(), arch, symbol: SYMBOL.into(), kernargs,
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: true, workgroup_size: 512, group_segment_fixed_size: 0,
        wave32: true, cu_mode: true };
    let mut b = Builder::new(spec, plan()?);
    b.enable_delay_alu();
    declare_lds(&mut b)?;
    prologue(&mut b)?;
    b.loop_(LOOP, |b| { phase1(b)?; phase2(b)?; phase3(b)?; phase4(b)?; phase5(b)?; phase7(b) })?;
    epilogue(&mut b)?;
    b.finish()
}
