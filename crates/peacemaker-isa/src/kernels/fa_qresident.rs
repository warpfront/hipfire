// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201 register-resident-Q FA2-GQA prefill attention, native-fp8 E4M3 K/V, E4M3 Q codes + f32 row scales:
//! the PeaceMaker twin of the hipcc `attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_gfx1201` (and its `_range_`
//! twin), `kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip` (`HIPFIRE_FA2_QRESIDENT_V2_Q8_A3B`, design comment
//! "Register-resident-Q v2"). H = 16, Hkv = 2, D = 256 (GQA 8) is authored; the dense H24 / KV4 geometry of the same
//! body family differs in its fill (aligned 16-byte K rows, `global_load_tr_b64` V) and is refused by [`validate`].
//!
//! Contract: the OUTPUT bytes equal the hipcc object's (byte identity, NaN payloads included). The instruction
//! stream is the generator's own: every rounding step below is the hipcc DAG, the schedule is free.
//!
//! Geometry: 768 threads = 24 waves; a workgroup owns one KV head and 48 queries (384 query-head rows = 48 queries x
//! 8 heads); wave `w` owns rows `row_base + 16 w + (lane & 15)`, lane half `kg = lane >> 4` holds keys / dims
//! `8 kg + j`. The workgroup runs heaviest-first (`x` and `z` reversed). Launch grid
//! `[ceil(min(batch, 512) * 8 / 384), 2, ceil(batch / 512)]`, dynamic LDS 49 936 bytes, 68-byte kernarg
//! (`q_codes, q_scales, k_cache, v_cache, out, positions` pointers; `n_heads, n_kv_heads, head_dim, batch_size`
//! i32; `scale_attn` f32). Both exports share the body; the range export differs only by symbol (the caller
//! advances the Q planes, output and positions).
//!
//! Per 16-key sub: `S = K^T Q` as 16 fp8 WMMAs (K fragment first operand, Q fragment second, accumulated in
//! dimension-pair order), `s = (acc * qs) * sk` (qs = row scale * scale_attn; sk = f32 key scale), causal select to
//! `-inf`, online softmax with the hipcc DAG (`__expf` = `v_exp_f32(x * log2e)`, ascending serial sums, lane XOR 16
//! max / sum), P quantised to E4M3 as `cvt(p * yb_lo)` with the bracket `yb_lo / yb_hi = RN(RN(1/b)(1 -+ 2^-21))`
//! check and the IEEE-division recompute when the brackets disagree, `O += V^T P` as 16 fp8 WMMAs. Keys stream through
//! two LDS buffers of KT48 (K 12 KiB, V 12 KiB pre-transposed into fragment order, 48 f32 K scales, 48 f32 V scales),
//! one barrier per tile; the tile loop is unrolled twice so each LDS buffer is a static slot of the checked machine.
//!
//! LDS buffer `b` (`b * 24960`): K fragments `((p * 3 + sub) * 32 + lane) * 16` (kf0, kf1 of dim pair p), V fragments
//! `12288 + ((sub * 8 + p) * 32 + lane) * 16`, K scales `24576 + 4 * key`, V scales `24768 + 4 * key`.
//!
//! Group bounds (`gmax`, `gmin` over the workgroup's queries' positions) are computed redundantly by every wave
//! (loads + XOR butterfly) instead of through LDS atomics; they are workgroup-uniform values either way.

use crate::kernels::common::{lit, op, s, sop, sr, v, vr};
use crate::kernels::fa_prefill::{fdiv, gload, wait_alu};
use crate::insn::{Instruction, MemoryClass, Sop};
use crate::lds::Transition;
use crate::ledger::Counter;
use crate::plan::Access;
use crate::reg::{Live, RegRef};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// CLI kernel id.
pub const KERNEL_ID: &str = "fa_qresident_a3b_pm";
/// Module / cache key.
pub const MODULE: &str = "fa_qresident_a3b_pm_gfx1201";
/// Direct export (the PM twin of `attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_gfx1201`).
pub const SYMBOL_DIRECT: &str = "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_pm_gfx1201";
/// Range export (the PM twin of `attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_range_a3b_gfx1201`).
pub const SYMBOL_RANGE: &str = "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_range_a3b_pm_gfx1201";
/// Workgroup size (24 waves).
pub const BLOCK: u32 = 768;
/// Dynamic LDS of the launch (two KT48 buffers + the hipcc layout's 16-byte bounds tail).
pub const DYNAMIC_LDS_BYTES: u32 = 49_936;
/// Kernarg segment bytes.
pub const KERNARG_BYTES: u32 = 68;
/// Native-fp8 token row bytes (`2 x 256` codes + `2` f16 scales).
pub const ROW_BYTES: u32 = 516;

/// The authored geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaQresSpec {
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
}

impl FaQresSpec {
    /// The A3B geometry (qwen3.6-35b-a3b).
    pub const A3B: FaQresSpec = FaQresSpec {
        heads: 16,
        kv_heads: 2,
        head_dim: 256,
    };
    pub fn validate(&self) -> Result<(), String> {
        if *self != Self::A3B {
            return Err(format!(
                "fa_qresident: only heads 16, kv_heads 2, head_dim 256 are authored, got {}/{}/{} (the dense 24/4/256 \
                 body needs the aligned-row global_load_tr fill, which is not built)",
                self.heads, self.kv_heads, self.head_dim
            ));
        }
        Ok(())
    }
    pub fn gqa(&self) -> u32 {
        self.heads / self.kv_heads
    }
}

fn admit(arch: Arch, spec: &FaQresSpec) -> R {
    if arch != Arch::Gfx1201 {
        return Err("fa_qresident: built for gfx1201 only".into());
    }
    spec.validate()
}

/// Launch grid of the A3B kernel for `batch_size` rows (the hipcc launcher's formula).
pub fn grid(batch_size: u32) -> [u32; 3] {
    [
        (batch_size.min(512) * 8).div_ceil(384),
        2,
        batch_size.div_ceil(512),
    ]
}

// ------------------------------------------------------------------ constants of the arithmetic
const TK: u32 = 48;
const BUFB: u32 = 24_960;
const OFF_V: u32 = 12_288;
const OFF_SK: u32 = 24_576;
const OFF_SV: u32 = 24_768;
/// `1.0f / 448.0f` rounded to f32.
const INV448: u32 = 0x3b12_4925;
/// `-448.0f`.
const NEG448: u32 = 0xc3e0_0000;
const LOG2E: u32 = 0x3fb8_aa3b;
const NEG_INF: u32 = 0xff80_0000;
/// `0x1.fffffp-1f`, `0x1.000008p+0f`: the bracketing reciprocal factors `1 -+ 2^-21`.
const BRK_LO: u32 = 0x3f7f_fff8;
const BRK_HI: u32 = 0x3f80_0004;
/// `0x1p-64f` and its reciprocal `0x1p+64f` with the brackets applied (`RN(2^64 (1 -+ 2^-21))`).
const B_INIT: u32 = 0x1f80_0000;
const YB_LO_INIT: u32 = 0x5f7f_fff8;
const YB_HI_INIT: u32 = 0x5f80_0004;
const TWO_M64: u32 = 0x1f80_0000;

// ------------------------------------------------------------------ labels (per export: tag `d` / `r`)
struct L {
    tag: &'static str,
}
impl L {
    fn n(&self, name: &str) -> String {
        format!(".Lfq{}_{name}", self.tag)
    }
}

// ------------------------------------------------------------------ VGPR map (240)
const V_O: u8 = 0; // 16 x 8 f32 output accumulators, dc at 8 dc (block 0 aliases the entry work-item id)
const V_QF: u8 = 128; // 16 Q fragments x 2 regs
const V_SA: u8 = 160; // score accumulators / scores / exps / P
const V_KW: u8 = 168; // K fill words (4)
const V_VW: u8 = 172; // V fill words (8)
const V_SP: u8 = 180; // scale fill words: k0, v0, k1, v1 (u16 zero-extended)
const V_M: u8 = 184;
const V_L: u8 = 185;
const V_B: u8 = 186;
const V_YL: u8 = 187;
const V_YH: u8 = 188;
const V_PK: u8 = 189; // pos - 8 kg
const V_RDK: u8 = 190; // K fragment read base (lane * 16 + sub * 512)
const V_RDV: u8 = 191; // V fragment read base (lane * 16 + 12288 + sub * 4096)
const V_RSK: u8 = 192; // scale read base (24576 + 32 kg + 64 sub)
const V_WK: u8 = 193; // K commit address (buffer 0)
const V_WV: u8 = 194; // V commit address (buffer 0, V plane included)
const V_WS: u8 = 195; // scale commit address (buffer 0, K scale plane included)
const V_GK: u8 = 196; // K global lane offset (key index * 516 + head * 256 + 16 dcf)
const V_KI: u8 = 197; // K fill key index within the tile
const V_GV: u8 = 198; // V global lane offset (head * 256 + 16 vdc + 4 (lane & 3))
const V_VKB: u8 = 199; // V fill first key within the tile
const V_GS: u8 = 200; // scale global lane offset (lane * 516 + 512 + 2 head)
const V_QS: u8 = 201; // q row scale * scale_attn
const V_FR: u8 = 202; // 16 regs: fragment ring / scale staging
const V_T: u8 = 218; // 21 temporaries 218..=238
const V_LANE: u8 = 239; // lane id
const V_TID: u8 = 0;

fn t(i: u8) -> u8 {
    V_T + i
}

// ------------------------------------------------------------------ SGPR map
const S_KARG: u8 = 0;
const S_QC: u8 = 8; // q_codes (8), q_scales (10), k_cache (12), v_cache (14)
const S_QSC: u8 = 10;
const S_KB: u8 = 12;
const S_VB: u8 = 14;
const S_OUT: u8 = 16; // out (16), positions (18)
const S_POS: u8 = 18;
const S_NH: u8 = 20;
const S_NKV: u8 = 21;
const S_HD: u8 = 22;
const S_BATCH: u8 = 23;
const S_SCALE: u8 = 24;
const S_KVH: u8 = 25;
const S_QOFF: u8 = 26; // query_offset
const S_RUN: u8 = 27; // run_rows
const S_ROWB: u8 = 28; // row_base
const S_WAVE: u8 = 29;
const S_GMAX: u8 = 30;
const S_GMIN: u8 = 31;
const S_SEQ: u8 = 32; // seq_len = gmax + 1
const S_KT: u8 = 33; // ktile
const S_NSUB: u8 = 34;
const S_KTS: u8 = 35; // kt_sub
const S_R8: u8 = 36; // 8 * run_rows
const S_WLIVE: u8 = 37; // wave has a live row (0 / 1)
const S_QALL: u8 = 38; // all 384 rows live (0 / 1)
const S_PSEL: u8 = 39; // 0x76543210
const S_X8: u8 = 40; // kv_h * 256
const S_X2: u8 = 41; // 512 + kv_h * 2
const S_TA: u8 = 42; // scalar scratch 42..=47
const S_TB: u8 = 43;
const S_TC: u8 = 44;
const S_TD: u8 = 45;
const S_TE: u8 = 46;
const S_TF: u8 = 47;
const S_QOK: u8 = 48; // lane masks are pairs; the high word is never written in wave32
const S_SAVE: u8 = 50;
const S_MA: u8 = 52;
const S_MB: u8 = 54;
const S_MC: u8 = 56;
const S_MD: u8 = 58;
const S_MJ: u8 = 60; // eight pairs 60..=75
const S_SV2: u8 = 76; // second save pair for nested exec regions
const S_SV3: u8 = 78;
const S_SV4: u8 = 80;
const S_TG: u8 = 82; // wider scalar scratch
const S_TH: u8 = 83;
const S_TI: u8 = 84;
const S_TJ: u8 = 85;

fn mj(j: u8) -> u8 {
    S_MJ + 2 * j
}
fn m(x: u8) -> RegRef {
    sr(x, 2)
}

macro_rules! ins {
    ($b:expr, $t:expr; $($d:expr),* ; $($u:expr),*) => {
        op($b, $t, &[$($d),*], &[$($u),*])
    };
}

// ------------------------------------------------------------------ small emitters
fn exec_and(b: &mut Builder, save: u8, mask: u8) -> R {
    op(
        b,
        format!("s_mov_b32 s{save}, exec_lo"),
        &[m(save)],
        &[],
    )?;
    op(
        b,
        format!("s_and_b32 exec_lo, exec_lo, s{mask}"),
        &[],
        &[m(mask)],
    )
}
fn exec_restore(b: &mut Builder, save: u8) -> R {
    op(
        b,
        format!("s_or_b32 exec_lo, exec_lo, s{save}"),
        &[],
        &[m(save)],
    )
}
fn vmov(b: &mut Builder, d: u8, src: &str) -> R {
    ins!(b, format!("v_mov_b32_e32 v{d}, {src}"); v(d); )
}
fn vmov_s(b: &mut Builder, d: u8, sreg: u8) -> R {
    ins!(b, format!("v_mov_b32_e32 v{d}, s{sreg}"); v(d); s(sreg))
}
fn dload128(b: &mut Builder, slot: usize, dst: u8, addr: u8, imm: u32) -> R {
    b.ds_load(
        slot,
        Instruction::new(
            format!(
                "ds_load_b128 {}, v{addr}{}",
                vr(dst, 4),
                if imm == 0 {
                    String::new()
                } else {
                    format!(" offset:{imm}")
                }
            ),
            vec![vr(dst, 4)],
            vec![v(addr)],
        )
        .memory(MemoryClass::DsLoad),
    )
}
fn dstore(b: &mut Builder, slot: usize, kind: &str, addr: u8, data: RegRef, imm: u32) -> R {
    b.ds_store(
        slot,
        Instruction::new(
            format!(
                "ds_store_{kind} v{addr}, {data}{}",
                if imm == 0 {
                    String::new()
                } else {
                    format!(" offset:{imm}")
                }
            ),
            vec![],
            vec![v(addr), data],
        )
        .memory(MemoryClass::DsStore),
    )
}
fn smem(b: &mut Builder, text: &str, dst: RegRef) -> R {
    b.push(Instruction::new(text, vec![dst], vec![sr(S_KARG, 2)]).memory(MemoryClass::SmemLoad))
}
/// `global_store_b128 v{voff}, v[data:data+3], s[base:base+1] offset:imm` (SADDR form).
fn gstore128(b: &mut Builder, voff: u8, data: u8, sbase: u8, imm: u32) -> R {
    b.push(
        Instruction::new(
            format!(
                "global_store_b128 v{voff}, {}, s[{sbase}:{}]{}",
                vr(data, 4),
                sbase + 1,
                if imm == 0 {
                    String::new()
                } else {
                    format!(" offset:{imm}")
                }
            ),
            vec![],
            vec![v(voff), vr(data, 4), sr(sbase, 2)],
        )
        .memory(MemoryClass::VmemStore),
    )
}
/// `v_wmma_f32_16x16x16_fp8_fp8 dst(8), a(2), bb(2), acc|0`: A is the first operand.
fn wmma8(b: &mut Builder, dst: u8, a: u8, bb: u8, acc: bool) -> R {
    let c = if acc {
        format!("{}", vr(dst, 8))
    } else {
        "0".into()
    };
    let mut uses = vec![vr(a, 2), vr(bb, 2)];
    if acc {
        uses.push(vr(dst, 8));
    }
    op(
        b,
        format!(
            "v_wmma_f32_16x16x16_fp8_fp8 {}, {}, {}, {c}",
            vr(dst, 8),
            vr(a, 2),
            vr(bb, 2)
        ),
        &[vr(dst, 8)],
        &uses,
    )
}
fn xor16(b: &mut Builder, dst: u8, src: u8) -> R {
    ins!(b, format!("v_permlanex16_b32 v{dst}, v{src}, s{S_PSEL}, 0xfedcba98"); v(dst); v(src), s(S_PSEL))
}

// ------------------------------------------------------------------ plan
fn plan(l: &L) -> Result<RegPlan, String> {
    let mut p = RegPlan::new(240, 104)?;
    let w = || Live::Whole;
    let tid_done = l.n("tid_done");
    let end = l.n("end");
    let pre = || Live::Between("entry".into(), tid_done.clone());
    let post = || Live::Between(tid_done.clone(), end.clone());
    p.v::<1>("tid", V_TID, pre())?;
    p.v::<8>("o", V_O, post())?;
    for i in 1..16u8 {
        p.v::<8>("o", V_O + 8 * i, w())?;
    }
    for i in 0..4u8 {
        p.v::<8>("q_frag", V_QF + 8 * i, w())?;
    }
    p.v::<8>("score", V_SA, w())?;
    p.v::<4>("k_fill_words", V_KW, w())?;
    p.v::<8>("v_fill_words", V_VW, w())?;
    p.v::<4>("scale_fill_words", V_SP, w())?;
    for (name, r) in [
        ("m_run", V_M),
        ("l_run", V_L),
        ("b_ref", V_B),
        ("yb_lo", V_YL),
        ("yb_hi", V_YH),
        ("pos_k", V_PK),
        ("rd_k", V_RDK),
        ("rd_v", V_RDV),
        ("rd_s", V_RSK),
        ("wr_k", V_WK),
        ("wr_v", V_WV),
        ("wr_s", V_WS),
        ("g_k", V_GK),
        ("k_idx", V_KI),
        ("g_v", V_GV),
        ("v_key_base", V_VKB),
        ("g_s", V_GS),
    ] {
        p.v::<1>(name, r, w())?;
    }
    p.v::<1>("qs_tile", V_QS, w())?;
    p.v::<8>("frag_ring", V_FR, w())?;
    p.v::<8>("frag_ring", V_FR + 8, w())?;
    p.v::<8>("temp", V_T, w())?;
    p.v::<8>("temp", V_T + 8, w())?;
    p.v::<4>("temp", V_T + 16, w())?;
    p.v::<1>("temp", V_T + 20, w())?;
    p.v::<1>("lane", V_LANE, w())?;
    // SGPRs.
    p.s::<2>("kernarg_ptr", S_KARG, w())?;
    p.s::<8>("kernarg_ptrs", S_QC, w())?;
    p.s::<4>("out_pos", S_OUT, w())?;
    p.s::<4>("dims", S_NH, w())?;
    for n in (S_SCALE..=S_X2).chain(S_TA..=S_TF).chain(S_TG..=S_TJ) {
        p.s::<1>("scalar", n, w())?;
    }
    for n in [S_QOK, S_SAVE, S_MA, S_MB, S_MC, S_MD, S_SV2, S_SV3, S_SV4] {
        p.s::<2>("mask", n, w())?;
    }
    for j in 0..8u8 {
        p.s::<2>("mask", mj(j), w())?;
    }
    Ok(p)
}

fn kernel_spec(symbol: &str, variant: &str) -> KernelSpec {
    let kernargs = KernargLayout::new(KERNARG_BYTES)
        .pointer_access("q_codes", 0, Access::ReadOnly)
        .pointer_access("q_scales", 8, Access::ReadOnly)
        .pointer_access("k_cache", 16, Access::ReadOnly)
        .pointer_access("v_cache", 24, Access::ReadOnly)
        .pointer_access("out", 32, Access::WriteOnly)
        .pointer_access("positions", 40, Access::ReadOnly)
        .hidden("n_heads", 48, 4, "by_value")
        .hidden("n_kv_heads", 52, 4, "by_value")
        .hidden("head_dim", 56, 4, "by_value")
        .hidden("batch_size", 60, 4, "by_value")
        .hidden("scale_attn", 64, 4, "by_value");
    KernelSpec {
        kernel_id: KERNEL_ID.into(),
        variant: variant.into(),
        arch: Arch::Gfx1201,
        symbol: symbol.into(),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: BLOCK as u16,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    }
}

// ------------------------------------------------------------------ prologue
/// Everything before the tile loop: guards, launch decomposition (heaviest-first), lane constants, the group
/// position bounds, the resident Q fragments, the fresh online state, and the first K/V tile into buffer 0.
fn prologue(b: &mut Builder, l: &L) -> R {
    let end = l.n("end");
    let epi = l.n("epi");
    smem(
        b,
        &format!("s_load_b256 s[{S_QC}:{}], s[0:1], 0x0", S_QC + 7),
        sr(S_QC, 8),
    )?;
    smem(
        b,
        &format!("s_load_b128 s[{S_OUT}:{}], s[0:1], 0x20", S_OUT + 3),
        sr(S_OUT, 4),
    )?;
    smem(
        b,
        &format!("s_load_b128 s[{S_NH}:{}], s[0:1], 0x30", S_NH + 3),
        sr(S_NH, 4),
    )?;
    smem(
        b,
        &format!("s_load_b32 s{S_SCALE}, s[0:1], 0x40"),
        s(S_SCALE),
    )?;
    // Lane constants from the work-item id (the entry id is dead after this block).
    let (lane, ml, kg, wv) = (t(0), t(1), t(2), t(3));
    ins!(b, format!("v_and_b32_e32 v{lane}, 31, v{V_TID}"); v(lane); v(V_TID))?;
    ins!(b, format!("v_and_b32_e32 v{ml}, 15, v{V_TID}"); v(ml); v(V_TID))?;
    ins!(b, format!("v_bfe_u32 v{kg}, v{V_TID}, 4, 1"); v(kg); v(V_TID))?;
    ins!(b, format!("v_lshrrev_b32_e32 v{wv}, 5, v{V_TID}"); v(wv); v(V_TID))?;
    ins!(b, format!("v_readfirstlane_b32 s{S_WAVE}, v{wv}"); s(S_WAVE); v(wv))?;
    vmov(b, V_LANE, &format!("v{lane}"))?;
    b.label(&l.n("tid_done"))?;
    b.wait(Counter::Km, 0)?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    // Shape guards: workgroup-uniform, before any data access or barrier.
    for (reg, want) in [(S_NH, 16u32), (S_NKV, 2), (S_HD, 256)] {
        sop(b, format!("s_cmp_lg_u32 s{reg}, {}", lit(want)), &[], &[reg])?;
        op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
    }
    op(b, format!("s_and_b32 s{S_KVH}, ttmp7, 0xffff"), &[s(S_KVH)], &[])?;
    sop(b, format!("s_cmp_gt_u32 s{S_KVH}, 1"), &[], &[S_KVH])?;
    op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
    // bz = gridDim.z - 1 - z, gridDim.z = ceil(batch / 512); query_offset = batch > 512 ? 512 bz : 0.
    op(b, format!("s_lshr_b32 s{S_TA}, ttmp7, 16"), &[s(S_TA)], &[])?;
    sop(b, format!("s_add_co_i32 s{S_TB}, s{S_BATCH}, 0x1ff"), &[S_TB], &[S_BATCH])?;
    sop(b, format!("s_lshr_b32 s{S_TB}, s{S_TB}, 9"), &[S_TB], &[S_TB])?;
    sop(b, format!("s_add_co_i32 s{S_TB}, s{S_TB}, -1"), &[S_TB], &[S_TB])?;
    sop(b, format!("s_sub_co_i32 s{S_TB}, s{S_TB}, s{S_TA}"), &[S_TB], &[S_TB, S_TA])?;
    sop(b, format!("s_cmp_lt_i32 s{S_TB}, 0"), &[], &[S_TB])?;
    op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
    sop(b, format!("s_lshl_b32 s{S_QOFF}, s{S_TB}, 9"), &[S_QOFF], &[S_TB])?;
    sop(b, format!("s_cmp_gt_i32 s{S_BATCH}, 0x200"), &[], &[S_BATCH])?;
    sop(
        b,
        format!("s_cselect_b32 s{S_QOFF}, s{S_QOFF}, 0"),
        &[S_QOFF],
        &[S_QOFF],
    )?;
    sop(b, format!("s_cmp_ge_i32 s{S_QOFF}, s{S_BATCH}"), &[], &[S_QOFF, S_BATCH])?;
    op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
    sop(
        b,
        format!("s_sub_co_i32 s{S_RUN}, s{S_BATCH}, s{S_QOFF}"),
        &[S_RUN],
        &[S_BATCH, S_QOFF],
    )?;
    sop(b, format!("s_min_i32 s{S_RUN}, s{S_RUN}, 0x200"), &[S_RUN], &[S_RUN])?;
    sop(b, format!("s_lshl_b32 s{S_R8}, s{S_RUN}, 3"), &[S_R8], &[S_RUN])?;
    // bx = gridDim.x - 1 - x, gridDim.x = ceil(min(batch, 512) 8 / 384) (x/384 = (x/3) >> 7).
    sop(b, format!("s_min_i32 s{S_TC}, s{S_BATCH}, 0x200"), &[S_TC], &[S_BATCH])?;
    sop(b, format!("s_lshl_b32 s{S_TC}, s{S_TC}, 3"), &[S_TC], &[S_TC])?;
    sop(b, format!("s_add_co_i32 s{S_TC}, s{S_TC}, 0x17f"), &[S_TC], &[S_TC])?;
    sop(
        b,
        format!("s_mul_hi_u32 s{S_TD}, s{S_TC}, 0xaaaaaaab"),
        &[S_TD],
        &[S_TC],
    )?;
    sop(b, format!("s_lshr_b32 s{S_TD}, s{S_TD}, 8"), &[S_TD], &[S_TD])?;
    sop(b, format!("s_add_co_i32 s{S_TD}, s{S_TD}, -1"), &[S_TD], &[S_TD])?;
    op(b, format!("s_sub_co_i32 s{S_TD}, s{S_TD}, ttmp9"), &[s(S_TD)], &[s(S_TD)])?;
    sop(b, format!("s_cmp_lt_i32 s{S_TD}, 0"), &[], &[S_TD])?;
    op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
    sop(b, format!("s_mul_i32 s{S_ROWB}, s{S_TD}, 0x180"), &[S_ROWB], &[S_TD])?;
    sop(b, format!("s_cmp_ge_i32 s{S_ROWB}, s{S_R8}"), &[], &[S_ROWB, S_R8])?;
    op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
    // Wave-level liveness and the head terms of the K/V lane offsets.
    sop(b, format!("s_lshl_b32 s{S_TE}, s{S_WAVE}, 4"), &[S_TE], &[S_WAVE])?;
    sop(b, format!("s_add_co_i32 s{S_TE}, s{S_TE}, s{S_ROWB}"), &[S_TE], &[S_TE, S_ROWB])?;
    sop(b, format!("s_cmp_lt_i32 s{S_TE}, s{S_R8}"), &[], &[S_TE, S_R8])?;
    sop(b, format!("s_cselect_b32 s{S_WLIVE}, 1, 0"), &[S_WLIVE], &[])?;
    sop(b, format!("s_add_co_i32 s{S_TF}, s{S_ROWB}, 0x17f"), &[S_TF], &[S_ROWB])?;
    sop(b, format!("s_cmp_lt_i32 s{S_TF}, s{S_R8}"), &[], &[S_TF, S_R8])?;
    sop(b, format!("s_cselect_b32 s{S_QALL}, 1, 0"), &[S_QALL], &[])?;
    sop(b, format!("s_mov_b32 s{S_PSEL}, 0x76543210"), &[S_PSEL], &[])?;
    sop(b, format!("s_lshl_b32 s{S_X8}, s{S_KVH}, 8"), &[S_X8], &[S_KVH])?;
    sop(b, format!("s_lshl_b32 s{S_X2}, s{S_KVH}, 1"), &[S_X2], &[S_KVH])?;
    sop(b, format!("s_add_co_i32 s{S_X2}, s{S_X2}, 0x200"), &[S_X2], &[S_X2])?;
    sop(b, format!("s_lshl_b32 s{S_TG}, s{S_KVH}, 3"), &[S_TG], &[S_KVH])?;
    // Group bounds: first query of the workgroup, live queries (<= 48); positions of those queries.
    sop(b, format!("s_lshr_b32 s{S_TH}, s{S_ROWB}, 3"), &[S_TH], &[S_ROWB])?;
    sop(b, format!("s_sub_co_i32 s{S_TC}, s{S_RUN}, s{S_TH}"), &[S_TC], &[S_RUN, S_TH])?;
    sop(b, format!("s_min_i32 s{S_TC}, s{S_TC}, 48"), &[S_TC], &[S_TC])?;
    sop(b, format!("s_add_co_i32 s{S_TI}, s{S_QOFF}, s{S_TH}"), &[S_TI], &[S_QOFF, S_TH])?;
    // Row quantities of this lane: row = wave row base + ml.
    let (row, ql, h3, query, qc, head, qrow, kgx, qoff) =
        (t(4), t(5), t(6), t(7), t(8), t(9), t(10), t(11), t(12));
    ins!(b, format!("v_add_nc_u32_e32 v{row}, s{S_TE}, v{ml}"); v(row); s(S_TE), v(ml))?;
    ins!(b, format!("v_cmp_gt_i32_e64 s{S_QOK}, s{S_R8}, v{row}"); m(S_QOK); s(S_R8), v(row))?;
    ins!(b, format!("v_lshrrev_b32_e32 v{ql}, 3, v{row}"); v(ql); v(row))?;
    ins!(b, format!("v_and_b32_e32 v{h3}, 7, v{row}"); v(h3); v(row))?;
    ins!(b, format!("v_add_nc_u32_e32 v{query}, s{S_QOFF}, v{ql}"); v(query); s(S_QOFF), v(ql))?;
    ins!(b, format!("v_cndmask_b32_e64 v{qc}, 0, v{query}, s{S_QOK}"); v(qc); v(query), m(S_QOK))?;
    ins!(b, format!("v_add_nc_u32_e32 v{head}, s{S_TG}, v{h3}"); v(head); s(S_TG), v(h3))?;
    ins!(b, format!("v_lshl_add_u32 v{qrow}, v{qc}, 4, v{head}"); v(qrow); v(qc), v(head))?;
    ins!(b, format!("v_lshlrev_b32_e32 v{kgx}, 7, v{kg}"); v(kgx); v(kg))?;
    ins!(b, format!("v_lshl_add_u32 v{qoff}, v{qrow}, 8, v{kgx}"); v(qoff); v(qrow), v(kgx))?;
    // Own position (0 for a dead row); positions of the group's queries for the bounds.
    let (pos, poff) = (t(13), t(14));
    vmov(b, pos, "0")?;
    ins!(b, format!("v_lshlrev_b32_e32 v{poff}, 2, v{qc}"); v(poff); v(qc))?;
    exec_and(b, S_SAVE, S_QOK)?;
    gload(b, "b32", v(pos), poff, S_POS, 0)?;
    exec_restore(b, S_SAVE)?;
    let (offa, pa, pb) = (V_FR, V_FR + 1, V_FR + 2);
    let (lane32, wmx, wmn) = (V_FR + 3, V_FR + 4, V_FR + 5);
    ins!(b, format!("v_add_nc_u32_e32 v{offa}, s{S_TI}, v{V_LANE}"); v(offa); s(S_TI), v(V_LANE))?;
    ins!(b, format!("v_lshlrev_b32_e32 v{offa}, 2, v{offa}"); v(offa); v(offa))?;
    ins!(b, format!("v_cmp_gt_i32_e64 s{S_MA}, s{S_TC}, v{V_LANE}"); m(S_MA); s(S_TC), v(V_LANE))?;
    ins!(b, format!("v_add_nc_u32_e32 v{lane32}, 32, v{V_LANE}"); v(lane32); v(V_LANE))?;
    ins!(b, format!("v_cmp_gt_i32_e64 s{S_MB}, s{S_TC}, v{lane32}"); m(S_MB); s(S_TC), v(lane32))?;
    ins!(b, format!("v_cmp_gt_u32_e64 s{S_MC}, 16, v{V_LANE}"); m(S_MC); v(V_LANE))?;
    op(
        b,
        format!("s_and_b32 s{S_MB}, s{S_MB}, s{S_MC}"),
        &[m(S_MB)],
        &[m(S_MB), m(S_MC)],
    )?;
    vmov(b, pa, "0")?;
    vmov(b, pb, "0")?;
    exec_and(b, S_SAVE, S_MA)?;
    gload(b, "b32", v(pa), offa, S_POS, 0)?;
    exec_restore(b, S_SAVE)?;
    exec_and(b, S_SAVE, S_MB)?;
    gload(b, "b32", v(pb), offa, S_POS, 128)?;
    exec_restore(b, S_SAVE)?;
    // Resident Q codes (exact bytes produced upstream) and the row scale.
    for r in 0..32u8 {
        vmov(b, V_QF + r, "0")?;
    }
    vmov(b, V_QS, "1.0")?;
    let qso = t(15);
    ins!(b, format!("v_lshlrev_b32_e32 v{qso}, 2, v{qrow}"); v(qso); v(qrow))?;
    exec_and(b, S_SAVE, S_QOK)?;
    for mm in 0..8u8 {
        gload(b, "b128", vr(V_QF + 4 * mm, 4), qoff, S_QC, 16 * u32::from(mm))?;
    }
    gload(b, "b32", v(V_QS), qso, S_QSC, 0)?;
    exec_restore(b, S_SAVE)?;
    // Bounds: lanes outside the live queries carry the sentinels (-1 max, INT_MAX min).
    b.wait(Counter::Load, 0)?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    ins!(b, format!("v_cndmask_b32_e64 v{wmx}, -1, v{pa}, s{S_MA}"); v(wmx); v(pa), m(S_MA))?;
    ins!(b, format!("v_cndmask_b32_e64 v{wmn}, 0x7fffffff, v{pa}, s{S_MA}"); v(wmn); v(pa), m(S_MA))?;
    let (xa, xb) = (V_FR + 6, V_FR + 7);
    ins!(b, format!("v_cndmask_b32_e64 v{xa}, -1, v{pb}, s{S_MB}"); v(xa); v(pb), m(S_MB))?;
    ins!(b, format!("v_cndmask_b32_e64 v{xb}, 0x7fffffff, v{pb}, s{S_MB}"); v(xb); v(pb), m(S_MB))?;
    ins!(b, format!("v_max_i32_e32 v{wmx}, v{wmx}, v{xa}"); v(wmx); v(wmx), v(xa))?;
    ins!(b, format!("v_min_i32_e32 v{wmn}, v{wmn}, v{xb}"); v(wmn); v(wmn), v(xb))?;
    let (ta, tb) = (V_FR + 8, V_FR + 9);
    for k in [16u32, 8, 4, 2, 1] {
        for (src, tmp) in [(wmx, ta), (wmn, tb)] {
            if k == 16 {
                ins!(
                    b,
                    format!("v_permlanex16_b32 v{tmp}, v{src}, s{S_PSEL}, 0xfedcba98");
                    v(tmp); v(src), s(S_PSEL)
                )?;
            } else {
                ins!(
                    b,
                    format!("v_mov_b32_dpp v{tmp}, v{src} row_xmask:{k} row_mask:0xf bank_mask:0xf");
                    v(tmp); v(src)
                )?;
            }
        }
        ins!(b, format!("v_max_i32_e32 v{wmx}, v{wmx}, v{ta}"); v(wmx); v(wmx), v(ta))?;
        ins!(b, format!("v_min_i32_e32 v{wmn}, v{wmn}, v{tb}"); v(wmn); v(wmn), v(tb))?;
    }
    ins!(b, format!("v_readfirstlane_b32 s{S_GMAX}, v{wmx}"); s(S_GMAX); v(wmx))?;
    ins!(b, format!("v_readfirstlane_b32 s{S_GMIN}, v{wmn}"); s(S_GMIN); v(wmn))?;
    // pos - 8 kg for the diagonal masks.
    ins!(b, format!("v_lshlrev_b32_e32 v{xa}, 3, v{kg}"); v(xa); v(kg))?;
    ins!(b, format!("v_sub_nc_u32_e32 v{V_PK}, v{pos}, v{xa}"); v(V_PK); v(pos), v(xa))?;
    // q scale: scale_attn * row scale.
    ins!(b, format!("v_mul_f32_e32 v{V_QS}, s{S_SCALE}, v{V_QS}"); v(V_QS); s(S_SCALE), v(V_QS))?;
    sop(b, format!("s_add_co_i32 s{S_SEQ}, s{S_GMAX}, 1"), &[S_SEQ], &[S_GMAX])?;
    sop(b, format!("s_mov_b32 s{S_KT}, 0"), &[S_KT], &[])?;
    // Fresh online state: O = +0, m = -inf, l = +0, b = 2^-64 with its reciprocal brackets.
    for r in 0..128u8 {
        vmov(b, V_O + r, "0")?;
    }
    vmov(b, V_M, &format!("{NEG_INF:#x}"))?;
    vmov(b, V_L, "0")?;
    vmov(b, V_B, &format!("{B_INIT:#x}"))?;
    vmov(b, V_YL, &format!("{YB_LO_INIT:#x}"))?;
    vmov(b, V_YH, &format!("{YB_HI_INIT:#x}"))?;
    // No live key (gmax < 0): the hipcc loop body never runs.
    sop(b, format!("s_cmp_lt_i32 s{S_GMAX}, 0"), &[], &[S_GMAX])?;
    op(b, format!("s_cbranch_scc1 {epi}"), &[], &[])?;
    fill_constants(b, wv, lane)?;
    // Tile 0 into buffer 0, tile 48 in flight.
    sop(b, format!("s_mov_b32 s{S_TG}, 0"), &[S_TG], &[])?;
    issue(b, l, "p0", S_TG)?;
    b.wait(Counter::Load, 0)?;
    commit(b, l, "p0", S_TG, 0)?;
    sop(b, format!("s_mov_b32 s{S_TH}, {}", lit(TK)), &[S_TH], &[])?;
    issue(b, l, "p1", S_TH)?;
    b.barrier(&[Transition::Ready(0)])?;
    Ok(())
}

/// Lane-constant addresses of the fill roles and the fragment read bases.
fn fill_constants(b: &mut Builder, wv: u8, lane: u8) -> R {
    let (a, c, e, f, g) = (V_FR, V_FR + 1, V_FR + 2, V_FR + 3, V_FR + 4);
    let (kdcf, kml, ksub) = (V_FR + 6, V_FR + 7, V_FR + 8);
    // K roles: ksub = wave >> 3, kdcf = ((wave & 7) << 1) | (lane & 1), kml = lane >> 1.
    ins!(b, format!("v_lshrrev_b32_e32 v{ksub}, 3, v{wv}"); v(ksub); v(wv))?;
    ins!(b, format!("v_and_b32_e32 v{a}, 7, v{wv}"); v(a); v(wv))?;
    ins!(b, format!("v_lshlrev_b32_e32 v{a}, 1, v{a}"); v(a); v(a))?;
    ins!(b, format!("v_and_b32_e32 v{c}, 1, v{lane}"); v(c); v(lane))?;
    ins!(b, format!("v_or_b32_e32 v{kdcf}, v{a}, v{c}"); v(kdcf); v(a), v(c))?;
    ins!(b, format!("v_lshrrev_b32_e32 v{kml}, 1, v{lane}"); v(kml); v(lane))?;
    ins!(b, format!("v_lshl_add_u32 v{V_KI}, v{ksub}, 4, v{kml}"); v(V_KI); v(ksub), v(kml))?;
    ins!(b, format!("v_mad_u32_u24 v{V_GK}, v{V_KI}, {}, s{S_X8}", lit(ROW_BYTES)); v(V_GK); v(V_KI), s(S_X8))?;
    ins!(b, format!("v_lshl_add_u32 v{V_GK}, v{kdcf}, 4, v{V_GK}"); v(V_GK); v(kdcf), v(V_GK))?;
    // WK = (((kdcf >> 1) * 3 + ksub) * 32 + kml) * 16 + (kdcf & 1) * 8.
    ins!(b, format!("v_lshrrev_b32_e32 v{e}, 1, v{kdcf}"); v(e); v(kdcf))?;
    ins!(b, format!("v_mul_u32_u24_e32 v{e}, 3, v{e}"); v(e); v(e))?;
    ins!(b, format!("v_add_nc_u32_e32 v{e}, v{e}, v{ksub}"); v(e); v(e), v(ksub))?;
    ins!(b, format!("v_lshl_add_u32 v{e}, v{e}, 5, v{kml}"); v(e); v(e), v(kml))?;
    ins!(b, format!("v_lshlrev_b32_e32 v{e}, 4, v{e}"); v(e); v(e))?;
    ins!(b, format!("v_and_b32_e32 v{f}, 1, v{kdcf}"); v(f); v(kdcf))?;
    ins!(b, format!("v_lshl_add_u32 v{V_WK}, v{f}, 3, v{e}"); v(V_WK); v(f), v(e))?;
    // V roles (waves 12..23): vw = wave - 12, vsub = vw >> 2, vkg = (vw >> 1) & 1, vdc = ((vw & 1) << 3) | (lane >> 2).
    let (vw, vsub, vkg, vdc) = (V_FR + 9, V_FR + 10, V_FR + 11, V_FR + 12);
    ins!(b, format!("v_add_nc_u32_e32 v{vw}, -12, v{wv}"); v(vw); v(wv))?;
    ins!(b, format!("v_lshrrev_b32_e32 v{vsub}, 2, v{vw}"); v(vsub); v(vw))?;
    ins!(b, format!("v_bfe_u32 v{vkg}, v{vw}, 1, 1"); v(vkg); v(vw))?;
    ins!(b, format!("v_and_b32_e32 v{a}, 1, v{vw}"); v(a); v(vw))?;
    ins!(b, format!("v_lshrrev_b32_e32 v{c}, 2, v{lane}"); v(c); v(lane))?;
    ins!(b, format!("v_lshl_or_b32 v{vdc}, v{a}, 3, v{c}"); v(vdc); v(a), v(c))?;
    // VKB = vsub * 16 + vkg * 8.
    ins!(b, format!("v_lshlrev_b32_e32 v{a}, 4, v{vsub}"); v(a); v(vsub))?;
    ins!(b, format!("v_lshl_add_u32 v{V_VKB}, v{vkg}, 3, v{a}"); v(V_VKB); v(vkg), v(a))?;
    // GV = kvh * 256 + vdc * 16 + (lane & 3) * 4.
    ins!(b, format!("v_and_b32_e32 v{g}, 3, v{lane}"); v(g); v(lane))?;
    ins!(b, format!("v_lshl_add_u32 v{V_GV}, v{vdc}, 4, s{S_X8}"); v(V_GV); v(vdc), s(S_X8))?;
    ins!(b, format!("v_lshl_add_u32 v{V_GV}, v{g}, 2, v{V_GV}"); v(V_GV); v(g), v(V_GV))?;
    // WV = 12288 + ((((vsub * 8 + (vdc >> 1)) * 32 + vkg * 16 + 4 (lane & 3)) * 16) + (vdc & 1) * 8).
    ins!(b, format!("v_lshrrev_b32_e32 v{a}, 1, v{vdc}"); v(a); v(vdc))?;
    ins!(b, format!("v_lshl_add_u32 v{a}, v{vsub}, 3, v{a}"); v(a); v(vsub), v(a))?;
    ins!(b, format!("v_lshlrev_b32_e32 v{a}, 5, v{a}"); v(a); v(a))?;
    ins!(b, format!("v_lshl_add_u32 v{a}, v{vkg}, 4, v{a}"); v(a); v(vkg), v(a))?;
    ins!(b, format!("v_lshl_add_u32 v{a}, v{g}, 2, v{a}"); v(a); v(g), v(a))?;
    ins!(b, format!("v_lshlrev_b32_e32 v{a}, 4, v{a}"); v(a); v(a))?;
    ins!(b, format!("v_and_b32_e32 v{c}, 1, v{vdc}"); v(c); v(vdc))?;
    ins!(b, format!("v_lshl_add_u32 v{a}, v{c}, 3, v{a}"); v(a); v(c), v(a))?;
    ins!(b, format!("v_add_nc_u32_e32 v{V_WV}, {}, v{a}", lit(OFF_V)); v(V_WV); v(a))?;
    // Scale roles (wave 0): GS = lane * 516 + 512 + 2 head; WS = 24576 + 4 lane.
    ins!(b, format!("v_mad_u32_u24 v{V_GS}, v{lane}, {}, s{S_X2}", lit(ROW_BYTES)); v(V_GS); v(lane), s(S_X2))?;
    ins!(b, format!("v_lshl_add_u32 v{V_WS}, v{lane}, 2, {}", lit(OFF_SK)); v(V_WS); v(lane))?;
    // Fragment read bases: K lane * 16, V lane * 16 + 12288, scales 24576 + 32 kg.
    ins!(b, format!("v_lshlrev_b32_e32 v{V_RDK}, 4, v{lane}"); v(V_RDK); v(lane))?;
    ins!(b, format!("v_add_nc_u32_e32 v{V_RDV}, {}, v{V_RDK}", lit(OFF_V)); v(V_RDV); v(V_RDK))?;
    ins!(b, format!("v_bfe_u32 v{f}, v{lane}, 4, 1"); v(f); v(lane))?;
    ins!(b, format!("v_lshl_add_u32 v{V_RSK}, v{f}, 5, {}", lit(OFF_SK)); v(V_RSK); v(f))?;
    Ok(())
}

// ------------------------------------------------------------------ K/V tile fill
/// Issue the global loads of the tile whose first key is in scalar `kt`: every wave one 16-byte K chunk (four dword
/// loads of key `kt + KI`), waves 12..23 also eight V dwords (keys clamped to the last live key), wave 0 the tile's
/// K/V scale words. Dead lanes are exec-masked and their destinations pre-zeroed; nothing here depends on a branch
/// for the pending-load shape (the scalar role branches skip whole roles and their consumers alike).
fn issue(b: &mut Builder, l: &L, tag: &str, kt: u8) -> R {
    let (rem, off) = (S_TA, S_TB);
    sop(b, format!("s_sub_co_i32 s{rem}, s{S_SEQ}, s{kt}"), &[rem], &[S_SEQ, kt])?;
    sop(b, format!("s_mul_i32 s{off}, s{kt}, {}", lit(ROW_BYTES)), &[off], &[kt])?;
    for i in 0..4u8 {
        vmov(b, V_KW + i, "0")?;
    }
    ins!(b, format!("v_cmp_gt_i32_e64 s{S_MA}, s{rem}, v{V_KI}"); m(S_MA); s(rem), v(V_KI))?;
    ins!(b, format!("v_add_nc_u32_e32 v{}, s{off}, v{V_GK}", t(0)); v(t(0)); s(off), v(V_GK))?;
    exec_and(b, S_SAVE, S_MA)?;
    for i in 0..4u8 {
        gload(b, "b32", v(V_KW + i), t(0), S_KB, 4 * u32::from(i))?;
    }
    exec_restore(b, S_SAVE)?;
    // V (waves 12..23).
    let skip_v = l.n(&format!("{tag}_skipv"));
    sop(b, format!("s_cmp_lt_u32 s{S_WAVE}, 12"), &[], &[S_WAVE])?;
    op(b, format!("s_cbranch_scc1 {skip_v}"), &[], &[])?;
    let k0 = t(1);
    ins!(b, format!("v_add_nc_u32_e32 v{k0}, s{kt}, v{V_VKB}"); v(k0); s(kt), v(V_VKB))?;
    ins!(b, format!("v_cmp_gt_i32_e64 s{S_MB}, s{S_SEQ}, v{k0}"); m(S_MB); s(S_SEQ), v(k0))?;
    for i in 0..8u8 {
        vmov(b, V_VW + i, "0")?;
    }
    exec_and(b, S_SV2, S_MB)?;
    for r in 0..8u8 {
        let tmp = t(2 + r % 4);
        ins!(b, format!("v_add_nc_u32_e32 v{tmp}, {r}, v{k0}"); v(tmp); v(k0))?;
        ins!(b, format!("v_min_i32_e32 v{tmp}, s{S_GMAX}, v{tmp}"); v(tmp); s(S_GMAX), v(tmp))?;
        ins!(b, format!("v_mad_u32_u24 v{tmp}, v{tmp}, {}, v{V_GV}", lit(ROW_BYTES)); v(tmp); v(tmp), v(V_GV))?;
        gload(b, "b32", v(V_VW + r), tmp, S_VB, 0)?;
    }
    exec_restore(b, S_SV2)?;
    b.label(&skip_v)?;
    // Scales (wave 0).
    let skip_s = l.n(&format!("{tag}_skips"));
    sop(b, format!("s_cmp_lg_u32 s{S_WAVE}, 0"), &[], &[S_WAVE])?;
    op(b, format!("s_cbranch_scc1 {skip_s}"), &[], &[])?;
    for i in 0..4u8 {
        vmov(b, V_SP + i, "0")?;
    }
    let (a, bb) = (t(1), t(2));
    ins!(b, format!("v_add_nc_u32_e32 v{a}, s{off}, v{V_GS}"); v(a); s(off), v(V_GS))?;
    ins!(b, format!("v_cmp_gt_i32_e64 s{S_MC}, s{rem}, v{V_LANE}"); m(S_MC); s(rem), v(V_LANE))?;
    ins!(b, format!("v_add_nc_u32_e32 v{bb}, 32, v{V_LANE}"); v(bb); v(V_LANE))?;
    ins!(b, format!("v_cmp_gt_i32_e64 s{S_MD}, s{rem}, v{bb}"); m(S_MD); s(rem), v(bb))?;
    ins!(b, format!("v_cmp_gt_u32_e64 s{S_MA}, 16, v{V_LANE}"); m(S_MA); v(V_LANE))?;
    op(
        b,
        format!("s_and_b32 s{S_MD}, s{S_MD}, s{S_MA}"),
        &[m(S_MD)],
        &[m(S_MD), m(S_MA)],
    )?;
    exec_and(b, S_SV3, S_MC)?;
    gload(b, "u16", v(V_SP), a, S_KB, 0)?;
    gload(b, "u16", v(V_SP + 1), a, S_VB, 0)?;
    exec_restore(b, S_SV3)?;
    exec_and(b, S_SV4, S_MD)?;
    gload(b, "u16", v(V_SP + 2), a, S_KB, 32 * ROW_BYTES)?;
    gload(b, "u16", v(V_SP + 3), a, S_VB, 32 * ROW_BYTES)?;
    exec_restore(b, S_SV4)?;
    b.label(&skip_s)?;
    Ok(())
}

/// Commit the tile loaded by the matching [`issue`] into LDS buffer `wb` (the caller has drained the loads).
fn commit(b: &mut Builder, l: &L, tag: &str, kt: u8, wb: usize) -> R {
    let base = wb as u32 * BUFB;
    // K: key KI, dims 16 kdcf..+15 as the kg = 0 / kg = 1 fragment halves.
    dstore(b, wb, "b64", V_WK, vr(V_KW, 2), base)?;
    dstore(b, wb, "b64", V_WK, vr(V_KW + 2, 2), base + 256)?;
    // V (waves 12..23): 8 keys x 4 dims -> four fragment words (byte j = key k0 + j), dead keys zeroed.
    let skip_v = l.n(&format!("{tag}_cskipv"));
    sop(b, format!("s_cmp_lt_u32 s{S_WAVE}, 12"), &[], &[S_WAVE])?;
    op(b, format!("s_cbranch_scc1 {skip_v}"), &[], &[])?;
    let d = V_FR; // four (lo, hi) pairs
    for h in 0..2u8 {
        let w = V_VW + 4 * h;
        let (t0, t1, u0, u1) = (t(0), t(1), t(2), t(3));
        let perm = |b: &mut Builder, dst: u8, x: u8, y: u8, sel: u32| -> R {
            ins!(b, format!("v_perm_b32 v{dst}, v{x}, v{y}, {sel:#x}"); v(dst); v(x), v(y))
        };
        perm(b, t0, w + 1, w, 0x0501_0400)?;
        perm(b, t1, w + 1, w, 0x0703_0602)?;
        perm(b, u0, w + 3, w + 2, 0x0501_0400)?;
        perm(b, u1, w + 3, w + 2, 0x0703_0602)?;
        perm(b, d + h, u0, t0, 0x0504_0100)?;
        perm(b, d + 2 + h, u0, t0, 0x0706_0302)?;
        perm(b, d + 4 + h, u1, t1, 0x0504_0100)?;
        perm(b, d + 6 + h, u1, t1, 0x0706_0302)?;
    }
    // keep = bytes of live keys: nvalid = clamp(seq - (kt + VKB), 0, 8).
    let (k0, nv, sh, mlo, mhi) = (t(4), t(5), t(6), t(8), t(9));
    ins!(b, format!("v_add_nc_u32_e32 v{k0}, s{kt}, v{V_VKB}"); v(k0); s(kt), v(V_VKB))?;
    ins!(b, format!("v_sub_nc_u32_e32 v{nv}, s{S_SEQ}, v{k0}"); v(nv); s(S_SEQ), v(k0))?;
    ins!(b, format!("v_max_i32_e32 v{nv}, 0, v{nv}"); v(nv); v(nv))?;
    ins!(b, format!("v_min_i32_e32 v{nv}, 8, v{nv}"); v(nv); v(nv))?;
    ins!(b, format!("v_lshlrev_b32_e32 v{sh}, 3, v{nv}"); v(sh); v(nv))?;
    ins!(b, format!("v_cmp_lt_i32_e64 s{S_MA}, 7, v{nv}"); m(S_MA); v(nv))?;
    ins!(b, format!("v_lshlrev_b64_e64 {}, v{sh}, -1", vr(mlo, 2)); vr(mlo, 2); v(sh))?;
    ins!(b, format!("v_not_b32_e32 v{mlo}, v{mlo}"); v(mlo); v(mlo))?;
    ins!(b, format!("v_not_b32_e32 v{mhi}, v{mhi}"); v(mhi); v(mhi))?;
    ins!(b, format!("v_cndmask_b32_e64 v{mlo}, v{mlo}, -1, s{S_MA}"); v(mlo); v(mlo), m(S_MA))?;
    ins!(b, format!("v_cndmask_b32_e64 v{mhi}, v{mhi}, -1, s{S_MA}"); v(mhi); v(mhi), m(S_MA))?;
    for c in 0..4u8 {
        ins!(b, format!("v_and_b32_e32 v{}, v{mlo}, v{}", d + 2 * c, d + 2 * c); v(d + 2 * c); v(mlo), v(d + 2 * c))?;
        ins!(b, format!("v_and_b32_e32 v{}, v{mhi}, v{}", d + 2 * c + 1, d + 2 * c + 1); v(d + 2 * c + 1); v(mhi), v(d + 2 * c + 1))?;
        dstore(b, wb, "b64", V_WV, vr(d + 2 * c, 2), base + 16 * u32::from(c))?;
    }
    b.label(&skip_v)?;
    // Scales (wave 0).
    let skip_s = l.n(&format!("{tag}_cskips"));
    sop(b, format!("s_cmp_lg_u32 s{S_WAVE}, 0"), &[], &[S_WAVE])?;
    op(b, format!("s_cbranch_scc1 {skip_s}"), &[], &[])?;
    let (x0, x1) = (t(0), t(1));
    ins!(b, format!("v_cvt_f32_f16_e64 v{x0}, v{V_SP}.l"); v(x0); v(V_SP))?;
    ins!(b, format!("v_cvt_f32_f16_e64 v{x1}, v{}.l", V_SP + 1); v(x1); v(V_SP + 1))?;
    dstore(b, wb, "b32", V_WS, v(x0), base)?;
    dstore(b, wb, "b32", V_WS, v(x1), base + 192)?;
    ins!(b, format!("v_cmp_gt_u32_e64 s{S_MD}, 16, v{V_LANE}"); m(S_MD); v(V_LANE))?;
    ins!(b, format!("v_cvt_f32_f16_e64 v{x0}, v{}.l", V_SP + 2); v(x0); v(V_SP + 2))?;
    ins!(b, format!("v_cvt_f32_f16_e64 v{x1}, v{}.l", V_SP + 3); v(x1); v(V_SP + 3))?;
    exec_and(b, S_SAVE, S_MD)?;
    dstore(b, wb, "b32", V_WS, v(x0), base + 128)?;
    dstore(b, wb, "b32", V_WS, v(x1), base + 192 + 128)?;
    exec_restore(b, S_SAVE)?;
    b.label(&skip_s)?;
    // Drain the stores on every path (the role branches skip counted work): the buffer is published by the barrier
    // and the fill registers are redefined by the next issue.
    b.wait(Counter::Ds, 0)?;
    Ok(())
}

// ------------------------------------------------------------------ tile loop
/// One tile with LDS buffer `rb` read and `1 - rb` committed: commit of the next tile, issue of the one after,
/// the sub loop, the rendezvous (Ready next, Retire this) and the tile advance (exit to the epilogue past the
/// last tile).
fn tile(b: &mut Builder, l: &L, rb: usize) -> R {
    let wb = 1 - rb;
    let c = if rb == 0 { "a" } else { "b" };
    let epi = l.n("epi");
    let skip_cm = l.n(&format!("{c}_skipcommit"));
    let barrier = l.n(&format!("{c}_barrier"));
    sop(b, format!("s_add_co_i32 s{S_TG}, s{S_KT}, {}", lit(TK)), &[S_TG], &[S_KT])?;
    sop(b, format!("s_add_co_i32 s{S_TH}, s{S_KT}, {}", lit(2 * TK)), &[S_TH], &[S_KT])?;
    // Drain before the branch: every path (committing or not) reaches the issue with no load in flight.
    b.wait(Counter::Load, 0)?;
    sop(b, format!("s_cmp_ge_i32 s{S_TG}, s{S_SEQ}"), &[], &[S_TG, S_SEQ])?;
    op(b, format!("s_cbranch_scc1 {skip_cm}"), &[], &[])?;
    commit(b, l, &format!("{c}c"), S_TG, wb)?;
    b.label(&skip_cm)?;
    issue(b, l, &format!("{c}i"), S_TH)?;
    // Compute (dead waves only join the rendezvous).
    sop(b, format!("s_cmp_eq_u32 s{S_WLIVE}, 0"), &[], &[S_WLIVE])?;
    op(b, format!("s_cbranch_scc1 {barrier}"), &[], &[])?;
    // nsub = min(3, (gmax - ktile) / 16 + 1): subs with kt_sub <= gmax.
    sop(b, format!("s_sub_co_i32 s{S_TA}, s{S_GMAX}, s{S_KT}"), &[S_TA], &[S_GMAX, S_KT])?;
    sop(b, format!("s_lshr_b32 s{S_TA}, s{S_TA}, 4"), &[S_TA], &[S_TA])?;
    sop(b, format!("s_add_co_i32 s{S_TA}, s{S_TA}, 1"), &[S_TA], &[S_TA])?;
    sop(b, format!("s_min_u32 s{S_NSUB}, s{S_TA}, 3"), &[S_NSUB], &[S_TA])?;
    sop(b, format!("s_mov_b32 s{S_TI}, s{S_NSUB}"), &[S_TI], &[S_NSUB])?;
    sop(b, format!("s_mov_b32 s{S_KTS}, s{S_KT}"), &[S_KTS], &[S_KT])?;
    b.loop_(&l.n(&format!("{c}_sub")), |b| sub(b, l, rb, c))?;
    // Restore the fragment bases (nsub subs consumed).
    sop(b, format!("s_lshl_b32 s{S_TA}, s{S_TI}, 9"), &[S_TA], &[S_TI])?;
    ins!(b, format!("v_subrev_nc_u32_e32 v{V_RDK}, s{S_TA}, v{V_RDK}"); v(V_RDK); s(S_TA), v(V_RDK))?;
    sop(b, format!("s_lshl_b32 s{S_TA}, s{S_TI}, 12"), &[S_TA], &[S_TI])?;
    ins!(b, format!("v_subrev_nc_u32_e32 v{V_RDV}, s{S_TA}, v{V_RDV}"); v(V_RDV); s(S_TA), v(V_RDV))?;
    sop(b, format!("s_lshl_b32 s{S_TA}, s{S_TI}, 6"), &[S_TA], &[S_TI])?;
    ins!(b, format!("v_subrev_nc_u32_e32 v{V_RSK}, s{S_TA}, v{V_RSK}"); v(V_RSK); s(S_TA), v(V_RSK))?;
    b.label(&barrier)?;
    b.wait(Counter::Ds, 0)?;
    b.barrier(&[Transition::Ready(wb), Transition::Retire(rb)])?;
    sop(b, format!("s_add_co_i32 s{S_KT}, s{S_KT}, {TK}"), &[S_KT], &[S_KT])?;
    sop(b, format!("s_cmp_ge_i32 s{S_KT}, s{S_SEQ}"), &[], &[S_KT, S_SEQ])?;
    op(b, format!("s_cbranch_scc1 {epi}"), &[], &[])
}

fn body(b: &mut Builder, l: &L) -> R {
    tile(b, l, 0)?;
    tile(b, l, 1)?;
    op(b, format!("s_branch {}", l.n("tile")), &[], &[])
}

/// One 16-key sub: QK, scores, online softmax, P quantisation, PV. The fragment bases advance; the loop
/// counter and the fragment bases are restored by the caller.
fn sub(b: &mut Builder, l: &L, rb: usize, c: &str) -> R {
    let base = rb as u32 * BUFB;
    let head = l.n(&format!("{c}_sub"));
    let sa = V_SA;
    // ---- QK: S^T = K^T Q over 16 dimension fragments, in dimension order.
    op(b, "s_setprio 3", &[], &[])?;
    for p in 0..4u8 {
        dload128(b, rb, V_FR + 4 * p, V_RDK, base + u32::from(p) * 1536)?;
    }
    for p in 0..8u8 {
        let slot = V_FR + 4 * (p % 4);
        wmma8(b, sa, slot, V_QF + 4 * p, p > 0)?;
        wmma8(b, sa, slot + 2, V_QF + 4 * p + 2, true)?;
        if p + 4 < 8 {
            dload128(b, rb, slot, V_RDK, base + u32::from(p + 4) * 1536)?;
        }
    }
    op(b, "s_setprio 0", &[], &[])?;
    // Key/value scales of the sub: sk[0..8] -> FR[0..8], sv[0..8] -> FR[8..16].
    dload128(b, rb, V_FR, V_RSK, base)?;
    dload128(b, rb, V_FR + 4, V_RSK, base + 16)?;
    dload128(b, rb, V_FR + 8, V_RSK, base + 192)?;
    dload128(b, rb, V_FR + 12, V_RSK, base + 208)?;
    // ---- scores: s = (acc * qs) * sk, masked to -inf past the row's position.
    let full_path = l.n(&format!("{c}_fast"));
    let scored = l.n(&format!("{c}_scored"));
    sop(b, format!("s_add_co_i32 s{S_TA}, s{S_KTS}, 15"), &[S_TA], &[S_KTS])?;
    sop(b, format!("s_cmp_le_i32 s{S_TA}, s{S_GMIN}"), &[], &[S_TA, S_GMIN])?;
    sop(b, format!("s_cselect_b32 s{S_TB}, 1, 0"), &[S_TB], &[])?;
    sop(b, format!("s_and_b32 s{S_TC}, s{S_TB}, s{S_QALL}"), &[S_TC], &[S_TB, S_QALL])?;
    sop(b, format!("s_cmp_eq_u32 s{S_TC}, 1"), &[], &[S_TC])?;
    // Both score paths consume the scale registers: drain the DS loads before the branch.
    b.wait(Counter::Ds, 0)?;
    op(b, format!("s_cbranch_scc1 {full_path}"), &[], &[])?;
    // Masked: lim = sub_full ? 8 : pos - 8 kg - kt_sub; ok_j = qok && (j <= lim).
    let lim = t(16);
    ins!(b, format!("v_subrev_nc_u32_e32 v{lim}, s{S_KTS}, v{V_PK}"); v(lim); s(S_KTS), v(V_PK))?;
    let keep_lim = l.n(&format!("{c}_keeplim"));
    sop(b, format!("s_cmp_eq_u32 s{S_TB}, 0"), &[], &[S_TB])?;
    op(b, format!("s_cbranch_scc1 {keep_lim}"), &[], &[])?;
    vmov(b, lim, "8")?;
    b.label(&keep_lim)?;
    for j in 0..8u8 {
        ins!(b, format!("v_cmp_lt_i32_e64 s{}, {}, v{lim}", mj(j), i32::from(j) - 1); m(mj(j)); v(lim))?;
    }
    for j in 0..8u8 {
        op(
            b,
            format!("s_and_b32 s{0}, s{0}, s{S_QOK}", mj(j)),
            &[m(mj(j))],
            &[m(mj(j)), m(S_QOK)],
        )?;
    }
    for j in 0..8u8 {
        let x = sa + j;
        ins!(b, format!("v_mul_f32_e32 v{x}, v{V_QS}, v{x}"); v(x); v(V_QS), v(x))?;
        ins!(b, format!("v_mul_f32_e32 v{x}, v{x}, v{}", V_FR + j); v(x); v(x), v(V_FR + j))?;
        ins!(b, format!("v_cndmask_b32_e64 v{x}, {NEG_INF:#x}, v{x}, s{}", mj(j)); v(x); v(x), m(mj(j)))?;
    }
    op(b, format!("s_branch {scored}"), &[], &[])?;
    b.label(&full_path)?;
    for j in 0..8u8 {
        let x = sa + j;
        ins!(b, format!("v_mul_f32_e32 v{x}, v{V_QS}, v{x}"); v(x); v(V_QS), v(x))?;
        ins!(b, format!("v_mul_f32_e32 v{x}, v{x}, v{}", V_FR + j); v(x); v(x), v(V_FR + j))?;
    }
    b.label(&scored)?;
    softmax(b, l, c)?;
    pv(b, rb)?;
    // Advance the fragment bases to the next sub.
    ins!(b, format!("v_add_nc_u32_e32 v{V_RDK}, 0x200, v{V_RDK}"); v(V_RDK); v(V_RDK))?;
    ins!(b, format!("v_add_nc_u32_e32 v{V_RDV}, 0x1000, v{V_RDV}"); v(V_RDV); v(V_RDV))?;
    ins!(b, format!("v_add_nc_u32_e32 v{V_RSK}, 64, v{V_RSK}"); v(V_RSK); v(V_RSK))?;
    sop(b, format!("s_add_co_i32 s{S_KTS}, s{S_KTS}, 16"), &[S_KTS], &[S_KTS])?;
    sop(b, format!("s_sub_co_i32 s{S_NSUB}, s{S_NSUB}, 1"), &[S_NSUB], &[S_NSUB])?;
    sop(b, format!("s_cmp_gt_i32 s{S_NSUB}, 0"), &[], &[S_NSUB])?;
    op(b, format!("s_cbranch_scc1 {head}"), &[], &[])
}

/// Online softmax of one sub (the hipcc DAG): max, running max with the alpha rescale, exponentials, the serial
/// normaliser, the V-scaled P, the E4M3 scale `needed`, the O rescale and the bracketed P codes into `t(10), t(11)`.
fn softmax(b: &mut Builder, l: &L, c: &str) -> R {
    let sa = V_SA;
    let sk = |j: u8| V_FR + j;
    let sv = |j: u8| V_FR + 8 + j;
    let mx = t(0);
    let (x1, x2, x3) = (t(1), t(2), t(3));
    // mx = max over the lane's 8 keys (the hipcc max3 chain), then against the other key half.
    ins!(b, format!("v_max3_num_f32 v{mx}, v{}, {NEG_INF:#x}, v{sa}", sa + 1); v(mx); v(sa), v(sa + 1))?;
    for j in [2u8, 4, 6] {
        ins!(b, format!("v_max3_num_f32 v{mx}, v{mx}, v{}, v{}", sa + j, sa + j + 1); v(mx); v(mx), v(sa + j), v(sa + j + 1))?;
    }
    xor16(b, x1, mx)?;
    ins!(b, format!("v_max_num_f32_e32 v{x1}, v{x1}, v{x1}"); v(x1); v(x1))?;
    ins!(b, format!("v_max_num_f32_e32 v{mx}, v{mx}, v{x1}"); v(mx); v(mx), v(x1))?;
    // Running max: m == -inf -> mx; mx > m + 4 -> alpha rescale (only lanes with mx != -inf).
    ins!(b, format!("v_cmp_neq_f32_e64 s{S_MA}, {NEG_INF:#x}, v{mx}"); m(S_MA); v(mx))?;
    ins!(b, format!("v_cmp_eq_f32_e64 s{S_MB}, {NEG_INF:#x}, v{V_M}"); m(S_MB); v(V_M))?;
    ins!(b, format!("v_add_f32_e32 v{x1}, 4.0, v{V_M}"); v(x1); v(V_M))?;
    ins!(b, format!("v_cmp_gt_f32_e64 s{S_MD}, v{mx}, v{x1}"); m(S_MD); v(mx), v(x1))?;
    op(b, format!("s_and_b32 s{S_MC}, s{S_MA}, s{S_MB}"), &[m(S_MC)], &[m(S_MA), m(S_MB)])?;
    op(b, format!("s_and_b32 s{S_SV2}, s{S_MA}, s{S_MD}"), &[m(S_SV2)], &[m(S_MA), m(S_MD)])?;
    op(b, format!("s_and_not1_b32 s{S_SV2}, s{S_SV2}, s{S_MB}"), &[m(S_SV2)], &[m(S_SV2), m(S_MB)])?;
    let alpha_done = l.n(&format!("{c}_alpha_done"));
    op(b, format!("s_and_saveexec_b32 s{S_SAVE}, s{S_SV2}"), &[m(S_SAVE)], &[m(S_SV2)])?;
    op(b, format!("s_cbranch_execz {alpha_done}"), &[], &[])?;
    {
        let (al, yb) = (x1, x2);
        ins!(b, format!("v_sub_f32_e32 v{al}, v{V_M}, v{mx}"); v(al); v(V_M), v(mx))?;
        ins!(b, format!("v_mul_f32_e32 v{al}, {LOG2E:#x}, v{al}"); v(al); v(al))?;
        ins!(b, format!("v_exp_f32_e32 v{al}, v{al}"); v(al); v(al))?;
        ins!(b, format!("v_mul_f32_e32 v{V_L}, v{V_L}, v{al}"); v(V_L); v(V_L), v(al))?;
        ins!(b, format!("v_mul_f32_e32 v{V_B}, v{V_B}, v{al}"); v(V_B); v(V_B), v(al))?;
        fdiv(b, yb, "1.0", &format!("v{V_B}"), &[v(V_B)], [t(17), t(18), t(19), t(20)])?;
        ins!(b, format!("v_mul_f32_e32 v{V_YL}, {BRK_LO:#x}, v{yb}"); v(V_YL); v(yb))?;
        ins!(b, format!("v_mul_f32_e32 v{V_YH}, {BRK_HI:#x}, v{yb}"); v(V_YH); v(yb))?;
    }
    b.label(&alpha_done)?;
    exec_restore(b, S_SAVE)?;
    op(b, format!("s_or_b32 s{S_MC}, s{S_MC}, s{S_SV2}"), &[m(S_MC)], &[m(S_MC), m(S_SV2)])?;
    ins!(b, format!("v_cndmask_b32_e64 v{V_M}, v{V_M}, v{mx}, s{S_MC}"); v(V_M); v(V_M), v(mx), m(S_MC))?;
    // e_j = exp(s_j - m): lanes still at m == -inf produce 0.
    ins!(b, format!("v_cmp_eq_f32_e64 s{S_MA}, {NEG_INF:#x}, v{V_M}"); m(S_MA); v(V_M))?;
    for j in 0..8u8 {
        let x = sa + j;
        ins!(b, format!("v_sub_f32_e32 v{x}, v{x}, v{V_M}"); v(x); v(x), v(V_M))?;
        ins!(b, format!("v_mul_f32_e32 v{x}, {LOG2E:#x}, v{x}"); v(x); v(x))?;
        ins!(b, format!("v_exp_f32_e32 v{x}, v{x}"); v(x); v(x))?;
    }
    let fast_e = l.n(&format!("{c}_fast_e"));
    sop(b, format!("s_cmp_eq_u32 s{S_MA}, 0"), &[], &[S_MA])?;
    op(b, format!("s_cbranch_scc1 {fast_e}"), &[], &[])?;
    for j in 0..8u8 {
        let x = sa + j;
        ins!(b, format!("v_cndmask_b32_e64 v{x}, v{x}, 0, s{S_MA}"); v(x); v(x), m(S_MA))?;
    }
    b.label(&fast_e)?;
    // sum = ((e0 + e1) + e2) + ... (ascending, new term first), then the other key half, then l += sum.
    let sum = x2;
    ins!(b, format!("v_add_f32_e32 v{sum}, v{sa}, v{}", sa + 1); v(sum); v(sa), v(sa + 1))?;
    for j in 2..8u8 {
        ins!(b, format!("v_add_f32_e32 v{sum}, v{}, v{sum}", sa + j); v(sum); v(sa + j), v(sum))?;
    }
    xor16(b, x3, sum)?;
    ins!(b, format!("v_add_f32_e32 v{sum}, v{sum}, v{x3}"); v(sum); v(sum), v(x3))?;
    ins!(b, format!("v_add_f32_e32 v{V_L}, v{V_L}, v{sum}"); v(V_L); v(V_L), v(sum))?;
    // P = sv * e.
    for j in 0..8u8 {
        let x = sa + j;
        ins!(b, format!("v_mul_f32_e32 v{x}, v{}, v{x}", sv(j)); v(x); v(sv(j)), v(x))?;
    }
    // wmax = max(0, p0..p7) over both halves; needed = wmax > 0 ? max(wmax / 448, 2^-64) : b.
    let wm = x1;
    ins!(b, format!("v_max3_num_f32 v{wm}, v{sa}, 0, v{}", sa + 1); v(wm); v(sa), v(sa + 1))?;
    for j in [2u8, 4, 6] {
        ins!(b, format!("v_max3_num_f32 v{wm}, v{wm}, v{}, v{}", sa + j, sa + j + 1); v(wm); v(wm), v(sa + j), v(sa + j + 1))?;
    }
    xor16(b, x3, wm)?;
    ins!(b, format!("v_max_num_f32_e32 v{x3}, v{x3}, v{x3}"); v(x3); v(x3))?;
    ins!(b, format!("v_max_num_f32_e32 v{wm}, v{wm}, v{x3}"); v(wm); v(wm), v(x3))?;
    let (q, needed) = (x2, x3);
    ins!(b, format!("v_mul_f32_e32 v{q}, {INV448:#x}, v{wm}"); v(q); v(wm))?;
    ins!(b, format!("v_cmp_lt_f32_e64 s{S_MB}, 0, v{wm}"); m(S_MB); v(wm))?;
    ins!(b, format!("v_fmac_f32_e32 v{wm}, {NEG448:#x}, v{q}"); v(wm); v(wm), v(q))?;
    ins!(b, format!("v_fmac_f32_e32 v{q}, {INV448:#x}, v{wm}"); v(q); v(q), v(wm))?;
    ins!(b, format!("v_max_num_f32_e32 v{q}, {TWO_M64:#x}, v{q}"); v(q); v(q))?;
    ins!(b, format!("v_cndmask_b32_e64 v{needed}, v{V_B}, v{q}, s{S_MB}"); v(needed); v(V_B), v(q), m(S_MB))?;
    // needed > b: rescale O by rho = b / needed, b = needed, refresh the reciprocal brackets.
    let resc_done = l.n(&format!("{c}_resc_done"));
    ins!(b, format!("v_cmp_gt_f32_e64 s{S_MC}, v{needed}, v{V_B}"); m(S_MC); v(needed), v(V_B))?;
    op(b, format!("s_and_saveexec_b32 s{S_SAVE}, s{S_MC}"), &[m(S_SAVE)], &[m(S_MC)])?;
    op(b, format!("s_cbranch_execz {resc_done}"), &[], &[])?;
    {
        let (rho, yb) = (t(4), t(5));
        fdiv(b, rho, &format!("v{V_B}"), &format!("v{needed}"), &[v(V_B), v(needed)], [t(17), t(18), t(19), t(20)])?;
        fdiv(b, yb, "1.0", &format!("v{needed}"), &[v(needed)], [t(17), t(18), t(19), t(20)])?;
        for i in 0..128u8 {
            let o = V_O + i;
            ins!(b, format!("v_mul_f32_e32 v{o}, v{o}, v{rho}"); v(o); v(o), v(rho))?;
        }
        ins!(b, format!("v_mov_b32_e32 v{V_B}, v{needed}"); v(V_B); v(needed))?;
        ins!(b, format!("v_mul_f32_e32 v{V_YL}, {BRK_LO:#x}, v{yb}"); v(V_YL); v(yb))?;
        ins!(b, format!("v_mul_f32_e32 v{V_YH}, {BRK_HI:#x}, v{yb}"); v(V_YH); v(yb))?;
    }
    b.label(&resc_done)?;
    exec_restore(b, S_SAVE)?;
    // P codes: cvt(p * yb_lo) vs cvt(p * yb_hi); equal codes are the IEEE-division codes.
    let (pw0, pw1, hw0, hw1) = (t(10), t(11), t(12), t(13));
    let (u0, u1) = (t(14), t(15));
    for (words, y) in [([pw0, pw1], V_YL), ([hw0, hw1], V_YH)] {
        for (wi, w) in words.into_iter().enumerate() {
            for half in 0..2u8 {
                let j = (4 * wi as u8) + 2 * half;
                ins!(b, format!("v_mul_f32_e32 v{u0}, v{}, v{y}", sa + j); v(u0); v(sa + j), v(y))?;
                ins!(b, format!("v_mul_f32_e32 v{u1}, v{}, v{y}", sa + j + 1); v(u1); v(sa + j + 1), v(y))?;
                if half == 0 {
                    ins!(b, format!("v_cvt_pk_fp8_f32 v{w}.l, v{u0}, v{u1}"); v(w); v(u0), v(u1), v(w))?;
                } else {
                    ins!(b, format!("v_cvt_pk_fp8_f32 v{w}.h, v{u0}, v{u1} op_sel:[0,0,1]"); v(w); v(u0), v(u1), v(w))?;
                }
            }
        }
    }
    ins!(b, format!("v_cmp_ne_u32_e64 s{S_MA}, v{pw0}, v{hw0}"); m(S_MA); v(pw0), v(hw0))?;
    ins!(b, format!("v_cmp_ne_u32_e64 s{S_MB}, v{pw1}, v{hw1}"); m(S_MB); v(pw1), v(hw1))?;
    op(b, format!("s_or_b32 s{S_MA}, s{S_MA}, s{S_MB}"), &[m(S_MA)], &[m(S_MA), m(S_MB)])?;
    let p_done = l.n(&format!("{c}_pcode_done"));
    op(b, format!("s_and_saveexec_b32 s{S_SAVE}, s{S_MA}"), &[m(S_SAVE)], &[m(S_MA)])?;
    op(b, format!("s_cbranch_execz {p_done}"), &[], &[])?;
    for (wi, w) in [pw0, pw1].into_iter().enumerate() {
        for half in 0..2u8 {
            let j = (4 * wi as u8) + 2 * half;
            fdiv(b, u0, &format!("v{}", sa + j), &format!("v{V_B}"), &[v(sa + j), v(V_B)], [t(17), t(18), t(19), t(20)])?;
            fdiv(b, u1, &format!("v{}", sa + j + 1), &format!("v{V_B}"), &[v(sa + j + 1), v(V_B)], [t(17), t(18), t(19), t(20)])?;
            if half == 0 {
                ins!(b, format!("v_cvt_pk_fp8_f32 v{w}.l, v{u0}, v{u1}"); v(w); v(u0), v(u1), v(w))?;
            } else {
                ins!(b, format!("v_cvt_pk_fp8_f32 v{w}.h, v{u0}, v{u1} op_sel:[0,0,1]"); v(w); v(u0), v(u1), v(w))?;
            }
        }
    }
    b.label(&p_done)?;
    exec_restore(b, S_SAVE)?;
    Ok(())
}

/// `O += V^T P`: V fragments first operand, the P words (`t(10), t(11)`) second, per 16-dim chunk pair.
fn pv(b: &mut Builder, rb: usize) -> R {
    let base = rb as u32 * BUFB;
    let pb = t(10);
    op(b, "s_setprio 3", &[], &[])?;
    for p in 0..4u8 {
        dload128(b, rb, V_FR + 4 * p, V_RDV, base + u32::from(p) * 512)?;
    }
    for p in 0..8u8 {
        let slot = V_FR + 4 * (p % 4);
        wmma8(b, V_O + 16 * p, slot, pb, true)?;
        wmma8(b, V_O + 16 * p + 8, slot + 2, pb, true)?;
        if p + 4 < 8 {
            dload128(b, rb, slot, V_RDV, base + u32::from(p + 4) * 512)?;
        }
    }
    op(b, "s_setprio 0", &[], &[])
}

// ------------------------------------------------------------------ epilogue
/// `out = O * (l > 0 ? b / l : 0)` for live rows, stored as two 16-byte halves per 16-dim chunk.
fn epilogue(b: &mut Builder, l: &L) -> R {
    let (epi, end) = (l.n("epi"), l.n("end"));
    b.label(&epi)?;
    sop(b, format!("s_cmp_eq_u32 s{S_WLIVE}, 0"), &[], &[S_WLIVE])?;
    op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
    let inv = t(0);
    ins!(b, format!("v_cmp_lt_f32_e64 s{S_MA}, 0, v{V_L}"); m(S_MA); v(V_L))?;
    fdiv(b, inv, &format!("v{V_B}"), &format!("v{V_L}"), &[v(V_B), v(V_L)], [t(1), t(2), t(3), t(4)])?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    ins!(b, format!("v_cndmask_b32_e64 v{inv}, 0, v{inv}, s{S_MA}"); v(inv); v(inv), m(S_MA))?;
    for i in 0..128u8 {
        let o = V_O + i;
        ins!(b, format!("v_mul_f32_e32 v{o}, v{o}, v{inv}"); v(o); v(o), v(inv))?;
    }
    // Byte offset of the lane's first element: (query * 16 + head) * 1024 + 32 kg.
    let (row, a, c) = (t(5), t(6), t(7));
    sop(b, format!("s_lshl_b32 s{S_TA}, s{S_WAVE}, 4"), &[S_TA], &[S_WAVE])?;
    sop(b, format!("s_add_co_i32 s{S_TA}, s{S_TA}, s{S_ROWB}"), &[S_TA], &[S_TA, S_ROWB])?;
    ins!(b, format!("v_and_b32_e32 v{c}, 15, v{V_LANE}"); v(c); v(V_LANE))?;
    ins!(b, format!("v_add_nc_u32_e32 v{row}, s{S_TA}, v{c}"); v(row); s(S_TA), v(c))?;
    ins!(b, format!("v_lshrrev_b32_e32 v{a}, 3, v{row}"); v(a); v(row))?;
    ins!(b, format!("v_add_nc_u32_e32 v{a}, s{S_QOFF}, v{a}"); v(a); s(S_QOFF), v(a))?;
    sop(b, format!("s_lshl_b32 s{S_TB}, s{S_KVH}, 3"), &[S_TB], &[S_KVH])?;
    ins!(b, format!("v_and_b32_e32 v{c}, 7, v{row}"); v(c); v(row))?;
    ins!(b, format!("v_add_nc_u32_e32 v{c}, s{S_TB}, v{c}"); v(c); s(S_TB), v(c))?;
    ins!(b, format!("v_lshl_add_u32 v{a}, v{a}, 4, v{c}"); v(a); v(a), v(c))?;
    ins!(b, format!("v_lshlrev_b32_e32 v{a}, 10, v{a}"); v(a); v(a))?;
    ins!(b, format!("v_bfe_u32 v{c}, v{V_LANE}, 4, 1"); v(c); v(V_LANE))?;
    ins!(b, format!("v_lshl_add_u32 v{a}, v{c}, 5, v{a}"); v(a); v(c), v(a))?;
    exec_and(b, S_SAVE, S_QOK)?;
    for dc in 0..16u8 {
        gstore128(b, a, V_O + 8 * dc, S_OUT, u32::from(dc) * 64)?;
        gstore128(b, a, V_O + 8 * dc + 4, S_OUT, u32::from(dc) * 64 + 16)?;
    }
    exec_restore(b, S_SAVE)?;
    b.wait_all()?;
    b.label(&end)?;
    b.control(Sop::End.encode(Arch::Gfx1201)?)
}

// ------------------------------------------------------------------ emission
/// One export (`range` selects the symbol; the body is shared).
pub fn emit(arch: Arch, spec: &FaQresSpec, range: bool) -> Result<Emitted, String> {
    admit(arch, spec)?;
    let l = L {
        tag: if range { "r" } else { "d" },
    };
    let (symbol, variant) = if range {
        (SYMBOL_RANGE, "range")
    } else {
        (SYMBOL_DIRECT, "direct")
    };
    let mut b = Builder::new(kernel_spec(symbol, variant), plan(&l)?);
    b.enable_delay_alu();
    b.enable_workgroup_id_z();
    for (id, (name, base)) in [("buf0", 0u32), ("buf1", BUFB)].into_iter().enumerate() {
        if b.lds_slot(name, base, BUFB)? != id {
            return Err("LDS slot order".into());
        }
    }
    prologue(&mut b, &l)?;
    b.loop_(&l.n("tile"), |b| body(b, &l))?;
    epilogue(&mut b, &l)?;
    b.finish()
}

/// The module: the direct then the range export.
pub fn emit_module(
    arch: Arch,
    spec: &FaQresSpec,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    admit(arch, spec)?;
    let emitted = vec![emit(arch, spec, false)?, emit(arch, spec, true)?];
    let (text, proof) = super::iu4_gemm::module(&emitted, MODULE)?;
    Ok((emitted, text, proof))
}


// ------------------------------------------------------------------ contracts
const COUNTED: [&str; 24] = [
    "v_wmma_f32_16x16x16_fp8_fp8",
    "v_cvt_pk_fp8_f32",
    "s_barrier_signal",
    "s_barrier_wait",
    "ds_load_b128",
    "ds_store_b64",
    "ds_store_b32",
    "global_load_u16",
    "global_load_b32",
    "global_load_b128",
    "global_store_b128",
    "s_load_b256",
    "s_load_b128",
    "s_load_b32",
    "v_exp_f32_e32",
    "v_permlanex16_b32",
    "v_max3_num_f32",
    "v_perm_b32",
    "v_div_scale_f32",
    "v_div_fmas_f32",
    "v_div_fixup_f32",
    "v_rcp_f32_e32",
    "s_setprio",
    "s_endpgm",
];

fn counts(e: &Emitted) -> serde_json::Map<String, serde_json::Value> {
    let mut counts = serde_json::Map::new();
    for name in COUNTED {
        let n = e
            .s_text
            .lines()
            .filter(|l| l.trim().split_whitespace().next() == Some(name))
            .count();
        counts.insert(name.into(), n.into());
    }
    counts
}

/// Shape contract of one export (`toolchain::IsaShapeContract`): counts are derived from the emission.
pub fn contract(arch: Arch, emitted: &Emitted, symbol: &str) -> serde_json::Value {
    let _ = arch;
    serde_json::json!({
        "symbol": symbol, "counts": counts(emitted),
        "forbidden": ["scratch_*", "flat_*", "buffer_*", "global_atomic*", "ds_atomic*", "ds_bpermute*", "ds_swizzle*",
                      "v_writelane*", "s_setpc*", "s_swappc*", "s_call*", "s_waitcnt*", "global_load_tr*"],
        "vgpr_max": 240, "sgpr_max": 104,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": DYNAMIC_LDS_BYTES,
        "group_segment_fixed_bytes": 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_matches_the_hipcc_launcher() {
        assert_eq!(grid(1), [1, 2, 1]);
        assert_eq!(grid(48), [1, 2, 1]);
        assert_eq!(grid(49), [2, 2, 1]);
        assert_eq!(grid(512), [11, 2, 1]);
        assert_eq!(grid(513), [11, 2, 2]);
        assert_eq!(grid(8192), [11, 2, 16]);
    }

    #[test]
    fn the_direct_export_emits() {
        let e = emit(Arch::Gfx1201, &FaQresSpec::A3B, false).unwrap();
        assert!(e.s_text.contains(SYMBOL_DIRECT));
        eprintln!("{:?}", e.shape);
    }

    #[test]
    fn the_module_assembles_natively() {
        let (_, text, _) = emit_module(Arch::Gfx1201, &FaQresSpec::A3B).unwrap();
        std::fs::write(
            std::env::var("FAQRES_DUMP").unwrap_or_else(|_| "/dev/null".into()),
            &text,
        )
        .unwrap();
        let elf = crate::native::assemble(&text, Arch::Gfx1201).unwrap();
        eprintln!("{} bytes", elf.len());
    }
}
