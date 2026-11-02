// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! gfx1201 GQA-fused FA2 prefill attention (Q8_0 K/V, f16 pre-converted Q) for
//! H = 16, Hkv = 2, D = 256, emitted with the checked ISA builder. Two exports:
//!
//!  * `attention_fa2_q_preconvert_pm_gfx1201`: f32 Q -> f16 Q16 scratch (RNE),
//!    one wave32 per (row, head), 4 per 128-thread block;
//!  * `attention_q8_0_fa2_gqa_pm_gfx1201`: causal direct attention
//!    (implicit `n_splits = 1`), 160 threads, dynamic LDS 65 536 bytes.
//!
//! The direct kernel is the twin of the 16/2/256 direct oracle
//! `lab_kernels/fa_a3b_fa2.hip` (`fa2_gqa_body<false>`): one workgroup owns one
//! KV head and eight query positions (8 x 8 query/head rows = four 16-row WMMA
//! tiles, one compute wave each) plus one trailing helper wave that never
//! fills or computes but takes part in every barrier. K and V are filled as
//! f16 fragment planes (K bytes `[0, 32768)`, V bytes `[32768, 65536)`):
//! `koff(sub,dc,lane,j) = ((dc*4+sub)*32+lane)*8+j`,
//! `voff(sub,dc,lane,j) = ((sub*16+dc)*32+lane)*8+j` (half elements).
//!
//! Flow per KT64 tile: `fill -> barrier(Ready K,V) -> QK -> [softmax -> PV] x
//! live subtiles -> barrier(Retire K,V)`. Rounding DAG of the oracle object:
//!  * Q8 decode: sign-extended byte -> f32; `f16(fma(scale_f32, code, -0))`
//!    (`v_fma_mixlo/hi_f16`, addend `neg(0)`, exact product, one rounding);
//!  * score `s * scale_attn` (f32 multiply), causal select to `-inf`;
//!  * `max` via `v_max3_num_f32`; `__expf(x)` = `v_exp_f32(x * 0x3fb8aa3b)`
//!    with the oracle's `m == -inf` guards on alpha and on every probability;
//!  * probabilities round once to f16 (RNE) for the PV B operand; the
//!    normalizer sums the f32 exponents serially (ascending j) then XOR-16,
//!    and `l = fma(l, alpha, sum)` is one fused rounding;
//!  * every O element is scaled by its lane's alpha before each subtile's PV;
//!  * epilogue `inv = l > 0 ? 1/l : 0` with the full f32 division graph, then
//!    a separate `O * inv` multiply.
//!
//! Causality is defined by `positions[]` (each row's own position), the group
//! bounds `gmin/gmax` come from the eight group positions (every wave: lanes 0..7
//! with a live query hold their position, all other lanes the sentinels -1 /
//! INT_MAX, then lane XOR 16/8/4/2/1 integer max/min and readfirstlane) and are
//! workgroup-uniform. A subtile is `full` when `kt_sub + 15 <= gmin` (uniform
//! branch): its key mask is `qok` for every key; otherwise the per-row mask is
//! `qok && key <= position` (invalid query lanes carry position `-1`).
//!
//! Q8 decode keeps the oracle DAG: sign-extended byte (exact integer
//! `(x ^ 0x80) - 0x80` of the unsigned byte) -> `v_cvt_f32_i32`, unmodified f32 scale,
//! `v_fma_mixlo_f16 ..., neg(0)` (one rounding, signed zero preserved); the high half
//! of each packed dword is joined from a second `mixlo` with a byte permute.
//!
//! Address strategy: every LDS address VGPR (`v1`, `v2`, `v3`) is derived from
//! the work-item id in the entry block (before the first branch) and never
//! redefined, so `pm_check::lds_bounds` bounds every lane of all five waves.
//!
//! K/V global addresses are 64-bit (`SGPR base + ktile * 544`), the per-lane
//! in-tile offsets are 32-bit.
//!
//! FP8 variant (`fa_prefill_fp8_pm`, [`FaKvCodec::Fp8E4M3`], [`emit_fp8_module`]): one direct export
//! `attention_fp8_e4m3_fa2_gqa_f16_pm_gfx1201`, the PeaceMaker twin of the production Q0 hipcc entry
//! (`attention_q8_0_fa2_gqa.gfx1201.hip`, `KMODE=8`, `-DHIPFIRE_FA2_Q0_A3B`). Same 60-byte ABI, grid, block,
//! QK, softmax, PV and epilogue shape as the Q8 direct entry; only the K/V global addressing, the fill-time
//! dequantisation and the LDS/register staging differ. K and V rows are 516-byte native-fp8 token rows `[512
//! head-major E4M3 code bytes][2 f16 scales]`: code byte `key*516 + head*256 + d`, scale at `key*516 + 512 +
//! head*2`. The code and scale bases are separate per-lane VGPR offsets (`head*256` versus `512 + head*2`) over
//! one 64-bit tile base (`k_cache + ktile*516`). Decode: `v_cvt_pk_f32_fp8` (gfx1201 OCP E4M3FN, no FNUZ fold)
//! of two bytes per instruction (`e32 .l`, `e64 .h op_sel:[1,0]`) to f32, f32 value of the f16 scale, then the
//! unchanged [`dequant_pair`] (`v_fma_mixlo_f16 .., neg(0)`: exact product, one rounding, as the hipcc
//! `(_Float16)(sf * d)`). One 32-KiB LDS stage (base 0, [`FP8_DYNAMIC_LDS_BYTES`]) is reused sequentially
//! for K then V: `fill K -> Ready -> QK -> Retire -> fill V -> Ready -> [softmax -> PV] -> Retire`, four
//! barrier pairs per KT64 tile, max 224 VGPRs. Scores 16..47 survive V-fill; Q 48..63 whole; O 96..223 whole.
//! The Q preconvert export is not part of this module: it is the Q8 module's `attention_fa2_q_preconvert_pm_gfx1201`.
//!

use crate::insn::{Instruction, MemoryClass, Sop};
use crate::kernels::common::{lit, op, s, sop, sr, v, vr};
use crate::lds::Transition;
use crate::ledger::Counter;
use crate::reg::{Live, RegRef};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

pub mod blocks;

/// CLI kernel id.
pub const KERNEL_ID: &str = "fa_prefill_pm";
/// Module / cache key.
pub const MODULE: &str = "fa_prefill_pm_gfx1201";
/// Direct attention symbol.
pub const SYMBOL_DIRECT: &str = "attention_q8_0_fa2_gqa_pm_gfx1201";
/// Q preconvert symbol.
pub const SYMBOL_PRECONVERT: &str = "attention_fa2_q_preconvert_pm_gfx1201";
/// CLI kernel id of the native-fp8 variant.
pub const KERNEL_ID_FP8: &str = "fa_prefill_fp8_pm";
/// Module / cache key of the native-fp8 variant (one direct export).
pub const MODULE_FP8: &str = "fa_prefill_fp8_pm_gfx1201";
/// Native-fp8 direct attention symbol (the PM twin of the Q0 hipcc entry).
pub const SYMBOL_FP8_DIRECT: &str = "attention_fp8_e4m3_fa2_gqa_f16_pm_gfx1201";
/// Bytes per native-fp8 token row: 2 KV heads x 256 E4M3 code bytes, then 2 f16 scales.
pub const FP8_ROW_BYTES: u32 = 516;
/// Byte offset of the first f16 scale in a native-fp8 row.
pub const FP8_SCALE_OFFSET: u32 = 512;
/// Direct block size (four compute waves plus one helper wave).
pub const DIRECT_BLOCK: u32 = 160;
/// Preconvert block size.
pub const PRECONVERT_BLOCK: u32 = 128;
/// Direct launch dynamic LDS bytes (K plane + V plane).
pub const DYNAMIC_LDS_BYTES: u32 = 65_536;
/// FP8 direct launch dynamic LDS bytes: one 32-KiB stage reused sequentially for K then V.
pub const FP8_DYNAMIC_LDS_BYTES: u32 = 32_768;
/// Direct kernarg segment bytes (the oracle's, not rounded).
pub const DIRECT_KERNARG_BYTES: u32 = 60;
/// Preconvert kernarg segment bytes.
pub const PRECONVERT_KERNARG_BYTES: u32 = 40;
/// Largest launch row count of the authoring domain.
pub const MAX_BATCH: usize = 8192;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaKvCodec {
    Q8_0,
    /// Native OCP E4M3FN codes with one f16 scale per (token, KV head): 516-byte token rows.
    Fp8E4M3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaPrefillSpec {
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub kv: FaKvCodec,
}

impl FaPrefillSpec {
    /// Exactly `(16, 2, 256)` with a supported K/V codec (`Q8_0` or `Fp8E4M3`).
    pub fn validate(&self) -> Result<(), String> {
        if self.heads != 16 || self.kv_heads != 2 || self.head_dim != 256 {
            return Err(format!(
                "fa_prefill: only heads 16, kv_heads 2, head_dim 256 are supported, got {}/{}/{}/{:?}",
                self.heads, self.kv_heads, self.head_dim, self.kv
            ));
        }
        Ok(())
    }
}

/// Pure length-level launch validation: valid spec, `1 <= batch_size <= 8192`
/// and `positions_len == batch_size`. Production routing keeps its own
/// 64..=8192 admission; device contents are not inspected here.
pub fn validate(
    spec: &FaPrefillSpec,
    batch_size: usize,
    positions_len: usize,
) -> Result<(), String> {
    spec.validate()?;
    if batch_size < 1 || batch_size > MAX_BATCH {
        return Err(format!(
            "fa_prefill: batch_size {batch_size} is outside 1..={MAX_BATCH}"
        ));
    }
    if positions_len != batch_size {
        return Err(format!(
            "fa_prefill: positions length {positions_len} != batch_size {batch_size}"
        ));
    }
    Ok(())
}

fn admit(arch: Arch, spec: &FaPrefillSpec) -> R {
    if arch != Arch::Gfx1201 {
        return Err("fa_prefill: built for gfx1201 only".into());
    }
    spec.validate()
}

// ------------------------------------------------------------------ labels
// Labels are unique per kernel: the module combiner shares one assembler unit.
// Q8 keeps the two-barrier schedule (D_LOOP, D_FILL_DONE, D_QK, D_TURN). FP8
// reuses one 32-KiB stage sequentially (D_LOOP, D_KFILL_DONE, D_KREADY, D_QK,
// D_QK_DONE, D_VFILL_DONE, D_VREADY, D_TURN): K-fill, QK, V-fill, softmax/PV,
// with all five waves joining all four barriers.
const D_LOOP: &str = ".Lfapd_tile";
const D_FILL_DONE: &str = ".Lfapd_fill_done";
const D_KFILL_DONE: &str = ".Lfapd_kfill_done";
const D_KREADY: &str = ".Lfapd_kready";
const D_QK: &str = ".Lfapd_qk";
const D_QK_DONE: &str = ".Lfapd_qk_done";
const D_VFILL_DONE: &str = ".Lfapd_vfill_done";
const D_VREADY: &str = ".Lfapd_vready";
const D_TURN: &str = ".Lfapd_turn";
const D_EPI: &str = ".Lfapd_epi";
const D_END: &str = ".Lfapd_end";
const P_END: &str = ".Lfapc_end";

// LDS slots (ids follow declaration order). Q8 declares two simultaneously
// resident 32768-byte planes (L_K, L_V). FP8 declares one 32768-byte stage
// (L_STAGE, id 0) reused sequentially for K then V.
const L_K: usize = 0;
const L_V: usize = 1;
const L_STAGE: usize = 0;

// ------------------------------------------------------------------ VGPR map (direct)
const TID: u8 = 0;
const V_LK: u8 = 1; // K fill store base (immutable LDS address)
const V_LV: u8 = 2; // V fill store base (immutable LDS address)
const V_LF: u8 = 3; // fragment read base 16*lane (immutable LDS address)
const V_GK: u8 = 4; // global K in-tile offset
const V_GV: u8 = 5; // global V in-tile offset
const V_QO: u8 = 6; // q16 byte offset (qr*8192 + h*512 + kg*16)
const V_POS: u8 = 7; // effective row position (-1 for an invalid query)
const V_KG8: u8 = 8; // 8 * (lane >> 4)
const V_M: u8 = 9; // running max
const V_L: u8 = 10; // running normalizer
const V_VK: u8 = 11; // (tid & 127) >> 3
const V_VK2: u8 = 12; // 2 * V_VK
const V_GSK: u8 = 13; // fp8 only: K scale lane offset (516*vk, then + 512 + 2*head); spare in the Q8 kernel
const V_GSV: u8 = 14; // fp8 only: V scale lane offset (1032*vk, then + 512 + 2*head); spare in the Q8 kernel
const S0: u8 = 16; // four score accumulators, 16 + 8 * sub
const QF: u8 = 48; // four Q half8 fragments, 48 + 4 * u (invalid lanes stay zero)
const KF: u8 = 64; // four K half8 fragments, 64 + 4 * u
const PROW: u8 = 80; // packed probabilities
const VFA: u8 = 84; // V fragment double buffer
const VFB: u8 = 88;
const V_MX: u8 = 92;
const V_SWZ: u8 = 93;
const V_MN: u8 = 94;
const V_AL: u8 = 95;
const V_SUM: u8 = 96;
const V_T1: u8 = 97;
const V_KEY: u8 = 98;
const V_KJ: u8 = 99;
const O: u8 = 128; // 16 x 8 f32 output accumulators
                   // Fill phase (aliases the score / fragment / softmax region; dead outside the fill).
const FK: u8 = 16; // K codes, slot t at 16 + 8t
const FSC: u8 = 64; // K scales 64 + t, V scales 68 + 2t' (+1)
const FV: u8 = 72; // V codes: slot t' u0 = 72 + 16t', u1 = 80 + 16t'
const FOUT: u8 = 104; // store-data ring
const FTMP: u8 = 112; // integer / float temporaries
                      // FP8 sequential-stage aliases (max 224; Q8 aliases above are untouched).
                      // K-fill (live D_LOOP..D_KREADY): codes 16..47, scales 64..67,
                      // output ring 68..71 (pairs (68,69),(70,71)), temporaries 72..79.
const FK8: u8 = 16;
const FSC8: u8 = 64;
const FOUT8: u8 = 68;
const FTMP8: u8 = 72;
// V-fill streamed one t' group at a time (live D_QK_DONE..D_VREADY):
// codes 64..79, scales 80..81, output ring 84..87, temporaries 88..95.
const VFC8: u8 = 64;
const VFSC8: u8 = 80;
const VFOUT8: u8 = 84;
const VFTMP8: u8 = 88;
// Softmax/PV (live D_VREADY..D_TURN): temporaries 64..71, packed
// probabilities 80..83, V fragment buffers 84..87/88..91, O 96..223.
const SMX8: u8 = 64;
const SPRO8: u8 = 80;
const SVFA8: u8 = 84;
const SVFB8: u8 = 88;
const O8: u8 = 96;

// ------------------------------------------------------------------ SGPR map (direct)
const KARG: u8 = 0;
const SQ16: u8 = 8; // q16, k_cache (10), v_cache (12), out (14)
const SK: u8 = 10;
const SV: u8 = 12;
const SO: u8 = 14;
const SPOS: u8 = 16;
const NHEADS: u8 = 20;
const NKV: u8 = 21;
const HDIM: u8 = 22;
const BATCH: u8 = 23;
const SCALE: u8 = 24;
const QBASE: u8 = 25;
const KVH: u8 = 26;
const GMAX: u8 = 27;
const GMIN: u8 = 28;
const KT: u8 = 29;
const REM: u8 = 30; // gmax - ktile
const REMMIN: u8 = 31; // gmin - ktile (kept for the contract of the oracle's `full` test)
const KB: u8 = 32;
const VB: u8 = 34;
const KB0: u8 = 36;
const VB0: u8 = 38;
const PSEL: u8 = 40;
const WAVE: u8 = 41;
const T: [u8; 8] = [42, 43, 44, 45, 46, 47, 80, 81];
const QOK: u8 = 48; // lane masks are pairs; the high word is never written in wave32
const SAVE: u8 = 50;
const CMPA: u8 = 52;
const CMPB: u8 = 54;
const MJ: u8 = 56; // eight pairs
const FM: u8 = 72; // four pairs

fn mj(j: u8) -> u8 {
    MJ + 2 * j
}
fn fm(t: u8) -> u8 {
    FM + 2 * t
}
fn m(x: u8) -> RegRef {
    sr(x, 2)
}

// ------------------------------------------------------------------ emission helpers
fn off(imm: u32) -> String {
    if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    }
}
fn imm_i32(x: i32) -> String {
    lit(x as u32)
}

/// `global_load_<kind> dst, v{voff}, s[sbase:sbase+1] offset:imm` (SADDR form, 32-bit lane offset).
pub(super) fn gload(b: &mut Builder, kind: &str, dst: RegRef, voff: u8, sbase: u8, imm: u32) -> R {
    b.push(
        Instruction::new(
            format!(
                "global_load_{kind} {dst}, v{voff}, s[{sbase}:{}]{}",
                sbase + 1,
                off(imm)
            ),
            vec![dst],
            vec![v(voff), sr(sbase, 2)],
        )
        .memory(MemoryClass::VmemLoad),
    )
}
/// One dword store per register of the lane's contiguous eight f32 / four f16x2 run:
/// `global_store_b32` at `imm + 4i` (the same bytes as one `global_store_b128`).
fn gstore128(b: &mut Builder, voff: u8, data: u8, sbase: u8, imm: u32) -> R {
    for i in 0..4u8 {
        b.push(
            Instruction::new(
                format!(
                    "global_store_b32 v{voff}, v{}, s[{sbase}:{}]{}",
                    data + i,
                    sbase + 1,
                    off(imm + 4 * u32::from(i))
                ),
                vec![],
                vec![v(voff), v(data + i), sr(sbase, 2)],
            )
            .memory(MemoryClass::VmemStore),
        )?;
    }
    Ok(())
}
fn dload128(b: &mut Builder, slot: usize, dst: u8, addr: u8, imm: u32) -> R {
    b.ds_load(
        slot,
        Instruction::new(
            format!("ds_load_b128 {}, v{addr}{}", vr(dst, 4), off(imm)),
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
            format!("ds_store_{kind} v{addr}, {data}{}", off(imm)),
            vec![],
            vec![v(addr), data],
        )
        .memory(MemoryClass::DsStore),
    )
}
fn smem(b: &mut Builder, text: &str, dst: RegRef) -> R {
    b.push(Instruction::new(text, vec![dst], vec![sr(KARG, 2)]).memory(MemoryClass::SmemLoad))
}
pub(super) fn wait_alu(b: &mut Builder, what: &str) -> R {
    op(b, format!("s_wait_alu {what}"), &[], &[])
}
fn vmov0(b: &mut Builder, r: u8) -> R {
    op(b, format!("v_mov_b32_e32 v{r}, 0"), &[v(r)], &[])
}
/// v_wmma_f32_16x16x16_f16 dst, a, b, (dst | 0): A is the 4-register first operand.
fn wmma(b: &mut Builder, dst: u8, a: u8, bb: u8, acc: bool) -> R {
    let c = if acc {
        format!("{}", vr(dst, 8))
    } else {
        "0".into()
    };
    let mut uses = vec![vr(a, 4), vr(bb, 4)];
    if acc {
        uses.push(vr(dst, 8));
    }
    op(
        b,
        format!(
            "v_wmma_f32_16x16x16_f16 {}, {}, {}, {c}",
            vr(dst, 8),
            vr(a, 4),
            vr(bb, 4)
        ),
        &[vr(dst, 8)],
        &uses,
    )
}
/// hipcc's correctly rounded f32 division num/den (`v_div_scale` / `v_rcp` / fma
/// Newton steps / `v_div_fmas` / `v_div_fixup`), op for op. Uses VCC.
pub(super) fn fdiv(
    b: &mut Builder,
    dst: u8,
    num: &str,
    den: &str,
    uses: &[RegRef],
    t: [u8; 4],
) -> R {
    let [sc, r, q, e] = t;
    op(
        b,
        format!("v_div_scale_f32 v{sc}, null, {den}, {den}, {num}"),
        &[v(sc)],
        uses,
    )?;
    op(b, format!("v_rcp_f32_e32 v{r}, v{sc}"), &[v(r)], &[v(sc)])?;
    op(
        b,
        format!("v_fma_f32 v{e}, -v{sc}, v{r}, 1.0"),
        &[v(e)],
        &[v(sc), v(r)],
    )?;
    op(
        b,
        format!("v_fmac_f32_e32 v{r}, v{e}, v{r}"),
        &[v(r)],
        &[v(r), v(e)],
    )?;
    op(
        b,
        format!("v_div_scale_f32 v{e}, vcc_lo, {num}, {den}, {num}"),
        &[v(e)],
        uses,
    )?;
    op(
        b,
        format!("v_mul_f32_e32 v{q}, v{e}, v{r}"),
        &[v(q)],
        &[v(e), v(r)],
    )?;
    op(
        b,
        format!("v_fma_f32 v{dst}, -v{sc}, v{q}, v{e}"),
        &[v(dst)],
        &[v(sc), v(q), v(e)],
    )?;
    op(
        b,
        format!("v_fmac_f32_e32 v{q}, v{dst}, v{r}"),
        &[v(q)],
        &[v(q), v(dst), v(r)],
    )?;
    op(
        b,
        format!("v_fma_f32 v{sc}, -v{sc}, v{q}, v{e}"),
        &[v(sc)],
        &[v(sc), v(q), v(e)],
    )?;
    wait_alu(b, "depctr_va_vcc(0)")?;
    op(
        b,
        format!("v_div_fmas_f32 v{sc}, v{sc}, v{r}, v{q}"),
        &[v(sc)],
        &[v(sc), v(r), v(q)],
    )?;
    let mut fix = vec![v(sc)];
    fix.extend_from_slice(uses);
    op(
        b,
        format!("v_div_fixup_f32 v{dst}, v{sc}, {den}, {num}"),
        &[v(dst)],
        &fix,
    )
}
/// One f32 -> f16 conversion into the low or high half of `dst`.
pub(super) fn cvt_half(b: &mut Builder, dst: u8, high: bool, src: u8) -> R {
    let h = if high { "h" } else { "l" };
    op(
        b,
        format!("v_cvt_f16_f32_e32 v{dst}.{h}, v{src}"),
        &[v(dst)],
        &[v(src), v(dst)],
    )
}
/// Byte `k` (0..=3) of `word`, sign-extended, as an f32 (the oracle's `(float)(int8_t)`):
/// exact integer sign extension of the unsigned byte `x` as `(x ^ 0x80) - 0x80` (the emulator
/// models no `v_bfe_i32` / `v_ashrrev_i32`), then `v_cvt_f32_i32`.
fn code_f32(b: &mut Builder, dst: u8, word: u8, k: u8) -> R {
    op(
        b,
        format!("v_bfe_u32 v{dst}, v{word}, {}, 8", 8 * k),
        &[v(dst)],
        &[v(word)],
    )?;
    op(
        b,
        format!("v_xor_b32_e32 v{dst}, {}, v{dst}", lit(0x80)),
        &[v(dst)],
        &[v(dst)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{dst}, {}, v{dst}", lit(0xffff_ff80)),
        &[v(dst)],
        &[v(dst)],
    )?;
    op(
        b,
        format!("v_cvt_f32_i32_e32 v{dst}, v{dst}"),
        &[v(dst)],
        &[v(dst)],
    )
}
/// Two consecutive code bytes of `word` (`high` false: bytes 0,1; true: bytes 2,3) as two f32 in `dst, dst + 1`: one native
/// `v_cvt_pk_f32_fp8` (gfx1201 OCP E4M3FN; `e32` selects the low half, the high half needs the `e64` `op_sel:[1,0]`
/// form). NaN codes (`S.1111.111`) become the default NaN; every finite code is its exact OCP value, so the
/// following f32 product with the f16 scale is exact in f32 (<= 4 + 11 significand bits).
fn fp8_pair(b: &mut Builder, dst: u8, word: u8, high: bool) -> R {
    let text = if high {
        format!(
            "v_cvt_pk_f32_fp8_e64 {}, v{word}.h op_sel:[1,0]",
            vr(dst, 2)
        )
    } else {
        format!("v_cvt_pk_f32_fp8_e32 {}, v{word}.l", vr(dst, 2))
    };
    op(b, text, &[vr(dst, 2)], &[v(word)])
}
/// A loaded f16 scale (zero-extended dword) as an f32, unmodified.
fn scale_f32(b: &mut Builder, r: u8) -> R {
    op(
        b,
        format!("v_cvt_f32_f16_e64 v{r}, v{r}.l"),
        &[v(r)],
        &[v(r)],
    )
}
/// `dst = {f16(fma(scale_hi, code_hi, -0)) : f16(fma(scale_lo, code_lo, -0))}`: the oracle's
/// single-rounding Q8 dequantisation of two halves, as two `v_fma_mixlo_f16` into the
/// zero-initialised temporaries `ta`/`tb` and one byte permute (`ta.lo`, `tb.lo`).
fn dequant_pair(b: &mut Builder, dst: u8, lo: (u8, u8), hi: (u8, u8), ta: u8, tb: u8) -> R {
    for (t, (scale, code)) in [(ta, lo), (tb, hi)] {
        op(
            b,
            format!("v_fma_mixlo_f16 v{t}, v{scale}, v{code}, neg(0)"),
            &[v(t)],
            &[v(t), v(scale), v(code)],
        )?;
    }
    op(
        b,
        format!("v_perm_b32 v{dst}, v{tb}, v{ta}, 0x5040100"),
        &[v(dst)],
        &[v(tb), v(ta)],
    )
}
/// `exec &= mask`, saving the old exec (restore with [`exec_restore`]).
fn exec_and(b: &mut Builder, mask: u8) -> R {
    op(b, format!("s_mov_b32 s{SAVE}, exec_lo"), &[m(SAVE)], &[])?;
    op(
        b,
        format!("s_and_b32 exec_lo, exec_lo, s{mask}"),
        &[],
        &[m(mask)],
    )
}
fn exec_restore(b: &mut Builder) -> R {
    op(
        b,
        format!("s_or_b32 exec_lo, exec_lo, s{SAVE}"),
        &[],
        &[m(SAVE)],
    )
}
/// 64-bit `dst = base + lo + (hi << 32)` over SGPR pairs (`lo`/`hi` single SGPRs).
fn add64(b: &mut Builder, dst: u8, base: u8, lo: u8, hi: Option<u8>) -> R {
    sop(
        b,
        format!("s_add_co_u32 s{dst}, s{base}, s{lo}"),
        &[dst],
        &[base, lo],
    )?;
    match hi {
        Some(h) => sop(
            b,
            format!("s_add_co_ci_u32 s{}, s{}, s{h}", dst + 1, base + 1),
            &[dst + 1],
            &[base + 1, h],
        ),
        None => sop(
            b,
            format!("s_add_co_ci_u32 s{}, s{}, 0", dst + 1, base + 1),
            &[dst + 1],
            &[base + 1],
        ),
    }
}

// ------------------------------------------------------------------ plans
fn plan_direct(fp8: bool) -> Result<RegPlan, String> {
    let mut p = RegPlan::new(if fp8 { 224 } else { 256 }, 104)?;
    let w = || Live::Whole;
    let between = |a: &str, z: &str| Live::Between(a.into(), z.into());
    let pro = || between("entry", D_LOOP);
    let fill = || between(D_LOOP, D_QK);
    let tile = || between(D_QK, D_TURN);
    let epi = || between(D_EPI, D_END);
    // FP8 sequential-stage lifetimes: K-fill, QK, V-fill, softmax/PV.
    let kfill = || between(D_LOOP, D_KREADY);
    let qk = || between(D_KREADY, D_QK_DONE);
    let vfill = || between(D_QK_DONE, D_VREADY);
    let sm = || between(D_VREADY, D_TURN);
    p.v::<1>("tid", TID, pro())?;
    for (name, base) in [
        ("lds_k_store", V_LK),
        ("lds_v_store", V_LV),
        ("lds_frag_read", V_LF),
        ("global_k_off", V_GK),
        ("global_v_off", V_GV),
        ("q16_off", V_QO),
        ("row_pos", V_POS),
        ("kg8", V_KG8),
        ("m_run", V_M),
        ("l_run", V_L),
        ("fill_vk", V_VK),
        ("fill_vk2", V_VK2),
    ] {
        p.v::<1>(name, base, w())?;
    }
    if fp8 {
        p.v::<1>("global_k_scale_off", V_GSK, w())?;
        p.v::<1>("global_v_scale_off", V_GSV, w())?;
        p.v::<1>("spare", 15, w())?;
    } else {
        for n in 13..16u8 {
            p.v::<1>("spare", n, w())?;
        }
    }
    // Both codecs use v16..v47 as prologue temporaries before their tile lifetimes.
    for i in 0..4u8 {
        p.v::<8>("prologue_temp", S0 + 8 * i, pro())?;
    }
    if fp8 {
        // K-fill only: codes 16..47, scales 64..67, output ring 68..71, temps 72..79.
        for i in 0..4u8 {
            p.v::<8>("fill_k_codes", FK8 + 8 * i, kfill())?;
        }
        p.v::<4>("fill_k_scales", FSC8, kfill())?;
        p.v::<4>("fill_k_out_ring", FOUT8, kfill())?;
        p.v::<8>("fill_k_temp", FTMP8, kfill())?;
        // Scores span QK through softmax consumption.
        for i in 0..4u8 {
            p.v::<8>("score", S0 + 8 * i, between(D_KREADY, D_TURN))?;
        }
        for i in 0..4u8 {
            p.v::<4>("q_frag", QF + 4 * i, w())?;
        }
        // QK only: K fragments 64..79.
        for i in 0..4u8 {
            p.v::<4>("k_frag", KF + 4 * i, qk())?;
        }
        // V-fill streams one t' group: codes 64..79, scales 80..81, ring 84..87, temps 88..95.
        p.v::<8>("fill_v_codes_lo", VFC8, vfill())?;
        p.v::<8>("fill_v_codes_hi", VFC8 + 8, vfill())?;
        p.v::<2>("fill_v_scales", VFSC8, vfill())?;
        p.v::<4>("fill_v_out_ring", VFOUT8, vfill())?;
        p.v::<8>("fill_v_temp", VFTMP8, vfill())?;
        // Softmax/PV: temporaries 64..71, probabilities 80..83, V buffers 84..91.
        p.v::<8>("sm_temp", SMX8, sm())?;
        p.v::<4>("prow", SPRO8, sm())?;
        p.v::<4>("v_frag_a", SVFA8, sm())?;
        p.v::<4>("v_frag_b", SVFB8, sm())?;
        for i in 0..16u8 {
            p.v::<8>("o", O8 + 8 * i, w())?;
        }
        // Epilogue temporaries.
        p.v::<8>("epilogue_temp", S0, epi())?;
        p.v::<8>("epilogue_temp", S0 + 8, epi())?;
    } else {
        // Fill phase.
        for i in 0..4u8 {
            p.v::<8>("fill_k_codes", FK + 8 * i, fill())?;
        }
        p.v::<8>("fill_scales", FSC, fill())?;
        for i in 0..4u8 {
            p.v::<8>("fill_v_codes", FV + 16 * (i / 2) + 8 * (i % 2), fill())?;
        }
        p.v::<8>("fill_out_ring", FOUT, fill())?;
        p.v::<8>("fill_temp", FTMP, fill())?;
        // Tile phase.
        for i in 0..4u8 {
            p.v::<8>("score", S0 + 8 * i, tile())?;
        }
        for i in 0..4u8 {
            p.v::<4>("q_frag", QF + 4 * i, w())?;
        }
        for i in 0..4u8 {
            p.v::<4>("k_frag", KF + 4 * i, tile())?;
        }
        p.v::<4>("prow", PROW, tile())?;
        p.v::<4>("v_frag_a", VFA, tile())?;
        p.v::<4>("v_frag_b", VFB, tile())?;
        for (name, base) in [
            ("row_max", V_MX),
            ("row_max_peer", V_SWZ),
            ("m_new", V_MN),
            ("alpha", V_AL),
            ("sum", V_SUM),
            ("peer", V_T1),
            ("key_base", V_KEY),
            ("key_j", V_KJ),
        ] {
            p.v::<1>(name, base, tile())?;
        }
        for i in 0..16u8 {
            p.v::<8>("o", O + 8 * i, w())?;
        }
        // Epilogue temporaries.
        p.v::<8>("epilogue_temp", S0, epi())?;
        p.v::<8>("epilogue_temp", S0 + 8, epi())?;
    }
    // SGPRs.
    p.s::<2>("kernarg_ptr", KARG, w())?;
    p.s::<8>("kernarg_ptrs", SQ16, w())?;
    p.s::<2>("positions", SPOS, w())?;
    p.s::<4>("dims", NHEADS, w())?;
    for n in [SCALE, QBASE, KVH, GMAX, GMIN, KT, REM, REMMIN, PSEL, WAVE] {
        p.s::<1>("scalar", n, w())?;
    }
    for n in T {
        p.s::<1>("scratch", n, w())?;
    }
    for n in [KB, VB, KB0, VB0] {
        p.s::<2>("ptr_pair", n, w())?;
    }
    for n in [QOK, SAVE, CMPA, CMPB] {
        p.s::<2>("mask", n, w())?;
    }
    for j in 0..8u8 {
        p.s::<2>("mask", mj(j), w())?;
    }
    for t in 0..4u8 {
        p.s::<2>("mask", fm(t), w())?;
    }
    Ok(p)
}

// ------------------------------------------------------------------ direct kernel
/// Wave-uniform exits, immutable LDS addresses and the per-launch state.
fn prologue(b: &mut Builder, fp8: bool) -> R {
    smem(
        b,
        &format!("s_load_b256 s[{SQ16}:{}], s[0:1], 0x0", SQ16 + 7),
        sr(SQ16, 8),
    )?;
    smem(
        b,
        &format!("s_load_b64 s[{SPOS}:{}], s[0:1], 0x20", SPOS + 1),
        sr(SPOS, 2),
    )?;
    smem(
        b,
        &format!("s_load_b128 s[{NHEADS}:{}], s[0:1], 0x28", NHEADS + 3),
        sr(NHEADS, 4),
    )?;
    smem(b, &format!("s_load_b32 s{SCALE}, s[0:1], 0x38"), s(SCALE))?;
    // Entry block: every LDS address VGPR comes from the work-item id alone.
    let (t0, a, bb, c, d, e, f, g, h, x, y, w) = (
        16u8, 17u8, 18u8, 19u8, 20u8, 21u8, 22u8, 23u8, 24u8, 25u8, 26u8, 27u8,
    );
    op(
        b,
        format!("v_and_b32_e32 v{t0}, {}, v{TID}", lit(127)),
        &[v(t0)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{a}, 7, v{t0}"),
        &[v(a)],
        &[v(t0)],
    )?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{V_VK}, 3, v{t0}"),
        &[v(V_VK)],
        &[v(t0)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{bb}, 4, v{V_VK}"),
        &[v(bb)],
        &[v(V_VK)],
    )?;
    // K store base: 4096*(t0&7) + 16*(t0>>3).
    op(
        b,
        format!("v_lshl_add_u32 v{V_LK}, v{a}, 12, v{bb}"),
        &[v(V_LK)],
        &[v(a), v(bb)],
    )?;
    // V store base: 8192*(t0>>6) + 1024*(t0&7) + 256*((t0>>5)&1) + 4*((t0>>3)&3).
    op(
        b,
        format!("v_lshrrev_b32_e32 v{c}, 6, v{t0}"),
        &[v(c)],
        &[v(t0)],
    )?;
    op(b, format!("v_bfe_u32 v{d}, v{t0}, 5, 1"), &[v(d)], &[v(t0)])?;
    op(b, format!("v_bfe_u32 v{e}, v{t0}, 3, 2"), &[v(e)], &[v(t0)])?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{e}, 2, v{e}"),
        &[v(e)],
        &[v(e)],
    )?;
    op(
        b,
        format!("v_lshl_add_u32 v{f}, v{a}, 10, v{e}"),
        &[v(f)],
        &[v(a), v(e)],
    )?;
    op(
        b,
        format!("v_lshl_add_u32 v{f}, v{d}, 8, v{f}"),
        &[v(f)],
        &[v(d), v(f)],
    )?;
    op(
        b,
        format!("v_lshl_add_u32 v{V_LV}, v{c}, 13, v{f}"),
        &[v(V_LV)],
        &[v(c), v(f)],
    )?;
    // Fragment read base 16 * lane.
    op(
        b,
        format!("v_and_b32_e32 v{g}, 31, v{TID}"),
        &[v(g)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{V_LF}, 4, v{g}"),
        &[v(V_LF)],
        &[v(g)],
    )?;
    if fp8 {
        // Native-fp8 rows (516 B): code offset 516*vk + 32*a (+ 256*head below), scale offset 516*vk (+ 512 + 2*head
        // below); V pairs 1032*vk. The head terms join once the KV head is known.
        op(
            b,
            format!(
                "v_mul_u32_u24_e32 v{V_GSK}, {}, v{V_VK}",
                lit(FP8_ROW_BYTES)
            ),
            &[v(V_GSK)],
            &[v(V_VK)],
        )?;
        op(
            b,
            format!("v_lshlrev_b32_e32 v{y}, 5, v{a}"),
            &[v(y)],
            &[v(a)],
        )?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{V_GK}, v{V_GSK}, v{y}"),
            &[v(V_GK)],
            &[v(V_GSK), v(y)],
        )?;
        op(
            b,
            format!("v_lshlrev_b32_e32 v{V_VK2}, 1, v{V_VK}"),
            &[v(V_VK2)],
            &[v(V_VK)],
        )?;
        op(
            b,
            format!(
                "v_mul_u32_u24_e32 v{V_GSV}, {}, v{V_VK}",
                lit(2 * FP8_ROW_BYTES)
            ),
            &[v(V_GSV)],
            &[v(V_VK)],
        )?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{V_GV}, v{V_GSV}, v{y}"),
            &[v(V_GV)],
            &[v(V_GSV), v(y)],
        )?;
    } else {
        // Global fill offsets: K row stride 544, V pairs 1088, block stride 34.
        op(
            b,
            format!("v_mul_u32_u24_e32 v{x}, {}, v{V_VK}", lit(544)),
            &[v(x)],
            &[v(V_VK)],
        )?;
        op(
            b,
            format!("v_mul_u32_u24_e32 v{y}, 34, v{a}"),
            &[v(y)],
            &[v(a)],
        )?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{V_GK}, v{x}, v{y}"),
            &[v(V_GK)],
            &[v(x), v(y)],
        )?;
        op(
            b,
            format!("v_lshlrev_b32_e32 v{V_VK2}, 1, v{V_VK}"),
            &[v(V_VK2)],
            &[v(V_VK)],
        )?;
        op(
            b,
            format!("v_mul_u32_u24_e32 v{x}, {}, v{V_VK}", lit(1088)),
            &[v(x)],
            &[v(V_VK)],
        )?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{V_GV}, v{x}, v{y}"),
            &[v(V_GV)],
            &[v(x), v(y)],
        )?;
    }
    // Lane quantities: kg, 8*kg, 16*kg, lane & 7, (lane >> 3) & 1, wave.
    op(
        b,
        format!("v_lshrrev_b32_e32 v{h}, 4, v{g}"),
        &[v(h)],
        &[v(g)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{V_KG8}, 3, v{h}"),
        &[v(V_KG8)],
        &[v(h)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{h}, 4, v{h}"),
        &[v(h)],
        &[v(h)],
    )?;
    op(b, format!("v_and_b32_e32 v{x}, 7, v{g}"), &[v(x)], &[v(g)])?;
    op(b, format!("v_bfe_u32 v{y}, v{g}, 3, 1"), &[v(y)], &[v(g)])?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{w}, 5, v{TID}"),
        &[v(w)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{WAVE}, v{w}"),
        &[s(WAVE)],
        &[v(w)],
    )?;
    b.wait(Counter::Km, 0)?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    // Workgroup-uniform shape and grid guards, before any data access or barrier.
    for (reg, want) in [(NHEADS, 16u32), (NKV, 2), (HDIM, 256)] {
        sop(
            b,
            format!("s_cmp_lg_u32 s{reg}, {}", lit(want)),
            &[],
            &[reg],
        )?;
        op(b, format!("s_cbranch_scc1 {D_END}"), &[], &[])?;
    }
    op(b, "s_cmp_gt_u32 ttmp7, 1", &[], &[])?;
    op(b, format!("s_cbranch_scc1 {D_END}"), &[], &[])?;
    op(b, format!("s_mov_b32 s{KVH}, ttmp7"), &[s(KVH)], &[])?;
    op(
        b,
        format!("s_lshl_b32 s{QBASE}, ttmp9, 3"),
        &[s(QBASE)],
        &[],
    )?;
    sop(
        b,
        format!("s_cmp_ge_i32 s{QBASE}, s{BATCH}"),
        &[],
        &[QBASE, BATCH],
    )?;
    op(b, format!("s_cbranch_scc1 {D_END}"), &[], &[])?;
    // Scalar launch state.
    sop(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[PSEL], &[])?;
    sop(
        b,
        format!("s_add_co_i32 s{}, s{BATCH}, -1", T[0]),
        &[T[0]],
        &[BATCH],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{}, s{KVH}, 3", T[1]),
        &[T[1]],
        &[KVH],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{}, s{WAVE}, 1", T[2]),
        &[T[2]],
        &[WAVE],
    )?;
    sop(
        b,
        format!("s_add_co_i32 s{0}, s{0}, s{1}", T[1], T[2]),
        &[T[1]],
        &[T[1], T[2]],
    )?;
    if fp8 {
        // One 64-bit tile base per cache (`k_cache + ktile*516`); the head enters through the lane offsets:
        // code head base 256*kvh in T[3], scale head base 512 + 2*kvh in T[2] (T[2] is dead after T[1] above).
        sop(
            b,
            format!("s_lshl_b32 s{}, s{KVH}, 8", T[3]),
            &[T[3]],
            &[KVH],
        )?;
        sop(
            b,
            format!("s_lshl_b32 s{}, s{KVH}, 1", T[2]),
            &[T[2]],
            &[KVH],
        )?;
        sop(
            b,
            format!(
                "s_add_co_i32 s{0}, s{0}, {1}",
                T[2],
                imm_i32(FP8_SCALE_OFFSET as i32)
            ),
            &[T[2]],
            &[T[2]],
        )?;
    } else {
        sop(
            b,
            format!("s_mul_i32 s{}, s{KVH}, {}", T[3], lit(272)),
            &[T[3]],
            &[KVH],
        )?;
        add64(b, KB0, SK, T[3], None)?;
        add64(b, VB0, SV, T[3], None)?;
    }
    sop(b, format!("s_mov_b32 s{KT}, 0"), &[KT], &[])?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    if fp8 {
        // Head terms of the lane offsets: codes at + 256*kvh, scales at + 512 + 2*kvh (separate offsets).
        for r in [V_GK, V_GV] {
            op(
                b,
                format!("v_add_nc_u32_e32 v{r}, s{}, v{r}", T[3]),
                &[v(r)],
                &[s(T[3]), v(r)],
            )?;
        }
        for r in [V_GSK, V_GSV] {
            op(
                b,
                format!("v_add_nc_u32_e32 v{r}, s{}, v{r}", T[2]),
                &[v(r)],
                &[s(T[2]), v(r)],
            )?;
        }
    }
    // Query row, validity, head and the q16 offset of this lane.
    let (qr, qc, hh, q13, po) = (28u8, 29u8, 30u8, 31u8, 32u8);
    op(
        b,
        format!("v_add_nc_u32_e32 v{qr}, s{QBASE}, v{x}"),
        &[v(qr)],
        &[s(QBASE), v(x)],
    )?;
    op(
        b,
        format!("v_cmp_gt_i32_e64 s{QOK}, s{BATCH}, v{qr}"),
        &[m(QOK)],
        &[s(BATCH), v(qr)],
    )?;
    op(
        b,
        format!("v_min_u32_e32 v{qc}, s{}, v{qr}", T[0]),
        &[v(qc)],
        &[s(T[0]), v(qr)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{hh}, s{}, v{y}", T[1]),
        &[v(hh)],
        &[s(T[1]), v(y)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{q13}, 13, v{qr}"),
        &[v(q13)],
        &[v(qr)],
    )?;
    op(
        b,
        format!("v_lshl_add_u32 v{q13}, v{hh}, 9, v{q13}"),
        &[v(q13)],
        &[v(hh), v(q13)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{V_QO}, v{q13}, v{h}"),
        &[v(V_QO)],
        &[v(q13), v(h)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{po}, 2, v{qc}"),
        &[v(po)],
        &[v(qc)],
    )?;
    gload(b, "b32", v(V_POS), po, SPOS, 0)?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    // Group bounds, every wave (helper included): lanes 0..7 with a live query hold their
    // position, every other lane the sentinels (max -1, min INT_MAX). The position of a live
    // lane is the one loaded above (its clamped row is its own row, so no access is out of range).
    // Then lane XOR 16, 8, 4, 2, 1 integer max / min and readfirstlane: the same scalar
    // gmax / gmin in all five waves.
    let (wmx, wmn, ta, tb) = (33u8, 34u8, 35u8, 36u8);
    op(
        b,
        format!("v_cmp_gt_u32_e64 s{CMPA}, 8, v{g}"),
        &[m(CMPA)],
        &[v(g)],
    )?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    op(
        b,
        format!("s_and_b32 s{CMPA}, s{CMPA}, s{QOK}"),
        &[m(CMPA)],
        &[m(CMPA), m(QOK)],
    )?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{wmx}, -1, v{V_POS}, s{CMPA}"),
        &[v(wmx)],
        &[v(V_POS), m(CMPA)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{wmn}, 0x7fffffff, v{V_POS}, s{CMPA}"),
        &[v(wmn)],
        &[v(V_POS), m(CMPA)],
    )?;
    for k in [16u32, 8, 4, 2, 1] {
        for (src, tmp) in [(wmx, ta), (wmn, tb)] {
            if k == 16 {
                op(
                    b,
                    format!("v_permlanex16_b32 v{tmp}, v{src}, s{PSEL}, 0xfedcba98"),
                    &[v(tmp)],
                    &[v(src), s(PSEL)],
                )?;
            } else {
                op(
                    b,
                    format!(
                        "v_mov_b32_dpp v{tmp}, v{src} row_xmask:{k} row_mask:0xf bank_mask:0xf"
                    ),
                    &[v(tmp)],
                    &[v(src)],
                )?;
            }
        }
        op(
            b,
            format!("v_max_i32_e32 v{wmx}, v{wmx}, v{ta}"),
            &[v(wmx)],
            &[v(wmx), v(ta)],
        )?;
        op(
            b,
            format!("v_cmp_lt_i32_e64 s{CMPB}, v{wmn}, v{tb}"),
            &[m(CMPB)],
            &[v(wmn), v(tb)],
        )?;
        wait_alu(b, "depctr_va_sdst(0)")?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{wmn}, v{tb}, v{wmn}, s{CMPB}"),
            &[v(wmn)],
            &[v(tb), v(wmn), m(CMPB)],
        )?;
    }
    op(
        b,
        format!("v_readfirstlane_b32 s{GMAX}, v{wmx}"),
        &[s(GMAX)],
        &[v(wmx)],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{GMIN}, v{wmn}"),
        &[s(GMIN)],
        &[v(wmn)],
    )?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    // Invalid query lanes get position -1: no key is <= it.
    op(
        b,
        format!("v_cndmask_b32_e64 v{V_POS}, -1, v{V_POS}, s{QOK}"),
        &[v(V_POS)],
        &[v(V_POS), m(QOK)],
    )?;
    // Fresh online state: O = +0, m = -inf, l = +0; Q fragments zero (invalid lanes stay zero).
    for r in 0..128u8 {
        vmov0(b, regs(fp8).o + r)?;
    }
    for r in 0..16u8 {
        vmov0(b, QF + r)?;
    }
    op(
        b,
        format!("v_mov_b32_e32 v{V_M}, 0xff800000"),
        &[v(V_M)],
        &[],
    )?;
    vmov0(b, V_L)?;
    // No live tile (gmax < 0): the oracle loop body never runs.
    sop(b, format!("s_cmp_lt_i32 s{GMAX}, 0"), &[], &[GMAX])?;
    op(b, format!("s_cbranch_scc1 {D_EPI}"), &[], &[])
}

/// Per-tile scalars: remaining rows and the 64-bit K/V tile bases.
fn tile_setup(b: &mut Builder, fp8: bool) -> R {
    // Key stride of the K/V caches (Q8: 2 heads x 8 blocks x 34 B, fp8: 516 B token rows) and the base pairs:
    // Q8 folds the head into the base (`KB0`/`VB0`), fp8 keeps one head-free base and offsets lanes instead.
    let (stride, kb0, vb0) = if fp8 {
        (FP8_ROW_BYTES, SK, SV)
    } else {
        (544, KB0, VB0)
    };
    sop(
        b,
        format!("s_sub_co_i32 s{REM}, s{GMAX}, s{KT}"),
        &[REM],
        &[GMAX, KT],
    )?;
    sop(
        b,
        format!("s_sub_co_i32 s{REMMIN}, s{GMIN}, s{KT}"),
        &[REMMIN],
        &[GMIN, KT],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{}, s{KT}, {}", T[0], lit(stride)),
        &[T[0]],
        &[KT],
    )?;
    sop(
        b,
        format!("s_mul_hi_u32 s{}, s{KT}, {}", T[1], lit(stride)),
        &[T[1]],
        &[KT],
    )?;
    add64(b, KB, kb0, T[0], Some(T[1]))?;
    add64(b, VB, vb0, T[0], Some(T[1]))?;
    wait_alu(b, "depctr_sa_sdst(0)")
}

/// Cooperative KT64 fill by the 128 filler threads (waves 0..3); wave 4 skips the
/// stores but not the rendezvous. Masked keys produce exact +0 halves.
/// Q8 fills K and V jointly (two planes); FP8 fills K then (after QK) V into
/// the single reused stage via [`fill_k_fp8`] / [`fill_v_fp8`].
fn fill(b: &mut Builder, fp8: bool) -> R {
    let _ = fp8;
    sop(b, format!("s_cmp_ge_u32 s{WAVE}, 4"), &[], &[WAVE])?;
    op(b, format!("s_cbranch_scc1 {D_FILL_DONE}"), &[], &[])?;
    // K masks: key vk + 16t is live iff vk < rem + 1 - 16t.
    for t in 0..4u8 {
        sop(
            b,
            format!(
                "s_add_co_i32 s{}, s{REM}, {}",
                T[t as usize],
                imm_i32(1 - 16 * i32::from(t))
            ),
            &[T[t as usize]],
            &[REM],
        )?;
    }
    wait_alu(b, "depctr_sa_sdst(0)")?;
    for t in 0..4u8 {
        op(
            b,
            format!("v_cmp_gt_i32_e64 s{}, s{}, v{V_VK}", fm(t), T[t as usize]),
            &[m(fm(t))],
            &[s(T[t as usize]), v(V_VK)],
        )?;
    }
    wait_alu(b, "depctr_va_sdst(0)")?;
    for r in 0..32u8 {
        vmov0(b, FK + r)?;
    }
    for t in 0..4u8 {
        vmov0(b, FSC + t)?;
    }
    for t in 0..4u8 {
        exec_and(b, fm(t))?;
        let row = 8704 * u32::from(t);
        gload(b, "u16", v(FSC + t), V_GK, KB, row)?;
        for w in 0..8u8 {
            gload(
                b,
                "b32",
                v(FK + 8 * t + w),
                V_GK,
                KB,
                row + 2 + 4 * u32::from(w),
            )?;
        }
        exec_restore(b)?;
    }
    // V masks: pair m = vk + 16t', even key live iff 2vk < rem + 1 - 32t', odd iff 2vk < rem - 32t'.
    for t in 0..2u8 {
        let a = T[4 + 2 * t as usize];
        let c = T[5 + 2 * t as usize];
        sop(
            b,
            format!(
                "s_add_co_i32 s{a}, s{REM}, {}",
                imm_i32(1 - 32 * i32::from(t))
            ),
            &[a],
            &[REM],
        )?;
        sop(
            b,
            format!("s_add_co_i32 s{c}, s{REM}, {}", imm_i32(-32 * i32::from(t))),
            &[c],
            &[REM],
        )?;
    }
    wait_alu(b, "depctr_sa_sdst(0)")?;
    for t in 0..2u8 {
        let a = T[4 + 2 * t as usize];
        let c = T[5 + 2 * t as usize];
        op(
            b,
            format!("v_cmp_gt_i32_e64 s{}, s{a}, v{V_VK2}", fm(2 * t)),
            &[m(fm(2 * t))],
            &[s(a), v(V_VK2)],
        )?;
        op(
            b,
            format!("v_cmp_gt_i32_e64 s{}, s{c}, v{V_VK2}", fm(2 * t + 1)),
            &[m(fm(2 * t + 1))],
            &[s(c), v(V_VK2)],
        )?;
    }
    wait_alu(b, "depctr_va_sdst(0)")?;
    for r in 0..4u8 {
        vmov0(b, FSC + 4 + r)?;
    }
    for r in 0..32u8 {
        vmov0(b, FV + r)?;
    }
    for t in 0..2u8 {
        for (half, mask) in [(0u8, fm(2 * t)), (1u8, fm(2 * t + 1))] {
            let scale = FSC + 4 + 2 * t + half;
            let codes = FV + 16 * t + 8 * half;
            exec_and(b, mask)?;
            let block = 17_408 * u32::from(t) + 544 * u32::from(half);
            gload(b, "u16", v(scale), V_GV, VB, block)?;
            for w in 0..8u8 {
                gload(
                    b,
                    "b32",
                    v(codes + w),
                    V_GV,
                    VB,
                    block + 2 + 4 * u32::from(w),
                )?;
            }
            exec_restore(b)?;
        }
    }
    // The half-writes below preserve the other half of their destination: define the whole
    // temporary block first.
    for r in 0..8u8 {
        vmov0(b, FTMP + r)?;
    }
    // K dequantisation into the fragment plane: one 8-byte store per code quad.
    let mut quad = 0u8;
    for t in 0..4u8 {
        let scale = FSC + t;
        scale_f32(b, scale)?;
        for w in 0..8u8 {
            let word = FK + 8 * t + w;
            let [a0, a1, a2, a3] = [FTMP, FTMP + 1, FTMP + 2, FTMP + 3];
            for (k, a) in [a0, a1, a2, a3].into_iter().enumerate() {
                code_f32(b, a, word, k as u8)?;
            }
            let out = FOUT + 2 * (quad % 4);
            quad += 1;
            dequant_pair(b, out, (scale, a0), (scale, a1), FTMP + 4, FTMP + 5)?;
            dequant_pair(b, out + 1, (scale, a2), (scale, a3), FTMP + 4, FTMP + 5)?;
            let imm = 512 * u32::from(t)
                + 2048 * u32::from(w >> 2)
                + 256 * u32::from((w >> 1) & 1)
                + 8 * u32::from(w & 1);
            dstore(b, L_K, "b64", V_LK, vr(out, 2), imm)?;
        }
    }
    // V dequantisation: (even key, odd key) halves of one dimension share a dword.
    let mut word_no = 0u8;
    for t in 0..2u8 {
        let (s0, s1) = (FSC + 4 + 2 * t, FSC + 5 + 2 * t);
        for r in [s0, s1] {
            scale_f32(b, r)?;
        }
        for w in 0..8u8 {
            let (u0, u1) = (FV + 16 * t + w, FV + 16 * t + 8 + w);
            for c in 0..4u8 {
                let base = FTMP + 4 * (word_no % 2);
                let out = FOUT + (word_no % 8);
                word_no = word_no.wrapping_add(1);
                code_f32(b, base, u0, c)?;
                code_f32(b, base + 1, u1, c)?;
                dequant_pair(b, out, (s0, base), (s1, base + 1), base + 2, base + 3)?;
                let imm = 32_768
                    + 16_384 * u32::from(t)
                    + 512 * u32::from(w >> 2)
                    + 64 * u32::from(w & 3)
                    + 16 * u32::from(c);
                dstore(b, L_V, "b32", V_LV, v(out), imm)?;
            }
        }
    }
    Ok(())
}
/// FP8 K-fill into the reused stage (base 0): codes 16..47, scales 64..67,
/// output ring 68..71 as pairs (68,69),(70,71), temporaries 72..79.
/// Same global loads and conversion/dequant sequence as the Q8-joined path's
/// fp8 branch, with the plane base removed from the store immediates.
fn fill_k_fp8(b: &mut Builder) -> R {
    sop(b, format!("s_cmp_ge_u32 s{WAVE}, 4"), &[], &[WAVE])?;
    op(b, format!("s_cbranch_scc1 {D_KFILL_DONE}"), &[], &[])?;
    // K masks: key vk + 16t is live iff vk < rem + 1 - 16t.
    for t in 0..4u8 {
        sop(
            b,
            format!(
                "s_add_co_i32 s{}, s{REM}, {}",
                T[t as usize],
                imm_i32(1 - 16 * i32::from(t))
            ),
            &[T[t as usize]],
            &[REM],
        )?;
    }
    wait_alu(b, "depctr_sa_sdst(0)")?;
    for t in 0..4u8 {
        op(
            b,
            format!("v_cmp_gt_i32_e64 s{}, s{}, v{V_VK}", fm(t), T[t as usize]),
            &[m(fm(t))],
            &[s(T[t as usize]), v(V_VK)],
        )?;
    }
    wait_alu(b, "depctr_va_sdst(0)")?;
    for r in 0..32u8 {
        vmov0(b, FK8 + r)?;
    }
    for t in 0..4u8 {
        vmov0(b, FSC8 + t)?;
    }
    for t in 0..4u8 {
        exec_and(b, fm(t))?;
        // Row 516*16*t; the scale offset (`512 + 2*head`) and the code offset (`256*head + 32*a`) are separate.
        let row = 16 * FP8_ROW_BYTES * u32::from(t);
        gload(b, "u16", v(FSC8 + t), V_GSK, KB, row)?;
        for w in 0..8u8 {
            gload(
                b,
                "b32",
                v(FK8 + 8 * t + w),
                V_GK,
                KB,
                row + 4 * u32::from(w),
            )?;
        }
        exec_restore(b)?;
    }
    // The half-writes below preserve the other half of their destination: define the whole
    // temporary block first.
    for r in 0..8u8 {
        vmov0(b, FTMP8 + r)?;
    }
    // K dequantisation into the fragment plane: one 8-byte store per code quad.
    for t in 0..4u8 {
        let scale = FSC8 + t;
        scale_f32(b, scale)?;
        for w in 0..8u8 {
            let word = FK8 + 8 * t + w;
            let (a0, a2) = (FTMP8, FTMP8 + 2);
            fp8_pair(b, a0, word, false)?;
            fp8_pair(b, a2, word, true)?;
            // Ring holds two quads: out (68,69) then out+1 (70,71), alternating per quad.
            let pair = (4 * t + w) % 2;
            let out = FOUT8 + 2 * pair;
            dequant_pair(b, out, (scale, a0), (scale, a0 + 1), FTMP8 + 4, FTMP8 + 5)?;
            dequant_pair(
                b,
                out + 1,
                (scale, a2),
                (scale, a2 + 1),
                FTMP8 + 4,
                FTMP8 + 5,
            )?;
            let imm = 512 * u32::from(t)
                + 2048 * u32::from(w >> 2)
                + 256 * u32::from((w >> 1) & 1)
                + 8 * u32::from(w & 1);
            dstore(b, L_STAGE, "b64", V_LK, vr(out, 2), imm)?;
        }
    }
    Ok(())
}

/// FP8 V-fill into the reused stage (base 0), streamed one t' group at a time:
/// codes 64..79, scales 80..81, output ring 84..87, temporaries 88..95.
/// Same global loads and conversion/dequant sequence as the Q8-joined path's
/// fp8 branch, with the 32768 plane base removed from the store immediates.
fn fill_v_fp8(b: &mut Builder) -> R {
    sop(b, format!("s_cmp_ge_u32 s{WAVE}, 4"), &[], &[WAVE])?;
    op(b, format!("s_cbranch_scc1 {D_VFILL_DONE}"), &[], &[])?;
    for tp in 0..2u8 {
        // V masks: pair m = vk + 16t', even key live iff 2vk < rem + 1 - 32t', odd iff 2vk < rem - 32t'.
        let a = T[4 + 2 * tp as usize];
        let c = T[5 + 2 * tp as usize];
        sop(
            b,
            format!(
                "s_add_co_i32 s{a}, s{REM}, {}",
                imm_i32(1 - 32 * i32::from(tp))
            ),
            &[a],
            &[REM],
        )?;
        sop(
            b,
            format!(
                "s_add_co_i32 s{c}, s{REM}, {}",
                imm_i32(-32 * i32::from(tp))
            ),
            &[c],
            &[REM],
        )?;
        wait_alu(b, "depctr_sa_sdst(0)")?;
        op(
            b,
            format!("v_cmp_gt_i32_e64 s{}, s{a}, v{V_VK2}", fm(2 * tp)),
            &[m(fm(2 * tp))],
            &[s(a), v(V_VK2)],
        )?;
        op(
            b,
            format!("v_cmp_gt_i32_e64 s{}, s{c}, v{V_VK2}", fm(2 * tp + 1)),
            &[m(fm(2 * tp + 1))],
            &[s(c), v(V_VK2)],
        )?;
        wait_alu(b, "depctr_va_sdst(0)")?;
        for r in 0..2u8 {
            vmov0(b, VFSC8 + r)?;
        }
        for r in 0..16u8 {
            vmov0(b, VFC8 + r)?;
        }
        for (half, mask) in [(0u8, fm(2 * tp)), (1u8, fm(2 * tp + 1))] {
            let scale = VFSC8 + half;
            let codes = VFC8 + 8 * half;
            exec_and(b, mask)?;
            // Key pair stride 2 * 516; the odd key is one row (516 B) further.
            let block = 32 * FP8_ROW_BYTES * u32::from(tp) + FP8_ROW_BYTES * u32::from(half);
            gload(b, "u16", v(scale), V_GSV, VB, block)?;
            for w in 0..8u8 {
                gload(b, "b32", v(codes + w), V_GV, VB, block + 4 * u32::from(w))?;
            }
            exec_restore(b)?;
        }
        // The half-writes below preserve the other half of their destination: define the whole
        // temporary block first.
        for r in 0..8u8 {
            vmov0(b, VFTMP8 + r)?;
        }
        // V dequantisation: (even key, odd key) halves of one dimension share a dword.
        let (s0, s1) = (VFSC8, VFSC8 + 1);
        scale_f32(b, s0)?;
        scale_f32(b, s1)?;
        // Two code bytes of each key per pair conversion: (p0, p1) = bytes 2cp, 2cp+1 of the even key,
        // (q0, q1) the odd key's; dims 4w + 2cp + k.
        let mut word_no = 0u8;
        for w in 0..8u8 {
            let (u0, u1) = (VFC8 + w, VFC8 + 8 + w);
            for cp in 0..2u8 {
                let (p, q) = (VFTMP8, VFTMP8 + 2);
                fp8_pair(b, p, u0, cp == 1)?;
                fp8_pair(b, q, u1, cp == 1)?;
                for k in 0..2u8 {
                    let c = 2 * cp + k;
                    let out = VFOUT8 + (word_no % 4);
                    word_no = word_no.wrapping_add(1);
                    dequant_pair(b, out, (s0, p + k), (s1, q + k), VFTMP8 + 4, VFTMP8 + 5)?;
                    let imm = 16_384 * u32::from(tp)
                        + 512 * u32::from(w >> 2)
                        + 64 * u32::from(w & 3)
                        + 16 * u32::from(c);
                    dstore(b, L_STAGE, "b32", V_LV, v(out), imm)?;
                }
            }
        }
    }
    Ok(())
}

/// Ordered QK: S^T = K * Q^T. Each Q fragment group is loaded once and shared by
/// the live subtiles; every subtile accumulates dc = 0..15 in order.
/// `fp8`: O lives at 96 (`O8`), V buffers/temps excluded; Q8 uses 128 (`O`) with
/// its own fragment/softmax registers.
struct CodecRegs {
    qf: u8,
    kf: u8,
    scores: u8,
    o: u8,
    prow: u8,
    vfa: u8,
    vfb: u8,
    smx: u8,
    swz: u8,
    smn: u8,
    sal: u8,
    ssum: u8,
    st1: u8,
    skey: u8,
    skj: u8,
    k_slot: usize,
    v_slot: usize,
}
const Q8_REGS: CodecRegs = CodecRegs {
    qf: QF,
    kf: KF,
    scores: S0,
    o: O,
    prow: PROW,
    vfa: VFA,
    vfb: VFB,
    smx: V_MX,
    swz: V_SWZ,
    smn: V_MN,
    sal: V_AL,
    ssum: V_SUM,
    st1: V_T1,
    skey: V_KEY,
    skj: V_KJ,
    k_slot: L_K,
    v_slot: L_V,
};
const FP8_REGS: CodecRegs = CodecRegs {
    qf: QF,
    kf: KF,
    scores: S0,
    o: O8,
    prow: SPRO8,
    vfa: SVFA8,
    vfb: SVFB8,
    smx: SMX8,
    swz: SMX8 + 1,
    smn: SMX8 + 2,
    sal: SMX8 + 3,
    ssum: SMX8 + 4,
    st1: SMX8 + 5,
    skey: SMX8 + 6,
    skj: SMX8 + 7,
    k_slot: L_STAGE,
    v_slot: L_STAGE,
};
fn regs(fp8: bool) -> CodecRegs {
    if fp8 {
        FP8_REGS
    } else {
        Q8_REGS
    }
}
fn qk(b: &mut Builder, fp8: bool) -> R {
    let r = regs(fp8);
    for dg in 0..4u8 {
        exec_and(b, QOK)?;
        for u in 0..4u8 {
            let dc = u32::from(4 * dg + u);
            gload(b, "b128", vr(r.qf + 4 * u, 4), V_QO, SQ16, dc * 32)?;
        }
        exec_restore(b)?;
        let end = format!(".Lfapd_dg{dg}_end");
        for sub in 0..4u8 {
            if sub > 0 {
                sop(
                    b,
                    format!("s_cmp_lt_i32 s{REM}, {}", 16 * u32::from(sub)),
                    &[],
                    &[REM],
                )?;
                op(b, format!("s_cbranch_scc1 {end}"), &[], &[])?;
            }
            for u in 0..4u8 {
                let dc = u32::from(4 * dg + u);
                dload128(
                    b,
                    r.k_slot,
                    r.kf + 4 * u,
                    V_LF,
                    (dc * 4 + u32::from(sub)) * 512,
                )?;
            }
            for u in 0..4u8 {
                wmma(
                    b,
                    r.scores + 8 * sub,
                    r.kf + 4 * u,
                    r.qf + 4 * u,
                    dg > 0 || u > 0,
                )?;
            }
        }
        b.label(&end)?;
    }
    Ok(())
}

/// One live subtile: mask + scale, online softmax update, O rescale, PV.
/// `fp8`: O at 96, softmax temps 64..71, probabilities 80..83, V buffers
/// 84..91, V reads from the reused stage (base 0); Q8 keeps its own map and V plane.
fn softmax_sub(b: &mut Builder, sub: u8, fp8: bool) -> R {
    let r = regs(fp8);
    let sb = r.scores + 8 * sub;
    let (smx, swz, smn, sal, ssum, st1, skey, skj) =
        (r.smx, r.swz, r.smn, r.sal, r.ssum, r.st1, r.skey, r.skj);
    let (prow, vfa, vfb, ob) = (r.prow, r.vfa, r.vfb, r.o);
    if sub > 0 {
        sop(
            b,
            format!("s_cmp_lt_i32 s{REM}, {}", 16 * u32::from(sub)),
            &[],
            &[REM],
        )?;
        op(b, format!("s_cbranch_scc1 {D_TURN}"), &[], &[])?;
    }
    // Causal mask of this lane's eight keys: ktile + 16 sub + 8 kg + j.
    //  * full subtile (workgroup-uniform: kt_sub + 15 <= gmin): ok_j = qok for every j;
    //  * otherwise per-row: ok_j = qok && key <= position (position -1 for invalid queries).
    let (full, masked) = (format!(".Lfapd_full{sub}"), format!(".Lfapd_mask{sub}"));
    sop(
        b,
        format!("s_cmp_ge_i32 s{REMMIN}, {}", 16 * u32::from(sub) + 15),
        &[],
        &[REMMIN],
    )?;
    op(b, format!("s_cbranch_scc1 {full}"), &[], &[])?;
    sop(
        b,
        format!("s_add_co_i32 s{}, s{KT}, {}", T[0], 16 * u32::from(sub)),
        &[T[0]],
        &[KT],
    )?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{skey}, s{}, v{V_KG8}", T[0]),
        &[v(skey)],
        &[s(T[0]), v(V_KG8)],
    )?;
    for j in 0..8u8 {
        let key = if j == 0 {
            skey
        } else {
            op(
                b,
                format!("v_add_nc_u32_e32 v{skj}, {j}, v{skey}"),
                &[v(skj)],
                &[v(skey)],
            )?;
            skj
        };
        op(
            b,
            format!("v_cmp_le_i32_e64 s{}, v{key}, v{V_POS}", mj(j)),
            &[m(mj(j))],
            &[v(key), v(V_POS)],
        )?;
    }
    wait_alu(b, "depctr_va_sdst(0)")?;
    op(b, format!("s_branch {masked}"), &[], &[])?;
    b.label(&full)?;
    for j in 0..8u8 {
        op(
            b,
            format!("s_mov_b32 s{}, s{QOK}", mj(j)),
            &[s(mj(j))],
            &[s(QOK)],
        )?;
    }
    wait_alu(b, "depctr_sa_sdst(0)")?;
    b.label(&masked)?;
    for j in 0..8u8 {
        op(
            b,
            format!("v_mul_f32_e32 v{0}, s{SCALE}, v{0}", sb + j),
            &[v(sb + j)],
            &[s(SCALE), v(sb + j)],
        )?;
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{0}, 0xff800000, v{0}, s{1}",
                sb + j,
                mj(j)
            ),
            &[v(sb + j)],
            &[v(sb + j), m(mj(j))],
        )?;
    }
    // Row max: serial chain, then the other key half through lane XOR 16.
    op(
        b,
        format!("v_max3_num_f32 v{smx}, v{sb}, 0xff800000, v{}", sb + 1),
        &[v(smx)],
        &[v(sb), v(sb + 1)],
    )?;
    for j in [2u8, 4, 6] {
        op(
            b,
            format!(
                "v_max3_num_f32 v{smx}, v{smx}, v{}, v{}",
                sb + j,
                sb + j + 1
            ),
            &[v(smx)],
            &[v(smx), v(sb + j), v(sb + j + 1)],
        )?;
    }
    op(
        b,
        format!("v_permlanex16_b32 v{swz}, v{smx}, s{PSEL}, 0xfedcba98"),
        &[v(swz)],
        &[v(smx), s(PSEL)],
    )?;
    op(
        b,
        format!("v_max3_num_f32 v{smn}, v{V_M}, v{smx}, v{swz}"),
        &[v(smn)],
        &[v(V_M), v(smx), v(swz)],
    )?;
    // alpha = (m_old == -inf) ? 0 : expf(m_old - m_new)
    op(
        b,
        format!("v_cmp_neq_f32_e64 s{CMPA}, 0xff800000, v{V_M}"),
        &[m(CMPA)],
        &[v(V_M)],
    )?;
    op(
        b,
        format!("v_sub_f32_e32 v{sal}, v{V_M}, v{smn}"),
        &[v(sal)],
        &[v(V_M), v(smn)],
    )?;
    op(
        b,
        format!("v_mul_f32_e32 v{sal}, 0x3fb8aa3b, v{sal}"),
        &[v(sal)],
        &[v(sal)],
    )?;
    op(
        b,
        format!("v_exp_f32_e32 v{sal}, v{sal}"),
        &[v(sal)],
        &[v(sal)],
    )?;
    op(
        b,
        format!("v_cmp_eq_f32_e64 s{CMPB}, 0xff800000, v{smn}"),
        &[m(CMPB)],
        &[v(smn)],
    )?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{sal}, 0, v{sal}, s{CMPA}"),
        &[v(sal)],
        &[v(sal), m(CMPA)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{V_M}, v{smn}"),
        &[v(V_M)],
        &[v(smn)],
    )?;
    // Exponent origin only: V_M already holds the real running maximum.
    // Existing s54 means m_new == -inf (compared above, waited above), so
    // sanitize the smn scratch to +0 there; masked (-inf) scores stay +0
    // while a live NaN score survives subtraction and EXP. smn's only
    // remaining uses are the probability subtractions; the next subtile
    // reduction overwrites it. Never stored back to V_M.
    op(
        b,
        format!("v_cndmask_b32_e64 v{smn}, v{smn}, 0, s{CMPB}"),
        &[v(smn)],
        &[v(smn), m(CMPB)],
    )?;
    for j in 0..8u8 {
        let x = sb + j;
        op(
            b,
            format!("v_sub_f32_e32 v{x}, v{x}, v{smn}"),
            &[v(x)],
            &[v(x), v(smn)],
        )?;
        op(
            b,
            format!("v_mul_f32_e32 v{x}, 0x3fb8aa3b, v{x}"),
            &[v(x)],
            &[v(x)],
        )?;
        op(b, format!("v_exp_f32_e32 v{x}, v{x}"), &[v(x)], &[v(x)])?;
    }
    for r in 0..4u8 {
        vmov0(b, prow + r)?;
    }
    for j in 0..8u8 {
        cvt_half(b, prow + j / 2, j % 2 == 1, sb + j)?;
    }
    // sum = e0 + e1 + ... + e7 (ascending), + XOR-16 peer, l = fma(l, alpha, sum).
    op(
        b,
        format!("v_add_f32_e32 v{ssum}, v{sb}, v{}", sb + 1),
        &[v(ssum)],
        &[v(sb), v(sb + 1)],
    )?;
    for j in 2..8u8 {
        op(
            b,
            format!("v_add_f32_e32 v{ssum}, v{}, v{ssum}", sb + j),
            &[v(ssum)],
            &[v(sb + j), v(ssum)],
        )?;
    }
    op(
        b,
        format!("v_permlanex16_b32 v{st1}, v{ssum}, s{PSEL}, 0xfedcba98"),
        &[v(st1)],
        &[v(ssum), s(PSEL)],
    )?;
    op(
        b,
        format!("v_add_f32_e32 v{ssum}, v{ssum}, v{st1}"),
        &[v(ssum)],
        &[v(ssum), v(st1)],
    )?;
    op(
        b,
        format!("v_fmac_f32_e32 v{ssum}, v{V_L}, v{sal}"),
        &[v(ssum)],
        &[v(ssum), v(V_L), v(sal)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{V_L}, v{ssum}"),
        &[v(V_L)],
        &[v(ssum)],
    )?;
    // PV: O^T = V^T * P^T, O rescaled by this lane's own alpha before each dc update.
    let buf = |dc: u8| if dc % 2 == 0 { vfa } else { vfb };
    let vbase: u32 = if fp8 { 0 } else { 32_768 };
    let vslot = r.v_slot;
    let vo = |dc: u8| vbase + (16 * u32::from(sub) + u32::from(dc)) * 512;
    dload128(b, vslot, vfa, V_LF, vo(0))?;
    for dc in 0..16u8 {
        if dc + 1 < 16 {
            dload128(b, vslot, buf(dc + 1), V_LF, vo(dc + 1))?;
        }
        for i in 0..8u8 {
            let r = ob + 8 * dc + i;
            op(
                b,
                format!("v_mul_f32_e32 v{r}, v{sal}, v{r}"),
                &[v(r)],
                &[v(sal), v(r)],
            )?;
        }
        wmma(b, ob + 8 * dc, buf(dc), prow, true)?;
    }
    Ok(())
}

fn body(b: &mut Builder, g: &blocks::Gen) -> R {
    use blocks as bl;
    bl::tile_setup(b, g)?;
    if !g.fp8() {
        bl::fill(b, g)?;
        b.label(bl::FILL_DONE)?;
        b.barrier(&[Transition::Ready(bl::SLOT_K), Transition::Ready(bl::SLOT_V)])?;
        b.label(bl::QK)?;
        bl::skip_if_helper(b, bl::TURN)?;
        bl::qk(b, g)?;
        for sub in 0..4u8 {
            bl::softmax_sub(b, g, sub)?;
        }
        b.label(bl::TURN)?;
        b.barrier(&[
            Transition::Retire(bl::SLOT_K),
            Transition::Retire(bl::SLOT_V),
        ])?;
        // Continue while ktile <= gmax (`s_cmp_gt_i32` + scc0: the compare the emulator models).
        bl::next_tile(b)
    } else {
        // Sequential single-stage schedule: K-fill, QK, V-fill, softmax/PV.
        // All five waves join every barrier; causal exits target TURN,
        // which retires the stage after the final tile's V use.
        bl::fill_k(b, g)?;
        b.label(bl::KFILL_DONE)?;
        b.barrier(&[Transition::Ready(bl::SLOT_STAGE)])?;
        b.label(bl::KREADY)?;
        bl::skip_if_helper(b, bl::QK_DONE)?;
        bl::qk(b, g)?;
        b.label(bl::QK_DONE)?;
        b.barrier(&[Transition::Retire(bl::SLOT_STAGE)])?;
        bl::fill_v(b, g)?;
        b.label(bl::VFILL_DONE)?;
        b.barrier(&[Transition::Ready(bl::SLOT_STAGE)])?;
        b.label(bl::VREADY)?;
        bl::skip_if_helper(b, bl::TURN)?;
        for sub in 0..4u8 {
            bl::softmax_sub(b, g, sub)?;
        }
        b.label(bl::TURN)?;
        b.barrier(&[Transition::Retire(bl::SLOT_STAGE)])?;
        bl::next_tile(b)
    }
}
/// Compute waves: out = O * (l > 0 ? 1/l : 0) for live queries only.
/// `fp8`: O at 96; Q8 at 128.
fn epilogue(b: &mut Builder, fp8: bool) -> R {
    let ob = if fp8 { O8 } else { O };
    b.label(D_EPI)?;
    sop(b, format!("s_cmp_ge_u32 s{WAVE}, 4"), &[], &[WAVE])?;
    op(b, format!("s_cbranch_scc1 {D_END}"), &[], &[])?;
    let (sc, r, q, e, div, oo) = (S0, S0 + 1, S0 + 2, S0 + 3, S0 + 4, S0 + 5);
    op(
        b,
        format!("v_cmp_lt_f32_e64 s{CMPA}, 0, v{V_L}"),
        &[m(CMPA)],
        &[v(V_L)],
    )?;
    fdiv(b, div, "1.0", &format!("v{V_L}"), &[v(V_L)], [sc, r, q, e])?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{div}, 0, v{div}, s{CMPA}"),
        &[v(div)],
        &[v(div), m(CMPA)],
    )?;
    for i in 0..128u8 {
        op(
            b,
            format!("v_mul_f32_e32 v{0}, v{div}, v{0}", ob + i),
            &[v(ob + i)],
            &[v(div), v(ob + i)],
        )?;
    }
    op(
        b,
        format!("v_lshlrev_b32_e32 v{oo}, 1, v{V_QO}"),
        &[v(oo)],
        &[v(V_QO)],
    )?;
    exec_and(b, QOK)?;
    for dc in 0..16u8 {
        gstore128(b, oo, ob + 8 * dc, SO, u32::from(dc) * 64)?;
        gstore128(b, oo, ob + 8 * dc + 4, SO, u32::from(dc) * 64 + 16)?;
    }
    exec_restore(b)?;
    b.wait_all()?;
    b.label(D_END)?;
    b.control(Sop::End.encode(Arch::Gfx1201)?)
}

/// The direct attention entry: `attention_q8_0_fa2_gqa_pm_gfx1201` for [`FaKvCodec::Q8_0`],
/// `attention_fp8_e4m3_fa2_gqa_f16_pm_gfx1201` for [`FaKvCodec::Fp8E4M3`]. The codec selects only the prologue's
/// global offsets, the tile base stride and the fill; QK, softmax, PV and the epilogue are shared.
pub fn emit(arch: Arch, spec: &FaPrefillSpec) -> Result<Emitted, String> {
    let g = blocks::Gen::new(arch, *spec)?;
    let mut b = Builder::new(g.kernel_spec(), g.plan()?);
    b.enable_delay_alu();
    for (id, (name, base, bytes)) in g.lds_slots().into_iter().enumerate() {
        if b.lds_slot(name, base, bytes)? != id {
            return Err("LDS slot order".into());
        }
    }
    blocks::prologue(&mut b, &g)?;
    b.loop_(blocks::LOOP, |b| body(b, &g))?;
    blocks::epilogue(&mut b, &g)?;
    b.finish()
}

// ------------------------------------------------------------------ preconvert kernel
const PV_TID: u8 = 0;
const PV_LANE: u8 = 1;
const PV_WAVE: u8 = 2;
const PV_OFFQ: u8 = 3; // lane * 32
const PV_OFFO: u8 = 4; // lane * 16
const PV_IN: u8 = 8; // eight f32
const PV_OUT: u8 = 16; // four packed f16x2
                       // SGPRs
const PS_PTRS: u8 = 8; // q (8), q16 (10)
const PS_BATCH: u8 = 12;
const PS_WAVE: u8 = 13;
const PS_ROW: u8 = 14;
const PS_LIMIT: u8 = 15;
const PS_QB: u8 = 16;
const PS_OB: u8 = 18;
const PS_T0: u8 = 20;
const PS_T1: u8 = 21;

fn plan_preconvert() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(24, 104)?;
    let w = || Live::Whole;
    for (name, base) in [
        ("tid", PV_TID),
        ("lane", PV_LANE),
        ("wave", PV_WAVE),
        ("off_q", PV_OFFQ),
        ("off_o", PV_OFFO),
    ] {
        p.v::<1>(name, base, w())?;
    }
    p.v::<8>("q_in", PV_IN, w())?;
    p.v::<4>("q16_out", PV_OUT, w())?;
    // The descriptor reserves VGPRs in granules of eight: declare the whole 24.
    p.v::<4>("granule_pad", PV_OUT + 4, w())?;
    p.s::<2>("kernarg_ptr", 0, w())?;
    p.s::<4>("ptrs", PS_PTRS, w())?;
    for n in [PS_BATCH, PS_WAVE, PS_ROW, PS_LIMIT, PS_T0, PS_T1] {
        p.s::<1>("scalar", n, w())?;
    }
    p.s::<2>("q_base", PS_QB, w())?;
    p.s::<2>("q16_base", PS_OB, w())?;
    Ok(p)
}

fn preconvert_body(b: &mut Builder) -> R {
    b.push(
        Instruction::new(
            format!("s_load_b128 s[{PS_PTRS}:{}], s[0:1], 0x0", PS_PTRS + 3),
            vec![sr(PS_PTRS, 4)],
            vec![sr(0, 2)],
        )
        .memory(MemoryClass::SmemLoad),
    )?;
    b.push(
        Instruction::new(
            format!("s_load_b32 s{PS_BATCH}, s[0:1], 0x20"),
            vec![s(PS_BATCH)],
            vec![sr(0, 2)],
        )
        .memory(MemoryClass::SmemLoad),
    )?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{PV_WAVE}, 5, v{PV_TID}"),
        &[v(PV_WAVE)],
        &[v(PV_TID)],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{PS_WAVE}, v{PV_WAVE}"),
        &[s(PS_WAVE)],
        &[v(PV_WAVE)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{PV_LANE}, 31, v{PV_TID}"),
        &[v(PV_LANE)],
        &[v(PV_TID)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{PV_OFFQ}, 5, v{PV_LANE}"),
        &[v(PV_OFFQ)],
        &[v(PV_LANE)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{PV_OFFO}, 4, v{PV_LANE}"),
        &[v(PV_OFFO)],
        &[v(PV_LANE)],
    )?;
    b.wait(Counter::Km, 0)?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    // r = 4 * block + wave; a wave past batch * 16 rows exits (uniform, no barrier here).
    op(
        b,
        format!("s_lshl_b32 s{PS_ROW}, ttmp9, 2"),
        &[s(PS_ROW)],
        &[],
    )?;
    sop(
        b,
        format!("s_add_co_i32 s{PS_ROW}, s{PS_ROW}, s{PS_WAVE}"),
        &[PS_ROW],
        &[PS_ROW, PS_WAVE],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{PS_LIMIT}, s{PS_BATCH}, 4"),
        &[PS_LIMIT],
        &[PS_BATCH],
    )?;
    sop(
        b,
        format!("s_cmp_ge_i32 s{PS_ROW}, s{PS_LIMIT}"),
        &[],
        &[PS_ROW, PS_LIMIT],
    )?;
    op(b, format!("s_cbranch_scc1 {P_END}"), &[], &[])?;
    // Row r owns f32 elements [r * 256, r * 256 + 256): qr * 4096 + h * 256 = r * 256.
    sop(
        b,
        format!("s_lshl_b32 s{PS_T0}, s{PS_ROW}, 10"),
        &[PS_T0],
        &[PS_ROW],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{PS_T1}, s{PS_ROW}, 9"),
        &[PS_T1],
        &[PS_ROW],
    )?;
    add64(b, PS_QB, PS_PTRS, PS_T0, None)?;
    add64(b, PS_OB, PS_PTRS + 2, PS_T1, None)?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    gload(b, "b128", vr(PV_IN, 4), PV_OFFQ, PS_QB, 0)?;
    gload(b, "b128", vr(PV_IN + 4, 4), PV_OFFQ, PS_QB, 16)?;
    for r in 0..4u8 {
        vmov0(b, PV_OUT + r)?;
    }
    for i in 0..8u8 {
        cvt_half(b, PV_OUT + i / 2, i % 2 == 1, PV_IN + i)?;
    }
    gstore128(b, PV_OFFO, PV_OUT, PS_OB, 0)?;
    b.wait_all()?;
    b.label(P_END)?;
    b.control(Sop::End.encode(Arch::Gfx1201)?)
}

/// The Q preconvert entry `attention_fa2_q_preconvert_pm_gfx1201`. `signs1`,
/// `signs2` and `do_rotate` keep the oracle ABI and are never read.
pub fn emit_preconvert(arch: Arch, spec: &FaPrefillSpec) -> Result<Emitted, String> {
    let g = blocks::Gen::new(arch, *spec)?;
    let mut b = Builder::new(g.preconvert_kernel_spec(), g.preconvert_plan()?);
    b.enable_delay_alu();
    blocks::preconvert(&mut b, &g)?;
    b.finish()
}

/// Both exports of the Q8_0 module as one code object, preconvert first, then direct. Q8_0 only: the native-fp8
/// kernel is [`emit_fp8_module`].
pub fn emit_module(
    arch: Arch,
    spec: &FaPrefillSpec,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    admit(arch, spec)?;
    if spec.kv != FaKvCodec::Q8_0 {
        return Err(format!(
            "fa_prefill: emit_module is the Q8_0 module, got {:?} (use emit_fp8_module)",
            spec.kv
        ));
    }
    let emitted = vec![emit_preconvert(arch, spec)?, emit(arch, spec)?];
    let (text, proof) = super::iu4_gemm::module(&emitted, MODULE)?;
    Ok((emitted, text, proof))
}

/// The native-fp8 module `fa_prefill_fp8_pm_gfx1201`: ONE direct export
/// `attention_fp8_e4m3_fa2_gqa_f16_pm_gfx1201` (no preconvert: the Q8 module's
/// `attention_fa2_q_preconvert_pm_gfx1201` and the F5 qprep produce the same f16 Q16). Same return shape as
/// [`emit_module`]. [`FaKvCodec::Fp8E4M3`] only.
pub fn emit_fp8_module(
    arch: Arch,
    spec: &FaPrefillSpec,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    admit(arch, spec)?;
    if spec.kv != FaKvCodec::Fp8E4M3 {
        return Err(format!(
            "fa_prefill: emit_fp8_module is the Fp8E4M3 module, got {:?} (use emit_module)",
            spec.kv
        ));
    }
    let emitted = vec![emit(arch, spec)?];
    let (text, proof) = super::iu4_gemm::module(&emitted, MODULE_FP8)?;
    Ok((emitted, text, proof))
}

// ------------------------------------------------------------------ contracts
const DIRECT_COUNTED: [&str; 28] = [
    "v_wmma_f32_16x16x16_f16",
    "s_barrier_signal",
    "s_barrier_wait",
    "ds_load_b128",
    "ds_store_b64",
    "ds_store_b32",
    "global_load_u16",
    "global_load_b32",
    "global_load_b128",
    "global_store_b32",
    "s_load_b256",
    "s_load_b128",
    "s_load_b64",
    "s_load_b32",
    "v_exp_f32_e32",
    "v_permlanex16_b32",
    "v_mov_b32_dpp",
    "v_max3_num_f32",
    "v_fma_mixlo_f16",
    "v_perm_b32",
    "v_cvt_f16_f32_e32",
    "v_fmac_f32_e32",
    "v_div_scale_f32",
    "v_div_fmas_f32",
    "v_div_fixup_f32",
    "v_rcp_f32_e32",
    "v_mul_f32_e32",
    "s_endpgm",
];
const PRECONVERT_COUNTED: [&str; 7] = [
    "global_load_b128",
    "global_store_b32",
    "v_cvt_f16_f32_e32",
    "s_load_b128",
    "s_load_b32",
    "v_readfirstlane_b32",
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

/// Shape contract of the direct export (`toolchain::IsaShapeContract`): counts are derived from the emission.
pub fn contract(arch: Arch, emitted: &Emitted) -> serde_json::Value {
    let _ = arch;
    serde_json::json!({
        "symbol": SYMBOL_DIRECT, "counts": counts(emitted, &DIRECT_COUNTED),
        "forbidden": ["scratch_*", "flat_*", "buffer_*", "global_atomic*", "ds_load_2addr*", "ds_store_2addr*",
                      "ds_bpermute*", "ds_swizzle*", "v_writelane*", "s_setpc*", "s_swappc*", "s_call*", "s_waitcnt*"],
        "vgpr_max": 256, "sgpr_max": 104,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": DYNAMIC_LDS_BYTES,
        "group_segment_fixed_bytes": 0,
    })
}

/// Shape contract of the native-fp8 direct export: the Q8 contract plus the exact counts of the two native
/// `v_cvt_pk_f32_fp8` forms (`e32` low half, `e64` `op_sel:[1,0]` high half), and every other FP8/BF8/integer
/// conversion route forbidden (the Q8 sign-extension `v_cvt_f32_i32`, the single-byte `v_cvt_f32_fp8`, the BF8 decode
/// and every FP8/BF8 encode). Counts are derived from the emission.
pub fn fp8_contract(arch: Arch, emitted: &Emitted) -> serde_json::Value {
    let _ = arch;
    let mut counted: Vec<&str> = DIRECT_COUNTED.to_vec();
    counted.extend([
        "v_cvt_pk_f32_fp8_e32",
        "v_cvt_pk_f32_fp8_e64",
        "v_cvt_f32_f16_e64",
    ]);
    serde_json::json!({
        "symbol": SYMBOL_FP8_DIRECT, "counts": counts(emitted, &counted),
        "forbidden": ["scratch_*", "flat_*", "buffer_*", "global_atomic*", "ds_load_2addr*", "ds_store_2addr*",
                      "ds_bpermute*", "ds_swizzle*", "v_writelane*", "s_setpc*", "s_swappc*", "s_call*", "s_waitcnt*",
                      "v_cvt_f32_i32*", "v_cvt_f32_fp8*", "v_cvt_f32_bf8*", "v_cvt_pk_f32_bf8*", "v_cvt_pk_fp8_f32*",
                      "v_cvt_pk_bf8_f32*", "v_cvt_sr_*"],
        "vgpr_max": 224, "sgpr_max": 104,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": FP8_DYNAMIC_LDS_BYTES,
        "group_segment_fixed_bytes": 0,
    })
}

/// Shape contract of the preconvert export: no DS, barrier or WMMA.
pub fn preconvert_contract(arch: Arch, emitted: &Emitted) -> serde_json::Value {
    let _ = arch;
    serde_json::json!({
        "symbol": SYMBOL_PRECONVERT, "counts": counts(emitted, &PRECONVERT_COUNTED),
        "forbidden": ["scratch_*", "flat_*", "buffer_*", "global_atomic*", "ds_*", "s_barrier*", "v_wmma*", "v_swmmac*",
                      "v_writelane*", "s_setpc*", "s_swappc*", "s_call*", "s_waitcnt*"],
        "vgpr_max": 24, "sgpr_max": 104,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0,
        "group_segment_fixed_bytes": 0,
    })
}
