// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx11 (gfx1100, gfx1151) NT1 grouped symmetric IU4 MoE GEMMs (Consumer24 B4; Halo E8P port): gate/up+SwiGLU
//! and down, over MQ4 (QT44/QT53 pointer-addressed), planar W2 (`d*(2c-3)`),
//! planar W3 (`d*(2c-7)`) and planar E8P-I4 (`d*(2C+s)/4`, LUT/MASK stage in LDS; see [`e8p`])
//! expert tensors, each in a BF16 entry and a
//! qualification-only `_acc` entry (F32 accumulators, never dispatched).
//!
//! This module imports only public `peacemaker_isa` / `peacemaker_author`
//! items, so a standalone example can include it by `#[path]`.
//!
//! ABI, grid, block (frozen, SOL-B4-B5 §2.1 plus the rc5 `n_experts` guard word): 64-byte kernargs
//! `expert_weight_ptrs@0, expert_tile_ids@8, sorted_slot_index@16, Xq@24,
//! Y_grouped@32, M@40, K@44, x_row_div@48, m_total@52, x_src_rows@56, n_experts@60`; wave32;
//! a tile whose expert id is `>= n_experts` as an unsigned compare (the -1 sentinel, any negative or
//! corrupt id) exits before the expert pointer load, table prefetch and first barrier, with no store;
//! GateUp 64 threads, Down 128; grid `(M/64, m_total/16)`. Lane `l`:
//! `lr = l & 15`, `hi = l >> 4`; `rbase = 32*x + 16*wave` (GateUp) or
//! `64*x + 16*wave` (Down); accumulator `j` of lane `(hi, lr)` is output row
//! `rbase + 2j + hi` of grouped token `16*tile + lr`.
//!
//! # Address proof (per lane, per epoch `e`, `E = K/128`, all offsets in bytes)
//!
//! * weights: `expert_ptr(e_id) + row*ROWB + code_offset(e)` plus the
//!   immediates of [`code_loads`]. `row = rbase + lr` (GateUp up rows: `+ M/2`).
//!   `ROWB` = [`row_code_bytes`]. [`code_offset`]: W3 `48e`, W2 `32e`, MQ4 Down
//!   `68e + 4`, MQ4 GateUp `(e>>1)*136 + 8 + (e&1)*64`. No `hi` term: both
//!   half waves carry all 16 operand dwords of the epoch.
//! * scales (16-bit loads, never a dword): row `(lr & 14) | hi`,
//!   [`scale_offset`]: planar `2e` into the scale plane at `M*ROWB`; MQ4 Down
//!   `68e`; MQ4 GateUp `(e>>1)*136 + (e&1)*4` (the header word holds the f16
//!   scale in its low half and the zero point above; only the low 16 bits are
//!   read). The last epoch of K640 (odd Down epoch count) therefore reads
//!   exactly two bytes at `2*(E-1)` of the scale row: no pair load crosses the
//!   final f16.
//! * activations: `Xq + e*x_src_rows*72 + slot_row*72`; `d` f32 at `+0`, the 64
//!   nibble bytes at `+8..+72`. A padding slot is forced to `GATHER_OOB` after
//!   every epoch add, so it can never wrap into memory.
//!
//! Every descriptor's `num_records` is the exact extent from its base to the
//! end of the tensor (see `prologue.rs`), the expert pointer is a 64-bit
//! `s_add_u32/s_addc_u32` sum (tensors above 4 GiB), and the K loop issues no
//! load past epoch `E-1`.
//!
//! # Register proof (frozen, SOL-B4-B5 §2.1)
//!
//! VGPR: `v0` tid; `v1..15` prologue scratch (`v2` lane id Whole in `_acc`);
//! `v16` X row offset; `v17` per-epoch X offset; `v18` WOFF; `v19` HOFF;
//! `v20` YOFF (Whole); `v21`/`v22` per-epoch code/scale offsets; `v24..31`
//! MAGIC; `v32..47` CACC (8/tensor); `v48..63` SUM (Whole; `v48+j` = row
//! `rbase+2j+hi`, GateUp up sums at `v56+j`); `v64..79` SCF; `v80..87` fold
//! products; `v88,v89` f32 scales; `v92..` operand set (A per tensor, X, scale
//! dwords, `d`, four unpack scratch). SGPR: `s0:1` kernarg; the native workgroup ids at `s2,s3` (gfx1151) or
//! `s14,s15` (gfx1100: LLVM's `FeatureUserSGPRInit16Bug` makes the descriptor declare 14 user SGPRs so that
//! user + system >= 16, and the ids follow the declared count; see [`Gen::wg_in`]); the eight-dword kernarg
//! `s_load_b256` at `s8:15` (gfx1151) or `s88:95` (gfx1100, clear of the ids; see [`Gen::args0`]);
//! `s16:17` Y (Whole); `s18` M; `s47` TILE16; `s49` RBASE;
//! `s52:53` LIVE (Whole); `s60:63` rounding masks (Whole). The epilogue's
//! aliases (RT_TMP v32.., OWN v64.., SEND/RECV/SEL v68.., OUT v76.., PACK
//! v80.., SiLU v92.. and s24..47, HIMASK s54:55, PSEL s56, SRD_Y s64:67) are
//! reserved by [`epilogue::reserve_bf16`] / [`epilogue::reserve_acc`] with
//! `Live::Between(epilogue, end)`; every K-loop range ends at `epilogue`.
//!
//! # Numerics
//!
//! Exactly [`fold`]: exact int32 WMMA chain per epoch, `S = fma(RN(d*sc),
//! float(D), S)` in ascending epoch order from `+0`, via the shared
//! `iu4_fold` helpers. Dot bounds: [`dot_bound`] (W2 3072, W3 7168, MQ4 8192, E8P 6144).
//!
//! E8P ([`e8p`], `Codec::E8p`) is not an unpack-only codec: a word's signed nibble is
//! `o = (v - 1)/2` of the codeword `v` (LUT/MASK unpack from a 1,536 B per-CTA LDS stage), so the
//! epoch is `sum(v q) = 2C + s` with the activation sum `s` (`i32` at `Xq + 4` of each A4 block),
//! and the fold is `S = fma(RN(d*sc), (2C + s)/4, S)` (`2a + (s - MAGIC)`, `+ -MAGIC`, `* 0.25`). The
//! D-only `iu4_fold::fold_pass` is not used for it. Wire/ABI: planar W2 geometry (`K/4` code bytes + `K/64`
//! scale bytes per row), the same 64-byte kernargs and grid; the entries are
//! `qwen4_moe_{gate_up_silu,down}_iu4_sym_e8p_pm_<arch>[_acc]` in `qwen4_moe_iu4_sym_e8p_pm_<arch>` (`<arch>` = `gfx1100` or
//! `gfx1151`: [`Spec::arch`] selects the typed target and the names; the instruction path is one).
//!
//! # Negative controls
//!
//! [`Negative`] builds deliberately wrong kernels for the G1 negative gates:
//! `WrongScaleRow` loads the gfx12 scale row; `ReversedEpoch` folds epochs in
//! descending order with the matching operands; `WrongPackSelector` is the
//! epilogue's one wrong `v_perm` selector (BF16 entry only, not E8P); `WrongExpertPointer`
//! reads expert slot 0 instead of the looked-up id. E8P adds `WrongLut` (stage `LUT[i] := LUT[i^1]`),
//! `WrongSignMask` (stage `MASK[s] := MASK[s^1]`) and `WrongFoldEpoch` (the per-epoch `s - MAGIC` add becomes
//! a plain `-MAGIC`: the activation sum `s` is dropped but the signed dot and the `0.25` multiplier stay,
//! so each epoch contributes `2C/4` instead of `(2C + s)/4`). The symbol is unchanged so
//! the object is drop-in; keep these out of every candidate module.
use peacemaker_author::{
    prime, retire_cur, rotate, End, Free, Full, Gfx1100, Gfx1151, Gfx11Waits, MmaIu4, Published,
    Ring, Scc, Target, WgUniform, Workgroup,
};
use peacemaker_isa::{
    insn::{Instruction, MemoryClass},
    kernels::{
        common::{
            add64, bload, gather_offset, lit, mem, op, s, s_add_i32, s_add_u32, smem, sop, sr,
            srd_tail, v, GatherTemps, GATHER_OOB, SRD_WORD3,
        },
        iu4_fold::{self, MAGIC, REBIAS},
        qwen4_moe_sym::MAX_SLOTS,
    },
    reg::Live,
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan, V,
};

pub use peacemaker_isa::kernels::qwen4_moe_sym::Kind;

pub mod e8p;
pub mod epilogue;
mod fold;
mod kloop;
mod pipe;
mod prologue;
mod unpack;
pub use unpack::{bfi, mq4_unpack_model, w2_spread_model, w3_spread_model};

/// The gfx11 targets this module is authored for (wave32, one `s_barrier`, VMcnt/LGKMcnt, native IU4 WMMA).
pub trait Gfx11: Target<Waits = Gfx11Waits, Barrier = Full> + MmaIu4 {}
impl<T: Target<Waits = Gfx11Waits, Barrier = Full> + MmaIu4> Gfx11 for T {}

type Wg<'b, T> = Workgroup<'b, T, Builder>;

/// The architectures [`emit`] builds: gfx1100 (RDNA3) and gfx1151 (RDNA3.5).
pub const ARCHES: [Arch; 2] = [Arch::Gfx1100, Arch::Gfx1151];

pub const KERNARG_BYTES: u32 = 64;
pub const VGPR_CEILING: u16 = 256;

/// Weight codec of the expert tensors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Codec {
    Mq4,
    W2,
    W3,
    E8p,
}
impl Codec {
    /// The MQ4/W2/W3 modules (the E8P module is emitted separately: [`Codec::E8p`]).
    pub const ALL: [Codec; 3] = [Codec::Mq4, Codec::W2, Codec::W3];
    /// Symbol infix after `sym`: `""`, `"_w2"`, `"_w3"`, `"_e8p"`.
    pub fn infix(self) -> &'static str {
        match self {
            Self::Mq4 => "",
            Self::W2 => "_w2",
            Self::W3 => "_w3",
            Self::E8p => "_e8p",
        }
    }
    /// Code object module: `qwen4_moe_iu4_sym{,_w2,_w3,_e8p}_pm_<arch>`.
    pub fn module(self, arch: Arch) -> String {
        format!("qwen4_moe_iu4_sym{}_pm_{}", self.infix(), arch.name())
    }
    /// Static LDS bytes of every entry: the E8P `LUT[256]` + `MASK[128]` stage, none otherwise.
    pub fn lds_bytes(self) -> u32 {
        if self == Self::E8p {
            e8p::STAGE_BYTES
        } else {
            0
        }
    }
}

/// Deliberately broken variants for the G1 negative controls.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Negative {
    None,
    WrongScaleRow,
    ReversedEpoch,
    WrongPackSelector,
    WrongExpertPointer,
    WrongLut,
    WrongSignMask,
    WrongFoldEpoch,
}
impl Negative {
    /// The four MQ4/W2/W3 mutations (no `None`).
    pub const ALL: [Negative; 4] = [
        Negative::WrongScaleRow,
        Negative::ReversedEpoch,
        Negative::WrongPackSelector,
        Negative::WrongExpertPointer,
    ];
    /// The six E8P mutations: no pack-selector one, plus `WrongLut` (stage `LUT[i] := LUT[i ^ 1]`),
    /// `WrongSignMask` (stage `MASK[s] := MASK[s ^ 1]`) and `WrongFoldEpoch` (fold term `2C/4`, no `s`).
    pub const E8P: [Negative; 6] = [
        Negative::WrongLut,
        Negative::WrongSignMask,
        Negative::ReversedEpoch,
        Negative::WrongFoldEpoch,
        Negative::WrongScaleRow,
        Negative::WrongExpertPointer,
    ];
    /// Whether the mutation exists for `codec`: `WrongLut`/`WrongSignMask`/`WrongFoldEpoch` are E8P-only, `WrongPackSelector` is not E8P.
    pub fn applies(self, codec: Codec) -> bool {
        match self {
            Self::None | Self::WrongScaleRow | Self::ReversedEpoch | Self::WrongExpertPointer => {
                true
            }
            Self::WrongPackSelector => codec != Codec::E8p,
            Self::WrongLut | Self::WrongSignMask | Self::WrongFoldEpoch => codec == Codec::E8p,
        }
    }
    pub fn tag(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::WrongScaleRow => "wrong-scale-row",
            Self::ReversedEpoch => "reversed-epoch",
            Self::WrongPackSelector => "wrong-pack-selector",
            Self::WrongExpertPointer => "wrong-expert-pointer",
            Self::WrongLut => "wrong-lut",
            Self::WrongSignMask => "wrong-sign-mask",
            Self::WrongFoldEpoch => "wrong-fold-epoch",
        }
    }
}

/// One entry: target arch, codec, kind, BF16 or `_acc`, and an optional mutation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Spec {
    pub arch: Arch,
    pub codec: Codec,
    pub kind: Kind,
    pub acc: bool,
    pub negative: Negative,
}

fn tensors(kind: Kind) -> u8 {
    match kind {
        Kind::GateUp => 2,
        Kind::Down => 1,
    }
}

impl Spec {
    /// `qwen4_moe_gate_up_silu_iu4_sym{,_w2,_w3,_e8p}_pm_<arch>` / `qwen4_moe_down_iu4_sym..`, `_acc` suffixed.
    pub fn symbol(self) -> String {
        let base = match self.kind {
            Kind::GateUp => format!(
                "qwen4_moe_gate_up_silu_iu4_sym{}_pm_{}",
                self.codec.infix(),
                self.arch.name()
            ),
            Kind::Down => format!(
                "qwen4_moe_down_iu4_sym{}_pm_{}",
                self.codec.infix(),
                self.arch.name()
            ),
        };
        if self.acc {
            format!("{base}_acc")
        } else {
            base
        }
    }
    /// Contract variant tag: `gate_up{infix}[_acc][__neg_<tag>]`.
    pub fn variant(self) -> String {
        let mut t = format!(
            "{}{}{}",
            self.kind.tag(),
            self.codec.infix(),
            if self.acc { "_acc" } else { "" }
        );
        if self.negative != Negative::None {
            t.push_str(&format!("__neg_{}", self.negative.tag().replace('-', "_")));
        }
        t
    }
    /// Threads per workgroup: GateUp 64 (two waves), Down 128 (four waves).
    pub fn threads(self) -> u32 {
        match self.kind {
            Kind::GateUp => 64,
            Kind::Down => 128,
        }
    }
    /// The frozen 64-byte kernarg layout: the 60-byte ABI plus a trailing `n_experts` at 60.
    pub fn kernargs(self) -> KernargLayout {
        KernargLayout::new(KERNARG_BYTES)
            .pointer("expert_weight_ptrs", 0)
            .pointer("expert_tile_ids", 8)
            .pointer("sorted_slot_index", 16)
            .pointer("Xq", 24)
            .pointer("Y_grouped", 32)
            .hidden("M", 40, 4, "by_value")
            .hidden("K", 44, 4, "by_value")
            .hidden("x_row_div", 48, 4, "by_value")
            .hidden("m_total", 52, 4, "by_value")
            .hidden("x_src_rows", 56, 4, "by_value")
            .hidden("n_experts", 60, 4, "by_value")
    }
    /// Dispatch grid `(M/64, m_total/16)`.
    pub fn grid(self, m: u32, m_total: u32) -> (u32, u32) {
        (m / 64, m_total / 16)
    }
    /// Spec-level admission (independent of launch dimensions): a gfx11 arch, and a mutation that exists for the codec.
    pub fn validate(self) -> Result<(), String> {
        if !ARCHES.contains(&self.arch) {
            return Err(format!(
                "qwen4_moe_sym_gfx11: no gfx11 target for {}",
                self.arch.name()
            ));
        }
        if !self.negative.applies(self.codec) {
            return Err(format!(
                "{}: mutation {} does not exist for codec {:?}",
                self.arch.name(),
                self.negative.tag(),
                self.codec
            ));
        }
        Ok(())
    }
    /// Static LDS bytes of the entry (E8P: the 1,536 B table stage; else 0). The GateUp lockstep
    /// regions are typed bookkeeping only: the ABI keeps the allocation at 0 bytes, so their
    /// one-dword accesses are out of range (stores discarded, loads return 0).
    pub fn lds_bytes(self) -> u32 {
        self.codec.lds_bytes()
    }
}

/// Exact K positivity/multiples, output-row divisibility, source gather and
/// extent constraints of one launch. No generic tail is admitted: every
/// quantity the kernel multiplies in 32 bits is bounded here, so no scalar or
/// 24-bit lane product wraps and every descriptor extent fits `num_records`.
///
/// `m`: kernarg M (GateUp: gate+up rows). GateUp needs `K % 256 == 0`, Down
/// `K % 128 == 0`; both need `M % 64 == 0` and `m_total % 16 == 0`.
pub fn validate_launch(
    spec: Spec,
    m: u32,
    k: u32,
    x_row_div: u32,
    m_total: u32,
    x_src_rows: u32,
) -> Result<(), String> {
    spec.validate()?;
    let n = spec.arch.name();
    let kmul = match spec.kind {
        Kind::GateUp => 256,
        Kind::Down => 128,
    };
    if k == 0 || k % kmul != 0 {
        return Err(format!(
            "{n} {}: K {k} must be a positive multiple of {kmul}",
            spec.kind.tag()
        ));
    }
    if m == 0 || m % 64 != 0 {
        return Err(format!(
            "{n} {}: M {m} must be a positive multiple of 64",
            spec.kind.tag()
        ));
    }
    if m_total == 0 || m_total % 16 != 0 || m_total > i32::MAX as u32 {
        return Err(format!(
            "{n}: m_total {m_total} must be a positive multiple of 16 below 2^31"
        ));
    }
    if x_row_div == 0 || x_src_rows == 0 {
        return Err(format!("{n}: x_row_div and x_src_rows must be positive"));
    }
    let slots = u64::from(x_row_div) * u64::from(x_src_rows);
    if slots > u64::from(MAX_SLOTS) {
        return Err(format!(
            "{n}: x_row_div*x_src_rows {slots} exceeds the exact gather limit {MAX_SLOTS}"
        ));
    }
    let x_extent = u64::from(k / 128) * u64::from(x_src_rows) * 72;
    if x_extent >= u64::from(GATHER_OOB) {
        return Err(format!(
            "{n}: activation extent {x_extent} reaches the gather out-of-range offset"
        ));
    }
    let rowb = u64::from(row_code_bytes(spec.codec, k));
    let srow = u64::from(row_scale_bytes(spec.codec, k));
    if rowb >= 1 << 24 || srow >= 1 << 24 {
        return Err(format!(
            "{n}: row bytes {rowb}/{srow} exceed the 24-bit lane multiply"
        ));
    }
    let total = tensor_bytes(spec.codec, m, k);
    if total >= 1 << 32 {
        return Err(format!(
            "{n}: tensor extent {total} exceeds the 32-bit descriptor extent"
        ));
    }
    let output_rows = if spec.kind == Kind::GateUp && !spec.acc {
        m / 2
    } else {
        m
    };
    let output_extent = u64::from(m_total) * u64::from(output_rows) * if spec.acc { 4 } else { 2 };
    if output_extent >= 1 << 32 {
        return Err(format!(
            "{n}: output extent {output_extent} exceeds 32-bit store offsets"
        ));
    }
    Ok(())
}

/// Physical code row bytes: MQ4 `K/128*68`, W3 `K*3/8`, W2 and E8P `K/4` (E8P: `K/8` u16 words).
pub fn row_code_bytes(codec: Codec, k: u32) -> u32 {
    match codec {
        Codec::Mq4 => (k >> 7) * 68,
        Codec::W3 => ((u64::from(k) * 3) >> 3) as u32,
        Codec::W2 | Codec::E8p => k >> 2,
    }
}
/// Planar scale row bytes `K/128*2` (0 for MQ4, whose scales live in the code rows).
pub fn row_scale_bytes(codec: Codec, k: u32) -> u32 {
    if codec == Codec::Mq4 {
        0
    } else {
        k >> 6
    }
}
/// Whole expert tensor bytes: `M*(row_code_bytes + row_scale_bytes)`.
pub fn tensor_bytes(codec: Codec, m: u32, k: u32) -> u64 {
    u64::from(m) * (u64::from(row_code_bytes(codec, k)) + u64::from(row_scale_bytes(codec, k)))
}
/// Largest `|D|` of one K128 epoch: `128 * max|w| * 8` (E8P: `o` in `[-6, 5]`, so `128 * 6 * 8`).
pub fn dot_bound(codec: Codec) -> u32 {
    match codec {
        Codec::W2 => 3072,
        Codec::W3 => 7168,
        Codec::Mq4 => 8192,
        Codec::E8p => 6144,
    }
}

/// Byte offset of epoch `e`'s code bytes inside a weight row.
pub fn code_offset(codec: Codec, kind: Kind, e: u32) -> u32 {
    match (codec, kind) {
        (Codec::W3, _) => 48 * e,
        (Codec::W2 | Codec::E8p, _) => 32 * e,
        (Codec::Mq4, Kind::Down) => 68 * e + 4,
        (Codec::Mq4, Kind::GateUp) => (e >> 1) * 136 + 8 + (e & 1) * 64,
    }
}
/// Byte offset of epoch `e`'s 16-bit scale inside the scale row (planar: of the
/// scale plane row; MQ4: of the code row).
pub fn scale_offset(codec: Codec, kind: Kind, e: u32) -> u32 {
    match (codec, kind) {
        (Codec::W3 | Codec::W2 | Codec::E8p, _) => 2 * e,
        (Codec::Mq4, Kind::Down) => 68 * e,
        (Codec::Mq4, Kind::GateUp) => (e >> 1) * 136 + (e & 1) * 4,
    }
}
/// The weight loads of one epoch as `(immediate byte offset from code_offset, width in dwords)`,
/// in destination order: operand dword `k` of the epoch is filled by the load whose
/// cumulative width precedes it. MQ4: four `b128` over the 64 bytes. W3: dword
/// at `6i` and `6i+3` per K16 step `i` (low 24 bits used). W2: dword at `4i`
/// and `4i+2` (low 16 bits used). E8P: two `b128` over the epoch's 32 bytes; raw dword
/// `i` (words `2i`, `2i+1`) feeds K16 step `i`.
pub fn code_loads(codec: Codec) -> Vec<(u32, u8)> {
    match codec {
        Codec::Mq4 => (0..4u32).map(|q| (16 * q, 4)).collect(),
        Codec::W3 => (0..8u32)
            .flat_map(|i| [(6 * i, 1), (6 * i + 3, 1)])
            .collect(),
        Codec::W2 => (0..8u32)
            .flat_map(|i| [(4 * i, 1), (4 * i + 2, 1)])
            .collect(),
        Codec::E8p => vec![(0, 4), (16, 4)],
    }
}
/// Scale row loaded by lane `(hi, lr)`: gfx11 `(lr & 14) | hi`; with
/// `wrong` the gfx12 row `8*hi + (lr & 7)` of [`Negative::WrongScaleRow`].
pub fn scale_row(lr: u32, hi: u32, wrong: bool) -> u32 {
    if wrong {
        8 * hi + (lr & 7)
    } else {
        (lr & 14) | hi
    }
}
/// Output row (within the 16-row fragment) of accumulator `j` of half `hi`.
pub fn accumulator_row(j: u32, hi: u32) -> u32 {
    2 * j + hi
}
/// `ds_swizzle_b32 BROADCAST,16,k` lane selector serving accumulator `j`.
pub fn swizzle_lane(j: u32) -> u32 {
    2 * j
}
/// Epoch visiting order of one output: ascending, or descending for [`Negative::ReversedEpoch`].
pub fn epoch_order(spec: Spec, epochs: u32) -> Vec<u32> {
    if spec.negative == Negative::ReversedEpoch {
        (0..epochs).rev().collect()
    } else {
        (0..epochs).collect()
    }
}

// VGPRs.
const TID: u8 = 0;
const XOFF: u8 = 16; // d offset of this lane's slot row (or GATHER_OOB)
const XE: u8 = 17; // per-epoch activation offset
const WOFF: u8 = 18; // lr*ROWB
const HOFF: u8 = 19; // scale row * scale row bytes
const YOFF: u8 = 20;
const WE: u8 = 21; // per-epoch code offset
const HE: u8 = 22; // per-epoch scale offset
const MAGIC8: u8 = 24;
const CACC: u8 = 32; // int32 chains, 8 per tensor
const SUM: u8 = 48; // f32 sums, 8 per tensor
const SCF: u8 = 64; // broadcast row scales, 8 per tensor
const TPROD: u8 = 80; // fold products d*sc
const HF: u8 = 88; // f32 scale of this lane's header row, per tensor
const SET0: u8 = 92; // operand set base

// SGPRs.
const KARG: u8 = 0;
const T0: u8 = 4;
const T1: u8 = 5;
const T64: u8 = 6;
/// Kernarg words 0x00..0x20 on gfx1151 (ptrs 8:9, tiles 10:11, sorted 12:13, xq 14:15); gfx1100: see [`ARGS0_PADDED`].
const ARGS0: u8 = 8;
/// gfx1100 kernarg block: s8:15 would overlap the workgroup ids at s14/s15 (padded user SGPRs); s88:95 is above
/// every other allocation, and keeps the descriptor's SGPR count within the contracts' 104 ceiling (the E8P table
/// literals, formerly s88:95, move to the freed s8..s13, see `e8p::LITS_PADDED`).
const ARGS0_PADDED: u8 = 88;
/// Real user SGPRs (the kernarg pointer) and enabled system SGPRs (workgroup ids X, Y).
const USER_SGPRS: u8 = 2;
const SYSTEM_SGPRS: u8 = 2;
const ARGS1: u8 = 16; // Y 16:17
const M: u8 = 18;
const K: u8 = 19;
const ARGS2: u8 = 20; // x_row_div 20, m_total 21
const ARGS3: u8 = 22; // x_src_rows
const EID: u8 = 23;
const SRD_W: u8 = 24;
const SRD_U: u8 = 28;
const SRD_X: u8 = 32;
const SRD_S: u8 = 40;
const EIDX: u8 = 44;
const ECNT: u8 = 45;
const ROWB: u8 = 46;
const TILE16: u8 = 47;
const WAVE: u8 = 48;
const RBASE: u8 = 49;
const WGX: u8 = 50;
const WGY: u8 = 51;
const LIVE: u8 = 52;
const WPTR: u8 = 58;
const MASKT: [u8; 2] = [60, 62];
const SRD_WS: u8 = 68;
const SRD_US: u8 = 72;
const SROW: u8 = 76;
const XSTR: u8 = 77;
const COFF: u8 = 78;
const SOFF: u8 = 79;
const XEO: u8 = 80;
const TOTB: u8 = 81;
const CBYTES: u8 = 82;

struct Gen {
    spec: Spec,
}

impl Gen {
    /// `.amdhsa_user_sgpr_count` of this arch (`Arch::declared_user_sgprs`: gfx1100 pads user + system to 16).
    fn declared_user_sgprs(&self) -> u8 {
        self.spec.arch.declared_user_sgprs(USER_SGPRS, SYSTEM_SGPRS)
    }
    /// SGPR of the native workgroup id X (`+ 1`: Y): after the DECLARED user SGPRs.
    fn wg_in(&self) -> u8 {
        self.declared_user_sgprs()
    }
    /// First SGPR of the eight kernarg dwords: s8 (the padding-free arches), [`ARGS0_PADDED`] where the
    /// padded user SGPRs put the workgroup ids on s8..s15.
    fn args0(&self) -> u8 {
        if self.declared_user_sgprs() > USER_SGPRS {
            ARGS0_PADDED
        } else {
            ARGS0
        }
    }
    fn label(&self, name: &str) -> String {
        format!(".Lc24_{}_{name}", self.spec.symbol())
    }
    fn tn(&self) -> u8 {
        tensors(self.spec.kind)
    }
    /// The GateUp MQ4 K loop carries the per-epoch lockstep barrier.
    fn lockstep(&self) -> bool {
        self.spec.kind == Kind::GateUp && self.spec.codec == Codec::Mq4 && kloop::LOCK.is_some()
    }
    fn planar(&self) -> bool {
        self.spec.codec != Codec::Mq4
    }
    fn e8p(&self) -> bool {
        self.spec.codec == Codec::E8p
    }
    /// Operand registers of tensor `t` (MQ4/W2/W3: 16 dwords; E8P: the 8 raw code dwords of the epoch).
    fn a(&self, t: u8) -> u8 {
        SET0 + if self.e8p() { 8 } else { 16 } * t
    }
    fn x(&self) -> u8 {
        SET0 + if self.e8p() { 8 } else { 16 } * self.tn()
    }
    fn hs(&self, t: u8) -> u8 {
        self.x() + 16 + t
    }
    /// `d_act` register; E8P: a multiple of four, with `s = d + 1` (one `b64` activation header).
    fn d(&self) -> u8 {
        if self.e8p() {
            self.hs(self.tn()).div_ceil(4) * 4
        } else {
            self.hs(self.tn())
        }
    }
    /// Unpack scratch pair `n` (0 or 1): `(x, y)`.
    fn scratch(&self, n: u8) -> (u8, u8) {
        let base = (self.d() + 1).div_ceil(4) * 4;
        (base + 2 * n, base + 2 * n + 1)
    }
    fn code_srd(&self, t: u8) -> u8 {
        if t == 0 {
            SRD_W
        } else {
            SRD_U
        }
    }
    fn scale_srd(&self, t: u8) -> u8 {
        if !self.planar() {
            self.code_srd(t)
        } else if t == 0 {
            SRD_WS
        } else {
            SRD_US
        }
    }

    fn plan(&self) -> Result<RegPlan, String> {
        let tn = self.tn();
        let acc = self.spec.acc;
        let mut p = RegPlan::new(VGPR_CEILING, 104)?;
        let l = |n: &str| self.label(n);
        let kernel = || Live::Between("entry".into(), l("epilogue"));
        let pro = || Live::Between("entry".into(), l("k_begin"));
        p.v::<1>("tid", TID, pro())?;
        for i in 1..16u8 {
            // The `_acc` epilogue reads `tid & 31` from v2 after the K loop.
            if acc && i == 2 {
                continue;
            }
            p.v::<1>("prologue_tmp", i, pro())?;
        }
        if acc {
            p.v::<1>("lane_id", 2, Live::Whole)?;
        }
        for (name, r) in [
            ("x_off", XOFF),
            ("x_epoch_off", XE),
            ("w_off", WOFF),
            ("h_off", HOFF),
            ("w_epoch_off", WE),
            ("h_epoch_off", HE),
        ] {
            p.v::<1>(name, r, kernel())?;
        }
        p.v::<1>("y_off", YOFF, Live::Whole)?;
        p.v::<8>("magic8", MAGIC8, kernel())?;
        for t in 0..tn {
            p.v::<8>("cacc", CACC + 8 * t, kernel())?;
            p.v::<8>("sum", SUM + 8 * t, Live::Whole)?;
            p.v::<8>("scale_rows", SCF + 8 * t, kernel())?;
            p.v::<1>("scale_f32", HF + t, kernel())?;
        }
        p.v::<8>("fold_t", TPROD, kernel())?;
        if self.e8p() {
            e8p::plan_operands(self, &mut p, &kernel)?;
        } else {
            // Operand set, at each load's granularity: MQ4 codes by b128, planar codes
            // by dword (registered as the WMMA pairs they feed), X by b128.
            for t in 0..tn {
                if self.planar() {
                    for q in 0..8u8 {
                        p.v::<2>("a_nibbles", self.a(t) + 2 * q, kernel())?;
                    }
                } else {
                    for q in 0..4u8 {
                        p.v::<4>("a_nibbles", self.a(t) + 4 * q, kernel())?;
                    }
                }
            }
            for q in 0..4u8 {
                p.v::<4>("x_nibbles", self.x() + 4 * q, kernel())?;
            }
            for t in 0..tn {
                p.v::<1>("scale_dword", self.hs(t), kernel())?;
            }
            p.v::<1>("d_act", self.d(), kernel())?;
            // MQ4 unpacks with one in-place XOR: its scratch pairs are never touched. The pipelined
            // Down path places operand set 1 over them.
            if !pipe::applies(self) {
                for n in 0..2u8 {
                    let (x, y) = self.scratch(n);
                    p.v::<1>("unpack_tmp", x, kernel())?;
                    p.v::<1>("unpack_tmp", y, kernel())?;
                }
            } else {
                pipe::plan_sets(&mut p, self, &kernel)?;
            }
        }
        // Epilogue scratch (aliases dead K-loop ranges) with its real lifetime.
        if acc {
            epilogue::reserve_acc(&mut p, self.spec.kind, &l("epilogue"), &l("end"))?;
        } else {
            epilogue::reserve_bf16(&mut p, self.spec.kind, &l("epilogue"), &l("end"))?;
        }
        // SGPRs.
        p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
        p.s::<1>("wg_x_in", self.wg_in(), Live::Whole)?;
        p.s::<1>("wg_y_in", self.wg_in() + 1, Live::Whole)?;
        p.s::<1>("s_tmp0", T0, kernel())?;
        p.s::<1>("s_tmp1", T1, kernel())?;
        p.s::<2>("s_tmp64", T64, kernel())?;
        p.s::<8>("kernargs_0x00", self.args0(), kernel())?;
        p.s::<2>("y_base", ARGS1, Live::Whole)?;
        // `_acc` recomputes its addresses from M, TILE16 and RBASE after the K loop.
        let keep = |first: bool| if acc && first { Live::Whole } else { kernel() };
        p.s::<2>("kernargs_m_k", M, keep(true))?;
        p.s::<2>("kernargs_0x30", ARGS2, kernel())?;
        p.s::<1>("x_src_rows", ARGS3, kernel())?;
        p.s::<1>("expert", EID, kernel())?;
        for t in 0..tn {
            p.s::<4>("srd_codes", self.code_srd(t), kernel())?;
            if self.planar() {
                p.s::<4>("srd_scales", self.scale_srd(t), kernel())?;
            }
        }
        p.s::<4>("srd_x", SRD_X, kernel())?;
        p.s::<4>("srd_sorted", SRD_S, kernel())?;
        for (name, r) in [
            ("epoch", EIDX),
            ("epochs", ECNT),
            ("row_bytes", ROWB),
            ("wave", WAVE),
            ("wg_x", WGX),
            ("wg_y", WGY),
        ] {
            p.s::<1>(name, r, kernel())?;
        }
        p.s::<1>("tile16", TILE16, keep(true))?;
        p.s::<1>("row_base", RBASE, keep(true))?;
        p.s::<2>("w_ptr", WPTR, kernel())?;
        p.s::<2>("live", LIVE, Live::Whole)?;
        for m in MASKT {
            p.s::<2>("cmp_mask", m, Live::Whole)?;
        }
        for (name, r) in [
            ("x_epoch_stride", XSTR),
            ("code_epoch_off", COFF),
            ("scale_epoch_off", SOFF),
            ("x_epoch_off", XEO),
            ("tensor_bytes", TOTB),
            ("code_plane_bytes", CBYTES),
        ] {
            p.s::<1>(name, r, kernel())?;
        }
        if self.planar() {
            p.s::<1>("scale_row_bytes", SROW, kernel())?;
        }
        // Keep the reservation a whole 8-VGPR granule, so the descriptor's
        // allocation equals the metadata count.
        let top = p.next_free_vgpr();
        if top % 8 != 0 {
            p.v::<1>("granule_pad", (top.div_ceil(8) * 8 - 1) as u8, Live::Whole)?;
        }
        Ok(p)
    }
}

fn epilogue_tail<T: Gfx11>(wg: &mut Wg<'_, T>, g: &Gen) -> Result<(), String> {
    let b = wg.isa();
    // The epilogue starts from an empty ledger: nothing aliased is still being written.
    b.wait_all()?;
    b.label(&g.label("epilogue"))?;
    if g.spec.acc {
        epilogue::emit_acc(b, g.spec.kind, &epilogue::AccRegs::default())?;
    } else {
        let sel = if g.spec.negative == Negative::WrongPackSelector {
            epilogue::PackSelectors::wrong_pack_selector()
        } else {
            epilogue::PackSelectors::frozen()
        };
        epilogue::emit_bf16(b, g.spec.kind, &sel)?;
    }
    Ok(())
}

fn body<T: Gfx11>(wg: &mut Wg<'_, T>, g: &Gen) -> Result<(), String> {
    if g.e8p() {
        // The table region precedes the exit reservation; stage and loop thread its typed state.
        let tables = wg.lds::<e8p::Tables>("e8p_tables", 0, e8p::STAGE_BYTES)?;
        let end = wg.exit(&g.label("end"))?;
        prologue::emit(wg, g, &end)?;
        let tables = e8p::stage(wg, g, tables)?;
        e8p::run_loop(wg, g, tables)?;
        epilogue_tail(wg, g)?;
        return wg.end(end);
    }
    let lock = if g.lockstep() {
        let r0 = wg.lds::<kloop::Lock>("lock0", 0, 4)?;
        let r1 = wg.lds::<kloop::Lock>("lock1", 4, 4)?;
        Some(Ring::new(r0, r1))
    } else {
        None
    };
    let end = wg.exit(&g.label("end"))?;
    prologue::emit(wg, g, &end)?;
    kloop::emit(wg, g, lock)?;
    epilogue_tail(wg, g)?;
    wg.end(end)
}

/// Emit one gfx11 kernel for `spec.arch` (gfx1100 or gfx1151): one instruction path, the arch-typed builder and arch names.
pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    match spec.arch {
        Arch::Gfx1100 => emit_on::<Gfx1100>(spec),
        Arch::Gfx1151 => emit_on::<Gfx1151>(spec),
        a => Err(format!(
            "qwen4_moe_sym_gfx11: no gfx11 target for {}",
            a.name()
        )),
    }
}

fn emit_on<T: Gfx11>(spec: Spec) -> Result<Emitted, String> {
    let g = Gen { spec };
    let kspec = KernelSpec {
        kernel_id: format!("qwen4_moe_sym_{}", spec.arch.name()),
        variant: spec.variant(),
        arch: spec.arch,
        symbol: spec.symbol(),
        kernargs: spec.kernargs(),
        user_sgpr_count: USER_SGPRS,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: spec.threads() as u16,
        group_segment_fixed_size: spec.lds_bytes(),
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, g.plan()?);
    b.enable_delay_alu();
    body(&mut Workgroup::<T, Builder>::new(&mut b)?, &g)?;
    b.finish()
}
