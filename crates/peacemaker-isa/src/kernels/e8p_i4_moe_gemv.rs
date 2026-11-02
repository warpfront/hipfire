// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact F4 read-once C1–C8 lane-union decode GEMV over pointer-addressed **E8P-I4** expert
//! tensors (`WeightCodec::E8pI4`, K-E2), emitted with the checked ISA builder for gfx1201,
//! gfx1100 and gfx1151: two symbols `e8p_i4_moe_gemv_{r256_gate_up,r128_down}` in the module
//! `e8p_i4_moe_gemv_pm_gfx1201`, and on gfx1100 / gfx1151 the two symbols
//! `e8p_i4_moe_gemv_{r256_gate_up,r128_down}_pm_<arch>` in the module `e8p_i4_moe_gemv_pm_<arch>`.
//!
//! The ABI, grid, guards, slot semantics, read-once structure and the F4 reduction tree are those of
//! [`super::mq_i4_moe_gemv`] (`Mq3I4` planar path); only the weight side differs. This file is
//! self-contained (the existing MQ4/W3 modules are untouched, their code objects byte-exact).
//!
//! # gfx1100 arm
//! The gfx1100 arm is the gfx1201 emitter with exactly the architecture-forced deltas, so the gfx1201
//! emitted text and object are unchanged by construction (every delta is behind `arch.gfx12()`):
//! workgroup ids are the system SGPRs after the DECLARED user SGPRs (`KernelSpec::workgroup_ids`, declared in the
//! register plan, the qualified `mq_i4_moe_gemv_gfx1100` pattern; gfx1100's `FeatureUserSGPRInit16Bug` pads the
//! user SGPRs to user + system >= 16, so x / y are `s14` / `s15` and the `_t8` entries' x / y / z are
//! `s13` / `s14` / `s15`, and the kernarg pointer block moves from `s[8:15]` to `s[88:95]`; gfx1151 keeps `s2` / `s3`
//! (/ `s4`) and `s[8:15]`) instead of `ttmp9` / `ttmp7[15:0]`
//! (the y mask does not exist: the system SGPR is the whole id); the F16 scale widening keeps the
//! qualified `v_cvt_f32_f16_e64 vD, vD.l` spelling of both architectures; the barrier is the builder's
//! `s_barrier` and the waits are the builder-inserted gfx11 `s_waitcnt` / `s_waitcnt_depctr` /
//! `s_delay_alu` (never hand-placed). The gfx1100 symbols carry a `_pm_gfx1100` suffix so the two architectures'
//! exports never collide. The arithmetic, the LDS table stage, the decode network and the reduction
//! tree are unchanged.
//!
//! # gfx1151 arm
//! gfx1151 (RDNA3.5, wave32) is the same gfx11 ISA family as gfx1100 and takes the gfx11 branch of every
//! fork above unchanged: its emitted text is the gfx1100 text with only the arch-forced deltas
//! (symbol / module / label names carrying `gfx1151` instead of `gfx1100`, the `.amdgcn_target` /
//! metadata target, and the builder's hazard model: `VALUTransUseHazard` is a gfx1100-only feature, so
//! the `s_waitcnt_depctr depctr_va_vdst(0)` the builder places on gfx1100 before the `v_mul_f32` that
//! consumes each entry's `v_rcp_iflag_f32` result is absent on gfx1151), the same instruction sequence otherwise,
//! register plan, LDS tables, reduction trees, ABI and launch geometry. The gfx1100 and gfx1201 emitted
//! text are unchanged.
//!
//! # Frozen 96-byte kernarg ABI (identical to `mq_i4_moe_gemv`)
//! u64 pointers `expert_weight_ptrs@0` (table indexed by expert id; each entry is the planar E8P
//! tensor base of that expert, never `expert_id * stride`), `expert_ids@8` (`u32[U]`, ascending),
//! `expert_offsets@16` (`u32[U+1]`), `slot_ids@24` (`u32[10*C]`, original slots `lane*10+rank`),
//! `Xrot@32` (F32 rotated input), `Y0@40`, `Y1@48`; u32 `M@56`, `K@60`, `U@64`, `C@68`,
//! `x_row_div@72`, `x_src_rows@76`, `x_row_stride@80`, `y_slot_stride@84`, `n_experts@88` (the extent of the
//! `expert_weight_ptrs` table, a nonnegative signed-i32 count) and `reserved@92` (zero, pads the block to 8
//! bytes). Strides are F32 elements.
//!
//! Launch: grid `[M, U, 1]` (x = output row, y = distinct expert), block `[32, 1, 1]` (one wave32 per
//! workgroup). Production shape: gate `M` is the full fused gate+up row count (`M = 1280`, `mi = M/2`),
//! `K = 2560`, `x_row_div = 10`, `Y0/Y1` split at `M/2`; down `M = 2560`,
//! `K = 640`, `x_row_div = 1`, unweighted `Y0`. A workgroup whose row `>= M`, expert index `>= U`,
//! whose slot count `n = offsets[u+1] - offsets[u]` is `0`, exceeds [`MAX_SLOTS`]` = 8` or has
//! `offsets[u+1] > 10*C`, or whose expert id is `>= n_experts` (one unsigned compare, so a negative id too; made
//! before `expert_id * 8`, the table lookup and the LDS stage) returns without a store (before any LDS access).
//! `n_experts` is read into `s4`, a scalar register no other instruction of the entry defines or reads.
//! Domain (host-validated, never re-checked): `K > 0`, `K % 128 == 0` (`K % 256 == 0` gate), `x_row_div >= 1`,
//! slots `< 2^22`, `x_src_rows*x_row_stride*4`, `M*K/4 + M*K/64` and every output offset below `2^32`, every
//! scale a finite FP16 `d >= 0x0400` (the writer guard), `n_experts` the table extent (`<= i32::MAX`).
//!
//! # Wire layout
//! `expert_weight_ptrs[e]` is the planar E8P-I4 tensor `[M, K]` of `qwen4_moe_sym_e8p`: `M*K/8`
//! little-endian `u16` words (row-major; word = `idx | sign7 << 8 | shift << 15`, eight weights),
//! immediately followed by `M*K/128` FP16 scales (row-major). Code row bytes `K/4`; one K128 group is
//! 32 bytes (16 words) of a row.
//!
//! # Lane layout and exactness
//! Lane `l` owns the four weights `g*128 + 4l .. +3` of every 128-column group `g`: word `l >> 1` of
//! the group (group byte `2*(l >> 1)`, an exact aligned `global_load_u16`) and half `l & 1`. Both
//! lanes of a pair decode the whole word through the staged tables and keep their half
//! (`unpack_model`): `x = LUT[idx] ^ MASK[sign7]`, `y = MASK & 0x11111111`, `n = shift ? 0xeeeeeeef : 0`,
//! operand `x + y + n` (eight two's-complement nibbles `o = (v-1)/2`), `t = operand >> (16*half)`,
//! element `i` `v = 2*bfe_i32(t, 4i, 4) + 1` (odd, `|v| <= 11`), weight `w = RN((f32(d)*0.25) * f32(v))`
//! (the scale multiply is an exact power-of-two scaling because `d >= 0x0400`; no FP16 `d/4`).
//! Per group `p = RN(w0*x0)` (`v_mul_f32`) then `p = fma(w_j, x_j, p)` for `j = 1..3` (`v_fmac_f32`);
//! the partial is added separately into the chain accumulator `acc[g&3]` (gate) or the single chain
//! (down); accumulators start at `+0`; lane value gate `RN(RN(a0+a1)+RN(a2+a3))`; then the
//! snapshot shuffle-down at offsets 16, 8, 4, 2, 1 with out-of-range lanes unchanged
//! (`v_add_f32` into scratch, `v_cndmask_b32` mask select), lane 0 stores. Bit-equal to the frozen
//! K-E0 twin `decode_gemv_f32_lanepair`. No F32-IU4 prefill magic fold exists in the GEMV.
//!
//! # Read-once C8
//! The weight word, scale and table lookups of a group are issued and decoded once; the up-to-eight
//! slots of the expert then run their own multiply/FMA chains on the shared weights, each in its own
//! accumulator register(s). Slots are skipped with scalar branches (`i >= n`), never by re-reading
//! weights: no global/LDS load sits inside a slot branch. The weight-class load count of a group
//! (`u16` word, `u16` scale, two `ds_load_b32`) is the same for `n = 1` and `n = 8`.
//!
//! # LDS stage (1,536 B, once per workgroup; `STAGE_BYTES`)
//! The K-E1 tables, byte-identical: `LUT[256]` at 0 (`build_lut256u32`, one u32 of eight nibbles
//! `2|g|` per abs row, lane 7 two's-complement negated on odd-parity rows) and `MASK[128]` at 1024
//! (`build_mask128u32`, `0xF` in the lanes `sign7` negates, lane 7 = parity of `popcount(sign7)`).
//! One checked LDS slot (`e8p_tables`, base 0, 1536 B) is stored by the wave, drained and published
//! at one workgroup barrier (`Transition::Ready`) and afterwards only read. MASK is computed
//! arithmetically (xor/shift parity, no `v_bcnt_u32_b32`: the native encoder lacks it; see
//! [`stage_mask_model`]), LUT is eight chunks of 32 dwords from SGPR literals combined into one
//! VGPR with `v_writelane_b32` per chunk. Store addresses are `4 * tid` derived in the entry block
//! (before the first branch); table reads use immediate-masked addresses (`& 0x3fc` LUT, `& 0x1fc`
//! MASK + 1024), both forms the `pm_check::lds_bounds` check proves: LUT reads end `<= 1024`, MASK
//! reads end `<= 1536`.
//!
//! # Registers (gate / down)
//! VGPR 112 / 88 (granule 8), SGPR 88 (the 80 SGPRs of `mq_i4_moe_gemv` before its `n_experts` register plus eight
//! literal staging SGPRs at 80..87; `n_experts` lives in the unused `s4`, so the ceiling is unchanged). The stage scratch
//! lives in the dead X-value block.
//!
//! # Memory bounds (no over-read, no tail pad)
//! Weight word: group byte `g*32 + 2*(l>>1)`, 2 bytes, inside `[0, M*K/4)` (max end `K/4` of the
//! row); scale: `M*K/4 + 2*(r*K/128 + g)`, 2 bytes, inside `[M*K/4, M*K/4 + M*K/64)`; the last byte
//! read of the tensor is its last byte, so the required tail padding is zero ([`word_end`],
//! [`scale_end`]). X rows go through the raw descriptor of `x_src_rows*x_row_stride*4` bytes (a row
//! at or past it reads 0), exactly as the incumbent.
//!
//! # Cn eight-output-wave workgroup ([`emit_cn_rows8`], gfx1201 only, candidate; production dispatch unchanged)
//! The one-wave entries above stage the whole 1,536-byte table once **per output row** (256 `s_mov` + 256
//! `v_writelane` + 67 MASK VALU per wave). Production K gives 20 gate / 5 down K128 groups; the gate body
//! has seven static group instances (four loop instances plus three possible tails).
//! The candidate serves eight rows per workgroup: launch grid `[ceil(M/8), U, 1]`, block `[256, 1, 1]`, `tid = v0`,
//! `lane = tid & 31`, wave `w = tid >> 5` (uniform, `v_readfirstlane_b32` into `s5`), `row = 8 * wg_x + w`. The eight
//! waves publish ONE table copy cooperatively (`stage_rows8`): each wave writes only its own LUT chunk `w`
//! (32 `s_mov` + 32 `v_writelane`, behind a uniform wave-index branch) and waves 0-3 each write one 32-entry quarter of
//! MASK, after which all eight waves meet at the one barrier. Everything workgroup-uniform (`row`-independent) still
//! returns before the stage and its barrier: expert index `>= U`, `n == 0` or `n > 8`, `offsets[u+1] > 10*C`, expert id
//! `>= n_experts`; only `row >= M` moves behind the barrier (such waves load nothing and store nothing). Group body, decode
//! network, per-slot chains, fold, shuffle tree and stores are the incumbent's text on the wave's own registers (the
//! wave's `lane` is `tid & 31`, kept in the dead `x_address` register); the same 96-byte kernarg ABI and VGPR / SGPR
//! counts as the incumbent entry (112 / 88 VGPR gate / down, SGPR 88), LDS 1,536 B per workgroup instead of per row.
//! Both new SGPR/VGPR roles reuse registers the incumbent already reserves, so the resource ceilings are unchanged.
//! No speedup is claimed.

use crate::arch::WgIdSource;
use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{
    add64, bload, gather_offset, lit, op, s, s_add_i32, smem, sop, sr, srd_tail, v, GatherTemps,
};
use crate::kernels::qwen4_moe_sym::e8p::{
    build_lut256u32, build_mask128u32, STAGE_BYTES as K_E1_STAGE_BYTES,
};
use crate::lds::Transition;
use crate::plan::Access;
use crate::reg::{Kind as RKind, Live, RegRef};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// The kernarg segment size.
pub const KERNARG_BYTES: u32 = 96;
/// Workgroup size (threads): one wave32.
pub const WORKGROUP_SIZE: u32 = 32;
/// Most distinct original slots one expert workgroup serves (C8: at most one expert row per lane).
pub const MAX_SLOTS: u32 = 8;
/// Planar scale / activation group width in columns.
pub const GROUP: u32 = 128;
/// `LUT[256]` bytes (offset 0).
pub const LUT_BYTES: u32 = 256 * 4;
/// `MASK[128]` bytes (offset [`LUT_BYTES`]).
pub const MASK_BYTES: u32 = 128 * 4;
/// Static LDS of every entry: the per-workgroup table stage (equal to the K-E1 `STAGE_BYTES`).
pub const STAGE_BYTES: u32 = LUT_BYTES + MASK_BYTES;
const _: () = assert!(STAGE_BYTES == K_E1_STAGE_BYTES && STAGE_BYTES == 1536);
/// Weights (and X elements) one lane owns per group.
pub const LANE_ELEMS: u8 = 4;

/// Nibble ones: the `+1` of a two's-complement negation and the shift unit.
const ONES: u32 = 0x1111_1111;
/// `-ONES mod 2^32`: the shift subtraction as an addend.
const NEG_ONES: u32 = 0xeeee_eeef;
/// `0.25f` bits.
const QUARTER: u32 = 0x3e80_0000;

/// Projection kind; selects the reduction tree (the profile's rotation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// Gate/up (`K2560`, rotation 256): four group chains `acc[g&3]`, fold `(a0+a1)+(a2+a3)`, Y0/Y1 split at `M/2`.
    GateUp,
    /// Down (`K640`, rotation 128): one chain, Y0 only.
    Down,
}
impl Kind {
    pub const ALL: [Kind; 2] = [Kind::GateUp, Kind::Down];
    pub fn tag(self) -> &'static str {
        match self {
            Self::GateUp => "r256_gate_up",
            Self::Down => "r128_down",
        }
    }
    /// Accumulator chains per slot.
    pub fn chains(self) -> u8 {
        match self {
            Self::GateUp => 4,
            Self::Down => 1,
        }
    }
}

/// One entry: `(arch, kind)`; gfx1201, gfx1100 and gfx1151.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
    pub kind: Kind,
}
impl Spec {
    /// The exported symbol: gfx1201 `e8p_i4_moe_gemv_<kind>`, gfx1100 / gfx1151 `e8p_i4_moe_gemv_<kind>_pm_<arch>`.
    pub fn symbol(&self) -> String {
        if self.arch.gfx12() {
            format!("e8p_i4_moe_gemv_{}", self.tag())
        } else {
            format!("e8p_i4_moe_gemv_{}_pm_{}", self.tag(), self.arch.name())
        }
    }
    /// `<kind>`: the `--epi` name, the label prefix and the contract file tag.
    pub fn tag(&self) -> String {
        self.kind.tag().to_owned()
    }
    pub fn module(arch: Arch) -> String {
        format!("e8p_i4_moe_gemv_pm_{}", arch.name())
    }
    pub fn validate(&self) -> R {
        if !matches!(self.arch, Arch::Gfx1201 | Arch::Gfx1100 | Arch::Gfx1151) {
            return Err(
                "e8p_i4_moe_gemv builder is exact-gfx1201 / exact-gfx1100 / exact-gfx1151 only"
                    .into(),
            );
        }
        Ok(())
    }
    /// Group instances in the emitted body: gate four loop groups plus three tail groups, down one.
    pub fn group_instances(&self) -> u32 {
        match self.kind {
            Kind::GateUp => 7,
            Kind::Down => 1,
        }
    }
}

/// Every entry of one architecture in module order: gate, down.
pub fn module_specs(arch: Arch) -> Vec<Spec> {
    Kind::ALL
        .into_iter()
        .map(|kind| Spec { arch, kind })
        .collect()
}

// ----------------------------------------------------------------------- SGPRs
// s0:1 kernarg pointer; gfx1201 workgroup ids are ttmp9 (x) and ttmp7[15:0] (y); gfx1100 / gfx1151 ids are the
// system SGPRs after the declared user SGPRs: s2 (x) / s3 (y) (/ s4 z) on gfx1151, and on gfx1100, whose
// `FeatureUserSGPRInit16Bug` pads the user SGPRs to user + system >= 16 (`Arch::declared_user_sgprs`),
// s14 / s15 (x / y, no z entry) and s13 / s14 / s15 (x / y / z, the `_t8` entries).
/// User SGPRs: the kernarg pointer pair.
const USER_SGPRS: u8 = 2;
const KARG: u8 = 0;
/// Kernargs (s_load_b256 @0, b256 @0x20, b128 @0x40, b64 @0x50): `Lay::wptrs()`, `+2`, `+4`, `+6`.
/// gfx1151 keeps them at s8..15; gfx1100's workgroup ids occupy s13..15, so the block lives past the plan's
/// other SGPRs, at s[88:95] ([`Lay::sgprs`] 96).
const WPTRS: u8 = 8;
const WPTRS_GFX1100: u8 = 88;
const XROT: u8 = 16;
const Y0: u8 = 18;
const Y1: u8 = 20;
const SM: u8 = 22;
const SK: u8 = 23;
const SU: u8 = 24;
const SC: u8 = 25;
const SDIV: u8 = 26;
const SXSRC: u8 = 27;
const SXSTR: u8 = 28;
const SYSTR: u8 = 29;
const ROW: u8 = 30;
const UIDX: u8 = 31;
const OFF0: u8 = 32;
const OFF1: u8 = 33;
const EID: u8 = 34;
const NS: u8 = 35;
const TPTR: u8 = 36;
const WPTR: u8 = 38;
const SRDX: u8 = 40;
const ROW_PTR: u8 = 44;
const SCALE_PTR: u8 = 46;
const GROUPS: u8 = 48;
const ROW_BYTES: u8 = 49;
const CODES_BYTES: u8 = 50;
const TMP: u8 = 51;
const TMP2: u8 = 52;
const GI: u8 = 53;
const QUADS: u8 = 54;
const G_CODE: u8 = 55;
const G_X: u8 = 56;
const G_SCALE: u8 = 57;
const YSEL: u8 = 58;
const ROWIN4: u8 = 60;
const SXS4: u8 = 61;
const SYS4: u8 = 62;
const NSM1: u8 = 63;
const LIVE: u8 = 64;
const MASK: u8 = 66;
const SHM: u8 = 68;
const TL: u8 = 73;
const MI: u8 = 77;
const ROWM: u8 = 78;
const ROWIN: u8 = 79;
/// Eight literal staging SGPRs of the LUT stage.
const LITS: u8 = 80;
/// SGPRs the plan declares (the honest ceiling, `sgpr_max` of the contract) on gfx1151 / gfx1201; gfx1100 declares
/// [`SGPRS_GFX1100`] (the relocated kernarg pointer block, [`WPTRS_GFX1100`]).
pub const SGPRS: u8 = 88;
/// SGPRs the gfx1100 plan declares.
pub const SGPRS_GFX1100: u8 = WPTRS_GFX1100 + 8;
/// `n_experts@88`, the extent of the `expert_weight_ptrs` table: `s4`, the first SGPR after the system workgroup ids,
/// which no other instruction of the entry touches (the plan's SGPR ceiling and the physical assignment of every
/// other register are unchanged). An `expert_ids` entry at or past it is never used to index the table.
const SNEXP: u8 = 4;
/// The uniform wave index of the eight-output-wave entry (`tid >> 5` via `v_readfirstlane_b32`): `s5`, a gfx12
/// SGPR no other instruction of the entry names (declared in the plan only for that entry).
const WAVE: u8 = 5;
/// Tiled gfx1151 z id occupies s4; move the expert extent to the otherwise unused s6.
const TILE_NEXP: u8 = 6;
const TILE: u8 = 7;
/// Fixed union descriptor capacity and byte stride, including both codec buckets.
const TILE_SLOTS: u32 = 80;
const TILE_BYTES: u32 = 1928;

// ----------------------------------------------------------------------- VGPRs
const LANE: u8 = 0;

/// VGPR layout of one entry, assigned sequentially so no range can overlap.
#[derive(Clone, Copy, Debug)]
struct Lay {
    arch: Arch,
    /// Eight-output-wave workgroup (`Lay::lane` is then `tid & 31`, not the work-item id).
    rows8: bool,
    rowtiles: bool,
    /// VGPR holding the lane index `0..32`.
    lane: u8,
    /// Accumulator chains per slot.
    chains: u8,
    caddr: u8,
    saddr: u8,
    xl: u8,
    cl: u8,
    hsh: u8,
    xb: u8,
    xa: u8,
    wrd: u8,
    d: u8,
    la: u8,
    ma: u8,
    lv: u8,
    mv: u8,
    sh: u8,
    opr: u8,
    dot: u8,
    exch: u8,
    shuf: u8,
    lds: u8,
    w: u8,
    x: u8,
    acc: u8,
    xo: u8,
    ya: u8,
    end: u8,
}
impl Lay {
    fn new(spec: Spec, rows8: bool) -> Self {
        let chains = spec.kind.chains();
        let mut n = 1u8;
        let mut take = |len: u8, align: u8| {
            n = n.div_ceil(align) * align;
            let base = n;
            n += len;
            base
        };
        let caddr = take(1, 1);
        let saddr = take(1, 1);
        let xl = take(1, 1);
        let cl = take(1, 1);
        let hsh = take(1, 1);
        let xb = take(1, 1);
        let xa = take(1, 1);
        let wrd = take(1, 1);
        let d = take(1, 1);
        let la = take(1, 1);
        let ma = take(1, 1);
        let lv = take(1, 1);
        let mv = take(1, 1);
        let sh = take(1, 1);
        let opr = take(1, 1);
        let dot = take(2, 1);
        let exch = take(1, 1);
        let shuf = take(5, 1);
        let lds = take(1, 1);
        let w = take(LANE_ELEMS, 4);
        let x = take(MAX_SLOTS as u8 * LANE_ELEMS, 4);
        let acc = take(MAX_SLOTS as u8 * chains, 1);
        let xo = take(MAX_SLOTS as u8, 1);
        let ya = take(MAX_SLOTS as u8, 1);
        let end = n.div_ceil(8) * 8;
        Self {
            arch: spec.arch,
            rows8,
            rowtiles: false,
            // The lane index: the work-item id in the one-wave entry; `tid & 31`, held in `x_address`
            // (dead until the first group; recomputed in the epilogue), in the eight-wave entry.
            lane: if rows8 { xa } else { LANE },
            chains,
            caddr,
            saddr,
            xl,
            cl,
            hsh,
            xb,
            xa,
            wrd,
            d,
            la,
            ma,
            lv,
            mv,
            sh,
            opr,
            dot,
            exch,
            shuf,
            lds,
            w,
            x,
            acc,
            xo,
            ya,
            end,
        }
    }
    /// First SGPR of the gfx11 workgroup ids: the declared user SGPRs (`Arch::declared_user_sgprs`; the `_t8` entries
    /// also enable the z id).
    fn wg_x(&self) -> u8 {
        let system = if self.rowtiles { 3 } else { 2 };
        self.arch.declared_user_sgprs(USER_SGPRS, system)
    }
    fn wg_y(&self) -> u8 {
        self.wg_x() + 1
    }
    fn wg_z(&self) -> u8 {
        self.wg_x() + 2
    }
    /// The `expert_weight_ptrs` / `expert_ids` / `expert_offsets` / `slot_ids` kernarg block (`s_load_b256`).
    fn wptrs(&self) -> u8 {
        if self.arch.min_user_system_sgprs() > 0 {
            WPTRS_GFX1100
        } else {
            WPTRS
        }
    }
    fn eids(&self) -> u8 {
        self.wptrs() + 2
    }
    fn offs(&self) -> u8 {
        self.wptrs() + 4
    }
    fn slots(&self) -> u8 {
        self.wptrs() + 6
    }
    /// SGPRs the plan declares.
    fn sgprs(&self) -> u8 {
        if self.arch.min_user_system_sgprs() > 0 {
            SGPRS_GFX1100
        } else {
            SGPRS
        }
    }
    fn label(&self, spec: Spec, name: &str) -> String {
        if self.rowtiles {
            format!(".Le8pi4moe_{}_t8_{name}", spec.tag())
        } else {
            format!(".Le8pi4moe_{}_{name}", spec.tag())
        }
    }
    /// First X register of slot `i` (four consecutive registers).
    fn xr(&self, i: u32) -> u8 {
        self.x + i as u8 * LANE_ELEMS
    }
    /// Accumulator register of slot `i`, chain `c`.
    fn ac(&self, i: u32, c: u8) -> u8 {
        self.acc + i as u8 * self.chains + c
    }
}

/// Declare `len` registers from `base` as the legal ranges the plan allows (widths 1, 2, 4 or 8, SGPR
/// ranges aligned to their width): greedy largest-first granules of the same name, so every multi-register
/// access (kernarg loads, pointer pairs, 128-bit X loads) lies inside one declared range.
fn declare(regs: &mut RegPlan, name: &str, kind: RKind, base: u8, len: u8) -> R {
    let (mut at, end) = (u16::from(base), u16::from(base) + u16::from(len));
    while at < end {
        let width = [8u16, 4, 2, 1]
            .into_iter()
            .find(|&w| at + w <= end && (kind == RKind::V || at % w == 0))
            .unwrap_or(1);
        regs.add_range(name, kind, at as u8, width as u8, Live::Whole)?;
        at += width;
    }
    Ok(())
}
fn plan(l: &Lay, arch: Arch) -> Result<RegPlan, String> {
    let mut regs = RegPlan::new(u16::from(l.end), u16::from(l.sgprs()))?;
    if !arch.gfx12() {
        for (name, base) in [("workgroup_id_x", l.wg_x()), ("workgroup_id_y", l.wg_y())] {
            declare(&mut regs, name, RKind::S, base, 1)?;
        }
        if l.rowtiles {
            declare(&mut regs, "workgroup_id_z", RKind::S, l.wg_z(), 1)?;
        }
    }
    for (name, base, len) in [
        ("kernarg", KARG, 2),
        ("kernel_arguments_ptrs", l.wptrs(), 8),
        ("kernel_arguments_x_y_m_k", XROT, 8),
        ("kernel_arguments_u_c_div_src", SU, 4),
        ("kernel_arguments_strides", SXSTR, 2),
        (
            "kernel_arguments_n_experts",
            if l.rowtiles { TILE_NEXP } else { SNEXP },
            1,
        ),
        ("row", ROW, 1),
        ("expert_index", UIDX, 1),
        ("expert_offsets_pair", OFF0, 2),
        ("expert_id", EID, 1),
        ("slot_count", NS, 1),
        ("table_pointer", TPTR, 2),
        ("expert_pointer", WPTR, 2),
        ("x_descriptor", SRDX, 4),
        ("row_pointer", ROW_PTR, 2),
        ("scale_pointer", SCALE_PTR, 2),
        ("groups", GROUPS, 1),
        ("row_bytes", ROW_BYTES, 1),
        ("codes_bytes", CODES_BYTES, 1),
        ("temp", TMP, 1),
        ("temp2", TMP2, 1),
        ("group_index", GI, 1),
        ("quad_end", QUADS, 1),
        ("group_code_offset", G_CODE, 1),
        ("group_x_offset", G_X, 1),
        ("group_scale_offset", G_SCALE, 1),
        ("y_select", YSEL, 2),
        ("row_in_y_bytes", ROWIN4, 1),
        ("x_stride_bytes", SXS4, 1),
        ("y_stride_bytes", SYS4, 1),
        ("slots_minus_one", NSM1, 1),
        ("gather_live", LIVE, 2),
        ("gather_mask", MASK, 2),
        ("shuffle_masks", SHM, 5),
        ("lane_temps", TL, 4),
        ("half_rows", MI, 1),
        ("row_minus_half", ROWM, 1),
        ("row_in_y", ROWIN, 1),
        ("table_literals", LITS, 8),
    ] {
        declare(&mut regs, name, RKind::S, base, len)?;
    }
    if l.rows8 {
        declare(&mut regs, "wave_index", RKind::S, WAVE, 1)?;
    }
    if l.rowtiles {
        declare(&mut regs, "row_tile", RKind::S, TILE, 1)?;
    }
    for (name, base, len) in [
        ("lane", LANE, 1),
        ("code_address", l.caddr, 1),
        ("scale_address", l.saddr, 1),
        ("x_lane_offset", l.xl, 1),
        ("code_lane_offset", l.cl, 1),
        ("half_shift", l.hsh, 1),
        ("x_base", l.xb, 1),
        ("x_address", l.xa, 1),
        ("word", l.wrd, 1),
        ("scale", l.d, 1),
        ("lut_address", l.la, 1),
        ("mask_address", l.ma, 1),
        ("lut_value", l.lv, 1),
        ("mask_value", l.mv, 1),
        ("shift_addend", l.sh, 1),
        ("operand", l.opr, 1),
        ("dots", l.dot, 2),
        ("exchange", l.exch, 1),
        ("shuffle_addresses", l.shuf, 5),
        ("lds_store_address", l.lds, 1),
        ("weights", l.w, LANE_ELEMS),
        ("x_values", l.x, MAX_SLOTS as u8 * LANE_ELEMS),
        ("accumulators", l.acc, MAX_SLOTS as u8 * l.chains),
        ("slot_x_offsets", l.xo, MAX_SLOTS as u8),
        ("slot_y_addresses", l.ya, MAX_SLOTS as u8),
    ] {
        declare(&mut regs, name, RKind::V, base, len)?;
    }
    let top = l.xo.max(l.ya) + MAX_SLOTS as u8;
    if l.end > top {
        declare(&mut regs, "granule_pad", RKind::V, top, l.end - top)?;
    }
    Ok(regs)
}

/// Emit one entry.
pub fn emit(spec: Spec) -> Result<Emitted, String> {
    emit_with(spec, false, false)
}

/// The shared emitter: `rows8` selects the eight-output-wave workgroup (module docs), `false` the incumbent text.
fn emit_with(spec: Spec, rows8: bool, rowtiles: bool) -> Result<Emitted, String> {
    spec.validate()?;
    let mut l = Lay::new(spec, rows8);
    l.rowtiles = rowtiles;
    let kernargs = KernargLayout::new(KERNARG_BYTES)
        .pointer_access("expert_weight_ptrs", 0, Access::ReadOnly)
        .pointer_access("expert_ids", 8, Access::ReadOnly)
        .pointer_access("expert_offsets", 16, Access::ReadOnly)
        .pointer_access("slot_ids", 24, Access::ReadOnly)
        .pointer_access("Xrot", 32, Access::ReadOnly)
        .pointer_access("Y0", 40, Access::WriteOnly)
        .pointer_access("Y1", 48, Access::WriteOnly)
        .hidden("M", 56, 4, "by_value")
        .hidden("K", 60, 4, "by_value")
        .hidden("U", 64, 4, "by_value")
        .hidden("C", 68, 4, "by_value")
        .hidden("x_row_div", 72, 4, "by_value")
        .hidden("x_src_rows", 76, 4, "by_value")
        .hidden("x_row_stride", 80, 4, "by_value")
        .hidden("y_slot_stride", 84, 4, "by_value")
        .hidden("n_experts", 88, 4, "by_value")
        .hidden("reserved", 92, 4, "by_value");
    let (symbol, variant, workgroup_size) = if rowtiles {
        (
            format!("{}_t8", spec.symbol()),
            format!("{}_t8", spec.tag()),
            CN_ROWS8_WORKGROUP_SIZE,
        )
    } else if rows8 {
        (
            cn_rows8_symbol(spec),
            format!("{}_cn_rows8", spec.tag()),
            CN_ROWS8_WORKGROUP_SIZE,
        )
    } else {
        (spec.symbol(), spec.tag(), WORKGROUP_SIZE)
    };
    let kspec = KernelSpec {
        kernel_id: "e8p_i4_moe_gemv".into(),
        variant,
        arch: spec.arch,
        symbol,
        kernargs,
        user_sgpr_count: USER_SGPRS,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: workgroup_size as u16,
        group_segment_fixed_size: STAGE_BYTES,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan(&l, spec.arch)?);
    b.enable_delay_alu();
    if rowtiles {
        b.enable_workgroup_id_z();
    }
    if b.lds_slot("e8p_tables", 0, STAGE_BYTES)? != 0 {
        return Err("LDS slot order".into());
    }
    prologue(&mut b, spec, &l)?;
    b.wait_all()?;
    if rows8 {
        stage_rows8(&mut b, spec, &l)?;
        // Rows past `M` leave only now: every wave of a live workgroup runs the table barrier above.
        sop(
            &mut b,
            format!("s_cmp_ge_u32 s{ROW}, s{SM}"),
            &[],
            &[ROW, SM],
        )?;
        branch(&mut b, &format!("s_cbranch_scc1 {}", l.label(spec, "exit")))?;
    } else {
        stage(&mut b, &l)?;
    }
    match spec.kind {
        Kind::GateUp => {
            // Groups strictly ascending; group g feeds chain g & 3: whole quads in the
            // loop, then up to three tail groups on chains 0, 1, 2.
            sop(
                &mut b,
                format!("s_and_b32 s{QUADS}, s{GROUPS}, -4"),
                &[QUADS],
                &[GROUPS],
            )?;
            sop(&mut b, format!("s_cmp_eq_u32 s{QUADS}, 0"), &[], &[QUADS])?;
            branch(&mut b, &format!("s_cbranch_scc1 {}", l.label(spec, "tail")))?;
            b.loop_(&l.label(spec, "quad"), |b| {
                for chain in 0..4 {
                    group(b, spec, &l, chain, &format!("q{chain}"))?;
                }
                sop(
                    b,
                    format!("s_cmp_lt_u32 s{GI}, s{QUADS}"),
                    &[],
                    &[GI, QUADS],
                )?;
                branch(b, &format!("s_cbranch_scc1 {}", l.label(spec, "quad")))
            })?;
            b.label(&l.label(spec, "tail"))?;
            for chain in 0..3 {
                sop(
                    &mut b,
                    format!("s_cmp_lt_u32 s{GI}, s{GROUPS}"),
                    &[],
                    &[GI, GROUPS],
                )?;
                branch(
                    &mut b,
                    &format!("s_cbranch_scc0 {}", l.label(spec, "reduce")),
                )?;
                group(&mut b, spec, &l, chain, &format!("t{chain}"))?;
            }
        }
        Kind::Down => {
            sop(&mut b, format!("s_cmp_eq_u32 s{GROUPS}, 0"), &[], &[GROUPS])?;
            branch(
                &mut b,
                &format!("s_cbranch_scc1 {}", l.label(spec, "reduce")),
            )?;
            b.loop_(&l.label(spec, "group"), |b| {
                group(b, spec, &l, 0, "g")?;
                sop(
                    b,
                    format!("s_cmp_lt_u32 s{GI}, s{GROUPS}"),
                    &[],
                    &[GI, GROUPS],
                )?;
                branch(b, &format!("s_cbranch_scc1 {}", l.label(spec, "group")))
            })?;
        }
    }
    b.label(&l.label(spec, "reduce"))?;
    epilogue(&mut b, spec, &l)?;
    b.finish()
}

/// The two entries as one code object (`Spec::module`).
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = module_specs(arch)
        .into_iter()
        .map(emit)
        .collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, &Spec::module(arch))?;
    Ok((emitted, text, proof))
}

/// Name of the gfx1201 Cn eight-output-wave module ([`emit_cn_rows8_module`]).
pub const CN_ROWS8_MODULE: &str = "e8p_i4_moe_gemv_cn_rows8_pm_gfx1201";
/// Output rows (waves) one workgroup of the eight-output-wave entry serves.
pub const CN_ROWS8_ROWS: u32 = 8;
/// Workgroup size (threads) of the eight-output-wave entry: eight wave32.
pub const CN_ROWS8_WORKGROUP_SIZE: u32 = CN_ROWS8_ROWS * 32;

/// The exported symbol of the Cn eight-output-wave entry of `spec`: the incumbent [`Spec::symbol`] plus `_cn_rows8`.
pub fn cn_rows8_symbol(spec: Spec) -> String {
    format!("{}_cn_rows8", spec.symbol())
}

/// One entry of the gfx1201 **Cn eight-output-wave** candidate (module docs, "Cn eight-output-wave workgroup"):
/// the incumbent network with one workgroup serving eight output rows, one wave32 each, over a single cooperatively
/// staged E8P table. Launch grid `[ceil(M/8), U, 1]`, block `[256, 1, 1]`, wave `w` of workgroup `x` = row `8x + w`;
/// same 96-byte ABI (final reserved word included), guards, per-row arithmetic, fold and stores as [`emit`];
/// symbol [`cn_rows8_symbol`]. Errors for any spec that is not gfx1201.
pub fn emit_cn_rows8(spec: Spec) -> Result<Emitted, String> {
    if spec.arch != Arch::Gfx1201 {
        return Err(format!(
            "e8p_i4_moe_gemv cn_rows8 is gfx1201 only, got {:?}",
            spec.arch
        ));
    }
    emit_with(spec, true, false)
}

/// The two Cn eight-output-wave entries (gate, down) as one code object [`CN_ROWS8_MODULE`].
pub fn emit_cn_rows8_module() -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String>
{
    let emitted = module_specs(Arch::Gfx1201)
        .into_iter()
        .map(emit_cn_rows8)
        .collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, CN_ROWS8_MODULE)?;
    Ok((emitted, text, proof))
}

/// Fixed-eight-row unions over every logical width 1..64. The 96-byte ABI is unchanged:
/// U=80, C=total logical rows, grid `[ceil(M/8),80,ceil(C/8)]`, block256.
/// Only descriptor pointers advance by `1928*blockIdx.z`; slot ids remain absolute, so
/// X/Y bases and their original slot/divisor addressing never receive a second tile offset.
/// gfx11 uses its existing singleton ISA arm and cooperative `s_barrier`, not a retagged image.
/// Availability here does not change product codec admission or certify hardware identity.
pub fn emit_rowtiles(spec: Spec) -> Result<Emitted, String> {
    emit_with(spec, true, true)
}

fn s_sub_i32(a: Arch) -> &'static str {
    if a.gfx12() {
        "s_sub_co_i32"
    } else {
        "s_sub_i32"
    }
}

/// Kernarg loads, the guards, the expert's pointer and the row/slot bookkeeping, per-lane
/// address constants, zeroed accumulators and the per-slot X offsets / Y addresses.
/// The LDS store address `4 * tid` is derived first, in the entry block (before the first branch).
fn prologue(b: &mut Builder, spec: Spec, l: &Lay) -> R {
    let a = b.spec.arch;
    let exit = l.label(spec, "exit");
    if l.rows8 {
        // Eight-wave workgroup: tid = 32 * wave + lane. The LDS store address folds to `4 * (tid & 127)` (see
        // `stage_rows8`); the lane index and the uniform wave index come from the work-item id (dead `x_address` /
        // `x_base` registers until the first group).
        valu(
            b,
            &format!("v_and_b32_e32 v{}, {}, v{LANE}", l.lds, lit(127)),
            &[l.lds],
            &[LANE],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{0}, 2, v{0}", l.lds),
            &[l.lds],
            &[l.lds],
            &[],
        )?;
        valu(
            b,
            &format!("v_and_b32_e32 v{}, 31, v{LANE}", l.lane),
            &[l.lane],
            &[LANE],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshrrev_b32_e32 v{}, 5, v{LANE}", l.xb),
            &[l.xb],
            &[LANE],
            &[],
        )?;
        op(
            b,
            format!("v_readfirstlane_b32 s{WAVE}, v{}", l.xb),
            &[s(WAVE)],
            &[v(l.xb)],
        )?;
    } else {
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{}, 2, v{LANE}", l.lds),
            &[l.lds],
            &[LANE],
            &[],
        )?;
    }
    smem(b, l.wptrs(), 8, KARG, 0)?;
    smem(b, XROT, 8, KARG, 0x20)?;
    smem(b, SU, 4, KARG, 0x40)?;
    smem(b, SXSTR, 2, KARG, 0x50)?;
    let snexp = if l.rowtiles { TILE_NEXP } else { SNEXP };
    smem(b, snexp, 1, KARG, 0x58)?;
    if a.gfx12() {
        if l.rows8 {
            sop(b, format!("s_lshl_b32 s{ROW}, ttmp9, 3"), &[ROW], &[])?;
            sop(
                b,
                format!("{} s{ROW}, s{ROW}, s{WAVE}", s_add_i32(a)),
                &[ROW],
                &[ROW, WAVE],
            )?;
        } else {
            sop(b, format!("s_mov_b32 s{ROW}, ttmp9"), &[ROW], &[])?;
        }
        sop(b, format!("s_and_b32 s{UIDX}, ttmp7, 0xffff"), &[UIDX], &[])?;
    } else {
        // gfx11 workgroup ids are system SGPRs (after the user SGPRs), not ttmp registers.
        let ids = b.spec.workgroup_ids(l.rowtiles)?;
        if (ids.x, ids.y) != (WgIdSource::Sgpr(l.wg_x()), WgIdSource::Sgpr(l.wg_y())) {
            return Err(format!("unexpected workgroup id ABI {ids:?}"));
        }
        if l.rows8 {
            sop(
                b,
                format!("s_lshl_b32 s{ROW}, s{}, 3", l.wg_x()),
                &[ROW],
                &[l.wg_x()],
            )?;
            sop(
                b,
                format!("{} s{ROW}, s{ROW}, s{WAVE}", s_add_i32(a)),
                &[ROW],
                &[ROW, WAVE],
            )?;
        } else {
            sop(
                b,
                format!("s_mov_b32 s{ROW}, s{}", l.wg_x()),
                &[ROW],
                &[l.wg_x()],
            )?;
        }
        sop(
            b,
            format!("s_mov_b32 s{UIDX}, s{}", l.wg_y()),
            &[UIDX],
            &[l.wg_y()],
        )?;
    }
    if l.rowtiles {
        if a.gfx12() {
            sop(b, format!("s_lshr_b32 s{TILE}, ttmp7, 16"), &[TILE], &[])?;
        } else {
            sop(
                b,
                format!("s_mov_b32 s{TILE}, s{}", l.wg_z()),
                &[TILE],
                &[l.wg_z()],
            )?;
        }
        // A dead tile/domain exits before any descriptor or table access, uniformly.
        sop(b, format!("s_cmp_eq_u32 s{SC}, 0"), &[], &[SC])?;
        branch(b, &format!("s_cbranch_scc1 {exit}"))?;
        sop(b, format!("s_cmp_gt_u32 s{SC}, 64"), &[], &[SC])?;
        branch(b, &format!("s_cbranch_scc1 {exit}"))?;
        sop(
            b,
            format!("s_cmp_eq_u32 s{SU}, {}", lit(TILE_SLOTS)),
            &[],
            &[SU],
        )?;
        branch(b, &format!("s_cbranch_scc0 {exit}"))?;
        sop(b, format!("s_lshl_b32 s{TMP}, s{TILE}, 3"), &[TMP], &[TILE])?;
        sop(b, format!("s_cmp_ge_u32 s{TMP}, s{SC}"), &[], &[TMP, SC])?;
        branch(b, &format!("s_cbranch_scc1 {exit}"))?;
        sop(b, format!("s_and_b32 s{TMP}, s{ROW}, -8"), &[TMP], &[ROW])?;
        sop(b, format!("s_cmp_ge_u32 s{TMP}, s{SM}"), &[], &[TMP, SM])?;
        branch(b, &format!("s_cbranch_scc1 {exit}"))?;
        sop(
            b,
            format!("s_mul_i32 s{TMP}, s{TILE}, {}", lit(TILE_BYTES)),
            &[TMP],
            &[TILE],
        )?;
        for ptr in [l.eids(), l.offs(), l.slots()] {
            add64(b, ptr, ptr, TMP)?;
        }
    }
    if !l.rows8 {
        sop(b, format!("s_cmp_ge_u32 s{ROW}, s{SM}"), &[], &[ROW, SM])?;
        branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    }
    sop(b, format!("s_cmp_ge_u32 s{UIDX}, s{SU}"), &[], &[UIDX, SU])?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    // offsets[u], offsets[u+1] and the expert id; the expert's tensor pointer from the table.
    sop(b, format!("s_lshl_b32 s{TMP}, s{UIDX}, 2"), &[TMP], &[UIDX])?;
    add64(b, TPTR, l.offs(), TMP)?;
    smem(b, OFF0, 2, TPTR, 0)?;
    add64(b, TPTR, l.eids(), TMP)?;
    smem(b, EID, 1, TPTR, 0)?;
    sop(
        b,
        format!("{} s{NS}, s{OFF1}, s{OFF0}", s_sub_i32(a)),
        &[NS],
        &[OFF1, OFF0],
    )?;
    sop(b, format!("s_cmp_eq_u32 s{NS}, 0"), &[], &[NS])?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    sop(b, format!("s_cmp_gt_u32 s{NS}, {MAX_SLOTS}"), &[], &[NS])?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    if l.rowtiles {
        sop(
            b,
            format!("s_mov_b32 s{TMP}, {}", lit(TILE_SLOTS)),
            &[TMP],
            &[],
        )?;
    } else {
        sop(b, format!("s_mul_i32 s{TMP}, s{SC}, 10"), &[TMP], &[SC])?;
    }
    sop(
        b,
        format!("s_cmp_gt_u32 s{OFF1}, s{TMP}"),
        &[],
        &[OFF1, TMP],
    )?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    // Expert id range check before the table index (and before the LDS stage and its barrier): an id at or past
    // `n_experts` (unsigned, so a negative id too) exits without a table load or a store.
    sop(
        b,
        format!("s_cmp_ge_u32 s{EID}, s{snexp}"),
        &[],
        &[EID, snexp],
    )?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    sop(b, format!("s_lshl_b32 s{TMP}, s{EID}, 3"), &[TMP], &[EID])?;
    add64(b, TPTR, l.wptrs(), TMP)?;
    smem(b, WPTR, 2, TPTR, 0)?;
    // Group count, row bytes (K/4) and the row's word and scale pointers (scales follow the M*K/4 word bytes).
    sop(
        b,
        format!("s_lshr_b32 s{GROUPS}, s{SK}, 7"),
        &[GROUPS],
        &[SK],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{ROW_BYTES}, s{SK}, 2"),
        &[ROW_BYTES],
        &[SK],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{CODES_BYTES}, s{SM}, s{ROW_BYTES}"),
        &[CODES_BYTES],
        &[SM, ROW_BYTES],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{TMP}, s{ROW}, s{ROW_BYTES}"),
        &[TMP],
        &[ROW, ROW_BYTES],
    )?;
    add64(b, ROW_PTR, WPTR, TMP)?;
    sop(
        b,
        format!("s_mul_i32 s{TMP2}, s{ROW}, s{GROUPS}"),
        &[TMP2],
        &[ROW, GROUPS],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{TMP2}, s{TMP2}, 1"),
        &[TMP2],
        &[TMP2],
    )?;
    sop(
        b,
        format!("{} s{TMP2}, s{TMP2}, s{CODES_BYTES}", s_add_i32(a)),
        &[TMP2],
        &[TMP2, CODES_BYTES],
    )?;
    add64(b, SCALE_PTR, WPTR, TMP2)?;
    // Y pointer and the row's byte offset inside a slot's output row.
    match spec.kind {
        Kind::GateUp => {
            sop(b, format!("s_lshr_b32 s{MI}, s{SM}, 1"), &[MI], &[SM])?;
            sop(
                b,
                format!("{} s{ROWM}, s{ROW}, s{MI}", s_sub_i32(a)),
                &[ROWM],
                &[ROW, MI],
            )?;
            sop(b, format!("s_cmp_lt_u32 s{ROW}, s{MI}"), &[], &[ROW, MI])?;
            op(
                b,
                format!(
                    "s_cselect_b64 s[{YSEL}:{}], s[{Y0}:{}], s[{Y1}:{}]",
                    YSEL + 1,
                    Y0 + 1,
                    Y1 + 1
                ),
                &[sr(YSEL, 2)],
                &[sr(Y0, 2), sr(Y1, 2)],
            )?;
            sop(
                b,
                format!("s_cselect_b32 s{ROWIN}, s{ROW}, s{ROWM}"),
                &[ROWIN],
                &[ROW, ROWM],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{ROWIN4}, s{ROWIN}, 2"),
                &[ROWIN4],
                &[ROWIN],
            )?;
        }
        Kind::Down => {
            sop(b, format!("s_mov_b32 s{YSEL}, s{Y0}"), &[YSEL], &[Y0])?;
            sop(
                b,
                format!("s_mov_b32 s{}, s{}", YSEL + 1, Y0 + 1),
                &[YSEL + 1],
                &[Y0 + 1],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{ROWIN4}, s{ROW}, 2"),
                &[ROWIN4],
                &[ROW],
            )?;
        }
    }
    sop(
        b,
        format!("s_lshl_b32 s{SXS4}, s{SXSTR}, 2"),
        &[SXS4],
        &[SXSTR],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{SYS4}, s{SYSTR}, 2"),
        &[SYS4],
        &[SYSTR],
    )?;
    sop(
        b,
        format!("{} s{NSM1}, s{NS}, -1", s_add_i32(a)),
        &[NSM1],
        &[NS],
    )?;
    // X descriptor: base Xrot, x_src_rows*x_row_stride*4 records (rows past it read 0).
    sop(b, format!("s_mov_b32 s{SRDX}, s{XROT}"), &[SRDX], &[XROT])?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRDX + 1, XROT + 1),
        &[SRDX + 1],
        &[XROT + 1],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{TMP}, s{SXSRC}, s{SXS4}"),
        &[TMP],
        &[SXSRC, SXS4],
    )?;
    srd_tail(b, SRDX, Some(TMP))?;
    sop(b, format!("s_mov_b32 s{GI}, 0"), &[GI], &[])?;
    // Per-lane constants: x byte offset within a group (16*l), the lane's word byte offset within a group
    // (2*(l>>1) = l & 30) and the half shift 16*(l&1).
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{}, 4, v{}", l.xl, l.lane),
        &[l.xl],
        &[l.lane],
        &[],
    )?;
    valu(
        b,
        &format!("v_and_b32_e32 v{}, 30, v{}", l.cl, l.lane),
        &[l.cl],
        &[l.lane],
        &[],
    )?;
    valu(
        b,
        &format!("v_and_b32_e32 v{}, 1, v{}", l.hsh, l.lane),
        &[l.hsh],
        &[l.lane],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{0}, 4, v{0}", l.hsh),
        &[l.hsh],
        &[l.hsh],
        &[],
    )?;
    for k in 0..MAX_SLOTS as u8 * l.chains {
        valu(
            b,
            &format!("v_mov_b32_e32 v{}, 0", l.acc + k),
            &[l.acc + k],
            &[],
            &[],
        )?;
    }
    // Slot lanes: lane i < 8 loads slot position min(i, n-1) (never past the expert's slots),
    // derives the X-row byte offset (slot / x_row_div * x_row_stride * 4) and the Y byte
    // offset (slot * y_slot_stride * 4 + row-in-Y * 4); each is then broadcast into per-slot VGPRs.
    let (sid, xq, t0, yq, p) = (l.x, l.x + 1, l.x + 2, l.x + 6, l.x + 7);
    valu(
        b,
        &format!("v_min_u32_e32 v{p}, s{NSM1}, v{}", l.lane),
        &[p],
        &[l.lane],
        &[NSM1],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{p}, s{OFF0}, v{p}"),
        &[p],
        &[p],
        &[OFF0],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{p}, 2, v{p}"),
        &[p],
        &[p],
        &[],
    )?;
    gload(b, "global_load_b32", sid, p, l.slots(), "")?;
    if l.rowtiles {
        // Each wave reads the same eight slot lanes (inactive lanes duplicate the final
        // live slot). A bad absolute slot therefore exits the entire workgroup before
        // the table barrier, without deriving an out-of-window X/Y address.
        sop(b, format!("s_mul_i32 s{TMP2}, s{SC}, 10"), &[TMP2], &[SC])?;
        for i in 0..MAX_SLOTS {
            op(
                b,
                format!("v_readlane_b32 s{TMP}, v{sid}, {i}"),
                &[s(TMP)],
                &[v(sid)],
            )?;
            sop(
                b,
                format!("s_cmp_ge_u32 s{TMP}, s{TMP2}"),
                &[],
                &[TMP, TMP2],
            )?;
            branch(b, &format!("s_cbranch_scc1 {exit}"))?;
        }
    }
    gather_offset(
        b,
        xq,
        sid,
        Some(SDIV),
        1,
        LIVE,
        GatherTemps {
            v: [t0, t0 + 1, t0 + 2, t0 + 3],
            mask: MASK,
        },
    )?;
    valu(
        b,
        &format!("v_mul_lo_u32 v{xq}, v{xq}, s{SXS4}"),
        &[xq],
        &[xq],
        &[SXS4],
    )?;
    valu(
        b,
        &format!("v_mul_lo_u32 v{yq}, v{sid}, s{SYS4}"),
        &[yq],
        &[sid],
        &[SYS4],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{yq}, s{ROWIN4}, v{yq}"),
        &[yq],
        &[yq],
        &[ROWIN4],
    )?;
    for i in 0..MAX_SLOTS as u8 {
        let tl = TL + i % 4;
        op(
            b,
            format!("v_readlane_b32 s{tl}, v{xq}, {i}"),
            &[s(tl)],
            &[v(xq)],
        )?;
        valu(
            b,
            &format!("v_mov_b32_e32 v{}, s{tl}", l.xo + i),
            &[l.xo + i],
            &[],
            &[tl],
        )?;
        op(
            b,
            format!("v_readlane_b32 s{tl}, v{yq}, {i}"),
            &[s(tl)],
            &[v(yq)],
        )?;
        valu(
            b,
            &format!("v_mov_b32_e32 v{}, s{tl}", l.ya + i),
            &[l.ya + i],
            &[],
            &[tl],
        )?;
    }
    Ok(())
}

/// Model of the arithmetic MASK entry the stage emits for table index `src` (`< 128`):
/// `x = src | parity(src) << 7`, bit `k` spread to nibble `k`, `0xF` per set bit. Equals
/// `build_mask128u32()[src]` (lane 7 = parity of `popcount(sign7)`).
pub fn stage_mask_model(src: u32) -> u32 {
    let mut pc = src ^ (src >> 4);
    for sh in [2u32, 1] {
        pc ^= pc >> sh;
    }
    let pc = pc << 7;
    let mut x = (pc & 0x80) | (src & !0x80);
    for (shift, mask) in [(12u32, 0x000f_000fu32), (6, 0x0303_0303), (3, ONES)] {
        x = (x << shift | x) & mask;
    }
    (x << 4).wrapping_sub(x)
}

/// Stage `LUT` and `MASK` into the checked LDS slot: the wave stores, drains, and one workgroup
/// barrier publishes. Scratch is the dead X-value block (`e, pc, x`, four MASK data registers, eight
/// LUT chunk registers, all distinct so no store source is redefined).
fn stage(b: &mut Builder, l: &Lay) -> R {
    let lut = build_lut256u32();
    let (e, pc, x) = (l.x, l.x + 1, l.x + 2);
    let ys = l.x + 3;
    let chunks = l.x + 7;
    for j in 0..4u32 {
        let y = ys + j as u8;
        let src = if j == 0 {
            LANE
        } else {
            valu(
                b,
                &format!("v_add_nc_u32_e32 v{e}, {}, v{LANE}", lit(32 * j)),
                &[e],
                &[LANE],
                &[],
            )?;
            e
        };
        mask_word(b, src, pc, x, y)?;
        b.ds_store(0, ds_store_b32(l.lds, y, LUT_BYTES + 128 * j))?;
    }
    // LUT: eight chunks of 32 dwords, one chunk register each, lane q of chunk c = LUT[32c + q].
    for c in 0..8u32 {
        let vc = chunks + c as u8;
        lut_chunk(b, &lut, c, vc)?;
        b.ds_store(0, ds_store_b32(l.lds, vc, 128 * c))?;
    }
    b.barrier(&[Transition::Ready(0)])
}

/// LUT chunk `c` (32 dwords, lane `q` = `LUT[32c + q]`) written lane by lane into `vc` from SGPR literals.
fn lut_chunk(b: &mut Builder, lut: &[u32; 256], c: u32, vc: u8) -> R {
    for batch in 0..4u32 {
        for q in 0..8u32 {
            let r = LITS + q as u8;
            sop(
                b,
                format!(
                    "s_mov_b32 s{r}, {}",
                    lit(lut[(32 * c + 8 * batch + q) as usize])
                ),
                &[r],
                &[],
            )?;
        }
        for q in 0..8u32 {
            let r = LITS + q as u8;
            op(
                b,
                format!("v_writelane_b32 v{vc}, s{r}, {}", 8 * batch + q),
                &[v(vc)],
                &[v(vc), s(r)],
            )?;
        }
    }
    Ok(())
}

/// The MASK dword `y` (`0xF` per set bit of `x = sign7 | parity(sign7) << 7`, bit `k` spread to nibble `k`, then
/// `(x << 4) - x`) of table index `src` (`< 128`), through the scratch `pc`, `x`; [`stage_mask_model`] is its model.
fn mask_word(b: &mut Builder, src: u8, pc: u8, x: u8, y: u8) -> R {
    // x = sign7 | parity(sign7) << 7; parity by xor-shift reduction (sign7 < 128).
    valu(
        b,
        &format!("v_lshrrev_b32_e32 v{x}, 4, v{src}"),
        &[x],
        &[src],
        &[],
    )?;
    valu(
        b,
        &format!("v_xor_b32_e32 v{pc}, v{src}, v{x}"),
        &[pc],
        &[src, x],
        &[],
    )?;
    for sh in [2u8, 1] {
        valu(
            b,
            &format!("v_lshrrev_b32_e32 v{x}, {sh}, v{pc}"),
            &[x],
            &[pc],
            &[],
        )?;
        valu(
            b,
            &format!("v_xor_b32_e32 v{pc}, v{pc}, v{x}"),
            &[pc],
            &[pc, x],
            &[],
        )?;
    }
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{pc}, 7, v{pc}"),
        &[pc],
        &[pc],
        &[],
    )?;
    valu(
        b,
        &format!("v_bfi_b32 v{x}, 0x80, v{pc}, v{src}"),
        &[x],
        &[pc, src],
        &[],
    )?;
    // Spread bit k to nibble k.
    for (shift, mask) in [(12u32, 0x000f_000fu32), (6, 0x0303_0303), (3, ONES)] {
        valu(
            b,
            &format!("v_lshl_or_b32 v{x}, v{x}, {shift}, v{x}"),
            &[x],
            &[x],
            &[],
        )?;
        valu(
            b,
            &format!("v_and_b32_e32 v{x}, {}, v{x}", lit(mask)),
            &[x],
            &[x],
            &[],
        )?;
    }
    // 0xF per set bit: (x << 4) - x.
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{y}, 4, v{x}"),
        &[y],
        &[x],
        &[],
    )?;
    valu(
        b,
        &format!("v_sub_nc_u32_e32 v{y}, v{y}, v{x}"),
        &[y],
        &[y, x],
        &[],
    )
}

/// The eight-wave entry's cooperative table stage. The workgroup's eight waves publish ONE copy of the 1,536-byte
/// tables: wave `w` writes LUT chunk `w` (its 32 lanes = `LUT[32w ..]`, from SGPR literals selected by the uniform
/// wave index) and stores it at `4 * (tid & 127) + 512 * (w >> 2)` (the entry-block LDS address `l.lds`; waves 0-3
/// at offset 0, waves 4-7 at offset 512); waves 0-3 also compute the MASK dword of table index `tid` and store it at
/// `l.lds + 1024`; waves 4-7 branch over the MASK work. Every address is a function of the work-item id derived in the
/// entry block and every access ends `<= 1536`. Each path drains its stores and all eight waves meet at the one
/// barrier that publishes the checked slot. Scratch is the dead X-value block (`pc, x, y, chunk`).
fn stage_rows8(b: &mut Builder, spec: Spec, l: &Lay) -> R {
    let lut = build_lut256u32();
    let (pc, x, y) = (l.x + 1, l.x + 2, l.x + 3);
    let vc = l.x + 4;
    for c in 0..8u32 {
        let skip = l.label(spec, &format!("lut{c}_skip"));
        sop(b, format!("s_cmp_eq_u32 s{WAVE}, {c}"), &[], &[WAVE])?;
        branch(b, &format!("s_cbranch_scc0 {skip}"))?;
        lut_chunk(b, &lut, c, vc)?;
        b.label(&skip)?;
    }
    let (hi, join) = (l.label(spec, "stage_hi"), l.label(spec, "stage_join"));
    sop(b, format!("s_cmp_gt_u32 s{WAVE}, 3"), &[], &[WAVE])?;
    branch(b, &format!("s_cbranch_scc1 {hi}"))?;
    mask_word(b, LANE, pc, x, y)?;
    b.ds_store(0, ds_store_b32(l.lds, y, LUT_BYTES))?;
    b.ds_store(0, ds_store_b32(l.lds, vc, 0))?;
    b.wait_all()?;
    branch(b, &format!("s_branch {join}"))?;
    b.label(&hi)?;
    b.ds_store(0, ds_store_b32(l.lds, vc, 512))?;
    b.wait_all()?;
    b.label(&join)?;
    b.barrier(&[Transition::Ready(0)])
}

/// `if n <= i skip to target` (scalar, uniform): SCC = (n > i), branch when clear.
fn slot_gate(b: &mut Builder, i: u32, target: &str) -> R {
    sop(b, format!("s_cmp_gt_u32 s{NS}, {i}"), &[], &[NS])?;
    branch(b, &format!("s_cbranch_scc0 {target}"))
}

/// One group of the row, loaded and decoded once, then applied to every live slot.
/// Chain `chain` (always 0 for down) takes the group partial of every slot.
fn group(b: &mut Builder, spec: Spec, l: &Lay, chain: u8, inst: &str) -> R {
    let arch = b.spec.arch;
    // Scalar group offsets (32 code bytes, 128 X floats, 2 scale bytes per group), then the lane
    // addresses, then the weight loads (all before any slot branch: slot branches never re-read weights).
    sop(
        b,
        format!("s_lshl_b32 s{G_CODE}, s{GI}, 5"),
        &[G_CODE],
        &[GI],
    )?;
    sop(b, format!("s_lshl_b32 s{G_X}, s{GI}, 9"), &[G_X], &[GI])?;
    sop(
        b,
        format!("s_lshl_b32 s{G_SCALE}, s{GI}, 1"),
        &[G_SCALE],
        &[GI],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{}, s{G_CODE}, v{}", l.caddr, l.cl),
        &[l.caddr],
        &[l.cl],
        &[G_CODE],
    )?;
    valu(
        b,
        &format!("v_mov_b32_e32 v{}, s{G_SCALE}", l.saddr),
        &[l.saddr],
        &[],
        &[G_SCALE],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{}, s{G_X}, v{}", l.xb, l.xl),
        &[l.xb],
        &[l.xl],
        &[G_X],
    )?;
    // Exact u16 word and u16 scale: the weight-class loads of the group.
    gload(b, "global_load_u16", l.wrd, l.caddr, ROW_PTR, "")?;
    gload(b, "global_load_u16", l.d, l.saddr, SCALE_PTR, "")?;
    // X loads of the live slots (slot 0 always: n >= 1), each slot's own X row.
    let xdone = l.label(spec, &format!("{inst}_xdone"));
    for i in 0..MAX_SLOTS {
        if i > 0 {
            slot_gate(b, i, &xdone)?
        }
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{}, v{}, v{}", l.xa, l.xo + i as u8, l.xb),
            &[l.xa],
            &[l.xo + i as u8, l.xb],
            &[],
        )?;
        bload(b, 4, l.xr(i), l.xa, SRDX, 0)?;
    }
    b.label(&xdone)?;
    b.wait_all()?;
    unpack(b, l)?;
    // Per-slot chains on the shared weights: p = RN(w0*x0), then fma(w_j, x_j, p) for j = 1..3.
    let done = l.label(spec, &format!("{inst}_done"));
    for i in 0..MAX_SLOTS {
        if i > 0 {
            slot_gate(b, i, &done)?
        }
        let dot = l.dot + (i % 2) as u8;
        let x = l.xr(i);
        valu(
            b,
            &format!("v_mul_f32_e32 v{dot}, v{}, v{x}", l.w),
            &[dot],
            &[l.w, x],
            &[],
        )?;
        for j in 1..LANE_ELEMS {
            valu(
                b,
                &format!("v_fmac_f32_e32 v{dot}, v{}, v{}", l.w + j, x + j),
                &[dot],
                &[dot, l.w + j, x + j],
                &[],
            )?;
        }
        let acc = l.ac(i, chain);
        valu(
            b,
            &format!("v_add_f32_e32 v{acc}, v{acc}, v{dot}"),
            &[acc],
            &[acc, dot],
            &[],
        )?;
    }
    b.label(&done)?;
    sop(
        b,
        format!("{} s{GI}, s{GI}, 1", s_add_i32(arch)),
        &[GI],
        &[GI],
    )?;
    b.wait_all()
}

/// Decode the lane's word to its four F32 weights `l.w ..`: masked table addresses and the two
/// table loads first, the shift addend and the scale `f32(d)*0.25` independent of them, then
/// `x = lut ^ mask`, `y = mask & 0x11111111`, `operand = x + y + n`, the lane's half, and per element
/// `v = 2*bfe_i32(t, 4i, 4) + 1`, `w = RN(dq * float(v))`.
fn unpack(b: &mut Builder, l: &Lay) -> R {
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{}, 2, v{}", l.la, l.wrd),
        &[l.la],
        &[l.wrd],
        &[],
    )?;
    valu(
        b,
        &format!("v_and_b32_e32 v{0}, {1}, v{0}", l.la, lit(0x3fc)),
        &[l.la],
        &[l.la],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshrrev_b32_e32 v{}, 6, v{}", l.ma, l.wrd),
        &[l.ma],
        &[l.wrd],
        &[],
    )?;
    valu(
        b,
        &format!("v_and_b32_e32 v{0}, {1}, v{0}", l.ma, lit(0x1fc)),
        &[l.ma],
        &[l.ma],
        &[],
    )?;
    b.ds_load(0, ds_load_b32(l.lv, l.la, 0))?;
    b.ds_load(0, ds_load_b32(l.mv, l.ma, LUT_BYTES))?;
    // Shift addend `-ones` from the shift bit (bit 15 of the zero-extended word).
    valu(
        b,
        &format!("v_bfe_i32 v{}, v{}, 15, 1", l.sh, l.wrd),
        &[l.sh],
        &[l.wrd],
        &[],
    )?;
    valu(
        b,
        &format!("v_and_b32_e32 v{0}, {1}, v{0}", l.sh, lit(NEG_ONES)),
        &[l.sh],
        &[l.sh],
        &[],
    )?;
    // dq = f32(d) * 0.25, exact for finite d >= 0x0400.
    valu(
        b,
        &format!("v_cvt_f32_f16_e64 v{0}, v{0}.l", l.d),
        &[l.d],
        &[l.d],
        &[],
    )?;
    valu(
        b,
        &format!("v_mul_f32_e32 v{0}, {1}, v{0}", l.d, lit(QUARTER)),
        &[l.d],
        &[l.d],
        &[],
    )?;
    valu(
        b,
        &format!("v_xor_b32_e32 v{0}, v{1}, v{0}", l.lv, l.mv),
        &[l.lv],
        &[l.mv, l.lv],
        &[],
    )?;
    valu(
        b,
        &format!("v_and_b32_e32 v{0}, {1}, v{0}", l.mv, lit(ONES)),
        &[l.mv],
        &[l.mv],
        &[],
    )?;
    valu(
        b,
        &format!("v_add3_u32 v{}, v{}, v{}, v{}", l.opr, l.lv, l.mv, l.sh),
        &[l.opr],
        &[l.lv, l.mv, l.sh],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshrrev_b32_e32 v{0}, v{1}, v{0}", l.opr, l.hsh),
        &[l.opr],
        &[l.hsh, l.opr],
        &[],
    )?;
    for i in 0..LANE_ELEMS {
        let w = l.w + i;
        valu(
            b,
            &format!("v_bfe_i32 v{w}, v{}, {}, 4", l.opr, u32::from(i) * 4),
            &[w],
            &[l.opr],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{w}, 1, v{w}"),
            &[w],
            &[w],
            &[],
        )?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{w}, 1, v{w}"),
            &[w],
            &[w],
            &[],
        )?;
        valu(b, &format!("v_cvt_f32_i32_e32 v{w}, v{w}"), &[w], &[w], &[])?;
        valu(
            b,
            &format!("v_mul_f32_e32 v{w}, v{}, v{w}", l.d),
            &[w],
            &[l.d, w],
            &[],
        )?;
    }
    Ok(())
}

/// Shuffle tables, the per-slot fold + snapshot shuffle-down tree, then the lane-0 stores.
fn epilogue(b: &mut Builder, spec: Spec, l: &Lay) -> R {
    // The lane index: the work-item id itself in the one-wave entry; `tid & 31` (into the dead `x_address`) when
    // the workgroup is eight waves.
    let lane = l.lane;
    if l.rows8 {
        valu(
            b,
            &format!("v_and_b32_e32 v{lane}, 31, v{LANE}"),
            &[lane],
            &[LANE],
            &[],
        )?;
    }
    // Shuffle step k (offset 16 >> k): mask lane < 32 - off, address (lane + off or lane) * 4.
    let offsets = [16u32, 8, 4, 2, 1];
    for (k, off) in offsets.into_iter().enumerate() {
        let (m, sh) = (SHM + k as u8, l.shuf + k as u8);
        op(
            b,
            format!("v_cmp_gt_u32_e64 s{m}, {}, v{lane}", 32 - off),
            &[s(m)],
            &[v(lane)],
        )?;
        valu(
            b,
            &format!("v_cndmask_b32_e64 v{sh}, 0, {off}, s{m}"),
            &[sh],
            &[],
            &[m],
        )?;
        valu(
            b,
            &format!("v_add_lshl_u32 v{sh}, v{sh}, v{lane}, 2"),
            &[sh],
            &[sh, lane],
            &[],
        )?;
    }
    let red_done = l.label(spec, "red_done");
    for i in 0..MAX_SLOTS {
        if i > 0 {
            slot_gate(b, i, &red_done)?
        }
        let a0 = l.ac(i, 0);
        if spec.kind == Kind::GateUp {
            let (a1, a2, a3) = (l.ac(i, 1), l.ac(i, 2), l.ac(i, 3));
            valu(
                b,
                &format!("v_add_f32_e32 v{a0}, v{a0}, v{a1}"),
                &[a0],
                &[a0, a1],
                &[],
            )?;
            valu(
                b,
                &format!("v_add_f32_e32 v{a2}, v{a2}, v{a3}"),
                &[a2],
                &[a2, a3],
                &[],
            )?;
            valu(
                b,
                &format!("v_add_f32_e32 v{a0}, v{a0}, v{a2}"),
                &[a0],
                &[a0, a2],
                &[],
            )?;
        }
        for k in 0..5u8 {
            let (m, sh) = (SHM + k, l.shuf + k);
            b.ds_crosslane(
                Instruction::new(
                    format!("ds_bpermute_b32 v{}, v{sh}, v{a0}", l.exch),
                    vec![vreg(l.exch, 1)],
                    vec![vreg(sh, 1), vreg(a0, 1)],
                )
                .memory(MemoryClass::DsLoad),
            )?;
            // sum = acc + acc[lane + off]; lanes with lane + off > 31 (mask clear) keep acc unchanged.
            valu(
                b,
                &format!("v_add_f32_e32 v{}, v{a0}, v{}", l.exch, l.exch),
                &[l.exch],
                &[a0, l.exch],
                &[],
            )?;
            valu(
                b,
                &format!("v_cndmask_b32_e64 v{a0}, v{a0}, v{}, s{m}", l.exch),
                &[a0],
                &[a0, l.exch],
                &[m],
            )?;
        }
    }
    b.label(&red_done)?;
    // Lane 0 stores each live slot's value at its original slot's row.
    valu(
        b,
        &format!("v_cmpx_eq_u32_e32 0, v{lane}"),
        &[],
        &[lane],
        &[],
    )?;
    let st_done = l.label(spec, "st_done");
    for i in 0..MAX_SLOTS {
        if i > 0 {
            slot_gate(b, i, &st_done)?
        }
        let (ya, a0) = (l.ya + i as u8, l.ac(i, 0));
        b.push(
            Instruction::new(
                format!("global_store_b32 v{ya}, v{a0}, s[{YSEL}:{}]", YSEL + 1),
                vec![],
                vec![vreg(ya, 1), vreg(a0, 1), sreg(YSEL, 2)],
            )
            .memory(MemoryClass::VmemStore),
        )?;
    }
    b.label(&st_done)?;
    b.wait_all()?;
    b.label(&l.label(spec, "exit"))?;
    branch(b, "s_endpgm")
}

// ----------------------------------------------------------------------- models and bounds (tests)

/// The decode network on one wire word for one lane half, exactly as emitted: masked table addresses,
/// `x = lut ^ mask`, `y = mask & ones`, `n = (word bit 15) ? -ones : 0`, `x + y + n`, shift by
/// `16*half`, per element `2*sext4(t >> 4i) + 1`. Equals `decode8(idx, sign7, shift)[4*half..]`.
pub fn unpack_model(lut: &[u32; 256], mask: &[u32; 128], word: u16, half: u32) -> [i32; 4] {
    let w = u32::from(word);
    let (la, ma) = ((w << 2) & 0x3fc, (w >> 6) & 0x1fc);
    let (lv, mv) = (lut[(la / 4) as usize], mask[(ma / 4) as usize]);
    let sh = (((w >> 15) & 1) as i32).wrapping_neg() as u32 & NEG_ONES;
    let t = (lv ^ mv).wrapping_add(mv & ONES).wrapping_add(sh) >> (16 * half);
    std::array::from_fn(|i| {
        2 * (((t >> (4 * i)) & 0xf) as i32 - if (t >> (4 * i)) & 0x8 != 0 { 16 } else { 0 }) + 1
    })
}

/// The four weights of one lane: `RN((f32(d)*0.25) * f32(v))` with `f32(d)` from FP16 bits `d_bits` (finite).
pub fn weights_model(
    lut: &[u32; 256],
    mask: &[u32; 128],
    word: u16,
    d_bits: u16,
    half: u32,
) -> [f32; 4] {
    let dq = f16_bits_to_f32(d_bits) * f32::from_bits(QUARTER);
    unpack_model(lut, mask, word, half).map(|v| dq * v as f32)
}

/// Exact FP16 -> F32 of finite bits (no `half` dependency in the builder crate).
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let (sign, exp, frac) = (
        u32::from(bits >> 15),
        u32::from((bits >> 10) & 0x1f),
        u32::from(bits & 0x3ff),
    );
    let magnitude = match exp {
        0 => frac as f32 * f32::from_bits(0x3380_0000),
        0x1f => f32::NAN,
        _ => f32::from_bits(((exp + 112) << 23) | (frac << 13)),
    };
    if sign == 1 {
        -magnitude
    } else {
        magnitude
    }
}

/// The lane/word/half access of weight word `(lane, g)` of a row: group byte `32*g + 2*(lane >> 1)`.
pub fn word_byte(lane: u32, g: u32) -> u32 {
    32 * g + 2 * (lane >> 1)
}
/// One past the last byte of the code plane a row `r` of a `[M, K]` tensor reads (`word_end` of the last
/// lane/group): inside `[r*K/4, (r+1)*K/4)`.
pub fn word_end(k: u32, r: u32) -> u32 {
    r * (k / 4) + word_byte(31, k / 128 - 1) + 2
}
/// One past the last byte the scale loads of row `r` read, inside the scale plane `[M*K/4, M*K/4 + M*K/64)`.
pub fn scale_end(m: u32, k: u32, r: u32) -> u32 {
    m * (k / 4) + 2 * (r * (k / 128) + k / 128 - 1) + 2
}

fn vreg(base: u8, len: u8) -> RegRef {
    RegRef {
        kind: RKind::V,
        base,
        len,
    }
}
fn sreg(base: u8, len: u8) -> RegRef {
    RegRef {
        kind: RKind::S,
        base,
        len,
    }
}
fn refs(kind: RKind, indices: &[u8]) -> Vec<RegRef> {
    indices
        .iter()
        .map(|&base| RegRef { kind, base, len: 1 })
        .collect()
}
fn valu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8]) -> R {
    let mut inputs = refs(RKind::V, uses);
    inputs.extend(refs(RKind::S, scalar));
    b.push(Instruction::new(text, refs(RKind::V, defs), inputs))
}
/// `mnemonic v{dst}, v{voff}, s[{base}:{base+1}]{suffix}` of one dword (or sub-dword) load.
fn gload(b: &mut Builder, mnemonic: &str, dst: u8, voff: u8, base: u8, suffix: &str) -> R {
    b.push(
        Instruction::new(
            format!("{mnemonic} v{dst}, v{voff}, s[{base}:{}]{suffix}", base + 1),
            vec![vreg(dst, 1)],
            vec![vreg(voff, 1), sreg(base, 2)],
        )
        .memory(MemoryClass::VmemLoad),
    )
}
fn branch(b: &mut Builder, text: &str) -> R {
    b.control(Instruction::new(text, vec![], vec![]))
}
fn ds_store_b32(addr: u8, data: u8, off: u32) -> Instruction {
    let o = if off == 0 {
        String::new()
    } else {
        format!(" offset:{off}")
    };
    Instruction::new(
        format!("ds_store_b32 v{addr}, v{data}{o}"),
        vec![],
        vec![v(addr), v(data)],
    )
    .memory(MemoryClass::DsStore)
}
fn ds_load_b32(dst: u8, addr: u8, off: u32) -> Instruction {
    let o = if off == 0 {
        String::new()
    } else {
        format!(" offset:{off}")
    };
    Instruction::new(
        format!("ds_load_b32 v{dst}, v{addr}{o}"),
        vec![v(dst)],
        vec![v(addr)],
    )
    .memory(MemoryClass::DsLoad)
}

/// The shape contract of one emitted entry: counts of its characteristic instructions, the forbidden
/// families, honest register ceilings of the C8 slot state, wave32, no spills, no private memory, and
/// exactly the 1,536-byte static table stage (no dynamic LDS). Committed as
/// `kernels/e8p_i4_moe_gemv.<arch>.<tag>.contract.json`. The gfx1100 contract counts the gfx11
/// `s_barrier`, `s_waitcnt`, `s_waitcnt_depctr` and `s_delay_alu` the builder inserts, forbids the gfx12
/// spellings (split barriers, `s_wait_*` counters, `s_add_co_*` / `s_sub_co_*`), and does not forbid
/// `s_waitcnt*`, which gfx11 requires after every load; the gfx1201 contract is unchanged.
pub fn contract(spec: Spec, e: &Emitted) -> serde_json::Value {
    let gfx12 = spec.arch.gfx12();
    let symbol = if e.proof.variant.ends_with("_t8") {
        format!("{}_t8", spec.symbol())
    } else if e.proof.variant.ends_with("_cn_rows8") {
        cn_rows8_symbol(spec)
    } else {
        spec.symbol()
    };
    let counted: &[&str] = if gfx12 { &COUNTED } else { &COUNTED_GFX11 };
    let mut counts = serde_json::Map::new();
    for &name in counted {
        let n = e
            .s_text
            .lines()
            .filter(|l| l.split_whitespace().next() == Some(name))
            .count();
        if n > 0 {
            counts.insert(name.into(), n.into());
        }
    }
    let forbidden = if gfx12 {
        serde_json::json!([
            "scratch_*",
            "flat_*",
            "ds_swizzle*",
            "ds_read*",
            "ds_write*",
            "v_bcnt*",
            "v_wmma*",
            "v_swmmac*",
            "v_fma_mix*",
            "s_waitcnt*",
            "s_setpc*",
            "s_swappc*"
        ])
    } else {
        serde_json::json!([
            "scratch_*",
            "flat_*",
            "ds_swizzle*",
            "ds_read*",
            "ds_write*",
            "v_bcnt*",
            "v_wmma*",
            "v_swmmac*",
            "v_fma_mix*",
            "s_setpc*",
            "s_swappc*",
            "s_barrier_signal*",
            "s_barrier_wait*",
            "s_wait_loadcnt*",
            "s_wait_dscnt*",
            "s_wait_storecnt*",
            "s_wait_kmcnt*",
            "s_wait_alu",
            "s_add_co_*",
            "s_sub_co_*"
        ])
    };
    serde_json::json!({
        "symbol": symbol, "counts": counts,
        "forbidden": forbidden,
        "vgpr_max": e.shape.next_free_vgpr, "sgpr_max": e.shape.next_free_sgpr,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0, "group_segment_fixed_bytes": STAGE_BYTES,
    })
}
const COUNTED: [&str; 34] = [
    "s_load_b256",
    "s_load_b128",
    "s_load_b64",
    "s_load_b32",
    "global_load_u16",
    "global_load_b32",
    "buffer_load_b128",
    "global_store_b32",
    "ds_bpermute_b32",
    "ds_load_b32",
    "ds_store_b32",
    "s_barrier_signal",
    "s_barrier_wait",
    "v_writelane_b32",
    "v_readlane_b32",
    "v_cvt_f32_f16_e64",
    "v_cvt_f32_i32_e32",
    "v_bfe_i32",
    "v_add3_u32",
    "v_xor_b32_e32",
    "v_bfi_b32",
    "v_mul_f32_e32",
    "v_fmac_f32_e32",
    "v_add_f32_e32",
    "v_cmpx_eq_u32_e32",
    "v_cndmask_b32_e64",
    "v_add_lshl_u32",
    "v_mul_lo_u32",
    "v_min_u32_e32",
    "s_cbranch_scc0",
    "s_cbranch_scc1",
    "s_cselect_b64",
    "s_endpgm",
    "s_mov_b32",
];
/// The gfx11 counted set: [`COUNTED`] with the split barrier replaced by `s_barrier` and the builder-inserted
/// gfx11 wait / hazard / issue-hint instructions added.
const COUNTED_GFX11: [&str; 36] = [
    "s_load_b256",
    "s_load_b128",
    "s_load_b64",
    "s_load_b32",
    "global_load_u16",
    "global_load_b32",
    "buffer_load_b128",
    "global_store_b32",
    "ds_bpermute_b32",
    "ds_load_b32",
    "ds_store_b32",
    "s_barrier",
    "v_writelane_b32",
    "v_readlane_b32",
    "v_cvt_f32_f16_e64",
    "v_cvt_f32_i32_e32",
    "v_bfe_i32",
    "v_add3_u32",
    "v_xor_b32_e32",
    "v_bfi_b32",
    "v_mul_f32_e32",
    "v_fmac_f32_e32",
    "v_add_f32_e32",
    "v_cmpx_eq_u32_e32",
    "v_cndmask_b32_e64",
    "v_add_lshl_u32",
    "v_mul_lo_u32",
    "v_min_u32_e32",
    "s_cbranch_scc0",
    "s_cbranch_scc1",
    "s_cselect_b64",
    "s_endpgm",
    "s_mov_b32",
    "s_waitcnt",
    "s_waitcnt_depctr",
    "s_delay_alu",
];

/// Both K-E1 tables, for tests: the exact data the stage writes.
pub fn tables() -> ([u32; 256], [u32; 128]) {
    (build_lut256u32(), build_mask128u32())
}
