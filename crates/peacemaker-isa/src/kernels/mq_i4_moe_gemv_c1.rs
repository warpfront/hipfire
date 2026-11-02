// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! K1-C1: the gfx1151 (Halo) one-lane slab decode GEMV with the MQ4 `rows8` schedule and K1's exact arithmetic.
//! Two symbols `mq_i4_moe_gemv_c1_{r256_gate_up,r128_down}` in the module `mq_i4_moe_gemv_c1_pm_gfx1151`. Each serves BOTH
//! codecs (`Mq4Sym` = codec 0, `Mq3I4` = W3 = codec 1) from one launch through a workgroup-uniform branch, with no union
//! plan and no per-codec bucket launches. It replaces, for `lanes == 1` (AR decode) only, the K1 path
//! (`mq_i4_moe_gemv.rs`: union plan + `mq_i4_moe_gemv_{mq3i4,mq4sym}_*` buckets: one wave32 per (row, expert), X re-read from
//! global by every wave per group, one weight group in flight per wave).
//!
//! # Kernarg ABI (80 bytes, frozen)
//! `expert_weight_ptrs@0` (u64: `[n_experts]` u64 table, each entry the projection tensor base of that expert: code-plane base
//! for W3, first group for MQ4), `topk@8` (u64: `[10]` i32 routed expert ids, slot `u` = rank `u`), `codec_table@16`
//! (u64: `[n_experts]` u8 per-expert codec, 0 = MQ4, 1 = W3; read ONLY when `uniform_codec < 0`), `X@24` (F32 rotated input),
//! `Y0@32`, `Y1@40` (gate only; down ignores), u32 `M@48` (gate 1280, down 2560; `M % 8 == 0`), u32 `K@52` (gate 2560,
//! down 640), i32 `uniform_codec@56` (-1 = table; 0 = all MQ4; 1 = all W3), u32 `x_row_div@60` (gate 10, down 1), u32
//! `x_row_stride@64` (F32 elements, = K), u32 `y_slot_stride@68` (F32 elements; gate `M/2` = 640, down `M` = 2560), u32
//! `n_experts@72` (the extent of the `expert_weight_ptrs` table; of `codec_table` too), u32 `reserved@76` (zero, pads the
//! block to 8 bytes; never read).
//! The host validates every value; the kernel hardcodes its group counts and exits (uniformly, before any barrier) when
//! `K` differs from the symbol's K, `wg_y >= 10`, or `topk[wg_y] >= n_experts` (UNSIGNED compare, so a negative id too;
//! tested on the loaded id before it is shifted, so neither the weight-pointer table nor the codec table is indexed by an
//! out-of-range id, no expert tensor is dereferenced, and no store is issued).
//!
//! # Geometry
//! Grid `[M/8, 10, 1]`, block `[256, 1, 1]` (eight wave32), fixed LDS (`group_segment_fixed_size`): gate 10,240 B (one X
//! row of 2560 F32), down 2,560 B (640 F32). Workgroup `(wg_x, u = wg_y)`: slot `u` (= rank), `eid = topk[u]`,
//! `codec = uniform_codec >= 0 ? uniform_codec : codec_table[eid]`, `base = expert_weight_ptrs[eid]`, `wave = tid >> 5`
//! (scalar, `v_readfirstlane_b32`), `row = wg_x*8 + wave`, X row `xr = u / x_row_div` at `X + 4*xr*x_row_stride`. Output (lane
//! 0 of each wave): gate `row < M/2 ? Y0[u*y_slot_stride + row] : Y1[u*y_slot_stride + row - M/2]` (`M/2` is a multiple of
//! 8, so a workgroup is all-gate or all-up), down `Y0[u*y_slot_stride + row]`. No route weight, no atomics, no other writes.
//!
//! # Schedule (per workgroup; each codec body is complete, so every wave reaches the same barrier)
//! 1. Scalar: kernargs (the `n_experts` word included), guards, `eid`, then, before `eid` is shifted or used to index any
//!    table, the unsigned `eid >= n_experts` exit; the codec (table dword at `codec_table + (eid & !3)`, byte `eid & 3`, only when
//!    `uniform_codec < 0`), `base`, `xr`, the X-row and Y pointers, every lane-constant VGPR and the shuffle masks (all of
//!    this but the codec/base resolution hides under the scalar-load latency). Then ONE `s_waitcnt lgkmcnt(0)` and the uniform branch.
//! 2. X staging loads: gate 640 `float4` (thread `t` loads `t`, `t+256`, and `t+512` only for `t < 128`), down 160
//!    (`t < 160`), `global_load_b128`; the partial loads run under `exec = EMASK` (a wave-uniform mask: waves 0..3 gate, 0..4
//!    down) and `exec` is restored afterwards.
//! 3. EVERY weight load of the wave's row, in group order, before any wait.
//! 4. `s_waitcnt vmcnt(<weight loads>)` (global loads return in issue order, so this waits for the X loads only), the
//!    `ds_store_b128` of the staged X, `s_waitcnt lgkmcnt(0)` + `s_barrier` (builder-inserted by `barrier`).
//! 5. Groups ascending: X for group `g+1` is read from LDS (`ds_load_b128`) before group `g` is computed; the builder
//!    waits `vmcnt(<loads issued after this group's>)` before the first use of each group's weights.
//! 6. Fold, 5-step reduction, lane-0 store, `s_endpgm`.
//!
//! # Exactness law (identical to K1 at `lanes = 1`)
//! For every (slot, row) the stored F32 is bit-identical to `mq_i4_moe_gemv.rs` at `lanes = 1`:
//!  * lane `l` owns `g*128 + 4l ..+3` (W3 both kinds, MQ4 down) or `g*256 + 8l ..+7` (MQ4 gate);
//!  * W3: `d = cvt_f32_f16(scale)`; per code `i` `c = bfe(pkt, 3i, 3)`, `t = 2c - 7` (here one `v_lshl_add_u32 t, c, 1, -7`, the
//!    same integer as K1's `v_lshlrev_b32` + `v_add_nc_u32 -7`), `w_i = v_mul_f32(d, cvt_f32_i32(t))`; `p = v_mul_f32(w0, x0)`
//!    then `v_fmac_f32(p, w_j, x_j)`, `j = 1..3`;
//!  * MQ4: `d = cvt(h.lo16)`, `z = cvt(h.hi16)`, `w_j = v_fma_f32(d, cvt_f32_u32(bfe(pkt, 4j, 4)), z)`, `p = v_mul_f32(w1, x1)`,
//!    `v_fmac(w0, x0)`, then `v_fmac(w_j, x_j)`, `j = 2..nw-1`;
//!  * accumulators start at `+0`, `acc[c] = v_add_f32(acc[c], p)` (gate chain `c = g & 3`, down chain 0), groups ascending;
//!    gate fold `a0 = a0 + a1; a2 = a2 + a3; a0 = a0 + a2`; the snapshot shuffle-down at offsets 16, 8, 4, 2, 1 with
//!    `ds_bpermute_b32` (address `(lane + off)*4` if `lane < 32 - off` else `lane*4`), `exch = v_add_f32(a0, exch)`,
//!    `a0 = v_cndmask(a0, exch, mask)`; lane 0 stores. No packed math, dot instruction, reassociation or other rounding
//!    touches the value path.
//!
//! # W3 b64 packet (the only structural change to the unpack network) and the over-read argument
//! K1's W3 packet of lane `l` is `(b0 | b1 << 8) >> (4*(l & 1))` with `b0` the group byte `(3l) >> 1`; it therefore starts at
//! group bit `8*((3l)>>1) + 4*(l&1) = 12 l`, and only bits `[0, 12)` of it are read (the four `bfe(pkt, 3i, 3)`). The C1
//! packet is ONE `global_load_b64` at group byte `4*((12 l) >> 5)` and, at compute time, `v_lshrrev_b64 pkt:pkt+1,
//! (12 l) & 31, hi:lo` (the 64-bit funnel shift; the llvm gfx11.5 `v_alignbit_b32` carries a 16-bit shift operand the native
//! codec does not model, so the same bits come from the 64-bit shift): bits `[0, 12)` of the low dword `pkt` are group bits `[12 l, 12 l + 12)` of the 8-byte little-endian window (the window
//! covers bits `32*((12l)>>5) .. +63`, and `(12l)&31 + 12 <= 43 < 64`), i.e. exactly K1's twelve bits. The last lane's
//! window (`l = 31`: byte 44) extends four bytes past its 48-byte group: into the next group, the next row, or for the last
//! row into the scale plane that follows the code plane in the tensor (planar layout, `M*K*3/8` code bytes then `2*M*K/128`
//! scale bytes); the read is therefore always inside the expert tensor and its bytes are never used. Scales (gate): the
//! row's 40 scale bytes at `base + M*R + 40*row` (`R = K*3/8`) are 8-byte aligned and loaded as `s_load_b256` +
//! `s_load_b64` into ten SGPRs (group `g`'s f16 is the low half of SGPR `g>>1` for even `g`, the high half for odd `g`);
//! this REQUIRES the scale plane base (`base + M*R`) to be 4-byte aligned (the scalar load drops the low two address bits).
//! Down (G = 5): one `global_load_u16` per group, same address in every lane (K1's own load).
//!
//! # LDS and the retire barrier
//! The staged X lives in one declared slot. The two codec bodies are separate straight-line streams over one linear
//! builder state, so the first-emitted body (W3) retires the slot with a final `s_barrier` (after its lane-0 store, no
//! work behind it) before the second body (MQ4) may store to it again; MQ4 is the later body and ends without it.
//!
//! The kernel contract (`contract`) lists the counts of the characteristic instructions, the forbidden families, `vgpr_max`,
//! `sgpr_max` and the fixed LDS bytes; it is committed as `kernels/mq_i4_moe_gemv_c1.gfx1151.<tag>.contract.json`.

use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{add64, lit, op, s, s_add_i32, s_addc_u32, smem, sop, sr, v, vr};
use crate::lds::Transition;
use crate::ledger::Counter;
use crate::plan::Access;
use crate::reg::{Kind as RKind, Live, RegRef};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// The kernarg segment size.
pub const KERNARG_BYTES: u32 = 80;
/// Workgroup size (threads): eight wave32.
pub const WORKGROUP_SIZE: u32 = 256;
/// Output rows (waves) of one workgroup.
pub const ROWS_PER_WORKGROUP: u32 = 8;
/// Routed slots (grid y).
pub const SLOTS: u32 = 10;
/// The module name.
pub const MODULE: &str = "mq_i4_moe_gemv_c1_pm_gfx1151";
/// The two exported symbols in module order.
pub const SYMBOLS: [&str; 2] = [
    "mq_i4_moe_gemv_c1_r256_gate_up",
    "mq_i4_moe_gemv_c1_r128_down",
];

/// Projection kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// Gate/up (`K2560`): four group chains, fold `(a0+a1)+(a2+a3)`, Y0/Y1 split at `M/2`.
    GateUp,
    /// Down (`K640`): one chain, Y0 only.
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
    /// Columns per row.
    pub fn k(self) -> u32 {
        match self {
            Self::GateUp => 2560,
            Self::Down => 640,
        }
    }
    /// Fixed LDS bytes: one F32 X row.
    pub fn lds_bytes(self) -> u32 {
        self.k() * 4
    }
    /// Accumulator chains.
    pub fn chains(self) -> u8 {
        match self {
            Self::GateUp => 4,
            Self::Down => 1,
        }
    }
    /// 128-column W3 groups per row.
    pub fn w3_groups(self) -> u32 {
        self.k() / 128
    }
    /// MQ4 groups per row (gate G256, down G128).
    pub fn mq4_groups(self) -> u32 {
        match self {
            Self::GateUp => 10,
            Self::Down => 5,
        }
    }
    /// `float4` X staging loads per thread (the last gate load and the single down load are partial).
    fn x_loads(self) -> u8 {
        match self {
            Self::GateUp => 3,
            Self::Down => 1,
        }
    }
    /// Waves whose threads own a partial X load (`t < 128`: four waves, `t < 160`: five).
    fn partial_waves(self) -> u32 {
        match self {
            Self::GateUp => 4,
            Self::Down => 5,
        }
    }
}

/// Weight codec of a workgroup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Codec {
    Mq4,
    W3,
}

/// One entry: gfx1151 only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
    pub kind: Kind,
}
impl Spec {
    pub fn new(kind: Kind) -> Self {
        Self {
            arch: Arch::Gfx1151,
            kind,
        }
    }
    /// The exported symbol, `mq_i4_moe_gemv_c1_<kind>`.
    pub fn symbol(&self) -> String {
        format!("mq_i4_moe_gemv_c1_{}", self.kind.tag())
    }
    /// The `--epi` name and the contract file tag.
    pub fn tag(&self) -> &'static str {
        self.kind.tag()
    }
    pub fn module() -> String {
        MODULE.into()
    }
    pub fn validate(&self) -> R {
        if self.arch != Arch::Gfx1151 {
            return Err("mq_i4_moe_gemv_c1 builder is exact-gfx1151 only".into());
        }
        Ok(())
    }
}

/// Both entries in module order.
pub fn module_specs() -> Vec<Spec> {
    Kind::ALL.into_iter().map(Spec::new).collect()
}

// ----------------------------------------------------------------------- SGPRs
// s0:1 kernarg pointer; s2, s3 workgroup ids x, y (gfx11 system SGPRs after the two user SGPRs).
const KARG: u8 = 0;
const WGX: u8 = 2;
const WGY: u8 = 3;
/// Kernargs: `s_load_b256 s[8:15]` (@0), `s_load_b256 s[16:23]` (@0x20), `s_load_b64 s[24:25]` (@0x40), `s_load_b32 s76` (@0x48).
const WPTRS: u8 = 8;
const TOPK: u8 = 10;
const CTAB: u8 = 12;
const XP: u8 = 14;
const Y0: u8 = 16;
const Y1: u8 = 18;
const SM: u8 = 20;
const SK: u8 = 21;
const SUC: u8 = 22;
const SDIV: u8 = 23;
const SXSTR: u8 = 24;
const SYSTR: u8 = 25;
const EID: u8 = 26;
const CODEC: u8 = 27;
const TMP: u8 = 28;
const TMP2: u8 = 29;
const TMP3: u8 = 30;
const WAVE: u8 = 31;
const TPTR: u8 = 32;
const BPTR: u8 = 34;
const CPTR: u8 = 36;
const BASE: u8 = 38;
const ROW: u8 = 40;
const XR: u8 = 41;
const ROWIN: u8 = 42;
const MI: u8 = 43;
const ROWM: u8 = 44;
const EMASK: u8 = 45;
const XROW: u8 = 46;
const ROWPTR: u8 = 48;
const YP: u8 = 50;
/// Five shuffle masks (one SGPR each, wave32).
const SHM: u8 = 52;
// W3-body-only SGPRs (the window aliases nothing else): scale pointer, ten scale dwords (gate), two shift temporaries.
const SPTR: u8 = 58;
const SC: u8 = 64;
const SC8: u8 = 72;
const STMP: u8 = 74;
/// `n_experts@72`: dedicated SGPR, aliased with nothing (above every other scalar window of both bodies).
const SNEXP: u8 = 76;
/// SGPRs the plan declares (the honest ceiling, `sgpr_max` of the contract).
const SGPRS: u8 = 77;

// ----------------------------------------------------------------------- VGPRs
/// The VGPRs shared by both bodies (live in the whole kernel): the lane-derived constants are computed once, before the
/// branch, under the scalar-load latency.
#[derive(Clone, Copy, Debug)]
struct Com {
    tid: u8,
    lane: u8,
    xo: [u8; 3],
    zero: u8,
    shuf: u8,
    exch: u8,
    /// `16*lane`: LDS X address of the W3 and MQ4-down lanes.
    ldsq: u8,
    /// W3 packet byte offset `4*((12*lane) >> 5)` and bit shift `(12*lane) & 31`.
    w3_off: u8,
    w3_sh: u8,
    /// MQ4 payload byte offset (`4*lane` gate, `2*lane` down), header byte offset `4*(lane >> 4)` and the `32*lane` LDS X address (gate).
    mq_poff: u8,
    mq_hoff: u8,
    mq_ldsq: u8,
    end: u8,
}
impl Com {
    fn new(kind: Kind) -> Self {
        let gate = kind == Kind::GateUp;
        let mut n = 0u8;
        let mut take = |len: u8| {
            let base = n;
            n += len;
            base
        };
        let tid = take(1);
        let lane = take(1);
        let xo = [
            take(1),
            if gate { take(1) } else { 0 },
            if gate { take(1) } else { 0 },
        ];
        let zero = take(1);
        let shuf = take(5);
        let exch = take(1);
        let ldsq = take(1);
        let w3_off = take(1);
        let w3_sh = take(1);
        let mq_poff = take(1);
        let mq_hoff = if gate { take(1) } else { 0 };
        let mq_ldsq = if gate { take(1) } else { 0 };
        Self {
            tid,
            lane,
            xo,
            zero,
            shuf,
            exch,
            ldsq,
            w3_off,
            w3_sh,
            mq_poff,
            mq_hoff,
            mq_ldsq,
            end: n,
        }
    }
}

/// The per-codec VGPR window (the two windows alias each other: they are live in disjoint label ranges).
#[derive(Clone, Debug)]
struct Body {
    /// `float4` X staging registers (four VGPRs each; down uses the first only).
    stage: [u8; 3],
    /// Per-group first load: W3 packet pair, MQ4 header.
    a: Vec<u8>,
    /// Per-group second load: down-W3 scale u16, MQ4 payload; empty for gate W3 (its scales are SGPRs).
    b: Vec<u8>,
    /// Two X ping-pong buffers of `nw` registers.
    xb: [u8; 2],
    /// Weights, scale, zero point, packet, dot, accumulators.
    w: u8,
    d: u8,
    z: u8,
    pk: u8,
    dot: u8,
    acc: u8,
    nw: u8,
    end: u8,
}
impl Body {
    fn new(kind: Kind, codec: Codec, start: u8) -> Self {
        let mut n = start;
        let mut take = |len: u8, align: u8| {
            n = n.div_ceil(align) * align;
            let base = n;
            n += len;
            base
        };
        let gate = kind == Kind::GateUp;
        let groups = match codec {
            Codec::W3 => kind.w3_groups(),
            Codec::Mq4 => kind.mq4_groups(),
        } as usize;
        let nw = if codec == Codec::Mq4 && gate { 8 } else { 4 };
        let stage = [
            take(4, 4),
            if gate { take(4, 4) } else { 0 },
            if gate { take(4, 4) } else { 0 },
        ];
        let (a, b) = match codec {
            Codec::W3 => (
                (0..groups).map(|_| take(2, 2)).collect(),
                if gate {
                    vec![]
                } else {
                    (0..groups).map(|_| take(1, 1)).collect()
                },
            ),
            Codec::Mq4 => (
                (0..groups).map(|_| take(1, 1)).collect(),
                (0..groups).map(|_| take(1, 1)).collect(),
            ),
        };
        let xb = [take(nw, 4), take(nw, 4)];
        let w = take(nw, 4);
        let d = take(1, 1);
        let z = if codec == Codec::Mq4 { take(1, 1) } else { 0 };
        let pk = if codec == Codec::W3 { take(2, 2) } else { 0 };
        let dot = take(1, 1);
        let acc = take(kind.chains(), 1);
        Self {
            stage,
            a,
            b,
            xb,
            w,
            d,
            z,
            pk,
            dot,
            acc,
            nw,
            end: n,
        }
    }
}

/// Everything one symbol needs: its kind, the register layout of both windows and the label names.
struct Ctx {
    kind: Kind,
    com: Com,
    w3: Body,
    mq4: Body,
    vgprs: u16,
}
impl Ctx {
    fn new(kind: Kind) -> Self {
        let com = Com::new(kind);
        let start = com.end.div_ceil(4) * 4;
        let (w3, mq4) = (
            Body::new(kind, Codec::W3, start),
            Body::new(kind, Codec::Mq4, start),
        );
        let vgprs = u16::from(w3.end.max(mq4.end)).div_ceil(8) * 8;
        Self {
            kind,
            com,
            w3,
            mq4,
            vgprs,
        }
    }
    fn lbl(&self, name: &str) -> String {
        format!(".Lc1_{}_{name}", self.kind.tag())
    }
    fn body(&self, codec: Codec) -> &Body {
        match codec {
            Codec::W3 => &self.w3,
            Codec::Mq4 => &self.mq4,
        }
    }
}

fn decl(plan: &mut RegPlan, name: &str, kind: RKind, base: u8, len: u8, live: &Live) -> R {
    plan.add_range(name, kind, base, len, live.clone())
}

fn plan(c: &Ctx) -> Result<RegPlan, String> {
    let mut regs = RegPlan::new(c.vgprs, u16::from(SGPRS))?;
    let whole = Live::Whole;
    for (name, base) in [("workgroup_id_x", WGX), ("workgroup_id_y", WGY)] {
        decl(&mut regs, name, RKind::S, base, 1, &whole)?
    }
    for (name, base, len) in [
        ("kernarg", KARG, 2),
        ("kernel_arguments_ptrs", WPTRS, 8),
        ("kernel_arguments_y_m_k", Y0, 8),
        ("kernel_arguments_strides", SXSTR, 2),
        ("expert_id", EID, 1),
        ("codec", CODEC, 1),
        ("temp", TMP, 1),
        ("temp2", TMP2, 1),
        ("temp3", TMP3, 1),
        ("wave", WAVE, 1),
        ("topk_pointer", TPTR, 2),
        ("table_pointer", BPTR, 2),
        ("codec_pointer", CPTR, 2),
        ("expert_pointer", BASE, 2),
        ("row", ROW, 1),
        ("x_row", XR, 1),
        ("row_in_y", ROWIN, 1),
        ("half_rows", MI, 1),
        ("row_minus_half", ROWM, 1),
        ("exec_mask", EMASK, 1),
        ("x_row_pointer", XROW, 2),
        ("row_pointer", ROWPTR, 2),
        ("y_pointer", YP, 2),
    ] {
        decl(&mut regs, name, RKind::S, base, len, &whole)?
    }
    decl(
        &mut regs,
        "kernel_arguments_n_experts",
        RKind::S,
        SNEXP,
        1,
        &whole,
    )?;
    for k in 0..5 {
        decl(&mut regs, "shuffle_mask", RKind::S, SHM + k, 1, &whole)?
    }
    let com = &c.com;
    for (name, base) in [
        ("tid", com.tid),
        ("lane", com.lane),
        ("x_offset0", com.xo[0]),
        ("zero", com.zero),
        ("exchange", com.exch),
        ("lds_lane_address", com.ldsq),
        ("w3_packet_offset", com.w3_off),
        ("w3_packet_shift", com.w3_sh),
        ("mq4_payload_offset", com.mq_poff),
    ] {
        decl(&mut regs, name, RKind::V, base, 1, &whole)?
    }
    if c.kind == Kind::GateUp {
        for (name, base) in [
            ("x_offset1", com.xo[1]),
            ("x_offset2", com.xo[2]),
            ("mq4_header_offset", com.mq_hoff),
            ("mq4_lds_lane_address", com.mq_ldsq),
        ] {
            decl(&mut regs, name, RKind::V, base, 1, &whole)?
        }
    }
    for k in 0..5 {
        decl(
            &mut regs,
            "shuffle_address",
            RKind::V,
            com.shuf + k,
            1,
            &whole,
        )?
    }
    for (codec, start, end) in [
        (Codec::W3, c.lbl("w3"), c.lbl("mq4")),
        (Codec::Mq4, c.lbl("mq4"), c.lbl("exit")),
    ] {
        let live = Live::Between(start, end);
        let body = c.body(codec);
        for (i, &base) in body
            .stage
            .iter()
            .enumerate()
            .take(usize::from(c.kind.x_loads()))
        {
            decl(&mut regs, &format!("x_stage{i}"), RKind::V, base, 4, &live)?
        }
        for (i, &base) in body.a.iter().enumerate() {
            let (name, width) = if codec == Codec::W3 {
                ("w3_packet", 2)
            } else {
                ("mq4_header", 1)
            };
            decl(
                &mut regs,
                &format!("{name}{i}"),
                RKind::V,
                base,
                width,
                &live,
            )?;
        }
        for (i, &base) in body.b.iter().enumerate() {
            decl(
                &mut regs,
                &format!(
                    "{}{i}",
                    if codec == Codec::W3 {
                        "w3_scale"
                    } else {
                        "mq4_payload"
                    }
                ),
                RKind::V,
                base,
                1,
                &live,
            )?
        }
        for (i, &base) in body.xb.iter().enumerate() {
            decl(
                &mut regs,
                &format!("x_values{i}"),
                RKind::V,
                base,
                body.nw,
                &live,
            )?
        }
        decl(&mut regs, "weights", RKind::V, body.w, body.nw, &live)?;
        decl(&mut regs, "scale", RKind::V, body.d, 1, &live)?;
        if codec == Codec::Mq4 {
            decl(&mut regs, "zero_point", RKind::V, body.z, 1, &live)?
        } else {
            decl(&mut regs, "packet", RKind::V, body.pk, 2, &live)?
        }
        decl(&mut regs, "dot", RKind::V, body.dot, 1, &live)?;
        decl(
            &mut regs,
            "accumulators",
            RKind::V,
            body.acc,
            c.kind.chains(),
            &live,
        )?;
    }
    // W3 scales: pointer pair, ten scale dwords (gate), two shift temporaries.
    let live = Live::Between(c.lbl("w3"), c.lbl("mq4"));
    decl(&mut regs, "scale_pointer", RKind::S, SPTR, 2, &live)?;
    if c.kind == Kind::GateUp {
        decl(&mut regs, "scale_dwords", RKind::S, SC, 8, &live)?;
        decl(&mut regs, "scale_dwords_tail", RKind::S, SC8, 2, &live)?;
        for k in 0..2 {
            decl(&mut regs, "scale_shift_temp", RKind::S, STMP + k, 1, &live)?
        }
    }
    Ok(regs)
}

/// Production entry.
pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    let c = Ctx::new(spec.kind);
    let kernargs = KernargLayout::new(KERNARG_BYTES)
        .pointer_access("expert_weight_ptrs", 0, Access::ReadOnly)
        .pointer_access("topk", 8, Access::ReadOnly)
        .pointer_access("codec_table", 16, Access::ReadOnly)
        .pointer_access("X", 24, Access::ReadOnly)
        .pointer_access("Y0", 32, Access::WriteOnly)
        .pointer_access("Y1", 40, Access::WriteOnly)
        .hidden("M", 48, 4, "by_value")
        .hidden("K", 52, 4, "by_value")
        .hidden("uniform_codec", 56, 4, "by_value")
        .hidden("x_row_div", 60, 4, "by_value")
        .hidden("x_row_stride", 64, 4, "by_value")
        .hidden("y_slot_stride", 68, 4, "by_value")
        .hidden("n_experts", 72, 4, "by_value")
        .hidden("reserved", 76, 4, "by_value");
    let kspec = KernelSpec {
        kernel_id: "mq_i4_moe_gemv_c1".into(),
        variant: spec.tag().into(),
        arch: spec.arch,
        symbol: spec.symbol(),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: WORKGROUP_SIZE as u16,
        group_segment_fixed_size: spec.kind.lds_bytes(),
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan(&c)?);
    b.enable_delay_alu();
    let slot = b.lds_slot("x", 0, spec.kind.lds_bytes())?;
    prologue(&mut b, &c)?;
    b.label(&c.lbl("w3"))?;
    body(&mut b, &c, Codec::W3, slot)?;
    b.label(&c.lbl("mq4"))?;
    body(&mut b, &c, Codec::Mq4, slot)?;
    b.label(&c.lbl("exit"))?;
    op(&mut b, "s_endpgm", &[], &[])?;
    b.finish()
}

/// The two entries as one code object.
pub fn emit_module() -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = module_specs()
        .into_iter()
        .map(emit)
        .collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, &Spec::module())?;
    Ok((emitted, text, proof))
}

fn s_sub_i32() -> &'static str {
    "s_sub_i32"
}

/// Kernarg loads, the lane constants and shuffle masks, guards, expert/codec/base resolution and the codec branch.
fn prologue(b: &mut Builder, c: &Ctx) -> R {
    let arch = b.spec.arch;
    let (kind, com) = (c.kind, &c.com);
    let gate = kind == Kind::GateUp;
    let exit = c.lbl("exit");
    smem(b, WPTRS, 8, KARG, 0)?;
    smem(b, Y0, 8, KARG, 0x20)?;
    smem(b, SXSTR, 2, KARG, 0x40)?;
    smem(b, SNEXP, 1, KARG, 0x48)?;
    match arch.workgroup_ids(2)? {
        crate::arch::WgIds {
            x: crate::arch::WgIdSource::Sgpr(x),
            y: crate::arch::WgIdSource::Sgpr(y),
        } if (x, y) == (WGX, WGY) => {}
        ids => {
            return Err(format!(
                "mq_i4_moe_gemv_c1: unexpected workgroup id sources {ids:?}"
            ))
        }
    }
    // Lane-derived constants (VALU, under the kernarg latency).
    valu(
        b,
        &format!("v_and_b32_e32 v{}, 31, v{}", com.lane, com.tid),
        &[com.lane],
        &[com.tid],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{}, 4, v{}", com.xo[0], com.tid),
        &[com.xo[0]],
        &[com.tid],
        &[],
    )?;
    if gate {
        valu(
            b,
            &format!(
                "v_add_nc_u32_e32 v{}, {}, v{}",
                com.xo[1],
                lit(4096),
                com.xo[0]
            ),
            &[com.xo[1]],
            &[com.xo[0]],
            &[],
        )?;
        valu(
            b,
            &format!(
                "v_add_nc_u32_e32 v{}, {}, v{}",
                com.xo[2],
                lit(8192),
                com.xo[0]
            ),
            &[com.xo[2]],
            &[com.xo[0]],
            &[],
        )?;
    }
    valu(
        b,
        &format!("v_mov_b32_e32 v{}, 0", com.zero),
        &[com.zero],
        &[],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshrrev_b32_e32 v{}, 5, v{}", com.exch, com.tid),
        &[com.exch],
        &[com.tid],
        &[],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{WAVE}, v{}", com.exch),
        &[s(WAVE)],
        &[v(com.exch)],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{}, 4, v{}", com.ldsq, com.lane),
        &[com.ldsq],
        &[com.lane],
        &[],
    )?;
    // W3: packet byte offset 4*((12 l) >> 5) and bit shift (12 l) & 31.
    valu(
        b,
        &format!("v_mul_u32_u24_e32 v{}, 12, v{}", com.exch, com.lane),
        &[com.exch],
        &[com.lane],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshrrev_b32_e32 v{}, 5, v{}", com.w3_off, com.exch),
        &[com.w3_off],
        &[com.exch],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{0}, 2, v{0}", com.w3_off),
        &[com.w3_off],
        &[com.w3_off],
        &[],
    )?;
    valu(
        b,
        &format!("v_and_b32_e32 v{}, 31, v{}", com.w3_sh, com.exch),
        &[com.w3_sh],
        &[com.exch],
        &[],
    )?;
    // MQ4: gate payload word 4l, header half 4*(l >> 4), X at 32*l; down payload u16 at 2l.
    if gate {
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{}, 2, v{}", com.mq_poff, com.lane),
            &[com.mq_poff],
            &[com.lane],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshrrev_b32_e32 v{}, 4, v{}", com.mq_hoff, com.lane),
            &[com.mq_hoff],
            &[com.lane],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{0}, 2, v{0}", com.mq_hoff),
            &[com.mq_hoff],
            &[com.mq_hoff],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{}, 5, v{}", com.mq_ldsq, com.lane),
            &[com.mq_ldsq],
            &[com.lane],
            &[],
        )?;
    } else {
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{}, 1, v{}", com.mq_poff, com.lane),
            &[com.mq_poff],
            &[com.lane],
            &[],
        )?;
    }
    // Shuffle step k (offset 16 >> k): mask lane < 32 - off, address (lane + off or lane) * 4.
    for (k, off) in [16u32, 8, 4, 2, 1].into_iter().enumerate() {
        let (m, sh) = (SHM + k as u8, com.shuf + k as u8);
        op(
            b,
            format!("v_cmp_gt_u32_e64 s{m}, {}, v{}", 32 - off, com.lane),
            &[s(m)],
            &[v(com.lane)],
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
            &format!("v_add_lshl_u32 v{sh}, v{sh}, v{}, 2", com.lane),
            &[sh],
            &[sh, com.lane],
            &[],
        )?;
    }
    // Guards (workgroup-uniform, before any barrier).
    sop(b, format!("s_cmp_ge_u32 s{WGY}, {SLOTS}"), &[], &[WGY])?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    sop(
        b,
        format!("s_cmp_lg_u32 s{SK}, {}", lit(kind.k())),
        &[],
        &[SK],
    )?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    // eid = topk[u].
    sop(b, format!("s_lshl_b32 s{TMP}, s{WGY}, 2"), &[TMP], &[WGY])?;
    add64(b, TPTR, TOPK, TMP)?;
    smem(b, EID, 1, TPTR, 0)?;
    // Under the eid load: row, X row, X/Y pointers, the partial-load exec mask.
    sop(b, format!("s_lshl_b32 s{ROW}, s{WGX}, 3"), &[ROW], &[WGX])?;
    sop(
        b,
        format!("{} s{ROW}, s{ROW}, s{WAVE}", s_add_i32(arch)),
        &[ROW],
        &[ROW, WAVE],
    )?;
    // xr = u / x_row_div for u < 10: the number of i in 1..=9 with i*div <= u.
    sop(b, format!("s_mov_b32 s{XR}, 0"), &[XR], &[])?;
    for i in 1..=9u32 {
        sop(
            b,
            format!("s_mul_i32 s{TMP}, s{SDIV}, {i}"),
            &[TMP],
            &[SDIV],
        )?;
        sop(b, format!("s_cmp_ge_u32 s{WGY}, s{TMP}"), &[], &[WGY, TMP])?;
        sop(
            b,
            format!("{} s{XR}, s{XR}, 0", s_addc_u32(arch)),
            &[XR],
            &[XR],
        )?;
    }
    sop(
        b,
        format!("s_mul_i32 s{TMP}, s{XR}, s{SXSTR}"),
        &[TMP],
        &[XR, SXSTR],
    )?;
    sop(b, format!("s_lshl_b32 s{TMP}, s{TMP}, 2"), &[TMP], &[TMP])?;
    add64(b, XROW, XP, TMP)?;
    let rowin = if gate {
        sop(b, format!("s_lshr_b32 s{MI}, s{SM}, 1"), &[MI], &[SM])?;
        sop(
            b,
            format!("{} s{ROWM}, s{ROW}, s{MI}", s_sub_i32()),
            &[ROWM],
            &[ROW, MI],
        )?;
        sop(b, format!("s_cmp_lt_u32 s{ROW}, s{MI}"), &[], &[ROW, MI])?;
        op(
            b,
            format!(
                "s_cselect_b64 s[{YP}:{}], s[{Y0}:{}], s[{Y1}:{}]",
                YP + 1,
                Y0 + 1,
                Y1 + 1
            ),
            &[sr(YP, 2)],
            &[sr(Y0, 2), sr(Y1, 2)],
        )?;
        sop(
            b,
            format!("s_cselect_b32 s{ROWIN}, s{ROW}, s{ROWM}"),
            &[ROWIN],
            &[ROW, ROWM],
        )?;
        ROWIN
    } else {
        sop(b, format!("s_mov_b32 s{YP}, s{Y0}"), &[YP], &[Y0])?;
        sop(
            b,
            format!("s_mov_b32 s{}, s{}", YP + 1, Y0 + 1),
            &[YP + 1],
            &[Y0 + 1],
        )?;
        ROW
    };
    sop(
        b,
        format!("s_mul_i32 s{TMP}, s{WGY}, s{SYSTR}"),
        &[TMP],
        &[WGY, SYSTR],
    )?;
    sop(
        b,
        format!("{} s{TMP}, s{TMP}, s{rowin}", s_add_i32(arch)),
        &[TMP],
        &[TMP, rowin],
    )?;
    sop(b, format!("s_lshl_b32 s{TMP}, s{TMP}, 2"), &[TMP], &[TMP])?;
    add64(b, YP, YP, TMP)?;
    sop(
        b,
        format!("s_cmp_lt_u32 s{WAVE}, {}", kind.partial_waves()),
        &[],
        &[WAVE],
    )?;
    sop(b, format!("s_cselect_b32 s{EMASK}, -1, 0"), &[EMASK], &[])?;
    // Expert id range check on the loaded id, before it is shifted or used to index any table: an id at or past
    // `n_experts` (UNSIGNED, so a negative id too) exits the workgroup uniformly (no barrier reached, no load, no store).
    sop(
        b,
        format!("s_cmp_ge_u32 s{EID}, s{SNEXP}"),
        &[],
        &[EID, SNEXP],
    )?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    // base = expert_weight_ptrs[eid].
    sop(b, format!("s_lshl_b32 s{TMP}, s{EID}, 3"), &[TMP], &[EID])?;
    add64(b, BPTR, WPTRS, TMP)?;
    smem(b, BASE, 2, BPTR, 0)?;
    // codec = uniform_codec >= 0 ? uniform_codec : codec_table[eid] (byte eid & 3 of the aligned dword).
    sop(b, format!("s_mov_b32 s{CODEC}, s{SUC}"), &[CODEC], &[SUC])?;
    sop(b, format!("s_cmp_lt_i32 s{SUC}, 0"), &[], &[SUC])?;
    let (known, join) = (c.lbl("known"), c.lbl("join"));
    branch(b, &format!("s_cbranch_scc0 {known}"))?;
    sop(b, format!("s_and_b32 s{TMP}, s{EID}, -4"), &[TMP], &[EID])?;
    add64(b, CPTR, CTAB, TMP)?;
    smem(b, TMP2, 1, CPTR, 0)?;
    sop(b, format!("s_lshl_b32 s{TMP3}, s{EID}, 3"), &[TMP3], &[EID])?;
    sop(
        b,
        format!("s_and_b32 s{TMP3}, s{TMP3}, 24"),
        &[TMP3],
        &[TMP3],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{TMP2}, s{TMP2}, s{TMP3}"),
        &[TMP2],
        &[TMP2, TMP3],
    )?;
    sop(
        b,
        format!("s_and_b32 s{CODEC}, s{TMP2}, 0xff"),
        &[CODEC],
        &[TMP2],
    )?;
    b.wait_all()?;
    branch(b, &format!("s_branch {join}"))?;
    b.label(&known)?;
    // The builder's ledger is linear: the table path above already drained the pending `base` load, so the uniform
    // path's wait for it is explicit.
    b.wait(Counter::Lgkm, 0)?;
    b.label(&join)?;
    sop(b, format!("s_cmp_eq_u32 s{CODEC}, 0"), &[], &[CODEC])?;
    branch(b, &format!("s_cbranch_scc1 {}", c.lbl("mq4")))
}

/// One complete codec body (steps 2-6); the workgroup's waves all take the same body.
fn body(b: &mut Builder, c: &Ctx, codec: Codec, slot: usize) -> R {
    let arch = b.spec.arch;
    let (kind, com, bd) = (c.kind, &c.com, c.body(codec));
    let gate = kind == Kind::GateUp;
    // Row pointers (and the gate W3 scale loads, issued first so they land under the X / weight loads).
    match codec {
        Codec::W3 => {
            let r = kind.k() * 3 / 8;
            sop(
                b,
                format!("s_mul_i32 s{TMP}, s{ROW}, {}", lit(r)),
                &[TMP],
                &[ROW],
            )?;
            add64(b, ROWPTR, BASE, TMP)?;
            // scale plane at base + M*R, f16 of (row, g) at +2*(row*G + g)
            sop(
                b,
                format!("s_mul_i32 s{TMP}, s{SM}, {}", lit(r)),
                &[TMP],
                &[SM],
            )?;
            sop(
                b,
                format!("s_mul_i32 s{TMP2}, s{ROW}, {}", 2 * kind.w3_groups()),
                &[TMP2],
                &[ROW],
            )?;
            sop(
                b,
                format!("{} s{TMP}, s{TMP}, s{TMP2}", s_add_i32(arch)),
                &[TMP],
                &[TMP, TMP2],
            )?;
            add64(b, SPTR, BASE, TMP)?;
            if gate {
                smem(b, SC, 8, SPTR, 0)?;
                smem(b, SC8, 2, SPTR, 0x20)?;
            }
        }
        Codec::Mq4 => {
            let stride = kind.mq4_groups() * if gate { 136 } else { 68 };
            sop(
                b,
                format!("s_mul_i32 s{TMP}, s{ROW}, {}", lit(stride)),
                &[TMP],
                &[ROW],
            )?;
            add64(b, ROWPTR, BASE, TMP)?;
        }
    }
    // Step 2: X staging loads. Gate: float4 t, t+256 (all threads), t+512 (t < 128); down: float4 t (t < 160). The
    // partial ones run under the wave-uniform exec mask (the instruction stream is identical in every wave, so the
    // in-order VM counter sees the same number of loads everywhere).
    if gate {
        gload(b, "global_load_b128", bd.stage[0], 4, com.xo[0], XROW, 0)?;
        gload(b, "global_load_b128", bd.stage[1], 4, com.xo[1], XROW, 0)?;
    }
    op(b, format!("s_mov_b32 exec_lo, s{EMASK}"), &[], &[s(EMASK)])?;
    if gate {
        gload(b, "global_load_b128", bd.stage[2], 4, com.xo[2], XROW, 0)?
    } else {
        gload(b, "global_load_b128", bd.stage[0], 4, com.xo[0], XROW, 0)?
    }
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
    // Step 3: every weight load of the row, in group order.
    let weight_loads = emit_weight_loads(b, c, codec)?;
    // Accumulators start at +0.
    for k in 0..kind.chains() {
        valu(
            b,
            &format!("v_mov_b32_e32 v{}, 0", bd.acc + k),
            &[bd.acc + k],
            &[],
            &[],
        )?
    }
    // Step 4: the X loads retire (global loads return in issue order), then the stores, the barrier.
    b.wait(Counter::Vm, weight_loads)?;
    if gate {
        ds_st(b, slot, com.xo[0], bd.stage[0], 0)?;
        ds_st(b, slot, com.xo[0], bd.stage[1], 4096)?;
    }
    op(b, format!("s_mov_b32 exec_lo, s{EMASK}"), &[], &[s(EMASK)])?;
    ds_st(
        b,
        slot,
        com.xo[0],
        if gate { bd.stage[2] } else { bd.stage[0] },
        if gate { 8192 } else { 0 },
    )?;
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
    b.barrier(&[Transition::Ready(slot)])?;
    // Step 5: groups ascending, X of the next group read before this group is computed.
    let groups = match codec {
        Codec::W3 => kind.w3_groups(),
        Codec::Mq4 => kind.mq4_groups(),
    };
    xload(b, c, codec, slot, 0)?;
    for g in 0..groups {
        if g + 1 < groups {
            xload(b, c, codec, slot, g + 1)?
        }
        match codec {
            Codec::W3 => w3_group(b, c, g)?,
            Codec::Mq4 => mq4_group(b, c, g)?,
        }
    }
    epilogue(b, c, codec)?;
    if codec == Codec::W3 {
        // The next-emitted (MQ4) body stores to the same slot: retire it (see the module doc).
        b.barrier(&[Transition::Retire(slot)])?;
    }
    op(b, "s_endpgm", &[], &[])
}

/// Every weight load of the row in group order; returns their count (the explicit `vmcnt` of step 4).
///  * W3: one `global_load_b64` packet window per group (+ the down-only scale `global_load_u16`);
///  * MQ4 gate: header dword at `+136g + 4*(l >= 16)`, payload dword at `+136g + 8 + 4l`;
///  * MQ4 down: header dword at `+68g` (same address in every lane), payload u16 at `+68g + 4 + 2l`.
fn emit_weight_loads(b: &mut Builder, c: &Ctx, codec: Codec) -> Result<u8, String> {
    let (kind, com, bd) = (c.kind, &c.com, c.body(codec));
    let gate = kind == Kind::GateUp;
    let mut n = 0u8;
    match codec {
        Codec::W3 => {
            for g in 0..kind.w3_groups() {
                gload(
                    b,
                    "global_load_b64",
                    bd.a[g as usize],
                    2,
                    com.w3_off,
                    ROWPTR,
                    48 * g,
                )?;
                n += 1;
                if !gate {
                    gload(
                        b,
                        "global_load_u16",
                        bd.b[g as usize],
                        1,
                        com.zero,
                        SPTR,
                        2 * g,
                    )?;
                    n += 1;
                }
            }
        }
        Codec::Mq4 => {
            for g in 0..kind.mq4_groups() {
                if gate {
                    gload(
                        b,
                        "global_load_b32",
                        bd.a[g as usize],
                        1,
                        com.mq_hoff,
                        ROWPTR,
                        136 * g,
                    )?;
                    gload(
                        b,
                        "global_load_b32",
                        bd.b[g as usize],
                        1,
                        com.mq_poff,
                        ROWPTR,
                        136 * g + 8,
                    )?;
                } else {
                    gload(
                        b,
                        "global_load_b32",
                        bd.a[g as usize],
                        1,
                        com.zero,
                        ROWPTR,
                        68 * g,
                    )?;
                    gload(
                        b,
                        "global_load_u16",
                        bd.b[g as usize],
                        1,
                        com.mq_poff,
                        ROWPTR,
                        68 * g + 4,
                    )?;
                }
                n += 2;
            }
        }
    }
    Ok(n)
}

/// The LDS read of group `g`'s X (`ds_load_b128` at `4*(g*128 + 4l)`; MQ4 gate two at `4*(g*256 + 8l)` and `+16`) into
/// the ping-pong buffer `g & 1`.
fn xload(b: &mut Builder, c: &Ctx, codec: Codec, slot: usize, g: u32) -> R {
    let (kind, com, bd) = (c.kind, &c.com, c.body(codec));
    let buf = bd.xb[(g & 1) as usize];
    if codec == Codec::Mq4 && kind == Kind::GateUp {
        ds_ld(b, slot, buf, com.mq_ldsq, 1024 * g)?;
        ds_ld(b, slot, buf + 4, com.mq_ldsq, 1024 * g + 16)
    } else {
        ds_ld(b, slot, buf, com.ldsq, 512 * g)
    }
}

/// One W3 group: scale, packet, the four dequantized weights (stage-major: the four chains are independent), the
/// partial and the chain add. K1's arithmetic exactly (see the module doc).
fn w3_group(b: &mut Builder, c: &Ctx, g: u32) -> R {
    let (kind, com, bd) = (c.kind, &c.com, &c.w3);
    let gi = g as usize;
    if kind == Kind::GateUp {
        // Group g's f16 is the low (even g) or high (odd g) half of SGPR g>>1; cvt reads the low 16 bits.
        let src = SC + (g >> 1) as u8;
        let from = if g & 1 == 1 {
            let t = STMP + ((g >> 1) & 1) as u8;
            sop(b, format!("s_lshr_b32 s{t}, s{src}, 16"), &[t], &[src])?;
            t
        } else {
            src
        };
        valu(
            b,
            &format!("v_mov_b32_e32 v{}, s{from}", bd.d),
            &[bd.d],
            &[],
            &[from],
        )?;
        valu(
            b,
            &format!("v_cvt_f32_f16_e64 v{0}, v{0}.l", bd.d),
            &[bd.d],
            &[bd.d],
            &[],
        )?;
    } else {
        valu(
            b,
            &format!("v_cvt_f32_f16_e64 v{}, v{}.l", bd.d, bd.b[gi]),
            &[bd.d],
            &[bd.b[gi]],
            &[],
        )?;
    }
    // packet = bits [12l, 12l + 64) of the lane's 8-byte window, low dword: `v_lshrrev_b64 pk:pk+1, shift, hi:lo` (the funnel
    // shift of the window; bits [0, 12) of the result are the lane's twelve code bits)
    let (pk, window) = (vr(bd.pk, 2), vr(bd.a[gi], 2));
    b.push(Instruction::new(
        format!("v_lshrrev_b64 {pk}, v{}, {window}", com.w3_sh),
        vec![pk],
        vec![v(com.w3_sh), window],
    ))?;
    for i in 0..4u8 {
        valu(
            b,
            &format!(
                "v_bfe_u32 v{}, v{}, {}, 3",
                bd.w + i,
                bd.pk,
                u32::from(i) * 3
            ),
            &[bd.w + i],
            &[bd.pk],
            &[],
        )?
    }
    for i in 0..4u8 {
        let w = bd.w + i;
        valu(
            b,
            &format!("v_lshl_add_u32 v{w}, v{w}, 1, -7"),
            &[w],
            &[w],
            &[],
        )?
    }
    for i in 0..4u8 {
        let w = bd.w + i;
        valu(b, &format!("v_cvt_f32_i32_e32 v{w}, v{w}"), &[w], &[w], &[])?
    }
    for i in 0..4u8 {
        let w = bd.w + i;
        valu(
            b,
            &format!("v_mul_f32_e32 v{w}, v{}, v{w}", bd.d),
            &[w],
            &[bd.d, w],
            &[],
        )?
    }
    let x = bd.xb[(g & 1) as usize];
    valu(
        b,
        &format!("v_mul_f32_e32 v{}, v{}, v{x}", bd.dot, bd.w),
        &[bd.dot],
        &[bd.w, x],
        &[],
    )?;
    for j in 1..4u8 {
        valu(
            b,
            &format!("v_fmac_f32_e32 v{}, v{}, v{}", bd.dot, bd.w + j, x + j),
            &[bd.dot],
            &[bd.dot, bd.w + j, x + j],
            &[],
        )?
    }
    let acc = bd.acc
        + if kind == Kind::GateUp {
            (g & 3) as u8
        } else {
            0
        };
    valu(
        b,
        &format!("v_add_f32_e32 v{acc}, v{acc}, v{}", bd.dot),
        &[acc],
        &[acc, bd.dot],
        &[],
    )
}

/// One MQ4 group: `d`, `z` from the header halves, `nw` weights `fma(d, q, z)`, the partial in the production
/// contraction order (`p = w1*x1`, `fma(w0, x0, p)`, `fma(w_j, x_j, p)`), the chain add.
fn mq4_group(b: &mut Builder, c: &Ctx, g: u32) -> R {
    let (kind, bd) = (c.kind, &c.mq4);
    let gi = g as usize;
    let (hdr, pay) = (bd.a[gi], bd.b[gi]);
    valu(
        b,
        &format!("v_cvt_f32_f16_e64 v{}, v{hdr}.l", bd.d),
        &[bd.d],
        &[hdr],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshrrev_b32_e32 v{}, 16, v{hdr}", bd.z),
        &[bd.z],
        &[hdr],
        &[],
    )?;
    valu(
        b,
        &format!("v_cvt_f32_f16_e64 v{0}, v{0}.l", bd.z),
        &[bd.z],
        &[bd.z],
        &[],
    )?;
    for i in 0..bd.nw {
        valu(
            b,
            &format!("v_bfe_u32 v{}, v{pay}, {}, 4", bd.w + i, u32::from(i) * 4),
            &[bd.w + i],
            &[pay],
            &[],
        )?
    }
    for i in 0..bd.nw {
        let w = bd.w + i;
        valu(b, &format!("v_cvt_f32_u32_e32 v{w}, v{w}"), &[w], &[w], &[])?
    }
    for i in 0..bd.nw {
        let w = bd.w + i;
        valu(
            b,
            &format!("v_fma_f32 v{w}, v{}, v{w}, v{}", bd.d, bd.z),
            &[w],
            &[bd.d, w, bd.z],
            &[],
        )?
    }
    let x = bd.xb[(g & 1) as usize];
    valu(
        b,
        &format!("v_mul_f32_e32 v{}, v{}, v{}", bd.dot, bd.w + 1, x + 1),
        &[bd.dot],
        &[bd.w + 1, x + 1],
        &[],
    )?;
    valu(
        b,
        &format!("v_fmac_f32_e32 v{}, v{}, v{x}", bd.dot, bd.w),
        &[bd.dot],
        &[bd.dot, bd.w, x],
        &[],
    )?;
    for j in 2..bd.nw {
        valu(
            b,
            &format!("v_fmac_f32_e32 v{}, v{}, v{}", bd.dot, bd.w + j, x + j),
            &[bd.dot],
            &[bd.dot, bd.w + j, x + j],
            &[],
        )?
    }
    let acc = bd.acc
        + if kind == Kind::GateUp {
            (g & 3) as u8
        } else {
            0
        };
    valu(
        b,
        &format!("v_add_f32_e32 v{acc}, v{acc}, v{}", bd.dot),
        &[acc],
        &[acc, bd.dot],
        &[],
    )
}

/// The gate fold, the 5-step snapshot shuffle-down and lane 0's store.
fn epilogue(b: &mut Builder, c: &Ctx, codec: Codec) -> R {
    let (kind, com, bd) = (c.kind, &c.com, c.body(codec));
    let a0 = bd.acc;
    if kind == Kind::GateUp {
        let (a1, a2, a3) = (bd.acc + 1, bd.acc + 2, bd.acc + 3);
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
        let (m, sh) = (SHM + k, com.shuf + k);
        b.ds_crosslane(
            Instruction::new(
                format!("ds_bpermute_b32 v{}, v{sh}, v{a0}", com.exch),
                vec![v(com.exch)],
                vec![v(sh), v(a0)],
            )
            .memory(MemoryClass::DsLoad),
        )?;
        // sum = acc + acc[lane + off]; lanes with lane + off > 31 (mask clear) keep acc unchanged.
        valu(
            b,
            &format!("v_add_f32_e32 v{}, v{a0}, v{}", com.exch, com.exch),
            &[com.exch],
            &[a0, com.exch],
            &[],
        )?;
        valu(
            b,
            &format!("v_cndmask_b32_e64 v{a0}, v{a0}, v{}, s{m}", com.exch),
            &[a0],
            &[a0, com.exch],
            &[m],
        )?;
    }
    valu(
        b,
        &format!("v_cmpx_eq_u32_e32 0, v{}", com.lane),
        &[],
        &[com.lane],
        &[],
    )?;
    b.push(
        Instruction::new(
            format!("global_store_b32 v{}, v{a0}, s[{YP}:{}]", com.zero, YP + 1),
            vec![],
            vec![v(com.zero), v(a0), sr(YP, 2)],
        )
        .memory(MemoryClass::VmemStore),
    )
}

/// The shape contract of one emitted entry: counts of its characteristic instructions, the forbidden families,
/// the honest register ceilings, wave32, no spills, no private memory, and the fixed LDS. Committed as
/// `kernels/mq_i4_moe_gemv_c1.gfx1151.<tag>.contract.json`.
pub fn contract(spec: Spec, e: &Emitted) -> serde_json::Value {
    let mut counts = serde_json::Map::new();
    for name in COUNTED {
        let n = e
            .s_text
            .lines()
            .filter(|l| l.split_whitespace().next() == Some(name))
            .count();
        if n > 0 {
            counts.insert(name.into(), n.into());
        }
    }
    let forbidden = [
        "scratch_*",
        "flat_*",
        "v_wmma*",
        "v_swmmac*",
        "v_fma_mix*",
        "v_dot*",
        "v_pk_*",
        "v_writelane*",
        "s_wait_loadcnt*",
        "s_wait_storecnt*",
        "s_wait_dscnt*",
        "s_wait_kmcnt*",
        "s_wait_alu*",
        "s_setpc*",
        "s_swappc*",
    ];
    serde_json::json!({
        "symbol": spec.symbol(), "counts": counts,
        "forbidden": forbidden,
        "vgpr_max": e.shape.next_free_vgpr, "sgpr_max": e.shape.next_free_sgpr,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0, "group_segment_fixed_bytes": spec.kind.lds_bytes(),
    })
}
const COUNTED: [&str; 38] = [
    "s_load_b256",
    "s_load_b64",
    "s_load_b32",
    "global_load_b128",
    "global_load_b64",
    "global_load_b32",
    "global_load_u16",
    "global_store_b32",
    "ds_store_b128",
    "ds_load_b128",
    "ds_bpermute_b32",
    "s_barrier",
    "v_readfirstlane_b32",
    "v_lshrrev_b64",
    "v_lshl_add_u32",
    "v_cvt_f32_f16_e64",
    "v_cvt_f32_i32_e32",
    "v_cvt_f32_u32_e32",
    "v_bfe_u32",
    "v_mul_f32_e32",
    "v_fma_f32",
    "v_fmac_f32_e32",
    "v_add_f32_e32",
    "v_cmpx_eq_u32_e32",
    "v_cndmask_b32_e64",
    "v_add_lshl_u32",
    "s_cbranch_scc0",
    "s_cbranch_scc1",
    "s_branch",
    "s_cselect_b64",
    "s_cselect_b32",
    "s_endpgm",
    "s_lshr_b32",
    "v_mov_b32_e32",
    "v_lshrrev_b32_e32",
    "v_lshlrev_b32_e32",
    "v_mul_u32_u24_e32",
    "v_cmp_gt_u32_e64",
];

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
fn off(imm: u32) -> String {
    if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    }
}
/// `mnemonic v[dst..dst+width), v{voff}, s[base:base+1]{ offset}` global load.
fn gload(b: &mut Builder, mnemonic: &str, dst: u8, width: u8, voff: u8, base: u8, imm: u32) -> R {
    let data = vreg(dst, width);
    b.push(
        Instruction::new(
            format!(
                "{mnemonic} {data}, v{voff}, s[{base}:{}]{}",
                base + 1,
                off(imm)
            ),
            vec![data],
            vec![vreg(voff, 1), sreg(base, 2)],
        )
        .memory(MemoryClass::VmemLoad),
    )
}
fn ds_st(b: &mut Builder, slot: usize, addr: u8, src: u8, imm: u32) -> R {
    let data = vr(src, 4);
    b.ds_store(
        slot,
        Instruction::new(
            format!("ds_store_b128 v{addr}, {data}{}", off(imm)),
            vec![],
            vec![v(addr), data],
        )
        .memory(MemoryClass::DsStore),
    )
}
fn ds_ld(b: &mut Builder, slot: usize, dst: u8, addr: u8, imm: u32) -> R {
    let data = vr(dst, 4);
    b.ds_load(
        slot,
        Instruction::new(
            format!("ds_load_b128 {data}, v{addr}{}", off(imm)),
            vec![data],
            vec![v(addr)],
        )
        .memory(MemoryClass::DsLoad),
    )
}
fn branch(b: &mut Builder, text: &str) -> R {
    b.control(Instruction::new(text, vec![], vec![]))
}
