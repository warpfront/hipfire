// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! K1-Cn: the gfx1151 (Halo) multi-lane (2..8 decode lanes, MTP verify) slab decode GEMV with the MQ4 `rows8` schedule and
//! K1's exact arithmetic. Four symbols `mq_i4_moe_gemv_cn_{mq3i4,mq4sym}_{r256_gate_up,r128_down}` in the module
//! `mq_i4_moe_gemv_cn_pm_gfx1151`: a drop-in replacement, on gfx1151 and for `2 <= lanes <= 8`, of the four K1 symbols of
//! `mq_i4_moe_gemv.rs` (module `mq_i4_moe_gemv_pm_gfx1151`) with the SAME 96-byte kernarg ABI, the SAME per-bucket launches
//! over the SAME union plan and the SAME outputs bit for bit. Only the schedule changes, exactly as in K1-C1: 8 output rows
//! per 256-thread workgroup, the expert's slots' X staged in LDS once per workgroup, every weight load of a row issued before
//! the first wait, each weight group dequantized once and then applied to every live slot.
//!
//! # Kernarg ABI (96 bytes, K1's, frozen)
//! `expert_weight_ptrs@0` (u64 table indexed by expert id; each entry the projection tensor's code-plane base of that expert,
//! first group for MQ4), `expert_ids@8` (`u32[U]`), `expert_offsets@16` (`u32[U+1]`), `slot_ids@24` (`u32[10*C]`, original
//! slots `lane*10+rank`), `Xrot@32`, `Y0@40`, `Y1@48`, u32 `M@56`, `K@60`, `U@64`, `C@68`, `x_row_div@72`, `x_src_rows@76`,
//! `x_row_stride@80`, `y_slot_stride@84` (strides in F32 elements), u32 `n_experts@88` (the extent of the `expert_weight_ptrs`
//! table) and a zero u32 `reserved@92` (pads the block to 8 bytes; never read). One codec per symbol, as K1.
//!
//! # Geometry
//! Grid `[M/8, U, 1]`, block `[256, 1, 1]` (eight wave32), fixed LDS (`group_segment_fixed_size`): gate 40,960 B = 8 slots x
//! 1,280 F32 (one column phase of every live slot), down 20,480 B = 8 x 640 F32. Workgroup `(wg_x, u = wg_y)`: wave `w`
//! (scalar `v_readfirstlane_b32` of `tid >> 5`) owns output row `row = wg_x*8 + w` (`M % 8 == 0` is host-validated, so no row exits
//! early) and the expert's slot `w` as its X staging slot. The whole workgroup exits (before any barrier) when `u >= U`,
//! `wg_x*8 >= M`, `K` differs from the symbol's K, `n = offsets[u+1] - offsets[u]` is 0 or exceeds 8, `offsets[u+1] > 10*C`, or
//! `expert_ids[u] >= n_experts` (UNSIGNED compare, so a negative id too; tested on the loaded id right after the offsets
//! guards and before it is shifted, so the weight-pointer table is never indexed by an out-of-range id, no expert tensor,
//! slot id or X row is read and no store is issued). The expert id is a single scalar per workgroup (one expert per workgroup).
//! The expert's `n` original slot ids `slot_ids[offsets[u] .. +n]` are loaded with scalar loads
//! (clamped to `n - 1`, so no load leaves the expert's slots), slot `s` reads the X row `s / x_row_div` (K1's semantics: a row
//! at or past `x_src_rows` reads 0 and never touches memory; here the wave stages zeros instead of loading) and lane 0 of each wave stores Y at `Y[s*y_slot_stride + row-in-Y]` (gate `row < M/2` goes
//! to `Y0`, else `Y1[... + row - M/2]`; down `Y0`). No route weight, no atomics, no other writes.
//!
//! # Schedule (gate, two column phases; down is the first phase alone)
//! 1. Scalar prologue: kernarg loads, guards, the offsets pair and expert id, `n`, the expert base, the nine slot-id loads
//!    (the eight slots and the wave's own staging slot), the wave's X row by the f32-reciprocal quotient of K1's gather
//!    (exact below 2^22), the staging masks, Y pointers, the lane constants.
//! 2. Phase 0 X loads (columns `[0, 1280)`; down all 640): wave `w < n` stages slot `w`, 10 (down 5) `global_load_b128` of
//!    the wave's slot: `float4 = lane + 32 j`. Waves `w >= n` run the same loads under `exec = 0`, so every wave issues the
//!    same load instructions; a wave whose X row is at or past `x_src_rows` zero-fills the staging registers (scalar-branch
//!    skipped when no wave needs it) and loads nothing. X and weight loads are all `global_load*`, one in-order `vmcnt` family.
//! 3. EVERY weight load of the wave's row, in group order (W3 gate: 20 `global_load_b64` packets + the 40 scale bytes by
//!    `s_load_b256` + `s_load_b64`; W3 down: 5 packets + 5 `global_load_u16` scales; MQ4: K1's header/payload loads).
//! 4. `s_waitcnt vmcnt(<weight loads>)` (loads return in order, so this waits for the X loads only), the `ds_store_b128` of
//!    the staged rows (LDS slot `i` holds slot `i`'s columns of the phase, slot-major), `s_barrier`.
//! 5. Compute: groups ascending; per group the weights are dequantized ONCE, then slots `i = 0..n-1` (scalar `n` gates, never
//!    re-reading weights) read their X from LDS (`ds_load_b128`) and run their partial into `acc[i][chain]`.
//!    The builder inserts `vmcnt(<loads issued after this group>)` before the first use of each group's weights.
//! 6. Gate only: the phase-1 X loads are issued right after the last phase-0 group (the staging registers alias the compute
//!    temporaries, free again at that point), then the retire `s_barrier` (every wave is done with phase-0 LDS; the load latency
//!    hides behind it), `vmcnt(0)`, the stores, the publish `s_barrier`, and groups 10..19 (MQ4 5..9) on the same
//!    accumulators.
//! 7. Per live slot the gate fold, the 5-step bpermute reduction; lane 0 stores each live slot's Y; `s_endpgm`.
//!
//! # Exactness law (K1's, per (slot, row))
//!  * lane `l` owns `g*128 + 4l ..+3` (W3 both kinds, MQ4 down) or `g*256 + 8l ..+7` (MQ4 gate);
//!  * W3: `d = cvt_f32_f16(scale)`; per code `i` `c = bfe(pkt, 3i, 3)`, `t = 2c - 7` (one `v_lshl_add_u32 t, c, 1, -7`, the same
//!    integer as K1's `v_lshlrev_b32` + `v_add_nc_u32 -7`), `w_i = v_mul_f32(d, cvt_f32_i32(t))`; `p = v_mul_f32(w0, x0)`
//!    then `v_fmac_f32(p, w_j, x_j)`, `j = 1..3`;
//!  * MQ4: `d = cvt(h.lo16)`, `z = cvt(h.hi16)`, `w_j = v_fma_f32(d, cvt_f32_u32(bfe(pkt, 4j, 4)), z)`, `p = v_mul_f32(w1, x1)`,
//!    `v_fmac(w0, x0)`, then `v_fmac(w_j, x_j)`, `j = 2..nw-1`;
//!  * accumulators start at `+0`, `acc[c] = v_add_f32(acc[c], p)` (gate chain `c = g & 3`, down chain 0), groups ascending;
//!    gate fold `a0 = a0 + a1; a2 = a2 + a3; a0 = a0 + a2`; snapshot shuffle-down at 16, 8, 4, 2, 1 with `ds_bpermute_b32`,
//!    `exch = v_add_f32(a0, exch)`, `a0 = v_cndmask(a0, exch, mask)`; lane 0 stores. No packed math, dot instruction,
//!    reassociation or other rounding touches the value path.
//!
//! # W3 b64 packet and its over-read
//! As K1-C1: ONE `global_load_b64` at group byte `4*((12 l) >> 5)` and `v_lshrrev_b64 pkt:pkt+1, (12 l) & 31, hi:lo` (bits
//! `[0, 12)` of the low dword are group bits `[12 l, 12 l + 12)`, exactly K1's twelve code bits). The last lane's window extends
//! four bytes past its 48-byte group (next group, next row, or for the last row the scale plane that follows the code plane in the
//! planar tensor), always inside the expert tensor, its bytes never used. Gate scales (K=2560, G=20): the row's 40 bytes at
//! `base + M*R + 40*row` (`R = K*3/8`, 8-byte aligned) by `s_load_b256` + `s_load_b64`; this REQUIRES the scale plane base
//! to be 4-byte aligned. Down (G=5): one `global_load_u16` per group, same address in every lane.
//!
//! # LDS and the barriers
//! One declared slot `x`. Phase stores publish it (`Ready`), the gate's retire barrier frees it (`Retire`) for the second
//! phase's stores, which publish it again; the down projection has the single publish barrier. Slot-state is tracked by the
//! builder linearly; every X read of a slot gate is closed (explicit `lgkmcnt(0)`) before the gated compute, so skipped slots
//! never leave the builder's wait ledger out of step with the executed stream.
//!
//! # Fixed eight-row unions
//! `emit_rowtiles` adds `_t8` entries without changing `emit` or its narrow ABI/bytes.
//! Launch grid `[ceil(M/8), 80, ceil(C/8)]`, block256, with `1 <= C <= 64` and `U = 80`.
//! Workgroup z adds `z * 1928` bytes to each of expert_ids, expert_offsets and slot_ids only.
//! The descriptor's slots are absolute (`slot < 10*C`), so X/Y keep their full-window bases.
//! Both offsets must be at most80, the expert must be in range and its count must be1..8;
//! every loaded slot is checked uniformly before any barrier or X access. A dead tile/group
//! exits uniformly. A tail output wave still stages its slot's X and reaches every barrier,
//! but skips row weights, dot arithmetic and output stores. The register bank stays at8 slots
//! and the gate four-chain/down one-chain DAG, operand orientation and M/2 gate split are unchanged.

use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{
    add64, gather_offset, lit, op, s, s_add_i32, smem, sop, sr, v, vr, GatherTemps,
};
use crate::lds::Transition;
use crate::ledger::Counter;
use crate::plan::Access;
use crate::reg::{Kind as RKind, Live, RegRef};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// The kernarg segment size (K1's).
pub const KERNARG_BYTES: u32 = 96;
/// Workgroup size (threads): eight wave32.
pub const WORKGROUP_SIZE: u32 = 256;
/// Output rows (waves) of one workgroup.
pub const ROWS_PER_WORKGROUP: u32 = 8;
/// Most distinct original slots one expert workgroup serves.
pub const MAX_SLOTS: u32 = 8;
/// The module name.
pub const MODULE: &str = "mq_i4_moe_gemv_cn_pm_gfx1151";
/// The four exported symbols in module order (Mq3I4 gate, Mq3I4 down, Mq4Sym gate, Mq4Sym down).
pub const SYMBOLS: [&str; 4] = [
    "mq_i4_moe_gemv_cn_mq3i4_r256_gate_up",
    "mq_i4_moe_gemv_cn_mq3i4_r128_down",
    "mq_i4_moe_gemv_cn_mq4sym_r256_gate_up",
    "mq_i4_moe_gemv_cn_mq4sym_r128_down",
];

/// Weight codec of the tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codec {
    /// Planar 3-bit codes, `2c - 7`, one f16 scale per row-group.
    Mq3I4,
    /// Pointer-addressed MQ4 (gate group 136 B, down group 68 B), `d*q + z`.
    Mq4Sym,
}
impl Codec {
    pub const ALL: [Codec; 2] = [Codec::Mq3I4, Codec::Mq4Sym];
    pub fn tag(self) -> &'static str {
        match self {
            Self::Mq3I4 => "mq3i4",
            Self::Mq4Sym => "mq4sym",
        }
    }
}

/// Projection kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// Gate/up (`K2560`): four group chains, fold `(a0+a1)+(a2+a3)`, Y0/Y1 split at `M/2`, two column phases.
    GateUp,
    /// Down (`K640`): one chain, Y0 only, one column phase.
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
    /// Column phases (the LDS holds one phase of every slot).
    pub fn phases(self) -> u32 {
        match self {
            Self::GateUp => 2,
            Self::Down => 1,
        }
    }
    /// F32 columns per slot per phase.
    pub fn phase_cols(self) -> u32 {
        self.k() / self.phases()
    }
    /// Bytes of one slot's LDS row (one phase).
    pub fn slot_bytes(self) -> u32 {
        self.phase_cols() * 4
    }
    /// Fixed LDS bytes: eight slots of one phase.
    pub fn lds_bytes(self) -> u32 {
        MAX_SLOTS * self.slot_bytes()
    }
    /// Accumulator chains per slot.
    pub fn chains(self) -> u8 {
        match self {
            Self::GateUp => 4,
            Self::Down => 1,
        }
    }
    /// `global_load_b128` X staging loads per wave per phase (`float4 = lane + 32 j`).
    pub fn x_loads(self) -> u8 {
        (self.phase_cols() / 4 / 32) as u8
    }
}

/// One entry: gfx1151 only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
    pub codec: Codec,
    pub kind: Kind,
}
impl Spec {
    pub fn new(codec: Codec, kind: Kind) -> Self {
        Self {
            arch: Arch::Gfx1151,
            codec,
            kind,
        }
    }
    /// The exported symbol, `mq_i4_moe_gemv_cn_<codec>_<kind>`.
    pub fn symbol(&self) -> String {
        format!("mq_i4_moe_gemv_cn_{}", self.tag())
    }
    /// `<codec>_<kind>`: the `--epi` name, the label prefix and the contract file tag.
    pub fn tag(&self) -> String {
        format!("{}_{}", self.codec.tag(), self.kind.tag())
    }
    pub fn module() -> String {
        MODULE.into()
    }
    pub fn validate(&self) -> R {
        if self.arch != Arch::Gfx1151 {
            return Err("mq_i4_moe_gemv_cn builder is exact-gfx1151 only".into());
        }
        Ok(())
    }
    /// Weights (and X elements) one lane owns per group: eight for the `Mq4Sym` gate G256 group, four otherwise.
    pub fn lane_elems(&self) -> u8 {
        if self.codec == Codec::Mq4Sym && self.kind == Kind::GateUp {
            8
        } else {
            4
        }
    }
    /// Groups of the whole row.
    pub fn groups(&self) -> u32 {
        match (self.codec, self.kind) {
            (Codec::Mq3I4, Kind::GateUp) => 20,
            (Codec::Mq4Sym, Kind::GateUp) => 10,
            (_, Kind::Down) => 5,
        }
    }
    /// Groups of one column phase.
    pub fn groups_per_phase(&self) -> u32 {
        self.groups() / self.kind.phases()
    }
    /// Slots whose X is read together (register window): all eight, four for the wide MQ4 gate group.
    pub fn batch(&self) -> u32 {
        if self.lane_elems() == 8 {
            4
        } else {
            8
        }
    }
}

/// Every entry in module order.
pub fn module_specs() -> Vec<Spec> {
    Codec::ALL
        .into_iter()
        .flat_map(|codec| {
            Kind::ALL
                .into_iter()
                .map(move |kind| Spec::new(codec, kind))
        })
        .collect()
}

// ----------------------------------------------------------------------- SGPRs
// s0:1 kernarg pointer; s2, s3 workgroup ids x, y (gfx11 system SGPRs after the two user SGPRs).
const KARG: u8 = 0;
const WGX: u8 = 2;
const WGY: u8 = 3;
const WGZ: u8 = 4;
/// Kernargs: `s_load_b256 s[8:15]` (@0), `s_load_b256 s[16:23]` (@0x20), `s_load_b128 s[24:27]` (@0x40), `s_load_b64 s[28:29]` (@0x50), `s_load_b32 s100` (@0x58).
const WPTRS: u8 = 8;
const EIDS: u8 = 10;
const OFFS: u8 = 12;
const SLOTS: u8 = 14;
const XP: u8 = 16;
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
const OFF0: u8 = 32;
const EID: u8 = 34;
const NS: u8 = 35;
const TPTR: u8 = 36;
const BASE: u8 = 38;
const WAVE: u8 = 40;
const TMP: u8 = 41;
const TMP2: u8 = 42;
const NSM1: u8 = 43;
/// Tiled entry only: whether this wave owns a valid output row.
const OUTPUT_LIVE: u8 = 44;
const ROWPTR: u8 = 48;
const SPTR: u8 = 50;
const YSEL: u8 = 52;
const ROWIN4: u8 = 54;
const SXS4: u8 = 55;
const SYS4: u8 = 56;
const WSID: u8 = 57;
/// The eight slot ids (`s58..s65`).
const SID: u8 = 58;
const XR: u8 = 66;
const XOFFW: u8 = 67;
const EMASK: u8 = 68;
const MI: u8 = 69;
const ROWM: u8 = 70;
const ROWIN: u8 = 71;
/// Five shuffle masks (one SGPR each, wave32).
const SHM: u8 = 72;
/// Staging masks: waves that stage their slot (`EMASK`), of those the ones whose X row exists (`LMASK`, loads), the rest (`ZMASK`, zero fill).
const LMASK: u8 = 77;
const ZMASK: u8 = 82;
const LIVE: u8 = 78;
const MASK: u8 = 80;
// W3 gate scales: ten scale dwords and two shift temporaries.
const SCW: u8 = 88;
const SC8: u8 = 96;
const STMP: u8 = 98;
/// `n_experts@88`: dedicated SGPR, aliased with nothing (above every other scalar window).
const SNEXP: u8 = 100;
/// SGPRs the plan declares (the honest ceiling, `sgpr_max` of the contract).
const SGPRS: u8 = 101;

// ----------------------------------------------------------------------- VGPRs
/// The register layout of one entry.
///
/// Persistent: the lane constants, the accumulators (`8 * chains`) and every weight-load register. The rest is one
/// "window" at `win`, reused by lifetime: prologue temporaries, the X staging registers of a phase, the compute registers
/// of a phase, the epilogue registers.
#[derive(Clone, Debug)]
struct Lay {
    lane: u8,
    ldsst: u8,
    vxo: u8,
    vxo2: u8,
    zero: u8,
    ldsq: u8,
    w3_off: u8,
    w3_sh: u8,
    poff: u8,
    hoff: u8,
    acc: u8,
    /// Per-group first load: W3 packet pair, MQ4 header.
    a: Vec<u8>,
    /// Per-group second load: down-W3 scale u16, MQ4 payload; empty for gate W3 (its scales are SGPRs).
    b: Vec<u8>,
    /// X staging `float4` registers of a phase.
    stage: Vec<u8>,
    /// Compute window: weights, scale, zero point, packet, two dot registers, the X registers of a batch of slots.
    w: u8,
    d: u8,
    z: u8,
    pk: u8,
    dot: [u8; 2],
    xb: Vec<u8>,
    nw: u8,
    /// Epilogue overlay: shuffle addresses, exchange, the eight store offsets.
    shuf: u8,
    exch: u8,
    vy: u8,
    /// Prologue overlay: quotient scratch (index, result, four temporaries).
    t_idx: u8,
    t_dst: u8,
    t_q: u8,
    t_r: u8,
    t_w: u8,
    t_f: u8,
    end: u8,
}
impl Lay {
    fn new(spec: Spec) -> Self {
        let (gate, mq4) = (spec.kind == Kind::GateUp, spec.codec == Codec::Mq4Sym);
        let nw = spec.lane_elems();
        let mut n = 0u8;
        let mut take = |len: u8, align: u8| {
            n = n.div_ceil(align) * align;
            let base = n;
            n += len;
            base
        };
        let lane = take(1, 1);
        let ldsst = take(1, 1);
        let vxo = take(1, 1);
        let vxo2 = if gate { take(1, 1) } else { 0 };
        let zero = take(1, 1);
        let ldsq = take(1, 1);
        let (w3_off, w3_sh) = if mq4 {
            (0, 0)
        } else {
            (take(1, 1), take(1, 1))
        };
        let (poff, hoff) = if mq4 {
            (take(1, 1), if gate { take(1, 1) } else { 0 })
        } else {
            (0, 0)
        };
        let acc = take(MAX_SLOTS as u8 * spec.kind.chains(), 1);
        let groups = spec.groups() as usize;
        let (a, b): (Vec<u8>, Vec<u8>) = match spec.codec {
            Codec::Mq3I4 => (
                (0..groups).map(|_| take(2, 2)).collect(),
                if gate {
                    vec![]
                } else {
                    (0..groups).map(|_| take(1, 1)).collect()
                },
            ),
            Codec::Mq4Sym => (
                (0..groups).map(|_| take(1, 1)).collect(),
                (0..groups).map(|_| take(1, 1)).collect(),
            ),
        };
        let win = take(0, 4);
        let stage: Vec<u8> = (0..spec.kind.x_loads()).map(|j| win + 4 * j).collect();
        let stage_end = win + 4 * spec.kind.x_loads();
        n = win;
        let mut take = |len: u8, align: u8| {
            n = n.div_ceil(align) * align;
            let base = n;
            n += len;
            base
        };
        let w = take(nw, 4);
        let d = take(1, 1);
        let z = if mq4 { take(1, 1) } else { 0 };
        let pk = if mq4 { 0 } else { take(2, 2) };
        let dot = [take(1, 1), take(1, 1)];
        let xb: Vec<u8> = (0..spec.batch()).map(|_| take(nw, 4)).collect();
        let compute_end = n;
        // Epilogue overlay: 5 shuffle addresses, the exchange, 8 store offsets.
        let (shuf, exch, vy) = (win, win + 5, win + 6);
        let epi_end = win + 14;
        let (t_idx, t_dst, t_q, t_r, t_w, t_f) = (win, win + 1, win + 2, win + 3, win + 4, win + 5);
        let end = stage_end.max(compute_end).max(epi_end);
        Self {
            lane,
            ldsst,
            vxo,
            vxo2,
            zero,
            ldsq,
            w3_off,
            w3_sh,
            poff,
            hoff,
            acc,
            a,
            b,
            stage,
            w,
            d,
            z,
            pk,
            dot,
            xb,
            nw,
            shuf,
            exch,
            vy,
            t_idx,
            t_dst,
            t_q,
            t_r,
            t_w,
            t_f,
            end,
        }
    }
    /// First accumulator of slot `i` (`chains` consecutive registers).
    fn ac(&self, i: u32, chains: u8) -> u8 {
        self.acc + i as u8 * chains
    }
}

/// Everything one symbol needs.
struct Ctx {
    spec: Spec,
    l: Lay,
    vgprs: u16,
    tiled: bool,
}
impl Ctx {
    fn new(spec: Spec) -> Self {
        let l = Lay::new(spec);
        let vgprs = u16::from(l.end).div_ceil(8) * 8;
        Self {
            spec,
            l,
            vgprs,
            tiled: false,
        }
    }
    fn lbl(&self, name: &str) -> String {
        if self.tiled {
            format!(".Lcn_{}_t8_{name}", self.spec.tag())
        } else {
            format!(".Lcn_{}_{name}", self.spec.tag())
        }
    }
}

fn decl(plan: &mut RegPlan, name: &str, kind: RKind, base: u8, len: u8, live: &Live) -> R {
    plan.add_range(name, kind, base, len, live.clone())
}

fn plan(c: &Ctx) -> Result<RegPlan, String> {
    let (spec, l) = (c.spec, &c.l);
    let (gate, mq4) = (spec.kind == Kind::GateUp, spec.codec == Codec::Mq4Sym);
    let mut regs = RegPlan::new(c.vgprs, u16::from(SGPRS))?;
    let whole = Live::Whole;
    for (name, base) in [("workgroup_id_x", WGX), ("workgroup_id_y", WGY)] {
        decl(&mut regs, name, RKind::S, base, 1, &whole)?
    }
    if c.tiled {
        decl(&mut regs, "workgroup_id_z", RKind::S, WGZ, 1, &whole)?;
        decl(&mut regs, "output_live", RKind::S, OUTPUT_LIVE, 1, &whole)?;
    }
    for (name, base, len) in [
        ("kernarg", KARG, 2),
        ("kernel_arguments_ptrs", WPTRS, 8),
        ("kernel_arguments_x_y_m_k", XP, 8),
        ("kernel_arguments_u_c_div_src", SU, 4),
        ("kernel_arguments_strides", SXSTR, 2),
        ("row", ROW, 1),
        ("offsets_pair", OFF0, 2),
        ("expert_id", EID, 1),
        ("slot_count", NS, 1),
        ("table_pointer", TPTR, 2),
        ("expert_pointer", BASE, 2),
        ("wave", WAVE, 1),
        ("temp", TMP, 1),
        ("temp2", TMP2, 1),
        ("slots_minus_one", NSM1, 1),
        ("row_pointer", ROWPTR, 2),
        ("y_select", YSEL, 2),
        ("row_in_y_bytes", ROWIN4, 1),
        ("x_stride_bytes", SXS4, 1),
        ("y_stride_bytes", SYS4, 1),
        ("wave_slot_id", WSID, 1),
        ("x_row", XR, 1),
        ("x_offset_wave", XOFFW, 1),
        ("exec_mask", EMASK, 1),
        ("load_mask", LMASK, 1),
        ("zero_mask", ZMASK, 1),
        ("half_rows", MI, 1),
        ("row_minus_half", ROWM, 1),
        ("row_in_y", ROWIN, 1),
        ("gather_live", LIVE, 2),
        ("gather_mask", MASK, 2),
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
    for i in 0..MAX_SLOTS as u8 {
        decl(&mut regs, "slot_id", RKind::S, SID + i, 1, &whole)?
    }
    for k in 0..5 {
        decl(&mut regs, "shuffle_mask", RKind::S, SHM + k, 1, &whole)?
    }
    if !mq4 {
        decl(&mut regs, "scale_pointer", RKind::S, SPTR, 2, &whole)?;
        if gate {
            decl(&mut regs, "scale_dwords", RKind::S, SCW, 8, &whole)?;
            decl(&mut regs, "scale_dwords_tail", RKind::S, SC8, 2, &whole)?;
            for k in 0..2 {
                decl(&mut regs, "scale_shift_temp", RKind::S, STMP + k, 1, &whole)?
            }
        }
    }
    // Persistent VGPRs.
    for (name, base) in [
        ("lane", l.lane),
        ("lds_store_address", l.ldsst),
        ("x_offset", l.vxo),
        ("zero", l.zero),
        ("lds_lane_address", l.ldsq),
    ] {
        decl(&mut regs, name, RKind::V, base, 1, &whole)?
    }
    if gate {
        decl(&mut regs, "x_offset2", RKind::V, l.vxo2, 1, &whole)?
    }
    if mq4 {
        decl(&mut regs, "mq4_payload_offset", RKind::V, l.poff, 1, &whole)?;
        if gate {
            decl(&mut regs, "mq4_header_offset", RKind::V, l.hoff, 1, &whole)?
        }
    } else {
        decl(&mut regs, "w3_packet_offset", RKind::V, l.w3_off, 1, &whole)?;
        decl(&mut regs, "w3_packet_shift", RKind::V, l.w3_sh, 1, &whole)?;
    }
    for k in 0..MAX_SLOTS as u8 * spec.kind.chains() {
        decl(&mut regs, "accumulator", RKind::V, l.acc + k, 1, &whole)?
    }
    for (i, &base) in l.a.iter().enumerate() {
        if mq4 {
            decl(
                &mut regs,
                &format!("mq4_header{i}"),
                RKind::V,
                base,
                1,
                &whole,
            )?
        } else {
            decl(
                &mut regs,
                &format!("w3_packet{i}"),
                RKind::V,
                base,
                2,
                &whole,
            )?
        }
    }
    for (i, &base) in l.b.iter().enumerate() {
        decl(
            &mut regs,
            &format!("{}{i}", if mq4 { "mq4_payload" } else { "w3_scale" }),
            RKind::V,
            base,
            1,
            &whole,
        )?
    }
    // Window lifetimes: prologue scratch, then (stage, compute) per column phase, then the epilogue overlay.
    let entry = Live::Between("entry".into(), c.lbl("body"));
    for (name, base) in [
        ("quot_index", l.t_idx),
        ("quot_result", l.t_dst),
        ("quot_q", l.t_q),
        ("quot_r", l.t_r),
        ("quot_w", l.t_w),
        ("quot_f", l.t_f),
    ] {
        decl(&mut regs, name, RKind::V, base, 1, &entry)?
    }
    let mut spans = vec![c.lbl("body"), c.lbl("c0")];
    if gate {
        spans.extend([c.lbl("s1"), c.lbl("c1")])
    }
    spans.push(c.lbl("epi"));
    for phase in 0..spec.kind.phases() as usize {
        let stage = Live::Between(spans[2 * phase].clone(), spans[2 * phase + 1].clone());
        let compute = Live::Between(spans[2 * phase + 1].clone(), spans[2 * phase + 2].clone());
        for (j, &base) in l.stage.iter().enumerate() {
            decl(&mut regs, &format!("x_stage{j}"), RKind::V, base, 4, &stage)?
        }
        decl(&mut regs, "weights", RKind::V, l.w, l.nw, &compute)?;
        decl(&mut regs, "scale", RKind::V, l.d, 1, &compute)?;
        if mq4 {
            decl(&mut regs, "zero_point", RKind::V, l.z, 1, &compute)?
        } else {
            decl(&mut regs, "packet", RKind::V, l.pk, 2, &compute)?
        }
        for k in 0..2 {
            decl(&mut regs, "dot", RKind::V, l.dot[k], 1, &compute)?
        }
        for (k, &base) in l.xb.iter().enumerate() {
            decl(
                &mut regs,
                &format!("x_values{k}"),
                RKind::V,
                base,
                l.nw,
                &compute,
            )?
        }
    }
    let epi = Live::Between(c.lbl("epi"), c.lbl("exit"));
    for k in 0..5 {
        decl(&mut regs, "shuffle_address", RKind::V, l.shuf + k, 1, &epi)?
    }
    decl(&mut regs, "exchange", RKind::V, l.exch, 1, &epi)?;
    for i in 0..MAX_SLOTS as u8 {
        decl(&mut regs, "store_offset", RKind::V, l.vy + i, 1, &epi)?
    }
    Ok(regs)
}

/// Production entry.
pub fn emit(spec: Spec) -> Result<Emitted, String> {
    emit_impl(spec, false)
}

/// Emit the gfx1151 fixed-eight-row descriptor entry (`<narrow symbol>_t8`).
///
/// Uses the same96-byte kernargs, cap8 register bank and exact arithmetic as `emit`.
/// `C` is the full1..64 row count; only descriptor pointers receive the grid-z offset.
pub fn emit_rowtiles(spec: Spec) -> Result<Emitted, String> {
    emit_impl(spec, true)
}

fn emit_impl(spec: Spec, tiled: bool) -> Result<Emitted, String> {
    spec.validate()?;
    let mut c = Ctx::new(spec);
    c.tiled = tiled;
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
    let kspec = KernelSpec {
        kernel_id: "mq_i4_moe_gemv_cn".into(),
        variant: if tiled {
            format!("{}_t8", spec.tag())
        } else {
            spec.tag()
        },
        arch: spec.arch,
        symbol: if tiled {
            format!("{}_t8", spec.symbol())
        } else {
            spec.symbol()
        },
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: WORKGROUP_SIZE as u16,
        group_segment_fixed_size: spec.kind.lds_bytes(),
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan(&c)?);
    if tiled {
        b.enable_workgroup_id_z();
    }
    b.enable_delay_alu();
    let slot = b.lds_slot("x", 0, spec.kind.lds_bytes())?;
    prologue(&mut b, &c)?;
    b.label(&c.lbl("body"))?;
    body(&mut b, &c, slot)?;
    b.label(&c.lbl("exit"))?;
    op(&mut b, "s_endpgm", &[], &[])?;
    b.finish()
}

/// The four entries as one code object.
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

/// Kernarg loads, the lane constants, guards, the expert, the slot ids, the wave's X row, the X descriptor and the Y pointers.
fn prologue(b: &mut Builder, c: &Ctx) -> R {
    let arch = b.spec.arch;
    let (spec, l) = (c.spec, &c.l);
    let (gate, mq4) = (spec.kind == Kind::GateUp, spec.codec == Codec::Mq4Sym);
    let exit = c.lbl("exit");
    smem(b, WPTRS, 8, KARG, 0)?;
    smem(b, XP, 8, KARG, 0x20)?;
    smem(b, SU, 4, KARG, 0x40)?;
    smem(b, SXSTR, 2, KARG, 0x50)?;
    smem(b, SNEXP, 1, KARG, 0x58)?;
    match arch.workgroup_ids(2)? {
        crate::arch::WgIds {
            x: crate::arch::WgIdSource::Sgpr(x),
            y: crate::arch::WgIdSource::Sgpr(y),
        } if (x, y) == (WGX, WGY) => {}
        ids => {
            return Err(format!(
                "mq_i4_moe_gemv_cn: unexpected workgroup id sources {ids:?}"
            ))
        }
    }
    // Wave (scalar) and lane; lane constants (VALU, under the kernarg latency). `lane` lives in v0 (the workitem id) in place.
    valu(
        b,
        &format!("v_lshrrev_b32_e32 v{}, 5, v{}", l.t_idx, l.lane),
        &[l.t_idx],
        &[l.lane],
        &[],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{WAVE}, v{}", l.t_idx),
        &[s(WAVE)],
        &[v(l.t_idx)],
    )?;
    valu(
        b,
        &format!("v_and_b32_e32 v{0}, 31, v{0}", l.lane),
        &[l.lane],
        &[l.lane],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{}, 4, v{}", l.vxo, l.lane),
        &[l.vxo],
        &[l.lane],
        &[],
    )?;
    valu(
        b,
        &format!(
            "v_lshlrev_b32_e32 v{}, {}, v{}",
            l.ldsq,
            if spec.lane_elems() == 8 { 5 } else { 4 },
            l.lane
        ),
        &[l.ldsq],
        &[l.lane],
        &[],
    )?;
    valu(
        b,
        &format!("v_mov_b32_e32 v{}, 0", l.zero),
        &[l.zero],
        &[],
        &[],
    )?;
    if mq4 {
        if gate {
            valu(
                b,
                &format!("v_lshlrev_b32_e32 v{}, 2, v{}", l.poff, l.lane),
                &[l.poff],
                &[l.lane],
                &[],
            )?;
            valu(
                b,
                &format!("v_lshrrev_b32_e32 v{}, 4, v{}", l.hoff, l.lane),
                &[l.hoff],
                &[l.lane],
                &[],
            )?;
            valu(
                b,
                &format!("v_lshlrev_b32_e32 v{0}, 2, v{0}", l.hoff),
                &[l.hoff],
                &[l.hoff],
                &[],
            )?;
        } else {
            valu(
                b,
                &format!("v_lshlrev_b32_e32 v{}, 1, v{}", l.poff, l.lane),
                &[l.poff],
                &[l.lane],
                &[],
            )?;
        }
    } else {
        // W3: packet byte offset 4*((12 l) >> 5) and bit shift (12 l) & 31.
        valu(
            b,
            &format!("v_mul_u32_u24_e32 v{}, 12, v{}", l.t_idx, l.lane),
            &[l.t_idx],
            &[l.lane],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshrrev_b32_e32 v{}, 5, v{}", l.w3_off, l.t_idx),
            &[l.w3_off],
            &[l.t_idx],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{0}, 2, v{0}", l.w3_off),
            &[l.w3_off],
            &[l.w3_off],
            &[],
        )?;
        valu(
            b,
            &format!("v_and_b32_e32 v{}, 31, v{}", l.w3_sh, l.t_idx),
            &[l.w3_sh],
            &[l.t_idx],
            &[],
        )?;
    }
    // LDS store address: wave * slot_bytes + 16 * lane.
    sop(
        b,
        format!("s_mul_i32 s{TMP}, s{WAVE}, {}", lit(spec.kind.slot_bytes())),
        &[TMP],
        &[WAVE],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{}, s{TMP}, v{}", l.ldsst, l.vxo),
        &[l.ldsst],
        &[l.vxo],
        &[TMP],
    )?;
    // Guards (workgroup-uniform, before any barrier).
    if c.tiled {
        sop(b, format!("s_cmp_eq_u32 s{SC}, 0"), &[], &[SC])?;
        branch(b, &format!("s_cbranch_scc1 {exit}"))?;
        sop(b, format!("s_cmp_gt_u32 s{SC}, 64"), &[], &[SC])?;
        branch(b, &format!("s_cbranch_scc1 {exit}"))?;
        sop(b, format!("s_cmp_lg_u32 s{SU}, {}", lit(80)), &[], &[SU])?;
        branch(b, &format!("s_cbranch_scc1 {exit}"))?;
        sop(b, format!("s_sub_i32 s{TMP}, s{SC}, 1"), &[TMP], &[SC])?;
        sop(b, format!("s_lshr_b32 s{TMP}, s{TMP}, 3"), &[TMP], &[TMP])?;
        sop(b, format!("s_cmp_gt_u32 s{WGZ}, s{TMP}"), &[], &[WGZ, TMP])?;
        branch(b, &format!("s_cbranch_scc1 {exit}"))?;
        sop(
            b,
            format!("s_mul_i32 s{TMP}, s{WGZ}, {}", lit(1928)),
            &[TMP],
            &[WGZ],
        )?;
        for ptr in [EIDS, OFFS, SLOTS] {
            add64(b, ptr, ptr, TMP)?;
        }
    }
    sop(b, format!("s_lshl_b32 s{ROW}, s{WGX}, 3"), &[ROW], &[WGX])?;
    sop(b, format!("s_cmp_ge_u32 s{WGY}, s{SU}"), &[], &[WGY, SU])?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    sop(b, format!("s_cmp_ge_u32 s{ROW}, s{SM}"), &[], &[ROW, SM])?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    sop(
        b,
        format!("s_cmp_lg_u32 s{SK}, {}", lit(spec.kind.k())),
        &[],
        &[SK],
    )?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    // offsets[u], offsets[u+1] and the expert id.
    sop(b, format!("s_lshl_b32 s{TMP}, s{WGY}, 2"), &[TMP], &[WGY])?;
    add64(b, TPTR, OFFS, TMP)?;
    smem(b, OFF0, 2, TPTR, 0)?;
    add64(b, TPTR, EIDS, TMP)?;
    smem(b, EID, 1, TPTR, 0)?;
    sop(
        b,
        format!("{} s{ROW}, s{ROW}, s{WAVE}", s_add_i32(arch)),
        &[ROW],
        &[ROW, WAVE],
    )?;
    if c.tiled {
        sop(b, format!("s_cmp_lt_u32 s{ROW}, s{SM}"), &[], &[ROW, SM])?;
        sop(
            b,
            format!("s_cselect_b32 s{OUTPUT_LIVE}, 1, 0"),
            &[OUTPUT_LIVE],
            &[],
        )?;
        for offset in [OFF0, OFF0 + 1] {
            sop(
                b,
                format!("s_cmp_gt_u32 s{offset}, {}", lit(80)),
                &[],
                &[offset],
            )?;
            branch(b, &format!("s_cbranch_scc1 {exit}"))?;
        }
    }
    sop(
        b,
        format!("{} s{NS}, s{}, s{OFF0}", s_sub_i32(), OFF0 + 1),
        &[NS],
        &[OFF0 + 1, OFF0],
    )?;
    sop(b, format!("s_cmp_eq_u32 s{NS}, 0"), &[], &[NS])?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    sop(b, format!("s_cmp_gt_u32 s{NS}, {MAX_SLOTS}"), &[], &[NS])?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    sop(b, format!("s_mul_i32 s{TMP}, s{SC}, 10"), &[TMP], &[SC])?;
    sop(
        b,
        format!("s_cmp_gt_u32 s{}, s{TMP}", OFF0 + 1),
        &[],
        &[OFF0 + 1, TMP],
    )?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    // Expert id range check on the loaded id, before it is shifted or used to index any table: an id at or past `n_experts`
    // (UNSIGNED, so a negative id too) exits the workgroup uniformly (no barrier reached, no load, no store).
    sop(
        b,
        format!("s_cmp_ge_u32 s{EID}, s{SNEXP}"),
        &[],
        &[EID, SNEXP],
    )?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    // base = expert_weight_ptrs[eid].
    if !c.tiled {
        sop(b, format!("s_lshl_b32 s{TMP}, s{EID}, 3"), &[TMP], &[EID])?;
        add64(b, TPTR, WPTRS, TMP)?;
        smem(b, BASE, 2, TPTR, 0)?;
    }
    // The eight slot ids (position clamped to n - 1) and the wave's own staging slot.
    sop(
        b,
        format!("{} s{NSM1}, s{NS}, -1", s_add_i32(arch)),
        &[NSM1],
        &[NS],
    )?;
    for i in 0..MAX_SLOTS as u8 {
        sop(
            b,
            format!("s_min_u32 s{TMP}, s{NSM1}, {i}"),
            &[TMP],
            &[NSM1],
        )?;
        slot_id_load(b, SID + i)?;
    }
    sop(
        b,
        format!("s_min_u32 s{TMP}, s{NSM1}, s{WAVE}"),
        &[TMP],
        &[NSM1, WAVE],
    )?;
    slot_id_load(b, WSID)?;
    if c.tiled {
        sop(b, format!("s_mul_i32 s{TMP2}, s{SC}, 10"), &[TMP2], &[SC])?;
        for i in 0..MAX_SLOTS as u8 {
            sop(
                b,
                format!("s_cmp_ge_u32 s{}, s{TMP2}", SID + i),
                &[],
                &[SID + i, TMP2],
            )?;
            branch(b, &format!("s_cbranch_scc1 {exit}"))?;
        }
        sop(b, format!("s_lshl_b32 s{TMP}, s{EID}, 3"), &[TMP], &[EID])?;
        add64(b, TPTR, WPTRS, TMP)?;
        smem(b, BASE, 2, TPTR, 0)?;
    }
    // The wave's X row = slot / x_row_div (K1's f32-reciprocal quotient), its byte offset and the LDS-staging X offsets.
    valu(
        b,
        &format!("v_mov_b32_e32 v{}, s{WSID}", l.t_idx),
        &[l.t_idx],
        &[],
        &[WSID],
    )?;
    gather_offset(
        b,
        l.t_dst,
        l.t_idx,
        Some(SDIV),
        1,
        LIVE,
        GatherTemps {
            v: [l.t_q, l.t_r, l.t_w, l.t_f],
            mask: MASK,
        },
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{XR}, v{}", l.t_dst),
        &[s(XR)],
        &[v(l.t_dst)],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{SXS4}, s{SXSTR}, 2"),
        &[SXS4],
        &[SXSTR],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{XOFFW}, s{XR}, s{SXS4}"),
        &[XOFFW],
        &[XR, SXS4],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{0}, s{XOFFW}, v{0}", l.vxo),
        &[l.vxo],
        &[l.vxo],
        &[XOFFW],
    )?;
    if gate {
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{}, {}, v{}", l.vxo2, lit(4096), l.vxo),
            &[l.vxo2],
            &[l.vxo],
            &[],
        )?
    }
    // Y pointer and the row's byte offset inside a slot's output row.
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
        ROWIN
    } else {
        sop(b, format!("s_mov_b32 s{YSEL}, s{Y0}"), &[YSEL], &[Y0])?;
        sop(
            b,
            format!("s_mov_b32 s{}, s{}", YSEL + 1, Y0 + 1),
            &[YSEL + 1],
            &[Y0 + 1],
        )?;
        ROW
    };
    sop(
        b,
        format!("s_lshl_b32 s{ROWIN4}, s{rowin}, 2"),
        &[ROWIN4],
        &[rowin],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{SYS4}, s{SYSTR}, 2"),
        &[SYS4],
        &[SYSTR],
    )?;
    // Staging masks: waves 0..n-1 stage their slot (the others run loads and stores under exec = 0); of those the waves whose X
    // row exists (`row < x_src_rows`) load it, the rest zero-fill their staging registers (K1: such rows read 0, no memory).
    sop(b, format!("s_cmp_lt_u32 s{WAVE}, s{NS}"), &[], &[WAVE, NS])?;
    sop(b, format!("s_cselect_b32 s{EMASK}, -1, 0"), &[EMASK], &[])?;
    sop(
        b,
        format!("s_cmp_lt_u32 s{XR}, s{SXSRC}"),
        &[],
        &[XR, SXSRC],
    )?;
    sop(b, format!("s_cselect_b32 s{LMASK}, -1, 0"), &[LMASK], &[])?;
    sop(
        b,
        format!("s_and_b32 s{LMASK}, s{LMASK}, s{EMASK}"),
        &[LMASK],
        &[LMASK, EMASK],
    )?;
    sop(
        b,
        format!("s_xor_b32 s{ZMASK}, s{EMASK}, s{LMASK}"),
        &[ZMASK],
        &[EMASK, LMASK],
    )
}

/// `s_load_b32 dst, slot_ids + 4*(offsets[u] + s{TMP})`, with `s{TMP}` the clamped slot position.
fn slot_id_load(b: &mut Builder, dst: u8) -> R {
    let arch = b.spec.arch;
    sop(
        b,
        format!("{} s{TMP}, s{TMP}, s{OFF0}", s_add_i32(arch)),
        &[TMP],
        &[TMP, OFF0],
    )?;
    sop(b, format!("s_lshl_b32 s{TMP}, s{TMP}, 2"), &[TMP], &[TMP])?;
    add64(b, TPTR, SLOTS, TMP)?;
    smem(b, dst, 1, TPTR, 0)
}

/// `if n <= i skip to target` (scalar, uniform): SCC = (n > i), branch when clear.
fn slot_gate(b: &mut Builder, i: u32, target: &str) -> R {
    sop(b, format!("s_cmp_gt_u32 s{NS}, {i}"), &[], &[NS])?;
    branch(b, &format!("s_cbranch_scc0 {target}"))
}

/// Row pointers and the W3 scale loads, the staged X, every weight load, the groups of each phase, the epilogue.
fn body(b: &mut Builder, c: &Ctx, slot: usize) -> R {
    let arch = b.spec.arch;
    let (spec, l) = (c.spec, &c.l);
    let (kind, gate) = (spec.kind, spec.kind == Kind::GateUp);
    if c.tiled {
        output_gate(b, &c.lbl("rowptr_done"))?;
    }
    match spec.codec {
        Codec::Mq3I4 => {
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
                format!("s_mul_i32 s{TMP2}, s{ROW}, {}", 2 * spec.groups()),
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
                smem(b, SCW, 8, SPTR, 0)?;
                smem(b, SC8, 2, SPTR, 0x20)?;
            }
        }
        Codec::Mq4Sym => {
            let stride = spec.groups() * if gate { 136 } else { 68 };
            sop(
                b,
                format!("s_mul_i32 s{TMP}, s{ROW}, {}", lit(stride)),
                &[TMP],
                &[ROW],
            )?;
            add64(b, ROWPTR, BASE, TMP)?;
        }
    }
    if c.tiled {
        b.label(&c.lbl("rowptr_done"))?;
    }
    // Phase-0 X loads, then every weight load of the row (in group order), all before the first wait.
    stage_loads(b, c, 0)?;
    if c.tiled {
        output_gate(b, &c.lbl("weights_done"))?;
    }
    let weight_loads = emit_weight_loads(b, c)?;
    for k in 0..MAX_SLOTS as u8 * kind.chains() {
        valu(
            b,
            &format!("v_mov_b32_e32 v{}, 0", l.acc + k),
            &[l.acc + k],
            &[],
            &[],
        )?
    }
    if c.tiled {
        b.label(&c.lbl("weights_done"))?;
    }
    // The X loads retire (global loads return in issue order), then the stores and the publish barrier.
    // Tail waves skipped weight loads, so their X loads cannot use the live waves'
    // vmcnt(weight_loads) threshold. Drain all loads before publishing the staged X.
    b.wait(Counter::Vm, if c.tiled { 0 } else { weight_loads })?;
    stage_stores(b, c, slot)?;
    b.barrier(&[Transition::Ready(slot)])?;
    b.label(&c.lbl("c0"))?;
    if c.tiled {
        output_gate(b, &c.lbl(if gate { "s1" } else { "epi" }))?;
    }
    for gl in 0..spec.groups_per_phase() {
        group(b, c, slot, gl, gl)?
    }
    if gate {
        b.label(&c.lbl("s1"))?;
        valu(
            b,
            &format!(
                "v_add_nc_u32_e32 v{0}, {1}, v{0}",
                l.vxo,
                lit(kind.phase_cols() * 4)
            ),
            &[l.vxo],
            &[l.vxo],
            &[],
        )?;
        valu(
            b,
            &format!(
                "v_add_nc_u32_e32 v{0}, {1}, v{0}",
                l.vxo2,
                lit(kind.phase_cols() * 4)
            ),
            &[l.vxo2],
            &[l.vxo2],
            &[],
        )?;
        stage_loads(b, c, 1)?;
        b.barrier(&[Transition::Retire(slot)])?;
        b.wait(Counter::Vm, 0)?;
        stage_stores(b, c, slot)?;
        b.barrier(&[Transition::Ready(slot)])?;
        b.label(&c.lbl("c1"))?;
        if c.tiled {
            output_gate(b, &c.lbl("epi"))?;
        }
        for gl in 0..spec.groups_per_phase() {
            group(b, c, slot, spec.groups_per_phase() + gl, gl)?
        }
    }
    b.label(&c.lbl("epi"))?;
    if c.tiled {
        output_gate(b, &c.lbl("exit"))?;
    }
    epilogue(b, c)
}

/// Skip output-only work for a tail wave; never place a workgroup barrier in the skipped region.
fn output_gate(b: &mut Builder, target: &str) -> R {
    sop(
        b,
        format!("s_cmp_eq_u32 s{OUTPUT_LIVE}, 0"),
        &[],
        &[OUTPUT_LIVE],
    )?;
    branch(b, &format!("s_cbranch_scc1 {target}"))
}

/// The wave's `x_loads` `global_load_b128` of its staging slot's current phase (`float4 = lane + 32 j`, base `Xrot`, per-lane
/// byte offset `xoff_wave + 16*lane (+ phase columns)`), under the load mask. A wave whose X row is at or past `x_src_rows`
/// first zero-fills its staging registers under the zero mask (skipped by a scalar branch when no wave needs it) and loads
/// nothing, so the staged row is 0 and no memory is read, exactly K1's raw-descriptor semantics. Immediates stop at 4095, so
/// `j >= 8` uses the second offset register (`+4096`).
fn stage_loads(b: &mut Builder, c: &Ctx, phase: u32) -> R {
    let l = &c.l;
    let skip = c.lbl(&format!("z{phase}"));
    sop(b, format!("s_cmp_eq_u32 s{ZMASK}, 0"), &[], &[ZMASK])?;
    branch(b, &format!("s_cbranch_scc1 {skip}"))?;
    op(b, format!("s_mov_b32 exec_lo, s{ZMASK}"), &[], &[s(ZMASK)])?;
    for &dst in &l.stage {
        for r in 0..4 {
            valu(
                b,
                &format!("v_mov_b32_e32 v{}, 0", dst + r),
                &[dst + r],
                &[],
                &[],
            )?
        }
    }
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
    b.label(&skip)?;
    op(b, format!("s_mov_b32 exec_lo, s{LMASK}"), &[], &[s(LMASK)])?;
    for (j, &dst) in l.stage.iter().enumerate() {
        let (voff, imm) = if j < 8 {
            (l.vxo, 512 * j as u32)
        } else {
            (l.vxo2, 512 * (j as u32 - 8))
        };
        gload(b, "global_load_b128", dst, 4, voff, XP, imm)?;
    }
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])
}

/// The staged float4s into LDS slot `wave` (slot-major: LDS address `wave * slot_bytes + 16 * (lane + 32 j)`).
fn stage_stores(b: &mut Builder, c: &Ctx, slot: usize) -> R {
    let l = &c.l;
    op(b, format!("s_mov_b32 exec_lo, s{EMASK}"), &[], &[s(EMASK)])?;
    for (j, &src) in l.stage.iter().enumerate() {
        ds_st(b, slot, l.ldsst, src, 512 * j as u32)?
    }
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])
}

/// Every weight load of the row in group order; returns their count (the explicit `vmcnt` of the first wait).
///  * W3: one `global_load_b64` packet window per group (+ the down-only scale `global_load_u16`);
///  * MQ4 gate: header dword at `+136g + 4*(l >= 16)`, payload dword at `+136g + 8 + 4l`;
///  * MQ4 down: header dword at `+68g` (same address in every lane), payload u16 at `+68g + 4 + 2l`.
fn emit_weight_loads(b: &mut Builder, c: &Ctx) -> Result<u8, String> {
    let (spec, l) = (c.spec, &c.l);
    let gate = spec.kind == Kind::GateUp;
    let mut n = 0u8;
    match spec.codec {
        Codec::Mq3I4 => {
            for g in 0..spec.groups() {
                gload(
                    b,
                    "global_load_b64",
                    l.a[g as usize],
                    2,
                    l.w3_off,
                    ROWPTR,
                    48 * g,
                )?;
                n += 1;
                if !gate {
                    gload(
                        b,
                        "global_load_u16",
                        l.b[g as usize],
                        1,
                        l.zero,
                        SPTR,
                        2 * g,
                    )?;
                    n += 1;
                }
            }
        }
        Codec::Mq4Sym => {
            for g in 0..spec.groups() {
                if gate {
                    gload(
                        b,
                        "global_load_b32",
                        l.a[g as usize],
                        1,
                        l.hoff,
                        ROWPTR,
                        136 * g,
                    )?;
                    gload(
                        b,
                        "global_load_b32",
                        l.b[g as usize],
                        1,
                        l.poff,
                        ROWPTR,
                        136 * g + 8,
                    )?;
                } else {
                    gload(
                        b,
                        "global_load_b32",
                        l.a[g as usize],
                        1,
                        l.zero,
                        ROWPTR,
                        68 * g,
                    )?;
                    gload(
                        b,
                        "global_load_u16",
                        l.b[g as usize],
                        1,
                        l.poff,
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

/// One group of the row (global index `g`, index `gl` inside the current LDS phase): the weights are dequantized once, then
/// every live slot applies its partial from its own X. Slots are read and computed in batches of `batch()`; the X reads of
/// a batch are closed by one explicit `lgkmcnt(0)` OUTSIDE every slot gate, so a skipped slot never desynchronizes the
/// builder's wait ledger (no wait or pending DS read crosses a gate).
fn group(b: &mut Builder, c: &Ctx, slot: usize, g: u32, gl: u32) -> R {
    let (spec, l) = (c.spec, &c.l);
    let chains = spec.kind.chains();
    let bsz = spec.batch();
    let batches = MAX_SLOTS / bsz;
    let done = c.lbl(&format!("g{g}_done"));
    for batch in 0..batches {
        let base = batch * bsz;
        if batch > 0 {
            slot_gate(b, base, &done)?
        }
        let (lw, ld) = (
            c.lbl(&format!("g{g}_b{batch}_lw")),
            c.lbl(&format!("g{g}_b{batch}_ld")),
        );
        for k in 0..bsz {
            if k > 0 {
                slot_gate(b, base + k, &lw)?
            }
            xload(b, c, slot, base + k, k as usize, gl)?;
        }
        b.label(&lw)?;
        if batch == 0 {
            match spec.codec {
                Codec::Mq3I4 => w3_unpack(b, c, g)?,
                Codec::Mq4Sym => mq4_unpack(b, c, g)?,
            }
        }
        b.wait(Counter::Lgkm, 0)?;
        for k in 0..bsz {
            if k > 0 {
                slot_gate(b, base + k, &ld)?
            }
            let chain = if chains == 4 { (g & 3) as u8 } else { 0 };
            let acc = l.ac(base + k, chains) + chain;
            match spec.codec {
                Codec::Mq3I4 => w3_partial(b, c, k as usize, acc)?,
                Codec::Mq4Sym => mq4_partial(b, c, k as usize, acc)?,
            }
        }
        b.label(&ld)?;
    }
    if batches > 1 {
        b.label(&done)?
    }
    Ok(())
}

/// The LDS read of slot `i`'s X of group `gl` into buffer `k`: `ds_load_b128` at `i*slot_bytes + 512*gl + 16*lane`
/// (MQ4 gate: two at `i*slot_bytes + 1024*gl + 32*lane` and `+16`).
fn xload(b: &mut Builder, c: &Ctx, slot: usize, i: u32, k: usize, gl: u32) -> R {
    let (spec, l) = (c.spec, &c.l);
    let base = i * spec.kind.slot_bytes();
    if spec.lane_elems() == 8 {
        ds_ld(b, slot, l.xb[k], l.ldsq, base + 1024 * gl)?;
        ds_ld(b, slot, l.xb[k] + 4, l.ldsq, base + 1024 * gl + 16)
    } else {
        ds_ld(b, slot, l.xb[k], l.ldsq, base + 512 * gl)
    }
}

/// One W3 group's dequantization: scale, packet, the four weights (stage-major: the four chains are independent).
fn w3_unpack(b: &mut Builder, c: &Ctx, g: u32) -> R {
    let (spec, l) = (c.spec, &c.l);
    let gi = g as usize;
    if spec.kind == Kind::GateUp {
        // Group g's f16 is the low (even g) or high (odd g) half of SGPR g>>1; cvt reads the low 16 bits.
        let src = SCW + (g >> 1) as u8;
        let from = if g & 1 == 1 {
            let t = STMP + ((g >> 1) & 1) as u8;
            sop(b, format!("s_lshr_b32 s{t}, s{src}, 16"), &[t], &[src])?;
            t
        } else {
            src
        };
        valu(
            b,
            &format!("v_mov_b32_e32 v{}, s{from}", l.d),
            &[l.d],
            &[],
            &[from],
        )?;
        valu(
            b,
            &format!("v_cvt_f32_f16_e64 v{0}, v{0}.l", l.d),
            &[l.d],
            &[l.d],
            &[],
        )?;
    } else {
        valu(
            b,
            &format!("v_cvt_f32_f16_e64 v{}, v{}.l", l.d, l.b[gi]),
            &[l.d],
            &[l.b[gi]],
            &[],
        )?;
    }
    // packet = bits [12l, 12l + 64) of the lane's 8-byte window, low dword: `v_lshrrev_b64 pk:pk+1, shift, hi:lo`.
    let (pk, window) = (vr(l.pk, 2), vr(l.a[gi], 2));
    b.push(Instruction::new(
        format!("v_lshrrev_b64 {pk}, v{}, {window}", l.w3_sh),
        vec![pk],
        vec![v(l.w3_sh), window],
    ))?;
    for i in 0..4u8 {
        valu(
            b,
            &format!("v_bfe_u32 v{}, v{}, {}, 3", l.w + i, l.pk, u32::from(i) * 3),
            &[l.w + i],
            &[l.pk],
            &[],
        )?
    }
    for i in 0..4u8 {
        let w = l.w + i;
        valu(
            b,
            &format!("v_lshl_add_u32 v{w}, v{w}, 1, -7"),
            &[w],
            &[w],
            &[],
        )?
    }
    for i in 0..4u8 {
        let w = l.w + i;
        valu(b, &format!("v_cvt_f32_i32_e32 v{w}, v{w}"), &[w], &[w], &[])?
    }
    for i in 0..4u8 {
        let w = l.w + i;
        valu(
            b,
            &format!("v_mul_f32_e32 v{w}, v{}, v{w}", l.d),
            &[w],
            &[l.d, w],
            &[],
        )?
    }
    Ok(())
}

/// One slot's W3 partial from X buffer `k`: `p = w0*x0`, `fmac(w_j, x_j)` for `j = 1..3`, then the chain add.
fn w3_partial(b: &mut Builder, c: &Ctx, k: usize, acc: u8) -> R {
    let l = &c.l;
    let (x, dot) = (l.xb[k], l.dot[k & 1]);
    valu(
        b,
        &format!("v_mul_f32_e32 v{dot}, v{}, v{x}", l.w),
        &[dot],
        &[l.w, x],
        &[],
    )?;
    for j in 1..4u8 {
        valu(
            b,
            &format!("v_fmac_f32_e32 v{dot}, v{}, v{}", l.w + j, x + j),
            &[dot],
            &[dot, l.w + j, x + j],
            &[],
        )?
    }
    valu(
        b,
        &format!("v_add_f32_e32 v{acc}, v{acc}, v{dot}"),
        &[acc],
        &[acc, dot],
        &[],
    )
}

/// One MQ4 group's dequantization: `d`, `z` from the header halves, `nw` weights `fma(d, q, z)`.
fn mq4_unpack(b: &mut Builder, c: &Ctx, g: u32) -> R {
    let l = &c.l;
    let gi = g as usize;
    let (hdr, pay) = (l.a[gi], l.b[gi]);
    valu(
        b,
        &format!("v_cvt_f32_f16_e64 v{}, v{hdr}.l", l.d),
        &[l.d],
        &[hdr],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshrrev_b32_e32 v{}, 16, v{hdr}", l.z),
        &[l.z],
        &[hdr],
        &[],
    )?;
    valu(
        b,
        &format!("v_cvt_f32_f16_e64 v{0}, v{0}.l", l.z),
        &[l.z],
        &[l.z],
        &[],
    )?;
    for i in 0..l.nw {
        valu(
            b,
            &format!("v_bfe_u32 v{}, v{pay}, {}, 4", l.w + i, u32::from(i) * 4),
            &[l.w + i],
            &[pay],
            &[],
        )?
    }
    for i in 0..l.nw {
        let w = l.w + i;
        valu(b, &format!("v_cvt_f32_u32_e32 v{w}, v{w}"), &[w], &[w], &[])?
    }
    for i in 0..l.nw {
        let w = l.w + i;
        valu(
            b,
            &format!("v_fma_f32 v{w}, v{}, v{w}, v{}", l.d, l.z),
            &[w],
            &[l.d, w, l.z],
            &[],
        )?
    }
    Ok(())
}

/// One slot's MQ4 partial from X buffer `k` in the production contraction order (`p = w1*x1`, `fma(w0, x0, p)`,
/// `fma(w_j, x_j, p)`, `j = 2..nw-1`), then the chain add.
fn mq4_partial(b: &mut Builder, c: &Ctx, k: usize, acc: u8) -> R {
    let l = &c.l;
    let (x, dot) = (l.xb[k], l.dot[k & 1]);
    valu(
        b,
        &format!("v_mul_f32_e32 v{dot}, v{}, v{}", l.w + 1, x + 1),
        &[dot],
        &[l.w + 1, x + 1],
        &[],
    )?;
    valu(
        b,
        &format!("v_fmac_f32_e32 v{dot}, v{}, v{x}", l.w),
        &[dot],
        &[dot, l.w, x],
        &[],
    )?;
    for j in 2..l.nw {
        valu(
            b,
            &format!("v_fmac_f32_e32 v{dot}, v{}, v{}", l.w + j, x + j),
            &[dot],
            &[dot, l.w + j, x + j],
            &[],
        )?
    }
    valu(
        b,
        &format!("v_add_f32_e32 v{acc}, v{acc}, v{dot}"),
        &[acc],
        &[acc, dot],
        &[],
    )
}

/// Shuffle tables, per live slot the gate fold and the 5-step snapshot shuffle-down, then lane 0's stores.
fn epilogue(b: &mut Builder, c: &Ctx) -> R {
    let arch = b.spec.arch;
    let (spec, l) = (c.spec, &c.l);
    let chains = spec.kind.chains();
    // Shuffle step k (offset 16 >> k): mask lane < 32 - off, address (lane + off or lane) * 4.
    for (k, off) in [16u32, 8, 4, 2, 1].into_iter().enumerate() {
        let (m, sh) = (SHM + k as u8, l.shuf + k as u8);
        op(
            b,
            format!("v_cmp_gt_u32_e64 s{m}, {}, v{}", 32 - off, l.lane),
            &[s(m)],
            &[v(l.lane)],
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
            &format!("v_add_lshl_u32 v{sh}, v{sh}, v{}, 2", l.lane),
            &[sh],
            &[sh, l.lane],
            &[],
        )?;
    }
    let red_done = c.lbl("red_done");
    for i in 0..MAX_SLOTS {
        if i > 0 {
            slot_gate(b, i, &red_done)?
        }
        let a0 = l.ac(i, chains);
        if spec.kind == Kind::GateUp {
            let (a1, a2, a3) = (a0 + 1, a0 + 2, a0 + 3);
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
                    vec![v(l.exch)],
                    vec![v(sh), v(a0)],
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
    // Lane 0 stores each live slot's value at its original slot's row (each store its own offset register).
    valu(
        b,
        &format!("v_cmpx_eq_u32_e32 0, v{}", l.lane),
        &[],
        &[l.lane],
        &[],
    )?;
    let st_done = c.lbl("st_done");
    for i in 0..MAX_SLOTS {
        if i > 0 {
            slot_gate(b, i, &st_done)?
        }
        let (vy, a0) = (l.vy + i as u8, l.ac(i, chains));
        sop(
            b,
            format!("s_mul_i32 s{TMP}, s{}, s{SYS4}", SID + i as u8),
            &[TMP],
            &[SID + i as u8, SYS4],
        )?;
        sop(
            b,
            format!("{} s{TMP}, s{TMP}, s{ROWIN4}", s_add_i32(arch)),
            &[TMP],
            &[TMP, ROWIN4],
        )?;
        valu(
            b,
            &format!("v_mov_b32_e32 v{vy}, s{TMP}"),
            &[vy],
            &[],
            &[TMP],
        )?;
        b.push(
            Instruction::new(
                format!("global_store_b32 v{vy}, v{a0}, s[{YSEL}:{}]", YSEL + 1),
                vec![],
                vec![v(vy), v(a0), sr(YSEL, 2)],
            )
            .memory(MemoryClass::VmemStore),
        )?;
    }
    b.label(&st_done)
}

/// The shape contract of one emitted entry: counts of its characteristic instructions, the forbidden families, the honest
/// register ceilings, wave32, no spills, no private memory, and the fixed LDS. Committed as
/// `crates/peacemaker-isa/kernels/mq_i4_moe_gemv_cn.gfx1151.<tag>.contract.json`.
pub fn contract(spec: Spec, e: &Emitted) -> serde_json::Value {
    contract_impl(spec, e, false)
}

/// Shape contract for the fixed-eight-row descriptor entry, including its `_t8` symbol.
pub fn contract_rowtiles(spec: Spec, e: &Emitted) -> serde_json::Value {
    contract_impl(spec, e, true)
}

fn contract_impl(spec: Spec, e: &Emitted, tiled: bool) -> serde_json::Value {
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
        "symbol": if tiled { format!("{}_t8", spec.symbol()) } else { spec.symbol() }, "counts": counts,
        "forbidden": forbidden,
        "vgpr_max": e.shape.next_free_vgpr, "sgpr_max": e.shape.next_free_sgpr,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0, "group_segment_fixed_bytes": spec.kind.lds_bytes(),
    })
}
const COUNTED: [&str; 42] = [
    "s_load_b256",
    "s_load_b128",
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
    "s_min_u32",
    "v_mul_lo_u32",
    "s_mul_i32",
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
