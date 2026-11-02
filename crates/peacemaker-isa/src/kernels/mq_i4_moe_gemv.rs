// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact F4 read-once C8 lane-union decode GEMV over pointer-addressed expert
//! tensors, emitted with the checked ISA builder for gfx1151 and gfx1201: four symbols
//! `mq_i4_moe_gemv_{mq3i4,mq4sym}_{r256_gate_up,r128_down}` in the module
//! `mq_i4_moe_gemv_pm_<arch>`. The two bodies are one instruction sequence; they differ only where the
//! architectures do (workgroup-id source, `s_sub`/`s_add` spellings, SMEM zero offset, buffer `soffset`,
//! wait/hazard instructions the builder inserts).
//!
//! Frozen 96-byte kernarg ABI (DESIGN §4.2 plus the trailing `n_experts` and a reserved zero word, identical for all four symbols):
//! u64 pointers `expert_weight_ptrs@0` (table indexed by expert id; each entry is
//! the projection tensor's code-plane base of that expert, never
//! `expert_id * stride`), `expert_ids@8` (`u32[U]`, planner first-occurrence order), `expert_offsets@16`
//! (`u32[U+1]`), `slot_ids@24` (`u32[10*C]`, original slots `lane*10+rank`),
//! `Xrot@32` (F32 rotated input), `Y0@40`, `Y1@48`; u32 `M@56`, `K@60`, `U@64`,
//! `C@68`, `x_row_div@72`, `x_src_rows@76`, `x_row_stride@80`, `y_slot_stride@84`, `n_experts@88` (the extent of
//! the `expert_weight_ptrs` table), `reserved@92` (zero, pads the block to 8 bytes). Strides are F32 elements.
//!
//! Launch: grid `[M, U, 1]` (x = output row, y = distinct expert), block `[32, 1, 1]`
//! (one wave32 per workgroup, no LDS: the only cross-lane traffic is `ds_bpermute_b32`).
//! A workgroup whose row `>= M`, expert index `>= U`, whose expert id is `>= n_experts` (unsigned compare, so a
//! negative id too; checked before the table is indexed), or whose slot count
//! `n = expert_offsets[u+1] - expert_offsets[u]` is `0`, exceeds [`MAX_SLOTS`]` = 8`
//! or `expert_offsets[u+1] > 10*C` returns without a store.
//!
//! Slots. The expert's `n <= 8` original slots are `slot_ids[offsets[u] .. +n]`. Slot
//! `s` reads the X row `s / x_row_div` (gate: `x_row_div = 10`, `X[C,K]`; down:
//! `x_row_div = 1`, `X[10C,K]`) through a raw buffer descriptor of
//! `x_src_rows * x_row_stride * 4` bytes, so a row at or past `x_src_rows` reads `0`
//! and never touches memory; it writes its F32 result at original slot `s`:
//! `Y[s*y_slot_stride + r]`. Gate (`M` fused gate+up rows, `mi = M/2`): `r < mi`
//! goes to `Y0[s*y_slot_stride + r]`, otherwise to `Y1[s*y_slot_stride + r - mi]`.
//! Down: `Y0[s*y_slot_stride + r]` (`Y1` unused). No route weight is applied (the
//! existing combine owns it); no atomics; output is indexed by original slot, not by
//! sorted position. Domain (host-validated, never re-checked): `K > 0`,
//! `K % 128 == 0` (`K % 256 == 0` for `Mq4Sym` gate), `x_row_div >= 1`, slots
//! `< 2^22`, `x_src_rows*x_row_stride*4`, `M*K*b/8 + 2*M*K/128` (planar) and every
//! output offset below `2^32`, scales finite (every expert id is range-checked against `n_experts` in the kernel).
//!
//! Read-once C8. The wave loads and dequantizes each 128-column (`Mq4Sym` gate:
//! 256-column) row group exactly once; the up-to-eight slots of the expert then run
//! their own multiply/FMA chains on the shared dequantized weights, each into its own
//! accumulator register(s) in its own K order. Slots are skipped with scalar
//! branches (`i >= n`), never by re-reading weights: no weight load sits inside a
//! slot branch (the census test asserts it). The weight-class loads of a group are the
//! same instruction count for `n = 1` and `n = 8`.
//!
//! Exactness. Each (slot, output row) value is the frozen F4 tree. Lane `l` owns the
//! contiguous elements below; per group `Mq3I4`: `p = RN(w0*x0)` (`v_mul_f32`) followed by
//! left-associated `p = fma(w_j, x_j, p)` (`v_fmac_f32`), `Mq4Sym`: see below; the group partial is added
//! separately into the chain accumulator; accumulators start at `+0`; lane value
//! gate `RN(RN(a0+a1)+RN(a2+a3))` over four chains `acc[g&3]`, down the single chain;
//! then the snapshot shuffle-down at offsets 16, 8, 4, 2, 1 with out-of-range lanes
//! unchanged (`v_add_f32` into scratch, `v_cndmask_b32` mask select), lane 0 stores.
//!  * `Mq3I4` (planar qt131/qt133, exactly `mq_i4_rows`): lane `l` owns `g*128 + 4l ..+3`
//!    of every 128-column group `g` (both kinds); `w_i = RN(float(d) * float(2c-7))`;
//!    codes `v_bfe_u32(packet, 3i, 3)` from the 12-bit packet at group bit `12l`
//!    (two byte loads, never straddling the group or over-reading the plane); scale
//!    plane at `base + M*K*3/8`, f16 of row `r` group `g` at `2*(r*K/128+g)`.
//!  * `Mq4Sym` (pointer-addressed MQ4, the existing MQ4 arithmetic): `w = fma(d, float(q), z)`
//!    with the header's f16 `d` and `z` (`z = -8*d` for symmetric tensors; never an odd
//!    grid), `q` the unsigned nibble, element `j` the low-nibble-first nibble `j`.
//!    Gate (legacy G256 group, 136 B: f16 `d,z` for columns 0..127 at `+0`, 128..255 at
//!    `+4`, payload at `+8`): lane `l` owns the eight elements `g*256 + 8l ..+7`, header
//!    half `l >= 16`, chain `acc[g&3]`, eight terms in the production contraction order
//!    `p = RN(w1*x1)`, `p = fma(w0,x0,p)`, then `fma(w_j,x_j,p)` for `j = 2..7`. Down
//!    (G128 group, 68 B: f16 `d,z` at `+0`, payload at `+4`, row stride `K/128*68`): lane
//!    `l` owns `g*128 + 4l ..+3`, the u16 at payload byte `2l`, one chain, the same order over four terms.
//!
//! Unpack: one `v_bfe_u32` per code plus the affine conversion, in [`unpack`]. Production [`emit`]
//! is the simple exact reference network: the `Mq3I4` packet is two `global_load_u8` (group byte
//! `(3l)>>1` and `+1`) combined by `v_lshl_or_b32` (`b0 | b1 << 8`). One revised network is retained as
//! [`emit_optimized`]: a single `global_load_u16` at the same byte offset (bytes `(3l)>>1` and `+1`, at most
//! byte 47 of the 48-byte group, so it never crosses the group or over-reads the plane; the hardware load may be
//! unaligned), the same 16 bits and hence the same 12 after the `4*(l&1)` shift. It was measured byte-exact on
//! GPU but slower than the reference, so the optimization was rejected by timing and is kept only for
//! comparison; no speed claim is made for either network. The network is an emitter choice, not a
//! symbol axis or ABI field; `Mq4Sym` and every arithmetic step are unchanged.
//!
//! Legacy diagnostic twins (`_c16`, `_c32`, [`Cap`]; [`wide`]) retain the same four
//! symbols for 16 / 32 token lanes (`n <= 16 | 32`, `U <= 10*C`). Same 96-byte ABI
//! and the same arithmetic; grid `[ceil(M/8), U, 1]`, block 256 (one wave per matrix-local output row, eight rows per
//! workgroup, every slot of the expert per wave), the slots' X staged through LDS eight at a time. The C8 entries above are
//! unchanged byte for byte.
//!
//! Row tiles ([`emit_rowtiles`]) use fixed eight-slot unions at every logical width
//! 1..=64, not larger accumulator banks. The unchanged 96-byte ABI has `U=80`,
//! `C=total rows`; grid `[ceil(M/8),80,ceil(C/8)]`, block256. Only descriptor
//! pointers advance by `1928*wg_z` bytes. Slots remain absolute and X/Y retain
//! their whole-window bases. gfx1151 uses its incumbent Cn schedule and symbol
//! family; gfx1201 uses the eight-output-wave read-once schedule in [`wide`].

use crate::arch::{WgIdSource, WgIds};
use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{
    add64, bload, gather_offset, lit, op, s, s_add_i32, smem, sop, sr, srd_tail, v, vr, GatherTemps,
};
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

pub mod wide;

/// Weight codec of the tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codec {
    /// Planar 3-bit codes (qt131/qt133), `2c - 7`, one f16 scale per row-group.
    Mq3I4,
    /// Pointer-addressed MQ4 (qt44 gate group 136 B / qt53 down group 68 B), `d*q + z`.
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

/// Token-lane capacity of an entry: `C8` is the narrow entry (one wave32 per workgroup, at most eight slots per expert),
/// `C16` / `C32` the wide twins ([`wide`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cap {
    C8,
    C16,
    C32,
}
impl Cap {
    pub const ALL: [Cap; 3] = [Cap::C8, Cap::C16, Cap::C32];
    /// Most distinct original slots one expert workgroup serves.
    pub fn slots(self) -> u32 {
        match self {
            Self::C8 => MAX_SLOTS,
            Self::C16 => 16,
            Self::C32 => 32,
        }
    }
    /// Symbol / tag suffix (empty for the narrow entry).
    pub fn suffix(self) -> &'static str {
        match self {
            Self::C8 => "",
            Self::C16 => "_c16",
            Self::C32 => "_c32",
        }
    }
}

/// One entry: `(arch, codec, kind, cap)`; gfx1151 or gfx1201.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
    pub codec: Codec,
    pub kind: Kind,
    pub cap: Cap,
}
impl Spec {
    /// The narrow (C8) entry.
    pub fn narrow(arch: Arch, codec: Codec, kind: Kind) -> Self {
        Self {
            arch,
            codec,
            kind,
            cap: Cap::C8,
        }
    }
    /// The exported symbol, `mq_i4_moe_gemv_<codec>_<kind>` plus `_c16` / `_c32` for a wide twin.
    pub fn symbol(&self) -> String {
        format!("mq_i4_moe_gemv_{}", self.tag())
    }
    /// `<codec>_<kind>[_c16|_c32]`: the `--epi` name, the label prefix and the contract file tag.
    pub fn tag(&self) -> String {
        format!(
            "{}_{}{}",
            self.codec.tag(),
            self.kind.tag(),
            self.cap.suffix()
        )
    }
    pub fn module(arch: Arch) -> String {
        format!("mq_i4_moe_gemv_pm_{}", arch.name())
    }
    pub fn validate(&self) -> R {
        if !matches!(self.arch, Arch::Gfx1151 | Arch::Gfx1201) {
            return Err("mq_i4_moe_gemv builder is exact-gfx1151 / gfx1201 only".into());
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
    /// Group instances in the emitted body: gate four loop groups plus three tail groups, down one.
    pub fn group_instances(&self) -> u32 {
        match self.kind {
            Kind::GateUp => 7,
            Kind::Down => 1,
        }
    }
    /// Threads per workgroup: one wave32 (narrow) or eight (wide).
    pub fn workgroup_size(&self) -> u32 {
        if self.cap == Cap::C8 {
            WORKGROUP_SIZE
        } else {
            wide::WORKGROUP_SIZE
        }
    }
    /// Static LDS bytes: none (narrow) or the X tile of eight slots (`nw` f32 per lane per slot).
    pub fn lds_bytes(&self) -> u32 {
        if self.cap == Cap::C8 {
            0
        } else {
            wide::SLOT_TILE * 32 * 4 * u32::from(self.lane_elems())
        }
    }
}

/// Diagnostic module order: four C8 entries, then four C16 and four C32 entries.
/// Production image composition uses [`narrow_specs`] plus [`emit_rowtiles`],
/// not the legacy wide diagnostics.
pub fn module_specs(arch: Arch) -> Vec<Spec> {
    Cap::ALL
        .into_iter()
        .flat_map(|cap| {
            Codec::ALL.into_iter().flat_map(move |codec| {
                Kind::ALL.into_iter().map(move |kind| Spec {
                    arch,
                    codec,
                    kind,
                    cap,
                })
            })
        })
        .collect()
}

/// The four narrow (C8) entries of one architecture.
pub fn narrow_specs(arch: Arch) -> Vec<Spec> {
    module_specs(arch)
        .into_iter()
        .filter(|s| s.cap == Cap::C8)
        .collect()
}

// ----------------------------------------------------------------------- SGPRs
// s0:1 kernarg pointer; s2, s3 workgroup ids on gfx11 (the first system SGPRs after the two user SGPRs; gfx12:
// ttmp9 (x) and ttmp7[15:0] (y)).
const KARG: u8 = 0;
const WGX: u8 = 2;
const WGY: u8 = 3;
/// Kernargs (s_load_b256 @0, b256 @0x20, b128 @0x40, b64 @0x50).
const WPTRS: u8 = 8;
const EIDS: u8 = 10;
const OFFS: u8 = 12;
const SLOTS: u8 = 14;
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
/// SGPRs the plan declares (the honest ceiling, `sgpr_max` of the contract).
const SGPRS: u8 = 81;
/// `n_experts@88`: the extent of the `expert_weight_ptrs` table; an `expert_ids` entry at or past it is never indexed.
const SNEXP: u8 = 80;

// ----------------------------------------------------------------------- VGPRs
const LANE: u8 = 0;

/// VGPR layout of one entry, assigned sequentially so no range can overlap.
#[derive(Clone, Copy, Debug)]
struct Lay {
    /// Weights / X elements per lane per group, accumulator chains per slot.
    nw: u8,
    chains: u8,
    caddr: u8,
    saddr: u8,
    xl: u8,
    cl: u8,
    shift: u8,
    hl: u8,
    xb: u8,
    xa: u8,
    pkt0: u8,
    pkt1: u8,
    d: u8,
    z: u8,
    dot: u8,
    exch: u8,
    shuf: u8,
    w: u8,
    x: u8,
    acc: u8,
    xo: u8,
    ya: u8,
    end: u8,
}
impl Lay {
    fn new(spec: Spec) -> Self {
        let (nw, chains) = (spec.lane_elems(), spec.kind.chains());
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
        let shift = take(1, 1);
        let hl = take(1, 1);
        let xb = take(1, 1);
        let xa = take(1, 1);
        let pkt0 = take(1, 1);
        let pkt1 = take(1, 1);
        let d = take(1, 1);
        let z = take(1, 1);
        let dot = take(2, 1);
        let exch = take(1, 1);
        let shuf = take(5, 1);
        let w = take(nw, 4);
        let x = take(MAX_SLOTS as u8 * nw, 4);
        let acc = take(MAX_SLOTS as u8 * chains, 1);
        let xo = take(MAX_SLOTS as u8, 1);
        let ya = take(MAX_SLOTS as u8, 1);
        let end = n.div_ceil(8) * 8;
        Self {
            nw,
            chains,
            caddr,
            saddr,
            xl,
            cl,
            shift,
            hl,
            xb,
            xa,
            pkt0,
            pkt1,
            d,
            z,
            dot,
            exch,
            shuf,
            w,
            x,
            acc,
            xo,
            ya,
            end,
        }
    }
    /// First X register of slot `i` (`nw` consecutive registers).
    fn xr(&self, i: u32) -> u8 {
        self.x + i as u8 * self.nw
    }
    /// Accumulator register of slot `i`, chain `c`.
    fn ac(&self, i: u32, c: u8) -> u8 {
        self.acc + i as u8 * self.chains + c
    }
}

fn lbl(spec: Spec, name: &str) -> String {
    format!(".Lmqi4moe_{}_{name}", spec.tag())
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
fn plan(arch: Arch, l: &Lay) -> Result<RegPlan, String> {
    let mut regs = RegPlan::new(u16::from(l.end), u16::from(SGPRS))?;
    // gfx11 receives the workgroup ids in the first two system SGPRs; gfx12 in ttmp9 / ttmp7.
    if !arch.gfx12() {
        for (name, base) in [("workgroup_id_x", WGX), ("workgroup_id_y", WGY)] {
            declare(&mut regs, name, RKind::S, base, 1)?;
        }
    }
    for (name, base, len) in [
        ("kernarg", KARG, 2),
        ("kernel_arguments_ptrs", WPTRS, 8),
        ("kernel_arguments_x_y_m_k", XROT, 8),
        ("kernel_arguments_u_c_div_src", SU, 4),
        ("kernel_arguments_strides", SXSTR, 2),
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
    ] {
        declare(&mut regs, name, RKind::S, base, len)?;
    }
    declare(&mut regs, "kernel_arguments_n_experts", RKind::S, SNEXP, 1)?;
    for (name, base, len) in [
        ("lane", LANE, 1),
        ("code_address", l.caddr, 1),
        ("scale_address", l.saddr, 1),
        ("x_lane_offset", l.xl, 1),
        ("code_lane_offset", l.cl, 1),
        ("packet_shift", l.shift, 1),
        ("header_lane_offset", l.hl, 1),
        ("x_base", l.xb, 1),
        ("x_address", l.xa, 1),
        ("packet0", l.pkt0, 1),
        ("packet1", l.pkt1, 1),
        ("scale", l.d, 1),
        ("zero_point", l.z, 1),
        ("dots", l.dot, 2),
        ("exchange", l.exch, 1),
        ("shuffle_addresses", l.shuf, 5),
        ("weights", l.w, l.nw),
        ("x_values", l.x, MAX_SLOTS as u8 * l.nw),
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

/// Production entry: the simple exact reference network (two byte loads + `v_lshl_or_b32`).
pub fn emit(spec: Spec) -> Result<Emitted, String> {
    emit_with_network(spec, Network::Reference)
}

/// Fixed-eight-row descriptor tiles for total logical rows 1..=64.
///
/// `cap` does not select a register-bank width here: the physical capacity is
/// always eight. The architecture's incumbent projection symbol gains `_t8`.
/// gfx1151 keeps its Cn symbol/module family; gfx1201 keeps this module family.
pub fn emit_rowtiles(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    if spec.arch == Arch::Gfx1151 {
        use super::mq_i4_moe_gemv_cn as cn;
        return cn::emit_rowtiles(cn::Spec::new(
            match spec.codec {
                Codec::Mq3I4 => cn::Codec::Mq3I4,
                Codec::Mq4Sym => cn::Codec::Mq4Sym,
            },
            match spec.kind {
                Kind::GateUp => cn::Kind::GateUp,
                Kind::Down => cn::Kind::Down,
            },
        ));
    }
    wide::emit(
        Spec {
            cap: Cap::C8,
            ..spec
        },
        Network::Reference,
    )
}

/// The retained revised network (one `global_load_u16` packet); byte-exact on GPU but slower by measurement, so not production.
pub fn emit_optimized(spec: Spec) -> Result<Emitted, String> {
    emit_with_network(spec, Network::Optimized)
}

/// Packet network of the `Mq3I4` unpack. A compile-time choice of the emitter, never a symbol axis or ABI field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Network {
    /// Two `global_load_u8` (offsets 0, 1) combined by `v_lshl_or_b32`.
    Reference,
    /// One `global_load_u16` at the same byte offset (the same 16 bits, so the same 12 after the shift); retained, not production.
    Optimized,
}

fn emit_with_network(spec: Spec, net: Network) -> Result<Emitted, String> {
    spec.validate()?;
    if spec.cap != Cap::C8 {
        return wide::emit(spec, net);
    }
    let l = Lay::new(spec);
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
        kernel_id: "mq_i4_moe_gemv".into(),
        variant: spec.tag(),
        arch: spec.arch,
        symbol: spec.symbol(),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: WORKGROUP_SIZE as u16,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan(spec.arch, &l)?);
    b.enable_delay_alu();
    prologue(&mut b, spec, &l)?;
    b.wait_all()?;
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
            branch(&mut b, &format!("s_cbranch_scc1 {}", lbl(spec, "tail")))?;
            b.loop_(&lbl(spec, "quad"), |b| {
                for chain in 0..4 {
                    group(b, spec, &l, net, chain, &format!("q{chain}"))?;
                }
                sop(
                    b,
                    format!("s_cmp_lt_u32 s{GI}, s{QUADS}"),
                    &[],
                    &[GI, QUADS],
                )?;
                branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "quad")))
            })?;
            b.label(&lbl(spec, "tail"))?;
            for chain in 0..3 {
                sop(
                    &mut b,
                    format!("s_cmp_lt_u32 s{GI}, s{GROUPS}"),
                    &[],
                    &[GI, GROUPS],
                )?;
                branch(&mut b, &format!("s_cbranch_scc0 {}", lbl(spec, "reduce")))?;
                group(&mut b, spec, &l, net, chain, &format!("t{chain}"))?;
            }
        }
        Kind::Down => {
            sop(&mut b, format!("s_cmp_eq_u32 s{GROUPS}, 0"), &[], &[GROUPS])?;
            branch(&mut b, &format!("s_cbranch_scc1 {}", lbl(spec, "reduce")))?;
            b.loop_(&lbl(spec, "group"), |b| {
                group(b, spec, &l, net, 0, "g")?;
                sop(
                    b,
                    format!("s_cmp_lt_u32 s{GI}, s{GROUPS}"),
                    &[],
                    &[GI, GROUPS],
                )?;
                branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "group")))
            })?;
        }
    }
    b.label(&lbl(spec, "reduce"))?;
    epilogue(&mut b, spec, &l)?;
    b.finish()
}

/// Diagnostic code object containing the four C8 entries and C16/C32 comparisons.
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

fn s_sub_i32(a: Arch) -> &'static str {
    if a.gfx12() {
        "s_sub_co_i32"
    } else {
        "s_sub_i32"
    }
}

/// Kernarg loads, the guards, the expert's pointer and the row/slot bookkeeping, per-lane
/// address constants, zeroed accumulators and the per-slot X offsets / Y addresses.
fn prologue(b: &mut Builder, spec: Spec, l: &Lay) -> R {
    let a = b.spec.arch;
    let exit = lbl(spec, "exit");
    smem(b, WPTRS, 8, KARG, 0)?;
    smem(b, XROT, 8, KARG, 0x20)?;
    smem(b, SU, 4, KARG, 0x40)?;
    smem(b, SXSTR, 2, KARG, 0x50)?;
    smem(b, SNEXP, 1, KARG, 0x58)?;
    // Workgroup ids: gfx12 `ttmp9` (x) and `ttmp7[15:0]` (y); gfx11 the first two system SGPRs after the user SGPRs.
    match a.workgroup_ids(2)? {
        WgIds {
            x: WgIdSource::Ttmp9,
            y: WgIdSource::Ttmp7,
        } => {
            sop(b, format!("s_mov_b32 s{ROW}, ttmp9"), &[ROW], &[])?;
            sop(b, format!("s_and_b32 s{UIDX}, ttmp7, 0xffff"), &[UIDX], &[])?;
        }
        WgIds {
            x: WgIdSource::Sgpr(x),
            y: WgIdSource::Sgpr(y),
        } if (x, y) == (WGX, WGY) => {
            sop(b, format!("s_mov_b32 s{ROW}, s{WGX}"), &[ROW], &[WGX])?;
            sop(b, format!("s_mov_b32 s{UIDX}, s{WGY}"), &[UIDX], &[WGY])?;
        }
        ids => {
            return Err(format!(
                "mq_i4_moe_gemv: unexpected workgroup id sources {ids:?}"
            ))
        }
    }
    sop(b, format!("s_cmp_ge_u32 s{ROW}, s{SM}"), &[], &[ROW, SM])?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    sop(b, format!("s_cmp_ge_u32 s{UIDX}, s{SU}"), &[], &[UIDX, SU])?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    // offsets[u], offsets[u+1] and the expert id; the expert's tensor pointer from the table.
    sop(b, format!("s_lshl_b32 s{TMP}, s{UIDX}, 2"), &[TMP], &[UIDX])?;
    add64(b, TPTR, OFFS, TMP)?;
    smem(b, OFF0, 2, TPTR, 0)?;
    add64(b, TPTR, EIDS, TMP)?;
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
    sop(b, format!("s_mul_i32 s{TMP}, s{SC}, 10"), &[TMP], &[SC])?;
    sop(
        b,
        format!("s_cmp_gt_u32 s{OFF1}, s{TMP}"),
        &[],
        &[OFF1, TMP],
    )?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    // Expert id range check before the table index: an id at or past `n_experts` (unsigned, so a negative id too)
    // exits without a load or a store.
    sop(
        b,
        format!("s_cmp_ge_u32 s{EID}, s{SNEXP}"),
        &[],
        &[EID, SNEXP],
    )?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    sop(b, format!("s_lshl_b32 s{TMP}, s{EID}, 3"), &[TMP], &[EID])?;
    add64(b, TPTR, WPTRS, TMP)?;
    smem(b, WPTR, 2, TPTR, 0)?;
    // Group count, row bytes and the row's code (and, planar, scale) pointers.
    match (spec.codec, spec.kind) {
        (Codec::Mq3I4, _) => {
            sop(
                b,
                format!("s_lshr_b32 s{GROUPS}, s{SK}, 7"),
                &[GROUPS],
                &[SK],
            )?;
            sop(
                b,
                format!("s_lshr_b32 s{ROW_BYTES}, s{SK}, 3"),
                &[ROW_BYTES],
                &[SK],
            )?;
            sop(
                b,
                format!("s_mul_i32 s{ROW_BYTES}, s{ROW_BYTES}, 3"),
                &[ROW_BYTES],
                &[ROW_BYTES],
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
        }
        (Codec::Mq4Sym, kind) => {
            let (shift, stride) = if kind == Kind::GateUp {
                (8, 136)
            } else {
                (7, 68)
            };
            sop(
                b,
                format!("s_lshr_b32 s{GROUPS}, s{SK}, {shift}"),
                &[GROUPS],
                &[SK],
            )?;
            sop(
                b,
                format!("s_mul_i32 s{ROW_BYTES}, s{GROUPS}, {}", lit(stride)),
                &[ROW_BYTES],
                &[GROUPS],
            )?;
            sop(
                b,
                format!("s_mul_i32 s{TMP}, s{ROW}, s{ROW_BYTES}"),
                &[TMP],
                &[ROW, ROW_BYTES],
            )?;
            add64(b, ROW_PTR, WPTR, TMP)?;
        }
    }
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
    // Per-lane constants: x byte offset within a group (16*l, or 32*l for the 8-element lane),
    // the code byte offset of the lane's packet within a group and its extras.
    let xshift = if l.nw == 8 { 5 } else { 4 };
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{}, {xshift}, v{LANE}", l.xl),
        &[l.xl],
        &[LANE],
        &[],
    )?;
    match (spec.codec, spec.kind) {
        (Codec::Mq3I4, _) => {
            // (3l) >> 1 and the W3 packet's bit shift 4*(l & 1).
            valu(
                b,
                &format!("v_mul_u32_u24_e32 v{}, 3, v{LANE}", l.cl),
                &[l.cl],
                &[LANE],
                &[],
            )?;
            valu(
                b,
                &format!("v_lshrrev_b32_e32 v{0}, 1, v{0}", l.cl),
                &[l.cl],
                &[l.cl],
                &[],
            )?;
            valu(
                b,
                &format!("v_and_b32_e32 v{}, 1, v{LANE}", l.shift),
                &[l.shift],
                &[LANE],
                &[],
            )?;
            valu(
                b,
                &format!("v_lshlrev_b32_e32 v{0}, 2, v{0}", l.shift),
                &[l.shift],
                &[l.shift],
                &[],
            )?;
        }
        (Codec::Mq4Sym, Kind::GateUp) => {
            // Payload word 4*l; header half (bytes 0 or 4) 4*(l >> 4).
            valu(
                b,
                &format!("v_lshlrev_b32_e32 v{}, 2, v{LANE}", l.cl),
                &[l.cl],
                &[LANE],
                &[],
            )?;
            valu(
                b,
                &format!("v_lshrrev_b32_e32 v{}, 4, v{LANE}", l.hl),
                &[l.hl],
                &[LANE],
                &[],
            )?;
            valu(
                b,
                &format!("v_lshlrev_b32_e32 v{0}, 2, v{0}", l.hl),
                &[l.hl],
                &[l.hl],
                &[],
            )?;
        }
        (Codec::Mq4Sym, Kind::Down) => valu(
            b,
            &format!("v_lshlrev_b32_e32 v{}, 1, v{LANE}", l.cl),
            &[l.cl],
            &[LANE],
            &[],
        )?,
    }
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
        &format!("v_min_u32_e32 v{p}, s{NSM1}, v{LANE}"),
        &[p],
        &[LANE],
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
    gload(b, "global_load_b32", sid, p, SLOTS, "")?;
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

/// `if n <= i skip to target` (scalar, uniform): SCC = (n > i), branch when clear.
fn slot_gate(b: &mut Builder, i: u32, target: &str) -> R {
    sop(b, format!("s_cmp_gt_u32 s{NS}, {i}"), &[], &[NS])?;
    branch(b, &format!("s_cbranch_scc0 {target}"))
}

/// One group of the row, loaded and dequantized once, then applied to every live slot.
/// Chain `chain` (always 0 for down) takes the group partial of every slot.
fn group(b: &mut Builder, spec: Spec, l: &Lay, net: Network, chain: u8, inst: &str) -> R {
    let arch = b.spec.arch;
    let wide = l.nw == 8;
    // Scalar group offsets, then the lane addresses, then the weight loads (all before
    // any slot branch: slot branches never re-read weights).
    match spec.codec {
        Codec::Mq3I4 => {
            sop(
                b,
                format!("s_mul_i32 s{G_CODE}, s{GI}, 48"),
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
        }
        Codec::Mq4Sym => {
            sop(
                b,
                format!(
                    "s_mul_i32 s{G_CODE}, s{GI}, {}",
                    lit(if wide { 136 } else { 68 })
                ),
                &[G_CODE],
                &[GI],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{G_X}, s{GI}, {}", if wide { 10 } else { 9 }),
                &[G_X],
                &[GI],
            )?;
        }
    }
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{}, s{G_CODE}, v{}", l.caddr, l.cl),
        &[l.caddr],
        &[l.cl],
        &[G_CODE],
    )?;
    match (spec.codec, spec.kind) {
        (Codec::Mq3I4, _) => valu(
            b,
            &format!("v_mov_b32_e32 v{}, s{G_SCALE}", l.saddr),
            &[l.saddr],
            &[],
            &[G_SCALE],
        )?,
        (Codec::Mq4Sym, Kind::GateUp) => valu(
            b,
            &format!("v_add_nc_u32_e32 v{}, s{G_CODE}, v{}", l.saddr, l.hl),
            &[l.saddr],
            &[l.hl],
            &[G_CODE],
        )?,
        (Codec::Mq4Sym, Kind::Down) => valu(
            b,
            &format!("v_mov_b32_e32 v{}, s{G_CODE}", l.saddr),
            &[l.saddr],
            &[],
            &[G_CODE],
        )?,
    }
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{}, s{G_X}, v{}", l.xb, l.xl),
        &[l.xb],
        &[l.xl],
        &[G_X],
    )?;
    match (spec.codec, spec.kind) {
        (Codec::Mq3I4, _) => {
            match net {
                Network::Reference => {
                    gload(b, "global_load_u8", l.pkt0, l.caddr, ROW_PTR, "")?;
                    gload(b, "global_load_u8", l.pkt1, l.caddr, ROW_PTR, " offset:1")?;
                }
                // One unaligned u16 at the same byte offset: bytes (3l)>>1 and +1 never leave the 48-byte group.
                Network::Optimized => gload(b, "global_load_u16", l.pkt0, l.caddr, ROW_PTR, "")?,
            }
            gload(b, "global_load_u16", l.d, l.saddr, SCALE_PTR, "")?;
        }
        (Codec::Mq4Sym, Kind::GateUp) => {
            gload(b, "global_load_b32", l.pkt0, l.caddr, ROW_PTR, " offset:8")?;
            gload(b, "global_load_b32", l.z, l.saddr, ROW_PTR, "")?;
        }
        (Codec::Mq4Sym, Kind::Down) => {
            gload(b, "global_load_u16", l.pkt0, l.caddr, ROW_PTR, " offset:4")?;
            gload(b, "global_load_b32", l.z, l.saddr, ROW_PTR, "")?;
        }
    }
    // X loads of the live slots (slot 0 always: n >= 1), each slot's own X row.
    let xdone = lbl(spec, &format!("{inst}_xdone"));
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
        if wide {
            bload(b, 4, l.xr(i) + 4, l.xa, SRDX, 16)?
        }
    }
    b.label(&xdone)?;
    b.wait_all()?;
    unpack(b, spec, l, net)?;
    // Per-slot chains on the shared weights.
    let done = lbl(spec, &format!("{inst}_done"));
    for i in 0..MAX_SLOTS {
        if i > 0 {
            slot_gate(b, i, &done)?
        }
        let dot = l.dot + (i % 2) as u8;
        let x = l.xr(i);
        // Mq3I4: p = RN(w0*x0), then fma(w_j, x_j, p) for j = 1..; Mq4Sym (the production contraction of the
        // old MQ4 kernels): p = RN(w1*x1), then fma(w0, x0, p), then fma(w_j, x_j, p) for j = 2.. ascending.
        let first = if spec.codec == Codec::Mq4Sym { 1 } else { 0 };
        valu(
            b,
            &format!("v_mul_f32_e32 v{dot}, v{}, v{}", l.w + first, x + first),
            &[dot],
            &[l.w + first, x + first],
            &[],
        )?;
        if first == 1 {
            valu(
                b,
                &format!("v_fmac_f32_e32 v{dot}, v{}, v{x}", l.w),
                &[dot],
                &[dot, l.w, x],
                &[],
            )?
        }
        for j in 2 - u8::from(first == 0)..l.nw {
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

/// The simple exact unpack: the lane's packet to `nw` dequantized weights `l.w ..`
/// (the single seam a later optimized network replaces).
///  * `Mq3I4`: `d = float(f16)`, packet `b0 | b1 << 8` shifted by `4*(l&1)`, per code
///    `w = RN(d * float(2*bfe(packet, 3i, 3) - 7))`.
///  * `Mq4Sym`: `d`, `z` the header's f16 halves converted exactly, per nibble
///    `w = fma(d, float(bfe(packet, 4i, 4)), z)`.
fn unpack(b: &mut Builder, spec: Spec, l: &Lay, net: Network) -> R {
    match spec.codec {
        Codec::Mq3I4 => {
            valu(
                b,
                &format!("v_cvt_f32_f16_e64 v{0}, v{0}.l", l.d),
                &[l.d],
                &[l.d],
                &[],
            )?;
            // Reference network: the packet is two byte loads combined here; optimized: one u16 load already holds b0 | b1 << 8.
            if net == Network::Reference {
                valu(
                    b,
                    &format!("v_lshl_or_b32 v{0}, v{1}, 8, v{0}", l.pkt0, l.pkt1),
                    &[l.pkt0],
                    &[l.pkt1, l.pkt0],
                    &[],
                )?;
            }
            valu(
                b,
                &format!("v_lshrrev_b32_e32 v{0}, v{1}, v{0}", l.pkt0, l.shift),
                &[l.pkt0],
                &[l.shift, l.pkt0],
                &[],
            )?;
            for i in 0..4u8 {
                let w = l.w + i;
                valu(
                    b,
                    &format!("v_bfe_u32 v{w}, v{}, {}, 3", l.pkt0, u32::from(i) * 3),
                    &[w],
                    &[l.pkt0],
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
                    &format!("v_add_nc_u32_e32 v{w}, -7, v{w}"),
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
        }
        Codec::Mq4Sym => {
            valu(
                b,
                &format!("v_cvt_f32_f16_e64 v{}, v{}.l", l.d, l.z),
                &[l.d],
                &[l.z],
                &[],
            )?;
            valu(
                b,
                &format!("v_lshrrev_b32_e32 v{0}, 16, v{0}", l.z),
                &[l.z],
                &[l.z],
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
                let w = l.w + i;
                valu(
                    b,
                    &format!("v_bfe_u32 v{w}, v{}, {}, 4", l.pkt0, u32::from(i) * 4),
                    &[w],
                    &[l.pkt0],
                    &[],
                )?;
                valu(b, &format!("v_cvt_f32_u32_e32 v{w}, v{w}"), &[w], &[w], &[])?;
                valu(
                    b,
                    &format!("v_fma_f32 v{w}, v{}, v{w}, v{}", l.d, l.z),
                    &[w],
                    &[l.d, w, l.z],
                    &[],
                )?;
            }
        }
    }
    Ok(())
}

/// Shuffle tables, the per-slot fold + snapshot shuffle-down tree, then the lane-0 stores.
fn epilogue(b: &mut Builder, spec: Spec, l: &Lay) -> R {
    // Shuffle step k (offset 16 >> k): mask lane < 32 - off, address (lane + off or lane) * 4.
    let offsets = [16u32, 8, 4, 2, 1];
    for (k, off) in offsets.into_iter().enumerate() {
        let (m, sh) = (SHM + k as u8, l.shuf + k as u8);
        op(
            b,
            format!("v_cmp_gt_u32_e64 s{m}, {}, v{LANE}", 32 - off),
            &[s(m)],
            &[v(LANE)],
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
            &format!("v_add_lshl_u32 v{sh}, v{sh}, v{LANE}, 2"),
            &[sh],
            &[sh, LANE],
            &[],
        )?;
    }
    let red_done = lbl(spec, "red_done");
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
        &format!("v_cmpx_eq_u32_e32 0, v{LANE}"),
        &[],
        &[LANE],
        &[],
    )?;
    let st_done = lbl(spec, "st_done");
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
    b.label(&lbl(spec, "exit"))?;
    branch(b, "s_endpgm")
}

/// The shape contract of one emitted entry: counts of its characteristic
/// instructions, the forbidden families, honest register ceilings (accumulators, X registers and, for the
/// narrow entries, per-slot offsets/addresses are inside
/// `vgpr_max`; the scalar bookkeeping is `sgpr_max`), wave32, no spills, no private
/// memory. The narrow entries use no LDS and no barrier; the wide twins declare their static X tile
/// (`group_segment_fixed_bytes`) and may use LDS and barriers (`ds_swizzle`, `flat_*` stay forbidden).
/// Committed as `kernels/mq_i4_moe_gemv.<arch>.<tag>.contract.json`. The forbidden wait
/// family is the other architecture's: `s_waitcnt*` on gfx12, the gfx12 counter waits on gfx11.
pub fn contract(spec: Spec, e: &Emitted) -> serde_json::Value {
    contract_impl(spec, e, false)
}

/// Contract for [`emit_rowtiles`], including its architecture-local symbol and
/// fixed-eight-slot LDS resources rather than the legacy `cap` schedule.
pub fn contract_rowtiles(spec: Spec, e: &Emitted) -> serde_json::Value {
    contract_impl(spec, e, true)
}

fn contract_impl(spec: Spec, e: &Emitted, rowtiles: bool) -> serde_json::Value {
    if rowtiles && spec.arch == Arch::Gfx1151 {
        use super::mq_i4_moe_gemv_cn as cn;
        return cn::contract_rowtiles(
            cn::Spec::new(
                match spec.codec {
                    Codec::Mq3I4 => cn::Codec::Mq3I4,
                    Codec::Mq4Sym => cn::Codec::Mq4Sym,
                },
                match spec.kind {
                    Kind::GateUp => cn::Kind::GateUp,
                    Kind::Down => cn::Kind::Down,
                },
            ),
            e,
        );
    }
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
    // Narrow entries: the committed list, in its committed order; the wide twins use LDS and barriers.
    let narrow = spec.cap == Cap::C8 && !rowtiles;
    let mut forbidden = vec!["scratch_*", "flat_*"];
    if narrow {
        forbidden.extend(["ds_load*", "ds_store*"])
    }
    forbidden.push("ds_swizzle*");
    if narrow {
        forbidden.extend(["ds_read*", "ds_write*", "s_barrier*"])
    }
    forbidden.extend(["v_writelane*", "v_wmma*", "v_swmmac*", "v_fma_mix*"]);
    if spec.arch.gfx12() {
        forbidden.push("s_waitcnt*")
    } else {
        forbidden.extend([
            "s_wait_loadcnt*",
            "s_wait_storecnt*",
            "s_wait_dscnt*",
            "s_wait_kmcnt*",
            "s_wait_alu*",
        ])
    }
    forbidden.extend(["s_setpc*", "s_swappc*"]);
    serde_json::json!({
        "symbol": if rowtiles { format!("mq_i4_moe_gemv_{}_{}_t8", spec.codec.tag(), spec.kind.tag()) } else { spec.symbol() }, "counts": counts,
        "forbidden": forbidden,
        "vgpr_max": e.shape.next_free_vgpr, "sgpr_max": e.shape.next_free_sgpr,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0,
        "group_segment_fixed_bytes": if rowtiles { wide::SLOT_TILE * 32 * 4 * u32::from(spec.lane_elems()) } else { spec.lds_bytes() },
    })
}
const COUNTED: [&str; 35] = [
    "s_load_b256",
    "s_load_b128",
    "s_load_b64",
    "s_load_b32",
    "global_load_u8",
    "global_load_u16",
    "global_load_b32",
    "buffer_load_b128",
    "global_store_b32",
    "ds_bpermute_b32",
    "v_readlane_b32",
    "v_cvt_f32_f16_e64",
    "v_cvt_f32_i32_e32",
    "v_cvt_f32_u32_e32",
    "v_bfe_u32",
    "v_lshl_or_b32",
    "v_mul_f32_e32",
    "v_fma_f32",
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
    "ds_load_b128",
    "ds_store_b128",
    "s_barrier",
    "s_barrier_signal",
    "s_barrier_wait",
    "v_readfirstlane_b32",
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
