// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! K1 decode GEMV `mq_i4_moe_gemv` (gfx1201, gfx1151): mixed-codec F4 lane-union planner,
//! frozen 96-byte kernarg ABI and the real per-bucket launch.
//!
//! The module is deliberately independent: it depends only on `std`,
//! `hip-bridge` and `rdna-compute` (never on the loader, the sealed route
//! files or `pipeline/mod.rs`), so it compiles unchanged both registered in
//! `pipeline` and as a `#[path]` module of an integration test.
//!
//! # Contract (W3-GPU-PLAN K1, DESIGN §4.2)
//!
//! * Codecs: [`Codec::Mq3I4`] (qt 131 gate_up R256 / qt 132 down R128) and
//!   [`Codec::Mq4Sym`] (qt 44 / qt 53). Anything else is refused; there is no
//!   representative-dtype fallback.
//! * Kernel symbols (one per codec x kind, fixed): see [`BucketKey::symbol`].
//!   Workgroup 32, grid `x = M` output rows (gate/up `M = 1280`: rows
//!   `0..640` gate -> `Y0`, `640..1280` up -> `Y1`; down `M = 2560`),
//!   `y = U_b` distinct experts of the bucket.
//! * Descriptor (device-owned, immutable during the launch, prepared by the
//!   caller; the host never reads routing back):
//!   * `expert_ids[U]` u32, concatenated bucket after bucket;
//!   * `expert_offsets[U + B]` u32: one `U_b + 1` run per non-empty bucket,
//!     each starting at `0` and ending at that bucket's slot count (offsets
//!     span only the bucket's own slots);
//!   * `slot_ids[10 C]` u32, concatenated bucket after bucket, slot =
//!     `lane * 10 + rank` (stable original order inside every expert).
//!   * `weight_ptrs[512]` u64 per projection tensor: the expert's code-plane
//!     base, indexed by **expert id**. There is no uniform stride.
//! * Buckets use the fixed codec order [`Codec::ORDER`]; the sorted compact
//!   [`LaneUnionPlan`] lists expert ids strictly increasing inside a bucket,
//!   whereas the production device planner (modelled by [`FixedUnion`]) lists
//!   them in first-occurrence order in fixed-capacity `6 S + 2` word arrays.
//!   Both partition the `10 C` original slots. `U <= min(10 C, 512)`,
//!   `C in 1..=32` (`S = 10 C <= 320`), ids in `0..512`, unique per lane.
//! * Outputs are indexed by the **original slot**. Down writes the unweighted
//!   `Y0 [C, 10, 2560]`; the existing combine stage alone applies the route
//!   weights. `X` row is `slot / x_row_div` (gate 10, down 1).
//!
//! [`LaneUnionPlan`] is the CPU oracle of that descriptor. In production the
//! caller owns device arrays with the same layout and describes them to
//! [`SlabMoeDecodeModule::launch`] through a [`UnionLayout`] (counts only).

use hip_bridge::{Function, HipError, HipResult, Module};
use rdna_compute::{Gpu, GpuTensor};
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Routed experts per lane (token).
pub const TOP_K: usize = 10;
/// Largest lane count of the decode union (`C32`: wide verify). Capacity and
/// CPU validation only: per-architecture admission of `C > 8` is a separate
/// route decision.
pub const MAX_LANES: usize = 32;
/// Exclusive upper bound of expert ids (`0..512`).
pub const MAX_EXPERTS: usize = 512;
/// Gate/up output rows of the fused projection (gate `640` + up `640`).
pub const GATE_UP_M: usize = 1280;
/// Gate/up reduction width.
pub const GATE_UP_K: usize = 2560;
/// Rows of each of gate and up.
pub const GATE_UP_OUT_ROWS: usize = GATE_UP_M / 2;
/// Down output rows.
pub const DOWN_M: usize = 2560;
/// Down reduction width.
pub const DOWN_K: usize = 640;
/// Threads per workgroup of every K1 decode kernel (one wave32).
pub const WORKGROUP: u32 = 32;
/// Size of the frozen F4 GEMV kernarg.
pub const KERNARG_BYTES: usize = 96;
const F32_BYTES: usize = 4;
const U32_BYTES: usize = 4;
const PTR_BYTES: usize = 8;

/// `U <= min(10 C, 512)`: the largest distinct-expert count of `c` lanes.
pub const fn max_union(c: usize) -> usize {
    let slots = TOP_K * c;
    if slots < MAX_EXPERTS {
        slots
    } else {
        MAX_EXPERTS
    }
}

// ---------------------------------------------------------------------------
// Typed axes
// ---------------------------------------------------------------------------

/// Weight codec of an expert. Compile-time symbol axis, never a runtime enum
/// handed to a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Codec {
    /// W3 planar MQ3-I4 (qt 131 gate_up, qt 132 down).
    Mq3I4,
    /// Pointer-addressed symmetric MQ4 (qt 44 gate_up, qt 53 down).
    Mq4Sym,
}

impl Codec {
    /// Fixed bucket order of the union descriptor.
    pub const ORDER: [Codec; 2] = [Codec::Mq3I4, Codec::Mq4Sym];

    /// Position in [`Codec::ORDER`].
    pub const fn order_index(self) -> usize {
        match self {
            Codec::Mq3I4 => 0,
            Codec::Mq4Sym => 1,
        }
    }

    /// Codec of one projection's HFQ quant type. `qt` of the other kind, a W2
    /// type (133/134), a legacy aggregate type or anything unknown is refused.
    pub fn from_qt(kind: Kind, qt: u8) -> Result<Codec, SlabMoeError> {
        match (kind, qt) {
            (Kind::GateUp, 131) | (Kind::Down, 132) => Ok(Codec::Mq3I4),
            (Kind::GateUp, 44) | (Kind::Down, 53) => Ok(Codec::Mq4Sym),
            _ => Err(SlabMoeError::UnsupportedQt { kind, qt }),
        }
    }

    /// Codec of an expert's `(gate_up_qt, down_qt)` pair: `(131, 132)` or
    /// `(44, 53)` only.
    pub fn from_qt_pair(gate_up_qt: u8, down_qt: u8) -> Result<Codec, SlabMoeError> {
        let gate = Codec::from_qt(Kind::GateUp, gate_up_qt)?;
        let down = Codec::from_qt(Kind::Down, down_qt)?;
        if gate == down {
            Ok(gate)
        } else {
            Err(SlabMoeError::QtPairMismatch {
                gate_up_qt,
                down_qt,
            })
        }
    }
}

/// Rotation profile of a projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Profile {
    /// 256-wide rotation (gate/up, K2560).
    R256,
    /// 128-wide row-local rotation (down, K640).
    R128,
}

impl Profile {
    /// Rotation width.
    pub const fn width(self) -> usize {
        match self {
            Profile::R256 => 256,
            Profile::R128 => 128,
        }
    }
}

/// Expert projection a launch computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// Fused gate/up, `[1280, 2560]`, R256.
    GateUp,
    /// Down, `[2560, 640]`, R128.
    Down,
}

impl Kind {
    /// The only profile the kind admits.
    pub const fn profile(self) -> Profile {
        match self {
            Kind::GateUp => Profile::R256,
            Kind::Down => Profile::R128,
        }
    }

    /// Output rows `M` (kernel grid x).
    pub const fn m(self) -> usize {
        match self {
            Kind::GateUp => GATE_UP_M,
            Kind::Down => DOWN_M,
        }
    }

    /// Reduction width `K`.
    pub const fn k(self) -> usize {
        match self {
            Kind::GateUp => GATE_UP_K,
            Kind::Down => DOWN_K,
        }
    }

    /// `X` row of slot `s` is `s / x_row_div`: gate 10 (`X[C, K]`), down 1
    /// (`X[10 C, K]`).
    pub const fn x_row_div(self) -> usize {
        match self {
            Kind::GateUp => TOP_K,
            Kind::Down => 1,
        }
    }

    /// F32 elements between consecutive original slots of each output.
    pub const fn y_slot_stride(self) -> usize {
        match self {
            Kind::GateUp => GATE_UP_OUT_ROWS,
            Kind::Down => DOWN_M,
        }
    }

    /// `X` rows for `c` lanes (`x_src_rows`).
    pub const fn x_src_rows(self, c: usize) -> usize {
        match self {
            Kind::GateUp => c,
            Kind::Down => TOP_K * c,
        }
    }

    /// Whether the launch writes a second output (`Y1` = up).
    pub const fn has_y1(self) -> bool {
        matches!(self, Kind::GateUp)
    }
}

/// One `(codec, profile)` bucket identity, in the fixed bucket order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BucketKey {
    pub codec: Codec,
    pub profile: Profile,
}

impl BucketKey {
    /// Fixed `(codec, profile)` order: all of a codec's profiles, codecs in
    /// [`Codec::ORDER`].
    pub const ORDER: [BucketKey; 4] = [
        BucketKey {
            codec: Codec::Mq3I4,
            profile: Profile::R256,
        },
        BucketKey {
            codec: Codec::Mq3I4,
            profile: Profile::R128,
        },
        BucketKey {
            codec: Codec::Mq4Sym,
            profile: Profile::R256,
        },
        BucketKey {
            codec: Codec::Mq4Sym,
            profile: Profile::R128,
        },
    ];

    /// Key of `codec` for the projection `kind`.
    pub const fn new(codec: Codec, kind: Kind) -> Self {
        BucketKey {
            codec,
            profile: kind.profile(),
        }
    }

    /// Projection this key computes (profile determines kind).
    pub const fn kind(self) -> Kind {
        match self.profile {
            Profile::R256 => Kind::GateUp,
            Profile::R128 => Kind::Down,
        }
    }

    /// Position in [`BucketKey::ORDER`].
    pub const fn order_index(self) -> usize {
        self.codec.order_index() * 2
            + match self.profile {
                Profile::R256 => 0,
                Profile::R128 => 1,
            }
    }

    /// The fixed kernel symbol of the key.
    pub const fn symbol(self) -> &'static str {
        match (self.codec, self.profile) {
            (Codec::Mq3I4, Profile::R256) => "mq_i4_moe_gemv_mq3i4_r256_gate_up",
            (Codec::Mq3I4, Profile::R128) => "mq_i4_moe_gemv_mq3i4_r128_down",
            (Codec::Mq4Sym, Profile::R256) => "mq_i4_moe_gemv_mq4sym_r256_gate_up",
            (Codec::Mq4Sym, Profile::R128) => "mq_i4_moe_gemv_mq4sym_r128_down",
        }
    }
}

/// Every K1 decode symbol a code object must export.
pub const SYMBOLS: [&str; 4] = [
    BucketKey::ORDER[0].symbol(),
    BucketKey::ORDER[1].symbol(),
    BucketKey::ORDER[2].symbol(),
    BucketKey::ORDER[3].symbol(),
];

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Planner, admission and launch refusal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SlabMoeError {
    /// Lane count outside `1..=32`.
    InvalidLaneCount { c: usize },
    /// Top-k id outside `0..512`.
    ExpertOutOfRange { lane: usize, rank: usize, id: u32 },
    /// The same expert twice in one lane.
    DuplicateExpert { lane: usize, id: u32 },
    /// A selected expert has no declared codec.
    MissingCodec { expert: u32 },
    /// A quant type that is not an admitted MQ-I4/MQ4 projection type.
    UnsupportedQt { kind: Kind, qt: u8 },
    /// `(gate_up_qt, down_qt)` of different codecs or an unadmitted pair.
    QtPairMismatch { gate_up_qt: u8, down_qt: u8 },
    /// Profile does not belong to the kind.
    ProfileMismatch { kind: Kind, profile: Profile },
    /// Explicit `M`/`K` differ from the kind's frozen geometry.
    Geometry {
        what: &'static str,
        got: usize,
        want: usize,
    },
    /// Layout violates the union contract.
    Layout(&'static str),
    /// An operand is null, misaligned, too small or aliases another operand.
    Operand {
        name: &'static str,
        reason: &'static str,
    },
    /// Device is neither gfx1201 nor gfx1151.
    Arch(String),
    /// The launch cannot be issued in this execution mode.
    Unavailable(&'static str),
}

impl fmt::Display for SlabMoeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLaneCount { c } => write!(f, "lane count {c} not in 1..={MAX_LANES}"),
            Self::ExpertOutOfRange { lane, rank, id } => {
                write!(
                    f,
                    "lane {lane} rank {rank}: expert id {id} not in 0..{MAX_EXPERTS}"
                )
            }
            Self::DuplicateExpert { lane, id } => {
                write!(f, "lane {lane}: expert {id} selected twice")
            }
            Self::MissingCodec { expert } => write!(f, "expert {expert} has no declared codec"),
            Self::UnsupportedQt { kind, qt } => {
                write!(f, "qt {qt} is not an admitted {kind:?} type")
            }
            Self::QtPairMismatch {
                gate_up_qt,
                down_qt,
            } => {
                write!(
                    f,
                    "qt pair ({gate_up_qt}, {down_qt}) is not (131, 132) or (44, 53)"
                )
            }
            Self::ProfileMismatch { kind, profile } => {
                write!(f, "{kind:?} does not admit profile {profile:?}")
            }
            Self::Geometry { what, got, want } => write!(f, "{what} {got} != frozen {want}"),
            Self::Layout(why) => write!(f, "union layout: {why}"),
            Self::Operand { name, reason } => write!(f, "operand {name}: {reason}"),
            Self::Arch(got) => write!(
                f,
                "mq_i4_moe_gemv is exact gfx1201/gfx1151 only, device is {got:?}"
            ),
            Self::Unavailable(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for SlabMoeError {}

fn hip_err(e: SlabMoeError) -> HipError {
    HipError::new(1, &format!("slab_moe_decode: {e}"))
}

// ---------------------------------------------------------------------------
// Frozen kernarg ABI
// ---------------------------------------------------------------------------

/// The frozen F4 GEMV kernarg, 96 bytes. Strides are F32 elements.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MoeGemvKernargs {
    /// `@0` per-projection table of expert code-plane bases, indexed by expert id.
    pub expert_weight_ptrs: u64,
    /// `@8` u32 expert ids of this bucket (distinct; order is the producer's:
    /// ascending in the compact CPU plan, first-occurrence on device).
    pub expert_ids: u64,
    /// `@16` u32 offsets, `U + 1` entries spanning only this bucket's slots.
    pub expert_offsets: u64,
    /// `@24` u32 original slot ids of this bucket.
    pub slot_ids: u64,
    /// `@32` rotated F32 activations.
    pub xrot: u64,
    /// `@40` gate (gate/up) or unweighted down output.
    pub y0: u64,
    /// `@48` up output; `0` for down.
    pub y1: u64,
    /// `@56` output rows.
    pub m: u32,
    /// `@60` reduction width.
    pub k: u32,
    /// `@64` distinct experts of this bucket.
    pub u: u32,
    /// `@68` lanes.
    pub c: u32,
    /// `@72` `X` row is `slot / x_row_div`.
    pub x_row_div: u32,
    /// `@76` `X` rows available.
    pub x_src_rows: u32,
    /// `@80` F32 elements between `X` rows.
    pub x_row_stride: u32,
    /// `@84` F32 elements between original-slot output rows.
    pub y_slot_stride: u32,
    /// `@88` extent of the `expert_weight_ptrs` table; an `expert_ids` entry at or past it is never indexed.
    pub n_experts: u32,
    /// `@92` reserved, always zero (pads the block to 8 bytes).
    pub reserved: u32,
}

const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(size_of::<MoeGemvKernargs>() == KERNARG_BYTES);
    assert!(align_of::<MoeGemvKernargs>() == 8);
    assert!(offset_of!(MoeGemvKernargs, expert_weight_ptrs) == 0);
    assert!(offset_of!(MoeGemvKernargs, expert_ids) == 8);
    assert!(offset_of!(MoeGemvKernargs, expert_offsets) == 16);
    assert!(offset_of!(MoeGemvKernargs, slot_ids) == 24);
    assert!(offset_of!(MoeGemvKernargs, xrot) == 32);
    assert!(offset_of!(MoeGemvKernargs, y0) == 40);
    assert!(offset_of!(MoeGemvKernargs, y1) == 48);
    assert!(offset_of!(MoeGemvKernargs, m) == 56);
    assert!(offset_of!(MoeGemvKernargs, k) == 60);
    assert!(offset_of!(MoeGemvKernargs, u) == 64);
    assert!(offset_of!(MoeGemvKernargs, c) == 68);
    assert!(offset_of!(MoeGemvKernargs, x_row_div) == 72);
    assert!(offset_of!(MoeGemvKernargs, x_src_rows) == 76);
    assert!(offset_of!(MoeGemvKernargs, x_row_stride) == 80);
    assert!(offset_of!(MoeGemvKernargs, y_slot_stride) == 84);
    assert!(offset_of!(MoeGemvKernargs, n_experts) == 88);
    assert!(offset_of!(MoeGemvKernargs, reserved) == 92);
};

impl MoeGemvKernargs {
    /// The exact little-endian kernarg bytes (all fields are naturally
    /// aligned; there is no padding).
    pub fn to_bytes(&self) -> [u8; KERNARG_BYTES] {
        let mut out = [0u8; KERNARG_BYTES];
        let ptrs = [
            self.expert_weight_ptrs,
            self.expert_ids,
            self.expert_offsets,
            self.slot_ids,
            self.xrot,
            self.y0,
            self.y1,
        ];
        for (i, p) in ptrs.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&p.to_le_bytes());
        }
        let words = [
            self.m,
            self.k,
            self.u,
            self.c,
            self.x_row_div,
            self.x_src_rows,
            self.x_row_stride,
            self.y_slot_stride,
            self.n_experts,
            self.reserved,
        ];
        for (i, w) in words.iter().enumerate() {
            out[56 + i * 4..56 + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Union layout and CPU oracle planner
// ---------------------------------------------------------------------------

/// Counts and bases of one non-empty codec bucket inside the concatenated
/// descriptor arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bucket {
    pub codec: Codec,
    /// First entry of the bucket in `expert_ids`.
    pub expert_start: usize,
    /// Distinct experts `U_b >= 1`.
    pub expert_count: usize,
    /// First entry of the bucket's `U_b + 1` offsets run in `expert_offsets`.
    pub offsets_start: usize,
    /// First entry of the bucket in `slot_ids`.
    pub slot_start: usize,
    /// Slots `n_b >= 1` (`expert_offsets[offsets_start + U_b]`).
    pub slot_count: usize,
}

/// Host-known shape of a device-owned union descriptor: counts only, no
/// routing data. Both gate/up and down launches of one step share it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnionLayout {
    /// Lanes `C in 1..=32`.
    pub c: usize,
    /// Total distinct experts `U <= min(10 C, 512)`.
    pub u_total: usize,
    /// Non-empty buckets in [`Codec::ORDER`].
    pub buckets: Vec<Bucket>,
}

impl UnionLayout {
    /// Entries of `expert_ids` (`U`).
    pub fn expert_id_words(&self) -> usize {
        self.u_total
    }

    /// Entries of `expert_offsets` (`U + B`).
    pub fn expert_offset_words(&self) -> usize {
        self.u_total + self.buckets.len()
    }

    /// Entries of `slot_ids` (`10 C`).
    pub fn slot_words(&self) -> usize {
        TOP_K * self.c
    }

    /// Validates the structural union contract: `C`, `U` bound, fixed bucket
    /// order without repeats or empties, contiguous spans, and a partition of
    /// the `10 C` slots.
    pub fn validate(&self) -> Result<(), SlabMoeError> {
        if !(1..=MAX_LANES).contains(&self.c) {
            return Err(SlabMoeError::InvalidLaneCount { c: self.c });
        }
        if self.buckets.is_empty() || self.buckets.len() > Codec::ORDER.len() {
            return Err(SlabMoeError::Layout("bucket count outside 1..=codec count"));
        }
        let (mut e, mut s, mut prev) = (0usize, 0usize, None::<usize>);
        for (i, b) in self.buckets.iter().enumerate() {
            if prev.is_some_and(|p| b.codec.order_index() <= p) {
                return Err(SlabMoeError::Layout(
                    "buckets not in fixed strictly increasing codec order",
                ));
            }
            prev = Some(b.codec.order_index());
            if b.expert_count == 0 || b.slot_count == 0 || b.expert_count > b.slot_count {
                return Err(SlabMoeError::Layout(
                    "bucket must be non-empty with 1 <= U_b <= n_b",
                ));
            }
            if b.expert_start != e || b.slot_start != s || b.offsets_start != e + i {
                return Err(SlabMoeError::Layout(
                    "bucket spans are not contiguous in descriptor order",
                ));
            }
            e = e
                .checked_add(b.expert_count)
                .ok_or(SlabMoeError::Layout("expert span overflows"))?;
            s = s
                .checked_add(b.slot_count)
                .ok_or(SlabMoeError::Layout("slot span overflows"))?;
        }
        if e != self.u_total {
            return Err(SlabMoeError::Layout(
                "u_total differs from the bucket expert counts",
            ));
        }
        if self.u_total > max_union(self.c) {
            return Err(SlabMoeError::Layout("U exceeds min(10 C, 512)"));
        }
        if s != TOP_K * self.c {
            return Err(SlabMoeError::Layout(
                "buckets do not partition the 10 C slots",
            ));
        }
        Ok(())
    }
}

/// CPU oracle of the device union descriptor. Every vector has exactly the
/// layout a device array must have (see the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneUnionPlan {
    pub layout: UnionLayout,
    /// `[U]`, buckets in codec order, ids strictly increasing inside a bucket.
    pub expert_ids: Vec<u32>,
    /// `[U + B]`, per bucket `U_b + 1` offsets from `0` to `n_b`.
    pub expert_offsets: Vec<u32>,
    /// `[10 C]`, original slot ids; ascending inside each expert.
    pub slot_ids: Vec<u32>,
    /// `inverse[slot]` = position of `slot` in `slot_ids`.
    pub inverse: Vec<u32>,
}

impl LaneUnionPlan {
    /// Plans the union of `top_ids` (one `[u32; 10]` per lane, rank order).
    ///
    /// `expert_codecs[e]` is expert `e`'s declared codec (`None` = undeclared;
    /// selecting it is refused). Ids must be `< 512`, unique per lane.
    pub fn build(
        expert_codecs: &[Option<Codec>],
        top_ids: &[[u32; TOP_K]],
    ) -> Result<Self, SlabMoeError> {
        let c = top_ids.len();
        if !(1..=MAX_LANES).contains(&c) {
            return Err(SlabMoeError::InvalidLaneCount { c });
        }
        let mut slots_of: Vec<Vec<u32>> = vec![Vec::new(); MAX_EXPERTS];
        for (lane, ids) in top_ids.iter().enumerate() {
            let mut seen = [0u64; MAX_EXPERTS / 64];
            for (rank, &id) in ids.iter().enumerate() {
                if id as usize >= MAX_EXPERTS {
                    return Err(SlabMoeError::ExpertOutOfRange { lane, rank, id });
                }
                let (word, bit) = (id as usize / 64, 1u64 << (id % 64));
                if seen[word] & bit != 0 {
                    return Err(SlabMoeError::DuplicateExpert { lane, id });
                }
                seen[word] |= bit;
                if expert_codecs.get(id as usize).copied().flatten().is_none() {
                    return Err(SlabMoeError::MissingCodec { expert: id });
                }
                // Lane-major, rank-minor iteration appends increasing slots.
                slots_of[id as usize].push((lane * TOP_K + rank) as u32);
            }
        }

        let mut buckets = Vec::new();
        let mut expert_ids = Vec::new();
        let mut expert_offsets = Vec::new();
        let mut slot_ids = Vec::new();
        for codec in Codec::ORDER {
            let expert_start = expert_ids.len();
            let offsets_start = expert_offsets.len();
            let slot_start = slot_ids.len();
            expert_offsets.push(0u32);
            for (id, slots) in slots_of.iter().enumerate() {
                if slots.is_empty() || expert_codecs[id] != Some(codec) {
                    continue;
                }
                expert_ids.push(id as u32);
                slot_ids.extend_from_slice(slots);
                expert_offsets.push((slot_ids.len() - slot_start) as u32);
            }
            let expert_count = expert_ids.len() - expert_start;
            if expert_count == 0 {
                expert_offsets.pop();
                continue;
            }
            buckets.push(Bucket {
                codec,
                expert_start,
                expert_count,
                offsets_start,
                slot_start,
                slot_count: slot_ids.len() - slot_start,
            });
        }

        let mut inverse = vec![0u32; TOP_K * c];
        for (pos, &slot) in slot_ids.iter().enumerate() {
            inverse[slot as usize] = pos as u32;
        }
        let layout = UnionLayout {
            c,
            u_total: expert_ids.len(),
            buckets,
        };
        layout.validate()?;
        Ok(Self {
            layout,
            expert_ids,
            expert_offsets,
            slot_ids,
            inverse,
        })
    }

    /// Expert ids of bucket `b`.
    pub fn bucket_experts(&self, b: usize) -> &[u32] {
        let bk = &self.layout.buckets[b];
        &self.expert_ids[bk.expert_start..bk.expert_start + bk.expert_count]
    }

    /// The `U_b + 1` offsets of bucket `b`.
    pub fn bucket_offsets(&self, b: usize) -> &[u32] {
        let bk = &self.layout.buckets[b];
        &self.expert_offsets[bk.offsets_start..bk.offsets_start + bk.expert_count + 1]
    }

    /// Slot ids of bucket `b`.
    pub fn bucket_slots(&self, b: usize) -> &[u32] {
        let bk = &self.layout.buckets[b];
        &self.slot_ids[bk.slot_start..bk.slot_start + bk.slot_count]
    }

    /// Slots owned by the `u`-th expert of bucket `b`.
    pub fn expert_slots(&self, b: usize, u: usize) -> &[u32] {
        let off = self.bucket_offsets(b);
        &self.bucket_slots(b)[off[u] as usize..off[u + 1] as usize]
    }
}

// ---------------------------------------------------------------------------
// CPU model of the production fixed-capacity device planner
// ---------------------------------------------------------------------------

/// Threads of the device planner launch (one workgroup, one thread per slot;
/// `kernels/src/qwen4_slab_union_plan.hip` `SLAB_PLAN_THREADS`).
pub const PLAN_THREADS: usize = 512;
/// Most slots (`10 * MAX_LANES`) the device planner stores (`SLAB_PLAN_MAX_SLOTS`).
pub const PLAN_MAX_SLOTS: usize = TOP_K * MAX_LANES;
/// Device planner buckets (fixed codec order, see [`Codec::ORDER`]).
pub const PLAN_BUCKETS: usize = 2;

const _: () = assert!(PLAN_MAX_SLOTS <= PLAN_THREADS);

/// Words of the fixed-capacity descriptor for `c` lanes (`S = 10 c`):
/// ids `2 S`, offsets `2 (S + 1)`, slots `2 S` = `6 S + 2`. Refuses `c`
/// outside `1..=MAX_LANES` (C16 = 962 words = 3848 B, C32 = 1922 words =
/// 7688 B).
pub fn fixed_union_words(c: usize) -> Result<usize, SlabMoeError> {
    if !(1..=MAX_LANES).contains(&c) {
        return Err(SlabMoeError::InvalidLaneCount { c });
    }
    Ok(6 * TOP_K * c + 2)
}

/// Bytes of [`fixed_union_words`].
pub fn fixed_union_bytes(c: usize) -> Result<usize, SlabMoeError> {
    Ok(fixed_union_words(c)? * U32_BYTES)
}

/// Host-side extent check before a planner launch: `c` must be `1..=32`, the
/// routed-id buffer must hold `10 c` i32 words and the descriptor buffer
/// `6 * 10 c + 2` u32 words. Returns the descriptor word count.
pub fn check_plan_extents(
    c: usize,
    topk_bytes: usize,
    union_bytes: usize,
) -> Result<usize, SlabMoeError> {
    let words = fixed_union_words(c)?;
    if topk_bytes < TOP_K * c * U32_BYTES {
        return Err(SlabMoeError::Operand {
            name: "topk",
            reason: "allocation smaller than 10 * lanes ids",
        });
    }
    if union_bytes < words * U32_BYTES {
        return Err(SlabMoeError::Operand {
            name: "union scratch",
            reason: "allocation smaller than 6 * 10 * lanes + 2 words",
        });
    }
    Ok(words)
}

/// Sentinel of a descriptor word no thread has written.
const UNWRITTEN: u32 = 0xDEAD_BEEF;

/// Thread-faithful CPU model of `qwen4_slab_union_plan`: the production
/// **first-occurrence** fixed-capacity layout (distinct from the sorted compact
/// [`LaneUnionPlan`]), computed phase by phase over all [`PLAN_THREADS`]
/// threads with a write log, so tests can assert that every one of the
/// `6 S + 2` words has exactly one owner. Use [`FixedUnion::canonical`] to
/// compare with the sorted representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedUnion {
    /// Slots `S = 10 c`.
    pub s: usize,
    /// The `6 S + 2` descriptor words (`ids | offsets | slots`).
    pub words: Vec<u32>,
    /// Number of thread writes each word received (must all be `1`).
    pub writes: Vec<u8>,
}

impl FixedUnion {
    /// Models the device planner for the first `10 * lanes` entries of `topk`.
    /// `uniform_bucket >= 0` puts every slot in that bucket (`0` or `1`) and
    /// ignores `codec`; otherwise `codec[id] == 1` selects bucket 0 (W3) and
    /// any other byte bucket 1, exactly as the kernel does. Out-of-range ids
    /// are clamped to expert 0 like the kernel.
    pub fn model(
        topk: &[i32],
        lanes: usize,
        codec: Option<&[u8]>,
        uniform_bucket: i32,
    ) -> Result<Self, SlabMoeError> {
        let words_len = fixed_union_words(lanes)?;
        let s = TOP_K * lanes;
        if topk.len() < s {
            return Err(SlabMoeError::Operand {
                name: "topk",
                reason: "fewer than 10 * lanes ids",
            });
        }
        if uniform_bucket >= PLAN_BUCKETS as i32 {
            return Err(SlabMoeError::Layout("uniform bucket must be 0 or 1"));
        }
        if uniform_bucket < 0 && !codec.is_some_and(|t| t.len() >= MAX_EXPERTS) {
            return Err(SlabMoeError::Operand {
                name: "codec",
                reason: "table shorter than 512 bytes",
            });
        }

        let mut words = vec![UNWRITTEN; words_len];
        let mut writes = vec![0u8; words_len];
        let mut put = |at: usize, v: u32| {
            assert!(
                at < words_len,
                "planner write {at} outside the 6 S + 2 words"
            );
            words[at] = v;
            writes[at] += 1;
        };

        // Workgroup-shared state (LDS) and per-thread registers.
        let mut s_id = [0i32; PLAN_MAX_SLOTS];
        let mut s_b = [0usize; PLAN_MAX_SLOTS];
        let mut s_first = [false; PLAN_MAX_SLOTS];
        let mut s_total = [0usize; PLAN_MAX_SLOTS];
        let mut s_eidx = [0usize; PLAN_MAX_SLOTS];
        let mut rank = [0usize; PLAN_THREADS];
        let mut first = [0usize; PLAN_THREADS];

        // Phase 1: stage ids and buckets (threads t >= S are inactive).
        for t in 0..s {
            let id = topk[t];
            let id = if (id as u32) < MAX_EXPERTS as u32 {
                id
            } else {
                0
            };
            s_id[t] = id;
            s_b[t] = if uniform_bucket >= 0 {
                uniform_bucket as usize
            } else if codec.expect("checked above")[id as usize] == 1 {
                0
            } else {
                1
            };
        }
        // barrier
        // Phase 2: per-slot first-occurrence scan.
        for t in 0..s {
            let id = s_id[t];
            let (mut r, mut total, mut f) = (0usize, 0usize, None::<usize>);
            for j in 0..s {
                if s_id[j] == id {
                    f.get_or_insert(j);
                    total += 1;
                    if j < t {
                        r += 1;
                    }
                }
            }
            rank[t] = r;
            first[t] = f.expect("a slot matches itself");
            s_first[t] = r == 0;
            s_total[t] = total;
        }
        // barrier
        // Phase 3: expert index of each first occurrence inside its bucket.
        for t in 0..s {
            if rank[t] == 0 {
                let b = s_b[t];
                s_eidx[t] = (0..t).filter(|&j| s_first[j] && s_b[j] == b).count();
            }
        }
        // barrier
        // Phase 4: exclusive-owner writes. Every thread re-derives the bucket
        // totals (U_b, n_b) from the shared state.
        let (mut ub, mut nb) = ([0usize; PLAN_BUCKETS], [0usize; PLAN_BUCKETS]);
        for j in 0..s {
            if s_first[j] {
                ub[s_b[j]] += 1;
                nb[s_b[j]] += s_total[j];
            }
        }
        let offs_base = 2 * s;
        let slots_base = 2 * s + 2 * (s + 1);
        for t in 0..PLAN_THREADS {
            if t < s {
                let b = s_b[t];
                let e = s_eidx[first[t]];
                let start: usize = (0..s)
                    .filter(|&j| s_first[j] && s_b[j] == b && s_eidx[j] < e)
                    .map(|j| s_total[j])
                    .sum();
                put(slots_base + b * s + start + rank[t], t as u32);
                if rank[t] == 0 {
                    put(b * s + e, s_id[t] as u32);
                    put(offs_base + b * (s + 1) + e, start as u32);
                }
                for bk in 0..PLAN_BUCKETS {
                    if t >= ub[bk] {
                        put(bk * s + t, 0);
                        put(offs_base + bk * (s + 1) + t, nb[bk] as u32);
                    }
                    if t >= nb[bk] {
                        put(slots_base + bk * s + t, 0);
                    }
                }
            }
            if t == 0 {
                for bk in 0..PLAN_BUCKETS {
                    put(offs_base + bk * (s + 1) + s, nb[bk] as u32);
                }
            }
        }
        Ok(Self { s, words, writes })
    }

    /// Models the planner for `top` lanes of ten ids. `codecs[e]` is expert
    /// `e`'s declared codec; a selected undeclared expert is refused. With
    /// `uniform`, every slot goes in that codec's bucket and `codecs` is not
    /// consulted (a layer whose experts all share one codec).
    pub fn from_routes(
        codecs: &[Option<Codec>],
        top: &[[u32; TOP_K]],
        uniform: Option<Codec>,
    ) -> Result<Self, SlabMoeError> {
        let mut table = vec![0u8; MAX_EXPERTS];
        let mut topk = Vec::with_capacity(top.len() * TOP_K);
        for (lane, ids) in top.iter().enumerate() {
            for (rank, &id) in ids.iter().enumerate() {
                if id as usize >= MAX_EXPERTS {
                    return Err(SlabMoeError::ExpertOutOfRange { lane, rank, id });
                }
                if uniform.is_none() {
                    match codecs.get(id as usize).copied().flatten() {
                        Some(Codec::Mq3I4) => table[id as usize] = 1,
                        Some(Codec::Mq4Sym) => table[id as usize] = 0,
                        None => return Err(SlabMoeError::MissingCodec { expert: id }),
                    }
                }
                topk.push(id as i32);
            }
        }
        let ub = uniform.map_or(-1, |c| c.order_index() as i32);
        Self::model(&topk, top.len(), Some(&table), ub)
    }

    /// `ids` of bucket `b` (`S` words, first-occurrence order, tail 0).
    pub fn ids(&self, b: usize) -> &[u32] {
        &self.words[b * self.s..(b + 1) * self.s]
    }

    /// `offsets` of bucket `b` (`S + 1` words, `0 ..= n_b`, tail `n_b`).
    pub fn offsets(&self, b: usize) -> &[u32] {
        let base = 2 * self.s + b * (self.s + 1);
        &self.words[base..base + self.s + 1]
    }

    /// `slot_ids` of bucket `b` (`S` words, grouped by expert, tail 0).
    pub fn slots(&self, b: usize) -> &[u32] {
        let base = 2 * self.s + 2 * (self.s + 1) + b * self.s;
        &self.words[base..base + self.s]
    }

    /// Distinct experts `U_b` of bucket `b`.
    pub fn expert_count(&self, b: usize) -> usize {
        self.offsets(b).windows(2).filter(|w| w[1] > w[0]).count()
    }

    /// Slots `n_b` of bucket `b`.
    pub fn slot_count(&self, b: usize) -> usize {
        self.offsets(b)[self.s] as usize
    }

    /// Structural contract of the device descriptor: every word has exactly one
    /// owner and is initialized; per bucket offsets start at 0, rise strictly
    /// over the `U_b` live experts and stay at `n_b` afterwards; ids are
    /// distinct, tail ids and tail slots are 0; per-expert slots ascend and
    /// belong to the expert's own run; experts are in first-occurrence order;
    /// the two buckets partition the original slots exactly once.
    pub fn validate(&self) -> Result<(), SlabMoeError> {
        if self.words.len() != 6 * self.s + 2 || self.writes.len() != self.words.len() {
            return Err(SlabMoeError::Layout("descriptor is not 6 S + 2 words"));
        }
        if self.writes.iter().any(|&n| n != 1) || self.words.contains(&UNWRITTEN) {
            return Err(SlabMoeError::Layout(
                "a descriptor word has zero or several owners",
            ));
        }
        let mut seen = vec![false; self.s];
        for b in 0..PLAN_BUCKETS {
            let (ids, offs, slots) = (self.ids(b), self.offsets(b), self.slots(b));
            let (u, n) = (self.expert_count(b), self.slot_count(b));
            if offs[0] != 0 || offs.windows(2).any(|w| w[1] < w[0]) || n > self.s || u > n {
                return Err(SlabMoeError::Layout(
                    "offsets are not monotonic from 0 to n_b",
                ));
            }
            if offs[..=u].windows(2).any(|w| w[1] <= w[0])
                || offs[u..].iter().any(|&o| o as usize != n)
            {
                return Err(SlabMoeError::Layout(
                    "live offsets must rise strictly, tail must equal n_b",
                ));
            }
            if ids[u..].iter().any(|&i| i != 0) || slots[n..].iter().any(|&p| p != 0) {
                return Err(SlabMoeError::Layout("unused ids/slots tail is not zero"));
            }
            let mut prev_first = None::<u32>;
            for e in 0..u {
                if ids[..e].contains(&ids[e]) || ids[e] as usize >= MAX_EXPERTS {
                    return Err(SlabMoeError::Layout("expert ids repeat or exceed 511"));
                }
                let run = &slots[offs[e] as usize..offs[e + 1] as usize];
                if run.windows(2).any(|w| w[0] >= w[1]) {
                    return Err(SlabMoeError::Layout("slots of an expert do not ascend"));
                }
                if prev_first.is_some_and(|p| run[0] <= p) {
                    return Err(SlabMoeError::Layout(
                        "experts are not in first-occurrence order",
                    ));
                }
                prev_first = Some(run[0]);
                for &slot in run {
                    if slot as usize >= self.s || std::mem::replace(&mut seen[slot as usize], true)
                    {
                        return Err(SlabMoeError::Layout(
                            "a slot is out of range or listed twice",
                        ));
                    }
                }
            }
        }
        if seen.iter().any(|&x| !x) {
            return Err(SlabMoeError::Layout(
                "buckets do not partition the 10 C slots",
            ));
        }
        Ok(())
    }

    /// The descriptor as `(codec, expert, slots)` sorted by bucket then expert
    /// id: the order of the sorted compact [`LaneUnionPlan`] representation,
    /// independent of first-occurrence order.
    pub fn canonical(&self) -> Vec<(Codec, u32, Vec<u32>)> {
        let mut out = Vec::new();
        for b in 0..PLAN_BUCKETS {
            let (ids, offs, slots) = (self.ids(b), self.offsets(b), self.slots(b));
            let mut rows: Vec<(Codec, u32, Vec<u32>)> = (0..self.expert_count(b))
                .map(|e| {
                    (
                        Codec::ORDER[b],
                        ids[e],
                        slots[offs[e] as usize..offs[e + 1] as usize].to_vec(),
                    )
                })
                .collect();
            rows.sort_by_key(|r| r.1);
            out.extend(rows);
        }
        out
    }
}

/// Little-endian bytes of a `u32` descriptor array, ready for a device upload.
pub fn u32_le_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

// ---------------------------------------------------------------------------
// Launch planning (pure, GPU-free)
// ---------------------------------------------------------------------------

/// A device allocation as the launch sees it: base address and byte extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DevView {
    pub ptr: u64,
    pub bytes: usize,
}

impl DevView {
    pub const fn new(ptr: u64, bytes: usize) -> Self {
        Self { ptr, bytes }
    }

    /// The whole tensor allocation.
    pub fn from_tensor(t: &GpuTensor) -> Self {
        Self {
            ptr: t.buf.as_ptr() as usize as u64,
            bytes: t.buf.size(),
        }
    }

    fn check(&self, name: &'static str, need_bytes: usize, align: u64) -> Result<(), SlabMoeError> {
        if self.ptr == 0 {
            return Err(SlabMoeError::Operand {
                name,
                reason: "null device pointer",
            });
        }
        if self.ptr % align != 0 {
            return Err(SlabMoeError::Operand {
                name,
                reason: "misaligned device pointer",
            });
        }
        if self.bytes < need_bytes {
            return Err(SlabMoeError::Operand {
                name,
                reason: "allocation smaller than the launch extent",
            });
        }
        self.ptr
            .checked_add(need_bytes as u64)
            .ok_or(SlabMoeError::Operand {
                name,
                reason: "address range overflows",
            })?;
        Ok(())
    }
}

/// Immutable device-owned descriptor arrays (see the module docs for layout).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescriptorArrays {
    pub expert_ids: DevView,
    pub expert_offsets: DevView,
    pub slot_ids: DevView,
}

/// One launch of a projection over every bucket of a union.
#[derive(Clone, Copy, Debug)]
pub struct DecodeRequest<'a> {
    /// Projection.
    pub kind: Kind,
    /// Explicit rotation profile; must be the kind's.
    pub profile: Profile,
    /// Explicit output rows; must be the kind's frozen `M`.
    pub m: usize,
    /// Explicit reduction width; must be the kind's frozen `K`.
    pub k: usize,
    /// Shape of `descriptors`.
    pub layout: &'a UnionLayout,
    pub descriptors: DescriptorArrays,
    /// `[512]` u64 projection table of expert code-plane bases (by expert id).
    pub weight_ptrs: DevView,
    /// F32 rotated activations: gate `[C, 2560]`, down `[10 C, 640]`.
    pub xrot: DevView,
    /// F32 gate `[C, 10, 640]` / unweighted down `[C, 10, 2560]`.
    pub y0: DevView,
    /// F32 up `[C, 10, 640]` for gate/up; `None` for down.
    pub y1: Option<DevView>,
}

/// One concrete kernel launch of a bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BucketLaunch {
    pub key: BucketKey,
    pub symbol: &'static str,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub kernargs: MoeGemvKernargs,
}

/// At most one bucket per codec: the launches of one projection over a union.
pub const MAX_BUCKETS: usize = Codec::ORDER.len();

/// Fixed-capacity, allocation-free sequence of bucket launches in the fixed
/// bucket order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchPlan {
    entries: [Option<BucketLaunch>; MAX_BUCKETS],
    len: usize,
}

impl LaunchPlan {
    const EMPTY: LaunchPlan = LaunchPlan {
        entries: [None; MAX_BUCKETS],
        len: 0,
    };

    // `UnionLayout::validate` bounds the bucket count by `MAX_BUCKETS`
    // before any entry is pushed.
    fn push(&mut self, launch: BucketLaunch) {
        self.entries[self.len] = Some(launch);
        self.len += 1;
    }

    /// Number of launches.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The `i`-th launch, if any.
    pub fn get(&self, i: usize) -> Option<&BucketLaunch> {
        self.entries[..self.len].get(i).and_then(Option::as_ref)
    }

    /// Launches in bucket order.
    pub fn iter(&self) -> impl Iterator<Item = &BucketLaunch> + '_ {
        self.entries[..self.len].iter().flatten()
    }
}

impl std::ops::Index<usize> for LaunchPlan {
    type Output = BucketLaunch;

    fn index(&self, i: usize) -> &BucketLaunch {
        self.get(i).expect("launch index out of range")
    }
}

fn mul(a: usize, b: usize, name: &'static str) -> Result<usize, SlabMoeError> {
    a.checked_mul(b).ok_or(SlabMoeError::Operand {
        name,
        reason: "extent overflows usize",
    })
}

fn disjoint(
    a: (&'static str, DevView, usize),
    b: (&'static str, DevView, usize),
) -> Result<(), SlabMoeError> {
    let (a_end, b_end) = (a.1.ptr + a.2 as u64, b.1.ptr + b.2 as u64);
    if a.1.ptr < b_end && b.1.ptr < a_end {
        return Err(SlabMoeError::Operand {
            name: b.0,
            reason: "overlaps another operand of the launch",
        });
    }
    Ok(())
}

/// Validates `req` (explicit profile/geometry, layout, every operand extent,
/// alignment and aliasing) and builds one [`BucketLaunch`] per bucket, in the
/// fixed bucket order. Performs no device access.
pub fn plan_launches(req: &DecodeRequest<'_>) -> Result<LaunchPlan, SlabMoeError> {
    let kind = req.kind;
    if req.profile != kind.profile() {
        return Err(SlabMoeError::ProfileMismatch {
            kind,
            profile: req.profile,
        });
    }
    if req.m != kind.m() {
        return Err(SlabMoeError::Geometry {
            what: "M",
            got: req.m,
            want: kind.m(),
        });
    }
    if req.k != kind.k() {
        return Err(SlabMoeError::Geometry {
            what: "K",
            got: req.k,
            want: kind.k(),
        });
    }
    let layout = req.layout;
    layout.validate()?;
    let c = layout.c;

    let d = &req.descriptors;
    d.expert_ids.check(
        "expert_ids",
        mul(layout.expert_id_words(), U32_BYTES, "expert_ids")?,
        4,
    )?;
    d.expert_offsets.check(
        "expert_offsets",
        mul(layout.expert_offset_words(), U32_BYTES, "expert_offsets")?,
        4,
    )?;
    d.slot_ids.check(
        "slot_ids",
        mul(layout.slot_words(), U32_BYTES, "slot_ids")?,
        4,
    )?;
    req.weight_ptrs
        .check("weight_ptrs", MAX_EXPERTS * PTR_BYTES, 8)?;

    let x_bytes = mul(
        mul(kind.x_src_rows(c), kind.k(), "xrot")?,
        F32_BYTES,
        "xrot",
    )?;
    let y_bytes = mul(
        mul(mul(c, TOP_K, "y0")?, kind.y_slot_stride(), "y0")?,
        F32_BYTES,
        "y0",
    )?;
    req.xrot.check("xrot", x_bytes, 4)?;
    req.y0.check("y0", y_bytes, 4)?;
    disjoint(("xrot", req.xrot, x_bytes), ("y0", req.y0, y_bytes))?;
    let y1_ptr = match (kind.has_y1(), req.y1) {
        (true, Some(y1)) => {
            y1.check("y1", y_bytes, 4)?;
            disjoint(("xrot", req.xrot, x_bytes), ("y1", y1, y_bytes))?;
            disjoint(("y0", req.y0, y_bytes), ("y1", y1, y_bytes))?;
            y1.ptr
        }
        (true, None) => {
            return Err(SlabMoeError::Operand {
                name: "y1",
                reason: "gate/up requires the up output",
            })
        }
        (false, None) => 0,
        (false, Some(_)) => {
            return Err(SlabMoeError::Operand {
                name: "y1",
                reason: "down has no second output",
            })
        }
    };

    let word = |view: DevView, index: usize, name: &'static str| -> Result<u64, SlabMoeError> {
        (index as u64)
            .checked_mul(U32_BYTES as u64)
            .and_then(|o| view.ptr.checked_add(o))
            .ok_or(SlabMoeError::Operand {
                name,
                reason: "bucket base overflows",
            })
    };

    let u32_of = |v: usize, name: &'static str| -> Result<u32, SlabMoeError> {
        u32::try_from(v).map_err(|_| SlabMoeError::Operand {
            name,
            reason: "value exceeds u32",
        })
    };
    let mut plan = LaunchPlan::EMPTY;
    for b in &layout.buckets {
        let key = BucketKey::new(b.codec, kind);
        plan.push(BucketLaunch {
            key,
            symbol: key.symbol(),
            grid: [u32_of(kind.m(), "m")?, u32_of(b.expert_count, "u")?, 1],
            block: [WORKGROUP, 1, 1],
            kernargs: MoeGemvKernargs {
                expert_weight_ptrs: req.weight_ptrs.ptr,
                expert_ids: word(d.expert_ids, b.expert_start, "expert_ids")?,
                expert_offsets: word(d.expert_offsets, b.offsets_start, "expert_offsets")?,
                slot_ids: word(d.slot_ids, b.slot_start, "slot_ids")?,
                xrot: req.xrot.ptr,
                y0: req.y0.ptr,
                y1: y1_ptr,
                m: u32_of(kind.m(), "m")?,
                k: u32_of(kind.k(), "k")?,
                u: u32_of(b.expert_count, "u")?,
                c: u32_of(c, "c")?,
                x_row_div: u32_of(kind.x_row_div(), "x_row_div")?,
                x_src_rows: u32_of(kind.x_src_rows(c), "x_src_rows")?,
                x_row_stride: u32_of(kind.k(), "x_row_stride")?,
                y_slot_stride: u32_of(kind.y_slot_stride(), "y_slot_stride")?,
                n_experts: u32_of(MAX_EXPERTS, "n_experts")?,
                reserved: 0,
            },
        });
    }
    Ok(plan)
}

// ---------------------------------------------------------------------------
// Real launch through the HIP module API
// ---------------------------------------------------------------------------

/// The four K1 decode functions loaded from a gfx1201 or gfx1151 code object, indexed by
/// [`BucketKey::order_index`] (the order of [`SYMBOLS`]).
///
/// The HIP module owns its functions: dropping this value may unload them.
/// See the Safety section of [`SlabMoeDecodeModule::launch`].
pub struct SlabMoeDecodeModule {
    // Keeps the HIP module (and so every function) alive.
    _module: Module,
    functions: [Function; 4],
}

fn require_slab_arch(gpu: &Gpu) -> Result<(), SlabMoeError> {
    if gpu.arch_caps.is_gfx1201() || gpu.arch_caps.is_gfx1151() {
        Ok(())
    } else {
        Err(SlabMoeError::Arch(gpu.arch.clone()))
    }
}

impl SlabMoeDecodeModule {
    /// Loads `image` (the code object of this device's architecture exporting every
    /// [`SYMBOLS`] entry) on `gpu`. Refuses any device that is not exactly gfx1201 or gfx1151 and
    /// fails closed if a symbol is missing.
    pub fn load(gpu: &mut Gpu, image: &[u8]) -> HipResult<Self> {
        require_slab_arch(gpu).map_err(hip_err)?;
        gpu.bind_thread()?;
        gpu.ensure_route_modules_preloaded()?;
        let module = gpu.hip.module_load_data(image)?;
        let functions = [
            gpu.hip.module_get_function(&module, SYMBOLS[0])?,
            gpu.hip.module_get_function(&module, SYMBOLS[1])?,
            gpu.hip.module_get_function(&module, SYMBOLS[2])?,
            gpu.hip.module_get_function(&module, SYMBOLS[3])?,
        ];
        Ok(Self {
            _module: module,
            functions,
        })
    }

    /// Launches `req` on the device's active stream, one kernel per bucket in
    /// the fixed bucket order. Everything shape-related is validated by
    /// [`plan_launches`] before the first launch; no device readback happens.
    /// Refused (safely, before any launch) while a replay recording or graph
    /// capture is active: this raw path would neither be recorded nor retain
    /// its kernarg bytes.
    ///
    /// # Safety
    ///
    /// `plan_launches` checks only shapes, extents, alignment and aliasing of
    /// the raw [`DevView`] addresses; it cannot know they name live
    /// allocations. The caller guarantees, until all work enqueued on the
    /// active stream by this call has completed:
    ///
    /// * every address in `req` (the three descriptor arrays, `weight_ptrs`,
    ///   `xrot`, `y0`, `y1`) is a valid device allocation of at least the
    ///   extent it declares, live and on this `gpu`'s device, and no other
    ///   work writes `xrot`, the descriptor arrays or `weight_ptrs`, nor
    ///   reads/writes `y0`/`y1` unordered with this stream;
    /// * the descriptor arrays and `weight_ptrs` are immutable and describe
    ///   exactly `req.layout` (ids `< 512`, bucket offsets and slot ids
    ///   consistent, slots partitioning `0..10 C`), and the expert generation
    ///   they were prepared for is unchanged;
    /// * every `weight_ptrs[id]` of a selected expert points to a live code
    ///   plane of that expert's declared codec/profile (`Mq3I4` or `Mq4Sym`,
    ///   matching the bucket) with the frozen `M x K` extent;
    /// * **this `SlabMoeDecodeModule` outlives all of that work**: dropping it
    ///   may unload the HIP module and functions, so the K4 binding owner must
    ///   retain it until the stream has synchronized or the work has been
    ///   ordered before the drop.
    pub unsafe fn launch(&self, gpu: &mut Gpu, req: &DecodeRequest<'_>) -> HipResult<()> {
        require_slab_arch(gpu).map_err(hip_err)?;
        if gpu.replay.is_recording() || gpu.graphs.capture_mode {
            return Err(hip_err(SlabMoeError::Unavailable(
                "mq_i4_moe_gemv launch is not recordable or capturable yet",
            )));
        }
        let launches = plan_launches(req).map_err(hip_err)?;
        gpu.bind_thread()?;
        gpu.ensure_route_modules_preloaded()?;
        for l in launches.iter() {
            let func = &self.functions[l.key.order_index()];
            let mut bytes = l.kernargs.to_bytes();
            // SAFETY: `bytes` is the exact frozen 96-byte kernarg of
            // `l.symbol`; shapes were validated by `plan_launches`, and the
            // address-validity, immutability and module-lifetime
            // preconditions are this function's contract.
            unsafe {
                gpu.hip.launch_kernel_blob(
                    func,
                    l.grid,
                    l.block,
                    0,
                    gpu.active_stream.as_ref(),
                    &mut bytes,
                )
            }
            .map_err(|e| e.with_kernel(l.symbol))?;
        }
        Ok(())
    }
}
