// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! K-E2 E8P-on-IU4 pointer-addressed MoE decode GEMV launcher, gfx1201 and gfx1151.
//!
//! Counterpart of `peacemaker-isa::kernels::e8p_i4_moe_gemv` (gfx1201: module
//! `e8p_i4_moe_gemv_pm_gfx1201`, symbols `e8p_i4_moe_gemv_r256_gate_up` and
//! `e8p_i4_moe_gemv_r128_down`; gfx1151: module `e8p_i4_moe_gemv_pm_gfx1151`, symbols
//! `e8p_i4_moe_gemv_{r256_gate_up,r128_down}_pm_gfx1151`, the gfx1100 arm with only the
//! arch-forced deltas; [`E8pArch`] selects module, symbols and image per device, the launch ABI
//! and geometry are identical). The launch ABI is the frozen F4 96-byte
//! kernarg of the `mq_i4_moe_gemv_*` symbols (the 88-byte block plus the trailing `n_experts` and a zero word):
//!
//! | offset | field |
//! |---|---|
//! | `@0`  | `expert_weight_ptrs`, `u64[512]` table indexed by **expert id**, each entry the expert's projection blob base |
//! | `@8`  | `expert_ids`, `u32[U]`, strictly ascending |
//! | `@16` | `expert_offsets`, `u32[U+1]`, from `0` to the bucket's slot count |
//! | `@24` | `slot_ids`, `u32[10*C]`, original slots `lane*10 + rank`, ascending inside an expert |
//! | `@32` | `Xrot`, F32 activations |
//! | `@40` | `Y0` (gate / unweighted down) |
//! | `@48` | `Y1` (up; null for down) |
//! | `@56..@88` | u32 `M, K, U, C, x_row_div, x_src_rows, x_row_stride, y_slot_stride` (strides in F32 elements) |
//! | `@88` | u32 `n_experts`, the extent of the `expert_weight_ptrs` table (`1..=i32::MAX`); an id at or past it (or negative) exits with no table load and no store |
//! | `@92` | u32 reserved, zero |
//!
//! Grid `[M, U, 1]`, block `[32, 1, 1]` (one wave32 per `(output row, distinct
//! expert)`), no LDS argument. Every distinct expert's weight blob is read once and
//! serves up to eight original slots (C1–C8); outputs are written at the
//! original slot, `Y[slot * y_slot_stride + row]`. Down applies no route weight.
//!
//! # Expert blob wire (planar, per projection, no tail pad)
//!
//! `[words: M*K/8 little-endian u16][scales: M*K/128 little-endian FP16]`
//! (`hipfire_runtime::e8p_i4_reference::pack_word`, `idx | sign7 << 8 | shift << 15`).
//! Gate/up `[1280, 2560]` is 870 400 B, down `[2560, 640]` is 435 200 B. The
//! kernel reads exact `u16` words and scales only, so no padding bytes follow the
//! blob (the final scale of the final row is the last byte read). Every scale
//! must satisfy the writer guard (finite, `>= 0x0400`).
//!
//! # Interface for the route owner
//!
//! * One-time (before graph capture / recording): [`Gpu::ensure_e8p_i4_moe_gemv_kernels`]
//!   with the code-object bytes of the emitted `e8p_i4_moe_gemv_pm_<arch>` bundle
//!   (`impl Into<CodeObjectBytes>`: `Arc<[u8]>` read from a file, or a `&'static [u8]`
//!   `include_bytes!` const). It binds both symbols to that one image digest.
//! * Per projection: [`Gpu::e8p_i4_moe_gemv_slab`] — argument order
//!   `(down, buckets, bucket, weight_ptrs, xrot, y0, y1, lanes)`. `buckets`/`bucket`
//!   locate the E8P bucket inside the slab lane-union scratch, whose layout for `B`
//!   buckets and `S = 10 * lanes` slots is, in u32 words (see [`union_words`] and
//!   [`union_bucket_words`]): `ids[b*S ..]`, `offsets[B*S + b*(S+1) ..]`,
//!   `slot_ids[B*S + B*(S+1) + b*S ..]`, `B*(3S+1)` words in all (`B = 2` is the
//!   incumbent `6S+2`). The launch grid's `U` is the capacity `S`: an expert whose slot
//!   count is `0` (the planner's empty tail), above eight, or whose end offset exceeds
//!   `10*C` returns without a store, so a data-dependent bucket count needs no readback.
//! * Scratch zeroing: **none required.** Every `Y` word of a selected slot's output rows
//!   is written exactly once; the kernel never reads `Y`, `xrot` is read only inside its
//!   `x_src_rows * x_row_stride` extent (rows past `x_src_rows` read `0`), and the
//!   launch has no global scratch or atomics.
//! * Tail pad: **none required** (see the wire above). A pool that keeps a 16-byte
//!   alignment pad after each blob is allowed, never read.
//! * Host-planned descriptors (no device planner) use [`E8pUnionHost`] and
//!   [`E8pMoeDecodeModule::launch`] with explicit [`DevView`]s.

use crate::dispatch::{Gpu, GpuTensor};
use hip_bridge::{Function, HipError, HipResult, Module};
use std::ffi::c_void;
use std::fmt;

/// Module key of the emitted K-E2 image (gfx1201; per arch: [`E8pArch::module`]).
pub const E8P_I4_MOE_GEMV_MODULE: &str = "e8p_i4_moe_gemv_pm_gfx1201";

/// `[gate_up (R256), down (R128)]` entry symbols of the gfx1201 module (per arch:
/// [`E8pArch::symbols`]).
pub const E8P_I4_MOE_GEMV_SYMBOLS: [&str; 2] =
    ["e8p_i4_moe_gemv_r256_gate_up", "e8p_i4_moe_gemv_r128_down"];

/// Module key of the emitted gfx1151 K-E2 image (the gfx1100 arm with only arch-forced deltas).
pub const E8P_I4_MOE_GEMV_GFX1151_MODULE: &str = "e8p_i4_moe_gemv_pm_gfx1151";

/// `[gate_up (R256), down (R128)]` entry symbols of the gfx1151 module; contracts
/// `kernels/e8p_i4_moe_gemv.gfx1151.{r256_gate_up,r128_down}.contract.json`.
pub const E8P_I4_MOE_GEMV_GFX1151_SYMBOLS: [&str; 2] = [
    "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1151",
    "e8p_i4_moe_gemv_r128_down_pm_gfx1151",
];

/// Architectures this launcher serves: its module key, entry symbols and embedded image.
/// Everything else about the launch (ABI, geometry, wire) is arch-independent. Presence of a
/// row is not admission: the sealed slab route is gated by `qwen4_slab_codec_admitted`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum E8pArch {
    Gfx1201,
    Gfx1151,
}

impl E8pArch {
    pub const ALL: [E8pArch; 2] = [E8pArch::Gfx1201, E8pArch::Gfx1151];

    /// Exact architecture name match (`"gfx1201"`, `"gfx1151"`); everything else is `None`.
    pub fn of_name(arch: &str) -> Option<Self> {
        match arch {
            "gfx1201" => Some(Self::Gfx1201),
            "gfx1151" => Some(Self::Gfx1151),
            _ => None,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Gfx1201 => "gfx1201",
            Self::Gfx1151 => "gfx1151",
        }
    }
    /// Module key of the arch's emitted image.
    pub const fn module(self) -> &'static str {
        match self {
            Self::Gfx1201 => E8P_I4_MOE_GEMV_MODULE,
            Self::Gfx1151 => E8P_I4_MOE_GEMV_GFX1151_MODULE,
        }
    }
    /// `[gate_up (R256), down (R128)]` entry symbols of the arch's module.
    pub const fn symbols(self) -> [&'static str; 2] {
        match self {
            Self::Gfx1201 => E8P_I4_MOE_GEMV_SYMBOLS,
            Self::Gfx1151 => E8P_I4_MOE_GEMV_GFX1151_SYMBOLS,
        }
    }
    /// The arch's embedded certified builder image.
    pub const fn image(self) -> &'static [u8] {
        match self {
            Self::Gfx1201 => crate::kernels::E8P_I4_MOE_GEMV_PM_GFX1201,
            Self::Gfx1151 => crate::kernels::E8P_I4_MOE_GEMV_PM_GFX1151,
        }
    }
}

/// Size of the frozen F4 GEMV kernarg (`n_experts@88`, zero `reserved@92`).
pub const KERNARG_BYTES: usize = 96;
/// Threads per workgroup (one wave32).
pub const WORKGROUP: u32 = 32;
/// Most original slots one expert workgroup serves (C8).
pub const MAX_SLOTS: usize = 8;
/// Routed experts per lane.
pub const TOP_K: usize = 10;
/// Most lanes of the decode union.
pub const MAX_LANES: usize = 8;
/// Experts per layer (`0..512` ids; size of the pointer table).
pub const MAX_EXPERTS: usize = 512;
/// Fused gate+up output rows and reduction width.
pub const GATE_UP_M: usize = 1280;
pub const GATE_UP_K: usize = 2560;
/// Down output rows and reduction width.
pub const DOWN_M: usize = 2560;
pub const DOWN_K: usize = 640;

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

/// Projection a launch computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum E8pKind {
    /// Fused gate/up `[1280, 2560]`, R256 (four group chains).
    GateUp,
    /// Down `[2560, 640]`, R128 (one chain).
    Down,
}

impl E8pKind {
    pub const ALL: [E8pKind; 2] = [E8pKind::GateUp, E8pKind::Down];

    /// Index into [`E8P_I4_MOE_GEMV_SYMBOLS`].
    pub const fn index(self) -> usize {
        match self {
            E8pKind::GateUp => 0,
            E8pKind::Down => 1,
        }
    }
    /// The gfx1201 entry symbol (see [`E8pKind::symbol_on`] for the other arches).
    pub const fn symbol(self) -> &'static str {
        E8P_I4_MOE_GEMV_SYMBOLS[self.index()]
    }
    /// Entry symbol of this projection in `arch`'s module.
    pub const fn symbol_on(self, arch: E8pArch) -> &'static str {
        arch.symbols()[self.index()]
    }
    /// Output rows `M` (grid x).
    pub const fn m(self) -> usize {
        match self {
            E8pKind::GateUp => GATE_UP_M,
            E8pKind::Down => DOWN_M,
        }
    }
    /// Reduction width `K`.
    pub const fn k(self) -> usize {
        match self {
            E8pKind::GateUp => GATE_UP_K,
            E8pKind::Down => DOWN_K,
        }
    }
    /// Rows written to `Y0` (gate/up: the first half, rest to `Y1`).
    pub const fn out_rows(self) -> usize {
        match self {
            E8pKind::GateUp => GATE_UP_M / 2,
            E8pKind::Down => DOWN_M,
        }
    }
    /// Slot `s` reads X row `s / x_row_div`: gate 10 (`X[C, K]`), down 1 (`X[10 C, K]`).
    pub const fn x_row_div(self) -> usize {
        match self {
            E8pKind::GateUp => TOP_K,
            E8pKind::Down => 1,
        }
    }
    /// F32 elements between consecutive original slots of each output (tight layout).
    pub const fn y_slot_stride(self) -> usize {
        self.out_rows()
    }
    /// X rows for `c` lanes.
    pub const fn x_src_rows(self, c: usize) -> usize {
        match self {
            E8pKind::GateUp => c,
            E8pKind::Down => TOP_K * c,
        }
    }
    pub const fn has_y1(self) -> bool {
        matches!(self, E8pKind::GateUp)
    }
    /// Bytes of one expert's planar blob: words `M*K/4` + scales `M*K/64`.
    pub const fn blob_bytes(self) -> usize {
        self.m() * self.k() / 4 + self.m() * self.k() / 64
    }
    /// Byte offset of the FP16 scale plane inside a blob.
    pub const fn scales_offset(self) -> usize {
        self.m() * self.k() / 4
    }
}

/// Refusal of a launch before any device work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum E8pMoeError {
    /// Lane count outside `1..=8`.
    InvalidLaneCount { c: usize },
    /// Distinct-expert count outside `1..=min(10 C, 512)`.
    InvalidUnion { u: usize, c: usize },
    /// An operand is null, misaligned, too small, overflowing or aliases another one.
    Operand {
        name: &'static str,
        reason: &'static str,
    },
    /// The device is not exactly gfx1201 or gfx1151.
    Arch(String),
    /// The launch cannot be issued in this execution mode.
    Unavailable(&'static str),
    /// Union scratch / bucket geometry is inconsistent.
    Layout(&'static str),
}

impl fmt::Display for E8pMoeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLaneCount { c } => write!(f, "lane count {c} not in 1..={MAX_LANES}"),
            Self::InvalidUnion { u, c } => write!(
                f,
                "distinct experts {u} not in 1..={} for C{c}",
                max_union(*c)
            ),
            Self::Operand { name, reason } => write!(f, "operand {name}: {reason}"),
            Self::Arch(got) => write!(
                f,
                "e8p_i4_moe_gemv is exact gfx1201 or gfx1151 only, device is {got:?}"
            ),
            Self::Unavailable(why) => write!(f, "{why}"),
            Self::Layout(why) => write!(f, "union layout: {why}"),
        }
    }
}

impl std::error::Error for E8pMoeError {}

fn hip_err(e: E8pMoeError) -> HipError {
    HipError::new(1, &format!("e8p_moe_decode: {e}"))
}

/// The frozen F4 GEMV kernarg, 96 bytes. Strides are F32 elements.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct E8pKernargs {
    /// `@0` per-projection table of expert blob bases, indexed by expert id.
    pub expert_weight_ptrs: u64,
    /// `@8` u32 expert ids (ascending).
    pub expert_ids: u64,
    /// `@16` u32 offsets, `U + 1` entries.
    pub expert_offsets: u64,
    /// `@24` u32 original slot ids.
    pub slot_ids: u64,
    /// `@32` F32 activations.
    pub xrot: u64,
    /// `@40` gate / unweighted down output.
    pub y0: u64,
    /// `@48` up output; `0` for down.
    pub y1: u64,
    /// `@56` output rows.
    pub m: u32,
    /// `@60` reduction width.
    pub k: u32,
    /// `@64` distinct experts (grid y).
    pub u: u32,
    /// `@68` lanes.
    pub c: u32,
    /// `@72` X row is `slot / x_row_div`.
    pub x_row_div: u32,
    /// `@76` X rows available.
    pub x_src_rows: u32,
    /// `@80` F32 elements between X rows.
    pub x_row_stride: u32,
    /// `@84` F32 elements between original-slot output rows.
    pub y_slot_stride: u32,
    /// `@88` extent of the `expert_weight_ptrs` table (`1..=i32::MAX`); `@92` is the reserved zero word (the
    /// struct's tail padding, written as zero by [`E8pKernargs::to_bytes`]).
    pub n_experts: u32,
}

const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(size_of::<E8pKernargs>() == KERNARG_BYTES);
    assert!(align_of::<E8pKernargs>() == 8);
    assert!(offset_of!(E8pKernargs, expert_weight_ptrs) == 0);
    assert!(offset_of!(E8pKernargs, expert_ids) == 8);
    assert!(offset_of!(E8pKernargs, expert_offsets) == 16);
    assert!(offset_of!(E8pKernargs, slot_ids) == 24);
    assert!(offset_of!(E8pKernargs, xrot) == 32);
    assert!(offset_of!(E8pKernargs, y0) == 40);
    assert!(offset_of!(E8pKernargs, y1) == 48);
    assert!(offset_of!(E8pKernargs, m) == 56);
    assert!(offset_of!(E8pKernargs, k) == 60);
    assert!(offset_of!(E8pKernargs, u) == 64);
    assert!(offset_of!(E8pKernargs, c) == 68);
    assert!(offset_of!(E8pKernargs, x_row_div) == 72);
    assert!(offset_of!(E8pKernargs, x_src_rows) == 76);
    assert!(offset_of!(E8pKernargs, x_row_stride) == 80);
    assert!(offset_of!(E8pKernargs, y_slot_stride) == 84);
    assert!(offset_of!(E8pKernargs, n_experts) == 88);
};

impl E8pKernargs {
    /// The exact little-endian kernarg bytes (`reserved@92` is the zero word).
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
        ];
        for (i, w) in words.iter().enumerate() {
            out[56 + i * 4..56 + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// Grid of the launch: `[M, U, 1]`.
    pub fn grid(&self) -> [u32; 3] {
        [self.m, self.u, 1]
    }
}

/// A device allocation as the launch sees it: address and byte extent.
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

    fn check(&self, name: &'static str, need_bytes: usize, align: u64) -> Result<(), E8pMoeError> {
        if self.ptr == 0 {
            return Err(E8pMoeError::Operand {
                name,
                reason: "null device pointer",
            });
        }
        if self.ptr % align != 0 {
            return Err(E8pMoeError::Operand {
                name,
                reason: "misaligned device pointer",
            });
        }
        if self.bytes < need_bytes {
            return Err(E8pMoeError::Operand {
                name,
                reason: "allocation smaller than the launch extent",
            });
        }
        self.ptr
            .checked_add(need_bytes as u64)
            .ok_or(E8pMoeError::Operand {
                name,
                reason: "address range overflows",
            })?;
        Ok(())
    }
}

/// One projection launch over a single bucket (the E8P experts of a step).
#[derive(Clone, Copy, Debug)]
pub struct E8pDecodeRequest {
    pub kind: E8pKind,
    /// Lanes `C in 1..=8`.
    pub c: usize,
    /// Distinct experts of the bucket (kernarg `U`, grid y), `1..=min(10 C, 512)`.
    pub u: usize,
    /// `u32[U]` ascending expert ids.
    pub expert_ids: DevView,
    /// `u32[U+1]` offsets.
    pub expert_offsets: DevView,
    /// `u32[10 C]` original slot ids.
    pub slot_ids: DevView,
    /// `u64[512]` blob-base table indexed by expert id.
    pub weight_ptrs: DevView,
    /// Extent of `weight_ptrs` in experts (kernarg `n_experts@88`): every expert id of the bucket must be below it,
    /// `1..=i32::MAX`, and `weight_ptrs` must hold that many `u64` entries. The kernel exits without a table load or
    /// a store for any id at or past it (or negative).
    pub n_experts: usize,
    /// F32 activations: gate `[C, 2560]`, down `[10 C, 640]`.
    pub xrot: DevView,
    /// F32 `[C, 10, out_rows]` gate (or unweighted down `[C, 10, 2560]`).
    pub y0: DevView,
    /// F32 up `[C, 10, 640]` for gate/up; `None` for down.
    pub y1: Option<DevView>,
}

/// One concrete kernel launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct E8pLaunch {
    pub kind: E8pKind,
    pub symbol: &'static str,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub kernargs: E8pKernargs,
}

fn mul(a: usize, b: usize, name: &'static str) -> Result<usize, E8pMoeError> {
    a.checked_mul(b).ok_or(E8pMoeError::Operand {
        name,
        reason: "extent overflows usize",
    })
}

fn u32_of(v: usize, name: &'static str) -> Result<u32, E8pMoeError> {
    u32::try_from(v).map_err(|_| E8pMoeError::Operand {
        name,
        reason: "value exceeds u32",
    })
}

fn disjoint(
    a: (&'static str, DevView, usize),
    b: (&'static str, DevView, usize),
) -> Result<(), E8pMoeError> {
    let (a_end, b_end) = (a.1.ptr + a.2 as u64, b.1.ptr + b.2 as u64);
    if a.1.ptr < b_end && b.1.ptr < a_end {
        return Err(E8pMoeError::Operand {
            name: b.0,
            reason: "overlaps another operand of the launch",
        });
    }
    Ok(())
}

/// Validates `req` (lane/union bounds, every operand's extent, alignment and
/// aliasing) and builds the launch for the gfx1201 module's symbols
/// ([`plan_launch_on`] with [`E8pArch::Gfx1201`]). Performs no device access. The geometry
/// (`M`, `K`, `x_row_div`, strides) is the kind's frozen one.
pub fn plan_launch(req: &E8pDecodeRequest) -> Result<E8pLaunch, E8pMoeError> {
    plan_launch_on(E8pArch::Gfx1201, req)
}

/// [`plan_launch`] naming `arch`'s entry symbol; the geometry and kernargs are arch-independent.
pub fn plan_launch_on(arch: E8pArch, req: &E8pDecodeRequest) -> Result<E8pLaunch, E8pMoeError> {
    let kind = req.kind;
    let c = req.c;
    if !(1..=MAX_LANES).contains(&c) {
        return Err(E8pMoeError::InvalidLaneCount { c });
    }
    if req.u == 0 || req.u > max_union(c) {
        return Err(E8pMoeError::InvalidUnion { u: req.u, c });
    }
    req.expert_ids
        .check("expert_ids", mul(req.u, U32_BYTES, "expert_ids")?, 4)?;
    req.expert_offsets.check(
        "expert_offsets",
        mul(req.u + 1, U32_BYTES, "expert_offsets")?,
        4,
    )?;
    req.slot_ids
        .check("slot_ids", mul(TOP_K * c, U32_BYTES, "slot_ids")?, 4)?;
    // The table extent is the caller's (the layer's expert count), a nonnegative signed-i32: the kernel's one
    // unsigned compare then rejects every id `>= n_experts` and every negative id before the table is indexed.
    if req.n_experts == 0 || req.n_experts > i32::MAX as usize {
        return Err(E8pMoeError::Operand {
            name: "n_experts",
            reason: "outside 1..=i32::MAX",
        });
    }
    req.weight_ptrs.check(
        "weight_ptrs",
        mul(req.n_experts, PTR_BYTES, "weight_ptrs")?,
        8,
    )?;

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
            return Err(E8pMoeError::Operand {
                name: "y1",
                reason: "gate/up requires the up output",
            })
        }
        (false, None) => 0,
        (false, Some(_)) => {
            return Err(E8pMoeError::Operand {
                name: "y1",
                reason: "down has no second output",
            })
        }
    };
    let m = u32_of(kind.m(), "m")?;
    let u = u32_of(req.u, "u")?;
    Ok(E8pLaunch {
        kind,
        symbol: kind.symbol_on(arch),
        grid: [m, u, 1],
        block: [WORKGROUP, 1, 1],
        kernargs: E8pKernargs {
            expert_weight_ptrs: req.weight_ptrs.ptr,
            expert_ids: req.expert_ids.ptr,
            expert_offsets: req.expert_offsets.ptr,
            slot_ids: req.slot_ids.ptr,
            xrot: req.xrot.ptr,
            y0: req.y0.ptr,
            y1: y1_ptr,
            m,
            k: u32_of(kind.k(), "k")?,
            u,
            c: u32_of(c, "c")?,
            x_row_div: u32_of(kind.x_row_div(), "x_row_div")?,
            x_src_rows: u32_of(kind.x_src_rows(c), "x_src_rows")?,
            x_row_stride: u32_of(kind.k(), "x_row_stride")?,
            y_slot_stride: u32_of(kind.y_slot_stride(), "y_slot_stride")?,
            n_experts: u32_of(req.n_experts, "n_experts")?,
        },
    })
}

// ---------------------------------------------------------------------------
// Slab lane-union scratch geometry
// ---------------------------------------------------------------------------

/// Words of a lane-union scratch of `buckets` buckets for `slots = 10 * lanes`:
/// `B * (3 S + 1)` (the incumbent two-bucket planner writes `6 S + 2`).
pub const fn union_words(slots: usize, buckets: usize) -> usize {
    buckets * (3 * slots + 1)
}

/// `(ids, offsets, slot_ids)` word offsets of `bucket` inside a lane-union
/// scratch of `buckets` buckets: `ids[b*S]`, `offsets[B*S + b*(S+1)]`,
/// `slot_ids[B*S + B*(S+1) + b*S]`.
pub const fn union_bucket_words(
    slots: usize,
    buckets: usize,
    bucket: usize,
) -> (usize, usize, usize) {
    (
        bucket * slots,
        buckets * slots + bucket * (slots + 1),
        buckets * slots + buckets * (slots + 1) + bucket * slots,
    )
}

/// CPU form of one bucket's descriptor: ascending expert ids, offsets
/// `0..=n`, and the original slot ids grouped by expert in ascending slot
/// order (`slot = lane * 10 + rank`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct E8pUnionHost {
    pub expert_ids: Vec<u32>,
    pub expert_offsets: Vec<u32>,
    pub slot_ids: Vec<u32>,
}

impl E8pUnionHost {
    /// Groups the slots of `route` (`route[lane*10 + rank]` = expert id, ids
    /// `< 512`) whose expert `select`s true.
    pub fn from_route(route: &[u32], select: impl Fn(u32) -> bool) -> Result<Self, E8pMoeError> {
        if route.is_empty()
            || route.len() % TOP_K != 0
            || !(1..=MAX_LANES).contains(&(route.len() / TOP_K))
        {
            return Err(E8pMoeError::InvalidLaneCount {
                c: route.len() / TOP_K,
            });
        }
        if route.iter().any(|&e| e as usize >= MAX_EXPERTS) {
            return Err(E8pMoeError::Layout("expert id outside 0..512"));
        }
        let mut ids: Vec<u32> = route.iter().copied().filter(|&e| select(e)).collect();
        ids.sort_unstable();
        ids.dedup();
        let mut out = E8pUnionHost {
            expert_ids: ids,
            expert_offsets: vec![0],
            slot_ids: Vec::new(),
        };
        for &id in &out.expert_ids {
            for (slot, &e) in route.iter().enumerate() {
                if e == id {
                    out.slot_ids.push(slot as u32);
                }
            }
            out.expert_offsets.push(out.slot_ids.len() as u32);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Loader-independent launch through the HIP module API
// ---------------------------------------------------------------------------

fn require_arch(gpu: &Gpu) -> Result<E8pArch, E8pMoeError> {
    if gpu.arch_caps.is_gfx1201() {
        Ok(E8pArch::Gfx1201)
    } else if gpu.arch_caps.is_gfx1151() {
        Ok(E8pArch::Gfx1151)
    } else {
        Err(E8pMoeError::Arch(gpu.arch.clone()))
    }
}

/// The two K-E2 functions loaded from a gfx1201 or gfx1151 code object (`.hxaco` bundle or
/// bare `.co`), indexed by [`E8pKind::index`]. Independent of the loader, the
/// route files and the embedded-kernel registry; the HIP module owns its
/// functions, so this value must outlive every launch enqueued through it.
pub struct E8pMoeDecodeModule {
    _module: Module,
    functions: [Function; 2],
    arch: E8pArch,
}

impl E8pMoeDecodeModule {
    /// Loads `image` on `gpu`. Refuses any device that is not exactly gfx1201 or gfx1151
    /// and fails closed if a symbol of that arch's module is missing.
    pub fn load(gpu: &mut Gpu, image: &[u8]) -> HipResult<Self> {
        let arch = require_arch(gpu).map_err(hip_err)?;
        gpu.bind_thread()?;
        gpu.ensure_route_modules_preloaded()?;
        let module = gpu.hip.module_load_data(image)?;
        let symbols = arch.symbols();
        let functions = [
            gpu.hip.module_get_function(&module, symbols[0])?,
            gpu.hip.module_get_function(&module, symbols[1])?,
        ];
        Ok(Self {
            _module: module,
            functions,
            arch,
        })
    }

    /// The architecture whose symbols this module bound.
    pub fn arch(&self) -> E8pArch {
        self.arch
    }

    /// Launches exactly `kernargs` over `grid` for `kind` on the active stream,
    /// with **no validation of the operands** (the oracle's out-of-contract
    /// descriptor tests use this). Refused while a replay recording or graph
    /// capture is active.
    ///
    /// # Safety
    ///
    /// Every pointer in `kernargs` is either null where the kernel never
    /// dereferences it or a live device allocation on this device that stays
    /// live (and is not written by other unordered work, for the inputs) until
    /// the enqueued work completes; this value outlives that work.
    pub unsafe fn launch_kernargs(
        &self,
        gpu: &mut Gpu,
        kind: E8pKind,
        grid: [u32; 3],
        kernargs: &E8pKernargs,
    ) -> HipResult<()> {
        if require_arch(gpu).map_err(hip_err)? != self.arch {
            return Err(hip_err(E8pMoeError::Arch(gpu.arch.clone())));
        }
        if gpu.replay.is_recording() || gpu.graphs.capture_mode {
            return Err(hip_err(E8pMoeError::Unavailable(
                "E8pMoeDecodeModule launch is not recordable or capturable; use Gpu::e8p_i4_moe_gemv_slab",
            )));
        }
        gpu.bind_thread()?;
        gpu.ensure_route_modules_preloaded()?;
        let mut bytes = kernargs.to_bytes();
        // SAFETY: `bytes` is the exact frozen 96-byte kernarg of the symbol; the pointer
        // and lifetime preconditions are this function's contract.
        unsafe {
            gpu.hip.launch_kernel_blob(
                &self.functions[kind.index()],
                grid,
                [WORKGROUP, 1, 1],
                0,
                gpu.active_stream.as_ref(),
                &mut bytes,
            )
        }
        .map_err(|e| e.with_kernel(kind.symbol_on(self.arch)))
    }

    /// Validates `req` with [`plan_launch_on`] (this module's arch) and launches it.
    ///
    /// # Safety
    ///
    /// `plan_launch_on` checks only shapes, extents, alignment and aliasing of the
    /// raw [`DevView`] addresses. The caller guarantees they name live allocations
    /// of at least the declared extent on this device, until the enqueued work
    /// completes; the descriptor arrays and `weight_ptrs` are immutable and
    /// consistent (ids `< n_experts` ascending, offsets from `0` ascending, slot ids
    /// `< 10 C`); every selected `weight_ptrs[id]` points to a live blob of
    /// exactly [`E8pKind::blob_bytes`] bytes with writer-guarded scales; this value
    /// outlives the work.
    pub unsafe fn launch(&self, gpu: &mut Gpu, req: &E8pDecodeRequest) -> HipResult<()> {
        let launch = plan_launch_on(self.arch, req).map_err(hip_err)?;
        // SAFETY: forwarded contract.
        unsafe { self.launch_kernargs(gpu, launch.kind, launch.grid, &launch.kernargs) }
    }
}

// ---------------------------------------------------------------------------
// Recorded / capturable launch through the shared function cache
// ---------------------------------------------------------------------------

impl Gpu {
    /// Binds both K-E2 symbols of this device's arch to `image` (the emitted
    /// `e8p_i4_moe_gemv_pm_gfx1201` / `e8p_i4_moe_gemv_pm_gfx1151` code object, see
    /// [`E8pArch`]) in the shared module/function
    /// cache. Call before graph capture. `image` is `Arc<[u8]>` (read from a file)
    /// or a `&'static [u8]` embedded constant; a later call with a different image
    /// is refused.
    pub fn ensure_e8p_i4_moe_gemv_kernels(
        &mut self,
        image: impl Into<redline_dispatch::aql::CodeObjectBytes> + Clone,
    ) -> HipResult<()> {
        let arch = require_arch(self).map_err(hip_err)?;
        self.bind_thread()?;
        for symbol in arch.symbols() {
            self.ensure_embedded_kernel(arch.module(), image.clone(), symbol)?;
        }
        Ok(())
    }

    /// Launch one projection over the E8P bucket `bucket` of `buckets` of the slab
    /// lane-union scratch the planner wrote for `lanes` lanes (layout:
    /// [`union_bucket_words`]). `weight_ptrs` is the layer's `[512]` u64 blob-base
    /// table of the projection. Gate/up: `xrot` `[lanes, 2560]`, `y0`/`y1`
    /// `[lanes, 10, 640]`; down: `xrot` `[10 * lanes, 640]`, `y0` `[lanes, 10, 2560]`
    /// (unweighted), `y1` `None`. Stream-ordered, recorded by the replay recorder and
    /// captured by graphs; the kernels must already be bound
    /// ([`Gpu::ensure_e8p_i4_moe_gemv_kernels`]).
    #[allow(clippy::too_many_arguments)]
    pub fn e8p_i4_moe_gemv_slab(
        &mut self,
        down: bool,
        buckets: usize,
        bucket: usize,
        weight_ptrs: &GpuTensor,
        xrot: &GpuTensor,
        y0: &GpuTensor,
        y1: Option<&GpuTensor>,
        lanes: usize,
    ) -> HipResult<()> {
        let arch = require_arch(self).map_err(hip_err)?;
        if buckets == 0 || bucket >= buckets {
            return Err(hip_err(E8pMoeError::Layout(
                "bucket index outside the bucket count",
            )));
        }
        let kind = if down { E8pKind::Down } else { E8pKind::GateUp };
        if !(1..=MAX_LANES).contains(&lanes) {
            return Err(hip_err(E8pMoeError::InvalidLaneCount { c: lanes }));
        }
        let slots = TOP_K * lanes;
        let (ids_w, offs_w, slots_w) = union_bucket_words(slots, buckets, bucket);
        let union_buf = self
            .scratch
            .qwen4_slab_union_scratch
            .as_ref()
            .ok_or_else(|| hip_err(E8pMoeError::Layout("slab union scratch is not reserved")))?;
        if union_buf.size() < union_words(slots, buckets) * U32_BYTES {
            return Err(hip_err(E8pMoeError::Layout(
                "slab union scratch is smaller than the bucket layout",
            )));
        }
        let base = union_buf.as_ptr() as usize as u64;
        let word = |w: usize| base + (w * U32_BYTES) as u64;
        let req = E8pDecodeRequest {
            kind,
            c: lanes,
            u: slots,
            expert_ids: DevView::new(word(ids_w), slots * U32_BYTES),
            expert_offsets: DevView::new(word(offs_w), (slots + 1) * U32_BYTES),
            slot_ids: DevView::new(word(slots_w), slots * U32_BYTES),
            weight_ptrs: DevView::from_tensor(weight_ptrs),
            xrot: DevView::from_tensor(xrot),
            y0: DevView::from_tensor(y0),
            y1: y1.map(DevView::from_tensor),
            // The layer's full pointer-table extent: the slab convention (`mq_i4_moe_gemv_slab`: `SLAB_EXPERTS`, with
            // `weight_ptrs` required to hold that many entries, re-checked by `plan_launch`), never the bucket's `U`.
            n_experts: crate::slab_moe::SLAB_EXPERTS,
        };
        let launch = plan_launch_on(arch, &req).map_err(hip_err)?;
        let symbol = launch.symbol;
        if !self.functions.contains_key(symbol) {
            return Err(hip_err(E8pMoeError::Unavailable(
                "e8p_i4_moe_gemv symbols are not bound; call ensure_e8p_i4_moe_gemv_kernels first",
            )));
        }
        self.bind_thread()?;
        let k = launch.kernargs;
        // Metadata lists reserved@92 as argument 17, not implicit struct padding.
        let reserved = 0u32;
        let mut params: [*mut c_void; 17] = [
            &k.expert_weight_ptrs as *const _ as *mut c_void,
            &k.expert_ids as *const _ as *mut c_void,
            &k.expert_offsets as *const _ as *mut c_void,
            &k.slot_ids as *const _ as *mut c_void,
            &k.xrot as *const _ as *mut c_void,
            &k.y0 as *const _ as *mut c_void,
            &k.y1 as *const _ as *mut c_void,
            &k.m as *const _ as *mut c_void,
            &k.k as *const _ as *mut c_void,
            &k.u as *const _ as *mut c_void,
            &k.c as *const _ as *mut c_void,
            &k.x_row_div as *const _ as *mut c_void,
            &k.x_src_rows as *const _ as *mut c_void,
            &k.x_row_stride as *const _ as *mut c_void,
            &k.y_slot_stride as *const _ as *mut c_void,
            &k.n_experts as *const _ as *mut c_void,
            &reserved as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(symbol, launch.grid, launch.block, 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            for p in [
                k.expert_weight_ptrs,
                k.expert_ids,
                k.expert_offsets,
                k.slot_ids,
                k.xrot,
                k.y0,
                k.y1,
            ] {
                b.push_ptr(p as usize as *mut c_void);
            }
            for w in [
                k.m,
                k.k,
                k.u,
                k.c,
                k.x_row_div,
                k.x_src_rows,
                k.x_row_stride,
                k.y_slot_stride,
                k.n_experts,
                0,
            ] {
                b.push_u32(w);
            }
            b
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn views(c: usize, kind: E8pKind) -> E8pDecodeRequest {
        let u = TOP_K * c;
        // Disjoint synthetic addresses, 1 MiB apart each of >= 32 MiB: no device access.
        let at = |i: u64, bytes: usize| DevView::new(0x1000_0000 + i * 0x1000_0000, bytes);
        E8pDecodeRequest {
            kind,
            c,
            u,
            expert_ids: at(0, u * 4),
            expert_offsets: at(1, (u + 1) * 4),
            slot_ids: at(2, TOP_K * c * 4),
            weight_ptrs: at(3, MAX_EXPERTS * 8),
            n_experts: MAX_EXPERTS,
            xrot: at(4, kind.x_src_rows(c) * kind.k() * 4),
            y0: at(5, c * TOP_K * kind.y_slot_stride() * 4),
            y1: kind
                .has_y1()
                .then(|| at(6, c * TOP_K * kind.y_slot_stride() * 4)),
        }
    }

    #[test]
    fn blob_geometry_matches_the_wire() {
        assert_eq!(E8pKind::GateUp.blob_bytes(), 870_400);
        assert_eq!(E8pKind::Down.blob_bytes(), 435_200);
        assert_eq!(E8pKind::GateUp.scales_offset(), 819_200);
        assert_eq!(E8pKind::Down.scales_offset(), 409_600);
        assert_eq!(
            E8P_I4_MOE_GEMV_SYMBOLS[E8pKind::GateUp.index()],
            "e8p_i4_moe_gemv_r256_gate_up"
        );
        assert_eq!(
            E8P_I4_MOE_GEMV_SYMBOLS[E8pKind::Down.index()],
            "e8p_i4_moe_gemv_r128_down"
        );
    }

    fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
        haystack.windows(needle.len()).any(|w| w == needle)
    }

    #[test]
    fn arch_table_is_exactly_gfx1201_and_gfx1151_with_the_gfx1201_names_unchanged() {
        assert_eq!(E8pArch::of_name("gfx1201"), Some(E8pArch::Gfx1201));
        assert_eq!(E8pArch::of_name("gfx1151"), Some(E8pArch::Gfx1151));
        for other in [
            "gfx1100", "gfx1101", "gfx1152", "gfx1200", "gfx942", "GFX1151", "gfx1151x", "",
        ] {
            assert_eq!(E8pArch::of_name(other), None, "{other:?}");
        }
        for arch in E8pArch::ALL {
            assert_eq!(E8pArch::of_name(arch.name()), Some(arch));
        }
        // gfx1201: the historical module, symbols and image, bit for bit.
        assert_eq!(E8pArch::Gfx1201.module(), E8P_I4_MOE_GEMV_MODULE);
        assert_eq!(E8pArch::Gfx1201.module(), "e8p_i4_moe_gemv_pm_gfx1201");
        assert_eq!(E8pArch::Gfx1201.symbols(), E8P_I4_MOE_GEMV_SYMBOLS);
        assert_eq!(
            E8pArch::Gfx1201.image(),
            crate::kernels::E8P_I4_MOE_GEMV_PM_GFX1201
        );
        // gfx1151: its own module, the `_pm_gfx1151` symbols and its own image.
        assert_eq!(E8pArch::Gfx1151.module(), "e8p_i4_moe_gemv_pm_gfx1151");
        assert_eq!(
            E8pArch::Gfx1151.symbols(),
            [
                "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1151",
                "e8p_i4_moe_gemv_r128_down_pm_gfx1151"
            ]
        );
        assert_eq!(
            E8pArch::Gfx1151.image(),
            crate::kernels::E8P_I4_MOE_GEMV_PM_GFX1151
        );
        for kind in E8pKind::ALL {
            assert_eq!(kind.symbol(), kind.symbol_on(E8pArch::Gfx1201));
            assert_eq!(
                kind.symbol_on(E8pArch::Gfx1151),
                E8pArch::Gfx1151.symbols()[kind.index()]
            );
            assert_eq!(
                kind.symbol_on(E8pArch::Gfx1151),
                format!("{}_pm_gfx1151", kind.symbol())
            );
        }
        for arch in E8pArch::ALL {
            for symbol in arch.symbols() {
                assert!(
                    contains_bytes(arch.image(), format!("{symbol}.kd").as_bytes()),
                    "{} {symbol}.kd",
                    arch.name()
                );
            }
        }
        // Each image carries its own symbols only.
        for symbol in E8pArch::Gfx1151.symbols() {
            assert!(
                !contains_bytes(E8pArch::Gfx1201.image(), symbol.as_bytes()),
                "{symbol} in the gfx1201 image"
            );
        }
        assert_ne!(E8pArch::Gfx1201.image(), E8pArch::Gfx1151.image());
    }

    #[test]
    fn plan_launch_on_changes_only_the_symbol_per_arch() {
        for c in 1..=MAX_LANES {
            for kind in E8pKind::ALL {
                let req = views(c, kind);
                let base = plan_launch(&req).unwrap();
                assert_eq!(base, plan_launch_on(E8pArch::Gfx1201, &req).unwrap());
                assert_eq!(base.symbol, kind.symbol());
                let halo = plan_launch_on(E8pArch::Gfx1151, &req).unwrap();
                assert_eq!(halo.symbol, kind.symbol_on(E8pArch::Gfx1151));
                assert_eq!(
                    (halo.kind, halo.grid, halo.block, halo.kernargs),
                    (base.kind, base.grid, base.block, base.kernargs)
                );
            }
        }
    }

    #[test]
    fn kernarg_bytes_follow_the_frozen_abi() {
        let k = E8pKernargs {
            expert_weight_ptrs: 0x11,
            expert_ids: 0x22,
            expert_offsets: 0x33,
            slot_ids: 0x44,
            xrot: 0x55,
            y0: 0x66,
            y1: 0x77,
            m: 1280,
            k: 2560,
            u: 10,
            c: 1,
            x_row_div: 10,
            x_src_rows: 1,
            x_row_stride: 2560,
            y_slot_stride: 640,
            n_experts: 512,
        };
        let b = k.to_bytes();
        assert_eq!(b.len(), 96);
        for (i, want) in [0x11u64, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77]
            .iter()
            .enumerate()
        {
            assert_eq!(
                u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap()),
                *want
            );
        }
        let w = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        assert_eq!(
            [
                w(56),
                w(60),
                w(64),
                w(68),
                w(72),
                w(76),
                w(80),
                w(84),
                w(88)
            ],
            [1280, 2560, 10, 1, 10, 1, 2560, 640, 512]
        );
        assert_eq!(w(92), 0, "the reserved word is zero");
        assert_eq!(k.grid(), [1280, 10, 1]);
    }

    #[test]
    fn plan_matches_the_incumbent_f4_geometry_for_c1_to_c8() {
        for c in 1..=MAX_LANES {
            let gate = plan_launch(&views(c, E8pKind::GateUp)).unwrap();
            let u = (TOP_K * c) as u32;
            assert_eq!(gate.grid, [1280, u, 1]);
            assert_eq!(gate.block, [32, 1, 1]);
            assert_eq!(gate.symbol, "e8p_i4_moe_gemv_r256_gate_up");
            let k = gate.kernargs;
            assert_eq!(
                (
                    k.m,
                    k.k,
                    k.u,
                    k.c,
                    k.x_row_div,
                    k.x_src_rows,
                    k.x_row_stride,
                    k.y_slot_stride,
                    k.n_experts
                ),
                (
                    1280,
                    2560,
                    u,
                    c as u32,
                    10,
                    c as u32,
                    2560,
                    640,
                    MAX_EXPERTS as u32
                )
            );
            assert_ne!(k.y1, 0);
            let down = plan_launch(&views(c, E8pKind::Down)).unwrap();
            assert_eq!(down.grid, [2560, u, 1]);
            assert_eq!(down.symbol, "e8p_i4_moe_gemv_r128_down");
            let k = down.kernargs;
            assert_eq!(
                (
                    k.m,
                    k.k,
                    k.x_row_div,
                    k.x_src_rows,
                    k.x_row_stride,
                    k.y_slot_stride,
                    k.y1
                ),
                (2560, 640, 1, u, 640, 2560, 0)
            );
        }
    }

    #[test]
    fn plan_refuses_out_of_contract_requests() {
        let mut r = views(1, E8pKind::GateUp);
        r.c = 0;
        assert_eq!(plan_launch(&r), Err(E8pMoeError::InvalidLaneCount { c: 0 }));
        let mut r = views(1, E8pKind::GateUp);
        r.c = 9;
        assert!(matches!(
            plan_launch(&r),
            Err(E8pMoeError::InvalidLaneCount { c: 9 })
        ));
        let mut r = views(1, E8pKind::GateUp);
        r.u = 0;
        assert!(matches!(
            plan_launch(&r),
            Err(E8pMoeError::InvalidUnion { .. })
        ));
        let mut r = views(1, E8pKind::GateUp);
        r.u = 11;
        assert!(matches!(
            plan_launch(&r),
            Err(E8pMoeError::InvalidUnion { .. })
        ));
        // U is bounded by 512 for C8 (80 slots) trivially, and by 10 C below it.
        let mut r = views(8, E8pKind::Down);
        r.u = 80;
        assert!(plan_launch(&r).is_ok());
        r.u = 81;
        assert!(matches!(
            plan_launch(&r),
            Err(E8pMoeError::InvalidUnion { .. })
        ));

        let mut r = views(2, E8pKind::GateUp);
        r.y1 = None;
        assert!(matches!(
            plan_launch(&r),
            Err(E8pMoeError::Operand { name: "y1", .. })
        ));
        let mut r = views(2, E8pKind::Down);
        r.y1 = Some(DevView::new(0x9000_0000, 1 << 24));
        assert!(matches!(
            plan_launch(&r),
            Err(E8pMoeError::Operand { name: "y1", .. })
        ));

        let mut r = views(2, E8pKind::GateUp);
        r.xrot.ptr = 0;
        assert!(matches!(
            plan_launch(&r),
            Err(E8pMoeError::Operand {
                name: "xrot",
                reason: "null device pointer"
            })
        ));
        let mut r = views(2, E8pKind::GateUp);
        r.weight_ptrs.ptr += 4;
        assert!(matches!(
            plan_launch(&r),
            Err(E8pMoeError::Operand {
                name: "weight_ptrs",
                ..
            })
        ));
        let mut r = views(2, E8pKind::GateUp);
        r.xrot.bytes -= 4;
        assert!(matches!(
            plan_launch(&r),
            Err(E8pMoeError::Operand {
                name: "xrot",
                reason: "allocation smaller than the launch extent"
            })
        ));
        let mut r = views(2, E8pKind::Down);
        r.y0.ptr = r.xrot.ptr + 4;
        assert!(matches!(
            plan_launch(&r),
            Err(E8pMoeError::Operand {
                name: "y0",
                reason: "overlaps another operand of the launch"
            })
        ));
        let mut r = views(2, E8pKind::GateUp);
        r.slot_ids.bytes = 10 * 2 * 4 - 4;
        assert!(matches!(
            plan_launch(&r),
            Err(E8pMoeError::Operand {
                name: "slot_ids",
                ..
            })
        ));
    }

    #[test]
    fn plan_validates_the_table_extent_against_the_signed_i32_contract() {
        let r = views(1, E8pKind::GateUp);
        assert_eq!(
            plan_launch(&r).unwrap().kernargs.n_experts,
            MAX_EXPERTS as u32
        );
        // The extent is the caller's table extent, not a constant: a smaller table is launched as such.
        let mut r = views(1, E8pKind::GateUp);
        r.n_experts = 16;
        r.weight_ptrs.bytes = 16 * PTR_BYTES;
        assert_eq!(plan_launch(&r).unwrap().kernargs.n_experts, 16);
        r.weight_ptrs.bytes -= 1;
        assert!(matches!(
            plan_launch(&r),
            Err(E8pMoeError::Operand {
                name: "weight_ptrs",
                ..
            })
        ));
        for bad in [0usize, i32::MAX as usize + 1, u32::MAX as usize, usize::MAX] {
            let mut r = views(1, E8pKind::GateUp);
            r.n_experts = bad;
            assert!(
                matches!(
                    plan_launch(&r),
                    Err(E8pMoeError::Operand {
                        name: "n_experts",
                        ..
                    })
                ),
                "n_experts {bad} must be refused"
            );
        }
        let mut r = views(1, E8pKind::GateUp);
        r.n_experts = i32::MAX as usize;
        assert!(matches!(
            plan_launch(&r),
            Err(E8pMoeError::Operand {
                name: "weight_ptrs",
                ..
            })
        ));
    }

    #[test]
    fn module_extent_is_the_slab_layer_extent() {
        assert_eq!(MAX_EXPERTS, crate::slab_moe::SLAB_EXPERTS);
    }

    #[test]
    fn union_scratch_layout_extends_the_incumbent_two_bucket_planner() {
        for lanes in 1..=MAX_LANES {
            let s = TOP_K * lanes;
            assert_eq!(union_words(s, 2), 6 * s + 2);
            // The incumbent `slab_moe` formulas for B = 2.
            for b in 0..2 {
                assert_eq!(
                    union_bucket_words(s, 2, b),
                    (b * s, 2 * s + b * (s + 1), 2 * s + 2 * (s + 1) + b * s)
                );
            }
            // B = 3: disjoint, in-bounds, packed.
            let total = union_words(s, 3);
            assert_eq!(total, 9 * s + 3);
            let mut last = 0;
            for b in 0..3 {
                let (i, o, sl) = union_bucket_words(s, 3, b);
                assert!(i + s <= 3 * s && o + s + 1 <= 3 * s + 3 * (s + 1) && sl + s <= total);
                last = last.max(sl + s);
            }
            assert_eq!(last, total);
        }
    }

    #[test]
    fn host_union_groups_slots_by_expert_in_original_order() {
        // C2: lane 0 = 0..10, lane 1 = 5..15 (experts 5..9 shared).
        let route: Vec<u32> = (0..10).chain(5..15).collect();
        let u = E8pUnionHost::from_route(&route, |_| true).unwrap();
        assert_eq!(u.expert_ids, (0..15).collect::<Vec<u32>>());
        assert_eq!(u.expert_offsets.len(), 16);
        assert_eq!(u.slot_ids.len(), 20);
        // Expert 5 sits at slots 5 (lane 0) and 10 (lane 1).
        let at = u.expert_ids.iter().position(|&e| e == 5).unwrap();
        assert_eq!(
            &u.slot_ids[u.expert_offsets[at] as usize..u.expert_offsets[at + 1] as usize],
            &[5, 10]
        );
        assert_eq!(*u.expert_offsets.last().unwrap(), 20);
        // A selecting predicate partitions the slots.
        let even = E8pUnionHost::from_route(&route, |e| e % 2 == 0).unwrap();
        let odd = E8pUnionHost::from_route(&route, |e| e % 2 == 1).unwrap();
        assert_eq!(even.slot_ids.len() + odd.slot_ids.len(), 20);
        let none = E8pUnionHost::from_route(&route, |_| false).unwrap();
        assert!(
            none.expert_ids.is_empty() && none.expert_offsets == [0] && none.slot_ids.is_empty()
        );
        assert!(E8pUnionHost::from_route(&[0; 7], |_| true).is_err());
        assert!(E8pUnionHost::from_route(&[512; 10], |_| true).is_err());
    }
}
