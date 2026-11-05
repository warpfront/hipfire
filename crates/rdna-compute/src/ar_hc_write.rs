// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201 TP2 all-reduce fused into the BF16 Hyper-Connection write
//! (`kernels/src/ar_hc_write.gfx1201.hip`). The runtime owns the fine-grained
//! mailboxes and the generation words ([`ArHcWriteMailbox`] layout below); this module
//! resolves the kernel, picks the launch geometry, launches it, and carries the
//! host numerics model used by the oracles.
//!
//! Wire: one 16-byte granule is `{bf16(p0) | bf16(p1) << 16, tag, bf16(p2) | bf16(p3) << 16,
//! tag}` for four consecutive partial floats; a granule is accepted only when both tags equal
//! the launch's generation. A rank's mailbox holds two banks (generation parity) of
//! [`ar_hc_bank_bytes`] bytes that the PEER writes.
//!
//! Numerics (the identity contract of the TP2 trunk shard): the fold is
//! `m = 0.0f + (bf16(p_first) + bf16(p_second))` in rank order, fp32 add of two
//! bf16-rounded partials, identical bits on both ranks, followed by the production
//! `hyper_write_bf16x2` expression. TP2 is therefore NOT byte-equal to one card (the
//! partials split a K sum and each is rounded to bf16 once); the per-element bound against
//! the exact f32 sum is [`ar_hc_fold_bound`].

use crate::dispatch::Gpu;
use crate::kernels;
use hip_bridge::{DeviceBuffer, HipError, HipResult, KernargBlob};
use std::ffi::c_void;

/// Hidden width of one HC stream (Flash-Next).
pub const AR_HC_HIDDEN: usize = 2560;
/// Bytes per tagged granule (four bf16 partial values and the tag twice).
pub const AR_HC_GRANULE_BYTES: usize = 16;
/// Partial floats per granule.
pub const AR_HC_GRANULE_VALUES: usize = 4;
/// Granules per lane the kernel is instantiated for.
pub const AR_HC_KMAX: usize = 6;
/// Largest block count of a launch.
pub const AR_HC_MAX_BLOCKS: usize = 64;
/// Largest lane count of a block.
pub const AR_HC_MAX_LANES: usize = 512;
/// Most rows one launch serves (granules per launch must fit `AR_HC_MAX_BLOCKS` blocks of
/// `AR_HC_MAX_LANES` lanes at `AR_HC_KMAX` each: 307 rows; 256 keeps a power of two).
pub const AR_HC_ROW_CAP: usize = 256;
/// Default launch shape, the Gate 0b best at 8 rows (16 blocks x 256 lanes).
pub const AR_HC_DEFAULT_BLOCKS: u32 = 16;
pub const AR_HC_DEFAULT_LANES: u32 = 256;
/// Word index (u32) of the arrival ticket inside a generation block (own 64-byte line).
pub const AR_HC_TICKET_WORD: usize = 16;
/// Bytes of a rank's generation block; word 0 is the generation, word 16 the ticket.
pub const AR_HC_GEN_BYTES: usize = 256;

const KERNEL: &str = "ar_hc_write_gfx1201";
const KERNEL_F32S: &str = "ar_hc_write_f32s_gfx1201";
const KERNEL_D8: &str = "ar_hc_write_d8_gfx1201";
const KERNEL_D8_F32S: &str = "ar_hc_write_d8_f32s_gfx1201";
const KERNEL_S: &str = "ar_hc_write_s_gfx1201";
const KERNEL_S_F32S: &str = "ar_hc_write_s_f32s_gfx1201";
/// Most rows of a branch-split launch (tagged wire, `HIPFIRE_EP_AR_FAST_APPLY`): one lane per
/// (granule, branch) item, 4 * rows * 640 items within [`AR_HC_MAX_BLOCKS`] blocks of at most
/// [`AR_HC_MAX_LANES`] lanes (12 rows: 64 x 480).
// selector-receipt: receipts/selectors/ar_hc_split_max_rows.json
pub const AR_HC_SPLIT_MAX_ROWS: usize = 12;

/// The mailbox wire format of one transport (fixed per [`ArHcExchange`] at construction).
///
/// * `Tagged`: the original 16-byte granule `{d0d1, tag, d2d3, tag}` (8 payload bytes per 16
///   wire bytes), validated by the generation tag. The kill-switch format.
/// * `Dense`: an 8-byte granule `{d0d1, d2d3}` (all 8 bytes payload), validated per dword by
///   the EMPTY marker `0xFFFF_FFFF`; the consumer re-arms what it read, so the mailbox is
///   EMPTY at the start of every launch (see the kernel header for the argument).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArHcWire {
    Tagged,
    Dense,
}

impl ArHcWire {
    /// Wire bytes of one granule (four bf16 partial values).
    pub fn granule_bytes(self) -> usize {
        match self {
            Self::Tagged => 16,
            Self::Dense => 8,
        }
    }

    /// Byte every mailbox byte holds before a launch writes it: tag 0 never matches a live
    /// generation (`0x00`), and the dense EMPTY dword is all ones (`0xFF`).
    pub fn idle_fill(self) -> u8 {
        match self {
            Self::Tagged => 0x00,
            Self::Dense => 0xFF,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Tagged => "tagged",
            Self::Dense => "dense",
        }
    }
}

/// Storage of the HC streams the fused write updates in place (the two production HC
/// writes): BF16 bits (`hyper_write_bf16x2`, the F16 prefill route, rows at or above
/// `QWEN4_F16_WMMA_MIN_TOKENS`) or F32 words holding BF16-exact values (`hyper_write_f32`:
/// decode, few-row verify and short chunks). The arithmetic per element is the same; only
/// the stream load/store width and row stride differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StreamKind {
    Bf16,
    F32,
}

impl StreamKind {
    /// The kind of the forward's HC streams (`state_bf16` of the HC ops).
    pub fn from_state_bf16(state_bf16: bool) -> Self {
        if state_bf16 {
            Self::Bf16
        } else {
            Self::F32
        }
    }

    /// The device entry of this kind on `wire`.
    pub fn symbol(self, wire: ArHcWire) -> &'static str {
        match (wire, self) {
            (ArHcWire::Tagged, Self::Bf16) => KERNEL,
            (ArHcWire::Tagged, Self::F32) => KERNEL_F32S,
            (ArHcWire::Dense, Self::Bf16) => KERNEL_D8,
            (ArHcWire::Dense, Self::F32) => KERNEL_D8_F32S,
        }
    }
    /// The branch-split tagged entry of this kind ([`ar_hc_split_geometry`]).
    pub fn split_symbol(self) -> &'static str {
        match self {
            Self::Bf16 => KERNEL_S,
            Self::F32 => KERNEL_S_F32S,
        }
    }
    /// Bytes per stream element.
    pub fn element_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }

    /// Bytes of one row of the four streams (`[4][2560]`).
    pub fn row_bytes(self) -> usize {
        4 * AR_HC_HIDDEN * self.element_bytes()
    }

    /// Required byte alignment of the stream base (8 for the u16 pair loads, 16 for the
    /// four-float loads).
    pub fn alignment(self) -> usize {
        match self {
            Self::Bf16 => 8,
            Self::F32 => 16,
        }
    }

    /// Both kinds, in report order.
    pub const ALL: [Self; 2] = [Self::Bf16, Self::F32];

    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
        }
    }
}

/// Bytes of one mailbox bank able to hold `max_rows` rows of partials on the TAGGED wire (the
/// larger of the two: a dense bank fits in it).
pub fn ar_hc_bank_bytes(max_rows: usize) -> usize {
    ar_hc_bank_bytes_for(ArHcWire::Tagged, max_rows)
}

/// Bytes of one mailbox bank able to hold `max_rows` rows of partials on `wire`.
pub fn ar_hc_bank_bytes_for(wire: ArHcWire, max_rows: usize) -> usize {
    (max_rows * AR_HC_HIDDEN / AR_HC_GRANULE_VALUES * wire.granule_bytes()).next_multiple_of(256)
}

/// Launch geometry for `rows` rows: `(blocks, lanes)`. Granules per lane stay within
/// [`AR_HC_KMAX`] (more blocks first, at most [`AR_HC_MAX_BLOCKS`]); the kernel slices
/// `per = ceil(G / blocks)` granules per block with lane stride `lanes`.
pub fn ar_hc_geometry_with(rows: usize, blocks0: u32, lanes0: u32) -> Option<(u32, u32)> {
    ar_hc_geometry_exact(rows, blocks0, lanes0).or_else(|| {
        // 64 blocks of the requested lanes cannot hold this many granules at KMAX each:
        // widen the blocks to the maximum lane count.
        (lanes0 as usize != AR_HC_MAX_LANES)
            .then(|| ar_hc_geometry_exact(rows, blocks0, AR_HC_MAX_LANES as u32))
            .flatten()
    })
}

fn ar_hc_geometry_exact(rows: usize, blocks0: u32, lanes0: u32) -> Option<(u32, u32)> {
    if rows == 0 || rows > AR_HC_ROW_CAP || lanes0 == 0 || blocks0 == 0 {
        return None;
    }
    let granules = rows * AR_HC_HIDDEN / AR_HC_GRANULE_VALUES;
    let mut b = (blocks0 as usize)
        .min(granules.div_ceil(32))
        .clamp(1, AR_HC_MAX_BLOCKS);
    loop {
        let per = granules.div_ceil(b);
        let t = (lanes0 as usize)
            .min(per.div_ceil(32) * 32)
            .clamp(32, AR_HC_MAX_LANES);
        if per.div_ceil(t) <= AR_HC_KMAX {
            return Some((b as u32, t as u32));
        }
        if b >= AR_HC_MAX_BLOCKS {
            return None;
        }
        b += 1;
    }
}

/// [`ar_hc_geometry_with`] at the default shape.
pub fn ar_hc_geometry(rows: usize) -> Option<(u32, u32)> {
    ar_hc_geometry_with(rows, AR_HC_DEFAULT_BLOCKS, AR_HC_DEFAULT_LANES)
}

/// Launch geometry `(blocks, lanes)` of a branch-split launch of `rows` rows (1 ..=
/// [`AR_HC_SPLIT_MAX_ROWS`]): `items = 4 * granules` (one per lane), up to 64 lanes per block
/// spread over at most [`AR_HC_MAX_BLOCKS`] blocks; the kernel's per-block item count
/// `per = round_up4(ceil(items / blocks))` fits the lanes (a granule's four items share a
/// block and a wave).
pub fn ar_hc_split_geometry(rows: usize) -> Option<(u32, u32)> {
    if rows == 0 || rows > AR_HC_SPLIT_MAX_ROWS {
        return None;
    }
    let items = rows * AR_HC_HIDDEN;
    let b = items.div_ceil(64).min(AR_HC_MAX_BLOCKS);
    let per = items.div_ceil(b).next_multiple_of(4);
    let t = per.next_multiple_of(32);
    (t <= AR_HC_MAX_LANES).then_some((b as u32, t as u32))
}

impl Gpu {
    /// Resolve the kernel before graph capture (JIT is not capture-safe).
    pub fn ensure_ar_hc_write_gfx1201_kernel(
        &mut self,
        kind: StreamKind,
        wire: ArHcWire,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch != "gfx1201" {
            return Err(HipError::new(
                0,
                &format!("ar_hc_write_gfx1201 requires gfx1201, got {}", self.arch),
            ));
        }
        let src = if self.flags.ep_ar_fast_apply_enabled() {
            kernels::AR_HC_WRITE_GFX1201_SRC
        } else {
            kernels::AR_HC_WRITE_GFX1201_HIPBF16_SRC
        };
        self.ensure_kernel(KERNEL, src, kind.symbol(wire))?;
        if wire == ArHcWire::Tagged && self.flags.ep_ar_fast_apply_enabled() {
            self.ensure_kernel(KERNEL, src, kind.split_symbol())?;
        }
        Ok(())
    }

    /// Push this rank's f32 partial `[rows, 2560]` (bf16-rounded) into the peer's mailbox,
    /// wait (bounded) for the peer's, fold in rank order and apply the production BF16
    /// HC write to `streams` (`[rows, 4, 2560]` of `kind`, in place) with the row gate
    /// logits `gates` (`[rows, 4]` f32). Operands are device pointers (the caller slices
    /// long row ranges into <= [`AR_HC_ROW_CAP`] launches by offsetting them).
    /// `peer_mailbox` is the PEER's mailbox (peer access enabled), `local_mailbox` this
    /// rank's own; each holds two banks `bank_stride` bytes apart. `generation` is this
    /// rank's [`AR_HC_GEN_BYTES`] block. `mixed_dbg` (oracle only, null otherwise) receives
    /// the folded `m`. One launch. Tagged launches of at most [`AR_HC_SPLIT_MAX_ROWS`] rows
    /// take the branch-split entry at [`ar_hc_split_geometry`] instead of `geometry`
    /// (`HIPFIRE_EP_AR_FAST_APPLY`, byte-identical).
    ///
    /// # Safety
    /// Every pointer must address live device memory of the extents above on this rank's
    /// device (the mailboxes through peer access), and both ranks must issue the same
    /// launches in the same order.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ar_hc_write_gfx1201(
        &mut self,
        partial: *mut c_void,
        streams: *mut c_void,
        kind: StreamKind,
        wire: ArHcWire,
        gates: *const c_void,
        peer_mailbox: *mut c_void,
        local_mailbox: *const c_void,
        bank_stride: usize,
        generation: &DeviceBuffer,
        local_is_first: bool,
        rows: usize,
        geometry: (u32, u32),
        mixed_dbg: *mut c_void,
    ) -> HipResult<()> {
        self.ensure_ar_hc_write_gfx1201_kernel(kind, wire)?;
        let fail = |what: &str| HipError::new(0, &format!("ar_hc_write_gfx1201: {what}"));
        if ar_hc_geometry_with(rows, geometry.0, geometry.1).is_none() {
            return Err(fail(&format!("unsupported row count {rows}")));
        }
        if geometry.0 as usize > AR_HC_MAX_BLOCKS
            || geometry.1 as usize > AR_HC_MAX_LANES
            || geometry.0 == 0
            || geometry.1 == 0
            || ar_hc_granules_per_lane(rows, geometry) > AR_HC_KMAX
        {
            return Err(fail(&format!(
                "invalid geometry {geometry:?} for {rows} rows"
            )));
        }
        let count = rows * AR_HC_HIDDEN;
        if bank_stride < ar_hc_bank_bytes_for(wire, rows) || !bank_stride.is_multiple_of(256) {
            return Err(fail(&format!(
                "bank stride {bank_stride} cannot hold {rows} rows"
            )));
        }
        if (streams as usize) % kind.alignment() != 0
            || (partial as usize) % 16 != 0
            || (gates as usize) % 16 != 0
        {
            return Err(fail(&format!(
                "operands are not aligned ({} streams {}, partial/gates 16 bytes)",
                kind.name(),
                kind.alignment()
            )));
        }
        let split = (wire == ArHcWire::Tagged && self.flags.ep_ar_fast_apply_enabled())
            .then(|| ar_hc_split_geometry(rows))
            .flatten();
        let (symbol, geometry) = match split {
            Some(g) => (kind.split_symbol(), g),
            None => (kind.symbol(wire), geometry),
        };
        let gen_ptr = generation.as_ptr();
        let stride_arg = u32::try_from(bank_stride).map_err(|_| fail("bank stride exceeds u32"))?;
        let first_arg = i32::from(local_is_first);
        let count_arg = count as i32;
        let wait_block = self.device_wait_block()?;
        let mut params: [*mut c_void; 11] = [
            &partial as *const _ as *mut c_void,
            &streams as *const _ as *mut c_void,
            &gates as *const _ as *mut c_void,
            &peer_mailbox as *const _ as *mut c_void,
            &local_mailbox as *const _ as *mut c_void,
            &stride_arg as *const _ as *mut c_void,
            &gen_ptr as *const _ as *mut c_void,
            &wait_block as *const _ as *mut c_void,
            &first_arg as *const _ as *mut c_void,
            &count_arg as *const _ as *mut c_void,
            &mixed_dbg as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            symbol,
            [geometry.0, 1, 1],
            [geometry.1, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(partial);
                b.push_ptr(streams);
                b.push_ptr(gates);
                b.push_ptr(peer_mailbox);
                b.push_ptr(local_mailbox);
                b.push_u32(stride_arg);
                b.push_ptr(gen_ptr);
                b.push_ptr(wait_block);
                b.push_i32(first_arg);
                b.push_i32(count_arg);
                b.push_ptr(mixed_dbg);
                b
            },
        )
    }
}

/// Granules per lane of the busiest block for `rows` at `geometry` (the kernel's slicing).
pub fn ar_hc_granules_per_lane(rows: usize, geometry: (u32, u32)) -> usize {
    let granules = rows * AR_HC_HIDDEN / AR_HC_GRANULE_VALUES;
    let per = granules.div_ceil(geometry.0 as usize);
    per.div_ceil(geometry.1 as usize)
}

// ----------------------------------------------------------------------------------------
// Sequence-parallel (SP) reduce-scatter + all-gather (`kernels/src/sp_hc_rs_ag.gfx1201.hip`)
// ----------------------------------------------------------------------------------------
//
// Dense wire only. A launch names an OWN row range (rows this rank folds / receives) and a
// PEER row range (rows this rank pushes) as `(row0, rows)` of the full tile; mailbox slots are
// granule indices relative to the first row of the pushed range, so the ranges the two ranks
// pass are mirror images: rank A's `peer` is rank B's `own` and vice versa.

const SP_MODULE: &str = "sp_hc_rs_ag_gfx1201";
const SP_RS_D8: &str = "sp_hc_rs_d8_gfx1201";
const SP_RS_D8_F32S: &str = "sp_hc_rs_d8_f32s_gfx1201";
const SP_AG_BF16: &str = "sp_ag_rows_bf16_d8_gfx1201";
const SP_AG_F32BF16: &str = "sp_ag_rows_f32bf16_d8_gfx1201";

/// Most rows of EACH range (own / peer) one SP launch serves; the caller slices longer ranges
/// with [`sp_slices`].
pub const SP_ROW_CAP: usize = AR_HC_ROW_CAP;
/// Bytes of one dense granule (four bf16 values).
pub const SP_GRANULE_BYTES: usize = 8;
/// Granules per row of a hidden-width tile.
pub const SP_GRANULES_PER_ROW: usize = AR_HC_HIDDEN / AR_HC_GRANULE_VALUES;
/// Most granules a launch can poll (`blocks x lanes x AR_HC_KMAX`).
pub const SP_MAX_POLL_GRANULES: usize = AR_HC_MAX_BLOCKS * AR_HC_MAX_LANES * AR_HC_KMAX;

/// Element format of the all-gathered buffer.
///
/// * `Bf16`: u16 bits (`[rows][row_elems]`, two per wire dword), base 8-byte aligned.
/// * `F32Bf16`: f32 words holding BF16-exact values (`hc_mixed`: `hyper_read` stores
///   `bf16r(value / 4)`); the wire carries the upper 16 bits of each word and the receiver
///   stores `bits << 16`. Base 16-byte aligned.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpAgElem {
    Bf16,
    F32Bf16,
}

impl SpAgElem {
    pub const ALL: [Self; 2] = [Self::Bf16, Self::F32Bf16];

    /// Device entry of this element format.
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Bf16 => SP_AG_BF16,
            Self::F32Bf16 => SP_AG_F32BF16,
        }
    }

    /// Bytes per buffer element.
    pub fn element_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32Bf16 => 4,
        }
    }

    /// Required byte alignment of the buffer base.
    pub fn alignment(self) -> usize {
        match self {
            Self::Bf16 => 8,
            Self::F32Bf16 => 16,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F32Bf16 => "f32bf16",
        }
    }
}

/// Device entry of the reduce-scatter for `kind`.
pub fn sp_rs_symbol(kind: StreamKind) -> &'static str {
    match kind {
        StreamKind::Bf16 => SP_RS_D8,
        StreamKind::F32 => SP_RS_D8_F32S,
    }
}

/// Block / lane shape for `granules` polled granules: the [`ar_hc_geometry_exact`] search
/// over a granule count (more blocks first, then the 512-lane fallback of
/// [`ar_hc_geometry_with`]), at most [`AR_HC_KMAX`] granules per lane.
fn sp_geometry_granules(granules: usize) -> Option<(u32, u32)> {
    fn exact(granules: usize, blocks0: usize, lanes0: usize) -> Option<(u32, u32)> {
        let mut b = blocks0
            .min(granules.div_ceil(32))
            .clamp(1, AR_HC_MAX_BLOCKS);
        loop {
            let per = granules.div_ceil(b);
            let t = lanes0.min(per.div_ceil(32) * 32).clamp(32, AR_HC_MAX_LANES);
            if per.div_ceil(t) <= AR_HC_KMAX {
                return Some((b as u32, t as u32));
            }
            if b >= AR_HC_MAX_BLOCKS {
                return None;
            }
            b += 1;
        }
    }
    if granules == 0 {
        return None;
    }
    exact(
        granules,
        AR_HC_DEFAULT_BLOCKS as usize,
        AR_HC_DEFAULT_LANES as usize,
    )
    .or_else(|| exact(granules, AR_HC_DEFAULT_BLOCKS as usize, AR_HC_MAX_LANES))
}

/// Launch geometry `(blocks, lanes)` of a reduce-scatter launch with `own_rows` rows folded
/// and `peer_rows` rows pushed (each at most [`SP_ROW_CAP`], not both zero). It is the
/// geometry of the larger range, so the poll side (own rows) stays within [`AR_HC_KMAX`]
/// granules per lane and the push side is spread over the same blocks.
pub fn sp_rs_geometry(own_rows: usize, peer_rows: usize) -> Option<(u32, u32)> {
    if own_rows > SP_ROW_CAP || peer_rows > SP_ROW_CAP {
        return None;
    }
    sp_geometry_granules(own_rows.max(peer_rows) * SP_GRANULES_PER_ROW)
}

/// Launch geometry `(blocks, lanes)` of an all-gather launch pushing `own_rows` and polling
/// `peer_rows` rows of `row_elems` elements (each at most [`SP_ROW_CAP`] rows, not both zero,
/// `row_elems % 4 == 0`).
pub fn sp_ag_geometry(own_rows: usize, peer_rows: usize, row_elems: usize) -> Option<(u32, u32)> {
    if row_elems == 0
        || !row_elems.is_multiple_of(AR_HC_GRANULE_VALUES)
        || own_rows > SP_ROW_CAP
        || peer_rows > SP_ROW_CAP
    {
        return None;
    }
    sp_geometry_granules(own_rows.max(peer_rows) * (row_elems / AR_HC_GRANULE_VALUES))
}

/// Granules per lane of the busiest block of `granules` polled granules at `geometry`.
pub fn sp_granules_per_lane(granules: usize, geometry: (u32, u32)) -> usize {
    granules
        .div_ceil(geometry.0 as usize)
        .div_ceil(geometry.1 as usize)
}

/// Most rows of one all-gather launch for `row_elems`-wide rows: the row cap, the granule
/// capacity of one mailbox bank and the poll capacity of one launch.
pub fn sp_ag_rows_per_launch(row_elems: usize, bank_stride: usize) -> usize {
    let gpr = row_elems / AR_HC_GRANULE_VALUES;
    if gpr == 0 {
        return 0;
    }
    SP_ROW_CAP
        .min(bank_stride / SP_GRANULE_BYTES / gpr)
        .min(SP_MAX_POLL_GRANULES / gpr)
}

/// Slices of a tile exchange: the launches covering an own range of `own_rows` rows and a peer
/// range of `peer_rows` rows in chunks of at most `cap` rows per range. Each item is
/// `((own_off, own_n), (peer_off, peer_n))`, offsets relative to the start of the range. The
/// ranks of a pair compute mirrored lists (rank A's peer chunks are rank B's own chunks), so
/// launch `i` pushes exactly the granules the other rank's launch `i` polls.
pub fn sp_slices(
    own_rows: usize,
    peer_rows: usize,
    cap: usize,
) -> Vec<((usize, usize), (usize, usize))> {
    assert!(cap > 0, "slice cap must be positive");
    let n = own_rows.div_ceil(cap).max(peer_rows.div_ceil(cap));
    (0..n)
        .map(|i| {
            let chunk = |total: usize| {
                let off = (i * cap).min(total);
                (off, cap.min(total - off))
            };
            (chunk(own_rows), chunk(peer_rows))
        })
        .collect()
}

/// Even two-way split of `rows` rows: `[(row0, n); 2]`, rank 0 first (it takes the extra row
/// of an odd count).
pub fn sp_even_split(rows: usize) -> [(usize, usize); 2] {
    let first = rows.div_ceil(2);
    [(0, first), (first, rows - first)]
}

/// The wire dword of a payload dword: a NaN-pair dword `0xFFFF_FFFF` (the EMPTY marker)
/// travels as `0xFFFF_FFFE`; everything else is unchanged.
pub fn sp_wire_dword(w: u32) -> u32 {
    if w == u32::MAX {
        u32::MAX - 1
    } else {
        w
    }
}

/// What an f32 word of a [`SpAgElem::F32Bf16`] buffer arrives as: its upper 16 bits widened
/// back (identity for BF16-exact values).
pub fn sp_f32bf16_wire(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0xFFFF_0000)
}

impl Gpu {
    /// Resolve the reduce-scatter kernel of `kind` before graph capture.
    pub fn ensure_sp_hc_rs_gfx1201_kernel(&mut self, kind: StreamKind) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch != "gfx1201" {
            return Err(HipError::new(
                0,
                &format!("sp_hc_rs_gfx1201 requires gfx1201, got {}", self.arch),
            ));
        }
        let src = self.sp_hc_rs_ag_src();
        self.ensure_kernel(SP_MODULE, src, sp_rs_symbol(kind))
    }

    /// Resolve the all-gather kernel of `elem` before graph capture.
    pub fn ensure_sp_ag_rows_gfx1201_kernel(&mut self, elem: SpAgElem) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch != "gfx1201" {
            return Err(HipError::new(
                0,
                &format!("sp_ag_rows_gfx1201 requires gfx1201, got {}", self.arch),
            ));
        }
        let src = self.sp_hc_rs_ag_src();
        self.ensure_kernel(SP_MODULE, src, elem.symbol())
    }

    /// SP module source: bf16 rounding by selects, or the `hip_bfloat16` casts under
    /// `HIPFIRE_EP_AR_FAST_APPLY=0` (bit-identical either way).
    fn sp_hc_rs_ag_src(&self) -> &'static str {
        if self.flags.ep_ar_fast_apply_enabled() {
            kernels::SP_HC_RS_AG_GFX1201_SRC
        } else {
            kernels::SP_HC_RS_AG_GFX1201_HIPBF16_SRC
        }
    }

    /// Reduce-scatter fused into the HC write: push `bf16(partial)` of the PEER row range
    /// into the peer's dense mailbox, wait (bounded) for the peer's partial of the OWN range,
    /// fold in rank order and apply the production HC write to the own rows of `streams`
    /// only. `partial` (`[rows, 2560]` f32), `streams` (`[rows, 4, 2560]` of `kind`) and
    /// `gates` (`[rows, 4]` f32) are the full-tile base pointers; `own` / `peer` are
    /// `(row0, rows)` of the tile (disjoint, each at most [`SP_ROW_CAP`], not both empty; an
    /// empty own range still pushes). `peer_mailbox` / `local_mailbox` address the DENSE
    /// region of the peer's / this rank's mailbox, `bank_stride` its bank size. `mixed_dbg`
    /// (oracle only) receives the folded `m` of the own rows at the full-tile offset. The
    /// own rows are bit-identical to those of `ar_hc_write_d8_*` over the same partials.
    /// One launch; both ranks must issue mirrored launches in the same order.
    ///
    /// # Safety
    /// Every pointer must address live device memory of the full-tile extents on this rank's
    /// device (the mailboxes through peer access).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sp_hc_rs_write_gfx1201(
        &mut self,
        partial: *mut c_void,
        streams: *mut c_void,
        kind: StreamKind,
        gates: *const c_void,
        peer_mailbox: *mut c_void,
        local_mailbox: *const c_void,
        bank_stride: usize,
        generation: &DeviceBuffer,
        local_is_first: bool,
        own: (usize, usize),
        peer: (usize, usize),
        geometry: (u32, u32),
        mixed_dbg: *mut c_void,
    ) -> HipResult<()> {
        self.ensure_sp_hc_rs_gfx1201_kernel(kind)?;
        let fail = |what: &str| HipError::new(0, &format!("sp_hc_rs_gfx1201: {what}"));
        if own.1 + peer.1 == 0 {
            return Err(fail("empty own and peer ranges"));
        }
        if sp_rs_geometry(own.1, peer.1).is_none() {
            return Err(fail(&format!(
                "unsupported ranges own {own:?} peer {peer:?} (cap {SP_ROW_CAP} rows each)"
            )));
        }
        if own.1 > 0 && peer.1 > 0 && own.0 < peer.0 + peer.1 && peer.0 < own.0 + own.1 {
            return Err(fail(&format!("own {own:?} and peer {peer:?} overlap")));
        }
        if geometry.0 == 0
            || geometry.1 == 0
            || geometry.0 as usize > AR_HC_MAX_BLOCKS
            || geometry.1 as usize > AR_HC_MAX_LANES
            || sp_granules_per_lane(own.1 * SP_GRANULES_PER_ROW, geometry) > AR_HC_KMAX
        {
            return Err(fail(&format!(
                "invalid geometry {geometry:?} for own {own:?} peer {peer:?}"
            )));
        }
        let granules = own.1.max(peer.1) * SP_GRANULES_PER_ROW;
        if granules * SP_GRANULE_BYTES > bank_stride || !bank_stride.is_multiple_of(256) {
            return Err(fail(&format!(
                "bank stride {bank_stride} cannot hold {granules} dense granules"
            )));
        }
        if (streams as usize) % kind.alignment() != 0
            || (partial as usize) % 16 != 0
            || (gates as usize) % 16 != 0
        {
            return Err(fail(&format!(
                "operands are not aligned ({} streams {}, partial/gates 16 bytes)",
                kind.name(),
                kind.alignment()
            )));
        }
        let int = |v: usize| i32::try_from(v).map_err(|_| fail("row index exceeds i32"));
        let (own_row0, own_rows, peer_row0, peer_rows) =
            (int(own.0)?, int(own.1)?, int(peer.0)?, int(peer.1)?);
        let gen_ptr = generation.as_ptr();
        let stride_arg = u32::try_from(bank_stride).map_err(|_| fail("bank stride exceeds u32"))?;
        let first_arg = i32::from(local_is_first);
        let wait_block = self.device_wait_block()?;
        let mut params: [*mut c_void; 14] = [
            &partial as *const _ as *mut c_void,
            &streams as *const _ as *mut c_void,
            &gates as *const _ as *mut c_void,
            &peer_mailbox as *const _ as *mut c_void,
            &local_mailbox as *const _ as *mut c_void,
            &stride_arg as *const _ as *mut c_void,
            &gen_ptr as *const _ as *mut c_void,
            &wait_block as *const _ as *mut c_void,
            &first_arg as *const _ as *mut c_void,
            &own_row0 as *const _ as *mut c_void,
            &own_rows as *const _ as *mut c_void,
            &peer_row0 as *const _ as *mut c_void,
            &peer_rows as *const _ as *mut c_void,
            &mixed_dbg as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            sp_rs_symbol(kind),
            [geometry.0, 1, 1],
            [geometry.1, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(partial);
                b.push_ptr(streams);
                b.push_ptr(gates);
                b.push_ptr(peer_mailbox);
                b.push_ptr(local_mailbox);
                b.push_u32(stride_arg);
                b.push_ptr(gen_ptr);
                b.push_ptr(wait_block);
                b.push_i32(first_arg);
                b.push_i32(own_row0);
                b.push_i32(own_rows);
                b.push_i32(peer_row0);
                b.push_i32(peer_rows);
                b.push_ptr(mixed_dbg);
                b
            },
        )
    }

    /// All-gather of rows of `buf` (`[rows][row_elems]` of `elem`, `row_elems % 4 == 0`):
    /// push the OWN rows (`(row0, rows)`) into the peer's dense mailbox, wait (bounded) for
    /// the PEER range's rows and write them into `buf`. Both ranks end with the same buffer.
    /// Finite values are byte-exact; a NaN-pair wire dword `0xFFFF_FFFF` arrives as
    /// `0xFFFF_FFFE`. Ranges are disjoint, each at most [`SP_ROW_CAP`] rows and within the
    /// bank ([`sp_ag_rows_per_launch`]). Pointers and generation as for
    /// [`Self::sp_hc_rs_write_gfx1201`]. One launch.
    ///
    /// # Safety
    /// Every pointer must address live device memory of the full buffer on this rank's
    /// device (the mailbox through peer access).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sp_ag_rows_gfx1201(
        &mut self,
        buf: *mut c_void,
        elem: SpAgElem,
        row_elems: usize,
        peer_mailbox: *mut c_void,
        local_mailbox: *const c_void,
        bank_stride: usize,
        generation: &DeviceBuffer,
        own: (usize, usize),
        peer: (usize, usize),
        geometry: (u32, u32),
    ) -> HipResult<()> {
        self.ensure_sp_ag_rows_gfx1201_kernel(elem)?;
        let fail = |what: &str| HipError::new(0, &format!("sp_ag_rows_gfx1201: {what}"));
        if own.1 + peer.1 == 0 {
            return Err(fail("empty own and peer ranges"));
        }
        if sp_ag_geometry(own.1, peer.1, row_elems).is_none() {
            return Err(fail(&format!(
                "unsupported ranges own {own:?} peer {peer:?} row_elems {row_elems}"
            )));
        }
        if own.1 > 0 && peer.1 > 0 && own.0 < peer.0 + peer.1 && peer.0 < own.0 + own.1 {
            return Err(fail(&format!("own {own:?} and peer {peer:?} overlap")));
        }
        let gpr = row_elems / AR_HC_GRANULE_VALUES;
        if geometry.0 == 0
            || geometry.1 == 0
            || geometry.0 as usize > AR_HC_MAX_BLOCKS
            || geometry.1 as usize > AR_HC_MAX_LANES
            || sp_granules_per_lane(peer.1 * gpr, geometry) > AR_HC_KMAX
        {
            return Err(fail(&format!(
                "invalid geometry {geometry:?} for own {own:?} peer {peer:?} row_elems {row_elems}"
            )));
        }
        let granules = own.1.max(peer.1) * gpr;
        if granules * SP_GRANULE_BYTES > bank_stride || !bank_stride.is_multiple_of(256) {
            return Err(fail(&format!(
                "bank stride {bank_stride} cannot hold {granules} dense granules"
            )));
        }
        if (buf as usize) % elem.alignment() != 0 {
            return Err(fail(&format!(
                "buffer is not {}-byte aligned ({})",
                elem.alignment(),
                elem.name()
            )));
        }
        let int = |v: usize| i32::try_from(v).map_err(|_| fail("index exceeds i32"));
        let (row_elems_arg, own_row0, own_rows, peer_row0, peer_rows) = (
            int(row_elems)?,
            int(own.0)?,
            int(own.1)?,
            int(peer.0)?,
            int(peer.1)?,
        );
        let gen_ptr = generation.as_ptr();
        let stride_arg = u32::try_from(bank_stride).map_err(|_| fail("bank stride exceeds u32"))?;
        let wait_block = self.device_wait_block()?;
        let mut params: [*mut c_void; 11] = [
            &buf as *const _ as *mut c_void,
            &row_elems_arg as *const _ as *mut c_void,
            &peer_mailbox as *const _ as *mut c_void,
            &local_mailbox as *const _ as *mut c_void,
            &stride_arg as *const _ as *mut c_void,
            &gen_ptr as *const _ as *mut c_void,
            &wait_block as *const _ as *mut c_void,
            &own_row0 as *const _ as *mut c_void,
            &own_rows as *const _ as *mut c_void,
            &peer_row0 as *const _ as *mut c_void,
            &peer_rows as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            elem.symbol(),
            [geometry.0, 1, 1],
            [geometry.1, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(buf);
                b.push_i32(row_elems_arg);
                b.push_ptr(peer_mailbox);
                b.push_ptr(local_mailbox);
                b.push_u32(stride_arg);
                b.push_ptr(gen_ptr);
                b.push_ptr(wait_block);
                b.push_i32(own_row0);
                b.push_i32(own_rows);
                b.push_i32(peer_row0);
                b.push_i32(peer_rows);
                b
            },
        )
    }
}

// ----------------------------------------------------------------------------------------
// Host numerics model
// ----------------------------------------------------------------------------------------

/// Round-to-nearest-even f32 -> bf16 bits (NaN passes through the rounding add like the
/// device `bf16bits`; partial sums are finite).
pub fn bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    ((u.wrapping_add(0x7FFF + ((u >> 16) & 1))) >> 16) as u16
}

/// `bf16(x)` widened back to f32.
pub fn bf16_round(x: f32) -> f32 {
    f32::from_bits(u32::from(bf16_bits(x)) << 16)
}

/// The folded block output of one element: `0.0 + (first + second)` over the two
/// bf16-rounded partials, `first` being rank 0's.
pub fn ar_hc_fold(first: f32, second: f32) -> f32 {
    0.0f32 + (bf16_round(first) + bf16_round(second))
}

/// Per-element bound of [`ar_hc_fold`] against the exact sum `p0 + p1` (one bf16 rounding
/// per partial, unit roundoff `2^-8`, plus the fp32 add): `(2^-8 + 2^-23)(|p0| + |p1|)`.
pub fn ar_hc_fold_bound(p0: f32, p1: f32) -> f64 {
    (2f64.powi(-8) + 2f64.powi(-23)) * (f64::from(p0).abs() + f64::from(p1).abs())
}

/// The row gate of one HC branch (`hyper_write_bf16x2`): three bf16 roundings around a
/// sigmoid. `expf` here is the host libm (the device uses its own; oracles that compare
/// bytes use the device `hyper_write` for the HC half).
pub fn hc_gate(logit: f32) -> f32 {
    let gate_logit = bf16_round(logit);
    let gate_scaled = bf16_round(gate_logit / 4.0);
    bf16_round(2.0 * bf16_round(1.0 / (1.0 + (-gate_scaled).exp())))
}

/// One stream word update of `hyper_write_bf16x2`: `bf16(bf16(g * bf16(m)) + stream)`.
pub fn hc_inject(gate: f32, m: f32, stream_bits: u16) -> u16 {
    let inj = bf16_round(gate * bf16_round(m));
    bf16_bits(inj + f32::from_bits(u32::from(stream_bits) << 16))
}

/// One stream word update of `hyper_write_f32`: `bf16(bf16(stream) + bf16(g * bf16(m)))`,
/// stored as an f32 (the stream holds BF16-exact values; the input is rounded on read).
pub fn hc_inject_f32(gate: f32, m: f32, stream: f32) -> f32 {
    let inj = bf16_round(gate * bf16_round(m));
    bf16_round(bf16_round(stream) + inj)
}

/// [`ar_hc_write_model`] for F32 streams (`[rows][4][2560]` f32, `hyper_write_f32`'s
/// arithmetic): the folded `m` is the same; every stream goes through [`hc_inject_f32`].
pub fn ar_hc_write_model_f32(
    p0: &[f32],
    p1: &[f32],
    streams: &mut [f32],
    gates: &[f32],
    rows: usize,
) -> Vec<f32> {
    assert_eq!(p0.len(), rows * AR_HC_HIDDEN);
    assert_eq!(p1.len(), rows * AR_HC_HIDDEN);
    assert_eq!(streams.len(), rows * 4 * AR_HC_HIDDEN);
    assert_eq!(gates.len(), rows * 4);
    let mut mixed = vec![0f32; rows * AR_HC_HIDDEN];
    for row in 0..rows {
        let g = [
            hc_gate(gates[row * 4]),
            hc_gate(gates[row * 4 + 1]),
            hc_gate(gates[row * 4 + 2]),
            hc_gate(gates[row * 4 + 3]),
        ];
        for col in 0..AR_HC_HIDDEN {
            let i = row * AR_HC_HIDDEN + col;
            let m = ar_hc_fold(p0[i], p1[i]);
            mixed[i] = m;
            for (br, gate) in g.iter().enumerate() {
                let s = &mut streams[row * 4 * AR_HC_HIDDEN + br * AR_HC_HIDDEN + col];
                *s = hc_inject_f32(*gate, m, *s);
            }
        }
    }
    mixed
}

/// Host model of the whole fused launch for one rank-agnostic result: folds the two ranks'
/// partials (`[rows * 2560]` each, rank 0 first) and applies the HC write to `streams`
/// (`[rows][4][2560]` bf16 bits) with `gates` (`[rows][4]`). Returns the folded `m`.
pub fn ar_hc_write_model(
    p0: &[f32],
    p1: &[f32],
    streams: &mut [u16],
    gates: &[f32],
    rows: usize,
) -> Vec<f32> {
    assert_eq!(p0.len(), rows * AR_HC_HIDDEN);
    assert_eq!(p1.len(), rows * AR_HC_HIDDEN);
    assert_eq!(streams.len(), rows * 4 * AR_HC_HIDDEN);
    assert_eq!(gates.len(), rows * 4);
    let mut mixed = vec![0f32; rows * AR_HC_HIDDEN];
    for row in 0..rows {
        let g = [
            hc_gate(gates[row * 4]),
            hc_gate(gates[row * 4 + 1]),
            hc_gate(gates[row * 4 + 2]),
            hc_gate(gates[row * 4 + 3]),
        ];
        for col in 0..AR_HC_HIDDEN {
            let i = row * AR_HC_HIDDEN + col;
            let m = ar_hc_fold(p0[i], p1[i]);
            mixed[i] = m;
            for (br, gate) in g.iter().enumerate() {
                let s = &mut streams[row * 4 * AR_HC_HIDDEN + br * AR_HC_HIDDEN + col];
                *s = hc_inject(*gate, m, *s);
            }
        }
    }
    mixed
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// The kernel's slicing (`per = ceil(G / grid)`, lane stride `blockDim`) covers every
    /// granule exactly once and never exceeds `AR_HC_KMAX` per lane, for every row count
    /// and several requested shapes; the mailbox bank holds the largest launch.
    #[test]
    fn geometry_covers_every_granule_once() {
        for rows in [
            1usize, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 100, 128, 255, 256,
        ] {
            for (b0, t0) in [
                (16u32, 256u32),
                (8, 256),
                (4, 256),
                (16, 512),
                (2, 128),
                (64, 512),
            ] {
                let granules = rows * AR_HC_HIDDEN / AR_HC_GRANULE_VALUES;
                let Some((blocks, lanes)) = ar_hc_geometry_with(rows, b0, t0) else {
                    // legitimately uncoverable: even 64 blocks of `t0` lanes at KMAX granules
                    // each fall short of this row count
                    assert!(
                        granules > AR_HC_MAX_BLOCKS * AR_HC_MAX_LANES * AR_HC_KMAX,
                        "rows {rows} shape {b0}x{t0}: no geometry"
                    );
                    continue;
                };
                assert!(blocks as usize <= AR_HC_MAX_BLOCKS && lanes as usize <= AR_HC_MAX_LANES);
                let g = rows * AR_HC_HIDDEN / AR_HC_GRANULE_VALUES;
                let per = g.div_ceil(blocks as usize);
                let mut seen = vec![0u8; g];
                let mut worst = 0;
                for blk in 0..blocks as usize {
                    let (g0, g1) = (blk * per, g.min(blk * per + per));
                    if g0 >= g1 {
                        continue;
                    }
                    let k = (g1 - g0).div_ceil(lanes as usize);
                    worst = worst.max(k);
                    assert!((1..=AR_HC_KMAX).contains(&k), "rows {rows} K {k}");
                    for tid in 0..lanes as usize {
                        for kk in 0..k {
                            let j = g0 + tid + kk * lanes as usize;
                            if j < g1 {
                                seen[j] += 1;
                            }
                        }
                    }
                }
                assert_eq!(worst, ar_hc_granules_per_lane(rows, (blocks, lanes)));
                assert!(seen.iter().all(|&c| c == 1), "rows {rows} {blocks}x{lanes}");
                assert!(g * AR_HC_GRANULE_BYTES <= ar_hc_bank_bytes(rows));
                // a granule never crosses a row
                assert_eq!(AR_HC_HIDDEN % AR_HC_GRANULE_VALUES, 0);
            }
        }
        assert!(ar_hc_geometry(0).is_none());
        assert!(ar_hc_geometry(AR_HC_ROW_CAP + 1).is_none());
        // the whole launch cap is coverable at the default shape (widened to 512 lanes)
        assert!(ar_hc_geometry(AR_HC_ROW_CAP).is_some());
        assert_eq!(
            ar_hc_geometry(8),
            Some((16, 256)).filter(|&(b, t)| ar_hc_granules_per_lane(8, (b, t)) <= AR_HC_KMAX)
        );
    }

    /// A branch-split launch gives every (granule, branch) item exactly one lane, keeps the
    /// four items of a granule inside one block and one wave (aligned group of four lanes),
    /// and exists exactly for 1 ..= AR_HC_SPLIT_MAX_ROWS rows. Mirrors arhc_split_entry.
    #[test]
    fn split_geometry_covers_every_item_once() {
        assert!(ar_hc_split_geometry(0).is_none());
        assert!(ar_hc_split_geometry(AR_HC_SPLIT_MAX_ROWS + 1).is_none());
        for rows in 1..=AR_HC_SPLIT_MAX_ROWS {
            let (blocks, lanes) = ar_hc_split_geometry(rows).expect("split geometry");
            assert!(blocks as usize <= AR_HC_MAX_BLOCKS && lanes as usize <= AR_HC_MAX_LANES);
            assert_eq!(lanes % 32, 0);
            let items = rows * AR_HC_HIDDEN;
            let per = (items.div_ceil(blocks as usize) + 3) & !3;
            assert!(
                per <= lanes as usize,
                "rows {rows}: per {per} > lanes {lanes}"
            );
            let mut seen = vec![0u8; items];
            for blk in 0..blocks as usize {
                let (i0, i1) = (blk * per, items.min(blk * per + per));
                assert_eq!(i0 % 4, 0);
                for tid in 0..per {
                    let i = i0 + tid;
                    if i < i1 {
                        seen[i] += 1;
                        // the granule's four items: same block, lanes tid & !3 .. +3 (one wave)
                        assert_eq!(i & 3, tid & 3);
                    }
                }
            }
            assert!(seen.iter().all(|&c| c == 1), "rows {rows}");
        }
        assert_eq!(ar_hc_split_geometry(1), Some((40, 64)));
        assert_eq!(ar_hc_split_geometry(8), Some((64, 320)));
    }

    #[test]
    fn bank_bytes_are_aligned_and_sized() {
        assert_eq!(ar_hc_bank_bytes(1), (640 * 16usize).next_multiple_of(256));
        assert_eq!(ar_hc_bank_bytes(8), 8 * 640 * 16);
        assert_eq!(ar_hc_bank_bytes(256), 256 * 640 * 16);
        assert!(ar_hc_bank_bytes(37) % 256 == 0);
    }

    /// One bf16 rounding per partial and one fp32 add stay within the stated bound of the
    /// exact sum, across 8 binades of partial magnitude and skewed ratios.
    #[test]
    fn fold_bound_holds_for_random_partials() {
        let mut rng = Rng(0x0b0b_0b0b_1234_5678);
        for _ in 0..2_000_000 {
            let mag = 2f32.powi(-6 + rng.below(10) as i32);
            let a = (rng.unit() - 0.5) * 2.0 * mag;
            let b = (rng.unit() - 0.5) * 2.0 * mag * 2f32.powi(rng.below(5) as i32 - 2);
            let m = ar_hc_fold(a, b);
            let exact = f64::from(a) + f64::from(b);
            assert!(
                (f64::from(m) - exact).abs() <= ar_hc_fold_bound(a, b),
                "a {a} b {b} m {m}"
            );
        }
    }

    /// Both ranks fold the same two bf16 values in the same (rank) order, so the folded value
    /// is the same bits whichever rank computes it; swapping the operands is a different
    /// (equally bounded) sum only through fp32 non-associativity of -0.0/rounding, which the
    /// rank order removes. The identity the kernel needs: `fold(first, second)` is a pure
    /// function of the two rounded values.
    #[test]
    fn fold_is_a_pure_function_of_the_two_rounded_partials() {
        let mut rng = Rng(77);
        for _ in 0..200_000 {
            let a = (rng.unit() - 0.5) * 8.0;
            let b = (rng.unit() - 0.5) * 8.0;
            let m = ar_hc_fold(a, b);
            assert_eq!(
                m.to_bits(),
                ar_hc_fold(bf16_round(a), bf16_round(b)).to_bits()
            );
            // commutative in fp32 for two operands
            assert_eq!(m.to_bits(), ar_hc_fold(b, a).to_bits());
        }
        // negative zero: 0.0 + (-0.0 + -0.0) is +0.0, as the kernel's `xi = 0.0f; xi + (..)`
        assert_eq!(ar_hc_fold(-0.0, -0.0).to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn bf16_rounding_is_round_to_nearest_even() {
        assert_eq!(bf16_bits(1.0), 0x3F80);
        // halfway between two bf16 values rounds to even
        assert_eq!(bf16_bits(f32::from_bits(0x3F80_8000)), 0x3F80);
        assert_eq!(bf16_bits(f32::from_bits(0x3F81_8000)), 0x3F82);
        assert_eq!(bf16_bits(f32::from_bits(0x3F80_8001)), 0x3F81);
    }

    /// The whole-launch model is deterministic, applies the production HC expression to every
    /// stream, and keeps each stream within one bf16 ulp of `stream + gate * m` computed in f64.
    #[test]
    fn write_model_matches_the_hc_expression() {
        let mut rng = Rng(5);
        for rows in [1usize, 3, 8] {
            let n = rows * AR_HC_HIDDEN;
            let p0: Vec<f32> = (0..n).map(|_| (rng.unit() - 0.5) * 0.5).collect();
            let p1: Vec<f32> = (0..n).map(|_| (rng.unit() - 0.5) * 0.5).collect();
            let gates: Vec<f32> = (0..rows * 4).map(|_| (rng.unit() - 0.5) * 8.0).collect();
            let s0: Vec<u16> = (0..rows * 4 * AR_HC_HIDDEN)
                .map(|_| bf16_bits((rng.unit() - 0.5) * 4.0))
                .collect();
            let mut a = s0.clone();
            let mut b = s0.clone();
            let ma = ar_hc_write_model(&p0, &p1, &mut a, &gates, rows);
            let mb = ar_hc_write_model(&p0, &p1, &mut b, &gates, rows);
            assert_eq!(a, b);
            assert_eq!(ma, mb);
            for row in 0..rows {
                for br in 0..4 {
                    let g = f64::from(hc_gate(gates[row * 4 + br]));
                    for col in (0..AR_HC_HIDDEN).step_by(97) {
                        let m = f64::from(ma[row * AR_HC_HIDDEN + col]);
                        let s = f64::from(f32::from_bits(
                            u32::from(s0[row * 4 * AR_HC_HIDDEN + br * AR_HC_HIDDEN + col]) << 16,
                        ));
                        let out = f64::from(f32::from_bits(
                            u32::from(a[row * 4 * AR_HC_HIDDEN + br * AR_HC_HIDDEN + col]) << 16,
                        ));
                        let want = s + g * m;
                        // gate*m rounds twice (two bf16 roundings), the sum once: <= 3 half-ulps
                        let tol =
                            (2f64.powi(-8)) * (want.abs() + (g * m).abs() + s.abs()) * 1.5 + 1e-30;
                        assert!(
                            (out - want).abs() <= tol,
                            "row {row} br {br} col {col}: {out} vs {want}"
                        );
                    }
                }
            }
        }
    }

    /// The F32-stream model is the `hyper_write_f32` expression verbatim (written out here
    /// independently: gate rounds, injection `bf16(gate * bf16(mixed))`, value
    /// `bf16(input) + injection`, one bf16 round on store), element for element, and agrees
    /// with the BF16-bit model when the streams hold the same bf16 values.
    #[test]
    fn f32_stream_model_is_hyper_write_f32() {
        let mut rng = Rng(0xf32f_0000_0000_0001);
        for rows in [1usize, 2, 7] {
            let n = rows * AR_HC_HIDDEN;
            let p0: Vec<f32> = (0..n).map(|_| (rng.unit() - 0.5) * 3.0).collect();
            let p1: Vec<f32> = (0..n).map(|_| (rng.unit() - 0.5) * 3.0).collect();
            let gates: Vec<f32> = (0..rows * 4).map(|_| (rng.unit() - 0.5) * 10.0).collect();
            // F32 streams hold bf16-exact values mostly; a few unrounded ones prove the read
            // rounding (`bf16(input)`).
            let s0: Vec<f32> = (0..rows * 4 * AR_HC_HIDDEN)
                .map(|i| {
                    let v = (rng.unit() - 0.5) * 8.0;
                    if i % 17 == 0 {
                        v
                    } else {
                        bf16_round(v)
                    }
                })
                .collect();
            let mut got = s0.clone();
            let m = ar_hc_write_model_f32(&p0, &p1, &mut got, &gates, rows);
            for row in 0..rows {
                for br in 0..4 {
                    let gate_logit = bf16_round(gates[row * 4 + br]);
                    let gate_scaled = bf16_round(gate_logit / 4.0);
                    let gate = bf16_round(2.0 * bf16_round(1.0 / (1.0 + (-gate_scaled).exp())));
                    for col in 0..AR_HC_HIDDEN {
                        let mixed = m[row * AR_HC_HIDDEN + col];
                        let injection = bf16_round(gate * bf16_round(mixed));
                        let at = row * 4 * AR_HC_HIDDEN + br * AR_HC_HIDDEN + col;
                        let value = bf16_round(s0[at]) + injection;
                        assert_eq!(got[at].to_bits(), bf16_round(value).to_bits());
                    }
                }
            }
            // Same bf16 streams: the F32 and BF16-bit models agree bit for bit.
            let bits: Vec<u16> = s0.iter().map(|&v| bf16_bits(v)).collect();
            let mut bf = bits.clone();
            let mut f = bits
                .iter()
                .map(|&b| f32::from_bits(u32::from(b) << 16))
                .collect::<Vec<_>>();
            let mb = ar_hc_write_model(&p0, &p1, &mut bf, &gates, rows);
            let mf = ar_hc_write_model_f32(&p0, &p1, &mut f, &gates, rows);
            assert_eq!(mb, mf);
            for (b, f) in bf.iter().zip(&f) {
                assert_eq!(f32::from_bits(u32::from(*b) << 16).to_bits(), f.to_bits());
            }
        }
    }

    #[test]
    fn stream_kinds_have_the_documented_strides() {
        assert_eq!(StreamKind::Bf16.row_bytes(), 4 * 2560 * 2);
        assert_eq!(StreamKind::F32.row_bytes(), 4 * 2560 * 4);
        assert_eq!(StreamKind::from_state_bf16(true), StreamKind::Bf16);
        assert_eq!(StreamKind::from_state_bf16(false), StreamKind::F32);
        for wire in [ArHcWire::Tagged, ArHcWire::Dense] {
            assert_ne!(StreamKind::Bf16.symbol(wire), StreamKind::F32.symbol(wire));
        }
        assert_ne!(
            StreamKind::Bf16.symbol(ArHcWire::Tagged),
            StreamKind::Bf16.symbol(ArHcWire::Dense)
        );
        assert_eq!(StreamKind::F32.alignment(), 16);
    }

    /// The device source declares one extern entry per kind and the F32 entry is
    /// the F32 instantiation.
    #[test]
    fn kernel_source_has_both_entries() {
        for src in [
            kernels::AR_HC_WRITE_GFX1201_SRC,
            kernels::AR_HC_WRITE_GFX1201_HIPBF16_SRC,
        ] {
            for wire in [ArHcWire::Tagged, ArHcWire::Dense] {
                for kind in StreamKind::ALL {
                    assert!(
                        src.contains(&format!("void {}(", kind.symbol(wire))),
                        "{}",
                        kind.symbol(wire)
                    );
                }
            }
            for kind in StreamKind::ALL {
                assert!(src.contains(&format!("void {}(", kind.split_symbol())));
            }
        }
        assert!(kernels::AR_HC_WRITE_GFX1201_HIPBF16_SRC.starts_with("#define ARHC_RND_HIP_BF16"));
        assert!(kernels::SP_HC_RS_AG_GFX1201_HIPBF16_SRC.starts_with("#define ARHC_RND_HIP_BF16"));
        let src = kernels::AR_HC_WRITE_GFX1201_SRC;
        for inst in [
            "arhc_entry<true, false>",
            "arhc_entry<false, false>",
            "arhc_entry<true, true>",
            "arhc_entry<false, true>",
        ] {
            assert!(src.contains(inst), "{inst}");
        }
    }

    /// The kernel source compiles offline for gfx1201 (hipcc `--genco`, no GPU touched) and
    /// exports the symbol. Skipped when hipcc is absent.
    #[test]
    fn kernel_source_compiles_for_gfx1201() {
        let hipcc = ["/opt/rocm/bin/hipcc", "hipcc"].into_iter().find(|p| {
            std::process::Command::new(p)
                .arg("--version")
                .output()
                .is_ok()
        });
        let Some(hipcc) = hipcc else {
            eprintln!("hipcc not found: skipped");
            return;
        };
        let dir = std::env::temp_dir().join(format!("ar-hc-write-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("ar_hc_write.hip");
        let obj = dir.join("ar_hc_write.hsaco");
        std::fs::write(&src, kernels::AR_HC_WRITE_GFX1201_SRC).unwrap();
        let out = std::process::Command::new(hipcc)
            .args([
                "-x",
                "hip",
                "--genco",
                "--offload-arch=gfx1201",
                "-O3",
                "--no-offload-compress",
                "-fuse-cuid=none",
                "-mcode-object-version=6",
            ])
            .arg(&src)
            .arg("-o")
            .arg(&obj)
            .env("HIP_VISIBLE_DEVICES", "-1")
            .env("ROCR_VISIBLE_DEVICES", "-1")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "hipcc: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let image = std::fs::read(&obj).unwrap();
        for wire in [ArHcWire::Tagged, ArHcWire::Dense] {
            for kind in StreamKind::ALL {
                let symbol = kind.symbol(wire);
                assert!(
                    image.windows(symbol.len()).any(|w| w == symbol.as_bytes()),
                    "{symbol} missing"
                );
            }
        }
        for kind in StreamKind::ALL {
            let symbol = kind.split_symbol();
            assert!(
                image.windows(symbol.len()).any(|w| w == symbol.as_bytes()),
                "{symbol} missing"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The device source keeps the invariants this module's host model depends on.
    #[test]
    fn kernel_source_pins_the_protocol_constants() {
        let src = kernels::AR_HC_WRITE_GFX1201_SRC;
        assert!(src.contains(&format!("#define AR_HC_KMAX {AR_HC_KMAX}")));
        assert!(src.contains(&format!("#define AR_HC_HIDDEN {AR_HC_HIDDEN}")));
        assert!(src.contains(&format!("#define AR_HC_TICKET_WORD {AR_HC_TICKET_WORD}")));
        assert!(src.contains("HF_WAIT_SITE_AR_HC_WRITE_GFX1201"));
        assert!(AR_HC_TICKET_WORD * 4 + 4 <= AR_HC_GEN_BYTES);
    }

    #[test]
    fn sp_geometry_keeps_the_poll_side_within_kmax() {
        for own in 0..=SP_ROW_CAP {
            for peer in [0usize, 1, 2, 7, 16, 255, 256] {
                if own + peer == 0 {
                    assert_eq!(sp_rs_geometry(own, peer), None);
                    continue;
                }
                let g = sp_rs_geometry(own, peer).expect("every capped range pair has a geometry");
                assert!((1..=AR_HC_MAX_BLOCKS as u32).contains(&g.0), "{own} {peer}");
                assert!((32..=AR_HC_MAX_LANES as u32).contains(&g.1), "{own} {peer}");
                assert!(sp_granules_per_lane(own * SP_GRANULES_PER_ROW, g) <= AR_HC_KMAX);
                assert!(sp_granules_per_lane(peer * SP_GRANULES_PER_ROW, g) <= AR_HC_KMAX);
            }
        }
        assert_eq!(sp_rs_geometry(SP_ROW_CAP + 1, 1), None);
        assert_eq!(sp_rs_geometry(1, SP_ROW_CAP + 1), None);
        // the symmetric shape equals the all-reduce launch of the same row count
        for rows in [1usize, 2, 7, 16, 255, 256] {
            assert_eq!(sp_rs_geometry(rows, rows), ar_hc_geometry(rows), "{rows}");
        }
    }

    #[test]
    fn sp_ag_geometry_covers_the_hidden_and_narrow_rows() {
        for row_elems in [4usize, 640, 2560] {
            for own in [0usize, 1, 3, 128, 256] {
                for peer in [1usize, 2, 255, 256] {
                    let g = sp_ag_geometry(own, peer, row_elems).expect("geometry");
                    let gpr = row_elems / 4;
                    assert!(sp_granules_per_lane(own * gpr, g) <= AR_HC_KMAX);
                    assert!(sp_granules_per_lane(peer * gpr, g) <= AR_HC_KMAX);
                    assert!(g.0 as usize <= AR_HC_MAX_BLOCKS && g.1 as usize <= AR_HC_MAX_LANES);
                }
            }
        }
        assert_eq!(sp_ag_geometry(0, 0, 2560), None);
        assert_eq!(sp_ag_geometry(1, 1, 0), None);
        assert_eq!(sp_ag_geometry(1, 1, 2), None);
        assert_eq!(sp_ag_geometry(1, 1, 2562), None);
        assert_eq!(sp_ag_geometry(SP_ROW_CAP + 1, 1, 2560), None);
        // a full 256-row hidden-width launch fits one dense bank of the exchange
        let bank = ar_hc_bank_bytes_for(ArHcWire::Dense, AR_HC_ROW_CAP);
        assert_eq!(sp_ag_rows_per_launch(2560, bank), SP_ROW_CAP);
        assert_eq!(sp_ag_rows_per_launch(640, bank), SP_ROW_CAP);
        assert_eq!(sp_ag_rows_per_launch(2560, 128 * 640 * 8), 128);
        assert_eq!(sp_ag_rows_per_launch(2, bank), 0);
    }

    #[test]
    fn sp_slices_mirror_between_the_two_ranks() {
        for (a, b) in [
            (1usize, 0usize),
            (0, 1),
            (256, 256),
            (257, 256),
            (306, 307),
            (0, 613),
            (613, 1),
        ] {
            let rank_a = sp_slices(a, b, SP_ROW_CAP);
            let rank_b = sp_slices(b, a, SP_ROW_CAP);
            assert_eq!(rank_a.len(), rank_b.len());
            let (mut own_sum, mut peer_sum) = (0, 0);
            for (sa, sb) in rank_a.iter().zip(&rank_b) {
                assert_eq!(sa.0, sb.1, "A's own chunk is B's peer chunk");
                assert_eq!(sa.1, sb.0);
                assert!(sa.0 .1 <= SP_ROW_CAP && sa.1 .1 <= SP_ROW_CAP);
                assert!(sa.0 .1 + sa.1 .1 > 0);
                own_sum += sa.0 .1;
                peer_sum += sa.1 .1;
            }
            assert_eq!((own_sum, peer_sum), (a, b));
        }
        assert!(sp_slices(0, 0, SP_ROW_CAP).is_empty());
    }

    #[test]
    fn sp_even_split_and_wire_models() {
        assert_eq!(sp_even_split(1), [(0, 1), (1, 0)]);
        assert_eq!(sp_even_split(7), [(0, 4), (4, 3)]);
        assert_eq!(sp_even_split(512), [(0, 256), (256, 256)]);
        assert_eq!(sp_wire_dword(u32::MAX), u32::MAX - 1);
        assert_eq!(sp_wire_dword(0x3f80_3f80), 0x3f80_3f80);
        assert_eq!(
            sp_f32bf16_wire(bf16_round(1.2345)).to_bits(),
            bf16_round(1.2345).to_bits()
        );
    }

    #[test]
    fn sp_wait_sites_are_defined_in_the_sp_source() {
        let src = kernels::SP_HC_RS_AG_GFX1201_SRC;
        assert!(src.contains("#define HF_WAIT_SITE_SP_HC_RS_GFX1201 12u"));
        assert!(src.contains("#define HF_WAIT_SITE_SP_AG_ROWS_GFX1201 13u"));
        assert!(src.contains(&format!("#define AR_HC_KMAX {AR_HC_KMAX}")));
        for kind in StreamKind::ALL {
            assert!(src.contains(&format!("void {}(", sp_rs_symbol(kind))));
        }
        for elem in SpAgElem::ALL {
            assert!(src.contains(&format!("void {}(", elem.symbol())));
        }
        // the header and the AR kernel file the SP source builds on are not edited by it
        assert!(!kernels::DEVICE_WAIT_DEADLINE_H.contains("SP_HC_RS"));
        assert!(!kernels::AR_HC_WRITE_GFX1201_SRC.contains("SP_HC_RS"));
    }

    /// The SP kernel source compiles offline for gfx1201 (hipcc `--genco`, no GPU touched)
    /// and exports every entry. Skipped when hipcc is absent.
    #[test]
    fn sp_kernel_source_compiles_for_gfx1201() {
        let hipcc = ["/opt/rocm/bin/hipcc", "hipcc"].into_iter().find(|p| {
            std::process::Command::new(p)
                .arg("--version")
                .output()
                .is_ok()
        });
        let Some(hipcc) = hipcc else {
            eprintln!("hipcc not found: skipped");
            return;
        };
        let dir = std::env::temp_dir().join(format!("sp-hc-rs-ag-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("sp_hc_rs_ag.hip");
        let obj = dir.join("sp_hc_rs_ag.hsaco");
        std::fs::write(&src, kernels::SP_HC_RS_AG_GFX1201_SRC).unwrap();
        let out = std::process::Command::new(hipcc)
            .args([
                "-x",
                "hip",
                "--genco",
                "--offload-arch=gfx1201",
                "-O3",
                "--no-offload-compress",
                "-fuse-cuid=none",
                "-mcode-object-version=6",
            ])
            .arg(&src)
            .arg("-o")
            .arg(&obj)
            .env("HIP_VISIBLE_DEVICES", "-1")
            .env("ROCR_VISIBLE_DEVICES", "-1")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "hipcc: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let image = std::fs::read(&obj).unwrap();
        let symbols = StreamKind::ALL
            .map(sp_rs_symbol)
            .into_iter()
            .chain(SpAgElem::ALL.map(SpAgElem::symbol));
        for symbol in symbols {
            assert!(
                image.windows(symbol.len()).any(|w| w == symbol.as_bytes()),
                "{symbol} missing"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
