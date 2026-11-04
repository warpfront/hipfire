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
        self.ensure_kernel(KERNEL, kernels::AR_HC_WRITE_GFX1201_SRC, kind.symbol(wire))
    }

    /// Push this rank's f32 partial `[rows, 2560]` (bf16-rounded) into the peer's mailbox,
    /// wait (bounded) for the peer's, fold in rank order and apply the production BF16
    /// HC write to `streams` (`[rows, 4, 2560]` of `kind`, in place) with the row gate
    /// logits `gates` (`[rows, 4]` f32). Operands are device pointers (the caller slices
    /// long row ranges into <= [`AR_HC_ROW_CAP`] launches by offsetting them).
    /// `peer_mailbox` is the PEER's mailbox (peer access enabled), `local_mailbox` this
    /// rank's own; each holds two banks `bank_stride` bytes apart. `generation` is this
    /// rank's [`AR_HC_GEN_BYTES`] block. `mixed_dbg` (oracle only, null otherwise) receives
    /// the folded `m`. One launch.
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
            kind.symbol(wire),
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
        let src = kernels::AR_HC_WRITE_GFX1201_SRC;
        for wire in [ArHcWire::Tagged, ArHcWire::Dense] {
            for kind in StreamKind::ALL {
                assert!(
                    src.contains(&format!("void {}(", kind.symbol(wire))),
                    "{}",
                    kind.symbol(wire)
                );
            }
        }
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
}
