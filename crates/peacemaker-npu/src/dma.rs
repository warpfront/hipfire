// SPDX-License-Identifier: Apache-2.0 AND MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// DMA register fields transcribed from AMD/Xilinx aie-rt (MIT); see NOTICE.
//! AIE2P DMA descriptors, channel queues and lock initialization.
//!
//! Register fields transcribed from AMD/Xilinx aie-rt `xaie2pgbl_params.h`
//! and `xaie_dma_aieml.c` (MIT). No vendor runtime is required.
//! Addresses are bytes; lengths and strides are 32-bit words. Dimensions are
//! inner-first, wraps are counts (zero disables), and steps encode as `step - 1`.
//! Queue repeat counts and iteration wraps are one-based.
//!
//! Construct a [`Bd`] with `Bd::new`, set its dimensions/locks/next descriptor,
//! then `emit_cdo` for boot configuration or `emit_txn` for firmware commands.
//! [`Task`] enqueues a chain; enable compute/memtile channels separately after
//! queueing via `enable_cdo` / `enable_txn`. Shim queues start the task directly.
//! Offline oracle gate: `cargo test -p peacemaker-npu reproduces_vendor_whole_array`.

/// Lock action on a BD: acquire `acq_id` with signed `acq_value` (AIE-ML semantics: negative =
/// wait until value >= -v, then add v), release `rel_id` by signed `rel_value`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BdLocks {
    pub acq: Option<(u32, i32)>,
    pub rel: Option<(u32, i32)>,
}

fn s7(v: i32) -> u32 {
    assert!((-64..=63).contains(&v));
    (v as u32) & 0x7f
}

/// Compute-tile memory-module BD (MEMORY_MODULE_DMA_BDi_0..5), contiguous 1-D transfer.
/// `addr` is the byte address in the tile's 64 KiB data memory (DMA view), `len_words` 32-bit words.
pub fn tile_bd(addr: u32, len_words: u32, locks: BdLocks, next: Option<u32>) -> [u32; 6] {
    Bd {
        locks,
        next,
        ..Bd::new(addr as u64, len_words)
    }
    .tile_words()
}

/// Shim NOC BD (NOC_MODULE_DMA_BDi_0..7), contiguous 1-D transfer of `len_words` 32-bit words.
/// The host address words (1, 2) are left at `addr_off` and patched by the firmware (DDR_PATCH).
pub fn shim_bd(len_words: u32, addr_off: u64) -> [u32; 8] {
    Bd::new(addr_off, len_words).shim_words()
}

use crate::{cdo::Cdo, regs, txn::Txn};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileType {
    Compute,
    Memtile,
    Shim,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Location {
    pub col: u32,
    pub row: u32,
}

impl Location {
    pub const fn new(col: u32, row: u32) -> Self {
        Self { col, row }
    }
    pub const fn kind(self) -> TileType {
        match self.row {
            0 => TileType::Shim,
            1 => TileType::Memtile,
            _ => TileType::Compute,
        }
    }
    pub fn address(self, offset: u32) -> u32 {
        assert!(self.col < 8 && self.row < 6);
        regs::tile(self.col, self.row, offset)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dim {
    pub step: u32,
    pub wrap: u32,
}
impl Default for Dim {
    fn default() -> Self {
        Self { step: 1, wrap: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iteration {
    pub step: u32,
    pub wrap: u32,
    pub current: u32,
}
impl Default for Iteration {
    fn default() -> Self {
        Self {
            step: 1,
            wrap: 1,
            current: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Packet {
    pub id: u32,
    pub kind: u32,
}

/// Shim NOC BD AXI-MM attributes, written verbatim into the BD words. Ignored by tile and memtile BDs.
///
/// Fields (AIE2P `NOC_MODULE_DMA_BDi_*`, `ref/aie-rt/driver/src/global/xaie2pgbl_params.h`):
/// - `burst`: `BDi_4` BURST_LENGTH, bits 30..31 (lines 16363-16366). This is the *register encoding*, not a byte
///   count. aie-rt only accepts 4/8/16/32-beat bursts (`dma/xaie_dma_aie2p.c:40-51`) and maps them to 0/1/2/3
///   (`dma/xaie_dma.c:765-782`); the AIE2P (BaseNPU2) table in mlir-aie `lib/Dialect/AIE/IR/AIETargetModel.cpp`
///   lines 1744-1758 (upstream main; exact lines drift) gives encodings 0/1/2/3 = 64/128/256/512 B (not the AIE2PS table).
/// - `cache`: `BDi_5` AXCACHE, bits 24..27 (lines 16383-16386). The raw 4-bit AXI AxCACHE value.
/// - `qos`: `BDi_5` AXQOS, bits 20..23 (lines 16387-16390). The raw 4-bit AXI AxQoS value.
///
/// Other bits are not exposed here: `BDi_5` SMID (bits 28..31, lines 16379-16382) and `BDi_3` SECURE_ACCESS
/// (bit 30, lines 16347-16350) are always emitted as 0, and the functional simulator rejects non-zero values
/// explicitly.
///
/// Defaults: [`ShimAxi::default`] is the *software* default every design uses (`burst 3`, `cache 2`, `qos 0`). It
/// is not the hardware reset value, which is 0 for all three fields (`*_DEFVAL 0x0`: 64 B bursts, AxCACHE 0).
///
/// Semantics: [`ShimAxi::validate`] only checks that each value fits its register field; it does not judge whether
/// a bit pattern is meaningful. AxCACHE values with allocate bits (e.g. 0xb, 0xf) are a hint to the memory
/// system and do not establish coherence or correctness for host buffers; making a host buffer coherent with
/// the NPU is solely the caller's responsibility, and nothing here guarantees coherent DMA for any host BO mode.
/// `cache` 0, 2 and 3 are conservative non-allocating candidates for experiments, not a hardware cache-safety
/// guarantee: hardware exactness must still be checked. Normal emitters require vendor word-5 attributes
/// (AxCACHE 2, AxQoS 0). Other values require an explicit hazardous-probe emitter: AxQoS 15 wedged the Halo
/// fabric and subsequent SMU resume failed with -22. Field legality and functional simulation do not prove safety.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShimAxi {
    pub burst: u32,
    pub cache: u32,
    pub qos: u32,
}
impl ShimAxi {
    /// Largest legal `burst` encoding (2-bit field).
    pub const MAX_BURST: u32 = 3;
    /// Largest legal `cache` value (4-bit field).
    pub const MAX_CACHE: u32 = 15;
    /// Largest legal `qos` value (4-bit field).
    pub const MAX_QOS: u32 = 15;

    /// Rejects values that do not fit their register field (they would otherwise be truncated or spill into
    /// neighbouring BD bits).
    pub fn validate(&self) -> Result<(), String> {
        if self.burst > Self::MAX_BURST {
            return Err(format!(
                "shim AXI burst encoding {} not in 0..={} (0/1/2/3 = 64/128/256/512 B)",
                self.burst,
                Self::MAX_BURST
            ));
        }
        if self.cache > Self::MAX_CACHE {
            return Err(format!(
                "shim AXI AxCACHE {} not in 0..={} (4-bit field)",
                self.cache,
                Self::MAX_CACHE
            ));
        }
        if self.qos > Self::MAX_QOS {
            return Err(format!(
                "shim AXI AxQoS {} not in 0..={} (4-bit field)",
                self.qos,
                Self::MAX_QOS
            ));
        }
        Ok(())
    }

    /// Reject non-vendor word-5 attributes before emitting a normal design.
    pub fn validate_vendor_word5(&self) -> Result<(), String> {
        self.validate()?;
        if self.cache != 2 || self.qos != 0 {
            return Err(format!("shim AXI AxCACHE {} / AxQoS {} requires an explicit hazardous probe; vendor word-5 requires AxCACHE 2 / AxQoS 0 (AxQoS 15 wedged Halo; SMU resume -22)", self.cache, self.qos));
        }
        Ok(())
    }
}
impl Default for ShimAxi {
    /// Software default (not hardware reset 0): 512 B bursts, AxCACHE 2 (mlir-aie `AIETargetModel.h:654-659`
    /// default; a NoC upsizing rationale only, not a cache-safety or coherence claim), QoS 0: the words every
    /// design used so far.
    fn default() -> Self {
        Self {
            burst: 3,
            cache: 2,
            qos: 0,
        }
    }
}

/// Structural BD. Unused dimensions must remain at `Dim::default()`.
/// The outermost dimension has no wrap field: transfer length terminates it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bd {
    pub addr: u64,
    pub len_words: u32,
    pub dims: [Dim; 4],
    pub iteration: Iteration,
    pub locks: BdLocks,
    pub next: Option<u32>,
    pub packet: Option<Packet>,
    pub suppress_tlast: bool,
    pub axi: ShimAxi,
}

fn field(value: u32, bits: u32, shift: u32) -> u32 {
    assert!(
        bits == 32 || value < (1 << bits),
        "field overflow: {value} / {bits}"
    );
    value << shift
}
fn step(value: u32, bits: u32) -> u32 {
    assert!(value > 0, "zero DMA step");
    field(value - 1, bits, 0)
}
fn iteration(i: Iteration, bits: u32) -> u32 {
    assert!(i.wrap > 0 && i.current < i.wrap);
    step(i.step, bits) | field(i.wrap - 1, 6, bits) | field(i.current, 6, bits + 6)
}
fn locks(locks: BdLocks, mem: bool) -> u32 {
    let (id_bits, acq_en, acq_val, rel_id, rel_val) = if mem {
        (8, 15, 8, 16, 24)
    } else {
        (4, 12, 5, 13, 18)
    };
    let mut word = 0;
    if let Some((id, value)) = locks.acq {
        word |= 1 << acq_en | s7(value) << acq_val | field(id, id_bits, 0);
    }
    if let Some((id, value)) = locks.rel {
        word |= s7(value) << rel_val | field(id, id_bits, rel_id);
    }
    word
}
fn tail(bd: &Bd) -> u32 {
    let mut word = 1 << 25 | locks(bd.locks, false) | (bd.suppress_tlast as u32) << 31;
    if let Some(next) = bd.next {
        word |= 1 << 26 | field(next, 4, 27);
    }
    word
}
fn packet(packet: Option<Packet>, mem: bool) -> u32 {
    match packet {
        None => 0,
        Some(p) => {
            if mem {
                1 << 31 | field(p.id, 5, 23) | field(p.kind, 3, 28)
            } else {
                1 << 30 | field(p.id, 5, 19) | field(p.kind, 3, 16)
            }
        }
    }
}

impl Bd {
    pub fn new(addr: u64, len_words: u32) -> Self {
        Self {
            addr,
            len_words,
            dims: [Dim::default(); 4],
            iteration: Iteration::default(),
            locks: BdLocks::default(),
            next: None,
            packet: None,
            suppress_tlast: false,
            axi: ShimAxi::default(),
        }
    }

    pub fn tile_words(&self) -> [u32; 6] {
        assert!(self.addr % 4 == 0 && self.addr < 0x1_0000);
        assert_eq!(self.dims[3], Dim::default());
        assert_eq!(self.dims[2].wrap, 0);
        let d = &self.dims;
        [
            (self.addr as u32 / 4) << 14 | field(self.len_words, 14, 0),
            packet(self.packet, false),
            step(d[0].step, 13) | step(d[1].step, 13) << 13,
            step(d[2].step, 13) | field(d[0].wrap, 8, 13) | field(d[1].wrap, 8, 21),
            iteration(self.iteration, 13),
            tail(self),
        ]
    }

    /// Address space includes neighboring memtiles; lock IDs are the 8-bit DMA view.
    /// Own memory is `0x80000 + byte_offset`; own lock IDs are `64 + lock_id`.
    pub fn memtile_words(&self) -> [u32; 8] {
        assert!(self.addr % 4 == 0 && self.addr < 0x20_0000);
        assert_eq!(self.dims[3].wrap, 0);
        let d = &self.dims;
        let mut address = self.addr as u32 / 4;
        if let Some(next) = self.next {
            assert!(next < 48);
            address |= 1 << 19 | next << 20;
        }
        [
            field(self.len_words, 17, 0) | packet(self.packet, true),
            address,
            step(d[0].step, 17) | field(d[0].wrap, 10, 17) | (self.suppress_tlast as u32) << 31,
            step(d[1].step, 17) | field(d[1].wrap, 10, 17),
            step(d[2].step, 17) | field(d[2].wrap, 10, 17),
            step(d[3].step, 17),
            iteration(self.iteration, 17),
            1 << 31 | locks(self.locks, true),
        ]
    }

    pub fn shim_words(&self) -> [u32; 8] {
        if let Err(e) = self.axi.validate_vendor_word5() {
            panic!("invalid shim BD: {e}");
        }
        self.shim_words_for_hazardous_probe()
    }

    /// Raw field encoding for an explicitly opted-in hazardous probe, never a default design.
    /// Non-vendor word-5 values can wedge the platform; register fit is not a safety guarantee.
    pub fn shim_words_for_hazardous_probe(&self) -> [u32; 8] {
        if let Err(e) = self.axi.validate() {
            panic!("invalid shim BD: {e}");
        }
        assert!(self.addr % 4 == 0 && self.addr < (1 << 48));
        assert_eq!(self.dims[3], Dim::default());
        assert_eq!(self.dims[2].wrap, 0);
        let d = &self.dims;
        [
            self.len_words,
            self.addr as u32,
            (self.addr >> 32) as u32 | packet(self.packet, false),
            step(d[0].step, 20) | field(d[0].wrap, 10, 20),
            field(self.axi.burst, 2, 30) | step(d[1].step, 20) | field(d[1].wrap, 10, 20),
            field(self.axi.cache, 4, 24) | field(self.axi.qos, 4, 20) | step(d[2].step, 20),
            iteration(self.iteration, 20),
            tail(self),
        ]
    }

    pub fn address(loc: Location, id: u32) -> u32 {
        let (base, count) = match loc.kind() {
            TileType::Memtile => (0xa0000, 48),
            _ => (0x1d000, 16),
        };
        assert!(id < count);
        loc.address(base + id * 0x20)
    }

    pub fn emit_cdo(&self, loc: Location, id: u32, cdo: &mut Cdo) {
        let addr = Self::address(loc, id);
        match loc.kind() {
            TileType::Compute => {
                cdo.dma_write(addr, &self.tile_words());
            }
            TileType::Memtile => {
                cdo.dma_write(addr, &self.memtile_words());
            }
            TileType::Shim => {
                cdo.dma_write(addr, &self.shim_words());
            }
        }
    }

    pub fn emit_txn(&self, loc: Location, id: u32, txn: &mut Txn) {
        let addr = Self::address(loc, id);
        match loc.kind() {
            TileType::Compute => {
                txn.block_write(addr, loc.col, loc.row, &self.tile_words());
            }
            TileType::Memtile => {
                txn.block_write(addr, loc.col, loc.row, &self.memtile_words());
            }
            TileType::Shim => {
                txn.block_write(addr, loc.col, loc.row, &self.shim_words());
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    S2mm,
    Mm2s,
}

/// Enqueue a BD chain `repeat` times. Infinite ping-pong uses a cyclic BD chain.
/// Emission only queues; compute/memtile channels must also be enabled.
#[derive(Clone, Copy, Debug)]
pub struct Task {
    pub direction: Direction,
    pub channel: u32,
    pub bd: u32,
    pub repeat: u32,
    pub issue_token: bool,
}
impl Task {
    pub fn write(self, loc: Location) -> (u32, u32) {
        let (base, channels, count) = match loc.kind() {
            TileType::Compute => (0x1de00, 2, 16),
            TileType::Memtile => (0xa0600, 6, 48),
            TileType::Shim => (0x1d200, 2, 16),
        };
        assert!(self.channel < channels && self.bd < count && (1..=256).contains(&self.repeat));
        if loc.kind() == TileType::Memtile {
            assert_eq!(
                self.bd >= 24,
                self.channel % 2 == 1,
                "memtile BD/channel bank mismatch"
            );
        }
        let dir = if self.direction == Direction::Mm2s {
            channels * 8
        } else {
            0
        };
        (
            loc.address(base + dir + self.channel * 8 + 4),
            (self.issue_token as u32) << 31 | (self.repeat - 1) << 16 | self.bd,
        )
    }
    pub fn emit_cdo(self, loc: Location, cdo: &mut Cdo) {
        let (addr, value) = self.write(loc);
        cdo.write(addr, value);
    }
    pub fn emit_txn(self, loc: Location, txn: &mut Txn) {
        let (addr, value) = self.write(loc);
        txn.write32(addr, value);
    }
    /// Enable a compute/memtile channel after queueing its initial descriptor.
    pub fn enable_cdo(self, loc: Location, cdo: &mut Cdo) {
        assert_ne!(loc.kind(), TileType::Shim);
        cdo.mask_write(self.write(loc).0 - 4, 1, 1);
    }
    pub fn enable_txn(self, loc: Location, txn: &mut Txn) {
        assert_ne!(loc.kind(), TileType::Shim);
        txn.mask_write(self.write(loc).0 - 4, 1, 1);
    }
}

/// Initialize a tile's own lock (not a neighboring DMA lock ID).
pub fn lock_write(loc: Location, id: u32, value: u32) -> (u32, u32) {
    let (base, count) = match loc.kind() {
        TileType::Compute => (0x1f000, 16),
        TileType::Memtile => (0xc0000, 64),
        TileType::Shim => (0x14000, 16),
    };
    assert!(id < count && value < 64);
    (loc.address(base + id * 0x10), value)
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Words from the vendor add_one PDI/insts (mlir-aie npu2), used as a field-placement check.
    #[test]
    fn matches_vendor_bds() {
        // S2MM BD0: buf 0x8000, 16 words, acq lock0 -1, rel lock1 +1, next BD1
        let w = tile_bd(
            0x8000,
            16,
            BdLocks {
                acq: Some((0, -1)),
                rel: Some((1, 1)),
            },
            Some(1),
        );
        assert_eq!(w, [0x0800_0010, 0, 0, 0, 0, 0x0e04_3fe0]);
        // MM2S BD3: buf 0x4000, 16 words, acq lock3 -1, rel lock2 +1, next BD2
        let w = tile_bd(
            0x4000,
            16,
            BdLocks {
                acq: Some((3, -1)),
                rel: Some((2, 1)),
            },
            Some(2),
        );
        assert_eq!(w, [0x0400_0010, 0, 0, 0, 0, 0x1604_5fe3]);
        assert_eq!(
            shim_bd(0x1000, 0),
            [0x1000, 0, 0, 0, 0xc000_0000, 0x0200_0000, 0, 0x0200_0000]
        );
    }

    #[test]
    fn shim_axi_validate_rejects_each_overflowing_field() {
        assert!(ShimAxi {
            burst: 3,
            cache: 15,
            qos: 15
        }
        .validate()
        .is_ok());
        assert!(ShimAxi {
            burst: 0,
            cache: 0,
            qos: 0
        }
        .validate()
        .is_ok());
        let e = ShimAxi {
            burst: 4,
            ..ShimAxi::default()
        }
        .validate()
        .unwrap_err();
        assert!(
            e.contains("burst") && e.contains('4') && e.contains("0..=3"),
            "{e}"
        );
        let e = ShimAxi {
            cache: 16,
            ..ShimAxi::default()
        }
        .validate()
        .unwrap_err();
        assert!(
            e.contains("AxCACHE") && e.contains("16") && e.contains("0..=15"),
            "{e}"
        );
        let e = ShimAxi {
            qos: 16,
            ..ShimAxi::default()
        }
        .validate()
        .unwrap_err();
        assert!(
            e.contains("AxQoS") && e.contains("16") && e.contains("0..=15"),
            "{e}"
        );
        // A value that only spills into a neighbouring field (0x10 << 24 would corrupt SMID bit 28) is still caught.
        assert!(ShimAxi {
            cache: 0x10,
            ..ShimAxi::default()
        }
        .validate()
        .is_err());
        assert!(ShimAxi {
            qos: u32::MAX,
            ..ShimAxi::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn normal_shim_emitter_rejects_non_vendor_word5() {
        for axi in [
            ShimAxi {
                qos: 15,
                ..ShimAxi::default()
            },
            ShimAxi {
                cache: 3,
                ..ShimAxi::default()
            },
        ] {
            assert!(axi.validate().is_ok());
            assert!(axi.validate_vendor_word5().is_err());
            let bd = Bd {
                axi,
                ..Bd::new(0, 16)
            };
            assert!(std::panic::catch_unwind(|| bd.shim_words()).is_err());
            let words = bd.shim_words_for_hazardous_probe();
            assert_eq!(
                (words[5] >> 24 & 15, words[5] >> 20 & 15),
                (axi.cache, axi.qos)
            );
        }
    }

    #[test]
    #[should_panic(expected = "AxCACHE 16")]
    fn shim_words_rejects_overflowing_cache_instead_of_truncating() {
        Bd {
            axi: ShimAxi {
                cache: 16,
                ..ShimAxi::default()
            },
            ..Bd::new(0, 16)
        }
        .shim_words();
    }

    #[test]
    #[should_panic(expected = "AxQoS 16")]
    fn shim_words_rejects_overflowing_qos_instead_of_truncating() {
        Bd {
            axi: ShimAxi {
                qos: 16,
                ..ShimAxi::default()
            },
            ..Bd::new(0, 16)
        }
        .shim_words();
    }

    #[test]
    #[should_panic(expected = "burst encoding 4")]
    fn shim_words_rejects_overflowing_burst_instead_of_truncating() {
        Bd {
            axi: ShimAxi {
                burst: 4,
                ..ShimAxi::default()
            },
            ..Bd::new(0, 16)
        }
        .shim_words();
    }

    /// Hand-computed words: AXI attributes sit next to the D1 wrap/step and D2 step fields.
    #[test]
    fn shim_axi_words_with_nontrivial_strides() {
        let mut bd = Bd::new(0x1_2345_6780, 96);
        bd.dims = [
            Dim { step: 3, wrap: 5 },
            Dim {
                step: 0x100,
                wrap: 7,
            },
            Dim {
                step: 0x2000,
                wrap: 0,
            },
            Dim::default(),
        ];
        bd.axi = ShimAxi {
            burst: 2,
            cache: 0xb,
            qos: 9,
        };
        let w = bd.shim_words_for_hazardous_probe();
        assert_eq!(w[3], 2 | 5 << 20);
        assert_eq!(w[4], 2 << 30 | 7 << 20 | 0xff);
        assert_eq!(w[5], 0xb << 24 | 9 << 20 | 0x1fff);
    }

    /// Every legal field value, including the largest, changes only its own bits even when the neighbouring
    /// wrap/step fields and locks/next/tlast are all populated at their maxima.
    #[test]
    fn shim_axi_fields_do_not_leak_into_neighbouring_bd_bits() {
        let mut bd = Bd::new(0xffff_ffff_fffc, 0xdead_beef);
        bd.dims = [
            Dim {
                step: 1 << 20,
                wrap: 1023,
            },
            Dim {
                step: 1 << 20,
                wrap: 1023,
            },
            Dim {
                step: 0xabcde,
                wrap: 0,
            },
            Dim::default(),
        ];
        bd.iteration = Iteration {
            step: 1 << 20,
            wrap: 64,
            current: 63,
        };
        bd.locks = BdLocks {
            acq: Some((15, -64)),
            rel: Some((14, 63)),
        };
        bd.next = Some(15);
        bd.suppress_tlast = true;
        let base = Bd {
            axi: ShimAxi {
                burst: 0,
                cache: 0,
                qos: 0,
            },
            ..bd
        }
        .shim_words_for_hazardous_probe();
        for burst in 0..=ShimAxi::MAX_BURST {
            for cache in [0, 1, 2, 3, 0xb, 0xf] {
                for qos in [0, 1, 8, 0xf] {
                    let w = Bd {
                        axi: ShimAxi { burst, cache, qos },
                        ..bd
                    }
                    .shim_words_for_hazardous_probe();
                    let mut want = base;
                    want[4] |= burst << 30;
                    want[5] |= cache << 24 | qos << 20;
                    assert_eq!(w, want, "burst {burst} cache {cache:#x} qos {qos}");
                }
            }
        }
        // The software default is bytes-identical to the pre-AXI-attribute encoding (burst 3, cache 2, qos 0).
        let d = Bd::new(0x40, 16).shim_words();
        assert_eq!((d[4] >> 30, d[5] >> 24 & 0xf, d[5] >> 20 & 0xf), (3, 2, 0));
    }
}

/// Offline vendor gate helpers. The input is the placed structural MLIR, not
/// decoded BD words: buffer allocation, lock actions and tensor sizes/strides.
#[cfg(test)]
pub(crate) mod vendor_gate {
    use super::*;
    use std::collections::BTreeMap;

    /// The vendor GEMM design used as the BD oracle, under `AIE2P_VENDOR_CORPUS`.
    pub const CORPUS: &str = "032eb944d6cdbf2df2d4b8d9";

    fn number(text: &str, key: &str) -> u32 {
        text.split_once(key)
            .unwrap()
            .1
            .trim_start()
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .unwrap()
            .parse()
            .unwrap()
    }
    fn name(line: &str) -> &str {
        line.split_whitespace().next().unwrap()
    }
    fn argument<'a>(line: &'a str, key: &str) -> &'a str {
        line.split_once(key)
            .unwrap()
            .1
            .split(|c: char| c == ',' || c == ')' || c.is_whitespace())
            .next()
            .unwrap()
    }
    fn list(line: &str, key: &str) -> Vec<u32> {
        line.split_once(key)
            .unwrap()
            .1
            .split(']')
            .next()
            .unwrap()
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect()
    }
    fn descriptor(line: &str, address: u64, dimensions: usize) -> Bd {
        let bytes = if line.split_once('>').unwrap().0.ends_with("xi8") {
            1
        } else {
            4
        };
        let mut bd = Bd::new(
            address + number(line, "offset = ") as u64 * bytes as u64,
            number(line, "len = ") * bytes / 4,
        );
        if line.contains("sizes = [") {
            let sizes = list(line, "sizes = [");
            let strides = list(line, "strides = [");
            assert_eq!(sizes.len(), strides.len());
            for (i, (&size, &stride)) in sizes.iter().zip(&strides).rev().enumerate() {
                bd.dims[i] = Dim {
                    step: (stride * bytes / 4).max(1),
                    wrap: if i + 1 == dimensions {
                        0
                    } else if i == 0 {
                        size * bytes / 4
                    } else {
                        size
                    },
                };
            }
        }
        bd
    }

    pub fn structural_cdo(source: &str) -> Cdo {
        let lines: Vec<_> = source.lines().collect();
        let mut tiles = BTreeMap::new();
        let mut buffers = BTreeMap::new();
        let mut lock_ids = BTreeMap::new();
        for line in &lines {
            if line.contains(" = aie.tile(") {
                tiles.insert(
                    name(line),
                    Location::new(
                        number(line, "aie.tile("),
                        number(line.split_once("aie.tile(").unwrap().1, ","),
                    ),
                );
            } else if line.contains(" = aie.buffer(") {
                let tile = tiles[argument(line, "aie.buffer(")];
                buffers.insert(name(line), (tile, number(line, "address = ")));
            } else if line.contains(" = aie.lock(") {
                let tile = tiles[argument(line, "aie.lock(")];
                lock_ids.insert(
                    name(line),
                    (tile, number(line.split_once("aie.lock(").unwrap().1, ",")),
                );
            }
        }
        let mut out = Cdo::new();
        let mut current = None;
        for (i, line) in lines.iter().enumerate() {
            if line.contains(" = aie.mem(") {
                current = Some(tiles[argument(line, "aie.mem(")]);
            } else if line.contains(" = aie.memtile_dma(") {
                current = Some(tiles[argument(line, "aie.memtile_dma(")]);
            } else if line.contains("aie.dma_bd(") && line.contains("bd_id = ") {
                let tile = current.unwrap();
                let (buffer_tile, address) = buffers[argument(line, "aie.dma_bd(")];
                assert_eq!(buffer_tile, tile);
                let mem = tile.kind() == TileType::Memtile;
                let mut bd = descriptor(
                    line,
                    address as u64 + if mem { 0x80000 } else { 0 },
                    if mem { 4 } else { 3 },
                );
                bd.next = Some(number(line, "next_bd_id = "));
                let acquire = lock_ids[argument(lines[i - 1], "aie.use_lock(")];
                let release = lock_ids[argument(lines[i + 1], "aie.use_lock(")];
                assert_eq!(acquire.0, tile);
                assert_eq!(release.0, tile);
                assert!(lines[i - 1].contains("AcquireGreaterEqual"));
                assert!(lines[i + 1].contains("Release"));
                bd.locks = BdLocks {
                    acq: Some((acquire.1 + if mem { 64 } else { 0 }, -1)),
                    rel: Some((release.1 + if mem { 64 } else { 0 }, 1)),
                };
                bd.emit_cdo(tile, number(line, "bd_id = "), &mut out);
            }
        }
        out
    }

    pub fn structural_txn(source: &str) -> Txn {
        let mut tiles = BTreeMap::new();
        let mut allocations = BTreeMap::new();
        for line in source.lines() {
            if line.contains(" = aie.tile(") {
                tiles.insert(
                    name(line),
                    Location::new(
                        number(line, "aie.tile("),
                        number(line.split_once("aie.tile(").unwrap().1, ","),
                    ),
                );
            } else if line.contains("aie.shim_dma_allocation @") {
                let (_, tail) = line.split_once("aie.shim_dma_allocation @").unwrap();
                let (symbol, args) = tail.split_once('(').unwrap();
                let loc = tiles[args.split(',').next().unwrap()];
                let dir = if args.contains("MM2S") {
                    Direction::Mm2s
                } else {
                    Direction::S2mm
                };
                let channel = args
                    .split(',')
                    .nth(2)
                    .unwrap()
                    .split(')')
                    .next()
                    .unwrap()
                    .trim()
                    .parse::<u32>()
                    .unwrap();
                allocations.insert(symbol, (loc, dir, channel));
            }
        }
        let mut out = Txn::aie2p_8col();
        let mut allocation = None;
        let mut next = [0; 8];
        for line in source.lines() {
            if line.contains("aiex.dma_configure_task_for @") {
                let symbol = line
                    .split_once("aiex.dma_configure_task_for @")
                    .unwrap()
                    .1
                    .split_whitespace()
                    .next()
                    .unwrap();
                allocation = Some(allocations[symbol]);
            } else if line.contains("aie.dma_bd(%arg") {
                let (loc, dir, channel) = allocation.unwrap();
                let id = next[loc.col as usize];
                next[loc.col as usize] += 1;
                let bd = descriptor(line, 0, 3);
                bd.emit_txn(loc, id, &mut out);
                let arg = number(line, "aie.dma_bd(%arg");
                out.ddr_patch(Bd::address(loc, id) + 4, arg, bd.addr);
                if dir == Direction::S2mm {
                    out.mask_write(loc.address(0x1d200 + channel * 8), 15 << 8, 31 << 8);
                }
                Task {
                    direction: dir,
                    channel,
                    bd: id,
                    repeat: 1,
                    issue_token: dir == Direction::S2mm,
                }
                .emit_txn(loc, &mut out);
            }
        }
        out
    }

    pub fn words(bytes: &[u8]) -> Vec<u32> {
        assert_eq!(bytes.len() % 4, 0);
        bytes
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect()
    }
    pub fn cdo_writes(words: &[u32]) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        for command in crate::cdo::parse(words).unwrap() {
            match command {
                crate::cdo::Cmd::Write(a, v) => out.push((a, v)),
                crate::cdo::Cmd::DmaWrite(a, data) => {
                    assert!(a <= u32::MAX as u64);
                    out.extend(
                        data.iter()
                            .enumerate()
                            .map(|(i, &v)| (a as u32 + i as u32 * 4, v)),
                    );
                }
                crate::cdo::Cmd::Other(0x108, p) => {
                    assert_eq!(p[0], 0);
                    out.push((p[1], p[2]));
                }
                _ => {}
            }
        }
        out
    }
    pub fn txn_writes(bytes: &[u8]) -> Vec<(u32, u32)> {
        let w = words(bytes);
        assert_eq!(w[3] as usize, bytes.len());
        let mut out = Vec::new();
        let mut i = 4;
        while i < w.len() {
            let len = match w[i] {
                0 => {
                    out.push((w[i + 2], w[i + 4]));
                    6
                }
                1 => {
                    let n = w[i + 3] as usize / 4;
                    out.extend(
                        w[i + 4..i + n]
                            .iter()
                            .enumerate()
                            .map(|(j, &v)| (w[i + 2] + j as u32 * 4, v)),
                    );
                    n
                }
                3 | 4 => 7,
                0x80 | 0x81 => w[i + 1] as usize / 4,
                op => panic!("unknown TXN opcode {op}"),
            };
            i += len;
        }
        out
    }
    pub fn bd_register(a: u32) -> bool {
        let offset = a & 0xfffff;
        if a >> 20 & 31 == 1 {
            (0xa0000..0xa0600).contains(&offset)
        } else {
            (0x1d000..0x1d200).contains(&offset)
        }
    }
    /// Stable address sort retains repeated writes and each register's transition order.
    pub fn compare(mut actual: Vec<(u32, u32)>, mut expected: Vec<(u32, u32)>, label: &str) {
        actual.sort_by_key(|&(a, _)| a);
        expected.sort_by_key(|&(a, _)| a);
        assert_eq!(actual.len(), expected.len(), "{label}: write count");
        for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert!(a == e, "{label}: write {i}: actual={a:x?}, expected={e:x?}");
        }
        println!("{label}: {} decoded writes; diff empty", actual.len());
    }

    #[test]
    fn reproduces_vendor_whole_array_dma() {
        let Some(corpus) = crate::vendor_corpus() else {
            return;
        };
        let root = corpus.join("cache").join(CORPUS);
        let root = root.as_path();
        let source = std::fs::read_to_string(root.join("input_with_addresses.mlir")).unwrap();
        let original = std::fs::read(root.join("cdo_main/main_aie_cdo_init.bin")).unwrap();
        let produced = structural_cdo(&source);
        compare(
            cdo_writes(&produced.to_words()),
            cdo_writes(&words(&original))
                .into_iter()
                .filter(|&(a, _)| bd_register(a))
                .collect(),
            "CDO BDs",
        );
        let original = std::fs::read(root.join("insts.bin")).unwrap();
        let produced = structural_txn(&source).to_bytes();
        compare(
            txn_writes(&produced),
            txn_writes(&original),
            "TXN BDs + task queues",
        );
    }
}
