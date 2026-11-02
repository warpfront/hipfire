// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//! Shim DMA bandwidth probe: pure data movement DDR -> memtile sink and memtile pattern -> DDR, no cores.
//!
//! Question answered on hardware: is the ~1/3..1/2 of the 4 B/cycle stream rate seen by the V6 whole-array
//! GEMM a per-shim-channel limit (latency / outstanding requests) or an aggregate (NoC / DDR) limit, and what
//! are the separate read and write ceilings? Sweep columns x channels x direction with [`BwConfig`]; per-channel
//! rate vs. number of concurrently active channels separates the two.
//!
//! ## Arguments (firmware DDR-patches arg0, arg1)
//! * arg0 = read buffer, `cols * read_ch * bytes` B, one contiguous segment of `bytes` per (column, channel):
//!   segment index `col * read_ch + ch`. Its contents are never inspected (the sink discards them).
//! * arg1 = write buffer, `cols * write_ch * bytes` B, segment index `col * write_ch + ch`; every segment must
//!   end up equal to the memtile pattern repeated ([`pattern_word`], [`BwDesign::check_write`]).
//!
//! A direction with zero channels still has an argument slot (`bytes: 0`); the host must still pass a BO.
//!
//! ## Data movement (per column `0..cols`, per lane `k < channels`)
//! * Read lane `k`: shim `MM2S k` -> shim `North(k)` -> memtile `South(k)` -> memtile `S2MM (4 + k)` (V6's
//!   B/A S2MM channels) writing a 16 KiB sink buffer (`0x8000 + k*0x4000`) with a cyclic lock-free BD
//!   (`next = itself`): it accepts data at line rate forever. Shim BD id `k`, one finite 1-D BD of `bytes`,
//!   DDR-patched to arg0 at its segment offset, completion token to the controller.
//! * Write lane `k`: memtile `MM2S k` -> memtile `South(k)` -> shim `North(k)` -> shim `S2MM k`. The source is a
//!   16 KiB memtile buffer (`0x14000 + k*0x4000`) initialised in the CDO with word `i` =
//!   `0xB0000000 | col << 20 | ch << 16 | i`. Shim BD id `2 + k`, one finite BD of `bytes` into arg1, token.
//! * Lanes never pass through the memtile switch (every lane ends in a memtile DMA), so the AIE2P rule that a
//!   memtile North/South pass-through must keep its lane is not exercised. Every switch master is driven by
//!   exactly one slave; shim ports are duplex (MM2S out `North(k)` vs S2MM in `North(k)`, memtile `South(k)` in
//!   vs out).
//! * Memtile BD banks: even channel BD < 24, odd >= 24. Sink BDs: S2MM4 -> 23, S2MM5 -> 47. Write chains:
//!   MM2S0 -> 0..=16, MM2S1 -> 24..=40.
//!
//! ## Deviation from "cyclic BD" for the write source
//! A cyclic source cannot be used when the shim S2MM is finite: after the shim BD completes the memtile keeps
//! emitting, and the words stranded in the stream-switch FIFOs would precede the next submit's stream
//! (rotated pattern, wrong result on a resubmitted context). The write source is therefore finite and emits
//! exactly `bytes`: a chain of [`CHAIN`] lock-free BDs over the same 16 KiB buffer, queued as at most three tasks
//! (chain x `bytes/(16 KiB * CHAIN)` repeats, the suffix of the chain for the remaining whole buffers, one short BD
//! for `bytes % 16 KiB`). The cyclic sink needs no such care (it discards).
//!
//! ## Submit / context reuse
//! The CDO loads state that survives a run: memtile pattern, memtile BDs (the short tail BD is rewritten per
//! submit), stream-switch circuits, shim NOC mux/demux and the token route. It queues nothing. Every submit (TXN):
//! 1. assert + deassert RESET (CTRL bit1) on every used memtile channel (V6 pattern; shim channels are never
//!    reset, their finite queues drained before the tokens that ended the previous submit);
//! 2. write the tail BD, set the shim token controller id (0xF) on every used shim channel, rewrite the finite
//!    shim BDs and DDR-patch them;
//! 3. queue the memtile sinks / sources, then (all columns) the shim tasks with `issue_token`;
//! 4. one aggregate `SYNC(col 0, ncol = cols)` per active (direction, channel): MM2S for reads, S2MM for writes.
//!
//! ## Shim AXI attributes
//! [`BwConfig::axi`] sets shim BD burst length, AxCACHE and AxQoS ([`ShimAxi`]). Default designs use vendor
//! word-5 attributes (AxCACHE 2, AxQoS 0); burst encodings 0..=3 remain configurable. Non-vendor word-5
//! attributes require [`design_for_hazardous_probe`] and can wedge the platform: AxQoS 15 wedged Halo and
//! subsequent SMU resume failed with -22. The functional simulator proves only encoding and movement, not
//! hardware safety, coherence, or performance.
use super::gemm_i8::{ArgKind, ArgSpec};
use crate::{
    cdo::Cdo,
    dma::{
        Bd,
        Direction::{self, Mm2s, S2mm},
        Location, ShimAxi, Task,
    },
    regs,
    route::{Circuit, Port, ShimDma},
    txn::Txn,
};

/// Memtile DMA own-memory view base.
const MEM_BASE: u32 = 0x80000;
/// Bytes / words of every memtile source and sink buffer.
pub const BUF_BYTES: usize = 16 * 1024;
const BUF_WORDS: u32 = (BUF_BYTES / 4) as u32;
/// Largest per-channel transfer (the 256-repeat task limit times [`CHAIN`] 16 KiB BDs).
pub const MAX_BYTES: usize = 64 << 20;
/// Whole 16 KiB source BDs in one write chain.
pub const CHAIN: usize = 16;
/// Memtile S2MM channel of read lane 0 (V6 B S2MM4, A S2MM5); lane `k` uses `READ_S2MM0 + k`.
const READ_S2MM0: u32 = 4;
const SINK_BD: [u32; 2] = [23, 47];
const SINK_OFF: u32 = 0x8000;
// V9's established B staging region starts at 0x14000. Keep persistent sources away from memtile word 0:
// hardware observed 1024 bad words in a 16 MiB lane, matching one corrupted word per 16 KiB source wrap.
// The origin of 0x00cd0cd0 is unproven (firmware execution is closed); this is an address intervention,
// not a claim that the complete low region is documented as firmware-reserved. Sinks remain disjoint.
const SOURCE_OFF: u32 = 0x14000;
const LANE_STRIDE: u32 = 0x4000;
const WRITE_BD0: [u32; 2] = [0, 24];
const SHIM_READ_BD: u32 = 0;
const SHIM_WRITE_BD: u32 = 2;
pub const READ_ARG: u32 = 0;
pub const WRITE_ARG: u32 = 1;
/// Aggregate-clock array cycles per second used by callers for "bytes per AIE cycle".
pub const AIE_HZ: f64 = 1.8e9;

/// Probe geometry: columns `0..cols`, `read_ch` shim MM2S and `write_ch` shim S2MM channels per column,
/// `bytes` per channel, shim BD AXI attributes `axi`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BwConfig {
    pub cols: usize,
    pub read_ch: usize,
    pub write_ch: usize,
    pub bytes: usize,
    pub axi: ShimAxi,
}

impl BwConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.axi.validate_vendor_word5()?;
        self.validate_for_hazardous_probe()
    }
    /// Geometry and raw register-fit checks only, for an explicitly opted-in hazardous probe.
    pub fn validate_for_hazardous_probe(&self) -> Result<(), String> {
        self.axi.validate()?;
        if !(1..=8).contains(&self.cols) {
            return Err(format!("cols {} not in 1..=8", self.cols));
        }
        if self.read_ch > 2 || self.write_ch > 2 {
            return Err("read/write channels per column must be 0..=2".into());
        }
        if self.read_ch + self.write_ch == 0 {
            return Err("at least one of read/write channels must be > 0".into());
        }
        if self.bytes == 0 || self.bytes % 4096 != 0 || self.bytes > MAX_BYTES {
            return Err(format!(
                "bytes {} must be a multiple of 4096 in 4096..={MAX_BYTES}",
                self.bytes
            ));
        }
        Ok(())
    }
    pub fn read_total(&self) -> usize {
        self.cols * self.read_ch * self.bytes
    }
    pub fn write_total(&self) -> usize {
        self.cols * self.write_ch * self.bytes
    }
}

/// Built probe image + command stream; no NPU access occurs here.
pub struct BwDesign {
    pub pdi: Vec<u8>,
    pub insts: Vec<u8>,
    /// arg0 = read buffer (In), arg1 = write buffer (Out); either may be 0 bytes.
    pub args: Vec<ArgSpec>,
    pub cfg: BwConfig,
}

/// Result of [`BwDesign::check_write`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WriteCheck {
    pub bad_words: usize,
    /// `(byte offset in arg1, got, want)` of the first mismatching word.
    pub first_bad: Option<(usize, u32, u32)>,
}

/// Word `i` (any index; the pattern repeats every 4096 words) of the memtile source of `(col, ch)`.
pub fn pattern_word(col: u32, ch: u32, i: usize) -> u32 {
    0xB000_0000 | col << 20 | ch << 16 | (i as u32 % BUF_WORDS)
}

impl BwDesign {
    /// Byte range of read segment `(col, ch)` in arg0.
    pub fn read_segment(&self, col: usize, ch: usize) -> std::ops::Range<usize> {
        assert!(col < self.cfg.cols && ch < self.cfg.read_ch);
        let at = (col * self.cfg.read_ch + ch) * self.cfg.bytes;
        at..at + self.cfg.bytes
    }
    /// Byte range of write segment `(col, ch)` in arg1.
    pub fn write_segment(&self, col: usize, ch: usize) -> std::ops::Range<usize> {
        assert!(col < self.cfg.cols && ch < self.cfg.write_ch);
        let at = (col * self.cfg.write_ch + ch) * self.cfg.bytes;
        at..at + self.cfg.bytes
    }
    /// The 16 KiB memtile pattern of `(col, ch)`.
    pub fn pattern(&self, col: usize, ch: usize) -> Vec<u8> {
        (0..BUF_WORDS as usize)
            .flat_map(|i| pattern_word(col as u32, ch as u32, i).to_le_bytes())
            .collect()
    }
    /// Compare the whole arg1 against the memtile pattern repeated in every segment.
    pub fn check_write(&self, buf: &[u8]) -> WriteCheck {
        assert_eq!(buf.len(), self.cfg.write_total(), "write buffer size");
        let mut out = WriteCheck {
            bad_words: 0,
            first_bad: None,
        };
        for col in 0..self.cfg.cols {
            for ch in 0..self.cfg.write_ch {
                let seg = self.write_segment(col, ch);
                let pattern = self.pattern(col, ch);
                for (n, chunk) in buf[seg.clone()].chunks(BUF_BYTES).enumerate() {
                    if chunk == &pattern[..chunk.len()] {
                        continue;
                    }
                    for (w, got) in chunk.chunks_exact(4).enumerate() {
                        let got = u32::from_le_bytes(got.try_into().unwrap());
                        let want =
                            u32::from_le_bytes(pattern[w * 4..w * 4 + 4].try_into().unwrap());
                        if got != want {
                            out.bad_words += 1;
                            out.first_bad.get_or_insert((
                                seg.start + n * BUF_BYTES + w * 4,
                                got,
                                want,
                            ));
                        }
                    }
                }
            }
        }
        out
    }
}

fn shim_loc(col: u32) -> Location {
    Location::new(col, 0)
}
fn mem_loc(col: u32) -> Location {
    Location::new(col, 1)
}

/// Source (write lane) memtile channel and its BDs.
fn write_chain_bd(ch: usize, i: usize) -> u32 {
    WRITE_BD0[ch] + i as u32
}
fn write_rem_bd(ch: usize) -> u32 {
    WRITE_BD0[ch] + CHAIN as u32
}
fn source_off(ch: usize) -> u32 {
    SOURCE_OFF + ch as u32 * LANE_STRIDE
}
fn sink_off(ch: usize) -> u32 {
    SINK_OFF + ch as u32 * LANE_STRIDE
}

fn circuits(cfg: BwConfig, col: u32) -> Vec<Circuit> {
    let (shim, mem) = (shim_loc(col), mem_loc(col));
    let mut v = Vec::new();
    for k in 0..cfg.read_ch as u8 {
        let mm = ShimDma {
            tile: shim,
            direction: Mm2s,
            channel: k,
        };
        v.push(Circuit {
            tile: shim,
            slave: mm.port(),
            master: Port::North(k),
        });
        v.push(Circuit {
            tile: mem,
            slave: Port::South(k),
            master: Port::Dma(READ_S2MM0 as u8 + k),
        });
    }
    for k in 0..cfg.write_ch as u8 {
        let sm = ShimDma {
            tile: shim,
            direction: S2mm,
            channel: k,
        };
        v.push(Circuit {
            tile: mem,
            slave: Port::Dma(k),
            master: Port::South(k),
        });
        v.push(Circuit {
            tile: shim,
            slave: Port::North(k),
            master: sm.port(),
        });
    }
    v
}

fn shim_routes(cfg: BwConfig, col: u32) -> Vec<ShimDma> {
    let shim = shim_loc(col);
    (0..cfg.read_ch as u8)
        .map(|k| ShimDma {
            tile: shim,
            direction: Mm2s,
            channel: k,
        })
        .chain((0..cfg.write_ch as u8).map(|k| ShimDma {
            tile: shim,
            direction: S2mm,
            channel: k,
        }))
        .collect()
}

/// Memtile BDs of a column: `(id, bd)`; independent of `bytes`.
fn memtile_descriptors(cfg: BwConfig) -> Vec<(u32, Bd)> {
    let mut v = Vec::new();
    for k in 0..cfg.read_ch {
        let mut bd = Bd::new((MEM_BASE + sink_off(k)) as u64, BUF_WORDS);
        bd.next = Some(SINK_BD[k]); // cyclic, no locks
        v.push((SINK_BD[k], bd));
    }
    for k in 0..cfg.write_ch {
        for i in 0..CHAIN {
            let mut bd = Bd::new((MEM_BASE + source_off(k)) as u64, BUF_WORDS);
            if i + 1 < CHAIN {
                bd.next = Some(write_chain_bd(k, i + 1));
            }
            v.push((write_chain_bd(k, i), bd));
        }
    }
    v
}

/// Task plan of one write lane: `(start bd, repeat)` in queue order. Emits exactly `bytes`.
fn write_tasks(ch: usize, bytes: usize) -> Vec<(u32, u32)> {
    let (q, rem) = (bytes / BUF_BYTES, bytes % BUF_BYTES);
    let (repeat, tail) = (q / CHAIN, q % CHAIN);
    let mut t = Vec::new();
    if repeat > 0 {
        t.push((write_chain_bd(ch, 0), repeat as u32));
    }
    // The suffix of the chain ends at the last BD (no next), so it is a valid shorter chain.
    if tail > 0 {
        t.push((write_chain_bd(ch, CHAIN - tail), 1));
    }
    if rem > 0 {
        t.push((write_rem_bd(ch), 1));
    }
    t
}

/// CTRL register of a channel (bit1 reset on memtile, bits 8..15 token controller id on shim).
fn channel_ctrl(loc: Location, direction: Direction, channel: u32) -> u32 {
    let bd = if loc.row == 1 && channel % 2 == 1 {
        24
    } else {
        0
    };
    Task {
        direction,
        channel,
        bd,
        repeat: 1,
        issue_token: false,
    }
    .write(loc)
    .0 - 4
}

/// Memtile channels used by the probe `(direction, channel)`.
fn memtile_channels(cfg: BwConfig) -> Vec<(Direction, u32)> {
    (0..cfg.read_ch as u32)
        .map(|k| (S2mm, READ_S2MM0 + k))
        .chain((0..cfg.write_ch as u32).map(|k| (Mm2s, k)))
        .collect()
}

fn build_cdo(cfg: BwConfig) -> Cdo {
    let mut cdo = Cdo::new();
    for col in 0..cfg.cols as u32 {
        emit_column_cdo(&mut cdo, cfg, col);
    }
    cdo
}

/// Persistent state of one column; only `cfg.read_ch` / `cfg.write_ch` (the lanes of this column) are used.
fn emit_column_cdo(cdo: &mut Cdo, cfg: BwConfig, col: u32) {
    for (id, bd) in memtile_descriptors(cfg) {
        bd.emit_cdo(mem_loc(col), id, cdo);
    }
    for k in 0..cfg.write_ch {
        let words: Vec<u32> = (0..BUF_WORDS as usize)
            .map(|i| pattern_word(col, k as u32, i))
            .collect();
        cdo.dma_write(mem_loc(col).address(source_off(k)), &words);
    }
    for c in circuits(cfg, col) {
        c.emit_cdo(cdo);
    }
    for r in shim_routes(cfg, col) {
        r.emit_cdo(cdo);
    }
    for (addr, value) in regs::shim_token_route(col) {
        cdo.write(addr, value);
    }
}

fn build_txn(cfg: BwConfig) -> Txn {
    build_txn_for_probe(cfg, false)
}

fn build_txn_for_probe(cfg: BwConfig, hazardous: bool) -> Txn {
    let mut txn = Txn::aie2p_8col();
    let words = (cfg.bytes / 4) as u32;
    let rem_words = ((cfg.bytes % BUF_BYTES) / 4) as u32;
    // 1. reset used memtile channels, then release the reset.
    let chans: Vec<(Location, Direction, u32)> = (0..cfg.cols as u32)
        .flat_map(|c| {
            memtile_channels(cfg)
                .into_iter()
                .map(move |(d, ch)| (mem_loc(c), d, ch))
        })
        .collect();
    for &(loc, d, ch) in &chans {
        txn.mask_write(channel_ctrl(loc, d, ch), 2, 2);
    }
    for &(loc, d, ch) in &chans {
        txn.mask_write(channel_ctrl(loc, d, ch), 0, 2);
    }
    // 2. tail BD, shim token ids, shim BDs + DDR patches.
    for col in 0..cfg.cols as u32 {
        if rem_words > 0 {
            for k in 0..cfg.write_ch {
                Bd::new((MEM_BASE + source_off(k)) as u64, rem_words).emit_txn(
                    mem_loc(col),
                    write_rem_bd(k),
                    &mut txn,
                );
            }
        }
        let shim = shim_loc(col);
        for k in 0..cfg.read_ch {
            txn.mask_write(channel_ctrl(shim, Mm2s, k as u32), 0xf00, 0x1f00);
            let id = SHIM_READ_BD + k as u32;
            emit_shim_bd(
                Bd {
                    axi: cfg.axi,
                    ..Bd::new(0, words)
                },
                shim,
                id,
                &mut txn,
                hazardous,
            );
            let off = ((col as usize * cfg.read_ch + k) * cfg.bytes) as u64;
            txn.ddr_patch(Bd::address(shim, id) + 4, READ_ARG, off);
        }
        for k in 0..cfg.write_ch {
            txn.mask_write(channel_ctrl(shim, S2mm, k as u32), 0xf00, 0x1f00);
            let id = SHIM_WRITE_BD + k as u32;
            emit_shim_bd(
                Bd {
                    axi: cfg.axi,
                    ..Bd::new(0, words)
                },
                shim,
                id,
                &mut txn,
                hazardous,
            );
            let off = ((col as usize * cfg.write_ch + k) * cfg.bytes) as u64;
            txn.ddr_patch(Bd::address(shim, id) + 4, WRITE_ARG, off);
        }
    }
    // 3. memtile sinks / sources, then the shim tasks of every column.
    for col in 0..cfg.cols as u32 {
        let mem = mem_loc(col);
        for k in 0..cfg.read_ch {
            Task {
                direction: S2mm,
                channel: READ_S2MM0 + k as u32,
                bd: SINK_BD[k],
                repeat: 1,
                issue_token: false,
            }
            .emit_txn(mem, &mut txn);
        }
        for k in 0..cfg.write_ch {
            for (bd, repeat) in write_tasks(k, cfg.bytes) {
                Task {
                    direction: Mm2s,
                    channel: k as u32,
                    bd,
                    repeat,
                    issue_token: false,
                }
                .emit_txn(mem, &mut txn);
            }
        }
    }
    for col in 0..cfg.cols as u32 {
        let shim = shim_loc(col);
        for k in 0..cfg.write_ch {
            Task {
                direction: S2mm,
                channel: k as u32,
                bd: SHIM_WRITE_BD + k as u32,
                repeat: 1,
                issue_token: true,
            }
            .emit_txn(shim, &mut txn);
        }
        for k in 0..cfg.read_ch {
            Task {
                direction: Mm2s,
                channel: k as u32,
                bd: SHIM_READ_BD + k as u32,
                repeat: 1,
                issue_token: true,
            }
            .emit_txn(shim, &mut txn);
        }
    }
    // 4. one aggregate SYNC per active (direction, channel).
    for k in 0..cfg.read_ch as u32 {
        txn.sync(0, 0, 1, k, cfg.cols as u32, 1);
    }
    for k in 0..cfg.write_ch as u32 {
        txn.sync(0, 0, 0, k, cfg.cols as u32, 1);
    }
    txn
}

fn emit_shim_bd(bd: Bd, loc: Location, id: u32, txn: &mut Txn, hazardous: bool) {
    if hazardous {
        txn.block_write(
            Bd::address(loc, id),
            loc.col,
            loc.row,
            &bd.shim_words_for_hazardous_probe(),
        );
    } else {
        bd.emit_txn(loc, id, txn);
    }
}

/// Build the probe (see the module header). Panics on an invalid [`BwConfig`] (use [`BwConfig::validate`]).
pub fn design(cfg: BwConfig) -> BwDesign {
    cfg.validate()
        .unwrap_or_else(|e| panic!("invalid bw_probe config: {e}"));
    build_design(cfg, build_txn(cfg))
}

/// Build one explicitly opted-in hazardous probe. Non-vendor AXI attributes can wedge the platform.
/// This is not a safety guarantee and must never be selected by an automatic/default experiment.
pub fn design_for_hazardous_probe(cfg: BwConfig) -> BwDesign {
    cfg.validate_for_hazardous_probe()
        .unwrap_or_else(|e| panic!("invalid bw_probe config: {e}"));
    build_design(cfg, build_txn_for_probe(cfg, true))
}

fn build_design(cfg: BwConfig, txn: Txn) -> BwDesign {
    BwDesign {
        pdi: crate::pdi::build(&build_cdo(cfg).to_words()),
        insts: txn.to_bytes(),
        args: vec![
            ArgSpec {
                bytes: cfg.read_total(),
                kind: ArgKind::In,
            },
            ArgSpec {
                bytes: cfg.write_total(),
                kind: ArgKind::Out,
            },
        ],
        cfg,
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Concurrency probe: additional to (not a replacement of) the `BwConfig` probe above.
//
// `read_channels` / `write_channels` count ACTIVE shim channels in total, indexed `0..N`; index `i` is column
// `i % 8`, lane `i / 8` (see [`channel_location`]), so 16 channels fill both lanes of all 8 columns. Every active
// channel queues `queued_tasks` (Q) independent finite shim BDs of `bd_bytes` each, repeat 1, a completion token only
// on the last. Data movement per channel is the established one (memtile sink / 16 KiB source, same routes,
// descriptors, buffers and finite `write_tasks`); the memtile source emits `Q * bd_bytes` as ONE continuous stream
// (never restarted per task), so task `t` of a write channel receives words `t * bd_bytes / 4 ..`.
//
// Shim BD ids per column (max [`CONCURRENCY_SHIM_BDS`] = 16): read lane 0 `0..4`, read lane 1 `4..8`, write lane 0
// `8..12`, write lane 1 `12..16`; task `t` is `lane * 4 + t` within its block.
//
// Arguments: arg0 (read) segment `(index * Q + task)`, arg1 (write) segment `(index * Q + task)`, each `bd_bytes`.
// ---------------------------------------------------------------------------------------------------------------------

/// Per-channel shim DMA task-queue depth the probe is sized for (Q must be 1, 2 or 4, never above it).
/// Source: aie-rt `driver/src/global/xaie2pgbl_reginit.c`, `Aie2PShimDmaChProp` `.StartQSizeMax = 4U`; the task-queue
/// token bit is the `.EnToken` field of the same task-queue register description.
pub const CONCURRENCY_QUEUE_CAPACITY: usize = 4;
/// Shim BDs per column; mixed 16 channels x Q 4 allocates exactly this many (read lanes 0..8, write lanes 8..16).
/// Source: aie-rt `driver/src/global/xaie2pgbl_reginit.c`, `Aie2PShimDmaMod` `.NumBds = 16U` (`.NumChannels = 2U`,
/// `.EnTokenIssue` available).
pub const CONCURRENCY_SHIM_BDS: usize = 16;
const _: () = assert!(2 * 2 * CONCURRENCY_QUEUE_CAPACITY == CONCURRENCY_SHIM_BDS);
/// Smallest and largest per-task BD size, bytes (powers of two between them).
pub const CONCURRENCY_MIN_BD_BYTES: usize = 4096;
pub const CONCURRENCY_MAX_BD_BYTES: usize = 1 << 20;
/// Columns of the array; channel index `i` lives in column `i % COLUMNS`.
const COLUMNS: usize = 8;
const ACTIVE_CHANNELS: [usize; 6] = [0, 1, 2, 4, 8, 16];

/// `(column, lane)` of active-channel index `index` (`index < 16`): column `index % 8`, lane `index / 8`.
pub fn channel_location(index: usize) -> (usize, usize) {
    assert!(
        index < 2 * COLUMNS,
        "channel index {index} must be < {}",
        2 * COLUMNS
    );
    (index % COLUMNS, index / COLUMNS)
}

/// Active channels of `lane` for `total` active channels: contiguous columns `0..n`.
fn lane_columns(total: usize, lane: usize) -> usize {
    total.saturating_sub(lane * COLUMNS).min(COLUMNS)
}

/// Concurrency probe geometry: `read_channels` shim MM2S and `write_channels` shim S2MM channels active in total
/// (each 0 or one of 1, 2, 4, 8, 16), each queueing `queued_tasks` BDs of `bd_bytes`, with shim BD AXI attributes `axi`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConcurrencyConfig {
    pub read_channels: usize,
    pub write_channels: usize,
    pub queued_tasks: usize,
    pub bd_bytes: usize,
    pub axi: ShimAxi,
}

impl ConcurrencyConfig {
    /// Rejects non-vendor AXI word-5 attributes, unsupported channel counts, queue depths and BD sizes.
    pub fn validate(&self) -> Result<(), String> {
        self.axi.validate_vendor_word5()?;
        self.axi.validate()?;
        for (name, n) in [
            ("read_channels", self.read_channels),
            ("write_channels", self.write_channels),
        ] {
            if !ACTIVE_CHANNELS.contains(&n) {
                return Err(format!("{name} {n} must be one of {ACTIVE_CHANNELS:?}"));
            }
        }
        if self.read_channels + self.write_channels == 0 {
            return Err("at least one of read_channels/write_channels must be > 0".into());
        }
        if ![1, 2, 4].contains(&self.queued_tasks) || self.queued_tasks > CONCURRENCY_QUEUE_CAPACITY
        {
            return Err(format!(
                "queued_tasks {} must be 1, 2 or 4 (queue capacity {CONCURRENCY_QUEUE_CAPACITY})",
                self.queued_tasks
            ));
        }
        if !self.bd_bytes.is_power_of_two()
            || !(CONCURRENCY_MIN_BD_BYTES..=CONCURRENCY_MAX_BD_BYTES).contains(&self.bd_bytes)
        {
            return Err(format!(
                "bd_bytes {} must be a power of two in {CONCURRENCY_MIN_BD_BYTES}..={CONCURRENCY_MAX_BD_BYTES}", self.bd_bytes));
        }
        // Q <= queue capacity and 2 directions x 2 lanes x capacity == CONCURRENCY_SHIM_BDS (const-asserted above).
        Ok(())
    }
    pub fn read_total(&self) -> usize {
        self.read_channels * self.queued_tasks * self.bd_bytes
    }
    pub fn write_total(&self) -> usize {
        self.write_channels * self.queued_tasks * self.bd_bytes
    }
    /// `(column, lane)` of active-channel `index`; same as the free [`channel_location`].
    pub fn channel_location(&self, index: usize) -> (usize, usize) {
        channel_location(index)
    }
    /// Columns with at least one active channel: `0..active_columns()`.
    fn active_columns(&self) -> usize {
        self.read_channels.max(self.write_channels).min(COLUMNS)
    }
    /// Lane counts of one column as a [`BwConfig`] so the established per-column helpers apply; `bytes` is the
    /// whole per-channel stream (`Q * bd_bytes`).
    fn column(&self, col: usize) -> BwConfig {
        let lanes = |total| {
            (0..2)
                .filter(|&lane| col < lane_columns(total, lane))
                .count()
        };
        BwConfig {
            cols: 1,
            read_ch: lanes(self.read_channels),
            write_ch: lanes(self.write_channels),
            bytes: self.queued_tasks * self.bd_bytes,
            axi: self.axi,
        }
    }
}

/// Built concurrency probe image + command stream; no NPU access occurs here.
pub struct ConcurrencyDesign {
    pub pdi: Vec<u8>,
    pub insts: Vec<u8>,
    /// arg0 = read buffer (In), arg1 = write buffer (Out); either may be 0 bytes.
    pub args: Vec<ArgSpec>,
    pub cfg: ConcurrencyConfig,
}

impl ConcurrencyDesign {
    /// Byte range of read task `task` of active channel `index` in arg0: contiguous, `(index * Q + task) * bd_bytes`.
    pub fn read_segment(&self, index: usize, task: usize) -> std::ops::Range<usize> {
        assert!(index < self.cfg.read_channels && task < self.cfg.queued_tasks);
        let at = (index * self.cfg.queued_tasks + task) * self.cfg.bd_bytes;
        at..at + self.cfg.bd_bytes
    }
    /// Byte range of write task `task` of active channel `index` in arg1: contiguous, `(index * Q + task) * bd_bytes`.
    pub fn write_segment(&self, index: usize, task: usize) -> std::ops::Range<usize> {
        assert!(index < self.cfg.write_channels && task < self.cfg.queued_tasks);
        let at = (index * self.cfg.queued_tasks + task) * self.cfg.bd_bytes;
        at..at + self.cfg.bd_bytes
    }
    /// Compare the whole arg1: word `w` of task `t` of channel `(col, lane)` must be
    /// `pattern_word(col, lane, t * bd_bytes / 4 + w)`.
    pub fn check_write(&self, buf: &[u8]) -> WriteCheck {
        assert_eq!(buf.len(), self.cfg.write_total(), "write buffer size");
        let task_words = self.cfg.bd_bytes / 4;
        let mut out = WriteCheck {
            bad_words: 0,
            first_bad: None,
        };
        for index in 0..self.cfg.write_channels {
            let (col, lane) = channel_location(index);
            for task in 0..self.cfg.queued_tasks {
                let seg = self.write_segment(index, task);
                for (w, got) in buf[seg.clone()].chunks_exact(4).enumerate() {
                    let got = u32::from_le_bytes(got.try_into().unwrap());
                    let want = pattern_word(col as u32, lane as u32, task * task_words + w);
                    if got != want {
                        out.bad_words += 1;
                        out.first_bad.get_or_insert((seg.start + w * 4, got, want));
                    }
                }
            }
        }
        out
    }
}

/// Shim BD id of read `(lane, task)`: lane 0 `0..4`, lane 1 `4..8`.
fn concurrency_read_bd(lane: usize, task: usize) -> u32 {
    (lane * CONCURRENCY_QUEUE_CAPACITY + task) as u32
}
/// Shim BD id of write `(lane, task)`: lane 0 `8..12`, lane 1 `12..16`.
fn concurrency_write_bd(lane: usize, task: usize) -> u32 {
    (2 * CONCURRENCY_QUEUE_CAPACITY + lane * CONCURRENCY_QUEUE_CAPACITY + task) as u32
}

fn build_concurrency_cdo(cfg: ConcurrencyConfig) -> Cdo {
    let mut cdo = Cdo::new();
    for col in 0..cfg.active_columns() {
        emit_column_cdo(&mut cdo, cfg.column(col), col as u32);
    }
    cdo
}

fn build_concurrency_txn(cfg: ConcurrencyConfig) -> Txn {
    let mut txn = Txn::aie2p_8col();
    let words = (cfg.bd_bytes / 4) as u32;
    let total = cfg.queued_tasks * cfg.bd_bytes;
    let rem_words = ((total % BUF_BYTES) / 4) as u32;
    let cols: Vec<(u32, BwConfig)> = (0..cfg.active_columns())
        .map(|c| (c as u32, cfg.column(c)))
        .collect();
    // 1. reset used memtile channels, then release the reset.
    let chans: Vec<(Location, Direction, u32)> = cols
        .iter()
        .flat_map(|&(c, lanes)| {
            memtile_channels(lanes)
                .into_iter()
                .map(move |(d, ch)| (mem_loc(c), d, ch))
        })
        .collect();
    for &(loc, d, ch) in &chans {
        txn.mask_write(channel_ctrl(loc, d, ch), 2, 2);
    }
    for &(loc, d, ch) in &chans {
        txn.mask_write(channel_ctrl(loc, d, ch), 0, 2);
    }
    // 2. tail BD, shim token ids, one distinct shim BD + DDR patch per (channel, task).
    for &(col, lanes) in &cols {
        if rem_words > 0 {
            for k in 0..lanes.write_ch {
                Bd::new((MEM_BASE + source_off(k)) as u64, rem_words).emit_txn(
                    mem_loc(col),
                    write_rem_bd(k),
                    &mut txn,
                );
            }
        }
        let shim = shim_loc(col);
        for lane in 0..lanes.read_ch {
            txn.mask_write(channel_ctrl(shim, Mm2s, lane as u32), 0xf00, 0x1f00);
            let index = col as usize + lane * COLUMNS;
            for task in 0..cfg.queued_tasks {
                let id = concurrency_read_bd(lane, task);
                emit_shim_bd(
                    Bd {
                        axi: cfg.axi,
                        ..Bd::new(0, words)
                    },
                    shim,
                    id,
                    &mut txn,
                    false,
                );
                let off = ((index * cfg.queued_tasks + task) * cfg.bd_bytes) as u64;
                txn.ddr_patch(Bd::address(shim, id) + 4, READ_ARG, off);
            }
        }
        for lane in 0..lanes.write_ch {
            txn.mask_write(channel_ctrl(shim, S2mm, lane as u32), 0xf00, 0x1f00);
            let index = col as usize + lane * COLUMNS;
            for task in 0..cfg.queued_tasks {
                let id = concurrency_write_bd(lane, task);
                emit_shim_bd(
                    Bd {
                        axi: cfg.axi,
                        ..Bd::new(0, words)
                    },
                    shim,
                    id,
                    &mut txn,
                    false,
                );
                let off = ((index * cfg.queued_tasks + task) * cfg.bd_bytes) as u64;
                txn.ddr_patch(Bd::address(shim, id) + 4, WRITE_ARG, off);
            }
        }
    }
    // 3. memtile sinks / finite sources (one continuous `Q * bd_bytes` stream), then the shim queues of every column.
    for &(col, lanes) in &cols {
        let mem = mem_loc(col);
        for k in 0..lanes.read_ch {
            Task {
                direction: S2mm,
                channel: READ_S2MM0 + k as u32,
                bd: SINK_BD[k],
                repeat: 1,
                issue_token: false,
            }
            .emit_txn(mem, &mut txn);
        }
        for k in 0..lanes.write_ch {
            for (bd, repeat) in write_tasks(k, total) {
                Task {
                    direction: Mm2s,
                    channel: k as u32,
                    bd,
                    repeat,
                    issue_token: false,
                }
                .emit_txn(mem, &mut txn);
            }
        }
    }
    for &(col, lanes) in &cols {
        let shim = shim_loc(col);
        for lane in 0..lanes.write_ch {
            for task in 0..cfg.queued_tasks {
                Task {
                    direction: S2mm,
                    channel: lane as u32,
                    bd: concurrency_write_bd(lane, task),
                    repeat: 1,
                    issue_token: task + 1 == cfg.queued_tasks,
                }
                .emit_txn(shim, &mut txn);
            }
        }
        for lane in 0..lanes.read_ch {
            for task in 0..cfg.queued_tasks {
                Task {
                    direction: Mm2s,
                    channel: lane as u32,
                    bd: concurrency_read_bd(lane, task),
                    repeat: 1,
                    issue_token: task + 1 == cfg.queued_tasks,
                }
                .emit_txn(shim, &mut txn);
            }
        }
    }
    // 4. one SYNC per active (direction, lane) over its contiguous active columns `0..n`.
    for lane in 0..2 {
        let n = lane_columns(cfg.read_channels, lane) as u32;
        if n > 0 {
            txn.sync(0, 0, 1, lane as u32, n, 1);
        }
    }
    for lane in 0..2 {
        let n = lane_columns(cfg.write_channels, lane) as u32;
        if n > 0 {
            txn.sync(0, 0, 0, lane as u32, n, 1);
        }
    }
    txn
}

/// Build the concurrency probe (see the section header above). Vendor-safe AXI attributes only. Panics on an invalid
/// [`ConcurrencyConfig`] (use [`ConcurrencyConfig::validate`]).
pub fn concurrency_design(cfg: ConcurrencyConfig) -> ConcurrencyDesign {
    cfg.validate()
        .unwrap_or_else(|e| panic!("invalid bw_probe concurrency config: {e}"));
    ConcurrencyDesign {
        pdi: crate::pdi::build(&build_concurrency_cdo(cfg).to_words()),
        insts: build_concurrency_txn(cfg).to_bytes(),
        args: vec![
            ArgSpec {
                bytes: cfg.read_total(),
                kind: ArgKind::In,
            },
            ArgSpec {
                bytes: cfg.write_total(),
                kind: ArgKind::Out,
            },
        ],
        cfg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Op {
        Write(u32, u32),
        Block(u32, usize),
        Mask(u32, u32, u32),
        Sync {
            col: u32,
            dir: u32,
            chan: u32,
            ncol: u32,
        },
        Patch {
            addr: u32,
            arg: u32,
            plus: u64,
        },
    }

    fn txn_ops(bytes: &[u8]) -> Vec<Op> {
        let w: Vec<u32> = bytes
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let (count, mut i) = (w[2], 4);
        let mut ops = Vec::new();
        for _ in 0..count {
            let (op, len) = match w[i] {
                0 => (Op::Write(w[i + 2], w[i + 4]), 6),
                1 => (
                    Op::Block(w[i + 2], (w[i + 3] as usize - 16) / 4),
                    (w[i + 3] / 4) as usize,
                ),
                3 => (Op::Mask(w[i + 2], w[i + 4], w[i + 5]), 7),
                0x80 => (
                    Op::Sync {
                        col: w[i + 2] >> 16,
                        dir: w[i + 2] & 0xff,
                        chan: w[i + 3] >> 24,
                        ncol: (w[i + 3] >> 16) & 0xff,
                    },
                    4,
                ),
                0x81 => (
                    Op::Patch {
                        addr: w[i + 6],
                        arg: w[i + 8],
                        plus: w[i + 10] as u64 | (w[i + 11] as u64) << 32,
                    },
                    12,
                ),
                other => panic!("unknown txn op {other:#x}"),
            };
            ops.push(op);
            i += len;
        }
        assert_eq!(i, w.len());
        ops
    }

    fn cfg(cols: usize, read_ch: usize, write_ch: usize, bytes: usize) -> BwConfig {
        BwConfig {
            cols,
            read_ch,
            write_ch,
            bytes,
            axi: ShimAxi::default(),
        }
    }

    #[test]
    fn validate_rejects_bad_geometry() {
        for bad in [
            cfg(0, 1, 0, 4096),
            cfg(9, 1, 0, 4096),
            cfg(1, 3, 0, 4096),
            cfg(1, 0, 3, 4096),
            cfg(1, 0, 0, 4096),
            cfg(1, 1, 0, 0),
            cfg(1, 1, 0, 4000),
            cfg(1, 1, 0, MAX_BYTES + 4096),
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
        cfg(8, 2, 2, MAX_BYTES).validate().unwrap();
    }

    #[test]
    fn validate_checks_axi_field_ranges_independently_of_geometry() {
        let axi = |burst, cache, qos| BwConfig {
            axi: ShimAxi { burst, cache, qos },
            ..cfg(2, 1, 1, 8192)
        };
        for ok in [axi(0, 2, 0), axi(3, 2, 0), axi(2, 2, 0)] {
            ok.validate().unwrap_or_else(|e| panic!("{ok:?}: {e}"));
        }
        for (bad, field) in [
            (axi(4, 2, 0), "burst"),
            (axi(3, 16, 0), "AxCACHE"),
            (axi(3, 2, 16), "AxQoS"),
        ] {
            let e = bad.validate().unwrap_err();
            assert!(e.contains(field), "{bad:?}: {e}");
        }
        // Geometry errors still reported for an otherwise valid AXI setting.
        assert!(BwConfig {
            cols: 0,
            ..axi(3, 2, 0)
        }
        .validate()
        .unwrap_err()
        .contains("cols"));
    }

    #[test]
    fn non_vendor_word5_requires_explicit_hazardous_probe() {
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
            let c = BwConfig {
                axi,
                ..cfg(1, 0, 1, 4096)
            };
            assert!(c.validate().is_err());
            assert!(std::panic::catch_unwind(|| design(c)).is_err());
            assert!(c.validate_for_hazardous_probe().is_ok());
        }
    }

    #[test]
    fn every_switch_master_is_driven_once_and_lanes_end_in_memtile_dma() {
        let c = cfg(8, 2, 2, 4096);
        for col in 0..8 {
            let mut driven: HashSet<(u32, u32, String)> = HashSet::new();
            let all = circuits(c, col);
            for ci in &all {
                let w = ci.writes(); // panics on an absent port
                assert!(w[0].1 >> 31 == 1);
                assert!(
                    driven.insert((ci.tile.col, ci.tile.row, format!("{:?}", ci.master))),
                    "master driven twice {ci:?}"
                );
            }
            let by_master: HashMap<_, _> = all
                .iter()
                .map(|ci| ((ci.tile.row, format!("{:?}", ci.master)), ci.slave))
                .collect();
            for k in 0..2u8 {
                // read: shim MM2S k -> North(k) -> memtile South(k) -> S2MM 4+k
                let mm = ShimDma {
                    tile: shim_loc(col),
                    direction: Mm2s,
                    channel: k,
                }
                .port();
                assert_eq!(by_master[&(0, format!("{:?}", Port::North(k)))], mm);
                assert_eq!(
                    by_master[&(1, format!("{:?}", Port::Dma(4 + k)))],
                    Port::South(k)
                );
                // write: memtile MM2S k -> South(k) -> shim North(k) -> S2MM k
                let sm = ShimDma {
                    tile: shim_loc(col),
                    direction: S2mm,
                    channel: k,
                }
                .port();
                assert_eq!(
                    by_master[&(1, format!("{:?}", Port::South(k)))],
                    Port::Dma(k)
                );
                assert_eq!(by_master[&(0, format!("{:?}", sm))], Port::North(k));
            }
        }
    }

    #[test]
    fn memtile_bds_follow_bank_rule_and_do_not_collide() {
        let c = cfg(1, 2, 2, 4096);
        let bds = memtile_descriptors(c);
        let ids: Vec<u32> = bds.iter().map(|b| b.0).collect();
        assert_eq!(ids.iter().collect::<HashSet<_>>().len(), ids.len());
        assert!(ids
            .iter()
            .chain([write_rem_bd(0), write_rem_bd(1)].iter())
            .all(|&i| i < 48));
        assert!(!ids.contains(&write_rem_bd(0)) && !ids.contains(&write_rem_bd(1)));
        // sinks are cyclic (next = itself); chains end without next; banks match the channel parity.
        for (k, &id) in SINK_BD.iter().enumerate() {
            let bd = bds.iter().find(|b| b.0 == id).unwrap().1;
            assert_eq!(bd.next, Some(id));
            assert_eq!(id >= 24, (READ_S2MM0 as usize + k) % 2 == 1);
        }
        for k in 0..2 {
            assert_eq!(write_rem_bd(k) >= 24, k % 2 == 1);
            assert_eq!(
                bds.iter()
                    .find(|b| b.0 == write_chain_bd(k, CHAIN - 1))
                    .unwrap()
                    .1
                    .next,
                None
            );
            for i in 0..CHAIN {
                assert_eq!(write_chain_bd(k, i) >= 24, k % 2 == 1);
            }
        }
        // Task::write asserts the bank rule for every queued task.
        build_txn(cfg(2, 2, 2, 5 * BUF_BYTES + 4096));
    }

    #[test]
    fn write_tasks_emit_exactly_bytes() {
        for bytes in [
            4096,
            16384,
            20480,
            16 * 16384,
            16 * 16384 + 4096,
            19 * 16384 + 8192,
            100 * 16384 + 12288,
            MAX_BYTES,
        ] {
            let t = write_tasks(0, bytes);
            assert!(t.len() <= 3);
            let mut total = 0usize;
            for &(bd, repeat) in &t {
                assert!((1..=256).contains(&repeat));
                let len = if bd == write_rem_bd(0) {
                    bytes % BUF_BYTES
                } else {
                    (CHAIN - (bd - write_chain_bd(0, 0)) as usize) * BUF_BYTES
                };
                total += len * repeat as usize;
            }
            assert_eq!(total, bytes, "{bytes}");
        }
    }

    #[test]
    fn txn_patches_segments_and_syncs_every_active_channel() {
        let d = design(cfg(3, 2, 1, 32768 + 4096));
        let ops = txn_ops(&d.insts);
        let patches: Vec<_> = ops
            .iter()
            .filter_map(|o| {
                if let Op::Patch { addr, arg, plus } = o {
                    Some((*addr, *arg, *plus))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(patches.len(), 3 * 3);
        for col in 0..3u32 {
            for k in 0..2u32 {
                let addr = Bd::address(shim_loc(col), SHIM_READ_BD + k) + 4;
                assert!(
                    patches.contains(&(addr, 0, ((col * 2 + k) as u64) * (32768 + 4096))),
                    "read col {col} ch {k}"
                );
            }
            let addr = Bd::address(shim_loc(col), SHIM_WRITE_BD) + 4;
            assert!(
                patches.contains(&(addr, 1, col as u64 * (32768 + 4096))),
                "write col {col}"
            );
        }
        let syncs: Vec<_> = ops
            .iter()
            .filter_map(|o| {
                if let Op::Sync {
                    col,
                    dir,
                    chan,
                    ncol,
                } = o
                {
                    Some((*col, *dir, *chan, *ncol))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(syncs, vec![(0, 1, 0, 3), (0, 1, 1, 3), (0, 0, 0, 3)]);
        assert!(matches!(ops.last(), Some(Op::Sync { .. })));
        assert_eq!(
            d.args,
            vec![
                ArgSpec {
                    bytes: 3 * 2 * 36864,
                    kind: ArgKind::In
                },
                ArgSpec {
                    bytes: 3 * 36864,
                    kind: ArgKind::Out
                }
            ]
        );
        // Reset is asserted on every used memtile channel, then released, before anything is queued.
        let first_queue = ops.iter().position(|o| matches!(o, Op::Write(..))).unwrap();
        let resets = ops[..first_queue]
            .iter()
            .filter(|o| matches!(o, Op::Mask(_, 2, 2)))
            .count();
        let releases = ops[..first_queue]
            .iter()
            .filter(|o| matches!(o, Op::Mask(_, 0, 2)))
            .count();
        assert_eq!((resets, releases), (3 * 3, 3 * 3));
    }

    #[test]
    fn pdi_pattern_and_unused_directions() {
        // Read-only: no pattern or source BDs; write-only: no sink BD.
        let r = design(cfg(1, 1, 0, 4096));
        let w = design(cfg(1, 0, 1, 4096));
        assert_eq!(r.args[1].bytes, 0);
        assert_eq!(w.args[0].bytes, 0);
        let cdo = crate::pdi::extract_cdo(&w.pdi).unwrap();
        let cmds = crate::cdo::parse(&cdo).unwrap();
        let want: Vec<u32> = (0..4096).map(|i| pattern_word(0, 0, i)).collect();
        let found = cmds.iter().any(|c| matches!(c, crate::cdo::Cmd::DmaWrite(a, d) if *a == mem_loc(0).address(source_off(0)) as u64 && *d == want));
        assert!(found, "pattern DMA_WRITE missing");
        assert_eq!(want[0], 0xB000_0000);
        assert_eq!(pattern_word(2, 1, 5), 0xB000_0000 | 2 << 20 | 1 << 16 | 5);
        assert!(r.pdi.len() < w.pdi.len());
    }

    #[test]
    fn check_write_reports_first_mismatch() {
        let d = design(cfg(2, 0, 2, 20480));
        let mut buf = vec![0u8; d.cfg.write_total()];
        for col in 0..2 {
            for ch in 0..2 {
                let p = d.pattern(col, ch);
                let seg = d.write_segment(col, ch);
                for (i, b) in buf[seg].iter_mut().enumerate() {
                    *b = p[i % BUF_BYTES];
                }
            }
        }
        assert_eq!(
            d.check_write(&buf),
            WriteCheck {
                bad_words: 0,
                first_bad: None
            }
        );
        let at = d.write_segment(1, 1).start + BUF_BYTES + 8;
        buf[at] ^= 1;
        let got = d.check_write(&buf);
        assert_eq!(got.bad_words, 1);
        assert_eq!(got.first_bad.unwrap().0, at & !3);
    }
}
