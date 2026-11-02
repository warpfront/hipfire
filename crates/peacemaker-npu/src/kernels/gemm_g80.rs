// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//! G80 whole-array deployment ([`Variant::G80`](super::gemm_array::Variant::G80), [`design_g80`]): the vertical-pair A-sharing
//! V8 structure on the 128x80 core of [`gemm_core_g80`]. Nothing here is a hardware claim: every statement below is
//! what the encoders in this file emit; behaviour on silicon is not established.
//!
//! # Problem shape and wave
//! `N = 1280` exactly, `K = 64 * kc` with `kc` in `1..=40` (`K <= 2560`), `M >= 1` padded with zero rows to `256 * mw`;
//! `waves = mw` (there is one N wave) and `waves <= 256`. Core `(col c, core row r)` of wave `w` computes the 128x80
//! output tile with M block `r / 2` (rows `w*256 + (r/2)*128 ..+128`) and N tile `2c + r % 2` (columns
//! `(2c + r%2)*80 ..+80`). The two cores of a vertical pair (core rows `2q`, `2q+1`) share one 128-row A block through
//! neighbour data memory (`PairRole::Lower` row `2q` receives the even-`mb` half, `Upper` row `2q+1` the odd half; the
//! peer lock protocol and reverse int8 epilogue are [`gemm_core_g80`]'s). Output is int8, two-stripe I32, or integer
//! direct BF16. `Control` may be Fast or Slow; legacy paths also accept layout0 timing probes.
//! The legacy layouts below remain byte-identical. BF16 sidecars, paired whole-tile C staging and byte totals are
//! documented in `docs/npu/npu-trunk.md`; [`pack_a_bf16`] / [`pack_b_bf16`] supply the required scale metadata.
//!
//! # Legacy int8/I32 host arguments (byte counts, `waves = W`, `kc`)
//! * arg0 = A, **4 segments** of `W*kc*4096` B, segment `c = 0..4` is injected by column `c`. Inside a segment: wave,
//!   K chunk, one 4 KiB E/O half chunk (8 `mb_local` x 8 `kb` 8x8 blocks, [`Geometry::pack_a_half_chunk`]) of rows
//!   `w*256 + q*128 ..` with `q = c/2`, half `h = c%2`. A total is `4*W*kc*4096`. Columns 4..7 read no A.
//! * arg1 = B, **8 column segments** of `kc*10240` B (total `8*kc*10240`), chunk-major, per chunk `[even N tile 5120 B,
//!   odd N tile 5120 B]` = N tiles `2c`, `2c+1`; each 5120 B tile is `(kb 0..8, nb 0..10)` 8x8 blocks. B is read from DDR
//!   exactly once per submit, whatever `mw` is.
//! * arg2 = C, **8 column segments** of `W*4*cb` B (total `8*W*4*cb`; `cb` = 10240 int8, 40960 I32): the S2MM0 stream (core
//!   rows 0, 1) followed by the S2MM1 stream (core rows 2, 3), each `(wave, rho % 2)` tiles of `cb` B in `(mb 0..16, nb 0..10)`
//!   8x8 blocks ([`Geometry::pair_c_offset`]): int8 values, or for `Epilogue::I32` little-endian i32 (the core's raw `C`; the
//!   tile is its two 20480 B stripes back to back: `mb 0..8` then `mb 8..16`). Host unpack ([`unpack_c`]) is row-major with the
//!   M padding discarded. For `K = 2560, M = 4096`: A 10 MiB, B 3.125 MiB, C 5 MiB int8 / 20 MiB I32.
//!
//! # Memtile map (own-view `MEM_BASE + off`, lock ids `MEM_LOCK + id`, per column)
//! | region | half-open offsets |
//! |---|---|
//! | **reserved for the lean ring scratch (X `0x1000`, Y `0x3000`) and spare BD queues, never accessed by G80** | `[0, 0x4000)` |
//! | C slot `s` row `r` (int8, 10240 B) | `0x4000 + s*0xa000 + r*0x2800`, both slots end `0x18000` |
//! | C stage row `r` (**I32**, one 20480 B slot per row, both stripes) | `0x4000 + r*0x5000`, row 3 ends `0x18000` |
//! | A ping / pong (4096 B live of an 8 KiB reservation each; columns 0..3 only) | `[0x18000,0x1a000)` / `[0x1a000,0x1c000)` |
//! | B parity 0 (`kc*5120` B) | `[0x1c000, 0x1c000 + kc*5120)` |
//! | B parity 1 (`kc*5120` B) | `[0x1c000 + kc*5120, 0x1c000 + 2*kc*5120)`, ends `<= 0x80000` (`= 0x80000` at `kc = 40`) |
//!
//! Parity `p` of B is the N-tile parity, i.e. the core-row parity that consumes it (rows 0, 2 take parity 0, rows 1, 3
//! parity 1). Every B chunk of the whole K stays resident for the whole submit; the B regions are single buffered.
//!
//! # Memtile BDs (26 BDs, ids < 48; even channel => id < 24, odd channel => id >= 24) and locks (22 active)
//! Channels: S2MM0..5 and MM2S0, 1, 2, 3, 5. BD ids 10 and 11 and shim BDs 4, 5, 6 are never touched (the lean ring's
//! spare Q / DQ / P / SENT / D descriptors).
//! | function | channel | BD ids | length | locks (memtile-local ids) |
//! |---|---|---|---|---|
//! | C producer row 0 / 1 / 2 / 3 (int8 only) | S2MM0 / 1 / 2 / 3 | `0<->1` / `24<->25` / `2<->3` / `26<->27` (slot 0 / 1, cyclic) | 2560 words | acquire `c_lock(slot,row)` (empty) -1, release `+1` (full); `c_lock = 8 + slot*8 + row*2` |
//! | B fill | S2MM4 | `4 -> 5 -> end` (parity 0 / 1), **finite, task `repeat = kc`** | 1280 words each | BD 4 releases ready0 (`4`) +1, BD 5 releases ready1 (`6`) +1, no acquire |
//! | A fill (cols 0..3) | S2MM5 | `28<->29` (ping / pong, cyclic) | 1024 words | acquire `A_EMPTY[s]` -1, release `A_FULL[s]` +1 |
//! | C drain rows 0, 1 | MM2S0 | `6->7->8->9->6`: slot 0 row 0, slot 0 row 1, slot 1 row 0, slot 1 row 1 (cyclic; int8 only, see the I32 rows below) | 2560 words | acquire `c_lock+1` (full) -1, release `c_lock` +1 |
//! | C drain rows 2, 3 | MM2S3 | `32->33->34->35->32` (same pattern, rows 2, 3) | 2560 words | as above |
//! | **I32** C producer row `r` | S2MM `r` | `0 -> 1` / `24 -> 25` / `2 -> 3` / `26 -> 27` (stripe 0 -> stripe 1), **finite, task `repeat = mw`**, second BD has no `next` | 5120 words each | both BDs: acquire `c_lock(0,r)` (empty) -1, release `+1` (full); `c_lock(0, r) = 8 + 2r`; both write the **same** stage address |
//! | **I32** C drain rows 0, 1 / rows 2, 3 | MM2S0 / MM2S3 | `6->7->8->9` / `32->33->34->35`: row 0 stripe 0, row 0 stripe 1, row 1 stripe 0, row 1 stripe 1 (rows 2, 3), **finite, task `repeat = mw`**, last BD has no `next` | 5120 words each | acquire `c_lock+1` (full) -1, release `c_lock` (empty) +1 of its row; each stripe is staged and locked independently |
//! | B parity 0 guard / whole | MM2S1 | `30` finite (`repeat = kc`), then `31` finite (`repeat = mw - 1`, omitted when `mw = 1`) | 1280 / `kc*1280` words | BD 30 acquires ready0 (`4`) -1; BD 31 no lock |
//! | B parity 1 guard / whole | MM2S5 | `42` finite (`repeat = kc`), then `43` finite (`repeat = mw - 1`, omitted when `mw = 1`) | 1280 / `kc*1280` words | BD 42 acquires ready1 (`6`) -1; BD 43 no lock |
//! | A replay (cols 0..3) | MM2S2 | `14<->15` (cyclic) | 1024 words | acquire `A_FULL[s]` -1, release `A_EMPTY[s]` +1 |
//!
//! Columns 4..7 have no A rows: 22 BDs and 18 locks there; columns 0..3 have the 26 BDs and 22 locks. Lock ids used:
//! `0..3` (A), `4`, `6` (B ready), `8..23` (C int8) / `8..15` (C I32). Initial values: every `*_EMPTY` 1, every `*_FULL` 0,
//! ready0 = ready1 = 0.
//! A ready lock never exceeds `kc <= 40 < 63`.
//!
//! ## B fill, guarded read, whole read (iteration contract)
//! Fill BD 4 / 5: base `0x1c000` / `0x1c000 + kc*5120`, 1280 words, iteration `step = 1280` words, `wrap = kc`,
//! `current = 0`; BD 4 links to BD 5, BD 5 has no `next`; the queued task repeats the chain `kc` times, so each BD is
//! loaded exactly `kc` times and its iteration `current` returns to 0 with the last load. Guard BD 30 / 42: same base,
//! length, iteration (`step = 1280, wrap = kc, current = 0`) and no `next`, task `repeat = kc`. Whole BD 31 / 43: same
//! base, `kc*1280` words, **default** iteration (`step 1, wrap 1`), no lock, no `next`, task `repeat = mw - 1`, queued
//! on the same channel strictly after the guard task (ordered queue, depth 2 of the memtile limit 4; `repeat <= 256`).
//! Reader chunk `j` therefore starts only after fill chunk `j` of its own parity landed
//! (`ready_p = fills_p - guarded reads_p`), and the unguarded whole pass starts only after the guard task consumed
//! all `kc` credits, so B is immutable afterwards and needs no lock. The two parity readers are independent queues.
//! All plain 1-D transfers, no dimension wrap (so a chunk of parity 1 can be read before all parity 0 chunks landed).
//!
//! # Completed-submit end state and lean repair recipe
//! Contract for a lean/persistent Railgun integration. "Start" is the canonical start state `S0` written by the full
//! dynamic TXN (`ArrayDesign::insts`): cores reset, every listed channel reset and queued at its first BD, **all G80
//! memtile BDs rewritten after the channel reset and before any queue write**, every lock at its initial value, shim
//! tokens armed. With `J = kc * waves` chunk transfers and `W = waves` tiles per column the end state of a submit whose C
//! token SYNCs (both S2MM channels, all 8 columns) returned is:
//! * **B fill S2MM4** (BD 4, 5, task `repeat = kc`): finite, idle, queue empty, **never parked**. `current` of BD 4 and BD 5
//!   returned to 0 (`kc` loads, `wrap = kc`). ready0 = ready1 = `kc - kc = 0`. There is no lookahead credit (finite tasks load
//!   no BD after the last repeat). Requeue is mandatory for every submit: `Task{S2mm, 4, BD 4, repeat = kc}`.
//! * **B readers MM2S1 / MM2S5**: guard task (`repeat = kc`) and whole task (`repeat = mw-1`) both finished, idle, queue
//!   empty; `current` of BD 30 / 42 is 0 again; BD 31 / 43 carry no iteration or lock. Requeue is mandatory for every submit,
//!   in this order on each channel: guard `repeat = kc`, then whole `repeat = mw - 1` only if `mw > 1`.
//! * **A fill S2MM5 {28,29} / A replay MM2S2 {14,15}** (cols 0..3): `J` loads each, parked at BD `28 + J%2` /
//!   `14 + J%2`; `A_FULL = 0`; the S2MM parked BD holds the lookahead acquire (`A_EMPTY[J%2] = 0`, the other slot's empty 1).
//!   Equivalent to S0 iff `J` is even. Minimal lean repair for odd `J` (slot 1): reset (CTRL bit1 pulse) both channels, restore only the
//!   dropped lookahead credit `A_EMPTY[1] = 1` (the canonical full reset writes every A lock to its initial value), requeue BD 28 / 14.
//! * **C producers S2MM0..3 (int8)** (`{0,1}`, `{24,25}`, `{2,3}`, `{26,27}`) : `W` loads, parked at slot `W%2` with its empty-lock
//!   lookahead taken; **C drains MM2S0 / MM2S3**: `2W` loads, parked at BD `6 + 2*(W%2)` / `32 + 2*(W%2)`, full locks unacquired.
//!   Equivalent to S0 iff `W` is even. Minimal lean repair for odd `W` (parked at slot 1): reset all six channels (S2MM0..3, MM2S0,
//!   MM2S3), restore only the dropped slot-1 `EMPTY` lock of each row (`c_lock(1, r) = 1`), requeue BD `C_S2MM_BD[r][0]`, 6, 32.
//!   C rings use no iteration, so there is no `current` to repair.
//! * **`Epilogue::I32` C** (replaces the two items above for I32): producer S2MM `r` runs `W` passes of the two-BD chain, each
//!   drain channel `W` passes of its four-BD chain; all finite, so after the last repeat they are idle, empty-queued, **not
//!   parked**, with no lookahead credit. `c_lock(0, r)` is back at empty 1 / full 0 for any `W` parity (acquire / release
//!   pairs balance), the BDs carry no iteration. A lean TXN therefore needs **no** C parity repair, but must requeue
//!   `S2MM r BD C_S2MM_BD[r][0] repeat = W`, `MM2S0 BD 6 repeat = W`, `MM2S3 BD 32 repeat = W` on every submit.
//! * **Cores**: `DONE`; always halt+reset, `CORE_PC = 0`, enable last, as the full TXN. Canonical full reset reloads every core lock
//!   (`A/B_EMPTY = 1`, `C_EMPTY = 1`, rest 0); the minimal lean repair reloads only what the end state lost: the dropped pong
//!   `A_EMPTY[1]` / `B_EMPTY[1]` lookahead credit when `J` is odd (`C_EMPTY` stays 1, FULL and PEER locks are 0). Core S2MM0 (A)
//!   {0,1} / S2MM1 (B) {2,3} are parked at BD `J%2` (reset and requeue BD 0 / 2 when `J` is odd); core MM2S0 C (BD 4 self-cyclic)
//!   parks at BD 4 with `C_FULL` unacquired for any `W`.
//! * **Shim**: finite BDs/tasks consumed (shim channels are never reset); BDs are rewritten with new DDR patches each submit.
//! * **Streams**: no residue after the C tokens: the cores consume exactly `kc*W` A and B chunks, A/B shim BDs have drained,
//!   the C streams carry exactly `W*2*cb` B per channel (`cb = 10240` int8, `40960` I32); every switch lane is empty.
//!
//! A lean TXN is valid **only** directly after a *completed* submit (both aggregate C SYNCs returned). After an interrupted or
//! aborted run the state above does not hold and only the canonical full reset is valid: full channel resets, a rewrite of
//! every G80 memtile BD (clearing the `current` of BD 4, 5, 30, 42, which a channel reset does not clear), and initialisation of
//! every lock. After a completed submit a lean TXN, for every column, requeues the finite B fill / guard / whole tasks (above,
//! `repeat = kc` for fill and guard, `mw - 1` for whole) and applies only the parity repairs listed (`J` odd: A / core rings; `W`
//! odd: C rings / drains). The ready locks 4 and 6 end at 0 and the completed finite fill holds **no** lookahead credit, so no lock
//! write is needed for them: a forced reset of S2MM4 (POLL) or MM2S0 (DONE) by the lean ring does not touch locks 4 / 6 and needs no
//! lock repair, only the requeue of BD 4 with `repeat = kc` (retain `kc`) and of all B tasks, and BD 6 for MM2S0 (int8: valid only
//! while the drain chain is parked at BD 6, i.e. `W` even; for odd `W` the parity plan above already resets and requeues it; I32:
//! the finite drain task `repeat = W` is requeued on every submit anyway and is kept as is). Ring
//! scratch X / Y are never touched (`[0, 0x4000)`).
//! A submit must start only after the previous submit's two aggregate C SYNCs returned (quiescent array) and the next
//! expert's B is not overlapped with the previous submit: the B regions are single buffered per submit.
//!
//! # Shim (DDR patch points, built by `gemm_array`)
//! Per column: BD 0 B on MM2S0 (`kc*2560` words, arg1 + `col*kc*10240`), BD 1 C stream 0 on S2MM0 (`W*5120` words int8 /
//! `W*20480` words I32, arg2 + `col*W*4*cb`, token), BD 2 A on MM2S1 (columns 0..3 only, `W*kc*1024` words, arg0 +
//! `col*W*kc*4096`), BD 3 C stream 1 on S2MM1 (same words, arg2 + `col*W*4*cb + W*2*cb`, token), `cb` the tile bytes (10240 int8,
//! 40960 I32). Each BD's address word (`Bd::address(shim, id) + 4`) is DDR-patched with `column_offset (+ arena_base[arg])` in
//! the grouped launch.
//!
//! # Switch circuits (column `c`, lanes preserved, no master has two slaves)
//! * shim: MM2S0 -> North0 (B), MM2S1 -> North1 (A, columns 0..3), North0 -> S2MM0 and North1 -> S2MM1 (C).
//! * memtile: South0 -> Dma4 (B fill); South1 -> Dma5 and Dma2 -> North5 (A, columns 0..3); Dma1 -> North4 (parity 0 B);
//!   Dma5 -> North3 (parity 1 B); North(r) -> Dma(r) (C of core row r, r = 0..3); Dma0 -> South0 (rows 0, 1 drain); Dma3 -> South1
//!   (rows 2, 3 drain).
//! * core row `r`: C `Dma0 -> South(r)` and `North(s) -> South(s)` for `s > r`. B parity 0 on lane 4: `South4 -> Dma1` on rows 0, 2,
//!   `South4 -> North4` on rows 0, 1. B parity 1 on lane 3: `South3 -> Dma1` on rows 1, 3, `South3 -> North3` on rows 0, 1, 2.
//!   A of row `r` is injected by column `r` (columns 0..3) on North5, passed `South5 -> North5` by rows below `r`, selected at row `r`
//!   as `South5` (column `r`), `East0` (columns `< r`) or `West0` (columns `> r`), driving `Dma0`, `West0` (`0 < col <= r`) and
//!   `East0` (`col < 7`, `col >= r`), so every column of the row receives the same half.
//!
//! # Estimate
//! [`estimate`] is a model, never a measurement; see its documentation. It uses the explicit 128x80 geometry and the
//! G80 core's issue cycles, not any TN64 constant. For `Epilogue::I32` it adds the serial single-slot stripe hand-off chain
//! (six 5120-cycle stripe phases exposed per wave before the core regains `C_EMPTY`, no overlap credit) against the int8
//! `2560`-cycle `Cout` drain plus `663`-cycle epilogue.
use super::gemm_array::{
    self as array, ArrayDesign, CycleEstimate, Geometry, Kind, Topo, Variant, MAX_WAVES,
};
use super::gemm_core::{self, Control, CoreVariant, Epilogue, PairRole};
use super::gemm_core_g80 as g80_core;
use super::gemm_i8::{peak_ops_per_second, ArgKind, ArgSpec};
use super::iu4_ief15_core::g80_bf16 as bf16;
use crate::{
    dma::{
        Bd, BdLocks,
        Direction::{Mm2s, S2mm},
        Iteration, Location, Task,
    },
    route::{Circuit, Port, ShimDma},
};

const TM: usize = g80_core::TM;
const TN: usize = g80_core::TN;
const CHUNK_K: usize = g80_core::CHUNK_K;
const A_HALF_BYTES: usize = g80_core::A_HALF_BYTES;
const B_BYTES: usize = g80_core::B_BYTES;
const COUT_BYTES: usize = g80_core::COUT_BYTES;
const MB: usize = TM / 8;
const NB: usize = TN / 8;
const KB: usize = CHUNK_K / 8;
const ROWS: usize = array::ROWS;
const COLS: usize = array::COLS;
/// Columns that inject A (core row `c`'s half is injected by column `c`).
const A_COLS: usize = array::G80_A_COLS;
/// Rows / columns of one wave: two 128-row M tiles x sixteen 80-column N tiles.
pub const WAVE_M: usize = array::G80_WAVE_M;
pub const WAVE_N: usize = array::G80_WAVE_N;
const _: () = assert!(
    WAVE_M == 256 && WAVE_N == 1280 && A_COLS == 4 && A_HALF_BYTES == 4096 && B_BYTES == 5120
);
const _: () =
    assert!(COUT_BYTES == 10240 && MB == 16 && NB == 10 && KB == 8 && ROWS == 4 && COLS == 8);

/// Host BF16 destination. Strides and starting columns must be even because
/// AIE2P DMA addresses pairs of BF16 elements as 32-bit words.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Bf16Layout {
    #[default]
    NativeTiles,
    RowMajor {
        pitch_elems: usize,
        col0_elems: usize,
    },
}

impl Bf16Layout {
    fn validate(self, padded_rows: usize) -> Result<(), String> {
        self.validate_for(padded_rows, WAVE_N)
    }

    /// `validate` for a wave `wave_n` columns wide (1280 for G80, 512 for the N512 design).
    pub(crate) fn validate_for(self, padded_rows: usize, wave_n: usize) -> Result<(), String> {
        if let Self::RowMajor {
            pitch_elems: pitch,
            col0_elems: col0,
        } = self
        {
            if pitch % 2 != 0
                || col0 % 2 != 0
                || pitch / 2 > 1 << 20
                || col0.checked_add(wave_n).is_none_or(|end| end > pitch)
                || padded_rows
                    .checked_mul(pitch)
                    .and_then(|n| n.checked_mul(2))
                    .is_none()
            {
                return Err(format!("BF16 row-major needs even pitch/col0, col0+{wave_n} <= pitch <= {}, and a representable arena", 2 << 20));
            }
        }
        Ok(())
    }
}

// Memtile map (byte offsets, see the module header).
const C_OFF: u32 = 0x4000;
const C_SLOT_STRIDE: u32 = 0xa000;
const C_ROW_STRIDE: u32 = 0x2800;
const A_OFF: [u32; 2] = [0x18000, 0x1a000];
const B_OFF: u32 = 0x1c000;
const MEM_END: u32 = 0x80000;
const _: () = assert!(C_ROW_STRIDE as usize == COUT_BYTES && C_SLOT_STRIDE == 4 * C_ROW_STRIDE);
const _: () = assert!(
    C_OFF + 2 * C_SLOT_STRIDE == A_OFF[0]
        && A_OFF[0] + 0x2000 == A_OFF[1]
        && A_OFF[1] + 0x2000 == B_OFF
);
const _: () = assert!(B_OFF as usize + 2 * g80_core::MAX_KC * B_BYTES == MEM_END as usize);
const _: () = assert!(A_HALF_BYTES <= 0x2000);

// BD ids and channels (banks: even channel < 24, odd >= 24).
pub(crate) const C_S2MM_BD: [[u32; 2]; ROWS] = [[0, 1], [24, 25], [2, 3], [26, 27]];
pub(crate) const C_DRAIN_CH: [u32; 2] = [0, 3];
pub(crate) const C_DRAIN_BD0: [u32; 2] = [6, 32];
pub(crate) const B_FILL_CH: u32 = 4;
pub(crate) const B_FILL_BD: [u32; 2] = [4, 5];
/// B readers per parity: MM2S1 (parity 0) and MM2S5 (parity 1).
pub(crate) const G80_B_MM2S_CH: [u32; 2] = [1, 5];
/// Finite guarded B readers (`repeat = kc`, iteration `step 1280, wrap kc`) and whole-stream readers (`repeat = mw - 1`).
pub(crate) const G80_B_GUARD_BD: [u32; 2] = [30, 42];
pub(crate) const G80_B_WHOLE_BD: [u32; 2] = [31, 43];
const B_READ_CH: [u32; 2] = G80_B_MM2S_CH;
const B_GUARD_BD: [u32; 2] = G80_B_GUARD_BD;
const B_WHOLE_BD: [u32; 2] = G80_B_WHOLE_BD;
/// B ready locks per parity (`gemm_core::B_EMPTY` ids 4 and 6; the `B_FULL` ids 5 and 7 are unused).
pub(crate) const B_READY: [u32; 2] = [gemm_core::B_EMPTY[0], gemm_core::B_EMPTY[1]];
pub(crate) const A_S2MM_CH: u32 = 5;
pub(crate) const A_S2MM_BD: [u32; 2] = [28, 29];
pub(crate) const A_MM2S_CH: u32 = 2;
pub(crate) const A_MM2S_BD: [u32; 2] = [14, 15];
pub(crate) const C_LOCK0: u32 = 8;
const _: () = assert!(B_READY[0] == 4 && B_READY[1] == 6 && g80_core::MAX_KC < 63);

pub(crate) fn c_lock(slot: usize, row: usize) -> u32 {
    C_LOCK0 + (slot * 8 + row * 2) as u32
}
fn c_off(slot: usize, row: usize) -> u32 {
    C_OFF + slot as u32 * C_SLOT_STRIDE + row as u32 * C_ROW_STRIDE
}
/// `Epilogue::I32` stage of row `r`: one 20480 B slot (`g80_core::C_STRIPE_BYTES`), rows contiguous from `0x4000`, ending `0x18000`.
fn c_i32_off(row: usize) -> u32 {
    C_OFF + row as u32 * g80_core::C_STRIPE_BYTES as u32
}
/// Serial single-slot stripe phases between `C_FULL` and the `C_EMPTY` return of the last row of a pair (see [`estimate`]).
const I32_HANDOFF_PHASES: u64 = 6;
fn b_off(kc: usize, parity: usize) -> u32 {
    B_OFF + (parity * kc * B_BYTES) as u32
}
fn a_locks() -> ([u32; 2], [u32; 2]) {
    (gemm_core::A_EMPTY, gemm_core::A_FULL)
}

/// `(kc, mw)` of a G80 topology; panics for any other kind.
fn shape(topo: Topo) -> (usize, usize) {
    match topo.kind {
        Kind::G80 { kc, mw, .. } => (kc, mw),
        other => panic!("gemm_g80 needs a G80 topology, got {other:?}"),
    }
}

/// `Epilogue::I32` topology (raw 40,960 B tiles, one 20,480 B memtile stage per row, finite C tasks).
fn is_i32(topo: Topo) -> bool {
    matches!(
        topo.kind,
        Kind::G80 {
            epi: Epilogue::I32,
            ..
        }
    )
}

pub(crate) fn a_chunk_bytes(topo: Topo) -> usize {
    if topo.is_g80_bf16() {
        bf16::A_BYTES
    } else {
        A_HALF_BYTES
    }
}
pub(crate) fn b_chunk_bytes(topo: Topo) -> usize {
    if topo.is_g80_n512() {
        return bf16::NARROW.b_bytes();
    }
    if topo.is_g80_bf16() {
        bf16::B_BYTES
    } else {
        B_BYTES
    }
}
fn a_offsets(topo: Topo) -> [u32; 2] {
    if topo.is_g80_bf16() {
        [0xe000, 0x10000]
    } else {
        A_OFF
    }
}
fn b_offset(topo: Topo, kc: usize, parity: usize) -> u32 {
    if topo.is_g80_bf16() {
        0x12000 + (parity * kc * bf16::B_BYTES) as u32
    } else {
        b_off(kc, parity)
    }
}

fn check_geometry(geo: &Geometry) -> (usize, usize) {
    let (kc, mw) = shape(geo.topo);
    assert!(
        geo.kc == kc
            && geo.mw == mw
            && geo.nw == 1
            && geo.waves == mw
            && geo.n == WAVE_N
            && geo.k == kc * CHUNK_K,
        "inconsistent G80 geometry {geo:?}"
    );
    (kc, mw)
}

/// One memtile BD in a cyclic chain: `idx` of `len`, ids from `first`, acquiring `acq` (-1) and releasing `rel` (+1).
fn chain_bd(
    first: u32,
    idx: usize,
    len: usize,
    addr: u32,
    words: u32,
    acq: u32,
    rel: u32,
) -> (u32, Bd) {
    let mut bd = Bd::new((array::MEM_BASE + addr) as u64, words);
    bd.locks = BdLocks {
        acq: Some((array::MEM_LOCK + acq, -1)),
        rel: Some((array::MEM_LOCK + rel, 1)),
    };
    bd.next = Some(first + ((idx + 1) % len) as u32);
    (first + idx as u32, bd)
}

/// Static stream-switch circuits of one column (see the module header, "Switch circuits").
pub(crate) fn column_circuits(topo: Topo, col: u32) -> Vec<Circuit> {
    if topo.is_g80_n512() {
        return super::gemm_g80_n512::column_circuits(topo, col);
    }
    shape(topo);
    let has_a = topo.has_a(col);
    let mut v = Vec::new();
    let (shim, mem) = (array::shim_loc(col), array::mem_loc(col));
    let sh = |direction, channel| ShimDma {
        tile: shim,
        direction,
        channel,
    };
    v.push(Circuit {
        tile: shim,
        slave: sh(Mm2s, 0).port(),
        master: Port::North(0),
    });
    v.push(Circuit {
        tile: shim,
        slave: Port::North(0),
        master: sh(S2mm, 0).port(),
    });
    v.push(Circuit {
        tile: shim,
        slave: Port::North(1),
        master: sh(S2mm, 1).port(),
    });
    v.push(Circuit {
        tile: mem,
        slave: Port::South(0),
        master: Port::Dma(B_FILL_CH as u8),
    });
    v.push(Circuit {
        tile: mem,
        slave: Port::Dma(B_READ_CH[0] as u8),
        master: Port::North(4),
    });
    v.push(Circuit {
        tile: mem,
        slave: Port::Dma(B_READ_CH[1] as u8),
        master: Port::North(3),
    });
    v.push(Circuit {
        tile: mem,
        slave: Port::Dma(C_DRAIN_CH[0] as u8),
        master: Port::South(0),
    });
    v.push(Circuit {
        tile: mem,
        slave: Port::Dma(C_DRAIN_CH[1] as u8),
        master: Port::South(1),
    });
    for r in 0..ROWS as u8 {
        v.push(Circuit {
            tile: mem,
            slave: Port::North(r),
            master: Port::Dma(r),
        });
    }
    if has_a {
        v.push(Circuit {
            tile: shim,
            slave: sh(Mm2s, 1).port(),
            master: Port::North(1),
        });
        v.push(Circuit {
            tile: mem,
            slave: Port::South(1),
            master: Port::Dma(A_S2MM_CH as u8),
        });
        v.push(Circuit {
            tile: mem,
            slave: Port::Dma(A_MM2S_CH as u8),
            master: Port::North(5),
        });
    }
    let c = col as usize;
    for r in 0..ROWS {
        let tile = array::core_loc(col, r);
        // C down: own tile, then the rows above forward on their own lanes.
        v.push(Circuit {
            tile,
            slave: Port::Dma(0),
            master: Port::South(r as u8),
        });
        for s in r + 1..ROWS {
            v.push(Circuit {
                tile,
                slave: Port::North(s as u8),
                master: Port::South(s as u8),
            });
        }
        // B: parity 0 on lane 4 (rows 0, 2), parity 1 on lane 3 (rows 1, 3); lanes are passed up through the rows between.
        if r % 2 == 0 {
            v.push(Circuit {
                tile,
                slave: Port::South(4),
                master: Port::Dma(1),
            });
        }
        if r < 2 {
            v.push(Circuit {
                tile,
                slave: Port::South(4),
                master: Port::North(4),
            });
        }
        if r % 2 == 1 {
            v.push(Circuit {
                tile,
                slave: Port::South(3),
                master: Port::Dma(1),
            });
        }
        if r < 3 {
            v.push(Circuit {
                tile,
                slave: Port::South(3),
                master: Port::North(3),
            });
        }
        // A of row r is injected by column r and broadcast along the row on horizontal lane 0.
        let slave = match c.cmp(&r) {
            std::cmp::Ordering::Equal => Port::South(5),
            std::cmp::Ordering::Less => Port::East(0),
            std::cmp::Ordering::Greater => Port::West(0),
        };
        v.push(Circuit {
            tile,
            slave,
            master: Port::Dma(0),
        });
        if c > 0 && c <= r {
            v.push(Circuit {
                tile,
                slave,
                master: Port::West(0),
            });
        }
        if c < COLS - 1 && c >= r {
            v.push(Circuit {
                tile,
                slave,
                master: Port::East(0),
            });
        }
        if c < A_COLS && r < c {
            v.push(Circuit {
                tile,
                slave: Port::South(5),
                master: Port::North(5),
            });
        }
    }
    v
}

/// All memtile BDs of a column `(id, bd)`: B fill / guarded / whole readers, C rings and drains, A rings (columns 0..3).
pub(crate) fn memtile_descriptors(topo: Topo, col: u32) -> Vec<(u32, Bd)> {
    if topo.is_g80_n512() {
        return super::gemm_g80_n512::memtile_descriptors(topo, col);
    }
    let (kc, _) = shape(topo);
    let mut v = Vec::new();
    let chunk_words = (b_chunk_bytes(topo) / 4) as u32;
    let iteration = Iteration {
        step: chunk_words,
        wrap: kc as u32,
        current: 0,
    };
    for p in 0..2 {
        let addr = (array::MEM_BASE + b_offset(topo, kc, p)) as u64;
        let mut fill = Bd::new(addr, chunk_words);
        fill.iteration = iteration;
        fill.locks = BdLocks {
            acq: None,
            rel: Some((array::MEM_LOCK + B_READY[p], 1)),
        };
        fill.next = (p == 0).then_some(B_FILL_BD[1]);
        v.push((B_FILL_BD[p], fill));
        let mut guard = Bd::new(addr, chunk_words);
        guard.iteration = iteration;
        guard.locks = BdLocks {
            acq: Some((array::MEM_LOCK + B_READY[p], -1)),
            rel: None,
        };
        v.push((B_GUARD_BD[p], guard));
        v.push((B_WHOLE_BD[p], Bd::new(addr, kc as u32 * chunk_words)));
    }
    if topo.has_a(col) {
        let (empty, full) = a_locks();
        let words = (a_chunk_bytes(topo) / 4) as u32;
        v.extend(array::ring(
            A_S2MM_BD,
            a_offsets(topo),
            words,
            empty,
            full,
            true,
        ));
        v.extend(array::ring(
            A_MM2S_BD,
            a_offsets(topo),
            words,
            empty,
            full,
            false,
        ));
    }
    if topo.is_g80_bf16() {
        // A drain hands the shared whole-tile stage to its pair's other row.
        let words = (bf16::OUT_BYTES / 4) as u32;
        for r in 0..ROWS {
            let addr = array::MEM_BASE + C_OFF + (r / 2 * bf16::OUT_BYTES) as u32;
            let mut producer = Bd::new(addr as u64, words);
            producer.locks = BdLocks {
                acq: Some((array::MEM_LOCK + c_lock(0, r), -1)),
                rel: Some((array::MEM_LOCK + c_lock(0, r) + 1, 1)),
            };
            v.push((C_S2MM_BD[r][0], producer));
            let id = C_DRAIN_BD0[r / 2] + (r % 2) as u32;
            let mut drain = Bd::new(addr as u64, words);
            if topo.bf16_rowmajor().is_some() {
                // Native (mb, nb, i, pair) -> rows (mb, i, nb, pair),
                // retaining the exact same 5120 stream words and shared stage.
                drain.dims = [
                    crate::dma::Dim { step: 1, wrap: 4 },
                    crate::dma::Dim { step: 32, wrap: 10 },
                    crate::dma::Dim { step: 4, wrap: 8 },
                    crate::dma::Dim { step: 320, wrap: 0 },
                ];
            }
            drain.locks = BdLocks {
                acq: Some((array::MEM_LOCK + c_lock(0, r) + 1, -1)),
                rel: Some((array::MEM_LOCK + c_lock(0, r ^ 1), 1)),
            };
            drain.next = (r % 2 == 0).then_some(id + 1);
            v.push((id, drain));
        }
        return v;
    }
    if is_i32(topo) {
        // One 20480 B staging slot per row: the two stripe BDs of the producer and the two of the drain share its address and
        // its empty / full lock pair, so every stripe hand-off is serialised through that one slot. Producers are finite
        // two-BD chains (first -> second, second has no `next`), the drains finite four-BD chains (row 0 / 2 stripe 0, 1,
        // row 1 / 3 stripe 0, 1); each is queued once per wave (`repeat = mw`).
        let stripe_words = (g80_core::C_STRIPE_BYTES / 4) as u32;
        let stripe = |addr: u32, acq: u32, rel: u32, next: Option<u32>| {
            let mut bd = Bd::new((array::MEM_BASE + addr) as u64, stripe_words);
            bd.locks = BdLocks {
                acq: Some((array::MEM_LOCK + acq, -1)),
                rel: Some((array::MEM_LOCK + rel, 1)),
            };
            bd.next = next;
            bd
        };
        for r in 0..ROWS {
            let (empty, full) = (c_lock(0, r), c_lock(0, r) + 1);
            for s in 0..2 {
                v.push((
                    C_S2MM_BD[r][s],
                    stripe(
                        c_i32_off(r),
                        empty,
                        full,
                        (s == 0).then_some(C_S2MM_BD[r][1]),
                    ),
                ));
            }
        }
        for g in 0..2 {
            for idx in 0..4usize {
                let row = 2 * g + idx / 2;
                let (empty, full) = (c_lock(0, row), c_lock(0, row) + 1);
                let id = C_DRAIN_BD0[g] + idx as u32;
                v.push((
                    id,
                    stripe(c_i32_off(row), full, empty, (idx < 3).then_some(id + 1)),
                ));
            }
        }
        return v;
    }
    let words = (COUT_BYTES / 4) as u32;
    for r in 0..ROWS {
        let empty = [c_lock(0, r), c_lock(1, r)];
        v.extend(array::ring(
            C_S2MM_BD[r],
            [c_off(0, r), c_off(1, r)],
            words,
            empty,
            [empty[0] + 1, empty[1] + 1],
            true,
        ));
    }
    for g in 0..2 {
        for slot in 0..2 {
            for rr in 0..2 {
                let row = 2 * g + rr;
                let lock = c_lock(slot, row);
                v.push(chain_bd(
                    C_DRAIN_BD0[g],
                    slot * 2 + rr,
                    4,
                    c_off(slot, row),
                    words,
                    lock + 1,
                    lock,
                ));
            }
        }
    }
    v
}

/// Initial memtile lock values: `*_EMPTY` 1, `*_FULL` 0, B ready 0.
pub(crate) fn memtile_locks(topo: Topo, col: u32) -> Vec<(u32, u32)> {
    if topo.is_g80_n512() {
        return super::gemm_g80_n512::memtile_locks(topo, col);
    }
    shape(topo);
    let mut v = vec![(B_READY[0], 0), (B_READY[1], 0)];
    if topo.has_a(col) {
        let (empty, full) = a_locks();
        for i in 0..2 {
            v.extend([(empty[i], 1), (full[i], 0)]);
        }
    }
    // I32 has one stage per row (locks 8 + 2r empty, 9 + 2r full); int8 two slots (8 + 8s + 2r).
    let c_slots = if is_i32(topo) || topo.is_g80_bf16() {
        1
    } else {
        2
    };
    for slot in 0..c_slots {
        for r in 0..ROWS {
            let empty = u32::from(!topo.is_g80_bf16() || r % 2 == 0);
            v.extend([(c_lock(slot, r), empty), (c_lock(slot, r) + 1, 0)]);
        }
    }
    v
}

/// Core ring tasks plus the memtile tasks of a column (the shim tasks depend on the arguments). Finite B tasks come in queue
/// order: per reader channel the guarded task, then the whole-stream task (omitted for `mw = 1`).
pub(crate) fn ring_tasks(topo: Topo, col: u32) -> Vec<(Location, Task)> {
    if topo.is_g80_n512() {
        return super::gemm_g80_n512::ring_tasks(topo, col);
    }
    let (kc, mw) = shape(topo);
    let rep = |direction, channel, bd, repeat| Task {
        direction,
        channel,
        bd,
        repeat,
        issue_token: false,
    };
    let task = |direction, channel, bd| rep(direction, channel, bd, 1);
    let mut v = Vec::new();
    for r in 0..ROWS {
        let tile = array::core_loc(col, r);
        v.push((tile, task(S2mm, gemm_core::A_CHANNEL, gemm_core::A_BD[0])));
        v.push((tile, task(S2mm, gemm_core::B_CHANNEL, gemm_core::B_BD[0])));
        v.push((tile, task(Mm2s, gemm_core::C_CHANNEL, gemm_core::C_BD)));
    }
    let mem = array::mem_loc(col);
    for r in 0..ROWS {
        // I32: finite two-stripe chain, one pass per wave.
        v.push((
            mem,
            if is_i32(topo) || topo.is_g80_bf16() {
                rep(S2mm, r as u32, C_S2MM_BD[r][0], mw as u32)
            } else {
                task(S2mm, r as u32, C_S2MM_BD[r][0])
            },
        ));
    }
    v.push((mem, rep(S2mm, B_FILL_CH, B_FILL_BD[0], kc as u32)));
    if topo.has_a(col) {
        v.push((mem, task(S2mm, A_S2MM_CH, A_S2MM_BD[0])));
    }
    for g in 0..2 {
        v.push((
            mem,
            if is_i32(topo) || topo.is_g80_bf16() {
                rep(Mm2s, C_DRAIN_CH[g], C_DRAIN_BD0[g], mw as u32)
            } else {
                task(Mm2s, C_DRAIN_CH[g], C_DRAIN_BD0[g])
            },
        ));
    }
    for p in 0..2 {
        v.push((mem, rep(Mm2s, B_READ_CH[p], B_GUARD_BD[p], kc as u32)));
        if mw > 1 {
            v.push((mem, rep(Mm2s, B_READ_CH[p], B_WHOLE_BD[p], (mw - 1) as u32)));
        }
    }
    if topo.has_a(col) {
        v.push((mem, task(Mm2s, A_MM2S_CH, A_MM2S_BD[0])));
    }
    v
}

/// A argument: 4 injecting-column segments, `(wave, chunk, E/O half 4 KiB)`.
pub(crate) fn pack_a(geo: &Geometry, a: &[i8]) -> Vec<u8> {
    if geo.topo.is_g80_n512() {
        return super::gemm_g80_n512::pack_a(geo, a);
    }
    let (kc, _) = check_geometry(geo);
    assert_eq!(a.len(), geo.m * geo.k, "A must be m*k");
    let chunk = a_chunk_bytes(geo.topo);
    let segment = geo.waves * kc * chunk;
    let mut out = vec![0u8; A_COLS * segment];
    assert_eq!(out.len(), geo.a_bytes(), "A stream size");
    for c in 0..A_COLS {
        let (q, h) = (c / 2, c % 2);
        for w in 0..geo.waves {
            for ch in 0..kc {
                let at = c * segment + (w * kc + ch) * chunk;
                geo.pack_a_half_chunk(
                    a,
                    w * WAVE_M + q * TM,
                    h,
                    ch,
                    &mut out[at..at + A_HALF_BYTES],
                );
            }
        }
    }
    out
}

/// B argument: 8 column segments, chunk-major `[even N tile, odd N tile]`, each tile `(kb 0..8, nb 0..10)` 8x8 blocks.
pub(crate) fn pack_b(geo: &Geometry, b: &[i8]) -> Vec<u8> {
    if geo.topo.is_g80_n512() {
        return super::gemm_g80_n512::pack_b(geo, b);
    }
    let (kc, _) = check_geometry(geo);
    assert_eq!(b.len(), geo.k * geo.n, "B must be k*n");
    let chunk = b_chunk_bytes(geo.topo);
    let mut out = vec![0u8; COLS * kc * 2 * chunk];
    assert_eq!(out.len(), geo.b_bytes(), "B stream size");
    for col in 0..COLS {
        for ch in 0..kc {
            for par in 0..2 {
                let n0 = (2 * col + par) * TN;
                let base = (col * kc + ch) * 2 * chunk + par * chunk;
                for kb in 0..KB {
                    for nb in 0..NB {
                        for kk in 0..8 {
                            let c0 = n0 + nb * 8;
                            if c0 >= geo.n {
                                continue;
                            }
                            let width = (geo.n - c0).min(8);
                            let row = ch * CHUNK_K + kb * 8 + kk;
                            let dst = base + (kb * NB + nb) * 64 + kk * 8;
                            let s = row * geo.n + c0;
                            for (x, y) in out[dst..dst + width].iter_mut().zip(&b[s..s + width]) {
                                *x = *y as u8;
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Row-major `m*n` C from the packed C stream (M padding discarded): int8 tiles are sign extended, `Epilogue::I32` tiles are
/// 40960 B (stripe 0 = `mb 0..8`, stripe 1 = `mb 8..16`, contiguous) of little-endian i32 8x8 blocks in `(mb 0..16, nb 0..10)`
/// order.
pub(crate) fn unpack_c(geo: &Geometry, output: &[u8]) -> Vec<i32> {
    if geo.topo.is_g80_n512() {
        return super::gemm_g80_n512::unpack_c(geo, output);
    }
    check_geometry(geo);
    assert_eq!(output.len(), geo.c_bytes(), "C stream size");
    let i32_tiles = is_i32(geo.topo);
    if let Some((pitch, col0)) = geo.topo.bf16_rowmajor() {
        return (0..geo.m)
            .flat_map(|row| {
                (0..geo.n).map(move |col| {
                    let off = 2 * (row * pitch + col0 + col);
                    i32::from(u16::from_le_bytes(output[off..off + 2].try_into().unwrap()))
                })
            })
            .collect();
    }
    let mut c = vec![0i32; geo.m * geo.n];
    for col in 0..COLS {
        for w in 0..geo.waves {
            for rho in 0..ROWS {
                let m0 = w * WAVE_M + (rho / 2) * TM;
                let n0 = (2 * col + rho % 2) * TN;
                let base = geo.pair_c_offset(col, w, rho);
                if m0 >= geo.m || n0 >= geo.n {
                    continue;
                }
                for mb in 0..MB {
                    for nb in 0..NB {
                        for i in 0..8 {
                            let row = m0 + mb * 8 + i;
                            if row >= geo.m {
                                break;
                            }
                            for j in 0..8 {
                                let cc = n0 + nb * 8 + j;
                                if cc >= geo.n {
                                    break;
                                }
                                let at = (mb * NB + nb) * 64 + i * 8 + j;
                                c[row * geo.n + cc] = if i32_tiles {
                                    let off = base + at * 4;
                                    i32::from_le_bytes(output[off..off + 4].try_into().unwrap())
                                } else if geo.topo.is_g80_bf16() {
                                    let off = base + at * 2;
                                    i32::from(u16::from_le_bytes(
                                        output[off..off + 2].try_into().unwrap(),
                                    ))
                                } else {
                                    output[base + at] as i8 as i32
                                };
                            }
                        }
                    }
                }
            }
        }
    }
    c
}

/// Issue-cycle growth of a K64 chunk against the 128x64 pair core: `1280 - 1024` VMACs per chunk (the wavefront is
/// `1315` against `1059` bundles), control unchanged (`docs/g80-feasibility.md` Gate 5c: later chunk ping / pong 1387 / 1379).
const CHUNK_GROWTH: u64 =
    (TM * TN * CHUNK_K / 512 - gemm_core::TM * gemm_core::TN * gemm_core::CHUNK_K / 512) as u64;
/// Issue cycles of the G80 reverse int8 epilogue `[Fast, Slow]`: 663 emitted for `Fast` (Gate 5c); `Slow` adds the baseline's
/// `550 - 530` serial-control delta ([INFERENCE]: not separately executed).
const EPILOGUE_CYCLES: [u64; 2] = [
    663,
    663 + gemm_core::PAIR_EPILOGUE_CYCLES[1] - gemm_core::PAIR_EPILOGUE_CYCLES[0],
];
/// One full wavefront of the compute region, the cost of each extra `Probe::repeat` pass.
const WAVEFRONT_BUNDLES: u64 = (32 * 5 * 8 + 35) as u64;

/// ESTIMATE only (never a measurement). Per K64 chunk a core needs `max(issue, 1280)`: the G80 issue cycles
/// ([`gemm_core::PAIR_CHUNK_CYCLES`] + [`CHUNK_GROWTH`], ping and pong averaged, per control discipline) against the
/// 1280-cycle B stream of the core (5120 B at 4 B/cycle; the 1024-cycle A half is hidden).
///
/// `Epilogue::Int8`: per wave the core adds the reverse-epilogue issue cycles and the exposed `Cout` drain
/// (`10240 / 4 = 2560` cycles: `Cout` shares the top of the i32 `C`, so chunk 0 of the next wave waits for the previous
/// wave's drain). The memtile -> shim C transfer (2 channels x 4 B/cycle, two 10240 B tiles each = 5120 cycles) overlaps the
/// next wave through the two memtile slots, so the wave period is `max(core wave cycles, shim C cycles)` and the last shim
/// transfer is exposed.
///
/// `Epilogue::I32`: no epilogue body and **no overlap credit**. One 20480 B memtile slot per row serialises every stripe
/// hand-off, each stripe phase taking `P = 20480 / 4 = 5120` cycles at 4 B/cycle. From `C_FULL` the `C_EMPTY` of core row 1
/// (rows 1 and 3 gate their pair through the peer lock protocol) returns only after six chained phases: row 0 stripe 0 writes
/// the slot; it drains; row 0 stripe 1 writes; it drains; row 1 stripe 0 drains (written concurrently with row 0 stripe 0 but
/// drained behind row 0); row 1 stripe 1 writes, which is when the core DMA finishes and returns `C_EMPTY`. The seventh phase
/// (row 1 stripe 1 drain) is hidden by the next wave's compute (`kc >= 3`) and exposed only after the last wave. The wave
/// period is `max(kc * chunk + 6 P, 4 P)` and the total adds `7 P` for the last wave. Stripe phases are never overlapped
/// beyond what the two independent row streams of a drain channel allow (row 1 stripe 0 write), so this is an upper-end
/// estimate of the I32 drain.
/// `Probe::no_compute` removes the compute and the epilogue body, `Probe::repeat = N` adds `N - 1` wavefronts to every
/// later chunk. Ignores lock waits beyond this, bank conflicts, DDR contention and start-up beyond the first chunk.
/// `output_dma_cycles` reports the per-wave epilogue plus exposed `Cout` drain cycles summed over the waves (I32: the
/// six-phase `C_EMPTY` stall per wave plus one final drain phase).
pub(crate) fn estimate(geo: &Geometry, ctl: Control, clock_hz: u64) -> CycleEstimate {
    if geo.topo.is_g80_n512() {
        return super::gemm_g80_n512::estimate(geo, ctl, clock_hz);
    }
    assert!(clock_hz > 0);
    let (kc, _) = check_geometry(geo);
    let (kc, waves) = (kc as u64, geo.waves as u64);
    let probe = ctl.probe();
    let slow = usize::from(probe.serial);
    let input_chunk = (b_chunk_bytes(geo.topo) / 4) as u64;
    let chunk = |cycles: u64| {
        if probe.no_compute {
            input_chunk
        } else {
            (cycles + CHUNK_GROWTH + (u64::from(probe.repeat.max(1)) - 1) * WAVEFRONT_BUNDLES)
                .max(input_chunk)
        }
    };
    let [ping, pong] = gemm_core::PAIR_CHUNK_CYCLES[slow].map(chunk);
    let compute = (kc * (ping + pong)).div_ceil(2);
    let (estimated_cycles, output_dma_cycles) = if is_i32(geo.topo) {
        let stripe = (g80_core::C_STRIPE_BYTES / 4) as u64;
        let wave_core = compute + I32_HANDOFF_PHASES * stripe;
        let shim_c = 4 * stripe;
        (
            input_chunk
                + (waves - 1) * wave_core.max(shim_c)
                + compute
                + (I32_HANDOFF_PHASES + 1) * stripe,
            (I32_HANDOFF_PHASES * waves + 1) * stripe,
        )
    } else if geo.topo.is_g80_bf16() {
        let phase = (bf16::OUT_BYTES / 4) as u64;
        let epilogue = bf16::wave_issue_cycles();
        // Shared pair stage: row0 write/drain, row1 write before its core can
        // reuse C. Row1's final drain is exposed once at the submit tail.
        let wave_core = compute + epilogue + 3 * phase;
        (
            input_chunk + waves * wave_core + phase,
            waves * (epilogue + 3 * phase) + phase,
        )
    } else {
        let epilogue = if probe.no_compute {
            0
        } else {
            EPILOGUE_CYCLES[slow]
        };
        let cout_drain = (COUT_BYTES / 4) as u64;
        let shim_c = (2 * COUT_BYTES / 4) as u64;
        let wave_core = compute + epilogue + cout_drain;
        (
            input_chunk + (waves - 1) * wave_core.max(shim_c) + wave_core + shim_c,
            (epilogue + cout_drain) * waves,
        )
    };
    let estimated_seconds = estimated_cycles as f64 / clock_hz as f64;
    let useful_ops = 2.0 * geo.m as f64 * geo.n as f64 * geo.k as f64;
    CycleEstimate {
        ideal_mac_cycles: (TM * TN * CHUNK_K / 512) as u64 * kc * waves,
        input_dma_cycles: input_chunk * kc * waves,
        output_dma_cycles,
        estimated_cycles,
        estimated_seconds,
        useful_tops: useful_ops / estimated_seconds / 1e12,
        peak_ops_per_second: peak_ops_per_second(clock_hz),
    }
}

/// Build the G80 whole-array design (module header): `n = 1280`, `k` a multiple of 64 in `64..=2560`, `m >= 1` padded to 256
/// rows per wave with at most 256 waves, int8 (`shift <=31`), I32 or direct BF16, and `Probe::layout ==0`.
/// BF16 requires ordinary compute, no timing-only probe. Invalid shapes panic before any emitted byte.
pub fn design_g80(m: usize, n: usize, k: usize, epi: Epilogue, ctl: Control) -> ArrayDesign {
    design_g80_bf16_layout(m, n, k, epi, ctl, Bf16Layout::NativeTiles)
}

/// G80 builder with an explicit BF16 destination; other epilogues require native tiles.
pub fn design_g80_bf16_layout(
    m: usize,
    n: usize,
    k: usize,
    epi: Epilogue,
    ctl: Control,
    layout: Bf16Layout,
) -> ArrayDesign {
    assert!(
        epi == Epilogue::Bf16 || layout == Bf16Layout::NativeTiles,
        "row-major is BF16-only"
    );
    if let Epilogue::Int8 { shift } = epi {
        assert!(shift <= 31, "G80 SRS shift {shift} exceeds 31");
    }
    assert_eq!(
        ctl.probe().layout,
        0,
        "G80 has one fixed data-memory map; Probe::layout={} is not defined",
        ctl.probe().layout
    );
    assert!(m > 0, "G80 needs M >= 1");
    assert_eq!(n, WAVE_N, "G80 needs N = {WAVE_N} exactly, got {n}");
    assert!(
        k % CHUNK_K == 0 && (CHUNK_K..=g80_core::MAX_KC * CHUNK_K).contains(&k),
        "G80 K must be a multiple of {CHUNK_K} in {CHUNK_K}..={}, got {k}",
        g80_core::MAX_KC * CHUNK_K
    );
    let waves = m.div_ceil(WAVE_M);
    assert!(
        waves <= MAX_WAVES,
        "G80 M={m} needs {waves} waves, exceeding {MAX_WAVES}"
    );
    layout
        .validate(waves * WAVE_M)
        .unwrap_or_else(|e| panic!("{e}"));
    let mut geo = Geometry::g80(m, n, k, epi, ctl);
    if let Kind::G80 { bf16_layout, .. } = &mut geo.topo.kind {
        *bf16_layout = layout;
    }
    let topo = geo.topo;
    let program = |role| g80_core::program_pair(geo.kc, geo.waves, role, epi, ctl).finish();
    let (lower, upper) = (program(PairRole::Lower), program(PairRole::Upper));
    for p in [&lower, &upper] {
        assert!(
            p.len() <= 16 * 1024,
            "G80 core program {} B exceeds 16 KiB",
            p.len()
        );
    }
    let insts = array::build_txn_dynamic(&geo, false);
    ArrayDesign {
        pdi: crate::pdi::build(&array::build_cdo(topo, [&lower, &upper], false).to_words()),
        insts: insts.to_bytes(),
        args: vec![
            ArgSpec {
                bytes: geo.a_bytes(),
                kind: ArgKind::In,
            },
            ArgSpec {
                bytes: geo.b_bytes(),
                kind: ArgKind::In,
            },
            ArgSpec {
                bytes: geo.c_bytes(),
                kind: ArgKind::Out,
            },
        ],
        geo,
        variant: Variant::G80,
        program: lower,
        upper_program: upper,
        core: if ctl.is_slow() {
            CoreVariant::FastSlowCtl
        } else {
            CoreVariant::Fast
        },
        v8: Some((epi, ctl)),
        shim_axi: crate::dma::ShimAxi::default(),
    }
}

/// CPU-only deployment builder for one K2560, N1280 panel. Pads rows to a
/// complete M256 wave; no Qwen-specific shape or device ownership lives here.
pub fn build_panel_program(rows_max: usize, epilogue: Epilogue) -> Result<ArrayDesign, String> {
    if !(1..=256 * WAVE_M).contains(&rows_max) {
        return Err(format!(
            "G80 rows_max {rows_max} outside 1..={}",
            256 * WAVE_M
        ));
    }
    Ok(design_g80(
        rows_max.div_ceil(WAVE_M) * WAVE_M,
        WAVE_N,
        2560,
        epilogue,
        Control::Fast,
    ))
}

/// K2560 BF16 panel into a fused row-major arena (M padded to 256).
/// Build once with col0=0 and advance arg2's base by `panel_col0 * 2`
/// through `append_run_body` / `append_lean_run_body` for every panel.
/// The caller must retain the full arena allocation behind that offset.
pub fn build_panel_program_rowmajor(
    rows_max: usize,
    pitch: usize,
    col0: usize,
) -> Result<ArrayDesign, String> {
    if !(1..=256 * WAVE_M).contains(&rows_max) {
        return Err(format!(
            "G80 rows_max {rows_max} outside 1..={}",
            256 * WAVE_M
        ));
    }
    let rows = rows_max.div_ceil(WAVE_M) * WAVE_M;
    let layout = Bf16Layout::RowMajor {
        pitch_elems: pitch,
        col0_elems: col0,
    };
    layout.validate(rows)?;
    Ok(design_g80_bf16_layout(
        rows,
        WAVE_N,
        2560,
        Epilogue::Bf16,
        Control::Fast,
        layout,
    ))
}

/// BF16 activation stream. Each K64 E/O half has 4096 code bytes followed
/// by Dm[128] u16 and De[128] i16; both halves repeat the full token scale row.
/// Padding tokens have zero scales. Scale metadata is deliberately repeated per
/// chunk to reuse the existing ping/pong descriptors and locks.
pub fn pack_a_bf16(design: &ArrayDesign, a: &[i8], scales: &[(u16, i16)]) -> Vec<u8> {
    if design.geo.topo.is_g80_n512() {
        return super::gemm_g80_n512::pack_a_bf16(design, a, scales);
    }
    let geo = &design.geo;
    assert!(
        geo.topo.is_g80_bf16(),
        "BF16 sidecars require a BF16 G80 design"
    );
    assert_eq!(scales.len(), geo.m);
    let mut out = pack_a(geo, a);
    let segment = geo.a_segment_bytes();
    for col in 0..A_COLS {
        for wave in 0..geo.waves {
            for ch in 0..geo.kc {
                let base = col * segment + (wave * geo.kc + ch) * bf16::A_BYTES + A_HALF_BYTES;
                let row0 = wave * WAVE_M + (col / 2) * TM;
                for t in 0..TM {
                    let (m, e) = scales.get(row0 + t).copied().unwrap_or((0, 0));
                    out[base + 2 * t..base + 2 * t + 2].copy_from_slice(&m.to_le_bytes());
                    out[base + 256 + 2 * t..base + 258 + 2 * t].copy_from_slice(&e.to_le_bytes());
                }
            }
        }
    }
    out
}

/// BF16 resident B panel: each 5120-byte K64 N80 tile is followed by Sm[96]
/// u16 and Se[96] i16 (80 live +16 zero padding for aligned vector loads).
pub fn pack_b_bf16(design: &ArrayDesign, b: &[i8], scales: &[(u16, i16)]) -> Vec<u8> {
    if design.geo.topo.is_g80_n512() {
        return super::gemm_g80_n512::pack_b_bf16(design, b, scales);
    }
    let geo = &design.geo;
    assert!(
        geo.topo.is_g80_bf16(),
        "BF16 sidecars require a BF16 G80 design"
    );
    assert_eq!(scales.len(), geo.n);
    let mut out = pack_b(geo, b);
    for col in 0..COLS {
        for ch in 0..geo.kc {
            for parity in 0..2 {
                let base = ((col * geo.kc + ch) * 2 + parity) * bf16::B_BYTES + B_BYTES;
                let n0 = (2 * col + parity) * TN;
                for n in 0..TN {
                    let (m, e) = scales[n0 + n];
                    out[base + 2 * n..base + 2 * n + 2].copy_from_slice(&m.to_le_bytes());
                    out[base + 192 + 2 * n..base + 194 + 2 * n].copy_from_slice(&e.to_le_bytes());
                }
            }
        }
    }
    out
}

/// Byte offset of one BF16 value in a G80 output panel, padded to M256.
/// Panels are concatenated by the host; this offset is relative to one panel.
pub fn bf16_output_offset(padded_rows: usize, row: usize, col: usize) -> usize {
    assert!(padded_rows > 0 && padded_rows % WAVE_M == 0);
    assert!(row < padded_rows && col < WAVE_N);
    let waves = padded_rows / WAVE_M;
    let tile = col / TN;
    let rho = 2 * ((row % WAVE_M) / TM) + tile % 2;
    let base = ((tile / 2) * waves * ROWS + (rho / 2) * waves * 2 + (row / WAVE_M) * 2 + rho % 2)
        * bf16::OUT_BYTES;
    let (r, c) = (row % TM, col % TN);
    base + 2 * ((r / 8 * NB + c / 8) * 64 + (r % 8) * 8 + c % 8)
}
