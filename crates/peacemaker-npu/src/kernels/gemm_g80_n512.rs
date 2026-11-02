// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//! G80 N512 deployment: direct-BF16 `K <= 6336` GEMM with the whole 64-column-per-column B panel resident in the memtiles,
//! on the V8 wave of 512 rows x 512 columns and the 128x64 pair core ([`gemm_core_g80::program_bf16_n64`]). Nothing here
//! is a hardware claim: every statement is what the encoders emit; behaviour on silicon is not established.
//!
//! # Why this geometry
//! The B panel of the out-projections (`N = 2560`, `K = 6144`) is 3.4 MB per 512 columns: it fits the eight 512 KiB memtiles
//! (4.19 MB) only as one 64-column tile per column. That is exactly the V8 wave (`gemm_array::Geometry::pair_tile`): core
//! `(col, rho)` computes the 128 x 64 tile of A block `2*(rho/2) + col%2` and B half `rho%2` of B block `col/2`, vertical
//! pairs share one A block through neighbour data memory, and the two columns of a B block exchange their halves on the
//! East/West lane 2. The switch circuits are V8's ([`gemm_array::pair_column_circuits`]), unchanged.
//!
//! # Host arguments (byte counts, `W = waves`, `kc` K64 chunks)
//! * arg0 = A, **8 segments** of `W*kc*4608` B; column `c` injects half `h = (c/2)%2` (even / odd 8-row blocks) of A block
//!   `2*(c/4) + c%2`, rows `w*512 + block*128 ..`. Per `(wave, chunk)`: 4096 code bytes (`mbl 0..8 x kb 0..8` 8x8 blocks)
//!   then `Dm[128]` u16 and `De[128]` i16 (both halves repeat the token scales, [`pack_a_bf16`]).
//! * arg1 = B, **8 segments** of `kc*4352` B: column `c` holds B half `c%2` of block `c/2`, i.e. features
//!   `(c/2)*128 + (c%2)*64 ..+64`, per chunk 4096 code bytes (`kb 0..8 x nb 0..8` 8x8 blocks) then `Sm[64]` u16, `Se[64]`
//!   i16 ([`pack_b_bf16`]). B is read from DDR once per submit whatever `W` is.
//! * arg2 = C: native tiles (`Bf16Layout::NativeTiles`, per column S2MM0 stream (rows 0, 1) then S2MM1 stream (rows 2, 3),
//!   each `(wave, rho%2)` tiles of 16,384 B in `(mb 0..16, nb 0..8)` 8x8 BF16 blocks) or row-major at a caller pitch / col0
//!   (`Bf16Layout::RowMajor`): one shim BD per channel and wave scatters both 64-column halves with
//!   `(1,32), (pitch/2,128), (32,0)` (8192 words), rows `w*512 + (2g + col%2)*128`, columns `col0 + (col/2)*128`.
//!
//! # Memtile map (own view `MEM_BASE + off`, per column)
//! | region | half-open offsets |
//! |---|---|
//! | reserved for the lean ring scratch, never accessed | `[0, 0x4000)` |
//! | C stage row `r` (one 16,384 B BF16 tile, finite single slot) | `0x4000 + r*0x4000`, ends `0x14000` |
//! | A ping / pong (4608 B each) | `[0x14000, 0x15200)` / `[0x15200, 0x16400)` |
//! | B resident (`kc*4352` B, single copy) | `[0x16400, 0x16400 + kc*4352)`, ends `0x7c400` at `kc = 96` (`MAX_KC = 99`) |
//!
//! # Memtile BDs (even channel => id < 24, odd => id >= 24) and locks
//! | function | channel | BD ids | length | locks (memtile-local) |
//! |---|---|---|---|---|
//! | C producer row `r` | S2MM `r` | `0 / 24 / 2 / 26`, finite task `repeat = W` | 4096 words | acquire `c_lock(r)` (empty), release `+1` (full) |
//! | C drain rows 0, 1 / 2, 3 | MM2S0 / MM2S3 | `6 -> 7` / `32 -> 33`, finite `repeat = W` | 4096 words | acquire full `-1`, release empty `+1` of its row |
//! | B fill | S2MM4 | `4`, `5` (groups of <= 64 chunks), finite `repeat = len`, iteration `step = 1088, wrap = len` | 1088 words | acquire credit (`6`) `-1`, release ready (`4`) `+1` |
//! | B guarded read | MM2S1 | `30`, `31` (same groups), finite `repeat = len`, same iteration | 1088 words | acquire ready `-1`, release credit `+1` |
//! | B whole replay | MM2S1 | `36`, finite `repeat = W - 1` (omitted for `W = 1`), queued after the guard task | `kc*1088` words | none |
//! | A fill / replay | S2MM5 / MM2S2 | `28 <-> 29` / `14 <-> 15` (cyclic) | 1152 words | `A_EMPTY/A_FULL` ping / pong |
//!
//! 17 memtile BDs (15 at kc <= 64) and 14 locks (ids `0..3` A, `4` B ready, `6` B credit, `8..15` C) on every column. The B guard / whole
//! structure is the 128x80 G80's (reader chunk `j` starts only after fill chunk `j` landed, then the lock-free whole pass
//! replays an immutable panel), with one parity instead of two. A memtile lock holds at most 63, and the fill can run
//! ahead of the stalled reader by the whole panel (`kc = 96`), so the fill additionally acquires a credit lock initialised
//! to `min(kc, 63)` that the guarded reader returns (the V10 B-prefix scheme): ready never exceeds 63 and the credit ends at
//! its initial value. Completed-submit end state: all tasks finite; the A rings and core rings park at
//! BD `J%2` with `J = kc*W` (the lean repair of the G80 BF16 design applies unchanged), C locks end empty 1 / full 0.
use super::gemm_array::{
    self as array, ArrayDesign, CycleEstimate, Geometry, Kind, Topo, Variant, MAX_WAVES,
};
use super::gemm_core::{self, Control, CoreVariant, Epilogue, PairRole};
use super::gemm_core_g80 as g80_core;
use super::gemm_g80::{self as g80, Bf16Layout};
use super::gemm_i8::{peak_ops_per_second, ArgKind, ArgSpec};
use super::iu4_ief15_core::g80_bf16 as bf16;
use crate::{
    dma::{
        Bd, BdLocks, Dim,
        Direction::{Mm2s, S2mm},
        Iteration, Location, Task,
    },
    route::Circuit,
};

const TILE: bf16::Tile = bf16::NARROW;
const TM: usize = 128;
const TN: usize = 64;
const CHUNK_K: usize = 64;
const MB: usize = TM / 8;
const NB: usize = TN / 8;
const KB: usize = CHUNK_K / 8;
const ROWS: usize = array::ROWS;
const COLS: usize = array::COLS;
/// Rows / columns of one wave: four 128-row A blocks x eight 64-column B halves.
pub const WAVE_M: usize = array::G80_N512_WAVE_M;
pub const WAVE_N: usize = array::G80_N512_WAVE_N;
/// One A half chunk (4096 codes + `Dm[128]` + `De[128]`), one B chunk (4096 codes + `Sm[64]` + `Se[64]`), one BF16 tile.
pub const A_CHUNK_BYTES: usize = TILE.a_bytes();
pub const B_CHUNK_BYTES: usize = TILE.b_bytes();
pub const TILE_BYTES: usize = TILE.out_bytes();
const _: () = assert!(
    WAVE_M == 512
        && WAVE_N == 512
        && A_CHUNK_BYTES == 4608
        && B_CHUNK_BYTES == 4352
        && TILE_BYTES == 16384
);

// Memtile map (byte offsets).
const C_OFF: u32 = 0x4000;
const C_STAGE: u32 = TILE_BYTES as u32;
const A_OFF: [u32; 2] = [
    C_OFF + ROWS as u32 * C_STAGE,
    C_OFF + ROWS as u32 * C_STAGE + A_CHUNK_BYTES as u32,
];
const B_OFF: u32 = A_OFF[1] + A_CHUNK_BYTES as u32;
const MEM_END: u32 = 0x80000;
/// Largest K64 chunk count: the resident B panel must fit behind the A rings.
pub const MAX_KC: usize = (MEM_END - B_OFF) as usize / B_CHUNK_BYTES;
const _: () = assert!(A_OFF[0] == 0x14000 && B_OFF == 0x16400 && MAX_KC == 99);
/// B ready credit lock (the unused `B_EMPTY[1]` id of the two-parity G80) and its window: a memtile lock holds at most 63.
const B_CREDIT: u32 = gemm_core::B_EMPTY[1];
const B_CREDIT_MAX: u32 = 63;
/// A memtile BD iteration wraps at most 64 times: the B fill / guard chunk iterations come in groups of at most 64 chunks
/// (`kc <= 128`), one finite BD each (fill BD `B_FILL_BD[g]` = 4, 5; guarded reader BD `B_GUARD_BD[g]` = 30, 31); the lock-free
/// whole-panel replay is BD 36 (odd bank), queued after the guarded tasks (3 of the 4 MM2S queue slots).
const B_GROUP_MAX: usize = 64;
const B_GUARD_BD: [u32; 2] = [30, 31];
const B_WHOLE_BD: u32 = 36;

/// `(first chunk, chunks)` of the B groups: `ceil(kc/64)` groups of near-equal size.
fn b_groups(kc: usize) -> Vec<(usize, usize)> {
    let n = kc.div_ceil(B_GROUP_MAX);
    assert!(n <= B_GUARD_BD.len(), "N512 kc {kc} needs {n} B groups");
    let (base, extra) = (kc / n, kc % n);
    let mut first = 0;
    (0..n)
        .map(|g| {
            let len = base + usize::from(g < extra);
            let at = first;
            first += len;
            (at, len)
        })
        .collect()
}

fn shape(topo: Topo) -> (usize, usize) {
    match topo.kind {
        Kind::G80 {
            kc, mw, n512: true, ..
        } => (kc, mw),
        other => panic!("gemm_g80_n512 needs an N512 topology, got {other:?}"),
    }
}

fn c_off(row: usize) -> u32 {
    C_OFF + row as u32 * C_STAGE
}
fn c_lock(row: usize) -> u32 {
    g80::c_lock(0, row)
}

/// Static stream-switch circuits of one column: V8's.
pub(crate) fn column_circuits(topo: Topo, col: u32) -> Vec<Circuit> {
    shape(topo);
    array::pair_column_circuits(col)
}

/// All memtile BDs of a column `(id, bd)`.
pub(crate) fn memtile_descriptors(topo: Topo, col: u32) -> Vec<(u32, Bd)> {
    let _ = col;
    let (kc, _) = shape(topo);
    let mut v = Vec::new();
    let chunk_words = (B_CHUNK_BYTES / 4) as u32;
    for (g, (first, len)) in b_groups(kc).into_iter().enumerate() {
        let addr = (array::MEM_BASE + B_OFF) as u64 + (first * B_CHUNK_BYTES) as u64;
        let iteration = Iteration {
            step: chunk_words,
            wrap: len as u32,
            current: 0,
        };
        let mut fill = Bd::new(addr, chunk_words);
        fill.iteration = iteration;
        fill.locks = BdLocks {
            acq: Some((array::MEM_LOCK + B_CREDIT, -1)),
            rel: Some((array::MEM_LOCK + g80::B_READY[0], 1)),
        };
        fill.next = None;
        v.push((g80::B_FILL_BD[g], fill));
        let mut guard = Bd::new(addr, chunk_words);
        guard.iteration = iteration;
        guard.locks = BdLocks {
            acq: Some((array::MEM_LOCK + g80::B_READY[0], -1)),
            rel: Some((array::MEM_LOCK + B_CREDIT, 1)),
        };
        v.push((B_GUARD_BD[g], guard));
    }
    v.push((
        B_WHOLE_BD,
        Bd::new((array::MEM_BASE + B_OFF) as u64, kc as u32 * chunk_words),
    ));
    let words = (A_CHUNK_BYTES / 4) as u32;
    let (empty, full) = (gemm_core::A_EMPTY, gemm_core::A_FULL);
    v.extend(array::ring(g80::A_S2MM_BD, A_OFF, words, empty, full, true));
    v.extend(array::ring(
        g80::A_MM2S_BD,
        A_OFF,
        words,
        empty,
        full,
        false,
    ));
    let tile_words = (TILE_BYTES / 4) as u32;
    for r in 0..ROWS {
        let addr = (array::MEM_BASE + c_off(r)) as u64;
        let mut producer = Bd::new(addr, tile_words);
        producer.locks = BdLocks {
            acq: Some((array::MEM_LOCK + c_lock(r), -1)),
            rel: Some((array::MEM_LOCK + c_lock(r) + 1, 1)),
        };
        v.push((g80::C_S2MM_BD[r][0], producer));
        let id = g80::C_DRAIN_BD0[r / 2] + (r % 2) as u32;
        let mut drain = Bd::new(addr, tile_words);
        if topo.bf16_rowmajor().is_some() {
            // Native (mb, nb, i, 4 words) -> rows (mb, i, nb, 4 words): the same 4096 stream words, 64 columns per row.
            drain.dims = [
                Dim { step: 1, wrap: 4 },
                Dim {
                    step: 32,
                    wrap: NB as u32,
                },
                Dim { step: 4, wrap: 8 },
                Dim {
                    step: (NB * 32) as u32,
                    wrap: 0,
                },
            ];
        }
        drain.locks = BdLocks {
            acq: Some((array::MEM_LOCK + c_lock(r) + 1, -1)),
            rel: Some((array::MEM_LOCK + c_lock(r), 1)),
        };
        drain.next = (r % 2 == 0).then_some(id + 1);
        v.push((id, drain));
    }
    v
}

/// Initial memtile lock values: `*_EMPTY` 1, `*_FULL` 0, B ready 0, B credit `min(kc, 63)`.
pub(crate) fn memtile_locks(topo: Topo, col: u32) -> Vec<(u32, u32)> {
    let _ = col;
    let (kc, _) = shape(topo);
    let mut v = vec![
        (g80::B_READY[0], 0),
        (B_CREDIT, (kc as u32).min(B_CREDIT_MAX)),
    ];
    for i in 0..2 {
        v.extend([(gemm_core::A_EMPTY[i], 1), (gemm_core::A_FULL[i], 0)]);
    }
    for r in 0..ROWS {
        v.extend([(c_lock(r), 1), (c_lock(r) + 1, 0)]);
    }
    v
}

/// Core ring tasks plus the memtile tasks of a column (the shim tasks depend on the arguments). Finite B tasks come in queue
/// order: the guarded task, then the whole-stream task (omitted for `mw = 1`).
pub(crate) fn ring_tasks(topo: Topo, col: u32) -> Vec<(Location, Task)> {
    let _ = col;
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
        v.push((mem, rep(S2mm, r as u32, g80::C_S2MM_BD[r][0], mw as u32)));
    }
    for (g, (_, len)) in b_groups(kc).into_iter().enumerate() {
        v.push((
            mem,
            rep(S2mm, g80::B_FILL_CH, g80::B_FILL_BD[g], len as u32),
        ));
    }
    v.push((mem, task(S2mm, g80::A_S2MM_CH, g80::A_S2MM_BD[0])));
    for g in 0..2 {
        v.push((
            mem,
            rep(Mm2s, g80::C_DRAIN_CH[g], g80::C_DRAIN_BD0[g], mw as u32),
        ));
    }
    for (g, (_, len)) in b_groups(kc).into_iter().enumerate() {
        v.push((
            mem,
            rep(Mm2s, g80::G80_B_MM2S_CH[0], B_GUARD_BD[g], len as u32),
        ));
    }
    if mw > 1 {
        v.push((
            mem,
            rep(Mm2s, g80::G80_B_MM2S_CH[0], B_WHOLE_BD, (mw - 1) as u32),
        ));
    }
    v.push((mem, task(Mm2s, g80::A_MM2S_CH, g80::A_MM2S_BD[0])));
    v
}

/// The completed finite B tasks a lean body requeues (fill groups, guarded groups, whole-panel replay iff `mw > 1`, in the
/// full submit's queue order and with its repeats) and the BDs whose iteration `current` it clears. After a completed submit
/// each of these channels is idle with an empty queue and `current = len mod len = 0`; the ready / credit locks are back at
/// `0 / min(kc, 63)`, so no B lock write or reset is ever needed.
pub(crate) fn lean_b_tasks(kc: usize, mw: usize) -> (Vec<Task>, Vec<u32>) {
    let rep = |direction, channel, bd, repeat| Task {
        direction,
        channel,
        bd,
        repeat,
        issue_token: false,
    };
    let groups = b_groups(kc);
    let mut tasks = Vec::new();
    let mut currents = Vec::new();
    for (g, &(_, len)) in groups.iter().enumerate() {
        tasks.push(rep(S2mm, g80::B_FILL_CH, g80::B_FILL_BD[g], len as u32));
        currents.extend([g80::B_FILL_BD[g], B_GUARD_BD[g]]);
    }
    for (g, &(_, len)) in groups.iter().enumerate() {
        tasks.push(rep(Mm2s, g80::G80_B_MM2S_CH[0], B_GUARD_BD[g], len as u32));
    }
    if mw > 1 {
        tasks.push(rep(
            Mm2s,
            g80::G80_B_MM2S_CH[0],
            B_WHOLE_BD,
            (mw - 1) as u32,
        ));
    }
    (tasks, currents)
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
        "inconsistent G80 N512 geometry {geo:?}"
    );
    (kc, mw)
}

/// Rows of A block `2*(c/4) + c%2` injected by column `c` in wave `w`, and its half.
fn a_source(col: usize, wave: usize) -> (usize, usize) {
    let (q, h, par) = (col / 4, (col / 2) % 2, col % 2);
    (wave * WAVE_M + (2 * q + par) * TM, h)
}

/// A argument: 8 injecting-column segments, `(wave, chunk, E/O half 4096 B)` (sidecar space left zero).
pub(crate) fn pack_a(geo: &Geometry, a: &[i8]) -> Vec<u8> {
    let (kc, _) = check_geometry(geo);
    assert_eq!(a.len(), geo.m * geo.k, "A must be m*k");
    let segment = geo.waves * kc * A_CHUNK_BYTES;
    let mut out = vec![0u8; COLS * segment];
    assert_eq!(out.len(), geo.a_bytes(), "A stream size");
    for c in 0..COLS {
        for w in 0..geo.waves {
            for ch in 0..kc {
                let (row0, h) = a_source(c, w);
                let at = c * segment + (w * kc + ch) * A_CHUNK_BYTES;
                geo.pack_a_half_chunk(a, row0, h, ch, &mut out[at..at + 4096]);
            }
        }
    }
    out
}

/// B argument: 8 column segments, chunk-major, each chunk `(kb 0..8, nb 0..8)` 8x8 blocks of features
/// `(c/2)*128 + (c%2)*64 ..+64` (sidecar space left zero).
pub(crate) fn pack_b(geo: &Geometry, b: &[i8]) -> Vec<u8> {
    let (kc, _) = check_geometry(geo);
    assert_eq!(b.len(), geo.k * geo.n, "B must be k*n");
    let mut out = vec![0u8; COLS * kc * B_CHUNK_BYTES];
    assert_eq!(out.len(), geo.b_bytes(), "B stream size");
    for col in 0..COLS {
        for ch in 0..kc {
            let n0 = (col / 2) * 2 * TN + (col % 2) * TN;
            let base = (col * kc + ch) * B_CHUNK_BYTES;
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
    out
}

/// BF16 activation stream: [`pack_a`] plus `Dm[128]` u16 / `De[128]` i16 after each half's 4096 code bytes (both halves
/// of a block repeat the full token scale row). Padding tokens have zero scales.
pub fn pack_a_bf16(design: &ArrayDesign, a: &[i8], scales: &[(u16, i16)]) -> Vec<u8> {
    let geo = &design.geo;
    assert!(
        geo.topo.is_g80_n512(),
        "N512 sidecars require an N512 design"
    );
    assert_eq!(scales.len(), geo.m);
    let mut out = pack_a(geo, a);
    let segment = geo.a_segment_bytes();
    for col in 0..COLS {
        for wave in 0..geo.waves {
            for ch in 0..geo.kc {
                let base = col * segment + (wave * geo.kc + ch) * A_CHUNK_BYTES + 4096;
                let (row0, _) = a_source(col, wave);
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

/// BF16 resident B panel: each 4096-byte K64 N64 tile is followed by `Sm[64]` u16 and `Se[64]` i16.
pub fn pack_b_bf16(design: &ArrayDesign, b: &[i8], scales: &[(u16, i16)]) -> Vec<u8> {
    let geo = &design.geo;
    assert!(
        geo.topo.is_g80_n512(),
        "N512 sidecars require an N512 design"
    );
    assert_eq!(scales.len(), geo.n);
    let mut out = pack_b(geo, b);
    for col in 0..COLS {
        for ch in 0..geo.kc {
            let base = (col * geo.kc + ch) * B_CHUNK_BYTES + 4096;
            let n0 = (col / 2) * 2 * TN + (col % 2) * TN;
            for n in 0..TN {
                let (m, e) = scales[n0 + n];
                out[base + 2 * n..base + 2 * n + 2].copy_from_slice(&m.to_le_bytes());
                out[base + 128 + 2 * n..base + 130 + 2 * n].copy_from_slice(&e.to_le_bytes());
            }
        }
    }
    out
}

/// Row-major `m*n` BF16 bit patterns (as i32) from the packed C stream (M padding discarded).
pub(crate) fn unpack_c(geo: &Geometry, output: &[u8]) -> Vec<i32> {
    check_geometry(geo);
    assert_eq!(output.len(), geo.c_bytes(), "C stream size");
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
                let m0 = w * WAVE_M + (2 * (rho / 2) + col % 2) * TM;
                let n0 = (col / 2) * 2 * TN + (rho % 2) * TN;
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
                                let off = base + ((mb * NB + nb) * 64 + i * 8 + j) * 2;
                                c[row * geo.n + cc] = i32::from(u16::from_le_bytes(
                                    output[off..off + 2].try_into().unwrap(),
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    c
}

/// ESTIMATE only (never a measurement). Per K64 chunk a core needs `max(issue, 1152)`: the pair-core issue cycles
/// ([`gemm_core::PAIR_CHUNK_CYCLES`], ping and pong averaged) against the 1152-cycle A half stream (4608 B at 4 B/cycle; the
/// 1088-cycle B chunk is hidden). Per wave the core adds the integer BF16 epilogue ([`bf16::wave_issue_cycles_for`]) and
/// the exposed 4096-cycle drain of its BF16 tile (the next wave's zeroing waits for `C_EMPTY`); the last wave's second
/// tile per drain channel is exposed once. Ignores lock waits beyond this, bank conflicts, DDR contention and start-up.
pub(crate) fn estimate(geo: &Geometry, ctl: Control, clock_hz: u64) -> CycleEstimate {
    assert!(clock_hz > 0);
    let (kc, _) = check_geometry(geo);
    let (kc, waves) = (kc as u64, geo.waves as u64);
    let slow = usize::from(ctl.probe().serial);
    let input_chunk = (A_CHUNK_BYTES / 4) as u64;
    let [ping, pong] = gemm_core::PAIR_CHUNK_CYCLES[slow].map(|c| c.max(input_chunk));
    let compute = (kc * (ping + pong)).div_ceil(2);
    let phase = (TILE_BYTES / 4) as u64;
    let epilogue = bf16::wave_issue_cycles_for(TILE);
    let wave_core = compute + epilogue + phase;
    let estimated_cycles = input_chunk + waves * wave_core + phase;
    let estimated_seconds = estimated_cycles as f64 / clock_hz as f64;
    let useful_ops = 2.0 * geo.m as f64 * geo.n as f64 * geo.k as f64;
    CycleEstimate {
        ideal_mac_cycles: (TM * TN * CHUNK_K / 512) as u64 * kc * waves,
        input_dma_cycles: input_chunk * kc * waves,
        output_dma_cycles: waves * (epilogue + phase) + phase,
        estimated_cycles,
        estimated_seconds,
        useful_tops: useful_ops / estimated_seconds / 1e12,
        peak_ops_per_second: peak_ops_per_second(clock_hz),
    }
}

/// Build the G80 N512 whole-array design (module header): `n = 512` exactly, `k` a multiple of 64 in `64..=6336`, `m >= 1`
/// padded to 512 rows per wave with at most 256 waves, direct BF16 only, native-tile C.
pub fn design_g80_n512(m: usize, n: usize, k: usize, epi: Epilogue, ctl: Control) -> ArrayDesign {
    design_g80_n512_layout(m, n, k, epi, ctl, Bf16Layout::NativeTiles)
}

/// [`design_g80_n512`] with an explicit BF16 destination (native tiles, or row-major at `pitch` / `col0` with
/// `col0 + 512 <= pitch`).
pub fn design_g80_n512_layout(
    m: usize,
    n: usize,
    k: usize,
    epi: Epilogue,
    ctl: Control,
    layout: Bf16Layout,
) -> ArrayDesign {
    assert_eq!(
        epi,
        Epilogue::Bf16,
        "G80 N512 is a direct-BF16 design, got {epi:?}"
    );
    assert_eq!(
        ctl.probe().layout,
        0,
        "G80 N512 has one fixed data-memory map; Probe::layout={} is not defined",
        ctl.probe().layout
    );
    assert!(m > 0, "G80 N512 needs M >= 1");
    let waves = m.div_ceil(WAVE_M);
    assert!(
        waves <= MAX_WAVES,
        "G80 N512 M={m} needs {waves} waves, exceeding {MAX_WAVES}"
    );
    layout
        .validate_for(waves * WAVE_M, WAVE_N)
        .unwrap_or_else(|e| panic!("{e}"));
    let geo = Geometry::g80_n512(m, n, k, ctl, layout);
    let topo = geo.topo;
    let program = |role| g80_core::program_bf16_n64(geo.kc, geo.waves, role, ctl).finish();
    let (lower, upper) = (program(PairRole::Lower), program(PairRole::Upper));
    for p in [&lower, &upper] {
        assert!(
            p.len() <= 16 * 1024,
            "G80 N512 core program {} B exceeds 16 KiB",
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

/// CPU-only deployment builder for one N512 BF16 panel of reduction `k` into a fused row-major arena (`pitch` even,
/// `col0` even, `col0 + 512 <= pitch`; M padded to 512): the Flash-Next out projections ([`OUT_K`]) and the 27B dense
/// in-projections (K = 5120). Build once with `col0 = 0` and advance arg2's base by `panel_col0 * 2` through
/// `append_run_body` / `append_lean_run_body` for every panel; the caller retains the full arena behind that offset.
pub fn build_panel_program_n512(
    rows_max: usize,
    pitch: usize,
    col0: usize,
    k: usize,
) -> Result<ArrayDesign, String> {
    if !(1..=MAX_WAVES * WAVE_M).contains(&rows_max) {
        return Err(format!(
            "G80 N512 rows_max {rows_max} outside 1..={}",
            MAX_WAVES * WAVE_M
        ));
    }
    if k == 0 || k % CHUNK_K != 0 || k / CHUNK_K > MAX_KC {
        return Err(format!(
            "G80 N512 K {k} is not a multiple of 64 in 64..={}",
            MAX_KC * CHUNK_K
        ));
    }
    let rows = rows_max.div_ceil(WAVE_M) * WAVE_M;
    let layout = Bf16Layout::RowMajor {
        pitch_elems: pitch,
        col0_elems: col0,
    };
    layout.validate_for(rows, WAVE_N)?;
    Ok(design_g80_n512_layout(
        rows,
        WAVE_N,
        k,
        Epilogue::Bf16,
        Control::Fast,
        layout,
    ))
}

/// K of the out-projection programs.
pub const OUT_K: usize = 6144;
