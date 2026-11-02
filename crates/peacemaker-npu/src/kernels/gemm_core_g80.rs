// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// AIE2P instruction fields and scheduling are derived from Xilinx/llvm-aie
// AIE2PGenInstrInfo.td and AIE2PGenSchedule.td (Apache-2.0 WITH LLVM-exception).
//! G80 pair core: the V8 pair wavefront of [`gemm_core`] widened from a 128x64 to a 128x80 output tile (five N
//! pairs per M pair, 40 groups of 32 cycles, 1,315-cycle wavefront), with int8, raw i32 or integer direct-BF16 output.
//!
//! Everything that does not depend on the tile width is reused from `gemm_core`: the [`Assembler`], the
//! 9-step `PEER_FULL`/`PEER_EMPTY` chunk lock protocol ([`pair_acquire`]/[`pair_release`]), the
//! `Probe::repeat` pointer reload ([`pair_pointers`]), instruction encoders, lock ids, DMA channel/BD ids and
//! the `Control`/`Probe` knobs. This file owns the N=80 wavefront, its loop plan, the reverse int8 epilogue and
//! the relocated data-memory map.
//!
//! ## Legacy int8/I32 data-memory map (`Probe::layout` must be 0)
//! `C` i32 accumulator `0x0000..0xa000` (128 x 80 x 4 = 40,960 B); int8 `Cout` `0x7800..0xa000` (10,240 B, inside
//! the top of `C`, `Epilogue::Int8` only); A half slots `0xa000`/`0xb000` (4 KiB); B slots `0xc000`/`0xd400` (5 KiB).
//! `Epilogue::I32` has no `Cout`: the DMA drains `C` itself as two 20,480 B stripes (`[0, 0x5000)` then `[0x5000, 0xa000)`,
//! core DMA BD 4 -> 5 -> 4, see [`tile_bds_pair`]). The int8 epilogue body is not emitted for I32, and `C_FULL` is released
//! right after the last chunk (as the V8 I32 protocol of [`gemm_core::program_pair`]).
//!
//! `Epilogue::Bf16` uses C's upper half `[0x5000,0xa000)` for descending 32-lane stores, A slots
//! `0xa000/0xb200` (4608 B), B slots `0xc400/0xd980` (5504 B), and scratch `[0xef00,0x10000)`.
//! One accumulating wavefront body follows a C-zero loop, leaving room for the integer epilogue in program memory.
//! Sidecars are captured under input locks; DMA BD4 drains one whole 20480-B BF16 tile.
//! See `docs/npu/npu-trunk.md` for the sidecar and shared memtile-stage contract. Simulator-only.
//!
//! ## Wavefront changes against the 128x64 core
//! * B column stride is `128 * NP = 640` B (`NP = 5`), a C row is `512 * NP = 2560` B. The B pointer steps need the
//!   eight modifiers [`MODIFIERS`] (`-512` resets A to repeat it for the next N pair, `3072` plus a `-512`
//!   immediate is the C row skip after five contiguous blocks, `640` is the B stride).
//! * The 1,315-bundle timeline is emitted as a 37-bundle peel, seven trips of a 160-bundle `jnz` body, and a
//!   158-bundle tail (the 128x64 scheme's 170/160x6/185 split overflows the 16 KiB program memory). The last
//!   group's row-1 C-skip pointer op falls past the end of the timeline and is the only dropped op; every other
//!   pointer update is needed for the periodicity of the repeated body. Program size is 12,848 B.
//! * The int8 epilogue converts the 160 blocks in descending order (`VLDA bm` quarters 3..0 with a `-64 B`
//!   post-modify, `VST.SRS.4x` with `-64 B` steps) so `Cout`, which overlaps the upper part of `C`, only
//!   overwrites i32 blocks that were already loaded; the closest (store, load) pair is ten cycles apart.
//! * `C_EMPTY` is acquired before chunk 0 of every wave (also with `Probe::no_compute`): `Cout` lives inside
//!   `C`, so the previous wave's `Cout` DMA drain must be finished before the first chunk writes `C`. For `Epilogue::I32`
//!   the same acquire is the real stall: `C_EMPTY` is returned by the second stripe BD only after both 20,480 B stripes
//!   were drained, so chunk 0 of the next wave waits for the whole 40,960 B drain. `C_FULL` is released after the last
//!   `VST.SRS` has committed (int8) or after the last chunk (I32).
//!
//! Lane order of the dm accumulator vs the four bm quarters is the same simulator assumption as in `gemm_core`.
use super::gemm_core::{
    self, descriptor, instruction, pair_acquire, pair_pointers, pair_release, Assembler, Control,
    Cycle, Epilogue, PairLayout, PairRole, A_LOAD_TIMES, EPI_BODY, EPI_DRAIN_PAD, EPI_LOOP_DEC,
    EPI_LOOP_JNZ, EPI_STORE_DELAY, MAC_TIMES,
};
use super::gemm_core::{emit_cycle, load_a, matrix, pointer, pointer_immediate, put, store};
use super::gemm_i8::SIGNED_8X8;
use super::iu4_ief15_core::g80_bf16 as bf16;
use crate::{
    dma::{self, BdLocks},
    isa::{
        self,
        bundle::encode_slot,
        gen::{Encoding, Slot},
        Program,
    },
};
use std::sync::LazyLock;

pub const TM: usize = 128;
pub const TN: usize = 80;
pub const CHUNK_K: usize = 64;
/// One E/O half of a 128-row A block per K64 chunk: 8 mb x 8 kb 8x8 int8 blocks.
pub const A_HALF_BYTES: usize = TM * CHUNK_K / 2;
pub const B_BYTES: usize = CHUNK_K * TN;
pub const C_BYTES: usize = TM * TN * 4;
pub const COUT_BYTES: usize = TM * TN;
/// Largest K64 chunk count per output tile (`K = 64 * kc`).
pub const MAX_KC: usize = 40;
pub const A_ADDR: [u32; 2] = [0xa000, 0xb000];
pub const B_ADDR: [u32; 2] = [0xc000, 0xd400];
pub const C_ADDR: u32 = 0x0000;
pub const COUT_ADDR: u32 = 0x7800;
const LAYOUT: PairLayout = PairLayout {
    a: A_ADDR,
    b: B_ADDR,
    c: C_ADDR,
    cout: COUT_ADDR,
};
const _: () = assert!(
    COUT_ADDR as usize + COUT_BYTES == C_ADDR as usize + C_BYTES && A_ADDR[0] as usize == C_BYTES
);
/// `Epilogue::I32` drains `C` as two stripes of half the tile (`[C_ADDR, +C_STRIPE_BYTES)` then the upper half) on core BD 4 -> 5.
pub const C_STRIPE_BYTES: usize = C_BYTES / 2;
pub const C_STRIPE_BD: [u32; 2] = [gemm_core::C_BD, gemm_core::C_BD + 1];
const _: () = assert!(C_STRIPE_BYTES == 20480 && 2 * C_STRIPE_BYTES == C_BYTES);

/// N pairs per M pair (each pair is 16 columns).
const NP: i32 = 5;
const GROUPS: i32 = 8 * NP;
const B_STRIDE: i32 = 128 * NP;
/// One C row of blocks (`mb` pair step) in bytes.
const ROW_BYTES: u32 = 512 * NP as u32;
/// Pointer modifier table (`m0..m7`): sorted B steps `{-3072, -2560, -2432, 640, 3072, 3200, 3712}` plus the A reset
/// (`-512`), the C row skip (`512 * NP + 512`) and the B stride (`640`), which are already in the set or merge into it.
const MODIFIERS: [i32; 8] = [-3072, -2560, -2432, -512, 640, 3072, 3200, 3712];
const A_RESET: u8 = 3;
const C_SKIP: u8 = 5;
const _: () = assert!(
    MODIFIERS[A_RESET as usize] == -512
        && MODIFIERS[C_SKIP as usize] == 512 * NP + 512
        && MODIFIERS[4] == B_STRIDE
);

/// Compressed-timeline plan, identical for the overwrite and accumulate wavefronts: `PEEL` straight bundles,
/// `TRIPS` trips of a `PERIOD`-bundle `jnz` body (its counter decrement rides in the free ALU slot of body bundle
/// `DEC`, the branch in the LNG slot of bundle `PERIOD - 1 - JUMP_DELAY_SLOTS`), then the tail.
const PEEL: usize = 37;
const PERIOD: usize = 32 * NP as usize;
const TRIPS: usize = 7;
const DEC: usize = 146;
const WAVE_BUNDLES: usize = GROUPS as usize * 32 + 35;
const _: () = assert!(PEEL + PERIOD * TRIPS < WAVE_BUNDLES);

/// Like `gemm_core`'s `free_store`, but a store-slot op that falls past the end of the timeline is dropped and
/// reported as `false`.
fn free_store(cycles: &mut [Cycle], mut time: i32, instructions: &[(Slot, u64)]) -> bool {
    let mut all = true;
    for &instruction in instructions {
        while ((time + 10) as usize) < cycles.len()
            && cycles[(time + 10) as usize][Slot::St as usize].is_some()
        {
            time += 1;
        }
        if ((time + 10) as usize) >= cycles.len() {
            all = false;
            time += 1;
            continue;
        }
        put(cycles, time, instruction);
        time += 1;
    }
    all
}

/// The 128x80 pair wavefront (timeline offset 10: bundle index = cycle + 10).
fn wavefront(overwrite: bool) -> Vec<Cycle> {
    let mut cycles = vec![[None; 8]; WAVE_BUNDLES];
    let mut b_loads = [Vec::new(), Vec::new()];
    for group in 0..GROUPS {
        for dm in 0..4u8 {
            let start = group * 32 + dm as i32 * 8;
            for k in 0..8u8 {
                let time = start + MAC_TIMES[k as usize];
                put(&mut cycles, time, matrix(dm, k, overwrite && k == 0));
                b_loads[(dm % 2) as usize].push((
                    time - 7,
                    k as i32 * B_STRIDE + group % NP * 128 + (dm % 2) as i32 * 64,
                ));
                if dm % 2 == 0 {
                    let row = dm / 2;
                    put(
                        &mut cycles,
                        start + A_LOAD_TIMES[row as usize][k as usize],
                        load_a(row * 4 + k % 4, row),
                    );
                }
            }
            let last = start + MAC_TIMES[7];
            for quarter in 0..4u8 {
                put(&mut cycles, last + 6 + quarter as i32, store(dm, quarter));
                if !overwrite {
                    put(
                        &mut cycles,
                        start - 9 + quarter as i32,
                        instruction!(
                            "VLDA_dmx_lda_bm_pstm_nrm_imm",
                            &[
                                ("dst", (dm * 4 + quarter) as u64),
                                ("ptr", (6 + dm / 2) as u64),
                                ("imm", 1)
                            ]
                        ),
                    );
                }
            }
        }
    }
    for column in 0..2 {
        b_loads[column].sort_unstable();
        for (index, &(time, offset)) in b_loads[column].iter().enumerate() {
            let step = b_loads[column]
                .get(index + 1)
                .map_or(B_STRIDE, |next| next.1 - offset);
            let modifier = MODIFIERS
                .iter()
                .position(|&m| m == step)
                .expect("G80 B wavefront modifier");
            put(
                &mut cycles,
                time,
                instruction!(
                    "VLDB_dmx_ldb_x_pstm_nrm",
                    &[
                        ("dst", 11),
                        ("ptr", (2 + column) as u64),
                        ("mod", modifier as u64)
                    ]
                ),
            );
        }
    }
    // A cursors repeat the five N pairs before advancing two M blocks (the pair E/O walk already sits on the next
    // mb_local after the fifth). C cursors independently skip the intervening row after five contiguous C blocks.
    for group in 0..GROUPS {
        for row in 0..2u8 {
            let start = group * 32 + row as i32 * 16;
            let last_load = start + A_LOAD_TIMES[row as usize][7];
            if group % NP != NP - 1 {
                assert!(free_store(
                    &mut cycles,
                    last_load + 1,
                    &[pointer(Slot::St, row, A_RESET)]
                ));
            }
        }
        if group % NP == NP - 1 {
            for row in 0..2u8 {
                let store_end = group * 32 + row as i32 * 16 + 40;
                let ok = free_store(
                    &mut cycles,
                    store_end + 1,
                    &[
                        pointer(Slot::St, 4 + row, C_SKIP),
                        pointer_immediate(4 + row, -512),
                    ],
                );
                assert!(
                    ok || group == GROUPS - 1,
                    "dropped C-skip op for non-final group {group}"
                );
                if !overwrite {
                    let load_end = group * 32 + row as i32 * 16 + 2;
                    assert!(free_store(
                        &mut cycles,
                        load_end + 1,
                        &[
                            pointer(Slot::St, 6 + row, C_SKIP),
                            pointer_immediate(6 + row, -512)
                        ]
                    ));
                }
            }
        }
    }
    cycles
}

/// One K64 chunk compute: pointer setup, then peel / `jnz` body / tail. Always the VLIW wavefront, whatever the
/// control discipline (serial gaps are suspended).
fn chunk(a: &mut Assembler, overwrite: bool) {
    static WAVES: LazyLock<[Vec<Cycle>; 2]> = LazyLock::new(|| [wavefront(false), wavefront(true)]);
    let cycles = &WAVES[overwrite as usize];
    let serial = a.serial;
    a.serial = false;
    a.mov(1, 4, gemm_core::CORE_BASE + C_ADDR);
    a.mov(1, 5, gemm_core::CORE_BASE + C_ADDR + ROW_BYTES);
    if !overwrite {
        a.mov(1, 6, gemm_core::CORE_BASE + C_ADDR);
        a.mov(1, 7, gemm_core::CORE_BASE + C_ADDR + ROW_BYTES);
    }
    a.mov(0, 12, TRIPS as u32);
    while a.program.pc() % 16 != 0 {
        a.nop(1);
    }
    for cycle in &cycles[..PEEL] {
        emit_cycle(a, cycle);
    }
    let begin = a.program.pc();
    let jnz = PERIOD - 1 - isa::sched::JUMP_DELAY_SLOTS;
    let mut body: Vec<Cycle> = cycles[PEEL..PEEL + PERIOD].to_vec();
    assert!(
        body[DEC][Slot::Alu as usize].is_none(),
        "loop counter needs the ALU slot"
    );
    body[DEC][Slot::Alu as usize] =
        Some(instruction!("ADD_add_r_ri", &[("d0", 12), ("s0", 12), ("imm", 0x7f)]).1);
    assert!(
        DEC + 8 <= jnz,
        "loop counter decrement must retire before the branch"
    );
    assert!(
        body[jnz][Slot::Alu as usize].is_none()
            && body[jnz][Slot::Lng as usize].is_none()
            && body[jnz][Slot::Mv as usize].is_none(),
        "loop branch needs the LNG slot"
    );
    body[jnz][Slot::Lng as usize] = Some(instruction!("JNZ", &[("i", begin as u64), ("s0", 12)]).1);
    for repeat in 1..TRIPS {
        assert_eq!(
            &cycles[PEEL..PEEL + PERIOD],
            &cycles[PEEL + repeat * PERIOD..PEEL + (repeat + 1) * PERIOD]
        );
    }
    for cycle in &body {
        emit_cycle(a, cycle);
    }
    for cycle in &cycles[PEEL + PERIOD * TRIPS..] {
        emit_cycle(a, cycle);
    }
    a.serial = serial;
}

const EPI_BLOCKS: usize = TM / 8 * TN / 8;
/// `Imm` field of a `-64 B` post-modify (4-bit two's complement of `-1` in 64 B units).
const EPI_BACK: u64 = 15;
/// Straight-line reverse epilogue timeline: processing index `j` handles block `159 - j`; it loads the four 16 x i32
/// quarters `3, 2, 1, 0` into `bm[4(j%4)..]` of `dm(j%4)` on cycles `4j..4j+3` and stores them as 64 int8 lanes with
/// one `VST.SRS.4x` `EPI_STORE_DELAY` cycles after the last quarter (pointers walk backwards, `-64 B` per op).
fn epilogue_wavefront() -> Vec<Cycle> {
    let mut cycles = vec![[None; 8]; (EPI_BLOCKS - 1) * 4 + 3 + EPI_STORE_DELAY + 1];
    for block in 0..EPI_BLOCKS {
        let dm = (block % 4) as u8;
        for i in 0..4u8 {
            let quarter = 3 - i;
            let (slot, bits) = instruction!(
                "VLDA_dmx_lda_bm_pstm_nrm_imm",
                &[
                    ("dst", (dm * 4 + quarter) as u64),
                    ("ptr", 6),
                    ("imm", EPI_BACK)
                ]
            );
            assert!(cycles[block * 4 + i as usize][slot as usize]
                .replace(bits)
                .is_none());
        }
        let (slot, bits) = instruction!(
            "VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_imm_srsSign1",
            &[("src", dm as u64), ("su", 0), ("ptr", 4), ("imm", EPI_BACK)]
        );
        assert!(cycles[block * 4 + 3 + EPI_STORE_DELAY][slot as usize]
            .replace(bits)
            .is_none());
    }
    cycles
}
/// Int8 epilogue: peeled first 16 cycles, a `jnz` loop of 39 identical 16-cycle bodies (`add r12,-1` / `jnz r12`
/// in free slots, delay slots are body bundles), then the store drain and `EPI_DRAIN_PAD` NOPs so `rel C_FULL`
/// follows the last `VST.SRS` commit. `p6` starts at the top quarter of the last C block, `p4` at the last Cout block.
fn epilogue(a: &mut Assembler) {
    static CYCLES: LazyLock<Vec<Cycle>> = LazyLock::new(epilogue_wavefront);
    let cycles = &*CYCLES;
    let trips = EPI_BLOCKS * 4 / EPI_BODY - 1;
    for repeat in 2..=trips {
        assert_eq!(
            &cycles[EPI_BODY..2 * EPI_BODY],
            &cycles[repeat * EPI_BODY..(repeat + 1) * EPI_BODY]
        );
    }
    let last = EPI_BLOCKS as u32 - 1;
    a.mov(1, 6, gemm_core::CORE_BASE + C_ADDR + 256 * last + 192);
    a.mov(1, 4, gemm_core::CORE_BASE + COUT_ADDR + 64 * last);
    a.mov(0, 12, trips as u32);
    let serial = a.serial;
    a.serial = false;
    while a.program.pc() % 16 != 0 {
        a.nop(1);
    }
    for cycle in &cycles[..EPI_BODY] {
        emit_cycle(a, cycle);
    }
    let begin = a.program.pc();
    for (i, cycle) in cycles[EPI_BODY..2 * EPI_BODY].iter().enumerate() {
        let mut cycle = *cycle;
        if i == EPI_LOOP_DEC {
            assert!(cycle[Slot::Alu as usize].is_none());
            cycle[Slot::Alu as usize] =
                Some(instruction!("ADD_add_r_ri", &[("d0", 12), ("s0", 12), ("imm", 0x7f)]).1);
        }
        if i == EPI_LOOP_JNZ {
            assert!(
                cycle[Slot::Alu as usize].is_none()
                    && cycle[Slot::Lng as usize].is_none()
                    && cycle[Slot::Mv as usize].is_none()
            );
            cycle[Slot::Lng as usize] =
                Some(instruction!("JNZ", &[("i", begin as u64), ("s0", 12)]).1);
        }
        emit_cycle(a, &cycle);
    }
    for cycle in &cycles[(trips + 1) * EPI_BODY..] {
        emit_cycle(a, cycle);
    }
    a.nop(EPI_DRAIN_PAD);
    a.serial = serial;
}

/// Validate the epilogue/control knobs shared by [`program_pair`] and [`tile_bds_pair`]; returns the SRS shift (`None` for
/// `Epilogue::I32`, which has no SRS conversion).
fn check(epi: Epilogue, ctl: Control) -> Option<u8> {
    assert_eq!(ctl.probe().layout, 0,
        "G80 128x80 has one fixed data-memory map (C 0x0000, Cout 0x7800, A 0xa000/0xb000, B 0xc000/0xd400); layout={} is not defined",
        ctl.probe().layout);
    match epi {
        Epilogue::Int8 { shift } => {
            assert!(shift <= 31, "SRS shift {shift}");
            Some(shift)
        }
        Epilogue::I32 | Epilogue::Bf16 => None,
    }
}

/// G80 pair core (128x80 tile, int8 or i32 epilogue). Same chunk lock protocol, `Control` disciplines and `Probe` knobs
/// (`serial`, `nocompute`, `repeat=N`) as [`gemm_core::program_pair`]; `kc` in `1..=MAX_KC`, `tiles` in `1..=256`,
/// `Int8 { shift <= 31 }` with `clamp(c >> shift, -128, 127)` semantics, `Probe::layout == 0`.
/// `C_EMPTY` is acquired before chunk 0 of every wave (also with `no_compute`, which skips only the chunk compute
/// and the epilogue body). `Int8`: `C_FULL` is released after the epilogue stores retire. `I32`: no SRS setup and no
/// epilogue body; `C_FULL` is released right after the last chunk and `C` is drained in place by the two stripe BDs.
pub fn program_pair(
    kc: usize,
    tiles: usize,
    role: PairRole,
    epi: Epilogue,
    ctl: Control,
) -> Program {
    assert!(
        (1..=MAX_KC).contains(&kc) && (1..=256).contains(&tiles),
        "G80 kc {kc} (1..={MAX_KC}) tiles {tiles} (1..=256)"
    );
    if matches!(epi, Epilogue::Bf16) {
        return program_bf16(kc, tiles, role, ctl);
    }
    let shift = check(epi, ctl);
    let probe = ctl.probe();
    let mut a = Assembler::default();
    a.serial = probe.serial;
    for (r, v) in [
        (0, -1i32 as u32),
        (1, 1),
        (2, SIGNED_8X8),
        (3, tiles as u32),
        (4, 0),
    ] {
        a.mov(0, r, v);
    }
    for (m, v) in MODIFIERS.into_iter().enumerate() {
        a.mov(2, m, v as u32);
    }
    if let Some(shift) = shift {
        a.set_cr("crSat", 1);
        a.set_cr("crRnd", 0);
        a.set_cr("crSRSMode", 0);
        a.mov_named("s0", u32::from(shift));
    }
    let tiles_loop = a.label();
    let accumulated = a.label();
    let drain = a.label();
    a.bind(tiles_loop);
    a.lock(true, gemm_core::C_EMPTY);
    a.mov(0, 5, kc as u32);
    pair_acquire(&mut a, role, LAYOUT);
    if !probe.no_compute {
        chunk(&mut a, true);
    }
    pair_release(&mut a, role);
    a.branch(drain, Some((5, false)));
    a.delay();
    a.bind(accumulated);
    pair_acquire(&mut a, role, LAYOUT);
    if probe.repeat > 1 {
        a.mov(0, 11, u32::from(probe.repeat));
        let again = a.label();
        a.bind(again);
        pair_pointers(&mut a, role, LAYOUT);
        chunk(&mut a, false);
        a.add(11, -1);
        if !a.serial {
            a.nop(7);
        }
        a.branch(again, Some((11, true)));
        a.delay();
    } else if !probe.no_compute {
        chunk(&mut a, false);
    }
    pair_release(&mut a, role);
    a.branch(accumulated, Some((5, true)));
    a.delay();
    a.bind(drain);
    if shift.is_some() && !probe.no_compute {
        epilogue(&mut a);
    }
    a.lock(false, gemm_core::C_FULL);
    a.add(3, -1);
    a.branch(tiles_loop, Some((3, true)));
    a.delay();
    a.emit(&[instruction!("DONE", &[])]);
    a.finish()
}

/// DMA-view descriptors of the G80 pair core, indexed by BD id, as a fixed six-slot backing array plus the number of valid BDs
/// (`5` for `Int8`, `6` for `I32`; no heap allocation): A halves (4 KiB) on S2MM0 (BD 0, 1), B (5 KiB) on S2MM1 (BD 2, 3) and C on
/// MM2S0, with the BD ids, channels and lock numbering of `gemm_core`. `Int8`: the int8 `Cout` (10 KiB) as the self-cyclic BD 4.
/// `I32`: `C` drained in place as two [`C_STRIPE_BYTES`] stripes, BD 4 (`C` bytes `[0, 0x5000)`, acquires `C_FULL`) -> BD 5
/// (`[0x5000, 0xa000)`, releases `C_EMPTY` only after it finished) -> BD 4; the two stripes make one 20,480 B memtile C staging
/// slot per row sufficient. Emit only `bds[..len]` (the int8 image has no BD 5).
pub fn tile_bds_pair(epi: Epilogue, ctl: Control) -> ([[u32; 6]; 6], usize) {
    let shift = check(epi, ctl);
    let mut bds = [[0; 6]; 6];
    let direct = matches!(epi, Epilogue::Bf16);
    let (aa, ab, ba, bb) = if direct {
        (bf16::A, bf16::A_BYTES, bf16::B, bf16::B_BYTES)
    } else {
        (A_ADDR, A_HALF_BYTES, B_ADDR, B_BYTES)
    };
    for slot in 0..2 {
        bds[gemm_core::A_BD[slot] as usize] = dma::tile_bd(
            aa[slot],
            (ab / 4) as u32,
            BdLocks {
                acq: Some((gemm_core::A_EMPTY[slot], -1)),
                rel: Some((gemm_core::A_FULL[slot], 1)),
            },
            Some(gemm_core::A_BD[1 - slot]),
        );
        bds[gemm_core::B_BD[slot] as usize] = dma::tile_bd(
            ba[slot],
            (bb / 4) as u32,
            BdLocks {
                acq: Some((gemm_core::B_EMPTY[slot], -1)),
                rel: Some((gemm_core::B_FULL[slot], 1)),
            },
            Some(gemm_core::B_BD[1 - slot]),
        );
    }
    if shift.is_some() || direct {
        let (addr, bytes) = if direct {
            (bf16::OUT, bf16::OUT_BYTES)
        } else {
            (COUT_ADDR, COUT_BYTES)
        };
        bds[gemm_core::C_BD as usize] = dma::tile_bd(
            addr,
            (bytes / 4) as u32,
            BdLocks {
                acq: Some((gemm_core::C_FULL, -1)),
                rel: Some((gemm_core::C_EMPTY, 1)),
            },
            Some(gemm_core::C_BD),
        );
        (bds, 5)
    } else {
        let words = (C_STRIPE_BYTES / 4) as u32;
        bds[C_STRIPE_BD[0] as usize] = dma::tile_bd(
            C_ADDR,
            words,
            BdLocks {
                acq: Some((gemm_core::C_FULL, -1)),
                rel: None,
            },
            Some(C_STRIPE_BD[1]),
        );
        bds[C_STRIPE_BD[1] as usize] = dma::tile_bd(
            C_ADDR + C_STRIPE_BYTES as u32,
            words,
            BdLocks {
                acq: None,
                rel: Some((gemm_core::C_EMPTY, 1)),
            },
            Some(C_STRIPE_BD[0]),
        );
        (bds, 6)
    }
}

/// Same lock numbering and initial values as `gemm_core` (`C_EMPTY` starts at 1, `PEER_*` at 0).
pub fn initial_locks() -> [i32; 16] {
    gemm_core::initial_locks_pair()
}

fn program_bf16(kc: usize, tiles: usize, role: PairRole, ctl: Control) -> Program {
    assert_eq!(ctl.probe().layout, 0);
    assert!(
        !ctl.probe().no_compute && ctl.probe().repeat <= 1,
        "BF16 requires ordinary GEMM compute"
    );
    let layout = PairLayout {
        a: bf16::A,
        b: bf16::B,
        c: C_ADDR,
        cout: bf16::OUT,
    };
    let mut a = Assembler::default();
    a.serial = ctl.probe().serial;
    for (r, v) in [
        (0, -1i32 as u32),
        (1, 1),
        (2, SIGNED_8X8),
        (3, tiles as u32),
        (4, 0),
    ] {
        a.mov(0, r, v);
    }
    bf16::initialize(&mut a);
    let wave = a.label();
    let chunks = a.label();
    let captured = a.label();
    let pong = a.label();
    let copied = a.label();
    a.bind(wave);
    for (m, value) in MODIFIERS.into_iter().enumerate() {
        a.mov(2, m, value as u32);
    }
    a.lock(true, gemm_core::C_EMPTY);
    bf16::zero(&mut a);
    a.mov(0, 5, kc as u32);
    a.bind(chunks);
    pair_acquire(&mut a, role, layout);
    chunk(&mut a, false);
    a.emit(&[instruction!(
        "ADD_add_r_ri",
        &[("d0", 11), ("s0", 5), ("imm", 0x7f)]
    )]);
    a.nop(7);
    a.branch(captured, Some((11, true)));
    a.delay();
    a.branch(pong, Some((4, true)));
    a.delay();
    bf16::capture(&mut a, 0);
    a.branch(copied, None);
    a.delay();
    a.bind(pong);
    bf16::capture(&mut a, 1);
    a.bind(copied);
    a.bind(captured);
    pair_release(&mut a, role);
    a.branch(chunks, Some((5, true)));
    a.delay();
    bf16::epilogue(&mut a);
    a.lock(false, gemm_core::C_FULL);
    a.add(3, -1);
    a.branch(wave, Some((3, true)));
    a.delay();
    a.emit(&[instruction!("DONE", &[])]);
    a.finish()
}

/// Bytes of one N=512 BF16 core tile as it reaches memtile / host (128 x 64 x 2 B).
pub const fn bf16_n64_tile_bytes() -> usize {
    bf16::NARROW.out_bytes()
}

/// The N=512 / K=6144 BF16 core: the V8 128x64 pair wavefront of [`gemm_core`] (E/O A halves through the peer's data memory,
/// the same 9-step lock protocol and chunk issue schedule as the silicon-proven V8 pair core) with the direct-BF16 integer
/// epilogue narrowed to 64 features ([`bf16::NARROW`]): per wave the 128 x 64 i32 tile is zeroed, accumulated over `kc`
/// chunks (`kc <= gemm_core::MAX_KC`, K up to 10,240), then reduced to BF16 into the upper half of `C` and drained by the
/// core DMA as one 16,384 B tile. The last chunk's sidecars are captured before its input locks are returned.
pub fn program_bf16_n64(kc: usize, tiles: usize, role: PairRole, ctl: Control) -> Program {
    assert_eq!(ctl.probe().layout, 0);
    assert!(
        !ctl.probe().no_compute && ctl.probe().repeat <= 1,
        "BF16 requires ordinary GEMM compute"
    );
    assert!(
        (1..=gemm_core::MAX_KC).contains(&kc) && (1..=256).contains(&tiles),
        "N64 kc {kc} tiles {tiles}"
    );
    let t = bf16::NARROW;
    let layout = PairLayout {
        a: t.a,
        b: t.b,
        c: t.c,
        cout: t.out,
    };
    let mut a = Assembler::default();
    a.serial = ctl.probe().serial;
    for (r, v) in [
        (0, -1i32 as u32),
        (1, 1),
        (2, SIGNED_8X8),
        (3, tiles as u32),
        (4, 0),
    ] {
        a.mov(0, r, v);
    }
    bf16::initialize_for(&mut a, t);
    let wave = a.label();
    let chunks = a.label();
    let captured = a.label();
    let pong = a.label();
    let copied = a.label();
    a.bind(wave);
    for (m, value) in gemm_core::MODIFIERS.into_iter().enumerate() {
        a.mov(2, m, value as u32);
    }
    a.lock(true, gemm_core::C_EMPTY);
    bf16::zero_for(&mut a, t);
    a.mov(0, 5, kc as u32);
    a.bind(chunks);
    pair_acquire(&mut a, role, layout);
    gemm_core::pair_chunk(&mut a, false, t.c);
    a.emit(&[instruction!(
        "ADD_add_r_ri",
        &[("d0", 11), ("s0", 5), ("imm", 0x7f)]
    )]);
    a.nop(7);
    a.branch(captured, Some((11, true)));
    a.delay();
    a.branch(pong, Some((4, true)));
    a.delay();
    bf16::capture_for(&mut a, 0, t);
    a.branch(copied, None);
    a.delay();
    a.bind(pong);
    bf16::capture_for(&mut a, 1, t);
    a.bind(copied);
    a.bind(captured);
    pair_release(&mut a, role);
    a.branch(chunks, Some((5, true)));
    a.delay();
    bf16::epilogue_for(&mut a, t);
    a.lock(false, gemm_core::C_FULL);
    a.add(3, -1);
    a.branch(wave, Some((3, true)));
    a.delay();
    a.emit(&[instruction!("DONE", &[])]);
    a.finish()
}

/// DMA-view descriptors of the N=512 BF16 core: A halves (4608 B) on S2MM0 (BD 0, 1), B (4352 B) on S2MM1 (BD 2, 3) and the
/// 16,384 B BF16 tile on MM2S0 as the self-cyclic BD 4 (five BDs, same ids, channels and locks as [`tile_bds_pair`]).
pub fn tile_bds_pair_n64() -> ([[u32; 6]; 6], usize) {
    let t = bf16::NARROW;
    let mut bds = [[0; 6]; 6];
    for slot in 0..2 {
        bds[gemm_core::A_BD[slot] as usize] = dma::tile_bd(
            t.a[slot],
            (t.a_bytes() / 4) as u32,
            BdLocks {
                acq: Some((gemm_core::A_EMPTY[slot], -1)),
                rel: Some((gemm_core::A_FULL[slot], 1)),
            },
            Some(gemm_core::A_BD[1 - slot]),
        );
        bds[gemm_core::B_BD[slot] as usize] = dma::tile_bd(
            t.b[slot],
            (t.b_bytes() / 4) as u32,
            BdLocks {
                acq: Some((gemm_core::B_EMPTY[slot], -1)),
                rel: Some((gemm_core::B_FULL[slot], 1)),
            },
            Some(gemm_core::B_BD[1 - slot]),
        );
    }
    bds[gemm_core::C_BD as usize] = dma::tile_bd(
        t.out,
        (t.out_bytes() / 4) as u32,
        BdLocks {
            acq: Some((gemm_core::C_FULL, -1)),
            rel: Some((gemm_core::C_EMPTY, 1)),
        },
        Some(gemm_core::C_BD),
    );
    (bds, 5)
}

/// Arithmetic issue cycles per 32-output BF16 pack group (simulator schedule).
pub fn bf16_group_cycles() -> usize {
    bf16::group_cycles()
}

/// Standalone BF16 epilogue diagnostics, following the IEF15 pack probe.
/// I32[32] at offset0, BF16[32] at0x5000; returns Dm/De/Sm/Se offsets.
#[doc(hidden)]
pub fn bf16_pack_debug_program() -> (Vec<u8>, [u32; 4]) {
    bf16::debug_program()
}
