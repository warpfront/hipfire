// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Persistent-ring command streams: one TXN that runs `S` GEMMs, each gated on a ring slot sequence number
//! published by the host / GPU and acknowledged by a DONE line written by the NPU.
//!
//! The protocol (ring BO layout, seq / slot numbering, producer rule) is frozen in the shared spec
//! (`railgun-spec.md`, "Persistent ring protocol"); this module encodes the NPU side for the V9 design
//! ([`gemm_array::design_v9`], `Epilogue::Int8 { shift: 12 }`, `Control::Fast`).
//!
//! # Arguments
//! `args = [A arena (In), B arena (In), C arena (Out), ring (In)]`: arg 3 is the ring BO of
//! [`RingLayout::bytes`] bytes (it is read AND written by the NPU, but `ArgKind::Out` would make a harness overwrite
//! the header, hence `In`). `SlotPlan::{a_off, b_off, c_off}` are byte offsets into args 0 / 1 / 2: every shim DDR patch
//! of run `i` is the vendor patch (`column offset`) plus the run's arena offset. Arena sizes are `max offset + one
//! run's packed bytes`.
//!
//! # Per-run command stream
//! ```text
//! prologue (once)   reset+release memtile(0) S2MM4 and MM2S0; shim(0) S2MM0 token controller id;
//!                   memtile BD Q (self loop, 1 word -> X); memtile BD DQ (16 words <- Y);
//!                   shim BD SENT (1 word <- ring sentinel, DDR patched); shim BD D (16 words, patched per run)
//! run i:  POLL      reset+release memtile S2MM4; WRITE32 X := 0 (never a seq: a stale seq or sentinel left by an earlier
//!                   run/submission can never satisfy the poll); shim BD P (1 word <- ring slot line, DDR patched, next = P) queued on
//!                   shim MM2S0, memtile BD Q queued on S2MM4: the shim re-reads the slot seq word from DDR forever and the
//!                   memtile re-stores it into X. MASKPOLL X == seq(i)                              <- PollSeq(i) site
//!         DRAIN     WRITE32 shim BD P tail word: next = SENT. The loop finishes its current iteration, then SENT reads the
//!                   sentinel word 0xFFFF_FFFF (never a seq) and ends, so MASKPOLL X == SENTINEL proves the last word
//!                   the shim will ever send for this poll has landed in X (stream order); nothing stray can reach the
//!                   B producer queued by the body.
//!         BODY      (GEMM runs) vendor V9 reset+restart body, shim patches advanced by the slot arena offsets, ends with
//!                   the two aggregate C SYNCs. [`persistent_v9_lean`] runs `1..` use the lean body instead (below).
//!         DONE      reset+release memtile MM2S0 (strictly after the C SYNC: every C word has reached DDR, so the stalled
//!                   cyclic C drain chain is no longer needed); 16 WRITE32 build the done line at Y
//!                   (`done_seq` <- DoneSeq(i) site, slot, 'DONE', 0..); shim S2MM0 BD D (token) and memtile MM2S0 BD DQ
//!                   move it as one 64-byte DMA write to the ring done line; SYNC(shim 0 S2MM0).
//! ```
//! `Neg::SkipDone(i)` omits the whole DONE block of run `i` (and its `DoneSeq(i)` patch site).
//!
//! Channel reuse (approved P1 deviation, vendor CTRL registers only): after a V9 run memtile S2MM4 (B producer ring)
//! and MM2S0 (C drain chain) stay active but stalled on locks, so the spare tasks above could never run. They are reset
//! and released (CTRL bit 1, the same register the body uses) before every POLL (S2MM4) and every DONE (MM2S0).
//! The body resets them again at its start, which also kills the leftover self-loop Q.
//!
//! # Latch argument for the single-word tail flip
//! DMA descriptor registers are latched when a BD is loaded (channel start or chain follow), never re-read mid-BD
//! (the simulator models exactly this: `Engine::step` decodes the BD registers only when `current` is empty). P is a
//! single BD whose `next` is itself, so every iteration is a fresh load of P's eight registers. The flip changes ONE
//! register, tail word 7 (`next` P -> SENT; P has no locks, so no other field of that word changes), by ONE 32-bit
//! register write; every other P word is identical before and after. Hence a load sees either the old BD (`next = P`,
//! one more seq-line read) or the new BD (`next = SENT`) and never a mixed state, and the iteration that loaded the new
//! tail is the last one of P. A load that happened before the write keeps its latched `next = P` and so costs exactly one
//! more P iteration, which still reads the slot seq line, never the sentinel. SENT is finite and is the only BD that reads
//! the sentinel, so X can only become SENTINEL after the last word the shim sends: X == SENTINEL implies every earlier
//! word is already in X (single in-order stream, memtile Q stores them in arrival order) and none follows. That the
//! hardware latches the same way is an assumption of the NPU's BD model (aie-rt BD semantics), checked only in the
//! simulator.
//!
//! # Spare ids (derived, see `gemm_array::v9_spare_memtile_bds`, `PAIR_SHIM_BDS_USED`)
//! * shim BDs 0..4 are used by V9 (B, C0, A, C1); P = 4, SENT = 5, D = 6.
//! * memtile BD ids come from the V9 descriptors at the maximal topology (`mw 16, nw 8, kc 54`): even bank used
//!   0..=9 and 14, 15, so the first two free even-bank ids are Q = 10 (S2MM4) and DQ = 11 (MM2S0). Both channels are
//!   even, hence BD < 24 (`Task::write` bank rule); the choice is independent of `mw`, `nw`, `kc`.
//!
//! # Scratch addresses (memtile column 0)
//! X = [`POLL_SCRATCH`] `0x1000` (one word) and Y = [`DONE_SCRATCH`] `0x3000` (64 bytes). The V9 A ring slots are
//! `0x0000` and `0x2000` but each BD moves only `A_HALF_BYTES = 4096` B, so `0x1000..0x2000` and `0x3000..0x4000` are
//! never accessed by any V9 memtile DMA for any `kc`/`mw`/`nw` (the compact C buffers start at `0x4000`, B slots at
//! `0x14000`; cores never address memtile memory). They are therefore dead between GEMMs AND during them: no run
//! can clobber the poll word or the staged done line, and the scratch writes cannot corrupt operands. The unit test
//! `scratch_is_outside_v9_footprint` checks this against the real descriptors. (V10 puts C at `0` and would collide;
//! the ring is V9 only.)
//!
//! # Lean bodies ([`persistent_v9_lean`])
//! Contract: run 0 of every submission is the FULL vendor body (valid right after the PDI load and after any completed
//! submission); run `i >= 1` is `ArrayDesign::append_lean_run_body`, which assumes the array was left by a COMPLETED
//! run of the same design: previous body + DONE of the same submission (its DONE SYNC returned strictly after the C
//! SYNC) and then this run's POLL / DRAIN. POLL / DONE clobber exactly two memtile column-0 channels the lean plan
//! believes are parked at their first BD: S2MM4 (reset, then left running the self-loop Q that stalls for stream data)
//! and MM2S0 (reset, one 64-byte DQ transfer, idle). They are passed as forced repairs
//! (`ArrayDesign::lean_insts_requeue` semantics): CTRL reset + release, the `B_EMPTY[0]` credit lock rewritten to its
//! initial value (S2MM4 only; the reset dropped a lookahead credit the hardware may or may not have taken), and requeue
//! at BD 4 / BD 6. A channel the shape's own parity plan already resets (S2MM4 for odd `NW`, MM2S0 for odd `waves`)
//! is deduplicated and keeps the plan's repair, so forced ops are only emitted where they are needed. MM2S0 has no
//! credit lock and no iteration `current` (FULL locks are unacquired when parked). The result is a state equal to
//! the canonical start state of the full body, so the end state after every run is identical to a full ring
//! (`Config::state_snapshot`). Other channels are not touched by POLL / DONE (Q / DQ / X / Y use spare BDs and dead
//! memtile bytes); shim BDs P / SENT / D are disjoint from the four shim BDs the body rewrites. `Neg::SkipLeanRequeue(i)`
//! drops the forced ops of lean run `i` only. The patch inventory is unchanged: poll `PollSeq(i)` + `DoneSeq(i)` per
//! run, exactly 2 sites per run for every run, lean or full.
//!
//! Bytes: every op is a fixed number of u32 words (`write32` 24 B, `mask_write` / `mask_poll` 28 B, shim / memtile
//! `block_write` 48 B, `ddr_patch` 48 B, `sync` 16 B). Lean bytes/run = 856 (POLL + DRAIN 304, DONE 552) + lean body,
//! with lean body = 7,776 (32 cores x (halt 28 + release 28 + PC 24 + enable 28), shim 8 x 536, SYNC 32)
//! + per-parity repairs, per column unless stated: odd `kc*waves` 8,448 (cores) + 184 x 8; odd `NW` 8 x (184 + 28 `MW`);
//! odd `waves` 8 x 576; forced S2MM4 (NW even) 104 and MM2S0 (waves even) 80 at column 0 only. The same model gives the
//! measured lean sizes 7,792 B and 11,056 B (header 16 B) and the measured 856 B protocol-only run.
//!
//! # Patch inventory
//! Exactly the value word of each poll MASKPOLL (`PollSeq(i)`) and of the done-seq WRITE32 (`DoneSeq(i)`), both
//! u32 word indices counted from the first byte of `insts` (header included). Run `i` of a submission starting at
//! `seq0` uses `seq0 + i`. 2 sites per run, 1 for a `SkipDone` run's poll only. Everything else, including the slot index,
//! offsets and ring lines, is fixed by the build.
//!
//! # Instruction footprint
//! Each submission has a 396-byte header/prologue plus the following bytes per run (16 sequence patch words for S=8):
//! | V9 shape | Bytes/run | Bytes, S=8 |
//! | --- | ---: | ---: |
//! | 512x512x64 | 49,208 | 394,060 |
//! | 4096x1280x2560 | 51,896 | 415,564 |
//! | 4096x2560x640 | 51,896 | 415,564 |
//! | Empty echo | 856 | 7,244 |
//! These are measured encoder lengths, not hardware latency measurements. Slot offsets do not affect encoded length.
//!
//! [`persistent_v9_lean`] keeps run 0 (full) and replaces runs `1..` by the lean body plus the forced col-0 repairs
//! (see "Lean bodies" for the byte method, which reproduces the two measured lean TXN sizes of the lean module,
//! 7,792 B and 11,056 B). "Measured" rows are encoder lengths of the built `insts` (`S=8`, run 0 full, 7 lean);
//! "derived" rows come from the op-size model only:
//! | V9 shape | Lean bytes/run | Bytes, S=8 lean | Bytes, S=8 full | Source |
//! | --- | ---: | ---: | ---: | --- |
//! | 512x512x64 | 24,856 | 223,596 | 394,060 | derived (full row measured above) |
//! | 4096x1280x2560 | 11,976 | 136,124 | 415,564 | measured |
//! | 4096x2560x640 | 11,976 | 136,124 | 415,564 | measured |
//! | 512x512x128 | 14,936 | 154,156 | 394,060 | measured |
//! | 1024x512x128 | 10,632 | 396 + full + 7 x 10,632 | | derived |
//! | 512x1024x128 (all parities even) | 8,816 | 396 + full + 7 x 8,816 | | derived |
//! The 16 sequence patch words of S=8 are unchanged (2 per run); `patch_seq` output equals a fresh build.
use super::gemm_array::{
    self, ArrayDesign, MEMTILE_DMA_BASE, PAIR_SHIM_BDS_USED, V8_DEFAULT_EPILOGUE,
};
use super::gemm_core::Control;
use super::gemm_i8::{ArgKind, ArgSpec};
use crate::{
    dma::{
        Bd,
        Direction::{Mm2s, S2mm},
        Location, Task,
    },
    regs,
    txn::Txn,
};

/// `'RGNP'` little endian.
pub const RING_MAGIC: u32 = 0x504e_4752;
pub const RING_VERSION: u32 = 1;
/// `'DONE'` little endian.
pub const DONE_MAGIC: u32 = 0x454e_4f44;
/// The NPU-read constant at ring byte 16; never a valid seq.
pub const SENTINEL: u32 = 0xFFFF_FFFF;
const LINE: usize = 64;
/// Ring BO is argument 3 of every persistent design.
pub const RING_ARG: u32 = 3;
/// Memtile (column 0) byte offset of the poll word X and of the staged done line Y (see the module header).
pub const POLL_SCRATCH: u32 = 0x1000;
pub const DONE_SCRATCH: u32 = 0x3000;
/// Shim (column 0) BDs: poll self loop, sentinel drain, done line.
const SHIM_BD_P: u32 = PAIR_SHIM_BDS_USED;
const SHIM_BD_SENT: u32 = PAIR_SHIM_BDS_USED + 1;
const SHIM_BD_D: u32 = PAIR_SHIM_BDS_USED + 2;
/// Memtile channels reused after a run: S2MM4 (poll in) and MM2S0 (done out).
const POLL_CH: u32 = 4;
const DONE_CH: u32 = 0;
const DONE_WORDS: u32 = (LINE / 4) as u32;
/// Channels the POLL / DONE tasks leave in a state a lean body does not expect: `(column, direction, channel)`.
const LEAN_FORCED: [(u32, crate::dma::Direction, u32); 2] =
    [(0, S2mm, POLL_CH), (0, Mm2s, DONE_CH)];

// Scratch placement vs the V9 memtile layout (A slots at 0x0000 / 0x2000 of `A_HALF_BYTES` each, compact C from 0x4000):
// X lies in the gap after A slot 0, Y (64 B) in the gap after A slot 1. `scratch_is_outside_v9_footprint` repeats the
// check against the real descriptors at several shapes including the maximal one.
const _: () =
    assert!(POLL_SCRATCH as usize >= gemm_array::A_HALF_BYTES && POLL_SCRATCH + 4 <= 0x2000);
const _: () = assert!(
    DONE_SCRATCH as usize >= 0x2000 + gemm_array::A_HALF_BYTES
        && DONE_SCRATCH + LINE as u32 <= 0x4000
);
const _: () = assert!(DONE_SCRATCH % LINE as u32 == 0 && POLL_SCRATCH % 4 == 0);

/// Ring geometry: `nslots` (a power of two) slot lines and `nslots` done lines behind one header line.
#[derive(Clone, Copy, Debug)]
pub struct RingLayout {
    pub nslots: usize,
}

impl RingLayout {
    fn check(&self) {
        assert!(
            self.nslots.is_power_of_two(),
            "ring nslots {} must be a power of two",
            self.nslots
        );
    }
    pub fn slot_line(&self, slot: usize) -> usize {
        assert!(slot < self.nslots, "slot {slot} >= nslots {}", self.nslots);
        LINE * (1 + slot)
    }
    pub fn done_line(&self, slot: usize) -> usize {
        assert!(slot < self.nslots, "slot {slot} >= nslots {}", self.nslots);
        LINE * (1 + self.nslots + slot)
    }
    /// Byte offset of the sentinel word in the header line.
    pub fn sentinel(&self) -> usize {
        16
    }
    pub fn bytes(&self) -> usize {
        LINE * (1 + 2 * self.nslots)
    }
    /// Zero the whole ring and write the header (`magic, version, nslots, slot_line_bytes, sentinel`).
    pub fn initialize(&self, bytes: &mut [u8]) {
        self.check();
        assert!(
            bytes.len() >= self.bytes(),
            "ring buffer {} B < {} B",
            bytes.len(),
            self.bytes()
        );
        bytes[..self.bytes()].fill(0);
        for (i, v) in [
            RING_MAGIC,
            RING_VERSION,
            self.nslots as u32,
            LINE as u32,
            SENTINEL,
        ]
        .into_iter()
        .enumerate()
        {
            bytes[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
}

/// One run of a submission: ring slot and arena byte offsets (args 0 / 1 / 2) of its operands / result.
#[derive(Clone, Copy, Debug)]
pub struct SlotPlan {
    pub slot: usize,
    pub a_off: u64,
    pub b_off: u64,
    pub c_off: u64,
}

/// Patchable value word of run `i` (run index inside the submission).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchKind {
    PollSeq(usize),
    DoneSeq(usize),
}

/// Negative-test builders. `SkipDone(i)`: run `i` never publishes its DONE line. `SkipSeqPatch` is a HOST-side control
/// (the host omits [`patch_seq`]); the builder output is the normal one. `SkipLeanRequeue(i)`
/// ([`persistent_v9_lean`] only, `1 <= i < runs`): lean run `i` omits the forced col0 S2MM4 / MM2S0 reset + lock + requeue
/// repairs (channels its own parity plan already repairs are unaffected), every other run is normal.
#[derive(Clone, Copy, Debug)]
pub enum Neg {
    SkipDone(usize),
    SkipSeqPatch,
    SkipLeanRequeue(usize),
}

pub struct PersistentDesign {
    pub pdi: Vec<u8>,
    pub insts: Vec<u8>,
    /// `(u32 word index in insts, kind)`.
    pub patch_sites: Vec<(usize, PatchKind)>,
    /// `[A arena, B arena, C arena, ring]`.
    pub args: Vec<ArgSpec>,
    design: ArrayDesign,
}

impl PersistentDesign {
    /// Packed `[A, B]` of ONE run (copy to arg0 at `a_off` / arg1 at `b_off`); see [`ArrayDesign::pack_in`].
    pub fn pack_in(&self, a: &[i8], b: &[i8]) -> [Vec<u8>; 2] {
        self.design.pack_in(a, b)
    }
    /// Row-major `m*n` result of ONE run from its `c_bytes()` slice of arg2 (starting at `c_off`).
    pub fn unpack_out(&self, c_run: &[u8]) -> Vec<i32> {
        self.design.unpack_out(c_run)
    }
    /// Exact CPU result in the form [`PersistentDesign::unpack_out`] returns.
    pub fn reference(&self, a: &[i8], b: &[i8]) -> Vec<i32> {
        self.design.reference(a, b)
    }
    /// Packed bytes of ONE run in args 0 / 1 / 2.
    pub fn a_bytes(&self) -> usize {
        self.design.args[0].bytes
    }
    pub fn b_bytes(&self) -> usize {
        self.design.args[1].bytes
    }
    pub fn c_bytes(&self) -> usize {
        self.design.args[2].bytes
    }
}

/// Ring-gated V9 GEMM runs: `slots.len()` runs of the `m x n x k` GEMM, run `i` waiting for `seq0 + i`.
pub fn persistent_v9(
    m: usize,
    n: usize,
    k: usize,
    ring: RingLayout,
    slots: &[SlotPlan],
    seq0: u32,
    neg: Option<Neg>,
) -> PersistentDesign {
    build(
        gemm_array::design_v9(m, n, k, V8_DEFAULT_EPILOGUE, Control::Fast),
        Some((Bodies::Full, false)),
        ring,
        slots,
        seq0,
        neg,
    )
}

/// [`persistent_v9`] with the same arguments, operand offsets, patch inventory (exactly [`PatchKind::PollSeq`] and
/// [`PatchKind::DoneSeq`] per run) and ring protocol, but run 0 of the submission keeps the FULL vendor body and runs
/// `1..` use the lean body ([`ArrayDesign::append_lean_run_body`]) with the memtile channels that the POLL / DONE tasks
/// clobber (column 0 S2MM4 and MM2S0) forced through reset + requeue (module documentation, "Lean bodies").
/// Every submission built by this function starts with a full body, so it is valid for the first submit after the PDI
/// and for any submit after a completed submit of the same design; [`persistent_with`] builds the lean-first form.
/// `Neg::SkipLeanRequeue(i)` (1 <= i < runs) omits only the forced repairs of lean run `i`.
pub fn persistent_v9_lean(
    m: usize,
    n: usize,
    k: usize,
    ring: RingLayout,
    slots: &[SlotPlan],
    seq0: u32,
    neg: Option<Neg>,
) -> PersistentDesign {
    build(
        gemm_array::design_v9(m, n, k, V8_DEFAULT_EPILOGUE, Control::Fast),
        Some((Bodies::Lean, false)),
        ring,
        slots,
        seq0,
        neg,
    )
}

/// Which body each run of a persistent V9 submission executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bodies {
    /// Every run the full body ([`persistent_v9`]).
    Full,
    /// Run 0 full, runs `1..` lean ([`persistent_v9_lean`]).
    Lean,
    /// Every run lean, run 0 too: the re-arm form, valid ONLY directly after a completed lean submission of the same
    /// design on the same hardware context (the array is still configured and every channel is where a completed lean
    /// run leaves it). Saves one full re-initialisation of the array per re-armed submission.
    LeanFirst,
}

/// Persistent ring of an explicit V9 or G80 `design` (e.g. [`ArrayDesign::with_a_repeat`]); same patch inventory and ring
/// protocol as [`persistent_v9`]. `b_shared` (G80, lean bodies only): every run reads the same B bytes (one `b_off` for
/// all slots) and the lean runs keep B resident in the memtile instead of refilling it
/// ([`ArrayDesign::append_lean_run_body_b_resident`]); a `Bodies::LeanFirst` submission is then valid only after one
/// whose B (at that offset) held the same bytes.
pub fn persistent_with(
    design: ArrayDesign,
    ring: RingLayout,
    slots: &[SlotPlan],
    seq0: u32,
    bodies: Bodies,
    b_shared: bool,
) -> PersistentDesign {
    if b_shared {
        assert_eq!(
            design.variant,
            gemm_array::Variant::G80,
            "B-resident lean bodies are derived for G80 only"
        );
        assert!(bodies != Bodies::Full, "B-resident needs lean bodies");
        assert!(
            slots.iter().all(|s| s.b_off == slots[0].b_off),
            "b_shared needs one B offset for every run"
        );
    }
    assert!(
        matches!(
            design.variant,
            gemm_array::Variant::V9 | gemm_array::Variant::G80
        ),
        "the ring is derived for the V9 / G80 pair designs, not {}",
        design.variant
    );
    build(design, Some((bodies, b_shared)), ring, slots, seq0, None)
}

/// Poll + DONE only (no GEMM, no body) on the `512x512x64` V9 image: measures the protocol.
pub fn empty_persistent(ring: RingLayout, slots: &[SlotPlan], seq0: u32) -> PersistentDesign {
    build(
        gemm_array::design_v9(512, 512, 64, V8_DEFAULT_EPILOGUE, Control::Fast),
        None,
        ring,
        slots,
        seq0,
        None,
    )
}

fn check_seq_range(seq0: u32, runs: usize) {
    assert!(
        runs > 0 && seq0 >= 1 && seq0 as u64 + runs as u64 - 1 < SENTINEL as u64,
        "seq range {seq0}+{runs} must stay inside 1..={}",
        SENTINEL - 1
    );
}

fn ctrl_addr(loc: Location, direction: crate::dma::Direction, channel: u32) -> u32 {
    Task {
        direction,
        channel,
        bd: 0,
        repeat: 1,
        issue_token: false,
    }
    .write(loc)
    .0 - 4
}

/// CTRL bit 1 reset then release (the body's own mechanism) of a memtile channel.
fn reset_release(txn: &mut Txn, loc: Location, direction: crate::dma::Direction, channel: u32) {
    assert!(channel % 2 == 0, "ctrl_addr assumes the even-bank BD 0");
    let ctrl = ctrl_addr(loc, direction, channel);
    txn.mask_write(ctrl, 2, 2);
    txn.mask_write(ctrl, 0, 2);
}

/// `body`: `None` = protocol only (no GEMM body); else the bodies and whether lean runs keep a shared B resident.
fn build(
    mut design: ArrayDesign,
    body: Option<(Bodies, bool)>,
    ring: RingLayout,
    slots: &[SlotPlan],
    seq0: u32,
    neg: Option<Neg>,
) -> PersistentDesign {
    ring.check();
    assert!(
        !slots.is_empty(),
        "a persistent submission needs at least one run"
    );
    check_seq_range(seq0, slots.len());
    if let Some(Neg::SkipDone(i)) = neg {
        assert!(i < slots.len(), "SkipDone({i}) but {} runs", slots.len());
    }
    if let Some(Neg::SkipLeanRequeue(i)) = neg {
        assert!(
            matches!(body, Some((Bodies::Lean, _))) && i >= 1 && i < slots.len(),
            "SkipLeanRequeue({i}) needs a lean run: persistent_v9_lean with 1 <= i < {} runs",
            slots.len()
        );
    }
    for s in slots {
        assert!(
            s.slot < ring.nslots,
            "slot {} >= nslots {}",
            s.slot,
            ring.nslots
        );
        assert!(
            s.a_off % 4 == 0 && s.b_off % 4 == 0 && s.c_off % 4 == 0,
            "arena offsets must be word aligned"
        );
    }
    let spare = gemm_array::v9_spare_memtile_bds(false);
    assert!(
        spare.len() >= 2,
        "no spare even-bank memtile BDs: {spare:?}"
    );
    let (q_bd, dq_bd) = (spare[0], spare[1]);
    assert!(
        !design.uses_memtile_bd(q_bd) && !design.uses_memtile_bd(dq_bd),
        "ring spare memtile BDs {q_bd}/{dq_bd} collide with {}",
        design.variant
    );
    assert!(q_bd < 24 && dq_bd < 24);

    let (shim, mem) = (Location::new(0, 0), Location::new(0, 1));
    let shim_bd = |id| Bd::address(shim, id);
    let poll_loop = {
        let mut bd = Bd::new(0, 1);
        bd.next = Some(SHIM_BD_P);
        bd
    };
    // Same BD with `next` = SENT: only the tail word (7) differs.
    let drain_tail = Bd {
        next: Some(SHIM_BD_SENT),
        ..poll_loop
    }
    .shim_words()[7];
    let x_addr = mem.address(POLL_SCRATCH);

    let mut txn = Txn::aie2p_8col();
    let mut sites = Vec::with_capacity(2 * slots.len());

    // Prologue.
    reset_release(&mut txn, mem, S2mm, POLL_CH);
    reset_release(&mut txn, mem, Mm2s, DONE_CH);
    txn.mask_write(shim.address(regs::shim::DMA_S2MM_0_CTRL), 0xf00, 0x1f00);
    let mut q = Bd::new((MEMTILE_DMA_BASE + POLL_SCRATCH) as u64, 1);
    q.next = Some(q_bd);
    q.emit_txn(mem, q_bd, &mut txn);
    Bd::new((MEMTILE_DMA_BASE + DONE_SCRATCH) as u64, DONE_WORDS).emit_txn(mem, dq_bd, &mut txn);
    Bd::new(0, 1).emit_txn(shim, SHIM_BD_SENT, &mut txn);
    txn.ddr_patch(shim_bd(SHIM_BD_SENT) + 4, RING_ARG, ring.sentinel() as u64);
    Bd::new(0, DONE_WORDS).emit_txn(shim, SHIM_BD_D, &mut txn);

    for (i, plan) in slots.iter().enumerate() {
        let seq = seq0 + i as u32;
        // POLL.
        reset_release(&mut txn, mem, S2mm, POLL_CH);
        txn.write32(x_addr, 0);
        poll_loop.emit_txn(shim, SHIM_BD_P, &mut txn);
        txn.ddr_patch(
            shim_bd(SHIM_BD_P) + 4,
            RING_ARG,
            ring.slot_line(plan.slot) as u64,
        );
        Task {
            direction: S2mm,
            channel: POLL_CH,
            bd: q_bd,
            repeat: 1,
            issue_token: false,
        }
        .emit_txn(mem, &mut txn);
        Task {
            direction: Mm2s,
            channel: 0,
            bd: SHIM_BD_P,
            repeat: 1,
            issue_token: false,
        }
        .emit_txn(shim, &mut txn);
        sites.push((
            txn.next_op_word() + Txn::MASKPOLL_VALUE_WORD,
            PatchKind::PollSeq(i),
        ));
        txn.mask_poll(x_addr, seq, 0xFFFF_FFFF);
        // DRAIN.
        txn.write32(shim_bd(SHIM_BD_P) + 4 * 7, drain_tail);
        txn.mask_poll(x_addr, SENTINEL, 0xFFFF_FFFF);
        // BODY.
        if let Some((body, b_shared)) = body {
            let arena = [plan.a_off, plan.b_off, plan.c_off];
            if body == Bodies::LeanFirst || (body == Bodies::Lean && i > 0) {
                let forced: &[(u32, crate::dma::Direction, u32)] = if matches!(neg, Some(Neg::SkipLeanRequeue(skip)) if skip == i)
                {
                    &[]
                } else {
                    &LEAN_FORCED
                };
                let lean = if b_shared {
                    ArrayDesign::append_lean_run_body_b_resident
                } else {
                    ArrayDesign::append_lean_run_body_same_design
                };
                lean(&design, &mut txn, arena, forced)
                    .unwrap_or_else(|e| panic!("lean body of run {i}: {e}"));
            } else {
                design.append_run_body(&mut txn, arena);
            }
        }
        // DONE.
        if matches!(neg, Some(Neg::SkipDone(skip)) if skip == i) {
            continue;
        }
        reset_release(&mut txn, mem, Mm2s, DONE_CH);
        for w in 0..DONE_WORDS {
            let value = match w {
                0 => seq,
                1 => plan.slot as u32,
                2 => DONE_MAGIC,
                _ => 0,
            };
            if w == 0 {
                sites.push((
                    txn.next_op_word() + Txn::WRITE32_VALUE_WORD,
                    PatchKind::DoneSeq(i),
                ));
            }
            txn.write32(mem.address(DONE_SCRATCH + 4 * w), value);
        }
        txn.ddr_patch(
            shim_bd(SHIM_BD_D) + 4,
            RING_ARG,
            ring.done_line(plan.slot) as u64,
        );
        Task {
            direction: S2mm,
            channel: 0,
            bd: SHIM_BD_D,
            repeat: 1,
            issue_token: true,
        }
        .emit_txn(shim, &mut txn);
        Task {
            direction: Mm2s,
            channel: DONE_CH,
            bd: dq_bd,
            repeat: 1,
            issue_token: false,
        }
        .emit_txn(mem, &mut txn);
        txn.sync(0, 0, 0, 0, 1, 1);
    }

    let arena = |bytes: usize, off: fn(&SlotPlan) -> u64| {
        let hi = slots
            .iter()
            .map(|s| usize::try_from(off(s)).expect("arena offset"))
            .max()
            .unwrap();
        hi.checked_add(bytes).expect("arena size")
    };
    let [a, b, c] = [0, 1, 2].map(|i| design.args[i].bytes);
    let args = vec![
        ArgSpec {
            bytes: arena(a, |s| s.a_off),
            kind: ArgKind::In,
        },
        ArgSpec {
            bytes: arena(b, |s| s.b_off),
            kind: ArgKind::In,
        },
        ArgSpec {
            bytes: arena(c, |s| s.c_off),
            kind: ArgKind::Out,
        },
        ArgSpec {
            bytes: ring.bytes(),
            kind: ArgKind::In,
        },
    ];
    PersistentDesign {
        pdi: std::mem::take(&mut design.pdi),
        insts: txn.to_bytes(),
        patch_sites: sites,
        args,
        design,
    }
}

/// Rewrite the declared value words for a submission whose first run has sequence `new_seq0`: `PollSeq(i)` and
/// `DoneSeq(i)` become `new_seq0 + i`. No other byte of `insts` is touched.
pub fn patch_seq(insts: &mut [u8], sites: &[(usize, PatchKind)], new_seq0: u32) {
    let runs = sites
        .iter()
        .map(|&(_, kind)| match kind {
            PatchKind::PollSeq(i) | PatchKind::DoneSeq(i) => i + 1,
        })
        .max()
        .unwrap_or(0);
    if runs == 0 {
        return;
    }
    check_seq_range(new_seq0, runs);
    for &(word, kind) in sites {
        let i = match kind {
            PatchKind::PollSeq(i) | PatchKind::DoneSeq(i) => i,
        };
        let at = word.checked_mul(4).expect("site");
        let dst = insts
            .get_mut(at..at + 4)
            .unwrap_or_else(|| panic!("patch site word {word} outside insts"));
        dst.copy_from_slice(&(new_seq0 + i as u32).to_le_bytes());
    }
}

/// Producer rule for GLOBAL run index `run_index` (seq `run_index + 1`, slot `run_index % nslots`): the first
/// `nslots` runs may always publish, later ones only once `done[s].done_seq` equals the seq of run `run_index - nslots`.
pub fn producer_may_publish(ring_bytes: &[u8], layout: RingLayout, run_index: usize) -> bool {
    layout.check();
    if run_index < layout.nslots {
        return true;
    }
    let at = layout.done_line(run_index % layout.nslots);
    let done = u32::from_le_bytes(ring_bytes[at..at + 4].try_into().unwrap());
    done == ((run_index - layout.nslots) as u32).wrapping_add(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(b: &[u8]) -> Vec<u32> {
        b.chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    fn plans(n: usize, nslots: usize) -> Vec<SlotPlan> {
        (0..n)
            .map(|i| SlotPlan {
                slot: i % nslots,
                a_off: 4096 * i as u64,
                b_off: 0,
                c_off: 8192 * i as u64,
            })
            .collect()
    }

    #[test]
    fn ring_layout_and_header() {
        let r = RingLayout { nslots: 4 };
        assert_eq!(
            (
                r.slot_line(0),
                r.slot_line(3),
                r.done_line(0),
                r.done_line(3),
                r.sentinel(),
                r.bytes()
            ),
            (64, 256, 320, 512, 16, 576)
        );
        let mut b = vec![0xAAu8; r.bytes() + 8];
        r.initialize(&mut b);
        let w = words(&b[..20]);
        assert_eq!(w, [RING_MAGIC, 1, 4, 64, SENTINEL]);
        assert!(
            b[20..r.bytes()].iter().all(|&x| x == 0) && b[r.bytes()..].iter().all(|&x| x == 0xAA)
        );
    }

    #[test]
    fn producer_rule_is_wrap_safe_equality() {
        let r = RingLayout { nslots: 2 };
        let mut b = vec![0u8; r.bytes()];
        r.initialize(&mut b);
        assert!(producer_may_publish(&b, r, 0) && producer_may_publish(&b, r, 1));
        assert!(!producer_may_publish(&b, r, 2));
        // run 2 reuses slot 0 and needs done[0] == seq(0) == 1.
        b[r.done_line(0)..r.done_line(0) + 4].copy_from_slice(&1u32.to_le_bytes());
        assert!(producer_may_publish(&b, r, 2));
        assert!(!producer_may_publish(&b, r, 3));
        // A later seq in the line (not equal to the awaited one) does not release it.
        b[r.done_line(1)..r.done_line(1) + 4].copy_from_slice(&4u32.to_le_bytes());
        assert!(!producer_may_publish(&b, r, 3));
        b[r.done_line(1)..r.done_line(1) + 4].copy_from_slice(&2u32.to_le_bytes());
        assert!(producer_may_publish(&b, r, 3));
    }

    /// Patching a build to a new seq0 equals a fresh build, and exactly the declared words differ.
    #[test]
    fn patch_seq_matches_fresh_build_and_inventory_is_exact() {
        let r = RingLayout { nslots: 2 };
        let p = plans(3, 2);
        let a = empty_persistent(r, &p, 1);
        let b = empty_persistent(r, &p, 40);
        assert_eq!(a.patch_sites.len(), 6);
        assert_eq!(a.patch_sites, b.patch_sites);
        let (wa, wb) = (words(&a.insts), words(&b.insts));
        let diff: Vec<usize> = (0..wa.len()).filter(|&i| wa[i] != wb[i]).collect();
        let mut declared: Vec<usize> = a.patch_sites.iter().map(|s| s.0).collect();
        declared.sort_unstable();
        assert_eq!(diff, declared);
        let mut patched = a.insts.clone();
        patch_seq(&mut patched, &a.patch_sites, 40);
        assert_eq!(patched, b.insts);
        for &(w, kind) in &a.patch_sites {
            let i = match kind {
                PatchKind::PollSeq(i) | PatchKind::DoneSeq(i) => i,
            };
            assert_eq!(wa[w], 1 + i as u32);
            assert_eq!(wb[w], 40 + i as u32);
        }
    }

    #[test]
    fn skip_done_drops_only_that_runs_done_site() {
        let r = RingLayout { nslots: 2 };
        let p = plans(3, 2);
        let full = empty_persistent(r, &p, 1);
        let skip = build(
            gemm_array::design_v9(512, 512, 64, V8_DEFAULT_EPILOGUE, Control::Fast),
            None,
            r,
            &p,
            1,
            Some(Neg::SkipDone(1)),
        );
        assert_eq!(skip.patch_sites.len(), 5);
        assert!(!skip
            .patch_sites
            .iter()
            .any(|s| s.1 == PatchKind::DoneSeq(1)));
        assert!(skip
            .patch_sites
            .iter()
            .any(|s| s.1 == PatchKind::PollSeq(1)));
        // SkipSeqPatch is host-only: the builder output is the normal one.
        let host = build(
            gemm_array::design_v9(512, 512, 64, V8_DEFAULT_EPILOGUE, Control::Fast),
            None,
            r,
            &p,
            1,
            Some(Neg::SkipSeqPatch),
        );
        assert_eq!(
            (host.insts.clone(), host.patch_sites.clone()),
            (full.insts.clone(), full.patch_sites.clone())
        );
    }

    /// Public arena allocation: each arena holds the largest slot offset plus one run's packed bytes, ring arg last.
    #[test]
    fn arenas_cover_offsets() {
        let r = RingLayout { nslots: 2 };
        let p2 = persistent_v9(512, 512, 64, r, &plans(2, 2), 1, None);
        // Args: arenas hold the largest offset plus one run, ring arg last.
        let (a, b, c) = (p2.a_bytes(), p2.b_bytes(), p2.c_bytes());
        assert_eq!(
            p2.args.iter().map(|s| s.bytes).collect::<Vec<_>>(),
            [4096 + a, b, 8192 + c, r.bytes()]
        );
        assert_eq!(
            p2.args.iter().map(|s| s.kind).collect::<Vec<_>>(),
            [ArgKind::In, ArgKind::In, ArgKind::Out, ArgKind::In]
        );
        assert!(p2.pdi.len() > 0);
    }

    #[test]
    fn scratch_is_outside_v9_footprint() {
        for (kc, mw, nw) in [(1, 1, 1), (10, 8, 5), (40, 8, 3), (54, 16, 8)] {
            for range in gemm_array::v9_memtile_footprint(kc, mw, nw) {
                for (at, len) in [(POLL_SCRATCH, 4), (DONE_SCRATCH, 64)] {
                    assert!(
                        range.end <= at || range.start >= at + len,
                        "{range:?} overlaps {at:#x} (kc {kc} mw {mw} nw {nw})"
                    );
                }
            }
        }
        assert_eq!(DONE_SCRATCH % 64, 0);
        // The selected memtile BDs are the first two ids no V9 descriptor of the maximal topology uses (even bank),
        // the selected shim BDs sit above the four V9 uses.
        let even = gemm_array::v9_spare_memtile_bds(false);
        assert!(even.len() >= 2 && even.iter().all(|&id| id < 24));
        assert!(
            SHIM_BD_P >= PAIR_SHIM_BDS_USED
                && SHIM_BD_SENT > SHIM_BD_P
                && SHIM_BD_D > SHIM_BD_SENT
                && SHIM_BD_D < 16
        );
        assert_eq!(PAIR_SHIM_BDS_USED, 4);
    }
}
