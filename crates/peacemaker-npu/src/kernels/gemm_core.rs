// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// AIE2P instruction fields and scheduling are derived from Xilinx/llvm-aie
// AIE2PGenInstrInfo.td and AIE2PGenSchedule.td (Apache-2.0 WITH LLVM-exception).
//! Whole-array core: resident 128x64 i32 output, streamed K64 input chunks.
//! Four accumulator streams form a wavefront: each holds one 8x8 C block for
//! all eight reduction steps, then overlaps its drain with the next block's
//! loads. Paired N blocks reuse A vectors; timed B loads stream through x11.
//! The steady state issues one matrix op per bundle. Counts are simulator
//! issue cycles, not measured NPU cycles; DMA/fetch/bank stalls are excluded.
//! Offline gates: `cargo test -p peacemaker-npu gemm_core` and
//! `cargo test -p peacemaker-npu gemm_core`. The aligned program is 14,048 bytes
//! for kc=40/tiles=256; measured compute regions are 1,070 first-chunk and
//! 1,067 later-chunk issue cycles, with a 1,008-cycle consecutive VMAC run.
//! `program_variant` also exposes Serial (2,000 bytes, no ZOL/co-issue),
//! ClockProbe (`program_probe`: LockOnly plus an N-iteration, `clock_probe_cycles_per_iter()`-cycle delay loop
//! before each C-full release, for measuring the AIE clock; program size independent of N),
//! LockOnly (688 bytes, scalar pattern in the first 64 packed C words),
//! and FastSlowCtl (Fast's `chunk()` compute with Serial control discipline: seven-NOP
//! lock/branch/move gaps, NOP-filled branch delay slots). The Fast-core silicon failure was unaligned branch
//! targets (fixed in `Assembler::bind`); every program is checked against `isa::rules` when finished.
//! V8 pair core (`program_pair`): the Fast wavefront with E/O A addressing (p0 walks the even-mb half, p1 the
//! odd-mb half, 512 B per mb_local, so the group%4==3 A step is absent), each core of a vertical pair reading
//! the other's half through the South/North data-memory view, the 9-step per-chunk `PEER_FULL`/`PEER_EMPTY`
//! lock protocol, and an optional int8 `VST.SRS.4x` C epilogue into `COUT_ADDR`. Lane order of the dm
//! accumulator vs the four bm quarters (dm lane 16q+l = bm q lane l) is the simulator's modelling assumption;
//! it is not yet confirmed on silicon. Gates: `cargo test -p peacemaker-npu gemm_core` and the two-core
//! `cargo test -p peacemaker-npu --test gemm_core_pair`.
//!
//! ## Pair data-memory layouts
//! `--variant V8|V9 --ctl-probe layout=N` selects the maps in [`pair_layout`]; add `serial,` for Slow control.
//! Layout alone preserves GEMM verification (`Probe::timing_only` remains false). All four maps keep contiguous C
//! blocks, unchanged `MODIFIERS`, and the same instruction schedule/issue-cycle constants. **Layout 0 is the default**
//! (C at 0x8000, Cout at 0x6000): measured on silicon (window 2026-10-02 21:56 UTC, hipx `probe5-02f7cbd.log`) it cut the
//! pair chunk compute from ~1530 to ~1360 cycles (repeat-slope) and raised V9 gate_up from 22.7 to 25.2 TOPS; layout 1 is
//! the original V6 map (C 0x6000, Cout 0xE000). In the model table below the rows are numbered by the current layout ids.
//!
//! AIE-ML DM has eight **single-port**, 512-word x 128-bit banks (8 KiB each), with an even/odd
//! interleaved pair forming a 256-bit double bank:
//! [UG1603, AI Engine-ML Memory](https://docs.amd.com/r/2024.1-English/ug1603-ai-engine-ml-kernel-graph/AI-Engine-ML-Memory).
//! Four contiguous 16 KiB placement windows are used by mlir-aie (`AIETargetModel.h:960-962`,
//! `AIETargetShared.cpp:248-286`) and LLVM A-D annotations (`aie2/AIE2AddrSpace.h:25-111`;
//! `aie2p/AIE2PSubtarget.h:36-38`). Thus `logical_bank = (own_offset >> 14)`.
//! [AM020, Load and Store Unit](https://docs.amd.com/r/en-US/am020-versal-aie-ml/Load-and-Store-Unit)
//! shows even/odd words at 0x0000/0x0010. Combining that with the contiguous double banks gives
//! `physical_bank = 2 * (offset >> 14) + ((offset >> 4) & 1)` (16-byte interleave).
//! [AM020, AIE-ML Memory Module](https://docs.amd.com/r/en-US/am020-versal-aie-ml/AIE-ML-Memory-Module)
//! specifies per-bank **round-robin** arbitration among all requesters: one grant/cycle, other cores
//! or DMA stall for one cycle and retry. There is no fixed core-versus-DMA-versus-neighbour winner.
//! AIE2P confirms eight conflict events (`xaie_events_aie2p.h:236-244`) and reuses LLVM's AIE2 bank
//! model; exact AIE2P 512-bit memory beat occupancy is not documented in the inspected sources.
//! LLVM's load-only scoreboard (`AIEHazardRecognizer.cpp:133-140,790-794`) is not a physical
//! dual-port guarantee: the TRM's single-port model takes precedence for hardware contention.
//!
//! `gemm_core_pair_bank_access_model` traces the actual wavefront at memory stage 5 and prints
//! physical-bank reservation collisions for both cores (512-bit accesses reserve both banks of their
//! double bank at the single LLVM memory stage; beat expansion is excluded). Accumulate-chunk results:
//!
//! | Layout | Peer A / B-C / C load-store collisions | Compute stall slots split/shared | +AB DMA | +AB DMA+Cout |
//! |---|---:|---:|---:|---:|
//! | 0 | 1024 / 0 / 944 | 512 / 748 | 512 / 1523 | 1016 / 2040 |
//! | 1 | 1024 / 488 / 904 | 634 / 870 | 644 / 1715 | 770 / 1769 |
//! | 2 | 1024 / 0 / 944 | 512 / 748 | 512 / 1523 | 1019 / 2043 |
//! | 3 | 1024 / 0 / 944 | 512 / 748 | 512 / 1523 | ping 1016 / 2040; pong 1019 / 2043 |
//!
//! These are **model estimates**, not elapsed-cycle predictions: split/shared are alternative port
//! assumptions; cores are nominally lockstep; DMA runs at 4 B/cycle/channel into the opposite slot.
//! Bank arbitration, phase drift and backpressure are not simulated. Layout 0 removes the B/C read overlap of the
//! V6 map with the smallest map change (estimated compute-only pressure -122 split-port slots). Base relocation alone
//! cannot remove identical peer A reads. Layout 2 probes 8 KiB address-half placement (not independent
//! physical banks in an interleaved double bank); layout 3 crosses ping/pong A/B double banks.
use crate::{
    dma::{self, BdLocks},
    isa::{
        self,
        bundle::{encode_slot, Bundle},
        gen::{Encoding, Format, Slot},
        Program,
    },
};
use std::sync::LazyLock;

/// Largest K64 chunk count per output tile (`K = 64 * kc`, K <= 10240). The chunk counter is a 32-bit register and
/// the resident i32 C cannot overflow: |c| <= 128 * 128 * 10240 < 2^31.
pub const MAX_KC: usize = 160;
pub const TM: usize = 128;
pub const TN: usize = 64;
pub const CHUNK_K: usize = 64;
pub const A_BYTES: usize = TM * CHUNK_K;
pub const B_BYTES: usize = CHUNK_K * TN;
pub const C_BYTES: usize = TM * TN * 4;
pub const CORE_BASE: u32 = 0x70000;
pub const A_ADDR: [u32; 2] = [0x0000, 0x2000];
pub const B_ADDR: [u32; 2] = [0x4000, 0x5000];
pub const C_ADDR: u32 = 0x6000;
pub const SCRATCH_ADDR: u32 = 0xe000;
pub const A_BD: [u32; 2] = [0, 1];
pub const B_BD: [u32; 2] = [2, 3];
pub const C_BD: u32 = 4;
pub const A_CHANNEL: u32 = 0;
pub const B_CHANNEL: u32 = 1;
pub const C_CHANNEL: u32 = 0;
pub const A_EMPTY: [u32; 2] = [0, 2];
pub const A_FULL: [u32; 2] = [1, 3];
pub const B_EMPTY: [u32; 2] = [4, 6];
pub const B_FULL: [u32; 2] = [5, 7];
pub const C_EMPTY: u32 = 8;
pub const C_FULL: u32 = 9;
pub const CORE_LOCK_BASE: u32 = 48;
pub const SIGNED_8X8: u32 = super::gemm_i8::SIGNED_8X8;
/// V8 pair lock IDs (own numbering, identical in the lower and upper core of a pair): the owner of a half
/// raises `PEER_FULL[s]` once its A half is resident; the peer raises `PEER_EMPTY[s]` once it has read it.
pub const PEER_FULL: [u32; 2] = [10, 11];
pub const PEER_EMPTY: [u32; 2] = [12, 13];
/// Int8 C output buffer (own-tile offset): `(mb 0..16, nb 0..8)` 8x8 int8 blocks, 64 B each.
pub const COUT_ADDR: u32 = 0xe000;
/// One E/O half of a 128-row A block per K64 chunk: 8 mb x 8 kb 8x8 int8 blocks.
pub const A_HALF_BYTES: usize = A_BYTES / 2;
pub const COUT_BYTES: usize = TM * TN;
/// AIE2 core data-memory views (mlir-aie `AIETargetModel.h:875-878`, `AIE2TargetModel`) and the matching core
/// lock-ID views: IDs 0..15 South, 16..31 West, 32..47 North, 48..63 own, in the same South/West/North/East
/// order as the memory views (`getLockLocalBaseIndex`, `AIETargetModel.h:333-337`; the 16-locks-per-core-tile
/// size is `getNumLocks`, `AIETargetModel.h:887-889`). `CORE_LOCK_BASE = 48` is the hardware-proven own view. The
/// concrete table is `AIE2TargetModel::getLockLocalBaseIndex`, upstream mlir-aie `AIETargetModel.cpp:1693-1701`
/// (not in the checked-out `ref/mlir-aie`; confirmed by SimPair against upstream main).
const SOUTH_MEM: u32 = 0x40000;
const NORTH_MEM: u32 = 0x60000;
const SOUTH_LOCK_BASE: u32 = 0;
const NORTH_LOCK_BASE: u32 = 32;

/// Own-tile data-memory map of a V8 pair core, selected by `Probe::layout` ([`pair_layout`]): A half slots (4 KiB
/// each), B slots (4 KiB each), the 32 KiB i32 `c` accumulator and the 8 KiB int8 `cout` buffer. Every region
/// stays contiguous C-order; layouts only move bases, so all maps assemble to the same instruction types, size
/// and issue cycles. 0 (default): `c` moved above `cout` (C / B load separation). 1: the V6 map. 2: A slots 4 KiB apart,
/// B slots in different 8 KiB halves. 3: A and B ping/pong slots crossed between logical groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairLayout {
    pub a: [u32; 2],
    pub b: [u32; 2],
    pub c: u32,
    pub cout: u32,
}
const PAIR_LAYOUTS: [PairLayout; 4] = [
    PairLayout {
        a: [0x0000, 0x2000],
        b: [0x4000, 0x5000],
        c: 0x8000,
        cout: 0x6000,
    },
    PairLayout {
        a: A_ADDR,
        b: B_ADDR,
        c: C_ADDR,
        cout: COUT_ADDR,
    },
    PairLayout {
        a: [0x0000, 0x1000],
        b: [0x4000, 0x6000],
        c: 0x8000,
        cout: 0x2000,
    },
    PairLayout {
        a: [0x0000, 0x5000],
        b: [0x4000, 0x1000],
        c: 0x8000,
        cout: 0x6000,
    },
];
/// The data-memory map `layout` (`0..=3`); panics on any other value.
pub fn pair_layout(layout: u8) -> PairLayout {
    assert!(
        usize::from(layout) < PAIR_LAYOUTS.len(),
        "pair layout {layout} (valid 0..={})",
        PAIR_LAYOUTS.len() - 1
    );
    PAIR_LAYOUTS[layout as usize]
}

/// Which core of a vertical pair: the lower core (core row 0/2) reads the upper core's odd-mb A half through
/// its North views; the upper core reads the lower core's even-mb half through its South views.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairRole {
    Lower,
    Upper,
}
/// C output path. `I32`: the 32 KiB i32 C buffer is drained by the C DMA (V6 behaviour). `Int8 { shift }`: after
/// the last chunk the i32 block accumulators are converted with `VST.SRS.4x` (shift in `s0`, `crSat = 1`
/// saturate, `crRnd = 0` floor, `crSRSMode = 0` 32-bit lanes, signed) into the 8 KiB `COUT_ADDR` buffer.
/// CPU reference: `clamp(c >> shift, -128, 127)` with an arithmetic shift. `shift <= 31`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Epilogue {
    I32,
    Int8 { shift: u8 },
    Bf16,
}
/// `Fast`: the Fast core's control spacing. `Slow`: FastSlowCtl discipline (seven-NOP gaps after every control
/// instruction, NOP-filled branch delay slots); only the VLIW loops keep co-issue. `Probe`: control,
/// layout and timing knobs ([`Probe`]); only `no_compute` and `repeat > 1` invalidate the GEMM result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    Fast,
    Slow,
    Probe(Probe),
}
/// Timing probes for the pair core (hardware measurement of the data-movement and compute rates). `Probe::default()`
/// assembles byte-identically to `Control::Fast` and `Probe { serial: true, ..Default::default() }` to
/// `Control::Slow`; `no_compute` / `repeat` change the work, so the C output is then not a GEMM.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Probe {
    /// Slow (serial) base discipline.
    pub serial: bool,
    /// Timing only: no chunk compute and no epilogue body; the lock / DMA protocol is unchanged.
    pub no_compute: bool,
    /// Timing only: every accumulate chunk runs its compute `repeat` times (0 and 1 mean once).
    pub repeat: u8,
    /// Data-memory map of the pair core ([`pair_layout`], `0..=3`); 0 is the default map, 1 the V6 map.
    pub layout: u8,
}
impl Probe {
    /// Output is not the GEMM result.
    pub fn timing_only(self) -> bool {
        self.no_compute || self.repeat > 1
    }
    /// Parse a comma list of knob names (`serial,nocompute,repeat=N,layout=N`).
    pub fn parse(s: &str) -> Result<Probe, String> {
        let mut p = Probe::default();
        for t in s.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            match t {
                "serial" => p.serial = true,
                "nocompute" => p.no_compute = true,
                _ => match (
                    t.strip_prefix("repeat=").map(str::parse::<u8>),
                    t.strip_prefix("layout=").map(str::parse::<u8>),
                ) {
                    (Some(Ok(n)), _) if (1..=16).contains(&n) => p.repeat = n,
                    (_, Some(Ok(n))) if usize::from(n) < PAIR_LAYOUTS.len() => p.layout = n,
                    _ => return Err(format!("unknown control probe knob {t:?}")),
                },
            }
        }
        Ok(p)
    }
}
impl Control {
    /// Knobs of this control; `Fast`/`Slow` are the two fixed points.
    pub fn probe(self) -> Probe {
        match self {
            Control::Fast => Probe::default(),
            Control::Slow => Probe {
                serial: true,
                ..Probe::default()
            },
            Control::Probe(p) => p,
        }
    }
    /// Serial (Slow) base discipline.
    pub fn is_slow(self) -> bool {
        self.probe().serial
    }
}

/// Offline/hardware bisect choices; `program` retains the unchanged fast path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoreVariant {
    #[default]
    Fast,
    /// One non-NOP instruction per bundle, seven retirement NOPs, no ZOL.
    Serial,
    /// Consume the same AB/C locks and publish a scalar diagnostic C pattern.
    LockOnly,
    /// `Fast`'s compute (VLIW wavefront + JNZ back edge) with Serial control discipline outside `chunk()`:
    /// seven-NOP gaps after every control instruction, NOP-filled branch delay slots, no delay-slot work.
    FastSlowCtl,
    /// `LockOnly` plus a calibrated delay loop (`program_probe`) right before `rel C_FULL` of every tile:
    /// exactly `probe_iters` iterations of [`clock_probe_cycles_per_iter`] issue cycles each, used to
    /// measure the real AIE clock from the submit→completion slope over `probe_iters`.
    ClockProbe,
}

pub(super) fn descriptor(name: &str) -> &'static Encoding {
    isa::gen::ENCODINGS
        .iter()
        .find(|e| e.name == name)
        .expect("generated core instruction")
}
macro_rules! instruction {
    ($name:literal, $args:expr) => {{
        static ENC: LazyLock<&'static Encoding> = LazyLock::new(|| descriptor($name));
        (
            ENC.slot,
            encode_slot(&ENC, $args, 0).expect("core instruction operands"),
        )
    }};
}
pub(super) use instruction;
fn move_reg(kind: usize, n: usize) -> u64 {
    static VALUES: LazyLock<[[u64; 32]; 3]> = LazyLock::new(|| {
        let mut result = [[u64::MAX; 32]; 3];
        for r in descriptor("MOVXM").operands[0].registers {
            for (kind, prefix) in ["r", "p", "m"].iter().enumerate() {
                if let Some(n) = r
                    .name
                    .strip_prefix(prefix)
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    if n < 32 {
                        result[kind][n] = r.value;
                    }
                }
            }
        }
        result
    });
    let value = VALUES[kind][n];
    assert_ne!(value, u64::MAX);
    value
}

#[derive(Default)]
pub(super) struct Assembler {
    pub(super) program: Program,
    labels: Vec<Option<usize>>,
    branches: Vec<(usize, usize, Option<(u8, bool)>)>,
    pub(super) serial: bool,
}
impl Assembler {
    pub(super) fn emit(&mut self, instructions: &[(Slot, u64)]) {
        self.emit_sized(instructions, false);
    }
    fn emit_sized(&mut self, instructions: &[(Slot, u64)], full: bool) {
        static FORMATS: LazyLock<[[Option<&'static Format>; 256]; 2]> = LazyLock::new(|| {
            std::array::from_fn(|full| {
                std::array::from_fn(|mask| {
                    isa::gen::FORMATS
                        .iter()
                        .filter(|f| {
                            let slots = f.slots.iter().fold(0u8, |m, s| m | (1 << s.slot as usize));
                            slots & mask as u8 == mask as u8 && (full == 0 || f.bits == 128)
                        })
                        .min_by_key(|f| f.bits)
                })
            })
        });
        static NOPS: LazyLock<[u64; 8]> = LazyLock::new(|| {
            [
                "NOPB", "NOPX", "NOPXM", "NOPA", "NOPM", "NOPS", "NOPV", "NOP",
            ]
            .map(|n| descriptor(n).value)
        });
        let mut slots = [None; 8];
        let mut mask = 0usize;
        for &(slot, bits) in instructions {
            assert!(
                slots[slot as usize].replace(bits).is_none(),
                "duplicate core slot"
            );
            mask |= 1 << slot as usize;
        }
        let mut bundle = Bundle::new(FORMATS[full as usize][mask].expect("legal core VLIW format"));
        for f in bundle.format.slots {
            bundle
                .set(
                    f.slot,
                    slots[f.slot as usize].unwrap_or(NOPS[f.slot as usize]),
                )
                .unwrap();
        }
        self.program
            .bytes
            .extend_from_slice(bundle.pack().unwrap().as_slice());
        if self.serial && instructions.iter().any(|&(slot, _)| slot != Slot::Nop16) {
            assert_eq!(
                instructions.len(),
                1,
                "serial core must issue one instruction"
            );
            self.nop(7);
        }
    }
    pub(super) fn nop(&mut self, cycles: usize) {
        for _ in 0..cycles {
            self.emit(&[instruction!("NOP", &[])]);
        }
    }
    pub(super) fn mov(&mut self, kind: usize, n: usize, value: u32) {
        self.emit(&[instruction!(
            "MOVXM",
            &[("dst", move_reg(kind, n)), ("i", u64::from(value))]
        )]);
    }
    pub(super) fn add(&mut self, r: u8, value: i8) {
        self.emit(&[instruction!(
            "ADD_add_r_ri",
            &[
                ("d0", r as u64),
                ("s0", r as u64),
                ("imm", (value as u8 & 0x7f) as u64)
            ]
        )]);
    }
    /// `r4 ^= r1`: toggles the ping/pong phase.
    fn toggle_phase(&mut self) {
        self.emit(&[instruction!("XOR", &[("d0", 4), ("s0", 4), ("s1", 1)])]);
    }
    pub(super) fn label(&mut self) -> usize {
        let label = self.labels.len();
        self.labels.push(None);
        label
    }
    /// Every branch target starts on a 16-byte boundary (NOP padding on the fall-through path). AIE2P requires it:
    /// llvm-aie `AIEMachineAlignment` pads every jump-target block to `getMachineBlockAlignmentBytes() == 16` and
    /// asserts "Jump Candidate Alignment is wrong" otherwise; on silicon (hardware window 2026-10-02 21:17 UTC) the
    /// Fast pair core hung or corrupted C with unaligned targets and was exact with this padding alone.
    pub(super) fn bind(&mut self, label: usize) {
        while self.program.pc() % 16 != 0 {
            self.nop(1);
        }
        assert!(self.labels[label].replace(self.program.pc()).is_none());
    }
    pub(super) fn branch(&mut self, label: usize, condition: Option<(u8, bool)>) {
        let pc = self.program.pc();
        self.emit(&[jump(0, condition)]);
        self.branches.push((pc, label, condition));
    }
    pub(super) fn lock(&mut self, acquire: bool, id: u32) {
        self.lock_at(acquire, CORE_LOCK_BASE, id);
    }
    fn lock_at(&mut self, acquire: bool, base: u32, id: u32) {
        // "imm" means immediate lock ID; the lock value is always in s1.
        // AIE2PGenInstrInfo.td:41/4397 and Schedule:4319/7668.
        let args = [
            ("id", (base + id) as u64),
            ("s1", if acquire { 0 } else { 1 }),
        ];
        self.emit(&[if acquire {
            instruction!("ACQ_mLockId_imm", &args)
        } else {
            instruction!("REL_mLockId_imm", &args)
        }]);
        // LCKREQ occupies four cycles; serial uses seven NOPs.
        if !self.serial {
            self.nop(3);
        }
    }
    /// `MOVXM` into a named special register (`s0`, ...).
    pub(super) fn mov_named(&mut self, name: &str, value: u32) {
        let dst = descriptor("MOVXM").operands[0]
            .registers
            .iter()
            .find(|r| r.name == name)
            .unwrap()
            .value;
        self.emit(&[instruction!(
            "MOVXM",
            &[("dst", dst), ("i", u64::from(value))]
        )]);
    }
    /// `MOVX crName, #imm` (ALU slot, 5-bit unsigned immediate).
    pub(super) fn set_cr(&mut self, name: &str, value: u8) {
        assert!(value < 32);
        let dst = descriptor("MOVX_mvx_cr_imm").operands[0]
            .registers
            .iter()
            .find(|r| r.name == name)
            .unwrap()
            .value;
        self.emit(&[instruction!(
            "MOVX_mvx_cr_imm",
            &[("dst", dst), ("src", u64::from(value))]
        )]);
    }
    /// Branch delay slots: NOPs under Fast spacing, already covered by the serial gap under Slow.
    pub(super) fn delay(&mut self) {
        if !self.serial {
            self.nop(isa::sched::JUMP_DELAY_SLOTS);
        }
    }
    pub(super) fn finish(mut self) -> Program {
        for &(pc, label, condition) in &self.branches {
            let target = self.labels[label].expect("bound core branch target");
            let (slot, bits) = jump(target as u64, condition);
            let packed = isa::bundle::pack_slots(&[(slot, bits)]).unwrap();
            assert_eq!(packed.as_slice().len(), 6);
            self.program.bytes[pc..pc + 6].copy_from_slice(packed.as_slice());
        }
        while self.program.bytes.len() % 16 != 0 {
            self.nop(1);
        }
        self.program
    }
}
fn jump(pc: u64, condition: Option<(u8, bool)>) -> (Slot, u64) {
    match condition {
        None => instruction!("J_lng", &[("i", pc)]),
        Some((r, true)) => instruction!("JNZ", &[("i", pc), ("s0", r as u64)]),
        Some((r, false)) => instruction!("JZ", &[("i", pc), ("s0", r as u64)]),
    }
}
pub(super) fn pointer(slot: Slot, p: u8, modifier: u8) -> (Slot, u64) {
    let args = [("ptr", p as u64), ("mod", modifier as u64)];
    match slot {
        Slot::Lda => instruction!("PADDA_pstm_nrm", &args),
        Slot::Ldb => instruction!("PADDB_pstm_nrm", &args),
        Slot::St => instruction!("PADDS_pstm_nrm", &args),
        _ => unreachable!(),
    }
}
pub(super) fn load_a(x: u8, p: u8) -> (Slot, u64) {
    instruction!(
        "VLDA_dmx_lda_x_pstm_nrm_imm",
        &[("dst", x as u64), ("ptr", p as u64), ("imm", 1)]
    )
}
pub(super) fn matrix(dm: u8, k: u8, overwrite: bool) -> (Slot, u64) {
    let args = [
        ("dst", dm as u64),
        ("s1", (dm / 2 * 4 + k % 4) as u64),
        ("s2", 11),
        ("acc", 2),
    ];
    if overwrite {
        instruction!("VMUL_vmul_cm_core_X_X", &args)
    } else {
        instruction!(
            "VMAC_vmul_cm_core_X_X",
            &[args[0], args[1], args[2], args[3], ("acc1", dm as u64)]
        )
    }
}
pub(super) fn store(dm: u8, quarter: u8) -> (Slot, u64) {
    // Raw 512-bit OP_mBMs store, not VSRS: InstrInfo:11481-11488 has no
    // rounding/saturation controls or srSRS_of definition.
    instruction!(
        "VST_dmx_sts_bm_pstm_nrm_imm",
        &[
            ("src", (dm * 4 + quarter) as u64),
            ("ptr", (4 + dm / 2) as u64),
            ("imm", 1)
        ]
    )
}

fn acquire_inputs(a: &mut Assembler) {
    let pong = a.label();
    let ready = a.label();
    a.branch(pong, Some((4, true)));
    a.nop(isa::sched::JUMP_DELAY_SLOTS);
    for slot in 0..2 {
        if slot == 1 {
            a.bind(pong);
        }
        a.lock(true, A_FULL[slot]);
        a.lock(true, B_FULL[slot]);
        a.mov(1, 0, CORE_BASE + A_ADDR[slot]);
        a.mov(1, 1, CORE_BASE + A_ADDR[slot] + 512);
        a.mov(1, 2, CORE_BASE + B_ADDR[slot]);
        a.mov(1, 3, CORE_BASE + B_ADDR[slot] + 64);
        if slot == 0 {
            a.branch(ready, None);
            a.nop(isa::sched::JUMP_DELAY_SLOTS);
        }
    }
    a.bind(ready);
}
fn release_inputs(a: &mut Assembler) {
    let pong = a.label();
    let ready = a.label();
    a.branch(pong, Some((4, true)));
    a.emit(&[instruction!("XOR", &[("d0", 4), ("s0", 4), ("s1", 1)])]);
    a.add(5, -1);
    a.nop(isa::sched::JUMP_DELAY_SLOTS - 2);
    a.lock(false, A_EMPTY[0]);
    a.lock(false, B_EMPTY[0]);
    a.branch(ready, None);
    a.nop(isa::sched::JUMP_DELAY_SLOTS);
    a.bind(pong);
    a.lock(false, A_EMPTY[1]);
    a.lock(false, B_EMPTY[1]);
    a.bind(ready);
}

pub(super) const MAC_TIMES: [i32; 8] = [0, 3, 6, 10, 13, 17, 20, 23];
pub(super) const A_LOAD_TIMES: [[i32; 8]; 2] = [
    [-10, -5, -4, 3, 5, 6, 13, 14],
    [-12, -5, -4, 3, 4, 5, 13, 14],
];
pub(super) const MODIFIERS: [i32; 8] = [512, -2048, 2560, 2432, -1920, 2944, -2432, -512];
pub(super) type Cycle = [Option<u64>; 8];
pub(super) fn put(cycles: &mut [Cycle], time: i32, instruction: (Slot, u64)) {
    let (slot, bits) = instruction;
    assert!(
        cycles[(time + 10) as usize][slot as usize]
            .replace(bits)
            .is_none(),
        "wavefront slot collision at {time} {slot:?}"
    );
}
pub(super) fn pointer_immediate(p: u8, step: i32) -> (Slot, u64) {
    instruction!(
        "PADDS_pstm_nrm_imm",
        &[("ptr", p as u64), ("imm", (step / 64) as u64 & 15)]
    )
}
fn free_store(cycles: &mut [Cycle], mut time: i32, instructions: &[(Slot, u64)]) {
    for &instruction in instructions {
        while cycles[(time + 10) as usize][Slot::St as usize].is_some() {
            time += 1;
        }
        put(cycles, time, instruction);
        time += 1;
    }
}
fn wavefront(overwrite: bool, pair: bool) -> Vec<Cycle> {
    let mut cycles = vec![[None; 8]; 1059];
    let mut b_loads = [Vec::new(), Vec::new()];
    for group in 0..32i32 {
        for dm in 0..4u8 {
            let start = group * 32 + dm as i32 * 8;
            for k in 0..8u8 {
                let time = start + MAC_TIMES[k as usize];
                put(&mut cycles, time, matrix(dm, k, overwrite && k == 0));
                b_loads[(dm % 2) as usize].push((
                    time - 7,
                    k as i32 * 512 + group % 4 * 128 + (dm % 2) as i32 * 64,
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
                .map_or(512, |next| next.1 - offset);
            let modifier = MODIFIERS
                .iter()
                .position(|&m| m == step)
                .expect("B wavefront modifier");
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
    // A cursors repeat four N pairs before advancing two M blocks. C cursors
    // independently skip the intervening row after eight contiguous C blocks.
    for group in 0..31i32 {
        for row in 0..2u8 {
            let start = group * 32 + row as i32 * 16;
            let last_load = start + A_LOAD_TIMES[row as usize][7];
            // Pair (E/O) addressing: p0/p1 each walk one half at 512 B per mb_local, so the pointer already
            // sits on the next mb_local after eight loads and the group%4==3 step is simply absent.
            if !(pair && group % 4 == 3) {
                free_store(
                    &mut cycles,
                    last_load + 1,
                    &[pointer(Slot::St, row, if group % 4 == 3 { 0 } else { 7 })],
                );
            }
        }
        if group % 4 == 3 {
            for row in 0..2u8 {
                let store_end = group * 32 + row as i32 * 16 + 40;
                free_store(
                    &mut cycles,
                    store_end + 1,
                    &[
                        pointer(Slot::St, 4 + row, 2),
                        pointer_immediate(4 + row, -512),
                    ],
                );
                if !overwrite {
                    let load_end = group * 32 + row as i32 * 16 + 2;
                    free_store(
                        &mut cycles,
                        load_end + 1,
                        &[
                            pointer(Slot::St, 6 + row, 2),
                            pointer_immediate(6 + row, -512),
                        ],
                    );
                }
            }
        }
    }
    cycles
}
pub(super) fn emit_cycle(a: &mut Assembler, cycle: &Cycle) {
    let mut instructions = [(Slot::Nop16, 0); 8];
    let mut n = 0;
    for (slot, bits) in cycle.iter().enumerate() {
        if let Some(bits) = bits {
            instructions[n] = (
                [
                    Slot::Ldb,
                    Slot::Alu,
                    Slot::Lng,
                    Slot::Lda,
                    Slot::Mv,
                    Slot::St,
                    Slot::Vec,
                    Slot::Nop16,
                ][slot],
                *bits,
            );
            n += 1;
        }
    }
    a.emit_sized(&instructions[..n], true);
}
/// `c`: own-tile base of the resident i32 C accumulator.
fn chunk(a: &mut Assembler, overwrite: bool, pair: bool, c: u32) {
    static WAVES: LazyLock<[Vec<Cycle>; 4]> = LazyLock::new(|| {
        [
            wavefront(false, false),
            wavefront(true, false),
            wavefront(false, true),
            wavefront(true, true),
        ]
    });
    let cycles = &WAVES[overwrite as usize + 2 * pair as usize];
    a.mov(1, 4, CORE_BASE + c);
    a.mov(1, 5, CORE_BASE + c + 2048);
    if !overwrite {
        a.mov(1, 6, CORE_BASE + c);
        a.mov(1, 7, CORE_BASE + c + 2048);
    }
    // Six-trip loop over cycles[138..266] without the hardware ZOL (ls/le/lc setups execute but never
    // loop on silicon, hardware runs 4 and 5): `add r12,-1` rides in the free ALU slot of body bundle
    // LOOP_DEC and `jnz r12, begin` in the free LNG slot of body bundle 122, so its five delay slots are
    // body bundles 123..127 and the back edge costs no cycle (same scheme as llvm-aie's JNZ loops).
    const LOOP_DEC: usize = 100;
    const LOOP_JNZ: usize = 128 - 1 - isa::sched::JUMP_DELAY_SLOTS;
    a.mov(0, 12, 6);
    // Keep the body 16-byte aligned like LLVM loop headers.
    while a.program.pc() % 16 != 0 {
        a.nop(1);
    }
    for cycle in &cycles[..138] {
        emit_cycle(a, cycle);
    }
    let begin = a.program.pc();
    for (i, cycle) in cycles[138..266].iter().enumerate() {
        let mut cycle = *cycle;
        if i == LOOP_DEC {
            assert!(cycle[1].is_none(), "loop counter needs the ALU slot");
            cycle[1] =
                Some(instruction!("ADD_add_r_ri", &[("d0", 12), ("s0", 12), ("imm", 0x7f)]).1);
        }
        if i == LOOP_JNZ {
            assert!(
                cycle[1].is_none() && cycle[2].is_none() && cycle[4].is_none(),
                "loop branch needs the LNG slot"
            );
            cycle[2] = Some(instruction!("JNZ", &[("i", begin as u64), ("s0", 12)]).1);
        }
        emit_cycle(a, &cycle);
    }
    // Six copies cover periods 4..27; pointers make each M pair identical.
    for repeat in 1..6 {
        assert_eq!(
            &cycles[138..266],
            &cycles[138 + repeat * 128..266 + repeat * 128]
        );
    }
    for cycle in &cycles[906..] {
        emit_cycle(a, cycle);
    }
}

fn slow_acquire_inputs(a: &mut Assembler, pointers: bool) {
    let pong = a.label();
    let ready = a.label();
    a.branch(pong, Some((4, true)));
    for slot in 0..2 {
        if slot == 1 {
            a.bind(pong);
        }
        a.lock(true, A_FULL[slot]);
        a.lock(true, B_FULL[slot]);
        if pointers {
            a.mov(1, 0, CORE_BASE + A_ADDR[slot]);
            a.mov(1, 2, CORE_BASE + B_ADDR[slot]);
        }
        if slot == 0 {
            a.branch(ready, None);
        }
    }
    a.bind(ready);
}
fn slow_release_inputs(a: &mut Assembler) {
    let pong = a.label();
    let ready = a.label();
    a.branch(pong, Some((4, true)));
    a.lock(false, A_EMPTY[0]);
    a.lock(false, B_EMPTY[0]);
    a.branch(ready, None);
    a.bind(pong);
    a.lock(false, A_EMPTY[1]);
    a.lock(false, B_EMPTY[1]);
    a.bind(ready);
    // Deliberately outside branch delay slots in both slow variants.
    a.emit(&[instruction!("XOR", &[("d0", 4), ("s0", 4), ("s1", 1)])]);
    a.add(5, -1);
}
fn serial_chunk(a: &mut Assembler, overwrite: bool) {
    a.mov(1, 4, CORE_BASE + C_ADDR);
    a.mov(0, 6, 16);
    let rows = a.label();
    a.bind(rows);
    a.mov(0, 7, 8);
    let columns = a.label();
    a.bind(columns);
    if !overwrite {
        for quarter in 0..4 {
            a.emit(&[instruction!(
                "VLDA_dmx_lda_bm_pstm_nrm_imm",
                &[("dst", quarter), ("ptr", 4), ("imm", 1)]
            )]);
        }
        a.emit(&[pointer(Slot::Lda, 4, 5)]);
    }
    a.emit(&[load_a(0, 0)]);
    a.emit(&[instruction!(
        "VLDB_dmx_ldb_x_pstm_nrm",
        &[("dst", 11), ("ptr", 2), ("mod", 0)]
    )]);
    a.emit(&[matrix(0, 0, overwrite)]);
    a.mov(0, 8, 7);
    let reduction = a.label();
    a.bind(reduction);
    a.emit(&[load_a(0, 0)]);
    a.emit(&[instruction!(
        "VLDB_dmx_ldb_x_pstm_nrm",
        &[("dst", 11), ("ptr", 2), ("mod", 0)]
    )]);
    a.emit(&[matrix(0, 0, false)]);
    a.add(8, -1);
    a.branch(reduction, Some((8, true)));
    a.emit(&[pointer(Slot::Lda, 0, 1)]);
    a.emit(&[pointer(Slot::Ldb, 2, 2)]);
    for quarter in 0..4 {
        a.emit(&[store(0, quarter)]);
    }
    a.add(7, -1);
    a.branch(columns, Some((7, true)));
    a.emit(&[pointer(Slot::Lda, 0, 0)]);
    a.emit(&[pointer(Slot::Ldb, 2, 1)]);
    a.add(6, -1);
    a.branch(rows, Some((6, true)));
}
/// NOP16 bundles per [`clock_probe`] iteration (1 issue cycle each).
const PROBE_NOPS: usize = 4000;
/// Issue cycles of one `ClockProbe` delay-loop iteration, taken or not: 4000 `NOP` (I16) bundles, `add r11,-1`
/// (1) + 7 serial retirement NOPs, `jnz r11` (1) + exactly [`isa::sched::JUMP_DELAY_SLOTS`] (5) NOP delay
/// bundles = 4000 + 1 + 7 + 1 + 5 = 4014. The jnz is not followed by the serial 7-NOP gap (only its five delay
/// slots), so the final, not-taken iteration costs the same 4014 as the taken ones and a tile spends exactly
/// `N * 4014` cycles in the loop on top of an N-independent constant (mov r11, the guarding `jz`, its seven
/// NOPs and 16-byte alignment padding; `jz` skips the loop for N == 0).
pub fn clock_probe_cycles_per_iter() -> u64 {
    (PROBE_NOPS + 1 + 7 + 1 + isa::sched::JUMP_DELAY_SLOTS) as u64
}
/// `r11 = iters; if r11 != 0 { do { 4000 x NOP16; r11 -= 1 } while r11 != 0 }`. Plain (non-ZOL) loop.
fn clock_probe(a: &mut Assembler, iters: u32) {
    a.mov(0, 11, iters);
    let skip = a.label();
    let body = a.label();
    a.branch(skip, Some((11, false)));
    while a.program.pc() % 16 != 0 {
        a.nop(1);
    }
    a.bind(body);
    a.nop(PROBE_NOPS);
    a.add(11, -1); // serial: followed by 7 NOPs
    a.serial = false;
    a.branch(body, Some((11, true)));
    a.nop(isa::sched::JUMP_DELAY_SLOTS);
    a.serial = true;
    a.bind(skip);
}
fn lock_pattern(a: &mut Assembler) {
    a.mov(1, 4, CORE_BASE + C_ADDR);
    let isa::Inst::St(bits) = isa::st_post_imm(isa::Reg::R(10), isa::Reg::P(4), 4) else {
        unreachable!()
    };
    a.mov(0, 9, 64);
    let words = a.label();
    a.bind(words);
    a.emit(&[(Slot::St, bits)]);
    a.add(10, 1);
    a.add(9, -1);
    a.branch(words, Some((9, true)));
    // The 64 word increments plus 192 produce the next tile's +0x100 base.
    for _ in 0..4 {
        a.add(10, 48);
    }
}
/// Fast's control skeleton rewritten with Serial discipline; only `chunk()` keeps VLIW co-issue.
fn fast_slow_ctl_acquire(a: &mut Assembler) {
    let pong = a.label();
    let ready = a.label();
    a.branch(pong, Some((4, true)));
    for slot in 0..2 {
        if slot == 1 {
            a.bind(pong);
        }
        a.lock(true, A_FULL[slot]);
        a.lock(true, B_FULL[slot]);
        a.mov(1, 0, CORE_BASE + A_ADDR[slot]);
        a.mov(1, 1, CORE_BASE + A_ADDR[slot] + 512);
        a.mov(1, 2, CORE_BASE + B_ADDR[slot]);
        a.mov(1, 3, CORE_BASE + B_ADDR[slot] + 64);
        if slot == 0 {
            a.branch(ready, None);
        }
    }
    a.bind(ready);
}
fn fast_chunk_unserialised(a: &mut Assembler, overwrite: bool) {
    a.serial = false;
    chunk(a, overwrite, false, C_ADDR);
    a.serial = true;
}
fn fast_slow_ctl_program(kc: usize, tiles: usize) -> Program {
    assert!((1..=MAX_KC).contains(&kc) && (1..=256).contains(&tiles));
    let mut a = Assembler {
        serial: true,
        ..Assembler::default()
    };
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
    let tiles_loop = a.label();
    let accumulated = a.label();
    let drain = a.label();
    a.bind(tiles_loop);
    a.lock(true, C_EMPTY);
    a.mov(0, 5, kc as u32);
    fast_slow_ctl_acquire(&mut a);
    fast_chunk_unserialised(&mut a, true);
    slow_release_inputs(&mut a);
    a.branch(drain, Some((5, false)));
    a.bind(accumulated);
    fast_slow_ctl_acquire(&mut a);
    fast_chunk_unserialised(&mut a, false);
    slow_release_inputs(&mut a);
    a.branch(accumulated, Some((5, true)));
    a.bind(drain);
    a.lock(false, C_FULL);
    a.add(3, -1);
    a.branch(tiles_loop, Some((3, true)));
    a.emit(&[instruction!("DONE", &[])]);
    a.finish()
}
fn slow_program(kc: usize, tiles: usize, variant: CoreVariant, probe_iters: u32) -> Program {
    assert!((1..=MAX_KC).contains(&kc) && (1..=256).contains(&tiles));
    let lock_only = matches!(variant, CoreVariant::LockOnly | CoreVariant::ClockProbe);
    let mut a = Assembler {
        serial: true,
        ..Assembler::default()
    };
    for (r, v) in [(0, -1i32 as u32), (1, 1), (3, tiles as u32), (4, 0)] {
        a.mov(0, r, v);
    }
    if lock_only {
        a.mov(0, 10, 0xc0de0000);
    } else {
        a.mov(0, 2, SIGNED_8X8);
        for (m, v) in [(0, 512i32), (1, -512), (2, -4032), (5, -256)] {
            a.mov(2, m, v as u32);
        }
    }
    let tiles_loop = a.label();
    a.bind(tiles_loop);
    a.lock(true, C_EMPTY);
    a.mov(0, 5, kc as u32);
    if lock_only {
        let chunks = a.label();
        a.bind(chunks);
        slow_acquire_inputs(&mut a, false);
        slow_release_inputs(&mut a);
        a.branch(chunks, Some((5, true)));
        lock_pattern(&mut a);
        if variant == CoreVariant::ClockProbe {
            clock_probe(&mut a, probe_iters);
        }
    } else {
        let accumulated = a.label();
        let drain = a.label();
        slow_acquire_inputs(&mut a, true);
        serial_chunk(&mut a, true);
        slow_release_inputs(&mut a);
        a.branch(drain, Some((5, false)));
        a.bind(accumulated);
        slow_acquire_inputs(&mut a, true);
        serial_chunk(&mut a, false);
        slow_release_inputs(&mut a);
        a.branch(accumulated, Some((5, true)));
        a.bind(drain);
    }
    a.lock(false, C_FULL);
    a.add(3, -1);
    a.branch(tiles_loop, Some((3, true)));
    a.emit(&[instruction!("DONE", &[])]);
    a.finish()
}

/// Select a diagnostic core while retaining the AB/C DMA and lock contract.
/// LockOnly publishes word `i = 0xC0DE0000 + tile*0x100 + i` for the first
/// 64 packed C words; other C words are unspecified. Tile is zero-based.
pub fn program_variant(kc: usize, tiles: usize, variant: CoreVariant) -> Program {
    match variant {
        CoreVariant::Fast => program(kc, tiles),
        CoreVariant::FastSlowCtl => fast_slow_ctl_program(kc, tiles),
        CoreVariant::Serial | CoreVariant::LockOnly | CoreVariant::ClockProbe => {
            slow_program(kc, tiles, variant, 0)
        }
    }
}

/// The `ClockProbe` program: `LockOnly` plus, once per tile right before `rel C_FULL`, a delay loop of exactly
/// `probe_iters` iterations of [`clock_probe_cycles_per_iter`] issue cycles (skipped entirely for 0).
/// `program_variant(.., ClockProbe)` is `program_probe(.., 0)`.
pub fn program_probe(kc: usize, tiles: usize, probe_iters: u32) -> Program {
    slow_program(kc, tiles, CoreVariant::ClockProbe, probe_iters)
}

/// `kc=K/64`, `tiles=waves`. Ping/pong phase persists across all output tiles.
/// Signed int8 x int8 -> wrapping i32. C is overwritten on the first chunk and
/// accumulated in resident memory thereafter; C-full is published once/tile.
pub fn program(kc: usize, tiles: usize) -> Program {
    assert!((1..=MAX_KC).contains(&kc) && (1..=256).contains(&tiles));
    let mut a = Assembler::default();
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
    let tiles_loop = a.label();
    let accumulated = a.label();
    let drain = a.label();
    a.bind(tiles_loop);
    a.lock(true, C_EMPTY);
    a.mov(0, 5, kc as u32);
    acquire_inputs(&mut a);
    chunk(&mut a, true, false, C_ADDR);
    release_inputs(&mut a);
    a.branch(drain, Some((5, false)));
    a.nop(isa::sched::JUMP_DELAY_SLOTS);
    a.bind(accumulated);
    acquire_inputs(&mut a);
    chunk(&mut a, false, false, C_ADDR);
    release_inputs(&mut a);
    a.branch(accumulated, Some((5, true)));
    a.nop(isa::sched::JUMP_DELAY_SLOTS);
    a.bind(drain);
    a.lock(false, C_FULL);
    a.add(3, -1);
    a.branch(tiles_loop, Some((3, true)));
    a.nop(isa::sched::JUMP_DELAY_SLOTS);
    a.emit(&[instruction!("DONE", &[])]);
    a.finish()
}

/// Straight-through (lock-never-blocks) issue cycles of one later pair chunk, `[Fast, Slow]` x `[ping, pong]`
/// (control + compute, steady state with the back edge taken; ping = slot 0, pong = slot 1 chunk). The first
/// chunk of a tile is ~3 cycles longer (`overwrite` wavefront); checked by `gemm_core_pair_issue_cycles`.
pub const PAIR_CHUNK_CYCLES: [[u64; 2]; 2] = [[1131, 1123], [1211, 1203]];
/// Issue cycles of the `Epilogue::Int8` per-wave epilogue (setup, 128 x `VLDA bm` x4 + `VST.SRS`), `[Fast, Slow]`.
/// LDA-bound: 512 loads + seven-cycle store drain.
pub const PAIR_EPILOGUE_CYCLES: [u64; 2] = [530, 550];
/// Own-view A walk bases of a pair core for one ping/pong slot: `(p0 = E half, p1 = O half)`. The lower core
/// owns E (own view) and reads O from the upper core through its North view; the upper core owns O (own
/// view) and reads E from the lower core through its South view.
fn pair_a_bases(role: PairRole, layout: PairLayout, slot: usize) -> (u32, u32) {
    match role {
        PairRole::Lower => (CORE_BASE + layout.a[slot], NORTH_MEM + layout.a[slot]),
        PairRole::Upper => (SOUTH_MEM + layout.a[slot], CORE_BASE + layout.a[slot]),
    }
}
/// Lock-ID view base through which this core reaches its peer's locks.
fn peer_lock_base(role: PairRole) -> u32 {
    match role {
        PairRole::Lower => NORTH_LOCK_BASE,
        PairRole::Upper => SOUTH_LOCK_BASE,
    }
}
/// Owner protocol steps 1-4 of one chunk: acq own A_FULL, rel own PEER_FULL, acq peer PEER_FULL, acq own B_FULL,
/// then the E/O A pointers (p0/p1) and B pointers (p2/p3) of the current ping/pong slot.
pub(super) fn pair_acquire(a: &mut Assembler, role: PairRole, layout: PairLayout) {
    let pong = a.label();
    let ready = a.label();
    let peer = peer_lock_base(role);
    a.branch(pong, Some((4, true)));
    a.delay();
    for slot in 0..2 {
        if slot == 1 {
            a.bind(pong);
        }
        a.lock(true, A_FULL[slot]);
        a.lock(false, PEER_FULL[slot]);
        a.lock_at(true, peer, PEER_FULL[slot]);
        a.lock(true, B_FULL[slot]);
        pair_pointers_slot(a, role, layout, slot);
        if slot == 0 {
            a.branch(ready, None);
            a.delay();
        }
    }
    a.bind(ready);
}
fn pair_pointers_slot(a: &mut Assembler, role: PairRole, layout: PairLayout, slot: usize) {
    let (p0, p1) = pair_a_bases(role, layout, slot);
    a.mov(1, 0, p0);
    a.mov(1, 1, p1);
    a.mov(1, 2, CORE_BASE + layout.b[slot]);
    a.mov(1, 3, CORE_BASE + layout.b[slot] + 64);
}
/// `Probe::repeat`: reload the A/B walk pointers of the current slot (r4) before a repeated chunk compute.
pub(super) fn pair_pointers(a: &mut Assembler, role: PairRole, layout: PairLayout) {
    let pong = a.label();
    let ready = a.label();
    a.branch(pong, Some((4, true)));
    a.delay();
    pair_pointers_slot(a, role, layout, 0);
    a.branch(ready, None);
    a.delay();
    a.bind(pong);
    pair_pointers_slot(a, role, layout, 1);
    a.bind(ready);
}
/// Owner protocol steps 6-9: rel own B_EMPTY, rel peer PEER_EMPTY (+1), acq own PEER_EMPTY, rel own A_EMPTY;
/// then toggle the ping/pong phase (r4) and count the chunk (r5).
pub(super) fn pair_release(a: &mut Assembler, role: PairRole) {
    let pong = a.label();
    let ready = a.label();
    let peer = peer_lock_base(role);
    let in_delay_slots = !a.serial;
    a.branch(pong, Some((4, true)));
    if in_delay_slots {
        // Fast: the phase/counter updates ride in the five delay slots, like `release_inputs`.
        a.emit(&[instruction!("XOR", &[("d0", 4), ("s0", 4), ("s1", 1)])]);
        a.emit(&[instruction!(
            "ADD_add_r_ri",
            &[("d0", 5), ("s0", 5), ("imm", 0x7f)]
        )]);
        a.nop(isa::sched::JUMP_DELAY_SLOTS - 2);
    } else {
        a.delay();
    }
    for slot in 0..2 {
        if slot == 1 {
            a.bind(pong);
        }
        a.lock(false, B_EMPTY[slot]);
        a.lock_at(false, peer, PEER_EMPTY[slot]);
        a.lock(true, PEER_EMPTY[slot]);
        a.lock(false, A_EMPTY[slot]);
        if slot == 0 {
            a.branch(ready, None);
            a.delay();
        }
    }
    a.bind(ready);
    if !in_delay_slots {
        // Slow: deliberately outside branch delay slots, as in `slow_release_inputs`.
        a.toggle_phase();
        a.add(5, -1);
    }
}
/// Pair chunk compute is always the VLIW wavefront (no serial gaps), whatever the control discipline.
pub(super) fn pair_chunk(a: &mut Assembler, overwrite: bool, c: u32) {
    let serial = a.serial;
    a.serial = false;
    chunk(a, overwrite, true, c);
    a.serial = serial;
}

const EPI_BLOCKS: usize = 128;
/// `VLDA bm` writes its destination at pipeline stage 7 and `VST.SRS` reads its source at stage 1 (AIE2P
/// itinerary operand cycles; RAW needs a strictly earlier write, `sched.rs` module doc), so the store issues
/// 7 + 1 - 1 = seven cycles after the last quarter load of its block. `check_decoded_program` does not alias
/// `bm` quarters with their `dm`, so this delay is asserted in `epilogue_wavefront` and by the simulator tests.
pub(super) const EPI_STORE_DELAY: usize = 7;
const _: () = assert!(
    EPI_STORE_DELAY >= 7,
    "VST.SRS must issue >= 7 cycles after the last VLDA bm of its dm"
);
/// NOP bundles after the last `VST.SRS` of a wave before `rel C_FULL` (store memory stage 7, `sched.rs`
/// `VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_imm_srsSign1` `memory_cycles: &[7]`).
pub(super) const EPI_DRAIN_PAD: usize = 6;
/// Loop body: four blocks, one iteration of software-pipeline skew (modulo schedule, period 16 cycles).
pub(super) const EPI_BODY: usize = 16;
pub(super) const EPI_LOOP_DEC: usize = 0;
pub(super) const EPI_LOOP_JNZ: usize = EPI_BODY - 1 - isa::sched::JUMP_DELAY_SLOTS;
/// Straight-line epilogue timeline: block `b` loads its four 16 x i32 quarters into `bm[4(b%4)..]` of `dm(b%4)`
/// on consecutive cycles `4b..4b+3` (`VLDA` is the only bm load, so the loop is LDA-bound at 4 cycles/block) and
/// stores them as 64 int8 lanes with one `VST.SRS.4x` `EPI_STORE_DELAY` cycles after the last quarter.
fn epilogue_wavefront() -> Vec<Cycle> {
    let mut cycles = vec![[None; 8]; (EPI_BLOCKS - 1) * 4 + 3 + EPI_STORE_DELAY + 1];
    for block in 0..EPI_BLOCKS {
        let dm = (block % 4) as u8;
        for quarter in 0..4u8 {
            let (slot, bits) = instruction!(
                "VLDA_dmx_lda_bm_pstm_nrm_imm",
                &[("dst", (dm * 4 + quarter) as u64), ("ptr", 6), ("imm", 1)]
            );
            assert!(cycles[block * 4 + quarter as usize][slot as usize]
                .replace(bits)
                .is_none());
        }
        let (slot, bits) = instruction!(
            "VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_imm_srsSign1",
            &[("src", dm as u64), ("su", 0), ("ptr", 4), ("imm", 1)]
        );
        assert!(cycles[block * 4 + 3 + EPI_STORE_DELAY][slot as usize]
            .replace(bits)
            .is_none());
    }
    cycles
}
/// Int8 epilogue: peeled first 16 cycles (loads plus the first two stores), a JNZ loop of 31 identical
/// 16-cycle bodies (no hardware ZOL; `add r12,-1` / `jnz r12` ride in free slots, delay slots are body
/// bundles), then the seven-cycle store drain. `p6` walks C i32 (256 B per block), `p4` walks Cout (64 B).
/// Issue cycles: `EPI_BLOCKS * 4 + EPI_STORE_DELAY` = 519 plus pointer/counter setup.
fn epilogue(a: &mut Assembler, c: u32, cout: u32) {
    static CYCLES: LazyLock<Vec<Cycle>> = LazyLock::new(epilogue_wavefront);
    let cycles = &*CYCLES;
    let trips = EPI_BLOCKS * 4 / EPI_BODY - 1;
    for repeat in 2..=trips {
        assert_eq!(
            &cycles[EPI_BODY..2 * EPI_BODY],
            &cycles[repeat * EPI_BODY..(repeat + 1) * EPI_BODY]
        );
    }
    a.mov(1, 6, CORE_BASE + c);
    a.mov(1, 4, CORE_BASE + cout);
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
    // The last VST.SRS commits to memory `memory_cycles` = 7 stages after issue; keep `rel C_FULL` behind it so the
    // Cout DMA (and the simulator harness, which drains at the release) never sees a pending final block.
    a.nop(EPI_DRAIN_PAD);
    a.serial = serial;
}

/// V8 pair core: `program` with the pair A addressing (E/O halves, role-dependent views) and the 9-step lock
/// protocol per chunk (`PEER_FULL`/`PEER_EMPTY` through the neighbour lock view). `Epilogue::I32` keeps the V6
/// C protocol (acq C_EMPTY before chunk 0, rel C_FULL after the last chunk, the C DMA drains 32 KiB). With
/// `Epilogue::Int8` the i32 C is private: after the last chunk the core acquires C_EMPTY, converts the 128
/// blocks into the layout's `cout` buffer (`epilogue`) and releases C_FULL; the next wave starts immediately.
/// `ctl.probe().layout` selects the [`PairLayout`] data-memory map (default 0: C 0x8000, Cout 0x6000).
pub fn program_pair(
    kc: usize,
    tiles: usize,
    role: PairRole,
    epi: Epilogue,
    ctl: Control,
) -> Program {
    assert!((1..=MAX_KC).contains(&kc) && (1..=256).contains(&tiles));
    assert!(
        !matches!(epi, Epilogue::Bf16),
        "direct BF16 is a G80-only epilogue"
    );
    let probe = ctl.probe();
    let layout = pair_layout(probe.layout);
    let mut a = Assembler {
        serial: probe.serial,
        ..Assembler::default()
    };
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
    if let Epilogue::Int8 { shift } = epi {
        assert!(shift <= 31, "SRS shift {shift}");
        a.set_cr("crSat", 1);
        a.set_cr("crRnd", 0);
        a.set_cr("crSRSMode", 0);
        a.mov_named("s0", u32::from(shift));
    }
    let tiles_loop = a.label();
    let accumulated = a.label();
    let drain = a.label();
    a.bind(tiles_loop);
    if epi == Epilogue::I32 {
        a.lock(true, C_EMPTY);
    }
    a.mov(0, 5, kc as u32);
    pair_acquire(&mut a, role, layout);
    if !probe.no_compute {
        pair_chunk(&mut a, true, layout.c);
    }
    pair_release(&mut a, role);
    a.branch(drain, Some((5, false)));
    a.delay();
    a.bind(accumulated);
    pair_acquire(&mut a, role, layout);
    if probe.repeat > 1 {
        a.mov(0, 11, u32::from(probe.repeat));
        let again = a.label();
        a.bind(again);
        pair_pointers(&mut a, role, layout);
        pair_chunk(&mut a, false, layout.c);
        a.add(11, -1);
        if !a.serial {
            a.nop(7);
        }
        a.branch(again, Some((11, true)));
        a.delay();
    } else if !probe.no_compute {
        pair_chunk(&mut a, false, layout.c);
    }
    pair_release(&mut a, role);
    a.branch(accumulated, Some((5, true)));
    a.delay();
    a.bind(drain);
    if epi != Epilogue::I32 {
        a.lock(true, C_EMPTY);
        if !probe.no_compute {
            epilogue(&mut a, layout.c, layout.cout);
        }
    }
    a.lock(false, C_FULL);
    a.add(3, -1);
    a.branch(tiles_loop, Some((3, true)));
    a.delay();
    a.emit(&[instruction!("DONE", &[])]);
    a.finish()
}
/// DMA-view descriptors: A ring on S2MM0, B ring on S2MM1, C ring on MM2S0.
pub fn tile_bds() -> [[u32; 6]; 5] {
    let mut bds = [[0; 6]; 5];
    for slot in 0..2 {
        bds[A_BD[slot] as usize] = dma::tile_bd(
            A_ADDR[slot],
            (A_BYTES / 4) as u32,
            BdLocks {
                acq: Some((A_EMPTY[slot], -1)),
                rel: Some((A_FULL[slot], 1)),
            },
            Some(A_BD[1 - slot]),
        );
        bds[B_BD[slot] as usize] = dma::tile_bd(
            B_ADDR[slot],
            (B_BYTES / 4) as u32,
            BdLocks {
                acq: Some((B_EMPTY[slot], -1)),
                rel: Some((B_FULL[slot], 1)),
            },
            Some(B_BD[1 - slot]),
        );
    }
    bds[C_BD as usize] = dma::tile_bd(
        C_ADDR,
        (C_BYTES / 4) as u32,
        BdLocks {
            acq: Some((C_FULL, -1)),
            rel: Some((C_EMPTY, 1)),
        },
        Some(C_BD),
    );
    bds
}
pub fn initial_locks() -> [i32; 16] {
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
}
/// `tile_bds` for the pair core: A halves (4 KiB) on S2MM0, B on S2MM1, and C MM2S0 draining either the 32 KiB
/// i32 buffer or the 8 KiB int8 `cout` buffer, at the bases of the [`PairLayout`] selected by `ctl.probe().layout`.
pub fn tile_bds_pair(epi: Epilogue, ctl: Control) -> [[u32; 6]; 5] {
    let layout = pair_layout(ctl.probe().layout);
    let mut bds = tile_bds();
    for slot in 0..2 {
        bds[A_BD[slot] as usize] = dma::tile_bd(
            layout.a[slot],
            (A_HALF_BYTES / 4) as u32,
            BdLocks {
                acq: Some((A_EMPTY[slot], -1)),
                rel: Some((A_FULL[slot], 1)),
            },
            Some(A_BD[1 - slot]),
        );
        bds[B_BD[slot] as usize] = dma::tile_bd(
            layout.b[slot],
            (B_BYTES / 4) as u32,
            BdLocks {
                acq: Some((B_EMPTY[slot], -1)),
                rel: Some((B_FULL[slot], 1)),
            },
            Some(B_BD[1 - slot]),
        );
    }
    let (address, bytes) = match epi {
        Epilogue::I32 => (layout.c, C_BYTES),
        Epilogue::Int8 { .. } => (layout.cout, COUT_BYTES),
        Epilogue::Bf16 => panic!("direct BF16 is a G80-only epilogue"),
    };
    bds[C_BD as usize] = dma::tile_bd(
        address,
        (bytes / 4) as u32,
        BdLocks {
            acq: Some((C_FULL, -1)),
            rel: Some((C_EMPTY, 1)),
        },
        Some(C_BD),
    );
    bds
}
/// Pair cores use the same lock numbering and initial values as V6; `PEER_*` start at 0.
pub fn initial_locks_pair() -> [i32; 16] {
    initial_locks()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gemm_core_packed_physical_schedule() {
        let bytes = program(40, 24).finish();
        assert!(bytes.len() <= 16384);
        let mut bundles = Vec::new();
        let mut offset = 0;
        while offset < bytes.len() {
            let bundle = isa::decode::decode(&bytes[offset..], offset as u64).unwrap();
            offset += bundle.len;
            bundles.push(bundle);
        }
        let report = isa::sched::check_decoded_program(&bundles).unwrap();
        assert!(report.is_clean(), "{:?}", report.hazards);
        let vm = bundles.iter().filter(|b| {
            b.instructions
                .iter()
                .any(|i| i.encoding.name.starts_with("VMAC"))
        });
        assert!(vm.clone().any(
            |b| b.instructions.iter().any(|i| i.encoding.slot == Slot::Lda)
                && b.instructions.iter().any(|i| i.encoding.slot == Slot::Ldb)
        ));
        assert!(vm.clone().any(|b| b
            .instructions
            .iter()
            .any(|i| i.encoding.name.starts_with("VST"))));
        println!("gemm_core: {} bytes, {} static bundles, {} physical hazards, {} unannotated timing observations", bytes.len(), report.bundles, report.hazards.len(), report.timing_observations.len());
    }
    #[test]
    fn gemm_core_slow_variants_physical_schedule() {
        for variant in [
            CoreVariant::Serial,
            CoreVariant::LockOnly,
            CoreVariant::FastSlowCtl,
            CoreVariant::ClockProbe,
        ] {
            // ClockProbe is checked with a nonzero N (the loop body is identical for every N; N is a MOVXM immediate).
            let bytes = if variant == CoreVariant::ClockProbe {
                program_probe(40, 24, 2000)
            } else {
                program_variant(40, 24, variant)
            }
            .finish();
            assert!(bytes.len() <= 16384);
            assert_eq!(
                program_variant(1, 1, variant).finish().len(),
                bytes.len(),
                "{variant:?} size depends on kc/tiles/iters"
            );
            let mut bundles = Vec::new();
            let mut offset = 0;
            while offset < bytes.len() {
                let bundle = isa::decode::decode(&bytes[offset..], offset as u64).unwrap();
                let count = bundle
                    .instructions
                    .iter()
                    .filter(|i| !i.encoding.mnemonic.starts_with("nop"))
                    .count();
                assert!(
                    count <= 1 || variant == CoreVariant::FastSlowCtl,
                    "{variant:?} contains multiple instructions"
                );
                offset += bundle.len;
                bundles.push(bundle);
            }
            let is_nop = |b: &isa::decode::DecodedBundle| {
                b.instructions
                    .iter()
                    .all(|i| i.encoding.mnemonic.starts_with("nop"))
            };
            if variant == CoreVariant::FastSlowCtl {
                // Outside chunk(): lock ops are followed by 7 NOP bundles, jumps by NOP-only delay slots.
                // chunk()'s 128-bit wavefront bundles (incl. its JNZ back edge) are exempt.
                for (n, bundle) in bundles.iter().enumerate().filter(|(_, b)| b.len != 16) {
                    let has = |p: &str| {
                        bundle
                            .instructions
                            .iter()
                            .any(|i| i.encoding.name.starts_with(p))
                    };
                    let gap = if has("ACQ_") || has("REL_") {
                        7
                    } else if has("J") || has("DONE") {
                        isa::sched::JUMP_DELAY_SLOTS
                    } else {
                        0
                    };
                    for follow in &bundles[n + 1..(n + 1 + gap).min(bundles.len())] {
                        assert!(
                            is_nop(follow),
                            "{variant:?}: non-NOP within gap after bundle at {:#x}",
                            bundle.pc
                        );
                    }
                }
            }
            let report = isa::sched::check_decoded_program(&bundles).unwrap();
            assert!(report.is_clean(), "{variant:?}: {:?}", report.hazards);
            println!("gemm_core {variant:?}: {} bytes, {} static bundles, {} physical hazards, {} timing observations",
                bytes.len(), report.bundles, report.hazards.len(), report.timing_observations.len());
        }
    }
    fn decode(bytes: &[u8]) -> Vec<isa::decode::DecodedBundle> {
        let mut bundles = Vec::new();
        let mut offset = 0;
        while offset < bytes.len() {
            let bundle = isa::decode::decode(&bytes[offset..], offset as u64).unwrap();
            offset += bundle.len;
            bundles.push(bundle);
        }
        bundles
    }
    fn operand(inst: &isa::decode::DecodedInst, name: &str) -> i64 {
        inst.operands.iter().find(|o| o.name == name).unwrap().value
    }
    /// Every pair role x epilogue x control x data-memory layout. Layout 0 is the plain Fast/Slow control; layouts
    /// 1..=3 are `Control::Probe` knobs with `serial` selecting the Slow discipline.
    fn all() -> Vec<(PairRole, Epilogue, Control)> {
        let mut all = Vec::new();
        for layout in 0..PAIR_LAYOUTS.len() as u8 {
            for serial in [false, true] {
                for epi in [Epilogue::I32, Epilogue::Int8 { shift: 9 }] {
                    for role in [PairRole::Lower, PairRole::Upper] {
                        let ctl = match (layout, serial) {
                            (0, false) => Control::Fast,
                            (0, true) => Control::Slow,
                            _ => Control::Probe(Probe {
                                serial,
                                layout,
                                ..Probe::default()
                            }),
                        };
                        all.push((role, epi, ctl));
                    }
                }
            }
        }
        all
    }
    /// Issue cycles of a straight-through run (locks never block): follows MOVXM/ADD/XOR on the scalar registers
    /// and the JNZ/JZ/J branches with their five delay slots.
    fn issue_cycles(bundles: &[isa::decode::DecodedBundle]) -> u64 {
        let mut regs = [0i32; 32];
        let index = |pc: u64| bundles.iter().position(|b| b.pc == pc).unwrap();
        let (mut i, mut cycles, mut pending): (usize, u64, Option<(usize, usize)>) = (0, 0, None);
        loop {
            let mut jump = None;
            let mut done = false;
            for inst in &bundles[i].instructions {
                match inst.encoding.name {
                    "MOVXM" => {
                        let dst = operand(inst, "dst") as u64;
                        let register =
                            isa::decode::resolve_register(&inst.encoding.operands[0], dst).unwrap();
                        if let Some(n) = register
                            .name
                            .strip_prefix('r')
                            .and_then(|s| s.parse::<usize>().ok())
                        {
                            regs[n] = operand(inst, "i") as i32;
                        }
                    }
                    "ADD_add_r_ri" => {
                        let imm = ((operand(inst, "imm") as i32) << 25) >> 25;
                        regs[operand(inst, "d0") as usize] =
                            regs[operand(inst, "s0") as usize].wrapping_add(imm);
                    }
                    "XOR" => {
                        regs[operand(inst, "d0") as usize] =
                            regs[operand(inst, "s0") as usize] ^ regs[operand(inst, "s1") as usize]
                    }
                    "J_lng" => jump = Some(operand(inst, "i") as u64),
                    "JNZ" => {
                        if regs[operand(inst, "s0") as usize] != 0 {
                            jump = Some(operand(inst, "i") as u64)
                        }
                    }
                    "JZ" => {
                        if regs[operand(inst, "s0") as usize] == 0 {
                            jump = Some(operand(inst, "i") as u64)
                        }
                    }
                    "DONE" => done = true,
                    _ => {}
                }
            }
            cycles += 1;
            if done {
                return cycles;
            }
            let mut next = i + 1;
            if let Some((target, left)) = pending {
                if left == 1 {
                    next = index(target as u64);
                    pending = None;
                } else {
                    pending = Some((target, left - 1));
                }
            } else if let Some(target) = jump {
                pending = Some((target as usize, isa::sched::JUMP_DELAY_SLOTS));
            }
            i = next;
        }
    }
    #[test]
    fn gemm_core_pair_physical_schedule() {
        for (role, epi, ctl) in all() {
            let bytes = program_pair(40, 256, role, epi, ctl).finish();
            let label = format!("{role:?} {epi:?} {ctl:?}");
            assert!(bytes.len() <= 16384, "{label}: {} bytes", bytes.len());
            assert_eq!(
                program_pair(1, 1, role, epi, ctl).finish().len(),
                bytes.len(),
                "{label}: size depends on kc/tiles"
            );
            let bundles = decode(&bytes);
            let is_nop = |b: &isa::decode::DecodedBundle| {
                b.instructions
                    .iter()
                    .all(|i| i.encoding.mnemonic.starts_with("nop"))
            };
            if ctl.is_slow() {
                // Outside the VLIW loops (128-bit bundles): lock ops are followed by 7 NOPs, jumps by NOP-only delay slots.
                for (n, bundle) in bundles.iter().enumerate().filter(|(_, b)| b.len != 16) {
                    let has = |p: &str| {
                        bundle
                            .instructions
                            .iter()
                            .any(|i| i.encoding.name.starts_with(p))
                    };
                    let gap = if has("ACQ_") || has("REL_") {
                        7
                    } else if has("J") || has("DONE") {
                        isa::sched::JUMP_DELAY_SLOTS
                    } else {
                        0
                    };
                    for follow in &bundles[n + 1..(n + 1 + gap).min(bundles.len())] {
                        assert!(
                            is_nop(follow),
                            "{label}: non-NOP within gap after {:#x}",
                            bundle.pc
                        );
                    }
                }
            }
            // Lock IDs of one chunk protocol: own 48+, peer through the neighbour view (north 32+ / south 0+).
            let mut ids = std::collections::BTreeSet::new();
            for bundle in &bundles {
                for inst in &bundle.instructions {
                    if inst.encoding.name.starts_with("ACQ_")
                        || inst.encoding.name.starts_with("REL_")
                    {
                        ids.insert((
                            inst.encoding.name.starts_with("ACQ_"),
                            operand(inst, "id") as u32,
                        ));
                    }
                }
            }
            let peer_base = if role == PairRole::Lower { 32 } else { 0 };
            for s in 0..2 {
                assert!(
                    ids.contains(&(true, 48 + A_FULL[s]))
                        && ids.contains(&(false, 48 + PEER_FULL[s]))
                );
                assert!(
                    ids.contains(&(true, peer_base + PEER_FULL[s]))
                        && ids.contains(&(true, 48 + B_FULL[s]))
                );
                assert!(
                    ids.contains(&(false, 48 + B_EMPTY[s]))
                        && ids.contains(&(false, peer_base + PEER_EMPTY[s]))
                );
                assert!(
                    ids.contains(&(true, 48 + PEER_EMPTY[s]))
                        && ids.contains(&(false, 48 + A_EMPTY[s]))
                );
            }
            assert!(ids.contains(&(true, 48 + C_EMPTY)) && ids.contains(&(false, 48 + C_FULL)));
            let srs = bundles.iter().any(|b| {
                b.instructions.iter().any(|i| {
                    i.encoding
                        .name
                        .starts_with("VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_imm_srsSign1")
                })
            });
            assert_eq!(srs, epi != Epilogue::I32, "{label}: SRS store presence");
            let report = isa::sched::check_decoded_program(&bundles).unwrap();
            assert!(report.is_clean(), "{label}: {:?}", report.hazards);
            println!("gemm_core pair {label}: {} bytes, {} static bundles, {} physical hazards, {} timing observations",
                bytes.len(), report.bundles, report.hazards.len(), report.timing_observations.len());
        }
    }
    #[test]
    fn gemm_core_pair_issue_cycles() {
        for (role, epi, ctl) in all() {
            let t = |kc, tiles| {
                issue_cycles(&decode(&program_pair(kc, tiles, role, epi, ctl).finish()))
            };
            // Chunks alternate ping (slot 0) / pong (slot 1); the paths differ in branch overhead. A taken branch skips
            // the last two NOPs of the Slow gap, so steady state is measured with the back edge taken throughout.
            let step = |k| t(k + 1, 1) - t(k, 1);
            let (ping, pong) = (step(4), step(5));
            assert_eq!(step(6), ping);
            let c = ctl.is_slow() as usize;
            assert_eq!([ping, pong], PAIR_CHUNK_CYCLES[c]);
            let tile = |kc| t(kc, 2) - t(kc, 1);
            if let Epilogue::Int8 { shift } = epi {
                let i32_tile = issue_cycles(&decode(
                    &program_pair(2, 2, role, Epilogue::I32, ctl).finish(),
                )) - issue_cycles(&decode(
                    &program_pair(2, 1, role, Epilogue::I32, ctl).finish(),
                ));
                assert_eq!(tile(2) - i32_tile, PAIR_EPILOGUE_CYCLES[c], "shift {shift}");
            }
            println!("gemm_core pair {role:?} {epi:?} {ctl:?}: later chunk pong {pong} ping {ping}, tile kc=2 {} kc=4 {} (tile overhead + epilogue)",
                tile(2), tile(4));
        }
    }
    #[test]
    fn gemm_core_pair_layouts_are_disjoint_and_probe_validates() {
        assert_eq!(
            pair_layout(1),
            PairLayout {
                a: A_ADDR,
                b: B_ADDR,
                c: C_ADDR,
                cout: COUT_ADDR
            }
        );
        for n in 0..PAIR_LAYOUTS.len() as u8 {
            let l = pair_layout(n);
            let mut regions = [
                (l.a[0], A_HALF_BYTES),
                (l.a[1], A_HALF_BYTES),
                (l.b[0], B_BYTES),
                (l.b[1], B_BYTES),
                (l.c, C_BYTES),
                (l.cout, COUT_BYTES),
            ];
            regions.sort();
            assert!(
                regions
                    .iter()
                    .all(|&(base, bytes)| base % 64 == 0 && base as usize + bytes <= 0x10000),
                "layout {n}: {regions:?}"
            );
            for w in regions.windows(2) {
                assert!(
                    w[0].0 as usize + w[0].1 <= w[1].0 as usize,
                    "layout {n} overlaps: {w:?}"
                );
            }
            assert_eq!(
                Probe::parse(&format!("serial,layout={n}")).unwrap(),
                Probe {
                    serial: true,
                    layout: n,
                    ..Probe::default()
                }
            );
        }
        assert_eq!(Probe::parse("").unwrap().layout, 0);
        assert!(!Probe::parse("layout=3").unwrap().timing_only());
        for bad in ["layout=4", "layout=-1", "layout=", "layout=x", "layout=256"] {
            assert!(Probe::parse(bad).is_err(), "{bad}");
        }
    }
    /// Nominal, lockstep pair access model, NOT a silicon simulator. AM020's memory-module chapter and
    /// load/store interleaving figure give eight single-port physical banks: double bank = off>>14,
    /// parity = (off>>4)&1. Arbitration is per-bank round-robin, one grant/cycle; losers stall and retry.
    ///
    /// Reserve both physical banks of each 512-bit access at LLVM's single memory stage 5. Exact AIE2P
    /// beat occupancy is unknown, so these are reservation collisions, not a byte-bandwidth simulation.
    /// DMA writes 4 B/cycle/channel for 1024 cycles into the opposite A/B slot, touching just one physical
    /// bank; optional Cout MM2S reads 4 B/cycle throughout the chunk (half of an overlapping 8 KiB drain).
    /// Both cores start together: identical peer A reads collide; base relocation cannot remove them.
    /// `split` reproduces LLVM's separate-read/write scoreboard hypothesis; `shared` uses the TRM's
    /// single-port bank rule. Arbitration winner state and retried requests are not simulated.
    /// Port pressure = sum over nominal cycles of max(requests - 1) over the pair's banks. This is an
    /// additive serialized-slot stall ESTIMATE, not a bound on elapsed time: arbitration, beats, core
    /// phase drift and DMA backpressure can change it. Pairwise collisions are not additive stalls.
    /// The printed rows separate compute-only repeats from next-chunk DMA and next-wave Cout traffic.
    #[test]
    fn gemm_core_pair_bank_access_model() {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Source {
            A,
            B,
            CLoad,
            CStore,
            DmaA,
            DmaB,
            Cout,
        }
        #[derive(Clone, Copy)]
        struct Access {
            source: Source,
            core: usize,
            write: bool,
        }
        for layout in 0..PAIR_LAYOUTS.len() as u8 {
            for slot in 0..2 {
                for overwrite in [true, false] {
                    let l = pair_layout(layout);
                    let mut a = Assembler::default();
                    for cycle in wavefront(overwrite, true) {
                        emit_cycle(&mut a, &cycle);
                    }
                    let bundles = decode(&a.program.bytes);
                    let trace_len = bundles.len() + 5;
                    let mut trace: Vec<[Vec<Access>; 16]> = (0..trace_len)
                        .map(|_| std::array::from_fn(|_| Vec::new()))
                        .collect();
                    for (core, role) in [(0, PairRole::Lower), (1, PairRole::Upper)] {
                        let (p0, p1) = pair_a_bases(role, l, slot);
                        let mut p = [
                            p0,
                            p1,
                            CORE_BASE + l.b[slot],
                            CORE_BASE + l.b[slot] + 64,
                            CORE_BASE + l.c,
                            CORE_BASE + l.c + 2048,
                            CORE_BASE + l.c,
                            CORE_BASE + l.c + 2048,
                        ];
                        for (cycle, bundle) in bundles.iter().enumerate() {
                            let old = p;
                            for inst in &bundle.instructions {
                                let name = inst.encoding.name;
                                let source = match name {
                                    "VLDA_dmx_lda_x_pstm_nrm_imm" => Some(Source::A),
                                    "VLDB_dmx_ldb_x_pstm_nrm" => Some(Source::B),
                                    "VLDA_dmx_lda_bm_pstm_nrm_imm" => Some(Source::CLoad),
                                    "VST_dmx_sts_bm_pstm_nrm_imm" => Some(Source::CStore),
                                    _ => None,
                                };
                                if let Some(source) = source {
                                    let ptr = operand(inst, "ptr") as usize;
                                    let address = old[ptr];
                                    let owner = if address & 0xf0000 == CORE_BASE {
                                        core
                                    } else {
                                        1 - core
                                    };
                                    let off = address & 0xffff;
                                    assert!(
                                        off + 64 <= 0x10000,
                                        "layout {layout}: access {address:#x}"
                                    );
                                    let pair = 2 * (off / 0x4000) as usize;
                                    for bank in pair..pair + 2 {
                                        trace[cycle + 5][owner * 8 + bank].push(Access {
                                            source,
                                            core,
                                            write: source == Source::CStore,
                                        });
                                    }
                                    let step = if source == Source::B {
                                        MODIFIERS[operand(inst, "mod") as usize]
                                    } else {
                                        64
                                    };
                                    p[ptr] = old[ptr].wrapping_add_signed(step);
                                } else if name == "PADDS_pstm_nrm" {
                                    let ptr = operand(inst, "ptr") as usize;
                                    p[ptr] = old[ptr].wrapping_add_signed(
                                        MODIFIERS[operand(inst, "mod") as usize],
                                    );
                                } else if name == "PADDS_pstm_nrm_imm" {
                                    let ptr = operand(inst, "ptr") as usize;
                                    let step = operand(inst, "imm") as i32; // Decoder already sign-extends/scales.
                                    p[ptr] = old[ptr].wrapping_add_signed(step);
                                }
                            }
                        }
                    }
                    for traffic in 0..3 {
                        let mut trace = trace.clone();
                        if traffic > 0 {
                            for cycle in 0..1024 {
                                for core in 0..2 {
                                    for (source, base, write) in [
                                        (Source::DmaA, l.a[1 - slot], true),
                                        (Source::DmaB, l.b[1 - slot], true),
                                        (Source::Cout, l.cout, false),
                                    ] {
                                        if source == Source::Cout && traffic < 2 {
                                            continue;
                                        }
                                        let address = base + cycle as u32 * 4;
                                        let bank = (2 * (address / 0x4000) + ((address >> 4) & 1))
                                            as usize;
                                        trace[cycle][core * 8 + bank].push(Access {
                                            source,
                                            core,
                                            write,
                                        });
                                    }
                                }
                            }
                        }
                        let (
                            mut peer,
                            mut bc,
                            mut load_store,
                            mut dma_core,
                            mut split,
                            mut shared,
                            mut pairs,
                        ) = (0, 0, 0, 0, 0, 0, 0);
                        for banks in &trace {
                            let (mut split_cycle, mut shared_cycle) = (0, 0);
                            for accesses in banks {
                                let reads = accesses.iter().filter(|a| !a.write).count();
                                let writes = accesses.len() - reads;
                                split_cycle = split_cycle
                                    .max(reads.saturating_sub(1).max(writes.saturating_sub(1)));
                                shared_cycle = shared_cycle.max(accesses.len().saturating_sub(1));
                                for (i, x) in accesses.iter().enumerate() {
                                    for y in &accesses[i + 1..] {
                                        pairs += 1;
                                        peer += usize::from(
                                            x.source == Source::A
                                                && y.source == Source::A
                                                && x.core != y.core,
                                        );
                                        bc += usize::from(matches!(
                                            (x.source, y.source),
                                            (Source::B, Source::CLoad) | (Source::CLoad, Source::B)
                                        ));
                                        load_store += usize::from(matches!(
                                            (x.source, y.source),
                                            (Source::CStore, Source::CLoad)
                                                | (Source::CLoad, Source::CStore)
                                        ));
                                        let is_dma = |s| {
                                            matches!(s, Source::DmaA | Source::DmaB | Source::Cout)
                                        };
                                        dma_core +=
                                            usize::from(is_dma(x.source) != is_dma(y.source));
                                    }
                                }
                            }
                            split += split_cycle;
                            shared += shared_cycle;
                        }
                        assert_eq!(
                            peer, 1024,
                            "both cores read both banks of the same E/O halves"
                        );
                        if layout != 1 {
                            assert_eq!(bc, 0, "C and B logical banks are disjoint");
                        }
                        println!("layout={layout} slot={slot} overwrite={overwrite} traffic={traffic}: total same-bank reservation pairs={pairs}; pair collisions peerA={peer} B/C={bc} Cload/store={load_store} DMA/core={dma_core}; estimated split/shared stall slots={split}/{shared}");
                    }
                }
            }
        }
    }
}
