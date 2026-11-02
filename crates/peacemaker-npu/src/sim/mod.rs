// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// Encoding layouts derived from Xilinx/llvm-aie AIE2P TableGen; see NOTICE.
//! Offline functional AIE2P execution. No device access or vendor runtime.
//! Encoding layouts transcribed from Xilinx llvm-aie AIE2PGenInstrInfo.td,
//! licensed Apache-2.0 WITH LLVM-exception. Not a hardware-cycle simulator.
//!
//! Scalar, dense int8 VMUL/VMAC, vector/accumulator memory operations, pointer
//! updates, counting locks and five-bundle branch delay slots execute from the
//! generated typed decoder. VLIW instructions sample pre-bundle state; register
//! writes retire at imported itinerary cycles. Later operand sampling and
//! matching vector bypasses can observe earlier bundles' pending results.
//! Positive LC counts body iterations; LS/LE create an immediate backedge after
//! decrement unless LC reaches zero. This follows llvm.set.loop.iterations
//! lowering in AIEBaseInstructionSelector/AIEBaseHardwareLoops, not a hardware
//! measurement. Blocked bundles do not decrement LC.
//! Memory snapshots are functional (not a bank/arbitration model); DMA ticks,
//! lock stalls and instruction fetch must not be interpreted as hardware cycles.
//!
//! `dma` models contiguous tile BD rings and host-offset shim BDs over bounded
//! word FIFOs, including lock waits, completion release and backpressure.
//! `config::Config` loads exact PDI/CDO and firmware TXN bytes, derives circuit
//! routes from switch registers, binds shim DDR_PATCH arguments and waits for
//! routed task-complete tokens in SYNC. Data packet streams, memtile DMA and
//! noncontiguous DMA remain unsupported. Functional ticks are not hardware cycles.
//!
//! Neighbour data memory and locks (V8 pair A-sharing): every core memory operation (scalar
//! `lda`/`st`, `vlda`/`vldb`/`vst` incl. accumulator `bm` forms and `vst.srs`) resolves its address to a
//! [`View`]: 0x40000 South (row-1, same column), 0x50000 West (column-1, same row), 0x60000
//! North (row+1), 0x70000 own, 64 KiB each (mlir-aie `AIETargetModel.h` AIE2 base addresses,
//! lines 875-878; missing neighbours per `AIE2TargetModel::getMemSouth`, `AIETargetModel.cpp`
//! line 781-788: a row-2 core has a memtile below, not core memory). `acq`/`rel` lock IDs map
//! 0..15 South, 16..31 West, 32..47 North, 48..63 own (`AIETargetModel.cpp:1693-1701`
//! `getLockLocalBaseIndex`: 0, NumLocks, 2*NumLocks, 3*NumLocks; upstream main, the file is not
//! vendored under `ref/mlir-aie`; the own base 48 is also silicon-proven by `gemm_core`). A core
//! tile is stepped with [`Tile::step_with`], which borrows the neighbours' memories and locks as
//! [`Neighbours`]; a missing neighbour or a plain [`Tile::step`] yields `Error::Bounds`. Blocking
//! `acq` and delayed `rel`/store retirement behave exactly as for own locks/memory.
//!
//! `vst.srs.4x ... srsSign1` (`VST_SRS_4x_dmx_sts_srs_dm_{pstm_nrm_imm,pstm_nrm,idx_imm}_srsSign1`)
//! stores a 64 x i32 `dm` accumulator as 64 x int8: shift from `s0..s3` (`su`), rounding `crRnd`,
//! saturation `crSat`, 32-bit lane mode (`crSRSMode` = 0, `getSRSModeForIntrinsic` in
//! `AIE2PInstructionSelector.cpp:2543`); see [`srs_lane`] for the sourced mode set. `srsSign0`
//! (unsigned; the sign is an instruction constant, `AIE2PInstrPatterns.td:804-830`) and unsourced
//! modes return `Error::Unsupported`. Rounding values: llvm-aie `lib/clang/*/include/aie2p/aie2p_defines.h:25-47`;
//! saturation: `aie2p_set_mode.h:25-26` (set_sat = 1, set_symsat = 3; `crSat` is 2 bits, `crRnd` 4,
//! `AIE2PRegisterInfo.cpp:687-690`); vendored copy under the oracle wheel `lib/clang/22/include`. Registers are
//! written by `MOVX_mvx_cr_imm`/`MOVX_mvx_cr_r` (crSat, crRnd, crSRSMode, crUPSMode, crUnpackSize) and `MOVXM`/
//! `MOV_alu_mv_mv_mv_cg`/`MOV_alu_mv_mv_mv_scl` (s0..s3); all reset to 0. `srSRS_of` is not modelled.
//!
//! IEF15 additions (`tests/ief15_ops.rs`; itinerary timing as for every other op, source: llvm-aie headers
//! `aie2p_vmult.h`, `aie2p_srs.h`, `aie2p_ups.h`, `aie2p_ldst.h`, `aie2p_enums.h`, `aie2p_vadd.h`): `vmul`/`vmac`/`vaddmac`
//! elementwise 32-lane 16x16 -> acc64 (`compute_control` amode 1, bmode 3, variant 2; sign bits 9/8, zero_acc bit 0,
//! shift16 bit 10 on acc1; sub bits unsupported); `vneg` (conf amode bit 1: acc64, else acc32 lanes); `vsrs.4x/.2x` to x and
//! `vst.srs.4x/.2x` for `crSRSMode` 0 (acc32 -> int8 / int16) and 1 (acc64 -> int16 / int32), see [`srs_round`]; `vups.4x`
//! (`crUPSMode` 1) and `vups.2x` (modes 1 and 0), overflow = `Unsupported` (`crSat` behaviour unsourced);
//! `vldb.unpack`/`vunpack` (`crUnpackSize` 0 int4 -> int8, 1 int8 -> int16, y register = two x); x / bm
//! loads and stores with `idx_imm`, `pstm_nrm_imm` and `pstm_nrm` addressing; `vshuffle` modes 0..=21; vector compare, select,
//! add/sub/and/or, `vmax_lt`/`vmin_ge`, `vbcst.16/.32`, x/bm `vmov`; scalar `and`/`or`/`add`/`lshl`/`eqz`/`nez`
//! (`lshl` amount is signed, negative = logical right shift; |n| >= 32 unsupported); `MOVX` into `crUPSMode`/`crUnpackSize`.
//! Gates: `cargo test -p peacemaker-npu npu_hello_64_words` and
//! `cargo test -p peacemaker-npu gemm_i8_27_shapes_dma_cpu_exact`.
//! Smoke: dump `npu-hello`'s artifacts, then `cargo run -p peacemaker-npu -- <core.bin>`;
//! `cargo run -p peacemaker-npu -- gemm 64` executes three streamed kernel jobs.
//! Submitted-design smoke: `cargo run -p peacemaker-npu -- gemm-design 64`.
//! Exact hello gate: dump current `npu-hello` artifacts, then run
//! `NPU_HELLO_ARTIFACT_DIR=<dir> cargo test -p peacemaker-npu npu_hello_exact_pdi_insts_64_words`.
use crate::isa::Reg;
pub mod config;
pub mod decode;
pub mod dma;
pub mod gemm;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Decode(usize),
    Bounds,
    Alignment,
    Unsupported,
    Limit,
    InvalidLock,
    InvalidBd,
}
pub type Result<T, E = Error> = std::result::Result<T, E>;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Advanced,
    Blocked,
    Done,
}
#[derive(Debug, Clone, Copy)]
pub enum Op {
    Nop,
    Done,
    Move(Reg, u32),
    Add(u8, u8, i32),
    Xor(u8, u8, u8),
    Load(Reg, u8, i32),
    Store(Reg, u8, i32, bool),
    Lock(u8, u8, bool),
    Jump(usize, Option<u8>, bool),
    JumpZero(usize, u8),
    VectorLoad(u8, u8, i32),
    VectorLoadModifier(u8, u8, u8),
    AccumulatorLoad(u8, u8, i32),
    VectorStore(u8, u8, i32),
    PointerAdd(u8, u8),
    PointerAddImmediate(u8, i32),
    LoopRegister(u8, u32),
    /// `add.nc ls|le|lc, rN, #imm` (llvm-aie's only LC form): loop register `n` = rN + imm.
    LoopAdd(u8, u8, i32),
    /// `vmul`/`vmac`/`vaddmac` (`acc`/`acc2` = 7: no accumulator operand). `config` = r register holding
    /// `aie2p_compute_control` bits: int8 8x8x8 matrix (`amode=0,bmode=1,variant=0`, sign bits 9/8) or the
    /// elementwise 32-lane 16x16 -> acc64 mode (`amode=1,bmode=3,variant=2`, plus zero_acc bit 0, shift16 bit 10).
    Multiply {
        dst: u8,
        acc: u8,
        acc2: u8,
        a: u8,
        b: u8,
        config: u8,
    },
    /// `vneg dst, src, rconf`: `-acc` per acc64 (conf amode bit 1 set) or acc32 lane.
    Negate {
        dst: u8,
        src: u8,
        config: u8,
    },
    /// `vsrs.4x`/`vsrs.2x` to an x register (srsSign1).
    ShiftRoundSaturate {
        dst: u8,
        src: AccSource,
        shift: u8,
    },
    /// `vups.4x` (`dst` = [`AccSource::Dm`]) / `vups.2x` (`dst` = [`AccSource::Cm`]) from an x register.
    Ups {
        dst: AccSource,
        src: u8,
        shift: u8,
        signed: bool,
    },
    /// `vldb.unpack` / `vunpack` into the register pair y`dst` = (x`2dst`, x`2dst+1`).
    Unpack {
        dst: u8,
        src: UnpackSource,
        signed: bool,
    },
    /// x / bm-quarter loads and stores with immediate (`idx_imm`, `pstm_nrm_imm`) or modifier addressing.
    VectorMemory {
        access: VectorAccess,
        reg: u8,
        ptr: u8,
        addressing: SrsAddressing,
    },
    /// `vshuffle dst, a, b, rMode`.
    Shuffle {
        dst: u8,
        a: u8,
        b: u8,
        mode: u8,
    },
    /// Lane-wise `vadd/vsub/vband/vbor`.
    VectorAlu {
        kind: VectorAlu,
        dst: u8,
        a: u8,
        b: u8,
    },
    /// `vlt/vge/veqz` mask into r register `mask` (bit i = lane i); `wide` = 32-bit lanes. Eqz reads `b` only.
    Compare {
        kind: Compare,
        wide: bool,
        signed: bool,
        mask: u8,
        a: u8,
        b: u8,
    },
    /// `vsel`: lane i = mask bit i ? b : a.
    Select {
        wide: bool,
        dst: u8,
        a: u8,
        b: u8,
        mask: u8,
    },
    /// `vmax_lt` (`max`) / `vmin_ge`: result plus the r16 compare mask (a<b resp. a>=b).
    MinMax {
        max: bool,
        wide: bool,
        signed: bool,
        dst: u8,
        a: u8,
        b: u8,
    },
    /// `vbcst.16/.32 dst, rN`.
    Broadcast {
        wide: bool,
        dst: u8,
        src: u8,
    },
    /// `vmov dst, src` between x registers and bm quarters.
    VectorMove {
        dst: V512,
        src: V512,
    },
    /// 32-bit scalar ALU (`and`, `or`, `add`, `lshl`, `eqz`, `nez`).
    Scalar {
        kind: ScalarAlu,
        dst: u8,
        a: u8,
        b: u8,
    },
    /// `movxm`/`mov` immediate or `mov s, rN` into a shift or control register.
    MoveSpecial(Special, u32),
    MoveSpecialRegister(Special, u8),
    /// `vst.srs.4x`/`vst.srs.2x ... srsSign{0,1}`: accumulator `src` (dm or cm half) -> 64 B of lanes,
    /// shift from `s{shift}`, rounding `crRnd`, saturation `crSat`, lane shape from `crSRSMode`.
    ShiftRoundSaturateStore {
        src: AccSource,
        shift: u8,
        ptr: u8,
        addressing: SrsAddressing,
        signed: bool,
    },
}
/// Addressing of x / bm vector memory operations (also used by the SRS stores); immediates are in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrsAddressing {
    PostImmediate(i32),
    Immediate(i32),
    PostModifier(u8),
}
/// Accumulator operand: a whole 256-byte `dm`, or half `half` (0 = `cml`, 1 = `cmh`, bytes 128*half..) of `dm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccSource {
    Dm(u8),
    Cm(u8, u8),
}
/// Source of an unpack: a 64-byte memory load (post-increment `step` bytes) or an x register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnpackSource {
    Memory { ptr: u8, step: i32 },
    Register(u8),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorAccess {
    LoadX,
    StoreX,
    LoadBm,
    StoreBm,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorAlu {
    Add16,
    Add32,
    Sub16,
    Sub32,
    And,
    Or,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compare {
    Lt,
    Ge,
    Eqz,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarAlu {
    And,
    Or,
    Add,
    Lshl,
    Eqz,
    Nez,
}
/// 512-bit operand of `vmov`: x register, or bm quarter `4*dm + q`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V512 {
    X(u8),
    Bm(u8),
}
/// Shift (`s0..s3`) and control registers a core can write and an SRS store reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Special {
    S(u8),
    CrSat,
    CrRnd,
    CrSrsMode,
    CrUpsMode,
    CrUnpackSize,
}
/// Data-memory view: a core addresses its own memory at 0x70000 and the South (row-1),
/// West (column-1) and North (row+1) core tiles' data memories at 0x40000/0x50000/0x60000
/// (mlir-aie `AIETargetModel.h` AIE2 `getMemSouth/West/North/EastBaseAddress`, lines 875-878).
/// Lock IDs follow the same order in 16-ID blocks: South 0..15, West 16..31, North 32..47,
/// own 48..63 (mlir-aie `AIETargetModel.cpp` `getLockLocalBaseIndex`, lines 1693-1701).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    South,
    West,
    North,
    Own,
}
impl View {
    /// Window and byte offset (within the 64 KiB tile memory) of a core data address.
    pub fn of_address(addr: u32) -> Result<(View, usize)> {
        let view = match addr {
            0x40000..=0x4ffff => View::South,
            0x50000..=0x5ffff => View::West,
            0x60000..=0x6ffff => View::North,
            0x70000..=0x7ffff => View::Own,
            _ => return Err(Error::Bounds),
        };
        Ok((view, (addr & 0xffff) as usize))
    }
    /// Lock view and local lock index of an `acq`/`rel` lock ID.
    pub fn of_lock(id: u8) -> Result<(View, usize)> {
        let view = match id {
            0..=15 => View::South,
            16..=31 => View::West,
            32..=47 => View::North,
            48..=63 => View::Own,
            _ => return Err(Error::InvalidLock),
        };
        Ok((view, (id % 16) as usize))
    }
}
/// Data memory and locks of one neighbouring core tile, lent to the stepping tile.
pub struct Neighbour<'a> {
    pub memory: &'a mut [u8],
    pub locks: &'a mut [i32; 16],
}
/// Core neighbours of the stepping tile; `None` = no core tile there (memtile below row 2,
/// outside the array, or not instantiated): any access fails with `Error::Bounds`.
#[derive(Default)]
pub struct Neighbours<'a> {
    pub south: Option<Neighbour<'a>>,
    pub west: Option<Neighbour<'a>>,
    pub north: Option<Neighbour<'a>>,
}
fn neighbour<'s, 'a>(nb: &'s mut Neighbours<'a>, view: View) -> Result<&'s mut Neighbour<'a>> {
    let slot = match view {
        View::South => &mut nb.south,
        View::West => &mut nb.west,
        View::North => &mut nb.north,
        View::Own => return Err(Error::Bounds),
    };
    slot.as_mut().ok_or(Error::Bounds)
}
fn memory_of<'s>(
    own: &'s mut [u8],
    nb: &'s mut Neighbours<'_>,
    view: View,
) -> Result<&'s mut [u8]> {
    if view == View::Own {
        Ok(own)
    } else {
        Ok(&mut *neighbour(nb, view)?.memory)
    }
}
fn lock_of<'s>(
    own: &'s mut [i32; 16],
    nb: &'s mut Neighbours<'_>,
    view: View,
    id: usize,
) -> Result<&'s mut i32> {
    let locks = if view == View::Own {
        own
    } else {
        &mut *neighbour(nb, view)?.locks
    };
    locks.get_mut(id).ok_or(Error::InvalidLock)
}
/// Shift-round-saturate one signed accumulator lane (acc32 values are sign-extended, acc64 as is) to a
/// signed `bits`-wide (8, 16 or 32) lane, returned sign-extended.
/// Rounding `crRnd` values and their meaning are the `rnd_*` definitions in llvm-aie's
/// `aie2p_defines.h` (floor 0, ceil 1, sym_floor 2 = toward zero, sym_ceil 3 = away from zero,
/// neg_inf 8 / pos_inf 9 / sym_zero 10 / sym_inf 11 / conv_even 12 / conv_odd 13: halfway
/// cases to -inf/+inf/zero/away/even/odd, other fractions to nearest). Saturation `crSat`:
/// 0 wrap (low `bits` bits), 1 saturate signed, 3 symmetric (`set_sat`/`set_symsat` in `aie2p_set_mode.h`).
/// Any other rounding/saturation value, or shift > 63, is unsourced. The arithmetic is exact (i128).
pub fn srs_round(value: i64, shift: u32, rounding: u32, saturation: u32, bits: u32) -> Result<i64> {
    if shift > 63
        || !matches!(rounding,0..=3|8..=13)
        || !matches!(saturation, 0 | 1 | 3)
        || !matches!(bits, 8 | 16 | 32)
    {
        return Err(Error::Unsupported);
    }
    let v = i128::from(value);
    let floor = v >> shift;
    let rest = v & ((1i128 << shift) - 1);
    let half = if shift == 0 { 0 } else { 1i128 << (shift - 1) };
    let inexact = rest != 0;
    let negative = v < 0;
    let rounded = if shift == 0 {
        v
    } else {
        floor
            + i128::from(match rounding {
                0 => false,
                1 => inexact,
                2 => negative && inexact,
                3 => !negative && inexact,
                _ => {
                    if rest > half {
                        true
                    } else if rest < half {
                        false
                    } else {
                        match rounding {
                            8 => false,
                            9 => true,
                            10 => negative,
                            11 => !negative,
                            12 => floor & 1 == 1,
                            _ => floor & 1 == 0,
                        }
                    }
                }
            })
    };
    let high = (1i128 << (bits - 1)) - 1;
    Ok(match saturation {
        0 => ((rounded as i64) << (64 - bits)) >> (64 - bits),
        1 => rounded.clamp(-high - 1, high) as i64,
        _ => rounded.clamp(-high, high) as i64,
    })
}
/// Shift-round-saturate one signed 32-bit accumulator lane to a signed 8-bit lane ([`srs_round`] with
/// `bits` = 8); acc32 lanes allow shifts up to 31 only.
pub fn srs_lane(value: i32, shift: u32, rounding: u32, saturation: u32) -> Result<i8> {
    if shift > 31 {
        return Err(Error::Unsupported);
    }
    Ok(srs_round(i64::from(value), shift, rounding, saturation, 8)? as i8)
}
#[derive(Clone, Copy)]
pub struct TimedOp {
    pub op: Op,
    pub dst: u8,
    pub pointer_write: u8,
    pub source: u8,
    pub pointer: u8,
    pub modifier: u8,
    pub a: u8,
    pub b: u8,
    pub accumulator: u8,
    pub configuration: u8,
    pub forwarding: u8,
    pub acc_forwarding: u8,
    pub memory: u8,
    /// `acc2` operand read cycle / bypass class (vaddmac).
    pub acc2: u8,
    pub acc2_forwarding: u8,
    /// Bypass class of the `s1` / `s2` vector operands (0 none, 1 MV, 2 VEC).
    pub a_bypass: u8,
    pub b_bypass: u8,
    /// Read cycle of `crUPSMode` / `crUnpackSize`, write cycle of the `cmp` mask result.
    pub control: u8,
    pub cmp: u8,
}
impl TimedOp {
    /// Compatibility for user-supplied functional decoders: one bundle, one
    /// operation, all writes committed at the following bundle boundary.
    pub fn functional(op: Op) -> Self {
        Self {
            op,
            dst: 1,
            pointer_write: 1,
            source: 1,
            pointer: 1,
            modifier: 1,
            a: 1,
            b: 1,
            accumulator: 1,
            configuration: 1,
            forwarding: 0,
            acc_forwarding: 0,
            memory: 1,
            acc2: 1,
            acc2_forwarding: 0,
            a_bypass: 0,
            b_bypass: 0,
            control: 1,
            cmp: 1,
        }
    }
}
pub enum Operations<'a> {
    Single(TimedOp),
    Prepared(&'a [TimedOp]),
}
impl Operations<'_> {
    fn as_slice(&self) -> &[TimedOp] {
        match self {
            Self::Single(op) => std::slice::from_ref(op),
            Self::Prepared(ops) => ops,
        }
    }
}
pub struct DecodedOps<'a> {
    pub operations: Operations<'a>,
    pub len: usize,
}
pub trait Decoder {
    fn decode(&self, bytes: &[u8], pc: usize) -> Result<(Op, usize)>;
    fn bundle(&self, bytes: &[u8], pc: usize) -> Result<DecodedOps<'_>> {
        let (op, len) = self.decode(bytes, pc)?;
        Ok(DecodedOps {
            operations: Operations::Single(TimedOp::functional(op)),
            len,
        })
    }
}

const PIPELINE: usize = 12;
const WRITES_PER_CYCLE: usize = 16;
#[derive(Clone, Copy)]
enum Effect {
    Empty,
    Register(Reg, u32),
    Vector(u8, [u8; 64]),
    Accumulator(u8, [i32; 64]),
    Quarter(u8, [i32; 16]),
    ScalarStore(View, usize, [u8; 4]),
    VectorStore(View, usize, [u8; 64]),
    Release(View, usize, i32),
    LoopRegister(u8, u32),
    Special(Special, u32),
}
#[derive(Clone, Copy)]
struct Write {
    issued: u64,
    at: u64,
    forwarding: u8,
    effect: Effect,
}
impl Write {
    const EMPTY: Self = Self {
        issued: 0,
        at: 0,
        forwarding: 0,
        effect: Effect::Empty,
    };
    fn readable(&self, now: u64, at: u64, bypass: u8) -> bool {
        let ready = self.at - u64::from(bypass != 0 && bypass == self.forwarding);
        self.issued < now && ready <= at
    }
}
struct Frame {
    writes: [Write; WRITES_PER_CYCLE],
    len: usize,
}
impl Frame {
    fn new() -> Self {
        Self {
            writes: [Write::EMPTY; WRITES_PER_CYCLE],
            len: 0,
        }
    }
}

pub struct Tile {
    pub memory: Vec<u8>,
    pub program: Vec<u8>,
    pub r: [u32; 32],
    pub p: [u32; 8],
    pub m: [u32; 8],
    pub dn: [u32; 8],
    pub dj: [u32; 8],
    pub dc: [u32; 8],
    pub sp: u32,
    pub lr: u32,
    pub locks: [i32; 16],
    pub pc: usize,
    pub status: u32,
    pub x: [[u8; 64]; 16],
    pub dm: [[i32; 64]; 8],
    /// Shift registers `s0..s3` (the `su` operand of SRS stores).
    pub s: [u32; 4],
    /// Control registers `crSat` (2 bits), `crRnd` (4 bits), `crSRSMode` (1 bit), `crUPSMode` (1 bit),
    /// `crUnpackSize` (1 bit); reset to 0 (no saturation, floor, 32-bit accumulator lanes, int4 unpack).
    /// Programs set them explicitly.
    pub cr_sat: u32,
    pub cr_rnd: u32,
    pub cr_srs_mode: u32,
    pub cr_ups_mode: u32,
    pub cr_unpack_size: u32,
    /// Five branch delay bundles, per llvm-aie AIEBaseInstrInfo.cpp NumDelaySlots.
    pub branch_delay: usize,
    pending: Option<(usize, usize)>,
    /// Functional timeline for exposed register write-back (not hardware time).
    pub cycles: u64,
    writes: [Frame; PIPELINE],
    loop_registers: [u32; 3],
}
impl Tile {
    pub fn new(program: Vec<u8>) -> Result<Self> {
        if program.len() > 16384 {
            return Err(Error::Bounds);
        }
        Ok(Self {
            memory: vec![0; 65536],
            program,
            r: [0; 32],
            p: [0; 8],
            m: [0; 8],
            dn: [0; 8],
            dj: [0; 8],
            dc: [0; 8],
            sp: 0,
            lr: 0,
            locks: [0; 16],
            pc: 0,
            status: 0,
            branch_delay: 5,
            pending: None,
            x: [[0; 64]; 16],
            dm: [[0; 64]; 8],
            s: [0; 4],
            cr_sat: 0,
            cr_rnd: 0,
            cr_srs_mode: 0,
            cr_ups_mode: 0,
            cr_unpack_size: 0,
            cycles: 0,
            writes: std::array::from_fn(|_| Frame::new()),
            loop_registers: [0; 3],
        })
    }
    pub(crate) fn reset_execution(&mut self) {
        self.pc = 0;
        self.status = 0;
        self.pending = None;
        self.r.fill(0);
        self.p.fill(0);
        self.m.fill(0);
        self.dn.fill(0);
        self.dj.fill(0);
        self.dc.fill(0);
        self.sp = 0;
        self.lr = 0;
        self.x.fill([0; 64]);
        self.dm.fill([0; 64]);
        self.cycles = 0;
        for frame in &mut self.writes {
            frame.len = 0;
        }
        self.loop_registers.fill(0);
        self.s.fill(0);
        self.cr_sat = 0;
        self.cr_rnd = 0;
        self.cr_srs_mode = 0;
        self.cr_ups_mode = 0;
        self.cr_unpack_size = 0;
    }
    fn get(&self, r: Reg) -> Result<u32> {
        Ok(match r {
            Reg::R(n) => *self.r.get(n as usize).ok_or(Error::Bounds)?,
            Reg::P(n) => *self.p.get(n as usize).ok_or(Error::Bounds)?,
            Reg::M(n) => *self.m.get(n as usize).ok_or(Error::Bounds)?,
            Reg::Dn(n) => *self.dn.get(n as usize).ok_or(Error::Bounds)?,
            Reg::Dj(n) => *self.dj.get(n as usize).ok_or(Error::Bounds)?,
            Reg::Dc(n) => *self.dc.get(n as usize).ok_or(Error::Bounds)?,
            Reg::Sp => self.sp,
            Reg::Lr => self.lr,
        })
    }
    fn set(&mut self, r: Reg, v: u32) -> Result<()> {
        *match r {
            Reg::R(n) => self.r.get_mut(n as usize),
            Reg::P(n) => self.p.get_mut(n as usize),
            Reg::M(n) => self.m.get_mut(n as usize),
            Reg::Dn(n) => self.dn.get_mut(n as usize),
            Reg::Dj(n) => self.dj.get_mut(n as usize),
            Reg::Dc(n) => self.dc.get_mut(n as usize),
            Reg::Sp => Some(&mut self.sp),
            Reg::Lr => Some(&mut self.lr),
        }
        .ok_or(Error::Bounds)? = v;
        Ok(())
    }
    fn acquire_in(
        &mut self,
        nb: &mut Neighbours<'_>,
        view: View,
        id: usize,
        value: i32,
    ) -> Result<bool> {
        let lock = lock_of(&mut self.locks, nb, view, id)?;
        let ready = if value < 0 {
            *lock >= value.checked_neg().ok_or(Error::InvalidLock)?
        } else {
            *lock == value
        };
        if ready && value < 0 {
            *lock += value
        }
        Ok(ready)
    }
    fn set_special(&mut self, sp: Special, v: u32) -> Result<()> {
        match sp {
            Special::S(n) => *self.s.get_mut(n as usize).ok_or(Error::Bounds)? = v,
            Special::CrSat => self.cr_sat = v & 3,
            Special::CrRnd => self.cr_rnd = v & 15,
            Special::CrSrsMode => self.cr_srs_mode = v & 1,
            Special::CrUpsMode => self.cr_ups_mode = v & 1,
            Special::CrUnpackSize => self.cr_unpack_size = v & 1,
        }
        Ok(())
    }
    fn get_special(&self, sp: Special) -> Result<u32> {
        Ok(match sp {
            Special::S(n) => *self.s.get(n as usize).ok_or(Error::Bounds)?,
            Special::CrSat => self.cr_sat,
            Special::CrRnd => self.cr_rnd,
            Special::CrSrsMode => self.cr_srs_mode,
            Special::CrUpsMode => self.cr_ups_mode,
            Special::CrUnpackSize => self.cr_unpack_size,
        })
    }
    fn read_special(&self, sp: Special, cycle: u8) -> Result<u32> {
        let mut value = self.get_special(sp)?;
        let mut latest = self.cycles;
        let at = self.read_time(cycle);
        for frame in &self.writes {
            for write in &frame.writes[..frame.len] {
                if write.readable(self.cycles, at, 0) && write.at >= latest {
                    if let Effect::Special(dst, v) = write.effect {
                        if dst == sp {
                            value = match dst {
                                Special::S(_) => v,
                                Special::CrSat => v & 3,
                                Special::CrRnd => v & 15,
                                Special::CrSrsMode | Special::CrUpsMode | Special::CrUnpackSize => {
                                    v & 1
                                }
                            };
                            latest = write.at;
                        }
                    }
                }
            }
        }
        Ok(value)
    }
    pub fn acquire(&mut self, id: usize, value: i32) -> Result<bool> {
        let lock = self.locks.get_mut(id).ok_or(Error::InvalidLock)?;
        let ready = if value < 0 {
            *lock >= value.checked_neg().ok_or(Error::InvalidLock)?
        } else {
            *lock == value
        };
        if ready && value < 0 {
            *lock += value
        }
        Ok(ready)
    }
    pub fn release(&mut self, id: usize, value: i32) -> Result<()> {
        let l = self.locks.get_mut(id).ok_or(Error::InvalidLock)?;
        *l = l.checked_add(value).ok_or(Error::InvalidLock)?;
        Ok(())
    }
    fn queue(&mut self, effect: Effect, latency: u8, forwarding: u8) -> Result<()> {
        if latency == 0 || usize::from(latency) >= PIPELINE {
            return Err(Error::Unsupported);
        }
        let at = self.cycles + u64::from(latency);
        let frame = &mut self.writes[at as usize % PIPELINE];
        if frame.len == WRITES_PER_CYCLE {
            return Err(Error::Limit);
        }
        frame.writes[frame.len] = Write {
            issued: self.cycles,
            at,
            forwarding,
            effect,
        };
        frame.len += 1;
        Ok(())
    }
    fn advance(&mut self, nb: &mut Neighbours<'_>) -> Result<()> {
        self.cycles += 1;
        let index = self.cycles as usize % PIPELINE;
        let count = self.writes[index].len;
        self.writes[index].len = 0;
        for i in 0..count {
            let effect = std::mem::replace(&mut self.writes[index].writes[i].effect, Effect::Empty);
            match effect {
                Effect::Empty => {}
                Effect::Register(r, v) => self.set(r, v)?,
                Effect::Vector(x, v) => *self.x.get_mut(x as usize).ok_or(Error::Bounds)? = v,
                Effect::Accumulator(dm, v) => {
                    *self.dm.get_mut(dm as usize).ok_or(Error::Bounds)? = v
                }
                Effect::Quarter(bm, v) => {
                    let dm = self.dm.get_mut((bm / 4) as usize).ok_or(Error::Bounds)?;
                    dm[(bm % 4) as usize * 16..(bm % 4) as usize * 16 + 16].copy_from_slice(&v);
                }
                Effect::ScalarStore(view, a, v) => memory_of(&mut self.memory, nb, view)?
                    .get_mut(a..a + 4)
                    .ok_or(Error::Bounds)?
                    .copy_from_slice(&v),
                Effect::VectorStore(view, a, v) => memory_of(&mut self.memory, nb, view)?
                    .get_mut(a..a + 64)
                    .ok_or(Error::Bounds)?
                    .copy_from_slice(&v),
                Effect::Release(view, id, v) => {
                    let l = lock_of(&mut self.locks, nb, view, id)?;
                    *l = l.checked_add(v).ok_or(Error::InvalidLock)?
                }
                Effect::LoopRegister(n, v) => {
                    *self
                        .loop_registers
                        .get_mut(n as usize)
                        .ok_or(Error::Bounds)? = v
                }
                Effect::Special(sp, v) => self.set_special(sp, v)?,
            }
        }
        Ok(())
    }
    fn read_time(&self, cycle: u8) -> u64 {
        self.cycles + u64::from(cycle.saturating_sub(1))
    }
    fn read_register(&self, r: Reg, cycle: u8) -> Result<u32> {
        let mut value = self.get(r)?;
        let mut latest = self.cycles;
        let at = self.read_time(cycle);
        for frame in &self.writes {
            for write in &frame.writes[..frame.len] {
                if write.readable(self.cycles, at, 0) && write.at >= latest {
                    if let Effect::Register(dst, v) = write.effect {
                        if dst == r {
                            value = v;
                            latest = write.at;
                        }
                    }
                }
            }
        }
        Ok(value)
    }
    fn read_vector(&self, x: u8, cycle: u8, bypass: u8) -> Result<&[u8; 64]> {
        let mut value = self.x.get(x as usize).ok_or(Error::Bounds)?;
        let mut latest = self.cycles;
        let at = self.read_time(cycle);
        for frame in &self.writes {
            for write in &frame.writes[..frame.len] {
                if write.readable(self.cycles, at, bypass) && write.at >= latest {
                    if let Effect::Vector(dst, ref v) = write.effect {
                        if dst == x {
                            value = v;
                            latest = write.at;
                        }
                    }
                }
            }
        }
        Ok(value)
    }
    fn read_accumulator(&self, dm: u8, cycle: u8, bypass: u8) -> Result<[i32; 64]> {
        let mut base = self.dm.get(dm as usize).ok_or(Error::Bounds)?;
        let mut last_full = self.cycles;
        let at = self.read_time(cycle);
        for frame in &self.writes {
            for write in &frame.writes[..frame.len] {
                if write.readable(self.cycles, at, bypass) && write.at >= last_full {
                    if let Effect::Accumulator(dst, ref v) = write.effect {
                        if dst == dm {
                            base = v;
                            last_full = write.at;
                        }
                    }
                }
            }
        }
        let mut value = *base;
        let mut latest = [last_full; 4];
        for frame in &self.writes {
            for write in &frame.writes[..frame.len] {
                if !write.readable(self.cycles, at, bypass) {
                    continue;
                }
                if let Effect::Quarter(bm, ref v) = write.effect {
                    let q = (bm % 4) as usize;
                    if bm / 4 == dm && write.at >= latest[q] {
                        value[q * 16..q * 16 + 16].copy_from_slice(v);
                        latest[q] = write.at;
                    }
                }
            }
        }
        Ok(value)
    }
    fn read_quarter(&self, bm: u8, cycle: u8) -> Result<&[i32]> {
        let dm = self.dm.get((bm / 4) as usize).ok_or(Error::Bounds)?;
        let q = (bm % 4) as usize;
        let mut value = &dm[q * 16..q * 16 + 16];
        let mut latest = self.cycles;
        let at = self.read_time(cycle);
        for frame in &self.writes {
            for write in &frame.writes[..frame.len] {
                if !write.readable(self.cycles, at, 0) || write.at < latest {
                    continue;
                }
                match write.effect {
                    Effect::Accumulator(dst, ref v) if dst == bm / 4 => {
                        value = &v[q * 16..q * 16 + 16];
                        latest = write.at;
                    }
                    Effect::Quarter(dst, ref v) if dst == bm => {
                        value = v;
                        latest = write.at;
                    }
                    _ => {}
                }
            }
        }
        Ok(value)
    }
    fn issue(&mut self, t: &TimedOp, nb: &mut Neighbours<'_>) -> Result<()> {
        match t.op {
            Op::Nop | Op::Done | Op::Jump(..) | Op::JumpZero(..) => {}
            Op::Move(r, v) => self.queue(Effect::Register(r, v), t.dst, t.forwarding)?,
            Op::LoopRegister(n, v) => {
                self.queue(Effect::LoopRegister(n, v), t.dst, t.forwarding)?
            }
            Op::LoopAdd(n, src, v) => {
                let value = self
                    .read_register(Reg::R(src), t.source)?
                    .wrapping_add(v as u32);
                self.queue(Effect::LoopRegister(n, value), t.dst, t.forwarding)?;
            }
            Op::Add(dst, src, v) => {
                let value = self
                    .read_register(Reg::R(src), t.source)?
                    .wrapping_add(v as u32);
                self.queue(Effect::Register(Reg::R(dst), value), t.dst, t.forwarding)?;
            }
            Op::Xor(dst, a, b) => {
                let value =
                    self.read_register(Reg::R(a), t.a)? ^ self.read_register(Reg::R(b), t.b)?;
                self.queue(Effect::Register(Reg::R(dst), value), t.dst, t.forwarding)?;
            }
            Op::PointerAdd(p, m) => {
                let v = self
                    .read_register(Reg::P(p), t.pointer)?
                    .wrapping_add(self.read_register(Reg::M(m), t.modifier)?);
                self.queue(Effect::Register(Reg::P(p), v), t.pointer_write, 0)?;
            }
            Op::PointerAddImmediate(p, step) => {
                let v = self
                    .read_register(Reg::P(p), t.pointer)?
                    .wrapping_add(step as u32);
                self.queue(Effect::Register(Reg::P(p), v), t.pointer_write, 0)?;
            }
            Op::VectorLoad(x, p, _)
            | Op::AccumulatorLoad(x, p, _)
            | Op::VectorLoadModifier(x, p, _) => {
                let step = match t.op {
                    Op::VectorLoad(_, _, v) | Op::AccumulatorLoad(_, _, v) => v,
                    Op::VectorLoadModifier(_, _, m) => {
                        self.read_register(Reg::M(m), t.modifier)? as i32
                    }
                    _ => unreachable!(),
                };
                let ptr = self.read_register(Reg::P(p), t.pointer)?;
                let (view, address) = View::of_address(ptr)?;
                if address % 64 != 0 {
                    return Err(Error::Alignment);
                }
                let bytes = memory_of(&mut self.memory, nb, view)?
                    .get(address..address + 64)
                    .ok_or(Error::Bounds)?;
                let effect = if matches!(t.op, Op::AccumulatorLoad(..)) {
                    let mut words = [0; 16];
                    for (word, b) in words.iter_mut().zip(bytes.chunks_exact(4)) {
                        *word = i32::from_le_bytes(b.try_into().unwrap());
                    }
                    Effect::Quarter(x, words)
                } else {
                    Effect::Vector(x, bytes.try_into().unwrap())
                };
                self.queue(effect, t.dst, t.forwarding)?;
                self.queue(
                    Effect::Register(Reg::P(p), ptr.wrapping_add(step as u32)),
                    t.pointer_write,
                    0,
                )?;
            }
            Op::VectorStore(src, p, step) => {
                let ptr = self.read_register(Reg::P(p), t.pointer)?;
                let (view, address) = View::of_address(ptr)?;
                if address % 64 != 0 {
                    return Err(Error::Alignment);
                }
                memory_of(&mut self.memory, nb, view)?
                    .get(address..address + 64)
                    .ok_or(Error::Bounds)?;
                let words = self.read_quarter(src, t.source)?;
                let mut bytes = [0; 64];
                for (dst, v) in bytes.chunks_exact_mut(4).zip(words) {
                    dst.copy_from_slice(&v.to_le_bytes());
                }
                self.queue(Effect::VectorStore(view, address, bytes), t.memory, 0)?;
                self.queue(
                    Effect::Register(Reg::P(p), ptr.wrapping_add(step as u32)),
                    t.pointer_write,
                    0,
                )?;
            }
            Op::Multiply {
                dst,
                acc,
                acc2,
                a,
                b,
                config,
            } => {
                let mode = self.read_register(Reg::R(config), t.configuration)?;
                let result = if mode & !0x300 == 8 {
                    if acc2 != NO_ACC {
                        return Err(Error::Unsupported);
                    }
                    let mut result = if acc == NO_ACC {
                        [0; 64]
                    } else {
                        self.read_accumulator(acc, t.accumulator, t.acc_forwarding)?
                    };
                    let a = self.read_vector(a, t.a, t.a_bypass)?;
                    let b = self.read_vector(b, t.b, t.b_bypass)?;
                    for row in 0..8 {
                        for col in 0..8 {
                            let mut sum = result[row * 8 + col];
                            for k in 0..8 {
                                let av = if mode & 0x200 != 0 {
                                    a[row * 8 + k] as i8 as i32
                                } else {
                                    a[row * 8 + k] as i32
                                };
                                let bv = if mode & 0x100 != 0 {
                                    b[k * 8 + col] as i8 as i32
                                } else {
                                    b[k * 8 + col] as i32
                                };
                                sum = sum.wrapping_add(av * bv);
                            }
                            result[row * 8 + col] = sum;
                        }
                    }
                    result
                } else if mode & !(0x300 | 0x400 | 1) == ELEMENTWISE_16X16 {
                    let acc1 = if acc == NO_ACC {
                        [0; 64]
                    } else {
                        self.read_accumulator(acc, t.accumulator, t.acc_forwarding)?
                    };
                    let add = if acc2 == NO_ACC {
                        [0; 64]
                    } else {
                        self.read_accumulator(acc2, t.acc2, t.acc2_forwarding)?
                    };
                    let a = *self.read_vector(a, t.a, t.a_bypass)?;
                    let b = *self.read_vector(b, t.b, t.b_bypass)?;
                    elementwise_16x16(mode, &acc1, &add, &a, &b)
                } else {
                    return Err(Error::Unsupported);
                };
                self.queue(Effect::Accumulator(dst, result), t.dst, t.forwarding)?;
            }
            Op::Negate { dst, src, config } => {
                let mode = self.read_register(Reg::R(config), t.configuration)?;
                if mode & !2 != 0 {
                    return Err(Error::Unsupported);
                }
                let acc = self.read_accumulator(src, t.accumulator, t.acc_forwarding)?;
                let mut result = [0; 64];
                if mode & 2 != 0 {
                    for i in 0..32 {
                        set_acc64(&mut result, i, acc64(&acc, i).wrapping_neg());
                    }
                } else {
                    for (r, v) in result.iter_mut().zip(acc) {
                        *r = v.wrapping_neg();
                    }
                }
                self.queue(Effect::Accumulator(dst, result), t.dst, t.forwarding)?;
            }
            Op::ShiftRoundSaturate { dst, src, shift } => {
                let bytes = self.srs_bytes(src, shift, t)?;
                self.queue(Effect::Vector(dst, bytes), t.dst, t.forwarding)?;
            }
            Op::Ups {
                dst,
                src,
                shift,
                signed,
            } => {
                let mode = self.read_special(Special::CrUpsMode, t.control)?;
                let shift = self.read_special(Special::S(shift), t.source)?;
                let x = *self.read_vector(src, t.source, 0)?;
                if shift > 63 {
                    return Err(Error::Unsupported);
                }
                match (dst, mode) {
                    (AccSource::Dm(d), 1) => {
                        let mut result = [0; 64];
                        for i in 0..32 {
                            set_acc64(
                                &mut result,
                                i,
                                ups_lane(lane(&x, i, false), signed.then_some(16), shift, 64)?,
                            );
                        }
                        self.queue(Effect::Accumulator(d, result), t.dst, t.forwarding)?;
                    }
                    (AccSource::Cm(d, h), 1) => {
                        let mut words = [0; 32];
                        for i in 0..16 {
                            set_acc64(
                                &mut words,
                                i,
                                ups_lane(lane(&x, i, true), signed.then_some(32), shift, 64)?,
                            );
                        }
                        self.queue_half(d, h, words, t)?;
                    }
                    (AccSource::Cm(d, h), _) => {
                        let mut words = [0; 32];
                        for i in 0..32 {
                            words[i] =
                                ups_lane(lane(&x, i, false), signed.then_some(16), shift, 32)?
                                    as i32;
                        }
                        self.queue_half(d, h, words, t)?;
                    }
                    _ => return Err(Error::Unsupported),
                }
            }
            Op::Unpack { dst, src, signed } => {
                let size = self.read_special(Special::CrUnpackSize, t.control)?;
                let (bytes, update) = match src {
                    UnpackSource::Memory { ptr, step } => {
                        let p = self.read_register(Reg::P(ptr), t.pointer)?;
                        let (view, address) = View::of_address(p)?;
                        if address % 64 != 0 {
                            return Err(Error::Alignment);
                        }
                        let bytes: [u8; 64] = memory_of(&mut self.memory, nb, view)?
                            .get(address..address + 64)
                            .ok_or(Error::Bounds)?
                            .try_into()
                            .unwrap();
                        (bytes, Some((ptr, p.wrapping_add(step as u32))))
                    }
                    UnpackSource::Register(x) => (*self.read_vector(x, t.source, 0)?, None),
                };
                let y = unpack(&bytes, size, signed)?;
                self.queue(Effect::Vector(2 * dst, y[0]), t.dst, t.forwarding)?;
                self.queue(Effect::Vector(2 * dst + 1, y[1]), t.dst, t.forwarding)?;
                if let Some((ptr, value)) = update {
                    self.queue(Effect::Register(Reg::P(ptr), value), t.pointer_write, 0)?;
                }
            }
            Op::VectorMemory {
                access,
                reg,
                ptr,
                addressing,
            } => {
                let p = self.read_register(Reg::P(ptr), t.pointer)?;
                let (offset, update) = match addressing {
                    SrsAddressing::PostImmediate(step) => (0, Some(step as u32)),
                    SrsAddressing::Immediate(offset) => (offset as u32, None),
                    SrsAddressing::PostModifier(m) => {
                        (0, Some(self.read_register(Reg::M(m), t.modifier)?))
                    }
                };
                let (view, address) = View::of_address(p.wrapping_add(offset))?;
                if address % 64 != 0 {
                    return Err(Error::Alignment);
                }
                let loaded: [u8; 64] = memory_of(&mut self.memory, nb, view)?
                    .get(address..address + 64)
                    .ok_or(Error::Bounds)?
                    .try_into()
                    .unwrap();
                match access {
                    VectorAccess::LoadX => {
                        self.queue(Effect::Vector(reg, loaded), t.dst, t.forwarding)?
                    }
                    VectorAccess::LoadBm => {
                        let mut words = [0; 16];
                        for (word, b) in words.iter_mut().zip(loaded.chunks_exact(4)) {
                            *word = i32::from_le_bytes(b.try_into().unwrap());
                        }
                        self.queue(Effect::Quarter(reg, words), t.dst, t.forwarding)?;
                    }
                    VectorAccess::StoreX => {
                        let bytes = *self.read_vector(reg, t.source, 0)?;
                        self.queue(Effect::VectorStore(view, address, bytes), t.memory, 0)?;
                    }
                    VectorAccess::StoreBm => {
                        let words = self.read_quarter(reg, t.source)?;
                        let mut bytes = [0; 64];
                        for (dst, v) in bytes.chunks_exact_mut(4).zip(words) {
                            dst.copy_from_slice(&v.to_le_bytes());
                        }
                        self.queue(Effect::VectorStore(view, address, bytes), t.memory, 0)?;
                    }
                }
                if let Some(step) = update {
                    self.queue(
                        Effect::Register(Reg::P(ptr), p.wrapping_add(step)),
                        t.pointer_write,
                        0,
                    )?;
                }
            }
            Op::Shuffle { dst, a, b, mode } => {
                let mode = self.read_register(Reg::R(mode), t.modifier)?;
                let a = *self.read_vector(a, t.a, t.a_bypass)?;
                let b = *self.read_vector(b, t.b, t.b_bypass)?;
                self.queue(
                    Effect::Vector(dst, shuffle(&a, &b, mode)?),
                    t.dst,
                    t.forwarding,
                )?;
            }
            Op::VectorAlu { kind, dst, a, b } => {
                let a = *self.read_vector(a, t.a, t.a_bypass)?;
                let b = *self.read_vector(b, t.b, t.b_bypass)?;
                let mut out = [0; 64];
                match kind {
                    VectorAlu::And => {
                        for i in 0..64 {
                            out[i] = a[i] & b[i];
                        }
                    }
                    VectorAlu::Or => {
                        for i in 0..64 {
                            out[i] = a[i] | b[i];
                        }
                    }
                    VectorAlu::Add16 | VectorAlu::Sub16 | VectorAlu::Add32 | VectorAlu::Sub32 => {
                        let wide = matches!(kind, VectorAlu::Add32 | VectorAlu::Sub32);
                        let add = matches!(kind, VectorAlu::Add16 | VectorAlu::Add32);
                        for i in 0..lane_count(wide) {
                            let (x, y) = (lane(&a, i, wide), lane(&b, i, wide));
                            set_lane(
                                &mut out,
                                i,
                                wide,
                                if add {
                                    x.wrapping_add(y)
                                } else {
                                    x.wrapping_sub(y)
                                },
                            );
                        }
                    }
                }
                self.queue(Effect::Vector(dst, out), t.dst, t.forwarding)?;
            }
            Op::Compare {
                kind,
                wide,
                signed,
                mask,
                a,
                b,
            } => {
                let a = if kind == Compare::Eqz {
                    [0; 64]
                } else {
                    *self.read_vector(a, t.a, t.a_bypass)?
                };
                let b = *self.read_vector(b, t.b, t.b_bypass)?;
                let mut bits = 0u32;
                for i in 0..lane_count(wide) {
                    let (x, y) = (lane(&a, i, wide), lane(&b, i, wide));
                    let set = match kind {
                        Compare::Eqz => y == 0,
                        Compare::Lt => less(x, y, wide, signed),
                        Compare::Ge => !less(x, y, wide, signed),
                    };
                    bits |= u32::from(set) << i;
                }
                self.queue(Effect::Register(Reg::R(mask), bits), t.cmp, 0)?;
            }
            Op::Select {
                wide,
                dst,
                a,
                b,
                mask,
            } => {
                let a = *self.read_vector(a, t.a, t.a_bypass)?;
                let b = *self.read_vector(b, t.b, t.b_bypass)?;
                let bits = self.read_register(Reg::R(mask), t.source)?;
                let mut out = [0; 64];
                for i in 0..lane_count(wide) {
                    set_lane(
                        &mut out,
                        i,
                        wide,
                        if bits >> i & 1 != 0 {
                            lane(&b, i, wide)
                        } else {
                            lane(&a, i, wide)
                        },
                    );
                }
                self.queue(Effect::Vector(dst, out), t.dst, t.forwarding)?;
            }
            Op::MinMax {
                max,
                wide,
                signed,
                dst,
                a,
                b,
            } => {
                let a = *self.read_vector(a, t.a, t.a_bypass)?;
                let b = *self.read_vector(b, t.b, t.b_bypass)?;
                let (mut out, mut bits) = ([0; 64], 0u32);
                for i in 0..lane_count(wide) {
                    let (x, y) = (lane(&a, i, wide), lane(&b, i, wide));
                    let lt = less(x, y, wide, signed);
                    bits |= u32::from(lt == max) << i;
                    set_lane(&mut out, i, wide, if lt == max { y } else { x });
                }
                self.queue(Effect::Vector(dst, out), t.dst, t.forwarding)?;
                self.queue(Effect::Register(Reg::R(16), bits), t.cmp, 0)?;
            }
            Op::Broadcast { wide, dst, src } => {
                let value = self.read_register(Reg::R(src), t.source)?;
                let mut out = [0; 64];
                for i in 0..lane_count(wide) {
                    set_lane(&mut out, i, wide, value);
                }
                self.queue(Effect::Vector(dst, out), t.dst, t.forwarding)?;
            }
            Op::VectorMove { dst, src } => {
                let bytes = match src {
                    V512::X(x) => *self.read_vector(x, t.source, 0)?,
                    V512::Bm(bm) => {
                        let words = self.read_quarter(bm, t.source)?;
                        let mut bytes = [0; 64];
                        for (dst, v) in bytes.chunks_exact_mut(4).zip(words) {
                            dst.copy_from_slice(&v.to_le_bytes());
                        }
                        bytes
                    }
                };
                match dst {
                    V512::X(x) => self.queue(Effect::Vector(x, bytes), t.dst, t.forwarding)?,
                    V512::Bm(bm) => {
                        let mut words = [0; 16];
                        for (word, b) in words.iter_mut().zip(bytes.chunks_exact(4)) {
                            *word = i32::from_le_bytes(b.try_into().unwrap());
                        }
                        self.queue(Effect::Quarter(bm, words), t.dst, t.forwarding)?;
                    }
                }
            }
            Op::Scalar { kind, dst, a, b } => {
                let x = self.read_register(Reg::R(a), t.a)?;
                let y = if matches!(kind, ScalarAlu::Eqz | ScalarAlu::Nez) {
                    0
                } else {
                    self.read_register(Reg::R(b), t.b)?
                };
                let value = match kind {
                    ScalarAlu::And => x & y,
                    ScalarAlu::Or => x | y,
                    ScalarAlu::Add => x.wrapping_add(y),
                    ScalarAlu::Eqz => u32::from(x == 0),
                    ScalarAlu::Nez => u32::from(x != 0),
                    // `a << n` lowers to `lshl rd, ra, rn` and unsigned `a >> n` to `lshl rd, ra, -n`
                    // (llvm-aie AIE2P codegen of plain C shifts): signed amount, negative = logical right shift.
                    // |n| >= 32 is unsourced.
                    ScalarAlu::Lshl => match y as i32 {
                        n @ 0..=31 => x << n,
                        n @ -31..=-1 => x >> -n,
                        _ => return Err(Error::Unsupported),
                    },
                };
                self.queue(Effect::Register(Reg::R(dst), value), t.dst, t.forwarding)?;
            }
            Op::Load(r, p, v) => {
                let (view, address) = View::of_address(
                    self.read_register(Reg::P(p), t.pointer)?
                        .wrapping_add(v as u32),
                )?;
                if address % 4 != 0 {
                    return Err(Error::Alignment);
                }
                let value = u32::from_le_bytes(
                    memory_of(&mut self.memory, nb, view)?
                        .get(address..address + 4)
                        .ok_or(Error::Bounds)?
                        .try_into()
                        .unwrap(),
                );
                self.queue(Effect::Register(r, value), t.dst, t.forwarding)?;
            }
            Op::Store(r, p, v, post) => {
                let ptr = self.read_register(Reg::P(p), t.pointer)?;
                let (view, address) =
                    View::of_address(ptr.wrapping_add(if post { 0 } else { v as u32 }))?;
                if address % 4 != 0 {
                    return Err(Error::Alignment);
                }
                memory_of(&mut self.memory, nb, view)?
                    .get(address..address + 4)
                    .ok_or(Error::Bounds)?;
                let bytes = self.read_register(r, t.source)?.to_le_bytes();
                self.queue(Effect::ScalarStore(view, address, bytes), t.memory, 0)?;
                if post {
                    self.queue(
                        Effect::Register(Reg::P(p), ptr.wrapping_add(v as u32)),
                        t.pointer_write,
                        0,
                    )?;
                }
            }
            Op::Lock(id, r, true) => {
                let (view, id) = View::of_lock(id)?;
                let value = self.read_register(Reg::R(r), t.source)? as i32;
                if !self.acquire_in(nb, view, id, value)? {
                    return Err(Error::InvalidLock);
                }
            }
            Op::Lock(id, r, false) => {
                let (view, id) = View::of_lock(id)?;
                let value = self.read_register(Reg::R(r), t.source)? as i32;
                lock_of(&mut self.locks, nb, view, id)?;
                self.queue(Effect::Release(view, id, value), t.memory, 0)?;
            }
            Op::MoveSpecial(sp, v) => self.queue(Effect::Special(sp, v), t.dst, t.forwarding)?,
            Op::MoveSpecialRegister(sp, r) => {
                let value = self.read_register(Reg::R(r), t.source)?;
                self.queue(Effect::Special(sp, value), t.dst, t.forwarding)?;
            }
            Op::ShiftRoundSaturateStore {
                src,
                shift,
                ptr,
                addressing,
                signed,
            } => {
                // Unsigned SRS (`srsSign0`) has no sourced saturation range: refuse, never guess.
                if !signed {
                    return Err(Error::Unsupported);
                }
                let p = self.read_register(Reg::P(ptr), t.pointer)?;
                let (offset, update) = match addressing {
                    SrsAddressing::PostImmediate(step) => (0, Some(step as u32)),
                    SrsAddressing::Immediate(offset) => (offset as u32, None),
                    SrsAddressing::PostModifier(m) => {
                        (0, Some(self.read_register(Reg::M(m), t.modifier)?))
                    }
                };
                let (view, address) = View::of_address(p.wrapping_add(offset))?;
                if address % 64 != 0 {
                    return Err(Error::Alignment);
                }
                memory_of(&mut self.memory, nb, view)?
                    .get(address..address + 64)
                    .ok_or(Error::Bounds)?;
                let bytes = self.srs_bytes(src, shift, t)?;
                self.queue(Effect::VectorStore(view, address, bytes), t.memory, 0)?;
                if let Some(step) = update {
                    self.queue(
                        Effect::Register(Reg::P(ptr), p.wrapping_add(step)),
                        t.pointer_write,
                        0,
                    )?;
                }
            }
        }
        Ok(())
    }
    /// Shift-round-saturate an accumulator operand to 64 bytes per `crSRSMode` (see [`srs_vector`]); shift
    /// from `s{shift}`, `crRnd`, `crSat`, all sampled at the `src` operand cycle.
    fn srs_bytes(&self, src: AccSource, shift: u8, t: &TimedOp) -> Result<[u8; 64]> {
        let mode = self.read_special(Special::CrSrsMode, t.source)?;
        let rounding = self.read_special(Special::CrRnd, t.source)?;
        let saturation = self.read_special(Special::CrSat, t.source)?;
        let shift = self.read_special(Special::S(shift), t.source)?;
        let (AccSource::Dm(dm) | AccSource::Cm(dm, _)) = src;
        let acc = self.read_accumulator(dm, t.source, 0)?;
        srs_vector(&acc, src, mode, shift, rounding, saturation)
    }
    /// Queue a write of cm half `half` of `dm` as its two bm quarters.
    fn queue_half(&mut self, dm: u8, half: u8, words: [i32; 32], t: &TimedOp) -> Result<()> {
        let bm = 4 * dm + 2 * half;
        self.queue(
            Effect::Quarter(bm, words[..16].try_into().unwrap()),
            t.dst,
            t.forwarding,
        )?;
        self.queue(
            Effect::Quarter(bm + 1, words[16..].try_into().unwrap()),
            t.dst,
            t.forwarding,
        )
    }
    /// Single-tile step: no neighbour views, so any access through the South/West/North
    /// windows or neighbour lock IDs fails with `Error::Bounds`.
    pub fn step<D: Decoder>(&mut self, d: &D) -> Result<Step> {
        self.step_with(d, &mut Neighbours::default())
    }
    /// Step with the data memories and locks of the South/West/North neighbour core
    /// tiles (`None` for a missing neighbour) visible through their address/lock windows.
    pub fn step_with<D: Decoder>(&mut self, d: &D, nb: &mut Neighbours<'_>) -> Result<Step> {
        if self.status & (1 << 20) != 0 {
            return Ok(Step::Done);
        }
        let decoded = d.bundle(&self.program, self.pc)?;
        let operations = decoded.operations.as_slice();
        // Preflight before issuing any slot: retrying a blocked bundle must not
        // duplicate stores, pointer updates or releases from its other slots.
        for op in operations {
            if let Op::Lock(id, r, true) = op.op {
                let (view, id) = View::of_lock(id)?;
                let value = self.read_register(Reg::R(r), op.source)? as i32;
                let count = *lock_of(&mut self.locks, nb, view, id)?;
                let ready = if value < 0 {
                    count >= value.checked_neg().ok_or(Error::InvalidLock)?
                } else {
                    count == value
                };
                if !ready {
                    self.advance(nb)?;
                    return Ok(Step::Blocked);
                }
            }
        }
        let old = self.pending;
        let mut jump = None;
        let mut done = false;
        for op in operations {
            match op.op {
                Op::Done => done = true,
                Op::Jump(target, cond, link) => {
                    let taken = match cond {
                        None => true,
                        Some(r) => self.read_register(Reg::R(r), op.source)? != 0,
                    };
                    if taken {
                        if self.pending.is_some() || jump.is_some() {
                            return Err(Error::Unsupported);
                        }
                        if link {
                            let mut return_pc = self.pc + decoded.len;
                            for _ in 0..self.branch_delay {
                                return_pc += d.bundle(&self.program, return_pc)?.len;
                            }
                            self.queue(
                                Effect::Register(Reg::Lr, return_pc as u32),
                                op.dst,
                                op.forwarding,
                            )?;
                        }
                        jump = Some((target, self.branch_delay));
                    }
                }
                Op::JumpZero(target, r) => {
                    if self.read_register(Reg::R(r), op.source)? == 0 {
                        if self.pending.is_some() || jump.is_some() {
                            return Err(Error::Unsupported);
                        }
                        jump = Some((target, self.branch_delay));
                    }
                }
                _ => {}
            }
            self.issue(op, nb)?;
        }
        let loop_back = self.pc == self.loop_registers[1] as usize && self.loop_registers[2] != 0;
        if loop_back && (old.is_some() || jump.is_some() || done) {
            return Err(Error::Unsupported);
        }
        self.pc += decoded.len;
        if let Some((target, left)) = old {
            if left <= 1 {
                self.pc = target;
                self.pending = None
            } else {
                self.pending = Some((target, left - 1))
            }
        }
        if let Some((target, left)) = jump {
            if left == 0 {
                self.pc = target
            } else {
                self.pending = Some((target, left))
            }
        }
        if loop_back {
            self.loop_registers[2] -= 1;
            if self.loop_registers[2] != 0 {
                self.pc = self.loop_registers[0] as usize;
            }
        }
        self.advance(nb)?;
        if done {
            // DONE retires at E6. Flush already-issued effects, without
            // interpreting the following padded bytes as instructions.
            for _ in 1..6 {
                self.advance(nb)?;
            }
            self.status |= 1 << 20;
            return Ok(Step::Done);
        }
        Ok(Step::Advanced)
    }
    pub fn run<D: Decoder>(&mut self, d: &D, limit: usize) -> Result<Step> {
        self.run_with(d, &mut Neighbours::default(), limit)
    }
    pub fn run_with<D: Decoder>(
        &mut self,
        d: &D,
        nb: &mut Neighbours<'_>,
        limit: usize,
    ) -> Result<Step> {
        for _ in 0..limit {
            let s = self.step_with(d, nb)?;
            if s != Step::Advanced {
                return Ok(s);
            }
        }
        Err(Error::Limit)
    }
}

/// `vmul`/`vmac`/`vaddmac` "no accumulator operand" marker (dm index 7 is not an encodable `eDM` register).
const NO_ACC: u8 = 7;
/// `aie2p_compute_control(.., amode=1, bmode=3, variant=2, ..)` without the sign (bits 8/9), zero_acc (bit 0)
/// and shift16 (bit 10) bits: `aie2p_vmult.h:15-22` (amode<<1 | bmode<<3 | variant<<5).
const ELEMENTWISE_16X16: u32 = (1 << 1) | (3 << 3) | (2 << 5);
/// acc64 lane `i` of a 64-word accumulator: words `2i` (low) and `2i+1` (high), i.e. bytes `8i..8i+8` LE.
fn acc64(words: &[i32], i: usize) -> i64 {
    (u64::from(words[2 * i] as u32) | u64::from(words[2 * i + 1] as u32) << 32) as i64
}
fn set_acc64(words: &mut [i32], i: usize, v: i64) {
    words[2 * i] = v as i32;
    words[2 * i + 1] = (v >> 32) as i32;
}
/// 32 lanes `p_i = A[i]*B[i]` (16-bit lanes, signed iff conf bit 9 for `a` / bit 8 for `b`); result lane
/// `sh(acc1) + acc2 + p` with `sh(x)` = 0 if zero_acc (bit 0), `x << 16` if shift16 (bit 10), else `x`.
fn elementwise_16x16(
    mode: u32,
    acc1: &[i32; 64],
    acc2: &[i32; 64],
    a: &[u8; 64],
    b: &[u8; 64],
) -> [i32; 64] {
    let mut out = [0; 64];
    for i in 0..32 {
        let p = lane_ext(a, i, mode & 0x200 != 0) * lane_ext(b, i, mode & 0x100 != 0);
        let base = acc64(acc1, i);
        let base = if mode & 1 != 0 {
            0
        } else if mode & 0x400 != 0 {
            base.wrapping_shl(16)
        } else {
            base
        };
        set_acc64(
            &mut out,
            i,
            base.wrapping_add(acc64(acc2, i)).wrapping_add(p),
        );
    }
    out
}
fn lane_count(wide: bool) -> usize {
    if wide {
        16
    } else {
        32
    }
}
/// 16-bit (`wide` = false) or 32-bit lane `i` of a vector register, zero-extended.
fn lane(v: &[u8; 64], i: usize, wide: bool) -> u32 {
    if wide {
        u32::from_le_bytes(v[4 * i..4 * i + 4].try_into().unwrap())
    } else {
        u32::from(u16::from_le_bytes(v[2 * i..2 * i + 2].try_into().unwrap()))
    }
}
fn set_lane(v: &mut [u8; 64], i: usize, wide: bool, x: u32) {
    if wide {
        v[4 * i..4 * i + 4].copy_from_slice(&x.to_le_bytes())
    } else {
        v[2 * i..2 * i + 2].copy_from_slice(&(x as u16).to_le_bytes())
    }
}
fn lane_ext(v: &[u8; 64], i: usize, signed: bool) -> i64 {
    let w = lane(v, i, false);
    if signed {
        i64::from(w as u16 as i16)
    } else {
        i64::from(w)
    }
}
fn less(x: u32, y: u32, wide: bool, signed: bool) -> bool {
    if !signed {
        x < y
    } else if wide {
        (x as i32) < (y as i32)
    } else {
        (x as u16 as i16) < (y as u16 as i16)
    }
}
/// `vups`: `(sext_w|zext)(raw) << shift` into an acc`bits` (32 or 64) lane. `signed` = `Some(source width)`.
/// A result outside the accumulator lane range (signed range for signed sources, unsigned range otherwise)
/// is `Error::Unsupported`: the instruction reads `crSat`, whose overflow behaviour is not sourced.
fn ups_lane(raw: u32, signed: Option<u32>, shift: u32, bits: u32) -> Result<i64> {
    let v = match signed {
        Some(16) => i128::from(raw as u16 as i16),
        Some(_) => i128::from(raw as i32),
        None => i128::from(raw),
    } << shift;
    let (low, high) = if signed.is_some() {
        (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
    } else {
        (0, (1i128 << bits) - 1)
    };
    if v < low || v > high {
        return Err(Error::Unsupported);
    }
    Ok(v as i64)
}
/// `vsrs` lane shape: dm (4x) mode 0: 64 acc32 -> 64 int8, mode 1: 32 acc64 -> 32 int16; cm half (2x)
/// mode 0: 32 acc32 -> 32 int16, mode 1: 16 acc64 -> 16 int32. acc32 lanes allow shift <= 31 only.
fn srs_vector(
    acc: &[i32; 64],
    src: AccSource,
    mode: u32,
    shift: u32,
    rounding: u32,
    saturation: u32,
) -> Result<[u8; 64]> {
    let mut bytes = [0u8; 64];
    match (src, mode) {
        (AccSource::Dm(_), 0) => {
            for (byte, lane) in bytes.iter_mut().zip(acc) {
                *byte = srs_lane(*lane, shift, rounding, saturation)? as u8;
            }
        }
        (AccSource::Dm(_), _) => {
            for i in 0..32 {
                bytes[2 * i..2 * i + 2].copy_from_slice(
                    &(srs_round(acc64(acc, i), shift, rounding, saturation, 16)? as i16)
                        .to_le_bytes(),
                );
            }
        }
        (AccSource::Cm(_, h), 0) => {
            if shift > 31 {
                return Err(Error::Unsupported);
            }
            for i in 0..32 {
                let lane = i64::from(acc[32 * usize::from(h) + i]);
                bytes[2 * i..2 * i + 2].copy_from_slice(
                    &(srs_round(lane, shift, rounding, saturation, 16)? as i16).to_le_bytes(),
                );
            }
        }
        (AccSource::Cm(_, h), _) => {
            for i in 0..16 {
                let lane = acc64(acc, 16 * usize::from(h) + i);
                bytes[4 * i..4 * i + 4].copy_from_slice(
                    &(srs_round(lane, shift, rounding, saturation, 32)? as i32).to_le_bytes(),
                );
            }
        }
    }
    Ok(bytes)
}
/// `vunpack`: `crUnpackSize` 0 = 128 int4 nibbles (byte n/2, low nibble first) -> 128 int8 (y bytes 0..63 =
/// x_2k, 64..127 = x_2k+1); 1 = 64 int8 -> 64 int16 LE. Sign or zero extension per `unpackSign1/0`
/// (llvm-aie `aie2p_ldst.h` `unpack`, `__builtin_aie2p_unpack_I1024_I8_I4` / `_I1024_I16_I8`).
fn unpack(src: &[u8; 64], size: u32, signed: bool) -> Result<[[u8; 64]; 2]> {
    let mut y = [[0u8; 64]; 2];
    match size {
        0 => {
            for n in 0..128 {
                let nibble = if n % 2 == 0 {
                    src[n / 2] & 15
                } else {
                    src[n / 2] >> 4
                };
                y[n / 64][n % 64] = if signed {
                    ((nibble << 4) as i8 >> 4) as u8
                } else {
                    nibble
                };
            }
        }
        1 => {
            for n in 0..64 {
                let v = if signed {
                    i16::from(src[n] as i8) as u16
                } else {
                    u16::from(src[n])
                };
                y[n / 32][2 * (n % 32)..2 * (n % 32) + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        _ => return Err(Error::Unsupported),
    }
    Ok(y)
}
/// `vshuffle` (`aie2p_enums.h`): modes 0..=11 de-interleave elements of width 1,2,4,8,16,32 bytes
/// (`T8_64x2` .. `T256_2x2`, lo/hi): lo[j] = cat[2j], hi[j] = cat[2j+1] with cat = s1:s2;
/// modes 12..=21 interleave widths 16,8,4,2,1 (`T128_2x4` .. `T8_2x64`): seq = s1[0],s2[0],s1[1],s2[1],..,
/// lo = first 64 B, hi = second. Other modes are `Error::Unsupported`.
fn shuffle(a: &[u8; 64], b: &[u8; 64], mode: u32) -> Result<[u8; 64]> {
    let mut out = [0u8; 64];
    let hi = mode & 1 == 1;
    match mode {
        0..=11 => {
            let w = 1usize << (mode / 2);
            let n = 64 / w;
            for j in 0..n {
                let e = 2 * j + usize::from(hi);
                let src = if e < n {
                    &a[e * w..(e + 1) * w]
                } else {
                    &b[(e - n) * w..(e - n + 1) * w]
                };
                out[j * w..(j + 1) * w].copy_from_slice(src);
            }
        }
        12..=21 => {
            let w = 16usize >> ((mode - 12) / 2);
            let n = 64 / w;
            for j in 0..n {
                let e = j + if hi { n } else { 0 };
                let src = if e % 2 == 0 {
                    &a[(e / 2) * w..(e / 2 + 1) * w]
                } else {
                    &b[(e / 2) * w..(e / 2 + 1) * w]
                };
                out[j * w..(j + 1) * w].copy_from_slice(src);
            }
        }
        _ => return Err(Error::Unsupported),
    }
    Ok(out)
}
