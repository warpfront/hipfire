// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// AIE2P instruction fields and scheduling are derived from Xilinx/llvm-aie
// AIE2PGenInstrInfo.td and AIE2PGenSchedule.td (Apache-2.0 WITH LLVM-exception).
//! IEF15 N0 single-core **semantics probe** (design doc `docs/ief15-n0.md` §4 / §7 step 1).
//!
//! One core (column `col`, row 2, `gemm_i8`-style In/Out buffers through the shim and the memtile pass circuit)
//! acquires its input buffer, executes a fixed sequence of probe cases on edge operands, stores every result
//! register into the output buffer, releases it and halts. The expected output is *not* modelled here: it is the
//! output of the exact PDI + TXN run in `peacemaker_npu::sim::config::Config` (see `npu-gemm --ief15-probe` and
//! `crates/peacemaker-npu/tests/ief15_probe.rs`), so the silicon comparison is literally simulator vs silicon, bit for bit, and
//! every mismatch is named by a row of [`CASES`] (case id, instruction, operands, output byte range).
//!
//! ## Program discipline
//! * No branches (straight-line program), so there is no branch target; `Program::finish` still runs `isa::rules`.
//! * The program is emitted through a small dependency tracker ([`Gen`]): every register written by an
//!   instruction becomes readable [`LAT`] (8) bundles after its issue and every register is rewritten at least
//!   [`WAR`] bundles after its last read, so dependent instructions are always >= 8 cycles apart ("serial style",
//!   semantics, not pipeline timing) while independent instructions are packed back to back. The only tighter
//!   chains are the post-incremented pointer updates (one bundle), the form proven on silicon by `gemm_i8`'s
//!   four back-to-back accumulator stores.
//! * Locks: `C_EMPTY` is acquired, then the input `IN_FULL`; `C_FULL` is released after the last store retired.
//!
//! ## Core data memory (own view)
//! `IN_ADDR = 0` (the input pool, [`pool`]) then the output buffer at [`OUT_ADDR`]: a scalar area (masks and
//! scalar ALU results, 4 B each, `st [p2], #4`) followed by the vector area (`vst [p1], #64`).
use super::gemm_core::{descriptor, Assembler, CORE_BASE, C_EMPTY, C_FULL};
use super::gemm_i8::{ArgKind, ArgSpec};
use crate::{
    cdo::Cdo,
    dma::{self, BdLocks},
    isa::{self, bundle::encode_slot, gen::Slot, Inst, Program, Reg},
    regs,
    route::{Circuit, Port, ShimDma},
    txn::Txn,
};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Input DMA buffer base in the core's data memory.
pub const IN_ADDR: u32 = 0;
/// Operand-to-use latency assumed by the scheduler (>= every itinerary latency).
pub const LAT: usize = 8;
/// Minimum bundles between the last read of a register and its next write.
pub const WAR: usize = 4;
/// Marks an unused operand slot of a [`Case`].
pub const NONE: u16 = u16::MAX;

/// Operand pool: the input buffer is a list of 64-byte vectors; pool items are consecutive runs of them.
pub mod pool {
    pub const A16: usize = 0;
    pub const B16_0: usize = 1;
    pub const B16_1: usize = 2;
    pub const RA16: usize = 3;
    pub const RB16: usize = 4;
    pub const ACC1: usize = 5;
    pub const ACC2: usize = 6;
    pub const A32: usize = 7;
    pub const B32_0: usize = 8;
    pub const B32_1: usize = 9;
    pub const B32_2: usize = 10;
    pub const B32_3: usize = 11;
    pub const RA32: usize = 12;
    pub const RB32: usize = 13;
    pub const H0: usize = 14;
    pub const H1: usize = 15;
    pub const NEG: usize = 16;
    pub const SRS4M1: usize = 17;
    pub const SRS4M0: usize = 18;
    pub const SRS2M1: usize = 19;
    pub const SRS2M0: usize = 20;
    pub const UPS4: usize = 21;
    pub const UPS2M1: usize = 22;
    pub const UPS2M0: usize = 23;
    pub const UPS2M0U: usize = 24;
    pub const UNP: usize = 25;
    pub const SHUF: usize = 26;
    pub const I8: usize = 27;
    pub const ITEMS: usize = 28;
    /// Vectors per item: ACC1/ACC2/NEG/SRS* are 4-vector (256 B) accumulators; H0/H1 = L3, L2, L1, L0, P;
    /// UNP = 5 nibble patterns; SHUF = (s1, s2); I8 = A8, B8, A8b, B8b.
    pub const SIZE: [usize; ITEMS] = [
        1, 1, 1, 1, 1, 4, 4, 1, 1, 1, 1, 1, 1, 1, 5, 5, 4, 4, 4, 4, 4, 1, 1, 1, 1, 5, 2, 4,
    ];
    /// Pool vector index of vector `k` of `item`.
    pub const fn vec(item: usize, k: usize) -> u16 {
        let mut i = 0;
        let mut base = 0;
        while i < item {
            base += SIZE[i];
            i += 1;
        }
        (base + k) as u16
    }
    /// Total pool vectors.
    pub const VECS: usize = vec(ITEMS - 1, SIZE[ITEMS - 1]) as usize;
}
/// Input buffer bytes (pool rounded up to 256 B).
pub const IN_BYTES: usize = (pool::VECS * 64 + 255) / 256 * 256;
/// Output buffer base in core data memory.
pub const OUT_ADDR: u32 = IN_BYTES as u32;

/// Probe instruction families. Parameter meaning (`Case::p`, operand pool vectors `Case::s = [A, B, acc1, acc2]`):
/// * `Vmul/Vmac/Vaddmac`: `p0` = conf (`0x5a | sx<<9 | sy<<8 | shift16<<10 | zero_acc`), 256 B (32 acc64 lanes).
/// * `HornerMul/HornerMac`: `p0` = conf, `p1` = step; `s0` = limb vector, `s1` = power vector; chain on dm0.
/// * `Vneg`: `p0` = conf (2: acc64, 0: acc32), `s2` = acc source.
/// * `AccRt`: raw `VLDA bm` / `VST bm` round trip of `s2`.
/// * `Srs4/Srs2`: `p0` = crSRSMode, `p1` = shift, `p2` = crRnd, `p3` = crSat | half << 8 (`Srs2`: cml/cmh), `s2` source.
/// * `Ups4/Ups2`: `p0` = crUPSMode, `p1` = upsSign, `p2` = shift, `p3` = half (`Ups2`), `s0` source x vector.
/// * `VldbUnpack/Vunpack`: `p0` = unpackSign, `p1` = crUnpackSize, `s0` source (128 B out).
/// * `VldbX`: plain `VLDB x` load of `s0` then `VST x` (64 B round trip).
/// * `Shuffle`: `p0` = mode, `s0`/`s1` = s1/s2.
/// * `Vlt*/Vge*`: `p0` = sign (0 unsigned, 1 signed); `Veqz*` tests `s1`'s vector (the s2 operand); masks.
/// * `Vsel*`: `p0` = 0: constant mask `p1`; 1: mask from `VLT` signed on (A, B).
/// * `Vadd/Vsub/Vband/Vbor/Vmax/Vmin`: lane ops on (A, B); `Vmax/Vmin` have `part` 0 = vector, 1 = r16 mask.
/// * `Vbcst*`: `p0` = scalar value.
/// * `I8Mul/I8Mac`: `p0` = conf 0x308, `p1` = step; `s` = the four I8 vectors.
/// * `SAnd/SOr/SLshl/SAdd/SEqz/SNez`: `p0`, `p1` = scalar operands (a, b).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Vmul,
    Vmac,
    Vaddmac,
    HornerMul,
    HornerMac,
    Vneg,
    AccRt,
    Srs4,
    Srs2,
    Ups4,
    Ups2,
    VldbUnpack,
    Vunpack,
    VldbX,
    Shuffle,
    Vlt16,
    Vge16,
    Vlt32,
    Vge32,
    Veqz16,
    Veqz32,
    Vsel16,
    Vsel32,
    Vadd16,
    Vsub16,
    Vadd32,
    Vsub32,
    Vband,
    Vbor,
    Vmax16,
    Vmin16,
    Vmax32,
    Vmin32,
    Vbcst16,
    Vbcst32,
    I8Mul,
    I8Mac,
    SAnd,
    SOr,
    SLshl,
    SAdd,
    SEqz,
    SNez,
}

/// One probe case: a named byte range of the output buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Case {
    /// Row index (dense, in program order).
    pub id: u16,
    /// Rows of one group share operand loads and are emitted together.
    pub group: u16,
    pub op: Op,
    /// 1 for the r16 mask half of `Vmax/Vmin`.
    pub part: u8,
    pub p: [i32; 4],
    /// Pool vector indices `[A, B, acc1/src, acc2]`, [`NONE`] if unused.
    pub s: [u16; 4],
    /// Byte offset / length in the output buffer.
    pub out_off: u32,
    pub out_len: u16,
}
impl Case {
    const EMPTY: Case = Case {
        id: 0,
        group: 0,
        op: Op::Vmul,
        part: 0,
        p: [0; 4],
        s: [NONE; 4],
        out_off: 0,
        out_len: 0,
    };
    /// The (primary) encoding name executed by the row.
    pub fn instruction(&self) -> &'static str {
        let sign = |a: &'static str, b: &'static str| if self.p[1] == 0 { a } else { b };
        match (self.op, self.part) {
            (Op::Vmul | Op::HornerMul | Op::I8Mul, _) => "VMUL_vmul_cm_core_X_X",
            (Op::Vmac | Op::HornerMac | Op::I8Mac, _) => "VMAC_vmul_cm_core_X_X",
            (Op::Vaddmac, _) => "VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_X",
            (Op::Vneg, _) => "VNEG",
            (Op::AccRt, _) => "VST_dmx_sts_bm_pstm_nrm_imm",
            (Op::Srs4, _) => "VSRS_4x_mv_x_srs_dm_srsSign1",
            (Op::Srs2, _) => "VSRS_2x_mv_x_srs_cm_srsSign1",
            (Op::Ups4, _) => sign("VUPS_4x_mv_ups_x2d_upsSign0", "VUPS_4x_mv_ups_x2d_upsSign1"),
            (Op::Ups2, _) => sign("VUPS_2x_mv_ups_x2c_upsSign0", "VUPS_2x_mv_ups_x2c_upsSign1"),
            (Op::VldbUnpack, _) => {
                if self.p[0] == 0 {
                    "VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_imm_unpackSign0"
                } else {
                    "VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_imm_unpackSign1"
                }
            }
            (Op::Vunpack, _) => {
                if self.p[0] == 0 {
                    "VUNPACK_mv_unpack_x_unpackSign0"
                } else {
                    "VUNPACK_mv_unpack_x_unpackSign1"
                }
            }
            (Op::VldbX, _) => "VLDB_dmx_ldb_x_pstm_nrm_imm",
            (Op::Shuffle, _) => "VSHUFFLE_vec_shuffle_x",
            (Op::Vlt16, _) => {
                if self.p[0] == 0 {
                    "VLT_16_vaddSign0"
                } else {
                    "VLT_16_vaddSign1"
                }
            }
            (Op::Vge16, _) => {
                if self.p[0] == 0 {
                    "VGE_16_vaddSign0"
                } else {
                    "VGE_16_vaddSign1"
                }
            }
            (Op::Vlt32, _) => {
                if self.p[0] == 0 {
                    "VLT_32_vaddSign0"
                } else {
                    "VLT_32_vaddSign1"
                }
            }
            (Op::Vge32, _) => {
                if self.p[0] == 0 {
                    "VGE_32_vaddSign0"
                } else {
                    "VGE_32_vaddSign1"
                }
            }
            (Op::Veqz16, _) => "VEQZ_16",
            (Op::Veqz32, _) => "VEQZ_32",
            (Op::Vsel16, _) => "VSEL_16",
            (Op::Vsel32, _) => "VSEL_32",
            (Op::Vadd16, _) => "VADD_16",
            (Op::Vsub16, _) => "VSUB_16",
            (Op::Vadd32, _) => "VADD_32",
            (Op::Vsub32, _) => "VSUB_32",
            (Op::Vband, _) => "VBAND",
            (Op::Vbor, _) => "VBOR",
            (Op::Vmax16, _) => "VMAX_LT_16_vaddSign1",
            (Op::Vmin16, _) => "VMIN_GE_16_vaddSign1",
            (Op::Vmax32, _) => "VMAX_LT_32_vaddSign1",
            (Op::Vmin32, _) => "VMIN_GE_32_vaddSign1",
            (Op::Vbcst16, _) => "VBCST_16",
            (Op::Vbcst32, _) => "VBCST_32",
            (Op::SAnd, _) => "AND",
            (Op::SOr, _) => "OR",
            (Op::SLshl, _) => "LSHL",
            (Op::SAdd, _) => "ADD_alu_r_rr",
            (Op::SEqz, _) => "EQZ",
            (Op::SNez, _) => "NEZ",
        }
    }
    /// Human-readable operands (parameters and operand vectors) for the mismatch table.
    pub fn operands(&self) -> String {
        let v = |i: usize| {
            if self.s[i] == NONE {
                String::from("-")
            } else {
                format!("v{}", self.s[i])
            }
        };
        let half = |h: i32| if h == 0 { "lo" } else { "hi" };
        let body = match self.op {
            Op::Vmul | Op::Vmac | Op::Vaddmac => format!(
                "conf={:#x} A={} B={} acc1={} acc2={}",
                self.p[0],
                v(0),
                v(1),
                v(2),
                v(3)
            ),
            Op::HornerMul | Op::HornerMac => format!(
                "step={} conf={:#x} limb={} p={}",
                self.p[1],
                self.p[0],
                v(0),
                v(1)
            ),
            Op::Vneg => format!("conf={:#x} acc={}", self.p[0], v(2)),
            Op::AccRt => format!("acc={}", v(2)),
            Op::Srs4 | Op::Srs2 => format!(
                "crSRSMode={} shift={} crRnd={} crSat={}{} src={}",
                self.p[0],
                self.p[1],
                self.p[2],
                self.p[3] & 0xff,
                if self.op == Op::Srs2 {
                    format!(" half={}", half(self.p[3] >> 8))
                } else {
                    String::new()
                },
                v(2)
            ),
            Op::Ups4 | Op::Ups2 => format!(
                "crUPSMode={} sign={} shift={}{} src={}",
                self.p[0],
                self.p[1],
                self.p[2],
                if self.op == Op::Ups2 {
                    format!(" half={}", half(self.p[3]))
                } else {
                    String::new()
                },
                v(0)
            ),
            Op::VldbUnpack | Op::Vunpack => format!(
                "unpackSign={} crUnpackSize={} src={}",
                self.p[0],
                self.p[1],
                v(0)
            ),
            Op::VldbX => format!("src={}", v(0)),
            Op::Shuffle => format!("mode={} s1={} s2={}", self.p[0], v(0), v(1)),
            Op::Vlt16 | Op::Vge16 | Op::Vlt32 | Op::Vge32 => {
                format!("sign={} s1={} s2={}", self.p[0], v(0), v(1))
            }
            Op::Veqz16 | Op::Veqz32 => format!("s2={}", v(1)),
            Op::Vsel16 | Op::Vsel32 => {
                if self.p[0] == 0 {
                    format!("mask={:#x} s1={} s2={}", self.p[1] as u32, v(0), v(1))
                } else {
                    format!("mask=VLT.signed s1={} s2={}", v(0), v(1))
                }
            }
            Op::Vmax16 | Op::Vmin16 | Op::Vmax32 | Op::Vmin32 => format!(
                "{} s1={} s2={}",
                if self.part == 0 { "d" } else { "r16" },
                v(0),
                v(1)
            ),
            Op::Vbcst16 | Op::Vbcst32 => format!("src={:#x}", self.p[0] as u32),
            Op::I8Mul | Op::I8Mac => format!("conf={:#x} step={}", self.p[0], self.p[1]),
            Op::SEqz | Op::SNez => format!("a={:#x}", self.p[0] as u32),
            _ if matches!(
                self.op,
                Op::Vadd16 | Op::Vsub16 | Op::Vadd32 | Op::Vsub32 | Op::Vband | Op::Vbor
            ) =>
            {
                format!("s1={} s2={}", v(0), v(1))
            }
            _ => format!("a={:#x} b={:#x}", self.p[0] as u32, self.p[1] as u32),
        };
        body
    }
    /// True when the row's result is a 4-byte scalar (mask / ALU) in the scalar output area.
    pub fn scalar(&self) -> bool {
        self.out_len == 4
    }
    /// Width in bytes of one result lane (acc64 lane 8, acc32 / int32 4, int16 2, byte-granular 1, scalar word 4).
    pub fn lane_bytes(&self) -> usize {
        if self.scalar() {
            return 4;
        }
        match self.op {
            Op::Vmul
            | Op::Vmac
            | Op::Vaddmac
            | Op::HornerMul
            | Op::HornerMac
            | Op::Ups4
            | Op::AccRt => 8,
            Op::Vneg => {
                if self.p[0] == 2 {
                    8
                } else {
                    4
                }
            }
            Op::Srs4 => {
                if self.p[0] == 1 {
                    2
                } else {
                    1
                }
            }
            Op::Srs2 => {
                if self.p[0] == 1 {
                    4
                } else {
                    2
                }
            }
            Op::Ups2 => {
                if self.p[0] == 1 {
                    8
                } else {
                    4
                }
            }
            Op::I8Mul
            | Op::I8Mac
            | Op::Vadd32
            | Op::Vsub32
            | Op::Vsel32
            | Op::Vmax32
            | Op::Vmin32
            | Op::Vbcst32 => 4,
            Op::Vadd16 | Op::Vsub16 | Op::Vsel16 | Op::Vmax16 | Op::Vmin16 | Op::Vbcst16 => 2,
            _ => 1,
        }
    }
    /// Little-endian value of result lane `lane` in an output buffer.
    pub fn lane_value(&self, buf: &[u8], lane: usize) -> u64 {
        let w = self.lane_bytes();
        let o = self.out_off as usize + lane * w;
        buf[o..o + w]
            .iter()
            .rev()
            .fold(0u64, |v, &b| v << 8 | u64::from(b))
    }
}

const MAX_CASES: usize = 640;
struct Built {
    cases: [Case; MAX_CASES],
    n: usize,
    vec_end: u32,
    sc_end: u32,
}
struct Builder {
    cases: [Case; MAX_CASES],
    n: usize,
    group: u16,
    vec_off: u32,
    sc_off: u32,
}
impl Builder {
    const fn push(&mut self, op: Op, part: u8, p: [i32; 4], s: [u16; 4], len: u16) {
        let scalar = len == 4;
        let off = if scalar {
            let o = self.sc_off;
            self.sc_off += 4;
            o
        } else {
            let o = self.vec_off;
            self.vec_off += len as u32;
            o
        };
        self.cases[self.n] = Case {
            id: self.n as u16,
            group: self.group,
            op,
            part,
            p,
            s,
            out_off: off,
            out_len: len,
        };
        self.n += 1;
    }
    const fn group(&mut self) {
        self.group += 1;
    }
}
const fn conf(sg: i32, sh: i32, z: i32) -> i32 {
    0x5a | (((sg >> 1) & 1) << 9) | ((sg & 1) << 8) | (sh << 10) | z
}
const fn s4(a: u16, b: u16, c: u16, d: u16) -> [u16; 4] {
    [a, b, c, d]
}
const fn p4(a: i32, b: i32, c: i32, d: i32) -> [i32; 4] {
    [a, b, c, d]
}
const P16: [(u16, u16); 3] = [
    (pool::vec(pool::A16, 0), pool::vec(pool::B16_0, 0)),
    (pool::vec(pool::A16, 0), pool::vec(pool::B16_1, 0)),
    (pool::vec(pool::RA16, 0), pool::vec(pool::RB16, 0)),
];
const P32: [(u16, u16); 5] = [
    (pool::vec(pool::A32, 0), pool::vec(pool::B32_0, 0)),
    (pool::vec(pool::A32, 0), pool::vec(pool::B32_1, 0)),
    (pool::vec(pool::A32, 0), pool::vec(pool::B32_2, 0)),
    (pool::vec(pool::A32, 0), pool::vec(pool::B32_3, 0)),
    (pool::vec(pool::RA32, 0), pool::vec(pool::RB32, 0)),
];
/// SRS shift sets per crSRSMode: acc64 -> 7 shifts, acc32 -> shifts <= 31.
const SRS_SHIFTS1: [i32; 7] = [0, 7, 16, 23, 32, 39, 48];
const SRS_SHIFTS0: [i32; 4] = [0, 7, 16, 23];
const UPS_SHIFTS: [i32; 3] = [0, 7, 16];
const SCALAR_ARITH: [(u32, u32); 6] = [
    (0, 0),
    (0xffff_ffff, 1),
    (0x8000_0000, 0x8000_0000),
    (0x7fff_ffff, 1),
    (0x1234_5678, 0xf0f0_f0f0),
    (0xffff_ffff, 0xffff_ffff),
];
const SCALAR_SHIFT: [(u32, u32); 6] = [
    (1, 0),
    (1, 31),
    (0xffff_ffff, 1),
    (0x8000_0001, 16),
    (0x1234_5678, 7),
    (0xf000_0000, 4),
];
const SCALAR_UNARY: [u32; 4] = [0, 1, 0xffff_ffff, 0x8000_0000];
const BCST16: [u32; 3] = [0x0001_abcd, 0xffff_8001, 0x1234_5678];
const BCST32: [u32; 3] = [0x8000_0001, 0x7fff_ffff, 0];
/// Constant select masks (16-bit lanes: 32 bits; 32-bit lanes: 16 bits).
const SEL_MASK16: [u32; 2] = [0xa5c3_0f96, !0xa5c3_0f96];
const SEL_MASK32: [u32; 2] = [0x0000_9c35, 0x0000_63ca];

const fn build(vec_base: u32) -> Built {
    use pool::vec;
    let mut b = Builder {
        cases: [Case::EMPTY; MAX_CASES],
        n: 0,
        group: 0,
        vec_off: vec_base,
        sc_off: 0,
    };
    // F1: elementwise 16x16 -> acc64 over edge operand sets.
    let mut si = 0;
    while si < 3 {
        let (a, bv) = P16[si];
        let (acc1, acc2) = (vec(pool::ACC1, 0), vec(pool::ACC2, 0));
        let full = si < 2;
        b.group();
        let mut sg = 0;
        while sg < 4 {
            b.push(
                Op::Vmul,
                0,
                p4(conf(sg, 0, 0), 0, 0, 0),
                s4(a, bv, NONE, NONE),
                256,
            );
            sg += 1;
        }
        let mut kind = 0;
        while kind < 2 {
            b.group();
            let mut sg = 0;
            while sg < 4 {
                let mut sh = 0;
                while sh < 2 {
                    let mut z = 0;
                    while z < 2 {
                        if full || (sh == 1 && z == 0) {
                            if kind == 0 {
                                b.push(
                                    Op::Vmac,
                                    0,
                                    p4(conf(sg, sh, z), 0, 0, 0),
                                    s4(a, bv, acc1, NONE),
                                    256,
                                );
                            } else {
                                b.push(
                                    Op::Vaddmac,
                                    0,
                                    p4(conf(sg, sh, z), 0, 0, 0),
                                    s4(a, bv, acc1, acc2),
                                    256,
                                );
                            }
                        }
                        z += 1;
                    }
                    sh += 1;
                }
                sg += 1;
            }
            kind += 1;
        }
        si += 1;
    }
    // F2: 4-step Horner chains acc = (acc << 16) + limb * p.
    let mut set = 0;
    while set < 2 {
        let item = if set == 0 { pool::H0 } else { pool::H1 };
        b.group();
        let mut step = 0;
        while step < 4 {
            let op = if step == 0 {
                Op::HornerMul
            } else {
                Op::HornerMac
            };
            let cf = if step == 0 {
                conf(0, 0, 0)
            } else {
                conf(0, 1, 0)
            };
            b.push(
                op,
                0,
                p4(cf, step, 0, 0),
                s4(vec(item, step as usize), vec(item, 4), NONE, NONE),
                256,
            );
            step += 1;
        }
        set += 1;
    }
    // F3/F4: VNEG (acc64 and acc32) and the raw bm round trip.
    b.group();
    b.push(
        Op::Vneg,
        0,
        p4(2, 0, 0, 0),
        s4(NONE, NONE, vec(pool::NEG, 0), NONE),
        256,
    );
    b.push(
        Op::Vneg,
        0,
        p4(0, 0, 0, 0),
        s4(NONE, NONE, vec(pool::NEG, 0), NONE),
        256,
    );
    b.group();
    b.push(
        Op::AccRt,
        0,
        p4(0, 0, 0, 0),
        s4(NONE, NONE, vec(pool::ACC1, 0), NONE),
        256,
    );
    // F5: VSRS_4x dm->x and VSRS_2x cm->x.
    let mut kind = 0;
    while kind < 2 {
        let mut mode = 1;
        while mode >= 0 {
            b.group();
            let src = match (kind, mode) {
                (0, 1) => pool::SRS4M1,
                (0, _) => pool::SRS4M0,
                (_, 1) => pool::SRS2M1,
                _ => pool::SRS2M0,
            };
            let nsh = if mode == 1 { 7 } else { 4 };
            let halves = if kind == 0 { 1 } else { 2 };
            let mut rnd = 0;
            while rnd < 2 {
                let mut sat = 0;
                while sat < 2 {
                    let mut k = 0;
                    while k < nsh {
                        let shift = if mode == 1 {
                            SRS_SHIFTS1[k]
                        } else {
                            SRS_SHIFTS0[k]
                        };
                        let mut h = 0;
                        while h < halves {
                            let op = if kind == 0 { Op::Srs4 } else { Op::Srs2 };
                            b.push(
                                op,
                                0,
                                p4(mode, shift, if rnd == 0 { 0 } else { 12 }, sat | (h << 8)),
                                s4(NONE, NONE, vec(src, 0), NONE),
                                64,
                            );
                            h += 1;
                        }
                        k += 1;
                    }
                    sat += 1;
                }
                rnd += 1;
            }
            mode -= 1;
        }
        kind += 1;
    }
    // F6: VUPS_4x x->dm (crUPSMode 1) and VUPS_2x x->cm (crUPSMode 1 and 0).
    b.group();
    let mut sign = 0;
    while sign < 2 {
        let mut k = 0;
        while k < 3 {
            b.push(
                Op::Ups4,
                0,
                p4(1, sign, UPS_SHIFTS[k], 0),
                s4(vec(pool::UPS4, 0), NONE, NONE, NONE),
                256,
            );
            k += 1;
        }
        sign += 1;
    }
    let mut mode = 1;
    while mode >= 0 {
        b.group();
        let mut sign = 0;
        while sign < 2 {
            let mut k = 0;
            while k < 3 {
                let src = if mode == 1 {
                    pool::UPS2M1
                } else if sign == 0 && k == 2 {
                    pool::UPS2M0U
                } else {
                    pool::UPS2M0
                };
                let mut h = 0;
                while h < 2 {
                    b.push(
                        Op::Ups2,
                        0,
                        p4(mode, sign, UPS_SHIFTS[k], h),
                        s4(vec(src, 0), NONE, NONE, NONE),
                        128,
                    );
                    h += 1;
                }
                k += 1;
            }
            sign += 1;
        }
        mode -= 1;
    }
    // F7: VLDB.UNPACK / VUNPACK, crUnpackSize 0.
    let mut kind = 0;
    while kind < 2 {
        b.group();
        let mut sign = 1;
        while sign >= 0 {
            let mut k = 0;
            while k < 5 {
                let op = if kind == 0 {
                    Op::VldbUnpack
                } else {
                    Op::Vunpack
                };
                b.push(
                    op,
                    0,
                    p4(sign, 0, 0, 0),
                    s4(vec(pool::UNP, k), NONE, NONE, NONE),
                    128,
                );
                k += 1;
            }
            sign -= 1;
        }
        kind += 1;
    }
    // F7b: plain VLDB x load round trip.
    b.group();
    b.push(
        Op::VldbX,
        0,
        p4(0, 0, 0, 0),
        s4(vec(pool::A16, 0), NONE, NONE, NONE),
        64,
    );
    b.push(
        Op::VldbX,
        0,
        p4(0, 0, 0, 0),
        s4(vec(pool::RA16, 0), NONE, NONE, NONE),
        64,
    );
    b.push(
        Op::VldbX,
        0,
        p4(0, 0, 0, 0),
        s4(vec(pool::RB32, 0), NONE, NONE, NONE),
        64,
    );
    // F8: VSHUFFLE interleave modes 12..19 and de-interleave modes 2..9.
    b.group();
    let mut m = 0;
    while m < 16 {
        let mode = (if m < 8 { 12 + m } else { 2 + (m - 8) }) as i32;
        b.push(
            Op::Shuffle,
            0,
            p4(mode, 0, 0, 0),
            s4(vec(pool::SHUF, 0), vec(pool::SHUF, 1), NONE, NONE),
            64,
        );
        m += 1;
    }
    // F9: lane compares (masks), VEQZ.
    let mut i = 0;
    while i < 8 {
        let (a, bv, wide) = if i < 3 {
            (P16[i].0, P16[i].1, false)
        } else {
            (P32[i - 3].0, P32[i - 3].1, true)
        };
        b.group();
        let (lt, ge, eqz) = if wide {
            (Op::Vlt32, Op::Vge32, Op::Veqz32)
        } else {
            (Op::Vlt16, Op::Vge16, Op::Veqz16)
        };
        let mut sg = 0;
        while sg < 2 {
            b.push(lt, 0, p4(sg, 0, 0, 0), s4(a, bv, NONE, NONE), 4);
            sg += 1;
        }
        let mut sg = 0;
        while sg < 2 {
            b.push(ge, 0, p4(sg, 0, 0, 0), s4(a, bv, NONE, NONE), 4);
            sg += 1;
        }
        b.push(eqz, 0, p4(0, 0, 0, 0), s4(a, bv, NONE, NONE), 4);
        i += 1;
    }
    // F10: VSEL (constant masks and VLT-derived masks).
    let mut wide = 0;
    while wide < 2 {
        let (a, bv) = if wide == 0 { P16[0] } else { P32[0] };
        b.group();
        let op = if wide == 0 { Op::Vsel16 } else { Op::Vsel32 };
        let mut k = 0;
        while k < 2 {
            let mask = if wide == 0 {
                SEL_MASK16[k]
            } else {
                SEL_MASK32[k]
            };
            b.push(op, 0, p4(0, mask as i32, 0, 0), s4(a, bv, NONE, NONE), 64);
            k += 1;
        }
        b.push(op, 0, p4(1, 0, 0, 0), s4(a, bv, NONE, NONE), 64);
        wide += 1;
    }
    // F11: VADD/VSUB (wraparound), VBAND, VBOR, VMAX_LT / VMIN_GE (+ r16 mask).
    let mut i = 0;
    while i < 8 {
        let (a, bv, wide) = if i < 3 {
            (P16[i].0, P16[i].1, false)
        } else {
            (P32[i - 3].0, P32[i - 3].1, true)
        };
        b.group();
        if wide {
            b.push(Op::Vadd32, 0, p4(0, 0, 0, 0), s4(a, bv, NONE, NONE), 64);
            b.push(Op::Vsub32, 0, p4(0, 0, 0, 0), s4(a, bv, NONE, NONE), 64);
        } else {
            b.push(Op::Vadd16, 0, p4(0, 0, 0, 0), s4(a, bv, NONE, NONE), 64);
            b.push(Op::Vsub16, 0, p4(0, 0, 0, 0), s4(a, bv, NONE, NONE), 64);
            b.push(Op::Vband, 0, p4(0, 0, 0, 0), s4(a, bv, NONE, NONE), 64);
            b.push(Op::Vbor, 0, p4(0, 0, 0, 0), s4(a, bv, NONE, NONE), 64);
        }
        b.group();
        let (mx, mn) = if wide {
            (Op::Vmax32, Op::Vmin32)
        } else {
            (Op::Vmax16, Op::Vmin16)
        };
        b.push(mx, 0, p4(0, 0, 0, 0), s4(a, bv, NONE, NONE), 64);
        b.push(mx, 1, p4(0, 0, 0, 0), s4(a, bv, NONE, NONE), 4);
        b.push(mn, 0, p4(0, 0, 0, 0), s4(a, bv, NONE, NONE), 64);
        b.push(mn, 1, p4(0, 0, 0, 0), s4(a, bv, NONE, NONE), 4);
        i += 1;
    }
    // F12: VBCST.
    b.group();
    let mut k = 0;
    while k < 3 {
        b.push(
            Op::Vbcst16,
            0,
            p4(BCST16[k] as i32, 0, 0, 0),
            s4(NONE, NONE, NONE, NONE),
            64,
        );
        k += 1;
    }
    let mut k = 0;
    while k < 3 {
        b.push(
            Op::Vbcst32,
            0,
            p4(BCST32[k] as i32, 0, 0, 0),
            s4(NONE, NONE, NONE, NONE),
            64,
        );
        k += 1;
    }
    // F13: int8 8x8x8 matrix control (conf 0x308).
    b.group();
    let i8s = s4(
        vec(pool::I8, 0),
        vec(pool::I8, 1),
        vec(pool::I8, 2),
        vec(pool::I8, 3),
    );
    b.push(Op::I8Mul, 0, p4(0x308, 0, 0, 0), i8s, 256);
    b.push(Op::I8Mac, 0, p4(0x308, 1, 0, 0), i8s, 256);
    b.push(Op::I8Mac, 0, p4(0x308, 2, 0, 0), i8s, 256);
    // F14: scalar ALU.
    b.group();
    let mut k = 0;
    while k < 6 {
        b.push(
            Op::SAnd,
            0,
            p4(SCALAR_ARITH[k].0 as i32, SCALAR_ARITH[k].1 as i32, 0, 0),
            s4(NONE, NONE, NONE, NONE),
            4,
        );
        k += 1;
    }
    let mut k = 0;
    while k < 6 {
        b.push(
            Op::SOr,
            0,
            p4(SCALAR_ARITH[k].0 as i32, SCALAR_ARITH[k].1 as i32, 0, 0),
            s4(NONE, NONE, NONE, NONE),
            4,
        );
        k += 1;
    }
    let mut k = 0;
    while k < 6 {
        b.push(
            Op::SAdd,
            0,
            p4(SCALAR_ARITH[k].0 as i32, SCALAR_ARITH[k].1 as i32, 0, 0),
            s4(NONE, NONE, NONE, NONE),
            4,
        );
        k += 1;
    }
    let mut k = 0;
    while k < 6 {
        b.push(
            Op::SLshl,
            0,
            p4(SCALAR_SHIFT[k].0 as i32, SCALAR_SHIFT[k].1 as i32, 0, 0),
            s4(NONE, NONE, NONE, NONE),
            4,
        );
        k += 1;
    }
    let mut k = 0;
    while k < 4 {
        b.push(
            Op::SEqz,
            0,
            p4(SCALAR_UNARY[k] as i32, 0, 0, 0),
            s4(NONE, NONE, NONE, NONE),
            4,
        );
        k += 1;
    }
    let mut k = 0;
    while k < 4 {
        b.push(
            Op::SNez,
            0,
            p4(SCALAR_UNARY[k] as i32, 0, 0, 0),
            s4(NONE, NONE, NONE, NONE),
            4,
        );
        k += 1;
    }
    Built {
        cases: b.cases,
        n: b.n,
        vec_end: b.vec_off,
        sc_end: b.sc_off,
    }
}
const fn round64(x: u32) -> u32 {
    (x + 63) / 64 * 64
}
/// Bytes of the scalar output area (the vector area follows it).
pub const SCALAR_BYTES: usize = round64(build(0).sc_end) as usize;
const BUILT: Built = build(SCALAR_BYTES as u32);
/// Number of probe cases (rows).
pub const CASE_COUNT: usize = BUILT.n;
/// Output buffer bytes (scalar area + vector area, rounded up to 256 B so the DMA lengths are round).
pub const OUT_BYTES: usize = (BUILT.vec_end as usize + 255) / 256 * 256;
/// The case table (program order = output order within each area).
pub const CASES: [Case; CASE_COUNT] = {
    let mut a = [Case::EMPTY; CASE_COUNT];
    let mut i = 0;
    while i < CASE_COUNT {
        a[i] = BUILT.cases[i];
        i += 1;
    }
    a
};

// ---------------------------------------------------------------------------------------------------------------
// Input pool
// ---------------------------------------------------------------------------------------------------------------
const EV16: [u16; 8] = [
    0x0000, 0x0001, 0xffff, 0x7fff, 0x8000, 0x8001, 0x00ff, 0x4000,
];
const EV32: [u32; 8] = [
    0,
    1,
    0xffff_ffff,
    0x7fff_ffff,
    0x8000_0000,
    0x8000,
    0xffff,
    0x8000_0001,
];
const EDGE64: [i64; 8] = [
    0,
    1,
    -1,
    i64::MAX,
    i64::MIN,
    1 << 32,
    0xffff_ffff,
    i64::MIN + 1,
];
const EDGE8: [i8; 8] = [-128, 127, 0, 1, -1, -127, 2, -2];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// One accumulator lane for the SRS probes: pattern `q` at shift `s` (halfway ties, saturation / wrap edges of the
/// `out_bits` result, just below / above ties), wrapped to the accumulator width. Patterns >= 16 are random.
fn srs_value(q: usize, s: u32, acc_bits: u32, out_bits: u32, rng: &mut Rng) -> i64 {
    let one = 1i128 << s;
    let half = if s > 0 { 1i128 << (s - 1) } else { 0 };
    let maxo = (1i128 << (out_bits - 1)) - 1;
    let mino = -(1i128 << (out_bits - 1));
    let v: i128 = match q {
        0 => one + half,
        1 => 2 * one + half,
        2 => -3 * one + half,
        3 => (maxo << s) + half,
        4 => one + half - 1,
        5 => one + half + 1,
        6 => (mino << s) - half,
        7 => -one + half,
        8 => half,
        9 => (maxo + 1) << s,
        10 => (mino - 1) << s,
        11 => (((1i128 << out_bits) + 5) << s) + half,
        12 => -2 * one + half,
        13 => 3 * one + half,
        14 => (maxo << s) + one - 1,
        15 => (mino << s) - one + 1,
        _ => (rng.next() as i64 >> (rng.next() % 40)) as i128,
    };
    if acc_bits == 32 {
        v as i32 as i64
    } else {
        v as i64
    }
}
/// Lanes of one SRS source half: lane `j` uses shift `shifts[j % n]` and pattern `pat_off + j / n`.
fn srs_half(
    lanes: usize,
    acc_bits: u32,
    out_bits: u32,
    shifts: &[i32],
    pat_off: usize,
    rng: &mut Rng,
) -> Vec<i64> {
    (0..lanes)
        .map(|j| {
            srs_value(
                pat_off + j / shifts.len(),
                shifts[j % shifts.len()] as u32,
                acc_bits,
                out_bits,
                rng,
            )
        })
        .collect()
}
fn put64(out: &mut Vec<u8>, lanes: &[i64]) {
    for v in lanes {
        out.extend_from_slice(&v.to_le_bytes());
    }
}
fn put32(out: &mut Vec<u8>, lanes: &[i64]) {
    for v in lanes {
        out.extend_from_slice(&(*v as i32).to_le_bytes());
    }
}

/// The input buffer (`IN_BYTES`): every pool item in [`pool`] order, deterministic.
pub fn input() -> Vec<u8> {
    let mut rng = Rng(0x1ef1_5eed_0000_0001);
    let mut out: Vec<u8> = Vec::with_capacity(IN_BYTES);
    let u16s = |out: &mut Vec<u8>, n: usize, mut f: Box<dyn FnMut(usize) -> u16>| {
        for i in 0..n {
            out.extend_from_slice(&f(i).to_le_bytes());
        }
    };
    let u32s = |out: &mut Vec<u8>, n: usize, mut f: Box<dyn FnMut(usize) -> u32>| {
        for i in 0..n {
            out.extend_from_slice(&f(i).to_le_bytes());
        }
    };
    for item in 0..pool::ITEMS {
        let start = out.len();
        match item {
            pool::A16 => u16s(&mut out, 32, Box::new(|i| EV16[i & 7])),
            pool::B16_0 => u16s(&mut out, 32, Box::new(|i| EV16[(i >> 3) & 3])),
            pool::B16_1 => u16s(&mut out, 32, Box::new(|i| EV16[4 + ((i >> 3) & 3)])),
            pool::RA16 | pool::RB16 => {
                for _ in 0..32 {
                    out.extend_from_slice(&(rng.next() as u16).to_le_bytes());
                }
            }
            pool::ACC1 | pool::ACC2 | pool::NEG => {
                let rot = if item == pool::ACC2 { 3 } else { 0 };
                let lanes: Vec<i64> = (0..32)
                    .map(|i| {
                        if i < 8 {
                            EDGE64[(i + rot) & 7]
                        } else if i < 16 && item == pool::NEG {
                            [
                                1i64 << 31,
                                -(1i64 << 31),
                                (1 << 31) - 1,
                                -(1i64 << 31) - 1,
                                1 << 62,
                                -(1 << 62),
                                0x7fff_ffff_0000_0000,
                                -0x7fff_ffff_0000_0000,
                            ][i - 8]
                        } else {
                            rng.next() as i64 >> (rng.next() % 40)
                        }
                    })
                    .collect();
                put64(&mut out, &lanes);
            }
            pool::A32 => u32s(&mut out, 16, Box::new(|i| EV32[i & 7])),
            pool::B32_0..=pool::B32_3 => {
                let j = item - pool::B32_0;
                u32s(&mut out, 16, Box::new(move |i| EV32[2 * j + (i >> 3)]))
            }
            pool::RA32 | pool::RB32 => {
                for _ in 0..16 {
                    out.extend_from_slice(&(rng.next() as u32).to_le_bytes());
                }
            }
            pool::H0 => {
                for k in 0..4 {
                    u16s(
                        &mut out,
                        32,
                        Box::new(move |i| EV16[(i + [0, 3, 5, 6][k]) & 7]),
                    );
                }
                u16s(&mut out, 32, Box::new(|i| 1u16 << (i & 15)));
            }
            pool::H1 => {
                for _ in 0..4 {
                    for _ in 0..32 {
                        out.extend_from_slice(&(rng.next() as u16).to_le_bytes());
                    }
                }
                u16s(&mut out, 32, Box::new(|i| 1u16 << ((i * 5 + 1) & 15)));
            }
            pool::SRS4M1 => put64(&mut out, &srs_half(32, 64, 16, &SRS_SHIFTS1, 0, &mut rng)),
            pool::SRS4M0 => put32(&mut out, &srs_half(64, 32, 8, &SRS_SHIFTS0, 0, &mut rng)),
            pool::SRS2M1 => {
                for h in 0..2 {
                    put64(
                        &mut out,
                        &srs_half(16, 64, 32, &SRS_SHIFTS1, 3 * h, &mut rng),
                    )
                }
            }
            pool::SRS2M0 => {
                for h in 0..2 {
                    put32(
                        &mut out,
                        &srs_half(32, 32, 16, &SRS_SHIFTS0, 8 * h, &mut rng),
                    )
                }
            }
            pool::UPS4 => u16s(
                &mut out,
                32,
                Box::new(|i| {
                    EV16[i & 7]
                        ^ if i >= 16 {
                            (i as u16).wrapping_mul(0x9e37)
                        } else {
                            0
                        }
                }),
            ),
            pool::UPS2M1 => u32s(
                &mut out,
                16,
                Box::new(|i| EV32[i & 7] ^ if i >= 8 { 0x0001_0001 * i as u32 } else { 0 }),
            ),
            pool::UPS2M0 => u16s(
                &mut out,
                32,
                Box::new(|i| {
                    EV16[i & 7]
                        ^ if i >= 16 {
                            (i as u16).wrapping_mul(0x4f1b)
                        } else {
                            0
                        }
                }),
            ),
            pool::UPS2M0U => u16s(
                &mut out,
                32,
                Box::new(|i| {
                    (EV16[i & 7]
                        ^ if i >= 16 {
                            (i as u16).wrapping_mul(0x4f1b)
                        } else {
                            0
                        })
                        & 0x7fff
                }),
            ),
            pool::UNP => {
                for r in [0usize, 1, 5, 11] {
                    for j in 0..64 {
                        out.push((((2 * j + r) & 15) | (((2 * j + 1 + r) & 15) << 4)) as u8);
                    }
                }
                for _ in 0..64 {
                    out.push(rng.next() as u8);
                }
            }
            pool::SHUF => out.extend((0..128u8).map(|b| b)),
            pool::I8 => {
                for i in 0..64 {
                    out.push(EDGE8[(i * 3 + i / 8) & 7] as u8);
                }
                for i in 0..64 {
                    out.push(EDGE8[(i + (i >> 3) * 5 + 1) & 7] as u8);
                }
                for _ in 0..128 {
                    out.push(rng.next() as u8);
                }
            }
            _ => unreachable!(),
        }
        assert_eq!(out.len() - start, pool::SIZE[item] * 64, "pool item {item}");
    }
    out.resize(IN_BYTES, 0);
    out
}

// ---------------------------------------------------------------------------------------------------------------
// Program generator
// ---------------------------------------------------------------------------------------------------------------
static BY_NAME: LazyLock<HashMap<&'static str, &'static isa::gen::Encoding>> =
    LazyLock::new(|| isa::gen::ENCODINGS.iter().map(|e| (e.name, e)).collect());
fn enc(name: &str, ops: &[(&str, u64)]) -> (Slot, u64) {
    let e = BY_NAME
        .get(name)
        .unwrap_or_else(|| panic!("no AIE2P encoding {name}"));
    (
        e.slot,
        encode_slot(e, ops, 0).unwrap_or_else(|_| panic!("operands of {name}: {ops:?}")),
    )
}
fn inst(i: Inst) -> (Slot, u64) {
    match i {
        Inst::Nop16 => (Slot::Nop16, 0),
        Inst::Alu(b) => (Slot::Alu, b),
        Inst::Lda(b) => (Slot::Lda, b),
        Inst::Ldb(b) => (Slot::Ldb, b),
        Inst::St(b) => (Slot::St, b),
        Inst::Mv(b) => (Slot::Mv, b),
        Inst::Vec(b) => (Slot::Vec, b),
        Inst::Lng(b) => (Slot::Lng, b),
    }
}
fn reg_value(encoding: &str, operand: usize, name: &str) -> u64 {
    descriptor(encoding).operands[operand]
        .registers
        .iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("{encoding}: no register {name}"))
        .value
}

// Dependency-tracked resources.
const RX: usize = 0; // x0..x11
const RQ: usize = 12; // bm quarters of dm0..dm4: RQ + 4 dm + q (a dm / cm half is its quarters)
const RR: usize = 32; // r0..r31
const RP: usize = 64; // p0..p7
const RS: usize = 72; // s0..s3
const CR_SAT: usize = 76;
const CR_RND: usize = 77;
const CR_SRS: usize = 78;
const CR_UPS: usize = 79;
const CR_UNP: usize = 80;
const MEM_IN: usize = 81;
const MEM_OUT: usize = 82;
const NRES: usize = 83;
const CR_NAMES: [&str; 5] = ["crSat", "crRnd", "crSRSMode", "crUPSMode", "crUnpackSize"];
const IN_FULL: u32 = 1;
const LOCK_GAP: usize = 6;

fn dmq(dm: usize) -> Vec<usize> {
    (0..4).map(|q| RQ + dm * 4 + q).collect()
}
fn cmq(cm: usize) -> Vec<usize> {
    (0..2)
        .map(|q| RQ + (cm / 2) * 4 + (cm % 2) * 2 + q)
        .collect()
}
fn wr_all(rs: &[usize], lat: usize) -> Vec<(usize, usize)> {
    rs.iter().map(|&r| (r, lat)).collect()
}

struct Gen {
    a: Assembler,
    t: usize,
    ready: [usize; NRES],
    last_read: [Option<usize>; NRES],
    last_lock: Option<usize>,
    /// Vector currently held by each x register (from `ld_x`), current p0 address, c-regs and s-regs values.
    xv: [Option<u16>; 12],
    p0: Option<u32>,
    crs: [Option<u32>; 5],
    sregs: [Option<u32>; 4],
}
impl Gen {
    fn new() -> Gen {
        Gen {
            a: Assembler::default(),
            t: 0,
            ready: [0; NRES],
            last_read: [None; NRES],
            last_lock: None,
            xv: [None; 12],
            p0: None,
            crs: [None; 5],
            sregs: [None; 4],
        }
    }
    fn issue_at(&mut self, i: (Slot, u64), rd: &[usize], wr: &[(usize, usize)], min_t: usize) {
        let mut need = self.t.max(min_t);
        for &r in rd {
            need = need.max(self.ready[r]);
        }
        for &(w, lat) in wr {
            if lat > 1 {
                if let Some(l) = self.last_read[w] {
                    need = need.max(l + WAR);
                }
            }
            need = need.max((self.ready[w] + 1).saturating_sub(lat));
        }
        while self.t < need {
            self.a.nop(1);
            self.t += 1;
        }
        self.a.emit(&[i]);
        let t = self.t;
        self.t += 1;
        for &r in rd {
            self.last_read[r] = Some(t);
        }
        for &(w, lat) in wr {
            self.ready[w] = t + lat;
            if w < 12 {
                self.xv[w] = None;
            }
        }
    }
    fn issue(&mut self, i: (Slot, u64), rd: &[usize], wr: &[(usize, usize)]) {
        self.issue_at(i, rd, wr, 0);
    }
    fn mov_r(&mut self, r: usize, v: u32) {
        self.issue(
            enc(
                "MOVXM",
                &[
                    ("dst", reg_value("MOVXM", 0, &format!("r{r}"))),
                    ("i", u64::from(v)),
                ],
            ),
            &[],
            &[(RR + r, LAT)],
        );
    }
    fn mov_p(&mut self, p: usize, v: u32) {
        self.issue(
            enc(
                "MOVXM",
                &[
                    ("dst", reg_value("MOVXM", 0, &format!("p{p}"))),
                    ("i", u64::from(v)),
                ],
            ),
            &[],
            &[(RP + p, LAT)],
        );
    }
    fn mov_s(&mut self, s: usize, v: u32) {
        self.issue(
            enc(
                "MOVXM",
                &[
                    ("dst", reg_value("MOVXM", 0, &format!("s{s}"))),
                    ("i", u64::from(v)),
                ],
            ),
            &[],
            &[(RS + s, LAT)],
        );
        self.sregs[s] = Some(v);
    }
    fn set_cr(&mut self, cr: usize, v: u32) {
        let n = cr - CR_SAT;
        if self.crs[n] == Some(v) {
            return;
        }
        self.issue(
            enc(
                "MOVX_mvx_cr_imm",
                &[
                    ("dst", reg_value("MOVX_mvx_cr_imm", 0, CR_NAMES[n])),
                    ("src", u64::from(v)),
                ],
            ),
            &[],
            &[(cr, LAT)],
        );
        self.crs[n] = Some(v);
    }
    /// Shift registers holding `shifts` (reusing resident values; others are loaded into registers not needed).
    fn assign_s(&mut self, shifts: &[i32]) -> Vec<usize> {
        let mut regs = Vec::new();
        for &sh in shifts {
            let reg = match self.sregs.iter().position(|v| *v == Some(sh as u32)) {
                Some(r) if !regs.contains(&r) => r,
                _ => {
                    let r = (0..4)
                        .find(|r| {
                            !regs.contains(r)
                                && !self.sregs[*r].is_some_and(|v| shifts.contains(&(v as i32)))
                        })
                        .or_else(|| (0..4).find(|r| !regs.contains(r)))
                        .unwrap();
                    self.mov_s(r, sh as u32);
                    r
                }
            };
            regs.push(reg);
        }
        regs
    }
    fn lock(&mut self, acquire: bool, id: u32) {
        let name = if acquire {
            "ACQ_mLockId_imm"
        } else {
            "REL_mLockId_imm"
        };
        let i = enc(
            name,
            &[
                ("id", u64::from(48 + id)),
                ("s1", if acquire { 0 } else { 1 }),
            ],
        );
        let min_t = self.last_lock.map_or(0, |l| l + LOCK_GAP);
        let (rd, wr): (Vec<usize>, Vec<(usize, usize)>) = if acquire {
            (vec![RR], vec![(MEM_IN, LAT)])
        } else {
            (vec![RR + 1, MEM_OUT], vec![])
        };
        self.issue_at(i, &rd, &wr, min_t);
        self.last_lock = Some(self.t - 1);
    }
    fn rebase(&mut self, addr: u32) {
        if self.p0 != Some(addr) {
            self.mov_p(0, addr);
            self.p0 = Some(addr);
        }
    }
    fn ld_x(&mut self, x: usize, vec: u16) {
        if self.xv[x] == Some(vec) {
            return;
        }
        self.rebase(CORE_BASE + IN_ADDR + u32::from(vec) * 64);
        self.issue(
            enc(
                "VLDA_dmx_lda_x_pstm_nrm_imm",
                &[("dst", x as u64), ("ptr", 0), ("imm", 1)],
            ),
            &[RP, MEM_IN],
            &[(RX + x, LAT), (RP, 1)],
        );
        self.p0 = Some(self.p0.unwrap() + 64);
        self.xv[x] = Some(vec);
    }
    fn ld_acc(&mut self, dm: usize, vec: u16) {
        self.rebase(CORE_BASE + IN_ADDR + u32::from(vec) * 64);
        for q in 0..4 {
            self.issue(
                enc(
                    "VLDA_dmx_lda_bm_pstm_nrm_imm",
                    &[("dst", (dm * 4 + q) as u64), ("ptr", 0), ("imm", 1)],
                ),
                &[RP, MEM_IN],
                &[(RQ + dm * 4 + q, LAT), (RP, 1)],
            );
            self.p0 = Some(self.p0.unwrap() + 64);
        }
    }
    fn st_x(&mut self, x: usize) {
        self.issue(
            enc(
                "VST_dmx_sts_x_pstm_nrm_imm",
                &[("src", x as u64), ("ptr", 1), ("imm", 1)],
            ),
            &[RX + x, RP + 1],
            &[(MEM_OUT, LAT), (RP + 1, 1)],
        );
    }
    fn st_acc(&mut self, dm: usize, q0: usize, nq: usize) {
        for q in q0..q0 + nq {
            self.issue(
                enc(
                    "VST_dmx_sts_bm_pstm_nrm_imm",
                    &[("src", (dm * 4 + q) as u64), ("ptr", 1), ("imm", 1)],
                ),
                &[RQ + dm * 4 + q, RP + 1],
                &[(MEM_OUT, LAT), (RP + 1, 1)],
            );
        }
    }
    fn st_scalar(&mut self, r: usize) {
        self.issue(
            inst(isa::st_post_imm(Reg::R(r as u8), Reg::P(2), 4)),
            &[RR + r, RP + 2],
            &[(MEM_OUT, LAT), (RP + 2, 1)],
        );
    }
    fn alu(&mut self, name: &str, d: usize, s0: usize, s1: Option<usize>) {
        let (ops, rd) = match s1 {
            Some(s1) => (
                vec![("d0", d as u64), ("s0", s0 as u64), ("s1", s1 as u64)],
                vec![RR + s0, RR + s1],
            ),
            None => (vec![("d0", d as u64), ("s0", s0 as u64)], vec![RR + s0]),
        };
        self.issue(enc(name, &ops), &rd, &[(RR + d, LAT)]);
    }

    // ---- families ----
    fn elem(&mut self, rows: &[Case]) {
        let r = rows[0];
        self.ld_x(0, r.s[0]);
        self.ld_x(1, r.s[1]);
        if r.op != Op::Vaddmac && r.s[2] != NONE {
            self.ld_acc(0, r.s[2]);
        }
        if r.s[3] != NONE {
            self.ld_acc(1, r.s[3]);
        }
        for chunk in rows.chunks(3) {
            for (i, c) in chunk.iter().enumerate() {
                self.mov_r(2 + i, c.p[0] as u32);
                if c.op == Op::Vaddmac {
                    self.ld_acc(2 + i, c.s[2]);
                }
            }
            for (i, c) in chunk.iter().enumerate() {
                let (dm, cr) = (2 + i, 2 + i);
                let xs = [RX, RX + 1, RR + cr];
                match c.op {
                    Op::Vmul => self.issue(
                        enc(
                            "VMUL_vmul_cm_core_X_X",
                            &[("dst", dm as u64), ("s1", 0), ("s2", 1), ("acc", cr as u64)],
                        ),
                        &xs,
                        &wr_all(&dmq(dm), LAT),
                    ),
                    Op::Vmac => self.issue(
                        enc(
                            "VMAC_vmul_cm_core_X_X",
                            &[
                                ("dst", dm as u64),
                                ("acc1", 0),
                                ("s1", 0),
                                ("s2", 1),
                                ("acc", cr as u64),
                            ],
                        ),
                        &[&xs[..], &dmq(0)].concat(),
                        &wr_all(&dmq(dm), LAT),
                    ),
                    _ => self.issue(
                        enc(
                            "VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_X",
                            &[
                                ("acc1", dm as u64),
                                ("acc2", 1),
                                ("s1", 0),
                                ("s2", 1),
                                ("acc", cr as u64),
                            ],
                        ),
                        &[&xs[..], &dmq(dm), &dmq(1)].concat(),
                        &wr_all(&dmq(dm), LAT),
                    ),
                }
            }
            for i in 0..chunk.len() {
                self.st_acc(2 + i, 0, 4);
            }
        }
    }
    fn horner(&mut self, rows: &[Case]) {
        self.ld_x(1, rows[0].s[1]);
        for (i, c) in rows.iter().enumerate() {
            self.ld_x(2 + i, c.s[0]);
        }
        self.mov_r(2, rows[0].p[0] as u32);
        self.mov_r(3, rows[1].p[0] as u32);
        for (i, c) in rows.iter().enumerate() {
            if c.op == Op::HornerMul {
                self.issue(
                    enc(
                        "VMUL_vmul_cm_core_X_X",
                        &[("dst", 0), ("s1", (2 + i) as u64), ("s2", 1), ("acc", 2)],
                    ),
                    &[RX + 2 + i, RX + 1, RR + 2],
                    &wr_all(&dmq(0), LAT),
                );
            } else {
                self.issue(
                    enc(
                        "VMAC_vmul_cm_core_X_X",
                        &[
                            ("dst", 0),
                            ("acc1", 0),
                            ("s1", (2 + i) as u64),
                            ("s2", 1),
                            ("acc", 3),
                        ],
                    ),
                    &[&[RX + 2 + i, RX + 1, RR + 3][..], &dmq(0)].concat(),
                    &wr_all(&dmq(0), LAT),
                );
            }
            self.st_acc(0, 0, 4);
        }
    }
    fn neg(&mut self, rows: &[Case]) {
        self.ld_acc(0, rows[0].s[2]);
        for (i, c) in rows.iter().enumerate() {
            self.mov_r(2 + i, c.p[0] as u32);
        }
        for i in 0..rows.len() {
            self.issue(
                enc(
                    "VNEG",
                    &[
                        ("dst", (2 + i) as u64),
                        ("acc1", 0),
                        ("acc", (2 + i) as u64),
                    ],
                ),
                &[&dmq(0)[..], &[RR + 2 + i]].concat(),
                &wr_all(&dmq(2 + i), LAT),
            );
        }
        for i in 0..rows.len() {
            self.st_acc(2 + i, 0, 4);
        }
    }
    fn acc_round_trip(&mut self, rows: &[Case]) {
        self.ld_acc(4, rows[0].s[2]);
        self.st_acc(4, 0, 4);
    }
    fn srs(&mut self, rows: &[Case]) {
        self.ld_acc(0, rows[0].s[2]);
        let key = |c: &Case| (c.p[0], c.p[2], c.p[3] & 0xff);
        let mut i = 0;
        while i < rows.len() {
            let mut j = i;
            let mut shifts: Vec<i32> = Vec::new();
            while j < rows.len() && j - i < 8 && key(&rows[j]) == key(&rows[i]) {
                let sh = rows[j].p[1];
                if !shifts.contains(&sh) {
                    if shifts.len() == 4 {
                        break;
                    }
                    shifts.push(sh);
                }
                j += 1;
            }
            let (mode, rnd, sat) = key(&rows[i]);
            self.set_cr(CR_SRS, mode as u32);
            self.set_cr(CR_RND, rnd as u32);
            self.set_cr(CR_SAT, sat as u32);
            let regs = self.assign_s(&shifts);
            for (n, c) in rows[i..j].iter().enumerate() {
                let su = regs[shifts.iter().position(|&s| s == c.p[1]).unwrap()];
                let dst = 4 + n;
                let (name, src, srcs) = if c.op == Op::Srs4 {
                    ("VSRS_4x_mv_x_srs_dm_srsSign1", 0, dmq(0))
                } else {
                    let cm = (c.p[3] >> 8) as usize;
                    ("VSRS_2x_mv_x_srs_cm_srsSign1", cm, cmq(cm))
                };
                self.issue(
                    enc(
                        name,
                        &[("dst", dst as u64), ("src", src as u64), ("su", su as u64)],
                    ),
                    &[&srcs[..], &[RS + su, CR_SAT, CR_RND, CR_SRS]].concat(),
                    &[(RX + dst, LAT)],
                );
            }
            for n in 0..j - i {
                self.st_x(4 + n);
            }
            i = j;
        }
    }
    fn ups(&mut self, rows: &[Case]) {
        let mut i = 0;
        while i < rows.len() {
            let mut j = i;
            let mut shifts: Vec<i32> = Vec::new();
            while j < rows.len()
                && j - i < 3
                && rows[j].p[0] == rows[i].p[0]
                && rows[j].s[0] == rows[i].s[0]
            {
                let sh = rows[j].p[2];
                if !shifts.contains(&sh) {
                    if shifts.len() == 4 {
                        break;
                    }
                    shifts.push(sh);
                }
                j += 1;
            }
            self.ld_x(0, rows[i].s[0]);
            self.set_cr(CR_UPS, rows[i].p[0] as u32);
            let regs = self.assign_s(&shifts);
            for (n, c) in rows[i..j].iter().enumerate() {
                let su = regs[shifts.iter().position(|&s| s == c.p[2]).unwrap()];
                let sign = c.p[1] as usize;
                let (name, dst, dsts) = if c.op == Op::Ups4 {
                    (
                        ["VUPS_4x_mv_ups_x2d_upsSign0", "VUPS_4x_mv_ups_x2d_upsSign1"][sign],
                        2 + n,
                        dmq(2 + n),
                    )
                } else {
                    let cm = 2 * (2 + n) + c.p[3] as usize;
                    (
                        ["VUPS_2x_mv_ups_x2c_upsSign0", "VUPS_2x_mv_ups_x2c_upsSign1"][sign],
                        cm,
                        cmq(cm),
                    )
                };
                self.issue(
                    enc(name, &[("dst", dst as u64), ("src", 0), ("su", su as u64)]),
                    &[RX, RS + su, CR_UPS],
                    &wr_all(&dsts, LAT),
                );
            }
            for (n, c) in rows[i..j].iter().enumerate() {
                if c.op == Op::Ups4 {
                    self.st_acc(2 + n, 0, 4);
                } else {
                    self.st_acc(2 + n, 2 * c.p[3] as usize, 2);
                }
            }
            i = j;
        }
    }
    fn unpack(&mut self, rows: &[Case]) {
        self.set_cr(CR_UNP, 0);
        for chunk in rows.chunks(3) {
            for (i, c) in chunk.iter().enumerate() {
                let y = 1 + i;
                let sign = c.p[0] as usize;
                if c.op == Op::VldbUnpack {
                    self.rebase(CORE_BASE + IN_ADDR + u32::from(c.s[0]) * 64);
                    let name = [
                        "VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_imm_unpackSign0",
                        "VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_imm_unpackSign1",
                    ][sign];
                    self.issue(
                        enc(name, &[("dst", y as u64), ("ptr", 0), ("imm", 1)]),
                        &[RP, MEM_IN, CR_UNP],
                        &[(RX + 2 * y, LAT), (RX + 2 * y + 1, LAT), (RP, 1)],
                    );
                    self.p0 = Some(self.p0.unwrap() + 64);
                } else {
                    self.ld_x(8 + i, c.s[0]);
                    let name = [
                        "VUNPACK_mv_unpack_x_unpackSign0",
                        "VUNPACK_mv_unpack_x_unpackSign1",
                    ][sign];
                    self.issue(
                        enc(name, &[("dst", y as u64), ("src", (8 + i) as u64)]),
                        &[RX + 8 + i, CR_UNP],
                        &[(RX + 2 * y, LAT), (RX + 2 * y + 1, LAT)],
                    );
                }
            }
            for i in 0..chunk.len() {
                self.st_x(2 * (1 + i));
                self.st_x(2 * (1 + i) + 1);
            }
        }
    }
    fn vldb_x(&mut self, rows: &[Case]) {
        for (i, c) in rows.iter().enumerate() {
            self.rebase(CORE_BASE + IN_ADDR + u32::from(c.s[0]) * 64);
            self.issue(
                enc(
                    "VLDB_dmx_ldb_x_pstm_nrm_imm",
                    &[("dst", (2 + i) as u64), ("ptr", 0), ("imm", 1)],
                ),
                &[RP, MEM_IN],
                &[(RX + 2 + i, LAT), (RP, 1)],
            );
            self.p0 = Some(self.p0.unwrap() + 64);
        }
        for i in 0..rows.len() {
            self.st_x(2 + i);
        }
    }
    fn shuffle(&mut self, rows: &[Case]) {
        self.ld_x(0, rows[0].s[0]);
        self.ld_x(1, rows[0].s[1]);
        for chunk in rows.chunks(8) {
            for (i, c) in chunk.iter().enumerate() {
                self.mov_r(2 + i, c.p[0] as u32);
            }
            for i in 0..chunk.len() {
                self.issue(
                    enc(
                        "VSHUFFLE_vec_shuffle_x",
                        &[
                            ("dst", (2 + i) as u64),
                            ("s1", 0),
                            ("s2", 1),
                            ("mod", (2 + i) as u64),
                        ],
                    ),
                    &[RX, RX + 1, RR + 2 + i],
                    &[(RX + 2 + i, LAT)],
                );
            }
            for i in 0..chunk.len() {
                self.st_x(2 + i);
            }
        }
    }
    fn compare(&mut self, rows: &[Case]) {
        self.ld_x(0, rows[0].s[0]);
        self.ld_x(1, rows[0].s[1]);
        let wide = matches!(rows[0].op, Op::Vlt32 | Op::Vge32 | Op::Veqz32);
        if wide {
            self.mov_r(8, 0xffff);
        }
        for chunk in rows.chunks(5) {
            for (i, c) in chunk.iter().enumerate() {
                let cmp = 17 + i;
                if matches!(c.op, Op::Veqz16 | Op::Veqz32) {
                    self.issue(
                        enc(c.instruction(), &[("cmp", (cmp - 16) as u64), ("s2", 1)]),
                        &[RX + 1],
                        &[(RR + cmp, LAT)],
                    );
                } else {
                    self.issue(
                        enc(
                            c.instruction(),
                            &[("cmp", (cmp - 16) as u64), ("s1", 0), ("s2", 1)],
                        ),
                        &[RX, RX + 1],
                        &[(RR + cmp, LAT)],
                    );
                }
            }
            for i in 0..chunk.len() {
                if wide {
                    self.alu("AND", 10 + i, 17 + i, Some(8));
                }
                self.st_scalar(if wide { 10 + i } else { 17 + i });
            }
        }
    }
    fn select(&mut self, rows: &[Case]) {
        self.ld_x(0, rows[0].s[0]);
        self.ld_x(1, rows[0].s[1]);
        let wide = rows[0].op == Op::Vsel32;
        for (i, c) in rows.iter().enumerate() {
            let sel = 16 + i;
            if c.p[0] == 0 {
                self.mov_r(sel, c.p[1] as u32);
            } else {
                let name = if wide {
                    "VLT_32_vaddSign1"
                } else {
                    "VLT_16_vaddSign1"
                };
                self.issue(
                    enc(name, &[("cmp", (sel - 16) as u64), ("s1", 0), ("s2", 1)]),
                    &[RX, RX + 1],
                    &[(RR + sel, LAT)],
                );
            }
        }
        for (i, c) in rows.iter().enumerate() {
            self.issue(
                enc(
                    c.instruction(),
                    &[
                        ("d", (2 + i) as u64),
                        ("s1", 0),
                        ("s2", 1),
                        ("sel", i as u64),
                    ],
                ),
                &[RX, RX + 1, RR + 16 + i],
                &[(RX + 2 + i, LAT)],
            );
        }
        for i in 0..rows.len() {
            self.st_x(2 + i);
        }
    }
    fn lane_op(&mut self, rows: &[Case]) {
        self.ld_x(0, rows[0].s[0]);
        self.ld_x(1, rows[0].s[1]);
        for (i, c) in rows.iter().enumerate() {
            self.issue(
                enc(
                    c.instruction(),
                    &[("d", (2 + i) as u64), ("s1", 0), ("s2", 1)],
                ),
                &[RX, RX + 1],
                &[(RX + 2 + i, LAT)],
            );
        }
        for i in 0..rows.len() {
            self.st_x(2 + i);
        }
    }
    fn max_min(&mut self, rows: &[Case]) {
        self.ld_x(0, rows[0].s[0]);
        self.ld_x(1, rows[0].s[1]);
        let wide = matches!(rows[0].op, Op::Vmax32 | Op::Vmin32);
        if wide {
            self.mov_r(8, 0xffff);
        }
        let pairs = rows.len() / 2;
        for p in 0..pairs {
            self.issue(
                enc(
                    rows[2 * p].instruction(),
                    &[("d", (2 + p) as u64), ("s1", 0), ("s2", 1)],
                ),
                &[RX, RX + 1],
                &[(RX + 2 + p, LAT), (RR + 16, LAT)],
            );
            if wide {
                self.alu("AND", 10 + p, 16, Some(8));
            } else {
                self.alu("OR", 10 + p, 16, Some(16));
            }
        }
        for p in 0..pairs {
            self.st_x(2 + p);
            self.st_scalar(10 + p);
        }
    }
    fn bcst(&mut self, rows: &[Case]) {
        for (i, c) in rows.iter().enumerate() {
            self.mov_r(2 + i, c.p[0] as u32);
            self.issue(
                enc(
                    c.instruction(),
                    &[("dst", (2 + i) as u64), ("src", (2 + i) as u64)],
                ),
                &[RR + 2 + i],
                &[(RX + 2 + i, LAT)],
            );
        }
        for i in 0..rows.len() {
            self.st_x(2 + i);
        }
    }
    fn i8_control(&mut self, rows: &[Case]) {
        for k in 0..4 {
            self.ld_x(k, rows[0].s[k]);
        }
        self.mov_r(2, 0x308);
        let steps: [(Op, usize, usize, usize, usize); 3] = [
            (Op::I8Mul, 2, 0, 0, 1),
            (Op::I8Mac, 3, 2, 2, 3),
            (Op::I8Mac, 4, 3, 0, 3),
        ];
        for (n, (op, dm, acc1, a, b)) in steps.into_iter().enumerate() {
            let _ = n;
            let xs = [RX + a, RX + b, RR + 2];
            if op == Op::I8Mul {
                self.issue(
                    enc(
                        "VMUL_vmul_cm_core_X_X",
                        &[
                            ("dst", dm as u64),
                            ("s1", a as u64),
                            ("s2", b as u64),
                            ("acc", 2),
                        ],
                    ),
                    &xs,
                    &wr_all(&dmq(dm), LAT),
                );
            } else {
                self.issue(
                    enc(
                        "VMAC_vmul_cm_core_X_X",
                        &[
                            ("dst", dm as u64),
                            ("acc1", acc1 as u64),
                            ("s1", a as u64),
                            ("s2", b as u64),
                            ("acc", 2),
                        ],
                    ),
                    &[&xs[..], &dmq(acc1)].concat(),
                    &wr_all(&dmq(dm), LAT),
                );
            }
        }
        for dm in 2..5 {
            self.st_acc(dm, 0, 4);
        }
    }
    fn scalar(&mut self, rows: &[Case]) {
        for chunk in rows.chunks(4) {
            for (i, c) in chunk.iter().enumerate() {
                let (a, b, d) = (2 + 2 * i, 3 + 2 * i, 10 + i);
                self.mov_r(a, c.p[0] as u32);
                let unary = matches!(c.op, Op::SEqz | Op::SNez);
                if !unary {
                    self.mov_r(b, c.p[1] as u32);
                }
                self.alu(c.instruction(), d, a, if unary { None } else { Some(b) });
            }
            for i in 0..chunk.len() {
                self.st_scalar(10 + i);
            }
        }
    }
}

/// The probe core program (unfinished `Program`; [`program_bytes`] runs `finish()` and the `isa::rules`).
pub fn program() -> Program {
    let mut g = Gen::new();
    g.mov_r(0, u32::MAX);
    g.mov_r(1, 1);
    g.lock(true, C_EMPTY);
    g.lock(true, IN_FULL);
    g.mov_p(2, CORE_BASE + OUT_ADDR);
    g.mov_p(1, CORE_BASE + OUT_ADDR + SCALAR_BYTES as u32);
    let mut i = 0;
    while i < CASES.len() {
        let mut j = i;
        while j < CASES.len() && CASES[j].group == CASES[i].group {
            j += 1;
        }
        let rows = &CASES[i..j];
        match rows[0].op {
            Op::Vmul | Op::Vmac | Op::Vaddmac => g.elem(rows),
            Op::HornerMul | Op::HornerMac => g.horner(rows),
            Op::Vneg => g.neg(rows),
            Op::AccRt => g.acc_round_trip(rows),
            Op::Srs4 | Op::Srs2 => g.srs(rows),
            Op::Ups4 | Op::Ups2 => g.ups(rows),
            Op::VldbUnpack | Op::Vunpack => g.unpack(rows),
            Op::VldbX => g.vldb_x(rows),
            Op::Shuffle => g.shuffle(rows),
            Op::Vlt16 | Op::Vge16 | Op::Vlt32 | Op::Vge32 | Op::Veqz16 | Op::Veqz32 => {
                g.compare(rows)
            }
            Op::Vsel16 | Op::Vsel32 => g.select(rows),
            Op::Vadd16 | Op::Vsub16 | Op::Vadd32 | Op::Vsub32 | Op::Vband | Op::Vbor => {
                g.lane_op(rows)
            }
            Op::Vmax16 | Op::Vmin16 | Op::Vmax32 | Op::Vmin32 => g.max_min(rows),
            Op::Vbcst16 | Op::Vbcst32 => g.bcst(rows),
            Op::I8Mul | Op::I8Mac => g.i8_control(rows),
            Op::SAnd | Op::SOr | Op::SLshl | Op::SAdd | Op::SEqz | Op::SNez => g.scalar(rows),
        }
        i = j;
    }
    g.lock(false, C_FULL);
    g.a.nop(7);
    g.a.emit(&[enc("DONE", &[])]);
    g.a.finish()
}
/// The finished program image (padded to 16 B, `isa::rules` checked).
pub fn program_bytes() -> Vec<u8> {
    program().finish()
}

// ---------------------------------------------------------------------------------------------------------------
// Design (single core, gemm_i8-style In/Out buffers)
// ---------------------------------------------------------------------------------------------------------------
/// Core DMA descriptors: BD0 = S2MM0 input (lock 0 acquire, lock 1 release), BD4 = MM2S0 output (lock 9 acquire,
/// lock 8 release), both single-shot.
pub fn tile_bds() -> [[u32; 6]; 5] {
    let mut bds = [[0; 6]; 5];
    bds[0] = dma::tile_bd(
        IN_ADDR,
        (IN_BYTES / 4) as u32,
        BdLocks {
            acq: Some((0, -1)),
            rel: Some((1, 1)),
        },
        None,
    );
    bds[4] = dma::tile_bd(
        OUT_ADDR,
        (OUT_BYTES / 4) as u32,
        BdLocks {
            acq: Some((C_FULL, -1)),
            rel: Some((C_EMPTY, 1)),
        },
        None,
    );
    bds
}

/// The deployable probe image: PDI (fresh-context CDO), TXN and the two DDR-patched arguments.
pub struct ProbeDesign {
    pub pdi: Vec<u8>,
    pub insts: Vec<u8>,
    /// `[input (In), output (Out)]`.
    pub args: Vec<ArgSpec>,
    pub col: u32,
    /// Core program bytes (16-byte padded).
    pub program_bytes: usize,
}

/// Single-core design at `(col, 2)`; shim BDs keep the vendor word 5 (`ShimAxi::default()`: AxCACHE 2, AxQoS 0).
pub fn design(col: u32) -> ProbeDesign {
    use dma::{
        Direction::{Mm2s, S2mm},
        Location, Task,
    };
    assert!(col < 8);
    let program = program_bytes();
    assert!(
        program.len() <= 16 * 1024,
        "probe program {} B exceeds the 16 KiB program memory",
        program.len()
    );
    let core = Location::new(col, 2);
    let mem = Location::new(col, 1);
    let shim = Location::new(col, 0);
    let mut cdo = Cdo::new();
    cdo.mask_write(core.address(regs::core::CORE_CONTROL), 1, 0);
    for off in [regs::core::DMA_S2MM_0_CTRL, regs::core::DMA_MM2S_0_CTRL] {
        cdo.mask_write(core.address(off), 2, 2);
    }
    let words: Vec<u32> = program
        .chunks(4)
        .map(|chunk| {
            let mut word = [0; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(word)
        })
        .collect();
    cdo.dma_write(core.address(regs::core::PROGRAM_MEMORY), &words);
    for off in [regs::core::DMA_S2MM_0_CTRL, regs::core::DMA_MM2S_0_CTRL] {
        cdo.mask_write(core.address(off), 2, 0);
    }
    cdo.mask_write(core.address(regs::core::CORE_CONTROL), 2, 2)
        .mask_write(core.address(regs::core::CORE_CONTROL), 2, 0)
        .write(core.address(regs::core::CORE_PC), 0);
    for (id, value) in super::gemm_i8::initial_locks().into_iter().enumerate() {
        let (addr, value) = dma::lock_write(core, id as u32, value as u32);
        cdo.write(addr, value);
    }
    for (id, bd) in tile_bds().iter().enumerate() {
        cdo.dma_write(dma::Bd::address(core, id as u32), bd);
    }
    let mm = ShimDma {
        tile: shim,
        direction: Mm2s,
        channel: 0,
    };
    let sm = ShimDma {
        tile: shim,
        direction: S2mm,
        channel: 0,
    };
    for circuit in [
        Circuit {
            tile: shim,
            slave: mm.port(),
            master: Port::North(0),
        },
        Circuit {
            tile: mem,
            slave: Port::South(0),
            master: Port::North(0),
        },
        Circuit {
            tile: core,
            slave: Port::South(0),
            master: Port::Dma(0),
        },
        Circuit {
            tile: core,
            slave: Port::Dma(0),
            master: Port::South(0),
        },
        Circuit {
            tile: mem,
            slave: Port::North(0),
            master: Port::South(0),
        },
        Circuit {
            tile: shim,
            slave: Port::North(0),
            master: sm.port(),
        },
    ] {
        circuit.emit_cdo(&mut cdo);
    }
    mm.emit_cdo(&mut cdo);
    sm.emit_cdo(&mut cdo);
    for (direction, bd) in [(Mm2s, 4), (S2mm, 0)] {
        let task = Task {
            direction,
            channel: 0,
            bd,
            repeat: 1,
            issue_token: false,
        };
        task.emit_cdo(core, &mut cdo);
        task.enable_cdo(core, &mut cdo);
    }
    for (addr, value) in regs::shim_token_route(col) {
        cdo.write(addr, value);
    }
    cdo.mask_write(core.address(regs::core::CORE_CONTROL), 1, 1);
    let mut txn = Txn::aie2p_8col();
    for (id, words, arg) in [(0, IN_BYTES / 4, 0), (1, OUT_BYTES / 4, 1)] {
        dma::Bd::new(0, words as u32).emit_txn(shim, id, &mut txn);
        txn.ddr_patch(dma::Bd::address(shim, id) + 4, arg, 0);
    }
    txn.mask_write(shim.address(regs::shim::DMA_S2MM_0_CTRL), 0xf00, 0x1f00);
    for (direction, bd, token) in [(S2mm, 1, true), (Mm2s, 0, false)] {
        Task {
            direction,
            channel: 0,
            bd,
            repeat: 1,
            issue_token: token,
        }
        .emit_txn(shim, &mut txn);
    }
    txn.sync(col, 0, 0, 0, 1, 1);
    ProbeDesign {
        pdi: crate::pdi::build(&cdo.to_words()),
        insts: txn.to_bytes(),
        args: vec![
            ArgSpec {
                bytes: IN_BYTES,
                kind: ArgKind::In,
            },
            ArgSpec {
                bytes: OUT_BYTES,
                kind: ArgKind::Out,
            },
        ],
        col,
        program_bytes: program.len(),
    }
}

/// One differing case: the first differing byte, and how many bytes of the case differ.
#[derive(Clone, Copy, Debug)]
pub struct Mismatch {
    pub case: &'static Case,
    /// Absolute offset in the output buffer of the first differing byte.
    pub byte: usize,
    pub expected: u8,
    pub got: u8,
    pub bad_bytes: usize,
}
impl Mismatch {
    /// Lane index (of [`Case::lane_bytes`] width) of the first differing byte within its case.
    pub fn lane(&self) -> usize {
        (self.byte - self.case.out_off as usize) / self.case.lane_bytes()
    }
}

/// Host side of the probe.
pub struct SemanticsProbe;
impl SemanticsProbe {
    pub fn design(col: u32) -> ProbeDesign {
        design(col)
    }
    pub fn input() -> Vec<u8> {
        input()
    }
    pub fn cases() -> &'static [Case] {
        &CASES
    }
    /// Compare two output buffers case by case (bytes outside every case range are ignored); one entry per
    /// differing case in table order.
    pub fn compare(expected: &[u8], got: &[u8]) -> Vec<Mismatch> {
        assert!(
            expected.len() >= OUT_BYTES && got.len() >= OUT_BYTES,
            "probe output shorter than {OUT_BYTES} B"
        );
        CASES
            .iter()
            .filter_map(|c| {
                let r = c.out_off as usize..c.out_off as usize + c.out_len as usize;
                let bad = expected[r.clone()]
                    .iter()
                    .zip(&got[r.clone()])
                    .filter(|(e, g)| e != g)
                    .count();
                let first = (r.start..r.end).find(|&i| expected[i] != got[i])?;
                Some(Mismatch {
                    case: c,
                    byte: first,
                    expected: expected[first],
                    got: got[first],
                    bad_bytes: bad,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_table_is_dense_and_partitions_the_output() {
        let mut spans: Vec<(u32, u32)> = Vec::new();
        for (i, c) in CASES.iter().enumerate() {
            assert_eq!(c.id as usize, i);
            assert!(
                i == 0 || c.group >= CASES[i - 1].group,
                "groups must be contiguous in program order"
            );
            assert!(
                isa::gen::ENCODINGS
                    .iter()
                    .any(|e| e.name == c.instruction()),
                "{}",
                c.instruction()
            );
            for &v in &c.s {
                assert!(v == NONE || (v as usize) < pool::VECS);
            }
            assert!(
                c.out_len > 0
                    && c.out_off % 4 == 0
                    && (c.out_off + c.out_len as u32) as usize <= OUT_BYTES
            );
            assert_eq!(
                c.out_off % 64 == 0 || c.scalar(),
                true,
                "vector results are 64-byte aligned: case {i}"
            );
            if !c.scalar() {
                assert!(c.out_off as usize >= SCALAR_BYTES);
            } else {
                assert!((c.out_off as usize) < SCALAR_BYTES);
            }
            spans.push((c.out_off, c.out_off + c.out_len as u32));
        }
        spans.sort();
        for w in spans.windows(2) {
            assert!(w[0].1 <= w[1].0, "output ranges overlap: {w:?}");
        }
        assert_eq!(OUT_BYTES % 256, 0);
        assert!(spans.last().unwrap().1 as usize <= OUT_BYTES);
    }

    #[test]
    fn program_assembles_and_passes_the_isa_rules() {
        let bytes = program_bytes();
        assert!(
            bytes.len() % 16 == 0 && bytes.len() <= 16 * 1024,
            "program {} B",
            bytes.len()
        );
        crate::isa::rules::check(&bytes).unwrap();
        println!(
            "probe: {} cases, {} B program, in {} B, out {} B",
            CASE_COUNT,
            bytes.len(),
            IN_BYTES,
            OUT_BYTES
        );
    }

    #[test]
    fn input_is_deterministic_and_pool_sized() {
        let a = input();
        assert_eq!(a.len(), IN_BYTES);
        assert_eq!(a, input());
        assert!(
            IN_BYTES + OUT_BYTES <= 64 * 1024,
            "in + out must fit the 64 KiB data memory"
        );
    }

    #[test]
    fn design_builds_and_compare_names_the_case() {
        let d = design(0);
        assert!(!d.pdi.is_empty() && !d.insts.is_empty());
        assert_eq!(
            d.args.iter().map(|a| a.bytes).collect::<Vec<_>>(),
            vec![IN_BYTES, OUT_BYTES]
        );
        let want = vec![0u8; OUT_BYTES];
        let mut got = want.clone();
        let c = CASES[7];
        got[c.out_off as usize + 3] = 0x5a;
        let m = SemanticsProbe::compare(&want, &got);
        assert_eq!(m.len(), 1);
        assert_eq!(
            (
                m[0].case.id,
                m[0].byte,
                m[0].expected,
                m[0].got,
                m[0].bad_bytes
            ),
            (7, c.out_off as usize + 3, 0, 0x5a, 1)
        );
    }
}
