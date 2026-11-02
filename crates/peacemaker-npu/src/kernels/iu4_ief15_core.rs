// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// AIE2P instruction fields and scheduling are derived from Xilinx/llvm-aie
// AIE2PGenInstrInfo.td and AIE2PGenSchedule.td (Apache-2.0 WITH LLVM-exception).
//! IEF15 pair core (TM64 x TN32): per K128 epoch an int4-streamed int8 GEMM into `C`, then the exact integer
//! fold `I += S*D*C` in 64-bit accumulator lanes, and after the last epoch one RNE to binary32 per output.
//! Data movement and lock protocol are V8's pair protocol (`gemm_core`), only the core memory map, the chunk
//! lengths and the program differ. Every statement about hardware behaviour is a simulator model until the
//! declared silicon window.
use super::gemm_core::{
    emit_cycle, pair_acquire, pair_release, Assembler, Control, Cycle, PairLayout, PairRole, A_BD,
    A_EMPTY, A_FULL, B_BD, B_EMPTY, B_FULL, CORE_BASE, C_BD, C_EMPTY, C_FULL,
};
use crate::{
    dma::{self, BdLocks},
    isa::{
        self,
        bundle::encode_slot,
        gen::{Encoding, Slot},
        sched, Program,
    },
};
use std::{collections::HashMap, sync::LazyLock};

#[path = "g80_bf16.rs"]
pub(super) mod g80_bf16;

/// Token rows of one core tile.
pub const TM: usize = 64;
/// Feature columns of one core tile.
pub const TN: usize = 32;
/// K of one epoch (one IEF15 scale step).
pub const EPOCH_K: usize = 128;
/// Largest epoch count (|I| < 2^51, fold-contract §3).
pub const MAX_EPOCHS: usize = 136;
/// One A half chunk: 4 `mbl` x 16 `kb` blocks of 8x8 int4 (2048 B), then D[32] u16, then b[32] i16.
pub const A_HALF_BYTES: usize = 2176;
/// One B chunk: 16 `kb` x 4 `nb` blocks of 8x8 int4 (2048 B), then S[32] u16, then a[32] i16.
pub const B_BYTES: usize = 2176;
/// Offset of the D (A chunk) / S (B chunk) section inside a chunk.
pub const SCALE_OFF: usize = 2048;
/// Offset of the b (A chunk) / a (B chunk) section inside a chunk.
pub const EXP_OFF: usize = 2112;
/// f32 output tile (64 x 32 x 4 B), the V8 int8 C tile size.
pub const COUT_BYTES: usize = TM * TN * 4;

/// Own-view data-memory map (core address = `gemm_core::CORE_BASE + offset`).
pub const A_ADDR: [u32; 2] = [0x0000, 0x0880];
pub const B_ADDR: [u32; 2] = [0x1100, 0x1980];
pub const C16_ADDR: u32 = 0x2200;
/// D patterns of one epoch (`[mb][half]`, 64 B each); after the last epoch the b patterns in the same layout.
pub const DPAT_ADDR: u32 = 0x3200;
/// b(E), b(O), a copies of the current epoch's chunks (64 B each).
pub const AB_COPY_ADDR: u32 = 0x3600;
/// Constant vectors of the final pack (`Konst`, 64 B each).
pub const CONST_ADDR: u32 = 0x3700;
pub const I_ADDR: u32 = 0x4000;
pub const COUT_ADDR: u32 = 0x8000;
/// Second epoch slot of a fold pair: C16 and Dpat of the odd epoch; Spat of both epochs (`[slot][nb]`).
pub const C16_B_ADDR: u32 = 0xa000;
pub const DPAT_B_ADDR: u32 = 0xb000;
pub const SPAT_ADDR: u32 = 0xb400;
const _: () =
    assert!(COUT_ADDR as usize + COUT_BYTES <= C16_B_ADDR as usize && SPAT_ADDR + 512 <= 0x10000);
const _: () = assert!(A_ADDR[1] as usize == A_HALF_BYTES && B_ADDR[0] as usize == 2 * A_HALF_BYTES);
const _: () = assert!(
    B_ADDR[1] as usize == B_ADDR[0] as usize + B_BYTES
        && C16_ADDR as usize == B_ADDR[1] as usize + B_BYTES
);
const _: () = assert!(
    A_HALF_BYTES == SCALE_OFF + 128 && B_BYTES == SCALE_OFF + 128 && EXP_OFF == SCALE_OFF + 64
);
const _: () = assert!(I_ADDR as usize + TM * TN * 8 == COUT_ADDR as usize && COUT_BYTES == 8192);

/// Core tile DMA descriptors: A ring S2MM0 BD0/1, B ring S2MM1 BD2/3, Cout MM2S0 BD4 (V8 pair numbering).
pub fn tile_bds() -> [[u32; 6]; 5] {
    let mut bds = [[0; 6]; 5];
    for slot in 0..2 {
        bds[A_BD[slot] as usize] = dma::tile_bd(
            A_ADDR[slot],
            (A_HALF_BYTES / 4) as u32,
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
        COUT_ADDR,
        (COUT_BYTES / 4) as u32,
        BdLocks {
            acq: Some((C_FULL, -1)),
            rel: Some((C_EMPTY, 1)),
        },
        Some(C_BD),
    );
    bds
}

/// Largest |a_r + b_t| the core keeps; beyond it every result is already 0 / -0 / +-inf (|I| < 2^51).
pub const X_CLAMP: i16 = 300;
/// One lane of the final pack, operation for operation as the core emits it (16-bit lanes wrap, `srs16` = floor SRS
/// with wrap to 16 bits, `srs32_rne` = conv_even SRS saturated to 32 bits). `x` is `a_r + b_t` (|a|, |b| <= 16383).
/// Bit-exact with `round_scaled(i, x)` for |i| < 2^51 (tests below).
pub fn pack_lane_model(i: i64, x: i32) -> u32 {
    let srs16 = |v: i64, s: u32| (v >> s) as u16;
    let x = (x as i16).clamp(-X_CLAMP, X_CLAMP);
    let l = [srs16(i, 0), srs16(i, 16), srs16(i, 32), srs16(i, 48)];
    let neg = (l[3] as i16) < 0;
    let n = i.wrapping_neg();
    let nl = [srs16(n, 0), srs16(n, 16), srs16(n, 32), srs16(n, 48)];
    let m = if neg { nl } else { l };
    // Highest nonzero limb and its bit offset.
    let (z3, z2, z1) = (m[3] == 0, m[2] == 0, m[1] == 0);
    let (mut v, mut base) = (m[3], 48i16);
    if z3 {
        v = m[2];
        base = 32;
    }
    if z3 && z2 {
        v = m[1];
        base = 16;
    }
    if z3 && z2 && z1 {
        v = m[0];
        base = 0;
    }
    let mut lg = 0i16;
    for k in [8u32, 4, 2, 1] {
        if v >= 1 << k {
            v = srs16(i64::from(v), k);
            lg = lg.wrapping_add(k as i16);
        }
    }
    let big_l = base.wrapping_add(lg);
    let sh = (big_l.wrapping_sub(23))
        .max((-149i16).wrapping_sub(x))
        .min(55);
    let (c_e, c7, c23, c39) = (sh < -8, sh < 8, sh < 24, sh < 40);
    let mut s = 55i16;
    if c39 {
        s = 39;
    }
    if c23 {
        s = 23;
    }
    if c7 {
        s = 7;
    }
    if c_e {
        s = -9;
    }
    let k = s.wrapping_sub(sh) as u16;
    debug_assert!(k <= 15);
    let pick = |bit: u16, on: u16| if k & bit != 0 { on } else { 1 };
    let p = srs16(
        i64::from(pick(8, 256) * pick(4, 16)) * i64::from(pick(2, 4) * pick(1, 2)),
        0,
    );
    let c_r = !c39;
    let m1s = if m[0] != 0 { m[1] | 1 } else { m[1] };
    let limbs = if c_r {
        [m1s, m[2], m[3], 0]
    } else if c_e {
        [0, 0, m[0], 0]
    } else {
        m
    };
    let mut acc = i64::from(limbs[3]) * i64::from(p);
    for &limb in limbs[..3].iter().rev() {
        acc = (acc << 16).wrapping_add(i64::from(limb) * i64::from(p));
    }
    let srs32_rne = |v: i64, s: u32| {
        let fl = v >> s;
        let rest = v - (fl << s);
        let half = 1i64 << (s - 1);
        let r = fl + i64::from(rest > half || (rest == half && fl & 1 == 1));
        r.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
    };
    let (q7, q23, q39) = (srs32_rne(acc, 7), srs32_rne(acc, 23), srs32_rne(acc, 39));
    let mut q = q23;
    if c7 && !c_e {
        q = q7;
    }
    if !c23 {
        q = q39;
    }
    // Ef <= 254 keeps `q + Ef<<23 <= 2^31` (q <= 2^24); the unsigned min maps every overflow to +inf bits.
    let ef = sh.wrapping_add(x).wrapping_add(149).min(254);
    let e23 = i32::from(srs16(i64::from(ef) << 7, 0) as i16) << 16;
    let mut bits = ((q.wrapping_add(e23)) as u32).min(0x7f80_0000) as i32;
    if q == 0 {
        bits = 0;
    }
    if neg {
        bits |= i32::MIN;
    }
    bits as u32
}

// ---------------------------------------------------------------------------------------------------------------
// Instruction builder: operand fields from `isa::gen`, operand read/write stages from the itineraries.
// ---------------------------------------------------------------------------------------------------------------

/// A scheduling resource. Every sub-register of an accumulator maps to its `dm`, every `y`/`w` to its `x`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Res {
    X(u8),
    Dm(u8),
    R(u8),
    P(u8),
    M(u8),
    S(u8),
    Cr(&'static str),
    Mem(u8),
}

/// Memory regions, ordered independently by the list scheduler.
const MEM_A: u8 = 0;
const MEM_B: u8 = 1;
const MEM_C16: u8 = 2;
const MEM_PAT: u8 = 3;
const MEM_AB: u8 = 4;
const MEM_CONST: u8 = 5;
const MEM_I: u8 = 6;
const MEM_OUT: u8 = 7;

#[derive(Clone, Copy, Debug)]
enum Arg {
    R(&'static str),
    I(i64),
}
use Arg::{I as Im, R as Rg};

/// One encoded slot instruction with its 0-based read / write stage offsets.
#[derive(Clone, Debug)]
struct Ins {
    slot: Slot,
    bits: u64,
    reads: Vec<(Res, i64)>,
    writes: Vec<(Res, i64)>,
    ports: Vec<(u8, i64, i64, bool)>,
}

fn res_of(name: &'static str) -> Vec<Res> {
    let split = name
        .find(|c: char| c.is_ascii_digit())
        .unwrap_or(name.len());
    let (prefix, digits) = name.split_at(split);
    let n: u8 = match digits.parse() {
        Ok(n) => n,
        Err(_) => {
            return if prefix.starts_with("cr") {
                vec![Res::Cr(name)]
            } else {
                vec![]
            }
        }
    };
    match prefix {
        "x" => vec![Res::X(n)],
        "y" => vec![Res::X(2 * n), Res::X(2 * n + 1)],
        "wl" | "wh" => vec![Res::X(n)],
        "dm" | "cml" | "cmh" | "bmll" | "bmlh" | "bmhl" | "bmhh" => vec![Res::Dm(n)],
        "r" => vec![Res::R(n)],
        "p" => vec![Res::P(n)],
        "m" => vec![Res::M(n)],
        "s" => vec![Res::S(n)],
        _ => vec![],
    }
}

fn encoding(name: &str) -> &'static Encoding {
    static ALL: LazyLock<HashMap<&'static str, &'static Encoding>> =
        LazyLock::new(|| isa::gen::ENCODINGS.iter().map(|e| (e.name, e)).collect());
    ALL.get(name)
        .copied()
        .unwrap_or_else(|| panic!("no encoding {name}"))
}

/// Encode `name` with named operands (`Rg("x3")` registers by name, `Im(v)` immediates) and derive its timing.
fn ins(name: &str, args: &[(&str, Arg)]) -> Ins {
    ins_mem(name, args, None)
}
fn ins_mem(name: &str, args: &[(&str, Arg)], region: Option<u8>) -> Ins {
    let enc = encoding(name);
    let width = |op: &str| {
        enc.fields
            .iter()
            .filter(|f| f.name == op)
            .map(|f| f.source_lsb + f.width)
            .max()
            .unwrap_or(0)
    };
    let mut fields: Vec<(&str, u64)> = Vec::new();
    let mut regs: Vec<(&'static str, &'static str, bool)> = Vec::new();
    let mut canonical: Vec<Option<&str>> = Vec::new();
    for op in enc.operands {
        match args.iter().find(|(n, _)| *n == op.name).map(|(_, a)| *a) {
            Some(Arg::R(reg)) => {
                let r = op
                    .registers
                    .iter()
                    .find(|r| r.name == reg)
                    .unwrap_or_else(|| panic!("{name}: register {reg} not in operand {}", op.name));
                let w = width(op.name);
                if w > 0 {
                    fields.push((op.name, r.value & ((1u64 << w) - 1)));
                }
                regs.push((op.name, r.name, op.output));
                canonical.push(Some(r.name));
            }
            Some(Arg::I(v)) => {
                let w = width(op.name);
                fields.push((op.name, (v as u64) & ((1u64 << w) - 1)));
                canonical.push(None);
            }
            None => {
                assert!(op.width == 0, "{name}: missing operand {}", op.name);
                canonical.push(None);
            }
        }
    }
    // Tied implicit operands (`$ptr_out = $ptr`, `$dst = $acc1`), resolved once every explicit register is known.
    for (index, op) in enc.operands.iter().enumerate() {
        if canonical[index].is_some() || op.width != 0 || args.iter().any(|(n, _)| *n == op.name) {
            continue;
        }
        let key = format!("${} = $", op.name);
        let tie = enc
            .constraints
            .split(',')
            .map(str::trim)
            .find_map(|c| c.strip_prefix(key.as_str()));
        if let Some(reg) =
            tie.and_then(|t| regs.iter().find(|(n, _, _)| *n == t).map(|(_, r, _)| *r))
        {
            regs.push((op.name, reg, op.output));
            canonical[index] = Some(reg);
        } else if let [only] = op.registers {
            // Fixed implicit register (e.g. the r16 compare output of VMAX_LT / VMIN_GE).
            regs.push((op.name, only.name, op.output));
            canonical[index] = Some(only.name);
        }
    }
    let bits = encode_slot(enc, &fields, 0).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    let info = sched::opcode_info(name).unwrap_or_else(|| panic!("{name}: no itinerary"));
    let it = info.select(&canonical);
    let mut i = Ins {
        slot: enc.slot,
        bits,
        reads: Vec::new(),
        writes: Vec::new(),
        ports: it
            .resources
            .iter()
            .map(|u| {
                (
                    u.resource,
                    i64::from(u.start),
                    i64::from(u.cycles),
                    u.reserved,
                )
            })
            .collect(),
    };
    for t in it.operands {
        let stage = i64::from(t.cycle) - 1;
        if let Some((_, reg, out)) = regs.iter().find(|(n, _, _)| *n == t.name) {
            for r in res_of(reg) {
                if *out {
                    i.writes.push((r, stage))
                } else {
                    i.reads.push((r, stage))
                }
            }
        } else if t.name.starts_with("cr") {
            i.reads.push((Res::Cr(t.name), stage));
        }
    }
    for imp in info.implicit {
        let stage = it
            .operands
            .get(imp.operand)
            .map_or(0, |o| i64::from(o.cycle) - 1);
        for r in res_of(imp.register) {
            if imp.write {
                i.writes.push((r, stage))
            } else {
                i.reads.push((r, stage))
            }
        }
    }
    if it.load || it.store {
        let region = region.unwrap_or_else(|| panic!("{name}: memory op needs a region"));
        let stage = i64::from(it.memory_cycles.last().copied().unwrap_or(1)) - 1;
        if it.store {
            i.writes.push((Res::Mem(region), stage))
        } else {
            i.reads.push((Res::Mem(region), stage))
        }
    }
    i
}

// Register names.
fn x(n: usize) -> &'static str {
    [
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11",
    ][n]
}
fn y(n: usize) -> &'static str {
    ["y0", "y1", "y2", "y3", "y4", "y5"][n]
}
fn dm(n: usize) -> &'static str {
    ["dm0", "dm1", "dm2", "dm3", "dm4"][n]
}
fn cm(n: usize, high: bool) -> &'static str {
    if high {
        ["cmh0", "cmh1", "cmh2", "cmh3", "cmh4"][n]
    } else {
        ["cml0", "cml1", "cml2", "cml3", "cml4"][n]
    }
}
fn bm(n: usize, q: usize) -> &'static str {
    [
        ["bmll0", "bmlh0", "bmhl0", "bmhh0"],
        ["bmll1", "bmlh1", "bmhl1", "bmhh1"],
        ["bmll2", "bmlh2", "bmhl2", "bmhh2"],
        ["bmll3", "bmlh3", "bmhl3", "bmhh3"],
        ["bmll4", "bmlh4", "bmhl4", "bmhh4"],
    ][n][q]
}
fn r(n: usize) -> &'static str {
    [
        "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "r13",
        "r14", "r15", "r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23", "r24", "r25", "r26",
        "r27", "r28", "r29", "r30", "r31",
    ][n]
}
fn p(n: usize) -> &'static str {
    ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"][n]
}
fn s(n: usize) -> &'static str {
    ["s0", "s1", "s2", "s3"][n]
}

/// Elementwise 16x16 -> acc64 configurations (`aie2p_compute_control`: amode 1, bmode 3, variant 2; bit 9 signs
/// s1, bit 8 signs s2, bit 10 shift16 of acc1).
const ELEM: u32 = (1 << 1) | (3 << 3) | (2 << 5);
const CONF_INT8: u32 = super::gemm_core::SIGNED_8X8;
const CONF_UU: u32 = ELEM;
const CONF_US: u32 = ELEM | (1 << 8);
const CONF_US16: u32 = ELEM | (1 << 8) | (1 << 10);
const CONF_UU16: u32 = ELEM | (1 << 10);
const CONF_NEG64: u32 = 1 << 1;
// Scalar register plan: r0 = -1, r1 = 1 (lock values), r2 / r6..r10 multiplier configurations, r3 wave counter,
// r4 ping/pong phase, r5 epoch counter, r12 loop counter, r13 = 0, r14 = -16, r15 scratch,
// r20..r27 shuffle modes 12..19, r16..r19 / r28..r31 compare masks.
const R_INT8: usize = 2;
const R_UU: usize = 6;
const R_US: usize = 7;
const R_US16: usize = 8;
const R_UU16: usize = 9;
const R_NEG: usize = 10;
const R_LOOP: usize = 12;
/// Position inside a fold pair (0: even epoch of the pair next, 1: odd).
const R_PAIR: usize = 11;
const R_ZERO: usize = 13;
const R_M16: usize = 14;
/// The final pack keeps its sign mask in the T128_HI mode register (r21, unused by the pack) and restores it after.
const R_NEGMASK: usize = 21;
/// `VSHUFFLE` modes (`aie2p_enums.h`): register r20 + (mode - 12).
const T128_LO: usize = 12;
const T128_HI: usize = 13;
const T64_LO: usize = 14;
const T64_HI: usize = 15;
const T32_LO: usize = 16;
const T32_HI: usize = 17;
const T16_LO: usize = 18;
const T16_HI: usize = 19;
fn mode(m: usize) -> &'static str {
    r(20 + m - 12)
}

fn vmac(dst: usize, acc1: Option<usize>, s1: &'static str, s2: &'static str, conf: usize) -> Ins {
    match acc1 {
        None => ins(
            "VMUL_vmul_cm_core_X_X",
            &[
                ("dst", Rg(dm(dst))),
                ("s1", Rg(s1)),
                ("s2", Rg(s2)),
                ("acc", Rg(r(conf))),
            ],
        ),
        Some(a) => ins(
            "VMAC_vmul_cm_core_X_X",
            &[
                ("dst", Rg(dm(dst))),
                ("acc1", Rg(dm(a))),
                ("s1", Rg(s1)),
                ("s2", Rg(s2)),
                ("acc", Rg(r(conf))),
            ],
        ),
    }
}
fn vaddmac(acc1: usize, acc2: usize, s1: &'static str, s2: &'static str, conf: usize) -> Ins {
    ins(
        "VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_X",
        &[
            ("acc1", Rg(dm(acc1))),
            ("acc2", Rg(dm(acc2))),
            ("s1", Rg(s1)),
            ("s2", Rg(s2)),
            ("acc", Rg(r(conf))),
        ],
    )
}
fn vneg(dst: usize, src: usize) -> Ins {
    ins(
        "VNEG",
        &[
            ("dst", Rg(dm(dst))),
            ("acc1", Rg(dm(src))),
            ("acc", Rg(r(R_NEG))),
        ],
    )
}
fn srs4(dst: usize, src: usize, sh: usize) -> Ins {
    ins(
        "VSRS_4x_mv_x_srs_dm_srsSign1",
        &[("dst", Rg(x(dst))), ("src", Rg(dm(src))), ("su", Rg(s(sh)))],
    )
}
fn srs2(dst: usize, src: usize, high: bool, sh: usize) -> Ins {
    ins(
        "VSRS_2x_mv_x_srs_cm_srsSign1",
        &[
            ("dst", Rg(x(dst))),
            ("src", Rg(cm(src, high))),
            ("su", Rg(s(sh))),
        ],
    )
}
fn st_srs2(src: usize, high: bool, sh: usize, ptr: usize, region: u8) -> Ins {
    ins_mem(
        "VST_SRS_2x_dm_sts_srs_cm_pstm_nrm_imm_srsSign1",
        &[
            ("src", Rg(cm(src, high))),
            ("su", Rg(s(sh))),
            ("ptr", Rg(p(ptr))),
            ("imm", Im(1)),
        ],
        Some(region),
    )
}
fn ups4(dst: usize, src: usize, sh: usize, signed: bool) -> Ins {
    let name = if signed {
        "VUPS_4x_mv_ups_x2d_upsSign1"
    } else {
        "VUPS_4x_mv_ups_x2d_upsSign0"
    };
    ins(
        name,
        &[("dst", Rg(dm(dst))), ("src", Rg(x(src))), ("su", Rg(s(sh)))],
    )
}
/// 64 B x load with post-increment `step` (units of 64 B) on the LDA or LDB slot.
fn ld(dst: usize, ptr: usize, step: i64, b_slot: bool, region: u8) -> Ins {
    let name = if b_slot {
        "VLDB_dmx_ldb_x_pstm_nrm_imm"
    } else {
        "VLDA_dmx_lda_x_pstm_nrm_imm"
    };
    ins_mem(
        name,
        &[("dst", Rg(x(dst))), ("ptr", Rg(p(ptr))), ("imm", Im(step))],
        Some(region),
    )
}
/// 64 B x load at `[ptr + 64*off]`, `off` in -8..=7 (no pointer update).
fn ld_at(dst: usize, ptr: usize, off: i64, b_slot: bool, region: u8) -> Ins {
    assert!((-8..=7).contains(&off));
    let name = if b_slot {
        "VLDB_dmx_ldb_x_idx_imm"
    } else {
        "VLDA_dmx_lda_x_idx_imm"
    };
    ins_mem(
        name,
        &[("dst", Rg(x(dst))), ("ptr", Rg(p(ptr))), ("imm", Im(off))],
        Some(region),
    )
}
fn st(src: usize, ptr: usize, step: i64, region: u8) -> Ins {
    ins_mem(
        "VST_dmx_sts_x_pstm_nrm_imm",
        &[("src", Rg(x(src))), ("ptr", Rg(p(ptr))), ("imm", Im(step))],
        Some(region),
    )
}
fn st_at(src: usize, ptr: usize, off: i64, region: u8) -> Ins {
    assert!((-8..=7).contains(&off));
    ins_mem(
        "VST_dmx_sts_x_idx_imm",
        &[("src", Rg(x(src))), ("ptr", Rg(p(ptr))), ("imm", Im(off))],
        Some(region),
    )
}
fn ld_bm(acc: usize, q: usize, ptr: usize, region: u8) -> Ins {
    ins_mem(
        "VLDA_dmx_lda_bm_pstm_nrm_imm",
        &[("dst", Rg(bm(acc, q))), ("ptr", Rg(p(ptr))), ("imm", Im(1))],
        Some(region),
    )
}
fn st_bm(acc: usize, q: usize, ptr: usize, region: u8) -> Ins {
    ins_mem(
        "VST_dmx_sts_bm_pstm_nrm_imm",
        &[("src", Rg(bm(acc, q))), ("ptr", Rg(p(ptr))), ("imm", Im(1))],
        Some(region),
    )
}
fn unpack_ld(dst_y: usize, ptr: usize, region: u8) -> Ins {
    ins_mem(
        "VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_imm_unpackSign1",
        &[("dst", Rg(y(dst_y))), ("ptr", Rg(p(ptr))), ("imm", Im(1))],
        Some(region),
    )
}
fn unpack(dst_y: usize, src: usize) -> Ins {
    ins(
        "VUNPACK_mv_unpack_x_unpackSign1",
        &[("dst", Rg(y(dst_y))), ("src", Rg(x(src)))],
    )
}
fn shuffle(dst: usize, s1: usize, s2: usize, m: usize) -> Ins {
    ins(
        "VSHUFFLE_vec_shuffle_x",
        &[
            ("dst", Rg(x(dst))),
            ("s1", Rg(x(s1))),
            ("s2", Rg(x(s2))),
            ("mod", Rg(mode(m))),
        ],
    )
}
/// Two-source vector ALU op `d = op(s1, s2)` (`VADD_16`, `VSUB_32`, `VBAND`, `VMAX_LT_16_vaddSign1`, ...).
fn alu(name: &str, d: usize, s1: usize, s2: usize) -> Ins {
    ins(
        name,
        &[("d", Rg(x(d))), ("s1", Rg(x(s1))), ("s2", Rg(x(s2)))],
    )
}
/// Vector compare into mask register `mask` (`VLT_16_vaddSign1`, ...).
fn cmp(name: &str, mask: usize, s1: usize, s2: usize) -> Ins {
    ins(
        name,
        &[("cmp", Rg(r(mask))), ("s1", Rg(x(s1))), ("s2", Rg(x(s2)))],
    )
}
fn eqz(name: &str, mask: usize, src: usize) -> Ins {
    ins(name, &[("cmp", Rg(r(mask))), ("s2", Rg(x(src)))])
}
/// `d = mask ? s2 : s1` lane-wise.
fn sel(name: &str, d: usize, s1: usize, s2: usize, mask: usize) -> Ins {
    ins(
        name,
        &[
            ("d", Rg(x(d))),
            ("s1", Rg(x(s1))),
            ("s2", Rg(x(s2))),
            ("sel", Rg(r(mask))),
        ],
    )
}
fn bcst(name: &str, dst: usize, src: usize) -> Ins {
    ins(name, &[("dst", Rg(x(dst))), ("src", Rg(r(src)))])
}
fn movxm_r(dst: usize, v: u32) -> Ins {
    ins("MOVXM", &[("dst", Rg(r(dst))), ("i", Im(i64::from(v)))])
}
fn movxm_p(dst: usize, v: u32) -> Ins {
    ins("MOVXM", &[("dst", Rg(p(dst))), ("i", Im(i64::from(v)))])
}
fn movxm_s(dst: usize, v: u32) -> Ins {
    ins("MOVXM", &[("dst", Rg(s(dst))), ("i", Im(i64::from(v)))])
}
fn movx_cr(name: &'static str, v: i64) -> Ins {
    ins("MOVX_mvx_cr_imm", &[("dst", Rg(name)), ("src", Im(v))])
}
fn sca(name: &str, d: usize, s0: usize, s1: usize) -> Ins {
    ins(
        name,
        &[("d0", Rg(r(d))), ("s0", Rg(r(s0))), ("s1", Rg(r(s1)))],
    )
}
fn padd(slot: Slot, ptr: usize, modifier: usize) -> Ins {
    let name = match slot {
        Slot::Lda => "PADDA_pstm_nrm",
        Slot::Ldb => "PADDB_pstm_nrm",
        _ => "PADDS_pstm_nrm",
    };
    ins(
        name,
        &[
            ("ptr", Rg(p(ptr))),
            (
                "mod",
                Rg(["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7"][modifier]),
            ),
        ],
    )
}

// ---------------------------------------------------------------------------------------------------------------
// Cycle tables and the list scheduler.
// ---------------------------------------------------------------------------------------------------------------

/// A slot combination is encodable iff some bundle format holds all of its slots (`full`: a 128-bit format).
fn legal(mask: u8, full: bool) -> bool {
    static MASKS: LazyLock<Vec<(u8, bool)>> = LazyLock::new(|| {
        isa::gen::FORMATS
            .iter()
            .map(|f| {
                (
                    f.slots.iter().fold(0u8, |m, s| m | (1 << s.slot as usize)),
                    f.bits == 128,
                )
            })
            .collect()
    });
    MASKS
        .iter()
        .any(|&(m, wide)| m & mask == mask && (wide || !full))
}
fn mask_of(c: &Cycle) -> u8 {
    c.iter()
        .enumerate()
        .fold(0, |m, (i, s)| if s.is_some() { m | (1 << i) } else { m })
}

/// Itinerary resource reservations `(resource, cycle) -> reserved` of placed ops (register ports, LCKREQ, ...).
#[derive(Default)]
struct Ports(HashMap<(u8, i64), bool>);
impl Ports {
    /// Reserved vs reserved does not conflict; either vs required does (`sched::ResourceUse`).
    fn free(&self, t: i64, i: &Ins) -> bool {
        i.ports.iter().all(|&(res, start, cycles, reserved)| {
            (0..cycles).all(|k| {
                self.0
                    .get(&(res, t + start + k))
                    .is_none_or(|&other| other && reserved)
            })
        })
    }
    fn take(&mut self, t: i64, i: &Ins) {
        for &(res, start, cycles, reserved) in &i.ports {
            for k in 0..cycles {
                let e = self.0.entry((res, t + start + k)).or_insert(reserved);
                *e = *e && reserved;
            }
        }
    }
}

/// Explicitly timed table (phases G and F): `put` panics on a slot collision, a resource (register port) conflict or
/// an unencodable 128-bit bundle.
#[derive(Default)]
struct Table {
    cycles: Vec<Cycle>,
    ports: Ports,
}
impl Table {
    fn put(&mut self, t: usize, i: &Ins) {
        if self.cycles.len() <= t {
            self.cycles.resize(t + 1, [None; 8]);
        }
        self.ports.take(t as i64, i);
        let c = &mut self.cycles[t];
        assert!(
            c[i.slot as usize].replace(i.bits).is_none(),
            "table slot collision at {t} {:?}",
            i.slot
        );
        assert!(
            legal(mask_of(c), true),
            "no 128-bit bundle format for slots {:#010b} at {t}",
            mask_of(c)
        );
    }
}

/// In-order list scheduler: every op goes to the earliest cycle that keeps program-order semantics (RAW and WAW
/// need a strictly later stage, WAR may write in the read cycle; no bypass is assumed) in a free, encodable slot.
/// `finish` pads the block until every placed op has retired, so blocks and loop iterations compose sequentially.
#[derive(Default)]
struct List {
    cycles: Vec<Cycle>,
    writes: HashMap<Res, i64>,
    reads: HashMap<Res, i64>,
    issued: HashMap<Res, i64>,
    ports: Ports,
    end: i64,
    floor: i64,
    limit: Option<usize>,
    pushed: usize,
}
impl List {
    fn ok(&self, t: i64, i: &Ins) -> bool {
        if let Some(c) = self.cycles.get(t as usize) {
            if c[i.slot as usize].is_some() {
                return false;
            }
            let mut c = *c;
            c[i.slot as usize] = Some(0);
            if !legal(mask_of(&c), false) {
                return false;
            }
        }
        self.ports.free(t, i)
            && i.reads
                .iter()
                .all(|&(r, s)| self.writes.get(&r).is_none_or(|&w| t + s > w))
            && i.writes.iter().all(|&(r, s)| {
                self.writes.get(&r).is_none_or(|&w| t + s > w)
                // WAR strictly later: a write with a matching bypass class is visible one cycle early.
                && self.reads.get(&r).is_none_or(|&rd| t + s > rd)
                // Writers of one register issue in program order, never in one bundle.
                && self.issued.get(&r).is_none_or(|&i| t > i)
            })
    }
    fn push(&mut self, i: Ins) -> i64 {
        self.pushed += 1;
        if self.limit.is_some_and(|n| self.pushed > n) {
            return -1;
        }
        let mut t = self.floor;
        while !self.ok(t, &i) {
            t += 1;
        }
        if self.cycles.len() <= t as usize {
            self.cycles.resize(t as usize + 1, [None; 8]);
        }
        self.cycles[t as usize][i.slot as usize] = Some(i.bits);
        self.ports.take(t, &i);
        for &(r, s) in &i.reads {
            let e = self.reads.entry(r).or_insert(t + s);
            *e = (*e).max(t + s);
        }
        for &(r, s) in &i.writes {
            let e = self.writes.entry(r).or_insert(t + s);
            *e = (*e).max(t + s);
            let e = self.issued.entry(r).or_insert(t);
            *e = (*e).max(t);
        }
        let last = i
            .reads
            .iter()
            .chain(&i.writes)
            .map(|&(_, s)| s)
            .max()
            .unwrap_or(0);
        self.end = self.end.max(t + last + 1);
        t
    }
    /// Every later op issues after every op pushed so far (control-register changes).
    fn barrier(&mut self) {
        self.floor = self.cycles.len() as i64;
    }
    fn finish(mut self) -> Vec<Cycle> {
        let len = (self.end as usize).max(self.cycles.len());
        self.cycles.resize(len, [None; 8]);
        self.cycles
    }
}

const SLOT_ORDER: [Slot; 8] = [
    Slot::Ldb,
    Slot::Alu,
    Slot::Lng,
    Slot::Lda,
    Slot::Mv,
    Slot::St,
    Slot::Vec,
    Slot::Nop16,
];
fn emit_bundle(a: &mut Assembler, c: &Cycle) {
    let v: Vec<(Slot, u64)> = c
        .iter()
        .enumerate()
        .filter_map(|(i, b)| b.map(|b| (SLOT_ORDER[i], b)))
        .collect();
    if v.is_empty() {
        a.nop(1)
    } else {
        a.emit(&v)
    }
}
fn emit_block(a: &mut Assembler, cycles: &[Cycle]) {
    let serial = a.serial;
    a.serial = false;
    for c in cycles {
        emit_bundle(a, c);
    }
    a.serial = serial;
}
/// Insert `add counter, -1` (first free ALU slot) and `jnz counter, begin` (LNG slot, five bundles before the end, so
/// the delay slots are body bundles) into a loop body; pads the body when those slots are taken.
fn close_loop(body: &mut Vec<Cycle>, begin: u64, counter: usize, full: bool) {
    while body.len() < 8 {
        body.push([None; 8]);
    }
    let fits = |c: &Cycle, slot: Slot| {
        c[slot as usize].is_none() && {
            let mut c = *c;
            c[slot as usize] = Some(0);
            legal(mask_of(&c), full)
        }
    };
    while !fits(
        &body[body.len() - 1 - isa::sched::JUMP_DELAY_SLOTS],
        Slot::Lng,
    ) {
        body.push([None; 8]);
    }
    let jnz_at = body.len() - 1 - isa::sched::JUMP_DELAY_SLOTS;
    let dec_at = (0..jnz_at - 1)
        .find(|&t| fits(&body[t], Slot::Alu))
        .expect("loop counter needs a free ALU slot");
    body[dec_at][Slot::Alu as usize] = Some(
        ins(
            "ADD_add_r_ri",
            &[
                ("d0", Rg(r(counter))),
                ("s0", Rg(r(counter))),
                ("imm", Im(-1)),
            ],
        )
        .bits,
    );
    body[jnz_at][Slot::Lng as usize] =
        Some(ins("JNZ", &[("i", Im(begin as i64)), ("s0", Rg(r(counter)))]).bits);
}
/// `count` iterations of a list-scheduled `body` as a JNZ loop on `counter` (no hardware loop).
fn emit_loop(a: &mut Assembler, mut body: Vec<Cycle>, count: u32, counter: usize) {
    assert!(count >= 1);
    let serial = a.serial;
    a.serial = false;
    let m = movxm_r(counter, count);
    a.emit(&[(m.slot, m.bits)]);
    a.nop(2);
    while a.program.pc() % 16 != 0 {
        a.nop(1);
    }
    let begin = a.program.pc() as u64;
    close_loop(&mut body, begin, counter, false);
    for c in &body {
        emit_bundle(a, c);
    }
    a.serial = serial;
}
/// Emit a fully timed table whose cycles `[start, start + period * reps)` repeat exactly: prologue, a JNZ loop over
/// one period, and the tail, all as 128-bit bundles so the timing and the 16-byte loop alignment are exact.
fn emit_periodic(
    a: &mut Assembler,
    cycles: &[Cycle],
    start: usize,
    period: usize,
    reps: usize,
    counter: usize,
) {
    for k in 1..reps {
        assert_eq!(
            &cycles[start..start + period],
            &cycles[start + k * period..start + (k + 1) * period],
            "table period {k} differs"
        );
    }
    let serial = a.serial;
    a.serial = false;
    let m = movxm_r(counter, reps as u32);
    a.emit(&[(m.slot, m.bits)]);
    a.nop(2);
    while a.program.pc() % 16 != 0 {
        a.nop(1);
    }
    for c in &cycles[..start] {
        emit_cycle(a, c);
    }
    let begin = a.program.pc() as u64;
    let mut body = cycles[start..start + period].to_vec();
    close_loop(&mut body, begin, counter, true);
    assert_eq!(
        body.len(),
        period,
        "periodic table body has no free branch slots"
    );
    for c in &body {
        emit_cycle(a, c);
    }
    for c in &cycles[start + period * reps..] {
        emit_cycle(a, c);
    }
    a.serial = serial;
}

// ---------------------------------------------------------------------------------------------------------------
// Phase G: per epoch, 8 rows (mb) x 16 kb x 4 nb int8 VMACs; finished blocks leave as int16 through VST.SRS.2x.
// ---------------------------------------------------------------------------------------------------------------

/// Cycles of one GEMM row: 64 VMACs plus one bubble (the drain window of the four accumulators).
const G_ROW: usize = 65;
const G_T0: usize = 12;
/// x0..x3 A (y0, y1 alternate per kb pair), x4..x7 B (y2 = nb 0/1, y3 = nb 2/3), x8 raw int4 A; dm0..dm3 C.
/// Pointers: p0 / p1 walk the E / O A halves (`pair_acquire`), p2 the B chunk (m0 = -2048 per row), p3 C16.
fn phase_g_table() -> Table {
    let mut t = Table::default();
    for row in 0..8 {
        let base = G_T0 + G_ROW * row;
        let pa = row % 2;
        for j in 0..8 {
            t.put(base + 8 * j - 12, &ld(8, pa, 1, false, MEM_A));
            t.put(base + 8 * j - 9, &unpack(j % 2, 8));
        }
        for kb in 0..16 {
            t.put(base + 4 * kb - 8, &unpack_ld(2, 2, MEM_B));
            t.put(base + 4 * kb - 6, &unpack_ld(3, 2, MEM_B));
            let ax = x(2 * ((kb / 2) % 2) + kb % 2);
            for nb in 0..4 {
                t.put(
                    base + 4 * kb + nb,
                    &vmac(nb, (kb > 0).then_some(nb), ax, x(4 + nb), R_INT8),
                );
            }
        }
        t.put(base + 55, &padd(Slot::Ldb, 2, 0));
        for nb in 0..4 {
            for half in 0..2 {
                t.put(
                    base + 66 + 2 * nb + half,
                    &st_srs2(nb, half == 1, 0, 3, MEM_C16),
                );
            }
        }
    }
    t
}

// ---------------------------------------------------------------------------------------------------------------
// Phase F: 64 half blocks (mb, nb, h) of 32 lanes, II = 6:  P = Dpat*Spat;  T = Phi*C;  T = (T<<16) + I + Plo*C.
// ---------------------------------------------------------------------------------------------------------------

const F_II: usize = 6;
const F_T0: usize = 14;
/// x0..x3 Spat[nb], x4 / x5 Dpat[mb][half], x6 C16, x7 Plo, x8 Phi; dm4 P, dm0 / dm1 T, dm2 I.
/// Pointers: p4 Dpat, p5 C16, p6 I loads, p7 I stores (all +64 B); s0 = 0, s1 = 16.
fn phase_f_table() -> Table {
    let mut t = Table::default();
    t.put(0, &ld(4, 4, 1, true, MEM_PAT));
    t.put(2, &ld(5, 4, 1, true, MEM_PAT));
    for h in 0..64 {
        let t0 = F_T0 + F_II * h;
        let (nb, half) = ((h / 2) % 4, h % 2);
        if h % 8 >= 6 && h < 56 {
            t.put(t0, &ld(4 + h % 2, 4, 1, true, MEM_PAT));
        }
        t.put(t0 + 4, &ld(6, 5, 1, true, MEM_C16));
        t.put(t0, &vmac(4, None, x(4 + half), x(nb), R_UU));
        t.put(t0 + 6, &srs4(7, 4, 0));
        t.put(t0 + 7, &srs4(8, 4, 1));
        for q in 0..4 {
            t.put(t0 + 6 + q, &ld_bm(2, q, 6, MEM_I));
        }
        t.put(t0 + 11, &vmac(half, None, x(8), x(6), R_US));
        t.put(t0 + 14, &vaddmac(half, 2, x(7), x(6), R_US16));
        for q in 0..4 {
            t.put(t0 + 20 + q, &st_bm(half, q, 7, MEM_I));
        }
    }
    t
}

// Phase F2: the fold of two epochs per pass (I loaded and stored once), 64 half blocks, II = 8.
//   T = Phi_a*Ca;  T += Phi_b*Cb;  T = (T<<16) + I + Plo_a*Ca;  T += Plo_b*Cb.
const F2_II: usize = 8;
const F2_T0: usize = 14;
/// x0 Sa, x1 Da, x2 Sb, x3 Db, x4 Ca, x5 Cb, x6 Plo_a, x7 Phi_a, x8 Phi_b, x9 / x10 Plo_b (half parity);
/// dm3 Pa, dm4 Pb, dm0 / dm1 T (half parity), dm2 I. Pointers: p0 Spat (`[slot][nb]`), p1 / p2 Dpat a / b (+128 B
/// per mb through m2), p3 / p4 C16 a / b, p5 I loads, p6 I stores; s0 = 0, s1 = 16.
fn phase_f2_table() -> Table {
    let mut t = Table::default();
    for h in 0..64 {
        let t0 = F2_T0 + F2_II * h;
        let (nb, half) = ((h / 2) % 4, h % 2);
        let plo_b = 9 + h % 2;
        t.put(t0 - 14, &ld_at(0, 0, nb as i64, true, MEM_PAT));
        t.put(t0 - 8, &ld_at(1, 1, half as i64, false, MEM_PAT));
        t.put(t0 - 7, &ld_at(2, 0, 4 + nb as i64, true, MEM_PAT));
        t.put(t0 - 10, &ld_at(3, 2, half as i64, true, MEM_PAT));
        if h % 8 == 7 {
            t.put(t0 - 3, &padd(Slot::Lda, 1, 2));
            t.put(t0 - 5, &padd(Slot::Ldb, 2, 2));
        }
        t.put(t0 + 5, &ld(4, 3, 1, true, MEM_C16));
        t.put(t0 + 8, &ld(5, 4, 1, true, MEM_C16));
        t.put(t0, &vmac(3, None, x(1), x(0), R_UU));
        t.put(t0 + 1, &vmac(4, None, x(3), x(2), R_UU));
        t.put(t0 + 7, &srs4(6, 3, 0));
        t.put(t0 + 8, &srs4(7, 3, 1));
        t.put(t0 + 9, &srs4(plo_b, 4, 0));
        t.put(t0 + 10, &srs4(8, 4, 1));
        for q in 0..4 {
            t.put(t0 + 9 + q, &ld_bm(2, q, 5, MEM_I));
        }
        t.put(t0 + 12, &vmac(half, None, x(7), x(4), R_US));
        t.put(t0 + 15, &vmac(half, Some(half), x(8), x(5), R_US));
        t.put(t0 + 18, &vaddmac(half, 2, x(6), x(4), R_US16));
        t.put(t0 + 21, &vmac(half, Some(half), x(plo_b), x(5), R_US));
        for q in 0..4 {
            t.put(t0 + 27 + q, &st_bm(half, q, 6, MEM_I));
        }
    }
    t
}

// ---------------------------------------------------------------------------------------------------------------
// Patterns, constants and the final pack (list scheduled).
// ---------------------------------------------------------------------------------------------------------------

/// Interleave chain: from a 32-lane vector of 4 x 8 per-token values, the 8 half-row patterns `(mbl, half)` (each
/// token repeated over the 8 lanes of its row). Temporaries x9, x10, x11; `src` is preserved.
fn row_patterns(l: &mut List, src: usize, mut emit: impl FnMut(&mut List, usize, usize, usize)) {
    for (hk, m16) in [(0, T16_LO), (1, T16_HI)] {
        l.push(shuffle(9, src, src, m16));
        for (qk, m32) in [(0, T32_LO), (1, T32_HI)] {
            l.push(shuffle(10, 9, 9, m32));
            for (half, m64) in [(0, T64_LO), (1, T64_HI)] {
                l.push(shuffle(11, 10, 10, m64));
                emit(l, 2 * hk + qk, half, 11);
            }
        }
    }
}
/// The 4 column patterns (8 features repeated 4x) of a 32-lane vector into `dst..dst+4`; temporaries x9, x10.
fn col_patterns(l: &mut List, src: usize, dst: usize) {
    l.push(shuffle(9, src, src, T128_LO));
    l.push(shuffle(10, src, src, T128_HI));
    l.push(shuffle(dst, 9, 9, T128_LO));
    l.push(shuffle(dst + 1, 9, 9, T128_HI));
    l.push(shuffle(dst + 2, 10, 10, T128_LO));
    l.push(shuffle(dst + 3, 10, 10, T128_HI));
}
/// Row patterns of the E (`x[e]`) and O (`x[o]`) halves stored to DPAT `[mb][half]` (p4 = DPAT + 512).
fn store_row_patterns(l: &mut List, e: usize, o: usize) {
    for (src, parity) in [(e, 0), (o, 1)] {
        row_patterns(l, src, |l, mbl, half, reg| {
            l.push(st_at(
                reg,
                4,
                ((2 * mbl + parity) * 2 + half) as i64 - 8,
                MEM_PAT,
            ));
        });
    }
}

/// After phase G: p0 / p1 point at the E / O halves' D sections (+2048), p2 at the B chunk start; the caller sets
/// p4 = Dpat destination + 512 and p6 = Spat destination. Builds Dpat `[mb][half]` and Spat `[nb]` (memory) and copies
/// b(E), b(O), a.
fn patterns() -> Vec<Cycle> {
    let mut l = List::default();
    l.push(movxm_p(5, CORE_BASE + AB_COPY_ADDR));
    l.push(ld(6, 0, 1, false, MEM_A)); // D(E), p0 -> b(E)
    l.push(ld(7, 1, 1, true, MEM_A)); // D(O), p1 -> b(O)
    l.push(padd(Slot::Ldb, 2, 1)); // p2 += 2048 -> S
    l.push(ld(8, 2, 1, true, MEM_B)); // S, p2 -> a
    store_row_patterns(&mut l, 6, 7);
    col_patterns(&mut l, 8, 0);
    for nb in 0..4 {
        l.push(st_at(nb, 6, nb as i64, MEM_PAT));
    }
    l.push(ld(9, 0, 0, false, MEM_A));
    l.push(ld(10, 1, 0, true, MEM_A));
    l.push(ld(11, 2, 0, true, MEM_B));
    l.push(st(9, 5, 1, MEM_AB));
    l.push(st(10, 5, 1, MEM_AB));
    l.push(st(11, 5, 1, MEM_AB));
    l.finish()
}
/// Spat `[nb]` of epoch slot A (`SPAT_ADDR`) into x0..x3 for the single-epoch fold.
fn load_spat() -> Vec<Cycle> {
    let mut l = List::default();
    l.push(movxm_p(6, CORE_BASE + SPAT_ADDR));
    for nb in 0..4 {
        l.push(ld_at(nb, 6, nb as i64, nb % 2 == 1, MEM_PAT));
    }
    l.finish()
}

/// Constant vectors of the final pack, indices into `CONST_ADDR` (64 B each; 16-bit lanes unless `wide`).
#[derive(Clone, Copy)]
#[repr(usize)]
enum Konst {
    Zero,
    One,
    Two,
    Four,
    Eight,
    Sixteen,
    K256,
    K4096,
    K23,
    KM149,
    K55,
    KM8,
    K24,
    K40,
    K39,
    K7,
    KM9,
    K48,
    K32,
    K149,
    K254,
    Inf32,
    Sign32,
    K300,
    KM300,
}
const KONST: [(Konst, u32, bool); 25] = [
    (Konst::Zero, 0, false),
    (Konst::One, 1, false),
    (Konst::Two, 2, false),
    (Konst::Four, 4, false),
    (Konst::Eight, 8, false),
    (Konst::Sixteen, 16, false),
    (Konst::K256, 256, false),
    (Konst::K4096, 4096, false),
    (Konst::K23, 23, false),
    (Konst::KM149, (-149i32) as u32, false),
    (Konst::K55, 55, false),
    (Konst::KM8, (-8i32) as u32, false),
    (Konst::K24, 24, false),
    (Konst::K40, 40, false),
    (Konst::K39, 39, false),
    (Konst::K7, 7, false),
    (Konst::KM9, (-9i32) as u32, false),
    (Konst::K48, 48, false),
    (Konst::K32, 32, false),
    (Konst::K149, 149, false),
    (Konst::K254, 254, false),
    (Konst::Inf32, 0x7f80_0000, true),
    (Konst::Sign32, 0x8000_0000, true),
    (Konst::K300, 300, false),
    (Konst::KM300, (-300i32) as u32, false),
];
const _: () =
    assert!((CONST_ADDR as usize + 64 * KONST.len()) <= I_ADDR as usize && KONST.len() <= 32);
/// Constant base pointers: p3 = CONST + 8*64 reaches constants 0..16, p2 = CONST + 24*64 reaches 16..32.
const KONST_P3: u32 = CORE_BASE + CONST_ADDR + 8 * 64;
const KONST_P2: u32 = CORE_BASE + CONST_ADDR + 24 * 64;
fn konst(l: &mut List, dst: usize, k: Konst) {
    let k = k as i64;
    l.push(if k < 16 {
        ld_at(dst, 3, k - 8, true, MEM_CONST)
    } else {
        ld_at(dst, 2, k - 24, true, MEM_CONST)
    });
}

/// Program start: the constant table (`VBCST` + store).
fn constants() -> Vec<Cycle> {
    let mut l = List::default();
    l.push(movxm_p(7, CORE_BASE + CONST_ADDR));
    for (i, &(k, v, wide)) in KONST.iter().enumerate() {
        assert_eq!(k as usize, i);
        l.push(movxm_r(15, v));
        l.push(bcst(if wide { "VBCST_32" } else { "VBCST_16" }, 9, 15));
        l.push(st(9, 7, 1, MEM_CONST));
    }
    l.finish()
}

/// One final-pack group, exactly [`pack_lane_model`] per lane: 32 lanes of I (dm0 from `[p6]`, +256 B) and X
/// (`[p5]`, +64 B) to 32 f32 at `[p4]` (+128 B). p3 / p2 are the constant bases.
fn final_pack_group(limit: Option<usize>) -> Vec<Cycle> {
    use Konst::*;
    let mut l = List {
        limit,
        ..List::default()
    };
    let (vlt, vge, veqz, vsel, vadd, vsub) = (
        "VLT_16_vaddSign1",
        "VGE_16_vaddSign0",
        "VEQZ_16",
        "VSEL_16",
        "VADD_16",
        "VSUB_16",
    );
    let (vmax, vmin) = ("VMAX_LT_16_vaddSign1", "VMIN_GE_16_vaddSign1");
    l.push(movx_cr("crRnd", 0));
    l.barrier();
    for q in 0..4 {
        l.push(ld_bm(0, q, 6, MEM_I));
    }
    l.push(ld(0, 5, 1, true, MEM_C16));
    l.push(vneg(1, 0));
    l.push(movxm_s(0, 0));
    l.push(movxm_s(1, 16));
    l.push(movxm_s(2, 32));
    l.push(movxm_s(3, 48));
    konst(&mut l, 3, Zero);
    // Magnitude limbs m3..m0 in x4..x7 (r16 = negative).
    l.push(srs4(1, 0, 3));
    l.push(srs4(2, 1, 3));
    l.push(cmp(vlt, R_NEGMASK, 1, 3));
    l.push(sel(vsel, 4, 1, 2, R_NEGMASK));
    for (dst, sh) in [(5, 2), (6, 1), (7, 0)] {
        l.push(srs4(1, 0, sh));
        l.push(srs4(2, 1, sh));
        l.push(sel(vsel, dst, 1, 2, R_NEGMASK));
    }
    // Highest nonzero limb v (x1) and its bit offset (x2).
    l.push(eqz(veqz, 17, 4));
    l.push(eqz(veqz, 18, 5));
    l.push(eqz(veqz, 19, 6));
    l.push(sca("AND", 28, 17, 18));
    l.push(sca("AND", 29, 28, 19));
    l.push(sel(vsel, 1, 4, 5, 17));
    l.push(sel(vsel, 1, 1, 6, 28));
    l.push(sel(vsel, 1, 1, 7, 29));
    konst(&mut l, 2, K48);
    konst(&mut l, 8, K32);
    l.push(sel(vsel, 2, 2, 8, 17));
    konst(&mut l, 8, Sixteen);
    l.push(sel(vsel, 2, 2, 8, 28));
    l.push(sel(vsel, 2, 2, 3, 29));
    // Ladder: v >= 2^k -> v >>= k, L += k.
    for (thr, inc, k) in [
        (K256, Eight, 8u32),
        (Sixteen, Four, 4),
        (Four, Two, 2),
        (Two, One, 1),
    ] {
        konst(&mut l, 8, thr);
        l.push(cmp(vge, 30, 1, 8));
        l.push(movxm_s(2, k));
        l.push(ups4(2, 1, 0, false));
        l.push(srs4(9, 2, 2));
        l.push(sel(vsel, 1, 1, 9, 30));
        konst(&mut l, 8, inc);
        l.push(alu(vadd, 9, 2, 8));
        l.push(sel(vsel, 2, 2, 9, 30));
    }
    // sh = min(max(L - 23, -149 - X), 55) in x9.
    konst(&mut l, 8, K23);
    l.push(alu(vsub, 9, 2, 8));
    konst(&mut l, 8, KM149);
    l.push(alu(vsub, 10, 8, 0));
    l.push(alu(vmax, 9, 9, 10));
    konst(&mut l, 8, K55);
    l.push(alu(vmin, 9, 9, 8));
    // Classes: r17 = sh < -8 (E), r18 = sh < 8, r19 = sh < 24, r28 = sh < 40 (not R); S in x10, k = S - sh.
    konst(&mut l, 8, KM8);
    l.push(cmp(vlt, 17, 9, 8));
    konst(&mut l, 8, Eight);
    l.push(cmp(vlt, 18, 9, 8));
    konst(&mut l, 8, K24);
    l.push(cmp(vlt, 19, 9, 8));
    konst(&mut l, 8, K40);
    l.push(cmp(vlt, 28, 9, 8));
    konst(&mut l, 10, K55);
    konst(&mut l, 8, K39);
    l.push(sel(vsel, 10, 10, 8, 28));
    konst(&mut l, 8, K23);
    l.push(sel(vsel, 10, 10, 8, 19));
    konst(&mut l, 8, K7);
    l.push(sel(vsel, 10, 10, 8, 18));
    konst(&mut l, 8, KM9);
    l.push(sel(vsel, 10, 10, 8, 17));
    l.push(alu(vsub, 10, 10, 9));
    // p = 2^k in x11: hi = (b3 ? 256 : 1) * (b2 ? 16 : 1), lo = (b1 ? 4 : 1) * (b0 ? 2 : 1), p = hi * lo.
    konst(&mut l, 8, Eight);
    l.push(alu("VBAND", 11, 10, 8));
    l.push(eqz(veqz, 29, 11));
    konst(&mut l, 8, Four);
    l.push(alu("VBAND", 11, 10, 8));
    l.push(eqz(veqz, 30, 11));
    konst(&mut l, 8, Sixteen);
    konst(&mut l, 1, One);
    l.push(sel(vsel, 8, 8, 1, 30));
    konst(&mut l, 11, K4096);
    konst(&mut l, 1, K256);
    l.push(sel(vsel, 11, 11, 1, 30));
    l.push(sel(vsel, 11, 11, 8, 29));
    konst(&mut l, 8, Two);
    l.push(alu("VBAND", 1, 10, 8));
    l.push(eqz(veqz, 29, 1));
    konst(&mut l, 8, One);
    l.push(alu("VBAND", 1, 10, 8));
    l.push(eqz(veqz, 30, 1));
    konst(&mut l, 8, Two);
    konst(&mut l, 1, One);
    l.push(sel(vsel, 8, 8, 1, 30));
    konst(&mut l, 10, Eight);
    konst(&mut l, 1, Four);
    l.push(sel(vsel, 10, 10, 1, 30));
    l.push(sel(vsel, 10, 10, 8, 29));
    l.push(vmac(2, None, x(11), x(10), R_UU));
    l.push(movxm_s(2, 0));
    l.push(srs4(11, 2, 2));
    // Class limbs: R (sh >= 40) (0, m3, m2, m1|sticky); E (sh < -8) (0, m0, 0, 0); else (m3, m2, m1, m0).
    l.push(eqz(veqz, 29, 7));
    konst(&mut l, 8, One);
    l.push(alu("VBOR", 8, 6, 8));
    l.push(sel(vsel, 8, 8, 6, 29));
    l.push(sel(vsel, 10, 7, 3, 17));
    l.push(sel(vsel, 10, 8, 10, 28));
    l.push(sel(vsel, 1, 6, 3, 17));
    l.push(sel(vsel, 1, 5, 1, 28));
    l.push(sel(vsel, 2, 5, 7, 17));
    l.push(sel(vsel, 2, 4, 2, 28));
    l.push(sel(vsel, 4, 4, 3, 17));
    l.push(sel(vsel, 4, 3, 4, 28));
    // Horner acc = limbs * p (dm3), exact.
    l.push(vmac(3, None, x(4), x(11), R_UU));
    l.push(vmac(3, Some(3), x(2), x(11), R_UU16));
    l.push(vmac(3, Some(3), x(1), x(11), R_UU16));
    l.push(vmac(3, Some(3), x(10), x(11), R_UU16));
    // Ef7 = min(sh + X + 149, 254) << 7 in x5 (exact UPS / SRS, crRnd still floor).
    l.push(alu(vadd, 5, 9, 0));
    konst(&mut l, 8, K149);
    l.push(alu(vadd, 5, 5, 8));
    konst(&mut l, 8, K254);
    l.push(alu(vmin, 5, 5, 8));
    l.push(movxm_s(1, 7));
    l.push(ups4(4, 5, 1, true));
    l.push(srs4(5, 4, 0));
    // Candidates q7 / q23 / q39 as two 16-lane int32 halves each, conv_even.
    l.barrier();
    l.push(movx_cr("crRnd", 12));
    l.barrier();
    l.push(movxm_s(2, 23));
    l.push(movxm_s(3, 39));
    l.push(srs2(1, 3, false, 1));
    l.push(srs2(2, 3, true, 1));
    l.push(srs2(6, 3, false, 2));
    l.push(srs2(7, 3, true, 2));
    l.push(srs2(10, 3, false, 3));
    l.push(srs2(11, 3, true, 3));
    l.barrier();
    l.push(movx_cr("crRnd", 0));
    // Masks: r31 = c7 & !cE, r30 = !c23, r16 = negative; the high lane half uses mask >> 16.
    l.push(sca("XOR", 31, 17, 0));
    l.push(sca("AND", 31, 18, 31));
    l.push(sca("XOR", 30, 19, 0));
    for (lo, qa, qb, qc) in [(true, 1, 6, 10), (false, 2, 7, 11)] {
        if !lo {
            for m in [31, 30, R_NEGMASK] {
                l.push(sca("LSHL", m, m, R_M16));
            }
        }
        l.push(sel("VSEL_32", qb, qb, qa, 31));
        l.push(sel("VSEL_32", qb, qb, qc, 30));
        l.push(shuffle(8, 3, 5, if lo { T16_LO } else { T16_HI }));
        l.push(alu("VADD_32", 9, qb, 8));
        konst(&mut l, 8, Inf32);
        l.push(alu("VMIN_GE_32_vaddSign0", 9, 9, 8));
        l.push(eqz("VEQZ_32", 29, qb));
        l.push(sel("VSEL_32", 9, 9, 3, 29));
        konst(&mut l, 8, Sign32);
        l.push(alu("VBOR", 8, 9, 8));
        l.push(sel("VSEL_32", 9, 9, 8, R_NEGMASK));
        l.push(st(9, 4, 1, MEM_OUT));
    }
    l.push(movxm_r(R_NEGMASK, T128_HI as u32));
    l.finish()
}

/// Wave end, part 1: b patterns to DPAT (`[mb][half]`), a column patterns to x0..x3, +-X_CLAMP in x6 / x7.
/// Part 2 (loop body, 8 iterations over mb): X = clamp(b[mb][half] + a[nb]) into the C16 area in I order.
fn x_patterns() -> (Vec<Cycle>, Vec<Cycle>) {
    let mut l = List::default();
    l.push(movxm_p(4, CORE_BASE + DPAT_ADDR + 512));
    l.push(movxm_p(5, CORE_BASE + AB_COPY_ADDR));
    l.push(movxm_p(2, KONST_P2));
    l.push(ld(6, 5, 1, true, MEM_AB));
    l.push(ld(7, 5, 1, true, MEM_AB));
    l.push(ld(8, 5, 1, true, MEM_AB));
    store_row_patterns(&mut l, 6, 7);
    col_patterns(&mut l, 8, 0);
    konst(&mut l, 6, Konst::K300);
    konst(&mut l, 7, Konst::KM300);
    l.push(movxm_p(5, CORE_BASE + C16_ADDR));
    l.push(movxm_p(4, CORE_BASE + DPAT_ADDR));
    let head = l.finish();
    let mut b = List::default();
    b.push(ld(4, 4, 1, true, MEM_PAT));
    b.push(ld(5, 4, 1, true, MEM_PAT));
    for g in 0..8 {
        let (nb, half, t) = (g / 2, g % 2, 8 + g % 4);
        b.push(alu("VADD_16", t, 4 + half, nb));
        b.push(alu("VMAX_LT_16_vaddSign1", t, t, 7));
        b.push(alu("VMIN_GE_16_vaddSign1", t, t, 6));
        b.push(st(t, 5, 1, MEM_C16));
    }
    (head, b.finish())
}

/// Every scheduled block, built once.
struct Blocks {
    constants: Vec<Cycle>,
    zero_head: Vec<Cycle>,
    zero_body: Vec<Cycle>,
    g: Vec<Cycle>,
    patterns: Vec<Cycle>,
    f: Vec<Cycle>,
    f2: Vec<Cycle>,
    spat: Vec<Cycle>,
    x_head: Vec<Cycle>,
    x_body: Vec<Cycle>,
    pack: Vec<Cycle>,
}
impl Blocks {
    fn new() -> Self {
        let mut zh = List::default();
        zh.push(movxm_r(15, 0));
        zh.push(bcst("VBCST_32", 11, 15));
        let mut zb = List::default();
        for _ in 0..16 {
            zb.push(st(11, 7, 1, MEM_I));
        }
        // Tables end with eight empty bundles: every write (stage <= 8) retires before the next block.
        let pad = |mut c: Vec<Cycle>| {
            c.resize(c.len() + 8, [None; 8]);
            c
        };
        let (x_head, x_body) = x_patterns();
        Blocks {
            constants: constants(),
            zero_head: zh.finish(),
            zero_body: zb.finish(),
            g: pad(phase_g_table().cycles),
            patterns: patterns(),
            f: pad(phase_f_table().cycles),
            f2: pad(phase_f2_table().cycles),
            spat: load_spat(),
            x_head,
            x_body,
            pack: final_pack_group(None),
        }
    }
}
static BLOCKS: LazyLock<Blocks> = LazyLock::new(Blocks::new);
/// Diagnostics only: a single-tile program that runs the program prologue (registers, control registers, the
/// constant table) and the first `ops` operations of one final-pack group on I at `I_ADDR` and X at `C16_ADDR`, then
/// `DONE`. Registers are left for inspection; with `ops = usize::MAX` the group's f32 land at `COUT_ADDR`.
#[doc(hidden)]
pub fn final_pack_debug_program(ops: usize) -> Vec<u8> {
    let mut a = Assembler::default();
    for (reg, v) in [
        (0, -1i32 as u32),
        (1, 1),
        (R_UU, CONF_UU),
        (R_US, CONF_US),
        (R_US16, CONF_US16),
        (R_UU16, CONF_UU16),
        (R_NEG, CONF_NEG64),
        (R_ZERO, 0),
        (R_M16, (-16i32) as u32),
    ] {
        a.mov(0, reg, v);
    }
    for m in 12..=19 {
        a.mov(0, 20 + m - 12, m as u32);
    }
    for (cr, v) in [
        ("crSat", 0),
        ("crRnd", 0),
        ("crUPSMode", 1),
        ("crUnpackSize", 0),
        ("crSRSMode", 1),
    ] {
        a.set_cr(cr, v);
    }
    emit_block(&mut a, &constants());
    a.mov(1, 2, KONST_P2);
    a.mov(1, 3, KONST_P3);
    a.mov(1, 4, CORE_BASE + COUT_ADDR);
    a.mov(1, 5, CORE_BASE + C16_ADDR);
    a.mov(1, 6, CORE_BASE + I_ADDR);
    emit_block(&mut a, &final_pack_group(Some(ops)));
    a.nop(8);
    a.emit(&[(encoding("DONE").slot, encoding("DONE").value)]);
    a.finish().finish()
}

/// The IEF15 pair core program for `waves` output tiles of `epochs` K128 epochs (unfinished `Program`; the caller
/// runs `finish()`). `role` selects the E/O half and the neighbour views exactly as `gemm_core::program_pair`.
/// Only the control discipline of `ctl` is used (Fast or serial lock / branch spacing); compute is always VLIW.
pub fn program_pair(epochs: usize, waves: usize, role: PairRole, ctl: Control) -> Program {
    assert!(
        (1..=MAX_EPOCHS).contains(&epochs) && (1..=256).contains(&waves),
        "IEF15 core: epochs {epochs}, waves {waves}"
    );
    let probe = ctl.probe();
    assert!(
        !probe.no_compute && probe.repeat <= 1 && probe.layout == 0,
        "IEF15 core supports Fast / Slow control only"
    );
    let blocks = &*BLOCKS;
    let layout = PairLayout {
        a: A_ADDR,
        b: B_ADDR,
        c: I_ADDR,
        cout: COUT_ADDR,
    };
    let mut a = Assembler::default();
    a.serial = probe.serial;
    for (reg, v) in [
        (0, -1i32 as u32),
        (1, 1),
        (R_INT8, CONF_INT8),
        (3, waves as u32),
        (4, 0),
        (R_UU, CONF_UU),
        (R_US, CONF_US),
        (R_US16, CONF_US16),
        (R_UU16, CONF_UU16),
        (R_NEG, CONF_NEG64),
        (R_ZERO, 0),
        (R_M16, (-16i32) as u32),
    ] {
        a.mov(0, reg, v);
    }
    for m in 12..=19 {
        a.mov(0, 20 + m - 12, m as u32);
    }
    a.mov(2, 0, (-2048i32) as u32);
    a.mov(2, 1, 2048);
    for (cr, v) in [
        ("crSat", 0),
        ("crRnd", 0),
        ("crUPSMode", 1),
        ("crUnpackSize", 0),
        ("crSRSMode", 0),
    ] {
        a.set_cr(cr, v);
    }
    emit_block(&mut a, &blocks.constants);
    a.mov(2, 2, 128);
    let wave = a.label();
    let epoch = a.label();
    let second = a.label();
    let slots = a.label();
    let fold2 = a.label();
    let folded = a.label();
    a.bind(wave);
    a.mov(1, 7, CORE_BASE + I_ADDR);
    emit_block(&mut a, &blocks.zero_head);
    emit_loop(
        &mut a,
        blocks.zero_body.clone(),
        (TM * TN * 8 / (16 * 64)) as u32,
        R_LOOP,
    );
    a.mov(0, 5, epochs as u32);
    a.mov(0, R_PAIR, 0);
    // Epochs run in fold pairs: the even epoch of a pair leaves C16 / Dpat / Spat in slot A, the odd one in slot B,
    // and one F2 pass folds both into I. An odd epoch count ends with a single-epoch fold (F) of slot A.
    a.bind(epoch);
    pair_acquire(&mut a, role, layout);
    a.branch(second, Some((R_PAIR as u8, true)));
    a.delay();
    a.mov(1, 3, CORE_BASE + C16_ADDR);
    a.mov(1, 4, CORE_BASE + DPAT_ADDR + 512);
    a.mov(1, 6, CORE_BASE + SPAT_ADDR);
    a.branch(slots, None);
    a.delay();
    a.bind(second);
    a.mov(1, 3, CORE_BASE + C16_B_ADDR);
    a.mov(1, 4, CORE_BASE + DPAT_B_ADDR + 512);
    a.mov(1, 6, CORE_BASE + SPAT_ADDR + 256);
    a.bind(slots);
    a.mov_named("s0", 0);
    a.set_cr("crSRSMode", 0);
    emit_periodic(
        &mut a,
        &blocks.g,
        G_T0 + 2 * G_ROW - 12,
        2 * G_ROW,
        3,
        R_LOOP,
    );
    emit_block(&mut a, &blocks.patterns);
    a.set_cr("crSRSMode", 1);
    pair_release(&mut a, role);
    a.emit(&[(
        ins(
            "XOR",
            &[
                ("d0", Rg(r(R_PAIR))),
                ("s0", Rg(r(R_PAIR))),
                ("s1", Rg(r(1))),
            ],
        )
        .slot,
        ins(
            "XOR",
            &[
                ("d0", Rg(r(R_PAIR))),
                ("s0", Rg(r(R_PAIR))),
                ("s1", Rg(r(1))),
            ],
        )
        .bits,
    )]);
    if !a.serial {
        a.nop(3);
    }
    a.mov_named("s0", 0);
    a.mov_named("s1", 16);
    a.branch(fold2, Some((R_PAIR as u8, false)));
    a.delay();
    a.branch(epoch, Some((5, true)));
    a.delay();
    // Odd tail: fold slot A alone.
    emit_block(&mut a, &blocks.spat);
    a.mov(1, 4, CORE_BASE + DPAT_ADDR);
    a.mov(1, 5, CORE_BASE + C16_ADDR);
    a.mov(1, 6, CORE_BASE + I_ADDR);
    a.mov(1, 7, CORE_BASE + I_ADDR);
    // Period k of 8 half blocks repeats for k = 1..=6 (k = 0 lacks the previous tails, k = 7 the next Dpat loads).
    emit_periodic(&mut a, &blocks.f, F_T0 + 8 * F_II, 8 * F_II, 6, R_LOOP);
    a.branch(folded, None);
    a.delay();
    a.bind(fold2);
    a.mov(1, 0, CORE_BASE + SPAT_ADDR);
    a.mov(1, 1, CORE_BASE + DPAT_ADDR);
    a.mov(1, 2, CORE_BASE + DPAT_B_ADDR);
    a.mov(1, 3, CORE_BASE + C16_ADDR);
    a.mov(1, 4, CORE_BASE + C16_B_ADDR);
    a.mov(1, 5, CORE_BASE + I_ADDR);
    a.mov(1, 6, CORE_BASE + I_ADDR);
    // Periods 1..=7 of 8 half blocks repeat (period 0 lacks the previous tails).
    emit_periodic(
        &mut a,
        &blocks.f2,
        F2_T0 + 8 * F2_II - 14,
        8 * F2_II,
        7,
        R_LOOP,
    );
    a.branch(epoch, Some((5, true)));
    a.delay();
    a.bind(folded);
    a.lock(true, C_EMPTY);
    emit_block(&mut a, &blocks.x_head);
    emit_loop(&mut a, blocks.x_body.clone(), 8, R_LOOP);
    a.mov(1, 2, KONST_P2);
    a.mov(1, 3, KONST_P3);
    a.mov(1, 4, CORE_BASE + COUT_ADDR);
    a.mov(1, 5, CORE_BASE + C16_ADDR);
    a.mov(1, 6, CORE_BASE + I_ADDR);
    emit_loop(&mut a, blocks.pack.clone(), (TM * TN / 32) as u32, R_LOOP);
    a.set_cr("crSRSMode", 0);
    a.nop(8);
    a.lock(false, C_FULL);
    a.add(3, -1);
    a.branch(wave, Some((3, true)));
    a.delay();
    a.emit(&[(encoding("DONE").slot, encoding("DONE").value)]);
    a.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tools/npu/fold-model/src/lib.rs::round_scaled`, copied as the independent reference.
    fn round_scaled(x: i128, exponent: i32) -> f32 {
        if x == 0 {
            return 0.0;
        }
        let sign = if x < 0 { 1u32 << 31 } else { 0 };
        let magnitude = x.unsigned_abs();
        let top = 127 - magnitude.leading_zeros() as i32;
        let mut result_exponent = top + exponent;
        if result_exponent > 127 {
            return f32::from_bits(sign | 0x7f800000);
        }
        let shift = if result_exponent >= -126 {
            top - 23
        } else {
            -149 - exponent
        };
        let mut mantissa = if shift <= 0 {
            magnitude << (-shift as u32)
        } else if shift >= 128 {
            0
        } else {
            let quotient = magnitude >> shift;
            let remainder = magnitude - (quotient << shift);
            let midpoint = 1u128 << (shift - 1);
            quotient
                + u128::from(remainder > midpoint || (remainder == midpoint && quotient & 1 != 0))
        };
        if result_exponent < -126 {
            return f32::from_bits(sign | mantissa as u32);
        }
        if mantissa == 1 << 24 {
            mantissa >>= 1;
            result_exponent += 1;
        }
        if result_exponent > 127 {
            return f32::from_bits(sign | 0x7f800000);
        }
        f32::from_bits(sign | ((result_exponent + 127) as u32) << 23 | (mantissa as u32 & 0x7fffff))
    }

    fn check(i: i64, x: i32) {
        let want = round_scaled(i128::from(i), x).to_bits();
        assert_eq!(pack_lane_model(i, x), want, "I={i} X={x}");
    }

    #[test]
    fn final_pack_model_is_round_scaled() {
        let limit = (1i64 << 51) - 1;
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let xs: Vec<i32> = (-310..=310).collect();
        // Every power-of-two boundary, its neighbours and halfway patterns, at every exponent.
        for bit in 0..51 {
            let p = 1i64 << bit;
            for v in [
                p - 1,
                p,
                p + 1,
                p | (p >> 1),
                (p | (p >> 1)) + 1,
                p + (p >> 24),
                p + (p >> 25),
                p + (p >> 23),
            ] {
                if v <= 0 || v > limit {
                    continue;
                }
                for &x in &xs {
                    check(v, x);
                    check(-v, x);
                }
            }
        }
        for &x in &xs {
            check(0, x);
            check(1, x);
            check(-1, x);
            check(limit, x);
            check(-limit, x);
        }
        // Exact ties of the final rounding: 24 significant bits plus a halfway bit, shifted to every position.
        for _ in 0..20000 {
            let mant = (next() >> 40) as i64 | (1 << 24);
            let shift = (next() % 27) as u32;
            let tie = ((mant << 1) | 1) << shift;
            if tie > limit {
                continue;
            }
            let x = (next() % 621) as i32 - 310;
            check(tie, x);
            check(-tie, x);
        }
        for _ in 0..2_000_000 {
            let bits = (next() % 52) as u32;
            let v = (next() >> 13) as i64 & ((1i64 << bits) - 1).max(0);
            let v = if next() & 1 == 1 { -v } else { v };
            let x = (next() % 621) as i32 - 310;
            check(v, x);
        }
        // Wide exponents (host limit |a|,|b| <= 16383).
        for x in [-32766, -16383, -1000, -301, 301, 1000, 16383, 32766] {
            for v in [1i64, 12345, limit, -limit, -7] {
                check(v, x);
            }
        }
    }

    #[test]
    fn program_assembles_fits_and_passes_rules() {
        for (epochs, waves) in [(1, 1), (5, 2), (40, 32), (136, 128)] {
            for role in [PairRole::Lower, PairRole::Upper] {
                for ctl in [Control::Fast, Control::Slow] {
                    let bytes = program_pair(epochs, waves, role, ctl).finish();
                    assert!(bytes.len() <= 16 * 1024, "program {} B", bytes.len());
                    if epochs == 40 && ctl == Control::Fast {
                        eprintln!("IEF15 core {role:?}: {} B", bytes.len());
                    }
                }
            }
        }
    }

    #[test]
    fn program_physical_schedule_is_clean() {
        for role in [PairRole::Lower, PairRole::Upper] {
            let bytes = program_pair(40, 4, role, Control::Fast).finish();
            let mut bundles = Vec::new();
            let mut offset = 0;
            while offset < bytes.len() {
                let bundle = isa::decode::decode(&bytes[offset..], offset as u64).unwrap();
                offset += bundle.len;
                bundles.push(bundle);
            }
            let report = isa::sched::check_decoded_program(&bundles).unwrap();
            assert!(
                report.is_clean(),
                "{:?}",
                &report.hazards[..report.hazards.len().min(8)]
            );
            eprintln!(
                "IEF15 core {role:?}: {} B, {} bundles, {} timing observations",
                bytes.len(),
                report.bundles,
                report.timing_observations.len()
            );
        }
    }
}
